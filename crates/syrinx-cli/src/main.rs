//! `syrinx` -- the command-line front-end.
//!
//! Deliberately thin: everything of substance lives in `syrinx-client`, so
//! the CLI and the GUI cannot drift apart.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use syrinx_client::{Config, OutputMode, SessionOptions, save, state};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "syrinx", about = "Live speech to text")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,

    /// Config file. Defaults to ~/.config/syrinx/config.toml
    #[arg(long, global = true)]
    config: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum FormatArg {
    /// Continuous prose.
    Plain,
    /// Each fragment prefixed with [MM:SS].
    Timestamped,
}

impl From<FormatArg> for save::Format {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Plain => save::Format::Plain,
            FormatArg::Timestamped => save::Format::Timestamped,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ModeArg {
    /// Accumulate a transcript and print it when the session ends.
    Transcribe,
    /// Type at the cursor as you speak.
    Type,
    /// Both.
    Both,
}

impl From<ModeArg> for OutputMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Transcribe => OutputMode::Transcribe,
            ModeArg::Type => OutputMode::Type,
            ModeArg::Both => OutputMode::Both,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Start a session and block until stopped.
    Start {
        /// What to do with the text. Defaults to the config file's setting.
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        /// Source to capture, as shown by `syrinx sources`. Defaults to the
        /// remembered one, else a microphone.
        #[arg(long)]
        source: Option<String>,
        /// Write the transcript to this file when the session ends.
        #[arg(long)]
        save: Option<PathBuf>,
        /// Write the transcript to the default transcripts folder.
        #[arg(long, conflicts_with = "save")]
        save_default: bool,
        /// Layout of the saved file. Timestamped prefixes each fragment with
        /// its time, which makes the transcript an index into the recording.
        #[arg(long, value_enum, default_value = "plain")]
        format: FormatArg,
    },
    /// Stop a running session.
    Stop,
    /// Start if idle, stop if running. Bind this to a key.
    Toggle {
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        save: Option<PathBuf>,
        #[arg(long, conflicts_with = "save")]
        save_default: bool,
        #[arg(long, value_enum, default_value = "plain")]
        format: FormatArg,
    },
    /// Report whether a session is active.
    Status,
    /// Run headless in the system tray: no window, click the icon to
    /// start and stop. This is what keeps running in the background.
    Tray {
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        #[arg(long)]
        source: Option<String>,
        /// Save each session to the default transcripts folder as it ends.
        #[arg(long)]
        save_default: bool,
        #[arg(long, value_enum, default_value = "plain")]
        format: FormatArg,
    },

    /// List capturable audio sources.
    Sources,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "syrinx=info,syrinx_client=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let pid_path = state::default_pid_path();

    match cli.command {
        Cmd::Tray {
            mode,
            source,
            save_default,
            format,
        } => run_tray(cli.config, mode, source, save_default, format.into()),

        Cmd::Sources => {
            let mut last = None;
            for s in syrinx_client::list_sources()? {
                if last != Some(s.kind) {
                    println!("\n{}:", s.kind.label());
                    last = Some(s.kind);
                }
                println!("  {:<52} {}", s.display(), s.stable_key());
            }
            Ok(())
        }

        Cmd::Status => {
            match state::running_pid(&pid_path) {
                Some(pid) => println!("running (pid {pid})"),
                None => println!("idle"),
            }
            Ok(())
        }

        Cmd::Stop => {
            match state::running_pid(&pid_path) {
                Some(pid) => {
                    state::signal_stop(pid)?;
                    info!("asked pid {pid} to stop");
                }
                None => info!("not running"),
            }
            Ok(())
        }

        Cmd::Toggle {
            mode,
            source,
            save,
            save_default,
            format,
        } => {
            if let Some(pid) = state::running_pid(&pid_path) {
                state::signal_stop(pid)?;
                info!("stopping pid {pid}");
                return Ok(());
            }
            run(cli.config, mode, source, save, save_default, format.into(), pid_path)
        }

        Cmd::Start {
            mode,
            source,
            save,
            save_default,
            format,
        } => {
            if let Some(pid) = state::running_pid(&pid_path) {
                anyhow::bail!("already running (pid {pid})");
            }
            run(cli.config, mode, source, save, save_default, format.into(), pid_path)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    config: Option<PathBuf>,
    mode: Option<ModeArg>,
    source: Option<String>,
    save_to: Option<PathBuf>,
    save_default: bool,
    format: save::Format,
    pid_path: PathBuf,
) -> Result<()> {
    let cfg = Config::load(config)?;
    let mode = mode.map(OutputMode::from).unwrap_or(cfg.mode);

    let sources = syrinx_client::list_sources()?;
    let remembered = source.as_deref().or(cfg.source_key.as_deref());
    let source = syrinx_client::choose_source(&sources, remembered)?;

    state::write_pid(&pid_path, std::process::id() as i32)?;
    state::refresh_waybar(cfg.waybar_signal);

    let mut handle = syrinx_client::session::start(
        SessionOptions {
            url: cfg.url.clone(),
            token: cfg.token.clone(),
            source,
            mode,
        },
        // The CLI has nothing to repaint, so state changes need no callback.
        || {},
    );

    // SIGTERM is how `stop` and `toggle` ask us to finish, and Ctrl-C is the
    // interactive equivalent. Handling them rather than dying lets the session
    // flush its final words instead of truncating mid-utterance.
    let (sig_tx, sig_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        wait_for_stop_signal();
        let _ = sig_tx.send(());
    });

    while handle.is_running() {
        if sig_rx.try_recv().is_ok() {
            handle.stop();
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let final_state = handle.state();
    state::clear_pid(&pid_path);
    state::refresh_waybar(cfg.waybar_signal);

    if mode.keeps_transcript() && !final_state.transcript.trim().is_empty() {
        println!("{}", final_state.transcript.trim());

        // Saving uses the same code the GUI's Save button does, so the two
        // produce byte-identical files.
        if save_to.is_some() || save_default {
            let p = save::save_rendered(
                save_to.as_deref(),
                &final_state.segments,
                &final_state.transcript,
                format,
            )?;
            eprintln!("saved to {}", p.display());
        }
    } else if save_to.is_some() || save_default {
        // Asking to save in a mode that keeps no transcript is a mistake worth
        // naming rather than silently doing nothing.
        anyhow::bail!(
            "--save has nothing to write in {} mode; use --mode transcribe or both",
            mode.label()
        );
    }
    if let Some(e) = final_state.error {
        anyhow::bail!("{e}");
    }
    Ok(())
}

/// Block until SIGTERM or Ctrl-C.
fn wait_for_stop_signal() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("building a runtime for signal handling");
    rt.block_on(async {
        let mut term =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("installing a SIGTERM handler");
        tokio::select! {
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
        }
    });
}

/// Headless tray loop.
///
/// The GUI cannot serve this purpose: winit documents `set_visible` as
/// unsupported on Wayland, so a window cannot hide itself and keep running.
/// Background operation needs a process that never had a window to begin with.
fn run_tray(
    config: Option<PathBuf>,
    mode: Option<ModeArg>,
    source: Option<String>,
    save_default: bool,
    format: save::Format,
) -> Result<()> {
    use syrinx_client::tray::{TrayCommand, TrayState};

    let cfg = Config::load(config)?;
    let mut mode = mode.map(OutputMode::from).unwrap_or(cfg.mode);
    let source_pref = source.or_else(|| cfg.source_key.clone());

    let Some((tray, mut rx)) = syrinx_client::tray::start() else {
        anyhow::bail!(
            "no system tray available. A StatusNotifierItem host must be running \
             (waybar with its tray module, or a desktop panel)."
        );
    };
    info!("tray running; click the icon to start and stop");

    let mut session: Option<syrinx_client::SessionHandle> = None;
    let mut quit = false;

    // Signals stop the whole tray, not just a session, so ^C behaves.
    let (sig_tx, sig_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        wait_for_stop_signal();
        let _ = sig_tx.send(());
    });

    while !quit {
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                TrayCommand::Toggle | TrayCommand::Start | TrayCommand::Stop => {
                    let running = session.as_ref().is_some_and(|s| s.is_running());
                    let want_stop = matches!(cmd, TrayCommand::Stop)
                        || (running && matches!(cmd, TrayCommand::Toggle));
                    if want_stop {
                        if let Some(s) = &mut session {
                            s.stop();
                        }
                    } else if !running {
                        let sources = syrinx_client::list_sources()?;
                        let src =
                            syrinx_client::choose_source(&sources, source_pref.as_deref())?;
                        session = Some(syrinx_client::session::start(
                            SessionOptions {
                                url: cfg.url.clone(),
                                token: cfg.token.clone(),
                                source: src,
                                mode,
                            },
                            || {},
                        ));
                    }
                }
                TrayCommand::SetMode(m) => {
                    if !session.as_ref().is_some_and(|s| s.is_running()) {
                        mode = m;
                    }
                }
                // Nothing to show: this process has no window by design.
                TrayCommand::ShowWindow => {
                    info!("no window in tray mode; run `syrinx-gui` for one");
                }
                TrayCommand::Quit => quit = true,
            }
        }

        if sig_rx.try_recv().is_ok() {
            quit = true;
        }

        // A finished session is reaped here, which is also where its transcript
        // gets saved -- the tray has no other moment to notice it ended.
        if let Some(s) = &session
            && !s.is_running()
        {
            let st = s.state();
            if save_default && mode.keeps_transcript() {
                match save::save_rendered(None, &st.segments, &st.transcript, format) {
                    Ok(p) => info!("saved to {}", p.display()),
                    // An empty transcript is the normal "stopped without
                    // speaking" case, not an error worth shouting about.
                    Err(e) => info!("not saved: {e}"),
                }
            }
            if let Some(e) = st.error {
                tracing::error!("session ended with an error: {e}");
            }
            session = None;
        }

        let st = session
            .as_ref()
            .map(|s| s.state())
            .unwrap_or_default();
        tray.update(TrayState {
            status: st.status,
            mode,
            last_fragment: st.last_fragment.clone(),
        });

        std::thread::sleep(std::time::Duration::from_millis(120));
    }

    if let Some(s) = &mut session {
        s.stop();
    }
    info!("tray stopped");
    Ok(())
}
