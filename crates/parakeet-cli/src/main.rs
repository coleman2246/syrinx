//! `parakeet` -- the command-line front-end.
//!
//! Deliberately thin: everything of substance lives in `parakeet-client`, so
//! the CLI and the GUI cannot drift apart.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use parakeet_client::{Config, OutputMode, SessionOptions, state};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "parakeet", about = "Live speech to text")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,

    /// Config file. Defaults to ~/.config/parakeet/config.toml
    #[arg(long, global = true)]
    config: Option<PathBuf>,
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
        /// Source to capture, as shown by `parakeet sources`. Defaults to the
        /// remembered one, else a microphone.
        #[arg(long)]
        source: Option<String>,
    },
    /// Stop a running session.
    Stop,
    /// Start if idle, stop if running. Bind this to a key.
    Toggle {
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        #[arg(long)]
        source: Option<String>,
    },
    /// Report whether a session is active.
    Status,
    /// List capturable audio sources.
    Sources,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "parakeet=info,parakeet_client=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let pid_path = state::default_pid_path();

    match cli.command {
        Cmd::Sources => {
            let mut last = None;
            for s in parakeet_client::list_sources()? {
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

        Cmd::Toggle { mode, source } => {
            if let Some(pid) = state::running_pid(&pid_path) {
                state::signal_stop(pid)?;
                info!("stopping pid {pid}");
                return Ok(());
            }
            run(cli.config, mode, source, pid_path)
        }

        Cmd::Start { mode, source } => {
            if let Some(pid) = state::running_pid(&pid_path) {
                anyhow::bail!("already running (pid {pid})");
            }
            run(cli.config, mode, source, pid_path)
        }
    }
}

fn run(
    config: Option<PathBuf>,
    mode: Option<ModeArg>,
    source: Option<String>,
    pid_path: PathBuf,
) -> Result<()> {
    let cfg = Config::load(config)?;
    let mode = mode.map(OutputMode::from).unwrap_or(cfg.mode);

    let sources = parakeet_client::list_sources()?;
    let remembered = source.as_deref().or(cfg.source_key.as_deref());
    let source = parakeet_client::choose_source(&sources, remembered)?;

    state::write_pid(&pid_path, std::process::id() as i32)?;
    state::refresh_waybar(cfg.waybar_signal);

    let mut handle = parakeet_client::session::start(
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
