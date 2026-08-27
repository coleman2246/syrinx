//! `syrinx` -- the command-line front-end.
//!
//! Deliberately thin: everything of substance lives in `syrinx-client`, so
//! the CLI and the GUI cannot drift apart.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use syrinx_client::{Config, OutputMode, SessionOptions, save, state};
use std::path::{Path, PathBuf};
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
    /// Time and source on each line, for several sources at once.
    Labelled,
}

/// The layout to use: the flag if given, otherwise the config.
///
/// The flag used to carry `default_value = "plain"`, which made it impossible
/// for clap to tell "not given" from "asked for plain" -- so `format` in the
/// config was never consulted and streaming was always plain prose.
fn resolve_format(flag: Option<FormatArg>, cfg: &Config) -> save::Format {
    flag.map(save::Format::from).unwrap_or(cfg.format)
}

impl From<FormatArg> for save::Format {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Plain => save::Format::Plain,
            FormatArg::Timestamped => save::Format::Timestamped,
            FormatArg::Labelled => save::Format::Labelled,
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
        /// Source to capture, as shown by `syrinx sources`. Repeat for
        /// several. Defaults to the remembered one, else a microphone.
        #[arg(long)]
        source: Vec<String>,
        /// Transcribe each source as its own stream, labelled, instead of
        /// mixing them into one. Only the first source types at the cursor.
        #[arg(long)]
        separate: bool,
        /// Append the transcript to this file as it is dictated, rather than
        /// only saving at the end. Overrides `stream_to` in the config.
        #[arg(long, value_name = "FILE")]
        stream: Option<PathBuf>,
        /// Write the transcript to this file when the session ends.
        #[arg(long)]
        save: Option<PathBuf>,
        /// Write the transcript to the default transcripts folder.
        #[arg(long, conflicts_with = "save")]
        save_default: bool,
        /// Layout of the saved file. Timestamped prefixes each fragment with
        /// its time, which makes the transcript an index into the recording.
        /// Layout for saved and streamed transcripts. Defaults to `format`
        /// in the config.
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
        /// With --separate, write one file per source instead of one combined
        /// file.
        #[arg(long)]
        split: bool,
    },
    /// Stop a running session.
    Stop,
    /// Start if idle, stop if running. Bind this to a key.
    Toggle {
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        #[arg(long)]
        source: Vec<String>,
        #[arg(long)]
        separate: bool,
        /// Append the transcript to this file as it is dictated.
        #[arg(long, value_name = "FILE")]
        stream: Option<PathBuf>,
        #[arg(long)]
        save: Option<PathBuf>,
        #[arg(long, conflicts_with = "save")]
        save_default: bool,
        /// Layout for saved and streamed transcripts. Defaults to `format`
        /// in the config.
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
        #[arg(long)]
        split: bool,
    },
    /// Report whether a session is active.
    Status,
    /// Stop the background daemon.
    ///
    /// `stop` ends dictation but leaves the daemon running, the same as the
    /// tray's Stop. This is the tray's Quit.
    Quit,
    /// Run the background daemon: tray icon, no window. The GUI attaches to
    /// this, so closing the GUI leaves dictation running.
    Daemon {
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
        #[arg(long)]
        source: Option<String>,
        /// Save each session to the default transcripts folder as it ends.
        #[arg(long)]
        save_default: bool,
        /// Layout for saved and streamed transcripts. Defaults to `format`
        /// in the config.
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
    },

    /// Transcribe an audio file. Anything ffmpeg can decode: wav, mp3, m4a,
    /// opus, flac.
    Transcribe {
        /// File to transcribe.
        file: PathBuf,
        /// Write the transcript here instead of printing it.
        #[arg(long)]
        save: Option<PathBuf>,
        /// Layout for saved and streamed transcripts. Defaults to `format`
        /// in the config.
        #[arg(long, value_enum)]
        format: Option<FormatArg>,
    },

    /// Show a live level meter for a source, to confirm it is carrying audio.
    /// Runs until interrupted.
    Meter {
        /// Source to meter. Defaults to the configured one.
        #[arg(long)]
        source: Option<String>,
        /// Stop after this many seconds instead of running until interrupted.
        #[arg(long)]
        seconds: Option<u64>,
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
        Cmd::Daemon {
            mode,
            source,
            save_default,
            format,
        } => run_daemon(cli.config, mode, source, save_default, format),

        Cmd::Transcribe { file, save, format } => {
            run_transcribe(cli.config, file, save, format)
        }

        Cmd::Meter { source, seconds } => run_meter(cli.config, source, seconds),

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
            // A daemon owns its session and writes no PID file, so looking only
            // at the PID file reports "idle" while the GUI is actively
            // dictating. The CLI is meant to be equivalent to the GUI, and that
            // starts with the two agreeing about what is happening.
            match state::running_pid(&pid_path) {
                Some(pid) => println!("running (pid {pid})"),
                None => match daemon_state() {
                    Some(s) => println!(
                        "{} (daemon, mode {})",
                        s.status.label().to_lowercase(),
                        s.mode.name()
                    ),
                    None => println!("idle"),
                },
            }
            Ok(())
        }

        Cmd::Quit => {
            if !syrinx_client::ipc::daemon_running() {
                info!("no daemon running");
                return Ok(());
            }
            ask_daemon(syrinx_client::ipc::Request::Quit)?;
            info!("daemon asked to quit");
            Ok(())
        }

        Cmd::Stop => {
            match state::running_pid(&pid_path) {
                Some(pid) => {
                    state::signal_stop(pid, &pid_path)?;
                    info!("asked pid {pid} to stop");
                }
                // Fall through to the daemon, which holds its session without a
                // PID file. Without this, `stop` is a no-op whenever dictation
                // was started from the GUI or the tray.
                None if syrinx_client::ipc::daemon_running() => {
                    ask_daemon(syrinx_client::ipc::Request::Stop)?;
                    info!("asked the daemon to stop dictating");
                }
                None => info!("not running"),
            }
            Ok(())
        }

        Cmd::Toggle {
            mode,
            source,
            separate,
            stream,
            save,
            save_default,
            format,
            split,
        } => {
            if let Some(pid) = state::running_pid(&pid_path) {
                state::signal_stop(pid, &pid_path)?;
                info!("stopping pid {pid}");
                return Ok(());
            }
            // A running daemon owns the session, so toggling has to go through
            // it rather than start a second one competing for the microphone.
            if syrinx_client::ipc::daemon_running() {
                ask_daemon(syrinx_client::ipc::Request::Toggle)?;
                return Ok(());
            }
            run(cli.config, mode, source, separate, stream, save, save_default, format, split, pid_path)
        }

        Cmd::Start {
            mode,
            source,
            separate,
            stream,
            save,
            save_default,
            format,
            split,
        } => {
            if let Some(pid) = state::running_pid(&pid_path) {
                anyhow::bail!("already running (pid {pid})");
            }
            run(cli.config, mode, source, separate, stream, save, save_default, format, split, pid_path)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    config: Option<PathBuf>,
    mode: Option<ModeArg>,
    sources_wanted: Vec<String>,
    separate: bool,
    stream: Option<PathBuf>,
    save_to: Option<PathBuf>,
    save_default: bool,
    format: Option<FormatArg>,
    split: bool,
    pid_path: PathBuf,
) -> Result<()> {
    let cfg = Config::load(config)?;
    let mode = mode.map(OutputMode::from).unwrap_or(cfg.mode);

    let available = syrinx_client::list_sources()?;
    let chosen: Vec<syrinx_client::Source> = if sources_wanted.is_empty() {
        vec![syrinx_client::choose_source(
            &available,
            cfg.source_key.as_deref(),
        )?]
    } else {
        let mut v = Vec::new();
        for key in &sources_wanted {
            v.push(
                syrinx_client::resolve(&available, key)
                    .with_context(|| format!("no source matching {key:?}"))?,
            );
        }
        v
    };
    let source_mode = if separate {
        syrinx_client::mode::SourceMode::Separate
    } else {
        syrinx_client::mode::SourceMode::Combined
    };

    // The flag wins over the config, so a one-off run can stream somewhere
    // else without editing anything.
    let format = resolve_format(format, &cfg);
    let stream_target = stream
        .map(|p| (p, format))
        .or_else(|| cfg.stream_path().map(|p| (p, format)));
    if let Some((path, _)) = &stream_target {
        info!("appending the transcript to {}", path.display());
    }

    state::write_pid(&pid_path, std::process::id() as i32)?;
    state::refresh_waybar(cfg.waybar_signal);

    // Separate mode is one session per source; combined is one session fed by
    // a mix. Only the first source may type, since several streams typing into
    // one cursor interleave into nonsense.
    let mut handles: Vec<syrinx_client::SessionHandle> = match source_mode {
        syrinx_client::mode::SourceMode::Combined => vec![syrinx_client::session::start(
            SessionOptions {
                url: cfg.url.clone(),
                token: cfg.token.clone(),
                sources: chosen,
                mode,
                diarize: cfg.diarize,
                label: None,
                inject: cfg.inject,
                stream: stream_target.clone(),
                external_audio: None,
            },
            // The CLI has nothing to repaint, so state changes need no callback.
            || {},
        )],
        syrinx_client::mode::SourceMode::Separate => chosen
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let label = Some(s.short_label());
                syrinx_client::session::start(
                    SessionOptions {
                        url: cfg.url.clone(),
                        token: cfg.token.clone(),
                        sources: vec![s],
                        mode: if i == 0 { mode } else { OutputMode::Transcribe },
                        diarize: cfg.diarize,
                        label,
                        inject: cfg.inject,
                        stream: stream_target.clone(),
                        external_audio: None,
                    },
                    || {},
                )
            })
            .collect(),
    };

    // SIGTERM is how `stop` and `toggle` ask us to finish, and Ctrl-C is the
    // interactive equivalent. Handling them rather than dying lets the session
    // flush its final words instead of truncating mid-utterance.
    let (sig_tx, sig_rx) = std::sync::mpsc::channel();
    let stop_path = pid_path.clone();
    std::thread::spawn(move || {
        wait_for_stop_signal(&stop_path);
        let _ = sig_tx.send(());
    });

    while handles.iter().any(|h| h.is_running()) {
        if sig_rx.try_recv().is_ok() {
            for h in &mut handles {
                h.stop();
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let final_state = merge_cli_states(&handles);
    state::clear_pid(&pid_path);
    state::refresh_waybar(cfg.waybar_signal);

    if mode.keeps_transcript() && !final_state.transcript.trim().is_empty() {
        println!("{}", final_state.transcript.trim());

        // Saving uses the same code the GUI's Save button does, so the two
        // produce byte-identical files.
        if save_to.is_some() || save_default {
            if split {
                let base = save_to.clone().unwrap_or_else(|| {
                    save::default_dir().join(save::filename_for(&save::timestamp()))
                });
                for p in save::save_per_source(&base, &final_state.segments, format)? {
                    eprintln!("saved to {}", p.display());
                }
            } else {
                let p = save::save_rendered(
                    save_to.as_deref(),
                    &final_state.segments,
                    &final_state.transcript,
                    format,
                )?;
                eprintln!("saved to {}", p.display());
            }
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

/// The daemon's current state, or `None` if no daemon is listening.
fn daemon_state() -> Option<syrinx_client::ipc::DaemonState> {
    // Always the whole state: this runs once per command and holds nothing
    // between runs, so there is no earlier revision to ask against.
    match syrinx_client::ipc::request(&syrinx_client::ipc::Request::GetState { since: None }) {
        Ok(syrinx_client::ipc::Response::State(s)) => Some(s),
        _ => None,
    }
}

/// Send one request to the daemon, turning its error reply into ours.
fn ask_daemon(req: syrinx_client::ipc::Request) -> Result<()> {
    match syrinx_client::ipc::request(&req)? {
        syrinx_client::ipc::Response::Error { message } => {
            anyhow::bail!("the daemon refused: {message}")
        }
        _ => Ok(()),
    }
}

/// Block until asked to stop: SIGTERM or Ctrl-C on Unix, a stop request or
/// Ctrl-C on Windows.
fn wait_for_stop_signal(pid_path: &Path) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("building a runtime for signal handling");
    rt.block_on(async {
        #[cfg(unix)]
        {
            let _ = pid_path;
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("installing a SIGTERM handler");
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        }
        #[cfg(windows)]
        {
            // Windows has no SIGTERM, so `stop` leaves a file instead. Polled
            // rather than watched: this is a human pressing a key, so a fifth
            // of a second is imperceptible and a directory watcher would be a
            // lot of machinery for it.
            let path = pid_path.to_path_buf();
            let watch = async {
                loop {
                    if state::take_stop_request(&path) {
                        return;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                }
            };
            tokio::select! {
                _ = watch => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        }
    });
}

/// Run the background daemon: session, tray and IPC socket.
///
/// The GUI cannot serve this purpose. winit documents `set_visible` as
/// unsupported on Wayland, so a window cannot hide itself and keep running;
/// something that never had a window has to own the session.
fn run_daemon(
    config: Option<PathBuf>,
    mode: Option<ModeArg>,
    source: Option<String>,
    save_each: bool,
    format: Option<FormatArg>,
) -> Result<()> {
    let cfg = Config::load(config)?;
    let format = resolve_format(format, &cfg);
    let mode = mode.map(OutputMode::from).unwrap_or(cfg.mode);
    let source_key = source.or_else(|| cfg.source_key.clone());

    // Signals stop the daemon, so ^C and `systemctl stop` behave.
    let stop_path = state::default_pid_path();
    std::thread::spawn(move || {
        wait_for_stop_signal(&stop_path);
        // Ask politely over IPC so the socket is cleaned up on the way out.
        let _ = syrinx_client::ipc::request(&syrinx_client::ipc::Request::Quit);
    });

    syrinx_client::daemon::run(syrinx_client::daemon::DaemonOptions {
        config: cfg,
        mode,
        source_keys: source_key.into_iter().collect(),
        source_mode: Default::default(),
        save_each,
        format,
        // Sits beside this binary, so a viewer opens from the same build.
        gui_command: std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("syrinx-gui")))
            .filter(|p| p.exists())
            .map(|p| p.display().to_string())
            .or_else(|| Some("syrinx-gui".into())),
    })
}

/// Print a live meter until interrupted.
///
/// Uses the same analysis the GUI draws, so the two agree about whether a
/// source is carrying audio.
fn run_meter(
    config: Option<PathBuf>,
    source: Option<String>,
    seconds: Option<u64>,
) -> Result<()> {
    use syrinx_audio::meter;

    let cfg = Config::load(config).ok();
    let want = source.or_else(|| cfg.and_then(|c| c.source_key));
    let sources = syrinx_client::list_sources()?;
    let src = syrinx_client::choose_source(&sources, want.as_deref())?;
    eprintln!("metering: {}  (ctrl-c to stop)", src.display());

    let preview = syrinx_client::preview::Preview::start(&src)?;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let stop = stop.clone();
        let stop_path = state::default_pid_path();
        std::thread::spawn(move || {
            wait_for_stop_signal(&stop_path);
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }

    // On a terminal, redraw one line in place. Redirected to a file or a pipe,
    // emit a line per sample instead -- carriage returns would produce one
    // enormous unreadable line.
    let interactive = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let deadline = seconds.map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));

    loop {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let bands = preview.levels();
        let rms = preview.rms();
        let line = format!("  [{}]  {:>4.0}%", meter::bars(&bands), rms * 100.0);
        use std::io::Write;
        if interactive {
            eprint!("\r{line}   ");
        } else {
            eprintln!("{line}");
        }
        let _ = std::io::stderr().flush();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if interactive {
        eprintln!();
    }
    Ok(())
}

/// Transcribe a file by streaming it through a normal session.
fn run_transcribe(
    config: Option<PathBuf>,
    file: PathBuf,
    save_to: Option<PathBuf>,
    format: Option<FormatArg>,
) -> Result<()> {
    let cfg = Config::load(config)?;
    let format = resolve_format(format, &cfg);
    eprintln!("decoding {}...", file.display());
    let samples = syrinx_client::bulk::decode(&file)?;
    let secs = syrinx_client::bulk::duration_secs(&samples);
    eprintln!("{secs:.1}s of audio; transcribing...");

    let started = std::time::Instant::now();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building a runtime")?;

    let text = rt.block_on(async {
        let mut last_shown = -1i32;
        syrinx_client::bulk::transcribe(&cfg.url, &cfg.token, &samples, |p| {
            // Whole percent only: a progress line that redraws hundreds of
            // times a second is just flicker.
            let pct = (p * 100.0) as i32;
            if pct != last_shown {
                last_shown = pct;
                eprint!("\r  {pct:>3}%");
                use std::io::Write;
                let _ = std::io::stderr().flush();
            }
        })
        .await
    })?;
    eprintln!();

    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "done in {elapsed:.1}s ({:.0}x real time)",
        if elapsed > 0.0 { secs / elapsed } else { 0.0 }
    );

    // Timestamped output needs per-fragment timings, which this path does not
    // record: a file is sent far faster than real time, so arrival order says
    // nothing about position in the recording. Say so rather than writing
    // times that would be wrong.
    if format == save::Format::Timestamped {
        eprintln!("note: timestamps are not available for file transcription; saving as plain");
    }

    match save_to {
        Some(p) => {
            save::write(&p, &text)?;
            eprintln!("saved to {}", p.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

/// Merge several concurrent sessions into one view, ordered by time.
fn merge_cli_states(
    handles: &[syrinx_client::SessionHandle],
) -> syrinx_client::SessionState {
    let states: Vec<_> = handles.iter().map(|h| h.state()).collect();
    if states.len() == 1 {
        return states[0].clone();
    }
    let mut segments: Vec<syrinx_client::session::Segment> =
        states.iter().flat_map(|s| s.segments.clone()).collect();
    // Ordering by time reconstructs what was said in what order across
    // sources, which is the point of transcribing them separately.
    segments.sort_by(|a, b| a.at.partial_cmp(&b.at).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = states.into_iter().next().unwrap_or_default();
    out.transcript = save::render(&segments, "", save::Format::Labelled);
    out.segments = segments;
    out
}

#[cfg(test)]
mod format_tests {
    use super::*;

    fn cfg_with(format: save::Format) -> Config {
        let mut c: Config = toml::from_str("token = \"t\"").unwrap();
        c.format = format;
        c
    }

    #[test]
    fn the_config_decides_when_no_flag_is_given() {
        // The flag carried default_value = "plain", so clap could not tell
        // "not given" from "asked for plain" and `format` in the config was
        // never consulted -- streaming was always plain prose.
        let cfg = cfg_with(save::Format::Timestamped);
        assert_eq!(resolve_format(None, &cfg), save::Format::Timestamped);
    }

    #[test]
    fn the_flag_wins_over_the_config() {
        let cfg = cfg_with(save::Format::Timestamped);
        assert_eq!(
            resolve_format(Some(FormatArg::Labelled), &cfg),
            save::Format::Labelled
        );
    }

    #[test]
    fn asking_for_plain_is_not_the_same_as_saying_nothing() {
        // The distinction the old default_value destroyed.
        let cfg = cfg_with(save::Format::Labelled);
        assert_eq!(
            resolve_format(Some(FormatArg::Plain), &cfg),
            save::Format::Plain
        );
        assert_eq!(resolve_format(None, &cfg), save::Format::Labelled);
    }

    #[test]
    fn every_flag_value_maps_to_its_format() {
        // A mismatch here would silently save in the wrong layout.
        for (arg, want) in [
            (FormatArg::Plain, save::Format::Plain),
            (FormatArg::Timestamped, save::Format::Timestamped),
            (FormatArg::Labelled, save::Format::Labelled),
        ] {
            assert_eq!(save::Format::from(arg), want);
        }
    }
}
