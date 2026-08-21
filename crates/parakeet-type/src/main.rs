//! Headless live dictation for Wayland: speak, and text appears at the cursor.
//!
//! Replaces nerd-dictation + Vosk. Runs in live mode, which is append-only by
//! design: the server never asks this client to delete text, because it types
//! into whatever window has focus.

mod capture;
mod client;
mod inject;
mod state;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

/// Audio frames queued between capture and the socket. Shallow on purpose: if
/// the network stalls, dropping audio beats drifting further behind real time.
const AUDIO_QUEUE_DEPTH: usize = 32;

#[derive(Parser)]
#[command(about = "Live voice dictation that types at the cursor")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,

    /// Config file. Defaults to ~/.config/parakeet-type/config.toml
    #[arg(long, global = true)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start dictating (blocks until stopped).
    Start,
    /// Stop a running instance.
    Stop,
    /// Start if idle, stop if running. Bind this to a key.
    Toggle,
    /// Print whether dictation is active.
    Status,
}

#[derive(Debug, Deserialize)]
struct Config {
    #[serde(default = "default_url")]
    url: String,
    token: String,
    /// waybar realtime signal number for the indicator, matching
    /// `"signal": N` in the waybar module.
    #[serde(default = "default_waybar_signal")]
    waybar_signal: u8,
}

fn default_url() -> String {
    "ws://127.0.0.1:8770/v1/stream".into()
}
fn default_waybar_signal() -> u8 {
    8
}

fn config_path(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| {
        let base = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".config")
            });
        base.join("parakeet-type/config.toml")
    })
}

fn load_config(path: &PathBuf) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config from {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "parakeet_type=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let pid_path = state::default_pid_path();

    match cli.command {
        Cmd::Status => {
            match state::running_pid(&pid_path) {
                Some(pid) => println!("dictating (pid {pid})"),
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

        Cmd::Toggle => {
            if let Some(pid) = state::running_pid(&pid_path) {
                state::signal_stop(pid)?;
                info!("stopping pid {pid}");
                return Ok(());
            }
            let cfg = load_config(&config_path(cli.config))?;
            dictate(cfg, pid_path).await
        }

        Cmd::Start => {
            if let Some(pid) = state::running_pid(&pid_path) {
                anyhow::bail!("already dictating (pid {pid})");
            }
            let cfg = load_config(&config_path(cli.config))?;
            dictate(cfg, pid_path).await
        }
    }
}

async fn dictate(cfg: Config, pid_path: PathBuf) -> Result<()> {
    // Fail before the mic goes live, not after the user has spoken a sentence.
    inject::preflight()?;

    state::write_pid(&pid_path, std::process::id() as i32)?;
    state::refresh_waybar(cfg.waybar_signal);

    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(AUDIO_QUEUE_DEPTH);
    let (stop_tx, stop_rx) = oneshot::channel();

    // The cpal stream must stay alive for the whole session; dropping it stops
    // capture. It is not Send, so it stays on this thread.
    let _stream = capture::start(audio_tx)?;

    // SIGTERM is how `stop` and `toggle` ask us to finish. Handling it (rather
    // than dying) lets the session flush its final words before exiting.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("installing SIGTERM handler")?;
    tokio::spawn(async move {
        sigterm.recv().await;
        let _ = stop_tx.send(());
    });

    info!("dictating -- speak now");
    let result = client::run_session(&cfg.url, &cfg.token, audio_rx, stop_rx).await;

    // Always clean up, even on error: a stale PID file would make the next
    // toggle believe dictation is still running and refuse to start.
    state::clear_pid(&pid_path);
    state::refresh_waybar(cfg.waybar_signal);

    if let Err(e) = &result {
        error!("session ended with an error: {e:#}");
    }
    info!("stopped");
    result
}
