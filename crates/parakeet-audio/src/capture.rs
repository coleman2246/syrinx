//! Capturing from any PipeWire node via `pw-record`.
//!
//! `pw-record` is used as a subprocess rather than binding libpipewire because
//! it already handles graph negotiation, format conversion and resampling, and
//! can target any node -- microphone, monitor, or a single application's stream.
//! It writes raw PCM to stdout, which is exactly the shape needed here.
//!
//! Capturing an application's stream is a **tap**: the application keeps playing
//! to its normal output. Nothing is rerouted, so transcribing a video does not
//! silence it.

use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Bytes read per poll. 16 kHz mono s16 is 32 kB/s, so this is ~0.1s of audio:
/// small enough to stay responsive, large enough to avoid syscall churn.
const READ_CHUNK: usize = 4096;

/// A running capture. Dropping it stops the capture.
pub struct Capture {
    child: Child,
}

impl Capture {
    /// Start capturing 16 kHz mono f32 samples from a PipeWire node.
    ///
    /// `pw-record` handles the rate conversion, so whatever the node runs at
    /// natively, samples arrive at the rate the model expects.
    pub fn start(node_id: u32, tx: mpsc::Sender<Vec<f32>>) -> Result<Self> {
        let mut child = Command::new("pw-record")
            .args([
                "--target",
                &node_id.to_string(),
                "--rate",
                &parakeet_proto::SAMPLE_RATE.to_string(),
                "--channels",
                "1",
                "--format",
                "s16",
                // "-" writes raw PCM to stdout, with no WAV header to skip.
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning pw-record (is PipeWire installed?)")?;

        let mut stdout = child.stdout.take().context("pw-record stdout missing")?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(log_stderr(stderr));
        }

        tokio::spawn(async move {
            // s16 samples can straddle reads, so carry the odd byte over rather
            // than decoding it as half a sample.
            let mut carry: Vec<u8> = Vec::new();
            let mut buf = vec![0u8; READ_CHUNK];
            loop {
                match stdout.read(&mut buf).await {
                    Ok(0) => {
                        debug!("pw-record closed its output");
                        break;
                    }
                    Ok(n) => {
                        carry.extend_from_slice(&buf[..n]);
                        let usable = carry.len() - (carry.len() % 2);
                        let samples = parakeet_proto::pcm_s16le_to_f32(&carry[..usable]);
                        carry.drain(..usable);
                        // try_send: if the consumer is behind, dropping audio
                        // beats growing a backlog and drifting off real time.
                        if tx.try_send(samples).is_err() && tx.is_closed() {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("reading from pw-record: {e}");
                        break;
                    }
                }
            }
        });

        info!("capturing from PipeWire node {node_id}");
        Ok(Self { child })
    }

    /// Stop capture. Also happens on drop; this just allows awaiting it.
    pub async fn stop(mut self) {
        let _ = self.child.kill().await;
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // start_kill rather than kill().await: Drop cannot be async, and
        // leaking a pw-record process would hold the mic open indefinitely.
        let _ = self.child.start_kill();
    }
}

async fn log_stderr(stderr: tokio::process::ChildStderr) {
    let mut s = String::new();
    let mut r = tokio::io::BufReader::new(stderr);
    if r.read_to_string(&mut s).await.is_ok() && !s.trim().is_empty() {
        warn!("pw-record: {}", s.trim());
    }
}
