//! Transcribing an audio file.
//!
//! Feeds the file through the same streaming session a microphone uses, rather
//! than a separate upload-and-poll job API. The model is a streaming one: it
//! consumes 560 ms chunks and emits text as it goes, so a file is just a source
//! of chunks that happens to arrive faster than real time. Reusing the session
//! path means one protocol, one code path, and a file transcribes with exactly
//! the behaviour a live session has.
//!
//! Decoding is delegated to ffmpeg. Supporting MP3, M4A, Opus, FLAC and the
//! rest by hand would be a large amount of code to reimplement something every
//! machine already has.

use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use std::path::Path;
use syrinx_proto::{ClientMessage, Encoding, Mode, SAMPLE_RATE, ServerMessage};

/// How many samples to send per WebSocket frame.
///
/// One model chunk. Larger frames would be fine over a socket, but matching the
/// server's chunk size keeps its buffering behaviour identical to a live
/// session.
const CHUNK: usize = 8960;

/// Decode any audio file ffmpeg understands into 16 kHz mono f32.
pub fn decode(path: &Path) -> Result<Vec<f32>> {
    if !path.exists() {
        bail!("no such file: {}", path.display());
    }
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            // Raw PCM on stdout, no container to parse.
            "-",
        ])
        .output()
        .map_err(ffmpeg_spawn_error)?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        bail!("ffmpeg could not decode {}: {}", path.display(), err.trim());
    }
    let samples = syrinx_proto::pcm_s16le_to_f32(&out.stdout);
    if samples.is_empty() {
        bail!("{} contains no audio", path.display());
    }
    Ok(samples)
}

/// Explain why ffmpeg could not be started.
///
/// "is it installed?" was the whole message, which is wrong whenever it is
/// installed but unreachable -- and on Windows that is common: winget puts a
/// zero-length reparse point on PATH, and spawning through it fails with
/// "untrusted mount point" even though `ffmpeg -version` works in a shell.
/// Being told to install what you already installed sends you the wrong way.
fn ffmpeg_spawn_error(e: std::io::Error) -> anyhow::Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        return anyhow::anyhow!("ffmpeg is not on PATH. Install it to transcribe files.");
    }
    anyhow::Error::new(e).context(
        "could not run ffmpeg. It appears to be on PATH but could not be started; \
         if PATH points at a shim or symlink, put the real ffmpeg directory on PATH instead",
    )
}

/// Duration of a decoded buffer, in seconds.
pub fn duration_secs(samples: &[f32]) -> f64 {
    samples.len() as f64 / SAMPLE_RATE as f64
}

/// Transcribe decoded audio, reporting progress as a fraction from 0.0 to 1.0.
pub async fn transcribe(
    url: &str,
    token: &str,
    samples: &[f32],
    mut progress: impl FnMut(f32),
) -> Result<String> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message as Ws;

    crate::install_crypto_provider();
    let mut req = url
        .into_client_request()
        .with_context(|| format!("building a request for {url}"))?;
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {token}")
            .parse()
            .context("token is not a valid header value")?,
    );
    let (ws, _) = tokio_tungstenite::connect_async(req)
        .await
        .with_context(|| format!("connecting to {url}"))?;
    let (mut tx, mut rx) = ws.split();

    // Transcript mode: nothing is typed, and the server is free to revise.
    tx.send(Ws::Text(
        serde_json::to_string(&ClientMessage::SessionStart {
            mode: Mode::Transcript,
            sample_rate: SAMPLE_RATE,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
        })?
        .into(),
    ))
    .await
    .context("sending session.start")?;

    match rx.next().await {
        Some(Ok(Ws::Text(t))) => match serde_json::from_str::<ServerMessage>(&t)? {
            ServerMessage::SessionReady { .. } => {}
            ServerMessage::Error { code, message, .. } => {
                bail!("server refused the session ({code:?}): {message}")
            }
            other => bail!("expected session.ready, got {other:?}"),
        },
        Some(Ok(other)) => bail!("expected a text frame, got {other:?}"),
        Some(Err(e)) => return Err(e).context("reading session.ready"),
        None => bail!("server closed the connection before session.ready"),
    }

    // Collect text while sending, rather than after. The server emits as it
    // decodes, and a socket whose replies are never read will eventually stall
    // on a full buffer.
    let (text_tx, mut text_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let reader = tokio::spawn(async move {
        while let Some(Ok(Ws::Text(t))) = rx.next().await {
            match serde_json::from_str::<ServerMessage>(&t) {
                Ok(ServerMessage::TranscriptCommit { text, .. })
                | Ok(ServerMessage::TranscriptProvisional { text, .. }) => {
                    let _ = text_tx.send(text);
                }
                Ok(ServerMessage::SessionClosed { .. }) => break,
                Ok(ServerMessage::Error { message, .. }) => {
                    let _ = text_tx.send(format!("\n[server error: {message}]"));
                    break;
                }
                _ => {}
            }
        }
    });

    let total = samples.len().max(1);
    for (i, chunk) in samples.chunks(CHUNK).enumerate() {
        let mut buf = chunk.to_vec();
        // The model wants whole chunks; the tail is padded rather than dropped
        // so the last words are not lost.
        if buf.len() < CHUNK {
            buf.resize(CHUNK, 0.0);
        }
        if tx
            .send(Ws::Binary(to_pcm_s16le(&buf).into()))
            .await
            .is_err()
        {
            break;
        }
        progress(((i + 1) * CHUNK).min(total) as f32 / total as f32);
    }

    tx.send(Ws::Text(
        serde_json::to_string(&ClientMessage::SessionStop)?.into(),
    ))
    .await
    .ok();

    // Generous: the server may still be decoding a backlog of chunks sent far
    // faster than real time.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(120), reader).await;

    let mut out = String::new();
    while let Ok(t) = text_rx.try_recv() {
        out.push_str(&t);
    }
    progress(1.0);
    Ok(out.trim().to_string())
}

fn to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_a_clear_error() {
        let e = decode(Path::new("/nonexistent/audio.mp3")).unwrap_err();
        assert!(e.to_string().contains("no such file"), "got {e}");
    }

    #[test]
    fn a_missing_ffmpeg_says_to_install_it() {
        let e = ffmpeg_spawn_error(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(format!("{e:#}").contains("not on PATH"), "got {e:#}");
    }

    #[test]
    fn an_unreachable_ffmpeg_does_not_say_to_install_it() {
        // The winget shim case: telling the user to install what they have
        // installed sends them looking in the wrong place.
        let e = ffmpeg_spawn_error(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        let text = format!("{e:#}");
        assert!(!text.contains("Install it"), "got {text}");
        assert!(text.contains("shim or symlink"), "got {text}");
    }

    #[test]
    fn duration_is_computed_from_the_sample_rate() {
        assert!((duration_secs(&vec![0.0; 16_000]) - 1.0).abs() < 1e-9);
        assert_eq!(duration_secs(&[]), 0.0);
    }

    #[test]
    fn encoding_round_trips_through_the_shared_decoder() {
        let original = [0.0f32, 0.5, -0.5];
        let decoded = syrinx_proto::pcm_s16le_to_f32(&to_pcm_s16le(&original));
        for (a, b) in original.iter().zip(&decoded) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}
