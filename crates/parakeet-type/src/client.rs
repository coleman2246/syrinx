//! WebSocket client: stream captured audio up, type committed text down.

use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use parakeet_proto::{ClientMessage, Encoding, Mode, SAMPLE_RATE, ServerMessage};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;
use tracing::{error, info, warn};

use crate::inject::type_text;

/// Convert f32 samples to little-endian s16 for the wire.
///
/// s16 halves bandwidth against f32 and is what the server's decoder expects.
fn to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        // 32767 rather than 32768: scaling by 32768 lets +1.0 overflow i16.
        let v = (clamped * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Run one dictation session until `audio_rx` closes or the server ends it.
pub async fn run_session(
    url: &str,
    token: &str,
    mut audio_rx: mpsc::Receiver<Vec<f32>>,
    mut stop_rx: tokio::sync::oneshot::Receiver<()>,
) -> Result<()> {
    let mut req = url
        .into_client_request()
        .with_context(|| format!("building request for {url}"))?;
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

    tx.send(WsMessage::Text(
        serde_json::to_string(&ClientMessage::SessionStart {
            mode: Mode::Live,
            sample_rate: SAMPLE_RATE,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
        })?
        .into(),
    ))
    .await
    .context("sending session.start")?;

    // The server may refuse before readying: bad token, at capacity, or the
    // VRAM guard protecting another tenant on the GPU.
    match rx.next().await {
        Some(Ok(WsMessage::Text(t))) => match serde_json::from_str::<ServerMessage>(&t)? {
            ServerMessage::SessionReady { model, chunk_ms, .. } => {
                info!("session ready: model={model} chunk_ms={chunk_ms}");
            }
            ServerMessage::Error { code, message, .. } => {
                bail!("server refused the session ({code:?}): {message}")
            }
            other => bail!("expected session.ready, got {other:?}"),
        },
        Some(Ok(other)) => bail!("expected a text frame, got {other:?}"),
        Some(Err(e)) => return Err(e).context("reading session.ready"),
        None => bail!("server closed the connection before session.ready"),
    }

    // Reader: type each committed fragment as it arrives.
    let reader = tokio::spawn(async move {
        while let Some(Ok(msg)) = rx.next().await {
            let WsMessage::Text(t) = msg else { continue };
            match serde_json::from_str::<ServerMessage>(&t) {
                Ok(ServerMessage::TranscriptCommit { text, .. }) => {
                    if let Err(e) = type_text(&text) {
                        error!("failed to type {text:?}: {e:#}");
                    }
                }
                Ok(ServerMessage::SessionClosed { reason }) => {
                    info!("server closed the session: {reason}");
                    break;
                }
                Ok(ServerMessage::Error { code, message, .. }) => {
                    error!("server error ({code:?}): {message}");
                    break;
                }
                // Live mode is append-only; the server never sends these.
                Ok(other) => warn!("unexpected message in live mode: {other:?}"),
                Err(e) => warn!("undecodable server frame: {e}"),
            }
        }
    });

    // Writer: pump audio until stopped.
    loop {
        tokio::select! {
            _ = &mut stop_rx => {
                info!("stop requested");
                break;
            }
            chunk = audio_rx.recv() => {
                match chunk {
                    Some(samples) => {
                        if tx.send(WsMessage::Binary(to_pcm_s16le(&samples).into())).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
    }

    // Ask for the tail of the utterance rather than cutting mid-word.
    let _ = tx
        .send(WsMessage::Text(
            serde_json::to_string(&ClientMessage::SessionStop)?.into(),
        ))
        .await;

    // Bounded: a hung server must not leave the client running forever with the
    // mic live.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(10), reader).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_encodes_to_zero_bytes() {
        assert_eq!(to_pcm_s16le(&[0.0, 0.0]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn full_scale_does_not_wrap() {
        // Scaling by 32768 would make +1.0 overflow to -32768, turning the
        // loudest sample into the quietest -- an audible click, and a bug that
        // only shows up on loud speech.
        let out = to_pcm_s16le(&[1.0]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
    }

    #[test]
    fn negative_full_scale_is_clamped() {
        let out = to_pcm_s16le(&[-1.0]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), -32767);
    }

    #[test]
    fn out_of_range_input_is_clamped_not_wrapped() {
        let out = to_pcm_s16le(&[2.5, -2.5]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
        assert_eq!(i16::from_le_bytes([out[2], out[3]]), -32767);
    }

    #[test]
    fn round_trips_through_the_shared_decoder() {
        // The server decodes with parakeet_proto::pcm_s16le_to_f32, so encoding
        // and decoding must agree.
        let original = [0.0f32, 0.5, -0.5];
        let decoded = parakeet_proto::pcm_s16le_to_f32(&to_pcm_s16le(&original));
        for (a, b) in original.iter().zip(&decoded) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}
