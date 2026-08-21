//! The dictation session: capture, stream, and report back to the UI.
//!
//! Runs on a tokio runtime on its own thread. The UI never blocks on the
//! network, and communicates only through [`SessionHandle`], so a stalled
//! server cannot freeze the window.

use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use parakeet_audio::{Capture, Source};
use parakeet_proto::{ClientMessage, Encoding, Mode, SAMPLE_RATE, ServerMessage};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

/// What the UI needs to draw. Cheap to clone and read every frame.
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    pub status: Status,
    /// Everything transcribed this session.
    pub transcript: String,
    /// The most recent fragment, shown so the user can see it is alive.
    pub last_fragment: String,
    pub model: Option<String>,
    pub chunk_ms: Option<u32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Status {
    #[default]
    Idle,
    Connecting,
    Listening,
    Stopping,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Idle => "Idle",
            Status::Connecting => "Connecting…",
            Status::Listening => "LISTENING",
            Status::Stopping => "Stopping…",
        }
    }
}

/// Handle to a running session. Dropping it stops the session.
pub struct SessionHandle {
    state: Arc<Mutex<SessionState>>,
    stop: Option<oneshot::Sender<()>>,
}

impl SessionHandle {
    pub fn state(&self) -> SessionState {
        self.state.lock().expect("state lock poisoned").clone()
    }

    pub fn stop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start a session against `source`, typing nothing -- the GUI displays text
/// rather than injecting it.
///
/// `on_change` is called whenever state moves, so the UI can request a repaint
/// instead of polling at a fixed rate.
pub fn start(
    url: String,
    token: String,
    source: Source,
    on_change: impl Fn() + Send + Sync + 'static,
) -> SessionHandle {
    let state = Arc::new(Mutex::new(SessionState {
        status: Status::Connecting,
        ..Default::default()
    }));
    let (stop_tx, stop_rx) = oneshot::channel();

    let st = state.clone();
    let notify = Arc::new(on_change);
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                set_error(&st, format!("starting the async runtime: {e}"));
                notify();
                return;
            }
        };
        if let Err(e) = rt.block_on(run(url, token, source, st.clone(), stop_rx, notify.clone())) {
            error!("session failed: {e:#}");
            set_error(&st, format!("{e:#}"));
        } else {
            let mut s = st.lock().expect("state lock poisoned");
            s.status = Status::Idle;
        }
        notify();
    });

    SessionHandle {
        state,
        stop: Some(stop_tx),
    }
}

fn set_error(state: &Arc<Mutex<SessionState>>, msg: String) {
    let mut s = state.lock().expect("state lock poisoned");
    s.error = Some(msg);
    s.status = Status::Idle;
}

async fn run(
    url: String,
    token: String,
    source: Source,
    state: Arc<Mutex<SessionState>>,
    mut stop_rx: oneshot::Receiver<()>,
    notify: Arc<impl Fn() + Send + Sync + 'static>,
) -> Result<()> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message as Ws;

    let mut req = url
        .as_str()
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

    // Transcript mode: the GUI owns its buffer, so the server may revise it.
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
            ServerMessage::SessionReady {
                model, chunk_ms, ..
            } => {
                let mut s = state.lock().expect("state lock poisoned");
                s.status = Status::Listening;
                s.model = Some(model);
                s.chunk_ms = Some(chunk_ms);
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
    notify();

    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(32);
    let _capture = Capture::start(&source, audio_tx).context("starting audio capture")?;
    info!("session running against {}", source.display());

    let st = state.clone();
    let n = notify.clone();
    let reader = tokio::spawn(async move {
        while let Some(Ok(msg)) = rx.next().await {
            let Ws::Text(t) = msg else { continue };
            match serde_json::from_str::<ServerMessage>(&t) {
                Ok(ServerMessage::TranscriptCommit { text, .. })
                | Ok(ServerMessage::TranscriptProvisional { text, .. }) => {
                    let mut s = st.lock().expect("state lock poisoned");
                    s.transcript.push_str(&text);
                    s.last_fragment = text;
                    drop(s);
                    n();
                }
                Ok(ServerMessage::TranscriptRevise {
                    retract_n, text, ..
                }) => {
                    // Transcript mode permits revision because the client owns
                    // this buffer. No v1 server path emits it, but honouring it
                    // here means a future post-processing layer needs no client
                    // change.
                    let mut s = st.lock().expect("state lock poisoned");
                    let keep = s.transcript.chars().count().saturating_sub(retract_n);
                    s.transcript = s.transcript.chars().take(keep).collect();
                    s.transcript.push_str(&text);
                    drop(s);
                    n();
                }
                Ok(ServerMessage::SessionClosed { .. }) => break,
                Ok(ServerMessage::Error { code, message, .. }) => {
                    warn!("server error ({code:?}): {message}");
                    let mut s = st.lock().expect("state lock poisoned");
                    s.error = Some(message);
                    drop(s);
                    n();
                    break;
                }
                Ok(_) => {}
                Err(e) => warn!("undecodable server frame: {e}"),
            }
        }
    });

    loop {
        tokio::select! {
            _ = &mut stop_rx => break,
            chunk = audio_rx.recv() => match chunk {
                Some(samples) => {
                    if tx.send(Ws::Binary(to_pcm_s16le(&samples).into())).await.is_err() {
                        break;
                    }
                }
                None => break,
            }
        }
    }

    {
        let mut s = state.lock().expect("state lock poisoned");
        s.status = Status::Stopping;
    }
    notify();

    let _ = tx
        .send(Ws::Text(
            serde_json::to_string(&ClientMessage::SessionStop)?.into(),
        ))
        .await;
    // Bounded: a hung server must not leave the microphone open forever.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(10), reader).await;
    Ok(())
}

/// Encode to little-endian s16 for the wire.
fn to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        // 32767, not 32768: scaling by 32768 lets +1.0 overflow i16 and wrap to
        // the most negative value, turning the loudest sample into a click.
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_does_not_wrap() {
        let out = to_pcm_s16le(&[1.0, -1.0]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
        assert_eq!(i16::from_le_bytes([out[2], out[3]]), -32767);
    }

    #[test]
    fn out_of_range_input_is_clamped() {
        let out = to_pcm_s16le(&[9.0]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
    }

    #[test]
    fn status_labels_are_distinct() {
        // The status line is the only thing telling the user whether the mic is
        // live; two states reading the same would be actively misleading.
        let all = [
            Status::Idle,
            Status::Connecting,
            Status::Listening,
            Status::Stopping,
        ];
        let mut labels: Vec<&str> = all.iter().map(|s| s.label()).collect();
        labels.sort_unstable();
        let before = labels.len();
        labels.dedup();
        assert_eq!(labels.len(), before);
    }
}
