//! The dictation session. One implementation, shared by the CLI and the GUI.
//!
//! Runs its own tokio runtime on a dedicated thread and communicates through
//! shared state, so a front-end never blocks on the network: a stalled server
//! cannot freeze a GUI or wedge a CLI.

use crate::inject;
use crate::mode::OutputMode;
use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use syrinx_audio::{Capture, Source};
use syrinx_proto::{ClientMessage, Encoding, SAMPLE_RATE, ServerMessage};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    #[default]
    Idle,
    Connecting,
    Listening,
    Stopping,
    /// Working through an audio file rather than a live source.
    Transcribing,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Idle => "Idle",
            Status::Connecting => "Connecting…",
            Status::Listening => "LISTENING",
            Status::Stopping => "Stopping…",
            Status::Transcribing => "TRANSCRIBING FILE",
        }
    }

    pub fn is_active(self) -> bool {
        !matches!(self, Status::Idle)
    }
}

/// One fragment of transcript with the time it arrived.
///
/// Kept alongside the flat string so a transcript can be saved with timestamps
/// without the caller having to reconstruct timing it never saw.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Seconds since the session started.
    pub at: f64,
    pub text: String,
    /// Which source produced this, when more than one is in play. `None` for a
    /// single source or a combined mix, where attribution is meaningless.
    pub source: Option<String>,
}

/// Everything a front-end needs to render. Cheap to clone.
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    pub status: Status,
    /// Accumulated text. Empty in [`OutputMode::Type`], which keeps none.
    pub transcript: String,
    /// The same text, split by arrival with timings.
    pub segments: Vec<Segment>,
    /// Most recent fragment, so a UI can show it is alive.
    pub last_fragment: String,
    pub model: Option<String>,
    pub chunk_ms: Option<u32>,
    pub error: Option<String>,
    /// Live spectrum of the audio being sent, so a viewer can see the session
    /// is receiving sound rather than only that it is running.
    pub levels: Vec<f32>,
    pub rms: f32,
}

/// Parameters for a run.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub url: String,
    pub token: String,
    /// Sources to capture. More than one is mixed into a single stream; for
    /// independent streams the caller runs a session per source.
    pub sources: Vec<Source>,
    pub mode: OutputMode,
    /// Label applied to this session's segments, for separate mode.
    pub label: Option<String>,
    /// How text is typed at the cursor.
    pub inject: crate::inject::Method,
    /// Append each committed fragment to this file as it arrives.
    pub stream: Option<(std::path::PathBuf, crate::save::Format)>,
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

    pub fn is_running(&self) -> bool {
        self.state().status.is_active()
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start a session on a background thread.
///
/// `on_change` fires whenever state moves, so a UI can repaint on demand rather
/// than polling.
pub fn start(
    opts: SessionOptions,
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
                fail(&st, format!("starting the async runtime: {e}"));
                notify();
                return;
            }
        };
        match rt.block_on(run(opts, st.clone(), stop_rx, notify.clone())) {
            Ok(()) => st.lock().expect("state lock poisoned").status = Status::Idle,
            Err(e) => {
                error!("session failed: {e:#}");
                fail(&st, format!("{e:#}"));
            }
        }
        notify();
    });

    SessionHandle {
        state,
        stop: Some(stop_tx),
    }
}

fn fail(state: &Arc<Mutex<SessionState>>, msg: String) {
    let mut s = state.lock().expect("state lock poisoned");
    s.error = Some(msg);
    s.status = Status::Idle;
}

async fn run(
    opts: SessionOptions,
    state: Arc<Mutex<SessionState>>,
    mut stop_rx: oneshot::Receiver<()>,
    notify: Arc<impl Fn() + Send + Sync + 'static>,
) -> Result<()> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message as Ws;

    if opts.mode.types_at_cursor() {
        // Fail before the microphone opens rather than after the user speaks.
        inject::preflight(opts.inject)?;
    }

    let mut req = opts
        .url
        .as_str()
        .into_client_request()
        .with_context(|| format!("building a request for {}", opts.url))?;
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", opts.token)
            .parse()
            .context("token is not a valid header value")?,
    );

    let (ws, _) = tokio_tungstenite::connect_async(req)
        .await
        .with_context(|| format!("connecting to {}", opts.url))?;
    let (mut tx, mut rx) = ws.split();

    tx.send(Ws::Text(
        serde_json::to_string(&ClientMessage::SessionStart {
            // Typing forces the append-only wire mode; see OutputMode::wire_mode.
            mode: opts.mode.wire_mode(),
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

    // Timestamps are relative to the moment audio starts flowing, not to
    // connection, so they line up with the recording rather than including
    // model load time.
    let started = std::time::Instant::now();

    let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(32);
    // One source opens directly; several are mixed. A single source through the
    // mixer would work but adds a queue and a timer for nothing.
    let _capture: Box<dyn std::any::Any + Send> = if opts.sources.len() == 1 {
        Box::new(Capture::start(&opts.sources[0], audio_tx).context("starting audio capture")?)
    } else {
        Box::new(
            syrinx_audio::mixer::MixedCapture::start(&opts.sources, audio_tx)
                .context("starting the combined capture")?,
        )
    };
    info!(
        "session running: {} -> {}",
        opts.sources
            .iter()
            .map(|s| s.display())
            .collect::<Vec<_>>()
            .join(" + "),
        opts.mode.label()
    );

    // Opened before the session starts, so an unwritable path is refused now
    // rather than discovered after an hour of talking.
    let mut stream = match &opts.stream {
        Some((path, format)) => Some(
            crate::stream::StreamWriter::open(path, *format)
                .context("opening the transcript stream")?,
        ),
        None => None,
    };

    let st = state.clone();
    let n = notify.clone();
    let mode = opts.mode;
    let label = opts.label.clone();
    let inject_method = opts.inject;
    let reader = tokio::spawn(async move {
        while let Some(Ok(msg)) = rx.next().await {
            let Ws::Text(t) = msg else { continue };
            match serde_json::from_str::<ServerMessage>(&t) {
                Ok(ServerMessage::TranscriptCommit { text, .. })
                | Ok(ServerMessage::TranscriptProvisional { text, .. }) => {
                    if mode.types_at_cursor()
                        && let Err(e) = inject::type_text(&text, inject_method)
                    {
                        error!("failed to type {text:?}: {e:#}");
                    }
                    let seg = Segment {
                        at: started.elapsed().as_secs_f64(),
                        text: text.clone(),
                        source: label.clone(),
                    };
                    // Written whatever the mode: the point of streaming is a
                    // copy on disk, and typing at the cursor is exactly when
                    // no transcript is kept in memory to save later.
                    if let Some(w) = stream.as_mut()
                        && let Err(e) = w.append(&seg)
                    {
                        error!("failed to append to the transcript stream: {e:#}");
                    }
                    let mut s = st.lock().expect("state lock poisoned");
                    if mode.keeps_transcript() {
                        s.transcript.push_str(&text);
                        s.segments.push(seg);
                    }
                    s.last_fragment = text;
                    drop(s);
                    n();
                }
                Ok(ServerMessage::TranscriptRevise {
                    retract_n, text, ..
                }) => {
                    // Only reachable in transcribe-only mode, where the client
                    // owns the buffer. A typing session runs in Live mode, so
                    // the server never sends this.
                    let mut s = st.lock().expect("state lock poisoned");
                    let keep = s.transcript.chars().count().saturating_sub(retract_n);
                    s.transcript = s.transcript.chars().take(keep).collect();
                    s.transcript.push_str(&text);
                    // Revisions rewrite the tail, so the affected segments are
                    // replaced rather than appended to.
                    let mut dropped = 0usize;
                    while dropped < retract_n {
                        match s.segments.pop() {
                            Some(seg) => dropped += seg.text.chars().count(),
                            None => break,
                        }
                    }
                    s.segments.push(Segment {
                        at: started.elapsed().as_secs_f64(),
                        text: text.clone(),
                        source: label.clone(),
                    });
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

    // Metered from the audio actually being sent, so what a viewer sees is the
    // stream the server receives rather than a second capture of the same
    // device.
    let mut window: Vec<f32> = Vec::with_capacity(4096);
    loop {
        tokio::select! {
            _ = &mut stop_rx => break,
            chunk = audio_rx.recv() => match chunk {
                Some(samples) => {
                    window.extend_from_slice(&samples);
                    if window.len() > 4096 {
                        let excess = window.len() - 4096;
                        window.drain(..excess);
                    }
                    {
                        let mut s = state.lock().expect("state lock poisoned");
                        s.levels = syrinx_audio::meter::spectrum(&window, SAMPLE_RATE).to_vec();
                        s.rms = syrinx_audio::meter::rms(&samples);
                    }
                    if tx.send(Ws::Binary(to_pcm_s16le(&samples).into())).await.is_err() {
                        break;
                    }
                }
                None => break,
            }
        }
    }

    state.lock().expect("state lock poisoned").status = Status::Stopping;
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
    fn out_of_range_input_is_clamped_not_wrapped() {
        let out = to_pcm_s16le(&[9.0, -9.0]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
        assert_eq!(i16::from_le_bytes([out[2], out[3]]), -32767);
    }

    #[test]
    fn encoding_round_trips_through_the_shared_decoder() {
        // The server decodes with syrinx_proto::pcm_s16le_to_f32, so the two
        // must agree or every transcript is subtly wrong.
        let original = [0.0f32, 0.5, -0.5, 0.25];
        let decoded = syrinx_proto::pcm_s16le_to_f32(&to_pcm_s16le(&original));
        for (a, b) in original.iter().zip(&decoded) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn status_labels_are_distinct() {
        let all = [
            Status::Idle,
            Status::Connecting,
            Status::Listening,
            Status::Stopping,
        ];
        let mut l: Vec<&str> = all.iter().map(|s| s.label()).collect();
        l.sort_unstable();
        let n = l.len();
        l.dedup();
        assert_eq!(l.len(), n);
    }

    #[test]
    fn only_idle_is_inactive() {
        assert!(!Status::Idle.is_active());
        assert!(Status::Connecting.is_active());
        assert!(Status::Listening.is_active());
        assert!(Status::Stopping.is_active());
    }
}
