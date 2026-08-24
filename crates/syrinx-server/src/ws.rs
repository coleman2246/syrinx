//! WebSocket transport.
//!
//! Owns framing and knows nothing about recognition; [`crate::session`] owns
//! semantics and knows nothing about sockets.
//!
//! Each session runs three cooperating parts:
//!
//! - the socket reader, which decodes frames and pushes audio into a **bounded**
//!   channel, so a client that outruns the server produces explicit
//!   backpressure rather than unbounded memory growth;
//! - an inference task on the blocking pool, because inference is synchronous
//!   and CPU/GPU-bound, and running it on the async runtime would stall every
//!   other session;
//! - the socket writer, draining emitted messages back to the client.

use crate::asr::lifecycle::ModelHandle;
use crate::auth::check_bearer;
use crate::config::Config;
use crate::session::Session;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use syrinx_proto::{ClientMessage, ErrorCode, Mode, ServerMessage};
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};
use tracing::{debug, info, warn};

/// How many audio chunks may queue between the socket and inference before the
/// reader blocks. Small on purpose: a large buffer just hides that the server is
/// falling behind while drifting further from real time.
const AUDIO_QUEUE_DEPTH: usize = 8;

#[derive(Clone)]
pub struct AppState {
    /// Lazily loaded. The server holds zero VRAM until a session arrives.
    pub model: Arc<ModelHandle>,
    pub config: Arc<Config>,
    /// Admission control. Sessions beyond the limit are refused outright rather
    /// than accepted into a pool that degrades everyone.
    pub slots: Arc<Semaphore>,
}

impl AppState {
    pub fn new(model: Arc<ModelHandle>, config: Arc<Config>) -> Self {
        let slots = Arc::new(Semaphore::new(config.max_sessions));
        Self {
            model,
            config,
            slots,
        }
    }
}

pub async fn stream_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let provided = headers.get("authorization").and_then(|v| v.to_str().ok());
    if !check_bearer(provided, &state.config.token) {
        warn!("rejected connection: bad or missing bearer token");
        // Reject before upgrading. A client that cannot authenticate should
        // never reach a session.
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut tx, mut rx) = socket.split();

    // Admission control before any work is done.
    let Ok(_slot) = state.slots.clone().try_acquire_owned() else {
        let _ = tx.send(json_msg(&ServerMessage::Error {
            code: ErrorCode::Capacity,
            message: "server at session capacity".into(),
            retryable: true,
        }))
        .await;
        return;
    };

    // Await session.start before anything else.
    let mode = match wait_for_start(&mut rx, &mut tx).await {
        Some(m) => m,
        None => return,
    };

    // Load the model now, not at startup. Loading is blocking and can take a
    // couple of seconds, so it must not run on the async runtime.
    let handle = state.model.clone();
    let backend = match tokio::task::spawn_blocking(move || handle.get_or_load()).await {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => {
            // Capacity (a busy GPU) is transient and worth retrying; a failed
            // load is a misconfiguration that will fail identically forever.
            let retryable = e.is_retryable();
            warn!(retryable, "model unavailable: {e}");
            let _ = tx
                .send(json_msg(&ServerMessage::Error {
                    code: if retryable {
                        ErrorCode::Capacity
                    } else {
                        ErrorCode::Internal
                    },
                    message: e.message(),
                    retryable,
                }))
                .await;
            return;
        }
        Err(e) => {
            warn!("model load task failed: {e}");
            let _ = tx
                .send(json_msg(&ServerMessage::Error {
                    code: ErrorCode::Internal,
                    message: "model load failed".into(),
                    retryable: false,
                }))
                .await;
            return;
        }
    };

    let session_id = uuid::Uuid::new_v4().to_string();
    info!(%session_id, ?mode, "session started");

    let _ = tx
        .send(json_msg(&ServerMessage::SessionReady {
            session_id: session_id.clone(),
            chunk_ms: backend.chunk_ms(),
            model: backend.model_name().to_string(),
            // Always false for now; wiring this to a real diarizer is Task 7.
            diarize: false,
        }))
        .await;

    let (audio_tx, mut audio_rx) = mpsc::channel::<AudioEvent>(AUDIO_QUEUE_DEPTH);
    let (msg_tx, mut msg_rx) = mpsc::channel::<ServerMessage>(32);

    // Inference on the blocking pool: it is synchronous and GPU-bound.
    let sid = session_id.clone();
    let infer = tokio::task::spawn_blocking(move || {
        let mut session = Session::new(mode, backend.as_ref(), sid);
        while let Some(ev) = audio_rx.blocking_recv() {
            let is_finish = matches!(ev, AudioEvent::Finish);
            let produced = match ev {
                AudioEvent::Samples(s) => session.push_audio(&s),
                AudioEvent::Finish => session.finish(),
            };
            match produced {
                Ok(msgs) => {
                    for m in msgs {
                        if msg_tx.blocking_send(m).is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    let _ = msg_tx.blocking_send(ServerMessage::Error {
                        code: ErrorCode::Internal,
                        message: e.to_string(),
                        retryable: false,
                    });
                    return;
                }
            }
            // session.finish() drains the model; nothing follows it.
            if is_finish {
                return;
            }
        }
    });

    // Writer.
    let writer = tokio::spawn(async move {
        while let Some(m) = msg_rx.recv().await {
            if tx.send(json_msg(&m)).await.is_err() {
                break;
            }
        }
        let _ = tx
            .send(json_msg(&ServerMessage::SessionClosed {
                reason: "ended".into(),
            }))
            .await;
        let _ = tx.close().await;
    });

    // Reader.
    while let Some(Ok(frame)) = rx.next().await {
        match frame {
            Message::Binary(bytes) => {
                let samples = syrinx_proto::pcm_s16le_to_f32(&bytes);
                if audio_tx.send(AudioEvent::Samples(samples)).await.is_err() {
                    break;
                }
            }
            Message::Text(t) => match serde_json::from_str::<ClientMessage>(&t) {
                Ok(ClientMessage::SessionStop) | Ok(ClientMessage::SessionFlush) => {
                    let _ = audio_tx.send(AudioEvent::Finish).await;
                    break;
                }
                Ok(ClientMessage::SessionStart { .. }) => {
                    debug!("ignoring duplicate session.start");
                }
                Err(e) => {
                    warn!("malformed control frame: {e}");
                    break;
                }
            },
            Message::Close(_) => break,
            _ => {}
        }
    }

    drop(audio_tx);
    let _ = infer.await;
    let _ = writer.await;
    info!(%session_id, "session closed");
}

enum AudioEvent {
    Samples(Vec<f32>),
    Finish,
}

/// Wait for `session.start`. Anything else is a protocol error.
async fn wait_for_start(
    rx: &mut futures_util::stream::SplitStream<WebSocket>,
    tx: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) -> Option<Mode> {
    while let Some(Ok(frame)) = rx.next().await {
        match frame {
            Message::Text(t) => match serde_json::from_str::<ClientMessage>(&t) {
                Ok(ClientMessage::SessionStart { mode, .. }) => return Some(mode),
                _ => {
                    let _ = tx
                        .send(json_msg(&ServerMessage::Error {
                            code: ErrorCode::BadRequest,
                            message: "expected session.start".into(),
                            retryable: false,
                        }))
                        .await;
                    return None;
                }
            },
            Message::Binary(_) => {
                // Audio before session.start: the server has no mode, sample
                // rate or encoding yet, so it cannot interpret these bytes.
                let _ = tx
                    .send(json_msg(&ServerMessage::Error {
                        code: ErrorCode::BadRequest,
                        message: "audio received before session.start".into(),
                        retryable: false,
                    }))
                    .await;
                return None;
            }
            Message::Close(_) => return None,
            _ => {}
        }
    }
    None
}

fn json_msg(m: &ServerMessage) -> Message {
    Message::Text(
        serde_json::to_string(m)
            .expect("ServerMessage is always serialisable")
            .into(),
    )
}
