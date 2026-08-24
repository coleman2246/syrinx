//! Protocol conformance, driven over a real socket against the real router.
//!
//! Uses the mock ASR backend, so this whole file runs in CI on a machine with
//! no CUDA. That is the point of the `AsrBackend` boundary.

use futures_util::{SinkExt, StreamExt};
use syrinx_proto::{ClientMessage, Encoding, ErrorCode, Mode, ServerMessage};
use syrinx_server::app::build_router;
use syrinx_server::asr::AsrBackend;
use syrinx_server::asr::lifecycle::{FixedVramProbe, ModelHandle, VramGuard};
use syrinx_server::asr::mock::MockBackend;
use syrinx_server::config::Config;
use std::sync::Arc;
use std::time::Duration;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message as TMessage;

const TOKEN: &str = "test-token";

/// Start the server on an ephemeral port; returns its address.
async fn spawn_server(max_sessions: usize) -> String {
    let config = Arc::new(
        Config::from_toml(&format!(
            r#"
            token = "{TOKEN}"
            model_dir = "/nonexistent"
            max_sessions = {max_sessions}
        "#
        ))
        .unwrap(),
    );
    // Small chunks keep the tests fast: 4 samples per inference chunk.
    // FixedVramProbe(None) means "no GPU", so the VRAM guard is skipped -- these
    // tests exercise the protocol, not the tenancy policy.
    let model = Arc::new(ModelHandle::new(
        Arc::new(|| {
            Ok(Arc::new(MockBackend::new(&["alpha", "beta", "gamma"]).with_chunk_samples(4))
                as Arc<dyn AsrBackend>)
        }),
        VramGuard::new(1536),
        Arc::new(FixedVramProbe(None)),
        3400,
        Duration::from_secs(600),
    ));
    let app = build_router(model, config);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("ws://{addr}/v1/stream")
}

async fn connect(
    url: &str,
    token: Option<&str>,
) -> Result<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    tokio_tungstenite::tungstenite::Error,
> {
    let mut req = url.into_client_request().unwrap();
    if let Some(t) = token {
        req.headers_mut()
            .insert("authorization", format!("Bearer {t}").parse().unwrap());
    }
    let (ws, _) = tokio_tungstenite::connect_async(req).await?;
    Ok(ws)
}

fn text(m: &ClientMessage) -> TMessage {
    TMessage::Text(serde_json::to_string(m).unwrap().into())
}

/// 4 samples of s16le silence = one mock chunk.
fn one_chunk() -> TMessage {
    TMessage::Binary(vec![0u8; 8].into())
}

async fn next_msg(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Option<ServerMessage> {
    while let Some(Ok(m)) = ws.next().await {
        if let TMessage::Text(t) = m {
            return Some(serde_json::from_str(&t).unwrap());
        }
    }
    None
}

#[tokio::test]
async fn bad_token_is_rejected_before_any_session() {
    let url = spawn_server(4).await;
    assert!(
        connect(&url, Some("wrong")).await.is_err(),
        "a bad token must not reach a session"
    );
    assert!(
        connect(&url, None).await.is_err(),
        "a missing token must not reach a session"
    );
}

#[tokio::test]
async fn happy_path_emits_ready_then_ordered_commits_then_closed() {
    let url = spawn_server(4).await;
    let mut ws = connect(&url, Some(TOKEN)).await.unwrap();

    ws.send(text(&ClientMessage::SessionStart {
        mode: Mode::Live,
        sample_rate: 16000,
        encoding: Encoding::PcmS16le,
        language: None,
        vocabulary: None,
        diarize: false,
    }))
    .await
    .unwrap();

    match next_msg(&mut ws).await.unwrap() {
        ServerMessage::SessionReady { model, .. } => assert_eq!(model, "mock"),
        other => panic!("expected session.ready, got {other:?}"),
    }

    for _ in 0..3 {
        ws.send(one_chunk()).await.unwrap();
    }
    ws.send(text(&ClientMessage::SessionStop)).await.unwrap();

    let mut commits = Vec::new();
    let mut closed = false;
    while let Some(m) = next_msg(&mut ws).await {
        match m {
            ServerMessage::TranscriptCommit { seq, text } => commits.push((seq, text)),
            ServerMessage::SessionClosed { .. } => {
                closed = true;
                break;
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    assert!(closed, "server must send session.closed");
    let seqs: Vec<u64> = commits.iter().map(|(s, _)| *s).collect();
    assert_eq!(seqs, vec![1, 2, 3], "commits must be sequential from 1");
    assert_eq!(commits[0].1, "alpha ");
}

#[tokio::test]
async fn audio_before_session_start_is_a_protocol_error() {
    let url = spawn_server(4).await;
    let mut ws = connect(&url, Some(TOKEN)).await.unwrap();

    // The server has no mode, sample rate or encoding yet, so it cannot
    // interpret these bytes and must say so rather than guessing.
    ws.send(one_chunk()).await.unwrap();

    match next_msg(&mut ws).await.unwrap() {
        ServerMessage::Error { code, .. } => assert_eq!(code, ErrorCode::BadRequest),
        other => panic!("expected bad_request error, got {other:?}"),
    }
}

#[tokio::test]
async fn live_mode_never_emits_revision_messages_over_the_wire() {
    // The same invariant as tests/modes.rs, asserted end to end through the
    // transport rather than at the session boundary.
    let url = spawn_server(4).await;
    let mut ws = connect(&url, Some(TOKEN)).await.unwrap();

    ws.send(text(&ClientMessage::SessionStart {
        mode: Mode::Live,
        sample_rate: 16000,
        encoding: Encoding::PcmS16le,
        language: None,
        vocabulary: None,
        diarize: false,
    }))
    .await
    .unwrap();

    for _ in 0..3 {
        ws.send(one_chunk()).await.unwrap();
    }
    ws.send(text(&ClientMessage::SessionStop)).await.unwrap();

    while let Some(m) = next_msg(&mut ws).await {
        match m {
            ServerMessage::TranscriptProvisional { .. } | ServerMessage::TranscriptRevise { .. } => {
                panic!("live mode emitted a revision message over the wire: {m:?}")
            }
            ServerMessage::SessionClosed { .. } => break,
            _ => {}
        }
    }
}

#[tokio::test]
async fn sessions_beyond_capacity_are_refused_with_a_retryable_error() {
    let url = spawn_server(1).await;

    let mut first = connect(&url, Some(TOKEN)).await.unwrap();
    first
        .send(text(&ClientMessage::SessionStart {
            mode: Mode::Live,
            sample_rate: 16000,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
            diarize: false,
        }))
        .await
        .unwrap();
    assert!(matches!(
        next_msg(&mut first).await.unwrap(),
        ServerMessage::SessionReady { .. }
    ));

    // Second connection exceeds max_sessions = 1.
    let mut second = connect(&url, Some(TOKEN)).await.unwrap();
    match next_msg(&mut second).await.unwrap() {
        ServerMessage::Error {
            code, retryable, ..
        } => {
            assert_eq!(code, ErrorCode::Capacity);
            assert!(retryable, "capacity is transient, so it must be retryable");
        }
        other => panic!("expected capacity error, got {other:?}"),
    }
}
