//! Router construction, shared by the binary and the integration tests.
//!
//! Tests build the same router the binary does, with a mock backend, so what
//! CI exercises is the real transport rather than a simplified stand-in.

use crate::asr::AsrBackend;
use crate::config::Config;
use crate::ws::{AppState, stream_handler};
use axum::Router;
use axum::routing::get;
use std::sync::Arc;

pub fn build_router(backend: Arc<dyn AsrBackend>, config: Arc<Config>) -> Router {
    let state = AppState::new(backend, config);
    Router::new()
        .route("/v1/stream", get(stream_handler))
        .route("/health", get(|| async { "ok" }))
        .with_state(state)
}
