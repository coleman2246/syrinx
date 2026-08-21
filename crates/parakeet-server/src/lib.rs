//! parakeet-stt server.
//!
//! Layering, from the bottom up:
//!
//! - [`asr`] wraps speech recognition behind a trait, with a deterministic mock
//!   so everything above it is testable without a GPU.
//! - [`session`] owns protocol semantics and knows nothing about transport.
//! - [`ws`] owns transport and knows nothing about recognition.
//!
//! That split is deliberate: it is what allows the session lifecycle, mode
//! invariants, auth and backpressure to run in CI on a machine with no CUDA.

pub mod asr;
pub mod auth;
pub mod config;
pub mod session;
