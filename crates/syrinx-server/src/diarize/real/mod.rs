//! ONNX-backed VAD and speaker-embedding wrappers.
//!
//! Ported from `spike/diarize/src/{vad,fbank,embed}.rs`, which validated this
//! pipeline end-to-end against speaker-verification pairs and AMI meetings.
//! Gated behind the `diarize` feature so the default build stays free of
//! `ort`, mirroring `asr::parakeet`'s `cuda` gate. [`super::Diarizer`] is
//! implemented on top of these two structs by [`RealDiarizer`], which owns
//! the pipeline; `vad` and `embed` own only the model sessions, and the
//! `session` helper below the one decision they share.
//!
//! The fbank front end these wrap lives one level up, at
//! [`super::fbank`]: it is pure arithmetic with no `ort` in it, so it is
//! not gated here and gets CI coverage in the default build too.

mod diarizer;
mod embed;
mod vad;

/// Re-exported for convenience: callers of this module's `Embedder::new`
/// need `Norm` without reaching past it into `diarize::fbank` themselves.
pub use super::fbank::Norm;
pub use diarizer::{RealDiarizer, RealDiarizerFactory, norm_for};
pub use embed::Embedder;
pub use vad::Vad;

use anyhow::{Context, Result};
use ort::execution_providers::CPU;
use ort::session::Session;

/// Build a session with the CPU execution provider registered explicitly.
///
/// Same lesson as `asr::parakeet::ParakeetBackend::load_cuda`: never let
/// ONNX Runtime's default choose. In this ort build CPU already *is* the
/// default when no provider is registered, so this call changes no observed
/// behaviour today -- but writing it out keeps that a decision on record
/// rather than an accident that stops being true the moment `cuda` and
/// `diarize` are built together and cargo unifies this crate's `ort` with
/// parakeet-rs's own (feature-unified to exactly one `ort` version; the two
/// features otherwise know nothing about each other).
fn session(path: &str, threads: usize) -> Result<Session> {
    // Each `with_*` call returns `Result<SessionBuilder, Error<SessionBuilder>>`
    // -- the generic error carries the builder back out for potential
    // recovery -- so `.map_err(<ort::Error>::from)` drops that and gets a
    // plain `ort::Error`, which is what lets `?` hand off to `anyhow`.
    // `commit_from_file` takes `&mut self` (unlike the fluent `with_*`
    // methods, which consume and return `Self`), hence the `mut` binding.
    let mut builder = Session::builder()?
        .with_execution_providers([CPU::default().build()])
        .map_err(<ort::Error>::from)?
        .with_intra_threads(threads)
        .map_err(<ort::Error>::from)?;
    builder
        .commit_from_file(path)
        .with_context(|| format!("loading ONNX model {path}"))
}
