//! Speech recognition behind a trait.
//!
//! This boundary is what keeps the protocol layer testable without a GPU. The
//! session state machine, mode semantics, backpressure and error paths all sit
//! above `AsrBackend`, so they can be exercised in CI on any machine using
//! [`mock::MockBackend`].

pub mod lifecycle;
pub mod mock;

/// The real GPU backend. Optional so the default build needs no CUDA.
#[cfg(feature = "cuda")]
pub mod parakeet;

use anyhow::Result;

/// One independent decoding stream, holding per-session decoder state.
pub trait AsrStream: Send {
    /// Feed one chunk of 16 kHz mono f32 audio, returning any newly emitted
    /// text. Returns an empty string when the model emitted nothing this chunk.
    ///
    /// Append-only by contract: implementations never retract previously
    /// returned text. A transducer emits a token and it is final, which is why
    /// revision cannot originate at this layer.
    fn push(&mut self, audio: &[f32]) -> Result<String>;

    /// Flush any trailing buffered audio at end of stream.
    fn finish(&mut self) -> Result<String>;
}

/// A loaded model able to spawn independent streams sharing its weights.
///
/// Sharing matters: one Nemotron model is ~2.5 GB, so N sessions must share one
/// set of weights rather than loading N copies onto a contended GPU.
pub trait AsrBackend: Send + Sync {
    fn stream(&self) -> Box<dyn AsrStream>;

    /// Samples per inference chunk. 8960 = 560 ms at 16 kHz for Nemotron.
    fn chunk_samples(&self) -> usize;

    fn model_name(&self) -> &str;

    /// Chunk duration in milliseconds, reported to clients in `session.ready`.
    fn chunk_ms(&self) -> u32 {
        (self.chunk_samples() as u64 * 1000 / syrinx_proto::SAMPLE_RATE as u64) as u32
    }
}
