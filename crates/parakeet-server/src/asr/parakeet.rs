//! The real ASR backend: NVIDIA Nemotron cache-aware streaming via parakeet-rs.
//!
//! Feature-gated behind `cuda` so the default build, and therefore CI, needs no
//! GPU stack at all.

use super::{AsrBackend, AsrStream};
use anyhow::{Context, Result};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron, NemotronHandle};
use std::path::Path;

/// Samples per inference chunk: 560 ms at 16 kHz.
const CHUNK_SAMPLES: usize = 8960;

/// Zero-chunks pushed at end of stream to drain the decoder's remaining tokens.
const FLUSH_CHUNKS: usize = 3;

pub struct ParakeetBackend {
    handle: NemotronHandle,
    model_name: String,
}

impl ParakeetBackend {
    /// Load the model onto the GPU.
    ///
    /// The execution provider is passed **explicitly**. `from_pretrained(path,
    /// None)` resolves to `ExecutionProvider::default()`, which is `Cpu` -- so
    /// omitting it produces a server that works but runs an order of magnitude
    /// too slowly, with no error to indicate why. parakeet-rs also falls back to
    /// CPU if the CUDA provider fails to initialise, so a working-but-slow
    /// server is a real failure mode. Verify with `nvidia-smi` after starting;
    /// no unit test can catch it.
    pub fn load_cuda(dir: &Path) -> Result<Self> {
        let exec = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
        let handle = NemotronHandle::from_pretrained(dir, Some(exec))
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading Nemotron model from {}", dir.display()))?;
        Ok(Self {
            handle,
            model_name: "nemotron-en-0.6b".to_string(),
        })
    }
}

impl AsrBackend for ParakeetBackend {
    /// Spawn an independent stream over the shared model.
    ///
    /// One ~2.5 GB set of weights serves every session; only decoder state is
    /// per-stream. Loading a copy per session would exhaust an 8 GB card at
    /// three clients.
    fn stream(&self) -> Box<dyn AsrStream> {
        Box::new(ParakeetStream {
            inner: Nemotron::from_shared(&self.handle),
        })
    }

    fn chunk_samples(&self) -> usize {
        CHUNK_SAMPLES
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

pub struct ParakeetStream {
    inner: Nemotron,
}

impl AsrStream for ParakeetStream {
    fn push(&mut self, audio: &[f32]) -> Result<String> {
        self.inner
            .transcribe_chunk(audio)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn finish(&mut self) -> Result<String> {
        // The transducer holds tokens until enough context arrives; feeding
        // silence drains them so the tail of an utterance is not lost.
        let mut out = String::new();
        for _ in 0..FLUSH_CHUNKS {
            let text = self
                .inner
                .transcribe_chunk(&vec![0.0; CHUNK_SAMPLES])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.push_str(&text);
        }
        Ok(out)
    }
}
