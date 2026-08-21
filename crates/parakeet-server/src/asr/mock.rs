//! Deterministic backend used to test the protocol layer without CUDA.

use super::{AsrBackend, AsrStream};
use anyhow::Result;
use std::sync::Arc;

/// Emits one scripted word per chunk, then nothing.
///
/// Deterministic on purpose: protocol tests assert exact message sequences, so
/// the backend must not introduce variability.
pub struct MockBackend {
    script: Arc<Vec<String>>,
    chunk_samples: usize,
}

impl MockBackend {
    pub fn new(words: &[&str]) -> Self {
        Self {
            script: Arc::new(words.iter().map(|s| s.to_string()).collect()),
            chunk_samples: 8960,
        }
    }

    /// Smaller chunks keep tests fast and readable.
    pub fn with_chunk_samples(mut self, n: usize) -> Self {
        self.chunk_samples = n;
        self
    }
}

impl AsrBackend for MockBackend {
    fn stream(&self) -> Box<dyn AsrStream> {
        Box::new(MockStream {
            script: self.script.clone(),
            idx: 0,
        })
    }
    fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }
    fn model_name(&self) -> &str {
        "mock"
    }
}

pub struct MockStream {
    script: Arc<Vec<String>>,
    idx: usize,
}

impl AsrStream for MockStream {
    fn push(&mut self, _audio: &[f32]) -> Result<String> {
        match self.script.get(self.idx) {
            Some(w) => {
                self.idx += 1;
                Ok(format!("{w} "))
            }
            None => Ok(String::new()),
        }
    }

    fn finish(&mut self) -> Result<String> {
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_emits_one_word_per_chunk() {
        let b = MockBackend::new(&["hello", "world"]);
        let mut s = b.stream();
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "hello ");
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "world ");
    }

    #[test]
    fn mock_returns_empty_when_script_exhausted() {
        let b = MockBackend::new(&["only"]);
        let mut s = b.stream();
        let _ = s.push(&[0.0; 8960]).unwrap();
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "");
    }

    #[test]
    fn streams_from_one_backend_are_independent() {
        // Mirrors the real backend: one shared model, independent decoder state
        // per session. If streams shared state, session B would resume where
        // session A left off.
        let b = MockBackend::new(&["alpha", "beta"]);
        let (mut a, mut c) = (b.stream(), b.stream());
        assert_eq!(a.push(&[0.0; 8960]).unwrap(), "alpha ");
        assert_eq!(c.push(&[0.0; 8960]).unwrap(), "alpha ");
    }

    #[test]
    fn chunk_ms_derives_from_sample_count() {
        assert_eq!(MockBackend::new(&[]).chunk_ms(), 560);
    }
}
