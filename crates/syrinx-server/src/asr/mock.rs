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
    chunks_per_word: usize,
    tail: Option<String>,
}

impl MockBackend {
    pub fn new(words: &[&str]) -> Self {
        Self {
            script: Arc::new(words.iter().map(|s| s.to_string()).collect()),
            chunk_samples: 8960,
            chunks_per_word: 1,
            tail: None,
        }
    }

    /// Smaller chunks keep tests fast and readable.
    pub fn with_chunk_samples(mut self, n: usize) -> Self {
        self.chunk_samples = n;
        self
    }

    /// Emit a word every `n` chunks rather than every chunk, so a commit's
    /// text spans several chunks of audio.
    ///
    /// The shape a real transducer has and the default does not: it says
    /// nothing for a while and then emits a phrase covering everything it has
    /// heard since. One word per chunk collapses a commit's first and last
    /// chunk onto each other, which hides every question about which chunks a
    /// commit's words actually came from.
    pub fn with_chunks_per_word(mut self, n: usize) -> Self {
        self.chunks_per_word = n.max(1);
        self
    }

    /// Text the model is still holding when the stream ends, drained once by
    /// [`AsrStream::finish`]. The real transducer can emit on that flush, so
    /// anything the session does with a tail needs a way to provoke one.
    pub fn with_tail(mut self, text: &str) -> Self {
        self.tail = Some(text.to_string());
        self
    }
}

impl AsrBackend for MockBackend {
    fn stream(&self) -> Box<dyn AsrStream> {
        Box::new(MockStream {
            script: self.script.clone(),
            idx: 0,
            pushes: 0,
            chunks_per_word: self.chunks_per_word,
            tail: self.tail.clone(),
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
    pushes: usize,
    chunks_per_word: usize,
    tail: Option<String>,
}

impl AsrStream for MockStream {
    fn push(&mut self, _audio: &[f32]) -> Result<String> {
        self.pushes += 1;
        if !self.pushes.is_multiple_of(self.chunks_per_word) {
            return Ok(String::new());
        }
        match self.script.get(self.idx) {
            Some(w) => {
                self.idx += 1;
                Ok(format!("{w} "))
            }
            None => Ok(String::new()),
        }
    }

    fn finish(&mut self) -> Result<String> {
        // Once: a second finish drains nothing, as the real stream would.
        Ok(self.tail.take().unwrap_or_default())
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
    fn a_scripted_tail_is_drained_once() {
        let b = MockBackend::new(&["only"]).with_tail("tail ");
        let mut s = b.stream();
        assert_eq!(s.finish().unwrap(), "tail ");
        assert_eq!(s.finish().unwrap(), "", "the tail is not re-emitted");
    }

    #[test]
    fn without_a_scripted_tail_finish_drains_nothing() {
        let mut s = MockBackend::new(&["only"]).stream();
        assert_eq!(s.finish().unwrap(), "");
    }

    #[test]
    fn a_word_can_span_several_chunks() {
        // What a real transducer does, and what a commit's chunk span is
        // measured against: silence, silence, then the phrase all three
        // chunks contributed to.
        let b = MockBackend::new(&["hello", "world"]).with_chunks_per_word(3);
        let mut s = b.stream();
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "");
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "");
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "hello ");
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "");
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "");
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "world ");
    }

    #[test]
    fn one_chunk_per_word_is_the_default() {
        let b = MockBackend::new(&["hello"]);
        let mut s = b.stream();
        assert_eq!(s.push(&[0.0; 8960]).unwrap(), "hello ");
    }

    #[test]
    fn chunk_ms_derives_from_sample_count() {
        assert_eq!(MockBackend::new(&[]).chunk_ms(), 560);
    }
}
