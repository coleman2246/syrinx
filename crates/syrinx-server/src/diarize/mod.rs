//! Speaker attribution behind a trait.
//!
//! Mirrors the `AsrBackend` boundary and exists for the same reason: the
//! session's labelling semantics -- lag, majority, strike-out -- are testable
//! in CI with [`MockDiarizer`], with no models anywhere near the tests.

pub mod cluster;
/// The Kaldi-compatible fbank front end. Pure arithmetic, unconditional like
/// `cluster` -- `real::embed::Embedder` is its only in-crate consumer, but it
/// needs no `ort` itself, so there is no reason to hide it behind the
/// `diarize` feature and lose its CI coverage.
pub mod fbank;

/// The real ONNX-backed VAD and embedding wrappers. Optional so the default
/// build needs no `ort`, mirroring `asr::parakeet`'s `cuda` gate.
#[cfg(feature = "diarize")]
pub mod real;

use anyhow::Result;

/// One session's speaker-attribution state.
///
/// `push` is called once per ASR chunk, in order, with exactly the samples the
/// ASR saw. Ok(None) is honest uncertainty (silence, cross-talk) and is
/// normal; Err means the diarizer itself failed. The distinction is
/// load-bearing: the session counts consecutive errors to decide when to give
/// up on labelling, and must not count uncertainty.
pub trait Diarizer: Send {
    fn push(&mut self, audio: &[f32]) -> Result<Option<u32>>;
}

/// Spawns an independent [`Diarizer`] per session, sharing loaded models.
pub trait DiarizerFactory: Send + Sync {
    fn diarizer(&self) -> Box<dyn Diarizer>;
}

/// Scripted diarizer for protocol and session tests. Deterministic on
/// purpose, like [`crate::asr::mock::MockStream`]: tests assert exact
/// message sequences.
pub struct MockDiarizer {
    script: std::collections::VecDeque<Result<Option<u32>>>,
}

impl MockDiarizer {
    pub fn new(script: Vec<Result<Option<u32>>>) -> Self {
        Self {
            script: script.into(),
        }
    }

    /// The common case: one label per chunk, no errors.
    pub fn labels(labels: &[Option<u32>]) -> Self {
        Self::new(labels.iter().map(|l| Ok(*l)).collect())
    }
}

impl Diarizer for MockDiarizer {
    fn push(&mut self, _audio: &[f32]) -> Result<Option<u32>> {
        // Past the script's end: unknown, not an error. A session outliving
        // its script is normal in tests that then call finish().
        self.script.pop_front().unwrap_or(Ok(None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_replays_its_script_then_reports_unknown() {
        let mut d = MockDiarizer::labels(&[Some(1), None, Some(2)]);
        assert_eq!(d.push(&[]).unwrap(), Some(1));
        assert_eq!(d.push(&[]).unwrap(), None);
        assert_eq!(d.push(&[]).unwrap(), Some(2));
        assert_eq!(d.push(&[]).unwrap(), None);
    }

    #[test]
    fn mock_can_script_a_failure() {
        let mut d = MockDiarizer::new(vec![Err(anyhow::anyhow!("boom"))]);
        assert!(d.push(&[]).is_err());
    }
}
