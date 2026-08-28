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
// `window` and `real` carry their own summaries in their `//!` headers rather
// than here. Not a style wobble: an outer doc comment on a `mod` line moves
// the whole module's doc block into *this* module's link scope, so every
// `[`Framer`]` in the file's own header silently stops resolving. Modules
// whose headers link to their own items are documented in the file.
pub mod window;

#[cfg(feature = "diarize")]
pub mod real;

use anyhow::Result;

/// The diarization settings a deployment can change, travelling together
/// because they are set together and every one of them is read exactly once,
/// at session start.
///
/// Held as a struct rather than passed as five arguments because four of them
/// are cosines and a caller that swapped them would compile. For the same
/// reason it is handed on whole: [`cluster::OnlineClusterer::with_config`]
/// takes this type rather than the three fields it reads, so the last place a
/// transposition could still have happened is a type error instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiarizeTuning {
    /// `diarize_min_pool`: agreeing windows before a speaker is minted.
    pub min_pool: usize,
    /// `diarize_margin`: how far the best centroid must beat the second.
    pub margin: f32,
    /// `diarize_mint_ceiling`: how close to an incumbent a pool's mean may sit
    /// and still be somebody new. A separate setting from `margin` because it
    /// is a separate question on a separate scale -- see
    /// [`cluster::T_MINT_CEILING`].
    pub mint_ceiling: f32,
    /// `diarize_change_threshold`: the cosine drop between hops that marks a
    /// turn change.
    pub change_threshold: f32,
    /// How alike the hops either side of a silence must be before a correction
    /// is allowed to reach across it. See [`cluster::T_GAP_CHANGE`].
    ///
    /// The one field here with no config key behind it. It decides how far
    /// back a correction reaches and nothing else, its value was chosen by a
    /// sweep of `diarize_probe live` rather than estimated, and a deployment
    /// has no measurement of its own to beat that with; `--gap-change` is
    /// where it is varied until one does.
    pub gap_change_threshold: f32,
}

impl Default for DiarizeTuning {
    /// The calibrated and estimated constants, read from where each one's
    /// justification lives rather than repeated here.
    fn default() -> Self {
        Self {
            min_pool: cluster::MIN_POOL,
            margin: cluster::T_MARGIN,
            mint_ceiling: cluster::T_MINT_CEILING,
            change_threshold: cluster::T_CHANGE,
            gap_change_threshold: cluster::T_GAP_CHANGE,
        }
    }
}

/// A stretch of already-pushed chunks whose speaker has just become known.
///
/// Chunks are counted from zero in [`Diarizer::push`] order, which is the same
/// count `Session` keeps, because the session pushes exactly one chunk per
/// chunk and in order. Two things produce one: a speaker being minted, whose
/// first few seconds were committed before they had a number, and a full
/// window contradicting a guess that was being offered.
///
/// How far back the range may reach is the diarizer's to bound, and it bounds
/// it by what it can vouch for rather than by what it happened to detect --
/// `real::RealDiarizer` records exactly what that means and what it does not
/// cover. The session's own rule, that a settled label is never overwritten,
/// is the second guard and not the first: it says nothing about text nobody
/// was named for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Relabel {
    /// First chunk covered, inclusive.
    pub from_chunk: u64,
    /// Last chunk covered, inclusive.
    pub to_chunk: u64,
    /// Who it was. Always a speaker that exists.
    pub speaker: u32,
}

/// What one chunk of audio told the diarizer.
///
/// `speaker: None` is honest uncertainty (silence, cross-talk, a voice not yet
/// minted) and is normal; `Err` from [`Diarizer::push`] means the diarizer
/// itself failed. The distinction is load-bearing: the session counts
/// consecutive errors to decide when to give up on labelling, and must not
/// count uncertainty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Attribution {
    /// Who is speaking in this chunk.
    pub speaker: Option<u32>,
    /// Whether `speaker` is a guess rather than an answer the diarizer will
    /// stand behind.
    ///
    /// Two things produce one, and they are the same promise: a 0.75 s hop
    /// naming a turn before any window has confirmed it, and a full window
    /// that no centroid stood out clearly enough for. Either may be
    /// contradicted by a later window.
    ///
    /// The session keeps it per chunk so that a later correction knows what it
    /// may overwrite: a gap and a guess are both correctable, a label a full
    /// window settled is not.
    pub provisional: bool,
    /// Whether the voice changed inside this chunk.
    ///
    /// The session stops the commit vote at a boundary, because a vote that
    /// reaches across a turn change is a vote the outgoing speaker wins by
    /// construction -- they were there first, and ties go to the earliest.
    pub boundary: bool,
    /// Speakers now known for audio the session has already committed.
    pub relabels: Vec<Relabel>,
}

impl Attribution {
    /// The common case: a label and nothing else to say.
    pub fn speaker(speaker: Option<u32>) -> Self {
        Self {
            speaker,
            ..Default::default()
        }
    }
}

/// One session's speaker-attribution state.
///
/// `push` is called once per ASR chunk, in order, with exactly the samples the
/// ASR saw. That ordering is what lets [`Relabel`] name chunks by number
/// without the two sides having to exchange one.
pub trait Diarizer: Send {
    fn push(&mut self, audio: &[f32]) -> Result<Attribution>;
}

/// Spawns an independent [`Diarizer`] per session, sharing loaded models.
pub trait DiarizerFactory: Send + Sync {
    fn diarizer(&self) -> Box<dyn Diarizer>;
}

/// Scripted diarizer for protocol and session tests. Deterministic on
/// purpose, like [`crate::asr::mock::MockStream`]: tests assert exact
/// message sequences.
pub struct MockDiarizer {
    script: std::collections::VecDeque<Result<Attribution>>,
}

impl MockDiarizer {
    /// A script of labels and failures, which is what most session tests need
    /// and all of them needed before boundaries and relabels existed.
    pub fn new(script: Vec<Result<Option<u32>>>) -> Self {
        Self::scripted(
            script
                .into_iter()
                .map(|r| r.map(Attribution::speaker))
                .collect(),
        )
    }

    /// The common case: one label per chunk, no errors.
    pub fn labels(labels: &[Option<u32>]) -> Self {
        Self::new(labels.iter().map(|l| Ok(*l)).collect())
    }

    /// A script of whole answers, for the tests that are about boundaries or
    /// corrections rather than about labels.
    pub fn scripted(script: Vec<Result<Attribution>>) -> Self {
        Self {
            script: script.into(),
        }
    }
}

impl Diarizer for MockDiarizer {
    fn push(&mut self, _audio: &[f32]) -> Result<Attribution> {
        // Past the script's end: unknown, not an error. A session outliving
        // its script is normal in tests that then call finish().
        self.script
            .pop_front()
            .unwrap_or_else(|| Ok(Attribution::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_replays_its_script_then_reports_unknown() {
        let mut d = MockDiarizer::labels(&[Some(1), None, Some(2)]);
        assert_eq!(d.push(&[]).unwrap().speaker, Some(1));
        assert_eq!(d.push(&[]).unwrap().speaker, None);
        assert_eq!(d.push(&[]).unwrap().speaker, Some(2));
        assert_eq!(d.push(&[]).unwrap().speaker, None);
    }

    #[test]
    fn mock_can_script_a_failure() {
        let mut d = MockDiarizer::new(vec![Err(anyhow::anyhow!("boom"))]);
        assert!(d.push(&[]).is_err());
    }

    #[test]
    fn a_label_script_says_nothing_about_boundaries_or_corrections() {
        // The shorthand every session test predating them uses, so it has to
        // mean "and nothing else happened" rather than leaving the extra
        // fields to whatever `Default` happens to be.
        let mut d = MockDiarizer::labels(&[Some(1)]);
        assert_eq!(
            d.push(&[]).unwrap(),
            Attribution {
                speaker: Some(1),
                provisional: false,
                boundary: false,
                relabels: Vec::new(),
            }
        );
    }

    #[test]
    fn the_tuning_defaults_are_the_constants_the_code_names() {
        let t = DiarizeTuning::default();
        assert_eq!(t.min_pool, cluster::MIN_POOL);
        assert_eq!(t.margin, cluster::T_MARGIN);
        assert_eq!(t.mint_ceiling, cluster::T_MINT_CEILING);
        assert_eq!(t.change_threshold, cluster::T_CHANGE);
        assert_eq!(t.gap_change_threshold, cluster::T_GAP_CHANGE);
    }

    #[test]
    fn crossing_a_silence_is_judged_more_strictly_than_crossing_a_hop() {
        // The two thresholds are not interchangeable and the ordering between
        // them is the whole argument: a 0.75 s embedding either side of a
        // pause is the noisiest comparison the pipeline makes, and the error
        // it must not make -- reaching a correction across a real speaker
        // change -- is the expensive one.
        let t = DiarizeTuning::default();
        assert!(
            t.gap_change_threshold > t.change_threshold,
            "a silence must be harder to call one voice than a hop boundary is"
        );
    }
}
