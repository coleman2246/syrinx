//! Silero-VAD v5 over fixed 512-sample (32 ms) frames.
//!
//! v5 carries a single `state` tensor between calls instead of v4's separate
//! `h`/`c` pair, and it only accepts 512-sample frames at 16 kHz -- feeding it
//! anything else produces silently wrong probabilities rather than an error.
//! Ported from `spike/diarize/src/vad.rs`, validated there against AMI
//! meetings and speaker-verification pairs.

use anyhow::Result;
use ort::{session::Session, value::Tensor};

use super::session;

pub const FRAME: usize = 512;
const STATE: usize = 2 * 128;
/// v5 does its own STFT inside the graph and expects the caller to prepend
/// the tail of the previous frame, so the tensor it actually wants is 576
/// wide, not 512. Feeding it a bare 512 samples returns near-zero speech
/// probability for everything -- wrong, and quietly so.
const CONTEXT: usize = 64;

/// Silero's own `VADIterator` defaults: enter speech at 0.5, leave it at
/// 0.35, so a wobble around the threshold does not chop a turn into pieces.
const ENTER: f32 = 0.5;
const LEAVE: f32 = 0.35;
/// Keep the tail of a turn: trailing low-energy speech carries speaker
/// identity, and cutting it costs embedding quality.
const HANGOVER_FRAMES: usize = 4; // ~128 ms

pub struct Vad {
    session: Session,
    state: Vec<f32>,
    context: Vec<f32>,
}

impl Vad {
    pub fn new(path: &str) -> Result<Self> {
        // One thread: silero is far too small for intra-op parallelism to pay
        // for its own synchronisation, and this runs 500 times a second.
        Ok(Self {
            session: session(path, 1)?,
            state: vec![0.0; STATE],
            context: vec![0.0; CONTEXT],
        })
    }

    pub fn prob(&mut self, frame: &[f32]) -> Result<f32> {
        let buf = with_context(&self.context, frame);
        self.context.copy_from_slice(&buf[buf.len() - CONTEXT..]);

        let input = Tensor::from_array(([1usize, CONTEXT + FRAME], buf))?;
        let state = Tensor::from_array(([2usize, 1, 128], self.state.clone()))?;
        let sr = Tensor::from_array(((), vec![16_000i64]))?;

        let out = self.session.run(ort::inputs![
            "input" => input,
            "state" => state,
            "sr" => sr,
        ])?;

        let (_, next) = out["stateN"].try_extract_tensor::<f32>()?;
        self.state.copy_from_slice(next);
        let (_, p) = out["output"].try_extract_tensor::<f32>()?;
        Ok(p[0])
    }

    /// One speech/non-speech decision per 512-sample frame of `samples`.
    pub fn run(&mut self, samples: &[f32]) -> Result<Vec<bool>> {
        let frames = samples.len() / FRAME;
        let mut voiced = Vec::with_capacity(frames);
        let mut speaking = false;
        let mut hangover = 0usize;

        for f in 0..frames {
            let p = self.prob(&samples[f * FRAME..(f + 1) * FRAME])?;
            if speaking {
                if p < LEAVE {
                    if hangover == 0 {
                        speaking = false;
                    } else {
                        hangover -= 1;
                    }
                } else {
                    hangover = HANGOVER_FRAMES;
                }
            } else if p >= ENTER {
                speaking = true;
                hangover = HANGOVER_FRAMES;
            }
            voiced.push(speaking);
        }
        Ok(voiced)
    }
}

/// Prepend the previous call's trailing `CONTEXT` samples to `frame`,
/// producing the 576-wide buffer silero actually wants. Pulled out of
/// [`Vad::prob`] as a free function -- with no `Session` field to thread
/// through -- so this module's central trap (see the `CONTEXT` doc comment
/// above) is unit-testable without an ONNX Runtime session.
fn with_context(context: &[f32], frame: &[f32]) -> Vec<f32> {
    debug_assert_eq!(
        context.len(),
        CONTEXT,
        "silero's context is always 64 samples"
    );
    let mut buf = Vec::with_capacity(CONTEXT + frame.len());
    buf.extend_from_slice(context);
    buf.extend_from_slice(frame);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_is_prepended_and_the_result_is_576_wide() {
        let context = vec![9.0; CONTEXT];
        let frame: Vec<f32> = (0..FRAME).map(|i| i as f32).collect();

        let buf = with_context(&context, &frame);

        assert_eq!(buf.len(), 576, "silero wants 576 samples, not a bare 512");
        assert_eq!(&buf[..CONTEXT], context.as_slice());
        assert_eq!(&buf[CONTEXT..], frame.as_slice());
    }

    #[test]
    fn the_next_calls_context_is_this_frames_own_tail() {
        // The 64 samples carried into the *next* call must come from the
        // frame just processed, not linger from whatever came before it --
        // this is the state Vad::prob threads via `self.context.copy_from_slice`.
        let stale_context = vec![-1.0; CONTEXT];
        let frame: Vec<f32> = (0..FRAME).map(|i| i as f32).collect();

        let buf = with_context(&stale_context, &frame);
        let next_context = &buf[buf.len() - CONTEXT..];

        assert_eq!(next_context, &frame[FRAME - CONTEXT..]);
    }
}
