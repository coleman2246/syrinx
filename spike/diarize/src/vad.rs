//! Silero-VAD v5 over fixed 512-sample (32 ms) frames.
//!
//! v5 carries a single `state` tensor between calls instead of v4's separate
//! `h`/`c` pair, and it only accepts 512-sample frames at 16 kHz — feeding it
//! anything else produces silently wrong probabilities rather than an error.

use anyhow::Result;
use ort::{session::Session, value::Tensor};

use crate::embed::session;

pub const FRAME: usize = 512;
const STATE: usize = 2 * 128;
/// v5 does its own STFT inside the graph and expects the caller to prepend
/// the tail of the previous frame, so the tensor it actually wants is 576
/// wide, not 512. Feeding it a bare 512 samples returns near-zero speech
/// probability for everything — wrong, and quietly so.
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
        let mut buf = Vec::with_capacity(CONTEXT + FRAME);
        buf.extend_from_slice(&self.context);
        buf.extend_from_slice(frame);
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
