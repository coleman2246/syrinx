//! Silero-VAD over fixed 512-sample (32 ms) frames. The shipped weights are
//! v6.2; the interface described below arrived in v5 and is unchanged since.
//!
//! v5 carries a single `state` tensor between calls instead of v4's separate
//! `h`/`c` pair, and it only accepts 512-sample frames at 16 kHz -- feeding it
//! anything else produces silently wrong probabilities rather than an error.
//! Ported from the go/no-go spike -- deleted, `git log -- 'spike/diarize'` --
//! which validated it against AMI meetings and speaker-verification pairs.

use anyhow::{Result, anyhow, ensure};
use ort::{session::Session, value::Tensor};

use super::session;
// The frame size is shared with the windowing that counts in it, and lives
// there so it needs no `ort` to be tested against.
use crate::diarize::window::FRAME;

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
    /// Samples carried over from the previous [`Vad::run`] call because they
    /// did not fill a whole 512-sample frame -- the same idea as `context`,
    /// just at the caller's boundary instead of silero's.
    /// [`super::RealDiarizer`] calls `run` once per ASR chunk (8960 samples
    /// = 17.5 frames); without this, the trailing 256 samples of every
    /// single chunk would be silently dropped instead of joining the next.
    remainder: Vec<f32>,
    gate: Gate,
}

impl Vad {
    pub fn new(path: &str) -> Result<Self> {
        // One thread: silero is far too small for intra-op parallelism to
        // pay for its own synchronisation, and one 512-sample frame is
        // 32 ms, so this runs about 31 times a second per stream.
        Ok(Self {
            session: session(path, 1)?,
            state: vec![0.0; STATE],
            context: vec![0.0; CONTEXT],
            remainder: Vec::new(),
            gate: Gate::default(),
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

        // `SessionOutputs`' `Index<&str>` panics on a missing name. The model
        // file is user config (`diarize_model_dir`), so a wrong or
        // differently-exported one must fail this one `push` with an `Err`,
        // not unwind the session task -- session.rs's strike-out design
        // exists precisely so a diarizer fault degrades the speaker labels,
        // never the transcript, and that only holds if `Diarizer::push` can
        // actually return the error instead of panicking past it.
        let (_, next) = out
            .get("stateN")
            .ok_or_else(|| anyhow!("silero model has no \"stateN\" output"))?
            .try_extract_tensor::<f32>()?;
        self.state.copy_from_slice(next);

        let (_, p) = out
            .get("output")
            .ok_or_else(|| anyhow!("silero model has no \"output\" output"))?
            .try_extract_tensor::<f32>()?;
        ensure!(!p.is_empty(), "silero's \"output\" tensor was empty");
        Ok(p[0])
    }

    /// One speech/non-speech decision per 512-sample frame accumulated from
    /// `samples` and whatever was left over from the previous call.
    ///
    /// Hysteresis state ([`Gate`]) and the frame remainder both live on
    /// `self`, so calling this repeatedly with successive chunks of one
    /// stream is equivalent to calling it once with the whole stream
    /// concatenated -- in particular, a turn that starts in one chunk and
    /// continues into the next is not treated as two turns. Following from
    /// that: the returned `Vec<bool>` indexes the concatenated stream, not
    /// `samples` -- after a call that left a remainder, `voiced[0]` on the
    /// *next* call describes a frame that began before that call's own
    /// `samples` did.
    ///
    /// If a frame's inference fails partway through a call, whatever frames
    /// had not yet been decoded are dropped rather than retried on the next
    /// call -- defensible, since `state` and `context` are already threaded
    /// per frame and a mid-call failure leaves them synchronised only up to
    /// the point of failure anyway, so there is no clean buffer left to
    /// resume from. The caller sees the `Err` and moves on with fresh audio.
    pub fn run(&mut self, samples: &[f32]) -> Result<Vec<bool>> {
        let buf = take_frames(&mut self.remainder, samples);

        let frames = buf.len() / FRAME;
        let mut voiced = Vec::with_capacity(frames);
        for f in 0..frames {
            let p = self.prob(&buf[f * FRAME..(f + 1) * FRAME])?;
            voiced.push(self.gate.step(p));
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

/// Combine `remainder` with `samples`, split off as many complete
/// `FRAME`-sized frames as the combined length allows, and leave whatever
/// is left over in `remainder` for the next call. Pulled out of
/// [`Vad::run`] as a free function -- the third application of the
/// `with_context` pattern -- so the framing arithmetic that dropped a
/// chunk's trailing partial frame before this module hoisted `remainder`
/// onto `Vad` is unit-testable without a session.
fn take_frames(remainder: &mut Vec<f32>, samples: &[f32]) -> Vec<f32> {
    let mut buf = std::mem::take(remainder);
    buf.extend_from_slice(samples);
    let frames = buf.len() / FRAME;
    *remainder = buf.split_off(frames * FRAME);
    buf
}

/// Speech/non-speech hysteresis: enter at `ENTER`, leave only after
/// `HANGOVER_FRAMES` consecutive frames below `LEAVE`. Pulled out of
/// [`Vad::run`] as a plain state machine -- no `Session`, no buffer -- for
/// the same reason `with_context` was: this is the actual onset/hangover
/// policy, and it deserves a unit test that isn't also exercising an ONNX
/// session.
#[derive(Default)]
struct Gate {
    speaking: bool,
    hangover: usize,
}

impl Gate {
    /// Feed one frame's speech probability; returns whether that frame
    /// counts as speech after applying the hysteresis.
    fn step(&mut self, p: f32) -> bool {
        if self.speaking {
            if p < LEAVE {
                if self.hangover == 0 {
                    self.speaking = false;
                } else {
                    self.hangover -= 1;
                }
            } else {
                self.hangover = HANGOVER_FRAMES;
            }
        } else if p >= ENTER {
            self.speaking = true;
            self.hangover = HANGOVER_FRAMES;
        }
        self.speaking
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------- with_context

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

    // --------------------------------------------------------- take_frames

    /// Pins the bug this module used to have: an 8960-sample ASR chunk
    /// (`asr::parakeet::CHUNK_SAMPLES`) is 17.5 frames, and the trailing
    /// 256-sample half-frame must be carried into the next call rather than
    /// silently dropped -- and the next call's first frame must actually
    /// open with those carried samples, not just have the right length.
    #[test]
    fn a_chunk_and_a_half_carries_its_tail_into_the_next_calls_first_frame() {
        let mut remainder = Vec::new();
        let chunk: Vec<f32> = (0..8960).map(|i| i as f32).collect();

        let framed = take_frames(&mut remainder, &chunk);
        assert_eq!(framed.len(), 17 * FRAME, "8960 samples is 17.5 frames");
        assert_eq!(remainder.len(), 256, "the trailing half-frame is carried");
        assert_eq!(remainder, chunk[17 * FRAME..]);
        let carried = remainder.clone();

        // Just enough new samples (512 - 256) to complete exactly one more
        // frame out of the carried tail.
        let next_chunk = vec![-1.0; FRAME - 256];
        let next_framed = take_frames(&mut remainder, &next_chunk);

        assert_eq!(next_framed.len(), FRAME);
        assert_eq!(
            &next_framed[..256],
            carried.as_slice(),
            "the next call's first frame must open with the carried samples"
        );
        assert_eq!(&next_framed[256..], next_chunk.as_slice());
        assert!(remainder.is_empty(), "the frame completed exactly");
    }

    // -------------------------------------------------------------- Gate

    #[test]
    fn a_probability_below_enter_does_not_onset_from_silence() {
        let mut gate = Gate::default();
        assert!(!gate.step(0.4));
    }

    #[test]
    fn once_speaking_a_probability_between_leave_and_enter_sustains() {
        let mut gate = Gate::default();
        assert!(gate.step(0.9), "0.9 is above ENTER, so this onsets");
        assert!(
            gate.step(0.4),
            "0.4 is above LEAVE (0.35) so an already-open turn stays open, \
             even though 0.4 alone would not have onset it from silence"
        );
    }

    #[test]
    fn a_four_frame_dip_bridges_and_a_fifth_ends_the_turn() {
        let mut gate = Gate::default();
        assert!(gate.step(0.9), "onset");

        for n in 1..=HANGOVER_FRAMES {
            assert!(
                gate.step(0.0),
                "dip frame {n}/{HANGOVER_FRAMES} is within the hangover \
                 budget and must still read as speaking"
            );
        }
        assert!(
            !gate.step(0.0),
            "the hangover budget is exhausted; the turn ends on this frame"
        );
    }
}
