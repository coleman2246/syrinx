//! Voiced audio into embedding windows: the diarizer's bookkeeping half.
//!
//! Ported from the go/no-go spike's `windows()`, the reference the AMI numbers
//! in the design doc were measured against. The spike itself is gone -- `git
//! log -- 'spike/diarize'` for it -- and `examples/diarize_probe` is what
//! reproduces those numbers now. Pure arithmetic -- no models, no `ort`, no
//! feature gate -- for the same reason [`super::fbank`] has none: this is
//! where the alignment bugs live, so the default `cargo test` should be
//! catching them, not only the `diarize` build.
//!
//! Two pieces, because they fail differently. [`Framer`] answers "which
//! samples do the VAD's flags describe?", which is a question about the
//! *stream*; [`WindowAssembler`] answers "is there a window's worth of one
//! person talking yet?", which is about the *speech*.

use std::collections::VecDeque;

/// Samples per VAD frame: 32 ms at 16 kHz. Silero v5 accepts nothing else --
/// `real::vad` records why -- and this module counts in the same frames, so
/// the two share one definition rather than agreeing by coincidence.
///
/// Named in prose rather than linked, here and below: `real` is behind the
/// `diarize` feature and this module is not, so a link would dangle in the
/// default build, which is the one CI runs.
pub const FRAME: usize = 512;

/// Voiced frames per embedding window and per hop: the spike's calibrated 1.5 s
/// and 0.75 s (design doc, "Chosen model and constants"). Whole frames only,
/// so 1.5 s rounds to 47 frames -- 24064 samples rather than exactly 24000 --
/// and 0.75 s to 23; the spike rounded the same way, and these are the frame
/// counts every published number was measured with.
const WINDOW_FRAMES: usize = 47;
const HOP_FRAMES: usize = 23;
/// Samples in one completed window. Public because the startup self-check
/// probes the embedder with exactly this length: the shape it proves should be
/// the shape a session runs.
pub const WINDOW_SAMPLES: usize = WINDOW_FRAMES * FRAME;
const HOP_SAMPLES: usize = HOP_FRAMES * FRAME;

/// Voiced frames further apart than this start the window over instead of
/// splicing across the silence between them. Silence of half a second ends a
/// turn more often than not, and a window straddling two speakers embeds
/// neither of them. 15 frames is 0.48 s, the spike's truncating conversion.
///
/// This protects the *embedding*, and is not a turn-change detector -- a
/// tempting second use it is measurably bad at. Over 13 minutes of AMI
/// (ES2002a and IS1000a), the accumulator broke here 151 times; of the 51
/// breaks with a decided label on both sides, 48 had the same speaker either
/// side of the gap. Real speaker changes in a meeting mostly arrive *without*
/// half a second of silence in front of them, so the diarizer keeps carrying
/// its label across a break rather than blanking it: blanking doubled the
/// unlabelled share of voiced chunks (30% to 59%) to catch three changes.
///
/// There is now a real turn-change detector, and it is somewhere else: the
/// diarizer compares the voice in one [`Cut::Hop`] against the voice in the
/// one before it against `cluster::T_CHANGE`, and calls
/// [`WindowAssembler::cut_at_boundary`] when they differ. Two rules that look
/// alike and are not -- this one asks how long the silence was, that one asks
/// who is talking -- so the measurement above still stands and this constant
/// is unchanged.
const MAX_GAP_FRAMES: usize = 15;

/// A piece of voiced audio the assembler has finished with.
///
/// Two sizes, because they answer different questions and are trusted
/// differently.
#[derive(Debug, PartialEq)]
pub enum Cut {
    /// 0.75 s of new voiced audio -- the hop, on its own, and never
    /// overlapping the hop before it.
    ///
    /// It exists to be embedded a second time, which costs one more 45 ms
    /// embedding per 0.75 s of speech and buys two things a 1.5 s window
    /// cannot deliver in time: whether the voice has changed since the last
    /// hop, and a provisional label about a second into a turn instead of
    /// after a full window and a vote. A 0.75 s embedding is noisier than a
    /// 1.5 s one, so it may do neither of the things that need to be right --
    /// it never updates a centroid and never enters the mint pool.
    Hop(Vec<f32>),
    /// 1.5 s of voiced audio: the window every number in the design doc was
    /// measured on, and still the only thing allowed to move a centroid or to
    /// mint a speaker.
    Window(Vec<f32>),
}

/// One chunk's worth of whole frames, and where they sit in the stream.
pub struct Framed {
    /// Index of the first frame in `samples`, counted from the start of the
    /// session over the concatenated stream -- not from the start of the chunk
    /// it came in on.
    pub first_frame: usize,
    /// Whole frames only: the length is always a multiple of [`FRAME`].
    pub samples: Vec<f32>,
}

/// Splits a stream of chunks into whole VAD frames, carrying the remainder.
///
/// This repeats, deliberately, what `real::Vad::run` does with its own
/// remainder -- and it has to, because the flags that call returns index the
/// *concatenated* stream rather than the chunk handed to it:
/// an ASR chunk is 8960 samples, which is 17.5 frames, so from the second
/// chunk onwards `voiced[0]` describes a frame that began in the chunk before.
/// The diarizer needs the audio those flags describe, so it splits the stream
/// the same way here.
///
/// The alternative considered was giving that remainder one owner -- `Vad`
/// holding a `Framer` and returning the frames it decided on alongside the
/// flags -- which would make disagreement impossible rather than detectable.
/// It was not taken because the two are not really one piece of state: the VAD
/// consumes a chunk into its remainder *before* its first inference, so a
/// mid-chunk failure leaves the model's own position ahead of the frames it
/// managed to decide, and the diarizer has to track the stream even for chunks
/// the VAD never reported on. Two independent counts that must agree, checked
/// on every chunk, catch that class of bug where a single shared count would
/// quietly absorb it.
#[derive(Default)]
pub struct Framer {
    carry: Vec<f32>,
    frames_seen: usize,
}

impl Framer {
    /// Take everything in `audio` that completes a frame, keeping the rest for
    /// the next call.
    ///
    /// The stream position advances whether or not the caller goes on to use
    /// the frames. A chunk the VAD failed on is consumed all the same -- the
    /// VAD swallowed it into its own remainder before its first inference --
    /// so it has to read downstream as a gap, not as audio that never existed.
    pub fn take(&mut self, audio: &[f32]) -> Framed {
        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(audio);
        let frames = buf.len() / FRAME;
        self.carry = buf.split_off(frames * FRAME);
        let first_frame = self.frames_seen;
        self.frames_seen += frames;
        Framed {
            first_frame,
            samples: buf,
        }
    }
}

/// Accumulates voiced audio into fixed-length windows with a fixed hop.
///
/// Only voiced frames go in: embedding silence or keyboard noise produces
/// vectors that characterise the room rather than a voice, and those pollute
/// every centroid they reach. So a window is 1.5 s of *speech*, which may have
/// taken any amount of wall-clock time to collect -- that is exactly why the
/// session's lag was measured against real window timings rather than derived
/// from the hop.
#[derive(Default)]
pub struct WindowAssembler {
    /// Voiced audio so far, oldest first, whole frames only.
    voiced: VecDeque<f32>,
    /// The hop being assembled: voiced audio since the last [`Cut::Hop`],
    /// which is *not* a suffix of `voiced` -- a boundary cut throws away part
    /// of one and none of the other.
    hop: Vec<f32>,
    /// Index of the most recent voiced frame, for the gap rule.
    last: Option<usize>,
}

impl WindowAssembler {
    /// Feed one chunk's frames and their speech/non-speech flags, and take
    /// whatever they completed, oldest first.
    ///
    /// Usually nothing: a hop is 23 voiced frames and a chunk carries at most
    /// 18 frames of any kind, so no single chunk can ever complete two of
    /// either. It can complete one of each, and a caller that acts on a
    /// [`Cut::Hop`] by calling [`WindowAssembler::cut_at_boundary`] has to
    /// discard the [`Cut::Window`]s after it in the same batch -- they were
    /// built before the cut, which is precisely the mixture the cut says they
    /// are. `real::RealDiarizer` is the one caller and does exactly that.
    ///
    /// `voiced[i]` describes `framed.samples[i * FRAME..]`; anything past the
    /// shorter of the two is ignored rather than panicked over, since the
    /// flags come from a model and the audio does not.
    pub fn push(&mut self, framed: &Framed, voiced: &[bool]) -> Vec<Cut> {
        let mut cuts = Vec::new();

        // as_chunks, not chunks_exact: both truncate to whole frames, and this
        // one keeps the frame size a constant the compiler can see.
        let (frames, _) = framed.samples.as_chunks::<FRAME>();
        for (i, (frame, &speech)) in frames.iter().zip(voiced).enumerate() {
            if !speech {
                continue;
            }
            let index = framed.first_frame + i;
            // saturating: frames arrive in order, so this is a plain
            // subtraction in practice -- but a caller that ever handed over an
            // out-of-order chunk would get an underflow panic in the middle of
            // a session, and reading it as "no gap" is the safe answer.
            if self
                .last
                .is_some_and(|previous| index.saturating_sub(previous) > MAX_GAP_FRAMES)
            {
                self.voiced.clear();
                // The hop goes with it, for the same reason and not merely by
                // analogy: a hop spliced across half a second of silence is
                // two voices in one vector, and the whole use of a hop is
                // being compared against its neighbour as if it were one.
                self.hop.clear();
            }
            self.last = Some(index);
            self.voiced.extend(frame);
            self.hop.extend_from_slice(frame);

            // Before the window, because the hop is what decides whether the
            // window about to complete is worth anything -- see the note above
            // about discarding the rest of the batch.
            if self.hop.len() >= HOP_SAMPLES {
                cuts.push(Cut::Hop(std::mem::take(&mut self.hop)));
            }
            if self.voiced.len() >= WINDOW_SAMPLES {
                cuts.push(Cut::Window(
                    self.voiced.iter().take(WINDOW_SAMPLES).copied().collect(),
                ));
                self.voiced.drain(..HOP_SAMPLES);
            }
        }
        cuts
    }

    /// Throw away everything accumulated before the most recent hop, because
    /// the voice changed at the start of it.
    ///
    /// The two mechanisms this fixes are stacked and were both measured. The
    /// only thing that used to reset the accumulator was `MAX_GAP_FRAMES`, and
    /// a normal turn change arrives with less than half a second of silence in
    /// front of it -- so at the handover the accumulator still held up to 46
    /// frames of the outgoing speaker, and every window for the next 1.5 s of
    /// the incoming speaker's *speech* was an A/B mixture. Cutting here makes
    /// the next full window one voice, and its resolution is the hop: 0.736 s,
    /// against an effective resolution of a window and a half.
    ///
    /// A hop's worth survives, and that is the point of cutting *at* the
    /// boundary rather than clearing outright: the hop that found the change
    /// is the incoming speaker's, so keeping it starts the next window 0.75 s
    /// ahead. Less than a hop in hand means the accumulator holds nothing
    /// older than the boundary anyway, and all of it stays.
    pub fn cut_at_boundary(&mut self) {
        let keep = self.voiced.len().min(HOP_SAMPLES);
        self.voiced.drain(..self.voiced.len() - keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 8960 samples: one ASR chunk, and 17.5 frames.
    const CHUNK: usize = 8960;

    // ------------------------------------------------------------- geometry

    #[test]
    fn the_frame_counts_are_the_spikes_seconds() {
        let seconds = |frames: usize| frames as f32 * FRAME as f32 / 16_000.0;

        // Window and hop round to the nearest frame, so half a frame is as
        // close as they get to 1.5 s and 0.75 s.
        for (frames, target, what) in [(WINDOW_FRAMES, 1.5, "window"), (HOP_FRAMES, 0.75, "hop")] {
            assert!(
                (seconds(frames) - target).abs() <= 0.016,
                "{what}: {frames} frames is {:.3}s, not {target}s",
                seconds(frames)
            );
        }

        // The gap truncates instead -- the spike's own conversion -- so it is
        // the largest whole number of frames that still fits in half a second.
        assert!(seconds(MAX_GAP_FRAMES) <= 0.5);
        assert!(seconds(MAX_GAP_FRAMES + 1) > 0.5);
    }

    // --------------------------------------------------------------- Framer

    #[test]
    fn frames_are_numbered_across_chunks_not_within_them() {
        // The bug this prevents: 8960 samples is 17.5 frames, so a diarizer
        // that restarted its numbering every chunk would drift half a frame
        // per chunk against the flags the VAD returns.
        let mut framer = Framer::default();
        let chunk = vec![0.0f32; CHUNK];

        let first = framer.take(&chunk);
        assert_eq!(first.first_frame, 0);
        assert_eq!(first.samples.len(), 17 * FRAME);

        let second = framer.take(&chunk);
        assert_eq!(
            second.first_frame, 17,
            "numbering continues, it does not reset"
        );
        assert_eq!(
            second.samples.len(),
            18 * FRAME,
            "the carried half-frame completes"
        );
    }

    #[test]
    fn the_frames_reassemble_the_stream_in_order() {
        let mut framer = Framer::default();
        let stream: Vec<f32> = (0..CHUNK * 3).map(|i| i as f32).collect();

        let mut seen = Vec::new();
        for chunk in stream.chunks(CHUNK) {
            let framed = framer.take(chunk);
            assert_eq!(framed.first_frame, seen.len() / FRAME);
            seen.extend_from_slice(&framed.samples);
        }

        // Everything but the trailing partial frame, in the original order.
        let whole = seen.len();
        assert_eq!(seen, stream[..whole]);
        assert!(stream.len() - whole < FRAME);
    }

    // ------------------------------------------------- WindowAssembler

    /// `n` frames of a distinguishable constant, starting at frame `first`.
    fn frames(first: usize, n: usize, value: f32) -> Framed {
        Framed {
            first_frame: first,
            samples: vec![value; n * FRAME],
        }
    }

    /// Just the completed windows, for the tests that predate hops and are
    /// about window geometry alone.
    fn windows(cuts: Vec<Cut>) -> Vec<Vec<f32>> {
        cuts.into_iter()
            .filter_map(|c| match c {
                Cut::Window(w) => Some(w),
                Cut::Hop(_) => None,
            })
            .collect()
    }

    fn hops(cuts: Vec<Cut>) -> Vec<Vec<f32>> {
        cuts.into_iter()
            .filter_map(|c| match c {
                Cut::Hop(h) => Some(h),
                Cut::Window(_) => None,
            })
            .collect()
    }

    #[test]
    fn a_window_needs_a_windows_worth_of_voiced_audio() {
        let mut a = WindowAssembler::default();

        let short = windows(a.push(
            &frames(0, WINDOW_FRAMES - 1, 1.0),
            &[true; WINDOW_FRAMES - 1],
        ));
        assert!(short.is_empty());

        let complete = windows(a.push(&frames(WINDOW_FRAMES - 1, 1, 1.0), &[true]));
        assert_eq!(complete.len(), 1);
        assert_eq!(complete[0].len(), WINDOW_SAMPLES);
        assert!(complete[0].iter().all(|&x| x == 1.0));
    }

    // ----------------------------------------------------------------- hops

    #[test]
    fn a_hop_of_voiced_audio_is_cut_on_its_own() {
        // The extra embedding this module exists to make possible: 0.75 s of
        // new voiced audio, handed over as soon as it exists rather than only
        // as part of the 1.5 s window it will later belong to. It is what
        // answers "has the voice changed" a window and a half sooner than the
        // window could.
        let mut a = WindowAssembler::default();
        let short = a.push(&frames(0, HOP_FRAMES - 1, 1.0), &[true; HOP_FRAMES - 1]);
        assert!(hops(short).is_empty(), "a hop is a whole hop or nothing");

        let cut = hops(a.push(&frames(HOP_FRAMES - 1, 1, 1.0), &[true]));
        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].len(), HOP_SAMPLES);
        assert!(cut[0].iter().all(|&x| x == 1.0));
    }

    #[test]
    fn each_hop_is_the_audio_since_the_last_one_and_no_more() {
        // Consecutive hops must not overlap, or the cosine between them would
        // measure their shared audio rather than the voice on either side of
        // the boundary, and every change would look smaller than it is.
        let mut a = WindowAssembler::default();
        let mut cut = Vec::new();
        for i in 0..HOP_FRAMES * 3 {
            cut.extend(hops(a.push(&frames(i, 1, i as f32), &[true])));
        }
        assert_eq!(cut.len(), 3);
        for (n, hop) in cut.iter().enumerate() {
            assert_eq!(hop.len(), HOP_SAMPLES);
            assert_eq!(hop[0] as usize, n * HOP_FRAMES, "hop {n} starts late");
            assert_eq!(hop[hop.len() - 1] as usize, (n + 1) * HOP_FRAMES - 1);
        }
    }

    #[test]
    fn unvoiced_frames_never_reach_a_hop_either() {
        // Same reason as the window: a hop of room tone characterises the
        // room, and comparing two of those finds a turn change wherever the
        // air conditioning changed note.
        let mut a = WindowAssembler::default();
        let mut cut = Vec::new();
        for i in 0..HOP_FRAMES * 2 {
            let value = if i % 2 == 0 { 1.0 } else { -1.0 };
            cut.extend(hops(a.push(&frames(i, 1, value), &[i % 2 == 0])));
        }
        assert_eq!(cut.len(), 1);
        assert!(cut[0].iter().all(|&x| x == 1.0));
    }

    #[test]
    fn a_long_silence_starts_the_hop_over_as_well_as_the_window() {
        // A hop spliced across half a second of silence is the same two-voice
        // mixture the window rule exists to prevent, and it would be compared
        // against its neighbour as if it were one voice.
        let mut a = WindowAssembler::default();
        a.push(&frames(0, HOP_FRAMES - 1, 1.0), &[true; HOP_FRAMES - 1]);
        let last_voiced = HOP_FRAMES - 2;

        let resumes_at = last_voiced + MAX_GAP_FRAMES + 1;
        let cut = hops(a.push(&frames(resumes_at, 1, 2.0), &[true]));
        assert!(cut.is_empty(), "the hop completed across the silence");

        let mut out = Vec::new();
        for i in 1..HOP_FRAMES {
            out.extend(hops(a.push(&frames(resumes_at + i, 1, 2.0), &[true])));
        }
        assert_eq!(out.len(), 1);
        assert!(out[0].iter().all(|&x| x == 2.0), "two turns in one hop");
    }

    // ------------------------------------------------------- boundary cuts

    #[test]
    fn a_boundary_cut_keeps_the_hop_that_found_it() {
        // The hop whose voice differs from the one before it is the *new*
        // speaker's, so the boundary is at its start: everything older is the
        // outgoing speaker and goes, and the hop itself stays. Throwing it
        // away too would cost 0.75 s of the new speaker for nothing.
        let mut a = WindowAssembler::default();
        for i in 0..HOP_FRAMES * 2 {
            let value = if i < HOP_FRAMES { 1.0 } else { 2.0 };
            a.push(&frames(i, 1, value), &[true]);
        }
        assert_eq!(a.voiced.len(), 2 * HOP_SAMPLES);

        a.cut_at_boundary();
        assert_eq!(a.voiced.len(), HOP_SAMPLES, "a hop's worth should survive");
        assert!(
            a.voiced.iter().all(|&x| x == 2.0),
            "the outgoing speaker's audio survived the cut"
        );
    }

    #[test]
    fn a_window_after_a_boundary_cut_holds_one_voice_only() {
        // The point of cutting at all. Without it the accumulator carries up
        // to 46 frames of the previous speaker, so the next window is an A/B
        // mixture and a window that is purely B is 1.5 s of B's speech away.
        let mut a = WindowAssembler::default();
        for i in 0..HOP_FRAMES * 2 {
            let value = if i < HOP_FRAMES { 1.0 } else { 2.0 };
            a.push(&frames(i, 1, value), &[true]);
        }
        a.cut_at_boundary();

        // A hop survived the cut, so a window needs only the rest of one --
        // which is the 0.75 s the cut is there to save.
        let mut out = Vec::new();
        for i in HOP_FRAMES * 2..HOP_FRAMES * 2 + (WINDOW_FRAMES - HOP_FRAMES) {
            out.extend(windows(a.push(&frames(i, 1, 2.0), &[true])));
        }
        assert_eq!(out.len(), 1, "the cut should have left a hop to build on");
        assert!(
            out[0].iter().all(|&x| x == 2.0),
            "the window still mixes two speakers"
        );
    }

    #[test]
    fn a_boundary_cut_with_nothing_to_cut_is_harmless() {
        // The diarizer cuts on a change point, and a change point can be
        // found on the very first pair of hops, when the accumulator holds
        // less than a hop's worth of anything.
        let mut a = WindowAssembler::default();
        a.cut_at_boundary();
        assert_eq!(a.voiced.len(), 0);
        a.push(&frames(0, 3, 1.0), &[true; 3]);
        a.cut_at_boundary();
        assert_eq!(a.voiced.len(), 3 * FRAME, "less than a hop is all of it");
    }

    #[test]
    fn unvoiced_frames_never_reach_a_window() {
        // The whole point of the VAD: silence and keyboard noise must not be
        // embedded, so the window is made of the voiced frames spliced
        // together, not of the audio between the first and the last.
        let mut a = WindowAssembler::default();
        let mut out = Vec::new();
        // Alternate voiced (1.0) and unvoiced (-1.0) frames, well inside the
        // gap budget, until a window falls out.
        for i in 0..WINDOW_FRAMES * 2 {
            let value = if i % 2 == 0 { 1.0 } else { -1.0 };
            out.extend(windows(a.push(&frames(i, 1, value), &[i % 2 == 0])));
        }
        assert_eq!(out.len(), 1);
        assert!(
            out[0].iter().all(|&x| x == 1.0),
            "an unvoiced frame was embedded"
        );
    }

    #[test]
    fn windows_slide_by_the_hop() {
        let mut a = WindowAssembler::default();
        let mut completed = 0;
        // Each frame carries its own index, so a window's contents say exactly
        // which frames it was built from.
        for i in 0..WINDOW_FRAMES + HOP_FRAMES {
            let window = windows(a.push(&frames(i, 1, i as f32), &[true]));
            if let Some(w) = window.first() {
                completed += 1;
                let first = w[0] as usize;
                assert_eq!(
                    first,
                    (completed - 1) * HOP_FRAMES,
                    "window {completed} started at frame {first}"
                );
                assert_eq!(w[w.len() - 1] as usize, first + WINDOW_FRAMES - 1);
            }
        }
        assert_eq!(completed, 2, "one window, then one more a hop later");
    }

    /// The gap rule's exact boundary, from both sides, since `>` and `>=` are
    /// one keystroke apart and the spike's comparison is `>`: a distance of
    /// MAX_GAP_FRAMES between two voiced frames still splices them, and one
    /// frame further is the first that does not. A test that jumps well past
    /// the boundary would pass against either comparison.
    #[test]
    fn the_gap_boundary_is_where_the_spike_put_it() {
        for (distance, kept) in [(MAX_GAP_FRAMES, 2), (MAX_GAP_FRAMES + 1, 1)] {
            let mut a = WindowAssembler::default();
            a.push(&frames(0, 1, 1.0), &[true]);
            a.push(&frames(distance, 1, 1.0), &[true]);
            assert_eq!(
                a.voiced.len(),
                kept * FRAME,
                "at a distance of {distance} frames the accumulator should hold {kept}"
            );
        }
    }

    #[test]
    fn a_long_silence_starts_the_window_over() {
        let mut a = WindowAssembler::default();
        // Nearly a window's worth of one speaker...
        a.push(
            &frames(0, WINDOW_FRAMES - 1, 1.0),
            &[true; WINDOW_FRAMES - 1],
        );
        let last_voiced = WINDOW_FRAMES - 2;

        // ...then a silence long enough that splicing across it would build a
        // window out of two different turns. Measured from the last voiced
        // frame, which is what the rule compares against.
        let resumes_at = last_voiced + MAX_GAP_FRAMES + 1;
        let resumed = windows(a.push(&frames(resumes_at, 1, 2.0), &[true]));
        assert!(resumed.is_empty());
        assert_eq!(
            a.voiced.len(),
            FRAME,
            "the half-built window went with the silence"
        );

        // And the next window is built from the new turn alone: a full
        // WINDOW_FRAMES of it, not a splice across the silence.
        let mut out = Vec::new();
        for i in 1..WINDOW_FRAMES {
            out.extend(windows(a.push(&frames(resumes_at + i, 1, 2.0), &[true])));
        }
        assert_eq!(out.len(), 1);
        assert!(out[0].iter().all(|&x| x == 2.0));
    }

    #[test]
    fn frames_lost_to_a_failed_chunk_read_as_a_gap() {
        // When Vad::run fails part-way through a chunk, the diarizer drops
        // that chunk's frames but still advances the stream position. The
        // frames are gone, so the accumulator must see the hole rather than
        // splice the audio either side of it together.
        let mut a = WindowAssembler::default();
        a.push(&frames(0, 10, 1.0), &[true; 10]);
        let last_voiced = 9;

        a.push(&frames(last_voiced + MAX_GAP_FRAMES + 1, 1, 2.0), &[true]);
        assert_eq!(a.voiced.len(), FRAME, "the audio either side was spliced");
    }

    #[test]
    fn a_flag_without_audio_behind_it_is_ignored() {
        // as_chunks truncates to the audio that is actually there: a
        // mismatch is the caller's bug to report, never a panic in here.
        let mut a = WindowAssembler::default();
        let out = a.push(&frames(0, 1, 1.0), &[true; 4]);
        assert!(out.is_empty());
        assert_eq!(a.voiced.len(), FRAME);
    }

    #[test]
    fn chunked_and_whole_stream_pushes_agree() {
        // The property the streaming path rests on: feeding the assembler one
        // ASR chunk at a time must produce exactly the windows that feeding it
        // the whole stream at once would.
        let frames_total = WINDOW_FRAMES * 3;
        let flags: Vec<bool> = (0..frames_total).map(|i| i % 11 != 3).collect();
        let audio: Vec<f32> = (0..frames_total * FRAME).map(|i| i as f32).collect();

        let mut whole = WindowAssembler::default();
        let at_once = whole.push(
            &Framed {
                first_frame: 0,
                samples: audio.clone(),
            },
            &flags,
        );

        let mut piecewise = WindowAssembler::default();
        let mut windows = Vec::new();
        // 17 frames at a time: the shape an ASR chunk actually arrives in.
        for (c, chunk) in audio.chunks(17 * FRAME).enumerate() {
            let first_frame = c * 17;
            let n = chunk.len() / FRAME;
            windows.extend(piecewise.push(
                &Framed {
                    first_frame,
                    samples: chunk.to_vec(),
                },
                &flags[first_frame..first_frame + n],
            ));
        }

        assert!(
            !windows.is_empty(),
            "the fixture produced no windows at all"
        );
        assert_eq!(windows, at_once);
    }
}
