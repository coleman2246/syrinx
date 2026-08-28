//! Replaying a real session over a recording, so the two latencies the
//! 2026-08-27 design targets can be measured.
//!
//! Every number the design doc publishes was produced by [`super::label_frames`],
//! which paints each 10 ms frame with the majority of the windows covering it.
//! That is backwards painting, and `main.rs` says so in its header. It is
//! honest about what it measures -- a batch diarization's accuracy -- and it
//! is structurally incapable of measuring either latency the complaints are
//! about, because it labels a frame using windows that had not finished when
//! that frame was spoken.
//!
//! So the live path has always been the outlier, and this module is what stops
//! it being one. It runs **the server's own code**: `RealDiarizerFactory` for
//! the pipeline, and a real [`Session`] for the lag buffer, the majority vote,
//! the carry-forward and the corrections. The only thing faked is the ASR --
//! `MockBackend` emits one scripted word per chunk, and the words are chunk
//! numbers, so each commit says which chunk it came from. Nothing about
//! labelling is reimplemented here, because a harness that reimplemented it
//! would be measuring the reimplementation.
//!
//! **These numbers have never been produced.** The constants the design added
//! are engineering estimates, and running this is what replaces them.
//!
//! ```text
//! ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so cargo run --release \
//!   -p syrinx-server --features diarize --example diarize_probe -- \
//!   live ES2002a.Mix-Headset.wav
//! ```

use anyhow::{Context, Result, ensure};

use syrinx_proto::{Mode, ServerMessage};
use syrinx_server::asr::mock::MockBackend;
use syrinx_server::diarize::real::RealDiarizerFactory;
use syrinx_server::diarize::{DiarizeTuning, DiarizerFactory};
use syrinx_server::session::{Session, SessionTuning};

use crate::reference::{self, Reference};

/// Samples per 10 ms reference frame at 16 kHz. Everything spliced or painted
/// below is aligned to this, so a slice boundary is always a frame boundary.
const SAMPLES_PER_FRAME: usize = 160;

/// What one recording's live replay produced.
pub struct Replay {
    /// The label the session committed for each chunk, as it went out.
    pub live: Vec<Option<u32>>,
    /// The same after every `transcript.relabel` has been applied -- what the
    /// GUI shows and what Save-as writes.
    pub corrected: Vec<Option<u32>>,
    /// Corrections sent, and commits they covered.
    pub relabels: usize,
    pub relabelled_commits: usize,
}

/// Drive a real session over `samples` and record what it committed.
///
/// The chunk each commit belongs to is recovered from its text, because
/// `MockBackend` emits one scripted word per chunk and the words here are the
/// chunk numbers. That is the whole trick: it lets the session be used exactly
/// as the server uses it, with no extra seam cut into it for the harness.
pub fn replay(
    samples: &[f32],
    factory: &RealDiarizerFactory,
    chunk_samples: usize,
    tuning: SessionTuning,
) -> Result<Replay> {
    let chunks = samples.len().div_ceil(chunk_samples);
    let words: Vec<String> = (0..chunks + 1).map(|i| i.to_string()).collect();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    let backend = MockBackend::new(&refs).with_chunk_samples(chunk_samples);

    let mut session = Session::with_tuning(
        Mode::Transcript,
        &backend,
        "probe".into(),
        Some(factory.diarizer()),
        tuning,
    );

    let mut messages = Vec::new();
    // Fed in chunk-sized pieces rather than all at once, so the diarizer sees
    // the arrival pattern a socket produces rather than one enormous push.
    for piece in samples.chunks(chunk_samples) {
        messages.extend(session.push_audio(piece)?);
    }
    messages.extend(session.finish()?);

    let mut live = vec![None; chunks];
    let mut corrected = vec![None; chunks];
    let mut chunk_of_seq: Vec<(u64, usize)> = Vec::new();
    let (mut relabels, mut relabelled_commits) = (0usize, 0usize);

    for m in &messages {
        match m {
            ServerMessage::TranscriptCommit {
                seq, text, speaker, ..
            } => {
                let Ok(chunk) = text.trim().parse::<usize>() else {
                    continue;
                };
                if chunk < chunks {
                    live[chunk] = *speaker;
                    corrected[chunk] = *speaker;
                    chunk_of_seq.push((*seq, chunk));
                }
            }
            ServerMessage::TranscriptRelabel {
                from_seq,
                to_seq,
                speaker,
            } => {
                relabels += 1;
                for (seq, chunk) in &chunk_of_seq {
                    if seq >= from_seq && seq <= to_seq {
                        corrected[*chunk] = Some(*speaker);
                        relabelled_commits += 1;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Replay {
        live,
        corrected,
        relabels,
        relabelled_commits,
    })
}

/// Per-chunk labels painted onto the 10 ms grid the reference is scored on.
///
/// Forwards, unlike [`super::label_frames`]: a chunk's label covers the audio
/// that chunk carried and nothing earlier. That is the whole difference this
/// module exists to expose.
///
/// This says *which audio a label describes*, not when it arrived. A commit is
/// released `lag_chunks` after the chunk its words end on, so the two latency
/// measurements below add that back; the accuracy figures scored off this grid
/// are unaffected, since they ask who was speaking rather than when anybody
/// found out.
pub fn paint(labels: &[Option<u32>], chunk_samples: usize, frames: usize) -> Vec<Option<u32>> {
    let mut out = vec![None; frames];
    let per_chunk = chunk_samples / SAMPLES_PER_FRAME;
    for (c, label) in labels.iter().enumerate() {
        let lo = (c * per_chunk).min(frames);
        let hi = (lo + per_chunk).min(frames);
        for f in &mut out[lo..hi] {
            *f = *label;
        }
    }
    out
}

/// Which reference speaker each emitted label mostly covered.
///
/// A greedy best-overlap rather than the exact assignment `reference::score`
/// runs, and adequate here: this answers "whose label is this" so a latency
/// can be attributed, not "how well did the mapping do".
fn speaker_of_label(hyp: &[Option<u32>], reference: &Reference) -> Vec<(u32, usize)> {
    let n = reference.frames.len().min(hyp.len());
    let mut counts: Vec<(u32, Vec<usize>)> = Vec::new();
    for (&mask, label) in reference.frames[..n].iter().zip(&hyp[..n]) {
        if mask.count_ones() != 1 {
            continue;
        }
        let Some(label) = label else { continue };
        let speaker = mask.trailing_zeros() as usize;
        match counts.iter_mut().find(|(l, _)| l == label) {
            Some((_, tally)) => tally[speaker] += 1,
            None => {
                let mut tally = vec![0usize; reference.names.len()];
                tally[speaker] = 1;
                counts.push((*label, tally));
            }
        }
    }
    counts
        .into_iter()
        .filter_map(|(label, tally)| {
            tally
                .iter()
                .enumerate()
                .max_by_key(|(_, n)| **n)
                .filter(|(_, n)| **n > 0)
                .map(|(speaker, _)| (label, speaker))
        })
        .collect()
}

/// Seconds of speaker `who`'s own reference speech in `[from, to)` frames.
///
/// The right denominator for both latencies below, and not wall-clock: the
/// diarizer needs a window's worth of *that voice*, so a speaker who is silent
/// for a minute has not made the diarizer any later. It is also the unit the
/// mint rule is written in -- four windows is about 3.7 s of one voice.
fn own_speech(reference: &Reference, who: usize, from: usize, to: usize) -> f32 {
    let to = to.min(reference.frames.len());
    if from >= to {
        return 0.0;
    }
    reference.frames[from..to]
        .iter()
        .filter(|m| m.count_ones() == 1 && m.trailing_zeros() as usize == who)
        .count() as f32
        * reference::FRAME_MS
}

/// The first frame at which each reference speaker is heard alone.
fn first_heard(reference: &Reference) -> Vec<Option<usize>> {
    let mut out = vec![None; reference.names.len()];
    for (f, &mask) in reference.frames.iter().enumerate() {
        if mask.count_ones() == 1 {
            let s = mask.trailing_zeros() as usize;
            out[s].get_or_insert(f);
        }
    }
    out
}

/// Every point at which the person holding the floor changes, as
/// `(frame, incoming speaker)`.
///
/// Silence and overlap do not end a turn -- the speaker is whoever last held
/// the floor alone -- so a pause mid-sentence is not counted as a boundary the
/// diarizer had to find. Same rule `reference::boundaries` uses, restated here
/// because this needs to know *who* the turn changed to.
fn turn_changes(reference: &Reference) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut current: Option<usize> = None;
    for (f, &mask) in reference.frames.iter().enumerate() {
        if mask.count_ones() != 1 {
            continue;
        }
        let who = mask.trailing_zeros() as usize;
        if current.is_some_and(|c| c != who) {
            out.push((f, who));
        }
        current = Some(who);
    }
    out
}

/// Voiced seconds from each speaker's first word to their first label
/// reaching a client, or `None` for a speaker who never got one.
///
/// Charged to the moment of *emission*, which is `lag_chunks` after the chunk
/// the label describes: the session holds a commit back while its label
/// settles, so a label painted onto chunk `c` is not on anybody's screen until
/// the commit released during chunk `c + lag_chunks` goes out. At the shipped
/// depth that is 1.12 s of the answer, and leaving it out understates every
/// number this subcommand exists to produce.
pub fn first_label_latency(
    hyp: &[Option<u32>],
    reference: &Reference,
    chunk_samples: usize,
    lag_chunks: usize,
) -> Vec<Option<f32>> {
    let mapping = speaker_of_label(hyp, reference);
    let first = first_heard(reference);
    let per_chunk = chunk_samples / SAMPLES_PER_FRAME;

    (0..reference.names.len())
        .map(|who| {
            let start = first[who]?;
            let label = mapping.iter().find(|(_, s)| *s == who)?.0;
            // The first frame carrying that speaker's label at or after they
            // started talking.
            let at =
                (start..reference.frames.len().min(hyp.len())).find(|&f| hyp[f] == Some(label))?;
            // Charged to the end of the commit that carries it: a label
            // reaches a client as a whole commit rather than as a frame, and
            // that commit waits out the lag first.
            let at = emitted_at(at, per_chunk, lag_chunks);
            Some(own_speech(reference, who, start, at))
        })
        .collect()
}

/// The frame at which a label painted onto frame `at` actually goes out.
///
/// Rounded up to the chunk that carries it, then forward by the lag the
/// session holds every commit for.
fn emitted_at(at: usize, per_chunk: usize, lag_chunks: usize) -> usize {
    let per_chunk = per_chunk.max(1);
    at.next_multiple_of(per_chunk) + lag_chunks * per_chunk
}

/// Voiced seconds from each real turn change to the emitted label changing to
/// the incoming speaker's, sorted. A turn the diarizer never picked up
/// contributes nothing, which is the miss rate's business rather than this
/// number's.
///
/// Charged at emission, exactly as [`first_label_latency`] is and for the same
/// reason: this is a latency a person waits, not a property of the labelling.
pub fn turn_switch_latency(
    hyp: &[Option<u32>],
    reference: &Reference,
    chunk_samples: usize,
    lag_chunks: usize,
) -> (Vec<f32>, usize) {
    let mapping = speaker_of_label(hyp, reference);
    let changes = turn_changes(reference);
    let per_chunk = chunk_samples / SAMPLES_PER_FRAME;
    let n = reference.frames.len().min(hyp.len());

    let mut out = Vec::new();
    for &(at, who) in &changes {
        let Some(&(label, _)) = mapping.iter().find(|(_, s)| *s == who) else {
            continue;
        };
        // Bounded by the next change, so a turn the diarizer picked up three
        // turns later does not count as a slow reaction to this one.
        let until = changes
            .iter()
            .find(|(f, _)| *f > at)
            .map_or(n, |(f, _)| (*f).min(n));
        if let Some(found) = (at..until).find(|&f| hyp[f] == Some(label)) {
            let found = emitted_at(found, per_chunk, lag_chunks);
            out.push(own_speech(reference, who, at, found));
        }
    }
    out.sort_by(f32::total_cmp);
    (out, changes.len())
}

/// Interleave several recordings into one room with all their speakers in it.
///
/// AMI gave at most five speakers and the problem statement says three to ten,
/// so the regime the design is most worried about is one the corpus cannot
/// supply. Splicing rather than summing, deliberately: summing two meetings
/// would put two people on top of each other for the whole recording, and
/// overlapped speech is excluded from every measured number in the design doc
/// and out of scope for the diarizer. Splicing produces a room where more
/// people take turns, which is the thing being measured.
///
/// Slices are whole reference frames, so the audio and the annotation stay
/// aligned to the sample.
pub fn splice(
    sources: Vec<(Vec<f32>, Reference)>,
    slice_frames: usize,
) -> Result<(Vec<f32>, Reference)> {
    ensure!(slice_frames > 0, "a slice has to be at least one frame");
    let mut names = Vec::new();
    let mut offsets = Vec::new();
    for (_, r) in &sources {
        offsets.push(names.len() as u32);
        names.extend(r.names.iter().cloned());
    }
    // The reference is a bitmask per frame, and a u8 holds eight speakers.
    // Widening it is a change to `reference.rs` and to every consumer of it,
    // so this refuses rather than silently dropping the ninth person.
    ensure!(
        names.len() <= 8,
        "{} speakers across {} recordings; the reference mask holds 8",
        names.len(),
        sources.len()
    );

    let mut samples: Vec<f32> = Vec::new();
    let mut frames: Vec<u8> = Vec::new();
    let mut cursor = vec![0usize; sources.len()];
    loop {
        let mut any = false;
        for (i, (audio, r)) in sources.iter().enumerate() {
            let start = cursor[i];
            if start >= r.frames.len() {
                continue;
            }
            any = true;
            let end = (start + slice_frames).min(r.frames.len());
            frames.extend(r.frames[start..end].iter().map(|m| m << offsets[i]));
            // Zero-padded where the wav is shorter than its annotation, so the
            // two never drift apart by a slice's worth of silence.
            let mut piece = vec![0.0f32; (end - start) * SAMPLES_PER_FRAME];
            let (a, b) = (start * SAMPLES_PER_FRAME, end * SAMPLES_PER_FRAME);
            if a < audio.len() {
                let n = (b.min(audio.len()) - a).min(piece.len());
                piece[..n].copy_from_slice(&audio[a..a + n]);
            }
            samples.extend_from_slice(&piece);
            cursor[i] = end;
        }
        if !any {
            break;
        }
    }
    Ok((samples, Reference { frames, names }))
}

/// Load the diarization models from the probe directory at the tuning a run
/// asked for.
pub fn factory(dir: &str, tuning: DiarizeTuning) -> Result<RealDiarizerFactory> {
    RealDiarizerFactory::load(std::path::Path::new(dir), tuning)
        .with_context(|| format!("loading the diarization models from {dir}"))
}
