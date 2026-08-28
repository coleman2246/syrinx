//! Auditing the gap guard one decision at a time.
//!
//! `live` scores the guard by its consequences: corrections emitted, corrected
//! miss and confusion. That is the right way to ask whether the feature is
//! worth having and the wrong way to place its threshold, because the two
//! populations are different sizes by two orders of magnitude. A meeting
//! produces a few hundred gap decisions and a handful of corrections, so almost
//! every decision the guard makes lands on audio no correction ever reaches --
//! it moves no aggregate, and a threshold chosen from aggregates is chosen from
//! the dozen decisions that happened to matter.
//!
//! This asks the other question. Every comparison the guard makes is recorded
//! with the AMI annotation's answer beside it, and a threshold is then scored
//! on the decisions themselves: how many same-speaker seams it correctly
//! discards, and how many times it discards a seam across a *real* speaker
//! change. The second column is the one the guard exists to keep at zero, and
//! it is invisible from the outcome metrics.
//!
//! **What is the server's code here.** The VAD, `WindowAssembler`, the
//! embedder and `cluster::cosine` are the server's own. What this module
//! carries is the twenty lines of `RealDiarizer::hop` and `batch` that decide
//! *when* a comparison happens: keep the last completed hop, count the
//! restarts since it, and compare the next hop to complete against it. The
//! assembler is driven one frame at a time rather than one chunk at a time,
//! which is what makes `WindowAssembler::restarted` name the exact frame the
//! silence broke at -- the same rule, asked at a finer grain.
//!
//! ```text
//! ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so cargo run --release \
//!   -p syrinx-server --features diarize --example diarize_probe -- \
//!   gaps --wav ES2002a.Mix-Headset.wav
//! ```

use anyhow::{Result, ensure};

use syrinx_server::diarize::cluster::cosine;
use syrinx_server::diarize::real::Embedder;
use syrinx_server::diarize::window::{Cut, FRAME, Framed, WindowAssembler};

use crate::reference::{self, Reference};

/// Samples per 10 ms reference frame at 16 kHz.
const SAMPLES_PER_FRAME: usize = 160;

/// One comparison the guard made, with the annotation's answer beside it.
pub struct Decision {
    /// Chunk the hop before the silence completed in.
    pub before_chunk: u64,
    /// Chunk the hop after it completed in -- where the seam would be laid.
    pub after_chunk: u64,
    /// What the guard compares against its threshold.
    pub cos: f32,
    /// Reference speaker of each hop, where the annotation is clear enough to
    /// say. A decision with either side `None` is not decidable and is counted
    /// but never scored.
    pub before: Option<usize>,
    pub after: Option<usize>,
    /// Silences this one comparison was asked to answer for. One is the shape
    /// the design describes; more means speech went by that never completed a
    /// hop, so it is not on either side of the comparison at all.
    pub restarts: u32,
}

impl Decision {
    /// Whether the annotation can say if this was one voice or two.
    pub fn decidable(&self) -> Option<bool> {
        Some(self.before? == self.after?)
    }
    /// Chunks between the two hops: what a discarded seam lets a correction
    /// reach back over.
    pub fn bridged(&self) -> u64 {
        self.after_chunk.saturating_sub(self.before_chunk)
    }
}

/// The voiced frames `frames` names, spliced back together the way the
/// assembler splices them.
fn audio_of(samples: &[f32], frames: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames.len() * FRAME);
    for &f in frames {
        out.extend_from_slice(&samples[f * FRAME..(f + 1) * FRAME]);
    }
    out
}

/// A completed hop: its embedding, and the reference frames it was built from.
struct Hop {
    embedding: Vec<f32>,
    /// One range of 10 ms reference frames per 32 ms voiced frame, in order.
    spans: Vec<(usize, usize)>,
    chunk: u64,
}

impl Hop {
    fn speaker(&self, reference: &Reference) -> Option<usize> {
        reference::span_speaker(reference, &self.spans)
    }
}

/// Replay one recording and record every comparison the gap guard makes.
///
/// Threshold-free on purpose: the guard's *input* is a cosine and its
/// population does not depend on where the bar is, so one pass produces the
/// evidence for every candidate value at once. That is also what keeps the
/// table honest -- every row scores the same recorded decisions rather than a
/// set generated afresh under its own value.
pub fn audit(
    samples: &[f32],
    voiced: &[bool],
    reference: &Reference,
    embedder: &mut Embedder,
    chunk_samples: usize,
) -> Result<Vec<Decision>> {
    let frames = voiced.len().min(samples.len() / FRAME);
    let mut assembler = WindowAssembler::default();

    // The frames of each hop as it accumulates, mirroring the assembler's own
    // accumulator: cleared when it restarts, taken when it emits a hop.
    let mut building: Vec<usize> = Vec::new();
    let mut pending: Vec<(Vec<usize>, u64, u32)> = Vec::new();
    let mut restarts = 0u32;

    for (i, &speech) in voiced[..frames].iter().enumerate() {
        if !speech {
            continue;
        }
        let framed = Framed {
            first_frame: i,
            samples: samples[i * FRAME..(i + 1) * FRAME].to_vec(),
        };
        let cuts = assembler.push(&framed, &[true]);
        // Before the frame is recorded, because the restart happens before the
        // assembler extends: this frame is the first of the new hop, not the
        // last of the old one.
        if assembler.restarted() {
            building.clear();
            restarts += 1;
        }
        building.push(i);
        for cut in cuts {
            let Cut::Hop(hop) = cut else { continue };
            // The mirror is what every ground-truth attribution below rests
            // on, so it is proved against the assembler's own output rather
            // than reasoned about -- the same discipline
            // `agrees_with_the_shipped_assembler` applies to windows. If the
            // two ever disagree, the frames named here are not the audio that
            // was embedded and the annotation is being read at the wrong time.
            ensure!(
                hop == audio_of(samples, &building),
                "the hop completing at frame {i} is not the audio this module \
                 thinks it is"
            );
            let chunk = ((i * FRAME) / chunk_samples) as u64;
            pending.push((std::mem::take(&mut building), chunk, restarts));
            restarts = 0;
        }
    }

    // Embedded in batches after the walk rather than during it: the walk is
    // arithmetic over cached VAD flags and takes seconds, while embedding is
    // the whole cost of this subcommand and wants every core.
    let mut hops: Vec<Hop> = Vec::with_capacity(pending.len());
    for batch in pending.chunks(16) {
        let audio: Vec<Vec<f32>> = batch
            .iter()
            .map(|(frames, _, _)| audio_of(samples, frames))
            .collect();
        let refs: Vec<&[f32]> = audio.iter().map(Vec::as_slice).collect();
        for (embedding, (frames, chunk, _)) in embedder.embed_batch(&refs)?.into_iter().zip(batch) {
            hops.push(Hop {
                embedding,
                spans: frames
                    .iter()
                    .map(|&f| {
                        (
                            f * FRAME / SAMPLES_PER_FRAME,
                            (f + 1) * FRAME / SAMPLES_PER_FRAME,
                        )
                    })
                    .collect(),
                chunk: *chunk,
            });
        }
    }

    let mut out = Vec::new();
    for (i, hop) in hops.iter().enumerate() {
        let restarts = pending[i].2;
        // No restart since the last hop is no silence to answer for, and the
        // first hop of a session has nothing before it -- the guard commits
        // that seam without a comparison, so there is no decision to score.
        if restarts == 0 || i == 0 {
            continue;
        }
        let before = &hops[i - 1];
        out.push(Decision {
            before_chunk: before.chunk,
            after_chunk: hop.chunk,
            cos: cosine(&before.embedding, &hop.embedding),
            before: before.speaker(reference),
            after: hop.speaker(reference),
            restarts,
        });
    }
    Ok(out)
}

/// How one threshold would have decided a recording's gaps.
pub struct Tally {
    /// Same-speaker seams correctly discarded: the whole benefit, counted in
    /// the decisions that produce it rather than in the corrections that
    /// happen to land on one.
    pub correct: usize,
    /// Seams discarded across a real speaker change. The number this
    /// threshold exists to hold at zero.
    pub crossings: usize,
}

/// Score every decision at one threshold, under a bound on how many silences
/// one pending seam may bridge. `max_restarts` of 0 leaves the chaining
/// unbounded, which is what the rule looked like before
/// `real::MAX_BRIDGED_SILENCES` and is how its cost is measured.
pub fn tally(decisions: &[Decision], threshold: f32, max_restarts: u32) -> Tally {
    let mut out = Tally {
        correct: 0,
        crossings: 0,
    };
    for d in decisions {
        // The guard's own rule, and the two fail-safes with it: a comparison
        // that cannot be made commits the seam, and so does one asked to
        // answer for more silences than the bound allows.
        let bounded = max_restarts > 0 && d.restarts > max_restarts;
        let discarded = !bounded && d.cos.is_finite() && d.cos >= threshold;
        if !discarded {
            continue;
        }
        match d.decidable() {
            Some(true) => out.correct += 1,
            Some(false) => out.crossings += 1,
            None => {}
        }
    }
    out
}
