//! AMI ground truth and the three measurements the design doc asks for.
//!
//! Reference turns come from the AMI manual annotations' word-level
//! alignments rather than the transcriber's `segments`, because segments
//! include the pauses inside a turn and would count silence as speech.
//!
//! Ported unchanged from the go/no-go spike -- `git log --follow` on this file
//! reaches it -- because every number in the design doc's "Spike results" was
//! produced by this scoring code, so it moves into the tree as it was rather
//! than being rewritten around the tree's types. Nothing here is production
//! code -- the server never scores itself against an annotation -- which is
//! why it lives beside the probe instead of in `src/diarize`.

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Everything is scored on a 10 ms grid -- the fbank frame rate, and finer
/// than the ~1 s boundary accuracy the design cares about.
pub const FRAME_MS: f32 = 0.01;

/// A hypothesis label must hold this share of a speaker's frames before it
/// counts as a real split rather than a stray window.
const SIGNIFICANT: f32 = 0.10;

pub struct Reference {
    /// Bitmask of active reference speakers per 10 ms frame.
    pub frames: Vec<u8>,
    pub names: Vec<String>,
}

impl Reference {
    /// Read `words/<meeting>.<X>.words.xml` for every speaker of a meeting.
    pub fn load_ami(annot_dir: &str, meeting: &str) -> Result<Self> {
        let mut names = Vec::new();
        let mut turns: Vec<(usize, f32, f32)> = Vec::new();

        for letter in ["A", "B", "C", "D", "E", "F", "G", "H"] {
            let path = format!("{annot_dir}/words/{meeting}.{letter}.words.xml");
            let Ok(xml) = std::fs::read_to_string(&path) else {
                continue;
            };

            let mut any = false;
            let idx = names.len();
            for line in xml.lines() {
                let line = line.trim();
                // Punctuation carries a zero-length timestamp and no voice.
                if !line.starts_with("<w ") || line.contains("punc=\"true\"") {
                    continue;
                }
                let (Some(start), Some(end)) = (attr(line, "starttime"), attr(line, "endtime"))
                else {
                    continue; // unaligned word
                };
                if end > start {
                    turns.push((idx, start, end));
                    any = true;
                }
            }
            if any {
                names.push(format!("{meeting}.{letter}"));
            }
        }
        anyhow::ensure!(!names.is_empty(), "no reference words found for {meeting}");

        let end = turns.iter().map(|t| t.2).fold(0.0, f32::max);
        let mut frames = vec![0u8; (end / FRAME_MS) as usize + 1];
        for (idx, start, stop) in turns {
            let lo = (start / FRAME_MS) as usize;
            let hi = ((stop / FRAME_MS) as usize).min(frames.len() - 1);
            for f in &mut frames[lo..=hi] {
                *f |= 1 << idx;
            }
        }
        Ok(Self { frames, names })
    }
}

fn attr(line: &str, name: &str) -> Option<f32> {
    let key = format!("{name}=\"");
    let rest = &line[line.find(&key)? + key.len()..];
    rest[..rest.find('"')?].parse().ok()
}

#[derive(Debug)]
pub struct Metrics {
    pub ref_speakers: usize,
    /// Labels the clusterer emitted at least once.
    pub hyp_speakers: usize,
    pub scored_frames: usize,
    /// Share of single-speaker reference frames the diarizer left unlabelled.
    pub miss: f32,
    /// Share attributed to the wrong speaker under the best 1-1 mapping.
    pub confusion: f32,
    /// Extra labels beyond one per real speaker, counting only labels that
    /// hold >=10% of that speaker's frames.
    pub splits: usize,
    /// Labels that significantly cover more than one real speaker.
    pub merges: usize,
    /// Dominant label per reference speaker in each third of the meeting;
    /// the design's "does Speaker 2 stay Speaker 2 in hour three" question.
    pub thirds: Vec<[Option<u32>; 3]>,
    /// Seconds of single-speaker reference speech per speaker. A speaker with
    /// only a few seconds cannot be split or merged meaningfully, and saying
    /// so is more honest than reporting a clean zero.
    pub per_speaker: Vec<f32>,
}

/// Score a per-frame hypothesis against the reference, ignoring frames where
/// two or more people talk at once -- overlap is explicitly out of scope.
pub fn score(hyp: &[Option<u32>], reference: &Reference) -> Metrics {
    let n = reference.frames.len().min(hyp.len());
    let nspk = reference.names.len();

    // overlap[speaker][label] = frames of that speaker carrying that label.
    let mut overlap: Vec<HashMap<u32, usize>> = vec![HashMap::new(); nspk];
    let mut per_speaker = vec![0usize; nspk];
    let mut thirds: Vec<[HashMap<u32, usize>; 3]> = (0..nspk).map(|_| Default::default()).collect();
    let mut scored = 0usize;
    let mut miss = 0usize;

    for (f, &mask) in reference.frames[..n].iter().enumerate() {
        if mask == 0 || mask.count_ones() != 1 {
            continue; // silence, or overlapped speech
        }
        let spk = mask.trailing_zeros() as usize;
        scored += 1;
        per_speaker[spk] += 1;

        match hyp[f] {
            None => miss += 1,
            Some(label) => {
                *overlap[spk].entry(label).or_default() += 1;
                let third = (f * 3 / n).min(2);
                *thirds[spk][third].entry(label).or_default() += 1;
            }
        }
    }

    let labels: Vec<u32> = {
        let mut all: Vec<u32> = overlap.iter().flat_map(|m| m.keys().copied()).collect();
        all.sort_unstable();
        all.dedup();
        all
    };

    let correct = best_mapping(&overlap, &labels, nspk);
    let labelled = scored - miss;

    let splits = overlap
        .iter()
        .zip(&per_speaker)
        .map(|(m, total)| {
            let significant = m
                .values()
                .filter(|&&c| c as f32 >= SIGNIFICANT * *total as f32)
                .count();
            significant.saturating_sub(1)
        })
        .sum();

    let merges = labels
        .iter()
        .map(|label| {
            let total: usize = overlap.iter().filter_map(|m| m.get(label)).sum();
            let significant = overlap
                .iter()
                .filter_map(|m| m.get(label))
                .filter(|&&c| c as f32 >= SIGNIFICANT * total as f32)
                .count();
            significant.saturating_sub(1)
        })
        .sum();

    Metrics {
        ref_speakers: nspk,
        hyp_speakers: labels.len(),
        scored_frames: scored,
        miss: miss as f32 / scored.max(1) as f32,
        confusion: (labelled - correct) as f32 / scored.max(1) as f32,
        splits,
        merges,
        thirds: thirds
            .iter()
            .map(|t| std::array::from_fn(|i| dominant(&t[i])))
            .collect(),
        per_speaker: per_speaker.iter().map(|&c| c as f32 * FRAME_MS).collect(),
    }
}

/// The reference speaker a window belongs to, or `None` if the window is
/// silence, or straddles a turn change, or covers overlapped speech. Used to
/// measure what the embeddings can do before any clustering is applied.
pub fn window_speaker(reference: &Reference, t0: f32, t1: f32) -> Option<usize> {
    let lo = (t0 / FRAME_MS) as usize;
    let hi = ((t1 / FRAME_MS) as usize).min(reference.frames.len());
    if lo >= hi {
        return None;
    }
    let mut counts = vec![0usize; reference.names.len()];
    let mut clean = 0usize;
    for &mask in &reference.frames[lo..hi] {
        if mask.count_ones() == 1 {
            counts[mask.trailing_zeros() as usize] += 1;
            clean += 1;
        }
    }
    let (best, &n) = counts.iter().enumerate().max_by_key(|(_, c)| **c)?;
    // Purity gate: the window must be one voice, not a handover.
    (n as f32 >= 0.9 * clean as f32 && n as f32 >= 0.5 * (hi - lo) as f32).then_some(best)
}

/// Frames of each reference speaker carrying each hypothesis label, in
/// seconds -- the raw picture behind splits and merges.
pub fn overlap_matrix(hyp: &[Option<u32>], reference: &Reference) -> (Vec<u32>, Vec<Vec<f32>>) {
    let n = reference.frames.len().min(hyp.len());
    let nspk = reference.names.len();
    let mut counts: Vec<HashMap<u32, usize>> = vec![HashMap::new(); nspk];

    for (&mask, label) in reference.frames[..n].iter().zip(&hyp[..n]) {
        if mask.count_ones() != 1 {
            continue;
        }
        if let Some(label) = label {
            *counts[mask.trailing_zeros() as usize]
                .entry(*label)
                .or_default() += 1;
        }
    }
    let mut labels: Vec<u32> = counts.iter().flat_map(|m| m.keys().copied()).collect();
    labels.sort_unstable();
    labels.dedup();

    let rows = counts
        .iter()
        .map(|m| {
            labels
                .iter()
                .map(|l| m.get(l).copied().unwrap_or(0) as f32 * FRAME_MS)
                .collect()
        })
        .collect();
    (labels, rows)
}

/// Times at which the person speaking changes. Silence and overlap do not
/// end a turn -- the speaker is whoever last held the floor alone -- so a pause
/// mid-sentence is not counted as a boundary the diarizer had to find.
fn changes<T: PartialEq + Copy>(seq: impl Iterator<Item = Option<T>>) -> Vec<f32> {
    let mut out = Vec::new();
    let mut current: Option<T> = None;
    for (f, value) in seq.enumerate() {
        let Some(value) = value else { continue };
        if current.is_some_and(|c| c != value) {
            out.push(f as f32 * FRAME_MS);
        }
        current = Some(value);
    }
    out
}

/// Distance from each time in `from` to the nearest time in `to`:
/// median, 90th percentile, and the share within 1 s.
fn nearest(from: &[f32], to: &[f32]) -> (f32, f32, f32) {
    if from.is_empty() || to.is_empty() {
        return (f32::NAN, f32::NAN, 0.0);
    }
    let mut offsets: Vec<f32> = from
        .iter()
        .map(|t| {
            let i = to.partition_point(|f| f < t);
            let before = if i > 0 { t - to[i - 1] } else { f32::MAX };
            let after = to.get(i).map_or(f32::MAX, |f| f - t);
            before.min(after)
        })
        .collect();
    offsets.sort_by(f32::total_cmp);
    (
        offsets[offsets.len() / 2],
        offsets[offsets.len() * 9 / 10],
        offsets.iter().filter(|d| **d <= 1.0).count() as f32 / offsets.len() as f32,
    )
}

pub struct Boundaries {
    pub reference_turns: usize,
    pub emitted_turns: usize,
    /// Reference change -> nearest emitted change. Answers "did the diarizer
    /// react to this turn at all", so short turns it never labels dominate it.
    pub recall: (f32, f32, f32),
    /// Emitted change -> nearest reference change. Answers "when the diarizer
    /// starts a new paragraph, is there really a new speaker there" -- the
    /// direction the transcript's readability depends on.
    pub precision: (f32, f32, f32),
}

pub fn boundaries(hyp: &[Option<u32>], reference: &Reference) -> Boundaries {
    let n = reference.frames.len().min(hyp.len());
    let truth = changes(
        reference.frames[..n]
            .iter()
            .map(|&m| (m.count_ones() == 1).then(|| m.trailing_zeros() as u8)),
    );
    let found = changes(hyp[..n].iter().copied());
    Boundaries {
        reference_turns: truth.len(),
        emitted_turns: found.len(),
        recall: nearest(&truth, &found),
        precision: nearest(&found, &truth),
    }
}

fn dominant(counts: &HashMap<u32, usize>) -> Option<u32> {
    counts.iter().max_by_key(|(_, c)| **c).map(|(l, _)| *l)
}

/// Frames correct under the best one-to-one label/speaker mapping. Exact:
/// a DP over subsets of reference speakers, of which there are at most 2^8.
fn best_mapping(overlap: &[HashMap<u32, usize>], labels: &[u32], nspk: usize) -> usize {
    let mut dp = vec![0usize; 1 << nspk];
    for label in labels {
        let mut next = dp.clone();
        for (mask, &best) in dp.iter().enumerate() {
            for (spk, counts) in overlap.iter().enumerate() {
                if mask & (1 << spk) != 0 {
                    continue;
                }
                let gain = counts.get(label).copied().unwrap_or(0);
                let m = mask | (1 << spk);
                next[m] = next[m].max(best + gain);
            }
        }
        dp = next;
    }
    dp.into_iter().max().unwrap_or(0)
}

/// Turn per-frame labels into printable `[t0-t1] Speaker N` runs.
pub fn segments(hyp: &[Option<u32>]) -> Vec<(f32, f32, u32)> {
    let mut out: Vec<(f32, f32, u32)> = Vec::new();
    for (f, label) in hyp.iter().enumerate() {
        let Some(label) = *label else { continue };
        let t = f as f32 * FRAME_MS;
        match out.last_mut() {
            Some(last) if last.2 == label && (t - last.1) < FRAME_MS * 1.5 => {
                last.1 = t + FRAME_MS;
            }
            _ => out.push((t, t + FRAME_MS, label)),
        }
    }
    out
}

pub fn read_wav(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("opening {path}"))?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate == 16_000 && spec.channels == 1,
        "{path}: want 16 kHz mono, got {} Hz / {} ch",
        spec.sample_rate,
        spec.channels
    );
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32_768.0))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    })
}
