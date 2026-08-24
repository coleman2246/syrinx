//! Go/no-go spike for embedding-based online speaker diarization.
//!
//! Pipeline: wav -> silero VAD -> sliding windows of voiced audio -> 80-dim
//! log-mel fbank -> speaker embedding -> the spec's online clusterer.
//!
//! Embeddings are the expensive part and depend only on (model, wav, window,
//! hop) — never on the clustering thresholds — so they are computed once and
//! cached to disk. That is what makes a threshold sweep cost seconds instead
//! of an afternoon, and it is why `sweep` is a real sweep rather than a few
//! hand-run configurations.
//!
//! Models and audio live in `$DIARIZE_SPIKE_DIR` (default
//! `~/models/diarize-spike`), never in the repository.

use anyhow::{Context, Result, bail};
use std::collections::VecDeque;
use std::io::{Read, Write};

mod cluster;
mod embed;
mod fbank;
mod reference;
mod vad;

use cluster::{Clusterer, Params};
use embed::Embedder;
use reference::Reference;

/// A voiced window: the audio, and the wall-clock range it was drawn from.
struct Window {
    t0: f32,
    t1: f32,
    samples: Vec<f32>,
}

/// Silence longer than this ends a turn more often than not, so splicing
/// across it would build windows straddling two speakers.
const MAX_GAP_SECONDS: f32 = 0.5;
/// Windows overlapping one 10 ms frame that get a vote on its label.
const VOTE_SLOTS: usize = 8;

fn spike_dir() -> String {
    std::env::var("DIARIZE_SPIKE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/models/diarize-spike",
            std::env::var("HOME").unwrap_or_default()
        )
    })
}

// ---------------------------------------------------------------- windowing

/// Accumulate voiced audio into fixed-length windows with a fixed hop.
fn windows(samples: &[f32], voiced: &[bool], window_s: f32, hop_s: f32) -> Vec<Window> {
    let per_frame = vad::FRAME as f32 / fbank::SAMPLE_RATE;
    let window_frames = (window_s / per_frame).round() as usize;
    let hop_frames = (hop_s / per_frame).round().max(1.0) as usize;
    let max_gap = (MAX_GAP_SECONDS / per_frame) as usize;

    let mut out = Vec::new();
    let mut acc: VecDeque<usize> = VecDeque::new();
    let mut last = None;

    for (i, &v) in voiced.iter().enumerate() {
        if !v {
            continue;
        }
        if last.is_some_and(|p| i - p > max_gap) {
            acc.clear();
        }
        last = Some(i);
        acc.push_back(i);

        if acc.len() >= window_frames {
            let mut audio = Vec::with_capacity(window_frames * vad::FRAME);
            for &f in &acc {
                audio.extend_from_slice(&samples[f * vad::FRAME..(f + 1) * vad::FRAME]);
            }
            out.push(Window {
                t0: acc[0] as f32 * per_frame,
                t1: (acc[acc.len() - 1] + 1) as f32 * per_frame,
                samples: audio,
            });
            for _ in 0..hop_frames.min(acc.len()) {
                acc.pop_front();
            }
        }
    }
    out
}

// -------------------------------------------------------------------- cache

/// Windows and their embeddings, as cached on disk.
struct Embeddings {
    dim: usize,
    times: Vec<(f32, f32)>,
    vectors: Vec<f32>,
}

impl Embeddings {
    fn len(&self) -> usize {
        self.times.len()
    }

    fn get(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }

    fn load(path: &str) -> Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut head = [0u8; 12];
        f.read_exact(&mut head)?;
        if &head[0..4] != b"DZEM" {
            bail!("{path}: not an embedding cache");
        }
        let dim = u32::from_le_bytes(head[4..8].try_into()?) as usize;
        let count = u32::from_le_bytes(head[8..12].try_into()?) as usize;

        let mut rest = Vec::new();
        f.read_to_end(&mut rest)?;
        let floats: Vec<f32> = rest
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect();
        anyhow::ensure!(floats.len() == count * (2 + dim), "{path}: truncated cache");

        let times = floats[..count * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (c[0], c[1]))
            .collect();
        Ok(Self {
            dim,
            times,
            vectors: floats[count * 2..].to_vec(),
        })
    }

    fn store(&self, path: &str) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"DZEM")?;
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        f.write_all(&(self.len() as u32).to_le_bytes())?;
        for (t0, t1) in &self.times {
            f.write_all(&t0.to_le_bytes())?;
            f.write_all(&t1.to_le_bytes())?;
        }
        for v in &self.vectors {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
}

fn cache_path(wav: &str, model: &str, window_s: f32, hop_s: f32) -> String {
    let stem = |p: &str| {
        p.rsplit('/')
            .next()
            .unwrap_or(p)
            .rsplit_once('.')
            .map_or(p, |(s, _)| s)
            .to_string()
    };
    format!(
        "{}/cache/{}.{}.w{:.0}h{:.0}.emb",
        spike_dir(),
        stem(wav),
        stem(model),
        window_s * 1000.0,
        hop_s * 1000.0
    )
}

/// Voiced-frame flags depend only on the wav, so they outlive every sweep.
fn voiced_frames(wav: &str) -> Result<Vec<bool>> {
    let stem = wav.rsplit('/').next().unwrap_or(wav);
    let path = format!("{}/cache/{stem}.vad", spike_dir());
    if let Ok(bytes) = std::fs::read(&path) {
        return Ok(bytes.iter().map(|&b| b != 0).collect());
    }

    let samples = reference::read_wav(wav)?;
    let mut vad = vad::Vad::new(&format!("{}/silero_vad.onnx", spike_dir()))?;
    let started = std::time::Instant::now();
    let voiced = vad.run(&samples)?;
    eprintln!(
        "  vad: {:.1}% voiced over {:.1} min in {:.0}s",
        100.0 * voiced.iter().filter(|v| **v).count() as f32 / voiced.len() as f32,
        samples.len() as f32 / fbank::SAMPLE_RATE / 60.0,
        started.elapsed().as_secs_f32()
    );
    std::fs::create_dir_all(format!("{}/cache", spike_dir()))?;
    std::fs::write(&path, voiced.iter().map(|&v| v as u8).collect::<Vec<_>>())?;
    Ok(voiced)
}

fn embeddings(wav: &str, model: &str, window_s: f32, hop_s: f32) -> Result<Embeddings> {
    let path = cache_path(wav, model, window_s, hop_s);
    if let Ok(cached) = Embeddings::load(&path) {
        return Ok(cached);
    }

    let voiced = voiced_frames(wav)?;
    let samples = reference::read_wav(wav)?;
    let wins = windows(&samples, &voiced, window_s, hop_s);

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut embedder = Embedder::new(model, threads)?;
    let started = std::time::Instant::now();

    let mut vectors = Vec::with_capacity(wins.len() * embedder.dim);
    for batch in wins.chunks(16) {
        let audio: Vec<Vec<f32>> = batch.iter().map(|w| w.samples.clone()).collect();
        for v in embedder.embed_batch(&audio)? {
            vectors.extend_from_slice(&v);
        }
    }
    eprintln!(
        "  embed: {} windows in {:.0}s ({:.1}x real time)",
        wins.len(),
        started.elapsed().as_secs_f32(),
        (samples.len() as f32 / fbank::SAMPLE_RATE) / started.elapsed().as_secs_f32()
    );

    let out = Embeddings {
        dim: embedder.dim,
        times: wins.iter().map(|w| (w.t0, w.t1)).collect(),
        vectors,
    };
    std::fs::create_dir_all(format!("{}/cache", spike_dir()))?;
    out.store(&path)?;
    Ok(out)
}

// ------------------------------------------------------------------ scoring

/// Run the clusterer over cached embeddings and paint labels onto a 10 ms
/// grid, each frame taking the majority vote of the windows covering it.
fn label_frames(emb: &Embeddings, params: Params, frames: usize) -> (Vec<Option<u32>>, Clusterer) {
    let mut votes = vec![0u32; frames * VOTE_SLOTS];
    let mut counts = vec![0u8; frames];
    let mut clusterer = Clusterer::new(params);

    for i in 0..emb.len() {
        let Some(label) = clusterer.push(emb.get(i)) else {
            continue;
        };
        let (t0, t1) = emb.times[i];
        let lo = (t0 / reference::FRAME_MS) as usize;
        let hi = ((t1 / reference::FRAME_MS) as usize).min(frames);
        for f in lo..hi {
            let n = counts[f] as usize;
            if n < VOTE_SLOTS {
                votes[f * VOTE_SLOTS + n] = label;
                counts[f] += 1;
            }
        }
    }

    let labels = (0..frames)
        .map(|f| {
            let slots = &votes[f * VOTE_SLOTS..f * VOTE_SLOTS + counts[f] as usize];
            slots
                .iter()
                .max_by_key(|l| slots.iter().filter(|x| x == l).count())
                .copied()
        })
        .collect();
    (labels, clusterer)
}

fn meeting_of(wav: &str) -> String {
    wav.rsplit('/')
        .next()
        .unwrap_or(wav)
        .split('.')
        .next()
        .unwrap_or("")
        .to_string()
}

fn evaluate(
    wav: &str,
    model: &str,
    window_s: f32,
    hop_s: f32,
    params: Params,
) -> Result<(reference::Metrics, Vec<Option<u32>>, Clusterer)> {
    let emb = embeddings(wav, model, window_s, hop_s)?;
    let reference = Reference::load_ami(&format!("{}/annot", spike_dir()), &meeting_of(wav))?;
    let (hyp, clusterer) = label_frames(&emb, params, reference.frames.len());
    let metrics = reference::score(&hyp, &reference);
    Ok((metrics, hyp, clusterer))
}

// ---------------------------------------------------------------- verify

/// Same-speaker pairs must score far above different-speaker pairs. If they
/// do not, the fbank front-end is wrong and every meeting number downstream
/// is noise.
fn verify(model: &str) -> Result<()> {
    let dir = format!("{}/verify", spike_dir());
    let speakers = ["fangjun", "leijun", "liudehua"];
    let mut embedder = Embedder::new(model, 4)?;

    let mut vecs = Vec::new();
    for spk in speakers {
        for file in [format!("{spk}-sr-1.wav"), format!("{spk}-sr-2.wav")] {
            let samples = reference::read_wav(&format!("{dir}/{file}"))?;
            let v = embedder.embed_batch(&[samples])?.remove(0);
            vecs.push((spk, v));
        }
    }

    let (mut same, mut diff) = (Vec::new(), Vec::new());
    for i in 0..vecs.len() {
        for j in i + 1..vecs.len() {
            let sim = embed::cosine(&vecs[i].1, &vecs[j].1);
            if vecs[i].0 == vecs[j].0 {
                &mut same
            } else {
                &mut diff
            }
            .push(sim);
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    println!(
        "{:<52} same {:.3} (min {:.3})  diff {:.3} (max {:.3})",
        model.rsplit('/').next().unwrap_or(model),
        mean(&same),
        same.iter().cloned().fold(f32::MAX, f32::min),
        mean(&diff),
        diff.iter().cloned().fold(f32::MIN, f32::max),
    );
    Ok(())
}

// -------------------------------------------------------------------- main

struct Args(Vec<String>);

impl Args {
    fn get(&self, name: &str) -> Option<&str> {
        let i = self.0.iter().position(|a| a == name)?;
        self.0.get(i + 1).map(|s| s.as_str())
    }
    fn num(&self, name: &str, default: f32) -> f32 {
        self.get(name)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
    fn list(&self, name: &str, default: &str) -> Vec<String> {
        self.get(name)
            .unwrap_or(default)
            .split(',')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|a| a == name)
    }
}

fn resolve(kind: &str, name: &str) -> String {
    if name.contains('/') {
        name.to_string()
    } else {
        format!("{}/{kind}/{name}", spike_dir())
    }
}

fn describe(t: &ort::value::ValueType) -> String {
    match t {
        ort::value::ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
            ..
        } => {
            let dims: Vec<String> = shape
                .iter()
                .zip(dimension_symbols.iter())
                .map(|(d, s)| {
                    if s.is_empty() {
                        format!("{d}")
                    } else {
                        s.clone()
                    }
                })
                .collect();
            format!("{ty:?}[{}]", dims.join(","))
        }
        other => format!("{other:?}"),
    }
}

fn model_path(name: &str) -> String {
    if name.contains('/') {
        name.to_string()
    } else {
        format!("{}/{name}", spike_dir())
    }
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = argv
        .split_first()
        .context("usage: diarize-spike <probe|verify|run|sweep>")?;
    let args = Args(rest.to_vec());

    match cmd.as_str() {
        // How the feature layouts below were established: no candidate
        // documents its tensor names or axis order.
        "probe" => {
            for name in args.list("--model", "") {
                let session =
                    ort::session::Session::builder()?.commit_from_file(model_path(&name))?;
                println!("== {name}");
                for i in session.inputs() {
                    println!("  in  {:20} {}", i.name(), describe(i.dtype()));
                }
                for o in session.outputs() {
                    println!("  out {:20} {}", o.name(), describe(o.dtype()));
                }
            }
        }

        "verify" => {
            for m in args.list("--model", "") {
                verify(&model_path(&m))?;
            }
        }

        "run" => {
            let wav = resolve("audio", args.get("--wav").context("--wav")?);
            let model = model_path(args.get("--model").context("--model")?);
            let (window_s, hop_s) = (args.num("--window", 1.5), args.num("--hop", 0.0));
            let hop_s = if hop_s == 0.0 { window_s / 2.0 } else { hop_s };
            let params = Params {
                t_assign: args.num("--t-assign", 0.6),
                t_retire: args.num("--t-retire", 0.85),
                alpha: args.num("--alpha", 0.05),
                min_pool: args.num("--min-pool", 4.0) as usize,
            };

            eprintln!(
                "{} / {}",
                meeting_of(&wav),
                model.rsplit('/').next().unwrap()
            );
            let (m, hyp, clusterer) = evaluate(&wav, &model, window_s, hop_s, params)?;
            let (minted, active) = (clusterer.minted(), clusterer.active());
            if args.has("--print") {
                for (t0, t1, label) in reference::segments(&hyp) {
                    println!("[{t0:8.2}-{t1:8.2}] Speaker {label}");
                }
            }
            println!("{params:?} window {window_s} hop {hop_s}");
            println!(
                "  ref {} speakers, hyp {} labels, splits {}, merges {}, \
                 miss {:.1}%, confusion {:.1}%",
                m.ref_speakers,
                m.hyp_speakers,
                m.splits,
                m.merges,
                100.0 * m.miss,
                100.0 * m.confusion
            );
            println!(
                "  {minted} centroids minted, {active} still active ({} retired), \
                 {:.0}s of scorable speech",
                minted as usize - active,
                m.scored_frames as f32 * reference::FRAME_MS
            );
            if let Some((a, b, sim)) = clusterer.crowding().first() {
                println!(
                    "  closest centroid pair: {a} vs {b} at {sim:.3} \
                     (T_assign {:.2}, so {:.2} of margin left)",
                    params.t_assign,
                    params.t_assign - sim
                );
            }
            let reference =
                Reference::load_ami(&format!("{}/annot", spike_dir()), &meeting_of(&wav))?;
            let b = reference::boundaries(&hyp, &reference);
            println!(
                "  turns: {} in reference, {} emitted",
                b.reference_turns, b.emitted_turns
            );
            for (what, (median, p90, within)) in
                [("emitted->real", b.precision), ("real->emitted", b.recall)]
            {
                println!(
                    "    {what}: median {median:.2}s, p90 {p90:.2}s, {:.0}% within 1s",
                    100.0 * within
                );
            }
            let (labels, rows) = reference::overlap_matrix(&hyp, &reference);
            println!(
                "  {:<12}{}",
                "seconds",
                labels.iter().map(|l| format!("{l:>8}")).collect::<String>()
            );
            for (((name, row), thirds), total) in reference
                .names
                .iter()
                .zip(&rows)
                .zip(&m.thirds)
                .zip(&m.per_speaker)
            {
                println!(
                    "  {name:<12}{}   ({total:.0}s ref)  thirds {thirds:?}",
                    row.iter().map(|s| format!("{s:8.0}")).collect::<String>()
                );
            }
        }

        // Silero is the one component with no ground truth of its own, so its
        // probabilities get eyeballed against a stretch of known speech.
        "vadprobe" => {
            let wav = resolve("audio", args.get("--wav").context("--wav")?);
            let samples = reference::read_wav(&wav)?;
            let mut vad = vad::Vad::new(&format!("{}/silero_vad.onnx", spike_dir()))?;
            let until = (args.num("--until", 120.0) * fbank::SAMPLE_RATE) as usize;
            let mut probs = Vec::new();
            for f in 0..(samples.len().min(until) / vad::FRAME) {
                probs.push(vad.prob(&samples[f * vad::FRAME..(f + 1) * vad::FRAME])?);
            }
            let over = |t: f32| {
                100.0 * probs.iter().filter(|p| **p > t).count() as f32 / probs.len() as f32
            };
            println!(
                "{} frames: max {:.3}, >0.5 {:.1}%, >0.1 {:.1}%",
                probs.len(),
                probs.iter().cloned().fold(0.0, f32::max),
                over(0.5),
                over(0.1)
            );
            for (i, p) in probs.iter().enumerate().skip(2400).take(24) {
                println!("  t={:7.2}s p={p:.3}", i as f32 * 0.032);
            }
        }

        // The measurement the go/no-go call actually rests on: how far apart
        // are two windows of the same voice, versus two windows of different
        // voices, on this audio? Clustering cannot beat this ceiling.
        "separability" => {
            let window_s = args.num("--window", 1.5);
            let hop_s = window_s / 2.0;
            for model in args.list(
                "--model",
                "wespeaker_en_voxceleb_resnet34_LM.onnx,\
                 3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx,\
                 nemo_en_titanet_small.onnx",
            ) {
                let model = model_path(&model);
                for wav in args.list("--wav", "ES2002a.Mix-Headset.wav") {
                    let wav = resolve("audio", &wav);
                    let emb = embeddings(&wav, &model, window_s, hop_s)?;
                    let reference =
                        Reference::load_ami(&format!("{}/annot", spike_dir()), &meeting_of(&wav))?;

                    let tagged: Vec<(usize, usize)> = (0..emb.len())
                        .filter_map(|i| {
                            let (t0, t1) = emb.times[i];
                            reference::window_speaker(&reference, t0, t1).map(|s| (i, s))
                        })
                        .collect();

                    let (mut same, mut diff) = (Vec::new(), Vec::new());
                    for a in 0..tagged.len() {
                        for b in a + 1..tagged.len() {
                            let sim = embed::cosine(emb.get(tagged[a].0), emb.get(tagged[b].0));
                            if tagged[a].1 == tagged[b].1 {
                                &mut same
                            } else {
                                &mut diff
                            }
                            .push(sim);
                        }
                    }
                    same.sort_by(f32::total_cmp);
                    diff.sort_by(f32::total_cmp);
                    let pct = |v: &Vec<f32>, p: f32| v[((v.len() - 1) as f32 * p) as usize];

                    println!(
                        "{:<40} {} w={window_s} pure {}/{} windows",
                        model.rsplit('/').next().unwrap().replace(".onnx", ""),
                        meeting_of(&wav),
                        tagged.len(),
                        emb.len()
                    );
                    println!(
                        "  same: p10 {:.3} p50 {:.3} p90 {:.3}   diff: p10 {:.3} p50 {:.3} p90 {:.3}",
                        pct(&same, 0.1),
                        pct(&same, 0.5),
                        pct(&same, 0.9),
                        pct(&diff, 0.1),
                        pct(&diff, 0.5),
                        pct(&diff, 0.9),
                    );
                    let (eer_t, eer) = equal_error(&same, &diff);
                    println!(
                        "  best split at {eer_t:.2} -> {:.1}% of pairs wrong",
                        100.0 * eer
                    );
                }
            }
        }

        // The server embeds one window every hop on one thread, so throughput
        // with a batch of 16 is the wrong number for the spec's CPU budget.
        // This measures the shape the server will actually run in.
        "bench" => {
            let window_s = args.num("--window", 1.5);
            let hop_s = args.num("--hop", window_s / 2.0);
            let samples = reference::read_wav(&resolve(
                "audio",
                args.get("--wav").unwrap_or("ES2002a.Mix-Headset.wav"),
            ))?;
            let window: Vec<f32> = samples
                [16_000 * 100..16_000 * 100 + (window_s * fbank::SAMPLE_RATE) as usize]
                .to_vec();

            let mut vad = vad::Vad::new(&format!("{}/silero_vad.onnx", spike_dir()))?;
            let started = std::time::Instant::now();
            for f in 0..1000 {
                vad.prob(&samples[f * vad::FRAME..(f + 1) * vad::FRAME])?;
            }
            let vad_ms = started.elapsed().as_secs_f32() * 1000.0 / 1000.0;
            // Silero sees every frame, so its cost is per 32 ms of audio.
            println!(
                "silero_vad: {vad_ms:.3} ms/frame -> {:.1}% of one core",
                100.0 * vad_ms / 32.0
            );

            for name in args.list(
                "--model",
                "wespeaker_en_voxceleb_resnet34_LM.onnx,\
                 3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
            ) {
                let mut embedder = Embedder::new(&model_path(&name), 1)?;
                embedder.embed_batch(std::slice::from_ref(&window))?; // warm up
                let started = std::time::Instant::now();
                for _ in 0..20 {
                    embedder.embed_batch(std::slice::from_ref(&window))?;
                }
                let ms = started.elapsed().as_secs_f32() * 1000.0 / 20.0;
                println!(
                    "{:<46} {ms:6.1} ms/window (1 thread) -> {:.1}% of one core \
                     at a {hop_s}s hop",
                    name.replace(".onnx", ""),
                    100.0 * ms / (hop_s * 1000.0)
                );
            }
        }

        // How long the session must hold an ASR chunk before a label covering
        // it exists. Measured against the real window timings rather than
        // derived from the hop, because voiced-audio accumulation stretches
        // wall-clock time by however silent the meeting is.
        "lag" => {
            let window_s = args.num("--window", 1.5);
            let hop_s = args.num("--hop", window_s / 2.0);
            let chunk = args.num("--chunk", 0.56);
            for wav in args.list(
                "--wav",
                "ES2002a.Mix-Headset.wav,IS1000a.Mix-Headset.wav,EN2001a.Mix-Headset.wav",
            ) {
                let wav = resolve("audio", &wav);
                let emb = embeddings(
                    &wav,
                    &model_path("wespeaker_en_voxceleb_resnet34_LM.onnx"),
                    window_s,
                    hop_s,
                )?;
                let reference =
                    Reference::load_ami(&format!("{}/annot", spike_dir()), &meeting_of(&wav))?;

                // Only chunks that carry speech need a label at all.
                let mut delays = Vec::new();
                let chunks = (reference.frames.len() as f32 * reference::FRAME_MS / chunk) as usize;
                for c in 0..chunks {
                    let (lo, hi) = (c as f32 * chunk, (c + 1) as f32 * chunk);
                    let flo = (lo / reference::FRAME_MS) as usize;
                    let fhi = ((hi / reference::FRAME_MS) as usize).min(reference.frames.len());
                    if !reference.frames[flo..fhi].iter().any(|&m| m != 0) {
                        continue;
                    }
                    // First window that both overlaps the chunk and is finished.
                    if let Some(t1) = emb
                        .times
                        .iter()
                        .find(|(t0, t1)| *t1 >= hi && *t0 < hi)
                        .map(|w| w.1)
                    {
                        delays.push(t1 - hi);
                    }
                }
                delays.sort_by(f32::total_cmp);
                let pct = |p: f32| delays[((delays.len() - 1) as f32 * p) as usize];
                println!(
                    "{:<10} {} speech chunks: delay p50 {:.2}s p90 {:.2}s p99 {:.2}s \
                     -> {:.1} chunks at p90",
                    meeting_of(&wav),
                    delays.len(),
                    pct(0.5),
                    pct(0.9),
                    pct(0.99),
                    pct(0.9) / chunk
                );
            }
        }

        "sweep" => sweep(&args)?,

        _ => bail!("unknown command {cmd}"),
    }
    Ok(())
}

/// The threshold minimising total pair errors, and the error rate there.
/// Both inputs must be sorted ascending.
fn equal_error(same: &[f32], diff: &[f32]) -> (f32, f32) {
    let mut best = (0.0, 1.0);
    let mut t = 0.0;
    while t <= 1.0 {
        let false_reject = same.partition_point(|&s| s < t) as f32 / same.len() as f32;
        let false_accept =
            (diff.len() - diff.partition_point(|&s| s < t)) as f32 / diff.len() as f32;
        let total = 0.5 * (false_reject + false_accept);
        if total < best.1 {
            best = (t, total);
        }
        t += 0.01;
    }
    best
}

fn sweep(args: &Args) -> Result<()> {
    let wavs = args.list("--wav", "ES2002a.Mix-Headset.wav,IS1000a.Mix-Headset.wav");
    let models = args.list(
        "--model",
        "wespeaker_en_voxceleb_resnet34_LM.onnx,\
         3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    );
    let windows: Vec<f32> = args
        .list("--windows", "1.0,1.5,2.0")
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let pools: Vec<usize> = args
        .list("--pools", "2,3,4,5,6")
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    // The spec guessed T_assign ~0.6 from the literature. On 1.5 s windows of
    // real meeting audio these models put same-speaker pairs at a median of
    // 0.52 and different-speaker pairs at 0.03, so the useful range is well
    // below that guess — hence a sweep that starts at 0.20.
    let assigns: Vec<f32> = args
        .list("--assigns", "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55")
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let retires: Vec<f32> = args
        .list("--retires", "0.60,0.70,0.80,0.85")
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let alphas: Vec<f32> = args
        .list("--alphas", "0.05")
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    println!(
        "model\tmeeting\twindow\tt_assign\tt_retire\tmin_pool\talpha\tref\thyp\tsplits\tmerges\tmiss%\tconf%"
    );

    for model in &models {
        let model = model_path(model);
        for wav in &wavs {
            let wav = resolve("audio", wav);
            let meeting = meeting_of(&wav);
            let reference = Reference::load_ami(&format!("{}/annot", spike_dir()), &meeting)?;

            for &window_s in &windows {
                let hop_s = window_s / 2.0;
                eprintln!(
                    "== {} {meeting} window {window_s}",
                    model.rsplit('/').next().unwrap()
                );
                let emb = embeddings(&wav, &model, window_s, hop_s)?;

                for &min_pool in &pools {
                    for &t_assign in &assigns {
                        for &t_retire in &retires {
                            for &alpha in &alphas {
                                let params = Params {
                                    t_assign,
                                    t_retire,
                                    alpha,
                                    min_pool,
                                };
                                let (hyp, ..) = label_frames(&emb, params, reference.frames.len());
                                let m = reference::score(&hyp, &reference);
                                println!(
                                    "{}\t{meeting}\t{window_s}\t{t_assign}\t{t_retire}\t{min_pool}\t{alpha}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}",
                                    model.rsplit('/').next().unwrap().replace(".onnx", ""),
                                    m.ref_speakers,
                                    m.hyp_speakers,
                                    m.splits,
                                    m.merges,
                                    100.0 * m.miss,
                                    100.0 * m.confusion
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
