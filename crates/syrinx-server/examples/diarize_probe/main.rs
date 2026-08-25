//! The diarizer's evaluation harness: what the constants in
//! [`syrinx_server::diarize::cluster`] were measured with, and what any change
//! to them has to be re-measured with.
//!
//! ```text
//! ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so cargo run --release \
//!   -p syrinx-server --features diarize --example diarize_probe -- \
//!   run ES2002a.Mix-Headset.wav
//! ```
//!
//! | subcommand | question it answers |
//! |---|---|
//! | `run <wav>` | what does the shipped configuration label this meeting? |
//! | `verify` | is the fbank front end right at all? |
//! | `separability` | how far apart are two voices *before* clustering? |
//! | `sweep` | which constants are best, and how sharp is the cliff? |
//! | `bench` | what does one session cost on one core? |
//! | `lag` | how long must a chunk wait for a label to exist? |
//!
//! **Models and audio live outside the repository**, in `$DIARIZE_SPIKE_DIR`
//! (default `~/models/diarize-spike`) -- the directory the spike left behind,
//! keeping its name so its caches stay valid:
//!
//! ```text
//! silero_vad.onnx                 the VAD, and the only one this drives
//! *.onnx                          embedding models, named by family
//! audio/<meeting>.<variant>.wav   16 kHz mono
//! annot/words/<meeting>.<X>.words.xml   AMI manual annotations, v1.6.2
//! verify/<speaker>-sr-{1,2}.wav   sherpa-onnx's verification pairs
//! cache/                          written here, deletable at any time
//! ```
//!
//! **What is the server's code and what is not.** The VAD, the embedder (and
//! through it the fbank front end), the filename-to-normalisation mapping and
//! the clusterer are `syrinx_server::diarize`'s own, imported and run
//! unmodified -- a harness that measured a copy would be measuring the copy.
//! Two things are re-ported from the spike instead, both because scoring is
//! offline and the server's are not: [`windows`], because
//! `window::WindowAssembler` is calibrated to one geometry and the sweep
//! varies it (they are run side by side and required to agree wherever the
//! geometry does match), and the backwards label painting in [`label_frames`],
//! which gives each 10 ms frame the majority vote of the windows covering it.
//! A live session cannot paint labels onto audio it has already sent, and the
//! published numbers were measured this way, so the harness keeps the spike's
//! batch semantics rather than the session's carry-forward.
//!
//! Two of the spike's subcommands did not come across: `probe`, which dumped a
//! model's input and output tensors, and `vadprobe`, which printed silero's
//! raw probabilities. Both existed to work out what the models wanted; the
//! answers are now in `real::embed::Embedder` and `real::vad::Vad`, and the
//! design doc cites neither.

use anyhow::{Context, Result, bail, ensure};
use std::collections::VecDeque;
use std::io::{Read, Write};

use syrinx_server::diarize::cluster::{
    EMA_ALPHA, MIN_POOL, OnlineClusterer, T_ASSIGN, T_RETIRE, cosine,
};
use syrinx_server::diarize::fbank::SAMPLE_RATE;
use syrinx_server::diarize::real::{Embedder, Vad, norm_for};
use syrinx_server::diarize::window::{FRAME, Framed, WINDOW_SAMPLES, WindowAssembler};

mod reference;
use reference::Reference;

/// The calibrated model, and this harness's default. Kept here as a literal
/// rather than read from `real::diarizer`'s own constant because a harness
/// that compares candidates has to name each one anyway -- `sweep` and
/// `separability` below list the others. If the server ever changes model,
/// this is the line to follow it with; every subcommand prints the model it
/// used, so a mismatch shows up in the output rather than hiding in it.
const CHOSEN: &str = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx";
/// The two models the design doc swept against each other. TitaNet is absent
/// on purpose: `separability` measured it at 3x the pair error of these two
/// and it was dropped before the sweep rather than carried through it.
const SWEPT: &str = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx,\
                     wespeaker_en_voxceleb_resnet34_LM.onnx";
/// Every candidate, including the one that lost, for the two subcommands that
/// exist to compare them.
const CANDIDATES: &str = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx,\
                          wespeaker_en_voxceleb_resnet34_LM.onnx,\
                          nemo_en_titanet_small.onnx";

/// The geometry the server ships (`window::WINDOW_SAMPLES` and half of it).
/// A run at exactly this window and hop *is* a run of the production
/// windowing, and [`windows`] proves as much rather than assuming it.
const PRODUCTION_WINDOW_S: f32 = 1.5;
const PRODUCTION_HOP_S: f32 = 0.75;

/// Silence longer than this ends a turn more often than not, so splicing
/// across it would build windows straddling two speakers. The seconds behind
/// `window::MAX_GAP_FRAMES`, which truncates it the same way.
const MAX_GAP_SECONDS: f32 = 0.5;
/// Windows overlapping one 10 ms frame that get a vote on its label.
const VOTE_SLOTS: usize = 8;

fn probe_dir() -> String {
    std::env::var("DIARIZE_SPIKE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/models/diarize-spike",
            std::env::var("HOME").unwrap_or_default()
        )
    })
}

// ---------------------------------------------------------------- windowing

/// A voiced window: the audio, and the wall-clock range it was drawn from.
struct Window {
    t0: f32,
    t1: f32,
    samples: Vec<f32>,
}

/// Accumulate voiced audio into fixed-length windows with a fixed hop.
///
/// The spike's `windows()`, re-ported rather than delegated to
/// `window::WindowAssembler` for two reasons that are both consequences of
/// scoring a whole file at once: the assembler is fixed at the production
/// geometry and `sweep` varies it, and the assembler answers with audio alone,
/// while painting labels back over a meeting needs to know which seconds each
/// window was drawn from. What it must *not* be is a second algorithm, so at
/// the production geometry the shipped assembler is fed the same frames and
/// required to produce the same windows.
fn windows(samples: &[f32], voiced: &[bool], window_s: f32, hop_s: f32) -> Result<Vec<Window>> {
    let per_frame = FRAME as f32 / SAMPLE_RATE;
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
            let mut audio = Vec::with_capacity(window_frames * FRAME);
            for &f in &acc {
                audio.extend_from_slice(&samples[f * FRAME..(f + 1) * FRAME]);
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

    if window_s == PRODUCTION_WINDOW_S && hop_s == PRODUCTION_HOP_S {
        ensure!(
            window_frames * FRAME == WINDOW_SAMPLES,
            "{window_s}s is {window_frames} frames here but {} in window.rs; the \
             harness and the server disagree about the production geometry",
            WINDOW_SAMPLES / FRAME
        );
        agrees_with_the_shipped_assembler(samples, voiced, &out)?;
    }
    Ok(out)
}

/// Feed `window::WindowAssembler` the same voiced frames and require it to
/// produce the same windows, sample for sample.
///
/// This is the check that keeps the numbers below attributable to the server:
/// the harness carries its own windower, so nothing but this would notice the
/// day one of them changed. Fed one ASR chunk's worth of frames at a time,
/// which is the shape the assembler actually runs in, and compared window by
/// window so neither side has to be held in memory whole.
fn agrees_with_the_shipped_assembler(
    samples: &[f32],
    voiced: &[bool],
    mine: &[Window],
) -> Result<()> {
    /// 8960 samples, an ASR chunk, is 17.5 frames.
    const CHUNK_FRAMES: usize = 17;

    let mut assembler = WindowAssembler::default();
    let frames = voiced.len().min(samples.len() / FRAME);
    let mut matched = 0usize;

    for first_frame in (0..frames).step_by(CHUNK_FRAMES) {
        let n = CHUNK_FRAMES.min(frames - first_frame);
        let framed = Framed {
            first_frame,
            samples: samples[first_frame * FRAME..(first_frame + n) * FRAME].to_vec(),
        };
        for window in assembler.push(&framed, &voiced[first_frame..first_frame + n]) {
            let expected = mine.get(matched).with_context(|| {
                format!(
                    "the shipped assembler produced window {matched}, the harness \
                     produced only {}",
                    mine.len()
                )
            })?;
            ensure!(
                window == expected.samples,
                "window {matched} ({:.2}-{:.2}s) differs between the harness and \
                 window::WindowAssembler",
                expected.t0,
                expected.t1
            );
            matched += 1;
        }
    }
    ensure!(
        matched == mine.len(),
        "the harness produced {} windows, the shipped assembler {matched}",
        mine.len()
    );
    // Said out loud, because a check nobody sees run is a check nobody trusts.
    eprintln!("  windowing: {matched} windows agree with window::WindowAssembler");
    Ok(())
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
        let tmp = format!("{path}.partial");
        let mut f = std::fs::File::create(&tmp)?;
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
        drop(f);
        Ok(std::fs::rename(&tmp, path)?)
    }
}

/// **Bump this whenever `diarize::{fbank, window}`, `diarize::real::{vad,
/// embed}` or [`windows`] changes.** The cache keys name the wav, model and
/// window geometry, but nothing about the code that produced the contents, so
/// without a version in the filename a pipeline fix silently reads back
/// results from the pipeline it replaced. That is not hypothetical: fixing
/// silero's context window during the spike invalidated every `.vad` file
/// already written, and nothing but remembering to delete them by hand stood
/// in the way.
///
/// 3 rather than the spike's 2 because the pipeline is now the server's own
/// code rather than the spike's copy of it. The two were expected to agree --
/// and did, to the digit, on the design doc's ES2002a row -- but expecting is
/// what the version number exists not to have to do.
const PIPELINE_VERSION: u32 = 3;

fn cache_path(wav: &str, model: &str, window_s: f32, hop_s: f32) -> String {
    format!(
        "{}/cache/{}.{}.w{:.0}h{:.0}.v{PIPELINE_VERSION}.emb",
        probe_dir(),
        stem(wav),
        stem(model),
        window_s * 1000.0,
        hop_s * 1000.0
    )
}

/// Voiced-frame flags depend only on the wav and the VAD, so they outlive
/// every sweep but not a change to `real::vad`.
fn voiced_frames(wav: &str) -> Result<Vec<bool>> {
    let name = basename(wav);
    let path = format!("{}/cache/{name}.v{PIPELINE_VERSION}.vad", probe_dir());
    if let Ok(bytes) = std::fs::read(&path) {
        return Ok(bytes.iter().map(|&b| b != 0).collect());
    }

    let samples = reference::read_wav(wav)?;
    let mut vad = Vad::new(&format!("{}/silero_vad.onnx", probe_dir()))?;
    let started = std::time::Instant::now();
    let voiced = vad.run(&samples)?;
    eprintln!(
        "  vad: {:.1}% voiced over {:.1} min in {:.0}s",
        100.0 * voiced.iter().filter(|v| **v).count() as f32 / voiced.len() as f32,
        samples.len() as f32 / SAMPLE_RATE / 60.0,
        started.elapsed().as_secs_f32()
    );
    // Write-then-rename: a run killed mid-write would otherwise leave a short
    // file that loads without complaint and silently truncates the meeting.
    // Unlike the embedding cache, these bytes carry no length to check against.
    std::fs::create_dir_all(format!("{}/cache", probe_dir()))?;
    let tmp = format!("{path}.partial");
    std::fs::write(&tmp, voiced.iter().map(|&v| v as u8).collect::<Vec<_>>())?;
    std::fs::rename(&tmp, &path)?;
    Ok(voiced)
}

/// Embeddings depend only on (model, wav, window, hop) -- never on the
/// clustering thresholds -- so they are computed once and cached. That is what
/// makes a 2640-configuration sweep cost seconds instead of an afternoon.
fn embeddings(wav: &str, model: &str, window_s: f32, hop_s: f32) -> Result<Embeddings> {
    let path = cache_path(wav, model, window_s, hop_s);
    if let Ok(cached) = Embeddings::load(&path) {
        return Ok(cached);
    }

    let voiced = voiced_frames(wav)?;
    let samples = reference::read_wav(wav)?;
    let wins = windows(&samples, &voiced, window_s, hop_s)?;

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut embedder = embedder(model, threads)?;
    let started = std::time::Instant::now();

    let mut vectors = Vec::with_capacity(wins.len() * embedder.dim());
    // Batched, unlike the server, because throughput is all that matters here;
    // `bench` measures the one-window-at-a-time shape a session runs in.
    for batch in wins.chunks(16) {
        let audio: Vec<&[f32]> = batch.iter().map(|w| w.samples.as_slice()).collect();
        for v in embedder.embed_batch(&audio)? {
            vectors.extend_from_slice(&v);
        }
    }
    eprintln!(
        "  embed: {} windows in {:.0}s ({:.1}x real time)",
        wins.len(),
        started.elapsed().as_secs_f32(),
        (samples.len() as f32 / SAMPLE_RATE) / started.elapsed().as_secs_f32()
    );

    let out = Embeddings {
        dim: embedder.dim(),
        times: wins.iter().map(|w| (w.t0, w.t1)).collect(),
        vectors,
    };
    std::fs::create_dir_all(format!("{}/cache", probe_dir()))?;
    out.store(&path)?;
    Ok(out)
}

/// An embedder for a model named on the command line, normalised the way the
/// server would normalise it. `real::norm_for` is the tree's one filename-to-
/// recipe mapping and refuses names it does not recognise, which is the point:
/// the wrong normalisation produces embeddings that look plausible and
/// separate nobody, so an unknown model has to stop the run rather than
/// quietly produce a number for the design doc.
fn embedder(model: &str, threads: usize) -> Result<Embedder> {
    let norm = norm_for(basename(model)).with_context(|| {
        format!(
            "{}: no known feature normalisation. The name has to identify the \
             family (3dspeaker/eres2net, wespeaker, nemo/titanet).",
            basename(model)
        )
    })?;
    Embedder::new(model, threads, norm)
}

// ------------------------------------------------------------------ scoring

/// Run the clusterer over cached embeddings and paint labels onto a 10 ms
/// grid, each frame taking the majority vote of the windows covering it.
///
/// Backwards painting is the one deliberate difference from the session, which
/// has no past to paint and carries the last label forward instead. It is how
/// every published number was measured, so it stays.
fn label_frames(
    emb: &Embeddings,
    params: Params,
    frames: usize,
) -> (Vec<Option<u32>>, OnlineClusterer) {
    let mut votes = vec![0u32; frames * VOTE_SLOTS];
    let mut counts = vec![0u8; frames];
    let mut clusterer = params.clusterer();

    for i in 0..emb.len() {
        let Some(label) = clusterer.observe(emb.get(i)) else {
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

/// The clusterer's four thresholds, as the sweep varies them.
///
/// Deliberately not a type in `diarize::cluster`: the server has no
/// configuration surface here, and the one place these four travel together is
/// this harness. [`Default`] reads the shipped constants, so a no-flags `run`
/// is the shipped configuration and there is no second copy of the numbers.
#[derive(Clone, Copy, Debug)]
struct Params {
    t_assign: f32,
    t_retire: f32,
    alpha: f32,
    min_pool: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            t_assign: T_ASSIGN,
            t_retire: T_RETIRE,
            alpha: EMA_ALPHA,
            min_pool: MIN_POOL,
        }
    }
}

impl Params {
    fn clusterer(&self) -> OnlineClusterer {
        OnlineClusterer::with_params(self.t_assign, self.t_retire, self.alpha, self.min_pool)
    }
}

// -------------------------------------------------------------------- paths

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// A filename without its directory or its extension, for cache keys and
/// table columns.
fn stem(path: &str) -> String {
    basename(path)
        .rsplit_once('.')
        .map_or(path, |(s, _)| s)
        .to_string()
}

/// The AMI meeting a recording belongs to: `ES2002a.Mix-Headset.wav` and
/// `ES2002a.Opus24k.wav` are both scored against `ES2002a`'s annotations.
fn meeting_of(wav: &str) -> String {
    basename(wav).split('.').next().unwrap_or("").to_string()
}

/// A bare name is looked up in the probe directory; anything with a `/` in it
/// is taken as written.
fn resolve(kind: &str, name: &str) -> String {
    if name.contains('/') {
        name.to_string()
    } else {
        format!("{}/{kind}/{name}", probe_dir())
    }
}

fn model_path(name: &str) -> String {
    if name.contains('/') {
        name.to_string()
    } else {
        format!("{}/{name}", probe_dir())
    }
}

// --------------------------------------------------------------------- args

struct Args(Vec<String>);

impl Args {
    fn get(&self, name: &str) -> Option<&str> {
        let i = self.0.iter().position(|a| a == name)?;
        self.0.get(i + 1).map(|s| s.as_str())
    }
    /// The subcommand's own argument, when it is not a flag: `run meeting.wav`.
    /// Every flag starts with `--`, so a leading bare word is unambiguous.
    fn positional(&self) -> Option<&str> {
        self.0
            .first()
            .filter(|a| !a.starts_with('-'))
            .map(String::as_str)
    }
    /// A value that will not parse is fatal, never a silent fall back to the
    /// default. This harness exists to attribute numbers to constants, so
    /// quietly sweeping `0.45` because `--t-assign 0,45` was unreadable would
    /// invalidate the output while looking like a clean run.
    fn num(&self, name: &str, default: f32) -> Result<f32> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .with_context(|| format!("{name}: expected a number, got {v:?}")),
        }
    }
    fn list(&self, name: &str, default: &str) -> Vec<String> {
        self.get(name)
            .unwrap_or(default)
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
    /// Comma-separated numbers, where one bad entry fails the run rather than
    /// silently shortening the grid.
    fn nums<T: std::str::FromStr>(&self, name: &str, default: &str) -> Result<Vec<T>> {
        self.list(name, default)
            .iter()
            .map(|s| {
                s.parse::<T>()
                    .map_err(|_| anyhow::anyhow!("{name}: expected a number, got {s:?}"))
            })
            .collect()
    }
    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|a| a == name)
    }
    /// The four clustering thresholds, each defaulting to the shipped
    /// constant. Overriding one leaves the other three at what the server
    /// runs, which is what makes `--t-assign 0.6` answer "what would raising
    /// only this cost?".
    fn params(&self) -> Result<Params> {
        let shipped = Params::default();
        Ok(Params {
            t_assign: self.num("--t-assign", shipped.t_assign)?,
            t_retire: self.num("--t-retire", shipped.t_retire)?,
            alpha: self.num("--alpha", shipped.alpha)?,
            min_pool: self.num("--min-pool", shipped.min_pool as f32)? as usize,
        })
    }
    /// Window and hop in seconds; a hop of zero means half the window.
    fn geometry(&self) -> Result<(f32, f32)> {
        let window_s = self.num("--window", PRODUCTION_WINDOW_S)?;
        let hop_s = self.num("--hop", 0.0)?;
        Ok((window_s, if hop_s == 0.0 { window_s / 2.0 } else { hop_s }))
    }
}

// --------------------------------------------------------------------- main

const USAGE: &str = "usage: diarize_probe <run|verify|separability|sweep|bench|lag> [args]";

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = argv.split_first().context(USAGE)?;
    let args = Args(rest.to_vec());

    match cmd.as_str() {
        "run" => run(&args),
        "verify" => verify(&args),
        "separability" => separability(&args),
        "sweep" => sweep(&args),
        "bench" => bench(&args),
        "lag" => lag(&args),
        _ => bail!("unknown command {cmd}\n{USAGE}"),
    }
}

/// Label one meeting with one configuration, and say how well it went.
///
/// With annotations for the recording present, this is the row the design
/// doc's results table holds; without them it is still a working diarization
/// of the file, which is what makes the harness usable on a recording nobody
/// has annotated.
fn run(args: &Args) -> Result<()> {
    let wav = resolve(
        "audio",
        args.positional()
            .or_else(|| args.get("--wav"))
            .context("usage: run <wav> [--model M] [--window S] [--quiet]")?,
    );
    let model = model_path(args.get("--model").unwrap_or(CHOSEN));
    let (window_s, hop_s) = args.geometry()?;
    let params = args.params()?;
    let meeting = meeting_of(&wav);

    eprintln!("{meeting} / {}", basename(&model));
    let emb = embeddings(&wav, &model, window_s, hop_s)?;
    let reference = match Reference::load_ami(&format!("{}/annot", probe_dir()), &meeting) {
        Ok(reference) => Some(reference),
        Err(e) => {
            eprintln!("  no reference ({e}); labelling only, nothing to score against");
            None
        }
    };

    // With a reference, the grid is the reference's, so the scoring below
    // compares like with like; without one it covers the last window.
    let frames = match &reference {
        Some(reference) => reference.frames.len(),
        None => emb
            .times
            .last()
            .map_or(0, |(_, t1)| (t1 / reference::FRAME_MS) as usize + 1),
    };
    let (hyp, clusterer) = label_frames(&emb, params, frames);

    if !args.has("--quiet") {
        for (t0, t1, label) in reference::segments(&hyp) {
            println!("[{t0:8.2}-{t1:8.2}] Speaker {label}");
        }
    }
    println!("{params:?} window {window_s} hop {hop_s}");

    let (minted, active) = (clusterer.minted(), clusterer.active());
    let Some(reference) = reference else {
        println!("  {minted} centroids minted, {active} still active");
        return Ok(());
    };

    let m = reference::score(&hyp, &reference);
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
    Ok(())
}

/// Same-speaker pairs must score far above different-speaker pairs. If they
/// do not, the fbank front end is wrong and every meeting number downstream
/// is noise -- which is a failure with no other symptom, since a wrong front
/// end still produces plausible-looking embeddings.
fn verify(args: &Args) -> Result<()> {
    let dir = format!("{}/verify", probe_dir());
    let speakers = ["fangjun", "leijun", "liudehua"];

    for model in args.list("--model", CANDIDATES) {
        let model = model_path(&model);
        let mut embedder = embedder(&model, 4)?;

        let mut vecs = Vec::new();
        for spk in speakers {
            for file in [format!("{spk}-sr-1.wav"), format!("{spk}-sr-2.wav")] {
                let samples = reference::read_wav(&format!("{dir}/{file}"))?;
                vecs.push((spk, embedder.embed(&samples)?));
            }
        }

        let (mut same, mut diff) = (Vec::new(), Vec::new());
        for i in 0..vecs.len() {
            for j in i + 1..vecs.len() {
                let sim = cosine(&vecs[i].1, &vecs[j].1);
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
            basename(&model),
            mean(&same),
            same.iter().cloned().fold(f32::MAX, f32::min),
            mean(&diff),
            diff.iter().cloned().fold(f32::MIN, f32::max),
        );
    }
    Ok(())
}

/// The measurement the go/no-go call rested on: how far apart are two windows
/// of the same voice, versus two windows of different voices, on real meeting
/// audio? Clustering cannot beat this ceiling, and `T_assign` has to sit
/// inside the gap it measures.
fn separability(args: &Args) -> Result<()> {
    let (window_s, hop_s) = args.geometry()?;
    for model in args.list("--model", CANDIDATES) {
        let model = model_path(&model);
        for wav in args.list("--wav", "ES2002a.Mix-Headset.wav") {
            let wav = resolve("audio", &wav);
            let emb = embeddings(&wav, &model, window_s, hop_s)?;
            let reference =
                Reference::load_ami(&format!("{}/annot", probe_dir()), &meeting_of(&wav))?;

            let tagged: Vec<(usize, usize)> = (0..emb.len())
                .filter_map(|i| {
                    let (t0, t1) = emb.times[i];
                    reference::window_speaker(&reference, t0, t1).map(|s| (i, s))
                })
                .collect();

            let (mut same, mut diff) = (Vec::new(), Vec::new());
            for a in 0..tagged.len() {
                for b in a + 1..tagged.len() {
                    let sim = cosine(emb.get(tagged[a].0), emb.get(tagged[b].0));
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
                stem(&model),
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
            let (threshold, wrong) = best_split(&same, &diff);
            println!(
                "  best split at {threshold:.2} -> {:.1}% of pairs wrong",
                100.0 * wrong
            );
        }
    }
    Ok(())
}

/// The threshold minimising total pair errors, and the error rate there.
/// Both inputs must be sorted ascending.
fn best_split(same: &[f32], diff: &[f32]) -> (f32, f32) {
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

/// Every configuration in the grid, one row each, cheap because the
/// embeddings are cached and the clustering is arithmetic.
///
/// The committed defaults are the grid the design doc's numbers came from:
/// 2 models x 2 meetings x 3 windows x 5 pools x 11 assign thresholds x
/// 4 retire thresholds = 2640 rows.
fn sweep(args: &Args) -> Result<()> {
    let wavs = args.list("--wav", "ES2002a.Mix-Headset.wav,IS1000a.Mix-Headset.wav");
    let models = args.list("--model", SWEPT);
    let windows: Vec<f32> = args.nums("--windows", "1.0,1.5,2.0")?;
    let pools: Vec<usize> = args.nums("--pools", "2,3,4,5,6")?;
    // The design asked for 0.4-0.7, which this covers. It also runs down to
    // 0.20 because that is where the answer turned out to live: on 1.5 s
    // windows of real meeting audio these models put same-speaker pairs at a
    // median of 0.52 and different-speaker pairs at 0.03, so the design's
    // ~0.6 guess sits above most true matches rather than between the two
    // distributions.
    let assigns: Vec<f32> = args.nums(
        "--assigns",
        "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
    )?;
    let retires: Vec<f32> = args.nums("--retires", "0.60,0.70,0.80,0.85")?;
    let alphas: Vec<f32> = args.nums("--alphas", "0.05")?;

    println!(
        "model\tmeeting\twindow\tt_assign\tt_retire\tmin_pool\talpha\tref\thyp\tsplits\tmerges\tmiss%\tconf%\tminted\tactive\tcrowd"
    );

    for model in &models {
        let model = model_path(model);
        for wav in &wavs {
            let wav = resolve("audio", wav);
            let meeting = meeting_of(&wav);
            let reference = Reference::load_ami(&format!("{}/annot", probe_dir()), &meeting)?;

            for &window_s in &windows {
                let hop_s = window_s / 2.0;
                eprintln!("== {} {meeting} window {window_s}", basename(&model));
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
                                let (hyp, clusterer) =
                                    label_frames(&emb, params, reference.frames.len());
                                let m = reference::score(&hyp, &reference);
                                // The closest surviving pair: how much room a
                                // meeting with more people would have had.
                                let crowd = clusterer
                                    .crowding()
                                    .first()
                                    .map_or(f32::NAN, |&(_, _, sim)| sim);
                                println!(
                                    "{}\t{meeting}\t{window_s}\t{t_assign}\t{t_retire}\t{min_pool}\t{alpha}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{}\t{}\t{crowd:.3}",
                                    stem(&model),
                                    m.ref_speakers,
                                    m.hyp_speakers,
                                    m.splits,
                                    m.merges,
                                    100.0 * m.miss,
                                    100.0 * m.confusion,
                                    clusterer.minted(),
                                    clusterer.active(),
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

/// What one session costs on one core.
///
/// The server embeds one window every hop on one thread, so throughput with a
/// batch of 16 is the wrong number for the CPU budget. This measures the shape
/// the server actually runs in: `Vad::prob` per frame, `Embedder::embed` per
/// window, one thread each.
fn bench(args: &Args) -> Result<()> {
    let (window_s, hop_s) = args.geometry()?;
    let wav = resolve(
        "audio",
        args.positional()
            .or_else(|| args.get("--wav"))
            .unwrap_or("ES2002a.Mix-Headset.wav"),
    );
    let samples = reference::read_wav(&wav)?;

    // A window from well inside the meeting, so it is speech rather than
    // whatever the room sounded like before anyone arrived.
    const SKIP_SECONDS: usize = 100;
    const VAD_FRAMES: usize = 1000;
    let start = SAMPLE_RATE as usize * SKIP_SECONDS;
    let len = (window_s * SAMPLE_RATE) as usize;
    ensure!(
        samples.len() >= start + len && samples.len() >= VAD_FRAMES * FRAME,
        "{wav}: only {:.0}s of audio; bench needs more than {SKIP_SECONDS}s",
        samples.len() as f32 / SAMPLE_RATE
    );
    let window = &samples[start..start + len];

    let mut vad = Vad::new(&format!("{}/silero_vad.onnx", probe_dir()))?;
    let started = std::time::Instant::now();
    for f in 0..VAD_FRAMES {
        vad.prob(&samples[f * FRAME..(f + 1) * FRAME])?;
    }
    let vad_ms = started.elapsed().as_secs_f32() * 1000.0 / VAD_FRAMES as f32;
    // Silero sees every frame, so its cost is per 32 ms of audio.
    let frame_ms = 1000.0 * FRAME as f32 / SAMPLE_RATE;
    println!(
        "silero_vad: {vad_ms:.3} ms/frame -> {:.1}% of one core",
        100.0 * vad_ms / frame_ms
    );

    // A mean over 20 windows, which is what the design doc's per-window
    // figures are -- and it is a noisy estimator on a machine doing anything
    // else: successive runs of this same loop have landed anywhere from 57 to
    // 80 ms for ERes2Net on one idle-ish laptop. Re-measure on a quiet machine
    // and raise `--runs` before believing a difference of less than about a
    // third.
    let runs = args.num("--runs", 20.0)? as usize;
    for name in args.list("--model", SWEPT) {
        let mut embedder = embedder(&model_path(&name), 1)?;
        embedder.embed(window)?; // warm up
        let started = std::time::Instant::now();
        for _ in 0..runs {
            embedder.embed(window)?;
        }
        let ms = started.elapsed().as_secs_f32() * 1000.0 / runs as f32;
        println!(
            "{:<46} {ms:6.1} ms/window (1 thread) -> {:.1}% of one core \
             at a {hop_s}s hop",
            stem(&name),
            100.0 * ms / (hop_s * 1000.0)
        );
    }
    Ok(())
}

/// How long a session must hold an ASR chunk before a label covering it
/// exists -- the measurement behind `session::LAG_CHUNKS`.
///
/// Measured against the real window timings rather than derived from the hop,
/// because a window is a window's worth of *voiced* audio and accumulating it
/// stretches wall-clock time by however silent the meeting is.
fn lag(args: &Args) -> Result<()> {
    let (window_s, hop_s) = args.geometry()?;
    let chunk = args.num("--chunk", 0.56)?;
    // Window timings do not depend on the embedding model; naming one here
    // only picks which cache file the times are read out of.
    let model = model_path(args.get("--model").unwrap_or(CHOSEN));

    for wav in args.list(
        "--wav",
        "ES2002a.Mix-Headset.wav,IS1000a.Mix-Headset.wav,EN2001a.Mix-Headset.wav",
    ) {
        let wav = resolve("audio", &wav);
        let emb = embeddings(&wav, &model, window_s, hop_s)?;
        let reference = Reference::load_ami(&format!("{}/annot", probe_dir()), &meeting_of(&wav))?;

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
    Ok(())
}
