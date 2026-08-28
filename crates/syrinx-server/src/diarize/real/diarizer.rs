//! The whole pipeline behind one trait: VAD, voiced windows, embeddings,
//! clustering.
//!
//! Everything here is assembly. The parts that decide anything live
//! elsewhere and are tested without a model: the windowing in
//! [`crate::diarize::window`], the labelling policy in
//! [`crate::diarize::cluster`], the front end in [`crate::diarize::fbank`].

use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use super::{Embedder, Norm, Vad};
use crate::diarize::cluster::{OnlineClusterer, cosine};
use crate::diarize::window::{Cut, FRAME, Framer, WINDOW_SAMPLES, WindowAssembler};
use crate::diarize::{Attribution, DiarizeTuning, Diarizer, DiarizerFactory, Relabel};

/// The VAD, by name, in `diarize_model_dir`. Fixed rather than sniffed:
/// silero is the only VAD this code knows how to drive -- 512-sample frames,
/// a `state` tensor, an `sr` input -- so a file that is not it does not work
/// whatever it is called.
const VAD_FILE: &str = "silero_vad.onnx";
/// The embedding model the design settled on, by name, in the same directory.
/// Unlike the VAD this one is genuinely swappable (see
/// [`Models::resolve`]), and the design records WeSpeaker ResNet34-LM as the
/// near-equal alternative.
const EMBED_FILE: &str = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx";

/// ONNX intra-op threads for the embedder. One, because there is one diarizer
/// per session and the spike measured 45 ms per window single-threaded, ~6% of
/// a core; letting each of `max_sessions` embedders spin up its own thread
/// pool would trade that for contention with the ASR, which is the work that
/// actually has to keep up. [`Vad::new`] makes the same call for itself, on
/// stronger grounds: silero is far too small for intra-op parallelism to pay
/// for its own synchronisation.
const EMBED_THREADS: usize = 1;

/// Where the models are and how to read them. Cheap to clone -- two paths and
/// a flag -- which is what a per-session diarizer gets handed.
#[derive(Clone, Debug)]
struct Models {
    vad: PathBuf,
    embed: PathBuf,
    /// Not inferable from the ONNX file: see [`Embedder::new`].
    norm: Norm,
}

/// Spawns a [`RealDiarizer`] per session over one checked model directory.
///
/// The models themselves are *not* held loaded here, which is the one place
/// this departs from `asr::parakeet::ParakeetBackend`. That backend shares one
/// `NemotronHandle` across every session because the alternative is 2.5 GB of
/// VRAM per client; here the alternative is ~28 MB of host RAM per session,
/// while sharing would mean putting an `ort::Session` behind a mutex --
/// `Session::run` takes `&mut self` precisely because ONNX Runtime's
/// concurrency is not to be trusted, and ort's own guidance is one session per
/// thread. Four sessions of unshared models cost ~112 MB and no lock.
pub struct RealDiarizerFactory {
    models: Models,
    /// The server's diarization settings, handed to every session.
    tuning: DiarizeTuning,
}

impl RealDiarizerFactory {
    /// Resolve the models in `dir` and prove they work, or explain why not.
    ///
    /// Blocking, and slow enough to matter (both graphs are committed and both
    /// are run), so this belongs at startup and nowhere near a request.
    ///
    /// `tuning` is carried rather than read from the constants so a deployment
    /// can trade pickup speed against splitting one voice in two, and caution
    /// against coverage; each key's default is the constant that documents it,
    /// and each range is checked at config load.
    pub fn load(dir: &Path, tuning: DiarizeTuning) -> Result<Self> {
        let models = Models::resolve(dir)?;
        let dim = models.self_check()?;
        info!(
            vad = %models.vad.display(),
            embed = %models.embed.display(),
            norm = ?models.norm,
            dim,
            min_pool = tuning.min_pool,
            margin = tuning.margin,
            change_threshold = tuning.change_threshold,
            "speaker labelling available"
        );
        Ok(Self { models, tuning })
    }
}

impl DiarizerFactory for RealDiarizerFactory {
    fn diarizer(&self) -> Box<dyn Diarizer> {
        Box::new(RealDiarizer::new(self.models.clone(), self.tuning))
    }
}

impl Models {
    /// Everything that touches the filesystem, and nothing that decides
    /// anything: the VAD is checked by name, the directory is listed, and
    /// [`pick_embed`] makes the choice.
    fn resolve(dir: &Path) -> Result<Self> {
        let vad = dir.join(VAD_FILE);
        ensure!(
            vad.is_file(),
            "{}: no {VAD_FILE}. The directory must hold silero VAD under exactly \
             that name -- v6.2.1 is what the README fetches, on the 512+64-sample \
             interface unchanged since v5 -- alongside a speaker-embedding model \
             (ideally {EMBED_FILE}).",
            dir.display()
        );

        let entries = std::fs::read_dir(dir).with_context(|| {
            format!("reading the diarization model directory {}", dir.display())
        })?;
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.with_context(|| format!("listing {}", dir.display()))?;
            names.push(entry.file_name().to_string_lossy().to_string());
        }

        let (name, norm) = pick_embed(&names)
            .with_context(|| format!("choosing an embedding model in {}", dir.display()))?;
        if name != EMBED_FILE {
            warn!(
                "{}: using {name} for speaker embeddings ({norm:?} normalisation, from \
                 its name) -- the calibrated model is {EMBED_FILE}",
                dir.display()
            );
        }
        Ok(Self {
            vad,
            embed: dir.join(name),
            norm,
        })
    }

    /// Load both models and run them, returning the embedding width.
    ///
    /// The `verify_gpu` pattern: a model file that loads is not a model file
    /// that works, and the failures worth catching here are silent ones. It
    /// pushes a second of digital silence and a second of a synthesised vowel
    /// through the VAD and requires it to disagree about them, then embeds one
    /// window of that vowel.
    ///
    /// **What this catches:** a missing or corrupt file; a VAD whose graph is
    /// not silero's (wrong input or output names); the silero context bug the
    /// spike hit, where feeding 512 samples instead of 576 returns near-zero
    /// speech probability for *everything* -- the vowel probe fails and
    /// nothing else would have noticed; an embedding model whose output is not
    /// a 2-D embedding, produces non-finite values, or produces an all-zero
    /// vector.
    ///
    /// **What it cannot catch:** the wrong weights. A model trained on another
    /// language, at another sample rate, or read with the wrong `Norm` loads,
    /// runs, and returns a plausible vector -- and a synthetic vowel cannot
    /// tell the difference, because these models hear all synthetic vowels as
    /// roughly one voice (measured: ERes2Net puts two synthesised speakers at
    /// cosine 0.81, against 0.97 for one; WeSpeaker at 0.91 against 0.93 --
    /// too close to assert on). Separability needs real voices, which is what
    /// the `diarize_probe` example's `verify` and `separability` are for.
    fn self_check(&self) -> Result<usize> {
        let path = |p: &Path| p.to_string_lossy().to_string();
        let mut vad = Vad::new(&path(&self.vad))?;

        let silent = vad
            .run(&vec![0.0f32; 16_000])
            .context("running the VAD over a second of silence")?;
        let voiced_in_silence = silent.iter().filter(|v| **v).count();
        ensure!(
            voiced_in_silence == 0,
            "{}: the VAD called {voiced_in_silence} of {} silent frames speech",
            path(&self.vad),
            silent.len()
        );

        let speech = speech_like(16_000);
        let voiced = vad
            .run(&speech)
            .context("running the VAD over synthesised speech")?;
        let voiced_frames = voiced.iter().filter(|v| **v).count();
        // Measured at 18 of 31 frames on this signal; a third is the margin
        // against a model revision hearing it slightly differently, while
        // still failing loudly for a VAD that hears nothing at all.
        ensure!(
            voiced_frames * 3 >= voiced.len(),
            "{}: the VAD found speech in only {voiced_frames} of {} frames of a \
             synthesised vowel. A VAD that calls everything silence labels nobody.",
            path(&self.vad),
            voiced.len()
        );

        let mut embedder = Embedder::new(&path(&self.embed), EMBED_THREADS, self.norm)?;
        // Exactly the window a session will hand it, so the shape checked here
        // is the shape that runs -- an embedder that rejects the production
        // frame count should fail at startup, not on someone's first meeting.
        let window = speech_like(WINDOW_SAMPLES);
        let embedding = embedder
            .embed(&window)
            .context("embedding a window of synthesised speech")?;
        ensure!(
            embedding.len() == embedder.dim(),
            "{}: embedding is {} wide, but the model declares {}",
            path(&self.embed),
            embedding.len(),
            embedder.dim()
        );
        // `Embedder::embed` normalises, so anything but a unit vector means it
        // was handed zeros -- which l2_normalize's zero guard turns into a
        // finite vector that the NaN check would let through.
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        ensure!(
            (norm - 1.0).abs() < 1e-3,
            "{}: embedding of a voiced window has length {norm:.3}, not 1",
            path(&self.embed)
        );
        Ok(embedder.dim())
    }
}

/// Which embedding model in a directory listing to use, and how to normalise
/// its features -- the four-outcome decision behind [`Models::resolve`], with
/// the filesystem left on the other side of it.
///
/// Pulled out for the same reason `vad`'s `take_frames` and `Gate` were: the
/// arithmetic-free half is where the interesting cases are, and it should be
/// testable without a directory to build first.
///
/// The rules, in order: the calibrated model wins by name; otherwise exactly
/// one other ONNX file whose name identifies its family wins; zero or several
/// is an error that says what to do about it. Guessing is what this refuses --
/// normalisation is a property of the training recipe that the ONNX file does
/// not record, and the wrong answer produces embeddings that separate nobody
/// without failing anywhere.
fn pick_embed(names: &[impl AsRef<str>]) -> Result<(String, Norm)> {
    // The calibrated model needs no guessing: the design doc records that
    // 3D-Speaker is trained with plain cepstral mean normalisation.
    if names.iter().any(|n| n.as_ref() == EMBED_FILE) {
        return Ok((EMBED_FILE.to_string(), Norm::Mean));
    }

    let candidates = names
        .iter()
        .map(AsRef::as_ref)
        .filter(|n| n.ends_with(".onnx") && *n != VAD_FILE);
    let (known, unknown): (Vec<_>, Vec<_>) = candidates.partition(|n| norm_for(n).is_some());

    match known.as_slice() {
        [only] => Ok((
            (*only).to_string(),
            norm_for(only).expect("partitioned on being recognised"),
        )),
        [] => bail!(
            "no speaker-embedding model. Expected {EMBED_FILE}, or one file whose name \
             identifies its family (3dspeaker/eres2net, wespeaker, nemo/titanet) so its \
             feature normalisation is known. Found: {unknown:?}",
        ),
        several => bail!(
            "{} candidate embedding models ({}); leave exactly one, or name the chosen \
             one {EMBED_FILE}",
            several.len(),
            several.join(", ")
        ),
    }
}

/// The normalisation a model family's recipe used, from its filename.
///
/// The one such mapping in the tree, on purpose: `embed.rs` used to carry a
/// second, looser copy that defaulted an unrecognised name to `Mean`, which is
/// the silently-wrong outcome its own doc comment warned about. This is a
/// naming convention rather than a fact about the file, so a name it does not
/// recognise turns the feature off loudly instead of picking one. Matching is
/// case-insensitive: `Titanet-Large.onnx` is the same model as
/// `titanet-large.onnx`.
///
/// `pub` for `examples/diarize_probe`, which compares model families against
/// each other and so names each file on the command line rather than resolving
/// a directory. Keeping it the only mapping matters more there than anywhere:
/// the probe is where a model's numbers are measured, and measuring one under
/// the wrong normalisation would put a wrong number in the design doc.
#[doc(hidden)]
pub fn norm_for(name: &str) -> Option<Norm> {
    let name = name.to_ascii_lowercase();
    if name.contains("nemo") || name.contains("titanet") {
        Some(Norm::MeanVar)
    } else if name.contains("3dspeaker") || name.contains("eres2net") || name.contains("wespeaker")
    {
        Some(Norm::Mean)
    } else {
        None
    }
}

/// `n` samples of something a VAD trained on speech will call speech: the
/// harmonics of a 120 Hz glottal pulse shaped by three formants, under a 4 Hz
/// syllable envelope.
///
/// Synthetic on purpose -- shipping a wav file to check a model at startup
/// would put audio in the repository and one more thing in the container that
/// has to be found at run time. White noise was the obvious alternative and
/// does not work: silero puts it at a speech probability of 0.02, which is
/// indistinguishable from the silence probe.
fn speech_like(n: usize) -> Vec<f32> {
    const F0: f32 = 120.0;
    const FORMANTS: [f32; 3] = [700.0, 1220.0, 2600.0];
    const NYQUIST: f32 = 8_000.0;
    let sample_rate = crate::diarize::fbank::SAMPLE_RATE;

    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let mut x = 0.0;
            for harmonic in 1..=40 {
                let f = F0 * harmonic as f32;
                if f >= NYQUIST {
                    break;
                }
                // A cheap resonance: each formant contributes a peak 100 Hz
                // wide, and the source itself rolls off as 1/harmonic.
                let gain: f32 = FORMANTS
                    .iter()
                    .map(|centre| 1.0 / (1.0 + ((f - centre) / 100.0).powi(2)))
                    .sum();
                x += gain * (std::f32::consts::TAU * f * t).sin() / harmonic as f32;
            }
            let envelope = 0.5 + 0.5 * (std::f32::consts::TAU * 4.0 * t).sin();
            0.3 * x * envelope
        })
        .collect()
}

/// Consecutive embeddings the embedder may fail before `RealDiarizer` calls
/// itself broken.
///
/// Six, and not five to match the session's strikes, because the two count
/// different things and stack: each of these failures is also a chunk the
/// session warns about, so the whole episode is at most six plus five lines.
/// Six consecutive embeddings is a little over two seconds of *voiced* audio,
/// which is enough that a failure with a cause outside the model -- a failed
/// allocation under memory pressure is the plausible one -- does not cost a
/// session its labels. Nothing else transient can reach here: audio arrives as
/// `pcm_s16le_to_f32` output, so a non-finite embedding is the model's doing
/// and will not stop being the model's doing.
///
/// It was three while a window was the only thing embedded. There are now two
/// embeddings per hop -- the hop itself and, on most hops, the window it
/// completes -- so three would have halved the tolerance measured in seconds
/// of speech, which is the quantity the paragraph above is reasoning about.
/// The number moved so the reasoning would not have to.
const MAX_EMBED_FAILURES: u32 = 6;

/// Whether the embedder is still worth asking, counted in embeddings.
///
/// A latch, not a policy. It answers one question -- can this diarizer still
/// produce a label? -- and once the answer is no it stays no. What to *do*
/// about a diarizer that keeps failing is not decided here: that is
/// `session.rs`, which counts consecutive failed chunks and retires the
/// diarizer after five of them, and it stays the only place that decides it.
///
/// The two counts are deliberately different quantities. The session's is
/// chunks, because a chunk is what it hands over and a warning is what a
/// failed one costs. This one is embeddings, because an embedding is what the
/// embedder is actually asked for. Counting embeddings here is what lets the
/// chunk count downstream terminate; see `RealDiarizer`'s failure notes.
#[derive(Default)]
struct Fuse {
    /// Failed embeddings since the last one that worked.
    consecutive: u32,
    /// The failure that blew it. Set once, at [`MAX_EMBED_FAILURES`] in a
    /// row, and never cleared.
    terminal: Option<String>,
}

impl Fuse {
    /// The failure this blew on, or `None` while the embedder is still worth
    /// asking.
    fn blown(&self) -> Option<&str> {
        self.terminal.as_deref()
    }

    /// An embedding that worked: whatever came before it was a blip.
    fn passed(&mut self) {
        self.consecutive = 0;
    }

    /// One that did not, returning how many that makes in a row.
    fn failed(&mut self, e: &anyhow::Error) -> u32 {
        self.consecutive += 1;
        if self.consecutive >= MAX_EMBED_FAILURES {
            self.terminal.get_or_insert_with(|| format!("{e:#}"));
        }
        self.consecutive
    }
}

/// One session's speaker attribution.
///
/// Per [`Diarizer::push`]: one ASR chunk in, one [`Attribution`] out. The
/// chunk is framed, the VAD marks which frames carry speech, and the voiced
/// ones accumulate into two sizes of audio, each of which is embedded.
///
/// **1.5 s windows** are what the spike measured, and they keep exclusive
/// ownership of everything that has to be right: only a window updates a
/// centroid, and only a window can mint a speaker.
///
/// **0.75 s hops** are the second embedding, one per hop of voiced audio, and
/// they answer two questions a window answers too late. Comparing a hop
/// against the one before it detects a turn change at hop resolution, 0.736 s,
/// against an effective resolution of a window and a half; and a hop matched
/// against the existing centroids under a stricter margin offers a provisional
/// label about 0.75 s into a turn rather than after a clean window plus the
/// session's vote. A hop is noisier than a window, so it is never allowed to
/// write anything down -- `OnlineClusterer::observe_short` takes `&self`.
///
/// A chunk that completes neither -- most of them, since a hop is 0.75 s of
/// *voiced* audio and a chunk is 0.56 s of any -- answers with the last label
/// settled on. That carry-forward is what keeps a turn from arriving in
/// labelled and unlabelled halves, and it is the one place the streaming
/// pipeline cannot follow the spike: the spike paints each window's label back
/// over the audio the window was built from, and a live session has no past to
/// paint. Three things end it: a chunk with no voiced frame in it, which
/// answers `None` rather than attributing a pause to whoever spoke before it;
/// a window the clusterer cannot place, which makes the previous answer stale;
/// and a detected turn change, because asserting the outgoing speaker's name
/// over the incoming speaker's opening sentence is worse than saying nothing.
///
/// **Corrections.** A live session has no past to paint, but it does have a
/// past it can *name*. Two things become knowable after the fact and are
/// reported as [`Relabel`]s for the session to turn into
/// `transcript.relabel`: a speaker being minted, whose first seconds were
/// committed with no name because four windows had not yet agreed, and a full
/// window contradicting the provisional label a hop had been offering. Neither
/// crosses a detected turn change, so a correction can never put one person's
/// name on another's sentence.
///
/// **On failure:** `session.rs` warns per failed chunk and retires the
/// diarizer after five *consecutive* ones, so every error path here has to be
/// able to reach five in a row. Model loading, a graph missing an output, and
/// a VAD/framing disagreement all fail identically on every chunk, so they
/// strike out inside three seconds and log five lines.
///
/// Embedding was the path that could not, and `Fuse` is why it now can. It
/// fails per *window*, and only about three chunks in four complete one under
/// continuous speech -- 17.5 frames arrive per chunk against a 23-frame hop --
/// so the chunks in between answered `Ok` and reset the session's count. An
/// embedder producing non-finite vectors warned roughly once a second for the
/// length of the meeting and never retired. The fuse counts the quantity that
/// fault actually has, consecutive failed windows, and once it blows this
/// diarizer stops answering at all: every later push fails, so the session's
/// five strikes land within five chunks and the episode costs at most eight
/// warnings rather than one a second. Retiring the diarizer is still the
/// session's decision, in the one place it lives; all this reports is that
/// there is nothing left to ask.
pub struct RealDiarizer {
    models: Models,
    /// Built on the first `push`, not in `diarizer()`.
    ///
    /// `ws.rs` calls `diarizer()` on the async runtime, where committing two
    /// ONNX graphs would stall every other session sharing that worker;
    /// `push` runs on the blocking pool, which is where a slow call belongs.
    /// The cost is that a load failure surfaces as a push error rather than at
    /// construction -- and `Diarizer::push` returning `Result` is exactly the
    /// shape that failure already has to take, since `diarizer()` cannot
    /// return one.
    state: Option<State>,
    framer: Framer,
    assembler: WindowAssembler,
    clusterer: OnlineClusterer,
    /// The cosine below which two consecutive hops are two different people:
    /// the server's `diarize_change_threshold`.
    change_threshold: f32,
    /// The label most recently settled on, and the answer for every chunk with
    /// voice in it until something changes it. `None` while the clusterer is
    /// undecided -- before the first speaker is minted, after any window it
    /// could not place, and after a detected turn change.
    last_label: Option<u32>,
    /// The previous hop's embedding, for the change-point comparison. `None`
    /// before the first hop and after a gap long enough to have cleared the
    /// accumulator, where there is nothing to compare against.
    previous_hop: Option<Vec<f32>>,
    /// The chunk at which the run of chunks answered with no speaker began,
    /// and `None` while a speaker is being named. What a mint relabels: those
    /// chunks were committed unattributed and now have somebody to attribute
    /// them to.
    ///
    /// Reset at a detected turn change, which is what keeps a correction
    /// inside one turn.
    unlabelled_since: Option<u64>,
    /// The provisional label a hop is currently asserting and the chunk it
    /// started at. What a contradicting window relabels.
    provisional: Option<(u32, u64)>,
    /// Chunks pushed so far. `Relabel` names chunks by this count, which is
    /// the same one `Session` keeps: the session pushes exactly one chunk per
    /// chunk, in order, so the two agree without exchanging anything.
    chunks_seen: u64,
    /// Whether the embedder is still worth asking. See the failure notes
    /// above: this is what makes a per-embedding fault visible to a session
    /// that counts chunks.
    fuse: Fuse,
}

/// The loaded models, once the first chunk has paid for them.
struct State {
    vad: Vad,
    embedder: Embedder,
}

impl State {
    fn load(models: &Models) -> Result<Self> {
        Ok(Self {
            vad: Vad::new(&models.vad.to_string_lossy())?,
            embedder: Embedder::new(&models.embed.to_string_lossy(), EMBED_THREADS, models.norm)?,
        })
    }
}

impl RealDiarizer {
    fn new(models: Models, tuning: DiarizeTuning) -> Self {
        Self {
            models,
            state: None,
            framer: Framer::default(),
            assembler: WindowAssembler::default(),
            clusterer: OnlineClusterer::with_config(tuning.min_pool, tuning.margin),
            change_threshold: tuning.change_threshold,
            last_label: None,
            previous_hop: None,
            unlabelled_since: None,
            provisional: None,
            chunks_seen: 0,
            fuse: Fuse::default(),
        }
    }

    /// What an embedding the model could not produce costs: one mark against
    /// the fuse, and the error back to the caller with the count in it, in the
    /// shape `session.rs` already logs failures in.
    ///
    /// Split out of `push` so the failure path can be driven without a model.
    /// The only thing left on the other side of it is the `embed` call itself.
    fn window_failed(&mut self, e: anyhow::Error) -> anyhow::Error {
        let n = self.fuse.failed(&e);
        e.context(format!(
            "embedding voiced audio ({n} of {MAX_EMBED_FAILURES} in a row)"
        ))
    }

    /// A hop: the turn-change test, then a provisional label if there is
    /// nothing better on offer.
    ///
    /// Returns whether this hop was a change point, which the caller needs in
    /// order to throw away the windows behind it.
    fn hop(&mut self, embedding: Vec<f32>, chunk: u64, out: &mut Attribution) -> bool {
        let changed = self
            .previous_hop
            .as_ref()
            .is_some_and(|previous| cosine(previous, &embedding) < self.change_threshold);
        self.previous_hop = Some(embedding.clone());

        if changed {
            out.boundary = true;
            // Everything accumulated before this hop is the outgoing speaker,
            // so a window built from it would be a mixture of two voices.
            self.assembler.cut_at_boundary();
            // And nothing known about the outgoing speaker is true of the
            // incoming one. Saying nothing beats saying the wrong name.
            self.last_label = None;
            self.provisional = None;
            self.unlabelled_since = Some(chunk);
        }

        // Only where there is nothing better. A hop's guess never overwrites
        // an answer a full window settled: it is the faster of the two, not
        // the more trustworthy one.
        if self.last_label.is_none()
            && let Some(label) = self.clusterer.observe_short(&embedding)
        {
            self.last_label = Some(label);
            self.provisional = Some((label, chunk));
        }
        changed
    }

    /// A full 1.5 s window: the only thing that moves a centroid, mints a
    /// speaker, or corrects one.
    fn window(&mut self, embedding: &[f32], chunk: u64, out: &mut Attribution) {
        let before = self.clusterer.minted();
        let settled = self.clusterer.observe(embedding);
        let minted = self.clusterer.minted() > before;

        match (settled, minted) {
            // A speaker exists who did not before. The chunks since this turn
            // began were committed with nobody's name on them, and now there
            // is one -- which is the whole of complaint 1, fixed without
            // touching the four-window reluctance that made it.
            (Some(speaker), true) => {
                let from = self.unlabelled_since.unwrap_or(chunk);
                out.relabels.push(Relabel {
                    from_chunk: from,
                    to_chunk: chunk,
                    speaker,
                });
            }
            // A window disagreeing with the guess a hop had been offering.
            // Correcting it is better than leaving it wrong, and the hop is
            // allowed to be wrong precisely because this exists.
            (Some(speaker), false) => {
                if let Some((provisional, since)) = self.provisional
                    && provisional != speaker
                {
                    out.relabels.push(Relabel {
                        from_chunk: since,
                        to_chunk: chunk,
                        speaker,
                    });
                }
            }
            (None, _) => {}
        }

        // An undecided clusterer overwrites a known label with None on
        // purpose: it has just seen 1.5 s of speech it cannot place, so the
        // previous answer is stale rather than still true.
        self.last_label = settled;
        self.provisional = None;
        match settled {
            Some(_) => self.unlabelled_since = None,
            None => {
                self.unlabelled_since.get_or_insert(chunk);
            }
        }
    }
}

impl Diarizer for RealDiarizer {
    fn push(&mut self, audio: &[f32]) -> Result<Attribution> {
        // Counted before anything can fail, so the count keeps step with the
        // session's however this call ends: `Relabel` names chunks by number
        // and the two sides never exchange one.
        let chunk = self.chunks_seen;
        self.chunks_seen += 1;

        // A blown fuse answers before anything else, the lazy load and the
        // framer included. Failing every chunk is the whole point: it is what
        // turns a fault the embedder raises per embedding into one the session
        // sees per chunk, so its consecutive count can reach the end. Nothing
        // below is worth doing on the way -- the stream position the framer
        // keeps is only of use to a diarizer that might answer again.
        if let Some(why) = self.fuse.blown() {
            bail!(
                "the embedder failed {MAX_EMBED_FAILURES} embeddings in a row and is \
                 out of service; the last of them: {why}"
            );
        }

        // The lazy load has to fail *before* the framer sees this chunk, and
        // that ordering is load-bearing rather than incidental: the framer
        // mirrors a remainder that only exists once the VAD does, so consuming
        // a chunk the VAD never saw would offset every later frame index by
        // however much audio arrived before the models failed to load. Any
        // tidying that groups the two uses of `audio` together breaks this
        // silently -- the `ensure!` below would not fire, because both sides
        // stay internally consistent while describing different audio.
        if self.state.is_none() {
            self.state = Some(State::load(&self.models).context("loading the diarization models")?);
        }
        // Framed before the VAD runs, and unconditionally: `Vad::run` takes
        // the whole chunk into its own remainder before its first inference,
        // so a failure part-way through still consumed the audio. Advancing
        // here regardless keeps the two sides agreeing about where the stream
        // is, and leaves the lost frames looking like the gap they are.
        let framed = self.framer.take(audio);
        let frames = framed.samples.len() / FRAME;
        let voiced = self.state.as_mut().expect("loaded above").vad.run(audio)?;
        ensure!(
            voiced.len() == frames,
            "the VAD reported {} frames for a chunk framed into {frames}; its flags \
             index the concatenated stream and are no longer aligned with the audio",
            voiced.len()
        );

        let mut out = Attribution::default();
        // A change point invalidates every window still to come out of this
        // batch: they were assembled before the cut, from audio either side of
        // the boundary, which is exactly the mixture the cut says they are.
        // (A hop and a window never complete on the same frame -- 23 and 47
        // frames are coprime in the relevant sense -- but they can complete on
        // the same chunk, which is what this is for.)
        let mut cut = false;
        for piece in self.assembler.push(&framed, &voiced) {
            let (Cut::Hop(audio) | Cut::Window(audio)) = &piece;
            if matches!(piece, Cut::Window(_)) && cut {
                continue;
            }
            let embedding = match self
                .state
                .as_mut()
                .expect("loaded above")
                .embedder
                .embed(audio)
            {
                Ok(embedding) => embedding,
                Err(e) => return Err(self.window_failed(e)),
            };
            self.fuse.passed();
            match piece {
                Cut::Hop(_) => cut |= self.hop(embedding, chunk, &mut out),
                Cut::Window(_) => self.window(&embedding, chunk, &mut out),
            }
        }

        // Silence answers None rather than repeating itself. Carrying a label
        // across a chunk with no voice in it would attribute the pause to
        // whoever spoke before it, and `speaker: None` exists for exactly this.
        if voiced.iter().any(|&v| v) {
            out.speaker = self.last_label;
            out.provisional = self.provisional.is_some();
        }
        if out.speaker.is_none() {
            self.unlabelled_since.get_or_insert(chunk);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_calibrated_model_names_are_recognised() {
        assert_eq!(norm_for(EMBED_FILE), Some(Norm::Mean));
        assert_eq!(
            norm_for("wespeaker_en_voxceleb_resnet34_LM.onnx"),
            Some(Norm::Mean)
        );
        assert_eq!(norm_for("nemo_en_titanet_small.onnx"), Some(Norm::MeanVar));
    }

    #[test]
    fn an_unrecognised_name_is_not_guessed_at() {
        // The failure this prevents is silent: the wrong normalisation
        // produces embeddings that look fine and separate nobody.
        assert_eq!(norm_for("model.onnx"), None);
        assert_eq!(norm_for("embedding.onnx"), None);
    }

    #[test]
    fn the_calibrated_model_wins_by_name() {
        // Even in the directory the spike left behind, which holds all three
        // candidates: the name that was measured is not up for a vote.
        let listing = [
            VAD_FILE,
            "wespeaker_en_voxceleb_resnet34_LM.onnx",
            EMBED_FILE,
            "nemo_en_titanet_small.onnx",
            "checksum.txt",
        ];
        assert_eq!(
            pick_embed(&listing).unwrap(),
            (EMBED_FILE.to_string(), Norm::Mean)
        );
    }

    #[test]
    fn one_recognisable_model_is_taken_with_its_own_normalisation() {
        let listing = [VAD_FILE, "Titanet-Large.onnx", "notes.md"];
        assert_eq!(
            pick_embed(&listing).unwrap(),
            ("Titanet-Large.onnx".to_string(), Norm::MeanVar),
            "the mapping is case-insensitive, so a capitalised checkpoint \
             still gets NeMo's per-feature normalisation"
        );
    }

    #[test]
    fn a_directory_with_no_embedding_model_says_what_is_missing() {
        // The VAD alone is not enough, and neither is an unrecognisable name:
        // both leave the operator needing to know which file to add or rename.
        let err = pick_embed(&[VAD_FILE, "model.onnx"]).expect_err("nothing recognisable");
        let message = format!("{err:#}");
        assert!(message.contains(EMBED_FILE), "{message}");
        assert!(message.contains("model.onnx"), "{message}");
    }

    #[test]
    fn two_candidates_are_ambiguous_rather_than_arbitrary() {
        // Picking one silently would make the labels depend on directory
        // order, so this refuses -- and names both files, since the fix is to
        // remove or rename one of them.
        let err = pick_embed(&[
            VAD_FILE,
            "wespeaker_en_voxceleb_resnet34_LM.onnx",
            "nemo_en_titanet_small.onnx",
        ])
        .expect_err("two recognisable candidates");
        let message = format!("{err:#}");
        assert!(
            message.contains("wespeaker_en_voxceleb_resnet34_LM.onnx"),
            "{message}"
        );
        assert!(message.contains("nemo_en_titanet_small.onnx"), "{message}");
        assert!(message.contains(EMBED_FILE), "{message}");
    }

    #[test]
    fn a_directory_without_a_vad_is_refused_by_name() {
        let err = Models::resolve(Path::new("/nonexistent-diarize-models"))
            .expect_err("no directory, so no VAD");
        assert!(err.to_string().contains(VAD_FILE), "{err}");
    }

    // ----------------------------------------------------------------- fuse

    #[test]
    fn a_window_that_embeds_wipes_the_slate() {
        // Only *consecutive* failures say anything about the model. The
        // failures this has to survive have causes outside it -- a failed
        // allocation under memory pressure -- and a session that still gets
        // an embedding out of every other window is still labelling.
        let mut fuse = Fuse::default();
        for _ in 0..MAX_EMBED_FAILURES * 3 {
            fuse.failed(&anyhow::anyhow!("a blip"));
            fuse.passed();
        }
        assert!(fuse.blown().is_none(), "a blip retired the embedder");
    }

    #[test]
    fn windows_that_keep_failing_blow_the_fuse_for_good() {
        let mut fuse = Fuse::default();
        for i in 1..MAX_EMBED_FAILURES {
            assert_eq!(fuse.failed(&anyhow::anyhow!("window {i}")), i);
            assert!(fuse.blown().is_none(), "blown after only {i}");
        }
        fuse.failed(&anyhow::anyhow!("the last straw"));
        assert_eq!(fuse.blown(), Some("the last straw"));

        // A latch: a later window that embeds does not buy the model its job
        // back, and the reason recorded is the one that ended it rather than
        // whatever happened last.
        fuse.passed();
        fuse.failed(&anyhow::anyhow!("something else"));
        assert_eq!(fuse.blown(), Some("the last straw"));
    }

    #[test]
    fn a_blown_fuse_fails_every_push_after_it() {
        // The point of the latch, and the half of it that lives in `push`:
        // `session.rs` counts consecutive failed *chunks*, so a diarizer that
        // has concluded it is broken has to fail at chunk granularity for that
        // count to reach five. `tests/diarize.rs` has the other half -- a
        // diarizer failing every chunk is dropped within five of them.
        //
        // No models are involved, and that is itself the contract: the check
        // comes before the lazy load, so a diarizer that will never answer
        // again does not commit two ONNX graphs in order to say so. Paths that
        // cannot load prove it -- if the check moved below the load, the error
        // would be about the missing file instead.
        let mut d = RealDiarizer::new(
            Models {
                vad: PathBuf::from("/nonexistent/silero_vad.onnx"),
                embed: PathBuf::from("/nonexistent/embed.onnx"),
                norm: Norm::Mean,
            },
            DiarizeTuning::default(),
        );
        for _ in 0..MAX_EMBED_FAILURES {
            let _ = d.window_failed(anyhow::anyhow!("embedding 0 is not finite at dimension 3"));
        }

        // Five chunks, because five consecutive failures is what the session
        // needs, and every one of them has to be a failure.
        for chunk in 1..=5 {
            let err = d
                .push(&[0.0; 8960])
                .expect_err("a blown fuse never answers again");
            let message = format!("{err:#}");
            assert!(
                message.contains("out of service") && message.contains("not finite"),
                "chunk {chunk} failed for the wrong reason: {message}"
            );
        }
    }

    #[test]
    fn the_speech_probe_is_a_signal_not_silence() {
        // The VAD half of the self-check is only as good as this: if it were
        // all zeros, "the VAD found speech in it" would be unfalsifiable.
        // (That it reads as *speech* is silero's judgement, and only the
        // self-check itself can ask for it.)
        let s = speech_like(16_000);
        assert_eq!(s.len(), 16_000);
        assert!(s.iter().all(|x| x.is_finite()));
        let peak = s.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!((0.05..1.0).contains(&peak), "peak amplitude {peak}");
    }
}
