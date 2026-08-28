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
use crate::diarize::cluster::{Heard, OnlineClusterer, cosine};
use crate::diarize::window::{Cut, FRAME, Framer, HOP_SAMPLES, WINDOW_SAMPLES, WindowAssembler};
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
            mint_ceiling = tuning.mint_ceiling,
            change_threshold = tuning.change_threshold,
            gap_change_threshold = tuning.gap_change_threshold,
            "speaker labelling available"
        );
        Ok(Self { models, tuning })
    }

    /// The embedding model this factory resolved, by path.
    ///
    /// For `examples/diarize_probe`, which reports the model beside the
    /// numbers it measured. The probe cannot name one here -- the server's own
    /// rules choose it from the directory -- so reading it back is the only
    /// way for the name printed to be the model that ran.
    #[doc(hidden)]
    pub fn embed_model(&self) -> &Path {
        &self.models.embed
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
    /// vector; and an embedder that accepts one of the two input lengths a
    /// session hands it and not the other.
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
        // Both shapes a session hands it, because it hands over both: a 1.5 s
        // window when one completes, and a 0.75 s hop on *every* hop of voiced
        // audio. A fixed-input-length graph that only accepts the window would
        // pass a window-only check at startup and then fail every hop for the
        // length of the meeting.
        for samples in [WINDOW_SAMPLES, HOP_SAMPLES] {
            let embedding = embedder
                .embed(&speech_like(samples))
                .with_context(|| format!("embedding {samples} samples of synthesised speech"))?;
            ensure!(
                embedding.len() == embedder.dim(),
                "{}: embedding of {samples} samples is {} wide, but the model \
                 declares {}",
                path(&self.embed),
                embedding.len(),
                embedder.dim()
            );
            // `Embedder::embed` normalises, so anything but a unit vector
            // means it was handed zeros -- which l2_normalize's zero guard
            // turns into a finite vector that the NaN check would let through.
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            ensure!(
                (norm - 1.0).abs() < 1e-3,
                "{}: embedding of {samples} samples of voiced audio has length \
                 {norm:.3}, not 1",
                path(&self.embed)
            );
        }
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
/// Three, and not five to match the session's strikes, because the two count
/// different things and stack: each of these failures is also a chunk the
/// session warns about, so the whole episode is at most three plus five lines.
/// Three consecutive embeddings is a little over two seconds of *voiced*
/// audio, which is enough that a failure with a cause outside the model -- a
/// failed allocation under memory pressure is the plausible one -- does not
/// cost a session its labels. Nothing else transient can reach here: audio
/// arrives as `pcm_s16le_to_f32` output, so a non-finite embedding is the
/// model's doing and will not stop being the model's doing.
///
/// A chunk embeds a hop and, on most hops, the window it completes -- but
/// `push` returns on the *first* embedding that fails, so at most one of them
/// can ever mark the fuse. A mark is a chunk, which is what lets the paragraph
/// above reason in seconds of speech.
const MAX_EMBED_FAILURES: u32 = 3;

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

/// What a silence that cleared the window accumulator is still owed.
///
/// Three states rather than an `Option`, because "the hop I am holding is from
/// before a pause" and "a seam is waiting on evidence" stop being the same fact
/// the moment a correction has to be emitted in between. Losing that
/// distinction would let the hop after a pause claim a *turn boundary* against
/// the hop before it, which is a decision `window::MAX_GAP_FRAMES` measured and
/// declined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gap {
    /// Nothing outstanding: the next hop adjoins the one being held.
    Closed,
    /// A silence at this chunk, with no hop yet able to say whether the voice
    /// changed across it.
    ///
    /// The chunk is where the seam goes if the answer never comes -- see
    /// [`RealDiarizer::assume_gap_hid_a_change`]. A hop that *does* answer
    /// puts it at itself instead, which is later and so tighter.
    Pending(u64),
    /// The same silence, after something had to assume it hid a change and
    /// committed the seam at the pause.
    ///
    /// The evidence is still wanted. A bound already given out cannot be
    /// *loosened* -- a correction may already have been emitted against it --
    /// but it can still be tightened, and when the hop finally arrives and the
    /// voices differ it should be: the hop is later than the pause, and the
    /// chunk a silence begins in is the one place a bound must not be drawn
    /// (see [`RealDiarizer::hop`]). What this state stops is a second forced
    /// commit of the same silence, not the comparison.
    Assumed,
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
/// window contradicting the provisional label a hop had been offering.
///
/// How far back one may reach is `correctable_since`, and what bounds it is
/// worth stating exactly rather than as "it never crosses a turn change".
/// Three things move it forward, and they are the three points at which the
/// audio before it has provenance this diarizer cannot vouch for: a *detected*
/// turn change; the first completed hop of the session, which had nothing
/// before it to be compared against, so no change could have been detected
/// there even in principle; and a silence that cleared the accumulator with a
/// different voice on the far side of it from the one that stopped. A run is
/// also ended by any window that settles a label, since a confident label is
/// not correctable.
///
/// **The silence is the one of the three that is measured rather than
/// assumed**, and it is measured late. Half a second of quiet is constant in
/// conversation -- a breath, an "um", a thought -- so treating every one of
/// them as a seam truncated the backfill to whatever followed the most recent
/// pause: 2 corrections over 21 minutes of ES2002a, where the complaint that
/// it "doesn't go far enough back" came from. The seam is therefore recorded
/// as *pending* at the silence rather than committed, the hop from before the
/// pause is kept instead of dropped, and the next hop to complete is compared
/// against it under [`crate::diarize::cluster::T_GAP_CHANGE`]. Voices that
/// match discard the seam, and a correction reaches through the pause; voices
/// that differ commit it, which is the behaviour that shipped. A silence still
/// claims no *boundary* either way -- that decision is `MAX_GAP_FRAMES`'s and
/// its measurement stands.
///
/// **Anything that would emit a correction while the answer is outstanding
/// commits the seam first.** No correction ever crosses an unresolved seam,
/// because the case a seam exists for is precisely the one where the missing
/// evidence would have said "somebody else". Evidence arriving after that is
/// still allowed to *tighten* the bound and never to loosen it, which is what
/// [`Gap::Assumed`] carries.
///
/// **What that does not guarantee.** A turn change entirely inside one hop is
/// invisible to a detector that compares whole hops, so a correction can in
/// principle reach over an interjection shorter than 0.736 s of voiced audio
/// that carried no label of its own. The exposure is bounded to one hop and
/// to the opening of a run -- anywhere else the incumbent's carried-forward
/// label is on those chunks, and `session.rs` will not overwrite one. Deferring
/// the gap seam adds one shape to that bound and no others: an interjection too
/// short to complete a hop, sitting *between* two silences, is never embedded
/// on its own -- so the comparison that resolves the seam is made across it,
/// between the voices either side, and a correction may reach over it when
/// those two agree.
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
    /// The cosine below which the hops either side of a silence are two
    /// different people. Stricter than `change_threshold`, and used for one
    /// decision only: whether a pending gap seam is committed or discarded.
    gap_change_threshold: f32,
    /// The label most recently settled on, and the answer for every chunk with
    /// voice in it until something changes it. `None` while the clusterer is
    /// undecided -- before the first speaker is minted, after any window it
    /// could not place, and after a detected turn change.
    last_label: Option<u32>,
    /// The previous hop's embedding, for the change-point comparison. `None`
    /// only before the first hop of the session, which is the one place there
    /// is genuinely nothing to compare against.
    ///
    /// It survives a silence, and what a silence changes is the question being
    /// asked of it: `gap` says whether the hop about to arrive adjoins this one
    /// or merely follows it across a pause, and the two are held to different
    /// thresholds and decide different things.
    previous_hop: Option<Vec<f32>>,
    /// What a silence that cleared the accumulator is still owed an answer
    /// about.
    gap: Gap,
    /// The earliest chunk a correction may still reach back to, or `None`
    /// while the current chunk carries a settled label.
    ///
    /// Not simply "where the unlabelled run began": it is bounded by the last
    /// point this diarizer can vouch for the audio's provenance, which is a
    /// detected turn change, the first hop of the session, or a silence with a
    /// different voice on the far side of it. See the type's own notes -- the
    /// safety of every relabel emitted here rests on this field, on nothing
    /// else, and only while `gap` holds no unresolved seam.
    correctable_since: Option<u64>,
    /// The provisional label currently being asserted and the chunk it started
    /// at. What a contradicting window relabels.
    ///
    /// Either a hop's guess or a full window the clusterer would not stand
    /// behind; the two are the same promise to the reader and the same promise
    /// to `session.rs`, which is that a later window may take it back.
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
            clusterer: OnlineClusterer::with_config(tuning),
            change_threshold: tuning.change_threshold,
            gap_change_threshold: tuning.gap_change_threshold,
            last_label: None,
            previous_hop: None,
            gap: Gap::Closed,
            correctable_since: None,
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

    /// Mark a point the audio's provenance cannot be carried across.
    ///
    /// A correction may reach back to here and no further, because whatever
    /// was said before it might have been somebody else and this diarizer has
    /// no way to know. Two callers, and they are the two points where that is
    /// known at the time: a detected turn change, and the first hop of a
    /// session. The third, a silence, is decided later -- see [`Gap`] and
    /// [`RealDiarizer::commit_gap_seam`].
    fn seam(&mut self, chunk: u64) {
        self.correctable_since = Some(chunk);
    }

    /// Give up on a pending gap seam and apply it, wherever the run has got to
    /// since.
    ///
    /// A floor rather than an assignment, which is the one way this differs
    /// from [`RealDiarizer::seam`] and it matters: chunks have gone by since
    /// the silence, and one of them may have carried a window that settled a
    /// label and ended the correctable run outright. Writing the seam's chunk
    /// in would reopen that run and let a correction reach back over text a
    /// full window placed.
    fn commit_gap_seam(&mut self, chunk: u64) {
        if let Some(since) = self.correctable_since {
            self.correctable_since = Some(since.max(chunk));
        }
    }

    /// Resolve any pending gap seam the cautious way, because something is
    /// about to emit a correction and the evidence has not arrived.
    ///
    /// The hop that would have answered is still to come, so the only safe
    /// answer is the one the seam was recorded for: assume the silence hid a
    /// speaker change. `Gap::Assumed` rather than `Gap::Closed`, so the hop
    /// when it does arrive still knows it follows a pause -- and can still
    /// tighten the bound this hands out, which is the only direction left open
    /// once a correction may have been emitted against it.
    ///
    /// **Not an unreachable precaution**, though the shipped geometry hides
    /// how it is reached: a hop completes at 23 voiced frames and a window at
    /// 47, so the answer normally arrives first. It does not when the hop's
    /// embedding *failed* -- the one transient [`Fuse`] exists to survive --
    /// leaving the window that follows it to mint with the silence still
    /// unaccounted for. Making the guard a rule rather than a consequence of
    /// that arithmetic is the point: the promise being kept here is the one
    /// the protocol makes about never renaming somebody's sentence.
    fn assume_gap_hid_a_change(&mut self) {
        if let Gap::Pending(chunk) = self.gap {
            self.commit_gap_seam(chunk);
            self.gap = Gap::Assumed;
        }
    }

    /// Whether the voice in `embedding` is a different one from the hop held
    /// across the silence before it.
    ///
    /// Four ways to answer yes, and each is a reason the seam has to stand.
    /// Comparison may be switched off, at `diarize_change_threshold = 0`,
    /// where no cosine between hops is allowed to decide anything and the
    /// reach of a correction is a decision. There may be no earlier hop to
    /// compare against, which is the opening of a session and the case the
    /// seam was invented for. The comparison may not be answerable at all. Or
    /// the two may genuinely differ, at the stricter bar a comparison across
    /// silence is held to.
    fn voice_changed_across_the_gap(&self, embedding: &[f32]) -> bool {
        self.change_threshold <= 0.0
            || self.previous_hop.as_ref().is_none_or(|before| {
                let cos = cosine(before, embedding);
                // Checked here rather than inferred from `real::embed`
                // refusing non-finite vectors two modules away. `cosine` is a
                // bare dot product, so a NaN reaching it makes `cos <
                // threshold` *false* -- "same voice, discard the seam", which
                // is the unsafe direction and the opposite of every other
                // default in this file. A comparison that cannot be made has
                // answered nothing, and nothing is what a seam is for.
                //
                // Not a panic: a session's contract with a failing model is to
                // stop labelling, not to take the process down, and `Fuse`
                // already owns that decision.
                !cos.is_finite() || cos < self.gap_change_threshold
            })
    }

    /// A hop: the turn-change test, then a provisional label if there is
    /// nothing better on offer.
    ///
    /// Returns whether this hop was an accepted change point, which the caller
    /// needs in order to throw away the windows behind it.
    fn hop(&mut self, embedding: Vec<f32>, chunk: u64, out: &mut Attribution) -> bool {
        // The first hop after a silence is asked a different question from
        // every other one -- not "has the voice changed since the hop before"
        // but "is this the voice that stopped" -- and the only thing that
        // turns on the answer is how far back a correction may reach. It
        // claims no boundary whatever it finds: a silence is a poor
        // turn-change detector, which is what the 51 decidable breaks behind
        // `window::MAX_GAP_FRAMES` measured, and none of this reopens that.
        let gap = std::mem::replace(&mut self.gap, Gap::Closed);
        if gap != Gap::Closed {
            if self.voice_changed_across_the_gap(&embedding) {
                // At this hop rather than at the pause, which is the tighter
                // of the two and exactly where the rule this replaces put it.
                // Half a second is under half a chunk, so the chunk a silence
                // begins in can carry the outgoing speaker's last words as
                // well as the incoming speaker's first, and a bound drawn
                // there would let a correction reach into the sentence it is
                // meant to stop at.
                //
                // `Gap::Assumed` takes this branch too. Something has already
                // committed that seam at the pause, and a bound handed out
                // cannot be loosened -- but this is a floor, so applying it
                // can only move the bound forward, off the chunk the silence
                // began in and onto the hop that measured the change. The rule
                // this replaces did exactly that, by clearing the held hop and
                // letting the next one seam as the first of a run.
                self.commit_gap_seam(chunk);
            }
            self.previous_hop = Some(embedding.clone());
            self.guess(&embedding, chunk);
            return false;
        }

        // A threshold of 0 is the documented way to switch detection off, and
        // it has to be checked for rather than reached by arithmetic:
        // different-speaker cosines are routinely negative, so `cosine < 0`
        // fires often and no value of a bare threshold would ever mean "never".
        let first = self.previous_hop.is_none();
        let differs = self.change_threshold > 0.0
            && self
                .previous_hop
                .as_ref()
                .is_some_and(|previous| cosine(previous, &embedding) < self.change_threshold);
        self.previous_hop = Some(embedding.clone());

        // Everything accumulated before this hop is the outgoing speaker, so a
        // window built from it would be a mixture of two voices -- but the
        // assembler refuses a cut that would leave no room for a window to
        // complete, and a refused cut is not a boundary at all. Acting on one
        // anyway would blank the label and stop the vote for a turn change
        // nothing downstream can ever confirm.
        let changed = differs && self.assembler.cut_at_boundary();

        if changed {
            out.boundary = true;
            // Nothing known about the outgoing speaker is true of the incoming
            // one. Saying nothing beats saying the wrong name.
            self.last_label = None;
            self.provisional = None;
            self.seam(chunk);
        } else if first {
            // The first hop of the session had nothing before it to be
            // compared against, so no change could have been detected inside
            // it however short the opening turn was. Whatever was said before
            // this point is of unknown provenance and stays uncorrected.
            self.seam(chunk);
        }

        self.guess(&embedding, chunk);
        changed
    }

    /// Offer a hop's provisional label, where there is nothing better.
    ///
    /// A hop's guess never overwrites an answer a full window settled: it is
    /// the faster of the two, not the more trustworthy one. Reached from both
    /// halves of [`RealDiarizer::hop`], because a hop across a silence is
    /// still 0.75 s of somebody talking and the reader wants a name for it.
    fn guess(&mut self, embedding: &[f32], chunk: u64) {
        if self.last_label.is_none()
            && let Some(label) = self.clusterer.observe_short(embedding)
        {
            self.last_label = Some(label);
            self.provisional = Some((label, chunk));
        }
    }

    /// One chunk's assembled pieces, and the silence the batch ended with.
    ///
    /// **The order of the two halves is the whole correctness of the
    /// deferral**, in both directions, which is why they are one function with
    /// one doc comment rather than two stretches of `push`.
    ///
    /// *Pieces first*, because every piece in a batch that restarted was
    /// assembled **before** the silence -- `push` refuses a chunk that is not
    /// shorter than a hop, so nothing can complete on the far side of a
    /// restart in the same batch. Recording the gap first would hand the
    /// pre-gap hop to the comparison meant for the post-gap one, which would
    /// answer "same voice" for the trivial reason that both sides of it are
    /// the same stretch of speech.
    ///
    /// *And the silence second whatever a piece did*, because
    /// [`WindowAssembler::restarted`] is cleared at the top of the next
    /// `push`: a silence not recorded here is not recorded at all -- not
    /// committed, not pending, not assumed. The next hop would be treated as
    /// adjoining, held to the looser [`Self::change_threshold`], and a
    /// correction could reach straight through the pause with no gap test made
    /// at all. The way a piece fails is a failed embedding, which is the
    /// transient [`Fuse`] exists to survive and the case [`Gap::Assumed`] was
    /// written for, so it must not also be the case that loses the gap.
    ///
    /// `embed` is a parameter for that second reason: the failure it stands
    /// for has no other way into a test, and the ordering above is what a test
    /// needs to be able to ask about.
    fn batch(
        &mut self,
        pieces: Vec<Cut>,
        restarted: bool,
        chunk: u64,
        out: &mut Attribution,
        mut embed: impl FnMut(&[f32]) -> Result<Vec<f32>>,
    ) -> Result<()> {
        // A change point invalidates every window still to come out of this
        // batch: they were assembled before the cut, from audio either side of
        // the boundary, which is exactly the mixture the cut says they are.
        //
        // They can share a chunk, which is what this is for. They cannot share
        // a frame -- windows complete at 47 + 23k voiced frames and hops at
        // 23m, and 47 is 1 mod 23 -- but that is arithmetic about the shipped
        // geometry rather than a property worth relying on, and the flag costs
        // nothing either way.
        let mut cut = false;
        let mut failed = None;
        for piece in pieces {
            let (Cut::Hop(audio) | Cut::Window(audio)) = &piece;
            if matches!(piece, Cut::Window(_)) && cut {
                continue;
            }
            let embedding = match embed(audio) {
                Ok(embedding) => embedding,
                Err(e) => {
                    failed = Some(self.window_failed(e));
                    break;
                }
            };
            self.fuse.passed();
            match piece {
                Cut::Hop(_) => cut |= self.hop(embedding, chunk, out),
                Cut::Window(_) => self.window(&embedding, chunk, out),
            }
        }

        if restarted {
            // Half a second of silence cleared the accumulator, so the hop
            // being held is from before it. That makes it the wrong thing to
            // ask "did the voice just change" of -- two stretches of a meeting
            // either side of a pause are not the voice now and the voice a
            // moment ago -- and the right thing to ask "is this the same person
            // again" of, which is the only question a correction's reach turns
            // on. So it is kept, and the seam waits for it to be asked. A
            // second silence before that happens moves the seam forward to the
            // later one, which is the tighter bound.
            self.gap = Gap::Pending(chunk);
        }
        match failed {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// A full 1.5 s window: the only thing that moves a centroid, mints a
    /// speaker, or corrects one.
    fn window(&mut self, embedding: &[f32], chunk: u64, out: &mut Attribution) {
        let before = self.clusterer.minted();
        let heard = self.clusterer.observe(embedding);
        let minted = self.clusterer.minted() > before;

        match (heard, minted) {
            // A speaker exists who did not before. The chunks since the last
            // provenance seam were committed with nobody's name on them, or
            // with a guess, and now there is a name -- which is the whole of
            // complaint 1, fixed without touching the four-window reluctance
            // that made it.
            (Heard::Settled(speaker), true) => {
                self.assume_gap_hid_a_change();
                out.relabels.push(Relabel {
                    from_chunk: self.correctable_since.unwrap_or(chunk),
                    to_chunk: chunk,
                    speaker,
                });
            }
            // A window disagreeing with the guess that was being offered.
            // Correcting it is better than leaving it wrong, and a guess is
            // allowed to be wrong precisely because this exists.
            (Heard::Settled(speaker), false) => {
                if let Some((provisional, since)) = self.provisional
                    && provisional != speaker
                {
                    self.assume_gap_hid_a_change();
                    out.relabels.push(Relabel {
                        from_chunk: since.max(self.correctable_since.unwrap_or(since)),
                        to_chunk: chunk,
                        speaker,
                    });
                }
            }
            (Heard::Guessed(_) | Heard::Unknown, _) => {}
        }

        // An undecided clusterer overwrites a known label with None on
        // purpose: it has just seen 1.5 s of speech it cannot place, so the
        // previous answer is stale rather than still true. A guess replaces it
        // too, and stays correctable.
        self.last_label = heard.label();
        self.provisional = match heard {
            Heard::Guessed(label) => Some(match self.provisional {
                Some((held, since)) if held == label => (label, since),
                _ => (label, chunk),
            }),
            Heard::Settled(_) | Heard::Unknown => None,
        };
        match heard {
            // Only a settled label ends a correctable run. A guess is text a
            // later mint is still entitled to take back.
            Heard::Settled(_) => self.correctable_since = None,
            Heard::Guessed(_) | Heard::Unknown => {
                self.correctable_since.get_or_insert(chunk);
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

        // Read off the raw chunk, before the load and before the framer, so
        // that neither is disturbed by asking and the answer is the same on
        // every chunk of a session. A chunk shorter than a hop is what lets
        // the gap be recorded *after* the pieces in `RealDiarizer::batch`,
        // and that ordering is the whole correctness of the deferral.
        //
        // Not a constant to be reasoned from: the length is the ASR backend's,
        // via `AsrBackend::chunk_samples()`, and `asr::parakeet` merely
        // happens to fix it at 0.56 s against a hop's 0.736. A longer chunk
        // could complete a hop on the far side of a silence within one push;
        // that hop would then be put to the adjoining-hop test at the looser
        // `T_CHANGE`, could claim a turn boundary `MAX_GAP_FRAMES` measured
        // and declined to claim, and would leave the gap recorded pointing at
        // a silence whose far side had already gone by. Refusing is the honest
        // answer: the deferral cannot be done at that geometry, and doing it
        // wrongly is invisible.
        const HOP_FRAMES: usize = HOP_SAMPLES / FRAME;
        ensure!(
            audio.len().div_ceil(FRAME) < HOP_FRAMES,
            "an ASR chunk of {} samples can carry {} of the {HOP_FRAMES} frames a \
             hop needs; a chunk has to be shorter than a hop for a silence inside \
             one to still have its far side ahead of it",
            audio.len(),
            audio.len().div_ceil(FRAME)
        );

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
        let pieces = self.assembler.push(&framed, &voiced);
        let restarted = self.assembler.restarted();
        // The embedder is lifted out of `self` for the call and put back
        // straight after, which is what lets it be borrowed alongside the
        // diarizer -- and what lets a test hand `batch` a closure that fails.
        let mut state = self.state.take().expect("loaded above");
        let consumed = self.batch(pieces, restarted, chunk, &mut out, |audio| {
            state.embedder.embed(audio)
        });
        self.state = Some(state);
        consumed?;

        // Silence answers None rather than repeating itself. Carrying a label
        // across a chunk with no voice in it would attribute the pause to
        // whoever spoke before it, and `speaker: None` exists for exactly this.
        if voiced.iter().any(|&v| v) {
            out.speaker = self.last_label;
            out.provisional = self.provisional.is_some();
        }
        if out.speaker.is_none() || out.provisional {
            self.correctable_since.get_or_insert(chunk);
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

    // ------------------------------------------------- provenance and seams
    //
    // `hop` and `window` decide everything a correction is allowed to reach,
    // and neither touches `state` -- so they can be driven directly, with the
    // models left on the other side of the boundary the fuse test already
    // crosses. What cannot be driven this way is `push` itself, which is why
    // the assembler's own half of the refractory floor and of the gap seam is
    // tested in `window.rs`.

    /// A diarizer with no models behind it.
    fn headless(tuning: DiarizeTuning) -> RealDiarizer {
        RealDiarizer::new(
            Models {
                vad: PathBuf::from("/nonexistent/silero_vad.onnx"),
                embed: PathBuf::from("/nonexistent/embed.onnx"),
                norm: Norm::Mean,
            },
            tuning,
        )
    }

    /// Embeddings live in four dimensions here: enough for two orthogonal
    /// voices and the vector exactly between them.
    fn axis(i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 4];
        v[i] = 1.0;
        v
    }

    /// Give the assembler enough voiced audio that a boundary cut is inside
    /// the refractory floor. The samples themselves never matter -- these
    /// tests hand `hop` and `window` their embeddings directly.
    fn fill_the_accumulator(d: &mut RealDiarizer) {
        let framed = crate::diarize::window::Framed {
            first_frame: 0,
            samples: vec![0.1f32; WINDOW_SAMPLES],
        };
        d.assembler.push(&framed, &[true; WINDOW_SAMPLES / FRAME]);
    }

    /// Mint a speaker on `voice` by handing the clusterer windows directly,
    /// leaving the diarizer's own bookkeeping untouched.
    fn mint(d: &mut RealDiarizer, voice: &[f32]) {
        for _ in 0..crate::diarize::cluster::MIN_POOL {
            d.clusterer.observe(voice);
        }
    }

    #[test]
    fn the_first_hop_of_a_session_is_a_seam_a_correction_cannot_cross() {
        // A meeting opens with somebody saying "Right." -- under 0.736 s, so
        // no hop ever completes on it and it is committed with no name. The
        // next speaker's first hop has nothing before it to be compared
        // against, so no turn change is detected there and none could be. If
        // "no boundary detected" were read as "same speaker", the correction
        // that names the second speaker would put their name on the first
        // speaker's line.
        let mut d = headless(DiarizeTuning::default());
        // Chunks 0 and 1 answered with nobody, exactly as `push` records it.
        d.correctable_since.get_or_insert(0);
        d.correctable_since.get_or_insert(1);

        let mut out = Attribution::default();
        assert!(
            !d.hop(axis(1), 2, &mut out),
            "there is nothing for a first hop to have changed from"
        );
        assert_eq!(
            d.correctable_since,
            Some(2),
            "the first hop has to end the run it cannot vouch for"
        );

        // Four windows of the new voice, and the mint that names them.
        let mut relabels = Vec::new();
        for chunk in 3..7 {
            let mut out = Attribution::default();
            d.window(&axis(1), chunk, &mut out);
            relabels.extend(out.relabels);
        }
        assert_eq!(
            relabels,
            vec![Relabel {
                from_chunk: 2,
                to_chunk: 6,
                speaker: 1
            }],
            "the correction reached back over somebody else's words"
        );
    }

    #[test]
    fn a_correction_never_reaches_back_over_a_settled_label() {
        // The other half: a window that settles a label ends the correctable
        // run, so text a full window placed is never up for reassignment
        // however far the next mint would like to reach.
        let mut d = headless(DiarizeTuning::default());
        mint(&mut d, &axis(0));

        let mut out = Attribution::default();
        d.window(&axis(0), 5, &mut out);
        assert_eq!(out.speaker, None, "`window` reports through `push`");
        assert_eq!(d.last_label, Some(1));
        assert_eq!(d.correctable_since, None, "a settled label ends the run");

        // Two chunks with nothing in them, then a second voice arriving.
        d.correctable_since.get_or_insert(6);
        let mut relabels = Vec::new();
        for chunk in 8..12 {
            let mut out = Attribution::default();
            d.window(&axis(1), chunk, &mut out);
            relabels.extend(out.relabels);
        }
        assert_eq!(
            relabels,
            vec![Relabel {
                from_chunk: 6,
                to_chunk: 11,
                speaker: 2
            }]
        );
    }

    #[test]
    fn an_ambiguous_window_offers_a_provisional_label_rather_than_a_gap() {
        // What the clusterer withholding an assignment costs, and what it does
        // not. The window names nobody confidently -- so it moves no centroid,
        // which is the drift protection -- but the argmax still reaches the
        // reader, marked as correctable, and the chunk stays inside the run a
        // later mint may rewrite.
        let mut d = headless(DiarizeTuning::default());
        mint(&mut d, &axis(0));
        mint(&mut d, &axis(1));

        let between: Vec<f32> = vec![0.5f32.sqrt(), 0.5f32.sqrt(), 0.0, 0.0];
        let mut out = Attribution::default();
        d.window(&between, 4, &mut out);
        assert!(
            out.relabels.is_empty(),
            "a guess corrects nothing by itself"
        );
        assert!(
            matches!(d.provisional, Some((1 | 2, 4))),
            "an ambiguous window should offer its best guess: {:?}",
            d.provisional
        );
        assert_eq!(d.last_label, d.provisional.map(|(l, _)| l));
        assert_eq!(
            d.correctable_since,
            Some(4),
            "a guess is text a mint may still take back"
        );
    }

    #[test]
    fn a_change_threshold_of_zero_detects_no_turn_change() {
        // Documented as the way to switch detection off, and it has to be
        // checked for rather than reached by arithmetic: different-speaker
        // cosines are routinely negative, so `cosine < 0` fires constantly and
        // no value of a bare threshold would ever mean "never".
        let off = DiarizeTuning {
            change_threshold: 0.0,
            ..Default::default()
        };
        let mut d = headless(off);
        fill_the_accumulator(&mut d);
        let mut out = Attribution::default();
        d.hop(axis(0), 0, &mut out);
        let opposite: Vec<f32> = axis(0).iter().map(|x| -x).collect();
        assert!(!d.hop(opposite.clone(), 1, &mut out));
        assert!(!out.boundary, "a cosine of -1 was called a turn change");

        // The same pair at the shipped threshold, so the difference is the
        // configuration and not the fixture.
        let mut d = headless(DiarizeTuning::default());
        fill_the_accumulator(&mut d);
        let mut out = Attribution::default();
        d.hop(axis(0), 0, &mut out);
        assert!(d.hop(opposite, 1, &mut out));
        assert!(out.boundary);
    }

    #[test]
    fn a_boundary_the_accumulator_refuses_is_not_a_boundary_at_all() {
        // The refractory floor lives in `window.rs`, but acting on a boundary
        // it declined would be worse than not detecting one: the label would
        // be blanked and the session's vote clipped for a turn change no
        // window can ever confirm, since no window would complete.
        let mut d = headless(DiarizeTuning::default());
        mint(&mut d, &axis(0));
        let mut out = Attribution::default();
        d.hop(axis(0), 0, &mut out);
        d.last_label = Some(1);

        // Nothing in the accumulator, so a cut has no room to leave a window
        // behind it.
        let opposite: Vec<f32> = axis(0).iter().map(|x| -x).collect();
        assert!(!d.hop(opposite, 1, &mut out));
        assert!(!out.boundary);
        assert_eq!(d.last_label, Some(1), "a refused cut blanked the label");
    }

    // --------------------------------------------------- the deferred seam
    //
    // Half a second of quiet is constant in conversation, and treating every
    // one of them as a provenance seam is what held the backfill to 2
    // corrections over 21 minutes of ES2002a. These drive the state machine
    // that measures the silence instead of assuming about it; `window.rs`
    // owns the half that decides when the accumulator restarted at all.

    /// A unit vector at exactly `cos` from `axis(0)`: the two thresholds a
    /// pair of hops can fall between are 0.30 and 0.45, so the interesting
    /// fixtures are the ones in there.
    fn at_cosine(cos: f32) -> Vec<f32> {
        vec![cos, (1.0 - cos * cos).sqrt(), 0.0, 0.0]
    }

    #[test]
    fn a_silence_the_same_voice_resumes_after_lets_a_correction_reach_through_it() {
        // The complaint this exists for: a correction "doesn't go far enough
        // back -- it might get a few words but it's still missing a couple",
        // because the breath in the middle of the sentence was read as a
        // change of speaker.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        assert_eq!(d.correctable_since, Some(3), "the session's first hop");

        // Half a second of quiet at chunk 10, and the same voice at chunk 13.
        d.gap = Gap::Pending(10);
        let mut out = Attribution::default();
        assert!(!d.hop(axis(0), 13, &mut out));
        assert_eq!(d.gap, Gap::Closed, "the silence has its answer");
        assert_eq!(
            d.correctable_since,
            Some(3),
            "the same person resumed, so the reach should survive the pause"
        );

        // And the mint that finally names them reaches over it.
        let mut relabels = Vec::new();
        for chunk in 14..18 {
            let mut out = Attribution::default();
            d.window(&axis(0), chunk, &mut out);
            relabels.extend(out.relabels);
        }
        assert_eq!(
            relabels,
            vec![Relabel {
                from_chunk: 3,
                to_chunk: 17,
                speaker: 1
            }]
        );
    }

    #[test]
    fn a_silence_a_different_voice_follows_holds_the_seam_at_the_pause() {
        // The bug the seam exists for, and the one this change must keep
        // impossible: A talks and is never minted, so their words are
        // committed with nobody's name on them; a pause; B talks and *is*
        // minted. If the correction reached back over the pause it would
        // rename A's sentence to B, which the protocol promises cannot happen.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);

        d.gap = Gap::Pending(10);
        let mut out = Attribution::default();
        assert!(!d.hop(axis(1), 13, &mut out));
        assert_eq!(
            d.correctable_since,
            Some(13),
            "a different voice resumed, so the pause is a seam after all -- \
             drawn at the hop that measured the change, not at the pause"
        );

        let mut relabels = Vec::new();
        for chunk in 14..18 {
            let mut out = Attribution::default();
            d.window(&axis(1), chunk, &mut out);
            relabels.extend(out.relabels);
        }
        assert_eq!(
            relabels,
            vec![Relabel {
                from_chunk: 13,
                to_chunk: 17,
                speaker: 1
            }],
            "the correction reached back over somebody else's words"
        );
    }

    #[test]
    fn a_mint_while_the_silence_is_unanswered_gets_the_cautious_answer() {
        // Reachable when the hop that would have answered failed to embed,
        // which `Fuse` exists to survive: the window that mints arrives with
        // the pause still unaccounted for, and there is no evidence to be had
        // in time. A correction may not cross a silence nobody has vouched
        // for, so the seam is committed rather than waited on.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        d.gap = Gap::Pending(10);

        let mut relabels = Vec::new();
        for chunk in 14..18 {
            let mut out = Attribution::default();
            d.window(&axis(0), chunk, &mut out);
            relabels.extend(out.relabels);
        }
        assert_eq!(
            relabels,
            vec![Relabel {
                from_chunk: 10,
                to_chunk: 17,
                speaker: 1
            }],
            "a correction crossed a silence nothing had answered for"
        );
        assert_eq!(
            d.gap,
            Gap::Assumed,
            "the hop still has to know it follows a pause"
        );
    }

    #[test]
    fn a_silence_with_no_hop_before_it_is_a_seam() {
        // A meeting opening with "Right." -- under 0.736 s, so no hop ever
        // completes on it -- and then a pause. There is nothing on the near
        // side of the silence to compare the far side against, so no evidence
        // can arrive and the seam stands. It stands at the hop rather than at
        // the pause, which is the tighter of the two and exactly where the
        // first hop of a session puts it anyway.
        let mut d = headless(DiarizeTuning::default());
        d.correctable_since = Some(0);
        d.gap = Gap::Pending(4);

        let mut out = Attribution::default();
        assert!(!d.hop(axis(1), 7, &mut out));
        assert_eq!(d.correctable_since, Some(7));
    }

    #[test]
    fn a_deferred_seam_never_reopens_a_run_a_settled_label_ended() {
        // The seam is applied as a floor rather than written in, and this is
        // why: chunks go by between the pause and the answer, and one of them
        // may carry a window that settles a label. Writing the pause's chunk
        // in would let a later correction reach back over text a full window
        // placed, which is the one thing `session.rs` cannot catch on its own.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        mint(&mut d, &axis(0));
        d.gap = Gap::Pending(10);

        let mut out = Attribution::default();
        d.window(&axis(0), 12, &mut out);
        assert_eq!(d.correctable_since, None, "a settled label ends the run");

        let mut out = Attribution::default();
        d.hop(axis(1), 13, &mut out);
        assert_eq!(
            d.correctable_since, None,
            "a deferred seam reopened a run a settled label had ended"
        );
    }

    #[test]
    fn the_silence_and_the_hop_boundary_are_judged_at_different_thresholds() {
        // A pair halfway between the two thresholds: alike enough to be one
        // person talking on from one hop to the next, not alike enough to
        // carry somebody's name across half a second of silence. Two 0.75 s
        // embeddings with a pause between them are the noisiest comparison
        // this pipeline makes, and the error that costs a sentence its author
        // is not the error that costs a few words of backfill.
        //
        // Read from the shipped values rather than written down, so that
        // retuning either one re-aims this test instead of rotting it.
        let shipped = DiarizeTuning::default();
        let voice = at_cosine(0.5 * (shipped.change_threshold + shipped.gap_change_threshold));

        let mut d = headless(shipped);
        fill_the_accumulator(&mut d);
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        assert!(
            !d.hop(voice.clone(), 4, &mut out),
            "the pair is above the hop threshold, and the accumulator would \
             have accepted the cut"
        );

        let mut d = headless(shipped);
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        d.gap = Gap::Pending(10);
        d.hop(voice, 13, &mut out);
        assert_eq!(
            d.correctable_since,
            Some(13),
            "the same pair should not carry a correction across a pause"
        );
    }

    #[test]
    fn a_hop_after_a_silence_claims_no_turn_boundary() {
        // What a silence decides is how far a correction reaches, and nothing
        // else. It is a poor turn-change detector -- 48 of 51 decidable breaks
        // had the same speaker either side, which is the measurement behind
        // `window::MAX_GAP_FRAMES` -- so it still never blanks a label or cuts
        // the accumulator, whichever way the comparison goes.
        for gap in [Gap::Pending(10), Gap::Assumed] {
            let mut d = headless(DiarizeTuning::default());
            let mut out = Attribution::default();
            d.hop(axis(0), 3, &mut out);
            fill_the_accumulator(&mut d);
            d.last_label = Some(1);
            d.gap = gap;

            let mut out = Attribution::default();
            assert!(!d.hop(axis(1), 13, &mut out), "{gap:?} claimed a boundary");
            assert!(!out.boundary, "{gap:?}");
            assert_eq!(d.last_label, Some(1), "{gap:?} blanked the label");
        }
    }

    #[test]
    fn a_gap_threshold_of_one_makes_every_silence_a_seam() {
        // The sweep's own control, and the reason it can have one: two 0.75 s
        // embeddings of real audio are never the same vector, so at 1 the
        // pause is committed whatever followed it -- which is the rule that
        // shipped before any of this. The fixture is 0.999 rather than an
        // identical vector for exactly that reason: the identical pair is the
        // one case the corpus cannot produce, and asserting on it would pin
        // the boundary condition instead of the rule.
        let old = DiarizeTuning {
            gap_change_threshold: 1.0,
            ..Default::default()
        };
        let mut d = headless(old);
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        d.gap = Gap::Pending(10);
        d.hop(at_cosine(0.999), 13, &mut out);
        assert_eq!(d.correctable_since, Some(13));
    }

    #[test]
    fn switching_change_detection_off_leaves_every_silence_a_seam() {
        // `diarize_change_threshold = 0` is documented as the way to stop any
        // cosine between two hops deciding anything, and how far a correction
        // reaches is something decided. A deployment that has taken that hatch
        // gets the rule that shipped before the deferral existed, rather than
        // a second comparison it never asked for.
        let off = DiarizeTuning {
            change_threshold: 0.0,
            ..Default::default()
        };
        let mut d = headless(off);
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        d.gap = Gap::Pending(10);
        d.hop(axis(0), 13, &mut out);
        assert_eq!(d.correctable_since, Some(13));
    }

    #[test]
    fn a_seam_committed_early_is_still_tightened_by_the_hop_that_arrives_late() {
        // `Gap::Assumed` means a correction went out while the answer was
        // outstanding, so the bound at the pause has been given out and cannot
        // be loosened. It can still be *tightened*, and when the voices turn
        // out to differ it must be: the rule this replaced cleared the held
        // hop, so the post-gap hop seamed as the first of a run -- at itself,
        // which is a chunk later than the pause. Leaving it at the pause
        // exposes exactly the chunk `hop` refuses to draw a bound at, the one
        // that can carry both speakers' words.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        d.gap = Gap::Pending(10);
        d.assume_gap_hid_a_change();
        assert_eq!(d.gap, Gap::Assumed);
        assert_eq!(d.correctable_since, Some(10), "the bound handed out");

        let mut out = Attribution::default();
        d.hop(axis(1), 13, &mut out);
        assert_eq!(
            d.correctable_since,
            Some(13),
            "the evidence said somebody else and the bound stayed on the chunk \
             the silence began in"
        );
    }

    #[test]
    fn a_seam_committed_early_is_never_loosened_by_it() {
        // The other direction, and the one that is not open: a correction was
        // emitted against the bound at the pause, so evidence arriving
        // afterwards to say the same voice resumed cannot buy the reach back.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);
        d.gap = Gap::Pending(10);
        d.assume_gap_hid_a_change();

        let mut out = Attribution::default();
        d.hop(axis(0), 13, &mut out);
        assert_eq!(d.correctable_since, Some(10));
    }

    #[test]
    fn a_piece_that_fails_to_embed_still_records_the_silence_it_arrived_with() {
        // The ordering `batch` exists for. A batch can both cut a piece and
        // report a restart -- `window.rs` proves the precondition against the
        // assembler's own API -- and if the piece fails to embed on the way
        // through, the silence has to survive it. `WindowAssembler::restarted`
        // is cleared on the next push, so a gap lost here is lost for good:
        // not committed, not pending, not assumed, and the next hop compared
        // as though it adjoined the one before the pause.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(axis(0), 3, &mut out);

        let mut out = Attribution::default();
        let err = d
            .batch(
                vec![Cut::Hop(vec![0.1; HOP_SAMPLES])],
                true,
                10,
                &mut out,
                |_| bail!("the embedder is having a moment"),
            )
            .expect_err("the piece did not embed");
        assert!(format!("{err:#}").contains("having a moment"), "{err:#}");
        assert_eq!(
            d.gap,
            Gap::Pending(10),
            "the silence vanished with the embedding that failed"
        );

        // And the seam it was recorded for still lands: a different voice on
        // the far side of the pause commits it.
        let mut out = Attribution::default();
        d.hop(axis(1), 13, &mut out);
        assert_eq!(d.correctable_since, Some(13));
    }

    #[test]
    fn a_comparison_that_cannot_be_made_commits_the_seam() {
        // `cosine` is a bare dot product, so a non-finite embedding makes
        // `cos < threshold` false -- "same voice, discard the seam", which is
        // the unsafe direction. `real::embed` refuses non-finite vectors, so
        // this is an invariant two modules away being asserted where it is
        // relied on rather than a case that arises.
        let mut d = headless(DiarizeTuning::default());
        let mut out = Attribution::default();
        d.hop(vec![f32::NAN; 4], 3, &mut out);
        d.gap = Gap::Pending(10);

        let mut out = Attribution::default();
        d.hop(axis(0), 13, &mut out);
        assert_eq!(
            d.correctable_since,
            Some(13),
            "a comparison that answered NaN was read as \"the same person\""
        );
    }

    #[test]
    fn a_chunk_that_is_not_shorter_than_a_hop_is_refused() {
        // The arithmetic the deferral rests on, checked rather than assumed:
        // the chunk length is `AsrBackend::chunk_samples()`, and at 0.8 s it
        // is longer than a hop. A hop could then complete on the far side of a
        // silence inside one push -- it would be compared at the looser
        // `T_CHANGE`, could claim a boundary `MAX_GAP_FRAMES` declines to
        // claim, and the gap recorded afterwards would point at a silence
        // whose far side had already gone by. All of it silent.
        let mut d = headless(DiarizeTuning::default());
        let err = d
            .push(&vec![0.0; HOP_SAMPLES])
            .expect_err("a chunk as long as a hop");
        let message = format!("{err:#}");
        assert!(message.contains("shorter than a hop"), "{message}");

        // Before the lazy load, so asking costs nothing and the answer does
        // not depend on a model directory: the shipped 0.56 s chunk gets past
        // this check and fails on the models instead.
        let err = d.push(&[0.0; 8960]).expect_err("no models to load");
        assert!(
            format!("{err:#}").contains("diarization models"),
            "the chunk-length check moved below the load: {err:#}"
        );
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
