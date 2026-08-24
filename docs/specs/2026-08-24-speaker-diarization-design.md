# Speaker diarization for transcribe mode

**Date:** 2026-08-24
**Status:** Approved design, pre-implementation

## Problem

In meetings — Microsoft Teams over system audio, three to ten participants —
the transcript is one undifferentiated stream of text. The user wants each
stretch of transcript attributed to an anonymous speaker label ("Speaker 1",
"Speaker 2", …) so that an LLM reading the transcript afterwards can work out
who said what and attach names from context.

Constraints that shaped the design:

- **No Teams integration.** Corporate IT rules out the Teams APIs entirely.
  The only input is the audio stream syrinx already captures.
- **Live, no recordings.** Labels appear as the transcript streams. The user
  will not record meeting audio for post-processing; the transcript text is
  the only artifact.
- **Label stability preferred.** Once someone is Speaker 2 they should stay
  Speaker 2. Occasional splits (one person acquiring a second label) are
  acceptable; the downstream LLM can reconcile them. Rough turn boundaries
  are acceptable.
- **Transcribe mode only.** Type-at-cursor never sees labels — typing
  "Speaker 2:" at the cursor would be wrong. Live mode is unaffected.
- **The GPU tenancy policy holds.** The server is the lowest-priority tenant
  on a shared 8 GB card. Diarization must not add VRAM pressure.

The chosen approach diarizes the **single mixed mono stream the server
already receives**. No client or capture changes: every client (GUI, CLI,
iOS) gets the feature with a protocol field, and it works regardless of how
the audio was captured. The consequence, accepted explicitly: the user's own
voice is just another Speaker N, not a distinguished "Me" — though in
practice the microphone's acoustic path differs enough from Teams' compressed
downlink that the user tends to be the most reliably clustered voice.
A client-side mic/system channel split (which would have given an exact "Me"
label) was considered and rejected in favour of keeping clients unchanged.

## Architecture

The diarizer is a sidecar on the server's session pipeline. The ASR path is
untouched; the same 16 kHz mono samples that are buffered into 560 ms ASR
chunks are also fed to a diarization pipeline running alongside:

```
client audio ──► Session::push_audio
                    ├──► ASR chunks (560 ms) ──► Nemotron ──► text
                    └──► Diarizer
                           ├─ VAD (silero-vad ONNX, ~2 MB) — is anyone speaking?
                           ├─ sliding voiced window (~1.5 s, ~0.75 s hop)
                           │     ──► speaker-embedding model (ONNX, CPU)
                           └─ online clustering (cosine to running centroids)
                                 ──► per-chunk speaker label
                    then: commit text + majority label over the chunk's audio
                          ──► transcript.commit { seq, text, speaker: 2 }
```

Properties:

- **Chunk-granular attribution.** Every commit corresponds to a known range
  of 560 ms chunks; the diarizer answers "who dominated this stretch". No
  token timestamps are needed, so nothing depends on what parakeet-rs does
  or does not expose.
- **CPU only.** VAD and embedding models are tiny (~2 MB + ~20–100 MB) and
  run faster than real time on CPU via `ort` (already in the dependency
  tree). Zero new VRAM; no interaction with the Frigate/Jellyfin tenancy.
- **Live and append-only.** Labels are assigned at commit time and never
  retroactively renumbered. `transcript.revise` stays reserved; a future
  merge-correction pass could use it, but that is out of scope here.
- **Opt-in, transcribe mode only.** The client requests it in
  `session.start`; the server does nothing extra otherwise.
- **Testable without models**, following the `AsrBackend`/`MockBackend`
  pattern: the diarizer sits behind a trait, session-level label attachment
  is tested in CI with a mock, and clustering is tested as pure functions
  over synthetic embedding vectors.

## Protocol and configuration

All changes are backward-compatible additions in `syrinx-proto`. An old
client against a new server, or the reverse, keeps working.

**`session.start`** gains `diarize: bool`, `#[serde(default)]` so absent
means `false`. If diarization is requested but unavailable (models missing,
or live mode), the server proceeds without it rather than failing the
session: dictation working matters more than labels.

**`session.ready`** gains `diarize: bool`, reporting what the session will
actually do — the client can ask and still not receive, and the honest
answer belongs in the handshake. Also `#[serde(default)]`, so a new client
parses an old server's handshake as `diarize: false`. The GUI shows
"speaker labels unavailable" instead of silently producing an unlabeled
transcript. Note that the GUI's "both" output mode runs the wire in live
mode, so its transcript never carries labels — that is the mode working as
designed, not a bug for the GUI plan to fix.

**`transcript.commit` and `transcript.provisional`** gain
`speaker: Option<u32>` with `skip_serializing_if = "Option::is_none"` — the
existing `language`/`vocabulary` pattern. Semantics:

- `Some(n)`: speaker `n`, numbered 1, 2, 3… in order of first confident
  appearance. Rendering ("Speaker 2") is the client's job; the wire carries
  a number so a future rename feature costs nothing.
- `None` on some commits: the diarizer could not decide for that stretch
  (very short utterance, cross-talk, low confidence). The field is omitted
  rather than guessed: an honest gap, not a confident lie.
- `None` on every commit: no diarizer ran.

**Server config** gains one optional key:

```toml
# absent = diarization unavailable, feature off
diarize_model_dir = "~/models/diarize"   # silero-vad + embedding ONNX
```

Same deployment story as the ASR model: not in the repository, not in the
image, volume-mounted read-only, download commands documented in the README.
Loaded alongside the ASR model under the existing lifecycle policy.

## The diarizer

A new `diarize` module in `syrinx-server` (a crate later only if another
consumer appears), behind a trait:

```rust
pub trait Diarizer: Send {
    /// Feed audio, aligned with the ASR chunk stream.
    /// Ok(None) means "no confident label" (silence, cross-talk) and is
    /// normal; Err means the diarizer itself failed. The distinction is
    /// load-bearing: the session counts consecutive errors to decide when
    /// to drop the diarizer, and must not count honest uncertainty.
    fn push(&mut self, audio: &[f32]) -> Result<Option<u32>>;
}
```

Dependency note: `ort` is today only a transitive dependency via
`parakeet-rs`, which is optional behind the `cuda` feature — default and CI
builds contain no `ort`. The real diarizer therefore takes a **direct `ort`
dependency, version-aligned with parakeet-rs's pin** (rc-series versions do
not unify across majors), behind its own cargo feature. The clustering
logic and `MockDiarizer` are pure Rust with no `ort` and stay
unconditional, so CI keeps testing them in the default, CUDA-free build.

**Stage 1 — VAD.** Silero-VAD (ONNX, ~2 MB, via `ort`) classifies ~30 ms
frames as speech or not. Only voiced audio flows downstream: embedding
silence or keyboard noise produces garbage vectors that pollute clusters.
Mostly-silent chunks yield `None`.

**Stage 2 — Embedding windows.** Voiced audio accumulates into a sliding
window of ~1.5 s with a ~0.75 s hop. Each full window is embedded into one
vector (192–256 dims) characterising the voice, not the words. Model
candidates, in spike order: 3D-Speaker ERes2Net, WeSpeaker ResNet34 (both
published as ONNX in the sherpa-onnx model zoo), NeMo TitaNet as fallback.
Window length is the reaction-speed vs. embedding-quality trade-off; 1.5 s
is the standard compromise and is a tunable constant, not architecture.

**Stage 3 — Online clustering.** Running centroids, cosine similarity, and
deliberately asymmetric rules — this is where label stability lives:

- **Assigning is eager:** a new embedding joins its nearest centroid when
  similarity ≥ `T_assign` (~0.6; the spike calibrates it on real audio).
- **Creating is reluctant:** below `T_assign`, the vector enters a
  provisional pool. Only when several consecutive windows (~3–4 s of
  speech) agree with each other and match no existing centroid is
  "Speaker N+1" minted, from the pooled vectors. One cough or cross-talk
  window can never create a speaker.
- **Centroids update slowly:** exponential moving average with small alpha,
  so a centroid is dominated by its history, not its last sentence. This
  keeps Speaker 2 being Speaker 2 in hour three.
- **Never renumber, never merge visibly.** If two centroids drift together,
  the newer is silently retired and future audio assigns to the older; past
  labels stay put. The accepted failure mode is an occasional split, which
  the downstream LLM reconciles.

**Stage 4 — Attribution.** When the ASR emits text for a chunk range, the
label is the majority speaker across embedding windows overlapping that
range. Alignment subtlety: the diarizer needs ~1.5 s of voice before it can
speak, while the ASR emits after 560 ms — so the session holds a lag buffer
between ASR output and emission so the label has caught up by commit time.
The lag depth is a tunable alongside the window length, not a fixed
constant: one chunk (~560 ms) does not strictly cover a window that
overlaps a chunk's tail, and the spike decides where the latency/coverage
trade-off lands (likely one or two chunks; `None`-on-uncertainty covers
whatever the buffer does not). Transcribe-mode latency grows by the buffer
depth — from ~0.6 s to ~1.2–1.8 s, imperceptible in a meeting transcript.
Live mode has no diarizer and keeps its latency.

**Out of scope:** overlapping speech. Two simultaneous voices arrive mixed
into one waveform; the dominant voice wins the window and the other is
lost. That is a physics-of-the-mixed-stream limit, not a fixable bug.

## Error handling

The rule: **diarization failing never costs the transcript.**

- Every diarizer call is fallible-soft: on error, log, emit the commit with
  no `speaker` field, continue. After repeated consecutive failures the
  session drops the diarizer and continues unlabeled — the meeting is still
  transcribed, which is the part that cannot be redone.
- Model-loading failures surface at startup (same spirit as `verify_gpu`):
  `diarize_model_dir` set but models missing or broken → loud log line,
  `diarize: false` in every handshake, server still starts. A dictation
  server that will not boot over an optional feature's model file is the
  wrong trade.
- CPU budget: VAD plus one embedding every 0.75 s is a few percent of one
  core. If concurrent sessions ever made it matter, sessions degrade to
  unlabeled rather than backpressuring audio.

## GUI rendering and file formats

Transcribe mode coalesces consecutive same-speaker commits into paragraphs:

```
Speaker 1  We should ship on Thursday if the build is green.
Speaker 2  The build won't be green by Thursday.
           Marketing already announced it.
Speaker 1  Then Friday.
```

A speaker change starts a new paragraph; unlabeled commits attach to the
current paragraph without breaking it (they are usually short connective
fragments). **Save as…** and `stream_to` write the same shape —
`Speaker 2: text` per turn — since the streamed file is what the LLM
consumes.

Composition with the existing formats, when labels are present:

- **plain** — a `Speaker 2: ` prefix where a turn starts, prose otherwise.
  Without labels the format is byte-identical to today.
- **timestamped** — timestamp, then speaker, then text:
  `[12:04] Speaker 2: ...`
- **labelled** — time, source, then speaker: `[12:04] [teams] Speaker 2: ...`.
  Speaker numbering is **per-session**: with `SourceMode::Separate`, each
  source's session mints its own independent Speaker 1..N, and the source
  label is what distinguishes them. (For meetings this hardly matters —
  diarization is aimed at a single mixed capture.)

## Testing

House pattern: no models in CI.

- **Clustering as pure functions** over synthetic embedding vectors — the
  bulk of the tests. Verify: eager assign, reluctant create, no churn on
  borderline vectors, EMA drift resistance, retire-not-merge, numbering in
  first-appearance order.
- **Session tests with a `MockDiarizer`** alongside `MockBackend`: labels
  attach to the right commits, the lag buffer holds and releases, `None`
  omits the field, mid-session diarizer death degrades to unlabeled without
  dropping text.
- **Proto tests:** round-trips, omitted-when-`None`, old-client
  compatibility (`session.start` without `diarize` parses).
- **Offline evaluation harness, not CI:** a small binary (like the
  `gpu_probe` example) that runs the pipeline over a wav file and prints a
  labeled transcript. Validated against AMI meeting-corpus recordings plus
  one deliberate Teams test recording made once as a calibration fixture —
  a test asset, not a recording workflow.

## Build order

**Step 0 — spike (go/no-go gate).** Standalone binary: VAD → embeddings →
online clustering over wav files, printing labeled output. Run against two
AMI meetings and one Teams test recording. Answers: do the candidate
embedding models separate 4–8 Teams-compressed voices well enough, under
the conservative clustering, to be worth shipping? Model choice and
threshold calibration happen here, against real audio. If the spike
disappoints, the cost was roughly a day and the protocol is untouched.

**Phase 1 — plumbing (no research risk).** Proto fields, session lag
buffer and label attachment, `MockDiarizer`, config key, handshake
reporting, GUI paragraph rendering, `stream_to` format. All CI-testable
without models. Ships inert behind the `diarize = false` default.

**Phase 2 — the real diarizer.** Port the spike pipeline into the
`Diarizer` trait, model loading and startup verification, the evaluation
harness, README model-download section mirroring the ASR one. Turn it on.

## Known risks

- **Multi-hour label stability** with rarely-speaking participants is the
  one real uncertainty. Conservative thresholds carry most of the load; the
  spike measures how far. Worst realistic case: occasional speaker splits,
  reconciled downstream — explicitly accepted.
- **Embedding quality on Teams-compressed audio.** Codec artifacts blur
  voice characteristics relative to clean corpora. This is exactly what the
  spike exists to measure before any integration work is spent.
