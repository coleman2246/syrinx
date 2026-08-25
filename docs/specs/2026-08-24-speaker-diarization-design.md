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
diarize_model_dir = "/home/you/models/diarize"   # silero-vad + embedding ONNX; no tilde expansion
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
fragments). **Save as…** and `stream_to` write the same attribution — the
streamed file is what the LLM consumes — though where it goes differs by
format, below.

Composition with the existing formats, when labels are present:

- **plain** — a `Speaker 2: ` prefix where a turn starts, prose otherwise.
  Without labels the format is byte-identical to today.
- **timestamped** — timestamp, then speaker, then text:
  `[12:04] Speaker 2: ...`
- **labelled** — time, source, then speaker, where the source is the
  capture's existing `short_label()` — the device, application, or
  microphone name exactly as the format prints it today:
  `[12:04] [System audio] Speaker 2: ...` or `[12:04] [Yeti] Speaker 1: ...`.
  Diarization adds the speaker; it does not rename the source.
  Speaker numbering is **per-session**: with `SourceMode::Separate`, each
  source's session mints its own independent Speaker 1..N, and the source
  label is what distinguishes them. (For meetings this hardly matters —
  diarization is aimed at a single mixed capture.)

How often the prefix appears follows from what a line means in each format.
**timestamped** and **labelled** are one record per line — a line breaks on
a pause of `NEW_LINE_AFTER_SILENCE`, and whatever reads the file back has
that line and nothing else — so **every** line names a speaker, including
one merely reopened by a pause inside a turn. **plain** is prose, where the
paragraph is the record: the speaker is named once where the turn starts,
because repeating it every time somebody drew breath would wreck the
reading.

The name is the turn's, not just the labelling commit's: a line opened by
an unlabeled fragment inside an established turn carries that turn's
speaker, which is already what the paragraph view claims about it. The
exception is the start of a session, before any voice has been given a
number — there is nothing honest to attribute those lines to, and the
streamed file is append-only, so the first label to arrive is never applied
backwards over what is already on disk.

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

## Spike results

**Date:** 2026-08-24. **Verdict: go.** Measured in `spike/diarize`, which has
since graduated into the tree: the pipeline is `syrinx-server`'s own
`diarize` module, and the harness that produced every number below is
`crates/syrinx-server/examples/diarize_probe` (`cargo run --release -p
syrinx-server --features diarize --example diarize_probe -- <subcommand>`).
The subcommand names are unchanged, so each citation below still names the
command that reproduces it.

### What was run

The pipeline runs the whole design end to end — silero VAD, sliding voiced
windows, a hand-rolled Kaldi-compatible fbank, an ONNX speaker embedder, and
the clusterer from "Stage 3" above — over wav files, scored against the AMI
manual annotations (v1.6.2, word-level times, so pauses inside a turn are not
counted as speech).

Recordings, all AMI `Mix-Headset` at 16 kHz mono (corpus CC-BY-4.0):

| Recording | Length | Speakers | Notes |
|---|---|---|---|
| ES2002a | 21 min | 4 | very skewed: 446 s / 346 s / 76 s / 22 s |
| IS1000a | 26 min | 4 | balanced: 623 s / 196 s / 176 s / 121 s |
| EN2001a | 87 min | 5 | the long-run stability test |

Embeddings depend only on (model, recording, window, hop), never on the
clustering thresholds, so the binary caches them to disk. A 2640-configuration
sweep over `T_assign` × `T_retire` × `MIN_POOL` × window × model × recording
(1920 originally; the `T_assign` grid was later widened through 0.70)
therefore costs seconds, and every number below is sweep output rather than a
hand-picked run.

### The front-end had to be validated first

A wrong fbank does not fail loudly — it produces embeddings that look
plausible and separate nobody. `diarize_probe verify` checks each model
against the same-speaker / different-speaker wav pairs shipped in the
sherpa-onnx release. All three candidates cleared it (same-speaker mean
0.66, different-speaker mean 0.15–0.29, no overlap between the
distributions), which is what licenses trusting the meeting numbers.

Two front-end details cost real time and are worth recording: Kaldi's mel
bank runs to Nyquist (8000 Hz), not the 7600 Hz speech-synthesis convention;
and **silero v5 wants 576 samples per call, not 512** — 64 samples of
context from the previous frame prepended to the new 512. Feeding it a bare
512 returns near-zero speech probability for everything, silently.

### The embeddings are better than the design assumed, at a lower threshold

Measured over windows the reference marks as a single clean speaker
(`diarize_probe separability`, ES2002a, 1.5 s windows):

| Model | same-speaker p50 | different-speaker p50 | best split | pairs wrong |
|---|---|---|---|---|
| WeSpeaker ResNet34-LM | 0.522 | 0.026 | 0.24 | 2.3% |
| 3D-Speaker ERes2Net | 0.517 | 0.046 | 0.24 | 2.0% |
| NeMo TitaNet-small | 0.558 | 0.196 | 0.36 | 7.2% |

TitaNet separates 3× worse than the other two and was dropped after this
measurement rather than carried through a full sweep.

The important calibration finding: **the design's `T_assign ≈ 0.6` guess was
wrong for these models on this audio.** Same-speaker pairs sit at a median
of 0.52, so 0.6 rejects most true matches. The sweep covers the 0.40–0.70
the plan asked for and extends down to 0.20, which is where the answer
turned out to live; the usable plateau is 0.40–0.50.

The high end fails quietly rather than loudly, which is worth recording
because splits and merges alone will not reveal it. At the chosen model,
window, and pool, raising `T_assign` from 0.45 to 0.60 leaves splits and
merges at zero — but only because the clusterer stops speaking: miss climbs
from 26% to 37% on ES2002a and from 31% to 58% on IS1000a, and ES2002a
collapses to 2 labels for its 4 speakers. By 0.70 the miss rate is 51–82%
and every meeting is down to 2 or 3 labels. A configuration that says
almost nothing scores perfectly on stability, so coverage has to be read
alongside it.

### Chosen model and constants

**Model: `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`** (26.5 MB,
192-dim), from the sherpa-onnx model zoo:
`https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`
(the release tag really is spelled `recongition`). Apache-2.0, from the
3D-Speaker project. VAD is `silero_vad.onnx` v6.2.1 (2.2 MB, MIT) —
originally recorded here as "v5", but the spike fetched from silero's
`master`, which carried v6.2 weights by then; the hash matches the v6.2.1
tag, so that is what every number in this section was measured with. The
512+64-sample interface is unchanged since v5.

| Constant | Value | Why this value |
|---|---|---|
| window | 1.5 s | 1.0 s finds too few speakers, 2.0 s misses 5% more speech |
| hop | 0.75 s | half the window |
| `T_assign` | 0.45 | middle of the 0.40–0.50 plateau |
| `T_retire` | 0.80 | ≤0.60 merges real speakers; see below |
| EMA alpha | 0.05 | insensitive: 0.02–0.20 moves nothing by more than 1.5% |
| `MIN_POOL` | 4 | 2 is catastrophic; 3 fails the long meeting |
| `LAG_CHUNKS` | 2 | ≈1.12 s, covers the p90 label delay |

The last two later became server config keys — `diarize_min_pool` and
`diarize_lag_chunks` — defaulting to the values above, so a deployment can
trade latency and pickup speed against the stability measured here. The
numbers in this section are the measurements, not the only settings that
will ever run.

`MIN_POOL` deserves emphasis, because it is the design's "reluctant create"
rule earning its keep. All figures here are at the chosen model, window
(1.5 s), `T_assign` (0.45) and `T_retire` (0.80):

| `MIN_POOL` | ES2002a (4) | IS1000a (4) | EN2001a (5) |
|---|---|---|---|
| 2 | 6 labels, 1 merge | 20 labels, 1 split, 6 merges | 15 labels, 5 merges |
| 3 | 4 ✓ | 4 ✓ | 6 labels, 1 merge |
| 4 | 4 ✓ | 4 ✓ | 5 ✓ |

**Only 4 is correct on every recording** — 3 already fails the long meeting,
so it is not merely "close to the cliff". (Shortening the window to 1.0 s
makes `MIN_POOL` 2 far worse still, at 29, 42 and 55 labels, but 1.5 s is
what was chosen.) The design's instinct here was right, and the setting is
not a free parameter.

`T_retire` is the opposite story: **it never fired in any accepted
configuration** — no centroid was ever retired at the chosen constants. Its
only measured effect is harm when set too low: on EN2001a, `T_retire ≤ 0.60`
retires two genuinely different speakers into one, taking confusion from
2.0% to 4.8%. The cliff is between 0.60 and 0.65. 0.80 clears it with margin
while staying low enough to plausibly catch a real split. Its value as
insurance is unmeasured, because no split ever happened to insure against.

### Results at those constants

Splits count extra labels holding ≥10% of a real speaker's speech; merges
count labels covering ≥10% of more than one speaker. Miss is single-speaker
speech left unlabelled; confusion is speech attributed to the wrong speaker
under the best one-to-one mapping. Overlapped speech is excluded from
scoring, per "out of scope" above.

| Recording | Real | Labels | Splits | Merges | Miss | Confusion |
|---|---|---|---|---|---|---|
| ES2002a | 4 | 4 | 0 | 0 | 26.2% | 2.2% |
| ES2002a, Opus 24 kbps | 4 | 4 | 0 | 0 | 27.8% | 2.2% |
| ES2002a, Opus 16 kbps | 4 | **3** | 0 | 0 | 29.8% | 2.4% |
| IS1000a | 4 | 4 | 0 | 0 | 31.3% | 2.6% |
| IS1000a, Opus 24 kbps | 4 | 4 | 0 | 0 | 31.5% | 2.4% |
| IS1000a, Opus 16 kbps | 4 | 4 | 0 | 0 | 32.8% | 2.1% |
| EN2001a (87 min) | 5 | 5 | 0 | 0 | 12.5% | 2.0% |
| EN2001a, Opus 24 kbps | 5 | 5 | 0 | 0 | 13.2% | 2.0% |

WeSpeaker ResNet34-LM matches this on confusion to within 0.4 points and
runs 0–2 points higher on miss, with one real difference: on ES2002a under
Opus it loses that meeting's 76-second speaker at 24 kbps, where ERes2Net
keeps them. That single case is the whole margin between the two models —
they are otherwise interchangeable, single-threaded cost included (50 ms
versus 45 ms per window), and swapping them is a one-line change.

**Label stability, the spec's main worry:** on the 87-minute meeting every
one of the five speakers holds the same label across all three thirds of the
recording, under both models, clean and Opus-degraded. No renumbering, no
drift, no late splits.

**Turn boundaries:** when the diarizer starts a new turn, the median
distance to a real turn change is 0.49–0.97 s on clean audio, with 55–72%
inside 1 s; the worst case anywhere, ES2002a at Opus 24 kbps, is 1.01 s and
47%. The reverse direction is much worse (median 0.6–4.5 s) but measures
something else — short turns the diarizer never labels at all, which is the
miss rate, not a boundary error.

**Cost**, single-threaded, measured by `diarize_probe bench`: silero VAD
0.079 ms per 32 ms frame (0.2% of a core), ERes2Net 45 ms per window at a
0.75 s hop (6.0% of a core). Total ≈6% of one core per session, which
matches the "a few percent of one core" claim in the CPU-budget note above.

**Lag:** measured against real window timings rather than derived from the
hop, the delay between a 560 ms ASR chunk ending and a label covering it
existing is p50 0.42–0.45 s, p90 0.88–1.09 s, p99 ≈1.5 s. Two chunks
(1.12 s) covers the p90; three (1.68 s) covers the p99. Two is the
recommendation, with `speaker: None` absorbing the rest — which is what that
field is for.

### Verdict

**Go.** The pipeline finds exactly the right number of speakers on seven of
the eight conditions tested, attributes 96–98% of the speech it does label
to the right person, and holds every label stable over 87 minutes. The
conservative clustering rules the design specified are doing real work:
`MIN_POOL` at 4 is the difference between 4 labels and 20. Nothing here
argues for changing the architecture, and the two headline risks recorded
above both came out better than feared — Opus at Teams-like bitrates costs
about one point of miss rate, and multi-hour stability was perfect.

### What this does not establish

Honesty about the gaps, in rough order of how much they should worry Phase 2:

- **No real Teams audio was tested.** Opus at 24 and 16 kbps is a proxy for
  the codec, not for the acoustic path — Teams also brings automatic gain
  control, noise suppression, and speakerphone-in-a-meeting-room, none of
  which this measures. AMI `Mix-Headset` is a mix of close-talking headset
  microphones and is *cleaner* than anything the server will really see. A
  deliberate Teams calibration recording is still worth making before the
  constants are treated as final.
- **Five speakers, not ten.** The problem statement says three to ten
  participants; AMI gave at most five. The crowding measurement is the
  warning sign here: with five speakers on EN2001a the two closest centroids
  sit at 0.519 cosine — *above* `T_assign`. Separation survived because
  assignment goes to the nearest centroid rather than the first one over the
  threshold, but the margin is gone at five speakers, and eight is untested.
  If a meeting is going to fail, this is how.
- **Quiet participants are the fragile case.** ES2002a's fourth speaker
  (22 s of speech in 21 minutes) is never reliably labelled, and its
  76-second speaker is the one Opus 16 kbps loses. Someone who says three
  sentences in an hour may simply not get a label.
- **The miss rate is high and is a design consequence, not a bug.** Roughly
  a quarter to a third of single-speaker speech gets no label in the
  conversational meetings, because a window needs 1.5 s of voiced audio and
  many turns are shorter. It falls to 12.5% on the meeting with long turns.
  Backchannels and one-word interjections will mostly arrive as
  `speaker: None`.
- **Overlapped speech is excluded from every number above**, so the real
  transcript will look worse than 2% confusion during cross-talk.
