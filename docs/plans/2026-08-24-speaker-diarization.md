# Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live anonymous speaker labels ("Speaker 1", "Speaker 2", …) on transcribe-mode commits, per the approved spec at `docs/specs/2026-08-24-speaker-diarization-design.md`.

**Architecture:** A server-side sidecar on the session pipeline: VAD → sliding-window speaker embeddings → conservative online clustering, attributed at ASR-chunk granularity, carried as an optional `speaker` field on transcript messages. The ASR path is untouched. Phase 1 plumbing is inert and CI-tested with mocks; the real diarizer arrives in Phase 2 behind a cargo feature, gated on a spike.

**Tech Stack:** Rust workspace. New: `ort` (direct dep, version-matched to parakeet-rs's pin), silero-VAD ONNX, a speaker-embedding ONNX model (spike chooses between 3D-Speaker ERes2Net / WeSpeaker ResNet34 / TitaNet), `hound` for wav in the harness.

**Read the spec first.** Every design decision below is justified there; this plan only sequences it.

**House rules observed throughout:**
- Commit messages are prose in the repo's style ("Draw the meter on the display link, not on the socket") — no `feat:`/`fix:` prefixes.
- Comments state constraints the code can't show; match the codebase's density.
- No models, no GPU, no network in CI tests. `cargo test --workspace` must pass at every commit with no features enabled.
- Run `cargo fmt` and `cargo clippy --workspace` before each commit.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `spike/diarize/` | create (own crate, **not** a workspace member) | Go/no-go evaluation harness; later mined for Phase 2 |
| `crates/syrinx-proto/src/message.rs` | modify | `diarize` on start/ready, `speaker` on commit/provisional |
| `crates/syrinx-server/src/diarize/mod.rs` | create | `Diarizer` trait, `DiarizerFactory` trait, `MockDiarizer` (all unconditional, no `ort`) |
| `crates/syrinx-server/src/diarize/cluster.rs` | create (Phase 2) | Pure online clusterer, unconditional, fully CI-tested |
| `crates/syrinx-server/src/diarize/real.rs` | create (Phase 2, feature `diarize`) | VAD + embedding + clusterer assembled behind the trait |
| `crates/syrinx-server/src/session.rs` | modify | Lag buffer, label attachment, strike-out on diarizer errors |
| `crates/syrinx-server/src/ws.rs` | modify | Parse `diarize` request, report honest `diarize` in ready, pass diarizer into `Session` |
| `crates/syrinx-server/src/config.rs` | modify | `diarize_model_dir: Option<String>` |
| `crates/syrinx-server/src/main.rs` | modify (Phase 2) | Build the real factory when configured |
| `crates/syrinx-proto` / client call sites | modify | Compile fixes where messages are constructed |
| `crates/syrinx-client/src/session.rs` | modify | Request diarize, parse `speaker` into `Segment`, expose handshake result |
| `crates/syrinx-client/src/config.rs` | modify | Client `diarize = false` key + generated-config comment |
| `crates/syrinx-client/src/stream.rs` | modify | Speaker-aware line breaks and prefixes |
| `crates/syrinx-client/src/save.rs` | modify | Same rules for **Save as…** rendering |
| `crates/syrinx-gui/src/main.rs` | modify | Paragraph-per-turn rendering; "labels unavailable" notice |
| `docs/deploying.md`, `README.md` | modify (Phase 2) | Model download + config docs |

---

## Task 1: The spike (go/no-go gate)

**This task is exploratory, not TDD.** Its deliverable is a decision and calibrated constants, not merged library code. Per the spec, it runs **before** any integration work so a disappointing result costs a day and leaves the protocol untouched.

**Files:**
- Create: `spike/diarize/Cargo.toml`, `spike/diarize/src/main.rs`
- The `spike/` directory is **not** added to the workspace `members` (give the crate its own empty `[workspace]` table so cargo doesn't try to adopt it). Add `spike/` to `.gitignore`? **No** — commit it; it becomes the seed of the Phase 2 evaluation harness and the record of how constants were chosen.

- [ ] **Step 1: Scaffold the crate**

`spike/diarize/Cargo.toml`:

```toml
[package]
name = "diarize-spike"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
anyhow = "1"
hound = "3"
# ort: use the same version parakeet-rs pins (2.0.0-rc.13 at time of
# writing). Find it first:
#   cargo tree -p syrinx-server --features cuda -i ort
# rc-series ort versions do not unify across majors, so this MUST match
# what Phase 2 will need to coexist with. No feature list: CPU execution
# is ort's built-in default (there is no "cpu" feature), and default
# features keep download-binaries, which is what makes this crate build
# without a hand-installed onnxruntime.
ort = "<match parakeet-rs>"
```

- [ ] **Step 2: Fetch models and test audio**

- silero-vad ONNX (v5, from the snakers4/silero-vad releases).
- Embedding candidates from the sherpa-onnx model-zoo, in spec order: 3D-Speaker ERes2Net, WeSpeaker ResNet34 (both published as ONNX), NeMo TitaNet-small as fallback.
- Two AMI-corpus meetings (headset-mix wav, 16 kHz mono — resample with ffmpeg if needed) with their reference speaker turns.
- One deliberate Teams test recording (a calibration fixture, made once).
- Keep all of it in `~/models/diarize-spike/` — **not** in the repo.

- [ ] **Step 3: Build the pipeline in main.rs**

Quick-and-dirty is fine here: wav → 30 ms VAD frames → voiced sliding windows (start at 1.5 s / 0.75 s hop) → fbank features → embedding → the clustering rules from the spec (eager assign at `T_assign`, provisional pool of ~4 agreeing windows before minting, EMA centroids with small alpha, retire-don't-merge). Print `[t0–t1] Speaker N` segments.

Feature extraction note: every candidate embedding model wants 80-dim log-mel fbank, 25 ms window / 10 ms shift. Implement it directly (~80 lines) or use a kaldi-fbank crate if one builds cleanly; note which in the report — Phase 2 reuses the choice.

- [ ] **Step 4: Evaluate**

For each candidate model, against each recording: compare printed segments to reference turns. No formal DER tooling needed — measure the three things the spec cares about:
1. Does each real speaker get exactly one label over a full meeting (count splits/merges)?
2. Are turn boundaries within ~1 s?
3. Do labels stay stable hour-to-hour (run the long AMI meeting end to end)?

Sweep `T_assign` (0.4–0.7), window (1.0–2.0 s), provisional pool size (2–6).

- [ ] **Step 5: Write the go/no-go report**

Append a `## Spike results` section to `docs/specs/2026-08-24-speaker-diarization-design.md`: chosen model, chosen constants (`T_assign`, window, hop, EMA alpha, pool size, lag depth), split/merge counts per recording, and the go/no-go call. **If no-go: stop here**, present findings to the user, and do not proceed to Task 2 without their say-so.

- [ ] **Step 6: Commit**

```bash
git add spike/ docs/specs/2026-08-24-speaker-diarization-design.md
git commit -m "Answer whether small embedding models can tell Teams voices apart"
```

---

## Phase 1: plumbing (Tasks 2–10)

Everything below is mock-tested, CUDA-free, and ships inert.

## Task 2: Proto — `diarize` on `session.start`

**Files:**
- Modify: `crates/syrinx-proto/src/message.rs`
- Modify (compile fixes): `crates/syrinx-client/src/session.rs:216-223`, plus every other `SessionStart {` construction site — find them all with `grep -rn "SessionStart {" crates/`

- [ ] **Step 1: Write the failing tests** in the existing `mod tests` of `message.rs`:

```rust
#[test]
fn session_start_without_diarize_parses_as_false() {
    // Wire compatibility: every message an old client sends today.
    let s = r#"{"type":"session.start","mode":"live","sample_rate":16000,"encoding":"pcm_s16le"}"#;
    let m: ClientMessage = serde_json::from_str(s).unwrap();
    let ClientMessage::SessionStart { diarize, .. } = m else { panic!() };
    assert!(!diarize);
}

#[test]
fn diarize_false_is_omitted_from_the_wire() {
    // False is the overwhelmingly common case; a field on every message
    // saying "nothing special" is noise an old server need never see.
    let m = ClientMessage::SessionStart {
        mode: Mode::Live,
        sample_rate: 16000,
        encoding: Encoding::PcmS16le,
        language: None,
        vocabulary: None,
        diarize: false,
    };
    assert!(!serde_json::to_string(&m).unwrap().contains("diarize"));
}

#[test]
fn diarize_true_round_trips() {
    let m = ClientMessage::SessionStart {
        mode: Mode::Transcript,
        sample_rate: 16000,
        encoding: Encoding::PcmS16le,
        language: None,
        vocabulary: None,
        diarize: true,
    };
    let s = serde_json::to_string(&m).unwrap();
    assert_eq!(serde_json::from_str::<ClientMessage>(&s).unwrap(), m);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p syrinx-proto`
Expected: compile error — `diarize` field does not exist.

- [ ] **Step 3: Implement**

In `SessionStart`, after `vocabulary`:

```rust
        /// Ask for speaker labels on this session's transcript messages.
        /// Best-effort: the server answers what it will actually do in
        /// `session.ready`, and proceeds unlabelled rather than refusing.
        #[serde(default, skip_serializing_if = "is_false")]
        diarize: bool,
```

At module level (bottom of the file, above `mod tests`):

```rust
/// serde helper: lets a false bool vanish from the wire entirely.
fn is_false(b: &bool) -> bool {
    !*b
}
```

Fix every construction site found by the grep with `diarize: false` (the client starts requesting it in Task 8; existing proto tests get `diarize: false`). One of them is `crates/syrinx-client/src/bulk.rs` — the offline file-transcription path. Leaving it at `false` is deliberate: bulk transcription never gets labels in this design, consistent with the spec's live-meeting focus.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --workspace`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git commit -am "Let a client ask for speaker labels in session.start"
```

## Task 3: Proto — `diarize` on `session.ready`

**Files:**
- Modify: `crates/syrinx-proto/src/message.rs`
- Modify (compile fixes): `crates/syrinx-server/src/ws.rs:130-136`, plus grep `"SessionReady {"` for the rest (client `session.rs:231` destructures with `..`, so only constructors break)

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn session_ready_without_diarize_parses_as_false() {
    // A new client against an old server must not choke on the handshake.
    let s = r#"{"type":"session.ready","session_id":"x","chunk_ms":560,"model":"m"}"#;
    let m: ServerMessage = serde_json::from_str(s).unwrap();
    let ServerMessage::SessionReady { diarize, .. } = m else { panic!() };
    assert!(!diarize);
}

#[test]
fn session_ready_reports_diarize_when_on() {
    let m = ServerMessage::SessionReady {
        session_id: "x".into(),
        chunk_ms: 560,
        model: "m".into(),
        diarize: true,
    };
    assert!(serde_json::to_string(&m).unwrap().contains("\"diarize\":true"));
}
```

- [ ] **Step 2: Run, expect compile failure** — `cargo test -p syrinx-proto`

- [ ] **Step 3: Implement**

```rust
        /// Whether this session will attach speaker labels. The client can ask
        /// and still not receive -- missing models, or a mode without a
        /// transcript -- and the honest answer belongs in the handshake.
        #[serde(default, skip_serializing_if = "is_false")]
        diarize: bool,
```

Fix constructors: `ws.rs` sends `diarize: false` for now (Task 7 makes it honest).

- [ ] **Step 4: Run** `cargo test --workspace` — green.

- [ ] **Step 5: Commit**

```bash
git commit -am "Say in the handshake whether speaker labels will come"
```

## Task 4: Proto — `speaker` on commit and provisional

**Files:**
- Modify: `crates/syrinx-proto/src/message.rs`
- Modify (compile fixes): `crates/syrinx-server/src/session.rs:82-88`; grep `"TranscriptCommit {"` and `"TranscriptProvisional {"` — server tests in `crates/syrinx-server/tests/` construct these too. Client matches use `{ text, .. }` and survive.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn commit_without_speaker_parses_and_omits() {
    // Both directions of compatibility in one place: an old server's commit
    // parses, and an unlabelled commit from a new server looks identical to
    // an old client.
    let old = r#"{"type":"transcript.commit","seq":1,"text":"hi"}"#;
    let m: ServerMessage = serde_json::from_str(old).unwrap();
    let ServerMessage::TranscriptCommit { speaker, .. } = &m else { panic!() };
    assert_eq!(*speaker, None);
    assert!(!serde_json::to_string(&m).unwrap().contains("speaker"));
}

#[test]
fn commit_with_speaker_round_trips() {
    let m = ServerMessage::TranscriptCommit {
        seq: 7,
        text: "hello".into(),
        speaker: Some(2),
    };
    let s = serde_json::to_string(&m).unwrap();
    assert!(s.contains("\"speaker\":2"), "got: {s}");
    assert_eq!(serde_json::from_str::<ServerMessage>(&s).unwrap(), m);
}
```

- [ ] **Step 2: Run, expect compile failure.**

- [ ] **Step 3: Implement** — on **both** `TranscriptCommit` and `TranscriptProvisional`:

```rust
        /// Who said it, numbered from 1 in order of first confident
        /// appearance. Absent when no diarizer ran, or when it honestly
        /// could not tell for this stretch -- a gap, never a guess.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speaker: Option<u32>,
```

Fix constructors with `speaker: None`.

- [ ] **Step 4: Run** `cargo test --workspace` — green.

- [ ] **Step 5: Commit**

```bash
git commit -am "Carry an optional speaker on transcript messages"
```

## Task 5: Server — `Diarizer` trait, factory, and mock

**Files:**
- Create: `crates/syrinx-server/src/diarize/mod.rs`
- Modify: `crates/syrinx-server/src/lib.rs` (add `pub mod diarize;`)

No `ort`, no features — this module is unconditional, exactly like `asr::mock`.

- [ ] **Step 1: Write the module with its tests** (trait + mock are small enough to land together; the tests pin the mock's contract before the session depends on it):

```rust
//! Speaker attribution behind a trait.
//!
//! Mirrors the `AsrBackend` boundary and exists for the same reason: the
//! session's labelling semantics -- lag, majority, strike-out -- are testable
//! in CI with [`MockDiarizer`], with no models anywhere near the tests.

use anyhow::Result;

/// One session's speaker-attribution state.
///
/// `push` is called once per ASR chunk, in order, with exactly the samples the
/// ASR saw. Ok(None) is honest uncertainty (silence, cross-talk) and is
/// normal; Err means the diarizer itself failed. The distinction is
/// load-bearing: the session counts consecutive errors to decide when to give
/// up on labelling, and must not count uncertainty.
pub trait Diarizer: Send {
    fn push(&mut self, audio: &[f32]) -> Result<Option<u32>>;
}

/// Spawns an independent [`Diarizer`] per session, sharing loaded models.
pub trait DiarizerFactory: Send + Sync {
    fn diarizer(&self) -> Box<dyn Diarizer>;
}

/// Scripted diarizer for protocol and session tests. Deterministic on
/// purpose, like [`crate::asr::mock::MockBackend`]: tests assert exact
/// message sequences.
pub struct MockDiarizer {
    script: std::collections::VecDeque<Result<Option<u32>>>,
}

impl MockDiarizer {
    pub fn new(script: Vec<Result<Option<u32>>>) -> Self {
        Self { script: script.into() }
    }

    /// The common case: one label per chunk, no errors.
    pub fn labels(labels: &[Option<u32>]) -> Self {
        Self::new(labels.iter().map(|l| Ok(*l)).collect())
    }
}

impl Diarizer for MockDiarizer {
    fn push(&mut self, _audio: &[f32]) -> Result<Option<u32>> {
        // Past the script's end: unknown, not an error. A session outliving
        // its script is normal in tests that then call finish().
        self.script.pop_front().unwrap_or(Ok(None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_replays_its_script_then_reports_unknown() {
        let mut d = MockDiarizer::labels(&[Some(1), None, Some(2)]);
        assert_eq!(d.push(&[]).unwrap(), Some(1));
        assert_eq!(d.push(&[]).unwrap(), None);
        assert_eq!(d.push(&[]).unwrap(), Some(2));
        assert_eq!(d.push(&[]).unwrap(), None);
    }

    #[test]
    fn mock_can_script_a_failure() {
        let mut d = MockDiarizer::new(vec![Err(anyhow::anyhow!("boom"))]);
        assert!(d.push(&[]).is_err());
    }
}
```

- [ ] **Step 2: Run** `cargo test -p syrinx-server` — green.

- [ ] **Step 3: Commit**

```bash
git add crates/syrinx-server/src/diarize crates/syrinx-server/src/lib.rs
git commit -m "Put speaker attribution behind a trait, with a scripted mock"
```

## Task 6: Server — session lag buffer and label attachment

The heart of Phase 1. `Session` gains an optional diarizer; when present, commits are held back `LAG_CHUNKS` chunks so the label (which needs more audio context than the text does) has caught up, then emitted with the majority label over the chunks the *held text and its lag window* cover.

**Files:**
- Modify: `crates/syrinx-server/src/session.rs`
- Modify (compile fix): `crates/syrinx-server/src/ws.rs:144` — `Session::new(mode, backend.as_ref(), sid, None)`
- Tests: alongside the server's existing session-level tests in `crates/syrinx-server/tests/modes.rs` (read that file first and match its helpers/style; add a new test module or file `crates/syrinx-server/tests/diarize.rs` if `modes.rs` helpers don't fit)

Behavioural contract to encode in tests, one test each:

1. **No diarizer → byte-identical behaviour to today**, including zero added lag. (Guard test: with `None`, pushing one chunk of a scripted backend yields its commit immediately, `speaker: None`.)
2. **With a diarizer, commits lag by `LAG_CHUNKS` (= 2, the spike's calibrated value)**: pushing chunks 1 and 2 yields nothing; pushing chunk 3 releases chunk 1's text.
3. **The label is the majority over the text's chunk plus its lag window**: script `[Some(1), Some(1)]` → chunk 1's commit carries `Some(1)`; script `[Some(1), Some(2)]` → ties break toward the **earlier** chunk's label (the chunk the words actually live in), so `Some(1)`.
4. **Uncertainty is honest**: script `[None, None]` → commit carries `speaker: None`.
5. **`finish()` flushes held commits** with whatever labels exist — nothing is ever lost to the lag buffer.
6. **Errors strike, then strike out**: a script of `MAX_DIARIZER_STRIKES` consecutive `Err`s drops the diarizer; subsequent commits carry `None` and **no further `push` is attempted** (verify via a script that would panic if reached — e.g. a label after the errors that must NOT appear).
7. **An `Ok` resets the strike count**: `Err, Err, Ok(Some(1)), Err …` keeps the diarizer alive.
8. **Live mode never gets a diarizer** — enforced at the `ws.rs` layer (Task 7), but add a `Session` doc-comment stating the invariant: `Session` trusts its caller on mode gating.

Implementation shape (complete, to be adapted to the real file):

```rust
/// How many chunks a commit is held so its speaker label can settle. The
/// diarizer needs more audio context than the transducer does; the spec
/// makes this a tunable, and the spike calibrated it to 2 (≈1.12 s, covering
/// the p90 label delay of a 1.5 s embedding window).
const LAG_CHUNKS: usize = 2;

/// Consecutive diarizer failures before the session stops asking. An
/// occasional hiccup is survivable; a diarizer that fails every chunk is
/// dead weight on a session that must keep transcribing.
const MAX_DIARIZER_STRIKES: u32 = 5;

struct HeldCommit {
    text: String,
    /// Index of the last chunk that contributed audio to this text.
    chunk: u64,
}

pub struct Session {
    // ...existing fields...
    diarizer: Option<Box<dyn crate::diarize::Diarizer>>,
    strikes: u32,
    chunks_seen: u64,
    /// Chunk-index-aligned labels, one per chunk seen. Bounded: only the
    /// last few are ever consulted, so it is drained as commits leave.
    chunk_labels: std::collections::VecDeque<(u64, Option<u32>)>,
    held: std::collections::VecDeque<HeldCommit>,
}
```

Per whole chunk in `push_audio` (and the padded tail in `finish`):

```rust
// Label first, text second: the label for chunk N must exist before any
// commit that ends at chunk N can be released.
let label = match self.diarizer.as_mut() {
    Some(d) => match d.push(&chunk) {
        Ok(l) => { self.strikes = 0; l }
        Err(e) => {
            self.strikes += 1;
            tracing::warn!("diarizer failed ({} of {MAX_DIARIZER_STRIKES}): {e:#}", self.strikes);
            if self.strikes >= MAX_DIARIZER_STRIKES {
                // Labels are decoration; the transcript is the work. Drop
                // the decoration, keep the session.
                tracing::warn!("dropping the diarizer; the session continues unlabelled");
                self.diarizer = None;
            }
            None
        }
    },
    None => None,
};
self.chunk_labels.push_back((self.chunks_seen, label));
self.chunks_seen += 1;

let text = self.stream.push(&chunk)?;
if !text.is_empty() {
    self.held.push_back(HeldCommit { text, chunk: self.chunks_seen - 1 });
}
out.extend(self.release_ripe());
```

Release logic:

```rust
/// Emit every held commit whose lag window is complete.
fn release_ripe(&mut self) -> Vec<ServerMessage> {
    let mut out = Vec::new();
    // Without a diarizer there is nothing to wait for -- including after a
    // strike-out, where holding text back would add latency for no label.
    let lag = if self.diarizer.is_some() { LAG_CHUNKS as u64 } else { 0 };
    while let Some(h) = self.held.front() {
        if self.chunks_seen < h.chunk + lag + 1 {
            break;
        }
        let h = self.held.pop_front().expect("front was Some");
        let speaker = self.majority_label(h.chunk, h.chunk + lag);
        out.push(self.emit(h.text, speaker));
        // Labels older than any commit still held are done with.
        let floor = self.held.front().map(|n| n.chunk).unwrap_or(h.chunk + 1);
        while self.chunk_labels.front().is_some_and(|(c, _)| *c < floor) {
            self.chunk_labels.pop_front();
        }
    }
    out
}

/// Most frequent Some-label across [from, to]; earlier chunks win ties,
/// because that is where the words actually live. None when nothing is
/// known -- a gap, never a guess.
fn majority_label(&self, from: u64, to: u64) -> Option<u32> {
    let mut counts: Vec<(u32, usize, u64)> = Vec::new(); // (label, count, first_seen)
    for (c, l) in &self.chunk_labels {
        if (*c >= from && *c <= to)
            && let Some(l) = l
        {
            match counts.iter_mut().find(|(k, _, _)| k == l) {
                Some((_, n, _)) => *n += 1,
                None => counts.push((*l, 1, *c)),
            }
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2)))
        .map(|(l, _, _)| l)
}
```

`emit` gains the speaker parameter; `finish()` pads/pushes the tail chunk as today, then force-releases everything held (`lag = 0` drain) before draining the model, and any model tail text emits with the last known label window.

- [ ] **Step 1: Read `crates/syrinx-server/tests/modes.rs`** to learn the existing session-test helpers and style.
- [ ] **Step 2: Write the seven failing tests** (behaviours 1–7 above; behaviour 8 is a doc comment, not a test).
- [ ] **Step 3: Run** `cargo test -p syrinx-server` — expect compile failure (signature), then test failures.
- [ ] **Step 4: Implement** as sketched; keep `emit` the single funnel.
- [ ] **Step 5: Run** `cargo test --workspace` — green, including untouched golden/protocol tests (behaviour 1 guarantees this).
- [ ] **Step 6: Commit**

```bash
git commit -am "Hold commits one chunk so their speaker label can settle"
```

## Task 7: Server — config key and ws wiring

**Files:**
- Modify: `crates/syrinx-server/src/config.rs`, `crates/syrinx-server/src/ws.rs`
- Tests: config tests inline; ws behaviour via `crates/syrinx-server/tests/protocol.rs` (read it first — it likely drives a real socket against a mock-provider app)

- [ ] **Step 1: Failing config tests**

```rust
#[test]
fn diarize_model_dir_defaults_to_absent() {
    // Absent means the feature is off, which must be the zero-config state.
    assert!(base().diarize_model_dir.is_none());
}

#[test]
fn diarize_model_dir_parses() {
    let c = Config::from_toml("token = \"a\"\nmodel_dir = \"/m\"\ndiarize_model_dir = \"/d\"").unwrap();
    assert_eq!(c.diarize_model_dir.as_deref(), Some("/d"));
}
```

- [ ] **Step 2: Implement config**

```rust
    /// Directory holding the diarization models (silero-vad + speaker
    /// embedding). Absent = speaker labels unavailable, feature off.
    #[serde(default)]
    pub diarize_model_dir: Option<String>,
```

Not env-overridable — it's a behaviour setting, same reasoning as `max_sessions`.

- [ ] **Step 3: Wire ws.rs**

- `AppState` gains `pub diarize: Option<Arc<dyn crate::diarize::DiarizerFactory>>`; `AppState::new` takes it (all current callers pass `None` — grep `AppState::new`). Note the protocol tests build the server via `app.rs::build_router(model, config)`, which calls `AppState::new` internally — `build_router` must grow the factory parameter too, or Step 4's tests have no way to inject a `MockDiarizer` factory.
- `wait_for_start` returns `Option<(Mode, bool)>` — destructure `diarize` from `SessionStart`.
- Honest handshake and session wiring:

```rust
let labelling = diarize_requested && mode == Mode::Transcript && state.diarize.is_some();
// in SessionReady: diarize: labelling
// in the inference task:
let d = labelling.then(|| state.diarize.as_ref().expect("labelling implies a factory").diarizer());
let mut session = Session::new(mode, backend.as_ref(), sid, d);
```

- `main.rs`: pass `None` for the factory (Phase 2 replaces this).

- [ ] **Step 4: Protocol-level tests** (in the style of `tests/protocol.rs`, with the mock provider): a transcribe session started with `diarize: true` against a state carrying a `MockDiarizer` factory gets `diarize: true` in ready and labelled commits; the same request with no factory gets `diarize: false` and unlabelled commits; a **live**-mode session with `diarize: true` gets `diarize: false` (mode gating).

- [ ] **Step 5: Run** `cargo test --workspace` — green.

- [ ] **Step 6: Commit**

```bash
git commit -am "Grant a session a diarizer only when it can honestly have one"
```

## Task 8: Client — request labels, carry them into segments

**Files:**
- Modify: `crates/syrinx-client/src/session.rs`, `crates/syrinx-client/src/config.rs`
- Grep `Segment {` across `crates/` for construction sites (session.rs ×3, stream.rs tests, save.rs tests, GUI?)

- [ ] **Step 1: Changes**

- `Segment` gains `pub speaker: Option<u32>` (doc: "Anonymous speaker from the server's diarizer, when one ran.").
- `SessionState` gains `pub diarize: bool` — what the handshake actually granted.
- `SessionOptions` gains `pub diarize: bool`.
- `session.start` construction: `diarize: opts.diarize && opts.mode.wire_mode() == syrinx_proto::Mode::Transcript` — never request labels on a typing session.
- The ready match arm captures `diarize` into state.
- Both `Segment` constructions in the reader capture `speaker` from the message (destructure `TranscriptCommit { text, speaker, .. }`; the provisional arm too — note the current code merges both arms with `|`, which still works since both variants now have `speaker`).
- Client config: `diarize: bool` default `false`, with a generated-config comment block in the config template ("# Ask the server for anonymous speaker labels (Speaker 1, 2, …) in\n# transcribe mode. Needs a server with diarization models installed."). **Read `config.rs` first** — it has a config-generation mechanism and tests asserting the generated file round-trips; follow that pattern. Thread the value into every `SessionOptions` construction site (grep `SessionOptions {`; daemon.rs and GUI likely both build it — pass `cfg.diarize`).

- [ ] **Step 2: Tests**

Config: generated file contains the `diarize` comment and parses back with `diarize = false`; an explicit `diarize = true` parses. Session: `to_pcm`-style pure tests aren't available for the reader loop, so the segment plumbing is covered by Task 9's stream/save tests plus the wire-mode guard test:

```rust
#[test]
fn typing_modes_never_request_labels() {
    // "Speaker 2:" typed at the cursor would be the bug.
    for m in OutputMode::ALL {
        if m.types_at_cursor() {
            assert_eq!(m.wire_mode(), WireMode::Live);
        }
    }
}
```

(That invariant already exists as a mode test; extend the session.start construction so it holds by construction, and note it in the doc comment there.)

- [ ] **Step 3: Run** `cargo test --workspace` — green.
- [ ] **Step 4: Commit**

```bash
git commit -am "Ask for speaker labels and keep them on each segment"
```

## Task 9: Client — speaker-aware stream and save formats

Per the spec's per-format rules. Turn = maximal run of segments whose `speaker` matches the last labelled one (unlabelled segments attach to the current turn without breaking it).

**Files:**
- Modify: `crates/syrinx-client/src/stream.rs`, `crates/syrinx-client/src/save.rs`

- [ ] **Step 1: Failing stream tests**

```rust
#[test]
fn a_speaker_change_breaks_the_line_and_names_the_speaker() {
    let p = scratch("spk-change");
    let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
    w.append(&seg_spk(0.0, "we ship Thursday", Some(1))).unwrap();
    w.append(&seg_spk(0.6, "no we don't", Some(2))).unwrap();
    assert_eq!(
        std::fs::read_to_string(&p).unwrap(),
        "[00:00] Speaker 1: we ship Thursday\n[00:00] Speaker 2: no we don't"
    );
    let _ = std::fs::remove_file(&p);
}

#[test]
fn an_unlabelled_fragment_stays_on_the_current_turn() {
    // Usually a short connective the diarizer could not call; breaking the
    // paragraph for it would shred every transcript.
    let p = scratch("spk-none");
    let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
    w.append(&seg_spk(0.0, "so the plan", Some(1))).unwrap();
    w.append(&seg_spk(0.6, " is simple", None)).unwrap();
    assert_eq!(
        std::fs::read_to_string(&p).unwrap(),
        "[00:00] Speaker 1: so the plan is simple"
    );
    let _ = std::fs::remove_file(&p);
}

#[test]
fn plain_gains_prefixes_only_when_labels_exist() {
    // Without labels the plain format must stay byte-identical to today.
    let p = scratch("spk-plain");
    let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
    w.append(&seg_spk(0.0, "hello ", None)).unwrap();
    w.append(&seg_spk(0.6, "world", None)).unwrap();
    assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello world");
    drop(w);
    let _ = std::fs::remove_file(&p);

    let p = scratch("spk-plain2");
    let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
    w.append(&seg_spk(0.0, "hello", Some(1))).unwrap();
    w.append(&seg_spk(0.6, "hi there", Some(2))).unwrap();
    assert_eq!(
        std::fs::read_to_string(&p).unwrap(),
        "Speaker 1: hello\nSpeaker 2: hi there"
    );
    let _ = std::fs::remove_file(&p);
}

#[test]
fn labelled_keeps_the_source_and_adds_the_speaker() {
    // The bracket names the capture source exactly as today; diarization
    // adds the speaker, it does not rename the source.
    let p = scratch("spk-labelled");
    let mut w = StreamWriter::open(&p, Format::Labelled).unwrap();
    let mut s = seg_spk(0.0, "hello", Some(2));
    s.source = Some("System audio".into());
    w.append(&s).unwrap();
    assert_eq!(
        std::fs::read_to_string(&p).unwrap(),
        "[00:00] [System audio] Speaker 2: hello"
    );
    let _ = std::fs::remove_file(&p);
}
```

With a helper `fn seg_spk(at: f64, text: &str, speaker: Option<u32>) -> Segment`.

- [ ] **Step 2: Run** — failures.

- [ ] **Step 3: Implement in `stream.rs`**

- `StreamWriter.last` becomes `Option<(f64, Option<String>, Option<u32>)>` — time, source, **last labelled speaker carried forward** (an unlabelled segment does not overwrite it).
- `continues_line`: additionally break when `seg.speaker.is_some() && seg.speaker != last_speaker` — for **every** format, plain included (a plain speaker change also needs its newline).
- `line_prefix`: append `Speaker {n}: ` when the turn opens labelled; plain's arm produces just the speaker prefix (no stamp).
- Track "have we ever seen a label" so plain stays byte-identical when no labels flow.

- [ ] **Step 4: Same rules in `save.rs::render`** — group segments into turns first, then render per format; add the mirror tests (one per format, reusing the same fixtures' expectations).
- [ ] **Step 5: Run** `cargo test --workspace` — green, all pre-existing stream/save tests untouched and passing (they build `Segment` with `speaker: None`, exercising the byte-identical path).
- [ ] **Step 6: Commit**

```bash
git commit -am "Break lines on a speaker change and name the turn"
```

## Task 10: GUI — turns as paragraphs, honesty in the handshake

**Files:**
- Modify: `crates/syrinx-gui/src/main.rs` (`transcript_scroll` at ~line 706)

- [ ] **Step 1: Implement**

- In `transcript_scroll`: when any segment carries a speaker, render grouped turns — a bold `Speaker N` header (use the existing theme's emphasis, look at how `ui.weak`/`ui.label` are used nearby) followed by the turn's concatenated text as one label; otherwise the flat `self.state.transcript` exactly as today.
- Turn grouping: reuse the same rule as Task 9 — factor a small `pub fn turns(segments: &[Segment]) -> Vec<(Option<u32>, String)>` into `syrinx-client` (`save.rs` is the natural home, next to `by_source`) and unit-test it there rather than in the GUI.
- Below the status row, when the session requested labels but `state.diarize` is false: a `ui.weak("Speaker labels unavailable on this server")` line. "Requested" means the config flag is set **and the wire mode is Transcript** — condition on `mode.wire_mode()`, not `keeps_transcript()`. **Both** mode keeps a transcript but runs the wire live and never asks for labels; showing the notice there would wrongly blame the server for the mode working as designed.

- [ ] **Step 2: Test** — the `turns()` function gets unit tests in syrinx-client (labelled runs group; unlabelled attaches; no labels → single `(None, everything)` turn). GUI rendering is verified manually in Step 3.

- [ ] **Step 3: Manual verification against the local server**

Run the server with `provider = "mock"` (and, for this check, a `MockDiarizer` factory is not wired into main — so instead verify the **unavailable** path end-to-end): GUI in transcribe mode with `diarize = true` in the client config → transcript still works, the "unavailable" notice shows, file output is byte-identical to before. The labelled path was proven by protocol tests in Task 7 and renders from the same `turns()` function tested in Step 2.

- [ ] **Step 4: Run** `cargo test --workspace`, `cargo clippy --workspace` — green.
- [ ] **Step 5: Commit**

```bash
git commit -am "Render each speaker's turn as its own paragraph"
```

**Phase 1 is now complete and inert: every client works against every server, labels flow end-to-end under mocks, and nothing changes for anyone until a server has models.**

---

## Phase 2: the real diarizer (Tasks 11–15)

**Gated on Task 1's go decision.** Constants below marked *(spike)* are placeholders to be replaced by the calibrated values from the spike report.

## Task 11: The online clusterer, pure and CI-tested

**Files:**
- Create: `crates/syrinx-server/src/diarize/cluster.rs` (unconditional — no `ort`, no feature gate)

Port the spike's clusterer properly, TDD'd from scratch. Complete skeleton:

```rust
//! Online speaker clustering: embeddings in, stable labels out.
//!
//! The rules are deliberately asymmetric -- eager to assign, reluctant to
//! create -- because the product requirement is label *stability*: once
//! someone is Speaker 2 they stay Speaker 2. The accepted failure mode is an
//! occasional split, never churn.

/// Cosine similarity above which an embedding joins its nearest centroid.
/// 0.45, not the 0.6 this plan first guessed: the spike measured same-speaker
/// windows at a median cosine of 0.52, so 0.6 rejects most true matches.
const T_ASSIGN: f32 = 0.45;
/// Consecutive mutually-agreeing orphan windows before a new speaker is minted.
/// Not a free parameter: at 2 the spike minted 20 labels for a 4-speaker
/// meeting, and 3 still failed an 87-minute one.
const MIN_POOL: usize = 4;
/// How much one window moves a centroid. Small: a centroid is its history,
/// not its last sentence. Insensitive across 0.02-0.20 in the spike.
const EMA_ALPHA: f32 = 0.05;
/// Centroids closer than this are duplicates; the newer retires. Below 0.65
/// this retires genuinely different speakers into one.
const T_RETIRE: f32 = 0.80;

pub struct OnlineClusterer {
    centroids: Vec<Centroid>,
    /// Orphan windows that matched nothing, awaiting enough agreement to
    /// become a new speaker.
    pool: Vec<Vec<f32>>,
    next_label: u32,
}

struct Centroid {
    label: u32,
    vector: Vec<f32>,
    /// A retired centroid keeps its slot (labels are never reused) but
    /// forwards to the older centroid it duplicated.
    retired_into: Option<u32>,
}
```

`observe(&mut self, embedding: &[f32]) -> Option<u32>` implements: L2-normalise → best live centroid ≥ `T_ASSIGN` → assign, EMA-update, clear pool, return label (following `retired_into` if set) → else pool it; if the pool reaches `MIN_POOL` mutually-agreeing vectors (pairwise similarity ≥ `T_ASSIGN`), mint `next_label` from their mean; a non-agreeing arrival evicts the pool's oldest. After any EMA update, check `T_RETIRE` against every other live centroid; retire the **newer** of a too-close pair.

One guard the prose above does not imply, and which the spike found necessary: **an agreeing pool must still be checked against the live centroids before it mints.** A run of windows from a speaker who already has a label can individually fall just under `T_ASSIGN` while agreeing with each other perfectly; their *mean* is much less noisy and usually lands back inside that speaker's territory. Minting there splits someone who already has a label, which is the one failure the design is built to avoid. So: if the pooled mean is within `T_ASSIGN` of any live centroid, clear the pool and return `None` rather than minting.

- [ ] **Step 1: Failing tests** (synthetic embeddings: orthogonal unit vectors per "speaker" plus small noise):
  - two well-separated voices get labels 1 and 2 in first-appearance order
  - a single noisy voice never splits across 100 windows
  - one outlier window mints nothing; `MIN_POOL` agreeing ones mint exactly one
  - `MIN_POOL` windows that agree with each other but whose mean lands within `T_ASSIGN` of an existing centroid mint **nothing** — an established speaker is never split by a borderline run
  - an assignment clears the pool (a real turn change resets the evidence)
  - EMA: 100 windows of speaker 1 then one borderline window does not move the centroid past recognising speaker 1 again
  - retire: two centroids driven together → older label survives, newer forwards, no third label appears
  - labels are never reused after retirement
- [ ] **Step 2–4: red → implement → green** (`cargo test -p syrinx-server`).
- [ ] **Step 5: Commit**

```bash
git commit -am "Cluster voices eagerly to assign and reluctantly to create"
```

## Task 12: The `diarize` feature — VAD and embedding wrappers

**Files:**
- Modify: `crates/syrinx-server/Cargo.toml`
- Create: `crates/syrinx-server/src/diarize/real.rs` (gated `#[cfg(feature = "diarize")]`)

- [ ] **Step 1: Cargo wiring**

```toml
[features]
diarize = ["dep:ort"]
# cuda stays as-is; the two are independent.

[dependencies]
ort = { version = "<exactly what parakeet-rs pins>", optional = true }
```

No feature list on `ort`: there is no `cpu` feature (CPU execution is the
built-in default), and `default-features = false` would strip
`download-binaries` while parakeet-rs wants `load-dynamic` — let cargo's
feature unification with parakeet-rs's own `ort` spec settle the link
strategy rather than fighting it here. Selecting the CPU execution
provider happens in code, per session builder, in Step 2.

Verify with `cargo tree --features cuda,diarize -i ort` that exactly one `ort` version appears. If the fbank crate chosen in the spike is used, add it here (optional, in the `diarize` feature); if the spike hand-rolled fbank, port that code into `real.rs`'s module.

- [ ] **Step 2: Port the spike's VAD and embedding code** into `real.rs` as two structs with the shapes the spike settled: `Vad::new(path) / is_speech(&[f32; FRAME]) -> Result<bool>`, `Embedder::new(path) / embed(&[f32]) -> Result<Vec<f32>>`. CPU execution provider explicitly — same lesson as `ParakeetBackend::load_cuda`: never let a default choose.

- [ ] **Step 3: Unit-test the pure parts only** (windowing arithmetic, fbank against a couple of hand-computed values if hand-rolled). Model-touching paths are exercised by Task 14's harness, not CI. `cargo test --workspace` (no features) must stay green and `cargo check -p syrinx-server --features diarize` must compile.

- [ ] **Step 4: Commit**

```bash
git commit -am "Wrap the VAD and embedding models behind the diarize feature"
```

## Task 13: `RealDiarizer` and the lifecycle

**Files:**
- Modify: `crates/syrinx-server/src/diarize/real.rs`, `crates/syrinx-server/src/main.rs`

- [ ] **Step 1: Assemble** `RealDiarizer` implementing `Diarizer::push`: accumulate samples → VAD frames → voiced sliding window *(spike constants)* → `Embedder` → `OnlineClusterer::observe` → most recent label for the chunk, `Ok(None)` when the chunk was mostly unvoiced or no window completed. `RealDiarizerFactory` implements `DiarizerFactory`, holding the two ONNX sessions the way `ParakeetBackend` holds `NemotronHandle` — loaded once, shared per-session state spawned cheaply. Set `LAG_CHUNKS` in session.rs to the spike's calibrated value.

- [ ] **Step 2: Startup wiring in `main.rs`**: when `diarize_model_dir` is set **and** the binary was built with the feature, load the factory; on failure, log loudly and continue with `None` (spec: a dictation server that won't boot over an optional feature is the wrong trade). When the config key is set but the feature is compiled out, log that mismatch explicitly — a configured-but-silent feature is the confusing state. A load-time self-check embeds one second of silence and one of noise and verifies the VAD disagrees about them — the "verify it actually works" pattern `verify_gpu` set.

- [ ] **Step 3: Run** `cargo test --workspace` and `cargo check --features diarize,cuda` — green.
- [ ] **Step 4: Commit**

```bash
git commit -am "Load the diarizer at startup and hand one to each session"
```

## Task 14: Evaluation harness

**Files:**
- Create: `crates/syrinx-server/examples/diarize_probe.rs` (gated on the feature, like `gpu_probe`)
- Delete: `spike/` (its code now lives in the tree; its report lives in the spec)

- [ ] **Step 1: Port the spike's main.rs** into the example: `cargo run -p syrinx-server --features diarize --example diarize_probe -- meeting.wav` prints `[t0–t1] Speaker N: <transcript-less segments>`. This is the calibration and regression tool for every future threshold change.
- [ ] **Step 2: Verify against the AMI files** — output matches the spike report's quality.
- [ ] **Step 3: Commit**

```bash
git rm -r spike && git add crates/syrinx-server/examples/diarize_probe.rs
git commit -m "Keep the spike's harness as the diarizer's probe"
```

## Task 15: Documentation and deployment

**Files:**
- Modify: `README.md`, `docs/deploying.md`, `docker/Dockerfile` (build with the feature), `docker/compose.yaml` comments if the model mount needs a second path

- [ ] **Step 1:** README: a "Speaker labels" section mirroring "The model" — what it does (anonymous Speaker N in transcribe mode), the two model downloads with exact `curl` commands and sizes, the `diarize_model_dir` server key, the client `diarize = true` key, and the honest-handshake behaviour. Note the licences of the chosen models (silero-vad is MIT; check the embedding model's — record it the way the NVIDIA licence is recorded).
- [ ] **Step 2:** `docs/deploying.md`: the extra volume mount, read-only, and that the container image now builds with `--features cuda,diarize`.
- [ ] **Step 3:** Full manual pass against the local server: GUI, transcribe mode, `diarize = true`, two people talking (or one person + a played recording) → labelled paragraphs live, streamed file labelled, save labelled.
- [ ] **Step 4: Commit**

```bash
git commit -am "Document speaker labels, and ship the models to the container"
```

---

## Execution notes

- Tasks 2–4 are mechanical and could batch into one session; Task 6 is the subtle one — do it alone and read `tests/modes.rs` first.
- Task 1 (spike) and Task 10 Step 3 / Task 15 Step 3 need a human or a machine with audio and models; everything else is pure `cargo test`.
- If the spike's answer is no-go: Phase 1 is still worth discussing with the user (the protocol fields cost nothing and the plumbing would serve a future attempt), but do not build it on momentum — ask.
