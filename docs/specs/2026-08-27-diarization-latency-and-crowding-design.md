# Faster labels, cleaner turns, and a room with eight people in it

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation
**Supersedes constants in:** `docs/specs/2026-08-24-speaker-diarization-design.md`
(architecture and spike results there still stand; this changes how windows are
cut, how assignment decides, and when a label may be corrected)

## Problem

Three complaints from real use, all of which the original design either
predicted or can be traced to a named line:

1. **The first label takes too long.** A voice must assemble four mutually
   agreeing 1.5 s windows before it is minted, which is 1.504 + 3 × 0.736 ≈
   **3.71 s of voiced audio** — and by then the opening sentences have already
   been committed unlabelled and can never be corrected.
2. **A turn change takes about a sentence to show up.** Measured causes are
   stacked, and are enumerated below.
3. **It breaks apart at eight or more people.** The 2026-08-24 spec named this
   as the most likely failure: *"the margin is gone at five speakers, and eight
   is untested. If a meeting is going to fail, this is how."*

### Why the first label is slow

`OnlineClusterer::observe` (`crates/syrinx-server/src/diarize/cluster.rs:140-205`)
pools embeddings that match no centroid and mints a speaker only when
`MIN_POOL = 4` of them mutually agree. That reluctance is load-bearing and the
spike proved it: a pool of 3 found six speakers in a room of five, a pool of 2
found twenty in a room of four. `MIN_POOL` is not a free parameter and this
design does not lower it.

The fixable half is that the text is gone by then. `Session::emit`
(`session.rs:280-287`) only ever produces `TranscriptCommit`; once sent, a
commit's speaker is final. `TranscriptRevise` exists in the protocol but is
explicitly reserved and unemitted (`crates/syrinx-proto/src/message.rs:85-94`),
and carries no speaker field — the client hardcodes `speaker: None` for
revisions (`crates/syrinx-client/src/session.rs:498-501`).

### Why a turn change takes a sentence

Five mechanisms, each adding to the last:

1. `MAX_GAP_FRAMES = 15` (0.48 s) is the only thing that resets the window
   accumulator, and `window.rs:44-52` records that it is **deliberately not a
   turn-change detector** — over 13 minutes of AMI it broke 151 times, and 48
   of 51 decidable breaks had the same speaker on both sides. At a normal turn
   change with less than 0.48 s of silence, `WindowAssembler.voiced` still
   holds up to 46 frames of the previous speaker.
2. The following windows are therefore A/B mixtures. A window that is purely B
   needs 47 voiced frames ≈ **1.5 s of B's speech**, and windows only complete
   on 23-frame boundaries, adding up to 0.736 s of granularity on top.
3. Between window completions, `RealDiarizer` repeats `last_label` for every
   chunk containing any voiced frame (`real/diarizer.rs:549-553`) — so A's
   label is *actively asserted* over B's opening sentence rather than merely
   being stale.
4. `majority_label` votes across `[chunk, chunk + lag]` = three chunks and
   breaks ties toward the label seen **first** (`session.rs:269`), which at a
   turn boundary is by construction the outgoing speaker. `tests/diarize.rs:89-113`
   pins exactly this: at lag 0 the commit is `Some(1)`, at lag 1 still
   `Some(1)`, only at lag 2 `Some(2)`.
5. Then the 1.12 s release lag.

### Why eight people breaks it

- **A fixed threshold against a rising maximum.** `nearest()`
  (`cluster.rs:209-216`) is an argmax of cosine over all live centroids, tested
  against a constant `T_ASSIGN = 0.45`. Measured same-speaker p50 is 0.517 and
  different-speaker p50 is 0.046 — but with **five** speakers the two closest
  live centroids already sat at **0.519**, above the threshold. The margin
  between "same speaker" and "nearest wrong centroid" is already negative at
  five. As centroids are added, the probability that some incumbent exceeds
  0.45 rises monotonically, so a genuine eighth speaker's windows increasingly
  assign to somebody else instead of pooling.
- **The pool can never fill in a crowded room.** Every assignment executes
  `self.pool.clear()` (`cluster.rs:165`), so minting requires four
  *consecutive* orphan windows with no assignment in between. With eight people
  interleaving, and with the previous point making stray assignments more
  likely, that run is rarely obtained — the new speaker is not merely slow to
  appear, they may never be minted at all and arrive as `None` forever.
- **No compensating machinery.** There is no max-speaker cap, no k estimate, no
  adaptive threshold, no re-clustering, no split detection, no ambiguity band.
  A 0.451 versus 0.450 pair is decided by the third decimal with no rejection.
- **Drift.** EMA at α = 0.05 with no count weighting (`cluster.rs:157-161`):
  every wrongly-assigned window from a crowded neighbour pulls a centroid 5%
  toward the intruder, and sustained wrong assignment walks it across the
  space. The existing retirement test demonstrates the mechanism.

### One measurement that reframes all of it

The spike's published accuracy — 96–98% correct attribution, zero splits, zero
merges — was produced by `label_frames` (`examples/diarize_probe/main.rs:549-584`),
which paints each 10 ms frame with the majority of up to `VOTE_SLOTS = 8`
windows covering it. That is **backwards painting**, and `main.rs:41-45` says
so outright. The live session cannot do it; it carries the last label forward
instead.

So the live path has always been the outlier, and the probe has never measured
the first-label or turn-change latency these complaints describe. Retroactive
labelling is not a new liberty being taken — it is the mode every published
number was measured in.

## Design

### One extra embedding per hop, serving all three complaints

Today the 0.75 s of new voiced audio at each hop only ever contributes to the
next 1.5 s window. Instead it is *also* embedded on its own as `h_i`. Cost is
one more 45 ms embedding per 0.75 s of voiced audio, taking the diarizer from
≈6% to ≈12% of one core — still comfortably inside the budget the 2026-08-24
spec set, and still bounded by the audio handed to it, so the reasoning in that
document's error-handling section is unchanged.

`h_i` buys three things:

**Turn-change detection.** `cos(h_i, h_{i-1})` falling below `T_CHANGE` marks a
speaker boundary at that hop. When one is marked, the 1.5 s accumulator is
flushed at the boundary, so the next full window is composed of one voice
rather than two. This is the fix for mechanisms 1 and 2 above. Boundary
resolution is the hop, 0.736 s, against a current effective resolution of a
window and a half.

**A fast provisional label.** `h_i` is matched against existing centroids and,
if it clears a *stricter* bar than a full window must (see below), yields a
label roughly 0.75 s into a turn rather than after a full clean window plus the
vote.

**Nothing else.** Short embeddings **never** update a centroid and **never**
contribute to the mint pool. Full 1.5 s windows keep exclusive ownership of
both. A 0.75 s ERes2Net embedding is noisier than a 1.5 s one — adequate for
"did the voice change" and for "which known voice is this, probably", not for
"is this a person we have never heard". Keeping centroid quality on the long
windows is what preserves everything the spike measured.

The stricter bar for provisional labels is a higher margin requirement, not a
higher `T_ASSIGN`: raising the absolute threshold is the failure mode the spike
documented, where the clusterer stops speaking and scores perfectly on
stability by saying nothing (miss climbing from 26% to 37% while splits and
merges stay at zero).

### Assignment becomes adaptive

`nearest()` is replaced by a decision over the best two candidates. Let `s1` be
the highest cosine to any live centroid and `s2` the second highest.

- Assign to the best centroid only when `s1 >= T_ASSIGN` **and**
  `s1 - s2 >= T_MARGIN`.
- Otherwise return `None` — and critically, **do not update any centroid**.
  Ambiguity today is resolved by the third decimal place and then written into
  a centroid via EMA; that is the drift mechanism. An ambiguous window should
  change nothing.

Above `COHORT_MIN = 4` live centroids, `s1` is additionally normalised against
the cohort of scores to the other centroids — `z = (s1 - mean(others)) / std(others)`,
tested against `T_ZNORM`. This is adaptive score normalisation, the standard
remedy in speaker verification for precisely this crowding, and it is
self-tuning: it asks whether this centroid stands out from the field rather
than whether it clears a number chosen when the field was smaller. Below four
centroids the cohort is too small to estimate a spread from, so raw cosine
governs and behaviour at two to three speakers is unchanged.

`T_ASSIGN` stays at 0.45. The margin and z-norm only ever *withhold* an
assignment that the current code would make; they never manufacture one. So the
change cannot introduce a merge, and its worst case is a higher miss rate,
which `speaker: None` is designed to carry.

### The pool stops needing to be consecutive

`pool.clear()` on assignment is removed. The pool becomes a bounded ring of the
most recent `POOL_RING = 8` orphan embeddings, and minting fires when any
`MIN_POOL = 4` of them are mutually in agreement — not necessarily consecutive,
not necessarily uninterrupted by other speakers' windows.

`MIN_POOL` stays 4, so the reluctance the spike proved is intact: four windows
must still agree with each other before a speaker exists. What is dropped is
the additional, undocumented, unmeasured requirement that no other speaker talk
in between, which crowding turns into an impossibility. The mean-versus-nearest
re-test before minting (`cluster.rs:183-194`) is retained unchanged.

`pool_agrees()` currently requires **all** pairs to clear `T_ASSIGN` and evicts
only the oldest on failure (`cluster.rs:225-231`, `:176-181`), recovering one
window per attempt. With a ring, the search is for any agreeing subset of four,
which recovers immediately from a single poisoned window.

### The session stops arguing with the boundary

Two smoothers actively fight a turn change and both become boundary-aware:

- **Carry-forward stops at a boundary.** `RealDiarizer` currently repeats
  `last_label` for every voiced chunk until another window completes. Once a
  change point is detected, carry-forward ceases and `None` is emitted until a
  window or a provisional label resolves the new voice. Asserting the previous
  speaker's name across a detected boundary is worse than saying nothing.
- **The majority vote does not cross a boundary.** `majority_label(from, to)`
  is clipped at any detected change point inside its range, and the tie-break
  toward the earliest label applies only within a single turn. The existing
  behaviour pinned by `tests/diarize.rs:89-113` changes deliberately, and that
  test is updated to assert the new intent rather than deleted.

### Labels may be corrected

New additive server message:

```rust
ServerMessage::TranscriptRelabel { from_seq: u64, to_seq: u64, speaker: u32 }
```

`transcript.revise` stays reserved and untouched. Relabelling is not a text
revision, the two have different client handling, and overloading revise would
require adding a speaker field to a message whose existing semantics say the
text changed.

The session keeps a bounded ring of recently emitted commits — their `seq` and
the chunk range each covered — spanning `RELABEL_WINDOW = 30s`. Two events emit
a relabel:

- **A speaker is minted.** The pooled windows have known chunk ranges; the
  commits overlapping them were emitted with `speaker: None` and now have a
  number. This is the fix for complaint 1: the four-window reluctance is
  retained in full, and the opening sentences get their label anyway.
- **A provisional label is corrected.** A short-window guess that the
  confirming full window contradicts is relabelled rather than left wrong.

Older text is frozen. A relabel never renumbers an existing speaker and never
reassigns a commit that already carries a different confident label from a
different turn — it fills gaps and corrects provisionals, nothing else. The
"never renumber" property from the 2026-08-24 design is preserved exactly.

**Client handling, per surface:**

- **GUI** applies the relabel to its in-memory segments and repaints. Earlier
  lines acquire their speaker.
- **Save as…** writes the corrected labels, because it renders from the same
  in-memory segments.
- **`StreamWriter` ignores relabels entirely.** The streamed file stays
  strictly append-only. It keeps the honest record of what was known live, the
  tear-resistance property tested in `stream.rs:595-637` is untouched, and the
  README's existing paragraph about session-opening lines staying unattributed
  remains true — of the streamed file specifically, which is where that
  paragraph's reasoning came from.

That divergence is deliberate and must be documented: the GUI and the saved
file carry corrected attribution; the streamed file carries live attribution.

### Configuration

Joining `diarize_lag_chunks` and `diarize_min_pool`, with the same treatment —
optional, range-validated at startup, not environment-overridable:

```toml
diarize_margin = 0.10           # how far the best centroid must beat the second
diarize_change_threshold = 0.30 # cosine drop between hops that marks a turn
diarize_relabel_window = 30     # seconds of transcript still eligible for correction
```

Starting values are engineering estimates, not measurements, and the probe work
below exists to replace them. `diarize_relabel_window = 0` disables relabelling
entirely, which is the escape hatch if corrections prove more disruptive to
read than gaps.

## Testing

The most important existing gap: **no test exercises the real clusterer past
two synthetic speakers**, which is why the eight-speaker failure was a
prediction in a document rather than a red test.

- **Crowding, as pure functions.** Synthetic 8- and 12-speaker sets with
  controlled inter-speaker similarity, including a set deliberately packed near
  0.5 to reproduce the measured crowding. Assert: every speaker is minted, no
  merges, and the margin rule withholds rather than guesses on a deliberately
  ambiguous vector.
- **Ambiguity changes nothing.** A vector between two centroids returns `None`
  and leaves both centroids bit-identical — the drift regression.
- **The ring pool mints through interruption.** Four agreeing orphans separated
  by other speakers' assigned windows still mint, where the consecutive pool
  would not.
- **Z-norm is inert below the cohort minimum**, so two- and three-speaker
  behaviour is provably unchanged.
- **Change detection cuts the window**, and a window spanning a synthetic
  boundary is never embedded as a mixture.
- **Carry-forward stops at a boundary**, and the vote does not cross one.
- **Relabel** fills a `None` gap on mint, corrects a contradicted provisional,
  never renumbers, never crosses out of its window, and is omitted entirely at
  `diarize_relabel_window = 0`.
- **Client:** the GUI's segment store applies relabels; `save::render` reflects
  them; `StreamWriter` provably does not.
- **Proto:** round-trip, and an old client ignoring an unknown message type.

### The probe learns to measure the live path

`diarize_probe run` currently reports numbers obtained with backwards painting,
so it cannot see either latency this design targets. It gains a live-emulation
mode that replays the session's actual rules — carry-forward, the majority
vote, the lag, and now change detection and relabelling — and reports:

- **first-label latency** per speaker: voiced seconds from a speaker's first
  speech to their first label, before and after relabelling;
- **turn-switch latency**: voiced seconds from a real turn change to the label
  changing, which is the number complaint 2 is about;
- **crowding** at 8+, by mixing AMI recordings to synthesise larger rooms than
  the corpus provides, since the untested regime is the whole point.

These are the acceptance measurements. The constants above are provisional
until this mode has run.

## Risks

- **Correction is visible.** Text on screen acquiring a speaker name a few
  seconds late is a new behaviour and may read worse than a gap for some users.
  `diarize_relabel_window = 0` is the retreat.
- **0.75 s embeddings are noisier than the model likes.** Mitigated by never
  letting them touch a centroid or the mint pool, and by the stricter margin —
  but the provisional label is the part of this design most likely to need its
  threshold moved after measurement.
- **Doubling the embedding rate doubles the diarizer's CPU.** 12% of a core per
  session, still bounded by input audio. If `max_sessions` is ever raised far
  enough for diarizers to rival the ASR, the 2026-08-24 design's note on a time
  budget feeding the existing retirement is the shape to reach for.
- **Change detection on overlapped speech.** Cross-talk will produce embedding
  jumps that are not turn changes, cutting windows spuriously. The cost is a
  higher miss rate during overlap, which is already excluded from every measured
  number and already documented as out of scope.
