# Faster labels, cleaner turns, and a room with eight people in it

**Date:** 2026-08-27
**Status:** Implemented, with four of its own rules corrected during
implementation — see "Where this document was wrong" at the end. Read that
section alongside the design below; where the two disagree, it wins.
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
- Otherwise **do not update any centroid**. Ambiguity today is resolved by the
  third decimal place and then written into a centroid via EMA; that is the
  drift mechanism. An ambiguous window changes nothing.
- Otherwise **still report the argmax**, marked provisional, and **still pool
  the window**. Withholding buys the drift protection; it must not also be
  paid for out of the miss rate. See "Where this document was wrong", item 3.

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

`pool.clear()` on assignment is removed. Orphans are held **per candidate
speaker**: an orphan joins the pool every member of which it agrees with at
`T_ASSIGN`, or starts a pool of its own, and a pool reaching `MIN_POOL = 4`
members mints. Pools are bounded in number (`MAX_POOLS = 8`, one per person in
the room this is for) and in age (`POOL_AGE = 64` windows without a new
member).

`MIN_POOL` stays 4, so the reluctance the spike proved is intact: four windows
must still agree with each other before a speaker exists. What is dropped is
the additional, undocumented, unmeasured requirement that no other speaker talk
in between, which crowding turns into an impossibility.

Per candidate rather than one shared ring, because a shared ring has to hold
the mintable set *and* everything that arrives between its members: with `k`
foreign orphans between each of a newcomer's windows it needs
`MIN_POOL + (MIN_POOL - 1) * k` slots, which at `k = 7` is 25. See "Where this
document was wrong", item 2.

Because a pool is mutually agreeing by construction, there is no subset search
and no question of a poisoned window: a window that agrees with nobody starts
its own pool and waits there for agreement that never comes.

The mean-versus-nearest re-test before minting is **not** retained unchanged;
see item 1 below.

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

A third smoother is needed and was missing from this design: a **refractory
floor** on how often a boundary may be accepted. See item 4 below.

### Labels may be corrected

New additive server message:

```rust
ServerMessage::TranscriptRelabel { from_seq: u64, to_seq: u64, speaker: u32 }
```

`transcript.revise` stays reserved and untouched. Relabelling is not a text
revision, the two have different client handling, and overloading revise would
require adding a speaker field to a message whose existing semantics say the
text changed.

The session keeps a bounded ring of recently emitted commits — their `seq`,
the chunk range **their text came from**, and whether the speaker beside them
is a guess — spanning `RELABEL_WINDOW = 30s`. The chunk range is the text's
provenance and not the range its label was voted over: the vote runs from the
text's last chunk forward through the lag window, so it starts where the words
end and reaches `lag_chunks` into whoever spoke next. Matching a correction
against that is wrong in both directions. Two events emit a relabel:

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

That promise is only keepable if the **client** can tell which of its commits
carry a guess, so `transcript.commit` carries `speaker_provisional: bool`
(additive, omitted when false) and the client applies a correction only to
segments it holds as unlabelled or provisional. A rule enforced at one end of
a protocol is a comment about that end.

**How far back a correction may reach** is bounded by provenance, not by "no
turn change was detected". A change is undetectable where there was nothing to
compare against, and the first hop of a session is exactly that case, so a
meeting opening with a short unlabelled utterance would otherwise have the
*next* speaker's name written over it. The bound moves forward at three
points: a detected turn change, the first completed hop, and a silence long
enough to have cleared the window accumulator. A turn change entirely inside
one hop remains undetectable — the residual exposure is one hop of audio at
the opening of a run, and is documented in the code rather than argued away.

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
diarize_mint_ceiling = 0.55     # how close to an incumbent a pool's mean may mint
diarize_change_threshold = 0.30 # cosine drop between hops that marks a turn
diarize_relabel_window = 30     # seconds of transcript still eligible for correction
```

Starting values are engineering estimates, not measurements, and the probe work
below exists to replace them. So are `T_MINT_MARGIN = 0.20`, `MAX_POOLS = 8`
and `POOL_AGE = 64`, which have no keys of their own.

`diarize_mint_ceiling` is a key rather than a derivation, and the correction
below records why: this document specified the ceiling as `T_ASSIGN + margin`,
which put one key in charge of two unrelated decisions and put the ceiling on
the wrong scale to boot.

**Every documented "off" has to be off.** `diarize_relabel_window = 0` disables
relabelling entirely. `diarize_margin = 0` disables the margin, the cohort
test and the mint ceiling together — the cohort test has no key of its own, so
leaving it running would give a deployment reaching for the hatch no relief in
the only room the hatch is for, and the ceiling is switched off by a rule
`may_mint` states rather than by the arithmetic that used to imply it.
`diarize_change_threshold = 0` is *checked for* rather than compared against,
because different-speaker cosines are routinely negative (median 0.046) and no
value of a plain `cosine < threshold` could ever mean "never".

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
- **The pool mints through interruption.** Four agreeing orphans separated by
  other speakers' assigned windows still mint, where the consecutive pool would
  not — and, separately, still mint with up to `MAX_POOLS - 1` *foreign
  orphans* between each, which is the case a shared ring cannot hold.
- **An ambiguous pool mints nobody.** Four mutually agreeing windows sitting
  equidistant from two incumbents are not a third speaker, at 0.60, 0.65, 0.705
  or 0.79 — while the same four just inside the ceiling still are.
- **A pool resembles its own mean far more than its members resemble each
  other**, with the arithmetic stated: it is what decides whether the mint
  gate's second clause can bind at all, and a form that cannot bind is
  indistinguishable from no clause under every other test in the file.
- **A non-finite embedding is refused**, never assigned, and moves no centroid:
  one such assignment is an absorbing state, since the EMA makes that centroid
  non-finite and `total_cmp` then ranks it above every real one.
- **A window completes between any two accepted boundaries**, however often
  detection fires.
- **Z-norm is inert below the cohort minimum**, so two- and three-speaker
  behaviour is provably unchanged.
- **Change detection cuts the window**, and a window spanning a synthetic
  boundary is never embedded as a mixture.
- **Carry-forward stops at a boundary**, and the vote does not cross one.
- **Relabel** fills a `None` gap on mint, corrects a contradicted provisional,
  never renumbers, never crosses out of its window, and is omitted entirely at
  `diarize_relabel_window = 0`. At the **shipped lag depth**, with commits
  spanning several chunks each: at `lag_chunks = 0` and one word per chunk, a
  commit's first chunk, its last chunk and the end of its vote window are all
  the same number, and every question about which chunks a commit's words came
  from collapses.
- **Both documented "off" switches are off**, checked against behaviour rather
  than against the config parser.
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

Two things the harness must get right, because both understate what it is
measuring. It replays the server's own `Session`, so the label for a chunk
reaches a client `lag_chunks` later than the audio it describes — 1.12 s at
the shipped depth — and the latency figures charge that. And it drives
`RealDiarizerFactory`, which resolves the embedding model by the server's own
rules, so it does not accept a `--model` flag it would then ignore: it prints
the model that actually ran.

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
  number and already documented as out of scope. The refractory floor bounds
  how bad this can get: a spurious boundary costs at most the audio before the
  next one, and they cannot arrive faster than one per 0.768 s of speech.
- **A crowded room is improved, not solved.** The mint gate's ceiling,
  `T_MINT_CEILING`, is 0.55, and the spike measured the two closest live
  centroids at 0.519 with only five speakers in the room. A genuinely new voice
  whose pooled mean lands above 0.55 from an incumbent still cannot be minted;
  what it gets instead is that incumbent's number as a *guess*, which reads as
  a wrong label rather than as a gap. Whether that
  ceiling is in the right place is the single most important thing the live
  probe has to answer — which is the other reason it is now a number of its
  own, reachable from `diarize_mint_ceiling` and from `--mint-ceiling`, rather
  than something only movable by changing what assignment does.

## Where this document was wrong

Four of the rules above were found to be wrong during implementation and are
corrected in the code. They are recorded here rather than quietly edited out,
because each was reasoned from the same evidence the rest of the design was
and the reasoning is worth not repeating.

**1. Relaxing the mint gate manufactures splits.** The rule as designed —
refuse a mint when the pooled mean is *assignable*, or within `T_RETIRE` —
makes the whole band from `T_ASSIGN` to 0.80 mintable whenever the nearest
incumbent is merely ambiguous. Measured end to end with two incumbents, it
mints a fresh label at 0.500, 0.600, 0.650 and 0.705 where the old rule
assigned; against a same-speaker median of 0.517, a mint at 0.705 is
overwhelmingly the same person. A duplicate then retires into its neighbour
tens of windows later, by which time commits carrying its number are frozen —
so the transcript is left permanently naming a speaker the session no longer
has. The flaw is conceptual: **ambiguity between two incumbents is evidence of
a bad or mixed embedding, not evidence of a new person**, and the rule treats
it as novelty. The gate now asks whether the pooled group is more like itself
than like anybody known, inside a ceiling which at margin 0 collapses onto the
pre-2026-08-27 rule exactly. The mintable band narrows from `[0.45, 0.80)` to
`[0.45, 0.55)`.

**And that ceiling is `T_MINT_CEILING`, not `T_ASSIGN + margin`.** Deriving it
from the assignment margin, as specified above, was wrong twice over. It gave
`diarize_margin` two unrelated jobs pulling in opposite directions — the key is
documented as assignment caution, so raising it to name people more carefully
in a crowded room silently *widened* the band a pool could mint into, loosening
the split protection measured at zero. And it put the ceiling on the wrong
scale, which is the same error the paragraph below identifies on the cohesion
side: `T_ASSIGN` and `T_MARGIN` were calibrated on one noisy window against a
centroid, while what the ceiling compares is a mean of `MIN_POOL` windows
against a centroid, and for one voice that sits far higher. The shipped ceiling
is 0.55 either way, so nothing at the shipped settings changed; `margin = 0`
still switches it off, now because `may_mint` says so rather than because
`T_ASSIGN + 0` happened to make its clause unreachable.

A deployment at a *tuned* margin does see its mint policy move, which is the
correction rather than a side effect of it. With a perfectly cohesive pool the
mintable-rival band topped out at `min(T_ASSIGN + margin, T_RETIRE)` — 0.470 at
a margin of 0.02, 0.500 at 0.05, 0.600 at 0.15, 0.650 at 0.20, 0.750 at 0.30
and 0.800 from 0.40 up — and is 0.550 at every one of them now, tighter above
the shipped margin and looser below it. Only 0.10 and the hatch at 0 are
unchanged. `diarize_mint_ceiling = T_ASSIGN + margin` puts any of them back for
an operator who would rather wait for the probe than take the new value.

**"More like itself" has to be measured against the pool's own mean, on both
sides.** The obvious reading — worst *pairwise* agreement among the members,
against the mean's cosine to the nearest centroid — compares two different
scales. A pool is mutually agreeing at `T_ASSIGN` by construction, so its worst
pairwise figure is floored at exactly 0.45 and can be nothing else, while
averaging four windows removes most of their noise, so the same pool's mean
sits at 0.766 or better from its own members and correspondingly closer to any
centroid. Written that way the clause admits only
`rival <= T_ASSIGN - T_MINT_MARGIN` = 0.25, which is stricter than the rule it
is layered on: it reads as a gate and acts as a veto, leaving the ceiling
unreachable and the whole mint path exactly what it was before. Measured over
synthetic rooms with the spike's same-speaker statistic, the shipped
configuration and `diarize_margin = 0` then mint the *same* speakers at every
crowding level — the crowding fix does nothing at all. Comparing
`min cos(mean, member)` against `cos(mean, nearest centroid)` instead puts both
sides on the mean's scale; over the same rooms it lifts eight-speaker mint
counts from 6–7 to 7–8 and *lowers* the unlabelled share (16% to 9%), because
a newcomer with their own centroid is unambiguous where a newcomer merged into
an incumbent is not.

**2. `POOL_RING = 8` tolerates one interrupting speaker.** Holding `MIN_POOL`
newcomer windows with `k` foreign orphans between each needs `4 + 3k <= 8`, so
`k <= 1`; a room of eight has seven other people, and the margin and cohort
gates *convert other speakers' assignments into orphans*, so `k` is large in
exactly the room this design exists for. Measured at `k = 2` the newcomer was
never minted across twelve windows. Complaint 3's "the new speaker may never be
minted at all" was not fixed by this design as written — the mechanism merely
changed from `pool.clear()` to ring eviction. Per-candidate pools replace it.

**3. Withholding an assignment must not cost coverage.** At four live centroids
with one neighbour at 0.519 — the crowding the spike measured at five speakers
— the margin and cohort gates refuse a true match at the same-speaker median of
0.517 and again at 0.60, accepting only from 0.65. That is the spike's
documented failure mode verbatim: a configuration that says almost nothing
scores perfectly on stability while its miss rate climbs from 26% to 37%. The
two jobs a withheld window was doing are now separated. On ambiguity: no
centroid is updated, which is the drift protection and the real win; the argmax
*is* emitted, marked provisional; and the window *is* pooled, so a mint from
those windows corrects the text through the relabel machinery this design
already builds.

**4. Change detection can starve the clusterer.** `cut_at_boundary` keeps a hop
(23 frames) and a window needs 47, so between two accepted boundaries more than
one hop of voiced audio must arrive or **no window ever completes**. Simulated
over 112 s of continuous speech, a boundary every hop embedded zero windows —
no centroid updated, nobody minted, and nothing for a hop's provisional label to
name; a boundary every two hops doubles complaint 1's latency. In a room of
eight taking short turns, a boundary every other hop is the detector working
correctly. There is a refractory floor now, of `WINDOW_FRAMES - HOP_FRAMES` =
24 voiced frames (0.768 s), which is not a taste in turn lengths but exactly
what window formation needs to make progress. Its cost is stated rather than
hidden: a turn shorter than that cannot be detected as one.
