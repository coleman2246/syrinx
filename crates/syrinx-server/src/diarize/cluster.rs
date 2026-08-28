//! Online speaker clustering: embeddings in, stable labels out.
//!
//! The rules are deliberately asymmetric -- eager to assign, reluctant to
//! create -- because the product requirement is label *stability*: once
//! someone is Speaker 2 they stay Speaker 2. Joining an existing speaker is
//! cheap, since a wrong join costs one mislabelled sentence; minting one is
//! expensive, since a wrong mint costs an extra label that no later evidence
//! ever takes back. The accepted failure mode is an occasional split, never
//! churn.
//!
//! "Eager to assign" acquired a qualification on 2026-08-27, and it is the
//! only one: eager to assign *when it is clear who*. The original rule was an
//! argmax of cosine tested against a fixed threshold, which is a decision that
//! gets worse every time a centroid is added -- at five speakers the spike
//! already measured the two closest live centroids at 0.519, above `T_ASSIGN`,
//! so a genuinely sixth voice had incumbents above the bar to be handed to.
//! Assignment now also asks how far the winner beat the runner-up, and in a
//! crowded room how far it stands out from the whole field. Both rules can
//! only ever *withhold* an assignment the old code would have made, never
//! manufacture one, so neither can introduce a merge; their cost is a higher
//! miss rate, which `speaker: None` is the field for.
//!
//! Most constants below are measurements: see "Spike results" in
//! `docs/specs/2026-08-24-speaker-diarization-design.md`, where they were
//! swept against three hand-annotated AMI meetings. The four added by
//! `docs/specs/2026-08-27-diarization-latency-and-crowding-design.md`
//! -- [`T_MARGIN`], [`T_MARGIN_SHORT`], [`T_ZNORM`], [`POOL_RING`] -- are
//! **engineering estimates and have not been measured**; the probe's
//! live-emulation mode exists to replace them.
//!
//! Pure arithmetic. No models, no `ort`, no feature gate -- which is what lets
//! the whole of the labelling policy be tested in CI on synthetic embeddings.

// The constants are `#[doc(hidden)] pub` rather than private for one reason:
// `examples/diarize_probe` overrides them one at a time, and it is external to
// this crate, so a private const would force it to keep its own copy of the
// others. A stale copy there would attribute a sweep's numbers to constants
// the server no longer ships, which is precisely the failure the probe exists
// to prevent. Hidden from the docs because most of them are not a
// configuration surface -- see [`OnlineClusterer::with_params`] -- and the two
// that are, `MIN_POOL` and `T_MARGIN`, are only the defaults of config keys.

/// Cosine similarity above which an embedding joins its nearest centroid.
/// 0.45, not the 0.6 the design first guessed: the spike measured
/// same-speaker windows at a median cosine of 0.52, so 0.6 rejects most true
/// matches.
#[doc(hidden)]
pub const T_ASSIGN: f32 = 0.45;
/// Mutually agreeing orphan windows before a new speaker is minted. Not a
/// free parameter: at 2 the spike minted 20 labels for a 4-speaker meeting,
/// and 3 still failed an 87-minute one.
///
/// The default of the `diarize_min_pool` config key, which reads it from here
/// so the shipped number stays next to the measurement that chose it. A
/// deployment can trade it down for faster pickup on its own audio; nothing
/// here changes what was measured on AMI.
#[doc(hidden)]
pub const MIN_POOL: usize = 4;
/// How much one window moves a centroid. Small: a centroid is its history,
/// not its last sentence. Insensitive across 0.02-0.20 in the spike.
#[doc(hidden)]
pub const EMA_ALPHA: f32 = 0.05;
/// Centroids closer than this are duplicates; the newer retires. Below 0.65
/// this retires genuinely different speakers into one.
#[doc(hidden)]
pub const T_RETIRE: f32 = 0.80;

/// How far the best centroid must beat the second best before a full 1.5 s
/// window joins it. Below this the window is ambiguous, and an ambiguous
/// window is pooled rather than assigned -- and, critically, moves no
/// centroid at all.
///
/// This is the drift fix as much as the crowding fix. Resolving a 0.451
/// against a 0.450 by the third decimal and then writing the answer into a
/// centroid via EMA is how a centroid walks across the space one wrong
/// window at a time; the retirement machinery exists to clean up after
/// exactly that, and this is what stops it happening.
///
/// **Unmeasured.** 0.10 is an engineering estimate sized against the spike's
/// separability figures -- same-speaker windows at a median cosine of 0.517,
/// different-speaker at 0.046, so a true match normally beats the field by
/// far more than 0.10 while the crowded case the design warns about beats it
/// by almost nothing. The default of the `diarize_margin` config key.
#[doc(hidden)]
pub const T_MARGIN: f32 = 0.10;

/// The same, for a 0.75 s hop embedding offering a provisional label.
///
/// Stricter, because a 0.75 s ERes2Net embedding is noisier than a 1.5 s one
/// and this one is allowed to speak before any full window has confirmed it.
/// Deliberately a wider *margin* rather than a higher `T_ASSIGN`: the spike
/// measured what raising the absolute threshold does, and it is the failure
/// mode where the clusterer scores perfectly on stability by saying almost
/// nothing (miss climbing from 26% to 37% at 0.60, splits and merges staying
/// at zero throughout).
///
/// **Unmeasured**, and the design records this as the constant most likely to
/// need moving once the probe's live-emulation mode has run.
#[doc(hidden)]
pub const T_MARGIN_SHORT: f32 = 0.20;

/// Live centroids from which the cohort test switches on.
///
/// Below four there is no field to stand out from: a spread estimated from
/// one or two other scores says nothing, and normalising against it would be
/// arithmetic pretending to be evidence. So at two and three speakers raw
/// cosine governs and behaviour is exactly what the spike measured.
#[doc(hidden)]
pub const COHORT_MIN: usize = 4;

/// How many standard deviations above the rest of the field the best score
/// must sit, once there is a field ([`COHORT_MIN`]) to measure.
///
/// Adaptive score normalisation, the standard remedy in speaker verification
/// for precisely this crowding, and it is self-tuning: it asks whether this
/// centroid stands out from the others rather than whether it clears a number
/// chosen when the room was smaller.
///
/// **Unmeasured.** 2.0 is the cautious end of the 2-4 band such systems
/// normally operate in, and cautious is the right end: this is the second of
/// two gates, [`T_MARGIN`] already catches the common crowded case, and the
/// spike recorded what over-tightening costs -- a clusterer that says almost
/// nothing scores perfectly on stability while its miss rate climbs from 26%
/// to 37%. The job left over for this gate is the case the margin misses: a
/// winner that beats the runner-up comfortably while the field behind it is
/// broad enough that "comfortably" means very little.
#[doc(hidden)]
pub const T_ZNORM: f32 = 2.0;

/// Cosine between two consecutive 0.75 s hop embeddings below which the voice
/// is taken to have changed.
///
/// The pipeline's only turn-change detector. `window::MAX_GAP_FRAMES` looks
/// like one and measurably is not -- over 13 minutes of AMI the accumulator
/// broke there 151 times, and 48 of the 51 breaks with a decided label on both
/// sides had the *same* speaker either side of the gap -- because real turn
/// changes in a meeting mostly arrive without half a second of silence in
/// front of them. Comparing the voice on both sides of a hop asks the question
/// directly instead of inferring it from a pause.
///
/// Lives here rather than in `window`, next to the other cosine thresholds,
/// because that is what it is; `window` owns what to *do* about a boundary.
///
/// **Unmeasured.** 0.30 sits inside the gap between the spike's two measured
/// populations -- same-speaker window pairs at a median of 0.517,
/// different-speaker at 0.046 -- and nearer the bottom of it, because a missed
/// boundary costs one sentence attributed to the previous speaker while a
/// spurious one costs a window and the label it would have carried. Those
/// medians were measured on 1.5 s windows and these are 0.75 s hops, which are
/// noisier: of the constants added here this is the one most in need of the
/// probe's live-emulation mode. The default of the `diarize_change_threshold`
/// config key.
#[doc(hidden)]
pub const T_CHANGE: f32 = 0.30;

/// How many recent orphan windows are kept while they wait for enough
/// agreement to become a speaker.
///
/// Twice [`MIN_POOL`], which is the whole reasoning: the pool needs room for
/// one interrupting speaker's window between every orphan and the next.
/// Before this the pool was cleared on every assignment, so minting required
/// [`MIN_POOL`] *consecutive* orphans with nobody else talking in between --
/// an extra requirement that was never designed, never measured, and that a
/// room where people take turns turns into an impossibility. `MIN_POOL` is
/// unchanged and still 4: four windows must still agree with each other
/// before a speaker exists. Only the "and nobody else spoke" clause is gone.
///
/// A clusterer at a configured `min_pool` uses `2 * min_pool` when that is
/// larger, so the ring can always hold a mintable set.
#[doc(hidden)]
pub const POOL_RING: usize = 2 * MIN_POOL;

/// One speaker's running identity.
struct Centroid {
    label: u32,
    /// Unit-length, always: every update re-normalises.
    vector: Vec<f32>,
    /// A retired centroid keeps its slot (labels are never reused) but
    /// forwards to the older centroid it turned out to duplicate.
    retired_into: Option<u32>,
}

/// Assigns speaker labels to a stream of embeddings, in order.
///
/// Dimensionality is whatever the first embedding brings (192 for the
/// production ERes2Net model); it must not vary within one clusterer's life.
/// Inputs need not be normalised -- [`OnlineClusterer::observe`] normalises
/// what it is given -- but they must be non-zero to carry any direction.
pub struct OnlineClusterer {
    centroids: Vec<Centroid>,
    /// Orphan windows that matched nobody clearly enough, awaiting enough
    /// agreement to become a new speaker. A ring, oldest at the front: an
    /// assignment no longer empties it, so the evidence for a new voice
    /// survives other people taking their turn.
    pool: std::collections::VecDeque<Vec<f32>>,
    next_label: u32,
    /// The five thresholds, held rather than read from the consts so that the
    /// probe can sweep them and the server can configure `min_pool` and
    /// `margin`. All five are the consts above in every clusterer that ships.
    t_assign: f32,
    t_retire: f32,
    alpha: f32,
    min_pool: usize,
    margin: f32,
}

impl Default for OnlineClusterer {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineClusterer {
    pub fn new() -> Self {
        Self::with_config(MIN_POOL, T_MARGIN)
    }

    /// A clusterer at a configured pool size and margin: the server's
    /// `diarize_min_pool` and `diarize_margin`, whose defaults are
    /// [`MIN_POOL`] and [`T_MARGIN`].
    ///
    /// The two clustering parameters the configuration exposes. The others
    /// have no trade an operator could make with them -- `T_RETIRE` never
    /// fired in any accepted configuration, and `T_ASSIGN` fails quietly
    /// rather than loudly when it is wrong, which is not a knob to hand out.
    /// These two do: how fast a new voice is picked up against how often one
    /// person is split across two labels, and how sure the clusterer has to be
    /// before it names anybody at all.
    ///
    /// Both this and [`OnlineClusterer::new`] reach the fields through
    /// [`OnlineClusterer::with_params`], so there is still exactly one place
    /// the constants are assigned.
    pub fn with_config(min_pool: usize, margin: f32) -> Self {
        Self::with_params(T_ASSIGN, T_RETIRE, EMA_ALPHA, min_pool, margin)
    }

    /// A clusterer with all five thresholds overridden. **Not a configuration
    /// surface**: the server builds every clusterer through
    /// [`OnlineClusterer::new`] or [`OnlineClusterer::with_config`], so
    /// `T_ASSIGN`, `T_RETIRE` and `EMA_ALPHA` are the calibrated constants
    /// above in everything that ships.
    ///
    /// This exists for `examples/diarize_probe`'s sweep, which re-measures
    /// those constants against annotated meetings and therefore has to vary
    /// them without forking the algorithm -- a sweep against a second copy of
    /// this code would answer a question about the copy.
    #[doc(hidden)]
    pub fn with_params(
        t_assign: f32,
        t_retire: f32,
        alpha: f32,
        min_pool: usize,
        margin: f32,
    ) -> Self {
        Self {
            centroids: Vec::new(),
            pool: std::collections::VecDeque::new(),
            next_label: 1,
            t_assign,
            t_retire,
            alpha,
            min_pool,
            margin,
        }
    }

    /// The label this embedding belongs to, or `None` while the clusterer is
    /// still undecided.
    ///
    /// `None` is an honest answer rather than a failure: it means no known
    /// speaker stood out clearly enough to be worth asserting, and there is
    /// not yet enough agreeing evidence to call this a new one. Pooling a
    /// window is not an assignment, and neither is being ambiguous about it --
    /// a window that names nobody moves nothing.
    pub fn observe(&mut self, embedding: &[f32]) -> Option<u32> {
        if let Some(first) = self.centroids.first() {
            // Cosine zips, so a width that changes mid-session truncates in
            // silence and mislabels everything after it. Free in release, a
            // panic anywhere a test or a dev build can see it.
            debug_assert_eq!(
                embedding.len(),
                first.vector.len(),
                "embedding width changed mid-session"
            );
        }
        let embedding = l2_normalize(embedding);

        if let Some(index) = self.admits(&embedding, self.margin) {
            let alpha = self.alpha;
            let centroid = &mut self.centroids[index];
            for (x, e) in centroid.vector.iter_mut().zip(&embedding) {
                *x = (1.0 - alpha) * *x + alpha * e;
            }
            centroid.vector = l2_normalize(&centroid.vector);
            let label = centroid.label;

            // The pool is *not* cleared. An assignment says who was talking
            // just now, which is no evidence at all about the orphan windows
            // waiting to become somebody else -- and treating it as evidence
            // is what made a new voice in a busy room unmintable, since it
            // demanded MIN_POOL orphans with nobody else speaking between.
            //
            // That nudge may have walked this centroid into another one.
            self.retire_converged();
            return Some(self.resolve(label));
        }

        self.pool.push_back(embedding);
        while self.pool.len() > self.ring() {
            self.pool.pop_front();
        }

        let group = self.agreeing_group();
        if group.len() < self.min_pool {
            return None;
        }

        let mean = self.pool_mean(&group);
        // Two ways a pool that agreed with itself must still not mint.
        //
        // The first is the original rule: their mean is the less noisy vector
        // and it has landed inside a speaker who already has a label, so
        // minting would split that speaker. It asks the question with the same
        // rule assignment asks it with, rather than with a bare threshold --
        // in a crowded room "nearest incumbent is over 0.45" is true of
        // *every* new voice, and a bare threshold there refuses to ever mint
        // anybody again, which is the failure this design exists to fix.
        //
        // The second covers what the first gives up. An ambiguous mean is not
        // assigned, so the first test passes it; but a mean sitting within
        // `T_RETIRE` of a live centroid would mint a duplicate that retirement
        // would immediately fold back, so there was never a speaker there to
        // find.
        if self.admits(&mean, self.margin).is_some()
            || self
                .nearest(&mean)
                .is_some_and(|(_, similarity)| similarity >= self.t_retire)
        {
            self.forget(&group);
            return None;
        }

        let label = self.next_label;
        self.next_label += 1;
        self.centroids.push(Centroid {
            label,
            vector: mean,
            retired_into: None,
        });
        // Only the windows that became this speaker leave. Anything else in
        // the ring is still evidence about somebody else.
        self.forget(&group);
        Some(label)
    }

    /// The label a short hop embedding *probably* belongs to.
    ///
    /// Takes `&self`, and that is the contract rather than an accident: a
    /// 0.75 s embedding never updates a centroid, never enters the pool, and
    /// never mints. It exists to name a turn about a second earlier than a
    /// full window can, and the price of that speed is that it is noisier --
    /// so it is allowed to *read* the clusterer's state and never to write it.
    /// Keeping centroid quality on the 1.5 s windows is what preserves
    /// everything the spike measured.
    ///
    /// Held to [`T_MARGIN_SHORT`] rather than [`T_MARGIN`], because a guess
    /// made on half the evidence should have to be twice as clear.
    pub fn observe_short(&self, embedding: &[f32]) -> Option<u32> {
        let embedding = l2_normalize(embedding);
        self.admits(&embedding, T_MARGIN_SHORT)
            .map(|index| self.resolve(self.centroids[index].label))
    }

    /// How many orphans the ring holds. [`POOL_RING`] at the shipped pool, and
    /// never less than room for a mintable set plus one interruption each.
    fn ring(&self) -> usize {
        POOL_RING.max(2 * self.min_pool)
    }

    /// The index of the live centroid `embedding` may join, or `None` when
    /// nobody stands out far enough to be worth asserting.
    ///
    /// This is the whole of the assignment decision: threshold, margin and, in
    /// a crowded room, the cohort test. Retired centroids do not compete --
    /// they are forwarding addresses, not speakers.
    fn admits(&self, embedding: &[f32], margin: f32) -> Option<usize> {
        let live: Vec<usize> = (0..self.centroids.len())
            .filter(|&i| self.centroids[i].retired_into.is_none())
            .collect();
        let scores: Vec<f32> = live
            .iter()
            .map(|&i| cosine(embedding, &self.centroids[i].vector))
            .collect();
        stands_out(&scores, self.t_assign, margin).map(|k| live[k])
    }

    /// The nearest live centroid and its similarity, by index. Retired
    /// centroids are invisible here: they exist only to forward.
    ///
    /// The bare argmax, which assignment no longer uses on its own -- see
    /// [`stands_out`] for why. What is left is the two questions where
    /// "nearest" really is the whole question: whether a mint would duplicate
    /// an existing centroid, and the diagnostics the probe reports.
    fn nearest(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        self.centroids
            .iter()
            .enumerate()
            .filter(|(_, c)| c.retired_into.is_none())
            .map(|(i, c)| (i, cosine(embedding, &c.vector)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The largest set of pooled orphans that contains the newest and whose
    /// members all agree with each other, as ring indices, newest first.
    ///
    /// The pool holds evidence for *one* new speaker, so agreement is still
    /// every-pair and still measured at `T_ASSIGN`; what has gone is the
    /// requirement that the agreeing windows be the *whole* pool. With a ring
    /// that other speakers' windows no longer empty, one poisoned window used
    /// to cost a mint and then a window per attempt to recover from -- now it
    /// is simply not in the group.
    ///
    /// Greedy from the newest, and anchored there on purpose rather than as an
    /// approximation: the newest orphan is the window being decided, and a
    /// group that did not contain it would mint a speaker out of somebody
    /// else's audio and then hand this window the label.
    ///
    /// Quadratic in the ring, which is 8 as shipped.
    fn agreeing_group(&self) -> Vec<usize> {
        let mut group: Vec<usize> = Vec::new();
        for i in (0..self.pool.len()).rev() {
            if group
                .iter()
                .all(|&j| cosine(&self.pool[i], &self.pool[j]) >= self.t_assign)
            {
                group.push(i);
            }
        }
        group
    }

    /// Drop `group` from the ring. Indices descend so earlier removals cannot
    /// invalidate later ones.
    fn forget(&mut self, group: &[usize]) {
        let mut sorted = group.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for i in sorted {
            self.pool.remove(i);
        }
    }

    fn pool_mean(&self, group: &[usize]) -> Vec<f32> {
        let mut mean = vec![0.0f32; self.pool[group[0]].len()];
        for &i in group {
            for (m, x) in mean.iter_mut().zip(&self.pool[i]) {
                *m += x;
            }
        }
        l2_normalize(&mean)
    }

    /// Retire every live centroid that has converged on an older one. Past
    /// labels are never rewritten, so the newer of the pair is the one that
    /// gives way; retiring one can expose another pair, hence the loop.
    fn retire_converged(&mut self) {
        while let Some((older, newer)) = self.converged_pair() {
            self.centroids[newer].retired_into = Some(self.centroids[older].label);
        }
    }

    /// The first converged pair in mint order, as `(older, newer)` indices.
    /// Centroids are only ever appended, so a lower index is always the
    /// earlier speaker.
    ///
    /// Quadratic in the number of centroids, and deliberately: one session is
    /// a handful of speakers in a room, so this is a few dozen dot products
    /// on an assignment that already cost an embedding.
    fn converged_pair(&self) -> Option<(usize, usize)> {
        let live = |i: usize| self.centroids[i].retired_into.is_none();
        (0..self.centroids.len())
            .filter(|&i| live(i))
            .find_map(|i| {
                ((i + 1)..self.centroids.len())
                    .find(|&j| {
                        live(j)
                            && cosine(&self.centroids[i].vector, &self.centroids[j].vector)
                                >= self.t_retire
                    })
                    .map(|j| (i, j))
            })
    }

    /// Live labels, oldest first. Test-only: the contract callers see is
    /// [`OnlineClusterer::observe`]'s return value, but retirement happens
    /// *during* a call, and pinning what that call returns needs a look
    /// inside.
    #[cfg(test)]
    fn live_labels(&self) -> Vec<u32> {
        self.centroids
            .iter()
            .filter(|c| c.retired_into.is_none())
            .map(|c| c.label)
            .collect()
    }

    /// Every centroid's vector, retired ones included, in mint order.
    /// Test-only, and for one question: whether a window that named nobody
    /// moved anything. Drift is invisible from `observe`'s return values until
    /// long after it has happened, which is exactly why it went unnoticed.
    #[cfg(test)]
    fn vectors(&self) -> Vec<Vec<f32>> {
        self.centroids.iter().map(|c| c.vector.clone()).collect()
    }

    /// Follow retirement forwarding to the label a caller should see. Chains
    /// terminate: a centroid only ever forwards to an older one.
    fn resolve(&self, mut label: u32) -> u32 {
        while let Some(next) = self
            .centroids
            .iter()
            .find(|c| c.label == label)
            .and_then(|c| c.retired_into)
        {
            label = next;
        }
        label
    }

    // ------------------------------------------------------- diagnostics
    //
    // Three questions a caller cannot answer from `observe`'s return values,
    // and which `examples/diarize_probe` reports for every configuration it
    // scores. Hidden from the docs and used by nothing the server runs: the
    // shipped contract is still one label per embedding.

    /// Labels ever minted, retired ones included. Against
    /// [`OnlineClusterer::active`] this separates "found five speakers" from
    /// "found nine and retired four of them".
    #[doc(hidden)]
    pub fn minted(&self) -> u32 {
        self.next_label - 1
    }

    /// Centroids still attracting windows.
    #[doc(hidden)]
    pub fn active(&self) -> usize {
        self.centroids
            .iter()
            .filter(|c| c.retired_into.is_none())
            .count()
    }

    /// Cosine similarity between every pair of live centroids, closest pair
    /// first.
    ///
    /// The head of this list against `T_ASSIGN` is the margin the clusterer
    /// had left -- how much room a meeting with more people in it would still
    /// have. It is the measurement behind the design doc's warning that at
    /// five speakers the two closest centroids sit at 0.519, above `T_ASSIGN`,
    /// and that eight is untested.
    #[doc(hidden)]
    pub fn crowding(&self) -> Vec<(u32, u32, f32)> {
        let live: Vec<&Centroid> = self
            .centroids
            .iter()
            .filter(|c| c.retired_into.is_none())
            .collect();
        let mut pairs: Vec<(u32, u32, f32)> = live
            .iter()
            .enumerate()
            .flat_map(|(i, a)| {
                live[i + 1..]
                    .iter()
                    .map(move |b| (a.label, b.label, cosine(&a.vector, &b.vector)))
            })
            .collect();
        pairs.sort_by(|a, b| b.2.total_cmp(&a.2));
        pairs
    }
}

/// Which of `scores` -- one cosine per live centroid, in any order -- an
/// embedding may be assigned to, or `None` when the answer is not clear
/// enough to assert.
///
/// The whole assignment policy, as a pure function over a score vector,
/// because that is the shape it can be tested in: the interesting cases are
/// score patterns, not clusterer histories.
///
/// Three gates, in order of how often they bite:
///
/// 1. **The threshold.** `s1 >= t_assign`, unchanged and still 0.45 -- the
///    middle of the 0.40-0.50 plateau the spike swept.
/// 2. **The margin.** `s1 - s2 >= margin`. A fixed threshold against an
///    argmax gets worse with every centroid added: the more incumbents there
///    are, the likelier one of them clears 0.45 against a voice that is not
///    theirs, and the spike measured two live centroids at 0.519 with only
///    five speakers in the room. Beating the runner-up is the question that
///    does not rot as the room fills.
/// 3. **The cohort.** Once there are [`COHORT_MIN`] live centroids to measure
///    a field from, `(s1 - mean(rest)) / std(rest) >= T_ZNORM`. Adaptive
///    score normalisation: it asks whether this centroid stands out from the
///    field rather than whether it clears a number chosen when the field was
///    smaller. Below four centroids there is no field, so it is inert and two-
///    and three-speaker behaviour is exactly what was measured.
///
/// Every gate is a conjunction with the first, so this function can only ever
/// return `None` where the original bare argmax returned `Some`. It cannot
/// invent an assignment, and therefore cannot introduce a merge; its worst
/// case is a window left unlabelled, which is what `speaker: None` is for.
fn stands_out(scores: &[f32], t_assign: f32, margin: f32) -> Option<usize> {
    let (best, &s1) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if s1 < t_assign {
        return None;
    }

    // NEG_INFINITY when there is nothing to come second, which makes the
    // margin trivially satisfied -- correct, since a lone centroid is the only
    // answer there is.
    let s2 = scores
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best)
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    if s1 - s2 < margin {
        return None;
    }

    if scores.len() < COHORT_MIN {
        return Some(best);
    }
    let rest: Vec<f32> = scores
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best)
        .map(|(_, s)| *s)
        .collect();
    let mean = rest.iter().sum::<f32>() / rest.len() as f32;
    let variance = rest.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / rest.len() as f32;
    // A floor rather than a special case: a field with no spread at all makes
    // any lead infinitely significant, which is the right answer and also the
    // one the division would give if it did not divide by zero first.
    let z = (s1 - mean) / variance.sqrt().max(1e-6);
    (z >= T_ZNORM).then_some(best)
}

/// A unit-length copy. The zero guard keeps a silent window from producing
/// NaNs that would poison every later comparison.
///
/// `pub(super)`: `real::embed::Embedder` needs the same normalisation on its
/// raw ONNX output and shares this copy rather than keeping its own.
pub(super) fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / norm).collect()
}

/// Cosine similarity, which for unit-length inputs is just the dot product.
/// Every vector this module compares has been normalised first.
///
/// `pub`, unlike [`l2_normalize`]'s `pub(super)`, because the consumer is
/// outside the crate: `examples/diarize_probe` measures how far apart two
/// voices are before any clustering happens, and that number is only
/// comparable to `T_ASSIGN` if it is the same similarity the clusterer
/// decides with. `Embedder::embed` returns unit vectors, so the caller's
/// contract is met.
#[doc(hidden)]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic embeddings live in 8 dimensions, not the model's 192: small
    /// enough to reason about on paper, and proof that nothing here is
    /// hardcoded to the real width.
    const DIM: usize = 8;
    /// Per-axis jitter for a "noisy" window. Big enough that same-speaker
    /// windows differ, small enough that they stay well above `T_ASSIGN` of
    /// each other -- which the first test asserts rather than assumes.
    const JITTER: f32 = 0.2;

    /// An embedding with the given weights on the given axes, zero elsewhere.
    /// Every cosine in these tests can be worked out by hand this way.
    fn embedding(components: &[(usize, f32)]) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        for &(axis, weight) in components {
            v[axis] = weight;
        }
        v
    }

    /// One synthetic speaker: a unit vector on its own axis, so two speakers
    /// are exactly orthogonal.
    fn voice(speaker: usize) -> Vec<f32> {
        embedding(&[(speaker, 1.0)])
    }

    /// A window of `speaker` with jitter on every axis -- as close as these
    /// tests get to a real embedding. Deterministic on purpose: the same
    /// numbers on every machine and every run, so a failure is a bug and
    /// never a reroll.
    fn noisy(speaker: usize, index: u32) -> Vec<f32> {
        let mut v = voice(speaker);
        for (axis, x) in v.iter_mut().enumerate() {
            let seed = (speaker as u32) * 7919 + index * (DIM as u32) + axis as u32;
            *x += JITTER * pseudo_noise(seed);
        }
        v
    }

    /// An integer hash spread over [-1, 1). Not an RNG, and deliberately not
    /// one: no crate, no seed, no state.
    fn pseudo_noise(seed: u32) -> f32 {
        let mut h = seed.wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        h = h.wrapping_mul(2_246_822_519);
        h ^= h >> 13;
        (h >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// Cosine between two unnormalised synthetic embeddings.
    fn similarity(a: &[f32], b: &[f32]) -> f32 {
        cosine(&l2_normalize(a), &l2_normalize(b))
    }

    // ------------------------------------------------------- a crowded room
    //
    // Everything above this line runs on two synthetic speakers, which is how
    // the eight-speaker failure stayed a prediction in a design document for
    // as long as it did. These fixtures build rooms instead.

    /// Per-axis jitter for a window in a room. Smaller than [`JITTER`] because
    /// a room is up to sixteen dimensions wide and the noise adds up over all
    /// of them; every test asserts the geometry it needs rather than trusting
    /// this number.
    const ROOM_JITTER: f32 = 0.05;

    /// One window of `voice`, jittered deterministically. `seed` makes two
    /// windows of one voice differ without making them disagree.
    fn window_of(voice: &[f32], seed: u32) -> Vec<f32> {
        voice
            .iter()
            .enumerate()
            .map(|(axis, x)| {
                x + ROOM_JITTER
                    * pseudo_noise(seed.wrapping_mul(2_654_435_761).wrapping_add(axis as u32))
            })
            .collect()
    }

    /// `n` mutually orthogonal voices in `n` dimensions: the easy geometry,
    /// where nothing is about crowding and the pool rules are alone under
    /// test.
    fn separated_room(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; n];
                v[i] = 1.0;
                v
            })
            .collect()
    }

    /// A meeting where people arrive one at a time, each newcomer's opening
    /// windows interleaved with whoever is already talking, and then everyone
    /// takes short turns.
    ///
    /// The shape of a real room, and the shape the old pool could not mint
    /// from: it was emptied by every assignment, so a newcomer needed
    /// `MIN_POOL` windows with *nobody else speaking in between*, which is
    /// exactly what a conversation never supplies. Returns `(voice, window)`
    /// pairs so a test can say who each window really was.
    fn arrivals(voices: &[Vec<f32>]) -> Vec<(usize, Vec<f32>)> {
        let mut script = Vec::new();
        let mut seed = 1u32;
        let mut push = |script: &mut Vec<(usize, Vec<f32>)>, v: usize, voice: &Vec<f32>| {
            seed += 1;
            script.push((v, window_of(voice, seed)));
        };
        for (v, voice) in voices.iter().enumerate() {
            for _ in 0..MIN_POOL {
                push(&mut script, v, voice);
                for (p, prior) in voices[..v].iter().enumerate() {
                    push(&mut script, p, prior);
                }
            }
        }
        for _ in 0..3 {
            for (v, voice) in voices.iter().enumerate() {
                push(&mut script, v, voice);
            }
        }
        script
    }

    /// Run a script and report which label dominated each voice's own
    /// windows, plus the clusterer's live labels. `None` for a voice that was
    /// never labelled at all.
    fn who_got_what(
        mut c: OnlineClusterer,
        voices: usize,
        script: &[(usize, Vec<f32>)],
    ) -> (Vec<Option<u32>>, Vec<u32>) {
        let mut counts: Vec<Vec<(u32, usize)>> = vec![Vec::new(); voices];
        for (v, window) in script {
            if let Some(label) = c.observe(window) {
                match counts[*v].iter_mut().find(|(k, _)| *k == label) {
                    Some((_, n)) => *n += 1,
                    None => counts[*v].push((label, 1)),
                }
            }
        }
        let heard = counts
            .into_iter()
            .map(|c| c.into_iter().max_by_key(|(_, n)| *n).map(|(l, _)| l))
            .collect();
        (heard, c.live_labels())
    }

    /// Assert that every voice was labelled and no two voices share a label.
    fn everyone_kept_their_own_label(heard: &[Option<u32>], live: &[u32]) {
        assert!(
            heard.iter().all(Option::is_some),
            "somebody was never labelled at all: {heard:?}"
        );
        let mut distinct: Vec<Option<u32>> = heard.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            heard.len(),
            "two voices merged onto one label: {heard:?}"
        );
        assert_eq!(
            live.len(),
            heard.len(),
            "{live:?} live centroids for {} voices",
            heard.len()
        );
    }

    #[test]
    fn a_room_of_eight_arriving_one_at_a_time_mints_all_eight() {
        // The scale no test reached before, and the reason the eight-speaker
        // failure was a prediction rather than a red test. Orthogonal voices,
        // so nothing here is about crowding: this is the pool rule alone.
        let voices = separated_room(8);
        let script = arrivals(&voices);
        assert!(
            similarity(&script[0].1, &script[2].1) >= T_ASSIGN,
            "two windows of one voice have to agree, or nobody could ever mint"
        );
        assert!(similarity(&voices[0], &voices[1]) < T_ASSIGN);

        let (heard, live) = who_got_what(OnlineClusterer::new(), 8, &script);
        everyone_kept_their_own_label(&heard, &live);
    }

    #[test]
    fn a_room_of_twelve_arriving_one_at_a_time_mints_all_twelve() {
        // Past the design's stated ceiling of ten, because the failure mode
        // was that the probability of a stray assignment rises with every
        // centroid added -- so the test has to climb until it would bite.
        let voices = separated_room(12);
        let script = arrivals(&voices);
        let (heard, live) = who_got_what(OnlineClusterer::new(), 12, &script);
        everyone_kept_their_own_label(&heard, &live);
    }

    /// A room where four mutually orthogonal founders are already talking and
    /// four later voices each arrive sitting `CROWDED` from *several* of the
    /// incumbents at once.
    ///
    /// This is the failure the 2026-08-24 design named and never tested: "the
    /// margin is gone at five speakers, and eight is untested. If a meeting is
    /// going to fail, this is how." An argmax against a fixed threshold hands
    /// each newcomer to whichever incumbent happens to be nearest -- all of
    /// them are over 0.45 -- so the newcomer is assigned, the pool is emptied,
    /// and they are never minted at all.
    ///
    /// The crowding is deliberately between a newcomer and the *incumbents*
    /// rather than between everybody. A room where two people's raw windows
    /// all sit above `T_ASSIGN` of each other is a room where the embeddings
    /// do not separate those two people, and no clustering rule can fix that
    /// -- it would break the pool's agreement test in exactly the same way it
    /// breaks assignment. What the spike measured, and what this reproduces,
    /// is centroids crowding a newcomer, not voices that are indistinguishable.
    fn crowded_room() -> Vec<Vec<f32>> {
        /// Cosine between a newcomer and each founder it crowds. Above
        /// `T_ASSIGN`, which is the whole point.
        const CROWDED: f32 = 0.5;
        const DIM: usize = 12;
        let axis = |i: usize| {
            let mut v = vec![0.0f32; DIM];
            v[i] = 1.0;
            v
        };
        let founders: Vec<Vec<f32>> = (0..4).map(axis).collect();

        // A newcomer leans equally on founders 0 and 1 and keeps a private
        // axis of its own: with a share of `a` on each of two founders,
        // `2a^2 = CROWDED` puts it at `CROWDED` from both and 0 from the
        // other two.
        let share = (CROWDED / 2.0).sqrt();
        let newcomers: Vec<Vec<f32>> = (0..4)
            .map(|j| {
                let mut v = vec![0.0f32; DIM];
                v[0] = share;
                v[1] = share;
                v[4 + j] = (1.0 - CROWDED).sqrt();
                v
            })
            .collect();

        let room: Vec<Vec<f32>> = founders.into_iter().chain(newcomers).collect();
        // Stated rather than assumed, because the whole test rests on it.
        assert!(similarity(&room[0], &room[1]) < T_ASSIGN, "founders apart");
        for j in 4..8 {
            for f in 0..2 {
                let sim = similarity(&room[j], &room[f]);
                assert!(
                    sim > T_ASSIGN,
                    "newcomer {j} sits at {sim} from founder {f}, which has to \
                     be above the threshold to reproduce the failure"
                );
            }
        }
        room
    }

    #[test]
    fn a_newcomer_crowded_by_several_incumbents_is_still_minted() {
        let voices = crowded_room();
        let script = arrivals(&voices);
        let (heard, live) = who_got_what(OnlineClusterer::new(), voices.len(), &script);
        everyone_kept_their_own_label(&heard, &live);
    }

    #[test]
    fn a_crowded_newcomer_is_withheld_rather_than_handed_to_an_incumbent() {
        // The same room, watched one decision at a time, because "everybody
        // was minted" does not say *why*. A newcomer's first window has three
        // incumbents over `T_ASSIGN` and no reason to prefer any of them, so
        // the honest answer is nobody -- and being nobody is what lets it
        // reach the pool at all.
        let voices = crowded_room();
        let mut c = OnlineClusterer::new();
        let mut seed = 1u32;
        let mut window = |v: usize| {
            seed += 1;
            window_of(&voices[v], seed)
        };

        for v in 0..4 {
            for _ in 0..MIN_POOL {
                c.observe(&window(v));
            }
        }
        assert_eq!(c.live_labels(), vec![1, 2, 3, 4], "the founders are in");

        // The newcomer's opening window. Under the old rule this was an
        // assignment to whichever founder won a coin toss at the third
        // decimal place.
        let opening = window(4);
        let (best, similarity) = c.nearest(&opening).expect("four live centroids");
        assert!(
            similarity >= T_ASSIGN,
            "founder {best} sits at {similarity}, so the old rule would have \
             taken it"
        );
        assert_eq!(c.observe(&opening), None, "and the new rule withholds it");

        // Three more, and the newcomer is somebody.
        assert_eq!(c.observe(&window(4)), None);
        assert_eq!(c.observe(&window(4)), None);
        assert_eq!(c.observe(&window(4)), Some(5));
        // Their own windows assign from then on: standing out from a crowded
        // field is exactly what a minted centroid does.
        assert_eq!(c.observe(&window(4)), Some(5));
        // And the founders keep their numbers.
        for v in 0..4 {
            assert_eq!(c.observe(&window(v)), Some(v as u32 + 1));
        }
    }
    #[test]
    fn two_separated_voices_take_labels_in_first_appearance_order() {
        // The synthetic geometry the rest of these tests assume.
        assert!(similarity(&noisy(0, 1), &noisy(0, 2)) >= T_ASSIGN);
        assert!(similarity(&noisy(0, 1), &noisy(1, 1)) < T_ASSIGN);

        let mut c = OnlineClusterer::new();
        let first: Vec<Option<u32>> = (0..MIN_POOL as u32)
            .map(|i| c.observe(&noisy(0, i)))
            .collect();
        let second: Vec<Option<u32>> = (0..MIN_POOL as u32)
            .map(|i| c.observe(&noisy(1, i)))
            .collect();

        assert_eq!(first.last(), Some(&Some(1)));
        assert_eq!(second.last(), Some(&Some(2)));
        assert!(first[..MIN_POOL - 1].iter().all(Option::is_none));
        assert!(second[..MIN_POOL - 1].iter().all(Option::is_none));
        // The first speaker keeps label 1 once the second exists.
        assert_eq!(c.observe(&noisy(0, 99)), Some(1));
    }

    /// A release test build compiles the guard out, and this test with it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "embedding width changed mid-session")]
    fn a_changed_embedding_width_is_caught_before_it_mislabels() {
        let mut c = OnlineClusterer::new();
        for _ in 0..MIN_POOL {
            c.observe(&voice(0));
        }
        c.observe(&[1.0; DIM + 1]);
    }

    #[test]
    fn a_single_noisy_voice_never_splits() {
        let mut c = OnlineClusterer::new();
        let labels: Vec<Option<u32>> = (0..100).map(|i| c.observe(&noisy(0, i))).collect();

        assert!(labels[..MIN_POOL - 1].iter().all(Option::is_none));
        assert!(
            labels[MIN_POOL - 1..].iter().all(|l| *l == Some(1)),
            "one voice produced more than one label: {labels:?}"
        );
    }

    #[test]
    fn one_outlier_mints_nothing_and_an_agreeing_run_mints_once() {
        let mut c = OnlineClusterer::new();
        // A lone window from a voice never heard again.
        assert_eq!(c.observe(&noisy(0, 0)), None);

        // A different voice fills the pool. The outlier disagrees with it and
        // is evicted as the oldest, so the run still mints on its fourth
        // window -- and mints label 1, proving the outlier minted nothing.
        let labels: Vec<Option<u32>> = (0..=MIN_POOL as u32)
            .map(|i| c.observe(&noisy(1, i)))
            .collect();
        assert_eq!(labels, vec![None, None, None, Some(1), Some(1)]);
    }

    #[test]
    fn an_assignment_no_longer_throws_the_pool_away() {
        // This used to assert the opposite, and the opposite is the bug: an
        // assignment cleared the pool, so minting needed MIN_POOL orphans with
        // nobody else speaking in between. MIN_POOL is unchanged -- four
        // windows still have to agree with each other -- but they no longer
        // have to be consecutive, because in a room where people take turns
        // they never are.
        let mut c = OnlineClusterer::new();
        for i in 0..MIN_POOL as u32 {
            c.observe(&noisy(0, i));
        }

        // Three quarters of the evidence for a second speaker...
        for i in 0..MIN_POOL as u32 - 1 {
            assert_eq!(c.observe(&noisy(1, i)), None);
        }
        // ...then the first speaker takes a turn, which says nothing at all
        // about whether the three orphans are somebody.
        assert_eq!(c.observe(&noisy(0, 50)), Some(1));

        // So the fourth orphan still completes the set, interruption and all.
        assert_eq!(c.observe(&noisy(1, 9)), Some(2));
    }

    #[test]
    fn orphans_survive_several_interruptions_and_a_poisoned_window() {
        // The shape a real meeting has: a new voice gets a window in, somebody
        // else answers, they get another in. Plus one stray window that
        // belongs to nobody, which used to cost a whole pool and then one
        // window per attempt to recover from -- with a ring it is simply not
        // in the agreeing group.
        let mut c = OnlineClusterer::new();
        for i in 0..MIN_POOL as u32 {
            c.observe(&noisy(0, i));
        }

        assert_eq!(c.observe(&noisy(1, 0)), None);
        assert_eq!(c.observe(&noisy(0, 20)), Some(1), "the floor changes hands");
        assert_eq!(c.observe(&noisy(2, 77)), None, "a stray window from nobody");
        assert_eq!(c.observe(&noisy(1, 1)), None);
        assert_eq!(c.observe(&noisy(0, 21)), Some(1));
        assert_eq!(c.observe(&noisy(1, 2)), None);
        assert_eq!(c.observe(&noisy(0, 22)), Some(1));
        assert_eq!(
            c.observe(&noisy(1, 3)),
            Some(2),
            "four agreeing windows are four agreeing windows"
        );
        // And the stray is still nobody: it was never in the group that
        // minted, so it did not leave with it, and it has minted nothing.
        assert_eq!(c.live_labels(), vec![1, 2]);
    }

    #[test]
    fn the_ring_forgets_the_oldest_orphan_rather_than_growing() {
        // A meeting is hours long and most of what the diarizer hears is
        // nobody in particular. The ring is what keeps that bounded, and it
        // has to be big enough to hold a mintable set with room for the
        // interruptions between -- twice MIN_POOL, which is what POOL_RING is.
        assert_eq!(POOL_RING, 2 * MIN_POOL);

        let mut c = OnlineClusterer::new();
        // Ring-many windows that agree with nothing, including each other.
        for i in 0..POOL_RING as u32 {
            assert_eq!(c.observe(&voice(i as usize % DIM)), None);
        }
        assert_eq!(c.pool.len(), POOL_RING);
        for i in 0..POOL_RING as u32 {
            c.observe(&voice(i as usize % DIM));
        }
        assert_eq!(c.pool.len(), POOL_RING, "the ring grew past its bound");
    }

    #[test]
    fn one_borderline_window_does_not_unseat_a_settled_centroid() {
        let mut c = OnlineClusterer::new();
        for i in 0..100 {
            c.observe(&noisy(0, i));
        }

        // Well inside T_ASSIGN, but pulling hard towards another axis.
        let borderline = embedding(&[(0, 0.6), (1, 0.8)]);
        assert_eq!(c.observe(&borderline), Some(1));

        // A centroid is its history: that window moved it by EMA_ALPHA, which
        // is nowhere near enough to drag it into the borderline window's own
        // territory. Probed first, because the next speaker-0 window would
        // pull it back and hide a centroid that had moved all the way.
        assert_eq!(c.observe(&voice(1)), None);
        // And it still recognises the speaker it belongs to.
        assert_eq!(c.observe(&voice(0)), Some(1));
        assert_eq!(c.observe(&noisy(0, 200)), Some(1));
    }

    /// Two minted speakers 0.4 apart, plus the voice that drags the newer
    /// one's centroid into the older's: `(clusterer, speaker 2, the voice
    /// between them)`. The geometry the retirement tests rest on is asserted
    /// here rather than assumed.
    ///
    /// **The clusterer this returns has the margin rule switched off**, and
    /// that is the point of the fixture rather than a convenience. `between`
    /// sits at 0.85 from one centroid and 0.82 from the other, which is
    /// exactly the ambiguity `T_MARGIN` now refuses -- so at the shipped
    /// margin nothing here would drift and there would be nothing to retire.
    /// The drift is the mechanism retirement exists to clean up after; these
    /// tests are about the cleaning up, and
    /// `an_ambiguous_window_moves_no_centroid_at_all` is about the refusal.
    fn converging_pair() -> (OnlineClusterer, Vec<f32>, Vec<f32>) {
        // Speaker 1 sits on axis 0. Speaker 2 sits 0.4 away from it -- far
        // enough apart to mint separately.
        let second = embedding(&[(0, 0.4), (1, 0.84_f32.sqrt())]);
        // A voice between the two, but nearer the newer one, so every window
        // assigns to speaker 2 and drags its centroid towards speaker 1's --
        // to within T_RETIRE, in the end.
        let between = embedding(&[(0, 0.82), (1, 0.3276_f32.sqrt())]);
        assert!(similarity(&voice(0), &second) < T_ASSIGN);
        assert!(similarity(&between, &second) > similarity(&between, &voice(0)));
        assert!(similarity(&between, &voice(0)) >= T_RETIRE);
        assert!(
            similarity(&between, &second) - similarity(&between, &voice(0)) < T_MARGIN,
            "the fixture's whole premise is that `between` is ambiguous"
        );

        let mut c = OnlineClusterer::with_params(T_ASSIGN, T_RETIRE, EMA_ALPHA, MIN_POOL, 0.0);
        for _ in 0..MIN_POOL {
            c.observe(&voice(0));
        }
        for _ in 0..MIN_POOL {
            c.observe(&second);
        }
        assert_eq!(c.live_labels(), vec![1, 2]);
        (c, second, between)
    }

    #[test]
    fn converged_centroids_retire_the_newer_into_the_older() {
        let (mut c, second, between) = converging_pair();

        let mut labels = Vec::new();
        let mut retired_at = None;
        for i in 0..100 {
            let live = c.live_labels();
            labels.push(c.observe(&between));
            if c.live_labels() != live {
                assert!(retired_at.is_none(), "retired more than once");
                // The older label survives; the newer is the one that goes.
                assert_eq!(c.live_labels(), vec![1]);
                retired_at = Some(i);
            }
        }

        let retired_at = retired_at.expect("the centroids never converged");
        assert!(labels[..retired_at].iter().all(|l| *l == Some(2)));
        // The call that retires speaker 2 already answers with speaker 1: a
        // caller never sees a label that died inside its own call. No third
        // speaker is invented anywhere along the way.
        assert!(labels[retired_at..].iter().all(|l| *l == Some(1)));
        // And speaker 2's own voice now answers to speaker 1's label.
        assert_eq!(c.observe(&second), Some(1));
    }

    #[test]
    fn a_retired_centroid_no_longer_attracts_windows() {
        let (mut c, second, between) = converging_pair();
        let mut driven = 0;
        while c.live_labels() != vec![1] {
            c.observe(&between);
            driven += 1;
            assert!(driven < 100, "the centroids never converged");
        }

        // Speaker 1's centroid has not moved -- every window so far went to
        // speaker 2 -- so it is still on axis 0, while speaker 2's retired
        // vector is frozen somewhere between its own voice and `between`.
        // This probe is far from the live centroid and close to the retired
        // one, and a retired centroid is a forwarding address, not a
        // speaker: it must attract nothing.
        let probe = embedding(&[(0, 0.2), (1, 0.96_f32.sqrt())]);
        assert!(similarity(&probe, &voice(0)) < T_ASSIGN);
        assert!(similarity(&probe, &second) >= T_ASSIGN);
        assert!(similarity(&probe, &between) >= T_ASSIGN);
        assert_eq!(c.observe(&probe), None);
    }

    #[test]
    fn labels_are_never_reused_after_retirement() {
        let (mut c, second, between) = converging_pair();
        for _ in 0..100 {
            c.observe(&between);
        }
        assert_eq!(c.observe(&second), Some(1), "label 2 should have retired");

        // A genuinely new voice takes the next label, not the free one.
        let labels: Vec<Option<u32>> = (0..MIN_POOL).map(|_| c.observe(&voice(7))).collect();
        assert_eq!(labels.last(), Some(&Some(3)));
    }

    #[test]
    fn new_does_not_drift_from_the_calibrated_constants() {
        // Narrow on purpose, and worth being clear about what it does not
        // prove: `new` delegates through `with_config` to `with_params`, so
        // this can only fail if a later edit stops it delegating and passes
        // something other than the constants. It says nothing about whether
        // `with_params` reads the arguments it is handed -- the tests below
        // are what cover that, and they are the ones the sweep actually rests
        // on.
        let (_, second, between) = converging_pair();

        // One script through all five constants: pooling and minting
        // (MIN_POOL), assignment (T_ASSIGN), a nudged centroid (EMA_ALPHA), a
        // hundred ambiguous windows that the margin refuses (T_MARGIN) and
        // whose own mean is too close to a live centroid to mint (T_RETIRE),
        // and a fresh voice afterwards to show numbering carries on.
        let mut script: Vec<Vec<f32>> = (0..MIN_POOL as u32).map(|i| noisy(0, i)).collect();
        script.push(noisy(1, 50));
        script.push(noisy(0, 10));
        script.extend(std::iter::repeat_n(second.clone(), MIN_POOL));
        script.extend(std::iter::repeat_n(between, 100));
        script.push(second);
        script.extend(std::iter::repeat_n(voice(7), MIN_POOL));

        let run = |mut c: OnlineClusterer| {
            let labels: Vec<Option<u32>> = script.iter().map(|e| c.observe(e)).collect();
            (labels, c.live_labels())
        };
        let shipped = run(OnlineClusterer::new());
        let swept = run(OnlineClusterer::with_params(
            T_ASSIGN, T_RETIRE, EMA_ALPHA, MIN_POOL, T_MARGIN,
        ));
        assert_eq!(shipped, swept);

        // And the script is worth comparing: an equivalence that held only
        // because nothing happened would pass this test while proving nothing.
        let (labels, live) = shipped;
        assert_eq!(
            live,
            vec![1, 2, 3],
            "nobody retires here any more: the ambiguous windows that used to \
             drag speaker 2 into speaker 1 are now refused outright"
        );
        for expected in [None, Some(1), Some(2), Some(3)] {
            assert!(labels.contains(&expected), "{expected:?} never came up");
        }
    }

    // ---------------------------------------------- the swept parameters
    //
    // One test per parameter, each holding the other three at the shipped
    // constant and asserting that the swept value behaves *differently* from
    // `new()` on a script built so that this parameter alone decides the
    // outcome. Equality against `new()` cannot show this: `new` delegates, so
    // a `with_params` that accepted an argument and then ignored it would
    // satisfy every equality test in this file while silently pinning the
    // probe's 2640-cell sweep to a single configuration. These are the tests
    // that would fail if that happened, and the reason the sweep's axes can be
    // believed at all.

    /// The labels a fresh clusterer emits for a script, and the live centroids
    /// it is left holding.
    fn observe_all(mut c: OnlineClusterer, script: &[Vec<f32>]) -> (Vec<Option<u32>>, Vec<u32>) {
        let labels = script.iter().map(|e| c.observe(e)).collect();
        (labels, c.live_labels())
    }

    #[test]
    fn a_smaller_min_pool_mints_on_less_evidence() {
        // Two agreeing orphans: enough to mint at a pool of 2, not at 4.
        let script = vec![noisy(0, 0), noisy(0, 1)];
        assert!(
            similarity(&script[0], &script[1]) >= T_ASSIGN,
            "the two windows must agree, or neither pool would mint"
        );

        let (shipped, _) = observe_all(OnlineClusterer::new(), &script);
        let (swept, _) = observe_all(
            OnlineClusterer::with_params(T_ASSIGN, T_RETIRE, EMA_ALPHA, 2, T_MARGIN),
            &script,
        );
        assert_eq!(shipped, vec![None, None], "MIN_POOL 4 wants four windows");
        assert_eq!(swept, vec![None, Some(1)], "a pool of 2 should have minted");
    }

    #[test]
    fn the_configured_pool_is_the_one_that_decides() {
        // `with_config` is the constructor a configured server goes through,
        // so the two things worth pinning are that it is `new` when handed the
        // shipped value -- an unconfigured deployment must not drift -- and
        // that it is genuinely a different clusterer when handed another. The
        // same two agreeing windows separate the cases.
        let script = vec![noisy(0, 0), noisy(0, 1)];
        assert_eq!(
            observe_all(OnlineClusterer::with_config(MIN_POOL, T_MARGIN), &script),
            observe_all(OnlineClusterer::new(), &script),
            "the default must be the calibrated clusterer, exactly"
        );

        let (configured, _) = observe_all(OnlineClusterer::with_config(2, T_MARGIN), &script);
        assert_eq!(
            configured,
            vec![None, Some(1)],
            "a configured pool of 2 should have minted where 4 waits"
        );
    }

    #[test]
    fn a_higher_t_assign_refuses_an_assignment_the_shipped_one_takes() {
        // Four identical windows mint under any threshold, so both clusterers
        // reach the same centroid and only the fifth window is in question.
        let mut script = vec![voice(0); MIN_POOL];
        script.push(noisy(0, 99));

        let near = similarity(&noisy(0, 99), &voice(0));
        assert!(
            (T_ASSIGN..0.99).contains(&near),
            "the probe window sits at {near}, which has to be between the \
             shipped threshold and the swept one for this to discriminate"
        );

        let (shipped, _) = observe_all(OnlineClusterer::new(), &script);
        let (swept, _) = observe_all(
            OnlineClusterer::with_params(0.99, T_RETIRE, EMA_ALPHA, MIN_POOL, T_MARGIN),
            &script,
        );
        assert_eq!(shipped.last(), Some(&Some(1)), "0.45 should have assigned");
        assert_eq!(swept.last(), Some(&None), "0.99 should have refused");
    }

    #[test]
    fn a_larger_alpha_drags_the_centroid_further() {
        // Mint on speaker 0, assign one window two-thirds of the way towards
        // speaker 1, then ask whether speaker 1 is now close enough to join.
        // At 0.05 the centroid barely moved and is not; at 1.0 it jumped the
        // whole way and is.
        let between = embedding(&[(0, 0.6), (1, 0.8)]);
        assert!(
            similarity(&between, &voice(0)) >= T_ASSIGN,
            "the dragging window has to be assignable in the first place"
        );
        let mut script = vec![voice(0); MIN_POOL];
        script.push(between);
        script.push(voice(1));

        let (shipped, _) = observe_all(OnlineClusterer::new(), &script);
        let (swept, _) = observe_all(
            OnlineClusterer::with_params(T_ASSIGN, T_RETIRE, 1.0, MIN_POOL, T_MARGIN),
            &script,
        );
        assert_eq!(
            shipped.last(),
            Some(&None),
            "alpha 0.05 should have left speaker 1 out of reach"
        );
        assert_eq!(
            swept.last(),
            Some(&Some(1)),
            "alpha 1.0 should have moved the centroid onto the new window"
        );
    }

    #[test]
    fn a_lower_t_retire_folds_two_centroids_the_shipped_one_keeps() {
        // Two speakers 0.4 apart: far enough to mint separately (below
        // T_ASSIGN), close enough that a retire threshold of 0.35 calls them
        // duplicates while the shipped 0.80 does not. The last window assigns
        // rather than mints, which is what runs the convergence check.
        let (_, second, _) = converging_pair();
        let apart = similarity(&voice(0), &second);
        assert!(
            (0.35..T_ASSIGN).contains(&apart),
            "the two speakers sit at {apart}, which has to fall between the \
             swept retire threshold and T_ASSIGN"
        );

        let mut script = vec![voice(0); MIN_POOL];
        script.extend(std::iter::repeat_n(second, MIN_POOL));
        script.push(voice(0));

        let (shipped, shipped_live) = observe_all(OnlineClusterer::new(), &script);
        let (swept, swept_live) = observe_all(
            OnlineClusterer::with_params(T_ASSIGN, 0.35, EMA_ALPHA, MIN_POOL, T_MARGIN),
            &script,
        );
        assert_eq!(
            shipped_live,
            vec![1, 2],
            "0.80 should have kept both speakers"
        );
        assert_eq!(swept_live, vec![1], "0.35 should have retired the newer");
        // Both still answer the final window with speaker 1 -- the difference
        // is only visible in what survives, which is why this asserts on the
        // live centroids and not on the labels alone.
        assert_eq!(shipped.last(), Some(&Some(1)));
        assert_eq!(swept.last(), Some(&Some(1)));
    }

    // ------------------------------------------- the admission rules, alone
    //
    // `stands_out` is the whole assignment policy as a function of a score
    // vector, and score patterns are what the interesting cases are made of.
    // Testing it here rather than through a clusterer history is what lets a
    // case be written down as the four numbers that produce it.

    #[test]
    fn a_lone_centroid_has_nothing_to_beat() {
        // The margin is a comparison, and with one centroid there is nothing
        // to compare against -- so a single speaker's session must behave
        // exactly as it did before any of this existed.
        assert_eq!(stands_out(&[0.46], T_ASSIGN, T_MARGIN), Some(0));
        assert_eq!(stands_out(&[0.44], T_ASSIGN, T_MARGIN), None);
        assert_eq!(stands_out(&[], T_ASSIGN, T_MARGIN), None);
    }

    #[test]
    fn a_tie_between_two_centroids_names_neither() {
        // The 0.451-versus-0.450 case the design named: decided by the third
        // decimal, with no rejection, and then written into a centroid.
        assert_eq!(stands_out(&[0.451, 0.450], T_ASSIGN, T_MARGIN), None);
        // A clear winner is still a winner, and still the argmax rather than
        // the first over the line.
        assert_eq!(stands_out(&[0.450, 0.700], T_ASSIGN, T_MARGIN), Some(1));
    }

    #[test]
    fn the_margin_only_ever_withholds_what_the_bare_argmax_would_have_taken() {
        // The invariant the whole design rests on: these rules cannot
        // manufacture an assignment, so they cannot introduce a merge. Their
        // worst case is a window left unlabelled. Checked exhaustively over a
        // grid of score patterns rather than argued.
        let grid = [0.0f32, 0.2, 0.44, 0.45, 0.46, 0.5, 0.519, 0.6, 0.9];
        for &a in &grid {
            for &b in &grid {
                for &c in &grid {
                    for &d in &grid {
                        for &e in &grid {
                            let scores = [a, b, c, d, e];
                            for n in 1..=scores.len() {
                                let scores = &scores[..n];
                                let Some(i) = stands_out(scores, T_ASSIGN, T_MARGIN) else {
                                    continue;
                                };
                                // Whatever it named, the bare argmax would
                                // have named the same centroid and taken it.
                                let argmax = scores
                                    .iter()
                                    .enumerate()
                                    .max_by(|x, y| x.1.total_cmp(y.1))
                                    .expect("non-empty");
                                assert_eq!(i, argmax.0, "{scores:?}");
                                assert!(*argmax.1 >= T_ASSIGN, "{scores:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_cohort_test_is_inert_below_four_live_centroids() {
        // Two and three speakers must behave exactly as the spike measured
        // them, so the z-norm has to be provably switched off there. These
        // three scores have a wide, awkward spread on purpose: at four
        // centroids the same pattern is refused, and the only difference is
        // that there is now a field to measure.
        assert_eq!(COHORT_MIN, 4);
        let below = [0.65, 0.50, 0.30];
        assert!(below.len() < COHORT_MIN);
        assert_eq!(stands_out(&below, T_ASSIGN, T_MARGIN), Some(0));

        // One more centroid, positioned so the margin is untouched -- the
        // second best is still 0.50 -- and the answer changes.
        let at = [0.65, 0.50, 0.30, 0.00];
        assert_eq!(at.len(), COHORT_MIN);
        assert_eq!(
            stands_out(&at, T_ASSIGN, T_MARGIN),
            None,
            "the cohort has to be the thing that changed, since the margin did not"
        );
    }

    #[test]
    fn the_cohort_test_withholds_where_the_margin_alone_would_not() {
        // What the z-norm is for, and proof it is not dead weight behind the
        // margin. The winner beats the runner-up by twice the margin, so gate
        // two passes -- but the field behind it is broad, so "twice the
        // margin" is barely more than the spread of everybody else.
        let broad = [0.75, 0.55, 0.50, 0.05, 0.00];
        assert!(broad[0] - broad[1] > T_MARGIN, "the margin passes");
        assert_eq!(stands_out(&broad, T_ASSIGN, T_MARGIN), None);

        // The same winner against a field that agrees with itself: now it
        // genuinely stands out, and the same 0.75 is taken.
        let tight = [0.75, 0.06, 0.05, 0.04, 0.05];
        assert_eq!(stands_out(&tight, T_ASSIGN, T_MARGIN), Some(0));
    }

    #[test]
    fn a_field_with_no_spread_at_all_does_not_divide_by_zero() {
        // Identical centroids are impossible in practice and cheap to be
        // wrong about: the variance is zero, and a lead over a field with no
        // spread is as significant as a lead gets.
        assert_eq!(
            stands_out(&[0.90, 0.10, 0.10, 0.10, 0.10], T_ASSIGN, T_MARGIN),
            Some(0)
        );
    }

    #[test]
    fn an_ambiguous_window_moves_no_centroid_at_all() {
        // The drift regression, and the reason the margin exists at all.
        // `between` sits 0.85 from one centroid and 0.82 from the other; the
        // old rule resolved that by the third decimal and then wrote the
        // answer into a centroid via EMA, one window at a time, until the two
        // centroids had walked into each other. A window that names nobody
        // must leave the clusterer bit-identical.
        let (_, second, between) = converging_pair();
        let mut c = OnlineClusterer::new();
        for _ in 0..MIN_POOL {
            c.observe(&voice(0));
        }
        for _ in 0..MIN_POOL {
            c.observe(&second);
        }
        assert_eq!(c.live_labels(), vec![1, 2]);

        let before = c.vectors();
        for _ in 0..100 {
            assert_eq!(c.observe(&between), None, "ambiguity is not an answer");
        }
        assert_eq!(c.vectors(), before, "an ambiguous window moved a centroid");
        assert_eq!(c.live_labels(), vec![1, 2], "and nothing retired");
        // Both speakers still answer to their own numbers afterwards.
        assert_eq!(c.observe(&voice(0)), Some(1));
        assert_eq!(c.observe(&second), Some(2));
    }

    // ------------------------------------------------- the short embedding

    #[test]
    fn a_short_embedding_writes_nothing_it_reads() {
        // The contract that keeps a noisier 0.75 s vector from costing
        // anything the spike measured: it may name a speaker and may not
        // change one. `&self` states it, and this is the test that would fail
        // if the signature ever loosened.
        let mut c = OnlineClusterer::new();
        for i in 0..MIN_POOL as u32 {
            c.observe(&noisy(0, i));
        }
        let before = (c.vectors(), c.live_labels(), c.minted());

        for i in 0..50 {
            c.observe_short(&noisy(0, 100 + i));
            c.observe_short(&noisy(1, i));
            c.observe_short(&voice(3));
        }
        assert_eq!(
            (c.vectors(), c.live_labels(), c.minted()),
            before,
            "a short embedding changed the clusterer"
        );
    }

    #[test]
    fn a_short_embedding_names_a_speaker_it_is_sure_of() {
        // The point of it: a label about a second into a turn rather than
        // after a full window and a vote.
        let mut c = OnlineClusterer::new();
        for i in 0..MIN_POOL as u32 {
            c.observe(&noisy(0, i));
        }
        assert_eq!(c.observe_short(&noisy(0, 200)), Some(1));
        assert_eq!(c.observe_short(&voice(1)), None, "a voice it does not know");
    }

    #[test]
    fn a_short_embedding_is_held_to_a_wider_margin_than_a_window() {
        // Two centroids and a vector that beats the runner-up by more than a
        // window needs and less than a hop does. The full window takes it; the
        // hop says nothing and waits for the window.
        let mut c = OnlineClusterer::new();
        for _ in 0..MIN_POOL {
            c.observe(&voice(0));
        }
        for _ in 0..MIN_POOL {
            c.observe(&voice(1));
        }
        assert_eq!(c.live_labels(), vec![1, 2]);

        // cos = 0.7781 to speaker 1 and 0.6281 to speaker 2: a lead of 0.15,
        // which sits between T_MARGIN and T_MARGIN_SHORT.
        let leaning = embedding(&[(0, 0.7781), (1, 0.6281)]);
        let lead = similarity(&leaning, &voice(0)) - similarity(&leaning, &voice(1));
        assert!(
            (T_MARGIN..T_MARGIN_SHORT).contains(&lead),
            "the probe leads by {lead}, which has to fall between the two margins"
        );
        assert_eq!(c.observe_short(&leaning), None, "a hop is not sure enough");
        assert_eq!(c.observe(&leaning), Some(1), "a window is");
    }

    #[test]
    fn a_short_embedding_speaks_only_for_speakers_that_already_exist() {
        // It never mints, so before anyone has a number it has nothing to say
        // -- which is exactly the gap `transcript.relabel` fills afterwards,
        // not a gap a faster guess is allowed to fill by inventing somebody.
        let c = OnlineClusterer::new();
        assert_eq!(c.observe_short(&voice(0)), None);
        assert_eq!(c.minted(), 0);
    }

    #[test]
    fn a_short_embedding_follows_a_retired_label_to_its_forwarding_address() {
        let (mut c, second, between) = converging_pair();
        for _ in 0..100 {
            c.observe(&between);
        }
        assert_eq!(c.live_labels(), vec![1], "the pair should have converged");
        assert_eq!(
            c.observe_short(&second),
            Some(1),
            "speaker 2 retired, so its voice answers to speaker 1"
        );
    }

    #[test]
    fn a_pool_that_agrees_with_a_known_speaker_mints_nothing() {
        let mut c = OnlineClusterer::new();
        for _ in 0..MIN_POOL {
            c.observe(&voice(0));
        }

        // Four windows that agree with each other, each individually just
        // under T_ASSIGN of speaker 1 -- a shared pull off-axis plus private
        // noise on an axis of its own. Averaging cancels the private part, so
        // their mean lands back inside speaker 1's territory.
        let pooled: Vec<Vec<f32>> = (0..MIN_POOL)
            .map(|i| embedding(&[(0, 0.42), (1, 0.5_f32.sqrt()), (2 + i, 0.35_f32.sqrt())]))
            .collect();
        let mut mean = vec![0.0f32; DIM];
        for v in &pooled {
            for (m, x) in mean.iter_mut().zip(v) {
                *m += x;
            }
        }
        for v in &pooled {
            assert!(similarity(v, &voice(0)) < T_ASSIGN);
            assert!(similarity(v, &pooled[0]) >= T_ASSIGN);
        }
        assert!(similarity(&mean, &voice(0)) >= T_ASSIGN);

        // So none of them mints: splitting a speaker who already has a label
        // is the one failure this design exists to avoid.
        for v in &pooled {
            assert_eq!(c.observe(v), None);
        }
        // And the pool is gone, not lingering: the next new voice needs its
        // own full MIN_POOL, and takes label 2 because nothing else was ever
        // minted.
        let labels: Vec<Option<u32>> = (0..MIN_POOL).map(|_| c.observe(&voice(7))).collect();
        assert_eq!(labels, vec![None, None, None, Some(2)]);
    }
}
