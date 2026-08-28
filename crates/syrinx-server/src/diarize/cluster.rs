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
//! "Eager to assign" is qualified in one way: eager to assign *when it is
//! clear who*. A bare argmax of cosine against a fixed threshold is a decision
//! that gets worse every time a centroid is added -- at five speakers the
//! spike measured the two closest live centroids at 0.519, above `T_ASSIGN`,
//! so a genuinely sixth voice has incumbents above the bar to be handed to.
//! Assignment therefore also asks how far the winner beat the runner-up, and
//! in a crowded room how far it stands out from the whole field.
//!
//! Those two gates only ever *withhold*, never manufacture, an assignment, so
//! neither can introduce a merge. What a withheld window costs is bounded on
//! purpose: it names nobody *confidently*, but [`OnlineClusterer::observe`]
//! still reports the argmax as a guess and still files the window with the
//! candidate speaker it agrees with. Withholding buys drift protection -- an
//! ambiguous window moves no centroid -- without buying it out of the miss
//! rate, which is what the spike measured going from 26% to 37% the last time
//! a threshold was tightened.
//!
//! Most constants below are measurements: see "Spike results" in
//! `docs/specs/2026-08-24-speaker-diarization-design.md`, where they were
//! swept against three hand-annotated AMI meetings. The ones added by
//! `docs/specs/2026-08-27-diarization-latency-and-crowding-design.md`
//! -- [`T_MARGIN`], [`SHORT_MARGIN_FACTOR`], [`T_ZNORM`], [`T_MINT_MARGIN`],
//! [`MAX_POOLS`], [`POOL_AGE`] -- are **engineering estimates and have not
//! been measured**; the probe's live-emulation mode exists to replace them.
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

/// How much wider a 0.75 s hop's margin is than a 1.5 s window's.
///
/// A guess made on half the evidence should have to be twice as clear, and
/// that is a statement about the *configured* margin rather than about 0.10:
/// a deployment that raises `diarize_margin` to 0.30 has said it wants more
/// caution, and a hop that stayed at a hard-coded 0.20 would then be the
/// *loosest* rule in the clusterer. So the short margin is derived --
/// [`OnlineClusterer::short_margin`] -- and this is the factor.
///
/// Deliberately a wider *margin* rather than a higher `T_ASSIGN`: the spike
/// measured what raising the absolute threshold does, and it is the failure
/// mode where the clusterer scores perfectly on stability by saying almost
/// nothing (miss climbing from 26% to 37% at 0.60, splits and merges staying
/// at zero throughout).
///
/// **Unmeasured**, and the design records the resulting hop margin as the
/// number most likely to need moving once the probe's live-emulation mode has
/// run.
#[doc(hidden)]
pub const SHORT_MARGIN_FACTOR: f32 = 2.0;

/// The hop margin a clusterer at the shipped [`T_MARGIN`] holds a 0.75 s
/// embedding to. Derived, so the two cannot drift apart.
#[doc(hidden)]
pub const T_MARGIN_SHORT: f32 = SHORT_MARGIN_FACTOR * T_MARGIN;

/// Live centroids from which the cohort test switches on.
///
/// Below four there is no field to stand out from: a spread estimated from
/// one or two other scores says nothing, and normalising against it would be
/// arithmetic pretending to be evidence. So at two and three speakers the
/// cohort test is provably inert -- which is *not* the same as two- and
/// three-speaker behaviour being what the spike measured, because [`T_MARGIN`]
/// applies from two centroids upwards. Only a single speaker, or a
/// deployment at `diarize_margin = 0`, gets the pre-2026-08-27 rule exactly.
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
/// normally operate in, and cautious is affordable here for one reason: a
/// window this gate withholds is still reported as a guess and still filed
/// with a candidate speaker, so tightening it costs confidence rather than
/// coverage. The job left over for it is the case the margin misses: a winner
/// that beats the runner-up comfortably while the field behind it is broad
/// enough that "comfortably" means very little.
///
/// The spread it divides by is a *sample* standard deviation. At the cohort
/// minimum the field has three members, where the population formula
/// understates the spread by about 18% and would make "two standard
/// deviations" mean something looser than it reads.
///
/// Switched off, along with [`T_MARGIN`], at `diarize_margin = 0`: the cohort
/// test has no config key of its own, and a deployment reaching for the
/// documented way back to the pre-2026-08-27 rule has to actually get there.
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
///
/// **0 means off**, and is checked for rather than reached by arithmetic.
/// Different-speaker cosines are routinely negative -- the spike's median is
/// 0.046 and the distribution runs well below zero -- so `cosine < 0` fires
/// often, and no value of a plain threshold could ever disable detection.
/// `window::WindowAssembler` imposes a floor on how *often* an accepted
/// boundary may arrive; this is the only switch that turns them off entirely.
#[doc(hidden)]
pub const T_CHANGE: f32 = 0.30;

/// How much more like each other than like any incumbent a pool of orphans
/// must be before it becomes a speaker of its own.
///
/// The mint gate asks the question the room is actually posing. Ambiguity
/// between two incumbents is evidence of a bad or mixed embedding, not
/// evidence of a new person, so "the mean was not assignable" is no reason to
/// mint from it. What is a reason is the group being tighter with itself than
/// any of it is with somebody already known: `min pairwise agreement - cosine
/// to the nearest live centroid >= T_MINT_MARGIN`.
///
/// **Unmeasured.** 0.20 is sized against the spike's separability figures and
/// against what it must refuse: four windows that each sit just under
/// `T_ASSIGN` of an incumbent, whose private noise cancels in the mean, agree
/// with each other by about 0.18 more than their mean resembles that
/// incumbent -- and they are that incumbent, not a new person. It is
/// deliberately above that. On real audio, where a same-speaker window pair
/// sits at a median cosine of 0.517 and the agreeing-pool rule already floors
/// the group at `T_ASSIGN`, this gate rarely decides anything on its own; the
/// ceiling below is what usually binds. Both are guesses.
#[doc(hidden)]
pub const T_MINT_MARGIN: f32 = 0.20;

/// How many candidate speakers may be waiting for their fourth agreeing
/// window at once.
///
/// Orphans are held per candidate rather than in one flat ring, and that is
/// the whole of the crowding fix on this side. A single ring of eight held a
/// mintable set only if at most one other speaker's orphan landed between
/// each of a newcomer's windows: `4 + 3k <= 8` gives `k <= 1`, and a room of
/// eight has seven other people in it. Foreign orphans now land in their own
/// pool instead of evicting a newcomer's evidence.
///
/// **Unmeasured.** 8 is one candidate per person in the room this design
/// exists for. When a ninth would open, the pool that has waited longest for
/// a new member gives way -- the newest evidence is the evidence about
/// whoever is talking now.
#[doc(hidden)]
pub const MAX_POOLS: usize = 8;

/// Windows a candidate pool may go without gaining a member before it is
/// forgotten.
///
/// The bound that makes "not necessarily consecutive" stop short of "not
/// necessarily this hour". Four windows spread across a whole meeting are not
/// evidence of one voice; four spread across a minute of a busy room are.
///
/// **Unmeasured.** 64 windows is roughly 47 s of voiced audio anywhere in the
/// room, which is longer than the 30 s `session::RELABEL_WINDOW` a mint's
/// correction could reach back over and short enough that a pool cannot
/// survive a change of subject.
#[doc(hidden)]
pub const POOL_AGE: u64 = 64;

/// One speaker's running identity.
struct Centroid {
    label: u32,
    /// Unit-length, always: every update re-normalises.
    vector: Vec<f32>,
    /// A retired centroid keeps its slot (labels are never reused) but
    /// forwards to the older centroid it turned out to duplicate.
    retired_into: Option<u32>,
}

/// Orphan windows that agree with each other and with nobody who already has
/// a number: one candidate speaker, waiting for its fourth window.
///
/// Every member agrees with every other at `T_ASSIGN`, by construction rather
/// than by search -- a window joins a pool only if it agrees with all of it.
/// That is what removes the subset search the flat ring needed, and with it
/// the gap between the largest agreeing set and the one a greedy scan finds.
struct Candidate {
    /// Members, unit-length, in arrival order.
    members: Vec<Vec<f32>>,
    /// [`OnlineClusterer::windows_seen`] when this pool last gained a member.
    touched: u64,
}

/// What one embedding told the clusterer.
///
/// Three answers rather than two, because "nobody stood out clearly enough to
/// write down" and "nobody at all" are different facts about the audio and the
/// session does different things with them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heard {
    /// A speaker clear enough to record: this window moved that centroid, and
    /// may have minted it.
    Settled(u32),
    /// The nearest speaker over `T_ASSIGN`, named without enough of a lead to
    /// be sure of. **No centroid moved**, which is the half of withholding
    /// worth keeping; the window went into a candidate pool, so if it turns
    /// out to be somebody new the mint's correction will reach the text.
    Guessed(u32),
    /// Silence, cross-talk, or a voice with no number yet.
    Unknown,
}

impl Heard {
    /// The label to show a reader, guess or not.
    pub fn label(self) -> Option<u32> {
        match self {
            Heard::Settled(l) | Heard::Guessed(l) => Some(l),
            Heard::Unknown => None,
        }
    }

    /// The label only where the clusterer would stand behind it.
    pub fn settled(self) -> Option<u32> {
        match self {
            Heard::Settled(l) => Some(l),
            Heard::Guessed(_) | Heard::Unknown => None,
        }
    }

    /// Whether [`Heard::label`] is a guess a later window may contradict.
    pub fn is_guess(self) -> bool {
        matches!(self, Heard::Guessed(_))
    }
}

/// Assigns speaker labels to a stream of embeddings, in order.
///
/// Dimensionality is whatever the first embedding brings (192 for the
/// production ERes2Net model); it must not vary within one clusterer's life.
/// Inputs need not be normalised -- [`OnlineClusterer::observe`] normalises
/// what it is given -- but they must be non-zero to carry any direction.
pub struct OnlineClusterer {
    centroids: Vec<Centroid>,
    /// Candidate speakers, one pool of mutually agreeing orphans each.
    ///
    /// Per candidate rather than one shared ring: in a room of eight, six
    /// other people's orphans arrive between a newcomer's windows, and a
    /// shared ring spends its capacity evicting the evidence it exists to
    /// hold.
    pools: Vec<Candidate>,
    /// Embeddings [`OnlineClusterer::observe`] has seen. The clock pools are
    /// aged against -- the clusterer has no other notion of time, and windows
    /// are the unit its reluctance is written in.
    windows_seen: u64,
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
            pools: Vec::new(),
            windows_seen: 0,
            next_label: 1,
            t_assign,
            t_retire,
            alpha,
            min_pool,
            margin,
        }
    }

    /// Who this embedding belongs to, and how sure the clusterer is.
    ///
    /// Three outcomes, and the middle one is the point. A window clear enough
    /// to be sure of joins its centroid and moves it. A window that names
    /// nobody at all is honest uncertainty -- silence, cross-talk, a voice
    /// with no number yet. In between sits the ambiguous window, and it does
    /// three separable things: it moves **no** centroid, which is the drift
    /// protection; it reports the argmax as a [`Heard::Guessed`] label, so the
    /// text is not left blank for a decision the clusterer has merely declined
    /// to write down; and it joins the candidate pool it agrees with, so that
    /// if it was a new voice all along, the mint corrects the text it was
    /// guessed onto.
    pub fn observe(&mut self, embedding: &[f32]) -> Heard {
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
        self.windows_seen += 1;
        self.expire_pools();

        if let Some(index) = self.admits(&embedding, self.margin) {
            let alpha = self.alpha;
            let centroid = &mut self.centroids[index];
            for (x, e) in centroid.vector.iter_mut().zip(&embedding) {
                *x = (1.0 - alpha) * *x + alpha * e;
            }
            centroid.vector = l2_normalize(&centroid.vector);
            let label = centroid.label;

            // The pools are *not* cleared. An assignment says who was talking
            // just now, which is no evidence at all about the orphans waiting
            // to become somebody else.
            //
            // That nudge may have walked this centroid into another one.
            self.retire_converged();
            return Heard::Settled(self.resolve(label));
        }

        // Nobody stood out. The argmax is still the best guess there is, and
        // reporting it costs nothing that was not already spent: no centroid
        // moves either way, and a guess the next full window contradicts is
        // what `transcript.relabel` exists to take back.
        let guess = self
            .nearest(&embedding)
            .filter(|(_, similarity)| similarity.is_finite() && *similarity >= self.t_assign)
            .map(|(index, _)| self.resolve(self.centroids[index].label));

        match (self.file(embedding), guess) {
            (Some(minted), _) => Heard::Settled(minted),
            (None, Some(label)) => Heard::Guessed(label),
            (None, None) => Heard::Unknown,
        }
    }

    /// The label a short hop embedding *probably* belongs to.
    ///
    /// Takes `&self`, and that is the contract rather than an accident: a
    /// 0.75 s embedding never updates a centroid, never enters a pool, and
    /// never mints. It exists to name a turn about a second earlier than a
    /// full window can, and the price of that speed is that it is noisier --
    /// so it is allowed to *read* the clusterer's state and never to write it.
    /// Keeping centroid quality on the 1.5 s windows is what preserves
    /// everything the spike measured.
    ///
    /// Held to [`OnlineClusterer::short_margin`] rather than the window
    /// margin, because a guess made on half the evidence should have to be
    /// twice as clear.
    pub fn observe_short(&self, embedding: &[f32]) -> Option<u32> {
        let embedding = l2_normalize(embedding);
        self.admits(&embedding, self.short_margin())
            .map(|index| self.resolve(self.centroids[index].label))
    }

    /// The lead a 0.75 s hop must show, derived from the configured window
    /// margin by [`SHORT_MARGIN_FACTOR`] so that the two move together.
    fn short_margin(&self) -> f32 {
        SHORT_MARGIN_FACTOR * self.margin
    }

    /// File an orphan with the candidate speaker it agrees with, and mint that
    /// candidate if this was its last missing window.
    ///
    /// Returns the new label, or `None` while the candidate is still short of
    /// evidence or has just been shown to be somebody who already has a
    /// number.
    fn file(&mut self, embedding: Vec<f32>) -> Option<u32> {
        let index = match self.agreeing_pool(&embedding) {
            Some(index) => index,
            None => self.open_pool(),
        };
        self.pools[index].members.push(embedding);
        self.pools[index].touched = self.windows_seen;
        if self.pools[index].members.len() < self.min_pool {
            return None;
        }
        // Mint or not, this pool has had its answer: it is either a speaker
        // now or it is an incumbent's audio, and neither leaves evidence
        // behind for a later window to join.
        let candidate = self.pools.remove(index);
        self.mint(&candidate.members)
    }

    /// The pool `embedding` agrees with best, or `None` when it agrees with
    /// none of them.
    ///
    /// Agreement is with *every* member, at `T_ASSIGN`, which is what keeps a
    /// pool a single voice; "best" is the pool whose worst member still agrees
    /// most, so a window that could join two candidates joins the one it is
    /// least at the edge of.
    fn agreeing_pool(&self, embedding: &[f32]) -> Option<usize> {
        self.pools
            .iter()
            .enumerate()
            .filter_map(|(index, pool)| {
                let worst = pool
                    .members
                    .iter()
                    .map(|member| cosine(embedding, member))
                    .fold(f32::INFINITY, f32::min);
                (worst >= self.t_assign).then_some((index, worst))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
    }

    /// Start a candidate pool and return its index, evicting the stalest one
    /// if there is no room. The newest evidence is the evidence about whoever
    /// is talking now, so the pool that has waited longest is the one to lose.
    fn open_pool(&mut self) -> usize {
        if self.pools.len() >= MAX_POOLS
            && let Some(stalest) = self
                .pools
                .iter()
                .enumerate()
                .min_by_key(|(_, pool)| pool.touched)
                .map(|(index, _)| index)
        {
            self.pools.remove(stalest);
        }
        self.pools.push(Candidate {
            members: Vec::new(),
            touched: self.windows_seen,
        });
        self.pools.len() - 1
    }

    /// Forget candidates that have gone [`POOL_AGE`] windows without growing.
    fn expire_pools(&mut self) {
        let floor = self.windows_seen.saturating_sub(POOL_AGE);
        self.pools.retain(|pool| pool.touched >= floor);
    }

    /// Turn a full candidate pool into a speaker, if it has earned one.
    ///
    /// Two conditions, and the second is the whole of the 2026-08-27 crowding
    /// rethink. The first is the original rule and the only one at
    /// `diarize_margin = 0`: nobody who already has a number is within
    /// `T_ASSIGN` of the pool's mean, so minting cannot split an existing
    /// speaker. The second lets a pool mint *slightly* into that band -- as
    /// far above `T_ASSIGN` as the configured margin -- but only when the pool
    /// is more like itself than it is like that incumbent, by
    /// [`T_MINT_MARGIN`].
    ///
    /// What it deliberately does not do is treat ambiguity as novelty. A mean
    /// the assignment rule merely declined to place is evidence of a bad or
    /// mixed embedding; minting from it manufactures a speaker who does not
    /// exist, carries text away under their name, and -- once retirement
    /// folds the duplicate back -- leaves a transcript naming somebody the
    /// session no longer has.
    ///
    /// The old `T_RETIRE` re-test is gone because it is subsumed: a mean
    /// within 0.80 of a live centroid is far past `T_ASSIGN + margin`.
    fn mint(&mut self, members: &[Vec<f32>]) -> Option<u32> {
        let mean = mean_of(members);
        let rival = self.nearest(&mean).map(|(_, similarity)| similarity);
        if !self.may_mint(coherence(members), rival) {
            return None;
        }
        let label = self.next_label;
        self.next_label += 1;
        self.centroids.push(Centroid {
            label,
            vector: mean,
            retired_into: None,
        });
        Some(label)
    }

    /// Whether a pool with this internal agreement, whose mean sits at `rival`
    /// from the nearest live centroid, may become a speaker.
    fn may_mint(&self, coherence: f32, rival: Option<f32>) -> bool {
        match rival {
            // Nobody to split.
            None => true,
            Some(rival) if rival.is_nan() => false,
            // The pre-2026-08-27 rule, and the only clause that survives at
            // `diarize_margin = 0` -- where the arm below cannot fire, since
            // it needs `rival < t_assign + 0` against a `rival >= t_assign`.
            Some(rival) if rival < self.t_assign => true,
            Some(rival) => {
                rival < self.t_assign + self.margin && coherence - rival >= T_MINT_MARGIN
            }
        }
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
///    a field from, `(s1 - mean(rest)) / std(rest) >= T_ZNORM`, with `std` a
///    sample standard deviation. Adaptive score normalisation: it asks whether
///    this centroid stands out from the field rather than whether it clears a
///    number chosen when the field was smaller. Below four centroids there is
///    no field, so it is inert -- but gate 2 is not, so two- and
///    three-speaker behaviour is *not* what the spike measured unless the
///    margin is 0.
///
/// A margin of 0 switches gates 2 and 3 both off, leaving gate 1 alone: the
/// pre-2026-08-27 rule, exactly, which is what `diarize_margin = 0` is
/// documented to buy and what a deployment reaching for it needs.
///
/// Every gate is a conjunction with the first, so this function can only ever
/// return `None` where the original bare argmax returned `Some`. It cannot
/// invent an assignment, and therefore cannot introduce a merge. What a
/// refusal costs is bounded by the caller: [`OnlineClusterer::observe`] still
/// reports the argmax as a guess.
///
/// **Non-finite scores are refused, never passed through.** A NaN loses every
/// `<` it appears in, so a rule written as "return None when the lead is too
/// small" reads a NaN as "clear enough" and assigns to it; one assignment is
/// all it takes, because the EMA then makes that centroid NaN and every later
/// window is nearest to it. `embed.rs` rejects a non-finite embedding before
/// it reaches here, so this is defence in depth rather than a live path.
fn stands_out(scores: &[f32], t_assign: f32, margin: f32) -> Option<usize> {
    let (best, &s1) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if !s1.is_finite() || s1 < t_assign {
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
    let lead = s1 - s2;
    if lead.is_nan() || lead < margin {
        return None;
    }

    if margin <= 0.0 || scores.len() < COHORT_MIN {
        return Some(best);
    }
    let rest: Vec<f32> = scores
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best)
        .map(|(_, s)| *s)
        .collect();
    let mean = rest.iter().sum::<f32>() / rest.len() as f32;
    // Sample rather than population: `rest` is three scores at the cohort
    // minimum, where dividing by n instead of n-1 understates the spread by
    // about 18% and quietly makes "two standard deviations" a looser bar than
    // it reads.
    let degrees = rest.len().saturating_sub(1).max(1) as f32;
    let variance = rest.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / degrees;
    // A floor rather than a special case: a field with no spread at all makes
    // any lead infinitely significant, which is the right answer and also the
    // one the division would give if it did not divide by zero first.
    let z = (s1 - mean) / variance.sqrt().max(1e-6);
    if z.is_nan() || z < T_ZNORM {
        return None;
    }
    Some(best)
}

/// The unit-length mean of a set of vectors, all of one width.
fn mean_of(members: &[Vec<f32>]) -> Vec<f32> {
    let mut mean = vec![0.0f32; members.first().map_or(0, Vec::len)];
    for member in members {
        for (m, x) in mean.iter_mut().zip(member) {
            *m += x;
        }
    }
    l2_normalize(&mean)
}

/// The worst agreement between any two members: how much the set is like
/// itself. `INFINITY` for a set with no pair in it, which no caller has.
fn coherence(members: &[Vec<f32>]) -> f32 {
    members
        .iter()
        .enumerate()
        .flat_map(|(i, a)| members[i + 1..].iter().map(move |b| cosine(a, b)))
        .fold(f32::INFINITY, f32::min)
}

/// A unit-length copy.
///
/// The 1e-9 floor keeps an all-zero input -- a silent window -- finite rather
/// than NaN. It is not a guard against a non-finite *input*: a NaN anywhere in
/// `v` normalises to a vector of NaNs, and every rule downstream has to refuse
/// on its own account. [`stands_out`] does.
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

    /// A unit vector `degrees` round from axis 0, in the plane of axes 0 and 1.
    ///
    /// Separation as an angle rather than a cosine, because the retirement
    /// fixture below has to reason about two centroids walking towards each
    /// other and angles add where cosines do not.
    fn at(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        embedding(&[(0, radians.cos()), (1, radians.sin())])
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
    // Everything below the two-speaker line runs on two synthetic speakers,
    // which is how the eight-speaker failure stayed a prediction in a design
    // document for as long as it did. These fixtures build rooms instead.

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

    /// Run a script and report which label was *settled* on for each voice's
    /// own windows most often, plus the clusterer's live labels. `None` for a
    /// voice that was never confidently labelled at all.
    ///
    /// Guesses are excluded on purpose: a [`Heard::Guessed`] label is the
    /// clusterer saying it does not know, and counting it here would let a
    /// room score perfectly on a pile of shrugs.
    fn who_got_what(
        mut c: OnlineClusterer,
        voices: usize,
        script: &[(usize, Vec<f32>)],
    ) -> (Vec<Option<u32>>, Vec<u32>) {
        let mut counts: Vec<Vec<(u32, usize)>> = vec![Vec::new(); voices];
        for (v, window) in script {
            if let Some(label) = c.observe(window).settled() {
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
        /// `T_ASSIGN`, which is the whole point, and below the mint gate's
        /// ceiling of `T_ASSIGN + T_MARGIN`, which is what still lets a
        /// crowded newcomer become somebody.
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
        const {
            assert!(
                CROWDED < T_ASSIGN + T_MARGIN,
                "a newcomer this crowded has to stay inside the mint gate's ceiling"
            )
        };
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
    fn a_crowded_newcomer_is_guessed_at_rather_than_handed_to_an_incumbent() {
        // The same room, watched one decision at a time, because "everybody
        // was minted" does not say *why*. A newcomer's first window has three
        // incumbents over `T_ASSIGN` and no reason to prefer any of them.
        //
        // What that costs is bounded on purpose. The window names the argmax,
        // because a reader is better served by the best guess going than by a
        // blank -- but it names it as a guess, it moves no centroid, and it
        // goes into a candidate pool. Three windows later the pool mints and
        // the guesses are wrong text a correction can reach.
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
        // decimal place -- and, being an assignment, it moved that founder's
        // centroid and emptied the pool.
        let opening = window(4);
        let (best, similarity) = c.nearest(&opening).expect("four live centroids");
        assert!(
            similarity >= T_ASSIGN,
            "founder {best} sits at {similarity}, so the old rule would have \
             taken it"
        );
        let before = c.vectors();
        let heard = c.observe(&opening);
        assert!(
            heard.is_guess(),
            "the new rule does not settle it: {heard:?}"
        );
        assert!(
            matches!(heard.label(), Some(1..=4)),
            "and it still says who it would have been: {heard:?}"
        );
        assert_eq!(c.vectors(), before, "a guess moved a centroid");

        // Three more, and the newcomer is somebody.
        assert!(c.observe(&window(4)).is_guess());
        assert!(c.observe(&window(4)).is_guess());
        assert_eq!(c.observe(&window(4)), Heard::Settled(5));
        // Their own windows settle from then on: standing out from a crowded
        // field is exactly what a minted centroid does.
        assert_eq!(c.observe(&window(4)), Heard::Settled(5));
        // And the founders keep their numbers.
        for v in 0..4 {
            assert_eq!(c.observe(&window(v)), Heard::Settled(v as u32 + 1));
        }
    }

    /// How many other people may talk between a newcomer's windows before the
    /// newcomer stops being mintable. The answer has to be "all of them".
    ///
    /// The flat ring this replaced held eight orphans, so `MIN_POOL` windows
    /// with `k` foreign orphans between each needed `4 + 3k <= 8`: one
    /// interrupting speaker. A room of eight has seven, and the crowding
    /// gates turn other people's *assignments* into orphans, so `k` is large
    /// in exactly the room this is for.
    #[test]
    fn a_newcomer_is_minted_however_many_other_orphans_arrive_between_windows() {
        for others in 1..=MAX_POOLS - 1 {
            let voices = separated_room(4 + 1 + others);
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

            // The newcomer is voice 4; voices 5.. are strangers who each get
            // one window in between and so never accumulate a pool of their
            // own before the newcomer does.
            let mut heard = Vec::new();
            for _ in 0..MIN_POOL {
                heard.push(c.observe(&window(4)));
                for stranger in 0..others {
                    c.observe(&window(5 + stranger));
                }
            }
            assert_eq!(
                heard.last(),
                Some(&Heard::Settled(5)),
                "with {others} other orphans between windows the newcomer \
                 was never minted: {heard:?}"
            );
        }
    }

    #[test]
    fn the_number_of_candidate_pools_is_bounded() {
        // A meeting is hours long and most of what the diarizer hears is
        // nobody in particular. Per-candidate pools are what stopped other
        // people's orphans evicting a newcomer's evidence, so the bound that
        // keeps them finite has to be on the number of candidates rather than
        // on the evidence inside one.
        let wide = |i: usize| {
            let mut v = vec![0.0f32; MAX_POOLS * 2];
            v[i] = 1.0;
            v
        };
        let mut c = OnlineClusterer::new();
        for i in 0..MAX_POOLS * 2 {
            assert_eq!(c.observe(&wide(i)), Heard::Unknown);
        }
        assert_eq!(c.pools.len(), MAX_POOLS, "the pools grew past their bound");
        // And nothing was minted along the way: mutually disagreeing windows
        // are evidence of nobody.
        assert_eq!(c.minted(), 0);
    }

    #[test]
    fn a_candidate_pool_that_stops_growing_is_forgotten() {
        // "Not necessarily consecutive" has to stop short of "not necessarily
        // this hour": four windows spread across a whole meeting are not
        // evidence of one voice.
        let mut stale = OnlineClusterer::new();
        for _ in 0..MIN_POOL - 1 {
            stale.observe(&voice(0));
        }
        // Somebody else holds the floor for longer than a pool may wait.
        for _ in 0..POOL_AGE + 1 {
            stale.observe(&voice(1));
        }
        assert_eq!(stale.minted(), 1, "the other voice is speaker 1");
        assert_eq!(
            stale.observe(&voice(0)),
            Heard::Unknown,
            "a window from before the age bound completed a forgotten pool"
        );

        // The same script inside the bound, so the difference is the waiting
        // and not the interruption.
        let mut fresh = OnlineClusterer::new();
        for _ in 0..MIN_POOL - 1 {
            fresh.observe(&voice(0));
        }
        for _ in 0..MIN_POOL * 2 {
            fresh.observe(&voice(1));
        }
        assert_eq!(fresh.observe(&voice(0)), Heard::Settled(2));
    }

    // -------------------------------------------------------- two speakers

    #[test]
    fn two_separated_voices_take_labels_in_first_appearance_order() {
        // The synthetic geometry the rest of these tests assume.
        assert!(similarity(&noisy(0, 1), &noisy(0, 2)) >= T_ASSIGN);
        assert!(similarity(&noisy(0, 1), &noisy(1, 1)) < T_ASSIGN);

        let mut c = OnlineClusterer::new();
        let first: Vec<Heard> = (0..MIN_POOL as u32)
            .map(|i| c.observe(&noisy(0, i)))
            .collect();
        let second: Vec<Heard> = (0..MIN_POOL as u32)
            .map(|i| c.observe(&noisy(1, i)))
            .collect();

        assert_eq!(first.last(), Some(&Heard::Settled(1)));
        assert_eq!(second.last(), Some(&Heard::Settled(2)));
        assert!(first[..MIN_POOL - 1].iter().all(|h| *h == Heard::Unknown));
        assert!(second[..MIN_POOL - 1].iter().all(|h| *h == Heard::Unknown));
        // The first speaker keeps label 1 once the second exists.
        assert_eq!(c.observe(&noisy(0, 99)), Heard::Settled(1));
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
        let labels: Vec<Heard> = (0..100).map(|i| c.observe(&noisy(0, i))).collect();

        assert!(labels[..MIN_POOL - 1].iter().all(|h| *h == Heard::Unknown));
        assert!(
            labels[MIN_POOL - 1..]
                .iter()
                .all(|l| *l == Heard::Settled(1)),
            "one voice produced more than one label: {labels:?}"
        );
    }

    #[test]
    fn one_outlier_mints_nothing_and_an_agreeing_run_mints_once() {
        let mut c = OnlineClusterer::new();
        // A lone window from a voice never heard again.
        assert_eq!(c.observe(&noisy(0, 0)), Heard::Unknown);

        // A different voice fills a pool of its own -- the outlier agrees with
        // none of it, so it is in nobody's way -- and the run mints on its
        // fourth window, taking label 1 and proving the outlier minted
        // nothing.
        let labels: Vec<Heard> = (0..=MIN_POOL as u32)
            .map(|i| c.observe(&noisy(1, i)))
            .collect();
        assert_eq!(
            labels,
            vec![
                Heard::Unknown,
                Heard::Unknown,
                Heard::Unknown,
                Heard::Settled(1),
                Heard::Settled(1)
            ]
        );
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
            assert_eq!(c.observe(&noisy(1, i)), Heard::Unknown);
        }
        // ...then the first speaker takes a turn, which says nothing at all
        // about whether the three orphans are somebody.
        assert_eq!(c.observe(&noisy(0, 50)), Heard::Settled(1));

        // So the fourth orphan still completes the set, interruption and all.
        assert_eq!(c.observe(&noisy(1, 9)), Heard::Settled(2));
    }

    #[test]
    fn orphans_survive_several_interruptions_and_a_poisoned_window() {
        // The shape a real meeting has: a new voice gets a window in, somebody
        // else answers, they get another in. Plus one stray window that
        // belongs to nobody, which used to cost a whole pool and then one
        // window per attempt to recover from -- now it is simply in a pool of
        // its own.
        let mut c = OnlineClusterer::new();
        for i in 0..MIN_POOL as u32 {
            c.observe(&noisy(0, i));
        }

        assert_eq!(c.observe(&noisy(1, 0)), Heard::Unknown);
        assert_eq!(
            c.observe(&noisy(0, 20)),
            Heard::Settled(1),
            "the floor changes hands"
        );
        assert_eq!(
            c.observe(&noisy(2, 77)),
            Heard::Unknown,
            "a stray window from nobody"
        );
        assert_eq!(c.observe(&noisy(1, 1)), Heard::Unknown);
        assert_eq!(c.observe(&noisy(0, 21)), Heard::Settled(1));
        assert_eq!(c.observe(&noisy(1, 2)), Heard::Unknown);
        assert_eq!(c.observe(&noisy(0, 22)), Heard::Settled(1));
        assert_eq!(
            c.observe(&noisy(1, 3)),
            Heard::Settled(2),
            "four agreeing windows are four agreeing windows"
        );
        // And the stray is still nobody: it was never in the pool that
        // minted, and it has minted nothing of its own.
        assert_eq!(c.live_labels(), vec![1, 2]);
    }

    #[test]
    fn one_borderline_window_does_not_unseat_a_settled_centroid() {
        let mut c = OnlineClusterer::new();
        for i in 0..100 {
            c.observe(&noisy(0, i));
        }

        // Well inside T_ASSIGN, but pulling hard towards another axis. One
        // live centroid, so there is nothing for it to be ambiguous against.
        let borderline = embedding(&[(0, 0.6), (1, 0.8)]);
        assert_eq!(c.observe(&borderline), Heard::Settled(1));

        // A centroid is its history: that window moved it by EMA_ALPHA, which
        // is nowhere near enough to drag it into the borderline window's own
        // territory. Probed first, because the next speaker-0 window would
        // pull it back and hide a centroid that had moved all the way.
        assert_eq!(c.observe(&voice(1)), Heard::Unknown);
        // And it still recognises the speaker it belongs to.
        assert_eq!(c.observe(&voice(0)), Heard::Settled(1));
        assert_eq!(c.observe(&noisy(0, 200)), Heard::Settled(1));
    }

    // ------------------------------------------------------------ retirement

    /// How far apart the two speakers in [`converging_pair`] start: cosine
    /// 0.40, below `T_ASSIGN`, so they mint separately.
    const APART: f32 = 66.4;

    /// Two speakers minted at the **shipped** configuration, and the pair of
    /// windows that walks their centroids into each other.
    ///
    /// The fixture this replaced switched the margin off, which meant every
    /// retirement test ran at a configuration the server does not ship --
    /// precisely when the mint path can now reach retirement at the shipped
    /// one. So the drift here is caused by windows that are *not* ambiguous:
    /// each leads the runner-up by well over `T_MARGIN` from the first
    /// observation to the last, and each sits on the far side of its own
    /// centroid, so assigning it walks that centroid towards the other. They
    /// would meet 25 degrees apart; retirement fires first, at the 36.9
    /// degrees that is `T_RETIRE`.
    ///
    /// Ambiguity is the other fixture's business --
    /// `an_ambiguous_window_moves_no_centroid_at_all` -- and the two are
    /// deliberately separate now: one is about drift the margin refuses, this
    /// one is about cleaning up drift the margin permits.
    fn converging_pair() -> (OnlineClusterer, Vec<f32>, [Vec<f32>; 2]) {
        let second = at(APART);
        let drags = [at(20.0), at(45.0)];
        assert!(
            similarity(&voice(0), &second) < T_ASSIGN,
            "far enough apart"
        );
        for (drag, (mine, theirs)) in drags
            .iter()
            .zip([(&voice(0), &second), (&second, &voice(0))])
        {
            let lead = similarity(drag, mine) - similarity(drag, theirs);
            assert!(
                lead >= T_MARGIN,
                "a drag has to be unambiguous at the shipped margin, and this \
                 one leads by {lead}"
            );
        }

        let mut c = OnlineClusterer::new();
        for _ in 0..MIN_POOL {
            c.observe(&voice(0));
        }
        for _ in 0..MIN_POOL {
            c.observe(&second);
        }
        assert_eq!(c.live_labels(), vec![1, 2]);
        (c, second, drags)
    }

    /// Push the two drags alternately until the pair retires, and stop there.
    ///
    /// Stopping matters: once one centroid is left it takes both drags, so
    /// driving on would walk the survivor somewhere neither speaker ever was
    /// and make every later probe a question about the fixture instead of
    /// about retirement.
    fn drive_until_retired(c: &mut OnlineClusterer, drags: &[Vec<f32>; 2]) {
        for i in 0..200 {
            if c.live_labels().len() == 1 {
                return;
            }
            c.observe(&drags[i % 2]);
        }
        panic!("the centroids never converged");
    }

    #[test]
    fn converged_centroids_retire_the_newer_into_the_older() {
        let (mut c, second, drags) = converging_pair();

        let mut labels = Vec::new();
        let mut retired_at = None;
        for i in 0..200 {
            let live = c.live_labels();
            labels.push(c.observe(&drags[i % 2]));
            if c.live_labels() != live {
                assert!(retired_at.is_none(), "retired more than once");
                // The older label survives; the newer is the one that goes.
                assert_eq!(c.live_labels(), vec![1]);
                retired_at = Some(i);
            }
        }

        let retired_at = retired_at.expect("the centroids never converged");
        // Before the fold each drag answered with its own speaker, which is
        // what makes this drift rather than confusion.
        assert!(
            labels[..retired_at]
                .iter()
                .enumerate()
                .all(|(i, l)| *l == Heard::Settled(if i % 2 == 0 { 1 } else { 2 })),
            "a drag stopped being unambiguous before the fold: {labels:?}"
        );
        // A caller never sees a label that died inside its own call, and no
        // third speaker is invented anywhere along the way.
        assert!(labels[retired_at..].iter().all(|l| *l == Heard::Settled(1)));
        // And speaker 2's own voice now answers to speaker 1's label.
        assert_eq!(c.observe(&second), Heard::Settled(1));
    }

    #[test]
    fn a_retired_centroid_no_longer_attracts_windows() {
        let (mut c, _, drags) = converging_pair();
        drive_until_retired(&mut c, &drags);

        // Speaker 2's retired vector is frozen part-way round towards its own
        // drag, while speaker 1's has walked only as far as its own. This
        // probe is beyond both, close enough to the retired one to join it and
        // too far from the live one -- and a retired centroid is a forwarding
        // address, not a speaker, so it must attract nothing.
        let probe = at(90.0);
        assert!(similarity(&probe, &at(0.0)) < T_ASSIGN);
        assert!(similarity(&probe, &drags[1]) >= T_ASSIGN);
        assert_eq!(c.observe(&probe), Heard::Unknown);
    }

    #[test]
    fn labels_are_never_reused_after_retirement() {
        let (mut c, second, drags) = converging_pair();
        drive_until_retired(&mut c, &drags);
        assert_eq!(
            c.observe(&second),
            Heard::Settled(1),
            "label 2 should have retired"
        );

        // A genuinely new voice takes the next label, not the free one.
        let labels: Vec<Heard> = (0..MIN_POOL).map(|_| c.observe(&voice(7))).collect();
        assert_eq!(labels.last(), Some(&Heard::Settled(3)));
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
        let second = at(APART);
        let drags = [at(20.0), at(45.0)];

        // One script through all five constants: pooling and minting
        // (MIN_POOL), assignment (T_ASSIGN), a nudged centroid (EMA_ALPHA), a
        // run of ambiguous windows the margin refuses (T_MARGIN), the drift
        // the margin does permit and the fold that cleans it up (T_RETIRE),
        // and a fresh voice afterwards to show numbering carries on.
        let mut script: Vec<Vec<f32>> = (0..MIN_POOL as u32).map(|i| noisy(0, i)).collect();
        script.push(noisy(1, 50));
        script.push(noisy(0, 10));
        script.extend(std::iter::repeat_n(second, MIN_POOL));
        script.extend(std::iter::repeat_n(at(APART / 2.0), MIN_POOL * 2));
        script.extend((0..100).map(|i| drags[i % 2].clone()));
        script.extend(std::iter::repeat_n(voice(7), MIN_POOL));

        let run = |mut c: OnlineClusterer| {
            let labels: Vec<Heard> = script.iter().map(|e| c.observe(e)).collect();
            (labels, c.live_labels(), c.minted())
        };
        let shipped = run(OnlineClusterer::new());
        let swept = run(OnlineClusterer::with_params(
            T_ASSIGN, T_RETIRE, EMA_ALPHA, MIN_POOL, T_MARGIN,
        ));
        assert_eq!(shipped, swept);

        // And the script is worth comparing: an equivalence that held only
        // because nothing happened would pass this test while proving nothing.
        let (labels, live, minted) = shipped;
        assert_eq!(live, vec![1, 3], "speaker 2 drifted into speaker 1");
        assert_eq!(minted, 3, "and the ambiguous run minted nobody of its own");
        for expected in [
            Heard::Unknown,
            Heard::Settled(1),
            Heard::Settled(2),
            Heard::Settled(3),
            Heard::Guessed(1),
        ] {
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
    fn observe_all(mut c: OnlineClusterer, script: &[Vec<f32>]) -> (Vec<Heard>, Vec<u32>) {
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
        assert_eq!(
            shipped,
            vec![Heard::Unknown, Heard::Unknown],
            "MIN_POOL 4 wants four windows"
        );
        assert_eq!(
            swept,
            vec![Heard::Unknown, Heard::Settled(1)],
            "a pool of 2 should have minted"
        );
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
            vec![Heard::Unknown, Heard::Settled(1)],
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
        assert_eq!(
            shipped.last(),
            Some(&Heard::Settled(1)),
            "0.45 should have assigned"
        );
        assert_eq!(
            swept.last(),
            Some(&Heard::Unknown),
            "0.99 should have refused, and had nobody to guess at either"
        );
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
            Some(&Heard::Unknown),
            "alpha 0.05 should have left speaker 1 out of reach"
        );
        assert_eq!(
            swept.last(),
            Some(&Heard::Settled(1)),
            "alpha 1.0 should have moved the centroid onto the new window"
        );
    }

    #[test]
    fn a_lower_t_retire_folds_two_centroids_the_shipped_one_keeps() {
        // Two speakers 0.4 apart: far enough to mint separately (below
        // T_ASSIGN), close enough that a retire threshold of 0.35 calls them
        // duplicates while the shipped 0.80 does not. The last window assigns
        // rather than mints, which is what runs the convergence check.
        let second = at(APART);
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
        // Retired, not refused: the second speaker was minted either way, and
        // only the fold differs. `T_RETIRE` has no say in minting any more.
        assert_eq!(swept[MIN_POOL * 2 - 1], Heard::Settled(2));
        // Both still answer the final window with speaker 1 -- the rest of
        // the difference is only visible in what survives, which is why this
        // asserts on the live centroids and not on the labels alone.
        assert_eq!(shipped.last(), Some(&Heard::Settled(1)));
        assert_eq!(swept.last(), Some(&Heard::Settled(1)));
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
    fn a_non_finite_score_is_refused_rather_than_won_by() {
        // The polarity that matters. Every gate below the threshold is written
        // as "return None when this comparison fails", and a NaN fails every
        // comparison -- so a rule phrased as `if lead < margin { None }` reads
        // a NaN lead as a clear win and assigns to it. One such assignment is
        // terminal: the EMA makes that centroid NaN, `total_cmp` then ranks it
        // above every real one, and every later window joins it.
        assert_eq!(stands_out(&[f32::NAN, 0.9, 0.1], T_ASSIGN, T_MARGIN), None);
        assert_eq!(stands_out(&[0.9, f32::NAN, 0.1], T_ASSIGN, T_MARGIN), None);
        assert_eq!(stands_out(&[f32::NAN], T_ASSIGN, T_MARGIN), None);
        assert_eq!(
            stands_out(&[0.9, f32::NAN, 0.1, 0.0, 0.0], T_ASSIGN, T_MARGIN),
            None
        );
        assert_eq!(
            stands_out(&[f32::INFINITY, 0.1], T_ASSIGN, T_MARGIN),
            None,
            "an infinite cosine is not a cosine either"
        );
    }

    #[test]
    fn a_non_finite_embedding_never_reaches_a_centroid() {
        // The same property through the door a model would come in by.
        // `embed.rs` rejects a non-finite embedding before this, so this is
        // defence in depth -- but the failure it defends against is an
        // absorbing state, not a wrong sentence.
        let mut c = OnlineClusterer::new();
        for i in 0..MIN_POOL as u32 {
            c.observe(&noisy(0, i));
        }
        let before = c.vectors();

        for _ in 0..MIN_POOL * 4 {
            assert_eq!(c.observe(&[f32::NAN; DIM]), Heard::Unknown);
        }
        assert_eq!(c.vectors(), before, "a NaN window moved a centroid");
        assert_eq!(c.minted(), 1, "a NaN window minted a speaker");
        assert_eq!(c.observe(&noisy(0, 500)), Heard::Settled(1));
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
    fn a_margin_of_zero_is_the_pre_2026_08_27_rule_and_nothing_else() {
        // The documented escape hatch, and it has to be complete: the cohort
        // test has no config key of its own, so a deployment that turns the
        // margin off and still gets refused in a crowded room has been sold a
        // switch that does not switch. Checked exhaustively, because "the
        // z-norm is off too" is exactly the kind of claim that rots.
        let grid = [0.0f32, 0.2, 0.44, 0.45, 0.46, 0.5, 0.519, 0.55, 0.6, 0.9];
        for &a in &grid {
            for &b in &grid {
                for &c in &grid {
                    for &d in &grid {
                        let scores = [a, b, c, d];
                        for n in 1..=scores.len() {
                            let scores = &scores[..n];
                            let argmax = scores
                                .iter()
                                .enumerate()
                                .max_by(|x, y| x.1.total_cmp(y.1))
                                .expect("non-empty");
                            let bare = (*argmax.1 >= T_ASSIGN).then_some(argmax.0);
                            assert_eq!(stands_out(scores, T_ASSIGN, 0.0), bare, "{scores:?}");
                        }
                    }
                }
            }
        }
        // The pattern the review found: four centroids, a comfortable winner,
        // and a z-norm that refuses it anyway however low the margin is set.
        let crowded = [0.60, 0.55, 0.20, 0.05];
        assert_eq!(stands_out(&crowded, T_ASSIGN, T_MARGIN), None);
        assert_eq!(stands_out(&crowded, T_ASSIGN, 0.0), Some(0));
    }

    #[test]
    fn the_cohort_test_is_inert_below_four_live_centroids() {
        // Two and three speakers have no field to be normalised against, so
        // the z-norm has to be provably switched off there. It does *not*
        // follow that two- and three-speaker behaviour is what the spike
        // measured -- the margin applies from two centroids upwards, which is
        // what the line below pins.
        assert_eq!(COHORT_MIN, 4);
        let below = [0.65, 0.50, 0.30];
        assert!(below.len() < COHORT_MIN);
        assert_eq!(stands_out(&below, T_ASSIGN, T_MARGIN), Some(0));
        assert_eq!(
            stands_out(&[0.65, 0.60], T_ASSIGN, T_MARGIN),
            None,
            "the margin is not inert at two centroids, whatever the z-norm does"
        );

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
    fn the_cohort_spread_is_a_sample_standard_deviation() {
        // At the cohort minimum the field has three members, where dividing
        // by n rather than n-1 understates the spread by about 18% and makes
        // "two standard deviations" a materially looser bar than it reads.
        // This score pattern is on the wrong side of the line for one and the
        // right side for the other.
        let scores = [0.66, 0.50, 0.30, 0.10];
        let rest = [0.50, 0.30, 0.10];
        let mean = rest.iter().sum::<f32>() / 3.0;
        let spread = |degrees: f32| {
            (rest.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / degrees).sqrt()
        };
        assert!((scores[0] - mean) / spread(3.0) >= T_ZNORM, "population");
        assert!((scores[0] - mean) / spread(2.0) < T_ZNORM, "sample");
        assert_eq!(stands_out(&scores, T_ASSIGN, T_MARGIN), None);
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

    // ----------------------------------------------------- the mint gate
    //
    // What a pool of agreeing orphans has to show before it becomes a
    // speaker. The rule this replaced asked whether the pool's mean was
    // *assignable*, and minted whenever it was merely ambiguous -- so the
    // whole band from `T_ASSIGN` to `T_RETIRE` became mintable on the
    // strength of an incumbent being unclear, which is evidence of a bad
    // embedding and not evidence of a new person.

    /// Two speakers minted at the shipped configuration, [`APART`] degrees
    /// apart, and nothing else in the room.
    fn two_speakers() -> OnlineClusterer {
        let mut c = OnlineClusterer::new();
        for _ in 0..MIN_POOL {
            c.observe(&at(0.0));
        }
        for _ in 0..MIN_POOL {
            c.observe(&at(APART));
        }
        assert_eq!(c.live_labels(), vec![1, 2]);
        c
    }

    /// A voice sitting at exactly `similarity` from *both* of
    /// [`two_speakers`], with a private axis carrying whatever is left.
    ///
    /// Ambiguous by construction -- the lead is zero, so no margin however
    /// small assigns it -- which is the only way into the mint gate.
    fn crowded_by_both(similarity: f32) -> Vec<f32> {
        let bisector = at(APART / 2.0);
        let share = similarity / cosine(&bisector, &at(0.0));
        assert!(share <= 1.0, "no such voice exists at {similarity}");
        let mut v: Vec<f32> = bisector.iter().map(|x| x * share).collect();
        v[2] = (1.0 - share * share).sqrt();
        for other in [at(0.0), at(APART)] {
            let sim = cosine(&v, &other);
            assert!(
                (sim - similarity).abs() < 1e-5,
                "the fixture sits at {sim}, not {similarity}"
            );
        }
        v
    }

    #[test]
    fn a_pool_ambiguous_between_two_incumbents_mints_nobody() {
        // The measured failure: with two incumbents, the rule this replaced
        // minted a fresh label at 0.600, 0.650 and 0.705 where the old one
        // assigned -- and at 0.705, against a same-speaker median of 0.517,
        // that is overwhelmingly one of the two people already in the room.
        // Worse, such a mint retires back into its neighbour tens of windows
        // later, by which time commits carrying its number are frozen and the
        // transcript names somebody the session no longer has.
        for similarity in [0.60, 0.65, 0.705, 0.79] {
            let mut c = two_speakers();
            let voice = crowded_by_both(similarity);
            let heard: Vec<Heard> = (0..MIN_POOL * 3).map(|_| c.observe(&voice)).collect();
            assert!(
                heard.iter().all(|h| h.is_guess()),
                "at {similarity} the ambiguous run was not all guesses: {heard:?}"
            );
            assert_eq!(
                c.minted(),
                2,
                "at {similarity} an ambiguous pool minted a speaker"
            );
        }
    }

    #[test]
    fn a_pool_that_stands_out_from_a_crowd_still_mints() {
        // The other side of the gate, and the reason it is a ceiling rather
        // than a return to "any incumbent over T_ASSIGN blocks a mint": in a
        // crowded room that bare rule is true of every new voice, and a
        // clusterer that never mints again is the eight-speaker failure this
        // work exists to fix. A newcomer just inside the ceiling, whose
        // windows agree with each other far better than their mean resembles
        // either incumbent, is still somebody.
        let mut c = two_speakers();
        let voice = crowded_by_both(0.50);
        const CROWDED: f32 = 0.50;
        const { assert!(CROWDED < T_ASSIGN + T_MARGIN, "inside the ceiling") };
        let heard: Vec<Heard> = (0..MIN_POOL).map(|_| c.observe(&voice)).collect();
        assert!(heard[..MIN_POOL - 1].iter().all(|h| h.is_guess()));
        assert_eq!(heard.last(), Some(&Heard::Settled(3)));
    }

    #[test]
    fn a_pool_whose_mean_is_a_known_speaker_mints_nothing() {
        // Four windows that agree with each other, each individually just
        // under `T_ASSIGN` of speaker 1 -- a shared pull off-axis plus private
        // noise on an axis of its own. Averaging cancels the private part, so
        // their mean lands back inside speaker 1's territory, and the pool
        // agrees with itself by less than `T_MINT_MARGIN` more than that.
        //
        // Two live centroids, not one: with a single centroid the gate's
        // first clause is trivially satisfied and the case that is the whole
        // content of the rule never runs.
        let mut c = OnlineClusterer::new();
        for _ in 0..MIN_POOL {
            c.observe(&voice(0));
        }
        for _ in 0..MIN_POOL {
            c.observe(&voice(6));
        }
        assert_eq!(c.live_labels(), vec![1, 2]);

        let pooled: Vec<Vec<f32>> = (0..MIN_POOL)
            .map(|i| embedding(&[(0, 0.42), (1, 0.35_f32.sqrt()), (2 + i, 0.4736_f32.sqrt())]))
            .collect();
        let mean = mean_of(&pooled.iter().map(|v| l2_normalize(v)).collect::<Vec<_>>());
        for v in &pooled {
            assert!(similarity(v, &voice(0)) < T_ASSIGN, "individually nobody");
            assert!(similarity(v, &voice(6)) < T_ASSIGN);
            assert!(similarity(v, &pooled[0]) >= T_ASSIGN, "but agreeing");
        }
        let rival = cosine(&mean, &voice(0));
        assert!((T_ASSIGN..T_ASSIGN + T_MARGIN).contains(&rival), "{rival}");
        let coherence = similarity(&pooled[0], &pooled[1]);
        assert!(
            coherence - rival < T_MINT_MARGIN,
            "the pool agrees with itself by {} more than its mean resembles \
             speaker 1, which has to be under the gate for this to bite",
            coherence - rival
        );

        // So none of them mints: splitting a speaker who already has a label
        // is the one failure this design exists to avoid.
        for v in &pooled {
            assert_eq!(c.observe(v), Heard::Unknown);
        }
        assert_eq!(c.minted(), 2);
        // And the pool is gone, not lingering: the next new voice needs its
        // own full MIN_POOL, and takes label 3 because nothing else was ever
        // minted.
        let labels: Vec<Heard> = (0..MIN_POOL).map(|_| c.observe(&voice(7))).collect();
        assert_eq!(
            labels,
            vec![
                Heard::Unknown,
                Heard::Unknown,
                Heard::Unknown,
                Heard::Settled(3)
            ]
        );
    }

    #[test]
    fn a_margin_of_zero_mints_by_the_pre_2026_08_27_rule() {
        // The other half of the escape hatch. At margin 0 the ceiling is
        // `T_ASSIGN` exactly, so the only clause left is the original one --
        // nobody over the threshold near the mean -- and the crowded newcomer
        // the shipped configuration mints is refused, which is honest about
        // what the hatch costs.
        let mut c = OnlineClusterer::with_config(MIN_POOL, 0.0);
        for _ in 0..MIN_POOL {
            c.observe(&at(0.0));
        }
        for _ in 0..MIN_POOL {
            c.observe(&at(APART));
        }
        assert_eq!(c.live_labels(), vec![1, 2]);

        let voice = crowded_by_both(0.50);
        for _ in 0..MIN_POOL * 3 {
            // Not a guess either: at margin 0 the bare argmax assigns, which
            // is the behaviour being restored.
            assert!(c.observe(&voice).settled().is_some());
        }
        assert_eq!(c.minted(), 2, "the old rule refuses this mint");
    }

    #[test]
    fn an_ambiguous_window_moves_no_centroid_at_all() {
        // The drift regression, and the reason the margin exists at all. The
        // old rule resolved a near-tie by the third decimal and then wrote the
        // answer into a centroid via EMA, one window at a time, until the two
        // centroids had walked into each other. A window that names nobody
        // confidently must leave the clusterer bit-identical.
        let mut c = two_speakers();
        let bisector = at(APART / 2.0);
        let lead = (similarity(&bisector, &at(0.0)) - similarity(&bisector, &at(APART))).abs();
        assert!(lead < T_MARGIN, "the fixture leads by {lead}");

        let before = c.vectors();
        for _ in 0..100 {
            let heard = c.observe(&bisector);
            assert!(heard.is_guess(), "ambiguity is not an answer: {heard:?}");
        }
        assert_eq!(c.vectors(), before, "an ambiguous window moved a centroid");
        assert_eq!(c.live_labels(), vec![1, 2], "and nothing retired");
        assert_eq!(c.minted(), 2, "and nothing was minted out of the ambiguity");
        // Both speakers still answer to their own numbers afterwards.
        assert_eq!(c.observe(&at(0.0)), Heard::Settled(1));
        assert_eq!(c.observe(&at(APART)), Heard::Settled(2));
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
        assert_eq!(c.observe(&leaning), Heard::Settled(1), "a window is");
    }

    #[test]
    fn the_hop_margin_follows_the_configured_one() {
        // "Twice as clear" is a statement about the margin a deployment
        // chose, not about 0.10. A hard-coded 0.20 makes a server configured
        // at `diarize_margin = 0.30` hold its *hops* to a looser rule than its
        // windows, which inverts the reason the number exists.
        let build = |margin: f32| {
            let mut c = OnlineClusterer::with_config(MIN_POOL, margin);
            for _ in 0..MIN_POOL {
                c.observe(&voice(0));
            }
            for _ in 0..MIN_POOL {
                c.observe(&voice(1));
            }
            assert_eq!(c.live_labels(), vec![1, 2]);
            c
        };
        let leaning = embedding(&[(0, 0.85), (1, 0.40)]);
        let lead = similarity(&leaning, &voice(0)) - similarity(&leaning, &voice(1));

        let cautious = 0.30f32;
        assert!(
            (cautious..SHORT_MARGIN_FACTOR * cautious).contains(&lead),
            "the probe leads by {lead}, which has to clear the configured \
             margin and not twice it"
        );
        assert_eq!(build(cautious).observe_short(&leaning), None);
        assert_eq!(build(cautious).observe(&leaning), Heard::Settled(1));

        // At the shipped margin the same vector is clear enough for a hop,
        // so the difference is the configuration and not the vector.
        assert!(lead >= SHORT_MARGIN_FACTOR * T_MARGIN);
        assert_eq!(build(T_MARGIN).observe_short(&leaning), Some(1));
        assert_eq!(T_MARGIN_SHORT, SHORT_MARGIN_FACTOR * T_MARGIN);
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
        let (mut c, second, drags) = converging_pair();
        drive_until_retired(&mut c, &drags);
        assert_eq!(
            c.observe_short(&second),
            Some(1),
            "speaker 2 retired, so its voice answers to speaker 1"
        );
    }
}
