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
//! The constants below are measurements, not guesses: see "Spike results" in
//! `docs/specs/2026-08-24-speaker-diarization-design.md`, where they were
//! swept against three hand-annotated AMI meetings.
//!
//! Pure arithmetic. No models, no `ort`, no feature gate -- which is what lets
//! the whole of the labelling policy be tested in CI on synthetic embeddings.

// The four constants are `#[doc(hidden)] pub` rather than private for one
// reason: `examples/diarize_probe` overrides them one at a time, and it is
// external to this crate, so a private const would force it to keep its own
// copy of the other three. A stale copy there would attribute a sweep's
// numbers to constants the server no longer ships, which is precisely the
// failure the probe exists to prevent. Hidden from the docs because they are
// still not a configuration surface -- see [`OnlineClusterer::with_params`].

/// Cosine similarity above which an embedding joins its nearest centroid.
/// 0.45, not the 0.6 the design first guessed: the spike measured
/// same-speaker windows at a median cosine of 0.52, so 0.6 rejects most true
/// matches.
#[doc(hidden)]
pub const T_ASSIGN: f32 = 0.45;
/// Mutually agreeing orphan windows before a new speaker is minted. Not a
/// free parameter: at 2 the spike minted 20 labels for a 4-speaker meeting,
/// and 3 still failed an 87-minute one.
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
    /// Orphan windows that matched nothing, awaiting enough agreement to
    /// become a new speaker.
    pool: Vec<Vec<f32>>,
    next_label: u32,
    /// The four thresholds, held rather than read from the consts so that the
    /// probe can sweep them. Every shipped clusterer is built by
    /// [`OnlineClusterer::new`] and carries exactly the consts above.
    t_assign: f32,
    t_retire: f32,
    alpha: f32,
    min_pool: usize,
}

impl Default for OnlineClusterer {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineClusterer {
    pub fn new() -> Self {
        Self::with_params(T_ASSIGN, T_RETIRE, EMA_ALPHA, MIN_POOL)
    }

    /// A clusterer with the thresholds overridden. **Not a configuration
    /// surface**: shipped behaviour is [`OnlineClusterer::new`]'s, which is
    /// the calibrated constants above and the only construction the server
    /// ever performs.
    ///
    /// This exists for `examples/diarize_probe`'s sweep, which re-measures
    /// those constants against annotated meetings and therefore has to vary
    /// them without forking the algorithm -- a sweep against a second copy of
    /// this code would answer a question about the copy. `new` delegates here
    /// so there is one assignment of the constants to the fields rather than
    /// two that have to agree.
    #[doc(hidden)]
    pub fn with_params(t_assign: f32, t_retire: f32, alpha: f32, min_pool: usize) -> Self {
        Self {
            centroids: Vec::new(),
            pool: Vec::new(),
            next_label: 1,
            t_assign,
            t_retire,
            alpha,
            min_pool,
        }
    }

    /// The label this embedding belongs to, or `None` while the clusterer is
    /// still undecided.
    ///
    /// `None` is an honest answer rather than a failure: it means the window
    /// matched no known speaker and there is not yet enough agreeing evidence
    /// to call it a new one. Pooling a window is not an assignment.
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

        if let Some((index, similarity)) = self.nearest(&embedding)
            && similarity >= self.t_assign
        {
            let alpha = self.alpha;
            let centroid = &mut self.centroids[index];
            for (x, e) in centroid.vector.iter_mut().zip(&embedding) {
                *x = (1.0 - alpha) * *x + alpha * e;
            }
            centroid.vector = l2_normalize(&centroid.vector);
            let label = centroid.label;

            // An assignment settles the question the pool was asking.
            self.pool.clear();
            // That nudge may have walked this centroid into another one.
            self.retire_converged();
            return Some(self.resolve(label));
        }

        self.pool.push(embedding);
        if self.pool.len() < self.min_pool {
            return None;
        }

        if !self.pool_agrees() {
            // The oldest pooled window is the one least likely to belong with
            // whatever is arriving now.
            self.pool.remove(0);
            return None;
        }

        let mean = self.pool_mean();
        // A pool whose members agreed with each other while each fell just
        // short of a live centroid must not mint: their mean is the less
        // noisy vector, and it has landed back inside a speaker who already
        // has a label. Minting here would split that speaker.
        if self
            .nearest(&mean)
            .is_some_and(|(_, similarity)| similarity >= self.t_assign)
        {
            self.pool.clear();
            return None;
        }

        let label = self.next_label;
        self.next_label += 1;
        self.centroids.push(Centroid {
            label,
            vector: mean,
            retired_into: None,
        });
        self.pool.clear();
        Some(label)
    }

    /// The nearest live centroid and its similarity, by index. Retired
    /// centroids are invisible here: they exist only to forward.
    fn nearest(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        self.centroids
            .iter()
            .enumerate()
            .filter(|(_, c)| c.retired_into.is_none())
            .map(|(i, c)| (i, cosine(embedding, &c.vector)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Whether every pooled window is within `T_ASSIGN` of every other. The
    /// pool holds evidence for *one* new speaker, so a single disagreeing
    /// pair means it is not yet that evidence.
    ///
    /// Every-pair comparison, which is fine at this size: the pool never
    /// exceeds the pool threshold (4 as shipped) entries, since reaching it
    /// either mints, clears, or evicts the oldest.
    fn pool_agrees(&self) -> bool {
        self.pool.iter().enumerate().all(|(i, a)| {
            self.pool[i + 1..]
                .iter()
                .all(|b| cosine(a, b) >= self.t_assign)
        })
    }

    fn pool_mean(&self) -> Vec<f32> {
        let mut mean = vec![0.0f32; self.pool[0].len()];
        for v in &self.pool {
            for (m, x) in mean.iter_mut().zip(v) {
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
    fn an_assignment_clears_the_pool() {
        let mut c = OnlineClusterer::new();
        for i in 0..MIN_POOL as u32 {
            c.observe(&noisy(0, i));
        }

        // Three quarters of the evidence for a second speaker...
        for i in 0..MIN_POOL as u32 - 1 {
            assert_eq!(c.observe(&noisy(1, i)), None);
        }
        // ...then the first speaker takes a turn, which settles the question
        // the pool was asking.
        assert_eq!(c.observe(&noisy(0, 50)), Some(1));

        // So the second speaker starts over: three more windows are still not
        // enough, and the fourth is.
        for i in 10..10 + MIN_POOL as u32 - 1 {
            assert_eq!(c.observe(&noisy(1, i)), None);
        }
        assert_eq!(c.observe(&noisy(1, 20)), Some(2));
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

        let mut c = OnlineClusterer::new();
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
        // prove: `new` delegates to `with_params`, so this can only fail if a
        // later edit stops it delegating and passes something other than the
        // constants. It says nothing about whether `with_params` reads the
        // arguments it is handed -- the four tests below are what cover that,
        // and they are the ones the sweep actually rests on.
        let (_, second, between) = converging_pair();

        // One script through all four constants: pooling and minting
        // (MIN_POOL), assignment (T_ASSIGN), a centroid dragged into another
        // one window at a time (EMA_ALPHA) until it retires (T_RETIRE), and a
        // fresh voice afterwards to show the freed label is not reused.
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
            T_ASSIGN, T_RETIRE, EMA_ALPHA, MIN_POOL,
        ));
        assert_eq!(shipped, swept);

        // And the script is worth comparing: an equivalence that held only
        // because nothing happened would pass this test while proving nothing.
        let (labels, live) = shipped;
        assert_eq!(live, vec![1, 3], "speaker 2 should have retired into 1");
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
            OnlineClusterer::with_params(T_ASSIGN, T_RETIRE, EMA_ALPHA, 2),
            &script,
        );
        assert_eq!(shipped, vec![None, None], "MIN_POOL 4 wants four windows");
        assert_eq!(swept, vec![None, Some(1)], "a pool of 2 should have minted");
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
            OnlineClusterer::with_params(0.99, T_RETIRE, EMA_ALPHA, MIN_POOL),
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
            OnlineClusterer::with_params(T_ASSIGN, T_RETIRE, 1.0, MIN_POOL),
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
            OnlineClusterer::with_params(T_ASSIGN, 0.35, EMA_ALPHA, MIN_POOL),
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
