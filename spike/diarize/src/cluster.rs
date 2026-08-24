//! The spec's online clusterer: eager assign, reluctant create, EMA
//! centroids, retire-don't-merge.
//!
//! Deliberately asymmetric. Joining an existing speaker is cheap because a
//! wrong join costs one mislabelled sentence; minting a speaker is expensive
//! because a wrong mint costs a permanent extra label. This is the whole of
//! the spec's "label stability preferred" trade, and it is pure arithmetic —
//! Phase 2 lifts this file more or less verbatim into CI-tested code.

use crate::embed::{cosine, l2_normalize};

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Cosine similarity at which an embedding joins its nearest centroid.
    pub t_assign: f32,
    /// Two centroids this close have converged on one voice; the newer
    /// retires.
    pub t_retire: f32,
    /// EMA weight for a centroid's newest member. Small, so a centroid is
    /// dominated by its history.
    pub alpha: f32,
    /// Mutually agreeing pooled windows needed before a new speaker exists.
    pub min_pool: usize,
}

impl Default for Params {
    /// What the spike settled on, not what it started from. The design's
    /// original guess of 0.6/0.85 is the configuration the sweep disproved:
    /// at 0.6 this clusterer finds two speakers in a four-speaker meeting.
    fn default() -> Self {
        Self {
            t_assign: 0.45,
            t_retire: 0.80,
            alpha: 0.05,
            min_pool: 4,
        }
    }
}

struct Centroid {
    id: u32,
    vec: Vec<f32>,
    /// Set when this centroid has been retired into an older one.
    forwards_to: Option<u32>,
}

pub struct Clusterer {
    params: Params,
    centroids: Vec<Centroid>,
    pool: Vec<Vec<f32>>,
    next_id: u32,
}

impl Clusterer {
    pub fn new(params: Params) -> Self {
        Self {
            params,
            centroids: Vec::new(),
            pool: Vec::new(),
            next_id: 1,
        }
    }

    /// The label an embedding belongs to, or `None` while the clusterer is
    /// still undecided. `None` is an honest answer, not a failure.
    pub fn push(&mut self, emb: &[f32]) -> Option<u32> {
        let emb = l2_normalize(emb);

        if let Some((idx, sim)) = self.nearest(&emb)
            && sim >= self.params.t_assign
        {
            let a = self.params.alpha;
            let c = &mut self.centroids[idx];
            for (x, e) in c.vec.iter_mut().zip(&emb) {
                *x = (1.0 - a) * *x + a * e;
            }
            c.vec = l2_normalize(&c.vec);
            let id = c.id;

            // An assignment settles the question the pool was asking.
            self.pool.clear();
            self.retire_converged();
            return Some(self.resolve(id));
        }

        self.pool.push(emb);
        if self.pool.len() < self.params.min_pool {
            return None;
        }

        if self.pool_agrees() {
            let mean = self.pool_mean();
            // A pool that has drifted into an existing speaker's territory
            // while its members agreed with each other must not mint.
            if self
                .nearest(&mean)
                .is_some_and(|(_, sim)| sim >= self.params.t_assign)
            {
                self.pool.clear();
                return None;
            }
            let id = self.next_id;
            self.next_id += 1;
            self.centroids.push(Centroid {
                id,
                vec: mean,
                forwards_to: None,
            });
            self.pool.clear();
            return Some(id);
        }

        // Disagreement: the oldest pooled vector is the one least likely to
        // belong with whatever is arriving now.
        self.pool.remove(0);
        None
    }

    /// Nearest *active* centroid and its similarity.
    fn nearest(&self, emb: &[f32]) -> Option<(usize, f32)> {
        self.centroids
            .iter()
            .enumerate()
            .filter(|(_, c)| c.forwards_to.is_none())
            .map(|(i, c)| (i, cosine(emb, &c.vec)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn pool_agrees(&self) -> bool {
        self.pool.iter().enumerate().all(|(i, a)| {
            self.pool[i + 1..]
                .iter()
                .all(|b| cosine(a, b) >= self.params.t_assign)
        })
    }

    fn pool_mean(&self) -> Vec<f32> {
        let dim = self.pool[0].len();
        let mut mean = vec![0.0f32; dim];
        for v in &self.pool {
            for (m, x) in mean.iter_mut().zip(v) {
                *m += x;
            }
        }
        l2_normalize(&mean)
    }

    /// After an EMA nudge, two centroids may have converged on one voice.
    /// The newer one retires; past labels are never rewritten.
    fn retire_converged(&mut self) {
        loop {
            let mut found = None;
            'outer: for i in 0..self.centroids.len() {
                for j in i + 1..self.centroids.len() {
                    if self.centroids[i].forwards_to.is_some()
                        || self.centroids[j].forwards_to.is_some()
                    {
                        continue;
                    }
                    if cosine(&self.centroids[i].vec, &self.centroids[j].vec)
                        >= self.params.t_retire
                    {
                        found = Some((i, j));
                        break 'outer;
                    }
                }
            }
            let Some((older, newer)) = found else { return };
            self.centroids[newer].forwards_to = Some(self.centroids[older].id);
        }
    }

    /// Follow retirement forwarding to the label a caller should see.
    fn resolve(&self, mut id: u32) -> u32 {
        while let Some(next) = self
            .centroids
            .iter()
            .find(|c| c.id == id)
            .and_then(|c| c.forwards_to)
        {
            id = next;
        }
        id
    }

    /// Labels ever minted, including retired ones.
    pub fn minted(&self) -> u32 {
        self.next_id - 1
    }

    pub fn active(&self) -> usize {
        self.centroids
            .iter()
            .filter(|c| c.forwards_to.is_none())
            .count()
    }

    /// Cosine similarity between every pair of surviving centroids. The
    /// largest of these against `t_assign` is the margin the clusterer had
    /// left — how much room a meeting with more people would still have.
    pub fn crowding(&self) -> Vec<(u32, u32, f32)> {
        let active: Vec<&Centroid> = self
            .centroids
            .iter()
            .filter(|c| c.forwards_to.is_none())
            .collect();
        let mut out = Vec::new();
        for i in 0..active.len() {
            for j in i + 1..active.len() {
                out.push((
                    active[i].id,
                    active[j].id,
                    cosine(&active[i].vec, &active[j].vec),
                ));
            }
        }
        out.sort_by(|a, b| b.2.total_cmp(&a.2));
        out
    }
}
