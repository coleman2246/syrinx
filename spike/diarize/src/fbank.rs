//! Kaldi-compatible 80-dim log-mel filterbank.
//!
//! Every candidate embedding model was trained on features from Kaldi's
//! `compute-fbank-feats` — WeSpeaker and 3D-Speaker both go through
//! torchaudio's `compliance.kaldi.fbank` — so this reproduces Kaldi's exact
//! order of operations: DC removal, pre-emphasis, Povey window, power
//! spectrum, HTK-mel triangles, natural log. A mismatch here does not fail
//! loudly; it yields embeddings that look plausible and separate nobody,
//! which is what the `verify` subcommand exists to catch.

use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::sync::Arc;

pub const NUM_BINS: usize = 80;
pub const SAMPLE_RATE: f32 = 16_000.0;
pub const FRAME_LEN: usize = 400; // 25 ms
pub const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_LEN: usize = 512;
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
/// Kaldi's `high_freq = 0` means "Nyquist", which is what both training
/// recipes used. 7600 is the speech-synthesis convention and is *not* what
/// these models saw.
const HIGH_FREQ: f32 = 8_000.0;
/// Kaldi reads int16 PCM and never rescales it; WeSpeaker's recipe multiplies
/// its floats by 1<<15 to match. Cepstral mean normalisation cancels the
/// resulting constant offset, but only for models that apply CMN.
const INT16_SCALE: f32 = 32_768.0;

/// How a model expects its features to be normalised across time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Norm {
    /// Subtract the per-dimension mean over the utterance (WeSpeaker,
    /// 3D-Speaker).
    Mean,
    /// Subtract mean and divide by standard deviation (NeMo `per_feature`).
    MeanVar,
    None,
}

pub struct Fbank {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// One triangle per mel bin, stored sparsely as (first FFT bin, weights).
    melbank: Vec<(usize, Vec<f32>)>,
}

impl Fbank {
    pub fn new() -> Self {
        let fft = FftPlanner::new().plan_fft_forward(FFT_LEN);

        // Povey window: a Hann raised to 0.85, Kaldi's default for fbank.
        let a = std::f32::consts::TAU / (FRAME_LEN as f32 - 1.0);
        let window = (0..FRAME_LEN)
            .map(|i| (0.5 - 0.5 * (a * i as f32).cos()).powf(0.85))
            .collect();

        Self {
            fft,
            window,
            melbank: mel_filterbank(),
        }
    }

    /// Number of frames Kaldi would emit for `n` samples (`snip_edges=true`).
    pub fn num_frames(n: usize) -> usize {
        if n < FRAME_LEN {
            0
        } else {
            1 + (n - FRAME_LEN) / FRAME_SHIFT
        }
    }

    /// Log-mel features, row-major: frame 0's 80 bins, then frame 1's, …
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let frames = Self::num_frames(samples.len());
        let mut out = Vec::with_capacity(frames * NUM_BINS);
        let mut buf = vec![Complex32::default(); FFT_LEN];
        let mut power = vec![0.0f32; FFT_LEN / 2];

        for f in 0..frames {
            let frame = &samples[f * FRAME_SHIFT..f * FRAME_SHIFT + FRAME_LEN];

            let mean = frame.iter().sum::<f32>() / FRAME_LEN as f32;
            let mut w: Vec<f32> = frame.iter().map(|s| (s - mean) * INT16_SCALE).collect();

            // Kaldi pre-emphasises in place from the tail, so each sample sees
            // its *original* predecessor; the first sample uses itself.
            for i in (1..FRAME_LEN).rev() {
                w[i] -= PREEMPH * w[i - 1];
            }
            w[0] -= PREEMPH * w[0];

            buf.iter_mut().for_each(|c| *c = Complex32::default());
            for (i, (s, win)) in w.iter().zip(&self.window).enumerate() {
                buf[i].re = s * win;
            }
            self.fft.process(&mut buf);

            for (k, p) in power.iter_mut().enumerate() {
                *p = buf[k].re * buf[k].re + buf[k].im * buf[k].im;
            }

            for (offset, weights) in &self.melbank {
                let energy: f32 = weights
                    .iter()
                    .zip(&power[*offset..])
                    .map(|(w, p)| w * p)
                    .sum();
                out.push(energy.max(f32::EPSILON).ln());
            }
        }
        out
    }

    /// Normalise features in place across time, per dimension.
    pub fn normalize(feats: &mut [f32], norm: Norm) {
        if norm == Norm::None {
            return;
        }
        let frames = feats.len() / NUM_BINS;
        if frames == 0 {
            return;
        }
        for d in 0..NUM_BINS {
            let mut mean = 0.0;
            for f in 0..frames {
                mean += feats[f * NUM_BINS + d];
            }
            mean /= frames as f32;

            let scale = if norm == Norm::MeanVar {
                let mut var = 0.0;
                for f in 0..frames {
                    let x = feats[f * NUM_BINS + d] - mean;
                    var += x * x;
                }
                1.0 / (var / frames as f32).sqrt().max(1e-5)
            } else {
                1.0
            };

            for f in 0..frames {
                feats[f * NUM_BINS + d] = (feats[f * NUM_BINS + d] - mean) * scale;
            }
        }
    }
}

fn mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Kaldi's triangular mel bank over the first `FFT_LEN/2` power-spectrum bins.
fn mel_filterbank() -> Vec<(usize, Vec<f32>)> {
    let num_fft_bins = FFT_LEN / 2;
    let bin_width = SAMPLE_RATE / FFT_LEN as f32;
    let (mel_low, mel_high) = (mel(LOW_FREQ), mel(HIGH_FREQ));
    let delta = (mel_high - mel_low) / (NUM_BINS + 1) as f32;

    (0..NUM_BINS)
        .map(|m| {
            let left = mel_low + m as f32 * delta;
            let (center, right) = (left + delta, left + 2.0 * delta);
            let mut offset = None;
            let mut weights = Vec::new();

            for k in 0..num_fft_bins {
                let mf = mel(bin_width * k as f32);
                let w = if mf > left && mf <= center {
                    (mf - left) / delta
                } else if mf > center && mf < right {
                    (right - mf) / delta
                } else {
                    0.0
                };
                if w != 0.0 {
                    offset.get_or_insert(k);
                    weights.push(w);
                } else if offset.is_some() {
                    break;
                }
            }
            (offset.unwrap_or(0), weights)
        })
        .collect()
}
