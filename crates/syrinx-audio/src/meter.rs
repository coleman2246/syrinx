//! Level metering: a small spectrum for confirming a source is actually
//! carrying audio before committing to a session.
//!
//! Deliberately tiny. Ten bands is enough to tell speech from silence, from a
//! steady hum, from a source that is simply dead -- which is the whole question
//! being asked. A full analyser would be more code and no more informative.
//!
//! The FFT is implemented here rather than pulled in, because a radix-2
//! Cooley-Tukey over 1024 points is a few dozen lines and this is the only
//! place it is needed.

use std::f32::consts::PI;

/// Number of display bands.
pub const BANDS: usize = 10;

/// FFT window. At 16 kHz this is 64 ms -- long enough to resolve speech
/// fundamentals, short enough that the display tracks the voice.
const FFT_SIZE: usize = 1024;

/// Band edges in Hz, log-spaced across the range speech occupies.
///
/// Nyquist at 16 kHz is 8 kHz, so the top band closes there. Logarithmic
/// because pitch is: linear bands would spend eight of ten on the hiss above
/// 4 kHz and none on the vowels that carry a voice.
const BAND_EDGES: [f32; BANDS + 1] = [
    60.0, 120.0, 220.0, 380.0, 620.0, 1000.0, 1600.0, 2600.0, 4200.0, 6000.0, 8000.0,
];

/// Root-mean-square amplitude, 0.0 to 1.0.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Band magnitudes for a window of audio, each roughly 0.0 to 1.0.
///
/// Returns silence for an empty or too-short buffer rather than failing: a
/// meter that errors is worse than one that reads zero.
pub fn spectrum(samples: &[f32], sample_rate: u32) -> [f32; BANDS] {
    let mut bands = [0.0f32; BANDS];
    if samples.len() < FFT_SIZE || sample_rate == 0 {
        return bands;
    }

    // Most recent window: the meter should show now, not the start of the
    // buffer.
    let start = samples.len() - FFT_SIZE;
    let mut re: Vec<f32> = samples[start..].to_vec();

    // Hann window, to stop a tone that does not fit a whole number of cycles
    // smearing across every band.
    for (i, v) in re.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos();
        *v *= w;
    }
    let mut im = vec![0.0f32; FFT_SIZE];
    fft(&mut re, &mut im);

    let bin_hz = sample_rate as f32 / FFT_SIZE as f32;
    for b in 0..BANDS {
        let (lo, hi) = (BAND_EDGES[b], BAND_EDGES[b + 1]);
        let lo_bin = (lo / bin_hz).floor().max(1.0) as usize;
        let hi_bin = ((hi / bin_hz).ceil() as usize).min(FFT_SIZE / 2);
        if lo_bin >= hi_bin {
            continue;
        }
        let mut sum = 0.0;
        for k in lo_bin..hi_bin {
            sum += (re[k] * re[k] + im[k] * im[k]).sqrt();
        }
        // Mean rather than total, so a wide band is not louder just for being
        // wide.
        let mean = sum / (hi_bin - lo_bin) as f32;
        // Normalised by window size, then scaled: raw magnitudes are tiny for
        // speech and would render as a flat line.
        bands[b] = (mean / (FFT_SIZE as f32 / 4.0) * 8.0).clamp(0.0, 1.0);
    }
    bands
}

/// In-place iterative radix-2 Cooley-Tukey FFT. Length must be a power of two.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cur_r, mut cur_i) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cur_r - im[i + k + len / 2] * cur_i,
                    re[i + k + len / 2] * cur_i + im[i + k + len / 2] * cur_r,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let next_r = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = next_r;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Render bands as a bar string, for a terminal.
pub fn bars(bands: &[f32; BANDS]) -> String {
    // Eighth-blocks give eight levels per character without needing colour or
    // multiple lines.
    const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    bands
        .iter()
        .map(|v| {
            let idx = ((v.clamp(0.0, 1.0) * 8.0).round() as usize).min(8);
            BLOCKS[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / rate as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn silence_reads_as_zero() {
        let s = spectrum(&vec![0.0; FFT_SIZE], 16_000);
        assert!(s.iter().all(|v| *v < 1e-6), "got {s:?}");
        assert_eq!(rms(&vec![0.0; 100]), 0.0);
    }

    #[test]
    fn a_short_buffer_reads_as_zero_rather_than_failing() {
        // The meter is polled constantly; erroring on a partial buffer would be
        // noise, not information.
        assert!(spectrum(&[0.1, 0.2], 16_000).iter().all(|v| *v == 0.0));
        assert!(spectrum(&[], 16_000).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_zero_sample_rate_does_not_divide_by_zero() {
        assert!(spectrum(&vec![0.1; FFT_SIZE], 0).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_low_tone_lands_in_a_low_band() {
        // 150 Hz sits in band 1 (120-220). If banding or the FFT were wrong the
        // energy would show up somewhere else entirely.
        let s = spectrum(&tone(150.0, 16_000, FFT_SIZE), 16_000);
        let peak = s
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 1, "peak in band {peak}, spectrum {s:?}");
    }

    #[test]
    fn a_high_tone_lands_in_a_high_band() {
        // 5 kHz sits in band 8 (4200-6000).
        let s = spectrum(&tone(5000.0, 16_000, FFT_SIZE), 16_000);
        let peak = s
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 8, "peak in band {peak}, spectrum {s:?}");
    }

    #[test]
    fn a_louder_tone_reads_higher_than_a_quiet_one() {
        let loud = spectrum(&tone(1000.0, 16_000, FFT_SIZE), 16_000);
        let quiet: Vec<f32> = tone(1000.0, 16_000, FFT_SIZE)
            .iter()
            .map(|s| s * 0.1)
            .collect();
        let quiet = spectrum(&quiet, 16_000);
        let l: f32 = loud.iter().sum();
        let q: f32 = quiet.iter().sum();
        assert!(l > q * 2.0, "loud {l} vs quiet {q}");
    }

    #[test]
    fn only_the_most_recent_window_is_analysed() {
        // A long buffer of silence ending in a tone must read as a tone: the
        // meter shows now, not the average of everything captured.
        let mut buf = vec![0.0f32; FFT_SIZE * 4];
        buf.extend(tone(1000.0, 16_000, FFT_SIZE));
        let s = spectrum(&buf, 16_000);
        assert!(s.iter().sum::<f32>() > 0.01, "got {s:?}");
    }

    #[test]
    fn rms_of_a_half_amplitude_tone_is_about_a_third() {
        // sin at 0.5 amplitude has rms 0.5/sqrt(2) = 0.354.
        let r = rms(&tone(440.0, 16_000, 16_000));
        assert!((r - 0.354).abs() < 0.01, "got {r}");
    }

    #[test]
    fn bars_render_one_character_per_band() {
        assert_eq!(bars(&[0.0; BANDS]).chars().count(), BANDS);
        assert_eq!(bars(&[1.0; BANDS]).chars().count(), BANDS);
    }

    #[test]
    fn bars_grow_with_level() {
        let quiet = bars(&[0.0; BANDS]);
        let loud = bars(&[1.0; BANDS]);
        assert_ne!(quiet, loud);
        assert!(loud.contains('█'));
    }

    #[test]
    fn out_of_range_levels_do_not_panic() {
        // A miscomputed band must not index past the block table.
        let _ = bars(&[-5.0; BANDS]);
        let _ = bars(&[99.0; BANDS]);
    }
}
