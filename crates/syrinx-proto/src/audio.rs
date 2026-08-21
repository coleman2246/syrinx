//! Audio conversion shared by every client.
//!
//! Both clients must hand the server 16 kHz mono `f32` in `[-1.0, 1.0]`. Keeping
//! that conversion here means a client bug cannot masquerade as a model problem,
//! and every client converts identically.

/// Target sample rate. The model is trained at 16 kHz; anything else must be
/// resampled by the client before sending.
pub const SAMPLE_RATE: u32 = 16_000;

/// Decode little-endian signed 16-bit PCM into normalised f32 samples.
///
/// A trailing odd byte is dropped rather than panicking, since network frames
/// can split mid-sample.
pub fn pcm_s16le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

/// Average interleaved channels down to mono. A trailing partial frame is
/// dropped for the same reason as above.
pub fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let n = channels as usize;
    samples
        .chunks_exact(n)
        .map(|frame| frame.iter().sum::<f32>() / n as f32)
        .collect()
}

/// Resample to [`SAMPLE_RATE`] (16 kHz).
///
/// Microphones commonly run at 44.1 or 48 kHz and cpal reports the device
/// default, so a client that sends whatever it is given will feed the model
/// audio at the wrong rate. That does not fail loudly -- it just transcribes
/// badly, which is far harder to attribute later.
///
/// Integer ratios (48000 -> 16000 is exactly 3:1) use box-filter decimation:
/// averaging each group of N samples, which provides basic anti-aliasing rather
/// than naively dropping samples. Non-integer ratios fall back to linear
/// interpolation. Neither is a polyphase FIR, but both are well within what a
/// 16 kHz speech model needs.
pub fn resample_to_16k(samples: &[f32], from_rate: u32) -> Vec<f32> {
    if from_rate == SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }

    if from_rate % SAMPLE_RATE == 0 {
        let factor = (from_rate / SAMPLE_RATE) as usize;
        return samples
            .chunks_exact(factor)
            .map(|g| g.iter().sum::<f32>() / factor as f32)
            .collect();
    }

    let ratio = from_rate as f64 / SAMPLE_RATE as f64;
    let out_len = (samples.len() as f64 / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;
            let a = samples[idx];
            let b = *samples.get(idx + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s16_converts_to_unit_range() {
        assert!((pcm_s16le_to_f32(&[0x00, 0x00])[0] - 0.0).abs() < 1e-6);
        // i16::MAX little-endian
        assert!((pcm_s16le_to_f32(&[0xff, 0x7f])[0] - 1.0).abs() < 1e-4);
        // i16::MIN little-endian maps to exactly -1.0
        assert!((pcm_s16le_to_f32(&[0x00, 0x80])[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn odd_length_input_drops_trailing_byte() {
        // A partial sample must not panic or produce garbage.
        assert_eq!(pcm_s16le_to_f32(&[0x00, 0x00, 0x11]).len(), 1);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(pcm_s16le_to_f32(&[]).is_empty());
        assert!(downmix_to_mono(&[], 2).is_empty());
    }

    #[test]
    fn stereo_downmixes_to_mono_by_averaging() {
        assert_eq!(downmix_to_mono(&[1.0, -1.0, 0.5, 0.5], 2), vec![0.0, 0.5]);
    }

    #[test]
    fn mono_passthrough_is_unchanged() {
        assert_eq!(downmix_to_mono(&[0.25, 0.5], 1), vec![0.25, 0.5]);
    }

    #[test]
    fn resampling_at_the_target_rate_is_a_passthrough() {
        assert_eq!(resample_to_16k(&[0.1, 0.2, 0.3], 16_000), vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn forty_eight_k_decimates_by_exactly_three() {
        // The common case: nearly every mic defaults to 48 kHz.
        let input: Vec<f32> = (0..48_000).map(|i| (i % 7) as f32).collect();
        assert_eq!(resample_to_16k(&input, 48_000).len(), 16_000);
    }

    #[test]
    fn integer_decimation_averages_rather_than_dropping_samples() {
        // Averaging gives basic anti-aliasing; dropping 2 of every 3 samples
        // would fold high frequencies down into the speech band.
        let out = resample_to_16k(&[0.0, 3.0, 6.0, 1.0, 1.0, 1.0], 48_000);
        assert_eq!(out, vec![3.0, 1.0]);
    }

    #[test]
    fn forty_four_one_k_uses_interpolation_and_lands_close_to_the_ratio() {
        let input: Vec<f32> = (0..44_100).map(|i| (i % 5) as f32).collect();
        let out = resample_to_16k(&input, 44_100);
        // 44100 -> 16000 is not an integer ratio; allow a sample of slack.
        assert!((out.len() as i64 - 16_000).abs() <= 1, "got {}", out.len());
    }

    #[test]
    fn resampling_an_empty_buffer_does_not_panic() {
        assert!(resample_to_16k(&[], 48_000).is_empty());
    }

    #[test]
    fn a_buffer_shorter_than_the_decimation_factor_yields_nothing() {
        // chunks_exact drops the remainder; the caller must not assume output.
        assert!(resample_to_16k(&[1.0, 2.0], 48_000).is_empty());
    }

    #[test]
    fn zero_channels_is_treated_as_passthrough_not_a_divide_by_zero() {
        // Defensive: a miscomputed channel count must not panic the server.
        assert_eq!(downmix_to_mono(&[0.25, 0.5], 0), vec![0.25, 0.5]);
    }
}
