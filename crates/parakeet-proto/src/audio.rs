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
    fn zero_channels_is_treated_as_passthrough_not_a_divide_by_zero() {
        // Defensive: a miscomputed channel count must not panic the server.
        assert_eq!(downmix_to_mono(&[0.25, 0.5], 0), vec![0.25, 0.5]);
    }
}
