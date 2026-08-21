//! Microphone capture via cpal, normalised to what the server expects.
//!
//! The server wants 16 kHz mono `f32`. Conversion uses the helpers in
//! `parakeet-proto` so every client converts identically and a client-side bug
//! cannot look like a model problem.

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SupportedStreamConfig;
use parakeet_proto::{SAMPLE_RATE, downmix_to_mono, resample_to_16k};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Start capturing from the default input device.
///
/// Returns a receiver of 16 kHz mono f32 samples and the stream, which must be
/// kept alive: dropping it stops capture.
pub fn start(tx: mpsc::Sender<Vec<f32>>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default audio input device")?;
    let device_name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "unknown".into());
    info!("capturing from {device_name}");

    let default = device
        .default_input_config()
        .context("querying default input config")?;

    // Prefer a native 16 kHz stream: no resampling means no added latency and
    // no quality loss. Most devices default to 44.1 or 48 kHz, so this usually
    // falls through -- an earlier version assumed PipeWire would simply
    // negotiate 16 kHz and sent 48 kHz audio to a 16 kHz model, which does not
    // fail loudly, it just transcribes badly.
    let chosen = native_16k_config(&device).unwrap_or_else(|| default.clone());

    let channels = chosen.channels();
    // cpal 0.18: SampleRate is a plain u32.
    let in_rate = chosen.sample_rate();
    let sample_format = chosen.sample_format();
    let config: cpal::StreamConfig = chosen.into();

    if in_rate == SAMPLE_RATE {
        info!("device provides {SAMPLE_RATE} Hz natively; no resampling needed");
    } else {
        info!("device runs at {in_rate} Hz; resampling to {SAMPLE_RATE} Hz");
    }

    let err_fn = |e| warn!("audio stream error: {e}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| {
                let mono = resample_to_16k(&downmix_to_mono(data, channels), in_rate);
                // try_send, not send: if the network cannot keep up, dropping
                // audio is better than growing an unbounded backlog and
                // drifting further behind real time.
                let _ = tx.try_send(mono);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|s| *s as f32 / 32768.0).collect();
                let mono = resample_to_16k(&downmix_to_mono(&f, channels), in_rate);
                let _ = tx.try_send(mono);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U8 => device.build_input_stream(
            config,
            move |data: &[u8], _| {
                // U8 is unsigned with 128 as silence.
                let f: Vec<f32> = data.iter().map(|s| (*s as f32 - 128.0) / 128.0).collect();
                let mono = resample_to_16k(&downmix_to_mono(&f, channels), in_rate);
                let _ = tx.try_send(mono);
            },
            err_fn,
            None,
        ),
        other => bail!(
            "unsupported sample format {other:?}; supported formats are F32, I16 and U8"
        ),
    }
    .context("building input stream")?;

    stream.play().context("starting the input stream")?;
    Ok(stream)
}

/// Find a supported input config that runs natively at 16 kHz, if the device
/// offers one. Preferred over resampling.
///
/// Format preference matters as much as sample rate. Devices commonly advertise
/// several 16 kHz ranges, and taking the first match blindly picked a U8 range
/// on this hardware -- an unsupported format, so capture failed outright even
/// though a perfectly good F32 range was also on offer.
fn native_16k_config(device: &cpal::Device) -> Option<SupportedStreamConfig> {
    let ranges: Vec<_> = device.supported_input_configs().ok()?.collect();

    // Lower score is better. F32 needs no conversion; I16 is a cheap divide;
    // U8 is 8-bit and a real quality loss, so it is a last resort.
    let rank = |f: cpal::SampleFormat| match f {
        cpal::SampleFormat::F32 => 0,
        cpal::SampleFormat::I16 => 1,
        cpal::SampleFormat::U8 => 2,
        _ => 99,
    };

    ranges
        .into_iter()
        .filter(|r| r.min_sample_rate() <= SAMPLE_RATE && SAMPLE_RATE <= r.max_sample_rate())
        .filter(|r| rank(r.sample_format()) < 99)
        .min_by_key(|r| rank(r.sample_format()))
        .map(|r| r.with_sample_rate(SAMPLE_RATE))
}
