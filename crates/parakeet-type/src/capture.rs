//! Microphone capture via cpal, normalised to what the server expects.
//!
//! The server wants 16 kHz mono `f32`. Conversion uses the helpers in
//! `parakeet-proto` so every client converts identically and a client-side bug
//! cannot look like a model problem.

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parakeet_proto::{SAMPLE_RATE, downmix_to_mono};
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
    let channels = default.channels();
    // cpal 0.18: SampleRate is a plain u32.
    let in_rate = default.sample_rate();

    let config: cpal::StreamConfig = default.into();
    let sample_format = default.sample_format();

    // Resampling is not implemented: the model is trained at 16 kHz, and a
    // naive resampler would degrade accuracy in a way that is hard to attribute
    // later. PipeWire and WASAPI both negotiate 16 kHz happily.
    if in_rate != SAMPLE_RATE {
        warn!(
            "input device runs at {in_rate} Hz, not {SAMPLE_RATE} Hz; \
             audio will be sent at the wrong rate and transcription will suffer"
        );
    }

    let err_fn = |e| warn!("audio stream error: {e}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| {
                let mono = downmix_to_mono(data, channels);
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
                let _ = tx.try_send(downmix_to_mono(&f, channels));
            },
            err_fn,
            None,
        ),
        other => bail!("unsupported sample format {other:?}"),
    }
    .context("building input stream")?;

    stream.play().context("starting the input stream")?;
    Ok(stream)
}
