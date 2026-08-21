//! Starting a capture, dispatched to whichever backend the source came from.

use crate::source::{Source, SourceTarget};
use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tracing::info;

/// A running capture. Dropping it stops the capture.
pub enum Capture {
    #[cfg(target_os = "linux")]
    PipeWire(crate::pipewire::PwCapture),
    Cpal(CpalCapture),
}

impl Capture {
    /// Begin capturing 16 kHz mono f32 samples from `source`.
    pub fn start(source: &Source, tx: mpsc::Sender<Vec<f32>>) -> Result<Self> {
        info!("capturing {:?} ({})", source.kind, source.display());
        match &source.target {
            #[cfg(target_os = "linux")]
            SourceTarget::PipeWireNode(id) => {
                // Only a real capture device is addressable with `--target`.
                //
                // A sink's monitor and an application's stream both have to be
                // linked port-to-port instead. A sink exposes monitor_FL/FR as
                // *output* ports, and `--target <sink>` does not reliably wire
                // them into a capture stream -- selecting a monitor that way
                // produced silence while the same card's capture device
                // happened to work, which reads as the two being swapped.
                let cap = match source.kind {
                    crate::SourceKind::Microphone => {
                        crate::pipewire::PwCapture::start(*id, tx)?
                    }
                    crate::SourceKind::Monitor | crate::SourceKind::Application => {
                        crate::pipewire::PwCapture::start_linked(*id, tx)?
                    }
                };
                Ok(Capture::PipeWire(cap))
            }
            #[cfg(not(target_os = "linux"))]
            SourceTarget::PipeWireNode(_) => {
                anyhow::bail!("this source came from PipeWire, which is only available on Linux")
            }
            SourceTarget::CpalDevice { name, loopback } => {
                Ok(Capture::Cpal(CpalCapture::start(name, *loopback, tx)?))
            }
        }
    }
}

/// cpal capture. The stream is not `Send`, so it lives on its own thread and is
/// stopped by dropping the handle.
pub struct CpalCapture {
    _stop: std::sync::mpsc::Sender<()>,
}

impl CpalCapture {
    fn start(name: &str, loopback: bool, tx: mpsc::Sender<Vec<f32>>) -> Result<Self> {
        use cpal::traits::{DeviceTrait, StreamTrait};
        use syrinx_proto::{downmix_to_mono, resample_to_16k};

        let device = crate::cpal_backend::find_device(name, loopback)?;
        // A loopback source is a *render* endpoint being read. cpal builds that
        // stream correctly -- it sets AUDCLNT_STREAMFLAGS_LOOPBACK for an input
        // stream on a render device -- but it refuses to describe one:
        // `default_input_config` answers "Device does not support input" for
        // anything that is not a capture endpoint. The format to ask for is the
        // one the device renders in, so query the output side and open it for
        // input.
        let supported = if loopback {
            device
                .default_output_config()
                .context("querying the output config of a loopback device")?
        } else {
            device
                .default_input_config()
                .context("querying input config")?
        };
        let channels = supported.channels();
        let rate = supported.sample_rate();
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();

        // cpal::Stream is !Send, so it cannot be moved into a tokio task; it
        // gets a dedicated thread that holds it alive until asked to stop.
        std::thread::spawn(move || {
            let err_fn = |e| tracing::warn!("audio stream error: {e}");
            let stream = match format {
                cpal::SampleFormat::F32 => device.build_input_stream(
                    config,
                    move |d: &[f32], _| {
                        let _ = tx.try_send(resample_to_16k(&downmix_to_mono(d, channels), rate));
                    },
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::I16 => device.build_input_stream(
                    config,
                    move |d: &[i16], _| {
                        let f: Vec<f32> = d.iter().map(|s| *s as f32 / 32768.0).collect();
                        let _ = tx.try_send(resample_to_16k(&downmix_to_mono(&f, channels), rate));
                    },
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::U8 => device.build_input_stream(
                    config,
                    move |d: &[u8], _| {
                        // U8 is unsigned with 128 as silence.
                        let f: Vec<f32> =
                            d.iter().map(|s| (*s as f32 - 128.0) / 128.0).collect();
                        let _ = tx.try_send(resample_to_16k(&downmix_to_mono(&f, channels), rate));
                    },
                    err_fn,
                    None,
                ),
                other => {
                    tracing::error!("unsupported sample format {other:?}");
                    return;
                }
            };
            let Ok(stream) = stream else {
                tracing::error!("failed to build the input stream");
                return;
            };
            if stream.play().is_err() {
                tracing::error!("failed to start the input stream");
                return;
            }
            // Block until the handle is dropped; the stream stops with us.
            let _ = stop_rx.recv();
        });

        Ok(Self { _stop: stop_tx })
    }
}
