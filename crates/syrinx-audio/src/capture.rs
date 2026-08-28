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
                    crate::SourceKind::Microphone => crate::pipewire::PwCapture::start(*id, tx)?,
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

/// How long a device gets to open before it is called a failure.
///
/// Bounded because the whole point of this rendezvous is that the caller
/// learns the truth, and a device that never finishes opening would otherwise
/// substitute a hang for the silence it replaced. Ten seconds is far longer
/// than any real open takes, on WASAPI or anywhere else.
const OPEN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Run a stream on a thread of its own, and report whether it started.
///
/// `cpal::Stream` is not `Send` on every platform, which is the only reason
/// the thread exists. `start` runs on it, and everything a caller learns about
/// the attempt comes back over the rendezvous below -- so a stream that fails
/// to build or play reaches the caller as an error rather than as a live
/// handle producing nothing.
///
/// The whole of the cpal work runs here, device lookup and config query
/// included, so that a panic anywhere in it can only cost this thread. The
/// caller sees that as a failed open rather than as its own stack unwinding.
///
/// The stream is dropped when the returned sender is, which is what stops the
/// capture.
fn on_stream_thread<S: 'static>(
    start: impl FnOnce() -> Result<S> + Send + 'static,
) -> Result<std::sync::mpsc::Sender<()>> {
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    std::thread::spawn(move || {
        // Formatted here rather than sent whole: an `anyhow::Error` is not
        // `Send` across this channel in every shape it can take, and what the
        // caller needs is the chain, which `{:#}` carries.
        let stream = match crate::caught("opening the audio device", start) {
            Ok(s) => {
                let _ = ready_tx.send(Ok(()));
                s
            }
            Err(e) => {
                let _ = ready_tx.send(Err(format!("{e:#}")));
                return;
            }
        };
        // Block until the handle is dropped; the stream stops with us.
        let _ = stop_rx.recv();
        drop(stream);
    });

    match ready_rx.recv_timeout(OPEN_TIMEOUT) {
        Ok(Ok(())) => Ok(stop_tx),
        Ok(Err(e)) => Err(anyhow::anyhow!(e)),
        // Disconnected: the thread unwound before it could answer, which cpal
        // backends do -- the Windows one panics rather than erroring in
        // several places. Either way the caller gets an error, not a handle to
        // a stream that is not running.
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("the audio thread stopped before the stream started")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            anyhow::bail!("the audio device did not open within {OPEN_TIMEOUT:?}")
        }
    }
}

impl CpalCapture {
    fn start(name: &str, loopback: bool, tx: mpsc::Sender<Vec<f32>>) -> Result<Self> {
        use cpal::traits::{DeviceTrait, StreamTrait};
        use syrinx_proto::{downmix_to_mono, resample_to_16k};

        let name = name.to_string();
        let stop = on_stream_thread(move || {
            // Found here rather than by the caller because the lookup panics
            // on Windows instead of returning: it goes through
            // `CoCreateInstance(..).unwrap()` and `.Item(i).unwrap()`. On the
            // caller's thread that panic is the daemon's session thread dying
            // mid-start, which leaves the session reading Listening for ever
            // with no error to show. Here it is only a failed open.
            let device = crate::cpal_backend::find_device(&name, loopback)?;
            // A loopback source is a *render* endpoint being read. cpal builds
            // that stream correctly -- it sets AUDCLNT_STREAMFLAGS_LOOPBACK for
            // an input stream on a render device -- but it refuses to describe
            // one: `default_input_config` answers "Device does not support
            // input" for anything that is not a capture endpoint. The format to
            // ask for is the one the device renders in, so query the output
            // side and open it for input.
            //
            // This query panics on Windows too, at `.expect("could not query
            // IMMDevice interface for IMMEndpoint")`, for the same reason it
            // is here and not up there.
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
                        let f: Vec<f32> = d.iter().map(|s| (*s as f32 - 128.0) / 128.0).collect();
                        let _ = tx.try_send(resample_to_16k(&downmix_to_mono(&f, channels), rate));
                    },
                    err_fn,
                    None,
                ),
                other => {
                    anyhow::bail!("this device captures in {other:?}, which syrinx cannot convert")
                }
            };
            let stream = stream.context("building the input stream")?;
            stream.play().context("starting the input stream")?;
            Ok(stream)
        })?;

        Ok(Self { _stop: stop })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    /// Wait for `f`, or give up.
    ///
    /// Deadline-bounded rather than left to spin: what these tests check is
    /// that a thread reaches an end state, and one that never does would hang
    /// the run instead of failing it.
    fn within(deadline: Duration, f: impl Fn() -> bool) -> bool {
        let until = Instant::now() + deadline;
        while Instant::now() < until {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        f()
    }

    #[test]
    fn a_stream_that_cannot_be_built_reports_the_error_rather_than_ok() {
        // The fault this replaces: the failure was logged without its value
        // and `Ok` was returned anyway, so a session went on to its handshake
        // and reported Listening while capturing nothing.
        let e =
            on_stream_thread(|| -> Result<()> { anyhow::bail!("Device does not support input") })
                .expect_err("a stream that never built must not report success");
        assert!(
            format!("{e:#}").contains("does not support input"),
            "the real error has to survive: {e:#}"
        );
    }

    #[test]
    fn a_stream_that_starts_returns_a_handle_that_stops_it() {
        struct Stopper(Arc<AtomicBool>);
        impl Drop for Stopper {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let stopped = Arc::new(AtomicBool::new(false));
        let flag = stopped.clone();
        let handle = on_stream_thread(move || Ok(Stopper(flag))).unwrap();
        assert!(
            !stopped.load(Ordering::SeqCst),
            "it stopped before it was asked"
        );

        drop(handle);
        assert!(
            within(Duration::from_secs(5), || stopped.load(Ordering::SeqCst)),
            "dropping the handle must stop the stream"
        );
    }

    #[test]
    fn a_backend_that_panics_reaches_the_caller_as_an_error_with_its_message() {
        // cpal's Windows backend panics rather than erroring in several
        // places -- device lookup and the config query among them, which is
        // why both now run inside this closure. A caller that only learned
        // "it panicked" would have nothing to show; the payload is the
        // diagnosis. The panic message this prints is the test doing its job.
        let e = on_stream_thread(|| -> Result<()> {
            panic!("could not query IMMDevice interface for IMMEndpoint")
        })
        .expect_err("a backend that panicked must not report success");
        assert!(
            format!("{e:#}").contains("could not query IMMDevice interface"),
            "got: {e:#}"
        );
    }
}
