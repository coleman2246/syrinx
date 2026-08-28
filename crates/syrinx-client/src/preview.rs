//! Live level metering for the currently selected source.
//!
//! Runs while idle so a source can be checked before committing to a session:
//! it answers "is anything arriving from this device" without involving the
//! server or producing a transcript.
//!
//! Owned by the daemon rather than a front-end, because the daemon owns the
//! source selection and only one process can sensibly hold a capture open.

use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use syrinx_audio::meter::{self, BANDS};
use syrinx_audio::mixer::{STARVE_AFTER, SourceHealth};
use syrinx_audio::{Capture, Source};
use tokio::sync::{Notify, mpsc};

/// How a preview's device is getting on.
#[derive(Debug, Clone, PartialEq)]
pub enum Opening {
    /// Still opening. A WASAPI endpoint can take a moment, and a wedged one
    /// takes until the capture's own open timeout gives up on it.
    Pending,
    /// Open, and feeding the levels below.
    Open,
    /// The device would not open, and why.
    Failed(String),
}

/// How long a caller with nothing else to do waits for a device.
///
/// Longer than the capture's own open timeout, so what comes back is the real
/// reason rather than this having given up first.
const WAIT_FOR_OPEN: Duration = Duration::from_secs(15);

/// A running preview. Dropping it stops the capture.
pub struct Preview {
    levels: Arc<Mutex<[f32; BANDS]>>,
    rms: Arc<Mutex<f32>>,
    /// When a chunk last arrived, so a source delivering nothing can say so
    /// rather than reading as a source delivering silence.
    last_chunk: Arc<Mutex<Instant>>,
    /// What became of the attempt to open the device, once it is known.
    opening: Arc<Mutex<Opening>>,
    stop: Arc<Notify>,
    /// Which source this is metering, so the daemon can tell when to re-point.
    pub source_key: String,
    /// `Source::short_label`, so a meter row can name itself.
    pub label: String,
}

/// Fold arriving audio into the published levels until told to stop.
///
/// Split out from [`Preview::start`] so it can be driven without a device: the
/// fault it exists to prevent is a source that never delivers a chunk, and no
/// real device can be asked to behave that way on demand.
async fn meter_loop(
    mut rx: mpsc::Receiver<Vec<f32>>,
    stop: Arc<Notify>,
    levels: Arc<Mutex<[f32; BANDS]>>,
    rms: Arc<Mutex<f32>>,
    last_chunk: Arc<Mutex<Instant>>,
) {
    // A rolling window kept just long enough for one FFT frame. Holding more
    // would make the meter lag behind the sound.
    let mut window: Vec<f32> = Vec::with_capacity(4096);
    loop {
        let chunk = tokio::select! {
            // The stop signal is selected on rather than checked after the
            // await, because a source that delivers nothing never returns from
            // the await at all: `recv` gives neither Some nor None, since the
            // sender lives in the capture callback, the capture is held by
            // this task, and this task is the one waiting. A silent WASAPI
            // loopback -- an output with nothing playing -- is exactly that
            // source, and would otherwise pin its capture client, its thread
            // and its runtime until the process died.
            _ = stop.notified() => break,
            chunk = rx.recv() => match chunk {
                Some(c) => c,
                None => break,
            },
        };
        *last_chunk.lock().expect("preview clock poisoned") = Instant::now();
        window.extend_from_slice(&chunk);
        if window.len() > 4096 {
            let excess = window.len() - 4096;
            window.drain(..excess);
        }
        *levels.lock().expect("levels lock poisoned") =
            meter::spectrum(&window, syrinx_proto::SAMPLE_RATE);
        *rms.lock().expect("rms lock poisoned") = meter::rms(&chunk);
    }
}

impl Preview {
    /// Start metering `source`. Returns at once.
    ///
    /// The device is opened on the preview's own thread, and what became of
    /// that shows up in [`opening`](Self::opening) when it is known. Waiting
    /// for it here instead stalled whichever loop called it -- and the daemon
    /// calls it inline in the loop that answers the GUI, which gives every
    /// request five seconds before replying "daemon did not answer in time".
    /// One endpoint slow to open therefore made Start, Stop and the source
    /// picker fail and froze the tray, for as long as the open took, once per
    /// selected source.
    ///
    /// Returning a `Preview` whatever happens is the other half of that. The
    /// old error paths returned before anything owned the stop signal, so a
    /// thread still inside `Capture::start` went on to acquire a capture that
    /// nobody could ever release -- a silent WASAPI loopback pinned for the
    /// life of the process. Here the caller always holds the handle that
    /// stops it.
    pub fn start(source: &Source) -> Self {
        let levels = Arc::new(Mutex::new([0.0f32; BANDS]));
        let rms_v = Arc::new(Mutex::new(0.0f32));
        let last_chunk = Arc::new(Mutex::new(Instant::now()));
        let opening = Arc::new(Mutex::new(Opening::Pending));
        let stop = Arc::new(Notify::new());

        let (l, r, c, s, o) = (
            levels.clone(),
            rms_v.clone(),
            last_chunk.clone(),
            stop.clone(),
            opening.clone(),
        );
        let src = source.clone();

        // Everything happens inside the runtime. `Capture::start` spawns a
        // `tokio::process::Command`, which panics with "there is no reactor
        // running" if called from a plain thread -- and both the daemon loop
        // and the CLI are synchronous, so the capture cannot be created before
        // the runtime exists.
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    *o.lock().expect("preview state poisoned") =
                        Opening::Failed(format!("building a runtime: {e}"));
                    return;
                }
            };
            rt.block_on(async move {
                let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
                let capture = match Capture::start(&src, tx) {
                    Ok(cap) => {
                        *o.lock().expect("preview state poisoned") = Opening::Open;
                        cap
                    }
                    Err(e) => {
                        *o.lock().expect("preview state poisoned") =
                            Opening::Failed(format!("{e:#}"));
                        return;
                    }
                };
                meter_loop(rx, s, l, r, c).await;
                // Explicit because it is the whole point: this handle is what
                // was being held open, and the device is not released until it
                // goes.
                drop(capture);
            });
        });

        Self {
            levels,
            rms: rms_v,
            last_chunk,
            opening,
            stop,
            source_key: source.stable_key(),
            label: source.short_label(),
        }
    }

    /// What became of the attempt to open the device.
    pub fn opening(&self) -> Opening {
        self.opening.lock().expect("preview state poisoned").clone()
    }

    /// Block until the device opens, or say why it did not.
    ///
    /// For a caller with nothing else to do -- `syrinx meter` is one long wait
    /// on this preview and nothing else. Anything with a loop to keep turning
    /// polls [`opening`](Self::opening) instead.
    pub fn wait_until_open(&self) -> Result<()> {
        let deadline = Instant::now() + WAIT_FOR_OPEN;
        loop {
            match self.opening() {
                Opening::Open => return Ok(()),
                Opening::Failed(e) => anyhow::bail!(e),
                Opening::Pending if Instant::now() >= deadline => {
                    anyhow::bail!("the audio device did not open within {WAIT_FOR_OPEN:?}")
                }
                Opening::Pending => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    }

    pub fn levels(&self) -> [f32; BANDS] {
        *self.levels.lock().expect("levels lock poisoned")
    }

    pub fn rms(&self) -> f32 {
        *self.rms.lock().expect("rms lock poisoned")
    }

    /// Nothing has arrived from this source for [`STARVE_AFTER`].
    ///
    /// Distinct from an rms of zero, which is what a live microphone in a
    /// quiet room reads. This says the device has delivered no audio at all,
    /// which is what an idle Windows loopback does.
    pub fn silent(&self) -> bool {
        self.last_chunk
            .lock()
            .expect("preview clock poisoned")
            .elapsed()
            >= STARVE_AFTER
    }

    /// This source as a meter row.
    pub fn health(&self) -> SourceHealth {
        SourceHealth {
            label: self.label.clone(),
            rms: self.rms(),
            silent: self.silent(),
            // A preview holds one capture with no queue behind it, so there
            // is nothing to trim; and a source whose device would not open
            // has no `Preview` to ask, so its row comes from the daemon,
            // which is what knows the source was selected at all.
            dropped: 0,
            error: None,
        }
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        // `notify_one` rather than `notify_waiters`: a stop that arrives
        // before the task has reached its first `notified()` has to be
        // remembered, not dropped on the floor.
        self.stop.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The three things a meter publishes into, as `meter_loop` takes them.
    type Published = (
        Arc<Mutex<[f32; BANDS]>>,
        Arc<Mutex<f32>>,
        Arc<Mutex<Instant>>,
    );

    fn published() -> Published {
        (
            Arc::new(Mutex::new([0.0f32; BANDS])),
            Arc::new(Mutex::new(0.0f32)),
            Arc::new(Mutex::new(Instant::now())),
        )
    }

    #[tokio::test]
    async fn the_meter_lets_go_of_a_source_that_never_produced_a_chunk() {
        // The leak. `tx` stands in for the capture callback, which the capture
        // keeps alive and the task keeps alive in turn -- so `recv` never
        // returns and the old loop never looked at its stop flag. Held here
        // for the whole test, exactly as a silent WASAPI loopback holds it.
        let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
        let (levels, rms, clock) = published();
        let stop = Arc::new(Notify::new());
        let task = tokio::spawn(meter_loop(rx, stop.clone(), levels, rms, clock));

        stop.notify_one();

        // Deadline-bounded: the regression is a task that never ends, which
        // without one hangs the run rather than failing it.
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("the meter held its capture after being told to stop")
            .expect("the meter task panicked");
        drop(tx);
    }

    #[tokio::test]
    async fn a_stop_that_arrives_first_is_not_lost() {
        // The daemon drops a preview from a synchronous loop, which can happen
        // before the task has reached its first await. A signal that only woke
        // a waiter already waiting would leak exactly as before.
        let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
        let (levels, rms, clock) = published();
        let stop = Arc::new(Notify::new());
        stop.notify_one();

        tokio::time::timeout(
            Duration::from_secs(5),
            meter_loop(rx, stop, levels, rms, clock),
        )
        .await
        .expect("a stop signalled before the first await was dropped");
        drop(tx);
    }

    #[tokio::test]
    async fn the_meter_lets_go_when_its_source_ends() {
        // The other exit: the device went away rather than the user leaving.
        let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
        let (levels, rms, clock) = published();
        drop(tx);

        tokio::time::timeout(
            Duration::from_secs(5),
            meter_loop(rx, Arc::new(Notify::new()), levels, rms, clock),
        )
        .await
        .expect("the meter outlived its source");
    }

    /// A source naming a device that does not exist, so the open fails.
    fn missing_device() -> Source {
        Source {
            target: syrinx_audio::SourceTarget::CpalDevice {
                name: "no such device exists anywhere".into(),
                loopback: false,
            },
            name: "no such device exists anywhere".into(),
            kind: syrinx_audio::SourceKind::Microphone,
            detail: None,
            stable_name: None,
            sink_description: None,
        }
    }

    #[test]
    fn starting_a_preview_does_not_wait_for_the_device() {
        // The daemon starts these inline in the loop that answers the GUI,
        // and that loop gives every request five seconds before it replies
        // "daemon did not answer in time". A start that waited on the device
        // therefore made Start, Stop and the source picker fail and froze the
        // tray for as long as the open took -- once per selected source.
        let started = std::time::Instant::now();
        let preview = Preview::start(&missing_device());
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "starting a meter took {:?}; the daemon loop cannot afford that",
            started.elapsed()
        );
        assert_eq!(preview.opening(), Opening::Pending);
    }

    #[test]
    fn a_device_that_will_not_open_says_so_rather_than_reading_zero() {
        // Silently metering nothing is indistinguishable from a dead source,
        // which is the question the meter exists to answer.
        let preview = Preview::start(&missing_device());
        let e = preview
            .wait_until_open()
            .expect_err("a device that does not exist must not report success");
        assert!(
            format!("{e:#}").contains("no such device exists anywhere"),
            "got: {e:#}"
        );
        assert!(matches!(preview.opening(), Opening::Failed(_)));
    }

    #[tokio::test]
    async fn a_chunk_that_arrives_moves_the_level() {
        // The loop still has to do its job; letting go is only half of it.
        let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
        let (levels, rms, clock) = published();
        let stop = Arc::new(Notify::new());
        let task = tokio::spawn(meter_loop(
            rx,
            stop.clone(),
            levels.clone(),
            rms.clone(),
            clock.clone(),
        ));

        let tone: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.2).sin() * 0.5).collect();
        tx.send(tone).await.unwrap();

        let moved = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if *rms.lock().unwrap() > 0.0 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(moved.is_ok(), "the meter never registered the audio");
        assert!(
            levels.lock().unwrap().iter().any(|b| *b > 0.0),
            "the spectrum stayed flat"
        );

        stop.notify_one();
        let _ = tokio::time::timeout(Duration::from_secs(5), task).await;
    }
}
