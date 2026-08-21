//! Live level metering for the currently selected source.
//!
//! Runs while idle so a source can be checked before committing to a session:
//! it answers "is anything arriving from this device" without involving the
//! server or producing a transcript.
//!
//! Owned by the daemon rather than a front-end, because the daemon owns the
//! source selection and only one process can sensibly hold a capture open.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use syrinx_audio::meter::{self, BANDS};
use syrinx_audio::{Capture, Source};

/// A running preview. Dropping it stops the capture.
pub struct Preview {
    levels: Arc<Mutex<[f32; BANDS]>>,
    rms: Arc<Mutex<f32>>,
    stop: Arc<AtomicBool>,
    /// Which source this is metering, so the daemon can tell when to re-point.
    pub source_key: String,
}

impl Preview {
    pub fn start(source: &Source) -> Result<Self> {
        let levels = Arc::new(Mutex::new([0.0f32; BANDS]));
        let rms_v = Arc::new(Mutex::new(0.0f32));
        let stop = Arc::new(AtomicBool::new(false));

        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let (l, r, s) = (levels.clone(), rms_v.clone(), stop.clone());
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
                    let _ = ready_tx.send(Err(format!("building a runtime: {e}")));
                    return;
                }
            };
            rt.block_on(async move {
                let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<f32>>(32);
                // Held for the life of this task; dropping it stops capture.
                let _capture = match Capture::start(&src, tx) {
                    Ok(c) => {
                        let _ = ready_tx.send(Ok(()));
                        c
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("{e:#}")));
                        return;
                    }
                };

                // A rolling window kept just long enough for one FFT frame.
                // Holding more would make the meter lag behind the sound.
                let mut window: Vec<f32> = Vec::with_capacity(4096);
                while let Some(chunk) = rx.recv().await {
                    if s.load(Ordering::Relaxed) {
                        break;
                    }
                    window.extend_from_slice(&chunk);
                    if window.len() > 4096 {
                        let excess = window.len() - 4096;
                        window.drain(..excess);
                    }
                    *l.lock().expect("levels lock poisoned") =
                        meter::spectrum(&window, syrinx_proto::SAMPLE_RATE);
                    *r.lock().expect("rms lock poisoned") = meter::rms(&chunk);
                }
            });
        });

        // Surface a failed capture here rather than returning a handle that
        // silently reads zero, which is indistinguishable from a dead source.
        match ready_rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => anyhow::bail!(e),
            Err(e) => return Err(e).context("starting the level meter"),
        }

        Ok(Self {
            levels,
            rms: rms_v,
            stop,
            source_key: source.stable_key(),
        })
    }

    pub fn levels(&self) -> [f32; BANDS] {
        *self.levels.lock().expect("levels lock poisoned")
    }

    pub fn rms(&self) -> f32 {
        *self.rms.lock().expect("rms lock poisoned")
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
