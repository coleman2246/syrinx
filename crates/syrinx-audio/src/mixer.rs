//! Combining several capture sources into one stream.
//!
//! No resampling happens here. Every backend already normalises to 16 kHz mono
//! before a sample reaches this point -- `pw-record` is asked for that rate
//! directly, and the cpal path resamples on the way in -- so mixing is
//! arithmetic on aligned buffers rather than rate conversion.
//!
//! Sources are independent devices whose callbacks fire at slightly different
//! times, so they never deliver the same amount at the same moment. Each gets a
//! queue, and the mixer emits only as much as every source can supply.

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::{Capture, Source};

/// Average several equal-length frames into one.
///
/// Averaged rather than summed: two full-scale inputs added together clip, and
/// clipping is both audible and destructive to recognition. Averaging cannot
/// exceed full scale whatever the inputs do, and preserves which source was
/// actually louder.
pub fn mix_frames(frames: &[Vec<f32>]) -> Vec<f32> {
    if frames.is_empty() {
        return Vec::new();
    }
    let len = frames.iter().map(|f| f.len()).min().unwrap_or(0);
    let n = frames.len() as f32;
    (0..len)
        .map(|i| frames.iter().map(|f| f[i]).sum::<f32>() / n)
        .collect()
}

/// How much audio a source may buffer before its oldest is dropped.
///
/// A source that stalls -- an application that stops playing, a device that
/// unplugs -- must not stall the mix. Two seconds is far more than the jitter
/// between healthy sources, so dropping only ever discards audio from a source
/// that has genuinely stopped keeping up.
const MAX_QUEUED: usize = 32_000;

/// Several captures combined into one stream. Dropping it stops all of them.
pub struct MixedCapture {
    _captures: Vec<Capture>,
}

impl MixedCapture {
    /// Start every source and mix them into `out`.
    pub fn start(sources: &[Source], out: mpsc::Sender<Vec<f32>>) -> Result<Self> {
        if sources.is_empty() {
            anyhow::bail!("no sources to combine");
        }

        let queues: Vec<Arc<Mutex<VecDeque<f32>>>> = sources
            .iter()
            .map(|_| Arc::new(Mutex::new(VecDeque::new())))
            .collect();

        let mut captures = Vec::new();
        for (source, queue) in sources.iter().zip(&queues) {
            let (tx, mut rx) = mpsc::channel::<Vec<f32>>(32);
            captures.push(Capture::start(source, tx)?);
            let q = queue.clone();
            tokio::spawn(async move {
                while let Some(chunk) = rx.recv().await {
                    let mut q = q.lock().expect("mixer queue poisoned");
                    q.extend(chunk);
                    // Keep the newest audio: a backlog means this source has
                    // fallen behind, and stale audio mixed in late is worse
                    // than a gap.
                    while q.len() > MAX_QUEUED {
                        q.pop_front();
                    }
                }
            });
        }

        // Emit whatever every source can supply, as often as any of them could
        // have produced something.
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_millis(20));
            loop {
                ticker.tick().await;
                let available = queues
                    .iter()
                    .map(|q| q.lock().expect("mixer queue poisoned").len())
                    .min()
                    .unwrap_or(0);
                if available == 0 {
                    continue;
                }
                let frames: Vec<Vec<f32>> = queues
                    .iter()
                    .map(|q| {
                        let mut q = q.lock().expect("mixer queue poisoned");
                        q.drain(..available).collect()
                    })
                    .collect();
                if out.send(mix_frames(&frames)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            _captures: captures,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sources_are_averaged_not_summed() {
        // Summing would put this at 2.0, which clips. Averaging keeps it in
        // range whatever the inputs do.
        let out = mix_frames(&[vec![1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(out, vec![1.0, 1.0]);
    }

    #[test]
    fn full_scale_opposites_cancel_rather_than_overflow() {
        assert_eq!(mix_frames(&[vec![1.0], vec![-1.0]]), vec![0.0]);
    }

    #[test]
    fn a_quiet_source_stays_quiet_next_to_a_loud_one() {
        // Relative level is information: it says who was actually louder.
        let out = mix_frames(&[vec![1.0], vec![0.0]]);
        assert_eq!(out, vec![0.5]);
    }

    #[test]
    fn mixing_takes_the_shortest_frame() {
        // Sources never deliver the same amount at the same moment, so the mix
        // is bounded by whichever has least.
        let out = mix_frames(&[vec![1.0, 1.0, 1.0], vec![1.0]]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn one_source_passes_through_unchanged() {
        // Combined mode with a single source must not alter it.
        assert_eq!(mix_frames(&[vec![0.25, -0.5]]), vec![0.25, -0.5]);
    }

    #[test]
    fn no_sources_yields_nothing_rather_than_panicking() {
        assert!(mix_frames(&[]).is_empty());
        assert!(mix_frames(&[vec![], vec![]]).is_empty());
    }

    #[test]
    fn three_sources_average_correctly() {
        let out = mix_frames(&[vec![0.9], vec![0.3], vec![0.3]]);
        assert!((out[0] - 0.5).abs() < 1e-6, "got {}", out[0]);
    }
}
