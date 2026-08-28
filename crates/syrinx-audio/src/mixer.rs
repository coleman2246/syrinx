//! Combining several capture sources into one stream.
//!
//! No resampling happens here. Every backend already normalises to 16 kHz mono
//! before a sample reaches this point -- `pw-record` is asked for that rate
//! directly, and the cpal path resamples on the way in -- so mixing is
//! arithmetic on aligned buffers rather than rate conversion.
//!
//! Emission is driven by the clock, not by what the sources have in common.
//! Every 20 ms a frame goes out whatever arrived, and a source short of a full
//! frame contributes silence for the remainder. The mixer used to emit only as
//! much as *every* source could supply, which meant a source producing nothing
//! emitted nothing for any of them -- and a WASAPI loopback on an output with
//! nothing playing produces exactly nothing, so ticking the system-audio box
//! silenced the microphone beside it. Silence is the correct rendering of an
//! idle output; the clock is what makes it expressible.
//!
//! Gain stays at 1/N whether or not a source contributed. Averaging over only
//! the sources that had data would move the output 6 dB the moment one woke or
//! went quiet, mid-utterance, which costs the recogniser more than a constant
//! halving does.
//!
//! Sources are independent devices on independent clocks, so they never
//! deliver the same amount at the same moment. Each gets a short queue, trimmed
//! from the front so the mix stays near the present.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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

/// How often the mix emits, and therefore how much latency it adds.
const TICK: Duration = Duration::from_millis(20);

/// Samples emitted per tick: 20 ms at 16 kHz.
///
/// Derived rather than written as 320, so it cannot drift from the rate every
/// backend normalises to.
const FRAME: usize = syrinx_proto::SAMPLE_RATE as usize / 50;

/// How much audio a source may buffer before its oldest is dropped.
///
/// 200 ms, where this was two seconds while the mix waited for its slowest
/// source. With the clock driving emission, a queue that keeps growing means
/// that source runs fast relative to the tick -- which happens for real:
/// `resample_to_16k` decimates with `chunks_exact`, silently dropping up to
/// `factor - 1` samples per callback whenever the device's period is not a
/// multiple of the ratio, and two devices on independent crystals drift
/// regardless. Trimming here keeps the sources loosely aligned instead of
/// letting one run two seconds ahead of the other.
const MAX_QUEUED: usize = FRAME * 10;

/// How long a source may contribute nothing before it is reported as silent.
pub const STARVE_AFTER: Duration = Duration::from_secs(5);

/// How often one source may report that it is being trimmed.
const TRIM_LOG_EVERY: Duration = Duration::from_secs(5);

/// What one source is contributing to the mix.
///
/// The wording is *silent*, not *failed*. A Windows loopback on an output with
/// nothing playing delivers no packets at all, and that is correct behaviour
/// rather than an error -- a red light for it would be a lie. What the user
/// cannot otherwise discover is which of their sources is carrying anything,
/// because everything downstream of here has already been mixed into one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceHealth {
    /// `Source::short_label`, so a meter row can name itself.
    pub label: String,
    /// Level of what this source contributed last tick, 0.0 to 1.0.
    pub rms: f32,
    /// Nothing has arrived from this source for [`STARVE_AFTER`].
    pub silent: bool,
}

/// Lets a repeated message through at most once per interval.
///
/// A source that runs fast is trimmed on every callback, which is dozens of
/// times a second; logging each one would bury everything else in the file
/// while saying the same thing over and over.
struct Rate {
    every: Duration,
    last: Option<Instant>,
}

impl Rate {
    fn new(every: Duration) -> Self {
        Self { every, last: None }
    }

    fn allow(&mut self, now: Instant) -> bool {
        match self.last {
            Some(t) if now.duration_since(t) < self.every => false,
            _ => {
                self.last = Some(now);
                true
            }
        }
    }
}

/// One source's buffered audio, and what the mixer knows about it.
struct SourceQueue {
    label: String,
    samples: VecDeque<f32>,
    /// When something last arrived. Starts at creation, so a source is given
    /// [`STARVE_AFTER`] to say anything before it is called silent.
    last_data: Instant,
    /// Level of the last frame taken from this source.
    rms: f32,
    /// Samples discarded because this source ran ahead of the clock.
    dropped: u64,
    /// The capture behind this queue has ended and will send nothing more.
    finished: bool,
}

impl SourceQueue {
    fn new(label: String) -> Self {
        Self {
            label,
            samples: VecDeque::new(),
            last_data: Instant::now(),
            rms: 0.0,
            dropped: 0,
            finished: false,
        }
    }

    /// Take a chunk from the capture, returning how much had to be discarded.
    fn push(&mut self, chunk: Vec<f32>) -> usize {
        if chunk.is_empty() {
            return 0;
        }
        // Non-empty, not non-silent: a live microphone in a quiet room sends
        // zeroes, and that is a source contributing, not a source starved.
        self.last_data = Instant::now();
        self.samples.extend(chunk);
        let excess = self.samples.len().saturating_sub(MAX_QUEUED);
        if excess > 0 {
            // Keep the newest: a backlog means this source has run ahead, and
            // stale audio mixed in late is worse than a gap.
            self.samples.drain(..excess);
            self.dropped += excess as u64;
        }
        excess
    }

    /// One frame's worth of this source: what it has, padded with silence.
    fn take_frame(&mut self) -> Vec<f32> {
        let n = self.samples.len().min(FRAME);
        let mut frame: Vec<f32> = self.samples.drain(..n).collect();
        // Measured over what actually arrived rather than the padding, so a
        // source delivering half a frame of speech does not read as half as
        // loud as one delivering a whole frame of it.
        self.rms = if n == 0 {
            0.0
        } else {
            crate::meter::rms(&frame)
        };
        frame.resize(FRAME, 0.0);
        frame
    }

    fn silent_at(&self, now: Instant) -> bool {
        now.duration_since(self.last_data) >= STARVE_AFTER
    }

    /// Nothing left to emit and nothing more coming.
    fn spent(&self) -> bool {
        self.finished && self.samples.is_empty()
    }
}

/// A read-only view of what each source is contributing.
///
/// Cloned out of [`MixedCapture`] before it is boxed away behind `dyn Any`, so
/// a session can keep reading the per-source levels it no longer has the
/// capture itself to ask.
#[derive(Clone)]
pub struct Health(Vec<Arc<Mutex<SourceQueue>>>);

impl Health {
    pub fn read(&self) -> Vec<SourceHealth> {
        let now = Instant::now();
        self.0
            .iter()
            .map(|q| {
                let q = q.lock().expect("mixer queue poisoned");
                SourceHealth {
                    label: q.label.clone(),
                    rms: q.rms,
                    silent: q.silent_at(now),
                }
            })
            .collect()
    }
}

/// Drive the mix from `inputs` onto the clock, into `out`.
///
/// Split out from [`MixedCapture::start`] so it can be driven from synthetic
/// senders. Every fault this exists to prevent is about what happens when a
/// device produces nothing, and no test can arrange that through a real one.
fn spawn_mix(
    labels: &[String],
    inputs: Vec<mpsc::Receiver<Vec<f32>>>,
    out: mpsc::Sender<Vec<f32>>,
) -> Health {
    let queues: Vec<Arc<Mutex<SourceQueue>>> = labels
        .iter()
        .map(|l| Arc::new(Mutex::new(SourceQueue::new(l.clone()))))
        .collect();

    for (mut rx, queue) in inputs.into_iter().zip(&queues) {
        let q = queue.clone();
        tokio::spawn(async move {
            let mut trim_log = Rate::new(TRIM_LOG_EVERY);
            while let Some(chunk) = rx.recv().await {
                let (dropped, label) = {
                    let mut q = q.lock().expect("mixer queue poisoned");
                    (q.push(chunk), q.label.clone())
                };
                if dropped > 0 && trim_log.allow(Instant::now()) {
                    tracing::warn!(
                        "{label} is running ahead of the mix; trimming its queue \
                         (dropped {dropped} samples this time)"
                    );
                }
            }
            q.lock().expect("mixer queue poisoned").finished = true;
        });
    }

    let mix = queues.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(TICK);
        // Delay, not Burst: a consumer that stalls must not be repaid with a
        // flood of catch-up frames drawn from queues that only hold 200 ms,
        // which would be a burst of silence rather than the audio it missed.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let mut frames = Vec::with_capacity(mix.len());
            let mut spent = true;
            for q in &mix {
                let mut q = q.lock().expect("mixer queue poisoned");
                frames.push(q.take_frame());
                spent &= q.spent();
            }
            // Every capture has ended and been drained. Dropping `out` here is
            // what tells the session its audio is over; ticking silence for
            // ever would leave it listening to a dead device.
            if spent {
                break;
            }
            if out.send(mix_frames(&frames)).await.is_err() {
                break;
            }
        }
    });

    Health(queues)
}

/// Several captures combined into one stream. Dropping it stops all of them.
pub struct MixedCapture {
    _captures: Vec<Capture>,
    health: Health,
}

impl MixedCapture {
    /// Start every source and mix them into `out`.
    pub fn start(sources: &[Source], out: mpsc::Sender<Vec<f32>>) -> Result<Self> {
        if sources.is_empty() {
            anyhow::bail!("no sources to combine");
        }

        let labels: Vec<String> = sources.iter().map(|s| s.short_label()).collect();
        let mut captures = Vec::new();
        let mut inputs = Vec::new();
        for source in sources {
            let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
            captures.push(Capture::start(source, tx)?);
            inputs.push(rx);
        }

        Ok(Self {
            health: spawn_mix(&labels, inputs, out),
            _captures: captures,
        })
    }

    pub fn health(&self) -> Health {
        self.health.clone()
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

    /// Two synthetic sources and somewhere to listen.
    ///
    /// Nothing here opens a device: the faults this module exists to prevent
    /// are all about a device that produces nothing, which is precisely what a
    /// real one cannot be asked to do on demand.
    #[allow(clippy::type_complexity)]
    fn two_sources() -> (
        mpsc::Sender<Vec<f32>>,
        mpsc::Sender<Vec<f32>>,
        mpsc::Receiver<Vec<f32>>,
        Health,
    ) {
        let (a_tx, a_rx) = mpsc::channel::<Vec<f32>>(32);
        let (b_tx, b_rx) = mpsc::channel::<Vec<f32>>(32);
        let (out_tx, out_rx) = mpsc::channel::<Vec<f32>>(64);
        let health = spawn_mix(
            &["A".to_string(), "B".to_string()],
            vec![a_rx, b_rx],
            out_tx,
        );
        (a_tx, b_tx, out_rx, health)
    }

    /// Read frames until one carries something, or give up.
    ///
    /// Deadline-bounded because the fault every one of these guards is a mixer
    /// that emits *nothing*, and an unbounded read of a channel that never
    /// produces hangs rather than fails. A hang in CI is a timeout with no
    /// message on it; the deadline makes the same regression a named
    /// assertion.
    async fn first_audible(rx: &mut mpsc::Receiver<Vec<f32>>) -> Vec<f32> {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let frame = rx.recv().await.expect("the mixer stopped emitting");
                if frame.iter().any(|s| *s != 0.0) {
                    return frame;
                }
            }
        })
        .await
        .expect("no audio reached the mix before the deadline")
    }

    #[tokio::test]
    async fn a_source_that_never_produces_does_not_silence_the_others() {
        // The whole bug. A WASAPI loopback on an output with nothing playing
        // delivers no packets at all, and the mix used to wait for it -- so
        // ticking the system-audio box silenced the microphone beside it.
        let (live, _idle, mut out, _health) = two_sources();
        live.send(vec![1.0; FRAME]).await.unwrap();

        let heard = first_audible(&mut out).await;
        assert_eq!(heard.len(), FRAME);
        // Halved by the silent source's contribution: the gain stays at 1/N so
        // the level cannot jump 6 dB mid-utterance when a source wakes up.
        assert!((heard[0] - 0.5).abs() < 1e-6, "got {}", heard[0]);
    }

    #[tokio::test]
    async fn a_source_that_stops_mid_stream_does_not_silence_the_others() {
        // The same fault arriving later: an application stops playing, or a
        // loopback goes quiet, an hour into a meeting.
        let (a, b, mut out, _health) = two_sources();
        a.send(vec![1.0; FRAME]).await.unwrap();
        b.send(vec![1.0; FRAME]).await.unwrap();
        assert!((first_audible(&mut out).await[0] - 1.0).abs() < 1e-6);

        // B goes quiet for good; A keeps talking and must keep being heard.
        for _ in 0..5 {
            a.send(vec![1.0; FRAME]).await.unwrap();
        }
        assert!((first_audible(&mut out).await[0] - 0.5).abs() < 1e-6);

        // And B coming back needs no recovery step of its own.
        a.send(vec![1.0; FRAME]).await.unwrap();
        b.send(vec![1.0; FRAME]).await.unwrap();
        let back = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let f = out.recv().await.expect("the mixer stopped emitting");
                if (f[0] - 1.0).abs() < 1e-6 {
                    return f;
                }
            }
        })
        .await
        .expect("the returning source never rejoined the mix");
        assert_eq!(back.len(), FRAME);
    }

    #[tokio::test]
    async fn emission_is_paced_by_the_clock_not_by_what_arrived() {
        // Nothing is ever sent, so under the old rendezvous this produced
        // nothing at all. Under the clock it produces frames of silence, which
        // is the honest rendering of two idle devices.
        let (_a, _b, mut out, _health) = two_sources();

        let started = Instant::now();
        const FRAMES: usize = 5;
        for i in 0..FRAMES {
            let frame = tokio::time::timeout(Duration::from_secs(5), out.recv())
                .await
                .expect("the clock stopped driving the mix")
                .expect("the mixer stopped emitting");
            assert_eq!(frame.len(), FRAME, "frame {i} was not a whole frame");
            assert!(frame.iter().all(|s| *s == 0.0), "frame {i} invented audio");
        }
        // Paced, not spun: the first tick fires at once and the rest are
        // 20 ms apart, so five of them cannot arrive instantly. Bounded well
        // below the exact 80 ms, since the point is the pacing rather than the
        // timer's precision on a loaded machine.
        assert!(
            started.elapsed() >= Duration::from_millis(60),
            "{FRAMES} frames arrived in {:?}; the mix is spinning, not ticking",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn the_mix_ends_when_every_source_has_ended() {
        // Otherwise a session whose devices all died would listen to a stream
        // of manufactured silence for ever instead of noticing.
        let (a, b, mut out, _health) = two_sources();
        a.send(vec![0.5; FRAME]).await.unwrap();
        drop(a);
        drop(b);

        let ended = tokio::time::timeout(Duration::from_secs(5), async {
            while out.recv().await.is_some() {}
        })
        .await;
        assert!(
            ended.is_ok(),
            "the mix kept emitting after every source ended"
        );
    }

    #[tokio::test]
    async fn each_source_reports_its_own_level() {
        // The point of per-source health: a running session at 0% and one at
        // 40% are indistinguishable once everything has been mixed into one.
        let (live, _idle, mut out, health) = two_sources();
        for _ in 0..4 {
            live.send(vec![1.0; FRAME]).await.unwrap();
        }
        first_audible(&mut out).await;

        let rows = health.read();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "A");
        assert!(rows[0].rms > 0.5, "the live source reads {}", rows[0].rms);
        assert_eq!(rows[1].rms, 0.0, "the idle source invented a level");
    }

    #[test]
    fn a_fast_source_is_trimmed_rather_than_allowed_to_drift() {
        // `resample_to_16k` drops up to `factor - 1` samples per callback and
        // two devices run on independent crystals, so one source genuinely
        // does outrun the other. Two seconds of backlog would mix in audio
        // two seconds stale; 200 ms keeps them loosely aligned.
        let mut q = SourceQueue::new("fast".into());
        for round in 0..10 {
            q.push(vec![round as f32; MAX_QUEUED / 2]);
        }
        assert_eq!(q.samples.len(), MAX_QUEUED, "the queue grew without bound");
        assert!(q.dropped > 0, "nothing was trimmed");
        // The newest audio is what survives: a gap beats stale audio mixed in
        // late.
        assert_eq!(*q.samples.back().unwrap(), 9.0);
        assert_eq!(*q.samples.front().unwrap(), 8.0);
    }

    #[test]
    fn a_source_keeping_up_is_never_trimmed() {
        let mut q = SourceQueue::new("steady".into());
        for _ in 0..20 {
            assert_eq!(q.push(vec![0.25; FRAME]), 0);
            q.take_frame();
        }
        assert_eq!(q.dropped, 0);
    }

    #[test]
    fn a_repeated_trim_is_logged_at_most_once_per_interval() {
        // A source running fast is trimmed on every callback, dozens of times
        // a second. Logging each would bury the file in one repeated sentence.
        let mut rate = Rate::new(Duration::from_secs(5));
        let t0 = Instant::now();
        assert!(rate.allow(t0), "the first one has to be said");
        assert!(!rate.allow(t0 + Duration::from_millis(20)));
        assert!(!rate.allow(t0 + Duration::from_secs(4)));
        assert!(rate.allow(t0 + Duration::from_secs(5)));
        assert!(!rate.allow(t0 + Duration::from_secs(6)));
    }

    #[test]
    fn a_source_is_reported_silent_only_after_the_threshold() {
        // Not immediately: every source starts with nothing delivered, and
        // calling a device silent before it has had a chance to speak would
        // make the meter lie for the first frame of every session.
        let mut q = SourceQueue::new("mic".into());
        let now = Instant::now();
        assert!(!q.silent_at(now), "a source is given time to say something");

        q.last_data = now
            .checked_sub(STARVE_AFTER + Duration::from_secs(1))
            .expect("the clock must reach back a few seconds");
        assert!(q.silent_at(now));

        // And it clears the moment anything arrives, silence included: a live
        // microphone in a quiet room is contributing.
        q.push(vec![0.0; 16]);
        assert!(!q.silent_at(Instant::now()));
    }

    #[test]
    fn a_partial_frame_is_padded_rather_than_shortening_the_mix() {
        // Every source has to hand back the same length or `mix_frames` would
        // truncate the whole mix to the shortest one, which is the rendezvous
        // by another name.
        let mut q = SourceQueue::new("partial".into());
        q.push(vec![1.0; 10]);
        let frame = q.take_frame();
        assert_eq!(frame.len(), FRAME);
        assert_eq!(frame[9], 1.0);
        assert_eq!(frame[10], 0.0);
    }

    #[test]
    fn a_source_with_nothing_to_give_contributes_a_frame_of_silence() {
        let mut q = SourceQueue::new("idle".into());
        let frame = q.take_frame();
        assert_eq!(frame.len(), FRAME);
        assert!(frame.iter().all(|s| *s == 0.0));
        assert_eq!(q.rms, 0.0);
    }
}
