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
//! Gain is 1/N over the sources that are *live*, not over every source
//! selected. A source that has delivered nothing for [`STARVE_AFTER`] is left
//! out of the divisor, so a microphone beside a Windows loopback with nothing
//! playing reaches the server at full amplitude instead of half. Dividing by
//! whatever happened to arrive in the last 20 ms would be the dangerous
//! version of this: it would move the output 6 dB every time a source paused
//! between words. Starvation is five seconds of hysteresis, which no utterance
//! is long enough to straddle, so the divisor only moves on an edge that is
//! already slow.
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

/// Average several equal-length frames into one, dividing by `live`.
///
/// Averaged rather than summed: two full-scale inputs added together clip, and
/// clipping is both audible and destructive to recognition. Averaging cannot
/// exceed full scale whatever the inputs do, and preserves which source was
/// actually louder.
///
/// `live` is how many of these sources are currently carrying audio, which is
/// not always how many there are; see the module doc for why the divisor is
/// the smaller of the two.
pub fn mix_frames(frames: &[Vec<f32>], live: usize) -> Vec<f32> {
    if frames.is_empty() {
        return Vec::new();
    }
    let len = frames.iter().map(|f| f.len()).min().unwrap_or(0);
    // At least one, so a mix whose every source has starved still divides by
    // something. Every frame is silence in that case and the value cannot
    // change the answer, but a zero would.
    let n = live.max(1) as f32;
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

/// How much audio a source may buffer before its oldest is dropped: 200 ms.
///
/// With the clock driving emission, a queue that keeps growing means that
/// source runs fast relative to the tick -- which happens for real:
/// `resample_to_16k` decimates with `chunks_exact`, silently dropping up to
/// `factor - 1` samples per callback whenever the device's period is not a
/// multiple of the ratio, and two devices on independent crystals drift
/// regardless. A backlog is latency the listener pays for on every word, so
/// the bound is kept short enough that the mix stays near the present.
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
    /// Nothing has arrived from this source for [`STARVE_AFTER`], or nothing
    /// ever will because it did not open.
    pub silent: bool,
    /// Samples trimmed from this source's queue since the session began.
    ///
    /// Carried to the window rather than only logged: trimming is words going
    /// missing mid-utterance, continuously, and somebody reading a transcript
    /// with holes in it has no other way to discover that is what happened.
    #[serde(default)]
    pub dropped: u64,
    /// Why this source contributes nothing, when its device would not open.
    ///
    /// `None` for a working source and for one that is merely silent -- an
    /// idle loopback is not a fault, and saying so in red would be a lie.
    #[serde(default)]
    pub error: Option<String>,
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
    /// Samples discarded because this queue overflowed.
    dropped: u64,
    /// The capture behind this queue has ended and will send nothing more.
    finished: bool,
    /// Why there is no capture behind this queue at all.
    error: Option<String>,
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
            error: None,
        }
    }

    /// A queue for a source whose device would not open.
    ///
    /// It holds a place in the row order and carries the reason, so the
    /// window can name the source that is missing rather than simply showing
    /// one row where the user selected two.
    fn failed(label: String, why: String) -> Self {
        Self {
            finished: true,
            error: Some(why),
            ..Self::new(label)
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

    /// Nothing has arrived for [`STARVE_AFTER`], or nothing ever will.
    ///
    /// Also the mix's test for whether this source counts towards the gain
    /// divisor, which is why a source that never opened has to answer true
    /// here at once rather than after five seconds of pretending.
    fn silent_at(&self, now: Instant) -> bool {
        self.error.is_some() || now.duration_since(self.last_data) >= STARVE_AFTER
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
                    dropped: q.dropped,
                    error: q.error.clone(),
                }
            })
            .collect()
    }
}

/// What the mix has to work with for one source.
enum Feed {
    /// A capture that started, delivering into this channel.
    Live(mpsc::Receiver<Vec<f32>>),
    /// A device that would not open, and why. The mix runs without it.
    Failed(String),
}

/// Drive the mix from `feeds` onto the clock, into `out`.
///
/// Split out from [`MixedCapture::start`] so it can be driven from synthetic
/// senders. Every fault this exists to prevent is about what happens when a
/// device produces nothing, and no test can arrange that through a real one.
fn spawn_mix(feeds: Vec<(String, Feed)>, out: mpsc::Sender<Vec<f32>>) -> Health {
    let mut queues: Vec<Arc<Mutex<SourceQueue>>> = Vec::with_capacity(feeds.len());

    for (label, feed) in feeds {
        let mut rx = match feed {
            Feed::Live(rx) => rx,
            Feed::Failed(why) => {
                queues.push(Arc::new(Mutex::new(SourceQueue::failed(label, why))));
                continue;
            }
        };
        let queue = Arc::new(Mutex::new(SourceQueue::new(label.clone())));
        queues.push(queue.clone());
        tokio::spawn(async move {
            let mut trim_log = Rate::new(TRIM_LOG_EVERY);
            // `label` is read at most once every five seconds and cloning it
            // per chunk would be fifty clones a second to feed one of them.
            while let Some(chunk) = rx.recv().await {
                let (dropped, total) = {
                    let mut q = queue.lock().expect("mixer queue poisoned");
                    (q.push(chunk), q.dropped)
                };
                if dropped > 0 && trim_log.allow(Instant::now()) {
                    // The cumulative figure, not this call's: trimming happens
                    // on nearly every callback once it starts, so the count
                    // from the one call that won the rate limit understates
                    // the loss by orders of magnitude.
                    //
                    // Which side is at fault is genuinely not known here. A
                    // queue overflows when the source outruns the clock and
                    // equally when the mix falls behind its own tick, and the
                    // queue cannot tell the two apart.
                    tracing::warn!(
                        "trimming {label}'s queue: {total} samples dropped so far. \
                         Either this source is running ahead of the mix or the mix \
                         is running late; both overflow a queue that holds \
                         {MAX_QUEUED} samples."
                    );
                }
            }
            queue.lock().expect("mixer queue poisoned").finished = true;
        });
    }

    let mix = queues.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(TICK);
        // Burst, not Delay. A tick that runs late has the audio it missed
        // sitting in the queues waiting for it, and `Delay` forfeits that
        // audio outright: it restarts the schedule from now, so the frames
        // that were due are never taken and are trimmed away unheard.
        // Bursting hands them over across the next few ticks instead, which
        // is what the speaker actually said. Measured under a consumer
        // stalling 60 ms once a second, `Delay` discarded 180 ms per source
        // per ten seconds and pinned both queues at their bound; `Burst`
        // dropped nothing and left the queues empty.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
        loop {
            ticker.tick().await;
            let now = Instant::now();
            let mut frames = Vec::with_capacity(mix.len());
            let mut spent = true;
            let mut live = 0usize;
            for q in &mix {
                let mut q = q.lock().expect("mixer queue poisoned");
                // Read before `take_frame`, which is what empties the queue.
                // A source whose last samples this tick is about to emit is
                // not spent until they have gone out; reading it afterwards
                // threw away the final frame of every source at stop, and
                // threw away everything when a capture ended with less than
                // one frame buffered.
                spent &= q.spent();
                if !q.silent_at(now) {
                    live += 1;
                }
                frames.push(q.take_frame());
            }
            // Every capture has ended and been drained. Dropping `out` here is
            // what tells the session its audio is over; ticking silence for
            // ever would leave it listening to a dead device.
            if spent {
                break;
            }
            if out.send(mix_frames(&frames, live)).await.is_err() {
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
    /// Start every source that will start, and mix them into `out`.
    ///
    /// A device that refuses to open costs its own row and nothing else. The
    /// point of selecting two sources is that one of them is a Windows
    /// loopback, which is the one that fails -- and failing the whole session
    /// for it would take away the working microphone beside it, which is the
    /// opposite of what the user asked for. The reason travels out on that
    /// source's health row, so the window can say which one is missing and
    /// why rather than quietly recording one source where two were ticked.
    ///
    /// Every source failing is still an error: there is nothing to record,
    /// and an empty mix would report `Listening` for ever.
    pub fn start(sources: &[Source], out: mpsc::Sender<Vec<f32>>) -> Result<Self> {
        if sources.is_empty() {
            anyhow::bail!("no sources to combine");
        }

        let mut captures = Vec::new();
        let mut feeds: Vec<(String, Feed)> = Vec::with_capacity(sources.len());
        let mut failures: Vec<String> = Vec::new();
        for source in sources {
            let (tx, rx) = mpsc::channel::<Vec<f32>>(32);
            match Capture::start(source, tx) {
                Ok(c) => {
                    captures.push(c);
                    feeds.push((source.short_label(), Feed::Live(rx)));
                }
                Err(e) => {
                    let why = format!("{e:#}");
                    tracing::warn!("leaving {} out of the mix: {why}", source.display());
                    failures.push(format!("{}: {why}", source.display()));
                    feeds.push((source.short_label(), Feed::Failed(why)));
                }
            }
        }
        if captures.is_empty() {
            anyhow::bail!("no selected source would open ({})", failures.join("; "));
        }

        Ok(Self {
            health: spawn_mix(feeds, out),
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
        let out = mix_frames(&[vec![1.0, 1.0], vec![1.0, 1.0]], 2);
        assert_eq!(out, vec![1.0, 1.0]);
    }

    #[test]
    fn full_scale_opposites_cancel_rather_than_overflow() {
        assert_eq!(mix_frames(&[vec![1.0], vec![-1.0]], 2), vec![0.0]);
    }

    #[test]
    fn a_quiet_source_stays_quiet_next_to_a_loud_one() {
        // Relative level is information: it says who was actually louder.
        let out = mix_frames(&[vec![1.0], vec![0.0]], 2);
        assert_eq!(out, vec![0.5]);
    }

    #[test]
    fn a_source_left_out_of_the_divisor_costs_the_others_nothing() {
        // The permanently-idle loopback. Counting it in the divisor put a
        // microphone beside it on the wire at exactly half amplitude, for the
        // whole session, in the commonest two-source configuration there is.
        assert_eq!(mix_frames(&[vec![1.0], vec![0.0]], 1), vec![1.0]);
    }

    #[test]
    fn mixing_takes_the_shortest_frame() {
        // Sources never deliver the same amount at the same moment, so the mix
        // is bounded by whichever has least.
        let out = mix_frames(&[vec![1.0, 1.0, 1.0], vec![1.0]], 2);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn one_source_passes_through_unchanged() {
        // Combined mode with a single source must not alter it.
        assert_eq!(mix_frames(&[vec![0.25, -0.5]], 1), vec![0.25, -0.5]);
    }

    #[test]
    fn no_sources_yields_nothing_rather_than_panicking() {
        assert!(mix_frames(&[], 0).is_empty());
        assert!(mix_frames(&[vec![], vec![]], 0).is_empty());
    }

    #[test]
    fn a_divisor_of_zero_does_not_divide_by_zero() {
        // Reachable: every source starved at once is every source silent, so
        // the answer is silence either way -- but a zero divisor would make
        // it NaN, which is not silence and travels a long way downstream.
        assert_eq!(mix_frames(&[vec![0.0], vec![0.0]], 0), vec![0.0]);
    }

    #[test]
    fn three_sources_average_correctly() {
        let out = mix_frames(&[vec![0.9], vec![0.3], vec![0.3]], 3);
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
            vec![
                ("A".to_string(), Feed::Live(a_rx)),
                ("B".to_string(), Feed::Live(b_rx)),
            ],
            out_tx,
        );
        (a_tx, b_tx, out_rx, health)
    }

    /// Push a queue's clock back far enough that the mix reads it as starved.
    ///
    /// Reached into rather than waited out: [`STARVE_AFTER`] is five seconds
    /// by design, and a test that slept through it would be five seconds of
    /// nothing on every run.
    fn starve(health: &Health, which: usize) {
        health.0[which].lock().unwrap().last_data = Instant::now()
            .checked_sub(STARVE_AFTER + Duration::from_secs(1))
            .expect("the clock must reach back a few seconds");
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
        // Still halved, because the idle source has not yet been idle long
        // enough to be called starved: it is given STARVE_AFTER to say
        // something, exactly as the meter rows give it. The hysteresis is the
        // point -- a divisor that followed each frame would move 6 dB every
        // time a speaker paused. See the test below for the other side.
        assert!((heard[0] - 0.5).abs() < 1e-6, "got {}", heard[0]);
    }

    #[tokio::test]
    async fn a_starved_source_stops_costing_the_others_their_level() {
        // Mic plus a Windows loopback with nothing playing is the reported
        // configuration, and the loopback in it is silent for the whole
        // session. Dividing by it anyway put the microphone on the wire 6 dB
        // down for as long as the user cared to talk.
        let (live, _idle, mut out, health) = two_sources();
        starve(&health, 1);
        for _ in 0..4 {
            live.send(vec![1.0; FRAME]).await.unwrap();
        }

        let heard = first_audible(&mut out).await;
        assert!((heard[0] - 1.0).abs() < 1e-6, "got {}", heard[0]);
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
    async fn the_mix_ends_only_after_emitting_what_its_sources_left_behind() {
        // Two things at once, because they are the same tick. A session whose
        // devices all died must stop rather than listen to manufactured
        // silence for ever -- and the tick that drains the last queue still
        // has to send what it drained. `take_frame` is what empties a queue,
        // so a `spent` test taken after it is true on the very tick that
        // produced the final audio: this exact case, one frame pushed and
        // both sources gone, emitted nothing at all.
        let (a, b, mut out, _health) = two_sources();
        a.send(vec![0.5; FRAME]).await.unwrap();
        drop(a);
        drop(b);

        let heard = tokio::time::timeout(Duration::from_secs(5), async {
            let mut frames = 0usize;
            let mut audible = 0usize;
            while let Some(f) = out.recv().await {
                frames += 1;
                if f.iter().any(|s| *s != 0.0) {
                    audible += 1;
                }
            }
            (frames, audible)
        })
        .await
        .expect("the mix kept emitting after every source ended");
        assert!(heard.0 > 0, "the mix ended without emitting anything");
        assert_eq!(heard.1, 1, "the last frame of audio never left the mix");
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

    #[tokio::test]
    async fn a_mix_that_ran_late_catches_up_rather_than_forfeiting_the_backlog() {
        // tokio applies its missed-tick behaviour whenever a tick runs more
        // than 5 ms late, which on a loaded machine is routine. `Delay`
        // restarts the schedule from the moment it noticed, so the frames that
        // fell due while it was late are never taken -- and they are not
        // skipped either, because they are sitting in the source's queue.
        // They are held there and trimmed away later as overflow. Measured
        // under a consumer stalling 60 ms once a second: 180 ms of audio
        // discarded per source per ten seconds, both queues pinned at their
        // bound, and 200 ms of latency added for good.
        const PUSHED: usize = 9;
        let (a_tx, a_rx) = mpsc::channel::<Vec<f32>>(64);
        // One deep, so a consumer that stops reading stalls the mix at once
        // rather than after sixty-four frames of slack.
        let (out_tx, mut out) = mpsc::channel::<Vec<f32>>(1);
        let health = spawn_mix(vec![("A".to_string(), Feed::Live(a_rx))], out_tx);
        for _ in 0..PUSHED {
            a_tx.send(vec![0.5; FRAME]).await.unwrap();
        }

        // Nobody reads for fifteen ticks' worth. The mix gets two frames into
        // the channel and then blocks on the third.
        tokio::time::sleep(TICK * 15).await;

        // Everything owed has to come back at once now, not one frame per
        // 20 ms from a schedule that restarted.
        let (mut frames, mut audible) = (0usize, 0usize);
        let _ = tokio::time::timeout(TICK * 5, async {
            loop {
                let Some(f) = out.recv().await else { break };
                frames += 1;
                if f.iter().any(|s| *s != 0.0) {
                    audible += 1;
                }
            }
        })
        .await;

        assert_eq!(
            audible, PUSHED,
            "only {audible} of {PUSHED} pushed frames came back; the mix is \
             emitting on a restarted schedule rather than the one it owes"
        );
        assert_eq!(health.read()[0].dropped, 0, "audio was trimmed away unheard");
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

    #[tokio::test]
    async fn a_source_that_would_not_open_neither_silences_nor_hides_itself() {
        // The whole thesis of selecting two sources. A loopback that refuses
        // to open must cost its own row and nothing more -- not the working
        // microphone beside it, and not its own place in the meter, or the
        // window shows one row where two were ticked and the user is back to
        // guessing which one went missing.
        let (a_tx, a_rx) = mpsc::channel::<Vec<f32>>(32);
        let (out_tx, mut out) = mpsc::channel::<Vec<f32>>(64);
        let health = spawn_mix(
            vec![
                ("Yeti".to_string(), Feed::Live(a_rx)),
                (
                    "System audio".to_string(),
                    Feed::Failed("Device does not support input".to_string()),
                ),
            ],
            out_tx,
        );
        a_tx.send(vec![1.0; FRAME]).await.unwrap();

        // Full scale, not half: a source that never opened is not a live one,
        // so it takes no share of the gain.
        let heard = first_audible(&mut out).await;
        assert!((heard[0] - 1.0).abs() < 1e-6, "got {}", heard[0]);

        let rows = health.read();
        assert_eq!(rows.len(), 2, "the source that failed lost its row");
        assert_eq!(rows[1].label, "System audio");
        assert_eq!(
            rows[1].error.as_deref(),
            Some("Device does not support input"),
            "the reason has to reach the window, not only the log"
        );
        assert!(rows[1].silent);
    }

    #[tokio::test]
    async fn a_mix_whose_every_source_failed_is_an_error_not_an_empty_stream() {
        // The other half: skipping is only right while something is left. A
        // mix with nothing behind it would report Listening for ever and
        // record silence.
        let missing = |n: &str| Source {
            target: crate::source::SourceTarget::CpalDevice {
                name: n.to_string(),
                loopback: false,
            },
            name: n.to_string(),
            kind: crate::source::SourceKind::Microphone,
            detail: None,
            stable_name: None,
            sink_description: None,
        };
        let (out_tx, _out) = mpsc::channel::<Vec<f32>>(8);
        let started = MixedCapture::start(
            &[missing("no such device one"), missing("no such device two")],
            out_tx,
        );
        let text = match started {
            Ok(_) => panic!("a mix with nothing behind it must not report success"),
            Err(e) => format!("{e:#}"),
        };
        assert!(text.contains("no such device one"), "{text}");
        assert!(text.contains("no such device two"), "{text}");
    }
}
