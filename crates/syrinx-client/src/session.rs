//! The dictation session. One implementation, shared by the CLI and the GUI.
//!
//! Runs its own tokio runtime on a dedicated thread and communicates through
//! shared state, so a front-end never blocks on the network: a stalled server
//! cannot freeze a GUI or wedge a CLI.

use crate::inject;
use crate::mode::OutputMode;
use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use syrinx_audio::mixer::{STARVE_AFTER, SourceHealth};
use syrinx_audio::{Capture, Source};
use syrinx_proto::{ClientMessage, Encoding, SAMPLE_RATE, ServerMessage};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    #[default]
    Idle,
    Connecting,
    Listening,
    Stopping,
    /// Working through an audio file rather than a live source.
    Transcribing,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Idle => "Idle",
            Status::Connecting => "Connecting…",
            Status::Listening => "LISTENING",
            Status::Stopping => "Stopping…",
            Status::Transcribing => "TRANSCRIBING FILE",
        }
    }

    pub fn is_active(self) -> bool {
        !matches!(self, Status::Idle)
    }
}

/// One fragment of transcript with the time it arrived.
///
/// Kept alongside the flat string so a transcript can be saved with timestamps
/// without the caller having to reconstruct timing it never saw.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    /// Seconds since the session started.
    pub at: f64,
    pub text: String,
    /// Which source produced this, when more than one is in play. `None` for a
    /// single source or a combined mix, where attribution is meaningless.
    pub source: Option<String>,
    /// Anonymous speaker from the server's diarizer, when one ran.
    pub speaker: Option<u32>,
    /// Whether `speaker` is the diarizer's best guess rather than an answer it
    /// will stand behind -- the commit's `speaker_provisional`.
    ///
    /// Kept per segment because it is what makes `transcript.relabel`'s
    /// promise enforceable here: a correction may fill a gap or replace a
    /// guess, and may not touch a label a full window settled. Nothing else
    /// reads it, and nothing renders it: a reader is shown the best answer
    /// going either way.
    ///
    /// Defaults to false, which is what a state file written before the field
    /// existed meant and what an older server's commits mean.
    #[serde(default, skip_serializing_if = "is_not_provisional")]
    pub speaker_provisional: bool,
    /// The `seq` of the transcript message this came from, so a later
    /// `transcript.relabel` naming a range of seqs can find it.
    ///
    /// `None` for a segment that did not come from a commit -- a revision
    /// rewrites the tail, and its replacement is nobody's commit -- and for one
    /// read back from a state file written before corrections existed. Omitted
    /// from the wire when absent, for the same reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
}

/// serde helper: lets the common `false` vanish from a state file.
fn is_not_provisional(b: &bool) -> bool {
    !*b
}

/// Give `from_seq..=to_seq` a speaker, reporting whether anything moved.
///
/// The client half of `transcript.relabel`. A voice needs several agreeing
/// windows before the server can mint it, so a turn's opening sentences are
/// committed before anyone can be named for them; this is where the name
/// catches up. It changes attribution and never text, which is what makes it
/// safe to apply to a buffer somebody is reading.
///
/// A free function rather than a branch inside the reader loop because the
/// reader is a spawned task inside an async session and this is the part worth
/// testing. Segments carry their commit's `seq`, so the range is matched
/// exactly rather than by counting backwards from the end the way a revision
/// has to.
///
/// Deliberately *not* mirrored into the streamed file: `StreamWriter` is
/// append-only and keeps the honest record of what was known live, while these
/// segments -- which the GUI paints and Save-as writes -- carry the corrected
/// attribution. That divergence is the design's, and `stream.rs` proves the
/// writer ignores corrections rather than leaving it to be believed.
///
/// **A segment carrying a confident label is never touched.** The range says
/// which commits a correction *may* cover; each commit's own
/// `speaker_provisional` says whether it is one of them. Applying to
/// everything in range instead is how one speaker's settled sentence acquires
/// another's name, and the protocol promises it cannot -- a promise only this
/// end can keep, because only this end knows what it is holding.
pub fn apply_relabel(segments: &mut [Segment], from_seq: u64, to_seq: u64, speaker: u32) -> bool {
    let mut moved = false;
    for seg in segments {
        let in_range = seg.seq.is_some_and(|s| s >= from_seq && s <= to_seq);
        let correctable = seg.speaker.is_none() || seg.speaker_provisional;
        if in_range && correctable && seg.speaker != Some(speaker) {
            seg.speaker = Some(speaker);
            seg.speaker_provisional = false;
            moved = true;
        }
    }
    moved
}

/// Everything a front-end needs to render. Cheap to clone.
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    pub status: Status,
    /// Accumulated text. Empty in [`OutputMode::Type`], which keeps none.
    pub transcript: String,
    /// The same text, split by arrival with timings.
    pub segments: Vec<Segment>,
    /// Most recent fragment, so a UI can show it is alive.
    pub last_fragment: String,
    pub model: Option<String>,
    pub chunk_ms: Option<u32>,
    /// What the handshake actually granted -- the client may have asked and
    /// still not received, e.g. no diarization models on the server.
    pub diarize: bool,
    /// What `session.start` actually asked for -- computed once, in `run`,
    /// from the same expression that builds the wire message, so this and
    /// the request sent can never disagree. Separate mode runs one session
    /// per source with a per-source mode, so this is per-session like
    /// `diarize`, not read back from global options.
    pub diarize_requested: bool,
    pub error: Option<String>,
    /// Trouble appending to the transcript file, which is not [`Self::error`].
    ///
    /// A failed append costs the fragment it was carrying and nothing else:
    /// the writer stays open, the next fragment may well succeed, the
    /// transcript in memory is untouched and the session goes on. Recording
    /// it as an error made a USB stick blinking once into a failed run --
    /// `syrinx start --stream` exited non-zero after a complete, saved
    /// transcript -- and in separate mode it took the one `error` slot
    /// `merge_states` keeps, hiding a second source that had really died.
    ///
    /// Kept once set. The lost words stay lost, so a later success is not
    /// permission to report a complete file.
    pub stream_error: Option<String>,
    /// Live spectrum of the audio being sent, so a viewer can see the session
    /// is receiving sound rather than only that it is running.
    pub levels: Vec<f32>,
    pub rms: f32,
    /// What each selected source is contributing, one row per source.
    ///
    /// `levels` and `rms` above are measured downstream of the mixer, so with
    /// two sources they say nothing about which of the two is carrying
    /// anything -- a session with a live microphone and a dead loopback looks
    /// exactly like one with both live. These are taken upstream, per source.
    pub sources: Vec<SourceHealth>,
    /// How many times `transcript` and `segments` have moved.
    ///
    /// Read by the daemon on every tick to answer "has the transcript
    /// changed" without looking at the transcript, which at a meeting's
    /// length is tens of thousands of segments. A count rather than a
    /// length: a revision retracts and re-pushes, so both the segment count
    /// and the character count can come out the same on either side of a
    /// real change.
    pub changes: u64,
}

impl SessionState {
    /// The same state with the transcript left out.
    ///
    /// `transcript` and `segments` come back empty; everything else is
    /// copied. Cloning the whole thing costs a string per segment, which
    /// after two hours of meeting is tens of thousands of them -- far too
    /// much to pay forty times a second for a status line and a level meter.
    /// `changes` comes through, so a caller can tell whether the transcript
    /// it left behind has moved and clone the real thing only then.
    pub fn live(&self) -> SessionState {
        SessionState {
            status: self.status,
            transcript: String::new(),
            segments: Vec::new(),
            last_fragment: self.last_fragment.clone(),
            model: self.model.clone(),
            chunk_ms: self.chunk_ms,
            diarize: self.diarize,
            diarize_requested: self.diarize_requested,
            error: self.error.clone(),
            stream_error: self.stream_error.clone(),
            levels: self.levels.clone(),
            rms: self.rms,
            sources: self.sources.clone(),
            changes: self.changes,
        }
    }
}

/// Fold several concurrent sessions into one view.
///
/// Segments carry their own timings, so ordering them by time reconstructs what
/// was actually said in what order across sources -- which is the point of
/// separate mode. The flat transcript follows that same order, so what is read
/// back is a conversation rather than one speaker and then the other.
///
/// Lives here rather than in either front-end because both need it and the
/// answer must be the same: the daemon folds its sessions on every tick and
/// again to save, and `syrinx start --separate` folds its own at the end. It
/// touches nothing but `SessionState`, so neither of those is a concern of the
/// other's.
///
/// Every field is folded across all the states rather than taken from the
/// first. `error` especially: a second source whose session failed has to be
/// reported, or a run comes back successful having recorded half of what was
/// asked for.
pub fn merge_states(states: &[SessionState]) -> SessionState {
    if states.len() == 1 {
        return states[0].clone();
    }
    let mut out = SessionState {
        // Active if any session is; the aggregate is running until all stop.
        status: states
            .iter()
            .map(|s| s.status)
            .find(|s| s.is_active())
            .unwrap_or_default(),
        model: states.iter().find_map(|s| s.model.clone()),
        chunk_ms: states.iter().find_map(|s| s.chunk_ms),
        // Any session that got labels is enough to say so; separate mode
        // runs one session per source and only the non-typing ones request
        // diarization, so an all-false merge would wrongly blame the server.
        diarize: states.iter().any(|s| s.diarize),
        // Likewise for the request itself: in separate mode the primary
        // source keeps whatever mode the user picked while every other
        // source is forced to Transcribe, so it alone can be the one asking.
        diarize_requested: states.iter().any(|s| s.diarize_requested),
        error: states.iter().find_map(|s| s.error.clone()),
        // Folded separately, and this is the point of it being separate: a
        // stream blip on the first source used to take the one `error` slot
        // and hide a second source whose connection had really dropped.
        stream_error: states.iter().find_map(|s| s.stream_error.clone()),
        // Levels come from the first source; a merged spectrum would say less
        // than one real one.
        levels: states.first().map(|s| s.levels.clone()).unwrap_or_default(),
        rms: states.first().map(|s| s.rms).unwrap_or(0.0),
        // Concatenated rather than folded. Separate mode runs one session per
        // source and each knows only about its own, so the merged view is
        // every session's rows in selection order -- which is exactly the one
        // row per selected source a viewer needs.
        sources: states.iter().flat_map(|s| s.sources.clone()).collect(),
        last_fragment: states
            .iter()
            .map(|s| s.last_fragment.clone())
            .find(|f| !f.is_empty())
            .unwrap_or_default(),
        ..Default::default()
    };

    let mut segments: Vec<Segment> = states.iter().flat_map(|s| s.segments.clone()).collect();
    segments.sort_by(|a, b| a.at.partial_cmp(&b.at).unwrap_or(std::cmp::Ordering::Equal));
    out.transcript = crate::save::render(&segments, "", crate::save::Format::Labelled);
    out.segments = segments;
    out
}

/// Parameters for a run.
#[derive(Debug)]
pub struct SessionOptions {
    pub url: String,
    pub token: String,
    /// Sources to capture. More than one is mixed into a single stream; for
    /// independent streams the caller runs a session per source.
    pub sources: Vec<Source>,
    pub mode: OutputMode,
    /// Ask the server for anonymous speaker labels. Only takes effect when
    /// the mode's wire mode is `Transcript` -- `Both` keeps a transcript too,
    /// but types at the cursor and so runs the wire live and never requests
    /// labels; see the `session.start` construction in `run`.
    pub diarize: bool,
    /// Label applied to this session's segments, for separate mode.
    pub label: Option<String>,
    /// How text is typed at the cursor.
    pub inject: crate::inject::Method,
    /// Append each committed fragment to this file as it arrives.
    pub stream: Option<(std::path::PathBuf, crate::save::Format)>,
    /// Audio supplied by the caller rather than captured from a device.
    ///
    /// When set, `sources` is ignored and no capture is started. This is the
    /// seam that lets the session run where syrinx cannot open a microphone
    /// itself -- on iOS the audio comes from AVAudioEngine on the Swift side,
    /// and everything downstream of this channel is unchanged.
    ///
    /// Samples must be 16 kHz mono f32, the same as a capture backend
    /// produces.
    pub external_audio: Option<mpsc::Receiver<Vec<f32>>>,
}

/// Handle to a running session. Dropping it stops the session.
pub struct SessionHandle {
    state: Arc<Mutex<SessionState>>,
    stop: Option<oneshot::Sender<()>>,
}

impl SessionHandle {
    pub fn state(&self) -> SessionState {
        self.state.lock().expect("state lock poisoned").clone()
    }

    /// Everything about the session except the transcript.
    ///
    /// [`state`](Self::state) clones every segment, which after two hours of
    /// meeting is tens of thousands of strings -- far too much to pay forty
    /// times a second for a status line and a level meter. See
    /// [`SessionState::live`] for what is left out.
    pub fn live(&self) -> SessionState {
        self.state.lock().expect("state lock poisoned").live()
    }

    pub fn stop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
    }

    pub fn is_running(&self) -> bool {
        self.state().status.is_active()
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start a session on a background thread.
///
/// `on_change` fires whenever state moves, so a UI can repaint on demand rather
/// than polling.
pub fn start(
    opts: SessionOptions,
    on_change: impl Fn() + Send + Sync + 'static,
) -> SessionHandle {
    let state = Arc::new(Mutex::new(SessionState {
        status: Status::Connecting,
        ..Default::default()
    }));
    let (stop_tx, stop_rx) = oneshot::channel();
    let st = state.clone();
    let notify = Arc::new(on_change);

    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                fail(&st, format!("starting the async runtime: {e}"));
                notify();
                return;
            }
        };
        finish(&st, || rt.block_on(run(opts, st.clone(), stop_rx, notify.clone())));
        notify();
    });

    SessionHandle {
        state,
        stop: Some(stop_tx),
    }
}

/// Run a session to its end, and make sure the end is visible.
///
/// Split out from [`start`] so a test can drive every way of ending without a
/// server or a device. The one that mattered is the panic: cpal's Windows
/// backend panics rather than erroring, on the caller's thread, and this
/// thread is that caller. An unguarded panic killed it before anything could
/// be recorded, so the status stayed wherever it was last set -- `Listening`,
/// which the handshake sets before any device is opened. `Status::is_active`
/// then read the dead session as running for ever: the daemon never reaped it,
/// Stop sent into a oneshot nobody held, metering stayed off, and only
/// restarting the daemon got out of it.
///
/// So whatever happens in `body`, this leaves the session idle, and leaves the
/// reason where a window can show it.
fn finish(state: &Arc<Mutex<SessionState>>, body: impl FnOnce() -> Result<()>) {
    match syrinx_audio::caught("the session", body) {
        Ok(()) => state.lock().expect("state lock poisoned").status = Status::Idle,
        Err(e) => {
            error!("session failed: {e:#}");
            fail(state, format!("{e:#}"));
        }
    }
}

/// Record that a fragment did not reach the transcript file.
///
/// The failure used to reach `error!` and nothing further, so a file that had
/// stopped taking words an hour into a meeting looked exactly like one still
/// being written: the daemon log is not somewhere anyone has open while
/// dictating, and the file itself only says so once it is read.
///
/// What it says is what is known -- one append failed. The writer is not
/// closed and the next fragment may well succeed, so "the stream stopped" was
/// a claim about the future that a single lost fragment does not support.
///
/// The first message is the one kept. Those words are not written by the
/// fragment after them, so a message that cleared itself would report a
/// complete file that is missing a sentence.
fn record_lost_fragment(state: &mut SessionState, lost: Option<String>) {
    let Some(msg) = lost else { return };
    state.stream_error.get_or_insert(msg);
}

fn fail(state: &Arc<Mutex<SessionState>>, msg: String) {
    let mut s = state.lock().expect("state lock poisoned");
    s.error = Some(msg);
    s.status = Status::Idle;
}

async fn run(
    mut opts: SessionOptions,
    state: Arc<Mutex<SessionState>>,
    mut stop_rx: oneshot::Receiver<()>,
    notify: Arc<impl Fn() + Send + Sync + 'static>,
) -> Result<()> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message as Ws;

    if opts.mode.types_at_cursor() {
        // Fail before the microphone opens rather than after the user speaks.
        inject::preflight(opts.inject)?;
    }

    crate::install_crypto_provider();
    let mut req = opts
        .url
        .as_str()
        .into_client_request()
        .with_context(|| format!("building a request for {}", opts.url))?;
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {}", opts.token)
            .parse()
            .context("token is not a valid header value")?,
    );

    let (ws, _) = tokio_tungstenite::connect_async(req)
        .await
        .with_context(|| format!("connecting to {}", opts.url))?;
    let (mut tx, mut rx) = ws.split();

    // Never on a typing session, even if the caller asked: a typing mode runs
    // the wire live and types every fragment as it lands, so "Speaker 2:"
    // landing at the cursor would be destructive, not decorative.
    // Mode::Transcript is exactly the modes that don't type -- guaranteed by
    // OutputMode::wire_mode's own invariant test, so this holds by
    // construction. Computed once and stamped onto state immediately, so a
    // viewer's honest-handshake notice reads the same request this sends,
    // never a second, possibly-diverging computation of it.
    let diarize_requested = opts.diarize && opts.mode.wire_mode() == syrinx_proto::Mode::Transcript;
    state.lock().expect("state lock poisoned").diarize_requested = diarize_requested;

    tx.send(Ws::Text(
        serde_json::to_string(&ClientMessage::SessionStart {
            // Typing forces the append-only wire mode; see OutputMode::wire_mode.
            mode: opts.mode.wire_mode(),
            sample_rate: SAMPLE_RATE,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
            diarize: diarize_requested,
        })?
        .into(),
    ))
    .await
    .context("sending session.start")?;

    match rx.next().await {
        Some(Ok(Ws::Text(t))) => match serde_json::from_str::<ServerMessage>(&t)? {
            ServerMessage::SessionReady {
                model,
                chunk_ms,
                diarize,
                ..
            } => {
                // Everything the handshake settled, but not yet `Listening`:
                // no device has been opened at this point, and opening one is
                // where the failures are. Saying Listening here meant the
                // window read Listening for the whole of an open that was
                // going to fail.
                let mut s = state.lock().expect("state lock poisoned");
                s.model = Some(model);
                s.chunk_ms = Some(chunk_ms);
                s.diarize = diarize;
            }
            ServerMessage::Error { code, message, .. } => {
                bail!("server refused the session ({code:?}): {message}")
            }
            other => bail!("expected session.ready, got {other:?}"),
        },
        Some(Ok(other)) => bail!("expected a text frame, got {other:?}"),
        Some(Err(e)) => return Err(e).context("reading session.ready"),
        None => bail!("server closed the connection before session.ready"),
    }
    notify();

    // Timestamps are relative to the moment audio starts flowing, not to
    // connection, so they line up with the recording rather than including
    // model load time.
    let started = std::time::Instant::now();

    // Either the caller feeds us, or we open a device ourselves. The rest of
    // the session cannot tell the difference: it only ever sees a receiver.
    //
    // The mixer's own view of each source comes out here too, because the
    // capture itself disappears behind `dyn Any` a line later and the levels
    // that matter are the ones taken before everything was averaged together.
    type Capt = Option<Box<dyn std::any::Any + Send>>;
    let (_capture, mut audio_rx, mixed): (Capt, _, Option<syrinx_audio::mixer::Health>) =
        match opts.external_audio.take() {
            Some(rx) => (None, rx, None),
            None => {
                let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(32);
                // One source opens directly; several are mixed. A single source
                // through the mixer would work but adds a queue and a timer for
                // nothing.
                if opts.sources.len() == 1 {
                    let cap = Capture::start(&opts.sources[0], audio_tx)
                        .context("starting audio capture")?;
                    (Some(Box::new(cap) as _), audio_rx, None)
                } else {
                    let cap = syrinx_audio::mixer::MixedCapture::start(&opts.sources, audio_tx)
                        .context("starting the combined capture")?;
                    let health = cap.health();
                    (Some(Box::new(cap) as _), audio_rx, Some(health))
                }
            }
        };
    info!(
        "session running: {} -> {}",
        opts.sources
            .iter()
            .map(|s| s.display())
            .collect::<Vec<_>>()
            .join(" + "),
        opts.mode.label()
    );

    // Opened before the session starts, so an unwritable path is refused now
    // rather than discovered after an hour of talking.
    let mut stream = match &opts.stream {
        Some((path, format)) => Some(
            crate::stream::StreamWriter::open(path, *format)
                .context("opening the transcript stream")?,
        ),
        None => None,
    };

    // Only now: a device is open and a place to write it exists, so this is
    // the first moment at which "listening" is true rather than intended.
    state.lock().expect("state lock poisoned").status = Status::Listening;
    notify();

    let st = state.clone();
    let n = notify.clone();
    let mode = opts.mode;
    let label = opts.label.clone();
    let inject_method = opts.inject;
    let reader = tokio::spawn(async move {
        while let Some(Ok(msg)) = rx.next().await {
            let Ws::Text(t) = msg else { continue };
            match serde_json::from_str::<ServerMessage>(&t) {
                Ok(ServerMessage::TranscriptCommit {
                    seq,
                    text,
                    speaker,
                    speaker_provisional,
                })
                | Ok(ServerMessage::TranscriptProvisional {
                    seq,
                    text,
                    speaker,
                    speaker_provisional,
                }) => {
                    if mode.types_at_cursor()
                        && let Err(e) = inject::type_text(&text, inject_method)
                    {
                        error!("failed to type {text:?}: {e:#}");
                    }
                    let seg = Segment {
                        seq: Some(seq),
                        at: started.elapsed().as_secs_f64(),
                        text: text.clone(),
                        source: label.clone(),
                        speaker,
                        speaker_provisional,
                    };
                    // Written whatever the mode: the point of streaming is a
                    // copy on disk, and typing at the cursor is exactly when
                    // no transcript is kept in memory to save later.
                    let appended = match stream.as_mut() {
                        Some(w) => w.append(&seg),
                        None => Ok(()),
                    };
                    // Logged and worded out here, before the lock. The daemon
                    // reads this state forty times a second and neither a log
                    // line nor a formatted message is worth making it wait.
                    let lost = appended.err().map(|e| {
                        error!("failed to append to the transcript stream: {e:#}");
                        format!("a fragment was not written to the transcript file: {e:#}")
                    });
                    let mut s = st.lock().expect("state lock poisoned");
                    record_lost_fragment(&mut s, lost);
                    if mode.keeps_transcript() {
                        s.transcript.push_str(&text);
                        s.segments.push(seg);
                        s.changes += 1;
                    }
                    s.last_fragment = text;
                    drop(s);
                    n();
                }
                Ok(ServerMessage::TranscriptRevise {
                    retract_n, text, ..
                }) => {
                    // Only reachable in transcribe-only mode, where the client
                    // owns the buffer. A typing session runs in Live mode, so
                    // the server never sends this.
                    let mut s = st.lock().expect("state lock poisoned");
                    let keep = s.transcript.chars().count().saturating_sub(retract_n);
                    s.transcript = s.transcript.chars().take(keep).collect();
                    s.transcript.push_str(&text);
                    // Revisions rewrite the tail, so the affected segments are
                    // replaced rather than appended to.
                    let mut dropped = 0usize;
                    while dropped < retract_n {
                        match s.segments.pop() {
                            Some(seg) => dropped += seg.text.chars().count(),
                            None => break,
                        }
                    }
                    s.segments.push(Segment {
                        seq: None,
                        at: started.elapsed().as_secs_f64(),
                        text: text.clone(),
                        source: label.clone(),
                        // Revise carries no speaker by design: it is
                        // reserved for a future post-processing layer, not
                        // something the diarizer's lag buffer ever emits.
                        speaker: None,
                        speaker_provisional: false,
                    });
                    s.changes += 1;
                    drop(s);
                    n();
                }
                Ok(ServerMessage::TranscriptRelabel {
                    from_seq,
                    to_seq,
                    speaker,
                }) => {
                    // Attribution only: not a character of text moves, so this
                    // is safe to apply under a reader's eyes in a way a
                    // revision would not be. The streamed file is deliberately
                    // not told -- see `apply_relabel`.
                    let mut s = st.lock().expect("state lock poisoned");
                    if apply_relabel(&mut s.segments, from_seq, to_seq, speaker) {
                        s.changes += 1;
                        drop(s);
                        n();
                    }
                }
                Ok(ServerMessage::SessionClosed { .. }) => break,
                Ok(ServerMessage::Error { code, message, .. }) => {
                    warn!("server error ({code:?}): {message}");
                    let mut s = st.lock().expect("state lock poisoned");
                    s.error = Some(message);
                    drop(s);
                    n();
                    break;
                }
                Ok(_) => {}
                Err(e) => warn!("undecodable server frame: {e}"),
            }
        }
    });

    // Metered from the audio actually being sent, so what a viewer sees is the
    // stream the server receives rather than a second capture of the same
    // device.
    let mut window: Vec<f32> = Vec::with_capacity(4096);

    // A single source has no mixer to ask, so its row is assembled from the
    // metering below. Caller-supplied audio has no device behind it at all and
    // reports no rows.
    let solo = (mixed.is_none() && opts.sources.len() == 1).then(|| opts.sources[0].short_label());
    let mut last_audio = std::time::Instant::now();
    // On a timer rather than only when a chunk lands: a source going quiet is
    // exactly what these rows exist to report, and a session whose only source
    // died would otherwise never publish the fact.
    let mut health_tick = tokio::time::interval(std::time::Duration::from_millis(200));

    loop {
        tokio::select! {
            _ = &mut stop_rx => break,
            _ = health_tick.tick() => {
                // Built before the state lock is taken, so the mixer's queue
                // locks and this one are never held at the same time and no
                // ordering between them can arise to be got wrong later.
                let rows = match (&mixed, &solo) {
                    (Some(h), _) => h.read(),
                    (None, Some(label)) => {
                        let rms = state.lock().expect("state lock poisoned").rms;
                        vec![SourceHealth {
                            label: label.clone(),
                            rms,
                            silent: last_audio.elapsed() >= STARVE_AFTER,
                            // A lone source is opened directly, with no queue
                            // behind it to trim, and one that would not open
                            // failed the whole session rather than reaching
                            // here -- there is only the one.
                            dropped: 0,
                            error: None,
                        }]
                    }
                    (None, None) => Vec::new(),
                };
                state.lock().expect("state lock poisoned").sources = rows;
            }
            chunk = audio_rx.recv() => match chunk {
                Some(samples) => {
                    last_audio = std::time::Instant::now();
                    window.extend_from_slice(&samples);
                    if window.len() > 4096 {
                        let excess = window.len() - 4096;
                        window.drain(..excess);
                    }
                    {
                        let mut s = state.lock().expect("state lock poisoned");
                        s.levels = syrinx_audio::meter::spectrum(&window, SAMPLE_RATE).to_vec();
                        s.rms = syrinx_audio::meter::rms(&samples);
                    }
                    if tx.send(Ws::Binary(to_pcm_s16le(&samples).into())).await.is_err() {
                        break;
                    }
                }
                None => break,
            }
        }
    }

    state.lock().expect("state lock poisoned").status = Status::Stopping;
    notify();

    let _ = tx
        .send(Ws::Text(
            serde_json::to_string(&ClientMessage::SessionStop)?.into(),
        ))
        .await;
    // Bounded: a hung server must not leave the microphone open forever.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(10), reader).await;
    Ok(())
}

/// Encode to little-endian s16 for the wire.
fn to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        // 32767, not 32768: scaling by 32768 lets +1.0 overflow i16 and wrap to
        // the most negative value, turning the loudest sample into a click.
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_does_not_wrap() {
        let out = to_pcm_s16le(&[1.0, -1.0]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
        assert_eq!(i16::from_le_bytes([out[2], out[3]]), -32767);
    }

    #[test]
    fn out_of_range_input_is_clamped_not_wrapped() {
        let out = to_pcm_s16le(&[9.0, -9.0]);
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), 32767);
        assert_eq!(i16::from_le_bytes([out[2], out[3]]), -32767);
    }

    #[test]
    fn encoding_round_trips_through_the_shared_decoder() {
        // The server decodes with syrinx_proto::pcm_s16le_to_f32, so the two
        // must agree or every transcript is subtly wrong.
        let original = [0.0f32, 0.5, -0.5, 0.25];
        let decoded = syrinx_proto::pcm_s16le_to_f32(&to_pcm_s16le(&original));
        for (a, b) in original.iter().zip(&decoded) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn status_labels_are_distinct() {
        let all = [
            Status::Idle,
            Status::Connecting,
            Status::Listening,
            Status::Stopping,
        ];
        let mut l: Vec<&str> = all.iter().map(|s| s.label()).collect();
        l.sort_unstable();
        let n = l.len();
        l.dedup();
        assert_eq!(l.len(), n);
    }

    #[test]
    fn a_merge_reports_an_error_from_any_session() {
        // `syrinx start --separate` bails on the merged state's error. The
        // CLI's own copy of this fold took every field from the first session
        // alone, so a run whose second source never connected exited
        // successfully having recorded half of what was asked for.
        let failed = SessionState {
            error: Some("connecting to the server: refused".into()),
            ..Default::default()
        };
        let merged = merge_states(&[SessionState::default(), failed]);
        assert_eq!(
            merged.error.as_deref(),
            Some("connecting to the server: refused")
        );
    }

    fn lost(what: &str) -> Option<String> {
        Some(format!("a fragment was not written to the transcript file: {what}"))
    }

    #[test]
    fn a_fragment_that_never_reached_the_file_is_worth_saying() {
        // It only ever reached `error!` in the daemon log, which is not a
        // place anyone has open while dictating -- so a file that had stopped
        // taking words an hour into a meeting looked exactly like one still
        // being written, right up until it was read.
        let mut s = SessionState::default();
        record_lost_fragment(&mut s, lost("No space left on device"));
        let msg = s.stream_error.expect("a lost fragment is worth showing");
        assert!(msg.contains("No space left on device"), "{msg}");
        assert!(msg.contains("fragment"), "say what was lost: {msg}");
    }

    #[test]
    fn a_lost_fragment_does_not_fail_the_session() {
        // `error` is what the CLI exits non-zero on and what the window paints
        // in red. A USB stick blinking once during a meeting that was fully
        // transcribed and fully saved is neither of those.
        let mut s = SessionState::default();
        record_lost_fragment(&mut s, lost("No space left on device"));
        assert_eq!(s.error, None);
    }

    #[test]
    fn an_append_that_worked_says_nothing() {
        let mut s = SessionState::default();
        record_lost_fragment(&mut s, None);
        assert_eq!(s.stream_error, None);
        assert_eq!(s.error, None);
    }

    #[test]
    fn a_later_success_does_not_erase_a_lost_fragment() {
        // The fragment that could not be written is not written by the one
        // after it, so a message that cleared itself would report a complete
        // file that is missing a sentence.
        let mut s = SessionState::default();
        record_lost_fragment(&mut s, lost("No space left on device"));
        record_lost_fragment(&mut s, None);
        assert!(s.stream_error.is_some());
    }

    #[test]
    fn a_second_lost_fragment_does_not_replace_the_first() {
        // The first is the one that says what went wrong; the ones after it
        // are the same disk saying so again.
        let mut s = SessionState::default();
        record_lost_fragment(&mut s, lost("No space left on device"));
        record_lost_fragment(&mut s, lost("Input/output error"));
        assert!(s.stream_error.unwrap().contains("No space left"));
    }

    #[test]
    fn a_lost_fragment_on_one_source_does_not_hide_a_failure_on_another() {
        // Separate mode folds a state per source and keeps the first `error`
        // it finds. While a lost fragment lived in that field, a stream blip
        // on the first source masked a second source whose connection had
        // really dropped, and the run came back reporting the blip.
        let blipped = SessionState {
            stream_error: lost("No space left on device"),
            ..Default::default()
        };
        let dropped = SessionState {
            error: Some("the server closed the connection".into()),
            ..Default::default()
        };
        let merged = merge_states(&[blipped, dropped]);
        assert_eq!(
            merged.error.as_deref(),
            Some("the server closed the connection")
        );
        assert!(merged.stream_error.is_some(), "both are worth reporting");
    }

    #[test]
    fn a_merge_keeps_a_meter_row_for_every_source() {
        // Separate mode runs a session per source and each knows only about
        // its own, so a merge that took the first session's rows -- as it does
        // for `levels` -- would leave the second source unmetered everywhere,
        // which is the state that made this invisible in the first place.
        let row = |label: &str, silent: bool| syrinx_audio::mixer::SourceHealth {
            label: label.into(),
            rms: 0.3,
            silent,
            ..Default::default()
        };
        let merged = merge_states(&[
            SessionState {
                sources: vec![row("Yeti", false)],
                ..Default::default()
            },
            SessionState {
                sources: vec![row("System audio", true)],
                ..Default::default()
            },
        ]);
        assert_eq!(
            merged
                .sources
                .iter()
                .map(|s| (s.label.as_str(), s.silent))
                .collect::<Vec<_>>(),
            [("Yeti", false), ("System audio", true)]
        );
    }

    /// A session that has got as far as reporting `Listening`.
    fn listening() -> Arc<Mutex<SessionState>> {
        Arc::new(Mutex::new(SessionState {
            status: Status::Listening,
            ..Default::default()
        }))
    }

    #[test]
    fn a_session_thread_that_panics_ends_failed_rather_than_listening_for_ever() {
        // The wedge behind "I restart the daemon a few times". cpal's Windows
        // backend panics on the calling thread, and that thread is the one
        // running the session. Nothing recorded the ending, so the status
        // stayed at whatever it had reached -- Listening -- and `is_active`
        // read the dead session as running: never reaped, Stop into a oneshot
        // nobody held, metering off, no error anywhere to explain it. The
        // panic message this prints is the test doing its job.
        let state = listening();
        finish(&state, || panic!("could not get endpoint data_flow"));

        let s = state.lock().unwrap();
        assert_eq!(s.status, Status::Idle, "a dead session still looked active");
        assert!(
            s.error
                .as_deref()
                .is_some_and(|e| e.contains("could not get endpoint data_flow")),
            "the window has nothing to show for it: {:?}",
            s.error
        );
    }

    // ------------------------------------------------------------ relabels

    /// Segments as a commit stream would produce them: `seq` counting from 1,
    /// one second apart, unattributed unless the caller says otherwise.
    fn committed(speakers: &[Option<u32>]) -> Vec<Segment> {
        speakers
            .iter()
            .enumerate()
            .map(|(i, speaker)| Segment {
                at: i as f64,
                text: format!("fragment {i} "),
                source: None,
                speaker: *speaker,
                speaker_provisional: false,
                seq: Some(i as u64 + 1),
            })
            .collect()
    }

    /// The same, with each segment's `speaker_provisional` given explicitly.
    fn committed_with_guesses(speakers: &[(Option<u32>, bool)]) -> Vec<Segment> {
        speakers
            .iter()
            .enumerate()
            .map(|(i, (speaker, guess))| Segment {
                at: i as f64,
                text: format!("fragment {i} "),
                source: None,
                speaker: *speaker,
                speaker_provisional: *guess,
                seq: Some(i as u64 + 1),
            })
            .collect()
    }

    #[test]
    fn a_relabel_leaves_a_settled_label_inside_its_range_alone() {
        // The promise `transcript.relabel` documents, kept at the end that can
        // keep it. A correction's range says which commits it *may* cover; the
        // commit's own `speaker_provisional` says whether it is one of them.
        // Applying to everything in range instead is how one speaker's settled
        // sentence acquires another's name.
        let mut segs = committed_with_guesses(&[(Some(2), false), (None, false), (Some(3), true)]);
        assert!(apply_relabel(&mut segs, 1, 3, 1));
        assert_eq!(
            segs.iter()
                .map(|s| (s.speaker, s.speaker_provisional))
                .collect::<Vec<_>>(),
            vec![(Some(2), false), (Some(1), false), (Some(1), false)],
            "a confident label was overwritten by a correction"
        );
    }

    #[test]
    fn a_session_that_ends_cleanly_goes_idle_with_nothing_to_report() {
        let state = listening();
        finish(&state, || Ok(()));
        let s = state.lock().unwrap();
        assert_eq!(s.status, Status::Idle);
        assert_eq!(s.error, None, "a clean stop invented a failure");
    }

    #[test]
    fn a_session_that_returns_an_error_still_carries_it() {
        // The guard must not swallow the ordinary failure it sits beside.
        let state = listening();
        finish(&state, || anyhow::bail!("connecting to the server: refused"));
        let s = state.lock().unwrap();
        assert_eq!(s.status, Status::Idle);
        assert_eq!(s.error.as_deref(), Some("connecting to the server: refused"));
    }

    #[test]
    fn a_relabel_that_can_touch_nothing_reports_no_change() {
        // The return value drives a repaint, and a correction that lands
        // entirely on settled labels has to be as quiet as one that lands on
        // nothing at all.
        let mut segs = committed_with_guesses(&[(Some(2), false), (Some(3), false)]);
        assert!(!apply_relabel(&mut segs, 1, 2, 1));
        assert_eq!(
            segs.iter().map(|s| s.speaker).collect::<Vec<_>>(),
            vec![Some(2), Some(3)]
        );
    }

    #[test]
    fn a_corrected_segment_stops_being_correctable() {
        // A correction is a full window's answer, so what it writes is
        // settled: a second correction naming the same range must leave it
        // alone rather than trading the label back and forth.
        let mut segs = committed_with_guesses(&[(Some(2), true)]);
        assert!(apply_relabel(&mut segs, 1, 1, 1));
        assert!(!segs[0].speaker_provisional);
        assert!(!apply_relabel(&mut segs, 1, 1, 3));
        assert_eq!(segs[0].speaker, Some(1));
    }

    #[test]
    fn a_relabel_names_the_segments_in_its_range_and_no_others() {
        let mut segs = committed(&[None, None, None, Some(2)]);
        assert!(apply_relabel(&mut segs, 1, 2, 1));
        assert_eq!(
            segs.iter().map(|s| s.speaker).collect::<Vec<_>>(),
            vec![Some(1), Some(1), None, Some(2)]
        );
    }

    #[test]
    fn a_relabel_changes_attribution_and_never_a_character_of_text() {
        // What makes it safe to apply to a buffer somebody is reading, and the
        // whole reason it is not `transcript.revise`.
        let before = committed(&[None, None]);
        let mut after = before.clone();
        apply_relabel(&mut after, 1, 2, 3);
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.at, b.at);
            assert_eq!(a.source, b.source);
            assert_eq!(a.seq, b.seq);
        }
    }

    #[test]
    fn a_relabel_for_seqs_this_client_never_saw_moves_nothing() {
        // A viewer can join late, or have dropped frames, or be looking at a
        // merged view where another source's session numbered its own commits.
        let mut segs = committed(&[None, None]);
        assert!(!apply_relabel(&mut segs, 90, 99, 1));
        assert!(segs.iter().all(|s| s.speaker.is_none()));
    }

    #[test]
    fn a_relabel_that_agrees_with_what_is_there_reports_no_change() {
        // The return value drives a repaint, so a correction that corrects
        // nothing must not cause one.
        let mut segs = committed(&[Some(1), Some(1)]);
        assert!(!apply_relabel(&mut segs, 1, 2, 1));
    }

    #[test]
    fn a_segment_with_no_seq_is_never_relabelled() {
        // A revision's replacement segment is nobody's commit, and neither is
        // anything read back from a state file written before corrections
        // existed. A relabel has no way to know which commit either belongs
        // to, so it leaves both alone rather than guessing by position.
        let mut segs = committed(&[None, None]);
        segs[0].seq = None;
        assert!(apply_relabel(&mut segs, 0, 99, 1));
        assert_eq!(
            segs.iter().map(|s| s.speaker).collect::<Vec<_>>(),
            vec![None, Some(1)]
        );
    }

    #[test]
    fn a_relabel_reaches_the_saved_transcript() {
        // Save-as renders from these segments, so corrected attribution is
        // what lands in the file the user keeps. The other half of the
        // divergence -- that the streamed file does not move -- is stated in
        // `stream.rs`, from the side that would have had to do the moving.
        let mut segs = committed(&[None, None]);
        segs[1].at = 0.5;
        assert_eq!(
            crate::save::render(&segs, "", crate::save::Format::Plain),
            "fragment 0 fragment 1"
        );
        apply_relabel(&mut segs, 1, 2, 1);
        assert_eq!(
            crate::save::render(&segs, "", crate::save::Format::Plain),
            "Speaker 1: fragment 0 fragment 1"
        );
    }

    #[test]
    fn only_idle_is_inactive() {
        assert!(!Status::Idle.is_active());
        assert!(Status::Connecting.is_active());
        assert!(Status::Listening.is_active());
        assert!(Status::Stopping.is_active());
    }
}
