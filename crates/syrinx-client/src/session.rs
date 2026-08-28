//! The dictation session. One implementation, shared by the CLI and the GUI.
//!
//! Runs its own tokio runtime on a dedicated thread and communicates through
//! shared state, so a front-end never blocks on the network: a stalled server
//! cannot freeze a GUI or wedge a CLI.

use crate::inject;
use crate::mode::OutputMode;
use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use std::sync::{Arc, Mutex};
use syrinx_audio::{Capture, Source};
use syrinx_proto::{ClientMessage, Encoding, SAMPLE_RATE, ServerMessage};
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
pub fn apply_relabel(segments: &mut [Segment], from_seq: u64, to_seq: u64, speaker: u32) -> bool {
    let mut moved = false;
    for seg in segments {
        if seg.seq.is_some_and(|s| s >= from_seq && s <= to_seq) && seg.speaker != Some(speaker) {
            seg.speaker = Some(speaker);
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
    /// Live spectrum of the audio being sent, so a viewer can see the session
    /// is receiving sound rather than only that it is running.
    pub levels: Vec<f32>,
    pub rms: f32,
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
            levels: self.levels.clone(),
            rms: self.rms,
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
        // Levels come from the first source; a merged spectrum would say less
        // than one real one.
        levels: states.first().map(|s| s.levels.clone()).unwrap_or_default(),
        rms: states.first().map(|s| s.rms).unwrap_or(0.0),
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
pub fn start(opts: SessionOptions, on_change: impl Fn() + Send + Sync + 'static) -> SessionHandle {
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
        match rt.block_on(run(opts, st.clone(), stop_rx, notify.clone())) {
            Ok(()) => st.lock().expect("state lock poisoned").status = Status::Idle,
            Err(e) => {
                error!("session failed: {e:#}");
                fail(&st, format!("{e:#}"));
            }
        }
        notify();
    });

    SessionHandle {
        state,
        stop: Some(stop_tx),
    }
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
                let mut s = state.lock().expect("state lock poisoned");
                s.status = Status::Listening;
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
    let (_capture, mut audio_rx): (Option<Box<dyn std::any::Any + Send>>, _) =
        match opts.external_audio.take() {
            Some(rx) => (None, rx),
            None => {
                let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>(32);
                // One source opens directly; several are mixed. A single source
                // through the mixer would work but adds a queue and a timer for
                // nothing.
                let cap: Box<dyn std::any::Any + Send> = if opts.sources.len() == 1 {
                    Box::new(
                        Capture::start(&opts.sources[0], audio_tx)
                            .context("starting audio capture")?,
                    )
                } else {
                    Box::new(
                        syrinx_audio::mixer::MixedCapture::start(&opts.sources, audio_tx)
                            .context("starting the combined capture")?,
                    )
                };
                (Some(cap), audio_rx)
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

    let st = state.clone();
    let n = notify.clone();
    let mode = opts.mode;
    let label = opts.label.clone();
    let inject_method = opts.inject;
    let reader = tokio::spawn(async move {
        while let Some(Ok(msg)) = rx.next().await {
            let Ws::Text(t) = msg else { continue };
            match serde_json::from_str::<ServerMessage>(&t) {
                Ok(ServerMessage::TranscriptCommit { seq, text, speaker })
                | Ok(ServerMessage::TranscriptProvisional { seq, text, speaker }) => {
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
                    };
                    // Written whatever the mode: the point of streaming is a
                    // copy on disk, and typing at the cursor is exactly when
                    // no transcript is kept in memory to save later.
                    if let Some(w) = stream.as_mut()
                        && let Err(e) = w.append(&seg)
                    {
                        error!("failed to append to the transcript stream: {e:#}");
                    }
                    let mut s = st.lock().expect("state lock poisoned");
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
    loop {
        tokio::select! {
            _ = &mut stop_rx => break,
            chunk = audio_rx.recv() => match chunk {
                Some(samples) => {
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
                seq: Some(i as u64 + 1),
            })
            .collect()
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
