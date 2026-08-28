//! Local-socket IPC between the background daemon and the GUI.
//!
//! The daemon owns the session and the tray; the GUI is a viewer that attaches
//! and detaches. That split exists because the GUI cannot outlive its own
//! window: winit documents `set_visible` as unsupported on Wayland, so a window
//! cannot hide itself and keep running. Something that never had a window has
//! to hold the session instead.
//!
//! Line-delimited JSON. On Unix that travels over a socket in
//! `$XDG_RUNTIME_DIR`, chosen because the directory is cleared on logout so a
//! stale socket cannot outlive the session that made it. On Windows it is a
//! named pipe, which has no filesystem presence to go stale.
//!
//! Deliberately *not* the abstract namespace on Linux, though `interprocess`
//! offers it on both platforms: an abstract socket has no filesystem
//! permissions, so every user on the machine could drive the daemon. A named
//! pipe carries a security descriptor, so Windows loses nothing by it.

use crate::mode::OutputMode;
use crate::save::Format;
use crate::session::{SessionState, Status};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use interprocess::local_socket::traits::Stream as _;
use interprocess::local_socket::{Name, Stream};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// A request from a front-end to the daemon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Poll current state.
    ///
    /// `since` is the last [`DaemonState::revision`] this viewer saw. When
    /// it still matches, the daemon leaves the transcript and its turns out
    /// of the reply and the viewer keeps the copy it already has -- which is
    /// what stops a two-hour meeting being serialised, sent and parsed
    /// thirty times a second to say nothing new. `None` always gets the
    /// whole thing, which is what an older viewer's request decodes to.
    GetState {
        #[serde(default)]
        since: Option<u64>,
    },
    Start,
    Stop,
    Toggle,
    SetMode { mode: OutputMode },
    /// Ask the server for speaker labels on the next session. Refused while
    /// one is running, for the same reason as `SetMode`.
    SetDiarize { diarize: bool },
    SetSource { key: String },
    /// Replace the whole selection, and optionally how they are combined.
    SetSources {
        keys: Vec<String>,
        source_mode: Option<crate::mode::SourceMode>,
    },
    /// Point the daemon at a different machine. Applies to the next session.
    SetServer { server: String },
    /// Layout for saved and streamed transcripts.
    SetFormat { format: Format },
    /// Append the transcript to this file as it is dictated. `None` stops.
    /// Takes effect on the next session.
    SetStreamFile { path: Option<String> },
    /// Save the current transcript, returning the path written.
    Save { format: Format, path: Option<String> },
    /// Transcribe an audio file, replacing the current transcript.
    TranscribeFile { path: String },
    /// Save one file per source. Separate mode only.
    SaveSplit { format: Format },
    /// Discard the retained transcript.
    Clear,
    /// Stop the daemon entirely.
    Quit,
}

/// The daemon's reply. Every request gets exactly one.
///
/// `State` dwarfs the other variants, but boxing it would trade a stack copy
/// for a heap allocation on the one variant that is sent most often, and these
/// are serialised and dropped immediately rather than held in bulk.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    State(DaemonState),
    /// A path, for `Save`.
    Saved { path: String },
    Ok,
    Error { message: String },
}

/// Everything a viewer needs. Flattened from [`SessionState`] plus the
/// daemon-owned settings, so one round trip renders a whole frame.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DaemonState {
    pub status: Status,
    pub mode: OutputMode,
    pub transcript: String,
    /// The same text grouped into turns, as `save::turn_texts` produces them
    /// -- a speaker and that speaker's prose, ready to render.
    ///
    /// Turns rather than the raw segments they are grouped from. A viewer
    /// needs nothing per-segment: the only thing ever rendered from them was
    /// the paragraph view, so shipping the grouping ready-made drops the
    /// `at` and `source` fields nobody reads, collapses roughly nine
    /// segments into one object, and takes the grouping off the viewer's
    /// repaint path, where it used to run over the whole transcript thirty
    /// times a second.
    ///
    /// Empty when nothing carries a speaker: the flat `transcript` above is
    /// the whole story then, and filling this too would send every word
    /// twice.
    #[serde(default)]
    pub turns: Vec<(Option<u32>, String)>,
    /// Which transcript the two fields above are. Bumped on every change,
    /// and monotonic, so no two different transcripts can ever share one.
    ///
    /// A viewer passes the last one it saw back as [`Request::GetState`]'s
    /// `since`; an unchanged answer arrives without the text. Zero means
    /// "this daemon does not track revisions" -- which is what an older
    /// one's `serde(default)` produces -- so a viewer must never read a zero
    /// as a match and keep stale text on the strength of it.
    #[serde(default)]
    pub revision: u64,
    pub last_fragment: String,
    pub model: Option<String>,
    pub chunk_ms: Option<u32>,
    /// What the handshake actually granted; see `SessionState::diarize`.
    #[serde(default)]
    pub diarize: bool,
    /// Whether the session behind this state actually asked for labels; see
    /// `SessionState::diarize_requested`. A viewer must judge the
    /// honest-handshake notice against this, not its own config read, which
    /// is a second process reading a file the daemon may have read a moment
    /// earlier or later.
    #[serde(default)]
    pub diarize_requested: bool,
    /// What the daemon will ask for at the *next* session -- the setting
    /// itself, which is neither of the two above: `diarize` is what a server
    /// granted and `diarize_requested` is what one session sent, and both are
    /// false while nothing is running. A control that has to show its own
    /// state needs the setting, not either answer about a session.
    #[serde(default)]
    pub diarize_configured: bool,
    pub error: Option<String>,
    /// A fragment that never reached the transcript file; see
    /// `SessionState::stream_error`. Separate from `error` because it does not
    /// stop the session, and a window that painted it as one would be saying
    /// the recording had died when it is still running.
    #[serde(default)]
    pub stream_error: Option<String>,
    /// Stable key of the first source, kept for viewers that show only one.
    pub source_key: Option<String>,
    /// Every selected source, in order.
    #[serde(default)]
    pub source_keys: Vec<String>,
    #[serde(default)]
    pub source_mode: crate::mode::SourceMode,
    /// Ten-band spectrum of the selected source, for confirming there is signal
    /// before starting. Zeroed when nothing is being metered.
    #[serde(default)]
    pub levels: Vec<f32>,
    /// Overall level of the same, 0.0 to 1.0.
    #[serde(default)]
    pub rms: f32,
    /// One row per selected source: its label, its level, and whether it has
    /// gone quiet.
    ///
    /// The two fields above are measured downstream of the mixer while a
    /// session runs, and from the first source alone while idle, so with two
    /// sources selected neither says which one is carrying anything. That was
    /// impossible to see from a viewer at all: a running session at 0% and one
    /// at 40% looked identical.
    ///
    /// Empty from a daemon too old to report them, which a viewer must read as
    /// "not known" rather than "no sources".
    #[serde(default)]
    pub sources: Vec<syrinx_audio::mixer::SourceHealth>,
    /// Progress through a file, 0.0 to 1.0. Only meaningful while transcribing
    /// a file.
    #[serde(default)]
    pub progress: f32,
    /// What became of the configured global hotkey. Shown in the GUI's help.
    #[serde(default)]
    pub hotkey: crate::hotkey::Report,
    /// Where the transcript is being appended, if anywhere.
    #[serde(default)]
    pub stream_to: Option<String>,
    /// Layout in use for saving and streaming.
    #[serde(default)]
    pub format: Format,
}

impl DaemonState {
    /// Everything a session knows about itself, ready for the daemon to
    /// stamp its own settings onto.
    ///
    /// `transcript`, `turns` and `revision` are deliberately left empty: the
    /// daemon holds those, rebuilds them only when they change, and fills
    /// them in itself. Passing a `SessionHandle::live` state here is
    /// therefore no loss.
    pub fn from_session(s: &SessionState, mode: OutputMode, source_key: Option<String>) -> Self {
        Self {
            status: s.status,
            mode,
            transcript: String::new(),
            turns: Vec::new(),
            revision: 0,
            last_fragment: s.last_fragment.clone(),
            model: s.model.clone(),
            chunk_ms: s.chunk_ms,
            diarize: s.diarize,
            diarize_requested: s.diarize_requested,
            // Stamped by the daemon: a session knows what it asked for, not
            // what the setting says now.
            diarize_configured: false,
            error: s.error.clone(),
            stream_error: s.stream_error.clone(),
            source_key,
            source_keys: Vec::new(),
            source_mode: Default::default(),
            // A running session meters its own audio; the daemon overwrites
            // these from the idle preview when there is no session.
            levels: s.levels.clone(),
            rms: s.rms,
            sources: s.sources.clone(),
            progress: 0.0,
            // Stamped by the daemon, which is the only thing that knows.
            hotkey: crate::hotkey::Report::Unset,
            stream_to: None,
            format: Format::default(),
        }
    }
}

/// Path of the daemon socket.
///
/// Meaningful only on Unix. Windows named pipes live in their own namespace,
/// so the path is used purely as the source of the pipe's name.
pub fn socket_path() -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(r"\\.\pipe\syrinx.sock")
    }
    #[cfg(not(windows))]
    {
        let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(dir).join("syrinx.sock")
    }
}

/// The address the daemon listens on and clients connect to.
pub fn socket_name() -> Result<Name<'static>> {
    #[cfg(windows)]
    {
        use interprocess::local_socket::{GenericNamespaced, ToNsName};
        "syrinx.sock"
            .to_ns_name::<GenericNamespaced>()
            .context("building the daemon pipe name")
    }
    #[cfg(not(windows))]
    {
        use interprocess::local_socket::{GenericFilePath, ToFsName};
        socket_path()
            .to_fs_name::<GenericFilePath>()
            .context("building the daemon socket name")
    }
}

/// Send one request and read the reply.
///
/// A fresh connection per request: these are infrequent and tiny, and it means
/// a viewer that dies mid-request leaves nothing behind for the daemon to clean
/// up.
pub fn request(req: &Request) -> Result<Response> {
    let stream = Stream::connect(socket_name()?).with_context(|| {
        format!(
            "connecting to the syrinx daemon at {}",
            socket_path().display()
        )
    })?;
    // Bounded so a wedged daemon cannot hang a GUI frame indefinitely.
    stream
        .set_recv_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();

    // `&Stream` is both Read and Write, so one connection serves both
    // directions without a duplicate handle.
    let mut w = &stream;
    writeln!(w, "{}", serde_json::to_string(req)?).context("sending the request")?;
    w.flush().ok();

    let mut line = String::new();
    BufReader::new(&stream)
        .read_line(&mut line)
        .context("reading the reply")?;
    serde_json::from_str(&line).context("decoding the reply")
}

/// Whether a daemon is listening.
///
/// Connecting rather than checking the file exists: a stale socket left by a
/// killed daemon is still a file, and treating it as "running" would make every
/// front-end fail with a confusing connection error instead of just starting
/// one.
pub fn daemon_running() -> bool {
    socket_name().is_ok_and(|n| Stream::connect(n).is_ok())
}

/// Remove a socket left behind by a daemon that did not shut down cleanly.
///
/// Unix only. A named pipe exists only while its server holds it open, so
/// Windows has nothing that can be left behind.
pub fn clear_stale_socket() {
    #[cfg(not(windows))]
    {
        let p = socket_path();
        if p.exists() && !daemon_running() {
            let _ = std::fs::remove_file(&p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_round_trip() {
        for r in [
            Request::GetState { since: None },
            Request::GetState { since: Some(42) },
            Request::Toggle,
            Request::SetMode {
                mode: OutputMode::Both,
            },
            Request::SetDiarize { diarize: true },
            Request::Save {
                format: Format::Timestamped,
                path: Some("/tmp/x.txt".into()),
            },
            Request::Quit,
        ] {
            let s = serde_json::to_string(&r).unwrap();
            assert_eq!(serde_json::from_str::<Request>(&s).unwrap(), r);
        }
    }

    #[test]
    fn responses_round_trip() {
        let r = Response::State(DaemonState {
            status: Status::Listening,
            mode: OutputMode::Type,
            transcript: "hello".into(),
            ..Default::default()
        });
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<Response>(&s).unwrap(), r);
    }

    #[test]
    fn a_poll_from_an_older_viewer_still_decodes() {
        // A viewer built before revisions sends a bare `get_state`. It has to
        // keep working, and it has to mean "send me everything" -- which is
        // exactly what `since: None` asks for.
        assert_eq!(
            serde_json::from_str::<Request>(r#"{"type":"get_state"}"#).unwrap(),
            Request::GetState { since: None }
        );
    }

    #[test]
    fn a_state_from_an_older_daemon_reports_no_revision() {
        // The other direction: a daemon that predates revisions sends no such
        // field, and `serde(default)` fills in zero. A viewer reads zero as
        // "not tracked" and asks for the whole transcript every time, which
        // is what it used to get anyway.
        let old = r#"{"type":"state","status":"idle","mode":"transcribe",
            "transcript":"hello","last_fragment":"","model":null,
            "chunk_ms":null,"error":null,"source_key":null}"#;
        let Response::State(s) = serde_json::from_str::<Response>(old).unwrap() else {
            panic!("expected a state");
        };
        assert_eq!(s.revision, 0);
        assert_eq!(s.transcript, "hello");
        assert!(s.turns.is_empty());
    }

    #[test]
    fn per_source_meters_cross_the_wire() {
        // The viewer renders one row per source from these; a field that did
        // not survive serialisation would leave every source reading silent.
        let r = Response::State(DaemonState {
            sources: vec![
                syrinx_audio::mixer::SourceHealth {
                    label: "Yeti".into(),
                    rms: 0.42,
                    silent: false,
                    dropped: 4_800,
                    error: None,
                },
                // The two things a row can say beyond a level: how much of
                // this source has been trimmed away unheard, and why it is
                // contributing nothing at all.
                syrinx_audio::mixer::SourceHealth {
                    label: "System audio".into(),
                    rms: 0.0,
                    silent: true,
                    dropped: 0,
                    error: Some("Device does not support input".into()),
                },
            ],
            ..Default::default()
        });
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<Response>(&s).unwrap(), r);
    }

    #[test]
    fn a_state_from_a_daemon_without_per_source_meters_still_decodes() {
        // A daemon too old to report them sends no such field, and the viewer
        // must fall back to its single overall meter rather than refuse the
        // reply outright.
        let old = r#"{"type":"state","status":"idle","mode":"transcribe",
            "transcript":"","last_fragment":"","model":null,
            "chunk_ms":null,"error":null,"source_key":null}"#;
        let Response::State(s) = serde_json::from_str::<Response>(old).unwrap() else {
            panic!("expected a state");
        };
        assert!(s.sources.is_empty());
    }

    #[test]
    fn an_unknown_request_is_rejected() {
        // A viewer and daemon at different versions should fail loudly rather
        // than have the daemon guess at an unfamiliar command.
        assert!(serde_json::from_str::<Request>(r#"{"type":"self_destruct"}"#).is_err());
    }

    #[test]
    fn messages_are_single_line_so_the_framing_holds() {
        // The protocol is line-delimited; an embedded newline would desync it.
        let r = Request::Save {
            format: Format::Plain,
            path: Some("/tmp/a b.txt".into()),
        };
        assert!(!serde_json::to_string(&r).unwrap().contains('\n'));

        let big = Response::State(DaemonState {
            transcript: "line one\nline two\nline three".into(),
            ..Default::default()
        });
        // A transcript legitimately contains newlines; JSON must escape them.
        assert!(!serde_json::to_string(&big).unwrap().contains('\n'));
    }

    #[test]
    fn the_socket_lives_in_the_runtime_directory() {
        // XDG_RUNTIME_DIR is cleared on logout, so a stale socket cannot
        // outlive the session that created it.
        assert!(socket_path().ends_with("syrinx.sock"));
    }

    #[test]
    fn the_socket_name_is_constructible_on_this_platform() {
        // A name that cannot be built means no front-end can reach the daemon
        // at all, so it is worth failing here rather than at first use.
        socket_name().expect("the platform socket name must build");
    }

    #[test]
    fn no_daemon_means_not_running_rather_than_an_error() {
        // Called on every front-end start; it must answer, not fail.
        let _ = daemon_running();
    }
}
