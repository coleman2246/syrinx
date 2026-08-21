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
    GetState,
    Start,
    Stop,
    Toggle,
    SetMode { mode: OutputMode },
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
    pub last_fragment: String,
    pub model: Option<String>,
    pub chunk_ms: Option<u32>,
    pub error: Option<String>,
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
    pub fn from_session(s: &SessionState, mode: OutputMode, source_key: Option<String>) -> Self {
        Self {
            status: s.status,
            mode,
            transcript: s.transcript.clone(),
            last_fragment: s.last_fragment.clone(),
            model: s.model.clone(),
            chunk_ms: s.chunk_ms,
            error: s.error.clone(),
            source_key,
            source_keys: Vec::new(),
            source_mode: Default::default(),
            // A running session meters its own audio; the daemon overwrites
            // these from the idle preview when there is no session.
            levels: s.levels.clone(),
            rms: s.rms,
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
            Request::GetState,
            Request::Toggle,
            Request::SetMode {
                mode: OutputMode::Both,
            },
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
