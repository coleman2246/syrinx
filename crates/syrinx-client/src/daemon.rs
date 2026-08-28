//! The background daemon: owns the session, the tray, and the IPC socket.
//!
//! Everything persistent lives here. The GUI attaches as a viewer and may come
//! and go; closing its window does not stop dictation, because the daemon never
//! had a window to lose.

use crate::ipc::{self, DaemonState, Request, Response};
use crate::mode::OutputMode;
use crate::save;
use crate::session::{SessionHandle, SessionOptions, merge_states};
use crate::tray::{TrayCommand, TrayState};
use crate::{Config, choose_source, list_sources};
use anyhow::{Context, Result};
use interprocess::local_socket::traits::{ListenerExt as _, Stream as _};
use interprocess::local_socket::{ListenerOptions, Stream};
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex, mpsc};
use tracing::{error, info, warn};

/// Work out what to tell the user about their hotkey.
///
/// Three different things can be true and they are not interchangeable: none
/// was asked for, one was asked for and cannot work here, or one was asked for
/// and failed. Only the daemon knows the third, since registration happens
/// alongside the tray.
fn report_hotkey(
    hotkey: Option<&crate::hotkey::HotKey>,
    error: Option<String>,
) -> crate::hotkey::Report {
    let Some(h) = hotkey else {
        return crate::hotkey::Report::Unset;
    };
    let spelled = h.spelled.clone();

    if let Some(why) = crate::hotkey::supported_here() {
        return crate::hotkey::Report::Unavailable {
            spelled,
            why: why.to_string(),
        };
    }
    if let Some(error) = error {
        return crate::hotkey::Report::Failed { spelled, error };
    }
    // Linux under X11: the parser accepts it and the platform allows it, but
    // registration is only wired up on Windows so far. Saying "active" here
    // would be a lie the user could only catch by pressing the key.
    if !cfg!(windows) {
        return crate::hotkey::Report::Unavailable {
            spelled,
            why: "syrinx only registers hotkeys on Windows so far. Bind it in \
                  your window manager to run `syrinx toggle`."
                .into(),
        };
    }
    crate::hotkey::Report::Active { spelled }
}

/// Read and check the configured hotkey.
fn state_hotkey(config: &Config) -> Result<Option<crate::hotkey::HotKey>> {
    let Some(spec) = config.hotkey.as_deref() else {
        return Ok(None);
    };
    crate::hotkey::parse(spec)
        .map(Some)
        .with_context(|| format!("the `hotkey` setting in your config ({spec:?}) is not valid"))
}

/// What the daemon does when a session ends by itself.
pub struct DaemonOptions {
    pub config: Config,
    pub mode: OutputMode,
    /// Selected sources, in order. The first is the one that types at the
    /// cursor in separate mode.
    pub source_keys: Vec<String>,
    pub source_mode: crate::mode::SourceMode,
    /// Save each session automatically as it ends.
    pub save_each: bool,
    pub format: save::Format,
    /// Command used to open a viewer when the tray asks for one.
    pub gui_command: Option<String>,
}

/// Run until the tray or a client asks to quit.
pub fn run(opts: DaemonOptions) -> Result<()> {
    ipc::clear_stale_socket();
    let sock = ipc::socket_path();
    // Ask rather than infer: on Windows there is no file to look for, and on
    // Unix a leftover file is not proof of a live daemon.
    if ipc::daemon_running() {
        anyhow::bail!(
            "a syrinx daemon is already running (socket at {}). Use `syrinx stop` \
             or quit it from the tray.",
            sock.display()
        );
    }
    let listener = ListenerOptions::new()
        .name(ipc::socket_name()?)
        .create_sync()
        .with_context(|| format!("binding the daemon socket at {}", sock.display()))?;

    // A channel per source of commands, so the main loop has one place to look.
    let (tx, rx) = mpsc::channel::<(Request, Option<mpsc::Sender<Response>>)>();

    // Latest state, published by the loop and read directly by clients.
    //
    // GetState used to travel through the same channel as commands, so a poll
    // waited for the next loop tick before it was even looked at. Reading a
    // published snapshot decouples how fresh state is from how fast the loop
    // spins, which is what a 30 Hz meter needs.
    let published: Arc<Mutex<Published>> = Arc::new(Mutex::new(Published::default()));

    // Accept loop: one thread per connection, which is ample for a handful of
    // short-lived requests.
    {
        let tx = tx.clone();
        let pub_state = published.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(s) => {
                        let tx = tx.clone();
                        let pub_state = pub_state.clone();
                        std::thread::spawn(move || serve_client(s, tx, pub_state));
                    }
                    Err(e) => warn!("accepting a client: {e}"),
                }
            }
        });
    }

    // Parsed before anything is started, so a typo is reported at once rather
    // than after the daemon looks like it came up cleanly.
    let hotkey = match state_hotkey(&opts.config) {
        Ok(h) => h,
        Err(e) => {
            warn!("{e:#}");
            None
        }
    };

    let tray = crate::tray::start(hotkey.clone());
    let (tray_handle, mut tray_rx, hotkey_error) = match tray {
        Some(t) => (Some(t.handle), Some(t.commands), t.hotkey_error),
        None => {
            info!("no system tray available; running headless");
            (None, None, None)
        }
    };

    // What actually became of the hotkey, so the GUI's help can report the
    // truth rather than repeat the config back at the user.
    let hotkey_report = report_hotkey(hotkey.as_ref(), hotkey_error);
    info!("{}", hotkey_report.summary());

    // Linux registers separately from the tray: ksni has no message loop to
    // share. On Wayland there is nothing to register at all, and saying so is
    // the whole point -- a hotkey that silently does nothing is worse than one
    // that was never offered.
    let mut state = DaemonRuntime {
        opts,
        sessions: Vec::new(),
        overlay: None,
        file_job: None,
        preview: None,
        last_viewer: None,
        last: Default::default(),
        // One rather than zero, so the first tick's token cannot match the
        // cache's default and the published revision leaves zero -- which
        // viewers read as "this daemon does not track revisions" -- behind
        // immediately.
        generation: 1,
        text: TranscriptCache {
            // Seeded per process, so revisions are unique across daemons and
            // not merely within one. A viewer holds its last revision
            // through a disconnection; if a replacement daemon started
            // counting from one again, it could in principle reach the
            // number that viewer is still holding and be taken for the
            // transcript the viewer already has -- which would leave a dead
            // session's words on screen as though they were live. Numbering
            // from the pid means no two daemons ever share a revision.
            revision: (std::process::id() as u64) << 32,
            ..Default::default()
        },
    };

    // Resolve the source up front so viewers show what would actually be used
    // rather than the word "default". Enumerating is a subprocess call, so it
    // happens once here rather than on every state poll.
    if state.opts.source_keys.is_empty()
        && let Ok(sources) = list_sources()
        && let Ok(s) = choose_source(&sources, None)
    {
        state.opts.source_keys = vec![s.stable_key()];
    }

    // And again here, not only at the session that uses it, so a window
    // opened the morning after names the file that would really be written
    // rather than yesterday's. Costs a config write once a day at most: a
    // name already carrying today's date is left alone.
    state.refresh_stream_name(&crate::Config::default_path());
    let mut quit = false;

    while !quit {
        // Tray menu commands.
        if let Some(rx) = &mut tray_rx {
            while let Ok(cmd) = rx.try_recv() {
                match cmd {
                    TrayCommand::Toggle => state.toggle(),
                    TrayCommand::Start => state.start(),
                    TrayCommand::Stop => state.stop(),
                    TrayCommand::SetMode(m) => state.set_mode(m),
                    TrayCommand::ShowWindow => state.open_viewer(),
                    TrayCommand::Quit => quit = true,
                }
            }
        }

        // IPC requests.
        while let Ok((req, reply)) = rx.try_recv() {
            let resp = match req {
                // Only reaches the loop as a keep-alive; the reply came from
                // the published snapshot.
                Request::GetState { .. } => {
                    state.last_viewer = Some(std::time::Instant::now());
                    Response::Ok
                }
                Request::Start => {
                    state.start();
                    Response::Ok
                }
                Request::Stop => {
                    state.stop();
                    Response::Ok
                }
                Request::Toggle => {
                    state.toggle();
                    Response::Ok
                }
                Request::SetMode { mode } => {
                    state.set_mode(mode);
                    Response::Ok
                }
                Request::SetDiarize { diarize } => {
                    state.set_diarize(diarize);
                    Response::Ok
                }
                Request::SetSource { key } => {
                    state.opts.source_keys = vec![key];
                    Response::Ok
                }
                Request::SetSources { keys, source_mode } => {
                    if state.running() {
                        Response::Error {
                            message: "stop the session before changing sources".into(),
                        }
                    } else {
                        state.opts.source_keys = keys;
                        if let Some(m) = source_mode {
                            state.opts.source_mode = m;
                        }
                        Response::Ok
                    }
                }
                Request::SetFormat { format } => {
                    // Held on the daemon because it owns the session that is
                    // streaming; a format that only travelled with Save
                    // requests left the stream writing whatever the daemon
                    // happened to start with.
                    state.opts.format = format;
                    state.opts.config.format = format;
                    if let Err(e) = state.opts.config.save(&crate::Config::default_path()) {
                        warn!("saving the config: {e:#}");
                    }
                    Response::Ok
                }
                Request::SetStreamFile { path } => {
                    state.set_stream_file(path, &crate::Config::default_path());
                    Response::Ok
                }
                Request::SetServer { server } => {
                    // Sessions read the URL at start, so this takes effect on
                    // the next one rather than disturbing a running session.
                    state.opts.config.url = server;
                    Response::Ok
                }
                Request::Save { format, path } => match state.save(format, path.as_deref()) {
                    Ok(p) => Response::Saved { path: p },
                    Err(e) => Response::Error {
                        message: format!("{e:#}"),
                    },
                },
                Request::TranscribeFile { path } => match state.transcribe_file(&path) {
                    Ok(()) => Response::Ok,
                    Err(e) => Response::Error {
                        message: format!("{e:#}"),
                    },
                },
                Request::SaveSplit { format } => match state.save_split(format) {
                    Ok(paths) => Response::Saved {
                        path: paths.join(", "),
                    },
                    Err(e) => Response::Error {
                        message: format!("{e:#}"),
                    },
                },
                Request::Clear => {
                    state.clear();
                    Response::Ok
                }
                Request::Quit => {
                    quit = true;
                    Response::Ok
                }
            };
            if let Some(r) = reply {
                let _ = r.send(resp);
            }
        }

        state.reap_finished_session();
        state.reap_file_job();
        state.update_preview();

        let mut snap = state.live_snapshot();
        // Settled at startup and constant thereafter, so it is stamped on
        // rather than carried through the session machinery.
        snap.hotkey = hotkey_report.clone();

        // Built from `snap` before it moves into `published`, so a whole
        // extra `DaemonState` clone is not paid every 25 ms just to hand the
        // tray three small fields.
        if let Some(h) = &tray_handle {
            h.update(TrayState {
                status: snap.status,
                mode: snap.mode,
                last_fragment: snap.last_fragment.clone(),
            });
        }
        state.publish_into(
            &mut published.lock().expect("published state poisoned"),
            snap,
        );

        // 25 ms rather than 100: the loop publishes the state a 30 Hz meter
        // reads, and doing almost nothing forty times a second is cheap.
        std::thread::sleep(std::time::Duration::from_millis(25));
    }

    state.stop();
    state.stop_overlay();
    let _ = std::fs::remove_file(&sock);
    info!("daemon stopped");
    Ok(())
}

/// What clients read, split by how often it changes.
///
/// `live` is republished every tick and is small. The transcript is not: at
/// two hours of meeting it is around a megabyte of JSON, and it is
/// byte-identical from one tick to the next almost every time. Keeping it
/// beside the live state rather than inside it means the loop replaces it
/// only when it moves, and a poll copies it only when the viewer asking has
/// not already got it.
///
/// One mutex over both, so a reply can never pair a live state with a
/// transcript from a different revision.
#[derive(Default)]
struct Published {
    /// Everything except the transcript. `live.revision` says which
    /// transcript the two fields below are.
    live: DaemonState,
    transcript: String,
    turns: Vec<(Option<u32>, String)>,
}

impl Published {
    /// The reply to a poll that last saw `since`.
    ///
    /// The text travels only when the viewer has not already got it. A
    /// revision of zero never counts as a match: it is what a daemon that
    /// does not track revisions reports, and treating it as one would leave
    /// a viewer showing an empty transcript for ever.
    fn state_since(&self, since: Option<u64>) -> DaemonState {
        let mut out = self.live.clone();
        if self.live.revision == 0 || since != Some(self.live.revision) {
            out.transcript = self.transcript.clone();
            out.turns = self.turns.clone();
        }
        out
    }
}

fn serve_client(
    stream: Stream,
    tx: mpsc::Sender<(Request, Option<mpsc::Sender<Response>>)>,
    published: Arc<Mutex<Published>>,
) {
    // A client that connects and then goes quiet must not tie up a thread.
    stream
        .set_recv_timeout(Some(std::time::Duration::from_secs(30)))
        .ok();
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
        return;
    }
    let resp = match serde_json::from_str::<Request>(&line) {
        // Answered straight from the published snapshot. Polling is the common
        // case by far, and routing it through the loop would cap the meter at
        // the loop rate.
        Ok(Request::GetState { since }) => {
            // Still tell the loop a viewer is here, so metering keeps running.
            let _ = tx.send((Request::GetState { since: None }, None));
            Response::State(
                published
                    .lock()
                    .expect("published state poisoned")
                    .state_since(since),
            )
        }
        Ok(req) => {
            let (rtx, rrx) = mpsc::channel();
            if tx.send((req, Some(rtx))).is_err() {
                Response::Error {
                    message: "daemon is shutting down".into(),
                }
            } else {
                rrx.recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap_or(Response::Error {
                        message: "daemon did not answer in time".into(),
                    })
            }
        }
        Err(e) => Response::Error {
            message: format!("undecodable request: {e}"),
        },
    };
    let mut w = stream;
    if let Ok(s) = serde_json::to_string(&resp) {
        let _ = writeln!(w, "{s}");
        let _ = w.flush();
    }
}

/// State of a file transcription.
#[derive(Default)]
struct FileJob {
    progress: f32,
    done: bool,
    text: String,
    error: Option<String>,
}

struct DaemonRuntime {
    opts: DaemonOptions,
    /// Running sessions. One in combined mode, one per source in separate mode.
    sessions: Vec<SessionHandle>,
    /// The level overlay, shown only for typing sessions.
    overlay: Option<std::process::Child>,
    /// Live level meter, running only while idle and only while a viewer is
    /// watching. Holding a capture open otherwise would keep a microphone
    /// active for no reason.
    preview: Option<crate::preview::Preview>,
    /// When a viewer last asked for state. Metering stops shortly after the
    /// last one closes.
    last_viewer: Option<std::time::Instant>,
    /// A file transcription in flight: its progress, and the text when done.
    file_job: Option<Arc<Mutex<FileJob>>>,
    /// The last finished session's state.
    ///
    /// Without this, stopping wipes the transcript from every viewer at exactly
    /// the moment the user wants to read or save it: the session is dropped and
    /// a snapshot falls back to an empty default. The text survives until the
    /// next session starts or it is explicitly cleared.
    last: crate::session::SessionState,
    /// Bumped whenever `sessions` or `last` is replaced rather than merely
    /// added to -- a session starting or being reaped, a transcript cleared,
    /// a file job folded in. Sessions only ever count upwards while they
    /// run, so this is what tells the cache below that the text moved
    /// backwards or sideways underneath it.
    generation: u64,
    text: TranscriptCache,
}

/// The transcript as viewers see it, rebuilt only when it changes.
///
/// Building it on every tick meant merging and cloning every segment forty
/// times a second and then regrouping them into turns -- at two hours of
/// meeting, milliseconds of work per tick to produce a byte-identical
/// answer. `token` is what makes "has it changed" answerable without
/// looking at the text at all.
#[derive(Default)]
struct TranscriptCache {
    /// Published to viewers, and monotonic, so no two different transcripts
    /// can ever share one -- a viewer that slept through a whole session and
    /// woke during the next cannot be told its stale copy is current.
    revision: u64,
    /// What `revision` was last computed from: the generation, how many
    /// sessions are running, and how many changes they have made between
    /// them. Every one is a counter read under a lock, so comparing costs
    /// nothing.
    token: (u64, usize, u64),
    transcript: String,
    turns: Vec<(Option<u32>, String)>,
}

impl DaemonRuntime {
    fn running(&self) -> bool {
        self.sessions.iter().any(|s| s.is_running())
    }

    /// Start, stop or re-point the level meter.
    ///
    /// Metering runs only while idle and only while a viewer is watching: a
    /// session already has the source open, and holding a capture for nobody
    /// would keep a microphone live with no indication.
    fn update_preview(&mut self) {
        const VIEWER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

        let watched = self
            .last_viewer
            .is_some_and(|t| t.elapsed() < VIEWER_TIMEOUT);
        let want = watched && !self.running();

        if !want {
            self.preview = None;
            return;
        }

        let key = self.opts.source_keys.first().cloned();
        // Re-point when the selection changes, so the meter always shows what
        // pressing Start would actually record.
        let stale = match (&self.preview, &key) {
            (Some(p), Some(k)) => &p.source_key != k,
            (Some(_), None) => true,
            (None, _) => true,
        };
        if !stale {
            return;
        }

        self.preview = None;
        let Ok(sources) = list_sources() else { return };
        let Ok(source) = choose_source(&sources, key.as_deref()) else {
            return;
        };
        match crate::preview::Preview::start(&source) {
            Ok(p) => self.preview = Some(p),
            // A source that cannot be metered is not fatal; Start will report
            // the real error.
            Err(e) => warn!("could not meter {}: {e:#}", source.display()),
        }
    }

    /// Rebuild the published transcript, if and only if it has moved.
    ///
    /// `changes` is the running total across live sessions, read from the
    /// counters they keep. It, the session count and the generation together
    /// change on every edit and on nothing else, so an unchanged token is
    /// proof the text is unchanged -- and the expensive part below, which
    /// clones every segment, merges them and groups them into turns, is
    /// skipped entirely.
    fn refresh_text(&mut self, changes: u64) {
        let token = (self.generation, self.sessions.len(), changes);
        if token == self.text.token {
            return;
        }
        let (transcript, turns) = {
            // Falls back to the last finished session rather than to an empty
            // default, so a stopped transcript stays on screen.
            let merged;
            let s = if self.sessions.is_empty() {
                &self.last
            } else {
                merged =
                    merge_states(&self.sessions.iter().map(|s| s.state()).collect::<Vec<_>>());
                &merged
            };
            // Turns only once something carries a speaker. Without labels
            // they are one turn holding the entire transcript over again,
            // and a viewer renders the flat text in that case anyway.
            let turns = if s.segments.iter().any(|seg| seg.speaker.is_some()) {
                save::turn_texts(&s.segments)
            } else {
                Vec::new()
            };
            (s.transcript.clone(), turns)
        };
        self.text.token = token;
        self.text.revision += 1;
        self.text.transcript = transcript;
        self.text.turns = turns;
    }

    /// Everything a viewer needs except the transcript itself.
    ///
    /// `transcript` and `turns` come back empty and `revision` says which
    /// ones they would have been; [`publish_into`](Self::publish_into) is
    /// what pairs them up. Splitting it this way is the point: the fields
    /// here move on every tick and are all small, and the transcript moves
    /// when somebody speaks and is not.
    fn live_snapshot(&mut self) -> DaemonState {
        // `live` leaves the segments where they are, so the
        // forty-times-a-second path never touches the transcript. The
        // transcript reaches viewers from the cache instead, and
        // `refresh_text` is what decides whether that cache needs rebuilding.
        let lives: Vec<crate::session::SessionState> =
            self.sessions.iter().map(|s| s.live()).collect();
        self.refresh_text(lives.iter().map(|s| s.changes).sum());

        // `merge_states` folds live states exactly as it folds full ones:
        // with no segments to interleave, everything it does to them is a
        // no-op, and the one merge rule stays in one place.
        let s = if lives.is_empty() {
            self.last.live()
        } else {
            merge_states(&lives)
        };
        let mut out = DaemonState::from_session(
            &s,
            self.opts.mode,
            self.opts.source_keys.first().cloned(),
        );
        out.revision = self.text.revision;
        out.source_keys = self.opts.source_keys.clone();
        out.source_mode = self.opts.source_mode;
        // The settings, not the session's answers about them: a viewer's
        // controls have to read what the next session would do. Stamped here
        // rather than by the caller so every reader of a snapshot -- the loop,
        // and the tests -- sees the same one.
        out.diarize_configured = self.opts.config.diarize;
        out.stream_to = self.opts.config.stream_to.clone();
        out.format = self.opts.format;

        // A file job owns the status while it runs, so a viewer shows progress
        // rather than an idle window with nothing happening.
        if let Some(job) = &self.file_job {
            let j = job.lock().expect("file job poisoned");
            if !j.done {
                out.status = crate::session::Status::Transcribing;
                out.progress = j.progress;
                // Bulk transcription never requests labels (see bulk.rs).
                out.diarize_requested = false;
            }
        }
        // Preview levels apply only when idle; a running session meters the
        // audio it is actually sending.
        if !self.running()
            && let Some(p) = &self.preview
        {
            out.levels = p.levels().to_vec();
            out.rms = p.rms();
        }
        out
    }

    /// Hand a fresh live state to viewers, with the transcript beside it.
    ///
    /// The transcript is copied across only when it has actually moved. It
    /// is the one big thing here -- around a megabyte of it after two hours
    /// of meeting -- and copying it forty times a second to say nothing new
    /// is the cost the revision exists to avoid.
    fn publish_into(&self, p: &mut Published, live: DaemonState) {
        if p.live.revision != live.revision {
            p.transcript = self.text.transcript.clone();
            p.turns = self.text.turns.clone();
        }
        p.live = live;
    }

    fn start(&mut self) {
        if self.running() {
            return;
        }
        let available = match list_sources() {
            Ok(s) => s,
            Err(e) => {
                error!("listing sources: {e:#}");
                return;
            }
        };

        // Resolve every remembered key; anything that has gone away is skipped
        // rather than failing the whole start.
        let mut resolved: Vec<crate::Source> = Vec::new();
        for key in &self.opts.source_keys {
            match crate::resolve(&available, key) {
                Some(s) => resolved.push(s),
                None => warn!("source {key} is no longer present; skipping"),
            }
        }
        if resolved.is_empty() {
            match choose_source(&available, None) {
                Ok(s) => resolved.push(s),
                Err(e) => {
                    error!("choosing a source: {e:#}");
                    return;
                }
            }
        }
        self.opts.source_keys = resolved.iter().map(|s| s.stable_key()).collect();

        // A new session starts from a blank transcript; the previous one has
        // had its chance to be read and saved.
        self.last = Default::default();
        self.generation += 1;
        self.start_overlay();

        let (url, token) = (self.opts.config.url.clone(), self.opts.config.token.clone());
        let inject = self.opts.config.inject;
        // Every way of starting comes through here -- the tray, the global
        // hotkey, Ctrl+D, `syrinx toggle` -- and only one of them has ever
        // seen a file dialog. Re-seeding the dialog's default name therefore
        // fixes almost none of them, so the generated name is refreshed at
        // the session that uses it.
        self.refresh_stream_name(&crate::Config::default_path());
        let stream = self
            .opts
            .config
            .stream_path()
            .map(|p| (p, self.opts.format));
        // Separate mode runs a session per source, and every session opens
        // its own StreamWriter. Pointing them all at one file used to be
        // described as interleaving in arrival order rather than tearing;
        // it tears. No writer is ever shown another's fragments, so none
        // knows to break the line for a source it cannot see, and they all
        // open on the same empty file so none writes the newline that would
        // have separated them -- two people's words run together on one
        // line, under two Speaker 1s their own clusterers minted
        // independently. See `two_writers_on_one_file_tear_the_records` in
        // stream.rs.
        //
        // A file each, named as `save_per_source` names them, so streaming
        // a conversation and saving it split agree. Only when there is more
        // than one source: a single writer has nobody to collide with, and
        // renaming its file would buy nothing.
        let split_stream = matches!(self.opts.source_mode, crate::mode::SourceMode::Separate)
            && resolved.len() > 1;
        match self.opts.source_mode {
            crate::mode::SourceMode::Combined => {
                self.sessions.push(crate::session::start(
                    SessionOptions {
                        url,
                        token,
                        sources: resolved,
                        mode: self.opts.mode,
                        diarize: self.opts.config.diarize,
                        // Attribution is meaningless once mixed.
                        label: None,
                        inject,
                        stream,
                        external_audio: None,
                    },
                    || {},
                ));
            }
            crate::mode::SourceMode::Separate => {
                // Named as a set rather than one at a time: `short_label`
                // calls every monitor "System audio", and two of those would
                // build one filename and put two writers on it -- the tearing
                // the paragraph above exists to prevent.
                let names = syrinx_audio::source::short_labels(&resolved);
                for (i, source) in resolved.into_iter().enumerate() {
                    // Only the first source may type: several streams typing
                    // into one cursor interleave into nonsense. The rest are
                    // transcribed and labelled.
                    let mode = if i == 0 {
                        self.opts.mode
                    } else {
                        OutputMode::Transcribe
                    };
                    let name = names[i].clone();
                    let stream = stream.as_ref().map(|(base, format)| {
                        let p = if split_stream {
                            save::path_for_source(base, &name)
                        } else {
                            base.clone()
                        };
                        info!("appending {name}'s transcript to {}", p.display());
                        (p, *format)
                    });
                    self.sessions.push(crate::session::start(
                        SessionOptions {
                            url: url.clone(),
                            token: token.clone(),
                            sources: vec![source],
                            mode,
                            diarize: self.opts.config.diarize,
                            label: Some(name),
                            inject,
                            stream,
                            external_audio: None,
                        },
                        || {},
                    ));
                }
            }
        }
    }

    fn stop(&mut self) {
        for s in &mut self.sessions {
            s.stop();
        }
    }

    /// Show the level overlay, for typing modes only.
    ///
    /// Transcribe mode already shows text arriving in its own window, which is
    /// feedback enough. Typing gives none -- text lands in whatever has focus,
    /// and silence is indistinguishable from a dead microphone. An always-on-top
    /// window over whatever you are reading has to earn its place, so it is not
    /// shown when it would be redundant.
    fn start_overlay(&mut self) {
        if !self.opts.mode.types_at_cursor() {
            return;
        }
        let Some(cmd) = &self.opts.gui_command else {
            return;
        };
        match std::process::Command::new(cmd)
            .arg("--overlay")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(c) => self.overlay = Some(c),
            Err(e) => warn!("could not show the level overlay: {e}"),
        }
    }

    /// Close the overlay. It also closes itself when the session ends, but a
    /// daemon shutting down should not leave one behind.
    fn stop_overlay(&mut self) {
        if let Some(mut c) = self.overlay.take() {
            let _ = c.kill();
        }
    }

    /// Start transcribing a file on a background thread.
    ///
    /// Runs through the same streaming session a microphone uses: the model
    /// consumes fixed chunks and emits as it goes, so a file is just chunks
    /// arriving faster than real time.
    fn transcribe_file(&mut self, path: &str) -> Result<()> {
        if self.running() {
            anyhow::bail!("stop the current session before transcribing a file");
        }
        if self.file_job.as_ref().is_some_and(|j| {
            !j.lock().expect("file job poisoned").done
        }) {
            anyhow::bail!("already transcribing a file");
        }

        let samples = crate::bulk::decode(std::path::Path::new(path))?;
        let job = Arc::new(Mutex::new(FileJob::default()));
        self.file_job = Some(job.clone());
        let (url, token) = (self.opts.config.url.clone(), self.opts.config.token.clone());

        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let mut j = job.lock().expect("file job poisoned");
                    j.error = Some(format!("building a runtime: {e}"));
                    j.done = true;
                    return;
                }
            };
            let j2 = job.clone();
            let result = rt.block_on(async move {
                crate::bulk::transcribe(&url, &token, &samples, |p| {
                    j2.lock().expect("file job poisoned").progress = p;
                })
                .await
            });
            let mut j = job.lock().expect("file job poisoned");
            match result {
                Ok(text) => j.text = text,
                Err(e) => j.error = Some(format!("{e:#}")),
            }
            j.done = true;
        });
        Ok(())
    }

    /// Fold a finished file job into the retained transcript.
    fn reap_file_job(&mut self) {
        let Some(job) = &self.file_job else { return };
        let (done, text, error) = {
            let j = job.lock().expect("file job poisoned");
            (j.done, j.text.clone(), j.error.clone())
        };
        if !done {
            return;
        }
        if let Some(e) = error {
            error!("file transcription failed: {e}");
            self.last.error = Some(e);
        } else {
            self.last.transcript = text;
            // No segments: a file is sent far faster than real time, so arrival
            // order says nothing about position in the recording. Timestamped
            // saving would produce times that are simply wrong.
            self.last.segments.clear();
        }
        self.generation += 1;
        self.file_job = None;
    }

    /// Discard the retained transcript. Ignored while a session is running,
    /// where clearing would fight with text still arriving.
    fn clear(&mut self) {
        if !self.running() {
            self.last = Default::default();
            self.generation += 1;
        }
    }

    fn toggle(&mut self) {
        if self.running() {
            self.stop();
        } else {
            self.start();
        }
    }

    /// Mode is fixed at session.start on the wire, so changing it mid-session
    /// would need a reconnect. Ignored while running rather than silently
    /// half-applied.
    fn set_mode(&mut self, m: OutputMode) {
        if !self.running() {
            self.opts.mode = m;
        }
    }

    /// Speaker labels are asked for in session.start on the wire too, so this
    /// is `set_mode` exactly: ignored while running rather than applied to a
    /// session that could only honour it by reconnecting.
    ///
    /// Written back to the config, unlike the mode. `diarize` reaches a session
    /// from `opts.config`, and the whole point of this control is to spare
    /// someone editing that file -- a checkbox whose effect died with the
    /// daemon would send them straight back to it.
    fn set_diarize(&mut self, on: bool) {
        if self.running() {
            return;
        }
        self.opts.config.diarize = on;
        if let Err(e) = self.opts.config.save(&crate::Config::default_path()) {
            warn!("saving the config: {e:#}");
        }
    }

    /// Where the next session appends its transcript, or nothing at all.
    ///
    /// Applied and written down, like `set_diarize`: the setting reaches a
    /// session through `opts.config`, and one that died with the daemon would
    /// have to be chosen again every time. Unlike `set_diarize` it is accepted
    /// while running -- the writer is opened at session start, so this can
    /// only ever describe the next session anyway.
    ///
    /// The config path is a parameter rather than `Config::default_path()`
    /// because this writes a real file: a test that called it would otherwise
    /// edit the settings of whoever ran it.
    fn set_stream_file(&mut self, path: Option<String>, config: &std::path::Path) {
        self.opts.config.stream_to = path;
        if let Err(e) = self.opts.config.save(config) {
            warn!("saving the config: {e:#}");
        }
    }

    /// Give a generated stream name today's date, before a session opens it.
    ///
    /// `stream_to` is persisted, so the name accepted from the Save dialog on
    /// the 20th is still the name on the 27th, and every session in between
    /// appended to `2026-08-20_09-14-03.txt`. Only a name of that shape is
    /// touched: `notes.txt` was chosen deliberately, and continuing it is the
    /// whole reason the setting is remembered.
    ///
    /// Written back rather than only used, so that two sessions on one day
    /// still meet in one file -- a fresh stamp per session would give each
    /// its own -- and so that every viewer's label names the file that is
    /// really being written.
    ///
    /// Restamping the string as it was written keeps a `~` a `~`; expanding
    /// it here would quietly rewrite the config to an absolute path.
    fn refresh_stream_name(&mut self, config: &std::path::Path) {
        let Some(current) = self.opts.config.stream_to.clone() else {
            return;
        };
        let current = std::path::PathBuf::from(current);
        let fresh = save::restamped(&current, &save::timestamp());
        if fresh != current {
            info!("streaming to {} for today", fresh.display());
            self.set_stream_file(Some(fresh.display().to_string()), config);
        }
    }

    fn save(&self, format: save::Format, path: Option<&str>) -> Result<String> {
        let st = if self.sessions.is_empty() {
            self.last.clone()
        } else {
            merge_states(&self.sessions.iter().map(|s| s.state()).collect::<Vec<_>>())
        };
        let p = save::save_rendered(
            path.map(std::path::Path::new),
            &st.segments,
            &st.transcript,
            format,
        )?;
        Ok(p.display().to_string())
    }

    /// Save one file per source, returning the paths written.
    fn save_split(&self, format: save::Format) -> Result<Vec<String>> {
        let st = if self.sessions.is_empty() {
            self.last.clone()
        } else {
            merge_states(&self.sessions.iter().map(|s| s.state()).collect::<Vec<_>>())
        };
        let base = save::default_dir().join(save::filename_for(&save::timestamp()));
        Ok(save::save_per_source(&base, &st.segments, format)?
            .into_iter()
            .map(|p| p.display().to_string())
            .collect())
    }

    /// Launch a viewer window.
    fn open_viewer(&self) {
        let Some(cmd) = &self.opts.gui_command else {
            info!("no viewer command configured");
            return;
        };
        match std::process::Command::new(cmd).spawn() {
            Ok(_) => info!("opened a viewer"),
            Err(e) => warn!("could not open {cmd}: {e}"),
        }
    }

    /// Notice a session that ended on its own, saving it if asked.
    fn reap_finished_session(&mut self) {
        if self.sessions.is_empty() || self.running() {
            return;
        }
        let st = merge_states(&self.sessions.iter().map(|s| s.state()).collect::<Vec<_>>());
        if self.opts.save_each && self.opts.mode.keeps_transcript() {
            match save::save_rendered(None, &st.segments, &st.transcript, self.opts.format) {
                Ok(p) => info!("saved to {}", p.display()),
                // Stopping without speaking is normal, not an error.
                Err(e) => info!("not saved: {e}"),
            }
        }
        if let Some(e) = &st.error {
            error!("session ended with an error: {e}");
        }
        // Keep the finished transcript visible and saveable.
        self.last = st;
        self.sessions.clear();
        self.generation += 1;
        self.stop_overlay();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Segment, SessionState};

    fn seg(at: f64, text: &str, speaker: Option<u32>, source: Option<&str>) -> Segment {
        Segment {
            at,
            text: text.into(),
            source: source.map(Into::into),
            speaker,
        }
    }

    fn config() -> Config {
        Config {
            url: "ws://127.0.0.1:8770/v1/stream".into(),
            token: "t".into(),
            source_key: None,
            mode: OutputMode::Transcribe,
            diarize: true,
            inject: Default::default(),
            stream_to: None,
            format: save::Format::Plain,
            hotkey: None,
            waybar_signal: 8,
        }
    }

    /// A daemon with nothing running, holding `last` -- the state a Save
    /// reaches once a session has ended, which is when a transcript is
    /// actually saved.
    fn daemon_holding(last: SessionState) -> DaemonRuntime {
        DaemonRuntime {
            opts: DaemonOptions {
                config: config(),
                mode: OutputMode::Transcribe,
                source_keys: Vec::new(),
                source_mode: Default::default(),
                save_each: false,
                format: save::Format::Plain,
                gui_command: None,
            },
            sessions: Vec::new(),
            overlay: None,
            preview: None,
            last_viewer: None,
            file_job: None,
            last,
            generation: 1,
            text: Default::default(),
        }
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!("syrinx-daemon-{tag}-{}.txt", std::process::id()))
    }

    #[test]
    fn saving_writes_the_labels_the_window_is_showing() {
        // stream.rs and save.rs test their renderers directly; this is the
        // daemon's own `save`, which is what a Save button actually reaches,
        // and it is the step between them that was suspected of dropping the
        // labels. Every format, because `Plain` prefixes turns by a different
        // route than the two stamped ones.
        let path = scratch("labels");
        let d = daemon_holding(SessionState {
            transcript: "we ship Thursday no we don't".into(),
            segments: vec![
                seg(0.0, "we ship Thursday", Some(1), None),
                seg(2.0, "no we don't", Some(2), None),
            ],
            ..Default::default()
        });

        for format in save::Format::ALL {
            let written = d.save(format, Some(&path.display().to_string())).unwrap();
            assert_eq!(written, path.display().to_string());
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(
                text.contains("Speaker 1: we ship Thursday"),
                "{format:?} lost the first label:\n{text}"
            );
            assert!(
                text.contains("Speaker 2: no we don't"),
                "{format:?} lost the second label:\n{text}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unlabelled_transcript_still_saves_as_it_always_did() {
        // The other half of the same guarantee: nothing may start prefixing a
        // recording that never had a speaker on it.
        let path = scratch("unlabelled");
        let d = daemon_holding(SessionState {
            transcript: "just me talking".into(),
            segments: vec![seg(0.0, "just me talking", None, None)],
            ..Default::default()
        });
        d.save(save::Format::Plain, Some(&path.display().to_string()))
            .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "just me talking\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn merging_sessions_keeps_every_speaker() {
        // Separate mode's path into the same save. A merge that dropped
        // `speaker` here would leave the GUI labelling from the live session
        // while every file came out bare, which is exactly the report.
        let merged = merge_states(&[
            SessionState {
                segments: vec![
                    seg(0.0, "mic one", Some(1), Some("Mic")),
                    seg(4.0, "mic two", Some(2), Some("Mic")),
                ],
                ..Default::default()
            },
            SessionState {
                segments: vec![seg(2.0, "system", Some(1), Some("System"))],
                ..Default::default()
            },
        ]);

        let got: Vec<(&str, Option<u32>)> = merged
            .segments
            .iter()
            .map(|s| (s.text.as_str(), s.speaker))
            .collect();
        assert_eq!(
            got,
            [("mic one", Some(1)), ("system", Some(1)), ("mic two", Some(2))],
            "the merge must interleave by time and carry every speaker"
        );
    }

    #[test]
    fn a_merged_save_carries_the_prefixes_too() {
        // `save` composes these two itself; a SessionHandle needs a live
        // connection, so this is as close to the running daemon as a test
        // reaches -- the composition is checked, the Vec<SessionHandle> that
        // feeds it is not.
        let path = scratch("merged");
        let merged = merge_states(&[
            SessionState {
                segments: vec![seg(0.0, "we ship Thursday", Some(1), Some("Mic"))],
                ..Default::default()
            },
            SessionState {
                segments: vec![seg(2.0, "no we don't", Some(1), Some("System"))],
                ..Default::default()
            },
        ]);
        save::save_rendered(
            Some(&path),
            &merged.segments,
            &merged.transcript,
            save::Format::Labelled,
        )
        .unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[Mic] Speaker 1: we ship Thursday"), "{text}");
        // Two sources' first speakers are two different people, so the second
        // opens a turn of its own rather than joining the first.
        assert!(text.contains("[System] Speaker 1: no we don't"), "{text}");
        let _ = std::fs::remove_file(&path);
    }

    /// A daemon holding two speakers' worth of transcript, which is the case
    /// every one of these is really about.
    fn daemon_holding_a_conversation() -> DaemonRuntime {
        daemon_holding(SessionState {
            transcript: "we ship Thursday no we don't".into(),
            segments: vec![
                seg(0.0, "we ship Thursday", Some(1), None),
                seg(2.0, "no we don't", Some(2), None),
            ],
            ..Default::default()
        })
    }

    /// One tick of the daemon loop, publishing into `p` exactly as `run`
    /// does -- so what these tests read is what a viewer would be sent.
    fn tick(d: &mut DaemonRuntime, p: &mut Published) {
        let snap = d.live_snapshot();
        d.publish_into(p, snap);
    }

    #[test]
    fn an_unchanged_transcript_is_not_sent_again() {
        // The whole point of the revision. A viewer that already has the text
        // gets a reply with the live fields and nothing else, so a two-hour
        // meeting stops crossing the socket thirty times a second.
        let mut d = daemon_holding_a_conversation();
        let mut published = Published::default();
        tick(&mut d, &mut published);
        let rev = published.live.revision;
        assert_ne!(rev, 0, "a live daemon must report a real revision");

        let fresh = published.state_since(None);
        assert_eq!(fresh.transcript, "we ship Thursday no we don't");
        assert_eq!(fresh.turns.len(), 2);

        let repeat = published.state_since(Some(rev));
        assert!(repeat.transcript.is_empty(), "the text travelled twice");
        assert!(repeat.turns.is_empty(), "the turns travelled twice");
        // The live fields still come through, which is what a poll is for.
        assert_eq!(repeat.revision, rev);
        assert_eq!(repeat.status, fresh.status);

        // An older revision is not a match, so the text comes back.
        assert!(!published.state_since(Some(rev - 1)).transcript.is_empty());
    }

    #[test]
    fn a_zero_revision_is_never_read_as_unchanged() {
        // Zero is what a daemon too old to track revisions reports. If it
        // ever counted as a match, a viewer would hold an empty transcript
        // and never be sent another.
        let published = Published {
            live: DaemonState::default(),
            transcript: "words".into(),
            turns: Vec::new(),
        };
        assert_eq!(published.live.revision, 0);
        assert_eq!(published.state_since(Some(0)).transcript, "words");
    }

    #[test]
    fn the_revision_holds_still_while_the_transcript_does() {
        // Ticking without speaking must not invent a new revision: every one
        // costs every viewer a fresh copy of the whole meeting.
        let mut d = daemon_holding_a_conversation();
        let mut p = Published::default();
        tick(&mut d, &mut p);
        let first = p.live.revision;
        for _ in 0..10 {
            tick(&mut d, &mut p);
            assert_eq!(p.live.revision, first);
        }
        // And the text is still there to be handed out, not merely unsent.
        assert_eq!(p.transcript, "we ship Thursday no we don't");
    }

    #[test]
    fn clearing_the_transcript_moves_the_revision_on() {
        // And it must move when the text does, including when it is replaced
        // rather than added to -- a cleared transcript is a change a viewer
        // has to be told about, and it makes nothing longer.
        let mut d = daemon_holding_a_conversation();
        let mut p = Published::default();
        tick(&mut d, &mut p);
        let before = p.live.revision;
        d.clear();
        tick(&mut d, &mut p);
        assert!(p.live.revision > before, "{before} -> {}", p.live.revision);
        assert!(p.transcript.is_empty());
        assert!(p.turns.is_empty());
    }

    #[test]
    fn turns_are_published_ready_to_render() {
        // The viewer does no grouping of its own any more, so what arrives
        // has to be exactly what it used to compute for itself.
        let mut d = daemon_holding_a_conversation();
        let mut p = Published::default();
        tick(&mut d, &mut p);
        assert_eq!(
            p.turns,
            vec![
                (Some(1), "we ship Thursday".to_string()),
                (Some(2), "no we don't".to_string()),
            ]
        );
        assert_eq!(p.turns, save::turn_texts(&d.last.segments));
    }

    #[test]
    fn an_unlabelled_transcript_publishes_no_turns_at_all() {
        // Without a speaker anywhere, the turns are one turn holding the
        // whole transcript over again, and the viewer renders the flat text
        // instead. Sending them would send every word twice.
        let mut d = daemon_holding(SessionState {
            transcript: "just me talking".into(),
            segments: vec![seg(0.0, "just me talking", None, None)],
            ..Default::default()
        });
        let mut p = Published::default();
        tick(&mut d, &mut p);
        assert_eq!(p.transcript, "just me talking");
        assert!(
            p.turns.is_empty(),
            "nothing is labelled, so there is nothing to lay out in turns"
        );
    }

    #[test]
    fn a_live_state_carries_everything_except_the_transcript() {
        // What the 40 Hz path reads. Dropping a field from it by accident
        // would leave a viewer with a dead meter or a stale status, and the
        // saving is only worth having because nothing a viewer shows lives
        // in the two fields left behind.
        let full = SessionState {
            status: crate::session::Status::Listening,
            transcript: "words".into(),
            segments: vec![seg(0.0, "words", Some(1), None)],
            last_fragment: "words".into(),
            model: Some("nemotron".into()),
            chunk_ms: Some(560),
            diarize: true,
            diarize_requested: true,
            error: Some("boom".into()),
            levels: vec![0.5; 10],
            rms: 0.25,
            changes: 7,
        };
        let live = full.live();
        assert!(live.transcript.is_empty() && live.segments.is_empty());
        assert_eq!(
            (live.status, live.model.clone(), live.chunk_ms, live.changes),
            (full.status, full.model.clone(), full.chunk_ms, full.changes)
        );
        assert_eq!(
            (live.diarize, live.diarize_requested, live.rms),
            (full.diarize, full.diarize_requested, full.rms)
        );
        assert_eq!((live.error, live.levels), (full.error, full.levels));
        assert_eq!(live.last_fragment, full.last_fragment);
    }

    #[test]
    fn a_chosen_stream_file_is_applied_and_written_down() {
        // Nothing covered this handler end to end, and it has two halves that
        // can fail separately: the next session reads `opts.config`, and every
        // viewer reads the published state.
        let config_path = scratch("streamfile").with_extension("toml");
        let _ = std::fs::remove_file(&config_path);
        let mut d = daemon_holding(SessionState::default());
        let mut p = Published::default();

        d.set_stream_file(Some("/tmp/notes.txt".into()), &config_path);
        tick(&mut d, &mut p);

        assert_eq!(d.opts.config.stream_to.as_deref(), Some("/tmp/notes.txt"));
        assert_eq!(
            p.live.stream_to.as_deref(),
            Some("/tmp/notes.txt"),
            "a window has to see what it just asked for"
        );
        let written = std::fs::read_to_string(&config_path).unwrap();
        assert!(written.contains("/tmp/notes.txt"), "not persisted:\n{written}");

        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn stopping_the_stream_takes_the_setting_away_for_good() {
        // A cleared setting that survived in the file would come back on the
        // next start, and a transcript would be appended to a file nobody
        // asked for. `clearing_an_optional_setting_removes_it` covers the
        // config half; this is the daemon reaching it.
        let config_path = scratch("streamstop").with_extension("toml");
        let _ = std::fs::remove_file(&config_path);
        let mut d = daemon_holding(SessionState::default());
        let mut p = Published::default();

        d.set_stream_file(Some("/tmp/notes.txt".into()), &config_path);
        d.set_stream_file(None, &config_path);
        tick(&mut d, &mut p);

        assert_eq!(d.opts.config.stream_to, None);
        assert_eq!(p.live.stream_to, None);
        let written = std::fs::read_to_string(&config_path).unwrap();
        assert!(!written.contains("/tmp/notes.txt"), "still there:\n{written}");

        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn a_generated_stream_name_from_another_day_is_refreshed_before_a_session() {
        // The complaint. Only the GUI's Stream button ever opens a file
        // dialog, so re-seeding the dialog left the tray, the global hotkey,
        // Ctrl+D and `syrinx start` still appending today's meeting to a file
        // named for a fortnight ago. This runs at every session, whichever
        // asked for it.
        let config_path = scratch("streamrestamp").with_extension("toml");
        let _ = std::fs::remove_file(&config_path);
        let mut d = daemon_holding(SessionState::default());
        d.opts.config.stream_to = Some("/tmp/2026-08-20_09-14-03.txt".into());

        d.refresh_stream_name(&config_path);

        let now = d.opts.config.stream_to.clone().expect("still streaming");
        assert!(now.starts_with("/tmp/"), "the folder was chosen: {now}");
        assert!(
            now.contains(&save::timestamp()[..10]),
            "not today's date: {now}"
        );
        let written = std::fs::read_to_string(&config_path).unwrap();
        assert!(written.contains(&now), "not persisted:\n{written}");
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn two_sessions_on_one_day_still_meet_in_one_file() {
        // The refreshed name is written back rather than only used, so the
        // second session of the day finds a name that is already today's and
        // continues it. A stamp minted per session would give each its own
        // file and lose "stopping and starting continues where you left off".
        let config_path = scratch("streamsameday").with_extension("toml");
        let _ = std::fs::remove_file(&config_path);
        let mut d = daemon_holding(SessionState::default());
        d.opts.config.stream_to = Some("/tmp/2026-08-20_09-14-03.txt".into());

        d.refresh_stream_name(&config_path);
        let first = d.opts.config.stream_to.clone().unwrap();
        d.refresh_stream_name(&config_path);

        assert_eq!(d.opts.config.stream_to.as_deref(), Some(first.as_str()));
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn a_stream_file_the_user_named_is_never_renamed() {
        // `notes.txt` was chosen on purpose, and continuing it is why the
        // setting is remembered at all. Nothing is written either: an
        // untouched setting is not a reason to rewrite the config file.
        let config_path = scratch("streamkept").with_extension("toml");
        let _ = std::fs::remove_file(&config_path);
        let mut d = daemon_holding(SessionState::default());
        d.opts.config.stream_to = Some("~/transcripts/notes.txt".into());

        d.refresh_stream_name(&config_path);

        assert_eq!(
            d.opts.config.stream_to.as_deref(),
            Some("~/transcripts/notes.txt")
        );
        assert!(!config_path.exists(), "the config was rewritten for nothing");
    }

    #[test]
    fn a_viewer_is_told_what_the_next_session_would_ask_for() {
        // The checkbox reads this. It is not `diarize` or `diarize_requested`:
        // both are false with nothing running, and the setting is on.
        let mut d = daemon_holding(SessionState::default());
        let snap = d.live_snapshot();
        assert!(snap.diarize_configured);
        assert!(!snap.diarize);
        assert!(!snap.diarize_requested);
    }
}
