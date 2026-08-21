//! The background daemon: owns the session, the tray, and the IPC socket.
//!
//! Everything persistent lives here. The GUI attaches as a viewer and may come
//! and go; closing its window does not stop dictation, because the daemon never
//! had a window to lose.

use crate::ipc::{self, DaemonState, Request, Response};
use crate::mode::OutputMode;
use crate::save;
use crate::session::{SessionHandle, SessionOptions};
use crate::tray::{TrayCommand, TrayState};
use crate::{Config, choose_source, list_sources};
use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex, mpsc};
use tracing::{error, info, warn};

/// What the daemon does when a session ends by itself.
pub struct DaemonOptions {
    pub config: Config,
    pub mode: OutputMode,
    pub source_key: Option<String>,
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
    if sock.exists() {
        anyhow::bail!(
            "a syrinx daemon is already running (socket at {}). Use `syrinx stop` \
             or quit it from the tray.",
            sock.display()
        );
    }
    let listener = UnixListener::bind(&sock)
        .with_context(|| format!("binding the daemon socket at {}", sock.display()))?;

    // A channel per source of commands, so the main loop has one place to look.
    let (tx, rx) = mpsc::channel::<(Request, Option<mpsc::Sender<Response>>)>();

    // Latest state, published by the loop and read directly by clients.
    //
    // GetState used to travel through the same channel as commands, so a poll
    // waited for the next loop tick before it was even looked at. Reading a
    // published snapshot decouples how fresh state is from how fast the loop
    // spins, which is what a 30 Hz meter needs.
    let published: Arc<Mutex<DaemonState>> = Arc::new(Mutex::new(DaemonState::default()));

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

    let tray = crate::tray::start();
    let (tray_handle, mut tray_rx) = match tray {
        Some((h, rx)) => (Some(h), Some(rx)),
        None => {
            info!("no system tray available; running headless");
            (None, None)
        }
    };

    let mut state = DaemonRuntime {
        opts,
        session: None,
        overlay: None,
        file_job: None,
        preview: None,
        last_viewer: None,
        last: Default::default(),
    };

    // Resolve the source up front so viewers show what would actually be used
    // rather than the word "default". Enumerating is a subprocess call, so it
    // happens once here rather than on every state poll.
    if state.opts.source_key.is_none()
        && let Ok(sources) = list_sources()
        && let Ok(s) = choose_source(&sources, None)
    {
        state.opts.source_key = Some(s.stable_key());
    }
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
                Request::GetState => {
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
                Request::SetSource { key } => {
                    state.opts.source_key = Some(key);
                    Response::Ok
                }
                Request::SetUrl { url } => {
                    // Sessions read the URL at start, so this takes effect on
                    // the next one rather than disturbing a running session.
                    state.opts.config.url = url;
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

        let snap = state.snapshot();
        *published.lock().expect("published state poisoned") = snap.clone();

        if let Some(h) = &tray_handle {
            h.update(TrayState {
                status: snap.status,
                mode: snap.mode,
                last_fragment: snap.last_fragment.clone(),
            });
        }

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

fn serve_client(
    stream: UnixStream,
    tx: mpsc::Sender<(Request, Option<mpsc::Sender<Response>>)>,
    published: Arc<Mutex<DaemonState>>,
) {
    let peer = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            warn!("cloning a client socket: {e}");
            return;
        }
    };
    let mut reader = BufReader::new(peer);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
        return;
    }
    let resp = match serde_json::from_str::<Request>(&line) {
        // Answered straight from the published snapshot. Polling is the common
        // case by far, and routing it through the loop would cap the meter at
        // the loop rate.
        Ok(Request::GetState) => {
            // Still tell the loop a viewer is here, so metering keeps running.
            let _ = tx.send((Request::GetState, None));
            Response::State(published.lock().expect("published state poisoned").clone())
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
    session: Option<SessionHandle>,
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
}

impl DaemonRuntime {
    fn running(&self) -> bool {
        self.session.as_ref().is_some_and(|s| s.is_running())
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

        let key = self.opts.source_key.clone();
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

    fn snapshot(&self) -> DaemonState {
        // Falls back to the last finished session rather than to an empty
        // default, so a stopped transcript stays on screen.
        let s = self
            .session
            .as_ref()
            .map(|s| s.state())
            .unwrap_or_else(|| self.last.clone());
        let mut out = DaemonState::from_session(&s, self.opts.mode, self.opts.source_key.clone());

        // A file job owns the status while it runs, so a viewer shows progress
        // rather than an idle window with nothing happening.
        if let Some(job) = &self.file_job {
            let j = job.lock().expect("file job poisoned");
            if !j.done {
                out.status = crate::session::Status::Transcribing;
                out.progress = j.progress;
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

    fn start(&mut self) {
        if self.running() {
            return;
        }
        let sources = match list_sources() {
            Ok(s) => s,
            Err(e) => {
                error!("listing sources: {e:#}");
                return;
            }
        };
        let source = match choose_source(&sources, self.opts.source_key.as_deref()) {
            Ok(s) => s,
            Err(e) => {
                error!("choosing a source: {e:#}");
                return;
            }
        };
        // Remember what was actually used, so a viewer shows the real source
        // rather than the preference that may not have resolved.
        self.opts.source_key = Some(source.stable_key());
        // A new session starts from a blank transcript; the previous one has
        // had its chance to be read and saved.
        self.last = Default::default();
        self.start_overlay();
        self.session = Some(crate::session::start(
            SessionOptions {
                url: self.opts.config.url.clone(),
                token: self.opts.config.token.clone(),
                source,
                mode: self.opts.mode,
            },
            || {},
        ));
    }

    fn stop(&mut self) {
        if let Some(s) = &mut self.session {
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
        self.file_job = None;
    }

    /// Discard the retained transcript. Ignored while a session is running,
    /// where clearing would fight with text still arriving.
    fn clear(&mut self) {
        if !self.running() {
            self.last = Default::default();
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

    fn save(&self, format: save::Format, path: Option<&str>) -> Result<String> {
        let st = self
            .session
            .as_ref()
            .map(|s| s.state())
            .unwrap_or_else(|| self.last.clone());
        let p = save::save_rendered(
            path.map(std::path::Path::new),
            &st.segments,
            &st.transcript,
            format,
        )?;
        Ok(p.display().to_string())
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
        let Some(s) = &self.session else { return };
        if s.is_running() {
            return;
        }
        let st = s.state();
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
        self.session = None;
        self.stop_overlay();
    }
}
