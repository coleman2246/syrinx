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
use std::sync::mpsc;
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

    // Accept loop: one thread per connection, which is ample for a handful of
    // short-lived requests.
    {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(s) => {
                        let tx = tx.clone();
                        std::thread::spawn(move || serve_client(s, tx));
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
    };
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
                Request::GetState => Response::State(state.snapshot()),
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
                Request::Save { format, path } => match state.save(format, path.as_deref()) {
                    Ok(p) => Response::Saved { path: p },
                    Err(e) => Response::Error {
                        message: format!("{e:#}"),
                    },
                },
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

        if let Some(h) = &tray_handle {
            let snap = state.snapshot();
            h.update(TrayState {
                status: snap.status,
                mode: snap.mode,
                last_fragment: snap.last_fragment,
            });
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    state.stop();
    let _ = std::fs::remove_file(&sock);
    info!("daemon stopped");
    Ok(())
}

fn serve_client(stream: UnixStream, tx: mpsc::Sender<(Request, Option<mpsc::Sender<Response>>)>) {
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

struct DaemonRuntime {
    opts: DaemonOptions,
    session: Option<SessionHandle>,
}

impl DaemonRuntime {
    fn running(&self) -> bool {
        self.session.as_ref().is_some_and(|s| s.is_running())
    }

    fn snapshot(&self) -> DaemonState {
        let s = self
            .session
            .as_ref()
            .map(|s| s.state())
            .unwrap_or_default();
        DaemonState::from_session(&s, self.opts.mode, self.opts.source_key.clone())
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
            .unwrap_or_default();
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
        if let Some(e) = st.error {
            error!("session ended with an error: {e}");
        }
        self.session = None;
    }
}
