//! A window onto the syrinx daemon.
//!
//! The GUI owns no session. The daemon does, along with the tray icon, and this
//! attaches to it over a Unix socket. That is what lets the window be closed
//! without stopping dictation: winit documents `set_visible` as unsupported on
//! Wayland, so a window cannot hide itself and keep running, and something that
//! never had a window has to hold the session instead.
//!
//! Starting the GUI starts a daemon if none is running, so opening the window is
//! enough to get a tray, and closing it leaves both alive. The daemon is
//! detached from whatever started it -- a console on Windows, a session on Unix
//! -- so closing the terminal the GUI was launched from does not take dictation
//! with it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod overlay;
mod theme;

use anyhow::{Context, Result};
use eframe::egui;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use syrinx_client::{
    Config, OutputMode, Source, SourceKind, ipc,
    ipc::{DaemonState, Request, Response},
    mode::SourceMode,
    save,
    session::Status,
};

/// How often the daemon is polled.
///
/// 30 Hz, because the level meter is in this window too and 200 ms made it
/// step visibly. A round trip over the Unix socket measures about 0.1 ms, so
/// polling thirty times a second costs roughly 0.3% of one core -- the meter is
/// worth that, and the alternative is a display that lies about how live it is.
const POLL_INTERVAL: Duration = Duration::from_millis(33);

/// How often the source list is re-scanned.
///
/// Applications only exist in the graph while they are playing, so a list read
/// once at startup never shows an app that started afterwards -- which looks
/// like per-application capture is missing rather than merely stale. Slower
/// than the state poll because scanning shells out to `pw-dump`.
const SOURCE_RESCAN_INTERVAL: Duration = Duration::from_secs(2);

fn main() {
    if let Err(e) = run() {
        report_fatal(&e);
        std::process::exit(1);
    }
}

/// Show a failure that happened before there was a window to show it in.
///
/// A double-clicked application has no console to print to, so printing alone
/// means it simply fails to appear -- no window, no message, nothing to act on.
/// A bad token or an unreachable server would look identical to a broken
/// download.
fn report_fatal(e: &anyhow::Error) {
    let text = format!("{e:#}");
    tracing::error!("{text}");
    eprintln!("Error: {text}");
    rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title("Syrinx could not start")
        .set_description(text)
        .show();
}

fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "syrinx_gui=info,syrinx_client=info".into()),
        )
        .init();

    // The overlay is a readout over an existing daemon, not a viewer that
    // starts one: it only makes sense while a session is already running.
    if std::env::args().any(|a| a == "--overlay") {
        return overlay::run();
    }

    let config = Config::load(None)?;
    ensure_daemon()?;

    // Checked after the daemon, deliberately. Starting one is useful even
    // where a window cannot open -- a tray, a hotkey and `syrinx toggle` are
    // not a window -- and it is what makes the message below true rather
    // than a guess.
    if let Some(why) = no_window_here() {
        anyhow::bail!("{why}");
    }

    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([480.0, 380.0])
        .with_min_inner_size([400.0, 280.0])
        .with_title("Syrinx");
    let options = eframe::NativeOptions {
        viewport: match window_icon() {
            Some(icon) => viewport.with_icon(icon),
            None => viewport,
        },
        ..Default::default()
    };

    eframe::run_native(
        "Syrinx",
        options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(App::new(config)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// The icon the window, the taskbar and alt-tab show.
///
/// Built into the binary with `include_bytes!` rather than read from disk, for
/// the same reason the tray icon next door is generated rather than shipped
/// (see `icon_for` in syrinx-client): there is no asset pipeline here, and a
/// file loaded at runtime is one more thing an install can turn out to be
/// missing.
///
/// Decoding is allowed to fail. An icon is decoration and the window is the
/// product, so a PNG this build cannot read costs the window its icon and
/// nothing else -- the same trade the diarizer makes when a fault degrades the
/// speaker labels rather than the transcript.
///
/// The decoder is eframe's own, which spares this crate a dependency: `image`
/// with its `png` feature is already non-optional in eframe, so the code to
/// read a PNG is linked in whether or not this calls it.
fn window_icon() -> Option<egui::IconData> {
    match eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon-256.png")) {
        Ok(icon) => Some(icon),
        Err(e) => {
            tracing::warn!("the window icon did not decode, carrying on without one: {e}");
            None
        }
    }
}

/// Why no window can open here, if none can.
///
/// Run from a TTY or over SSH, winit answers `neither WAYLAND_DISPLAY nor
/// WAYLAND_SOCKET nor DISPLAY is set`. That is true, and it names three
/// variables and no next step -- and it reads as total failure at the one
/// moment when it is furthest from true: the daemon is started before the
/// window and outlives it, so dictation is already running by the time this
/// can be reported.
///
/// Read from the environment rather than from what `run_native` hands back.
/// The environment is what decides the outcome -- with no X11 display and no
/// Wayland socket there is no backend winit could succeed with -- so
/// checking it is exact, and it needs no guesses about which error variant
/// or wording a future eframe will use.
fn no_window_here() -> Option<String> {
    // Only Wayland and X11 work this way. Windows and macOS have no such
    // variables, and a window that fails to open there has some other cause
    // that this must not claim to explain.
    if !cfg!(all(unix, not(target_os = "macos"))) {
        return None;
    }
    // Empty counts as unset, which is how the display libraries read it too.
    let set = |k: &str| std::env::var_os(k).is_some_and(|v| !v.is_empty());
    no_window_advice(
        set("WAYLAND_DISPLAY") || set("WAYLAND_SOCKET") || set("DISPLAY"),
        ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"]
            .iter()
            .any(|k| set(k)),
    )
}

/// The wording, kept apart from reading the environment so that what a user
/// is told can be checked without a test reaching into the process it runs
/// in.
fn no_window_advice(graphical: bool, ssh: bool) -> Option<String> {
    if graphical {
        return None;
    }
    let mut why = String::from(
        "this shell has no graphical session, so there is nowhere to put a window.\n\n\
         Dictation is running regardless: the daemon started before this and is \
         listening now. Use the tray icon, the hotkey from your config, or \
         `syrinx toggle` from a script or key binding.\n\n\
         For the window, start syrinx-gui from a terminal inside the desktop session.",
    );
    if ssh {
        why.push_str(
            "\n\nThis is an SSH session, so the window could only ever have opened on \
             that machine's own screen: Wayland has no equivalent of X11 forwarding, and \
             there is nothing here to forward. The daemon is on the far machine too, \
             which is where the microphone is.",
        );
    }
    Some(why)
}

/// Start a daemon if none is listening, and wait for its socket.
fn ensure_daemon() -> Result<()> {
    if ipc::daemon_running() {
        return Ok(());
    }
    ipc::clear_stale_socket();

    // Same directory as this binary, so a viewer and its daemon come from one
    // build rather than whatever is first on PATH.
    let cmd = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().and_then(daemon_beside))
        .unwrap_or_else(|| PathBuf::from("syrinx"));

    tracing::info!("starting the syrinx daemon");

    // The daemon's output goes to a file rather than to /dev/null. It used to
    // be discarded, so a daemon that refused to start -- a bad config, a port
    // already taken, another daemon already running -- produced only "did not
    // start listening within 5s", with the actual reason thrown away.
    let log_path = daemon_log_path();
    let log = std::fs::File::create(&log_path)
        .with_context(|| format!("creating the daemon log at {}", log_path.display()))?;
    let mut command = std::process::Command::new(&cmd);
    command
        .arg("daemon")
        // Detached from this window's stdin: the daemon must outlive it.
        .stdin(std::process::Stdio::null())
        .stdout(
            log.try_clone()
                .context("duplicating the daemon log handle")?,
        )
        .stderr(log);
    detach(&mut command);
    let mut child = command
        .spawn()
        .with_context(|| format!("starting the daemon via {}", cmd.display()))?;

    // Binding the socket takes a moment; poll rather than guess at a sleep.
    for _ in 0..50 {
        if ipc::daemon_running() {
            // Reaped in the background. Without this the daemon becomes a
            // zombie the moment it exits and stays one for as long as this
            // window is open.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            return Ok(());
        }
        // Died before it ever listened: say why rather than time out.
        if let Ok(Some(status)) = child.try_wait() {
            anyhow::bail!(
                "the daemon exited immediately ({status}).\n{}",
                tail(&log_path)
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    anyhow::bail!(
        "the daemon did not start listening within 5s.\n{}",
        tail(&log_path)
    )
}

/// The daemon binary sitting next to this one, if it is there.
///
/// The extension is the whole point. This looked for a bare `syrinx`, which on
/// Windows is never the sibling's name -- so the check above it failed every
/// time and the launch fell through to `syrinx` on PATH, which is exactly the
/// older install the comment there means to avoid. A fresh window driving a
/// stale daemon looks like a bug in whatever the daemon does: it shows the
/// state and the transcript, so only the parts it writes -- saved and streamed
/// files -- come out of date.
fn daemon_beside(dir: &std::path::Path) -> Option<PathBuf> {
    let p = dir.join(format!("syrinx{}", std::env::consts::EXE_SUFFIX));
    p.exists().then_some(p)
}

/// Cut the daemon loose from whatever started this window.
///
/// Without it the daemon dies with the terminal. On Windows a child inherits
/// its parent's console and receives `CTRL_CLOSE_EVENT` when that console is
/// closed; on Unix a shell sends `SIGHUP` to its jobs when it exits. Either way
/// closing the window you launched from would stop dictation, which is the
/// opposite of the point of having a daemon.
#[cfg(windows)]
fn detach(command: &mut std::process::Command) {
    use std::os::windows::process::CommandExt;
    /// The child gets no console of its own and does not inherit ours, so no
    /// window flashes up and console close events cannot reach it.
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    /// Also out of our Ctrl-C group.
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
}

#[cfg(unix)]
fn detach(command: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    // A new session, not merely a new process group: a background job in the
    // same session is still sent SIGHUP when the shell exits.
    unsafe {
        command.pre_exec(|| {
            // Failure here is not fatal -- the daemon still runs, it is just
            // still attached -- so the error is swallowed rather than aborting
            // a process that is mid-exec.
            libc::setsid();
            Ok(())
        });
    }
}

#[cfg(not(any(windows, unix)))]
fn detach(_command: &mut std::process::Command) {}

/// Where the daemon's output is kept, beside its PID file so it is per-user.
fn daemon_log_path() -> PathBuf {
    syrinx_client::state::default_pid_path().with_extension("log")
}

/// The last few lines of the daemon log, for an error message.
fn tail(path: &std::path::Path) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return format!("(no output; see {})", path.display());
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return format!("(no output; see {})", path.display());
    }
    let start = lines.len().saturating_sub(5);
    lines[start..].join("\n")
}

/// The little round state indicator, painted rather than written.
///
/// It used to be `●` and `○` in a label, which renders as a missing-glyph box
/// on Windows -- the fonts egui bundles do not reliably cover that block there,
/// and it has no system fallback to reach for. A circle from the painter
/// depends on no font at all, so it cannot fail that way on any machine.
///
/// Allocating the space rather than painting free-hand keeps it in the flow of
/// the horizontal layout it sits in, so the label beside it stays vertically
/// centred against it.
fn state_dot(ui: &mut egui::Ui, colour: egui::Color32, filled: bool) {
    // Matches the 18pt glyph it replaces closely enough that no row moved.
    const SIZE: f32 = 12.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(SIZE, SIZE), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    if filled {
        painter.circle_filled(rect.center(), SIZE / 2.0 - 1.0, colour);
    } else {
        // Hollow, as `○` was: idle should read as an indicator that is off,
        // not as a second lit one in a duller colour.
        painter.circle_stroke(
            rect.center(),
            SIZE / 2.0 - 1.5,
            egui::Stroke::new(1.5, colour),
        );
    }
}

/// Render key/description pairs as an aligned two-column grid.
fn keys_table(ui: &mut egui::Ui, id: &str, rows: &[(&str, &str)]) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            for (key, what) in rows {
                ui.label(egui::RichText::new(*key).monospace().strong());
                ui.label(*what);
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod ensure_daemon_tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("syrinx-gui-test-{}-{tag}", std::process::id()))
    }

    #[test]
    fn a_missing_log_still_points_somewhere_useful() {
        // The daemon can fail before it writes anything. An empty message
        // would leave the user with nothing at all to go on.
        let p = scratch("missing");
        let _ = std::fs::remove_file(&p);
        let out = tail(&p);
        assert!(out.contains(&p.display().to_string()), "got: {out}");
    }

    #[test]
    fn an_empty_log_is_treated_as_no_output() {
        let p = scratch("empty");
        std::fs::write(&p, "   \n\n").unwrap();
        assert!(tail(&p).contains("no output"), "got: {}", tail(&p));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn only_the_last_few_lines_are_shown() {
        // A daemon that logged for an hour before dying should not paste an
        // hour of logs into an error dialog.
        let p = scratch("long");
        let body: String = (0..100).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&p, body).unwrap();
        let out = tail(&p);
        assert!(out.contains("line 99"), "the end is what matters: {out}");
        assert!(!out.contains("line 50"), "too much context: {out}");
        assert_eq!(out.lines().count(), 5);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_sibling_daemon_is_found_under_its_platform_name() {
        // The bug this covers is invisible on Linux, where EXE_SUFFIX is
        // empty: only on Windows did the old bare `syrinx` miss the sibling
        // and send every launch to PATH instead.
        let dir = scratch("beside");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            daemon_beside(&dir),
            None,
            "an empty directory has no daemon"
        );

        let exe = dir.join(format!("syrinx{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&exe, b"").unwrap();
        assert_eq!(daemon_beside(&dir), Some(exe));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_graphical_session_is_left_alone() {
        // The overwhelmingly common case, including SSH with X11 forwarding
        // set up, where DISPLAY is present and a window really does open.
        assert_eq!(no_window_advice(true, false), None);
        assert_eq!(no_window_advice(true, true), None);
    }

    #[test]
    fn a_tty_is_told_that_dictation_is_running_anyway() {
        // The point of the whole message. winit's version reads as a total
        // failure, and the daemon has already started by then -- so the one
        // thing a user most needs to know is the thing it never said.
        let why = no_window_advice(false, false).expect("a bare TTY has no window");
        for wanted in ["tray", "syrinx toggle", "desktop session", "hotkey"] {
            assert!(why.contains(wanted), "missing {wanted:?} from:\n{why}");
        }
    }

    #[test]
    fn an_ssh_session_is_told_the_window_cannot_follow_it() {
        // Otherwise the advice to "start it inside the desktop session" reads
        // as though forwarding were merely misconfigured.
        let why = no_window_advice(false, true).expect("SSH with no display has no window");
        assert!(why.contains("SSH"), "{why}");
        assert!(why.contains("forward"), "{why}");
        // The non-SSH advice is still there: it is additional, not instead.
        assert!(why.contains("syrinx toggle"), "{why}");
    }

    #[test]
    fn the_message_does_not_fall_back_to_naming_variables() {
        // What it replaces. Three environment variables is what winit said,
        // and knowing their names has never helped anyone start dictating.
        let why = no_window_advice(false, true).unwrap();
        for raw in ["WAYLAND_DISPLAY", "WAYLAND_SOCKET", "DISPLAY"] {
            assert!(!why.contains(raw), "{raw} leaked into:\n{why}");
        }
    }

    #[test]
    fn the_daemon_log_sits_beside_the_pid_file() {
        // Per-user, so two accounts on one machine cannot overwrite each
        // other's diagnostics.
        let log = daemon_log_path();
        let pid = syrinx_client::state::default_pid_path();
        assert_eq!(log.parent(), pid.parent());
        assert_ne!(log, pid);
    }
}

struct App {
    config: Config,
    /// Mirrored from the daemon; the GUI owns no session.
    state: DaemonState,
    sources: Vec<Source>,
    save_format: save::Format,
    last_poll: Instant,
    last_source_scan: Instant,
    /// Set when the socket drops, so the window can say so rather than showing
    /// stale state as if it were live.
    disconnected: bool,
    status_line: Option<String>,
    list_error: Option<String>,
    /// Server address being edited. Applied on Enter or the Apply button, not
    /// per keystroke, so a half-typed host is never dialled.
    url_edit: String,
    editing_url: bool,
    /// Whether the help window is open.
    show_help: bool,
}

impl App {
    fn new(config: Config) -> Self {
        let format = config.format;
        let mut app = Self {
            show_help: false,
            config,
            state: DaemonState::default(),
            sources: Vec::new(),
            // Replaced by the daemon's value on the first poll; this only
            // covers the frame before that arrives.
            save_format: format,
            last_poll: Instant::now() - POLL_INTERVAL,
            last_source_scan: Instant::now(),
            disconnected: false,
            status_line: None,
            list_error: None,
            url_edit: String::new(),
            editing_url: false,
        };
        app.refresh_sources();
        app.poll();
        app
    }

    fn refresh_sources(&mut self) {
        self.last_source_scan = Instant::now();
        match syrinx_client::list_sources() {
            Ok(list) => {
                self.list_error = None;
                self.sources = list;
            }
            Err(e) => self.list_error = Some(format!("{e:#}")),
        }
    }

    fn poll(&mut self) {
        self.last_poll = Instant::now();
        // Asking with the revision already on screen: an unchanged transcript
        // comes back without its text, which is what keeps a two-hour meeting
        // from being serialised, sent and parsed thirty times a second to say
        // nothing new.
        let since = self.state.revision;
        match ipc::request(&Request::GetState { since: Some(since) }) {
            Ok(Response::State(mut s)) => {
                // Zero is never a match: it is what a daemon too old to track
                // revisions reports, and keeping our own copy on the strength
                // of it would leave this window permanently blank.
                if since != 0 && s.revision == since {
                    s.transcript = std::mem::take(&mut self.state.transcript);
                    s.turns = std::mem::take(&mut self.state.turns);
                }
                // The daemon is the authority: it owns the session that
                // streams, and it persists the setting. Mirroring it here
                // keeps the dropdown honest across restarts and across two
                // windows open at once.
                self.save_format = s.format;
                self.state = s;
                self.disconnected = false;
            }
            Ok(Response::Error { message }) => self.status_line = Some(message),
            Ok(_) => {}
            Err(_) => self.disconnected = true,
        }
    }

    /// Send a command and refresh immediately, so the UI does not wait a poll
    /// interval to reflect a button the user just pressed.
    fn send(&mut self, req: Request) {
        match ipc::request(&req) {
            Ok(Response::Error { message }) => self.status_line = Some(message),
            Ok(Response::Saved { path }) => self.status_line = Some(format!("Saved to {path}")),
            Ok(_) => self.status_line = None,
            Err(e) => {
                self.disconnected = true;
                self.status_line = Some(format!("{e:#}"));
            }
        }
        self.poll();
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.last_poll.elapsed() >= POLL_INTERVAL {
            self.poll();
        }
        // Rescan while idle only: the list is frozen during a session anyway,
        // and shelling out to pw-dump every couple of seconds for no reason is
        // wasteful.
        if !self.state.status.is_active()
            && self.last_source_scan.elapsed() >= SOURCE_RESCAN_INTERVAL
        {
            self.refresh_sources();
        }
        let running = self.state.status.is_active();

        if let Some(req) = self.keyboard_shortcuts(ui) {
            self.send(req);
        }
        self.help_window(&ctx);

        ui.add_space(2.0);
        self.status_row(ui);
        if self.labels_unavailable() {
            ui.weak("Speaker labels unavailable on this server");
        }
        ui.add_space(6.0);
        self.source_row(ui, running);
        self.mode_row(ui, running);
        self.meter_row(ui, running);
        ui.add_space(8.0);

        // The card carries the visual weight now, so the rules above and below
        // it are redundant lines across the window.
        let reserved = 104.0;
        let height = ui.available_height() - reserved;
        self.transcript_box(ui, height);
        ui.add_space(8.0);
        self.controls(ui, &ctx, running);

        if self.disconnected {
            ui.colored_label(
                theme::palette::DANGER,
                "Lost the daemon. Close and reopen this window to restart it.",
            );
        }
        if let Some(e) = &self.state.error {
            ui.colored_label(theme::palette::DANGER, e);
        }
        if let Some(e) = &self.list_error {
            ui.colored_label(theme::palette::DANGER, e);
        }
        if let Some(s) = &self.status_line {
            ui.colored_label(theme::palette::SUCCESS, s);
        }

        ctx.request_repaint_after(POLL_INTERVAL);
    }
}

impl App {
    /// Server address editor. Hidden until the address is clicked, so the
    /// common case stays uncluttered.
    fn server_row(&mut self, ui: &mut egui::Ui, running: bool) {
        if !self.editing_url {
            return;
        }
        ui.horizontal(|ui| {
            ui.label("Server:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.url_edit)
                    .desired_width(280.0)
                    .hint_text("ws://192.168.1.10:8770/v1/stream"),
            );
            let submitted = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            // Changing servers mid-session would leave the daemon connected to
            // the old one, so it waits for idle.
            let apply = ui
                .add_enabled(!running, egui::Button::new("Apply"))
                .on_hover_text(if running {
                    "Stop the session first"
                } else {
                    "Save and use this server"
                })
                .clicked();
            if ui.button("Cancel").clicked() {
                self.editing_url = false;
            }
            if (submitted || apply) && !running {
                match self.apply_server() {
                    Ok(()) => {
                        self.status_line = Some(format!("Server set to {}", self.config.url));
                        self.editing_url = false;
                    }
                    Err(e) => self.status_line = Some(format!("{e:#}")),
                }
            }
        });
        ui.weak("The daemon reconnects on the next session.");
    }

    /// Normalise, persist, and hand the new address to the daemon.
    fn apply_server(&mut self) -> Result<()> {
        let server = normalise_server(&self.url_edit)?;
        self.config.url = server.clone();
        self.config.save(&Config::default_path())?;
        // The daemon holds its own copy, so telling it is what actually takes
        // effect; writing the file only makes it survive a restart.
        match ipc::request(&Request::SetServer { server })? {
            Response::Error { message } => anyhow::bail!(message),
            _ => Ok(()),
        }
    }

    fn status_row(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let (colour, filled) = match self.state.status {
                Status::Listening => (theme::palette::RECORDING, true),
                Status::Connecting | Status::Stopping | Status::Transcribing => {
                    (theme::palette::WARNING, true)
                }
                Status::Idle => (egui::Color32::GRAY, false),
            };
            state_dot(ui, colour, filled);
            ui.colored_label(
                colour,
                egui::RichText::new(self.state.status.label()).strong(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let (Some(m), Some(c)) = (&self.state.model, self.state.chunk_ms) {
                    ui.weak(format!("{m} · {c}ms chunks"));
                }
            });
        });
    }

    fn source_row(&mut self, ui: &mut egui::Ui, running: bool) {
        // Checkboxes rather than a single selection: several sources can be
        // active at once, and a combo box cannot express that.
        let mut new_keys: Option<Vec<String>> = None;
        let mut new_mode: Option<SourceMode> = None;
        // Cloned so the combo box closure does not hold a borrow of self while
        // also needing to mutate it.
        let selected: Vec<String> = self.state.source_keys.clone();

        ui.horizontal(|ui| {
            ui.label("Sources:");
            ui.add_enabled_ui(!running, |ui| {
                let summary = match selected.len() {
                    0 => "none".to_string(),
                    1 => self
                        .sources
                        .iter()
                        .find(|s| s.stable_key() == selected[0])
                        .map(|s| s.display())
                        .unwrap_or_else(|| selected[0].clone()),
                    n => format!("{n} sources"),
                };
                egui::ComboBox::from_id_salt("sources")
                    .selected_text(summary)
                    .width(290.0)
                    .show_ui(ui, |ui| {
                        let mut last: Option<SourceKind> = None;
                        for s in &self.sources {
                            if last != Some(s.kind) {
                                if last.is_some() {
                                    ui.separator();
                                }
                                ui.weak(s.kind.label());
                                last = Some(s.kind);
                            }
                            let key = s.stable_key();
                            let mut on = selected.contains(&key);
                            if ui.checkbox(&mut on, s.display()).changed() {
                                let mut keys = selected.clone();
                                if on {
                                    keys.push(key);
                                } else {
                                    keys.retain(|k| k != &key);
                                }
                                new_keys = Some(keys);
                            }
                        }
                        // An application exists in the graph only while it is
                        // playing, so an empty section means "nothing is
                        // playing", not "unsupported".
                        if !self
                            .sources
                            .iter()
                            .any(|s| s.kind == SourceKind::Application)
                        {
                            ui.separator();
                            ui.weak(SourceKind::Application.label());
                            ui.weak("  nothing is playing audio right now");
                        }
                    });
                // Latin text, not a glyph: the geometric and icon ranges are
                // not reliably covered by the bundled fonts on Windows, where
                // they render as missing-glyph boxes.
                if ui
                    .button("Rescan")
                    .on_hover_text("Rescan sources")
                    .clicked()
                {
                    self.refresh_sources();
                }
            });
        });

        // Only meaningful with more than one source, so it stays out of the way
        // until it means something.
        if selected.len() > 1 {
            ui.horizontal(|ui| {
                ui.add_space(56.0);
                ui.add_enabled_ui(!running, |ui| {
                    for m in SourceMode::ALL {
                        if ui
                            .selectable_label(self.state.source_mode == m, m.label())
                            .on_hover_text(match m {
                                SourceMode::Combined => "Mix into one stream",
                                SourceMode::Separate => {
                                    "One labelled stream each; only the first types"
                                }
                            })
                            .clicked()
                        {
                            new_mode = Some(m);
                        }
                    }
                });
            });
        }

        if new_keys.is_some() || new_mode.is_some() {
            self.send(Request::SetSources {
                keys: new_keys.unwrap_or_else(|| selected.clone()),
                source_mode: new_mode,
            });
        }
    }

    fn mode_row(&mut self, ui: &mut egui::Ui, running: bool) {
        let mut chosen: Option<OutputMode> = None;
        let mut labels: Option<bool> = None;
        // The daemon's setting, never a copy kept here: it is the thing that
        // reaches session.start, and two windows open at once must agree.
        let mut diarize = self.state.diarize_configured;
        ui.horizontal(|ui| {
            ui.label("Mode:");
            // Both of these are fixed at session start on the wire, so changing
            // either needs a reconnect.
            ui.add_enabled_ui(!running, |ui| {
                for m in OutputMode::ALL {
                    if ui
                        .selectable_label(self.state.mode == m, m.label())
                        .clicked()
                    {
                        chosen = Some(m);
                    }
                }
                if ui
                    .checkbox(&mut diarize, "Speaker labels")
                    .on_hover_text(if running {
                        "Stop the session first"
                    } else {
                        "Ask the server for Speaker 1, Speaker 2… on transcribed \
                         text. Typing at the cursor never gets labels."
                    })
                    .changed()
                {
                    labels = Some(diarize);
                }
            });
        });
        if self.state.mode.types_at_cursor() {
            ui.weak("Types into whatever window has focus. Append-only: it never deletes.");
        }
        if let Some(mode) = chosen {
            self.send(Request::SetMode { mode });
        }
        if let Some(diarize) = labels {
            self.send(Request::SetDiarize { diarize });
        }
    }

    /// Ten-band spectrum of the selected source.
    ///
    /// Answers "is this device actually carrying audio" before a session is
    /// started, which otherwise can only be discovered by recording and getting
    /// an empty transcript.
    fn meter_row(&mut self, ui: &mut egui::Ui, running: bool) {
        let bands = &self.state.levels;
        ui.horizontal(|ui| {
            ui.label("Level:");

            let (rect, _) = ui.allocate_exact_size(egui::vec2(220.0, 26.0), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);

            let n = syrinx_audio::meter::BANDS;
            let gap = 2.0;
            let bar_w = (rect.width() - gap * (n as f32 + 1.0)) / n as f32;
            for i in 0..n {
                let v = bands.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                // A visible floor, so an idle meter reads as "running, quiet"
                // rather than "not running at all".
                let h = (rect.height() - 4.0) * v.max(0.02);
                let x = rect.left() + gap + i as f32 * (bar_w + gap);
                let bar = egui::Rect::from_min_size(
                    egui::pos2(x, rect.bottom() - 2.0 - h),
                    egui::vec2(bar_w, h),
                );
                // Green through amber to red: loud enough to clip is worth
                // seeing at a glance.
                let colour = if v > 0.85 {
                    theme::palette::RECORDING
                } else if v > 0.6 {
                    theme::palette::WARNING
                } else {
                    theme::palette::SUCCESS
                };
                painter.rect_filled(bar, 1.0, colour);
            }

            if self.state.status == Status::Transcribing {
                ui.weak(format!("file {:.0}%", self.state.progress * 100.0));
            } else if running {
                ui.weak("(session running)");
            } else if self.state.rms > 0.001 {
                ui.weak(format!("{:.0}%", self.state.rms * 100.0));
            } else {
                ui.weak("silent");
            }
        });
    }

    fn transcript_box(&self, ui: &mut egui::Ui, height: f32) {
        // A surface of its own rather than bare canvas. The transcript is the
        // content of this window, and content that sits directly on the
        // background reads as an empty window with some text in the corner.
        egui::Frame::new()
            .fill(theme::palette::SURFACE)
            .stroke(egui::Stroke::new(1.0, theme::palette::BORDER))
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::symmetric(12, 10))
            .show(ui, |ui| {
                self.transcript_scroll(ui, height);
            });
    }

    fn transcript_scroll(&self, ui: &mut egui::Ui, height: f32) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .max_height(height.max(80.0))
            .show(ui, |ui| {
                if self.state.transcript.is_empty() {
                    if self.state.mode == OutputMode::Type {
                        ui.weak("Typing at the cursor; no transcript is kept in this mode.");
                        if !self.state.last_fragment.is_empty() {
                            ui.label(
                                egui::RichText::new(format!("last: {}", self.state.last_fragment))
                                    .italics(),
                            );
                        }
                    } else {
                        ui.weak("Nothing transcribed yet.");
                    }
                } else if self.state.turns.iter().any(|(s, _)| s.is_some()) {
                    // A paragraph per turn, headed by its speaker. A leading
                    // turn before the first label ever appears has nothing
                    // honest to call itself, so it renders bare rather than
                    // borrowing a number from the turn after it.
                    //
                    // Grouped by the daemon, which does it once per change
                    // rather than once per frame. This used to run over every
                    // segment of the whole transcript on every repaint.
                    for (speaker, text) in &self.state.turns {
                        if let Some(n) = speaker {
                            ui.label(egui::RichText::new(format!("Speaker {n}")).strong());
                        }
                        ui.label(text);
                    }
                } else {
                    ui.label(&self.state.transcript);
                }
            });
    }

    /// Whether to say the server did not honour a diarization request.
    ///
    /// Judged against `state.diarize_requested`, which the daemon stamps from
    /// what it actually sent -- not a GUI-side read of the config file, which
    /// is a second process reading a file the daemon may have read a moment
    /// earlier or later, and can disagree with it. `Listening` is the only
    /// status where the handshake has demonstrably already answered:
    /// `Connecting` precedes it entirely (every cold start would flash the
    /// notice for the whole model-load window), and `Transcribing` belongs to
    /// a bulk file job, which never requests labels in the first place.
    fn labels_unavailable(&self) -> bool {
        self.state.diarize_requested
            && !self.state.diarize
            && self.state.status == Status::Listening
    }

    /// Keys handled by this window.
    ///
    /// Every one takes a modifier. A bare key would fire while typing into the
    /// server address field, and egui reports key presses whether or not a text
    /// field has focus.
    fn keyboard_shortcuts(&mut self, ui: &mut egui::Ui) -> Option<Request> {
        let (toggle, save, help) = ui.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::CTRL, egui::Key::D),
                i.consume_key(egui::Modifiers::CTRL, egui::Key::S),
                i.consume_key(egui::Modifiers::NONE, egui::Key::F1),
            )
        });
        if help {
            self.show_help = !self.show_help;
        }
        if toggle {
            return Some(Request::Toggle);
        }
        if save && !self.state.transcript.trim().is_empty() {
            return Some(Request::Save {
                format: self.save_format,
                path: None,
            });
        }
        None
    }

    /// Everything that can drive syrinx, in one place.
    fn help_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_help;
        let mut close_clicked = false;
        egui::Window::new("Keys and controls")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.heading("In this window");
                keys_table(
                    ui,
                    "window-keys",
                    &[
                        ("Ctrl+D", "Start or stop dictation"),
                        ("Ctrl+S", "Save the transcript"),
                        ("F1", "Show or hide this window"),
                    ],
                );

                ui.add_space(10.0);
                ui.heading("Anywhere");
                // Reported by the daemon rather than read from the config: a
                // hotkey can be configured and still not be listening, and
                // saying so is the whole point of showing it here.
                let report = &self.state.hotkey;
                ui.horizontal_wrapped(|ui| {
                    let colour = if report.is_active() {
                        theme::palette::SUCCESS
                    } else {
                        theme::palette::WARNING
                    };
                    state_dot(ui, colour, report.is_active());
                    ui.label(report.summary());
                });
                if let Some(detail) = report.detail() {
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new(detail).weak().size(11.0));
                }
                if matches!(report, syrinx_client::hotkey::Report::Unset) {
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(
                            "Set one with `hotkey = \"ctrl+alt+d\"` in the config, \
                             then restart the daemon.",
                        )
                        .weak()
                        .size(11.0),
                    );
                }

                ui.add_space(10.0);
                ui.heading("Tray icon");
                keys_table(
                    ui,
                    "tray-keys",
                    &[
                        ("Left click", "Start or stop dictation"),
                        ("Right click", "Mode, open this window, quit"),
                    ],
                );

                ui.add_space(10.0);
                ui.heading("Command line");
                ui.label(
                    egui::RichText::new("The same actions, for a key binding or a script.")
                        .weak()
                        .size(11.0),
                );
                keys_table(
                    ui,
                    "cli-keys",
                    &[
                        ("syrinx toggle", "Start or stop"),
                        ("syrinx status", "What it is doing now"),
                        ("syrinx quit", "Stop the daemon"),
                    ],
                );

                ui.add_space(10.0);
                close_clicked = ui.button("Close").clicked();
            });
        self.show_help = open && !close_clicked;
    }

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, running: bool) {
        let mut action: Option<Request> = None;
        ui.horizontal(|ui| {
            if running {
                if ui.button("Stop").clicked() {
                    action = Some(Request::Stop);
                }
            } else if ui.button("Start").clicked() {
                action = Some(Request::Start);
            }

            if ui
                .button("Help")
                .on_hover_text("Keys and controls (F1)")
                .clicked()
            {
                self.show_help = !self.show_help;
            }

            let has_text = !self.state.transcript.trim().is_empty();
            if ui.button("Copy").clicked() {
                ctx.copy_text(self.state.transcript.clone());
            }
            if ui
                .add_enabled(has_text, egui::Button::new("Save"))
                .on_hover_text("Save to the default transcripts folder")
                .clicked()
            {
                action = Some(Request::Save {
                    format: self.save_format,
                    path: None,
                });
            }
            // Streaming is a property of the next session, so it can be
            // changed while one is running without disturbing it.
            let streaming = self.state.stream_to.is_some();
            // The word already carries the state, so the record dot was
            // redundant -- and it rendered as a box on Windows.
            let label = if streaming { "Streaming" } else { "Stream…" };
            // Separate mode writes one file per source beside the chosen
            // path, so a tooltip naming only that path would name a file
            // nothing is being written to.
            let split =
                self.state.source_mode == SourceMode::Separate && self.state.source_keys.len() > 1;
            let hover = match &self.state.stream_to {
                Some(p) if split => {
                    format!("Appending to one file per source beside {p}\nClick to stop")
                }
                Some(p) => format!("Appending to {p}\nClick to stop"),
                None => "Append the transcript to a file as you speak, so\n\
                         nothing is lost if this crashes"
                    .to_string(),
            };
            if ui.button(label).on_hover_text(hover).clicked() {
                if streaming {
                    action = Some(Request::SetStreamFile { path: None });
                    self.status_line = Some("Stopped streaming to file".into());
                } else if let Some(p) = rfd::FileDialog::new()
                    .set_file_name(save::filename_for(&save::timestamp()))
                    .set_directory(save::default_dir())
                    .save_file()
                {
                    let path = p.display().to_string();
                    self.status_line = Some(format!("Appending to {path}"));
                    action = Some(Request::SetStreamFile { path: Some(path) });
                }
            }

            if ui
                .add_enabled(has_text, egui::Button::new("Save as…"))
                .clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .set_file_name(save::filename_for(&save::timestamp()))
                    .set_directory(save::default_dir())
                    .save_file()
            {
                action = Some(Request::Save {
                    format: self.save_format,
                    path: Some(p.display().to_string()),
                });
            }
            egui::ComboBox::from_id_salt("save_format")
                .selected_text(self.save_format.label())
                .width(110.0)
                .show_ui(ui, |ui| {
                    for f in save::Format::ALL {
                        if ui
                            .selectable_label(self.save_format == f, f.label())
                            .on_hover_text(match f {
                                save::Format::Plain => "Continuous prose",
                                save::Format::Timestamped => "Each fragment prefixed [MM:SS]",
                                save::Format::Labelled => "Time and source on each line",
                            })
                            .clicked()
                        {
                            self.save_format = f;
                            // The daemon streams; it has to be told, or the
                            // dropdown only affects the Save button.
                            action = Some(Request::SetFormat { format: f });
                        }
                    }
                });
            if ui
                .add_enabled(!running, egui::Button::new("File…"))
                .on_hover_text("Transcribe an audio file: wav, mp3, m4a, opus, flac")
                .clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .add_filter(
                        "Audio",
                        &["wav", "mp3", "m4a", "opus", "flac", "ogg", "aac"],
                    )
                    .pick_file()
            {
                action = Some(Request::TranscribeFile {
                    path: p.display().to_string(),
                });
            }
            let has_text2 = !self.state.transcript.trim().is_empty();
            if self.state.source_mode == SourceMode::Separate
                && self.state.source_keys.len() > 1
                && ui
                    .add_enabled(has_text2, egui::Button::new("Save split"))
                    .on_hover_text("One file per source")
                    .clicked()
            {
                action = Some(Request::SaveSplit {
                    format: self.save_format,
                });
            }
            if ui
                .add_enabled(has_text2 && !running, egui::Button::new("Clear"))
                .on_hover_text("Discard the transcript")
                .clicked()
            {
                action = Some(Request::Clear);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Shortened for the corner of a window; the full address is
                // in the tooltip and in the edit field.
                let short = self
                    .config
                    .url
                    .trim_start_matches("ws://")
                    .trim_start_matches("wss://")
                    .trim_end_matches("/v1/stream");
                if ui
                    .link(short)
                    .on_hover_text(format!("{}\nclick to change", self.config.url))
                    .clicked()
                {
                    self.url_edit = self.config.url.clone();
                    self.editing_url = true;
                }
            });
        });
        self.server_row(ui, running);
        ui.weak("Closing this window leaves dictation running in the tray.");
        if let Some(a) = action {
            self.send(a);
        }
    }
}

/// Accept a bare host, `host:port`, or a full URL, and produce a full one.
///
/// Typing just an IP is the common case, and failing on it would be needless
/// friction when the rest of the address is always the same.
/// Check a typed address without rewriting it.
///
/// Whitespace is trimmed and nothing else changes. This field used to reduce
/// whatever was typed to a bare host and rebuild the URL around it, which is
/// convenient until it guesses wrong -- and then there is no way to type what
/// you actually meant.
fn normalise_server(input: &str) -> Result<String> {
    let s = input.trim();
    if s.is_empty() {
        anyhow::bail!("enter the server address, e.g. ws://192.168.1.10:8770/v1/stream");
    }
    if !(s.starts_with("ws://") || s.starts_with("wss://")) {
        anyhow::bail!("the address needs a ws:// or wss:// scheme");
    }
    if s.contains(char::is_whitespace) {
        anyhow::bail!("an address cannot contain spaces");
    }
    let rest = s.split_once("://").map(|(_, r)| r).unwrap_or("");
    if rest.is_empty() || rest.starts_with('/') {
        anyhow::bail!("the address names no host");
    }
    Ok(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_address_survives_untouched() {
        for a in [
            "ws://192.168.1.10:8770/v1/stream",
            "wss://dictate.example.com/v1/stream",
            "ws://localhost:9000/some/other/path",
        ] {
            assert_eq!(normalise_server(a).unwrap(), a);
        }
    }

    #[test]
    fn surrounding_whitespace_is_trimmed_and_nothing_else() {
        assert_eq!(
            normalise_server("  ws://h:1/v1/stream  ").unwrap(),
            "ws://h:1/v1/stream"
        );
    }

    #[test]
    fn an_incomplete_address_is_refused() {
        // These used to be repaired into a URL. Refusing is the point: a
        // guess that lands somewhere unintended is worse than a complaint.
        for bad in [
            "",
            "   ",
            "10.0.0.5",
            "dictate.example.com",
            "ws://",
            "ws:///v1/stream",
        ] {
            assert!(normalise_server(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn a_host_with_spaces_is_refused() {
        assert!(normalise_server("my server").is_err());
    }

    #[test]
    fn the_field_does_not_rewrite_what_was_typed() {
        // It used to reduce an address to a bare host and rebuild the URL.
        let typed = "wss://dictate.example.com/v1/stream";
        assert_eq!(normalise_server(typed).unwrap(), typed);
    }

    #[test]
    fn an_address_without_a_scheme_is_refused() {
        assert!(normalise_server("192.168.1.10").is_err());
        assert!(normalise_server("dictate.example.com").is_err());
    }

    #[test]
    fn the_window_icon_decodes() {
        // `window_icon` swallows a decode failure on purpose, so a truncated
        // or re-rendered asset would quietly cost every window its icon and
        // fail nothing. This is the only place that would ever notice.
        let icon = window_icon().expect("the built-in PNG should decode");
        assert_eq!((icon.width, icon.height), (256, 256));
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
    }
}
