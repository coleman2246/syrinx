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
    Config, OutputMode, Source, SourceKind, ipc, mode::SourceMode,
    ipc::{DaemonState, Request, Response},
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

/// Least the transcript is ever given. Below this a card with a scrollbar in
/// it says nothing at all, so the window is better off overflowing.
const MIN_TRANSCRIPT: f32 = 80.0;

/// Size the window opens at. Named because what it paints has to fit inside
/// it -- there is no scroll area around any of it -- and a test says so.
///
/// It opened at 380 tall, which is less than the window's own minimum
/// content: the rows above the transcript come to about 186, the transcript
/// will not go below [`MIN_TRANSCRIPT`] and its card adds 22 more, and the
/// controls and the lines of message under them are 130 again. The bottom row
/// went under the edge of the window, and the bottom row is where the errors
/// are.
///
/// 460 was that sum before the per-source meter rows existed. They are drawn
/// above the transcript, so they take their height from it rather than from
/// the reserve -- until it reaches [`MIN_TRANSCRIPT`], which it now does, and
/// the overflow reappears at the bottom. Two sources is the ordinary case and
/// the one that has to fit; a meeting metered across four still will not, for
/// the same reason six rows of message do not, and no default height fixes
/// that for every count.
const WINDOW_WIDTH: f32 = 480.0;
const WINDOW_HEIGHT: f32 = 480.0;

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
        .with_inner_size([WINDOW_WIDTH, WINDOW_HEIGHT])
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
        ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"].iter().any(|k| set(k)),
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
            syrinx_client::state::reap_in_background(child);
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
    // A daemon that came up but never listened is still a child of ours. The
    // failure below is reported in a modal dialog, which waits for a person to
    // dismiss it, so leaving the wait to this process exiting could hold a
    // defunct daemon for as long as the message sits on screen.
    syrinx_client::state::reap_in_background(child);
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
        std::env::temp_dir().join(format!(
            "syrinx-gui-test-{}-{tag}",
            std::process::id()
        ))
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
        assert_eq!(daemon_beside(&dir), None, "an empty directory has no daemon");

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

/// A label and a row for every selected source, paired up.
///
/// One entry per source in `selected`, in that order, because that is what the
/// user ticked and what the window has to account for. `reporting` is what the
/// daemon has rows for, which is not always the same list: a source that would
/// not open has no meter behind it, an older daemon reports fewer rows than
/// this one, and a selection can be a tick ahead of what the daemon has acted
/// on. Anything unaccounted for still needs a name, which is what `known` --
/// the enumerated sources behind the picker -- is for.
fn meter_rows(
    selected: &[String],
    reporting: &[syrinx_audio::mixer::SourceHealth],
    known: &[Source],
) -> Vec<(String, Option<syrinx_audio::mixer::SourceHealth>)> {
    (0..selected.len().max(reporting.len()))
        .map(|i| {
            let health = reporting.get(i).cloned();
            let label = match (&health, selected.get(i)) {
                (Some(h), _) => h.label.clone(),
                (None, Some(key)) => known
                    .iter()
                    .find(|s| &s.stable_key() == key)
                    .map(|s| s.short_label())
                    // The key itself, when the device is no longer there to
                    // be asked. It is at least what the user picked.
                    .unwrap_or_else(|| key.clone()),
                (None, None) => "unknown source".to_string(),
            };
            (label, health)
        })
        .collect()
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
    status_line: Option<(Severity, String)>,
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
            Err(_) => self.disconnected = true,
            // Never a `State`, so it is whatever the daemon says went wrong,
            // read the same way `send` reads it. Only ever set, never
            // cleared: the line belongs to the button that was last pressed,
            // not to a refresh happening thirty times a second behind it.
            other => {
                if let Some(s) = status_for(&other) {
                    self.status_line = Some(s);
                }
            }
        }
    }

    /// Send a command and refresh immediately, so the UI does not wait a poll
    /// interval to reflect a button the user just pressed.
    fn send(&mut self, req: Request) {
        let reply = ipc::request(&req);
        // Only ever set here, never cleared: the poll that follows is what
        // decides the socket is back.
        if reply.is_err() {
            self.disconnected = true;
        }
        self.status_line = status_for(&reply);
        self.poll();
    }
}

/// Whether a status line reports something that worked or something that did
/// not.
///
/// The line was painted green whatever it said, so a daemon error read as a
/// success. That matters most for streaming: the likeliest way choosing a file
/// fails is choosing one that cannot be opened, and the refusal arrives as an
/// ordinary `Response::Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Severity {
    Ok,
    Error,
}

impl Severity {
    fn colour(self) -> egui::Color32 {
        match self {
            Severity::Ok => theme::palette::SUCCESS,
            Severity::Error => theme::palette::DANGER,
        }
    }
}

/// What a reply leaves on the status line, and in what colour.
///
/// `None` clears it. An acknowledgement has nothing to say, and a message left
/// over from the last command reads as the answer to this one.
fn status_for(reply: &Result<Response>) -> Option<(Severity, String)> {
    match reply {
        Ok(Response::Error { message }) => Some((Severity::Error, message.clone())),
        Ok(Response::Saved { path }) => Some((Severity::Ok, format!("Saved to {path}"))),
        Ok(_) => None,
        Err(e) => Some((Severity::Error, format!("{e:#}"))),
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.window(ui);
    }
}

impl App {
    /// Everything the window paints, one frame of it.
    ///
    /// Separated from the `eframe::App` method it is the whole of, because an
    /// `eframe::Frame` cannot be built outside eframe and this needs none: a
    /// test can hand it a bare `Ui` of the window's size and measure what
    /// comes out.
    fn window(&mut self, ui: &mut egui::Ui) {
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
        self.meter_row(ui);
        ui.add_space(8.0);

        // The card carries the visual weight now, so the rules above and below
        // it are redundant lines across the window.
        let room = ui.available_height() - self.reserved_below(ui);
        let height = room.max(MIN_TRANSCRIPT);
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
        // Amber rather than red, and a line of its own. A lost fragment is not
        // a dead session: the words are still arriving, still being kept and
        // still saveable, and it is the copy on disk that has a hole in it.
        if let Some(e) = &self.state.stream_error {
            ui.colored_label(theme::palette::WARNING, e);
        }
        if let Some(e) = &self.list_error {
            ui.colored_label(theme::palette::DANGER, e);
        }
        if let Some((severity, s)) = &self.status_line {
            ui.colored_label(severity.colour(), s);
        }

        ctx.request_repaint_after(POLL_INTERVAL);
    }

    /// Room to leave below the transcript for everything painted after it.
    ///
    /// This was the constant 104, sized when there were fewer rows than
    /// there are now. Every row added since -- the stream target, a lost
    /// fragment, the status line -- came out of the transcript's height
    /// without being asked for, and the last one fell off the bottom of the
    /// window: `eframe` hands `window` a bare root `Ui` with no scroll area,
    /// so a row that does not fit is clipped rather than scrolled, and the
    /// rows at the end are the errors.
    ///
    /// Deliberately generous. The control row is laid out `Align::Center`
    /// across whatever height is left, so it stretches to absorb an
    /// over-estimate and costs nothing but a slightly shorter transcript,
    /// while an under-estimate is a row nobody can see.
    fn reserved_below(&self, ui: &egui::Ui) -> f32 {
        let gap = ui.spacing().item_spacing.y;
        let line = ui.text_style_height(&egui::TextStyle::Body) + gap;
        let button = ui.spacing().interact_size.y.max(
            ui.text_style_height(&egui::TextStyle::Button)
                + 2.0 * ui.spacing().button_padding.y,
        ) + gap;
        // The card's margin and border, which sit outside the height handed
        // to the scroll area inside it.
        const CARD: f32 = 22.0;
        // Two button-height rows, which is what the controls wrap onto at
        // the width this window opens at.
        CARD + 8.0 + gap + 2.0 * button + self.footer_rows() as f32 * line
    }

    /// How many single-line messages sit under the controls right now.
    fn footer_rows(&self) -> usize {
        // The tray hint, which is always painted.
        1 + usize::from(self.state.stream_to.is_some())
            + usize::from(self.disconnected)
            + usize::from(self.state.error.is_some())
            + usize::from(self.state.stream_error.is_some())
            + usize::from(self.list_error.is_some())
            + usize::from(self.status_line.is_some())
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
            let submitted =
                resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
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
                        self.status_line = Some((
                            Severity::Ok,
                            format!("Server set to {}", self.config.url),
                        ));
                        self.editing_url = false;
                    }
                    Err(e) => {
                        self.status_line = Some((Severity::Error, format!("{e:#}")))
                    }
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
                        if !self.sources.iter().any(|s| s.kind == SourceKind::Application) {
                            ui.separator();
                            ui.weak(SourceKind::Application.label());
                            ui.weak("  nothing is playing audio right now");
                        }
                    });
                // Latin text, not a glyph: the geometric and icon ranges are
                // not reliably covered by the bundled fonts on Windows, where
                // they render as missing-glyph boxes.
                if ui.button("Rescan").on_hover_text("Rescan sources").clicked() {
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

    /// Ten-band spectrum of the selected source, and a row per source under it.
    ///
    /// Answers "is this device actually carrying audio" before a session is
    /// started, which otherwise can only be discovered by recording and getting
    /// an empty transcript. It answers it while one runs, too, which is why
    /// nothing here is conditional on that.
    ///
    /// The spectrum is measured downstream of the mixer while a session runs,
    /// and from the first source alone while idle, so with two sources ticked
    /// it cannot say which of them is contributing. The per-source rows can.
    fn meter_row(&mut self, ui: &mut egui::Ui) {
        let bands = &self.state.levels;
        ui.horizontal(|ui| {
            ui.label("Level:");

            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(220.0, 26.0),
                egui::Sense::hover(),
            );
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
            } else if self.state.rms > 0.001 {
                // Shown while a session runs too. This used to read "(session
                // running)", which made a session at 0% and one at 40% look
                // identical -- so the one thing a user with a suspect source
                // wanted to know was the one thing the window would not say.
                ui.weak(format!("{:.0}%", self.state.rms * 100.0));
            } else {
                ui.weak("silent");
            }
        });

        // One row per *selected* source, once there is more than one. With a
        // single source the spectrum above is already that source's row, and
        // repeating it would only take space from the transcript.
        //
        // Gated on how many were selected rather than on how many are
        // reporting. Only sources that started report, so with two ticked and
        // one broken this drew nothing at all -- in precisely the case the
        // rows exist for, leaving the user with the same silence and the same
        // absence of explanation they came here to diagnose.
        if self.state.source_keys.len() > 1 {
            for (label, health) in
                meter_rows(&self.state.source_keys, &self.state.sources, &self.sources)
            {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        egui::vec2(104.0, 14.0),
                        egui::Label::new(egui::RichText::new(&label).small()),
                    );

                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(140.0, 10.0), egui::Sense::hover());
                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
                    let v = health.as_ref().map_or(0.0, |s| s.rms.clamp(0.0, 1.0));
                    let bar = egui::Rect::from_min_size(
                        rect.min,
                        egui::vec2(rect.width() * v.max(0.01), rect.height()),
                    );
                    let colour = if health.as_ref().is_some_and(|s| s.error.is_some()) {
                        // A device that would not open is a real fault, which
                        // a device with nothing to play is not. Distinct from
                        // the level scale below it, so a loud source and a
                        // broken one can never read as the same thing.
                        theme::palette::DANGER
                    } else if health.as_ref().is_none_or(|s| s.silent) {
                        theme::palette::BORDER
                    } else if v > 0.85 {
                        theme::palette::RECORDING
                    } else if v > 0.6 {
                        theme::palette::WARNING
                    } else {
                        theme::palette::SUCCESS
                    };
                    painter.rect_filled(bar, 1.0, colour);

                    match &health {
                        Some(s) if s.error.is_some() => {
                            ui.weak("not available")
                                .on_hover_text(s.error.clone().unwrap_or_default());
                        }
                        // "Silent", not "failed", and not in red. A Windows
                        // loopback on an output with nothing playing delivers
                        // nothing at all, and that is it working correctly.
                        Some(s) if s.silent => {
                            ui.weak("silent");
                        }
                        Some(s) => {
                            ui.weak(format!("{:.0}%", s.rms * 100.0));
                        }
                        // Selected, and nothing is reporting it. Said rather
                        // than left out: a row that is simply missing reads as
                        // a source that was never ticked.
                        None => {
                            ui.weak("not reporting");
                        }
                    }

                    // Trimming is words going missing mid-utterance, and this
                    // is the only place a reader can find out that happened.
                    if let Some(dropped) =
                        health.as_ref().map(|s| s.dropped).filter(|d| *d > 0)
                    {
                        let seconds = dropped as f32 / syrinx_proto::SAMPLE_RATE as f32;
                        ui.weak(format!("-{seconds:.1}s")).on_hover_text(
                            "Audio trimmed from this source, because the mix could not \
                             keep up with it or it could not keep up with the mix.",
                        );
                    }
                });
            }
        }
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
            // Clamped by the caller too, which needs to know the number it
            // gets: the room left for everything else is measured against it.
            .max_height(height.max(MIN_TRANSCRIPT))
            .show(ui, |ui| {
                if self.state.transcript.is_empty() {
                    if self.state.mode == OutputMode::Type {
                        ui.weak("Typing at the cursor; no transcript is kept in this mode.");
                        if !self.state.last_fragment.is_empty() {
                            ui.label(
                                egui::RichText::new(format!(
                                    "last: {}",
                                    self.state.last_fragment
                                ))
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
        // What the press meant, for the presses whose meaning the daemon's
        // reply does not carry. Applied after `send`, which writes the status
        // line from that reply; see the end of this function.
        let mut note: Option<(Severity, String)> = None;
        // Wrapped, because this row holds a dozen controls and the window is
        // 480 wide by default. A plain `horizontal` does not wrap, so
        // everything past the edge was drawn outside the clip rectangle and
        // simply never seen -- including the Clear button and the server
        // address in the corner.
        ui.horizontal_wrapped(|ui| {
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
            // Streaming is a property of the next session -- the writer opens
            // at session start -- so a file can be chosen while one is running
            // without disturbing it.
            //
            // The ellipsis is this window's convention for a button that opens
            // a dialog, and it is honoured on every press now. It used to be a
            // toggle, so the dialog was reached only while nothing was set;
            // since the setting is persisted, the second run of the GUI opened
            // reading "Streaming" before the user had touched anything, and
            // pressing it stopped rather than asked.
            let streaming = self.state.stream_to.is_some();
            if ui
                .button("Stream…")
                .on_hover_text(stream_hover(running))
                .clicked()
            {
                let (dir, name) =
                    stream_seed(self.state.stream_to.as_deref(), &save::timestamp());
                let chosen = rfd::FileDialog::new()
                    .set_file_name(name)
                    .set_directory(dir)
                    .save_file();
                // Cancelling leaves the setting alone: closing a dialog is not
                // a way of asking for anything.
                if let Some(req) = stream_choice(chosen) {
                    if let Request::SetStreamFile { path: Some(p) } = &req {
                        note = Some((Severity::Ok, stream_chosen_note(p)));
                    }
                    action = Some(req);
                }
            }
            // A bare verb, because it acts at once. Clearing is its own
            // control now that the button beside it always asks, and it is
            // shown only while there is a file set. "Clear" rather than
            // "Stop": it cannot stop the session that is running, and while
            // nothing is running there is no activity to stop either.
            if streaming
                && ui
                    .button("Clear stream file")
                    .on_hover_text(stream_clear_hover(running))
                    .clicked()
            {
                note = Some((Severity::Ok, stream_cleared_note(running).to_string()));
                action = Some(Request::SetStreamFile { path: None });
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
                    .add_filter("Audio", &["wav", "mp3", "m4a", "opus", "flac", "ogg", "aac"])
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
            // Right-aligned in whatever the row has left, which after
            // wrapping is the end of its last line. It is not an action like
            // its neighbours, and the corner is where an address belongs.
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
        // Under the row rather than in the button's tooltip: what is being
        // written is worth reading without hovering, and in separate mode the
        // path that was chosen is not one of the files that gets written.
        if let Some(target) = self.state.stream_to.clone() {
            let names = self.source_names();
            let targets =
                stream_targets(&target, self.state.source_mode, names.as_deref());
            if let Some(label) = stream_target_label(&targets, running) {
                ui.weak(label);
            }
        }
        ui.weak("Closing this window leaves dictation running in the tray.");
        if let Some(a) = action {
            self.send(a);
            // After the dispatch, and only over a line the reply left empty.
            // `send` fills the status line from what the daemon said, and for
            // these presses that is a bare acknowledgement with nothing to
            // report -- but a refusal must not be painted over with a
            // confirmation of something that did not happen.
            if self.status_line.is_none() {
                self.status_line = note;
            }
        }
    }

    /// The selected sources under the names their files would take, or
    /// `None` when this window cannot say.
    ///
    /// `DaemonRuntime::start` resolves every remembered key against the
    /// sources actually present, **skips the ones that have gone**, and only
    /// then decides whether there is more than one to split the path between.
    /// This window resolves against its own scan, which can be older, so it
    /// may only speak when every selected key is accounted for. With a device
    /// unplugged, two keys became two filenames here and one file there, so
    /// neither of the names shown was ever written -- and an unresolved key
    /// used to be turned into a filename of its own, which invented
    /// `notes-alsa-input-usb-blue-microphones-yeti-st.txt` out of nothing.
    ///
    /// `short_labels` rather than `Source::short_label` one at a time, because
    /// that is what the daemon uses and the two must name the same files.
    fn source_names(&self) -> Option<Vec<String>> {
        let resolved: Option<Vec<Source>> = self
            .state
            .source_keys
            .iter()
            .map(|k| self.sources.iter().find(|s| &s.stable_key() == k).cloned())
            .collect();
        resolved.map(|s| syrinx_audio::source::short_labels(&s))
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

/// Where the Save dialog opens, and what it proposes to call the file.
///
/// Recomputed on every press. The chosen path is persisted, so the last one is
/// still there on the next run -- and reusing its *name* is what appended
/// tonight's meeting to a file created a fortnight ago and named for that day.
/// The folder is worth remembering; the filename is not.
fn stream_seed(remembered: Option<&str>, stamp: &str) -> (PathBuf, String) {
    let dir = remembered
        // The config documents `~/transcripts/notes.txt`, and nothing has
        // expanded that by the time it reaches a dialog.
        .map(syrinx_client::config::expand_tilde)
        .and_then(|p| p.parent().map(PathBuf::from))
        // A dialog pointed at a folder that has since gone opens wherever the
        // toolkit decides, which is worse than opening where transcripts live.
        .filter(|p| p.is_dir())
        .unwrap_or_else(save::default_dir);
    (dir, save::filename_for(stamp))
}

/// What a press of Stream asks the daemon for.
///
/// Cancelling is not stopping. A dialog closed with nothing chosen leaves the
/// current target exactly as it was, so there is no request at all rather than
/// one clearing it.
fn stream_choice(chosen: Option<PathBuf>) -> Option<Request> {
    chosen.map(|p| Request::SetStreamFile {
        path: Some(p.display().to_string()),
    })
}

/// The files a session started now would actually append to.
///
/// Separate mode with more than one source writes one beside the chosen path
/// per source and never writes the chosen path itself. The rule is
/// `DaemonRuntime::start`'s, and has to stay the same one, or this names files
/// that never appear.
///
/// `None` for the sources means the window could not resolve every selected
/// key, so it does not know how many files the daemon will open. The chosen
/// path is then all there is to say honestly: it is at least the path the
/// split is built from, where a guessed list of names is nothing at all.
fn stream_targets(
    target: &str,
    mode: SourceMode,
    sources: Option<&[String]>,
) -> Vec<String> {
    let Some(sources) = sources else {
        return vec![target.to_string()];
    };
    if mode != SourceMode::Separate || sources.len() < 2 {
        return vec![target.to_string()];
    }
    let base = std::path::Path::new(target);
    sources
        .iter()
        .map(|s| save::path_for_source(base, s).display().to_string())
        .collect()
}

/// Hover text for the Stream button.
///
/// Two answers, because there are two truths. `session::run` opens the writer
/// once, at session start, from the options that session captured, so a file
/// chosen now cannot redirect a session already under way -- which the old
/// text claimed it did, in the present tense, from the moment the setting
/// existed.
fn stream_hover(running: bool) -> &'static str {
    if running {
        "Choose a file to append the transcript to.\n\
         The file is opened when a session starts, so this applies to\n\
         the next session and not to the one running."
    } else {
        "Choose a file to append the transcript to as you speak, so\n\
         nothing is lost if this crashes."
    }
}

/// Hover text for the control that clears the stream file.
///
/// It said "Stop appending to a file", which is a promise it cannot keep:
/// `session::run` opens the writer once, at session start, from the options
/// that session captured, so nothing sent now reaches a session already
/// writing. The word is "clear" in both states, because that is what the
/// button does in both -- while idle there is no activity to stop either, only
/// a setting that says where the next session would write.
fn stream_clear_hover(running: bool) -> &'static str {
    if running {
        "Clear the file the transcript is appended to.\n\
         The file is opened when a session starts, so the session running\n\
         now keeps writing to its own; this applies to the next one."
    } else {
        "Clear the file the transcript is appended to. Nothing is\n\
         appended until a file is chosen again."
    }
}

/// What the window says once a file has been chosen.
///
/// `rfd`'s `save_file()` raises the operating system's "replace this file?"
/// prompt, and this code appends. Now that the dialog opens on every press
/// that prompt is the ordinary path, so saying plainly that nothing is
/// replaced is worth a line.
fn stream_chosen_note(path: &str) -> String {
    format!("Appending to {path}. Anything already in it is kept.")
}

/// What the window says once the file has been cleared.
fn stream_cleared_note(running: bool) -> &'static str {
    if running {
        "The next session will not stream to a file. This one still is."
    } else {
        "No longer streaming to a file."
    }
}

/// The target as it reads beneath the controls.
///
/// Running or idle, because the two are different claims. The writer is opened
/// at session start, so while a session runs this names the file the *next*
/// one would open -- and a target changed mid-session would otherwise leave
/// the window reading `Stream target: B.txt` while every fragment went to
/// `A.txt`. The tooltip has always been careful about this; the label, which
/// the design moved out of the tooltip precisely because it is the more
/// visible surface, was not.
fn stream_target_label(targets: &[String], running: bool) -> Option<String> {
    let lead = if running { "Next session streams to" } else { "Stream target" };
    match targets {
        [] => None,
        [one] => Some(format!("{lead}: {one}")),
        many => Some(format!("{lead}, one file per source: {}", many.join(", "))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window with every control showing and nothing running.
    ///
    /// Built field by field rather than through `App::new`, which enumerates
    /// audio devices and dials the daemon socket. Neither is anything to do
    /// with how wide a row of buttons is.
    fn app_showing_everything() -> App {
        let config: Config = toml::from_str("token = ''").expect("a minimal config");
        App {
            config,
            state: DaemonState {
                // Separate mode with two sources is what shows Save split,
                // and a stream target is what shows Clear stream file: the
                // widest the row ever gets.
                source_mode: SourceMode::Separate,
                source_keys: vec!["a".into(), "b".into()],
                stream_to: Some("/tmp/notes.txt".into()),
                transcript: "something to save".into(),
                ..DaemonState::default()
            },
            sources: Vec::new(),
            save_format: save::Format::Plain,
            last_poll: Instant::now(),
            last_source_scan: Instant::now(),
            disconnected: false,
            status_line: None,
            list_error: None,
            url_edit: String::new(),
            editing_url: false,
            show_help: false,
        }
    }

    #[test]
    fn every_control_fits_inside_the_default_window() {
        // The row holds a dozen controls and the default window is 480 wide.
        // A plain `horizontal` does not wrap, so everything past the edge was
        // laid out beyond the clip rectangle and never drawn -- the Clear
        // button and the server address in the corner among them.
        let ctx = egui::Context::default();
        // The spacing is part of the answer: this theme sets its own button
        // padding and item spacing.
        theme::apply(&ctx);
        let mut app = app_showing_everything();
        let mut content = 0.0;
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(WINDOW_WIDTH, 380.0),
            )),
            ..Default::default()
        };
        // A bare root `Ui`, which is what `eframe` hands `App::ui` -- no
        // scroll area, so whatever is too wide is clipped away rather than
        // reachable. Its width is set explicitly because a headless context
        // with no window manager sizes its viewport to whatever it is given
        // to lay out, which is the one thing this must not do.
        let mut out = ctx.run_ui(input, |ui| {
            ui.set_max_width(WINDOW_WIDTH);
            let ctx = ui.ctx().clone();
            app.controls(ui, &ctx, false);
            content = ui.min_rect().width();
        });
        // A frame's font atlas has to be taken by whoever would upload it;
        // dropping it unread is a panic in epaint, and nothing here draws --
        // done before the assertions, because a panic while one is unwinding
        // aborts the process and loses the message.
        out.textures_delta.clear();

        assert!(
            content <= WINDOW_WIDTH,
            "the controls are {content} wide in a window {WINDOW_WIDTH} across"
        );

        // And the server address is still in the right-hand corner. It is
        // laid out right-to-left in whatever space the row has left, which is
        // the part of this that wrapping could have spoiled.
        let link = text_shape(&out.shapes, "127.0.0.1:8770")
            .expect("the server address is not drawn at all");
        assert!(
            // A pixel of slack: a galley's box carries the last glyph's
            // advance, which is not ink and is not what gets clipped.
            link.right() <= WINDOW_WIDTH + 1.0,
            "the server address runs off the window: {link:?}"
        );
        assert!(
            link.left() > WINDOW_WIDTH / 2.0,
            "the server address is no longer in the corner: {link:?}"
        );
    }

    /// How deep into a window of `height` this app paints, once its layout
    /// has settled.
    fn painted_depth(app: &mut App, height: f32) -> f32 {
        let ctx = egui::Context::default();
        theme::apply(&ctx);
        let mut bottom = 0.0;
        // Twice: the first frame of any egui context lays out against sizes
        // nothing has measured yet.
        for _ in 0..2 {
            // No poll and no rescan: this is about layout, and the daemon
            // this window would talk to is whatever happens to be running on
            // the machine running the test.
            app.last_poll = Instant::now();
            app.last_source_scan = Instant::now();
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(WINDOW_WIDTH, height),
                )),
                ..Default::default()
            };
            let mut out = ctx.run_ui(input, |ui| {
                ui.set_max_width(WINDOW_WIDTH);
                ui.set_max_height(height);
                app.window(ui);
                bottom = ui.min_rect().bottom();
            });
            out.textures_delta.clear();
        }
        bottom
    }

    #[test]
    fn nothing_the_window_paints_falls_off_the_bottom_of_it() {
        // `eframe` hands `window` a bare root `Ui` with no scroll area, so a
        // row that does not fit is clipped rather than scrolled -- and the
        // room left below the transcript was a constant, settled before the
        // stream target, the lost-fragment line and the status line existed.
        // The rows at the end are the errors.
        let mut app = app_showing_everything();
        app.status_line = Some((Severity::Ok, "Saved to /tmp/a.txt".into()));
        let depth = painted_depth(&mut app, WINDOW_HEIGHT);
        assert!(
            depth <= WINDOW_HEIGHT,
            "{depth} painted into {WINDOW_HEIGHT} of window"
        );

        // And with every message showing at once. In a taller window,
        // because at the size this one opens there is no room for six rows
        // of message above a transcript that will not go below 80 pixels --
        // which is not something a reserve can fix.
        app.disconnected = true;
        app.state.error = Some("the server closed the connection".into());
        app.state.stream_error = Some("a fragment was not written".into());
        app.list_error = Some("listing sources failed".into());
        let tall = 620.0;
        let depth = painted_depth(&mut app, tall);
        assert!(depth <= tall, "{depth} painted into {tall} of window");
    }

    /// Where a piece of text was actually drawn, if it was drawn at all.
    fn text_shape(
        shapes: &[egui::epaint::ClippedShape],
        want: &str,
    ) -> Option<egui::Rect> {
        fn walk(shape: &egui::Shape, want: &str) -> Option<egui::Rect> {
            match shape {
                egui::Shape::Text(t) if t.galley.job.text.contains(want) => {
                    Some(t.galley.rect.translate(t.pos.to_vec2()))
                }
                egui::Shape::Vec(v) => v.iter().find_map(|s| walk(s, want)),
                _ => None,
            }
        }
        shapes.iter().find_map(|c| walk(&c.shape, want))
    }

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
        for bad in ["", "   ", "10.0.0.5", "dictate.example.com", "ws://", "ws:///v1/stream"] {
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

    fn row(label: &str, silent: bool) -> syrinx_audio::mixer::SourceHealth {
        syrinx_audio::mixer::SourceHealth {
            label: label.into(),
            rms: if silent { 0.0 } else { 0.4 },
            silent,
            ..Default::default()
        }
    }

    #[test]
    fn two_sources_with_one_broken_still_get_two_rows() {
        // The case the rows exist for. Only sources that started report, so a
        // window drawing one row per reporting source drew a single row for
        // two ticked boxes -- and the row it left out was the broken one,
        // which is the only one the user was looking for.
        let selected = vec!["cpal:in:Yeti".to_string(), "cpal:out:Speakers".to_string()];
        let rows = meter_rows(&selected, &[row("Yeti", false)], &[]);
        assert_eq!(rows.len(), 2, "the source that is not reporting went missing");
        assert_eq!(rows[0].0, "Yeti");
        assert!(rows[0].1.is_some());
        // Named from the key, since nothing enumerated matches it here.
        assert_eq!(rows[1].0, "cpal:out:Speakers");
        assert!(rows[1].1.is_none(), "a row was invented for a silent source");
    }

    #[test]
    fn a_source_with_no_row_is_named_from_what_was_enumerated() {
        // The key is a last resort. When the picker still knows the device,
        // the row reads as the same name the picker shows.
        let speakers = Source {
            target: syrinx_audio::SourceTarget::CpalDevice {
                name: "Speakers".into(),
                loopback: true,
            },
            name: "Everything playing on Speakers".into(),
            kind: SourceKind::Monitor,
            detail: None,
            stable_name: Some("cpal:out:Speakers".into()),
            sink_description: Some("Speakers".into()),
        };
        let selected = vec!["cpal:in:Yeti".to_string(), speakers.stable_key()];
        let rows = meter_rows(
            &selected,
            &[row("Yeti", false)],
            std::slice::from_ref(&speakers),
        );
        assert_eq!(rows[1].0, speakers.short_label());
    }

    #[test]
    fn every_reporting_source_keeps_its_row_even_past_the_selection() {
        // A session's rows arrive before a selection change has been acted
        // on, and dropping the extra would blank a meter that is running.
        let rows = meter_rows(
            &["cpal:in:Yeti".to_string()],
            &[row("Yeti", false), row("System audio", true)],
            &[],
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].0, "System audio");
    }

    #[test]
    fn the_dialog_is_seeded_with_todays_name_in_the_remembered_folder() {
        // The bug this exists for. `stream_to` is persisted, so a file chosen
        // on the 20th came back on every run afterwards and a fortnight of
        // meetings piled into one file still called 2026-08-20_09-14-03.txt.
        let stale = std::env::temp_dir().join("2026-08-20_09-14-03.txt");
        let (folder, name) =
            stream_seed(Some(&stale.display().to_string()), "2026-08-27_18-02-11");
        assert_eq!(folder, stale.parent().unwrap(), "the folder is worth keeping");
        assert_eq!(name, "2026-08-27_18-02-11.txt", "the stale name came back");
    }

    #[test]
    fn a_remembered_folder_that_has_gone_falls_back_to_the_default() {
        // A dialog opened on a directory that no longer exists opens
        // somewhere arbitrary, which is worse than opening where transcripts
        // normally go.
        let (folder, name) =
            stream_seed(Some("/no-such-place-9d3f/notes.txt"), "2026-08-27_18-02-11");
        assert_eq!(folder, save::default_dir());
        assert_eq!(name, "2026-08-27_18-02-11.txt");
    }

    #[test]
    fn nothing_remembered_opens_where_transcripts_go() {
        assert_eq!(stream_seed(None, "s").0, save::default_dir());
    }

    #[test]
    fn a_remembered_path_written_with_a_tilde_still_names_a_folder() {
        // The generated config documents `~/transcripts/notes.txt`, and no
        // shell has expanded it by the time it reaches a file dialog.
        //
        // Asserted against the tilde rather than against `HOME`, which this
        // used to read and unwrap: a test that requires the machine running it
        // to have a variable set fails under `env -u HOME`, and what is
        // actually being pinned is that no dialog is ever pointed at a folder
        // literally called `~`.
        let (folder, _) = stream_seed(Some("~/notes.txt"), "s");
        assert!(
            !folder.starts_with("~"),
            "the tilde reached the dialog: {folder:?}"
        );
    }

    #[test]
    fn cancelling_the_dialog_is_not_a_stop() {
        // The button used to be a toggle, so a second press stopped
        // streaming. It always asks now, and a dialog closed with nothing
        // chosen has to leave the setting exactly as it was.
        assert_eq!(stream_choice(None), None);
        assert_eq!(
            stream_choice(Some(PathBuf::from("/tmp/notes.txt"))),
            Some(Request::SetStreamFile {
                path: Some("/tmp/notes.txt".into())
            })
        );
    }

    #[test]
    fn separate_mode_names_the_file_each_source_really_gets() {
        // `save::path_for_source` splits the chosen path, so naming only the
        // path that was chosen would name a file nothing is written to.
        assert_eq!(
            stream_targets(
                "/tmp/meeting.txt",
                SourceMode::Separate,
                Some(&["Yeti".to_string(), "System audio".to_string()]),
            ),
            ["/tmp/meeting-yeti.txt", "/tmp/meeting-system-audio.txt"]
        );
    }

    #[test]
    fn one_source_streams_to_the_path_that_was_chosen() {
        // The daemon splits only in separate mode and only with more than one
        // source; the window has to say the same or it names files that never
        // appear.
        assert_eq!(
            stream_targets(
                "/tmp/meeting.txt",
                SourceMode::Separate,
                Some(&["Yeti".to_string()])
            ),
            ["/tmp/meeting.txt"]
        );
        assert_eq!(
            stream_targets(
                "/tmp/meeting.txt",
                SourceMode::Combined,
                Some(&["Yeti".to_string(), "System audio".to_string()]),
            ),
            ["/tmp/meeting.txt"]
        );
    }

    #[test]
    fn a_source_that_cannot_be_resolved_names_no_files_at_all() {
        // The daemon skips a key whose device has gone and splits between
        // what is left, so with one of two unplugged it opens a single file
        // and this window named two -- neither of which was ever written. A
        // label naming files nothing goes to is worse than no label, so when
        // the count cannot be trusted only the chosen path is named.
        assert_eq!(
            stream_targets("/tmp/meeting.txt", SourceMode::Separate, None),
            ["/tmp/meeting.txt"]
        );
    }

    #[test]
    fn a_running_session_is_told_the_target_is_for_the_next_one() {
        // The writer is opened once, at session start, from the options the
        // session captured -- so choosing a file now cannot redirect the
        // session already running. The old text said "Appending to ..." the
        // moment the setting existed, which was a claim about a file this
        // session may never touch.
        let running = stream_hover(true);
        assert!(running.contains("next session"), "{running}");
        assert_ne!(running, stream_hover(false));
    }

    #[test]
    fn an_idle_window_is_told_what_the_button_is_for() {
        // Nothing is running, so there is no other session to distinguish it
        // from, and the sentence that explains why anyone would stream is the
        // useful one.
        let idle = stream_hover(false);
        assert!(!idle.contains("next session"), "{idle}");
        assert!(idle.contains("crashes"), "{idle}");
    }

    #[test]
    fn the_target_is_readable_without_hovering() {
        let one = stream_target_label(&["/tmp/notes.txt".to_string()], false)
            .expect("a chosen target has something to say");
        assert!(one.contains("/tmp/notes.txt"), "{one}");

        let split = stream_target_label(
            &[
                "/tmp/a-mic.txt".to_string(),
                "/tmp/a-system-audio.txt".to_string(),
            ],
            false,
        )
        .unwrap();
        assert!(split.contains("/tmp/a-mic.txt"), "{split}");
        assert!(split.contains("/tmp/a-system-audio.txt"), "{split}");

        assert_eq!(
            stream_target_label(&[], false),
            None,
            "nothing set, nothing to say"
        );
    }

    #[test]
    fn the_label_says_which_session_it_means() {
        // The setting can be changed mid-session and the running session
        // keeps the writer it opened, so a label that only stated the setting
        // read "Stream target: B.txt" while every fragment went to A.txt.
        let target = ["/tmp/b.txt".to_string()];
        let running = stream_target_label(&target, true).unwrap();
        let idle = stream_target_label(&target, false).unwrap();
        assert!(running.contains("Next session"), "{running}");
        assert_ne!(running, idle);
        assert!(idle.contains("/tmp/b.txt") && running.contains("/tmp/b.txt"));
    }

    #[test]
    fn clearing_the_file_promises_only_what_it_can_do() {
        // It said "Stop appending to a file" while a session was running,
        // which it cannot do: the writer is opened at session start and the
        // running session keeps it.
        let running = stream_clear_hover(true);
        assert!(running.contains("next one"), "{running}");
        assert!(
            !running.contains("Stop"),
            "it cannot stop the session running: {running}"
        );
        assert_ne!(running, stream_clear_hover(false));

        let note = stream_cleared_note(true);
        assert!(note.contains("next session"), "{note}");
        assert_ne!(note, stream_cleared_note(false));
    }

    #[test]
    fn a_chosen_file_is_confirmed_as_appended_to_and_not_replaced() {
        // `rfd`'s save dialog raises the operating system's "replace this
        // file?" prompt and this code appends. The dialog opens on every
        // press now, so that prompt is the ordinary path.
        let note = stream_chosen_note("/tmp/notes.txt");
        assert!(note.contains("/tmp/notes.txt"), "{note}");
        assert!(note.contains("Appending"), "{note}");
        assert!(note.contains("kept"), "say the file survives: {note}");
    }

    #[test]
    fn a_failure_is_not_reported_in_the_colour_of_success() {
        // The status line was painted green whatever it said. The likeliest
        // way this window fails is a stream target that cannot be opened, and
        // that was reported in the colour that means it worked.
        let (severity, text) = status_for(&Ok(Response::Error {
            message: "opening /mnt/gone/notes.txt to append: No such file".into(),
        }))
        .expect("an error has to reach the window");
        assert_eq!(severity, Severity::Error);
        assert_eq!(severity.colour(), theme::palette::DANGER);
        assert!(text.contains("No such file"), "{text}");
    }

    #[test]
    fn a_lost_daemon_is_a_failure_too() {
        // A transport failure took the same green path as a save.
        let (severity, text) =
            status_for(&Err(anyhow::anyhow!("the daemon is not listening")))
                .expect("a dropped socket has to reach the window");
        assert_eq!(severity, Severity::Error);
        assert!(text.contains("not listening"), "{text}");
    }

    #[test]
    fn a_completed_save_stays_green() {
        let (severity, text) = status_for(&Ok(Response::Saved {
            path: "/tmp/a.txt".into(),
        }))
        .expect("a save says where it went");
        assert_eq!(severity, Severity::Ok);
        assert_eq!(severity.colour(), theme::palette::SUCCESS);
        assert!(text.contains("/tmp/a.txt"), "{text}");
    }

    #[test]
    fn a_bare_acknowledgement_clears_the_line() {
        // Otherwise the last message stays up and reads as the answer to
        // whichever button was pressed next.
        assert!(status_for(&Ok(Response::Ok)).is_none());
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
