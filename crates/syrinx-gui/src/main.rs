//! A window onto the syrinx daemon.
//!
//! The GUI owns no session. The daemon does, along with the tray icon, and this
//! attaches to it over a Unix socket. That is what lets the window be closed
//! without stopping dictation: winit documents `set_visible` as unsupported on
//! Wayland, so a window cannot hide itself and keep running, and something that
//! never had a window has to hold the session instead.
//!
//! Starting the GUI starts a daemon if none is running, so opening the window is
//! enough to get a tray, and closing it leaves both alive.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use eframe::egui;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use syrinx_client::{
    Config, OutputMode, Source, SourceKind, ipc,
    ipc::{DaemonState, Request, Response},
    save,
    session::Status,
};

/// How often the daemon is polled. Fast enough to feel live, slow enough that
/// an idle window is not doing constant socket work.
const POLL_INTERVAL: Duration = Duration::from_millis(200);

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "syrinx_gui=info,syrinx_client=info".into()),
        )
        .init();

    let config = Config::load(None)?;
    ensure_daemon()?;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([480.0, 380.0])
            .with_min_inner_size([400.0, 280.0])
            .with_title("Syrinx"),
        ..Default::default()
    };

    eframe::run_native(
        "Syrinx",
        options,
        Box::new(|_cc| Ok(Box::new(App::new(config)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
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
        .and_then(|p| p.parent().map(|d| d.join("syrinx")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("syrinx"));

    tracing::info!("starting the syrinx daemon");
    std::process::Command::new(&cmd)
        .arg("daemon")
        // Detached: the daemon must outlive this window.
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("starting the daemon via {}", cmd.display()))?;

    // Binding the socket takes a moment; poll rather than guess at a sleep.
    for _ in 0..50 {
        if ipc::daemon_running() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    anyhow::bail!("the daemon did not start listening within 5s")
}

struct App {
    config: Config,
    /// Mirrored from the daemon; the GUI owns no session.
    state: DaemonState,
    sources: Vec<Source>,
    save_format: save::Format,
    last_poll: Instant,
    /// Set when the socket drops, so the window can say so rather than showing
    /// stale state as if it were live.
    disconnected: bool,
    status_line: Option<String>,
    list_error: Option<String>,
    /// Server address being edited. Applied on Enter or the Apply button, not
    /// per keystroke, so a half-typed host is never dialled.
    url_edit: String,
    editing_url: bool,
}

impl App {
    fn new(config: Config) -> Self {
        let mut app = Self {
            config,
            state: DaemonState::default(),
            sources: Vec::new(),
            save_format: save::Format::default(),
            last_poll: Instant::now() - POLL_INTERVAL,
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
        match ipc::request(&Request::GetState) {
            Ok(Response::State(s)) => {
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

    fn selected_source(&self) -> Option<&Source> {
        let key = self.state.source_key.as_deref()?;
        self.sources.iter().find(|s| s.stable_key() == key)
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.last_poll.elapsed() >= POLL_INTERVAL {
            self.poll();
        }
        let running = self.state.status.is_active();

        ui.add_space(4.0);
        self.status_row(ui);
        ui.add_space(8.0);
        self.source_row(ui, running);
        ui.add_space(4.0);
        self.mode_row(ui, running);
        ui.separator();

        let reserved = 96.0;
        let height = ui.available_height() - reserved;
        self.transcript_box(ui, height);
        ui.separator();
        self.controls(ui, &ctx, running);

        if self.disconnected {
            ui.colored_label(
                egui::Color32::from_rgb(220, 80, 80),
                "Lost the daemon. Close and reopen this window to restart it.",
            );
        }
        if let Some(e) = &self.state.error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), e);
        }
        if let Some(e) = &self.list_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), e);
        }
        if let Some(s) = &self.status_line {
            ui.colored_label(egui::Color32::from_rgb(120, 190, 120), s);
        }

        // Poll steadily rather than only on interaction, since state changes
        // originate in the daemon and the tray, not just here.
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
                    .hint_text("ws://host:8770/v1/stream"),
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
                match self.apply_url() {
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
    fn apply_url(&mut self) -> Result<()> {
        let url = normalise_url(&self.url_edit)?;
        self.config.url = url.clone();
        self.config.save(&Config::default_path())?;
        // The daemon holds its own copy, so telling it is what actually takes
        // effect; writing the file only makes it survive a restart.
        match ipc::request(&Request::SetUrl { url })? {
            Response::Error { message } => anyhow::bail!(message),
            _ => Ok(()),
        }
    }

    fn status_row(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let (colour, dot) = match self.state.status {
                Status::Listening => (egui::Color32::from_rgb(220, 60, 60), "●"),
                Status::Connecting | Status::Stopping => {
                    (egui::Color32::from_rgb(220, 170, 60), "●")
                }
                Status::Idle => (egui::Color32::GRAY, "○"),
            };
            ui.colored_label(colour, egui::RichText::new(dot).size(18.0));
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
        let mut chosen: Option<String> = None;
        ui.horizontal(|ui| {
            ui.label("Source:");
            ui.add_enabled_ui(!running, |ui| {
                let label = self
                    .selected_source()
                    .map(|s| s.display())
                    .unwrap_or_else(|| "default".into());
                egui::ComboBox::from_id_salt("source")
                    .selected_text(label)
                    .width(300.0)
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
                            let is = self.state.source_key.as_deref() == Some(&s.stable_key());
                            if ui.selectable_label(is, s.display()).clicked() {
                                chosen = Some(s.stable_key());
                            }
                        }
                    });
                if ui.button("⟳").on_hover_text("Rescan sources").clicked() {
                    self.refresh_sources();
                }
            });
        });
        if let Some(key) = chosen {
            self.send(Request::SetSource { key });
        }
    }

    fn mode_row(&mut self, ui: &mut egui::Ui, running: bool) {
        let mut chosen: Option<OutputMode> = None;
        ui.horizontal(|ui| {
            ui.label("Mode:");
            // Fixed at session start on the wire, so changing it needs a
            // reconnect.
            ui.add_enabled_ui(!running, |ui| {
                for m in OutputMode::ALL {
                    if ui
                        .selectable_label(self.state.mode == m, m.label())
                        .clicked()
                    {
                        chosen = Some(m);
                    }
                }
            });
        });
        if self.state.mode.types_at_cursor() {
            ui.weak("Types into whatever window has focus. Append-only: it never deletes.");
        }
        if let Some(mode) = chosen {
            self.send(Request::SetMode { mode });
        }
    }

    fn transcript_box(&self, ui: &mut egui::Ui, height: f32) {
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
                } else {
                    ui.label(&self.state.transcript);
                }
            });
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
                            })
                            .clicked()
                        {
                            self.save_format = f;
                        }
                    }
                });
            let has_text2 = !self.state.transcript.trim().is_empty();
            if ui
                .add_enabled(has_text2 && !running, egui::Button::new("Clear"))
                .on_hover_text("Discard the transcript")
                .clicked()
            {
                action = Some(Request::Clear);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let short = self
                    .config
                    .url
                    .trim_start_matches("ws://")
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
fn normalise_url(input: &str) -> Result<String> {
    let s = input.trim();
    if s.is_empty() {
        anyhow::bail!("enter a server address");
    }
    if s.starts_with("ws://") || s.starts_with("wss://") {
        return Ok(s.to_string());
    }
    let with_port = if s.contains(':') {
        s.to_string()
    } else {
        format!("{s}:8770")
    };
    Ok(format!("ws://{with_port}/v1/stream"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_host_becomes_a_full_url() {
        assert_eq!(
            normalise_url("192.168.0.11").unwrap(),
            "ws://192.168.0.11:8770/v1/stream"
        );
    }

    #[test]
    fn a_host_and_port_keeps_the_port() {
        assert_eq!(
            normalise_url("acdc.home.arpa:9000").unwrap(),
            "ws://acdc.home.arpa:9000/v1/stream"
        );
    }

    #[test]
    fn a_full_url_is_left_alone() {
        let u = "ws://host:1234/v1/stream";
        assert_eq!(normalise_url(u).unwrap(), u);
        assert_eq!(normalise_url("wss://host/v1/stream").unwrap(), "wss://host/v1/stream");
    }

    #[test]
    fn surrounding_whitespace_is_ignored() {
        assert_eq!(
            normalise_url("  10.0.0.5  ").unwrap(),
            "ws://10.0.0.5:8770/v1/stream"
        );
    }

    #[test]
    fn an_empty_address_is_an_error() {
        assert!(normalise_url("").is_err());
        assert!(normalise_url("   ").is_err());
    }
}
