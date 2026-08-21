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
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let short = self
                    .config
                    .url
                    .trim_start_matches("ws://")
                    .trim_end_matches("/v1/stream");
                ui.weak(short).on_hover_text(&self.config.url);
            });
        });
        ui.weak("Closing this window leaves dictation running in the tray.");
        if let Some(a) = action {
            self.send(a);
        }
    }
}
