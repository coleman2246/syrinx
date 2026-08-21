//! A small control panel for parakeet dictation.
//!
//! Shows what it is listening to, whether the server is up, and what it heard
//! last. The source picker is the reason this exists: it can target a
//! microphone, all system audio, or a single application -- so you can
//! transcribe a video call or one browser tab without touching system defaults.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod tray;

use anyhow::Result;
use eframe::egui;
use syrinx_audio::{Source, SourceKind};
use syrinx_client::{
    Config, OutputMode, SessionHandle, SessionOptions, SessionState, Status, save,
};
use tray::{TrayCommand, TrayHandle, TrayState};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "syrinx_gui=info".into()),
        )
        .init();

    let config = Config::load(None)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([460.0, 340.0])
            .with_min_inner_size([380.0, 260.0])
            .with_title("Parakeet"),
        ..Default::default()
    };

    eframe::run_native(
        "Parakeet",
        options,
        Box::new(|_cc| Ok(Box::new(App::new(config)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

struct App {
    config: Config,
    mode: OutputMode,
    tray: Option<TrayHandle>,
    tray_rx: Option<tokio::sync::mpsc::UnboundedReceiver<TrayCommand>>,
    /// Set when the tray asks to quit, so the next frame can close the window.
    quitting: bool,
    /// Last saved path, shown so the user knows where it went.
    saved_to: Option<String>,
    sources: Vec<Source>,
    selected: Option<Source>,
    session: Option<SessionHandle>,
    state: SessionState,
    list_error: Option<String>,
}

impl App {
    fn new(config: Config) -> Self {
        let mode = config.mode;
        let (tray, tray_rx) = match tray::start() {
            Some((h, rx)) => (Some(h), Some(rx)),
            None => (None, None),
        };
        let mut app = Self {
            config,
            mode,
            tray,
            tray_rx,
            quitting: false,
            saved_to: None,
            sources: Vec::new(),
            selected: None,
            session: None,
            state: SessionState::default(),
            list_error: None,
        };
        app.refresh_sources();
        app
    }

    /// Re-enumerate, keeping the current selection if it still exists.
    ///
    /// Applications come and go and their node ids change, so selection is
    /// matched on the stable key rather than held by reference.
    fn refresh_sources(&mut self) {
        match syrinx_audio::list_sources() {
            Ok(list) => {
                self.list_error = None;
                let want = self
                    .selected
                    .as_ref()
                    .map(|s| s.stable_key())
                    .or_else(|| self.config.source_key.clone());
                self.selected = want
                    .and_then(|k| syrinx_audio::resolve(&list, &k))
                    // Default to a microphone, never to system audio: starting
                    // to record the speakers because it happened to sort first
                    // would be a surprising thing to do unasked.
                    .or_else(|| {
                        // Prefer a noise-suppressed input where one exists: on
                        // this setup rnnoise_source is the same microphone with
                        // a much lower noise floor, so defaulting to the raw
                        // device would be a worse choice for the same hardware.
                        list.iter()
                            .find(|s| {
                                s.kind == SourceKind::Microphone
                                    && s.stable_key().contains("rnnoise")
                            })
                            .or_else(|| list.iter().find(|s| s.kind == SourceKind::Microphone))
                            .cloned()
                    })
                    .or_else(|| list.first().cloned());
                self.sources = list;
            }
            Err(e) => self.list_error = Some(format!("{e:#}")),
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        let Some(source) = self.selected.clone() else {
            return;
        };
        let ctx = ctx.clone();
        self.session = Some(syrinx_client::session::start(
            SessionOptions {
                url: self.config.url.clone(),
                token: self.config.token.clone(),
                source,
                mode: self.mode,
            },
            move || ctx.request_repaint(),
        ));
    }
}

impl eframe::App for App {
    // eframe 0.36 hands the app a Ui directly rather than a Context.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(s) = &self.session {
            self.state = s.state();
            if self.state.status == Status::Idle {
                self.session = None;
            }
        }
        let running = self.session.is_some();
        let ctx = ui.ctx().clone();

        self.pump_tray(&ctx, running);
        if self.quitting {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        ui.add_space(4.0);
        self.status_row(ui);
        ui.add_space(8.0);
        self.source_row(ui, running);
        ui.add_space(4.0);
        self.mode_row(ui, running);
        ui.separator();
        // Reserve room for the separator, the controls row and any error line.
        let reserved = 90.0;
        let height = ui.available_height() - reserved;
        self.transcript_box(ui, height);
        ui.separator();
        self.controls(ui, &ctx, running);

        if let Some(e) = &self.state.error {
            ui.add_space(4.0);
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), e);
        }
        if let Some(e) = &self.list_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), e);
        }
        if let Some(p) = &self.saved_to {
            ui.colored_label(egui::Color32::from_rgb(120, 190, 120), format!("Saved to {p}"));
        }

        // Repaint while listening so the status stays live; when idle the app
        // is fully event-driven and uses no CPU.
        if running {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }
}

impl App {
    /// Drain tray commands and mirror current state back to the tray.
    fn pump_tray(&mut self, ctx: &egui::Context, running: bool) {
        let mut cmds = Vec::new();
        if let Some(rx) = &mut self.tray_rx {
            while let Ok(c) = rx.try_recv() {
                cmds.push(c);
            }
        }
        for c in cmds {
            match c {
                TrayCommand::Toggle => {
                    if running {
                        self.stop();
                    } else {
                        self.start(ctx);
                    }
                }
                TrayCommand::Start if !running => self.start(ctx),
                TrayCommand::Stop if running => self.stop(),
                TrayCommand::Start | TrayCommand::Stop => {}
                TrayCommand::SetMode(m) if !running => self.mode = m,
                TrayCommand::SetMode(_) => {}
                TrayCommand::ShowWindow => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                TrayCommand::Quit => self.quitting = true,
            }
        }
        if let Some(tray) = &self.tray {
            tray.update(TrayState {
                status: self.state.status,
                mode: self.mode,
                last_fragment: self.state.last_fragment.clone(),
            });
        }
    }

    fn stop(&mut self) {
        if let Some(s) = &mut self.session {
            s.stop();
        }
    }

    /// Save the transcript, remembering where it went.
    fn save_transcript(&mut self, pick_path: bool) {
        let text = self.state.transcript.clone();
        let result = if pick_path {
            match rfd::FileDialog::new()
                .set_file_name(save::filename_for(&save::timestamp()))
                .set_directory(save::default_dir())
                .save_file()
            {
                Some(p) => save::write(&p, &text).map(|_| p),
                // Cancelled: not an error, and not something to report.
                None => return,
            }
        } else {
            save::save_default(&text)
        };
        match result {
            Ok(p) => {
                self.saved_to = Some(p.display().to_string());
                self.state.error = None;
            }
            Err(e) => self.state.error = Some(format!("{e:#}")),
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
            ui.colored_label(colour, egui::RichText::new(self.state.status.label()).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let (Some(m), Some(c)) = (&self.state.model, self.state.chunk_ms) {
                    ui.weak(format!("{m} · {c}ms chunks"));
                }
            });
        });
    }

    fn source_row(&mut self, ui: &mut egui::Ui, running: bool) {
        ui.horizontal(|ui| {
            ui.label("Source:");
            // Changing source mid-session would need a reconnect; simpler and
            // clearer to require stopping first.
            ui.add_enabled_ui(!running, |ui| {
                let label = self
                    .selected
                    .as_ref()
                    .map(|s| s.display())
                    .unwrap_or_else(|| "none".into());
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
                            let chosen = self.selected.as_ref() == Some(s);
                            if ui.selectable_label(chosen, s.display()).clicked() {
                                self.selected = Some(s.clone());
                            }
                        }
                    });
                if ui.button("⟳").on_hover_text("Rescan sources").clicked() {
                    self.refresh_sources();
                }
            });
        });
    }

    /// Fills whatever vertical space is left. A fixed height wastes most of the
    /// window when tiled, which on Sway is the normal case.
    fn mode_row(&mut self, ui: &mut egui::Ui, running: bool) {
        ui.horizontal(|ui| {
            ui.label("Mode:");
            // Changing mode mid-session would need a reconnect, since the wire
            // mode is fixed at session.start.
            ui.add_enabled_ui(!running, |ui| {
                for m in OutputMode::ALL {
                    if ui.selectable_label(self.mode == m, m.label()).clicked() {
                        self.mode = m;
                    }
                }
            });
        });
        if self.mode.types_at_cursor() {
            ui.weak("Types into whatever window has focus. Append-only: it never deletes.");
        }
    }

    fn transcript_box(&self, ui: &mut egui::Ui, height: f32) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .max_height(height.max(80.0))
            .show(ui, |ui| {
                if self.state.transcript.is_empty() {
                    if self.mode == OutputMode::Type {
                        ui.weak(if self.state.last_fragment.is_empty() {
                            "Typing at the cursor. Nothing spoken yet."
                        } else {
                            "Typing at the cursor."
                        });
                        if !self.state.last_fragment.is_empty() {
                            ui.label(
                                egui::RichText::new(format!("last: {}", self.state.last_fragment))
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
        ui.horizontal(|ui| {
            if running {
                if ui.button("Stop").clicked()
                    && let Some(s) = &mut self.session
                {
                    s.stop();
                }
            } else if ui
                .add_enabled(self.selected.is_some(), egui::Button::new("Start"))
                .clicked()
            {
                self.start(ctx);
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
                self.save_transcript(false);
            }
            if ui
                .add_enabled(has_text, egui::Button::new("Save as…"))
                .clicked()
            {
                self.save_transcript(true);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let short = self
                    .config
                    .url
                    .trim_start_matches("ws://")
                    .trim_end_matches("/v1/stream");
                ui.weak(short).on_hover_text(&self.config.url);
            });
        });
    }
}
