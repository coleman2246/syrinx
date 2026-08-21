//! A small control panel for parakeet dictation.
//!
//! Shows what it is listening to, whether the server is up, and what it heard
//! last. The source picker is the reason this exists: it can target a
//! microphone, all system audio, or a single application -- so you can
//! transcribe a video call or one browser tab without touching system defaults.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod session;

use anyhow::Result;
use eframe::egui;
use parakeet_audio::{Source, SourceKind};
use serde::Deserialize;
use session::{SessionHandle, SessionState, Status};
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
struct Config {
    #[serde(default = "default_url")]
    url: String,
    token: String,
    /// Remembered source, as a `Source::stable_key`. Node ids change between
    /// runs, so the key is what gets persisted.
    #[serde(default)]
    source_key: Option<String>,
}

fn default_url() -> String {
    "ws://127.0.0.1:8770/v1/stream".into()
}

fn config_path() -> PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".config")
        });
    base.join("parakeet-gui/config.toml")
}

fn load_config() -> Result<Config> {
    // Fall back to the dictation client's config: the two want the same server
    // and token, and making the user write it twice invites them to diverge.
    let mut candidates = vec![config_path()];
    if let Some(p) = config_path().parent().and_then(|p| p.parent()) {
        candidates.push(p.join("parakeet-type/config.toml"));
    }
    for p in &candidates {
        if let Ok(text) = std::fs::read_to_string(p) {
            return Ok(toml::from_str(&text)?);
        }
    }
    anyhow::bail!(
        "no config found. Create {} with:\n  url = \"ws://host:8770/v1/stream\"\n  token = \"...\"",
        config_path().display()
    )
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "parakeet_gui=info".into()),
        )
        .init();

    let config = load_config()?;
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
    sources: Vec<Source>,
    selected: Option<Source>,
    session: Option<SessionHandle>,
    state: SessionState,
    list_error: Option<String>,
}

impl App {
    fn new(config: Config) -> Self {
        let mut app = Self {
            config,
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
        match parakeet_audio::list_sources() {
            Ok(list) => {
                self.list_error = None;
                let want = self
                    .selected
                    .as_ref()
                    .map(|s| s.stable_key())
                    .or_else(|| self.config.source_key.clone());
                self.selected = want
                    .and_then(|k| parakeet_audio::resolve(&list, &k))
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
        self.session = Some(session::start(
            self.config.url.clone(),
            self.config.token.clone(),
            source,
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

        ui.add_space(4.0);
        self.status_row(ui);
        ui.add_space(8.0);
        self.source_row(ui, running);
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

        // Repaint while listening so the status stays live; when idle the app
        // is fully event-driven and uses no CPU.
        if running {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
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
    fn transcript_box(&self, ui: &mut egui::Ui, height: f32) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .max_height(height.max(80.0))
            .show(ui, |ui| {
                if self.state.transcript.is_empty() {
                    ui.weak("Nothing transcribed yet.");
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

            if ui.button("Copy").clicked() {
                ctx.copy_text(self.state.transcript.clone());
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
