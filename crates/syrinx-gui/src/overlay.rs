//! A small always-on-top level display for typing modes.
//!
//! In transcribe mode the window already shows text arriving, which is its own
//! feedback. Typing at the cursor gives none: text appears in whatever has
//! focus, and if nothing appears there is no way to tell whether the microphone
//! is dead, the server is down, or nobody has spoken yet.
//!
//! Deliberately not shown for transcribe-only. An always-on-top window over
//! whatever you are reading is an intrusion that has to earn its place.

use eframe::egui;
use syrinx_client::{ipc, ipc::Request, session::Status};

/// Poll interval. Faster than the main window because this exists purely to
/// show movement.
const POLL: std::time::Duration = std::time::Duration::from_millis(120);

pub fn run() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([260.0, 54.0])
            .with_decorations(false)
            .with_always_on_top()
            .with_resizable(false)
            // Compositors that honour it will skip this in the taskbar; the
            // overlay is a readout, not a window to manage.
            .with_taskbar(false)
            .with_title("Syrinx level")
            // A distinct app id so a tiling compositor can be told to float
            // this. Undecorated is not the same as floating: without a rule
            // Sway tiles it like any other window, which is the opposite of an
            // overlay. See the README for the one-line rule.
            .with_app_id("syrinx-overlay"),
        ..Default::default()
    };
    eframe::run_native(
        "syrinx-overlay",
        options,
        Box::new(|_cc| Ok(Box::new(Overlay::default()))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[derive(Default)]
struct Overlay {
    levels: Vec<f32>,
    status: Status,
    gone: bool,
}

impl eframe::App for Overlay {
    /// Transparent background so the readout floats rather than sitting in a
    /// grey slab.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.85]
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        match ipc::request(&Request::GetState) {
            Ok(ipc::Response::State(s)) => {
                self.levels = s.levels;
                self.status = s.status;
                // The session ending is what closes this, so it never outlives
                // the thing it is reporting on.
                if !s.status.is_active() {
                    self.gone = true;
                }
            }
            // A daemon that has gone away means there is nothing to report.
            Err(_) => self.gone = true,
            _ => {}
        }
        if self.gone {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        ui.horizontal(|ui| {
            ui.add_space(6.0);
            ui.colored_label(
                egui::Color32::from_rgb(220, 60, 60),
                egui::RichText::new("●").size(16.0),
            );
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(190.0, 34.0), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            let n = syrinx_audio::meter::BANDS;
            let gap = 2.0;
            let bar_w = (rect.width() - gap * (n as f32 + 1.0)) / n as f32;
            for i in 0..n {
                let v = self.levels.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let h = (rect.height() - 4.0) * v.max(0.03);
                let x = rect.left() + gap + i as f32 * (bar_w + gap);
                let bar = egui::Rect::from_min_size(
                    egui::pos2(x, rect.bottom() - 2.0 - h),
                    egui::vec2(bar_w, h),
                );
                let colour = if v > 0.85 {
                    egui::Color32::from_rgb(230, 90, 70)
                } else if v > 0.6 {
                    egui::Color32::from_rgb(230, 190, 70)
                } else {
                    egui::Color32::from_rgb(100, 200, 120)
                };
                painter.rect_filled(bar, 1.0, colour);
            }
        });

        ctx.request_repaint_after(POLL);
    }
}
