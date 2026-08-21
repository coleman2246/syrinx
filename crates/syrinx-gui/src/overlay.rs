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

/// Poll interval, ~30 Hz.
///
/// The audio path delivers a chunk every 32 ms and the daemon publishes state
/// every 25 ms, so this is the slowest link and the display is smooth rather
/// than stepping.
const POLL: std::time::Duration = std::time::Duration::from_millis(33);

/// How quickly a falling bar drops, as a fraction of the gap per frame.
///
/// Rising is instantaneous so a syllable registers immediately; falling is
/// eased, because bars that snap to zero between words read as dropouts rather
/// than as speech.
const FALL: f32 = 0.28;

pub fn run() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([250.0, 48.0])
            .with_decorations(false)
            .with_always_on_top()
            .with_resizable(false)
            // Real transparency rather than a dark fill: the readout sits over
            // whatever you are working in, and an opaque slab hides it.
            .with_transparent(true)
            .with_taskbar(false)
            .with_title("Syrinx level")
            // A distinct app id so a tiling compositor can be told to float
            // this. Undecorated is not the same as floating: without a rule
            // Sway tiles it like any other window, which is the opposite of an
            // overlay. See the README.
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
    /// Smoothed bar heights, so the display eases rather than flickers.
    shown: Vec<f32>,
    status: Status,
    gone: bool,
}

impl eframe::App for Overlay {
    /// Fully transparent: the panel below paints its own rounded, translucent
    /// background, which looks like an overlay rather than a window.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        let mut target: Vec<f32> = Vec::new();
        match ipc::request(&Request::GetState) {
            Ok(ipc::Response::State(s)) => {
                target = s.levels;
                self.status = s.status;
                // The session ending is what closes this, so it never outlives
                // the thing it reports on.
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

        let n = syrinx_audio::meter::BANDS;
        if self.shown.len() != n {
            self.shown = vec![0.0; n];
        }
        for i in 0..n {
            let t = target.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            if t >= self.shown[i] {
                self.shown[i] = t;
            } else {
                self.shown[i] += (t - self.shown[i]) * FALL;
            }
        }

        let full = ui.available_rect_before_wrap();
        let painter = ui.painter();
        painter.rect_filled(
            full,
            8.0,
            egui::Color32::from_rgba_unmultiplied(16, 16, 20, 170),
        );

        // Middle- or left-drag moves the window. A compositor-side move is the
        // only way that works on Wayland, where a client cannot position
        // itself.
        //
        // Sent once on drag START, never while dragging. StartDrag hands the
        // whole gesture to the compositor; repeating it every frame re-grabs
        // the pointer continuously, and the move never ends -- the window
        // follows the cursor with no way to release it short of killing the
        // process.
        let drag = ui.interact(
            full,
            egui::Id::new("overlay-drag"),
            egui::Sense::click_and_drag(),
        );
        if drag.drag_started_by(egui::PointerButton::Middle)
            || drag.drag_started_by(egui::PointerButton::Primary)
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
        }

        let pad = 8.0;
        let rect = full.shrink(pad);
        let gap = 3.0;
        let bar_w = (rect.width() - gap * (n as f32 - 1.0)) / n as f32;
        for i in 0..n {
            let v = self.shown[i];
            // A visible floor, so an idle meter reads as "running, quiet"
            // rather than "not running at all".
            let h = (rect.height() * v).max(2.0);
            let x = rect.left() + i as f32 * (bar_w + gap);
            let bar = egui::Rect::from_min_size(
                egui::pos2(x, rect.bottom() - h),
                egui::vec2(bar_w, h),
            );
            let colour = if v > 0.9 {
                egui::Color32::from_rgb(235, 90, 70)
            } else if v > 0.7 {
                egui::Color32::from_rgb(235, 190, 70)
            } else {
                egui::Color32::from_rgb(110, 205, 130)
            };
            painter.rect_filled(bar, 1.5, colour);
        }

        ctx.request_repaint_after(POLL);
    }
}
