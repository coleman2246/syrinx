//! A Fluent-style dark theme, in the register Microsoft Teams uses.
//!
//! The colours are one place rather than scattered through the widgets. They
//! were literals at every call site, which is how a interface drifts: two reds
//! that are nearly the same, a green that belongs to nothing else.
//!
//! The palette is Fluent 2's, which is what Teams is built from: a violet brand
//! colour, near-neutral greys for the surfaces, and status colours that carry
//! meaning rather than decoration.

use eframe::egui::{self, Color32, CornerRadius, Stroke};

/// Named colours. Use these rather than literals.
pub mod palette {
    use eframe::egui::Color32;

    /// Fluent brand violet. Selection, focus, the accent on anything active.
    pub const BRAND: Color32 = Color32::from_rgb(0x5B, 0x5F, 0xC7);
    /// Lifted for hover, because the brand colour on a dark ground needs to
    /// get brighter to read as "reacting", not darker.
    pub const BRAND_HOVER: Color32 = Color32::from_rgb(0x75, 0x79, 0xEB);
    pub const BRAND_PRESSED: Color32 = Color32::from_rgb(0x4F, 0x52, 0xB2);

    /// The window behind everything.
    pub const CANVAS: Color32 = Color32::from_rgb(0x1F, 0x1F, 0x1F);
    /// Cards, popups, the transcript box.
    pub const SURFACE: Color32 = Color32::from_rgb(0x29, 0x29, 0x29);
    /// A surface that sits on a surface: buttons, menu rows.
    pub const SURFACE_RAISED: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
    /// Sunken, for anything being typed into.
    pub const SUNKEN: Color32 = Color32::from_rgb(0x14, 0x14, 0x14);

    pub const BORDER: Color32 = Color32::from_rgb(0x40, 0x40, 0x40);

    pub const TEXT: Color32 = Color32::from_rgb(0xFF, 0xFF, 0xFF);
    pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0xC7, 0xC7, 0xC7);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x8A, 0x8A, 0x8A);

    /// Teams' red. Recording, and errors.
    pub const DANGER: Color32 = Color32::from_rgb(0xC4, 0x31, 0x4B);
    /// Brighter, for a dot that has to catch the eye against a dark ground.
    pub const RECORDING: Color32 = Color32::from_rgb(0xE0, 0x4A, 0x63);
    pub const WARNING: Color32 = Color32::from_rgb(0xF7, 0x9D, 0x3C);
    /// Teams' presence-available green.
    pub const SUCCESS: Color32 = Color32::from_rgb(0x6B, 0xB7, 0x00);
}

use palette as p;

/// Corner radius. Fluent rounds gently; anything more reads as a toy.
const RADIUS: u8 = 4;

/// Apply the theme to a context. Call once at startup.
pub fn apply(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();

    visuals.panel_fill = p::CANVAS;
    visuals.window_fill = p::SURFACE;
    visuals.extreme_bg_color = p::SUNKEN;
    visuals.faint_bg_color = p::SURFACE;
    visuals.code_bg_color = p::SUNKEN;
    visuals.window_stroke = Stroke::new(1.0, p::BORDER);
    visuals.window_corner_radius = CornerRadius::same(RADIUS + 2);
    visuals.menu_corner_radius = CornerRadius::same(RADIUS);

    visuals.selection.bg_fill = p::BRAND;
    visuals.selection.stroke = Stroke::new(1.0, p::TEXT);
    visuals.hyperlink_color = p::BRAND_HOVER;
    visuals.error_fg_color = p::DANGER;
    visuals.warn_fg_color = p::WARNING;

    let w = &mut visuals.widgets;

    // Labels and other things that are not interactive.
    w.noninteractive.bg_fill = p::SURFACE;
    w.noninteractive.weak_bg_fill = p::SURFACE;
    w.noninteractive.bg_stroke = Stroke::new(1.0, p::BORDER);
    w.noninteractive.fg_stroke = Stroke::new(1.0, p::TEXT_SECONDARY);
    w.noninteractive.corner_radius = CornerRadius::same(RADIUS);

    // Buttons at rest.
    w.inactive.bg_fill = p::SURFACE_RAISED;
    w.inactive.weak_bg_fill = p::SURFACE_RAISED;
    w.inactive.bg_stroke = Stroke::new(1.0, p::BORDER);
    w.inactive.fg_stroke = Stroke::new(1.0, p::TEXT);
    w.inactive.corner_radius = CornerRadius::same(RADIUS);

    // Hover lifts the surface and shows the brand colour on the edge, which is
    // how Fluent signals "this responds" without moving anything.
    w.hovered.bg_fill = Color32::from_rgb(0x3D, 0x3D, 0x3D);
    w.hovered.weak_bg_fill = Color32::from_rgb(0x3D, 0x3D, 0x3D);
    w.hovered.bg_stroke = Stroke::new(1.0, p::BRAND_HOVER);
    w.hovered.fg_stroke = Stroke::new(1.5, p::TEXT);
    w.hovered.corner_radius = CornerRadius::same(RADIUS);
    // No expansion: a button that grows on hover stops lining up with its
    // neighbours, which is worse than the emphasis is worth.
    w.hovered.expansion = 0.0;

    // Pressed and selected are the brand colour itself.
    w.active.bg_fill = p::BRAND_PRESSED;
    w.active.weak_bg_fill = p::BRAND_PRESSED;
    w.active.bg_stroke = Stroke::new(1.0, p::BRAND);
    w.active.fg_stroke = Stroke::new(1.5, p::TEXT);
    w.active.corner_radius = CornerRadius::same(RADIUS);
    w.active.expansion = 0.0;

    // An open dropdown.
    w.open.bg_fill = p::SURFACE_RAISED;
    w.open.weak_bg_fill = p::SURFACE_RAISED;
    w.open.bg_stroke = Stroke::new(1.0, p::BRAND);
    w.open.fg_stroke = Stroke::new(1.0, p::TEXT);
    w.open.corner_radius = CornerRadius::same(RADIUS);

    ctx.set_visuals(visuals);

    // Spacing applies to both themes; only the colours above are dark-specific.
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.spacing.menu_margin = egui::Margin::same(6);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_status_colour_is_distinct() {
        // These carry meaning -- recording, warning, done. Two that looked
        // alike would make the state unreadable at a glance.
        let colours = [p::RECORDING, p::WARNING, p::SUCCESS, p::BRAND];
        for (i, a) in colours.iter().enumerate() {
            for b in &colours[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn text_is_lighter_than_every_surface_it_sits_on() {
        // A palette where a text colour is darker than its background is
        // unreadable, and that is easy to do by mistyping one hex digit.
        let luma =
            |c: Color32| 0.2126 * c.r() as f32 + 0.7152 * c.g() as f32 + 0.0722 * c.b() as f32;
        for surface in [p::CANVAS, p::SURFACE, p::SURFACE_RAISED, p::SUNKEN] {
            for text in [p::TEXT, p::TEXT_SECONDARY, p::TEXT_MUTED] {
                assert!(
                    luma(text) > luma(surface) + 40.0,
                    "{text:?} on {surface:?} has too little contrast"
                );
            }
        }
    }

    #[test]
    fn hover_is_brighter_than_rest_and_pressed_is_darker() {
        // The direction matters: on a dark ground, brighter reads as
        // "reacting" and darker as "held down".
        let luma = |c: Color32| c.r() as u32 + c.g() as u32 + c.b() as u32;
        assert!(luma(p::BRAND_HOVER) > luma(p::BRAND));
        assert!(luma(p::BRAND_PRESSED) < luma(p::BRAND));
    }
}
