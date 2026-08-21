//! System tray icon via StatusNotifierItem.
//!
//! Uses `ksni` rather than a GTK-based tray library: SNI is pure D-Bus, which
//! is what waybar consumes, and it avoids running a GTK main loop alongside
//! eframe's winit loop.
//!
//! Linux uses ksni here. Windows needs a different crate and a message loop of
//! its own, which lives in `windows_ui`; both feed the same [`TrayCommand`]
//! channel, so nothing downstream knows which one is running.

use std::sync::Mutex;
use crate::{OutputMode, Status};
use tokio::sync::mpsc;

/// A request from the tray menu to the application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayCommand {
    Toggle,
    Start,
    Stop,
    SetMode(OutputMode),
    ShowWindow,
    Quit,
}

/// What the tray needs to render itself. Updated by the app each time state
/// moves.
#[derive(Debug, Clone, Default)]
pub struct TrayState {
    pub status: Status,
    pub mode: OutputMode,
    pub last_fragment: String,
}

impl TrayState {
    /// Whether a redraw would change anything the tray displays.
    fn same_as(&self, other: &TrayState) -> bool {
        self.status == other.status
            && self.mode == other.mode
            && self.last_fragment == other.last_fragment
    }
}

/// Handle to a running tray. Dropping it removes the icon.
pub struct TrayHandle {
    /// ksni's blocking handle. Synchronous `update`, no runtime to manage.
    ///
    /// The async API was tried first and panics: zbus spawns its own thread and
    /// calls `tokio::spawn` from it, outside any runtime context, so it dies
    /// with "there is no reactor running" regardless of what runtime the caller
    /// provides. The blocking API owns its threading and avoids the question.
    #[cfg(target_os = "linux")]
    handle: ksni::blocking::Handle<SyrinxTray>,
    /// The Windows tray lives on its own message-loop thread and is not `Send`,
    /// so it is updated by message rather than by call.
    #[cfg(windows)]
    updates: std::sync::mpsc::Sender<TrayState>,
    /// Last state sent, so an unchanged frame does not spam D-Bus at the
    /// repaint rate.
    last: Mutex<Option<TrayState>>,
}

impl TrayHandle {
    /// Push new state to the tray so the icon and menu reflect reality.
    pub fn update(&self, state: TrayState) {
        let mut last = self.last.lock().expect("tray state lock poisoned");
        if last.as_ref().is_some_and(|l| l.same_as(&state)) {
            return;
        }
        *last = Some(state.clone());
        #[cfg(target_os = "linux")]
        {
            self.handle.update(move |t: &mut SyrinxTray| t.state = state);
        }
        #[cfg(windows)]
        {
            let _ = self.updates.send(state);
        }
    }
}

#[cfg(target_os = "linux")]
struct SyrinxTray {
    state: TrayState,
    tx: mpsc::UnboundedSender<TrayCommand>,
}

#[cfg(target_os = "linux")]
impl ksni::Tray for SyrinxTray {
    fn id(&self) -> String {
        "syrinx".into()
    }

    fn title(&self) -> String {
        match self.state.status {
            Status::Listening => format!("Syrinx — listening ({})", self.state.mode.label()),
            Status::Idle => "Syrinx — idle".into(),
            other => format!("Syrinx — {}", other.label()),
        }
    }

    /// Named icons rather than bundled pixmaps: they follow the user's icon
    /// theme, so the tray matches the rest of the desktop.
    ///
    /// These particular names are chosen for availability, not just meaning.
    /// `audio-input-microphone-muted` reads perfectly but is absent from
    /// breeze-dark, where it rendered as a generic "prohibited" glyph -- an icon
    /// that says "broken" rather than "idle". `media-record`,
    /// `audio-input-microphone` and `microphone-sensitivity-muted` are all
    /// present in Breeze and Adwaita.
    fn icon_name(&self) -> String {
        match self.state.status {
            // A record dot is unambiguous about the microphone being live.
            Status::Listening => "media-record".into(),
            Status::Connecting | Status::Stopping | Status::Transcribing => {
                "audio-input-microphone".into()
            }
            Status::Idle => "microphone-sensitivity-muted".into(),
        }
    }

    /// Left click is the fast path: start or stop without opening a menu.
    fn activate(&mut self, _x: i32, _y: i32) {
        let _ = self.tx.send(TrayCommand::Toggle);
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{CheckmarkItem, MenuItem, StandardItem, SubMenu};

        let listening = self.state.status.is_active();
        // A disabled header showing status and mode. The tooltip is not always
        // visible -- waybar shows it only on hover -- so the menu has to state
        // what mode it is in rather than only letting you choose one.
        let mut items: Vec<MenuItem<Self>> = vec![
            StandardItem {
                label: format!(
                    "{}  ·  {}",
                    self.state.status.label(),
                    self.state.mode.label()
                ),
                enabled: false,
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: if listening { "Stop" } else { "Start" }.into(),
                icon_name: if listening {
                    "media-playback-stop".into()
                } else {
                    "media-record".into()
                },
                activate: Box::new(|t: &mut Self| {
                    let _ = t.tx.send(if t.state.status.is_active() {
                        TrayCommand::Stop
                    } else {
                        TrayCommand::Start
                    });
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
        ];

        // Mode is a submenu of checkmarks so the current one is visible at a
        // glance, which matters when the window is hidden.
        let mode_items: Vec<MenuItem<Self>> = OutputMode::ALL
            .iter()
            .map(|m| {
                let m = *m;
                CheckmarkItem {
                    label: m.label().into(),
                    checked: self.state.mode == m,
                    // Changing mode needs a reconnect, so it is only offered
                    // while idle.
                    enabled: !self.state.status.is_active(),
                    activate: Box::new(move |t: &mut Self| {
                        let _ = t.tx.send(TrayCommand::SetMode(m));
                    }),
                    ..Default::default()
                }
                .into()
            })
            .collect();

        items.push(
            SubMenu {
                // Named with the current value so the mode is legible without
                // opening the submenu.
                label: format!("Mode: {}", self.state.mode.label()),
                submenu: mode_items,
                ..Default::default()
            }
            .into(),
        );

        if !self.state.last_fragment.is_empty() {
            items.push(MenuItem::Separator);
            items.push(
                StandardItem {
                    label: truncate(&self.state.last_fragment, 40),
                    enabled: false,
                    ..Default::default()
                }
                .into(),
            );
        }

        items.push(MenuItem::Separator);
        items.push(
            StandardItem {
                label: "Show window".into(),
                activate: Box::new(|t: &mut Self| {
                    let _ = t.tx.send(TrayCommand::ShowWindow);
                }),
                ..Default::default()
            }
            .into(),
        );
        items.push(
            StandardItem {
                label: "Quit".into(),
                icon_name: "application-exit".into(),
                activate: Box::new(|t: &mut Self| {
                    let _ = t.tx.send(TrayCommand::Quit);
                }),
                ..Default::default()
            }
            .into(),
        );
        items
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    format!("{}…", s.chars().take(max - 1).collect::<String>())
}

/// Start the tray, returning a handle and the channel its menu sends on.
///
/// Returns `None` where no tray is available -- no SNI host running, or a
/// platform without an implementation. A missing tray must never stop the GUI
/// starting.
pub struct Tray {
    pub handle: TrayHandle,
    pub commands: mpsc::UnboundedReceiver<TrayCommand>,
    /// Why the hotkey did not register, where it was tried and failed.
    ///
    /// Carried out of here because registration happens on the tray's thread
    /// on Windows -- the two share a message loop -- and the daemon has to
    /// report the outcome rather than assume it.
    pub hotkey_error: Option<String>,
}

pub fn start(hotkey: Option<crate::hotkey::HotKey>) -> Option<Tray> {
    // Annotated: on platforms without a tray, nothing else pins the element
    // type and inference has nowhere to look.
    let (tx, rx) = mpsc::unbounded_channel::<TrayCommand>();

    #[cfg(target_os = "linux")]
    {
        // Linux hotkeys are registered separately: ksni has no message loop to
        // share, and on Wayland there is nothing to register at all.
        let _ = hotkey;
        use ksni::blocking::TrayMethods;
        let tray = SyrinxTray {
            state: TrayState::default(),
            tx,
        };
        match tray.spawn() {
            Ok(handle) => Some(Tray {
                handle: TrayHandle {
                    handle,
                    last: Mutex::new(None),
                },
                commands: rx,
                // Registered separately on Linux, so nothing to report here.
                hotkey_error: None,
            }),
            Err(e) => {
                // No SNI host running is normal, not an error worth failing on.
                tracing::info!("running without a system tray: {e}");
                None
            }
        }
    }

    #[cfg(windows)]
    {
        // The Windows tray makes its own channel, because its thread has to
        // own both ends of the message loop.
        let _ = (tx, rx);
        let (updates, cmds, hotkey_error) = crate::windows_ui::start(hotkey)?;
        Some(Tray {
            handle: TrayHandle {
                updates,
                last: Mutex::new(None),
            },
            commands: cmds,
            hotkey_error,
        })
    }

    #[cfg(not(any(target_os = "linux", windows)))]
    {
        // Both halves are dropped: without a tray there is nothing to send on
        // and nothing to receive from.
        let _ = (tx, rx, hotkey);
        tracing::info!("system tray is not implemented on this platform");
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_fragments_are_truncated_for_the_menu() {
        assert!(truncate(&"x".repeat(100), 40).chars().count() <= 40);
    }

    #[test]
    fn short_fragments_are_left_alone() {
        assert_eq!(truncate("hello", 40), "hello");
    }

    #[test]
    fn identical_state_is_recognised_so_the_tray_is_not_spammed() {
        // update() is called every repaint; without this check the GUI would
        // hammer D-Bus at the frame rate.
        let a = TrayState::default();
        assert!(a.same_as(&TrayState::default()));
    }

    #[test]
    fn a_changed_status_is_not_treated_as_identical() {
        let a = TrayState::default();
        let b = TrayState {
            status: Status::Listening,
            ..Default::default()
        };
        assert!(!a.same_as(&b));
    }

    #[test]
    fn truncation_counts_characters_not_bytes() {
        // Byte-slicing a multi-byte transcript would panic mid-character.
        let _ = truncate(&"日本語".repeat(40), 10);
    }
}
