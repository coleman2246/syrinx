//! The Windows tray icon and global hotkey, on one thread.
//!
//! Both `tray-icon` and `global-hotkey` are thread-affine on Windows: each
//! creates a hidden window, and the messages that drive it are delivered to the
//! thread that created it. They therefore share a single thread running a
//! single message loop. Splitting them would mean two pumps for no gain.
//!
//! The daemon's own loop is a poll over channels, not a message loop, so this
//! cannot live there. Events are forwarded to the daemon as [`TrayCommand`],
//! the same type the Linux tray sends, so the daemon does not know or care
//! which platform produced a command.

use crate::tray::{TrayCommand, TrayState};
use crate::{OutputMode, Status};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use std::sync::mpsc::{Receiver as StdReceiver, Sender as StdSender};
use tokio::sync::mpsc;
use tracing::{info, warn};
use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIconBuilder, TrayIconEvent};

/// Size of the generated tray icon, in pixels.
///
/// Windows asks for 16x16 in the notification area and scales what it is given.
/// Drawing at 32 and letting it downscale looks better than drawing at 16.
const ICON: u32 = 32;

/// Start the tray and, if asked, a global hotkey.
///
/// Returns the update channel for the tray and the command stream from it.
pub fn start(
    hotkey: Option<crate::hotkey::HotKey>,
) -> Option<(StdSender<TrayState>, mpsc::UnboundedReceiver<TrayCommand>)> {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<TrayCommand>();
    let (state_tx, state_rx) = std::sync::mpsc::channel::<TrayState>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<bool>();

    std::thread::spawn(move || run(hotkey, cmd_tx, state_rx, ready_tx));

    // Wait for the thread to say whether it got a tray at all, so the daemon can
    // log "running headless" truthfully rather than optimistically.
    match ready_rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(true) => Some((state_tx, cmd_rx)),
        _ => None,
    }
}

/// Menu item ids, so a click can be matched back to a command.
struct Ids {
    toggle: MenuId,
    transcribe: MenuId,
    type_: MenuId,
    both: MenuId,
    show: MenuId,
    quit: MenuId,
}

fn run(
    hotkey: Option<crate::hotkey::HotKey>,
    cmd_tx: mpsc::UnboundedSender<TrayCommand>,
    state_rx: StdReceiver<TrayState>,
    ready_tx: StdSender<bool>,
) {
    let toggle = MenuItem::new("Start / stop dictation", true, None);
    let transcribe = MenuItem::new("Mode: transcribe", true, None);
    let type_ = MenuItem::new("Mode: type at cursor", true, None);
    let both = MenuItem::new("Mode: both", true, None);
    let show = MenuItem::new("Open window", true, None);
    let quit = MenuItem::new("Quit syrinx", true, None);
    let ids = Ids {
        toggle: toggle.id().clone(),
        transcribe: transcribe.id().clone(),
        type_: type_.id().clone(),
        both: both.id().clone(),
        show: show.id().clone(),
        quit: quit.id().clone(),
    };

    let menu = Menu::new();
    let built = menu.append_items(&[
        &toggle,
        &PredefinedMenuItem::separator(),
        &transcribe,
        &type_,
        &both,
        &PredefinedMenuItem::separator(),
        &show,
        &quit,
    ]);
    if let Err(e) = built {
        warn!("building the tray menu: {e}");
        let _ = ready_tx.send(false);
        return;
    }

    let tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Syrinx — idle")
        .with_icon(icon_for(Status::Idle))
        .build();

    let tray = match tray {
        Ok(t) => t,
        Err(e) => {
            // No notification area is survivable; the CLI still works.
            info!("running without a system tray: {e}");
            let _ = ready_tx.send(false);
            return;
        }
    };

    // Registered on this thread, because the hotkey's window belongs to it.
    // Kept alive for the life of the thread: dropping the manager unregisters.
    let _hotkeys = hotkey.and_then(|h| match register(&h) {
        Ok(m) => {
            info!("hotkey {} registered", h.spelled);
            Some(m)
        }
        Err(e) => {
            // Almost always another application already owns the combination.
            warn!("could not register the hotkey {}: {e}", h.spelled);
            None
        }
    });

    let _ = ready_tx.send(true);

    let menu_events = MenuEvent::receiver();
    let tray_events = TrayIconEvent::receiver();
    let key_events = GlobalHotKeyEvent::receiver();
    let mut shown = TrayState::default();

    loop {
        pump();

        while let Ok(e) = menu_events.try_recv() {
            let cmd = if e.id == ids.toggle {
                Some(TrayCommand::Toggle)
            } else if e.id == ids.transcribe {
                Some(TrayCommand::SetMode(OutputMode::Transcribe))
            } else if e.id == ids.type_ {
                Some(TrayCommand::SetMode(OutputMode::Type))
            } else if e.id == ids.both {
                Some(TrayCommand::SetMode(OutputMode::Both))
            } else if e.id == ids.show {
                Some(TrayCommand::ShowWindow)
            } else if e.id == ids.quit {
                Some(TrayCommand::Quit)
            } else {
                None
            };
            if let Some(c) = cmd
                && cmd_tx.send(c).is_err()
            {
                return; // the daemon has gone
            }
        }

        // Left click starts or stops, matching the Linux tray.
        while let Ok(e) = tray_events.try_recv() {
            if let TrayIconEvent::Click { button, .. } = e
                && button == tray_icon::MouseButton::Left
                && cmd_tx.send(TrayCommand::Toggle).is_err()
            {
                return;
            }
        }

        // Only the press, not the release: otherwise one keystroke toggles
        // twice and dictation ends the instant it starts.
        while let Ok(e) = key_events.try_recv() {
            if e.state == HotKeyState::Pressed && cmd_tx.send(TrayCommand::Toggle).is_err() {
                return;
            }
        }

        // Latest state wins; intermediate frames are not worth drawing.
        let mut latest = None;
        while let Ok(s) = state_rx.try_recv() {
            latest = Some(s);
        }
        if let Some(s) = latest
            && (s.status != shown.status || s.mode != shown.mode)
        {
            let _ = tray.set_icon(Some(icon_for(s.status)));
            let _ = tray.set_tooltip(Some(tooltip(&s)));
            shown = s;
        }

        std::thread::sleep(std::time::Duration::from_millis(30));
    }
}

fn tooltip(s: &TrayState) -> String {
    match s.status {
        Status::Listening => format!("Syrinx — listening ({})", s.mode.label()),
        Status::Idle => "Syrinx — idle".into(),
        other => format!("Syrinx — {}", other.label()),
    }
}

fn register(h: &crate::hotkey::HotKey) -> anyhow::Result<GlobalHotKeyManager> {
    use global_hotkey::hotkey::{Code, HotKey as GHotKey, Modifiers};
    use std::str::FromStr;

    let mut mods = Modifiers::empty();
    if h.mods.ctrl {
        mods |= Modifiers::CONTROL;
    }
    if h.mods.alt {
        mods |= Modifiers::ALT;
    }
    if h.mods.shift {
        mods |= Modifiers::SHIFT;
    }
    if h.mods.meta {
        mods |= Modifiers::META;
    }
    let code = Code::from_str(&h.code)
        .map_err(|_| anyhow::anyhow!("{} is not a key this platform knows", h.code))?;

    let manager = GlobalHotKeyManager::new()?;
    manager.register(GHotKey::new(Some(mods), code))?;
    Ok(manager)
}

/// Drain the thread's message queue so the hidden windows behind the tray and
/// the hotkey actually receive their messages.
fn pump() {
    use windows::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, MSG, PM_REMOVE, PeekMessageW, TranslateMessage,
    };
    let mut msg = MSG::default();
    unsafe {
        while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// Draw the tray icon.
///
/// Generated rather than shipped as a file: it is a coloured disc, and a build
/// that renders one in nine lines does not need an asset pipeline, an icon that
/// can go missing, or a second thing to keep in step with the Linux names.
fn icon_for(status: Status) -> Icon {
    Icon::from_rgba(icon_rgba(status), ICON, ICON)
        .expect("the generated icon is always well formed")
}

/// The icon's pixels, split out so they can be checked without a Windows
/// handle: `Icon` exposes nothing once built.
fn icon_rgba(status: Status) -> Vec<u8> {
    let (r, g, b) = match status {
        // A record dot, the same meaning as the Linux `media-record` icon.
        Status::Listening => (235u8, 70u8, 60u8),
        Status::Connecting | Status::Stopping | Status::Transcribing => (235, 190, 70),
        Status::Idle => (150, 150, 158),
    };

    let mut rgba = Vec::with_capacity((ICON * ICON * 4) as usize);
    let centre = (ICON as f32 - 1.0) / 2.0;
    let radius = ICON as f32 / 2.0 - 1.0;
    for y in 0..ICON {
        for x in 0..ICON {
            let dx = x as f32 - centre;
            let dy = y as f32 - centre;
            let d = (dx * dx + dy * dy).sqrt();
            // One pixel of feathering, so the disc is not visibly jagged.
            let a = ((radius - d).clamp(0.0, 1.0) * 255.0) as u8;
            rgba.extend_from_slice(&[r, g, b, a]);
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Status; 5] = [
        Status::Idle,
        Status::Listening,
        Status::Connecting,
        Status::Stopping,
        Status::Transcribing,
    ];

    #[test]
    fn every_icon_has_exactly_the_pixels_its_dimensions_promise() {
        // Icon::from_rgba rejects a mismatch, and it is built with `expect`.
        for s in ALL {
            assert_eq!(icon_rgba(s).len(), (ICON * ICON * 4) as usize, "{s:?}");
        }
    }

    #[test]
    fn listening_is_visibly_different_from_idle() {
        // The icon is the only signal that the microphone is live, so these two
        // must never render the same.
        assert_ne!(icon_rgba(Status::Idle), icon_rgba(Status::Listening));
    }

    #[test]
    fn the_icon_is_a_disc_not_a_square() {
        // A square would be a filled block in the tray rather than a dot.
        let px = icon_rgba(Status::Listening);
        let at = |x: u32, y: u32| px[((y * ICON + x) * 4 + 3) as usize];
        assert_eq!(at(0, 0), 0, "the corner should be transparent");
        assert_eq!(at(ICON - 1, ICON - 1), 0, "the corner should be transparent");
        assert_eq!(at(ICON / 2, ICON / 2), 255, "the centre should be opaque");
    }
}
