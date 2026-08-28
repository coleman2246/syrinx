//! Does `SendInput` actually put text into a window?
//!
//! Ignored by default: it needs an interactive desktop, so it fails in CI and
//! over SSH, where a process has no window station to draw on. Run it on a real
//! login with:
//!
//! ```text
//! cargo test -p syrinx-client --test windows_typing -- --ignored --nocapture
//! ```
//!
//! It creates its own window with an edit control rather than driving Notepad.
//! Notepad on Windows 11 no longer exposes a plain EDIT control, so reading the
//! text back out of it is unreliable -- and a test that cannot read back what it
//! typed proves nothing. Owning the window means the assertion is exact.
#![cfg(windows)]

use windows::Win32::Foundation::HWND;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, DispatchMessageW, GetWindowTextW, MSG, PM_REMOVE, PeekMessageW,
    SW_SHOW, SetForegroundWindow, ShowWindow, TranslateMessage, WINDOW_EX_STYLE, WS_CHILD,
    WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};
// SetFocus lives with the keyboard APIs rather than the window ones.
use windows::Win32::UI::Input::KeyboardAndMouse::SetFocus;
use windows::core::{PCWSTR, w};

/// Run the message loop briefly so queued input is delivered.
fn pump(ms: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(ms);
    while std::time::Instant::now() < deadline {
        let mut msg = MSG::default();
        unsafe {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn read_text(hwnd: HWND) -> String {
    let mut buf = [0u16; 1024];
    let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
    String::from_utf16_lossy(&buf[..n as usize])
}

#[test]
#[ignore = "needs an interactive desktop"]
fn sendinput_types_into_a_focused_window() {
    // "Static" and "Edit" are pre-registered classes, so no window class of our
    // own has to be registered.
    let hinstance = unsafe { GetModuleHandleW(PCWSTR::null()) }.expect("module handle");

    let parent = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("Static"),
            w!("syrinx typing test"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            100,
            100,
            420,
            140,
            None,
            None,
            Some(hinstance.into()),
            None,
        )
    }
    .expect("creating the test window");

    let edit = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("Edit"),
            PCWSTR::null(),
            WS_CHILD | WS_VISIBLE,
            10,
            10,
            380,
            60,
            Some(parent),
            None,
            Some(hinstance.into()),
            None,
        )
    }
    .expect("creating the edit control");

    unsafe {
        let _ = ShowWindow(parent, SW_SHOW);
        let _ = SetForegroundWindow(parent);
        let _ = SetFocus(Some(edit));
    }
    pump(300);

    // Deliberately mixed: ASCII, a non-ASCII character that would come out
    // wrong if a virtual key code were used instead of a Unicode scan code,
    // and an astral character that needs a surrogate pair.
    let text = "hello syrinx \u{e9} \u{1F600}";
    syrinx_client::inject::type_text(text, syrinx_client::inject::Method::SendInput)
        .expect("SendInput should deliver every event");
    pump(600);

    let got = read_text(edit);
    unsafe {
        let _ = DestroyWindow(parent);
    }

    assert_eq!(
        got, text,
        "what arrived in the window differs from what was typed"
    );
}
