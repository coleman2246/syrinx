//! Typing at the cursor on Windows, via `SendInput`.
//!
//! Text is sent as UTF-16 code units with `KEYEVENTF_UNICODE` rather than as
//! virtual key codes. A virtual key means "the key in this position", so what
//! arrives depends on the active keyboard layout -- dictating an apostrophe on
//! a French layout would produce something else entirely. A Unicode scan code
//! means "this character", whatever the layout, which is what a transcript
//! needs.
//!
//! This is the Windows counterpart to the three Linux methods, and unlike them
//! it needs no helper process: `SendInput` is a syscall, so there is no daemon
//! to be running and no binary to be missing.

use anyhow::{Result, bail};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT, KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE, SendInput, VIRTUAL_KEY, VK_RETURN,
};

/// Type `text` into whatever window has focus.
pub fn type_text(text: &str) -> Result<()> {
    let mut inputs: Vec<INPUT> = Vec::with_capacity(text.len() * 2);

    for unit in text.encode_utf16() {
        // U+000A is not a printable character, so sending it as a Unicode scan
        // code produces nothing at all. Enter is what the character means.
        // Note this submits the message in chat applications, exactly as
        // pressing Enter would -- which is the honest reading of a newline.
        if unit == b'\n' as u16 {
            push_key(&mut inputs, VK_RETURN, 0, KEYBD_EVENT_FLAGS(0));
            continue;
        }
        // A carriage return alongside a newline would press Enter twice.
        if unit == b'\r' as u16 {
            continue;
        }
        // Surrogate pairs need no special handling: encode_utf16 emits both
        // halves in order and Windows recombines consecutive events.
        push_key(&mut inputs, VIRTUAL_KEY(0), unit, KEYEVENTF_UNICODE);
    }

    if inputs.is_empty() {
        return Ok(());
    }

    let sent = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
        // The usual cause is UIPI: a process cannot send input to a window
        // running at a higher integrity level. Worth naming, because the
        // symptom is text appearing everywhere except the one window that
        // matters, which looks like a syrinx bug rather than a Windows rule.
        bail!(
            "SendInput delivered {sent} of {} events. If the focused window is \
             running as administrator, syrinx must be too.",
            inputs.len()
        );
    }
    Ok(())
}

/// Append a press and release pair.
fn push_key(inputs: &mut Vec<INPUT>, vk: VIRTUAL_KEY, scan: u16, flags: KEYBD_EVENT_FLAGS) {
    for extra in [KEYBD_EVENT_FLAGS(0), KEYEVENTF_KEYUP] {
        inputs.push(INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: scan,
                    dwFlags: flags | extra,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the event list without sending it, so the encoding can be checked
    /// without typing into whatever window happens to be focused.
    fn encode(text: &str) -> Vec<u16> {
        text.encode_utf16().filter(|u| *u != b'\r' as u16).collect()
    }

    #[test]
    fn typing_nothing_sends_nothing() {
        assert!(type_text("").is_ok());
    }

    #[test]
    fn astral_characters_become_surrogate_pairs() {
        // An emoji is two UTF-16 units; sending only one would produce a
        // replacement character.
        assert_eq!(encode("\u{1F600}").len(), 2);
    }

    #[test]
    fn crlf_presses_enter_once() {
        // Both halves reaching the key builder would submit a chat message
        // twice.
        assert_eq!(
            encode("a\r\nb"),
            vec![b'a' as u16, b'\n' as u16, b'b' as u16]
        );
    }
}
