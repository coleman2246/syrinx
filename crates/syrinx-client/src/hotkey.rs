//! Parsing a hotkey written in a config file.
//!
//! One spelling everywhere -- `hotkey = "ctrl+alt+d"` means the same thing on
//! every platform -- even though what can be done with it does not.
//!
//! **A global hotkey is not portable, and cannot be made portable.** Windows and
//! X11 let any process claim a key combination. Wayland deliberately does not:
//! the compositor owns input, and a client that could grab keys globally could
//! keylog every other client. Under Sway, GNOME or KDE the binding belongs in
//! the compositor's own config, running `syrinx toggle`. The daemon says so
//! rather than failing quietly, because a hotkey that does nothing and reports
//! nothing is worse than one that was never offered.
//!
//! Parsing lives here, away from any platform code, so it can be tested
//! everywhere and so the error messages are the same everywhere.

use anyhow::{Result, bail};

/// Modifier keys, as a set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mods {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// Super, Windows, Command, Meta -- one key, four names.
    pub meta: bool,
}

impl Mods {
    pub fn any(self) -> bool {
        self.ctrl || self.alt || self.shift || self.meta
    }
}

/// A parsed hotkey: some modifiers and exactly one ordinary key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotKey {
    pub mods: Mods,
    /// The key as a W3C `KeyboardEvent.code` name, e.g. `KeyD`, `F9`, `Space`.
    ///
    /// Stored as the canonical name rather than a platform key code so this
    /// type stays free of platform types and can be tested on any machine.
    pub code: String,
    /// The text the user wrote, for error messages and logging.
    pub spelled: String,
}

/// Parse a hotkey such as `ctrl+alt+d`, `super+shift+space` or `f9`.
pub fn parse(spec: &str) -> Result<HotKey> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        bail!("a hotkey cannot be empty");
    }

    let mut mods = Mods::default();
    let mut key: Option<String> = None;

    for part in trimmed.split('+') {
        let p = part.trim().to_ascii_lowercase();
        if p.is_empty() {
            bail!("`{spec}` has an empty part; write it like `ctrl+alt+d`");
        }
        match p.as_str() {
            "ctrl" | "control" => mods.ctrl = true,
            "alt" | "option" => mods.alt = true,
            "shift" => mods.shift = true,
            "super" | "win" | "windows" | "cmd" | "command" | "meta" => mods.meta = true,
            other => {
                if key.is_some() {
                    bail!(
                        "`{spec}` names more than one ordinary key; a hotkey takes \
                         modifiers plus exactly one key"
                    );
                }
                key = Some(code_for(other).ok_or_else(|| {
                    anyhow::anyhow!("`{other}` in `{spec}` is not a key syrinx recognises")
                })?);
            }
        }
    }

    let Some(code) = key else {
        bail!("`{spec}` is only modifiers; it needs a key as well, like `ctrl+alt+d`");
    };

    // A bare letter would fire while typing that letter, which for a dictation
    // toggle means it stops the moment you dictate the letter.
    if !mods.any() && !is_standalone(&code) {
        bail!(
            "`{spec}` has no modifier. Add one (`ctrl+alt+{}`), or use a \
             function key, which is safe on its own",
            spec.trim().to_ascii_lowercase()
        );
    }

    Ok(HotKey {
        mods,
        code,
        spelled: trimmed.to_string(),
    })
}

/// Keys that are safe to bind without a modifier, because nothing types them.
fn is_standalone(code: &str) -> bool {
    code.starts_with('F') && code[1..].chars().all(|c| c.is_ascii_digit())
}

/// Map a written key name to its W3C `KeyboardEvent.code`.
fn code_for(name: &str) -> Option<String> {
    // Single letters and digits are the common case.
    if name.len() == 1 {
        let c = name.chars().next()?;
        if c.is_ascii_alphabetic() {
            return Some(format!("Key{}", c.to_ascii_uppercase()));
        }
        if c.is_ascii_digit() {
            return Some(format!("Digit{c}"));
        }
    }
    // Function keys.
    if let Some(n) = name.strip_prefix('f')
        && let Ok(n) = n.parse::<u8>()
        && (1..=24).contains(&n)
    {
        return Some(format!("F{n}"));
    }
    Some(
        match name {
            "space" => "Space",
            "enter" | "return" => "Enter",
            "tab" => "Tab",
            "esc" | "escape" => "Escape",
            "backspace" => "Backspace",
            "delete" | "del" => "Delete",
            "insert" | "ins" => "Insert",
            "home" => "Home",
            "end" => "End",
            "pageup" | "pgup" => "PageUp",
            "pagedown" | "pgdn" => "PageDown",
            "up" => "ArrowUp",
            "down" => "ArrowDown",
            "left" => "ArrowLeft",
            "right" => "ArrowRight",
            "comma" => "Comma",
            "period" | "dot" => "Period",
            "slash" => "Slash",
            "backslash" => "Backslash",
            "semicolon" => "Semicolon",
            "quote" => "Quote",
            "backquote" | "grave" => "Backquote",
            "minus" | "dash" => "Minus",
            "equal" | "equals" => "Equal",
            "leftbracket" => "BracketLeft",
            "rightbracket" => "BracketRight",
            _ => return None,
        }
        .to_string(),
    )
}

/// What a config file says about hotkeys, on every platform.
///
/// The same text everywhere. A note that appeared only on the machine that
/// generated the file would leave the other machine's user wondering why an
/// identical setting behaved differently.
pub const PORTABILITY_NOTE: &str = "\
Global hotkey to start and stop dictation, e.g. \"ctrl+alt+d\".
Modifiers: ctrl, alt, shift, super. Function keys work on their own.
Unset by default, because this claims the key for the whole desktop.

Registered by syrinx on Windows and on Linux under X11. Wayland does not let
an application claim a global hotkey -- the compositor owns input -- so
bind it in the compositor there instead, e.g. for Sway:
    bindsym $mod+n exec syrinx toggle";

/// Whether this platform lets a process claim a global hotkey at all.
///
/// Wayland is the interesting case: the protocol has no way for a client to
/// grab a key globally, by design.
pub fn supported_here() -> Option<&'static str> {
    if cfg!(windows) {
        return None;
    }
    if cfg!(target_os = "linux") {
        // WAYLAND_DISPLAY is set by the compositor for every client it starts.
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            return Some(
                "Wayland does not let an application claim a global hotkey -- the \
                 compositor owns input. Bind one in your compositor instead, e.g. \
                 for Sway add to ~/.config/sway/config:\n    \
                 bindsym $mod+n exec syrinx toggle",
            );
        }
        return None;
    }
    Some("global hotkeys are not implemented on this platform")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_combination_parses() {
        let h = parse("ctrl+alt+d").unwrap();
        assert!(h.mods.ctrl && h.mods.alt);
        assert!(!h.mods.shift && !h.mods.meta);
        assert_eq!(h.code, "KeyD");
    }

    #[test]
    fn spelling_and_spacing_are_forgiving() {
        // A config file is written by hand; case and spaces should not matter.
        for s in ["CTRL+ALT+D", "Ctrl + Alt + D", "  ctrl+alt+d  "] {
            assert_eq!(parse(s).unwrap().code, "KeyD", "failed on {s:?}");
        }
    }

    #[test]
    fn every_name_for_the_super_key_works() {
        // Users write whatever their keyboard says, and all four appear on real
        // keyboards.
        for s in ["super+n", "win+n", "cmd+n", "meta+n"] {
            assert!(parse(s).unwrap().mods.meta, "failed on {s:?}");
        }
    }

    #[test]
    fn function_keys_need_no_modifier() {
        // Nothing types F9, so it is safe alone.
        let h = parse("f9").unwrap();
        assert_eq!(h.code, "F9");
        assert!(!h.mods.any());
    }

    #[test]
    fn a_bare_letter_is_refused() {
        // Binding `d` alone means dictation stops the moment you dictate a `d`.
        let e = parse("d").unwrap_err().to_string();
        assert!(e.contains("no modifier"), "got: {e}");
    }

    #[test]
    fn modifiers_alone_are_refused() {
        let e = parse("ctrl+alt").unwrap_err().to_string();
        assert!(e.contains("needs a key"), "got: {e}");
    }

    #[test]
    fn two_ordinary_keys_are_refused() {
        let e = parse("ctrl+a+b").unwrap_err().to_string();
        assert!(e.contains("more than one"), "got: {e}");
    }

    #[test]
    fn an_unknown_key_names_itself() {
        // The message has to say which part was wrong, or the user is left
        // guessing at their own config.
        let e = parse("ctrl+alt+wibble").unwrap_err().to_string();
        assert!(e.contains("wibble"), "got: {e}");
    }

    #[test]
    fn empty_and_malformed_specs_are_refused() {
        assert!(parse("").is_err());
        assert!(parse("   ").is_err());
        assert!(parse("ctrl+").is_err());
        assert!(parse("+d").is_err());
    }

    #[test]
    fn named_keys_map_to_w3c_codes() {
        assert_eq!(parse("ctrl+space").unwrap().code, "Space");
        assert_eq!(parse("ctrl+enter").unwrap().code, "Enter");
        assert_eq!(parse("ctrl+pgup").unwrap().code, "PageUp");
        assert_eq!(parse("ctrl+7").unwrap().code, "Digit7");
    }

    #[test]
    fn the_original_spelling_is_kept_for_messages() {
        // Errors and logs should echo what the user wrote, not a normalised
        // form they would not recognise.
        assert_eq!(parse("Ctrl+Alt+D").unwrap().spelled, "Ctrl+Alt+D");
    }

    #[test]
    fn f24_is_the_last_function_key() {
        assert_eq!(parse("f24").unwrap().code, "F24");
        assert!(parse("f25").is_err());
    }
}
