//! Typing text at the cursor.
//!
//! Append-only by design. Live sessions never emit revisions, so there is no
//! retraction path here -- which is the whole point of that constraint. Typing
//! goes into whatever window has focus, where deleting characters could destroy
//! whatever the user typed in the meantime.
//!
//! Three methods, because no single one works everywhere. See [`Method`].

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::debug;

/// How text is delivered to the focused window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// `wtype`, the Wayland virtual-keyboard protocol.
    ///
    /// The default, and correct for most native Wayland applications. It fails
    /// in Electron and Chromium apps such as Teams, Discord and VS Code: each
    /// call creates and destroys a virtual keyboard, and Chromium re-evaluates
    /// focus when input devices appear, so the text field loses focus and the
    /// keystrokes are interpreted as global shortcuts. In Teams that shows up
    /// as the chat list jumping about instead of a message being typed.
    #[default]
    Wtype,
    /// `ydotool`, writing to `/dev/uinput`.
    ///
    /// A kernel-level virtual device, indistinguishable from a real keyboard,
    /// so applications that ignore or mishandle the virtual-keyboard protocol
    /// still receive it. This is the one to use for Electron apps. Needs the
    /// `ydotoold` daemon running and access to `/dev/uinput`.
    Ydotool,
    /// Copy to the clipboard and send Ctrl+V.
    ///
    /// The most broadly compatible, since pasting is layout-independent and
    /// handles any character an application accepts. It briefly replaces the
    /// clipboard, restoring it afterwards, and terminals need Ctrl+Shift+V so
    /// they are not a good fit.
    Paste,
}

impl Method {
    pub fn label(self) -> &'static str {
        match self {
            Method::Wtype => "wtype (Wayland)",
            Method::Ydotool => "ydotool (uinput)",
            Method::Paste => "clipboard paste",
        }
    }

    /// The command this method needs on PATH.
    fn binary(self) -> &'static str {
        match self {
            Method::Wtype => "wtype",
            Method::Ydotool => "ydotool",
            Method::Paste => "wl-copy",
        }
    }

    pub const ALL: [Method; 3] = [Method::Wtype, Method::Ydotool, Method::Paste];
}

/// Type `text` at the cursor.
pub fn type_text(text: &str, method: Method) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    debug!("typing {text:?} via {}", method.label());
    match method {
        Method::Wtype => run(Command::new("wtype").arg("--").arg(text), "wtype"),
        // --key-delay 0 keeps a long fragment from taking visibly long; the
        // uinput device is persistent so there is no settling to wait for.
        Method::Ydotool => run(
            Command::new("ydotool")
                .arg("type")
                .args(["--key-delay", "0"])
                .arg("--")
                .arg(text),
            "ydotool",
        ),
        Method::Paste => paste(text),
    }
}

/// Copy to the clipboard, paste, and put the clipboard back.
fn paste(text: &str) -> Result<()> {
    // Read what is there so the user's clipboard survives dictation.
    let previous = Command::new("wl-paste")
        .arg("--no-newline")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| o.stdout);

    let mut child = Command::new("wl-copy")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("running wl-copy")?;
    {
        use std::io::Write;
        child
            .stdin
            .as_mut()
            .context("wl-copy stdin missing")?
            .write_all(text.as_bytes())
            .context("writing to wl-copy")?;
    }
    child.wait().context("waiting for wl-copy")?;

    run(
        Command::new("wtype").args(["-M", "ctrl", "-k", "v", "-m", "ctrl"]),
        "wtype (paste)",
    )?;

    // Restore after the paste has been read. Without the pause the application
    // can still be reading the selection when it is overwritten, and pastes the
    // old clipboard instead.
    if let Some(prev) = previous {
        std::thread::sleep(std::time::Duration::from_millis(120));
        if let Ok(mut c) = Command::new("wl-copy")
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            use std::io::Write;
            if let Some(stdin) = c.stdin.as_mut() {
                let _ = stdin.write_all(&prev);
            }
            let _ = c.wait();
        }
    }
    Ok(())
}

fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("running {what} (is it installed?)"))?;
    if !status.success() {
        bail!("{what} exited with {status}");
    }
    Ok(())
}

/// Check the chosen method can work, before a session starts.
///
/// Surfaces a missing tool up front rather than after the user has spoken a
/// sentence into a session that was never going to type anything.
pub fn preflight(method: Method) -> Result<()> {
    let bin = method.binary();
    match Command::new(bin).arg("--help").output() {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("{bin} not found on PATH; it is required for {}", method.label())
        }
        Err(e) => return Err(e).with_context(|| format!("checking for {bin}")),
    }

    // ydotool needs its daemon; without it every call fails at the point where
    // the user is already talking.
    if method == Method::Ydotool && !ydotoold_running() {
        bail!(
            "ydotoold is not running. Start it with `systemctl --user start ydotool` \
             (or run `ydotoold` directly); ydotool cannot type without it."
        );
    }
    Ok(())
}

fn ydotoold_running() -> bool {
    Command::new("pgrep")
        .args(["-x", "ydotoold"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methods_round_trip_through_config() {
        for m in Method::ALL {
            let s = serde_json::to_string(&m).unwrap();
            assert_eq!(serde_json::from_str::<Method>(&s).unwrap(), m);
        }
    }

    #[test]
    fn wtype_is_the_default() {
        // Changing the default would alter behaviour for everyone whose setup
        // already works.
        assert_eq!(Method::default(), Method::Wtype);
    }

    #[test]
    fn every_method_names_a_distinct_binary() {
        let mut b: Vec<&str> = Method::ALL.iter().map(|m| m.binary()).collect();
        b.sort_unstable();
        let n = b.len();
        b.dedup();
        assert_eq!(b.len(), n);
    }

    #[test]
    fn labels_are_distinct() {
        let mut l: Vec<&str> = Method::ALL.iter().map(|m| m.label()).collect();
        l.sort_unstable();
        let n = l.len();
        l.dedup();
        assert_eq!(l.len(), n);
    }

    #[test]
    fn typing_nothing_is_a_no_op_for_every_method() {
        // Called for every empty fragment; it must not spawn a process or fail.
        for m in Method::ALL {
            assert!(type_text("", m).is_ok());
        }
    }
}
