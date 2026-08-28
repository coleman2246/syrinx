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
    /// Whatever is right for the platform this is running on.
    ///
    /// The default, and the reason a single config file works on every
    /// machine: the *setting* is identical everywhere, and only its
    /// resolution differs. Naming a concrete method as the default would
    /// have meant a config written on one platform being wrong on another.
    #[default]
    Auto,
    /// `wtype`, the Wayland virtual-keyboard protocol.
    ///
    /// The default, and correct for most native Wayland applications. It fails
    /// in Electron and Chromium apps such as Teams, Discord and VS Code: each
    /// call creates and destroys a virtual keyboard, and Chromium re-evaluates
    /// focus when input devices appear, so the text field loses focus and the
    /// keystrokes are interpreted as global shortcuts. In Teams that shows up
    /// as the chat list jumping about instead of a message being typed.
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
    /// The Win32 `SendInput` API.
    ///
    /// The Windows equivalent, and the only one there. Text is sent as
    /// UTF-16 key events with `KEYEVENTF_UNICODE`, which bypasses the
    /// keyboard layout entirely -- the character arrives as written whatever
    /// the user's layout is.
    SendInput,
}

/// What [`Method::Auto`] means here, worked out once per process.
///
/// Probed rather than assumed, and cached because this is asked once per
/// transcript fragment -- several times a second -- and the answer cannot
/// change without restarting the daemon that owns the session.
fn auto() -> Method {
    static CHOICE: std::sync::OnceLock<Method> = std::sync::OnceLock::new();
    *CHOICE.get_or_init(|| {
        if cfg!(windows) {
            return Method::SendInput;
        }
        // Running ydotoold is a deliberate act -- it is a daemon someone had
        // to install, enable and give access to /dev/uinput. Taking that as
        // consent means `auto` is right in Electron apps too, where wtype
        // silently loses focus. Without it, wtype needs no setup at all.
        if ydotoold_running() {
            return Method::Ydotool;
        }
        Method::Wtype
    })
}

impl Method {
    /// The concrete method to use on this machine.
    ///
    /// Only [`Method::Auto`] changes; anything chosen explicitly is honoured
    /// as written, and fails loudly in preflight if it cannot work here.
    pub fn resolve(self) -> Method {
        match self {
            Method::Auto => auto(),
            other => other,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Method::Wtype => "wtype (Wayland)",
            Method::Ydotool => "ydotool (uinput)",
            Method::Paste => "clipboard paste",
            Method::SendInput => "SendInput (Win32)",
            Method::Auto => "automatic",
        }
    }

    /// The value as written in the config file.
    ///
    /// Kept in step with serde by a test rather than by hope: these strings
    /// go into a generated config, and one that does not parse is worse than
    /// no generated config at all.
    pub fn name(self) -> &'static str {
        match self {
            Method::Wtype => "wtype",
            Method::Ydotool => "ydotool",
            Method::Paste => "paste",
            Method::SendInput => "sendinput",
            Method::Auto => "auto",
        }
    }

    /// One line for the generated config, explaining when to pick this.
    pub fn summary(self) -> &'static str {
        match self {
            Method::Wtype => "[Linux] Wayland virtual keyboard. Fails in Electron apps",
            Method::Ydotool => "[Linux] kernel uinput. Works in Electron, needs ydotoold",
            Method::Paste => "[Linux] clipboard then Ctrl+V. Wrong for terminals",
            Method::SendInput => "[Windows] layout-independent Unicode key events",
            Method::Auto => "the right one for this machine. Start here",
        }
    }

    /// Whether this method can work on the platform now running.
    pub fn supported_here(self) -> bool {
        match self {
            // Auto resolves to something that works, by construction.
            Method::Auto => true,
            Method::Wtype | Method::Ydotool | Method::Paste => cfg!(target_os = "linux"),
            Method::SendInput => cfg!(windows),
        }
    }

    /// The command this method needs on PATH, if it needs one.
    ///
    /// `SendInput` is an API call rather than a process, so it has none.
    fn binary(self) -> Option<&'static str> {
        match self {
            Method::Wtype => Some("wtype"),
            Method::Ydotool => Some("ydotool"),
            Method::Paste => Some("wl-copy"),
            Method::SendInput | Method::Auto => None,
        }
    }

    pub const ALL: [Method; 5] = [
        Method::Auto,
        Method::Wtype,
        Method::Ydotool,
        Method::Paste,
        Method::SendInput,
    ];
}

/// Type `text` at the cursor.
pub fn type_text(text: &str, method: Method) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let method = method.resolve();
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
        Method::SendInput => send_input(text),
        // resolve() never returns Auto.
        Method::Auto => unreachable!("Auto resolves before dispatch"),
    }
}

/// Type via the Win32 `SendInput` API.
#[cfg(windows)]
fn send_input(text: &str) -> Result<()> {
    crate::windows_input::type_text(text)
}

#[cfg(not(windows))]
fn send_input(_text: &str) -> Result<()> {
    bail!("the `sendinput` method is Windows-only")
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

    copy_to_clipboard(Command::new("wl-copy"), text.as_bytes())?;

    run(
        Command::new("wtype").args(["-M", "ctrl", "-k", "v", "-m", "ctrl"]),
        "wtype (paste)",
    )?;

    // Restore after the paste has been read. Without the pause the application
    // can still be reading the selection when it is overwritten, and pastes the
    // old clipboard instead.
    if let Some(prev) = previous {
        std::thread::sleep(std::time::Duration::from_millis(120));
        let _ = copy_to_clipboard(Command::new("wl-copy"), &prev);
    }
    Ok(())
}

/// Feed `bytes` to a clipboard tool on its stdin, waiting for it however it
/// goes.
///
/// The wait is unconditional because every path here holds the `Child`: a `?`
/// on the write would drop it, and a dropped handle is never waited for, so a
/// `wl-copy` that cannot reach the compositor -- no `WAYLAND_DISPLAY`, a
/// restarted compositor, a broken wl-clipboard -- would be left defunct. A
/// failure to type is logged and the session goes on (see `session.rs`), so
/// that is one defunct process per fragment spoken, several a minute, rather
/// than one per session.
///
/// The command is a parameter rather than a name so a test can hand it one
/// that fails; `wl-copy` is the only thing that reaches here in a real run.
fn copy_to_clipboard(mut cmd: Command, bytes: &[u8]) -> Result<()> {
    let what = cmd.get_program().to_string_lossy().into_owned();
    let mut child = cmd
        .stdin(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("running {what}"))?;
    let written = write_to_stdin(&mut child, bytes).with_context(|| format!("writing to {what}"));
    // `wait` closes stdin before it waits, so a tool still reading is told the
    // input has ended rather than left to block against a handle we hold.
    let waited = child.wait().with_context(|| format!("waiting for {what}"));
    written?;
    waited?;
    Ok(())
}

/// Hand `bytes` to a child's stdin.
///
/// Its own function so its `?` returns to a caller that still has the child to
/// wait for, which is the whole point of the split.
fn write_to_stdin(child: &mut std::process::Child, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let stdin = child.stdin.as_mut().context("stdin missing")?;
    Ok(stdin.write_all(bytes)?)
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
    let method = method.resolve();
    if !method.supported_here() {
        let usable: Vec<&str> = Method::ALL
            .iter()
            .filter(|m| m.supported_here())
            .map(|m| m.name())
            .collect();
        bail!(
            "`inject = \"{}\"` does not work on this platform. Use one of: {}",
            method.name(),
            usable.join(", ")
        );
    }

    if let Some(bin) = method.binary() {
        match Command::new(bin).arg("--help").output() {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("{bin} not found on PATH; it is required for {}", method.label())
            }
            Err(e) => return Err(e).with_context(|| format!("checking for {bin}")),
        }
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
    fn the_default_is_the_same_setting_on_every_platform() {
        // The point of Auto: one config file that is correct everywhere.
        // A concrete default would be wrong on the other machine.
        assert_eq!(Method::default(), Method::Auto);
    }

    #[test]
    fn auto_resolves_to_something_that_works_here() {
        let r = Method::Auto.resolve();
        assert_ne!(r, Method::Auto, "resolve must produce a concrete method");
        assert!(r.supported_here(), "{r:?} cannot work on this platform");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn auto_picks_a_linux_method() {
        // Which one depends on whether ydotoold is running, so assert the
        // property rather than the answer: a test that demands wtype fails on
        // a machine set up for Electron apps, and vice versa.
        let r = Method::Auto.resolve();
        assert!(
            matches!(r, Method::Wtype | Method::Ydotool),
            "auto chose {r:?} on Linux"
        );
        assert!(r.supported_here());
    }

    #[test]
    fn auto_is_stable_within_a_process() {
        // It is asked several times a second while dictating; an answer that
        // changed mid-session would type half a sentence by one route and
        // half by another.
        assert_eq!(Method::Auto.resolve(), Method::Auto.resolve());
    }

    #[test]
    #[cfg(windows)]
    fn auto_is_sendinput_on_windows() {
        assert_eq!(Method::Auto.resolve(), Method::SendInput);
    }

    #[test]
    fn an_explicit_choice_is_never_second_guessed() {
        // Someone who wrote `ydotool` gets ydotool, not what we would pick.
        for m in Method::ALL.iter().filter(|m| **m != Method::Auto) {
            assert_eq!(m.resolve(), *m);
        }
    }

    #[test]
    fn every_method_names_a_distinct_binary() {
        let mut b: Vec<&str> = Method::ALL.iter().filter_map(|m| m.binary()).collect();
        b.sort_unstable();
        let n = b.len();
        b.dedup();
        assert_eq!(b.len(), n);
    }

    #[test]
    fn config_names_match_what_serde_accepts() {
        // These strings are written into a generated config file. If one did
        // not round-trip, syrinx would emit a config it cannot itself read.
        for m in Method::ALL {
            let quoted = format!("\"{}\"", m.name());
            assert_eq!(
                serde_json::from_str::<Method>(&quoted).unwrap(),
                m,
                "name() disagrees with serde for {m:?}"
            );
        }
    }

    #[test]
    fn the_default_can_always_be_used() {
        assert!(Method::default().supported_here());
    }

    #[test]
    fn every_platform_has_at_least_one_usable_method() {
        assert!(Method::ALL.iter().any(|m| m.supported_here()));
    }

    #[test]
    fn labels_are_distinct() {
        let mut l: Vec<&str> = Method::ALL.iter().map(|m| m.label()).collect();
        l.sort_unstable();
        let n = l.len();
        l.dedup();
        assert_eq!(l.len(), n);
    }

    /// Whether the kernel still holds a record of this process.
    ///
    /// Signal 0 checks permission without sending anything. A zombie answers
    /// yes -- it stays in the table until someone reads its exit status -- and
    /// that is the distinction the test below is about. It also leaves the
    /// status where it is, which a `waitpid` would not: a test that reaped the
    /// child itself could not tell whether the code under test had.
    #[cfg(unix)]
    fn still_in_the_process_table(pid: i32) -> bool {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
    }

    #[test]
    #[cfg(unix)]
    fn a_clipboard_tool_that_fails_is_waited_for_rather_than_left_defunct() {
        // A stand-in that exits without reading a byte, which is what wl-copy
        // does when it cannot reach a compositor. It reports its own pid
        // because nothing hands one back: the handle never leaves the code
        // under test.
        let pid_file =
            std::env::temp_dir().join(format!("syrinx-copier-pid-{}", std::process::id()));
        let _ = std::fs::remove_file(&pid_file);
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(format!("echo $$ > {}", pid_file.display()));

        // More than a pipe will buffer, so the write cannot quietly land in
        // the buffer of a reader that has already gone: it blocks, the reader
        // exits, and the failure is a broken pipe every time rather than
        // sometimes.
        let err = copy_to_clipboard(cmd, &vec![b'x'; 256 * 1024])
            .expect_err("a tool that exits without reading fails the write");
        assert!(
            format!("{err:#}").contains("writing to sh"),
            "expected the write to be what failed: {err:#}"
        );

        let reported = std::fs::read_to_string(&pid_file)
            .expect("the stand-in never reported its pid; was `sh` runnable?");
        let _ = std::fs::remove_file(&pid_file);
        let pid: i32 = reported.trim().parse().expect("a pid from the stand-in");
        assert!(
            !still_in_the_process_table(pid),
            "the copy failed and {pid} was never waited for, so it is defunct"
        );
    }

    #[test]
    fn typing_nothing_is_a_no_op_for_every_method() {
        // Called for every empty fragment; it must not spawn a process or fail.
        for m in Method::ALL {
            assert!(type_text("", m).is_ok());
        }
    }
}
