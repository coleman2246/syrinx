//! Run-state tracking via a PID file.
//!
//! Deliberately **not** `pgrep -f`. A pattern like `pgrep -f parakeet-type`
//! matches the command line of the shell that invoked it, so any script or
//! terminal whose command line mentions the pattern reads as "already running".
//! A toggle built on that always takes the stop branch and never starts. This
//! bit the earlier nerd-dictation setup, and then bit again while testing the
//! server, where `pkill -f target/debug/parakeet-server` killed the invoking
//! shell.
//!
//! A PID file validated with `kill(pid, 0)` has neither problem, and correctly
//! reports "not running" for a stale file left by a crash.

use anyhow::{Context, Result};
use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use std::path::{Path, PathBuf};

/// Where the PID file lives. `XDG_RUNTIME_DIR` is preferred because it is
/// cleared on logout, so a stale file cannot survive a reboot.
pub fn default_pid_path() -> PathBuf {
    let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(dir).join("parakeet.pid")
}

/// Read the PID file and confirm the process is alive.
///
/// Returns `None` when the file is missing, unparseable, or names a process
/// that no longer exists.
pub fn running_pid(path: &Path) -> Option<i32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let pid: i32 = raw.trim().parse().ok()?;
    // Signal 0 performs error checking without sending anything.
    match kill(Pid::from_raw(pid), None) {
        Ok(()) => Some(pid),
        Err(_) => None,
    }
}

pub fn write_pid(path: &Path, pid: i32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, pid.to_string())
        .with_context(|| format!("writing pid file {}", path.display()))
}

pub fn clear_pid(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Ask a running instance to stop. It finishes the current utterance and exits.
pub fn signal_stop(pid: i32) -> Result<()> {
    kill(Pid::from_raw(pid), Signal::SIGTERM).context("sending SIGTERM to the running instance")
}

/// Nudge waybar to re-run its custom module immediately.
///
/// Polling would leave the indicator up to a second stale on both edges, which
/// is very visible when the whole point is knowing whether the mic is live.
pub fn refresh_waybar(signal: u8) {
    let _ = std::process::Command::new("pkill")
        .arg(format!("-RTMIN+{signal}"))
        // `-x` matches the process name exactly, never the invoking command
        // line -- the same self-match trap as pgrep -f.
        .arg("-x")
        .arg("waybar")
        .status();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("parakeet-test-{name}-{}", std::process::id()));
        p
    }

    #[test]
    fn missing_pid_file_reads_as_not_running() {
        let p = tmp("missing");
        let _ = std::fs::remove_file(&p);
        assert_eq!(running_pid(&p), None);
    }

    #[test]
    fn our_own_pid_reads_as_running() {
        let p = tmp("live");
        let me = std::process::id() as i32;
        write_pid(&p, me).unwrap();
        assert_eq!(running_pid(&p), Some(me));
        clear_pid(&p);
    }

    #[test]
    fn stale_pid_file_reads_as_not_running() {
        // A crash leaves the file behind. Treating that as "running" is what
        // makes a toggle permanently refuse to start.
        let p = tmp("stale");
        // PID 0 is never a normal process; kill(0, 0) targets a process group
        // so use an implausible high PID instead.
        write_pid(&p, 4_194_303).unwrap();
        assert_eq!(running_pid(&p), None);
        clear_pid(&p);
    }

    #[test]
    fn garbage_pid_file_reads_as_not_running() {
        let p = tmp("garbage");
        std::fs::write(&p, "not-a-number").unwrap();
        assert_eq!(running_pid(&p), None);
        clear_pid(&p);
    }

    #[test]
    fn empty_pid_file_reads_as_not_running() {
        let p = tmp("empty");
        std::fs::write(&p, "").unwrap();
        assert_eq!(running_pid(&p), None);
        clear_pid(&p);
    }

    #[test]
    fn clear_then_read_is_not_running() {
        let p = tmp("cleared");
        write_pid(&p, std::process::id() as i32).unwrap();
        clear_pid(&p);
        assert_eq!(running_pid(&p), None);
    }
}
