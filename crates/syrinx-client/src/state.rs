//! Run-state tracking via a PID file.
//!
//! Deliberately **not** `pgrep -f`. A pattern like `pgrep -f syrinx`
//! matches the command line of the shell that invoked it, so any script or
//! terminal whose command line mentions the pattern reads as "already running".
//! A toggle built on that always takes the stop branch and never starts. This
//! bit the earlier nerd-dictation setup, and then bit again while testing the
//! server, where `pkill -f target/debug/syrinx-server` killed the invoking
//! shell.
//!
//! A PID file validated with `kill(pid, 0)` has neither problem, and correctly
//! reports "not running" for a stale file left by a crash.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Where the PID file lives. `XDG_RUNTIME_DIR` is preferred because it is
/// cleared on logout, so a stale file cannot survive a reboot.
pub fn default_pid_path() -> PathBuf {
    #[cfg(windows)]
    {
        // No XDG_RUNTIME_DIR equivalent; the per-user temp directory is the
        // closest thing, and is likewise not shared between users.
        std::env::temp_dir().join("syrinx.pid")
    }
    #[cfg(not(windows))]
    {
        let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(dir).join("syrinx.pid")
    }
}

/// Read the PID file and confirm the process is alive.
///
/// Returns `None` when the file is missing, unparseable, or names a process
/// that no longer exists.
pub fn running_pid(path: &Path) -> Option<i32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let pid: i32 = raw.trim().parse().ok()?;
    is_alive(pid).then_some(pid)
}

/// Whether a process with this id exists.
#[cfg(unix)]
fn is_alive(pid: i32) -> bool {
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    // Signal 0 performs error checking without sending anything.
    kill(Pid::from_raw(pid), None).is_ok()
}

/// Whether a process with this id exists.
///
/// `OpenProcess` succeeding is not enough on its own: a handle stays valid for
/// a process that has exited but not yet been reaped, so the exit code has to
/// be checked too. Without that, a crashed session would read as running and a
/// toggle would never start again -- the same trap the PID file exists to
/// avoid.
#[cfg(windows)]
fn is_alive(pid: i32) -> bool {
    use windows::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let Ok(pid) = u32::try_from(pid) else {
        return false;
    };
    unsafe {
        let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return false;
        };
        let mut code = 0u32;
        let alive = GetExitCodeProcess(handle, &mut code).is_ok() && code == STILL_ACTIVE.0 as u32;
        let _ = CloseHandle(handle);
        alive
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

/// Where a stop request is left for a running instance to find.
///
/// Windows only. See [`signal_stop`].
pub fn stop_request_path(pid_path: &Path) -> PathBuf {
    pid_path.with_extension("stop")
}

/// Ask a running instance to stop. It finishes the current utterance and exits.
///
/// On Unix this is SIGTERM. Windows has no equivalent -- `TerminateProcess` is
/// not one, since it gives the session no chance to flush the last utterance or
/// remove its PID file -- so a request file is left beside the PID file and the
/// running instance watches for it. Slower to notice, but it stops cleanly,
/// which matters more for something whose whole job is not losing your words.
pub fn signal_stop(pid: i32, pid_path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        let _ = pid_path;
        kill(Pid::from_raw(pid), Signal::SIGTERM)
            .context("sending SIGTERM to the running instance")
    }
    #[cfg(windows)]
    {
        let _ = pid;
        let p = stop_request_path(pid_path);
        std::fs::write(&p, "stop")
            .with_context(|| format!("writing a stop request to {}", p.display()))
    }
}

/// Whether someone has asked this instance to stop. Clears the request.
///
/// Windows only; on Unix the signal handler does this job.
pub fn take_stop_request(pid_path: &Path) -> bool {
    let p = stop_request_path(pid_path);
    if p.exists() {
        let _ = std::fs::remove_file(&p);
        return true;
    }
    false
}

/// Nudge waybar to re-run its custom module immediately.
///
/// Polling would leave the indicator up to a second stale on both edges, which
/// is very visible when the whole point is knowing whether the mic is live.
pub fn refresh_waybar(signal: u8) {
    if !cfg!(target_os = "linux") {
        return;
    }
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
        p.push(format!("syrinx-test-{name}-{}", std::process::id()));
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
        assert!(running_pid(&p).is_none());
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
