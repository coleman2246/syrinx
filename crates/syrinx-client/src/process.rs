//! Child processes started and then left to themselves.
//!
//! The viewer, the overlay and the daemon are all launched and not spoken to
//! again: nothing keeps the handle, and nothing is waiting for the moment they
//! exit. Somebody still has to read the exit status, which is what this is for.

/// Wait for a child on a thread of its own, so it cannot be left defunct.
///
/// Dropping a `std::process::Child` deliberately does not wait -- the standard
/// library will not hide a blocking call inside a drop -- so a child nobody
/// waits for stays in the process table from the moment it exits until its
/// parent does. Where the parent is long-lived that accrues one entry per
/// child it has ever started; fourteen defunct viewers under one daemon is
/// what sent anyone looking.
///
/// The wait cannot go on the caller. These children exit when a person closes
/// a window, which may be hours, or not at all until they are killed -- and a
/// killed process in uninterruptible IO does not exit on request either. So it
/// gets a thread whose only job is to sit in the wait: one stack for as long
/// as the child lives, which is the price of not blocking whatever the caller
/// was doing.
///
/// A thread that cannot be created is reported rather than fatal.
/// `thread::spawn` panics when the OS refuses one, and this is called from the
/// daemon's loop: one leaked zombie is a far smaller thing than a daemon that
/// died because it could not make a thread.
pub fn reap_in_background(mut child: std::process::Child) {
    let started = std::thread::Builder::new()
        .name("reap-child".into())
        .spawn(move || {
            let _ = child.wait();
        });
    if let Err(e) = started {
        tracing::warn!("no thread to wait for a child process, leaving it to the parent: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether the kernel still holds a record of this process.
    ///
    /// Signal 0 checks permission without sending anything. A zombie answers
    /// yes -- it stays in the table until someone reads its exit status -- so
    /// this asks the question the helper exists to answer. Waiting here
    /// instead would consume the status and make the test pass whatever the
    /// helper did.
    #[cfg(unix)]
    fn still_in_the_process_table(pid: u32) -> bool {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
    }

    #[test]
    #[cfg(unix)]
    fn a_child_handed_over_is_waited_for_once_it_exits() {
        let child = std::process::Command::new("true")
            .spawn()
            .expect("spawning a stand-in child");
        let pid = child.id();

        reap_in_background(child);

        // Two seconds is far longer than a thread needs to be scheduled, so
        // exhausting it means nobody is waiting at all.
        let gone = (0..200).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            !still_in_the_process_table(pid)
        });
        assert!(gone, "{pid} exited and was never waited for");
    }
}
