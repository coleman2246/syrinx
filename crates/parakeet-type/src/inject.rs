//! Typing text at the cursor on Wayland via `wtype`.
//!
//! Injection is **append-only**. Live mode never emits `transcript.revise`, so
//! there is no retraction path to implement -- which is the entire point of that
//! design decision. Typing into arbitrary applications means a delete could
//! destroy whatever the user typed in the meantime.

use anyhow::{Context, Result, bail};
use std::process::Command;
use tracing::debug;

/// Type `text` at the cursor.
///
/// Uses `--` so text beginning with a dash is not parsed as a flag; a
/// transcript starting with "-" would otherwise make wtype fail or, worse, do
/// something unintended.
pub fn type_text(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    debug!("typing {text:?}");
    let status = Command::new("wtype")
        .arg("--")
        .arg(text)
        .status()
        .context("running wtype (is it installed?)")?;
    if !status.success() {
        bail!("wtype exited with {status}");
    }
    Ok(())
}

/// Check `wtype` is present before a session starts, so the failure surfaces up
/// front rather than after the user has already spoken a sentence.
///
/// Only spawnability is checked. wtype's exit status with no arguments is not a
/// health signal, so the meaningful question is simply whether it exists.
pub fn preflight() -> Result<()> {
    match Command::new("wtype").arg("").status() {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("wtype not found on PATH; it is required to type on Wayland")
        }
        Err(e) => Err(e).context("checking for wtype"),
    }
}
