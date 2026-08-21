//! Typing text at the cursor.
//!
//! Append-only by design. Live sessions never emit revisions, so there is no
//! retraction path here -- which is the whole point of that constraint. Typing
//! goes into whatever window has focus, where deleting characters could destroy
//! whatever the user typed in the meantime.

use anyhow::{Context, Result, bail};
use std::process::Command;
use tracing::debug;

/// Type `text` at the cursor.
///
/// `--` guards against a transcript beginning with a dash being parsed as a
/// flag.
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

/// Check the typing tool exists before a session starts, so a missing
/// dependency surfaces up front rather than after the user has spoken.
pub fn preflight() -> Result<()> {
    match Command::new("wtype").arg("").status() {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("wtype not found on PATH; it is required to type on Wayland")
        }
        Err(e) => Err(e).context("checking for wtype"),
    }
}
