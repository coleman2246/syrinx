//! Writing transcripts to disk.
//!
//! Lives in the shared library rather than either front-end, so `syrinx save`
//! and the GUI's Save button produce byte-identical files. A CLI that wrote
//! subtly different output from the GUI would be a trap for anyone scripting
//! around it.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Default directory for saved transcripts.
pub fn default_dir() -> PathBuf {
    // XDG_DOCUMENTS_DIR if the user has set one, else ~/Documents, else ~.
    if let Ok(d) = std::env::var("XDG_DOCUMENTS_DIR") {
        return PathBuf::from(d).join("syrinx");
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
    let docs = home.join("Documents");
    if docs.is_dir() {
        docs.join("syrinx")
    } else {
        home.join("syrinx")
    }
}

/// Build a filename from a timestamp.
///
/// Sortable by name, and safe on every filesystem: no colons, which Windows
/// rejects and which make shell quoting awkward everywhere else.
pub fn filename_for(stamp: &str) -> String {
    format!("transcript-{stamp}.txt")
}

/// Local time as `YYYY-MM-DD_HH-MM-SS`.
///
/// Shelling out to `date` avoids a chrono dependency for one format string,
/// and falls back to a UTC epoch if that fails so saving never breaks over a
/// timestamp.
pub fn timestamp() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%d_%H-%M-%S")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("epoch-{secs}")
        })
}

/// Write `transcript` to `path`, creating parent directories.
///
/// Refuses to write an empty transcript: a zero-byte file named as if it held a
/// recording is worse than an error, because it looks like the recording failed
/// silently rather than never having happened.
pub fn write(path: &Path, transcript: &str) -> Result<()> {
    let body = transcript.trim();
    if body.is_empty() {
        anyhow::bail!("nothing to save: the transcript is empty");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    // Trailing newline: this is a text file, and tools that read it line-wise
    // expect one.
    std::fs::write(path, format!("{body}\n"))
        .with_context(|| format!("writing {}", path.display()))
}

/// Save to the default directory under a timestamped name, returning the path.
pub fn save_default(transcript: &str) -> Result<PathBuf> {
    let path = default_dir().join(filename_for(&timestamp()));
    write(&path, transcript)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("syrinx-save-{name}-{}", std::process::id()))
    }

    #[test]
    fn writes_the_transcript_with_a_trailing_newline() {
        let p = tmp("basic").join("t.txt");
        write(&p, "hello world").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello world\n");
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn creates_missing_parent_directories() {
        let p = tmp("nested").join("a/b/c/t.txt");
        write(&p, "text").unwrap();
        assert!(p.exists());
        let _ = std::fs::remove_dir_all(tmp("nested"));
    }

    #[test]
    fn refuses_to_write_an_empty_transcript() {
        // A zero-byte file looks like a failed recording rather than one that
        // never happened.
        let p = tmp("empty").join("t.txt");
        assert!(write(&p, "").is_err());
        assert!(write(&p, "   \n\t ").is_err());
        assert!(!p.exists(), "must not create a file it refused to write");
    }

    #[test]
    fn surrounding_whitespace_is_trimmed() {
        let p = tmp("trim").join("t.txt");
        write(&p, "  spoken words  \n\n").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "spoken words\n");
        let _ = std::fs::remove_dir_all(tmp("trim"));
    }

    #[test]
    fn filenames_sort_chronologically_and_avoid_colons() {
        // Colons are illegal on Windows and awkward to quote in a shell.
        let a = filename_for("2026-08-21_09-00-00");
        let b = filename_for("2026-08-21_10-00-00");
        assert!(a < b, "names must sort by time");
        assert!(!a.contains(':'));
        assert!(a.ends_with(".txt"));
    }

    #[test]
    fn a_timestamp_is_always_produced() {
        // Saving must never fail because a clock or a subprocess misbehaved.
        assert!(!timestamp().is_empty());
    }
}
