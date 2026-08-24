//! Appending a transcript to a file as it arrives.
//!
//! Saving at the end is fine until something goes wrong at minute forty of a
//! meeting. This writes each fragment the moment the server commits it, so
//! whatever has been said is already on disk -- a crash, a power cut or a
//! closed laptop costs the last sentence rather than the whole session.
//!
//! It also makes stopping and resuming natural: the file is opened for append,
//! so a second session continues the first rather than replacing it.
//!
//! **Only committed text is written.** The server may one day send provisional
//! text that a later revision retracts, and a file that has been flushed cannot
//! take anything back. Writing commits only means the file lags the screen
//! slightly and is never wrong, which is the right way round for the copy you
//! keep.

use crate::save::{Format, stamp};
use crate::session::Segment;
use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// An open transcript file being appended to.
pub struct StreamWriter {
    file: File,
    format: Format,
    path: PathBuf,
    /// Whether the next write needs a newline in front of it, because the file
    /// already has content that did not end in one.
    needs_newline: bool,
    /// When the last fragment arrived, and what produced it, so a stamped line
    /// can be continued rather than restarted for every fragment.
    last: Option<(f64, Option<String>)>,
}

/// How long a silence has to be before a stamped line is considered finished.
///
/// The model emits a fragment roughly every 560 ms while speech continues, so
/// anything much longer than that is a real pause rather than a chunk
/// boundary. Without this, every fragment started its own line and a stamped
/// transcript broke mid-word: "brown fox j" then "umps over the".
const NEW_LINE_AFTER_SILENCE: f64 = 1.5;

impl StreamWriter {
    /// Open `path` for appending, creating it if it does not exist.
    pub fn open(path: &Path, format: Format) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {} to append", path.display()))?;

        // Resuming into prose that ended mid-sentence would otherwise join two
        // sessions into one run-on word.
        let existing = file.metadata().map(|m| m.len()).unwrap_or(0);
        let needs_newline = existing > 0 && !ends_with_newline(path);

        Ok(Self {
            file,
            format,
            path: path.to_path_buf(),
            needs_newline,
            last: None,
        })
    }

    /// Append one committed fragment.
    pub fn append(&mut self, seg: &Segment) -> Result<()> {
        if seg.text.trim().is_empty() {
            return Ok(());
        }
        let mut out = String::new();
        if self.needs_newline {
            out.push('\n');
            self.needs_newline = false;
        }

        match self.format {
            // Prose as it arrives, spacing and all, so the file reads as the
            // same text the screen shows.
            Format::Plain => out.push_str(&seg.text),
            _ => {
                if self.continues_line(seg) {
                    // Mid-utterance: keep the words on the line already open,
                    // with whatever spacing the model gave them.
                    out.push_str(&seg.text);
                } else {
                    if self.last.is_some() {
                        out.push('\n');
                    }
                    out.push_str(&self.line_prefix(seg));
                    out.push_str(seg.text.trim_start());
                }
            }
        }
        self.last = Some((seg.at, seg.source.clone()));

        // `File` is unbuffered, so this reaches the operating system before the
        // call returns: a process that dies afterwards loses nothing. Surviving
        // a power cut as well would need an fsync per fragment, which is a real
        // cost for a rare failure, so it is deliberately not done.
        self.file
            .write_all(out.as_bytes())
            .with_context(|| format!("appending to {}", self.path.display()))?;
        Ok(())
    }

    /// Whether this fragment belongs on the line already open.
    fn continues_line(&self, seg: &Segment) -> bool {
        let Some((last_at, last_source)) = &self.last else {
            return false;
        };
        // A different speaker always starts a new line, however close in time:
        // the label at the start of the line would otherwise be wrong for half
        // of it.
        if matches!(self.format, Format::Labelled) && *last_source != seg.source {
            return false;
        }
        seg.at - last_at < NEW_LINE_AFTER_SILENCE
    }

    /// The stamp, and source where relevant, that opens a line.
    fn line_prefix(&self, seg: &Segment) -> String {
        match (self.format, &seg.source) {
            (Format::Labelled, Some(src)) => format!("{} [{}] ", stamp(seg.at), src),
            _ => format!("{} ", stamp(seg.at)),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Whether a file's last byte is a newline.
fn ends_with_newline(path: &Path) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|b| b.last().copied())
        .is_some_and(|b| b == b'\n')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "syrinx-stream-{}-{tag}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn seg(at: f64, text: &str) -> Segment {
        Segment {
            at,
            text: text.into(),
            source: None,
            speaker: None,
        }
    }

    #[test]
    fn text_is_on_disk_before_the_session_ends() {
        // The whole point: a crash after this line must not lose the words.
        let p = scratch("during");
        let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
        w.append(&seg(0.0, "hello ")).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello ");
        w.append(&seg(1.0, "world")).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello world");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_second_session_continues_the_file() {
        // Stop and resume: the earlier text must still be there.
        let p = scratch("resume");
        let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
        w.append(&seg(0.0, "first session")).unwrap();
        drop(w);

        let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
        w.append(&seg(0.0, "second session")).unwrap();
        drop(w);

        let out = std::fs::read_to_string(&p).unwrap();
        assert!(out.starts_with("first session"), "got {out:?}");
        assert!(out.contains("second session"), "got {out:?}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn resuming_does_not_run_two_sessions_into_one_word() {
        // "first sessionsecond" would be the bug.
        let p = scratch("joins");
        let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
        w.append(&seg(0.0, "alpha")).unwrap();
        drop(w);
        let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
        w.append(&seg(0.0, "beta")).unwrap();
        drop(w);
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "alpha\nbeta");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_pause_starts_a_new_stamped_line() {
        let p = scratch("stamped");
        let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
        w.append(&seg(0.0, "one")).unwrap();
        w.append(&seg(65.0, "two")).unwrap();
        let out = std::fs::read_to_string(&p).unwrap();
        assert_eq!(out, "[00:00] one\n[01:05] two");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn continuous_speech_stays_on_one_line() {
        // The model splits on chunk boundaries, not words. A line per fragment
        // broke words in half: "brown fox j" then "umps over the".
        let p = scratch("continuous");
        let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
        w.append(&seg(0.0, "The quick brown fox j")).unwrap();
        w.append(&seg(0.6, "umps over the")).unwrap();
        w.append(&seg(1.2, " lazy dog")).unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "[00:00] The quick brown fox jumps over the lazy dog"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_new_speaker_starts_a_line_even_without_a_pause() {
        // The label opens the line, so it would otherwise be wrong for the
        // second half of it.
        let p = scratch("speakers");
        let mut w = StreamWriter::open(&p, Format::Labelled).unwrap();
        let mk = |at: f64, text: &str, src: &str| Segment {
            at,
            text: text.into(),
            source: Some(src.into()),
            speaker: None,
        };
        w.append(&mk(0.0, "hello", "Yeti")).unwrap();
        w.append(&mk(0.3, "goodbye", "System")).unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "[00:00] [Yeti] hello\n[00:00] [System] goodbye"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn every_fragment_reaches_disk_immediately_even_mid_line() {
        // Continuing a line must not mean buffering it: a crash halfway
        // through a sentence should still leave that half on disk.
        let p = scratch("midline");
        let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
        w.append(&seg(0.0, "first half")).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "[00:00] first half");
        w.append(&seg(0.5, " second half")).unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "[00:00] first half second half"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn labelled_output_names_the_source() {
        let p = scratch("labelled");
        let mut w = StreamWriter::open(&p, Format::Labelled).unwrap();
        w.append(&Segment {
            at: 0.0,
            text: "hello".into(),
            source: Some("Yeti".into()),
            speaker: None,
        })
        .unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "[00:00] [Yeti] hello");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn silence_writes_nothing() {
        // Empty commits would otherwise litter the file with blank lines.
        let p = scratch("silence");
        let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
        w.append(&seg(0.0, "   ")).unwrap();
        w.append(&seg(1.0, "")).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_first_stamped_line_has_no_leading_blank() {
        // Reading the newline flag that the renderer had just set put a blank
        // line at the top of every new file.
        let p = scratch("firstline");
        let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
        w.append(&seg(0.0, "one")).unwrap();
        let out = std::fs::read_to_string(&p).unwrap();
        assert!(!out.starts_with('\n'), "leading blank line: {out:?}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn resuming_a_stamped_file_starts_a_new_line() {
        let p = scratch("stampresume");
        let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
        w.append(&seg(0.0, "one")).unwrap();
        drop(w);
        let mut w = StreamWriter::open(&p, Format::Timestamped).unwrap();
        w.append(&seg(0.0, "two")).unwrap();
        drop(w);
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "[00:00] one\n[00:00] two"
        );
        // A resumed session must not continue the previous session's line: its
        // clock restarts at zero, so the times would be nonsense.
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_missing_directory_is_created() {
        // Streaming to ~/transcripts/today.txt should not fail because the
        // folder does not exist yet.
        let dir = scratch("mkdir");
        let p = dir.join("nested/out.txt");
        let mut w = StreamWriter::open(&p, Format::Plain).unwrap();
        w.append(&seg(0.0, "x")).unwrap();
        assert!(p.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unwritable_path_fails_at_open_not_mid_session() {
        // Better to refuse at start than to discover it after an hour of
        // talking. A directory is the portable way to be unwritable-as-a-file:
        // this used to point at /proc/version, which on Windows is not
        // protected but simply absent, so the test created C:\proc\version
        // and passed for the wrong reason.
        let dir = scratch("unwritable");
        std::fs::create_dir_all(&dir).unwrap();
        let e = StreamWriter::open(&dir, Format::Plain);
        assert!(e.is_err(), "opening a directory as a file should fail");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
