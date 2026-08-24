//! Writing transcripts to disk.
//!
//! Lives in the shared library rather than either front-end, so `syrinx save`
//! and the GUI's Save button produce byte-identical files. A CLI that wrote
//! subtly different output from the GUI would be a trap for anyone scripting
//! around it.

use crate::session::Segment;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// How a saved transcript is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// Continuous prose. What you want if the transcript is the point.
    #[default]
    Plain,
    /// Each fragment prefixed with its time, `[MM:SS]`. What you want when the
    /// transcript is an index into a recording.
    Timestamped,
    /// Time and source on each line. What you want for several sources
    /// transcribed separately, where who said it matters as much as when.
    Labelled,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::Plain => "Plain",
            Format::Timestamped => "Timestamped",
            Format::Labelled => "Timestamped + source",
        }
    }

    /// The value as written in the config file.
    pub fn name(self) -> &'static str {
        match self {
            Format::Plain => "plain",
            Format::Timestamped => "timestamped",
            Format::Labelled => "labelled",
        }
    }

    /// One line for the generated config.
    pub fn summary(self) -> &'static str {
        match self {
            Format::Plain => "continuous prose. For when the words are the point",
            Format::Timestamped => "each line prefixed [MM:SS]. An index into a recording",
            Format::Labelled => "time and source per line. For several sources at once",
        }
    }

    pub const ALL: [Format; 3] = [Format::Plain, Format::Timestamped, Format::Labelled];
}

/// Format seconds as `[MM:SS]`, or `[HH:MM:SS]` past an hour.
///
/// Hours are only shown when needed: prefixing every line of a two-minute note
/// with `00:` is noise.
pub fn stamp(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("[{h:02}:{m:02}:{s:02}]")
    } else {
        format!("[{m:02}:{s:02}]")
    }
}

/// Render segments in the requested format.
pub fn render(segments: &[Segment], fallback: &str, format: Format) -> String {
    match format {
        // Falls back to the flat transcript, which is all a session that
        // predates segment tracking would have.
        Format::Plain => {
            if segments.is_empty() {
                fallback.trim().to_string()
            } else {
                segments
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<String>()
                    .trim()
                    .to_string()
            }
        }
        Format::Timestamped => segments
            .iter()
            .filter(|s| !s.text.trim().is_empty())
            .map(|s| format!("{} {}", stamp(s.at), s.text.trim()))
            .collect::<Vec<_>>()
            .join("\n"),
        // Falls back to the plain timestamp when a segment has no source, so a
        // single-source recording saved this way is not littered with empty
        // brackets.
        Format::Labelled => segments
            .iter()
            .filter(|s| !s.text.trim().is_empty())
            .map(|s| match &s.source {
                Some(src) => format!("{} [{}] {}", stamp(s.at), src, s.text.trim()),
                None => format!("{} {}", stamp(s.at), s.text.trim()),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Split segments by source, in first-appearance order.
///
/// For saving each source to its own file. Order is by when a source was first
/// heard rather than alphabetical, so file numbering matches the conversation.
pub fn by_source(segments: &[Segment]) -> Vec<(String, Vec<Segment>)> {
    let mut order: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<Segment>> =
        std::collections::HashMap::new();
    for s in segments {
        let key = s.source.clone().unwrap_or_else(|| "transcript".into());
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(s.clone());
    }
    order
        .into_iter()
        .filter_map(|k| groups.remove(&k).map(|v| (k, v)))
        .collect()
}

/// Turn a source name into something safe to put in a filename.
pub fn slug(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    // Collapse runs and trim, so "Yeti (RNNoise)" does not become
    // "yeti--rnnoise-".
    let mut out = String::new();
    let mut last_dash = false;
    for c in s.chars() {
        if c == '-' {
            if !last_dash && !out.is_empty() {
                out.push('-');
            }
            last_dash = true;
        } else {
            out.push(c);
            last_dash = false;
        }
    }
    out.trim_end_matches('-').chars().take(40).collect()
}

/// Save each source to its own file beside `base`, returning the paths.
pub fn save_per_source(
    base: &Path,
    segments: &[Segment],
    format: Format,
) -> Result<Vec<PathBuf>> {
    let groups = by_source(segments);
    if groups.is_empty() {
        anyhow::bail!("nothing to save: the transcript is empty");
    }
    let stem = base
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "transcript".into());
    let ext = base
        .extension()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "txt".into());
    let dir = base.parent().unwrap_or(Path::new("."));

    let mut written = Vec::new();
    for (name, segs) in groups {
        let path = dir.join(format!("{stem}-{}.{ext}", slug(&name)));
        let body = render(&segs, "", format);
        // A source that produced nothing is skipped rather than writing an
        // empty file that looks like a failed recording.
        if body.trim().is_empty() {
            continue;
        }
        write(&path, &body)?;
        written.push(path);
    }
    if written.is_empty() {
        anyhow::bail!("nothing to save: no source produced any text");
    }
    Ok(written)
}

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
    format!("{stamp}.txt")
}

/// Local time as `YYYY-MM-DD_HH-MM-SS`.
///
/// Shelling out to `date` avoids a chrono dependency for one format string,
/// and falls back to a UTC epoch if that fails so saving never breaks over a
/// timestamp.
pub fn timestamp() -> String {
    chrono::Local::now().format(STAMP_FORMAT).to_string()
}

/// Local date and time, ordered so that sorting by name sorts by time.
///
/// Was `date +%Y-%m-%d_%H-%M-%S` in a subprocess, which is not a command on
/// Windows: every filename there fell back to `epoch-1755…`, which is neither
/// readable nor sortable by eye. Formatting it in-process gives the same answer
/// on both platforms and drops a process spawn from every save.
const STAMP_FORMAT: &str = "%Y-%m-%d_%H-%M-%S";

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

/// Render and save in one step. The path a front-end shows the user.
pub fn save_rendered(
    path: Option<&Path>,
    segments: &[Segment],
    fallback: &str,
    format: Format,
) -> Result<PathBuf> {
    let body = render(segments, fallback, format);
    let path = match path {
        Some(p) => p.to_path_buf(),
        None => default_dir().join(filename_for(&timestamp())),
    };
    write(&path, &body)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timestamp_is_the_local_date_and_time() {
        // Shape: 2026-08-21_14-53-07. Checked digit by digit because a wrong
        // format string produces something plausible-looking but useless, and
        // this ends up as a filename.
        let s = timestamp();
        assert_eq!(s.len(), 19, "got {s:?}");
        let bytes = s.as_bytes();
        for i in [4, 7] {
            assert_eq!(bytes[i], b'-', "expected a date separator at {i} in {s:?}");
        }
        assert_eq!(bytes[10], b'_', "expected date and time to be split in {s:?}");
        for i in [13, 16] {
            assert_eq!(bytes[i], b'-', "expected a time separator at {i} in {s:?}");
        }
        for (i, c) in s.chars().enumerate() {
            if ![4, 7, 10, 13, 16].contains(&i) {
                assert!(c.is_ascii_digit(), "{c:?} at {i} is not a digit in {s:?}");
            }
        }
    }

    #[test]
    fn a_timestamp_never_falls_back_to_an_epoch_count() {
        // It used to shell out to `date`, which does not exist on Windows, so
        // every filename there was `epoch-1755…`.
        assert!(!timestamp().contains("epoch"), "{}", timestamp());
    }

    #[test]
    fn timestamps_sort_the_same_way_as_time() {
        // The whole reason for this ordering: a folder of these listed
        // alphabetically is in the order they were recorded.
        let mut names = ["2026-08-21_09-00-00", "2026-01-02_23-59-59", "2026-08-21_10-00-00"];
        names.sort_unstable();
        assert_eq!(
            names,
            ["2026-01-02_23-59-59", "2026-08-21_09-00-00", "2026-08-21_10-00-00"]
        );
    }

    #[test]
    fn a_filename_is_the_timestamp_and_nothing_else() {
        assert_eq!(filename_for("2026-08-21_14-53-07"), "2026-08-21_14-53-07.txt");
    }

    #[test]
    fn a_generated_filename_is_legal_on_windows() {
        // Windows refuses these characters outright, and a colon is the
        // obvious way to write a time.
        let name = filename_for(&timestamp());
        for bad in ['<', '>', ':', '"', '/', '\\', '|', '?', '*'] {
            assert!(!name.contains(bad), "{name:?} contains {bad:?}");
        }
    }

    #[test]
    fn config_names_match_what_serde_accepts() {
        // Written into the generated config; a mismatch would emit a file
        // syrinx cannot read back.
        for f in Format::ALL {
            let quoted = format!("\"{}\"", f.name());
            assert_eq!(serde_json::from_str::<Format>(&quoted).unwrap(), f);
        }
    }

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

    fn seg(at: f64, text: &str) -> Segment {
        Segment {
            at,
            text: text.into(),
            source: None,
            speaker: None,
        }
    }

    fn seg_src(at: f64, text: &str, src: &str) -> Segment {
        Segment {
            at,
            text: text.into(),
            source: Some(src.into()),
            speaker: None,
        }
    }

    #[test]
    fn labelled_format_names_the_source_on_each_line() {
        let segs = [seg_src(0.0, "hello", "Mic"), seg_src(5.0, "hi", "System")];
        let out = render(&segs, "", Format::Labelled);
        assert_eq!(out, "[00:00] [Mic] hello\n[00:05] [System] hi");
    }

    #[test]
    fn labelled_format_omits_empty_brackets_for_a_single_source() {
        // A one-source recording saved this way should not be littered with
        // empty labels.
        let segs = [seg(1.0, "hello")];
        assert_eq!(render(&segs, "", Format::Labelled), "[00:01] hello");
    }

    #[test]
    fn segments_group_by_source_in_first_appearance_order() {
        let segs = [
            seg_src(0.0, "a", "System"),
            seg_src(1.0, "b", "Mic"),
            seg_src(2.0, "c", "System"),
        ];
        let groups = by_source(&segs);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, "System", "first heard should come first");
        assert_eq!(groups[0].1.len(), 2);
        assert_eq!(groups[1].0, "Mic");
    }

    #[test]
    fn source_names_become_safe_filenames() {
        assert_eq!(slug("Yeti (RNNoise)"), "yeti-rnnoise");
        assert_eq!(slug("Firefox — YouTube"), "firefox-youtube");
        assert!(!slug("a/b\\c").contains('/'));
    }

    #[test]
    fn a_very_long_source_name_is_truncated() {
        assert!(slug(&"x".repeat(200)).len() <= 40);
    }

    #[test]
    fn timestamps_omit_hours_until_they_matter() {
        assert_eq!(stamp(0.0), "[00:00]");
        assert_eq!(stamp(65.0), "[01:05]");
        // Prefixing a short note with 00: every line is noise.
        assert_eq!(stamp(3661.0), "[01:01:01]");
    }

    #[test]
    fn a_negative_time_does_not_underflow() {
        // as u64 on a negative float wraps to something enormous.
        assert_eq!(stamp(-5.0), "[00:00]");
    }

    #[test]
    fn plain_format_joins_segments_into_prose() {
        let segs = [seg(0.5, "hello "), seg(1.5, "world")];
        assert_eq!(render(&segs, "", Format::Plain), "hello world");
    }

    #[test]
    fn timestamped_format_prefixes_each_fragment() {
        let segs = [seg(0.0, "hello "), seg(65.0, "world")];
        assert_eq!(
            render(&segs, "", Format::Timestamped),
            "[00:00] hello\n[01:05] world"
        );
    }

    #[test]
    fn plain_falls_back_to_the_flat_transcript() {
        // A session that kept no segments must still be saveable.
        assert_eq!(render(&[], " some text ", Format::Plain), "some text");
    }

    #[test]
    fn empty_segments_are_skipped_in_timestamped_output() {
        // A blank line with a timestamp on it is just noise.
        let segs = [seg(0.0, "real "), seg(1.0, "   "), seg(2.0, "text")];
        assert_eq!(render(&segs, "", Format::Timestamped).lines().count(), 2);
    }

    #[test]
    fn a_timestamp_is_always_produced() {
        // Saving must never fail because a clock or a subprocess misbehaved.
        assert!(!timestamp().is_empty());
    }
}
