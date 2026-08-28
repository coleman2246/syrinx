//! Writing transcripts to disk.
//!
//! Lives in the shared library rather than either front-end, so `syrinx save`
//! and the GUI's Save button produce byte-identical files. A CLI that wrote
//! subtly different output from the GUI would be a trap for anyone scripting
//! around it.

use crate::session::Segment;
use crate::stream::NEW_LINE_AFTER_SILENCE;
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

/// Group segments into turns: a maximal run of one source whose speaker
/// matches the last labelled one. An unlabelled segment attaches to the turn
/// already open rather than starting a fresh one -- usually it is a short
/// connective fragment the diarizer could not call.
///
/// A change of source always opens a turn, whatever the labels say. Speaker
/// numbers are only meaningful within one session, and separate mode runs a
/// session per source, so the mic's "Speaker 1" and the system's "Speaker 1"
/// are two different people who happen to have been numbered first. Grouping
/// on the number alone spliced them into a single paragraph under one
/// heading.
///
/// Exposed so anything that needs to know where a turn starts -- the GUI's
/// paragraph view, which coalesces a turn regardless of any pause within it,
/// and `render` below, which additionally splits a turn back into lines on a
/// silence gap -- shares one rule for it.
///
/// Segments rather than the plan's sketched rendered `String`: `render`
/// below needs the raw segments to format each of `Plain`/`Timestamped`/
/// `Labelled` differently per turn; the GUI's paragraph view (Task 10) can
/// concatenate the text itself.
pub fn turns(segments: &[Segment]) -> Vec<(Option<u32>, Vec<&Segment>)> {
    let mut out: Vec<(Option<u32>, Vec<&Segment>)> = Vec::new();
    let mut last_speaker: Option<u32> = None;
    let mut last_source: Option<&Option<String>> = None;
    for seg in segments {
        let source_changed = last_source.is_some_and(|last| *last != seg.source);
        // The previous source's numbering says nothing about this one, so it
        // is forgotten rather than carried across the boundary: otherwise the
        // new source's own "Speaker 1" would match it and open no turn.
        if source_changed {
            last_speaker = None;
        }
        let opens_turn = out.is_empty()
            || source_changed
            || (seg.speaker.is_some() && seg.speaker != last_speaker);
        match (opens_turn, out.last_mut()) {
            (false, Some((_, segs))) => segs.push(seg),
            _ => out.push((seg.speaker, vec![seg])),
        }
        if seg.speaker.is_some() {
            last_speaker = seg.speaker;
        }
        last_source = Some(&seg.source);
    }
    out
}

/// Turns as `(speaker, concatenated text)` pairs -- for a caller that wants
/// each turn's prose and nothing else. `render` below needs the raw segments
/// instead, to format stamps and sources per line; the GUI's paragraph view
/// (Task 10) is the intended caller here.
pub fn turn_texts(segments: &[Segment]) -> Vec<(Option<u32>, String)> {
    turns(segments)
        .into_iter()
        .map(|(speaker, segs)| {
            let text: String = segs.iter().map(|s| s.text.as_str()).collect();
            (speaker, text)
        })
        .collect()
}

/// Render segments in the requested format.
pub fn render(segments: &[Segment], fallback: &str, format: Format) -> String {
    match format {
        // Falls back to the flat transcript, which is all a session that
        // predates segment tracking would have.
        Format::Plain => {
            if segments.is_empty() {
                return fallback.trim().to_string();
            }
            // A `Speaker N: ` prefix at each turn's start, where labels
            // exist. Without any, `turns` yields one turn holding every
            // segment, and this is byte-identical to a flat concatenation.
            turns(segments)
                .into_iter()
                .map(|(speaker, segs)| {
                    let prefix = speaker.map(|n| format!("Speaker {n}: ")).unwrap_or_default();
                    let text: String = segs.iter().map(|s| s.text.as_str()).collect();
                    format!("{prefix}{text}")
                })
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string()
        }
        // Without any label, this must render exactly as it did before
        // turns existed: one line per segment, unconditionally. Only once a
        // label is present does a turn start merging lines together --
        // otherwise every plain recording's timestamps would start silently
        // vanishing on any two fragments less than 1.5s apart.
        Format::Timestamped | Format::Labelled if segments.iter().all(|s| s.speaker.is_none()) => {
            render_stamped_flat(segments, format)
        }
        Format::Timestamped | Format::Labelled => render_stamped(segments, format),
    }
}

/// `Timestamped`/`Labelled` before turns existed: one line per segment,
/// regardless of how close in time two fragments are.
fn render_stamped_flat(segments: &[Segment], format: Format) -> String {
    segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .map(|s| match (format, &s.source) {
            (Format::Labelled, Some(src)) => format!("{} [{}] {}", stamp(s.at), src, s.text.trim()),
            _ => format!("{} {}", stamp(s.at), s.text.trim()),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render `Timestamped` or `Labelled` once a label is present: one line per
/// turn, split further on a silence gap or, for `Labelled`, a source change
/// -- the same rule `StreamWriter` uses to continue a line, so a saved file
/// and a streamed one agree. Every line names the speaker whose turn it
/// belongs to, including one merely reopened after a pause: these formats
/// are one record per line, and something reading the file back -- an LLM
/// asked who said what, a grep -- has only the line in front of it.
///
/// Turn boundaries come from `turns` itself, so this and the GUI's paragraph
/// view can never disagree about where one starts; only the further split
/// into physical lines -- on a gap or a source change -- is local to saving.
fn render_stamped(segments: &[Segment], format: Format) -> String {
    let filtered: Vec<Segment> = segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .cloned()
        .collect();

    turns(&filtered)
        .into_iter()
        .flat_map(|(speaker, segs)| {
            lines_in_turn(segs, format)
                .into_iter()
                .map(move |line| render_line(&line, speaker, format))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split one turn's segments into physical lines on a silence gap or, for
/// `Labelled`, a source change. Every line belongs to the same turn, and so
/// to the same speaker: which line came first no longer changes how it is
/// rendered.
///
/// The source check is a guard rather than the mechanism: `turns` already
/// ends a turn at a source change, so no turn reaching here spans two. It is
/// kept because `StreamWriter::continues_line` states the same rule, and the
/// two are meant to be readable as one.
fn lines_in_turn(segs: Vec<&Segment>, format: Format) -> Vec<Vec<&Segment>> {
    let mut lines: Vec<Vec<&Segment>> = Vec::new();
    for seg in segs {
        let breaks = match lines.last().and_then(|l| l.last()) {
            Some(prev) => {
                (format == Format::Labelled && prev.source != seg.source)
                    || seg.at - prev.at >= NEW_LINE_AFTER_SILENCE
            }
            None => false,
        };
        if lines.is_empty() || breaks {
            lines.push(vec![seg]);
        } else {
            lines.last_mut().unwrap().push(seg);
        }
    }
    lines
}

/// One rendered line: its stamp, source (`Labelled`) and `speaker` -- the
/// speaker of the turn the line belongs to, not merely the one its first
/// segment happened to carry, so a line opened by an unlabelled fragment is
/// still attributed to whoever is talking -- then the text, spliced across
/// the segments sharing the line the same way a stream continuation splices
/// them (the interior spacing between fragments is kept; only the outer
/// edges are trimmed).
fn render_line(segs: &[&Segment], speaker: Option<u32>, format: Format) -> String {
    let first = segs[0];
    let source_part = match (format, &first.source) {
        (Format::Labelled, Some(src)) => format!("[{src}] "),
        _ => String::new(),
    };
    let speaker_prefix = speaker.map(|n| format!("Speaker {n}: ")).unwrap_or_default();
    // Only the first segment is ever trimmed: fully, if it is also the last
    // (matching the pre-turns single-line-per-segment convention); from the
    // left only otherwise, since `StreamWriter` writes every continuation
    // after it raw -- trimming the last segment's trailing whitespace here
    // too would make a saved file disagree with a streamed one.
    let text: String = segs
        .iter()
        .enumerate()
        .map(|(i, s)| match i {
            0 if segs.len() == 1 => s.text.trim().to_string(),
            0 => s.text.trim_start().to_string(),
            _ => s.text.clone(),
        })
        .collect();
    format!("{} {source_part}{speaker_prefix}{text}", stamp(first.at))
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

/// Where one source's half of `base` goes: `meeting.txt` and `Yeti` give
/// `meeting-yeti.txt`.
///
/// Shared by the two things that split one requested path into a file per
/// source -- [`save_per_source`] and separate mode's streaming -- because
/// they must agree. Streaming a conversation and then saving it split
/// should land on the same names, not on two conventions for one idea.
pub fn path_for_source(base: &Path, source: &str) -> PathBuf {
    let stem = base
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "transcript".into());
    let ext = base
        .extension()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "txt".into());
    base.parent()
        .unwrap_or(Path::new("."))
        .join(format!("{stem}-{}.{ext}", slug(source)))
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

    let mut written = Vec::new();
    for (name, segs) in groups {
        let path = path_for_source(base, &name);
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
            seq: None,
            speaker_provisional: false,
            at,
            text: text.into(),
            source: None,
            speaker: None,
        }
    }

    fn seg_src(at: f64, text: &str, src: &str) -> Segment {
        Segment {
            seq: None,
            speaker_provisional: false,
            at,
            text: text.into(),
            source: Some(src.into()),
            speaker: None,
        }
    }

    fn seg_spk(at: f64, text: &str, speaker: Option<u32>) -> Segment {
        Segment {
            seq: None,
            speaker_provisional: false,
            at,
            text: text.into(),
            source: None,
            speaker,
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
    fn a_source_gets_its_own_file_beside_the_one_asked_for() {
        let p = path_for_source(Path::new("/tmp/meeting.txt"), "Yeti (RNNoise)");
        assert_eq!(p, Path::new("/tmp/meeting-yeti-rnnoise.txt"));
    }

    #[test]
    fn a_streamed_split_lands_where_a_saved_split_would() {
        // Separate mode streams through `path_for_source` and saves through
        // `save_per_source`. Two conventions for one idea would leave a
        // conversation streamed to one set of files and saved to another.
        let base = tmp("split-agree").join("meeting.txt");
        let segs = [
            seg_src(0.0, "hello", "Mic"),
            seg_src(1.0, "hi", "System audio"),
        ];
        let saved = save_per_source(&base, &segs, Format::Plain).unwrap();
        let streamed: Vec<PathBuf> = ["Mic", "System audio"]
            .iter()
            .map(|s| path_for_source(&base, s))
            .collect();
        assert_eq!(saved, streamed);
        let _ = std::fs::remove_dir_all(tmp("split-agree"));
    }

    #[test]
    fn a_path_without_an_extension_still_splits() {
        // Streaming targets are whatever was typed into a file dialog.
        assert_eq!(
            path_for_source(Path::new("notes"), "Mic"),
            Path::new("notes-mic.txt")
        );
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

    #[test]
    fn turns_group_labelled_runs_and_attach_unlabelled_fragments() {
        let segs = [
            seg_spk(0.0, "we ship Thursday", Some(1)),
            seg_spk(0.6, " right", None),
            seg_spk(1.0, "no we don't", Some(2)),
        ];
        let t = turns(&segs);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].0, Some(1));
        assert_eq!(t[0].1.len(), 2, "the unlabelled fragment attaches");
        assert_eq!(t[1].0, Some(2));
        assert_eq!(t[1].1.len(), 1);
    }

    #[test]
    fn turns_without_any_label_is_a_single_turn() {
        let segs = [seg(0.0, "a"), seg(1.0, "b")];
        let t = turns(&segs);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].0, None);
        assert_eq!(t[0].1.len(), 2);
    }

    #[test]
    fn two_sources_numbered_the_same_are_two_turns() {
        // Regression: separate mode runs a session per source, each minting
        // its own Speaker 1, and merge_states interleaves them into one list.
        // Grouping on the number alone concatenated two different people's
        // words into a single paragraph under one heading.
        let mut mic = seg_spk(0.0, "we ship Thursday", Some(1));
        mic.source = Some("Mic".into());
        let mut system = seg_spk(0.6, "no we don't", Some(1));
        system.source = Some("System audio".into());

        let segs = [mic, system];
        let t = turns(&segs);
        assert_eq!(t.len(), 2, "a source change ends the turn");
        assert_eq!(t[0].1.len(), 1);
        assert_eq!(t[1].1.len(), 1);
    }

    #[test]
    fn a_new_source_does_not_inherit_the_previous_one_s_numbering() {
        // The second source opens unlabelled and only then produces its own
        // Speaker 1. If the previous source's 1 were still remembered, that
        // labelled fragment would match it, open no turn, and the turn would
        // stay reported as unlabelled.
        let mut mic = seg_spk(0.0, "we ship Thursday", Some(1));
        mic.source = Some("Mic".into());
        let mut system_first = seg_spk(0.6, "hmm", None);
        system_first.source = Some("System audio".into());
        let mut system_then = seg_spk(1.0, "no we don't", Some(1));
        system_then.source = Some("System audio".into());

        let segs = [mic, system_first, system_then];
        let t = turns(&segs);
        assert_eq!(t.len(), 3);
        assert_eq!(t[1].0, None, "the new source's opening fragment");
        assert_eq!(t[2].0, Some(1), "its own Speaker 1, freshly announced");
    }

    #[test]
    fn plain_keeps_two_sources_apart_even_sharing_a_number() {
        // What the bug actually looked like: one heading over both halves.
        let mut mic = seg_spk(0.0, "we ship Thursday", Some(1));
        mic.source = Some("Mic".into());
        let mut system = seg_spk(0.6, "no we don't", Some(1));
        system.source = Some("System audio".into());
        assert_eq!(
            render(&[mic, system], "", Format::Plain),
            "Speaker 1: we ship Thursday\nSpeaker 1: no we don't"
        );
    }

    #[test]
    fn turn_texts_concatenates_each_turns_segments() {
        let segs = [
            seg_spk(0.0, "we ship Thursday", Some(1)),
            seg_spk(0.6, " right", None),
            seg_spk(1.0, "no we don't", Some(2)),
        ];
        assert_eq!(
            turn_texts(&segs),
            vec![
                (Some(1), "we ship Thursday right".to_string()),
                (Some(2), "no we don't".to_string()),
            ]
        );
    }

    #[test]
    fn turn_texts_leaves_an_unlabelled_leading_turn_alone() {
        // Before any speaker has appeared there is nothing honest to call the
        // turn, so it stays unlabelled rather than borrowing the next one's.
        let segs = [seg_spk(0.0, "hello ", None), seg_spk(0.6, "there", Some(1))];
        assert_eq!(
            turn_texts(&segs),
            vec![
                (None, "hello ".to_string()),
                (Some(1), "there".to_string()),
            ]
        );
    }

    #[test]
    fn turn_texts_without_any_label_is_one_turn() {
        let segs = [seg(0.0, "a"), seg(1.0, "b")];
        assert_eq!(turn_texts(&segs), vec![(None, "ab".to_string())]);
    }

    #[test]
    fn a_speaker_change_breaks_the_line_and_names_the_speaker() {
        let segs = [
            seg_spk(0.0, "we ship Thursday", Some(1)),
            seg_spk(0.6, "no we don't", Some(2)),
        ];
        assert_eq!(
            render(&segs, "", Format::Timestamped),
            "[00:00] Speaker 1: we ship Thursday\n[00:00] Speaker 2: no we don't"
        );
    }

    #[test]
    fn an_unlabelled_fragment_stays_on_the_current_turn() {
        // Usually a short connective the diarizer could not call; breaking
        // the paragraph for it would shred every transcript.
        let segs = [
            seg_spk(0.0, "so the plan", Some(1)),
            seg_spk(0.6, " is simple", None),
        ];
        assert_eq!(
            render(&segs, "", Format::Timestamped),
            "[00:00] Speaker 1: so the plan is simple"
        );
    }

    #[test]
    fn plain_gains_prefixes_only_when_labels_exist() {
        // Without labels the plain format must stay byte-identical to today.
        let unlabelled = [seg_spk(0.0, "hello ", None), seg_spk(0.6, "world", None)];
        assert_eq!(render(&unlabelled, "", Format::Plain), "hello world");

        let labelled = [
            seg_spk(0.0, "hello", Some(1)),
            seg_spk(0.6, "hi there", Some(2)),
        ];
        assert_eq!(
            render(&labelled, "", Format::Plain),
            "Speaker 1: hello\nSpeaker 2: hi there"
        );
    }

    #[test]
    fn labelled_keeps_the_source_and_adds_the_speaker() {
        // The bracket names the capture source exactly as today;
        // diarization adds the speaker, it does not rename the source.
        let mut s = seg_spk(0.0, "hello", Some(2));
        s.source = Some("System audio".into());
        assert_eq!(
            render(&[s], "", Format::Labelled),
            "[00:00] [System audio] Speaker 2: hello"
        );
    }

    #[test]
    fn unlabelled_close_segments_still_get_their_own_stamped_line() {
        // Regression: turn-merging must not engage when no segment carries a
        // speaker, or a plain recording's timestamps would start silently
        // vanishing on any two fragments less than 1.5s apart.
        let segs = [seg(0.0, "hello "), seg(0.6, "world")];
        assert_eq!(
            render(&segs, "", Format::Timestamped),
            "[00:00] hello\n[00:00] world"
        );
    }

    #[test]
    fn plain_does_not_break_a_turn_on_a_silence_gap() {
        // A turn is a speaker run, not a time window: unlike Timestamped and
        // Labelled, Plain has no stamps to go stale, so nothing about a
        // pause should split one speaker's paragraph in two. The speaker is
        // named once, at the top of the paragraph -- the other half of the
        // format split that the next test states.
        let segs = [
            seg_spk(0.0, "hello ", Some(1)),
            seg_spk(65.0, "world", Some(1)),
        ];
        assert_eq!(render(&segs, "", Format::Plain), "Speaker 1: hello world");
    }

    #[test]
    fn a_silence_gap_within_a_turn_repeats_the_speaker() {
        // The stamped formats used to leave the speaker off a line reopened
        // by a pause, on the grounds that the turn had already been named.
        // But a pause mid-turn is the ordinary shape of a meeting, so most
        // lines came out with a time and nobody attached to them. A stamped
        // line is a record on its own, and the record has to say who.
        let segs = [
            seg_spk(0.0, "hello", Some(1)),
            seg_spk(65.0, "still there", Some(1)),
        ];
        assert_eq!(
            render(&segs, "", Format::Timestamped),
            "[00:00] Speaker 1: hello\n[01:05] Speaker 1: still there"
        );
    }

    #[test]
    fn labelled_repeats_the_speaker_after_a_gap_as_well() {
        // Same rule as Timestamped: the source label never stood in for the
        // speaker, and one microphone can carry several people.
        let mut first = seg_spk(0.0, "hello", Some(1));
        first.source = Some("Yeti".into());
        let mut then = seg_spk(65.0, "still there", Some(1));
        then.source = Some("Yeti".into());
        assert_eq!(
            render(&[first, then], "", Format::Labelled),
            "[00:00] [Yeti] Speaker 1: hello\n[01:05] [Yeti] Speaker 1: still there"
        );
    }

    #[test]
    fn a_line_reopened_by_an_unlabelled_fragment_names_the_turn_anyway() {
        // The diarizer says nothing about stretches it has not heard enough
        // of, so the fragment that happens to follow a pause often carries
        // no label of its own. `turns` already counts it as part of the turn
        // -- the line it opens is attributed to that turn's speaker.
        let segs = [
            seg_spk(0.0, "hello", Some(1)),
            seg_spk(65.0, "still there", None),
        ];
        assert_eq!(
            render(&segs, "", Format::Timestamped),
            "[00:00] Speaker 1: hello\n[01:05] Speaker 1: still there"
        );
    }

    #[test]
    fn lines_before_any_speaker_is_known_stay_unattributed() {
        // A voice needs a few seconds before it is given a number, so the
        // opening of a session arrives unlabelled. There is nothing to
        // attribute it to yet, and the streamed file it must agree with is
        // append-only -- the first label to arrive cannot be applied
        // backwards over lines already written.
        let segs = [
            seg_spk(0.0, "hello", None),
            seg_spk(65.0, "we ship Thursday", Some(1)),
        ];
        assert_eq!(
            render(&segs, "", Format::Timestamped),
            "[00:00] hello\n[01:05] Speaker 1: we ship Thursday"
        );
    }

    #[test]
    fn a_pause_mid_turn_is_attributed_the_same_way_saved_and_streamed() {
        // Two independent renderers state the same rule, so this is what
        // catches them drifting: an unlabelled opening, a gap mid-turn, and
        // a gap the diarizer could not label either side of.
        let segs = [
            seg_spk(0.0, "hello", None),
            seg_spk(65.0, "we ship Thursday", Some(1)),
            seg_spk(130.0, "or Friday", Some(1)),
            seg_spk(200.0, " probably", None),
        ];
        let saved = render(&segs, "", Format::Timestamped);
        assert_eq!(
            saved,
            "[00:00] hello\n[01:05] Speaker 1: we ship Thursday\n\
             [02:10] Speaker 1: or Friday\n[03:20] Speaker 1: probably"
        );

        let dir = tmp("stream-parity-gap");
        let p = dir.join("t.txt");
        let mut w = crate::stream::StreamWriter::open(&p, Format::Timestamped).unwrap();
        for s in &segs {
            w.append(s).unwrap();
        }
        drop(w);
        let streamed = std::fs::read_to_string(&p).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(saved, streamed);
    }

    #[test]
    fn a_merged_lines_trailing_whitespace_matches_between_save_and_stream() {
        // Regression: render_line used to trim_end the last segment of a
        // merged line, but StreamWriter writes a continuation's text raw --
        // so a saved file and a streamed one disagreed on trailing
        // whitespace, contradicting render_stamped's own claim that they
        // agree.
        let segs = [
            seg_spk(0.0, "hello ", Some(1)),
            seg_spk(0.6, "world  ", Some(1)),
        ];
        let saved = render(&segs, "", Format::Timestamped);

        let dir = tmp("stream-parity");
        let p = dir.join("t.txt");
        let mut w = crate::stream::StreamWriter::open(&p, Format::Timestamped).unwrap();
        for s in &segs {
            w.append(s).unwrap();
        }
        drop(w);
        let streamed = std::fs::read_to_string(&p).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(saved, streamed);
    }
}
