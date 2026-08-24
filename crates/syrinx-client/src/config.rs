//! Shared client configuration.
//!
//! One file for every front-end. The CLI and the GUI want the same server and
//! token, and making the user write them twice invites them to diverge.

use crate::mode::OutputMode;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Config {
    /// The address to dial, used exactly as written.
    ///
    /// Nothing is inferred: no scheme, no port, no path. syrinx used to take a
    /// bare host and build the URL around it, which is convenient right up to
    /// the first address it guesses wrong -- and then there is no way to say
    /// what you actually meant. What is in the file is what gets dialled.
    ///
    /// `server` is accepted as a name for the same setting.
    #[serde(default = "default_url", alias = "server")]
    pub url: String,
    pub token: String,
    /// Remembered source, as a `Source::stable_key`. Node ids change between
    /// runs, so a key is stored rather than an id.
    #[serde(default)]
    pub source_key: Option<String>,
    #[serde(default)]
    pub mode: OutputMode,
    /// Ask the server for anonymous speaker labels (Speaker 1, Speaker 2,
    /// ...) in transcribe mode. Best-effort: a server with no diarization
    /// models installed still works, just unlabelled -- see
    /// `SessionState::diarize` for the honest answer once connected.
    #[serde(default)]
    pub diarize: bool,
    /// How text is typed at the cursor. Electron applications such as Teams
    /// need `ydotool`; see the Method docs.
    #[serde(default)]
    pub inject: crate::inject::Method,
    /// Append the transcript to this file while it is being dictated.
    ///
    /// Unset by default. Set it and every session appends to the same file, so
    /// stopping and starting continues where it left off.
    #[serde(default)]
    pub stream_to: Option<String>,
    /// Layout for saved and streamed transcripts.
    #[serde(default)]
    pub format: crate::save::Format,
    /// Global hotkey that starts and stops dictation, e.g. `ctrl+alt+d`.
    ///
    /// Unset by default: claiming a key combination for the whole desktop is
    /// not something to do to someone without being asked.
    #[serde(default)]
    pub hotkey: Option<String>,
    /// waybar realtime signal for the status indicator.
    #[serde(default = "default_waybar_signal")]
    pub waybar_signal: u8,
}

fn default_url() -> String {
    "ws://127.0.0.1:8770/v1/stream".into()
}

fn default_waybar_signal() -> u8 {
    8
}

impl Config {
    /// Where to stream the transcript, with `~` expanded.
    pub fn stream_path(&self) -> Option<PathBuf> {
        self.stream_to.as_deref().map(expand_tilde)
    }

    /// Canonical location.
    pub fn default_path() -> PathBuf {
        config_base().join("syrinx/config.toml")
    }

    /// Load from the canonical path, falling back to the older per-binary
    /// locations so an existing setup keeps working after the consolidation.
    pub fn load(explicit: Option<PathBuf>) -> Result<Self> {
        let explicit_path = explicit.clone();
        let candidates = match explicit {
            Some(p) => vec![p],
            None => vec![
                Self::default_path(),
                config_base().join("parakeet/config.toml"),
                config_base().join("parakeet-type/config.toml"),
                config_base().join("syrinx-gui/config.toml"),
            ],
        };
        // An explicit --config is a claim the file exists; silently creating a
        // different one there would hide the typo that caused it.
        let explicit_given = explicit_path.is_some();

        for p in &candidates {
            match std::fs::read_to_string(p) {
                Ok(text) => {
                    // A parse failure is nearly always a missing required
                    // field, and naming it without showing a whole valid file
                    // invites replacing the file with just that field.
                    let cfg: Config = toml::from_str(&text).map_err(|e| {
                        anyhow::anyhow!("{}\n\n{}", e, EXAMPLE_HINT).context(format!(
                            "the config at {} is not valid",
                            p.display()
                        ))
                    })?;
                    // Loading a legacy path is announced. It used to be
                    // silent, so deleting the canonical config did not
                    // regenerate it -- an old file quietly took over, with
                    // whatever settings it happened to carry. Now that the
                    // default mode types at the cursor, an unnoticed config is
                    // not a harmless surprise.
                    if p != &Self::default_path() {
                        tracing::warn!(
                            "using the older config at {}. Rename or copy it to {} \
                             to stop it taking precedence.",
                            p.display(),
                            Self::default_path().display()
                        );
                    }
                    if let Err(e) = check_url(&cfg.url) {
                        anyhow::bail!("`url` in {} {e}", p.display());
                    }
                    // The generated file ships an empty token, so this is the
                    // normal second step of first-run rather than an edge case.
                    if cfg.token.trim().is_empty() {
                        anyhow::bail!(
                            "`token` is empty in {}.\n\n\
                             Set it to the same value as `token` in the server's \
                             config.toml, then run again.",
                            p.display()
                        );
                    }
                    return Ok(cfg);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e).with_context(|| format!("reading {}", p.display())),
            }
        }
        let path = Self::default_path();
        if explicit_given {
            anyhow::bail!("no config at {}", path.display());
        }

        // Nothing anywhere: write a documented starter file rather than make
        // the user derive its shape from an error message.
        write_template(&path).with_context(|| format!("creating {}", path.display()))?;
        anyhow::bail!(
            "No config existed, so a documented one was written to:\n  {}\n\n\
             Set `token` to match the server's token, then run again. \
             Every other setting is already at its default, and the file lists \
             the values each one accepts.",
            path.display()
        )
    }

    /// Write the config back, keeping comments and layout.
    ///
    /// The GUI saves whenever a source is chosen. Serialising the struct would
    /// be simpler, but it would erase the generated documentation on the first
    /// save -- the comments would survive exactly until the config was used.
    /// So values are edited into the existing document, and only a file that
    /// does not exist yet is written from the template.
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let existing = std::fs::read_to_string(path).unwrap_or_else(|_| template());
        let mut doc = existing
            .parse::<toml_edit::DocumentMut>()
            // A hand-edited file that no longer parses should not cost the user
            // their settings; fall back to a fresh document.
            .unwrap_or_else(|_| template().parse().expect("the template must parse"));

        // Written under whichever name the file already uses, so a config
        // saying `server` is not silently rewritten to `url`.
        let key = if doc.get("server").is_some() && doc.get("url").is_none() {
            "server"
        } else {
            "url"
        };
        doc[key] = toml_edit::value(&self.url);
        doc["token"] = toml_edit::value(&self.token);
        doc["mode"] = toml_edit::value(self.mode.name());
        doc["diarize"] = toml_edit::value(self.diarize);
        doc["inject"] = toml_edit::value(self.inject.name());
        doc["format"] = toml_edit::value(self.format.name());
        doc["waybar_signal"] = toml_edit::value(i64::from(self.waybar_signal));

        // Every optional field has to be handled, or a setting changed from a
        // front-end is applied and then lost on restart. Three of these were
        // missing, which is why `format` chosen in the GUI never survived.
        set_or_remove(&mut doc, "source_key", self.source_key.as_deref());
        set_or_remove(&mut doc, "stream_to", self.stream_to.as_deref());
        set_or_remove(&mut doc, "hotkey", self.hotkey.as_deref());

        std::fs::write(path, doc.to_string())
            .with_context(|| format!("writing {}", path.display()))
    }
}

/// Write a key, or remove it entirely when there is no value.
fn set_or_remove(doc: &mut toml_edit::DocumentMut, key: &str, value: Option<&str>) {
    match value {
        Some(v) => doc[key] = toml_edit::value(v),
        None => {
            doc.remove(key);
        }
    }
}

/// Reject an address that cannot possibly work, without rewriting it.
///
/// The value is used exactly as given, so this only says no -- it never
/// substitutes. Catching it here means a typo is reported against the config
/// file rather than surfacing later as a connection failure.
fn check_url(url: &str) -> Result<(), String> {
    let u = url.trim();
    if u.is_empty() {
        return Err("is empty".into());
    }
    if !(u.starts_with("ws://") || u.starts_with("wss://")) {
        return Err(format!(
            "is {u:?}, which has no ws:// or wss:// scheme. \
             Write the whole address, e.g. \"ws://192.168.1.10:8770/v1/stream\""
        ));
    }
    let rest = u.split_once("://").map(|(_, r)| r).unwrap_or("");
    if rest.is_empty() || rest.starts_with('/') {
        return Err(format!("is {u:?}, which names no host"));
    }
    Ok(())
}

/// Write the starter config, without clobbering anything already there.
fn write_template(path: &std::path::Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    // create_new: two front-ends starting at once must not race, and an
    // existing file must never be overwritten by a template.
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut f) => {
            use std::io::Write;
            f.write_all(template().as_bytes())?;
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// The starter config written when none exists.
///
/// Generated from the enums rather than kept as a string literal, so a new
/// variant cannot silently go undocumented -- the file is only as good as its
/// guarantee that it lists every value that works.
///
/// **The same on every platform.** It once listed only what the generating
/// machine could do, which read well until you kept one config for a desktop
/// and a laptop: the file was then wrong on whichever machine did not write
/// it. Every option is listed with the platform it applies to, and `inject`
/// defaults to `auto` so the default itself needs no platform of its own.
pub fn template() -> String {
    let mut s = String::new();
    s.push_str(
        "# Syrinx client configuration.\n\
         #\n\
         # Written automatically because no config existed. Every setting below is\n\
         # at its default except `token`, which has no sensible default.\n\
         # Delete any line to go back to the default.\n\n",
    );

    s.push_str(&comment_wrap(
        "The address to dial, used exactly as written: scheme, host, port and \
         path. Nothing is inferred. Behind TLS this is the public name and \
         the port is left off, because wss means 443 -- for example \
         wss://dictate.example.com/v1/stream",
    ));
    s.push_str(&format!("url = \"{}\"\n\n", default_url()));

    s.push_str(
        "# Shared secret, matching `token` in the server's config.toml.\n\
         # Required: the server refuses a session without it.\n",
    );
    s.push_str("token = \"\"\n\n");

    s.push_str("# What to do with the text that comes back.\n");
    for m in OutputMode::ALL {
        s.push_str(&format!("#   \"{}\"{} -- {}\n", m.name(), pad(m.name()), m.summary()));
    }
    s.push_str(&format!("mode = \"{}\"\n\n", OutputMode::default().name()));

    s.push_str(&comment_wrap(
        "Ask the server for anonymous speaker labels (Speaker 1, Speaker 2, \
         ...) on the transcript, for telling meeting participants apart. \
         Only applies to \"transcribe\" mode, and needs a server with \
         diarization models installed -- the server says in its handshake \
         whether labels will come.",
    ));
    s.push_str("diarize = false\n\n");

    s.push_str("# How text is typed at the cursor, for the modes above that type.\n");
    for m in crate::inject::Method::ALL {
        s.push_str(&format!("#   \"{}\"{} -- {}\n", m.name(), pad(m.name()), m.summary()));
    }
    s.push_str(&format!(
        "inject = \"{}\"\n\n",
        crate::inject::Method::default().name()
    ));

    // Stated unconditionally rather than only on Wayland, so the file reads
    // the same everywhere and explains why the same setting behaves
    // differently on the machine at the other end.
    s.push_str(&comment_wrap(crate::hotkey::PORTABILITY_NOTE));
    s.push_str("# hotkey = \"ctrl+alt+d\"\n\n");

    s.push_str(&comment_wrap(
        "Append the transcript to this file as it is dictated, rather than \
         only saving at the end. Every session appends to the same file, so \
         stopping and starting continues where you left off, and a crash \
         costs the last sentence rather than the whole session.",
    ));
    s.push_str("# stream_to = \"~/transcripts/notes.txt\"\n\n");

    s.push_str("# Layout for saved and streamed transcripts.\n");
    for f in crate::save::Format::ALL {
        s.push_str(&format!(
            "#   \"{}\"{} -- {}\n",
            f.name(),
            pad(f.name()),
            f.summary()
        ));
    }
    s.push_str(&format!(
        "format = \"{}\"\n\n",
        crate::save::Format::default().name()
    ));

    s.push_str(
        "# Capture source to use, as printed by `syrinx sources`.\n\
         # Left unset, syrinx asks or uses the default input.\n\
         # source_key = \"...\"\n\n",
    );

    s.push_str("# Realtime signal number for the waybar status indicator.\n");
    s.push_str("# Linux only; ignored elsewhere.\n");
    s.push_str(&format!("waybar_signal = {}\n", default_waybar_signal()));
    s
}

/// Wrap prose into `#` comment lines that fit in a terminal.
///
/// Indented lines are passed through unwrapped: they are examples meant to be
/// copied, and reflowing a config line would break it.
fn comment_wrap(text: &str) -> String {
    const WIDTH: usize = 74;
    let mut out = String::new();
    // Consecutive ordinary lines form one paragraph and are reflowed
    // together. Wrapping each source line on its own re-breaks text that was
    // already broken, which produces a line containing a single word.
    let mut para: Vec<&str> = Vec::new();

    let flush = |para: &mut Vec<&str>, out: &mut String| {
        if para.is_empty() {
            return;
        }
        let mut line = String::new();
        for word in para.join(" ").split_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > WIDTH {
                out.push_str(&format!("# {line}\n"));
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if !line.is_empty() {
            out.push_str(&format!("# {line}\n"));
        }
        para.clear();
    };

    for raw in text.lines() {
        if raw.trim().is_empty() {
            flush(&mut para, &mut out);
            out.push_str("#\n");
        } else if raw.starts_with(' ') || raw.starts_with('\t') {
            // An indented line is an example meant to be copied; reflowing a
            // config line would break it.
            flush(&mut para, &mut out);
            out.push_str(&format!("#    {}\n", raw.trim()));
        } else {
            para.push(raw);
        }
    }
    flush(&mut para, &mut out);
    out
}

/// Pad a quoted value so the `--` descriptions line up in a column.
fn pad(name: &str) -> String {
    const WIDTH: usize = 12;
    " ".repeat(WIDTH.saturating_sub(name.len()))
}

/// A complete, valid config.
///
/// Shown whole rather than as the one field at fault, because a snippet reads
/// as "the file should contain this" and gets pasted over a working file.
const EXAMPLE_HINT: &str = "\
A complete config looks like this -- every field but `token` is optional,
but the file must contain all of the ones you want:

    url = \"ws://127.0.0.1:8770/v1/stream\"
    token = \"your-shared-token\"
    inject = \"wtype\"        # or \"ydotool\" for Electron apps, or \"paste\"
    mode = \"transcribe\"     # or \"type\", or \"both\"";

/// Where per-user config lives on this platform.
///
/// `%APPDATA%` on Windows, XDG on everything else. XDG_CONFIG_HOME is honoured
/// first on both, which is what lets the tests point at a scratch directory.
/// Expand a leading `~` to the home directory.
///
/// Done here rather than left to the shell: this value comes from a config
/// file and from a GUI text box, neither of which a shell has touched, and
/// a literal `~` directory is never what was meant.
pub fn expand_tilde(path: &str) -> PathBuf {
    let Some(rest) = path.strip_prefix('~') else {
        return PathBuf::from(path);
    };
    let rest = rest.trim_start_matches(['/', '\\']);
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from);
    match home {
        Ok(h) if rest.is_empty() => h,
        Ok(h) => h.join(rest),
        Err(_) => PathBuf::from(path),
    }
}

fn config_base() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(dir);
    }
    #[cfg(windows)]
    {
        if let Ok(dir) = std::env::var("APPDATA") {
            return PathBuf::from(dir);
        }
        PathBuf::from(std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into()))
            .join("AppData")
            .join("Roaming")
    }
    #[cfg(not(windows))]
    {
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch path unique to this test, so tests can run in parallel.
    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "syrinx-test-{}-{}-{:?}",
            std::process::id(),
            tag,
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn a_minimal_config_applies_defaults() {
        let c: Config = toml::from_str(r#"token = "abc""#).unwrap();
        assert_eq!(c.url, "ws://127.0.0.1:8770/v1/stream");
        assert_eq!(c.mode, OutputMode::Type);
        assert_eq!(c.waybar_signal, 8);
    }

    #[test]
    fn the_address_is_used_exactly_as_written() {
        // The whole contract. syrinx used to build a URL around a bare host,
        // which is convenient until it guesses wrong -- and then there is no
        // way to express what you actually meant.
        for written in [
            "ws://192.168.1.10:8770/v1/stream",
            "wss://dictate.example.com/v1/stream",
            "ws://localhost:9000/some/other/path",
            "wss://edge.example.com:8443/asr",
        ] {
            let c: Config =
                toml::from_str(&format!("token = \"t\"\nurl = \"{written}\"")).unwrap();
            assert_eq!(c.url, written, "the address was rewritten");
        }
    }

    #[test]
    fn server_is_accepted_as_a_name_for_the_same_setting() {
        // Configs written before this change say `server`. They must keep
        // working, and mean exactly the same thing.
        let c: Config = toml::from_str(
            "token = \"t\"\nserver = \"wss://dictate.example.com/v1/stream\"",
        )
        .unwrap();
        assert_eq!(c.url, "wss://dictate.example.com/v1/stream");
    }

    #[test]
    fn an_address_without_a_scheme_is_refused_not_repaired() {
        // Previously "192.168.1.10" became a full URL. Now it is an error,
        // reported against the config file rather than at connect time.
        let e = check_url("192.168.1.10").unwrap_err();
        assert!(e.contains("ws://"), "got: {e}");
        assert!(e.contains("192.168.1.10"), "should quote what was written: {e}");
    }

    #[test]
    fn an_empty_or_hostless_address_is_refused() {
        assert!(check_url("").is_err());
        assert!(check_url("   ").is_err());
        assert!(check_url("ws://").is_err());
        assert!(check_url("ws:///v1/stream").is_err());
    }

    #[test]
    fn a_well_formed_address_passes() {
        for u in [
            "ws://h:1/v1/stream",
            "wss://h/v1/stream",
            "ws://[::1]:8770/v1/stream",
            "  ws://h:1/v1/stream  ",
        ] {
            assert!(check_url(u).is_ok(), "{u} should be accepted");
        }
    }

    #[test]
    fn a_bad_address_names_the_file_it_came_from() {
        // The error has to say where to go and fix it.
        let dir = scratch("badurl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "token = \"t\"\nurl = \"192.168.1.10\"\n").unwrap();

        let e = Config::load(Some(path.clone())).unwrap_err();
        let text = format!("{e:#}");
        assert!(text.contains(&path.display().to_string()), "got: {text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generated_comments_fit_in_a_terminal() {
        // A config is read in a terminal; a 200-column comment wraps into
        // something unreadable.
        for line in template().lines() {
            assert!(line.len() <= 100, "line too long ({}): {line}", line.len());
        }
    }

    #[test]
    fn wrapped_comments_reflow_a_whole_paragraph() {
        // Wrapping each source line separately leaves orphans: a line with
        // one word on it, mid-sentence.
        let text = "one two three four five six seven eight nine ten\n\
                    eleven twelve thirteen fourteen fifteen sixteen";
        for line in comment_wrap(text).lines() {
            let words = line.trim_start_matches("# ").split_whitespace().count();
            assert!(words > 1, "orphaned line: {line:?}");
        }
    }

    #[test]
    fn wrapped_comments_keep_examples_copyable() {
        // An indented example is meant to be pasted; reflowing it breaks it.
        let out = comment_wrap("some prose here\n    bindsym $mod+n exec syrinx toggle");
        assert!(
            out.contains("#    bindsym $mod+n exec syrinx toggle"),
            "the example was reflowed: {out}"
        );
    }

    #[test]
    fn the_generated_template_is_a_config_syrinx_can_read() {
        // The whole point of generating it. A template that does not parse
        // would turn first-run into a bug report.
        let text = template().replace("token = \"\"", "token = \"t\"");
        let c: Config = toml::from_str(&text)
            .unwrap_or_else(|e| panic!("template does not parse: {e}\n---\n{text}"));
        assert_eq!(c.url, default_url());
        assert_eq!(c.mode, OutputMode::default());
        assert_eq!(c.inject, crate::inject::Method::default());
    }

    #[test]
    fn the_template_documents_every_value_there_is() {
        let text = template();
        for m in OutputMode::ALL {
            assert!(
                text.contains(&format!("\"{}\"", m.name())),
                "mode {} is undocumented:\n{text}",
                m.name()
            );
        }
        for m in crate::inject::Method::ALL {
            assert!(
                text.contains(&format!("\"{}\"", m.name())),
                "inject {} is undocumented:\n{text}",
                m.name()
            );
        }
    }

    #[test]
    fn the_template_is_the_same_on_every_platform() {
        // One config kept for a desktop and a laptop has to be right on both.
        // Anything only true of the generating machine breaks that, so the
        // template names no platform-specific default and hides no option.
        let text = template();
        assert!(text.contains("waybar_signal"), "a setting was hidden");
        assert!(
            text.contains("inject = \"auto\""),
            "the default must not name a platform: {text}"
        );
        for m in crate::inject::Method::ALL {
            assert!(text.contains(m.name()), "{} was hidden", m.name());
        }
        // The Wayland note belongs in every copy, not just the Wayland one.
        assert!(text.contains("Wayland"), "the portability note is missing");
    }

    #[test]
    fn the_canonical_path_is_tried_before_the_legacy_ones() {
        // Order is the whole contract: a leftover parakeet config must never
        // win over the real one.
        let base = config_base();
        assert_eq!(
            Config::default_path(),
            base.join("syrinx/config.toml"),
            "the canonical path moved"
        );
    }

    #[test]
    fn a_leading_tilde_becomes_the_home_directory() {
        // The value comes from a config file and a text box; no shell has
        // expanded it, and a literal `~` folder is never what was meant.
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap();
        assert_eq!(expand_tilde("~/notes.txt"), PathBuf::from(&home).join("notes.txt"));
        assert_eq!(expand_tilde("~"), PathBuf::from(&home));
    }

    #[test]
    fn a_path_without_a_tilde_is_untouched() {
        assert_eq!(expand_tilde("/tmp/a.txt"), PathBuf::from("/tmp/a.txt"));
        // Not a prefix, so not an expansion.
        assert_eq!(expand_tilde("/tmp/~x"), PathBuf::from("/tmp/~x"));
    }

    #[test]
    fn a_first_run_writes_a_config_and_says_where() {
        let dir = scratch("firstrun");
        let path = dir.join("config.toml");
        assert!(!path.exists());

        write_template(&path).unwrap();

        assert!(path.exists(), "the config should have been created");
        let e = Config::load(Some(path.clone())).unwrap_err();
        let text = format!("{e:#}");
        // The token is the one thing it cannot guess, so it must say so.
        assert!(text.contains("token"), "got: {text}");
        assert!(text.contains(&path.display().to_string()), "got: {text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writing_a_template_never_clobbers_an_existing_config() {
        // Two front-ends can start at once; neither may destroy a real config.
        let dir = scratch("clobber");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "token = \"mine\"\n").unwrap();

        write_template(&path).unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "token = \"mine\"\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_setting_survives_a_save_and_reload() {
        // Guards against the bug this test was written for: a field added to
        // Config but not to `save`, so a setting changed in the GUI applied
        // once and was gone after a restart. Every field is given a
        // non-default value, so a missing one cannot pass by coincidence.
        let dir = scratch("roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, template()).unwrap();

        let original = Config {
            url: "ws://laptop.lan:9001/v1/stream".into(),
            token: "a-token".into(),
            source_key: Some("some-source".into()),
            stream_to: Some("~/notes.txt".into()),
            format: crate::save::Format::Labelled,
            mode: OutputMode::Both,
            diarize: true,
            inject: crate::inject::Method::Paste,
            hotkey: Some("ctrl+alt+k".into()),
            waybar_signal: 3,
        };
        original.save(&path).unwrap();

        let reloaded = Config::load(Some(path.clone())).unwrap();
        assert_eq!(reloaded, original, "a setting was lost by save/load");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clearing_an_optional_setting_removes_it() {
        // Setting it back to None has to delete the line, or the old value
        // comes back on the next load.
        let dir = scratch("clearopt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "token = \"t\"\nstream_to = \"/tmp/x.txt\"\nhotkey = \"ctrl+alt+d\"\n",
        )
        .unwrap();

        let mut c = Config::load(Some(path.clone())).unwrap();
        assert!(c.stream_to.is_some() && c.hotkey.is_some());
        c.stream_to = None;
        c.hotkey = None;
        c.save(&path).unwrap();

        let reloaded = Config::load(Some(path.clone())).unwrap();
        assert_eq!(reloaded.stream_to, None);
        assert_eq!(reloaded.hotkey, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saving_keeps_the_documentation_in_the_file() {
        // The GUI saves as soon as a source is picked. Serialising the struct
        // would erase every comment on that first save.
        let dir = scratch("save");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, template()).unwrap();

        let mut c: Config = toml::from_str(&template().replace("token = \"\"", "token = \"t\""))
            .unwrap();
        c.source_key = Some("some_source".into());
        c.save(&path).unwrap();

        let after = std::fs::read_to_string(&path).unwrap();
        assert!(after.contains('#'), "comments were lost:\n{after}");
        assert!(after.contains("What to do with the text"), "lost the mode docs");
        assert!(after.contains("some_source"), "the new value was not saved");

        // And it must still load.
        let reloaded = Config::load(Some(path.clone())).unwrap();
        assert_eq!(reloaded.source_key.as_deref(), Some("some_source"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_token_names_the_file_to_edit() {
        let dir = scratch("emptytok");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "token = \"\"\n").unwrap();

        let e = Config::load(Some(path.clone())).unwrap_err();
        let text = format!("{e:#}");
        assert!(text.contains("token"), "got: {text}");
        assert!(text.contains(&path.display().to_string()), "got: {text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_injection_method_parses_from_config() {
        let c: Config =
            toml::from_str("token = \"a\"\ninject = \"ydotool\"").unwrap();
        assert_eq!(c.inject, crate::inject::Method::Ydotool);
    }

    #[test]
    fn mode_parses_from_config() {
        let c: Config = toml::from_str("token = \"a\"\nmode = \"type\"").unwrap();
        assert_eq!(c.mode, OutputMode::Type);
    }

    #[test]
    fn diarize_defaults_to_false() {
        // Zero-config state: no labels requested until asked for.
        let c: Config = toml::from_str(r#"token = "abc""#).unwrap();
        assert!(!c.diarize);
    }

    #[test]
    fn diarize_true_parses() {
        let c: Config = toml::from_str("token = \"a\"\ndiarize = true").unwrap();
        assert!(c.diarize);
    }

    #[test]
    fn the_template_documents_diarize() {
        let text = template();
        assert!(text.contains("diarize = false"), "got:\n{text}");
        assert!(
            text.contains("speaker labels"),
            "the diarize comment is missing:\n{text}"
        );
    }

    #[test]
    fn config_round_trips_through_toml() {
        // The GUI writes this file back when a source is chosen, so a value it
        // cannot re-read would silently lose the setting.
        let c = Config {
            url: "ws://h:1/v1/stream".into(),
            stream_to: Some("~/notes.txt".into()),
            format: crate::save::Format::Timestamped,
            token: "t".into(),
            hotkey: Some("ctrl+alt+d".into()),
            source_key: Some("rnnoise_source".into()),
            mode: OutputMode::Both,
            diarize: true,
            inject: Default::default(),
            waybar_signal: 3,
        };
        let back: Config = toml::from_str(&toml::to_string_pretty(&c).unwrap()).unwrap();
        assert_eq!(back.source_key, c.source_key);
        assert_eq!(back.mode, c.mode);
        assert_eq!(back.waybar_signal, c.waybar_signal);
    }
}
