//! Shared client configuration.
//!
//! One file for every front-end. The CLI and the GUI want the same server and
//! token, and making the user write them twice invites them to diverge.

use crate::mode::OutputMode;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default = "default_url")]
    pub url: String,
    pub token: String,
    /// Remembered source, as a `Source::stable_key`. Node ids change between
    /// runs, so a key is stored rather than an id.
    #[serde(default)]
    pub source_key: Option<String>,
    #[serde(default)]
    pub mode: OutputMode,
    /// How text is typed at the cursor. Electron applications such as Teams
    /// need `ydotool`; see the Method docs.
    #[serde(default)]
    pub inject: crate::inject::Method,
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

        doc["url"] = toml_edit::value(&self.url);
        doc["token"] = toml_edit::value(&self.token);
        doc["mode"] = toml_edit::value(self.mode.name());
        doc["inject"] = toml_edit::value(self.inject.name());
        doc["waybar_signal"] = toml_edit::value(i64::from(self.waybar_signal));
        match &self.source_key {
            Some(k) => doc["source_key"] = toml_edit::value(k),
            None => {
                doc.remove("source_key");
            }
        }

        std::fs::write(path, doc.to_string())
            .with_context(|| format!("writing {}", path.display()))
    }
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
/// Only methods that work on the platform generating the file are listed.
/// Offering a Linux user `sendinput` is noise, and offering a Windows user
/// `wtype` is a wrong answer dressed as a choice.
pub fn template() -> String {
    let mut s = String::new();
    s.push_str(
        "# Syrinx client configuration.\n\
         #\n\
         # Written automatically because no config existed. Every setting below is\n\
         # at its default except `token`, which has no sensible default.\n\
         # Delete any line to go back to the default.\n\n",
    );

    s.push_str("# The server's address.\n");
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

    let usable: Vec<crate::inject::Method> = crate::inject::Method::ALL
        .iter()
        .copied()
        .filter(|m| m.supported_here())
        .collect();
    s.push_str("# How text is typed at the cursor, for the modes above that type.\n");
    for m in &usable {
        s.push_str(&format!("#   \"{}\"{} -- {}\n", m.name(), pad(m.name()), m.summary()));
    }
    s.push_str(&format!(
        "inject = \"{}\"\n\n",
        crate::inject::Method::default().name()
    ));

    s.push_str(
        "# Capture source to use, as printed by `syrinx sources`.\n\
         # Left unset, syrinx asks or uses the default input.\n\
         # source_key = \"...\"\n\n",
    );

    if cfg!(target_os = "linux") {
        s.push_str("# Realtime signal number for the waybar status indicator.\n");
        s.push_str(&format!("waybar_signal = {}\n", default_waybar_signal()));
    }
    s
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

    #[test]
    fn a_minimal_config_applies_defaults() {
        let c: Config = toml::from_str(r#"token = "abc""#).unwrap();
        assert_eq!(c.url, "ws://127.0.0.1:8770/v1/stream");
        assert_eq!(c.mode, OutputMode::Transcribe);
        assert_eq!(c.waybar_signal, 8);
    }

    #[test]
    fn a_missing_token_is_a_hard_error() {
        // Better to refuse than to connect unauthenticated and be rejected
        // with something less clear.
        assert!(toml::from_str::<Config>(r#"url = "ws://x""#).is_err());
    }

    #[test]
    fn a_parse_failure_shows_a_whole_valid_config() {
        // Naming only the field at fault invites replacing the whole file with
        // that one field, which then fails for a different reason. The hint has
        // to be something that works if pasted as-is.
        let path = std::env::temp_dir()
            .join(format!("syrinx-cfg-test-{}.toml", std::process::id()));
        std::fs::write(&path, "inject = \"ydotool\"\n").unwrap();

        let e = Config::load(Some(path.clone())).unwrap_err();
        let _ = std::fs::remove_file(&path);
        let text = format!("{e:#}");
        assert!(text.contains("token"), "should name the field: {text}");
        assert!(text.contains("url ="), "should show a whole config: {text}");

        // The hint itself must parse, or it is worse than no hint.
        let example: String = EXAMPLE_HINT
            .lines()
            .filter(|l| l.starts_with("    ") && l.contains('='))
            .map(|l| format!("{}\n", l.trim()))
            .collect();
        toml::from_str::<Config>(&example)
            .unwrap_or_else(|e| panic!("the suggested config does not parse: {e}\n{example}"));
    }

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
    fn the_template_documents_every_value_that_works_here() {
        let text = template();
        for m in OutputMode::ALL {
            assert!(
                text.contains(&format!("\"{}\"", m.name())),
                "mode {} is undocumented:\n{text}",
                m.name()
            );
        }
        for m in crate::inject::Method::ALL.iter().filter(|m| m.supported_here()) {
            assert!(
                text.contains(&format!("\"{}\"", m.name())),
                "inject {} is undocumented:\n{text}",
                m.name()
            );
        }
    }

    #[test]
    fn the_template_omits_methods_that_cannot_work_here() {
        // Offering a choice that cannot work is worse than offering none.
        let text = template();
        for m in crate::inject::Method::ALL.iter().filter(|m| !m.supported_here()) {
            assert!(
                !text.contains(&format!("inject = \"{}\"", m.name())),
                "{} is offered as the default but cannot work here",
                m.name()
            );
        }
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
    fn config_round_trips_through_toml() {
        // The GUI writes this file back when a source is chosen, so a value it
        // cannot re-read would silently lose the setting.
        let c = Config {
            url: "ws://h:1/v1/stream".into(),
            token: "t".into(),
            source_key: Some("rnnoise_source".into()),
            mode: OutputMode::Both,
            inject: Default::default(),
            waybar_signal: 3,
        };
        let back: Config = toml::from_str(&toml::to_string_pretty(&c).unwrap()).unwrap();
        assert_eq!(back.source_key, c.source_key);
        assert_eq!(back.mode, c.mode);
        assert_eq!(back.waybar_signal, c.waybar_signal);
    }
}
