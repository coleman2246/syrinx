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
        let candidates = match explicit {
            Some(p) => vec![p],
            None => vec![
                Self::default_path(),
                config_base().join("parakeet/config.toml"),
                config_base().join("parakeet-type/config.toml"),
                config_base().join("syrinx-gui/config.toml"),
            ],
        };
        for p in &candidates {
            match std::fs::read_to_string(p) {
                Ok(text) => {
                    // A parse failure is nearly always a missing required
                    // field, and naming it without showing a whole valid file
                    // invites replacing the file with just that field.
                    return toml::from_str(&text).map_err(|e| {
                        anyhow::anyhow!("{}\n\n{}", e, EXAMPLE_HINT).context(format!(
                            "the config at {} is not valid",
                            p.display()
                        ))
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e).with_context(|| format!("reading {}", p.display())),
            }
        }
        anyhow::bail!(
            "no config found at {}.\n\n{}",
            Self::default_path().display(),
            EXAMPLE_HINT
        )
    }

    pub fn save(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, toml::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }
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

fn config_base() -> PathBuf {
    std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".config")
        })
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
