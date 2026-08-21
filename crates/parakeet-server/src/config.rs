//! Server configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_bind")]
    pub bind: String,

    /// Shared bearer token. Required: an empty token fails every request
    /// rather than opening the service.
    pub token: String,

    pub model_dir: String,

    /// Unload models after this many seconds with no active session. The server
    /// should hold zero VRAM for most of the day.
    #[serde(default = "default_idle_unload")]
    pub idle_unload_secs: u64,

    /// Refuse to load a model if doing so would leave less than this much VRAM
    /// (MiB) free for other tenants on the GPU.
    #[serde(default = "default_vram_floor")]
    pub vram_floor_mib: u64,

    /// Maximum concurrent streaming sessions. Sessions beyond this are refused
    /// with `capacity` rather than degrading everyone already connected.
    #[serde(default = "default_max_sessions")]
    pub max_sessions: usize,
}

fn default_bind() -> String {
    "0.0.0.0:8770".into()
}
fn default_idle_unload() -> u64 {
    600
}
fn default_vram_floor() -> u64 {
    1536
}
fn default_max_sessions() -> usize {
    4
}

impl Config {
    pub fn from_toml(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_config_applies_defaults() {
        let c = Config::from_toml(
            r#"
            token = "abc"
            model_dir = "/models"
        "#,
        )
        .unwrap();
        assert_eq!(c.bind, "0.0.0.0:8770");
        assert_eq!(c.idle_unload_secs, 600);
        assert_eq!(c.vram_floor_mib, 1536);
        assert_eq!(c.max_sessions, 4);
    }

    #[test]
    fn missing_token_is_a_hard_error() {
        // Better to refuse to start than to start unauthenticated.
        assert!(Config::from_toml(r#"model_dir = "/models""#).is_err());
    }

    #[test]
    fn values_override_defaults() {
        let c = Config::from_toml(
            r#"
            token = "abc"
            model_dir = "/models"
            max_sessions = 9
            vram_floor_mib = 2048
        "#,
        )
        .unwrap();
        assert_eq!(c.max_sessions, 9);
        assert_eq!(c.vram_floor_mib, 2048);
    }
}
