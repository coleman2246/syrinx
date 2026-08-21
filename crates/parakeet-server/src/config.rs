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

    /// Which execution provider to use. `cuda` verifies at startup that
    /// inference is actually running on the GPU and refuses to start if it is
    /// not, rather than silently serving at CPU speed.
    #[serde(default)]
    pub provider: Provider,

    /// Expected VRAM footprint in MiB, used for the pre-load check. Measured at
    /// 3400 for Nemotron on CUDA: a 2515 MB model file plus ~900 MB ORT arena.
    #[serde(default = "default_model_mib")]
    pub model_mib: u64,

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

/// Execution provider. Defaults to CPU: it needs no GPU, cannot disturb other
/// tenants, and at ~149ms per 560ms chunk it still keeps up in real time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Cpu,
    Cuda,
    /// Deterministic stub. Testing only.
    Mock,
}

fn default_model_mib() -> u64 {
    3400
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
    fn provider_defaults_to_cpu_not_gpu() {
        // Defaulting to CUDA would mean a misconfigured server competes for a
        // contended GPU by accident.
        let c = Config::from_toml("token = \"a\"\nmodel_dir = \"/m\"").unwrap();
        assert_eq!(c.provider, Provider::Cpu);
        assert_eq!(c.model_mib, 3400);
    }

    #[test]
    fn provider_parses_from_config() {
        let c = Config::from_toml("token = \"a\"\nmodel_dir = \"/m\"\nprovider = \"cuda\"").unwrap();
        assert_eq!(c.provider, Provider::Cuda);
    }

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
