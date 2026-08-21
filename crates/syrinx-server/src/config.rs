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

    /// Load from TOML, then let the environment override.
    ///
    /// A container needs its token from outside the image. Baking one into a
    /// layer publishes it to anyone who can pull the image, and committing one
    /// to a config file publishes it to anyone who can read the repository, so
    /// the deployment path has to be able to supply it at run time.
    ///
    /// Only the settings that vary per deployment are overridable. Everything
    /// else describes how the service behaves, which belongs in a file that can
    /// be reviewed rather than in an environment nobody can see.
    pub fn from_toml_and_env(s: &str) -> anyhow::Result<Self> {
        let mut cfg = Self::from_toml(s)?;
        cfg.apply_env(|k| std::env::var(k).ok());
        Ok(cfg)
    }

    /// The override step, with the environment injected so it can be tested.
    ///
    /// Reading the real environment in a test is not safe: `set_var` is unsound
    /// with other threads running, and the test harness is threaded.
    pub fn apply_env(&mut self, get: impl Fn(&str) -> Option<String>) {
        if let Some(v) = get("SYRINX_TOKEN").filter(|v| !v.is_empty()) {
            self.token = v;
        }
        if let Some(v) = get("SYRINX_BIND").filter(|v| !v.is_empty()) {
            self.bind = v;
        }
        if let Some(v) = get("SYRINX_MODEL_DIR").filter(|v| !v.is_empty()) {
            self.model_dir = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Config {
        Config::from_toml("token = \"from-file\"\nmodel_dir = \"/models\"").unwrap()
    }

    #[test]
    fn the_environment_supplies_the_token() {
        // The reason this exists: a token must not be baked into an image or
        // committed to a config file.
        let mut c = base();
        c.apply_env(|k| (k == "SYRINX_TOKEN").then(|| "from-env".to_string()));
        assert_eq!(c.token, "from-env");
    }

    #[test]
    fn an_empty_override_does_not_blank_a_setting() {
        // An unset variable in a compose file arrives as "". Taking that
        // literally would replace a working token with one that fails closed,
        // and the failure would look like a client problem.
        let mut c = base();
        c.apply_env(|k| (k == "SYRINX_TOKEN").then(String::new));
        assert_eq!(c.token, "from-file");
    }

    #[test]
    fn the_file_wins_when_nothing_is_set() {
        let mut c = base();
        c.apply_env(|_| None);
        assert_eq!(c.token, "from-file");
        assert_eq!(c.model_dir, "/models");
    }

    #[test]
    fn bind_and_model_dir_are_overridable() {
        // The two other things that differ between a desktop and a container.
        let mut c = base();
        c.apply_env(|k| match k {
            "SYRINX_BIND" => Some("0.0.0.0:9000".into()),
            "SYRINX_MODEL_DIR" => Some("/mnt/models".into()),
            _ => None,
        });
        assert_eq!(c.bind, "0.0.0.0:9000");
        assert_eq!(c.model_dir, "/mnt/models");
    }

    #[test]
    fn behaviour_settings_are_not_overridable() {
        // VRAM floors and session caps describe how the service behaves under
        // pressure. They belong in a file that can be reviewed, not in an
        // environment nobody can see.
        let mut c = base();
        let before = (c.max_sessions, c.vram_floor_mib, c.idle_unload_secs);
        c.apply_env(|_| Some("999".into()));
        assert_eq!(
            (c.max_sessions, c.vram_floor_mib, c.idle_unload_secs),
            before
        );
    }

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
