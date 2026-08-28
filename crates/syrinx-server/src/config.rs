//! Server configuration.

use anyhow::ensure;
use serde::Deserialize;
use std::ops::RangeInclusive;

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

    /// Directory holding the diarization models. Absent = speaker labels
    /// unavailable, feature off.
    ///
    /// Two files are expected in it: `silero_vad.onnx` (v6.2; the v5-era
    /// 512+64-sample interface, unchanged since) and
    /// `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`, both from the
    /// sherpa-onnx model zoo. Another embedding model is accepted if its
    /// filename identifies its family -- `wespeaker`, `3dspeaker`/`eres2net`,
    /// `nemo`/`titanet` -- because feature normalisation is a property of the
    /// training recipe that the ONNX file does not record. Anything else, or
    /// more than one candidate, turns the feature off with an error in the log
    /// rather than guessing.
    #[serde(default)]
    pub diarize_model_dir: Option<String>,

    /// How many 560 ms chunks a commit is held back so its speaker label can
    /// catch up, since the diarizer needs more audio than the transducer does.
    ///
    /// 2 is ~1.12 s, which covers the p90 label delay the spike measured at
    /// 0.88-1.09 s; one chunk (0.56 s) does not. Lowering it buys that latency
    /// back and pays for it in fragments that arrive unlabelled and turn
    /// starts attributed to whoever was talking before -- 0 emits every commit
    /// the moment its text exists, labelled from its own chunk alone. Raising
    /// it to 3 (1.68 s) covers the p99 and buys nothing measurable beyond.
    ///
    /// Accepted range 0-16. See "Spike results" in
    /// `docs/specs/2026-08-24-speaker-diarization-design.md`, which calls this
    /// depth a tunable rather than a fixed constant.
    #[serde(default = "default_diarize_lag_chunks")]
    pub diarize_lag_chunks: usize,

    /// How many mutually agreeing 1.5 s windows it takes to mint a new
    /// speaker. At a 0.75 s hop, 4 is ~3.7 s of one voice before it is given a
    /// number.
    ///
    /// Lowering it picks a new speaker up faster -- the fragile case is the
    /// participant who says three sentences in an hour -- and pays by
    /// splitting one person across several labels, which is the failure this
    /// rule exists to prevent. The spike found 4 the only value correct on all
    /// three annotated meetings: 3 produced 6 labels for the 5 speakers of the
    /// 87-minute meeting, and 2 produced 20 labels for a 4-speaker one.
    ///
    /// Accepted range 2-16; a pool of one window has nothing to agree with.
    /// See "Spike results" in
    /// `docs/specs/2026-08-24-speaker-diarization-design.md`.
    #[serde(default = "default_diarize_min_pool")]
    pub diarize_min_pool: usize,

    /// How far the nearest centroid must beat the second nearest before a
    /// window is attributed to it.
    ///
    /// The crowding tunable. An argmax against a fixed threshold decays as
    /// centroids are added -- with five speakers the spike already measured
    /// two live centroids at 0.519, above `T_assign` -- so a genuinely new
    /// voice increasingly finds an incumbent over the bar to be handed to.
    /// Requiring a lead as well is the question that does not rot.
    ///
    /// Raising it labels less and guesses less, and does nothing else --
    /// minting has its own key below, because it is a different question on a
    /// different scale. 0 is the rule the server used before 2026-08-27, and
    /// the retreat if corrections and gaps turn out to read worse than the
    /// occasional wrong name; at 0, and only at 0, this key does govern the
    /// mint ceiling as well, because the hatch is documented as switching off
    /// everything that arrived with it. Accepted range 0 to 0.5.
    ///
    /// **The default is an engineering estimate, not a measurement.** See
    /// `docs/specs/2026-08-27-diarization-latency-and-crowding-design.md`; the
    /// probe's live-emulation mode exists to replace it.
    #[serde(default = "default_diarize_margin")]
    pub diarize_margin: f32,

    /// How close to a speaker who already has a number a group of agreeing
    /// windows may sit and still be given a number of its own.
    ///
    /// The split guard. Four windows that agree with each other but with
    /// nobody known are either a new voice or one known voice recorded badly,
    /// and the cosine between their average and the nearest existing speaker
    /// is what separates the two: past this value they are that speaker, and
    /// minting would split one person across two labels -- a label the
    /// duplicate carries text away under and does not give back, since
    /// retirement folds the centroid and cannot unfreeze a commit.
    ///
    /// Raising it mints more readily in a crowded room, at the risk of
    /// splitting; lowering it mints less, and a genuinely new voice that is
    /// refused gets an incumbent's number as a guess instead of a gap.
    ///
    /// **Not the same quantity as `diarize_margin`,** although it used to be
    /// derived from it. That one is a lead between one 1.5 s window's cosines,
    /// measured against the spike's same-speaker median of 0.517; this one is
    /// a cosine between an *average* of `diarize_min_pool` windows and a
    /// centroid, which for a single voice sits much higher. Accepted range 0
    /// to 1, a cosine's own range: at or below `T_assign` this clause is inert
    /// and the pre-2026-08-27 mint rule is all that is left, and from
    /// `T_retire` upwards the retirement threshold caps it, so both ends
    /// saturate into settings rather than into nonsense.
    ///
    /// **The default is an engineering estimate, not a measurement**, and the
    /// design names it the single most important number the probe has to
    /// answer.
    #[serde(default = "default_diarize_mint_ceiling")]
    pub diarize_mint_ceiling: f32,

    /// How far the cosine between two consecutive 0.75 s hop embeddings must
    /// fall before the diarizer calls it a turn change.
    ///
    /// A detected change flushes the 1.5 s accumulator at the boundary, so the
    /// next full window is one voice rather than two; stops the previous
    /// speaker's label being carried forward over the new speaker's opening
    /// sentence; and stops the commit vote reaching across the boundary. It is
    /// the only turn-change detector in the pipeline -- `MAX_GAP_FRAMES` is
    /// deliberately not one, and `window.rs` records the measurement saying so.
    ///
    /// A cosine, so 0 to 1. At 0 nothing is ever a turn change and the
    /// behaviour is what the server did before 2026-08-27; at 1 every hop is,
    /// and no window ever completes.
    ///
    /// **The default is an engineering estimate, not a measurement.**
    #[serde(default = "default_diarize_change_threshold")]
    pub diarize_change_threshold: f32,

    /// How many seconds of already-committed transcript stay eligible to have
    /// their speaker corrected.
    ///
    /// A voice needs four agreeing windows before it is minted, so the opening
    /// of a meeting is committed before anybody can be named for it. Within
    /// this window the server sends `transcript.relabel` when the name
    /// arrives, and the GUI and Save-as pick it up; the streamed file never
    /// does, because it is append-only on purpose.
    ///
    /// 0 turns corrections off entirely, which is the retreat if text
    /// acquiring a speaker name a few seconds late reads worse than a gap.
    /// Accepted range 0 to 600 seconds.
    ///
    /// **The default is an engineering estimate, not a measurement.**
    #[serde(default = "default_diarize_relabel_window")]
    pub diarize_relabel_window: u64,
}

/// What [`Config::diarize_lag_chunks`] will accept. The top is ~9 s, five
/// times the p99 label delay the spike measured, so every experiment worth
/// running fits inside it; past it the buffer is holding a paragraph of
/// transcript that a dropped connection would take with it.
const DIARIZE_LAG_CHUNKS: RangeInclusive<usize> = 0..=16;

/// What [`Config::diarize_min_pool`] will accept. One window cannot agree with
/// itself, so 2 is where "agreeing windows" starts meaning anything. The top
/// is ~12 s of one uninterrupted voice before a speaker is minted, by which
/// point a meeting labels nobody at all.
const DIARIZE_MIN_POOL: RangeInclusive<usize> = 2..=16;

/// What [`Config::diarize_margin`] will accept. The top is half a cosine,
/// which is already past useful: same-speaker windows sit at a median of 0.52
/// and different-speaker at 0.046, so a lead of more than about half is a
/// server that assigns nothing while looking perfectly healthy -- the quiet
/// failure the spike documented when `T_assign` was raised instead.
const DIARIZE_MARGIN: RangeInclusive<f32> = 0.0..=0.5;

/// What [`Config::diarize_mint_ceiling`] will accept: a cosine's whole range,
/// because unlike the margin this one has no interior value that stops meaning
/// anything. Both ends saturate into settings instead -- at or below `T_assign`
/// the ceiling clause is inert and minting is the pre-2026-08-27 rule alone,
/// and from `T_retire` up the retirement threshold is the cap, so the far end
/// says "let retirement decide" rather than "mint anything". Refusing the ends
/// would make an operator work out where the useful band stops from constants
/// no config file names.
const DIARIZE_MINT_CEILING: RangeInclusive<f32> = 0.0..=1.0;

/// What [`Config::diarize_change_threshold`] will accept: a cosine's range,
/// both ends meaningful. 0 detects no turn change ever; 1 would detect one at
/// every hop, which the window assembler's refractory floor then holds down to
/// one per 0.768 s of voiced audio so that windows keep completing.
const DIARIZE_CHANGE_THRESHOLD: RangeInclusive<f32> = 0.0..=1.0;

/// What [`Config::diarize_relabel_window`] will accept, in seconds. The top is
/// ten minutes, past which the session is holding a ring of commits nobody is
/// still looking at and the client is repainting scrollback.
const DIARIZE_RELABEL_WINDOW: RangeInclusive<u64> = 0..=600;

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
/// The calibrated values are the ones the code already names, read from where
/// their justification lives rather than copied here: a second literal would
/// eventually disagree with the comment explaining it.
fn default_diarize_lag_chunks() -> usize {
    crate::session::LAG_CHUNKS
}
fn default_diarize_min_pool() -> usize {
    crate::diarize::cluster::MIN_POOL
}
fn default_diarize_margin() -> f32 {
    crate::diarize::cluster::T_MARGIN
}
fn default_diarize_mint_ceiling() -> f32 {
    crate::diarize::cluster::T_MINT_CEILING
}
fn default_diarize_change_threshold() -> f32 {
    crate::diarize::cluster::T_CHANGE
}
fn default_diarize_relabel_window() -> u64 {
    crate::session::RELABEL_WINDOW
}

impl Config {
    pub fn from_toml(s: &str) -> anyhow::Result<Self> {
        let cfg: Self = toml::from_str(s)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Refuse settings that parse but cannot work, naming the key and what it
    /// accepts.
    ///
    /// Only the diarization tunables need this. The other numbers here are
    /// either checked where they are used or have no wrong value -- a small
    /// `max_sessions` is a cautious server, not a broken one. These are
    /// different: a pool of 1 mints a speaker from a single window, which is
    /// the one outcome the clustering design exists to prevent; a lag of 200
    /// holds two minutes of transcript hostage to a label; a margin of 0.9
    /// labels nobody at all. Every one of them looks exactly like a working
    /// server until a meeting is under way, so the load is where they have to
    /// fail.
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            DIARIZE_LAG_CHUNKS.contains(&self.diarize_lag_chunks),
            "diarize_lag_chunks = {} is out of range: it must be {} to {}, in chunks \
             of 560 ms. 0 emits each label as it comes; the calibrated value is {}.",
            self.diarize_lag_chunks,
            DIARIZE_LAG_CHUNKS.start(),
            DIARIZE_LAG_CHUNKS.end(),
            default_diarize_lag_chunks(),
        );
        ensure!(
            DIARIZE_MIN_POOL.contains(&self.diarize_min_pool),
            "diarize_min_pool = {} is out of range: it must be {} to {} windows. \
             Below {} there is no second window for the first to agree with; the \
             calibrated value is {}.",
            self.diarize_min_pool,
            DIARIZE_MIN_POOL.start(),
            DIARIZE_MIN_POOL.end(),
            DIARIZE_MIN_POOL.start(),
            default_diarize_min_pool(),
        );
        ensure!(
            DIARIZE_MARGIN.contains(&self.diarize_margin),
            "diarize_margin = {} is out of range: it must be {} to {}, in cosine. \
             0 turns off the margin, the cohort test and the mint ceiling together, \
             which is what the server did before speaker crowding was addressed; \
             the shipped value is {}.",
            self.diarize_margin,
            DIARIZE_MARGIN.start(),
            DIARIZE_MARGIN.end(),
            default_diarize_margin(),
        );
        ensure!(
            DIARIZE_MINT_CEILING.contains(&self.diarize_mint_ceiling),
            "diarize_mint_ceiling = {} is out of range: it must be {} to {}, in \
             cosine between a pool's mean and the nearest existing speaker. This \
             is not the assignment margin and is not on its scale; the shipped \
             value is {}.",
            self.diarize_mint_ceiling,
            DIARIZE_MINT_CEILING.start(),
            DIARIZE_MINT_CEILING.end(),
            default_diarize_mint_ceiling(),
        );
        ensure!(
            DIARIZE_CHANGE_THRESHOLD.contains(&self.diarize_change_threshold),
            "diarize_change_threshold = {} is out of range: it must be {} to {}, \
             in cosine between consecutive 0.75 s hops. 0 detects no turn change \
             ever, and is checked for rather than compared against, since \
             different-speaker cosines are often negative; the shipped value is {}.",
            self.diarize_change_threshold,
            DIARIZE_CHANGE_THRESHOLD.start(),
            DIARIZE_CHANGE_THRESHOLD.end(),
            default_diarize_change_threshold(),
        );
        ensure!(
            DIARIZE_RELABEL_WINDOW.contains(&self.diarize_relabel_window),
            "diarize_relabel_window = {} is out of range: it must be {} to {} \
             seconds. 0 turns speaker corrections off entirely; the shipped \
             value is {}.",
            self.diarize_relabel_window,
            DIARIZE_RELABEL_WINDOW.start(),
            DIARIZE_RELABEL_WINDOW.end(),
            default_diarize_relabel_window(),
        );
        Ok(())
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
        let before = (
            c.max_sessions,
            c.vram_floor_mib,
            c.idle_unload_secs,
            c.diarize_model_dir.clone(),
            c.diarize_lag_chunks,
            c.diarize_min_pool,
        );
        c.apply_env(|_| Some("999".into()));
        assert_eq!(
            (
                c.max_sessions,
                c.vram_floor_mib,
                c.idle_unload_secs,
                c.diarize_model_dir.clone(),
                c.diarize_lag_chunks,
                c.diarize_min_pool,
            ),
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

    #[test]
    fn diarize_model_dir_defaults_to_absent() {
        // Absent means the feature is off, which must be the zero-config state.
        assert!(base().diarize_model_dir.is_none());
    }

    #[test]
    fn diarize_model_dir_parses() {
        let c = Config::from_toml(
            "token = \"a\"\nmodel_dir = \"/m\"\ndiarize_model_dir = \"/d\"",
        )
        .unwrap();
        assert_eq!(c.diarize_model_dir.as_deref(), Some("/d"));
    }

    /// `base()` plus one line, for the tunables.
    fn with(line: &str) -> anyhow::Result<Config> {
        Config::from_toml(&format!("token = \"a\"\nmodel_dir = \"/m\"\n{line}"))
    }

    #[test]
    fn the_diarize_tunables_default_to_the_calibrated_values() {
        // A config that says nothing about them must behave exactly as the
        // server did before they existed, which is what makes them safe to add
        // to a deployment that has already been tuned by hand.
        let c = base();
        assert_eq!(c.diarize_lag_chunks, 2);
        assert_eq!(c.diarize_min_pool, 4);
        assert_eq!(c.diarize_lag_chunks, crate::session::LAG_CHUNKS);
        assert_eq!(c.diarize_min_pool, crate::diarize::cluster::MIN_POOL);
    }

    #[test]
    fn the_diarize_tunables_parse() {
        let c = with("diarize_lag_chunks = 1\ndiarize_min_pool = 3").unwrap();
        assert_eq!(c.diarize_lag_chunks, 1);
        assert_eq!(c.diarize_min_pool, 3);
    }

    #[test]
    fn a_lag_of_zero_is_a_setting_and_not_a_mistake() {
        // The bottom of the range is meaningful: no wait at all, every commit
        // labelled from its own chunk. Rejecting it as "off" would remove the
        // one setting that answers "how much of this delay is the diarizer?".
        assert_eq!(
            with("diarize_lag_chunks = 0").unwrap().diarize_lag_chunks,
            0
        );
        assert_eq!(
            with("diarize_lag_chunks = 16").unwrap().diarize_lag_chunks,
            16
        );
    }

    #[test]
    fn an_absurd_lag_is_refused_at_load_by_name() {
        // Nine seconds of held transcript is not a trade-off anyone is making
        // on purpose, and a server that accepted it would look healthy while
        // the client waited.
        let err = with("diarize_lag_chunks = 17").expect_err("above the range");
        let message = format!("{err:#}");
        assert!(message.contains("diarize_lag_chunks"), "{message}");
        assert!(message.contains("0 to 16"), "{message}");
    }

    #[test]
    fn a_pool_below_two_cannot_express_agreement() {
        // "Windows that agree with each other" needs two windows. At 1 every
        // stray cough mints a speaker, which is the one failure the clustering
        // rules were built around.
        let err = with("diarize_min_pool = 1").expect_err("below the range");
        let message = format!("{err:#}");
        assert!(message.contains("diarize_min_pool"), "{message}");
        assert!(message.contains("2 to 16"), "{message}");

        assert_eq!(with("diarize_min_pool = 2").unwrap().diarize_min_pool, 2);
        assert_eq!(with("diarize_min_pool = 16").unwrap().diarize_min_pool, 16);
    }

    #[test]
    fn an_absurd_pool_is_refused_at_load_by_name() {
        let err = with("diarize_min_pool = 17").expect_err("above the range");
        let message = format!("{err:#}");
        assert!(message.contains("diarize_min_pool"), "{message}");
        assert!(message.contains("2 to 16"), "{message}");
    }

    #[test]
    fn the_latency_tunables_default_to_what_the_code_names() {
        // Same rule as the two before them: a config that says nothing about
        // these must behave exactly as the server did before they existed.
        let c = base();
        assert_eq!(c.diarize_margin, crate::diarize::cluster::T_MARGIN);
        assert_eq!(
            c.diarize_mint_ceiling,
            crate::diarize::cluster::T_MINT_CEILING
        );
        assert_eq!(
            c.diarize_change_threshold,
            crate::diarize::cluster::T_CHANGE
        );
        assert_eq!(c.diarize_relabel_window, crate::session::RELABEL_WINDOW);
    }

    #[test]
    fn the_latency_tunables_parse() {
        let c = with(
            "diarize_margin = 0.2\ndiarize_mint_ceiling = 0.62\n\
             diarize_change_threshold = 0.5\ndiarize_relabel_window = 10",
        )
        .unwrap();
        assert_eq!(c.diarize_margin, 0.2);
        assert_eq!(c.diarize_mint_ceiling, 0.62);
        assert_eq!(c.diarize_change_threshold, 0.5);
        assert_eq!(c.diarize_relabel_window, 10);
    }

    #[test]
    fn a_margin_of_zero_is_the_documented_way_back_to_the_old_rule() {
        // The bottom of the range is meaningful: assign to the nearest
        // centroid over the threshold and never mind the runner-up, which is
        // exactly what the server did before 2026-08-27. It is the retreat if
        // withholding turns out to cost more than it saves, so it has to be
        // accepted rather than read as "unset".
        //
        // That the value *reaches* the clusterer and switches off all three of
        // the 2026-08-27 rules is `cluster.rs`'s business, and the tests there
        // say so exhaustively -- a switch nobody checks the far end of is how
        // a documented hatch stops working. The mint ceiling is the one that
        // needs saying out loud now that it has a key of its own: nothing
        // about the arithmetic switches it off any more.
        assert_eq!(with("diarize_margin = 0.0").unwrap().diarize_margin, 0.0);
    }

    #[test]
    fn a_relabel_window_of_zero_turns_corrections_off() {
        // The escape hatch the design names: if text acquiring a speaker name
        // a few seconds late reads worse than a gap, this is how a deployment
        // says so.
        assert_eq!(
            with("diarize_relabel_window = 0")
                .unwrap()
                .diarize_relabel_window,
            0
        );
    }

    #[test]
    fn a_margin_nothing_could_clear_is_refused_at_load_by_name() {
        // Cosines to two different centroids cannot differ by more than about
        // half in any configuration that assigns anything at all, so a margin
        // above that is a server that labels nobody while looking healthy --
        // the same quiet failure the spike measured when T_assign was raised.
        let err = with("diarize_margin = 0.6").expect_err("above the range");
        let message = format!("{err:#}");
        assert!(message.contains("diarize_margin"), "{message}");
        assert!(message.contains("0 to 0.5"), "{message}");
        assert!(with("diarize_margin = -0.1").is_err(), "a negative margin");
    }

    #[test]
    fn a_mint_ceiling_outside_a_cosine_is_refused_at_load_by_name() {
        // The whole of a cosine is accepted because both ends saturate into
        // settings: at or below T_assign the ceiling clause is inert, and from
        // T_retire up the retirement threshold is the cap. Outside that there
        // is no number left to mean anything.
        let err = with("diarize_mint_ceiling = 1.1").expect_err("above the range");
        let message = format!("{err:#}");
        assert!(message.contains("diarize_mint_ceiling"), "{message}");
        assert!(message.contains("0 to 1"), "{message}");
        assert!(with("diarize_mint_ceiling = -0.1").is_err(), "negative");
        assert!(with("diarize_mint_ceiling = 0.0").is_ok());
        assert!(with("diarize_mint_ceiling = 1.0").is_ok());
    }

    #[test]
    fn a_non_finite_mint_ceiling_is_refused_rather_than_swallowed_by_the_cap() {
        // TOML spells all three of these, and the range check is the only
        // thing between them and the clusterer.
        //
        // A NaN ceiling is never compared at all, which is why it is the
        // dangerous one. The gate reads `mint_ceiling.min(t_retire)`, and
        // `f32::min` discards a NaN operand, so the cap alone would decide:
        // the loosest setting the key can reach, with the split guard off,
        // rather than the "refuse every mint" a `<` against NaN would give.
        // `inf` folds to the same cap; only `-inf` refuses everything.
        // `contains` is false for all three, which is what makes the load the
        // place it fails, and
        // `cluster::tests::a_ceiling_of_nan_is_the_loosest_setting_and_not_the_strictest`
        // is what holds the behaviour this rests on.
        for value in ["nan", "inf", "-inf"] {
            assert!(
                with(&format!("diarize_mint_ceiling = {value}")).is_err(),
                "{value} was accepted"
            );
        }
    }

    #[test]
    fn an_out_of_range_change_threshold_is_refused_at_load_by_name() {
        // A cosine, so the range is a cosine's. Both ends are settings rather
        // than mistakes: 0 detects nothing, which `real::diarizer` checks for
        // rather than reaching by arithmetic, and 1 detects a change at every
        // hop, which the window assembler's refractory floor keeps from
        // starving the clusterer of windows.
        let err = with("diarize_change_threshold = 1.5").expect_err("above the range");
        let message = format!("{err:#}");
        assert!(message.contains("diarize_change_threshold"), "{message}");
        assert!(message.contains("0 to 1"), "{message}");
        assert!(with("diarize_change_threshold = 0.0").is_ok());
        assert!(with("diarize_change_threshold = 1.0").is_ok());
    }

    #[test]
    fn an_absurd_relabel_window_is_refused_at_load_by_name() {
        // Ten minutes of transcript still open to correction is a client
        // repainting text nobody is looking at any more, and a ring of
        // commits the session holds for as long.
        let err = with("diarize_relabel_window = 601").expect_err("above the range");
        let message = format!("{err:#}");
        assert!(message.contains("diarize_relabel_window"), "{message}");
        assert!(message.contains("0 to 600"), "{message}");
    }

    #[test]
    fn the_latency_tunables_are_not_overridable_either() {
        // They describe how the service behaves, like everything else that is
        // not a token, a bind address or a model path.
        let mut c = base();
        let before = (
            c.diarize_margin,
            c.diarize_mint_ceiling,
            c.diarize_change_threshold,
            c.diarize_relabel_window,
        );
        c.apply_env(|_| Some("999".into()));
        assert_eq!(
            (
                c.diarize_margin,
                c.diarize_mint_ceiling,
                c.diarize_change_threshold,
                c.diarize_relabel_window,
            ),
            before
        );
    }
}
