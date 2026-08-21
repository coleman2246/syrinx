//! Everything a parakeet front-end needs: config, sources, sessions, output.
//!
//! Exists so the CLI and the GUI are thin. Both previously carried their own
//! near-identical session loop, which meant every feature -- modes, source
//! selection, status -- had to be built twice and could drift between them.

pub mod config;
pub mod inject;
pub mod mode;
pub mod session;
pub mod state;

pub use config::Config;
pub use mode::OutputMode;
pub use session::{SessionHandle, SessionOptions, SessionState, Status};

pub use parakeet_audio::{Source, SourceKind, list_sources, resolve};

use anyhow::{Context, Result};

/// Pick the source to use: the remembered one if it still exists, otherwise a
/// sensible default.
///
/// Prefers a noise-suppressed microphone where one exists -- on a setup with
/// an RNNoise virtual source that is the same hardware with a far lower noise
/// floor. Never defaults to system audio: silently recording the speakers
/// because they sorted first would be a surprising thing to do unasked.
pub fn choose_source(sources: &[Source], remembered: Option<&str>) -> Result<Source> {
    if let Some(key) = remembered
        && let Some(s) = resolve(sources, key)
    {
        return Ok(s);
    }
    sources
        .iter()
        .find(|s| s.kind == SourceKind::Microphone && s.stable_key().contains("rnnoise"))
        .or_else(|| sources.iter().find(|s| s.kind == SourceKind::Microphone))
        .or_else(|| sources.first())
        .cloned()
        .context("no capturable audio sources found")
}

#[cfg(test)]
mod tests {
    use super::*;
    use parakeet_audio::SourceTarget;

    fn src(name: &str, kind: SourceKind, key: &str) -> Source {
        Source {
            target: SourceTarget::PipeWireNode(1),
            name: name.into(),
            kind,
            detail: None,
            stable_name: Some(key.into()),
        }
    }

    #[test]
    fn a_remembered_source_wins() {
        let list = vec![
            src("Yeti (RNNoise)", SourceKind::Microphone, "rnnoise_source"),
            src("Webcam", SourceKind::Microphone, "cam"),
        ];
        assert_eq!(choose_source(&list, Some("cam")).unwrap().name, "Webcam");
    }

    #[test]
    fn a_stale_remembered_key_falls_back_instead_of_failing() {
        // Devices come and go; a saved choice that no longer exists must not
        // stop the session starting.
        let list = vec![src("Webcam", SourceKind::Microphone, "cam")];
        assert_eq!(choose_source(&list, Some("gone")).unwrap().name, "Webcam");
    }

    #[test]
    fn a_denoised_microphone_is_preferred_over_the_raw_device() {
        let list = vec![
            src("Blue Microphones", SourceKind::Microphone, "alsa_input.blue"),
            src("Yeti (RNNoise)", SourceKind::Microphone, "rnnoise_source"),
        ];
        assert_eq!(
            choose_source(&list, None).unwrap().name,
            "Yeti (RNNoise)",
            "should prefer the denoised input over the same raw hardware"
        );
    }

    #[test]
    fn system_audio_is_never_chosen_by_default() {
        // Recording the speakers unasked would be a surprising default.
        let list = vec![
            src("Monitor of Speakers", SourceKind::Monitor, "mon"),
            src("Webcam", SourceKind::Microphone, "cam"),
        ];
        assert_eq!(choose_source(&list, None).unwrap().kind, SourceKind::Microphone);
    }

    #[test]
    fn with_only_monitors_available_one_is_still_usable() {
        // A machine with no microphone should still be able to transcribe
        // system audio rather than refusing outright.
        let list = vec![src("Monitor of Speakers", SourceKind::Monitor, "mon")];
        assert_eq!(choose_source(&list, None).unwrap().kind, SourceKind::Monitor);
    }

    #[test]
    fn an_empty_source_list_is_an_error() {
        assert!(choose_source(&[], None).is_err());
    }
}
