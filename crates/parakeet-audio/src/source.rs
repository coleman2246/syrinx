//! Enumerating capturable audio sources from the PipeWire graph.
//!
//! Deliberately not cpal. cpal's ALSA host reports raw ALSA devices and plugin
//! names ("Rate Converter Plugin Using Speex Resampler"), shows no monitor
//! sources at all, and cannot see per-application streams. Everything
//! interesting here -- transcribing system audio, or transcribing Firefox
//! specifically while it keeps playing to the speakers -- lives in the PipeWire
//! graph, which ALSA does not model.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// What kind of thing is being captured. Drives grouping in a picker, and
/// explains to the user what they are about to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    /// A real capture device: microphone, line in.
    Microphone,
    /// The output of a sink -- everything playing through that device, mixed.
    Monitor,
    /// A single application's playback stream, e.g. Firefox or Spotify.
    Application,
}

impl SourceKind {
    pub fn label(self) -> &'static str {
        match self {
            SourceKind::Microphone => "Microphone",
            SourceKind::Monitor => "System audio",
            SourceKind::Application => "Application",
        }
    }
}

/// A capturable node in the PipeWire graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// PipeWire node id. This is what capture targets.
    ///
    /// Not stable across restarts: an application's node id changes every time
    /// it restarts, and devices are renumbered on replug. Persist
    /// [`Self::stable_key`] instead and re-resolve on startup.
    pub id: u32,
    /// Human-readable name for a picker.
    pub name: String,
    pub kind: SourceKind,
    /// PipeWire `node.name`, stable for devices though not for applications.
    pub node_name: Option<String>,
    /// What the application is playing, when it says. Firefox reports the tab
    /// title, which is the only way to tell two Firefox streams apart.
    pub detail: Option<String>,
}

impl Source {
    /// Name for a picker, including detail where it disambiguates.
    pub fn display(&self) -> String {
        match &self.detail {
            Some(d) if !d.is_empty() => format!("{} \u{2014} {}", self.name, truncate(d, 48)),
            _ => self.name.clone(),
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}\u{2026}")
}

impl Source {
    /// A key worth persisting across runs.
    ///
    /// Devices keep their `node.name`, so that is used where available.
    /// Applications do not have a stable identity at all, so their display name
    /// is the best available handle -- "Firefox" will match the next Firefox,
    /// which is what a user expects from a remembered choice.
    pub fn stable_key(&self) -> String {
        match (&self.node_name, self.kind) {
            (Some(n), SourceKind::Microphone | SourceKind::Monitor) => n.clone(),
            _ => match &self.detail {
                Some(d) if !d.is_empty() => format!("app:{}:{d}", self.name),
                _ => format!("app:{}", self.name),
            },
        }
    }
}

/// Enumerate capturable sources from `pw-dump` output.
///
/// Split from the subprocess call so it can be tested against recorded JSON
/// without a running PipeWire.
pub fn parse_sources(pw_dump_json: &str) -> Result<Vec<Source>> {
    let objects: Vec<serde_json::Value> =
        serde_json::from_str(pw_dump_json).context("parsing pw-dump output")?;

    let mut out = Vec::new();
    for o in &objects {
        if o.get("type").and_then(|t| t.as_str()) != Some("PipeWire:Interface:Node") {
            continue;
        }
        let props = match o.pointer("/info/props") {
            Some(p) => p,
            None => continue,
        };
        let class = props
            .get("media.class")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let node_name = props
            .get("node.name")
            .and_then(|n| n.as_str())
            .map(str::to_string);
        let Some(id) = o.get("id").and_then(|i| i.as_u64()).map(|i| i as u32) else {
            continue;
        };

        let (kind, name) = match class {
            // A real capture device, or the monitor of a sink. Monitors are
            // distinguished by node.name rather than media.class, since
            // PipeWire reports both as Audio/Source.
            "Audio/Source" | "Audio/Source/Virtual" => {
                let desc = props
                    .get("node.description")
                    .and_then(|d| d.as_str())
                    .unwrap_or_else(|| node_name.as_deref().unwrap_or("unknown"));
                let is_monitor = node_name
                    .as_deref()
                    .is_some_and(|n| n.ends_with(".monitor"));
                (
                    if is_monitor {
                        SourceKind::Monitor
                    } else {
                        SourceKind::Microphone
                    },
                    desc.to_string(),
                )
            }
            // A sink. Its monitor ports carry everything playing through it,
            // and `pw-record --target <sink>` captures exactly those. Monitors
            // are NOT separate nodes in the PipeWire graph -- pactl synthesises
            // ".monitor" sources from sinks, which is why looking for
            // Audio/Source nodes with a .monitor suffix finds nothing.
            "Audio/Sink" => {
                let desc = props
                    .get("node.description")
                    .and_then(|d| d.as_str())
                    .unwrap_or_else(|| node_name.as_deref().unwrap_or("unknown"));
                (SourceKind::Monitor, format!("Monitor of {desc}"))
            }
            // An application playing audio. Capturing this taps the stream
            // without interrupting playback.
            "Stream/Output/Audio" => {
                let app = props
                    .get("application.name")
                    .and_then(|a| a.as_str())
                    .or_else(|| props.get("node.name").and_then(|n| n.as_str()))
                    .unwrap_or("unknown application");
                (SourceKind::Application, app.to_string())
            }
            _ => continue,
        };

        let detail = if kind == SourceKind::Application {
            props
                .get("media.name")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        } else {
            None
        };

        out.push(Source {
            id,
            name,
            kind,
            node_name,
            detail,
        });
    }

    // Group by kind for a stable, readable picker, then by name so the order
    // does not jump around as PipeWire renumbers nodes.
    out.sort_by(|a, b| {
        (a.kind as u8, a.name.to_lowercase()).cmp(&(b.kind as u8, b.name.to_lowercase()))
    });
    out.dedup_by(|a, b| a.id == b.id);
    Ok(out)
}

/// Enumerate sources from the running PipeWire daemon.
pub fn list_sources() -> Result<Vec<Source>> {
    let out = std::process::Command::new("pw-dump")
        .output()
        .context("running pw-dump (is PipeWire running?)")?;
    if !out.status.success() {
        anyhow::bail!("pw-dump exited with {}", out.status);
    }
    parse_sources(&String::from_utf8_lossy(&out.stdout))
}

/// Re-resolve a remembered source, since node ids change between runs.
pub fn resolve(sources: &[Source], stable_key: &str) -> Option<Source> {
    sources.iter().find(|s| s.stable_key() == stable_key).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"[
      {"id": 63, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source",
        "node.name": "alsa_output.usb-Blue.analog-stereo.monitor",
        "node.description": "Monitor of Blue Microphones"}}},
      {"id": 43, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source/Virtual",
        "node.name": "rnnoise_source",
        "node.description": "Yeti (RNNoise)"}}},
      {"id": 105, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Stream/Output/Audio",
        "node.name": "Firefox",
        "application.name": "Firefox"}}},
      {"id": 999, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Sink",
        "node.name": "some-speaker"}}},
      {"id": 111, "type": "PipeWire:Interface:Port", "info": {"props": {}}}
    ]"#;

    #[test]
    fn finds_microphones_monitors_and_applications() {
        let s = parse_sources(SAMPLE).unwrap();
        assert!(s.iter().any(|x| x.kind == SourceKind::Microphone));
        assert!(s.iter().any(|x| x.kind == SourceKind::Monitor));
        assert!(s.iter().any(|x| x.kind == SourceKind::Application));
    }

    #[test]
    fn sinks_are_included_as_monitors_not_ignored() {
        // An earlier version skipped Audio/Sink on the assumption that an
        // output is not capturable. It is: `pw-record --target <sink>` reads
        // the sink's monitor ports, and that is the only route to "transcribe
        // whatever is playing", since no separate monitor node exists.
        let s = parse_sources(SAMPLE).unwrap();
        let sink = s.iter().find(|x| x.id == 999).expect("sink should appear");
        assert_eq!(sink.kind, SourceKind::Monitor);
    }

    #[test]
    fn non_node_objects_are_ignored() {
        // Ports are not nodes and cannot be capture targets.
        let s = parse_sources(SAMPLE).unwrap();
        assert!(!s.iter().any(|x| x.id == 111));
    }

    #[test]
    fn monitors_are_classified_by_node_name_not_media_class() {
        // PipeWire reports a monitor as Audio/Source like any microphone; only
        // the .monitor suffix distinguishes it. Getting this wrong would offer
        // "record your speakers" as if it were a microphone.
        let s = parse_sources(SAMPLE).unwrap();
        let mon = s.iter().find(|x| x.id == 63).unwrap();
        assert_eq!(mon.kind, SourceKind::Monitor);
    }

    #[test]
    fn virtual_sources_are_treated_as_microphones() {
        // rnnoise_source is Audio/Source/Virtual but is a capture input.
        let s = parse_sources(SAMPLE).unwrap();
        let rn = s.iter().find(|x| x.id == 43).unwrap();
        assert_eq!(rn.kind, SourceKind::Microphone);
        assert_eq!(rn.name, "Yeti (RNNoise)");
    }

    #[test]
    fn applications_use_their_application_name() {
        let s = parse_sources(SAMPLE).unwrap();
        let ff = s.iter().find(|x| x.id == 105).unwrap();
        assert_eq!(ff.name, "Firefox");
    }

    #[test]
    fn devices_get_a_stable_key_but_applications_key_on_name() {
        // Node ids are renumbered constantly, so a remembered choice must not
        // be stored by id.
        let s = parse_sources(SAMPLE).unwrap();
        let rn = s.iter().find(|x| x.id == 43).unwrap();
        assert_eq!(rn.stable_key(), "rnnoise_source");
        let ff = s.iter().find(|x| x.id == 105).unwrap();
        assert_eq!(ff.stable_key(), "app:Firefox");
    }

    #[test]
    fn a_remembered_source_resolves_to_a_new_node_id() {
        // Firefox restarts with a different node id; the remembered key must
        // still find it.
        let s = parse_sources(SAMPLE).unwrap();
        let moved: Vec<Source> = s
            .iter()
            .cloned()
            .map(|mut x| {
                x.id += 1000;
                x
            })
            .collect();
        let found = resolve(&moved, "app:Firefox").unwrap();
        assert_eq!(found.id, 1105);
    }

    #[test]
    fn an_unknown_key_resolves_to_nothing() {
        let s = parse_sources(SAMPLE).unwrap();
        assert!(resolve(&s, "app:Spotify").is_none());
    }

    #[test]
    fn sinks_are_offered_as_system_audio_monitors() {
        // Monitors are not separate nodes: pactl synthesises ".monitor" sources
        // from sinks, so searching for Audio/Source nodes with that suffix
        // finds nothing and the user gets no way to capture system audio.
        let json = r#"[{"id": 64, "type": "PipeWire:Interface:Node", "info": {"props": {
            "media.class": "Audio/Sink",
            "node.name": "alsa_output.pci-0000_0c_00.4.analog-stereo",
            "node.description": "Starship Analog Stereo"}}}]"#;
        let s = parse_sources(json).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].kind, SourceKind::Monitor);
        assert_eq!(s[0].name, "Monitor of Starship Analog Stereo");
    }

    #[test]
    fn two_streams_from_one_app_are_distinguishable() {
        // Firefox reports the tab title in media.name. Without it a picker
        // shows "Firefox" twice with no way to tell which is which.
        let json = r#"[
          {"id": 105, "type": "PipeWire:Interface:Node", "info": {"props": {
            "media.class": "Stream/Output/Audio", "application.name": "Firefox",
            "media.name": "Some Video - YouTube"}}},
          {"id": 100, "type": "PipeWire:Interface:Node", "info": {"props": {
            "media.class": "Stream/Output/Audio", "application.name": "Firefox",
            "media.name": "Another Page"}}}]"#;
        let s = parse_sources(json).unwrap();
        assert_eq!(s.len(), 2);
        assert_ne!(s[0].display(), s[1].display());
        assert_ne!(s[0].stable_key(), s[1].stable_key());
        assert!(s.iter().any(|x| x.display().contains("YouTube")));
    }

    #[test]
    fn long_stream_titles_are_truncated_for_display() {
        let long = "x".repeat(200);
        let src = Source {
            id: 1,
            name: "Firefox".into(),
            kind: SourceKind::Application,
            node_name: None,
            detail: Some(long),
        };
        assert!(src.display().chars().count() < 70, "got {}", src.display());
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(parse_sources("not json").is_err());
    }

    #[test]
    fn nodes_missing_props_are_skipped() {
        let json = r#"[{"id": 1, "type": "PipeWire:Interface:Node"}]"#;
        assert!(parse_sources(json).unwrap().is_empty());
    }
}
