//! The platform-neutral description of something capturable.

use serde::{Deserialize, Serialize};

/// What kind of thing is being captured. Drives grouping in a picker, and tells
/// the user what they are about to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SourceKind {
    /// A real capture device: microphone, line in.
    Microphone,
    /// Everything playing through an output, mixed.
    Monitor,
    /// A single application's playback stream. Linux only.
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

/// How to actually open a source, which differs by backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceTarget {
    /// A PipeWire node id. Not stable across restarts.
    PipeWireNode(u32),
    /// A cpal device by name. `loopback` means it is an *output* device to be
    /// opened for input, which WASAPI turns into system-audio capture.
    CpalDevice { name: String, loopback: bool },
}

/// Something that can be captured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub target: SourceTarget,
    /// Human-readable name.
    pub name: String,
    pub kind: SourceKind,
    /// What an application is playing, when it says. Firefox reports the tab
    /// title, which is the only way to tell two Firefox streams apart.
    pub detail: Option<String>,
    /// Backend-stable identifier, when one exists. PipeWire `node.name` for
    /// devices; `None` for application streams, which have no stable identity.
    pub stable_name: Option<String>,
    /// For monitors, the underlying output's description. Used to spot a
    /// capture device sharing a name with an output on the same card.
    pub sink_description: Option<String>,
}

impl Source {
    /// Name for a picker, including detail where it disambiguates.
    pub fn display(&self) -> String {
        match &self.detail {
            Some(d) if !d.is_empty() => format!("{} — {}", self.name, truncate(d, 48)),
            _ => self.name.clone(),
        }
    }

    /// A compact name for tagging transcript lines.
    ///
    /// `display()` is built for a picker, where the full device name helps you
    /// choose. It is far too long to prefix every line of a transcript with:
    /// "Everything playing on Starship/Matisse HD Audio Controller Analog
    /// Stereo (default output)" buries the words it is labelling.
    pub fn short_label(&self) -> String {
        match self.kind {
            // Which output it came from rarely matters; that it was the system
            // rather than a person does.
            SourceKind::Monitor => "System audio".to_string(),
            // The application name alone, not the tab title, which changes
            // mid-recording and would make the same source look like several.
            SourceKind::Application => self.name.clone(),
            SourceKind::Microphone => {
                let n = self.name.trim_end_matches(" (input)");
                truncate(n, 24)
            }
        }
    }

    /// A key worth persisting across runs.
    ///
    /// Node ids and device indices are renumbered constantly, so a remembered
    /// choice must never be stored by those. Devices have a stable name;
    /// applications do not, so their display name is the best available handle
    /// -- "Firefox" matching the next Firefox is what a user expects from a
    /// remembered choice.
    pub fn stable_key(&self) -> String {
        if let Some(n) = &self.stable_name {
            return n.clone();
        }
        match &self.detail {
            Some(d) if !d.is_empty() => format!("app:{}:{d}", self.name),
            _ => format!("app:{}", self.name),
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(name: &str, detail: Option<&str>) -> Source {
        Source {
            target: SourceTarget::PipeWireNode(1),
            name: name.into(),
            kind: SourceKind::Application,
            detail: detail.map(str::to_string),
            stable_name: None,
            sink_description: None,
        }
    }

    #[test]
    fn a_device_keys_on_its_stable_name_not_its_id() {
        let s = Source {
            target: SourceTarget::PipeWireNode(43),
            name: "Yeti (RNNoise)".into(),
            kind: SourceKind::Microphone,
            detail: None,
            stable_name: Some("rnnoise_source".into()),
            sink_description: None,
        };
        assert_eq!(s.stable_key(), "rnnoise_source");
    }

    #[test]
    fn two_streams_of_one_app_get_different_keys() {
        assert_ne!(
            app("Firefox", Some("Tab A")).stable_key(),
            app("Firefox", Some("Tab B")).stable_key()
        );
    }

    #[test]
    fn an_app_without_detail_still_has_a_key() {
        assert_eq!(app("Spotify", None).stable_key(), "app:Spotify");
    }

    #[test]
    fn display_includes_detail_when_present() {
        assert_eq!(app("Firefox", Some("Video")).display(), "Firefox — Video");
        assert_eq!(app("Firefox", None).display(), "Firefox");
    }

    #[test]
    fn long_titles_are_truncated_for_display() {
        let s = app("Firefox", Some(&"x".repeat(200)));
        assert!(s.display().chars().count() < 70, "got {}", s.display());
    }

    #[test]
    fn truncation_counts_characters_not_bytes() {
        // Byte-slicing a multi-byte title would panic mid-character.
        let s = app("Firefox", Some(&"日本語のタイトル".repeat(20)));
        let _ = s.display();
    }

    #[test]
    fn a_monitor_is_labelled_by_what_it_is_not_which_device() {
        // Prefixing every line with the full sink name buries the words.
        let s = Source {
            target: SourceTarget::PipeWireNode(1),
            name: "Everything playing on Starship/Matisse HD Audio Controller Analog Stereo                    (default output)"
                .into(),
            kind: SourceKind::Monitor,
            detail: None,
            stable_name: None,
            sink_description: None,
        };
        assert_eq!(s.short_label(), "System audio");
    }

    #[test]
    fn an_application_is_labelled_by_name_not_by_what_it_is_playing() {
        // The tab title changes mid-recording; using it would make one source
        // look like several.
        let s = app("Firefox", Some("Some Very Long Video Title - YouTube"));
        assert_eq!(s.short_label(), "Firefox");
    }

    #[test]
    fn a_microphone_label_drops_the_input_suffix_and_is_bounded() {
        let s = Source {
            target: SourceTarget::PipeWireNode(1),
            name: "Blue Microphones Analog Stereo (input)".into(),
            kind: SourceKind::Microphone,
            detail: None,
            stable_name: None,
            sink_description: None,
        };
        assert!(!s.short_label().contains("(input)"));
        assert!(s.short_label().chars().count() <= 24);
    }

    #[test]
    fn kinds_sort_microphones_before_monitors_before_applications() {
        let mut k = [
            SourceKind::Application,
            SourceKind::Monitor,
            SourceKind::Microphone,
        ];
        k.sort();
        assert_eq!(
            k,
            [
                SourceKind::Microphone,
                SourceKind::Monitor,
                SourceKind::Application
            ]
        );
    }
}
