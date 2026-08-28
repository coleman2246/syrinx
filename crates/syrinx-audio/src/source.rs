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

/// Compact names for a whole set of sources, with collisions broken apart.
///
/// [`Source::short_label`] answers for one source and cannot see the others,
/// so every monitor comes back as "System audio" and two microphones sharing
/// their first 24 characters come back identical. In a transcript line that is
/// merely vague; in a filename it is a bug. Separate mode gives each source
/// its own stream file through `save::path_for_source`, built from this name,
/// and two sources under one name point two `StreamWriter`s at one file --
/// neither of which is ever shown the other's fragments, so the records tear
/// rather than interleave.
///
/// The first source to claim a name keeps it, so the ordinary case of one
/// microphone and one monitor reads exactly as it did before.
pub fn short_labels(sources: &[Source]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(sources.len());
    for s in sources {
        let base = s.short_label();
        let mut name = base.clone();
        // Counting from two, because whatever already holds the name is the
        // first of them: "System audio" and "System audio 2".
        let mut n = 2;
        while out.contains(&name) {
            name = format!("{base} {n}");
            n += 1;
        }
        out.push(name);
    }
    out
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

    fn monitor(name: &str, stable: &str) -> Source {
        Source {
            target: SourceTarget::PipeWireNode(1),
            name: name.into(),
            kind: SourceKind::Monitor,
            detail: None,
            stable_name: Some(stable.into()),
            sink_description: None,
        }
    }

    #[test]
    fn two_monitors_are_never_given_one_name() {
        // Every monitor is "System audio" on its own, and separate mode builds
        // a stream filename from that name. Two of them under one name is two
        // writers on one file, which tears the records.
        let names = short_labels(&[
            monitor("Speakers", "alsa_output.pci-0000_00.analog"),
            monitor("Headset", "alsa_output.usb-headset.analog"),
        ]);
        assert_eq!(names, ["System audio", "System audio 2"]);
    }

    #[test]
    fn a_lone_monitor_keeps_the_name_it_always_had() {
        // The common case must read exactly as before; a "System audio 1"
        // beside nothing else would be noise.
        let names = short_labels(&[monitor("Speakers", "alsa_output.pci")]);
        assert_eq!(names, ["System audio"]);
    }

    #[test]
    fn microphones_truncated_to_the_same_name_are_still_told_apart() {
        // The label is cut to 24 characters, so two devices from one maker
        // collide long before their full names do.
        let mic = |n: &str| Source {
            target: SourceTarget::PipeWireNode(1),
            name: n.into(),
            kind: SourceKind::Microphone,
            detail: None,
            stable_name: None,
            sink_description: None,
        };
        let names = short_labels(&[
            mic("Blue Microphones Yeti Stereo A"),
            mic("Blue Microphones Yeti Stereo B"),
        ]);
        assert_ne!(names[0], names[1]);
    }

    #[test]
    fn three_of_a_kind_get_three_names() {
        let names = short_labels(&[
            monitor("A", "a"),
            monitor("B", "b"),
            monitor("C", "c"),
        ]);
        assert_eq!(names, ["System audio", "System audio 2", "System audio 3"]);
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
