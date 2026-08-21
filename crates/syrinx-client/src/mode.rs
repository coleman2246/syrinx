//! What to do with transcribed text.

use syrinx_proto::Mode as WireMode;
use serde::{Deserialize, Serialize};

/// What the client does with text as it arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    /// Accumulate a transcript for the caller to read or save. Nothing is typed.
    #[default]
    Transcribe,
    /// Type at the cursor as text arrives. Nothing is accumulated for display.
    Type,
    /// Both: keep a transcript and type it.
    Both,
}

impl OutputMode {
    pub fn label(self) -> &'static str {
        match self {
            OutputMode::Transcribe => "Transcribe",
            OutputMode::Type => "Type at cursor",
            OutputMode::Both => "Transcribe + type",
        }
    }

    pub fn types_at_cursor(self) -> bool {
        matches!(self, OutputMode::Type | OutputMode::Both)
    }

    pub fn keeps_transcript(self) -> bool {
        matches!(self, OutputMode::Transcribe | OutputMode::Both)
    }

    /// Which protocol mode this output mode requires.
    ///
    /// Anything that types must run the wire session in [`WireMode::Live`],
    /// which is append-only. Keystrokes cannot be safely retracted: by the time
    /// a revision arrived the user may have typed or moved the cursor, and
    /// deleting characters would destroy their work. Only a client that owns
    /// its own buffer can accept revisions.
    pub fn wire_mode(self) -> WireMode {
        if self.types_at_cursor() {
            WireMode::Live
        } else {
            WireMode::Transcript
        }
    }

    pub const ALL: [OutputMode; 3] = [
        OutputMode::Transcribe,
        OutputMode::Type,
        OutputMode::Both,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_mode_that_types_must_use_the_append_only_wire_mode() {
        // The core safety invariant. If this ever returns Transcript for a
        // typing mode, the server becomes free to send retractions and the
        // client would delete whatever the user typed in the meantime.
        for m in OutputMode::ALL {
            if m.types_at_cursor() {
                assert_eq!(m.wire_mode(), WireMode::Live, "{m:?} types but is not Live");
            }
        }
    }

    #[test]
    fn transcribe_only_may_accept_revisions() {
        // The client owns that buffer, so rewriting it is safe and gives better
        // final text.
        assert_eq!(OutputMode::Transcribe.wire_mode(), WireMode::Transcript);
    }

    #[test]
    fn both_keeps_a_transcript_and_types() {
        assert!(OutputMode::Both.types_at_cursor());
        assert!(OutputMode::Both.keeps_transcript());
    }

    #[test]
    fn type_mode_does_not_accumulate() {
        assert!(!OutputMode::Type.keeps_transcript());
    }

    #[test]
    fn modes_round_trip_through_config() {
        for m in OutputMode::ALL {
            let s = serde_json::to_string(&m).unwrap();
            assert_eq!(serde_json::from_str::<OutputMode>(&s).unwrap(), m);
        }
    }

    #[test]
    fn labels_are_distinct() {
        let mut l: Vec<&str> = OutputMode::ALL.iter().map(|m| m.label()).collect();
        l.sort_unstable();
        let n = l.len();
        l.dedup();
        assert_eq!(l.len(), n);
    }
}
