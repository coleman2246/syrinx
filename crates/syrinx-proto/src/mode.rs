use serde::{Deserialize, Serialize};

/// Which revision semantics a session uses.
///
/// `Live` is append-only because the client types into arbitrary applications,
/// where retracting characters is destructive if the user has typed or moved the
/// cursor since. `Transcript` clients own their buffer, so revision is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Live,
    Transcript,
}

impl Mode {
    /// Whether this mode may receive `provisional` / `revise` messages.
    pub fn allows_revision(self) -> bool {
        matches!(self, Mode::Transcript)
    }
}

/// Sample format of the binary audio frames a client sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    PcmS16le,
    PcmF32le,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Unauthorized,
    /// No capacity: too many sessions, or loading would starve another GPU
    /// tenant. Refusing is correct behaviour, not a failure.
    Capacity,
    BadRequest,
    Internal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_serializes_as_lowercase_string() {
        assert_eq!(serde_json::to_string(&Mode::Live).unwrap(), r#""live""#);
        assert_eq!(
            serde_json::to_string(&Mode::Transcript).unwrap(),
            r#""transcript""#
        );
    }

    #[test]
    fn unknown_mode_is_rejected() {
        assert!(serde_json::from_str::<Mode>(r#""telepathy""#).is_err());
    }

    #[test]
    fn only_transcript_mode_allows_revision() {
        assert!(!Mode::Live.allows_revision());
        assert!(Mode::Transcript.allows_revision());
    }

    #[test]
    fn encoding_and_error_code_use_snake_case() {
        assert_eq!(
            serde_json::to_string(&Encoding::PcmS16le).unwrap(),
            r#""pcm_s16le""#
        );
        assert_eq!(
            serde_json::to_string(&ErrorCode::BadRequest).unwrap(),
            r#""bad_request""#
        );
    }
}
