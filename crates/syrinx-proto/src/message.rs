use crate::{Encoding, ErrorCode, Mode};
use serde::{Deserialize, Serialize};

/// Control messages sent by a client. Audio travels as binary frames, never
/// inside these, so the high-rate path pays no base64 tax.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "session.start")]
    SessionStart {
        mode: Mode,
        sample_rate: u32,
        encoding: Encoding,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vocabulary: Option<Vec<String>>,
    },
    /// Force emission of buffered audio without ending the session.
    #[serde(rename = "session.flush")]
    SessionFlush,
    #[serde(rename = "session.stop")]
    SessionStop,
}

/// Control messages sent by the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "session.ready")]
    SessionReady {
        session_id: String,
        chunk_ms: u32,
        model: String,
    },

    /// Final text. Never revised. The only transcript message live mode emits.
    #[serde(rename = "transcript.commit")]
    TranscriptCommit { seq: u64, text: String },

    /// Text that may still change. Transcript mode only.
    #[serde(rename = "transcript.provisional")]
    TranscriptProvisional { seq: u64, text: String },

    /// Retract `retract_n` characters and replace with `text`.
    ///
    /// RESERVED. The streaming ASR is append-only (`transcribe_chunk` returns
    /// only newly emitted tokens), so no v1 code path emits this. It exists so a
    /// future post-processing layer can revise without a breaking protocol
    /// change. Its absence in v1 is intended, not a bug.
    #[serde(rename = "transcript.revise")]
    TranscriptRevise {
        seq: u64,
        retract_n: usize,
        text: String,
    },

    #[serde(rename = "error")]
    Error {
        code: ErrorCode,
        message: String,
        retryable: bool,
    },

    #[serde(rename = "session.closed")]
    SessionClosed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_start_uses_dotted_tag() {
        let m = ClientMessage::SessionStart {
            mode: Mode::Live,
            sample_rate: 16000,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(v["type"], "session.start");
        assert_eq!(v["sample_rate"], 16000);
    }

    #[test]
    fn optional_fields_are_omitted_when_absent() {
        let m = ClientMessage::SessionStart {
            mode: Mode::Live,
            sample_rate: 16000,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(!s.contains("language"), "got: {s}");
        assert!(!s.contains("vocabulary"), "got: {s}");
    }

    #[test]
    fn commit_round_trips() {
        let m = ServerMessage::TranscriptCommit {
            seq: 7,
            text: "hello".into(),
        };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<ServerMessage>(&s).unwrap(), m);
    }

    #[test]
    fn revise_round_trips_even_though_v1_never_emits_it() {
        // Reserved for a future post-processing layer. Kept tested so it does
        // not rot into something unusable before it is needed.
        let m = ServerMessage::TranscriptRevise {
            seq: 2,
            retract_n: 13,
            text: "thirty seconds".into(),
        };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<ServerMessage>(&s).unwrap(), m);
    }

    #[test]
    fn unknown_message_type_is_rejected() {
        // Forward compatibility is explicitly NOT wanted here: an unknown
        // control message means client and server disagree, and silently
        // ignoring it would produce confusing behaviour far from the cause.
        let r = serde_json::from_str::<ClientMessage>(r#"{"type":"session.teleport"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn unit_variants_carry_only_a_tag() {
        assert_eq!(
            serde_json::to_string(&ClientMessage::SessionStop).unwrap(),
            r#"{"type":"session.stop"}"#
        );
    }
}
