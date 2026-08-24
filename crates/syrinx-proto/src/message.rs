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
        /// Ask for speaker labels on this session's transcript messages.
        /// Best-effort: the server answers what it will actually do in
        /// `session.ready`, and proceeds unlabelled rather than refusing.
        #[serde(default, skip_serializing_if = "is_false")]
        diarize: bool,
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
        /// Whether this session will attach speaker labels. The client can ask
        /// and still not receive -- missing models, or a mode without a
        /// transcript -- and the honest answer belongs in the handshake.
        #[serde(default, skip_serializing_if = "is_false")]
        diarize: bool,
    },

    /// Final text. Never revised. The only transcript message live mode emits.
    #[serde(rename = "transcript.commit")]
    TranscriptCommit {
        seq: u64,
        text: String,
        /// Who said it, numbered from 1 in order of first confident
        /// appearance. Absent when no diarizer ran, or when it honestly
        /// could not tell for this stretch -- a gap, never a guess.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speaker: Option<u32>,
    },

    /// Text that may still change. Transcript mode only.
    #[serde(rename = "transcript.provisional")]
    TranscriptProvisional {
        seq: u64,
        text: String,
        /// Who said it, numbered from 1 in order of first confident
        /// appearance. Absent when no diarizer ran, or when it honestly
        /// could not tell for this stretch -- a gap, never a guess.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speaker: Option<u32>,
    },

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

/// serde helper: lets a false bool vanish from the wire entirely.
fn is_false(b: &bool) -> bool {
    !*b
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
            diarize: false,
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
            diarize: false,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(!s.contains("language"), "got: {s}");
        assert!(!s.contains("vocabulary"), "got: {s}");
    }

    #[test]
    fn session_start_without_diarize_parses_as_false() {
        // Wire compatibility: every message an old client sends today.
        let s =
            r#"{"type":"session.start","mode":"live","sample_rate":16000,"encoding":"pcm_s16le"}"#;
        let m: ClientMessage = serde_json::from_str(s).unwrap();
        let ClientMessage::SessionStart { diarize, .. } = m else {
            panic!()
        };
        assert!(!diarize);
    }

    #[test]
    fn diarize_false_is_omitted_from_the_wire() {
        // False is the overwhelmingly common case; a field on every message
        // saying "nothing special" is noise an old server need never see.
        let m = ClientMessage::SessionStart {
            mode: Mode::Live,
            sample_rate: 16000,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
            diarize: false,
        };
        assert!(!serde_json::to_string(&m).unwrap().contains("diarize"));
    }

    #[test]
    fn diarize_true_round_trips() {
        let m = ClientMessage::SessionStart {
            mode: Mode::Transcript,
            sample_rate: 16000,
            encoding: Encoding::PcmS16le,
            language: None,
            vocabulary: None,
            diarize: true,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<ClientMessage>(&s).unwrap(), m);
    }

    #[test]
    fn commit_round_trips() {
        let m = ServerMessage::TranscriptCommit {
            seq: 7,
            text: "hello".into(),
            speaker: None,
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
    fn unknown_field_on_a_known_variant_is_ignored() {
        // Forward compatibility: a future client field this server does not
        // know about yet must not break parsing.
        let s = r#"{"type":"session.start","mode":"live","sample_rate":16000,"encoding":"pcm_s16le","future_field":true}"#;
        assert!(serde_json::from_str::<ClientMessage>(s).is_ok());
    }

    #[test]
    fn session_ready_without_diarize_parses_as_false() {
        // A new client against an old server must not choke on the handshake.
        let s = r#"{"type":"session.ready","session_id":"x","chunk_ms":560,"model":"m"}"#;
        let m: ServerMessage = serde_json::from_str(s).unwrap();
        let ServerMessage::SessionReady { diarize, .. } = m else { panic!() };
        assert!(!diarize);
    }

    #[test]
    fn session_ready_reports_diarize_when_on() {
        let m = ServerMessage::SessionReady {
            session_id: "x".into(),
            chunk_ms: 560,
            model: "m".into(),
            diarize: true,
        };
        assert!(serde_json::to_string(&m).unwrap().contains("\"diarize\":true"));
    }

    #[test]
    fn session_ready_omits_diarize_when_false() {
        let m = ServerMessage::SessionReady {
            session_id: "x".into(),
            chunk_ms: 560,
            model: "m".into(),
            diarize: false,
        };
        assert!(!serde_json::to_string(&m).unwrap().contains("diarize"));
    }

    #[test]
    fn commit_without_speaker_parses_and_omits() {
        // Both directions of compatibility in one place: an old server's commit
        // parses, and an unlabelled commit from a new server looks identical to
        // an old client.
        let old = r#"{"type":"transcript.commit","seq":1,"text":"hi"}"#;
        let m: ServerMessage = serde_json::from_str(old).unwrap();
        let ServerMessage::TranscriptCommit { speaker, .. } = &m else {
            panic!()
        };
        assert_eq!(*speaker, None);
        assert!(!serde_json::to_string(&m).unwrap().contains("speaker"));
    }

    #[test]
    fn commit_with_speaker_round_trips() {
        let m = ServerMessage::TranscriptCommit {
            seq: 7,
            text: "hello".into(),
            speaker: Some(2),
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"speaker\":2"), "got: {s}");
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
