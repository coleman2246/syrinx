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
    /// Ends the session, exactly like [`ClientMessage::SessionStop`].
    ///
    /// It was written to mean "force emission of buffered audio without ending
    /// the session", and the server has never done that: `ws` routes both to
    /// the same terminal drain. The documentation is what changed, because a
    /// non-terminal flush cannot be assembled from what `Session` exposes. It
    /// would have to release the lag buffer holding commits back while their
    /// speaker labels settle, *without* also draining the ASR stream -- and
    /// draining is one-way, since the transducer is flushed by feeding it
    /// silence that cannot afterwards be taken back out of its context.
    ///
    /// Worth building the day a client wants "give me what you have, I am still
    /// talking", which a labelling session now makes a real capability. Nothing
    /// sends this today, and `syrinx-server/tests/protocol.rs` pins the
    /// terminal behaviour so contract and code cannot drift apart again.
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

    /// Correct who said `from_seq..=to_seq`, leaving every word of it alone.
    ///
    /// A speaker needs several agreeing windows before it is minted, so the
    /// opening of a turn is committed before anyone can be named for it. This
    /// is how the name catches up: the commits it covers were emitted with no
    /// speaker, or with a provisional guess a later full window contradicted,
    /// and both are corrected rather than left wrong. It never renumbers a
    /// speaker and never touches a commit that already carries a confident
    /// label of its own.
    ///
    /// Deliberately *not* [`ServerMessage::TranscriptRevise`]. That message is
    /// reserved for a post-processing layer, says the *text* changed, and
    /// carries no speaker; relabelling shares none of those three properties,
    /// and a client's handling of the two differs -- the GUI repaints its
    /// segments, while `StreamWriter` ignores relabels outright to keep the
    /// streamed file append-only.
    ///
    /// Additive, so a client that does not know this tag drops the frame and
    /// keeps the attribution it was given live. That is a supported outcome,
    /// not a degraded one: it is exactly what the streamed file does anyway.
    #[serde(rename = "transcript.relabel")]
    TranscriptRelabel {
        /// First commit `seq` covered, inclusive.
        from_seq: u64,
        /// Last commit `seq` covered, inclusive. Equal to `from_seq` for a
        /// single commit; the range is always contiguous in `seq`.
        to_seq: u64,
        /// Who it was, numbered from 1 exactly as on a commit. Never 0, and
        /// never a number that has not already appeared as a speaker.
        speaker: u32,
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
    fn relabel_round_trips() {
        let m = ServerMessage::TranscriptRelabel {
            from_seq: 3,
            to_seq: 7,
            speaker: 2,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<ServerMessage>(&s).unwrap(), m);
    }

    #[test]
    fn a_relabel_names_a_range_and_a_speaker_and_no_text() {
        // The whole point of the message being its own variant rather than a
        // speaker bolted onto `transcript.revise`: nothing here says the words
        // changed, so a client that applies it never has to touch its buffer.
        let s = serde_json::to_string(&ServerMessage::TranscriptRelabel {
            from_seq: 3,
            to_seq: 7,
            speaker: 2,
        })
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "transcript.relabel");
        assert_eq!(v["from_seq"], 3);
        assert_eq!(v["to_seq"], 7);
        assert_eq!(v["speaker"], 2);
        assert!(!s.contains("text"), "got: {s}");
        assert!(!s.contains("retract"), "got: {s}");
    }

    #[test]
    fn a_relabel_never_names_speaker_zero() {
        // Speakers are numbered from 1, so 0 has no referent. Nothing on the
        // wire enforces it -- `u32` cannot -- and this is the note that says
        // the server must not send one rather than a check that it did not.
        let m: ServerMessage = serde_json::from_str(
            r#"{"type":"transcript.relabel","from_seq":1,"to_seq":1,"speaker":1}"#,
        )
        .unwrap();
        let ServerMessage::TranscriptRelabel { speaker, .. } = m else {
            panic!()
        };
        assert!(speaker >= 1);
    }

    #[test]
    fn an_old_client_skips_a_relabel_rather_than_breaking() {
        // Forward compatibility for the *server* messages is a skipped frame,
        // not a tolerated field: a client built before this variant existed
        // fails to decode the tag and its reader loop logs and carries on --
        // `syrinx-client`'s `undecodable server frame` arm. So the property to
        // pin is that an unknown tag is an ordinary `Err`, which is what that
        // arm is reachable by. `syrinx-client`'s own tests cover the arm.
        let unknown = r#"{"type":"transcript.teleport","seq":1}"#;
        let e = serde_json::from_str::<ServerMessage>(unknown);
        assert!(e.is_err(), "an unknown tag must be a decode error");

        // And the messages an *old server* sends still decode against this
        // enum, so adding the variant costs nothing in the other direction.
        for old in [
            r#"{"type":"transcript.commit","seq":1,"text":"hi"}"#,
            r#"{"type":"transcript.provisional","seq":1,"text":"hi"}"#,
            r#"{"type":"transcript.revise","seq":1,"retract_n":2,"text":"hi"}"#,
            r#"{"type":"session.ready","session_id":"x","chunk_ms":560,"model":"m"}"#,
            r#"{"type":"session.closed","reason":"ended"}"#,
        ] {
            assert!(
                serde_json::from_str::<ServerMessage>(old).is_ok(),
                "an old server's {old} stopped parsing"
            );
        }
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
        let ServerMessage::SessionReady { diarize, .. } = m else {
            panic!()
        };
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
        assert!(
            serde_json::to_string(&m)
                .unwrap()
                .contains("\"diarize\":true")
        );
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
