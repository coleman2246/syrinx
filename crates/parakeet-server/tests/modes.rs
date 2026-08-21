//! Mode invariants.
//!
//! The live-mode test here encodes the core safety property of the whole
//! design. If someone later wires revision into live mode, this fails loudly.

use parakeet_proto::{Mode, ServerMessage};
use parakeet_server::asr::mock::MockBackend;
use parakeet_server::session::Session;

fn drive(mode: Mode, chunks: usize) -> Vec<ServerMessage> {
    let backend = MockBackend::new(&["alpha", "beta", "gamma"]);
    let mut s = Session::new(mode, &backend, "sid-1".into());
    let mut out = Vec::new();
    for _ in 0..chunks {
        out.extend(s.push_audio(&vec![0.0; 8960]).unwrap());
    }
    out.extend(s.finish().unwrap());
    out
}

#[test]
fn live_mode_never_emits_provisional_or_revise() {
    // Live mode types into arbitrary applications, where deleting characters is
    // destructive if the user has typed or moved the cursor. It must never ask
    // a client to retract.
    for m in drive(Mode::Live, 3) {
        match m {
            ServerMessage::TranscriptProvisional { .. } | ServerMessage::TranscriptRevise { .. } => {
                panic!("live mode emitted a revision message: {m:?}")
            }
            _ => {}
        }
    }
}

#[test]
fn live_mode_commits_are_sequential_from_one() {
    let seqs: Vec<u64> = drive(Mode::Live, 3)
        .into_iter()
        .filter_map(|m| match m {
            ServerMessage::TranscriptCommit { seq, .. } => Some(seq),
            _ => None,
        })
        .collect();
    assert_eq!(seqs, vec![1, 2, 3]);
}

#[test]
fn transcript_mode_still_commits_text() {
    let committed: String = drive(Mode::Transcript, 3)
        .into_iter()
        .filter_map(|m| match m {
            ServerMessage::TranscriptCommit { text, .. } => Some(text),
            _ => None,
        })
        .collect();
    assert!(committed.contains("alpha"), "got: {committed}");
}

#[test]
fn audio_shorter_than_a_chunk_emits_nothing_until_finish() {
    // Network frames have no relationship to the model's chunk size, so a
    // partial chunk must buffer rather than emit a truncated inference.
    let backend = MockBackend::new(&["alpha"]);
    let mut s = Session::new(Mode::Live, &backend, "sid".into());

    assert!(s.push_audio(&vec![0.0; 100]).unwrap().is_empty());
    assert_eq!(s.finish().unwrap().len(), 1, "finish must flush the tail");
}

#[test]
fn one_oversized_push_emits_several_chunks() {
    // A client may send a large frame; it must decompose into whole chunks.
    let backend = MockBackend::new(&["a", "b", "c"]);
    let mut s = Session::new(Mode::Live, &backend, "sid".into());
    assert_eq!(s.push_audio(&vec![0.0; 8960 * 3]).unwrap().len(), 3);
}

#[test]
fn finish_on_an_empty_session_emits_nothing() {
    let backend = MockBackend::new(&["alpha"]);
    let mut s = Session::new(Mode::Live, &backend, "sid".into());
    assert!(s.finish().unwrap().is_empty());
}
