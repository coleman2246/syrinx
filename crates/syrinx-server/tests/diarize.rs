//! The lag buffer: how a commit waits for its speaker label.
//!
//! Labelling is invisible on the wire except as a label and a delay, so the
//! whole contract -- lag, majority, honest uncertainty, strike-out -- is pinned
//! here as exact message sequences, with no model anywhere near the tests.

use anyhow::anyhow;
use syrinx_proto::{Mode, ServerMessage};
use syrinx_server::asr::mock::MockBackend;
use syrinx_server::diarize::{Diarizer, MockDiarizer};
use syrinx_server::session::Session;

/// Tiny chunks: these tests count chunks, never samples.
const CHUNK: usize = 16;

/// One scripted word per chunk, for the tests that need runway: long enough
/// for a diarizer to strike out and for the lag window to still have somewhere
/// to go afterwards.
const WORDS: [&str; 8] = [
    "one", "two", "three", "four", "five", "six", "seven", "eight",
];

/// A transcript-mode session over a scripted backend.
fn over(backend: MockBackend, diarizer: Option<Box<dyn Diarizer>>) -> Session {
    let backend = backend.with_chunk_samples(CHUNK);
    Session::new(Mode::Transcript, &backend, "sid".into(), diarizer)
}

/// The common case: one scripted word per chunk from the backend, one scripted
/// label per chunk from the diarizer, and no tail left in the model.
fn session(words: &[&str], diarizer: Option<Box<dyn Diarizer>>) -> Session {
    over(MockBackend::new(words), diarizer)
}

fn diarizer(labels: &[Option<u32>]) -> Option<Box<dyn Diarizer>> {
    Some(Box::new(MockDiarizer::labels(labels)))
}

/// One whole chunk of silence in, whatever it commits out.
fn push(s: &mut Session) -> Vec<(String, Option<u32>)> {
    commits(s.push_audio(&[0.0; CHUNK]).unwrap())
}

fn finish(s: &mut Session) -> Vec<(String, Option<u32>)> {
    commits(s.finish().unwrap())
}

/// Commits as (text, speaker). The ASR is append-only, so anything else is a
/// test failure rather than a case to handle.
fn commits(msgs: Vec<ServerMessage>) -> Vec<(String, Option<u32>)> {
    msgs.into_iter()
        .map(|m| match m {
            ServerMessage::TranscriptCommit { text, speaker, .. } => (text, speaker),
            other => panic!("expected a commit, got {other:?}"),
        })
        .collect()
}

#[test]
fn without_a_diarizer_a_chunk_commits_immediately() {
    // The guard on every session that did not ask for labels: no lag, no
    // label, exactly the behaviour from before diarization existed. This is the
    // path live mode always takes, since it is never handed a diarizer.
    let mut s = session(&["alpha"], None);
    assert_eq!(push(&mut s), vec![("alpha ".into(), None)]);
}

#[test]
fn with_a_diarizer_a_commit_waits_out_the_lag() {
    // The label needs more audio than the text does, so the text waits for it.
    // Two chunks of waiting, per the spike's calibration of the embedding
    // window against the chunk size.
    let mut s = session(&["alpha"], diarizer(&[Some(1), Some(1), Some(1)]));
    assert!(
        push(&mut s).is_empty(),
        "the first chunk must not commit yet"
    );
    assert!(push(&mut s).is_empty(), "nor the second");
    assert_eq!(push(&mut s), vec![("alpha ".into(), Some(1))]);
}

#[test]
fn one_oversized_push_labels_and_releases_chunk_by_chunk() {
    // A client may send a frame holding several chunks. The diarizer must see
    // them one at a time and in order -- a label per chunk, not per frame --
    // and commits must ripen inside the call that completes their window
    // rather than waiting for the next frame to arrive.
    let mut s = session(&WORDS, diarizer(&[Some(1), Some(2), Some(2)]));

    // Chunks 1-3. The first ripens on the third, and is outvoted within the
    // same call: one label per frame would have left it Some(1).
    let out = commits(s.push_audio(&[0.0; CHUNK * 3]).unwrap());
    assert_eq!(out, vec![("one ".into(), Some(2))]);

    // Chunks 4-6, past the diarizer's script, so its labels are unknown. Three
    // more commits ripen inside this one call, each with its own window.
    let out = commits(s.push_audio(&[0.0; CHUNK * 3]).unwrap());
    assert_eq!(
        out,
        vec![
            ("two ".into(), Some(2)),
            ("three ".into(), Some(2)),
            ("four ".into(), None),
        ]
    );
}

#[test]
fn the_label_is_the_majority_over_the_lag_window() {
    // A commit is labelled by a vote of its own chunk and the lag window after
    // it, so one odd chunk cannot flip a turn.
    let label = |labels: &[Option<u32>]| {
        let mut s = session(&["alpha"], diarizer(labels));
        let mut out = Vec::new();
        for _ in 0..3 {
            out.extend(push(&mut s));
        }
        out.first().expect("the first chunk ripens on the third").1
    };

    assert_eq!(label(&[Some(1), Some(1), Some(1)]), Some(1), "unanimous");
    assert_eq!(label(&[Some(1), Some(2), Some(2)]), Some(2), "outvoted");
    // A tie goes to the earlier chunk: that is where the words actually live.
    assert_eq!(label(&[Some(1), Some(2), None]), Some(1), "tie");
}

#[test]
fn an_unknown_speaker_stays_unknown() {
    // Silence and cross-talk are normal, and a gap is honest: a guess would
    // reach the user as a speaker turn that never happened. (The wire form
    // omits the field entirely; syrinx-proto pins that.)
    let mut s = session(&["alpha"], diarizer(&[None, None, None]));
    let mut out = Vec::new();
    for _ in 0..3 {
        out.extend(push(&mut s));
    }
    assert_eq!(out, vec![("alpha ".into(), None)]);
}

#[test]
fn finish_flushes_what_the_lag_buffer_still_holds() {
    // Nothing is ever lost to the buffer. A session that ends mid-window still
    // emits its text, with whatever label had settled by then.
    let mut s = session(&["alpha", "beta"], diarizer(&[Some(1), Some(1)]));
    assert!(push(&mut s).is_empty());
    assert!(push(&mut s).is_empty());
    assert_eq!(
        finish(&mut s),
        vec![("alpha ".into(), Some(1)), ("beta ".into(), Some(1))]
    );

    // And the same when a partial chunk is still buffered, which takes a
    // different path through finish().
    let mut s = session(&["alpha"], diarizer(&[Some(3)]));
    assert!(s.push_audio(&[0.0; CHUNK / 2]).unwrap().is_empty());
    assert_eq!(finish(&mut s), vec![("alpha ".into(), Some(3))]);
}

#[test]
fn the_models_tail_is_labelled_from_the_last_window() {
    // finish() drains whatever text the model was still holding. That text
    // belongs to the last chunk that went in, so it is labelled from that
    // chunk's window -- which means that chunk's label has to survive the
    // pruning that follows the last release.
    let mut s = over(
        MockBackend::new(&["alpha"]).with_tail("tail "),
        diarizer(&[Some(1), Some(1), Some(1)]),
    );
    let mut out = Vec::new();
    for _ in 0..3 {
        out.extend(push(&mut s));
    }
    assert_eq!(out, vec![("alpha ".into(), Some(1))]);
    assert_eq!(finish(&mut s), vec![("tail ".into(), Some(1))]);

    // A session that ends with a tail and no chunks at all: chunk indices
    // start at zero, so the arithmetic has nowhere below to go.
    let mut s = over(
        MockBackend::new(&[]).with_tail("tail "),
        diarizer(&[Some(1)]),
    );
    assert_eq!(finish(&mut s), vec![("tail ".into(), None)]);
}

#[test]
fn a_diarizer_that_keeps_failing_is_dropped() {
    // Labels are decoration; the transcript is the work. Five consecutive
    // failures and the session stops asking -- and never asks again, so the
    // label scripted after them can never reach a commit.
    let mut script: Vec<anyhow::Result<Option<u32>>> =
        (0..5).map(|_| Err(anyhow!("boom"))).collect();
    script.extend((0..3).map(|_| Ok(Some(7))));
    let mut s = session(&WORDS, Some(Box::new(MockDiarizer::new(script))));

    let mut out = Vec::new();
    for _ in 0..WORDS.len() {
        out.extend(push(&mut s));
    }
    out.extend(finish(&mut s));

    let texts: Vec<String> = out.iter().map(|(t, _)| t.clone()).collect();
    assert_eq!(
        texts,
        WORDS.map(|w| format!("{w} ")),
        "the transcript survives"
    );
    assert!(
        out.iter().all(|(_, sp)| sp.is_none()),
        "a dropped diarizer labels nothing, ever again: {out:?}"
    );
}

#[test]
fn a_dropped_diarizer_takes_the_lag_with_it() {
    // Once the diarizer is gone there is no label coming, so there is nothing
    // left to wait for. Holding text back would be pure added latency, and a
    // session that has already lost its labels should not also feel slower.
    let script: Vec<anyhow::Result<Option<u32>>> = (0..5).map(|_| Err(anyhow!("boom"))).collect();
    let mut s = session(&WORDS, Some(Box::new(MockDiarizer::new(script))));
    for _ in 0..5 {
        let _ = push(&mut s);
    }

    // The sixth chunk arrives after the strike-out: its own text, in its own
    // push, with nothing held over.
    assert_eq!(push(&mut s), vec![("six ".into(), None)]);
}

#[test]
fn one_success_wipes_the_slate() {
    // Only *consecutive* failures count. An occasional hiccup is survivable, so
    // a diarizer that still answers sometimes keeps its job.
    let mut script: Vec<anyhow::Result<Option<u32>>> = vec![
        Err(anyhow!("boom")),
        Err(anyhow!("boom")),
        Ok(Some(1)),
        Err(anyhow!("boom")),
        Err(anyhow!("boom")),
        Err(anyhow!("boom")),
        Err(anyhow!("boom")),
    ];
    script.push(Ok(Some(2)));
    let mut s = session(&WORDS, Some(Box::new(MockDiarizer::new(script))));

    let mut out = Vec::new();
    for _ in 0..WORDS.len() {
        out.extend(push(&mut s));
    }
    out.extend(finish(&mut s));

    let speakers: Vec<Option<u32>> = out.iter().map(|(_, sp)| *sp).collect();
    assert_eq!(
        speakers.first(),
        Some(&Some(1)),
        "the third chunk's answer labels the first chunk: {speakers:?}"
    );
    assert_eq!(
        speakers.last(),
        Some(&Some(2)),
        "four more failures and the diarizer is still being asked: {speakers:?}"
    );
}

#[test]
fn seq_numbers_run_unbroken_through_the_lag_buffer() {
    // seq is assigned where the message leaves, not where the text arrived, so
    // holding commits back must not open a gap or swap an order. A client
    // reassembles by seq, and cannot tell a delayed commit from a lost one.
    let mut s = session(&WORDS, diarizer(&[Some(1); 8]));
    let mut msgs = Vec::new();
    for _ in 0..WORDS.len() {
        msgs.extend(s.push_audio(&[0.0; CHUNK]).unwrap());
    }
    msgs.extend(s.finish().unwrap());

    let seqs: Vec<u64> = msgs
        .iter()
        .map(|m| match m {
            ServerMessage::TranscriptCommit { seq, .. } => *seq,
            other => panic!("expected a commit, got {other:?}"),
        })
        .collect();
    assert_eq!(seqs, (1..=WORDS.len() as u64).collect::<Vec<_>>());
    assert_eq!(
        commits(msgs)
            .iter()
            .map(|(t, _)| t.clone())
            .collect::<Vec<_>>(),
        WORDS.map(|w| format!("{w} ")),
        "and the text is still in the order it was spoken"
    );
}
