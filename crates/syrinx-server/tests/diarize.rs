//! The lag buffer: how a commit waits for its speaker label.
//!
//! Labelling is invisible on the wire except as a label and a delay, so the
//! whole contract -- lag, majority, honest uncertainty, strike-out -- is pinned
//! here as exact message sequences, with no model anywhere near the tests.

use anyhow::anyhow;
use syrinx_proto::{Mode, ServerMessage};
use syrinx_server::asr::mock::MockBackend;
use syrinx_server::diarize::{Attribution, Diarizer, MockDiarizer, Relabel};
use syrinx_server::session::{LAG_CHUNKS, RELABEL_WINDOW, Session, SessionTuning};

/// Tiny chunks: these tests count chunks, never samples.
const CHUNK: usize = 16;

/// One scripted word per chunk, for the tests that need runway: long enough
/// for a diarizer to strike out and for the lag window to still have somewhere
/// to go afterwards.
const WORDS: [&str; 8] = [
    "one", "two", "three", "four", "five", "six", "seven", "eight",
];

/// A transcript-mode session over a scripted backend, at the shipped lag.
fn over(backend: MockBackend, diarizer: Option<Box<dyn Diarizer>>) -> Session {
    let backend = backend.with_chunk_samples(CHUNK);
    Session::new(Mode::Transcript, &backend, "sid".into(), diarizer)
}

/// The same at a configured lag depth -- `diarize_lag_chunks`, as `ws.rs`
/// passes it.
fn at_lag(words: &[&str], diarizer: Option<Box<dyn Diarizer>>, lag: usize) -> Session {
    tuned(
        words,
        diarizer,
        SessionTuning {
            lag_chunks: lag,
            ..Default::default()
        },
    )
}

/// A session at whatever settings a test is about.
fn tuned(words: &[&str], diarizer: Option<Box<dyn Diarizer>>, tuning: SessionTuning) -> Session {
    let backend = MockBackend::new(words).with_chunk_samples(CHUNK);
    Session::with_tuning(Mode::Transcript, &backend, "sid".into(), diarizer, tuning)
}

/// A diarizer whose answers are scripted whole, for the tests that are about
/// turn changes or corrections rather than about labels.
fn scripted(script: Vec<Attribution>) -> Option<Box<dyn Diarizer>> {
    Some(Box::new(MockDiarizer::scripted(
        script.into_iter().map(Ok).collect(),
    )))
}

/// One chunk's answer: a speaker, and a turn change at this chunk.
fn turn(speaker: Option<u32>) -> Attribution {
    Attribution {
        speaker,
        boundary: true,
        ..Default::default()
    }
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
/// test failure rather than a case to handle -- relabels included, since a
/// test that wants those asks for them with [`relabels`].
fn commits(msgs: Vec<ServerMessage>) -> Vec<(String, Option<u32>)> {
    msgs.into_iter()
        .map(|m| match m {
            ServerMessage::TranscriptCommit { text, speaker, .. } => (text, speaker),
            other => panic!("expected a commit, got {other:?}"),
        })
        .collect()
}

/// Relabels as (from_seq, to_seq, speaker), with commits filtered out.
fn relabels(msgs: &[ServerMessage]) -> Vec<(u64, u64, u32)> {
    msgs.iter()
        .filter_map(|m| match m {
            ServerMessage::TranscriptRelabel {
                from_seq,
                to_seq,
                speaker,
            } => Some((*from_seq, *to_seq, *speaker)),
            _ => None,
        })
        .collect()
}

/// Commits as (seq, text, speaker), for the tests that need to name a seq in
/// order to check a relabel against it.
fn sequenced(msgs: &[ServerMessage]) -> Vec<(u64, String, Option<u32>)> {
    msgs.iter()
        .filter_map(|m| match m {
            ServerMessage::TranscriptCommit {
                seq, text, speaker, ..
            } => Some((*seq, text.clone(), *speaker)),
            _ => None,
        })
        .collect()
}

/// Push one chunk and keep every message, relabels included.
fn push_all(s: &mut Session) -> Vec<ServerMessage> {
    s.push_audio(&[0.0; CHUNK]).unwrap()
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
fn the_lag_depth_is_what_the_session_was_built_with() {
    // `diarize_lag_chunks` reaches the wire as two things at once: how long the
    // first commit waits, and how wide a window votes on its label. One script
    // shows both, because the speaker changes at the second chunk -- so a
    // deeper lag delays the commit *and*, with nothing marking the change,
    // lets the new speaker outvote the old.
    //
    // That second half is the bug, not the feature, and the test below states
    // the intent: the words being labelled here are chunk 0's, which are the
    // *first* speaker's, and a deeper lag should not hand them to whoever
    // interrupted. What makes the difference is whether the diarizer detected
    // the turn change; this script does not, so the old behaviour stands and
    // is worth keeping pinned as the thing the detector is measured against.
    let script = [Some(1), Some(2), Some(2), Some(2)];
    let first_commit = |lag: usize| {
        let mut s = at_lag(&WORDS, diarizer(&script), lag);
        (0..4).find_map(|chunk| push(&mut s).first().map(|(_, speaker)| (chunk, *speaker)))
    };

    assert_eq!(
        first_commit(0),
        Some((0, Some(1))),
        "no wait at all, labelled from its own chunk"
    );
    assert_eq!(
        first_commit(1),
        Some((1, Some(1))),
        "one chunk of window, and a tie goes to the earlier one"
    );
    assert_eq!(first_commit(2), Some((2, Some(2))), "the calibrated depth");
    assert_eq!(first_commit(3), Some((3, Some(2))), "deeper still");
}

#[test]
fn the_vote_does_not_reach_across_a_turn_change() {
    // The same script, with the diarizer reporting the turn change it really
    // is. Chunk 0's words belong to speaker 1 and nobody else, so no lag depth
    // should be able to give them to speaker 2 -- the vote stops where the
    // turn does.
    //
    // Both halves of the old rule were wrong here and for the same reason. The
    // window reached into the next speaker's chunks, so at the calibrated
    // depth the outgoing speaker was outvoted on their own sentence; and the
    // tie-break towards the earliest label hands a tie to whoever was there
    // first, which at a boundary is the outgoing speaker by construction. A
    // vote that spans two turns is not a vote about either of them.
    let script = || {
        vec![
            Attribution::speaker(Some(1)),
            turn(Some(2)),
            Attribution::speaker(Some(2)),
            Attribution::speaker(Some(2)),
        ]
    };
    let first_commit = |lag: usize| {
        let mut s = at_lag(&WORDS, scripted(script()), lag);
        (0..4).find_map(|chunk| push(&mut s).first().map(|(_, speaker)| (chunk, *speaker)))
    };

    for lag in 0..4 {
        assert_eq!(
            first_commit(lag),
            Some((lag, Some(1))),
            "at lag {lag} the first speaker's own words went to the second"
        );
    }
}

#[test]
fn a_turn_change_at_the_commits_own_chunk_does_not_clip_its_vote() {
    // A boundary marks where a turn *starts*, so the commit that opens the
    // turn has to be allowed to vote over the chunks that follow it. Clipping
    // on the boundary itself would leave every turn's opening commit voting
    // on one chunk, which is the latency this design set out to remove rather
    // than a fix for it.
    let mut s = at_lag(
        &WORDS,
        scripted(vec![
            turn(Some(2)),
            Attribution::speaker(Some(2)),
            Attribution::speaker(Some(2)),
        ]),
        2,
    );
    let mut out = Vec::new();
    for _ in 0..3 {
        out.extend(push(&mut s));
    }
    assert_eq!(out.first(), Some(&("one ".into(), Some(2))));
}

// ---------------------------------------------------------------- relabels

#[test]
fn minting_a_speaker_names_the_text_that_was_committed_before_them() {
    // Complaint 1, fixed without touching the four-window reluctance that
    // causes it. The first two chunks are committed with nobody's name on
    // them, because nobody has been minted yet; the third chunk mints speaker
    // 1 and says so covered chunks 0 to 2.
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            Attribution::default(),
            Attribution::default(),
            Attribution {
                speaker: Some(1),
                relabels: vec![Relabel {
                    from_chunk: 0,
                    to_chunk: 2,
                    speaker: 1,
                }],
                ..Default::default()
            },
        ]),
        SessionTuning {
            lag_chunks: 0,
            ..Default::default()
        },
    );

    let mut msgs = Vec::new();
    for _ in 0..3 {
        msgs.extend(push_all(&mut s));
    }

    // The commits went out honestly unattributed, and are not rewritten.
    assert_eq!(
        sequenced(&msgs),
        vec![
            (1, "one ".into(), None),
            (2, "two ".into(), None),
            (3, "three ".into(), Some(1)),
        ]
    );
    // And one correction covers exactly the two that had no speaker, as a
    // single contiguous range.
    assert_eq!(relabels(&msgs), vec![(1, 2, 1)]);
}

#[test]
fn a_correction_goes_out_before_the_commit_that_provoked_it() {
    // A client applies messages in order, so a correction naming seq 1 has to
    // arrive before the commit that made it knowable -- otherwise the client
    // paints the new speaker's line and only then goes back to fix the old
    // one, which is the flicker this ordering exists to avoid.
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            Attribution::default(),
            Attribution {
                speaker: Some(1),
                relabels: vec![Relabel {
                    from_chunk: 0,
                    to_chunk: 1,
                    speaker: 1,
                }],
                ..Default::default()
            },
        ]),
        SessionTuning {
            lag_chunks: 0,
            ..Default::default()
        },
    );
    push_all(&mut s);
    let second = push_all(&mut s);
    assert!(
        matches!(
            second.first(),
            Some(ServerMessage::TranscriptRelabel { .. })
        ),
        "the correction should lead the batch: {second:?}"
    );
}

#[test]
fn a_correction_never_overwrites_a_label_a_window_settled() {
    // The rule that keeps a correction inside its own turn. Chunk 0 is
    // confidently speaker 2's; a later relabel claiming the range back for
    // speaker 1 must leave it alone and take only the gap after it.
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            Attribution::speaker(Some(2)),
            Attribution::default(),
            Attribution {
                speaker: Some(1),
                relabels: vec![Relabel {
                    from_chunk: 0,
                    to_chunk: 2,
                    speaker: 1,
                }],
                ..Default::default()
            },
        ]),
        SessionTuning {
            lag_chunks: 0,
            ..Default::default()
        },
    );
    let mut msgs = Vec::new();
    for _ in 0..3 {
        msgs.extend(push_all(&mut s));
    }
    assert_eq!(
        sequenced(&msgs),
        vec![
            (1, "one ".into(), Some(2)),
            (2, "two ".into(), None),
            (3, "three ".into(), Some(1)),
        ]
    );
    assert_eq!(
        relabels(&msgs),
        vec![(2, 2, 1)],
        "only the unattributed commit should have been corrected"
    );
}

#[test]
fn a_provisional_guess_is_corrected_rather_than_left_wrong() {
    // The other half of what a hop's speed costs. A 0.75 s embedding named
    // speaker 2 about a second into the turn; the full window says 1, and the
    // text that carried the guess is corrected rather than left carrying it.
    let guess = Attribution {
        speaker: Some(2),
        provisional: true,
        ..Default::default()
    };
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            guess.clone(),
            guess,
            Attribution {
                speaker: Some(1),
                relabels: vec![Relabel {
                    from_chunk: 0,
                    to_chunk: 2,
                    speaker: 1,
                }],
                ..Default::default()
            },
        ]),
        SessionTuning {
            lag_chunks: 0,
            ..Default::default()
        },
    );
    let mut msgs = Vec::new();
    for _ in 0..3 {
        msgs.extend(push_all(&mut s));
    }
    assert_eq!(
        sequenced(&msgs)
            .iter()
            .map(|(_, _, sp)| *sp)
            .collect::<Vec<_>>(),
        vec![Some(2), Some(2), Some(1)],
        "the guesses go out as they were made; the file of record keeps them"
    );
    assert_eq!(relabels(&msgs), vec![(1, 2, 1)]);
}

#[test]
fn a_correction_that_agrees_with_the_guess_says_nothing() {
    // A relabel is a correction, not a confirmation. Repeating a label the
    // client already has would be a repaint for nothing.
    let guess = Attribution {
        speaker: Some(1),
        provisional: true,
        ..Default::default()
    };
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            guess.clone(),
            guess,
            Attribution {
                speaker: Some(1),
                relabels: vec![Relabel {
                    from_chunk: 0,
                    to_chunk: 2,
                    speaker: 1,
                }],
                ..Default::default()
            },
        ]),
        SessionTuning {
            lag_chunks: 0,
            ..Default::default()
        },
    );
    let mut msgs = Vec::new();
    for _ in 0..3 {
        msgs.extend(push_all(&mut s));
    }
    assert!(relabels(&msgs).is_empty(), "{:?}", relabels(&msgs));
}

#[test]
fn a_relabel_window_of_zero_emits_no_corrections_at_all() {
    // The escape hatch, and it has to be complete: not a narrower window, not
    // a correction the client is expected to ignore. Nothing on the wire.
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            Attribution::default(),
            Attribution::default(),
            Attribution {
                speaker: Some(1),
                relabels: vec![Relabel {
                    from_chunk: 0,
                    to_chunk: 2,
                    speaker: 1,
                }],
                ..Default::default()
            },
        ]),
        SessionTuning {
            lag_chunks: 0,
            relabel_window: 0,
        },
    );
    let mut msgs = Vec::new();
    for _ in 0..3 {
        msgs.extend(push_all(&mut s));
    }
    assert!(relabels(&msgs).is_empty());
    // And the transcript is exactly what it would have been, gaps included.
    assert_eq!(
        sequenced(&msgs)
            .iter()
            .map(|(_, _, sp)| *sp)
            .collect::<Vec<_>>(),
        vec![None, None, Some(1)]
    );
}

#[test]
fn a_correction_cannot_reach_text_that_has_scrolled_out_of_the_window() {
    // Older text is frozen. The window is in seconds, so this session runs at
    // a second per chunk to make three seconds three chunks, and the relabel
    // arriving on chunk 4 names a range starting at chunk 0 -- which by then
    // is outside it.
    const SECOND: usize = 16_000;
    let mut script: Vec<Attribution> = (0..4).map(|_| Attribution::default()).collect();
    script.push(Attribution {
        speaker: Some(1),
        relabels: vec![Relabel {
            from_chunk: 0,
            to_chunk: 4,
            speaker: 1,
        }],
        ..Default::default()
    });
    let backend = MockBackend::new(&WORDS).with_chunk_samples(SECOND);
    let mut s = Session::with_tuning(
        Mode::Transcript,
        &backend,
        "sid".into(),
        scripted(script),
        SessionTuning {
            lag_chunks: 0,
            relabel_window: 3,
        },
    );

    let mut msgs = Vec::new();
    for _ in 0..5 {
        msgs.extend(s.push_audio(&vec![0.0; SECOND]).unwrap());
    }
    // Chunks 0 and 1 are more than three seconds behind chunk 4 and are gone;
    // chunks 2 and 3 are still in reach.
    assert_eq!(relabels(&msgs), vec![(3, 4, 1)]);
    assert_eq!(
        sequenced(&msgs)
            .iter()
            .map(|(_, _, sp)| *sp)
            .collect::<Vec<_>>(),
        vec![None, None, None, None, Some(1)],
        "the commits themselves went out unattributed either way"
    );
}

#[test]
fn corrections_split_around_a_commit_they_may_not_touch() {
    // A correction is one contiguous range of seq per message, so a range
    // interrupted by a commit that is off limits has to become two messages
    // rather than one that quietly includes it.
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            Attribution::default(),
            Attribution::speaker(Some(2)),
            Attribution::default(),
            Attribution {
                speaker: Some(1),
                relabels: vec![Relabel {
                    from_chunk: 0,
                    to_chunk: 3,
                    speaker: 1,
                }],
                ..Default::default()
            },
        ]),
        SessionTuning {
            lag_chunks: 0,
            ..Default::default()
        },
    );
    let mut msgs = Vec::new();
    for _ in 0..4 {
        msgs.extend(push_all(&mut s));
    }
    assert_eq!(relabels(&msgs), vec![(1, 1, 1), (3, 3, 1)]);
}

// ------------------------------------- corrections at the shipped lag depth
//
// Every test above runs at `lag_chunks: 0` and one word per chunk, where a
// commit's first chunk, its last chunk and the end of its vote window are all
// the same number. That collapse is what let a correction be matched against
// the range its *label* was voted over rather than the range its *words* came
// from: at lag 0 the vote window is a single chunk and the two spans coincide.
// At the shipped depth they do not, and the difference is wrong in both
// directions -- the vote window starts where the words end and reaches two
// chunks into whoever spoke next.

/// A session at the shipped lag with commits that span `chunks_per_word`
/// chunks each, which is the shape a real transducer's output has.
fn spanning(
    words: &[&str],
    diarizer: Option<Box<dyn Diarizer>>,
    chunks_per_word: usize,
) -> Session {
    let backend = MockBackend::new(words)
        .with_chunk_samples(CHUNK)
        .with_chunks_per_word(chunks_per_word);
    Session::with_tuning(
        Mode::Transcript,
        &backend,
        "sid".into(),
        diarizer,
        SessionTuning::default(),
    )
}

/// A diarizer that says nothing for `quiet` chunks and then reports one
/// correction.
fn corrects_at(quiet: usize, relabel: Relabel) -> Option<Box<dyn Diarizer>> {
    let mut script: Vec<Attribution> = (0..quiet).map(|_| Attribution::default()).collect();
    script.push(Attribution {
        speaker: Some(relabel.speaker),
        relabels: vec![relabel],
        ..Default::default()
    });
    scripted(script)
}

#[test]
fn a_correction_cannot_claim_the_words_of_the_chunks_before_it() {
    // Three chunks per word at the shipped lag of 2, so commit 1's words come
    // from chunks 0-2 and its label is voted over chunks 2-4. A correction
    // naming chunks 3 onwards is about the *next* commit, whose words are
    // chunks 3-5 -- and it must not reach commit 1, whose vote window happens
    // to extend into the same chunks.
    assert_eq!(LAG_CHUNKS, 2, "the collapse this test avoids returns at 0");
    let mut s = spanning(
        &WORDS,
        corrects_at(
            9,
            Relabel {
                from_chunk: 3,
                to_chunk: 9,
                speaker: 2,
            },
        ),
        3,
    );

    let mut msgs = Vec::new();
    for _ in 0..10 {
        msgs.extend(push_all(&mut s));
    }
    assert_eq!(
        sequenced(&msgs)
            .iter()
            .map(|(seq, t, _)| (*seq, t.clone()))
            .collect::<Vec<_>>(),
        vec![(1, "one ".into()), (2, "two ".into())],
        "the fixture has to have produced two multi-chunk commits"
    );
    assert_eq!(
        relabels(&msgs),
        vec![(2, 2, 2)],
        "the correction reached a commit whose words came from earlier chunks"
    );
}

#[test]
fn a_correction_reaches_a_commit_whose_words_started_before_its_label_did() {
    // The same collapse, from the other side. Commit 1's words come from
    // chunks 0-2; a correction naming chunks 0-1 is about those words, and
    // matching it against the vote window -- which starts at chunk 2 -- misses
    // it entirely. This is the opening of a meeting, which is the case the
    // whole correction mechanism exists for.
    let mut s = spanning(
        &WORDS,
        corrects_at(
            6,
            Relabel {
                from_chunk: 0,
                to_chunk: 1,
                speaker: 1,
            },
        ),
        3,
    );

    let mut msgs = Vec::new();
    for _ in 0..7 {
        msgs.extend(push_all(&mut s));
    }
    assert_eq!(
        relabels(&msgs),
        vec![(1, 1, 1)],
        "the opening commit was never named"
    );
}

#[test]
fn a_commit_that_carries_a_guess_says_so_on_the_wire() {
    // The bit `transcript.relabel`'s promise rests on at the far end. A commit
    // voted entirely out of provisional labels is one a later window may take
    // back; one a full window settled is not, and only the client knows which
    // of its segments is which.
    let guess = |speaker| Attribution {
        speaker: Some(speaker),
        provisional: true,
        ..Default::default()
    };
    let mut s = tuned(
        &WORDS,
        scripted(vec![
            guess(2),
            Attribution::speaker(Some(1)),
            guess(1),
            guess(1),
        ]),
        SessionTuning {
            lag_chunks: 0,
            ..Default::default()
        },
    );
    let mut msgs = Vec::new();
    for _ in 0..4 {
        msgs.extend(push_all(&mut s));
    }
    let flags: Vec<(Option<u32>, bool)> = msgs
        .iter()
        .filter_map(|m| match m {
            ServerMessage::TranscriptCommit {
                speaker,
                speaker_provisional,
                ..
            } => Some((*speaker, *speaker_provisional)),
            _ => None,
        })
        .collect();
    assert_eq!(
        flags,
        vec![
            (Some(2), true),
            (Some(1), false),
            (Some(1), true),
            (Some(1), true)
        ],
        "a guess and a settled label have to be distinguishable on the wire"
    );
}

#[test]
fn an_unconfigured_session_relabels_over_the_shipped_window() {
    // The same rule as the lag depth: a deployment that says nothing must not
    // be able to tell there is now something to say.
    assert_eq!(
        SessionTuning::default(),
        SessionTuning {
            lag_chunks: LAG_CHUNKS,
            relabel_window: RELABEL_WINDOW,
        }
    );
}

#[test]
fn an_unconfigured_session_runs_at_the_calibrated_depth() {
    // The default has to be the behaviour from before the key existed, chunk
    // for chunk: a deployment that says nothing about lag must not be able to
    // tell that there is now something to say.
    let script = [Some(1), Some(2), Some(2)];
    let run = |mut s: Session| (0..3).map(|_| push(&mut s)).collect::<Vec<_>>();
    assert_eq!(
        run(session(&WORDS, diarizer(&script))),
        run(at_lag(&WORDS, diarizer(&script), LAG_CHUNKS))
    );
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
