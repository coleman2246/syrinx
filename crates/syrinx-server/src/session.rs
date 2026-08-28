//! Per-session protocol state machine.
//!
//! Deliberately knows nothing about WebSockets: it takes audio samples and
//! returns messages. That is what lets the mode invariants below be tested as
//! plain function calls, with no socket and no GPU.

use crate::asr::{AsrBackend, AsrStream};
use crate::diarize::{Diarizer, Relabel};
use anyhow::Result;
use std::collections::VecDeque;
use syrinx_proto::{Mode, SAMPLE_RATE, ServerMessage};

/// How many chunks a commit is held so its speaker label can settle. The
/// diarizer needs more audio context than the transducer does; calibrated by
/// the spike, which wants a 1.5 s embedding window against 560 ms chunks.
///
/// `pub` because it is the default of the `diarize_lag_chunks` config key,
/// which reads it here rather than repeating the 2 next to a copy of this
/// paragraph. A deployment can tune the key against its own meetings; this is
/// the value the spike measured, and the one every session runs at unless the
/// configuration says otherwise.
pub const LAG_CHUNKS: usize = 2;

/// Consecutive diarizer failures before the session stops asking. An occasional
/// hiccup is survivable; a diarizer that fails every chunk is dead weight on a
/// session that must keep transcribing.
const MAX_DIARIZER_STRIKES: u32 = 5;

/// Seconds of already-emitted transcript that stay eligible for a speaker
/// correction.
///
/// A voice needs four agreeing 1.5 s windows before it is minted, which is
/// roughly 3.7 s of speech, and the text of those seconds has already been
/// committed by then -- unlabelled, and until now unlabellable forever. The
/// session keeps a ring of what it emitted over this many seconds so that
/// `transcript.relabel` can fill those gaps when the name finally arrives.
///
/// 30 s covers the mint delay many times over, which is deliberate: the case
/// it is really sized for is a quiet participant whose first few sentences are
/// spread across half a minute before four windows of them agree.
///
/// `pub` because it is the default of the `diarize_relabel_window` config key,
/// which reads it here rather than repeating the number next to a copy of this
/// paragraph. **An engineering estimate, not a measurement.**
pub const RELABEL_WINDOW: u64 = 30;

/// The session-level diarization settings a deployment can change.
///
/// Two fields rather than two arguments because both are read once, at
/// construction, and a caller that swapped them would compile: they are both
/// small counts, and both are about how long text waits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTuning {
    /// `diarize_lag_chunks`: how many chunks a commit is held so its label can
    /// settle.
    pub lag_chunks: usize,
    /// `diarize_relabel_window`: seconds of emitted transcript still eligible
    /// for a speaker correction. 0 turns corrections off.
    pub relabel_window: u64,
}

impl Default for SessionTuning {
    /// The calibrated and estimated values, read from where each one's
    /// justification lives rather than repeated here.
    fn default() -> Self {
        Self {
            lag_chunks: LAG_CHUNKS,
            relabel_window: RELABEL_WINDOW,
        }
    }
}

/// Text waiting for its speaker label to settle.
struct HeldCommit {
    text: String,
    /// Index of the last chunk that contributed audio to this text.
    chunk: u64,
}

/// What the diarizer said about one chunk.
struct ChunkLabel {
    chunk: u64,
    speaker: Option<u32>,
    /// Whether `speaker` came from a 0.75 s hop's guess rather than a full
    /// window. A commit voted entirely out of guesses is one a later window
    /// may correct; one a window settled is not.
    provisional: bool,
    /// Whether the voice changed at this chunk. The vote stops here.
    boundary: bool,
}

/// A commit already on the wire, and what it would take to correct its
/// speaker.
///
/// Held in a bounded ring covering `relabel_window` seconds, because a
/// speaker's first sentences are committed before four windows have agreed
/// that they are anybody -- and until `transcript.relabel` existed, that
/// attribution was lost for good.
struct Emitted {
    seq: u64,
    /// The chunk range its label was voted over, so an incoming correction can
    /// tell whether it covers this commit.
    from_chunk: u64,
    to_chunk: u64,
    speaker: Option<u32>,
    provisional: bool,
}

pub struct Session {
    mode: Mode,
    stream: Box<dyn AsrStream>,
    seq: u64,
    session_id: String,
    chunk_samples: usize,
    pending: Vec<f32>,
    diarizer: Option<Box<dyn Diarizer>>,
    strikes: u32,
    chunks_seen: u64,
    /// What the diarizer said about each chunk seen. Bounded: labels no held
    /// commit can still consult are dropped as commits leave.
    chunk_labels: VecDeque<ChunkLabel>,
    held: VecDeque<HeldCommit>,
    /// Commits already sent whose speaker may still be corrected. Bounded by
    /// `tuning.relabel_window`, and empty for good when that is 0.
    emitted: VecDeque<Emitted>,
    /// This session's diarization settings. Held rather than read from the
    /// constants so a deployment can tune them; every session built by
    /// [`Session::new`] carries exactly [`SessionTuning::default`].
    tuning: SessionTuning,
}

impl Session {
    /// Build a session at the calibrated lag depth. A `diarizer` makes it a
    /// labelling session, which costs [`LAG_CHUNKS`] of added latency on every
    /// commit.
    ///
    /// `Session` trusts its caller on mode gating: it will label a live-mode
    /// session if handed a diarizer, and must never be handed one, because live
    /// mode types into someone else's application where a speaker label has
    /// nowhere to go and the lag would be latency for nothing. Deciding that
    /// needs to know what the client asked for, so it lives in [`crate::ws`].
    pub fn new(
        mode: Mode,
        backend: &dyn AsrBackend,
        session_id: String,
        diarizer: Option<Box<dyn Diarizer>>,
    ) -> Self {
        Self::with_tuning(
            mode,
            backend,
            session_id,
            diarizer,
            SessionTuning::default(),
        )
    }

    /// The same, at configured settings -- the server's `diarize_lag_chunks`
    /// and `diarize_relabel_window`. `new` delegates here, so they reach the
    /// fields from one place rather than two.
    ///
    /// Neither is consulted without a diarizer, so a session that never asked
    /// for labels is unaffected by any value. A lag of 0 releases every commit
    /// in the call that produced its text, labelled from that chunk alone: the
    /// fastest setting, and the one that leaves the most turn starts
    /// attributed to whoever was speaking before. A relabel window of 0 sends
    /// no corrections, which leaves the opening of a meeting permanently
    /// unattributed -- honest, and what the server did before corrections
    /// existed.
    pub fn with_tuning(
        mode: Mode,
        backend: &dyn AsrBackend,
        session_id: String,
        diarizer: Option<Box<dyn Diarizer>>,
        tuning: SessionTuning,
    ) -> Self {
        Self {
            mode,
            stream: backend.stream(),
            seq: 0,
            session_id,
            chunk_samples: backend.chunk_samples(),
            pending: Vec::new(),
            diarizer,
            strikes: 0,
            chunks_seen: 0,
            chunk_labels: VecDeque::new(),
            held: VecDeque::new(),
            emitted: VecDeque::new(),
            tuning,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Buffer audio and emit messages for every whole chunk now available.
    ///
    /// Audio arrives in network-sized frames that have no relationship to the
    /// model's chunk size, so buffering here is what decouples the two.
    pub fn push_audio(&mut self, audio: &[f32]) -> Result<Vec<ServerMessage>> {
        self.pending.extend_from_slice(audio);
        let mut out = Vec::new();
        while self.pending.len() >= self.chunk_samples {
            let chunk: Vec<f32> = self.pending.drain(..self.chunk_samples).collect();
            out.extend(self.consume_chunk(&chunk)?);
        }
        Ok(out)
    }

    /// Zero-pad and flush any trailing partial chunk, then drain the model.
    pub fn finish(&mut self) -> Result<Vec<ServerMessage>> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let mut chunk: Vec<f32> = std::mem::take(&mut self.pending);
            chunk.resize(self.chunk_samples, 0.0);
            out.extend(self.consume_chunk(&chunk)?);
        }
        let tail = self.stream.finish()?;
        if !tail.is_empty() {
            // Whatever the model was still holding belongs to the last chunk
            // that went into it.
            let chunk = self.chunks_seen.saturating_sub(1);
            self.held.push_back(HeldCommit { text: tail, chunk });
        }
        // The session is over, so no label will settle any further. Flushing
        // unconditionally -- including when there was no partial chunk to push
        // -- is what keeps the lag buffer from swallowing text.
        out.extend(self.release_all());
        Ok(out)
    }

    /// One whole chunk through both models: label first, then text.
    ///
    /// The order is load-bearing. The label for chunk N must be recorded before
    /// any commit ending at chunk N can be released, or the commit would be
    /// labelled from a window with a hole in it.
    fn consume_chunk(&mut self, chunk: &[f32]) -> Result<Vec<ServerMessage>> {
        let heard = self.label(chunk);
        self.chunk_labels.push_back(ChunkLabel {
            chunk: self.chunks_seen,
            speaker: heard.speaker,
            provisional: heard.provisional,
            boundary: heard.boundary,
        });
        self.chunks_seen += 1;

        // Corrections go out before this chunk's own text, so a client sees a
        // commit's speaker fixed before the next commit arrives rather than
        // after it -- and so `seq` still only ever counts upwards.
        let mut out = self.apply_relabels(&heard.relabels);

        let text = self.stream.push(chunk)?;
        if !text.is_empty() {
            self.held.push_back(HeldCommit {
                text,
                chunk: self.chunks_seen - 1,
            });
        }
        out.extend(self.release_ripe());
        Ok(out)
    }

    /// Ask the diarizer who is speaking, or give up on it.
    ///
    /// Only errors are strikes: an `Attribution` with no speaker in it is
    /// honest uncertainty and says nothing about the diarizer's health.
    fn label(&mut self, chunk: &[f32]) -> crate::diarize::Attribution {
        let Some(d) = self.diarizer.as_mut() else {
            return crate::diarize::Attribution::default();
        };
        match d.push(chunk) {
            Ok(a) => {
                self.strikes = 0;
                a
            }
            Err(e) => {
                self.strikes += 1;
                tracing::warn!(
                    "diarizer failed ({} of {MAX_DIARIZER_STRIKES}): {e:#}",
                    self.strikes
                );
                if self.strikes >= MAX_DIARIZER_STRIKES {
                    // Labels are decoration; the transcript is the work. Drop
                    // the decoration, keep the session.
                    tracing::warn!("dropping the diarizer; the session continues unlabelled");
                    self.diarizer = None;
                }
                crate::diarize::Attribution::default()
            }
        }
    }

    /// Turn the diarizer's corrections into `transcript.relabel` messages.
    ///
    /// Three rules, and all three are about *not* rewriting history:
    ///
    /// - only commits still inside the relabel window are eligible, so text
    ///   the reader has scrolled past does not move under them;
    /// - only a commit with no speaker, or one carrying a provisional guess
    ///   this contradicts, is touched. A label a full window settled stands,
    ///   which is what stops a correction reaching out of its own turn and
    ///   putting one person's name on another's sentence;
    /// - a speaker is never renumbered. This names an existing speaker for
    ///   text that had none, and nothing else.
    ///
    /// Contiguous runs of `seq` are coalesced into one message, because that
    /// is the shape the correction really has -- a stretch of one person
    /// talking -- and because a client applying it to a range is doing one
    /// pass rather than one per commit.
    fn apply_relabels(&mut self, relabels: &[Relabel]) -> Vec<ServerMessage> {
        // Pruned here rather than as commits leave, so the window is measured
        // from the chunk the correction arrived on. Pruning on the way out
        // instead would measure it from the last chunk that produced text,
        // which in a quiet meeting is an arbitrary distance behind.
        self.forget_frozen_commits();
        let mut out = Vec::new();
        if self.tuning.relabel_window == 0 {
            return out;
        }
        for r in relabels {
            let mut run: Option<(u64, u64)> = None;
            for e in self.emitted.iter_mut() {
                let overlaps = e.from_chunk <= r.to_chunk && e.to_chunk >= r.from_chunk;
                let correctable =
                    e.speaker.is_none() || (e.provisional && e.speaker != Some(r.speaker));
                if overlaps && correctable {
                    e.speaker = Some(r.speaker);
                    e.provisional = false;
                    run = match run {
                        Some((from, to)) if to + 1 == e.seq => Some((from, e.seq)),
                        Some((from, to)) => {
                            out.push(ServerMessage::TranscriptRelabel {
                                from_seq: from,
                                to_seq: to,
                                speaker: r.speaker,
                            });
                            Some((e.seq, e.seq))
                        }
                        None => Some((e.seq, e.seq)),
                    };
                }
            }
            if let Some((from, to)) = run {
                out.push(ServerMessage::TranscriptRelabel {
                    from_seq: from,
                    to_seq: to,
                    speaker: r.speaker,
                });
            }
        }
        out
    }

    /// Drop commits too old to be corrected.
    ///
    /// Measured in chunks, from the seconds the config names: a chunk is
    /// `chunk_samples` at 16 kHz, and rounding up means the window is never
    /// shorter than what was asked for.
    fn forget_frozen_commits(&mut self) {
        if self.tuning.relabel_window == 0 {
            self.emitted.clear();
            return;
        }
        let per_chunk = self.chunk_samples.max(1) as f64 / SAMPLE_RATE as f64;
        let span = (self.tuning.relabel_window as f64 / per_chunk).ceil() as u64;
        let floor = self.chunks_seen.saturating_sub(span);
        while self.emitted.front().is_some_and(|e| e.to_chunk < floor) {
            self.emitted.pop_front();
        }
    }

    /// Emit every held commit whose lag window is complete.
    fn release_ripe(&mut self) -> Vec<ServerMessage> {
        self.release(false)
    }

    /// Emit everything still held, complete window or not. At the end of a
    /// session a label that never settled costs a label; text left in the
    /// buffer would cost the transcript.
    fn release_all(&mut self) -> Vec<ServerMessage> {
        self.release(true)
    }

    /// Shared body of both: `force` releases regardless of the lag window.
    fn release(&mut self, force: bool) -> Vec<ServerMessage> {
        let mut out = Vec::new();
        let lag = self.lag();
        while let Some(h) = self.held.front() {
            if !force && self.chunks_seen < h.chunk + lag + 1 {
                break;
            }
            let h = self.held.pop_front().expect("front was Some");
            let to = h.chunk + lag;
            let (speaker, provisional) = self.majority_label(h.chunk, to);
            out.push(self.emit(h.text, speaker, h.chunk, to, provisional));
        }
        self.forget_spent_labels();
        out
    }

    /// The lag a commit waits out before release: none without a diarizer,
    /// including after a strike-out, where holding text back would add latency
    /// for a label that is never coming.
    fn lag(&self) -> u64 {
        if self.diarizer.is_some() {
            self.tuning.lag_chunks as u64
        } else {
            0
        }
    }

    /// Drop labels no commit can consult again: those older than the oldest
    /// held commit, or than the current chunk when nothing is held. Without
    /// this, a long silence -- chunks that produce no text, so release nothing
    /// -- would grow the deque for the length of the session.
    fn forget_spent_labels(&mut self) {
        let floor = self
            .held
            .front()
            .map_or_else(|| self.chunks_seen.saturating_sub(1), |h| h.chunk);
        while self.chunk_labels.front().is_some_and(|l| l.chunk < floor) {
            self.chunk_labels.pop_front();
        }
    }

    /// Most frequent Some-label across `[from, to]`, and whether it was voted
    /// entirely out of provisional guesses. Earlier chunks win ties, because
    /// that is where the words actually live. None when nothing is known -- a
    /// gap, never a guess.
    ///
    /// **The vote stops at a turn change.** It used to run to `to`
    /// unconditionally, and both halves of that were wrong at a boundary: the
    /// window reaches into the next speaker's chunks, so the outgoing
    /// speaker's own words can be outvoted by the person who interrupted them,
    /// and the tie-break towards the earliest label hands a tie to whoever was
    /// there first, which at a boundary is the outgoing speaker by
    /// construction. Clipping fixes both at once, because both are the same
    /// mistake: a vote that spans two turns is not a vote about either of
    /// them. The tie-break is kept, and now means what it says -- earliest
    /// *within this turn*.
    ///
    /// A boundary at `from` does not clip, because that is where this turn
    /// starts rather than where it ends.
    fn majority_label(&self, from: u64, to: u64) -> (Option<u32>, bool) {
        // (label, count, first seen, provisional votes)
        let mut counts: Vec<(u32, usize, u64, usize)> = Vec::new();
        for l in &self.chunk_labels {
            if l.chunk < from || l.chunk > to {
                continue;
            }
            if l.boundary && l.chunk > from {
                break;
            }
            let Some(speaker) = l.speaker else { continue };
            match counts.iter_mut().find(|(k, _, _, _)| *k == speaker) {
                Some((_, n, _, p)) => {
                    *n += 1;
                    *p += usize::from(l.provisional);
                }
                None => counts.push((speaker, 1, l.chunk, usize::from(l.provisional))),
            }
        }
        counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2)))
            .map_or((None, false), |(l, n, _, p)| (Some(l), p == n))
    }

    /// All transcript emission funnels through here.
    ///
    /// The ASR is append-only, so every message is a commit today. When a
    /// post-processing layer is added it will emit provisional/revise from this
    /// point, gated on [`Mode::allows_revision`]. Routing every emission through
    /// one function is what makes the live-mode guarantee enforceable in a
    /// single place rather than at every call site.
    ///
    /// It is also where a commit is recorded as correctable. The chunk range
    /// is kept rather than recomputed because it is the range the *label* was
    /// voted over, which is the question an incoming correction asks -- not
    /// the range the text came from, which is narrower.
    fn emit(
        &mut self,
        text: String,
        speaker: Option<u32>,
        from_chunk: u64,
        to_chunk: u64,
        provisional: bool,
    ) -> ServerMessage {
        self.seq += 1;
        if self.tuning.relabel_window > 0 {
            self.emitted.push_back(Emitted {
                seq: self.seq,
                from_chunk,
                to_chunk,
                speaker,
                provisional,
            });
        }
        ServerMessage::TranscriptCommit {
            seq: self.seq,
            text,
            speaker,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::mock::MockBackend;
    use crate::diarize::MockDiarizer;

    #[test]
    fn silence_does_not_grow_the_label_buffer() {
        // A chunk that produces no text releases no commit, and releasing a
        // commit is what drains the labels behind it. A quiet meeting is hours
        // of exactly that, so labels have to be dropped on their own account
        // rather than as a side effect of text leaving.
        //
        // The bound is what matters, not the count: whatever is kept has to be
        // enough for a commit's window and no more. This lives inside the
        // module because the deque is private, and it is private because
        // nothing outside needs it.
        let backend = MockBackend::new(&[]).with_chunk_samples(16);
        let mut s = Session::new(
            Mode::Transcript,
            &backend,
            "sid".into(),
            Some(Box::new(MockDiarizer::labels(&[Some(1)]))),
        );
        for _ in 0..20 {
            assert!(s.push_audio(&[0.0; 16]).unwrap().is_empty());
            assert!(
                s.chunk_labels.len() <= LAG_CHUNKS + 1,
                "{} labels held after {} chunks of silence",
                s.chunk_labels.len(),
                s.chunks_seen
            );
        }
    }
}
