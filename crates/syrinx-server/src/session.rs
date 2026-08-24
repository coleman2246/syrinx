//! Per-session protocol state machine.
//!
//! Deliberately knows nothing about WebSockets: it takes audio samples and
//! returns messages. That is what lets the mode invariants below be tested as
//! plain function calls, with no socket and no GPU.

use crate::asr::{AsrBackend, AsrStream};
use crate::diarize::Diarizer;
use anyhow::Result;
use std::collections::VecDeque;
use syrinx_proto::{Mode, ServerMessage};

/// How many chunks a commit is held so its speaker label can settle. The
/// diarizer needs more audio context than the transducer does; calibrated by
/// the spike, which wants a 1.5 s embedding window against 560 ms chunks.
const LAG_CHUNKS: usize = 2;

/// Consecutive diarizer failures before the session stops asking. An occasional
/// hiccup is survivable; a diarizer that fails every chunk is dead weight on a
/// session that must keep transcribing.
const MAX_DIARIZER_STRIKES: u32 = 5;

/// Text waiting for its speaker label to settle.
struct HeldCommit {
    text: String,
    /// Index of the last chunk that contributed audio to this text.
    chunk: u64,
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
    /// One label per chunk seen, tagged with its chunk index. Bounded: labels
    /// no held commit can still consult are dropped as commits leave.
    chunk_labels: VecDeque<(u64, Option<u32>)>,
    held: VecDeque<HeldCommit>,
}

impl Session {
    /// Build a session. A `diarizer` makes it a labelling session, which costs
    /// [`LAG_CHUNKS`] of added latency on every commit.
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
        let label = self.label(chunk);
        self.chunk_labels.push_back((self.chunks_seen, label));
        self.chunks_seen += 1;

        let text = self.stream.push(chunk)?;
        if !text.is_empty() {
            self.held.push_back(HeldCommit {
                text,
                chunk: self.chunks_seen - 1,
            });
        }
        Ok(self.release_ripe())
    }

    /// Ask the diarizer who is speaking, or give up on it.
    ///
    /// Only errors are strikes: `Ok(None)` is honest uncertainty and says
    /// nothing about the diarizer's health.
    fn label(&mut self, chunk: &[f32]) -> Option<u32> {
        let d = self.diarizer.as_mut()?;
        match d.push(chunk) {
            Ok(l) => {
                self.strikes = 0;
                l
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
                None
            }
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
            let speaker = self.majority_label(h.chunk, h.chunk + lag);
            out.push(self.emit(h.text, speaker));
        }
        self.forget_spent_labels();
        out
    }

    /// The lag a commit waits out before release: none without a diarizer,
    /// including after a strike-out, where holding text back would add latency
    /// for a label that is never coming.
    fn lag(&self) -> u64 {
        if self.diarizer.is_some() {
            LAG_CHUNKS as u64
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
        while self.chunk_labels.front().is_some_and(|(c, _)| *c < floor) {
            self.chunk_labels.pop_front();
        }
    }

    /// Most frequent Some-label across `[from, to]`; earlier chunks win ties,
    /// because that is where the words actually live. None when nothing is
    /// known -- a gap, never a guess.
    fn majority_label(&self, from: u64, to: u64) -> Option<u32> {
        let mut counts: Vec<(u32, usize, u64)> = Vec::new(); // (label, count, first seen)
        for (c, l) in &self.chunk_labels {
            if (*c >= from && *c <= to)
                && let Some(l) = l
            {
                match counts.iter_mut().find(|(k, _, _)| k == l) {
                    Some((_, n, _)) => *n += 1,
                    None => counts.push((*l, 1, *c)),
                }
            }
        }
        counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2)))
            .map(|(l, _, _)| l)
    }

    /// All transcript emission funnels through here.
    ///
    /// The ASR is append-only, so every message is a commit today. When a
    /// post-processing layer is added it will emit provisional/revise from this
    /// point, gated on [`Mode::allows_revision`]. Routing every emission through
    /// one function is what makes the live-mode guarantee enforceable in a
    /// single place rather than at every call site.
    fn emit(&mut self, text: String, speaker: Option<u32>) -> ServerMessage {
        self.seq += 1;
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
