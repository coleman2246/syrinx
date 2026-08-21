//! Per-session protocol state machine.
//!
//! Deliberately knows nothing about WebSockets: it takes audio samples and
//! returns messages. That is what lets the mode invariants below be tested as
//! plain function calls, with no socket and no GPU.

use crate::asr::{AsrBackend, AsrStream};
use anyhow::Result;
use syrinx_proto::{Mode, ServerMessage};

pub struct Session {
    mode: Mode,
    stream: Box<dyn AsrStream>,
    seq: u64,
    session_id: String,
    chunk_samples: usize,
    pending: Vec<f32>,
}

impl Session {
    pub fn new(mode: Mode, backend: &dyn AsrBackend, session_id: String) -> Self {
        Self {
            mode,
            stream: backend.stream(),
            seq: 0,
            session_id,
            chunk_samples: backend.chunk_samples(),
            pending: Vec::new(),
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
            let text = self.stream.push(&chunk)?;
            if !text.is_empty() {
                out.push(self.emit(text));
            }
        }
        Ok(out)
    }

    /// Zero-pad and flush any trailing partial chunk, then drain the model.
    pub fn finish(&mut self) -> Result<Vec<ServerMessage>> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let mut chunk: Vec<f32> = std::mem::take(&mut self.pending);
            chunk.resize(self.chunk_samples, 0.0);
            let text = self.stream.push(&chunk)?;
            if !text.is_empty() {
                out.push(self.emit(text));
            }
        }
        let tail = self.stream.finish()?;
        if !tail.is_empty() {
            out.push(self.emit(tail));
        }
        Ok(out)
    }

    /// All transcript emission funnels through here.
    ///
    /// The ASR is append-only, so every message is a commit today. When a
    /// post-processing layer is added it will emit provisional/revise from this
    /// point, gated on [`Mode::allows_revision`]. Routing every emission through
    /// one function is what makes the live-mode guarantee enforceable in a
    /// single place rather than at every call site.
    fn emit(&mut self, text: String) -> ServerMessage {
        self.seq += 1;
        ServerMessage::TranscriptCommit {
            seq: self.seq,
            text,
        }
    }
}
