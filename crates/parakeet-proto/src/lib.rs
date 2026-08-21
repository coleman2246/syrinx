//! Wire protocol for parakeet-stt. One definition, shared by server and clients.
//!
//! This crate deliberately depends on nothing but `serde`, so any client -- a
//! headless typer, a GUI, a future mobile app -- can compile it cheaply. Keeping
//! the protocol in one place is what stops the server and its clients drifting:
//! a breaking change fails the build rather than failing at runtime on the far
//! side of a network.
//!
//! Transport shape: JSON text frames carry control messages, binary frames carry
//! raw PCM. Audio never travels inside JSON.

mod audio;
mod message;
mod mode;

pub use audio::{SAMPLE_RATE, downmix_to_mono, pcm_s16le_to_f32};
pub use message::{ClientMessage, ServerMessage};
pub use mode::{Encoding, ErrorCode, Mode};
