//! Audio source discovery and capture, shared by the dictation client and GUI.
//!
//! Linux-first and PipeWire-native on purpose. cpal cannot see monitor sources
//! or per-application streams, so it cannot express "transcribe Firefox" or
//! "transcribe whatever is playing" -- both of which are the point.

pub mod capture;
pub mod source;

pub use capture::Capture;
pub use source::{Source, SourceKind, list_sources, parse_sources, resolve};
