//! Audio source discovery and capture, shared by the dictation client and GUI.
//!
//! Two backends, chosen at compile time:
//!
//! - **PipeWire** (Linux) enumerates the graph with `pw-dump` and captures with
//!   `pw-record`. This is the only backend that can target a single
//!   application's stream, because per-application audio is a PipeWire concept
//!   with no ALSA or cpal equivalent.
//! - **cpal** (Windows, and anywhere else) enumerates input devices, plus output
//!   devices used as inputs -- which WASAPI transparently turns into loopback,
//!   giving system audio. It cannot isolate a single application.
//!
//! What each platform can capture:
//!
//! | | Linux | Windows |
//! |---|---|---|
//! | Microphone | yes | yes |
//! | System audio | yes (sink monitors) | yes (WASAPI loopback) |
//! | One application | yes | no |
//!
//! Windows lacking per-application capture is a real gap, not an oversight:
//! process loopback exists in the Windows API (10 2004+) but cpal does not
//! expose it, and adding it means hand-written WASAPI.

pub mod capture;
pub mod source;

#[cfg(target_os = "linux")]
pub mod pipewire;

pub mod cpal_backend;

pub use capture::Capture;
pub use source::{Source, SourceKind, SourceTarget};

use anyhow::Result;

/// Enumerate everything capturable on this machine.
///
/// Ordered by kind so a picker groups naturally: microphones, then system
/// audio, then individual applications.
pub fn list_sources() -> Result<Vec<Source>> {
    #[cfg(target_os = "linux")]
    {
        pipewire::list_sources()
    }
    #[cfg(not(target_os = "linux"))]
    {
        cpal_backend::list_sources()
    }
}

/// Re-resolve a remembered source. Node ids and device indices are not stable
/// across restarts, so a saved choice is stored as [`Source::stable_key`] and
/// looked up again here.
pub fn resolve(sources: &[Source], stable_key: &str) -> Option<Source> {
    sources
        .iter()
        .find(|s| s.stable_key() == stable_key)
        .cloned()
}
