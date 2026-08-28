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
//! Per-application capture on Linux works by linking the application's output
//! ports directly into a capture stream's inputs; see [`link`]. Windows lacking
//! it is a real gap, not an oversight: process loopback exists in the Windows
//! API (10 2004+) but cpal does not expose it, and adding it means hand-written
//! WASAPI.

pub mod capture;
pub mod link;
pub mod meter;
pub mod mixer;
pub mod source;

#[cfg(target_os = "linux")]
pub mod pipewire;

pub mod cpal_backend;

pub use capture::Capture;
pub use source::{Source, SourceKind, SourceTarget};

use anyhow::Result;

/// What a caught panic was about, rendered for a person to read.
///
/// The payload is the diagnosis: "could not get endpoint data_flow" says a
/// great deal more than "it panicked", and these messages end up in front of a
/// user rather than only in a log.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "no message".to_string()
}

/// Run `f`, turning a panic inside it into an error.
///
/// cpal's Windows backend does not confine itself to returning errors. It
/// panics on a transient failure at any endpoint, in at least four places this
/// code reaches: `Device::description` calls `.OpenPropertyStore(STGM_READ)
/// .expect("could not open property store")`; `default_input_config` and
/// `default_output_config` reach `.expect("could not query IMMDevice interface
/// for IMMEndpoint")`; `input_devices`/`output_devices` reach a
/// `CoCreateInstance(..).unwrap()`; and `Devices::next` does
/// `.Item(i).unwrap()`, which fails when an endpoint disappears between the
/// collection being snapshotted and the entry being read -- the device-swap
/// case exactly.
///
/// The threads that reach those are the GUI's UI thread every two seconds, the
/// daemon's main loop, and the thread that starts a session. The last is the
/// worst: a panic there kills the thread before it can record a failure, so
/// the session reads `Listening` with no error, Start and Stop do nothing, and
/// the only way back is restarting the daemon.
///
/// Losing one endpoint, or one session, is a far smaller thing than losing the
/// process. This works only where panics unwind; under `panic = "abort"` there
/// is nothing to catch, and nothing here can help.
pub fn caught<T>(what: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    // AssertUnwindSafe because what these closures touch is dropped on the way
    // out, so there is no half-updated state left for a later caller to see.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(p) => anyhow::bail!("{what} panicked: {}", panic_message(p.as_ref())),
    }
}

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
