//! Portable backend for platforms without PipeWire, principally Windows.
//!
//! Enumerates capture devices, plus **output** devices offered as system audio:
//! WASAPI transparently enables loopback when an output device is opened for
//! input, which is how "transcribe whatever is playing" works on Windows.
//!
//! Cannot isolate a single application. Windows has had process loopback since
//! 10 2004, but cpal does not expose it, so that would mean hand-written
//! WASAPI. On Linux the PipeWire backend covers this instead.
//!
//! Compiled on every platform rather than hidden behind `cfg(windows)`, so the
//! code at least type-checks and its tests run during Linux development. Only
//! the loopback *behaviour* is Windows-specific; the calls are portable.

use crate::source::{Source, SourceKind, SourceTarget};
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};

fn device_name(d: &cpal::Device) -> Option<String> {
    d.description().ok().map(|x| x.name().to_string())
}

/// Enumerate microphones, and outputs re-offered as system audio.
pub fn list_sources() -> Result<Vec<Source>> {
    let host = cpal::default_host();
    let mut out = Vec::new();

    for d in host.input_devices().context("enumerating input devices")? {
        let Some(name) = device_name(&d) else { continue };
        out.push(Source {
            target: SourceTarget::CpalDevice {
                name: name.clone(),
                loopback: false,
            },
            name,
            kind: SourceKind::Microphone,
            detail: None,
            // cpal has no id beyond the name, so the name is the stable key.
            stable_name: None,
        });
    }

    for d in host.output_devices().context("enumerating output devices")? {
        let Some(name) = device_name(&d) else { continue };
        out.push(Source {
            target: SourceTarget::CpalDevice {
                name: name.clone(),
                loopback: true,
            },
            name: format!("Monitor of {name}"),
            kind: SourceKind::Monitor,
            detail: None,
            stable_name: None,
        });
    }

    // Give every entry a stable key derived from its target, since cpal offers
    // no identifier of its own and a picker must survive a restart.
    for s in &mut out {
        if let SourceTarget::CpalDevice { name, loopback } = &s.target {
            s.stable_name = Some(format!(
                "cpal:{}:{name}",
                if *loopback { "out" } else { "in" }
            ));
        }
    }

    out.sort_by_key(|s| (s.kind, s.name.to_lowercase()));
    Ok(out)
}

/// Find a cpal device by name, on the correct side of the input/output split.
pub fn find_device(name: &str, loopback: bool) -> Result<cpal::Device> {
    let host = cpal::default_host();
    let mut devices: Box<dyn Iterator<Item = cpal::Device>> = if loopback {
        Box::new(host.output_devices().context("enumerating output devices")?)
    } else {
        Box::new(host.input_devices().context("enumerating input devices")?)
    };
    devices
        .find(|d| device_name(d).as_deref() == Some(name))
        .with_context(|| format!("no such audio device: {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_does_not_error_on_this_machine() {
        // Backends differ wildly; the contract is only that listing succeeds
        // and never panics, even where no device exists.
        let r = list_sources();
        assert!(r.is_ok(), "listing failed: {:?}", r.err());
    }

    #[test]
    fn every_source_has_a_stable_key() {
        // A picker that cannot persist a choice is a picker the user has to
        // redo on every launch.
        for s in list_sources().unwrap() {
            assert!(!s.stable_key().is_empty(), "empty key for {}", s.name);
        }
    }

    #[test]
    fn outputs_are_offered_as_monitors_and_inputs_as_microphones() {
        for s in list_sources().unwrap() {
            match &s.target {
                SourceTarget::CpalDevice { loopback: true, .. } => {
                    assert_eq!(s.kind, SourceKind::Monitor)
                }
                SourceTarget::CpalDevice {
                    loopback: false, ..
                } => assert_eq!(s.kind, SourceKind::Microphone),
                other => panic!("cpal backend produced a non-cpal target: {other:?}"),
            }
        }
    }

    #[test]
    fn a_missing_device_is_an_error_not_a_panic() {
        assert!(find_device("no such device exists anywhere", false).is_err());
    }
}
