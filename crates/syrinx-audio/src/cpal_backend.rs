//! Portable backend for platforms without PipeWire, principally Windows.
//!
//! Enumerates capture devices, plus **output** devices offered as system audio:
//! WASAPI enables loopback when an output device is opened for input, which is
//! how "transcribe whatever is playing" works on Windows.
//!
//! Only half transparent, and the half that is not cost a debugging session:
//! cpal sets the loopback flag when *building* an input stream on a render
//! endpoint, but `default_input_config` still refuses to describe one. The
//! config has to come from the output side. See `CpalCapture::start`.
//!
//! Cannot isolate a single application. Windows has had process loopback since
//! 10 2004, but cpal does not expose it, so that would mean hand-written
//! WASAPI. On Linux the PipeWire backend covers this instead.
//!
//! Compiled on every platform rather than hidden behind `cfg(windows)`, so the
//! code at least type-checks and its tests run during Linux development. Only
//! the loopback *behaviour* is Windows-specific; the calls are portable.

use crate::caught;
use crate::source::{Source, SourceKind, SourceTarget};
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};

/// Run `f`, treating a panic inside it as "no answer".
///
/// For the enumeration path, where one endpoint that will not answer should
/// drop out of the list rather than fail the whole listing. See [`caught`].
fn without_panicking<T>(what: &str, f: impl FnOnce() -> Option<T>) -> Option<T> {
    caught(what, || Ok(f())).unwrap_or_else(|e| {
        // No name to give it: the name is what panicked.
        tracing::warn!("skipping an audio endpoint: {e:#}");
        None
    })
}

fn device_name(d: &cpal::Device) -> Option<String> {
    without_panicking("describing the device", || {
        d.description().ok().map(|x| x.name().to_string())
    })
}

/// Every device on one side of the input/output split.
///
/// Both halves are guarded, because both panic. Getting the enumerator is a
/// `CoCreateInstance(..).unwrap()`, and reading an entry out of the collection
/// is an `.Item(i).unwrap()` that fails when an endpoint disappears between
/// the snapshot and the read.
///
/// A panic while reading an entry ends the enumeration rather than skipping
/// past it: cpal increments its index only *after* the unwrap, so asking again
/// would panic on the same entry for ever. A short list beats a spin.
fn devices(loopback: bool) -> Result<Vec<cpal::Device>> {
    let host = cpal::default_host();
    let mut it = caught("listing the audio devices", || {
        let side: Box<dyn Iterator<Item = cpal::Device>> = if loopback {
            Box::new(host.output_devices().context("enumerating output devices")?)
        } else {
            Box::new(host.input_devices().context("enumerating input devices")?)
        };
        Ok(side)
    })?;

    let mut out = Vec::new();
    loop {
        match caught("reading an audio endpoint", || Ok(it.next())) {
            Ok(Some(d)) => out.push(d),
            Ok(None) => break,
            Err(e) => {
                tracing::warn!("stopping the device listing early: {e:#}");
                break;
            }
        }
    }
    Ok(out)
}

/// Enumerate microphones, and outputs re-offered as system audio.
pub fn list_sources() -> Result<Vec<Source>> {
    let mut out = Vec::new();

    for d in devices(false)? {
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
            sink_description: None,
        });
    }

    for d in devices(true)? {
        let Some(name) = device_name(&d) else { continue };
        let name_for_sink = name.clone();
        out.push(Source {
            target: SourceTarget::CpalDevice {
                name: name.clone(),
                loopback: true,
            },
            name: format!("Everything playing on {name}"),
            kind: SourceKind::Monitor,
            detail: None,
            stable_name: None,
            sink_description: Some(name_for_sink),
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
    devices(loopback)?
        .into_iter()
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

    #[test]
    fn an_endpoint_that_panics_while_being_described_is_skipped_not_fatal() {
        // cpal's Windows backend panics rather than erroring when a property
        // store will not open, and this is reached from the GUI's UI thread
        // every two seconds. The panic message printed by this test is the
        // test doing its job.
        let got: Option<String> = without_panicking("a deliberately broken endpoint", || {
            panic!("could not open property store")
        });
        assert_eq!(got, None, "a panicking endpoint must drop out of the list");
    }

    #[test]
    fn a_panicking_query_becomes_an_error_carrying_what_it_said() {
        // The message is the diagnosis. cpal panics with "could not query
        // IMMDevice interface for IMMEndpoint" and the like, and that reaches
        // a user rather than only a log, so dropping it for "it panicked"
        // would throw away the only thing that says what went wrong. The
        // panic message printed by this test is the test doing its job.
        let e = caught("querying the device", || -> Result<()> {
            panic!("could not get endpoint data_flow")
        })
        .expect_err("a panicking query must not report success");
        let text = format!("{e:#}");
        assert!(text.contains("querying the device"), "{text}");
        assert!(text.contains("could not get endpoint data_flow"), "{text}");
    }

    #[test]
    fn a_query_that_answers_keeps_both_its_value_and_its_error() {
        // The guard must cost neither the ordinary answer nor an ordinary
        // failure, which still has to arrive as the error it already was.
        assert_eq!(caught("fine", || Ok(7)).unwrap(), 7);
        let e = caught("failing", || -> Result<u8> { anyhow::bail!("no device") })
            .expect_err("an ordinary error must survive the guard");
        assert!(format!("{e:#}").contains("no device"));
    }

    #[test]
    fn an_endpoint_that_answers_is_passed_straight_through() {
        // The guard must not cost the ordinary case its answer.
        assert_eq!(
            without_panicking("a working endpoint", || Some("Yeti".to_string())),
            Some("Yeti".to_string())
        );
        assert_eq!(
            without_panicking("an endpoint with no name", || None::<String>),
            None
        );
    }
}
