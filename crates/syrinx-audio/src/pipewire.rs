//! Linux backend: enumerate the PipeWire graph and capture from any node.
//!
//! The only backend that can target a single application, because
//! per-application audio is a PipeWire concept with no ALSA or cpal equivalent.

use crate::source::{Source, SourceKind, SourceTarget};
use anyhow::{Context, Result};

/// Enumerate capturable sources from `pw-dump` output.
///
/// Split from the subprocess call so it can be tested against recorded JSON
/// without a running PipeWire.
pub fn parse_sources(pw_dump_json: &str) -> Result<Vec<Source>> {
    let objects: Vec<serde_json::Value> =
        serde_json::from_str(pw_dump_json).context("parsing pw-dump output")?;

    let mut out = Vec::new();
    for o in &objects {
        if o.get("type").and_then(|t| t.as_str()) != Some("PipeWire:Interface:Node") {
            continue;
        }
        let props = match o.pointer("/info/props") {
            Some(p) => p,
            None => continue,
        };
        let class = props
            .get("media.class")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let node_name = props
            .get("node.name")
            .and_then(|n| n.as_str())
            .map(str::to_string);
        let Some(id) = o.get("id").and_then(|i| i.as_u64()).map(|i| i as u32) else {
            continue;
        };

        let mut sink_desc: Option<String> = None;
        let (kind, name) = match class {
            // A real capture device, or the monitor of a sink. Monitors are
            // distinguished by node.name rather than media.class, since
            // PipeWire reports both as Audio/Source.
            "Audio/Source" | "Audio/Source/Virtual" => {
                let desc = props
                    .get("node.description")
                    .and_then(|d| d.as_str())
                    .unwrap_or_else(|| node_name.as_deref().unwrap_or("unknown"));
                let is_monitor = node_name
                    .as_deref()
                    .is_some_and(|n| n.ends_with(".monitor"));
                (
                    if is_monitor {
                        SourceKind::Monitor
                    } else {
                        SourceKind::Microphone
                    },
                    desc.to_string(),
                )
            }
            // A sink. Its monitor ports carry everything playing through it,
            // and `pw-record --target <sink>` captures exactly those. Monitors
            // are NOT separate nodes in the PipeWire graph -- pactl synthesises
            // ".monitor" sources from sinks, which is why looking for
            // Audio/Source nodes with a .monitor suffix finds nothing.
            "Audio/Sink" => {
                let desc = props
                    .get("node.description")
                    .and_then(|d| d.as_str())
                    .unwrap_or_else(|| node_name.as_deref().unwrap_or("unknown"));
                // Named for what it captures, not how. "Monitor of X" is the
                // PipeWire mechanism; "everything playing through X" is what
                // the user is choosing.
                sink_desc = Some(desc.to_string());
                (SourceKind::Monitor, format!("Everything playing on {desc}"))
            }
            // An application playing audio. Captured by linking its output
            // ports into a capture stream; see the `link` module for why
            // `--target` does not work here.
            "Stream/Output/Audio" => {
                let app = props
                    .get("application.name")
                    .and_then(|a| a.as_str())
                    .or_else(|| props.get("node.name").and_then(|n| n.as_str()))
                    .unwrap_or("unknown application");
                (SourceKind::Application, app.to_string())
            }
            _ => continue,
        };

        let detail = if kind == SourceKind::Application {
            props
                .get("media.name")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        } else {
            None
        };

        out.push(Source {
            target: SourceTarget::PipeWireNode(id),
            name,
            kind,
            detail,
            sink_description: sink_desc,
            // Applications have no stable identity; devices keep node.name.
            stable_name: match kind {
                SourceKind::Application => None,
                _ => node_name.clone(),
            },
        });
    }

    disambiguate_card_inputs(&mut out);

    // Group by kind for a stable, readable picker, then by name so the order
    // does not jump around as PipeWire renumbers nodes.
    out.sort_by_key(|s| (s.kind, s.name.to_lowercase()));
    Ok(out)
}

/// Mark microphones that share a name with an output.
///
/// A motherboard's line-in jack and its speakers are separate nodes on one
/// card, and PipeWire gives both the same `node.description`. The capture side
/// then appears under Microphone bearing the name of the sound card, which
/// reads as the *output* having been filed under the wrong heading. It has not:
/// it is the capture side of the same device. Saying so is clearer than leaving
/// two identical names to be told apart by which list they are in.
///
/// "(input)" rather than anything more specific, because the collision happens
/// for a motherboard line-in jack and for a USB microphone with a headphone
/// output alike, and only "input" is true of both.
fn disambiguate_card_inputs(sources: &mut [Source]) {
    let sink_descriptions: Vec<String> = sources
        .iter()
        .filter(|s| s.kind == SourceKind::Monitor)
        .filter_map(|s| s.sink_description.clone())
        .collect();

    for s in sources.iter_mut() {
        if s.kind == SourceKind::Microphone && sink_descriptions.contains(&s.name) {
            s.name = format!("{} (input)", s.name);
        }
    }
}

/// Enumerate capturable sources from the running PipeWire daemon.
pub fn list_sources() -> Result<Vec<Source>> {
    let mut sources = list_all_sources()?;
    // Mark the default output, since a machine with several sinks otherwise
    // offers three indistinguishable "everything playing" entries.
    if let Some(default_sink) = default_sink_name() {
        for s in &mut sources {
            if s.kind == SourceKind::Monitor && s.stable_name.as_deref() == Some(&default_sink) {
                s.name = format!("{} (default output)", s.name);
            }
        }
    }
    Ok(sources)
}

/// The current default sink's `node.name`.
fn default_sink_name() -> Option<String> {
    let out = std::process::Command::new("pactl")
        .arg("get-default-sink")
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Alias kept for the diagnostics example.
pub fn list_all_sources() -> Result<Vec<Source>> {
    let out = std::process::Command::new("pw-dump")
        .output()
        .context("running pw-dump (is PipeWire running?)")?;
    if !out.status.success() {
        anyhow::bail!("pw-dump exited with {}", out.status);
    }
    parse_sources(&String::from_utf8_lossy(&out.stdout))
}


// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// A running `pw-record`. Dropping it stops the capture.
///
/// `pw-record` is used as a subprocess rather than binding libpipewire because
/// it already handles graph negotiation, format conversion and resampling, and
/// can target any node -- microphone, sink monitor, or one application's
/// stream. It writes raw PCM to stdout, which is the shape needed here.
///
/// Capturing an application is a **tap**: it keeps playing to its normal output.
/// Nothing is rerouted, so transcribing a video does not silence it.
pub struct PwCapture {
    child: tokio::process::Child,
}

/// Bytes per read. 16 kHz mono s16 is 32 kB/s, so this is ~0.1s of audio:
/// responsive without excessive syscalls.
const READ_CHUNK: usize = 4096;

impl PwCapture {
    /// Capture from a source or sink monitor, addressed by target.
    pub fn start(node_id: u32, tx: tokio::sync::mpsc::Sender<Vec<f32>>) -> Result<Self> {
        Self::start_inner(Some(node_id), tx, None)
    }

    /// Capture one application by linking its output ports into this stream.
    ///
    /// `--target` cannot do this: for a capture stream it means "read from this
    /// source", and an application's playback stream is not a source. See the
    /// `link` module for the measurements.
    pub fn start_linked(app_node: u32, tx: tokio::sync::mpsc::Sender<Vec<f32>>) -> Result<Self> {
        Self::start_inner(None, tx, Some(app_node))
    }

    fn start_inner(
        target: Option<u32>,
        tx: tokio::sync::mpsc::Sender<Vec<f32>>,
        link_from: Option<u32>,
    ) -> Result<Self> {
        use tokio::io::AsyncReadExt;

        // Target 0 when linking manually: pw-record wants a target argument,
        // and 0 leaves the stream unconnected for the link to fill.
        let node_id = target.unwrap_or(0);
        let capture_name = crate::link::unique_capture_name();
        let mut cmd = tokio::process::Command::new("pw-record");
        if link_from.is_some() {
            // pw-record publishes no process id, so give the node a name we can
            // find it by. Without this the link target cannot be identified.
            cmd.env("PIPEWIRE_PROPS", format!("{{ node.name = {capture_name} }}"));
        }
        let mut child = cmd
            .args([
                "--target",
                &node_id.to_string(),
                "--rate",
                &syrinx_proto::SAMPLE_RATE.to_string(),
                "--channels",
                "1",
                "--format",
                "s16",
                // "-" writes raw PCM to stdout, with no WAV header to skip.
                "-",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("spawning pw-record (is PipeWire installed?)")?;

        // Wire the application in once pw-record has registered its node.
        if let Some(app_node) = link_from {
            let mut linked = false;
            for _ in 0..40 {
                std::thread::sleep(std::time::Duration::from_millis(50));
                if let Ok(Some(capture_node)) = crate::link::capture_node_by_name(&capture_name) {
                    match crate::link::link_all(app_node, capture_node) {
                        Ok(n) => {
                            tracing::info!("linked {n} port(s) from node {app_node}");
                            linked = true;
                        }
                        Err(e) => tracing::warn!("linking application audio: {e:#}"),
                    }
                    break;
                }
            }
            if !linked {
                // Without the link the stream yields silence, which would look
                // like a broken microphone rather than a failed link.
                let _ = child.start_kill();
                anyhow::bail!("could not link application node {app_node} into the capture stream");
            }
        }

        let mut stdout = child.stdout.take().context("pw-record stdout missing")?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut s = String::new();
                let mut r = tokio::io::BufReader::new(stderr);
                if r.read_to_string(&mut s).await.is_ok() && !s.trim().is_empty() {
                    tracing::warn!("pw-record: {}", s.trim());
                }
            });
        }

        tokio::spawn(async move {
            // s16 samples can straddle reads, so carry the odd byte rather than
            // decoding half a sample.
            let mut carry: Vec<u8> = Vec::new();
            let mut buf = vec![0u8; READ_CHUNK];
            loop {
                match stdout.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => {
                        carry.extend_from_slice(&buf[..n]);
                        let usable = carry.len() - (carry.len() % 2);
                        let samples = syrinx_proto::pcm_s16le_to_f32(&carry[..usable]);
                        carry.drain(..usable);
                        // try_send: if the consumer is behind, dropping audio
                        // beats growing a backlog and drifting off real time.
                        if tx.try_send(samples).is_err() && tx.is_closed() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("reading from pw-record: {e}");
                        break;
                    }
                }
            }
        });

        Ok(Self { child })
    }
}

impl Drop for PwCapture {
    fn drop(&mut self) {
        // start_kill rather than an await: Drop cannot be async, and leaking
        // pw-record would hold the capture open indefinitely.
        let _ = self.child.start_kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Look a parsed source up by the PipeWire node id it came from.
    fn find(sources: &[Source], id: u32) -> &Source {
        sources
            .iter()
            .find(|s| s.target == SourceTarget::PipeWireNode(id))
            .unwrap_or_else(|| panic!("no source for node {id}"))
    }

    const SAMPLE: &str = r#"[
      {"id": 63, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source",
        "node.name": "alsa_output.usb-Blue.analog-stereo.monitor",
        "node.description": "Monitor of Blue Microphones"}}},
      {"id": 43, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source/Virtual",
        "node.name": "rnnoise_source",
        "node.description": "Yeti (RNNoise)"}}},
      {"id": 105, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Stream/Output/Audio",
        "node.name": "Firefox",
        "application.name": "Firefox"}}},
      {"id": 999, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Sink",
        "node.name": "some-speaker"}}},
      {"id": 111, "type": "PipeWire:Interface:Port", "info": {"props": {}}}
    ]"#;

    #[test]
    fn finds_microphones_monitors_and_applications() {
        let s = parse_sources(SAMPLE).unwrap();
        assert!(s.iter().any(|x| x.kind == SourceKind::Microphone));
        assert!(s.iter().any(|x| x.kind == SourceKind::Monitor));
        assert!(s.iter().any(|x| x.kind == SourceKind::Application));
    }

    #[test]
    fn sinks_are_included_as_monitors_not_ignored() {
        // An earlier version skipped Audio/Sink on the assumption that an
        // output is not capturable. It is: `pw-record --target <sink>` reads
        // the sink's monitor ports, and that is the only route to "transcribe
        // whatever is playing", since no separate monitor node exists.
        let s = parse_sources(SAMPLE).unwrap();
        let sink = find(&s, 999);
        assert_eq!(sink.kind, SourceKind::Monitor);
    }

    #[test]
    fn non_node_objects_are_ignored() {
        // Ports are not nodes and cannot be capture targets.
        let s = parse_sources(SAMPLE).unwrap();
        assert!(!s.iter().any(|x| x.target == SourceTarget::PipeWireNode(111)));
    }

    #[test]
    fn monitors_are_classified_by_node_name_not_media_class() {
        // PipeWire reports a monitor as Audio/Source like any microphone; only
        // the .monitor suffix distinguishes it. Getting this wrong would offer
        // "record your speakers" as if it were a microphone.
        let s = parse_sources(SAMPLE).unwrap();
        let mon = find(&s, 63);
        assert_eq!(mon.kind, SourceKind::Monitor);
    }

    #[test]
    fn virtual_sources_are_treated_as_microphones() {
        // rnnoise_source is Audio/Source/Virtual but is a capture input.
        let s = parse_sources(SAMPLE).unwrap();
        let rn = find(&s, 43);
        assert_eq!(rn.kind, SourceKind::Microphone);
        assert_eq!(rn.name, "Yeti (RNNoise)");
    }

    #[test]
    fn applications_use_their_application_name() {
        let s = parse_sources(SAMPLE).unwrap();
        let ff = find(&s, 105);
        assert_eq!(ff.name, "Firefox");
    }

    #[test]
    fn sinks_are_offered_as_system_audio_monitors() {
        // Monitors are not separate nodes: pactl synthesises ".monitor" sources
        // from sinks, so searching for Audio/Source nodes with that suffix
        // finds nothing and the user gets no way to capture system audio.
        let json = r#"[{"id": 64, "type": "PipeWire:Interface:Node", "info": {"props": {
            "media.class": "Audio/Sink",
            "node.name": "alsa_output.pci-0000_0c_00.4.analog-stereo",
            "node.description": "Starship Analog Stereo"}}}]"#;
        let s = parse_sources(json).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].kind, SourceKind::Monitor);
        assert_eq!(s[0].name, "Everything playing on Starship Analog Stereo");
    }

    #[test]
    fn two_streams_from_one_app_are_distinguishable() {
        // Firefox reports the tab title in media.name. Without it a picker
        // shows "Firefox" twice with no way to tell which is which.
        let json = r#"[
          {"id": 105, "type": "PipeWire:Interface:Node", "info": {"props": {
            "media.class": "Stream/Output/Audio", "application.name": "Firefox",
            "media.name": "Some Video - YouTube"}}},
          {"id": 100, "type": "PipeWire:Interface:Node", "info": {"props": {
            "media.class": "Stream/Output/Audio", "application.name": "Firefox",
            "media.name": "Another Page"}}}]"#;
        let s = parse_sources(json).unwrap();
        assert_eq!(s.len(), 2);
        assert_ne!(s[0].display(), s[1].display());
        assert_ne!(s[0].stable_key(), s[1].stable_key());
        assert!(s.iter().any(|x| x.display().contains("YouTube")));
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(parse_sources("not json").is_err());
    }

    #[test]
    fn nodes_missing_props_are_skipped() {
        let json = r#"[{"id": 1, "type": "PipeWire:Interface:Node"}]"#;
        assert!(parse_sources(json).unwrap().is_empty());
    }
}

