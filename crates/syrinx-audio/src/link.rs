//! Per-application capture by linking PipeWire ports directly.
//!
//! `pw-record --target <app node>` does not work: PipeWire creates a link but
//! no audio flows. Measured against a controlled 440 Hz tone, targeting an
//! application's node yields rms 26 while the sink monitor carrying the same
//! audio yields 341. A null sink plus `pactl move-sink-input` fails the same
//! way.
//!
//! What does work is linking the application's **output ports** straight into a
//! capture stream's **input ports**. `--target` on a capture stream means "read
//! from this source", and an application's playback stream is not a source; its
//! ports have to be wired in by hand. The same tone measured rms 5636 through a
//! direct port link.
//!
//! Linking is additive, so the application keeps playing to its normal output:
//! transcribing a video does not silence it.

use anyhow::{Context, Result};
use serde::Deserialize;

/// A PipeWire port belonging to a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Port {
    pub id: u32,
    pub node_id: u32,
    /// `out` or `in`.
    pub direction: String,
    pub name: String,
}

#[derive(Deserialize)]
struct RawObject {
    id: u32,
    #[serde(rename = "type")]
    kind: String,
    info: Option<RawInfo>,
}

#[derive(Deserialize)]
struct RawInfo {
    props: Option<serde_json::Value>,
}

/// Parse ports out of `pw-dump` output.
pub fn parse_ports(pw_dump_json: &str) -> Result<Vec<Port>> {
    let objects: Vec<RawObject> =
        serde_json::from_str(pw_dump_json).context("parsing pw-dump output")?;
    let mut out = Vec::new();
    for o in objects {
        if o.kind != "PipeWire:Interface:Port" {
            continue;
        }
        let Some(props) = o.info.and_then(|i| i.props) else {
            continue;
        };
        let node_id = props
            .get("node.id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let direction = props
            .get("port.direction")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let name = props
            .get("port.name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if let (Some(node_id), Some(direction)) = (node_id, direction) {
            out.push(Port {
                id: o.id,
                node_id,
                direction,
                name,
            });
        }
    }
    Ok(out)
}

/// Output ports of a node, in a stable order.
///
/// Sorted by name so a stereo pair is always linked FL then FR rather than in
/// whatever order the graph happens to report.
pub fn output_ports_of(ports: &[Port], node_id: u32) -> Vec<&Port> {
    let mut v: Vec<&Port> = ports
        .iter()
        .filter(|p| p.node_id == node_id && p.direction == "out")
        .collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// Input ports of a node, in a stable order.
pub fn input_ports_of(ports: &[Port], node_id: u32) -> Vec<&Port> {
    let mut v: Vec<&Port> = ports
        .iter()
        .filter(|p| p.node_id == node_id && p.direction == "in")
        .collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// Read the current graph.
pub fn dump() -> Result<Vec<Port>> {
    let out = std::process::Command::new("pw-dump")
        .output()
        .context("running pw-dump")?;
    parse_ports(&String::from_utf8_lossy(&out.stdout))
}

/// Find a capture stream by the unique `node.name` it was launched with.
///
/// Identity comes from a name we choose rather than the process id, because
/// `pw-record` does not publish `application.process.id` -- looking one up
/// silently finds nothing. Several captures can run at once, so the name must
/// be unique or an application's audio could be linked into someone else's
/// stream. See [`unique_capture_name`].
pub fn capture_node_by_name(name: &str) -> Result<Option<u32>> {
    let out = std::process::Command::new("pw-dump")
        .output()
        .context("running pw-dump")?;
    let objects: Vec<serde_json::Value> =
        serde_json::from_slice(&out.stdout).context("parsing pw-dump output")?;
    Ok(find_capture_node(&objects, name))
}

/// A name no other capture will share.
///
/// Process id plus a counter: two captures in one process would otherwise
/// collide, and the pid alone is not enough.
pub fn unique_capture_name() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    format!(
        "syrinx-capture-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

/// Split out for testing without a running PipeWire.
pub fn find_capture_node(objects: &[serde_json::Value], name: &str) -> Option<u32> {
    objects.iter().find_map(|o| {
        if o.get("type").and_then(|t| t.as_str()) != Some("PipeWire:Interface:Node") {
            return None;
        }
        let props = o.pointer("/info/props")?;
        if props.get("media.class").and_then(|c| c.as_str()) != Some("Stream/Input/Audio") {
            return None;
        }
        if props.get("node.name").and_then(|n| n.as_str())? != name {
            return None;
        }
        o.get("id").and_then(|i| i.as_u64()).map(|i| i as u32)
    })
}

/// Link one port to another.
pub fn link(output_port: u32, input_port: u32) -> Result<()> {
    let status = std::process::Command::new("pw-link")
        .arg(output_port.to_string())
        .arg(input_port.to_string())
        .status()
        .context("running pw-link")?;
    if !status.success() {
        anyhow::bail!("pw-link {output_port} -> {input_port} failed with {status}");
    }
    Ok(())
}

/// Wire every output port of `source_node` into the capture node's inputs.
///
/// A stereo application feeding a mono capture links both channels to the one
/// input, which PipeWire mixes -- the same downmix a monitor would give.
pub fn link_all(source_node: u32, capture_node: u32) -> Result<usize> {
    let ports = dump()?;
    let outs = output_ports_of(&ports, source_node);
    let ins = input_ports_of(&ports, capture_node);
    if outs.is_empty() {
        anyhow::bail!("source node {source_node} has no output ports");
    }
    if ins.is_empty() {
        anyhow::bail!("capture node {capture_node} has no input ports");
    }
    let mut n = 0;
    for (i, o) in outs.iter().enumerate() {
        // Round-robin so stereo into mono links both, and mono into stereo
        // feeds both sides rather than leaving one silent.
        let target = ins[i % ins.len()];
        link(o.id, target.id)?;
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PORTS: &str = r#"[
      {"id": 124, "type": "PipeWire:Interface:Port", "info": {"props": {
        "node.id": 123, "port.direction": "out", "port.name": "output_FR"}}},
      {"id": 122, "type": "PipeWire:Interface:Port", "info": {"props": {
        "node.id": 123, "port.direction": "out", "port.name": "output_FL"}}},
      {"id": 114, "type": "PipeWire:Interface:Port", "info": {"props": {
        "node.id": 110, "port.direction": "in", "port.name": "input_MONO"}}},
      {"id": 99, "type": "PipeWire:Interface:Node", "info": {"props": {}}}
    ]"#;

    #[test]
    fn parses_ports_and_ignores_other_objects() {
        let p = parse_ports(PORTS).unwrap();
        assert_eq!(p.len(), 3);
        assert!(!p.iter().any(|x| x.id == 99));
    }

    #[test]
    fn output_ports_come_back_in_channel_order() {
        // Unordered linking would put the right channel on the left.
        let p = parse_ports(PORTS).unwrap();
        let outs = output_ports_of(&p, 123);
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].name, "output_FL");
        assert_eq!(outs[1].name, "output_FR");
    }

    #[test]
    fn input_ports_are_found_for_the_capture_node() {
        let p = parse_ports(PORTS).unwrap();
        assert_eq!(input_ports_of(&p, 110).len(), 1);
    }

    #[test]
    fn a_node_with_no_ports_yields_nothing() {
        let p = parse_ports(PORTS).unwrap();
        assert!(output_ports_of(&p, 4242).is_empty());
    }

    #[test]
    fn a_capture_node_is_matched_by_its_unique_name() {
        // pw-record publishes no process id, so identity has to come from a
        // name we set. Several captures can run at once, and linking into the
        // wrong one would send an application's audio somewhere unrelated.
        let objs: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
              {"id": 200, "type": "PipeWire:Interface:Node", "info": {"props": {
                "media.class": "Stream/Input/Audio", "node.name": "syrinx-capture-1-0"}}},
              {"id": 201, "type": "PipeWire:Interface:Node", "info": {"props": {
                "media.class": "Stream/Input/Audio", "node.name": "syrinx-capture-2-0"}}}
            ]"#,
        )
        .unwrap();
        assert_eq!(find_capture_node(&objs, "syrinx-capture-2-0"), Some(201));
        assert_eq!(find_capture_node(&objs, "syrinx-capture-9-9"), None);
    }

    #[test]
    fn playback_streams_are_not_mistaken_for_captures() {
        // Stream/Output is an application playing audio, not somewhere to send
        // it.
        let objs: Vec<serde_json::Value> = serde_json::from_str(
            r#"[{"id": 300, "type": "PipeWire:Interface:Node", "info": {"props": {
                "media.class": "Stream/Output/Audio", "node.name": "syrinx-capture-1-0"}}}]"#,
        )
        .unwrap();
        assert_eq!(find_capture_node(&objs, "syrinx-capture-1-0"), None);
    }

    #[test]
    fn capture_names_are_unique_within_a_process() {
        // Two sessions in one daemon would otherwise share a name and race for
        // each other's links.
        assert_ne!(unique_capture_name(), unique_capture_name());
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(parse_ports("nonsense").is_err());
    }
}
