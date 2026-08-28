//! Capture from a named source for a few seconds and report the level.
//!
//! Verifies a source really produces audio, rather than trusting that a link
//! was made.
//!
//! ```text
//! cargo run -p syrinx-audio --example tap -- <substring of the source name>
//! ```
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("syrinx_audio=info")
        .init();
    let want = std::env::args().nth(1).unwrap_or_default().to_lowercase();
    let sources = syrinx_audio::list_sources()?;
    let src = sources
        .iter()
        .find(|s| s.display().to_lowercase().contains(&want))
        .ok_or_else(|| anyhow::anyhow!("no source matching {want:?}"))?;
    println!("capturing: {} ({:?})", src.display(), src.kind);

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let _cap = syrinx_audio::Capture::start(src, tx)?;

    let mut all: Vec<f32> = Vec::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        if let Ok(Some(chunk)) =
            tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
        {
            all.extend(chunk);
        }
    }
    let rms = if all.is_empty() {
        0.0
    } else {
        (all.iter().map(|s| s * s).sum::<f32>() / all.len() as f32).sqrt()
    };
    println!(
        "samples={} rms={:.4} -> {}",
        all.len(),
        rms,
        if rms > 0.01 {
            "AUDIO CAPTURED"
        } else {
            "silent"
        }
    );
    Ok(())
}
