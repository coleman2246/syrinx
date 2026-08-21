//! Print every capturable source on this machine, grouped by kind.

fn main() -> anyhow::Result<()> {
    // --all also shows application streams, which enumerate but cannot yet be
    // captured.
    let all = std::env::args().any(|a| a == "--all");
    #[cfg(target_os = "linux")]
    let sources = if all {
        parakeet_audio::pipewire::list_all_sources()?
    } else {
        parakeet_audio::list_sources()?
    };
    #[cfg(not(target_os = "linux"))]
    let sources = {
        let _ = all;
        parakeet_audio::list_sources()?
    };
    let mut last = None;
    for s in &sources {
        if last != Some(s.kind) {
            println!("\n{}:", s.kind.label());
            last = Some(s.kind);
        }
        println!("  {:<58} key={}", s.display(), s.stable_key());
    }
    println!("\n{} sources", sources.len());
    Ok(())
}
