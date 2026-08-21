//! Print every capturable source on this machine, grouped by kind.

fn main() -> anyhow::Result<()> {
    let sources = parakeet_audio::list_sources()?;
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
