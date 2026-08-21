//! Diagnostic: load the model on CUDA and hold it, so `nvidia-smi` can be
//! inspected while it runs.
//!
//! parakeet-rs registers the CUDA provider without `error_on_failure`, so a
//! provider that fails to initialise silently degrades to CPU. No unit test can
//! catch that; this probe is how you check.
//!
//! ```text
//! PARAKEET_MODEL_DIR=... ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so \
//!   cargo run -p parakeet-server --features cuda --example gpu_probe
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use parakeet_server::asr::AsrBackend;
    use parakeet_server::asr::parakeet::ParakeetBackend;
    use std::time::Instant;

    // ort logs through `tracing`. Without a subscriber its provider-registration
    // warnings are silently discarded, which is how a CPU fallback hides.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ort=debug,parakeet_rs=debug".into()),
        )
        .init();

    let dir = std::env::var("PARAKEET_MODEL_DIR").expect("set PARAKEET_MODEL_DIR");
    println!("pid = {}", std::process::id());

    let t = Instant::now();
    let backend = ParakeetBackend::load_cuda(std::path::Path::new(&dir))?;
    println!("loaded {} in {:.2}s", backend.model_name(), t.elapsed().as_secs_f32());

    // Run one chunk so the provider actually executes, not just loads.
    let mut s = backend.stream();
    let t = Instant::now();
    let _ = s.push(&vec![0.0; backend.chunk_samples()])?;
    println!("first chunk inference: {:.1}ms", t.elapsed().as_secs_f32() * 1000.0);

    println!("holding for 20s -- check nvidia-smi now");
    std::thread::sleep(std::time::Duration::from_secs(20));
    Ok(())
}
