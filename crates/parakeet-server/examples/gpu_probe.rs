//! Diagnostic: load the model, verify the execution provider, and report
//! steady-state inference speed.
//!
//! Exists because no unit test can catch a silent CPU fallback: ORT registers
//! the CUDA provider without `error_on_failure`, so a provider that fails to
//! initialise just runs ~4x slower with no error anywhere.
//!
//! ```text
//! PARAKEET_MODEL_DIR=... ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so \
//!   cargo run -p parakeet-server --features cuda --example gpu_probe [cpu]
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
    // errors are silently discarded, which is how a CPU fallback stays hidden.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "parakeet_server=info,ort::ep=info".into()),
        )
        .init();

    let dir = std::env::var("PARAKEET_MODEL_DIR").expect("set PARAKEET_MODEL_DIR");
    let want_cpu = std::env::args().nth(1).as_deref() == Some("cpu");
    println!("pid = {}", std::process::id());

    let t = Instant::now();
    let path = std::path::Path::new(&dir);
    let backend = if want_cpu {
        ParakeetBackend::load_cpu(path)?
    } else {
        ParakeetBackend::load_cuda(path)?
    };
    println!(
        "loaded {} on {} in {:.2}s",
        backend.model_name(),
        if want_cpu { "CPU" } else { "CUDA" },
        t.elapsed().as_secs_f32()
    );

    let median = backend.measure_chunk_ms(20)?;
    let chunk_ms = backend.chunk_ms() as f32;
    println!("steady state median: {median:.1}ms per {chunk_ms:.0}ms chunk");
    println!(
        "real-time factor:    {:.2}x -> {}",
        median / chunk_ms,
        if median < chunk_ms { "KEEPS UP" } else { "TOO SLOW" }
    );
    println!(
        "concurrent streams:  ~{:.0} before saturating",
        chunk_ms / median
    );

    if !want_cpu {
        match backend.verify_gpu() {
            Ok(ms) => println!("verify_gpu: PASS ({ms:.1}ms)"),
            Err(e) => println!("verify_gpu: FAIL -- {e}"),
        }
    }

    println!("holding 12s -- check nvidia-smi now");
    std::thread::sleep(std::time::Duration::from_secs(12));
    Ok(())
}
