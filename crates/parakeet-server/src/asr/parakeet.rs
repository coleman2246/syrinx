//! The real ASR backend: NVIDIA Nemotron cache-aware streaming via parakeet-rs.
//!
//! Feature-gated behind `cuda` so the default build, and therefore CI, needs no
//! GPU stack at all.

use super::{AsrBackend, AsrStream};
use anyhow::{Context, Result, bail};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron, NemotronHandle};
use std::ffi::CString;
use std::path::Path;
use std::time::Instant;
use tracing::info;

/// Samples per inference chunk: 560 ms at 16 kHz.
const CHUNK_SAMPLES: usize = 8960;

/// Zero-chunks pushed at end of stream to drain the decoder's remaining tokens.
const FLUSH_CHUNKS: usize = 3;

/// Measured steady-state inference on an RTX 3060 Ti is ~37 ms per chunk; CPU on
/// a Ryzen 5700X is ~163 ms. Anything above this threshold means we are not on
/// the GPU, whatever the configuration claimed.
const GPU_SANITY_THRESHOLD_MS: f32 = 90.0;

/// Put cuDNN's symbols in the global scope before ONNX Runtime loads its CUDA
/// provider.
///
/// `libonnxruntime_providers_cuda.so` references 8 cuDNN symbols
/// (`cudnnConvolutionBackwardFilter` and friends) but carries **no `DT_NEEDED`
/// entry for cuDNN** -- verified with `objdump -p`. It links cublas, cudart and
/// nccl, but the cuDNN link was dropped, most likely by `--as-needed` at package
/// build time. When ORT `dlopen`s the provider, those symbols are unresolved and
/// registration fails with:
///
/// ```text
/// undefined symbol: cudnnGetConvolutionBackwardDataAlgorithm_v7
/// ```
///
/// ORT then falls back to CPU **silently** -- inference still works, roughly 4x
/// slower, with nothing in the output to say so.
///
/// Loading cuDNN ourselves with `RTLD_GLOBAL` puts those symbols in the global
/// scope, so the provider resolves them when it is loaded. This is the same fix
/// as `LD_PRELOAD=/usr/lib/libcudnn.so.9`, done in-process so deployment does
/// not depend on remembering an environment variable.
///
/// Harmless where cuDNN is linked correctly: the library is simply already
/// loaded.
fn preload_cudnn() -> Result<()> {
    // Try the versioned name first; fall back to the unversioned symlink.
    for name in ["libcudnn.so.9", "libcudnn.so"] {
        let c = CString::new(name).expect("static string has no interior nul");
        // SAFETY: `c` is a valid NUL-terminated C string that outlives the call.
        let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
        if !handle.is_null() {
            info!("preloaded {name} into the global symbol scope for the CUDA provider");
            return Ok(());
        }
    }
    bail!(
        "could not dlopen libcudnn.so.9. ONNX Runtime's CUDA provider needs cuDNN \
         symbols in the global scope and does not link cuDNN itself, so without \
         this it silently falls back to CPU."
    )
}

pub struct ParakeetBackend {
    handle: NemotronHandle,
    model_name: String,
}

impl ParakeetBackend {
    /// Load the model onto the GPU.
    ///
    /// The execution provider is passed **explicitly**. `from_pretrained(path,
    /// None)` resolves to `ExecutionProvider::default()`, which is `Cpu` -- so
    /// omitting it produces a server that works but runs an order of magnitude
    /// too slowly, with no error to indicate why. parakeet-rs also falls back to
    /// CPU if the CUDA provider fails to initialise, so a working-but-slow
    /// server is a real failure mode. Verify with `nvidia-smi` after starting;
    /// no unit test can catch it.
    pub fn load_cuda(dir: &Path) -> Result<Self> {
        preload_cudnn()?;
        let exec = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cuda);
        let handle = NemotronHandle::from_pretrained(dir, Some(exec))
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading Nemotron model from {}", dir.display()))?;
        Ok(Self {
            handle,
            model_name: "nemotron-en-0.6b".to_string(),
        })
    }

    /// Load on CPU. Viable: ~163 ms per 560 ms chunk on a Ryzen 5700X, so it
    /// keeps up in real time, but at roughly a quarter of the GPU's throughput.
    pub fn load_cpu(dir: &Path) -> Result<Self> {
        let exec = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cpu);
        let handle = NemotronHandle::from_pretrained(dir, Some(exec))
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading Nemotron model from {}", dir.display()))?;
        Ok(Self {
            handle,
            model_name: "nemotron-en-0.6b".to_string(),
        })
    }

    /// Time steady-state inference, discarding the warmup chunk.
    ///
    /// The first chunk pays for CUDA context creation and cuDNN autotuning --
    /// ~280 ms against a ~37 ms steady state -- so including it would understate
    /// GPU performance by nearly an order of magnitude.
    pub fn measure_chunk_ms(&self, iterations: usize) -> Result<f32> {
        let mut stream = self.stream();
        let chunk = vec![0.0f32; CHUNK_SAMPLES];
        stream.push(&chunk)?; // warmup, discarded

        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t = Instant::now();
            stream.push(&chunk)?;
            times.push(t.elapsed().as_secs_f32() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).expect("timings are never NaN"));
        Ok(times[times.len() / 2])
    }

    /// Fail if a GPU was requested but inference is running at CPU speed.
    ///
    /// ORT registers the CUDA provider without `error_on_failure`, so a provider
    /// that fails to initialise degrades to CPU with no error anywhere. The
    /// symptom is a server that works but is ~4x slower, which is easy to ship
    /// without noticing. Timing is used rather than querying the provider
    /// because it measures the thing actually cared about, and catches every
    /// cause rather than the one known failure mode.
    pub fn verify_gpu(&self) -> Result<f32> {
        let median = self.measure_chunk_ms(8)?;
        if median > GPU_SANITY_THRESHOLD_MS {
            bail!(
                "GPU was requested but inference took {median:.0}ms per {CHUNK_SAMPLES}-sample \
                 chunk, above the {GPU_SANITY_THRESHOLD_MS:.0}ms threshold. This is CPU-speed: \
                 the CUDA provider almost certainly failed to register and ONNX Runtime fell \
                 back silently. Run the gpu_probe example with RUST_LOG=ort::ep=trace to see why."
            );
        }
        info!("GPU verified: {median:.1}ms per chunk");
        Ok(median)
    }
}

impl AsrBackend for ParakeetBackend {
    /// Spawn an independent stream over the shared model.
    ///
    /// One ~2.5 GB set of weights serves every session; only decoder state is
    /// per-stream. Loading a copy per session would exhaust an 8 GB card at
    /// three clients.
    fn stream(&self) -> Box<dyn AsrStream> {
        Box::new(ParakeetStream {
            inner: Nemotron::from_shared(&self.handle),
        })
    }

    fn chunk_samples(&self) -> usize {
        CHUNK_SAMPLES
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

pub struct ParakeetStream {
    inner: Nemotron,
}

impl AsrStream for ParakeetStream {
    fn push(&mut self, audio: &[f32]) -> Result<String> {
        self.inner
            .transcribe_chunk(audio)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn finish(&mut self) -> Result<String> {
        // The transducer holds tokens until enough context arrives; feeding
        // silence drains them so the tail of an utterance is not lost.
        let mut out = String::new();
        for _ in 0..FLUSH_CHUNKS {
            let text = self
                .inner
                .transcribe_chunk(&vec![0.0; CHUNK_SAMPLES])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.push_str(&text);
        }
        Ok(out)
    }
}
