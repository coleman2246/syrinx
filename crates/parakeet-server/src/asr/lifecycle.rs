//! Model lifecycle: lazy load, idle unload, and the VRAM tenancy guard.
//!
//! The governing rule from the spec: this server is the lowest-priority GPU
//! tenant on its host and must actively yield. The deployment box shares its
//! 2070 Super with Frigate (live camera recording) and Jellyfin (NVENC
//! transcoding), both of which fail visibly when starved. Refusing to load is
//! correct behaviour here, not a failure.

use super::AsrBackend;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Enforces that loading a model leaves at least `floor_mib` free for other
/// tenants on the GPU.
#[derive(Debug, Clone, Copy)]
pub struct VramGuard {
    floor_mib: u64,
}

impl VramGuard {
    pub fn new(floor_mib: u64) -> Self {
        Self { floor_mib }
    }

    /// Whether a model of `model_mib` may be loaded given `free_mib` available.
    ///
    /// Strictly greater-than: landing exactly on the floor leaves a neighbour
    /// with nothing to grow into, and Jellyfin's usage is bursty.
    pub fn can_load(&self, model_mib: u64, free_mib: u64) -> bool {
        free_mib > model_mib + self.floor_mib
    }

    pub fn floor_mib(&self) -> u64 {
        self.floor_mib
    }
}

/// Reports free GPU memory. Abstracted so the policy can be tested without a
/// GPU, and so CPU-only deployments can opt out of the check entirely.
pub trait VramProbe: Send + Sync {
    /// Free VRAM in MiB, or `None` when there is no GPU to speak of.
    fn free_mib(&self) -> Option<u64>;
}

/// Queries `nvidia-smi`. Returns `None` if it is missing or unparseable, which
/// is treated as "no GPU" rather than as an error.
pub struct NvidiaSmiProbe;

impl VramProbe for NvidiaSmiProbe {
    fn free_mib(&self) -> Option<u64> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        String::from_utf8(out.stdout)
            .ok()?
            .lines()
            .next()?
            .trim()
            .parse()
            .ok()
    }
}

/// Always reports the same figure. For tests.
pub struct FixedVramProbe(pub Option<u64>);

impl VramProbe for FixedVramProbe {
    fn free_mib(&self) -> Option<u64> {
        self.0
    }
}

type Loader = Arc<dyn Fn() -> Result<Arc<dyn AsrBackend>> + Send + Sync>;

/// Why a model could not be provided.
///
/// The distinction matters to clients: `Capacity` depends on what the GPU's
/// other tenants are doing right now and is worth retrying, whereas `Failed`
/// means a missing model directory or a binary built without GPU support, which
/// will fail identically forever. Reporting the second as retryable would have
/// clients retry a permanent misconfiguration in a loop.
#[derive(Debug)]
pub enum LoadError {
    /// Loading now would starve another GPU tenant. Transient.
    Capacity(String),
    /// The model could not be loaded at all. Permanent until fixed.
    Failed(anyhow::Error),
}

impl LoadError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, LoadError::Capacity(_))
    }

    pub fn message(&self) -> String {
        match self {
            LoadError::Capacity(m) => m.clone(),
            LoadError::Failed(e) => format!("{e:#}"),
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

/// Loads a model on demand and drops it once idle.
///
/// The server should hold zero VRAM for most of the day. A dictation session is
/// seconds of work in an otherwise quiet 24 hours, and holding 3.4 GB the whole
/// time to save a two-second load is a bad trade against neighbours that fail
/// visibly.
pub struct ModelHandle {
    loaded: Mutex<Option<Arc<dyn AsrBackend>>>,
    last_used: Mutex<Instant>,
    loader: Loader,
    guard: VramGuard,
    probe: Arc<dyn VramProbe>,
    /// Expected footprint, used for the pre-load check. Measured at 3400 MiB
    /// for Nemotron on CUDA: the 2515 MB model file plus ~900 MB of ORT arena.
    model_mib: u64,
    idle_unload: Duration,
}

impl ModelHandle {
    pub fn new(
        loader: Loader,
        guard: VramGuard,
        probe: Arc<dyn VramProbe>,
        model_mib: u64,
        idle_unload: Duration,
    ) -> Self {
        Self {
            loaded: Mutex::new(None),
            last_used: Mutex::new(Instant::now()),
            loader,
            guard,
            probe,
            model_mib,
            idle_unload,
        }
    }

    /// Return the loaded model, loading it if necessary.
    ///
    /// Fails with a capacity-style error when loading would leave too little
    /// VRAM for other tenants. Callers surface that as
    /// `error{code:"capacity", retryable:true}` -- it is transient, since it
    /// depends on what the neighbours are doing right now.
    pub fn get_or_load(&self) -> std::result::Result<Arc<dyn AsrBackend>, LoadError> {
        let mut slot = self.loaded.lock().expect("model lock poisoned");
        *self.last_used.lock().expect("last_used lock poisoned") = Instant::now();

        if let Some(b) = slot.as_ref() {
            return Ok(b.clone());
        }

        // Only meaningful on a GPU deployment. `None` means CPU, where there is
        // no shared VRAM to protect.
        if let Some(free) = self.probe.free_mib()
            && !self.guard.can_load(self.model_mib, free)
        {
            return Err(LoadError::Capacity(format!(
                "refusing to load: {} MiB free, need {} MiB for the model plus a \
                 {} MiB floor reserved for other GPU tenants",
                free,
                self.model_mib,
                self.guard.floor_mib()
            )));
        }

        info!("loading model");
        let backend = (self.loader)().map_err(LoadError::Failed)?;
        *slot = Some(backend.clone());
        Ok(backend)
    }

    /// Drop the model if it has been idle longer than the configured window.
    /// Returns whether it unloaded.
    pub fn unload_if_idle(&self) -> bool {
        let idle = self
            .last_used
            .lock()
            .expect("last_used lock poisoned")
            .elapsed();
        if idle < self.idle_unload {
            return false;
        }
        let mut slot = self.loaded.lock().expect("model lock poisoned");
        if slot.is_some() {
            info!(idle_secs = idle.as_secs(), "unloading idle model");
            *slot = None;
            return true;
        }
        false
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
            .lock()
            .expect("model lock poisoned")
            .is_some()
    }

    /// Background task dropping the model once idle. Runs for the process
    /// lifetime.
    pub async fn run_idle_reaper(self: Arc<Self>, tick: Duration) {
        loop {
            tokio::time::sleep(tick).await;
            if self.unload_if_idle() {
                warn!("model unloaded after idle timeout; next session pays a reload");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::mock::MockBackend;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn handle_with(
        free: Option<u64>,
        model_mib: u64,
        idle: Duration,
        counter: Arc<AtomicUsize>,
    ) -> ModelHandle {
        let loader: Loader = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(MockBackend::new(&["x"])) as Arc<dyn AsrBackend>)
        });
        ModelHandle::new(
            loader,
            VramGuard::new(1536),
            Arc::new(FixedVramProbe(free)),
            model_mib,
            idle,
        )
    }

    #[test]
    fn refuses_to_load_when_free_vram_below_floor() {
        let g = VramGuard::new(1536);
        assert!(!g.can_load(3400, 2000));
    }

    #[test]
    fn allows_load_with_headroom_above_floor() {
        let g = VramGuard::new(1536);
        assert!(g.can_load(3400, 6000));
    }

    #[test]
    fn boundary_is_exclusive_not_inclusive() {
        let g = VramGuard::new(1536);
        assert!(!g.can_load(3400, 3400 + 1536));
        assert!(g.can_load(3400, 3400 + 1536 + 1));
    }

    #[test]
    fn a_zero_floor_still_requires_room_for_the_model_itself() {
        let g = VramGuard::new(0);
        assert!(!g.can_load(3400, 3400));
        assert!(g.can_load(3400, 3401));
    }

    #[test]
    fn model_is_not_loaded_until_first_use() {
        let n = Arc::new(AtomicUsize::new(0));
        let h = handle_with(Some(8000), 3400, Duration::from_secs(600), n.clone());
        assert!(!h.is_loaded());
        assert_eq!(n.load(Ordering::SeqCst), 0, "must not load eagerly");

        h.get_or_load().unwrap();
        assert!(h.is_loaded());
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn repeated_use_loads_only_once() {
        let n = Arc::new(AtomicUsize::new(0));
        let h = handle_with(Some(8000), 3400, Duration::from_secs(600), n.clone());
        h.get_or_load().unwrap();
        h.get_or_load().unwrap();
        h.get_or_load().unwrap();
        assert_eq!(n.load(Ordering::SeqCst), 1, "must reuse the loaded model");
    }

    #[test]
    fn refuses_when_a_neighbour_is_using_the_gpu() {
        // 4000 MiB free, model needs 3400 plus a 1536 floor -> refuse.
        let n = Arc::new(AtomicUsize::new(0));
        let h = handle_with(Some(4000), 3400, Duration::from_secs(600), n.clone());
        // `unwrap_err` would require the Ok type to be Debug, which a trait
        // object is not.
        let err = match h.get_or_load() {
            Ok(_) => panic!("expected a refusal when VRAM is short"),
            Err(e) => e,
        };
        assert!(err.message().contains("refusing to load"), "got: {err}");
        assert!(err.is_retryable(), "VRAM pressure is transient, so retryable");
        assert_eq!(n.load(Ordering::SeqCst), 0, "must not have attempted a load");
    }

    #[test]
    fn a_broken_loader_is_reported_as_permanent_not_retryable() {
        // A missing model directory, or a binary built without GPU support,
        // fails identically forever. Telling a client to retry that would have
        // it loop on a permanent misconfiguration.
        let loader: Loader = Arc::new(|| anyhow::bail!("model directory not found"));
        let h = ModelHandle::new(
            loader,
            VramGuard::new(1536),
            Arc::new(FixedVramProbe(Some(8000))),
            3400,
            Duration::from_secs(600),
        );
        match h.get_or_load() {
            Ok(_) => panic!("expected the load to fail"),
            Err(e) => {
                assert!(!e.is_retryable(), "a broken loader is permanent");
                assert!(e.message().contains("model directory not found"));
            }
        }
    }

    #[test]
    fn no_gpu_skips_the_vram_check_entirely() {
        // CPU deployment: there is no shared VRAM to protect, so a probe
        // returning None must not block loading.
        let n = Arc::new(AtomicUsize::new(0));
        let h = handle_with(None, 3400, Duration::from_secs(600), n.clone());
        assert!(h.get_or_load().is_ok());
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn idle_model_is_unloaded_and_reloaded_on_next_use() {
        let n = Arc::new(AtomicUsize::new(0));
        // Zero idle window: anything already loaded is immediately stale.
        let h = handle_with(Some(8000), 3400, Duration::ZERO, n.clone());
        h.get_or_load().unwrap();
        assert!(h.is_loaded());

        assert!(h.unload_if_idle(), "should have unloaded");
        assert!(!h.is_loaded(), "server must hold zero VRAM when idle");

        h.get_or_load().unwrap();
        assert_eq!(n.load(Ordering::SeqCst), 2, "reload after unload");
    }

    #[test]
    fn a_model_still_within_its_idle_window_is_kept() {
        let n = Arc::new(AtomicUsize::new(0));
        let h = handle_with(Some(8000), 3400, Duration::from_secs(600), n.clone());
        h.get_or_load().unwrap();
        assert!(!h.unload_if_idle(), "keep-warm window must prevent thrash");
        assert!(h.is_loaded());
    }

    #[test]
    fn unloading_when_nothing_is_loaded_is_a_no_op() {
        let n = Arc::new(AtomicUsize::new(0));
        let h = handle_with(Some(8000), 3400, Duration::ZERO, n.clone());
        assert!(!h.unload_if_idle());
    }
}
