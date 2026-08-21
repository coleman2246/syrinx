//! Model lifecycle: lazy load, idle unload, and the VRAM tenancy guard.
//!
//! The governing rule from the spec: this server is the lowest-priority GPU
//! tenant on its host and must actively yield. The deployment box shares its
//! 2070 Super with Frigate (live camera recording) and Jellyfin (NVENC
//! transcoding), both of which fail visibly when starved. Refusing to load is
//! correct behaviour here, not a failure.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_to_load_when_free_vram_below_floor() {
        let g = VramGuard::new(1536);
        // model needs 2515 MiB, only 2000 MiB free -> must refuse
        assert!(!g.can_load(2515, 2000));
    }

    #[test]
    fn allows_load_with_headroom_above_floor() {
        let g = VramGuard::new(1536);
        assert!(g.can_load(2515, 6000));
    }

    #[test]
    fn boundary_is_exclusive_not_inclusive() {
        let g = VramGuard::new(1536);
        // exactly at the floor after loading is NOT acceptable
        assert!(!g.can_load(2515, 2515 + 1536));
        assert!(g.can_load(2515, 2515 + 1536 + 1));
    }

    #[test]
    fn a_zero_floor_still_requires_room_for_the_model_itself() {
        let g = VramGuard::new(0);
        assert!(!g.can_load(2515, 2515));
        assert!(g.can_load(2515, 2516));
    }
}
