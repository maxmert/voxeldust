/// Deterministic seed derivation for the universe hierarchy.
///
/// Uses the splitmix64 finalizer for excellent avalanche properties:
/// every input bit affects every output bit with ~50% probability.
/// Same (parent_seed, index) always produces the same child seed
/// regardless of platform.

/// Core seed derivation: combine a parent seed with an index.
pub fn derive_seed(parent_seed: u64, index: u32) -> u64 {
    let mut h = parent_seed ^ (index as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
    h ^ (h >> 31)
}

/// Derive a galaxy seed from a universe seed and galaxy index.
pub fn derive_galaxy_seed(universe_seed: u64, galaxy_index: u32) -> u64 {
    derive_seed(universe_seed, galaxy_index)
}

/// Derive a star system seed from a galaxy seed and star index.
pub fn derive_system_seed(galaxy_seed: u64, star_index: u32) -> u64 {
    derive_seed(galaxy_seed, star_index)
}

/// Derive a planet seed from a system seed and planet index.
pub fn derive_planet_seed(system_seed: u64, planet_index: u32) -> u64 {
    derive_seed(system_seed, planet_index)
}

/// Extract a deterministic f64 in [0, 1) from a seed.
pub fn seed_to_f64(seed: u64) -> f64 {
    (seed >> 11) as f64 / (1u64 << 53) as f64
}

/// Extract a deterministic f64 in [lo, hi) from a seed.
pub fn seed_to_range(seed: u64, lo: f64, hi: f64) -> f64 {
    lo + seed_to_f64(seed) * (hi - lo)
}

/// Extract a deterministic u32 in [0, max) from a seed.
pub fn seed_to_u32(seed: u64, max: u32) -> u32 {
    (seed % max as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = derive_seed(42, 0);
        let b = derive_seed(42, 0);
        assert_eq!(a, b);

        let c = derive_galaxy_seed(100, 3);
        let d = derive_galaxy_seed(100, 3);
        assert_eq!(c, d);
    }

    #[test]
    fn different_inputs_different_outputs() {
        let a = derive_seed(42, 0);
        let b = derive_seed(42, 1);
        assert_ne!(a, b);

        let c = derive_seed(43, 0);
        assert_ne!(a, c);
    }

    #[test]
    fn avalanche() {
        // Changing one bit in the parent should change ~half of output bits.
        let a = derive_seed(0, 0);
        let b = derive_seed(1, 0);
        let diff = (a ^ b).count_ones();
        // Expect roughly 32 bits different (±10 is generous).
        assert!(diff > 20, "avalanche too low: {diff} bits differ");
        assert!(diff < 50, "avalanche too high: {diff} bits differ");
    }

    #[test]
    fn no_trivial_collisions() {
        // First 1000 seeds from the same parent should all be unique.
        let mut seen = std::collections::HashSet::new();
        for i in 0..1000 {
            let s = derive_seed(42, i);
            assert!(seen.insert(s), "collision at index {i}");
        }
    }

    #[test]
    fn hierarchy_consistency() {
        let universe = 12345u64;
        let galaxy = derive_galaxy_seed(universe, 0);
        let system = derive_system_seed(galaxy, 7);
        let planet = derive_planet_seed(system, 2);

        // Re-derive and verify.
        assert_eq!(derive_planet_seed(derive_system_seed(derive_galaxy_seed(12345, 0), 7), 2), planet);
    }

    #[test]
    fn seed_to_f64_range() {
        for i in 0..100 {
            let s = derive_seed(42, i);
            let f = seed_to_f64(s);
            assert!((0.0..1.0).contains(&f), "f64 out of range: {f}");
        }
    }

    #[test]
    fn seed_to_range_bounds() {
        for i in 0..100 {
            let s = derive_seed(99, i);
            let v = seed_to_range(s, 10.0, 20.0);
            assert!((10.0..20.0).contains(&v), "range out of bounds: {v}");
        }
    }
}
