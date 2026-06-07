//! `SplitMix64` — the deterministic RNG for everything stochastic in the harness
//! and (later) seed-derived world decisions. NEVER `thread_rng` (clippy-banned in
//! sim/node): a seed fully determines every draw, which is what makes a failing
//! chaos run replayable byte-for-byte.
//!
//! SplitMix64 (Steele/Lea/Flood) — tiny, fast, well-distributed, and trivially
//! portable: no dependency, no platform variance.

/// A deterministic 64-bit generator. `Clone` forks an identical stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[must_use]
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    /// The canonical SplitMix64 step.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` (53-bit mantissa precision).
    pub fn next_f64(&mut self) -> f64 {
        // Take the top 53 bits — the IEEE-754 double mantissa width.
        #[allow(clippy::cast_precision_loss)] // 53 bits fit a double exactly
        let mantissa = (self.next_u64() >> 11) as f64;
        mantissa / (1u64 << 53) as f64
    }

    /// Bernoulli draw: true with probability `p` (clamped to [0, 1]).
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p.clamp(0.0, 1.0)
    }

    /// Uniform in `[lo, hi)` (empty ranges yield `lo`).
    pub fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo {
            return lo;
        }
        lo + self.next_u64() % (hi - lo)
    }

    /// Deterministic Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let n = items.len() as u64;
        for i in (1..n).rev() {
            let j = self.range_u64(0, i + 1);
            items.swap(
                usize::try_from(i).unwrap_or(usize::MAX),
                usize::try_from(j).unwrap_or(0),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn known_vector_pins_the_algorithm() {
        // SplitMix64 reference: seed 0 -> first output 0xE220A8397B1DCDAF.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn floats_land_in_unit_interval() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..256 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn chance_extremes_are_exact() {
        let mut rng = SplitMix64::new(1);
        assert!(!rng.chance(0.0));
        assert!(rng.chance(1.0));
        // Out-of-range probabilities clamp instead of misbehaving.
        assert!(!rng.chance(-3.0));
        assert!(rng.chance(2.0));
    }

    #[test]
    fn range_handles_empty_and_normal() {
        let mut rng = SplitMix64::new(9);
        assert_eq!(rng.range_u64(5, 5), 5);
        assert_eq!(rng.range_u64(7, 3), 7);
        for _ in 0..64 {
            let v = rng.range_u64(10, 20);
            assert!((10..20).contains(&v));
        }
    }

    #[test]
    fn shuffle_is_deterministic_and_a_permutation() {
        let mut a: Vec<u32> = (0..16).collect();
        let mut b: Vec<u32> = (0..16).collect();
        SplitMix64::new(3).shuffle(&mut a);
        SplitMix64::new(3).shuffle(&mut b);
        assert_eq!(a, b, "same seed, same permutation");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<u32>>());
        // A different seed produces a different permutation (for this size).
        let mut c: Vec<u32> = (0..16).collect();
        SplitMix64::new(4).shuffle(&mut c);
        assert_ne!(a, c);
        // Degenerate inputs are no-ops.
        let mut empty: Vec<u32> = vec![];
        SplitMix64::new(5).shuffle(&mut empty);
        let mut one = vec![1u32];
        SplitMix64::new(5).shuffle(&mut one);
        assert_eq!(one, vec![1]);
    }
}
