//! THE INTEGER HASH — SplitMix64's step and the corner hash, the one implementation every draw of
//! the world comes from. `vd_seed::rng::SplitMix64` wraps [`splitmix_step`] (it adds float
//! conveniences for the harness on top); the recipe's kernels call [`corner_hash`] directly. The
//! GPU spike measured this exact arithmetic on 3 936 256 corners against the CPU: 0 differ.

/// SplitMix64's increment.
pub const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// SplitMix64's first mixing multiplier.
pub const MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;
/// SplitMix64's second mixing multiplier.
pub const MIX_2: u64 = 0x94D0_49BB_1331_11EB;

/// The avalanche of one word: SplitMix64's mixing function.
#[must_use]
pub const fn mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(MIX_1);
    let z = (z ^ (z >> 27)).wrapping_mul(MIX_2);
    z ^ (z >> 31)
}

/// One SplitMix64 step from `state`: `(the next state, the output)`.
#[must_use]
pub const fn splitmix_step(state: u64) -> (u64, u64) {
    let next = state.wrapping_add(GOLDEN);
    (next, mix(next))
}

/// The first output of a SplitMix64 seeded with `seed`: one round of the hash.
#[must_use]
pub const fn hash1(seed: u64) -> u64 {
    splitmix_step(seed).1
}

/// The hash of one lattice corner: the seed and the three coordinates folded into one key with
/// SplitMix64's own three constants (so no second hash enters the tree), then ONE round.
#[must_use]
pub const fn corner_hash(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    let key = seed
        ^ (x as u64).wrapping_mul(GOLDEN)
        ^ (y as u64).wrapping_mul(MIX_1)
        ^ (z as u64).wrapping_mul(MIX_2);
    hash1(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_step_is_the_reference_splitmix64() {
        // SplitMix64 reference: seed 0 → 0xE220A8397B1DCDAF, then 0x6E789E6AA1B965F4.
        let (state, out) = splitmix_step(0);
        assert_eq!(out, 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix_step(state).1, 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(hash1(0), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn the_corner_hash_folds_every_coordinate_and_the_seed() {
        let h = corner_hash(1, 2, 3, 4);
        assert_eq!(h, corner_hash(1, 2, 3, 4));
        assert_ne!(h, corner_hash(2, 2, 3, 4));
        assert_ne!(h, corner_hash(1, 3, 3, 4));
        assert_ne!(h, corner_hash(1, 2, 4, 4));
        assert_ne!(h, corner_hash(1, 2, 3, 5));
        assert_ne!(corner_hash(0, -1, 0, 0), corner_hash(0, 1, 0, 0));
        // The fold is the documented one.
        let key =
            1u64 ^ 2u64.wrapping_mul(GOLDEN) ^ 3u64.wrapping_mul(MIX_1) ^ 4u64.wrapping_mul(MIX_2);
        assert_eq!(h, hash1(key));
    }
}
