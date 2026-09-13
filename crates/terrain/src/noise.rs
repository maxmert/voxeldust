//! ★ THE NOISE — ONE IMPLEMENTATION, in the recipe's kernels (ruling F7, 2026-09-12).
//!
//! The gradient noise, the value noise, the quintic fade, the blend and the corner hash all live in
//! [`vd_recipe::noise`] and [`vd_recipe::rng`], on 64-bit integers at
//! [`vd_recipe::noise::NOISE_BITS`] fraction bits, because that is the arithmetic a GPU and a CPU
//! agree on byte for byte. This module re-exports them at the paths the generator's callers have
//! always used, so there is ONE noise in the world and no second copy to drift.
//!
//! **What the pins say.** `NOISE_PIN` and `VALUE_PIN` below are the integer noise's own words at the
//! home world's own seed: the composition of the hash and the blend is part of the world identity
//! (Format D), so the two literals are the hills' foundation and a change to either opens a world
//! epoch.
//!
//! **Example.** The hill field asks for the noise at the direction of cell (face 2, 1181, 77) times
//! octave 3's frequency. The point falls in lattice cube (417, −88, 1203); its eight corners'
//! gradients come from the hash; the answer is the same word on the Mac, on the pod and on the
//! pilot's graphics card.

pub use vd_recipe::noise::{NOISE_BITS, NOISE_ONE, fade, lerp, noise3, unit_value, value3};
pub use vd_recipe::rng::corner_hash;

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]
    use super::*;
    use vd_recipe::Gi;

    /// A lattice point at the noise's fraction bits from three whole-number eighths.
    fn p(x: i64, y: i64, z: i64) -> [Gi; 3] {
        [
            Gi::new(x) * (NOISE_ONE >> 3),
            Gi::new(y) * (NOISE_ONE >> 3),
            Gi::new(z) * (NOISE_ONE >> 3),
        ]
    }

    #[test]
    fn a_known_vector_pins_the_integer_noise_forever() {
        // (0.25, 0.5, 0.75) — two, four and six eighths — under the home world's universe seed.
        assert_eq!(noise3(2298, p(2, 4, 6)).raw(), NOISE_PIN);
        assert_eq!(value3(2298, p(2, 4, 6)).raw(), VALUE_PIN);
        // The re-exports are the recipe's own kernels, not a copy.
        assert_eq!(fade(NOISE_ONE >> 1), vd_recipe::noise::fade(NOISE_ONE >> 1));
        assert_eq!(lerp(Gi::ZERO, NOISE_ONE, NOISE_ONE >> 1), NOISE_ONE >> 1);
        assert_eq!(unit_value(0), Gi::ZERO);
        assert_eq!(
            corner_hash(1, 2, 3, 4),
            vd_recipe::rng::corner_hash(1, 2, 3, 4)
        );
        assert_eq!(NOISE_BITS, 28);
    }

    const NOISE_PIN: i64 = -165_398_528;
    const VALUE_PIN: i64 = 121_016_942;
}
