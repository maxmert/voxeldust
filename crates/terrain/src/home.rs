//! ★ THE HOME PLANET — THE world's earth-like planet, stated as two literals (SL5: one world; the
//! voxel foundation, slice 5; re-named by ruling V13 L27 on 2026-09-09). The generator may name no
//! motion crate, so it cannot ask the forest which planet is home; it states the planet's seed and
//! the exact bits of its look radius, and `crates/bins/tests/home_body_pin.rs` proves the forest
//! still produces exactly those two numbers and that the census calls the body earth-like. Every
//! unit test of this crate runs on this body, never on an invented one, and the golden gate pins its
//! chunks.
//!
//! **Example.** The home planet is 6 371 km in radius at its look radius (Earth's own, to a
//! kilometre); the ladder snaps it to 6 341 670 m, nineteen rungs, fourteen octaves. A test that wants "a planet"
//! wants this one. The FIRST home planet (seed 7 701 581 858 760 374 086, 3 351 km, airless and
//! hot) was the first body of `System(7)` the ladder accepted, not a chosen world.

use crate::body::BodyDefinition;

/// ★ THE UNIVERSE SEED of THE world (SL5: one world), stated here so the client — which links no motion
/// crate — can fold its DECLARED world tag; `crates/bins/tests/home_body_pin.rs` proves it is the
/// forest's own seed.
pub const HOME_UNIVERSE_SEED: u64 = 2298;

/// The home planet's realm seed, as the forest draws it under the home universe seed
/// (`vd_core::worldgen::HOME_PLANET_SEED`, cross-pinned in `home_body_pin.rs`).
pub const HOME_PLANET_SEED: u64 = 4_030_111_653_607_004_909;
/// The home planet's look radius, bit for bit, as the forest draws it (6 370.7 km).
pub const HOME_PLANET_RADIUS_BITS: u64 = 0x4158_4d6e_d403_3833;

/// The home planet, defined by the recipe.
#[must_use]
pub fn home_planet() -> BodyDefinition {
    BodyDefinition::from_seed(HOME_PLANET_SEED, f64::from_bits(HOME_PLANET_RADIUS_BITS))
        .expect("the home planet is on the ladder")
}

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

    #[test]
    fn the_home_planet_is_on_the_ladder_with_nineteen_rungs() {
        let home = home_planet();
        assert_eq!(home.seed, HOME_PLANET_SEED);
        assert_eq!(home.ladder.rungs, 19);
        assert_eq!(home.octave_count, 14);
        // The ladder snaps Earth's radius to 2N/π at N = 9 961 472 cells: 6 341 670.0 m.
        assert!((home.radius_m() - 6_341_670.0).abs() < 1.0);
    }

    /// MEASURED (the refuter's finding 10): the radius is an INPUT from outside the fence, and the
    /// ladder's snap is what makes a drift in it harmless — a look radius moved by a millimetre, or
    /// by a thousand ulps, gives the SAME body byte for byte, because every number the recipe holds
    /// is a function of the seed and the integer edge count.
    #[test]
    fn a_look_radius_moved_below_the_snap_gives_the_same_body() {
        let r = f64::from_bits(HOME_PLANET_RADIUS_BITS);
        let home = home_planet();
        assert_eq!(
            BodyDefinition::from_seed(HOME_PLANET_SEED, r + 1e-3),
            Some(home)
        );
        assert_eq!(
            BodyDefinition::from_seed(HOME_PLANET_SEED, r - 1e-3),
            Some(home)
        );
        let mut ulps = r;
        let mut i = 0;
        while i < 1_000 {
            ulps = f64::from_bits(ulps.to_bits() + 1);
            i += 1;
        }
        assert_eq!(
            BodyDefinition::from_seed(HOME_PLANET_SEED, ulps),
            Some(home)
        );
        // The snap unit is one top-rung cell edge: a radius moved by half of it can change the body.
        let unit_m = f64::from(1u32 << (home.ladder.rungs - 1)) * std::f64::consts::FRAC_2_PI;
        assert_ne!(
            BodyDefinition::from_seed(HOME_PLANET_SEED, r + unit_m).map(|b| b.ladder.n),
            Some(home.ladder.n),
            "a whole snap unit moves the edge count"
        );
    }
}
