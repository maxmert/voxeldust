//! ★ THE HOME PLANET — THE world's first body, stated as two literals (SL5: one world; the voxel
//! foundation, slice 5). The generator may name no motion crate, so it cannot ask the forest which
//! planet is home; it states the planet's seed and the exact bits of its look radius, and
//! `crates/bins/tests/home_body_pin.rs` proves the forest still produces exactly those two numbers.
//! Every unit test of this crate runs on this body, never on an invented one, and the golden gate
//! pins its chunks.
//!
//! **Example.** The home planet is 3 351 km across at its look radius; the ladder snaps it to
//! 3 350 759 m, twelve rungs, fourteen octaves. A test that wants "a planet" wants this one.

use crate::body::BodyDefinition;

/// The home planet's realm seed, as the forest draws it under the home universe seed.
pub const HOME_PLANET_SEED: u64 = 7_701_581_858_760_374_086;
/// The home planet's look radius, bit for bit, as the forest draws it.
pub const HOME_PLANET_RADIUS_BITS: u64 = 0x4149_9139_1e69_2dfa;

/// The home planet, defined by the recipe.
#[must_use]
pub fn home_planet() -> BodyDefinition {
    BodyDefinition::from_seed(HOME_PLANET_SEED, f64::from_bits(HOME_PLANET_RADIUS_BITS))
        .expect("the home planet is on the ladder")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_home_planet_is_on_the_ladder_with_twelve_rungs() {
        let home = home_planet();
        assert_eq!(home.seed, HOME_PLANET_SEED);
        assert_eq!(home.ladder.rungs, 12);
        assert_eq!(home.octave_count, 14);
        assert!((home.radius_m - crate::gf::Gf::from_f64(3_350_759.0)).abs() < crate::gf::Gf::ONE);
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
