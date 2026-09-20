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

use crate::artifact::GoldenFields;
use crate::body::{BodyDefinition, BodyFacts};

/// ★ THE UNIVERSE SEED of THE world (SL5: one world), stated here so the client — which links no motion
/// crate — can fold its DECLARED world tag; `crates/bins/tests/home_body_pin.rs` proves it is the
/// forest's own seed.
pub const HOME_UNIVERSE_SEED: u64 = 2298;

/// ★ THE GOLDEN FIELDS' TEXT (slice 8c stage C4c): the home artifact's rows under the six rung-0
/// self-check keys and its coarsest pyramid level, recorded by `cargo run --release -p vd-bins
/// --example golden_z_record` after a deliberate change of the solve, and proved to be the solve's
/// own rows by `crates/bins/tests/home_artifact_pin.rs`.
const HOME_GOLDEN_Z: &str = include_str!("../tests/golden_home_z.txt");

/// ★ THE MEASURED HALF OF THE HOME WORLD'S IDENTITY (slice 8c stage C4c): the eight self-check
/// chunks read through the golden fields, folded. Recorded with the fields; a host that folds another
/// word has drifted, and its world hello is refused by name.
pub const HOME_IDENTITY_MEASURED: u64 = 0x9d23_0f49_bf93_67e8;

/// ★ THE HOME PLANET'S SEA (slice 8c stage C5; gate G-SEA): the level the solve finds for the
/// planet's water inventory over its eroded field, whole metres over the ladder radius, and the
/// share of the globe under it in 1/10 000. Recorded by the artifact pin; a change is a change of
/// the solve and is re-recorded on purpose.
pub const HOME_PLANET_SEA_M: i32 = 4_455;
pub const HOME_PLANET_OCEAN_SHARE_Q4: u32 = 8_995;

/// The golden fields, parsed from the literal in the build. A torn literal is a defect, not a
/// refusal: the build carries it, so it panics here and never ships.
#[must_use]
pub fn home_golden_fields() -> GoldenFields {
    GoldenFields::parse(HOME_GOLDEN_Z).expect("the golden fields' text is whole in the build")
}

/// The home planet's realm seed, as the forest draws it under the home universe seed
/// (`vd_core::worldgen::HOME_PLANET_SEED`, cross-pinned in `home_body_pin.rs`).
pub const HOME_PLANET_SEED: u64 = 4_030_111_653_607_004_909;
/// The home planet's look radius, bit for bit, as the forest draws it (6 370.7 km).
pub const HOME_PLANET_RADIUS_BITS: u64 = 0x4158_4d6e_d403_3833;

/// ★ THE HOME PLANET'S OWN CHARTER WORDS, stated here as literals for exactly the reason the seed
/// and the radius are (slice 8b stage 3; the design's §5.3): the generator may name no motion crate,
/// and the client's binary folds its DECLARED world tag before it connects to anything, so for the
/// home body there is no author to state a charter yet. `crates/bins/tests/home_body_pin.rs`
/// (`charter_pin`) proves the forest's own draw is exactly these two numbers, so the golden table
/// and THE world can never part company in silence.
///
/// 9.818 m/s². The design's §2.1 wrote 9 821 by hand from L27's prose; the forest draws 9 818, and
/// the forest wins.
pub const HOME_PLANET_GRAVITY_MM_S2: u32 = 9_818;
/// 5 513 kg/m³ — Earth's own 5 514 to a part in five thousand.
pub const HOME_PLANET_BULK_DENSITY_KGM3: u32 = 5_513;

/// ★ THE HOME PLANET'S MOON (the landform arc, slice 8c stage C1): the smallest round body the
/// home system holds that the ladder accepts — 353 km, airless, tidally locked — stated as the
/// same literals the planet is, so the solve's DRIVER test runs on a REAL SMALL BODY OF THE WORLD
/// (06 §3.3: not a variant, not a test world) in milliseconds, and `home_body_pin.rs` (`moon_pin`)
/// proves the forest draws exactly these numbers. The design counted on a 50 km body of 384
/// nodes; the home system holds none that small, and its moon's lattice is 68 nodes an edge,
/// 27 744 nodes (MEASURED 2026-09-19).
pub const HOME_MOON_SEED: u64 = 2_918_819_812_335_288_845;
/// The moon's look radius, bit for bit, as the forest draws it (353.0 km).
pub const HOME_MOON_RADIUS_BITS: u64 = 0x4115_8bc0_1ca5_d6aa;
/// 0.330 m/s² and 3 344 kg/m³ — the forest's own charter words for the moon.
pub const HOME_MOON_GRAVITY_MM_S2: u32 = 330;
pub const HOME_MOON_BULK_DENSITY_KGM3: u32 = 3_344;

/// ★ THE HOME SYSTEM'S AGE in years (the landform arc, slice 8c; the design's ask 3): the erosional
/// age the solve steps through is the SYSTEM'S OWN age from the census — a derived fact, never a
/// typed dial — and the generator, which may name no motion crate, states the census's number
/// here; `home_body_pin.rs` proves the census still says it.
pub const HOME_SYSTEM_AGE_YR: u64 = 5_000_000_000;

/// ★ THE HOME PLANET'S LAND WORDS (slice 8c stage C2): the two charter words the initial land
/// reads — the water inventory in whole km³ and the lithosphere's elastic thickness in whole
/// metres — as the census computes them (`home_body_pin.rs` proves it), stated here because the
/// generator may name no motion crate and the driver test needs them.
pub const HOME_PLANET_WATER_KM3: u64 = 2_735_928_089;
pub const HOME_PLANET_ELASTIC_THICKNESS_M: u32 = 36_789;
/// The moon's: no water, and a lithosphere three times its own radius thick — a stagnant lid.
pub const HOME_MOON_WATER_KM3: u64 = 0;
pub const HOME_MOON_ELASTIC_THICKNESS_M: u32 = 1_094_578;

/// ★ THE HOME PLANET'S SOLVE WORDS (slice 8c stage C3): every charter word the solve reads, as the
/// census computes them — `home_body_pin.rs` (`solve_words_pin`) proves each one.
#[must_use]
pub const fn home_solve_words() -> crate::solve::SolveWords {
    crate::solve::SolveWords {
        water_km3: HOME_PLANET_WATER_KM3,
        elastic_thickness_m: HOME_PLANET_ELASTIC_THICKNESS_M,
        insolation_q12: 3_065,
        t_eq_mk: 236_785,
        t_surface_mk: Some(288_000),
        bond_albedo_q12: 1_228,
        mu_q8: Some(7_168),
        scale_height_m: Some(7_160),
        p_surf_pa: Some(101_409),
        tau_ir_q12: Some(6_490),
        day_s: Some(259_597),
        obliquity_cos_q1024: Some(586),
        ecc_q16: 1_017,
        year_s: 34_727_239,
        flags: 1_027,
        age_yr: HOME_SYSTEM_AGE_YR,
    }
}

/// The moon's solve words: airless, tidally locked, dry.
#[must_use]
pub const fn home_moon_solve_words() -> crate::solve::SolveWords {
    crate::solve::SolveWords {
        water_km3: HOME_MOON_WATER_KM3,
        elastic_thickness_m: HOME_MOON_ELASTIC_THICKNESS_M,
        insolation_q12: 3_065,
        t_eq_mk: 252_139,
        t_surface_mk: Some(252_139),
        bond_albedo_q12: 409,
        mu_q8: None,
        scale_height_m: None,
        p_surf_pa: None,
        tau_ir_q12: Some(0),
        day_s: Some(54_899),
        obliquity_cos_q1024: Some(972),
        ecc_q16: 565,
        year_s: 54_899,
        flags: 1_030,
        age_yr: HOME_SYSTEM_AGE_YR,
    }
}

/// The home planet's land words, as its realm states them.
#[must_use]
pub const fn home_land_words() -> crate::land::LandWords {
    crate::land::LandWords {
        water_km3: HOME_PLANET_WATER_KM3,
        elastic_thickness_m: HOME_PLANET_ELASTIC_THICKNESS_M,
    }
}

/// The moon's land words.
#[must_use]
pub const fn home_moon_land_words() -> crate::land::LandWords {
    crate::land::LandWords {
        water_km3: HOME_MOON_WATER_KM3,
        elastic_thickness_m: HOME_MOON_ELASTIC_THICKNESS_M,
    }
}

/// The home planet's facts, as its realm states them.
#[must_use]
pub fn home_facts() -> BodyFacts {
    BodyFacts::new(HOME_PLANET_GRAVITY_MM_S2, HOME_PLANET_BULK_DENSITY_KGM3)
}

/// The home planet, defined by the recipe.
#[must_use]
pub fn home_planet() -> BodyDefinition {
    BodyDefinition::from_seed(
        HOME_PLANET_SEED,
        f64::from_bits(HOME_PLANET_RADIUS_BITS),
        home_facts(),
    )
    .expect("the home planet is on the ladder")
}

/// The home planet's moon, defined by the recipe.
#[must_use]
pub fn home_moon() -> BodyDefinition {
    BodyDefinition::from_seed(
        HOME_MOON_SEED,
        f64::from_bits(HOME_MOON_RADIUS_BITS),
        BodyFacts::new(HOME_MOON_GRAVITY_MM_S2, HOME_MOON_BULK_DENSITY_KGM3),
    )
    .expect("the home moon is on the ladder")
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
    fn the_home_moon_is_on_the_ladder() {
        let moon = home_moon();
        assert_eq!(moon.seed, HOME_MOON_SEED);
        // The ladder snaps the moon's radius to 2N/π at N = 557 056 cells: 354 632.9 m.
        assert!((moon.radius_m() - 354_632.9).abs() < 1.0);
        assert_eq!(moon.ladder.n, 557_056);
    }

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
            BodyDefinition::from_seed(HOME_PLANET_SEED, r + 1e-3, home_facts()),
            Some(home)
        );
        assert_eq!(
            BodyDefinition::from_seed(HOME_PLANET_SEED, r - 1e-3, home_facts()),
            Some(home)
        );
        let mut ulps = r;
        let mut i = 0;
        while i < 1_000 {
            ulps = f64::from_bits(ulps.to_bits() + 1);
            i += 1;
        }
        assert_eq!(
            BodyDefinition::from_seed(HOME_PLANET_SEED, ulps, home_facts()),
            Some(home)
        );
        // The snap unit is one top-rung cell edge: a radius moved by half of it can change the body.
        let unit_m = f64::from(1u32 << (home.ladder.rungs - 1)) * std::f64::consts::FRAC_2_PI;
        assert_ne!(
            BodyDefinition::from_seed(HOME_PLANET_SEED, r + unit_m, home_facts())
                .map(|b| b.ladder.n),
            Some(home.ladder.n),
            "a whole snap unit moves the edge count"
        );
    }
}
