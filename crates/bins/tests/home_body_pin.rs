//! ★ THE HOME BODY's CROSS-PIN (the voxel foundation, slice 5; SL5: one world). The generator's golden
//! gate (`crates/terrain/tests/terrain_pin.rs`) states the home planet as two literals, its seed and
//! the bits of its look radius, because the generator may name no motion crate. This test proves the
//! forest still produces exactly those two numbers, so the golden table and THE world can never drift
//! apart in silence: a forest change that moves the home planet turns this red, and the recipe's
//! version is bumped and the table re-recorded on purpose.

use vd_physics::worldgen::HOME_SEED;

const HOME_PLANET_SEED: u64 = 7_701_581_858_760_374_086;
const HOME_PLANET_RADIUS_BITS: u64 = 0x4149_9139_1e69_2dfa;

#[test]
fn the_forests_home_planet_is_the_golden_gates_home_planet() {
    let body = vd_bins::home_body(HOME_SEED).expect("the home system holds a round planet");
    assert_eq!(
        body.seed(),
        HOME_PLANET_SEED,
        "the home planet's seed moved"
    );
    let golden = vd_terrain::BodyDefinition::from_seed(
        HOME_PLANET_SEED,
        f64::from_bits(HOME_PLANET_RADIUS_BITS),
    )
    .expect("on the ladder");
    assert_eq!(
        body, golden,
        "the forest's home planet and the golden gate's home planet are the same body"
    );
    let identity = vd_bins::world_identity(HOME_SEED).expect("a world identity");
    assert_eq!(identity.declared, vd_terrain::declared_world_tag(HOME_SEED));
    assert_eq!(
        Some(identity.measured),
        vd_terrain::golden_self_check(&body)
    );
    // The generator's own literals are the same two numbers (SL5: one world, stated once per crate,
    // tied here).
    assert_eq!(body.seed(), vd_terrain::home::HOME_PLANET_SEED);
    assert_eq!(
        HOME_SEED,
        vd_terrain::home::HOME_UNIVERSE_SEED,
        "the universe seed the client folds"
    );
    assert_eq!(body, vd_terrain::home::home_planet());
    // ★ THE POLE AXIS (the refuter's finding 4): the biome's pole is the orbits' axis. The world's
    // perifocal plane is `z = 0` (`crates/physics/src/celestial.rs`): an orbit with no inclination
    // stays at `z = 0` at every instant, so the axis every orbit turns about is `+Z`, and the
    // generator's pole must be `+Z` too.
    let flat = vd_physics::celestial::OrbitalElements {
        sma: 1.0e9,
        ecc: 0.1,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: 1.0e30,
    };
    for t in [0.0, 1_000.0, 123_456.0] {
        assert_eq!(
            vd_physics::celestial::orbital_state(&flat, t).position.z,
            0.0,
            "an orbit without inclination lies in the z = 0 plane"
        );
    }
    assert_eq!(
        vd_terrain::height::POLE_AXIS,
        2,
        "the pole is the orbits' axis, +Z"
    );
    // The world label folds the recipe's version: a different version is a different world.
    assert_ne!(
        vd_bins::world_generation(),
        vd_core::store_stamp::world_generation(&vd_physics::worldgen::world_shape_constants()),
        "the recipe's version is folded into the world label"
    );
}
