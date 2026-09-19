//! ★ THE HOME BODY's CROSS-PIN (the voxel foundation, slice 5; SL5: one world). The generator's golden
//! gate (`crates/terrain/tests/terrain_pin.rs`) states the home planet as two literals, its seed and
//! the bits of its look radius, because the generator may name no motion crate. This test proves the
//! forest still produces exactly those two numbers, so the golden table and THE world can never drift
//! apart in silence: a forest change that moves the home planet turns this red, and the recipe's
//! version is bumped and the table re-recorded on purpose.

use vd_physics::worldgen::HOME_SEED;

const HOME_PLANET_SEED: u64 = 4_030_111_653_607_004_909;
const HOME_PLANET_RADIUS_BITS: u64 = 0x4158_4d6e_d403_3833;

/// ★ THE HOME MOON's CROSS-PIN (the landform arc, slice 8c stage C1): the generator states the
/// home planet's moon as literals so the solve's driver test runs on a REAL SMALL BODY OF THE
/// WORLD; this proves the forest still draws exactly those numbers — the seed, the bits of the
/// look radius, the two charter words — and that the census's system age is the age the
/// generator states as the erosional age (the design's ask 3).
#[test]
fn the_forests_home_moon_is_the_generators_home_moon() {
    let config =
        vd_physics::worldgen::UniverseConfig::world(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    let home_realm = vd_core::worldgen::HOME_PLANET;
    let held = std::collections::BTreeSet::from([home_realm]);
    let lineage = std::collections::BTreeSet::from([
        vd_core::worldgen::GALAXY,
        vd_core::worldgen::HOME_SYSTEM,
    ]);
    let (rows, _) =
        vd_physics::worldgen::shard_boot_world(HOME_SEED, &config, &held, home_realm, &lineage);
    let moons: Vec<(vd_core::pose::RealmId, f64)> = rows
        .iter()
        .filter(|r| r.parent == Some(home_realm))
        .filter_map(|r| match (r.realm, r.look) {
            (
                realm @ vd_core::pose::RealmId::Planet(_),
                Some(vd_core::geometry::Boundary::Shell { r: look }),
            ) => Some((realm, look)),
            _ => None,
        })
        .collect();
    assert_eq!(moons.len(), 1, "the home planet has one moon: {moons:?}");
    let (moon_realm, look) = moons[0];
    assert_eq!(
        moon_realm,
        vd_core::pose::RealmId::Planet(vd_terrain::home::HOME_MOON_SEED),
        "the moon's seed moved"
    );
    assert_eq!(
        look.to_bits(),
        vd_terrain::home::HOME_MOON_RADIUS_BITS,
        "the moon's look radius moved: bits {:#018x}",
        look.to_bits()
    );
    let facts = vd_physics::worldgen::body_facts_in_subtree(
        HOME_SEED, &config, &held, &lineage, moon_realm,
    )
    .expect("the moon has facts");
    let [g, rho] = vd_physics::worldgen::relief_words(facts.taxon.mass_kg, facts.taxon.radius_m);
    assert_eq!(g, Some(vd_terrain::home::HOME_MOON_GRAVITY_MM_S2));
    assert_eq!(rho, Some(vd_terrain::home::HOME_MOON_BULK_DENSITY_KGM3));
    let forest = vd_terrain::BodyDefinition::from_seed(
        vd_terrain::home::HOME_MOON_SEED,
        look,
        vd_terrain::BodyFacts::new(g.expect("g"), rho.expect("rho")),
    )
    .expect("on the ladder");
    assert_eq!(forest, vd_terrain::home::home_moon());
    // The erosional age is the census's own system age, to the year.
    assert_eq!(
        (vd_physics::taxonomy::SYSTEM_AGE_GYR * 1.0e9) as u64,
        vd_terrain::home::HOME_SYSTEM_AGE_YR
    );
}

#[test]
fn the_forests_home_planet_is_the_golden_gates_home_planet() {
    let body = vd_bins::home_body(HOME_SEED).expect("the home system holds a round planet");
    assert_eq!(
        body.seed(),
        HOME_PLANET_SEED,
        "the home planet's seed moved"
    );
    // ★ THE HOME PLANET IS EARTH-LIKE by the census's own predicate (ruling V13 L27): the named
    // system holds exactly one earth-like body and it is the named planet.
    assert_eq!(vd_core::worldgen::HOME_PLANET_SEED, HOME_PLANET_SEED);
    let config =
        vd_physics::worldgen::UniverseConfig::world(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    let held = std::collections::BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let facts = vd_physics::worldgen::body_facts_in_subtree(
        HOME_SEED,
        &config,
        &held,
        &lineage,
        vd_core::worldgen::HOME_PLANET,
    )
    .expect("the home planet has facts");
    assert!(facts.earth_like, "the home planet is earth-like: {facts:?}");
    let earth_like =
        vd_physics::worldgen::earth_like_in_subtree(HOME_SEED, &config, &held, &lineage);
    assert_eq!(
        earth_like.iter().map(|c| c.body).collect::<Vec<_>>(),
        vec![vd_core::worldgen::HOME_PLANET],
        "the home system's one earth-like body is the home planet"
    );
    let golden = vd_terrain::BodyDefinition::from_seed(
        HOME_PLANET_SEED,
        f64::from_bits(HOME_PLANET_RADIUS_BITS),
        vd_terrain::home::home_facts(),
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

/// ★ `charter_pin` — THE HOME BODY'S CHARTER, STATED AS LITERALS AND CROSS-CHECKED AGAINST THE
/// FOREST (the landform arc, slice 8b stage 2; `slice_8b_design.md` §6.2).
///
/// **WHY IT MUST EXIST.** MEASURED in §1.8 of the design: `digest_of` folds a chunk's cells and
/// folds NO charter word. So the golden table and the no-drift gate catch a charter word only where
/// it moves a cell — and in stage 2 NO kernel reads the charter at all. Without this pin every
/// integer below is untested data that four later slices will trust.
///
/// It reads the same way `the_forests_home_planet_is_the_golden_gates_home_planet` reads the seed
/// and the radius: the literals are the record, the forest is the live draw, and a change to either
/// turns this red and becomes a stated decision instead of a silent drift.
#[test]
fn the_home_bodys_charter_is_the_forests_charter() {
    let config =
        vd_physics::worldgen::UniverseConfig::world(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    let held = std::collections::BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let started = std::time::Instant::now();
    let charter = vd_physics::worldgen::body_charter_in_subtree(
        HOME_SEED,
        &config,
        &held,
        &lineage,
        vd_core::worldgen::HOME_PLANET,
    )
    .expect("the home planet's own shard derives its charter from its own subtree");
    // M-B, the boot cost (§6.1): the whole derivation, subtree build included, on this host.
    println!(
        "[charter_pin] the home planet's charter derived in {:?}: {charter:?}",
        started.elapsed()
    );
    // ★ THE LITERALS. Every one is MEASURED from the forest at this commit, not computed by hand.
    assert_eq!(
        charter,
        vd_core::look::BodyCharter {
            // 9.818 m/s² — the design's hand figure was 9 821 from L27's prose; the forest's own
            // draw is three millimetres per second squared lower, and the forest wins.
            gravity_mm_s2: 9_818,
            // 5 513 kg/m³, Earth's 5 514 to a part in five thousand.
            bulk_density_kgm3: 5_513,
            escape_velocity_mps: 11_185,
            // 0.748 S⊕ at rung 2 — the seed-free insolation ladder, in 1/4096.
            insolation_q12: 3_065,
            // 236.785 K equilibrium.
            t_eq_mk: 236_785,
            // 288.000 K: the thermostat's set point (stage 4, ask 2 ASSUMED).
            t_surface_mk: Some(288_000),
            // 0.2998 — the Rocky-with-atmosphere Bond albedo, in 1/4096.
            bond_albedo_q12: 1_228,
            // 28.0 u: an N₂-like secondary atmosphere (MU_N2), in 1/256.
            mu_q8: Some(7_168),
            scale_height_m: Some(7_160),
            // 101 409 Pa — DERIVED (ruling T9): Earth's, scaled by the mass, the gravity and the
            // area; Earth's twin reads one bar by construction (the draw once read 1 239 Pa).
            p_surf_pa: Some(101_409),
            // 0.1005 at one bar and 28 u (Earth's 0.0973 at 28.96 u), in 1/4096 — the design's 412.
            tau_vis_q12: Some(412),
            // 1.584: the greenhouse that lifts 236.785 K to the set point, in 1/4096.
            tau_ir_q12: Some(6_490),
            // 72.1 h, a log-uniform draw in [6 h, 100 h]; 1.049 AU is outside the 0.388 AU lock.
            day_s: Some(259_597),
            // cos ε = 0.572 (55°): the common band's draw, damped by the moon's small torque.
            obliquity_cos_q1024: Some(586),
            // 2 735 928 089 km³: the formation zone at 0.428 of the snow line (4.58e-4 of the
            // mass), retained by the census's own shoreline, under the runaway edge — about
            // TWO Earth oceans (stage 5; the design's hand estimate 2.737e9).
            water_km3: Some(2_735_928_089),
            sea_offset_mm: None,
            // 36.789 km from the density word 5 513 at 5.0 Gyr (the design's 36.8 km).
            elastic_thickness_m: Some(36_789),
            // e = 0.01552, in 1/65536.
            ecc_q16: 1_017,
            // 34 727 239 s = 401.9 days — the home planet's year.
            year_s: 34_727_239,
            // 1027 = air (1) + solid ground (2) + the G-class sun (4 << 8).
            flags: 1_027,
        },
        "the home body's charter moved"
    );
    // ★ THE TWO WORDS WITH NO AUTHOR YET are ABSENT, and that is asserted, not assumed: a zero in
    // either would be read by a later stage as a fact nobody drew. The seven words stage 4 draws
    // are PRESENT, pinned above to the forest's own numbers.
    // The physics charter carries NO sea: the sea is the OWNING REALM's solve over the body's own
    // shape (slice 8b stage 6), stated by the shard's boot through `charter_with_sea`.
    assert_eq!(
        charter.sea_offset_mm, None,
        "the sea is solved by the realm, not drawn by the census"
    );
    // ★ G-SEA (ruling T8 (a)): the home planet's stated sea. Its two oceans are LIQUID under the
    // derived pressure (101 409 Pa, boiling at 373 K, the surface at 288 K), so the shard solves the
    // level and states it: 4 957.341 m over the ladder radius — and on today's one-humped ground
    // that level stands over EVERY sample (the ocean share reads 1.0000 at 34 656 samples, the
    // escalated rung). The recipe does not read it until 8c; the pictures are judged dry.
    let home_look = 6_370_747.312_696_504;
    let body = vd_terrain::BodyDefinition::from_seed(
        HOME_PLANET_SEED,
        home_look,
        vd_bins::facts_of_charter(&charter),
    )
    .expect("the home body");
    let stated = vd_bins::charter_with_sea(charter, Some(&body));
    let level =
        vd_bins::sea::solve_sea_level(&body, charter.water_km3.expect("water") as f64 * 1.0e9)
            .expect("a sea");
    println!(
        "[charter_pin] G-SEA: the home planet's sea stands {:.3} m over the ladder radius, ocean share {:.4} at rung {} ({} samples)",
        level.offset_m, level.ocean_share, level.rung, level.samples
    );
    assert_eq!(
        stated.sea_offset_mm,
        Some(4_957_341),
        "the stated sea moved"
    );
    assert_eq!(
        stated.flags & vd_core::look::CHARTER_FLAG_HAS_SEA,
        vd_core::look::CHARTER_FLAG_HAS_SEA,
        "the home planet has a sea"
    );
    assert_eq!(
        vd_physics::worldgen::quantise_i32(level.offset_m, 1_000.0),
        stated.sea_offset_mm,
        "the stored offset is the derived one (G-SEA)"
    );
    assert_eq!(
        [
            charter.t_surface_mk.is_some(),
            charter.p_surf_pa.is_some(),
            charter.tau_vis_q12.is_some(),
            charter.tau_ir_q12.is_some(),
            charter.day_s.is_some(),
            charter.obliquity_cos_q1024.is_some(),
            charter.elastic_thickness_m.is_some(),
        ],
        [true; 7],
        "every word stage 4 draws is present on the home planet"
    );
    assert_eq!(
        charter.flags,
        vd_core::look::CHARTER_FLAG_HAS_AIR
            | vd_core::look::CHARTER_FLAG_SOLID_SURFACE
            | (4 << vd_core::look::CHARTER_STAR_CLASS_SHIFT),
        "the home planet has air, ground, and a yellow sun"
    );
    assert_eq!(
        vd_core::look::charter_star_class_code(charter.flags),
        4,
        "a G-class star"
    );
    // The charter the SHARD states is the charter the forest draws — one derivation, read twice.
    // M-B, the MARGINAL cost: this call runs with the layer cache already warm, which is the state a
    // shard's boot is in by the time it asks (the boot builds its world first).
    let warm = std::time::Instant::now();
    assert_eq!(
        vd_bins::boot_charter(
            HOME_SEED,
            &held,
            vd_core::worldgen::HOME_PLANET,
            vd_bins::DEV.move_speed,
            vd_bins::DEV.tick_dt,
            &lineage,
        ),
        Some(charter),
        "the shard's boot derives the forest's charter"
    );
    println!(
        "[charter_pin] M-B: the shard's boot charter, layer cache warm, in {:?}",
        warm.elapsed()
    );
    // A realm the seed gave no body — the star system that holds the planet — states no charter, so
    // its shard states no surface.
    assert_eq!(
        vd_bins::boot_charter(
            HOME_SEED,
            &held,
            vd_core::worldgen::HOME_SYSTEM,
            vd_bins::DEV.move_speed,
            vd_bins::DEV.tick_dt,
            &lineage,
        ),
        None,
        "a star system has no body charter"
    );
}
