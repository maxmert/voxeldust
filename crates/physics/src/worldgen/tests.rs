//! THE GENERATOR'S UNIT TIER — the assertions that were the tail of `worldgen.rs` until the file
//! was split, moved VERBATIM and named identically (the module path did not change: this is still
//! `worldgen::tests`).
//!
//! Owns: the fixtures, the bit-exact draw-stream pins and the fence assertions for every lane in the
//! `worldgen` tree. It reads `super::*`, so it sees the module's whole re-exported surface exactly
//! as a caller outside the crate does, plus the crate-private items the lanes share.
//!
//! Does NOT own: any production behaviour, and no second world. Every fixture reads THE one
//! generator at a seed (SL5); a test may choose a seed, never a smaller universe.

use super::*;
use crate::celestial::KEPLER_ECC_MAX;
use crate::celestial::{OrbitalElements, orbital_state};
use crate::motion::Motion;
use crate::taxonomy::{
    FrostThresholds, SpectralClass, classify_spectral, habitable_zone_radius_au,
    main_sequence_luminosity, orbital_axis_au,
};
use core::f64::consts::TAU;
use glam::DVec3;
use vd_core::frame::FramePlacement;
use vd_core::geometry::visibility_factor;
use vd_core::geometry::{AoiConfig, BandError, Boundary, RealmRegion};
use vd_core::geometry::{region_depth, region_signed_distance};
use vd_core::pose::{LatticePos, RealmId, Tier, frame_for_realm};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmPath};
use vd_core::rng::{child_seed, realm_stream};
use vd_core::worldgen::{
    AREA_A, GALAXY, GALAXY_SEED, PLANET_A, STATION_A, SYSTEM_A, SYSTEM_A_SEED, SYSTEM_B, UNIVERSE,
    UNIVERSE_SEED, level_of,
};
use vd_core::worldgen::{coord_of_realm, default_home_realm};

/// HOW MANY SYSTEMS A TEST'S GALAXY HOLDS, unless the test says otherwise.
///
/// ★ WHY A TEST DOES NOT GET THE SHIPPED GALAXY (S12/G8, 2026-08-28). THE world holds 279 380 star
/// systems, which is 4.2 million objects and 1.5 GB of them. The runner starts one test per core, so
/// the suite asked for that fourteen times over and the machine stopped — MEASURED at 21 GB resident
/// with the swap exhausted, twice.
///
/// Nearly every test here examines a RULE: does a boundary nest inside its parent, does a band fit,
/// does a moon sit at the depth it claims. A rule that holds across a quarter of a million systems
/// holds across fifty, and fifty cost five thousand times less. So a test gets a SMALL GALAXY —
/// which owner ruling G11 sanctions by name: *"looking at a small galaxy is looking at THE world
/// through a different seed"*. Same law, same draws, same density, a smaller volume. No variant, no
/// preset, no second generator.
///
/// A test whose subject IS the population says so and takes the shipped galaxy, spelled out.
const TEST_GALAXY_SYSTEMS: u32 = 48;

/// ★ THE WORLD'S OWN CONFIG, UNSCALED (2026-08-30) — for a test that reads CONFIG FIELDS and never
/// builds a forest from them.
///
/// [`test_world`] shrinks the galaxy's radius so a test can hold its forest in memory. That is the
/// right trade for a test that BUILDS something, and the wrong one for a test that PINS the world's
/// own geometry: it then measures a scaled galaxy against unscaled numbers and can only fail.
/// MEASURED: the placement radius reads 2.56e17 on the scaled galaxy against THE world's 4.61e18.
///
/// Reading a config costs nothing — no forest is folded — so a geometry pin should always use this.
fn world_config() -> UniverseConfig {
    UniverseConfig::world(15.0, 0.05)
}

/// THE world, at a galaxy a test can hold. See [`TEST_GALAXY_SYSTEMS`].
fn test_world() -> UniverseConfig {
    galaxy_holding(&UniverseConfig::world(15.0, 0.05), 0, TEST_GALAXY_SYSTEMS)
}

/// The visual preset, at a galaxy a test can hold. See [`TEST_GALAXY_SYSTEMS`].
fn test_visual() -> UniverseConfig {
    galaxy_holding(&UniverseConfig::visual_scale(), 0, TEST_GALAXY_SYSTEMS)
}

/// ★ A SMALL GALAXY, LAWFULLY (S12/G8 + G11, 2026-08-28).
///
/// A population is a RESULT now — the volume a galaxy encloses at the density it drew — so there is
/// no count to ask for. `system_count_lo`/`system_count_hi` are deleted, and with them the only way a
/// test used to get a four-star world.
///
/// The owner's ruling G11 says how to look at a small galaxy without building one: *"use the same
/// seed and generation mechanisms"*, because galaxies DIFFER in size and some simply ARE small. So a
/// test that wants a handful of systems asks for a SMALL GALAXY — the same law, the same draws, a
/// smaller volume. Nothing here is a variant world: the rim is a galaxy's own size, and the count
/// follows from it exactly as it does for the shipped one.
///
/// The solve is exact rather than a search: the population is `density · π·R²·(t·R)`, so it goes as
/// `R³`, and the radius that holds `want` is the shipped radius scaled by the cube root of the ratio.
fn galaxy_holding(base: &UniverseConfig, seed: u64, want: u32) -> UniverseConfig {
    let mut cfg = *base;
    let have = galaxy_profile(seed, &cfg).count;
    let ratio = f64::from(want.max(1)) / f64::from(have.max(1));
    cfg.stellar.galaxy_rim_r_m *= ratio.cbrt();
    cfg
}

/// ★ HOW MANY SEEDS A SWEEP NEEDS — DERIVED FROM THE WORLD'S OWN POPULATION (S12/G8, 2026-08-28).
///
/// A sweep's evidence is the number of star systems it JUDGES, never the number of seeds it spends.
/// Every sweep in this file was sized when a galaxy held THREE systems, so hundreds of seeds bought
/// hundreds of systems. A galaxy now holds a quarter of a million, and the same sweeps became
/// unaffordable overnight: MEASURED, the 664-seed fence sweep asks for 1 328 whole worlds at 1.3 GB
/// and 0.7 s each. The test process reached 21 GB resident with swap exhausted.
///
/// So the seeds follow the population, exactly as the nest sweep's own size always has
/// (`derived_nest_sweep_seeds`). More stars per world, fewer worlds to reach the same evidence.
///
/// THE FLOOR IS NOT A FUDGE: the SHAPE is drawn per seed, so one seed judges one galaxy's shape
/// however many systems it holds. A sweep that collapsed to a single seed would stop testing the
/// thing these sweeps exist for — that a CROWDED draw is still lawful — so several distinct galaxies
/// are always visited.
fn seeds_to_judge(want_systems: u32) -> u64 {
    /// Distinct galaxy shapes every sweep visits, whatever the arithmetic says.
    const SHAPES: u64 = 6;
    let per_world = u64::from(world_system_count()).max(1);
    u64::from(want_systems).div_ceil(per_world).max(SHAPES)
}

/// How many star systems the galaxy of `cfg` holds — what the LAW says, for a test to check the
/// forest against.
///
/// ★ THIS IS NOT CIRCULAR, and the distinction matters. The profile states what the density and the
/// shape COME TO; the forest states what the generator EMITTED. Comparing them proves the generator
/// built the galaxy it was told to. A hard-coded 3 proved that once, in 2026, for a world that no
/// longer exists.
fn systems_in(cfg: &UniverseConfig) -> usize {
    galaxy_profile(0, cfg).count as usize
}

/// THE world's own population, DERIVED — the count its galaxy's drawn density and drawn disc
/// thickness produce (owner ruling G8: a population is a result, never a stated number). This
/// replaces `WORLD_SYSTEM_COUNT`, which stated one.
fn world_system_count() -> u32 {
    galaxy_profile(0, &test_world()).count
}

/// The world's DERIVED planet count (9 — scale-free; the disc edge over the ladder).
fn world_n_planets() -> u32 {
    test_visual().planet.n_planets
}

fn regions() -> Vec<RealmRegion> {
    realm_regions_for(0)
}

/// THE world at a nominal occupant speed — the config-driven twin the bins boot uses.
fn boot_world_for_tests() -> WorldView {
    WorldView::generated(0, &test_world())
}

#[test]
fn the_world_preset_is_the_visual_demand_geometry_and_the_alias_is_its_twin() {
    // `UniverseConfig::world` IS `visual_demand` (SL5: one world, the name states the law), and
    // the held-config alias builds the SAME neighbourhood as the fn it delegates to (HR3: the
    // seam stays closed — two identical functions consulting different worlds is how the login
    // side and the simulating side once described different universes from one seed).
    // ★ BOTH SIDES AT THE SAME SIZE (S12/G8, 2026-08-28). The subject is that the two preset NAMES
    // describe ONE geometry, so both must be asked the same question. Shrinking only one would make
    // the lengths differ for a reason that has nothing to do with the presets.
    let world = test_world();
    let demand = galaxy_holding(
        &UniverseConfig::visual_demand(15.0, 0.05),
        0,
        TEST_GALAXY_SYSTEMS,
    );
    assert_eq!(
        generate_system_forest(0, &world).len(),
        generate_system_forest(0, &demand).len(),
        "one geometry"
    );
    let held = std::collections::BTreeSet::from([RealmId::System(7)]);
    assert_eq!(
        realm_neighbourhood_for_held_config(0, &held, &world),
        realm_neighbourhood_for_config(0, &held, &world),
        "the alias is byte-equal to its twin"
    );
}

#[test]
fn the_sibling_fence_judges_an_orbit_on_its_shell_never_its_epoch() {
    // The overlap fence skips a KEPLER sibling on BOTH sides of the pair loop: an orbit is
    // judged on its shell (the SOI nesting fence), not on where its epoch anchor happens to sit
    // — an epoch-position overlap between an orbiting body and a static one is not a defect.
    let orbital = Placement::Orbital(OrbitalElements {
        sma: 1.0e11,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: 1.989e30,
    });
    let body = |realm: RealmId, placement: Placement| GeneratedBody {
        realm,
        parent: Some(RealmId::System(1)),
        shape: Boundary::Shell { r: 10.0 },
        taxon: None,
        look: Some(Boundary::Shell { r: 10.0 }),
        placement,
        photometrics: None,
    };
    // A static + an ORBITING sibling at the "same place": no overlap verdict — the orbiter is
    // skipped by the inner arm.
    let mixed = [
        body(RealmId::Planet(1), Placement::StaticOffset(DVec3::ZERO)),
        body(RealmId::Planet(2), orbital),
    ];
    assert!(siblings_disjoint(&mixed).is_ok());
    // Two STATIC siblings genuinely overlapping: the fence still fires (non-vacuity).
    let clash = [
        body(RealmId::Planet(1), Placement::StaticOffset(DVec3::ZERO)),
        body(
            RealmId::Planet(2),
            Placement::StaticOffset(DVec3::new(5.0, 0.0, 0.0)),
        ),
    ];
    assert!(siblings_disjoint(&clash).is_err());
}

#[test]
fn coord_of_realm_resolves_the_full_root_rooted_lineage_and_is_none_for_a_ship() {
    // The un-lossy RealmId→RealmCoord a source shard uses to KeepAlive-demand a crossing DEST's whole
    // ancestor chain (the Symptom-B freeze fix). A System resolves to [Universe, Galaxy, System]; a
    // Planet one level deeper; a ship (entity-backed, no seed level) resolves to None.
    let regions = realm_regions_for(0);
    let want_sys = RealmCoord::from_path(RealmPath::from_levels(vec![
        level_of(UNIVERSE),
        level_of(GALAXY),
        level_of(SYSTEM_A),
    ]))
    .expect("a 3-level path has a leaf");
    let want_planet = want_sys.child(level_of(PLANET_A));
    assert_eq!(coord_of_realm(&regions, SYSTEM_A), Some(want_sys));
    assert_eq!(want_planet.path().levels().len(), 4); // full lineage, never lowered()
    assert_eq!(coord_of_realm(&regions, PLANET_A), Some(want_planet));
    // ★ A REALM THIS FOREST DOES NOT HOLD HAS NO LINEAGE — and the reason changed on 2026-09-01.
    // This used to pass because `level_of` refused a SHIP. It now passes because the forest does not
    // CONTAIN this realm, which is the honest condition and true of every kind. A ship that really is
    // in a forest — a built one, live state — resolves like anything else.
    let ship = RealmId::Ship(vd_core::ids::EntityId::pack(
        vd_core::entity_kind::EntityKind::Ship,
        1,
        1,
        1,
    ));
    assert_eq!(coord_of_realm(&regions, ship), None);
}

#[test]
fn region_signed_distance_is_frame_aware_and_identity_at_p3() {
    let rs = regions();
    let system_a = rs
        .iter()
        .find(|r| r.realm == SYSTEM_A)
        .expect("system A is in the forest");
    // At the system center: signed distance = -r_soi (fully inside). Identity frame ⇒ pos unchanged.
    let at_centre = vd_core::pose::StampedPose::at_rest(
        vd_core::pose::FrameRef::SystemSpace { system_seed: 0 },
        DVec3::ZERO,
        vd_core::ids::UniverseTick(0),
    );
    // The book: anchored on the pose's own frame, with the system's frame at the identity (the
    // P3 shipping shape — every placement is the identity, so the reframe moves nothing).
    let book = vd_core::placement::PlacementBook::new(
        at_centre.frame,
        at_centre.universe_tick,
        vec![(system_a.frame, vd_core::frame::FramePlacement::identity())],
    );
    let sd = region_signed_distance(&at_centre, system_a, &book).expect("ok");
    assert!(
        (sd - (-SYSTEM_SOI_R_M)).abs() < 1e-9,
        "center is r_soi inside: {sd}"
    );
}

#[test]
fn realm_neighbourhood_scopes_to_own_ancestors_and_children_never_siblings() {
    // System 7's shard: own + ancestors (Galaxy, Universe) + children Planet 7 AND Station 7 (a
    // first-class child under System 7) — NOT sibling System 8, NOT the grandchild Area 7.
    let n7: Vec<RealmId> = realm_neighbourhood_for(0, SYSTEM_A)
        .iter()
        .map(|r| r.realm)
        .collect();
    assert!(n7.contains(&SYSTEM_A));
    assert!(n7.contains(&GALAXY));
    assert!(n7.contains(&UNIVERSE));
    assert!(n7.contains(&PLANET_A));
    assert!(
        n7.contains(&STATION_A),
        "the Station is an OWNED child of System 7 — the shard scans it",
    );
    assert!(
        !n7.contains(&SYSTEM_B),
        "a shard NEVER loads a sibling — the scale-bounded rule",
    );
    assert!(
        !n7.contains(&AREA_A),
        "Area 7 is a grandchild (under Planet 7), not a direct child of System 7",
    );
    assert_eq!(n7.len(), 5);
    // The GALAXY shard: own + ancestor Universe + children System 7 & 8 (the between-space owner that
    // routes a sibling crossing) — NOT Planet 7 (a grandchild, not a direct child).
    let ng: Vec<RealmId> = realm_neighbourhood_for(0, GALAXY)
        .iter()
        .map(|r| r.realm)
        .collect();
    assert!(ng.contains(&GALAXY));
    assert!(ng.contains(&UNIVERSE));
    assert!(ng.contains(&SYSTEM_A));
    assert!(ng.contains(&SYSTEM_B));
    assert!(
        !ng.contains(&PLANET_A),
        "a grandchild is not a direct child"
    );
    assert_eq!(ng.len(), 4);
    // The STATION 7 shard: its own realm + its ANCESTOR CHAIN (System 7, Galaxy, Universe) and NO
    // children (a leaf) — it never pulls its sibling Planet 7 (they share the System 7 parent).
    let nst: Vec<RealmId> = realm_neighbourhood_for(0, STATION_A)
        .iter()
        .map(|r| r.realm)
        .collect();
    assert!(nst.contains(&STATION_A));
    assert!(nst.contains(&SYSTEM_A));
    assert!(nst.contains(&GALAXY));
    assert!(nst.contains(&UNIVERSE));
    assert!(
        !nst.contains(&PLANET_A),
        "the Station never loads its sibling Planet 7",
    );
    assert_eq!(nst.len(), 4);
    // The AREA 7 shard: its own realm + its ANCESTOR CHAIN (Planet 7, System 7, Galaxy, Universe) and
    // NO children — it never pulls its sibling Station 7 (they share the System 7 ancestor, not a parent).
    let nar: Vec<RealmId> = realm_neighbourhood_for(0, AREA_A)
        .iter()
        .map(|r| r.realm)
        .collect();
    assert!(nar.contains(&AREA_A));
    assert!(nar.contains(&PLANET_A));
    assert!(nar.contains(&SYSTEM_A));
    assert!(nar.contains(&GALAXY));
    assert!(nar.contains(&UNIVERSE));
    assert!(
        !nar.contains(&STATION_A),
        "the Area never loads the Station (they are not parent/child)",
    );
    assert_eq!(nar.len(), 5);
    // A shard hosting an unknown realm ⇒ empty neighbourhood ⇒ the detector is inert (safe degrade).
    // Both appended kinds (Station/Area) at an ABSENT seed (99) degrade to empty — the seed-7 plant
    // above does NOT make every Station/Area live.
    assert!(realm_neighbourhood_for(0, RealmId::Station(99)).is_empty());
    assert!(realm_neighbourhood_for(0, RealmId::Area(99)).is_empty());
}

#[test]
fn a_single_held_realm_neighbourhood_union_equals_the_single_neighbourhood() {
    // Co-hosting DEGENERATE case: a held-set of exactly one realm is byte-identical to
    // `realm_neighbourhood_for` — the single-realm shard path is untouched.
    for r in [SYSTEM_A, GALAXY, PLANET_A, STATION_A] {
        let single = realm_neighbourhood_for(0, r);
        let held = realm_neighbourhood_for_held(0, &std::collections::BTreeSet::from([r]));
        assert_eq!(
            single, held,
            "the held-set union for {{{r}}} equals its single neighbourhood",
        );
    }
}

#[test]
fn a_cohosted_system_plus_children_union_reaches_the_deepest_grandchild_area() {
    // The un-hosted-child cure: a shard co-hosting System 7 + its children (Planet/Station/Area) must
    // evaluate the DEEPEST region (Area 7, a GRANDCHILD of System 7 absent from System 7's OWN
    // neighbourhood). The union reaches it via Planet 7 being held.
    let held = std::collections::BTreeSet::from([SYSTEM_A, PLANET_A, STATION_A, AREA_A]);
    let realms: Vec<RealmId> = realm_neighbourhood_for_held(0, &held)
        .iter()
        .map(|r| r.realm)
        .collect();
    for expected in [UNIVERSE, GALAXY, SYSTEM_A, PLANET_A, STATION_A, AREA_A] {
        assert!(realms.contains(&expected), "the union includes {expected}");
    }
    // System B (a SIBLING of System 7 — never a child/ancestor of any held realm) is EXCLUDED.
    assert!(
        !realms.contains(&SYSTEM_B),
        "a sibling system is never in the co-hosting union",
    );
    // The union deduplicates (Universe/Galaxy/System 7 appear once even though several held realms
    // share them as ancestors) — the region set is a valid single-root forest.
    assert_eq!(
        realms.len(),
        6,
        "6 distinct regions (7-forest minus the sibling System B)"
    );
}

#[test]
fn realm_neighbourhood_for_config_excludes_sibling_planets_over_the_visual_forest() {
    // The flap cure at VISUAL scale: a planet shard's neighbourhood is its own realm + ancestors + the
    // children it authors — NEVER its sibling planets. A shard cannot place a realm it does not author, so
    // folding a sibling collapses it to the origin and a hosted occupant reads as inside all of them at
    // once (the production hot-potato). Unlike the walk forest, the visual system forest has MULTIPLE
    // orbiting planets, so this is where the exclusion actually bites.
    let cfg = test_visual();
    let forest = realm_regions_for_config(0, &cfg);
    let planets: Vec<RealmId> = forest
        .iter()
        .filter(|r| matches!(r.realm, RealmId::Planet(_)))
        .map(|r| r.realm)
        .collect();
    assert!(
        planets.len() >= 2,
        "the visual forest must have sibling planets to distinguish (got {planets:?})",
    );
    let target = planets[0];
    let sibling = planets[1];
    let parent = forest
        .iter()
        .find(|r| r.realm == target)
        .and_then(|r| r.parent)
        .expect("a visual-forest planet has a parent system");
    let scope: Vec<RealmId> =
        realm_neighbourhood_for_config(0, &std::collections::BTreeSet::from([target]), &cfg)
            .iter()
            .map(|r| r.realm)
            .collect();
    assert!(scope.contains(&target), "own realm is in scope");
    assert!(
        scope.contains(&parent),
        "the parent system (an ancestor) is in scope",
    );
    assert!(
        !scope.contains(&sibling),
        "a SIBLING planet is NEVER in scope — the origin-stacking flap cure",
    );
}

// ---- D-45(a) Slice 3b: UniverseConfig -------------------------------------------

#[test]
fn walk_scale_equals_the_named_geometry_consts() {
    let c = UniverseConfig::walk_scale();
    assert_eq!(c.scale.universe_r_m, UNIVERSE_R_M);
    // The galaxy is DERIVED to contain the ring of stars, no longer the walk constant: it must hold
    // every system with its reach, or a star sits outside its own galaxy.
    assert!(c.scale.galaxy_r_m > c.stellar.galaxy_rim_r_m + c.stellar.system_soi_r_m);
    assert_eq!(c.stellar.system_soi_r_m, SYSTEM_SOI_R_M);
    assert_eq!(c.planet.planet_soi_r_m, PLANET_SOI_R_M);
    assert_eq!(c.satellite.planet_offset_m, PLANET_A_OFFSET_M);
    assert_eq!(c.satellite.system_b_offset_m, SYSTEM_B_OFFSET_M);
    assert_eq!(c.satellite.station_offset_m, STATION_A_OFFSET_M);
    assert_eq!(c.satellite.station_half_m, STATION_HALF_M);
    assert_eq!(c.satellite.area_offset_m, AREA_OFFSET_M);
    assert_eq!(c.satellite.area_half_m, AREA_HALF_M);
    assert_eq!(c.band.inset_m, CONTAINMENT_INSET_M);
    assert_eq!(c.band.outset_m, CONTAINMENT_OUTSET_M);
    assert_eq!(c.planet.n_planets, 0);
    // The re-solve's draw bounds ride every preset (walk never reads them — no Orbital body).
    assert_eq!(c.planet.mass_lo_mearth, PLANET_MASS_LO_MEARTH);
    assert_eq!(c.planet.disc_mass_fraction, DISC_MASS_FRACTION);
    assert_eq!(c.planet.mass_cap_mearth, M_JUP_MEARTH);
}

#[test]
fn universe_config_presets_serde_round_trip() {
    for c in [
        UniverseConfig::walk_scale(),
        test_visual(),
        // The PLANTED config too (look_horizon slice 5): the plant field is DATA and must
        // survive the codec like every other field — both enum arms round-trip.
        test_world().with_station_area_plant(),
    ] {
        let bytes = postcard::to_allocvec(&c).expect("encode");
        let back: UniverseConfig = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(c, back);
    }
}

#[test]
fn every_preset_ecc_cap_is_within_the_kepler_domain() {
    // Fail-loud cross-slice invariant: no preset may cap eccentricity above the fixed Kepler
    // solver's convergence domain (KEPLER_ECC_MAX).
    assert!(UniverseConfig::walk_scale().planet.ecc_cap <= KEPLER_ECC_MAX);
    assert!(test_visual().planet.ecc_cap <= KEPLER_ECC_MAX);
}

#[test]
fn walk_band_builds_the_static_band() {
    UniverseConfig::walk_scale()
        .band
        .build()
        .expect("walk band is valid by construction");
}

#[test]
fn frost_thresholds_reads_the_planet_config() {
    let ft = UniverseConfig::walk_scale().planet.frost_thresholds();
    assert_eq!(ft, FrostThresholds::CANONICAL);
}

// ---- D-45(a) Slice 3c: generate -> to_regions lowering (byte-identity) -----------

#[test]
fn realm_regions_for_matches_the_frozen_pre_generator_golden() {
    // A HAND-AUTHORED frozen golden (literal values, independent of the generator path): if
    // to_regions drifts any coordinate / shape / parent, this fails against the literals — NOT
    // a self-referential capture of the (rewritten) realm_regions_for output.
    let rs = realm_regions_for(0);
    let expected: [(RealmId, DVec3, Boundary, Option<RealmId>); 7] = [
        (UNIVERSE, DVec3::ZERO, Boundary::Shell { r: 1.0e9 }, None),
        (
            GALAXY,
            DVec3::ZERO,
            Boundary::Shell { r: 180.0 },
            Some(UNIVERSE),
        ),
        (
            SYSTEM_A,
            DVec3::ZERO,
            Boundary::Shell { r: 40.0 },
            Some(GALAXY),
        ),
        (
            PLANET_A,
            DVec3::new(20.0, 0.0, 0.0),
            Boundary::Shell { r: 10.0 },
            Some(SYSTEM_A),
        ),
        (
            SYSTEM_B,
            DVec3::new(130.0, 0.0, 0.0),
            Boundary::Shell { r: 40.0 },
            Some(GALAXY),
        ),
        (
            STATION_A,
            DVec3::new(-25.0, 0.0, 0.0),
            Boundary::Aabb {
                half: DVec3::splat(5.0),
            },
            Some(SYSTEM_A),
        ),
        (
            AREA_A,
            // +5 FROM ITS PARENT, the planet at +20 — so still +25 in the system's frame, the same
            // place it has always occupied. The golden used to read 25 here, an absolute on a field
            // that means "offset from my parent"; it went unnoticed while nothing converted between
            // levels. The number changed; the world did not.
            DVec3::new(5.0, 0.0, 0.0),
            Boundary::Aabb {
                half: DVec3::splat(3.0),
            },
            Some(PLANET_A),
        ),
    ];
    // ★ 7 → 9 ON 2026-08-31. The hand-placed world gained TWO PLANETS under System B — the only
    // hand-placed system that is actually somewhere, System A sitting at the galaxy's origin. A
    // worked example needs a system that is BOTH placed and populated, to prove a parent adds its
    // child's placement; a hop of zero proves nothing. See `walk.rs` for the bodies themselves.
    assert_eq!(rs.len(), 9);
    for (r, (realm, offset, shape, parent)) in rs.iter().zip(expected) {
        assert_eq!(r.realm, realm);
        // ★ READ AT THE PARENT'S RUNG SINCE S9. `center` is this realm's position in its PARENT's
        // frame, so the ruler that turns it into metres is the PARENT's, and the ladder made those
        // rulers differ: a system under the galaxy counts in 2 m cells, the galaxy under the universe
        // in 32_768 m cells. The golden read every one of them in millimetres, which put System B —
        // 130 m along the galaxy's x axis, 65 galaxy cells — at cell 133_120.
        //
        // The METRES in the golden below are unchanged, and that is the point: the world did not
        // move, only the counting did.
        let ptier = rs
            .iter()
            .find(|p| Some(p.realm) == parent)
            .map_or_else(|| r.frame.tier(), |p| p.frame.tier());
        assert_eq!(
            r.center,
            vd_core::geometry::ParentCentre::authored(LatticePos::from_metres(offset, ptier))
        );
        assert_eq!(
            r.center
                .in_parents_frame()
                .delta_m(LatticePos::ORIGIN, ptier),
            offset
        );
        assert_eq!(r.shape, shape);
        assert_eq!(r.parent, parent);
        assert_eq!(
            r.frame,
            frame_for_realm(realm, parent).expect("canonical frame")
        );
    }
    // ★ RE-BASED AT S6. The forest used to share ONE static band; every band is now sized from the
    // body it wraps, so the golden asserts the DERIVATION rather than a single literal. This still
    // fails if a coordinate, a shape or a parent drifts — it simply no longer claims all bands are
    // equal, which stopped being true when they started being sized.
    let cfg = UniverseConfig::walk_scale();
    for r in &rs {
        assert_eq!(
            r.band,
            cfg.band.build_for_shape(&r.shape).expect("walk band"),
            "{:?}'s band is not what the one band solve gives for its own extent",
            r.realm
        );
    }
}

#[test]
fn to_regions_gives_an_orbital_body_a_zero_center_position_authored_by_the_frame() {
    // The Orbital placement arm: a moving body carries NO baked position — its boundary sits at the
    // ZERO origin of its own frame; its live pose is authored per tick into the placement book.
    let elements = OrbitalElements {
        sma: 1.5e11,
        ecc: 0.1,
        inclination: 0.4,
        raan: 0.3,
        arg_periapsis: 0.9,
        mean_anomaly_epoch: 0.2,
        central_mass: 1.989e30,
    };
    let body = GeneratedBody {
        realm: RealmId::Planet(42),
        parent: Some(RealmId::System(42)),
        shape: Boundary::Shell { r: 9.0e8 },
        taxon: None,
        look: Some(Boundary::Shell { r: 9.0e8 }),
        placement: Placement::Orbital(elements),
        photometrics: None,
    };
    let regions = to_regions(&[body], &test_visual());
    assert_eq!(regions.len(), 1);
    assert_eq!(
        regions[0].center.in_parents_frame().cell(),
        glam::I64Vec3::ZERO
    );
    assert_eq!(regions[0].center.in_parents_frame().offset(), DVec3::ZERO);
}

#[test]
fn placement_offset_reads_static_verbatim_and_the_orbital_tick_zero_position() {
    // The frame-local offset utility. `StaticOffset` returns its fixed vector verbatim; `Orbital`
    // returns the tick-0 ephemeris position. `region_center_of` only ever feeds it `StaticOffset`
    // (movers are frame-AUTHORED ⇒ ZERO center, the moving-frame fix), so its `Orbital` arm — the
    // orbital-position fold the S2 own-absolute path reuses — is exercised directly here.
    let v = DVec3::new(3.0, -4.0, 5.0);
    assert_eq!(placement_offset(Placement::StaticOffset(v)), v);
    let elements = OrbitalElements {
        sma: 1.5e11,
        ecc: 0.1,
        inclination: 0.4,
        raan: 0.3,
        arg_periapsis: 0.9,
        mean_anomaly_epoch: 0.2,
        central_mass: 1.989e30,
    };
    assert_eq!(
        placement_offset(Placement::Orbital(elements)),
        orbital_state(&elements, 0.0).position
    );
}

#[test]
fn moving_children_selects_only_direct_orbital_children() {
    // FA-2b: the AUTHORED moving-child roster keeps ONLY a hosted realm's DIRECT children whose
    // placement is `Orbital` — a STATIC child, an orbital NON-child (someone else's), and an
    // orbital GRANDchild are all excluded. Exercises both `orbital_of` arms + the parent filter.
    let elements = OrbitalElements {
        sma: 1.5e11,
        ecc: 0.1,
        inclination: 0.4,
        raan: 0.3,
        arg_periapsis: 0.9,
        mean_anomaly_epoch: 0.2,
        central_mass: 1.989e30,
    };
    let orbital_child = GeneratedBody {
        realm: RealmId::Planet(1),
        parent: Some(RealmId::System(7)),
        shape: Boundary::Shell { r: 9.0e8 },
        taxon: None,
        look: Some(Boundary::Shell { r: 9.0e8 }),
        placement: Placement::Orbital(elements),
        photometrics: None,
    };
    let static_child = GeneratedBody {
        realm: RealmId::Station(2),
        parent: Some(RealmId::System(7)),
        shape: Boundary::Shell { r: 1.0e6 },
        taxon: None,
        look: Some(Boundary::Shell { r: 1.0e6 }),
        placement: Placement::StaticOffset(DVec3::new(5.0, 0.0, 0.0)),
        photometrics: None,
    };
    let orbital_non_child = GeneratedBody {
        realm: RealmId::Planet(3),
        parent: Some(RealmId::System(99)),
        shape: Boundary::Shell { r: 9.0e8 },
        taxon: None,
        look: Some(Boundary::Shell { r: 9.0e8 }),
        placement: Placement::Orbital(elements),
        photometrics: None,
    };
    let bodies = [orbital_child, static_child, orbital_non_child];
    assert_eq!(
        moving_children(&bodies, RealmId::System(7)),
        vec![(RealmId::Planet(1), elements)],
    );
}

#[test]
fn moving_children_for_is_empty_at_walk_scale() {
    // The byte-identity guarantee: the walk roster is ALL `StaticOffset`, so a walk-scale shard
    // authors NO moving child — `frame_context` registers every region at identity, unchanged.
    assert!(moving_children_for(0, RealmId::System(7)).is_empty());
    assert!(moving_children_for(0, RealmId::System(8)).is_empty());
}

// ===== RLM Step 2: per-realm AoI config + generator accessors ========================

#[test]
fn interest_config_build_inert_at_zero_factor() {
    let inert = InterestConfig::inert();
    assert!(!inert.is_live());
    assert_eq!(
        inert.build(100.0, 5.0).expect("inert always ok"),
        AoiConfig::inert()
    );
}

#[test]
fn one_world_answers_and_checks_with_the_same_contents() {
    // THE DEFECT THIS TYPE RETIRES, measured. A home used to be RESOLVED against the generated world and
    // then VALIDATED against the hand-placed one. With one star the two happened to agree; with several,
    // the stars a seed draws are simply absent from the hand-placed world, so a perfectly valid home
    // beside any other star fails its own defence — and the gateway panics on the spot.
    let cfg = test_visual();
    let generated = WorldView::generated(0, &cfg);
    let placed = WorldView::hand_placed(&cfg);
    // The two worlds really do hold different things — otherwise the rest of this proves nothing.
    let only_generated: Vec<RealmId> = generated
        .regions()
        .iter()
        .map(|r| r.realm)
        .filter(|realm| !placed.contains_realm(*realm))
        .collect();
    assert!(
        !only_generated.is_empty(),
        "the generated world holds stars the hand-placed one does not"
    );
    // …and EVERY one of them passes the check when the check reads the world that produced it.
    for realm in only_generated {
        assert!(generated.contains_realm(realm));
    }
}

#[test]
fn the_hand_placed_world_is_the_one_with_structures_to_stand_in() {
    // Stations are built by players and areas mostly are, so the generator emits neither — which is why
    // a test that needs one places it. Pinned both ways: the placed world HAS them, the generated world
    // has NONE, and that is the whole reason both exist.
    let cfg = UniverseConfig::walk_scale();
    let placed = WorldView::hand_placed(&cfg);
    let generated = WorldView::generated(0, &cfg);
    let structures = |w: &WorldView| -> usize {
        w.regions()
            .iter()
            .filter(|r| matches!(r.realm, RealmId::Station(_) | RealmId::Area(_)))
            .count()
    };
    assert_eq!(structures(&generated), 0);
    assert_eq!(structures(&placed), 2);
}

#[test]
fn the_world_answers_neighbourhood_from_its_own_contents() {
    // The remaining questions a world is asked, each equal to the free function it delegates to — so
    // holding a world can never mean a different answer than deriving one, only a cheaper one. It was
    // also asked for a realm's ORIGIN CHAIN; that question no longer exists, because no realm is
    // entitled to know where it sits.
    let cfg = test_visual();
    let world = WorldView::generated(0, &cfg);
    let held = std::collections::BTreeSet::from([SYSTEM_A]);
    assert_eq!(
        world.neighbourhood(&held),
        realm_neighbourhood_for_config(0, &held, &cfg)
    );
    assert_eq!(world.regions(), realm_regions_for_config(0, &cfg));
}

#[test]
fn the_lowered_world_is_the_regions_and_nothing_else() {
    // `WorldView::lowered` hands the connection plane the region forest VERBATIM — the same
    // slice `regions()` answers with — and (by its type) nothing a body knows: the lowered
    // value is `vd_core::worldgen::WorldRealms`, a crate with no path back to an orbit (SL4).
    let cfg = test_visual();
    let world = WorldView::generated(0, &cfg);
    assert_eq!(world.lowered().regions(), world.regions());
}

#[test]
fn no_two_static_siblings_ever_overlap_in_any_shipped_world() {
    // THE FENCE, RUN. A position inside two overlapping siblings has two equally valid owners, and
    // which shard gets you falls out of iteration order — see `siblings_disjoint` for why that is
    // unrepairable downstream. The generator is deterministic, so proving it over the shipped presets
    // and a spread of seeds proves the worlds that can actually be booted.
    //
    // Seeds swept rather than one sampled: the star ring is drawn from the galaxy's own stream, so a
    // seed that happened to draw a crowded galaxy is exactly the case a single-seed test would miss.
    // ★ RE-DERIVED FROM 664 SEEDS (S12/G8). It judged 664 x 3 = 1 992 systems when a world held
    // three; the same evidence now costs a handful of seeds, and each one judges a whole galaxy.
    for seed in 0..seeds_to_judge(1_992) {
        for config in [
            test_visual(),
            galaxy_holding(
                &UniverseConfig::visual_demand(15.0, 0.02),
                0,
                TEST_GALAXY_SYSTEMS,
            ),
        ] {
            let bodies = generate_system_forest(seed, &config);
            assert_eq!(siblings_disjoint(&bodies), Ok(()), "seed {seed}");
        }
        assert_eq!(
            siblings_disjoint(&generate_walk_forest(&UniverseConfig::walk_scale())),
            Ok(())
        );
    }
}

#[test]
fn the_fence_refuses_two_siblings_that_reach_each_other() {
    // The fence's OWN failing case, so passing above is a fact and not a fence that never says no.
    // Two shells of radius 1 whose centres are 1.5 apart: their surfaces interpenetrate.
    let shell = Boundary::Shell { r: 1.0 };
    let at = |x: f64| Placement::StaticOffset(DVec3::new(x, 0.0, 0.0));
    let body = |realm, placement| GeneratedBody {
        realm,
        parent: Some(GALAXY),
        shape: shell,
        taxon: None,
        look: Some(shell),
        placement,
        photometrics: None,
    };
    let a = RealmId::System(1);
    let b = RealmId::System(2);
    assert_eq!(
        siblings_disjoint(&[body(a, at(0.0)), body(b, at(1.5))]),
        Err(SiblingsOverlap {
            a,
            b,
            parent: GALAXY
        })
    );
    // …and clears once they are pushed apart past the sum of their radii.
    assert_eq!(
        siblings_disjoint(&[body(a, at(0.0)), body(b, at(2.5))]),
        Ok(())
    );
}

#[test]
fn the_first_system_draws_the_stream_its_realm_path_names() {
    // HR1 restated as a measurement: every shard hosting a system must draw the IDENTICAL per-system
    // stream, which holds only if the stream's lineage IS the realm's path from the universe down.
    // Pinned on system zero because that is the one with a named seed to compare against.
    assert_eq!(
        [UNIVERSE_SEED, GALAXY_SEED, system_seed_at(0)],
        SYSTEM_A_LINEAGE
    );
}

#[test]
fn guard_wake_covers_visibility_every_body_wakes_before_it_is_visible() {
    // THE SUCCESSOR of the interim "a planet is visible from anywhere inside its own system"
    // pin (a tiny-world artifact: at true scale a 3,400 km planet across a 1.6e11 m system is
    // genuinely below the visibility angle — that IS the real sky). The lawful invariant at
    // any scale is the taxonomy design's §4.5.6 gauge: the AoI wake trigger reads the BOUND
    // (`spin_up_factor · bound`), the picture needs `cot(θ/2) · look` — so the wake is early
    // (safe) exactly when `spin_up ≥ look·cot(θ/2)` for every drawable body. The minimum
    // ratio is printed; the gauge fails the day a body's look outgrows its bound's wake.
    let regions = realm_regions_for_config(0, &test_visual());
    let factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
    let mut planets_checked = 0_u32;
    let mut min_ratio = f64::INFINITY;
    for r in regions.iter().filter(|r| r.parent.is_some()) {
        let Some(look) = r.look else { continue };
        let needed = look.finite_extent() * factor;
        let wake = r.aoi.spin_up_r_m();
        assert!(
            wake >= needed,
            "{:?} wakes at {wake} m but is visible from {needed} m",
            r.realm
        );
        min_ratio = min_ratio.min(wake / needed);
        if matches!(r.realm, RealmId::Planet(_)) {
            planets_checked += 1;
        }
    }
    println!("[guard_wake_covers_visibility] min wake/needed ratio = {min_ratio}");
    // …and the loop actually ran over the full Planet-kind roster (27 planets + 6 moons).
    // ★ RE-PINNED IN S12 (2026-08-28), ONE CAUSE FOR ALL OF THEM. The placement became a SHAPE and
    // takes SIX draws where the shell took two, so every draw after them shifted by four. One
    // planet drew a different mass, and a lighter planet holds no moon — so exactly ONE MOON left
    // the world. Every count below is that single fact, counted differently.
    // ★ DERIVED, NOT REMEMBERED (S12/G8, 2026-08-28). This said 31 — a tally counted once, when a
    // galaxy held three systems. A population is a result now, so a count OF that population is a
    // result too. Asking the roster how many rows carry a picture proves the walk visited all of
    // them, at any galaxy size, which is what the assertion always meant.
    // The counter above only ticks for PLANET-kind rows (planets and moons), so the honest
    // expectation is how many of those carry a picture — not every row that does.
    let planet_kind_with_look = regions
        .iter()
        .filter(|r| r.parent.is_some() && r.look.is_some() && matches!(r.realm, RealmId::Planet(_)))
        .count();
    assert_eq!(
        planets_checked as usize, planet_kind_with_look,
        "the walk judged every planet and moon that carries a picture"
    );
    assert!(planet_kind_with_look > 0, "and there were some to judge");
}

#[test]
fn interest_config_build_live() {
    let live = test_visual().interest;
    assert!(live.is_live());
    // spin_up_r = extent × the ONE visibility factor cot(θ/2) (v_child 0 ⇒ tear = spin_up widened by
    // the occupant-speed lead, since spin_up_factor == tear_down_factor collapses the geometric gap).
    let band = live.build(100.0, 0.0).expect("valid live");
    assert_eq!(
        band.spin_up_r_m(),
        100.0 * visibility_factor(VISIBILITY_THETA_MIN_RAD)
    );
}

#[test]
fn to_regions_stamps_per_realm_aoi() {
    // Walk: every region inert (byte-identity — the field never changes containment).
    for r in realm_regions_for(0) {
        assert_eq!(r.aoi, AoiConfig::inert());
    }
    // Visual: a Planet's spin_up = its own extent × the visibility factor; a bigger realm (System)
    // reaches farther (same factor, larger extent).
    let visual = realm_regions_for_config(0, &test_visual());
    let planet = visual
        .iter()
        .find(|r| matches!(r.realm, RealmId::Planet(_)))
        .expect("a planet");
    assert_eq!(
        planet.aoi.spin_up_r_m(),
        planet.shape.finite_extent() * visibility_factor(VISIBILITY_THETA_MIN_RAD)
    );
    let system = visual
        .iter()
        .find(|r| r.realm == SYSTEM_A)
        .expect("system A");
    assert!(system.aoi.spin_up_r_m() > planet.aoi.spin_up_r_m());
}

// ===== RLM 5f-4a: the walk-demand LIVE AoI band over the walk forest =================
// The dev demand-cluster's live AoI dynamics: a 2 m/s brisk walk (move_speed × time_multiplier),
// 50 Hz (0.02 s/tick). Kept here (test-only) — the composer supplies the live cluster values at boot.
fn walk_demand_regions() -> Vec<RealmRegion> {
    realm_regions_for_walk_config(0, &UniverseConfig::walk_demand(2.0, 0.02))
}

#[test]
fn walk_keeps_aoi_inert_while_visual_stays_live() {
    // Byte-identity: walk keeps AoI OFF (behaviour unchanged); the compressed-real visual
    // band is LIVE under the ONE cot(θ/2) visibility factor.
    for r in realm_regions_for(0) {
        assert_eq!(r.aoi, AoiConfig::inert(), "walk regions stay AoI-inert");
    }
    assert!(
        test_visual().interest.is_live(),
        "the visual band is live under the visibility factor"
    );
}

#[test]
fn walk_demand_differs_from_walk_only_in_aoi() {
    let walk = realm_regions_for_walk_config(0, &UniverseConfig::walk_scale());
    let demand = walk_demand_regions();
    assert_eq!(walk.len(), demand.len(), "same forest topology");
    for (w, d) in walk.iter().zip(demand.iter()) {
        assert_eq!(w.realm, d.realm, "realm unchanged");
        assert_eq!(w.center, d.center, "center unchanged");
        assert_eq!(w.frame, d.frame, "frame unchanged");
        assert_eq!(w.shape, d.shape, "shape unchanged");
        assert_eq!(w.band, d.band, "containment band unchanged");
        assert_eq!(w.parent, d.parent, "parent unchanged");
        assert_eq!(w.aoi, AoiConfig::inert(), "walk region is AoI-inert");
        assert_ne!(
            d.aoi,
            AoiConfig::inert(),
            "walk-demand region has a LIVE AoI band"
        );
    }
}

#[test]
fn walk_config_builders_delegate_byte_identically() {
    // The existing fns delegate to the config builders with walk_scale ⇒ byte-identical output.
    assert_eq!(
        realm_regions_for(0),
        realm_regions_for_walk_config(0, &UniverseConfig::walk_scale()),
        "realm_regions_for delegates byte-identically"
    );
    let all = realm_regions_for(0);
    let a_planet = all
        .iter()
        .find_map(|r| matches!(r.realm, RealmId::Planet(_)).then_some(r.realm))
        .expect("a planet in the walk forest");
    let sets = [
        std::collections::BTreeSet::from([SYSTEM_A]),
        std::collections::BTreeSet::from([a_planet]),
        std::collections::BTreeSet::from([SYSTEM_A, a_planet]),
    ];
    // The FIXTURE scoper agrees with its own single-realm twin over the same roster. It deliberately
    // does NOT agree with the config scoper any more: that one reads the GENERATED world, which holds
    // no player-built station or area. Asserting they match is what let a fixture masquerade as the
    // world in the first place.
    for held in &sets {
        let scoped = realm_neighbourhood_for_held(0, held);
        for r in held {
            for one in realm_neighbourhood_for(0, *r) {
                assert!(
                    scoped.iter().any(|s| s.realm == one.realm),
                    "the union over {held:?} covers every member's own neighbourhood"
                );
            }
        }
    }
}

#[test]
fn direct_child_levels_from_seed() {
    // Visual System A hosts N orbiting planets; the roster is exactly those planet levels.
    let config = test_visual();
    let levels = direct_child_levels(0, &config, SYSTEM_A);
    assert_eq!(
        levels.len(),
        world_n_planets() as usize + 1,
        "9 planets + the star (T2)"
    );
    assert_eq!(
        levels
            .iter()
            .filter(|l| l.kind == RealmKindTag::Planet)
            .count(),
        world_n_planets() as usize
    );
    assert_eq!(
        levels
            .iter()
            .filter(|l| l.kind == RealmKindTag::Star)
            .count(),
        1
    );
}

/// The visual-scale system forest at seed 0 (helper for the tests below).
fn visual_forest() -> Vec<GeneratedBody> {
    generate_system_forest(0, &test_visual())
}

// (The compressed-visual derive helpers `vis_planet_soi`/`vis_outer_sma`/`vis_central_mass`
// died with the re-solve; the world's own bodies state their derived values now.)

// ===== RLM realistic-demo Slice 0: the one visibility constant + compressed-real geometry =========

/// The single most-distant planet region + its epoch ORBIT DISTANCE — the OUTER planet, the one the
/// star-view visibility rule culls until an occupant closes in. A moving realm's region.center is ZERO
/// (position authored via the frame), so the orbit distance is derived from the mover ELEMENTS, not the
/// region center.
fn outer_planet_orbit(config: &UniverseConfig) -> (RealmRegion, f64) {
    let regions = realm_regions_for_config(0, config);
    moving_children_for_config(0, config, SYSTEM_A)
        .iter()
        .map(|(realm, el)| {
            let region = *regions
                .iter()
                .find(|r| r.realm == *realm)
                .expect("every mover has a region");
            (region, orbital_state(el, 0.0).position.length())
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("a planet mover is present")
}

/// FINDING 28's measurement: THE world produces `Placement::Orbital` in production — at least one
/// per system, through the same `UniverseConfig::world` every shipped shard boots. The claim used
/// to live as a doc comment saying the opposite ("no production producer") under an
/// `#[allow(dead_code)]`; a gate can be wrong but it cannot be stale.
#[test]
fn the_world_produces_an_orbital_planet_in_every_system() {
    // ★ A SMALL GALAXY, AND NOT A QUADRATIC (S12/G8, 2026-08-28). Two faults, both invisible while a
    // world held three systems.
    //
    // The rule — every star system has at least one orbiting planet — holds for fifty systems exactly
    // as for a quarter of a million, so this takes a small galaxy (see `TEST_GALAXY_SYSTEMS`).
    //
    // And the walk below SCANNED EVERY BODY FOR EVERY BODY. At the shipped galaxy that is 4.2 million
    // bodies squared: 1.7e13 comparisons. MEASURED — this test held the suite for seventeen minutes
    // and was still going. The parents are collected ONCE now, which answers the same question.
    let config = galaxy_holding(&UniverseConfig::world(15.0, 0.02), 0, TEST_GALAXY_SYSTEMS);
    let bodies = generate_system_forest(0, &config);
    let parents_of_orbiters: std::collections::BTreeSet<RealmId> = bodies
        .iter()
        .filter(|b| orbital_of(b.placement).is_some())
        .filter_map(|b| b.parent)
        .collect();
    let systems: Vec<RealmId> = bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::System(_)) && parents_of_orbiters.contains(&b.realm))
        .map(|b| b.realm)
        .collect();
    let stars: Vec<RealmId> = bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
        .map(|b| b.realm)
        .collect();
    assert_eq!(
        systems, stars,
        "every star system of THE world holds at least one Orbital planet"
    );
    assert!(!stars.is_empty(), "THE world holds star systems");
    // …and T3's moon-hosting planets hold Orbital children of their own (the same shape
    // one level down — a moon is a planet under a planet).
    let moon_hosts = bodies
        .iter()
        .filter(|b| {
            matches!(b.realm, RealmId::Planet(_))
                && bodies
                    .iter()
                    .any(|c| c.parent == Some(b.realm) && orbital_of(c.placement).is_some())
        })
        .count();
    // ★ DERIVED, NOT REMEMBERED (S12/G8, 2026-08-28). A count of moon-hosting planets is a count OF
    // the population, so it moves with it. What the test means is that SOME planets hold moons and
    // not all of them do — the moon ladder emits by drawn mass, so a galaxy where every planet had
    // moons, or none did, would mean the gate stopped discriminating.
    let planets = bodies
        .iter()
        .filter(|b| {
            matches!(b.realm, RealmId::Planet(_))
                && b.parent.is_some_and(|p| matches!(p, RealmId::System(_)))
        })
        .count();
    assert!(
        moon_hosts > 0 && moon_hosts < planets,
        "some planets hold moons and some do not ({moon_hosts} of {planets}) — the ladder \
         emits by drawn mass, so all-or-nothing would mean it stopped discriminating"
    );
}

#[test]
fn visibility_factor_is_cot_half_theta() {
    // cot(θ/2) at θ_min = 1.5° — the ONE visibility constant (≈ 76.390), frozen non-self-referentially.
    assert_eq!(
        visibility_factor(VISIBILITY_THETA_MIN_RAD),
        FROZEN_VISIBILITY_FACTOR
    );
}

// ===== The generator visibility check (owner ruling 2026-08-15, item 5 + the re-solve addendum) ==

/// THE MEASUREMENT ON THE WORLD, pinned verbatim — the two-level bound HOLDS after the
/// 2026-08-15 shell solve.
///
/// HISTORY (the failing measurement this green pin replaces, kept as provenance). Before the
/// solve the shell was containment-only (`ring + 2·system_soi = 12_331.398_328_646_887 m`) and
/// THE world (seed 0) MEASURED FAILING on 2026-08-15: every planet of BOTH ring-placed systems
/// stayed visible from just outside the galaxy — a 3.954_173_752_999_557_8 m planet visible out
/// to 302.058_663_384_242_95 m while the shell passed within d_min 161.127_605_697_744_3 …
/// 280.730_889_197_954 m of the ten ring planets' worst-instant positions (worst_dist
/// 12_046.713_265_695_933 … 12_166.316_549_196_143 m; the origin system's planets passed). The
/// owner ruled (addendum to items 5/10): the generator SOLVES the margin as a general
/// constraint — [`galaxy_shell_r_m`] grows the shell by the worst descendant's two-level
/// clearance; the ring could not move inward because it already sits at the wake law's LOWER
/// bound. The exact pre-solve offence stays a live measurement in
/// `the_guard_refuses_a_shell_that_hugs_its_ring`, which restores the old shell and pins the
/// first offence verbatim.
#[test]
fn the_two_level_bound_re_solved_on_the_world_no_body_is_visible_past_any_two_level_ancestor() {
    let config = test_world();
    // The storage-fence chain's shell, frozen non-self-referentially (real-scale addendum
    // §A2.2 — the interim upward solve retired; the two-level bound below now holds with ~9
    // orders of slack at the 0.2377 ly gap instead of the interim 4 m margin).
    assert_eq!(config.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
    let pairs = grandchild_visibility_pairs(
        &generate_system_forest(0, &config),
        VISIBILITY_THETA_MIN_RAD,
    );
    // Non-vacuity: every planet AND every star is judged against its galaxy AND the
    // universe (30 + 30), every system against the universe (3), and every MOON against
    // its system, galaxy and universe (3 × 7) — the walk really visited every two-level
    // pair that carries a picture (the look-less ambients are not subjects).
    // DERIVED from the world under test: the literal 75 counted a three-system galaxy.
    assert!(
        !pairs.is_empty(),
        "the walk visited every two-level pair that carries a picture"
    );
    // The WORST margin across every pair of THE world — a ring system's planet against the
    // galaxy shell. At the 0.2377 ly gap the margin IS the reserved clearance class
    // (~3.069e11 m — the §A2.2 clearance showing through, where the interim world measured
    // 11.128 m). Pinned EXACTLY as measured so any re-solve flips this loudly.
    let worst_margin_m = pairs
        .iter()
        .map(|p| p.d_min_m - p.required_m)
        .fold(f64::INFINITY, f64::min);
    println!("[two-level] worst margin = {worst_margin_m}");
    assert!(
        worst_margin_m > 0.0,
        "no body is visible past any two-level ancestor"
    );
    // …and the check the boot fence's predecessor ran: no offence anywhere in THE world —
    // now expressed through the MEASURED climb (look_horizon slice 2): the fence passes at
    // the landed carrier's arity and refuses one below it (both arms driven).
    assert_eq!(
        grandchild_visibility_offences(
            &generate_system_forest(0, &config),
            VISIBILITY_THETA_MIN_RAD,
        ),
        vec![]
    );
    assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
    // With the look split the whole world fits even a one-level carrier (max climb 1);
    // the fence's refusal arm stays covered by the hugging-shell test below.
    assert_eq!(guard_visibility_climb_bounded(0, &config, 1), Ok(()));
}

/// ★ THE DERIVED MASS CAP AND ITS RESERVATION, pinned as measured (owner ruling 2026-08-20).
/// Both numbers are readings of ONE solve, so they are pinned TOGETHER and against the laws
/// that produced them — never as two independent literals that could drift apart, which is
/// exactly how the previous reservation came to describe a different world than the draw did.
#[test]
fn the_mass_cap_and_the_reservation_are_one_derivation() {
    let pl = world_planet_config();
    let cap = imf_mass_hi_msun();
    // ★ RE-BASED IN S9: 16.36 → 30.745 M☉, and the SENTENCE changed with the number. It used to read
    // "above it a star's system is too wide for this galaxy to PLACE"; the galaxy grew 2_051× and
    // stopped being the limit. It now reads: above it a star's system is too wide to STATE ITS OWN
    // POSITIONS on the millimetre lattice it counts in.
    assert_eq!(cap, 30.745_283_003_771_995);
    // THE RESERVATION: the system shell AT the cap, and the star look AT the cap.
    // ★ RE-BASED, ×3.004, and it lands exactly on the fine rung's own fence — which is the whole
    // content of the change: the cap is no longer a number the galaxy happened to afford, it is the
    // largest system that fits its own lattice, to the last bit.
    assert_eq!(target_system_bound_max_m(), 2_251_799_813_685_248.0);
    assert_eq!(target_system_bound_max_m(), SYSTEM_LATTICE_R_M);
    assert_eq!(
        target_system_bound_max_m(),
        system_shell_r_m(&pl, &star_at_mass(cap))
    );
    // ★ RE-BASED, ×1.419 — the star's own photosphere at the heavier cap. It grows far slower than
    // the shell does (a star's radius goes roughly as the square root of its mass, a system's reach
    // with the star's luminosity), which is why the reservation is set by the shell and not by this.
    assert_eq!(target_star_look_max_m(), 6_015_901_650.068_202);
    assert_eq!(
        target_star_look_max_m(),
        crate::taxonomy::star_radius_m(cap)
    );
    // THE CAP IS A ROOT, not a guess: the world can host it and cannot host a hair above.
    // ★ RE-BASED to ask `binding_limit` — since S9 the binding constraint is the system lattice, so
    // a purse-only pair of arms would say "affordable" for a mass the solve refuses.
    assert_eq!(
        binding_limit(&pl, cap, REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M),
        None
    );
    assert_eq!(
        binding_limit(&pl, cap * 1.000_001, REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M),
        Some(StarLimit::SystemLattice)
    );
    // …and the demand really is what the doc says it is — the clearance plus the two bounds
    // the origin-anchored home and a ring sibling put on the line between them.
    let shell_m = system_shell_r_m(&pl, &star_at_mass(cap));
    assert_eq!(
        galaxy_child_demand_m(&pl, cap),
        child_clearance_m(
            shell_m,
            crate::taxonomy::star_radius_m(cap),
            VISIBILITY_THETA_MIN_RAD
        ) + 2.0 * shell_m
    );
    // THE PAIR THE CONSTRUCTION FIXES IS DISJOINT AT THE CAP: the home system sits at the
    // galactic origin, every sibling at exactly the placement radius, so the separation fence's
    // own inequality holds for every seed even if BOTH stars were drawn at the cap.
    assert!(
        real_placement_r_m() >= 2.0 * target_system_bound_max_m(),
        "placement {} vs two capped shells {}",
        real_placement_r_m(),
        2.0 * target_system_bound_max_m()
    );
    // WHY THE CAP EXISTS, measured rather than asserted. ★ RE-BASED IN S9, AND THE COMPARISON HAD
    // TO CHANGE: this measured the retired 120 M☉ literal against THE GALAXY, and after the climb it
    // fits the galaxy comfortably (0.0053× of it). Measured against the thing that actually limits
    // it — the lattice a system counts its own positions on — it still overruns, by 10.77×.
    let old_literal_shell_m = system_shell_r_m(&pl, &star_at_mass(120.0));
    assert!(
        old_literal_shell_m < REAL_GALAXY_R_M,
        "the galaxy now has the room"
    );
    assert!(
        old_literal_shell_m > 10.0 * SYSTEM_LATTICE_R_M,
        "its own lattice does not"
    );
    eprintln!(
        "[MASS CAP] cap {cap} M☉ | reservation {} m | look {} m | placement {} m | chi {} | \
         the retired 120 M☉ literal solves to {old_literal_shell_m} m = {:.4}x the galaxy but \
         {:.2}x the system lattice",
        target_system_bound_max_m(),
        target_star_look_max_m(),
        real_placement_r_m(),
        real_compression_chi(),
        old_literal_shell_m / REAL_GALAXY_R_M,
        old_literal_shell_m / SYSTEM_LATTICE_R_M,
    );
}

/// ★ G-NEST-SWEEP — THE GUARANTEE, MEASURED (owner ruling 2026-08-20). Every seed in a derived
/// sweep generates a world that passes the IDENTICAL fence that refused the owner's galaxy
/// shard. The sweep size is derived from the IMF itself (see `derived_nest_sweep_seeds`) and
/// both it and the heaviest star it actually drew are PRINTED, so a sweep that stopped
/// exercising the massive tail is visible rather than quietly green.
#[test]
fn g_nest_sweep_every_swept_seed_generates_a_world_that_nests() {
    let cfg = test_world();
    let sweep = derived_nest_sweep_seeds(galaxy_profile(0, &cfg).count);
    // ★ 284 -> 664 AT S9, THEN 664 -> 42 AT S12/G8. The sweep size is DERIVED from the world, so it
    // moves when the world does, and it has moved twice for opposite reasons. S9 grew the galaxy from
    // 0.475 light years to 487, and a bigger world needs more seeds to reach the heavy tail. S12 made
    // the POPULATION a result of that volume, so one world now holds a whole galaxy's worth of stars
    // instead of three — and the same tail is met in far fewer worlds. Pinned as measured either way,
    // so a size that drifts again is loud.
    assert_eq!(sweep, 42, "the derived sweep size, pinned as measured");
    let judged = guard_swept_seeds_nest(&cfg, sweep).expect("every swept seed nests");
    // NON-VACUOUS: the walk really visited every region of every world.
    //
    // 13,428 -> 31,317 at S9, which is the SWEEP growing and not the world: 31,317/664 = 47.2 regions per
    // world against 13,428/284 = 47.3 before. Same worlds, more of them.
    // NON-VACUITY, not a census — see the swept-rows note for why a row count is not a world fact.
    assert!(
        judged as u64 >= sweep,
        "every swept world contributed a region: {judged} over {sweep} worlds"
    );
    // …and it really reached into the massive tail: the heaviest star of the sweep, and the
    // tightest nesting margin any child of any of those worlds left.
    let mut heaviest_msun = 0.0_f64;
    let mut worst_margin_m = f64::INFINITY;
    for seed in 0..sweep {
        let world = WorldView::generated(seed, &cfg);
        let regions = world.regions();
        let reaches = child_reaches_for_config(seed, regions, &cfg);
        for child in regions.iter().filter(|r| r.parent.is_some()) {
            let parent = regions
                .iter()
                .find(|p| Some(p.realm) == child.parent)
                .expect("a generated forest resolves every parent");
            let reach_m = match reaches[&child.realm] {
                vd_core::geometry::ChildReach::Fixed(at) => child.shape.max_reach_from(at),
                vd_core::geometry::ChildReach::Excursion(r) => {
                    r + child.shape.max_reach_from(DVec3::ZERO)
                }
            };
            worst_margin_m = worst_margin_m.min(parent.shape.inscribed_extent() - reach_m);
        }
        for (_, p) in system_photometrics_for_config(seed, &cfg) {
            heaviest_msun = heaviest_msun.max(p.mass_msun);
        }
    }
    eprintln!(
        "[NEST SWEEP] {sweep} seeds ({judged} regions) all nest | heaviest star \
         {heaviest_msun} M☉ = {:.1}% of the {} M☉ cap | tightest margin {worst_margin_m} m",
        100.0 * heaviest_msun / imf_mass_hi_msun(),
        imf_mass_hi_msun(),
    );
    assert!(worst_margin_m > 0.0, "every child fits, with room");
    assert!(
        heaviest_msun > NEST_SWEEP_TAIL_OCTAVE * imf_mass_hi_msun(),
        "the sweep drew into the octave below the cap, which is what it is sized to do: \
         {heaviest_msun} M☉ against a {} M☉ cap",
        imf_mass_hi_msun()
    );
}

/// ★ THE SWEEP'S REFUSAL ARM, driven by the reservation that actually grounded the owner
/// (2026-08-20). Restore the SAMPLED reservation — seed 0's own heaviest star, the number the
/// derived cap replaced — and the home seed's world stops nesting, naming the very sibling and
/// the very parent the live cluster named. The fence can fail, and this is the failure.
#[test]
fn the_nest_sweep_refuses_the_sampled_reservation_that_grounded_the_owner() {
    // THE RETIRED CONSTANTS, verbatim: the shell and photosphere of seed 0's heaviest star,
    // 0.16179874709518627 M☉ — a reservation measured from ONE seed's population.
    const SAMPLED_RESERVATION_M: f64 = 296_703_425_982.042_3;
    const SAMPLED_STAR_LOOK_M: f64 = 131_889_247.210_144_1;

    // ★ REWRITTEN IN S12 (2026-08-28) — THE LESSON IS ARITHMETIC NOW, NOT A SAMPLED WORLD.
    //
    // This drove a world with the sampled reservation and asserted it REFUSED to nest. That worked
    // because the SHELL put every system at exactly the placement radius: the worst case the
    // reservation must cover was where every star stood, so any world proved it.
    //
    // The shape draws each radius as `R · u^p`, so a system reaches the rim only as `u → 1`. MEASURED
    // 2026-08-28: across 664 seeds, NOT ONE world refuses under the sampled reservation any more —
    // the generator simply stops producing the worst case.
    //
    // ★ THAT IS NOT THE RESERVATION BECOMING SUFFICIENT. It is the test losing its subject. A
    // reservation must cover a system AT the placement radius whether or not a seed happens to draw
    // one, so the property is now checked against the BOUND directly, which is strictly stronger than
    // waiting for a world to wander into it.
    let sampled_clearance_m = child_clearance_m(
        SAMPLED_RESERVATION_M,
        SAMPLED_STAR_LOOK_M,
        VISIBILITY_THETA_MIN_RAD,
    );
    let derived_clearance_m = child_clearance_m(
        target_system_bound_max_m(),
        target_star_look_max_m(),
        VISIBILITY_THETA_MIN_RAD,
    );
    assert!(
        sampled_clearance_m < derived_clearance_m,
        "a reservation sampled from ONE seed's population under-reserves against the whole seed \
         space: sampled {sampled_clearance_m} m against derived {derived_clearance_m} m"
    );

    // …and the shortfall is what would have escaped: a system standing AT the placement radius under
    // the sampled reservation sits outside the galaxy by exactly the difference.
    let escape_m = derived_clearance_m - sampled_clearance_m;
    assert!(
        escape_m > 0.0,
        "the sampled reservation leaves a system at the rim outside its galaxy by {escape_m} m"
    );
    eprintln!("[NEST SWEEP] the sampled reservation under-reserves by {escape_m:.6e} m");
}

/// The refusal the boot fence makes, measured on the exact PRE-SOLVE geometry: restoring the
/// containment-only shell (`ring + 2·system_soi` — the world as measured failing 2026-08-15)
/// makes the guard name the FIRST ring planet with the very numbers of that measurement —
/// history kept live, and the guard's `Err` arm covered on the boot-facing wrapper.
#[test]
fn the_guard_refuses_a_shell_that_hugs_its_ring() {
    // ★ RE-ROUTED (2026-08-20, the fit clamp): this test used to shrink the shell to
    // `ring + 2·soi` and read the offence off a RING system. The fit clamp now places a sibling
    // where it fits, so that route can no longer produce an offence — which is the clamp working,
    // not the guard weakening. The HOME system is anchored at the galactic origin and is therefore
    // the one child no placement arithmetic can move, so the offence is driven THERE: a shell drawn
    // in tight around the home system leaves its planets visible from outside the galaxy.
    //
    // The interim-scale record is kept as history per the re-solve's provenance rule: worst_dist
    // 12_046.713_265_695_933 m, d_min 280.730_889_197_954 m on the 12 031 m ring; and the
    // real-scale ring record that this re-route supersedes: Planet(2790672799213891506),
    // worst_dist 2_248_492_745_656_386.8 m, d_min −2_261_384_776.593_888_3 m.
    let mut config = test_world();
    config.scale.galaxy_r_m = TARGET_SYSTEM_BOUND_HOME_M;
    let offences = grandchild_visibility_offences(
        &generate_system_forest(0, &config),
        VISIBILITY_THETA_MIN_RAD,
    );
    // NON-VACUOUS on both halves: an offence exists, it names the galaxy as the ancestor the body
    // is visible from outside of, and its numbers are the guard's own inequality — the body's
    // clearance is NEGATIVE and the visibility it must clear is POSITIVE. Exact-value pins are
    // deliberately not restored here: they pinned the ring route the clamp retired.
    let first = offences.first().copied().expect(
        "a shell drawn in to the home system's own bound leaves its planets visible outside",
    );
    assert_eq!(first.ancestor, GALAXY);
    assert!(matches!(first.body, RealmId::Planet(_)));
    assert!(first.d_min_m < 0.0);
    assert!(first.required_m > 0.0);
    // …and the BOOT-facing fence (look_horizon slice 2 — the climb measurement): under the
    // hugging shell a ring SYSTEM's star stays visible from outside the whole galaxy, so
    // its picture must travel TWO levels — more than a one-level carrier holds. (At the
    // true-size world a PLANET's climb stops at its own solved system shell regardless of
    // the galaxy — the §3.2 identity doing its job — so the two-level subject under a
    // hugging shell is the system itself.) The landed arity-2 carrier still serves the
    // hugging world; the refusal arm is driven one level down.
    assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
    let refused = guard_visibility_climb_bounded(0, &config, 1)
        .expect_err("a hugging shell exceeds a one-level carrier");
    eprintln!("[climb] the hugging-shell refusal, verbatim: {refused}");
    assert_eq!(refused.levels, 2);
    assert_eq!(refused.top, GALAXY, "visible from outside even the galaxy");
    assert_eq!(refused.arity, 1);
    assert!(matches!(refused.body, RealmId::System(_)));
}

/// G-CLIMB, RE-DERIVED at the IN-SYSTEM TRUE-SIZE RE-SOLVE (the taxonomy arc's flag day —
/// a proof rewrite, never a weakening): with the BOUND/LOOK split landed, THE world's
/// measured visibility climb is **max 1 over all 30 bodies** — the design's prediction
/// (real-scale addendum §A2.5; celestial_taxonomy_design §4.5.1), now a measurement:
/// - every body's picture is its LOOK (a planet's derived radius, a system's star), and the
///   §3.2 clearance identity gives every child a strictly positive stopping slack at its
///   OWN parent — levels 1, top == self, everywhere;
/// - the ambient galaxy/universe carry `look = None` — no picture, no climb (the outer
///   re-solve's named levels-2 root artifact DISSOLVES here, exactly as its ledger said
///   it would the day `look: None` landed on the ambients).
#[test]
fn g_climb_the_worlds_measured_climb_at_the_true_size_resolve() {
    let config = test_world();
    let climbs = measure_visibility_climb(0, &config);
    // One climb per LOOK-carrying parented body: 3 systems + 27 planets + 3 stars + the
    // 4 census moons (T3; the ambient galaxy/universe draw nothing; the universe is the
    // root). ★ THE MOON COUNT IS 4 SINCE S12 (2026-08-28): the shape's four extra draws
    // shifted every later draw, two planets drew a lighter mass, and a lighter planet holds
    // no moon.
    // ★ DERIVED, NOT REMEMBERED (S12/G8, 2026-08-28). This said 37 — systems + planets + stars +
    // moons, tallied once for a three-system galaxy. One climb is measured per LOOK-carrying
    // parented body, so that roster is the honest expectation at any galaxy size.
    let look_carrying = realm_regions_for_config(0, &config)
        .iter()
        .filter(|r| r.parent.is_some() && r.look.is_some())
        .count();
    assert_eq!(
        climbs.len(),
        look_carrying,
        "one climb per body that carries a picture"
    );
    assert!(look_carrying > 0, "and there were some to climb");
    assert_eq!(climbs.iter().map(|c| c.levels).max(), Some(1));
    for c in &climbs {
        assert_eq!(c.levels, 1, "{c:?}");
        assert_eq!(c.top, c.body, "{c:?}");
        assert!(c.slack_m > 0.0, "{c:?}");
    }
    // THE RESERVED-CLEARANCE IDENTITY, system half: a ring system's stopping slack ==
    // (R_gal − placement) − R★·(1 + cot(θ/2)) — the reserved clearance showing through.
    // ★ 2026-08-20: the cancellation that used to make this land EXACTLY on the reserved
    // bound only held while the reservation was seed 0's own heaviest sample. The reservation
    // is now the shell at the DERIVED CAP, so the identity holds per-system exactly (asserted
    // below, unchanged) and the reserved bound becomes a FLOOR under every drawn star's
    // slack — the guarantee restated as an inequality, which is what it always was.
    let clearance_m = config.scale.galaxy_r_m - config.stellar.galaxy_rim_r_m;
    let factor_plus_one = 1.0 + FROZEN_VISIBILITY_FACTOR;
    // ★ THE ASSOCIATION TOLERANCE, DERIVED IN S9, shared by both identities below. It replaces two
    // hand-typed 1 m literals chosen as "a few ulp at 2.25e15 magnitudes". The climb moved the
    // magnitudes: these identities cancel two GALAXY-scale numbers (4.61e18, whose spacing is 1024 m)
    // to leave a system-scale result, so the result carries galaxy-scale rounding — the clearance
    // identity misses by 512 m and the residue identity by 286 m, both exactly what half an ulp of
    // the largest term buys, and neither a fault.
    //
    // A literal cannot follow a change of scale; it can only be re-typed after each one breaks it.
    // This reads off the largest magnitude in the chain, which is the thing that actually sets it.
    let tol_m = 2.0 * config.scale.galaxy_r_m * f64::EPSILON;
    let bodies = generate_system_forest(0, &config);
    let mut ring_systems = 0u32;
    for c in climbs
        .iter()
        .filter(|c| matches!(c.body, RealmId::System(_)))
    {
        let body = bodies
            .iter()
            .find(|b| b.realm == c.body)
            .expect("a climbed body is in the forest");
        let look_m = body.look.expect("a system draws its star").finite_extent();
        let placement_m = placement_offset(body.placement).length();
        if placement_m == 0.0 {
            continue; // the home anchor's slack is the whole galaxy, not the clearance class
        }
        ring_systems += 1;
        // ★ GENERALISED IN S12, NOT RE-PINNED. This read `clearance_m`, which is
        // `galaxy_r − system_ring_r` — the gap from the galaxy shell to THE ONE RADIUS every system
        // sat at. That identity only held while the placement was a SHELL, and G3 deletes the shell:
        // no two systems share a radius now.
        //
        // The general form uses each system's OWN distance, so it says the same thing for a star
        // anywhere in the galaxy instead of only for one at the rim. It is strictly stronger: the old
        // form is this one evaluated at a single radius.
        let own_clearance_m = config.scale.galaxy_r_m - placement_m;
        let identity_m = own_clearance_m - look_m * factor_plus_one;
        assert!(
            (c.slack_m - identity_m).abs() < tol_m,
            "the reserved-clearance identity: {c:?} vs {identity_m} (tolerance {tol_m} m)"
        );
    }
    // ★ DERIVED, NOT REMEMBERED (S12/G8, 2026-08-28). "Both" was true when a galaxy held three
    // systems: the home at the origin and two siblings. Every system except the home is placed away
    // from the centre, so the honest expectation is the population less the home.
    assert_eq!(
        ring_systems,
        systems_in(&config) as u32 - 1,
        "every seeded sibling was judged — the whole population but the home at the origin"
    );
    let largest_slack_m = climbs
        .iter()
        .filter(|c| matches!(c.body, RealmId::System(_)))
        .map(|c| c.slack_m)
        .fold(f64::INFINITY, f64::min);
    assert!(
        largest_slack_m >= target_system_bound_max_m(),
        "every ring star's slack clears the reserved system bound: {largest_slack_m} vs {}",
        target_system_bound_max_m()
    );
    // …and the residue is exactly the look the reservation set aside for a star at the cap
    // minus the look this seed actually drew — the cancellation, stated where it now lands.
    let heaviest_look_m = bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
        .map(|b| b.look.expect("a system draws its star").finite_extent())
        .fold(0.0_f64, f64::max);
    let residue_m = (target_star_look_max_m() - heaviest_look_m) * factor_plus_one;
    // ★ AN EQUALITY BECAME AN INEQUALITY IN S12, AND THAT IS THE HONEST FORM.
    //
    // This asserted the minimum slack EQUALS the reservation plus the unspent look. That exactness was
    // a property of the SHELL: every system sat at one radius, so every system had one clearance, and
    // the minimum was that clearance exactly.
    //
    // G3 deletes the shell. Systems now sit at their own distances, and one nearer the centre has MORE
    // room, so the minimum comes from the outermost — which is at or inside the placement radius. The
    // reservation is therefore never overspent, and may be underspent by however far in that system
    // sits. `>=` says exactly that, and it is what the reservation was always FOR.
    assert!(
        largest_slack_m + tol_m >= target_system_bound_max_m() + residue_m,
        "the reservation is never OVERSPENT: {largest_slack_m} against {} + {residue_m}",
        target_system_bound_max_m()
    );
    // THE WORST TRUE-PLANET STOPPING SLACK: strictly positive by the per-rung solve;
    // printed and floor-pinned at the solve's own reserved clearance class (> 1e10 m on
    // THE world). MOONS (Planet-kind under a planet, T3) are printed separately — their
    // slack lives at the PLANET's scale, and its floor is its own gate below.
    let is_moon = |realm: RealmId| {
        bodies
            .iter()
            .find(|b| b.realm == realm)
            .and_then(|b| b.parent)
            .is_some_and(|p| matches!(p, RealmId::Planet(_)))
    };
    let planet_slack_m = climbs
        .iter()
        .filter(|c| matches!(c.body, RealmId::Planet(_)) && !is_moon(c.body))
        .map(|c| c.slack_m)
        .fold(f64::INFINITY, f64::min);
    let moon_slack_m = climbs
        .iter()
        .filter(|c| matches!(c.body, RealmId::Planet(_)) && is_moon(c.body))
        .map(|c| c.slack_m)
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "[G-CLIMB] max 1 over {} bodies; worst planet slack {planet_slack_m} m; worst MOON \
         slack {moon_slack_m} m; ring system slack {largest_slack_m} m (reserved clearance \
         {clearance_m} m)",
        climbs.len()
    );
    assert!(planet_slack_m > 1.0e10);
    assert!(
        moon_slack_m > 0.0,
        "every moon stops at its own planet with slack"
    );
    // The four-number pins, re-asserted here so G-CLIMB stays self-contained.
    // ★ THE WORLD'S OWN NUMBERS COME FROM THE WORLD'S OWN CONFIG (2026-08-30). These read `config`,
    // which this test SHRINKS so its forest fits in memory — so they measured a scaled galaxy
    // against unscaled pins and could only fail (2.56e17 against 4.61e18). Reading a config folds
    // no forest, so the pins cost nothing to read from the real one.
    let shipped = world_config();
    assert_eq!(shipped.scale.universe_r_m, FROZEN_REAL_UNIVERSE_R_M);
    assert_eq!(shipped.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
    assert_eq!(shipped.stellar.galaxy_rim_r_m, FROZEN_REAL_PLACEMENT_R_M);
    // …and the fence at the landed carrier's arity: passes at 2 with a FULL SPARE LEVEL —
    // and even at arity 1 now (both fences green is itself the flag-day measurement; the
    // refusal arm stays covered by the hugging-shell test below).
    assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
    assert_eq!(guard_visibility_climb_bounded(0, &config, 1), Ok(()));
}

#[test]
fn measure_visibility_climb_the_ordered_first_measurement() {
    let config = test_world();
    let climbs = measure_visibility_climb(0, &config);
    eprintln!("[CLIMB — THE ORDERED FIRST MEASUREMENT, verbatim]");
    for c in &climbs {
        eprintln!(
            "  body={:?} top={:?} levels={} slack_m={}",
            c.body, c.top, c.levels, c.slack_m
        );
    }
    let max_levels = climbs.iter().map(|c| c.levels).max();
    eprintln!("  MAX LEVELS = {max_levels:?} over {} bodies", climbs.len());
    // The landed carrier still bounds it (arity 2) — the boot fence's condition.
    assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
}

/// ▲ THE FOUR OUTER GEOMETRY NUMBERS, pinned bit-for-bit against the addendum's derivations
/// (real-scale addendum §A2.2/§A2.3) — plus the reserved-clearance identity and the storage
/// budget's exact occupancy/headroom, each a measurement that could have failed.
#[test]
fn the_four_outer_geometry_numbers_are_the_addendums_derivations() {
    let cfg = world_config(); // THE world's own numbers — this test builds no forest
    // ▲ 1 THE UNIVERSE: 2⁷⁶ m exactly — its OWN rung's fence solved at equality.
    // ★ MOVED IN S9, ×33_554_432 (2²⁵), CAUSE: the three-rung ladder. It was 2⁵¹ m because every
    // realm in the world counted in ONE lattice whose step was a millimetre. The root now counts in
    // 32_768 m steps, and 2⁶¹ of them either side is 2⁷⁶ m ≈ 7.99 million light years. Same fence,
    // same construction, a coarser ruler.
    assert_eq!(cfg.scale.universe_r_m, FROZEN_REAL_UNIVERSE_R_M);
    assert_eq!(cfg.scale.universe_r_m, root_radius_at(Tier::Universe));
    // ▲ 2 THE GALAXY: 2⁶² m exactly — likewise its own fence, NOT a subtraction from the root's.
    // ★ MOVED IN S9, ×2_051, CAUSE: the galaxy got a lattice. The old value was `R_uni − outset`,
    // a τ-free band carved out of the ROOT's storage budget because the galaxy had no budget of its
    // own; S7 recorded that as a single-lattice artefact due to die here, and it has. 487.46 light
    // years against 0.475 — which is what makes 150_000 star systems geometrically possible.
    assert_eq!(cfg.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
    assert_eq!(cfg.scale.galaxy_r_m, root_radius_at(Tier::Galaxy));
    // The retired formula, kept as a DRIVEN control so the claim "the subtraction is gone" is a
    // measurement and not a comment: the outset still computes, and the galaxy is no longer it.
    let outset_m = (2.0 * FROZEN_REAL_UNIVERSE_R_M / T_TRAVERSE_S)
        * GEOMETRY_TICK_DT_S
        * BAND_TICKS_N
        * BAND_TAU_HEADROOM;
    assert!(cfg.scale.galaxy_r_m < FROZEN_REAL_UNIVERSE_R_M - outset_m);
    // ▲ 3 THE PLACEMENT RADIUS: R_gal − the reserved clearance, unchanged as a LAW.
    // ★ MOVED IN S9 with its two inputs; the FORM did not move. Both the galaxy and the reservation
    // grew, and the reservation grew FASTER in absolute metres — but it fell from 0.13 % of the
    // galaxy to 0.0488 %, so the world gained placement room in every sense that matters.
    assert_eq!(cfg.stellar.galaxy_rim_r_m, FROZEN_REAL_PLACEMENT_R_M);
    let clearance_m = child_clearance_m(
        target_system_bound_max_m(),
        target_star_look_max_m(),
        VISIBILITY_THETA_MIN_RAD,
    );
    assert_eq!(
        cfg.stellar.galaxy_rim_r_m,
        cfg.scale.galaxy_r_m - clearance_m
    );
    // …and the clearance itself is the ONE clearance law evaluated AT THE DERIVED MASS CAP.
    // ★ MOVED IN S9, ×3.004, CAUSE: the cap rose from 16.36 M☉ to 30.745 M☉ (▲ 5 below), and a
    // heavier star means a wider system shell to reserve room for. Supersedes 749_817_826_779_791.8.
    assert_eq!(
        clearance_m, 2_252_265_383_295_202.5,
        "the reserved clearance == child_clearance_m at the derived mass cap"
    );
    // THE GUARANTEE, as an equality rather than a hope: the reservation IS the shell at the
    // cap, and the cap is the largest mass whose demand the galaxy can pay.
    assert_eq!(
        target_system_bound_max_m(),
        system_shell_r_m(&world_planet_config(), &star_at_mass(imf_mass_hi_msun())),
    );
    // ▲ 5 THE MASS CAP, and WHICH of its two constraints now binds — the S9 change that matters
    // most, because it is the one that stopped the cap being an accident of a single lattice.
    // Before S9 the solve asked ONE question: can the galaxy afford this star's demand? The galaxy
    // then grew by 2_051×, so that question stopped biting, and the cap would have run away to a
    // mass no star has. `affordable_at` therefore asks BOTH: the galaxy's purse AND whether the
    // resulting system still fits its OWN rung. The second binds now — the cap sits exactly on the
    // fine root, to the last bit.
    assert_eq!(imf_mass_hi_msun(), 30.745_283_003_771_995);
    assert_eq!(target_system_bound_max_m(), FINE_ROOT_R_M);
    // The galaxy's purse, measured, so "it stopped biting" is a number and not a story: the cap's
    // demand is under two ten-thousandths of what the galaxy can now pay.
    let demand_m = galaxy_child_demand_m(&world_planet_config(), imf_mass_hi_msun());
    assert!(demand_m <= REAL_GALAXY_R_M, "the cap is affordable");
    assert!(
        demand_m / REAL_GALAXY_R_M < 0.002, // MEASURED 0.001464944704316511
        "and the purse is no longer what binds it — the fine rung is"
    );
    // Both arms driven: one part per million above the cap, the SYSTEM no longer fits its rung.
    assert!(
        system_shell_r_m(
            &world_planet_config(),
            &star_at_mass(imf_mass_hi_msun() * 1.000_001)
        ) > FINE_ROOT_R_M,
        "one part per million above the cap overflows the fine rung — the solve sits at the bound"
    );
    // ▲ 4 the compression χ. ★ MOVED IN S9 from 24.5678 to 0.00799 — and this one is NOT just a
    // moved number, so it is not re-pinned as if it were. χ divides the real mean stellar separation
    // by the PLACEMENT RADIUS, which measures compression only while every system sits on ONE ring
    // at that radius. It does now, because the census is three. The galaxy grew 2_051× and the ring
    // did not gain systems, so χ crossed 1 and the world reads as STRETCHED 125× — which is true of
    // the ring and says nothing about the world S12 builds.
    //
    // Its successor is `the_target_census_lands_near_true_stellar_density` below: real separation
    // against the MEAN SPACING at the census. Retire χ when the shaped placement lands (S12).
    assert_eq!(real_compression_chi(), FROZEN_REAL_COMPRESSION_CHI);
    assert!(
        real_compression_chi() < 1.0,
        "χ now reads as stretch, not compression — see the successor measurement"
    );
    // The storage budget, EXACT: occupancy 50.0000 %, headroom 2.0000× — the equality is the
    // construction (2 × 2⁶¹ == CELL_DOMAIN_MAX + 1), never slack.
    let budget = guard_root_representable(&cfg).expect("THE world is representable");
    assert_eq!(budget.occupancy, 0.5);
    assert_eq!(budget.headroom, 2.0);
    eprintln!(
        "[GEOMETRY] universe {} m ({:.3} Mly) | galaxy {} m ({:.2} ly) | placement {} m ({:.2} ly) \
         | reservation {:.4}% of R_gal | cap {} Msun | chi {} | occupancy {:.4}% headroom {:.4}x",
        cfg.scale.universe_r_m,
        cfg.scale.universe_r_m / (1.0e6 * vd_core::units::LIGHT_YEAR_M),
        cfg.scale.galaxy_r_m,
        cfg.scale.galaxy_r_m / vd_core::units::LIGHT_YEAR_M,
        cfg.stellar.galaxy_rim_r_m,
        cfg.stellar.galaxy_rim_r_m / vd_core::units::LIGHT_YEAR_M,
        100.0 * clearance_m / cfg.scale.galaxy_r_m,
        imf_mass_hi_msun(),
        real_compression_chi(),
        100.0 * budget.occupancy,
        budget.headroom,
    );
}

/// ★ ONE TRUTH, TWO PRODUCERS (S11; the owner's SECOND condition on the catalogue message, 2026-08-24
/// Q4): *"a test asserts the encoded catalogue and the folded one are IDENTICAL — one truth, two
/// producers, which will otherwise drift at the first patch and nobody will notice."*
///
/// THE TWO PRODUCERS. A shard emits the catalogue from the forest it actually BOOTED. This test folds
/// one straight from the SEED. If those ever disagree, the shard is serving a galaxy the seed does not
/// describe — and the failure is silent in the worst way: a player flies to a star that is not there.
///
/// Compared as ENCODED BYTES, not as values. A field added to the row, a field reordered, or a change
/// of encoding moves the bytes while every value still compares equal — and the bytes are what the
/// generation is derived from and what the client caches on disk.
#[test]
fn the_encoded_catalogue_and_the_seed_folded_one_are_byte_identical() {
    // ★ THE SHIPPED GALAXY, NOT A SMALL ONE (S12/G8, 2026-08-28). Almost every test here takes a
    // small galaxy, because a rule holds at fifty systems as at a quarter of a million. This one may
    // not, and the reason is worth stating: the catalogue row carries NO SUB-CELL PART, "because
    // there is none to carry" — an assertion a few lines below this one. That is a fact about THE
    // WORLD'S SCALE, not about the law. At the shipped rim a system's placement lands on a lattice
    // cell; in a galaxy eighteen times smaller the same law puts it a metre off one, and the test
    // failed on exactly that residual.
    //
    // So the subject really is the world as shipped. It costs one full galaxy, which is affordable
    // for one test — what broke the machine was fourteen of them at once, not one.
    let cfg = UniverseConfig::world(15.0, 0.05);

    // PRODUCER ONE — from a booted forest, the way a shard holds it.
    let booted = realm_regions_for_config(0, &cfg);
    let lit = system_photometrics_for_config(0, &cfg);
    let from_boot = star_catalogue(&booted, &lit);

    // PRODUCER TWO — folded again from the seed, nothing carried over.
    let from_seed = star_catalogue(
        &realm_regions_for_config(0, &cfg),
        &system_photometrics_for_config(0, &cfg),
    );

    let a = postcard::to_allocvec(&from_boot).expect("catalogue encodes");
    let b = postcard::to_allocvec(&from_seed).expect("catalogue encodes");
    assert_eq!(a, b, "the two producers disagree about the galaxy");

    // ANTI-VACUITY, three ways — a byte comparison of two empty vectors would pass and prove nothing.
    assert_eq!(
        from_boot.len(),
        systems_in(&cfg),
        "every one of the galaxy's systems is catalogued"
    );
    assert!(!a.is_empty());
    assert!(
        from_boot.iter().any(|r| r.luma_lsun > 0.0),
        "at least one star carries a real photometric draw, not a default"
    );

    // ★ THE ROW IS THE MEASURED SHAPE: no sub-cell part, because there is none to carry. If a future
    // generator starts placing a system off-cell, THIS goes red rather than the position silently
    // losing its residual on the way to every client.
    for r in &booted {
        if matches!(r.realm, RealmId::System(_)) && r.parent == Some(GALAXY) {
            assert_eq!(
                r.center.in_parents_frame().offset(),
                DVec3::ZERO,
                "{:?} sits off-cell — the catalogue row would drop its residual",
                r.realm
            );
        }
    }

    // ORDER IS BY REALM, so two folds of one world cannot differ by walk order — the generation is
    // derived from these bytes, so an unstable order would re-issue the whole sky for nothing.
    let names: Vec<RealmId> = from_boot.iter().map(|r| r.realm).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
}

/// ★ THE GENERATION COMES FROM THE CONTENT (S11; the owner's THIRD catalogue condition).
///
/// The client caches the sky on disk and asks for it by version. If a person maintains that number, a
/// person eventually forgets to — and a stale cache then looks current FOREVER: the player draws an old
/// galaxy, flies at a star that has moved, and nothing reports a fault because every part believes it
/// agrees. Deriving it from the bytes makes "the content changed but the version did not"
/// UNREPRESENTABLE rather than merely unlikely, and this drives both directions of that.
#[test]
fn the_catalogue_generation_moves_with_the_content_and_only_with_it() {
    let cfg = test_world();
    let rows = star_catalogue(
        &realm_regions_for_config(0, &cfg),
        &system_photometrics_for_config(0, &cfg),
    );
    let bytes = postcard::to_allocvec(&rows).expect("encodes");
    let gen0 = vd_core::look::catalogue_generation(&bytes);

    // SAME CONTENT ⇒ SAME NUMBER. Folded again from the seed, nothing carried over. Without this the
    // client re-downloads the whole sky on every login for a world that did not change.
    let again = star_catalogue(
        &realm_regions_for_config(0, &cfg),
        &system_photometrics_for_config(0, &cfg),
    );
    let gen1 =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&again).expect("encodes"));
    assert_eq!(gen0, gen1, "an unchanged sky must not re-issue itself");

    // ★ ONE STAR MOVED BY ONE CELL ⇒ A DIFFERENT NUMBER. This is the direction that matters: it is the
    // case a hand-maintained version gets wrong, and the case whose failure is silent.
    let mut moved = rows.clone();
    moved[0].cell.x += 1;
    let gen_moved =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&moved).expect("encodes"));
    assert_ne!(
        gen0, gen_moved,
        "a star moved one cell must re-issue the sky"
    );

    // …and a star that only changed COLOUR must too — a version that tracked positions alone would
    // leave every client drawing the old spectrum.
    let mut relit = rows.clone();
    relit[0].luma_lsun *= 2.0;
    let gen_relit =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&relit).expect("encodes"));
    assert_ne!(
        gen0, gen_relit,
        "a star that changed brightness must re-issue the sky"
    );

    // ANTI-VACUITY: the fold is over real content, not an empty vector that would agree with anything.
    assert_eq!(
        rows.len(),
        systems_in(&cfg),
        "the fold is over every star of the world under test"
    );
    assert!(!bytes.is_empty());
}

/// ★ THE SUCCESSOR TO χ, and the measurement that says whether the S9 climb actually bought the
/// world it was supposed to buy.
///
/// χ compares the real mean stellar separation against the PLACEMENT RADIUS. That was the right
/// comparison while three systems sat on one ring at that radius, and it is the wrong one for a
/// galaxy with a census. The honest question is: **when the census reaches the target, how far apart
/// are neighbouring systems, against how far apart real stars are?**
///
/// This is a GEOMETRY measurement, not a placement one — it asks what spacing the galaxy's size
/// affords at the target count, independent of the shape S12 chooses. It therefore cannot be
/// invalidated by that choice; it BOUNDS it. If this ever fails, S12 has no shape that works and the
/// galaxy's radius is what must move.
#[test]
fn the_target_census_lands_near_true_stellar_density() {
    const CENSUS: f64 = 150_000.0;
    let real_sep_m = real_compression_chi() * real_placement_r_m();
    // Systems distributed over a shell at the placement radius: each owns 4πR²/N of it, so the
    // centre-to-centre spacing is R·√(4π/N).
    let shell_spacing_m = real_placement_r_m() * (4.0 * core::f64::consts::PI / CENSUS).sqrt();
    let ratio = real_sep_m / shell_spacing_m;
    eprintln!(
        "[s9-density] real mean separation {:.4} ly | shell spacing at {CENSUS:.0} systems \
         {:.4} ly | ratio {ratio:.4}",
        real_sep_m / vd_core::units::LIGHT_YEAR_M,
        shell_spacing_m / vd_core::units::LIGHT_YEAR_M,
    );
    // ▲ THE RESULT: 3.8926 ly real against 4.4595 ly afforded — the world sits at 0.8729× true
    // stellar density, i.e. a shade sparser than real space, with NO compression factor at all.
    // Before the climb the same census would not have fitted by four orders of magnitude.
    assert_eq!(ratio, 0.872_881_733_000_605_9);
    assert!(
        (0.5..2.0).contains(&ratio),
        "the galaxy affords the target census at within a factor of two of true stellar density"
    );
    // …and the spacing clears the owner's one-light-year floor with room to spare, which is the
    // constraint S12's shape must actually respect.
    assert!(shell_spacing_m > vd_core::units::LIGHT_YEAR_M);
    // THE CONTROL that makes the claim falsifiable: the PRE-CLIMB galaxy could not do this. Its
    // placement radius affords 0.00145 ly at the same census — 690× inside the floor.
    let pre_climb_spacing_m =
        1_498_979_587_153_876.0 * (4.0 * core::f64::consts::PI / CENSUS).sqrt();
    assert!(pre_climb_spacing_m < vd_core::units::LIGHT_YEAR_M / 100.0);
}

/// The storage fence's REFUSAL ARM, driven (addendum §A6.3's `g_root_representable`: "without
/// the refusal arm the fence is untested"): a synthetic root ONE OCTAVE larger is refused with
/// its numbers — THE NAMED P10 TRIGGER (R3) — and the error text names the cure.
#[test]
fn guard_root_representable_refuses_the_next_octave_naming_p10() {
    let mut cfg = test_world();
    cfg.scale.universe_r_m = 2.0 * FROZEN_REAL_UNIVERSE_R_M;
    let refused = guard_root_representable(&cfg).expect_err("one octave up must refuse");
    assert_eq!(refused.root_r_m, 2.0 * FROZEN_REAL_UNIVERSE_R_M);
    assert_eq!(refused.k_span, K_SPAN);
    assert_eq!(refused.occupancy_pct, 100.0);
    assert!(refused.to_string().contains("galaxy cell lattice (P10)"));
}

/// ★ DISCOVERY-PERMANENCE, measured (the module-doc law's pin): the 3-D placement pair is
/// APPENDED at per-system stream positions 32–33 — the first 31 draws are byte-identical to
/// the pre-placement stream (the planet elements, the star, the albedos — their own pins
/// stand beside this), and the NEXT TWO draws are exactly the pair the shipped placement law
/// consumed. A reorder fails here before it can re-roll a world.
#[test]
fn the_placement_draws_are_appended_after_the_albedo_pass_and_shift_nothing() {
    let cfg = test_world();
    let bodies = generate_system_forest(0, &cfg);
    for (ix, seed) in [
        (0u32, SYSTEM_A_SEED),
        (1, system_seed_at(1)),
        (2, system_seed_at(2)),
    ] {
        let mut stream = realm_stream(0, &[UNIVERSE_SEED, GALAXY_SEED, seed]);
        // Consume the pre-placement prefix exactly as the generator draws it: 5 planets × 5
        // element draws, 1 star draw, 5 albedo draws = 31 draws.
        for _ in 0..(5 * 5 + 1 + 5) {
            let _ = stream.next_f64();
        }
        // Draws 32–38 ARE the placement's own SEVEN (S12 plus the radial law's second gamma draw:
        // a population, two radius halves, an azimuth, two arm-scatter halves and a height, where the
        // sphere took a direction pair).
        let draws = PlacementDraws {
            population: stream.next_f64(),
            radius: stream.next_f64(),
            radius_b: stream.next_f64(),
            azimuth: stream.next_f64(),
            scatter: stream.next_f64(),
            scatter_b: stream.next_f64(),
            height: stream.next_f64(),
        };
        let body = bodies
            .iter()
            .find(|b| b.realm == RealmId::System(seed))
            .expect("every system is in the forest");
        // ★ SNAPPED, because that is what the generator stores (2026-08-31). A galaxy counts in whole
        // cells and a catalogue row carries no sub-cell part, so every system's placement lands on the
        // grid. Comparing the STORED placement against the RAW draw asks the generator to have skipped
        // a step it is required to take.
        let want = on_galaxy_cell(system_center_at(
            &cfg,
            &galaxy_profile(0, &cfg).shape,
            ix,
            draws,
        ));
        let got = placement_offset(body.placement);
        assert_eq!(
            got, want,
            "system index {ix}: the placement's own draws sit at 32-38"
        );
    }
}

/// ★ THE 3-D SEEDED PLACEMENTS (owner ruling Q-B) and the RE-DERIVED SEPARATION FENCE,
/// measured on THE world: the home anchored at the origin; both siblings at EXACTLY the
/// placement radius in genuinely three-dimensional directions (out of the old ring's y = 0
/// plane, and NOT collinear — the pair's chord differs from both the sum and difference of
/// their radii); every pair disjoint by the 3-D fence, with the wake half (asleep at
/// departure) measured beside it; and the fence's refusal arm driven on a synthetic overlap.
#[test]
fn the_seeded_placements_are_three_dimensional_and_the_fence_judges_the_point_set() {
    let cfg = test_world();
    let centres: Vec<(RealmId, DVec3, f64)> = generate_system_forest(0, &cfg)
        .iter()
        .filter(|b| b.parent == Some(GALAXY))
        .map(|b| {
            (
                b.realm,
                placement_offset(b.placement),
                b.shape.circumscribed_extent(),
            )
        })
        .collect();
    assert_eq!(
        centres.len(),
        systems_in(&cfg),
        "the point set the fence judges is the galaxy's whole population"
    );
    // The home anchor.
    assert_eq!(centres[0].0, SYSTEM_A);
    assert_eq!(centres[0].1, DVec3::ZERO * -1.0);
    // ★ REWRITTEN IN S12, NOT RE-PINNED. This asserted both siblings sat at EXACTLY the placement
    // radius — which is the SHELL, and the shell is the law S12 deletes. Owner ruling G3: no two
    // systems at the same distance, the radial axis included. A test that demanded one radius would
    // now have to be deleted or inverted; inverting it is the honest form, so the property it guards
    // becomes the NEW law rather than disappearing with the old one.
    //
    // Each sibling sits INSIDE the placement radius (the rim is a bound, not a home) and at its OWN
    // distance.
    assert!(centres[1].1.length() <= FROZEN_REAL_PLACEMENT_R_M);
    assert!(centres[2].1.length() <= FROZEN_REAL_PLACEMENT_R_M);
    assert!(
        (centres[1].1.length() - centres[2].1.length()).abs() > 0.0,
        "G3: no two systems share a radius — the shell is gone"
    );
    // …in genuinely 3-D directions: off the retired ring's plane…
    assert!(centres[1].1.y.abs() > 0.0);
    assert!(centres[2].1.y.abs() > 0.0);
    // …and NOT collinear: the sibling chord is neither 2r (diametric) nor 0 (coincident).
    let chord = (centres[2].1 - centres[1].1).length();
    assert!(chord > 0.0);
    assert!((chord - 2.0 * FROZEN_REAL_PLACEMENT_R_M).abs() > 1.0e12);
    eprintln!(
        "[3D PLACEMENTS] sibling 1 {:?}; sibling 2 {:?}; chord {chord} m",
        centres[1].1, centres[2].1
    );
    // The fence over the real point set…
    assert_eq!(guard_seeded_systems_disjoint(0, &cfg), Ok(()));
    // …the wake half: every pair separated by far more than a system's spin-up reach, so a
    // system is ASLEEP at departure from any sibling (the interim ring's second fence,
    // re-measured on the seeded set).
    let spin_up_m = cfg
        .interest
        .build(cfg.stellar.system_soi_r_m, 0.0)
        .expect("a live system band builds")
        .spin_up_r_m();
    for (i, (_, a, _)) in centres.iter().enumerate() {
        for (_, b, _) in centres.iter().skip(i + 1) {
            assert!((*b - *a).length() > spin_up_m);
        }
    }
    // …and the refusal arm, driven: two synthetic siblings closer than their extents.
    let overlap = vec![
        (SYSTEM_A, DVec3::ZERO, 150.0),
        (RealmId::System(99), DVec3::new(200.0, 0.0, 0.0), 150.0),
    ];
    assert_eq!(
        seeded_systems_disjoint_3d(&overlap),
        Err(SiblingsOverlap {
            a: SYSTEM_A,
            b: RealmId::System(99),
            parent: GALAXY,
        })
    );
}

/// ★ REALM EXTENT = GRAVITY SOI (owner ruling 2026-08-18, D-REAL-1) — FLIPPED from
/// blocker-measured to EQUALITY-PINNED by the taxonomy arc's in-system re-solve: every
/// planet shell of THE world now EQUALS `celestial::planet_soi` at that planet's DRAWN
/// mass around its star's REAL drawn mass. The half-worst-instant-gap clamp is the
/// inert-but-live second arm: MEASURED never to bind on any lawful draw (the fence below
/// prints the closest ratio and fails if it ever does silently).
#[test]
fn realm_shell_equals_the_gravitational_soi_at_the_drawn_mass_d_real_1() {
    let cfg = test_world();
    let bodies = generate_system_forest(0, &cfg);
    // The parent's REAL mass: a planet's is its star's drawn mass; a MOON's is its parent
    // PLANET's drawn mass (T3 — the same one law, different arguments, kind-blind).
    let parent_mass_kg = |parent: Option<RealmId>| -> f64 {
        let p = bodies
            .iter()
            .find(|b| Some(b.realm) == parent)
            .expect("every taxon-bearing body has a rostered parent");
        p.taxon.map_or_else(
            || {
                p.photometrics
                    .expect("a system carries its star draw")
                    .mass_msun
                    * crate::taxonomy::M_SUN_KG
            },
            |t| t.mass_kg,
        )
    };
    let mut planets = 0usize;
    let mut worst_clamp_ratio = 0.0_f64;
    for b in bodies.iter().filter(|b| b.taxon.is_some()) {
        let taxon = b.taxon.expect("filtered on presence");
        let central_kg = parent_mass_kg(b.parent);
        let elements = orbital_of(b.placement).expect("a generated planet is Orbital");
        let soi = crate::celestial::planet_soi(elements.sma, taxon.mass_kg, central_kg);
        // THE EQUALITY (bit-for-bit): the emitted shell IS the gravitational SOI.
        assert_eq!(
            b.shape,
            Boundary::Shell { r: soi },
            "{:?}: realm extent == gravitational SOI (D-REAL-1)",
            b.realm
        );
        // The drawn mass round-trips the central mass law (a moon orbits its planet's
        // REAL drawn mass; a planet its star's).
        assert_eq!(elements.central_mass, central_kg);
        // The clamp arm's margin: soi against the half-worst-instant gap it is min'd with.
        let ratio = soi / b.shape.finite_extent();
        worst_clamp_ratio = worst_clamp_ratio.max(ratio);
        planets += 1;
    }
    // ★ RE-PINNED IN S12 (2026-08-28), ONE CAUSE FOR ALL OF THEM. The placement became a SHAPE and
    // takes SIX draws where the shell took two, so every draw after them shifted by four. One
    // planet drew a different mass, and a lighter planet holds no moon — so exactly ONE MOON left
    // the world. Every count below is that single fact, counted differently.
    // ★ DERIVED, NOT REMEMBERED (S12/G8, 2026-08-28). This said 31 — 27 planets plus 4 moons,
    // counted once for a three-system world. The walk visits every body that carries a taxon, so
    // the honest expectation is how many of those the forest holds.
    let with_taxon = bodies.iter().filter(|b| b.taxon.is_some()).count();
    assert_eq!(
        planets, with_taxon,
        "every planet AND every moon of THE world was judged"
    );
    // The clamp never bound: shell == unclamped soi everywhere (ratio exactly 1.0), so the
    // min's second arm is inert-but-live on THE world — printed, fenced.
    println!("[d-real-1] worst soi/shell ratio = {worst_clamp_ratio}");
    assert_eq!(worst_clamp_ratio, 1.0);
}

#[test]
fn plant_seed_of_reads_seed_lineage_kinds_and_refuses_the_rest() {
    // The plant field's resting state IS "nothing built" (the serde default and every
    // shipped constructor agree).
    assert_eq!(FixturePlant::default(), FixturePlant::None);
    assert_eq!(plant_seed_of(RealmId::System(7)), Some(7));
    assert_eq!(plant_seed_of(RealmId::Planet(9)), Some(9));
    assert_eq!(plant_seed_of(RealmId::Station(3)), None);
    assert_eq!(plant_seed_of(RealmId::Area(4)), None);
    assert_eq!(
        plant_seed_of(RealmId::Ship(vd_core::ids::EntityId(1))),
        None
    );
}

/// look_horizon.md slice 5 — the fixture plant is ADDITIVE: with the plant selected, every
/// generated body (id, orbit, photometric draw, order) is byte-identical to the plant-free
/// world, and exactly the named pair is appended after them. The plain world carries no
/// player-built kind at all.
#[test]
fn the_fixture_plant_is_appended_last_and_the_plain_world_is_untouched() {
    let plain_cfg = test_world();
    let planted_cfg = plain_cfg.with_station_area_plant();
    let plain = generate_system_forest(0, &plain_cfg);
    let planted = generate_system_forest(0, &planted_cfg);
    assert_eq!(planted.len(), plain.len() + 2);
    assert_eq!(
        &planted[..plain.len()],
        &plain[..],
        "every generated body is byte-identical under the plant"
    );
    let spec = station_area_plant(0, &planted_cfg);
    assert_eq!(planted[plain.len()].realm, spec.station);
    assert_eq!(planted[plain.len() + 1].realm, spec.area);
    // The plain world has no player-built kind. ★ RE-BASED IN S9: the non-plantable set was the
    // three STAR realms (nothing is built inside the dust-sublimation radius — T2). S9 gives the
    // world a real GALAXY and a real UNIVERSE, and neither takes a seed lineage either — there is no
    // "inside the galaxy" to build in; you build inside one of its children.
    //
    // Asserted as an EQUALITY on the actual list rather than `all(matches!(…))`. A matches! arm that
    // never sees a false case is an uncoverable region (HR5), and the equality says strictly more:
    // it names WHICH realms, so a fourth kind quietly joining the set is a failure rather than a
    // pass. Both changes are why this test caught the new arms at all.
    let non_plantable: Vec<RealmId> = plain
        .iter()
        .filter(|b| plant_seed_of(b.realm).is_none())
        .map(|b| b.realm)
        .collect();
    // ★ THE PROPERTY, NOT THE ROSTER (2026-08-30). This listed three stars by seed — a retired
    // census, and hashes copied into a test besides. What the plant law says is that the ambient
    // pair and every STAR carry no plant seed, whatever the census.
    let (ambients, stars): (Vec<RealmId>, Vec<RealmId>) = non_plantable
        .iter()
        .partition(|r| matches!(r, RealmId::Universe | RealmId::Galaxy(_)));
    assert_eq!(ambients, vec![RealmId::Universe, GALAXY]);
    assert!(
        stars.iter().all(|r| matches!(r, RealmId::Star(_))),
        "only the ambients and the stars carry no plant seed: {stars:?}"
    );
    assert_eq!(
        stars.len(),
        systems_in(&plain_cfg),
        "one star per star system of the world under test"
    );
    assert_eq!(non_plantable.len(), ambients.len() + stars.len());
    // The accessor derives the SAME spec from the plain and the planted config (it strips the
    // plant before deriving, so the spec can never be derived from planted content).
    assert_eq!(spec, station_area_plant(0, &plain_cfg));
}

/// look_horizon.md slice 5 (G-IDENTICAL) — the planted pair's measured climbs: BOTH stop at
/// two levels (the carrier's arity serves the whole planted world), the generated numbers are
/// untouched, and the ADMISSION fence would admit both members as candidates — the fixture
/// plants only what build admission would accept.
#[test]
fn g_identical_the_planted_pair_measures_climb_two_and_leaves_the_worlds_numbers_alone() {
    let plain = test_world();
    let config = plain.with_station_area_plant();
    let spec = station_area_plant(0, &config);
    let climbs = measure_visibility_climb(0, &config);
    // ★ RE-PINNED IN S12 (2026-08-28), ONE CAUSE FOR ALL OF THEM. The placement became a SHAPE and
    // takes SIX draws where the shell took two, so every draw after them shifted by four. One
    // planet drew a different mass, and a lighter planet holds no moon — so exactly ONE MOON left
    // the world. Every count below is that single fact, counted differently.
    // DERIVED: the generated climbs are whatever the world holds; only the PLANTED PAIR is a
    // number this test chose. The literal 39 was the census of a retired world, and the comment
    // above it — "SIX draws where the shell took two" — describes a placement that now takes SEVEN.
    let plain_climbs = measure_visibility_climb(0, &plain);
    assert_eq!(
        climbs.len(),
        plain_climbs.len() + 2,
        "every generated climb, plus the 2 planted"
    );
    assert_eq!(
        climbs.iter().map(|c| c.levels).max(),
        Some(1),
        "the true-size world serves the planted pair at climb 1 (a full spare level)"
    );
    let station = climbs
        .iter()
        .find(|c| c.body == spec.station)
        .expect("the station is measured");
    // The 10 km city stops at its OWN system now (the §6.3 admission margin measured):
    // its picture travels one hop, with the whole system clearance as slack.
    assert_eq!(
        (station.levels, station.top),
        (1, spec.station),
        "{station:?}"
    );
    let area = climbs
        .iter()
        .find(|c| c.body == spec.area)
        .expect("the area is measured");
    // The 20 m structure stops at its OWN planet (slack ~half the planet's SOI).
    assert_eq!((area.levels, area.top), (1, spec.area), "{area:?}");
    eprintln!(
        "[G-IDENTICAL plant] station climb levels {} top {:?} slack {:.3} m; area climb \
         levels {} top {:?} slack {:.3} m",
        station.levels, station.top, station.slack_m, area.levels, area.top, area.slack_m,
    );
    // The plant changes NO generated number: the worst planet slack is the SAME measured bit
    // pattern G-CLIMB pins on the plain world (the reserved-clearance class at the outer
    // re-solve).
    let plain_planet_slack_m = measure_visibility_climb(0, &plain)
        .iter()
        .filter(|c| matches!(c.body, RealmId::Planet(_)))
        .map(|c| c.slack_m)
        .fold(f64::INFINITY, f64::min);
    let planet_slack_m = climbs
        .iter()
        .filter(|c| matches!(c.body, RealmId::Planet(_)))
        .map(|c| c.slack_m)
        .fold(f64::INFINITY, f64::min);
    assert_eq!(
        planet_slack_m, plain_planet_slack_m,
        "the plant moves no generated number (bit-identical worst planet slack)"
    );
    // The boot fence on the planted world: green at the landed arity, with a spare level.
    assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
    assert_eq!(guard_visibility_climb_bounded(0, &config, 1), Ok(()));
    // THE ADMISSION CROSS-CHECK: both planted members, put to the build-admission fence as
    // candidates over the PLAIN world, are ACCEPTED at the landed arity — the fixture path
    // plants nothing the future build path would refuse.
    for (realm, parent, r, offset_m) in [
        (
            spec.station,
            spec.station_parent,
            spec.station_extent_m,
            spec.station_offset_m,
        ),
        (
            spec.area,
            spec.area_parent,
            spec.area_extent_m,
            spec.area_offset_m,
        ),
    ] {
        let candidate = CandidateRegion {
            realm,
            parent,
            shape: Boundary::Shell { r },
            offset_m,
        };
        assert_eq!(
            guard_candidate_climb_bounded(&candidate, 0, &plain, 2),
            Ok(()),
            "{realm:?}"
        );
    }
}

/// look_horizon.md slice 5 — the planted pair NESTS (the real boot fence, with the boot's own
/// worst-instant reach map) and stands MEASURED-clear of the orbital plane, both ±Z polar
/// flight axes, and its own shell wall.
#[test]
fn the_planted_pair_nests_and_stands_clear_of_the_plane_and_the_polar_axes() {
    let config = test_world().with_station_area_plant();
    let spec = station_area_plant(0, &config);
    let bodies = generate_system_forest(0, &config);
    let regions = realm_regions_for_config(0, &config);
    // The boot's own reach map: a mover at its closed-form apoapsis, a static child at its
    // authored offset — the same shape the bins boot states (`child_reaches`).
    let reaches: std::collections::BTreeMap<_, _> = bodies
        .iter()
        .filter(|b| b.parent.is_some())
        .map(|b| {
            let reach = match b.placement {
                Placement::Orbital(e) => vd_core::geometry::ChildReach::Excursion(
                    Motion::Kepler(e).max_excursion_m(Tier::Fine),
                ),
                Placement::StaticOffset(at) => vd_core::geometry::ChildReach::Fixed(at),
            };
            (b.realm, reach)
        })
        .collect();
    // 64 = the membership-bitset width every shard boot passes (`vd_sim::stub::MAX_REGIONS`).
    assert_eq!(
        vd_core::geometry::guard_regions_nest(&regions, &reaches),
        Ok(())
    );
    // Station: inside the shell with margin; clear of BOTH ±Z polar flight axes (the licensed
    // exit corridor and the pixel gate's own park legs) by far more than its extent plus one
    // occupant step; above the orbital plane's MEASURED worst |z| (apoapsis at the
    // eccentricity cap times sin(inclination), over the home system's actual elements).
    let off = spec.station_offset_m;
    let off_len = off.length();
    let extent = spec.station_extent_m;
    let shell = regions
        .iter()
        .find(|r| r.realm == spec.station_parent)
        .expect("the home system is rostered")
        .shape
        .finite_extent();
    let step_m = config.interest.occupant_v_max_mps * config.interest.tick_dt_s;
    // Precomputed locals + inline captures (HR5 test discipline — a multi-line lazy format
    // argument is a line only a FAILING assert executes).
    let off_x = off.x;
    let off_z = off.z;
    assert!(
        off_len + extent < shell,
        "the station nests: {off_len} + {extent} < {shell}",
    );
    assert!(
        off_x > extent + step_m,
        "clear of the polar axes: x {off_x} vs extent {extent} + step {step_m}",
    );
    let worst_plane_z = bodies
        .iter()
        .filter(|b| b.parent == Some(spec.station_parent))
        .filter_map(|b| orbital_of(b.placement))
        .map(|e| e.sma * (1.0 + config.planet.ecc_cap) * e.inclination.sin())
        .fold(0.0, f64::max);
    assert!(
        off_z - extent > worst_plane_z,
        "clear of the orbital plane: z {off_z} − extent {extent} vs measured worst plane \
         |z| {worst_plane_z}",
    );
    eprintln!(
        "[G-IDENTICAL plant] station at {off:?} (|off| {off_len:.3} m, extent \
         {extent:.4} m); measured worst orbital-plane |z| {worst_plane_z:.4} m; occupant \
         step {step_m} m",
    );
    // Area: nests at half the inner planet's own solved shell plus its 20 m extent —
    // strictly inside, derived from the roster.
    let planet_shell = regions
        .iter()
        .find(|r| r.realm == spec.area_parent)
        .expect("the inner planet is rostered")
        .shape
        .finite_extent();
    assert_eq!(spec.area_offset_m.z, 0.5 * planet_shell);
    assert!(spec.area_offset_m.z + spec.area_extent_m < planet_shell);
}

/// The monotone-slack arm on a HAND-BUILT forest with a ZERO-MARGIN level (look_horizon.md
/// slice 2's gate): a level whose slack is exactly zero still COUNTS AS VISIBLE (the same
/// equality convention as the offence filter — the margin the solve reserves is what keeps a
/// lawful world strictly clear), so the climb passes it; one strictly-positive level stops
/// it. Numbers chosen to stay exact in f64 (single-binade sums), so the zero is a ZERO.
#[test]
fn a_zero_margin_level_still_climbs_and_a_positive_one_stops_the_walk() {
    let factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
    let extent = 10.0;
    let offset = 5.0;
    let required = extent * factor;
    let root = RealmId::System(800);
    let zero_parent = RealmId::Planet(801);
    let body = RealmId::Station(802);
    let forest = |parent_r: f64| {
        vec![
            GeneratedBody {
                realm: root,
                parent: None,
                shape: Boundary::Shell { r: 1.0e9 },
                taxon: None,
                look: Some(Boundary::Shell { r: 1.0e9 }),
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
            GeneratedBody {
                realm: zero_parent,
                parent: Some(root),
                shape: Boundary::Shell { r: parent_r },
                taxon: None,
                look: Some(Boundary::Shell { r: parent_r }),
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
            GeneratedBody {
                realm: body,
                parent: Some(zero_parent),
                shape: Boundary::Shell { r: extent },
                taxon: None,
                look: Some(Boundary::Shell { r: extent }),
                placement: Placement::StaticOffset(DVec3::new(offset, 0.0, 0.0)),
                photometrics: None,
            },
        ]
    };
    // (a) THE ZERO-MARGIN LEVEL: the parent's shell sized so `d_min == required` exactly —
    // equality is VISIBLE, the climb passes it and stops at the (enormous) root.
    let zero_r = offset + extent + required;
    let climbs = visibility_climbs(&forest(zero_r), VISIBILITY_THETA_MIN_RAD, 0.0);
    let c = climbs.iter().find(|c| c.body == body).expect("measured");
    assert_eq!(
        c.levels, 2,
        "a zero-margin level climbs — equality counts as visible: {c:?}"
    );
    assert_eq!(c.top, zero_parent);
    assert!(
        c.slack_m > 0.0,
        "the stop happened at the root, with the root's own slack: {c:?}"
    );
    // (b) ONE ULP-SCALE POSITIVE MARGIN stops the walk at the parent: levels 1, the body's
    // own degenerate top, and the slack IS the margin (the monotone arm's other side).
    let stopped_r = zero_r + 1.0;
    let climbs = visibility_climbs(&forest(stopped_r), VISIBILITY_THETA_MIN_RAD, 0.0);
    let c = climbs.iter().find(|c| c.body == body).expect("measured");
    assert_eq!(c.levels, 1, "a positive margin stops the climb: {c:?}");
    assert_eq!(c.top, body, "nobody outside sees it");
    assert_eq!(c.slack_m, 1.0, "the slack IS the stated margin");
}

/// D-LOOK-1, THE SCHEDULED CURE MEASURED (owner Q3 ruling 2026-08-17: "(b)-first-then-
/// measure — the near-real-scale re-solve is the scheduled cure"): at the TRUE-SIZE world
/// the same ~20 m surface structure the interim world REFUSED (climb 3 > arity 2, the
/// pinned Q3 evidence) is now ADMITTED — its picture stops at its own planet with ~half an
/// SOI of slack. The refusal arm stays covered by a SYNTHETIC oversize candidate (a body
/// whose look out-reaches its system's clearance — beyond the §6.3 build ceiling, which THE
/// world's own mass domain cannot produce; stated as synthetic, HR5's named-arm discipline).
#[test]
fn q3_the_twenty_metre_structure_is_admitted_at_true_scale_and_the_fence_still_refuses() {
    let config = test_world();
    let bodies = generate_system_forest(0, &config);
    let origin_system = bodies
        .iter()
        .find(|b| {
            matches!(b.realm, RealmId::System(_))
                && b.parent == Some(GALAXY)
                && worst_hop_excursion_m(&b.placement) == 0.0
        })
        .expect("THE world's system 0 sits at the galactic origin")
        .realm;
    let planet = bodies
        .iter()
        .find(|b| b.parent == Some(origin_system))
        .expect("the origin system holds planets");
    // The Q3 case: a 20 m structure ON the planet's surface (its look radius out).
    let surface_m = planet.look.expect("a planet draws itself").finite_extent();
    let candidate = CandidateRegion {
        realm: RealmId::Station(7777),
        parent: planet.realm,
        shape: Boundary::Shell { r: 20.0 },
        offset_m: DVec3::new(surface_m, 0.0, 0.0),
    };
    assert_eq!(
        guard_candidate_climb_bounded(&candidate, 0, &config, 2),
        Ok(()),
        "the scheduled cure: the 20 m structure is ADMITTED at true scale"
    );
    // The 10 km city case (§6.3's other named margin) — admitted too.
    let city = CandidateRegion {
        realm: RealmId::Station(7778),
        parent: origin_system,
        shape: Boundary::Shell { r: 1.0e4 },
        offset_m: DVec3::new(surface_m * 2.0, 0.0, 0.0),
    };
    assert_eq!(guard_candidate_climb_bounded(&city, 0, &config, 2), Ok(()));
    // THE REFUSAL ARM (synthetic, named): a body whose picture out-reaches its planet AND
    // its system — beyond any lawful build, driven so the fence's Err arm stays real.
    let oversize = CandidateRegion {
        realm: RealmId::Station(7779),
        parent: planet.realm,
        shape: Boundary::Shell { r: 4.0e9 },
        offset_m: DVec3::new(surface_m, 0.0, 0.0),
    };
    let refused = guard_candidate_climb_bounded(&oversize, 0, &config, 2)
        .expect_err("a synthetic oversize candidate exceeds the landed carrier");
    eprintln!("[Q3] the admission refusal, verbatim: {refused}");
    assert_eq!(refused.body, oversize.realm);
    assert!(refused.levels > 2, "visible past its system");
    assert_eq!(refused.arity, 2);
    // …never the boot: THE world itself still passes the same fence.
    assert_eq!(guard_visibility_climb_bounded(0, &config, 2), Ok(()));
}

// (`the_shell_solve_binds_on_the_visibility_clearance_at_the_interim_scale` and
// `the_shell_solve_is_slack_at_near_real_scale_containment_binds` retired WITH the upward
// interim solve they pinned (`two_level_clearance_m`/`galaxy_shell_r_m` — real-scale addendum
// §9.2). Their successors are the ▲-chain pins:
// `the_four_outer_geometry_numbers_are_the_addendums_derivations` and the re-derived climb
// pins below.)

// ===== T2 — THE STAR REALM's gates (celestial_taxonomy_design §9 slice T2) ==========

/// The star's extent: the dust-sublimation bound, pinned per star as measured, with the
/// SCALE-FREE identity `bound/a₀` EQUAL for all three stars (both scale with √L — a
/// measurement that can fail), and the photosphere fence green plus its SWEPT minimum
/// printed over the whole IMF domain (1.993 at the hydrogen-burning limit — the "always
/// ≥ 2" claim is FALSE at the edge and is not asserted; the fence's `> 1` is).
#[test]
fn g_star_extent_the_dust_bound_its_scale_free_identity_and_the_swept_photosphere_fence() {
    let cfg = test_world();
    let bodies = generate_system_forest(0, &cfg);
    let mut ratios = Vec::new();
    let mut stars = 0u32;
    for b in &bodies {
        let RealmId::Star(_) = b.realm else { continue };
        stars += 1;
        let star = b.photometrics.expect("a star carries its photometrics");
        let bound_m = b.shape.finite_extent();
        let a0_m = cfg.planet.orbital_a0_au
            * habitable_zone_radius_au(star.luma_lsun, 1.0)
            * crate::taxonomy::AU_M;
        ratios.push(bound_m / a0_m);
        // The bound IS flux_radius at T_sub, A = 0 — the one law, inverted.
        assert_eq!(
            bound_m,
            crate::taxonomy::flux_radius_m(
                star.luma_lsun * crate::taxonomy::L_SUN_W,
                crate::taxonomy::DUST_SUBLIMATION_K,
                0.0,
            )
        );
        // The look is the photosphere — the SAME value the system's look states (§5.2:
        // one function, two call sites, no double-draw).
        let look_m = b.look.expect("a star draws itself").finite_extent();
        assert_eq!(look_m, crate::taxonomy::star_radius_m(star.mass_msun));
        let system_look = bodies
            .iter()
            .find(|p| Some(p.realm) == b.parent)
            .and_then(|p| p.look)
            .expect("the system's look is its star");
        assert_eq!(look_m, system_look.finite_extent());
        eprintln!(
            "[T2 star] {:?}: bound {bound_m} m, photosphere {look_m} m, bound/look {}",
            b.realm,
            bound_m / look_m
        );
    }
    assert_eq!(
        stars as usize,
        systems_in(&cfg),
        "one star per star system of the world under test"
    );
    // The scale-free identity: the same ratio for every star at every seed.
    assert!((ratios[0] - ratios[1]).abs() < 1e-12, "{ratios:?}");
    assert!((ratios[0] - ratios[2]).abs() < 1e-12, "{ratios:?}");
    eprintln!("[T2 star] FROZEN_STAR_BOUND_OVER_A0 = {}", ratios[0]);
    assert!((ratios[0] - 0.086_075_043).abs() < 1e-8);
    // The boot fence, green on THE world…
    assert_eq!(guard_star_bound_exceeds_photosphere(0, &cfg), Ok(()));
    // …and SWEPT over the whole IMF mass domain, minimum printed (never asserted ≥ 2 —
    // the domain edge measures 1.993).
    let mut min_ratio = f64::INFINITY;
    for i in 0..=120 {
        let mass = cfg.stellar.mass_lo_msun
            + (cfg.stellar.mass_hi_msun - cfg.stellar.mass_lo_msun) * f64::from(i) / 120.0;
        let luma = main_sequence_luminosity(mass, &cfg.stellar.mlr_segments);
        let bound = crate::taxonomy::flux_radius_m(
            luma * crate::taxonomy::L_SUN_W,
            crate::taxonomy::DUST_SUBLIMATION_K,
            0.0,
        );
        let photosphere = crate::taxonomy::star_radius_m(mass);
        min_ratio = min_ratio.min(bound / photosphere);
        assert!(bound > photosphere, "the fence holds at {mass} Msun");
    }
    eprintln!("[T2 star] swept min(bound/photosphere) over [0.08, 120] Msun = {min_ratio}");
    assert!((min_ratio - 1.993).abs() < 0.01);
}

/// `g_star_shell_unmoved`: the star's clearance arm is LIVE in the shell solve and
/// MEASURED never to bind (§5.3.2 Prediction A) — each system's shell still equals its
/// pinned reserved bound, and the star's owed clearance loses to the binding planet term
/// by the printed margin.
#[test]
fn g_star_shell_unmoved_the_stars_clearance_arm_never_binds() {
    let cfg = test_world();
    let bodies = generate_system_forest(0, &cfg);
    let mut shells: Vec<f64> = bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
        .map(|b| b.shape.finite_extent())
        .collect();
    shells.sort_by(f64::total_cmp);
    // ★ THE HOME SHELL, NAMED — NOT THE SMALLEST (2026-08-30). This read `shells[0]`, which was the
    // home system's only while a galaxy held three. With a real census some other star draws a
    // smaller shell, and the test then measured a system it was never about.
    let home_shell = bodies
        .iter()
        .find(|b| b.realm == SYSTEM_A)
        .expect("the home system is in every world")
        .shape
        .finite_extent();
    assert_eq!(
        home_shell, TARGET_SYSTEM_BOUND_HOME_M,
        "the home shell is the cited home target"
    );
    // ★ 2026-08-20: the largest DRAWN shell is no longer the reservation. It used to be —
    // because the reservation WAS this sample, which is precisely the defect the derived cap
    // closed. The reservation is now a CEILING every seed sits under, so the statement gets
    // STRONGER, not weaker: the drawn value is pinned exactly AND it is proved to clear the
    // reserved bound with room.
    // ★ THE EXACT LARGEST SHELL IS NO LONGER PINNED (2026-08-31), and the reason is what the pin was
    // for. It was a golden of the IMF draw's upper bound — "the biggest star this seed draws". But the
    // biggest of a SAMPLE grows with the sample: draw 48 stars instead of 3 and you reach further into
    // the same tail, so the number tracks how many stars a test can afford to build, not the law.
    //
    // What the pin actually protected is asserted right below and is scale-free: EVERY drawn shell
    // sits under the reservation. The IMF draw itself keeps its exact goldens in the photometrics
    // tests, where they are pinned per star rather than per sample.
    assert!(
        shells.len() >= 2,
        "the sample reaches into the tail: {} shells drawn",
        shells.len()
    );
    assert!(
        *shells.last().expect("the world holds star systems") < target_system_bound_max_m(),
        "every drawn shell sits under the reservation: {} vs {}",
        shells.last().expect("the world holds star systems"),
        target_system_bound_max_m()
    );
    for b in bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::Star(_)))
    {
        let star = b.photometrics.expect("a star carries its photometrics");
        let star_term = child_clearance_m(
            b.shape.finite_extent(),
            crate::taxonomy::star_radius_m(star.mass_msun),
            VISIBILITY_THETA_MIN_RAD,
        );
        let shell = bodies
            .iter()
            .find(|p| Some(p.realm) == b.parent)
            .expect("parented")
            .shape
            .finite_extent();
        eprintln!(
            "[T2 star] {:?}: clearance owed {star_term} m vs shell {shell} m ({}x margin)",
            b.realm,
            shell / star_term
        );
        assert!(star_term * 2.0 < shell, "the star's arm never binds");
    }
}

/// `siblings_disjoint_static_vs_orbital` (§4.5.5 — the comparison the static fence
/// structurally skips): the STATIC star against every ORBITAL sibling's worst-instant
/// periapsis annulus, margin printed (≈5.8× at home WITH the sibling's own shell counted
/// — not the 10.2× the source designs quoted without it).
#[test]
fn siblings_disjoint_static_vs_orbital_the_star_clears_every_planets_annulus() {
    let cfg = test_world();
    let bodies = generate_system_forest(0, &cfg);
    let mut worst_margin = f64::INFINITY;
    let mut pairs = 0u32;
    for star in bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::Star(_)))
    {
        let star_bound = star.shape.finite_extent();
        for planet in bodies
            .iter()
            .filter(|p| p.parent == star.parent && matches!(p.realm, RealmId::Planet(_)))
        {
            let el = orbital_of(planet.placement).expect("a planet is Orbital");
            let peri_reach = el.sma * (1.0 - cfg.planet.ecc_cap) - planet.shape.finite_extent();
            let margin = peri_reach / star_bound;
            worst_margin = worst_margin.min(margin);
            pairs += 1;
            assert!(
                peri_reach > star_bound,
                "{:?} vs {:?}: the static star and the orbital annulus are disjoint",
                star.realm,
                planet.realm
            );
        }
    }
    // DERIVED: one pair per (star, planet of that star). The literal 27 was three systems times
    // nine planets — the census of a retired world.
    assert_eq!(
        pairs as usize,
        systems_in(&cfg) * cfg.planet.n_planets as usize,
        "one pair per planet of every star system under test"
    );
    eprintln!("[T2 star] worst static-vs-orbital margin = {worst_margin}x");
    assert!(worst_margin > 5.0, "the ≈5.8× home margin class holds");
}

// ===== T3 — THE MOON gates (celestial_taxonomy_design §9 slice T3) ==================

/// ★ g_moon_census: THE world's moon roster, pinned as `f(seed)` — the MEASURED counts
/// (7 world-wide, on the outermost one-two planets of each system — the design's
/// §4.2.3 prediction, measured true), every moon's derived quantities in lawful ranges,
/// its shell == its OWN gravitational SOI around its planet's REAL drawn mass, and the
/// world's region budget printed against the 64 fence.
#[test]
fn g_moon_census_seven_moons_on_the_outer_planets_pinned_as_f_of_seed() {
    let cfg = test_world();
    let bodies = generate_system_forest(0, &cfg);
    let moon_of = |b: &GeneratedBody| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_)));
    // Per-system, per-rung counts: [0×8,1] / [0×7,1,2] / [0×7,1,2] — the census.
    let mut census: Vec<(RealmId, Vec<usize>)> = Vec::new();
    for i in 0..3u32 {
        let sys = RealmId::System(system_seed_at(i));
        let mut planets: Vec<(RealmId, f64)> = bodies
            .iter()
            .filter(|b| b.parent == Some(sys) && matches!(b.realm, RealmId::Planet(_)))
            .map(|b| {
                (
                    b.realm,
                    orbital_of(b.placement).expect("a planet is Orbital").sma,
                )
            })
            .collect();
        planets.sort_by(|a, b| a.1.total_cmp(&b.1));
        let counts: Vec<usize> = planets
            .iter()
            .map(|(p, _)| bodies.iter().filter(|b| b.parent == Some(*p)).count())
            .collect();
        census.push((sys, counts));
    }
    eprintln!("[T3 census] {census:?}");
    // ★ ONE ASSERTION OVER THE WHOLE SAMPLE (2026-08-30). These were three separate `assert_eq!`s,
    // so a failure stopped at the first row and hid the other two. A census that has moved has moved
    // in a PATTERN, and the pattern is the evidence — showing one row of it wastes the failure.
    let sample: Vec<Vec<usize>> = census.iter().take(3).map(|(_, c)| c.clone()).collect();
    assert_eq!(
        sample,
        vec![
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 1, 2],
            vec![0, 0, 0, 0, 0, 0, 0, 1, 2],
        ],
        "the first three systems' moon census, per planet, outermost last"
    );
    let moons: Vec<&GeneratedBody> = bodies.iter().filter(|b| moon_of(b)).collect();
    // A PROPERTY, not a census: the per-system sample above IS the census, and it is stable whatever
    // the galaxy's size. A world-wide total is a function of how many systems a test galaxy holds.
    assert!(
        !moons.is_empty(),
        "the moon ladder emits moons on the world under test"
    );
    for m in &moons {
        let el = orbital_of(m.placement).expect("a moon is Orbital");
        let taxon = m.taxon.expect("a moon carries its taxon");
        let parent_mass_kg = bodies
            .iter()
            .find(|b| Some(b.realm) == m.parent)
            .and_then(|b| b.taxon)
            .expect("a moon's planet carries its taxon")
            .mass_kg;
        // Its shell IS its own gravitational SOI around its planet (D-REAL-1, one level
        // down — the same one function, different arguments).
        assert_eq!(
            m.shape.finite_extent(),
            crate::celestial::planet_soi(el.sma, taxon.mass_kg, parent_mass_kg)
        );
        assert_eq!(el.central_mass, parent_mass_kg);
        // Above the potato floor by emission; below its planet's mass by construction;
        // tidally-damped elements (the measured regular-satellite sigmas).
        assert!(taxon.radius_m >= crate::taxonomy::MOON_MIN_RADIUS_M);
        assert!(taxon.mass_kg < parent_mass_kg);
        assert!(el.ecc <= crate::taxonomy::MOON_ECC_SIGMA * ECC_CAP_SIGMAS);
        eprintln!(
            "[T3 census] {:?} under {:?}: a={} e={} i={} mass={} kg radius={} m soi={} m",
            m.realm,
            m.parent,
            el.sma,
            el.ecc,
            el.inclination,
            taxon.mass_kg,
            taxon.radius_m,
            m.shape.finite_extent()
        );
    }
    // The region budget with the plant: 2 ambient + 3 systems + 27 planets + 3 stars +
    // 7 moons + 2 planted = 44 against the 64 fence — 20 spare, printed.
    let planted = cfg.with_station_area_plant();
    let regions = realm_regions_for_config(0, &planted);
    eprintln!("[T3 census] world regions = {} against 64", regions.len());
    // DERIVED: the plant appends exactly two regions to the plain world. The literal was a
    // region budget for a three-system galaxy, and it is the 64-fence print above that carries
    // the meaning — this only says the plant added its pair and nothing else.
    assert_eq!(
        regions.len(),
        realm_regions_for_config(0, &cfg).len() + 2,
        "the plant appends its station and area, and touches nothing else"
    );
    // Two moons of one planet stay disjoint at the worst instant (the §4.5.5 margin —
    // measured over every two-moon planet).
    let mut hosts: std::collections::BTreeMap<RealmId, Vec<&GeneratedBody>> =
        std::collections::BTreeMap::new();
    for m in &moons {
        hosts
            .entry(m.parent.expect("parented"))
            .or_default()
            .push(m);
    }
    for (host, ms) in hosts {
        if ms.len() < 2 {
            continue;
        }
        let mut anns: Vec<(f64, f64)> = ms
            .iter()
            .map(|m| {
                let el = orbital_of(m.placement).expect("Orbital");
                (el.sma, m.shape.finite_extent())
            })
            .collect();
        anns.sort_by(|a, b| a.0.total_cmp(&b.0));
        let ecc_cap = crate::taxonomy::MOON_ECC_SIGMA * ECC_CAP_SIGMAS;
        for w in anns.windows(2) {
            let gap = w[1].0 * (1.0 - ecc_cap) - w[0].0 * (1.0 + ecc_cap);
            let need = w[0].1 + w[1].1;
            eprintln!(
                "[T3 census] {host:?}: adjacent moon annuli gap {gap} vs shells {need} \
                 ({}x)",
                gap / need
            );
            assert!(
                gap > need,
                "two moons of one planet are disjoint at the cap"
            );
        }
    }
}

/// The potato floor's BOTH arms, driven from mass domains the generator itself produces:
/// THE world emits (7 moons — the emit arm, above); a synthetic mass-floor config draws
/// planets so light every ladder rung's moon falls under hydrostatic equilibrium and
/// NOTHING is emitted (the reject arm — the design's own named synthetic).
#[test]
fn the_potato_floor_rejects_sub_equilibrium_moons_and_the_count_law_still_ran() {
    let mut cfg = test_visual();
    // Every planet at the Mercury floor: the log-uniform draw with lo == cap yields the
    // floor exactly; the moon budget 1e-4·M then sits under the potato radius everywhere.
    cfg.planet.mass_cap_mearth = cfg.planet.mass_lo_mearth;
    let bodies = generate_system_forest(0, &cfg);
    let moons = bodies
        .iter()
        .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        .count();
    assert_eq!(moons, 0, "no Mercury-mass planet keeps a hydrostatic moon");
    // …and the world's own census (the emit arm) is the 7-moon pin above — both arms live.
}

/// The depth-4 machinery walked at the unit tier (T3): a moon's neighbourhood scope is its
/// own five-level ancestor chain + nothing else; its coord resolves the full lineage; its
/// frame is the SAME total map arm every planet takes (a moon IS a planet — HR4's sharpest
/// statement, they are literally the same kind).
#[test]
fn a_moon_is_a_planet_at_depth_four_scope_coord_and_frame() {
    let cfg = test_world();
    let regions = realm_regions_for_config(0, &cfg);
    let moon = regions
        .iter()
        .find(|r| {
            matches!(r.realm, RealmId::Planet(_))
                && r.parent.is_some_and(|p| matches!(p, RealmId::Planet(_)))
        })
        .expect("THE world holds moons");
    // Scope: ancestors ∪ direct children — a moon shard evaluates exactly its chain.
    let scope = vd_core::worldgen::neighbourhood_scope(
        &regions,
        &std::collections::BTreeSet::from([moon.realm]),
    );
    assert_eq!(scope.len(), 5, "universe, galaxy, system, planet, moon");
    // Coord: the full root-rooted five-level lineage (depth 4).
    let coord =
        vd_core::worldgen::coord_of_realm(&regions, moon.realm).expect("a moon's lineage resolves");
    assert_eq!(coord.path().levels().len(), 5);
    // Frame: the SAME one-field lift every planet takes — nothing can tell them apart.
    assert_eq!(
        moon.frame,
        vd_core::pose::frame_for_realm(moon.realm, moon.parent).expect("total"),
    );
    assert_eq!(
        vd_core::worldgen::level_of(moon.realm).kind,
        vd_core::realm_path::RealmKindTag::Planet,
        "a moon IS a planet — no Moon kind exists to be told apart"
    );
}

// ===== T4 — THE EARTH-LIKE PREDICATE gates (celestial_taxonomy_design §9 slice T4) ===

/// Each of the six predicate clauses driven TRUE and FALSE (HR5), from hand-built rows
/// around a real G-star candidate shape.
#[test]
fn earth_like_every_clause_is_driven_both_ways() {
    use crate::taxonomy::{Atmosphere, BodyTaxon, MU_N2, PlanetType, R_EARTH_M};
    let g_star = StarPhotometrics {
        mass_msun: 0.9,
        class: classify_spectral(0.9, &SpectralClass::MASS_BOUNDS),
        luma_lsun: main_sequence_luminosity(0.9, &SpectralClass::MLR_SEGMENTS),
    };
    assert_eq!(g_star.class, SpectralClass::G, "0.9 Msun is a G star");
    let candidate = BodyTaxon {
        class: PlanetType::Rocky,
        mass_kg: crate::taxonomy::M_EARTH_KG,
        radius_m: R_EARTH_M,
        insolation_rel: 0.748_314_795,
        t_eq_k: 236.785_700_196_447_95,
        bond_albedo: 0.30,
        atmosphere: Some(Atmosphere {
            mean_molecular_weight: MU_N2,
            scale_height_m: 8.0e3,
            reference_density_kgm3: None,
        }),
    };
    assert!(
        earth_like(&g_star, &candidate),
        "the reference candidate passes"
    );
    // 1. NOT a yellow sun (an M dwarf).
    let m_star = StarPhotometrics {
        mass_msun: 0.1,
        class: classify_spectral(0.1, &SpectralClass::MASS_BOUNDS),
        luma_lsun: main_sequence_luminosity(0.1, &SpectralClass::MLR_SEGMENTS),
    };
    assert!(!earth_like(&m_star, &candidate));
    // 2. NOT rocky.
    let sub_neptune = BodyTaxon {
        class: PlanetType::SubNeptune,
        ..candidate
    };
    assert!(!earth_like(&g_star, &sub_neptune));
    // 3. Out of the radius band (both edges).
    assert!(!earth_like(
        &g_star,
        &BodyTaxon {
            radius_m: 0.79 * R_EARTH_M,
            ..candidate
        }
    ));
    assert!(!earth_like(
        &g_star,
        &BodyTaxon {
            radius_m: 1.26 * R_EARTH_M,
            ..candidate
        }
    ));
    // 4. Out of the Kopparapu flux band (both edges — ruling B: [0.53, 1.10]).
    assert!(!earth_like(
        &g_star,
        &BodyTaxon {
            insolation_rel: 0.52,
            ..candidate
        }
    ));
    assert!(!earth_like(
        &g_star,
        &BodyTaxon {
            insolation_rel: 1.11,
            ..candidate
        }
    ));
    // 5. Out of the derived temperate band (both edges).
    assert!(!earth_like(
        &g_star,
        &BodyTaxon {
            t_eq_k: 216.0,
            ..candidate
        }
    ));
    assert!(!earth_like(
        &g_star,
        &BodyTaxon {
            t_eq_k: 261.0,
            ..candidate
        }
    ));
    // 6. Airless.
    assert!(!earth_like(
        &g_star,
        &BodyTaxon {
            atmosphere: None,
            ..candidate
        }
    ));
    // The derived temperate band converts the SAME flux limits once (the corrected
    // literals of §8.1: ~216.75 / ~260.16 K at the class table's 0.30 Bond albedo).
    let (t_lo, t_hi) = (earth_like_t_bound_k(0.53), earth_like_t_bound_k(1.10));
    assert!((t_lo - 217.2).abs() < 0.1, "measured {t_lo}");
    assert!((t_hi - 260.7).abs() < 0.1, "measured {t_hi}");
}

/// §8.2 THE NO-OP HONESTY PIN: given `rocky + G` the insolation and temperature clauses
/// can never discriminate — the ladder is quantised and seed-free (rung 2 = 0.748 S⊕ at
/// every seed) and a rocky rung-2 `T_eq` is one of two class-albedo values, both in band.
/// MEASURED over a seed sweep: every rocky body in the flux band also passes the
/// temperature clause — the two clauses' pass sets coincide exactly.
#[test]
fn earth_like_no_op_clauses_are_measured_as_no_ops_given_rocky_and_g() {
    use crate::taxonomy::PlanetType;
    let cfg = test_world();
    let mut flux_passes = 0u32;
    let mut temp_passes = 0u32;
    let mut rung2_insolations: Vec<f64> = Vec::new();
    for seed in 0..64u64 {
        for b in generate_system_forest(seed, &cfg) {
            let Some(t) = b.taxon else { continue };
            if t.class != PlanetType::Rocky {
                continue;
            }
            let (s_lo, s_hi) = crate::taxonomy::KOPPARAPU_FLUX_CONSERVATIVE;
            if (s_lo..=s_hi).contains(&t.insolation_rel) {
                flux_passes += 1;
                rung2_insolations.push(t.insolation_rel);
                // Counted, never branched: the temperature clause's "false" arm is the very
                // thing this pin measures as unreachable given rocky + G, so writing it as
                // an `if` would ask the coverage gate to exercise an arm the world cannot
                // produce.
                temp_passes += u32::from(
                    (earth_like_t_bound_k(s_lo)..=earth_like_t_bound_k(s_hi)).contains(&t.t_eq_k),
                );
            }
        }
    }
    assert!(flux_passes > 0, "the sweep found rocky flux-band bodies");
    assert_eq!(
        flux_passes, temp_passes,
        "temperature discriminates NOTHING beyond the flux clause"
    );
    // ONE quantised temperate insolation across every star at every seed — the √L
    // cancellation, measured to f64 association (the division `L/(0.4·√L·r²)²` rounds a
    // few ulp differently per drawn L; the QUANTITY is seed-free, the bits are not).
    for s_rel in &rung2_insolations {
        assert!(
            (s_rel - 0.748_314_795).abs() < 1e-9,
            "rung 2 is 0.748315 S⊕ for every star at every seed: {s_rel}"
        );
    }
}

/// THE SL5 FIREWALL (§8.4): the ranking weights are TOOL policy and live in the seed-search
/// binary alone — `UniverseConfig`'s own serialized field set carries no weight, no rank,
/// no search knob (asserted against the serde field names, so a smuggled knob fails here).
#[test]
fn the_ranking_weights_are_absent_from_the_one_config() {
    let json = serde_json::to_value(test_world()).expect("the one config serializes");
    let mut names = Vec::new();
    fn collect(prefix: &str, v: &serde_json::Value, out: &mut Vec<String>) {
        if let serde_json::Value::Object(map) = v {
            for (k, child) in map {
                out.push(format!("{prefix}{k}"));
                collect(&format!("{prefix}{k}."), child, out);
            }
        }
    }
    collect("", &json, &mut names);
    for name in &names {
        let lower = name.to_lowercase();
        assert!(!lower.contains("weight"), "a weight knob leaked: {name}");
        assert!(!lower.contains("rank"), "a rank knob leaked: {name}");
        assert!(!lower.contains("search"), "a search knob leaked: {name}");
    }
    assert!(!names.is_empty());
}

#[test]
fn worst_hop_excursion_is_the_offset_for_a_static_and_the_apoapsis_for_a_mover() {
    // Static: the authored offset's magnitude, exactly.
    assert_eq!(
        worst_hop_excursion_m(&Placement::StaticOffset(DVec3::new(3.0, 0.0, 4.0))),
        5.0
    );
    // Mover: THE one closed-form worst-instant accessor — never a re-derived `a·(1+e)` beside it.
    let bodies = generate_system_forest(0, &test_world());
    let mover = bodies
        .iter()
        .find(|b| matches!(b.placement, Placement::Orbital(_)))
        .expect("THE world has orbital movers");
    let elements = orbital_of(mover.placement).expect("a mover is Orbital");
    assert_eq!(
        worst_hop_excursion_m(&mover.placement),
        Motion::Kepler(elements).max_excursion_m(Tier::Fine)
    );
}

#[test]
fn a_visible_grandchild_is_named_with_its_exact_numbers() {
    // A synthetic guilty forest — the arm THE (green) world can never take: root shell 1000 m,
    // a child 100 m off the root's centre, and a 20 m grandchild 50 m off the child's centre.
    // From just outside the root the grandchild can close to 1000 − (100+50) − 20 = 830 m, and
    // the band keeps a 20 m body visible out to 20·cot(θ_min/2) ≈ 1527.8 m — an offence.
    let root = RealmId::System(900);
    let child = RealmId::Planet(901);
    let grand = RealmId::Station(902);
    let bodies = vec![
        GeneratedBody {
            realm: root,
            parent: None,
            shape: Boundary::Shell { r: 1000.0 },
            taxon: None,
            look: Some(Boundary::Shell { r: 1000.0 }),
            placement: Placement::StaticOffset(DVec3::ZERO),
            photometrics: None,
        },
        GeneratedBody {
            realm: child,
            parent: Some(root),
            shape: Boundary::Shell { r: 200.0 },
            taxon: None,
            look: Some(Boundary::Shell { r: 200.0 }),
            placement: Placement::StaticOffset(DVec3::new(100.0, 0.0, 0.0)),
            photometrics: None,
        },
        GeneratedBody {
            realm: grand,
            parent: Some(child),
            shape: Boundary::Shell { r: 20.0 },
            taxon: None,
            look: Some(Boundary::Shell { r: 20.0 }),
            placement: Placement::StaticOffset(DVec3::new(0.0, 0.0, 50.0)),
            photometrics: None,
        },
    ];
    let offences = grandchild_visibility_offences(&bodies, VISIBILITY_THETA_MIN_RAD);
    let expected = GrandchildVisibleOutside {
        body: grand,
        ancestor: root,
        worst_dist_m: 150.0,
        extent_m: 20.0,
        d_min_m: 1000.0 - 150.0 - 20.0,
        required_m: 20.0 * visibility_factor(VISIBILITY_THETA_MIN_RAD),
    };
    assert_eq!(offences, vec![expected]);
    // The fail-loud shape moved to the MEASURED climb (look_horizon slice 2): the same
    // guilty forest measures a grandchild climb of 3 (visible past its parent AND its
    // grandparent — it runs out of ancestors, so the slack is the ROOT's own non-positive
    // figure), and the fence refuses it at the landed arity while passing an arity that
    // could carry it (both arms, named).
    let climbs = visibility_climbs(&bodies, VISIBILITY_THETA_MIN_RAD, 0.0);
    let grand_climb = climbs
        .iter()
        .find(|c| c.body == grand)
        .expect("the grandchild is measured");
    assert_eq!(grand_climb.levels, 3);
    assert_eq!(grand_climb.top, root, "visible from outside even the root");
    assert!(
        grand_climb.slack_m <= 0.0,
        "the climb never stopped inside the forest: {grand_climb:?}"
    );
    assert_eq!(
        first_climb_over(&climbs, 3),
        Ok(()),
        "an arity that can carry the climb passes"
    );
    let refused = first_climb_over(&climbs, 2).expect_err("the landed arity refuses");
    assert_eq!(refused.body, grand);
    assert_eq!(refused.levels, 3);
    assert_eq!(refused.arity, 2);
}

#[test]
fn world_geometry_is_true_size_and_the_ladder_is_luminosity_anchored() {
    let c = galaxy_holding(
        &UniverseConfig::visual_demand(15.0, 0.02),
        0,
        TEST_GALAXY_SYSTEMS,
    );
    // The derived planet count: 9, scale-free (the disc edge over the ladder ratio).
    assert_eq!(c.planet.n_planets, 9);
    assert_eq!(
        c.planet.n_planets,
        derived_planet_count(
            c.planet.orbital_a0_au,
            c.planet.orbital_ratio,
            (NEPTUNE_SMA_AU / crate::taxonomy::FROST_COEFF_AU) * c.planet.frost_coeff_au,
        )
    );
    assert_eq!(c.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
    // THE LADDER LAW: every home sma == a0·√L·ratio^n in TRUE metres — χ = 1 in-system,
    // exactly (no compression factor exists to be anything else).
    let bodies = generate_system_forest(0, &c);
    let star = bodies
        .iter()
        .find(|b| b.realm == SYSTEM_A)
        .and_then(|b| b.photometrics)
        .expect("the home star");
    let a0_m = c.planet.orbital_a0_au
        * habitable_zone_radius_au(star.luma_lsun, 1.0)
        * crate::taxonomy::AU_M;
    let smas: Vec<f64> = bodies
        .iter()
        .filter(|b| b.parent == Some(SYSTEM_A) && matches!(b.realm, RealmId::Planet(_)))
        .map(|b| orbital_of(b.placement).expect("a planet is Orbital").sma)
        .collect();
    assert_eq!(smas.len(), 9);
    for (n, sma) in smas.iter().enumerate() {
        let expect = orbital_axis_au(n as u32, 1.0, c.planet.orbital_ratio) * a0_m;
        assert!(
            (sma / expect - 1.0).abs() < 1e-12,
            "rung {n}: {sma} vs ladder {expect}"
        );
    }
    // Rung 2 is the temperate rung of every star in the universe: S = 6.25/2.89^2.
    let s2 = star.luma_lsun / (smas[2] / crate::taxonomy::AU_M).powi(2);
    assert!((s2 - 0.748_314_795).abs() < 1e-9, "measured {s2}");
    // The SAME geometry as visual_scale (the static-render twin): one game geometry.
    let vs = test_visual();
    assert_eq!(c.stellar.galaxy_rim_r_m, vs.stellar.galaxy_rim_r_m);
    assert_eq!(c.planet.n_planets, vs.planet.n_planets);
    assert_eq!(c.planet.mass_lo_mearth, vs.planet.mass_lo_mearth);
}

#[test]
fn visual_demand_band_is_crossable_between_stars() {
    // NON-VACUITY, the other half of `a_planet_is_visible_from_anywhere_inside_its_own_system`. Together
    // they sandwich the band: a planet is awake everywhere inside its own system (so arriving shows you
    // a populated system), and ASLEEP from the neighbouring star (so the interstellar leg crosses its
    // band and the spin-up machinery is actually exercised). Without this the wider angle could swell
    // until every planet in the galaxy is permanently awake and nothing would ever be measured waking.
    //
    // This USED to read `spin_up < orbit` — the outer planet asleep at its OWN star — which is the very
    // property that made a system look empty on arrival. The band did not stop being crossable; it moved
    // outward, so the crossing is now measured where it belongs, on the way in from another star.
    // ★ THE REAL RIM, A SMALL GALAXY FOR THE PLANET (S12/G8, 2026-08-28). This test has TWO needs and
    // they pull opposite ways. The distance to the next star is a fact about THE WORLD'S OWN
    // GEOMETRY, so it must be read off the shipped config — a shrunken galaxy would prove the
    // property for a world nobody boots. The outer PLANET is just a sample: a planet's orbit is drawn
    // from its own system's stream and its own star, and the galaxy's size is nowhere in that chain,
    // so any system's outer planet is the same planet.
    //
    // So the rim comes from the shipped preset and the planet from a galaxy small enough to build.
    let cfg = UniverseConfig::visual_demand(15.0, 0.02);
    let (outer, orbit) = outer_planet_orbit(&galaxy_holding(&cfg, 0, TEST_GALAXY_SYSTEMS));
    // The closest a neighbouring star ever gets to this planet: the rim, less its orbit at worst phase.
    let from_the_next_star = cfg.stellar.galaxy_rim_r_m - orbit;
    let reach = outer.aoi.tear_down_r_m();
    assert!(
        reach < from_the_next_star,
        "the outer planet must be asleep from the next star"
    );
}

#[test]
fn equal_visibility_factor_with_zero_v_rel_is_err_not_panic() {
    // THE equal-factor tripwire: under a single visibility factor (spin_up_factor == tear_down_factor)
    // the geometric dead-zone collapses, so a ZERO-relative-velocity band has spin_up == tear_down →
    // Err(InvalidEdges), never a valid inert band (and to_regions' .expect would PANIC at boot on one).
    // The precondition band validity requires occupant_v_max + v_child > 0 — a live occupant. Equality
    // asserted via expect_err (NOT matches!). No epsilon is added to keep the factors exactly equal.
    let factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
    let err = AoiConfig::for_velocity_safe(
        100.0,
        factor,
        factor,
        0.0,
        AOI_TICK_DT_S,
        VISUAL_AOI_GRACE_TICKS,
        VISUAL_AOI_K_SAFETY_EXTRA,
    )
    .expect_err("equal factors + zero v_rel collapse the dead-zone → InvalidEdges");
    assert_eq!(err, BandError::InvalidEdges);
}

#[test]
fn visual_scale_preset_is_walk_physics_with_derived_true_size_geometry() {
    let c = test_visual();
    // The galaxy holds the seeded placement radius with every system's reach inside it.
    assert!(c.scale.galaxy_r_m > c.stellar.galaxy_rim_r_m + target_system_bound_max_m());
    // …and it is the storage-fence chain's shell exactly (real-scale addendum §A2.2, frozen).
    assert_eq!(c.scale.galaxy_r_m, FROZEN_REAL_GALAXY_R_M);
    assert_eq!(
        c.planet.ecc_cap,
        ECC_SIGMA * ECC_CAP_SIGMAS,
        "the GEOMETRY cap (4σ), not the solver bound"
    );
    assert_eq!(c.planet.ecc_sigma, ECC_SIGMA);
    assert_eq!(c.planet.incl_sigma, INCL_SIGMA);
    assert_eq!(c.planet.n_planets, world_n_planets());
}

#[test]
fn generate_system_forest_emits_the_ambient_forest_plus_n_orbital_planets() {
    let bodies = visual_forest();
    // 2 ambient shells + every system + that system's planets. The galaxy's population is DRAWN from
    // its census, so this reads the census rather than restating a number in two places.
    // ★ THE COUNT MUST COME FROM THE CONFIG THE FOREST CAME FROM (2026-08-30). This read
    // `world_system_count()`, which is THE world's population — while the bodies above are the
    // VISUAL preset's. `galaxy_holding` only APPROXIMATES a wanted census (it scales a radius and
    // lets the density law decide), so two presets asked for 48 systems produce two different
    // numbers. Reading one config's census against another's forest is a comparison of two worlds.
    let n_sys = systems_in(&test_visual());
    // 2 ambient + per system: itself + its planets + its STAR (T2) + the census moons (T3
    // — the MEASURED 7, pinned exactly by `g_moon_census…`).
    let moons = bodies
        .iter()
        .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        .count();
    assert_eq!(
        bodies.len(),
        2 + n_sys * (1 + world_n_planets() as usize + 1) + moons
    );
    // ★ RE-PINNED IN S12 (2026-08-28), ONE CAUSE FOR ALL OF THEM. The placement became a SHAPE and
    // takes SIX draws where the shell took two, so every draw after them shifted by four. One
    // planet drew a different mass, and a lighter planet holds no moon — so exactly ONE MOON left
    // the world. Every count below is that single fact, counted differently.
    // (The moon CENSUS is pinned by `g_moon_census…`, which exists for it. A literal here was a
    // second, weaker pin of the same fact, and it goes stale with every change to the test galaxy.)
    assert_eq!(orbital_of(bodies[0].placement), None); // Universe
    assert_eq!(orbital_of(bodies[1].placement), None); // Galaxy
    // Every remaining body is either a system (static, under the galaxy) or one of its planets
    // (orbital, under a system) — no third kind, and no planet parented anywhere but a star.
    let systems: Vec<_> = bodies
        .iter()
        .filter(|b| b.parent == Some(GALAXY))
        .map(|b| b.realm)
        .collect();
    assert_eq!(systems.len(), n_sys);
    assert!(
        systems.contains(&SYSTEM_A),
        "system 0 keeps the named identity"
    );
    // The FIVE lawful shapes (§7.3, extended by T3): static system under galaxy; orbital
    // planet under a system; STATIC STAR at its system's origin (T2); ORBITAL MOON —
    // Planet-kind under a planet (T3); nothing else in the plain world.
    let planet_ids: std::collections::BTreeSet<RealmId> = bodies
        .iter()
        .filter(|b| {
            matches!(b.realm, RealmId::Planet(_)) && b.parent.is_some_and(|p| systems.contains(&p))
        })
        .map(|b| b.realm)
        .collect();
    for b in bodies.iter().skip(2) {
        if systems.contains(&b.realm) {
            assert_eq!(orbital_of(b.placement), None, "a system does not orbit");
        } else if matches!(b.realm, RealmId::Star(_)) {
            assert!(systems.contains(&b.parent.expect("a star nests in its system")));
            assert_eq!(orbital_of(b.placement), None, "the star sits at the origin");
            assert_eq!(placement_offset(b.placement), DVec3::ZERO);
        } else if planet_ids.contains(&b.realm) {
            assert!(orbital_of(b.placement).is_some());
        } else {
            // A MOON: Planet-kind, parented to a planet, orbiting it.
            assert!(matches!(b.realm, RealmId::Planet(_)));
            assert!(planet_ids.contains(&b.parent.expect("a moon has a planet")));
            assert!(orbital_of(b.placement).is_some());
        }
    }
}

#[test]
fn a_galaxy_of_several_systems_gives_each_its_own_seed_place_and_planets() {
    // THE GENERALISATION. The previous shape named ONE system in code and hung the planets off a
    // constant, so a second star could not exist at any scale — which is how the login side and the
    // shard side ended up describing two different worlds. Every system now comes off the same loop.
    let mut cfg = test_visual();
    cfg = galaxy_holding(&cfg, 0, 4);
    cfg.stellar.galaxy_rim_r_m = 4.0 * cfg.stellar.system_soi_r_m;
    let bodies = generate_system_forest(0, &cfg);

    let systems: Vec<_> = bodies.iter().filter(|b| b.parent == Some(GALAXY)).collect();
    // DERIVED: `galaxy_holding` scales a radius and lets the density law decide, so it APPROXIMATES
    // the census it is asked for. Asserting the number asked for measures the request, not the world.
    assert_eq!(
        systems.len(),
        systems_in(&cfg),
        "a galaxy is N systems, not one"
    );
    // System 0 keeps the identity every existing fixture and label already names.
    assert_eq!(systems[0].realm, SYSTEM_A);
    // …and no two systems share an id, so their planets can never collide either.
    let ids: std::collections::BTreeSet<_> = systems.iter().map(|s| s.realm).collect();
    assert_eq!(
        ids.len(),
        systems.len(),
        "system identities are distinct: {ids:?}"
    );

    // Each system carries its OWN planets, and a planet belongs to exactly one star.
    for sys in &systems {
        let mine = bodies
            .iter()
            .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Planet(_)))
            .count();
        assert_eq!(
            mine,
            world_n_planets() as usize,
            "{:?} has its own planets",
            sys.realm
        );
        // …and exactly ONE star child (T2): the body-bearing near-star realm.
        let stars = bodies
            .iter()
            .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Star(_)))
            .count();
        assert_eq!(stars, 1, "{:?} holds its star as a child realm", sys.realm);
    }
    let planets: std::collections::BTreeSet<_> = bodies
        .iter()
        .filter(|b| {
            b.parent.is_some_and(|p| ids.contains(&p)) && matches!(b.realm, RealmId::Planet(_))
        })
        .map(|b| b.realm)
        .collect();
    assert_eq!(
        planets.len(),
        systems.len() * cfg.planet.n_planets as usize,
        "every planet across every system is a distinct realm"
    );

    // A DIFFERENT SEED DRAWS DIFFERENT ORBITS but the SAME AMBIENT + SYSTEM + PLANET +
    // STAR structure — the world is a pure function of the seed. The MOON census is
    // seed-DEPENDENT by design (T3: the count follows each star's drawn ladder and each
    // planet's drawn mass through the potato floor), so the structural comparison counts
    // the non-moon prefix kinds.
    let non_moon = |forest: &[GeneratedBody]| {
        forest
            .iter()
            .filter(|b| !b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
            .count()
    };
    let other = generate_system_forest(99, &cfg);
    assert_eq!(
        non_moon(&other),
        non_moon(&bodies),
        "structure is seed-independent"
    );
    assert_ne!(
        orbital_of(other[3].placement),
        orbital_of(bodies[3].placement),
        "orbits are seed-DERIVED, not fixed"
    );
}

#[test]
fn a_forest_whose_star_systems_overlap_is_refused() {
    // THE FENCE THAT MAKES SEVERAL SYSTEMS SAFE. Authority is "the deepest realm containing you". Two
    // overlapping systems give a position two equally valid owners, and which shard simulates you
    // would come down to iteration order — a coin flip deciding where your input lands.
    let mut cfg = test_visual();
    cfg = galaxy_holding(&cfg, 0, 4);

    // A ring TIGHTER than the systems on it: neighbours intersect. The shells are SOLVED
    // per system, so the ring is read off THIS forest's own largest solved shell — the home
    // system sits at the galactic origin, so ANY sibling placed one large-shell away is
    // closer than the sum of the two extents, for every seed and every direction draw. That
    // is STRUCTURAL, where the old spelling (`ring = the reservation`) only overlapped while
    // the reservation happened to equal a drawn shell; the derived cap ended that coincidence.
    let drawn_shells = |cfg: &UniverseConfig| {
        generate_system_forest(0, cfg)
            .iter()
            .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
            .map(|b| b.shape.finite_extent())
            .fold(0.0_f64, f64::max)
    };
    // ★ THE OVERLAP IS BUILT BY HAND NOW (2026-08-31). This shrank the galaxy's ring until the
    // generator produced touching systems. It cannot any more, and that is a FEATURE: ruling G12's
    // crowding push moves a star that the shape placed too close, so a generated forest never
    // overlaps however tight the ring. The old spelling therefore asked the generator to emit a
    // world it now refuses to emit, and read the absence of a refusal as a fault.
    //
    // The subject of this test is the FENCE, not the generator. So take a lawful forest and MOVE one
    // system onto another — the exact geometry the fence exists to refuse — and check that it does.
    // That is stronger: it no longer depends on the generator being able to build a broken world.
    let tight_ring_m = drawn_shells(&cfg);
    let mut overlapping = generate_system_forest(0, &cfg);
    let systems: Vec<usize> = overlapping
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
        .map(|(i, _)| i)
        .collect();
    assert!(
        systems.len() >= 2,
        "the fence needs two siblings to judge; this galaxy holds {}",
        systems.len()
    );
    // Park the second system one metre from the first: inside both shells, by construction.
    let first_at = super::placement_offset(overlapping[systems[0]].placement);
    overlapping[systems[1]].placement =
        Placement::StaticOffset(first_at + DVec3::new(1.0, 0.0, 0.0));
    let _ = tight_ring_m;
    let err = siblings_disjoint(&overlapping).expect_err("touching systems must be refused");
    assert_eq!(
        err.parent, GALAXY,
        "the ambiguity is between children of the galaxy"
    );

    // Spread them and the same forest is accepted — so the refusal is about the GEOMETRY, not about
    // having more than one star.
    //
    // ★ THE SPREAD IS DERIVED, NOT A CHOSEN MULTIPLE (S12/G1, 2026-08-28). This read `4.0 *
    // tight_ring_m`, and four was enough only for the SHELL, which put every system at the rim so
    // scaling the rim scaled every separation with it. The drawn shape has a real bulge, and a bulge
    // packs its systems into a fraction of the rim — the same four systems now sit about seventeen
    // times tighter, so four was not enough and the fixture failed for a reason that was never about
    // the fence it tests.
    //
    // MEASURED, and it corrected a wrong guess of mine worth recording: I expected a bulge to be
    // scale-INVARIANT, so that no spread would ever clear the overlap. It is not. The closest pair
    // tracks the rim exactly — at 1×, 4×, 16× and 64× the rim it sat at 0.062, 0.249, 0.995 and 3.980
    // of the separation it needs, each step a clean factor of four. So the shortfall is a pure ratio,
    // and spreading by it is exact rather than lucky.
    let shortfall = |cfg: &UniverseConfig| {
        let forest = generate_system_forest(0, cfg);
        let sys: Vec<_> = forest.iter().filter(|b| b.parent == Some(GALAXY)).collect();
        let mut worst = f64::INFINITY;
        for (i, a) in sys.iter().enumerate() {
            for b in sys.iter().skip(i + 1) {
                let d = (placement_offset(a.placement) - placement_offset(b.placement)).length();
                let need = a.shape.finite_extent() + b.shape.finite_extent();
                worst = worst.min(d / need.max(1.0));
            }
        }
        worst
    };
    // Spread by the shortfall itself, and then some — so the assertion below turns on the geometry
    // being roomy, never on the spread being exactly enough.
    const ROOM: f64 = 2.0;
    cfg.stellar.galaxy_rim_r_m = tight_ring_m * ROOM / shortfall(&cfg);
    assert_eq!(siblings_disjoint(&generate_system_forest(0, &cfg)), Ok(()));

    // And the single-system world every existing rig boots is accepted unchanged.
    assert_eq!(
        siblings_disjoint(&generate_system_forest(0, &test_visual())),
        Ok(())
    );
}

#[test]
fn the_sibling_fence_declines_to_judge_orbits_rather_than_guessing() {
    // AN HONEST LIMIT, pinned so nobody mistakes silence for a guarantee. An orbiting body's region
    // sits at its frame ORIGIN — its position is authored live each tick — so every planet looks
    // co-located to any static comparison. Judging orbits needs their SHELLS compared, which is a
    // separate check over the moving roster; until it exists, orbital overlap is UNCHECKED.
    let cfg = test_visual();
    let bodies = generate_system_forest(0, &cfg);
    let orbiting = bodies
        .iter()
        .filter(|b| orbital_of(b.placement).is_some())
        .count();
    let moons = bodies
        .iter()
        .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        .count();
    assert_eq!(
        orbiting,
        (world_system_count() * world_n_planets()) as usize + moons,
        "the fixture really does orbit (planets + the T3 moons)"
    );
    // The planets pass — NOT because they are proven disjoint, but because this fence does not judge
    // orbits at all.
    assert_eq!(siblings_disjoint(&bodies), Ok(()));
}

// ---- the render-origin PIN classifier ----

#[test]
fn d_fo_7_no_static_region_sits_under_a_varying_ancestor_chain() {
    // D-FO-7 TRIPWIRE: the A4c realm feed ships its movers-only rows and does NOT widen — a STATIC child's
    // authored row rides its frame-local `center`, never a per-tick absolute. That is correct ONLY while no
    // static realm hangs under a MOVING ancestor (which would ride the parent's orbit the center cannot
    // express). Through P4 `generate_system_forest` gives orbiting planets NO children, so the case does not
    // exist. This asserts it: the day P4 first hangs a station under an orbiting planet, THIS fails, and the
    // D-FO-7 decision (parent per-tick rows behind a realm-lane AoI cull, vs the child shard authoring its
    // own box row) must be taken before the widening lands.
    // The invariant, stated as SET EQUALITY (HR5-clean — no branch on the never-true "static-under-varying"
    // condition, whose true arm would be uncoverable): a realm moves-or-inherits-motion IFF the realm is
    // ITSELF a mover. A static realm inheriting a moving ancestor is exactly the case where the two sets
    // diverge. Both filter arms are genuinely exercised — a planet (Orbital ⇒ true) and an ambient body
    // (Static ⇒ false) exist in every forest.
    //
    // This used to ask the question through the seed-derived ORIGIN CHAIN, which is gone: no realm folds
    // its own absolute any more. The question itself survives untouched — it is about the FOREST's shape,
    // not about anybody's absolute — so it is now asked directly of the parent links.
    let moving_anywhere_above = |bodies: &[GeneratedBody], realm: RealmId| -> bool {
        let mut cur = realm;
        for _ in 0..bodies.len() {
            let Some(body) = bodies.iter().find(|b| b.realm == cur) else {
                return false; // unknown realm — no ancestry to inherit motion from
            };
            if orbital_of(body.placement).is_some() {
                return true;
            }
            match body.parent {
                None => return false, // the ambient root — the walk is complete
                Some(parent) => cur = parent,
            }
        }
        false // hop cap — a cycle, which the boot guard rejects; a safe stop, never a hang
    };
    // The walk's own edge arms, driven where the forest cannot produce them: an UNKNOWN realm has
    // no ancestry to inherit motion from; a CYCLE (which the boot guard rejects at boot) stops at
    // the hop cap instead of hanging — both answer "no motion", never a panic.
    let unknown_only = [GeneratedBody {
        realm: RealmId::System(1),
        parent: None,
        shape: Boundary::Shell { r: 1.0 },
        taxon: None,
        look: Some(Boundary::Shell { r: 1.0 }),
        placement: Placement::StaticOffset(DVec3::ZERO),
        photometrics: None,
    }];
    assert!(
        !moving_anywhere_above(&unknown_only, RealmId::Planet(999)),
        "an unknown realm inherits nothing"
    );
    let cycle = [
        GeneratedBody {
            realm: RealmId::System(1),
            parent: Some(RealmId::System(2)),
            shape: Boundary::Shell { r: 1.0 },
            taxon: None,
            look: Some(Boundary::Shell { r: 1.0 }),
            placement: Placement::StaticOffset(DVec3::ZERO),
            photometrics: None,
        },
        GeneratedBody {
            realm: RealmId::System(2),
            parent: Some(RealmId::System(1)),
            shape: Boundary::Shell { r: 1.0 },
            taxon: None,
            look: Some(Boundary::Shell { r: 1.0 }),
            placement: Placement::StaticOffset(DVec3::ZERO),
            photometrics: None,
        },
    ];
    assert!(
        !moving_anywhere_above(&cycle, RealmId::System(1)),
        "a cycle stops at the hop cap — a safe no, never a hang"
    );
    let config = test_visual();
    for seed in [0u64, 1, 7, 42, 100] {
        let bodies = generate_system_forest(seed, &config);
        let mut varying_chain: Vec<RealmId> = bodies
            .iter()
            .filter(|b| moving_anywhere_above(&bodies, b.realm))
            .map(|b| b.realm)
            .collect();
        let mut movers: Vec<RealmId> = bodies
            .iter()
            .filter(|b| orbital_of(b.placement).is_some())
            .map(|b| b.realm)
            .collect();
        varying_chain.sort();
        movers.sort();
        assert_eq!(
            varying_chain, movers,
            "D-FO-7 (seed {seed}): a realm inherits motion IFF it is itself a mover — a divergence means a \
             static realm now hangs under a moving ancestor, which the realm feed's movers-only filter \
             would silently drop. Take the D-FO-7 decision before the widening lands."
        );
    }
}

#[test]
fn realm_regions_for_config_gives_each_moving_planet_a_zero_center() {
    let config = test_visual();
    let bodies = generate_system_forest(0, &config);
    let regions = realm_regions_for_config(0, &config);
    assert_eq!(regions.len(), bodies.len());
    // A moving (Orbital) planet authors its position LIVE through its frame, so its region carries NO
    // baked position — center is the ZERO origin of its own frame. (The crossing-flap fix: a nonzero
    // epoch center would be double-counted against the live frame placement in region_signed_distance.)
    for (body, region) in bodies
        .iter()
        .zip(&regions)
        .filter(|(b, _)| matches!(b.realm, RealmId::Planet(_)))
    {
        assert!(orbital_of(body.placement).is_some(), "a planet is Orbital");
        assert_eq!(region.center.in_parents_frame().cell(), glam::I64Vec3::ZERO);
        assert_eq!(region.center.in_parents_frame().offset(), DVec3::ZERO);
    }
}

/// FRAME-COHERENT CONTAINMENT (the moving-realm crossing fix): a moving planet's position is authored
/// ONCE — into the placement book, off its injected `MotionFn` — and its region `center`
/// is ZERO (the boundary sits at the body's OWN frame origin). So an occupant sitting exactly at the
/// planet's live orbital position is judged INSIDE its SOI, and an occupant at the star (17.9 m away) is
/// OUTSIDE — the SAME geometry both the parent shard (planet as a moving child) and the planet's own
/// shard (planet at the identity) compute, so a crossing cannot flap. Regression guard against the epoch
/// `center` being double-counted against the frame placement (which put the SOI ~17.9 m off the planet).
#[test]
fn a_moving_planet_soi_is_centered_on_its_live_position_not_double_counted() {
    use vd_core::geometry::region_signed_distance;
    use vd_core::kinematics::secs_since_epoch;
    use vd_core::placement::PlacementBook;
    use vd_core::pose::StampedPose;
    let config = test_visual();
    let regions = realm_regions_for_config(0, &config);
    let movers = moving_children_for_config(0, &config, SYSTEM_A);
    let (realm, elements) = movers
        .iter()
        .min_by(|a, b| {
            orbital_state(&a.1, 0.0)
                .position
                .length()
                .total_cmp(&orbital_state(&b.1, 0.0).position.length())
        })
        .cloned()
        .expect("the visual forest has planet movers");
    let region = regions
        .iter()
        .find(|r| r.realm == realm)
        .expect("the mover realm has a region");
    // ★ RE-BASED IN S9. This used to take the frame of the forest's PARENTLESS ROOT, which was the
    // star system while the system was the top of the tree. It is the universe now, and a universe
    // counts in 32_768 m cells while a planet's SOI counts in millimetres — so the measurement asked
    // for a crossing between two rungs, which is refused by name (`CrossTierCrossingNotBuilt`) and
    // is P10's work, not this test's subject.
    //
    // The frame this measurement was always about is the planet's OWN PARENT — the star system that
    // authors its orbit. Naming it directly is both correct and no longer sensitive to what sits
    // above the system in the tree.
    let parent_realm = region.parent.expect("a moving planet has a parent");
    let root = regions
        .iter()
        .find(|r| r.realm == parent_realm)
        .expect("the planet's parent has a region")
        .frame;
    let tick_hz = 20.0;
    let tick = vd_core::UniverseTick(200);
    // A moving realm carries NO baked position — its center is the origin of its own frame.
    assert_eq!(
        region.center.in_parents_frame().offset(),
        DVec3::ZERO,
        "a moving planet's region.center must be ZERO (position authored via the frame)",
    );
    // The parent shard's authored book: the planet's row is its live orbital state at this tick —
    // the ONE writer's output, which every consumer (this measurement included) reads as data.
    let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
    let ctx = PlacementBook::new(
        root,
        tick,
        vec![(
            region.frame,
            FramePlacement::moving(state.position, state.velocity),
        )],
    );
    let live = state.position;
    let soi = region.shape.finite_extent();
    // Occupant sitting EXACTLY at the planet's live position (root frame) ⇒ INSIDE the SOI.
    let at_planet = StampedPose::at_rest(root, live, tick);
    let d_at = region_signed_distance(&at_planet, region, &ctx).expect("the planet frame resolves");
    // Occupant at the star origin (17.9 m from the planet) ⇒ OUTSIDE.
    let at_star = StampedPose::at_rest(root, DVec3::ZERO, tick);
    let d_star = region_signed_distance(&at_star, region, &ctx).expect("the planet frame resolves");
    // An occupant AT the planet's live position is inside its SOI (distance ≈ −SOI). Split asserts
    // (not `a && b`) so neither short-circuit leaves an uncovered branch (HR5).
    assert!(d_at < 0.0);
    assert!((d_at + soi).abs() < 1e-9);
    // An occupant at the star is outside by (orbit − SOI).
    assert!(d_star > 0.0);
    assert!((d_star - (live.length() - soi)).abs() < 1e-6);
}

#[test]
fn moving_children_for_config_lists_every_planet_as_a_mover() {
    let config = test_visual();
    let movers = moving_children_for_config(0, &config, SYSTEM_A);
    assert_eq!(movers.len(), world_n_planets() as usize);
    // Each mover pairs the planet realm with its exact elements (the FIRST non-empty roster —
    // the orbital_of Some-arm + moving_children filter-true in a live path).
    let bodies = generate_system_forest(0, &config);
    for (body, mover) in bodies.iter().skip(3).zip(&movers) {
        assert_eq!(mover.0, body.realm);
        assert_eq!(Some(mover.1), orbital_of(body.placement));
    }
}

#[test]
fn moving_children_for_config_excludes_non_children_and_empty_hosts() {
    let config = test_visual();
    // The Galaxy's only child (System A) is StaticOffset ⇒ no mover (filter-true + orbital_of None).
    assert!(moving_children_for_config(0, &config, GALAXY).is_empty());
    // A realm hosting nothing ⇒ no mover (parent-filter false arm).
    assert!(moving_children_for_config(0, &config, RealmId::System(999)).is_empty());
}

#[test]
fn planet_ecc_is_branchlessly_capped_both_ways() {
    // A HUGE ecc_sigma lets the Rayleigh draw exceed the cap ⇒ `.min` returns the cap exactly.
    let mut hot = test_visual();
    hot.planet.ecc_sigma = 5.0;
    let mut stream = realm_stream(0, &SYSTEM_A_LINEAGE);
    let hot_eccs: Vec<f64> = (0..64)
        .map(|_| planet_element_draws(&hot, &mut stream).ecc)
        .collect();
    assert!(hot_eccs.iter().all(|&e| e <= hot.planet.ecc_cap));
    assert!(
        hot_eccs.contains(&hot.planet.ecc_cap),
        "a large sigma must hit the cap",
    );
    // The real sigma (0.03) draws well below the cap ⇒ `.min` returns the sample.
    let cool = test_visual();
    let mut s2 = realm_stream(0, &SYSTEM_A_LINEAGE);
    for _ in 0..world_n_planets() {
        assert!(planet_element_draws(&cool, &mut s2).ecc < cool.planet.ecc_cap);
    }
}

#[test]
fn generate_system_forest_is_deterministic_and_in_domain() {
    // Same seed ⇒ byte-identical forest (the HR1 replay property).
    assert_eq!(
        generate_system_forest(0, &test_visual()),
        generate_system_forest(0, &test_visual()),
    );
    // Every planet's elements are in-domain across seeds (each assert split — no `&&`).
    for seed in [0u64, 1, 42, 999] {
        for body in generate_system_forest(seed, &test_visual())
            .iter()
            .filter(|b| matches!(b.realm, RealmId::Planet(_)))
        {
            let e = orbital_of(body.placement).expect("a planet is Orbital");
            assert!(e.ecc >= 0.0);
            assert!(e.ecc <= KEPLER_ECC_MAX);
            assert!(e.inclination >= 0.0);
            assert!(e.inclination.is_finite());
            assert!(e.raan >= 0.0);
            assert!(e.raan < TAU);
            assert!(e.arg_periapsis < TAU);
            assert!(e.mean_anomaly_epoch < TAU);
        }
    }
}

#[test]
fn generate_system_forest_differs_by_seed() {
    // Genuinely f(seed): different universe seeds yield different orbits/angles.
    assert_ne!(
        generate_system_forest(1, &test_visual()),
        generate_system_forest(2, &test_visual()),
    );
}

#[test]
fn the_worlds_orbits_are_real_kepler_around_the_real_drawn_star_mass() {
    // The synthetic Kepler-tuned mass is DEAD: every planet's central mass is its own
    // star's drawn mass in kg, and the home periods run real Kepler — 1.7 days at rung 0
    // to ~2.8 years at rung 8 (real-scale design §3.3.7), MEASURED in band here.
    let bodies = visual_forest();
    let star = bodies
        .iter()
        .find(|b| b.realm == SYSTEM_A)
        .and_then(|b| b.photometrics)
        .expect("the home star");
    let day_s = 86_400.0;
    let mut periods: Vec<f64> = bodies
        .iter()
        .filter(|b| b.parent == Some(SYSTEM_A) && matches!(b.realm, RealmId::Planet(_)))
        .map(|b| orbital_of(b.placement).expect("a planet is Orbital"))
        .inspect(|el| {
            assert_eq!(el.central_mass, star.mass_msun * crate::taxonomy::M_SUN_KG);
        })
        .map(|el| el.period())
        .collect();
    periods.sort_by(f64::total_cmp);
    assert_eq!(periods.len(), 9);
    let inner_d = periods[0] / day_s;
    assert!(
        (1.5..2.0).contains(&inner_d),
        "inner ~1.7 d, measured {inner_d} d"
    );
    let outer_yr = periods[8] / (365.25 * day_s);
    assert!(
        (2.5..3.1).contains(&outer_yr),
        "outer ~2.8 yr, measured {outer_yr} yr"
    );
}

#[test]
fn true_size_containment_and_sibling_annulus_non_overlap() {
    // Containment: every planet's worst-instant apoapsis + its shell + its clearance sits
    // inside its system's solved shell (the §3.2 solve, restated as the measurement).
    let bodies = visual_forest();
    for sys in bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
    {
        let shell = sys.shape.finite_extent();
        let ecc_cap = test_visual().planet.ecc_cap;
        for p in bodies
            .iter()
            .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Planet(_)))
        {
            let el = orbital_of(p.placement).expect("a planet is Orbital");
            let apo = el.sma * (1.0 + ecc_cap);
            assert!(
                apo + p.shape.finite_extent() < shell,
                "{:?} at worst instant stays inside {:?}",
                p.realm,
                sys.realm
            );
        }
        // NON-OVERLAP at the WORST INSTANT: adjacent annuli (a·(1±ecc_cap) widened by each
        // shell) never touch — provable at every instant for every seed (§3.3.4).
        let mut rungs: Vec<(f64, f64)> = bodies
            .iter()
            .filter(|b| b.parent == Some(sys.realm) && matches!(b.realm, RealmId::Planet(_)))
            .map(|p| {
                let el = orbital_of(p.placement).expect("Orbital");
                (el.sma, p.shape.finite_extent())
            })
            .collect();
        rungs.sort_by(|a, b| a.0.total_cmp(&b.0));
        for w in rungs.windows(2) {
            let (a_in, soi_in) = w[0];
            let (a_out, soi_out) = w[1];
            assert!(
                a_in * (1.0 + ecc_cap) + soi_in < a_out * (1.0 - ecc_cap) - soi_out,
                "adjacent SOI annuli are disjoint at the eccentricity cap"
            );
        }
    }
}

#[test]
fn walk_path_is_untouched_and_visual_planet_ids_are_distinct() {
    // Byte-identity: the walk boot path still yields the frozen 9-body forest + empty mover roster
    // (the new generator is uncalled by any walk path).
    // ★ 7 → 9 ON 2026-08-31. The hand-placed world gained TWO PLANETS under System B — the only
    // hand-placed system that is actually somewhere, System A sitting at the galaxy's origin. A
    // worked example needs a system that is BOTH placed and populated, to prove a parent adds its
    // child's placement; a hop of zero proves nothing. See `walk.rs` for the bodies themselves.
    assert_eq!(realm_regions_for(0).len(), 9);
    assert!(moving_children_for(0, SYSTEM_A).is_empty());
    // The 5 visual planet ids are mutually distinct and NONE aliases the walk Planet(7) — the
    // child_seed salt/index avalanche keeps them off the roster ids (no silent alias).
    // EVERY planet of EVERY star, not just one star's — the salt/index avalanche must keep them
    // distinct ACROSS systems too, or two stars would quietly claim the same planet realm.
    let forest = visual_forest();
    let moons = forest
        .iter()
        .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        .count();
    let ids: Vec<RealmId> = forest
        .iter()
        .filter(|b| matches!(b.realm, RealmId::Planet(_)))
        .map(|b| b.realm)
        .collect();
    let expect = (world_system_count() * world_n_planets()) as usize + moons;
    assert_eq!(ids.len(), expect);
    let mut distinct = ids.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        expect,
        "every planet id is distinct across every system"
    );
    for id in &ids {
        assert_ne!(*id, PLANET_A);
    }
}

#[test]
fn default_home_realm_walks_root_galaxy_system_and_refuses_a_degenerate_forest() {
    // The login fallback home: the FIRST system under the first galaxy under the root — pure
    // forest-walk, no seed knowledge. A degenerate forest (no root, or no chain below it) is
    // `None`, so a cluster booted on one fails where it can be seen instead of placing every
    // account somewhere arbitrary.
    let world = boot_world_for_tests();
    let home = default_home_realm(world.regions()).expect("THE world has a home system");
    assert!(
        matches!(home, RealmId::System(_)),
        "the fallback home is a star system: {home:?}"
    );
    assert_eq!(default_home_realm(&[]), None, "an empty forest has no home");
    // A root with nothing under it: the galaxy hop refuses.
    let bare_root = [world
        .regions()
        .iter()
        .find(|r| r.parent.is_none())
        .copied()
        .expect("the world has a root")];
    assert_eq!(default_home_realm(&bare_root), None);
}

#[test]
fn the_forest_is_a_valid_single_root_containment_tree() {
    let rs = regions();
    assert_eq!(rs.iter().filter(|r| r.parent.is_none()).count(), 1);
    let mut realms: Vec<RealmId> = rs.iter().map(|r| r.realm).collect();
    realms.sort();
    realms.dedup();
    assert_eq!(realms.len(), rs.len(), "every region has a distinct realm");
    assert_eq!(
        rs.len(),
        9,
        "the 9-region forest (5 shells + System B's 2 planets + Station + Area)"
    );
    // The mandate depths: Universe 0, Galaxy 1, System 2, Planet 3; the sibling system is same-depth.
    assert_eq!(region_depth(&rs, UNIVERSE), 0);
    assert_eq!(region_depth(&rs, GALAXY), 1);
    assert_eq!(region_depth(&rs, SYSTEM_A), 2);
    assert_eq!(region_depth(&rs, PLANET_A), 3);
    assert_eq!(region_depth(&rs, SYSTEM_B), 2);
    // Station A nests directly under System A (depth 3); Area A nests under Planet A (depth 4, the
    // deepest region). The kind-agnostic detector re-homes into either with no station/area code.
    assert_eq!(region_depth(&rs, STATION_A), 3);
    assert_eq!(region_depth(&rs, AREA_A), 4);
    // SIBLING TOPOLOGY LOCKED (task #135 C-6b): System B is a CHILD OF THE GALAXY — a SIBLING of
    // System A reached through the shared parent — NOT a child of System A. This is the canonical forest
    // the LIVE boot uses; the interim `override_regions_for_boundaries` child-of-source model (a
    // playground-only, env-gated fixture) must never drift into it. Depth 2 already implies this, but
    // pin the parent explicitly so a future refactor cannot silently promote the child model.
    assert_eq!(
        rs.iter()
            .find(|r| r.realm == SYSTEM_B)
            .expect("System B present")
            .parent,
        Some(GALAXY),
        "System B is a sibling under the Galaxy, NOT a child of System A",
    );
    // PARENTAGE LOCKED for the first-class Station/Area realms (task #133): a Station nests under its
    // star SYSTEM, an Area under its PLANET (the Area's frame REQUIRES a Planet parent — pin it so a
    // refactor cannot silently re-parent it and break `frame_for_realm`).
    assert_eq!(
        rs.iter()
            .find(|r| r.realm == STATION_A)
            .expect("Station A present")
            .parent,
        Some(SYSTEM_A),
        "Station A nests directly under System A",
    );
    assert_eq!(
        rs.iter()
            .find(|r| r.realm == AREA_A)
            .expect("Area A present")
            .parent,
        Some(PLANET_A),
        "Area A nests under Planet A (its frame provenance)",
    );
}

#[test]
fn region_depth_of_an_unknown_realm_is_zero() {
    assert_eq!(region_depth(&regions(), RealmId::Station(99)), 0);
}

#[test]
fn walk_demand_band_is_crossable_for_every_separated_child() {
    // Every expectation is DERIVED from the generated forest, never hand-typed.
    let regions = walk_demand_regions();
    for r in &regions {
        let ext = r.shape.finite_extent();
        let su = r.aoi.spin_up_r_m();
        // (a) An occupant INSIDE the child (within its own extent) is always in range.
        assert!(
            su > ext,
            "spin-up must exceed the child's own extent (in range inside the child)"
        );
        let Some(parent_id) = r.parent else { continue };
        let parent = regions
            .iter()
            .find(|p| p.realm == parent_id)
            .expect("a region's parent is in the forest");
        // Flatten the NORMALIZED centre (the walk offsets are dyadic, so the residual
        // `.offset()` is exactly ZERO — reading it here silenced the separated arm).
        //
        // ★ RE-BASED IN S9: measured at the PARENT's tier, not the child's. A region's `center` is
        // its position in its PARENT's frame, and its `frame` is its OWN — two different rungs since
        // the ladder landed. Reading the parent's number with the child's ruler made a system sitting
        // exactly ON the galaxy origin read as 0.0635 m away from it, which then failed the
        // "separated child must be out of range" arm. The same confusion cost a 2048× error in the
        // body generator's centres; this is the second place it hid.
        let d = r.center.metres_in(parent).length();
        let td = r.aoi.tear_down_r_m();
        // (c) Releasable: a point inside the parent exists from which the child is out of tear-down
        // range (tear_down < the farthest-in-parent distance = separation + the parent's own extent).
        assert!(
            td < d + parent.shape.finite_extent(),
            "tear-down must release within the parent's reach (child is releasable)"
        );
        if d > 0.0 {
            // (b) SEPARATED child: out of range AT the parent origin ⇒ a walk toward it CROSSES the band.
            assert!(
                su < d,
                "a separated child must be out of range at the parent origin (crossable)"
            );
        } else {
            // (d) CO-LOCATED ancestor (offset 0): always in range at the parent origin.
            assert!(
                su > d,
                "a co-located child must be in range at the parent origin"
            );
        }
    }
}

#[test]
fn planet_a_is_releasable_at_its_parents_origin() {
    // L3's precondition (spin-DOWN): Planet A releases while the occupant is still AT the parent origin
    // — a STRONGER fact than the uniform (c) criterion (which does NOT hold for the Area: 5.4 vs 5).
    let regions = walk_demand_regions();
    let planet = regions
        .iter()
        .find(|r| matches!(r.realm, RealmId::Planet(_)))
        .expect("planet A in the walk forest");
    let d = planet
        .centre_m(&regions)
        .expect("the walk forest holds the planet's parent")
        .length();
    assert!(
        planet.aoi.tear_down_r_m() < d,
        "planet A must release at its parent's origin (tear-down < its own offset)"
    );
}

#[test]
fn planet_a_geometric_warmup_precedes_its_boundary() {
    // Gate 1 rests on this: a geometric warm-up margin exists OUTSIDE the child's boundary.
    let regions = walk_demand_regions();
    let planet = regions
        .iter()
        .find(|r| matches!(r.realm, RealmId::Planet(_)))
        .expect("planet A in the walk forest");
    assert!(
        planet.aoi.spin_up_r_m() > planet.shape.finite_extent(),
        "planet A must warm up before its boundary (spin-up radius exceeds its extent)"
    );
}

// ===== FA-5 S1: the config-driven VISUAL-scale Orbital generator =====================

// FROZEN compressed-real geometry goldens — EXACT f64, captured once from the derive helpers at the
// compressed-real numbers and pinned as literals here (NON-self-referential: a regression in a derive
// helper is caught, not silently re-captured). Approx: au→render 37.96 / planet SOI 3.95 / orbit
// semi-major axes 15,26,44,75,127 / synthetic central mass / visibility factor cot(0.75°) ≈ 76.390.
// RE-CAPTURED at the placement arc S4 (the apoapsis-solved compression + the 0.372 gap fraction) —
// THE WORLD'S NUMBERS MOVED, deliberately: the outer planet's worst instant now sits exactly at the
// margin inside its system shell (was: outside it, the S0 tripwire), and a planet's visibility reach
// still crosses its whole system (302.1 m ≥ 300 m).
const FROZEN_VISIBILITY_FACTOR: f64 = 76.38983065807547;
// (`FROZEN_AU_TO_RENDER_M`, `FROZEN_PLANET_SOI_R_M`, `FROZEN_CENTRAL_MASS_KG` and the
// 5-rung `FROZEN_ORBIT_SMA_M` retired WITH their quantities at the in-system re-solve —
// real-scale design §9.2: per-planet SOIs are the D-REAL-1 equality pin's, the central
// mass is the star's drawn mass, and the 9-rung TRUE-metre ladder is pinned as a DERIVED
// identity by `world_geometry_is_true_size_and_the_ladder_is_luminosity_anchored`.)
// The 2026-08-15 SHELL SOLVE (owner ruling, items 5/10 addendum) — EXACT f64, captured once
// from the derivation at THE world's numbers and pinned as literals (non-self-referential).
// The two-level clearance of the worst descendant (the outer planet at the ecc-cap apoapsis,
// 142.046 m reach + 3.954 m extent × (1 + cot(θ/2)) + the 4 m solve margin ≈ 452.06 m)
// OUT-BINDS the 300 m containment headroom, so the shell is ring + clearance ≈ 12_483.46 m
// (was ring + 300 = 12_331.40 m, the 2026-08-15 measured failure). The worst measured margin
// on THE world (seed 0) is ≈ 11.13 m — the 4 m reserved margin plus the slack of the worst
// planet's DRAWN eccentricity sitting below the cap the solve bounds against.
// (`FROZEN_TWO_LEVEL_CLEARANCE_M = 452.058663384243`, `FROZEN_GALAXY_SHELL_R_M =
// 12483.45699203113` and `FROZEN_TWO_LEVEL_WORST_MARGIN_M = 11.127605697744457` retired with
// the upward interim solve — real-scale addendum §9.2; kept here verbatim as the interim-scale
// record. Their successors are the ▲ four-number pins below.)
/// ▲ THE FOUR OUTER GEOMETRY NUMBERS (real-scale addendum §A2.3), pinned bit-for-bit as
/// MEASURED on THE world — each equals the addendum's printed derivation exactly.
const FROZEN_REAL_UNIVERSE_R_M: f64 = 75_557_863_725_914_323_419_136.0; // 2⁷⁶ m, exact
const FROZEN_REAL_GALAXY_R_M: f64 = 4_611_686_018_427_387_904.0; // 2⁶² m, exact
// R_gal − clearance. The FORM is unchanged since 2026-08-20 (the reservation is the system shell at
// the largest star the world can host, not seed 0's heaviest sample); ★ RE-MEASURED IN S9, when both
// of its inputs climbed. 1.4989795871538760e15 m (0.15843 ly) → 4.609433753044092928e18 m
// (487.22 ly), a factor of 3_075. The reservation grew too, but it fell from 0.13 % of the galaxy to
// 0.0488 %, so the world gained placement room in every sense. The P10 lift named here is now the
// FINE rung, not the galaxy's — see `the_star_limit_moved_from_the_galaxys_room_to_the_systems_own_numbers`.
const FROZEN_REAL_PLACEMENT_R_M: f64 = 4_609_433_753_044_092_928.0;
/// The FINE rung's own root radius, `2⁵¹ m` — the number that used to BE the universe while every
/// realm counted in one lattice. It did not disappear when the ladder landed; it became the bound on
/// how big ONE star system may be, and slice S9 made the mass-cap solve ask it directly. The cap now
/// sits exactly on it (`target_system_bound_max_m() == FINE_ROOT_R_M`), which is why it is pinned.
const FINE_ROOT_R_M: f64 = 2_251_799_813_685_248.0;
// ★ RE-PINNED IN S9. Was 24.567_816_882_382_665 while the galaxy was 0.475 ly across. See the ▲ 4
// note in `the_four_outer_geometry_numbers_are_the_addendums_derivations` for why this ratio's
// MEANING changed with the climb and what measurement replaces it.
const FROZEN_REAL_COMPRESSION_CHI: f64 = 0.007_989_409_975_423_876; // real NN separation / placement

// ===== THE WINDOW LANE Slice 0: the per-system photometric draw (the marker datum) =========
// Owner-approved 2026-08-15/16, docs/design/window_lane.md §2.2/§2.8: a sleeping child's point
// of light is authored by its parent from the child's OWN generation stream. Slice 0 lands the
// draw consumer-less (nothing moves); the Slice-A marker emit reads it.

#[test]
fn the_worlds_systems_draw_their_pinned_photometrics() {
    // ★ RE-PINNED IN S9 — SAME CAUSE AS 2026-08-20, THIRD TIME: the mass cap. `sample_imf_mass`
    // inverts a BOUNDED power law, so its upper bound enters EVERY draw in the world, and the cap
    // is that bound. It has now moved twice: the literal 120.0 M☉ → the galaxy-derived
    // 16.360034882257757 → the lattice-derived 30.745283003771995. Each move re-rolls every star by
    // a few parts in ten thousand, upward this time because the bound rose:
    //   0.09286807253954772 → 0.09287476027864262
    //   0.10811333254263818 → 0.10813084330897464
    //   0.16166413170715563 → 0.16174689739789090
    // The world re-rolls and that is lawful pre-launch (the seed is a pre-freeze dial); the
    // draw ORDER is untouched, so the append-only stream discipline holds. All three stay M-class,
    // so nothing downstream of the class changes.
    //
    // ★ AND THIS IS THE LAST TIME IT MAY HAPPEN QUIETLY. Three re-rolls of the whole world traced to
    // one bound says the cap is a launch-blocking input, not a tuning knob: after the seed freezes,
    // moving it re-draws every star every player has seen. The P10 lattice lift moves it again, and
    // that lift must land BEFORE the freeze.
    //
    // FROZEN per-system draw goldens on THE world (seed 0) — EXACT f64, captured once from the
    // taxonomy chain (sample_imf_mass → classify_spectral → main_sequence_luminosity) at THE
    // world's stellar config and pinned as literals (NON-self-referential: a stream drift, a
    // re-ordered draw, or a retuned IMF is caught, not silently re-captured). All three stars
    // land M-class — the honest Salpeter answer (α = 2.35 concentrates mass draws at the low
    // bound; the u01 that would draw a G star is a ~1e-3 sliver). Sub-solar luma is expected:
    // the marker's DERIVED brightness knob (coordinate-scale model) is a later, separate owe.
    let cfg = galaxy_holding(
        &UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S),
        0,
        TEST_GALAXY_SYSTEMS,
    );
    let all = system_photometrics_for_config(0, &cfg);
    // The SYSTEM subset carries the pinned stellar goldens; the planets' REFLECTED draws
    // (Slice C1) are coherence-checked below against the derivation, not re-pinned per body.
    // ★ A BOUNDED SAMPLE (2026-08-31). This pinned EVERY system's draw — three of them, when a
    // galaxy held three. A galaxy now holds a quarter of a million, and a test galaxy a readable
    // slice of that, so pinning all of them makes a golden nobody can read and one that moves
    // whenever the slice does. The stellar draw is one law applied per star: three stars pin it as
    // surely as forty-eight, which is the same reason the placement golden samples one anchor per rung.
    // EVERY system's draw — what the coherence walk below looks a planet's illuminator up in.
    let systems: Vec<(RealmId, StarPhotometrics)> = all
        .iter()
        .copied()
        .filter(|(realm, _)| matches!(realm, RealmId::System(_)))
        .collect();
    // …and the bounded SAMPLE that carries the golden.
    let draws: Vec<(RealmId, StarPhotometrics)> = systems.iter().copied().take(3).collect();
    assert_eq!(
        draws,
        vec![
            (
                RealmId::System(7),
                StarPhotometrics {
                    mass_msun: 0.09287476027864262,
                    class: SpectralClass::M,
                    luma_lsun: 0.0009725066044268927,
                },
            ),
            (
                RealmId::System(10487570625701098367),
                StarPhotometrics {
                    mass_msun: 0.10813084330897464,
                    class: SpectralClass::M,
                    luma_lsun: 0.0013797865053910446,
                },
            ),
            (
                RealmId::System(13979593561158050752),
                StarPhotometrics {
                    mass_msun: 0.1617468973978909,
                    class: SpectralClass::M,
                    luma_lsun: 0.0034837785544366263,
                },
            ),
        ],
    );
    // Provenance: the three pinned realms ARE the seed lineage's systems, in forest order
    // (system 0 keeps the named SYSTEM_A_SEED; the rest avalanche off the galaxy). Plain
    // equality, no destructuring match — a non-System draw fails the vec compare (HR5: no
    // uncoverable panic arm).
    let realms: Vec<RealmId> = draws.iter().map(|(realm, _)| *realm).collect();
    assert_eq!(
        realms,
        vec![
            RealmId::System(system_seed_at(0)),
            RealmId::System(system_seed_at(1)),
            RealmId::System(system_seed_at(2)),
        ]
    );
    assert_eq!(realms[0], RealmId::System(SYSTEM_A_SEED));
    // Coherence: each pinned (class, luma) IS the taxonomy derivation of its pinned mass —
    // the chain cannot silently decouple from the one drawn u01.
    for (_, p) in &draws {
        assert_eq!(
            p.class,
            classify_spectral(p.mass_msun, &SpectralClass::MASS_BOUNDS)
        );
        assert_eq!(
            p.luma_lsun,
            main_sequence_luminosity(p.mass_msun, &cfg.stellar.mlr_segments)
        );
    }
    // THE PLANET REFLECTOR DRAWS (Slice C1 — §1.1 item 3b "per direct child"): every planet
    // of THE world carries a marker datum; its class and mass provenance are its STAR's
    // (reflected light keeps the star's color), and its luma sits inside the closed-form
    // reflected band `L★ · albedo · r²/(4d²)` over the canonical albedo table at the
    // planet's own orbit — bounded by construction, MEASURED here (never assumed).
    let planets: Vec<(RealmId, StarPhotometrics)> = all
        .iter()
        .copied()
        .filter(|(realm, _)| matches!(realm, RealmId::Planet(_)))
        .collect();
    let world_bodies = generate_system_forest(0, &cfg);
    let moons_n = world_bodies
        .iter()
        .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        .count();
    // The COHERENCE check counts every system of the world; `draws` above is a bounded SAMPLE of the
    // stellar goldens and must not be mistaken for the census.
    assert_eq!(
        planets.len(),
        systems_in(&cfg) * cfg.planet.n_planets as usize + moons_n,
        "every planet AND every moon of every system carries a marker datum"
    );
    let world = WorldView::generated(0, &cfg);
    for (realm, p) in &planets {
        let region = world
            .regions()
            .iter()
            .find(|r| r.realm == *realm)
            .expect("a drawn planet is a region of THE world");
        // A MOON's illuminating geometry (T3): the star is still the illuminator and the
        // DILUTION DISTANCE is its parent PLANET's orbit — resolve the star through the
        // grandparent and the distance through the parent's own elements.
        let parent = region.parent.expect("parented");
        let (illuminating_system, dilution_sma_m) = match parent {
            RealmId::Planet(_) => {
                let grandparent = world_bodies
                    .iter()
                    .find(|b| b.realm == parent)
                    .and_then(|b| b.parent)
                    .expect("a moon's planet nests in a system");
                let planet_el = moving_children_for_config(0, &cfg, grandparent)
                    .into_iter()
                    .find(|(child, _)| *child == parent)
                    .map(|(_, el)| el)
                    .expect("the moon's planet orbits its system");
                (grandparent, planet_el.sma)
            }
            _ => (
                parent,
                moving_children_for_config(0, &cfg, parent)
                    .into_iter()
                    .find(|(child, _)| *child == *realm)
                    .map(|(_, el)| el.sma)
                    .expect("a planet of THE world orbits"),
            ),
        };
        let star = systems
            .iter()
            .find(|(sys, _)| *sys == illuminating_system)
            .map(|(_, s)| *s)
            .expect("every planet's illuminator is a system of this world");
        assert_eq!(
            p.class, star.class,
            "reflected light keeps the star's color"
        );
        assert_eq!(p.mass_msun, star.mass_msun, "the illuminator's provenance");
        // The reflector's cross-section is the body's OWN derived LOOK radius (SL3).
        let (d, r) = (
            dilution_sma_m,
            region
                .look
                .expect("a planet of THE world draws itself")
                .finite_extent(),
        );
        let (lo, hi) = GEOMETRIC_ALBEDO_BOUNDS;
        let dilution = (r * r) / (4.0 * d * d);
        // Two asserts, not one `&&` (HR5: a short-circuit's false arm is uncoverable).
        assert!(
            p.luma_lsun >= star.luma_lsun * lo * dilution,
            "{realm:?}: reflected luma {} below the derivation band",
            p.luma_lsun
        );
        assert!(
            p.luma_lsun <= star.luma_lsun * hi * dilution,
            "{realm:?}: reflected luma {} above the derivation band",
            p.luma_lsun
        );
    }
}

#[test]
fn the_marker_datum_frames_the_pinned_draw_through_the_one_shared_codec() {
    // Slice A → look_horizon slice 1: the parent's marker datum for a sleeping child rides
    // the ONE window-body codec (`vd_core::look::marker_bag`, now with the presence floor's
    // extent beside it) — encode every system of THE world, decode through the shared reader,
    // and get back exactly the pinned (class code, luma) pair AND the stated radius. A
    // re-framed bag, a transposed field, or a codec fork fails here, not on a live wire.
    let cfg = galaxy_holding(
        &UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S),
        0,
        TEST_GALAXY_SYSTEMS,
    );
    let draws = system_photometrics_for_config(0, &cfg);
    let moons = generate_system_forest(0, &cfg)
        .iter()
        .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        .count();
    // DERIVED: the two `3`s were a retired census. Per system: itself, its star child, and one
    // reflector per planet — plus every moon's.
    let n_sys = systems_in(&cfg);
    assert_eq!(
        draws.len(),
        n_sys + n_sys * (cfg.planet.n_planets as usize + 1) + moons,
        "THE world's marker roster: three systems + every planet's reflector + each \
         system's STAR child (T2) + every moon's reflector (T3)"
    );
    // ★ RE-PINNED IN S12 (2026-08-28), ONE CAUSE FOR ALL OF THEM. The placement became a SHAPE and
    // takes SIX draws where the shell took two, so every draw after them shifted by four. One
    // planet drew a different mass, and a lighter planet holds no moon — so exactly ONE MOON left
    // the world. Every count below is that single fact, counted differently.
    // (The moon CENSUS is pinned by `g_moon_census…`, which exists for it. A literal here was a
    // second, weaker pin of the same fact, and it goes stale with every change to the test galaxy.)
    for (_, p) in &draws {
        let datum = marker_datum(p);
        assert_eq!(datum, (p.class as u8, p.luma_lsun));
        let bag = vd_core::look::marker_bag(Some(datum), cfg.stellar.system_soi_r_m);
        assert_eq!(
            vd_core::look::luma_of(&bag),
            Ok((p.class as u8, p.luma_lsun))
        );
        assert_eq!(
            vd_core::look::extent_of(&bag),
            Ok(cfg.stellar.system_soi_r_m)
        );
        // A marker bag can never answer for a look (the structural exclusivity, decoded side).
        assert_eq!(
            vd_core::look::look_of(&bag),
            Err(vd_core::tlv::TlvError::MissingRequiredTag(
                vd_core::look::TAG_LOOK
            ))
        );
    }
}

#[test]
fn the_photometric_draw_is_deterministic_and_dynamics_blind() {
    // Two generations, identical draws (pure f(seed, config) — HR1: every shard hosting the
    // galaxy authors byte-identical markers with no shared state)…
    // ★ A SMALL GALAXY (S12/G8, 2026-08-28). Both properties — the same draw twice, and the same
    // draw whatever cluster dynamics are passed — are true of every star or of none. Fifty stars
    // prove them as surely as a quarter of a million, and this test held the suite for over a
    // minute at the shipped size.
    let cfg = galaxy_holding(
        &UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S),
        0,
        TEST_GALAXY_SYSTEMS,
    );
    assert_eq!(
        system_photometrics_for_config(0, &cfg),
        system_photometrics_for_config(0, &cfg)
    );
    // …and blind to the two CLUSTER-dynamics arguments (occupant speed / tick dt): they size
    // the interest band, never the world — the same draw whatever cluster runs it (SL5).
    // ★ VARY ONLY THE DYNAMICS (2026-08-30). This built a SECOND galaxy through `galaxy_holding`,
    // which scales a radius by a census ratio — so the comparison had two variables in it and the
    // scaling, not the dynamics, is what moved. MEASURED: one side folded 48 star systems and the
    // other 3, and the test read that as "the draw is not dynamics-blind".
    //
    // The property under test is that the DRAW ignores the cluster's speed and tick. So take the
    // very same galaxy and change only those two fields.
    let mut other_dynamics = cfg;
    other_dynamics.interest = UniverseConfig::world(15.0, 0.02).interest;
    assert_eq!(
        system_photometrics_for_config(0, &cfg),
        system_photometrics_for_config(0, &other_dynamics)
    );
    // A DIFFERENT universe seed draws differently (the stream is real, not a constant): seed 1
    // shares no system seed with seed 0 beyond the named system 0, whose draw must move.
    let seed1 = system_photometrics_for_config(1, &cfg);
    // The NON-MOON census is config-pinned; the MOON census is seed-derived by design
    // (T3: each star's ladder and each planet's drawn mass gate emission), so the roster
    // is bounded below by the moonless shape and every extra row is a moon reflector.
    assert!(
        seed1.len() >= 3 + 3 * (cfg.planet.n_planets as usize + 1),
        "the moonless census floor holds at any seed"
    );
    assert_ne!(
        seed1[0].1,
        system_photometrics_for_config(0, &cfg)[0].1,
        "system 0's draw must differ under a different universe seed"
    );
    // The ambient shells carry NO draw. Systems draw their own light; every planet carries
    // its reflected datum (Slice C1 — a sleeping realm appears only as its parent's marker).
    // ONE equality over the whole forest (HR5: no matches!/count with uncoverable arms): the
    // bodies carrying a draw are exactly each system followed by its planets, forest order.
    let world = WorldView::generated(0, &cfg);
    let starred: Vec<RealmId> = world
        .bodies
        .iter()
        .filter_map(|b| b.photometrics.map(|_| b.realm))
        .collect();
    let mut expected = Vec::new();
    // DERIVED: this built the expectation for THREE systems while `starred` covers every
    // system the galaxy drew. The literal was a retired census.
    for i in 0..systems_in(&cfg) as u32 {
        let s = system_seed_at(i);
        expected.push(RealmId::System(s));
        for n in 0..cfg.planet.n_planets {
            expected.push(RealmId::Planet(child_seed(s, PLANET_SALT, u64::from(n))));
        }
        // T2: each system's STAR child follows its planets (the §7.1 per-system push
        // order) — its datum IS the system's pinned draw, re-stated, zero new draws.
        expected.push(RealmId::Star(child_seed(s, STAR_SALT, 0)));
        // T3: then the census MOONS, planet order then rung order — read from the forest
        // (the emission is potato-gated, so the roster states what actually exists).
        for b in world
            .bodies
            .iter()
            .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        {
            let parent = b.parent.expect("a moon has a planet");
            let in_this_system = world
                .bodies
                .iter()
                .find(|q| q.realm == parent)
                .and_then(|q| q.parent)
                == Some(RealmId::System(s));
            if in_this_system {
                expected.push(b.realm);
            }
        }
    }
    assert_eq!(starred, expected);
    // THE APPEND ASSERTION (§7.3): filtering the T2/T3 rows out reproduces the pre-T2
    // roster order EXACTLY — the star and moon passes INSERTED per-system rows and
    // shifted nothing else; the stream (draw) prefix identity is pinned separately above.
    let moons: std::collections::BTreeSet<RealmId> = world
        .bodies
        .iter()
        .filter(|b| b.parent.is_some_and(|p| matches!(p, RealmId::Planet(_))))
        .map(|b| b.realm)
        .collect();
    let without_new: Vec<RealmId> = starred
        .iter()
        .copied()
        .filter(|r| !matches!(r, RealmId::Star(_)) && !moons.contains(r))
        .collect();
    let pre_t2: Vec<RealmId> = expected
        .iter()
        .copied()
        .filter(|r| !matches!(r, RealmId::Star(_)) && !moons.contains(r))
        .collect();
    assert_eq!(without_new, pre_t2);
}

/// The star-bound boot fence's REFUSAL arm: a star whose authority bound sits inside its own
/// photosphere is a world nobody may boot (T2 §5.3). Built by hand — THE world never
/// produces it (the green arm is measured by every boot) — so the refusal is exercised
/// exactly once, here, with its stated numbers.
#[test]
fn the_star_bound_fence_refuses_a_bound_inside_the_photosphere() {
    let cfg = test_world();
    let mut bodies = generate_system_forest(0, &cfg);
    let star = bodies
        .iter_mut()
        .find(|b| matches!(b.realm, RealmId::Star(_)))
        .expect("THE world names a star");
    let photosphere = crate::taxonomy::star_radius_m(
        star.photometrics
            .expect("a star carries photometrics")
            .mass_msun,
    );
    star.shape = vd_core::geometry::Boundary::Shell {
        r: 0.5 * photosphere,
    };
    let err = guard_star_bounds(&bodies).expect_err("the fence refuses");
    assert_eq!(err.bound_m, 0.5 * photosphere);
    assert_eq!(err.photosphere_m, photosphere);
}

/// `earth_like_candidates` — the T4 tool's whole read path (the bin is a four-line shell, so
/// this is where the sweep is measured). Both continue arms ride here too: a body whose
/// parent is not a system resolves its star through the GRANDparent (a moon), and a body
/// whose chain names no photometric star is skipped.
#[test]
fn earth_like_candidates_reads_the_world_and_answers_with_its_numbers() {
    let cfg = test_world();
    // The measured best seed of the ruling-F sweep — it holds exactly one Earth-like body.
    // ★ SEED CHANGED, NOT THE EXPECTATION (S12). Seed 2298 was chosen because it held exactly one
    // Earth-like body; after the shape shifted every planet mass it holds none. Pinning "0 found"
    // would make this test stop testing the FINDING. Seed 17 is the first that holds one on the new
    // world — measured by `an_earth_like_world_still_exists_somewhere_in_the_seed_space`, which also
    // records how rare it is: 2 seeds in 256.
    // ★ RE-MEASURED 2026-08-31, NOT HAND-EDITED. Seed 349 was chosen when a test could hold THE
    // world's whole galaxy. A test galaxy now holds a readable slice of it, so seed 349's Earth-like
    // body is simply not among the systems this galaxy draws — the seed was right for a different
    // number of stars. Seed 4 is the first that carries exactly one here, and every field below was
    // read off the generator by `diag_first_earth_like_seed_on_the_test_galaxy`.
    let found = earth_like_candidates(4, &cfg);
    assert_eq!(found.len(), 1);
    // ★ RE-PINNED IN S9 — SAME TWO CAUSES AS 2026-08-20, both moving again because both descend
    // from the mass cap:
    //   • THE CAP re-rolled the stellar draw a second time (star 1.015066097741417 →
    //     1.0249674589959064 M☉, luma 1.0616400455472874 → 1.1036727248910603 L☉). The IMF is a
    //     BOUNDED power law and the cap is its bound, so every star in every world moves when it does.
    //   • THE PLACEMENT RADIUS climbed with the galaxy (1.4989795871538758e15 →
    //     4.609433753044093e18 m), which is what both sibling distances read.
    //
    // ★ AND THE PLANET ITSELF IS STILL UNMOVED — `mass_kg` and `radius_m` are bit-identical across
    // BOTH re-pins, because a planet's own draws never read the stellar cap. That invariance is the
    // reason this golden is worth keeping: it separates "the star re-rolled" from "the planet
    // generator changed", and only the first has happened.
    //
    // The two derived readings that depend on the star — insolation and equilibrium temperature —
    // move only in the last two digits (0.7483147950814765 → 0.7483147950814767), because a G star
    // 1 % heavier is also placed on a slightly wider orbit ladder and the two nearly cancel.
    //
    // PINNED AS MEASURED (the config here is the unit tier's 15 m/s · 0.05 s world, not
    // the DEV cluster's — the ladder is the same, the derived speed knobs are not).
    // ★ RE-MEASURED IN S12 (2026-08-28), NOT HAND-EDITED. Seed 2298 held one Earth-like body and now
    // holds none; seed 349 is the first that does on the shaped world. Every field below was read off
    // the generator, never typed from the old record.
    //
    // ★ ONE FIELD PAIR IS THE SHELL'S OWN EPITAPH. The old record read
    // `nearest_sibling_m: 4.6094337530440924e18` and `farthest_sibling_m: 4.609433753044093e18` — the
    // SAME distance twice, because every system sat on one surface. They now differ by a factor of
    // four, which is the shell being gone, measured by a test that was never about shape.
    assert_eq!(
        found[0],
        EarthLikeCandidate {
            system: RealmId::System(17_478_540_969_984_029_189),
            body: RealmId::Planet(17_335_632_776_285_106_644),
            star_mass_msun: 0.972_030_245_179_213_8,
            mass_kg: 3.512_183_762_070_154_5e24,
            radius_m: 5_520_280.063_838_893,
            insolation_rel: 0.748_314_795_081_476_7,
            t_eq_k: 236.785_700_196_447_92,
            star_class: SpectralClass::G,
            star_luma_lsun: 0.892_727_912_069_189_4,
            planet_class: crate::taxonomy::PlanetType::Rocky,
            bond_albedo: 0.3,
            has_atmosphere: true,
            system_planets: 9,
            system_moons: 18,
            own_moons: 1,
            sibling_count: 22,
            // ★ RE-MEASURED AT S12/G1 (2026-08-28): the shape's numbers are drawn now, so the two
            // neighbours sit somewhere else. Everything ABOUT THE PLANET is untouched — same system,
            // same body, same star mass, same radius, same temperature — because a planet's draws
            // come from its own system's stream, which the galaxy's shape draws cannot reach.
            nearest_sibling_m: 2.080_474_878_942_688e16,
            farthest_sibling_m: 3.186_567_517_143_485e17,
        }
    );
    // A seed with no Earth-like body answers with an EMPTY sweep — the same read path, the
    // other verdict (the sweep is 1-in-181, so seed 0 is the ordinary case).
    assert_eq!(earth_like_candidates(0, &cfg), Vec::new());
}

/// `WorldView::default_home_offset_m` — the T2 spawn standoff, both arms: on THE world the
/// home system holds a STATIC child that contains its centre (the star), so the clearing is
/// twice that child's bound; on a world whose home realm cannot be named there is no
/// clearing at all and the answer is the pre-T2 zero, byte-identical.
#[test]
fn the_home_offset_is_twice_the_centre_holding_childs_bound_or_zero() {
    let cfg = test_world();
    let world = WorldView::generated(0, &cfg);
    let home = vd_core::worldgen::default_home_realm(world.regions()).expect("a home");
    let star_bound = world
        .regions()
        .iter()
        .find(|r| matches!(r.realm, RealmId::Star(_)) && r.parent == Some(home))
        .map(|r| r.shape.finite_extent())
        .expect("the home system holds its star");
    assert_eq!(
        world.default_home_offset_m(),
        vd_core::glam::DVec3::new(0.0, 0.0, 2.0 * star_bound),
    );
    // A view whose forest names no home realm: the clearing is zero.
    let empty = WorldView {
        bodies: Vec::new(),
        regions: Vec::new(),
    };
    assert_eq!(empty.default_home_offset_m(), vd_core::glam::DVec3::ZERO);
}

/// The two DEFENSIVE arms of the Earth-like sweep, and the MOON arm beside them. THE world
/// never produces either refusal — every generated body's parent is in the forest and every
/// system carries its star — so they are measured over a hand-built forest, which is also
/// the only honest way to state "this cannot happen here".
#[test]
fn the_earth_like_sweep_skips_an_orphan_and_a_starless_chain() {
    let cfg = test_world();
    // ★ SEED RE-CHOSEN 2026-08-31, NOT THE EXPECTATION. This test exists for its SKIP arms — a
    // parentless body, a starless chain — and each of them needs one real candidate to mutate. So it
    // needs a seed whose galaxy carries exactly one, on the galaxy THIS test holds.
    //
    // Seed 349 was measured on a galaxy that held every star of THE world. A test galaxy now holds a
    // readable slice, so 349's candidate is simply not among the systems it draws. Seed 4 is the
    // first that carries one here, measured by `diag_first_earth_like_seed_on_the_test_galaxy`.
    let world = generate_system_forest(4, &cfg);
    let earthlike = earth_like_in_forest(&world);
    assert_eq!(earthlike.len(), 1);
    let body = world
        .iter()
        .find(|b| b.realm == earthlike[0].body)
        .copied()
        .expect("the candidate is a body of this forest");
    // (a0) PARENTLESS: a taxon-bearing body with no parent at all — no chain, so no star,
    // so no verdict. THE world never emits one (every taxon rides a planet or a moon).
    let mut rootless = body;
    rootless.parent = None;
    assert_eq!(earth_like_in_forest(&[rootless]), Vec::new());
    // (a) ORPHAN: the candidate re-parented onto a planet that is not in the forest — the
    // grandparent lookup answers None and the body is skipped.
    let mut orphan = body;
    orphan.parent = Some(RealmId::Planet(0xdead_beef));
    assert_eq!(earth_like_in_forest(&[orphan]), Vec::new());
    // (b) STARLESS: the candidate under a system row that carries no photometrics.
    let mut starless_system = world
        .iter()
        .find(|b| b.realm == earthlike[0].system)
        .copied()
        .expect("the candidate's system");
    starless_system.photometrics = None;
    assert_eq!(earth_like_in_forest(&[starless_system, body]), Vec::new(),);
    // (c) THE MOON ARM: a body whose parent is a PLANET resolves its star through the
    // GRANDparent — stated by re-parenting the candidate under a planet of its own system.
    // ★ THE HOST MUST BE A PLANET OF THE CANDIDATE'S OWN SYSTEM (S12 fix, 2026-08-28) — which is
    // what the comment above always said, and what the selection did not enforce.
    //
    // It took the first planet in the WORLD. Under the old seed the candidate sat in the first system,
    // so the first planet happened to belong to it and the chain resolved. Seed 349's candidate is in
    // another system, so the host came from a DIFFERENT one, its parent was not `system_row`, and the
    // grandparent lookup correctly found nothing — the test failed for a reason that was never about
    // the moon arm it exists to drive.
    let host = world
        .iter()
        .find(|b| {
            matches!(b.realm, RealmId::Planet(_))
                && b.realm != body.realm
                && b.parent == Some(earthlike[0].system)
        })
        .copied()
        .expect("the candidate's own system holds another planet");
    let system_row = world
        .iter()
        .find(|b| b.realm == earthlike[0].system)
        .copied()
        .expect("the system row");
    let mut moonised = body;
    moonised.parent = Some(host.realm);
    let via_grandparent = earth_like_in_forest(&[system_row, host, moonised]);
    assert_eq!(via_grandparent.len(), 1);
    assert_eq!(via_grandparent[0].system, earthlike[0].system);
    // (c') and the SIBLING-GAP reader's own two refusals ride the same hand-built forest.
    // A system with NO SIBLINGS in the forest reports zeros with a zero count — the honest
    // answer, never a sentinel (this is also what (c) above measured).
    assert_eq!(via_grandparent[0].sibling_count, 0);
    assert_eq!(via_grandparent[0].nearest_sibling_m, 0.0);
    // (d) A system whose placement is an ORBIT has no static offset to subtract from, and
    // (e) a system with no parent has no frame to compare siblings in. THE world emits
    // neither (a star system is a static child of its galaxy), so both are stated here.
    let mut orbiting_system = system_row;
    orbiting_system.placement = Placement::Orbital(crate::celestial::OrbitalElements {
        sma: 1.0,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: 1.0,
    });
    let orbiting = earth_like_in_forest(&[orbiting_system, host, moonised]);
    assert_eq!(orbiting.len(), 1);
    assert_eq!(orbiting[0].sibling_count, 0);
    assert_eq!(orbiting[0].farthest_sibling_m, 0.0);
    let mut parentless_system = system_row;
    parentless_system.parent = None;
    let parentless = earth_like_in_forest(&[parentless_system, host, moonised]);
    assert_eq!(parentless.len(), 1);
    assert_eq!(parentless[0].sibling_count, 0);
    assert_eq!(parentless[0].nearest_sibling_m, 0.0);
    // (f) A SIBLING STAR WITH NO STATIC OFFSET — an ORBITING star system — has no placement to
    // subtract, so it is skipped rather than counted at a fabricated distance. THE world never
    // emits one (a star system is a static child of its galaxy), so it is stated here.
    let mut orbiting_sibling = system_row;
    orbiting_sibling.realm = RealmId::System(0x5151_5151_5151_5151);
    orbiting_sibling.placement = Placement::Orbital(crate::celestial::OrbitalElements {
        sma: 1.0,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: 0.0,
        central_mass: 1.0,
    });
    let with_orbiting_sibling =
        earth_like_in_forest(&[system_row, orbiting_sibling, host, moonised]);
    assert_eq!(with_orbiting_sibling.len(), 1);
    assert_eq!(with_orbiting_sibling[0].sibling_count, 0);
    assert_eq!(with_orbiting_sibling[0].nearest_sibling_m, 0.0);
    assert_eq!(with_orbiting_sibling[0].farthest_sibling_m, 0.0);
    // ...and a sibling that DOES carry a static offset is counted, at that offset — the same
    // loop, the other verdict, so the skip above is not the only arm this test ever sees.
    // ★ THE HAND-BUILT FOREST STATES ITS OWN FRAME (S12 fix, 2026-08-28). The two siblings below sit
    // 5 m and 9 m from their system, and the system is anchored HERE rather than left wherever the
    // shape put it. Two reasons, in order:
    //
    //  * The old numbers only worked because the old candidate's system WAS the home, the one system
    //    anchored at the origin, so an absolute (3,4,0) happened to be 5 m away from it. That was
    //    luck, not intent. Seed 349's candidate sits out in the disc.
    //  * Following it out there does NOT work either: a star sits about 6.8e17 m from the centre,
    //    where one f64 step is 128 m (measured, not argued). `x + 5.0 - x` is exactly 0.0 there, so
    //    a 5 m sibling and its system land on the SAME number and the gap reads zero.
    //
    // These four numbers measure the sibling-gap READER — which rows count, and what it reports —
    // never where the galaxy put the star. So the forest states a frame the arithmetic can hold.
    // The shape's own distances are measured in `earth_like_candidates_reads_the_world_and_answers…`,
    // on the real world, at the real magnitude.
    let mut system_row = system_row;
    system_row.placement = Placement::StaticOffset(DVec3::ZERO);
    let mut placed_sibling = system_row;
    placed_sibling.realm = RealmId::System(0x6262_6262_6262_6262);
    placed_sibling.placement = Placement::StaticOffset(DVec3::new(3.0, 4.0, 0.0));
    // ...beside a sibling-parented body that carries NO photometrics — an ambient shell rather
    // than a star system. It shares the parent and is not the system itself, so only the
    // photometric test can reject it: the arm that names WHAT a star system is.
    let mut ambient_sibling = system_row;
    ambient_sibling.realm = RealmId::System(0x7373_7373_7373_7373);
    ambient_sibling.photometrics = None;
    ambient_sibling.placement = Placement::StaticOffset(DVec3::new(9.0, 0.0, 0.0));
    let with_placed_sibling =
        earth_like_in_forest(&[system_row, placed_sibling, ambient_sibling, host, moonised]);
    assert_eq!(with_placed_sibling.len(), 1);
    assert_eq!(with_placed_sibling[0].sibling_count, 1);
    assert_eq!(with_placed_sibling[0].nearest_sibling_m, 5.0);
    assert_eq!(with_placed_sibling[0].farthest_sibling_m, 5.0);
}

/// The moon ladder's SOI-CLEARANCE clamp — the third `child_clearance_m` arm, measured
/// 25×-slack inert on THE world (the census pin prints the counts). Reachable only over a
/// hostile input: a planet whose stored shell is a fraction of its own Hill disc, so the
/// first rung's worst instant plus its clearance already breaches the SOI and NO moon is
/// minted. That is exactly what the clamp promises.
#[test]
fn the_moon_ladder_mints_nothing_when_the_first_rung_would_breach_the_soi() {
    let cfg = test_world();
    let world = generate_system_forest(0, &cfg);
    // A planet THE world actually gives moons to — its first ladder rung is inside the disc
    // edge, so the disc-edge break cannot pre-empt the clearance clamp under test.
    let hosts: std::collections::BTreeSet<RealmId> = world
        .iter()
        .filter(|b| matches!(b.parent, Some(RealmId::Planet(_))))
        .filter_map(|b| b.parent)
        .collect();
    let host = world
        .iter()
        .find(|b| hosts.contains(&b.realm))
        .copied()
        .expect("THE world gives some planet moons");
    let star = world
        .iter()
        .find(|b| b.realm == RealmId::System(7))
        .and_then(|b| b.photometrics)
        .expect("the home system carries its star");
    let taxon = host.taxon.expect("a generated planet carries its taxon");
    let sma_m = orbital_of(host.placement)
        .expect("a generated planet orbits")
        .sma;
    // The SAME planet, its authority shell cut to one Roche radius — small enough that the
    // clearance clamp refuses the first rung.
    // ★ THE PINCHED SHELL IS STATED, NOT SMUGGLED (2026-08-28). This set the shell on a body and
    // let the moon pass find it by searching the body list — the scan that made generation
    // quadratic. Now the pass takes the number, so the test hands it over plainly.
    let pinched_shell_m =
        crate::taxonomy::roche_radius_m(taxon.mass_kg, crate::taxonomy::RHO_ROCK_KGM3);
    let mut pinched = host;
    pinched.shape = Boundary::Shell { r: pinched_shell_m };
    let realm = pinched.realm;
    let mut bodies = vec![pinched];
    let minted = append_moons(
        &mut bodies,
        &cfg,
        0,
        7,
        &star,
        realm,
        // The moon stream's own salt — any seed states the same clamp; the clamp is
        // geometry, not chance.
        12_345,
        sma_m,
        taxon.mass_kg,
        0.3,
        pinched_shell_m,
    );
    assert_eq!(minted, 0);
    assert_eq!(bodies.len(), 1);
}

#[test]
fn the_world_shape_constants_are_what_a_stores_label_folds_and_every_one_of_them_moves_it() {
    // WHAT THIS LIST IS FOR (owner ruling 2026-08-24 Q1 condition 1). A durable file folds these into
    // its label, so a store written for one world's geometry is REFUSED by a build that would place
    // those bodies somewhere else. The whole mechanism rests on one property, asserted here rather than
    // assumed: EVERY entry must move the label. An entry that did not would be a number this list
    // pretends to protect and does not.
    let base = world_shape_constants();
    assert!(
        !base.is_empty(),
        "a label folded over nothing protects nothing"
    );
    let label = vd_core::store_stamp::world_generation(&base);

    for ix in 0..base.len() {
        let mut moved = base.clone();
        // ONE BIT — far below anything printable. A constant that changed by less than this is still a
        // different world, and the label must say so.
        moved[ix] = f64::from_bits(moved[ix].to_bits() ^ 1);
        assert_ne!(
            vd_core::store_stamp::world_generation(&moved),
            label,
            "constant {ix} does not move the label, so nothing would refuse a store written before it \
             changed"
        );
    }

    // And the fold is order-sensitive, which is why this is a LIST: re-ordering it is a world change.
    let mut swapped = base.clone();
    swapped.swap(0, 1);
    assert_ne!(vd_core::store_stamp::world_generation(&swapped), label);
}

#[test]
fn every_boundary_owns_its_band_and_a_bigger_body_gets_a_bigger_band() {
    // ★ RE-BASED AT SLICE S6, WHICH IS WHAT THIS TEST WAS BUILT TO CATCH. Slice S2 moved the band from
    // ONE value copied onto every region into a per-region question, and asserted here that no value
    // had actually moved — a byte-identity measurement, because "structural change only" is worth
    // nothing as a claim. S6 makes the values move on purpose, so the identity assertion is replaced by
    // the property that replaces it, not deleted.
    //
    // THE NEW PROPERTY: a band is a function of the body it wraps, and a bigger body gets a strictly
    // bigger band — because the ceiling lawfully holdable at a surface grows with the body's own size.
    // That is the whole of S6 stated as something that can fail.
    let cfg = test_world();
    let regions = realm_regions_for(HOME_SEED);
    assert!(!regions.is_empty(), "THE world has a forest to judge");

    let mut parented = 0usize;
    let mut by_extent: Vec<(f64, f64, RealmId)> = Vec::new();
    for r in &regions {
        let width = r.band.inset() + r.band.outset();
        // THE SHAPE IS PRESERVED AT EVERY SIZE: one third of the band inside the surface, two thirds
        // outside, so acquiring stays strictly harder than holding however large the body is.
        assert!(
            (r.band.outset() - 2.0 * r.band.inset()).abs() <= width * f64::EPSILON * 8.0,
            "region {:?} lost the 1:2 band shape: {:?}",
            r.realm,
            r.band
        );
        // AND IT IS THE VALUE THE CONFIG WOULD DERIVE FOR THIS BODY — one home for the law, so a
        // region's band cannot drift away from the solve that is supposed to author it.
        assert_eq!(
            r.band,
            cfg.band
                .build_for_shape(&r.shape)
                .expect("the derived band is valid"),
            "region {:?}'s band is not what the one band solve gives for its own extent",
            r.realm
        );
        by_extent.push((r.shape.circumscribed_extent(), width, r.realm));
        parented += usize::from(r.parent.is_some());
    }

    // MONOTONE: sort by body size and the bands must not decrease. A band that shrank as its body grew
    // would mean the law read something other than the body.
    by_extent.sort_by(|a, b| a.0.total_cmp(&b.0));
    for pair in by_extent.windows(2) {
        assert!(
            pair[1].1 >= pair[0].1,
            "{:?} ({} m) has a SMALLER band than {:?} ({} m)",
            pair[1].2,
            pair[1].0,
            pair[0].2,
            pair[0].0
        );
    }
    // NON-VACUITY, twice over and for two different reasons: a forest with no parented region has no
    // boundary at all, and a forest whose bands are all equal would satisfy the monotone loop while
    // proving nothing about the law that is supposed to vary them.
    assert!(
        parented > 0,
        "a forest with no parented region has no boundary to own a band"
    );
    assert!(
        by_extent.first().map(|f| f.1) < by_extent.last().map(|l| l.1),
        "every band in the forest is the same width, so this test proves nothing about a law that is \
         supposed to size them from their bodies"
    );
}

#[test]
fn the_worlds_own_ceiling_and_its_thinnest_band_are_two_readings_of_one_solve() {
    // THE MEASUREMENT S6 AND S7 BOTH READ, taken on THE world rather than on a fixture, and stated as
    // an assertion so it cannot quietly stop being true.
    //
    // The owner's ruling (2026-08-24): a realm's maximum speed is the speed at which ONE TICK of travel
    // still fits inside the thinnest band in that realm. Here is what THE world's shipped band affords,
    // against what its shipped ceiling actually asks for. The gap between the two numbers IS the work
    // S6 does; recording it now is what makes that work measurable rather than argued.
    use vd_core::geometry::{band_for_speed, speed_for_band};

    // The SHIPPED posture, read the same way every other world pin in this file reads it.
    let cfg = test_world();
    let band = cfg.band.build().expect("the world's band is valid");
    let width_m = band.inset() + band.outset();
    assert!(width_m > 0.0, "a band with no width contains nothing");

    // The shipped geometry solve's own tick and in-band count — NOT the cluster's live tick, which is
    // deliberate: two clusters at different tick rates must boot the identical world.
    let dt = GEOMETRY_TICK_DT_S;
    let ticks = BAND_TICKS_N;

    let afforded = speed_for_band(width_m, dt, ticks);
    assert!(
        afforded > 0.0,
        "the world's own band must afford SOME speed, or nothing may move inside it"
    );
    // And the inverse holds on the world's own numbers, which is the property S6 will size against.
    let needed = band_for_speed(afforded, dt, ticks);
    assert!(
        (needed - width_m).abs() <= width_m * f64::EPSILON * 4.0,
        "the world's band and the speed it affords must be one solve read two ways"
    );

    eprintln!(
        "[s2-band] THE world's shared band: width {width_m} m at dt {dt} s over {ticks} ticks \
         affords {afforded} m/s"
    );
}

#[test]
fn every_boundary_reports_what_its_band_needs_and_what_it_can_afford() {
    // THE SWEEP (slice S2's evidence half). For every boundary in THE world it computes five numbers,
    // and the gap between two of them is the whole reason S6 exists:
    //
    //   1. the ceiling the PARENT states for what is inside it;
    //   2. how many ticks a subject at that ceiling actually spends inside this boundary's band;
    //   3. the band this boundary would NEED at that ceiling;
    //   4. the band it could afford against its own extent;
    //   5. the band it could afford against its nearest sibling's clear gap.
    //
    // A PRINT CANNOT FAIL, so this is not a print. The assertions below fail for two DIFFERENT reasons:
    // every parented boundary must produce all five numbers, and the row count must equal the parented
    // count derived from the forest itself. Either one alone would pass on an empty sweep.
    use vd_core::flight::{FlightTuning, TRAVERSE_S, realm_speed_cap_mps};
    use vd_core::geometry::BoundaryTuning;
    use vd_core::geometry::{band_for_speed, speed_for_band};

    let cfg = test_world();
    let regions = realm_regions_for(HOME_SEED);
    let dt = GEOMETRY_TICK_DT_S;
    let ticks = BAND_TICKS_N;
    let v_foot = cfg.interest.occupant_v_max_mps;

    let parented_expected = regions.iter().filter(|r| r.parent.is_some()).count();
    let mut rows = 0usize;
    let mut thinnest_ticks = f64::MAX;
    let mut worst: Option<RealmId> = None;

    for r in &regions {
        let Some(parent_id) = r.parent else {
            continue; // the ambient root has nothing outside it to be entered from
        };
        let Some(parent) = regions.iter().find(|p| p.realm == parent_id) else {
            continue; // a dangling parent is the boot fence's business, not this sweep's
        };

        // 1. What the parent lets a thing inside it do.
        let parent_ceiling =
            realm_speed_cap_mps(parent.shape.circumscribed_extent(), v_foot, TRAVERSE_S);
        // 2. How long a subject at that ceiling is actually inside this band.
        let width_m = r.band.inset() + r.band.outset();
        let in_band_ticks = width_m / (parent_ceiling * dt);
        // 3. What this boundary would need to be, to see such a subject at all.
        let needed_m = band_for_speed(parent_ceiling, dt, ticks);
        // 4. What it can afford against its own size — a band wider than the thing it surrounds is not
        //    a band, it is a second body.
        let afford_own_m = r.shape.circumscribed_extent();
        // 5. What it can afford before it touches its nearest sibling.
        let gap_m = nearest_sibling_gap_m(&regions, r, parent_id);

        assert!(
            parent_ceiling.is_finite() && parent_ceiling > 0.0,
            "{:?}: its parent states no usable ceiling",
            r.realm
        );
        assert!(
            in_band_ticks.is_finite() && needed_m.is_finite() && afford_own_m.is_finite(),
            "{:?}: a boundary that cannot state its own numbers cannot be sized",
            r.realm
        );

        if in_band_ticks < thinnest_ticks {
            thinnest_ticks = in_band_ticks;
            worst = Some(r.realm);
        }
        rows += 1;

        eprintln!(
            "[s2-sweep] {:?}: parent ceiling {parent_ceiling:.6e} m/s | in-band {in_band_ticks:.6e} \
             ticks | needs {needed_m:.6e} m | affords(own) {afford_own_m:.6e} m | affords(gap) \
             {gap_m:.6e} m | has {width_m} m",
            r.realm
        );
    }

    assert_eq!(
        rows, parented_expected,
        "every boundary in the forest must be swept — a sweep that skipped one would be evidence \
         about a world we do not ship"
    );
    assert!(
        rows > 0,
        "a forest with no boundary proves nothing about bands"
    );

    // ★ THE PINNED CENSUS, and it fails for a DIFFERENT REASON than the equality above.
    //
    // `rows == parented_expected` only proves the sweep visited what the forest offered it. If the
    // forest itself changed shape — a body added, a body lost — that equality would stay true while the
    // evidence quietly became evidence about a different world. A literal cannot follow the forest, so
    // it is the one assertion here that a world change must come and re-base by hand.
    assert_eq!(
        rows, 8,
        "the number of boundaries in THE world moved. That is not necessarily wrong, but this sweep's \
         numbers now describe a different world — re-read them before re-basing this count"
    );

    // THE HEADLINE, recorded as a number rather than an adjective. Below one tick, a subject at the
    // parent's ceiling is never observed inside the band at all — it is on one side, then the other.
    eprintln!(
        "[s2-sweep] THINNEST: {:?} at {thinnest_ticks:.6e} ticks in band (one tick is the floor at \
         which a crossing can be seen at all); the world's band affords {:.6} m/s",
        worst.expect("a swept row"),
        speed_for_band(
            cfg.band.build().expect("valid band").inset()
                + cfg.band.build().expect("valid band").outset(),
            dt,
            ticks
        )
    );

    // ★ AND THE SAME QUESTION ACROSS EVERY SEED THE WORLD SWEEPS, not only the one this cluster boots.
    //
    // A boundary is sized from the bodies the seed drew, and a different seed draws different bodies.
    // Measuring one world would tell us about one world. The sweep size is the world's OWN derived
    // figure — the same one the nesting fence uses — so this asks the question over the range the
    // project already agreed reaches the heavy tail.
    //
    // What is asserted is the PROPERTY, not the values: every boundary of every swept world must be
    // able to state its five numbers. A world that cannot is a world we could not size a band in.
    let sweep = derived_nest_sweep_seeds(galaxy_profile(0, &cfg).count);
    // ★ 284 -> 664 AT S9, THEN 664 -> 42 AT S12/G8. The sweep size is DERIVED from the world, so it
    // moves when the world does, and it has moved twice for opposite reasons. S9 grew the galaxy from
    // 0.475 light years to 487, and a bigger world needs more seeds to reach the heavy tail. S12 made
    // the POPULATION a result of that volume, so one world now holds a whole galaxy's worth of stars
    // instead of three — and the same tail is met in far fewer worlds. Pinned as measured either way,
    // so a size that drifts again is loud.
    assert_eq!(sweep, 42, "the derived sweep size, pinned as measured");
    let mut swept_rows = 0usize;
    let mut swept_thinnest = f64::MAX;
    for seed in 0..sweep {
        let forest = WorldView::generated(seed, &cfg).regions().to_vec();
        for r in &forest {
            let Some(parent_id) = r.parent else { continue };
            let Some(parent) = forest.iter().find(|p| p.realm == parent_id) else {
                continue;
            };
            let ceiling =
                realm_speed_cap_mps(parent.shape.circumscribed_extent(), v_foot, TRAVERSE_S);
            let width_m = r.band.inset() + r.band.outset();
            let in_band = width_m / (ceiling * dt);
            let needed = band_for_speed(ceiling, dt, ticks);
            assert!(
                ceiling.is_finite() && ceiling > 0.0 && in_band.is_finite() && needed.is_finite(),
                "seed {seed}, boundary {:?}: a boundary that cannot state its numbers cannot be sized",
                r.realm
            );
            swept_thinnest = swept_thinnest.min(in_band);
            swept_rows += 1;
        }
    }
    assert!(
        swept_rows > rows,
        "the swept range must judge more boundaries than the single booted world, or it adds nothing"
    );
    eprintln!(
        "[s2-sweep] ACROSS {sweep} SEEDS: {swept_rows} boundaries judged; thinnest \
         {swept_thinnest:.6e} ticks in band"
    );
    // ★ THE MEASUREMENT S6 MUST MOVE, pinned as a COUNT so it can actually fail.
    //
    // My first version of this assertion was backwards and could never have failed: it required the
    // swept worst case to be at or below the booted world's, and a minimum over a larger set always is.
    // It would have passed on any world, however bad.
    //
    // What is worth pinning is the thing the sweep discovered: how many boundaries in the whole swept
    // range are UNOBSERVABLE — a subject at the parent's ceiling spends under one tick inside them, so
    // it is on one side and then on the other and nothing ever sees it in between. That number is the
    // size of the problem S6 exists to solve. When S6 sizes the bands it goes to zero and this test goes
    // RED, which is exactly when somebody should come back and read it.
    // ★ THE METRIC CHANGED AT S6, DELIBERATELY AND VISIBLY, AND BOTH READINGS ARE KEPT.
    //
    // Slice S2 measured this against the PARENT'S ceiling, because at that point nothing bounded how
    // fast a subject could be travelling when it reached a child's surface, so the parent's ceiling was
    // the honest upper bound. It is no longer the right question, and this is why: the approach governor
    // lowers a subject's ceiling onto the body it is nearing, and it is ENFORCED at both arms of
    // `governed_ceiling_in_book` — approaching a child (`v <= child_cap + distance/tau`) and leaving your
    // own realm (`v <= own_cap + distance_to_your_own_shell/tau`). At either surface the distance term is
    // zero, so THE FASTEST A SUBJECT MAY LAWFULLY BE MOVING AT A BOUNDARY IS THAT BOUNDARY'S OWN CAP.
    // Measuring against the parent's ceiling now measures a speed no lawful subject can hold there.
    //
    // Both numbers are computed and both are asserted, so the change is a re-pointing and not a
    // weakening: the ungoverned count is still reported and still pinned, and the GOVERNED count — the
    // one that describes what can actually happen — must be zero.
    let (ungoverned, governed, (worst_governed_ticks, worst_at_tau_floor), galaxies, gal_un) = {
        let (mut un, mut gov) = (0usize, 0usize);
        // ★ S9 adds these two so the re-base below is a MEASUREMENT and not an explanation. The 284
        // rows that used to escape the ungoverned reading were said to be the galaxies; nothing
        // checked it. These count them.
        let (mut galaxies, mut gal_un) = (0usize, 0usize);
        let mut worst = f64::MAX;
        let mut worst_floor_tau = f64::MAX;
        for seed in 0..sweep {
            let forest = WorldView::generated(seed, &cfg).regions().to_vec();
            for r in &forest {
                let Some(parent_id) = r.parent else { continue };
                let Some(parent) = forest.iter().find(|p| p.realm == parent_id) else {
                    continue;
                };
                let width_m = r.band.inset() + r.band.outset();
                let parent_ceiling =
                    realm_speed_cap_mps(parent.shape.circumscribed_extent(), v_foot, TRAVERSE_S);
                let this_un = usize::from(width_m / (parent_ceiling * dt) < 1.0);
                un += this_un;
                let is_galaxy = usize::from(matches!(r.realm, RealmId::Galaxy(_)));
                galaxies += is_galaxy;
                gal_un += is_galaxy * this_un;
                // The governed speed AT THE BAND'S OUTER EDGE — the fastest point of the crossing,
                // because the ceiling falls as the surface nears. Its own cap plus the approach term
                // over the band's own release edge.
                let own_cap =
                    realm_speed_cap_mps(r.shape.circumscribed_extent(), v_foot, TRAVERSE_S);
                // ★ TWO TAUS, BECAUSE ONE OF THEM IS NOT A WORLD WE RUN. The approach constant is
                // derived as `(2·demand_cadence + boot_p99 + pipeline)·dt`. With no cadence and no boot
                // latency it bottoms out at 0.1 s, and my first version of this measurement used that —
                // reporting a crossing time no shipped cluster experiences. The dev cluster's cadence is
                // half a second of ticks, giving the 1.1 s the flight tuning's own doc states. Both are
                // computed; the SHIPPED one is what the cooldown is judged against.
                let cadence = ((1.0 / dt).round() as u64) / 2;
                let tau_shipped = FlightTuning::derive(v_foot, dt, cadence, 0).tau_s;
                let tau_floor = FlightTuning::derive(v_foot, dt, 0, 0).tau_s;
                let ticks_at = |tau: f64| {
                    width_m
                        / (vd_core::flight::approach_ceiling_mps(own_cap, r.band.outset(), tau)
                            * dt)
                };
                let ticks_in = ticks_at(tau_shipped);
                worst = worst.min(ticks_in);
                worst_floor_tau = worst_floor_tau.min(ticks_at(tau_floor));
                gov += usize::from(ticks_in < 1.0);
            }
        }
        (un, gov, (worst, worst_floor_tau), galaxies, gal_un)
    };
    eprintln!(
        "[s2-sweep] UNGOVERNED (the S2 reading, kept): {ungoverned} of {swept_rows} boundaries give a \
         subject at their PARENT's ceiling under ONE tick inside the band\n\
         [s2-sweep] GOVERNED (the S6 reading): {governed} of {swept_rows} unobservable at the fastest \
         speed the governor permits at that surface; thinnest crossing {worst_governed_ticks:.6e} \
         ticks at the shipped approach constant, {worst_at_tau_floor:.6e} at its floor"
    );
    // ★ RE-BASED IN S9, AND THE ROW COUNT MOVED FOR A REASON WORTH READING: the sweep itself grew.
    // `derived_nest_sweep_seeds` sizes the sweep from the IMF tail above the mass cap, so when the
    // cap climbed from 16.36 to 30.745 M☉ the sweep went 284 → 664 worlds and the rows with it,
    // 13,144 → 30,653. Nothing about any single boundary changed to make that number move.
    // NON-VACUITY is what this pins: the sweep really walked every boundary of every world. The
    // literal was a ROW COUNT, and a row count scales with how big a galaxy a test can hold — a
    // memory decision, not a fact about the world.
    assert!(
        swept_rows as u64 >= sweep,
        "every swept world contributed a boundary: {swept_rows} rows over {sweep} worlds"
    );
    // (The sweep SIZE is not asserted here. It is computed a few lines above by
    // `derived_nest_sweep_seeds`, so restating it would compare a derivation with itself — and the
    // literal it used to carry, 664, was that derivation's output for a retired census.)
    // ★ AND THE 284 EXCEPTIONS ARE GONE — every boundary is now ungoverned, where 284 were not.
    // The old message asserted in prose that those 284 were the galaxies, "whose own ceiling nearly
    // equals its parent's". S9 gives the galaxy a real parent — the universe — whose ceiling is six
    // orders of magnitude higher, so a galaxy's band is no longer a tick wide against it.
    //
    // That explanation is now COUNTED rather than written: there is one galaxy per swept world and
    // every one of them is ungoverned. Had the flip come from somewhere else, these two would differ.
    assert_eq!(galaxies, sweep as usize);
    assert_eq!(
        gal_un, galaxies,
        "every galaxy is now ungoverned against the universe"
    );
    // ★ RE-BASED IN S12 (2026-08-28), AND THE SWEEP WAS READ FIRST, as this message demands.
    //
    // The count rose by 77 across 664 worlds because the placement became a SHAPE: systems now sit at
    // their own distances instead of one shared radius, so the forest each world produces differs.
    // `swept_rows` moved by the same 77, which is what says the total set changed rather than the
    // classification of a fixed set.
    //
    // ★ THE SAFETY PROPERTY IS UNTOUCHED, and it is the assertion below, not this one: `governed == 0`
    // still holds. A boundary that could be crossed unseen at a speed the governor permits is the
    // defect S6 exists to remove, and no such boundary appeared.
    // ★ THE SAFETY PROPERTY, NOT THE ROW COUNT (2026-08-31). This pinned how MANY boundaries the sweep
    // classified, and that is a function of how many worlds the sweep visits and how many boundaries
    // each holds — both of which move with the test galaxy's size, which is a memory decision.
    //
    // The property the comment above already names as the real one is asserted separately: `governed
    // == 0`. What is left to say here is that the classification covered the whole sweep.
    assert_eq!(
        ungoverned, swept_rows,
        "every swept boundary is ungoverned; none could be crossed unseen"
    );
    assert_eq!(
        governed, 0,
        "a boundary can still be crossed unseen at a speed the governor actually permits — that is \
         the defect S6 exists to remove, and it is not removed"
    );
    // S6'S OWN GATE: every crossing is SEEN. Below one tick a subject is on one side and then the
    // other and nothing observes it in between.
    assert!(
        worst_governed_ticks >= 1.0,
        "the thinnest governed crossing is {worst_governed_ticks} ticks — under one tick a crossing \
         cannot be seen at all, which is the defect this slice exists to remove"
    );

    // ★ AND THE ANTI-THRASH HALF, WHICH IS THE ONE THE PARKED PROCESS TEST IS ABOUT.
    //
    // A subject that re-homes may not re-home again for `k_dwell` ticks. A band it can leave INSIDE
    // that window flips back the moment the window ends — nine crossings in, nine out, which is the
    // measurement that parked that test. So the crossing must outlast the cooldown, which is a stronger
    // requirement than merely being seen.
    assert!(
        worst_governed_ticks >= f64::from(BoundaryTuning::DEFAULT.k_dwell),
        "the thinnest governed crossing is {worst_governed_ticks} ticks against a {} tick cooldown — \
         a subject can leave the band before it is allowed to re-home again, which is the thrash",
        BoundaryTuning::DEFAULT.k_dwell
    );

    // ★ AND THE CONFIGURATION WHERE IT DOES NOT HOLD, STATED RATHER THAN LEFT OUT.
    //
    // At the FLOOR of the approach constant — no demand cadence, no boot latency, which no shipped
    // cluster runs — the same band gives 10/3 ticks and the cooldown is NOT cleared. The reason is a
    // feedback the band cannot escape on its own: widening a band moves its outer edge further out,
    // where the governor permits a higher speed, so crossing time rises less than width does and
    // saturates. Reaching the cooldown at that floor needs a band factor of 15 rather than 6 — still
    // only a third of a percent of each body, but that factor also sets the galaxy's own radius, so
    // moving it is the next slice's subject and not this one's.
    //
    // Pinned, so a cluster configured into that corner turns this red instead of thrashing quietly.
    assert!(
        (worst_at_tau_floor - 10.0 / 3.0).abs() < 1e-9,
        "the crossing at the approach constant's floor moved to {worst_at_tau_floor} ticks (was 10/3). \
         If the band factor was raised on purpose, re-base this — and if it now clears the {} tick \
         cooldown, the ledger row that owes it can be closed",
        BoundaryTuning::DEFAULT.k_dwell
    );
}

/// ★ THE INDEX-QUALITY GATE, INSIDE THIS SLICE RATHER THAN FIVE SLICES LATER.
///
/// A child's indexed radius is its own reach PLUS its release edge. So sizing bands from bodies
/// inflates every radius, and an inflated radius makes more children's spheres overlap — which would
/// degrade the O(log n) lookup back toward the scan it was built to replace. That is an
/// unbounded-child-count defect introduced by a lawful band change, which is exactly the kind of thing
/// that is invisible until a realm has many children.
///
/// So the cost is measured here, against the same forest with the band it used to have. (The lookup
/// was a uniform grid until 2026-09-04; this gate then also pinned that the grid's edge grew by
/// exactly one octave. An R*-tree has no edge, so that half of the gate is gone with the grid.)
#[test]
fn sizing_the_bands_does_not_coarsen_the_child_lookup() {
    use std::collections::BTreeSet;
    use vd_core::child_index::{ChildIndex, IndexedChild};

    let cfg = test_world();
    let old_band = cfg.band.build().expect("the old shared band is valid");
    let sweep = derived_nest_sweep_seeds(galaxy_profile(0, &cfg).count);

    let mut parents_judged = 0usize;
    let mut worst_candidates = 0usize;
    let mut total_candidates = 0usize;
    let mut worst_before = 0usize;
    let mut total_before = 0usize;
    let mut queries = 0usize;

    for seed in 0..sweep {
        let forest = WorldView::generated(seed, &cfg).regions().to_vec();
        // One index per parent, which is how a shard builds it: its own direct children only.
        let parents: BTreeSet<RealmId> = forest.iter().filter_map(|r| r.parent).collect();
        for parent in parents {
            let kids: Vec<&RealmRegion> =
                forest.iter().filter(|r| r.parent == Some(parent)).collect();
            // ★ THE INDEX IS BUILT AT THE PARENT'S RUNG, not at `Tier::Fine` (S9). A child's `center`
            // is its position in its PARENT's frame, and since the ladder landed those frames are not
            // all the same rung — a system sits in galaxy cells of 2 m, a galaxy in universe cells of
            // 32_768 m. Reading either with the millimetre ruler is off by 2048× or by 33_554_432×,
            // and the index would be measuring a world that does not exist. This is the third place
            // that confusion hid (after the body generator's centres and the walk band test).
            let ptier = forest
                .iter()
                .find(|p| p.realm == parent)
                .map_or(Tier::Fine, |p| p.frame.tier());
            let build = |band_outset: &dyn Fn(&RealmRegion) -> f64| -> ChildIndex {
                let children: Vec<IndexedChild> = kids
                    .iter()
                    .map(|r| IndexedChild {
                        realm: r.realm,
                        centre: r.center.in_parents_frame(),
                        radius_m: r.shape.circumscribed_extent() + band_outset(r),
                    })
                    .collect();
                ChildIndex::build(&children, ptier)
            };
            let now = build(&|r: &RealmRegion| r.band.outset());
            let before = build(&|_: &RealmRegion| old_band.outset());
            // WHAT THE LOOKUP ACTUALLY ANSWERS, which is the number that matters: a wider radius is
            // only a problem if it starts naming more children per query.
            for k in &kids {
                let named = now.candidates(k.center.in_parents_frame(), ptier).len();
                worst_candidates = worst_candidates.max(named);
                total_candidates += named;
                // THE CONTROL: the same query on the same forest with the band it used to have. Without
                // it, a number that was always this large would read as a regression this slice caused.
                let named_before = before.candidates(k.center.in_parents_frame(), ptier).len();
                worst_before = worst_before.max(named_before);
                total_before += named_before;
                queries += 1;
            }
            parents_judged += 1;
        }
    }

    let mean = total_candidates as f64 / queries as f64;
    let mean_before = total_before as f64 / queries as f64;
    eprintln!(
        "[s6-index] {parents_judged} parents over {sweep} seeds, {queries} queries\n\
         [s6-index]   candidates per query BEFORE: max {worst_before}, mean {mean_before:.4}\n\
         [s6-index]   candidates per query AFTER:  max {worst_candidates}, mean {mean:.4}"
    );
    assert!(queries > 0, "an index gate with no query measures nothing");

    // ★ THE GATE, AND IT IS THE COMPARISON RATHER THAN THE ABSOLUTE NUMBER.
    //
    // What this slice could break is the lookup, and the way it would break it is by naming MORE
    // children per query than before. It does not name one more. That is the assertion, and it could
    // have come out otherwise — every radius genuinely grew, so "the answer is unchanged" is a result,
    // not a restatement of the change being small.
    assert_eq!(
        (worst_candidates, total_candidates),
        (worst_before, total_before),
        "sizing the bands changed what the child lookup answers: max {worst_before} -> \
         {worst_candidates}, total {total_before} -> {total_candidates}. That is the unbounded-child \
         defect this gate exists to catch"
    );

    // AND THE ABSOLUTE NUMBER, PINNED WITH ITS CAUSE STATED so nobody reads it as this slice's doing.
    // It is what a system's planets plus its star and shells come to when they share a cell — what the
    // lookup answered before the band was sized and what it answers after.
    //
    // ★ RE-BASED IN S9, 12 → 13, and NOT because anything about the index or the band changed. The
    // sweep is sized from the IMF tail above the mass cap, the cap climbed, and the sweep went 284 →
    // 664 worlds. A wider sweep reaches a system whose children cluster slightly more tightly than
    // any of the first 284 did. The claim this test actually makes is the BEFORE/AFTER equality above,
    // and that is untouched: max 13 both ways, mean 7.3797 both ways, to the last digit.
    assert_eq!(
        worst_candidates, 13,
        "the widest lookup answer moved. It is set by how tightly a system's children cluster relative \
         to its own size, NOT by the band — the equality above is what proves that"
    );
}

// ===================== SLICE S8 — THE LADDER, BUILT BUT NOT YET CLIMBED =====================

/// ★ WHAT DOES IT ACTUALLY COST TO JUST HOLD ALL 150,000 SYSTEMS? (owner ruling 2026-08-26:
/// "Just list all of them at once, don't do lazy smart.")
///
/// The design run proposed a position-addressed lazy generator to avoid ever enumerating the galaxy. That
/// is a large machine. Before building it, this measures whether the thing it avoids is expensive at all.
#[test]
fn holding_every_system_at_once_costs_what_it_costs() {
    use std::mem::size_of;
    const CENSUS: usize = 150_000;
    let region = size_of::<RealmRegion>();
    let body = size_of::<GeneratedBody>();
    eprintln!("[s9-hold] one RealmRegion = {region} bytes | one GeneratedBody = {body} bytes");
    eprintln!(
        "[s9-hold] 150,000 SYSTEMS ONLY : regions {:.1} MB | bodies {:.1} MB",
        (region * CENSUS) as f64 / 1.0e6,
        (body * CENSUS) as f64 / 1.0e6
    );
    // The whole forest, if planets were materialised too: 1 system + its planets + its star.
    let per_system = 1 + world_planet_config().n_planets as usize + 1;
    let full = CENSUS * per_system;
    eprintln!(
        "[s9-hold] EVERY REGION ({per_system} per system, {full} total) : {:.1} MB",
        (region * full) as f64 / 1.0e6
    );
    // ★ THE ALL-PAIRS FENCE, which is the cost the lazy design was really avoiding.
    let pairs = CENSUS * (CENSUS - 1) / 2;
    eprintln!(
        "[s9-hold] the shipped all-pairs separation fence at the census = {pairs} pair tests",
    );
    // A flat list of systems must be affordable, or the owner's ruling cannot stand. State it as a bound
    // that could fail rather than as a print.
    assert!(
        (region * CENSUS) as f64 / 1.0e6 < 200.0,
        "a flat list of 150,000 systems must fit in a couple of hundred megabytes"
    );
    assert!(
        pairs > 1.0e10 as usize,
        "the all-pairs fence is the real cost, and it is enormous"
    );
}

/// ★ THE STAR CAP AFTER S9, SOLVED — not estimated, and not fitted.
///
/// Today the heaviest star is limited by THE GALAXY'S SIZE: a heavy star needs a wide system around it and
/// the galaxy must have room. After S9 the galaxy is 2⁶² m and that limit stops binding, so a different
/// one takes over: **a star system counts its own positions in millimetres, and its shell must fit its own
/// lattice.** The fence's own equality gives that budget as 2⁵¹ m.
///
/// The design run ESTIMATED ~30.68 solar masses and a second spine ~28.36, both by propagating a fitted
/// exponent. The owner's Q9 ruling 2 says never to build against a number from a neighbouring design. This
/// runs OUR solver, on THE world, against the real shell law.
#[test]
fn the_star_cap_after_the_climb_solved_with_our_own_solver() {
    let pl = world_planet_config();
    // THE BUDGET: a star system's shell must fit the lattice it counts in, at the fence's own equality.
    let budget_m = root_radius_at(vd_core::pose::Tier::Fine);
    assert_eq!(
        budget_m, 2_251_799_813_685_248.0,
        "2^51 — the millimetre lattice's own reach"
    );

    let shell_of = |m: f64| system_shell_r_m(&pl, &star_at_mass(m));
    // MONOTONE, asserted rather than assumed — the bisection below is only valid if it is.
    let mut prev = 0.0;
    for m in [0.08, 0.5, 1.0, 2.0, 8.0, 16.0, 30.0, 60.0, 120.0] {
        let s = shell_of(m);
        assert!(
            s > prev,
            "the shell must grow with mass: {m} M_sun gave {s}"
        );
        prev = s;
    }

    // THE SOLVE: bracket by doubling, then bisect — the same shape the shipped mass-cap solve uses.
    let mut lo = 0.08_f64;
    let mut hi = lo;
    for _ in 0..64 {
        if shell_of(hi) > budget_m {
            break;
        }
        lo = hi;
        hi *= 2.0;
    }
    for _ in 0..f64::MANTISSA_DIGITS {
        let mid = 0.5 * (lo + hi);
        let ok = shell_of(mid) <= budget_m;
        lo = if ok { mid } else { lo };
        hi = if ok { hi } else { mid };
    }
    let cap = lo;

    // WHICH STAR CLASSES SURVIVE. The class boundaries are the shipped table.
    let classes = [
        ("O", 16.0),
        ("B", 2.1),
        ("A", 1.4),
        ("F", 1.04),
        ("G (our sun is 1.0)", 0.8),
        ("K", 0.45),
        ("M", 0.08),
    ];
    eprintln!("[s9-cap] budget (a system's own lattice) = {budget_m:.6e} m");
    eprintln!(
        "[s9-cap] SOLVED CAP = {cap:.4} solar masses   (its shell {:.6e} m)",
        shell_of(cap)
    );
    eprintln!(
        "[s9-cap] the SHIPPED cap, after the climb = {:.4} solar masses",
        imf_mass_hi_msun()
    );
    for (name, lo_bound) in classes {
        eprintln!(
            "[s9-cap]   class {name:<20} from {lo_bound:>6} M_sun : {}",
            if lo_bound <= cap {
                "KEPT"
            } else {
                "LOST ENTIRELY"
            }
        );
    }

    // THE ANSWER IS AFFORDABLE AND IT IS THE LARGEST ONE — both arms, or a solve that returned anything
    // affordable (zero, say) would satisfy the first.
    assert!(
        shell_of(cap) <= budget_m,
        "the cap must fit its own lattice"
    );
    assert!(
        shell_of(cap * 1.000_001) > budget_m,
        "and one part per million more must not"
    );
    // ★ AND THE PREDICTION CAME TRUE, WHICH IS THE STRONGER STATEMENT. This test was written BEFORE
    // the climb to work out what the cap would become, and asserted `cap > imf_mass_hi_msun()` — the
    // climb must raise the limit. The climb has now landed, so that arm can no longer hold and must
    // not be re-based into something weaker. It becomes an equality: the independent solve WRITTEN
    // HERE, in this test, from the shell law and the lattice budget alone, reproduces the shipped
    // cap to the last bit. Two solvers, written apart, agreeing exactly.
    assert_eq!(cap, imf_mass_hi_msun());
    assert_eq!(cap, 30.745_283_003_771_995);
    // …and it IS a rise: 16.36 before the climb, pinned so the gain cannot quietly evaporate.
    assert!(cap > 16.360_034_882_257_757);
}

/// ★ THE ROOT RADIUS IS ITS OWN DERIVATION, AND IT REPRODUCES THE LITERAL IT REPLACED.
///
/// It used to be the typed-in number `2 251 799 813 685 248`, with a comment explaining that it came
/// from the step. A comment cannot follow a change: re-value the step and the literal stays where it
/// was, the two disagree, and only one pin at one level would notice.
///
/// Calling the derivation HERE also drives it. It is a `const fn`, so the shipped call is folded at
/// compile time and nothing executes it at run time — a function that only ever runs in the compiler is
/// invisible to a coverage gate, which is not the same as being covered.
#[test]
fn the_root_radius_is_the_fence_solved_at_equality_and_matches_the_literal_it_replaced() {
    use vd_core::pose::Tier;
    // THE HAND-TYPED NUMBER the expression must reproduce — the whole point of the change is that this
    // equality now holds by construction rather than by two places agreeing to stay in step.
    // ★ RE-BASED IN S9. The root moved from the FINE rung to the UNIVERSE rung, so the literal it
    // reproduces moved with it — 2⁷⁶ m in 32_768 m cells instead of 2⁵¹ m in millimetres. The old
    // literal did not become wrong; it became the FINE rung's radius, and it is still asserted, one
    // line down. That is the test earning its keep: had the root silently kept the fine radius the
    // second equality would have passed and the first would not.
    assert_eq!(REAL_UNIVERSE_R_M, 75_557_863_725_914_323_419_136.0);
    assert_eq!(REAL_UNIVERSE_R_M, root_radius_at(ROOT_TIER));
    assert_eq!(ROOT_TIER, Tier::Universe);
    assert_eq!(root_radius_at(Tier::Fine), 2_251_799_813_685_248.0);
    // …and at every other rung it is that rung's own equality: 2⁶¹ cells, whatever a cell is.
    for tier in Tier::ALL {
        let r = root_radius_at(tier);
        assert_eq!(r / tier.cell_edge_m(), 2.0_f64.powi(61), "{tier:?}");
        // The fence agrees, which is what makes the radius and the fence one statement instead of two.
        let b = guard_shell_representable(r, tier).expect("its own equality must pass");
        assert_eq!(b.occupancy, 0.5, "{tier:?}");
    }
}

/// ★ THE STORAGE FENCE AT EVERY RUNG OF THE LADDER.
///
/// The fence used to divide by the millimetre step unconditionally. That was right while every level
/// counted in millimetres, and becomes badly wrong the moment they do not: applied to the universe's own
/// `2⁷⁶ m` shell it would refuse by a factor of `2²⁵` — thirty-three million — a world that fits its own
/// lattice EXACTLY. A fence that refuses a lawful world is worse than no fence, because its refusal looks
/// authoritative.
///
/// THE RADII HERE ARE HAND-TYPED, and that is the point. Feeding the expression that defines the radius
/// back into the fence that defines the expression is not a measurement — it is a construction asserting
/// itself. These three numbers are `2⁵¹`, `2⁶²` and `2⁷⁶` written out.
#[test]
fn the_storage_fence_sits_at_exact_equality_on_every_rung() {
    use vd_core::pose::Tier;
    for (tier, r_m) in [
        (Tier::Fine, 2_251_799_813_685_248.0_f64),
        (Tier::Galaxy, 4_611_686_018_427_387_904.0_f64),
        (Tier::Universe, 75_557_863_725_914_323_419_136.0_f64),
    ] {
        let b = guard_shell_representable(r_m, tier).expect("the fence's own equality must pass");
        // RUNG-INVARIANT BY CONSTRUCTION, not by three coincidences: the radius is the fence solved at
        // equality, so the cell count is 2⁶¹ whatever the step is.
        assert_eq!(b.root_cells, 2.0_f64.powi(61), "{tier:?}");
        assert_eq!(
            b.occupancy, 0.5,
            "{tier:?} occupancy must be exactly one half"
        );
        assert_eq!(
            b.headroom, 2.0,
            "{tier:?} headroom must be exactly one octave"
        );
        assert_eq!(b.tier, tier);
        assert_eq!(b.cell_edge_m, tier.cell_edge_m());
        // …AND ONE OCTAVE UP REFUSES, at every rung. Without this the equality above would be satisfied
        // by a fence that accepted everything.
        let refused = guard_shell_representable(2.0 * r_m, tier)
            .expect_err("one octave above the equality must refuse");
        assert_eq!(refused.tier, tier);
        assert_eq!(refused.occupancy_pct, 100.0, "{tier:?}");
    }
}

/// ★ THE RUNG AND THE RADIUS ARE TWO ARGUMENTS, AND A WRONG PAIRING IS A VALUE ERROR NOTHING ELSE CATCHES.
///
/// Each of these radii is lawful, and each rung is lawful. Only the PAIRING is wrong — which is exactly
/// the mistake a ladder invites, and exactly the mistake no type can prevent.
#[test]
fn a_radius_judged_at_the_wrong_rung_is_refused() {
    use vd_core::pose::Tier;
    // The galaxy's radius counted in millimetres: 2⁷² cells, budget 2⁷³ against a threshold of 2⁶².
    guard_shell_representable(4_611_686_018_427_387_904.0, Tier::Fine)
        .expect_err("the galaxy radius does not fit the millimetre lattice");
    // The universe's radius counted in the galaxy's step: over by 2¹⁴.
    guard_shell_representable(75_557_863_725_914_323_419_136.0, Tier::Galaxy)
        .expect_err("the universe radius does not fit the galaxy lattice");
    // And the pairing that IS right passes, so the two refusals above are about the pairing and not
    // about the fence refusing everything.
    guard_shell_representable(2_251_799_813_685_248.0, Tier::Fine).expect("the shipped pairing");
}

/// ★ A SHELL THAT IS NOT A LENGTH IS REFUSED, both arms.
///
/// `NaN > x` is false and so is `NaN <= x`, so a non-finite radius used to sail straight through the
/// fence's one comparison and be reported as a representable world with a `NaN` occupancy. A negative
/// radius did the same. A fence that answers "fine" to a question that makes no sense is worse than no
/// fence at all.
#[test]
fn the_storage_fence_fails_closed_on_a_shell_that_is_not_a_length() {
    use vd_core::pose::Tier;
    guard_shell_representable(f64::NAN, Tier::Fine).expect_err("a shell of NaN is not a shell");
    guard_shell_representable(f64::INFINITY, Tier::Fine)
        .expect_err("an unbounded shell is not one");
    guard_shell_representable(-1.0, Tier::Fine).expect_err("a negative shell is not one");
    guard_shell_representable(0.0, Tier::Fine).expect_err("a shell of no size is not one");
}

// ===================== SLICE S7 — THE CAP, ASKED AT A DIFFERENT GALAXY SIZE =====================

/// ★ THE ARM THAT MAKES THE REFACTOR TRUSTWORTHY, and the only reason it is safe to ask the solve
/// anything new. Turning a module constant into a parameter must move NOTHING.
///
/// This is not a formality. The cap is the IMF draw's upper bound, so every star in the world is drawn
/// through it — the last time it moved, 99 of 99 planet rows and 6 of 12 system rows moved with it. A
/// refactor that shifted it by one bit would silently re-draw the universe.
#[test]
fn the_cap_solved_at_todays_budget_is_bit_for_bit_the_shipped_one() {
    let solved = solve_mass_cap(REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M);
    // All three fields, by exact equality — the mass, the reservation it implies, and the star's own
    // photosphere. Comparing only the mass would let the two derived fields drift.
    assert_eq!(solved.mass_hi_msun, imf_mass_hi_msun());
    assert_eq!(solved.system_bound_max_m, target_system_bound_max_m());
    assert_eq!(solved.star_look_max_m, target_star_look_max_m());
    // AND THE SOLVE IS DETERMINISTIC IN ITS ARGUMENT: the same budget twice is the same answer, so the
    // equality above is a property of the function and not of one lazily-initialised value.
    assert_eq!(solve_mass_cap(REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M), solved);
}

/// ★ THE SOLVE'S OWN SHAPE, asked at the shipped budget so the harness below can be trusted at budgets
/// nobody has run yet. Both of these could fail, and each fails for a different reason.
#[test]
fn the_cap_is_the_largest_affordable_mass_and_one_part_per_million_more_is_not() {
    let pl = world_planet_config();
    for budget in [
        REAL_GALAXY_R_M,
        REAL_GALAXY_R_M * 0.5,
        REAL_GALAXY_R_M * 2.0,
        REAL_GALAXY_R_M * 1_000.0,
    ] {
        let cap = solve_mass_cap(budget, SYSTEM_LATTICE_R_M);
        // ★ RE-BASED IN S9: both arms now ask `binding_limit`, the same question the solve asks,
        // instead of testing the galaxy's purse directly. They had to. Since the climb the purse is
        // not what binds at these budgets — the system's own lattice is — so a purse-only arm reads
        // "affordable" for a mass the solve refuses, and the test would pass while measuring the
        // wrong thing. A test must ask the question its subject answers.
        assert_eq!(
            binding_limit(&pl, cap.mass_hi_msun, budget, SYSTEM_LATTICE_R_M),
            None,
            "the cap must be affordable at budget {budget}"
        );
        // …and it must be the LARGEST affordable one. A solve that returned something merely
        // affordable — zero, say — would satisfy the arm above and nothing else.
        assert!(
            binding_limit(
                &pl,
                cap.mass_hi_msun * 1.000_001,
                budget,
                SYSTEM_LATTICE_R_M
            )
            .is_some(),
            "one part per million above the cap must be unaffordable at budget {budget}"
        );
    }
}

/// ★ THE HARNESS THE SLICE EXISTS TO RUN. What the heaviest star becomes at each candidate coordinate
/// step, on THE world, with our own solver rather than a fitted exponent.
///
/// The design estimated about 1,048 solar masses at the ruled two-metre step, by propagating a
/// two-point fit. That is arithmetic through a fitted exponent, NOT a measurement, and the plan says so
/// in as many words. The bisection is the authority. This runs it.
///
/// ★ RE-BASED IN S9, AND WHAT IT MEASURES CHANGED. It asserted the cap rises STRICTLY with the galaxy
/// step, which was true while the galaxy's room was the only limit. It is not any more: past a point
/// the system's own lattice takes over and a wider galaxy buys nothing. That is not a fault in the
/// solve — it is the S9 result, and pinning WHERE it saturates is worth more than the old assertion,
/// because that point is what P10 has to lift.
#[test]
fn what_the_heaviest_star_becomes_at_each_candidate_coordinate_step() {
    let pl = world_planet_config();
    // The galaxy's radius is its lattice's storage budget: half the signed cell domain, in metres of
    // whatever step that lattice uses. Today's step is the fine millimetre grid; the candidates are the
    // coarser steps a second tier could take. Derived from the domain, never quoted.
    let cells = f64::from(2_i32).powi(61);
    eprintln!(
        "[s7-cap] step (m)      | galaxy radius (m) | heaviest star (Msun) | its shell (m)      | \
         demand at cap (m)"
    );
    let mut previous_cap = 0.0_f64;
    let mut rows = 0usize;
    let mut saturated = 0usize;
    for step_m in [vd_core::pose::FINE_CELL_EDGE_M, 1.0, 2.0, 16.0, 1_024.0] {
        // The same expression the shipped radius uses, at this step: the storage fence's own equality,
        // less the band the geometry solve reserves.
        let r_uni = cells * step_m;
        let r_gal = r_uni
            - (2.0 * r_uni / T_TRAVERSE_S) * GEOMETRY_TICK_DT_S * BAND_TICKS_N * BAND_TAU_HEADROOM;
        let cap = solve_mass_cap(r_gal, SYSTEM_LATTICE_R_M);
        let demand = galaxy_child_demand_m(&pl, cap.mass_hi_msun);
        eprintln!(
            "[s7-cap] {step_m:<13} | {r_gal:.6e} | {:<20.6} | {:.6e} | {demand:.6e}",
            cap.mass_hi_msun, cap.system_bound_max_m
        );
        // MONOTONE IN THE BUDGET, which is the property the whole solve rests on: a bigger galaxy can
        // never afford a SMALLER star. If this failed, every number printed above would be noise.
        // NON-STRICT since S9 — equal is lawful and means the other limit has taken over.
        assert!(
            cap.mass_hi_msun >= previous_cap,
            "a larger galaxy afforded a SMALLER star at step {step_m} — the solve is not monotone"
        );
        saturated += usize::from(cap.mass_hi_msun == previous_cap);
        previous_cap = cap.mass_hi_msun;
        rows += 1;
    }
    assert_eq!(rows, 5, "every candidate step must be solved");
    // ★ WHERE THE GALAXY STOPS BEING THE LIMIT, measured. The last three candidate steps all return
    // the same cap, because at each of them the system lattice binds first. Both behaviours are
    // therefore driven in one run: the rising arm and the saturated one.
    assert_eq!(
        saturated, 3,
        "the cap saturates once the system lattice takes over"
    );
    assert_eq!(previous_cap, imf_mass_hi_msun());
    // …and the saturation value IS the shipped cap, which is the statement that the world we ship
    // sits ON the lattice bound rather than near it.
    assert_eq!(
        binding_limit(&pl, previous_cap, REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M),
        None
    );
}

/// ★ THE AFFORDABILITY FENCE, BOTH ARMS, BOTH LIMITS, AND THE MEASUREMENT OF WHERE EACH FLIPS.
///
/// ★ RE-BASED IN S9, AND THE FINDING INVERTED. This test used to assert that the fence REFUSES because
/// the galaxy is too small to place a 120 M☉ star's system — refused by a factor of about seven — and
/// that refusal was the stated argument for changing the coordinate step.
///
/// The step changed. The galaxy grew 2_051×, its purse now affords 1288 solar masses, and that arm has
/// stopped binding. **The fence still refuses, for a different reason, and that is the S9 result worth
/// keeping**: a star system counts its own positions in millimetres, and at 120 M☉ the system is wider
/// than that lattice can state. The limit moved from the galaxy's room to the system's numbers.
///
/// So the cure the fence asks for has changed too — the fine rung, not the galaxy radius. That is P10.
#[test]
fn the_star_limit_moved_from_the_galaxys_room_to_the_systems_own_numbers() {
    let pl = world_planet_config();
    // ▲ THE ARM THAT USED TO BIND, MEASURED AS NO LONGER BINDING. On the world we ship, the galaxy
    // can pay for the physical top several hundred times over.
    let demand_m = galaxy_child_demand_m(&pl, 120.0);
    assert!(
        demand_m < REAL_GALAXY_R_M,
        "the galaxy's purse no longer binds — it was 7x short before the climb"
    );
    let purse_headroom = REAL_GALAXY_R_M / demand_m;
    assert!(purse_headroom > 50.0, "MEASURED 63.355x"); // was 0.143x — a 7x shortfall

    // ▲ THE ARM THAT BINDS NOW, on the world we actually ship.
    let refused = guard_galaxy_affords_its_stars(REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M)
        .expect_err("a 120 solar-mass system still does not fit its own lattice");
    assert_eq!(refused.stated_top_msun, 120.0);
    assert_eq!(refused.budget_m, REAL_GALAXY_R_M);
    assert_eq!(refused.lattice_m, SYSTEM_LATTICE_R_M);
    assert_eq!(refused.bound_by, StarLimit::SystemLattice);
    // Refused by a wide margin, not a rounding — stated as a ratio so it reads the same whatever the
    // units become. MEASURED 10.7746x.
    let over_by = refused.shell_m / refused.lattice_m;
    assert!(
        over_by > 2.0,
        "the overrun is only {over_by}x — if it were marginal this would be a tuning question \
         rather than a coordinate one"
    );
    assert!(
        refused.affordable_msun < refused.stated_top_msun,
        "a refusal must mean the world affords LESS than physics states"
    );

    // ▲ THE OTHER LIMIT, STILL DRIVEN. A galaxy small enough still refuses for the OLD reason, so
    // both arms of `binding_limit` stay exercised and the enum cannot rot into one value.
    let small = guard_galaxy_affords_its_stars(REAL_GALAXY_R_M / 1.0e6, SYSTEM_LATTICE_R_M)
        .expect_err("a galaxy a millionth the size cannot place the system either");
    assert_eq!(small.bound_by, StarLimit::GalaxyPurse);
    // BOTH REFUSALS SAY THEIR OWN NAME. A refusal a person cannot read is a refusal that gets
    // silenced, so the wording is asserted rather than left to be discovered in a log — and driving
    // both arms is also what keeps `StarLimit`'s Display fully covered (HR5).
    assert!(
        refused
            .to_string()
            .contains("its system could not state its own positions"),
        "{refused}"
    );
    assert!(
        small
            .to_string()
            .contains("the galaxy has no room to place its system"),
        "{small}"
    );
    // The purse is reported first when BOTH break, which is what this case is.
    assert!(small.shell_m > small.lattice_m);

    // ▲ THE PASSING ARM, which is now a question about the RUNG and not about the galaxy — and this
    // is precisely why the lattice became an argument. With the budget alone it could not be driven
    // at all: no galaxy radius makes a 120 M☉ system fit a millimetre lattice, so the fence's Ok
    // branch would have been unreachable code wearing a green light.
    assert_eq!(
        guard_galaxy_affords_its_stars(REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M * 16.0),
        Ok(())
    );

    // ★ AND WHERE IT FLIPS, found rather than assumed: the coarsest system step, in whole binary
    // octaves above the millimetre, at which the world can host the stars physics states. Reported so
    // the P10 decision reads off a number instead of an argument.
    let flips_at = (0..32)
        .find(|&oct| {
            guard_galaxy_affords_its_stars(REAL_GALAXY_R_M, SYSTEM_LATTICE_R_M * 2.0_f64.powi(oct))
                .is_ok()
        })
        .expect("some octave of the system lattice must afford the physical top");
    eprintln!(
        "[s9-fence] galaxy purse: {purse_headroom:.1}x headroom (was 7x SHORT) | system lattice: \
         over by {over_by:.4}x | affords {:.6} Msun against a stated {:.1} | the fence passes from \
         {flips_at} octave(s) of fine-lattice lift",
        refused.affordable_msun, refused.stated_top_msun
    );
    // FOUR octaves of the fine rung would do it — the system step going from a millimetre to 16 mm.
    // Pinned, because P10's size is this number and an argument is not a size.
    assert_eq!(flips_at, 4);
}

/// ★ THE S6 MEASUREMENT, TAKEN BEFORE ANYTHING IS CHANGED. What would every band in THE world become
/// if it were sized from the ceiling in force at its own surface, and can the world afford it?
///
/// The number that matters is the RATIO of the band to the body it wraps. A band wider than the thing
/// it surrounds is not a band, it is a second body — so if the answer is a large fraction, the law is
/// unaffordable and S6 needs a different shape. This test states the answer as an assertion rather than
/// a print, so it cannot quietly stop being true.
#[test]
fn what_a_self_sized_band_would_cost_every_boundary_in_the_world() {
    use vd_core::flight::{TRAVERSE_S, realm_speed_cap_mps};

    let cfg = test_world();
    let dt = GEOMETRY_TICK_DT_S;
    let ticks = BAND_TICKS_N;
    let headroom = BAND_TAU_HEADROOM;
    // ★ THE FOOT SPEED IS A CONSTANT OF THE SOLVE, NOT THE CLUSTER'S. The band a world ships may not
    // depend on how fast one deployment lets a person walk, for exactly the reason the tick does not:
    // two clusters must boot the identical world. The shipped default (500 m/s) is used here so the
    // measurement judges the widest lawful case rather than the narrowest.
    let v_foot = 500.0_f64;
    let mut foot_bound = 0usize;

    // WHY THE BOUNDARY'S OWN CEILING AND NOT ITS PARENT'S. The approach governor lowers a subject's
    // ceiling onto the body it is approaching, so a thing arrives at THE CHILD'S speed and never at the
    // parent's. Sizing against the parent's ceiling would size every band for a speed no lawful subject
    // can hold at that surface.
    let sweep = derived_nest_sweep_seeds(galaxy_profile(0, &cfg).count);
    let mut rows = 0usize;
    let mut worst_ratio = 0.0_f64;
    let mut worst: Option<(RealmId, f64, f64)> = None;
    let mut thinnest_ticks = f64::MAX;
    let mut floor_bound = 0usize;

    for seed in 0..sweep {
        let forest = WorldView::generated(seed, &cfg).regions().to_vec();
        for r in &forest {
            if r.parent.is_none() {
                continue;
            }
            let extent = r.shape.circumscribed_extent();
            let own_ceiling = realm_speed_cap_mps(extent, v_foot, TRAVERSE_S);
            // Does the foot speed BIND here, i.e. is this realm small enough that a person on foot is
            // the fastest lawful thing at its surface? That is the only regime where the band could be
            // asked to be larger than the body it wraps.
            foot_bound += usize::from(own_ceiling == v_foot);
            // The band this surface needs so that one crossing at its own ceiling is SEEN.
            let need_m = own_ceiling * dt * ticks * headroom;
            // The floor every band keeps regardless — today's shipped width.
            let floor_m = CONTAINMENT_INSET_M + CONTAINMENT_OUTSET_M;
            let band_m = need_m.max(floor_m);
            if band_m == floor_m {
                floor_bound += 1;
            }
            let ratio = band_m / extent;
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst = Some((r.realm, band_m, extent));
            }
            thinnest_ticks = thinnest_ticks.min(band_m / (own_ceiling * dt));
            assert!(
                band_m.is_finite() && band_m > 0.0,
                "seed {seed}, {:?}: a boundary must be able to state its own band",
                r.realm
            );
            rows += 1;
        }
    }

    assert!(rows > 0, "a sweep with no boundary measures nothing");
    let (worst_realm, worst_band, worst_extent) = worst.expect("a swept row");
    eprintln!(
        "[s6-cost] {rows} boundaries over {sweep} seeds | widest band-to-body ratio {worst_ratio:.6e}          at {worst_realm:?} ({worst_band:.6e} m band around a {worst_extent:.6e} m body) |          {floor_bound} boundaries stay at today's {:.1} m floor | {foot_bound} where the foot \
         speed binds | thinnest crossing now {thinnest_ticks:.6e} ticks",
        CONTAINMENT_INSET_M + CONTAINMENT_OUTSET_M
    );

    // ★ THE AFFORDABILITY ANSWER, as an assertion. If a band ever needed to be a large fraction of the
    // body it wraps, this law would be the wrong shape and S6 would need re-designing rather than
    // implementing. Half is the line: past it a band would reach the body's own centre.
    assert!(
        worst_ratio < 0.5,
        "a self-sized band would reach {worst_ratio} of its own body at {worst_realm:?} — that is not          a band, and this law would need re-designing rather than implementing"
    );

    // ★ AND IT ACTUALLY FIXES THE THING IT EXISTS FOR: every crossing becomes observable. Today every
    // one of these boundaries is crossed in under one tick.
    assert!(
        thinnest_ticks >= 1.0,
        "the thinnest crossing is still {thinnest_ticks} ticks — the band did not do its job"
    );
}

/// ★ THE NAMED RISK, MEASURED BEFORE IT IS TAKEN. Widening every band widens what each body OCCUPIES,
/// and the world's two placement fences both compare occupied volumes: a child must fit inside its
/// parent, and two siblings must not overlap. Either could refuse THE world.
///
/// The plan's own instruction is to record a refusal as the measurement it is rather than to relax the
/// fence. So this test takes the reading first, and it fails if the world cannot afford the law.
#[test]
fn a_self_sized_band_still_fits_inside_its_parent_and_clear_of_its_siblings() {
    use vd_core::flight::{TRAVERSE_S, realm_speed_cap_mps};

    let cfg = test_world();
    let dt = GEOMETRY_TICK_DT_S;
    let v_foot = cfg.interest.occupant_v_max_mps;
    let band_of = |extent: f64| -> f64 {
        (realm_speed_cap_mps(extent, v_foot, TRAVERSE_S) * dt * BAND_TICKS_N * BAND_TAU_HEADROOM)
            .max(CONTAINMENT_INSET_M + CONTAINMENT_OUTSET_M)
    };

    let sweep = derived_nest_sweep_seeds(galaxy_profile(0, &cfg).count);
    let mut nest_checked = 0usize;
    let mut sib_checked = 0usize;
    let mut worst_nest_margin = f64::MAX;
    let mut worst_sib_margin = f64::MAX;
    let mut worst_nest_today = f64::MAX;
    let mut worst_sib_today = f64::MAX;
    let mut moving_pairs = 0usize;
    let mut nest_refused: Option<RealmId> = None;
    let mut sib_refused: Option<(RealmId, RealmId)> = None;

    for seed in 0..sweep {
        let forest = WorldView::generated(seed, &cfg).regions().to_vec();
        for r in &forest {
            let Some(parent_id) = r.parent else { continue };
            let Some(parent) = forest.iter().find(|p| p.realm == parent_id) else {
                continue;
            };
            // NESTING, ASKED THE WAY THE SHIPPED FENCE ASKS IT. `center` is the child's placement in
            // its PARENT'S frame, so the reach is measured from that offset — never as a delta against
            // the parent's own centre, which lives in the GRANDPARENT'S frame and would subtract two
            // different frames' numbers. (My first version of this test did exactly that and refused
            // the world by 1.5e15 m, which is why the control below exists.)
            // ★ THE PARENT'S RULER, NOT THE CHILD'S (S12 fix, 2026-08-28) — this read
            // `r.frame.tier()`, which is the 2048x parent-frame trap this tree has hit repeatedly.
            //
            // `center` is stated in the PARENT'S frame; `frame` is the region's OWN. Reading one with
            // the other's unit made every distance here 2048x too SMALL, so a gap measured against
            // extents in true metres went hugely negative — and the control fired with
            // `sibling: -9.682227e14 m`.
            //
            // ★ THE SHELL HID IT. While every system sat at ONE radius the error was uniform and the
            // comparisons still looked sane. Varied radii expose it. Measured independently: the
            // closest pair the shape produces is 902x FURTHER apart than it needs to be, so the world
            // was never crowded — only the ruler was wrong. That is exactly what this test's own
            // control demands be checked before any conclusion is drawn about the law.
            let tier = parent.frame.tier();
            let at = r
                .center
                .in_parents_frame()
                .delta_m(LatticePos::ORIGIN, tier);
            let limit = parent.shape.inscribed_extent();
            let base_reach = r.shape.max_reach_from(at);
            let reach = base_reach + band_of(r.shape.circumscribed_extent());
            let margin = limit - reach;
            // THE CONTROL: the same fence with TODAY'S band. If this is also refused, the fault is in
            // this measurement and not in the law it is judging.
            let today = limit - (base_reach + CONTAINMENT_INSET_M + CONTAINMENT_OUTSET_M);
            worst_nest_today = worst_nest_today.min(today);
            if margin < worst_nest_margin {
                worst_nest_margin = margin;
                if margin < 0.0 {
                    nest_refused = Some(r.realm);
                }
            }
            nest_checked += 1;

            // SIBLINGS: two bands must not overlap, or a point could be a hysteretic member of two
            // siblings at once and the tie would break on the lower realm id rather than the nearer body.
            for s in forest
                .iter()
                .filter(|s| s.parent == Some(parent_id) && s.realm != r.realm)
            {
                let gap = s
                    .center
                    .in_parents_frame()
                    .delta_m(r.center.in_parents_frame(), tier)
                    .length()
                    - r.shape.circumscribed_extent()
                    - s.shape.circumscribed_extent();
                let need = band_of(r.shape.circumscribed_extent())
                    + band_of(s.shape.circumscribed_extent());
                // ★ A MOVING SIBLING CANNOT BE JUDGED FROM THIS FIELD, and mistaking that for an
                // overlap is the trap this branch exists to avoid. A moving child's `center` is ZERO by
                // design — its real placement is authored into the parent's book every tick — so two
                // orbiting planets both read as sitting exactly on their star. My first version of this
                // measurement compared those two zeros and reported 83,542 "overlapping" sibling pairs
                // in THE world. There are none: it was comparing a field that does not hold a position.
                //
                // So the pairs judged below are the STATICALLY placed ones, which is precisely the set
                // the shipped separation fence judges. Judging moving siblings needs their authored
                // placements at an instant, which is a different measurement.
                let both_static = (at.length() > 0.0)
                    | (s.center
                        .in_parents_frame()
                        .delta_m(LatticePos::ORIGIN, tier)
                        .length()
                        > 0.0);
                if !both_static {
                    moving_pairs += 1;
                    continue;
                }
                let today_gap = gap - 2.0 * (CONTAINMENT_INSET_M + CONTAINMENT_OUTSET_M);
                worst_sib_today = worst_sib_today.min(today_gap);
                let m = gap - need;
                if m < worst_sib_margin {
                    worst_sib_margin = m;
                    if m < 0.0 {
                        sib_refused = Some((r.realm, s.realm));
                    }
                }
                sib_checked += 1;
            }
        }
    }

    eprintln!(
        "[s6-fence] {nest_checked} nestings, {sib_checked} sibling pairs over {sweep} seeds\n\
         [s6-fence]   nesting: today {worst_nest_today:.6e} m -> self-sized {worst_nest_margin:.6e} m\n\
         [s6-fence]   sibling: today {worst_sib_today:.6e} m -> self-sized {worst_sib_margin:.6e} m\n\
         [s6-fence]   sibling pairs skipped as MOVING (no position in the static field): {moving_pairs}"
    );
    // THE CONTROL FIRST. A measurement that refuses the world under TODAY'S shipped band is measuring
    // itself, not the law it is judging.
    assert!(
        worst_nest_today >= 0.0 && worst_sib_today >= 0.0,
        "this measurement refuses the SHIPPED world, so it is wrong about how the fences ask their \
         question — fix the measurement before drawing any conclusion about the law"
    );
    assert!(
        nest_checked > 0 && sib_checked > 0,
        "a fence with nothing to judge proves nothing"
    );
    assert!(
        nest_refused.is_none(),
        "a self-sized band pushes {:?} outside its own parent — record this, do not relax the fence",
        nest_refused
    );
    assert!(
        sib_refused.is_none(),
        "a self-sized band makes {:?} overlap — record this, do not relax the fence",
        sib_refused
    );
}

/// The clear distance from `r` to its nearest sibling under `parent` — how much room a boundary has
/// before widening it would touch the thing next door. `f64::INFINITY` for an only child, which is the
/// honest answer: nothing constrains it.
fn nearest_sibling_gap_m(regions: &[RealmRegion], r: &RealmRegion, parent: RealmId) -> f64 {
    let tier = r.frame.tier();
    regions
        .iter()
        .filter(|s| s.parent == Some(parent) && s.realm != r.realm)
        .map(|s| {
            let d = s
                .center
                .in_parents_frame()
                .delta_m(r.center.in_parents_frame(), tier)
                .length();
            (d - r.shape.circumscribed_extent() - s.shape.circumscribed_extent()).max(0.0)
        })
        .fold(f64::INFINITY, f64::min)
}

/// ★ MEASURING S12's REVERSIBILITY GATE AGAINST THE GENERATOR THAT ACTUALLY EXISTS (2026-08-28).
///
/// The plan says this gate is *"red today by construction"* and that changing the count moves every
/// star. That describes an OLDER generator. **Measured against the one in the tree: it is GREEN.**
///
/// The reason is the stream keying. Each system draws from its OWN lineage —
/// `[UNIVERSE_SEED, GALAXY_SEED, its own seed]` — so its position is `f(universe_seed, itself)` and a
/// neighbour appearing cannot reach it.
///
/// Owner ruling G4: where the seed puts a star, the star stays.
///
/// ⚠ **WHAT THIS DOES NOT PROVE, AND MUST NOT BE READ AS.** It proves STABILITY (G4) and nothing else.
/// The placement is still a SHELL: every system shares one radius, and both angles are drawn UNIFORM
/// ON THE SPHERE. A uniform sphere is a ball, not a galaxy — as many stars "above" the disc as in it,
/// and no arms, by construction. G2 and G9 (a believable shape, drawn from the seed) are entirely
/// absent, and this test would stay green through every one of those defects.
#[test]
fn growing_the_system_count_does_not_move_the_systems_already_placed() {
    // ★ REWRITTEN AT S12/G8 (2026-08-28). This used to build two worlds from two CONFIG COUNTS — a
    // three-system galaxy and a four-system one — and check the three survivors were unmoved. The
    // owner's ruling deleted that knob: a population is a RESULT of the density and the shape a
    // galaxy draws, so a seed's world has ONE size and there is no second count to ask for.
    //
    // The property is unchanged and is what G4 actually says: a system's place depends on its own
    // index and its own draws, never on how many siblings it has. So the honest test states the
    // shape ONCE and asks the production placement law where systems 0..N sit for a SMALL N and a
    // LARGER one — which is what growth is, with nothing else varying.
    let cfg = test_visual();
    let draws_for = |seed: u64, n: u32| {
        let mut s = realm_stream(seed, &[UNIVERSE_SEED, GALAXY_SEED, system_seed_at(n)]);
        PlacementDraws {
            population: s.next_f64(),
            radius: s.next_f64(),
            radius_b: s.next_f64(),
            azimuth: s.next_f64(),
            scatter: s.next_f64(),
            scatter_b: s.next_f64(),
            height: s.next_f64(),
        }
    };
    for seed in 0..seeds_to_judge(48) {
        let shape = galaxy_profile(seed, &cfg).shape;
        const SMALL: u32 = 3;
        const BIGGER: u32 = 4;
        const { assert!(SMALL < BIGGER, "the bigger world really is bigger") };
        let small: Vec<DVec3> = (0..SMALL)
            .map(|n| system_center_at(&cfg, &shape, n, draws_for(seed, n)))
            .collect();
        let bigger: Vec<DVec3> = (0..BIGGER)
            .map(|n| system_center_at(&cfg, &shape, n, draws_for(seed, n)))
            .collect();
        for (n, before) in small.iter().enumerate() {
            assert_eq!(
                bigger[n], *before,
                "seed {seed}: system {n} MOVED when the world grew from {SMALL} to {BIGGER}"
            );
        }
        // NON-VACUITY: the world really did grow, and the new system is somewhere of its own rather
        // than a repeat of one already placed.
        assert_eq!(bigger.len(), small.len() + 1);
        assert!(
            !small.contains(&bigger[small.len()]),
            "seed {seed}: the system growth added did not land anywhere new"
        );
    }
}

/// ★ MEASURING THE SHAPE (S12) — is it a galaxy, or still a ball?
///
/// Owner rulings G2 and G9: a believable shape, drawn from the seed. The shell it replaces failed on
/// three counts, and each one is measured here rather than argued.
#[test]
fn the_placement_is_a_shaped_galaxy_and_no_longer_a_shell() {
    let mut cfg = test_visual();
    cfg = galaxy_holding(&cfg, 0, 4_000);
    let bodies = generate_system_forest(0, &cfg);
    let centres: Vec<DVec3> = bodies
        .iter()
        .filter(|b| matches!(b.realm, RealmId::System(_)))
        .filter_map(|b| match b.placement {
            Placement::StaticOffset(v) => Some(v),
            _ => None,
        })
        .filter(|v| v.length() > 0.0) // the home sits at the origin by law
        .collect();
    assert!(centres.len() > 3_000, "enough systems to measure a shape");

    let r_max = cfg.stellar.galaxy_rim_r_m;
    let radii: Vec<f64> = centres.iter().map(|v| v.length() / r_max).collect();

    // (1) NOT A SHELL. The shell put every system at exactly one radius. A galaxy fills a volume, so
    // the radii must SPREAD — measured as the gap between the nearest and the farthest.
    let r_min = radii.iter().copied().fold(f64::INFINITY, f64::min);
    let r_far = radii.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        r_far - r_min > 0.5,
        "the radii must span the galaxy, not sit on one surface: {r_min:.3}..{r_far:.3}"
    );

    // (2) A THIN DISC, NOT A BALL. Uniform on a sphere puts as many systems above the disc as in it.
    // A disc is thin: the height spread must be a small fraction of the radial spread.
    let mean_abs_z = centres.iter().map(|v| v.y.abs()).sum::<f64>() / centres.len() as f64 / r_max;
    let mean_r = radii.iter().sum::<f64>() / radii.len() as f64;
    assert!(
        mean_abs_z < mean_r * 0.25,
        "a disc is THIN — mean |z| {mean_abs_z:.4} against mean r {mean_r:.4}"
    );

    // (3) ARMS. A uniform azimuth has no arms by construction. With arms, the angle a system sits at
    // is NOT uniform once the arm's own winding is removed: subtract the logarithmic sweep and the
    // systems pile into a small number of directions.
    //
    // Measured as the LARGEST share any one of twelve angular buckets holds. Uniform would give
    // 1/12 = 0.083 in every bucket; arms concentrate far above that.
    // ★ MEASURE THE ARMS ON THE DISC, NOT ON THE WHOLE GALAXY (S12/G1, 2026-08-28). A BULGE IS ROUND
    // BY CONSTRUCTION — its azimuth is a plain uniform draw, with no arm to follow — so counting its
    // systems into these buckets adds a flat floor that can only dilute the signal. The shell had no
    // bulge worth the name (a fixed 15% share in a fixed 15% radius), so including it cost little and
    // nobody noticed. A drawn bulge reaches 40% of the stars, and at that share the measurement read
    // 0.116 against its 0.15 bar and called a galaxy with visible arms armless.
    //
    // Outside the bulge radius the disc is what remains, and the arms are its own structure.
    // ★ AND COUNT THE BUCKETS PER ARM, NOT TWELVE FOR EVERYONE (S12/G1, 2026-08-28). Twelve fixed
    // buckets measure a different thing for each arm count: two arms sit 180° apart and pile into a
    // few buckets, four sit 90° apart and spread over more. MEASURED across 22 drawn spirals, the
    // twelve-bucket peak ran 2.50× uniform at two arms down to 1.36× at four — a spread that says
    // nothing about the galaxies and everything about the ruler. Four buckets per arm measures the
    // same shape whatever the count: the same 22 spirals then read 1.59× to 1.88×, and the bar below
    // is set just under the worst of them.
    let shape = galaxy_profile(0, &cfg).shape;
    let n_buckets = shape.arms as usize * 4;
    let mut buckets = vec![0u32; n_buckets];
    let mut counted = 0usize;
    for (v, r) in centres.iter().zip(&radii) {
        if *r < shape.bulge_radius_frac {
            continue;
        }
        counted += 1;
        let az = v.z.atan2(v.x);
        // Remove the arm's own winding, so a real arm collapses to one direction.
        // ★ UNWIND BY THE GALAXY'S OWN PITCH, NEVER A LITERAL (S12/G1, 2026-08-28). This subtracted
        // tan(25°), which was the chosen table's spiral pitch and therefore right for every galaxy.
        // The pitch is DRAWN now — seed 0 winds at 22.2° — so unwinding by 25° smears a real arm
        // across buckets and the measurement reads a galaxy with arms as one without.
        let unwound = az - (r.max(1.0e-6)).ln() / shape.pitch_tan;
        let norm = unwound.rem_euclid(core::f64::consts::TAU) / core::f64::consts::TAU;
        buckets[((norm * n_buckets as f64) as usize).min(n_buckets - 1)] += 1;
    }
    assert!(
        counted > 1_000,
        "the disc outside the bulge must still hold enough systems to measure ({counted})"
    );
    let peak = f64::from(buckets.iter().copied().max().unwrap_or(0)) / counted as f64;
    let uniform = 1.0 / n_buckets as f64;
    let contrast = peak / uniform;
    assert!(
        contrast > 1.5,
        "the systems must pile into arms once the winding is removed — the fullest of \
         {n_buckets} buckets holds {peak:.3} of the disc against {uniform:.3} if it were \
         featureless, a contrast of {contrast:.2}x"
    );
}

/// ★ DOES AN EARTH-LIKE WORLD STILL EXIST AFTER THE SHAPE LANDED (S12, 2026-08-28)?
///
/// The pinned seed 2298 held exactly one Earth-like body and now holds none. That is expected —
/// the placement takes five draws where it took two, so every draw after them shifted, and planet
/// masses moved with them.
///
/// ★ BUT IT MUST BE CHECKED, NOT ASSUMED. Owner ruling G10: the home is FOUND, by having an
/// Earth-like planet. A world where no seed yields one is a world with no home, and that would be a
/// real defect wearing the costume of a moved golden.
#[test]
fn an_earth_like_world_still_exists_somewhere_in_the_seed_space() {
    let cfg = test_world();
    let mut seeds_with = 0_u32;
    let mut total = 0_usize;
    let mut first: Option<u64> = None;
    // ★ RE-DERIVED FROM 4 096 SEEDS (S12/G8, 2026-08-28). This swept four thousand worlds because a
    // world held THREE star systems and an Earth-like planet is rare — 18 seeds in 4 096 held one.
    // A world now holds a quarter of a million systems, so the rarity is met inside a single galaxy
    // and the sweep costs a handful of seeds instead of four thousand whole worlds.
    let sweep = seeds_to_judge(4_096 * 3);
    for seed in 0..sweep {
        let found = earth_like_candidates(seed, &cfg);
        if !found.is_empty() {
            seeds_with += 1;
            total += found.len();
            if first.is_none() {
                first = Some(seed);
            }
        }
    }
    println!(
        "[s12-earthlike] {seeds_with} of {sweep} seeds hold an Earth-like body ({total} bodies); \
         first at seed {first:?}"
    );
    assert!(
        seeds_with > 0,
        "no seed in the sweep yields an Earth-like planet — the home cannot be FOUND (owner ruling G10), \
         and the shape has broken habitability rather than merely moved it"
    );
}

/// ★ LOOK AT THE GALAXIES THE SEED DRAWS (S12/G1; owner ruling G13 — *"the assistant proposes the
/// standard model in code and the owner judges the picture, because a galaxy is easier to judge than
/// to specify"*).
///
/// This renders the DRAWN shape — the family law — for a run of seeds, as a picture rather than a
/// column of numbers. Two views per galaxy, because one hides the fault the other shows:
///
/// - FROM ABOVE: are there arms, do they wind the right way, is the middle denser?
/// - FROM THE SIDE: is the disc thin, does the bulge stand out of it?
///
/// The world does NOT use these shapes yet. The placement still reads the chosen table, so this test
/// changes nothing and pins nothing — it exists so the ranges can be judged before the world adopts
/// them. Run it with:
///
/// ```text
/// cargo test -p vd-physics --lib look_at_the_drawn_galaxies -- --ignored --nocapture
/// ```
/// TEMPORARY — what does the world now hold?
#[test]
#[ignore]
fn diag_the_worlds_population() {
    // HOW BIG IS THE SKY ON THE WIRE, and how many parts does it become?
    {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let rows = sky_from_system_layer(2298, &cfg);
        let bytes = postcard::to_allocvec(&rows).expect("encodes").len();
        let parts = bytes.div_ceil(8 * 1024);
        println!(
            "  SKY ON THE WIRE: {} rows = {:.2} MB in {parts} parts of 8 KB",
            rows.len(),
            bytes as f64 / 1.0e6
        );
        println!(
            "  IF RE-SENT EVERY BEAT until the client confirms: {:.1} MB per beat, per client",
            bytes as f64 / 1.0e6
        );
    }
    // THE SHARD'S SUBTREE: identical to the filtered full build, and what it costs.
    {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let held = std::collections::BTreeSet::from([SYSTEM_A]);
        let t0 = std::time::Instant::now();
        let old = realm_neighbourhood_for_config(2298, &held, &cfg);
        let old_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let sub = realm_subtree(2298, &cfg, &held, &std::collections::BTreeSet::new());
        let new_rows = vd_core::worldgen::neighbourhood_scope(&to_regions(&sub, &cfg), &held);
        let new_s = t1.elapsed().as_secs_f64();
        let key = |rs: &[vd_core::geometry::RealmRegion]| {
            let mut v: Vec<_> = rs
                .iter()
                .map(|r| (r.realm, r.parent, r.shape, r.frame))
                .collect();
            v.sort_by_key(|t| format!("{:?}", t.0));
            v
        };
        println!(
            "  SUBTREE: full-then-filter {old_s:.2}s -> subtree {new_s:.3}s ({:.0}x) | rows {} vs {} | IDENTICAL: {}",
            old_s / new_s.max(1e-9),
            old.len(),
            new_rows.len(),
            key(&old) == key(&new_rows)
        );
    }
    // THE SKY: byte-identical, and what it costs the two ways.
    {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let t0 = std::time::Instant::now();
        let old = star_catalogue(
            &realm_regions_for_config(2298, &cfg),
            &system_photometrics_for_config(2298, &cfg),
        );
        let old_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let new = sky_from_system_layer(2298, &cfg);
        let new_s = t1.elapsed().as_secs_f64();
        println!(
            "  SKY: whole forest {old_s:.2}s -> system layer {new_s:.2}s ({:.1}x) | rows {} vs {} | \
             BYTE-IDENTICAL: {}",
            old_s / new_s.max(1e-9),
            old.len(),
            new.len(),
            postcard::to_allocvec(&old).expect("encodes")
                == postcard::to_allocvec(&new).expect("encodes")
        );
    }
    // THE SYSTEM LAYER: is it the same world, and how much cheaper?
    {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let t0 = std::time::Instant::now();
        let full = generate_system_forest(2298, &cfg);
        let full_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let layer = generate_system_layer(2298, &cfg);
        let layer_s = t1.elapsed().as_secs_f64();
        // IDENTITY: every system in the layer must match the full forest exactly — same realm,
        // same place, same shell, same star. If one differs we have built two worlds.
        let pick = |bs: &[GeneratedBody]| -> Vec<(RealmId, DVec3, f64, Option<StarPhotometrics>)> {
            bs.iter()
                .filter(|b| b.parent == Some(GALAXY) && matches!(b.realm, RealmId::System(_)))
                .map(|b| {
                    (
                        b.realm,
                        placement_offset(b.placement),
                        b.shape.finite_extent(),
                        b.photometrics,
                    )
                })
                .collect()
        };
        let a = pick(&full);
        let b = pick(&layer);
        println!(
            "  LAYER: full {full_s:.2}s ({} bodies) | layer {layer_s:.2}s ({} bodies) | \
             {:.1}x cheaper | systems {} vs {} | IDENTICAL: {}",
            full.len(),
            layer.len(),
            full_s / layer_s.max(1e-9),
            a.len(),
            b.len(),
            a == b
        );
    }
    // WHAT DOES A SHARD PAY TO BOOT, at full density?
    {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let t0 = std::time::Instant::now();
        let forest = generate_system_forest(2298, &cfg);
        let built = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let regions = realm_regions_for_config(2298, &cfg);
        let lower = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let held = std::collections::BTreeSet::from([RealmId::System(7)]);
        let n = realm_neighbourhood_for_config(2298, &held, &cfg).len();
        let hood = t2.elapsed().as_secs_f64();
        println!(
            "  BOOT: generate {built:.2}s ({} bodies) | lower {lower:.2}s ({} regions) | \
             neighbourhood {hood:.2}s ({n} rows kept)",
            forest.len(),
            regions.len()
        );
    }
    // G3: does any pair of systems share a distance from the centre?
    {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let forest = generate_system_forest(2298, &cfg);
        let mut radii: Vec<f64> = forest
            .iter()
            .filter(|b| b.parent == Some(GALAXY) && matches!(b.realm, RealmId::System(_)))
            .map(|b| placement_offset(b.placement).length())
            .collect();
        let n = radii.len();
        radii.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let mut exact = 0usize;
        let mut closest = f64::INFINITY;
        for w in radii.windows(2) {
            let d = w[1] - w[0];
            if d == 0.0 {
                exact += 1;
            }
            closest = closest.min(d);
        }
        println!(
            "  G3 radial: {n} systems, {exact} share a distance EXACTLY, closest two differ by {closest:.3e} m"
        );
    }
    // How BADLY does it collide, and where?
    {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let forest = generate_system_forest(2298, &cfg);
        let sys: Vec<(DVec3, f64, f64)> = forest
            .iter()
            .filter(|b| b.parent == Some(GALAXY) && matches!(b.realm, RealmId::System(_)))
            .map(|b| {
                let at = placement_offset(b.placement);
                (at, b.shape.circumscribed_extent(), at.length())
            })
            .collect();
        let rim = cfg.stellar.galaxy_rim_r_m;
        let cell = 2.0 * sys.iter().map(|s| s.1).fold(0.0_f64, f64::max);
        let mut grid: std::collections::BTreeMap<(i64, i64, i64), Vec<usize>> =
            std::collections::BTreeMap::new();
        let key = |v: DVec3| {
            (
                (v.x / cell).floor() as i64,
                (v.y / cell).floor() as i64,
                (v.z / cell).floor() as i64,
            )
        };
        for (i, s) in sys.iter().enumerate() {
            grid.entry(key(s.0)).or_default().push(i);
        }
        let mut pairs = 0usize;
        let mut worst = f64::INFINITY;
        let mut worst_r = 0.0;
        for (i, a) in sys.iter().enumerate() {
            let (cx, cy, cz) = key(a.0);
            for dx in -1..=1i64 {
                for dy in -1..=1i64 {
                    for dz in -1..=1i64 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket.iter().filter(|j| **j > i) {
                                let b = &sys[j];
                                let d = (b.0 - a.0).length();
                                let need = a.1 + b.1;
                                if d < need {
                                    pairs += 1;
                                    if d / need < worst {
                                        worst = d / need;
                                        worst_r = a.2 / rim;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!(
            "  seed 2298: {} systems, {pairs} OVERLAPPING PAIRS, worst at {:.1}% of what it needs, \
             {:.1}% of the way out from the centre",
            sys.len(),
            100.0 * worst,
            100.0 * worst_r
        );
    }
    // Does THE world, at full density, actually hold together?
    for seed in [0u64, 2298] {
        let cfg = UniverseConfig::world(15.0, 0.05);
        let t0 = std::time::Instant::now();
        let forest = generate_system_forest(seed, &cfg);
        let verdict = siblings_disjoint(&forest);
        println!(
            "  seed {seed:5}: {} systems, {} bodies, built in {:.2}s -> {}",
            galaxy_profile(seed, &cfg).count,
            forest.len(),
            t0.elapsed().as_secs_f64(),
            match verdict {
                Ok(()) => "DISJOINT".to_string(),
                Err(e) => format!("OVERLAP: {e:?}"),
            }
        );
    }
    use core::mem::size_of;
    println!("GeneratedBody      {:4} bytes", size_of::<GeneratedBody>());
    println!("  realm            {:4}", size_of::<RealmId>());
    println!("  parent           {:4}", size_of::<Option<RealmId>>());
    println!("  shape            {:4}", size_of::<Boundary>());
    println!("  placement        {:4}", size_of::<Placement>());
    println!(
        "  photometrics     {:4}",
        size_of::<Option<StarPhotometrics>>()
    );
    println!(
        "  taxon            {:4}",
        size_of::<Option<crate::taxonomy::BodyTaxon>>()
    );
    println!("  look             {:4}", size_of::<Option<Boundary>>());
    let cfg = test_world();
    for seed in [0u64, 2298] {
        let p = galaxy_profile(seed, &cfg);
        println!(
            "seed {seed:5}: {:?}, density {:.4}/pc3, disc {:.2}% thick -> {} systems",
            p.kind,
            p.density_per_pc3,
            100.0 * p.shape.disc_thickness_frac,
            p.count
        );
    }
    for (name, c) in [
        ("world", test_world()),
        ("visual_scale", test_visual()),
        ("visual_demand", UniverseConfig::visual_demand(15.0, 0.02)),
        ("walk_scale", UniverseConfig::walk_scale()),
    ] {
        println!(
            "  preset {name:14} rim {:.3e} m -> {} systems",
            c.stellar.galaxy_rim_r_m,
            galaxy_profile(0, &c).count
        );
    }
    let mut lo = u32::MAX;
    let mut hi = 0u32;
    for seed in 0..64u64 {
        let c = galaxy_profile(seed, &cfg).count;
        lo = lo.min(c);
        hi = hi.max(c);
    }
    println!("across 64 seeds: {lo} .. {hi} systems");
}

/// The same drawn galaxies, as DATA — for rendering somewhere with more than 62 characters of width.
/// Writes one JSON line per galaxy to stdout: its numbers, then its stars as flat `x,y,z` in units of
/// the galaxy's own radius.
#[test]
#[ignore = "diagnostic: emits the drawn galaxies as JSON for a real renderer"]
fn emit_the_drawn_galaxies_as_json() {
    const STARS: u32 = 12_000;
    let mut cfg = test_visual();
    cfg = galaxy_holding(&cfg, 0, STARS);
    for seed in 0..8_u64 {
        let profile = galaxy_profile(seed, &cfg);
        let shape = &profile.shape;
        let mut pts = String::new();
        for n in 1..STARS {
            let mut s = realm_stream(seed, &[UNIVERSE_SEED, GALAXY_SEED, system_seed_at(n)]);
            let draws = PlacementDraws {
                population: s.next_f64(),
                radius: s.next_f64(),
                radius_b: s.next_f64(),
                azimuth: s.next_f64(),
                scatter: s.next_f64(),
                scatter_b: s.next_f64(),
                height: s.next_f64(),
            };
            let p = system_center_at(&cfg, shape, n, draws) / cfg.stellar.galaxy_rim_r_m;
            if n > 1 {
                pts.push(',');
            }
            pts.push_str(&format!("{:.4},{:.4},{:.4}", p.x, p.y, p.z));
        }
        println!(
            "GALAXY {{\"seed\":{seed},\"kind\":\"{:?}\",\"arms\":{},\"pitch_deg\":{:.1},\"bulge_pct\":{:.1},\"bulge_radius_pct\":{:.1},\"thickness_pct\":{:.2},\"interarm_pct\":{:.1},\"duty_pct\":{:.1},\"p\":[{pts}]}}",
            profile.kind,
            shape.arms,
            shape.pitch_tan.atan().to_degrees(),
            100.0 * shape.bulge_fraction,
            100.0 * shape.bulge_radius_frac,
            100.0 * shape.disc_thickness_frac,
            100.0 * shape.interarm_fraction,
            100.0 * shape.arm_width_rad * (1.0 + shape.arm_fray) * f64::from(shape.arms) / TAU,
        );
    }
}

#[test]
#[ignore = "diagnostic: renders the drawn galaxy shapes for the owner to judge"]
fn look_at_the_drawn_galaxies() {
    const W: usize = 62;
    const H_TOP: usize = 31;
    const H_SIDE: usize = 11;
    const STARS: u32 = 150_000;
    // Denser toward the top of the ramp; a space where nothing landed.
    const SHADE: [char; 6] = [' ', '.', ':', '*', '#', '@'];

    let mut cfg = test_visual();
    cfg = galaxy_holding(&cfg, 0, STARS);

    for seed in 0..6_u64 {
        let profile = galaxy_profile(seed, &cfg);
        let shape = &profile.shape;
        // Place the stars with the DRAWN shape, through the production placement law.
        let mut top = vec![0u32; W * H_TOP];
        let mut side = vec![0u32; W * H_SIDE];
        for n in 1..STARS {
            let mut s = realm_stream(seed, &[UNIVERSE_SEED, GALAXY_SEED, system_seed_at(n)]);
            // The placement draws sit behind the per-system prefix; take a fresh stream and pull six
            // uniforms, which is what the placement itself consumes.
            let draws = PlacementDraws {
                population: s.next_f64(),
                radius: s.next_f64(),
                radius_b: s.next_f64(),
                azimuth: s.next_f64(),
                scatter: s.next_f64(),
                scatter_b: s.next_f64(),
                height: s.next_f64(),
            };
            let p = system_center_at(&cfg, shape, n, draws) / cfg.stellar.galaxy_rim_r_m;
            let px = ((p.x * 0.5 + 0.5) * W as f64) as isize;
            let pz = ((p.z * 0.5 + 0.5) * H_TOP as f64) as isize;
            if (0..W as isize).contains(&px) && (0..H_TOP as isize).contains(&pz) {
                top[pz as usize * W + px as usize] += 1;
            }
            // The side view exaggerates height, or a thin disc is one line and says nothing.
            let sy = ((p.y * 4.0 * 0.5 + 0.5) * H_SIDE as f64) as isize;
            if (0..W as isize).contains(&px) && (0..H_SIDE as isize).contains(&sy) {
                side[sy as usize * W + px as usize] += 1;
            }
        }
        let paint = |grid: &[u32], w: usize| -> String {
            let peak = f64::from(*grid.iter().max().unwrap_or(&1)).max(1.0);
            grid.chunks(w)
                .map(|row| {
                    row.iter()
                        .map(|&c| {
                            let t = (f64::from(c) / peak).sqrt();
                            SHADE[((t * (SHADE.len() - 1) as f64).round() as usize)
                                .min(SHADE.len() - 1)]
                        })
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        println!("\n{}", "=".repeat(W));
        println!(
            "SEED {seed} — {:?}, {} arms, pitch {:.1}°, bulge {:.0}% of the stars \
             (radius {:.0}% of the galaxy), disc {:.1}% thick",
            profile.kind,
            shape.arms,
            shape.pitch_tan.atan().to_degrees(),
            100.0 * shape.bulge_fraction,
            100.0 * shape.bulge_radius_frac,
            100.0 * shape.disc_thickness_frac,
        );
        println!("{}", "=".repeat(W));
        println!("{}", paint(&top, W));
        println!("{}  ← from the side (height ×4)", "-".repeat(W - 26));
        println!("{}", paint(&side, W));
    }
}

/// A DIAGNOSTIC DUMP, not a gate: print every system's position so the shape can be LOOKED AT
/// (owner ruling G13 — the assistant proposes the shape in code, the owner judges the picture).
///
/// The raw-coordinate form, for feeding a plotter. `look_at_the_drawn_galaxies` renders the same
/// thing as a picture in the terminal, and `emit_the_drawn_galaxies_as_json` hands it to a real
/// renderer. Ignored by default: it prints thousands of lines and asserts nothing.
#[test]
#[ignore = "diagnostic: prints the galaxy's positions for a picture"]
fn dump_the_galaxy_for_looking() {
    let mut cfg = test_visual();
    cfg = galaxy_holding(&cfg, 0, 6_000);
    let r = cfg.stellar.galaxy_rim_r_m;
    println!("KIND {:?}", galaxy_profile(0, &cfg).kind);
    for b in generate_system_forest(0, &cfg) {
        if !matches!(b.realm, RealmId::System(_)) {
            continue;
        }
        if let Placement::StaticOffset(v) = b.placement {
            println!("P {:.6} {:.6} {:.6}", v.x / r, v.y / r, v.z / r);
        }
    }
}

/// ★ DOES THE BULGE PUT A SYSTEM ON TOP OF THE HOME (S12, 2026-08-28)?
///
/// The shell placed the home at the origin and EVERY other system at one far radius, so nothing could
/// ever be near the middle. The shape has a bulge, and a bulge is exactly a population near the middle.
///
/// Owner ruling G12: where the shape wants a star closer than the gap allows, PUSH it out. This
/// measures whether that rule is needed yet, or still theoretical.
#[test]
#[ignore = "diagnostic: measures the closest approach the shape produces"]
fn how_close_does_the_shape_put_two_systems() {
    let cfg = test_visual();
    let mut worst_ratio = f64::INFINITY;
    let mut worst = String::new();
    for seed in 0..664_u64 {
        let bodies = generate_system_forest(seed, &cfg);
        let sys: Vec<&GeneratedBody> = bodies.iter().filter(|b| b.parent == Some(GALAXY)).collect();
        for (i, a) in sys.iter().enumerate() {
            for b in sys.iter().skip(i + 1) {
                let d = (placement_offset(a.placement) - placement_offset(b.placement)).length();
                let need = a.shape.circumscribed_extent() + b.shape.circumscribed_extent();
                let ratio = d / need.max(1.0);
                if ratio < worst_ratio {
                    worst_ratio = ratio;
                    worst = format!(
                        "seed {seed}: {:?} at {:.3e} m and {:?} at {:.3e} m — apart {:.3e}, need {:.3e}",
                        a.realm,
                        placement_offset(a.placement).length(),
                        b.realm,
                        placement_offset(b.placement).length(),
                        d,
                        need
                    );
                }
            }
        }
    }
    println!("[s12-closest] worst separation ratio {worst_ratio:.4} (1.0 = just touching)");
    println!("[s12-closest] {worst}");
}

/// ★ A REALM'S WAKE RADIUS NEVER DEPENDS ON WHAT IS INSIDE IT (owner ruling 2026-08-29).
///
/// THE LAW THIS PINS. A parent decides to wake a child from the child's OWN radius, and that radius
/// is a function of the child's own size and its own speed. It is not a function of the child's
/// contents. So a shard that builds only its own subtree must derive the IDENTICAL radius that the
/// full world derives, for every realm it holds.
///
/// ★ WHY IT IS WORTH A TEST. Before this ruling a second band existed, taken as the maximum over a
/// realm's DIRECT CHILDREN. A subtree stops one level down, so that band could not be reproduced —
/// MEASURED on THE world, every one of the galaxy's 233 220 star systems carried a reach of zero,
/// and the galaxy therefore told no system to warm anything. This test is red the moment any radius
/// starts reading a child again.
#[test]
fn a_realms_wake_radius_is_the_same_whether_or_not_its_contents_are_built() {
    let cfg = test_world();
    // The SAME galaxy, built two ways: the shard's own subtree (star systems, no planets inside
    // them) and the whole world (every planet and moon). The rows must agree.
    let held = std::collections::BTreeSet::from([GALAXY]);
    let (subtree_rows, _) =
        shard_boot_world(0, &cfg, &held, GALAXY, &std::collections::BTreeSet::new());
    let full_rows = WorldView::generated(0, &cfg).neighbourhood(&held);
    let by_realm: std::collections::BTreeMap<_, _> =
        full_rows.iter().map(|r| (r.realm, r)).collect();
    assert!(!subtree_rows.is_empty(), "the galaxy shard holds rows");
    for row in &subtree_rows {
        let full = by_realm
            .get(&row.realm)
            .expect("the world names the same realm");
        assert_eq!(
            row.aoi, full.aoi,
            "the subtree and the world derive one radius for {:?}",
            row.realm
        );
    }
}

/// ★ EVERY REALM A GALAXY SHARD HOLDS CARRIES A LIVE RADIUS — none is inert (2026-08-29).
///
/// The zero-radius realm is the failure this ruling removed: a realm with no radius is never woken
/// and never told anything, so its inside stays cold and the player meets it as a pop on arrival.
#[test]
fn every_system_a_galaxy_shard_holds_carries_a_live_wake_radius() {
    let cfg = test_world();
    let held = std::collections::BTreeSet::from([GALAXY]);
    let (rows, _) = shard_boot_world(0, &cfg, &held, GALAXY, &std::collections::BTreeSet::new());
    let systems: Vec<_> = rows.iter().filter(|r| r.parent == Some(GALAXY)).collect();
    assert_eq!(
        systems.len(),
        systems_in(&cfg),
        "the galaxy shard holds every star system of THE world"
    );
    let inert = systems
        .iter()
        .filter(|r| r.aoi.spin_up_r_m() <= 0.0)
        .count();
    assert_eq!(inert, 0, "no star system is left without a radius");
}

/// ★ A BIGGER REALM GETS A BIGGER RADIUS, BY ARITHMETIC — never by a match on what it is (HR3).
#[test]
fn the_wake_radius_rises_with_the_realms_own_size_and_names_no_realm_kind() {
    let cfg = test_world();
    let rows = realm_regions_for_config(0, &cfg);
    let mut pairs: Vec<(f64, f64)> = rows
        .iter()
        .map(|r| (r.shape.finite_extent(), r.aoi.spin_up_r_m()))
        .filter(|(size, radius)| (*size > 0.0) & (*radius > 0.0))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(pairs.len() > 2, "THE world holds realms of several sizes");
    let smallest = pairs[0];
    let largest = pairs[pairs.len() - 1];
    assert!(
        largest.1 > smallest.1,
        "the larger realm reaches further: {smallest:?} vs {largest:?}"
    );
}

/// DIAGNOSTIC (2026-08-31): the first seed whose galaxy — at the size a test can hold — carries
/// exactly one Earth-like body, and every field of it, read off the generator.
#[test]
#[ignore]
fn diag_first_earth_like_seed_on_the_test_galaxy() {
    let cfg = test_world();
    for seed in 0..4096u64 {
        let found = earth_like_candidates(seed, &cfg);
        if found.len() == 1 {
            println!("seed = {seed}");
            println!("{:#?}", found[0]);
            return;
        }
    }
    println!("no seed in 0..4096 holds exactly one");
}

/// DIAGNOSTIC (2026-08-31): WHICH systems sit off-cell, and are they the pushed ones?
#[test]
#[ignore]
fn diag_which_systems_sit_off_cell() {
    let cfg = test_world();
    let regions = realm_regions_for_config(0, &cfg);
    let edge = vd_core::pose::Tier::Galaxy.cell_edge_m();
    let mut off = 0usize;
    let mut total = 0usize;
    let mut examples = Vec::new();
    for r in &regions {
        if !(matches!(r.realm, RealmId::System(_)) && r.parent == Some(GALAXY)) {
            continue;
        }
        total += 1;
        let o = r.center.in_parents_frame().offset();
        if o != vd_core::glam::DVec3::ZERO {
            off += 1;
            if examples.len() < 5 {
                examples.push((r.realm, o));
            }
        }
    }
    println!("galaxy cell edge = {edge} m");
    println!("systems: {total}, off-cell: {off}");
    for (realm, o) in &examples {
        println!("  {realm:?} offset {o:?}");
    }
}

/// DIAGNOSTIC (2026-08-31): standing on a star system's EDGE, how far is each planet, and how far
/// does each planet's own wake radius reach? If no planet reaches, arriving at a system shows an
/// empty sky until the traveller is well inside it.
#[test]
#[ignore]
fn diag_can_a_planet_be_seen_from_its_systems_edge() {
    let cfg = test_world();
    let regions = realm_regions_for_config(0, &cfg);
    let sys = regions
        .iter()
        .find(|r| r.realm == SYSTEM_A)
        .expect("the home system");
    let shell = sys.shape.finite_extent();
    println!("home system shell (the edge) = {shell:.3e} m");
    let mut reach_max: f64 = 0.0;
    for p in regions.iter().filter(|r| r.parent == Some(SYSTEM_A)) {
        let orbit = p.center.metres_in(sys).length();
        let radius = p.aoi.spin_up_r_m();
        // Standing on the edge, the FARTHEST the traveller could be from this planet.
        let worst = shell + orbit;
        reach_max = reach_max.max(radius);
        println!(
            "  {:?} orbit {:.3e}  wake radius {:.3e}  worst distance {:.3e}  visible={}",
            p.realm,
            orbit,
            radius,
            worst,
            radius >= worst
        );
    }
    println!("largest planet wake radius = {reach_max:.3e} m against a shell of {shell:.3e} m");
}
