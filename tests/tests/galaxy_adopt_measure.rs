//! ★ THE GALAXY ADOPTS ONE HULL AMONG ALL ITS STAR SYSTEMS (SL9, measured — the ruler switch, slice 2's
//! owed measurement). A cost that grows with the number of children is a defect, and it must be
//! measured on a realm with many, not argued.
//!
//! THE world's system layer (SL5: one world, the shipped seed and config) gives the galaxy every star
//! system it holds. A hull region is adopted and released a hundred times on that galaxy, and a hundred
//! times on a ten-child realm; the per-operation cost on the galaxy must stay within a small factor of
//! the small realm's. A walk over all children would be tens of thousands of times slower, so the
//! bound is loose against machine noise and tight against the defect it guards.
use std::collections::BTreeSet;
use std::time::Instant;
use vd_core::built::Berth;
use vd_core::geometry::{AoiConfig, Boundary, ContainmentBand, ParentCentre, RealmRegion};
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, LatticePos, RealmId, Tier};
use vd_core::worldgen::WorldRealms;
use vd_physics::worldgen::{HOME_SEED, UniverseConfig, system_layer_view};
use vd_sim::stub::RealmRegions;

const ROUNDS: u32 = 100;
/// The galaxy may cost at most this many times the ten-child realm per adopt+release. A per-child walk
/// on 233 220 systems would be ~20 000× — the factor a regression would show.
const MAX_RATIO: f64 = 25.0;

fn hull_region(parent: RealmId, parent_tier: Tier) -> RealmRegion {
    let ship = vd_core::EntityId::pack(vd_core::entity_kind::EntityKind::Ship, 1, 1, 0);
    RealmRegion {
        realm: RealmId::Ship(ship),
        center: ParentCentre::authored(LatticePos::from_metres(
            DVec3::new(1.0e9, 2.0e9, -3.0e9),
            parent_tier,
        )),
        frame: FrameRef::ShipLocal { ship },
        shape: Boundary::Shell { r: 20.0 },
        look: Some(Boundary::Shell { r: 20.0 }),
        band: ContainmentBand::for_containment_velocity_safe(50.0, 100.0, 1.0, 0.05, 1.0)
            .expect("valid containment band"),
        aoi: AoiConfig::inert(),
        parent: Some(parent),
    }
}

/// Adopt + release `ROUNDS` times; the mean seconds per round.
fn measure(regions: &mut RealmRegions, hull: RealmRegion) -> f64 {
    let realm = hull.realm;
    let started = Instant::now();
    for _ in 0..ROUNDS {
        regions.adopt_child(hull);
        regions.release_child(realm);
    }
    started.elapsed().as_secs_f64() / f64::from(ROUNDS)
}

#[test]
fn the_galaxy_adopts_a_hull_at_a_cost_that_does_not_grow_with_its_children() {
    // THE world's system layer: the galaxy and every star system in it.
    let config = UniverseConfig::world(500.0, 0.05);
    let world: WorldRealms = system_layer_view(HOME_SEED, &config).lowered();
    let galaxy = world
        .regions()
        .iter()
        .find(|r| matches!(r.realm, RealmId::Galaxy(_)))
        .map(|r| r.realm)
        .expect("THE world has a galaxy");
    let held: BTreeSet<RealmId> = BTreeSet::from([galaxy]);
    let neighbourhood = world.neighbourhood(&held);
    let mut galaxy_regions = RealmRegions::new(neighbourhood).with_own_realm(galaxy);
    let children = galaxy_regions.direct_children(galaxy).count();
    assert!(
        children > 100_000,
        "the measurement needs a realm with MANY children: {children}"
    );
    let galaxy_tier = galaxy_regions
        .hosted_frame(galaxy)
        .expect("the galaxy has a frame")
        .tier();
    let galaxy_cost = measure(&mut galaxy_regions, hull_region(galaxy, galaxy_tier));
    assert_eq!(
        galaxy_regions.direct_children(galaxy).count(),
        children,
        "every adopt was released"
    );

    // The control: a star system with ten children, from the same world's shapes.
    let system = RealmId::System(7);
    let mut small = vec![RealmRegion {
        realm: system,
        center: ParentCentre::authored(LatticePos::from_metres(DVec3::ZERO, Tier::Galaxy)),
        frame: FrameRef::SystemSpace { system_seed: 7 },
        shape: Boundary::Shell { r: 1.0e12 },
        look: Some(Boundary::Shell { r: 1.0e12 }),
        band: ContainmentBand::for_containment_velocity_safe(50.0, 100.0, 1.0, 0.05, 1.0)
            .expect("valid containment band"),
        aoi: AoiConfig::inert(),
        parent: None,
    }];
    for i in 0..10u64 {
        small.push(RealmRegion {
            realm: RealmId::Planet(i),
            center: ParentCentre::authored(LatticePos::from_metres(
                DVec3::new(1.0e8 * (i as f64 + 1.0), 0.0, 0.0),
                Tier::Fine,
            )),
            frame: FrameRef::PlanetCentered { planet_seed: i },
            shape: Boundary::Shell { r: 1.0e6 },
            look: Some(Boundary::Shell { r: 1.0e6 }),
            band: ContainmentBand::for_containment_velocity_safe(50.0, 100.0, 1.0, 0.05, 1.0)
                .expect("valid containment band"),
            aoi: AoiConfig::inert(),
            parent: Some(system),
        });
    }
    let mut small_regions = RealmRegions::new(small).with_own_realm(system);
    let small_cost = measure(&mut small_regions, hull_region(system, Tier::Fine));

    let ratio = galaxy_cost / small_cost.max(1.0e-9);
    eprintln!(
        "galaxy adopt+release over {children} children: {:.1} µs per round; ten children: {:.1} µs; ratio {ratio:.1}",
        galaxy_cost * 1.0e6,
        small_cost * 1.0e6
    );
    assert!(
        ratio <= MAX_RATIO,
        "the galaxy's adopt+release costs {ratio:.1}× a ten-child realm's — a cost that grows with the children"
    );
    // The berth shape is what the store keeps for it; name it so the row type stays in the picture.
    let _berth_shape = Berth {
        child: hull_region(galaxy, galaxy_tier).realm,
        offset_m: DVec3::ZERO,
        bound: Boundary::Shell { r: 20.0 },
        look: Boundary::Shell { r: 20.0 },
        fence: vd_core::Fence(1),
    };
}
