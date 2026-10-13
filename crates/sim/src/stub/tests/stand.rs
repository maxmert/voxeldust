//! ★ A BODY STANDS ON ITS REALM'S GROUND (ruling B81 Q6, built 2026-10-13): the planet's parked children read against a
//! ground the test holds — a sphere of 1 000 m with a height for each of five patches — through the system itself.

use super::*;
use crate::stub::drive::{DrivenChild, DrivenState};
use crate::stub::stand::{GroundSource, STAND_CLEARANCE_M, Standing, stand_on_ground};
use bevy_ecs::system::RunSystemOnce;
use std::sync::{Arc, Mutex};
use vd_core::ids::EntityId;

const PLANET: RealmId = RealmId::Planet(9);
/// Hull 15: parked with its belly a metre IN the ground (a berth written over a lower ground).
const HULL: RealmId = RealmId::Ship(EntityId(15));
/// A hull stopped 500 m over the ground: it hovers, and is never moved until its ground rises into it.
const HIGH: RealmId = RealmId::Ship(EntityId(16));
/// A pad whose row is static (a station's): parked within its clearance, so it stands and is lowered half a metre.
const PAD: RealmId = RealmId::Station(5);
/// A moon this realm moves by an orbit: its place is its motion's, never the ground's.
const MOON: RealmId = RealmId::Planet(10);
/// A hull parked at the planet's very centre: it stands on no ground.
const CORE: RealmId = RealmId::Ship(EntityId(18));
/// A hull over ground the test does not hold yet (patch 4).
const DARK: RealmId = RealmId::Ship(EntityId(19));
/// A parked attitude kept for a realm that is not one of this realm's direct children (a hull inside hull 15).
const NESTED: RealmId = RealmId::Ship(EntityId(20));

const GROUND_R: f64 = 1_000.0;
const HULL_HALF: DVec3 = DVec3::new(6.0, 3.0, 20.0);

/// The test's ground: each patch's height over the sphere, patch 4 dark until lit, and the patches changed since the last
/// read.
#[derive(Default)]
struct Sphere {
    raised: Mutex<BTreeMap<u64, f64>>,
    lit: Mutex<bool>,
    changed: Mutex<Vec<u64>>,
}

impl Sphere {
    fn patch(dir: DVec3) -> u64 {
        if dir.y > 0.5 {
            1
        } else if dir.x > 0.5 {
            2
        } else if dir.z > 0.5 {
            3
        } else if dir.z < -0.5 {
            4
        } else {
            0
        }
    }

    fn raise(&self, patch: u64, by: f64) {
        *self
            .raised
            .lock()
            .expect("the ground's lock")
            .entry(patch)
            .or_default() += by;
        self.changed.lock().expect("the ground's lock").push(patch);
    }
}

struct Shared(Arc<Sphere>);

impl crate::stub::stand::Ground for Shared {
    fn radius_m(&self, dir: DVec3) -> Option<f64> {
        let p = Sphere::patch(dir);
        let dark = (p == 4) & !*self.0.lit.lock().expect("the ground's lock");
        (!dark).then(|| {
            GROUND_R
                + self
                    .0
                    .raised
                    .lock()
                    .expect("the ground's lock")
                    .get(&p)
                    .copied()
                    .unwrap_or(0.0)
        })
    }
    fn patches(&self, dir: DVec3) -> Vec<u64> {
        vec![Sphere::patch(dir)]
    }
    fn changed(&self) -> Vec<u64> {
        std::mem::take(&mut *self.0.changed.lock().expect("the ground's lock"))
    }
}

fn forest() -> Vec<RealmRegion> {
    vec![
        region(PLANET, None, DVec3::ZERO, 2.0 * GROUND_R),
        region_box(HULL, Some(PLANET), DVec3::new(0.0, 1_002.0, 0.0), HULL_HALF),
        region_box(HIGH, Some(PLANET), DVec3::new(1_500.0, 0.0, 0.0), HULL_HALF),
        region_box(
            PAD,
            Some(PLANET),
            DVec3::new(0.0, 0.0, 1_003.5),
            DVec3::new(10.0, 10.0, 2.0),
        ),
        region(MOON, Some(PLANET), DVec3::new(5_000.0, 0.0, 0.0), 50.0),
        region_box(CORE, Some(PLANET), DVec3::ZERO, HULL_HALF),
        region_box(
            DARK,
            Some(PLANET),
            DVec3::new(0.0, 0.0, -1_002.0),
            HULL_HALF,
        ),
        region_box(NESTED, Some(HULL), DVec3::ZERO, DVec3::ONE),
    ]
}

/// A world holding the planet's children as its boot reads them, and the ground.
fn world_over(sphere: &Arc<Sphere>) -> World {
    let motion = MotionFn(std::sync::Arc::new(|_| FramePlacement::identity()));
    let parked: BTreeMap<RealmId, DQuat> = [HULL, HIGH, PAD, MOON, CORE, DARK, NESTED]
        .into_iter()
        .map(|c| (c, DQuat::IDENTITY))
        .collect();
    let mut world = World::new();
    world.insert_resource(config());
    world.insert_resource(ClockSample::default());
    world.insert_resource(GroundSource(Some(Box::new(Shared(Arc::clone(sphere))))));
    world.insert_resource(Standing::default());
    world.insert_resource(crate::stub::drive::DrivenChildren::default());
    world.insert_resource(
        RealmRegions::new(forest())
            .with_berthed_facings(parked)
            .with_moving_children(BTreeMap::from([(MOON, motion)]))
            .with_own_realm(PLANET),
    );
    world
}

/// One tick at `tick` (the test config's 20 a second).
fn tick(world: &mut World, at: u64) {
    world.resource_mut::<ClockSample>().universe_tick = UniverseTick(at);
    world
        .run_system_once(stand_on_ground)
        .expect("the system runs");
}

fn lift(world: &World, child: RealmId) -> DVec3 {
    world.resource::<RealmRegions>().lift_of(child)
}

/// Where the realm stands a parked child: its authored centre and its lift.
fn place(world: &World, child: RealmId) -> DVec3 {
    let parked = world
        .resource::<RealmRegions>()
        .parked_at(child)
        .expect("a parked child");
    parked.centre + parked.lift
}

fn stands(world: &World, child: RealmId) -> bool {
    world.resource::<Standing>().stands(child)
}

fn close(a: DVec3, b: DVec3) {
    assert!((a - b).length() < 1e-6, "{a} against {b}");
}

/// A drive state at rest on its parked place (its facts arrived; no stick yet).
fn at_rest(world: &mut World, child: RealmId) {
    let held = DrivenChild {
        state: DrivenState {
            pos_m: lift(world, child),
            vel_mps: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            spin_radps: DVec3::ZERO,
        },
        drive: ([0; 3], [0; 3]),
        drive_at: (vd_core::fence::Fence::GENESIS, UniverseTick(0)),
        facts: None,
        facts_at: (vd_core::fence::Fence::GENESIS, UniverseTick(0)),
        frozen: false,
    };
    world
        .resource_mut::<crate::stub::drive::DrivenChildren>()
        .0
        .insert(child, held);
}

fn pos_of(world: &World, child: RealmId) -> DVec3 {
    world
        .resource::<crate::stub::drive::DrivenChildren>()
        .state_of(child)
        .expect("a drive state")
        .pos_m
}

/// ★ THE FIRST STATEMENT: with no ground nothing moves; once the ground is here, hull 15 (its belly a metre in the
/// ground) stands at once with its belly a metre over it, its drive state at rest with it; the pad within its clearance
/// is lowered half a metre and its static row carries it; the hull 500 m up, the moon, the hull at the centre, the hull
/// over dark ground and the nested hull are not moved.
#[test]
fn the_first_statement_stands_every_buried_or_touching_parked_child_on_its_ground_at_once() {
    let sphere = Arc::new(Sphere::default());
    let mut world = world_over(&sphere);
    world.resource_mut::<GroundSource>().0 = None;
    at_rest(&mut world, HULL);
    tick(&mut world, 100);
    close(place(&world, HULL), DVec3::new(0.0, 1_002.0, 0.0));
    assert_eq!(
        format!("{:?}", world.resource::<GroundSource>()),
        "GroundSource { present: false }"
    );
    world.resource_mut::<GroundSource>().0 = Some(Box::new(Shared(Arc::clone(&sphere))));
    sphere.raise(0, 0.0); // a change before the first statement is read with it
    tick(&mut world, 101);
    let need = GROUND_R + HULL_HALF.y + STAND_CLEARANCE_M;
    close(place(&world, HULL), DVec3::new(0.0, need, 0.0));
    assert_eq!(
        lift(&world, HULL),
        DVec3::ZERO,
        "the stand landed in the centre"
    );
    assert_eq!(
        pos_of(&world, HULL),
        DVec3::ZERO,
        "its drive state at rest on that centre"
    );
    assert!(stands(&world, HULL));
    close(place(&world, PAD), DVec3::new(0.0, 0.0, 1_003.0));
    let rows = world
        .resource::<RealmRegions>()
        .static_rows_of(PLANET)
        .expect("the planet's static rows")
        .clone();
    let pad_row = rows
        .iter()
        .find(|(f, _)| *f == frame_for_realm(PAD, None).expect("a frame"))
        .expect("the pad's row");
    close(pad_row.1.origin, DVec3::new(0.0, 0.0, 1_003.0));
    for (child, at, why) in [
        (HIGH, DVec3::new(1_500.0, 0.0, 0.0), "500 m up it hovers"),
        (CORE, DVec3::ZERO, "at the centre no ground is under it"),
        (
            DARK,
            DVec3::new(0.0, 0.0, -1_002.0),
            "its ground is not held",
        ),
    ] {
        close(place(&world, child), at);
        assert!(!stands(&world, child), "{why}");
    }
    for child in [MOON, NESTED] {
        assert_eq!(
            lift(&world, child),
            DVec3::ZERO,
            "an orbit moves it, or it is not a direct child"
        );
        assert!(!stands(&world, child));
    }
    assert!(format!("{:?}", world.resource::<GroundSource>()).contains("true"));
}

/// ★ THE GROUND CHANGES UNDER A STANDING CHILD: the rise is followed over half a second (halfway at a quarter), a fall
/// lowers it the same way, a child over another patch is not read, and a hovering hull is lifted only once its ground
/// rises into it; the dark ground, once lit, stands its hull.
#[test]
fn a_standing_child_follows_its_ground_over_the_ramp_and_a_hovering_one_moves_only_when_buried() {
    let sphere = Arc::new(Sphere::default());
    let mut world = world_over(&sphere);
    tick(&mut world, 100);
    let start = place(&world, HULL);
    sphere.raise(1, 10.0);
    tick(&mut world, 200);
    close(place(&world, HULL), start);
    tick(&mut world, 205);
    close(place(&world, HULL), start + DVec3::new(0.0, 5.0, 0.0));
    assert_eq!(
        lift(&world, HULL),
        DVec3::new(0.0, 5.0, 0.0),
        "mid-ramp the lift carries it"
    );
    tick(&mut world, 210);
    close(place(&world, HULL), start + DVec3::new(0.0, 10.0, 0.0));
    assert_eq!(lift(&world, HULL), DVec3::ZERO, "landed");
    tick(&mut world, 211);
    close(place(&world, HULL), start + DVec3::new(0.0, 10.0, 0.0));
    sphere.raise(1, -20.0);
    tick(&mut world, 300);
    tick(&mut world, 310);
    close(place(&world, HULL), start + DVec3::new(0.0, -10.0, 0.0));
    // The hull 500 m up: a 400 m rise leaves it hovering, a 600 m one buries it and it stands.
    sphere.raise(2, 400.0);
    tick(&mut world, 400);
    assert!(!stands(&world, HIGH));
    sphere.raise(2, 200.0);
    tick(&mut world, 500);
    tick(&mut world, 510);
    let need = GROUND_R + 600.0 + HULL_HALF.x + STAND_CLEARANCE_M;
    close(place(&world, HIGH), DVec3::new(need, 0.0, 0.0));
    // The dark ground is lit: its hull stands, following the ramp.
    *sphere.lit.lock().expect("the ground's lock") = true;
    sphere.changed.lock().expect("the ground's lock").push(4);
    tick(&mut world, 600);
    tick(&mut world, 610);
    assert!(stands(&world, DARK));
    let need = GROUND_R + HULL_HALF.y + STAND_CLEARANCE_M;
    close(place(&world, DARK), DVec3::new(0.0, 0.0, -need));
    // A patch nobody stands over changes: nothing is read.
    sphere.changed.lock().expect("the ground's lock").push(99);
    tick(&mut world, 700);
}

/// ★ A CHILD THAT FLIES STANDS ON NOTHING: one that left its place before its ground changed, and one that leaves
/// mid-ramp, leave the book and keep their place; one the realm no longer holds leaves it too.
#[test]
fn a_child_that_flies_or_leaves_the_realm_leaves_the_standing_book() {
    let sphere = Arc::new(Sphere::default());
    let mut world = world_over(&sphere);
    at_rest(&mut world, HULL);
    tick(&mut world, 100);
    let stood = place(&world, HULL);
    // Mid-ramp: the pilot pushes off.
    sphere.raise(1, 10.0);
    tick(&mut world, 200);
    world
        .resource_mut::<crate::stub::drive::DrivenChildren>()
        .0
        .get_mut(&HULL)
        .expect("held")
        .state
        .vel_mps = DVec3::new(0.0, 3.0, 0.0);
    tick(&mut world, 205);
    assert!(!stands(&world, HULL));
    close(place(&world, HULL), stood);
    // Flying when its ground changes: nothing moves and it stays out of the book.
    sphere.raise(1, 10.0);
    tick(&mut world, 300);
    close(place(&world, HULL), stood);
    // A child the realm no longer holds: its patch changes, and it leaves the book.
    assert!(stands(&world, PAD));
    let without: Vec<RealmRegion> = forest().into_iter().filter(|r| r.realm != PAD).collect();
    *world.resource_mut::<RealmRegions>() = RealmRegions::new(without).with_own_realm(PLANET);
    sphere.raise(3, 1.0);
    tick(&mut world, 400);
    assert!(!stands(&world, PAD));
}

/// The regions' side: a lift for a realm the forest does not hold is kept and moves no row; the parked children are the
/// direct children with a parked attitude; a realm whose parent region is missing has no parked place.
#[test]
fn the_regions_answer_lifts_and_parked_places_only_for_their_own_direct_children() {
    let mut regions = RealmRegions::new(forest())
        .with_berthed_facings(BTreeMap::from([
            (HULL, DQuat::IDENTITY),
            (CORE, DQuat::IDENTITY),
        ]))
        .with_own_realm(PLANET);
    regions.set_lift(RealmId::Ship(EntityId(99)), DVec3::X);
    assert_eq!(regions.lift_of(RealmId::Ship(EntityId(99))), DVec3::X);
    // A stand that lands for a realm the forest does not hold drops its lift and adopts nothing; so does one on a shard
    // that names no realm of its own.
    regions.fold_lift(RealmId::Ship(EntityId(99)));
    assert_eq!(regions.lift_of(RealmId::Ship(EntityId(99))), DVec3::ZERO);
    let mut unnamed = RealmRegions::new(forest());
    unnamed.set_lift(HULL, DVec3::Y);
    unnamed.fold_lift(HULL);
    assert_eq!(unnamed.lift_of(HULL), DVec3::ZERO);
    assert_eq!(regions.parked_children(), vec![HULL, CORE]);
    assert!(regions.parked_at(RealmId::Ship(EntityId(99))).is_none());
    assert!(regions.parked_at(NESTED).is_none());
    assert!(
        regions.parked_at(MOON).is_some(),
        "not moving in this forest"
    );
    let orphaned = RealmRegions::new(vec![region_box(HULL, Some(PLANET), DVec3::ZERO, HULL_HALF)])
        .with_own_realm(PLANET);
    assert!(orphaned.parked_at(HULL).is_none());
    // An oriented box and a shell stand by their own extents.
    let shapes = RealmRegions::new(vec![
        region(PLANET, None, DVec3::ZERO, 2.0 * GROUND_R),
        RealmRegion {
            shape: Boundary::Obb {
                half: HULL_HALF,
                orient: DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
            },
            ..region_box(HULL, Some(PLANET), DVec3::new(0.0, 1_000.0, 0.0), HULL_HALF)
        },
        region(CORE, Some(PLANET), DVec3::new(0.0, 1_000.0, 0.0), 4.0),
    ]);
    let sphere = Arc::new(Sphere::default());
    let mut world = world_over(&sphere);
    *world.resource_mut::<RealmRegions>() = shapes
        .with_berthed_facings(BTreeMap::from([
            (HULL, DQuat::IDENTITY),
            (CORE, DQuat::IDENTITY),
        ]))
        .with_own_realm(PLANET);
    tick(&mut world, 100);
    // The box turned on its end reaches 20 m down; the shell 4 m.
    close(
        place(&world, HULL),
        DVec3::new(0.0, GROUND_R + 20.0 + STAND_CLEARANCE_M, 0.0),
    );
    close(
        place(&world, CORE),
        DVec3::new(0.0, GROUND_R + 4.0 + STAND_CLEARANCE_M, 0.0),
    );
}
