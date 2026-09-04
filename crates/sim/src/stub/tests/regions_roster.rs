//! THE ROSTER'S EDGES (HR5, 2026-09-04): every arm of adopt, re-parent and release on the region
//! table, driven through the public API alone — a moving child, a live band, a root row, a row
//! that is not the last, a re-driven release. The six-shard crossing fixture flies the happy path;
//! these pin the edges the coverage gate found bare.

use super::*;
use crate::stub::drive::{DrivenChild, DrivenChildren, DrivenState};
use crate::stub::reach::StatedReach;
use crate::stub::realm_head::{ChildRealmNodes, ParentRealmNode};
use crate::stub::regions::ReachOutcome;
use glam::DQuat;
use vd_physics::motion::{Motion, motion_fn};

const MOVER: RealmId = RealmId::Planet(44);
const STATIC_FAR: RealmId = RealmId::Planet(45);
const STATIC_SAME: RealmId = RealmId::Planet(46);

/// A ballistic mover: one motion the table registers under `with_moving_children`.
fn a_motion() -> MotionFn {
    motion_fn(Motion::Integrated {
        placement: FramePlacement::moving(DVec3::new(2_000.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0)),
        acceleration: DVec3::ZERO,
    })
}

fn live(realm: RealmId, centre: DVec3, r: f64) -> RealmRegion {
    RealmRegion {
        aoi: aoi_band(2),
        ..region(realm, Some(OWN_REALM), centre, r)
    }
}

fn driven_at_rest() -> DrivenChild {
    DrivenChild {
        state: DrivenState {
            pos_m: DVec3::ZERO,
            vel_mps: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            spin_radps: DVec3::ZERO,
        },
        drive: ([0; 3], [0; 3]),
        drive_at: (Fence::GENESIS, UniverseTick(0)),
        facts: None,
        facts_at: (Fence::GENESIS, UniverseTick(0)),
        frozen: false,
    }
}

#[test]
fn a_moving_child_adopted_at_runtime_is_never_indexed_and_always_asked() {
    let mut regions = RealmRegions::new(vec![root_region(), own_region()])
        .with_moving_children(BTreeMap::from([(MOVER, a_motion())]))
        .with_own_realm(OWN_REALM);
    regions.adopt_child(live(MOVER, DVec3::new(2_000.0, 0.0, 0.0), 100.0));
    assert!(regions.unindexed_rows().contains(&MOVER));
    assert!(!regions.child_index().answers_for(MOVER));
    assert!(!regions.aoi_index().answers_for(MOVER));
    // And it is one of the realm's moving children, driven or not.
    let driven = DrivenChildren::default();
    assert!(
        regions
            .moving_children_of(OWN_REALM, &driven)
            .contains(&MOVER)
    );
}

#[test]
fn a_live_band_child_adopted_at_runtime_joins_the_range_index_and_the_radial_list() {
    let mut regions =
        RealmRegions::new(vec![root_region(), own_region()]).with_own_realm(OWN_REALM);
    regions.adopt_child(live(STATIC_FAR, DVec3::new(30_000.0, 0.0, 0.0), 100.0));
    regions.adopt_child(live(STATIC_SAME, DVec3::new(0.0, 30_000.0, 0.0), 100.0));
    assert!(regions.aoi_index().answers_for(STATIC_FAR));
    assert!(regions.aoi_index().answers_for(STATIC_SAME));
    // Two children at ONE radial distance share a slot; releasing one keeps the other's entry, and
    // releasing the second empties the slot. Neither is the last row when it goes, so the moved row's
    // tables are re-pointed.
    regions.release_child(STATIC_FAR);
    assert!(!regions.aoi_index().answers_for(STATIC_FAR));
    assert!(regions.aoi_index().answers_for(STATIC_SAME));
    assert!(regions.direct_child(OWN_REALM, STATIC_SAME).is_some());
    regions.release_child(STATIC_SAME);
    assert!(!regions.aoi_index().answers_for(STATIC_SAME));
    assert_eq!(regions.direct_children(OWN_REALM).count(), 0);
}

#[test]
fn releasing_a_row_that_is_not_the_last_re_points_the_moved_row_and_its_movers() {
    // Three direct children: a static, a MOVER, a static. Releasing the first moves the last into
    // its slot; the mover's index list is re-pointed; releasing the mover (now last) drops it from
    // the mover list too.
    let mut regions = RealmRegions::new(vec![root_region(), own_region()])
        .with_moving_children(BTreeMap::from([(MOVER, a_motion())]))
        .with_own_realm(OWN_REALM);
    regions.adopt_child(live(STATIC_FAR, DVec3::new(30_000.0, 0.0, 0.0), 100.0));
    regions.adopt_child(live(MOVER, DVec3::new(2_000.0, 0.0, 0.0), 100.0));
    regions.adopt_child(live(STATIC_SAME, DVec3::new(0.0, 30_000.0, 0.0), 100.0));
    let driven = DrivenChildren::default();
    regions.release_child(STATIC_FAR);
    assert!(
        regions
            .moving_children_of(OWN_REALM, &driven)
            .contains(&MOVER)
    );
    assert!(regions.direct_child(OWN_REALM, STATIC_SAME).is_some());
    regions.release_child(MOVER);
    assert!(
        !regions
            .moving_children_of(OWN_REALM, &driven)
            .contains(&MOVER)
    );
    regions.release_child(STATIC_SAME);
    assert_eq!(regions.direct_children(OWN_REALM).count(), 0);
    // Releasing a mover that IS the last row also drops it from the mover list.
    regions.adopt_child(live(STATIC_FAR, DVec3::new(30_000.0, 0.0, 0.0), 100.0));
    regions.adopt_child(live(MOVER, DVec3::new(2_000.0, 0.0, 0.0), 100.0));
    regions.release_child(MOVER);
    assert!(
        !regions
            .moving_children_of(OWN_REALM, &driven)
            .contains(&MOVER)
    );
    // And releasing a mover that is NOT the last row re-points the movers that sat at the end.
    regions.adopt_child(live(MOVER, DVec3::new(2_000.0, 0.0, 0.0), 100.0));
    regions.adopt_child(live(STATIC_SAME, DVec3::new(0.0, 30_000.0, 0.0), 100.0));
    regions.release_child(STATIC_FAR);
    regions.release_child(MOVER);
    assert!(regions.direct_child(OWN_REALM, STATIC_SAME).is_some());
}

#[test]
fn a_root_row_can_be_released_and_a_moved_root_row_keeps_its_place() {
    // The root has no parent: its release touches no child list; when a later row is released the
    // root may be the row that moves, and a moved root has no parent list to re-point.
    let mut regions = RealmRegions::new(vec![
        own_region(),
        region(
            STATIC_FAR,
            Some(OWN_REALM),
            DVec3::new(30_000.0, 0.0, 0.0),
            100.0,
        ),
        root_region(),
    ])
    .with_own_realm(OWN_REALM);
    regions.release_child(STATIC_FAR);
    assert!(regions.direct_child(OWN_REALM, STATIC_FAR).is_none());
    assert!(regions.coord_of(OWN_REALM).is_some());
    regions.release_child(ROOT_REALM);
    assert!(regions.coord_of(ROOT_REALM).is_none());
}

#[test]
fn re_parenting_the_own_realm_covers_every_refusal_and_the_move() {
    // No own realm named: nothing to re-parent.
    let mut nameless = RealmRegions::new(vec![root_region(), own_region()]);
    nameless.reparent_own(RealmId::Galaxy(1));
    assert_eq!(nameless.parent_of(OWN_REALM), Some(ROOT_REALM));
    // Named but not on the roster: nothing to re-parent.
    let mut absent = RealmRegions::new(vec![root_region()]).with_own_realm(OWN_REALM);
    absent.reparent_own(RealmId::Galaxy(1));
    assert!(absent.coord_of(OWN_REALM).is_none());
    // The same parent again: a no-op.
    let mut same = RealmRegions::new(vec![root_region(), own_region()]).with_own_realm(OWN_REALM);
    same.reparent_own(ROOT_REALM);
    assert_eq!(same.parent_of(OWN_REALM), Some(ROOT_REALM));
    // A real move from a rostered parent.
    same.reparent_own(RealmId::Galaxy(1));
    assert_eq!(same.parent_of(OWN_REALM), Some(RealmId::Galaxy(1)));
    // A move when the own realm was a ROOT (no old parent) and when the old parent is a stranger
    // this shard never rostered.
    let mut rooted = RealmRegions::new(vec![region(OWN_REALM, None, DVec3::ZERO, 100_000.0)])
        .with_own_realm(OWN_REALM);
    rooted.reparent_own(RealmId::Galaxy(1));
    assert_eq!(rooted.parent_of(OWN_REALM), Some(RealmId::Galaxy(1)));
    let mut orphan = RealmRegions::new(vec![region(
        OWN_REALM,
        Some(RealmId::Galaxy(9)),
        DVec3::ZERO,
        100_000.0,
    )])
    .with_own_realm(OWN_REALM);
    orphan.reparent_own(RealmId::Galaxy(1));
    assert_eq!(orphan.parent_of(OWN_REALM), Some(RealmId::Galaxy(1)));
}

#[test]
fn the_moving_set_and_the_book_overlay_ignore_strangers_and_other_parents_children() {
    let far_parent = RealmId::Planet(43);
    let mut regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(
            far_parent,
            Some(OWN_REALM),
            DVec3::new(-40_000.0, 0.0, 0.0),
            1_000.0,
        ),
        region(
            STATIC_FAR,
            Some(far_parent),
            DVec3::new(10.0, 0.0, 0.0),
            1.0,
        ),
    ])
    .with_moving_children(BTreeMap::from([(MOVER, a_motion())]))
    .with_own_realm(OWN_REALM);
    regions.adopt_child(live(MOVER, DVec3::new(2_000.0, 0.0, 0.0), 100.0));
    let mut driven = DrivenChildren::default();
    driven.0.insert(RealmId::Planet(99), driven_at_rest()); // never rostered
    driven.0.insert(STATIC_FAR, driven_at_rest()); // another parent's child
    driven.0.insert(MOVER, driven_at_rest()); // moving AND driven: the mover list wins
    // The own realm's moving set: the mover once, no stranger, no grandchild.
    let set = regions.moving_children_of(OWN_REALM, &driven);
    assert_eq!(set, BTreeSet::from([MOVER]));
    // A realm with no movers of its own answers only its driven children.
    assert_eq!(
        regions.moving_children_of(far_parent, &driven),
        BTreeSet::from([STATIC_FAR])
    );
    // The book over the own realm carries the mover once; the stranger and the grandchild are absent.
    let book = regions.author_book_driven(OWN_REALM, 20.0, UniverseTick(5), &driven);
    assert!(book.of(frame_of(MOVER)).is_some());
    assert!(book.of(frame_of(STATIC_FAR)).is_none());
}

// ===================== the reach (owner ruling 2026-09-02 R3/R6; built 2026-09-04) =====================

const AU_M: f64 = 1.495_978_707e11;

/// A planet-like child: 6 400 km across, 30 million km out, with a live boot band.
fn planet(realm: RealmId, x_m: f64) -> RealmRegion {
    RealmRegion {
        aoi: vd_core::geometry::AoiConfig::for_velocity_safe(
            6.4e6,
            vd_core::geometry::visibility_factor(vd_core::geometry::VISIBILITY_THETA_MIN_RAD),
            vd_core::geometry::visibility_factor(vd_core::geometry::VISIBILITY_THETA_MIN_RAD),
            10.0,
            0.05,
            2,
            0.0,
        )
        .expect("live"),
        ..region(realm, Some(OWN_REALM), DVec3::new(x_m, 0.0, 0.0), 6.4e6)
    }
}

#[test]
fn a_childs_planted_light_widens_its_band_to_its_light_reach_unless_the_parent_lights_it() {
    // An Earth-like reflected light: about 50 AU of reach, far past the 490 000 km its size makes.
    let earth_luma = 0.3 * 6.4e6_f64.powi(2) / (4.0 * AU_M * AU_M);
    let luma = BTreeMap::from([(STATIC_FAR, (7_u8, earth_luma))]);
    let regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(STATIC_FAR, 3.0e10),
    ])
    .with_child_light(&luma)
    .with_own_realm(OWN_REALM);
    let band = regions
        .direct_child(OWN_REALM, STATIC_FAR)
        .expect("rostered")
        .aoi;
    assert!(
        band.spin_up_r_m() > 45.0 * AU_M,
        "{}",
        band.spin_up_r_m() / AU_M
    );
    assert!(band.tear_down_r_m() > band.spin_up_r_m());
    // The leaf followed the band: a point 40 AU out names the planet.
    let far = LatticePos::from_metres(DVec3::new(3.0e10 + 40.0 * AU_M, 0.0, 0.0), Tier::Fine);
    assert!(
        regions
            .aoi_index()
            .candidates(far, Tier::Fine)
            .contains(&STATIC_FAR)
    );
    // A parent that draws its children's light leaves the band at the size reach.
    let galaxy_like = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(STATIC_FAR, 3.0e10),
    ])
    .with_child_light(&luma)
    .with_lights_children(true)
    .with_own_realm(OWN_REALM);
    let lit = galaxy_like
        .direct_child(OWN_REALM, STATIC_FAR)
        .expect("rostered")
        .aoi;
    assert!(
        (lit.spin_up_r_m() - 6.4e6 * 76.39).abs() < 1.0e6,
        "{}",
        lit.spin_up_r_m()
    );
    // Builder order cannot matter.
    let other_order = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(STATIC_FAR, 3.0e10),
    ])
    .with_own_realm(OWN_REALM)
    .with_child_light(&luma);
    assert_eq!(
        other_order
            .direct_child(OWN_REALM, STATIC_FAR)
            .expect("rostered")
            .aoi,
        band
    );
}

#[test]
fn a_realms_own_reach_folds_its_look_and_its_childrens_distance_plus_reach() {
    let regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(STATIC_FAR, 3.0e10),
        planet(STATIC_SAME, -5.0e10),
    ])
    .with_own_realm(OWN_REALM);
    // No light anywhere: the reach by light is zero; by size it is the farther child's distance plus
    // that child's own size reach (own look 100 000 m reaches only 7.6 million m).
    let (size, light) = regions.own_reach(OWN_REALM, None);
    let child_size =
        6.4e6 * vd_core::geometry::visibility_factor(vd_core::geometry::VISIBILITY_THETA_MIN_RAD);
    assert_eq!(size, (5.0e10 + child_size).round() as u64);
    assert_eq!(light, 0);
    // The realm's own light adds its own reach by light.
    let (_, lit) = regions.own_reach(OWN_REALM, Some(1.0));
    assert!(lit as f64 > 6.0e17, "a Sun's glow: {lit}");
}

#[test]
fn a_child_stating_its_reach_re_bands_it_refolds_the_parent_and_refuses_stale_and_strangers() {
    let mut regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(STATIC_FAR, 3.0e10),
        planet(STATIC_SAME, -5.0e10),
        region(MOVER, Some(STATIC_FAR), DVec3::new(1.0e8, 0.0, 0.0), 10.0), // a grandchild
    ])
    .with_own_realm(OWN_REALM);
    let (size0, _) = regions.own_reach(OWN_REALM, None);
    let at = |f: u64, t: u64| (Fence(f), UniverseTick(t));
    // The nearer child states a huge reach by light: its band widens, the parent's reach by light
    // moves (distance + reach), its reach by size does not.
    let out = regions.set_child_reach(STATIC_FAR, at(1, 10), 500_000_000, 8_000_000_000_000);
    assert_eq!(out, ReachOutcome::Applied { own_changed: true });
    let band = regions
        .direct_child(OWN_REALM, STATIC_FAR)
        .expect("rostered")
        .aoi;
    assert_eq!(band.spin_up_r_m(), 8.0e12);
    let (size1, light1) = regions.own_reach(OWN_REALM, None);
    assert_eq!(size1, size0);
    assert_eq!(light1, 3.0e10 as u64 + 8_000_000_000_000);
    // The same statement again, older: stale. A newer one that changes nothing at the top: applied,
    // own unchanged.
    assert_eq!(
        regions.set_child_reach(STATIC_FAR, at(1, 9), 1, 1),
        ReachOutcome::Stale
    );
    // A newer statement that restates the default exactly moves nothing at the top.
    let default_size = (6.4e6
        * vd_core::geometry::visibility_factor(vd_core::geometry::VISIBILITY_THETA_MIN_RAD))
    .round() as u64;
    assert_eq!(
        regions.set_child_reach(STATIC_SAME, at(1, 11), default_size, 0),
        ReachOutcome::Applied { own_changed: false }
    );
    // A grandchild and a stranger are not this realm's to re-band.
    assert_eq!(
        regions.set_child_reach(MOVER, at(1, 12), 1, 1),
        ReachOutcome::NotMine
    );
    assert_eq!(
        regions.set_child_reach(RealmId::Planet(99), at(1, 12), 1, 1),
        ReachOutcome::NotMine
    );
    // Releasing the bright child drops its term: the parent's reach by light falls back.
    regions.release_child(STATIC_FAR);
    let (_, light2) = regions.own_reach(OWN_REALM, None);
    assert!(light2 < light1);
    // A table with no own realm named refuses every statement.
    let mut nameless = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(STATIC_FAR, 3.0e10),
    ]);
    assert_eq!(
        nameless.set_child_reach(STATIC_FAR, at(1, 1), 1, 1),
        ReachOutcome::NotMine
    );
}

/// Every flow `sent` carries toward `to`, decoded.
fn flows_to(sent: &[(NodeId, MsgClass, Vec<u8>)], to: NodeId) -> Vec<InterShardFlow> {
    sent.iter()
        .filter(|(node, _, _)| *node == to)
        .filter_map(|(_, _, bytes)| postcard::from_bytes::<InterShardFlow>(bytes).ok())
        .collect()
}

/// The emitter and the consumer on one rig: the realm states its reach upward once, says nothing
/// while it holds, and restates when a child's statement moves its own fold; a child's statement
/// through the three guards.
#[test]
fn a_realm_states_its_reach_once_and_again_when_a_child_moves_it_and_guards_its_children() {
    const PARENT: NodeId = NodeId(40);
    const CHILD_NODE: NodeId = NodeId(41);
    let mut rig = Rig::new();
    rig.grant_realm();
    let earth_luma = 0.3 * 6.4e6_f64.powi(2) / (4.0 * AU_M * AU_M);
    let luma = BTreeMap::from([(OTHER_REALM, (7_u8, earth_luma))]);
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(OTHER_REALM, 3.0e10),
    ])
    .with_child_light(&luma)
    .with_own_realm(OWN_REALM);
    rig.world.insert_resource(ParentRealmNode(Some(PARENT)));
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, CHILD_NODE);
    // Tick one: the reach goes up, once, to the parent's node.
    let sent = rig.tick(vec![]);
    let stated: Vec<InterShardFlow> = flows_to(&sent, PARENT)
        .into_iter()
        .filter(|f| matches!(f, InterShardFlow::ReachStated(_)))
        .collect();
    assert_eq!(stated.len(), 1, "{stated:?}");
    let InterShardFlow::ReachStated(first) = &stated[0] else {
        unreachable!()
    };
    assert_eq!(first.child.lowered(), OWN_REALM);
    assert!(
        first.light_reach_m as f64 > 3.0e10 + 45.0 * AU_M,
        "{}",
        first.light_reach_m
    );
    assert_eq!(rig.world.resource::<StubStats>().reach_sent, 1);
    assert_eq!(
        rig.world.resource::<StatedReach>().0,
        Some((first.size_reach_m, first.light_reach_m))
    );
    // Tick two: nothing changed, nothing said.
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, PARENT)
            .iter()
            .all(|f| !matches!(f, InterShardFlow::ReachStated(_)))
    );
    // The child states a far bigger reach by light: applied, and the realm restates upward.
    let coord = rig
        .world
        .resource::<RealmRegions>()
        .coord_of(OTHER_REALM)
        .expect("rostered");
    let statement = |fence: u64, at: u64, light: u64| {
        InterShardFlow::ReachStated(vd_wire::intershard::ReachStated {
            child: coord.clone(),
            child_fence: Fence(fence),
            at: UniverseTick(at),
            size_reach_m: 1,
            light_reach_m: light,
        })
    };
    let sent = rig.tick(vec![wire_msg(
        CHILD_NODE,
        MsgClass::Saga,
        &statement(1, 5, 9_000_000_000_000_000),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().reach_received, 1);
    let restated: Vec<InterShardFlow> = flows_to(&sent, PARENT)
        .into_iter()
        .filter(|f| matches!(f, InterShardFlow::ReachStated(_)))
        .collect();
    assert_eq!(restated.len(), 1, "the parent hears the new fold");
    // Stale: older than the held statement. Unattested: from a node the directory does not place
    // there. Misrouted: a coord whose parent is not this realm.
    rig.tick(vec![wire_msg(
        CHILD_NODE,
        MsgClass::Saga,
        &statement(1, 4, 1),
    )]);
    rig.tick(vec![wire_msg(
        NodeId(99),
        MsgClass::Saga,
        &statement(1, 6, 1),
    )]);
    let stranger = InterShardFlow::ReachStated(vd_wire::intershard::ReachStated {
        child: StubConfig::root_coord(RealmId::Galaxy(1))
            .child(vd_core::worldgen::level_of(OTHER_REALM)),
        child_fence: Fence(1),
        at: UniverseTick(7),
        size_reach_m: 1,
        light_reach_m: 1,
    });
    rig.tick(vec![wire_msg(CHILD_NODE, MsgClass::Saga, &stranger)]);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(
        (
            stats.reach_stale,
            stats.reach_unattested,
            stats.reach_misrouted
        ),
        (1, 1, 1)
    );
    // ATTESTED AS MINE AND ADDRESSED TO ME, YET NOT ON MY ROSTER: the adopt has not landed here
    // yet. The statement is counted with the misroutes; the hull restates on its next lineage.
    let newcomer = RealmId::Planet(43);
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(newcomer, CHILD_NODE);
    let unrostered = InterShardFlow::ReachStated(vd_wire::intershard::ReachStated {
        child: StubConfig::root_coord(OWN_REALM).child(vd_core::worldgen::level_of(newcomer)),
        child_fence: Fence(1),
        at: UniverseTick(8),
        size_reach_m: 1,
        light_reach_m: 1,
    });
    rig.tick(vec![wire_msg(CHILD_NODE, MsgClass::Saga, &unrostered)]);
    assert_eq!(rig.world.resource::<StubStats>().reach_misrouted, 2);
}

#[test]
fn a_child_restating_the_reach_it_already_holds_leaves_its_band_exactly_as_it_is() {
    // A hull that states the same reach twice must not disturb its band: the second statement
    // computes the same radius, so the band is already the answer and nothing is re-indexed. The
    // re-statement is still APPLIED — it is not stale — it simply moves nothing.
    let mut regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        planet(STATIC_FAR, 3.0e10),
    ])
    .with_own_realm(OWN_REALM);
    let at = |f: u64, t: u64| (Fence(f), UniverseTick(t));
    assert_eq!(
        regions.set_child_reach(STATIC_FAR, at(1, 10), 4_000_000_000, 0),
        ReachOutcome::Applied { own_changed: true }
    );
    let band = regions
        .direct_child(OWN_REALM, STATIC_FAR)
        .expect("rostered")
        .aoi;
    assert_eq!(
        regions.set_child_reach(STATIC_FAR, at(1, 11), 4_000_000_000, 0),
        ReachOutcome::Applied { own_changed: false }
    );
    assert_eq!(
        regions
            .direct_child(OWN_REALM, STATIC_FAR)
            .expect("rostered")
            .aoi,
        band,
        "the same reach twice leaves the band alone"
    );
}

#[test]
fn a_moving_childs_band_follows_its_stated_reach_and_its_leaf_still_never_joins_the_index() {
    // A mover is ALWAYS asked, never looked up, so its band is a number the parent keeps and its
    // leaf stays out of the range index. Stating a reach must widen the band all the same — the
    // hull that flies is exactly the one whose reach changes.
    let mut regions = RealmRegions::new(vec![root_region(), own_region(), planet(MOVER, 3.0e10)])
        .with_moving_children(BTreeMap::from([(MOVER, a_motion())]))
        .with_own_realm(OWN_REALM);
    assert!(
        regions.child_moves(MOVER),
        "a mover, by the table's own word"
    );
    assert!(!regions.child_moves(STATIC_FAR));
    let out = regions.set_child_reach(MOVER, (Fence(1), UniverseTick(10)), 9_000_000_000, 0);
    assert_eq!(out, ReachOutcome::Applied { own_changed: true });
    assert_eq!(
        regions
            .direct_child(OWN_REALM, MOVER)
            .expect("rostered")
            .aoi
            .spin_up_r_m(),
        9.0e9,
        "the band followed the statement"
    );
    assert!(
        !regions.aoi_index().answers_for(MOVER),
        "and the mover is still asked, never looked up"
    );
}

#[test]
fn a_shard_that_names_no_realm_adopts_nobody() {
    // A shard with no realm of its own authors nobody, so a hull handed to it is refused in
    // silence — the same refusal the index makes.
    let mut nameless = RealmRegions::new(vec![root_region(), own_region()]);
    nameless.adopt_child(live(STATIC_FAR, DVec3::new(30_000.0, 0.0, 0.0), 100.0));
    assert!(nameless.coord_of(STATIC_FAR).is_none());
    assert_eq!(nameless.direct_children(OWN_REALM).count(), 0);
}

#[test]
fn releasing_a_row_ahead_of_a_mover_re_points_the_movers_that_sat_at_the_end() {
    // The mover is the LAST row: releasing an earlier row swaps the mover into that slot, and the
    // mover list must follow it. If it did not, the book's per-tick overlay would author the wrong
    // child — a hull flying on a planet's row.
    let mut regions = RealmRegions::new(vec![root_region(), own_region()])
        .with_moving_children(BTreeMap::from([(MOVER, a_motion())]))
        .with_own_realm(OWN_REALM);
    regions.adopt_child(live(STATIC_FAR, DVec3::new(30_000.0, 0.0, 0.0), 100.0));
    regions.adopt_child(live(STATIC_SAME, DVec3::new(0.0, 30_000.0, 0.0), 100.0));
    regions.adopt_child(live(MOVER, DVec3::new(2_000.0, 0.0, 0.0), 100.0));
    regions.release_child(STATIC_FAR);
    let driven = DrivenChildren::default();
    assert!(
        regions
            .moving_children_of(OWN_REALM, &driven)
            .contains(&MOVER)
    );
    let book = regions.author_book_driven(OWN_REALM, 20.0, UniverseTick(5), &driven);
    assert!(book.of(frame_of(MOVER)).is_some(), "the overlay found it");
    assert!(regions.direct_child(OWN_REALM, STATIC_SAME).is_some());
}
