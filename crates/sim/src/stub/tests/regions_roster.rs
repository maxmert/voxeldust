//! THE ROSTER'S EDGES (HR5, 2026-09-04): every arm of adopt, re-parent and release on the region
//! table, driven through the public API alone — a moving child, a live band, a root row, a row
//! that is not the last, a re-driven release. The six-shard crossing fixture flies the happy path;
//! these pin the edges the coverage gate found bare.

use super::*;
use crate::stub::drive::{DrivenChild, DrivenChildren, DrivenState};
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
