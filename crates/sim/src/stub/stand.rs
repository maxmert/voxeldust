//! ★ A BODY STANDS ON ITS REALM'S GROUND (the fine ground's way-in design §4.3 and its Q6, accepted by ruling B81; built
//! 2026-10-13, when the owner found hull 15 under the ground ruling B93 solved).
//!
//! A realm with ground (a planet, a moon) authors where its parked children stand. Their berths were written over the
//! ground of their day — hull 15's over the 8 km ground, a metre under its belly — and a finer ground solved later may
//! stand higher or lower there. The realm reads its own LIVE ground under each parked child (the fine tiles it holds,
//! the coarser ground elsewhere: what every client draws there) and stands the child on it, its belly
//! [`STAND_CLEARANCE_M`] over the highest ground under its footprint. The sim never names the generator: the composition
//! root injects a [`Ground`] (the shape of [`super::fine_ship::FineTiles`]).
//!
//! - **WHO STANDS.** A parked child (a berth this realm read at boot, or a child it adopted) stands when its belly is in
//!   the ground or within the clearance over it — a hull parked on a plain. A child parked high (a ship stopped over the
//!   planet) hovers and is never moved. A child the realm stood on its ground stays on it as the ground changes, lifted
//!   or lowered, until it flies: until its drive state leaves the place it was stood on, or moves.
//! - **WHEN.** Once, when the ground arrives (every parked child: the realm's first statement of where they stand), and
//!   then only for the children over a patch whose ground changed — a lookup, never a walk of every child (SL9).
//! - **HOW FAST.** A change after the first statement follows the ground over [`STAND_RAMP_S`], the fine ground's own
//!   arrival ramp (ruling B18 Q7), so the body and the ground under it move together.
//! - **WHAT IT WRITES.** The parent's own lift for the child (SL1: the parent is the only writer), added to the centre it
//!   authored; a drive state at rest on that place moves with it.

use super::{RealmRegions, StubConfig};
use crate::runtime::ClockSample;
use crate::stub::drive::DrivenChildren;
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::geometry::Boundary;
use vd_core::glam::{DQuat, DVec3};
use vd_core::kinematics::secs_since_epoch;
use vd_core::pose::RealmId;

/// The belly's clearance over the highest ground under a standing body's footprint, metres: the drawn surface stands
/// within the extractor's step of the field it is read from, and a metre keeps the belly out of it (the window stand's
/// own clearance, `examples/window_stand.rs`).
pub const STAND_CLEARANCE_M: f64 = 1.0;

/// The time a standing body takes to follow its ground to a new height, seconds: the fine ground's own arrival ramp
/// (ruling B18 Q7), so the body and the ground under it move together.
pub const STAND_RAMP_S: f64 = 0.5;

/// The footprint's samples along a body's length (its own `z`) and across it (its own `x`), as fractions of its half
/// extents — the window stand's five by three.
const ALONG: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
const ACROSS: [f64; 3] = [-1.0, 0.0, 1.0];

/// The realm's own live ground, injected by the composition root; the sim reads radii.
pub trait Ground: Send + Sync {
    /// The live ground's radius along the unit direction `dir` (this realm's own frame), metres from its centre — the
    /// ground every client draws there; `None` where none is held.
    fn radius_m(&self, dir: DVec3) -> Option<f64>;
    /// The patches the read along `dir` depends on: opaque ids of the units the ground changes in.
    fn patches(&self, dir: DVec3) -> Vec<u64>;
    /// The patches whose ground changed since the last call.
    fn changed(&self) -> Vec<u64>;
}

/// The injected ground: `None` until the realm's ground is here (a body still solving, a realm with no ground, a test
/// rig).
#[derive(Resource, Default)]
pub struct GroundSource(pub Option<Box<dyn Ground>>);

impl std::fmt::Debug for GroundSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroundSource")
            .field("present", &self.0.is_some())
            .finish()
    }
}

/// Where a parked child stands, as the realm holds it ([`RealmRegions::parked_at`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parked {
    /// Its authored centre, metres in this realm's own frame.
    pub centre: DVec3,
    /// The lift this realm stands it on its ground with.
    pub lift: DVec3,
    /// The attitude it was parked in.
    pub facing: DQuat,
    /// Its bound.
    pub shape: Boundary,
}

/// One child following its ground: from one lift to another over [`STAND_RAMP_S`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ramp {
    from: DVec3,
    to: DVec3,
    start_s: f64,
}

/// ★ THE REALM'S STANDING BOOK: which parked children stand on its ground, the patches each one's footprint reads (the
/// lookup a changed patch is answered by), and the children following their ground now.
#[derive(Resource, Debug, Default)]
pub struct Standing {
    seeded: bool,
    on_patch: BTreeMap<u64, BTreeSet<RealmId>>,
    patches_of: BTreeMap<RealmId, Vec<u64>>,
    standing: BTreeSet<RealmId>,
    ramps: BTreeMap<RealmId, Ramp>,
}

impl Standing {
    /// Whether this realm stands `child` on its ground.
    #[must_use]
    pub fn stands(&self, child: RealmId) -> bool {
        self.standing.contains(&child)
    }

    /// The children over any of `patches`, each once.
    fn over(&self, patches: &[u64]) -> Vec<RealmId> {
        patches
            .iter()
            .filter_map(|p| self.on_patch.get(p))
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// The child leaves the book: it is gone, or it flies.
    fn forget(&mut self, child: RealmId) {
        self.standing.remove(&child);
        self.ramps.remove(&child);
        for p in self.patches_of.remove(&child).unwrap_or_default() {
            if let Some(set) = self.on_patch.get_mut(&p) {
                set.remove(&child);
            }
        }
    }

    /// ★ ONE CHILD AGAINST ITS GROUND: stood on it at once on the `first` statement, else followed over the ramp.
    fn settle(
        &mut self,
        ground: &dyn Ground,
        regions: &mut RealmRegions,
        driven: &mut DrivenChildren,
        child: RealmId,
        now_s: f64,
        first: bool,
    ) {
        let Some(parked) = regions.parked_at(child) else {
            self.forget(child);
            return;
        };
        if flies(driven, child, parked.lift) {
            self.forget(child);
            return;
        }
        let at = parked.centre + parked.lift;
        let up = at.normalize_or_zero();
        if up == DVec3::ZERO {
            return; // a child at the realm's very centre stands on no ground
        }
        let (half, axes) = box_of(parked.shape, parked.facing);
        let dirs = footprint(at, half, axes);
        if first {
            let patches: Vec<u64> = dirs
                .iter()
                .flat_map(|d| ground.patches(*d))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            for p in &patches {
                self.on_patch.entry(*p).or_default().insert(child);
            }
            self.patches_of.insert(child, patches);
        }
        let Some(highest) = dirs
            .iter()
            .map(|d| ground.radius_m(*d))
            .try_fold(f64::NEG_INFINITY, |hi, r| r.map(|r| hi.max(r)))
        else {
            return; // some of its ground is not held: it waits for the change that brings it
        };
        let need = highest + support(half, axes, up) + STAND_CLEARANCE_M;
        if !(self.standing.contains(&child) | (at.length() <= need + STAND_CLEARANCE_M)) {
            return; // it hovers over its ground: parked high, never moved
        }
        self.standing.insert(child);
        let to = up * need - parked.centre;
        tracing::info!(
            ?child,
            from_radius_m = at.length(),
            to_radius_m = need,
            ground_m = highest,
            first,
            "a parked child stands on its realm's ground (ruling B81 Q6)"
        );
        if first {
            put(regions, driven, child, to);
            land(regions, driven, child);
        } else {
            self.ramps.insert(
                child,
                Ramp {
                    from: parked.lift,
                    to,
                    start_s: now_s,
                },
            );
        }
    }

    /// The ramps one tick on; a child that flew mid-ramp leaves the book.
    fn step(&mut self, regions: &mut RealmRegions, driven: &mut DrivenChildren, now_s: f64) {
        let (mut done, mut flown) = (Vec::new(), Vec::new());
        for (&child, ramp) in &self.ramps {
            if flies(driven, child, regions.lift_of(child)) {
                flown.push(child);
                continue;
            }
            let t = ((now_s - ramp.start_s) / STAND_RAMP_S).clamp(0.0, 1.0);
            put(regions, driven, child, ramp.from.lerp(ramp.to, t));
            if t >= 1.0 {
                done.push(child);
            }
        }
        for child in done {
            self.ramps.remove(&child);
            land(regions, driven, child);
        }
        for child in flown {
            self.forget(child);
        }
    }
}

/// Whether `child` flies: its drive state has left the place it was stood on (`lift`), or moves.
fn flies(driven: &DrivenChildren, child: RealmId, lift: DVec3) -> bool {
    driven
        .state_of(child)
        .is_some_and(|s| (s.pos_m != lift) | (s.vel_mps != DVec3::ZERO))
}

/// The child stood at `lift`: the realm's lift, and its drive state at rest with it.
fn put(regions: &mut RealmRegions, driven: &mut DrivenChildren, child: RealmId, lift: DVec3) {
    regions.set_lift(child, lift);
    if let Some(held) = driven.0.get_mut(&child) {
        held.state.pos_m = lift;
    }
}

/// The stand lands: the lift folds into the child's authored centre ([`RealmRegions::fold_lift`]), and its drive state
/// at rest stands at that centre.
fn land(regions: &mut RealmRegions, driven: &mut DrivenChildren, child: RealmId) {
    regions.fold_lift(child);
    if let Some(held) = driven.0.get_mut(&child) {
        held.state.pos_m = DVec3::ZERO;
    }
}

/// A bound as a box in this realm's frame: its half extents and its three axes (a shell its enclosing cube).
fn box_of(shape: Boundary, facing: DQuat) -> (DVec3, [DVec3; 3]) {
    let (half, orient) = match shape {
        Boundary::Shell { r } => (DVec3::splat(r), facing),
        Boundary::Aabb { half } => (half, facing),
        Boundary::Obb { half, orient } => (half, facing * orient),
    };
    (
        half,
        [orient * DVec3::X, orient * DVec3::Y, orient * DVec3::Z],
    )
}

/// How far the box reaches below its centre along `up`: its belly.
fn support(half: DVec3, axes: [DVec3; 3], up: DVec3) -> f64 {
    axes[0].dot(up).abs() * half.x + axes[1].dot(up).abs() * half.y + axes[2].dot(up).abs() * half.z
}

/// The directions under the box's footprint: five along its length, three across it.
fn footprint(at: DVec3, half: DVec3, axes: [DVec3; 3]) -> Vec<DVec3> {
    ALONG
        .iter()
        .flat_map(|a| {
            ACROSS
                .iter()
                .map(move |b| (at + axes[0] * (b * half.x) + axes[2] * (a * half.z)).normalize())
        })
        .collect()
}

/// ★ THE SYSTEM, before the placement writer each synced tick: the first statement of where every parked child stands
/// once the ground is here, then the children over the patches whose ground changed, then the ramps one tick on.
pub(crate) fn stand_on_ground(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    source: Res<GroundSource>,
    mut regions: ResMut<RealmRegions>,
    mut driven: ResMut<DrivenChildren>,
    mut standing: ResMut<Standing>,
) {
    let Some(ground) = source.0.as_deref() else {
        return;
    };
    let now_s = secs_since_epoch(clock.universe_tick.0, 1.0 / config.tick_dt_s);
    let changed = ground.changed();
    let first = !standing.seeded;
    let due = if first {
        standing.seeded = true;
        regions.parked_children()
    } else {
        standing.over(&changed)
    };
    for child in due {
        standing.settle(ground, &mut regions, &mut driven, child, now_s, first);
    }
    standing.step(&mut regions, &mut driven, now_s);
}
