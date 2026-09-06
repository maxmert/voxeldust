//! THE INTEREST: which of this shard's occupants each observer is shipped (D-9, foundation slice 2,
//! owner-approved 2026-09-05).
//!
//! Owns: the cell grid over the shard's own frame, the per-observer hold with its hysteresis, and
//! the per-tick plan — one body per cell, the observers that hold that cell, and the occupants that
//! left an observer's interest and must be told to it. Pure: ticks and positions in, a plan out.
//!
//! Does NOT own: the rows (frames.rs restates them), the wire (frames.rs ships the plan), or the
//! reach formula (`vd_core::geometry::visibility_reach_m`, the one every realm's reach reads) and
//! the drawable angle (`vd_core::geometry::drawable_theta_min_rad`).
//!
//! THE RULE. An occupant is shipped to an observer inside the occupant's reach — its look at THE
//! DRAWABLE ANGLE, one pixel at the reference view (owner 2026-09-06: a person is worth a row long
//! before a planet is worth a shard, so the wake angle is the wrong bar here) — plus a lead that
//! grows with the closing speed: the fastest row's speed plus the observer's own, over the
//! interpolation buffer plus one tick. The realm's occupants are sorted into cubes of one reach; an observer TAKES every cube
//! within its step count, and KEEPS a cube one step further out while it holds an occupant the
//! observer was shown last tick — so nobody flaps at a boundary, and a figure you were shown stays
//! shown until it is clearly out of reach.
//!
//! Example: three hundred people at a station. The pilot at the far dock stands in one cube; the
//! twelve people near the dock stand in it or next to it; those twelve are the rows the pilot
//! receives, in one body shared with the eleven others standing there.
//!
//! COST. Sorting is one pass over the occupants. Each observer then reads the cubes in a slab of
//! the key order around itself; with occupants spread over many cubes that is the slab's width,
//! never the whole realm. The state held per observer is bounded by its neighbours, never by the
//! population (SL9's spirit, applied to occupants).

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Resource;
use vd_core::glam::DVec3;
use vd_core::{EntityId, NodeId, SessionId};

/// One cube of the interest grid, by its integer coordinates in the shard's own frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellKey(pub i64, pub i64, pub i64);

/// The cube a position falls in, for cubes of side `side_m`.
#[must_use]
pub fn cell_of(pos_m: DVec3, side_m: f64) -> CellKey {
    CellKey(
        (pos_m.x / side_m).floor() as i64,
        (pos_m.y / side_m).floor() as i64,
        (pos_m.z / side_m).floor() as i64,
    )
}

/// The distance between two cubes in steps: the largest axis difference (a cube one step away
/// touches, by a face, an edge or a corner).
#[must_use]
pub fn cell_steps(a: CellKey, b: CellKey) -> i64 {
    (a.0 - b.0)
        .abs()
        .max((a.1 - b.1).abs())
        .max((a.2 - b.2).abs())
}

/// The interest geometry, all derived — the cube side is one reach, the horizon is the time a row
/// must be shipped ahead of the moment it comes within reach.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterestRule {
    /// One reach: the largest occupant look's visibility reach at the drawable angle.
    pub side_m: f64,
    /// The interpolation buffer plus one tick, in seconds.
    pub horizon_s: f64,
}

/// Derive the rule from the largest look an occupant of this realm draws as and the tick.
#[must_use]
pub fn interest_rule(max_look_extent_m: f64, tick_dt_s: f64) -> InterestRule {
    let side_m = vd_core::geometry::visibility_reach_m(
        max_look_extent_m,
        vd_core::geometry::drawable_theta_min_rad(),
    );
    let horizon_s = vd_wire::channels::INTERP_BUFFER_MS / 1000.0 + tick_dt_s;
    InterestRule { side_m, horizon_s }
}

/// How many steps an observer takes around its own cube: one for the reach itself, plus what the
/// closing speed covers within the horizon. Example: a standing crowd closes at zero and takes one
/// step; an observer walking at a kilometre a second toward a runner at the same speed, with an
/// 870 m cube, takes one more.
#[must_use]
pub fn take_steps(rule: InterestRule, closing_speed_mps: f64) -> i64 {
    1 + ((closing_speed_mps * rule.horizon_s) / rule.side_m).ceil() as i64
}

/// One observer this tick: the session its client rides, its gateway, its cube and its speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observer {
    pub session: SessionId,
    pub gateway: NodeId,
    pub cell: CellKey,
    pub speed_mps: f64,
}

/// One placed row this tick: which occupant, in which cube, at what speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedRow {
    pub entity: EntityId,
    pub cell: CellKey,
    pub speed_mps: f64,
}

/// The per-observer hold, carried tick to tick: the occupants each observer was shipped last tick —
/// the hysteresis (a shown figure is kept one step further out) and the removal diff in one set.
/// Keyed by the observer's session, bounded by its neighbours.
#[derive(Resource, Debug, Default, PartialEq)]
pub struct InterestHeld {
    /// The cube side the hold was computed for; a different side re-keys every cube, so the hold
    /// is cleared and counted when it changes.
    pub side_m: f64,
    pub delivered: BTreeMap<SessionId, BTreeSet<EntityId>>,
}

/// What one tick ships: per cube, the observers that hold it (sorted by session), and the
/// (observer, occupant) pairs that left an observer's interest.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct InterestPlan {
    pub recipients: BTreeMap<CellKey, Vec<SessionId>>,
    pub removals: Vec<(SessionId, EntityId)>,
}

/// THE PLAN. Monomorphic; every branch is covered in `tests/interest.rs`.
#[must_use]
pub fn plan_interest(
    rule: InterestRule,
    observers: &[Observer],
    rows: &[PlacedRow],
    held: &mut InterestHeld,
) -> InterestPlan {
    // The cubes that hold rows this tick, and the fastest row (the closing speed's other half).
    let mut cells: BTreeMap<CellKey, Vec<EntityId>> = BTreeMap::new();
    let mut fastest_row = 0.0_f64;
    for row in rows {
        cells.entry(row.cell).or_default().push(row.entity);
        fastest_row = fastest_row.max(row.speed_mps);
    }
    let mut plan = InterestPlan::default();
    let mut present: BTreeSet<SessionId> = BTreeSet::new();
    for obs in observers {
        present.insert(obs.session);
        let take = take_steps(rule, obs.speed_mps + fastest_row);
        let drop = take + 1;
        let delivered_last = held.delivered.remove(&obs.session).unwrap_or_default();
        // Held: every cube with rows within `take` steps, and every cube within `drop` steps that
        // holds an occupant this observer was shown last tick — read from the slab of keys whose
        // first coordinate is within `drop`, never the whole map.
        let now_held: Vec<CellKey> = cells
            .range(
                CellKey(obs.cell.0 - drop, i64::MIN, i64::MIN)
                    ..=CellKey(obs.cell.0 + drop, i64::MAX, i64::MAX),
            )
            .filter(|(cell, rows)| {
                let steps = cell_steps(**cell, obs.cell);
                (steps <= take)
                    | ((steps <= drop) & rows.iter().any(|e| delivered_last.contains(e)))
            })
            .map(|(cell, _)| *cell)
            .collect();
        let delivered_now: BTreeSet<EntityId> = now_held
            .iter()
            .flat_map(|cell| cells[cell].iter().copied())
            .collect();
        for entity in delivered_last.difference(&delivered_now) {
            plan.removals.push((obs.session, *entity));
        }
        for cell in &now_held {
            plan.recipients.entry(*cell).or_default().push(obs.session);
        }
        held.delivered.insert(obs.session, delivered_now);
    }
    // An observer that left the shard takes its hold with it: its client is gone or crossed, and
    // the realm it stands in now ships it what it sees.
    held.delivered
        .retain(|session, _| present.contains(session));
    plan
}
