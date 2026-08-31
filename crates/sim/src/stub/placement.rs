//! THE PLACEMENT LEDGER: the ONE writer, and the books every other lane reads.
//!
//! Owns: the per-tick authoring of a [`vd_core::placement::PlacementBook`] for every anchor this
//! shard can be asked about, and the two conversions between a book row and a stamped pose. ONE
//! writer publishes at the head of every synced tick; containment, crossing, AoI and the scene
//! lanes SELECT a book by an instant they already hold as data and read rows with no clock of their
//! own.
//!
//! Does NOT own: how anything moves. A mover's row runs an opaque closure injected at boot from a
//! crate this one has no edge to; a static child's row is its stored centre. Both arms write the
//! same kind of value into the same row and no reader can tell which ran — which is exactly SL4's
//! demand that a static flag may decide whether to recompute, never what anyone reads.

use super::{RealmRegions, StubConfig, StubStats};
use crate::runtime::ClockSample;
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::BTreeMap;
use vd_core::UniverseTick;
use vd_core::frame::FramePlacement;
use vd_core::geometry::RealmRegion;
use vd_core::glam::{DQuat, DVec3};
use vd_core::placement::{MotionFn, PlacementBook, PlacementLedger};
use vd_core::pose::{LatticePos, RealmId, StampedPose};

/// THE PLACEMENT LEDGER (the placement arc S2): every anchor this shard hosts × a bounded backward
/// window of exact per-instant [`PlacementBook`]s. ONE writer ([`author_placements`], at the head of
/// every synced tick) publishes; every consumer — containment, crossing, AoI, the scene lanes,
/// realm-frame authoring — SELECTS a book by an instant it already holds as data and reads rows with
/// no clock. `Res` everywhere downstream: Bevy's `Res`/`ResMut` split is itself a partial structural
/// guarantee that no consumer can write the store. SCOPE of the "one writer" fence, stated honestly
/// (batch review): the clippy `publish` ban binds THIS crate only (measured — see clippy.toml), and
/// this field is `pub` — the crates above the seam are held to it by the observation that none of
/// them names `Placements`, until the writer-in-node rework (D-PLACE-4) closes the surface.
#[derive(Resource, Debug, Default)]
pub struct Placements(pub PlacementLedger);

/// THE ONE PLACEMENT WRITER (the placement arc S2): at the head of every synced tick, author a
/// [`PlacementBook`] for every anchor this shard can be asked about — every region with a direct
/// child in its forest (its own realm, each co-hosted realm, and each ancestor whose visible child
/// chain it holds) plus every held realm (so a childless leaf still publishes an empty book and the
/// detector's head-selection is total). Everything downstream this tick reads THESE rows; nothing
/// re-derives a placement, so what a body is doing cannot decide what any consumer reads (SL4).
// THE ONE stated exemption from the crate-wide `publish` ban (clippy.toml): this IS the writer.
#[allow(clippy::disallowed_methods)]
pub(crate) fn author_placements(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    regions: Res<RealmRegions>,
    mut placements: ResMut<Placements>,
) {
    let tick_hz = 1.0 / config.tick_dt_s;
    for anchor in placement_anchors(&regions, &config) {
        let book = regions.author_book(anchor, tick_hz, clock.universe_tick);
        placements.0.publish(anchor, book);
    }
}

/// The anchors the writer authors for. Deterministic (forest order, then the held set), deduped.
pub(crate) fn placement_anchors(regions: &RealmRegions, config: &StubConfig) -> Vec<RealmId> {
    // ★ A SCAN INSIDE A SCAN, EVERY TICK (fixed 2026-08-30). This asked, for EVERY region, whether
    // ANY region named it as a parent — a full pass over the forest per region.
    //
    // MEASURED on THE world: a galaxy shard holds 279 380 direct children, so this is about
    // 78 000 000 000 comparisons PER TICK. A six-beat measurement did not finish its FIRST beat in
    // twelve minutes, and this was the whole of it.
    //
    // "Which realms have children" is precisely what the region table's parent index holds, and its
    // KEYS are the answer. SL9: a lookup, never a scan.
    let mut anchors: Vec<RealmId> = regions.parents_with_children().collect();
    for &held in &config.held_realms {
        anchors.push(held);
    }
    anchors.sort_unstable();
    anchors.dedup();
    anchors
}

/// Author ONE child's placement row (the placement arc) — THE single motion test in this crate: a
/// mover's row RUNS its injected [`MotionFn`] at the book's instant, a static child's row is its
/// stored region `center`. Both arms write the SAME kind of value into the SAME row; nobody
/// downstream can tell which arm ran — which is exactly SL4's demand (a static flag may decide
/// WHETHER TO RECOMPUTE, never WHAT ANYONE READS). And the mover arm cannot even NAME what it runs:
/// the closure was injected at boot from the motion crate, which this crate has no edge to.
/// Monomorphic — the `match` is covered once here (a mover test + a static test), NOT per generic
/// monomorphization (HR5).
///
/// The static arm carries the WHOLE stored position, integer cell anchor included. It used to read
/// only the f64 remainder and silently threw the anchor away — invisible while every region is
/// authored at cell ZERO, and it would have stayed invisible until the first child placed a
/// cell-block out drew at the wrong place.
pub(crate) fn placement_row(
    moving: &BTreeMap<RealmId, MotionFn>,
    r: &RealmRegion,
    secs: f64,
) -> FramePlacement {
    match moving.get(&r.realm) {
        Some(motion) => (motion.0)(secs),
        None => FramePlacement {
            // The centre AS THE PARENT AUTHORED IT — this row is a placement in the parent's own
            // frame, so it carries the parent's own numbers verbatim, unit and all.
            origin_cell: r.center.in_parents_frame().cell(),
            origin: r.center.in_parents_frame().offset(),
            velocity: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        },
    }
}

/// A book row read back as a stamped pose in the book's own anchor frame, at the book's own instant —
/// the ONE conversion between the two shapes of "where this child is" (the inverse of
/// [`placement_of`]), so no lane can end up stamping a different frame or instant onto the same fact.
pub(crate) fn pose_of_row(book: &PlacementBook, at: FramePlacement) -> StampedPose {
    StampedPose {
        frame: book.anchor(),
        pos: LatticePos::at(at.origin_cell, at.origin),
        vel: at.velocity,
        orient: DQuat::IDENTITY,
        universe_tick: book.at(),
    }
}

/// Select the book for an ARRIVING HAND-OFF's instant — clamped to the head in BOTH directions when
/// the window no longer (or does not yet) retain it, because a hand-off is a RETAINED, RETRIED message
/// whose pose stamp never changes across redeliveries: one missed delivery plus an exact-instant law
/// wedged the saga PERMANENTLY (MEASURED at process tier, 2026-08-14 — `at_tick=2105` refused against
/// a head that had advanced to 2450+, once per redelivery, forever; the demand round-trip froze on
/// it). The receiver reads its world of NOW instead: the occupant lands relative to where the realm
/// IS, not where it was at the stamp — the same "an occupant rides its realm" rule the up-observation
/// ride measurement pinned, and the same one-instant rule the flush follows. Both directions are
/// COUNTED with their size; `None` only before the writer's first pass (no head at all).
pub(crate) fn arrival_book<'a>(
    placements: &'a PlacementLedger,
    anchor: RealmId,
    stamp: UniverseTick,
    stats: &mut StubStats,
) -> Option<(&'a PlacementBook, UniverseTick)> {
    if let Ok(book) = placements.at(anchor, stamp) {
        return Some((book, stamp));
    }
    let head = placements.head(anchor)?;
    stats.placement_skew_clamped += 1;
    // PER-DIRECTION gauges (batch review): a forward stamp measures clock-phase lead (`span_ahead`);
    // a backward one measures redelivery staleness. One |Δ| magnitude conflated the two, and the
    // arrival lane's staleness (345 ticks at the measured wedge) drowned the forward bound.
    if stamp > head.at() {
        stats.placement_skew_ahead_max_ticks = stats
            .placement_skew_ahead_max_ticks
            .max(stamp.0.saturating_sub(head.at().0));
    } else {
        stats.placement_skew_behind_max_ticks = stats
            .placement_skew_behind_max_ticks
            .max(head.at().0.saturating_sub(stamp.0));
    }
    Some((head, head.at()))
}

/// The anchor whose authored book measures a subject standing in `owning`: the realm itself when this
/// shard hosts it (a co-hosted realm answers for its own children), else the shard's PRIMARY realm —
/// the fallback for a subject in a realm this shard carries no region for (a fault, counted as
/// `containment_prior_unhosted` by the scan). Monomorphic so both arms are covered once (HR5).
#[must_use]
pub(crate) fn book_anchor(
    ix_of: &BTreeMap<RealmId, usize>,
    config_realm: RealmId,
    owning: RealmId,
) -> RealmId {
    if owning != config_realm && ix_of.contains_key(&owning) {
        owning
    } else {
        config_realm
    }
}
