//! THE RETAINED GHOST: a dot this shard no longer owns but still draws, and the band that ends it.
//!
//! Owns: the destination-side registry of who retains a ghost of an entity we own, the source-side
//! lifecycle arms that spawn and despawn one, and the band-exit sweep that tears the pairing down
//! once the subject has travelled clear of the boundary it crossed.
//!
//! Does NOT own: the ghost's pose. Since the fed-ghost lane died there is exactly one pose truth —
//! the dot itself — and this module holds distances read live from it, never a copy. The overlap
//! band is also the retained skeleton the cross-boundary collision seam will hang off; it carries no
//! collision behaviour today and states so rather than pretending.

use super::{
    Dot, Dots, HandoffHolds, HoldRole, StubConfig, StubStats, close_hold_at_fence,
    push_entity_removed,
};
use crate::authority::Authority;
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::BTreeMap;
use vd_core::geometry::OverlapBand;
use vd_core::pose::{LatticePos, StampedPose};
use vd_core::{EntityId, Fence, NodeId};
use vd_wire::intershard::{GhostFlow, InterShardFlow};

/// One ghost-neighbor this shard registered at promote (1d.5b.3b → slice F): the node that RETAINS
/// a ghost dot of an entity we OWN. Since slice F no pose is ever fed to it — the entry exists only
/// to drive the band-exit `Despawn` (and it is the P5 cross-boundary collision seam's retained
/// skeleton). Holds NO pose/authority — distances are read LIVE from the owned `Dot` (FG-2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GhostNeighbor {
    /// The node hosting the ghost (the transfer SOURCE in 1d.5b.3b; any band-neighbor at P-band).
    pub source: NodeId,
    /// The boundary ANCHOR (1d.5b.3c): the fixed pose POSITION at which the entity crossed into this
    /// realm, captured once at promote. The dest measures the owned entity's distance from here each
    /// tick; when it exits the overlap band the dest Despawns the ghost (band-exit). IMMUTABLE
    /// bookkeeping — a reference point, NEVER a live pose (FG-2: the live pose is read from the `Dot`).
    /// Interim: a local-boundary approximation (the stub has no realm-center/SOI geometry); the real
    /// SOI band anchors at the realm center (`for_planet_soi`/`for_system_soi`, P4/P5).
    ///
    /// A POSITION, kept whole. It was a bare metre triple, and the distance from it was taken against
    /// another bare triple — so once the integrator began folding a position's whole-number part out, both
    /// terms were sub-millimetre remainders, the walk from here measured ~0 forever, and the ghost this
    /// anchor exists to retire could never be retired.
    pub anchor: LatticePos,
}

/// DEST-side ghost FEED registry (1d.5b.3b): for each entity this shard OWNS, the ghost-host
/// neighbor(s) to feed `GhostFlow::Delta`. Populated by the relocated promote (`on_saga_promote`)
/// in 1d.5b.3b — the dest, on becoming owner, registers the transfer source as a ghost-host. The
/// SAME machinery serves the future band-driven multi-neighbor ghost (the owner fans `Delta` to
/// every overlap neighbor — the registration generalizes to a neighbor SET, an additive extension).
/// Torn down on band-exit `Despawn` (1d.5b.3c). One entry per owned, ghosted entity.
#[derive(Resource, Debug, Default)]
pub struct GhostColliderRegistration(pub BTreeMap<EntityId, GhostNeighbor>);

/// SOURCE-side ghost lifecycle consumer (1d.5b.3b → slice F): the dest (new owner) drives
/// `GhostFlow` to this shard (the ghost-host) — `SpawnV2` is the pose-free take-over proof (closes
/// the hold, stops the retained ghost's emit, evicts the bystanders' figures) and `Despawn` ends
/// the lifecycle (1d.5b.3c band-exit tears the retained DOT out). NOTHING here ever writes a pose:
/// `refresh_source_ghost` — the §4u corruption's LAST writer, the fn that painted a foreign-frame
/// pose into a promotable dot — died with the feed (the tombstoned `Spawn`/`Delta` arms are counted
/// no-ops). A malformed body is counted + dropped, never mis-applied. Monomorphic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_ghost_flow(
    bytes: &[u8],
    clock: &ClockSample,
    realm_fence: Option<Fence>,
    dots: &mut Dots,
    holds: &mut HandoffHolds,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let flow = match postcard::from_bytes::<InterShardFlow>(bytes) {
        Ok(InterShardFlow::Ghost(flow)) => flow,
        // Any other arm misrouted onto a Ghost carrier, or a decode failure: counted + dropped.
        _ => {
            stats.undecodable += 1;
            return;
        }
    };
    match flow {
        // ★TOMBSTONED arms (slice F): the old pose-carrying proof and the 20 Hz pose feed. Their
        // frames still decode (reserved discriminants) and land here as counted no-ops — the pose
        // they carry is exactly the foreign-frame write this slice deleted, so nothing may apply it.
        GhostFlow::Spawn { .. } | GhostFlow::Delta { .. } => {
            stats.undecodable += 1;
        }
        // THE TAKE-OVER PROOF, pose-free (slice F). The destination owns the entity at
        // `source_fence`, so our part in the hand-off is positively done. A FENCE COMPARE, not a
        // flag: a replayed proof from a SUPERSEDED crossing cannot close a newer hold. Closing the
        // hold is what stops the retained ghost EMITTING (`emits` consults the hold), so this is
        // ALSO the moment every bystander's drawn copy of the leaver must be evicted — the remove
        // message, retimed here from the band-exit despawn (the leaver VANISHES at hold closure).
        // The retained DOT stays: it is the return-crossing target until band-exit tears it down.
        GhostFlow::SpawnV2 {
            entity,
            source_fence,
            ..
        } => {
            let closed = close_hold_at_fence(holds, (entity, HoldRole::Source), source_fence);
            let emitting_ghost = dots
                .0
                .values()
                .any(|d| (d.entity == entity) & is_retained_ghost(d));
            match (closed & emitting_ghost, realm_fence) {
                (true, Some(fence)) => {
                    push_entity_removed(dots, fence, entity, clock.universe_tick, outbox);
                }
                // The stop happened but this shard holds no lease: withheld (an unowned shard is
                // silent), counted — never silently (see the stat's doc).
                (true, None) => {
                    stats.entity_removals_suppressed_no_lease += 1;
                    tracing::warn!(%entity, "entity removal suppressed: no realm lease");
                }
                (false, _) => {}
            }
        }
        GhostFlow::Despawn { entity, .. } => {
            // Band-exit TEARDOWN (1d.5b.3c → slice F): the ghost lifecycle ENDS — remove the
            // retained ghost DOT (it stopped EMITTING at hold closure; this removes the
            // return-crossing target too). IDEMPOTENT: a reliable redelivery, or a stale Despawn
            // for an entity the source has since RE-OWNED (the `Owned` dot is structurally refused
            // by `remove_retained_ghost`), tears down nothing — a counted no-op
            // (`ghost_despawn_no_host`), never a panic.
            let removed_dot = remove_retained_ghost(dots, entity);
            if removed_dot {
                stats.ghost_despawns += 1;
            } else {
                stats.ghost_despawn_no_host += 1;
            }
            // THE REMOVE MESSAGE (D-4(a)): the retained ghost was the LAST emitter of this entity
            // here — the band-exit teardown is the moment a bystander's drawn figure would freeze
            // forever, so their clients are told to evict it now. Gated on the dot actually
            // removed (a stale Despawn tore down nothing ⇒ nothing changed for any client) and on
            // a held lease (an unowned shard is silent, never speculative — its subscribers are
            // being torn down by the sub machinery instead).
            match (removed_dot, realm_fence) {
                (true, Some(fence)) => {
                    push_entity_removed(dots, fence, entity, clock.universe_tick, outbox);
                }
                // The stop happened but this shard holds no lease (self-fenced mid-teardown):
                // the removal is withheld (an unowned shard is silent) but COUNTED — see the
                // stat's doc for what a bystander may see until the sub machinery catches up.
                (true, None) => {
                    stats.entity_removals_suppressed_no_lease += 1;
                    tracing::warn!(%entity, "entity removal suppressed: no realm lease");
                }
                (false, _) => {}
            }
        }
    }
}

/// Remove the RETAINED source ghost dot for `entity` (1d.5b.3c band-exit teardown), if one is hosted
/// here. Removes ONLY a dot whose authority is a `Ghost` (a retained source ghost); a dot the source
/// has RE-OWNED (`Owned` — a re-acquisition transfer brought the entity back) is structurally REFUSED,
/// so a stale Despawn from an earlier transfer can never tear out a live owner. This Ghost-only guard
/// is the shard-LOCAL stand-in for the orchestrator's in-transfer refusal (a `vd-sim` shard cannot see
/// the orchestrator live-saga set — `DEFERRED.md` D-2). Returns whether a dot was removed. Monomorphic
/// (the find + the `matches!` false arm — a non-Ghost dot — are covered here, not in the decode arm).
#[must_use]
fn remove_retained_ghost(dots: &mut Dots, entity: EntityId) -> bool {
    let Some(session) = dots
        .0
        .iter()
        .find(|(_, d)| (d.entity == entity) & matches!(d.authority, Authority::Ghost { .. }))
        .map(|(s, _)| *s)
    else {
        return false;
    };
    dots.0.remove(&session);
    true
}

/// DEST-side ghost band-exit SWEEP (1d.5b.3b → slice F): for each registered ghost-neighbor, watch
/// the OWNED entity's distance from its crossing anchor and DESPAWN the neighbour's retained ghost
/// when it exits the overlap band. NO pose ever streams (slice F deleted `GhostFlow::Delta` — every
/// fed pose was a foreign-frame write into a promotable dot, the §4u corruption); the registration
/// is also the P5 cross-boundary collision seam's retained skeleton. A registration whose entity is
/// not currently Owned here (no dot / a non-`simulates()` dot) is a counted no-op.
pub(crate) fn feed_source_ghosts(
    config: Res<StubConfig>,
    dots: Res<Dots>,
    mut registration: ResMut<GhostColliderRegistration>,
    mut stats: ResMut<StubStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    // The overlap band, seed-derived from the shard's per-tick travel (no inline literal — the
    // factors live in `core::geometry`). Velocity-safe by construction; its destroy edge is many
    // per-tick steps out, so a ghost registered in-band at the crossing exits only after the entity
    // has walked well past the demote→promote→release handoff (band-exit is strictly POST-release).
    let band = OverlapBand::for_motion(config.move_speed_mps * config.tick_dt_s);
    let mut exited: Vec<EntityId> = Vec::new();
    for (entity, neighbor) in registration.0.iter() {
        let Some(dot) = dots.0.values().find(|d| d.entity == *entity) else {
            stats.ghost_feed_skipped += 1;
            continue;
        };
        if !dot.authority.simulates() {
            stats.ghost_feed_skipped += 1;
            continue;
        }
        if ghost_band_exited(&band, neighbor.anchor, &dot.pose) {
            // BAND-EXIT (1d.5b.3c): the owned entity left the overlap band — DESPAWN the ghost on the
            // RELIABLE carrier (a lost Despawn would leak the collider) + DEREGISTER. The source
            // tears the ghost DOT down on receipt (`on_ghost_flow`); its clients were already told
            // to evict at hold closure (the SpawnV2 proof), so nothing visible changes there.
            outbox.push_flow_durable(
                neighbor.source,
                MsgClass::GhostReliable,
                &InterShardFlow::Ghost(GhostFlow::Despawn {
                    entity: *entity,
                    source_fence: dot.authority.fence(),
                }),
                Durability::Retained,
            );
            exited.push(*entity);
            stats.ghost_band_exits += 1;
        }
    }
    // Deregister the exited ghosts (the sweep stops; the source teardown is driven by the Despawn).
    for entity in exited {
        registration.0.remove(&entity);
    }
}

/// Whether the owned entity has EXITED the overlap band anchored at its boundary crossing (1d.5b.3c):
/// it is no longer a member — it has travelled past the band's destroy edge from `anchor`. A
/// branchless monomorphic shim: the membership hysteresis lives in `OverlapBand::update_membership`
/// (fully covered in `core`), so the dest's band-exit decision adds no uncovered branch here. The
/// ghost was SPAWNED in-band at the crossing (distance 0 = a member), so `was_member` is always `true`.
#[must_use]
fn ghost_band_exited(band: &OverlapBand, anchor: LatticePos, pose: &StampedPose) -> bool {
    !band.update_membership(true, pose.pos.delta_m(anchor, pose.frame.tier()).length())
}

/// Whether a dot is a RETAINED source ghost (1d.5b.3b): a granted, non-departing `Ghost` whose
/// `source_fence` is POST-GENESIS — it holds its OWN-frame demote pose (slice F: nothing ever
/// refreshes it; the pose feed is dead). The `source_fence != GENESIS` term EXCLUDES the
/// pre-promote DEST-adopt Ghost (which holds `GENESIS` and has no real pose — it must stay silent
/// until `on_saga_promote` flips it Owned); `granted & !departing` excludes a pre-grant provisional
/// Ghost. Monomorphic (the state destructure + guard live here, not in the filter closure).
#[must_use]
pub(crate) fn is_retained_ghost(d: &Dot) -> bool {
    let retained_source = matches!(
        d.authority,
        Authority::Ghost { source_fence, .. } if source_fence != Fence::GENESIS
    );
    retained_source & d.granted & !d.departing
}
