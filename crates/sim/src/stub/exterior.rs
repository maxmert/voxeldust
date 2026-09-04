//! ★ THE EXTERIOR CROSSING ON THE SHARDS (the ruler switch, slice 2; owner-approved 2026-09-03 —
//! `docs/design/realm_crossing_plan_2026-09-02.md` §3.5–3.6).
//!
//! A hull is a realm its parent authors. When the parent's swept verdict hands the hull to another
//! realm, three things move on the shards, on the one transfer machinery:
//! - **the source FLUSHES** the hull's exterior — its authored placement as a stamped pose, in the
//!   frame SL1 dictates (verbatim in its own frame going UP, subtracted into the child's frame going
//!   DOWN), plus the blob the pose cannot carry — and freezes the hull's drive so the pose it shipped
//!   stays the truth until the hand-over commits;
//! - **the destination ADOPTS** the exterior: it places the arriving pose (its parent's number as
//!   given, or its child's number added through its own book), re-advances it to its own tick,
//!   takes the hull onto its roster without a rebuild, seeds the driven state, records the exterior
//!   lease at the crossing's fence, writes the berth row into its own store, and asks the directory
//!   for the hull's node so the up-lanes admit it from its first tick;
//! - **the source RELEASES** the hull at the demote: off its roster, out of its driven set, its
//!   exterior lease dropped, its berth row deleted.
//!
//! Nothing here names the hull's motion, drive or kind (SL4): a station or a rock with an exterior
//! key would take the identical path. The hull's own shard is never touched; what it must learn —
//! its new parent — is slice 3.
//!
//! Example: System 7 flushes the hull's pose in System 7's frame. The galaxy adds System 7's
//! placement, which it authored, at its own two-metre rung; the hull sits on the galaxy's roster
//! next tick; System 7 forgets it at the demote.

use super::drive::{DrivenChild, DrivenChildren, DrivenState};
use super::realm_head::{ChildRealmNodes, ExteriorAuthority};
use super::{RealmRegions, StubConfig, StubStats};
use crate::io::{MsgClass, Store};
use crate::runtime::OutboundBox;
use bevy_ecs::prelude::Resource;
use vd_core::built::Berth;
use vd_core::geometry::ParentCentre;
use vd_core::glam::DVec3;
use vd_core::ids::UniverseTick;
use vd_core::pose::{LatticePos, RealmId, StampedPose};
use vd_core::{EntityId, Fence, TransferId};
use vd_wire::intershard::{ExteriorState, InterShardFlow, TransferAck};
use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};

/// ★ THIS REALM'S OWN DURABLE STORE, at runtime (the ruler switch, slice 2). The shard bin opens the
/// realm's file at boot and reads its berths and its body; until now the handle was dropped after the
/// read, because nothing wrote at runtime. An adopted hull's berth row is the first runtime write:
/// without it a restart re-plants the hull under its OLD parent (`shard.rs` reads berths at boot).
/// `None` for a shard with no store, which is what every test rig and every seeded-only shard has.
#[derive(Resource, Default)]
pub struct RealmStore(pub Option<Box<dyn Store + Send + Sync>>);

impl std::fmt::Debug for RealmStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RealmStore")
            .field(&self.0.is_some())
            .finish()
    }
}

/// The destination's mutable half of an adoption, bundled so the dispatch stays legible.
pub(crate) struct AdoptSide<'a> {
    pub driven: &'a mut DrivenChildren,
    pub exterior: &'a mut ExteriorAuthority,
    pub store: &'a mut RealmStore,
    pub owed: &'a mut super::lineage::LineageOwed,
}

/// The per-child tables a parent keeps beside its roster — the area-of-interest rows, the interest
/// byte latch, the in-band verdict and the child's luma. Every one is keyed by the child, so a release
/// must drop the child's rows from each, or the old parent keeps a stale record of a hull it no longer
/// authors (a stale row is small, and wrong).
pub(crate) struct ChildTables<'a> {
    pub aoi: &'a mut super::aoi::AoiMembership,
    pub interest_latch: &'a mut super::relay::InterestEmitLatch,
    pub in_band: &'a mut super::relay::InBandVerdict,
    pub luma: &'a mut super::window::ChildLuma,
}

/// The source's mutable half of a release, bundled likewise.
pub(crate) struct ReleaseSide<'a> {
    pub regions: &'a mut RealmRegions,
    pub driven: &'a mut DrivenChildren,
    pub exterior: &'a mut ExteriorAuthority,
    pub child_nodes: &'a mut ChildRealmNodes,
    pub child_liveness: &'a mut super::aoi::ChildLiveness,
    pub relay_held: &'a mut super::relay::RelayHeld,
    pub store: &'a mut RealmStore,
    pub tables: ChildTables<'a>,
    pub owed: &'a mut super::lineage::LineageOwed,
}

/// THE SOURCE FLUSHES a driven child's exterior — the reply to a `FlushSource` whose subject is a
/// `Ship` key. Refused and counted when this realm does not hold that exterior (a stale saga, or a
/// request for a hull that is not mine) or cannot place the destination.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flush_exterior(
    transfer: TransferId,
    step_id: u32,
    entity: EntityId,
    to_realm: RealmId,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &vd_core::placement::PlacementLedger,
    tick: UniverseTick,
    driven: &mut DrivenChildren,
    exterior: &ExteriorAuthority,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let child = RealmId::Ship(entity);
    let held = exterior.0.contains_key(&child);
    let region = regions.direct_child(config.realm, child).copied();
    let Some((region, state)) = region
        .filter(|_| held)
        .and_then(|r| driven.0.get(&child).map(|d| (r, d.state)))
    else {
        stats.exterior_flush_unheld += 1;
        tracing::warn!(
            %child,
            own = %config.realm,
            held,
            "FlushSource for an exterior this realm does not author — no pose to ship",
        );
        return;
    };
    let own_frame = regions.own_frame(config.realm);
    let pose = super::containment::exterior_pose(&region, &state, own_frame, tick);
    // The one conversion every flush takes, minus the stale-exit re-validation: the early start
    // means the hull is still INSIDE at flush time by design, and that is not a departure gone stale.
    let Some(pose) = super::conversion::flush_pose_for_dest(
        pose, to_realm, config, regions, placements, tick, stats, false,
    ) else {
        stats.exterior_flush_unplaceable += 1;
        return;
    };
    // FROZEN: the drive is no longer applied, so the pose shipped stays the truth the destination
    // re-advances from — both sides coast at the same velocity until the hand-over commits, and a
    // thrust the pilot pushes in those ticks is lost rather than doubled.
    if let Some(d) = driven.0.get_mut(&child) {
        d.frozen = true;
    }
    let mut row = region;
    row.center = ParentCentre::ORIGIN; // the child's placement in MY frame is not the destination's to read
    row.parent = None;
    let blob = ExteriorState {
        region: row,
        spin: state.spin_radps,
    };
    stats.exterior_flushed += 1;
    tracing::info!(
        %child,
        to = ?to_realm,
        own = %config.realm,
        pose_frame = ?pose.frame,
        at_tick = pose.universe_tick.0,
        "EXTERIOR FLUSHED: the hull's authored placement leaves for its new parent, and its drive freezes",
    );
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: transfer,
            step_id,
            pose,
            drained_seq: 0,
            state: blob.encode(),
        }),
    );
}

/// THE DESTINATION ADOPTS an exterior that arrived on a `StubCrossing` envelope — `pose` already
/// placed in THIS realm's frame by `place_arriving_pose`. Idempotent on the applied-steps journal,
/// so a re-emitted envelope adopts once and acks every time. Refused and counted when the blob does
/// not decode or the destination named is not this realm: an exterior lands only in the realm that
/// will author it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adopt_exterior(
    transfer: TransferId,
    step_id: u32,
    fence: Fence,
    entity: EntityId,
    to_realm: RealmId,
    pose: StampedPose,
    state: &[u8],
    config: &StubConfig,
    regions: &mut RealmRegions,
    side: AdoptSide<'_>,
    applied: &mut super::AppliedSteps,
    tick: UniverseTick,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let AdoptSide {
        driven,
        exterior,
        store,
        owed,
    } = side;
    let child = RealmId::Ship(entity);
    let first = applied.journal_step(transfer, step_id) == super::StepOutcome::FirstApply;
    if first {
        let blob = ExteriorState::decode(state);
        let (Some(blob), true) = (blob, to_realm == config.realm) else {
            stats.exterior_arrival_refused += 1;
            tracing::error!(
                %child,
                into = ?to_realm,
                own = %config.realm,
                decodes = ExteriorState::decode(state).is_some(),
                "refusing an exterior this realm cannot adopt — not its own realm, or an unreadable blob",
            );
            return;
        };
        // Re-advance to now: the flush was ticks ago and both sides coasted since (the source froze
        // the drive), so the placement continues at the flushed velocity, never with a jump back.
        let gap_s = tick.0.saturating_sub(pose.universe_tick.0) as f64 * config.tick_dt_s;
        let now_pose = pose.advanced_ballistic(DVec3::ZERO, gap_s, tick);
        let tier = now_pose.frame.tier();
        let mut region = blob.region;
        region.realm = child;
        region.parent = Some(config.realm);
        region.center = ParentCentre::authored(now_pose.pos);
        regions.adopt_child(region);
        driven.0.insert(
            child,
            DrivenChild {
                state: DrivenState {
                    pos_m: DVec3::ZERO,
                    vel_mps: now_pose.vel,
                    orient: now_pose.orient,
                    spin_radps: blob.spin,
                },
                drive: ([0; 3], [0; 3]),
                drive_at: (Fence::GENESIS, tick),
                facts: None, // the hull restates what it IS to its new parent (slice 3)
                facts_at: (Fence::GENESIS, tick),
                frozen: false,
            },
        );
        exterior.0.insert(child, fence);
        // The hull must be TOLD it is mine now (slice 3): owed until its facts arrive.
        super::lineage::lineage_owed(child, owed);
        // THE BERTH ROW, durable in my own store, so a restart re-plants the hull under ME.
        if let Some(store) = store.0.as_mut() {
            let berth = Berth {
                child,
                offset_m: now_pose.pos.delta_m(LatticePos::ORIGIN, tier),
                bound: region.shape,
                look: region.look.unwrap_or(region.shape),
                fence,
            };
            store.put(
                &super::built_store::berth_key(child),
                &crate::io::bytes(super::built_store::encode_berth(&berth)),
            );
            store.commit();
        }
        // WHO RUNS THE HULL: the directory says; the reply fills the up-lanes' admission map.
        outbox.push_flow(
            config.orchestrator,
            MsgClass::Saga,
            &InterShardFlow::Directory(DirectoryOp::HeadRead {
                key: DirectoryKey::Realm(child),
            }),
        );
        stats.exterior_adopted += 1;
        tracing::info!(
            %child,
            own = %config.realm,
            ?fence,
            at_tick = tick.0,
            pos_m = %vd_core::pose::describe(now_pose.pos, now_pose.frame),
            "EXTERIOR ADOPTED: this realm authors the hull's placement from now on",
        );
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: transfer,
            step_id,
        }),
    );
}

/// THE SOURCE RELEASES an exterior at the demote: off the roster, out of the driven set, the lease
/// and the admission dropped, the berth row deleted. A hull already released is a no-op, so a
/// re-driven demote is harmless.
#[allow(clippy::too_many_arguments)]
pub(crate) fn release_exterior(
    entity: EntityId,
    config: &StubConfig,
    side: ReleaseSide<'_>,
    stats: &mut StubStats,
) {
    let ReleaseSide {
        regions,
        driven,
        exterior,
        child_nodes,
        child_liveness,
        relay_held,
        store,
        tables,
        owed,
    } = side;
    let child = RealmId::Ship(entity);
    let was_mine = driven.0.remove(&child).is_some();
    regions.release_child(child);
    exterior.0.remove(&child);
    child_nodes.0.remove(&child);
    child_liveness.0.remove(&child);
    relay_held.0.remove(&child);
    // The per-child tables: the hull as a WATCHED realm and as an OBSERVER (the occupied-child proxy)
    // both leave the area-of-interest rows; its latch, its verdict, its luma and the lineage I owed it go.
    tables.aoi.0.retain(|(observer, watched), _| {
        (*watched != child) & (*observer != super::aoi::ObserverId::Child(child))
    });
    tables.interest_latch.0.remove(&child);
    tables.in_band.0.remove(&child);
    tables.luma.0.remove(&child);
    owed.0.remove(&child);
    if let Some(store) = store.0.as_mut() {
        store.delete(&super::built_store::berth_key(child));
        store.commit();
    }
    stats.exterior_released += u64::from(was_mine);
    tracing::info!(
        %child,
        own = %config.realm,
        was_mine,
        "EXTERIOR RELEASED: this realm no longer authors the hull's placement",
    );
}
