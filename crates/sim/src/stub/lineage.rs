//! ★ THE HULL IS TOLD (the ruler switch, slice 3; owner-approved 2026-09-03 — the plan's §3.7 and its
//! ask 2, THE ONE NEW DATUM).
//!
//! After an adoption the new parent authors the hull's placement, but the hull's own shard still
//! believes its OLD parent: every drive datagram, every occupancy bit and every fact it sends names
//! the old coord and is refused as misrouted, and its facts were stated to a parent that no longer
//! holds it. So the parent STATES the hull's new lineage to it — its own coord plus the hull's level —
//! once it learns the hull's node from the directory, and again on every head read of the hull until
//! the hull's facts arrive (which they do only after the hull applied the statement). The hull applies
//! it only from the node the directory names as its exterior's holder; a statement from anybody else
//! is held and the exterior head is re-read, so a deposed or hostile sender can name nothing.
//!
//! On apply the hull: replaces its own coord, points its up-lanes at the sender, forgets what facts it
//! stated (so they go out again to the new parent), re-arms its occupancy edge, and re-points its own
//! row's parent in its forest. Nothing here tells the hull where it is.
//!
//! Example: the galaxy adopts a hull that left System 7. The directory answers the galaxy with the
//! hull's node; the galaxy states `Universe / Galaxy 1 / Ship hull`. The hull's next drive datagram
//! names that coord, the galaxy admits it, and the hull's mass follows on the reliable lane.

use super::drive::{DrivenChildren, StatedFacts};
use super::realm_head::{ExteriorAuthority, ParentRealmNode};
use super::{RealmRegions, StubConfig, StubStats};
use crate::io::MsgClass;
use crate::runtime::OutboundBox;
use bevy_ecs::prelude::Resource;
use std::collections::BTreeSet;
use vd_core::ids::UniverseTick;
use vd_core::pose::RealmId;
use vd_core::{Fence, NodeId};
use vd_wire::intershard::{InterShardFlow, LineageStated};
use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};

/// THE PARENT'S SIDE: the children whose lineage this realm still owes a statement — inserted at an
/// adoption, cleared when the child's facts arrive (the proof it heard). Default EMPTY.
#[derive(Resource, Debug, Default)]
pub struct LineageOwed(pub BTreeSet<RealmId>);

/// THE CHILD'S SIDE: a statement held until the directory names its sender as this realm's exterior
/// holder. `None` almost always.
#[derive(Resource, Debug, Default)]
pub struct PendingLineage(pub Option<(LineageStated, NodeId)>);

/// The child's and the parent's lineage stores, bundled so the dispatch stays legible.
pub(crate) struct LineageSide<'a> {
    pub stated: &'a mut StatedFacts,
    pub was_occupied: &'a mut super::aoi::WasOccupied,
    pub owed: &'a mut LineageOwed,
    pub pending: &'a mut PendingLineage,
}

/// The parent states an owed child's lineage to the node the directory just named for it. Called from
/// the child-head reply arm, so it re-states on every read until the child's facts clear the debt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn state_lineage_if_owed(
    child: RealmId,
    node: Option<NodeId>,
    config: &StubConfig,
    realm_fence: Option<Fence>,
    tick: UniverseTick,
    owed: &LineageOwed,
    outbox: &mut OutboundBox,
    stats: &mut StubStats,
) {
    let (Some(node), Some(parent_fence), true) = (node, realm_fence, owed.0.contains(&child))
    else {
        return;
    };
    let coord = config.own_coord.child(vd_core::worldgen::level_of(child));
    outbox.push_flow(
        node,
        MsgClass::Saga,
        &InterShardFlow::LineageStated(LineageStated {
            child: coord,
            parent_fence,
            at: tick,
        }),
    );
    stats.lineage_stated += 1;
    tracing::info!(%child, own = %config.realm, ?node, "LINEAGE STATED to an adopted child");
}

/// The child's facts arrived: it has heard its lineage, and the debt is paid.
pub(crate) fn lineage_heard(child: RealmId, owed: &mut LineageOwed) {
    owed.0.remove(&child);
}

/// THE CHILD RECEIVES a lineage statement. Misrouted (not about me) ⇒ refused, counted. From the node
/// my exterior head names ⇒ applied at once. From anybody else ⇒ held, and my exterior head re-read;
/// the reply applies or discards it (`apply_pending_lineage`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_lineage_stated(
    ls: LineageStated,
    from: NodeId,
    config: &mut StubConfig,
    regions: &mut RealmRegions,
    parent_node: &mut ParentRealmNode,
    stated: &mut StatedFacts,
    was_occupied: &mut super::aoi::WasOccupied,
    pending: &mut PendingLineage,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    if ls.child.lowered() != config.realm {
        stats.lineage_misrouted += 1;
        return;
    }
    if parent_node.0 == Some(from) {
        apply_lineage(
            ls,
            from,
            config,
            regions,
            parent_node,
            stated,
            was_occupied,
            stats,
        );
        return;
    }
    stats.lineage_held_unattested += 1;
    if let Some(key) = DirectoryKey::exterior_of(config.realm) {
        outbox.push_flow(
            config.orchestrator,
            MsgClass::Saga,
            &InterShardFlow::Directory(DirectoryOp::HeadRead { key }),
        );
    }
    pending.0 = Some((ls, from));
}

/// After an exterior head reply moved `parent_node`: a held statement from that very node is applied;
/// one from any other node is discarded — the directory did not name its sender.
pub(crate) fn apply_pending_lineage(
    config: &mut StubConfig,
    regions: &mut RealmRegions,
    parent_node: &mut ParentRealmNode,
    stated: &mut StatedFacts,
    was_occupied: &mut super::aoi::WasOccupied,
    pending: &mut PendingLineage,
    stats: &mut StubStats,
) {
    let Some((ls, from)) = pending.0.take() else {
        return;
    };
    if parent_node.0 == Some(from) {
        apply_lineage(
            ls,
            from,
            config,
            regions,
            parent_node,
            stated,
            was_occupied,
            stats,
        );
    } else {
        stats.lineage_discarded += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_lineage(
    ls: LineageStated,
    from: NodeId,
    config: &mut StubConfig,
    regions: &mut RealmRegions,
    parent_node: &mut ParentRealmNode,
    stated: &mut StatedFacts,
    was_occupied: &mut super::aoi::WasOccupied,
    stats: &mut StubStats,
) {
    let new_parent = ls.child.parent().map(|p| p.lowered());
    tracing::info!(
        own = %config.realm,
        was = ?config.own_coord.parent().map(|p| p.lowered()),
        now = ?new_parent,
        ?from,
        "LINEAGE APPLIED: this realm moved house — its facts go out again to its new parent",
    );
    config.own_coord = ls.child;
    parent_node.0 = Some(from);
    stated.0 = None;
    was_occupied.0 = false;
    if let Some(p) = new_parent {
        regions.reparent_own(p);
    }
    stats.lineage_applied += 1;
}

/// The parent's debt begins at an adoption.
pub(crate) fn lineage_owed(child: RealmId, owed: &mut LineageOwed) {
    owed.0.insert(child);
}

// A driven set is what the parent authors; the type is named here so the module's doc reads whole.
#[allow(dead_code)]
type Authored = DrivenChildren;
#[allow(dead_code)]
type Lease = ExteriorAuthority;
