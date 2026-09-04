//! ★ THE REACH — a realm states how far it is seen, to its parent, on change (owner ruling
//! 2026-09-02 R3/R6; built 2026-09-04 under the owner's rule *"brightness counts only when no
//! ancestor already draws that light"*).
//!
//! Owns: the ONE emitter (this realm's own reach, by size and by light, folded over its children,
//! stated upward only when it changes) and the ONE consumer (a child's statement, admitted through
//! the same three guards every up-lane carries, then handed to the region table which re-bands the
//! child and refolds this realm's own terms).
//!
//! Does NOT own: the numbers. The region table folds them ([`RealmRegions::own_reach`],
//! [`RealmRegions::set_child_reach`]); the physics of a dot and of a magnitude live in `vd-core`.
//!
//! Example: a station is built on a planet and reaches 200 000 km. The planet's own reach does not
//! move (its own disc reaches farther), so the planet says nothing to its star system. A brighter
//! lamp on the station lifts its reach by light past the planet's: the planet restates once.

use super::realm_head::ParentRealmNode;
use super::regions::{ReachOutcome, RealmRegions};
use super::{ChildLuma, RealmAuthority, StubConfig, StubStats};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::BTreeMap;
use vd_core::ids::NodeId;
use vd_core::pose::RealmId;
use vd_wire::intershard::{InterShardFlow, ReachStated};

/// What this realm last stated upward — send-on-change's memory. `None` until the first statement.
#[derive(Resource, Debug, Default)]
pub struct StatedReach(pub Option<(u64, u64)>);

/// The emitter: this realm's own reach, upward, when it changed. Withheld until the parent's node is
/// known and this realm holds its lease (a statement needs a route and a fence). Producer-less
/// reliable: stated once, carried `Retained`, so a parent that restarts is restated to by the
/// lineage path, never by a timer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_own_reach(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    parent_node: Res<ParentRealmNode>,
    regions: Res<RealmRegions>,
    luma: Res<ChildLuma>,
    mut stated: ResMut<StatedReach>,
    mut outbox: ResMut<OutboundBox>,
    mut stats: ResMut<StubStats>,
) {
    let (Some(parent), Some(fence)) = (parent_node.0, authority.0) else {
        return;
    };
    let own_luma = luma.0.get(&config.realm).map(|(_, l)| *l);
    let reach = regions.own_reach(config.realm, own_luma);
    if stated.0 == Some(reach) {
        return; // unchanged — a reach says nothing twice
    }
    outbox.push_flow_durable(
        parent,
        MsgClass::Saga,
        &InterShardFlow::ReachStated(ReachStated {
            child: config.own_coord.clone(),
            child_fence: fence,
            at: clock.universe_tick,
            size_reach_m: reach.0,
            light_reach_m: reach.1,
        }),
        Durability::Retained,
    );
    stated.0 = Some(reach);
    stats.reach_sent += 1;
}

/// The consumer: a child's statement through the three guards (misroute, attestation, staleness),
/// then the table. Monomorphic, every arm counted (HR5).
pub(crate) fn on_reach_stated(
    rs: ReachStated,
    from: NodeId,
    own_realm: RealmId,
    child_nodes: &BTreeMap<RealmId, NodeId>,
    regions: &mut RealmRegions,
    stats: &mut StubStats,
) {
    let child = rs.child.lowered();
    if rs.child.parent().map(|p| p.lowered()) != Some(own_realm) {
        stats.reach_misrouted += 1;
        return;
    }
    if child_nodes.get(&child) != Some(&from) {
        stats.reach_unattested += 1;
        return;
    }
    match regions.set_child_reach(
        child,
        (rs.child_fence, rs.at),
        rs.size_reach_m,
        rs.light_reach_m,
    ) {
        ReachOutcome::Applied { .. } => stats.reach_received += 1,
        ReachOutcome::Stale => stats.reach_stale += 1,
        // Attested as my child by the directory, yet not on my roster: the adopt has not landed
        // here yet. Counted with the misroutes; the child restates on its next lineage.
        ReachOutcome::NotMine => stats.reach_misrouted += 1,
    }
}
