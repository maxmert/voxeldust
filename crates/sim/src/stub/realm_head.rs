//! THE DIRECTORY ROUND-TRIP: which fences this shard holds, which nodes hold the realms around it,
//! and when to stop believing either.
//!
//! Owns: the realm-authority stores (primary and co-hosted), the cached parent/child head nodes,
//! the grant/renew/read requests that keep them fresh, the reply arms that write them, and the
//! PROACTIVE self-fence — a holder that has lost contact with the directory hard-stops its own
//! authority before anyone else may be given it.
//!
//! Does NOT own: the directory itself (that is `crate::directory`, and at scale another node's).
//! Nothing here decides what a lane DOES with authority — it only decides whether authority is
//! held, and every emitting lane asks before it speaks.

use super::{
    AppliedSteps, CrossingProgress, Dot, Dots, GhostColliderRegistration, GrantFlip, HandoffHolds,
    InterestHeld, OwnedTransients, PendingCrossings, PendingInputSlots, RealmRegions, RelayHeld,
    RequestInFlight, StubConfig, StubStats, aoi_recheck_cadence, drain_pending_crossing,
    drain_pending_input_slots, flip_grant, on_crossing_aborted, on_flush_source, on_re_home,
    on_realm_interest, on_release_complete, on_saga_demote, on_saga_promote, on_transfer_envelope,
    on_transient_abandon, on_transient_crossing_grant, on_transient_discard, on_transient_promote,
    on_transient_release, on_window_relay, push_entity_removed, push_session_reply,
    self_fence_drop_transients,
};
use crate::io::MsgClass;
use crate::runtime::{ClockSample, NodeIdentity, OutboundBox};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use std::collections::BTreeMap;
use vd_core::placement::PlacementLedger;
use vd_core::pose::RealmId;
use vd_core::{Fence, NodeId, SessionId, TickId};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::session_flow::ShardToGateway;

/// The PRIMARY realm authority this shard holds (`config.realm`'s fence; None until the directory
/// grants it — no frames are emitted unowned). The self-fence / emit-frame / promote-guard machinery
/// all key on THIS, unchanged from the single-realm model. CO-HOSTED child realms (co-hosting) carry
/// their own fences in [`CoHostedAuthority`]; the local re-home short-circuit reads the union via
/// `CrossingCtx::held_here`.
#[derive(Resource, Debug, Default)]
pub struct RealmAuthority(pub Option<Fence>);

/// The fences for the CO-HOSTED CHILD realms this shard hosts beyond its own `config.realm` (the
/// un-hosted-child cure). Populated by the same directory grant/affirm path as `RealmAuthority` but
/// for `config.cohosted_realms()` — a per-realm map so each child's head is affirmed independently.
/// Default EMPTY: a single-realm shard (`held_realms == {realm}`) never populates it, so every
/// single-realm rig is byte-identical (the map is only touched when `cohosted_realms()` is non-empty).
#[derive(Resource, Debug, Default)]
pub struct CoHostedAuthority(pub BTreeMap<RealmId, Fence>);

/// ★ THE EXTERIOR LEASES THIS REALM HOLDS FOR ITS CHILDREN (the ruler switch, slice 0; owner-approved
/// 2026-09-02). For every direct child that has an exterior key (`DirectoryKey::exterior_of`), the
/// parent leases that key: it is the directory's record that THIS realm authors that child's placement.
/// Requested until affirmed, renewed with the realm's own heartbeat, dropped the moment the directory
/// names another node — which is what a later saga does when the child moves house.
///
/// Default EMPTY: a realm with no built children never touches it, so every seeded-only rig is
/// byte-identical (the loops over it never body).
///
/// Example: System 7 boots with the hull's berth in its store. Its forest holds the hull as a direct
/// child, so it requests `Ship(hull)` at genesis, and from the reply on it renews it beside
/// `Realm(System 7)`. The hull, meanwhile, reads `Ship(hull)` to learn who its parent is.
#[derive(Resource, Debug, Default)]
pub struct ExteriorAuthority(pub BTreeMap<RealmId, Fence>);

/// D-3 Slice 5 — the `local_tick` of the last realm-head ROUND-TRIP confirmation (the reply at which
/// the directory affirmed this shard still owns its realm). The partition detector for the proactive
/// self-fence: it advances only when a `realm_recheck` reply lands, so under a partition (the
/// orchestrator unreachable, no reply) it FREEZES while `local_tick` keeps advancing — the gap is the
/// evidence the holder has lost contact. Meaningful only while `RealmAuthority` is `Some` (the
/// self-fence timer guards on that); reset to the current `local_tick` on every (re)confirmation, so a
/// re-granted realm never inherits a stale deadline.
#[derive(Resource, Debug, Default)]
pub struct RealmConfirmedAt(pub TickId);

/// The resolved live `NodeId` owning this shard's PARENT realm (`own_coord.parent()`), learned via a
/// directory HeadRead round-trip and CACHED (resolve-once, re-read on the realm-recheck cadence so a
/// parent RE-HOME is observed). `None` = root shard / pre-resolve / mid-CAS gap / walk-static (never
/// armed) ⇒ every up-lane is inert: the SL7 `ChildLive` bit and the `RealmObservation`/
/// `RealmShapeObservation` mirrors all gate on it. Overwritten on every parent-realm Head reply (the
/// mirror of [`RealmAuthority`]'s fence overwrite). NOT a map — a shard has exactly one parent.
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentRealmNode(pub Option<NodeId>);

/// The resolved live `NodeId` owning each of this shard's DIRECT CHILD realms — [`ParentRealmNode`]'s
/// downward twin, learned via the SAME directory HeadRead round-trip on the SAME cadence (HR3) and
/// written by the SAME realm-Head reply arm. This is the up-lanes' ADMISSION authority: a `ChildLive`
/// bit / `RealmObservation` / `RealmShapeObservation` is believed ONLY when its sender matches the
/// directory's record for that child — the same head the fence CAS already trusts, resolved locally.
/// The learned frame sender stays the ROUTE (`ChildLiveEntry::home` — the mesh routes only to
/// booked-or-learned peers, so a directory-derived NodeId may be unroutable); the two are DELIBERATELY
/// different jobs. Absent entry = fail closed (refuse and count; a refusal arms a lazy re-read).
/// Keys on the LOSSY `lowered()` `RealmId`, exactly like the directory and [`ParentRealmNode`] — all
/// three migrate to `path()`-keying together (DEFERRED, D-RLM-10). EMPTY at walk/static (no read is
/// ever emitted) ⇒ byte-identical.
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub struct ChildRealmNodes(pub BTreeMap<RealmId, NodeId>);

/// Until the directory has granted this shard its realm — and every provisional
/// dot its entity — keep requesting (grants are idempotent by fence; a lost
/// reply costs one tick).
#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources
pub(crate) fn request_pending_grants(
    config: Res<StubConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    authority: Res<RealmAuthority>,
    cohosted: Res<CoHostedAuthority>,
    exterior: Res<ExteriorAuthority>,
    regions: Res<RealmRegions>,
    dots: Res<Dots>,
    mut outbox: ResMut<OutboundBox>,
) {
    // ★ THE EXTERIOR KEYS OF MY BUILT CHILDREN (the ruler switch, slice 0): request each until it is
    // affirmed, exactly like a co-hosted child's realm key. INERT for a realm with no built children —
    // `exterior_keys_wanted` yields nothing — so every seeded-only rig is byte-identical.
    for (_child, key) in exterior_keys_wanted(&config, &regions, &exterior) {
        outbox.push_flow(
            config.orchestrator,
            MsgClass::Saga,
            &InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                key,
                owner: AuthorityRef::Shard(identity.node_id),
                fence: Fence::GENESIS.next(),
            }),
        );
    }
    // Co-hosting (the un-hosted-child cure): keep requesting the head of every co-hosted CHILD realm not
    // yet affirmed (the un-affirmed child self-grants at `GENESIS.next`, exactly like the primary realm's
    // `None` arm; a re-request of an already-held child is idempotent by fence). INERT for a single-realm
    // shard — `cohosted_realms()` is empty, so this loop never bodies (byte-identical). A BRANCHLESS shim:
    // the sole branch is the `contains_key` membership, covered by both a held and an un-held child.
    for realm in config.cohosted_realms() {
        if !cohosted.0.contains_key(&realm) {
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Realm(realm),
                    owner: AuthorityRef::Shard(identity.node_id),
                    fence: Fence::GENESIS.next(),
                }),
            );
        }
    }
    match authority.0 {
        None => {
            let op = DirectoryOp::LeaseGrant {
                key: DirectoryKey::Realm(config.realm),
                owner: AuthorityRef::Shard(identity.node_id),
                fence: Fence::GENESIS.next(),
            };
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        }
        Some(realm_fence) => {
            // Periodically re-read the realm head: the reply reveals a lost lease so
            // the shard self-fences (the loss-reaction is otherwise unreachable). It is ALSO the
            // round-trip that re-arms `RealmConfirmedAt` for the proactive self-fence (D-3 Slice 5).
            if crate::directory::due_this_tick(config.realm_recheck_interval, clock.local_tick.0) {
                let op = DirectoryOp::HeadRead {
                    key: DirectoryKey::Realm(config.realm),
                };
                outbox.push_flow(
                    config.orchestrator,
                    MsgClass::Saga,
                    &InterShardFlow::Directory(op),
                );
            }
            // D-3 lease-renewal heartbeat: keep the Realm + every granted, non-departing Entity lease
            // alive on the holder's own LOCAL cadence (the orchestrator's reaper revokes a lapsed lease).
            // INERT when `lease_renew_interval_ticks == 0` (pre-D-3 default). A departing dot is excluded
            // (its lease is about to be revoked by the logout `LeaseRevoke`, not renewed).
            if crate::directory::due_this_tick(
                config.lease_renew_interval_ticks,
                clock.local_tick.0,
            ) {
                // The primary realm + every CO-HOSTED child realm this shard actually holds + every
                // granted non-departing Entity. The co-host chain is EMPTY for a single-realm shard
                // (`cohosted.0` is never populated), so the renewal set is byte-identical there.
                let renewals = std::iter::once((DirectoryKey::Realm(config.realm), realm_fence))
                    .chain(
                        cohosted
                            .0
                            .iter()
                            .map(|(realm, fence)| (DirectoryKey::Realm(*realm), *fence)),
                    )
                    .chain(
                        dots.0
                            .values()
                            .filter(|d| d.granted && !d.departing)
                            .map(|d| (DirectoryKey::Entity(d.entity), d.entity_fence)),
                    )
                    // The exterior leases of my built children ride the same heartbeat: a lapsed
                    // exterior would let the reaper forget who authors the hull's placement.
                    .chain(exterior.0.iter().filter_map(|(child, fence)| {
                        DirectoryKey::exterior_of(*child).map(|key| (key, *fence))
                    }));
                outbox.push_renewals(renewals, config.orchestrator);
            }
        }
    }
    // Per-dot grant/revoke requests (login LeaseGrant, adopt HeadRead, logout LeaseRevoke). The
    // 1c.8 SOURCE granted-key poll is GONE (1d.5b.2): the saga-pushed `Demote` (on_saga_demote) is
    // now the SOLE source-demote driver — the source no longer polls the directory to DISCOVER a
    // foreign takeover.
    for dot in dots.0.values() {
        if let Some(op) = pending_grant_op(dot, identity.node_id) {
            outbox.push_flow(
                config.orchestrator,
                MsgClass::Saga,
                &InterShardFlow::Directory(op),
            );
        }
    }
}

/// The per-dot directory op `request_pending_grants` should (re)issue, as a monomorphic 3-way so
/// the system loop stays a branchless shim (HR5):
/// - adopting + !granted → `HeadRead{Entity}` (1c.8): ADOPT the record the transfer CAS moved
///   here; a `LeaseGrant` at `GENESIS.next` would be Refused (the record is past genesis).
/// - !adopting + !granted → `LeaseGrant{Entity}` at `GENESIS.next` (the login fresh-mint path).
/// - departing → `LeaseRevoke{Entity}` at the RECORDED fence (FENCE-1/5/8; a literal would be
///   Refused once a transfer advanced the fence, stranding the logout).
/// - otherwise (granted, non-departing) → `None`. The 1c.8 source granted-key poll branch is GONE
///   (1d.5b.2): the source self-fence is driven by the saga-pushed `Demote`, not a directory poll.
#[must_use]
fn pending_grant_op(dot: &Dot, node: NodeId) -> Option<DirectoryOp> {
    if !dot.granted {
        if dot.adopting {
            Some(DirectoryOp::HeadRead {
                key: DirectoryKey::Entity(dot.entity),
            })
        } else {
            Some(DirectoryOp::LeaseGrant {
                key: DirectoryKey::Entity(dot.entity),
                owner: AuthorityRef::Shard(node),
                fence: Fence::GENESIS.next(),
            })
        }
    } else if dot.departing {
        Some(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Entity(dot.entity),
            fence: dot.entity_fence,
        })
    } else {
        None
    }
}

/// D-3 Slice 5 — the PROACTIVE self-fence (fence rule 4, the partition cure). When this shard HOLDS its
/// realm but has had no round-trip confirmation (`RealmConfirmedAt`) within `self_fence_grace_ticks` of
/// its own `local_tick` — i.e. it is partitioned from the orchestrator, so the reactive `realm_recheck`
/// reply never arrives — it HARD-STOPS its own authority BEFORE the orchestrator's reassign horizon
/// opens (`should_reap` reaps only past `lease_expires + max_self_fence_grace`; the split-brain-safe
/// ordering `lease_ttl < grace` with `THETA_MAX*grace < lease_ttl + max` is enforced orchestrator-side by
/// `DirectoryTuning::validate`). The schedule runs
/// this before the egress systems, so a holder that self-fences this tick drops its held transients and
/// emits NO frames — two owners can never both reach clients. The reactive `realm_recheck` reply path
/// (a reply showing a takeover) remains the prompt cure while the link is ALIVE; this is the only path
/// that fires when it is NOT. `local_tick` is the partition-surviving clock (it advances every
/// `step_tick` regardless of `ClockSync`), so the measurement holds even with the universe clock frozen.
pub(crate) fn self_fence_lapsed_realm(
    config: Res<StubConfig>,
    clock: Res<ClockSample>,
    confirmed: Res<RealmConfirmedAt>,
    mut authority: ResMut<RealmAuthority>,
    mut owned_transients: ResMut<OwnedTransients>,
    mut stats: ResMut<StubStats>,
) {
    // The split-brain-critical predicate is the ONE shared `lease_self_fence_due` (DRY across every
    // authority holder — the shard's Realm here, the gateway's Session in connection-plane — so the
    // fence-rule-4 timing can never drift between them). `held` = this shard holds its realm.
    if crate::directory::lease_self_fence_due(
        authority.0.is_some(),
        config.self_fence_grace_ticks,
        config.realm_recheck_interval,
        clock.local_tick,
        confirmed.0,
    ) {
        authority.0 = None;
        self_fence_drop_transients(&mut owned_transients, &mut stats);
        stats.realm_self_fenced_lapsed += 1;
    }
}

/// Apply a realm-head affirm to the right authority store — the PRIMARY realm (`config.realm` ↦
/// `RealmAuthority`, the single-realm machinery, byte-identical) OR a CO-HOSTED CHILD realm
/// (`CoHostedAuthority`, the additive co-host store). Monomorphic so every branch is covered ONCE
/// (HR5). `record` is the head at the moment of the read (`None` = the record was revoked):
/// - PRIMARY realm, OURS  → hold `RealmAuthority` at the recorded fence + re-arm the self-fence clock.
/// - PRIMARY realm, FOREIGN/None → self-fence (drop `RealmAuthority` + declare the transients lost).
/// - CO-HOSTED realm, OURS → hold this child's `CoHostedAuthority` fence (independent of the primary).
/// - CO-HOSTED realm, FOREIGN/None → drop this child's entry (the shard no longer co-hosts it). The
///   transient loss path is PRIMARY-only (transients are anchored to `RealmAuthority`, never a child).
#[allow(clippy::too_many_arguments)]
fn affirm_realm_head(
    realm: RealmId,
    record: Option<&vd_wire::seams::directory::OwnerRecord>,
    identity: &NodeIdentity,
    config: &StubConfig,
    clock: &ClockSample,
    authority: &mut RealmAuthority,
    confirmed: &mut RealmConfirmedAt,
    cohosted: &mut CoHostedAuthority,
    owned_transients: &mut OwnedTransients,
    stats: &mut StubStats,
) {
    let ours = record.map(|r| r.authority) == Some(AuthorityRef::Shard(identity.node_id));
    if realm == config.realm {
        // The PRIMARY realm — the single-realm authority path, UNCHANGED.
        if ours {
            if authority.0.is_none() {
                tracing::debug!(realm = %config.realm, fence = ?record.map(|r| r.fence), "REALM LEASE AFFIRMED");
            }
            authority.0 = record.map(|r| r.fence);
            // D-3 Slice 5: a round-trip that AFFIRMS ownership re-arms the self-fence deadline — the
            // holder has just heard from the directory, so it is provably not partitioned now.
            confirmed.0 = clock.local_tick;
        } else {
            // Taken over (P2 transfer / reassignment) or revoked (record None): SELF-FENCE immediately
            // (fence rule 4) — drop authority and stop emitting so a stale old owner cannot affect clients.
            tracing::warn!("realm lease no longer held by this shard — self-fencing");
            authority.0 = None;
            // D-7: the transients were anchored to the now-lost lease, with no hand-off — a counted LOSS
            // (the declared-loss path; durable dots are retained by authority.rs).
            self_fence_drop_transients(owned_transients, stats);
        }
    } else if ours {
        // A CO-HOSTED CHILD realm we still hold: refresh its own fence (independent of the primary).
        cohosted
            .0
            .insert(realm, record.map_or(Fence::GENESIS, |r| r.fence));
    } else {
        // A CO-HOSTED CHILD realm taken over or revoked: drop the co-host entry (no transient loss — a
        // child realm never anchors this shard's transients; those ride `RealmAuthority`).
        cohosted.0.remove(&realm);
    }
}

/// VU AoI S2a-2b — cache the node owning this shard's PARENT realm from a realm Head reply, IFF the replied
/// realm is THIS shard's parent (`own_coord.parent().lowered()`). Monomorphic (all branching HERE, HR5): the
/// parent-match yes/no + the Shard / non-Shard / None record resolve. OVERWRITE (not merge) so a re-home's new
/// owner replaces the old and a revoked/absent record clears to `None` (skip, never emit to a dead node). A
/// no-op for any non-parent realm ⇒ purely additive to `affirm_realm_head` (byte-identical). NOTE: keys on the
/// LOSSY `lowered()` `RealmId`, exactly like the directory itself does today — both must migrate to
/// `path()`-keying together (DEFERRED); a boot `debug_assert` in `register_stub_shard` arms the alias case.
pub(crate) fn update_parent_node(
    realm: RealmId,
    record: Option<&vd_wire::seams::directory::OwnerRecord>,
    config: &StubConfig,
    parent_node: &mut ParentRealmNode,
) {
    // A realm with an EXTERIOR key learns its parent from that key alone (`resolve_exterior_head`);
    // a realm-head reply for its lineage parent must not fight it. Bitwise `&`: both sides pure.
    let from_lineage = DirectoryKey::exterior_of(config.realm).is_none();
    if from_lineage & (Some(realm) == config.own_coord.parent().map(|p| p.lowered())) {
        parent_node.0 = match record.map(|r| r.authority) {
            Some(AuthorityRef::Shard(n)) => Some(n),
            _ => None,
        };
        tracing::debug!(
            parent = %realm,
            node = ?parent_node.0,
            realm = %config.realm,
            "PARENT RESOLVED: the up-observation relay target",
        );
    }
}

/// The exterior keys this realm should be leasing but is not yet: every direct child that has one and
/// is absent from [`ExteriorAuthority`]. Monomorphic; a realm with no built children yields nothing.
pub(crate) fn exterior_keys_wanted(
    config: &StubConfig,
    regions: &RealmRegions,
    exterior: &ExteriorAuthority,
) -> Vec<(RealmId, DirectoryKey)> {
    // Only the BUILT children can want a lease, and the roster keeps them as a short list (SL9: the
    // galaxy has 233 220 children and one hull; asking each child whether it is a hull, every tick,
    // was measured at a fifth of its tick).
    regions
        .built_children()
        .iter()
        .filter(|c| regions.parent_of(**c) == Some(config.realm))
        .filter(|c| !exterior.0.contains_key(c))
        .filter_map(|c| DirectoryKey::exterior_of(*c).map(|key| (*c, key)))
        .collect()
}

/// The directory key this realm reads to learn its PARENT's node: its own exterior key when it has
/// one (whoever authors my placement is my parent), else its lineage parent's realm key. `None` at
/// the root.
pub(crate) fn parent_head_key(config: &StubConfig) -> Option<DirectoryKey> {
    DirectoryKey::exterior_of(config.realm).or_else(|| {
        config
            .own_coord
            .parent()
            .map(|p| DirectoryKey::Realm(p.lowered()))
    })
}

/// ★ A `Ship` HEAD REPLY — the exterior key's two readers, one arm (the ruler switch, slice 0):
/// - **the child:** if the key is MY exterior, its authority is my parent's node — overwrite
///   `ParentRealmNode` (a re-home's new author replaces the old; an absent or non-shard record clears
///   it, so nothing is ever sent to a node the directory no longer names);
/// - **the parent:** if the key names one of my DIRECT CHILDREN, affirm the lease when the directory
///   names ME, else drop it (the child is somebody else's now, or nobody's yet).
///
/// Monomorphic: all branching here (HR5). A `Ship` head for a realm that is neither is ignored.
pub(crate) fn resolve_exterior_head(
    entity: vd_core::EntityId,
    record: Option<&vd_wire::seams::directory::OwnerRecord>,
    self_node: NodeId,
    config: &StubConfig,
    regions: &RealmRegions,
    parent_node: &mut ParentRealmNode,
    exterior: &mut ExteriorAuthority,
) {
    let key = DirectoryKey::Ship(entity);
    let child = RealmId::Ship(entity);
    if DirectoryKey::exterior_of(config.realm) == Some(key) {
        parent_node.0 = match record.map(|r| r.authority) {
            Some(AuthorityRef::Shard(n)) => Some(n),
            _ => None,
        };
        tracing::debug!(
            node = ?parent_node.0,
            realm = %config.realm,
            "PARENT RESOLVED from my exterior key: whoever authors my placement",
        );
        return;
    }
    if regions.parent_of(child) == Some(config.realm) {
        match record {
            Some(r) if r.authority == AuthorityRef::Shard(self_node) => {
                exterior.0.insert(child, r.fence);
            }
            _ => {
                exterior.0.remove(&child);
            }
        }
        tracing::debug!(
            child = %child,
            held = exterior.0.contains_key(&child),
            realm = %config.realm,
            "EXTERIOR LEASE: do I author this child's placement",
        );
    }
}

/// Lane cure (findings 0/43, up half) — [`update_parent_node`]'s DOWNWARD twin: cache the node owning one
/// of this shard's DIRECT CHILD realms from a realm Head reply, IFF the replied realm is on this shard's
/// own roster. Monomorphic (all branching HERE, HR5): the roster-membership yes/no + the Shard /
/// non-Shard / None record resolve. `Some(Shard)` inserts (OVERWRITE — a re-home's new owner replaces the
/// old); anything else REMOVES (a revoked/absent record fails the up-lanes closed — refuse, never believe
/// a node the directory no longer names). This is the ADMISSION side only; the learned frame sender stays
/// the ROUTE (`ChildLiveEntry::home` — see [`ChildRealmNodes`]). A no-op for any non-child realm ⇒ purely
/// additive to `affirm_realm_head`/`update_parent_node` (byte-identical). Keys on the LOSSY `lowered()`
/// `RealmId`, exactly like the directory itself (D-RLM-10).
pub(crate) fn update_child_node(
    realm: RealmId,
    record: Option<&vd_wire::seams::directory::OwnerRecord>,
    config: &StubConfig,
    regions: &RealmRegions,
    child_nodes: &mut ChildRealmNodes,
) {
    // A LOOKUP, never a scan of the children (SL9, measured 2026-09-04 on the galaxy).
    if regions.parent_of(realm) == Some(config.realm) {
        match record.map(|r| r.authority) {
            Some(AuthorityRef::Shard(n)) => {
                child_nodes.0.insert(realm, n);
            }
            _ => {
                child_nodes.0.remove(&realm);
            }
        }
        tracing::debug!(
            child = %realm,
            node = ?child_nodes.0.get(&realm),
            realm = %config.realm,
            "CHILD RESOLVED: the up-lanes' admission authority",
        );
    }
}

/// Handle a directory reply: realm-lease and entity-grant confirmations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_directory_reply(
    bytes: &[u8],
    // The frame's sender — the attestation input for every peer-to-peer arm on this carrier (the
    // Q2 relay), passed through exactly like the Control arm's (stub dispatch, findings 0/43).
    from: NodeId,
    identity: &NodeIdentity,
    config: &mut StubConfig,
    clock: &ClockSample,
    regions: &mut RealmRegions,
    placements: &PlacementLedger,
    authority: &mut RealmAuthority,
    confirmed: &mut RealmConfirmedAt,
    cohosted: &mut CoHostedAuthority,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    pending: &mut PendingCrossings,
    pending_slots: &mut PendingInputSlots,
    registration: &mut GhostColliderRegistration,
    owned_transients: &mut OwnedTransients,
    in_flight: &mut RequestInFlight,
    progress: &mut CrossingProgress,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
    parent_node: &mut ParentRealmNode,
    // Lane cure (findings 0/43, up half) — the directory-derived ADMISSION map for this shard's direct
    // children, written by the same realm-Head arm that writes `parent_node`.
    child_nodes: &mut ChildRealmNodes,
    // The ruler switch, slice 0 — the exterior leases this realm holds for its built children.
    exterior: &mut ExteriorAuthority,
    // The ruler switch, slice 2 — the driven set, the liveness table and the store the exterior arms
    // write (a flush freezes a child; an adoption seeds one and writes its berth; a release drops it).
    driven: &mut crate::stub::drive::DrivenChildren,
    child_liveness: &mut crate::stub::aoi::ChildLiveness,
    store: &mut crate::stub::exterior::RealmStore,
    // The ruler switch, slice 3 — the lineage stores on both sides of a statement.
    lineage: crate::stub::lineage::LineageSide<'_>,
    // The per-child tables a release must clear (the hull's rows in each).
    tables: crate::stub::exterior::ChildTables<'_>,
    // Slice C1 — the Q2 relay holder the `WindowRelay` receive writes (held sealed, unopened).
    relay_held: &mut RelayHeld,
    // Look horizon slice 4 — the interest byte's holder the `RealmInterest` receive writes.
    interest_held: &mut InterestHeld,
    holds: &mut HandoffHolds,
) {
    // Decode the Saga-class envelope once and dispatch by arm. The orchestrator wraps a directory
    // answer in DirectoryReply; the 1d.1 transfer machinery adds the SOURCE's FlushSource request
    // and the DEST's Transfer (StubCrossing) envelope. D-7 adds the DEST's `TransientBatch` adopt
    // (inside `on_transfer_envelope`) + the orchestrator's `TransientDrop`. Other arms
    // (Ghost/Directory/Saga/SagaAck/TransferAck) never target a stub inbound and are ignored.
    let reply = match postcard::from_bytes::<InterShardFlow>(bytes) {
        Ok(InterShardFlow::DirectoryReply(reply)) => reply,
        // ★TOMBSTONE (window lane Slice C2, minor 19): `ChildSceneSet` — the parent's down-reflect
        // of scenery — has NO arm here any more. It was the one message that could tell a realm
        // about itself, and it needed the SL1 self-placement filter to keep from doing so; both
        // retire together by the owner's Q3 amendment (2026-08-16, window_lane.md §5 RULINGS).
        // A received frame falls to the closed fall-through below and counts `undecodable` on
        // this, its real carrier.
        //
        // THE Q2 RELAY RECEIVE (window lane Slice C1, mesh minor 17; owner-approved 2026-08-16 —
        // owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): a live direct
        // child's sealed self-authored statements, held UNOPENED for the per-tick forwarder.
        // Rides the reliable peer-to-peer carrier the down-reflect used to share, with the same
        // mis-route + attestation admission.
        Ok(InterShardFlow::WindowRelay(wr)) => {
            on_window_relay(wr, from, config, clock, child_nodes, relay_held, stats);
            return;
        }
        // THE INTEREST BYTE (look horizon slice 4, mesh minor 21; Q1 APPROVED, owner-approved
        // 2026-08-17 — look_horizon.md §2 ASK B): the parent's one-byte "somebody may look
        // inside you", on the same reliable peer carrier the relay rides, admitted fail-closed
        // against the resolved PARENT head.
        // The ruler switch, slice 3 — my parent tells me where I now stand in the tree.
        Ok(InterShardFlow::LineageStated(ls)) => {
            crate::stub::lineage::on_lineage_stated(
                ls,
                from,
                config,
                regions,
                parent_node,
                lineage.stated,
                lineage.was_occupied,
                lineage.pending,
                stats,
                outbox,
            );
            return;
        }
        Ok(InterShardFlow::RealmInterest(ri)) => {
            on_realm_interest(
                ri,
                from,
                config,
                clock,
                parent_node,
                interest_held,
                stats,
                outbox,
            );
            return;
        }
        // SOURCE: ship the held subject's pose (1d.1).
        Ok(InterShardFlow::FlushSource(flush)) => {
            on_flush_source(
                flush,
                config,
                regions,
                placements,
                clock.universe_tick,
                dots,
                driven,
                exterior,
                stats,
                outbox,
            );
            return;
        }
        // DEST: adopt the crossed entity state — a durable `StubCrossing` (1d.1) OR a `TransientBatch`
        // adopt-as-Arriving (D-7); both ride the `Transfer` arm, split by payload inside.
        Ok(InterShardFlow::Transfer(env)) => {
            on_transfer_envelope(
                env,
                clock.epoch,
                config,
                regions,
                placements,
                dots,
                applied,
                pending,
                owned_transients,
                crate::stub::exterior::AdoptSide {
                    driven,
                    exterior,
                    store,
                    owed: lineage.owed,
                },
                clock.universe_tick,
                stats,
                outbox,
            );
            return;
        }
        // D-7b structural drop-before-promote: SOURCE release (`Held→Departing` + ack), DEST promote
        // (`Arriving→Held` + ack), SOURCE complete (retire the `Departing` copy). The holder set
        // transits `{source}→{}→{dest}`, never `{source,dest}`.
        Ok(InterShardFlow::TransientRelease(rel)) => {
            on_transient_release(rel, owned_transients, applied, stats, config, outbox);
            return;
        }
        Ok(InterShardFlow::TransientDrop(promote)) => {
            on_transient_promote(promote, owned_transients, applied, stats, config, outbox);
            return;
        }
        Ok(InterShardFlow::ReleaseComplete(rc)) => {
            on_release_complete(rc, owned_transients, stats, config, outbox);
            return;
        }
        // SOURCE: the D-7d dead-DEST resolution — abandon this batch's retained copy as an accounted
        // loss (no ack: the resolving saga is already terminal; the fabric delivers reliably to the
        // live source, and a lost abandon falls back to the realm self-fence loss path).
        Ok(InterShardFlow::TransientAbandon(abandon)) => {
            on_transient_abandon(abandon, owned_transients, applied, stats);
            return;
        }
        // DEST: the R-6d3c NEVER-restart resolution — the source died in AwaitAdopt (pre-adopt), so
        // remove any late-replayed Arriving copy of this batch AND poison the adopt as an accounted loss
        // (no ack: the resolving saga is already terminal, exactly like TransientAbandon).
        Ok(InterShardFlow::TransientDiscard(discard)) => {
            on_transient_discard(discard, owned_transients, applied, stats);
            return;
        }
        // SOURCE: the saga-pushed ordered Demote (1d.5b.1) — Owned→Frozen→Ghost + DemoteAck. Slice 3e:
        // the durable COMMIT terminal at the source, so it also POSITIVELY clears the subject's
        // `RequestInFlight` crossing latch (a triggered durable crossing has committed — the entity is
        // free to trigger again from its new home).
        Ok(InterShardFlow::Demote(cmd)) => {
            on_saga_demote(
                cmd,
                config,
                clock,
                dots,
                in_flight,
                holds,
                crate::stub::exterior::ReleaseSide {
                    regions,
                    driven,
                    exterior,
                    child_nodes,
                    child_liveness,
                    relay_held,
                    store,
                    tables,
                    owed: lineage.owed,
                },
                stats,
                outbox,
            );
            return;
        }
        // SOURCE (Slice 3e): the orchestrator GRANTED a resolved dest for a `TransientCrossingRequest`
        // — flip the source Transient to `Crossing` so `emit_transient_batch` ships it this/next tick.
        // The grant's four fields EXACTLY match `TransientStatus::Crossing`. A grant for a
        // non-Entity subject or an unknown/settled transient is a counted no-op (degrade, never panic).
        Ok(InterShardFlow::TransientCrossingGrant(g)) => {
            on_transient_crossing_grant(g, owned_transients, stats);
            return;
        }
        // SOURCE (Slice 3e): the crossing resolve/start saga ABORTED pre-CAS — CLEAR the subject's
        // `RequestInFlight` latch IF it still holds THIS transfer id (the positive re-cross signal; the
        // full abort EGRESS is Slice 3f, but the consumer arm lands now so the wire arm is used). A
        // mismatch (a stale abort for a superseded/re-latched transfer) is a counted no-op.
        Ok(InterShardFlow::CrossingAborted(a)) => {
            on_crossing_aborted(
                a,
                in_flight,
                progress,
                holds,
                driven,
                stats,
                outbox,
                config.orchestrator,
            );
            return;
        }
        // DEST: the saga-pushed ordered Promote (1d.5b.3b) — the REAL Ghost→Owned promoter + the
        // dest read-sub announce + the source-ghost feed registration/Spawn + PromoteAck. The dest
        // normally holds its realm before the orchestrator routes a Promote to it (the realm-owner
        // invariant); if it does NOT (a realm self-fence raced the Promote — unreachable in P2, no
        // realm-revoke producer; reachable only at P8/P10 multi-realm mobility), DROP the Promote as
        // a counted no-op so the saga's `Promoting`-timeout re-drives it — DEGRADE, never panic
        // (mirroring every sibling handler; the realm-mobility re-drive is owed with D-3/Slice-2).
        Ok(InterShardFlow::Promote(cmd)) => {
            let Some(realm_fence) = authority.0 else {
                stats.promote_without_realm += 1;
                return;
            };
            on_saga_promote(
                cmd,
                config,
                identity.node_id,
                dots,
                applied,
                registration,
                realm_fence,
                stats,
                outbox,
            );
            return;
        }
        // D-37 forward re-home ADOPT (the target creates the entity Owned from the carried pose). The
        // realm guard mirrors Promote (the re-home target is committed by the orchestrator CAS so it
        // normally holds its realm; the guard is a counted degrade-never-panic no-op, never an unwrap).
        Ok(InterShardFlow::ReHome(cmd)) => {
            let Some(_realm_fence) = authority.0 else {
                stats.re_home_without_realm += 1;
                return;
            };
            on_re_home(
                cmd,
                config,
                regions,
                placements,
                identity.node_id,
                clock,
                dots,
                applied,
                registration,
                stats,
                outbox,
            );
            return;
        }
        // CA-1 S3: the orchestrator's AwaitAdopt liveness PROBE. A COUNTED NO-OP — the probe's whole signal
        // is its SEND OUTCOME at the orchestrator (a dead source's send fails → `NodeUnreachable`); a LIVE
        // source that receives it need do nothing but exist (its regular lease-renewal inbound is what clears
        // stale unreachable evidence). Counted for ops visibility; never a state change.
        Ok(InterShardFlow::ReSolicitBatch(_)) => {
            stats.re_solicits_received += 1;
            return;
        }
        // ★TOMBSTONES on THIS carrier — the deleted per-occupant scene reflect (`ProxySceneSet`,
        // minor 12) and the deleted per-live-child scene reflect that replaced it
        // (`ChildSceneSet`, minor 19). Both DECODE fine (their discriminants are reserved
        // forever), so the `Err` arm below can never see them and the silent `Ok(_)`
        // fall-through would swallow them uncounted. Counted HERE, on their own carrier — the
        // same "meaning is gone" accounting their SignalDelta twins get from that dispatch's
        // closed fall-through.
        Ok(InterShardFlow::ProxySceneSet(_) | InterShardFlow::ChildSceneSet(_)) => {
            stats.undecodable += 1;
            tracing::error!("tombstoned scene-reflect frame received");
            return;
        }
        Ok(_) => return,
        Err(_) => {
            stats.undecodable += 1;
            tracing::error!("undecodable saga-class message");
            return;
        }
    };
    match reply {
        DirectoryReply::Head {
            key: DirectoryKey::Realm(realm),
            record,
        } => {
            // A realm-head affirm for the PRIMARY realm drives the single-realm authority machinery
            // (unchanged); one for a CO-HOSTED CHILD realm drives only its own map entry — so the
            // co-hosting affirm reuses THIS one arm (no new dispatch arm), byte-identical for a
            // single-realm shard (the `else` branch is never reached when `held_realms == {realm}`).
            // Branching is hoisted into the monomorphic `affirm_realm_head` (HR5): the arm body is a
            // branchless shim.
            let had_lease = authority.0.is_some();
            affirm_realm_head(
                realm,
                record.as_ref(),
                identity,
                config,
                clock,
                authority,
                confirmed,
                cohosted,
                owned_transients,
                stats,
            );
            // Stage B2: the lease just landed — drain every input slot buffered while it was absent,
            // through the identical adopt path. The adopt HeadRead pump then flips each grant and
            // `PendingCrossings` drains at the flip, completing a crossing the dropped-slot posture
            // stranded permanently (rehome_one_mechanism §4v fact 3).
            if !had_lease && authority.0.is_some() {
                drain_pending_input_slots(pending_slots, config, clock, dots, stats);
            }
            // VU AoI S2a-2b — ALSO cache the node IFF this head is THIS shard's PARENT realm (the up-flow
            // target). Purely additive: `update_parent_node` no-ops for a non-parent realm, so the primary /
            // co-hosted authority machinery above is byte-identical.
            update_parent_node(realm, record.as_ref(), config, parent_node);
            // Lane cure (findings 0/43, up half) — and IFF it is one of this shard's DIRECT CHILDREN,
            // cache its node as the up-lanes' admission authority (same reply arm, same cadence, HR3).
            update_child_node(realm, record.as_ref(), config, regions, child_nodes);
            // The ruler switch, slice 3 — an adopted child's node is known: state its lineage to it
            // (again on every read, until its facts arrive).
            crate::stub::lineage::state_lineage_if_owed(
                realm,
                child_nodes.0.get(&realm).copied(),
                config,
                authority.0,
                clock.universe_tick,
                lineage.owed,
                outbox,
                stats,
            );
        }
        DirectoryReply::Head {
            key: DirectoryKey::Ship(entity),
            record,
        } => {
            resolve_exterior_head(
                entity,
                record.as_ref(),
                identity.node_id,
                config,
                regions,
                parent_node,
                exterior,
            );
            // The ruler switch, slice 3 — my exterior's holder is known: a held lineage statement from
            // that node is applied, one from any other node discarded.
            crate::stub::lineage::apply_pending_lineage(
                config,
                regions,
                parent_node,
                lineage.stated,
                lineage.was_occupied,
                lineage.pending,
                stats,
            );
        }
        DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: Some(record),
        } => {
            // The avatar's authority is now RECORDED: it becomes held, visible,
            // and attachable (fence rule 2 — authority derives from the directory).
            let Some(realm_fence) = authority.0 else {
                return; // grant raced ahead of the realm lease: retry resolves it
            };
            if record.authority == AuthorityRef::Shard(identity.node_id) {
                // OURS: flip granted + stamp the recorded fence. A login attach pushes
                // SessionAttached and Promotes Ghost→Owned (now simulates); a transfer-dest ADOPT
                // (1c.8) suppresses the attach (R2 — source owns the client) and STAYS a Ghost (no
                // pose carried; the saga `Promote` flips it in `on_saga_promote`, 1d.5b.3b —
                // `apply_crossing` only STORES the crossed pose). The
                // branching is hoisted into `flip_grant` → `GrantFlip`: LoggedIn pushes
                // SessionAttached; Adopted drains the buffered crossing (if any) NOW; NoOp is a
                // duplicate-grant idempotent no-op.
                match flip_grant(&mut dots.0, entity, record.fence, config, realm_fence) {
                    GrantFlip::LoggedIn(egress) => {
                        push_session_reply(outbox, egress.gateway, &egress.reply);
                    }
                    GrantFlip::Adopted { session } => {
                        // 1d.5b.3b: NO SubscriptionReady here (it moved to on_saga_promote). Just
                        // drain any buffered crossing onto the now-adopted Ghost dot (it stays Ghost
                        // and emits nothing until the saga Promote flips it Owned).
                        drain_pending_crossing(
                            &mut dots.0,
                            session,
                            entity,
                            applied,
                            pending,
                            config,
                            stats,
                            outbox,
                        );
                    }
                    GrantFlip::NoOp => {}
                }
            } else {
                // FOREIGN owner of an entity this shard holds: IGNORED here (1d.5b.2). The source
                // self-fence is now driven SOLELY by the saga-pushed ordered `Demote`
                // (`on_saga_demote`) — the 1c.8 cooperative poll that DISCOVERED a foreign takeover
                // from a directory head is gone. The EXPLICIT else keeps the outer `if`'s FALSE arm
                // a covered branch (a pre-grant race still surfaces a foreign-owner head, exercised by
                // `foreign_entity_grants_and_pregrant_races_are_survived`) — never an implicit
                // uncovered else.
            }
        }
        DirectoryReply::Head {
            key: DirectoryKey::Entity(entity),
            record: None,
        } => {
            // The revoke is RECORDED (no record remains): finish the release —
            // despawn and confirm to the gateway.
            let departed: Vec<SessionId> = dots
                .0
                .iter()
                .filter(|(_, d)| (d.entity == entity) & d.departing)
                .map(|(s, _)| *s)
                .collect();
            let any_departed = !departed.is_empty();
            for session in departed {
                let dot = dots.0.remove(&session).expect("just found");
                push_session_reply(
                    outbox,
                    dot.gateway,
                    &ShardToGateway::SessionDetached { session },
                );
            }
            // THE REMOVE MESSAGE (D-4(a)): a completed detach is a permanent stop — the departed
            // player's figure must leave every bystander's screen, not freeze on it. Emitted once
            // per entity (not per departed session), after the removals, so the fan reaches
            // exactly the remaining bystanders. Lease-gated like every emit.
            match (any_departed, authority.0) {
                (true, Some(fence)) => {
                    push_entity_removed(dots, fence, entity, clock.universe_tick, outbox);
                }
                // Same loud suppression as the despawn site: a detach completing on a shard
                // whose lease lapsed withholds the removal, counted, never silent.
                (true, None) => {
                    stats.entity_removals_suppressed_no_lease += 1;
                    tracing::warn!(%entity, "entity removal suppressed: no realm lease");
                }
                (false, _) => {}
            }
        }
        // Headless realm reads, CAS results, clock answers: no obligation in P1.
        _ => {}
    }
}

/// VU AoI S2a-2b — is a parent-realm directory read DUE this tick? Returns the PARENT [`RealmCoord`] to
/// resolve (its authoritative node is cached in [`ParentRealmNode`] from the reply), or `None` at the
/// containment ROOT (no parent), when the interest bands are inert (walk/static), or off the recheck
/// cadence. The parent node is the send target of this shard's UP-lanes — the one-bit `ChildLive`
/// heartbeat and the up-observation rows/outlines. (The per-occupant interest up-relay this resolve was
/// built for is DELETED — Step 5 slice D; no occupant data crosses, SL2.)
pub(crate) fn parent_headread_due(
    config: &StubConfig,
    regions: &RealmRegions,
    clock: &ClockSample,
) -> Option<DirectoryKey> {
    let key = parent_head_key(config)?;
    head_reads_due(config, regions, clock).then_some(key)
}

/// THE ONE FOLD for "are directory head-reads due this tick" (lane cure, findings 0/43 — HR3: the parent
/// resolve and the child admission reads share one cadence and one gate, so the two can never drift).
/// BITWISE `&` (not `&&`): both operands are cheap + pure, and a short-circuit would leave the RHS a
/// region HR5 can never cover from the false-LHS side (the discipline `AoiConfig::in_range` uses). The
/// inert walk/static bands make `aoi_live()` false ⇒ no read is ever emitted ⇒ byte-identical.
pub(crate) fn head_reads_due(
    config: &StubConfig,
    regions: &RealmRegions,
    clock: &ClockSample,
) -> bool {
    regions.aoi_live()
        & crate::directory::due_this_tick(aoi_recheck_cadence(config), clock.local_tick.0)
}
