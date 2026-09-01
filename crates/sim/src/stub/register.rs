//! THE WORLD BUILD AND THE INBOUND DISPATCH: where every lane in this module tree is installed,
//! and where every delivered message is handed to one.
//!
//! Owns: the one registration a shard binary calls, the schedule order the systems run in, and the
//! per-tick drain that routes each inbound message to its arm.
//!
//! Does NOT own: any per-kind fork. Registration is NEVER gated on a node kind — a shard type is a
//! capability config, so the same systems are installed everywhere and the `ShardProfile` decides
//! what they do, not which of them exist (HR3, HR4).

use super::{
    AoiMembership, AppliedSteps, ChildLiveness, ChildLuma, ChildRealmNodes, CoHostedAuthority,
    ContainmentProgress, CrossingProgress, Dots, EntityMint, FrameCounter, GatewayMsgCtx,
    GhostColliderRegistration, HandoffHolds, HoldRole, InBandVerdict, InputLog, InterestEmitLatch,
    InterestHeld, OpenWindows, OwnedTransients, ParentRealmNode, PendingCrossings,
    PendingInputSlots, Placements, RealmAuthority, RealmConfirmedAt, RealmRegions, RelayHeld,
    RelayShip, RequestInFlight, StubConfig, StubStats, WasOccupied, announce_presence,
    author_placements, emit_frames, emit_realm_frames, emit_transient_batch, evaluate_realm_aoi,
    evaluate_realm_boundaries, feed_source_ghosts, is_retained_ghost, on_directory_reply,
    on_gateway_msg, on_ghost_flow, placement_window_ticks, prune_holds, push_entity_removed,
    readvance_dots, readvance_transients, redrive_pending_adoptions, redrive_stranded_crossings,
    request_pending_grants, retain_child_live, self_fence_lapsed_realm,
};
use crate::io::{Inbound, MsgClass};
use crate::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};
use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Schedule, World};
use vd_core::placement::PlacementLedger;
use vd_core::rng::SplitMix64;
use vd_wire::intershard::InterShardFlow;

/// Install the stub-shard systems and resources onto a node's world + schedule.
/// Called EXPLICITLY by the shard bin (RLM Step 5a: for a `NodeKind::Shard(profile)`; was
/// `NodeKind::StubShard` pre-5a) — never by feature code, and NEVER gated on the node kind (the
/// carried `ShardProfile` is capability-inert at P1–P3, proven byte-identical by the 5a inertness gate).
pub fn register_stub_shard(world: &mut World, schedule: &mut Schedule, config: StubConfig) {
    // Slice 3e — FAIL-LOUD boundary-tuning validation at boot (mirroring the tick-pair guard): a
    // zero dwell/pad/cell would silently disable anti-flap or divide the cell rebase. The registry is
    // EMPTY in prod (the trigger is inert), but the tuning is validated regardless so a misconfigured
    // deployment never boots a half-armed trigger.
    config
        .boundary
        .validate()
        .expect("StubConfig.boundary is a valid BoundaryTuning (n_entry/k_dwell/pad/cell > 0)");
    // RLM Step 2 (L3 co-hosting hygiene): the AoI loop (`aoi_decide`) builds exactly ONE `own_coord` per
    // shard from `config.realm`, so it names the PRIMARY realm's direct children — correct for
    // node-per-realm (`held_realms == {realm}`, the base; co-hosting is D-44 KEPT-unused). A co-hosting
    // shard's CO-HOSTED realms' children are simply not AoI-evaluated here (a per-held-realm coord loop is
    // owed if/when D-44 is revived) — an incompleteness, NOT a mis-key, so it is safe for dormant infra
    // and NOT a boot tripwire (an unconditional `held_realms.len() == 1` panic would break the dormant
    // co-hosting grant/affirm/re-home tests, which legitimately build multi-realm shards). The live-AoI
    // composer (RLM Step 5/6) builds node-per-realm shards, so the primary path is always complete.
    // RLM Step 2 (M-2 single-source): the AoI hysteresis BANDS are baked into each `RealmRegion.aoi` at
    // GENERATION from `UniverseConfig.interest` (occupant_v_max_mps + tick_dt_s), and the runtime horizon
    // reads `StubConfig.tick_dt_s` — so the two must agree on `tick_dt_s`. `register_stub_shard` sees only
    // `StubConfig` (never `UniverseConfig`), so the cross-check belongs at the COMPOSER boot where both
    // meet (the live-AoI shard wiring, RLM Step 5/6); `InterestConfig` already carries the inputs for it.
    // RLM 5f-4a: `UniverseConfig::walk_demand(occupant_v_max_mps, tick_dt_s)` now takes BOTH as RUNTIME
    // arguments (no hardcoded `AOI_TICK_DT_S` — which is 0.05 and would be WRONG at the dev cluster's 50 Hz =
    // 0.02), so the composer (5f-4b) passes the live cluster's `tick_dt_s`/`move_speed·time_multiplier` and
    // the two homes agree BY CONSTRUCTION; the `debug_assert!` cross-check lands with that composer wiring.
    // Capture the scalar params read AFTER the config move (`StubConfig` is no longer `Copy` — the
    // `held_realms` set is heap-backed).
    let mint_seed = config.mint_seed;
    let input_log_capacity = config.input_log_capacity;
    // VU AoI S2a-2b [MAJOR guard] — the parent-realm resolve keys the parent's `HeadRead` on the LOSSY
    // `RealmId` (`lowered()`), exactly like the directory. `RealmLevel::to_realm_id` collapses Galaxy →
    // System(1), so a shard whose PARENT lowers to the SAME `RealmId` as itself would drive its own
    // parent-Head reply through the PRIMARY-FOREIGN self-fence branch (a false self-fence of a healthy
    // shard). That universe is UNREPRESENTABLE in today's lowered-keyed directory (the two realms would
    // share a directory key), so this is a boot tripwire, not a runtime path — and it must migrate to
    // `path()`-keying together with the directory (see DEFERRED.md). `debug_assert` (dev/test only): the
    // release directory is already lowered-keyed, so a colliding universe cannot boot there either.
    debug_assert!(
        config.own_coord.parent().map(|p| p.lowered()) != Some(config.realm),
        "own parent aliases own RealmId under lowered() — the parent resolve would self-fence; \
         migrate parent-resolve + directory to path()-keying together (DEFERRED)",
    );
    // The placement ledger's window derives from the config BEFORE the config moves into the world.
    let placement_window = placement_window_ticks(&config);
    world.insert_resource(config);
    world.insert_resource(Placements(PlacementLedger::new(placement_window)));
    world.insert_resource(Dots::default());
    world.insert_resource(RealmAuthority::default());
    world.insert_resource(CoHostedAuthority::default());
    world.insert_resource(RealmConfirmedAt::default());
    world.insert_resource(EntityMint {
        seq: 0,
        rng: SplitMix64::new(mint_seed),
    });
    world.insert_resource(InputLog::new(input_log_capacity));
    world.insert_resource(FrameCounter::default());
    world.insert_resource(StubStats::default());
    world.insert_resource(AppliedSteps::default());
    world.insert_resource(PendingCrossings::default());
    world.insert_resource(PendingInputSlots::default());
    world.insert_resource(GhostColliderRegistration::default());
    world.insert_resource(OwnedTransients::default());
    // task #135 — the CONTAINMENT trigger state. `RealmRegions` defaults EMPTY, so
    // `evaluate_realm_boundaries` early-returns in prod (inert through C-3; seed boot-population is
    // C-5/C-6); tests populate it via `RealmRegions::new`.
    world.insert_resource(CrossingProgress::default());
    world.insert_resource(ContainmentProgress::default());
    world.insert_resource(RequestInFlight::default());
    world.insert_resource(HandoffHolds::default());
    world.insert_resource(RealmRegions::default());
    // RLM Step 2 — the per-CHILD AoI hysteresis ledger. Defaults EMPTY; `evaluate_realm_aoi` is inert
    // (early-returns) until regions are planted AND the shard is clock-synced, so this is byte-identical
    // through walk/canonical scale (inert AoI ⇒ no demand ⇒ never touched).
    world.insert_resource(AoiMembership::default());
    world.insert_resource(ParentRealmNode::default());
    world.insert_resource(ChildRealmNodes::default());
    world.insert_resource(WasOccupied::default());
    world.insert_resource(ChildLiveness::default());
    world.insert_resource(OpenWindows::default());
    world.insert_resource(RelayShip::default());
    world.insert_resource(RelayHeld::default());
    // Look horizon slice 3 — the §3.4.5 forward gate's store. Defaults EMPTY (no verdict, no
    // interior forward) until the AoI fold writes one; byte-identical wherever AoI is inert.
    world.insert_resource(InBandVerdict::default());
    // Look horizon slice 4 — the interest byte's two stores: the CHILD-side holder (the
    // down-proxy's power source) and the PARENT-side per-child emission latch. Both default
    // empty ⇒ byte-identical until the lane fires.
    world.insert_resource(InterestHeld::default());
    // D-MOVE-2 — every driven child this realm holds: where each is, how it is moving, and the
    // freshest push it stated. Empty on a realm that holds none, which is most of them.
    world.insert_resource(crate::stub::drive::DrivenChildren::default());
    // D-MOVE-2 — what this realm IS, read from its own file at boot, and what it last told its parent.
    // Both empty on a realm the seed made, which is most of them.
    world.insert_resource(crate::stub::drive::OwnBody::default());
    world.insert_resource(crate::stub::drive::StatedFacts::default());
    world.insert_resource(InterestEmitLatch::default());
    world.insert_resource(ChildLuma::default());
    // `feed_source_ghosts` runs AFTER `process_inbound` (this tick's promote has registered the
    // neighbor + the dest dot is Owned) and BEFORE `emit_frames` (the source consumes the Delta it
    // received this tick before emitting) — the dest→source ghost collider feed (1d.5b.3b).
    // `readvance_transients` (D-7b) runs AFTER `process_inbound` (this tick's promote/drop settled the
    // held-set) and BEFORE `emit_transient_batch` (a crossing item is emitted with its CURRENT
    // re-advanced pose, keeping the source/dest origins consistent). `emit_transient_batch` (D-7) runs
    // AFTER `process_inbound` and is independent of the ghost/frame egress — it ships the source's
    // pending transient crossings as ONE batch per dest realm.
    // `self_fence_lapsed_realm` (D-3 Slice 5) runs AFTER `process_inbound` (so THIS tick's realm-head
    // confirmation has already refreshed `RealmConfirmedAt` — a fresh reply pre-empts a spurious
    // self-fence) and BEFORE the egress systems (a holder that self-fences this tick drops its
    // transients and emits NO frames this tick — a stale partitioned owner stops affecting clients at
    // once). INERT unless `self_fence_grace_ticks > 0` (the pre-D-3 default and every single-shard rig).
    // `redrive_pending_adoptions` (CA-1 S3/S4) runs AFTER `process_inbound` (this tick's promote/discard
    // settled the Arriving set — a just-promoted item is Held, not re-driven) and AFTER
    // `self_fence_lapsed_realm` (a self-fenced holder already dropped its Arriving items, so it re-drives
    // nothing). It is an orchestrator-bound egress like `emit_transient_batch`.
    // `evaluate_realm_boundaries` (Slice 3e) runs AFTER `self_fence_lapsed_realm` (a holder that lost its
    // realm this tick emits NO crossing request — the authority gate short-circuits) and BEFORE
    // `readvance_transients` / `emit_transient_batch`: a TRANSIENT crossing it flags emits a
    // `TransientCrossingRequest` this tick, and the source's own position (read from the settled
    // post-inbound pose) drives the swept segment. Its `CrossingRequest`/`TransientCrossingRequest`
    // egress is orchestrator-bound like `emit_transient_batch`. INERT unless `RealmBoundaries` is
    // non-empty (the composer plants none through P3 — behaviour-identical).
    // The full per-tick order is a single strict chain. Bevy's system-tuple `.chain()` supports at most
    // 8 direct elements, so the 9 systems are expressed as two chained groups joined by
    // `.after(readvance_dots)` — the SAME total order as one flat `.chain()` (group A ends at
    // `self_fence_lapsed_realm`; group B is itself chained and runs strictly after it). Splitting here is
    // a mechanical arity workaround, NOT a semantics change.
    schedule.add_systems(
        (
            // THE ONE PLACEMENT WRITER, at the head of the tick (the placement arc S2): after
            // `observe_clock_syncs` (registered earlier on this SingleThreaded schedule), before any
            // consumer — so every system this tick reads the SAME authored rows. `has_synced`-gated:
            // a pre-sync clock must not author placements (D-Finding-1, same gate as every author).
            author_placements.run_if(has_synced),
            request_pending_grants,
            process_inbound,
            // D-MOVE-2 — a self-driven realm states its push AFTER this tick's inbound, so a stick that
            // arrived this tick is the one that travels. Placed here rather than at the head for that
            // reason: at the head it would always send yesterday's stick, adding a tick of lag to every
            // control input for no gain.
            crate::stub::drive::emit_own_drive.run_if(has_synced),
            // RLM 5f RG-1: the reactive greeting — UNGATED (fires pre-sync/pre-lease, reachability precedes
            // authority) and after `process_inbound` so THIS tick's inbound counts as contact. No-op unless a
            // demand boot inserted `PresenceAnnounce`, so group order is byte-identical for static rigs.
            announce_presence,
            self_fence_lapsed_realm,
            // Stage B4 (§4u refutation 8): re-stamp input-idle durable poses to the current tick, AFTER
            // `process_inbound` (this tick's input already integrated — an active dot is a dt=0 no-op) and
            // BEFORE `evaluate_realm_boundaries` (group B is `.after` this system), so the containment scan
            // and the flush measure a parked occupant against the LIVE world, never one frozen at its
            // last-input instant. `has_synced`-gated: a pre-sync clock must not stamp anything.
            readvance_dots.run_if(has_synced),
        )
            .chain(),
    );
    // RLM Step 2 (M-1 retrofit, behaviour-CHANGING by design): `evaluate_realm_boundaries` and
    // `emit_realm_frames` are the two AUTHORING systems — both previously gated ONLY on `RealmAuthority`
    // (`authority.0`), the D-Finding-1 hole where a fresh shard authors at tick 0 BEFORE its first
    // `ClockSync` (a pre-sync celestial pose is wrong). `.run_if(has_synced)` closes it. Both early-return
    // on empty snaps/regions, so at walk/static scale this is inert (byte-identical); at visual scale the
    // first-sync boundary is exactly what must be gated. `ClockSample` is persistent and both systems sit
    // AFTER `observe_clock_syncs` on the shared schedule, so reading `synced` is order-correct.
    // The per-tick FrameAbs producer used to be ordered in here, ahead of the feed group, so every reader
    // saw ONE folded absolute per tick. There is no fold and no table any more — a shard authors its
    // children in its own frame and ships that — so the ordering it needed is gone with it. Group B's own
    // `.after(readvance_dots)` (group A's tail) already expresses the order that remains.
    schedule.add_systems(
        (
            evaluate_realm_boundaries.run_if(has_synced),
            redrive_stranded_crossings,
            readvance_transients,
            emit_transient_batch,
            redrive_pending_adoptions,
            feed_source_ghosts,
            emit_frames,
            emit_realm_frames.run_if(has_synced),
        )
            .chain()
            .after(readvance_dots),
    );
    // RLM Step 2 — the demand-driven realm-lifecycle detector, a THIRD chained group (group B is at
    // Bevy's 8-`.chain()` arity limit). Runs strictly AFTER `emit_realm_frames` (the author tail — the AoI
    // decision reads the SAME per-tick child placements the observer feed just shipped, H-1) and
    // `.run_if(has_synced)` (a fresh shard demands nothing pre-sync — determinism).
    schedule.add_systems(
        (evaluate_realm_aoi,)
            .chain()
            .after(emit_realm_frames)
            .run_if(has_synced),
    );
}

/// The receive-side VU AoI stores, bundled into ONE tuple `SystemParam` because bevy caps a system at 16
/// top-level params and this is the last slot: the read-only region forest (its `frame_context` REBASES a
/// flushed crossing pose into the dest realm's live frame — the moving-realm crossing fix) and the cached
/// parent-realm node (parent Head reply) beside the lanes' own stores. All are receive-side; grouping
/// them is a mechanical arity fix, not a coupling change, and it is destructured straight back into the
/// individual borrows at the top of the system.
type VuAoiInbound<'w> = (
    Res<'w, RealmRegions>,
    // The placement ledger — READ-ONLY here (the one writer is `author_placements`): the arrival and
    // flush ingress arms select their books from it.
    Res<'w, Placements>,
    ResMut<'w, ParentRealmNode>,
    // Lane cure (findings 0/43) — the directory-derived admission map for this shard's direct children:
    // written by the realm-Head reply arm, consulted by every up-lane receive.
    ResMut<'w, ChildRealmNodes>,
    // Step 5 slice A + the Q2 relay: the two parent-side stores their receive arms write — the SL7
    // bit table and the sealed relay holder.
    ResMut<'w, ChildLiveness>,
    ResMut<'w, RelayHeld>,
    // Look horizon slice 4 — the CHILD-side interest holder the `RealmInterest` receive writes.
    ResMut<'w, InterestHeld>,
    ResMut<'w, crate::stub::drive::DrivenChildren>,
);

/// Drain and dispatch everything delivered this tick.
/// Was this reliable-carrier message a child's declared facts? If so it is consumed here and the
/// caller does nothing more with it (D-MOVE-2).
///
/// Decoded ONCE. Peeking and then decoding again would repeat the work on every directory reply in
/// the world, and the "cannot happen" branch that shape needs is an arm no input can reach —
/// permanently uncovered, which is exactly what the coverage discipline exists to prevent.
#[allow(clippy::too_many_arguments)]
fn took_child_facts(
    bytes: &[u8],
    from: vd_core::ids::NodeId,
    own_realm: vd_core::pose::RealmId,
    integrates: bool,
    child_nodes: &std::collections::BTreeMap<vd_core::pose::RealmId, vd_core::ids::NodeId>,
    driven: &mut crate::stub::drive::DrivenChildren,
    stats: &mut StubStats,
) -> bool {
    match postcard::from_bytes::<InterShardFlow>(bytes) {
        Ok(InterShardFlow::ChildFacts(cf)) => {
            crate::stub::drive::on_child_facts(
                cf, from, own_realm, integrates, child_nodes, driven, stats,
            );
            true
        }
        // Everything else on this carrier is the directory's business, INCLUDING a message that does
        // not decode: reporting that is the reply path's job, not this one's.
        _ => false,
    }
}

/// Does THIS shard do the physics for what is inside it (D-MOVE-2)?
///
/// Read from the node's own identity, which already carries its profile — so the capability decides
/// what an installed system DOES, never which systems exist. The same systems are installed on every
/// shard (HR3), and a realm that does no physics refuses a drive and counts the refusal rather than
/// silently having no arm for it.
fn integrates_children(identity: &NodeIdentity) -> bool {
    match identity.kind {
        crate::capability::NodeKind::Shard(profile) => profile.integrates_children(),
        // Every other node kind — a gateway, an orchestrator, the relay, the test stub — holds no
        // realm and therefore integrates nothing. A drive reaching one is a misroute by definition.
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn process_inbound(
    config: Res<StubConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    inbox: Res<InboundBox>,
    mut dots: ResMut<Dots>,
    // Bundled realm-authority tuple `SystemParam` (bevy's 16-param ceiling): the primary realm fence, its
    // confirmation timestamp, AND the co-hosted child-realm fence map are all the realm-authority stores,
    // so grouping them is a mechanical arity fix. Destructured to three `&mut` below.
    realm_auth: (
        ResMut<RealmAuthority>,
        ResMut<RealmConfirmedAt>,
        ResMut<CoHostedAuthority>,
    ),
    mut mint: ResMut<EntityMint>,
    mut log: ResMut<InputLog>,
    mut stats: ResMut<StubStats>,
    mut applied: ResMut<AppliedSteps>,
    // Bundled tuple `SystemParam` (bevy's 16-param ceiling): the two dest-side pending buffers — the
    // entity-state crossing awaiting its adopt (1d.1) and the input slot awaiting the realm lease
    // (Stage B2) — plus the window registry the gateway-lane control arms register into (Slice A;
    // riding this tuple is the same mechanical arity fix). Destructured below.
    pending: (
        ResMut<PendingCrossings>,
        ResMut<PendingInputSlots>,
        ResMut<OpenWindows>,
    ),
    // The ghost-path store (slice F shrank the pair to one: the source-side feed mirror died with
    // the pose feed).
    mut registration: ResMut<GhostColliderRegistration>,
    mut owned_transients: ResMut<OwnedTransients>,
    // Bundled tuple `SystemParam` (bevy's 16-param ceiling): 3f-D threads `CrossingProgress` into the
    // `CrossingAborted` demux (for the L1 dwell re-arm + attempt bump) — both are the crossing-trigger
    // latch/dwell stores, so grouping them is a mechanical arity fix. Destructured to two `&mut` below.
    crossing: (
        ResMut<RequestInFlight>,
        ResMut<CrossingProgress>,
        ResMut<HandoffHolds>,
    ),
    mut outbox: ResMut<OutboundBox>,
    // Bundled tuple `SystemParam` (bevy's 16-param, LAST slot) — see [`VuAoiInbound`].
    vu_aoi: VuAoiInbound,
) {
    let (
        regions,
        placements,
        mut parent_node,
        mut child_nodes,
        mut child_liveness,
        mut relay_held,
        mut interest_held,
        mut driven,
    ) = vu_aoi;
    let (mut in_flight, mut progress, mut holds) = crossing;
    let (mut pending, mut pending_slots, mut open_windows) = pending;
    let (mut authority, mut confirmed, mut cohosted) = realm_auth;
    // UNGATED, and first: an expired hold must be reclaimed even on a shard that has lost its lease and
    // is doing nothing else, or the ledger would outlive the thing it describes. Inert at a zero budget.
    // The EVICTION for an expired Source hold fans AFTER the inbound loop below (review-caught): the
    // lease it is gated on may arrive in THIS tick's directory replies, and a one-shot suppressed on a
    // lease that was one message away would strand the phantom it exists to clear.
    let expired_holds = prune_holds(&mut holds, clock.local_tick, config.handoff_hold_ttl_ticks);
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            // Unreachability notices are observed by the node shell (TickReport);
            // the stub has no retry obligations in P1.
            continue;
        };
        match class {
            MsgClass::Control | MsgClass::Input => {
                let ctx = GatewayMsgCtx {
                    config: &config,
                    identity: &identity,
                    clock: &clock,
                    realm_fence: authority.0,
                    regions: &regions,
                    placements: &placements.0,
                };
                on_gateway_msg(
                    bytes,
                    *from,
                    &ctx,
                    &mut dots,
                    &mut mint,
                    &mut log,
                    &mut pending_slots,
                    &mut open_windows,
                    &mut stats,
                    &mut outbox,
                );
            }
            // D-MOVE-2 — what a child IS: its mass, cross-section, drag coefficient and declared
            // states. On the RELIABLE carrier, because a change stated once and then lost would leave
            // this realm computing drag from a mass that is wrong for ever, with nothing to correct it.
            //
            // Decoded ONCE and matched, rather than peeked at and decoded again: a second decode of
            // the same bytes is wasted work on every directory reply in the world, and the `else`
            // branch it needs is an arm no input can reach — permanently uncovered, which the coverage
            // rule forbids for exactly this reason.
            // D-MOVE-2 — the reliable carrier now serves TWO arms, so it decodes once and asks which.
            // What a child IS (its mass, cross-section, drag coefficient and declared states) must
            // arrive reliably: a change stated once and then lost would leave this realm computing
            // drag from a mass that is wrong for ever, with nothing to correct it.
            MsgClass::Saga if took_child_facts(
                bytes,
                *from,
                config.realm,
                integrates_children(&identity),
                &child_nodes.0,
                &mut driven,
                &mut stats,
            ) => {}
            MsgClass::Saga => on_directory_reply(
                bytes,
                *from,
                &identity,
                &config,
                &clock,
                &regions,
                &placements.0,
                &mut authority,
                &mut confirmed,
                &mut cohosted,
                &mut dots,
                &mut applied,
                &mut pending,
                &mut pending_slots,
                &mut registration,
                &mut owned_transients,
                &mut in_flight,
                &mut progress,
                &mut stats,
                &mut outbox,
                &mut parent_node,
                &mut child_nodes,
                &mut relay_held,
                &mut interest_held,
                &mut holds,
            ),
            // 1d.5b.3b: the SOURCE-side ghost feed consumer — Spawn/Delta/Despawn from the dest owner
            // refresh this shard's RETAINED ghost dot (kinematic collider; pose+GhostRefresh, never a
            // second authority store). On the dedicated Ghost carriers, NOT the Saga dispatch.
            MsgClass::GhostReliable | MsgClass::GhostDelta => {
                on_ghost_flow(
                    bytes,
                    &clock,
                    authority.0,
                    &mut dots,
                    &mut holds,
                    &mut stats,
                    &mut outbox,
                );
            }
            // Membership (clock sync) is consumed by the node-level follower system;
            // Snapshot / RealmSnapshot are gateway→client render datagrams and never target a shard.
            MsgClass::Membership | MsgClass::Snapshot | MsgClass::RealmSnapshot => {}
            // The up-lanes' carrier. A malformed / mis-classed payload — and a frame of any
            // TOMBSTONED lane that rode it (`OccupantInterest`, slice D; `EntityInterest`/
            // `EntityCascade`, slice E; discriminants reserved forever) — is counted as
            // undecodable, never a panic.
            MsgClass::SignalDelta => match postcard::from_bytes::<InterShardFlow>(bytes) {
                // Step 5 slice A — a direct child's SL7 occupancy bit. Presence-is-the-bit; last-wins
                // by (fence, at); the sender NodeId is the return address for the down-lanes.
                // D-MOVE-2 — a child realm's per-tick push and turn, in its OWN frame. The carrier is
                // the unreliable up-lane because the next tick restates the whole intent: a lost
                // datagram is corrected before anybody could read the gap.
                Ok(InterShardFlow::ChildDrive(cd)) => {
                    crate::stub::drive::on_child_drive(
                        cd,
                        *from,
                        config.realm,
                        integrates_children(&identity),
                        &child_nodes.0,
                        &mut driven,
                        &mut stats,
                    );
                }
                Ok(InterShardFlow::ChildLive(cl)) => {
                    retain_child_live(
                        &mut child_liveness,
                        &config,
                        cl,
                        clock.local_tick,
                        *from,
                        &child_nodes,
                        &mut stats,
                        &mut outbox,
                    );
                }
                // ★TOMBSTONES (window lane Slice C1/C2, minors 17/19): `RealmShapeObservation`
                // (32), `RealmObservation` (31) and `RealmCascade` (26) have NO arm here any more
                // — a received frame of any of them falls to the `_` fall-through below and counts
                // `undecodable` on its real carrier, the established tombstone accounting. The
                // whole picture leaves this shard through its own windows now
                // (`ShardToGateway::WindowFrame`/`WindowBody`/`WindowMembership`), and for a realm
                // an observer is BESIDE rather than inside, through the SEALED `WindowRelay` — a
                // reliable Saga-class carrier received in `on_directory_reply`.
                //
                // ★TOMBSTONED lanes fall through here and are COUNTED: the per-occupant cull hint
                // (`OccupantInterest`, slice D), the entity lane's two legs
                // (`EntityInterest`/`EntityCascade`, slice E) and the three scenery lanes above
                // all rode this carrier; their frames still decode (reserved discriminants) and
                // land in this closed fall-through.
                _ => stats.undecodable += 1,
            },
        }
    }
    // The expired-hold evictions, fanned with the lease as THIS TICK's inbound left it (see the
    // prune note above): an expired Source hold with a still-hosted retained ghost is a
    // leaver-vanish the take-over proof never delivered (a logout mid-crossing) — lease-gated,
    // loud on suppression, exactly like the proof-driven emit.
    for (entity, role) in expired_holds {
        let ghost_hosted = dots
            .0
            .values()
            .any(|d| (d.entity == entity) & is_retained_ghost(d));
        match (matches!(role, HoldRole::Source) & ghost_hosted, authority.0) {
            (true, Some(fence)) => {
                push_entity_removed(&dots, fence, entity, clock.universe_tick, &mut outbox);
            }
            (true, None) => {
                stats.entity_removals_suppressed_no_lease += 1;
                tracing::warn!(%entity, "entity removal suppressed: no realm lease");
            }
            (false, _) => {}
        }
    }
}

/// RLM Step 2 — the run-condition: a fresh shard AUTHORS NOTHING (its celestial poses, boundary crossings,
/// AoI demands) until its clock is LIVE (D-Finding-1). `ClockSample.synced` latches `true` on the first
/// `ClockSync` and never resets, so once synced every subsequent tick runs the gated authors. Reading the
/// persistent `ClockSample` after `observe_clock_syncs` (same schedule) is order-correct.
fn has_synced(clock: Res<ClockSample>) -> bool {
    clock.synced
}
