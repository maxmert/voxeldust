//! P3 D-7a — the TRANSIENT/debris transfer class, first slice (PERMANENT gate; PLAN.md P3 + HR2).
//!
//! "A debris chunk crosses a shard boundary on the SAME machinery as a player — but batched: ONE
//! orchestrator go-token per batch (never a per-entity directory row), held-set anchored, handed
//! over adopt-before-drop." This is the second day-one transfer triple (`{Transient,
//! BallisticReadvance, …}`) proving HR2: a new durability class is policy fan-out on the ONE saga
//! commit point, not a second machine.
//!
//! D-7a SCOPE: a 1-item Debris batch crosses source→dest, adopt-before-drop, the short FSM path
//! reaches `Done`, exactly ONE go-token is recorded, TRANSIENT-AUTHORITY-HELD holds, no DoubleHeld,
//! no LOSS. The closed-form ballistic re-advance + per-tick conservation (D-7b), the burst/G-TIER
//! scale proof (D-7c), and the crash-matrix transient cells (D-7d) build on this.

use std::collections::BTreeSet;
use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_core::{AccountId, BatchId, EntityId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_harness::fabric::{CrashWhen, FaultFabric};
use vd_harness::oracle::{verify_transient_authority_held, verify_transient_conservation_tick};
use vd_harness::topology::{InspectReport, StaggerPlan, Topology};
use vd_sim::saga::SagaCtx;
use vd_tests::{
    DEST, EndState, Fault, SHARD, Scenario, TRANSIENT_SEED_POS0, TRANSIENT_SEED_TICK0,
    assert_transient_end_state, dest_stub_config, dest_unreachable_resolutions, durable_subset,
    held_at_dest, live_sagas, liveness_notices, p1_client, p2_cluster, p2_cluster_staggered,
    read_subject, realm_fence, run_transient_dest_flap, run_transient_fault_scenario,
    seed_transient_burst, seed_transient_crossing, source_transients_emitted,
    source_unreachable_resolutions, transient_dropped_total, trigger_transfer, walk_forward,
};
use vd_wire::seams::directory::DirectoryKey;

/// The `BatchHandoff` phase strings the D-7d crash cells target (matched as a `starts_with` prefix on
/// the saga `Debug` — the deterministic drive-to-phase observable). ONE home each (no inline literal).
const AT_AWAIT_RELEASE: &str = "BatchHandoff { phase: AwaitRelease";
const AT_AWAIT_PROMOTE: &str = "BatchHandoff { phase: AwaitPromote";

/// The durable-warmup client (one client per scenario; id is local to these tests).
const CLIENT: NodeId = NodeId(100);

/// A Transient batch saga ctx for `batch` (D-7c) — the session-less transient trigger shape, shared by
/// the G-TIER burst tests (DRY).
fn transient_batch_ctx(batch: TransferId, dst_fence: vd_core::Fence) -> SagaCtx {
    SagaCtx {
        transfer: batch,
        session: SessionId(0),
        subject: DirectoryKey::Realm(DST_REALM),
        expected_fence: dst_fence,
        source: SHARD,
        dest: DEST,
        class: DurabilityClass::Transient,
        needs_provision: false,
        from_realm: SRC_REALM,
        to_realm: DST_REALM,
    }
}

/// Assert the DEST holds `debris` at a pose that lies on its closed-form ballistic trajectory from the
/// seeded origin (D-7b: no teleport across the cut, correct re-advance, NO double-advance). The dest's
/// re-advanced pose at universe-tick `t` must equal `pos0 + vel·(t - tick0)·tick_dt_s` — proving the
/// source's incremental advance + the dest's re-advance compose to the one trajectory. `pos0`/`tick0`
/// come from the SHARED [`TRANSIENT_SEED_POS0`]/[`TRANSIENT_SEED_TICK0`] (one source with the seed
/// helper — the expected trajectory can never silently drift from what was seeded).
fn assert_debris_on_ballistic_trajectory(topo: &mut Topology, debris: EntityId, vel: DVec3) {
    let reports = topo.inspect_all();
    let dest_pose = reports
        .iter()
        .flat_map(|(_, r)| r.held_transient_poses.iter())
        .find(|(e, _)| *e == debris)
        .map(|(_, p)| *p)
        .expect("the dest holds the debris pose");
    let dt_s =
        (dest_pose.universe_tick.0 - TRANSIENT_SEED_TICK0.0) as f64 * dest_stub_config().tick_dt_s;
    let expected = TRANSIENT_SEED_POS0 + vel * dt_s;
    assert!(
        (dest_pose.pos.offset() - expected).length() < 1.0e-6,
        "debris off its ballistic trajectory: got {:?}, expected {expected:?} (dt_s={dt_s})",
        dest_pose.pos
    );
    assert!(
        (dest_pose.pos.offset() - TRANSIENT_SEED_POS0).length() > 1.0,
        "the debris actually MOVED from its origin (a static check would be vacuous)"
    );
}

/// Step `topo` `ticks` times, asserting per-tick TRANSIENT-CONSERVATION (no transient COUNTED-held by
/// more than one shard) at EVERY committed tick — the D-7b structural drop-before-promote proof that
/// the holder set transits `{source}→{}→{dest}`, never `{source,dest}`, even under StaggerPlan skew.
fn step_asserting_conservation(topo: &mut Topology, ticks: u64) {
    for i in 0..ticks {
        topo.step();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick(&reports, TickId(i))
            .expect("no transient is COUNTED-held by two shards at any tick");
    }
}

const SRC_REALM: RealmId = RealmId::System(7);
const DST_REALM: RealmId = RealmId::System(8);

/// THE D-7a headline: a single Debris batch crosses from the source shard to the dest shard via the
/// adopt-before-drop choreography — the dest holds it (and NOWHERE else: no DoubleHeld, no vanish),
/// the batched go-token committed exactly once (G-TIER: one write per batch), TRANSIENT-AUTHORITY-HELD
/// passes, and nothing was lost (the clean hand-off is not a drop). This becomes a permanent gate.
#[test]
fn p3_transient_debris_batch_crosses_adopt_before_drop() {
    let fabric = FaultFabric::new(0xD7A, 4);
    let mut topo = p2_cluster(&fabric, 4);
    // WARMUP: both shards win their realm leases (the source for the crossing, the dest to adopt into).
    for _ in 0..10 {
        topo.step();
    }
    let src_fence = realm_fence(&mut topo, SRC_REALM);
    let dst_fence = realm_fence(&mut topo, DST_REALM);

    // Seed a Debris transient crossing SHARD→DEST and trigger the matching batch saga (the same
    // orchestrator producer a player's saga uses — the Transient class just takes the short path).
    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 1, 0);
    let batch = TransferId(0xD7A_0001);
    seed_transient_crossing(&mut topo, debris, batch, src_fence, dst_fence, DVec3::ZERO);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0), // a transient batch is session-less; the short path never reads it
            subject: DirectoryKey::Realm(DST_REALM), // inert provenance — never enters the directory
            expected_fence: dst_fence,               // the go-token commits at the dest realm fence
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Transient,
            needs_provision: false,
            from_realm: SRC_REALM,
            to_realm: DST_REALM,
        },
    );

    // Step to quiescence through the D-7b structural drop-before-promote (release → DropApplied →
    // promote → DropApplied → ReleaseComplete), asserting per-tick conservation EVERY tick.
    step_asserting_conservation(&mut topo, 24);

    let reports = topo.inspect_all();

    // ONE batched go-token, at the dest realm-lease fence (G-TIER: one orchestrator write per batch).
    let go_tokens: Vec<(BatchId, _)> = reports
        .iter()
        .flat_map(|(_, r)| r.batch_goes.clone())
        .collect();
    assert_eq!(
        go_tokens,
        vec![(BatchId(batch), dst_fence)],
        "exactly one batched go-token committed, at the dest realm fence"
    );

    // The debris is held AUTHORITATIVELY by the DEST and NOWHERE else — no DoubleHeld, no vanish.
    let holders: Vec<NodeId> = reports
        .iter()
        .filter(|(_, r)| r.owned_transients.iter().any(|(e, _)| *e == debris))
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(
        holders,
        vec![DEST],
        "the debris settled at the DEST, held by exactly one shard"
    );

    // TRANSIENT-AUTHORITY-HELD: singly held, anchored to a live lease, backed by the go-token.
    assert_eq!(
        verify_transient_authority_held(&reports),
        Ok(()),
        "TRANSIENT-AUTHORITY-HELD holds after the crossing settles"
    );

    // The clean hand-off is NOT a loss — the declared-loss counter stays 0 on the happy path.
    assert_eq!(
        transient_dropped_total(&mut topo),
        0,
        "a clean adopt-before-drop hand-off loses nothing"
    );

    // The transient never created a directory row (burst isolation — HR2): the directory holds only
    // the two realm records + (from warmup) no entity, never a Debris key.
    let debris_rows = reports
        .iter()
        .flat_map(|(_, r)| r.directory.iter())
        .filter(|(k, _)| matches!(k, DirectoryKey::Entity(e) if *e == debris))
        .count();
    assert_eq!(
        debris_rows, 0,
        "a transient is NEVER a directory OwnerRecord"
    );
}

/// THE D-7b headline (the correctness blocker the structural drop-before-promote fixes): a Debris
/// batch crosses while the DEST shard LAGS the source by a StaggerPlan offset — and per-tick
/// TRANSIENT-CONSERVATION holds at EVERY tick (the holder set transits `{SHARD}→{}→{DEST}`, never
/// `{SHARD,DEST}`). On the OLD D-7a broadcast model the dest could promote while the source still
/// held, double-holding for the skew window; the structural release-before-promote makes that
/// unrepresentable. Becomes a permanent gate.
#[test]
fn p3_transient_crosses_without_double_holding_under_stagger() {
    let fabric = FaultFabric::new(0xD7B, 4);
    // DEST lags SHARD by 2 ticks — the source releases (uncounts) well before the lagging dest
    // would otherwise promote, so the per-tick gate exercises the skew window the fix closes.
    let mut topo = p2_cluster_staggered(&fabric, 4, StaggerPlan::lockstep().with_offset(DEST, 2));
    // WARMUP: both shards win their realm leases (the lagging dest needs more topology ticks).
    for _ in 0..18 {
        topo.step();
    }
    let src_fence = realm_fence(&mut topo, SRC_REALM);
    let dst_fence = realm_fence(&mut topo, DST_REALM);

    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 2, 0);
    let batch = TransferId(0xD7B_0001);
    seed_transient_crossing(&mut topo, debris, batch, src_fence, dst_fence, DVec3::ZERO);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(DST_REALM),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Transient,
            needs_provision: false,
            from_realm: SRC_REALM,
            to_realm: DST_REALM,
        },
    );

    // The whole handoff under skew, asserting NO double-held tick — the structural-fix proof.
    step_asserting_conservation(&mut topo, 40);

    let reports = topo.inspect_all();
    let holders: Vec<NodeId> = reports
        .iter()
        .filter(|(_, r)| r.owned_transients.iter().any(|(e, _)| *e == debris))
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(
        holders,
        vec![DEST],
        "the debris settled at the DEST under stagger — held by exactly one shard"
    );
    assert_eq!(
        verify_transient_authority_held(&reports),
        Ok(()),
        "TRANSIENT-AUTHORITY-HELD holds after the staggered crossing settles"
    );
    assert_eq!(
        transient_dropped_total(&mut topo),
        0,
        "a clean staggered hand-off loses nothing"
    );
}

/// D-7b.2: a MOVING debris crosses and re-advances CONTINUOUSLY — the dest's pose lies exactly on the
/// closed-form ballistic trajectory from the seeded origin (no teleport across the cut, no
/// double-advance: the source's incremental per-tick advance + the dest's re-advance from the adopted
/// pose compose to the one trajectory). Permanent gate.
#[test]
fn p3_moving_debris_re_advances_continuously_across_the_cut() {
    let fabric = FaultFabric::new(0xD7B2, 4);
    let mut topo = p2_cluster(&fabric, 4);
    for _ in 0..10 {
        topo.step();
    }
    let src_fence = realm_fence(&mut topo, SRC_REALM);
    let dst_fence = realm_fence(&mut topo, DST_REALM);

    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 3, 0);
    let batch = TransferId(0xD7B2_0001);
    let vel = DVec3::new(20.0, 0.0, -8.0); // a moving debris (m/s)
    seed_transient_crossing(&mut topo, debris, batch, src_fence, dst_fence, vel);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(DST_REALM),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Transient,
            needs_provision: false,
            from_realm: SRC_REALM,
            to_realm: DST_REALM,
        },
    );
    step_asserting_conservation(&mut topo, 24);

    // Settled at the DEST, and ON its ballistic trajectory (continuity / no-teleport / no-double-advance).
    let reports = topo.inspect_all();
    let holders: Vec<NodeId> = reports
        .iter()
        .filter(|(_, r)| r.owned_transients.iter().any(|(e, _)| *e == debris))
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(holders, vec![DEST], "the moving debris settled at the DEST");
    assert_debris_on_ballistic_trajectory(&mut topo, debris, vel);
}

/// D-7b.2: a FAST (projectile-speed) debris crosses UNDER STAGGER and STILL re-advances continuously —
/// no double-held tick AND its pose stays on the trajectory even when the dest lags. The
/// high-velocity + skew combination is the worst case for the closed-form re-advance continuity.
#[test]
fn p3_fast_debris_re_advances_continuously_under_stagger() {
    let fabric = FaultFabric::new(0xD7B2F, 4);
    let mut topo = p2_cluster_staggered(&fabric, 4, StaggerPlan::lockstep().with_offset(DEST, 2));
    for _ in 0..18 {
        topo.step();
    }
    let src_fence = realm_fence(&mut topo, SRC_REALM);
    let dst_fence = realm_fence(&mut topo, DST_REALM);

    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 4, 0);
    let batch = TransferId(0xD7B2_0002);
    let vel = DVec3::new(600.0, 120.0, 0.0); // projectile-speed
    seed_transient_crossing(&mut topo, debris, batch, src_fence, dst_fence, vel);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(DST_REALM),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Transient,
            needs_provision: false,
            from_realm: SRC_REALM,
            to_realm: DST_REALM,
        },
    );
    // While the lagging dest's universe-tick trails the source's emit-tick, readvance's
    // `now.saturating_sub(emit_tick)` is 0 — the adopted pose is held in place (no backward teleport)
    // and catches up in ONE closed-form step on the first tick where the dest's clock reaches it.
    step_asserting_conservation(&mut topo, 40);

    let reports = topo.inspect_all();
    let holders: Vec<NodeId> = reports
        .iter()
        .filter(|(_, r)| r.owned_transients.iter().any(|(e, _)| *e == debris))
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(
        holders,
        vec![DEST],
        "the fast debris settled at the DEST under stagger"
    );
    assert_debris_on_ballistic_trajectory(&mut topo, debris, vel);
}

/// THE D-7c G-TIER headline: a 1000-item Debris burst commits via EXACTLY ONE go-token WRITE (the
/// orchestrator write rate scales with BATCH count, NOT item count) and writes ZERO directory rows —
/// the burst-isolation that keeps the orchestrator directory off the hot path at volume in a
/// multi-shard cloud mesh. The write COUNT (not batch_goes.len(), which is false-green under the
/// idempotent or_insert) is the observable that turns a per-item-write regression RED. Permanent gate.
#[test]
fn p3_gtier_burst_write_rate_is_batch_count_not_item_count() {
    const BURST_SIZE: u32 = 1000;
    let fabric = FaultFabric::new(0xD7C, 4);
    let mut topo = p2_cluster(&fabric, 4);
    for _ in 0..10 {
        topo.step();
    }
    let src_fence = realm_fence(&mut topo, SRC_REALM);
    let dst_fence = realm_fence(&mut topo, DST_REALM);
    let batch = TransferId(0xD7C_0001);
    let burst = seed_transient_burst(&mut topo, batch, BURST_SIZE, 0, src_fence, dst_fence);
    trigger_transfer(&mut topo, transient_batch_ctx(batch, dst_fence));
    step_asserting_conservation(&mut topo, 24);

    let reports = topo.inspect_all();
    // (A) THE PROBE — the orchestrator wrote ONE go-token for the whole 1000-item batch (write RATE ==
    // batch COUNT). A per-item-write regression would inflate this to 1000 while batch_goes.len() (B)
    // stays 1 — which is precisely why the write COUNT, not the map size, is the real G-TIER observable.
    let writes: u64 = reports.iter().map(|(_, r)| r.batch_go_writes).sum();
    assert_eq!(
        writes, 1,
        "ONE go-token write for 1000 items (write rate == batch count)"
    );
    // (B) SIZE — one ledger entry at the dest realm fence.
    let go_tokens: Vec<(BatchId, _)> = reports
        .iter()
        .flat_map(|(_, r)| r.batch_goes.clone())
        .collect();
    assert_eq!(go_tokens, vec![(BatchId(batch), dst_fence)]);
    // (C) ZERO directory rows for ANY of the 1000 transients (burst isolation — HR2).
    let dir_rows = reports
        .iter()
        .flat_map(|(_, r)| r.directory.iter())
        .filter(|(k, _)| matches!(k, DirectoryKey::Entity(e) if burst.contains(e)))
        .count();
    assert_eq!(
        dir_rows, 0,
        "a transient is NEVER a directory OwnerRecord, even at 1000-item volume"
    );
    // (D) POSITIVE guards (close the drop-axis vacuity) — all 1000 settled SINGLY at the DEST, no loss.
    assert_eq!(
        held_at_dest(&reports, &burst),
        BURST_SIZE as usize,
        "all 1000 burst items settled at the DEST"
    );
    assert_eq!(
        transient_dropped_total(&mut topo),
        0,
        "no burst item was lost"
    );
}

/// THE D-7c G-TIER discriminator: K=3 DISTINCT batches commit K go-token WRITES (one EACH) — proving
/// the write rate tracks the BATCH count, not a fixed 1, and not the 15 items. The per-batch envelope
/// (ack/round-trip) cost is likewise O(batches) == 3, not O(items). (K>1 is K distinct seeded batches +
/// K triggers — NOT a cap-split of one batch, which is architecturally broken at HEAD; owed D-7d.)
#[test]
fn p3_gtier_write_rate_scales_with_batch_count() {
    let fabric = FaultFabric::new(0xD7C2, 4);
    let mut topo = p2_cluster(&fabric, 4);
    for _ in 0..10 {
        topo.step();
    }
    let src_fence = realm_fence(&mut topo, SRC_REALM);
    let dst_fence = realm_fence(&mut topo, DST_REALM);
    let mut all = BTreeSet::new();
    for (k, batch) in [TransferId(0xA1), TransferId(0xA2), TransferId(0xA3)]
        .into_iter()
        .enumerate()
    {
        let set = seed_transient_burst(&mut topo, batch, 5, (k as u64) * 100, src_fence, dst_fence);
        all.extend(&set);
        trigger_transfer(&mut topo, transient_batch_ctx(batch, dst_fence));
    }
    step_asserting_conservation(&mut topo, 24);

    let reports = topo.inspect_all();
    let writes: u64 = reports.iter().map(|(_, r)| r.batch_go_writes).sum();
    assert_eq!(
        writes, 3,
        "write rate == BATCH count (3), not item count (15), not 1"
    );
    let go_tokens: Vec<(BatchId, _)> = reports
        .iter()
        .flat_map(|(_, r)| r.batch_goes.clone())
        .collect();
    assert_eq!(go_tokens.len(), 3, "3 distinct go-tokens");
    assert_eq!(
        held_at_dest(&reports, &all),
        15,
        "all 15 items (3 batches × 5) settled at the DEST"
    );
    assert_eq!(
        source_transients_emitted(&mut topo),
        3,
        "3 envelopes — the ack/round-trip cost is O(batches), not O(items)"
    );
}

// ====================================================================================================
// D-7c 7c.3 — DURABLE-UNAFFECTED-BY-BURST. A concurrent 1000-item transient burst must leave EVERY
// concurrent Durable saga BYTE-IDENTICAL (the differential gate). The durable warmup is INLINED via the
// PUBLIC building blocks (p1_client/walk_forward/read_subject/trigger_transfer) — the cut choreography's
// private `run_cut_transfer` lives in p2_transfer_gates and is not cross-crate visible.
// ====================================================================================================

/// Find a node's report in an inspect slice.
fn report(reports: &[(NodeId, InspectReport)], node: NodeId) -> &InspectReport {
    reports
        .iter()
        .find(|(n, _)| *n == node)
        .map(|(_, r)| r)
        .expect("node present in the inspect slice")
}

/// Sum `batch_go_writes` across every node (the G-TIER write-rate observable; 0 ⇒ no burst ran).
fn total_go_writes(reports: &[(NodeId, InspectReport)]) -> u64 {
    reports.iter().map(|(_, r)| r.batch_go_writes).sum()
}

/// Log a player in over `fabric` and walk until SHARD grants the avatar AND DEST wins its realm lease —
/// the deterministic prelude shared by the transfer and no-transfer (sensitivity) durable paths. Returns
/// the topo + the durable subject + its session + the avatar's live directory fence (for the trigger).
/// Bounded + deterministic (same seed ⇒ same grant tick ⇒ the two variants stay tick-aligned).
fn warmup_durable(fabric: &FaultFabric) -> (Topology, EntityId, SessionId, Fence) {
    let mut topo = p2_cluster(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));
    let mut warmed = false;
    for _ in 0..80 {
        topo.step();
        let r = topo.inspect_all();
        if !report(&r, SHARD).held_entities.is_empty()
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == DST_REALM)
        {
            warmed = true;
            break;
        }
    }
    assert!(
        warmed,
        "durable warmup granted the avatar + the DEST realm lease"
    );
    let (session, entity, fence) = read_subject(&mut topo);
    (topo, entity, session, fence)
}

/// Drive a DURABLE player transfer SHARD→DEST to full settle over the ZERO-fault `fabric`. When
/// `with_burst`, a concurrent 1000-item Debris burst + its Transient saga ride alongside the durable saga
/// on the ONE `SagaRuntimeRes`. Both variants run the IDENTICAL deterministic warmup then step a FIXED
/// window — so the two are sampled at the IDENTICAL absolute tick and every tick-dependent field (a
/// renewed lease, a re-stamped pose tick) is burst-independent BY CONSTRUCTION, not by luck. Returns the
/// settled topo + the durable subject + session + the burst entity set (empty when `!with_burst`).
fn run_durable_to_settle(
    fabric: &FaultFabric,
    with_burst: bool,
) -> (Topology, EntityId, SessionId, BTreeSet<EntityId>) {
    let (mut topo, entity, session, fence) = warmup_durable(fabric);

    // CONCURRENT BURST (optional) — seed 1000 Debris + trigger its Transient saga BEFORE the durable
    // trigger (NO `step` between → both variants trigger the durable saga at the SAME tick), so the burst
    // is genuinely in flight WHILE the durable saga commits + demotes.
    let mut burst = BTreeSet::new();
    if with_burst {
        let src_fence = realm_fence(&mut topo, SRC_REALM);
        let dst_fence = realm_fence(&mut topo, DST_REALM);
        let batch = TransferId(0xB57);
        burst = seed_transient_burst(&mut topo, batch, 1000, 0, src_fence, dst_fence);
        trigger_transfer(&mut topo, transient_batch_ctx(batch, dst_fence));
    }

    // TRIGGER the durable avatar transfer — the SAME saga machinery; Durable takes the full path.
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: TransferId(1),
            session,
            subject: DirectoryKey::Entity(entity),
            expected_fence: fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Durable,
            needs_provision: false,
            from_realm: SRC_REALM,
            to_realm: DST_REALM,
        },
    );

    // FIXED window (tick-aligned across variants), asserting per-tick transient conservation throughout.
    for i in 0..120 {
        topo.step();
        let r = topo.inspect_all();
        verify_transient_conservation_tick(&r, TickId(i))
            .expect("transient conservation holds every tick (durable + burst)");
    }

    // CONFIRM full settle — fail loud if the fixed window was too short (never silently under-drive).
    let r = topo.inspect_all();
    assert!(
        !report(&r, SHARD)
            .held_entities
            .iter()
            .any(|(e, _)| *e == entity),
        "the source demoted the durable subject within the window",
    );
    assert!(
        report(&r, DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == entity),
        "the dest holds the durable subject within the window",
    );
    assert_eq!(
        live_sagas(&mut topo),
        0,
        "every saga (durable + burst) reached Done"
    );
    if with_burst {
        assert_eq!(
            held_at_dest(&r, &burst),
            burst.len(),
            "all 1000 burst items settled at the DEST within the window",
        );
    }
    (topo, entity, session, burst)
}

/// THE D-7c DURABLE-UNAFFECTED-BY-BURST headline: a concurrent 1000-item transient burst leaves the
/// durable player saga's footprint BYTE-IDENTICAL. Transients take NO directory lock and write ZERO
/// directory rows, so the durable saga and the burst share NO mutable orchestrator state except
/// `batch_goes` (excluded from the projection) and the wire fabric — so the durable SUBSET is invariant
/// while `trace_bytes` legitimately moves. Any subset diff is a burst leak into the durable path = an
/// HR1/HR2 violation. PERMANENT gate. (Byte-identity is valid ONLY under the zero `LinkPolicy` — see
/// `durable_subset`'s doc precondition.)
#[test]
fn p3_durable_transfer_byte_identical_under_concurrent_burst() {
    // SAME seed both runs ⇒ deterministic + tick-aligned; the ONLY difference is the concurrent burst.
    let (mut base, base_entity, base_session, _) =
        run_durable_to_settle(&FaultFabric::new(909, 2), false);
    let (mut burst, burst_entity, burst_session, burst_set) =
        run_durable_to_settle(&FaultFabric::new(909, 2), true);

    // The durable subject is minted identically (same seed + same warmup) — the projections are comparable.
    assert_eq!(
        base_entity, burst_entity,
        "same durable subject id in both runs"
    );
    assert_eq!(base_session, burst_session, "same session id in both runs");

    // THE GATE — the durable footprint is byte-identical with vs without the burst.
    let base_reports = base.inspect_all();
    let burst_reports = burst.inspect_all();
    assert_eq!(
        durable_subset(&base_reports, base_entity, base_session),
        durable_subset(&burst_reports, burst_entity, burst_session),
        "the durable saga footprint is UNAFFECTED by a concurrent 1000-item transient burst",
    );

    // SPECIFICITY (negative control) — the burst GENUINELY ran in the burst variant and was ABSENT in the
    // base variant, so the byte-identity above is not a false-green from a burst that never happened.
    assert_eq!(
        held_at_dest(&burst_reports, &burst_set),
        1000,
        "the burst variant actually moved all 1000 transients to the DEST",
    );
    assert_eq!(
        total_go_writes(&burst_reports),
        1,
        "the burst variant committed exactly one go-token",
    );
    assert_eq!(
        total_go_writes(&base_reports),
        0,
        "the base variant ran NO burst (zero go-token writes) — the control is real",
    );
}

/// The two-sided counterpart to the gate (SENSITIVITY): a REAL durable-path change MOVES the durable
/// subset — proving `durable_subset` is not an over-filtered inert constant that would pass the gate
/// above vacuously. BASE transfers the avatar to the DEST; PERTURBED runs the identical warmup but never
/// triggers the transfer, so the subject stays authoritative at the SHARD — the projections must differ.
#[test]
fn p3_durable_subset_moves_when_durable_saga_perturbed() {
    let (mut base, base_entity, base_session, _) =
        run_durable_to_settle(&FaultFabric::new(909, 2), false);
    // PERTURBED: the SAME warmup, but NO durable transfer — the subject stays at the SHARD.
    let (mut perturbed, p_entity, p_session, _) = warmup_durable(&FaultFabric::new(909, 2));
    for _ in 0..16 {
        perturbed.step();
    }
    assert_eq!(
        base_entity, p_entity,
        "same subject id (same seed + warmup)"
    );
    assert_eq!(base_session, p_session, "same session id");
    assert_ne!(
        durable_subset(&base.inspect_all(), base_entity, base_session),
        durable_subset(&perturbed.inspect_all(), p_entity, p_session),
        "a real durable-path change (transfer vs no-transfer) MOVES the durable subset",
    );
}

// ====================================================================================================
// D-7d Slice 1 — the kill-9 crash cells for the TRANSIENT class (the P3 headline for debris). Every
// stranded transient lands DETERMINISTICALLY in exactly one of {held-at-the-survivor,
// bucketed-as-accounted-loss}, proven under a permanent kill of the source or dest mid-handoff — plus a
// crash-resurrect control proving the dead-resolution is gated on the KILL-only notice, not a bare timeout.
// ====================================================================================================

/// THE D-7d SOURCE-kill headline: the source is permanently killed mid-handoff AFTER the dest adopted
/// (at `AwaitRelease`) — today's stranded-until-self-fence gap. The go-token authorizes the dest
/// self-promote, so the debris settles at the DEST with ZERO loss; the dead source's `Held` corpse is
/// excluded by the dead-aware oracle (never a phantom second holder). Permanent gate.
#[test]
fn p3_transient_source_kill_self_promotes_the_dest_zero_loss() {
    let (mut topo, debris, dead) = run_transient_fault_scenario(
        0xD7D_5041,
        Scenario {
            class: DurabilityClass::Transient,
            victim: SHARD,
            at_phase: AT_AWAIT_RELEASE,
            crash_when: CrashWhen::PostStep, // ignored for a permanent Kill
            fault: Fault::Kill,
            standing_rehome: false, // transients never standing-re-home (they live + die in one realm)
        },
    );
    assert_transient_end_state(
        &mut topo,
        debris,
        &dead,
        EndState::BatchCommittedAt { node: DEST },
    );
    // MECHANISM (anti-vacuity): the dead-SOURCE self-promote resolution actually fired exactly once —
    // this is the kill path, NOT the happy path that completes without a resolution.
    assert_eq!(
        source_unreachable_resolutions(&mut topo),
        1,
        "the self-promote resolution fired exactly once",
    );
    assert_eq!(
        dest_unreachable_resolutions(&mut topo),
        0,
        "no dest resolution"
    );
}

/// THE D-7d DEST-kill headline: the dest is permanently killed mid-handoff AFTER the source released
/// (at `AwaitPromote`) — the "vanished from the counted set" gap. The only promote target is gone, so
/// the source ABANDONS its retained `Departing` copy as a DETERMINISTIC accounted loss (within the
/// Debris budget, NON-ZERO — the retained copy is what makes the loss COUNTABLE, not silent). Permanent gate.
#[test]
fn p3_transient_dest_kill_abandons_within_budget() {
    let (mut topo, debris, dead) = run_transient_fault_scenario(
        0xD7D_4553,
        Scenario {
            class: DurabilityClass::Transient,
            victim: DEST,
            at_phase: AT_AWAIT_PROMOTE,
            crash_when: CrashWhen::PostStep, // ignored for a permanent Kill
            fault: Fault::Kill,
            standing_rehome: false, // transients never standing-re-home (they live + die in one realm)
        },
    );
    assert_transient_end_state(
        &mut topo,
        debris,
        &dead,
        EndState::BatchDroppedWithinBudget {
            kind: EntityKind::Debris,
        },
    );
    assert_eq!(
        dest_unreachable_resolutions(&mut topo),
        1,
        "the abandon resolution fired exactly once",
    );
    assert_eq!(
        source_unreachable_resolutions(&mut topo),
        0,
        "no source resolution",
    );
}

/// THE D-7d control: the source CRASHES mid-handoff (at `AwaitRelease`) then RESURRECTS — the fabric
/// redelivers the in-flight `TransientRelease` and the choreography completes NORMALLY to the DEST. The
/// source NEVER enters `dead_participants` (a crash emits NO kill-only `NodeUnreachable`), so NO
/// resolution fires — proving the dead-resolution is gated on the kill-only notice, NOT a bare timeout
/// (the held-set self-heals before any resolution could). The `after` outlasts the redrive deadline, so
/// the BatchHandoff Timeout re-drive (the "due but NOT dead" branch) is exercised during the crash.
#[test]
fn p3_transient_source_crash_resurrect_completes_without_resolving() {
    let (mut topo, debris, dead) = run_transient_fault_scenario(
        0xD7D_C0DE,
        Scenario {
            class: DurabilityClass::Transient,
            victim: SHARD,
            at_phase: AT_AWAIT_RELEASE,
            crash_when: CrashWhen::PostStep,
            fault: Fault::CrashResurrect { after: 12 }, // > redrive deadline (8) → Timeout re-drive fires
            standing_rehome: false, // transients never standing-re-home (they live + die in one realm)
        },
    );
    assert!(dead.is_empty(), "the resurrected source is not a dead node");
    // The SAME outcome as the kill cell (debris at the DEST, zero loss) — but reached by NORMAL
    // completion, NOT a resolution.
    assert_transient_end_state(
        &mut topo,
        debris,
        &dead,
        EndState::BatchCommittedAt { node: DEST },
    );
    // THE GATE: a crash is NOT a kill — neither dead-resolution fired (no spurious abandon/self-promote).
    assert_eq!(
        source_unreachable_resolutions(&mut topo),
        0,
        "a crash-then-resurrect must NOT trigger a dead-source resolution (kill-only gate)",
    );
    assert_eq!(
        dest_unreachable_resolutions(&mut topo),
        0,
        "no dest resolution either",
    );
}

/// D-3 CSCALE-1 (the dead-vs-slow cure, the headline regression cell): a recoverable BLIP toward a
/// HEALTHY dest during a transient `BatchHandoff` must NOT abandon the batch. The orchestrator runs the
/// prod confirmation margin (`n_consecutive_unreachable = 3`); the flap's `NodeUnreachable{DEST}` notices
/// never reach the threshold, the dest is never confirmed dead, the flap heals, and the batch lands.
/// RED before Slice 3 (one `NodeUnreachable` → `dead_participants` → cheap-redrive abandon → irreversible
/// loss of a HEALTHY batch); GREEN after (evidence-gated tracker + clear-on-ack + the abort-budget gate).
#[test]
fn p3_transient_a_dest_blip_does_not_abandon_a_healthy_batch() {
    let (mut topo, debris) = run_transient_dest_flap(0xC5A1_E001);
    // The batch landed at DEST, held singly, zero loss — no node was killed, so the dead set is empty.
    assert_transient_end_state(
        &mut topo,
        debris,
        &BTreeSet::new(),
        EndState::BatchCommittedAt { node: DEST },
    );
    assert_eq!(
        dest_unreachable_resolutions(&mut topo),
        0,
        "a recoverable blip never abandons a HEALTHY dest (the CSCALE-1 cure)",
    );
    // ANTI-VACUITY: the blip was genuinely observed (the orch→dest promote bounced ≥1) — the exact path
    // the old insert-only / cheap-redrive code would have abandoned the healthy batch on.
    assert!(
        liveness_notices(&mut topo) > 0,
        "the flap produced at least one NodeUnreachable — the cell is non-vacuous",
    );
}
