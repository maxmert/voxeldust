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

use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_core::{BatchId, EntityId, NodeId, SessionId, TickId, TransferId};
use vd_harness::fabric::FaultFabric;
use vd_harness::oracle::{verify_transient_authority_held, verify_transient_conservation_tick};
use vd_harness::topology::{StaggerPlan, Topology};
use vd_sim::saga::SagaCtx;
use vd_tests::{
    DEST, SHARD, TRANSIENT_SEED_POS0, TRANSIENT_SEED_TICK0, dest_stub_config, p2_cluster,
    p2_cluster_staggered, realm_fence, seed_transient_crossing, transient_dropped_total,
    trigger_transfer,
};
use vd_wire::seams::directory::DirectoryKey;

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
        (dest_pose.pos - expected).length() < 1.0e-6,
        "debris off its ballistic trajectory: got {:?}, expected {expected:?} (dt_s={dt_s})",
        dest_pose.pos
    );
    assert!(
        (dest_pose.pos - TRANSIENT_SEED_POS0).length() > 1.0,
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
