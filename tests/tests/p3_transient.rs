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
use vd_core::pose::RealmId;
use vd_core::{BatchId, EntityId, NodeId, SessionId, TransferId};
use vd_harness::fabric::FaultFabric;
use vd_harness::oracle::verify_transient_authority_held;
use vd_sim::saga::SagaCtx;
use vd_tests::{
    DEST, SHARD, p2_cluster, realm_fence, seed_transient_crossing, transient_dropped_total,
    trigger_transfer,
};
use vd_wire::seams::directory::DirectoryKey;

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
    seed_transient_crossing(&mut topo, debris, batch, src_fence, dst_fence);
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

    // Step to quiescence: SHARD emits the batch → DEST adopts as Arriving + acks BatchAdopted → the
    // orchestrator emits TransientDrop → SHARD drops + DEST promotes Arriving→Held.
    for _ in 0..20 {
        topo.step();
    }

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
