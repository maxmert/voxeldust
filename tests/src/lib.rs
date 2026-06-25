//! # vd-tests — accumulated scenario suites
//!
//! Shared scenario builders/fixtures used by the integration tests in `tests/tests/`.
//! Standing rule: every phase ADDS scenarios; nothing is deleted. The accumulated
//! suite re-running green is the release gate for every later phase.

use std::collections::BTreeSet;
use vd_connection_plane::gateway::{
    GatewayConfig, GatewayStats, TransportTuning, register_gateway,
};
use vd_connection_plane::tickets;

use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::{AccountId, EntityId, EpochId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_harness::client::{DeliveredWorldView, InputCmd, ScriptedClient};
use vd_harness::fabric::{CrashWhen, FabricTransport, FaultFabric};
use vd_harness::oracle::{
    AuthorityViolation, verify_authority_settled, verify_authority_unique_excluding,
    verify_transient_authority_held_excluding, verify_transient_conservation_tick_excluding,
    verify_transient_loss_budget,
};
use vd_harness::topology::{InspectReport, StaggerPlan, Topology};
use vd_node::ShardNode;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_node::orchestrator::{DirectoryRes, OrchestratorConfig, register_orchestrator_with_store};
use vd_node::saga_runtime::{ActiveTransfer, SagaRuntimeRes};
use vd_sim::capability::NodeKind;
use vd_sim::directory::DirectoryTuning;
use vd_sim::io::mem::MemStore;
use vd_sim::saga::SagaCtx;
use vd_sim::stub::{StubConfig, register_stub_shard};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, OwnerRecord};

/// The canonical P1 node ids (one orchestrator, one gateway, one stub shard;
/// clients from 100 upward).
pub const ORCH: NodeId = NodeId(1);
pub const GATEWAY: NodeId = NodeId(2);
pub const SHARD: NodeId = NodeId(3);
/// The transfer DESTINATION shard (P2): a second stub in a distinct realm a player crosses INTO.
pub const DEST: NodeId = NodeId(4);

/// The auth service's signing key for test-minted login tickets.
pub const AUTH_SIGNING_KEY: [u8; 32] = [0x42; 32];

#[must_use]
pub fn auth_verifying_key() -> [u8; 32] {
    ed25519_dalek::SigningKey::from_bytes(&AUTH_SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

/// P1 world parameters shared by every scenario.
#[must_use]
pub fn stub_config() -> StubConfig {
    StubConfig {
        realm: RealmId::System(7),
        frame: FrameRef::SystemSpace { system_seed: 7 },
        move_speed_mps: 2.0,
        tick_dt_s: 0.05,
        orchestrator: ORCH,
        mint_seed: 11,
        // Large window: the oracle must see a whole short test run.
        input_log_capacity: 1_000_000,
        // A small NON-ZERO recheck interval (the EXISTING knob, not a new magic number) so the
        // SOURCE periodically re-reads its REALM-lease head and self-fences the whole shard on a
        // lease revocation. The 1c.8 per-entity granted-key poll is GONE (1d.5b.2): the source
        // ENTITY self-fence is now driven by the saga-pushed ordered `Demote`, not a directory poll.
        realm_recheck_interval: 4,
        // D-3 INERT here: the cluster scenarios do not exercise the lease-renewal heartbeat (the D-3
        // cells set it explicitly). 0 = no heartbeat, matching pre-D-3 behavior.
        lease_renew_interval_ticks: 0,
        snapshot_datagram_budget: 1100,
    }
}

/// The DEST stub's params: a DISTINCT realm (`System(8)`) + frame + mint seed so it is a
/// genuinely separate shard the player transfers INTO (not a clone of the source).
#[must_use]
pub fn dest_stub_config() -> StubConfig {
    StubConfig {
        realm: RealmId::System(8),
        frame: FrameRef::SystemSpace { system_seed: 8 },
        mint_seed: 17,
        ..stub_config()
    }
}

/// The shared cluster spine: an orchestrator + a gateway + N stub shards on a fabric-backed
/// topology, built in a fixed order (orchestrator, gateway, then shards in list order — the
/// `p1_cluster` baseline must stay byte-identical). `clock_peers` are the nodes the
/// orchestrator drives the universe clock to; EVERY follower shard must be listed or its clock
/// never advances. Clients are added by the caller.
/// The cluster orchestrator's config (shared by `build_cluster` + the D-6 orchestrator-kill rebuild, so
/// a rebuilt orchestrator is built IDENTICALLY — same reserve_chunk/tuning → a clean recover).
#[must_use]
pub fn orch_config(clock_peers: Vec<NodeId>) -> OrchestratorConfig {
    OrchestratorConfig {
        epoch: EpochId(1),
        reserve_chunk: 1024,
        clock_peers,
        directory: DirectoryTuning {
            lease_ttl_ticks: 1_000,
            ..DirectoryTuning::default()
        },
        // Slice 2a: the deadline producer runs LIVE in the capstone cluster (dev values 8/24).
        saga: vd_sim::saga::SagaTuning::default(),
    }
}

fn build_cluster(
    fabric: &FaultFabric,
    max_sessions: usize,
    clock_peers: Vec<NodeId>,
    shards: Vec<(NodeId, StubConfig)>,
    stagger: StaggerPlan,
    orch_store: MemStore,
) -> Topology {
    let mut topo = Topology::new(fabric.clone(), stagger);

    let mut orch = build_app(
        NodeConfig {
            node_id: ORCH,
            kind: NodeKind::Orchestrator,
        },
        fabric.register(ORCH),
    );
    let (world, schedule) = orch.parts_mut();
    // D-6: the orchestrator is built against a durable Store. Normal clusters pass a fresh (genesis)
    // `MemStore` (transparent); the D-6 orchestrator-kill driver passes a RETAINED handle so a rebuilt
    // orchestrator re-hydrates the SAME committed WAL.
    register_orchestrator_with_store(
        world,
        schedule,
        &orch_config(clock_peers),
        Box::new(orch_store),
    );
    topo.add_node(Box::new(orch));

    let mut gateway = build_app(
        NodeConfig {
            node_id: GATEWAY,
            kind: NodeKind::Gateway,
        },
        fabric.register(GATEWAY),
    );
    // The STABLE routable-shard roster (FORK 5 / 1d.2): EVERY shard in the cluster, so a
    // (render-ready) dest's frames are node-class-dispatchable. The login shard `SHARD` is
    // always a member.
    let known_shards: std::collections::BTreeSet<NodeId> =
        shards.iter().map(|(id, _)| *id).collect();
    let (world, schedule) = gateway.parts_mut();
    register_clock_follower(world, schedule);
    register_gateway(
        world,
        schedule,
        GatewayConfig {
            orchestrator: ORCH,
            shard: SHARD,
            known_shards,
            auth_verifying_key: auth_verifying_key(),
            session_seed: 23,
            tick_hz: 50,
            // D-3 INERT: the cluster scenarios do not exercise the session heartbeat (the D-3 cells do).
            lease_renew_interval_ticks: 0,
            tuning: TransportTuning {
                max_sessions,
                max_buffered_inputs: TransportTuning::DEFAULT_MAX_BUFFERED_INPUTS,
            },
        },
    );
    topo.add_node(Box::new(gateway));

    for (id, cfg) in shards {
        let mut shard = build_app(
            NodeConfig {
                node_id: id,
                kind: NodeKind::StubShard,
            },
            fabric.register(id),
        );
        let (world, schedule) = shard.parts_mut();
        register_clock_follower(world, schedule);
        register_stub_shard(world, schedule, cfg);
        topo.add_node(Box::new(shard));
    }

    topo
}

/// Build the canonical P1 cluster (orchestrator + gateway + ONE stub shard) onto a
/// fabric-backed topology. Clients are added by the caller.
#[must_use]
pub fn p1_cluster(fabric: &FaultFabric, max_sessions: usize) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD],
        vec![(SHARD, stub_config())],
        StaggerPlan::lockstep(),
        MemStore::new(), // a fresh (genesis) store — this cluster is not rebuilt
    )
}

/// Build the P2 transfer cluster: P1 plus a SECOND stub shard at [`DEST`] in a distinct realm,
/// with `DEST` added to the orchestrator's clock peers so it follows the clock and wins its own
/// realm lease — the source/dest pair a cross-shard transfer hands a player between.
#[must_use]
pub fn p2_cluster(fabric: &FaultFabric, max_sessions: usize) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, DEST],
        vec![(SHARD, stub_config()), (DEST, dest_stub_config())],
        StaggerPlan::lockstep(),
        MemStore::new(), // a fresh (genesis) store — this cluster is not rebuilt
    )
}

/// The P2 transfer cluster whose orchestrator is built against a caller-RETAINED `MemStore` (D-6): the
/// returned handle lets the orchestrator-kill driver rebuild the orchestrator against the SAME committed
/// WAL (the kill-9 analog). Identical to [`p2_cluster`] otherwise.
#[must_use]
pub fn p2_cluster_durable_orch(fabric: &FaultFabric, max_sessions: usize) -> (Topology, MemStore) {
    let store = MemStore::new();
    let topo = build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, DEST],
        vec![(SHARD, stub_config()), (DEST, dest_stub_config())],
        StaggerPlan::lockstep(),
        store.clone(),
    );
    (topo, store)
}

/// The P2 transfer cluster under a deliberate `StaggerPlan` (D-7b): the source/dest process the
/// handoff at SKEWED local ticks, so a per-tick conservation gate can prove the structural
/// drop-before-promote never double-holds even when one shard lags the other.
#[must_use]
pub fn p2_cluster_staggered(
    fabric: &FaultFabric,
    max_sessions: usize,
    stagger: StaggerPlan,
) -> Topology {
    build_cluster(
        fabric,
        max_sessions,
        vec![GATEWAY, SHARD, DEST],
        vec![(SHARD, stub_config()), (DEST, dest_stub_config())],
        stagger,
        MemStore::new(), // a fresh (genesis) store — this cluster is not rebuilt
    )
}

/// Borrow a topology node downcast to its concrete `ShardNode<FabricTransport>` so scenario code
/// can drive/read it directly (orchestrator = saga producer + directory; gateway = route/stats).
fn with_node<R>(
    topo: &mut Topology,
    id: NodeId,
    f: impl FnOnce(&mut ShardNode<FabricTransport>) -> R,
) -> R {
    let node = topo.node_mut(id).expect("node present");
    let shard = node
        .as_any_mut()
        .expect("ShardNode opts into downcasting")
        .downcast_mut::<ShardNode<FabricTransport>>()
        .expect("the node is a ShardNode");
    f(shard)
}

/// Borrow the orchestrator's concrete node — the saga producer + the directory live there.
fn with_orchestrator<R>(
    topo: &mut Topology,
    f: impl FnOnce(&mut ShardNode<FabricTransport>) -> R,
) -> R {
    with_node(topo, ORCH, f)
}

/// Read the player's transfer SUBJECT live from the orchestrator directory: the `Session`, the
/// single `Entity` record (the avatar), and the fence it is recorded at — the CAS expectation
/// is the DIRECTORY record's fence, never a hardcoded literal or the shard's own copy.
#[must_use]
pub fn read_subject(topo: &mut Topology) -> (SessionId, EntityId, Fence) {
    with_orchestrator(topo, |orch| {
        let dir = orch.world_mut().resource::<DirectoryRes>();
        let mut session = None;
        let mut entity = None;
        for (key, record) in dir.0.entries() {
            match key {
                DirectoryKey::Session(s) => session = Some(*s),
                DirectoryKey::Entity(e) => entity = Some((*e, record.fence)),
                _ => {}
            }
        }
        let session = session.expect("the session is recorded");
        let (entity, fence) = entity.expect("the avatar entity is recorded");
        (session, entity, fence)
    })
}

/// Trigger a REAL transfer through the orchestrator's saga runtime — the same producer 1d
/// shards will use (never hand-fed acks): it emits the real
/// `PrepareSubscribe → RequestCut → FreezeSource → CommitAuthority` over the fabric.
pub fn trigger_transfer(topo: &mut Topology, ctx: SagaCtx) {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource_mut::<SagaRuntimeRes>()
            .start_transfer(ctx, GATEWAY);
    });
}

/// Read a realm's recorded fence from the orchestrator directory (D-7: the transient go-token /
/// adopt anchor fence is read from the directory, never a hardcoded literal).
#[must_use]
pub fn realm_fence(topo: &mut Topology, realm: RealmId) -> Fence {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<DirectoryRes>()
            .0
            .head(DirectoryKey::Realm(realm))
            .expect("realm recorded")
            .fence
    })
}

/// The seeded Debris origin (D-7b) — the SINGLE source of the crossing pose's `pos0`/`tick0`, shared
/// by [`seed_transient_crossing`] AND the e2e ballistic-trajectory assertion, so the expected
/// trajectory can NEVER silently drift from what the seed actually writes (no-drift discipline).
pub const TRANSIENT_SEED_POS0: vd_core::glam::DVec3 = vd_core::glam::DVec3::new(4.0, 5.0, 6.0);
/// See [`TRANSIENT_SEED_POS0`] — the seeded crossing pose's analytic-clock origin `tick0`.
pub const TRANSIENT_SEED_TICK0: vd_core::UniverseTick = vd_core::UniverseTick(1);

/// D-7: seed a Debris transient on the SOURCE shard ([`SHARD`]) as a pending Crossing to the DEST
/// realm — the TEST-driven boundary-heuristic stand-in (the autonomous geometric trigger is P4/P5).
/// The matching batch saga must be triggered separately via [`trigger_transfer`] with a Transient
/// ctx carrying the SAME `batch` id. `anchor` is the source's realm-lease fence; `dst_realm_fence`
/// the dest's (both via [`realm_fence`]).
pub fn seed_transient_crossing(
    topo: &mut Topology,
    entity: EntityId,
    batch: TransferId,
    anchor: Fence,
    dst_realm_fence: Fence,
    vel: vd_core::glam::DVec3,
) {
    with_node(topo, SHARD, |s| {
        let pose = vd_core::pose::StampedPose {
            vel,
            ..vd_core::pose::StampedPose::at_rest(
                stub_config().frame,
                TRANSIENT_SEED_POS0,
                TRANSIENT_SEED_TICK0,
            )
        };
        s.world_mut()
            .resource_mut::<vd_sim::stub::OwnedTransients>()
            .0
            .insert(
                entity,
                vd_sim::stub::Transient {
                    pose,
                    anchor_fence: anchor,
                    status: vd_sim::stub::TransientStatus::Crossing {
                        dest: DEST,
                        to_realm: dest_stub_config().realm,
                        dst_realm_fence,
                        batch,
                    },
                },
            );
    });
}

/// Total transients DROPPED as a LOSS (self-fence, no hand-off) across the source + dest shards
/// (D-7) — 0 on the happy path (a clean adopt-before-drop hand-off is NOT a loss).
#[must_use]
pub fn transient_dropped_total(topo: &mut Topology) -> u64 {
    [SHARD, DEST]
        .into_iter()
        .map(|n| {
            with_node(topo, n, |s| {
                s.world_mut()
                    .resource::<vd_sim::stub::StubStats>()
                    .transients_dropped
            })
        })
        .sum()
}

/// The SOURCE shard's count of `TransientBatch` envelopes EMITTED (D-7c G-TIER ack-cost dimension) —
/// one per batch, regardless of item count. The K-batch gate asserts this == K (round-trip cost is
/// O(batches), not O(items)).
#[must_use]
pub fn source_transients_emitted(topo: &mut Topology) -> u64 {
    with_node(topo, SHARD, |s| {
        s.world_mut()
            .resource::<vd_sim::stub::StubStats>()
            .transients_emitted
    })
}

/// D-7c G-TIER: seed a BURST of `count` distinct Debris transients on the SOURCE shard ([`SHARD`]),
/// ALL carrying the SAME `batch` id (so `emit_transient_batch` aggregates them into ONE envelope +
/// ONE go-token — the burst-isolation claim). Entity seqs are `seq_base..seq_base+count` (give each
/// batch a DISTINCT `seq_base` so K batches do not collide on entity id). Returns the seeded entity
/// set (for the `held_at_dest` positive guard). Reuses [`seed_transient_crossing`] (DRY); rest poses.
pub fn seed_transient_burst(
    topo: &mut Topology,
    batch: TransferId,
    count: u32,
    seq_base: u64,
    anchor: Fence,
    dst_realm_fence: Fence,
) -> BTreeSet<EntityId> {
    let mut seeded = BTreeSet::new();
    for i in 0..count {
        let entity = EntityId::pack(
            EntityKind::Debris,
            SHARD.0 as u32,
            seq_base + u64::from(i),
            0,
        );
        seed_transient_crossing(
            topo,
            entity,
            batch,
            anchor,
            dst_realm_fence,
            vd_core::glam::DVec3::ZERO,
        );
        seeded.insert(entity);
    }
    seeded
}

/// How many of the `entities` the DEST shard holds AUTHORITATIVELY (D-7c G-TIER positive guard — the
/// drop-axis vacuity closer: proves all N burst items actually settled at the DEST, not silently
/// dropped). Reads `owned_transients` (the `is_held()` subset) at DEST.
#[must_use]
pub fn held_at_dest(reports: &[(NodeId, InspectReport)], entities: &BTreeSet<EntityId>) -> usize {
    reports
        .iter()
        .filter(|(n, _)| *n == DEST)
        .flat_map(|(_, r)| r.owned_transients.iter())
        .filter(|(e, _)| entities.contains(e))
        .count()
}

/// The DURABLE-relevant projection of one node's [`InspectReport`], filtered to the durable subject
/// (D-7c DURABLE-UNAFFECTED-BY-BURST). Projects ONLY fields a transient burst must NEVER perturb (the
/// subject's directory row + held authority/pose/pending/departing/ghost + its live saga); EXCLUDES
/// the legitimately-moving transient/wire fields (`owned_transients`, `held_transient_poses`,
/// `batch_goes`, `batch_go_writes`, `transient_loss`, `held_realms`, `trace_bytes`). `PartialEq` (NOT
/// `Eq` — `StampedPose` carries `f64`); a deterministic same-seed run yields bit-identical poses.
#[derive(Clone, Debug, PartialEq)]
pub struct DurableProjection {
    pub directory_subject: Option<OwnerRecord>,
    pub directory_session: Option<OwnerRecord>,
    pub held_subject: Vec<(EntityId, Fence)>,
    pub held_pose_subject: Vec<(EntityId, StampedPose)>,
    pub pending_subject: Vec<EntityId>,
    pub departing_subject: Vec<EntityId>,
    pub ghost_subject: bool,
    pub active_durable: Vec<ActiveTransfer>,
}

/// Project the durable subject's footprint from every node's report (D-7c). The differential gate
/// asserts this is byte-identical with vs without a concurrent transient burst — any diff is a burst
/// leak into the durable path = an HR1/HR2 violation. PRECONDITION: byte-identity holds only under the
/// ZERO `LinkPolicy` (`p2_cluster` installs no faults); a faulted variant would need invariant-equality
/// (same terminal directory record + applied-input multiset), not raw `assert_eq!` on this projection.
#[must_use]
pub fn durable_subset(
    reports: &[(NodeId, InspectReport)],
    subject: EntityId,
    session: SessionId,
) -> Vec<(NodeId, DurableProjection)> {
    reports
        .iter()
        .map(|(node, r)| {
            (
                *node,
                DurableProjection {
                    directory_subject: r
                        .directory
                        .iter()
                        .find_map(|(k, rec)| (*k == DirectoryKey::Entity(subject)).then_some(*rec)),
                    directory_session: r.directory.iter().find_map(|(k, rec)| {
                        (*k == DirectoryKey::Session(session)).then_some(*rec)
                    }),
                    held_subject: r
                        .held_entities
                        .iter()
                        .filter(|(e, _)| *e == subject)
                        .copied()
                        .collect(),
                    held_pose_subject: r
                        .held_poses
                        .iter()
                        .filter(|(e, _)| *e == subject)
                        .copied()
                        .collect(),
                    pending_subject: r
                        .pending_entities
                        .iter()
                        .filter(|e| **e == subject)
                        .copied()
                        .collect(),
                    departing_subject: r
                        .departing_entities
                        .iter()
                        .filter(|e| **e == subject)
                        .copied()
                        .collect(),
                    ghost_subject: r.ghost_dots.contains(&subject),
                    active_durable: r
                        .active_transfers
                        .iter()
                        .filter(|at| at.subject == DirectoryKey::Entity(subject))
                        .copied()
                        .collect(),
                },
            )
        })
        .collect()
}

/// The `Debug` state strings of every live saga on the orchestrator (the admin `views()`
/// surface) — the deterministic phase observable the cut-window choreography polls on.
#[must_use]
pub fn saga_states(topo: &mut Topology) -> Vec<String> {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<SagaRuntimeRes>()
            .views()
            .into_iter()
            .map(|v| v.state)
            .collect()
    })
}

/// Live saga count on the orchestrator.
#[must_use]
pub fn live_sagas(topo: &mut Topology) -> usize {
    with_orchestrator(topo, |orch| {
        orch.world_mut().resource::<SagaRuntimeRes>().live()
    })
}

/// D-7d — the count of dead-SOURCE self-promote resolutions on the orchestrator (the SOURCE-kill cell's
/// anti-vacuity observable: `> 0` proves the resolution actually fired, not that the happy path ran).
#[must_use]
pub fn source_unreachable_resolutions(topo: &mut Topology) -> u64 {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<SagaRuntimeRes>()
            .source_unreachable_resolutions()
    })
}

/// D-7d — the count of dead-DEST abandon resolutions on the orchestrator (the DEST-kill cell's
/// anti-vacuity observable).
#[must_use]
pub fn dest_unreachable_resolutions(topo: &mut Topology) -> u64 {
    with_orchestrator(topo, |orch| {
        orch.world_mut()
            .resource::<SagaRuntimeRes>()
            .dest_unreachable_resolutions()
    })
}

/// D-7d — the total handover-attributable loss for `kind`, summed across every node's `transient_loss`
/// (the budget gate's population). Used by the DEST-kill cell to assert a NON-ZERO, within-budget loss.
#[must_use]
pub fn total_handover_loss(reports: &[(NodeId, InspectReport)], kind: EntityKind) -> u64 {
    reports
        .iter()
        .flat_map(|(_, r)| r.transient_loss.iter())
        .filter(|(k, _)| *k == kind)
        .map(|(_, n)| *n)
        .sum()
}

/// The gateway's count of client inputs BUFFERED for the transfer dest (the `seq > marker` cut
/// buffer fills) — so the conservation gate can assert the buffer/DRAIN path was actually
/// exercised, not merely that the dest applied SOMETHING post-marker (which a direct-forward
/// after the route swap also satisfies). Read via a gateway downcast (vd-harness does not depend
/// on connection-plane, so `GatewayStats` cannot ride `InspectReport`).
#[must_use]
pub fn gateway_buffered_count(topo: &mut Topology) -> u64 {
    with_node(topo, GATEWAY, |gw| {
        gw.world_mut()
            .resource::<GatewayStats>()
            .inputs_buffered_for_dest
    })
}

/// Build one scripted client with a freshly minted (real, Ed25519) login ticket.
#[must_use]
pub fn p1_client(
    fabric: &FaultFabric,
    id: NodeId,
    account: AccountId,
    script: impl FnMut(&DeliveredWorldView) -> Option<InputCmd> + Send + 'static,
) -> ScriptedClient {
    let ticket = tickets::mint_login(&AUTH_SIGNING_KEY, account, EpochId(1), id.0);
    ScriptedClient::new(fabric.register(id), GATEWAY, ticket, script)
}

/// A script that walks straight forward forever.
pub fn walk_forward() -> impl FnMut(&DeliveredWorldView) -> Option<InputCmd> + Send + 'static {
    |_view: &DeliveredWorldView| {
        Some(InputCmd {
            movement: [1.0, 0.0, 0.0],
            look: [0.0, 0.0],
        })
    }
}

// ====================================================================================================
// P3 Slice 1 — the crash/fault recovery matrix harness (the kill-9 thesis headline). ONE data-driven
// driver over a cell table (DRY, HR2-additive: parameterized on `DurabilityClass`); the dead-node-aware
// oracle (vd-harness) makes the deferred kill cells HONEST (a corpse never false-passes AUTHORITY-UNIQUE).
// SCOPE: crash+resurrect (fabric at-least-once recovers) + permanent kill (handled cell aborts to a live
// source; D-37 cells park/orphan honestly). Drop/partition + the seed-driven breadth chaos are P3 Slice 1b.
// ====================================================================================================

/// The crash-matrix client (distinct from the in-process capstone CLIENT 100, but same id is fine —
/// only one client per scenario).
pub const FAULT_CLIENT: NodeId = NodeId(100);

/// The fault a scenario injects at a chosen saga phase.
#[derive(Clone, Copy, Debug)]
pub enum Fault {
    /// Temporary crash (`CrashWhen` phase) then resurrect `after` ticks later — the fabric's
    /// at-least-once redelivers the unacked message on resurrection (the common-case recovery).
    CrashResurrect { after: u64 },
    /// Permanent death of the victim — sends toward it bounce `NodeUnreachable`; recovery depends on
    /// the cell (a pre-freeze dest-kill aborts to the live source; the rest park/orphan honestly, D-37).
    Kill,
}

/// One crash-matrix cell: crash/kill `victim` once the saga reaches `at_phase`, at the `crash_when`
/// sub-tick phase. `class` feeds `SagaCtx.class` (HR2 — Transient/Guided plug in additively, no
/// match-on-class in the driver).
#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    pub class: DurabilityClass,
    pub victim: NodeId,
    pub at_phase: &'static str,
    pub crash_when: CrashWhen,
    pub fault: Fault,
}

/// The asserted end state of a scenario. Data-carrying (not 3 bare arms) so the deferred D-37 cells
/// name WHERE the orphan/park sits — proven HONEST (not false-green) by the dead-node-aware oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndState {
    /// The transfer COMMITTED + settled at `node` (the recovered happy path).
    SettledAt(NodeId),
    /// A clean abort returned the avatar to the live SOURCE, lock cleared, saga terminal. NOTE: NO
    /// permanent-kill cell reaches this (killing the source/gateway makes THEM dead → an orphan; the
    /// abort-to-a-LIVE-source path is a NON-kill abort — a spatial rejection or a transient-fault
    /// pre-freeze timeout that heals). Exercised in P3 Slice 1b (when the Drop/Partition faults land).
    AbortedToSource,
    /// D-37 (RED, owed D-3+D-6): the saga PARKED (re-driving toward a dead node), directory names the
    /// dead `authority_at`; the dead-aware oracle surfaces the orphan rather than false-passing a corpse.
    ParkedHalfOpen { authority_at: NodeId },
    /// D-37 (RED): the saga TOMBSTONED to a dead-owner orphan (directory names the dead `authority_at`,
    /// lock cleared, no live saga) — the dead-aware oracle surfaces it (the orphan is a HeldNowhere: a
    /// dead owner holds nothing, so the fence value is moot here).
    DeadOwnerOrphan { authority_at: NodeId },
    /// D-7d TRANSIENT analogue of [`EndState::SettledAt`]: the batch committed + settled at `node`'s
    /// HELD set (a transient has NO directory row), proven by the dead-aware held-set + a fired
    /// SOURCE-unreachable self-promote resolution; ZERO loss. (Dead source mid-handoff → the go-token
    /// authorizes the dest self-promote.)
    BatchCommittedAt { node: NodeId },
    /// D-7d accounted-loss outcome: the batch was DROPPED (the dead-DEST abandon) — held at NO live
    /// shard, lost WITHIN the `kind`'s `LossBudget` AND non-zero (anti-vacuity), via a fired
    /// DEST-unreachable resolution. (Dead dest mid-handoff → drop-within-budget, the transient answer.)
    BatchDroppedWithinBudget { kind: EntityKind },
}

fn fault_report(reports: &[(NodeId, InspectReport)], id: NodeId) -> &InspectReport {
    &reports
        .iter()
        .find(|(n, _)| *n == id)
        .expect("node present")
        .1
}

/// Step at most `max` ticks until `cond` holds; panic if never (bounded, deterministic).
fn fault_step_until(topo: &mut Topology, max: u64, mut cond: impl FnMut(&mut Topology) -> bool) {
    for _ in 0..max {
        topo.step();
        if cond(topo) {
            return;
        }
    }
    panic!("crash-matrix condition not reached within {max} ticks");
}

/// The orchestrator's directory record for the subject entity (the authority-of-record).
#[must_use]
fn fault_entity_record(reports: &[(NodeId, InspectReport)], entity: EntityId) -> OwnerRecord {
    fault_report(reports, ORCH)
        .directory
        .iter()
        .find_map(|(k, r)| (*k == DirectoryKey::Entity(entity)).then_some(*r))
        .expect("the subject entity is recorded in the directory")
}

/// Drive ONE crash-matrix scenario over the real fabric: warm up + trigger the transfer, drive to
/// `sc.at_phase` (asserting the saga is genuinely there — anti-vacuity), inject the fault, then
/// quiesce (settle, or cap for a parked cell). Returns the topology, the transferred entity, and the
/// dead-node set for the dead-aware oracle.
#[must_use]
pub fn run_fault_scenario(seed: u64, sc: Scenario) -> (Topology, EntityId, BTreeSet<NodeId>) {
    let fabric = FaultFabric::new(seed, 2);
    let mut topo = p2_cluster(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        FAULT_CLIENT,
        AccountId(1000),
        walk_forward(),
    )));
    // WARMUP: the player logs in + walks; the SOURCE grants its avatar and the DEST wins its realm.
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        !fault_report(&r, SHARD).held_entities.is_empty()
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let (session, entity, fence) = read_subject(&mut topo);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: TransferId(1),
            session,
            subject: DirectoryKey::Entity(entity),
            expected_fence: fence,
            source: SHARD,
            dest: DEST,
            class: sc.class,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
        },
    );
    // Drive to the target saga phase (the client auto-stamps the CUT_MARKER on RequestCut; the saga
    // progresses on its acks — no marker pause/resume needed for the crash matrix).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with(sc.at_phase))
    });
    // ANTI-VACUITY: the saga is GENUINELY in the target phase on the fire tick (not advanced past it).
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with(sc.at_phase)),
        "the saga is in phase {} when the fault fires (anti-vacuity)",
        sc.at_phase,
    );
    match sc.fault {
        Fault::CrashResurrect { after } => {
            let fire = topo.tick().0 + 1;
            topo.schedule_crash(sc.victim, TickId(fire), sc.crash_when);
            topo.schedule_resurrect(sc.victim, TickId(fire + after));
        }
        Fault::Kill => fabric.kill(sc.victim),
    }
    // QUIESCE: settle to terminal, or cap (a D-37 parked cell never quiesces — the assert handles it).
    for _ in 0..120 {
        topo.step();
        if live_sagas(&mut topo) == 0 {
            break;
        }
    }
    let dead = topo.dead_nodes();
    (topo, entity, dead)
}

/// Assert the scenario reached its expected [`EndState`], using the dead-node-aware oracle so a killed
/// participant's corpse can neither false-pass uniqueness nor false-RED a legitimate park.
pub fn assert_end_state(
    topo: &mut Topology,
    entity: EntityId,
    dead: &BTreeSet<NodeId>,
    expected: EndState,
) {
    // The orphan that the dead-aware oracle MUST surface for a directory record naming a dead node:
    // the entity half returns HeldNowhere (the recorded owner holds nothing live) before the realm half.
    let orphan = |authority_at: NodeId| AuthorityViolation::HeldNowhere {
        entity,
        recorded: format!("{:?}", AuthorityRef::Shard(authority_at)),
    };
    match expected {
        EndState::SettledAt(node) => {
            // Post-recovery the LIVE view (a resurrect cell has no dead nodes, so inspect_live ==
            // inspect_all here) must be consistent + settled, with the avatar held once at `node`.
            let reports = topo.inspect_live();
            verify_authority_unique_excluding(&reports, dead)
                .expect("AUTHORITY-UNIQUE holds after recovery");
            verify_authority_settled(&reports).expect("settled after recovery");
            assert_eq!(
                live_sagas(topo),
                0,
                "the saga reached terminal (recovered to Done)"
            );
            assert!(
                fault_report(&reports, node)
                    .held_entities
                    .iter()
                    .any(|(e, _)| *e == entity),
                "the entity settled at {node} after recovery",
            );
        }
        EndState::AbortedToSource => {
            let reports = topo.inspect_all();
            let rec = fault_entity_record(&reports, entity);
            assert_eq!(
                rec.authority,
                AuthorityRef::Shard(SHARD),
                "aborted back to the source"
            );
            assert_eq!(rec.in_transfer, None, "the directory lock cleared on abort");
            assert_eq!(live_sagas(topo), 0, "the abort reached terminal");
            assert!(
                fault_report(&reports, SHARD)
                    .held_entities
                    .iter()
                    .any(|(e, _)| *e == entity),
                "the source still owns the avatar (never demoted)",
            );
            // FENCE-9 (the abort-path fence-neutrality fix): the surviving source's HELD fence equals the
            // directory's RECORDED fence — `abort_clear` did NOT bump (an abort is no ownership change), so
            // there is no owner-vs-directory split that would strand the source / wedge a logout LeaseRevoke.
            verify_authority_unique_excluding(&reports, dead)
                .expect("AUTHORITY-UNIQUE holds after the abort (no fence divergence — FENCE-9)");
            // SETTLED (parity with the SettledAt arm): the abort left NO lingering pending/departing/ghost
            // set — the AbortTransfer compensator tore the dest ghost down. Catches a leaked teardown that
            // the uniqueness check (which excludes Ghosts) would miss — load-bearing for a future
            // signal-grant / compound abort that is likelier to leak.
            verify_authority_settled(&reports).expect("settled after the abort (no leaked dest ghost)");
        }
        EndState::ParkedHalfOpen { authority_at } => {
            let reports = topo.inspect_all();
            assert!(
                live_sagas(topo) >= 1,
                "the saga is PARKED (no recovery without D-37)"
            );
            let rec = fault_entity_record(&reports, entity);
            assert_eq!(
                rec.authority,
                AuthorityRef::Shard(authority_at),
                "the directory names the dead node {authority_at}",
            );
            // The dead-aware oracle surfaces the EXACT orphan (HeldNowhere at the dead node) — pinned
            // to the variant, not a loose is_err (HR5): a real split-brain would be WrongHolderCount.
            assert_eq!(
                verify_authority_unique_excluding(&reports, dead),
                Err(orphan(authority_at)),
                "the dead-aware oracle surfaces the orphan (NOT a false-passing corpse)",
            );
        }
        EndState::DeadOwnerOrphan { authority_at } => {
            let reports = topo.inspect_all();
            assert_eq!(
                live_sagas(topo),
                0,
                "the saga tombstoned (gateway-acked abort)"
            );
            let rec = fault_entity_record(&reports, entity);
            assert_eq!(
                rec.authority,
                AuthorityRef::Shard(authority_at),
                "the directory names the dead owner {authority_at}",
            );
            assert_eq!(rec.in_transfer, None, "the lock cleared (abort_clear)");
            assert_eq!(
                verify_authority_unique_excluding(&reports, dead),
                Err(orphan(authority_at)),
                "the dead-aware oracle surfaces the exact dead-owner orphan",
            );
        }
        // The TRANSIENT end states are held-set / loss-budget shaped (no directory row) — asserted by
        // [`assert_transient_end_state`], never this directory-shaped durable asserter.
        EndState::BatchCommittedAt { .. } | EndState::BatchDroppedWithinBudget { .. } => {
            panic!(
                "transient end states are asserted by assert_transient_end_state, not assert_end_state"
            )
        }
    }
}

/// Drive ONE crash-matrix scenario for a TRANSIENT batch (D-7d) — the held-set/loss-budget sibling of
/// [`run_fault_scenario`] (which is directory-shaped to the bone; a transient writes ZERO directory rows,
/// so forcing both through one driver would need a forbidden match-on-class). REUSES the shared spine
/// (cluster, realm-lease warmup, `seed_transient_crossing`, a Transient `trigger_transfer`, the
/// `Fault`/`CrashWhen` enums, `schedule_crash`/`kill`/`dead_nodes`) verbatim; only the transient-shaped
/// 40% differs (warmup waits for BOTH realm leases; drive-to-phase matches the `BatchHandoff` string;
/// per-tick conservation is dead-aware; quiesce on `live_sagas == 0`). Returns the topo + the debris
/// entity + the dead-node set for [`assert_transient_end_state`].
#[must_use]
pub fn run_transient_fault_scenario(
    seed: u64,
    sc: Scenario,
) -> (Topology, EntityId, BTreeSet<NodeId>) {
    let fabric = FaultFabric::new(seed, 2);
    let mut topo = p2_cluster(&fabric, 8);
    // WARMUP: BOTH shards win their realm leases (the source to cross from, the dest to adopt into).
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        fault_report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == RealmId::System(7))
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let src_fence = realm_fence(&mut topo, RealmId::System(7));
    let dst_fence = realm_fence(&mut topo, RealmId::System(8));
    // Seed a 1-item Debris crossing SHARD→DEST and trigger the matching Transient batch saga.
    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 1, 0);
    let batch = TransferId(1);
    seed_transient_crossing(
        &mut topo,
        debris,
        batch,
        src_fence,
        dst_fence,
        vd_core::glam::DVec3::ZERO,
    );
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(RealmId::System(8)),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: sc.class,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
        },
    );
    // Drive to the target `BatchHandoff` phase (the saga progresses on the choreography acks).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with(sc.at_phase))
    });
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with(sc.at_phase)),
        "the saga is in phase {} when the fault fires (anti-vacuity)",
        sc.at_phase,
    );
    match sc.fault {
        Fault::CrashResurrect { after } => {
            let fire = topo.tick().0 + 1;
            topo.schedule_crash(sc.victim, TickId(fire), sc.crash_when);
            topo.schedule_resurrect(sc.victim, TickId(fire + after));
        }
        Fault::Kill => fabric.kill(sc.victim),
    }
    // PHASE 1 — drive to SAGA quiescence: the dead-resolution fires ~2 redrive windows after the kill
    // (the first Timeout re-emit bounces `NodeUnreachable` → marks the node dead; the next due injects
    // the resolution), tombstoning the saga. Dead-aware per-tick TRANSIENT-CONSERVATION holds EVERY tick
    // (a killed participant's corpse is excluded, so a stale Held claim is never a phantom holder).
    let mut settled = false;
    for i in 0..160 {
        topo.step();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, TickId(i))
            .expect("no transient COUNTED-held by two LIVE shards at any tick under the crash");
        if live_sagas(&mut topo) == 0 {
            settled = true;
            break;
        }
    }
    assert!(
        settled,
        "the transient crash scenario did not reach saga quiescence"
    );
    // PHASE 2 — let the RESOLUTION'S EGRESS settle: the saga tombstones the SAME tick it emits the
    // self-promote (→ the dest flips `Arriving→Held`) / abandon (→ the source drops + buckets the loss),
    // so the SURVIVOR processes it only AFTER quiescence. Conservation still holds each settle tick.
    for i in 0..24 {
        topo.step();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, TickId(i)).expect(
            "no transient COUNTED-held by two LIVE shards while the resolution egress settles",
        );
    }
    let dead = topo.dead_nodes();
    (topo, debris, dead)
}

/// Assert a TRANSIENT crash scenario reached its expected [`EndState`] (D-7d), held-set / loss-budget
/// shaped (a transient has no directory row). The dead-aware `TRANSIENT-AUTHORITY-HELD` holds in EVERY
/// outcome (no double-hold; corpses excluded). `BatchCommittedAt` = the debris held SINGLY at the live
/// survivor + a fired source-unreachable resolution + zero loss; `BatchDroppedWithinBudget` = held at NO
/// live shard + a within-budget NON-ZERO loss + a fired dest-unreachable resolution.
pub fn assert_transient_end_state(
    topo: &mut Topology,
    debris: EntityId,
    dead: &BTreeSet<NodeId>,
    expected: EndState,
) {
    let reports = topo.inspect_all();
    verify_transient_authority_held_excluding(&reports, dead)
        .expect("TRANSIENT-AUTHORITY-HELD (dead-aware) holds after the crash settles");
    let live_holders: Vec<NodeId> = reports
        .iter()
        .filter(|(n, _)| !dead.contains(n))
        .filter(|(_, r)| r.owned_transients.iter().any(|(e, _)| *e == debris))
        .map(|(n, _)| *n)
        .collect();
    match expected {
        EndState::BatchCommittedAt { node } => {
            // OUTCOME-only (reusable by BOTH the SOURCE-kill cell AND the crash-resurrect control, which
            // reaches the same outcome WITHOUT a resolution): the debris settled SINGLY at the live
            // survivor, zero loss. Each cell adds its mechanism check (resolution fired vs NOT fired).
            assert_eq!(
                live_holders,
                vec![node],
                "the debris settled at the live survivor, held by exactly one shard"
            );
            assert_eq!(
                transient_dropped_total(topo),
                0,
                "BatchCommittedAt is ZERO loss (the item was kept, not dropped)"
            );
        }
        EndState::BatchDroppedWithinBudget { kind } => {
            assert!(
                live_holders.is_empty(),
                "the abandoned debris is held at NO live shard: {live_holders:?}"
            );
            verify_transient_loss_budget(&reports, kind)
                .expect("the loss is within the kind's budget");
            assert!(
                total_handover_loss(&reports, kind) > 0,
                "anti-vacuity: a loss was actually counted (not a silent vanish)"
            );
            assert!(
                dest_unreachable_resolutions(topo) > 0,
                "the dead-dest abandon resolution actually fired (anti-vacuity)"
            );
        }
        _ => panic!("assert_transient_end_state handles only the transient end states"),
    }
}

/// REBUILD the orchestrator on its existing identity (D-6 kill-9 recovery): drop the running World (its
/// in-memory saga set / directory / clock — gone with the process RAM) and stand up a FRESH one that
/// RE-HYDRATES from `store`. Mirrors a real restart: [`FaultFabric::reregister`] hands the rebuilt process
/// a fresh transport on the prior peer links (the at-least-once unacked ledger redelivers what the dead
/// process never acked) and [`Topology::replace_node`] swaps the new World in. Built IDENTICALLY to the
/// original (same [`orch_config`]) so recovery is a clean re-hydrate. Pass the RETAINED handle from
/// [`p2_cluster_durable_orch`] to recover; pass a fresh `MemStore::new()` to model a NON-durable restart
/// (the anti-theater control — nothing is recovered). The caller has typically `kill`ed the node first.
pub fn rebuild_orchestrator(topo: &mut Topology, fabric: &FaultFabric, store: MemStore) {
    let mut orch = build_app(
        NodeConfig {
            node_id: ORCH,
            kind: NodeKind::Orchestrator,
        },
        fabric.reregister(ORCH),
    );
    let (world, schedule) = orch.parts_mut();
    register_orchestrator_with_store(
        world,
        schedule,
        &orch_config(vec![GATEWAY, SHARD, DEST]),
        Box::new(store),
    );
    topo.replace_node(Box::new(orch));
}

/// The outcome of an orchestrator kill-9 + rebuild ([`run_orch_kill_transient`] / [`run_orch_kill_durable`]).
pub struct OrchKillOutcome {
    /// The cluster, settled (recovered) or frozen right after the rebuild (anti-theater).
    pub topo: Topology,
    /// The transferred subject — a debris item (transient) or a player avatar (durable).
    pub subject: EntityId,
    /// Dead nodes after the rebuild — EMPTY (the orchestrator is alive again), for the dead-aware oracle.
    pub dead: BTreeSet<NodeId>,
    /// `saga_states` captured the INSTANT the rebuilt orchestrator boots, before it re-drives anything —
    /// THE recovery evidence: a `BatchHandoff` saga (re-hydrated from the retained WAL) when durable,
    /// EMPTY (nothing survived the dropped World) for the fresh-store anti-theater control.
    pub recovered_states: Vec<String>,
}

/// D-6 e2e: kill-9 the orchestrator MID-`BatchHandoff` of a live transient batch, then rebuild it. With
/// `durable`, the rebuilt orchestrator re-hydrates the retained durable WAL, re-drives the in-flight
/// handoff to `Done`, and the debris settles at [`DEST`] with zero loss — an orchestrator crash mid-handoff
/// never wedges nor loses the batch. With `!durable` (rebuilt against a FRESH empty store) it recovers
/// NOTHING — the anti-theater control proving the recovery rides the persisted WAL, not in-process World
/// survival across the kill. Conservation is asserted EVERY tick of the re-drive (no transient is ever
/// COUNTED-held by two live shards across the orchestrator's death + recovery).
#[must_use]
pub fn run_orch_kill_transient(seed: u64, durable: bool) -> OrchKillOutcome {
    let fabric = FaultFabric::new(seed, 2);
    let (mut topo, store) = p2_cluster_durable_orch(&fabric, 8);
    // WARMUP: BOTH shards win their realm leases (the source to cross from, the dest to adopt into).
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        fault_report(&r, SHARD)
            .held_realms
            .iter()
            .any(|(realm, _)| *realm == RealmId::System(7))
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let src_fence = realm_fence(&mut topo, RealmId::System(7));
    let dst_fence = realm_fence(&mut topo, RealmId::System(8));
    let debris = EntityId::pack(EntityKind::Debris, SHARD.0 as u32, 1, 0);
    let batch = TransferId(1);
    seed_transient_crossing(
        &mut topo,
        debris,
        batch,
        src_fence,
        dst_fence,
        vd_core::glam::DVec3::ZERO,
    );
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: batch,
            session: SessionId(0),
            subject: DirectoryKey::Realm(RealmId::System(8)),
            expected_fence: dst_fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Transient,
            needs_provision: false,
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
        },
    );
    // Drive to the `BatchHandoff` tail — the saga is quiescent-but-persisted (the kill target).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with("BatchHandoff"))
    });
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with("BatchHandoff")),
        "anti-vacuity: the saga is in BatchHandoff when the orchestrator is killed"
    );
    // The orchestrator PERSISTED durable state before the kill — else the recovery is vacuous. (The
    // BatchHandoff assert above proves the saga is the in-flight thing; the cell's `recovered_states`
    // assert proves the saga specifically rehydrated. This is the "there IS a committed WAL" floor.)
    assert!(
        !store.is_empty(),
        "the orchestrator committed durable state (incl. the in-flight saga) before the kill"
    );
    // KILL-9: `crash` (not `kill`) is the restart-able death — the orchestrator's in-process inbound is
    // cleared (RAM lost), but the at-least-once ledger HOLDS the in-flight shard acks (e.g. a dest
    // `BatchAdopted` sent while it is down) for redelivery once it is back; `kill` is permanent death
    // (bounces + DROPS those acks, which the producer-less `AwaitAdopt` phase could never re-solicit).
    // The World itself dies at `replace_node` below — `crash` + rehydrated `replace_node` = kill-9+restart.
    fabric.crash(ORCH);
    // OUTAGE: a few ticks down — the orchestrator is skipped; the dest adopts off the source's envelope
    // and acks `BatchAdopted`, which the fabric HOLDS (blocked on the crashed orchestrator, rescheduled).
    // Conservation holds even here (the dead orchestrator is excluded; the dest's `Arriving` is uncounted).
    for _ in 0..4 {
        topo.step();
        let tick = topo.tick();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, tick)
            .expect("no transient COUNTED-held by two LIVE shards while the orchestrator is down");
    }
    // REBUILD on the SAME identity: durable → re-hydrate the retained store; anti-theater → a FRESH
    // (empty) store, which recovers NOTHING (the proof the retained WAL is load-bearing).
    let recover_store = if durable {
        store.clone()
    } else {
        MemStore::new()
    };
    rebuild_orchestrator(&mut topo, &fabric, recover_store);
    // THE recovery evidence — captured the instant the rebuilt World boots, before it re-drives.
    let recovered_states = saga_states(&mut topo);
    if !durable {
        // The anti-theater control stops here: nothing re-hydrated, so there is nothing to re-drive.
        let dead = topo.dead_nodes();
        return OrchKillOutcome {
            topo,
            subject: debris,
            dead,
            recovered_states,
        };
    }
    // QUIESCE: the re-hydrated saga re-arms (since=0) → the deadline producer re-emits the `BatchHandoff`
    // egress, the redelivered acks land, the handoff completes. Conservation holds EVERY tick.
    let mut settled = false;
    for _ in 0..200 {
        topo.step();
        let tick = topo.tick();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, tick).expect(
            "no transient COUNTED-held by two LIVE shards at any tick across the orchestrator kill-9",
        );
        if live_sagas(&mut topo) == 0 {
            settled = true;
            break;
        }
    }
    assert!(
        settled,
        "the rebuilt orchestrator re-drove the in-flight transient batch to quiescence"
    );
    // Let the recovered handoff's resolution egress settle (mirror `run_transient_fault_scenario`).
    for _ in 0..24 {
        topo.step();
        let tick = topo.tick();
        let dead = topo.dead_nodes();
        let reports = topo.inspect_all();
        verify_transient_conservation_tick_excluding(&reports, &dead, tick)
            .expect("conservation holds while the recovered handoff's egress settles");
    }
    let dead = topo.dead_nodes();
    OrchKillOutcome {
        topo,
        subject: debris,
        dead,
        recovered_states,
    }
}

/// D-6 e2e: kill-9 the orchestrator mid-saga of a live DURABLE player transfer, then rebuild it — the
/// P3 headline for the class that carries PLAYERS (zero loss, AUTHORITY-UNIQUE). The durable twin of
/// [`run_orch_kill_transient`]: a separate driver because the durable path is directory-shaped (a client
/// logs in + walks, the avatar is a directory `Entity` record, the saga runs the full
/// `Preparing→…→Demoting→Promoting` choreography) where the transient is held-set-shaped — forcing both
/// through one driver would need a forbidden match-on-class. The SHARED spine is reused verbatim:
/// `p2_cluster_durable_orch` (retained Store), `fault_step_until`, `fabric.crash` + `rebuild_orchestrator`,
/// the `recovered_states` capture, the quiesce loop. Drive to `at_phase`, `crash` the orchestrator, rebuild
/// from the retained store (or a FRESH one for the anti-theater control), and let the rehydrated saga
/// re-drive. POST-commit phases (e.g. `Demoting`) re-drive forward to `Done`; the cell asserts via
/// [`assert_end_state`]. With `!durable_store` the rebuilt orchestrator recovers NOTHING (the control).
#[must_use]
pub fn run_orch_kill_durable(seed: u64, at_phase: &str, durable_store: bool) -> OrchKillOutcome {
    let fabric = FaultFabric::new(seed, 2);
    let (mut topo, store) = p2_cluster_durable_orch(&fabric, 8);
    topo.add_node(Box::new(p1_client(
        &fabric,
        FAULT_CLIENT,
        AccountId(1000),
        walk_forward(),
    )));
    // WARMUP: the player logs in + walks; the SOURCE grants its avatar and the DEST wins its realm.
    fault_step_until(&mut topo, 80, |t| {
        let r = t.inspect_all();
        !fault_report(&r, SHARD).held_entities.is_empty()
            && fault_report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });
    let (session, entity, fence) = read_subject(&mut topo);
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
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
        },
    );
    // Drive to the target saga phase (the client auto-stamps the CUT_MARKER on RequestCut).
    fault_step_until(&mut topo, 60, |t| {
        saga_states(t).iter().any(|s| s.starts_with(at_phase))
    });
    assert!(
        saga_states(&mut topo)
            .iter()
            .any(|s| s.starts_with(at_phase)),
        "anti-vacuity: the saga is in the target phase when the orchestrator is killed"
    );
    assert!(
        !store.is_empty(),
        "the orchestrator committed durable state (incl. the in-flight saga) before the kill"
    );
    // KILL-9: `crash` (restart-able) clears the orchestrator's in-process inbound (RAM lost) while the
    // at-least-once ledger HOLDS the in-flight shard acks for redelivery; the World dies at `replace_node`.
    fabric.crash(ORCH);
    // OUTAGE: a few ticks down. The client keeps emitting to the GATEWAY (unaffected by the orchestrator
    // crash); the source/dest acks are held for redelivery. (No per-tick AUTHORITY-UNIQUE here — that
    // oracle reads the directory on the crashed orchestrator; the end-state assert covers recovery.)
    for _ in 0..4 {
        topo.step();
    }
    let recover_store = if durable_store {
        store.clone()
    } else {
        MemStore::new()
    };
    rebuild_orchestrator(&mut topo, &fabric, recover_store);
    let recovered_states = saga_states(&mut topo);
    if !durable_store {
        let dead = topo.dead_nodes();
        return OrchKillOutcome {
            topo,
            subject: entity,
            dead,
            recovered_states,
        };
    }
    // QUIESCE: the re-hydrated saga re-arms (since=0) → the deadline producer re-drives the choreography
    // (POST-commit phases re-emit Demote/Promote idempotently → forward to Done at DEST; a PRE-commit phase
    // fires the abort deadline → ThawSource → terminal Aborted at SOURCE). Either way the saga tombstones.
    for _ in 0..200 {
        topo.step();
        if live_sagas(&mut topo) == 0 {
            break;
        }
    }
    // SETTLE the terminal egress: the saga tombstones the SAME tick it emits its last action (the final
    // Promote/ReleaseComplete on the commit path, or the ThawSource + AbortTransfer teardown on the abort
    // path), so the SOURCE/DEST applies it only AFTER quiescence (mirror `run_transient_fault_scenario`).
    // (The abort path is FENCE-NEUTRAL — no fence sync to wait for; this lets the dest ghost teardown land.)
    for _ in 0..24 {
        topo.step();
    }
    let dead = topo.dead_nodes();
    OrchKillOutcome {
        topo,
        subject: entity,
        dead,
        recovered_states,
    }
}
