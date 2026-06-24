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

use vd_core::entity_kind::DurabilityClass;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{AccountId, EntityId, EpochId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_harness::client::{DeliveredWorldView, InputCmd, ScriptedClient};
use vd_harness::fabric::{CrashWhen, FabricTransport, FaultFabric};
use vd_harness::oracle::{
    AuthorityViolation, verify_authority_settled, verify_authority_unique_excluding,
};
use vd_harness::topology::{InspectReport, StaggerPlan, Topology};
use vd_node::ShardNode;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_node::orchestrator::{DirectoryRes, OrchestratorConfig, register_orchestrator};
use vd_node::saga_runtime::SagaRuntimeRes;
use vd_sim::capability::NodeKind;
use vd_sim::directory::DirectoryTuning;
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
fn build_cluster(
    fabric: &FaultFabric,
    max_sessions: usize,
    clock_peers: Vec<NodeId>,
    shards: Vec<(NodeId, StubConfig)>,
    stagger: StaggerPlan,
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
    register_orchestrator(
        world,
        schedule,
        &OrchestratorConfig {
            epoch: EpochId(1),
            reserve_chunk: 1024,
            clock_peers,
            directory: DirectoryTuning {
                lease_ttl_ticks: 1_000,
            },
            // Slice 2a: the deadline producer runs LIVE in the capstone cluster (dev values 8/24).
            saga: vd_sim::saga::SagaTuning::default(),
        },
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
    )
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
    /// D-37 (RED): the saga TOMBSTONED to a dead-owner orphan (directory names the dead `authority_at`
    /// at a bumped fence, lock cleared, no live saga) — the dead-aware oracle surfaces it.
    DeadOwnerOrphan { authority_at: NodeId },
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
    }
}
