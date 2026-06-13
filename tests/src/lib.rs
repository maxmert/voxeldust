//! # vd-tests — accumulated scenario suites
//!
//! Shared scenario builders/fixtures used by the integration tests in `tests/tests/`.
//! Standing rule: every phase ADDS scenarios; nothing is deleted. The accumulated
//! suite re-running green is the release gate for every later phase.

use vd_connection_plane::gateway::{
    GatewayConfig, GatewayStats, TransportTuning, register_gateway,
};
use vd_connection_plane::tickets;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{AccountId, EntityId, EpochId, Fence, NodeId, SessionId};
use vd_harness::client::{DeliveredWorldView, InputCmd, ScriptedClient};
use vd_harness::fabric::{FabricTransport, FaultFabric};
use vd_harness::topology::{StaggerPlan, Topology};
use vd_node::ShardNode;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_node::orchestrator::{DirectoryRes, OrchestratorConfig, register_orchestrator};
use vd_node::saga_runtime::SagaRuntimeRes;
use vd_sim::capability::NodeKind;
use vd_sim::directory::DirectoryTuning;
use vd_sim::saga::SagaCtx;
use vd_sim::stub::{StubConfig, register_stub_shard};
use vd_wire::seams::directory::DirectoryKey;

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
        realm_recheck_interval: 0,
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
) -> Topology {
    let mut topo = Topology::new(fabric.clone(), StaggerPlan::lockstep());

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
    let (world, schedule) = gateway.parts_mut();
    register_clock_follower(world, schedule);
    register_gateway(
        world,
        schedule,
        GatewayConfig {
            orchestrator: ORCH,
            shard: SHARD,
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
