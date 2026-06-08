//! # vd-tests — accumulated scenario suites
//!
//! Shared scenario builders/fixtures used by the integration tests in `tests/tests/`.
//! Standing rule: every phase ADDS scenarios; nothing is deleted. The accumulated
//! suite re-running green is the release gate for every later phase.

use vd_connection_plane::gateway::{GatewayConfig, TransportTuning, register_gateway};
use vd_connection_plane::tickets;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{AccountId, EpochId, NodeId};
use vd_harness::client::{DeliveredWorldView, InputCmd, ScriptedClient};
use vd_harness::fabric::FaultFabric;
use vd_harness::topology::{StaggerPlan, Topology};
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_node::orchestrator::{OrchestratorConfig, register_orchestrator};
use vd_sim::capability::NodeKind;
use vd_sim::directory::DirectoryTuning;
use vd_sim::stub::{StubConfig, register_stub_shard};

/// The canonical P1 node ids (one orchestrator, one gateway, one stub shard;
/// clients from 100 upward).
pub const ORCH: NodeId = NodeId(1);
pub const GATEWAY: NodeId = NodeId(2);
pub const SHARD: NodeId = NodeId(3);

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

/// Build the canonical P1 cluster (orchestrator + gateway + stub shard) onto a
/// fabric-backed topology. Clients are added by the caller.
#[must_use]
pub fn p1_cluster(fabric: &FaultFabric, max_sessions: usize) -> Topology {
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
            clock_peers: vec![GATEWAY, SHARD],
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
            tuning: TransportTuning { max_sessions },
        },
    );
    topo.add_node(Box::new(gateway));

    let mut shard = build_app(
        NodeConfig {
            node_id: SHARD,
            kind: NodeKind::StubShard,
        },
        fabric.register(SHARD),
    );
    let (world, schedule) = shard.parts_mut();
    register_clock_follower(world, schedule);
    register_stub_shard(world, schedule, stub_config());
    topo.add_node(Box::new(shard));

    topo
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
