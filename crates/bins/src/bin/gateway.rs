//! The gateway binary: a thin shell over `vd-connection-plane` (the lib owns ALL
//! session/route logic). Config → mesh → build_app → register → tick loop.

use vd_connection_plane::gateway::{GatewayConfig, TransportTuning, register_gateway};
use vd_io_prod::mesh::{MeshConfig, spawn_mesh};
use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_sim::capability::NodeKind;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let env = EnvConfig::from_process_env();
    let local = env.node_id("VD_NODE_ID")?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let trust = ClusterTrust::from_der_dir(std::path::Path::new(&env.string("VD_TRUST_DIR")?))?;
    let (transport, _control) = spawn_mesh(
        runtime.handle(),
        &trust,
        &MeshConfig::new(
            local,
            env.parse("VD_BIND")?,
            env.peer_book("VD_PEERS")?,
            env.parse("VD_OUTBOUND_CAP")?,
            // R-2b: per-process incarnation stamped on reliable frames (default 0; the durable
            // monotone boot-counter lands at R-6). The dev cluster sets it per launch (DEFERRED D-6).
            env.parse_or("VD_PROCESS_INCARNATION", 0)?,
        ),
    )?;
    let mut node = build_app(
        NodeConfig {
            node_id: local,
            kind: NodeKind::Gateway,
        },
        transport,
    );
    let (world, schedule) = node.parts_mut();
    register_clock_follower(world, schedule);
    let shard = env.node_id("VD_SHARD")?;
    // D-3 Slice 5b: the proactive self-fence cadence is a node-side knob — validate its split-brain
    // safety (armed ⇒ a recheck channel exists AND grace >= 2 recheck cycles) HERE at boot, so a
    // mis-tuned config can never silently mass-self-fence healthy sessions.
    let session_recheck_interval: u64 = env.parse_or("VD_SESSION_RECHECK", 0)?;
    let self_fence_grace_ticks: u64 = env.parse_or("VD_SELF_FENCE_GRACE", 0)?;
    vd_sim::directory::validate_self_fence_cadence(
        self_fence_grace_ticks,
        session_recheck_interval,
    )?;
    register_gateway(
        world,
        schedule,
        GatewayConfig {
            orchestrator: env.node_id("VD_ORCH")?,
            shard,
            // P1 single-shard roster: the one login shard is the only routable shard
            // (Track R / 1d.2 multi-shard cluster rosters extend this set).
            known_shards: std::collections::BTreeSet::from([shard]),
            auth_verifying_key: env.hex32("VD_AUTH_PUBKEY")?,
            session_seed: env.parse("VD_SESSION_SEED")?,
            // The SAME VD_TICK_HZ that paces this node — relayed to clients via
            // UniverseRate so the render cursor tracks the cluster's rate (R1).
            tick_hz: env.parse("VD_TICK_HZ")?,
            // D-3 session-lease heartbeat cadence (the gateway's local copy). INERT (0) until D-3 is on.
            lease_renew_interval_ticks: env.parse_or("VD_LEASE_RENEW_INTERVAL", 0)?,
            // D-3 Slice 5b: session-head recheck cadence (the round-trip confirmation channel) + the
            // proactive self-fence grace. Both INERT (0) by default; validated above at boot (armed ⇒ a
            // recheck channel exists AND grace spans >= 2 recheck cycles — the no-mass-fence guard).
            session_recheck_interval,
            self_fence_grace_ticks,
            tuning: TransportTuning {
                max_sessions: env.parse("VD_MAX_SESSIONS")?,
                max_buffered_inputs: env.parse("VD_MAX_BUFFERED_INPUTS")?,
            },
        },
    );
    let mut pacer = TickPacer::new(env.parse("VD_TICK_HZ")?);
    loop {
        let _ = node.step_tick();
        let _ = pacer.wait();
    }
}
