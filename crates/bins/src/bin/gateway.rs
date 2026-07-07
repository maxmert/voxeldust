//! The gateway binary: a thin shell over `vd-connection-plane` (the lib owns ALL
//! session/route logic). Config → mesh → build_app → register → tick loop.

use vd_connection_plane::gateway::{GatewayConfig, TransportTuning, register_gateway};
use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_sim::capability::NodeKind;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let env = EnvConfig::from_process_env();
    let local = env.node_id("VD_NODE_ID")?;
    // Cloud-ready k3d Slice 2: the footgun preflight + resolved D-3, BEFORE `boot_mesh_and_replay` (the
    // preflight must veto a manual incarnation / ephemeral escape before that resolves the durable M3 boot-
    // counter). The gateway is the ONLY node that holds the session-auth verifying key, so it passes it in
    // for the cloud dev-key veto (a cloud deploy must never trust the checked-in dev signer).
    // Stringify so a cloud footgun (incl. the dev-auth-key veto) prints its actionable Display guidance in
    // `kubectl logs`, never a bare Debug variant dump.
    let d3 = vd_bins::resolve_node_d3(
        &env,
        vd_io_prod::boot::NodeRole::Gateway,
        Some(vd_bins::dev_auth_pubkey_bytes()),
    )
    .map_err(|e| e.to_string())?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let trust = ClusterTrust::from_der_dir(std::path::Path::new(&env.string("VD_TRUST_DIR")?))?;
    // R-6d3b-2b: the ONE shared boot sequence (HR3 — the SAME helper the shard uses). Resolves incarnation
    // once, opens+wraps the durable outbox, spawns the mesh WITH the sink, replays retained rows before
    // build_app. HR3 uniformity: the gateway opens one iff VD_OUTBOX_PATH is set (empty until it has a
    // producer-less flow). `_control` + `runtime` stay bound for the tick loop.
    let (transport, _control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
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
    // D-3 Slice 5b: the proactive self-fence cadence is a node-side knob whose split-brain safety
    // (armed ⇒ a recheck channel exists AND grace >= 2 recheck cycles) is validated inside `resolve_d3`
    // (via `validate_self_fence_cadence`) at the top, so a mis-tuned config can never silently mass-self-
    // fence healthy sessions. Cloud-ready k3d Slice 2: `d3.node_recheck` / `d3.node_self_fence_grace`
    // REPLACE the old direct env reads + the inline validate. In DevTest these are the inert 0/0.
    let session_recheck_interval: u64 = d3.node_recheck;
    let self_fence_grace_ticks: u64 = d3.node_self_fence_grace;
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
    // Cloud-ready k3d Slice 1: poll a SIGTERM/SIGINT flag each tick so a routine pod-stop / gateway
    // rolling-deploy drains cleanly instead of a hard SIGKILL crash-path.
    let shutdown = vd_bins::install_shutdown_flag(runtime.handle());
    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = node.step_tick();
        let _ = pacer.wait();
    }
    // GRACEFUL DRAIN: the gateway holds no un-fsynced durable state (its R-6d outbox — empty until it has a
    // producer-less flow — is fsync-before-send; sessions re-adopt via their ResumeTicket on reconnect). A
    // clean return drops `node`/`runtime`/`_control`, tearing down the endpoint + peer writers in order.
    tracing::info!("gateway drained on shutdown signal — exiting cleanly");
    Ok(())
}
