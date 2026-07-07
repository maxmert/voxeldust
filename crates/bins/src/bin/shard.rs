//! THE shard binary (HR3: one binary; shard types are `ShardProfile` configs —
//! P1 runs the stub profile). A thin shell: config → mesh → build_app → register
//! → tick loop. All logic lives in the tested libs.

use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::register_clock_follower;
use vd_sim::capability::NodeKind;
use vd_sim::stub::{StubConfig, register_stub_shard};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let env = EnvConfig::from_process_env();
    let local = env.node_id("VD_NODE_ID")?;
    // Cloud-ready k3d Slice 2: the footgun preflight + resolved D-3, BEFORE `boot_mesh_and_replay` (the
    // preflight must veto a manual incarnation / ephemeral escape before that resolves the durable M3 boot-
    // counter). The shard holds no auth key ⇒ dev_pubkey = None.
    // Stringify so a cloud footgun/incoherence prints its actionable Display guidance in `kubectl logs`.
    let d3 = vd_bins::resolve_node_d3(&env, vd_io_prod::boot::NodeRole::Shard, None)
        .map_err(|e| e.to_string())?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let trust = ClusterTrust::from_der_dir(std::path::Path::new(&env.string("VD_TRUST_DIR")?))?;
    // GW-1 §6.3: fail LOUD at boot if the snapshot budget exceeds the conservative datagram floor — a
    // misconfiguration must never become a silent runtime drop. Checked BEFORE the mesh/replay so the loud
    // guard is instantaneous (never after a bounded boot-replay fence).
    let snapshot_budget: usize = env.parse("VD_SNAPSHOT_BUDGET")?;
    assert!(
        snapshot_budget <= vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET,
        "VD_SNAPSHOT_BUDGET {snapshot_budget} exceeds the conservative datagram floor {}",
        vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET
    );
    // R-6d3b-2b: the ONE shared boot sequence — resolve incarnation once, open+wrap the durable outbox, spawn
    // the mesh WITH the sink (durable-before-send gate LIVE), replay retained rows BEFORE build_app. `_control`
    // + `runtime` stay bound for the whole tick loop (dropping either tears down the endpoint / peer-writers).
    let (transport, _control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
    let mut node = build_app(
        NodeConfig {
            node_id: local,
            kind: NodeKind::StubShard,
        },
        transport,
    );
    let (world, schedule) = node.parts_mut();
    register_clock_follower(world, schedule);
    let realm_seed: u64 = env.parse("VD_REALM_SEED")?;
    // D-3 Slice 5: the proactive self-fence cadence is a NODE-side knob the orchestrator never sees, so
    // its split-brain-safety (armed ⇒ a confirmation channel exists AND the grace spans >= 2 recheck
    // cycles, so a healthy holder never self-fences between on-time replies) is validated HERE at boot —
    // loud, never a silent mass-self-fence of healthy shards.
    // Cloud-ready k3d Slice 2: the node D-3 config comes from the shared resolver (footgun preflight +
    // derived-or-inert D-3, run at the top before `boot_mesh_and_replay`). `d3.node_recheck` /
    // `d3.node_self_fence_grace` REPLACE the old direct env reads + the inline `validate_self_fence_cadence`
    // (which now runs inside `resolve_d3`). In DevTest these are the inert 0/0 (byte-identical to before).
    let realm_recheck_interval: u64 = d3.node_recheck;
    let self_fence_grace_ticks: u64 = d3.node_self_fence_grace;
    register_stub_shard(
        world,
        schedule,
        StubConfig {
            realm: vd_core::pose::RealmId::System(realm_seed),
            frame: vd_core::pose::FrameRef::SystemSpace {
                system_seed: realm_seed,
            },
            move_speed_mps: env.parse("VD_SPEED")?,
            tick_dt_s: env.parse("VD_TICK_DT")?,
            orchestrator: env.node_id("VD_ORCH")?,
            mint_seed: env.parse("VD_MINT_SEED")?,
            // Bounded in production: the input-conservation log is a metrics ring,
            // not an unbounded audit trail (SCALE-3).
            input_log_capacity: env.parse("VD_INPUT_LOG_CAP")?,
            // How often to re-read the realm head to observe a lost lease (FENCE-1/5/8) — ALSO the
            // round-trip that re-arms RealmConfirmedAt for the proactive self-fence (D-3 Slice 5).
            realm_recheck_interval,
            // D-3 lease-renewal heartbeat cadence (the holder's local copy of the orchestrator's
            // lease_renew_interval_ticks). Defaults INERT (0 = no heartbeat) until D-3 is switched on.
            lease_renew_interval_ticks: env.parse_or("VD_LEASE_RENEW_INTERVAL", 0)?,
            // D-3 Slice 5 proactive self-fence grace (the holder's local copy of self_fence_grace_ticks).
            // Defaults INERT (0). The grace-vs-ttl ordering is validated orchestrator-side; the
            // grace-vs-recheck cadence (the no-mass-fence guard) is validated above at this node's boot.
            self_fence_grace_ticks,
            // Per-datagram snapshot budget — partitioned so none exceeds the MTU (GW-1).
            snapshot_datagram_budget: snapshot_budget,
        },
    );
    let mut pacer = TickPacer::new(env.parse("VD_TICK_HZ")?);
    // Cloud-ready k3d Slice 1: poll a SIGTERM/SIGINT flag each tick so a routine pod-stop breaks the loop
    // and drains cleanly instead of being a hard SIGKILL crash-path.
    let shutdown = vd_bins::install_shutdown_flag(runtime.handle());
    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = node.step_tick();
        let _ = pacer.wait();
    }
    // GRACEFUL DRAIN: the shard holds no un-fsynced durable state — its R-6d outbox is fsync-BEFORE-send and
    // any in-flight mesh send is recovery-covered by the outbox replay on restart. So a clean return
    // suffices: `node`, `runtime`, and `_control` drop, tearing down the endpoint + peer writers in order.
    tracing::info!("shard drained on shutdown signal — exiting cleanly");
    Ok(())
}
