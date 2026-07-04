//! THE shard binary (HR3: one binary; shard types are `ShardProfile` configs —
//! P1 runs the stub profile). A thin shell: config → mesh → build_app → register
//! → tick loop. All logic lives in the tested libs.

use vd_io_prod::mesh::{MeshConfig, spawn_mesh};
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
            // R-6a: the DURABLE MONOTONE process incarnation stamped on reliable frames (VD_PROCESS_INCARNATION
            // explicit wins for dev/test; else the VD_BOOT_STATE_DIR boot-counter that survives a CrashLoop).
            vd_bins::resolve_process_incarnation(&env)?,
        ),
        // R-6d3a: the durable outbox handle — `None` until R-6d3b opens + boot-replays the per-node store
        // (the durable-before-send gate is inert without it: the same 2c-style inert-safe posture).
        None,
    )?;
    // GW-1 §6.3: fail LOUD at boot if the snapshot budget exceeds the conservative
    // datagram floor — a misconfiguration must never become a silent runtime drop.
    let snapshot_budget: usize = env.parse("VD_SNAPSHOT_BUDGET")?;
    assert!(
        snapshot_budget <= vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET,
        "VD_SNAPSHOT_BUDGET {snapshot_budget} exceeds the conservative datagram floor {}",
        vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET
    );
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
    let realm_recheck_interval: u64 = env.parse("VD_REALM_RECHECK")?;
    let self_fence_grace_ticks: u64 = env.parse_or("VD_SELF_FENCE_GRACE", 0)?;
    vd_sim::directory::validate_self_fence_cadence(self_fence_grace_ticks, realm_recheck_interval)?;
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
    loop {
        let _ = node.step_tick();
        let _ = pacer.wait();
    }
}
