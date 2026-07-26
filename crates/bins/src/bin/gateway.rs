//! The gateway binary: a thin shell over `vd-connection-plane` (the lib owns ALL
//! session/route logic). Config → mesh → build_app → register → tick loop.

use vd_connection_plane::gateway::{
    GatewayConfig, GatewaySessions, SeedInjectorConfig, TransportTuning, register_gateway,
};
use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::{FollowerState, register_clock_follower};
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
    // producer-less flow). `control` (Arc-wrapped below for the auto-resolver) + `runtime` stay bound.
    let (transport, control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
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
    // Hoisted so the S3 readiness predicate gates on the SAME capacity the admission gate rejects on.
    let max_sessions: usize = env.parse("VD_MAX_SESSIONS")?;
    let tick_hz: u32 = env.parse("VD_TICK_HZ")?;
    // interplay-02 (holistic audit): the RUNTIME twin of the compile-time DRAIN-BURST assert. The gateway
    // drains up to `max_buffered_inputs` cut-buffered frames to the dest shard in ONE tick; that MUST stay
    // within the dest inbox floor `inbound_capacity_for(outbound_cap)`, or a cloud ConfigMap that lowers
    // VD_OUTBOUND_CAP (without the gateway's VD_MAX_BUFFERED_INPUTS) silently sheds a conserved resume-input
    // at the transfer-commit moment (D-8). VD_OUTBOUND_CAP is the cluster-wide dest cap (DEV value off-cloud).
    let max_buffered_inputs: usize = env.parse("VD_MAX_BUFFERED_INPUTS")?;
    let dest_outbound_cap: usize =
        env.parse_or("VD_OUTBOUND_CAP", vd_bins::DEV.outbound_cap as usize)?;
    vd_bins::validate_drain_burst(dest_outbound_cap, max_buffered_inputs)?;
    // RLM 5f-3c — the TRUSTED GATEWAY SEED INJECTOR inputs. `VD_DEMAND` ARMS it (unset ⇒ INERT, byte-
    // identical: a login emits no `RealmDemand`); the live-arming veto (mutual exclusion with a static
    // forest, mirroring the orchestrator's `VD_DEMAND` XOR `VD_STATIC_FOREST`) is 5f-3e — this is only the
    // on/off flag. `VD_UNIVERSE_SEED` is the SAME cluster seed the shard reads. The per-account STORED
    // spawn poses REUSE 5f-3b's `resolve_spawn_poses` (the SAME `VD_SPAWN_POSES` map the shard admits at,
    // so the gateway-DERIVED home coord and the shard-side admit pose agree). The forest is walk-scale (the
    // P3 scope of `container_coord_at`; the visual/canonical lazy generator is the P4 owe).
    let demand_armed = vd_bins::parse_bool_env(&env, "VD_DEMAND")?;
    let universe_seed: u64 = env.parse_or("VD_UNIVERSE_SEED", 0)?;
    let spawn_poses = vd_bins::resolve_spawn_poses(&env)?;
    register_gateway(
        world,
        schedule,
        GatewayConfig {
            orchestrator: env.node_id("VD_ORCH")?,
            shard,
            // Track R / 1d.2: the STABLE routable-shard roster — the login shard PLUS every shard in
            // VD_KNOWN_SHARDS (parsed by the EXISTING `EnvConfig::node_list`), so a (render-ready) DEST's
            // frames are node-class dispatchable (`is_known_shard` ⇒ they reach `on_shard_frame` instead
            // of dropping as an unknown peer). The login `shard` is ALWAYS a member. Absent VD_KNOWN_SHARDS
            // ⇒ just {shard} (a single-shard `up` stays byte-identical). SCOPE (M-2): this local playground
            // is 2-shard; the N-shard k3d roster is ledgered to cloud #123 in DEFERRED.md.
            known_shards: {
                let mut set = std::collections::BTreeSet::from([shard]);
                set.extend(env.node_list("VD_KNOWN_SHARDS").unwrap_or_default());
                set
            },
            auth_verifying_key: env.hex32("VD_AUTH_PUBKEY")?,
            session_seed: env.parse("VD_SESSION_SEED")?,
            // The SAME VD_TICK_HZ that paces this node — relayed to clients via
            // UniverseRate so the render cursor tracks the cluster's rate (R1).
            tick_hz,
            // D-3 session-lease heartbeat cadence (the gateway's local copy). INERT (0) until D-3 is on.
            lease_renew_interval_ticks: env.parse_or("VD_LEASE_RENEW_INTERVAL", 0)?,
            // D-3 Slice 5b: session-head recheck cadence (the round-trip confirmation channel) + the
            // proactive self-fence grace. Both INERT (0) by default; validated above at boot (armed ⇒ a
            // recheck channel exists AND grace spans >= 2 recheck cycles — the no-mass-fence guard).
            session_recheck_interval,
            self_fence_grace_ticks,
            // 3g abort-leg lever: INERT in production (a test-only one-shot; a real gateway never
            // rejects a prepare via this knob). Behaviour-identical to the pre-3g `Ready` stub.
            reject_next_prepare: None,
            // RLM 5f-3c — the trusted-gateway seed injector. UNARMED (VD_DEMAND unset) ⇒ INERT (byte-
            // identical). The forest is walk-scale (container_coord_at's P3 scope); the pose store is the
            // SAME map the shard admits at (5f-3b), so the derived home coord and the admit pose agree.
            seed_injector: SeedInjectorConfig {
                armed: demand_armed,
                universe_seed,
                universe_config: vd_core::worldgen::UniverseConfig::walk_scale(),
                spawn_poses,
            },
            tuning: TransportTuning {
                max_sessions,
                max_buffered_inputs,
            },
        },
    );
    let mut pacer = TickPacer::new(tick_hz);
    // Cloud-ready k3d Slice 3: the k8s probe surface (see shard.rs). Slice 1 SIGTERM flag → the with-health
    // variant so the shutdown EDGE de-routes + keeps /healthz LIVE through the gateway rolling-deploy drain.
    let health = vd_io_prod::probe::new_health_cell();
    let shutdown_linger = vd_bins::resolve_shutdown_linger(&env)?;
    let shutdown = vd_bins::install_shutdown_flag_with_health(runtime.handle(), health.clone());
    if let Some((probe_addr, probe_tuning)) = vd_bins::resolve_probe(&env)? {
        vd_bins::spawn_probe_server(
            runtime.handle(),
            probe_addr,
            std::sync::Arc::new(vd_io_prod::probe::PublishedHealth::new(
                health.clone(),
                probe_tuning.stall_deadline(tick_hz),
            )),
            // gateway carries no incarnation cookie (RLM 5c: /whoami is realm-shard-only).
            None,
        );
    }
    // Cloud reschedule re-plumb: the peer-addr auto-resolver (production caller of update_peer_addr) — spawned
    // iff VD_PEER_HOSTS is set (cloud deploy), a no-op in-process. Shares the tick loop's shutdown flag so it
    // drains on SIGTERM. `control` stays bound (Arc) so the endpoint lives for the whole run. So a rescheduled
    // shard's new IP is re-plumbed and the gateway can keep ROUTING inputs to it.
    let control = std::sync::Arc::new(control);
    vd_bins::spawn_peer_resolver_if_configured(
        &env,
        runtime.handle(),
        std::sync::Arc::clone(&control),
        std::sync::Arc::clone(&shutdown),
    )?;
    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = node.step_tick();
        // Readiness = clock-synced AND session capacity available (the SAME quantity the admission gate
        // rejects on) AND session-servicing live (S4: partition-aware — a fully directory-partitioned
        // gateway self-de-routes once its Active sessions' confirmed round-trips freeze, the shard-symmetric
        // closure of the S3 blind spot). Read on the sim thread; the probe task reads the cell lock-free.
        let local_tick = node.tick().0;
        let ready = {
            let world = node.world_mut();
            let clock_synced = world.resource::<FollowerState>().clock.is_some();
            let sessions = world.resource::<GatewaySessions>();
            let sessions_live = vd_node::health::gateway_sessions_live(
                sessions.freshest_session_confirmed(),
                local_tick,
                self_fence_grace_ticks,
            );
            vd_node::health::gateway_ready(
                clock_synced,
                sessions.len(),
                max_sessions,
                sessions_live,
            )
        };
        vd_io_prod::probe::publish_tick(&health, local_tick, ready);
        let _ = pacer.wait();
    }
    // S3 drain LINGER: the shutdown edge already set /readyz=503; linger in Terminating (default 0) so k8s
    // de-routes + in-flight requests finish before the process goes away (/healthz stays LIVE meanwhile).
    std::thread::sleep(shutdown_linger);
    // GRACEFUL DRAIN: the gateway holds no un-fsynced durable state (its R-6d outbox — empty until it has a
    // producer-less flow — is fsync-before-send; sessions re-adopt via their ResumeTicket on reconnect). A
    // clean return drops `node`/`runtime`/`control` (+ the resolver task's Arc clone), tearing down the
    // endpoint + peer writers in order (the resolver exits on the shared shutdown flag first).
    tracing::info!("gateway drained on shutdown signal — exiting cleanly");
    Ok(())
}
