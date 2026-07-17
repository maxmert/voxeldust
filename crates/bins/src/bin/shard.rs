//! THE shard binary (HR3: one binary; shard types are `ShardProfile` configs —
//! P1 runs the stub profile). A thin shell: config → mesh → build_app → register
//! → tick loop. All logic lives in the tested libs.

use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, build_app};
use vd_node::follower::{FollowerState, register_clock_follower};
use vd_sim::capability::NodeKind;
use vd_sim::stub::{RealmAuthority, RealmConfirmedAt, StubConfig, register_stub_shard};

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
    // the mesh WITH the sink (durable-before-send gate LIVE), replay retained rows BEFORE build_app. `control`
    // (Arc-wrapped below for the auto-resolver) + `runtime` stay bound for the whole tick loop (dropping either
    // tears down the endpoint / peer-writers).
    let (transport, control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
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
    // S4b (cloud manifest guard): VD_TICK_DT and VD_TICK_HZ are INDEPENDENT env vars; a cloud ConfigMap that
    // retunes one but not the other would silently integrate the avatar's motion at the wrong step (no crash).
    // The compile-time TICK-PAIR assert guards only the DEV const — cross-check the env-supplied pair here.
    let tick_hz: u32 = env.parse("VD_TICK_HZ")?;
    let tick_dt: f64 = env.parse("VD_TICK_DT")?;
    let move_speed: f64 = env.parse("VD_SPEED")?;
    vd_bins::validate_tick_pair(tick_hz, tick_dt)?;
    register_stub_shard(
        world,
        schedule,
        StubConfig {
            realm: vd_core::pose::RealmId::System(realm_seed),
            frame: vd_core::pose::FrameRef::SystemSpace {
                system_seed: realm_seed,
            },
            move_speed_mps: move_speed,
            tick_dt_s: tick_dt,
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
            // Slice 3e — the geometric transfer-trigger tuning. INERT in production through P3 (the
            // trigger's `RealmBoundaries` registry is empty ⇒ `evaluate_realm_boundaries` early-returns,
            // behaviour-identical); the tuning is validated at `register_stub_shard` regardless.
            boundary: vd_core::geometry::BoundaryTuning::DEFAULT,
            // Slice 3d — the crossing-latch TTL fallback. INERT (0): the POSITIVE saga-terminal clear is
            // the sole driver until the 3f abort/TTL egress lands.
            request_ttl_ticks: 0,
        },
    );
    // C-6b — the SEED-DERIVED containment boot (task #135). The shard computes its realm-region
    // NEIGHBOURHOOD closed-form from the shared universe seed (`realm_neighbourhood_for`: own realm +
    // ancestor chain to the ambient root + owned children — NEVER siblings, HR1 replicated-by-construction,
    // no inter-shard bytes) and plants it, so the containment detector is LIVE from boot (no longer inert).
    // `VD_UNIVERSE_SEED` (default 0) is the ONE seed every shard shares; a per-shard forest that fails
    // `guard_regions_nest` (two roots, a dangling parent, a cycle, count > MAX_REGIONS) is a CODE bug in the
    // generator — fail LOUD at boot (Display carries the actionable guidance for `kubectl logs`), never a
    // silent detector no-op on a malformed forest.
    let hosted_realm = vd_core::pose::RealmId::System(realm_seed);
    let universe_seed: u64 = env.parse_or("VD_UNIVERSE_SEED", 0)?;
    let regions = vd_core::worldgen::realm_neighbourhood_for(universe_seed, hosted_realm);
    // `VD_REALM_BOUNDARIES` OVERRIDE (kept for the dual-cluster / render-crossing PLAYGROUND smokes): an
    // authored `boundaries.json` (a `Vec<RealmBoundary>`, SINGLE-SOURCED with the client's `--realm-boxes`)
    // REPLACES the seed neighbourhood with a born-inside child crossing shell, so the process-tier smoke can
    // prove a DIRECT source→dest re-home without standing up a Galaxy shard. ABSENT ⇒ the seed forest (the
    // production default). A malformed file / a boundary for a realm this shard does NOT host fails LOUD.
    let regions = if let Some(boundaries) =
        vd_bins::resolve_realm_boundaries(&env, hosted_realm).map_err(|e| e.to_string())?
    {
        tracing::info!(
            count = boundaries.len(),
            realm = %hosted_realm,
            "planting VD_REALM_BOUNDARIES OVERRIDE — the authored playground crossing forest is ARMED",
        );
        vd_bins::override_regions_for_boundaries(&boundaries, hosted_realm, move_speed, tick_dt)
    } else {
        tracing::info!(
            count = regions.len(),
            realm = %hosted_realm,
            seed = universe_seed,
            "planting the SEED-DERIVED containment neighbourhood — the re-home detector is LIVE",
        );
        regions
    };
    // The BOOT FENCE (C-5): pure-topology validation BEFORE the infallible `RealmRegions::new`, so a
    // malformed forest fails LOUD here rather than degrading to the detector's rootless no-op.
    vd_core::geometry::guard_regions_nest(&regions, vd_sim::stub::MAX_REGIONS).map_err(|e| {
        format!("malformed realm-region forest for {hosted_realm}: {e} — refusing to boot")
    })?;
    *node
        .world_mut()
        .resource_mut::<vd_sim::stub::RealmRegions>() = vd_sim::stub::RealmRegions::new(regions);
    let mut pacer = TickPacer::new(tick_hz);
    // Cloud-ready k3d Slice 3: the k8s probe surface. A lock-free health cell the tick loop publishes (its
    // heartbeat + THIS shard's readiness) and the /healthz+/readyz HTTP task reads. Slice 1: the SIGTERM flag
    // — now the with-health variant, so the shutdown EDGE de-routes (NotReady) instantly + marks `draining`
    // (keeps /healthz LIVE through the drain, never a SIGKILL mid-fsync). The probe server binds iff
    // VD_PROBE_ADDR is set (the in-process rigs stay byte-identical until they opt in).
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
        );
    }
    // Cloud reschedule re-plumb: the peer-addr auto-resolver (the production caller of update_peer_addr) —
    // spawned iff VD_PEER_HOSTS is set (cloud deploy), a no-op in-process. Shares the tick loop's shutdown flag,
    // so it drains cleanly on SIGTERM. `control` stays bound (Arc) so the endpoint lives for the whole run.
    let control = std::sync::Arc::new(control);
    vd_bins::spawn_peer_resolver_if_configured(
        &env,
        runtime.handle(),
        std::sync::Arc::clone(&control),
        std::sync::Arc::clone(&shutdown),
    )?;
    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = node.step_tick();
        // Readiness = clock-synced AND realm-authority-ready (confirmed-fresh under active D-3 so a
        // partitioned shard self-de-routes via the frozen `RealmConfirmedAt`; authority-held when inert).
        // Read on the sim thread; the probe HTTP task reads the published cell lock-free (HR1: own World only).
        let local_tick = node.tick().0;
        let ready = {
            let world = node.world_mut();
            let clock_synced = world.resource::<FollowerState>().clock.is_some();
            let held = world.resource::<RealmAuthority>().0.is_some();
            let confirmed = world.resource::<RealmConfirmedAt>().0.0;
            vd_node::health::shard_ready(
                clock_synced,
                vd_node::health::shard_authority_ready(
                    held,
                    local_tick,
                    confirmed,
                    self_fence_grace_ticks,
                ),
            )
        };
        vd_io_prod::probe::publish_tick(&health, local_tick, ready);
        let _ = pacer.wait();
    }
    // S3 drain LINGER (default 0): the shutdown edge already set /readyz=503; linger in Terminating so k8s
    // de-routes before exit (/healthz stays LIVE meanwhile).
    std::thread::sleep(shutdown_linger);
    // GRACEFUL DRAIN: the shard holds no un-fsynced durable state — its R-6d outbox is fsync-BEFORE-send and
    // any in-flight mesh send is recovery-covered by the outbox replay on restart. So a clean return
    // suffices: `node`, `runtime`, and `control` (+ the resolver task's Arc clone) drop, tearing down the
    // endpoint + peer writers in order (the resolver task exits on the shared shutdown flag first).
    tracing::info!("shard drained on shutdown signal — exiting cleanly");
    Ok(())
}
