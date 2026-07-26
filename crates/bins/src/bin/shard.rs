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
    // RLM Step 5a: the shard's OWN lineage coord + its DERIVED ShardProfile, resolved BEFORE build_app.
    // `VD_OWN_COORD` (the real spawner's UN-collapsed RealmPath, Step 5c) preserves a Galaxy/Universe level
    // that `RealmId` would collapse — so a Galaxy shard keeps `signal_relay` (uncorners P9 Signals). ABSENT
    // ⇒ the root coord from VD_REALM_KIND/SEED (byte-identical for every existing rig until 5c emits
    // VD_OWN_COORD). The profile is DERIVED from the coord (HR3: the ONE kind→profile match lives in
    // `profile_kind`, never a feature/reconciler branch). The `StubShard → Shard(profile)` swap is
    // CAPABILITY-INERT at P1–P3 (no system reads the profile caps yet) — proven byte-identical for every
    // `ProfileKind` by the vd-node 5a inertness gate.
    let realm_seed: u64 = env.parse("VD_REALM_SEED")?;
    let realm_kind = env.string("VD_REALM_KIND").unwrap_or_default();
    let own_realm = vd_bins::realm_from_kind_seed(&realm_kind, realm_seed)?;
    let own_coord = match env.string("VD_OWN_COORD").ok().filter(|s| !s.is_empty()) {
        Some(s) => vd_core::realm_coord::RealmCoord::from_path(
            vd_core::realm_path::RealmPath::from_env_string(&s).map_err(|e| e.to_string())?,
        )
        .ok_or("VD_OWN_COORD is an empty lineage — refusing to boot")?,
        None => vd_sim::stub::StubConfig::root_coord(own_realm),
    };
    let profile =
        vd_sim::capability::profile_for(own_coord.profile_kind()).map_err(|e| e.to_string())?;
    let mut node = build_app(
        NodeConfig {
            node_id: local,
            kind: NodeKind::Shard(profile),
        },
        transport,
    );
    let (world, schedule) = node.parts_mut();
    register_clock_follower(world, schedule);
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
    // The realm's SUBJECTIVE time factor (D-45(a)): dilates OCCUPANT movement inside this realm (never the
    // celestial orbit). `VD_REALM_TIME_MULTIPLIER` override / `VD_TIME_MULTIPLIER` global / 1.0 default.
    let time_multiplier = vd_bins::resolve_time_multiplier(&env)?;
    // RLM 5f-3b: the per-account STORED spawn poses (the `VD_SPAWN_POSES` STAND-IN for the P7 durable pose
    // store). ABSENT ⇒ an EMPTY map ⇒ every login births origin-at-rest (byte-identical). A login then
    // loads the stored pose SHARD-SIDE (keyed by the account in the AttachSession arm) and admits at it.
    let spawn_poses = vd_bins::resolve_spawn_poses(&env)?;
    vd_bins::validate_tick_pair(tick_hz, tick_dt)?;
    // NODE-PER-REALM (task #149): a realm-shard hosts EXACTLY ONE realm. `own_realm` (its `RealmId`),
    // `own_coord` (its un-collapsed lineage), and its derived `ShardProfile` were all resolved ABOVE (before
    // build_app) from `VD_OWN_COORD` or, absent it, `VD_REALM_KIND`/`VD_REALM_SEED` (ABSENT/empty ⇒
    // `System(seed)` — the pre-NODE-PER-REALM default, so every legacy shard boot stays byte-identical).
    // CO-HOSTING (the un-hosted-child cure, KEPT for the --triple LEGACY shape): a shard may host its own
    // realm PLUS deeper CHILD realms it co-hosts (`VD_HELD_REALMS`, set by the --triple launcher). ABSENT ⇒
    // single-realm ({own realm} — the byte-identical default, and the NODE-PER-REALM Forest case: each shard
    // holds exactly its own realm). A malformed value fails LOUD at boot (a co-hosting misconfig must never
    // silently degrade to single-realm and re-open the orphan gap). The own realm is always included.
    let held_realms =
        vd_bins::parse_held_realms(&env.string("VD_HELD_REALMS").unwrap_or_default(), own_realm)?;
    // `VD_UNIVERSE_SEED` (default 0) is the ONE seed every shard shares — read ONCE here (reused for the
    // frame lookup below AND the seed-neighbourhood plant further down).
    let universe_seed: u64 = env.parse_or("VD_UNIVERSE_SEED", 0)?;
    // FA-5 (D-45(a)): the world SCALE (`VD_UNIVERSE_SCALE`, ABSENT ⇒ `Walk` = byte-identical). `Walk` ⇒
    // the seed neighbourhood + an EMPTY mover roster; `Visual` ⇒ the single-system forest whose planets
    // ORBIT. Build the containment forest + the moving-child roster from the SAME `(scale, seed, config)`,
    // so the authored own-frame and the moving planets can never derive from different elements.
    let scale = vd_bins::resolve_universe_scale(&env)?;
    // RLM 5f-4: the WalkDemand AoI band is measured against the LIVE occupant speed the sim integrates
    // (`move_speed · time_multiplier`) at the LIVE tick dt — closing the M-2 two-home owe. `Walk`/`Visual`
    // ignore both.
    let (seed_regions, moving) = vd_bins::boot_regions_and_movers(
        scale,
        universe_seed,
        &held_realms,
        own_realm,
        move_speed * time_multiplier,
        tick_dt,
    );
    // The shard's LOCAL authority frame = its realm's canonical frame from the seed forest (NODE-PER-REALM:
    // a Planet shard is `PlanetCentered`, a Station `StationLocal`, an Area `AreaLocal{planet,area}` — an Area
    // REQUIRES its Planet parent, which the forest region carries). Looked up from `seed_regions` — the
    // SAME scale-built forest the containment detector uses — so the frame + parent are the SAME
    // deterministic worldgen fact the detector + `rebind_pose_to_dest` use, never a per-kind inline (a
    // `System(seed)` shard resolves to `SystemSpace{seed}`, byte-identical to the old hardcoded frame, on
    // both scales). If the realm is absent from the forest (an unknown seed) fall back to its parentless
    // canonical frame (an Area then has no frame — rejected LOUD, never a silent wrong frame).
    let own_frame = seed_regions
        .iter()
        .find(|r| r.realm == own_realm)
        .map(|r| r.frame)
        .or_else(|| vd_core::pose::frame_for_realm(own_realm, None))
        .ok_or_else(|| {
            format!(
                "realm {own_realm} has no canonical frame (an Area realm needs a Planet parent, absent from \
                 the seed forest for this seed) — refusing to boot"
            )
        })?;
    register_stub_shard(
        world,
        schedule,
        StubConfig {
            realm: own_realm,
            // RLM Step 5a: the shard's REAL lineage coord (resolved above from `VD_OWN_COORD`, or the root
            // coord from VD_REALM_KIND/SEED when absent). Byte-identical to the old `root_coord(own_realm)`
            // for every existing rig (they set no `VD_OWN_COORD`); the spawner (5c) feeds the full lineage so
            // `evaluate_realm_aoi` can name this shard's TRUE children, not just root-level ones.
            own_coord: own_coord.clone(),
            // RLM 5f: the measured process-boot p99 (ticks) feeds `evaluate_realm_aoi`'s PREDICTIVE spin-up
            // horizon (a child is demanded `boot_ticks_p99*dt` before an occupant reaches it, so its shard is
            // warm on arrival). Default 0 ⇒ no look-ahead (byte-identical); 5f-4 bakes the measured value.
            boot_ticks_p99: env.parse_or("VD_BOOT_TICKS_P99", 0)?,
            held_realms: held_realms.clone(),
            frame: own_frame,
            move_speed_mps: move_speed,
            tick_dt_s: tick_dt,
            time_multiplier,
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
            // RLM 5f-3b — the per-account STORED spawn poses (the `VD_SPAWN_POSES` stand-in; the P7 durable
            // pose store swaps in behind this SAME map). ABSENT ⇒ empty ⇒ origin-at-rest (byte-identical).
            spawn_poses,
        },
    );
    // C-6b — the SEED-DERIVED containment boot (task #135). The shard computes its realm-region
    // NEIGHBOURHOOD closed-form from the shared universe seed (`realm_neighbourhood_for`: own realm +
    // ancestor chain to the ambient root + owned children — NEVER siblings, HR1 replicated-by-construction,
    // no inter-shard bytes) and plants it, so the containment detector is LIVE from boot (no longer inert).
    // `universe_seed` was read once above; a per-shard forest that fails `guard_regions_nest` (two roots, a
    // dangling parent, a cycle, count > MAX_REGIONS) is a CODE bug in the generator — fail LOUD at boot
    // (Display carries the actionable guidance for `kubectl logs`), never a silent detector no-op.
    let hosted_realm = own_realm;
    // `seed_regions` (built above from the scale) is the containment forest this shard EVALUATES against:
    // Walk ⇒ the seed NEIGHBOURHOOD (the UNION of every held realm's neighbourhood — a co-hosting shard must
    // see the deeper child regions, a Planet's Area being a GRANDCHILD, so the LOCAL re-home short-circuit
    // can fire; single-realm ⇒ `realm_neighbourhood_for(hosted)` exactly, byte-identical); Visual ⇒ the FA-5
    // single-system forest. `moving` is EMPTY on Walk, the orbiting planets on Visual.
    // `VD_REALM_BOUNDARIES` OVERRIDE (kept for the dual-cluster / render-crossing PLAYGROUND smokes): an
    // authored `boundaries.json` (a `Vec<RealmBoundary>`, SINGLE-SOURCED with the client's `--realm-boxes`)
    // REPLACES the scale forest with a born-inside child crossing shell — a self-contained Walk-scale forest,
    // so its mover roster is EMPTY (the override never combines with the Visual orbiting planets). A
    // malformed file / a boundary for a realm this shard does NOT host fails LOUD. ABSENT ⇒ the scale forest.
    let (regions, moving) = if let Some(boundaries) =
        vd_bins::resolve_realm_boundaries(&env, hosted_realm).map_err(|e| e.to_string())?
    {
        tracing::info!(
            count = boundaries.len(),
            realm = %hosted_realm,
            "planting VD_REALM_BOUNDARIES OVERRIDE — the authored playground crossing forest is ARMED",
        );
        (
            vd_bins::override_regions_for_boundaries(
                &boundaries,
                hosted_realm,
                move_speed,
                tick_dt,
            ),
            std::collections::BTreeMap::new(),
        )
    } else {
        tracing::info!(
            count = seed_regions.len(),
            realm = %hosted_realm,
            seed = universe_seed,
            scale = ?scale,
            "planting the scale-derived containment forest — the re-home detector is LIVE",
        );
        (seed_regions, moving)
    };
    // The BOOT FENCE (C-5): pure-topology validation BEFORE the infallible `RealmRegions::new`, so a
    // malformed forest fails LOUD here rather than degrading to the detector's rootless no-op.
    vd_core::geometry::guard_regions_nest(&regions, vd_sim::stub::MAX_REGIONS).map_err(|e| {
        format!("malformed realm-region forest for {hosted_realm}: {e} — refusing to boot")
    })?;
    *node
        .world_mut()
        .resource_mut::<vd_sim::stub::RealmRegions>() =
        vd_sim::stub::RealmRegions::new(regions).with_moving_children(moving);
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
            // RLM 5c: the realm spawner sets VD_INCARNATION_COOKIE (a real forked shard); ABSENT for the
            // in-process rigs (no /whoami mounted). Echoed verbatim on /whoami for the Step-5e pid-reuse
            // guard — we re-emit the exact string the parent minted (no decode/re-encode round-trip).
            env.string("VD_INCARNATION_COOKIE")
                .ok()
                .filter(|s| !s.is_empty()),
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
