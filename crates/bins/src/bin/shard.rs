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
    vd_bins::init_tracing();
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
    let declared_coord = match env.string("VD_OWN_COORD").ok().filter(|s| !s.is_empty()) {
        Some(s) => Some(
            vd_core::realm_coord::RealmCoord::from_path(
                vd_core::realm_path::RealmPath::from_env_string(&s).map_err(|e| e.to_string())?,
            )
            .ok_or("VD_OWN_COORD is an empty lineage — refusing to boot")?,
        ),
        None => None,
    };
    // The profile depends only on the LEAF level's kind, which is the same whether the lineage was declared
    // or is about to be derived below (both end at `own_realm`), so it can be settled here — before the
    // forest exists — without prejudging the chain above it.
    let profile = vd_sim::capability::profile_for(
        declared_coord
            .clone()
            .unwrap_or_else(|| vd_sim::stub::StubConfig::root_coord(own_realm))
            .profile_kind(),
    )
    .map_err(|e| e.to_string())?;
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
    vd_bins::validate_tick_pair(tick_hz, tick_dt)?;
    // NODE-PER-REALM (task #149): a realm-shard hosts EXACTLY ONE realm. `own_realm` (its `RealmId`),
    // `own_coord` (its un-collapsed lineage), and its derived `ShardProfile` were all resolved ABOVE (before
    // build_app) from `VD_OWN_COORD` or, absent it, `VD_REALM_KIND`/`VD_REALM_SEED` (ABSENT/empty ⇒
    // `System(seed)` — the pre-NODE-PER-REALM default, so every legacy shard boot stays byte-identical).
    // CO-HOSTING (`VD_HELD_REALMS`): a shard may host its own realm PLUS deeper CHILD realms it co-hosts.
    // NO launcher produces it since the co-hosting shapes retired into NODE-PER-REALM (D-WORLD-6 — the
    // parse keeps unit coverage so it cannot rot silently). ABSENT ⇒ single-realm ({own realm} — the
    // byte-identical default: each shard holds exactly its own realm). A malformed value fails LOUD at boot
    // (a co-hosting misconfig must never silently degrade to single-realm and re-open the orphan gap). The
    // own realm is always included.
    let held_realms =
        vd_bins::parse_held_realms(&env.string("VD_HELD_REALMS").unwrap_or_default(), own_realm)?;
    // `VD_UNIVERSE_SEED` (default 0) is the ONE seed every shard shares — read ONCE here (reused for the
    // frame lookup below AND the seed-neighbourhood plant further down).
    let universe_seed: u64 = env.parse_or("VD_UNIVERSE_SEED", 0)?;
    // The containment forest + the moving-child roster, from the ONE world (SL5). There is no scale to
    // resolve: this shard and its gateway build the same universe because there is only one, and the
    // interest band is measured against the speed this cluster actually flies at and the tick it
    // actually runs.
    let (seed_regions, moving) = vd_bins::boot_regions_and_movers(
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
    // deterministic worldgen fact the detector + the arrival placement (`place_arriving_pose`) use, never a per-kind inline (a
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
    let hosted_realm = own_realm;
    // C-6b — the SEED-DERIVED containment boot (task #135). The shard computes its realm-region
    // NEIGHBOURHOOD closed-form from the shared universe seed (`realm_neighbourhood_for`: own realm +
    // ancestor chain to the ambient root + owned children — NEVER siblings, HR1 replicated-by-construction,
    // no inter-shard bytes) and plants it, so the containment detector is LIVE from boot (no longer inert).
    // `universe_seed` was read once above; a per-shard forest that fails `guard_regions_nest` (two roots, a
    // dangling parent, a cycle, count > MAX_REGIONS) is a CODE bug in the generator — fail LOUD at boot
    // (Display carries the actionable guidance for `kubectl logs`), never a silent detector no-op.
    // There is NO boundary-file override any more (SL5): the authored playground forest a shard could once
    // load in place of THE world let a cluster simulate geometry the world does not contain, and the gates
    // that leaned on it proved nothing about the game as shipped. THE world's own shells are the only
    // crossing boundaries a shard ever arms.
    let regions = seed_regions;
    // This line used to print the SCALE this shard booted, and reading it across a live cluster is
    // how the two-worlds defect was caught: the orchestrator said one thing and its gateway another.
    // There is no scale to print now. The seed is, because one world generated from one seed is
    // exactly what has to be true, and it is the thing worth being able to compare across processes.
    tracing::info!(
        count = regions.len(),
        realm = %hosted_realm,
        seed = universe_seed,
        "planting the containment forest for THE world — the re-home detector is LIVE",
    );
    // THE SHARD'S LINEAGE, and where it comes from. `VD_OWN_COORD` when the launcher set one (the demand
    // spawner does); otherwise DERIVED from the very forest just planted, by walking the parent pointers of
    // this shard's own region up to the ambient root.
    //
    // It used to fall back to a ROOT-SHAPED coord — a one-level lineage claiming this realm has no parent —
    // and every shipped launcher takes that path, so every statically launched shard has been running with
    // a lineage that says it is the Universe. That is not a harmless label: leaving a realm now reads the
    // shard's own lineage to decide who to hand the occupant to, so a shard that believes it is a root
    // hands people to the ambient root instead of to the star system twenty metres away. (`outward_dest`
    // has been papering over exactly this by consulting the region row when the coord had nothing; with the
    // lineage derived from the same forest, the two sources agree by construction rather than by luck.)
    //
    // Deriving beats demanding an env var: the forest is the one source of truth about who contains whom,
    // and a lineage handed in separately can drift from it — which is precisely what the fence below
    // refuses. A realm absent from its own forest keeps the root shape, and the fence then has nothing to
    // object to (nothing claims it has a parent).
    let own_coord = match declared_coord {
        Some(declared) => declared,
        None => vd_core::worldgen::coord_of_realm(&regions, own_realm)
            .unwrap_or_else(|| vd_sim::stub::StubConfig::root_coord(own_realm)),
    };
    // THE LINEAGE FENCE — the owner's rule: we cannot boot a realm without its parents all the way to the
    // root. A shard whose world gives it a parent while its lineage claims none would MISROUTE every
    // occupant that leaves it, silently, with nothing crashing; refusing to start beats that. Reachable
    // only via an explicitly declared `VD_OWN_COORD` that disagrees with the forest, since a derived
    // lineage cannot disagree with the forest it was derived from.
    vd_core::geometry::guard_lineage_reaches_root(
        &regions,
        own_realm,
        own_coord.parent().map(|p| p.lowered()),
    )
    .map_err(|e| format!("{e} — refusing to boot"))?;
    // How long this shard keeps speaking for an occupant it has handed away — counting itself occupied,
    // so its one-bit ChildLive heartbeat keeps beating (no pose crosses; the per-occupant position
    // up-relay is DELETED, Step 5 slice D) — when the take-over never lands. Handed down by the
    // orchestrator that launched it, because only there are both budgets it depends on in scope: how long
    // a hand-off may take, and how fast a realm can be reclaimed. ABSENT ⇒ 0 ⇒ the shard lets go the
    // instant it is told to, exactly as before. Logged so a real run SHOWS which of the two it is running.
    let handoff_hold_ttl_ticks: u32 = env.parse_or("VD_HANDOFF_HOLD_TICKS", 0)?;
    tracing::info!(
        ticks = handoff_hold_ttl_ticks,
        armed = handoff_hold_ttl_ticks != 0,
        "hand-off hold budget resolved — how long this shard speaks for an occupant it has handed away",
    );
    // D-WORLD-2 cure — the crossing-request ttl + re-drive budget, handed down by the launcher
    // (`crossing_redrive_env`: derived from the deployment's SAGA deadlines, the only place both
    // budgets are in scope — a shard re-deriving from a default it happens to know would be guessing).
    // ABSENT ⇒ 0 ⇒ disarmed, exactly the pre-cure posture: an unresolved-dest crossing then strands
    // its entity's latch forever, which is why every shipped launcher sets the pair. Logged so a real
    // run SHOWS which posture it is flying.
    let request_ttl_ticks: u32 = env.parse_or("VD_CROSSING_TTL_TICKS", 0)?;
    let crossing_redrive_budget: u32 = env.parse_or("VD_CROSSING_REDRIVE_BUDGET", 0)?;
    tracing::info!(
        ttl = request_ttl_ticks,
        budget = crossing_redrive_budget,
        armed = request_ttl_ticks != 0,
        "crossing re-drive posture resolved — how a strand-latched crossing self-heals (D-WORLD-2)",
    );
    // Step 5 slice B — THE HOLD IS MANDATORY ON AN AoI-ARMED SHARD (the boot fence, like the lineage
    // fence above): armed interest bands mean live hand-offs, and a hand-off ledger with a zero
    // budget lets a source fall silent about a departing occupant while its parent still counts on
    // it — the exact gap the ledger closes. Every launcher of an armed world must state the budget
    // (the demand orchestrator derives it in `handoff_hold_anchor`); refusing to boot beats running
    // half a ledger. Walk/static worlds carry inert bands and boot exactly as before.
    if regions.iter().any(|r| r.aoi.spin_up_r_m() > 0.0) && handoff_hold_ttl_ticks == 0 {
        return Err(format!(
            "this world arms the interest bands (demand scale) on realm {own_realm} but \
             VD_HANDOFF_HOLD_TICKS is 0/unset — an AoI-armed shard must carry the hand-off hold \
             budget its launcher derives (see vd_bins::handoff_hold_anchor) — refusing to boot"
        )
        .into());
    }
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
            // D-WORLD-2 — the crossing-latch ttl + re-drive budget (resolved from the launcher-derived
            // env above): a delivered-but-unresolved crossing re-drives a bounded number of times, then
            // takes the LOCAL pre-CAS abort that clears the strand latch.
            request_ttl_ticks,
            crossing_redrive_budget,
            handoff_hold_ttl_ticks,
            // THIS REALM'S OWN stored poses — realm-local, in this realm's frame. EMPTY at boot, and
            // deliberately so: the only pose store that exists today is a cluster-wide env var holding
            // UNIVERSE-ABSOLUTE positions, which is not something a realm can hold. The shard used to load
            // it anyway and "convert" by relabelling, planting a player stored above a planet next to that
            // planet's star. A login's position now arrives already converted, in `AttachSession`, from the
            // gateway — the only party that holds the whole forest and can therefore do the subtraction.
            // The P7 durable PER-REALM store fills this map, whose contents are realm-local by construction.
            spawn_poses: std::collections::BTreeMap::new(),
        },
    );
    // RLM 5f RG-2: a DEMAND-spawned shard (it carries the spawner's VD_INCARNATION_COOKIE) greets its booked
    // peers so they learn its return connection — its spawn-minted NodeId is reachable without any pre-booked
    // address (the cloud-portable property). A STATIC shard carries no cookie ⇒ None ⇒ no greeting ⇒
    // byte-identical. Inserted via the still-live `world` borrow before the RealmRegions plant below.
    if let Some(presence) = vd_bins::resolve_presence(&env, tick_dt)? {
        world.insert_resource(presence);
    }
    // The BOOT FENCE (C-5): topology + WORST-INSTANT geometry validation BEFORE the infallible
    // `RealmRegions::new`, so a malformed forest fails LOUD here rather than degrading to the
    // detector's rootless no-op. The reach map is the boot's statement of how far each child's motion
    // can carry it — a mover judged at its APOAPSIS, a static child exactly at its authored offset
    // (the placement arc S4: the fence used to read a mover's zeroed centre and go size-only).
    vd_core::geometry::guard_regions_nest(
        &regions,
        vd_sim::stub::MAX_REGIONS,
        // The reach roster derives from THE world for EVERY parent this neighbourhood names —
        // never from this shard's own `moving` subset, whose gaps judged a non-hosted mover at
        // its zeroed centre (the batch-review zero-reach hole; one forest, one verdict, on
        // every shard).
        &vd_bins::child_reaches(
            universe_seed,
            &regions,
            move_speed * time_multiplier,
            tick_dt,
        ),
    )
    .map_err(|e| {
        format!("malformed realm-region forest for {hosted_realm}: {e} — refusing to boot")
    })?;
    // THE GENERATOR VISIBILITY CHECK (owner ruling 2026-08-15, items 5/10 + the re-solve addendum):
    // no body two or more levels deep may be visible from just outside its ancestor — the two-level
    // bound (SL3/SL7: a not-running realm is drawable ONLY as its parent's one-level placement
    // marker) turned into a boot refusal, the same fail-loud pattern as the nest fence above. THE
    // world satisfies it BY CONSTRUCTION (the shell solve `galaxy_shell_r_m`); a world whose numbers
    // stop satisfying it must never boot a shard that would owe unauthorable pixels.
    // THE LOOK HORIZON's boot fence (look_horizon.md §3.3.2/§3.3.4, slice 2 — replaces the
    // boolean two-level guard): MEASURE how many levels each body's picture must travel and
    // refuse a world whose climb exceeds what the look carrier can carry. A refusal is a
    // measurement; a wrong pixel is not (the owner's Q3 ruling: the arity stays 2 until the
    // near-real-scale re-solve measures otherwise).
    vd_physics::worldgen::guard_visibility_climb_bounded(
        universe_seed,
        &vd_bins::process_world_config(move_speed * time_multiplier, tick_dt),
        vd_wire::session_flow::LOOK_CARRIER_ARITY,
    )
    .map_err(|e| {
        format!(
            "THE world's measured visibility climb exceeds the look carrier: {e} — refusing to boot"
        )
    })?;
    // THE WINDOW LANE's marker roster (Slice A, docs/design/window_lane.md §2.2/§2.8): per DIRECT
    // child this shard parents, the pre-encoded TAG_LUMA bag drawn from the child's own generation
    // stream — plumbed at BOOT the way the region forest and the motion roster are (the boot/config
    // path; the sim crate receives two plain scalars and keeps no vd-physics edge). Computed
    // BEFORE the forest is moved into the resource below. Empty wherever no direct child carries
    // a draw — such a child still states an extent-only point of light (look_horizon.md slice 1,
    // the presence floor), built by the sim through the one marker-bag codec.
    let child_luma = vd_bins::child_luma_draws(
        universe_seed,
        move_speed * time_multiplier,
        tick_dt,
        &regions,
        &held_realms,
    );
    *node.world_mut().resource_mut::<vd_sim::stub::ChildLuma>() =
        vd_sim::stub::ChildLuma(child_luma);
    *node
        .world_mut()
        .resource_mut::<vd_sim::stub::RealmRegions>() = vd_sim::stub::RealmRegions::new(regions)
        .with_moving_children(vd_physics::motion::kepler_motion_fns(moving));
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
