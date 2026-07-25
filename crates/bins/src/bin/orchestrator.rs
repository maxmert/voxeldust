//! The orchestrator binary: directory + universe clock + the read-only admin
//! endpoint (the 2am `curl`). The admin snapshot is republished after every tick
//! through a lock-free cell — the HTTP task never touches the sim thread.

use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;
use vd_io_prod::admin::{MeshMetrics, SnapshotSource, admin_router};
use vd_io_prod::mesh::{MeshConfig, spawn_mesh};
use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::store::{RedbStore, StoreTuning};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, TickPrologue, build_app};
use vd_node::orchestrator::{OrchestratorConfig, admin_snapshot, register_orchestrator_with_store};
// D-6 D-delta (feature `store-test-hooks`, ABSENT from release): the crash-proof sentinel inject.
#[cfg(feature = "store-test-hooks")]
use vd_bins::SHARD;
#[cfg(feature = "store-test-hooks")]
use vd_core::pose::RealmId;
#[cfg(feature = "store-test-hooks")]
use vd_core::{Fence, UniverseTick};
#[cfg(feature = "store-test-hooks")]
use vd_node::orchestrator::DirectoryRes;
use vd_sim::capability::NodeKind;
use vd_wire::admin::AdminSnapshot;
#[cfg(feature = "store-test-hooks")]
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

struct Published(Arc<ArcSwap<AdminSnapshot>>);

impl SnapshotSource for Published {
    fn snapshot(&self) -> AdminSnapshot {
        self.0.load().as_ref().clone()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let env = EnvConfig::from_process_env();
    // Cloud-ready k3d Slice 2: run the footgun preflight FIRST — before ANY durable action (the M3 boot-counter
    // increment inside resolve_process_incarnation, the store open). DevTest = a no-op passthrough; Cloud fails
    // LOUD on an ephemeral escape / a manual VD_PROCESS_INCARNATION / a missing-or-under-temp VD_STORE_DURABLE_
    // ROOT. The orchestrator holds no auth key ⇒ dev_pubkey = None. `profile` threads into resolve_d3 below.
    // Stringify so main's default `{:?}` print surfaces the actionable Display guidance (e.g. "cloud profile
    // forbids VD_STORE_EPHEMERAL_OK …") in the pod's `kubectl logs`, never a bare Debug variant dump — the
    // SAME operator-facing posture as `check_durable_path` below.
    let profile =
        vd_io_prod::boot::enforce_cloud_preflight(&env, None).map_err(|e| e.to_string())?;
    let local = env.node_id("VD_NODE_ID")?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let trust = ClusterTrust::from_der_dir(std::path::Path::new(&env.string("VD_TRUST_DIR")?))?;
    // Named so R-4c can cross-validate the SAME transport redial backoff the mesh runs with (below).
    let mesh_cfg = MeshConfig::new(
        local,
        env.parse("VD_BIND")?,
        env.peer_book("VD_PEERS")?,
        env.parse("VD_OUTBOUND_CAP")?,
        // R-2b: per-process incarnation stamped on reliable frames (default 0; R-6 durable counter).
        // R-6a: the DURABLE MONOTONE process incarnation (VD_PROCESS_INCARNATION explicit wins for dev/test;
        // else the VD_BOOT_STATE_DIR boot-counter that survives a CrashLoop / clock rewind).
        vd_bins::resolve_process_incarnation(&env)?,
    );
    // R-4d M4: RETAIN the MeshControl (was discarded) so the admin `/metrics` endpoint can read
    // the live mesh reliability counters (`MeshControl::stats()` — a pure atomic load off the hot path).
    // R-6d3a: `spawn_mesh` takes an optional durable outbox; the orchestrator wires `None` — its saga flows
    // re-drive via `scan_deadlines`, so they need no transport outbox (the outbox is for the producer-less
    // shard one-shots, wired at R-6d3b).
    let (transport, control) = spawn_mesh(runtime.handle(), &trust, &mesh_cfg, None)?;
    let control = Arc::new(control);
    let mut node = build_app(
        NodeConfig {
            node_id: local,
            kind: NodeKind::Orchestrator,
        },
        transport,
    );
    let (world, schedule) = node.parts_mut();
    // Slice 2a saga deadline budget: defaults to the documented dev values, overridable per
    // deployment per the false-timeout formula (a present-but-bad value fails loud). `validate`
    // rejects a 0 / abort-below-redrive override at boot — LOUD config error, never a silent
    // production abort storm (audit wf_75a8d57d).
    let saga = vd_sim::saga::SagaTuning {
        redrive_deadline_ticks: env.parse_or(
            "VD_SAGA_REDRIVE_DEADLINE",
            vd_sim::saga::DEFAULT_REDRIVE_DEADLINE_TICKS,
        )?,
        abort_deadline_ticks: env.parse_or(
            "VD_SAGA_ABORT_DEADLINE",
            vd_sim::saga::DEFAULT_ABORT_DEADLINE_TICKS,
        )?,
    };
    saga.validate()?;
    // D-3 lease-liveness config (cloud-ready k3d Slice 2): ONE resolution (HR3) for this `profile`. DevTest =
    // the INERT defaults (byte-identical to before — dev/test/in-process rigs unbroken); Cloud = the DERIVED
    // active split-brain set (`DirectoryTuning::cloud` + `LivenessTuning::cloud`, whose window dominates the
    // confirmed-dead run) + the inert-D-3 rejection. `resolve_d3` runs `directory.validate` +
    // `liveness.validate` + `validate_against` (the R-4c cross-check, against the SAME DEFAULT redial backoff
    // the mesh dials with — now shared consts, so `mesh_cfg`'s values equal them) + `validate_self_fence_cadence`
    // internally — those individual validates MOVED there. `profile` was resolved by `enforce_cloud_preflight`
    // at the top (before any durable action).
    let tick_hz: u32 = env.parse("VD_TICK_HZ")?;
    let d3 = vd_io_prod::boot::resolve_d3(
        &env,
        profile,
        vd_io_prod::boot::NodeRole::Orchestrator,
        tick_hz,
    )
    .map_err(|e| e.to_string())?;
    let directory = d3.directory;
    let liveness = d3.liveness;
    // D-6 Slice D: the DURABLE redb Store (Store A — directory + saga WAL + clock ceiling). HR1: durable
    // state lives on a PERSISTENT volume keyed by path, NEVER under /tmp (the old /tmp/{shard_id} data-loss
    // bug). VD_STORE_PATH is REQUIRED (no silent in-memory fallback) and a temp path is REJECTED LOUD at
    // boot. A non-empty file is RE-HYDRATED (clock resumes forward, directory + in-flight sagas restore +
    // re-drive); an empty/new file is genesis. The off-tick writer fsyncs off the tick thread (C2).
    let store_path = PathBuf::from(env.string("VD_STORE_PATH")?);
    // Create the parent FIRST so the ephemeral check can canonicalize it (resolve symlinks).
    if let Some(parent) = store_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    // HR1 DURABLE-PATH GUARD — the SHARED `check_durable_path` (F2, DRY with the M3 boot-counter + the R-6d
    // outbox: ONE guard for every node durable path). When `VD_STORE_DURABLE_ROOT` is declared it is an
    // ALLOW-list (the store must canonicalize UNDER the mounted persistent volume — catching a k8s `emptyDir`
    // mounted outside the volume that a prefix DENY-list cannot enumerate); otherwise the temp deny-list (the
    // dev-safety net against the old `/tmp` data-loss bug). `VD_STORE_EPHEMERAL_OK=1` is the explicit dev/test
    // escape (a throwaway $TMPDIR cluster cleaned by `dev-cluster down`). The REQUIRED `VD_STORE_PATH` (no
    // in-memory fallback) still stands.
    let durable_root = env
        .string("VD_STORE_DURABLE_ROOT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from);
    let ephemeral_ok = vd_bins::parse_bool_env(&env, "VD_STORE_EPHEMERAL_OK")?;
    // Stringify so main's default `{:?}` error print surfaces the Display message (the "Refusing to boot"
    // guidance) rather than the struct-Debug of BootCounterError.
    vd_io_prod::boot::check_durable_path(&store_path, durable_root.as_deref(), ephemeral_ok)
        .map_err(|e| e.to_string())?;
    if ephemeral_ok {
        tracing::warn!(
            "orchestrator durable store at {} has VD_STORE_EPHEMERAL_OK set (a dev/test escape) — recovery \
             survives a restart but NOT a host reboot / tmp reap. Never production.",
            store_path.display()
        );
    }
    // D-6 D-delta sentinel (feature-gated): VD_STORE_TEST_SENTINEL_SEED enables the SIGKILL-mid-fsync
    // scenario — the orchestrator plants a distinct Realm grant whose durable store-key the writer pauses
    // on (pre-fsync) so the crash test can SIGKILL it in a deterministic submitted-but-not-durable window.
    // The pause-key bytes are computed via the ONE barrier encoder (no drift); the marker rides the store path.
    #[cfg(feature = "store-test-hooks")]
    let sentinel = match env.string("VD_STORE_TEST_SENTINEL_SEED") {
        Ok(s) => {
            let seed: u64 = s
                .parse()
                .map_err(|_| format!("VD_STORE_TEST_SENTINEL_SEED is not a u64: {s:?}"))?;
            let key = DirectoryKey::Realm(RealmId::System(seed));
            let prefix = vd_node::saga_runtime::directory_store_key(&key);
            let marker = store_path.with_extension("paused");
            Some((key, prefix, marker))
        }
        Err(_) => None,
    };
    let writer_channel_depth = env.parse_or(
        "VD_STORE_CHANNEL_DEPTH",
        StoreTuning::default().writer_channel_depth,
    )?;
    // Reject 0 LOUD: a 0-depth (rendezvous) channel couples the sim thread to the fsync — the exact
    // off-tick property C2 exists to provide. Never a silent mis-config (audit wf_ed40e95e MEDIUM).
    if writer_channel_depth == 0 {
        return Err(
            "VD_STORE_CHANNEL_DEPTH must be >= 1 — a 0 (rendezvous) channel couples the sim thread to the \
             off-tick fsync, defeating C2. Refusing to boot."
                .into(),
        );
    }
    let store_tuning = StoreTuning {
        writer_channel_depth,
        #[cfg(feature = "store-test-hooks")]
        pause_on_key_prefix: sentinel.as_ref().map(|(_, prefix, _)| prefix.clone()),
        #[cfg(feature = "store-test-hooks")]
        pause_marker_path: sentinel.as_ref().map(|(_, _, marker)| marker.clone()),
        // R-6d4-B4's fault / poll-override hooks are unit-test-only ⇒ never set by the orchestrator bin.
        #[cfg(feature = "store-test-hooks")]
        fail_fsync_on_key_prefix: None,
        #[cfg(feature = "store-test-hooks")]
        wait_poll_override: None,
    };
    let (store, durability) = RedbStore::open(&store_path, store_tuning)?;
    // The Store is now durable. REMAINING transport production precondition (DEFERRED.md D-6): the
    // redelivering mesh transport is at-least-once for a source that STAYS UP (R-1..R-5 + M3 durable
    // incarnation, both proven), but a SOURCE that CRASHES inside the producer-less `AwaitAdopt` recovery
    // phase still loses its un-acked batch until R-6d's durable outbox lands. Owed before a rolling deploy.
    tracing::warn!(
        "orchestrator durable store at {} (redb, off-tick fsync). REMAINING production precondition \
         (DEFERRED.md D-6): the redelivering mesh transport is at-least-once for a source that stays up \
         (R-1..R-5 + M3), but a SOURCE crash inside the producer-less AwaitAdopt phase still loses its \
         un-acked batch until R-6d's durable outbox lands.",
        store_path.display()
    );
    // RLM Step 5e: the REAL demand-driven realm spawner, wired LIVE but INERT (`RlmTuning::default()` below
    // ⇒ the reconciler never sweeps ⇒ `spawn_realm` is never called ⇒ boot is byte-identical). It forks real
    // `vd-shard` processes (`ProcLaunchBackend`) driven by the 100%-covered `SpawnCore` kernel. Its launch
    // ledger lives in a SEPARATE `launch.redb` beside the saga WAL — isolating the write-ahead-before-fork
    // durability barrier + decoupling launch fsyncs from the per-tick universe-clock barrier (zero edits to
    // the depth-1 D-6 writer core). 5f arms `RlmTuning` + adds the `--demand` launcher that assembles the
    // COMPLETE shard boot-env; the anchors here are the subset the orchestrator's own env carries (unused
    // while inert — the 5e kill-9 gate builds its own full anchors).
    let launch_store_path = store_path.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    vd_io_prod::boot::check_durable_path(&launch_store_path, durable_root.as_deref(), ephemeral_ok)
        .map_err(|e| e.to_string())?;
    // The launch ledger is low-write-rate (one write-ahead + one confirm per spawn), so the default writer
    // channel depth suffices — no need to mirror the saga WAL's tuned depth.
    let (launch_store, _launch_durability) =
        RedbStore::open(&launch_store_path, StoreTuning::default())?;
    let mut spawn_anchors: Vec<(&'static str, String)> = Vec::new();
    for key in [
        "VD_TRUST_DIR",
        "VD_TICK_HZ",
        "VD_TICK_DT",
        "VD_SPEED",
        "VD_SNAPSHOT_BUDGET",
        "VD_UNIVERSE_SEED",
        "VD_UNIVERSE_SCALE",
        "VD_OUTBOUND_CAP",
    ] {
        if let Some(v) = env.string(key).ok().filter(|v| !v.is_empty()) {
            spawn_anchors.push((key, v));
        }
    }
    spawn_anchors.push(("VD_ORCH", local.0.to_string()));
    // The child's VD_PEERS anchors: this orchestrator (self) + the gateway (if booked). The per-realm
    // ANCESTOR closure is computed per-spawn by SpawnCore; only these static anchors are held here.
    let mut anchor_peers: Vec<(vd_core::NodeId, std::net::SocketAddr)> =
        vec![(local, env.parse("VD_BIND")?)];
    if let Some(gw) = env.peer_book("VD_PEERS")?.get(&vd_bins::GATEWAY).copied() {
        anchor_peers.push((vd_bins::GATEWAY, gw));
    }
    let realm_spawner: Box<dyn vd_sim::io::RealmSpawner + Send + Sync> =
        Box::new(vd_node::rlm_spawn::SpawnCore::new(
            Box::new(launch_store),
            vd_bins::proc_launch::ProcLaunchBackend::new(
                Arc::clone(&control),
                vd_bins::proc_launch::ProcSpawnTuning {
                    exe: "vd-shard",
                    workdir: store_path.with_file_name("realm-logs"),
                    // Operational param (env-overridable, ONE default) — the SIGTERM→SIGKILL teardown grace.
                    drain_grace: std::time::Duration::from_millis(
                        env.parse_or("VD_REALM_DRAIN_GRACE_MS", 3_000)?,
                    ),
                    anchors: spawn_anchors,
                },
            ),
            vd_node::rlm_spawn::SpawnTuning::dev(),
            anchor_peers,
        ));
    register_orchestrator_with_store(
        world,
        schedule,
        &OrchestratorConfig {
            epoch: vd_core::EpochId(env.parse("VD_EPOCH")?),
            reserve_chunk: env.parse("VD_RESERVE_CHUNK")?,
            clock_peers: env.node_list("VD_CLOCK_PEERS")?,
            directory,
            saga,
            liveness,
            // Track R / 1d.2: the D-37 re-home roster is the routable-shard SET from VD_ROSTER (parsed by
            // the EXISTING `EnvConfig::node_list`), each an empty/stub `ShardProfile` (a P3 bare-point
            // subject's `CapRequest::default()` ⇒ the empty profile satisfies). Absent VD_ROSTER ⇒ empty
            // (a re-home parks; a single-shard `up` stays byte-identical). NOTE: `select_rehome_target` is
            // realm-BLIND (it picks the lowest live capable NodeId) — a D-37 concern, NOT the crossing,
            // which resolves via `head(Realm(to_realm))` (HR3-clean, roster-independent). SCOPE (M-2): this
            // local playground is 2-shard; the N-entry k3d roster is ledgered to cloud #123 in DEFERRED.md.
            roster: {
                let mut roster = std::collections::BTreeMap::new();
                for node in env.node_list("VD_ROSTER").unwrap_or_default() {
                    roster.insert(
                        node,
                        vd_sim::capability::ShardProfile::build(
                            vd_sim::capability::CapRequest::default(),
                        )
                        .expect("the empty ShardProfile is coherent"),
                    );
                }
                roster
            },
            rlm: vd_sim::rlm::RlmTuning::default(),
        },
        Box::new(store),
        // RLM Step 5e: the REAL `SpawnCore<ProcLaunchBackend>` built above (was a placeholder `MemSpawner`).
        // INERT while `RlmTuning::default()` keeps the reconciler from sweeping — 5f arms it via `--demand`.
        realm_spawner,
    );

    // The admin endpoint: republished after every tick, served off-thread.
    let cell = Arc::new(ArcSwap::from_pointee(AdminSnapshot::default()));
    let admin_addr: std::net::SocketAddr = env.parse("VD_ADMIN_ADDR")?;
    let served = Arc::clone(&cell);
    // R-4d M4: the live mesh-metrics source shares the retained MeshControl.
    let metrics = Arc::new(MeshMetrics(Arc::clone(&control)));
    runtime.spawn(async move {
        let listener = tokio::net::TcpListener::bind(admin_addr)
            .await
            .expect("admin endpoint binds");
        axum::serve(listener, admin_router(Arc::new(Published(served)), metrics))
            .await
            .expect("admin endpoint serves");
    });

    // PERSIST-BEFORE-EFFECT (Slice D parked-flush): split each tick into run_schedule (stages + submits
    // this tick's durable batch via the group-commit barrier) and flush_outbox (sends the tick's egress),
    // and DEFER the outbox by one tick — flush tick T's sends only after batch T is durable. The off-tick
    // writer ~always finishes within one tick at 50Hz, so the deferred flush waits on an already-durable
    // batch (the wait IS the disk-stall back-pressure, and fails LOUD if the writer died — a refusal is
    // never a loss). No effect ever leaves the orchestrator before the state authorizing it is durable.
    //
    // WHY gating on `last_submitted()` is correct (and stays correct under the owed idle-fsync-skip,
    // DEFERRED.md D-6 #2 — holistic audit `wf_ed40e95e`): every flushed effect depends on a PAST-or-same-tick
    // commit, and a same-tick state change STAGES a durable delta (the directory dirty-set / saga pending_writes)
    // ⇒ is submitted ⇒ reflected in `last_submitted`; a past commit is ≤ `last_submitted` by definition. So
    // `last_submitted()` at flush time ALWAYS covers the flushed tick's effect-state. Today the barrier also
    // stages the Clock key every tick, so `last_submitted` advances every tick — but even if a future write-amp
    // pass skips that on a FULLY IDLE tick, an idle tick has no state-dependent egress (only loss-tolerant
    // ClockSync, gated on the durable clock CEILING), so the gate holds. SHUTDOWN (cloud-ready k3d Slice 1):
    // the loop is now `while !shutdown` — on SIGTERM/SIGINT it BREAKS and runs the graceful drain below (flush
    // the final parked outbox + return → `RedbStore::Drop` joins the off-tick writer for a final fsync). A
    // HARD kill (SIGKILL / OOM / the SIGKILL escalation after the grace) still skips the drain, and that path
    // stays recovery-covered: rehydrate restores the last durable state + the Slice-2a producer re-drives
    // every saga command idempotently (all egress today is re-driven saga commands + loss-tolerant ClockSync;
    // a future RELIABLE non-re-driven effect would extend the drain's final-flush, the seam is already here).
    let mut pacer = TickPacer::new(tick_hz); // R-4c hoisted VD_TICK_HZ (also feeds validate_against)
    let mut parked: Option<(TickPrologue, u64)> = None;
    #[cfg(feature = "store-test-hooks")]
    let mut sentinel_injected = false;
    // Cloud-ready k3d Slice 3: the k8s probe surface (a SEPARATE port from the topology-rich /admin+/metrics).
    // Orchestrator readiness = its OWN serving capability (`orch_ready(true)` from the first tick), NOT the
    // whole-cluster `cluster_bootstrapped()` latch — which would deadlock a cold-start orchestrator (a shard
    // must dial IT to register) and stay Ready forever after every shard died. S4 note: the orchestrator's mesh
    // Service must be headless / publishNotReadyAddresses so a shard can dial it before the cluster is warm.
    let health = vd_io_prod::probe::new_health_cell();
    let shutdown_linger = vd_bins::resolve_shutdown_linger(&env)?;
    if let Some((probe_addr, probe_tuning)) = vd_bins::resolve_probe(&env)? {
        vd_bins::spawn_probe_server(
            runtime.handle(),
            probe_addr,
            Arc::new(vd_io_prod::probe::PublishedHealth::new(
                Arc::clone(&health),
                probe_tuning.stall_deadline(tick_hz),
            )),
            // the orchestrator carries no incarnation cookie (RLM 5c: /whoami is realm-shard-only).
            None,
        );
    }
    // Slice 1: SIGTERM/SIGINT flips this flag; the loop breaks to run the graceful drain below. The
    // with-health variant ALSO de-routes (/readyz 503) on the shutdown edge + marks `draining` so /healthz
    // stays LIVE through the final-fsync park (never a kubelet SIGKILL mid-write).
    let shutdown =
        vd_bins::install_shutdown_flag_with_health(runtime.handle(), Arc::clone(&health));
    // Cloud reschedule re-plumb: the peer-addr auto-resolver (production caller of update_peer_addr) — spawned
    // iff VD_PEER_HOSTS is set (cloud deploy), a no-op in-process. The orchestrator INITIATES to shards (realm
    // grants), so it must re-plumb a rescheduled shard's new IP. Shares the tick loop's shutdown flag + the
    // retained-for-/metrics `control` Arc.
    vd_bins::spawn_peer_resolver_if_configured(
        &env,
        runtime.handle(),
        Arc::clone(&control),
        Arc::clone(&shutdown),
    )?;
    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        // Flush the PREVIOUS tick's outbox now that its batch is durable. The world outbox still holds only
        // that tick's sends (this runs before the next run_schedule refills it), so the defer is exact.
        if let Some((prologue, seq)) = parked.take() {
            durability.wait_durable_through(seq);
            let _ = node.flush_outbox(prologue);
        }
        // D-delta: inject the sentinel grant ONCE, the first tick a shard's realm (grant A) is already in
        // the directory — i.e. in a PRIOR tick's batch, so by block-on-prior A is durable BEFORE this tick's
        // sentinel batch (B) is even submitted. The next run_schedule stages B; the writer pauses pre-fsync.
        #[cfg(feature = "store-test-hooks")]
        if let Some((key, _, _)) = sentinel.as_ref()
            && !sentinel_injected
        {
            let mut dir = node.world_mut().resource_mut::<DirectoryRes>();
            let a_durable = dir.0.entries().any(|(k, r)| {
                matches!(k, DirectoryKey::Realm(_)) && matches!(r.authority, AuthorityRef::Shard(_))
            });
            if a_durable {
                dir.0
                    .grant(*key, AuthorityRef::Shard(SHARD), Fence(1), UniverseTick(0));
                sentinel_injected = true;
            }
        }
        // Advance: run this tick's schedule (commit() submits this tick's batch); park its seq for next tick.
        let prologue = node.run_schedule();
        parked = Some((prologue, durability.last_submitted()));
        cell.store(Arc::new(admin_snapshot(node.world_mut())));
        // S3: publish the liveness heartbeat + readiness (orchestrator = own-serving, always ready once
        // ticking). `publish_tick` preserves `draining` so a post-SIGTERM tick cannot re-route.
        vd_io_prod::probe::publish_tick(&health, node.tick().0, vd_node::health::orch_ready(true));
        let _ = pacer.wait();
    }
    // S3 drain LINGER (default 0): the shutdown edge already set /readyz=503; linger in Terminating so k8s
    // de-routes + in-flight admin/mesh work settles before the final flush + exit (/healthz stays LIVE).
    std::thread::sleep(shutdown_linger);
    // GRACEFUL DRAIN (Slice 1): the loop broke on SIGTERM/SIGINT. Flush the FINAL parked outbox — wait for
    // its batch to be durable, then send it. Best-effort: this egress is re-driven saga commands +
    // loss-tolerant ClockSync, so an unsent frame is recovery-covered on the peer's next need, but flushing
    // it here cuts the successor's re-drive work. Then RETURN: dropping `node` drops the boxed `RedbStore`,
    // whose `Drop` drains the writer channel + JOINS the off-tick writer thread (a final fsync) — so the
    // durable directory / saga WAL / clock ceiling are consistent on disk before exit. `runtime` (declared
    // earlier) drops AFTER `node`, tearing down the mesh + admin tasks last. This makes a routine
    // rolling-deploy / pod-stop a clean flush-and-exit instead of a hard SIGKILL crash-path.
    if let Some((prologue, seq)) = parked.take() {
        durability.wait_durable_through(seq);
        let _ = node.flush_outbox(prologue);
    }
    tracing::info!("orchestrator drained on shutdown signal — flushed store, exiting cleanly");
    Ok(())
}
