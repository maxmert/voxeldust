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
use vd_sim::directory::DirectoryTuning;
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
    // D-3 lease-liveness budget: the heartbeat/reaper/self-fence knobs, env-overridable per deployment,
    // defaulting INERT (renew/reaper intervals 0 = pre-D-3 behavior). `validate` rejects a mis-tuned
    // ordering chain at boot (a too-sparse renewal, or a self-fence grace that overlaps the orchestrator's
    // reassign-after window = a split-brain window) — LOUD config error, never a silent liveness hole.
    let d3 = DirectoryTuning::default();
    let directory = DirectoryTuning {
        lease_ttl_ticks: env.parse("VD_LEASE_TTL")?,
        lease_renew_interval_ticks: env
            .parse_or("VD_LEASE_RENEW_INTERVAL", d3.lease_renew_interval_ticks)?,
        min_renews_before_lapse: env
            .parse_or("VD_MIN_RENEWS_BEFORE_LAPSE", d3.min_renews_before_lapse)?,
        self_fence_grace_ticks: env.parse_or("VD_SELF_FENCE_GRACE", d3.self_fence_grace_ticks)?,
        max_self_fence_grace_ticks: env
            .parse_or("VD_MAX_SELF_FENCE_GRACE", d3.max_self_fence_grace_ticks)?,
        reaper_interval_ticks: env.parse_or("VD_REAPER_INTERVAL", d3.reaper_interval_ticks)?,
        recovery_grace_ticks: env.parse_or("VD_RECOVERY_GRACE", d3.recovery_grace_ticks)?,
    };
    directory.validate()?;
    // D-3 dead-vs-slow confirmation tuning. PROD-SAFE default `n = 3` (the CSCALE-1 margin: a single
    // recoverable blip / idle-reap of a LIVE peer never confirms it dead) — DISTINCT from the dev/test
    // `LivenessTuning::default()` (`n = 1`, kill-equivalent). `validate` rejects a 0 confirmation count
    // or a window too tight to span the run at boot.
    let liveness = vd_sim::saga::LivenessTuning {
        n_consecutive_unreachable: env.parse_or("VD_LIVENESS_N", 3)?,
        unreachable_window_ticks: env.parse_or(
            "VD_LIVENESS_WINDOW",
            vd_sim::saga::LivenessTuning::default().unreachable_window_ticks,
        )?,
        retry_delay_ticks_hint: env.parse_or(
            "VD_LIVENESS_RETRY_HINT",
            vd_sim::saga::LivenessTuning::default().retry_delay_ticks_hint,
        )?,
    };
    liveness.validate()?;
    // R-4c: the orchestrator is the ONE process holding BOTH the transport redial backoff (the mesh it just
    // spawned) and the saga liveness window — cross-validate them LOUD. Nothing else asserts the two clocks
    // are ordered: a `VD_LIVENESS_WINDOW` narrower than the GEOMETRIC backoff spread of the confirmation run
    // lets `record_unreachable` reset the run before the n-th notice, so a genuinely-dead peer is NEVER
    // confirmed dead (the never-confirm strand) — the coarse `validate()` linear check (a hand-entered
    // `retry_delay_ticks_hint` disconnected from the real backoff) does NOT catch it.
    let tick_hz: u32 = env.parse("VD_TICK_HZ")?;
    liveness.validate_against(
        mesh_cfg.reliability.confirm_unreachable_after_retries,
        mesh_cfg.redial_backoff_min,
        mesh_cfg.redial_backoff_max,
        tick_hz,
    )?;
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
            // D-37: EMPTY in prod for now (ledgered DEFERRED.md D-37) — a re-home parks until the
            // per-shard-profile roster config lands (P3 is harness-driven; the cluster builder wires its
            // own roster from the shard list). No env knob added yet (no-unilateral-deps).
            roster: std::collections::BTreeMap::new(),
        },
        Box::new(store),
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
    // Cloud-ready k3d Slice 1: SIGTERM/SIGINT flips this flag; the loop breaks to run the graceful drain
    // below (flush the final parked outbox + let RedbStore::Drop join the off-tick writer for a final fsync).
    let shutdown = vd_bins::install_shutdown_flag(runtime.handle());
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
        let _ = pacer.wait();
    }
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
