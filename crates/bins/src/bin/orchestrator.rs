//! The orchestrator binary: directory + universe clock + the read-only admin
//! endpoint (the 2am `curl`). The admin snapshot is republished after every tick
//! through a lock-free cell — the HTTP task never touches the sim thread.

use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;
use vd_io_prod::admin::{SnapshotSource, admin_router};
use vd_io_prod::mesh::{MeshConfig, spawn_mesh};
use vd_io_prod::runtime::{EnvConfig, TickPacer};
use vd_io_prod::store::{RedbStore, StoreTuning};
use vd_io_prod::trust::ClusterTrust;
use vd_node::app::{NodeConfig, TickPrologue, build_app};
use vd_node::orchestrator::{OrchestratorConfig, admin_snapshot, register_orchestrator_with_store};
use vd_sim::capability::NodeKind;
use vd_sim::directory::DirectoryTuning;
use vd_wire::admin::AdminSnapshot;

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
    let (transport, _control) = spawn_mesh(
        runtime.handle(),
        &trust,
        &MeshConfig::new(
            local,
            env.parse("VD_BIND")?,
            env.peer_book("VD_PEERS")?,
            env.parse("VD_OUTBOUND_CAP")?,
        ),
    )?;
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
        min_renews_before_lapse: env.parse_or("VD_MIN_RENEWS_BEFORE_LAPSE", d3.min_renews_before_lapse)?,
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
    // D-6 Slice D: the DURABLE redb Store (Store A — directory + saga WAL + clock ceiling). HR1: durable
    // state lives on a PERSISTENT volume keyed by path, NEVER under /tmp (the old /tmp/{shard_id} data-loss
    // bug). VD_STORE_PATH is REQUIRED (no silent in-memory fallback) and a temp path is REJECTED LOUD at
    // boot. A non-empty file is RE-HYDRATED (clock resumes forward, directory + in-flight sagas restore +
    // re-drive); an empty/new file is genesis. The off-tick writer fsyncs off the tick thread (C2).
    let store_path = PathBuf::from(env.string("VD_STORE_PATH")?);
    let tmp = std::env::temp_dir();
    let under_temp = store_path.starts_with(&tmp) || store_path.starts_with("/tmp");
    // VD_STORE_EPHEMERAL_OK (present = opt-in) is the EXPLICIT dev/test escape: a throwaway local cluster
    // legitimately stores under $TMPDIR (cleaned by `dev-cluster down`). Production OMITS it, so a temp-dir
    // path is a HARD boot reject (HR1: durable state on a persistent volume, never the old /tmp data-loss
    // bug) — never a silent ephemeral production deployment.
    let ephemeral_ok = env.string("VD_STORE_EPHEMERAL_OK").is_ok();
    if under_temp && !ephemeral_ok {
        return Err(format!(
            "VD_STORE_PATH ({}) is under a temp dir ({}) — durable orchestrator state (directory + saga \
             WAL + clock ceiling) MUST live on a persistent volume (HR1; the old /tmp data-loss bug). Set \
             VD_STORE_EPHEMERAL_OK=1 ONLY for a throwaway dev/test cluster. Refusing to boot.",
            store_path.display(),
            tmp.display()
        )
        .into());
    }
    if under_temp {
        tracing::warn!(
            "orchestrator durable store is under a TEMP dir ({}) with VD_STORE_EPHEMERAL_OK set — this is a \
             dev/test cluster; recovery survives a restart but NOT a host reboot / tmp reap. Never production.",
            store_path.display()
        );
    }
    if let Some(parent) = store_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let store_tuning = StoreTuning {
        writer_channel_depth: env
            .parse_or("VD_STORE_CHANNEL_DEPTH", StoreTuning::default().writer_channel_depth)?,
    };
    let (store, durability) = RedbStore::open(&store_path, store_tuning)?;
    // The Store is now durable; the REMAINING production precondition (DEFERRED.md D-6) is the transport:
    // the mesh is at-most-once, so a producer-less recovery phase (`AwaitAdopt`) could wedge on a real
    // kill-9 even WITH this durable store — recovery is proven vs the harness at-least-once model. A
    // redelivering transport is owed before a rolling production deploy.
    tracing::warn!(
        "orchestrator durable store at {} (redb, off-tick fsync). REMAINING production precondition \
         (DEFERRED.md D-6): the mesh transport is at-most-once, so a producer-less recovery phase \
         (AwaitAdopt) could wedge on a real kill-9 even with this durable store; a redelivering transport \
         is owed before a rolling production deploy.",
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
    runtime.spawn(async move {
        let listener = tokio::net::TcpListener::bind(admin_addr)
            .await
            .expect("admin endpoint binds");
        axum::serve(listener, admin_router(Arc::new(Published(served))))
            .await
            .expect("admin endpoint serves");
    });

    // PERSIST-BEFORE-EFFECT (Slice D parked-flush): split each tick into run_schedule (stages + submits
    // this tick's durable batch via the group-commit barrier) and flush_outbox (sends the tick's egress),
    // and DEFER the outbox by one tick — flush tick T's sends only after batch T is durable. The off-tick
    // writer ~always finishes within one tick at 50Hz, so the deferred flush waits on an already-durable
    // batch (the wait IS the disk-stall back-pressure, and fails LOUD if the writer died — a refusal is
    // never a loss). No effect ever leaves the orchestrator before the state authorizing it is durable.
    let mut pacer = TickPacer::new(env.parse("VD_TICK_HZ")?);
    let mut parked: Option<(TickPrologue, u64)> = None;
    loop {
        // Flush the PREVIOUS tick's outbox now that its batch is durable. The world outbox still holds only
        // that tick's sends (this runs before the next run_schedule refills it), so the defer is exact.
        if let Some((prologue, seq)) = parked.take() {
            durability.wait_durable_through(seq);
            let _ = node.flush_outbox(prologue);
        }
        // Advance: run this tick's schedule (commit() submits this tick's batch); park its seq for next tick.
        let prologue = node.run_schedule();
        parked = Some((prologue, durability.last_submitted()));
        cell.store(Arc::new(admin_snapshot(node.world_mut())));
        let _ = pacer.wait();
    }
}
