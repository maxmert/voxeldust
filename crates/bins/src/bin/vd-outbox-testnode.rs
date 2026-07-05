//! R-6d4-D — the process-tier SIGKILL-restart proof harness (store-test-hooks ONLY; ABSENT from release —
//! a plain `[[bin]]` with a feature-gated `main` so `CARGO_BIN_EXE_vd-outbox-testnode` is always defined for
//! the `outbox_sigkill_restart` integration test). A minimal node whose ONLY job is to make the durable
//! outbox + boot replay crash-provable at the process tier, WITHOUT the shard bin's stub-realm/self-fence
//! noise:
//!
//! - boot 1 (`VD_OUTBOX_TEST_SEED` SET): resolve the process incarnation FIRST (mints + durably writes v1 via
//!   the R-6a BootCounter — so a later `v2 == v1 + 1` is provable), open the durable outbox, SEED one durable
//!   retained `ReliableFrame` row to peer B, confirm it is on disk, write the `.seeded` marker, then IDLE
//!   (park, holding the outbox open) — the test SIGKILLs it here. NO mesh, NO replay (that would gc the seed).
//! - boot 2 (`VD_OUTBOX_TEST_SEED` UNSET): `boot_mesh_and_replay` (resolves incarnation ONCE → v2 = v1 + 1,
//!   opens the SAME outbox, re-drives the retained row to B), then keep the mesh alive so the peer-writer
//!   actually delivers the re-driven frame.

#[cfg(feature = "store-test-hooks")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use vd_core::NodeId;
    use vd_io_prod::outbox::OutboxSink;
    use vd_io_prod::runtime::EnvConfig;
    use vd_io_prod::trust::ClusterTrust;
    use vd_sim::io::MsgClass;

    tracing_subscriber::fmt().with_env_filter("info").init();
    let env = EnvConfig::from_process_env();

    match env.string("VD_OUTBOX_TEST_SEED") {
        // BOOT 1: seed a durable-unacked row, then idle for the SIGKILL.
        Ok(spec) if !spec.trim().is_empty() => {
            let payload: u8 = spec.trim().parse()?; // the single payload byte the test will observe at B
            // Mint v1 durably (the R-6a counter) — boot 2's boot_mesh_and_replay mints v2 = v1 + 1.
            let incarnation = vd_bins::resolve_process_incarnation(&env)?;
            let from = env.node_id("VD_NODE_ID")?;
            let mut ob = vd_bins::open_node_outbox(&env)?
                .ok_or("VD_OUTBOX_PATH must be set to seed a durable row")?;
            // Fixed topology: peer B = NodeId(2), class Saga, seq 0.
            let seq = ob.seed_reliable_row(from, NodeId(2), MsgClass::Saga, incarnation, 0, &[payload]);
            assert!(
                ob.durability().is_durable_through(seq),
                "seeded outbox row is durable-on-disk before the SIGKILL window opens"
            );
            // The decoupled "durable window open" signal the test polls (the writer already fsynced above).
            let marker = std::path::PathBuf::from(env.string("VD_OUTBOX_PATH")?).with_extension("seeded");
            std::fs::write(&marker, b"seeded")?;
            tracing::info!(?marker, seq, "boot 1: seeded a durable outbox row; idling for SIGKILL");
            // Idle holding the outbox open — the test SIGKILLs here (do NOT drop: no graceful flush/join).
            loop {
                std::thread::park();
            }
        }
        // BOOT 2: replay the retained row across the restart + keep the mesh alive to deliver it. The tokio
        // runtime + trust are built ONLY here (where boot_mesh_and_replay consumes them), so "boot 1 builds no
        // mesh" is structurally true — the seed path above never constructs a runtime it does not use.
        _ => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?;
            let trust = ClusterTrust::from_der_dir(std::path::Path::new(&env.string("VD_TRUST_DIR")?))?;
            let (_transport, _control) =
                vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
            // The "boot done" signal (mesh up, incarnation minted, retained rows re-driven) — the
            // anti-theater twin polls this before reading the counter + killing (no seed ⇒ no `.seeded`).
            let ready = std::path::PathBuf::from(env.string("VD_OUTBOX_PATH")?).with_extension("ready");
            std::fs::write(&ready, b"ready")?;
            tracing::info!("boot 2: boot_mesh_and_replay done (retained rows re-driven); keeping mesh alive");
            // Hold the transport + control + runtime alive so the peer-writer delivers the re-driven frame
            // to the in-process receiver B the test holds. The test observes B, then SIGKILLs us at teardown.
            loop {
                std::thread::park();
            }
        }
    }
}

#[cfg(not(feature = "store-test-hooks"))]
fn main() {
    eprintln!(
        "vd-outbox-testnode is a store-test-hooks-only crash-test harness (R-6d4-D); \
         build with --features store-test-hooks"
    );
    std::process::exit(2);
}
