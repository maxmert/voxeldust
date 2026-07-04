//! R-6d3b-2b: exercise `boot_mesh_and_replay`'s WITH-outbox (Some-path) GLUE in-process — the ~15-line
//! assembly sub-slice C adds (resolve incarnation ONCE → `open_node_outbox` → wrap as `SharedOutbox` →
//! `spawn_mesh(Some)` → `replay_outbox` BEFORE returning the transport). The process-tier `process_parity` /
//! `boot_counter_crashloop` tests boot on the `None` path, so without this the new glue never RUNS in a test
//! and a `spawn_mesh`-arg / boot-ordering regression would ship green.
//!
//! This uses an EMPTY outbox ⇒ `replay_outbox` is a no-op (`Ok(ReplayCounts::default())`); it proves the glue
//! ASSEMBLES, opens the durable store, spawns the mesh WITH the sink, replays, and returns a live transport
//! without hanging. The re-drive + SIGKILL-restart + boot-counter-by-exactly-one e2e are R-6d4 (they need the
//! `pub(crate)` `ReliableFrame` to pre-seed a durable row, and belong with the SIGKILL process proof).

use std::collections::BTreeMap;

use vd_bins::{book, boot_mesh_and_replay, reserve_udp_addr};
use vd_core::NodeId;
use vd_io_prod::runtime::EnvConfig;
use vd_io_prod::trust::ClusterTrust;
use vd_sim::io::Transport;

#[test]
fn boot_mesh_and_replay_wires_a_live_outbox_and_returns() {
    let a = NodeId(1);
    let b = NodeId(2);
    let (addr_a, addr_b) = (reserve_udp_addr(), reserve_udp_addr());
    let outbox_path = std::env::temp_dir().join(format!("vd-bmr-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&outbox_path);

    // VD_PROCESS_INCARNATION explicit (1) so `resolve_process_incarnation` takes the non-mutating path (no
    // BootCounter); VD_OUTBOX_PATH + EPHEMERAL_OK ⇒ the Some-path glue opens a real (empty) durable outbox.
    let env = EnvConfig::new(BTreeMap::from([
        ("VD_NODE_ID".to_owned(), "1".to_owned()),
        ("VD_BIND".to_owned(), addr_a.to_string()),
        ("VD_PEERS".to_owned(), book(&[(a, addr_a), (b, addr_b)])),
        ("VD_OUTBOUND_CAP".to_owned(), "64".to_owned()),
        ("VD_PROCESS_INCARNATION".to_owned(), "1".to_owned()),
        (
            "VD_OUTBOX_PATH".to_owned(),
            outbox_path.display().to_string(),
        ),
        ("VD_OUTBOX_EPHEMERAL_OK".to_owned(), "1".to_owned()),
    ]));
    let trust = ClusterTrust::generate("vd-bmr").expect("trust");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("rt");

    // THE Some-path glue: open + wrap + spawn_mesh(Some) + replay (empty ⇒ no-op) + return. If the glue
    // mis-ordered replay after build_app, passed `None`, or hung, this would fail/hang.
    let (transport, control) =
        boot_mesh_and_replay(&env, rt.handle(), &trust).expect("boot with a live outbox");
    assert!(outbox_path.exists(), "the durable outbox was opened by the glue");
    assert_eq!(transport.local_id(), a, "the mesh came up as the configured node");

    drop((transport, control));
    drop(rt); // stop the peer-writer tasks so the outbox store writer joins + releases the file
    let _ = std::fs::remove_file(&outbox_path);
}
