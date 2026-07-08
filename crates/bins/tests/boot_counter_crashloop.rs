//! R-6b: the LOOPBACK CrashLoop e2e proof for the M3 durable monotone incarnation boot-counter.
//!
//! The hazard R-6a cures: a node that restarts must come up at a STRICTLY HIGHER process incarnation than
//! the value a still-running peer's mesh ledger holds, or the peer silently DEDUPS the restarted node's
//! reliable traffic (its per-`(peer,class)` seq resets to 0, and `seq <= hw` at an EQUAL incarnation is a
//! `Dedup`). Under k8s a wall-clock incarnation (`launch_incarnation`) collides on a sub-second CrashLoop
//! restart; the durable boot-counter increments instead.
//!
//! This proves the cure END-TO-END across a REAL process SIGKILL + restart, over the real QUIC mesh, by the
//! PAIRING of two runs against the ORCHESTRATOR's `/metrics` `vd_dedup_drop_total` (R-4d): the shard grants
//! its realm A to the orchestrator (a reliable directory op the orchestrator's ledger tracks), then is
//! SIGKILLed + restarted, so it re-grants at seq0.
//! - RED control (fixed `VD_PROCESS_INCARNATION`, reused across the restart): the re-grant arrives at the
//!   SAME incarnation ⇒ the orchestrator DEDUPS it ⇒ `dedup_drop` CLIMBS (the silent-loss hazard).
//! - GREEN (the durable boot-counter, `VD_BOOT_STATE_DIR`): the re-grant arrives at a HIGHER incarnation ⇒
//!   the receiver's A1 ladder RESETS the dedup high-water ⇒ the re-grant is Accepted ⇒ `dedup_drop` stays 0.
//!
//! Same shard behavior in both; only the incarnation source differs — so RED>0 while GREEN==0 demonstrates
//! the boot-counter fixes exactly what the wall-clock version breaks. No Docker/k3d, runs in CI.

use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, admin_get_body, common_env, orchestrator_env, reserve_tcp_addr,
    reserve_udp_addr, shard_env, spawn_node,
};
use vd_core::pose::RealmId;
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::AdminSnapshot;

const DEADLINE: Duration = Duration::from_secs(30);

/// RAII removal of a run's on-disk temp artifacts on exit (the run also clears them at START, which is
/// pid-stable — this stops distinct `cargo test` pids from accumulating `$TMPDIR` cruft).
struct TempPaths {
    store: std::path::PathBuf,
    trust_dir: std::path::PathBuf,
    boot_dir: std::path::PathBuf,
}
impl Drop for TempPaths {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.store);
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_dir_all(&self.boot_dir);
    }
}

/// Parse a bare Prometheus counter `<name> <value>` (skipping the `# TYPE` lines) — 0 if absent.
fn metric(body: &str, name: &str) -> u64 {
    body.lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| {
            let mut it = l.split_whitespace();
            if it.next() == Some(name) {
                it.next().and_then(|v| v.parse().ok())
            } else {
                None
            }
        })
        .unwrap_or(0)
}

/// GET the orchestrator admin snapshot (`None` while it is not yet serving). `admin_get_body` already
/// strips the HTTP headers, so the body is bare JSON.
fn admin(addr: std::net::SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

/// The orchestrator's `dedup_drop` + `stale_incarnation_drop` totals (the restarted-shard silent-loss
/// counters), or `None` while `/metrics` is unreachable.
fn drop_counters(admin_addr: std::net::SocketAddr) -> Option<(u64, u64)> {
    let body = admin_get_body(admin_addr, "/metrics", Some(Duration::from_secs(2)))?;
    Some((
        metric(&body, "vd_dedup_drop_total"),
        metric(&body, "vd_stale_incarnation_drop_total"),
    ))
}

/// Run one CrashLoop case and return the orchestrator's (dedup, stale-incarnation) drop DELTA across the
/// shard's SIGKILL + restart. `use_boot_counter=false` reuses a FIXED `VD_PROCESS_INCARNATION` (the RED
/// wall-clock hazard); `true` uses the durable `VD_BOOT_STATE_DIR` boot-counter (GREEN).
fn crashloop_drop_delta(use_boot_counter: bool) -> (u64, u64) {
    let tag = if use_boot_counter { "green" } else { "red" };
    let orch = reserve_udp_addr();
    let gateway = reserve_udp_addr();
    let shard = reserve_udp_addr();
    let admin_addr = reserve_tcp_addr();
    let pid = std::process::id();
    let trust_dir = std::env::temp_dir().join(format!("vd-crashloop-{tag}-{pid}"));
    let store = std::env::temp_dir().join(format!("vd-crashloop-{tag}-{pid}.redb"));
    let boot_dir = std::env::temp_dir().join(format!("vd-crashloop-boot-{tag}-{pid}"));
    // Clear at start (pid-stable) AND remove on exit (a distinct `cargo test` pid must not accumulate cruft).
    let _clean = TempPaths {
        store: store.clone(),
        trust_dir: trust_dir.clone(),
        boot_dir: boot_dir.clone(),
    };
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_dir_all(&boot_dir);
    let trust = ClusterTrust::generate("vd-crashloop").expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    let store_str = store.display().to_string();
    // The orchestrator keeps the fixed common env (its OWN incarnation is irrelevant — the test tracks the
    // SHARD's incarnation as recorded in the orchestrator's ledger).
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    let addrs = ClusterAddrs {
        orchestrator: orch,
        gateway,
        shard,
        admin: admin_addr,
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
    };
    let mut cluster = Cluster::new(); // RAII-reaped
    cluster.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(&addrs, &DEV, &store_str),
        )
        .expect("spawn orch"),
    );

    // The shard's shared env: RED keeps VD_PROCESS_INCARNATION (a fixed value, reused across the restart);
    // GREEN drops it and uses the durable boot-counter under a FIXED dir that survives the restart.
    let shard_common: Vec<(&'static str, String)> = if use_boot_counter {
        let mut v: Vec<_> = common
            .iter()
            .filter(|(k, _)| *k != "VD_PROCESS_INCARNATION")
            .cloned()
            .collect();
        v.push(("VD_BOOT_STATE_DIR", boot_dir.display().to_string()));
        v.push(("VD_BOOT_STATE_EPHEMERAL_OK", "1".to_owned())); // the boot dir is under $TMPDIR (a test)
        v
    } else {
        common.clone()
    };

    let spawn_shard = || {
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            &shard_common,
            &shard_env(&addrs, &DEV),
        )
        .expect("spawn shard")
    };
    cluster.push("vd-shard", spawn_shard());

    // Wait for the shard to grant its realm A to the directory over the QUIC mesh (the reliable-path-
    // established signal — the orchestrator's ledger now tracks the shard's incarnation).
    let a_key = RealmId::System(DEV.realm_seed).to_string();
    let started = Instant::now();
    loop {
        if let Some(s) = admin(admin_addr)
            && s.cluster_bootstrapped()
            && s.directory.iter().any(|e| e.key == a_key)
        {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "[{tag}] the shard never granted realm A (the reliable mesh path never established)"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let (dedup_before, stale_before) =
        drop_counters(admin_addr).expect("metrics reachable pre-kill");

    // SIGKILL the shard, reap (release the port), then restart it sub-second on the SAME bind + env.
    cluster.kill_and_reap("vd-shard");
    cluster.push("vd-shard-restart", spawn_shard());

    // Let the restarted shard reconnect + re-grant (re-sending seq0 at its incarnation). RED dedups it
    // (the orchestrator's persisted ledger holds a higher incarnation-equal watermark); GREEN accepts it.
    // Poll: RED's dedup climbs within the deadline; GREEN's never does. A generous settle for GREEN ensures
    // the re-grant WAS sent (RED, with identical shard behavior, proves it reaches the orchestrator).
    // 12s budget: RED breaks early the instant a drop climbs (the reconnect + re-grant is ~1-3s); GREEN
    // never breaks and waits the full budget, so a 0 result is over a window in which RED (identical shard
    // behavior) has demonstrably re-sent + been dropped — not a too-short settle.
    let settle = Instant::now();
    let mut last = (dedup_before, stale_before);
    while settle.elapsed() < Duration::from_secs(12) {
        if let Some(c) = drop_counters(admin_addr) {
            last = c;
            if c.0 > dedup_before || c.1 > stale_before {
                break; // a drop climbed (the RED hazard manifested) — no need to wait the full settle
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let delta = (
        last.0.saturating_sub(dedup_before),
        last.1.saturating_sub(stale_before),
    );
    eprintln!(
        "[{tag}] restarted-shard drop delta over the settle: dedup+{} stale_incarnation+{}",
        delta.0, delta.1
    );
    delta
    // `cluster` (orchestrator + restarted shard) is reaped here on drop.
}

#[test]
fn a_crashloop_restart_is_silently_deduped_at_a_fixed_incarnation_but_not_with_the_boot_counter() {
    // RED: a fixed (wall-clock-class) incarnation reused across the restart ⇒ the orchestrator dedups the
    // restarted shard's re-grant (the silent-loss hazard M3 exists to cure).
    let (red_dedup, red_stale) = crashloop_drop_delta(false);
    assert!(
        red_dedup > 0 || red_stale > 0,
        "RED: a fixed incarnation must make the restarted shard's re-grant be silently dropped \
         (dedup/stale) by the orchestrator — got dedup+{red_dedup}, stale+{red_stale}"
    );

    // GREEN: the durable boot-counter mints a HIGHER incarnation on restart ⇒ the orchestrator's A1 ladder
    // resets the dedup high-water ⇒ the re-grant is Accepted ⇒ ZERO silent drops. Same shard behavior as
    // RED (which proved the re-grant reaches the orchestrator), so 0 here is the fix, not a vacuous absence.
    let (green_dedup, green_stale) = crashloop_drop_delta(true);
    assert_eq!(
        (green_dedup, green_stale),
        (0, 0),
        "GREEN: the durable boot-counter must prevent ANY silent dedup/stale drop of the restarted shard's \
         reliable traffic — got dedup+{green_dedup}, stale+{green_stale}"
    );
}
