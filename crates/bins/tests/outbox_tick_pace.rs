//! THE PRICE OF DURABILITY, MEASURED (foundation slice 4; owner: "add durability, make sure performance
//! does not degrade").
//!
//! Turning the durable outbox on puts an fsync in front of every retained frame a shard sends. The claim
//! that this costs nothing is an argument, not a result — so this gate boots the SAME single-shard cluster
//! twice, with the outbox OFF and then ON, stands eight real clients in it each time, and reads the shard's
//! own pace line (`slowest_ms`, `mean_ms`, `ticks`) out of its log.
//!
//! WHY THE BOUND IS WHAT IT IS. Two different numbers are asserted, because two different things could go
//! wrong.
//!
//!   * THE MEAN may not grow by more than half again, plus a millisecond
//!     (`mean_on <= mean_off * 1.5 + 1.0`). This is deliberately generous: the runs are sequential on a
//!     shared developer machine, so the two means are never measured under identical load, and a tight
//!     bound would fail for reasons that have nothing to do with the outbox. The additive millisecond
//!     keeps a very fast shard (a sub-millisecond mean) from failing on ordinary jitter. The ratio is
//!     PRINTED, so the owner reads the real number rather than only "it passed".
//!   * THE SLOWEST TICK must stay inside the tick budget (`1 / tick_hz`). This is the one that matters:
//!     a shard that misses its tick stamps its statements sparsely, and a gateway ring that never sees a
//!     tick in common with its neighbours folds nothing. Durability may cost a little; it may never cost
//!     a tick.
//!
//! WHAT IS MEASURED. Only pace windows that START after every client is Active: the test waits for one
//! fresh pace line (which closes the window the logins fell in), remembers where the log then ends, lets
//! the clients stand for ten seconds, and reads only what was written after that mark. So the numbers
//! describe a shard doing the ordinary work of holding eight players, not a shard booting.
//!
//! Gated on `dev-control` (the clients are real `client` processes). Run with
//! `cargo test --release -p vd-bins --features dev-control --test outbox_tick_pace -- --nocapture`.
#![cfg(feature = "dev-control")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, common_env, dev_auth_pubkey_hex, dev_auth_signing_key_hex,
    dev_roundtrip, gateway_env, orchestrator_env, reserve_tcp_addr, reserve_udp_addr, shard_env,
};
use vd_devproto::{DevPhase, DevRequest, DevResponse, DevState};

/// The players standing in the realm while the pace is read. `DEV.max_sessions` is 8, so this is the whole
/// room the dev gateway admits.
const K: u16 = 8;

/// How long the clients stand still while the shard's pace is recorded.
const STAND: Duration = Duration::from_secs(10);

/// EIGHT logins, not one. The single-login budget below covers one client meeting a booted cluster;
/// this run stands the whole room up at once, on a cold shard that must fold the world first, so the
/// convergence wait is that budget four times over. A generous wait costs nothing when it passes.
fn convergence_deadline() -> Duration {
    login_deadline() * 4
}

/// One boot budget plus a margin — the same number every other process gate waits a login out on.
fn login_deadline() -> Duration {
    let confirm_window = vd_io_prod::mesh::DEFAULT_REDIAL_BACKOFF_MAX
        * vd_io_prod::mesh::DEFAULT_CONFIRM_UNREACHABLE_AFTER_RETRIES;
    let boot_budget = Duration::from_millis(DEV.boot_ticks_p99 * 1_000 / u64::from(DEV.tick_hz));
    confirm_window + boot_budget
}

/// The tick a shard must not miss.
fn tick_budget_ms() -> f64 {
    1_000.0 / f64::from(DEV.tick_hz)
}

/// What one run reports.
#[derive(Debug)]
struct Pace {
    /// The mean of every window's `mean_ms`.
    mean_ms: f64,
    /// The largest `slowest_ms` any window reported.
    slowest_ms: f64,
    /// How many pace windows were read (zero is a failed measurement, never a pass).
    windows: usize,
}

/// A process that dies with its guard (the clients are not in the node `Cluster`).
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn poll_state(port: u16) -> Option<DevState> {
    match dev_roundtrip(port, &DevRequest::State).ok()? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

/// A node writes its log with colour escapes even into a file, and an escape sits BETWEEN a field's name
/// and its `=`. Strip them first or every field reads as a different name than it has.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // A colour escape runs to its terminating `m`; drop the whole of it.
            for c2 in chars.by_ref() {
                if c2 == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// One `key=value` field of a tracing line, by exact field name — token by token, so `ticks` is never read
/// out of `local_tick` and `mean_ms` is never read out of anything else.
fn field(line: &str, key: &str) -> Option<f64> {
    line.split_whitespace()
        .filter_map(|token| token.split_once('='))
        .find(|(name, _)| *name == key)
        .and_then(|(_, value)| value.trim_end_matches(',').parse().ok())
}

/// Is this the shard's per-window pace line? The over-budget warning carries `slowest_ms` and `mean_ms`
/// too, but only the pace line counts its `ticks` — so a window is never read twice.
fn pace_of(line: &str) -> Option<(f64, f64)> {
    field(line, "ticks")?;
    Some((field(line, "mean_ms")?, field(line, "slowest_ms")?))
}

/// Every pace window written to the log after `from` bytes.
fn pace_windows(log: &Path, from: u64) -> Vec<(f64, f64)> {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let start = usize::try_from(from).unwrap_or(0).min(text.len());
    text[start..]
        .lines()
        .filter_map(|line| pace_of(&strip_ansi(line)))
        .collect()
}

/// The log's current length in bytes — the mark the measurement starts from.
fn log_len(log: &Path) -> u64 {
    std::fs::metadata(log).map(|m| m.len()).unwrap_or(0)
}

/// Boot the single-shard cluster, stand K clients in it, and read the shard's pace.
///
/// `outbox` selects the ONE difference between the two runs: with it false the two outbox keys are
/// stripped from the shard's env, which is exactly the state every shard ran in before this slice.
fn run_once(tag: &'static str, outbox: bool) -> Pace {
    let addrs = ClusterAddrs {
        orchestrator: reserve_udp_addr(),
        gateway: reserve_udp_addr(),
        shard: reserve_udp_addr(),
        admin: reserve_tcp_addr(),
        ..ClusterAddrs::reserve()
    };
    let trust_dir = std::env::temp_dir().join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&trust_dir);
    let trust = vd_io_prod::trust::ClusterTrust::generate(tag).expect("trust");
    trust.write_der_dir(&trust_dir).expect("trust dir");
    let work_dir = vd_bins::fresh_work_dir(tag);
    let orch_store = work_dir.join("orchestrator.redb").display().to_string();
    let common = common_env(&trust_dir.display().to_string(), &DEV);

    // THE ONE DIFFERENCE between the two runs.
    let mut shard_e = shard_env(&addrs, &DEV, vd_bins::ClusterShape::Single, &work_dir);
    if !outbox {
        shard_e.retain(|(k, _)| *k != "VD_OUTBOX_PATH" && *k != "VD_OUTBOX_EPHEMERAL_OK");
    }
    assert_eq!(
        shard_e.iter().any(|(k, _)| *k == "VD_OUTBOX_PATH"),
        outbox,
        "the run must differ from its twin in the outbox and nothing else"
    );

    // Each node's output goes to its own file in the work dir, exactly as the dev-cluster launcher does —
    // the shard's is the one this gate reads.
    let log_of = |label: &str| -> PathBuf { work_dir.join(format!("{label}.log")) };
    let spawn = |bin: &str, label: &str, node_env: Vec<(&'static str, String)>| -> Child {
        let file = std::fs::File::create(log_of(label)).expect("node log");
        vd_bins::spawn_node_grouped(Path::new(bin), &common, &node_env, file).expect("spawn node")
    };

    let mut nodes = Cluster::new();
    nodes.push(
        "vd-orchestrator",
        spawn(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            "vd-orchestrator",
            orchestrator_env(&addrs, &DEV, &orch_store, vd_bins::ClusterShape::Single),
        ),
    );
    nodes.push(
        "vd-gateway",
        spawn(
            env!("CARGO_BIN_EXE_vd-gateway"),
            "vd-gateway",
            gateway_env(
                &addrs,
                &dev_auth_pubkey_hex(),
                &DEV,
                vd_bins::ClusterShape::Single,
            ),
        ),
    );
    nodes.push(
        "vd-shard",
        spawn(env!("CARGO_BIN_EXE_vd-shard"), "vd-shard", shard_e),
    );
    let _nodes = nodes; // RAII: reaps the cluster on return or panic

    // ---- stand K real clients in the realm ------------------------------------
    let mut clients: Vec<KillOnDrop> = Vec::new();
    let mut ports: Vec<u16> = Vec::new();
    for i in 0..K {
        let quic = reserve_udp_addr();
        let devctl = reserve_tcp_addr().port();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
        for (k, v) in &common {
            cmd.env(k, v);
        }
        cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
        cmd.args([
            "--name",
            &format!("pace-{i}"),
            "--agent-index",
            &i.to_string(),
            "--gateway",
            &addrs.gateway.to_string(),
            "--client-quic",
            &quic.port().to_string(),
            "--trust-dir",
            &trust_dir.display().to_string(),
            "--dev-control",
            &devctl.to_string(),
            "--allow-dev-control",
        ]);
        clients.push(KillOnDrop(cmd.spawn().expect("spawn client")));
        ports.push(devctl);
    }
    let _clients = clients;

    let started = Instant::now();
    let deadline = convergence_deadline();
    loop {
        let active = ports
            .iter()
            .filter(|p| poll_state(**p).is_some_and(|s| s.phase == DevPhase::Active))
            .count();
        if active == usize::from(K) {
            break;
        }
        assert!(
            started.elapsed() < deadline,
            "{tag}: only {active} of {K} clients became Active"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // ---- read the pace of a shard doing ordinary work -------------------------
    let log = log_of("vd-shard");
    // Wait for one fresh line, which closes the window the logins fell in; measure only after it.
    let before = pace_windows(&log, 0).len();
    let waited = Instant::now();
    while pace_windows(&log, 0).len() == before {
        assert!(
            waited.elapsed() < deadline,
            "{tag}: the shard wrote no pace line after the logins"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    let mark = log_len(&log);
    std::thread::sleep(STAND);
    let windows = pace_windows(&log, mark);
    assert!(
        !windows.is_empty(),
        "{tag}: no pace window in the {STAND:?} the clients stood — a measurement, not a pass"
    );
    let mean_ms = windows.iter().map(|(m, _)| m).sum::<f64>() / windows.len() as f64;
    let slowest_ms = windows.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);

    let pace = Pace {
        mean_ms,
        slowest_ms,
        windows: windows.len(),
    };
    let _ = std::fs::remove_dir_all(&trust_dir);
    pace
}

#[test]
fn the_durable_outbox_does_not_slow_the_tick() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();

    let off = run_once("vd-pace-off", false);
    let on = run_once("vd-pace-on", true);
    let budget = tick_budget_ms();
    let ratio = on.mean_ms / off.mean_ms.max(f64::MIN_POSITIVE);
    println!(
        "tick pace with K={K} players standing, {STAND:?} per run, budget {budget:.2} ms\n  \
         outbox OFF: mean {:.3} ms, slowest {:.3} ms, {} windows\n  \
         outbox ON : mean {:.3} ms, slowest {:.3} ms, {} windows\n  \
         mean ON / mean OFF = {ratio:.3}",
        off.mean_ms, off.slowest_ms, off.windows, on.mean_ms, on.slowest_ms, on.windows,
    );

    let allowed = off.mean_ms * 1.5 + 1.0;
    assert!(
        on.mean_ms <= allowed,
        "the mean tick grew past the bound with the outbox on: {:.3} ms > {allowed:.3} ms \
         (off {:.3} ms, ratio {ratio:.3})",
        on.mean_ms,
        off.mean_ms
    );
    assert!(
        on.slowest_ms < budget,
        "a shard must never miss its tick because of the outbox: slowest {:.3} ms >= budget \
         {budget:.3} ms (off slowest {:.3} ms)",
        on.slowest_ms,
        off.slowest_ms
    );
}
