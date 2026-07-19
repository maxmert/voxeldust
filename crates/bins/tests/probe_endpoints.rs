//! Cloud-ready k3d Slice 3 — the real-binary proof of the k8s `/healthz` + `/readyz` probes.
//!
//! Asserts, against the ACTUAL server binaries over their real probe HTTP listeners (curled with the shared
//! `vd_bins::http_get_status`, which distinguishes 200 from 503 — `admin_get_body` cannot): (1) a full cluster
//! CONVERGES to Ready + Live; (2) an orchestrator alone is Ready WITHOUT a shard (the bootstrap-deadlock fix —
//! it keys on own-serving, not the whole-cluster `cluster_bootstrapped()` latch); (3) SIGTERM de-routes
//! `/readyz` (503) the instant it arrives while `/healthz` stays LIVE (the shutdown-edge publish + the
//! drain-safe `draining` bit); (4) a partitioned (orchestrator-killed) shard under ACTIVE D-3 goes NotReady
//! (its `RealmConfirmedAt` freezes past the grace) yet stays LIVE — never routed to a zombie, never restarted.
//! Tier-B process glue.

use std::process::Child;
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, DEV, common_env, dev_auth_pubkey_hex, gateway_env, http_get_status,
    orchestrator_env, reserve_tcp_addr, reserve_udp_addr, shard_env, spawn_node,
};
use vd_io_prod::trust::ClusterTrust;

fn addrs() -> ClusterAddrs {
    ClusterAddrs {
        orchestrator: reserve_udp_addr(),
        gateway: reserve_udp_addr(),
        shard: reserve_udp_addr(),
        admin: reserve_tcp_addr(),
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
        shard_b: reserve_tcp_addr(),
        shard_b_probe: reserve_tcp_addr(),
        galaxy: reserve_udp_addr(),
        galaxy_probe: reserve_tcp_addr(),
        planet: reserve_udp_addr(),
        planet_probe: reserve_tcp_addr(),
        station: reserve_udp_addr(),
        station_probe: reserve_tcp_addr(),
        area: reserve_udp_addr(),
        area_probe: reserve_tcp_addr(),
    }
}

fn write_trust(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vd-probe-{tag}-{}", std::process::id()));
    ClusterTrust::generate("vd-probe")
        .expect("trust")
        .write_der_dir(&dir)
        .expect("trust dir");
    dir
}

fn orch_store(tag: &str) -> String {
    let p = std::env::temp_dir().join(format!("vd-probe-{tag}-{}-orch.redb", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p.display().to_string()
}

/// Poll a probe endpoint until it returns `want`, or the deadline. Returns the last observed status.
fn poll_status(
    addr: std::net::SocketAddr,
    path: &str,
    want: u16,
    timeout: Duration,
) -> Option<u16> {
    let deadline = Instant::now() + timeout;
    let mut last = None;
    while Instant::now() < deadline {
        last = http_get_status(addr, path, Some(Duration::from_secs(1)));
        if last == Some(want) {
            return last;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    last
}

#[test]
fn cluster_converges_to_ready_and_all_endpoints_serve() {
    let addrs = addrs();
    let trust = write_trust("conv");
    let store = orch_store("conv");
    let common = common_env(&trust.display().to_string(), &DEV);
    let auth = dev_auth_pubkey_hex();

    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(&addrs, &DEV, &store, vd_bins::ClusterShape::Single),
        )
        .expect("spawn orch"),
    );
    cluster.push(
        "vd-gateway",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            &common,
            &gateway_env(&addrs, &[], &auth, &DEV, vd_bins::ClusterShape::Single),
        )
        .expect("spawn gateway"),
    );
    cluster.push(
        "vd-shard",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-shard"),
            &common,
            &shard_env(&addrs, &DEV, vd_bins::ClusterShape::Single),
        )
        .expect("spawn shard"),
    );
    let _guard = cluster;

    // Each node becomes READY: orchestrator (own-serving) at once, shard after ClockSync + the realm grant
    // (held, since dev D-3 is inert), gateway after ClockSync.
    for (name, addr) in [
        ("orchestrator", addrs.orchestrator_probe),
        ("gateway", addrs.gateway_probe),
        ("shard", addrs.shard_probe),
    ] {
        assert_eq!(
            poll_status(addr, "/readyz", 200, Duration::from_secs(15)),
            Some(200),
            "{name} /readyz must converge to 200"
        );
        // And LIVE.
        assert_eq!(
            http_get_status(addr, "/healthz", Some(Duration::from_secs(2))),
            Some(200),
            "{name} /healthz must be 200"
        );
    }
    let _ = std::fs::remove_dir_all(&trust);
    let _ = std::fs::remove_file(&store);
}

#[test]
fn orchestrator_alone_is_ready_without_a_shard() {
    // The bootstrap-deadlock fix: the orchestrator's readiness is its OWN serving capability, NOT the
    // whole-cluster cluster_bootstrapped() latch — so it is Ready with NO shard registered (otherwise a
    // readiness-gated Service would never let the first shard reach it, and the cluster could never warm).
    let addrs = addrs();
    let trust = write_trust("solo");
    let store = orch_store("solo");
    let common = common_env(&trust.display().to_string(), &DEV);

    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(&addrs, &DEV, &store, vd_bins::ClusterShape::Single),
        )
        .expect("spawn orch"),
    );
    let _guard = cluster;

    assert_eq!(
        poll_status(
            addrs.orchestrator_probe,
            "/readyz",
            200,
            Duration::from_secs(10)
        ),
        Some(200),
        "the orchestrator must be Ready without any shard (no bootstrap deadlock)"
    );
    assert_eq!(
        http_get_status(
            addrs.orchestrator_probe,
            "/healthz",
            Some(Duration::from_secs(2))
        ),
        Some(200)
    );
    let _ = std::fs::remove_dir_all(&trust);
    let _ = std::fs::remove_file(&store);
}

#[test]
fn sigterm_de_routes_readyz_while_healthz_stays_live() {
    // SIGTERM (the k8s pod-stop signal) must flip /readyz to 503 IMMEDIATELY (the shutdown-edge publish — the
    // loop-head `!shutdown` would otherwise skip the in-body publish on the exit tick), while /healthz stays
    // LIVE through the drain (the `draining` bit — no SIGKILL mid-fsync). Test on the gateway (no durable
    // store, so it drains fast and does not race the reap).
    let addrs = addrs();
    let trust = write_trust("term");
    let store = orch_store("term");
    let common = common_env(&trust.display().to_string(), &DEV);
    let auth = dev_auth_pubkey_hex();

    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(&addrs, &DEV, &store, vd_bins::ClusterShape::Single),
        )
        .expect("spawn orch"),
    );
    // The gateway as a STANDALONE child so we can grab its pid + SIGTERM it (Cluster only offers SIGKILL).
    // A drain LINGER holds it in Terminating for 1.5 s after de-routing, so the 503 window is observable
    // (without the linger a store-less gateway exits within ~1 tick, faster than a probe poll).
    let mut gw_env = gateway_env(&addrs, &[], &auth, &DEV, vd_bins::ClusterShape::Single);
    gw_env.push(("VD_SHUTDOWN_LINGER_MS", "1500".to_owned()));
    let mut gateway: Child =
        spawn_node(env!("CARGO_BIN_EXE_vd-gateway"), &common, &gw_env).expect("spawn gateway");
    let _guard = cluster;

    // Wait until the gateway is Ready, then SIGTERM it.
    assert_eq!(
        poll_status(addrs.gateway_probe, "/readyz", 200, Duration::from_secs(15)),
        Some(200),
        "gateway must be Ready before the SIGTERM"
    );
    let pid = gateway.id() as libc::pid_t;
    // SAFETY: `pid` is a live child we own (not yet reaped); SIGTERM is a valid signal.
    assert_eq!(
        unsafe { libc::kill(pid, libc::SIGTERM) },
        0,
        "SIGTERM the gateway"
    );

    // /readyz must go 503 (de-routed) while the pod is still draining; /healthz must stay 200 (drain-safe).
    assert_eq!(
        poll_status(addrs.gateway_probe, "/readyz", 503, Duration::from_secs(5)),
        Some(503),
        "SIGTERM must de-route /readyz to 503 during the drain"
    );
    assert_eq!(
        http_get_status(
            addrs.gateway_probe,
            "/healthz",
            Some(Duration::from_secs(2))
        ),
        Some(200),
        "/healthz must stay LIVE through the drain (no SIGKILL mid-fsync)"
    );
    let _ = gateway.wait();
    let _ = std::fs::remove_dir_all(&trust);
    let _ = std::fs::remove_file(&store);
}

#[test]
fn a_partitioned_shard_goes_not_ready_but_stays_live() {
    // THE partition-aware proof (the CRITICAL flap-fix's payoff): with an ACTIVE self-fence (grace/recheck),
    // the shard's readiness keys on RealmConfirmedAt staleness. Kill the orchestrator ⇒ the recheck round-trip
    // stops re-arming RealmConfirmedAt ⇒ after the grace it is stale ⇒ /readyz 503 (never route to a zombie),
    // while /healthz stays 200 (still ticking ⇒ do NOT restart, just de-route). Only the SHARD gets active D-3
    // (grace + recheck); the orchestrator would fail validate_self_fence_cadence with a grace but no recheck.
    let addrs = addrs();
    let trust = write_trust("part");
    let store = orch_store("part");
    let common = common_env(&trust.display().to_string(), &DEV);

    let mut shard_e = shard_env(&addrs, &DEV, vd_bins::ClusterShape::Single);
    // A small active self-fence: grace 20 ticks (~0.4 s @50Hz), recheck 8 (grace >= 2*recheck). RealmConfirmedAt
    // re-arms every recheck round-trip while the orchestrator is alive; freezes when it dies.
    shard_e.push(("VD_SELF_FENCE_GRACE", "20".to_owned()));
    shard_e.push(("VD_REALM_RECHECK", "8".to_owned()));

    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &common,
            &orchestrator_env(&addrs, &DEV, &store, vd_bins::ClusterShape::Single),
        )
        .expect("spawn orch"),
    );
    cluster.push(
        "vd-shard",
        spawn_node(env!("CARGO_BIN_EXE_vd-shard"), &common, &shard_e).expect("spawn shard"),
    );
    let mut guard = cluster;

    // Shard converges to Ready (clock synced + realm confirmed-fresh).
    assert_eq!(
        poll_status(addrs.shard_probe, "/readyz", 200, Duration::from_secs(15)),
        Some(200),
        "the shard must be Ready before the partition"
    );
    // Partition: kill the orchestrator. The shard's RealmConfirmedAt stops re-arming.
    guard.kill_and_reap("vd-orchestrator");

    // After the grace elapses the shard self-de-routes (/readyz 503) but stays LIVE (/healthz 200).
    assert_eq!(
        poll_status(addrs.shard_probe, "/readyz", 503, Duration::from_secs(10)),
        Some(503),
        "a partitioned shard must go NotReady (RealmConfirmedAt frozen past the grace)"
    );
    assert_eq!(
        http_get_status(addrs.shard_probe, "/healthz", Some(Duration::from_secs(2))),
        Some(200),
        "a partitioned-but-ticking shard must stay LIVE (de-route, do NOT restart)"
    );
    let _ = std::fs::remove_dir_all(&trust);
    let _ = std::fs::remove_file(&store);
}
