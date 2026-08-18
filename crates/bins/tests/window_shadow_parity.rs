//! THE WINDOW LANE's COMPOSED SELF-CONSISTENCY GATE, in real processes.
//!
//! WHAT THIS FILE USED TO BE, and why it is not that any more. Through Slice B it was the
//! SHADOW-PARITY gate (§4 Slice B, §4.5 Topic 5): the composition engine ran LIVE beside the old
//! inter-realm scenery lanes, and its picture was measured against the old-lane client feed, per
//! realm per tick. That measurement RAN and is RECORDED (its last green reading is in the
//! D-WINDOW-1 ledger), and it discharged its whole purpose — "shadow parity before any client
//! cut" (§4.5 Topic 5) — when Slice C1 cut the client over. Slice C2 then DELETED the lanes it
//! compared against, so its left-hand side no longer exists: §4's Slice-C2 gate list is
//! `intershard_closed` pins + a green suite + coverage, with no parity entry.
//!
//! THE SCENARIO IS NOT DELETED, IT IS RE-BASED — because three §2.12 NAMED INVARIANTS have no
//! other process-tier home and would have died with the comparator:
//!   * the EXACT-CADENCE pin — `window_full_chain_folds > 0` is direct proof two different shard
//!     processes stamped window levels at IDENTICAL universe ticks;
//!   * the DEDUP f64 agreement, both halves — hop-vs-child-row (`window_dedup_disagree == 0`,
//!     max deviation printed) and shared-vs-per-session fold (`window_fold_divergence == 0`,
//!     non-vacuous because `window_fold_hits > 0` with two co-located sessions);
//!   * the SHEAR LAW live — `window_instant_mismatch == 0` across the whole run (its CAN-fail
//!     half is the deliberate mixed-tick unit in `vd-connection-plane::window`).
//!
//! It also now asserts what the deletion promises: the tombstoned lanes are SILENT in a real
//! cluster (`old_realm_frames_dropped == 0`, `old_scene_deltas_dropped == 0` — a revived producer
//! shows up here, loudly).
//!
//! The run is unchanged, and it is the rlm_demand_login pattern: a REAL demand cluster
//! (orchestrator + gateway, no shard pre-booked), TWO real dev-control clients logging into the
//! same home star (two sessions sharing one origin — the §2.14 shared-fold measurement is
//! non-vacuous), a real flight leg (the shared rendezvous INTO the inner planet — a crossing, so
//! the chain re-derives, the epoch bumps, and the parent level composes through the
//! lineage-derived Child window), and a dwell on the planet. The proof is read from the gateway's
//! `/admin/snapshot`.
//!
//! `--test-threads=1` + the RLM demand port band (this file boots the same demand topology).
#![cfg(feature = "dev-control")]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::rendezvous_into_planet;
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr, world_roster,
};
use vd_core::NodeId;
use vd_devproto::{CLIENT_NODE_BASE, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::{AdminSnapshot, GatewayView};

/// Demand: the login must SPAWN real shard processes THEN converge (same budget as the
/// demand-login gate); the crossing leg gets the walk budget.
const LOGIN_DEADLINE: Duration = Duration::from_secs(60);
const FLIGHT_DEADLINE: Duration = Duration::from_secs(150);
/// A healthy client clears this in a second or two of 20 Hz once its home is up.
const SNAPSHOT_FLOOR: u64 = 5;
/// The anti-vacuity floor: the stationary settle alone composes rows at the 20 Hz realm-lane rate
/// across several planets — hundreds within a few seconds. Requiring this many COMPOSED rows makes
/// a composer that never folds a loud failure, while staying far under what any healthy run
/// produces. (It replaces the retired comparator's matched-row floor, at the same magnitude and
/// for the same reason.)
const COMPOSED_FLOOR: u64 = 200;

fn poll_state(port: u16) -> Option<DevState> {
    match dev_roundtrip(port, &DevRequest::State).ok()? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str::<AdminSnapshot>(&body).ok()?.gateway
}

struct Fixture {
    trust_dir: std::path::PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: std::path::PathBuf,
    store: std::path::PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_file(&self.store);
        let _ = std::fs::remove_file(&self.launch_path);
    }
}

fn fixture(tag: &str) -> Fixture {
    let trust = ClusterTrust::generate("vd-window-parity").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-window-parity-{tag}-{}", std::process::id()));
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let store = base.join("orchestrator.redb");
    let launch_path = store.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    let _ = std::fs::remove_file(&store);
    let _ = std::fs::remove_file(&launch_path);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    Fixture {
        store_str: store.display().to_string(),
        trust_dir,
        common,
        launch_path,
        store,
    }
}

/// SIGKILL + poll-until-gone every demand-spawned shard on drop (declared BEFORE the `Cluster`
/// so it drops AFTER it — the rlm_demand_login pattern, verbatim rationale).
struct ForkedReaper(std::path::PathBuf);
impl Drop for ForkedReaper {
    fn drop(&mut self) {
        if self.0.exists() {
            reap_forked(&launch_rows(&self.0));
        }
    }
}

fn spawn_client(
    f: &Fixture,
    gateway: SocketAddr,
    name: &str,
    agent_index: u32,
    quic_port: u16,
    devctl_port: u16,
) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.args([
        "--name",
        name,
        "--agent-index",
        &agent_index.to_string(),
        "--gateway",
        &gateway.to_string(),
        "--client-quic",
        &quic_port.to_string(),
        "--trust-dir",
        &f.trust_dir.display().to_string(),
        "--dev-control",
        &devctl_port.to_string(),
        "--allow-dev-control",
    ]);
    cmd.spawn().expect("spawn client")
}

fn await_active(devctl_port: u16, gateway_admin: SocketAddr, deadline: Duration) -> DevState {
    let started = Instant::now();
    loop {
        if let Some(st) = poll_state(devctl_port)
            && st.phase == DevPhase::Active
            && st.snapshots_applied >= SNAPSHOT_FLOOR
        {
            return st;
        }
        assert!(
            started.elapsed() < deadline,
            "the demand login never converged: last client state {:?}; gateway view {:?}",
            poll_state(devctl_port),
            gateway_view(gateway_admin),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// The gate's whole counter report, verbatim — every number by name, never a blur.
fn print_class_report(tag: &str, gw: &GatewayView) {
    eprintln!(
        "[window:{tag}] composer: folds={} fold_hits={} full_chain_folds={} composed_rows={} chains_held={}\n\
         [window:{tag}] composer: holds={} dead_hops={} stalled={} cycles={} unresolved_standing={}\n\
         [window:{tag}] shear/dedup: instant_mismatch={} fold_divergence={} dedup_disagree={} dedup_max_dev={} nm\n\
         [window:{tag}] ingest: rows_ingested={} level_refused={} body_stale={} body_preroster={} head_reads={}\n\
         [window:{tag}] egress: levels={} deltas={} datagrams={} relay_rows_composed={} \
         relay_descent_refused={} relay_stamp_missing={} relay_unrostered={} relay_skew_max={} relay_depth_max={}\n\
         [window:{tag}] DEAD LANES (must stay 0): old_realm_frames={} old_scene_deltas={}",
        gw.window_folds,
        gw.window_fold_hits,
        gw.window_full_chain_folds,
        gw.window_composed_rows,
        gw.window_chains_held,
        gw.window_compose_hold_ticks,
        gw.window_hop_dead,
        gw.window_t_monotone_stalled,
        gw.window_chain_cycle,
        gw.window_unresolved_standing,
        gw.window_instant_mismatch,
        gw.window_fold_divergence,
        gw.window_dedup_disagree,
        gw.window_dedup_max_dev_nm,
        gw.window_rows_ingested,
        gw.window_level_refused,
        gw.window_body_stale,
        gw.window_body_preroster,
        gw.window_head_reads_sent,
        gw.scene_levels_sent,
        gw.scene_deltas_sent,
        gw.scene_datagrams_sent,
        gw.window_relay_rows_composed,
        gw.window_relay_descent_refused,
        gw.window_relay_stamp_missing,
        gw.window_relay_unrostered,
        gw.window_relay_stamp_skew_ticks,
        gw.window_relay_depth_max,
        gw.old_realm_frames_dropped,
        gw.old_scene_deltas_dropped,
    );
}

/// Every assertion that must hold at EVERY probe point of the run (the gate's invariant half).
fn assert_no_unexplained(tag: &str, gw: &GatewayView) {
    assert_eq!(
        gw.window_instant_mismatch, 0,
        "[{tag}] the shear law fired on live data: a mixed-tick fold was attempted"
    );
    assert_eq!(
        gw.window_fold_divergence, 0,
        "[{tag}] §2.14 broken: a shared fold differed bit-level from its per-session recompute"
    );
    assert_eq!(
        gw.window_dedup_disagree, 0,
        "[{tag}] §2.12 broken: hop-derived and child-row-derived positions disagreed (max {} nm)",
        gw.window_dedup_max_dev_nm,
    );
    assert_eq!(
        gw.window_chain_cycle, 0,
        "[{tag}] a chain derivation cycled"
    );
    assert_eq!(gw.window_sender_mismatch, 0, "[{tag}] a forged window row");
    assert_eq!(gw.window_misauthored_body, 0, "[{tag}] a mis-authored body");
    // THE DELETION, in a real cluster: the tombstoned lanes have no producer left, so a shard
    // speaking either of them lands here — the audit trail the C2 ledger promises.
    assert_eq!(
        gw.old_realm_frames_dropped, 0,
        "[{tag}] a shard still speaks the TOMBSTONED realm datagram (window lane Slice C2)"
    );
    assert_eq!(
        gw.old_scene_deltas_dropped, 0,
        "[{tag}] a shard still speaks the TOMBSTONED per-observer scene delta (Slice C2)"
    );
    assert_eq!(
        gw.undecodable, 0,
        "[{tag}] undecodable frames at the gateway"
    );
}

/// THE GATE. Boot the demand cluster + two clients, settle at the star (the shared-fold
/// measurement), fly ONE client into the inner planet (the crossing leg), dwell on the planet
/// (the ≥2-level fold through the lineage-derived Child window), then assert the §2.12 pins:
/// the exact-cadence proof nonzero, the dedup and shear pins zero, the composer carrying real
/// volume — and the four deleted lanes silent in a real cluster.
#[test]
fn the_composed_picture_folds_one_chain_at_one_tick_with_the_dead_lanes_silent() {
    // FIRST statement: hold the process tier for the whole body (vd_bins::cluster_tier).
    let _tier = vd_bins::cluster_tier();
    let f = fixture("parity");
    let gw_admin = reserve_tcp_addr();
    let a = ClusterAddrs {
        gateway_admin: Some(gw_admin),
        ..ClusterAddrs::reserve()
    };
    let p: DevClusterParams = DEV;
    let quic_a = reserve_udp_addr();
    let quic_b = reserve_udp_addr();
    let devctl_a = reserve_tcp_addr().port();
    let devctl_b = reserve_tcp_addr().port();
    let client_book = [
        (NodeId(CLIENT_NODE_BASE), quic_a),
        (NodeId(CLIENT_NODE_BASE + 1), quic_b),
    ];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &f.common,
            &orchestrator_env(&a, &p, &f.store_str, ClusterShape::Demand),
        )
        .expect("spawn orchestrator"),
    );
    cluster.push(
        "vd-gateway",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            &f.common,
            &gateway_env(
                &a,
                &client_book,
                &dev_auth_pubkey_hex(),
                &p,
                ClusterShape::Demand,
            ),
        )
        .expect("spawn gateway"),
    );
    cluster.push(
        "client-a",
        spawn_client(&f, a.gateway, "parity-0", 0, quic_a.port(), devctl_a),
    );
    cluster.push(
        "client-b",
        spawn_client(&f, a.gateway, "parity-1", 1, quic_b.port(), devctl_b),
    );
    let _cluster = cluster;

    // ---- Phase 1: both real logins land at the star (two sessions, one origin realm). ----
    let st_a = await_active(devctl_a, gw_admin, LOGIN_DEADLINE);
    let st_b = await_active(devctl_b, gw_admin, LOGIN_DEADLINE);
    assert_eq!(st_a.location.as_deref(), Some("System 7"), "{st_a:?}");
    assert_eq!(st_b.location.as_deref(), Some("System 7"), "{st_b:?}");

    // ---- Phase 2: the stationary settle — poll until the comparator has REAL volume. ----
    let settle_deadline = Instant::now() + Duration::from_secs(30);
    let settled = loop {
        let gw = gateway_view(gw_admin).expect("gateway admin snapshot");
        if (gw.window_composed_rows >= COMPOSED_FLOOR) & (gw.window_folds > 0) {
            break gw;
        }
        assert!(
            Instant::now() < settle_deadline,
            "the composer never reached the composed-row floor ({COMPOSED_FLOOR}): {gw:?}"
        );
        std::thread::sleep(Duration::from_millis(250));
    };
    print_class_report("settled", &settled);
    assert_no_unexplained("settled", &settled);
    assert!(
        settled.window_fold_hits > 0,
        "two sessions share one origin: the §2.14 shared fold must have served the second \
         session at least once (non-vacuous fold-divergence measurement): {settled:?}"
    );

    // ---- Phase 3: the flight leg — client A crosses into the inner planet (full orbit speed,
    // the ONE shared rendezvous), while client B keeps watching from the star. ----
    let roster = world_roster(&p);
    let (planet, elements) = (roster.inner, roster.inner_elements);
    rendezvous_into_planet(devctl_a, &DEV, planet, &elements, FLIGHT_DEADLINE);
    eprintln!("[window] client A crossed into {planet:?}; dwelling for the ≥2-level fold");

    // ---- Phase 4: the on-planet dwell — the lineage-derived Child window on the parent joins
    // the leaf's own window, so the composer folds a ≥2-level chain. Poll until the FULL-CHAIN
    // fold count moves (the exact-cadence pin's direct proof: a ≥2-level fold exists only when
    // two shard processes stamped identical universe ticks). ----
    let full_chain_before = settled.window_full_chain_folds;
    let dwell_deadline = Instant::now() + Duration::from_secs(30);
    let dwelled = loop {
        let gw = gateway_view(gw_admin).expect("gateway admin snapshot");
        if gw.window_full_chain_folds > full_chain_before {
            break gw;
        }
        assert!(
            Instant::now() < dwell_deadline,
            "EXACT-CADENCE PIN / CHAIN PROOF failed: no ≥2-level fold landed after the crossing \
             — either the shards stamp levels on diverged cadences (no common tick exists) or \
             the lineage-derived Child window never confirmed: {gw:?}"
        );
        std::thread::sleep(Duration::from_millis(250));
    };
    std::thread::sleep(Duration::from_secs(5)); // accumulate on-planet compose volume
    let final_gw = gateway_view(gw_admin).expect("gateway admin snapshot");
    print_class_report("final", &final_gw);
    assert_no_unexplained("final", &final_gw);
    assert!(
        final_gw.window_composed_rows > dwelled.window_composed_rows,
        "the composer kept folding THROUGH the on-planet dwell: {final_gw:?}"
    );
    assert!(
        final_gw.window_full_chain_folds > full_chain_before,
        "the exact-cadence pin held through the dwell: {final_gw:?}"
    );
    assert!(
        final_gw.window_rows_ingested > 0 && final_gw.window_composed_rows > 0,
        "the engine consumed and composed real volume: {final_gw:?}"
    );
    eprintln!(
        "[window] GREEN: {} rows composed across {} folds; full-chain folds {} (the exact-cadence \
         pin), dedup max dev {} nm with 0 disagreements, fold divergence 0, shear mismatches 0; \
         the four deleted scenery lanes stayed silent (old realm frames 0, old scene deltas 0).",
        final_gw.window_composed_rows,
        final_gw.window_folds,
        final_gw.window_full_chain_folds,
        final_gw.window_dedup_max_dev_nm,
    );
}
