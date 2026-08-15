//! THE WINDOW LANE's SHADOW-PARITY GATE (Slice B — `docs/design/window_lane.md` §4 Slice B,
//! §4.5 Topic 5): the composition engine runs LIVE in real processes beside the old lane, and its
//! composed picture is MEASURED against the old-lane client feed — per realm, per tick, inside
//! the gateway that forwards both. Zero UNEXPLAINED mismatches; every explained divergence is a
//! NAMED CLASS with a count, printed below (classes, not blur — the owner's Topic-5 approval).
//!
//! The run is the rlm_demand_login pattern: a REAL demand cluster (orchestrator + gateway, no
//! shard pre-booked), TWO real dev-control clients logging into the same home star (two sessions
//! sharing one origin — the §2.14 shared-fold measurement is non-vacuous), a real flight leg
//! (the shared rendezvous INTO the inner planet — a crossing, so the chain re-derives, the epoch
//! bumps, and the parent level composes through the lineage-derived Child window while the old
//! cascade feeds the same rows), and a dwell on the planet (ancestor-row parity under the
//! cascade). The proof is read from the gateway's `/admin/snapshot`.
//!
//! The gate also carries three §2.12 pins the design demands land HERE:
//! - the EXACT-CADENCE boot pin: `window_full_chain_folds > 0` is direct proof two different
//!   shard processes stamped window levels at IDENTICAL universe ticks (a fold across a ≥2-level
//!   chain exists only at a shared stamp) — loud failure if per-realm cadence ever diverges;
//! - the DEDUP f64 agreement, both halves MEASURED: hop-vs-child-row (`window_dedup_disagree`
//!   == 0, max deviation printed) and shared-vs-per-session fold (`window_fold_divergence` == 0,
//!   non-vacuous because `window_fold_hits > 0` with the two co-located sessions);
//! - the shear law live: `window_instant_mismatch == 0` across the whole run (its CAN-fail half
//!   is the deliberate mixed-tick unit in `vd-connection-plane::window`).
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
/// The parity anti-vacuity floor: the stationary settle alone produces old-lane realm rows at
/// the 20 Hz realm-lane rate across several planets — hundreds within a few seconds. Requiring
/// this many MATCHED rows makes a silently idle comparator (or a composer that never folds) a
/// loud failure, while staying far under what any healthy run produces.
const MATCHED_FLOOR: u64 = 200;

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

/// The gate's whole class report, verbatim — §4.5 Topic 5: CLASSES, each by name, never a blur.
fn print_class_report(tag: &str, gw: &GatewayView) {
    eprintln!(
        "[parity:{tag}] MEASURED against the live old-lane feed, per realm per tick:\n\
         [parity:{tag}]   matched (bit-identical position at the row's own tick) = {}\n\
         [parity:{tag}]   UNEXPLAINED pose_mismatch = {} (max deviation {} nm)\n\
         [parity:{tag}]   UNEXPLAINED missing_composed = {}\n\
         [parity:{tag}]   EXCLUDED BY NAME sibling_interior (the recorded Q2-carrier gap, D-WINDOW-1) = {}\n\
         [parity:{tag}]   explained unwindowed_ancestor (head poll in flight) = {}\n\
         [parity:{tag}]   explained no_fold_at_tick (boot / ring-aged stamp) = {}\n\
         [parity:{tag}]   explained offframe_rows (crossing-overlap dual feed) = {}\n\
         [parity:{tag}]   origin_rows (SL1 self-filter regression signal) = {}\n\
         [parity:{tag}]   pending_shed = {}  parity_undecodable = {}\n\
         [parity:{tag}] composer: folds={} fold_hits={} full_chain_folds={} composed_rows={} chains_held={}\n\
         [parity:{tag}] composer: holds={} dead_hops={} stalled={} cycles={} unresolved_standing={}\n\
         [parity:{tag}] shear/dedup: instant_mismatch={} fold_divergence={} dedup_disagree={} dedup_max_dev={} nm\n\
         [parity:{tag}] ingest: rows_ingested={} level_refused={} body_stale={} body_preroster={} head_reads={}",
        gw.parity_rows_matched,
        gw.parity_pose_mismatch,
        gw.parity_max_pos_dev_nm,
        gw.parity_missing_composed,
        gw.parity_sibling_interior_excluded,
        gw.parity_unwindowed_ancestor,
        gw.parity_no_fold_at_tick,
        gw.parity_offframe_rows,
        gw.parity_origin_rows,
        gw.parity_pending_shed,
        gw.parity_undecodable,
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
    );
}

/// Every assertion that must hold at EVERY probe point of the run (the gate's invariant half).
fn assert_no_unexplained(tag: &str, gw: &GatewayView) {
    assert_eq!(
        gw.parity_pose_mismatch, 0,
        "[{tag}] UNEXPLAINED parity class: composed position differed from the old lane at the \
         same (realm, tick) — max deviation {} nm. The fold is not reproducing the cascade.",
        gw.parity_max_pos_dev_nm,
    );
    assert_eq!(
        gw.parity_missing_composed, 0,
        "[{tag}] UNEXPLAINED parity class: a realm a held window STATES was absent from its \
         tick's fold — the composer lost a row the old lane carried."
    );
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
    assert_eq!(
        gw.parity_undecodable, 0,
        "[{tag}] an undecodable old-lane body"
    );
    assert_eq!(
        gw.undecodable, 0,
        "[{tag}] undecodable frames at the gateway"
    );
}

/// THE GATE. Boot the demand cluster + two clients, settle at the star (stationary parity +
/// the shared-fold measurement), fly ONE client into the inner planet (the crossing leg), dwell
/// (ancestor-row parity through the lineage-derived Child window vs the live cascade), then
/// assert the class ledger: zero unexplained mismatches, the named exclusion printed, the
/// exact-cadence and dedup pins nonzero/zero as designed.
#[test]
fn the_shadow_composed_picture_reproduces_the_old_lane_with_zero_unexplained_mismatches() {
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
        if (gw.parity_rows_matched >= MATCHED_FLOOR) & (gw.window_folds > 0) {
            break gw;
        }
        assert!(
            Instant::now() < settle_deadline,
            "the comparator never reached the matched floor ({MATCHED_FLOOR}): {gw:?}"
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
    eprintln!("[parity] client A crossed into {planet:?}; dwelling for ancestor-row parity");

    // ---- Phase 4: the on-planet dwell — the star's cascade feeds the old lane while the
    // lineage-derived Child window feeds the composer; both must say the same rows. Poll until
    // the FULL-CHAIN fold count moves (the exact-cadence pin's direct proof: a ≥2-level fold
    // exists only when two shard processes stamped identical universe ticks). ----
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
    std::thread::sleep(Duration::from_secs(5)); // accumulate on-planet parity volume
    let final_gw = gateway_view(gw_admin).expect("gateway admin snapshot");
    print_class_report("final", &final_gw);
    assert_no_unexplained("final", &final_gw);
    assert!(
        final_gw.parity_rows_matched > dwelled.parity_rows_matched,
        "the comparator kept matching THROUGH the on-planet dwell: {final_gw:?}"
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
        "[parity] GREEN: {} rows matched bit-identically, 0 unexplained mismatches; \
         sibling-interior exclusion class counted {} (the recorded Q2-carrier gap, D-WINDOW-1); \
         full-chain folds {} (exact-cadence pin), dedup max dev {} nm, fold divergence 0.",
        final_gw.parity_rows_matched,
        final_gw.parity_sibling_interior_excluded,
        final_gw.window_full_chain_folds,
        final_gw.window_dedup_max_dev_nm,
    );
}
