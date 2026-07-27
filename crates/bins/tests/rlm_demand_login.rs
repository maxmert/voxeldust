//! RLM — THE demand-loop PROCESS proofs: the demand-LOGIN (RG-4c) + the demand-WALK.
//!
//! Boots a DEMAND cluster in real OS processes — orchestrator + gateway ONLY, with NO shard pre-booked
//! ([`ClusterShape::Demand`]) — then logs in ONE real dev-control client. The design pass (wf_5c3f7fbb) found
//! this path had NEVER run outside a private in-process test helper; this gate proves it end to end:
//!
//!   login → the gateway's dynamic-home route emits a `RealmDemand` → the orchestrator's ARMED reconciler
//!   spawns the home realm shard in a fresh process → that shard greets its booked peers (the reactive
//!   greeting, RG-0..3) so the gateway learns its return connection with NO pre-booked address → the gateway
//!   routes the login onto the spawn-minted node → the client goes Active and receives snapshots.
//!
//! The proof is read from the gateway's `/admin/snapshot` (the RG-4a observability): a nonzero
//! `dynamic_shards` means a login routed to a spawn-minted node (not a static `config.shard`), a nonzero
//! `presence_announces` means the demand shard's reactive greeting arrived, and `home_bootstrap_timeouts ==
//! 0` + `undecodable == 0` mean it converged cleanly. The orchestrator's `rlm.spins_requested` corroborates.
//!
//! `--test-threads=1` + a DEDICATED RLM port band ([`RLM_DEMAND_FIRST_PORT`]) so the demand-spawned shards'
//! deterministic ports never collide with `rlm_kill9_spawn`'s band when the two binaries run in parallel.
//!
//! Gated on `dev-control`: it drives the client through its loopback listener.
#![cfg(feature = "dev-control")]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr,
};
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_devproto::{CLIENT_NODE_BASE, DevEntityRow, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::{AdminSnapshot, GatewayView, RlmView};

/// Demand: the login must SPAWN a real shard process THEN converge — a longer budget than a static login.
const DEADLINE: Duration = Duration::from_secs(60);
/// A healthy client clears this in a second or two of 20 Hz once its home is up — below it is a stall.
const SNAPSHOT_FLOOR: u64 = 5;

fn devctl(port: u16, request: &DevRequest) -> Option<DevResponse> {
    dev_roundtrip(port, request).ok()
}

fn poll_state(port: u16) -> Option<DevState> {
    match devctl(port, &DevRequest::State)? {
        DevResponse::State { state } => Some(state),
        _ => None,
    }
}

fn admin(addr: SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    admin(addr)?.gateway
}

/// A full `ClusterAddrs` for a demand cluster: the orchestrator + gateway binds + admin endpoints. The
/// `shard`/`shard_b`/… fields are UNUSED (no static shard is spawned — the demand-spawned shard mints its own
/// ports from the RLM band), but the struct is total, so every field gets a fresh reservation. `gateway_admin`
/// is `Some` so the gateway serves the `/admin/snapshot` this test reads the proof from.
fn demand_addrs(gateway_admin: SocketAddr) -> ClusterAddrs {
    ClusterAddrs {
        orchestrator: reserve_udp_addr(),
        gateway: reserve_udp_addr(),
        shard: reserve_udp_addr(),
        admin: reserve_tcp_addr(),
        gateway_admin: Some(gateway_admin),
        orchestrator_probe: reserve_tcp_addr(),
        gateway_probe: reserve_tcp_addr(),
        shard_probe: reserve_tcp_addr(),
        shard_b: reserve_udp_addr(),
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

/// A test-private trust dir + the orchestrator's durable store + its `launch.redb` sidecar. Cleaned on drop.
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
    let trust = ClusterTrust::generate("vd-rlm-demand").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-rlm-demand-{tag}-{}", std::process::id()));
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

/// SIGKILL + poll-until-gone every demand-spawned shard the orchestrator forked, on drop. Declared BEFORE the
/// `Cluster` so it drops AFTER it — by then the `Cluster` has killed + reaped the orchestrator (releasing the
/// `launch.redb` lock), so `launch_rows` can reopen the ledger to name the forked pids. This is the panic-safe
/// cleanup: the forked shards lead their OWN process groups, so the `Cluster`'s group-kill never reaches them.
struct ForkedReaper(std::path::PathBuf);
impl Drop for ForkedReaper {
    fn drop(&mut self) {
        if self.0.exists() {
            reap_forked(&launch_rows(&self.0));
        }
    }
}

/// Spawn one real dev-control client that logs into `gateway` and drives itself through its loopback listener.
fn spawn_client(f: &Fixture, gateway: SocketAddr, quic_port: u16, devctl_port: u16) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.args([
        "--name",
        "demand-0",
        "--agent-index",
        "0",
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

/// Poll the client's dev-control listener until it is Active AND receiving, or fail LOUD with the last client
/// state + the gateway's view (so a stall names WHY — never went Active, or Active but starved).
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

/// Push a fresh demand gateway onto `cluster` — the SAME env each time, so a restart re-binds the SAME addrs.
fn push_demand_gateway(
    cluster: &mut Cluster,
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    client_book: &[(NodeId, SocketAddr)],
) {
    cluster.push(
        "vd-gateway",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            &f.common,
            &gateway_env(a, client_book, &dev_auth_pubkey_hex(), p, ClusterShape::Demand),
        )
        .expect("spawn gateway"),
    );
}

/// Boot a demand cluster (orchestrator + gateway) + one dev-control client — NO shard pre-booked. The only
/// way to a world is the armed reconciler spinning one up on the client's login demand (+ its AoI as it moves).
fn boot_demand_login(
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    client_book: &[(NodeId, SocketAddr)],
    client_quic: SocketAddr,
    devctl_port: u16,
) -> Cluster {
    let mut cluster = Cluster::new();
    cluster.push(
        "vd-orchestrator",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-orchestrator"),
            &f.common,
            &orchestrator_env(a, p, &f.store_str, ClusterShape::Demand),
        )
        .expect("spawn orchestrator"),
    );
    push_demand_gateway(&mut cluster, f, a, p, client_book);
    cluster.push("client", spawn_client(f, a.gateway, client_quic.port(), devctl_port));
    cluster
}

#[test]
fn a_demand_login_spins_up_a_fresh_home_and_lands_with_no_prebooked_shard() {
    let f = fixture("login");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    // The forked-shard reaper drops LAST (declared first) — after the Cluster has reaped the orchestrator.
    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &DEV, &client_book, client_quic, devctl_port);

    // ---- THE proof: the login went Active, and it did so by spawning + reaching a fresh home shard. ----
    let state = await_active(devctl_port, gw_admin, DEADLINE);
    assert!(state.own_entity.is_some(), "no authority after login: {state:?}");
    assert_eq!(state.decode_errors, 0, "decode faults: {state:?}");
    assert_eq!(state.foreign_peer_drops, 0, "cross-wired peer: {state:?}");

    let gw = gateway_view(gw_admin).expect("the demand gateway serves its admin snapshot");
    assert!(
        gw.dynamic_shards >= 1,
        "the login must have routed to a DEMAND-SPAWNED node (not a static shard): {gw:?}",
    );
    assert!(
        gw.presence_announces >= 1,
        "the demand shard's reactive greeting must have reached the gateway: {gw:?}",
    );
    assert_eq!(
        gw.home_bootstrap_timeouts, 0,
        "the home realm bootstrapped within its TTL (no timeout): {gw:?}",
    );
    assert_eq!(gw.undecodable, 0, "no undecodable frames: {gw:?}");
    assert_eq!(gw.sessions_open, 1, "exactly the one login session: {gw:?}");

    let orch = admin(a.admin).expect("the orchestrator serves its admin snapshot");
    assert!(
        orch.rlm.spins_requested >= 1,
        "the armed reconciler requested at least one spawn: {:?}",
        orch.rlm,
    );
    assert_eq!(
        orch.rlm.spins_failed, 0,
        "no spawn failed: {:?}",
        orch.rlm
    );
}

#[test]
fn a_gateway_restart_re_learns_the_running_demand_shard_via_the_reactive_greeting() {
    // The reactive greeting's SELF-HEAL property (RG-1): the connection between a demand-spawned shard and the
    // gateway is learned (not pre-booked), so it must survive a GATEWAY restart with no manual re-plumb. This
    // is the SERVER-side re-heal — the client's session does NOT resume (reconnect-without-replay is D-11,
    // deferred), but the still-running shard must re-teach a FRESH gateway where it is.
    let f = fixture("reheal");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];

    let _reaper = ForkedReaper(f.launch_path.clone());
    let mut cluster = boot_demand_login(&f, &a, &DEV, &client_book, client_quic, devctl_port);

    // The login spawned a home shard + the gateway learned it via the reactive greeting.
    await_active(devctl_port, gw_admin, DEADLINE);
    let before = gateway_view(gw_admin).expect("gateway view");
    assert!(
        before.presence_announces >= 1 && before.dynamic_shards >= 1,
        "the demand shard is up + learned before the restart: {before:?}",
    );

    // SIGKILL + reap the gateway, then boot a FRESH one on the SAME addrs (a fresh process ⇒ its
    // presence_announces resets to 0). The still-running demand shard is UNAFFECTED — it leads its OWN process
    // group, so the Cluster's per-child kill never touched it. D-34: we restart the GATEWAY (the learner), not
    // the shard.
    cluster.kill_and_reap("vd-gateway");
    push_demand_gateway(&mut cluster, &f, &a, &DEV, &client_book);
    let _cluster = cluster;

    // THE re-heal proof: the fresh gateway RE-LEARNS the still-running shard via the greeting ALONE. The shard
    // greets-on-silence (RG-1) — having stopped hearing from the old gateway it re-dials the SAME booked
    // address + re-sends its ShardPresence, so the fresh gateway learns the return connection with NO
    // pre-booked address + NO re-login. `presence_announces` climbs from 0 back to >= 1.
    let started = Instant::now();
    loop {
        if let Some(gw) = gateway_view(gw_admin)
            && gw.presence_announces >= 1
        {
            break;
        }
        assert!(
            started.elapsed() < DEADLINE,
            "the restarted gateway never re-learned the demand shard via the reactive greeting: {:?}",
            gateway_view(gw_admin),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

// The bootstrap-TTL fail-safe leg (a demand login whose home never boots must be CLOSED at
// `bootstrap_ttl_ticks`, not hang) is DEFERRED — see DEFERRED.md D-RLM-9. An empirical spike here found the
// obvious inductions do NOT reach the TTL: with no orchestrator (or the gateway dropped from its peers) the
// gateway never CLOCK-SYNCS, so its seed injector stays inert and no demand is ever emitted — the login sits
// with an open session and `home_bootstrap_timeouts == 0` (a separate pre-sync-hold concern). The TTL fires
// only for a SYNCED gateway whose home genuinely fails to boot, which CA-1 learning (the orchestrator learns
// the gateway from the demand frame and delivers the grant anyway) makes delicate to induce.

// ---- The demand-WALK: a MOVING occupant drives realm spin-up AHEAD + tear-down BEHIND -------------------

/// Walk-demand params: a NON-ZERO predictive AoI horizon so a child realm spins up + boots BEFORE the walking
/// occupant crosses its boundary. At 50 Hz, 150 ticks ≈ 3 s of look-ahead: Planet 7's spin-up band edge sits
/// at x≈8 (extent 10 × the 1.2 walk-demand factor), so a +X walker at ~2 m/s predicts into it around x≈2 —
/// ~4 s of walk to the physical boundary (x=10), covering the ~2–3 s the child needs to spawn + greet + route.
fn walk_params() -> DevClusterParams {
    DevClusterParams {
        boot_ticks_p99: 150,
        ..DEV
    }
}

fn own_row(state: &DevState) -> Option<&DevEntityRow> {
    let own = state.own_entity.as_deref()?;
    state.entities.iter().find(|r| r.entity == own)
}

fn own_pos(state: &DevState) -> Option<DVec3> {
    own_row(state).map(|r| DVec3::from_array(r.pos))
}

/// The orchestrator's RLM view (`spins_requested` / `spins_failed` / `teardowns_reaped` …); a zeroed view if
/// the admin endpoint blinks (a poll loop tolerates that).
fn orch_rlm(admin_addr: SocketAddr) -> RlmView {
    admin(admin_addr).map(|s| s.rlm).unwrap_or_default()
}

/// Drive ONE walk leg: `WalkTo` the world x-target on the +X axis, settle the sticky Move, read the delivered
/// own pose, and assert the occupant ARRIVED — movement never froze across the demand spin-up + cross-node
/// re-home. A generous `max_ticks` so a slow spawn+cross never times out mid-leg (the point is that the dot
/// keeps MOVING, and that the child booted in time to receive the cross — not raw speed).
fn walk_leg(devctl_port: u16, leg: &str, target_x: f64, max_ticks: u64) -> DVec3 {
    let target = DVec3::new(target_x, 0.0, 0.0);
    let _ = devctl(
        devctl_port,
        &DevRequest::WalkTo {
            target: target.to_array(),
            arrive_epsilon: 2.0,
            max_ticks,
        },
    )
    .unwrap_or_else(|| panic!("leg {leg}: no walk response"));
    let _ = devctl(
        devctl_port,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
    let landed = own_pos(&poll_state(devctl_port).expect("state after leg")).expect("own pos");
    assert!(
        (landed.x - target_x).abs() <= 2.0,
        "leg {leg} FROZE: the occupant stalled at x={:.3}, never reaching x={target_x} — the demand child did \
         not spin up + boot in time to receive the cross, or the re-home stalled.",
        landed.x,
    );
    landed
}

#[test]
fn a_walking_occupant_spins_up_a_child_ahead_crosses_in_then_the_vacated_realm_is_reaped() {
    // THE demand-WALK — warp as a consequence of movement. A logged-in occupant walks toward a CHILD realm
    // that does not exist yet; its Area-of-Interest (predictive) spins the child up AHEAD of it, the
    // orchestrator spawns it, the occupant CROSSES in, and when the occupant walks back out the vacated child
    // EMPTIES and its shard is reaped. No loading, no teleport — the demand loop driven by a moving player.
    let f = fixture("walk");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let p = walk_params();
    let client_quic = reserve_udp_addr();
    let devctl_port = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    // Spawn + walk + cross + the drain/quiesce teardown windows ⇒ a materially longer budget than a login.
    let deadline = Duration::from_secs(120);

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_login(&f, &a, &p, &client_book, client_quic, devctl_port);

    // The login lands the occupant in the home realm (System 7) at the origin.
    let landed = await_active(devctl_port, gw_admin, deadline);
    assert_eq!(
        landed.location.as_deref(),
        Some("System 7"),
        "the login lands in the home realm: {landed:?}",
    );
    let home_spins = orch_rlm(a.admin).spins_requested; // >= 1 (the home was demand-spawned)

    // ---- LEG 1: spin-up-AHEAD + cross-IN ----
    // Walk +X into Planet 7 (center +20, near-face x=10). The occupant's predictive AoI spins Planet 7 up while
    // it is still ~8 m short of the boundary, the orchestrator spawns it, and the occupant crosses in without
    // its movement freezing.
    walk_leg(devctl_port, "into-planet-7", 15.0, 2_000);
    let inside = poll_state(devctl_port).expect("state after leg 1");
    // The occupant's AUTHORITATIVE frame is now Planet 7 — it re-homed onto the demand-spawned child's shard
    // (the gateway routed it there). `location` changes ONLY on a real cross-realm move, so this IS the
    // spin-up-ahead + cross-in proof: the child could not have existed at login (NO shard was pre-booked).
    assert_eq!(
        inside.location.as_deref(),
        Some("Planet 7"),
        "the occupant crossed into the demand-spawned child realm: {inside:?}",
    );
    assert!(
        inside.own_entity.is_some(),
        "the occupant kept authority across the cross-in: {inside:?}",
    );
    assert_eq!(
        inside.decode_errors, 0,
        "the client saw no decode faults across the walk-in: {inside:?}",
    );
    let walk_rlm = orch_rlm(a.admin);
    assert!(
        walk_rlm.spins_requested > home_spins,
        "the WALK demanded a NEW child realm beyond the home (spins {} > {home_spins}): {walk_rlm:?}",
        walk_rlm.spins_requested,
    );
    assert_eq!(walk_rlm.spins_failed, 0, "no spawn failed: {walk_rlm:?}");
    // NOTE: the gateway's `dynamic_shards` counts LOGIN-home claims only (the `on_home_realm_head` resolve); a
    // crossing dest is admitted via the `PrepareSubscribe` path, so it is NOT counted there — `location`
    // above is the routing proof, not `dynamic_shards`.

    // ---- LEG 2: tear-down-BEHIND ----
    // Walk back to the origin (into System 7). The occupant leaves Planet 7 → it self-reports Empty → the
    // reconciler drains + reaps it, while the still-occupied home (System 7) stays up.
    walk_leg(devctl_port, "back-to-system-7", 0.0, 2_000);
    let back = poll_state(devctl_port).expect("state after leg 2");
    assert_eq!(
        back.location.as_deref(),
        Some("System 7"),
        "the occupant returned to the parent realm: {back:?}",
    );
    // The vacated Planet 7 is torn down + its real shard process reaped. The drain + quiesce windows make this
    // a polled assertion (not instant).
    let started = Instant::now();
    loop {
        let r = orch_rlm(a.admin);
        if r.teardowns_reaped >= 1 {
            break;
        }
        assert!(
            started.elapsed() < deadline,
            "the vacated Planet 7 was never torn down/reaped (teardowns_reaped stayed 0): {r:?}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}
