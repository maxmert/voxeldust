//! **THE WORLD IS SEEN FROM INSIDE ANY REALM** — owner ruling 2026-09-02 (`docs/design/
//! owner_decisions_2026-09-02_reach.md`, R9 step 1): the red test that must go green before the
//! reach design is done.
//!
//! MEASURED 2026-09-01: a player crossed into a player-built hull and saw a black sky. No error, no
//! counter, no warning. The stars were held in full and drawn not at all, and the picture composed
//! for the player held exactly one row — the hull. Every gate in the suite stayed green, because:
//!
//! * the one sky gate anchors on the observer's OWN STAR SYSTEM and cannot run any other origin;
//! * every convergence predicate accepts ONE realm box, and one box is exactly the failure;
//! * the client reported the stars it HELD, never the stars it DREW.
//!
//! So this file asserts the two things a player actually needs, and it asserts them from inside
//! TWO realms of different kinds on one cluster flight (HR4, G-IDENTICAL):
//!
//! 1. the **home star system**, a seeded realm, where the login lands: the stars are DRAWN
//!    (`stars_drawn > 0` — the new instrument, the drawn count and not the held count) and the
//!    system's own children are in the picture;
//! 2. a **player-built hull**, written by the shipyard's stand-in, berthed in the home system, and
//!    entered by an ordinary containment crossing forty metres from the spawn: the stars are still
//!    drawn, the picture names at least one realm the hull does NOT parent (the star system's own
//!    body, a world, the star), and the stars do not vanish across the crossing (after + 1 >= before).
//!
//! A THIRD subject — the home system's innermost planet, entered by the governed rendezvous — is
//! written and IGNORED, with its measurement: the movement ruling deleted the approach governor, a
//! walking occupant moves at foot speed, and MEASURED 2026-09-02 the chase ran at 1.0e3 m/s toward a
//! planet 6.1e10 m away (a 48 km/s orbit) — no budget reaches it. It runs again when a hull can fly
//! there (the temporary control seam, M-C) or the spawn moves (G10).
//!
//! Every wait is on a signal with a derived bound, never a sleep standing in for one. One DEMAND
//! cluster per subject (orchestrator + gateway, no shard pre-booked). GPU-required + LOCAL like the
//! other capture gates: the drawn-star count is written by the render thread.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::cross_leg;
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, realm_store_path, reap_forked, reserve_tcp_addr, reserve_udp_addr,
    world_roster,
};
use vd_core::NodeId;
use vd_core::entity_kind::EntityKind;
use vd_core::flight::{TRAVERSE_S, realm_speed_cap_mps};
use vd_core::glam::DVec3;
use vd_core::ids::EntityId;
use vd_core::pose::{RealmId, frame_for_realm};
use vd_devproto::{CLIENT_NODE_BASE, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::{AdminSnapshot, GatewayView};

// ---------------------------------------------------------------------------------------------
// Cluster scaffolding (the demand shape, as the RLM process gates boot it).
// ---------------------------------------------------------------------------------------------

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Fixture {
    /// The slot's own directory: the trust dir's PARENT, which is where every realm's file lives
    /// (`common_env` sets `VD_REALM_STORE_DIR` to exactly this).
    base: std::path::PathBuf,
    trust_dir: std::path::PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: std::path::PathBuf,
    cwd: std::path::PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // A RED RUN KEEPS ITS EVIDENCE. The forked shards' logs live under this directory, and a
        // failure that deletes them explains nothing — the first run of this gate did exactly that.
        if std::thread::panicking() {
            eprintln!(
                "world_from_inside: the fixture is KEPT for diagnosis at {}",
                self.base.display()
            );
            return;
        }
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn fixture(tag: &str) -> Fixture {
    let trust = ClusterTrust::generate("vd-world-from-inside").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-inside-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let cwd = base.join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("capture cwd");
    let store = base.join("orchestrator.redb");
    let launch_path = store.with_file_name(vd_bins::LAUNCH_STORE_NAME);
    let common = common_env(&trust_dir.display().to_string(), &DEV);
    Fixture {
        store_str: store.display().to_string(),
        base,
        trust_dir,
        common,
        launch_path,
        cwd,
    }
}

/// SIGKILL + reap every demand-spawned shard the orchestrator forked, on drop.
struct ForkedReaper(std::path::PathBuf);
impl Drop for ForkedReaper {
    fn drop(&mut self) {
        if self.0.exists() {
            reap_forked(&launch_rows(&self.0));
        }
    }
}

fn demand_addrs(gateway_admin: SocketAddr) -> ClusterAddrs {
    ClusterAddrs {
        gateway_admin: Some(gateway_admin),
        ..ClusterAddrs::reserve()
    }
}

fn spawn_capture_client(
    f: &Fixture,
    gateway: SocketAddr,
    name: &str,
    quic_port: u16,
    devctl_port: u16,
) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.current_dir(&f.cwd);
    cmd.args([
        "--name",
        name,
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
        "--capture",
        "--capture-pilot",
    ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().expect("spawn capture client")
}

fn boot_demand_cluster(
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    client_book: &[(NodeId, SocketAddr)],
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
    cluster.push(
        "vd-gateway",
        vd_bins::spawn_node(
            env!("CARGO_BIN_EXE_vd-gateway"),
            &f.common,
            &gateway_env(
                a,
                client_book,
                &dev_auth_pubkey_hex(),
                p,
                ClusterShape::Demand,
            ),
        )
        .expect("spawn gateway"),
    );
    cluster
}

fn admin(addr: SocketAddr) -> Option<AdminSnapshot> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str(&body).ok()
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    admin(addr)?.gateway
}

/// A red run says what the cluster thought: the orchestrator's lifecycle counters and the gateway's
/// view, printed at the failure point so the log can separate "never demanded" from "never launched"
/// from "launched and never seen".
fn dump_cluster_view(orch_admin: SocketAddr, gw_admin: SocketAddr) {
    eprintln!(
        "world_from_inside: orchestrator RLM view = {:?}",
        admin(orch_admin).map(|s| s.rlm)
    );
    eprintln!(
        "world_from_inside: gateway view = {:?}",
        gateway_view(gw_admin)
    );
}

/// The login must SPAWN a real shard process before it can converge.
const LOGIN_DEADLINE: Duration = Duration::from_secs(60);
/// A healthy client clears this in a second or two once its home is up — below it is a stall.
const SNAPSHOT_FLOOR: u64 = 5;

fn await_active(devctl_port: u16, gw_admin: SocketAddr, deadline: Duration) -> DevState {
    let started = Instant::now();
    loop {
        if let Ok(DevResponse::State { state }) = dev_roundtrip(devctl_port, &DevRequest::State)
            && state.phase == DevPhase::Active
            && state.snapshots_applied >= SNAPSHOT_FLOOR
            && !state.realm_boxes.is_empty()
        {
            return state;
        }
        assert!(
            started.elapsed() < deadline,
            "the demand login never converged: gateway view {:?}",
            gateway_view(gw_admin),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn await_listener(port: u16, child: &mut Child) {
    let started = Instant::now();
    loop {
        if std::net::TcpStream::connect(vd_bins::loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the capture client exited before serving dev-control ({status}) — GPU \
                 precondition: this gate needs a working adapter (WGPU_BACKENDS/WGPU_POWER_PREF)",
            );
        }
        assert!(
            started.elapsed() < Duration::from_secs(90),
            "the capture client never opened its dev-control listener on {port}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn label_of(realm: RealmId) -> String {
    frame_for_realm(realm, None)
        .expect("every realm of THE world has a frame")
        .label()
}

/// A realm's own governed speed ceiling, from its bound extent.
fn cap_of(bound_m: f64) -> f64 {
    realm_speed_cap_mps(bound_m, DEV.move_speed, TRAVERSE_S)
}

// ---------------------------------------------------------------------------------------------
// THE ONE BODY — what a player must see from inside a realm, whatever the realm is.
// ---------------------------------------------------------------------------------------------

/// How long the composed picture gets to settle after a crossing before the verdict is read: the
/// window keep-alive beat is the slowest thing on the path (a fresh `Child` window opens on it,
/// the parent answers on the next AoI beat, the composer folds on the tick after), so the bound
/// is a handful of beats — derived from the cluster's own tick rate, never a flat number.
fn settle_budget() -> Duration {
    let tick_hz = (1.0 / DEV.tick_dt).round().max(1.0);
    let beat_s = (tick_hz / 2.0).max(1.0) / tick_hz;
    // Eight beats: two to open the window, two for the parent's first fold, two for the relay,
    // two of slack for the reliable lane's own re-delivery.
    Duration::from_secs_f64(8.0 * beat_s).max(Duration::from_secs(10))
}

/// The verdict a player needs from inside `subject`, polled until it holds or the budget ends —
/// and then ASSERTED on the last state, so a red run names exactly what was missing.
fn assert_world_seen_from(devctl: u16, subject: RealmId, stars_before: u64) {
    let subject_label = label_of(subject);
    let budget = settle_budget();
    let started = Instant::now();
    let mut last;
    loop {
        last = vd_bins::pixel::poll(devctl);
        let inside = last.location.as_deref() == Some(subject_label.as_str());
        let stars = last.stars_drawn > 0;
        let outside_seen = last.realm_boxes.iter().any(|b| {
            b.realm != subject_label && b.parent.as_deref() != Some(subject_label.as_str())
        });
        if inside && stars && outside_seen {
            break;
        }
        if started.elapsed() > budget {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(
        last.location.as_deref(),
        Some(subject_label.as_str()),
        "the player stands inside the subject: {:?}",
        last.location
    );
    assert!(
        last.stars_drawn > 0,
        "THE STARS ARE DRAWN from inside {subject_label}: stars_drawn = {} while the client holds \
         sky {:?} (held is not drawn — that is the whole defect)",
        last.stars_drawn,
        last.sky,
    );
    assert!(
        last.stars_drawn + 1 >= stars_before,
        "the stars do not vanish across the crossing: {} before, {} inside {subject_label}",
        stars_before,
        last.stars_drawn,
    );
    let names: Vec<String> = last
        .realm_boxes
        .iter()
        .map(|b| format!("{} (parent {:?})", b.realm, b.parent))
        .collect();
    assert!(
        last.realm_boxes.iter().any(
            |b| b.realm != subject_label && b.parent.as_deref() != Some(subject_label.as_str())
        ),
        "THE WORLD OUTSIDE IS SEEN from inside {subject_label}: the picture must name a realm the \
         subject does not parent (its star system's body, a sibling world, the star). It names: \
         {names:?}",
    );
}

/// Log in, and read the stars the sky draws from the home system — the precondition every
/// subject is measured against.
fn login_under_the_stars(
    devctl: u16,
    gw_admin: SocketAddr,
    orch_admin: SocketAddr,
    home: RealmId,
) -> u64 {
    let landed = await_active(devctl, gw_admin, LOGIN_DEADLINE);
    assert_eq!(
        landed.location.as_deref(),
        Some(label_of(home).as_str()),
        "the login lands in the home star system: {landed:?}",
    );
    // SUBJECT 1, THE SEEDED REALM: from inside the home star system the stars are DRAWN and the
    // system's own children (its worlds, its star) are in the picture. The sky arrives paced, so
    // this waits for the cloud to be on screen before measuring against it.
    let started = Instant::now();
    loop {
        let st = vd_bins::pixel::poll(devctl);
        if st.stars_drawn > 0 {
            assert!(
                !st.realm_boxes.is_empty(),
                "from inside the home system its own children are drawn: {st:?}"
            );
            return st.stars_drawn;
        }
        if started.elapsed() >= LOGIN_DEADLINE {
            dump_cluster_view(orch_admin, gw_admin);
            panic!(
                "the sky never drew from the home system (held {:?}, watch {}, anchor {:?}, \
                 origin {:?}, boxes {}) — the precondition this gate measures against",
                st.sky,
                st.sky_watch,
                st.sky_anchor,
                st.origin,
                st.realm_boxes.len(),
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

// ---------------------------------------------------------------------------------------------
// SUBJECT 1 — a player-built hull, forty metres from the spawn.
// ---------------------------------------------------------------------------------------------

/// The shipyard stand-in's default mint (`--mint-shard 1 --seq 1`), restated so the test knows the
/// hull's name before the tool prints it — and checks the print against it.
fn minted_hull() -> RealmId {
    RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 1, 0))
}

/// How far from the spawn the hull is berthed, along +x: inside the hull's own wake radius by two
/// orders (a 21 m half-diagonal wakes from ~1.6 km), and a two-second walk.
const BERTH_STANDOFF_M: f64 = 40.0;

/// Write the hull into the home system's file and its own, where the shards will read them. The
/// spawn is the home clearing; a berth is measured from the star, so all three axes are stated.
fn plant_hull(f: &Fixture, home: RealmId, spawn_m: DVec3) -> RealmId {
    let parent_store = realm_store_path(&f.base, home);
    let ship_store = realm_store_path(&f.base, minted_hull());
    let out = Command::new(env!("CARGO_BIN_EXE_vd-build-ship"))
        .args([
            "--parent-store",
            &parent_store,
            "--ship-store",
            &ship_store,
            "--owner",
            "1000",
            "--berth-x-m",
            &(spawn_m.x + BERTH_STANDOFF_M).to_string(),
            "--berth-y-m",
            &spawn_m.y.to_string(),
            "--berth-z-m",
            &spawn_m.z.to_string(),
        ])
        .output()
        .expect("the shipyard's stand-in runs");
    assert!(
        out.status.success(),
        "vd-build-ship: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let printed = String::from_utf8_lossy(&out.stdout);
    let hull = minted_hull();
    assert!(
        printed.contains(&hull.to_string()),
        "the tool names the hull it built ({hull}); it printed {printed:?}",
    );
    hull
}

#[test]
fn a_player_inside_a_built_hull_sees_the_stars_and_the_world_outside() {
    let _tier = vd_bins::cluster_tier();
    let roster = world_roster(&DEV);
    // THE SPAWN, BY THE GATEWAY'S OWN EXPRESSION. `boot_world` is what the gateway derives its home
    // registry from, so its clearing is where the player lands. MEASURED on this gate's first run: the
    // system-LAYER view holds no children, so its clearing is ZERO, and a hull berthed forty metres
    // from that point sits inside the star — 1.08e10 m from the player, who then walked toward it for
    // a minute. A berth derived from any other view is a berth at the wrong place.
    let spawn_m =
        vd_bins::boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt).default_home_offset_m();

    let f = fixture("hull");
    let hull = plant_hull(&f, roster.home, spawn_m);

    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV, &client_book);
    let mut client = ChildGuard(spawn_capture_client(
        &f,
        a.gateway,
        "g-inside-hull",
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);
    let stars_before = login_under_the_stars(devctl, gw_admin, a.admin, roster.home);
    let landed = vd_bins::pixel::poll(devctl);
    eprintln!(
        "world_from_inside: landed at {:?}, the hull is berthed at {:?}",
        vd_bins::pixel::own_pose(&landed).map(|(p, _)| p),
        DVec3::new(spawn_m.x + BERTH_STANDOFF_M, spawn_m.y, spawn_m.z),
    );

    // Walk into the hull. The berth is stated in the home system's frame — the frame the player
    // stands in until the crossing commits — so the aim is fixed. The hull's shard spawns by the
    // ordinary demand (the player stands well inside its wake radius from the first tick); the
    // crossing fires once it runs. The budget is a login's worth: a shard must boot.
    let berth = DVec3::new(spawn_m.x + BERTH_STANDOFF_M, spawn_m.y, spawn_m.z);
    let entered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cross_leg(
            devctl,
            "into the hull",
            move |_tick| berth,
            &label_of(hull),
            LOGIN_DEADLINE,
        );
    }));
    if entered.is_err() {
        dump_cluster_view(a.admin, gw_admin);
        eprintln!(
            "world_from_inside: the client's last state = {:?}",
            vd_bins::pixel::poll(devctl)
        );
        panic!("the player never entered the hull");
    }

    assert_world_seen_from(devctl, hull, stars_before);
}

// ---------------------------------------------------------------------------------------------
// SUBJECT 2 — the home system's innermost planet, by the governed rendezvous.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "MEASURED 2026-09-02: a walking occupant cannot reach a planet (1.0e3 m/s toward 6.1e10 m; \
            the governor is deleted). Re-enable when a hull flies there (M-C) or the spawn moves (G10)."]
fn a_player_inside_the_home_planet_sees_the_stars_and_the_world_outside() {
    let _tier = vd_bins::cluster_tier();
    let roster = world_roster(&DEV);
    let system_bound_m = vd_physics::worldgen::realm_regions_for_config(
        DEV.universe_seed,
        &vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt),
    )
    .iter()
    .find(|r| r.realm == roster.home)
    .map(|r| r.shape.finite_extent())
    .expect("the home system is rostered on THE world");

    let f = fixture("planet");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let client_quic = reserve_udp_addr();
    let devctl = reserve_tcp_addr().port();
    let client_book = [(NodeId(CLIENT_NODE_BASE), client_quic)];
    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV, &client_book);
    let mut client = ChildGuard(spawn_capture_client(
        &f,
        a.gateway,
        "g-inside-planet",
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);
    let stars_before = login_under_the_stars(devctl, gw_admin, a.admin, roster.home);

    vd_bins::flight::rendezvous_into_planet(
        devctl,
        &DEV,
        roster.inner,
        &roster.inner_elements,
        vd_bins::flight::governed_leg_budget(&DEV, system_bound_m, cap_of(system_bound_m)),
    );

    assert_world_seen_from(devctl, roster.inner, stars_before);
}
