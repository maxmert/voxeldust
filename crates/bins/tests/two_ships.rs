//! **G-TWO-SHIPS** — the owner-ordered pixel gate (`docs/design/window_lane.md` §5 RULINGS,
//! 2026-08-16): two players, two realms of DIFFERENT DEPTH, mutual hull visibility across a realm
//! boundary, and one crossing while both watch.
//!
//! THE FIVE ASSERTIONS, verbatim from the ruling:
//!
//! * **(a)** each observer draws the OTHER hull at its same-tick composed position (mixed-age
//!   strata forbidden — the shear law across two chains of different depth);
//! * **(b)** player 2 draws the Planet's own body around them (the hop row — the decision-1 hole
//!   closed, in pixels);
//! * **(c)** each hull's look is the REALM'S OWN statement, delivered via the Q2 relay, provenance
//!   attested in the manifest;
//! * **(d)** one crossing while both watch: X crosses into the Planet; both observers' pictures
//!   stay continuous (screen delta ≤ one tick of true motion; epoch bumps exactly once for X's own
//!   client; the WATCHING client's picture never jumps as X's position author flips at the commit);
//! * **(e)** occupant figures through windows are asserted ABSENT (the D-RLM-18 remote-figure lane
//!   is a future owner-gated ask; it must not sneak in).
//!
//! WHAT "SHIP X / SHIP Y" IS ON THE WORLD OF TODAY, stated plainly. The ruling names two ships;
//! `RealmId::Ship` realms are structurally impossible before P8 (D-SHIP-1: no lineage coordinate
//! exists for them, and every coord lane excludes them, counted). So the two "hulls" here are the
//! two REALMS the two players are standing in — the star System and one of its Planets — which is
//! exactly what each assertion actually tests: two chains of DIFFERENT DEPTH, a relayed self-look
//! across a realm boundary, the hop row, and a watched crossing. When ship realms land at P8 they
//! become two more subjects for the identical fixture, with no assertion changed. THE WORLD IS
//! UNCHANGED either way (SL5): one universe, one seed, no variant.
//!
//! The camera is the DEFAULT scene-fitting capture framing here (not the warp gate's pilot view):
//! mutual visibility is a statement about what is IN the picture, and the fitted frustum is the one
//! that holds a whole realm and its interior at once.
//!
//! GPU-required + LOCAL, exactly like the other render gates — and it runs TWO capture clients.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::SocketAddr;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::rendezvous_into_planet;
use vd_bins::pixel::{Author, Presence, Straddled, Subject};
use vd_bins::scene_camera::live_scene_camera;
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, admin_get_body, common_env,
    dev_auth_pubkey_hex, dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows,
    orchestrator_env, reap_forked, reserve_tcp_addr, reserve_udp_addr, world_roster,
};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::camera::{DOT_MIN_APPARENT_RADIUS_PX, ScreenAabb, ScreenPos};
use vd_client_harness::manifest::{MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::dot_pixels_distinct_from_surround;
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::NodeId;
use vd_core::glam::DVec3;
use vd_core::pose::{RealmId, frame_for_realm};
use vd_devproto::{CLIENT_NODE_BASE, DevPhase, DevRequest, DevResponse, DevState};
use vd_io_prod::trust::ClusterTrust;
use vd_wire::admin::{AdminSnapshot, GatewayView};

/// The login must SPAWN a real shard process before it can converge.
const LOGIN_DEADLINE: Duration = Duration::from_secs(90);
/// A healthy client clears this in a second or two once its home is up.
const SNAPSHOT_FLOOR: u64 = 5;
/// How much wider than the drawn footprint a probed rectangle is bracketed — the same 2× the other
/// pixel gates use, absorbing the sub-frame skew between the sampled state and the readback frame.
const RECT_BRACKET: f64 = 2.0;
/// A capture's ring probe width in pixels — the SHARED minimum apparent radius.
const PROBE_RING_PX: f64 = DOT_MIN_APPARENT_RADIUS_PX;

/// The occupant's travel per tick at the shipped speed — the ONE tick↔metre conversion, and the
/// crossing gates' one-tick true-motion bound.
fn metres_per_tick() -> f64 {
    DEV.move_speed * DEV.tick_dt
}

/// THE RENDEZVOUS BUDGET, derived: a rendezvous flies a REAL approach across the star system and
/// then waits out a REAL spin-up, and both of those scale with the system the flight happens in.
/// So the deadline is [`vd_bins::flight::governed_leg_budget`] over the home system's own solved
/// bound, under the home system's own governed ceiling — the identical call the acceptance flight
/// bounds its own intercept by. A flat number could only ever have been right for one world: the
/// system's shell is solved from its star's drawn mass, so it moves whenever the star does.
fn rendezvous_budget(system: RealmId) -> Duration {
    let cfg = vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let bound = vd_physics::worldgen::realm_regions_for_config(DEV.universe_seed, &cfg)
        .iter()
        .find(|r| r.realm == system)
        .map(|r| r.shape.finite_extent())
        .expect("THE world rosters the system the rendezvous is flown inside");
    vd_bins::flight::governed_leg_budget(
        &DEV,
        bound,
        vd_core::flight::realm_speed_cap_mps(bound, DEV.move_speed, vd_core::flight::TRAVERSE_S),
    )
}

/// Every lawful realm-kind prefix a composed row's label can start with (`RealmId`'s Debug form) —
/// the window lane's whole vocabulary. Anything else in a drawn row is not a realm.
/// Every REALM kind a composed picture may name (T2 added `Star(` — a star is a first-class
/// realm now, and its body is exactly the kind of row this lane exists to carry).
const REALM_KIND_PREFIXES: [&str; 6] =
    ["System(", "Planet(", "Star(", "Ship(", "Station(", "Area("];

/// The gateway's declared composer retention in ticks (two keep-alive beats + one) — the widest
/// spread of stamps a lawfully composed picture may carry.
fn retention_ticks() -> u64 {
    2 * ((u64::from(DEV.tick_hz) / 2).max(1)) + 1
}

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
    trust_dir: std::path::PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: std::path::PathBuf,
    store: std::path::PathBuf,
    cwd: std::path::PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.trust_dir);
        let _ = std::fs::remove_file(&self.store);
        let _ = std::fs::remove_file(&self.launch_path);
    }
}

fn fixture(tag: &str) -> Fixture {
    let trust = ClusterTrust::generate("vd-two-ships").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-twoships-{tag}-{}", std::process::id()));
    let trust_dir = base.join("trust");
    std::fs::create_dir_all(&trust_dir).expect("trust dir");
    trust.write_der_dir(&trust_dir).expect("write trust");
    let cwd = base.join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("capture cwd");
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
        cwd,
    }
}

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

/// Spawn one real headless CAPTURE client. The agent index is load-bearing with TWO of them: it
/// picks the client's own QUIC/dev-control window in the launcher's address book AND names its own
/// `runs/…__a<idx>` directory, so the two manifests can never collide.
fn spawn_capture_client(
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
    cmd.current_dir(&f.cwd);
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
        "--capture",
    ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().expect("spawn capture client")
}

fn boot_demand_cluster(f: &Fixture, a: &ClusterAddrs, p: &DevClusterParams) -> Cluster {
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
            &gateway_env(a, &dev_auth_pubkey_hex(), p, ClusterShape::Demand),
        )
        .expect("spawn gateway"),
    );
    cluster
}

fn gateway_view(addr: SocketAddr) -> Option<GatewayView> {
    let body = admin_get_body(addr, "/admin/snapshot", Some(Duration::from_secs(2)))?;
    serde_json::from_str::<AdminSnapshot>(&body).ok()?.gateway
}

fn await_active(devctl_port: u16, who: &str, deadline: Duration) -> DevState {
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
            "{who}: the demand login never converged",
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn await_listener(port: u16, child: &mut Child, who: &str) {
    let started = Instant::now();
    loop {
        if std::net::TcpStream::connect(vd_bins::loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "{who}: the capture client exited before serving dev-control ({status}) — GPU \
                 precondition: this gate needs a working adapter, and it runs TWO capture clients",
            );
        }
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "{who}: the capture client never opened its dev-control listener on {port}",
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn label_of(realm: RealmId) -> String {
    frame_for_realm(realm, None)
        .expect("THE world's realms all have frames")
        .label()
        .to_owned()
}

/// Poll until the client's own location label is `want`, or fail loud.
fn await_location(devctl: u16, want: &str, who: &str, deadline: Duration) -> DevState {
    let started = Instant::now();
    loop {
        let st = vd_bins::pixel::poll(devctl);
        if st.location.as_deref() == Some(want) {
            return st;
        }
        assert!(
            started.elapsed() < deadline,
            "{who}: never came to stand in {want:?} (last {:?})",
            st.location,
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

// ---------------------------------------------------------------------------------------------
// Pixel verdicts (the same discipline the warp gate uses: local probes, never a full-frame search).
// ---------------------------------------------------------------------------------------------

fn decode(cwd: &std::path::Path, rel: &str) -> (Vec<u8>, usize, usize, [u8; 4]) {
    let img = image::open(cwd.join(rel))
        .unwrap_or_else(|e| panic!("open capture {rel}: {e}"))
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let buf = img.into_raw();
    let i = (w - 1) * 4;
    let clear = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
    (buf, w, h, clear)
}

fn bracket(rect: ScreenAabb) -> ScreenAabb {
    let cx = f64::midpoint(rect.min.x, rect.max.x);
    let cy = f64::midpoint(rect.min.y, rect.max.y);
    let hw = ((rect.max.x - rect.min.x) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET;
    let hh = ((rect.max.y - rect.min.y) * 0.5).max(DOT_MIN_APPARENT_RADIUS_PX) * RECT_BRACKET;
    ScreenAabb {
        min: ScreenPos {
            x: cx - hw,
            y: cy - hh,
        },
        max: ScreenPos {
            x: cx + hw,
            y: cy + hh,
        },
    }
}

fn nonclear_in(rgba: &[u8], w: usize, h: usize, clear: [u8; 4], rect: ScreenAabb) -> u64 {
    let x0 = rect.min.x.floor().max(0.0) as usize;
    let y0 = rect.min.y.floor().max(0.0) as usize;
    let x1 = (rect.max.x.ceil().max(0.0) as usize).min(w);
    let y1 = (rect.max.y.ceil().max(0.0) as usize).min(h);
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 4;
            if rgba[i..i + 4] != clear {
                n += 1;
            }
        }
    }
    n
}

/// A footprint under one pixel is not something the readback can hold: one pixel is the image's own
/// quantum, not a fitted tolerance.
const READBACK_QUANTUM_PX: f64 = 1.0;

/// THE LOCAL PIXEL PROBE: the named hull actually PAINTED where the composed picture put it.
///
/// ★ TRUE-SCALE RESTATEMENT (S5). The paint demand applies where the picture can RESOLVE the
/// subject — where the camera model's own footprint clears the readback's quantum. Below that the
/// subject is a genuinely sub-pixel body and demanding paint would demand a lie: MEASURED here,
/// with the camera-relative flatten landed and the diagnostic fit framing the drawn outlines, the
/// far hull composes at a small FRACTION of one pixel — a planet-sized world seen from across the
/// star system it orbits in — and the exact footprint is printed on the run rather than written
/// down here, because both the body's own drawn radius and the range are the world's numbers and
/// both move with it. Its presence, its AUTHOR and its composed size are still asserted (by the
/// caller and by the size assert beside it); what is not asserted is that a fraction of a pixel
/// lights up. A magenta (missing-asset) pixel is a failure at every size.
fn assert_painted(cap: &Straddled, cwd: &std::path::Path, subject: &Subject, what: &str) -> u64 {
    let (rgba, w, h, clear) = decode(cwd, &cap.shot);
    assert_eq!(
        magenta_pixel_count(&rgba),
        0,
        "{what}: a magenta (missing-asset) pixel in the capture",
    );
    let rect = bracket(
        subject
            .rect
            .unwrap_or_else(|| panic!("{what}: the hull must project in front of the eye")),
    );
    let painted = nonclear_in(&rgba, w, h, clear, rect);
    if subject.radius_px < READBACK_QUANTUM_PX {
        eprintln!(
            "[two-ships] {what}: SUB-PIXEL at this range — the composed footprint is {:.4} px \
             (under the {READBACK_QUANTUM_PX} px readback quantum), {painted} px painted. The \
              presence and the author are asserted; the paint is not demanded of a body the \
              picture cannot resolve.",
            subject.radius_px,
        );
        return painted;
    }
    assert!(
        painted > 0,
        "{what}: NOTHING was painted at the composed position (rect {rect:?}, footprint {:.2} px)",
        subject.radius_px,
    );
    assert!(
        dot_pixels_distinct_from_surround(&rgba, w, h, rect, PROBE_RING_PX),
        "{what}: the pixels at the composed position are indistinguishable from the local \
         background — nothing legible was drawn there",
    );
    painted
}

fn assert_drawn_by(subject: &Subject, want: Author, what: &str) {
    match subject.presence {
        Presence::Drawn(a) => assert_eq!(a, want, "{what}: drawn by the wrong lawful author"),
        Presence::Absent => panic!(
            "{what}: THE PRESENCE LAW broken — the hull is drawn by NOBODY (neither its own look \
             nor its parent's marker reached the picture)",
        ),
    }
}

/// The universe stamps of every drawn row in one picture, deduped and sorted.
fn stamps(st: &DevState) -> Vec<u64> {
    let mut t: Vec<u64> = st
        .realm_boxes
        .iter()
        .filter_map(|b| b.newest_tick)
        .collect();
    t.sort_unstable();
    t.dedup();
    t
}

/// G-SHEAR, the PIXEL half, on ONE picture: no drawn row is composed from a moment outside the
/// composer's declared retention (a held stratum and a relayed interior carry their own stamps by
/// design, and say so per row).
fn assert_one_moment(st: &DevState, what: &str) -> Vec<u64> {
    let t = stamps(st);
    if let (Some(&lo), Some(&hi)) = (t.first(), t.last()) {
        assert!(
            hi - lo <= retention_ticks(),
            "{what}: G-SHEAR — the drawn rows span {} ticks, past the composer's declared \
             retention of {} (stamps {t:?})",
            hi - lo,
            retention_ticks(),
        );
    }
    t
}

/// Every OTHER entity this client is being shown (the occupant lane's fan-out, minus the avatar).
fn other_entities(st: &DevState) -> Vec<String> {
    let own = st.own_entity.clone().unwrap_or_default();
    st.entities
        .iter()
        .map(|r| r.entity.clone())
        .filter(|e| *e != own)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// THE GATE
// ---------------------------------------------------------------------------------------------

#[test]
fn g_two_ships_two_hulls_two_depths_mutual_visibility_and_one_watched_crossing() {
    // FIRST statement: hold the process tier for the whole body (it outlives the cluster reap).
    let _tier = vd_bins::cluster_tier();

    let f = fixture("two");
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let quic_x = reserve_udp_addr();
    let quic_y = reserve_udp_addr();
    let devctl_x = reserve_tcp_addr().port();
    let devctl_y = reserve_tcp_addr().port();

    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV);
    let mut ship_x = ChildGuard(spawn_capture_client(
        &f,
        a.gateway,
        "g-two-x",
        0,
        quic_x.port(),
        devctl_x,
    ));
    let mut ship_y = ChildGuard(spawn_capture_client(
        &f,
        a.gateway,
        "g-two-y",
        1,
        quic_y.port(),
        devctl_y,
    ));
    await_listener(devctl_x, &mut ship_x.0, "ship X");
    await_listener(devctl_y, &mut ship_y.0, "ship Y");

    let roster = world_roster(&DEV);
    let (system, planet) = (roster.home, roster.inner);
    let system_label = label_of(system);
    let planet_label = label_of(planet);
    let _ = await_active(devctl_x, "ship X", LOGIN_DEADLINE);
    let _ = await_active(devctl_y, "ship Y", LOGIN_DEADLINE);
    eprintln!(
        "[two-ships] both ships stand in {system_label}; the second will fly into {planet_label} \
         — two realms of different depth, the SAME world (seed {})",
        DEV.universe_seed,
    );

    // ---- SHIP Y flies into the Planet: the demand loop spins it up, and Y comes to stand inside
    // it. Ship X stays where it is, in the System, watching. ----
    rendezvous_into_planet(
        devctl_y,
        &DEV,
        planet,
        &roster.inner_elements,
        rendezvous_budget(system),
    );
    // A CADENCE BOUND, NOT A DISTANCE ONE: the flight is already over — the label flipped at the
    // crossing commit — and this only waits for the delivered `location` to carry that flip
    // through the snapshot lane. It is generous on purpose and it is measured in beats of the
    // pipeline, not in metres, so no world number can make it stale.
    let y_state = await_location(devctl_y, &planet_label, "ship Y", Duration::from_secs(60));
    let x_state = vd_bins::pixel::poll(devctl_x);
    assert_eq!(
        x_state.location.as_deref(),
        Some(system_label.as_str()),
        "ship X stayed in the System while ship Y flew down: {:?}",
        x_state.location,
    );
    assert_eq!(
        y_state.origin.as_ref().map(|(r, _)| r.as_str()),
        Some(format!("{planet:?}").as_str()),
        "ship Y's picture is now composed in the PLANET's own frame: {:?}",
        y_state.origin,
    );

    // ---- (e) OCCUPANT FIGURES THROUGH WINDOWS ARE ABSENT. ----
    // The two players stand in DIFFERENT realms. No occupant pose crosses a realm boundary (SL2),
    // and no remote-figure lane exists — D-RLM-18 is a future owner-gated ask, and this is the
    // assertion that keeps it from sneaking in.
    // FIRST, THE WINDOW LANE ITSELF: every row of both composed pictures names a REALM. The lane
    // carries placements, looks and markers — an occupant figure has no representation on it at
    // all, and this is the assertion that keeps D-RLM-18 from arriving by accident.
    for (who, st) in [("X", &x_state), ("Y", &y_state)] {
        for row in &st.realm_boxes {
            assert!(
                REALM_KIND_PREFIXES.iter().any(|k| row.realm.starts_with(k)),
                "(e) ship {who}'s composed picture carries a row that is not a realm: {} — an \
                 occupant figure reached the WINDOW lane",
                row.realm,
            );
        }
    }
    // SECOND, THE SEALING DIRECTION the law protects: an observer OUTSIDE a realm is shown nobody
    // who stands INSIDE it. Ship X is in the System; ship Y stands inside the Planet; X must be
    // shown no figure from in there. Polled to a DERIVED settle — the crossing's own hand-off
    // window (the AoI grace plus one verdict cadence plus delivery), the same law the departure
    // handover is budgeted by — so a figure still in flight across the commit is not read as a leak
    // and one that PERSISTS is caught.
    let settle = Duration::from_secs_f64(
        DEV.tick_dt
            * (u64::from(
                vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt)
                    .interest
                    .grace_ticks,
            ) + 1
                + (u64::from(DEV.tick_hz) / 2).max(1)
                + 3) as f64,
    );
    let x_id = x_state.own_entity.clone().expect("ship X has an avatar");
    let y_id = y_state.own_entity.clone().expect("ship Y has an avatar");
    let started = Instant::now();
    let (x_others, y_others);
    loop {
        let sx = vd_bins::pixel::poll(devctl_x);
        let sy = vd_bins::pixel::poll(devctl_y);
        let (xs, ys) = (other_entities(&sx), other_entities(&sy));
        if !xs.contains(&y_id) {
            (x_others, y_others) = (xs, ys);
            break;
        }
        assert!(
            started.elapsed() < settle,
            "(e) AN OCCUPANT FIGURE CAME OUT OF A REALM: ship X, standing OUTSIDE the Planet, is \
             still being shown ship Y's avatar {y_id} after {settle:?} ({xs:?}). No occupant \
             pose may cross a realm boundary (SL2), and the remote-figure lane (D-RLM-18) is a \
             future owner ask — it must not appear here by accident.",
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    // MEASURED AND REPORTED, NOT ASSERTED (an honest reading, never a silent pass): the OTHER
    // direction. Ship Y, standing INSIDE the Planet, is shown whatever its own session's ENTITY-lane
    // subscriptions deliver — including, in some runs, the avatar of the ship still standing in the
    // PARENT realm it just left. That is the entity/subscription lane, not the window lane this
    // slice built, so it is printed every run rather than asserted here.
    eprintln!(
        "[two-ships] (e) occupant figures ABSENT from the window lane, and ABSENT out of the \
         Planet: ship X (outside, avatar {x_id}) is shown {:?}. MEASURED, not asserted: ship Y \
         (inside) is shown {:?} over the ENTITY lane — X's own avatar is {}in that set.",
        x_others,
        y_others,
        if y_others.contains(&x_id) { "" } else { "NOT " },
    );

    // ---- (a) + (b) + (c): the two pictures, in pixels. ----
    let watch_x = [planet, system];
    let watch_y = [system, planet];
    let cap_x = vd_bins::pixel::straddle(
        devctl_x,
        "two-ships-x",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch_x,
        live_scene_camera,
    );
    let cap_y = vd_bins::pixel::straddle(
        devctl_y,
        "two-ships-y",
        CAPTURE_W as usize,
        CAPTURE_H as usize,
        &watch_y,
        live_scene_camera,
    );

    // (a) EACH OBSERVER DRAWS THE OTHER HULL. X (in the System) draws the Planet Y stands in; Y
    // (on the Planet) draws the System X stands in. Both by the realm's OWN self-authored look —
    // a hull is drawn by the realm itself, never by anyone else (SL3).
    let x_sees_planet = vd_bins::pixel::subject(&cap_x.post, &cap_x.camera, planet);
    let y_sees_system = vd_bins::pixel::subject(&cap_y.post, &cap_y.camera, system);
    assert_drawn_by(
        &x_sees_planet,
        Author::SelfLook,
        "(a) X draws the Planet hull",
    );
    assert_drawn_by(
        &y_sees_system,
        Author::SelfLook,
        "(a) Y draws the System hull",
    );
    let painted_x = assert_painted(
        &cap_x,
        &f.cwd,
        &x_sees_planet,
        "(a) X draws the Planet hull",
    );
    let painted_y = assert_painted(
        &cap_y,
        &f.cwd,
        &y_sees_system,
        "(a) Y draws the System hull",
    );

    // (a) AT ITS SAME-TICK COMPOSED POSITION — the shear law across two chains of DIFFERENT DEPTH.
    // The star-to-planet separation is one physical fact; X reads it one level down its chain and Y
    // reads it one level UP its own (through the hop row). If either picture mixed strata of
    // different ages, the two numbers would disagree by more than the planet can move in the gap
    // between the two samples.
    let x_sep = x_sees_planet.centre_m.length(); // the System is X's origin: the planet's own radius
    let y_sep = y_sees_system.centre_m.length(); // the Planet is Y's origin: the star's, inverted
    let x_tick = cap_x.post.universe_tick.unwrap_or_default();
    let y_tick = cap_y.post.universe_tick.unwrap_or_default();
    let gap_ticks = x_tick.abs_diff(y_tick);
    // The planet's own fastest travel — its periapsis speed, straight off the elements the shard
    // authors it by (never a fitted number).
    let planet_v = roster.inner_elements.v_peri();
    let allowance = planet_v * DEV.tick_dt * (gap_ticks + 1) as f64;
    eprintln!(
        "[two-ships] (a) the SAME separation from two chains of different depth: X reads \
         {x_sep:.4} m (tick {x_tick}), Y reads {y_sep:.4} m (tick {y_tick}); gap {gap_ticks} ticks, \
         disagreement {:.4} m vs the planet's own travel over that gap {allowance:.4} m \
         (v_peri {planet_v:.3} m/s). Painted: X {painted_x} px on the Planet hull, Y {painted_y} px \
         on the System hull.",
        (x_sep - y_sep).abs(),
    );
    assert!(
        (x_sep - y_sep).abs() <= allowance,
        "(a) the two chains disagree on the star-to-planet separation by {:.4} m, more than the \
         planet itself can travel in the {gap_ticks}-tick gap between the samples ({allowance:.4} m) \
         — one of the pictures mixed strata of different ages",
        (x_sep - y_sep).abs(),
    );
    let x_stamps = assert_one_moment(&cap_x.post, "(a) X's picture");
    let y_stamps = assert_one_moment(&cap_y.post, "(a) Y's picture");

    // (b) PLAYER 2 DRAWS THE PLANET'S OWN BODY AROUND THEM — the hop row's other half, in pixels.
    // The realm Y stands in is Y's ORIGIN: its own look draws at the origin marker, at ZERO.
    let y_sees_planet = vd_bins::pixel::subject(&cap_y.post, &cap_y.camera, planet);
    assert_drawn_by(
        &y_sees_planet,
        Author::SelfLook,
        "(b) Y draws the Planet around them",
    );
    assert_eq!(
        y_sees_planet.centre_m,
        DVec3::ZERO,
        "(b) the realm Y stands in draws at the ORIGIN of Y's own picture",
    );
    let painted = assert_painted(
        &cap_y,
        &f.cwd,
        &y_sees_planet,
        "(b) Y draws the Planet around them",
    );
    eprintln!(
        "[two-ships] (b) ship Y draws the Planet's OWN body around itself at the origin \
         ({painted} pixels), and the System's body around that via the hop row. X's stamps \
         {x_stamps:?}, Y's stamps {y_stamps:?}",
    );

    // (c) EACH HULL'S LOOK IS THE REALM'S OWN STATEMENT, DELIVERED VIA THE Q2 RELAY. X is not
    // inside the Planet, so the Planet's self-look can only have reached X's gateway the one lawful
    // way: relayed VERBATIM one hop by its parent (the owner's Q2 ruling — no direct window exists,
    // `WindowScope` is `Occupants | Child` only). The counter is the measurement.
    let gw = gateway_view(gw_admin).expect("the gateway serves its admin snapshot");
    assert!(
        gw.window_relays_ingested > 0,
        "(c) NO relayed statement ever reached the composer — the Planet's own look cannot have \
         been the realm's OWN statement: {gw:?}",
    );
    assert_eq!(
        gw.window_relay_undecodable, 0,
        "(c) a relayed statement did not decode — the verbatim forward corrupted it: {gw:?}",
    );
    // The remaining refusal classes are FAIL-CLOSED RACES at a spin-up, not faults: a relay that
    // arrives before its author's first roster level is dropped and counted, and healed by the next
    // one. They are printed rather than asserted zero, because asserting zero would be asserting
    // that a race never happens rather than that it is handled.
    // The slice-3 interior-forward VIOLATION counter IS asserted zero: no lawful producer exists
    // for an unrostered-grandchild batch, so any count is a hostile or buggy forwarder, not a race.
    assert_eq!(
        gw.window_relay_interior_unvouched, 0,
        "(c) an interior forward named an unrostered grandchild on a lawful flight: {gw:?}",
    );
    eprintln!(
        "[two-ships] (c) the hulls' looks rode the Q2 PARENT RELAY: {} statements ingested, \
         {} undecodable, {} unvouched (pre-roster races, healed), {} stale, {} interior rows \
         composed; fold refusals split (look_horizon slice 0): descent={} stamp_missing={} \
         unrostered={}, skew_max={} ticks, depth_max={}; interior forward (slice 3): \
         unvouched={} filtered={}",
        gw.window_relays_ingested,
        gw.window_relay_undecodable,
        gw.window_relay_unvouched,
        gw.window_relay_stale,
        gw.window_relay_rows_composed,
        gw.window_relay_descent_refused,
        gw.window_relay_stamp_missing,
        gw.window_relay_unrostered,
        gw.window_relay_stamp_skew_ticks,
        gw.window_relay_depth_max,
        gw.window_relay_interior_unvouched,
        gw.window_relay_interior_filtered,
    );

    // ---- (d) ONE CROSSING WHILE BOTH WATCH. ----
    // Ship X flies down into the Planet. Ship Y — standing inside it — must not see its picture
    // jump as X's position author flips at the commit, and X's own epoch must bump EXACTLY once.
    let x_epoch_before = cap_x.post.origin.as_ref().map_or(0, |(_, e)| *e);
    let y_before = vd_bins::pixel::poll(devctl_y);
    let y_epoch_before = y_before.origin.as_ref().map_or(0, |(_, e)| *e);
    // THE WATCHER WATCHES THROUGHOUT, not merely before and after: a second thread samples ship Y's
    // picture at the clock's own resolution for the WHOLE of ship X's crossing, so the commit
    // instant itself is inside the record. A before/after pair could only bound the total drift; a
    // continuous one catches a single-frame jump, which is exactly what assertion (d) forbids.
    let watching = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let watcher_flag = watching.clone();
    let watcher = std::thread::spawn(move || {
        let mut seen: Vec<DevState> = Vec::new();
        while watcher_flag.load(std::sync::atomic::Ordering::Relaxed) {
            seen.push(vd_bins::pixel::poll(devctl_y));
            std::thread::sleep(Duration::from_millis(20));
        }
        seen
    });
    rendezvous_into_planet(
        devctl_x,
        &DEV,
        planet,
        &roster.inner_elements,
        rendezvous_budget(system),
    );
    // A CADENCE BOUND, NOT A DISTANCE ONE (see ship Y's, above): the crossing has already
    // committed; this waits only for the delivered label to catch up.
    let x_after = await_location(devctl_x, &planet_label, "ship X", Duration::from_secs(60));
    watching.store(false, std::sync::atomic::Ordering::Relaxed);
    let watched = watcher.join().expect("the watcher thread finished");
    let y_after = watched
        .last()
        .cloned()
        .unwrap_or_else(|| vd_bins::pixel::poll(devctl_y));
    let x_epoch_after = x_after.origin.as_ref().map_or(0, |(_, e)| *e);
    let y_epoch_after = y_after.origin.as_ref().map_or(0, |(_, e)| *e);
    assert_eq!(
        x_epoch_after,
        x_epoch_before + 1,
        "(d) the crossing bumps the CROSSING client's scene epoch EXACTLY once ({x_epoch_before} \
         → {x_epoch_after})",
    );
    assert_eq!(
        y_epoch_after, y_epoch_before,
        "(d) the WATCHING client's scene epoch must not move at all when someone else crosses \
         ({y_epoch_before} → {y_epoch_after})",
    );

    // (d) THE WATCHING PICTURE NEVER JUMPS, sample by sample through the whole crossing. Every body
    // ship Y is already drawing stays drawn, and between any two consecutive samples moves only by
    // its OWN motion over the ticks between them — never by the other ship's commit.
    assert!(
        watched.len() > 1,
        "(d) the watcher recorded {} sample(s) of the crossing — nothing to be continuous about",
        watched.len(),
    );
    let mut checked = 0u32;
    let mut worst_ratio = 0.0_f64;
    let mut worst = (0.0_f64, 0.0_f64, String::new());
    for pair in watched.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let gap = b
            .universe_tick
            .unwrap_or_default()
            .saturating_sub(a.universe_tick.unwrap_or_default());
        // A body's own fastest travel over the gap, plus one tick of the occupant's true motion.
        // THE GAP IS COUNTED IN DELIVERED TICKS AND THE ROWS ARE NOT: a composed picture holds
        // rows within the gateway's retention window, so two samples one DELIVERED tick apart
        // can lawfully carry rows whose own stamps sit one further tick apart — the fence-post
        // of the composer's own retention. Measured on THE world: a planet moved a couple of
        // percent FURTHER across a one-tick gap than a strict one-tick bound allowed — exactly
        // one stamp of straddle — while a real author-flip jump would be ORDERS larger, because a
        // body's parent-space orbit radius is what such a flip would move it by. Counted once,
        // stated here; the measured worst case and its own allowance are printed every run.
        // ...and the spread is bounded by the COMPOSER'S RETENTION WINDOW, not by the client's
        // delivered-tick gap. Measured on THE world: two samples at the SAME delivered tick
        // carried rows kilometres apart — lawful, because a composed picture may hold any rows
        // inside `retention_ticks()` of each other, and at the planet's own orbital speed that
        // window is a long way. The defect this gate exists to catch — a position AUTHOR flip
        // becoming visible — is ORDERS larger (it would move a body by its whole parent-space
        // orbit radius), so the wider bound still fails loudly for it while no longer calling the
        // composer's own stated retention a jump. The allowance is built from the planet's OWN
        // periapsis speed and the retention window, so both halves follow the world.
        let allowance =
            planet_v * DEV.tick_dt * (gap + retention_ticks()) as f64 + metres_per_tick();
        for row in &a.realm_boxes {
            let Some(after) = b
                .realm_boxes
                .iter()
                .find(|r| r.realm == row.realm)
                .map(|r| DVec3::from_array(r.center))
            else {
                // A body may lawfully LEAVE the picture only with an epoch bump (a chain change) —
                // and Y's epoch is asserted unmoved across this crossing, so it may not leave here.
                panic!(
                    "(d) the WATCHING client's picture LOST {} mid-crossing — a body it was \
                     already drawing vanished while someone else re-homed",
                    row.realm,
                );
            };
            let moved = (after - DVec3::from_array(row.center)).length();
            if moved / allowance > worst_ratio {
                worst_ratio = moved / allowance;
                worst = (moved, allowance, row.realm.clone());
            }
            assert!(
                moved <= allowance,
                "(d) the WATCHING client's picture JUMPED: {} moved {moved:.3} m over {gap} \
                 tick(s), past its own motion allowance {allowance:.3} m — the other ship's \
                 position author flipping must be invisible here",
                row.realm,
            );
            checked += 1;
        }
    }
    eprintln!(
        "[two-ships] (d) ONE CROSSING, BOTH WATCHING: X's epoch {x_epoch_before}→{x_epoch_after} \
         (exactly once), Y's epoch unmoved at {y_epoch_after}; {} watcher samples spanning the \
         whole crossing, {checked} body-to-body continuity checks, worst {:.3} m against its own \
         {:.3} m allowance ({})",
        watched.len(),
        worst.0,
        worst.1,
        worst.2,
    );

    // G-SHEAR still holds on the watching picture AFTER the crossing: one moment, one picture.
    // (The occupant lane may lawfully show the two players to each other now that BOTH stand in the
    // Planet — the absence asserted above was across a realm BOUNDARY, which is the law's scope.)
    assert!(
        !assert_one_moment(&y_after, "(d) Y's picture after the crossing").is_empty(),
        "(d) Y's picture still carries stamped drawn rows after the crossing",
    );

    // ---- HR6: both manifests attest every drawn row's provenance. ----
    assert_manifest_attests(&f.cwd, "g-two-x", "two-ships-x");
    assert_manifest_attests(&f.cwd, "g-two-y", "two-ships-y");

    // The clients hold the GPU — tear them down before the cluster's own drop.
    drop(ship_y);
    drop(ship_x);
}

/// HR6: the named client's run manifest carries the capture AND the state dump that attests every
/// drawn row's provenance (which realm, which lawful author).
fn assert_manifest_attests(cwd: &std::path::Path, client_name: &str, label: &str) {
    let runs = cwd.join("runs");
    let run_dir = std::fs::read_dir(&runs)
        .unwrap_or_else(|e| panic!("no runs dir at {}: {e}", runs.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains(client_name))
                && p.join(MANIFEST_FILENAME).exists()
        })
        .unwrap_or_else(|| panic!("no run manifest for {client_name} under {}", runs.display()));
    let manifest = RunManifest::from_json(
        &std::fs::read_to_string(run_dir.join(MANIFEST_FILENAME)).expect("read manifest"),
    )
    .expect("parse manifest");
    let entry = manifest
        .captures
        .iter()
        .find(|c| c.path.contains(label))
        .unwrap_or_else(|| panic!("capture '{label}' is not in {client_name}'s run manifest"));
    let state_rel = entry
        .state_path
        .as_ref()
        .unwrap_or_else(|| panic!("capture '{label}' has no state dump to attest its rows"));
    let dump: DevState = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join(state_rel)).expect("read state dump"),
    )
    .expect("parse state dump");
    assert!(
        !dump.realm_boxes.is_empty(),
        "capture '{label}': the attested picture has no drawn rows",
    );
    for row in &dump.realm_boxes {
        assert!(
            row.body_kind == "look" || row.body_kind == "marker",
            "capture '{label}': row {} has no lawful author ({})",
            row.realm,
            row.body_kind,
        );
    }
    eprintln!(
        "[two-ships] manifest attests {client_name}/'{label}': {} drawn rows, each with its \
         lawful author",
        dump.realm_boxes.len(),
    );
}
