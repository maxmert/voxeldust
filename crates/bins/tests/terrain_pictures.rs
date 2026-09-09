//! ★ THE FIRST PICTURES OF THE HOME PLANET (the voxel foundation, slice 7; ruling V12 S7-11, M7-4):
//! the client links the one generator, a login stands on the planet's surface, and the harness takes
//! the pictures the owner judges — from the ground at rung 0, from a hill at a middle rung, and from
//! aloft at a coarse rung.
//!
//! **The path is the shipped one.** A DEMAND cluster (orchestrator + gateway, no shard pre-booked);
//! the stand-in spawn poses put three accounts INSIDE the home planet's realm (slice 7 gave the
//! stand-in a realm name and a facing): on the planet's DAY SIDE — the star's direction comes from the
//! planet's own orbit, the same elements its shard authors — each born with the radial as its up and
//! its nose toward the star, a little below level. One stands 1.8 m over the surface the recipe
//! states, one 300 m over it, one 60 km over it. The login demands the planet's chain; the planet's
//! shard boots and states its surface; the capture client draws the chunks around the eye at the rung
//! the flag names (`D-TERRAIN-3`: one rung, for one slice) and answers the harness with
//! `terrain_chunks_drawn`, the instrument this gate waits on — never a sleep standing in for a signal.
//!
//! **What is asserted.** Chunks are on screen; the screenshot holds the terrain's own warm paint
//! across the lower half of the frame; zero magenta (the missing-shader colour). The pictures are
//! copied beside the slice document for the owner. GPU-required + LOCAL like every capture gate.
#![cfg(all(feature = "dev-control", feature = "render"))]
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, common_env, dev_auth_pubkey_hex,
    dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows, orchestrator_env,
    reap_forked, reserve_tcp_addr, reserve_udp_addr,
};
use vd_client_harness::assert::magenta_pixel_count;
use vd_core::glam::DVec3;
use vd_devproto::{DevPhase, DevRequest, DevResponse, DevState, WaitField, WaitOp, WaitPredicate};
use vd_io_prod::trust::ClusterTrust;
use vd_terrain::Gf;

/// The client's account for `--agent-index N` (the client binary's own rule).
const CLIENT_ACCOUNT_BASE: u64 = 1000;
/// Eye height over the surface for the ground picture, in metres.
const EYE_HEIGHT_M: f64 = 1.8;
/// Height over the surface for the hill picture, in metres.
const HILL_M: f64 = 300.0;
/// Altitude for the aloft picture, in metres.
const ALOFT_M: f64 = 60_000.0;
/// The star's height over the standing point's horizon, in degrees (M8-L, ruling V13 L23: a raking
/// light, 12°–18°): low enough that every spur throws a shadow, high enough that the ground is lit.
const SUN_ELEVATION_DEG: f64 = 15.0;
/// The star's azimuth from the camera's nose, in degrees (M8-L: 100°–140°, behind the shoulder — a
/// picture shot INTO the light shows no relief, MEASURED on the slice 7 pictures).
const SUN_OFF_NOSE_DEG: f64 = 120.0;
/// How far below level each picture looks, in degrees.
const GROUND_TILT_DEG: f64 = 8.0;
const HILL_TILT_DEG: f64 = 15.0;
const ALOFT_TILT_DEG: f64 = 15.0;
/// A day of universe ticks at the dev cluster's rate: the bound on how far from the clock's genesis
/// a picture may be taken while its day side is computed at genesis.
const TICKS_PER_DAY: u64 = 86_400 * 20;
/// The rung and radius per picture (the one-rung dev flag, `D-TERRAIN-3`).
const GROUND_RUNG: u8 = 0;
const GROUND_RADIUS: i32 = 6;
const HILL_RUNG: u8 = 3;
const HILL_RADIUS: i32 = 6;
const ALOFT_RUNG: u8 = 9;
const ALOFT_RADIUS: i32 = 12;
/// The login and the terrain's arrival, in client ticks (20 Hz): a planet shard must boot and the
/// workers must build a few dozen chunks.
const LOGIN_DEADLINE: Duration = Duration::from_secs(120);
const TERRAIN_WAIT_TICKS: u64 = 1_800;
/// Where the pictures go for the owner.
const PICTURE_DIR: &str = "docs/investigation/2026-09-07/pictures";

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Fixture {
    base: PathBuf,
    trust_dir: PathBuf,
    common: Vec<(&'static str, String)>,
    store_str: String,
    launch_path: PathBuf,
    cwd: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "terrain_pictures: the fixture is KEPT for diagnosis at {}",
                self.base.display()
            );
            return;
        }
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn fixture() -> Fixture {
    let trust = ClusterTrust::generate("vd-terrain-pictures").expect("trust");
    let base = std::env::temp_dir().join(format!("vd-terrain-pictures-{}", std::process::id()));
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

struct ForkedReaper(PathBuf);
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

/// Boot the demand cluster with the spawn poses inside the home planet.
fn boot_demand_cluster(
    f: &Fixture,
    a: &ClusterAddrs,
    p: &DevClusterParams,
    spawn_poses: &str,
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
    let mut gw = gateway_env(a, &dev_auth_pubkey_hex(), p, ClusterShape::Demand);
    gw.push(("VD_SPAWN_POSES", spawn_poses.to_owned()));
    cluster.push(
        "vd-gateway",
        vd_bins::spawn_node(env!("CARGO_BIN_EXE_vd-gateway"), &f.common, &gw)
            .expect("spawn gateway"),
    );
    cluster
}

/// One picture's parameters: which account stands where (the agent index picks the account), the
/// rung and radius the flags name, and the verdict's band and floor.
#[derive(Clone, Copy)]
struct Picture {
    name: &'static str,
    agent_index: u64,
    rung: u8,
    radius: i32,
    band: (f64, f64),
    min_share: f64,
}

/// The capture client for one picture: the agent index picks the account, the flags pick the rung.
fn spawn_capture_client(
    f: &Fixture,
    gateway: SocketAddr,
    pic: &Picture,
    quic_port: u16,
    devctl_port: u16,
) -> Child {
    let Picture {
        name,
        agent_index,
        rung,
        radius,
        ..
    } = *pic;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    cmd.env(vd_client_render::terrain::RUNG_ENV, rung.to_string());
    cmd.env(vd_client_render::terrain::RADIUS_ENV, radius.to_string());
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
        "--capture-pilot",
    ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().expect("spawn capture client")
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
                 precondition: this gate needs a working adapter (WGPU_BACKENDS/WGPU_POWER_PREF)"
            );
        }
        assert!(
            started.elapsed() < LOGIN_DEADLINE,
            "the capture client never served dev-control on {port}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| {
        panic!("dev-control round-trip on {port} failed: {e} — capture client gone?")
    })
}

fn await_active(devctl: u16) -> DevState {
    let started = Instant::now();
    loop {
        if let Ok(DevResponse::State { state }) = dev_roundtrip(devctl, &DevRequest::State)
            && state.phase == DevPhase::Active
            && state.snapshots_applied >= 5
            && !state.realm_boxes.is_empty()
        {
            return state;
        }
        assert!(
            started.elapsed() < LOGIN_DEADLINE,
            "the demand login onto the planet never converged"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// The GROUND in a screenshot: pixels of the terrain's own warm paint (red over blue by a margin —
/// the realm outlines are blue-ish, the sky black, the missing-shader colour magenta) in a row band
/// `[y0, y1)` of the frame, outside the HUD's corner, as a share of that band.
fn paint_share(rgba: &[u8], w: usize, h: usize, y0: usize, y1: usize) -> f64 {
    let mut painted = 0u64;
    let mut total = 0u64;
    let mut y = y0;
    while y < y1.min(h) {
        let mut x = 0;
        while x < w {
            let hud = x < 320 && y < 180;
            if !hud {
                let i = (y * w + x) * 4;
                let (r, g, b) = (rgba[i], rgba[i + 1], rgba[i + 2]);
                let warm = r >= g && g >= b && r >= b + 8 && r > 24;
                if warm {
                    painted += 1;
                }
                total += 1;
            }
            x += 1;
        }
        y += 1;
    }
    painted as f64 / total.max(1) as f64
}

/// One picture: login, wait for the terrain on screen, look, capture, judge, copy.
fn take_picture(f: &Fixture, gateway: SocketAddr, pic: &Picture) -> (u64, f64) {
    let Picture {
        name,
        band,
        min_share,
        ..
    } = *pic;
    let client_quic = reserve_udp_addr();
    let devctl = reserve_tcp_addr().port();
    let mut client = ChildGuard(spawn_capture_client(
        f,
        gateway,
        pic,
        client_quic.port(),
        devctl,
    ));
    await_listener(devctl, &mut client.0);
    let landed = await_active(devctl);
    eprintln!(
        "terrain_pictures/{name}: landed in {:?} at {:?}, universe tick {:?}",
        landed.location,
        vd_bins::pixel::own_pose(&landed).map(|(p, _)| p),
        landed.universe_tick
    );
    // THE STAND'S FACING, MEASURED (M8-L): the delivered orientation against the radial under the
    // eye — how far the nose is below level, how far the avatar's up leans off the radial, and the
    // ROLL (the camera's right axis lifted out of the local horizontal). A rolled horizon in a
    // picture is a defect of the stand, never of the ground.
    if let Some((p, q)) = vd_bins::pixel::own_pose(&landed) {
        let radial = p.normalize();
        let up = q * DVec3::Y;
        let fwd = q * DVec3::NEG_Z;
        let right = q * DVec3::X;
        eprintln!(
            "terrain_pictures/{name}: facing — nose {:.2}° below level, up {:.2}° off the radial, roll {:.2}° (right axis lifted)",
            (-fwd.dot(radial)).asin().to_degrees(),
            up.dot(radial).clamp(-1.0, 1.0).acos().to_degrees(),
            right.dot(radial).clamp(-1.0, 1.0).asin().to_degrees()
        );
    }
    // The day side was computed at the clock's genesis: the login must be within a day of it (an
    // orbit is a year long; a day moves the star under one degree).
    assert!(
        landed.universe_tick.unwrap_or(0) < TICKS_PER_DAY,
        "{name}: the clock is not near its genesis: {:?}",
        landed.universe_tick
    );
    // The terrain arrives: the instrument, never a sleep.
    let live = round_trip(
        devctl,
        &DevRequest::WaitUntil {
            predicate: WaitPredicate {
                field: WaitField::TerrainChunksDrawn,
                op: WaitOp::Ge,
                value: 1,
            },
            max_ticks: TERRAIN_WAIT_TICKS,
        },
    );
    assert!(
        matches!(live, DevResponse::State { .. }),
        "{name}: terrain never reached the screen: {live:?}"
    );
    // The whole wanted set lands: nothing is still building. An instrument, never a sleep.
    let settled = round_trip(
        devctl,
        &DevRequest::WaitUntil {
            predicate: WaitPredicate {
                field: WaitField::TerrainChunksPending,
                op: WaitOp::Le,
                value: 0,
            },
            max_ticks: TERRAIN_WAIT_TICKS,
        },
    );
    assert!(
        matches!(settled, DevResponse::State { .. }),
        "{name}: the terrain never settled: {settled:?}"
    );
    let last = vd_bins::pixel::poll(devctl).terrain_chunks_drawn;
    // The account was born facing where the picture wants: capture.
    let shot = round_trip(
        devctl,
        &DevRequest::Screenshot {
            at_tick: None,
            label: Some(name.to_owned()),
        },
    );
    let rel = match shot {
        DevResponse::Captured { path, .. } => path,
        other => panic!("{name}: screenshot was not captured (GPU precondition?): {other:?}"),
    };
    let png = f.cwd.join(&rel);
    let img = image::open(&png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgba = img.as_raw();
    assert_eq!(magenta_pixel_count(rgba), 0, "{name}: zero magenta");
    let share = paint_share(
        rgba,
        w,
        h,
        (band.0 * h as f64) as usize,
        (band.1 * h as f64) as usize,
    );
    eprintln!(
        "terrain_pictures/{name}: {last} chunks drawn, paint share {share:.3} in rows {:?} of {h}",
        band
    );
    assert!(
        share >= min_share,
        "{name}: the ground is not in the picture: paint share {share:.3} < {min_share}"
    );
    // For the owner: beside the slice document.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(PICTURE_DIR);
    std::fs::create_dir_all(&dir).expect("the picture directory");
    let dest = dir.join(format!("{name}.png"));
    std::fs::copy(&png, &dest).expect("copy the picture");
    eprintln!("terrain_pictures/{name}: written to {}", dest.display());
    (last, share)
}

/// One standing account: its offset from the planet's centre and the facing it is born with.
struct Stand {
    offset_m: DVec3,
    orient: vd_core::glam::DQuat,
}

/// A stand `height_m` over the surface along `d`, facing `forward` with `d` as its up — the
/// orientation whose own `-Z` is `forward` and own `+Y` is the radial (glam's frame convention, the
/// same one the shard's kinematics and the pilot camera read).
fn stand(d: DVec3, height_m: f64, surface_m: f64, forward: DVec3) -> Stand {
    let up = (d - forward * forward.dot(d)).normalize();
    let right = forward.cross(up).normalize();
    let basis = vd_core::glam::DMat3::from_cols(right, up, -forward);
    Stand {
        offset_m: d * (surface_m + height_m),
        orient: vd_core::glam::DQuat::from_mat3(&basis).normalize(),
    }
}

/// The stand-in's entry for one account.
fn spawn_entry(account: u64, planet_seed: u64, st: &Stand) -> String {
    format!(
        "{account}=Planet({planet_seed}):{},{},{}@{},{},{},{}",
        st.offset_m.x,
        st.offset_m.y,
        st.offset_m.z,
        st.orient.x,
        st.orient.y,
        st.orient.z,
        st.orient.w
    )
}

#[test]
fn the_home_planet_is_seen_from_the_ground_and_from_aloft() {
    let _tier = vd_bins::cluster_tier();
    let body = vd_bins::home_body(DEV.universe_seed).expect("the home planet");
    let planet = vd_core::pose::RealmId::Planet(body.seed());
    // THE DAY SIDE: the star's direction from the planet's centre is minus the planet's position on
    // its orbit at the clock's genesis (the cluster boots at tick 0 — the universe tick read after
    // the login is asserted small below; the picture is taken seconds later, an orbital motion far
    // under one cell). The planet's frame is its parent's while the planet does not spin — ASSUMED
    // here (the renderer's debug line showed the identity facing on 2026-09-08); a planet that
    // spins one day makes this gate red with "the ground is not in the picture", and the fix is to
    // read the row's facing off the state, which does not carry it yet.
    let orbit = vd_bins::home_orbit(DEV.universe_seed).expect("the home planet's orbit");
    let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
        .position
        .normalize();
    // THE SPOT: on the radial where the star stands `SUN_ELEVATION_DEG` over the horizon, in the
    // orbit's plane (the planet's equator: the pole is `+Z`).
    let along = sun.cross(DVec3::Z).normalize();
    let zenith = (90.0_f64 - SUN_ELEVATION_DEG).to_radians();
    let d = (sun * zenith.cos() + along * zenith.sin()).normalize();
    let dir = [Gf::from_f64(d.x), Gf::from_f64(d.y), Gf::from_f64(d.z)];
    let h = vd_terrain::height::height_m(&body, dir, 0).to_f64();
    // The nose: the star's azimuth turned `SUN_OFF_NOSE_DEG` about the radial (the light comes over
    // the camera's shoulder), tilted below level by the picture's angle.
    let toward_sun = (sun - d * sun.dot(d)).normalize();
    let ahead =
        vd_core::glam::DQuat::from_axis_angle(d, SUN_OFF_NOSE_DEG.to_radians()) * toward_sun;
    let nose = |tilt_deg: f64| {
        let t = tilt_deg.to_radians();
        (ahead * t.cos() - d * t.sin()).normalize()
    };
    let ground = stand(d, EYE_HEIGHT_M, h, nose(GROUND_TILT_DEG));
    let hill = stand(d, HILL_M, h, nose(HILL_TILT_DEG));
    let aloft = stand(d, ALOFT_M, h, nose(ALOFT_TILT_DEG));
    let spawn_poses = [
        spawn_entry(CLIENT_ACCOUNT_BASE, body.seed(), &ground),
        spawn_entry(CLIENT_ACCOUNT_BASE + 1, body.seed(), &hill),
        spawn_entry(CLIENT_ACCOUNT_BASE + 2, body.seed(), &aloft),
    ]
    .join(";");
    let face = vd_seed::bend::face_of([d.x, d.y, d.z]);
    let (t, s) = vd_seed::bend::face_coords(face, [d.x, d.y, d.z]);
    eprintln!(
        "terrain_pictures: {planet:?}, surface {h:.1} m, star at {sun:?} ({SUN_ELEVATION_DEG}° up, \
         {SUN_OFF_NOSE_DEG}° off the nose), standing on {face:?} at ({t:.3}, {s:.3}); spawns \
         {spawn_poses}"
    );

    let f = fixture();
    let gw_admin = reserve_tcp_addr();
    let a = demand_addrs(gw_admin);
    let _reaper = ForkedReaper(f.launch_path.clone());
    let _cluster = boot_demand_cluster(&f, &a, &DEV, &spawn_poses);

    let (ground_chunks, ground_share) = take_picture(
        &f,
        a.gateway,
        &Picture {
            name: "ground",
            agent_index: 0,
            rung: GROUND_RUNG,
            radius: GROUND_RADIUS,
            band: (0.55, 1.0),
            min_share: 0.30,
        },
    );
    let (hill_chunks, hill_share) = take_picture(
        &f,
        a.gateway,
        &Picture {
            name: "hill",
            agent_index: 1,
            rung: HILL_RUNG,
            radius: HILL_RADIUS,
            band: (0.5, 1.0),
            min_share: 0.30,
        },
    );
    let (aloft_chunks, aloft_share) = take_picture(
        &f,
        a.gateway,
        &Picture {
            name: "aloft",
            agent_index: 2,
            rung: ALOFT_RUNG,
            radius: ALOFT_RADIUS,
            band: (0.5, 1.0),
            min_share: 0.30,
        },
    );
    eprintln!(
        "terrain_pictures: ground {ground_chunks} chunks ({ground_share:.3}), hill {hill_chunks} \
         chunks ({hill_share:.3}), aloft {aloft_chunks} chunks ({aloft_share:.3})"
    );
    assert!(
        ground_chunks >= 9,
        "the ground picture holds the columns around the eye"
    );
    assert!(hill_chunks >= 9, "the hill picture holds the columns below");
    assert!(
        aloft_chunks >= 9,
        "the aloft picture holds the columns below the eye"
    );
}
