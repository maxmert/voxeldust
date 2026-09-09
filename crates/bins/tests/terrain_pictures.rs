//! ★ THE FIRST PICTURES OF THE HOME PLANET (the voxel foundation, slice 7; ruling V12 S7-11, M7-4),
//! WITH THE PICTURE INSTRUMENT (slice 8p; ruling V14 D8-7, M8-4): the client links the one
//! generator, a login stands on the planet's surface, and the harness takes the pictures the owner
//! judges — from the ground at rung 0, from a hill at a middle rung, and from aloft at a coarse rung.
//! Every picture carries the STAMP (what the renderer measured), the PROBE (a second picture in
//! which every pixel says what drew it and how far it is), the RULER (a ball of known size where the
//! centre ray meets the ground) and one LIGHT at a stated angle.
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
//! **What is asserted (M8-4, the stamp's own truth).** The probe says the lower half of the frame is
//! terrain, at the rung the flag named, on every pixel, AND THE PICTURE IS JUDGED THERE: where the
//! probe says ground, the picture's own pixels carry the ground's lit paint; where it says the ball,
//! the picture is red (the probe knows only the terrain and the ruler, so a marker or a box over the
//! ground is caught by the picture's paint under the probe, not by the probe). The stamp's altitude
//! and horizon agree with what this gate recomputes from the state file beside the picture and the
//! recipe, in the row's DELIVERED facing; the stand held still across the capture; the star stands
//! 12°–18° up and 100°–140° off the nose in the renderer's own reading; the ruler's disc on the probe
//! measures what the stamp's stated ball projects to through the pilot camera, within one pixel;
//! the probe's near and far distance under the ball are the stamp's within one cell; there is drawn
//! ground under the ball; zero magenta. The pictures and their probes are copied beside the slice
//! document for the owner. GPU-required + LOCAL like every capture gate.
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
use vd_client::chunks::eye_surface;
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::capture::{probe_rel_for, state_rel_for};
use vd_client_harness::probe::{
    PROBE_KIND_RULER, PROBE_KIND_TERRAIN, equivalent_radius_px, horizon_m, probe_blob,
    probe_share_in_rows,
};
use vd_client_harness::verdict::projected_point_aabb;
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
/// THE LIGHT, ASSERTED (slice 8p): the renderer's own reading of the star must fall in the bands the
/// ruling names. The stand is built for 15° and 120°; a spinning planet or a wrong up would move it.
const SUN_ELEVATION_BAND_DEG: (f64, f64) = (12.0, 18.0);
const SUN_OFF_NOSE_BAND_DEG: (f64, f64) = (100.0, 140.0);
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
/// M8-4's tolerances. The stamp's altitude against the gate's own reading of the same recipe at the
/// same eye: the two differ only by the lattice reduction's rounding of the eye. The horizon is a
/// formula of that altitude. The ruler's disc against its projection: one pixel, the rasteriser's
/// own edge (the projection offsets a point along the camera's right axis, `f·r/d`; the true
/// silhouette is `f·r/√(d²−r²)`, which at the ball's `r/d = tan 2°` differs by 0.06 %, a fiftieth
/// of a pixel — and stays under a pixel until `r/d` passes 0.2, a hit under three cells, where the
/// half-cell floor binds). The probe's nearest and farthest distance under the ball against
/// `(d − r)/cell` and `d/cell`: one cell, the channel's own rounding — so a wrong high byte (256
/// cells) or a wrong low byte is caught, which is the G/B channel's exactness measured.
const ALTITUDE_TOLERANCE_M: f64 = 0.05;
const HORIZON_TOLERANCE_M: f64 = 1.0;
const RULER_TOLERANCE_PX: f64 = 1.0;
const RULER_CELLS_TOLERANCE: f64 = 1.0;
/// The picture under the probe: the share of the probe's ground pixels that carry the ground's lit
/// paint in the picture, and of its ball pixels that are red. A black or unlit picture with a
/// perfect probe fails here (the refuter's finding 1); a marker over the ground fails here.
const PAINT_UNDER_PROBE_MIN: f64 = 0.95;
const RED_UNDER_PROBE_MIN: f64 = 0.90;
/// Rows under the ball's disc that must be drawn ground: from three pixels under its rim to eight.
const GROUND_UNDER_BALL_ROWS: (usize, usize) = (3, 8);
/// A still stand: the delivered position in the state file and in a poll after the capture agree to
/// this, so the stamp, the pose and the probe describe one moment (a moving leg needs the straddle).
const STILL_TOLERANCE_M: f64 = 1e-6;
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

/// THE GROUND'S LIT PAINT in the picture (slice 7's classifier, now applied only where the probe says
/// ground): red over blue by a margin, not black.
fn warm(px: &[u8]) -> bool {
    let (r, g, b) = (px[0], px[1], px[2]);
    r >= g && g >= b && r >= b + 8 && r > 24
}

/// THE BALL'S PAINT in the picture: a red HUE — red at least twice each other channel, above black —
/// so the ball's shaded side counts as the ball (MEASURED on the fourth flight: a bright-margin test
/// read the shade as "not the ball" on a quarter of the disc), while the ground's tan (red under
/// twice green) does not.
fn red(px: &[u8]) -> bool {
    let (r, g, b) = (u16::from(px[0]), u16::from(px[1]), u16::from(px[2]));
    r > 12 && r >= 2 * g && r >= 2 * b
}

/// The share of the probe's pixels of `kind` whose PICTURE pixel satisfies `paint`.
fn paint_under_probe(picture: &[u8], probe: &[u8], kind: u8, paint: fn(&[u8]) -> bool) -> f64 {
    let (mut hits, mut total) = (0u64, 0u64);
    for (pic, pr) in picture.chunks_exact(4).zip(probe.chunks_exact(4)) {
        let here = vd_client_harness::probe::decode_probe([pr[0], pr[1], pr[2]]).kind == kind;
        total += u64::from(here);
        hits += u64::from(here && paint(pic));
    }
    hits as f64 / total.max(1) as f64
}

/// Open a captured PNG as RGBA8 with its size.
fn open_rgba(png: &Path) -> (Vec<u8>, usize, usize) {
    let img = image::open(png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    (img.into_raw(), w, h)
}

/// One picture: login, wait for the terrain on screen, look, capture, judge the picture, the
/// probe, the stamp and the ruler, copy for the owner.
fn take_picture(
    f: &Fixture,
    gateway: SocketAddr,
    body: &vd_terrain::BodyDefinition,
    planet: vd_core::pose::RealmId,
    pic: &Picture,
) -> (u64, f64) {
    let Picture {
        name,
        band,
        min_share,
        rung,
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
    let (rgba, w, h) = open_rgba(&png);
    assert_eq!(magenta_pixel_count(&rgba), 0, "{name}: zero magenta");
    // THE PROBE beside the picture, the same size.
    let probe_png = f.cwd.join(probe_rel_for(&rel));
    let (probe, pw, ph) = open_rgba(&probe_png);
    assert_eq!((pw, ph), (w, h), "{name}: the probe is the picture's size");
    // THE STATE FILE beside the picture: the listener wrote the delivered state at capture time.
    let run_dir = png
        .parent()
        .and_then(Path::parent)
        .expect("shots/<name>.png sits in a run dir");
    let state_path = run_dir.join(state_rel_for(&rel));
    let state: DevState = serde_json::from_str(
        &std::fs::read_to_string(&state_path)
            .unwrap_or_else(|e| panic!("{name}: read {}: {e}", state_path.display())),
    )
    .expect("the state file decodes");
    let stamp = state
        .terrain_stamp
        .clone()
        .unwrap_or_else(|| panic!("{name}: the stamp is on the state file"));
    eprintln!("terrain_pictures/{name}: stamp {stamp:?}");
    // THE STAND HELD STILL: the state file (polled after the capture) and a poll now agree on the
    // delivered position, so the stamp, the pose and the probe describe one moment.
    let again = vd_bins::pixel::poll(devctl);
    let (p_file, _) = vd_bins::pixel::own_pose(&state).expect("the file has the pose");
    let (p_now, _) = vd_bins::pixel::own_pose(&again).expect("the poll has the pose");
    assert!(
        (p_file - p_now).length() <= STILL_TOLERANCE_M,
        "{name}: the stand moved across the capture: {p_file:?} then {p_now:?}"
    );
    // 1. THE GROUND IS IN THE PICTURE — by the probe, not by paint: the share of terrain pixels in
    //    the lower band, every one of them at the rung the flag named (a bit-exact reading of the
    //    probe's channel through the sRGB target).
    let share = probe_share_in_rows(
        &probe,
        w,
        h,
        (band.0 * h as f64) as usize,
        (band.1 * h as f64) as usize,
        PROBE_KIND_TERRAIN,
    );
    eprintln!(
        "terrain_pictures/{name}: {last} chunks drawn, terrain share {share:.3} in rows {:?} of {h}",
        band
    );
    assert!(
        share >= min_share,
        "{name}: the ground is not in the picture: terrain share {share:.3} < {min_share}"
    );
    let terrain = probe_blob(&probe, w, PROBE_KIND_TERRAIN);
    assert_eq!(
        (terrain.rung_min, terrain.rung_max),
        (rung, rung),
        "{name}: every terrain pixel states the drawn rung: {terrain:?}"
    );
    assert_eq!(stamp.rung, rung, "{name}: the stamp states the drawn rung");
    assert_eq!(
        stamp.chunks_pending, 0,
        "{name}: the stamp was taken settled"
    );
    // THE PICTURE, JUDGED WHERE THE PROBE POINTS: the ground's lit paint under the probe's ground,
    // red under its ball. The probe alone would pass a black picture.
    let ground_paint = paint_under_probe(&rgba, &probe, PROBE_KIND_TERRAIN, warm);
    let ball_paint = paint_under_probe(&rgba, &probe, PROBE_KIND_RULER, red);
    eprintln!(
        "terrain_pictures/{name}: paint under the probe — ground {ground_paint:.3}, ball {ball_paint:.3}"
    );
    assert!(
        ground_paint >= PAINT_UNDER_PROBE_MIN,
        "{name}: the picture is not lit ground where the probe says ground: {ground_paint:.3}"
    );
    assert!(
        ball_paint >= RED_UNDER_PROBE_MIN,
        "{name}: the picture is not the red ball where the probe says ball: {ball_paint:.3}"
    );
    // 2. THE STAMP'S OWN TRUTH (M8-4): the eye from the state's pose through the pilot camera, in
    //    the planet's frame — its row's centre and its DELIVERED facing, the same four numbers the
    //    renderer turns the terrain by — against the recipe.
    let camera = vd_bins::pixel::pilot_camera(&state, w, h);
    let label = format!("{planet:?}");
    let row = state
        .realm_boxes
        .iter()
        .find(|b| b.realm == label)
        .expect("the planet's row is drawn");
    let centre = DVec3::from_array(row.center);
    let facing =
        vd_core::glam::DQuat::from_xyzw(row.facing[0], row.facing[1], row.facing[2], row.facing[3])
            .normalize();
    let eye_body = facing.inverse() * (camera.eye - centre);
    let ground = eye_surface(body, eye_body.to_array()).expect("the eye has a direction");
    assert!(
        (stamp.altitude_m - ground.altitude_m).abs() <= ALTITUDE_TOLERANCE_M,
        "{name}: the stamp's altitude {:.3} m vs the state's {:.3} m",
        stamp.altitude_m,
        ground.altitude_m
    );
    assert!(
        (stamp.surface_m - ground.surface_m).abs() <= ALTITUDE_TOLERANCE_M,
        "{name}: the stamp's surface {:.3} m vs the state's {:.3} m",
        stamp.surface_m,
        ground.surface_m
    );
    assert_eq!(
        stamp.biome,
        format!("{:?}", ground.biome),
        "{name}: the stamp's biome"
    );
    let expected_horizon = horizon_m(ground.surface_m, ground.altitude_m);
    assert!(
        (stamp.horizon_m - expected_horizon).abs() <= HORIZON_TOLERANCE_M,
        "{name}: the stamp's horizon {:.1} m vs {:.1} m",
        stamp.horizon_m,
        expected_horizon
    );
    // 3. THE LIGHT, in the renderer's own reading.
    let star = stamp
        .star
        .unwrap_or_else(|| panic!("{name}: the ground is lit by the star, not a work light"));
    assert!(
        (SUN_ELEVATION_BAND_DEG.0..=SUN_ELEVATION_BAND_DEG.1).contains(&star.elevation_deg),
        "{name}: the star stands {:.2}° up, outside {SUN_ELEVATION_BAND_DEG:?}",
        star.elevation_deg
    );
    assert!(
        (SUN_OFF_NOSE_BAND_DEG.0..=SUN_OFF_NOSE_BAND_DEG.1).contains(&star.off_nose_deg),
        "{name}: the star stands {:.2}° off the nose, outside {SUN_OFF_NOSE_BAND_DEG:?}",
        star.off_nose_deg
    );
    // 4. THE RULER: the stamp's stated ball, projected through the pilot camera, against its disc on
    //    the probe — the radius within one pixel, the centroid within one pixel — and the probe's
    //    distance under it against the stamp's.
    let ruler = stamp
        .ruler
        .unwrap_or_else(|| panic!("{name}: the centre ray meets the drawn ground"));
    let rect = projected_point_aabb(
        &camera,
        camera.eye + DVec3::from_array(ruler.centre_m),
        ruler.radius_m,
    )
    .expect("the ruler is ahead of the eye");
    let predicted_px = (rect.max.x - rect.min.x) * 0.5;
    let predicted_centre = (
        f64::midpoint(rect.min.x, rect.max.x),
        f64::midpoint(rect.min.y, rect.max.y),
    );
    let disc = probe_blob(&probe, w, PROBE_KIND_RULER);
    let measured_px = equivalent_radius_px(disc.count);
    let centroid = disc.centroid.expect("the ruler is on the probe");
    eprintln!(
        "terrain_pictures/{name}: ruler r {:.3} m at {:.1} m — predicted {predicted_px:.2} px at \
         ({:.1}, {:.1}), measured {measured_px:.2} px at ({:.1}, {:.1}) from {} pixels, cells \
         {}..{} of {} m",
        ruler.radius_m,
        ruler.distance_m,
        predicted_centre.0,
        predicted_centre.1,
        centroid.0,
        centroid.1,
        disc.count,
        disc.cells_min,
        disc.cells_max,
        stamp.cell_m
    );
    assert!(
        (measured_px - predicted_px).abs() <= RULER_TOLERANCE_PX,
        "{name}: the ruler's disc measures {measured_px:.2} px, its projection {predicted_px:.2} px"
    );
    let drift = (centroid.0 - predicted_centre.0).hypot(centroid.1 - predicted_centre.1);
    assert!(
        drift <= RULER_TOLERANCE_PX,
        "{name}: the ruler's centroid stands {drift:.2} px from its projection"
    );
    assert_eq!(
        (disc.rung_min, disc.rung_max),
        (rung, rung),
        "{name}: the ruler's pixels state the drawn rung"
    );
    // The G/B channel, measured: the ball's nearest pixel is `d − r` off, its rim `√(d² − r²)`
    // (within a fiftieth of a cell of `d` at this angular size), each within one cell.
    let near_cells = (ruler.distance_m - ruler.radius_m) / stamp.cell_m;
    let rim_cells = (ruler.distance_m * ruler.distance_m - ruler.radius_m * ruler.radius_m).sqrt()
        / stamp.cell_m;
    assert!(
        (f64::from(disc.cells_min) - near_cells).abs() <= RULER_CELLS_TOLERANCE,
        "{name}: the probe's nearest ruler distance {} cells vs the stamp's {near_cells:.1}",
        disc.cells_min
    );
    assert!(
        (f64::from(disc.cells_max) - rim_cells).abs() <= RULER_CELLS_TOLERANCE,
        "{name}: the probe's farthest ruler distance {} cells vs the stamp's rim {rim_cells:.1}",
        disc.cells_max
    );
    // DRAWN GROUND UNDER THE BALL: the rows just under its rim, at its centroid's column, are
    // terrain on the probe — the ball hovers over ground the picture shows, not over nothing.
    let cx = centroid.0.round() as usize;
    let rim_y = (centroid.1 + measured_px).round() as usize;
    let mut k = GROUND_UNDER_BALL_ROWS.0;
    while k <= GROUND_UNDER_BALL_ROWS.1 {
        let y = rim_y + k;
        assert!(y < h, "{name}: the ball's rim is at the frame's bottom");
        let i = (y * w + cx) * 4;
        let px = vd_client_harness::probe::decode_probe([probe[i], probe[i + 1], probe[i + 2]]);
        assert_eq!(
            px.kind, PROBE_KIND_TERRAIN,
            "{name}: no drawn ground {k} px under the ball at ({cx}, {y}): {px:?}"
        );
        k += 1;
    }
    // For the owner: beside the slice document, the picture and its probe.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(PICTURE_DIR);
    std::fs::create_dir_all(&dir).expect("the picture directory");
    let dest = dir.join(format!("{name}.png"));
    std::fs::copy(&png, &dest).expect("copy the picture");
    std::fs::copy(&probe_png, dir.join(format!("{name}.probe.png"))).expect("copy the probe");
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
        &body,
        planet,
        &Picture {
            name: "ground",
            agent_index: 0,
            rung: GROUND_RUNG,
            radius: GROUND_RADIUS,
            band: (0.55, 1.0),
            min_share: 0.95,
        },
    );
    let (hill_chunks, hill_share) = take_picture(
        &f,
        a.gateway,
        &body,
        planet,
        &Picture {
            name: "hill",
            agent_index: 1,
            rung: HILL_RUNG,
            radius: HILL_RADIUS,
            band: (0.5, 1.0),
            min_share: 0.95,
        },
    );
    let (aloft_chunks, aloft_share) = take_picture(
        &f,
        a.gateway,
        &body,
        planet,
        &Picture {
            name: "aloft",
            agent_index: 2,
            rung: ALOFT_RUNG,
            radius: ALOFT_RADIUS,
            band: (0.5, 1.0),
            min_share: 0.95,
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
