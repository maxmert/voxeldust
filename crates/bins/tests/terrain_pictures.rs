//! ★ THE PICTURES OF THE HOME PLANET TO THE HORIZON (the voxel foundation, slice 7; ruling V12
//! S7-11, M7-4; slice 8 step 2, ruling V14 D8-1, D8-6), WITH THE PICTURE INSTRUMENT (slice 8p;
//! ruling V14 D8-7, M8-4): the client links the one generator, a login stands on the planet's
//! surface, and the harness takes the pictures the owner judges — from the ground, from a hill, from
//! aloft, and from orbit where the globe's limb is in frame. The client draws THE LADDER: every ring
//! from the rung under the eye out to the horizon and the peaks behind it, by the tier rule (one
//! cell = one pixel at the reference view), no flag. Every picture carries the STAMP (what the
//! renderer measured), the PROBE (a second picture in which every pixel says what drew it, at which
//! rung, and how far it is), the RULER (a ball of known size where the centre ray meets the ground)
//! and one LIGHT at a stated angle.
//!
//! **The path is the shipped one.** A DEMAND cluster (orchestrator + gateway, no shard pre-booked);
//! the stand-in spawn poses put three accounts INSIDE the home planet's realm (slice 7 gave the
//! stand-in a realm name and a facing): on the planet's DAY SIDE — the star's direction comes from the
//! planet's own orbit, the same elements its shard authors — each born with the radial as its up and
//! its nose toward the star, a little below level. One stands 1.8 m over the surface the recipe
//! states, one 300 m over it, one 60 km over it, one 2 000 km over it with its nose 45° down so the
//! limb is in frame. The login demands the planet's chain; the planet's shard boots and states its
//! surface; the capture client draws the ladder and answers the harness with `terrain_chunks_drawn`
//! and `terrain_chunks_pending`, the instruments this gate waits on — never a sleep standing in for
//! a signal.
//!
//! **What is asserted (M8-4, the stamp's own truth; step 2, the ladder).** The probe says the lower
//! half of the frame is terrain, and EVERY TERRAIN PIXEL'S RUNG IS THE TIER RULE'S RUNG FOR ITS OWN
//! DISTANCE within one rung (the boundary runs by the column's centre, a pixel by its own point);
//! the finest rung on screen is the rule's rung at the eye's height; the ladder reaches past the
//! horizon and the terrain in the picture's centre column reaches up to the horizon's own row, so
//! there is no black between the ground and the sky; AND THE PICTURE IS JUDGED THERE: where the
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
use vd_bins::memory::{
    MemoryRead, footprint_report, gpu_busy_mean, heap_summary, report_rows, resident_mb,
    vmmap_rows, vmmap_summary,
};
use vd_bins::{
    Cluster, ClusterAddrs, ClusterShape, DEV, DevClusterParams, common_env, dev_auth_pubkey_hex,
    dev_auth_signing_key_hex, dev_roundtrip, gateway_env, launch_rows, orchestrator_env,
    reap_forked, reserve_tcp_addr, reserve_udp_addr,
};
use vd_client::chunks::eye_surface;
use vd_client::ladder_view::{in_fade_band, rung_for_distance};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::capture::{probe_rel_for, state_rel_for};
use vd_client_harness::probe::{
    PROBE_KIND_NONE, PROBE_KIND_RULER, PROBE_KIND_TERRAIN, decode_probe, equivalent_radius_px,
    ground_holes, horizon_m, probe_blob, probe_share_in_rows,
};
use vd_client_harness::verdict::projected_point_aabb;
use vd_core::glam::DVec3;
use vd_devproto::{DevPhase, DevRequest, DevResponse, DevState, WaitField, WaitOp, WaitPredicate};
use vd_io_prod::trust::ClusterTrust;

/// The client's account for `--agent-index N` (the client binary's own rule).
const CLIENT_ACCOUNT_BASE: u64 = 1000;
/// Eye height over the surface for the ground picture, in metres.
const EYE_HEIGHT_M: f64 = 1.8;
/// Height over the surface for the hill picture, in metres.
const HILL_M: f64 = 300.0;
/// Altitude for the aloft picture, in metres.
const ALOFT_M: f64 = 60_000.0;
/// Altitude for the orbit picture, in metres: the globe's limb in frame (D8-6).
const ORBIT_M: f64 = 2_000_000.0;
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
/// (A FEATURE STAND — the sharpest metre-wide bump or pit within 60 m of the spot, seen from 4 m
/// with the low sun throwing its shadow — was built and MEASURED 2026-09-12: the sharpest cell
/// holds 0.01 m of relief over its neighbours two metres off; the recipe's finest octave has no
/// metre-wide feature there. The stand waits for the block store's own features.)
/// THE SEAM STAND: the eye on the edge of the spot's cube face, at the point whose sun elevation is
/// nearest the spot's, looking along the seam tilted as the ground stand is — the face bend must not
/// show (SL8: a seam is a defect). The edge is searched in this many steps.
const SEAM_STEPS: i32 = 400;
const HILL_TILT_DEG: f64 = 15.0;
const ALOFT_TILT_DEG: f64 = 15.0;
/// From orbit the horizon dips 40° below level; a nose 45° down puts the limb 5° over the frame's
/// centre and the centre ray on the ground inside it.
const ORBIT_TILT_DEG: f64 = 45.0;
/// A day of universe ticks at the dev cluster's rate: the bound on how far from the clock's genesis
/// a picture may be taken while its day side is computed at genesis.
const TICKS_PER_DAY: u64 = 86_400 * 20;
/// The login and the terrain's arrival, in client ticks (20 Hz): a planet shard must boot and the
/// workers must build the ladder — MEASURED about 4 300 chunks from the ground at 4 ms each on one
/// thread, so a few seconds on every core; the wait allows a slow machine three minutes.
const LOGIN_DEADLINE: Duration = Duration::from_secs(120);
const TERRAIN_WAIT_TICKS: u64 = 3_600;
/// A terrain pixel's rung against the tier rule at the pixel's own distance: inside a crossfade
/// band a pixel belongs to either of the band's two rungs by design, so the rule is asserted
/// within one rung there; OUTSIDE every band a pixel sits on the rule's rung, and the share of
/// such pixels that do must be high (MEASURED before the bands: 99.9 % on the ground, 96.6 % from
/// the hill, 93.8 % aloft), so a rule off by one everywhere cannot pass.
const RUNG_TOLERANCE: u8 = 1;
const RUNG_EXACT_SHARE_MIN: f64 = 0.9;

/// In EVERY column of the picture the topmost drawn pixel reaches at least up to the row of the
/// horizon of the LOWEST GROUND THE RECIPE CAN RAISE — the sphere of the surface under the eye
/// minus the recipe's relief bound — within this many rows. A skyline can stand under the sphere
/// through the eye's own surface wherever the ground toward the horizon is lower (MEASURED on the
/// ground stand: 28 rows, 1.75°, at the far left, where the stand's own hill overlooks a valley),
/// and never under the horizon of the lowest ground; from orbit the two differ by three rows, so a
/// missing cap at the limb is red there, and on the ground a missing chunk is a block.
const HORIZON_ROW_TOLERANCE: usize = 3;
/// The chunk count each stand draws, at least (MEASURED: 3 855 / 4 300 / 2 012 / 1 277).
const GROUND_CHUNKS_MIN: u64 = 2_000;
const HILL_CHUNKS_MIN: u64 = 2_000;
const ALOFT_CHUNKS_MIN: u64 = 1_000;
const ORBIT_CHUNKS_MIN: u64 = 500;
/// M8-4's tolerances. The stamp's altitude against the gate's own reading of the same recipe at the
/// same eye: the two differ only by the lattice reduction's rounding of the eye. The horizon is a
/// formula of that altitude. The ruler's disc against its projection: one pixel, the rasteriser's
/// own edge (the projection offsets a point along the camera's right axis, `f·r/d`; the true
/// silhouette is `f·r/√(d²−r²)`, which at the ball's `r/d = tan 2°` differs by 0.06 %, a fiftieth
/// of a pixel — and stays under a pixel until `r/d` passes 0.2, a hit under three cells, where the
/// half-cell floor binds). The probe's nearest and farthest distance under the ball against
/// `(d − r)/cell` and `√(d² − r²)/cell`: two cells — one of geometric rounding at the rim and one
/// step of the target's own 8-bit rounding on the low byte (MEASURED on the hill at 2 m cells:
/// 576 against 574.8) — so a wrong high byte (256 cells) is caught; the R byte's exactness is
/// asserted on every terrain pixel, the G/B channel's to within one byte.
const ALTITUDE_TOLERANCE_M: f64 = 0.05;
const HORIZON_TOLERANCE_M: f64 = 1.0;
const RULER_TOLERANCE_PX: f64 = 1.0;
const RULER_CELLS_TOLERANCE: f64 = 2.0;
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
/// Set in the environment, the gate refuses a picture that differs from the one on disk by one
/// pixel (ruling V16: a packing changes bytes, never the picture).
const PICTURE_IDENTICAL_ENV: &str = "VD_PICTURE_IDENTICAL";
/// A LOOK MEASUREMENT (D8-8): with this set the compare reports and keeps its pictures but
/// withholds the tolerance's verdict, and the owner's pictures are left untouched. Never set by
/// a gate.
const PICTURE_REPORT_ONLY_ENV: &str = "VD_PICTURE_REPORT_ONLY";
/// A look measurement's capture grid, in multiples of the gate's.
const LOOK_TICK_FACTOR: u64 = 3;
/// Whether this stand freezes on this run: `VD_PICTURE_FREEZE` empty or `1` freezes every stand; a
/// comma-separated list of names freezes those alone — a new stand's first reference, while the
/// others' look stays the owner's to accept.
fn freeze_wanted(name: &str) -> bool {
    std::env::var(PICTURE_FREEZE_ENV)
        .is_ok_and(|v| v.is_empty() || v == "1" || v.split(',').any(|s| s.trim() == name))
}

/// THE FREEZE (ruling V18): with this set the run writes its pictures as the frozen exact
/// references and compares nothing — used ONCE, on the owner's acceptance of a look from the
/// difference images, never by a gate. The next run compares against them.
const PICTURE_FREEZE_ENV: &str = "VD_PICTURE_FREEZE";
/// AN ABLATION FLIGHT (D8-8): with this set the census prints and the stand ends — no picture
/// judgement, no compare, the owner's pictures untouched. Never set by a gate.
const PICTURE_CENSUS_ONLY_ENV: &str = "VD_PICTURE_CENSUS_ONLY";
/// The GPU's busy share: readings and their spacing.
const GPU_SAMPLES: u32 = 10;
const GPU_SAMPLE_GAP_MS: u64 = 200;
/// How many ticks before the capture tick the test asks for the capture (under the client's own
/// bound on a deferred capture, 600 ticks).
const CAPTURE_ASK_AHEAD_TICKS: u64 = 200;
/// THE TOLERANCE (ruling V18): no content pixel may change by more than this many brightness
/// levels, in any channel, against the EXACT picture. A packing that flips a crease pixel to
/// the other slope (19 levels, the 16-bit radial) is refused; one that shades it a level darker
/// is not.
const TOLERANCE_LEVELS: u8 = 1;
/// The stands whose picture changed past the tolerance, judged together at the flight's end.
static PAST_TOLERANCE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
/// The exact pictures' directory beside the owner's: frozen from an exact flight, never
/// overwritten by a run, the reference every compare reads when it exists.
const EXACT_DIR: &str = "exact";
/// THE CAPTURE TICK: a stand's picture is taken at a CONSTANT universe tick — this many ticks per
/// stand, in the stands' order — so two flights of one stand capture the SAME moment of the world
/// (the star at the same angle) and their pixels compare. MEASURED before this: two flights of one
/// code differed by 277–593 pixels, the widest channel step up to 140, because each captured at
/// its own tick. A stand that settles past its tick is a red gate, never a silent shift to the
/// next (refutation P-4). One minute of ticks a stand: the longest settle is 12 s.
const CAPTURE_TICK_GRID: u64 = 1_200;
/// How far the overlay's rectangle grows on every side before the compare leaves it out: the
/// glyphs' antialiasing.
const HUD_MARGIN_PX: f32 = 2.0;

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
/// nose's tilt below level (for the horizon's row), and the verdict's band and floor.
#[derive(Clone, Copy)]
struct Picture {
    name: &'static str,
    agent_index: u64,
    tilt_deg: f64,
    band: (f64, f64),
    min_share: f64,
    /// Where the star may stand off the nose, in degrees: over the shoulder for the stands that
    /// choose their nose; anywhere behind the shoulder for the seam, whose nose the edge chooses
    /// (MEASURED: no edge point has the star both low and 100°–140° off a nose along the edge).
    off_nose_band: (f64, f64),
}

/// The capture client for one picture: the agent index picks the account. No flag: the client
/// draws the ladder (slice 8 step 2).
fn spawn_capture_client(
    f: &Fixture,
    gateway: SocketAddr,
    pic: &Picture,
    quic_port: u16,
    devctl_port: u16,
) -> Child {
    let Picture {
        name, agent_index, ..
    } = *pic;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    for (k, v) in &f.common {
        cmd.env(k, v);
    }
    cmd.env("VD_AUTH_SIGNING_KEY", dev_auth_signing_key_hex());
    // THE INSTRUMENT RUNS AT THE WHOLE MACHINE (ruling F6): the game's default is a share of the
    // cores, but the stands' capture grids and the flights' settle deadlines were measured at every
    // core, so the harness states the whole count unless the knob names another (the knob is how
    // the share itself is measured). MEASURED: at the share of three the ground stand settled at
    // tick 2 389 against a capture tick of 1 200.
    if std::env::var_os("VD_TERRAIN_WORKERS").is_none() {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        cmd.env("VD_TERRAIN_WORKERS", cores.to_string());
    }
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

/// The footprint table's rows the census prints: the largest categories after the header.
const FOOTPRINT_ROWS: usize = 12;
/// The heap summary's rows the census prints: the zone lines after its header.
const HEAP_ROWS: usize = 6;
/// The footprint table's header: the banner, the blank, the column names and their dashes.
const FOOTPRINT_HEADER_ROWS: usize = 6;
/// The heap summary's header: the process block, the footprint block and their blanks.
const HEAP_HEADER_ROWS: usize = 21;

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
        tilt_deg,
        off_nose_band,
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
    // The terrain arrives: the instrument, never a sleep. The fill is timed from here to the
    // settle (M8-2's census).
    let fill_started = Instant::now();
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
    let fill_s = fill_started.elapsed().as_secs_f64();
    let last = vd_bins::pixel::poll(devctl).terrain_chunks_drawn;
    // THE FRAME RATE on the still stand (M8-2): the renderer's own frame counter over a timed
    // interval, read from the stamp.
    let frames_at = |devctl: u16| {
        vd_bins::pixel::poll(devctl)
            .terrain_stamp
            .map_or(0, |s| s.frames)
    };
    let frames_before = frames_at(devctl);
    let frame_window = Instant::now();
    std::thread::sleep(Duration::from_secs(2));
    let frames_per_s =
        (frames_at(devctl) - frames_before) as f64 / frame_window.elapsed().as_secs_f64();
    // The account was born facing where the picture wants: capture, at the grid's next tick.
    let settled_tick = vd_bins::pixel::poll(devctl).universe_tick.unwrap_or(0);
    // A look measurement captures on a coarser grid (MEASURED: the splat's quad form, four copies
    // of every vertex, settles the ground stand at tick 1 734 to 2 847 against 710 for the mesh),
    // and its reference is an exact flight on the SAME coarser grid, never the gate's pictures.
    let look = std::env::var_os(PICTURE_REPORT_ONLY_ENV).is_some();
    let capture_tick =
        CAPTURE_TICK_GRID * (pic.agent_index + 1) * if look { LOOK_TICK_FACTOR } else { 1 };
    eprintln!(
        "terrain_pictures/{name}: settled at tick {settled_tick}; the capture waits for tick \
         {capture_tick}"
    );
    assert!(
        settled_tick < capture_tick,
        "{name}: the terrain settled at tick {settled_tick}, past this stand's capture tick \
         {capture_tick}: the picture would not be the same moment as the last flight's"
    );
    // The client holds a deferred capture for a bounded count of ticks; the test waits here for
    // the grid's tick to come near before it asks (a look measurement's coarse grid sits past
    // that bound).
    while vd_bins::pixel::poll(devctl).universe_tick.unwrap_or(0) + CAPTURE_ASK_AHEAD_TICKS
        < capture_tick
    {
        std::thread::sleep(Duration::from_millis(250));
    }
    let shot = round_trip(
        devctl,
        &DevRequest::Screenshot {
            at_tick: Some(capture_tick),
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
    // THE CLIENT'S RESIDENT MEMORY (M8-2): what the capture client holds in RAM at the settled
    // stand, from the operating system's own count.
    let footprint_report = footprint_report(client.0.id());
    let memory = MemoryRead {
        resident_mb: resident_mb(client.0.id()),
        ..MemoryRead::from_footprint(&footprint_report)
    };
    eprintln!(
        "terrain_pictures/{name}: M8-2 CENSUS — {} chunks, {} vertices, {:.1} MB on screen ({:.0} \
         KB a chunk), filled in {fill_s:.1} s, {frames_per_s:.1} frames/s on the still stand, the \
         client at {memory}",
        stamp.chunks_drawn,
        stamp.vertices,
        stamp.bytes_drawn as f64 / 1.0e6,
        if stamp.chunks_drawn > 0 {
            stamp.bytes_drawn as f64 / stamp.chunks_drawn as f64 / 1024.0
        } else {
            0.0
        }
    );
    // THE SHADOW LADDER: the coarse casters on the shadow layer and their bytes.
    eprintln!(
        "terrain_pictures/{name}: SHADOW LADDER — {} coarse casters, {:.1} MB",
        stamp.shadow_casters,
        stamp.shadow_bytes as f64 / 1.0e6
    );
    // THE FRAME'S ANATOMY (D8-8): the frame's time and every render pass's CPU and GPU time,
    // smoothed, so the wall of a still stand is named before anything is built against it.
    eprintln!(
        "terrain_pictures/{name}: FRAME ANATOMY — {:.1} ms a frame; passes (cpu ms / gpu ms): {}",
        stamp.frame_ms,
        stamp
            .passes_ms
            .iter()
            .map(|(p, c, g)| format!("{p} {c:.2}/{g:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // THE GPU'S BUSY SHARE while the stand holds still (D8-8's ablation): ten readings over two
    // seconds of the driver's own statistics.
    if let Some((device, renderer)) =
        gpu_busy_mean(GPU_SAMPLES, Duration::from_millis(GPU_SAMPLE_GAP_MS))
    {
        eprintln!(
            "terrain_pictures/{name}: GPU BUSY — device {device:.0} %, renderer {renderer:.0} % \
             over {GPU_SAMPLES} readings"
        );
    }
    if std::env::var_os(PICTURE_CENSUS_ONLY_ENV).is_some() {
        eprintln!(
            "terrain_pictures/{name}: CENSUS ONLY — the picture's judgements and the compare are \
             skipped (an ablation flight)"
        );
        return (last, 0.0);
    }
    // WHERE THE FOOTPRINT GOES: the tool's largest categories (the GPU's buffers on unified
    // memory, the allocator's large blocks, the compressed pages), for the memory report.
    for row in report_rows(&footprint_report, FOOTPRINT_HEADER_ROWS, FOOTPRINT_ROWS) {
        eprintln!("terrain_pictures/{name}: footprint {row}");
    }
    // WHAT THE ALLOCATOR HOLDS (`heap`): the blocks in use by zone and by size class, so a
    // footprint category can be read as held or as freed-and-retained.
    for row in report_rows(&heap_summary(client.0.id()), HEAP_HEADER_ROWS, HEAP_ROWS) {
        eprintln!("terrain_pictures/{name}: heap {row}");
    }
    // WHAT THE REGIONS ARE (`vmmap`): every region type with its sizes, so the footprint's
    // owned-but-unmapped memory gets a name.
    for row in vmmap_rows(&vmmap_summary(client.0.id())) {
        eprintln!("terrain_pictures/{name}: vmmap {row}");
    }
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
    //    the lower band; and THE LADDER: every terrain pixel's rung is the tier rule's rung for the
    //    pixel's own distance (its cells times its rung's cell), within one rung (a bit-exact reading
    //    of the probe's channel through the sRGB target, judged by the rule).
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
    // NO BLOCK OF NOTHING IN THE GROUND: a hole that survives a one-pixel erosion is a missing
    // chunk (MEASURED on the first ladder pictures: black rectangles at the ring boundaries,
    // 4 818 pixels on the hill, which the band's share let through).
    let holes = ground_holes(&probe, w, h);
    eprintln!(
        "terrain_pictures/{name}: {} hole pixels under drawn ground, {} in blocks",
        holes.pixels, holes.blocks
    );
    assert_eq!(
        holes.blocks, 0,
        "{name}: {} pixels of nothing under drawn ground survive erosion — a missing chunk",
        holes.blocks
    );
    // NO CRACK EITHER (step 3): two rungs never meet at a mesh edge now — the finer slides onto the
    // coarser surface across its band while the coarser is drawn whole beneath it — so no pixel
    // of nothing lies under drawn ground.
    assert_eq!(
        holes.pixels, 0,
        "{name}: {} pixels of nothing under drawn ground — a crack between two rungs",
        holes.pixels
    );
    let terrain = probe_blob(&probe, w, PROBE_KIND_TERRAIN);
    let rungs = body.ladder().rungs;
    let mut off_by_one = 0u64;
    let mut exact = 0u64;
    let mut outside = 0u64;
    for px in probe.chunks_exact(4) {
        let p = decode_probe([px[0], px[1], px[2]]);
        if p.kind == PROBE_KIND_TERRAIN {
            let d = f64::from(p.cells) * f64::from(vd_seed::ladder::cell_m(p.rung));
            let rule = rung_for_distance(d, rungs);
            let gap = rule.abs_diff(p.rung);
            assert!(
                gap <= RUNG_TOLERANCE,
                "{name}: a terrain pixel at rung {} states {} cells ({d:.0} m), where the rule says \
                 rung {rule}",
                p.rung,
                p.cells
            );
            let in_band = in_fade_band(d, rungs);
            off_by_one += u64::from(gap == 1);
            exact += u64::from((gap == 0) & !in_band);
            outside += u64::from(!in_band);
        }
    }
    let exact_share = exact as f64 / outside.max(1) as f64;
    assert!(
        exact_share >= RUNG_EXACT_SHARE_MIN,
        "{name}: only {exact_share:.3} of the terrain pixels outside the bands sit on the rule's \
         rung"
    );
    eprintln!(
        "terrain_pictures/{name}: rungs {}..{} on the probe, {} of {} terrain pixels one rung off \
         the rule at their own distance; the stamp draws {:?}",
        terrain.rung_min, terrain.rung_max, off_by_one, terrain.count, stamp.chunks_per_rung
    );
    // The finest rung ON SCREEN (the probe's) is the rule's rung at the eye's height — or, when
    // that height lies inside a crossfade band, the finer rung that is still fading out there
    // (step 3: MEASURED aloft, the eye at 60 km inside rung 6's band of 50–61 km, rung 6 under
    // it). The stamp counts what is RESIDENT, which may hold one finer rung more: a column that
    // straddles its fade-out edge is resident while every fragment of it lies past the edge and
    // is discarded (MEASURED from orbit: 29 rung-11 chunks under the eye, no rung-11 pixel).
    let rule_at_eye = rung_for_distance(stamp.altitude_m, rungs);
    let in_band_at_eye = in_fade_band(stamp.altitude_m, rungs);
    assert!(
        terrain.rung_min == rule_at_eye || (in_band_at_eye && terrain.rung_min + 1 == rule_at_eye),
        "{name}: the finest rung on screen ({}) is neither the rule's rung at the eye's height \
         ({rule_at_eye}) nor, inside a band ({in_band_at_eye}), the one below it",
        terrain.rung_min
    );
    assert!(
        stamp.rung_min + 1 >= rule_at_eye && stamp.rung_min <= terrain.rung_min,
        "{name}: the finest resident rung ({}) is more than one under the rule's rung at the \
         eye's height ({rule_at_eye}), or finer than what is on screen ({})",
        stamp.rung_min,
        terrain.rung_min
    );
    // More than one rung is on screen, unless the eye's own rung is the top of the ladder (from
    // orbit the whole cap is one ring).
    assert!(
        stamp.rung_max > stamp.rung_min || stamp.rung_min + 1 == rungs,
        "{name}: the ladder holds one rung {}..{} under a ladder of {rungs}",
        stamp.rung_min,
        stamp.rung_max
    );
    assert!(
        stamp.drawn_radius_m >= stamp.horizon_m,
        "{name}: the ladder reaches {:.0} m, short of the horizon at {:.0} m",
        stamp.drawn_radius_m,
        stamp.horizon_m
    );
    // THE GEOMORPH'S TARGETS stand on the parent mesh: at most one vertex in a hundred fell back
    // to the coarser field (a cave's own vertex, or a coarser surface past its parents).
    assert!(
        stamp.morph_fallbacks * 100 <= stamp.vertices,
        "{name}: {} of {} morph targets fell back to the field ({} on a face seam by rule)",
        stamp.morph_fallbacks,
        stamp.vertices,
        stamp.morph_seam
    );
    eprintln!(
        "terrain_pictures/{name}: morph targets — {} fallbacks and {} seam vertices of {}",
        stamp.morph_fallbacks, stamp.morph_seam, stamp.vertices
    );
    assert_eq!(
        stamp.chunks_pending, 0,
        "{name}: the stamp was taken settled"
    );
    // THE GROUND REACHES THE HORIZON IN EVERY COLUMN: the topmost drawn pixel of each column lies
    // at or above the horizon's predicted row for that column — the row where the pixel's own ray
    // dips below level by the horizon's dip, through the pilot camera the renderer used. (The
    // topmost DRAWN pixel: a crack does not read as the sky, and the ruler ball may cover the limb
    // in its columns.) The refuter's finding: a check on one column, and a hole counter blind to
    // the skyline, would miss a missing cap at the limb.
    let camera = vd_bins::pixel::pilot_camera(&state, w, h);
    let label = format!("{planet:?}");
    let row = state
        .realm_boxes
        .iter()
        .find(|b| b.realm == label)
        .expect("the planet's row is drawn");
    let centre = DVec3::from_array(row.center);
    let radial = (camera.eye - centre).normalize();
    // The horizon of the lowest ground the recipe can raise, from this eye.
    let lowest_m = stamp.surface_m - body.relief_bound_m(0);
    let over_lowest = (camera.eye - centre).length() - lowest_m;
    let dip = (lowest_m / (lowest_m + over_lowest))
        .clamp(-1.0, 1.0)
        .acos();
    let (right, cam_up, back) = camera.basis();
    let forward = -back;
    let tan_half = (camera.fov_y * 0.5).tan();
    let aspect = w as f64 / h as f64;
    // The ray through a pixel's centre, and how far below level it points.
    let below_level = |x: usize, y: usize| -> f64 {
        let ndc_x = (x as f64 + 0.5) / w as f64 * 2.0 - 1.0;
        let ndc_y = 1.0 - (y as f64 + 0.5) / h as f64 * 2.0;
        let dir = (forward + right * (ndc_x * tan_half * aspect) + cam_up * (ndc_y * tan_half))
            .normalize();
        (-dir.dot(radial)).clamp(-1.0, 1.0).asin()
    };
    let mut worst: Option<(usize, usize, usize)> = None;
    let mut x = 0;
    while x < w {
        // The horizon's row in this column: the first row whose ray dips by the horizon's dip.
        let mut horizon_row = h;
        let mut y = 0;
        while y < h {
            if below_level(x, y) >= dip {
                horizon_row = y;
                break;
            }
            y += 1;
        }
        // The topmost drawn pixel in this column.
        let mut sky_row = h;
        let mut y = 0;
        while y < h {
            let i = (y * w + x) * 4;
            if decode_probe([probe[i], probe[i + 1], probe[i + 2]]).kind != PROBE_KIND_NONE {
                sky_row = y;
                break;
            }
            y += 1;
        }
        let short = sky_row.saturating_sub(horizon_row);
        if worst.is_none_or(|(_, _, s)| short > s) {
            worst = Some((x, horizon_row, short));
        }
        x += 1;
    }
    let (worst_x, worst_row, worst_short) = worst.expect("a column");
    eprintln!(
        "terrain_pictures/{name}: the skyline's worst column {worst_x} falls {worst_short} rows \
         short of the lowest ground's horizon row {worst_row} (its dip {:.2}°, the eye's surface's \
         {:.2}°, nose {tilt_deg}° down)",
        dip.to_degrees(),
        stamp.horizon_dip_deg
    );
    assert!(
        worst_short <= HORIZON_ROW_TOLERANCE,
        "{name}: in column {worst_x} the ground ends {worst_short} rows under the horizon's row \
         {worst_row}"
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
        (off_nose_band.0..=off_nose_band.1).contains(&star.off_nose_deg),
        "{name}: the star stands {:.2}° off the nose, outside {off_nose_band:?}",
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
        ruler.cell_m
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
        (ruler.rung, ruler.rung),
        "{name}: the ruler's pixels state the ball's rung"
    );
    assert_eq!(
        ruler.rung,
        rung_for_distance(ruler.distance_m, rungs),
        "{name}: the ball stands on the rule's rung for its distance"
    );
    // The G/B channel, measured: the ball's nearest pixel is `d − r` off, its rim `√(d² − r²)`
    // (within a fiftieth of a cell of `d` at this angular size), each within one cell of the
    // ball's own rung.
    let near_cells = (ruler.distance_m - ruler.radius_m) / ruler.cell_m;
    let rim_cells = (ruler.distance_m * ruler.distance_m - ruler.radius_m * ruler.radius_m).sqrt()
        / ruler.cell_m;
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
    // THE PICTURE AGAINST THE EXACT ONE (ruling V18: a packing may move no content pixel by
    // more than `TOLERANCE_LEVELS`): every pixel compared, the count and the widest channel step
    // reported, a step above the tolerance a red gate; with `VD_PICTURE_IDENTICAL` set, one
    // differing pixel is. The reference is the frozen exact picture when it exists, else the
    // owner's picture from the previous run.
    let dest_probe = dir.join(format!("{name}.probe.png"));
    let dest_hud = dir.join(format!("{name}.hud.json"));
    // EVERY STAND HAS A FROZEN EXACT PICTURE (refutation N-10: a compare against the previous
    // run's picture would let two lossy steps ratchet one level each). A new stand freezes its
    // exact picture first — a missing reference is a red gate, never a skipped compare.
    let exact = dir.join(EXACT_DIR);
    let reference = exact.join(format!("{name}.png"));
    let reference_probe = exact.join(format!("{name}.probe.png"));
    let reference_hud = exact.join(format!("{name}.hud.json"));
    if freeze_wanted(name) {
        // A freeze records the GATE's moment: a look measurement's coarser grid may never become
        // the reference (refutation of the ladder, finding 5).
        assert!(
            std::env::var_os(PICTURE_REPORT_ONLY_ENV).is_none()
                && std::env::var_os(PICTURE_CENSUS_ONLY_ENV).is_none(),
            "{name}: a freeze runs on the gate's own grid — unset {PICTURE_REPORT_ONLY_ENV} and \
             {PICTURE_CENSUS_ONLY_ENV}"
        );
        std::fs::create_dir_all(&exact).expect("the exact directory");
        std::fs::copy(&png, &reference).expect("freeze the picture");
        std::fs::copy(&probe_png, &reference_probe).expect("freeze the probe");
        let rect = serde_json::to_string(&stamp.hud_rect_px).expect("the rectangle encodes");
        std::fs::write(&reference_hud, &rect).expect("freeze the overlay's rectangle");
        std::fs::copy(&png, &dest).expect("copy the picture");
        std::fs::copy(&probe_png, &dest_probe).expect("copy the probe");
        std::fs::write(&dest_hud, &rect).expect("write the overlay's rectangle beside the picture");
        eprintln!(
            "terrain_pictures/{name}: FROZEN as the exact reference under {} (the owner's \
             acceptance); nothing compared",
            exact.display()
        );
        return (last, share);
    }
    assert!(
        reference.exists() && reference_probe.exists(),
        "{name}: no frozen exact picture under {} — freeze one from an exact flight first",
        exact.display()
    );
    eprintln!("terrain_pictures/{name}: compared against the frozen exact picture");
    {
        let (before, bw, bh) = open_rgba(&reference);
        let (before_probe, pw2, ph2) = open_rgba(&reference_probe);
        assert_eq!(
            ((bw, bh), (pw2, ph2)),
            ((w, h), (w, h)),
            "{name}: the frozen exact picture is {bw}×{bh} and its probe {pw2}×{ph2}; this one \
             is {w}×{h}"
        );
        {
            // Only the CONTENT compares: a pixel either probe marks as terrain or ruler, and not
            // under the overlay. The HUD's stamp line carries the frame's tick, and that readout
            // differs between two runs of one code by design (MEASURED: 152 pixels of a
            // packed-against-packed ground picture, every one inside the HUD's text); the
            // renderer reports the HUD's rectangle on the stamp, and the compare leaves it out.
            // The UNION of both pictures' overlay rectangles (the reference's rides beside it
            // in a sidecar; refutation P-3: a HUD whose longest line changed with the code left
            // its extra text counted as content), grown by the glyphs' antialiasing.
            let this_hud = stamp.hud_rect_px;
            let before_hud: [f32; 4] = std::fs::read_to_string(&reference_hud)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(this_hud);
            let hud = [
                this_hud[0].min(before_hud[0]) - HUD_MARGIN_PX,
                this_hud[1].min(before_hud[1]) - HUD_MARGIN_PX,
                this_hud[2].max(before_hud[2]) + HUD_MARGIN_PX,
                this_hud[3].max(before_hud[3]) + HUD_MARGIN_PX,
            ];
            eprintln!(
                "terrain_pictures/{name}: the overlays' rectangle ({:.0}, {:.0})–({:.0}, {:.0}) \
                 is left out of the compare",
                hud[0], hud[1], hud[2], hud[3]
            );
            let mut differing = 0usize;
            let mut content = 0usize;
            let mut widest = 0u8;
            for (i, (((a, b), pa), pb)) in before
                .chunks_exact(4)
                .zip(rgba.chunks_exact(4))
                .zip(before_probe.chunks_exact(4))
                .zip(probe.chunks_exact(4))
                .enumerate()
            {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let under_hud = x >= hud[0] && x <= hud[2] && y >= hud[1] && y <= hud[3];
                let drawn = decode_probe([pa[0], pa[1], pa[2]]).kind != PROBE_KIND_NONE
                    || decode_probe([pb[0], pb[1], pb[2]]).kind != PROBE_KIND_NONE;
                if !drawn || under_hud {
                    continue;
                }
                content += 1;
                let step = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.abs_diff(*y))
                    .max()
                    .unwrap_or(0);
                differing += usize::from(step > 0);
                widest = widest.max(step);
            }
            eprintln!(
                "terrain_pictures/{name}: against the picture on disk — {differing} of {content} \
                 content pixels differ (the probe's terrain and ruler), the widest channel step \
                 {widest}"
            );
            // Where they differ: the previous picture kept beside the run, and a difference
            // image — a differing pixel white on black — so the owner sees WHAT changed.
            if differing > 0 {
                // Kept OUTSIDE the fixture (a green run deletes its fixture): under the
                // workspace's target directory.
                let kept_dir =
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/terrain_pictures");
                std::fs::create_dir_all(&kept_dir).expect("the kept pictures' directory");
                let keep = kept_dir.join(format!("{name}.before.png"));
                std::fs::copy(&reference, &keep).expect("keep the reference picture");
                std::fs::copy(&png, kept_dir.join(format!("{name}.after.png")))
                    .expect("keep the new picture");
                let mut diff = vec![0u8; before.len()];
                for (i, ((((a, b), pa), pb), d)) in before
                    .chunks_exact(4)
                    .zip(rgba.chunks_exact(4))
                    .zip(before_probe.chunks_exact(4))
                    .zip(probe.chunks_exact(4))
                    .zip(diff.chunks_exact_mut(4))
                    .enumerate()
                {
                    let (x, y) = ((i % w) as f32, (i / w) as f32);
                    let under_hud = x >= hud[0] && x <= hud[2] && y >= hud[1] && y <= hud[3];
                    let drawn = !under_hud
                        && (decode_probe([pa[0], pa[1], pa[2]]).kind != PROBE_KIND_NONE
                            || decode_probe([pb[0], pb[1], pb[2]]).kind != PROBE_KIND_NONE);
                    let step = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| x.abs_diff(*y))
                        .max()
                        .unwrap_or(0);
                    let v = if drawn && step > 0 { 255 } else { 0 };
                    d.copy_from_slice(&[v, v, v, 255]);
                }
                let diff_path = kept_dir.join(format!("{name}.diff.png"));
                image::save_buffer(
                    &diff_path,
                    &diff,
                    w as u32,
                    h as u32,
                    image::ColorType::Rgba8,
                )
                .expect("write the difference image");
                eprintln!(
                    "terrain_pictures/{name}: the previous picture is kept at {} and the \
                     difference image at {}",
                    keep.display(),
                    diff_path.display()
                );
            }
            if std::env::var_os(PICTURE_REPORT_ONLY_ENV).is_some() {
                eprintln!(
                    "terrain_pictures/{name}: LOOK MEASUREMENT — the tolerance's verdict is \
                     withheld ({differing} pixels differ, the widest step {widest} against the \
                     allowed {TOLERANCE_LEVELS}); the pictures are kept, the owner's untouched"
                );
            } else if widest > TOLERANCE_LEVELS {
                // THE VERDICT IS HELD until every stand has flown, so one flight reports every
                // stand (the packing's measurement reads all four); the test fails at its end.
                let verdict = format!(
                    "{name}: THE PICTURE CHANGED PAST THE TOLERANCE — {differing} pixels differ, \
                     the widest channel step {widest} against the allowed {TOLERANCE_LEVELS} \
                     (ruling V18)"
                );
                eprintln!("terrain_pictures/{verdict}");
                PAST_TOLERANCE.lock().expect("the verdicts").push(verdict);
            }
            if std::env::var_os(PICTURE_IDENTICAL_ENV).is_some() {
                assert_eq!(
                    differing, 0,
                    "{name}: THE PICTURE CHANGED — {differing} pixels differ (widest step \
                     {widest}) and {PICTURE_IDENTICAL_ENV} refuses any change"
                );
            }
        }
    }
    // The previous picture and probe stay beside the run for a later look, the new ones go to
    // the owner's directory — unless this is a look measurement, whose pictures stay beside the
    // run only.
    if std::env::var_os(PICTURE_REPORT_ONLY_ENV).is_some() {
        let kept_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/terrain_pictures");
        std::fs::create_dir_all(&kept_dir).expect("the kept pictures' directory");
        std::fs::copy(&png, kept_dir.join(format!("{name}.look.png"))).expect("keep the look");
        std::fs::copy(&probe_png, kept_dir.join(format!("{name}.look.probe.png")))
            .expect("keep the look's probe");
        eprintln!(
            "terrain_pictures/{name}: the look is kept under {}",
            kept_dir.display()
        );
        return (last, share);
    }
    std::fs::copy(&png, &dest).expect("copy the picture");
    std::fs::write(
        &dest_hud,
        serde_json::to_string(&stamp.hud_rect_px).expect("the rectangle encodes"),
    )
    .expect("write the overlay's rectangle beside the picture");
    std::fs::copy(&probe_png, &dest_probe).expect("copy the probe");
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
    let dir = [d.x, d.y, d.z];
    let h = vd_terrain::height::height_m(&body, dir, 0);
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
    let orbit = stand(d, ORBIT_M, h, nose(ORBIT_TILT_DEG));
    let height_at = |p: DVec3| vd_terrain::height::height_m(&body, [p.x, p.y, p.z], 0);
    // THE SEAM: the point on the cube's twelve edges (every face's four), looking along the edge
    // one way or the other, where the star stands INSIDE the gate's elevation band and nearest
    // `SUN_OFF_NOSE_DEG` off the nose (the light over the shoulder, as every stand has it); where
    // no point of any edge meets the band, the nearest by the sum of both misses. MEASURED on the
    // spot's face alone: the elevation's nearest point put the star straight behind (177.75° off
    // the nose), and the sum's nearest put it 37.65° up — the spot's own face has no edge point
    // with the star low AND over the shoulder.
    let seam_param = |i: i32| -1.0 + 2.0 * f64::from(i) / f64::from(SEAM_STEPS);
    let seam_at = |face: vd_seed::bend::Face, edge: i32, i: i32| {
        let (a, b) = match edge {
            0 => (1.0, seam_param(i)),
            1 => (-1.0, seam_param(i)),
            2 => (seam_param(i), 1.0),
            _ => (seam_param(i), -1.0),
        };
        DVec3::from_array(vd_seed::bend::direction(face, a, b))
    };
    // The nose along the edge at step `i`, one way (`sign` ±1), and the star's angles there.
    let seam_nose = |face: vd_seed::bend::Face, edge: i32, i: i32, sign: f64| {
        let p = seam_at(face, edge, i);
        let tangent = (seam_at(face, edge, (i + 1).min(SEAM_STEPS))
            - seam_at(face, edge, (i - 1).max(0)))
        .normalize()
            * sign;
        let level_sun = (sun - p * sun.dot(p)).normalize();
        let elevation = sun.dot(p).clamp(-1.0, 1.0).asin().to_degrees();
        let off_nose = tangent.dot(level_sun).clamp(-1.0, 1.0).acos().to_degrees();
        (p, tangent, elevation, off_nose)
    };
    let in_band =
        |elevation: f64| (SUN_ELEVATION_BAND_DEG.0..=SUN_ELEVATION_BAND_DEG.1).contains(&elevation);
    let mut seam = (f64::MAX, vd_seed::bend::Face::ALL[0], 0_i32, 0_i32, 1.0_f64);
    for face in vd_seed::bend::Face::ALL {
        let mut edge = 0;
        while edge < 4 {
            let mut i = 0;
            while i <= SEAM_STEPS {
                for sign in [1.0, -1.0] {
                    let (_, _, elevation, off_nose) = seam_nose(face, edge, i, sign);
                    let off_miss = (off_nose - SUN_OFF_NOSE_DEG).abs();
                    // Inside the band the off-nose miss alone; outside it, a full turn's worth
                    // more plus the elevation's miss, so any in-band point wins.
                    let miss = if in_band(elevation) {
                        off_miss
                    } else {
                        360.0 + (elevation - SUN_ELEVATION_DEG).abs() + off_miss
                    };
                    if miss < seam.0 {
                        seam = (miss, face, edge, i, sign);
                    }
                }
                i += 1;
            }
            edge += 1;
        }
    }
    let (seam_dir, seam_ahead, seam_elevation, seam_off_nose) =
        seam_nose(seam.1, seam.2, seam.3, seam.4);
    let seam_h = height_at(seam_dir);
    let seam_t = GROUND_TILT_DEG.to_radians();
    let seam_forward = (seam_ahead * seam_t.cos() - seam_dir * seam_t.sin()).normalize();
    let seam_stand = stand(seam_dir, EYE_HEIGHT_M, seam_h, seam_forward);
    eprintln!(
        "terrain_pictures: THE SEAM on edge {} of {:?} at step {} looking {} (the star \
         {seam_elevation:.2}° up, {seam_off_nose:.2}° off the nose)",
        seam.2,
        seam.1,
        seam.3,
        if seam.4 > 0.0 { "forward" } else { "back" }
    );
    let spawn_poses = [
        spawn_entry(CLIENT_ACCOUNT_BASE, body.seed(), &ground),
        spawn_entry(CLIENT_ACCOUNT_BASE + 1, body.seed(), &hill),
        spawn_entry(CLIENT_ACCOUNT_BASE + 2, body.seed(), &aloft),
        spawn_entry(CLIENT_ACCOUNT_BASE + 3, body.seed(), &orbit),
        spawn_entry(CLIENT_ACCOUNT_BASE + 4, body.seed(), &seam_stand),
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
            tilt_deg: GROUND_TILT_DEG,
            band: (0.55, 1.0),
            min_share: 0.95,
            off_nose_band: SUN_OFF_NOSE_BAND_DEG,
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
            tilt_deg: HILL_TILT_DEG,
            band: (0.5, 1.0),
            min_share: 0.95,
            off_nose_band: SUN_OFF_NOSE_BAND_DEG,
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
            tilt_deg: ALOFT_TILT_DEG,
            band: (0.5, 1.0),
            min_share: 0.95,
            off_nose_band: SUN_OFF_NOSE_BAND_DEG,
        },
    );
    let (orbit_chunks, orbit_share) = take_picture(
        &f,
        a.gateway,
        &body,
        planet,
        &Picture {
            name: "orbit",
            agent_index: 3,
            tilt_deg: ORBIT_TILT_DEG,
            // The limb arcs across the frame and dips to two thirds of the height at the sides
            // (MEASURED on the first orbit picture: 0.932 of the lower half was ground, the rest
            // the sky in the corners), so the band starts under it.
            band: (0.72, 1.0),
            min_share: 0.95,
            off_nose_band: SUN_OFF_NOSE_BAND_DEG,
        },
    );
    let (seam_chunks, seam_share) = take_picture(
        &f,
        a.gateway,
        &body,
        planet,
        &Picture {
            name: "seam",
            agent_index: 4,
            tilt_deg: GROUND_TILT_DEG,
            band: (0.55, 1.0),
            min_share: 0.95,
            off_nose_band: (SUN_OFF_NOSE_BAND_DEG.0, 180.0),
        },
    );
    eprintln!(
        "terrain_pictures: ground {ground_chunks} chunks ({ground_share:.3}), hill {hill_chunks} \
         chunks ({hill_share:.3}), aloft {aloft_chunks} chunks ({aloft_share:.3}), orbit \
         {orbit_chunks} chunks ({orbit_share:.3}), seam {seam_chunks} chunks ({seam_share:.3})"
    );
    let past = PAST_TOLERANCE.lock().expect("the verdicts");
    assert!(
        past.is_empty(),
        "THE PICTURE CHANGED PAST THE TOLERANCE on {} stand(s) (ruling V18):\n{}",
        past.len(),
        past.join("\n")
    );
    drop(past);
    assert!(
        ground_chunks >= GROUND_CHUNKS_MIN,
        "the ground picture holds the ladder"
    );
    assert!(
        hill_chunks >= HILL_CHUNKS_MIN,
        "the hill picture holds the ladder"
    );
    assert!(
        aloft_chunks >= ALOFT_CHUNKS_MIN,
        "the aloft picture holds the ladder"
    );
    assert!(
        orbit_chunks >= ORBIT_CHUNKS_MIN,
        "the orbit picture holds the globe"
    );
    assert!(
        seam_chunks >= GROUND_CHUNKS_MIN,
        "the seam picture holds the ladder"
    );
}
