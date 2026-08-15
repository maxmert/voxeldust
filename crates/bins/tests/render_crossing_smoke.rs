//! G-RENDER-CROSSING-SMOKE — the crossing pixel proof on THE WORLD ITSELF: a real dot flies the ±Z
//! polar corridor OUT of THE world's own 150 m home shell, the directory CAS re-homes its authority
//! onto the pre-booked galaxy shard, and the round trip is captured in actual pixels — INSIDE the
//! home system's drawn shell, OUTSIDE it in the between-space, and back INSIDE on the return. The
//! RETURN leg gets pixel coverage for the first time (the owner's "everything froze on the way
//! back").
//!
//! It stands up the DUAL process cluster (`up --dual`) with NO injected geometry — the old gate
//! planted an authored two-box playground through a boundary-override hook; THE world's own home
//! shell IS the crossing boundary now (SL5: one world, nothing to select). A headless
//! `client --capture --realm-boxes` draws the scene `emit-world-scene` writes (`regions.json`, the
//! home shard's own boot neighbourhood — the drawn geometry IS the detector's, single-sourced).
//!
//! THE CAMERA (judge fix A1): the client's offscreen capture camera is `fit_camera_to_scene` over
//! the LIVE overlaid scene, refit EVERY frame — and THE world's planets ORBIT, so a camera this
//! gate fitted over the static boot file would drift from the client's the moment an outer planet
//! swings wide. Each capture therefore RECONSTRUCTS the client's actual camera from the client's
//! own reported drawn boxes (`DevState.realm_boxes` centres, zipped with THE world's extents by
//! realm label → `bounds_union` → `fit_camera_to_bounds`) — the identical per-frame refit, fed the
//! identical inputs. Residual skew is sub-frame (one tick of orbital motion against a ~260 m scene
//! radius, ≤ ~0.14 m).
//!
//! J1, ASSERTED IN-TEST: the home system sits at the GALACTIC ORIGIN (`world_roster` asserts the
//! galaxy authors its placement at ZERO), so a home↔galaxy crossing is numerically an IDENTITY in
//! the drawn space — which is exactly why this ONE file-based scene stays valid across both
//! crossings, and why `expected_box == None` at the middle capture is honest: the galaxy's ~12483 m
//! extent exceeds the renderable ceiling and SL3 says a containment boundary is never drawn.
//!
//! GPU PRECONDITION (same as the other visual gates): renders through wgpu, REQUIRES a working GPU
//! adapter, LOCAL-only (no CI, no software-raster fallback). Run via `just render-crossing-smoke`.
//! Under a plain `cargo test --workspace` (no features) this file compiles to ZERO tests. Tier-B
//! process glue (coverage-exempt); the Tier-A verdicts it stands on are 100%-covered by their own
//! unit tests.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::flight::cross_leg;
use vd_bins::scene_camera::{extent_by_label, live_scene_camera};
use vd_bins::{
    ClusterAddrs, ClusterShape, DEV, DevClusterDown, admin_get_body, dev_auth_signing_key_hex,
    dev_roundtrip, devcluster, loopback, realm_shards, record_extra_pid, slot_trust_dir,
    slot_workdir, world_roster, write_world_regions,
};
use vd_client::realm_scene::{BoxShape, RealmScene};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::camera::marker_world_radius;
use vd_client_harness::camera::{CaptureCamera, ScreenAabb};
use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::{
    dot_pixels_distinct_from_surround, dot_pixels_within_box_region, expected_box,
    projected_point_aabb,
};
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::glam::DVec3;
use vd_core::pose::{RealmId, frame_for_realm};
use vd_devproto::{DevPortScheme, DevRequest, DevResponse, WORKTREE_SLOT_CEILING};

const CLIENT_NAME: &str = "g-render-crossing";
/// A DISTINCT test-reserved slot, above `WORKTREE_SLOT_CEILING` and clear of the other visual/process
/// gates (render-smoke +18, render-boxes/crossing-e2e +19) so a live cluster can never collide.
const RENDER_CROSSING_SLOT: u16 = WORKTREE_SLOT_CEILING + 22; // 86: G-RENDER-CROSSING-SMOKE

/// The dot rectangle's BRACKET over the marker's floored world radius
/// (`marker_world_radius(DOT_RADIUS, …)` — the SAME derivation the renderer scales the marker by,
/// so the asserted rectangle and the drawn footprint cannot drift): 2× covers the reconstruction
/// skew, the sphere-silhouette bulge and the rect pixelization.
const DOT_RECT_BRACKET: f64 = 2.0;
/// The surround-ring width (px) for the dot-presence verdict: the local background sampled just
/// outside the dot's rectangle (`dot_pixels_distinct_from_surround`).
const DOT_SURROUND_RING_PX: f64 = 4.0;
/// Extra clearance (px) beyond the dot's rectangle when choosing an in-home capture park clear of
/// every planet's whole projected orbit annulus (phase-independent — see `clean_home_park`).
const PARK_RECT_CLEARANCE_PX: f64 = 2.0;
/// The park plane's world |z| (m): planets hug the XY plane (inclination σ 0.02 rad) and their
/// containment reach is soi + outset ≈ 6 m, so 20 m of z clears every planet's SOI in WORLD space —
/// the park can never fire a planet crossing, whatever its (x, y).
const PARK_PLANE_Z_M: f64 = 20.0;
/// The park search range/step along +Y (the most screen-transverse world axis under the fitted
/// view), bounded well inside the home shell's ~149 m acquire edge.
const PARK_MAX_OFFSET_M: f64 = 135.0;
const PARK_SEARCH_STEP_M: f64 = 1.0;
/// A sphere's SILHOUETTE slightly exceeds its projected chord under perspective (≤ 2 % at this
/// scene's depth ratios) — the factor the annulus bounds inflate an SOI disc by.
const SILHOUETTE_BULGE: f64 = 1.02;
/// The OUTSIDE park (flight law leg A, continued down the corridor): stated in the space the
/// session stands in — home-frame on the way out, and numerically the SAME point in the galaxy
/// frame (J1). The crossing itself fires at the ~152 m release edge; the park is 2 km down the pole
/// because of the CAMERA's geometry (measured): the fitted view direction is mostly −Z — nearly
/// parallel to the corridor — so a dot parked just past the shell projects INSIDE the shell's
/// silhouette by foreshortening (at 220 m: a 91 px offset against a 167 px rect). Past ~1.2 km the
/// projection clears the rect; 2 km gives ~50 px of margin per axis. Still deep inside the galaxy's
/// own ~12483 m shell, still on the polar corridor (the sibling stars sit on the ±X ring).
const OUTSIDE_PARK: DVec3 = DVec3::new(0.0, 0.0, -2000.0);
/// The RETURN leg's AIM (flight law leg B): inside the ~149 m acquire edge so the label flips. The
/// capture itself then re-parks at the annulus-clear point (`clean_home_park`) — the aim only has to
/// commit the crossing, never to host pixels.
const RETURN_PARK: DVec3 = DVec3::new(0.0, 0.0, -60.0);
/// A generous LOCAL deadline per crossing leg: the ~2 km corridor flight + the re-home propagation.
const LEG_DEADLINE: Duration = Duration::from_secs(60);
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const READY_POLL: Duration = Duration::from_millis(200);
const DELIVERY_DEADLINE: Duration = Duration::from_secs(90);
const DELIVERY_POLL: Duration = Duration::from_millis(100);

/// Kill the capture client on drop — the 5th process beyond the dual cluster's 4 nodes.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| {
        panic!("dev-control round-trip on {port} failed: {e} — capture client gone? (GPU precondition?)")
    })
}

fn await_listener(port: u16, child: &mut Child) {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if TcpStream::connect(loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the --capture client exited early ({status}) before its dev-control listener came \
                 up — G-RENDER-CROSSING-SMOKE requires a working GPU adapter (see the module docs)"
            );
        }
        assert!(
            Instant::now() < deadline,
            "dev-control listener on {port} never came up within {READY_TIMEOUT:?} (GPU precondition?)"
        );
        std::thread::sleep(READY_POLL);
    }
}

/// The orchestrator admin directory as `(key, authority)` rows (diagnosis on a failed non-vacuity
/// gate: distinguishes "never fired / source still owns" from "committed but unobserved"). Empty on
/// a not-yet-answering endpoint.
fn directory_rows(admin: std::net::SocketAddr) -> Vec<(String, String)> {
    let Some(body) = admin_get_body(admin, "/admin/snapshot", Some(Duration::from_secs(2))) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    value["directory"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    Some((
                        e["key"].as_str()?.to_owned(),
                        e["authority"].as_str()?.to_owned(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The current delivered state (a `State` round-trip).
fn poll_state(port: u16) -> vd_devproto::DevState {
    match round_trip(port, &DevRequest::State) {
        DevResponse::State { state } => state,
        other => panic!("expected DevState, got {other:?}"),
    }
}

/// The OWN entity's delivered row (id + composited world pose), or panic naming the state.
fn own_pos(state: &vd_devproto::DevState) -> DVec3 {
    let own = state
        .own_entity
        .as_deref()
        .unwrap_or_else(|| panic!("no own_entity yet in {state:?}"));
    let row = state
        .entities
        .iter()
        .find(|r| r.entity == own)
        .unwrap_or_else(|| panic!("own entity {own} not in the delivered rows: {state:?}"));
    DVec3::new(row.pos[0], row.pos[1], row.pos[2])
}

/// The client-reported DRAWN centre of `realm`'s box (`DevState.realm_boxes`, the same chokepoint
/// the pixels go through) — panics if the client is not drawing it.
fn drawn_centre(state: &vd_devproto::DevState, realm: RealmId) -> DVec3 {
    let label = format!("{realm:?}");
    let row = state
        .realm_boxes
        .iter()
        .find(|b| b.realm == label)
        .unwrap_or_else(|| {
            panic!(
                "the client is not drawing {label} — drawn scene: {:?}",
                state.realm_boxes
            )
        });
    DVec3::from_array(row.center)
}

/// A realm box's projected screen rectangle through the RECONSTRUCTED live camera: the client's
/// reported drawn centre + the box's bounding-sphere radius from the file scene.
fn drawn_box_screen_aabb(
    state: &vd_devproto::DevState,
    scene: &RealmScene,
    camera: &CaptureCamera,
    realm: RealmId,
) -> ScreenAabb {
    let radius = match scene.get(realm).expect("box in scene").shape {
        BoxShape::Box { half } => half.length(),
        BoxShape::Sphere { r } => r,
    };
    projected_point_aabb(camera, drawn_centre(state, realm), radius)
        .expect("the box center projects in front of the camera")
}

/// The dot's projected rectangle at `pos`: [`DOT_RECT_BRACKET`] × the marker's floored world radius
/// (the SAME `marker_world_radius` derivation the renderer scales the marker by — batch review: the
/// pixel verdicts are dot-sensitive only because the marker has a floor AND the rectangle brackets
/// exactly that floor).
fn dot_screen_aabb(camera: &CaptureCamera, pos: DVec3) -> ScreenAabb {
    let dist_m = (pos - camera.eye).length();
    let marker_r = marker_world_radius(
        f64::from(vd_client_render::DOT_RADIUS),
        dist_m,
        camera.fov_y,
        camera.height as f64,
    );
    projected_point_aabb(camera, pos, DOT_RECT_BRACKET * marker_r)
        .expect("the dot projects in front of the camera")
}

/// Two screen rectangles share no pixel.
fn rects_disjoint(a: ScreenAabb, b: ScreenAabb) -> bool {
    a.max.x < b.min.x || b.max.x < a.min.x || a.max.y < b.min.y || b.max.y < a.min.y
}

/// Screen px per world metre, transverse, at the fitted camera's target depth — measured off the
/// camera itself (project the target and a point 1 m right of it), never re-derived arithmetic.
fn px_per_metre(camera: &CaptureCamera) -> f64 {
    let forward = (camera.target - camera.eye).normalize();
    let right = forward.cross(camera.up).normalize();
    let a = camera
        .project_point(camera.target)
        .expect("the target projects");
    let b = camera
        .project_point(camera.target + right)
        .expect("1 m off the target projects");
    ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt()
}

/// One planet's whole projected ORBIT ANNULUS `[min, max]`, in px of radial distance from the drawn
/// home centre: every screen position the planet's drawn disc can EVER occupy, over all orbital
/// phases. The min folds the worst XY-plane foreshortening (`|forward·ẑ|` — orbits hug the XY
/// plane); both edges inflate the SOI disc by [`SILHOUETTE_BULGE`]. Phase-independent, which is what
/// makes a park chosen outside every annulus immune to capture timing.
fn planet_annulus_px(
    camera: &CaptureCamera,
    elements: &vd_physics::celestial::OrbitalElements,
    soi_m: f64,
) -> (f64, f64) {
    let pxm = px_per_metre(camera);
    let fore = ((camera.target - camera.eye).normalize().z).abs();
    let soi_px = soi_m * SILHOUETTE_BULGE;
    let min = (elements.sma * (1.0 - elements.ecc) * fore - soi_px) * pxm;
    let max = (elements.sma * (1.0 + elements.ecc) + soi_px) * pxm;
    (min.max(0.0), max)
}

/// A capture park INSIDE the home shell whose dot RECTANGLE provably clears every planet's whole
/// projected orbit annulus — so no planet pixel can ever sit inside the dot's rectangle, at any
/// orbital phase, however long the flight to the park takes. The surround ring may still graze a
/// planet (it only ADDS colors to the background set; the pure-shell color always remains on the
/// ring's far side), so only the rectangle needs the clearance. Searched along +Y (the most
/// screen-transverse world axis under the fitted view) on the |z| = [`PARK_PLANE_Z_M`] plane (clear
/// of every planet's SOI in world space). Panics — loudly, naming the annuli — if THE world's
/// orbit layout ever tiles the whole disc.
fn clean_home_park(camera: &CaptureCamera, annuli: &[(f64, f64)]) -> DVec3 {
    let marker_px = vd_client_harness::camera::DOT_MIN_APPARENT_RADIUS_PX;
    let clearance = DOT_RECT_BRACKET * marker_px + PARK_RECT_CLEARANCE_PX;
    let centre = camera
        .project_point(DVec3::ZERO)
        .expect("the home centre projects");
    let mut offset_m = 0.0_f64;
    while offset_m <= PARK_MAX_OFFSET_M {
        let world = DVec3::new(0.0, offset_m, -PARK_PLANE_Z_M);
        let p = camera
            .project_point(world)
            .expect("an in-shell park projects");
        let r_px = ((p.x - centre.x).powi(2) + (p.y - centre.y).powi(2)).sqrt();
        if annuli
            .iter()
            .all(|(lo, hi)| r_px + clearance < *lo || r_px - clearance > *hi)
        {
            return world;
        }
        offset_m += PARK_SEARCH_STEP_M;
    }
    panic!(
        "no capture park inside the home shell clears every planet's projected orbit annulus \
         (annuli {annuli:?} px, clearance {clearance:.1} px) — THE world's orbit layout changed; \
         restate the park search, never the pixel asserts"
    )
}

/// Capture-state belt over the phase-independent park: every OTHER drawn box's projected rect at
/// THIS capture is disjoint from the dot's rectangle — so a pixel in the dot rect unlike its
/// surround can only be the dot.
fn assert_dot_rect_clear_of_planets(cap: &Capture, scene: &RealmScene, home: RealmId, label: &str) {
    for (realm, _) in scene.iter() {
        if realm == home {
            continue;
        }
        let rect = drawn_box_screen_aabb(&cap.state, scene, &cap.camera, realm);
        assert!(
            rects_disjoint(cap.dot_rect, rect),
            "{label}: the dot's rectangle {:?} must be clear of {realm:?}'s drawn disc {rect:?} — \
             the annulus-clear park guarantees this at every orbital phase, so a collision means \
             the park search and the world disagree",
            cap.dot_rect,
        );
    }
}

/// Park the dot at `target` (stated in the space the session currently stands in), throttle cut.
fn park_at(devctl: u16, target: DVec3) {
    let walk = round_trip(
        devctl,
        &DevRequest::WalkTo {
            target: target.to_array(),
            arrive_epsilon: 3.0,
            max_ticks: 3000,
            max_step_m: 4.0 * DEV.move_speed * DEV.tick_dt,
        },
    );
    assert!(
        matches!(walk, DevResponse::State { .. }),
        "the park WalkTo at {target} should arrive, got {walk:?}",
    );
    let _ = round_trip(
        devctl,
        &DevRequest::Move {
            axes: [0.0, 0.0, 0.0],
        },
    );
}

/// Decode a captured PNG → (rgba, w, h, self-calibrated clear color from the top-right corner).
fn decode_capture(cwd: &Path, rel_path: &str) -> (Vec<u8>, usize, usize, [u8; 4]) {
    let png = cwd.join(rel_path);
    let img = image::open(&png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let buf = img.into_raw();
    let corner = (w - 1) * 4;
    let clear = [
        buf[corner],
        buf[corner + 1],
        buf[corner + 2],
        buf[corner + 3],
    ];
    (buf, w, h, clear)
}

/// Screenshot at the current tick, returning the run-relative path (fails loud naming the GPU precondition).
fn screenshot(port: u16, label: &str) -> String {
    match round_trip(
        port,
        &DevRequest::Screenshot {
            at_tick: None,
            label: Some(label.to_owned()),
        },
    ) {
        DevResponse::Captured { path, .. } => path,
        other => {
            panic!("screenshot '{label}' was not captured (GPU precondition unmet?): {other:?}")
        }
    }
}

/// One capture: sample the state, screenshot, reconstruct the client's live camera from THAT state,
/// and return everything a verdict needs. The J1 drawn identity is asserted at EVERY capture: the
/// home box draws at the session origin on both sides of the crossing (the galaxy authors the home
/// placement at ZERO), or the one shared file scene would be lying about one of the frames.
struct Capture {
    state: vd_devproto::DevState,
    camera: CaptureCamera,
    home_rect: ScreenAabb,
    pos: DVec3,
    dot_rect: ScreenAabb,
    rgba: Vec<u8>,
    w: usize,
    h: usize,
    clear: [u8; 4],
    /// The run-relative screenshot path (the HR6 manifest check at the end).
    shot: String,
}

#[allow(clippy::too_many_arguments)]
fn capture(
    devctl: u16,
    cwd: &Path,
    label: &str,
    scene: &RealmScene,
    extents: &std::collections::BTreeMap<String, DVec3>,
    home: RealmId,
) -> Capture {
    let state = poll_state(devctl);
    let shot = screenshot(devctl, label);
    let camera = live_scene_camera(&state, extents, CAPTURE_W as usize, CAPTURE_H as usize);
    assert_eq!(
        drawn_centre(&state, home),
        DVec3::ZERO,
        "J1 (drawn identity, capture '{label}'): the home box draws at the session origin on BOTH \
         sides of the home↔galaxy crossing — the galaxy authors the home placement at ZERO, which \
         is why the one file scene stays valid across it",
    );
    let home_rect = drawn_box_screen_aabb(&state, scene, &camera, home);
    let pos = own_pos(&state);
    let dot_rect = dot_screen_aabb(&camera, pos);
    let (rgba, w, h, clear) = decode_capture(cwd, &shot);
    assert_eq!(
        magenta_pixel_count(&rgba),
        0,
        "the '{label}' frame must be magenta-free"
    );
    Capture {
        state,
        camera,
        home_rect,
        pos,
        dot_rect,
        rgba,
        w,
        h,
        clear,
        shot,
    }
}

#[test]
fn g_render_crossing_smoke_dot_pixels_leave_the_home_shell_and_return() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    let _ = devcluster(launcher, "down", RENDER_CROSSING_SLOT); // clean slate (idempotent)
    let _down = DevClusterDown::new(launcher, RENDER_CROSSING_SLOT);

    // THE ROSTER + THE LABELS. `world_roster` itself asserts I-AXIS/I-POLE/I-RADIAL and J1 (the
    // home placement is ZERO in the galaxy frame; the sibling's is not) — deriving it here IS the
    // in-test flight-law + J1 gate, run before any process spawns.
    let roster = world_roster(&DEV);
    let home_label = frame_for_realm(roster.home, None)
        .expect("the home realm has a frame")
        .label();
    let galaxy_label = frame_for_realm(roster.galaxy, None)
        .expect("the galaxy realm has a frame")
        .label();

    let ports = DevPortScheme::DEFAULT
        .slot_ports(RENDER_CROSSING_SLOT)
        .expect("slot ports");
    let gateway = loopback(ports.gateway);
    let devctl = ports.dev_control(0).expect("dev-control port");
    let client_quic = ports.client_quic(0).expect("client-quic port");
    let trust_dir = slot_trust_dir(RENDER_CROSSING_SLOT);
    let signing_key = dev_auth_signing_key_hex();
    let cwd = slot_workdir(RENDER_CROSSING_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    // ONE scene, from THE world (`emit-world-scene`'s body): the home shard's own boot
    // neighbourhood — the drawn geometry IS the detector's. The same file feeds the client
    // (`--realm-boxes`), the in-test membership verdict (`expected_box`), and the extent map the
    // camera reconstruction joins the drawn boxes against.
    let regions_path = write_world_regions(&cwd, &DEV).expect("emit THE world scene");
    let regions_json = std::fs::read_to_string(&regions_path).expect("read regions.json");
    let regions: Vec<vd_core::geometry::RealmRegion> =
        serde_json::from_str(&regions_json).expect("regions.json parses");
    let extents = extent_by_label(&regions);
    let scene = RealmScene::from_regions_json(&regions_json).expect("the world scene projects");
    // J1, on the file scene: the home box is AT the origin of the emitted scene (the shard's own
    // frame — SL1), so the drawn-identity assert below compares against a genuine zero.
    assert_eq!(
        scene
            .get(roster.home)
            .expect("the home box is renderable")
            .draw_center(),
        DVec3::ZERO,
        "J1: the emitted scene draws the home system at its own origin",
    );

    // Bring up the DUAL cluster — THE world as shipped, NO injected geometry. `up` exits 0 only
    // after every pre-booked realm (home + galaxy) is granted in the ONE directory (C1), so the
    // crossing's dest head can never resolve to nothing (J-0: an unresolved dest is a PERMANENT
    // STRAND, not a soft failure).
    assert!(
        Command::new(launcher)
            .args(["up", "--slot", &RENDER_CROSSING_SLOT.to_string(), "--dual"])
            .status()
            .expect("run vd-devcluster up --dual")
            .success(),
        "up --dual must reach the C1 all-realms ready gate and exit 0",
    );

    // The DEST authority DERIVED from the shape's own shard list — never a literal node string.
    let addrs = ClusterAddrs::for_slot(ports);
    let dest_node = realm_shards(ClusterShape::Dual, &addrs, &DEV)
        .first()
        .map(|s| s.node)
        .expect("Dual pre-books the galaxy realm-shard");
    let dest_authority = format!("shard:{dest_node}");

    // Launch the headless capture client drawing THE world scene (the 5th process).
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_client"));
    cmd.current_dir(&cwd)
        .env("VD_AUTH_SIGNING_KEY", &signing_key)
        .args([
            "--name",
            CLIENT_NAME,
            "--agent-index",
            "0",
            "--gateway",
            &gateway.to_string(),
            "--client-quic",
            &client_quic.to_string(),
            "--trust-dir",
            trust_dir.to_str().expect("utf8 trust dir"),
            "--dev-control",
            &devctl.to_string(),
            "--allow-dev-control",
            "--capture",
            "--realm-boxes",
            &regions_path,
        ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = ChildGuard(cmd.spawn().expect("spawn capture client"));
    record_extra_pid(RENDER_CROSSING_SLOT, child.0.id()).expect("record capture-client pid");
    await_listener(devctl, &mut child.0);

    // ---- INSIDE: the dot spawns at the home star's centre. Wait for a delivered frame. --
    {
        let deadline = Instant::now() + DELIVERY_DEADLINE;
        loop {
            let s = poll_state(devctl);
            if s.own_entity.is_some()
                && !s.entities.is_empty()
                && s.location.as_deref() == Some(home_label.as_str())
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the dot never delivered inside the home system ({home_label}): {s:?}"
            );
            std::thread::sleep(DELIVERY_POLL);
        }
    }
    // THE ANNULUS-CLEAR CAPTURE PARK (batch review: the pixel verdicts must isolate the DOT — its
    // rectangle may never share a pixel with a planet's disc, and that is guaranteed against every
    // ORBITAL PHASE, so no flight-time or capture-latency race can slide a planet under it). Both
    // in-home captures (INSIDE + RETURNED) park here.
    let home_park = {
        let plan_state = poll_state(devctl);
        let plan_camera = live_scene_camera(
            &plan_state,
            &extents,
            CAPTURE_W as usize,
            CAPTURE_H as usize,
        );
        let config = vd_physics::worldgen::UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
        let annuli: Vec<(f64, f64)> = vd_physics::worldgen::moving_children_for_config(
            DEV.universe_seed,
            &config,
            roster.home,
        )
        .iter()
        .map(|(_, e)| planet_annulus_px(&plan_camera, e, config.planet.planet_soi_r_m))
        .collect();
        clean_home_park(&plan_camera, &annuli)
    };
    park_at(devctl, home_park);
    let inside = capture(devctl, &cwd, "inside", &scene, &extents, roster.home);
    assert_eq!(
        inside.state.location.as_deref(),
        Some(home_label.as_str()),
        "INSIDE: the session stands in the home system",
    );
    assert_dot_rect_clear_of_planets(&inside, &scene, roster.home, "INSIDE");
    assert!(
        dot_pixels_within_box_region(
            &inside.rgba,
            inside.w,
            inside.h,
            inside.clear,
            inside.dot_rect,
            inside.home_rect
        ),
        "INSIDE: the dot's pixels must fall inside the home shell's projected region \
         (dot {:?} home {:?})",
        inside.dot_rect,
        inside.home_rect,
    );
    // DOT-SENSITIVE (batch review): the containment assert above is satisfied by the shell disc's
    // own pixels, dot or no dot — this one is not: at least one pixel inside the dot's rect must
    // differ from EVERYTHING on the ring around it (the shell disc is uniform there; the planets
    // provably cannot reach the rect), and that pixel can only be the dot.
    assert!(
        dot_pixels_distinct_from_surround(
            &inside.rgba,
            inside.w,
            inside.h,
            inside.dot_rect,
            DOT_SURROUND_RING_PX
        ),
        "INSIDE: the dot itself must be pixel-visible against the shell disc (rect {:?}) — an \
         empty or background-only dot region is a FAIL, never a vacuous pass",
        inside.dot_rect,
    );

    // ---- OUT (flight law leg A): the ±Z polar corridor to the galaxy — the label is the arrival. --
    cross_leg(
        devctl,
        "A home->galaxy (polar corridor)",
        |_tick| OUTSIDE_PARK,
        &galaxy_label,
        LEG_DEADLINE,
    );
    // Park AT the corridor waypoint. The target is numerically the same point in both frames (J1),
    // so restating it after the flip is sound — the one crossing where that is true.
    park_at(devctl, OUTSIDE_PARK);
    let admin = loopback(ports.admin);
    let outside = capture(devctl, &cwd, "outside", &scene, &extents, roster.home);
    assert_eq!(
        outside.state.location.as_deref(),
        Some(galaxy_label.as_str()),
        "OUTSIDE: the session stands in the galaxy between-space",
    );
    // NON-VACUITY: a REAL transfer, not a label cosmetic — the directory CAS committed the dot's
    // Entity authority onto the pre-booked galaxy shard.
    let rows = directory_rows(admin);
    assert!(
        rows.iter()
            .any(|(key, authority)| key.starts_with("ent-") && authority == &dest_authority),
        "the galaxy shard ({dest_authority}) must OWN the re-homed dot's Entity row: {rows:?}",
    );
    // HONEST at the middle capture: nothing drawable contains the dot out here — the galaxy's
    // ~12483 m shell exceeds the renderable ceiling and SL3 never draws a containment boundary.
    assert_eq!(
        expected_box(&scene, outside.pos),
        None,
        "OUTSIDE: no drawn box contains the parked dot at {}",
        outside.pos,
    );
    assert!(
        !dot_pixels_within_box_region(
            &outside.rgba,
            outside.w,
            outside.h,
            outside.clear,
            outside.dot_rect,
            outside.home_rect
        ),
        "OUTSIDE: the dot's pixels must NOT fall inside the home shell's projected region \
         (dot {:?} home {:?})",
        outside.dot_rect,
        outside.home_rect,
    );
    // DOT-SENSITIVE (batch review): the negated containment above passes on an EMPTY frame too (the
    // two rectangles are disjoint by construction), so it carried no pixel information. State the
    // geometry as its own assert…
    assert!(
        rects_disjoint(outside.dot_rect, outside.home_rect),
        "OUTSIDE: the 2 km park exists exactly so the dot's rect {:?} clears the home silhouette \
         {:?} — a collision means the park or the camera geometry changed",
        outside.dot_rect,
        outside.home_rect,
    );
    // …and require the dot to have DRAWN out there: pixels unlike the empty space around its rect.
    // An empty dot region FAILS this — the middle capture is no longer satisfiable by nothing.
    assert!(
        dot_pixels_distinct_from_surround(
            &outside.rgba,
            outside.w,
            outside.h,
            outside.dot_rect,
            DOT_SURROUND_RING_PX
        ),
        "OUTSIDE: the dot itself must be pixel-visible in the between-space (rect {:?}) — an empty \
         dot region is a FAIL, never a vacuous pass",
        outside.dot_rect,
    );

    // ---- RETURN (flight law leg B): back through the acquire edge — the leg the owner watched
    // freeze, now under pixels. --
    cross_leg(
        devctl,
        "B galaxy->home (0,0,-60)",
        |_tick| RETURN_PARK,
        &home_label,
        LEG_DEADLINE,
    );
    // Re-park at the SAME annulus-clear capture point the INSIDE capture used (the return AIM only
    // committed the crossing; pixels are always taken where the dot's rect is provably planet-free).
    park_at(devctl, home_park);
    let returned = capture(devctl, &cwd, "returned", &scene, &extents, roster.home);
    assert_eq!(
        returned.state.location.as_deref(),
        Some(home_label.as_str()),
        "RETURNED: the session stands in the home system again",
    );
    assert_dot_rect_clear_of_planets(&returned, &scene, roster.home, "RETURNED");
    assert!(
        dot_pixels_within_box_region(
            &returned.rgba,
            returned.w,
            returned.h,
            returned.clear,
            returned.dot_rect,
            returned.home_rect
        ),
        "RETURNED: the dot's pixels must fall inside the home shell's projected region again \
         (dot {:?} home {:?})",
        returned.dot_rect,
        returned.home_rect,
    );
    // DOT-SENSITIVE (batch review): the return leg is the one the owner watched freeze — the pixels
    // must show the DOT back inside, not merely the shell disc that was there all along.
    assert!(
        dot_pixels_distinct_from_surround(
            &returned.rgba,
            returned.w,
            returned.h,
            returned.dot_rect,
            DOT_SURROUND_RING_PX
        ),
        "RETURNED: the dot itself must be pixel-visible against the shell disc again (rect {:?})",
        returned.dot_rect,
    );

    // HR6: all three screenshots recorded in the run manifest (the capture pipeline ran end to end).
    let run_dir = cwd.join(&returned.shot);
    let run_dir = run_dir.parent().and_then(Path::parent).expect("run dir");
    let manifest = RunManifest::from_json(
        &std::fs::read_to_string(run_dir.join(MANIFEST_FILENAME)).expect("read manifest.json"),
    )
    .expect("parse manifest");
    for (label, shot) in [
        ("inside", &inside.shot),
        ("outside", &outside.shot),
        ("returned", &returned.shot),
    ] {
        assert!(
            manifest
                .captures
                .iter()
                .any(|c| c.kind == CaptureKind::Screenshot && shot.ends_with(&c.path)),
            "manifest must record the '{label}' screenshot, got {:?}",
            manifest.captures,
        );
    }

    println!(
        "G-RENDER-CROSSING-SMOKE: {}x{} · pos {} → {} → {} · location {:?} → {:?} → {:?} · \
         home rect (inside) {:?} · camera eye (inside) {:?}",
        inside.w,
        inside.h,
        inside.pos,
        outside.pos,
        returned.pos,
        inside.state.location,
        outside.state.location,
        returned.state.location,
        (inside.home_rect.min.x as i32, inside.home_rect.max.x as i32),
        inside.camera.eye,
    );
}
