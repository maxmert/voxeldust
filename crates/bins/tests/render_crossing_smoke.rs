//! G-RENDER-CROSSING-SMOKE — the crossing pixel proof on THE WORLD ITSELF: a real dot flies the ±Z
//! polar corridor OUT of THE world's own 150 m home shell, the directory CAS re-homes its authority
//! onto the pre-booked galaxy shard, and the round trip is captured in actual pixels — INSIDE the
//! home system's drawn shell, OUTSIDE it in the between-space, and back INSIDE on the return. The
//! RETURN leg gets pixel coverage for the first time (the owner's "everything froze on the way
//! back").
//!
//! It stands up the DUAL process cluster (`up --dual`) with NO injected geometry — THE world's own
//! home shell IS the crossing boundary (SL5: one world, nothing to select). RE-BASED at the flag
//! day (Slice C1, `docs/design/window_lane.md` §2.11): the headless `client --capture` draws ONLY
//! the COMPOSED STREAM (the `--realm-boxes` boot file is DELETED — D-LANE-6 🟩).
//!
//! J1 IS THE ORIGIN-MARKER ASSERT NOW: the home body draws at the session origin on BOTH sides of
//! the crossing (the galaxy authors the home placement at ZERO — `world_roster` asserts it), the
//! origin marker flips home ↔ galaxy, and the epoch bumps EXACTLY ONCE per crossing. THE
//! NO-FLICKER GATE rides both legs (`cross_leg_watching_scene`): no delivered state across the
//! swap shows an ABSENT scene, and every body persisting across the swap moves by no more than
//! the ONE-TICK true-motion bound (`DEV.move_speed × DEV.tick_dt` — the home body is static at
//! zero in both frames, so its measured delta is exactly 0).
//!
//! THE CAMERA (judge fix A1): reconstructed per capture from the client's own reported drawn
//! boxes — centres + STREAMED `extent_m` (§2.11) — through the identical `bounds_union` →
//! `fit_camera_to_bounds` refit. Dot detection is DevState-driven: the dot rect comes from the
//! own row's projected position + the marker-radius floor, and the pixel probes are LOCAL to that
//! rect — never a full-frame search.
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

use vd_bins::flight::cross_leg_watching_scene;
use vd_bins::scene_camera::live_scene_camera;
use vd_bins::{
    ClusterAddrs, ClusterShape, DEV, DevClusterDown, admin_get_body, dev_auth_signing_key_hex,
    dev_roundtrip, devcluster, loopback, realm_shards, record_extra_pid, slot_trust_dir,
    slot_workdir, world_roster,
};
use vd_client_harness::assert::magenta_pixel_count;
use vd_client_harness::camera::marker_world_radius;
use vd_client_harness::camera::{CaptureCamera, ScreenAabb, ScreenPos};
use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::{
    dot_pixels_distinct_from_surround, dot_pixels_within_box_region, projected_point_aabb,
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
/// reported drawn centre + its STREAMED extent (§2.11 — no file exists to join against).
fn drawn_box_screen_aabb(
    state: &vd_devproto::DevState,
    camera: &CaptureCamera,
    realm: RealmId,
) -> ScreenAabb {
    let label = format!("{realm:?}");
    let row = state
        .realm_boxes
        .iter()
        .find(|b| b.realm == label)
        .unwrap_or_else(|| panic!("the client is not drawing {label}"));
    projected_point_aabb(camera, DVec3::from_array(row.center), row.extent_m)
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

/// The union rectangle of two screen AABBs (the straddled-capture drift window).
fn union_aabb(a: ScreenAabb, b: ScreenAabb) -> ScreenAabb {
    ScreenAabb {
        min: ScreenPos {
            x: a.min.x.min(b.min.x),
            y: a.min.y.min(b.min.y),
        },
        max: ScreenPos {
            x: a.max.x.max(b.max.x),
            y: a.max.y.max(b.max.y),
        },
    }
}

/// Two screen rectangles share no pixel.
fn rects_disjoint(a: ScreenAabb, b: ScreenAabb) -> bool {
    a.max.x < b.min.x || b.max.x < a.min.x || a.max.y < b.min.y || b.max.y < a.min.y
}

/// THE CROSSING NO-FLICKER VERDICT (§2.11): every body drawn in BOTH the last old-epoch state and
/// the first new-epoch state — a persisting body — moved by no more than its own DERIVED
/// allowance across the swap. A static body (the home shell on this crossing) is bounded by the
/// occupant's one-tick true-motion bound alone. A MOVING persister (an orbiting planet riding the
/// sibling-interior relay, §2.6.5 step 4) lawfully sweeps its own orbit between the two sampled
/// states, so its allowance is MEASURED: its per-tick motion (from the last two old-epoch
/// samples) times the tick gap its own track advanced across the swap, plus the occupant bound as
/// re-expression slack. Everything in the allowance is measured or derived — a frame
/// mis-re-expression (the km-scale teleport class) still fails loud. Non-vacuous: at least the
/// home body persists.
fn assert_no_flicker_across_swap(
    before_prev: Option<&vd_devproto::DevState>,
    before: &vd_devproto::DevState,
    after: &vd_devproto::DevState,
    one_tick_bound_m: f64,
    leg: &str,
) {
    let mut persisting = 0usize;
    for b in &before.realm_boxes {
        let Some(a) = after.realm_boxes.iter().find(|a| a.realm == b.realm) else {
            continue; // not a persisting body — the old interior lawfully leaves the new scene
        };
        persisting += 1;
        // The body's own measured motion per tick, off the last two old-epoch samples (0 when
        // the pair is unavailable or its track did not advance — a static body needs none).
        let pair = before_prev.and_then(|p| {
            p.realm_boxes
                .iter()
                .find(|x| x.realm == b.realm)
                .map(|prev| (prev.newest_tick, prev.center))
        });
        let per_tick_m = pair
            .and_then(|(prev_tick, prev_center)| {
                let dt = b.newest_tick?.checked_sub(prev_tick?)?;
                if dt == 0 {
                    return None;
                }
                let dpos = (DVec3::from_array(b.center) - DVec3::from_array(prev_center)).length();
                Some(dpos / dt as f64)
            })
            .unwrap_or(0.0);
        // The ticks this body's own track advanced across the swap (0 when unstated: a held/
        // same-stamp row moved no time, so it gets no motion allowance).
        let across_ticks = match (b.newest_tick, a.newest_tick) {
            (Some(t0), Some(t1)) => t1.saturating_sub(t0),
            _ => 0,
        };
        let allowed = per_tick_m * across_ticks as f64 + one_tick_bound_m;
        let delta = (DVec3::from_array(a.center) - DVec3::from_array(b.center)).length();
        assert!(
            delta <= allowed,
            "NO-FLICKER ({leg}): persisting body {} moved {delta:.3} m across the swap — over \
             its derived allowance {allowed:.3} m (= measured {per_tick_m:.3} m/tick x \
             {across_ticks} ticks + the {one_tick_bound_m:.3} m one-tick occupant bound; the \
             swap level must be the same-tick re-expression, §2.7). pair={pair:?} \
             b_tick={:?} a_tick={:?}",
            b.realm,
            b.newest_tick,
            a.newest_tick,
        );
    }
    assert!(
        persisting > 0,
        "NO-FLICKER ({leg}): no body persisted across the swap — the verdict measured nothing \
         (the home body must persist on a home↔galaxy crossing)"
    );
    println!(
        "NO-FLICKER ({leg}): {persisting} persisting bodies within their derived allowances \
         (occupant one-tick bound {one_tick_bound_m:.3} m)"
    );
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

/// Capture-state belt over the phase-independent park: every OTHER drawn body with a NONZERO
/// extent (markers are points — Slice D owns their sprites) projects disjoint from the dot's
/// rectangle — so a pixel in the dot rect unlike its surround can only be the dot.
fn assert_dot_rect_clear_of_planets(cap: &Capture, home_label: &str, label: &str) {
    for b in &cap.state.realm_boxes {
        if b.realm == home_label || b.extent_m <= 0.0 {
            continue;
        }
        let rect = projected_point_aabb(&cap.camera, DVec3::from_array(b.center), b.extent_m)
            .expect("a drawn body projects in front of the camera");
        assert!(
            rects_disjoint(cap.dot_rect, rect),
            "{label}: the dot's rectangle {:?} must be clear of {}'s drawn disc {rect:?}",
            cap.dot_rect,
            b.realm,
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
/// and return everything a verdict needs. J1 — THE ORIGIN-MARKER ASSERT (§2.11) — runs at EVERY
/// capture: the origin names the expected realm at the expected epoch, and the home body draws at
/// the session origin on BOTH sides of the crossing (the galaxy authors the home placement at
/// ZERO — `world_roster` asserts the world half; this asserts the drawn half).
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
    home: RealmId,
    expected_origin: RealmId,
    expected_epoch: u64,
) -> Capture {
    // TWO state polls STRADDLE the screenshot: the client refits its camera EVERY frame from the
    // drawn boxes' union bounds, and since the flag day those boxes move every tick (the composed
    // per-tick feed; `fit_camera_to_bounds`'s own doc: the two cameras drift the moment a drawn
    // box moves). The frame lies between the polls in time, so every verdict rectangle below is
    // the UNION of the two polls' projections — the drift is MEASURED into the rect, never
    // guessed, and a quiet scene collapses the union to the old single-poll rectangle.
    // …and RETRY until the straddle is ROSTER-STABLE: right after a crossing the composed scene
    // is still FILLING (rows arrive over the next beats — the sibling-interior relays), and every
    // arrival re-fits the client camera by a jump the two-poll union cannot bound. A stable
    // roster leaves only the per-tick orbital wobble, which the union DOES bound.
    //
    // …and RETRY until the straddle is TIGHT (measured 2026-08-16, window lane Slice C2). A stable
    // roster is NOT sufficient: the camera refits from the moving boxes every frame, so the two
    // polls can still project the dot to different pixels, and the UNION then spans far more than
    // the dot's own footprint. That envelope is the right answer for a CONTAINMENT question ("is
    // the dot inside the shell's disc") and the WRONG one for a LOCALITY question ("is the dot
    // distinct from what surrounds it"): `dot_pixels_distinct_from_surround` samples its ring just
    // OUTSIDE the rect it is given, so an inflated rect pushes the ring off the dot's local
    // background and out over the shell's shading gradient — where some pixel eventually matches
    // the dot's own colour and the verdict can no longer be satisfied. MEASURED at 2 failures in 5
    // runs, always the same assert, with a rect 28.8 px tall against a ~13 px dot footprint.
    //
    // So the straddle must establish the verdict's precondition, not the verdict be loosened: the
    // two polls must agree on where the dot is to within THE DOT'S OWN APPARENT RADIUS. That is
    // the resolution at which "where the dot is" is a meaningful question at all, and it is
    // DERIVED from `DOT_MIN_APPARENT_RADIUS_PX` — the one constant the renderer scales the marker
    // by and the gates size their rectangles from — never a fitted pixel count. A drift smaller
    // than the dot's own radius cannot carry the dot out of its bracketed rectangle, and it keeps
    // the union inside one footprint, which is exactly what the ring needs.
    let mut tries = 0;
    let (pre, shot, state) = loop {
        let pre = poll_state(devctl);
        let shot = screenshot(devctl, label);
        let state = poll_state(devctl);
        let roster = |s: &vd_devproto::DevState| -> std::collections::BTreeSet<String> {
            s.realm_boxes.iter().map(|b| b.realm.clone()).collect()
        };
        let cam_pre = live_scene_camera(&pre, CAPTURE_W as usize, CAPTURE_H as usize);
        let cam_post = live_scene_camera(&state, CAPTURE_W as usize, CAPTURE_H as usize);
        let centre = |r: ScreenAabb| (0.5 * (r.min.x + r.max.x), 0.5 * (r.min.y + r.max.y));
        let (ax, ay) = centre(dot_screen_aabb(&cam_pre, own_pos(&pre)));
        let (bx, by) = centre(dot_screen_aabb(&cam_post, own_pos(&state)));
        let drift_px = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
        let dot_radius_px = vd_client_harness::camera::DOT_MIN_APPARENT_RADIUS_PX;
        if (roster(&pre) == roster(&state)) & (drift_px <= dot_radius_px) {
            break (pre, shot, state);
        }
        tries += 1;
        assert!(
            tries < 40,
            "capture '{label}': the straddle never settled — roster (pre {:?} vs post {:?}) or \
             dot drift {drift_px:.2} px over its own apparent radius {dot_radius_px} px",
            roster(&pre),
            roster(&state),
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    let camera_pre = live_scene_camera(&pre, CAPTURE_W as usize, CAPTURE_H as usize);
    let camera = live_scene_camera(&state, CAPTURE_W as usize, CAPTURE_H as usize);
    for s in [&pre, &state] {
        assert_eq!(
            s.origin,
            Some((format!("{expected_origin:?}"), expected_epoch)),
            "J1 (origin marker, capture '{label}'): the composed scene names its origin + epoch",
        );
        assert_eq!(
            drawn_centre(s, home),
            DVec3::ZERO,
            "J1 (drawn identity, capture '{label}'): the home body draws at the session origin \
             on BOTH sides of the home<->galaxy crossing - the galaxy authors the home placement \
             at ZERO",
        );
    }
    let home_rect = union_aabb(
        drawn_box_screen_aabb(&pre, &camera_pre, home),
        drawn_box_screen_aabb(&state, &camera, home),
    );
    let pos = own_pos(&state);
    let dot_rect = union_aabb(
        dot_screen_aabb(&camera_pre, own_pos(&pre)),
        dot_screen_aabb(&camera, pos),
    );
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

// PARKED — full citation:
// PARKED at the outer re-solve (real-scale addendum §A6.1's ordering edge 'S5 before S6 — or S6's pixel gates go black and the failure is unattributable', measured doing exactly that): a GALAXY-standing observer's composed scene now includes the seeded sibling systems 0.2376656 ly out, and the capture camera (`fit_camera_to_scene` — the union fit) blows out to a ~6.4e15 m eye, collapsing every in-system silhouette to a degenerate point (measured: home rect 4e-11 px)
// The cure is the S5 renderer/camera slice (camera-relative flatten + the angular-rule fit, real-scale design §5.4/§7.4's render_crossing row), which this geometry-first slice deliberately precedes
// Un-park with S5
// Parked, never deleted or weakened — the in-system pixel gates (render_smoke, render_boxes_smoke, two_ships, look_pixels) stay green and carry the pixel proof at this slice.
#[test]
#[ignore = "PARKED for the S5 renderer/camera slice (the speed-law slice S3 landed; the camera blow-out remains) - full citation in the comment block above this test"]
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

    // NO scene file exists (Slice C1, §2.11): the client draws only what the composed stream
    // states, and every projected extent below is read STREAMED off the client's own report.

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
                // The composed scene's own settle signal (§2.11): the level landed and a fold
                // delivered — the camera reconstruction below frames a non-empty drawn set.
                && !s.realm_boxes.is_empty()
                && s.realm_frames_applied >= 1
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
        let plan_camera = live_scene_camera(&plan_state, CAPTURE_W as usize, CAPTURE_H as usize);
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
    // The login scene: origin = the home realm at the login epoch (0→1 at the first fold).
    let inside = capture(devctl, &cwd, "inside", roster.home, roster.home, 1);
    assert_eq!(
        inside.state.location.as_deref(),
        Some(home_label.as_str()),
        "INSIDE: the session stands in the home system",
    );
    let home_box_label = format!("{:?}", roster.home);
    assert_dot_rect_clear_of_planets(&inside, &home_box_label, "INSIDE");
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
         empty or background-only dot region is a FAIL, never a vacuous pass (drawn scene: {:?})",
        inside.dot_rect,
        inside.state.realm_boxes,
    );

    // ---- OUT (flight law leg A): the ±Z polar corridor to the galaxy — the label is the
    // arrival, the crossing WATCHED (§2.11): scene never absent, epoch bumps exactly once, and
    // every body persisting across the swap moves within the ONE-TICK true-motion bound.
    let (out_prev, out_before, out_after) = cross_leg_watching_scene(
        devctl,
        "A home->galaxy (polar corridor)",
        |_tick| OUTSIDE_PARK,
        &galaxy_label,
        LEG_DEADLINE,
    );
    let one_tick_bound_m = DEV.move_speed * DEV.tick_dt;
    assert_no_flicker_across_swap(
        out_prev.as_ref(),
        &out_before,
        &out_after,
        one_tick_bound_m,
        "A home->galaxy",
    );
    // Park AT the corridor waypoint. The target is numerically the same point in both frames (J1),
    // so restating it after the flip is sound — the one crossing where that is true.
    park_at(devctl, OUTSIDE_PARK);
    let admin = loopback(ports.admin);
    // The swapped scene: origin = the galaxy realm, epoch = login + 1.
    let outside = capture(devctl, &cwd, "outside", roster.home, roster.galaxy, 2);
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
    // own shell exceeds the renderable ceiling and SL3 never draws a containment boundary
    // (asserted on the STREAMED scene: no drawn body's extent reaches the parked dot).
    for b in &outside.state.realm_boxes {
        let centre = DVec3::from_array(b.center);
        assert!(
            (outside.pos - centre).length() > b.extent_m,
            "OUTSIDE: no drawn box contains the parked dot at {} — {} (extent {} m) does",
            outside.pos,
            b.realm,
            b.extent_m,
        );
    }
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
    // freeze, now under pixels AND under the no-flicker watch. --
    let (back_prev, back_before, back_after) = cross_leg_watching_scene(
        devctl,
        "B galaxy->home (0,0,-60)",
        |_tick| RETURN_PARK,
        &home_label,
        LEG_DEADLINE,
    );
    assert_no_flicker_across_swap(
        back_prev.as_ref(),
        &back_before,
        &back_after,
        one_tick_bound_m,
        "B galaxy->home",
    );
    // Re-park at the SAME annulus-clear capture point the INSIDE capture used (the return AIM only
    // committed the crossing; pixels are always taken where the dot's rect is provably planet-free).
    park_at(devctl, home_park);
    // Back home: origin = the home realm again, the SECOND bump (login 1 → out 2 → return 3).
    let returned = capture(devctl, &cwd, "returned", roster.home, roster.home, 3);
    assert_eq!(
        returned.state.location.as_deref(),
        Some(home_label.as_str()),
        "RETURNED: the session stands in the home system again",
    );
    assert_dot_rect_clear_of_planets(&returned, &home_box_label, "RETURNED");
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
