//! G-RENDER-BOXES-SMOKE — the realm-box pixel proof on THE WORLD, RE-BASED at the flag day
//! (Slice C1, `docs/design/window_lane.md` §2.11): bring the local-process QUIC cluster up,
//! launch a HEADLESS `client --capture` drawing ONLY the COMPOSED STREAM (the `--realm-boxes`
//! boot file is DELETED — D-LANE-6 🟩, one world, one source), wait on the composed feed's OWN
//! settle signal (`RealmFramesApplied ≥ 1` — the demand loop's first delivered fold, never a
//! sleep literal), drive it to a real wgpu-readback screenshot, and assert THE world's
//! HOME-system shell is PIXEL-VISIBLE and correctly placed — its color region holds non-clear
//! pixels INSIDE its projected screen AABB (H2: "the box drew WHERE it should"). Zero magenta.
//!
//! STREAM anti-vacuity: the DRAWN set (the state the manifest's capture records) must equal the
//! gateway-emitted LEVEL set — on THE world's Single topology the origin's own LOOK body (the
//! home shell) plus one MARKER point per planet (the parent-authored reflected datum; markers
//! are points until Slice D's sprites) — and the ORIGIN MARKER must name the home realm. A lost
//! row, an extra row, or a wrong body kind cannot hide behind the pixel floor.
//!
//! THE CAMERA (the A1 discipline, shared with the crossing gate): the client's offscreen capture
//! camera refits `fit_camera_to_scene` over the LIVE overlaid scene every frame — so the
//! projection camera is RECONSTRUCTED from the client's own reported drawn boxes
//! (`DevState.realm_boxes` centres × their STREAMED `extent_m`), never from a file.
//!
//! GPU PRECONDITION (same as G-RENDER-SMOKE): renders through wgpu, REQUIRES a working GPU adapter,
//! LOCAL-only (no CI, no software-raster fallback). On a GPU-less host the capture client cannot
//! start; the test fails naming the precondition.
//!
//! Run via `just render-boxes-smoke` (builds the client with `--features dev-control,render`).
//! Under a plain `cargo test --workspace` (no features) this file compiles to ZERO tests.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::scene_camera::live_scene_camera;
use vd_bins::{
    DEV, DevClusterDown, boot_world, dev_auth_signing_key_hex, dev_roundtrip, devcluster, loopback,
    record_extra_pid, slot_trust_dir, slot_workdir, world_roster,
};
use vd_client_harness::assert::{MAGENTA, magenta_pixel_count};
use vd_client_harness::camera::ScreenAabb;
use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::projected_point_aabb;
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::glam::DVec3;
use vd_devproto::WORKTREE_SLOT_CEILING;
use vd_devproto::{DevPortScheme, DevRequest, DevResponse, WaitField, WaitOp, WaitPredicate};

const CLIENT_NAME: &str = "g-render-boxes";
/// A DISTINCT test-reserved slot (self-contained here so no shared registry needs touching): one
/// past the render-smoke slot, still above `WORKTREE_SLOT_CEILING` so a developer's live cluster
/// can never collide with it.
const RENDER_BOXES_SLOT: u16 = WORKTREE_SLOT_CEILING + 19; // 83: G-RENDER-BOXES-SMOKE
/// How long to wait for the capture client's dev-control listener to come up.
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const READY_POLL: Duration = Duration::from_millis(200);
/// Step-tick budget for the "delivered frame arrived" wait (≈30 s at the 20 Hz client step).
const CAPTURE_WAIT_TICKS: u64 = 600;

/// Kill the capture client on drop — the 4th process beyond the cluster's 3 nodes.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// One dev-control request → response over the SHARED `vd_bins::dev_roundtrip` framing.
fn round_trip(port: u16, req: &DevRequest) -> DevResponse {
    dev_roundtrip(port, req).unwrap_or_else(|e| {
        panic!("dev-control round-trip on {port} failed: {e} — capture client gone? (GPU precondition?)")
    })
}

/// Block until the dev-control listener accepts, or fail naming the GPU precondition.
fn await_listener(port: u16, child: &mut Child) {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if TcpStream::connect(loopback(port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!(
                "the --capture client exited early ({status}) before its dev-control listener \
                 came up — G-RENDER-BOXES-SMOKE requires a working GPU adapter (see the module docs \
                 / `just render-boxes-smoke`)"
            );
        }
        assert!(
            Instant::now() < deadline,
            "dev-control listener on {port} never came up within {READY_TIMEOUT:?} \
             (capture client stuck; GPU precondition?)"
        );
        std::thread::sleep(READY_POLL);
    }
}

/// The rim-probe half-size (px): small squares straddling the shell's silhouette edge.
const RIM_PROBE_HALF_PX: f64 = 3.0;
/// The inner probe's centre inset from the projected rect radius — deep enough that pixelization
/// cannot push it outside the disc (the silhouette only ever EXCEEDS the projected chord).
const RIM_INNER_INSET_PX: f64 = 7.0;
/// The outer probe's centre outset — past the silhouette bulge (≤ 2 % of the ~167 px radius
/// ≈ 3.3 px) plus MSAA edge blending, so the probe sits in provably-empty space.
const RIM_OUTER_OUTSET_PX: f64 = 9.0;
/// How much each planet's drawn rect is inflated in the probe-angle search — covers the planets'
/// orbital drift between the state sample and the screenshot's own frame.
const RIM_PLANET_DRIFT_PX: f64 = 12.0;
/// The probe-angle search step (deg) around the rim circle.
const RIM_ANGLE_STEP_DEG: f64 = 5.0;

/// A square probe rect of half-size `half`, centred `radius` px from `(cx, cy)` along `theta`
/// (pixel coords, y down).
fn rim_probe(cx: f64, cy: f64, theta_rad: f64, radius: f64, half: f64) -> ScreenAabb {
    let px = cx + theta_rad.cos() * radius;
    let py = cy + theta_rad.sin() * radius;
    ScreenAabb {
        min: vd_client_harness::camera::ScreenPos {
            x: px - half,
            y: py - half,
        },
        max: vd_client_harness::camera::ScreenPos {
            x: px + half,
            y: py + half,
        },
    }
}

/// Two screen rectangles share no pixel.
fn rects_disjoint(a: ScreenAabb, b: ScreenAabb) -> bool {
    a.max.x < b.min.x || b.max.x < a.min.x || a.max.y < b.min.y || b.max.y < a.min.y
}

/// Count non-clear pixels of `rgba` (top-left origin, `(y*w+x)*4`) inside `region`. The H2 verdict:
/// the box's color region must be NON-EMPTY inside its projected screen AABB — "the box drew where
/// it should", not merely "something drew somewhere".
fn nonclear_in_region(rgba: &[u8], w: usize, h: usize, clear: [u8; 4], region: ScreenAabb) -> u64 {
    let x0 = region.min.x.floor().clamp(0.0, w as f64) as usize;
    let y0 = region.min.y.floor().clamp(0.0, h as f64) as usize;
    let x1 = (region.max.x.ceil().clamp(0.0, w as f64) as usize).max(x0);
    let y1 = (region.max.y.ceil().clamp(0.0, h as f64) as usize).max(y0);
    let mut count = 0u64;
    for py in y0..y1 {
        for px in x0..x1 {
            let idx = (py * w + px) * 4;
            if let Some(pixel) = rgba.get(idx..idx + 4)
                && pixel != clear.as_slice()
            {
                count += 1;
            }
        }
    }
    count
}

#[test]
#[ignore = "PARKED under D-LOOK-3 (the true-scale camera far plane, MEASURED): no rim angle \
            clears the planets' discs because the home system's SHELL — the subject of the rim \
            assert — is real geometry 1.58e11 m across and is clipped by the 120 000 \
            render-metre far plane, while the bodies survive as backdrop-sphere markers clustered \
            near the view centre (the failure prints all ten discs). Same class as \
            look_pixels/warp_pixels/render_smoke; un-parks with the S5 render-scale slice."]
fn g_render_boxes_smoke_shows_the_home_system_shell_pixel_visible_in_its_screen_region() {
    // FIRST statement: hold the process tier for the whole body, so it outlives the cluster reap
    // that frees the ports. See `vd_bins::cluster_tier`.
    let _tier = vd_bins::cluster_tier();
    let launcher = env!("CARGO_BIN_EXE_vd-devcluster");
    // Clean slate (idempotent) — and the SIGKILL-orphan reaper.
    let _ = devcluster(launcher, "down", RENDER_BOXES_SLOT);
    let _down = DevClusterDown::new(launcher, RENDER_BOXES_SLOT);
    assert!(
        devcluster(launcher, "up", RENDER_BOXES_SLOT).success(),
        "dev-cluster up should reach ready and exit 0"
    );

    let ports = DevPortScheme::DEFAULT
        .slot_ports(RENDER_BOXES_SLOT)
        .expect("slot ports");
    let gateway = loopback(ports.gateway);
    let devctl = ports.dev_control(0).expect("dev-control port");
    let client_quic = ports.client_quic(0).expect("client-quic port");
    let trust_dir = slot_trust_dir(RENDER_BOXES_SLOT);
    let signing_key = dev_auth_signing_key_hex();
    // A contained cwd so the client's `runs/` (and the emitted regions.json) land here (reaped on `down`).
    let cwd = slot_workdir(RENDER_BOXES_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    // THE expected drawn set, from THE world itself (no file exists to emit): the home system's
    // own LOOK body + one MARKER point per planet of the home system — exactly what the composed
    // level states on the Single topology (the chain is the home realm alone; its planets are
    // dormant, so THE DRAW LAW gives each its parent's reflected marker and nothing else).
    let roster = world_roster(&DEV);
    let world = boot_world(DEV.universe_seed, DEV.move_speed, DEV.tick_dt);
    let home_planets: Vec<String> = world
        .regions()
        .iter()
        .filter(|r| r.parent == Some(roster.home))
        .map(|r| format!("{:?}", r.realm))
        .collect();

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
    record_extra_pid(RENDER_BOXES_SLOT, child.0.id()).expect("record capture-client pid");

    await_listener(devctl, &mut child.0);

    // THE SETTLE SIGNAL (§2.11 — never a sleep literal): the composed feed's own first delivered
    // fold. `RealmFramesApplied ≥ 1` means the level landed (the scene exists), the windows
    // confirmed, and a composed datagram was applied — the demand loop's own "the picture is on".
    let live = round_trip(
        devctl,
        &DevRequest::WaitUntil {
            predicate: WaitPredicate {
                field: WaitField::RealmFramesApplied,
                op: WaitOp::Ge,
                value: 1,
            },
            max_ticks: CAPTURE_WAIT_TICKS,
        },
    );
    assert!(
        matches!(live, DevResponse::State { .. }),
        "client should reach a delivered composed fold, got {live:?}"
    );

    // Sample the drawn scene, then capture — the camera reconstruction reads THIS state (the A1
    // discipline: the client refits per frame over the live overlaid scene; the residual skew is
    // sub-frame).
    let state = match round_trip(devctl, &DevRequest::State) {
        DevResponse::State { state } => state,
        other => panic!("expected DevState, got {other:?}"),
    };
    let shot = round_trip(
        devctl,
        &DevRequest::Screenshot {
            at_tick: None,
            label: Some("boxes".to_owned()),
        },
    );
    let (rel_path, tick) = match shot {
        DevResponse::Captured { path, tick } => (path, tick),
        other => panic!("screenshot was not captured (GPU precondition unmet?): {other:?}"),
    };

    // STREAM anti-vacuity (§2.11): the DRAWN set equals the gateway-emitted level set — the home
    // LOOK body + one MARKER per planet — and the origin marker names the home realm. Body kinds
    // asserted per row: a planet that silently gained a mesh (or a home that degraded to a
    // marker) fails here, not behind the pixel floor.
    let home_label = format!("{:?}", roster.home);
    let mut drawn: Vec<(String, String)> = state
        .realm_boxes
        .iter()
        .map(|b| (b.realm.clone(), b.body_kind.clone()))
        .collect();
    drawn.sort();
    let mut expected: Vec<(String, String)> = home_planets
        .iter()
        .map(|label| (label.clone(), "marker".to_owned()))
        .chain(std::iter::once((home_label.clone(), "look".to_owned())))
        .collect();
    expected.sort();
    assert_eq!(
        drawn, expected,
        "the drawn (realm, body_kind) set must equal the gateway-emitted level set",
    );
    assert_eq!(
        state.origin,
        Some((home_label.clone(), 1)),
        "the origin marker names the HOME realm at the login epoch",
    );

    // The home shell's projected screen rectangle through the RECONSTRUCTED live camera: the
    // drawn centre the client reports (zero — the origin row) + the shell's STREAMED extent.
    let camera = live_scene_camera(&state, CAPTURE_W as usize, CAPTURE_H as usize);
    let home = state
        .realm_boxes
        .iter()
        .find(|b| b.realm == home_label)
        .expect("the home shell is drawn");
    let home_centre = DVec3::from_array(home.center);
    let radius = home.extent_m;
    let region = projected_point_aabb(&camera, home_centre, radius)
        .expect("the home shell projects in front of the fitted camera");

    // Decode the captured PNG → RGBA8 (the only decode in the gate; the assertion is Tier-A).
    let png = cwd.join(&rel_path);
    let img = image::open(&png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let buf = img.into_raw();

    // Self-calibrate the clear color from the top-right corner (background — the camera frames the
    // union bounds with margin, so the corner sits outside the shell). A blank frame still fails:
    // every pixel equals the corner ⇒ zero non-clear anywhere.
    let corner = (w - 1) * 4;
    let clear = [
        buf[corner],
        buf[corner + 1],
        buf[corner + 2],
        buf[corner + 3],
    ];

    // (1) Zero magenta — nothing failed to draw with a real material.
    let magenta = magenta_pixel_count(&buf);
    // (2) The H2 pixel assertion: the home shell's color region is NON-EMPTY inside its projected
    // screen AABB (the reconstructed live camera). This pins "the shell drew WHERE it should".
    let box_pixels = nonclear_in_region(&buf, w, h, clear, region);

    println!(
        "G-RENDER-BOXES-SMOKE: {w}x{h} frame at tick {tick:?}, clear {clear:?}, \
         home screen AABB [{:.1},{:.1}]-[{:.1},{:.1}] → magenta {magenta}, box_region_pixels {box_pixels}",
        region.min.x, region.min.y, region.max.x, region.max.y,
    );
    assert_eq!(
        magenta, 0,
        "the frame must be magenta-free (no missing-material draws), got {magenta} (sentinel {MAGENTA:?})"
    );
    assert!(
        box_pixels > 0,
        "THE world's home shell region holds NO pixels at all inside \
         [{:.1},{:.1}]-[{:.1},{:.1}] — a bare content fraction would miss this. \
         Did anything draw, and did the reconstructed camera frame the scene?",
        region.min.x,
        region.min.y,
        region.max.x,
        region.max.y,
    );

    // ---- H2, ISOLATED TO THE SHELL (batch review): the whole-region count above is satisfied by
    // ANY paint inside the rect — five nested planet boxes and the avatar dot sit in it, so a single
    // planet disc used to pass it with the shell never rasterized (and before the capture schedule
    // despawned the reference scaffold, the 500 m ground slab did too). The shell-isolating proof is
    // the RIM: only the home shell can reach its own silhouette edge — the generator solves every
    // planet's worst apoapsis face strictly inside the shell, and the probe ANGLE is chosen clear of
    // every planet's CURRENT drawn disc (drift-inflated) — so a probe just inside the rim must hold
    // paint, and its mirror just outside must hold NONE ("the shell drew WHERE it should", both
    // edges of "where").
    let cx = (region.min.x + region.max.x) * 0.5;
    let cy = (region.min.y + region.max.y) * 0.5;
    let rim_r = (region.max.x - region.min.x) * 0.5;
    // The paint-exclusion rects for the rim search: every OTHER drawn body with a nonzero
    // extent, drift-inflated. On THE world today the planets are zero-extent MARKER points
    // (Slice D owns their sprites), so this list is empty — kept GENERIC so a planet that gains
    // a look (a spun-up neighbour in a future topology) is excluded again without an edit.
    let planet_rects: Vec<(String, ScreenAabb)> = state
        .realm_boxes
        .iter()
        .filter(|b| b.realm != home_label && b.extent_m > 0.0)
        .map(|b| {
            let rect = projected_point_aabb(&camera, DVec3::from_array(b.center), b.extent_m)
                .expect("a drawn body projects in front of the fitted camera");
            (
                b.realm.clone(),
                ScreenAabb {
                    min: vd_client_harness::camera::ScreenPos {
                        x: rect.min.x - RIM_PLANET_DRIFT_PX,
                        y: rect.min.y - RIM_PLANET_DRIFT_PX,
                    },
                    max: vd_client_harness::camera::ScreenPos {
                        x: rect.max.x + RIM_PLANET_DRIFT_PX,
                        y: rect.max.y + RIM_PLANET_DRIFT_PX,
                    },
                },
            )
        })
        .collect();
    // The probe angle: swept from the disc's BOTTOM (far from the top-left HUD; the rim's topmost
    // row already sits ~190 px below the HUD block, guarded below anyway) until both probes are
    // in-image and clear of every drift-inflated planet disc. The avatar dot sits at the projected
    // session origin — the disc centre, ~160 px from any rim probe — and cannot collide.
    let (inner_probe, outer_probe) = {
        let mut found = None;
        let mut step = 0u32;
        while step < 72 {
            let theta = (90.0 + f64::from(step) * RIM_ANGLE_STEP_DEG).to_radians();
            let inner = rim_probe(cx, cy, theta, rim_r - RIM_INNER_INSET_PX, RIM_PROBE_HALF_PX);
            let outer = rim_probe(
                cx,
                cy,
                theta,
                rim_r + RIM_OUTER_OUTSET_PX,
                RIM_PROBE_HALF_PX,
            );
            let in_image = |r: &ScreenAabb| {
                r.min.x >= 0.0 && r.min.y > 130.0 && r.max.x <= w as f64 && r.max.y <= h as f64
            };
            if in_image(&inner)
                && in_image(&outer)
                && planet_rects
                    .iter()
                    .all(|(_, p)| rects_disjoint(inner, *p) && rects_disjoint(outer, *p))
            {
                found = Some((inner, outer));
                break;
            }
            step += 1;
        }
        found.unwrap_or_else(|| {
            panic!(
                "no rim angle clears every planet's drift-inflated disc ({planet_rects:?}) — THE \
                 world's projected layout changed; restate the probe search, never the rim asserts"
            )
        })
    };
    let rim_inside = nonclear_in_region(&buf, w, h, clear, inner_probe);
    let rim_outside = nonclear_in_region(&buf, w, h, clear, outer_probe);
    println!(
        "G-RENDER-BOXES-SMOKE rim: inside probe [{:.1},{:.1}]-[{:.1},{:.1}] → {rim_inside} px, \
         outside probe [{:.1},{:.1}]-[{:.1},{:.1}] → {rim_outside} px",
        inner_probe.min.x,
        inner_probe.min.y,
        inner_probe.max.x,
        inner_probe.max.y,
        outer_probe.min.x,
        outer_probe.min.y,
        outer_probe.max.x,
        outer_probe.max.y,
    );
    assert!(
        rim_inside > 0,
        "H2 (shell-isolated): the home SHELL is not pixel-visible at its own rim — the probe just \
         inside the silhouette (clear of every planet disc) holds no paint, so whatever filled the \
         whole-region count was NOT the shell",
    );
    assert_eq!(
        rim_outside, 0,
        "H2 (shell-isolated): paint OUTSIDE the home shell's silhouette where nothing may draw \
         (the capture scaffold is despawned; the galaxy is never drawn) — the shell's edge is not \
         where the camera math says it is",
    );

    // The HR6 manifest aligns the capture to its tick + state — confirm it recorded this screenshot.
    let run_dir = png.parent().and_then(Path::parent).expect("run dir");
    let manifest_json =
        std::fs::read_to_string(run_dir.join(MANIFEST_FILENAME)).expect("read manifest.json");
    let manifest = RunManifest::from_json(&manifest_json).expect("parse manifest");
    assert!(
        manifest
            .captures
            .iter()
            .any(|c| c.kind == CaptureKind::Screenshot && rel_path.ends_with(&c.path)),
        "manifest must record the screenshot capture, got {:?}",
        manifest.captures,
    );
}
