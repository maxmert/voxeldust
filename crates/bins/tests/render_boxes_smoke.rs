//! G-RENDER-BOXES-SMOKE — the Visual Crossing Playground pixel proof (Slice V3). A NEW sibling to
//! `render_smoke.rs` (additive, a new consumer — the existing gate is untouched): it brings the
//! local-process QUIC cluster up, launches a HEADLESS `client --capture` with a boot-loaded
//! `boxes.json` (ONE translucent colored realm box), drives it to a real wgpu-readback screenshot,
//! decodes the PNG, and asserts the BOX is PIXEL-VISIBLE **and correctly placed** — not a bare
//! content fraction (adversary H2). The box center + extent are projected to a screen AABB via the
//! SAME [`fit_camera_to_scene`] camera the offscreen render framed the scene with (the ONE
//! legitimate screen-space step), and the box's color region must hold non-clear pixels INSIDE that
//! projected AABB. Zero magenta.
//!
//! WHY the box pins to its screen region: the capture app, seeing a loaded scene, points its
//! offscreen camera at `fit_camera_to_scene(scene)`, so the box lands where this test projects it.
//! A regression that drew SOMETHING elsewhere (or lost the box but kept the reference scene) fails
//! the region check even though a bare content-fraction would pass — that is the H2 fix.
//!
//! GPU PRECONDITION (same as G-RENDER-SMOKE): this renders through wgpu and REQUIRES a working GPU
//! adapter (the dev Apple-Silicon Metal GPU today). It is a LOCAL gate — no CI, no software-raster
//! fallback (that would poison the visual baseline). On a GPU-less host the capture client cannot
//! start; the test then fails with a message naming the precondition.
//!
//! Run via `just render-boxes-smoke` (it builds the client with `--features dev-control,render`).
//! Under a plain `cargo test --workspace` (no features) this file compiles to ZERO tests.
#![cfg(all(feature = "dev-control", feature = "render"))]

use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use vd_bins::{
    DevClusterDown, dev_auth_signing_key_hex, dev_roundtrip, devcluster, loopback,
    record_extra_pid, slot_trust_dir, slot_workdir,
};
use vd_client::realm_scene::{BoxShape, RealmScene};
use vd_client_harness::assert::{MAGENTA, magenta_pixel_count};
use vd_client_harness::camera::{ScreenAabb, fit_camera_to_scene};
use vd_client_harness::manifest::{CaptureKind, MANIFEST_FILENAME, RunManifest};
use vd_client_harness::verdict::projected_point_aabb;
use vd_client_render::{CAPTURE_H, CAPTURE_W};
use vd_core::geometry::{CrossEffect, RealmBoundary};
use vd_core::glam::{DVec3, I64Vec3};
use vd_core::pose::{LatticePos, RealmId};
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

/// The ONE realm box the scene carries: a Station AABB placed FAR from the origin (so the reference
/// ground/pillars around the origin fall off-screen once the camera frames the box), with a large
/// half-extent so it fills the framed view. The IDENTICAL Vec is written to `boxes.json` for the
/// client AND rebuilt in-test for the projection — single-sourced, so the pixels and the projected
/// AABB can never disagree.
const BOX_CENTER: DVec3 = DVec3::new(500.0, 0.0, 0.0);
const BOX_HALF: DVec3 = DVec3::new(60.0, 60.0, 60.0);
const BOX_REALM: RealmId = RealmId::Station(4242);

fn one_box_boundaries() -> Vec<RealmBoundary> {
    vec![
        RealmBoundary::aabb(
            BOX_REALM,
            LatticePos::local(BOX_CENTER),
            BOX_HALF,
            1.15,
            1.30,
            0.0,
            0.05,
            0.5,
            1.0,
            None,
            BOX_REALM,
            CrossEffect::Authority,
        )
        .expect("valid aabb boundary"),
    ]
}

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

/// The box's projected screen rectangle via the SAME `fit_camera_to_scene` camera the offscreen
/// render used — the box center + its bounding-sphere extent (the AABB corner distance) projected
/// through the Tier-A `CaptureCamera`. This is where the box's pixels MUST land (H2).
fn box_screen_aabb(scene: &RealmScene, origin: LatticePos) -> ScreenAabb {
    let camera = fit_camera_to_scene(scene, origin, CAPTURE_W as usize, CAPTURE_H as usize)
        .expect("the one-box scene frames to a camera");
    let rbox = scene.get(BOX_REALM).expect("box in scene");
    // The bounding-sphere radius of the box: the AABB corner distance from the center.
    let radius = match rbox.shape {
        BoxShape::Box { half } => half.length(),
        BoxShape::Sphere { r } => r,
    };
    // The box's centre reduced against the client's LIVE render origin — the same space the pixels
    // were drawn in. Projecting the ABSOLUTE centre would only agree while the origin is zero.
    projected_point_aabb(&camera, rbox.draw_center(origin), radius)
        .expect("the box center projects in front of the fitted camera")
}

/// The client's LIVE render origin: the space every position it reports, and every pixel it drew,
/// is expressed in. Read from the client rather than assumed to be zero, so this gate keeps
/// checking the same thing once real galactic coordinates make the origin non-zero.
fn live_render_origin(port: u16) -> LatticePos {
    let o = dev_roundtrip_state(port).render_origin;
    LatticePos::at(
        I64Vec3::new(o.cell[0], o.cell[1], o.cell[2]),
        DVec3::new(o.offset[0], o.offset[1], o.offset[2]),
    )
}

/// The current delivered state (a `State` round-trip).
fn dev_roundtrip_state(port: u16) -> vd_devproto::DevState {
    match round_trip(port, &DevRequest::State) {
        DevResponse::State { state } => state,
        other => panic!("expected DevState, got {other:?}"),
    }
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
fn g_render_boxes_smoke_shows_a_translucent_box_pixel_visible_in_its_screen_region() {
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
    // A contained cwd so the client's `runs/` (and our boxes.json) land here (reaped on `down`).
    let cwd = slot_workdir(RENDER_BOXES_SLOT).join("capture-cwd");
    std::fs::create_dir_all(&cwd).expect("make client cwd");

    // Write boxes.json — the IDENTICAL Vec we project in-test (single-sourced). A JSON array of
    // RealmBoundary, exactly what the shard plants; the client loads it via --realm-boxes.
    let boundaries = one_box_boundaries();
    // The load path must project this to a scene (fail LOUD here if the fixture is bad).
    let scene = RealmScene::from_boundaries(&boundaries).expect("the fixture projects to a scene");
    let boxes_json_path = cwd.join("boxes.json");
    std::fs::write(
        &boxes_json_path,
        serde_json::to_string(&boundaries).expect("serialize boxes.json"),
    )
    .expect("write boxes.json");

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
            boxes_json_path.to_str().expect("utf8 boxes.json path"),
        ]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = ChildGuard(cmd.spawn().expect("spawn capture client"));
    record_extra_pid(RENDER_BOXES_SLOT, child.0.id()).expect("record capture-client pid");

    await_listener(devctl, &mut child.0);

    // Drive to a live, DELIVERED frame, then capture (the box is boot-config, present from frame 0).
    let live = round_trip(
        devctl,
        &DevRequest::WaitUntil {
            predicate: WaitPredicate {
                field: WaitField::UniverseTick,
                op: WaitOp::Ge,
                value: 1,
            },
            max_ticks: CAPTURE_WAIT_TICKS,
        },
    );
    assert!(
        matches!(live, DevResponse::State { .. }),
        "client should reach a delivered frame, got {live:?}"
    );

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

    // Decode the captured PNG → RGBA8 (the only decode in the gate; the assertion is Tier-A).
    let png = cwd.join(&rel_path);
    let img = image::open(&png)
        .unwrap_or_else(|e| panic!("open captured PNG {}: {e}", png.display()))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let buf = img.into_raw();

    // Self-calibrate the clear color from the top-right corner (guaranteed background — the box is
    // framed centered, and the camera looks straight at it). A blank frame still fails: every pixel
    // equals the corner ⇒ zero non-clear anywhere.
    let corner = (w - 1) * 4;
    let clear = [
        buf[corner],
        buf[corner + 1],
        buf[corner + 2],
        buf[corner + 3],
    ];

    // (1) Zero magenta — nothing failed to draw with a real material.
    let magenta = magenta_pixel_count(&buf);
    // (2) The H2 box-pixel assertion: the box's color region is NON-EMPTY inside its projected
    // screen AABB (the SAME fit_camera_to_scene camera the render used). This pins "the box drew
    // WHERE it should", not a bare content fraction.
    let region = box_screen_aabb(&scene, live_render_origin(devctl));
    let box_pixels = nonclear_in_region(&buf, w, h, clear, region);

    println!(
        "G-RENDER-BOXES-SMOKE: {w}x{h} frame at tick {tick:?}, clear {clear:?}, \
         box screen AABB [{:.1},{:.1}]-[{:.1},{:.1}] → magenta {magenta}, box_region_pixels {box_pixels}",
        region.min.x, region.min.y, region.max.x, region.max.y,
    );
    assert_eq!(
        magenta, 0,
        "the frame must be magenta-free (no missing-material draws), got {magenta} (sentinel {MAGENTA:?})"
    );
    assert!(
        box_pixels > 0,
        "the translucent box is NOT pixel-visible inside its projected screen region \
         [{:.1},{:.1}]-[{:.1},{:.1}] — a bare content fraction would miss this (H2). \
         Did the box mesh/material draw, and did fit_camera_to_scene frame it?",
        region.min.x,
        region.min.y,
        region.max.x,
        region.max.y,
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
