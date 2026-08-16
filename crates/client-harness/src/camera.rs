//! The follow-own-dot first-person camera math (pure). Parameterized by an `up`-vector
//! — world-Y for P1.5, planet-radial / ship-local later — the ONE seam that must not
//! need reshaping (slice_3 plan §4). Mouse-look is a LOCAL view-only response: this
//! turns the camera immediately; the rendered ENTITY orientation stays server-delivered
//! (NOT prediction).

use glam::DVec3;
use vd_core::kinematics;

/// Max pitch (radians, ~89°) — just under straight-up to avoid the gimbal flip.
pub const PITCH_LIMIT: f64 = 1.553_343;

/// First-person eye height above the entity origin (m), along `up`.
pub const DEFAULT_EYE_OFFSET: f64 = 1.6;

/// A follow camera's orientation state (the entity position is supplied per frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FollowCamera {
    /// The up-vector the camera is aligned to (world-Y now; planet-radial later).
    pub up: DVec3,
    /// Heading around `up` (radians).
    pub yaw: f64,
    /// Tilt toward `up` (radians, clamped to ±[`PITCH_LIMIT`]).
    pub pitch: f64,
    /// First-person eye offset along `up`.
    pub eye_offset: f64,
}

impl FollowCamera {
    #[must_use]
    pub fn new(up: DVec3) -> FollowCamera {
        FollowCamera {
            up,
            yaw: 0.0,
            pitch: 0.0,
            eye_offset: DEFAULT_EYE_OFFSET,
        }
    }

    /// Accumulate a look delta — yaw is free, pitch clamps to ±[`PITCH_LIMIT`].
    pub fn apply_look(&mut self, delta_yaw: f64, delta_pitch: f64) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// The world-space look direction from `(yaw, pitch)` around `up`. A PURE CONSUMER
    /// of the one up-relative convention ([`vd_core::kinematics::forward_in_frame`]) —
    /// NOT a second basis — so it matches the stub server exactly (forward is `-Z` at
    /// rest for `up = +Y`; `+pitch` tilts toward `up`) and stays correct for the
    /// planet-radial / ship-local `up` the end goal needs, with no drift to maintain.
    #[must_use]
    pub fn forward(&self) -> DVec3 {
        kinematics::forward_in_frame(self.up, self.yaw, self.pitch)
    }

    /// The first-person eye position for an entity at `own_pos`.
    #[must_use]
    pub fn eye(&self, own_pos: DVec3) -> DVec3 {
        own_pos + self.up.normalize() * self.eye_offset
    }
}

// ---------------------------------------------------------------------------
// World→screen projection (Slice V1): the ONE legitimate screen-space projection.
// The transfer MEMBERSHIP verdict stays world-space (deterministic state artifact);
// this pinhole projection exists ONLY for the PIXEL-corroboration verdict
// (`dot_pixels_within_box_region`) — to compute where a world point lands on the
// readback so the dot's pixels can be checked against the box's projected screen box.
// Pure + deterministic (no GPU): a right-handed look-at view + a symmetric-perspective
// divide + a viewport map. Tier-A, fully coverable on synthetic cameras.
// ---------------------------------------------------------------------------

use glam::DMat4;

/// A pixel position in the readback image (origin top-left, `+y` DOWN — image-buffer
/// convention, matching `assert.rs`'s `(ry * width + rx)` indexing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenPos {
    pub x: f64,
    pub y: f64,
}

/// A pixel-space axis-aligned rectangle in the readback image, as inclusive `min`/`max`
/// corners (origin top-left). The projected screen extent of a world region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenAabb {
    pub min: ScreenPos,
    pub max: ScreenPos,
}

impl ScreenAabb {
    /// Whether `p` lies within the rectangle (inclusive on every edge).
    #[must_use]
    pub fn contains(&self, p: ScreenPos) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
}

/// A pinhole capture camera: a look-at view + a symmetric vertical-FOV perspective, sized to a
/// pixel viewport. Deterministic and GPU-free — the Tier-A projection the pixel-corroboration
/// verdict uses to place a world point on the readback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptureCamera {
    pub eye: DVec3,
    pub target: DVec3,
    pub up: DVec3,
    /// Vertical field of view in radians.
    pub fov_y: f64,
    /// Viewport width in pixels.
    pub width: usize,
    /// Viewport height in pixels.
    pub height: usize,
}

impl CaptureCamera {
    /// Project a world point to a pixel position, or `None` if it is behind the camera or the
    /// viewport is degenerate (zero-size). Right-handed look-at + symmetric perspective divide +
    /// a top-left-origin viewport map (NDC `+y` up flips to image `+y` down).
    #[must_use]
    pub fn project_point(&self, world_p: DVec3) -> Option<ScreenPos> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let aspect = self.width as f64 / self.height as f64;
        let view = DMat4::look_at_rh(self.eye, self.target, self.up);
        // A far/near pair wide enough for any capture; the divide only needs a positive w (= -z_view).
        let proj = DMat4::perspective_rh(self.fov_y, aspect, NEAR_PLANE, FAR_PLANE);
        let clip = proj * view * world_p.extend(1.0);
        // w = -z_view; a point on or behind the eye plane has w <= 0 and cannot be projected.
        if clip.w <= 0.0 {
            return None;
        }
        let ndc_x = clip.x / clip.w; // [-1, 1] left→right
        let ndc_y = clip.y / clip.w; // [-1, 1] bottom→top
        // Viewport map: NDC→pixels, flipping y so +ndc_y (up) → smaller pixel row (top).
        let px = (ndc_x * 0.5 + 0.5) * self.width as f64;
        let py = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height as f64;
        Some(ScreenPos { x: px, y: py })
    }
}

/// The perspective near plane (m) — small; only the `w`-sign gate and the divide matter for a
/// pixel map, so the exact value is not load-bearing (a named const, not a magic number).
pub const NEAR_PLANE: f64 = 0.1;
/// The perspective far plane (m) — generous so any capture-scale point projects.
pub const FAR_PLANE: f64 = 1.0e12;

/// The minimum apparent radius, in pixels, a DOT MARKER may shrink to under perspective — the VU
/// marker-phase visibility floor (a dot is the avatar's marker until meshes land, and a marker that
/// falls below a pixel is not a marker). Load-bearing for the pixel gates too (batch review): the
/// 0.5 m dot at a scene-fitted camera's ~780 m eye subtends ~0.6 px — whether it covered ANY sample
/// was luck — so no pixel verdict could be made dot-sensitive until the marker's footprint had a
/// floor. THE one constant both the renderer (the marker scale) and the gates (the projected rect)
/// derive from, via [`marker_world_radius`] — two derivations of it is how they would drift.
pub const DOT_MIN_APPARENT_RADIUS_PX: f64 = 3.0;

/// The dot marker's WORLD radius after the minimum-apparent-size floor: `base_radius_m`, or the
/// world size of [`DOT_MIN_APPARENT_RADIUS_PX`] at view distance `dist_m` under a symmetric vertical
/// `fov_y` over `viewport_h_px` rows — whichever is larger. Pure, branchless (HR5), shared by the
/// Tier-B renderer (which scales the marker mesh by it) and the pixel gates (which size the dot's
/// projected rectangle from it), so the drawn footprint and the asserted rectangle cannot disagree.
/// `dist_m` is the euclidean eye→marker distance on BOTH sides (a slight over-estimate of view
/// depth off-axis, absorbed by the callers' bracketing factor).
#[must_use]
pub fn marker_world_radius(base_radius_m: f64, dist_m: f64, fov_y: f64, viewport_h_px: f64) -> f64 {
    let m_per_px = 2.0 * dist_m.max(0.0) * (fov_y * 0.5).tan() / viewport_h_px;
    base_radius_m.max(DOT_MIN_APPARENT_RADIUS_PX * m_per_px)
}

// ---------------------------------------------------------------------------
// Scene framing (Slice V3): fit a CaptureCamera to the whole realm-box scene so
// every box is visible in the readback — Tier-A math the Tier-B renderer only APPLIES.
// ---------------------------------------------------------------------------

use vd_client::realm_scene::{BoxShape, RealmScene};

/// The default capture vertical FOV (radians, 45°) — the framing distance is derived from this,
/// so a box scene fits the same frustum the render camera uses. A named const (no magic number).
pub const FIT_FOV_Y: f64 = std::f64::consts::FRAC_PI_4;

/// Headroom multiplier on the fitted distance so no box edge kisses the frame border (a small,
/// deterministic margin — 15% padding around the union bounds).
pub const FIT_MARGIN: f64 = 1.15;

/// The fixed view direction the fitted camera looks ALONG (from eye toward the scene center):
/// a gentle downward-and-forward 3/4 view so nested boxes read as volumes, not flat quads. Unit,
/// not axis-parallel to world-Y (so `look_at_rh`'s up basis is well-defined). Named, deterministic.
const FIT_VIEW_DIR: DVec3 = DVec3::new(0.3, -0.35, -0.887);

/// THE PILOT'S OWN CAPTURE CAMERA (window lane Slice D — `docs/design/window_lane.md` §4): the
/// headless capture rendered from the AVATAR'S EYE, along the AVATAR'S DELIVERED FACING, instead
/// of the scene-fitting diagnostic framing.
///
/// WHY IT EXISTS. [`fit_camera_to_scene`] frames the union of everything drawn, so on a flight
/// BETWEEN two star systems — whose centres are fixed points of the world's ring — the fitted
/// frustum barely moves and a system you are flying toward cannot grow on screen. The warp
/// acceptance is a statement about what the PILOT sees, so the agent's eyes must look through the
/// pilot's eyes. The scene-fitting camera stays the default (every existing box gate frames its
/// whole scene by it); this is opted into per run.
///
/// It reads only DELIVERED state — the own entity's position and its SERVER-AUTHORED orientation
/// (`orient · -Z` is the one facing convention, shared with `vd_client_harness::nav::look_at` and
/// the stub server's `orient_from_yaw_pitch`) — so a `LookAt` injected through dev-control turns
/// the avatar AND the eyes together, and a pixel gate reconstructs the very camera the renderer
/// used from the same `DevState` poll it reads the positions from. No prediction, no local view
/// state: unlike the windowed [`FollowCamera`] (whose mouse-look is a local, view-only response),
/// a headless run has no mouse, so the delivered orientation is the only honest facing.
#[must_use]
pub fn pilot_capture_camera(
    own_pos: DVec3,
    own_orient: glam::DQuat,
    width: usize,
    height: usize,
) -> CaptureCamera {
    let eye = own_pos + DVec3::Y * DEFAULT_EYE_OFFSET;
    CaptureCamera {
        eye,
        target: eye + own_orient * DVec3::NEG_Z,
        up: own_orient * DVec3::Y,
        fov_y: FIT_FOV_Y,
        width,
        height,
    }
}

/// Fit a [`CaptureCamera`] of pixel size `width`×`height` to the WHOLE scene: frame the union of
/// every box's center ± its extent so all boxes land in the readback. Returns `None` for an empty
/// scene (nothing to frame) or a degenerate viewport (zero-size). Pure + deterministic — the box
/// render camera the pixel proof uses. A straight-line delegate: [`scene_bounds`] →
/// [`fit_camera_to_bounds`] (both guards live in the delegates; nothing branches here beyond the
/// bounds `?`).
#[must_use]
pub fn fit_camera_to_scene(
    scene: &RealmScene,
    width: usize,
    height: usize,
) -> Option<CaptureCamera> {
    let (center, radius) = scene_bounds(scene)?;
    fit_camera_to_bounds(center, radius, width, height)
}

/// Fit a [`CaptureCamera`] of pixel size `width`×`height` to a bounding SPHERE (`center`,
/// `radius`) — the extracted framing core `fit_camera_to_scene` delegates to. Returns `None` for a
/// degenerate viewport (zero-size). The eye sits back along [`FIT_VIEW_DIR`] far enough that the
/// sphere fits [`FIT_FOV_Y`] (× [`FIT_MARGIN`]), looking at `center`. EXPOSED so a process gate can
/// reconstruct the client's ACTUAL live camera from the client's own reported drawn boxes (their
/// live union bounds), instead of fitting its own camera over a static file — the two cameras drift
/// the moment a drawn box moves (an orbiting planet), and rectangles projected through the wrong
/// camera prove nothing.
#[must_use]
pub fn fit_camera_to_bounds(
    center: DVec3,
    radius: f64,
    width: usize,
    height: usize,
) -> Option<CaptureCamera> {
    if width == 0 || height == 0 {
        return None;
    }
    // Distance so the bounding sphere of `radius` subtends at most the vertical FOV: the half-angle
    // is fov_y/2, so `sin(half) = radius / dist` ⇒ `dist = radius / sin(half)`, padded by the
    // margin and floored so a single tiny (or zero-radius) box is still viewed from a sane range.
    let half_fov = FIT_FOV_Y * 0.5;
    let fit_dist = (radius / half_fov.sin()) * FIT_MARGIN;
    let dist = fit_dist.max(radius + NEAR_PLANE * 2.0);
    let dir = FIT_VIEW_DIR.normalize();
    let eye = center - dir * dist; // back off ALONG the view dir so the camera looks toward center
    Some(CaptureCamera {
        eye,
        target: center,
        up: DVec3::Y,
        fov_y: FIT_FOV_Y,
        width,
        height,
    })
}

/// The union bounding sphere over `(center, per-axis half-extent)` items: its center (the midpoint
/// of the union AABB) and radius (half the AABB diagonal, so the whole union fits). `None` for an
/// empty iterator — the empty guard lives HERE (moved from the old `scene_bounds` body, not
/// duplicated). EXPOSED beside [`fit_camera_to_bounds`] so a gate can union the client's reported
/// live box centres (zipped with their world extents) exactly as the renderer's per-frame refit
/// unions its drawn scene.
#[must_use]
pub fn bounds_union(items: impl Iterator<Item = (DVec3, DVec3)>) -> Option<(DVec3, f64)> {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    let mut any = false;
    for (c, extent) in items {
        any = true;
        min = min.min(c - extent);
        max = max.max(c + extent);
    }
    if !any {
        return None;
    }
    let center = (min + max) * 0.5;
    let radius = (max - min).length() * 0.5;
    Some((center, radius))
}

/// The union bounding sphere of every box in the scene — [`bounds_union`] over the boxes' DRAWN
/// centres ± extents. A straight-line delegate: the renderer draws in reduced space, so bounding
/// raw absolutes would aim the capture camera somewhere nothing is; the per-box extent branch is
/// [`box_extent`]'s.
fn scene_bounds(scene: &RealmScene) -> Option<(DVec3, f64)> {
    bounds_union(
        scene
            .iter()
            .map(|(_realm, rbox)| (rbox.draw_center(), box_extent(rbox.shape))),
    )
}

/// A box's per-axis half-extent: a sphere is `r` on every axis; a box is its `half`. A monomorphic
/// helper (the shape KIND branch is here, not in the framing body).
fn box_extent(shape: BoxShape) -> DVec3 {
    match shape {
        BoxShape::Sphere { r } => DVec3::splat(r),
        BoxShape::Box { half } => half,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-9
    }

    #[test]
    fn the_marker_radius_floors_at_the_apparent_size_and_never_shrinks_the_base() {
        // NEAR: the base radius already subtends more than the floor — returned unchanged. At
        // fov 90° over 100 rows, 1 px ≙ 2·d·tan(45°)/100 = d/50 m; at d = 10 m the 3 px floor is
        // 0.6 m, under the 1 m base.
        let fov = std::f64::consts::FRAC_PI_2;
        assert_eq!(marker_world_radius(1.0, 10.0, fov, 100.0), 1.0);
        // FAR: the floor wins — exactly DOT_MIN_APPARENT_RADIUS_PX pixels' worth of world at depth
        // (~60 m at d = 1000 m over 100 rows), stated by the same expression so no tan() ulp can
        // split the two sides.
        assert_eq!(
            marker_world_radius(1.0, 1000.0, fov, 100.0),
            DOT_MIN_APPARENT_RADIUS_PX * (2.0 * 1000.0 * (fov * 0.5).tan() / 100.0)
        );
        // A degenerate (behind-the-eye) distance clamps to zero depth ⇒ the base radius stands.
        assert_eq!(marker_world_radius(0.5, -5.0, fov, 100.0), 0.5);
    }

    #[test]
    fn the_pilot_capture_camera_sits_at_the_eye_and_looks_along_the_delivered_facing() {
        use glam::DQuat;
        // AT REST the delivered facing is the shared convention's `-Z`, the eye is one offset above
        // the avatar along world-up, and the framing FOV is the one both cameras declare.
        let pos = DVec3::new(10.0, 0.0, -5.0);
        let cam = pilot_capture_camera(pos, DQuat::IDENTITY, 1284, 720);
        assert_eq!(cam.eye, pos + DVec3::Y * DEFAULT_EYE_OFFSET);
        assert_eq!(cam.target, cam.eye + DVec3::NEG_Z);
        assert_eq!(cam.up, DVec3::Y);
        assert_eq!(cam.fov_y, FIT_FOV_Y);
        assert_eq!((cam.width, cam.height), (1284, 720));
        // TURNED (yaw +90° about world-up) the eye is unmoved and the facing follows the DELIVERED
        // orientation — which is what makes an injected `LookAt` turn the avatar and the agent's
        // eyes together, with no local view state in between.
        let yawed = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
        let turned = pilot_capture_camera(pos, yawed, 1284, 720);
        assert_eq!(turned.eye, cam.eye);
        assert!((turned.target - (cam.eye + DVec3::NEG_X)).length() < 1e-12);
        assert!((turned.up - DVec3::Y).length() < 1e-12);
        // A point straight ahead of the turned camera projects at the frame centre — the gate
        // reconstructs exactly the camera the renderer used, so its rectangles land on the pixels.
        let ahead = turned
            .project_point(cam.eye + DVec3::NEG_X * 100.0)
            .expect("in front");
        assert!((ahead.x - 642.0).abs() < 1e-6, "centred in x: {ahead:?}");
        assert!((ahead.y - 360.0).abs() < 1e-6, "centred in y: {ahead:?}");
    }

    #[test]
    fn default_forward_is_minus_z_and_eye_offsets_along_up() {
        let cam = FollowCamera::new(DVec3::Y);
        // up=Y (not parallel to Z) -> rest forward = -Z (matches the server convention).
        let fwd = cam.forward();
        assert!(close(fwd, DVec3::NEG_Z), "got {fwd:?}");
        assert!(close(cam.eye(DVec3::ZERO), DVec3::Y * DEFAULT_EYE_OFFSET));
    }

    #[test]
    fn yaw_rotates_forward_around_up() {
        let mut cam = FollowCamera::new(DVec3::Y);
        cam.apply_look(std::f64::consts::FRAC_PI_2, 0.0); // +90° yaw
        // rest forward -Z, yawed +90° about +Y -> -X.
        let fwd = cam.forward();
        assert!(close(fwd, DVec3::NEG_X), "got {fwd:?}");
    }

    #[test]
    fn pitch_tilts_toward_up_and_clamps() {
        let mut cam = FollowCamera::new(DVec3::Y);
        cam.apply_look(0.0, 0.5);
        assert!(cam.forward().y > 0.0, "pitched up");
        // Beyond the limit clamps (does not exceed PITCH_LIMIT).
        cam.apply_look(0.0, 100.0);
        assert!((cam.pitch - PITCH_LIMIT).abs() < 1e-9);
        cam.apply_look(0.0, -1000.0);
        assert!((cam.pitch + PITCH_LIMIT).abs() < 1e-9);
    }

    #[test]
    fn forward_matches_the_shared_convention_for_any_up() {
        // PARITY: the camera's forward MUST equal the ONE shared up-relative convention
        // (vd_core::kinematics::forward_in_frame) for EVERY up — world-Y now,
        // planet-radial / ship-local later. The camera is a pure consumer, so a drift in
        // either trips this red (closes the second-source hazard, incl. the non-Y case).
        use vd_core::kinematics::forward_in_frame;
        for up in [DVec3::Y, DVec3::Z, DVec3::new(1.0, 2.0, 3.0).normalize()] {
            for yi in -3..=3 {
                for pi in -3..=3 {
                    let yaw = f64::from(yi) * 0.4;
                    let pitch = f64::from(pi) * 0.4; // |pitch| <= 1.2 < PITCH_LIMIT
                    let cam = FollowCamera {
                        up,
                        yaw,
                        pitch,
                        eye_offset: 0.0,
                    };
                    let got = cam.forward();
                    let want = forward_in_frame(up, yaw, pitch);
                    assert!(
                        close(got, want),
                        "up {up:?} yaw {yaw} pitch {pitch}: {got:?} vs {want:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rest_forward_for_up_z_is_plus_y_an_independent_non_world_up_oracle() {
        // INDEPENDENT value oracle for the non-Y path, hand-derived (NOT via
        // forward_in_frame, so it is not tautological): frame_from_up(Z) is the minimal
        // rotation Y→Z (90° about +X), so rest forward = that · (-Z) = +Y. This is the
        // real cross-check now that `forward_matches_the_shared_convention_for_any_up` is
        // tautological-by-construction (a guard against re-introducing a second basis).
        let fwd = FollowCamera::new(DVec3::Z).forward(); // pre-computed (a call in a msg is uncovered)
        assert!(
            close(fwd, DVec3::Y),
            "rest forward for up=Z is +Y, got {fwd:?}"
        );
    }

    // ---- CaptureCamera::project_point + ScreenAabb ------------------------------

    /// A camera looking down -Z at the origin, 64×48 viewport, 90° vertical FOV.
    fn cam() -> CaptureCamera {
        CaptureCamera {
            eye: DVec3::new(0.0, 0.0, 10.0),
            target: DVec3::ZERO,
            up: DVec3::Y,
            fov_y: std::f64::consts::FRAC_PI_2,
            width: 64,
            height: 48,
        }
    }

    #[test]
    fn a_point_on_the_view_axis_projects_to_the_viewport_center() {
        let c = cam();
        let p = c.project_point(DVec3::ZERO).expect("origin is in front");
        assert!((p.x - 32.0).abs() < 1e-6, "centered x, got {}", p.x);
        assert!((p.y - 24.0).abs() < 1e-6, "centered y, got {}", p.y);
    }

    #[test]
    fn up_in_world_maps_to_a_smaller_pixel_row_top_left_origin() {
        // A point above the axis (+y world) must land in the UPPER half of the image (py < center):
        // the y-flip in the viewport map (image origin top-left).
        let c = cam();
        let above = c
            .project_point(DVec3::new(0.0, 2.0, 0.0))
            .expect("in front");
        assert!(
            above.y < 24.0,
            "above-center world → upper image row, got {}",
            above.y
        );
        let right = c
            .project_point(DVec3::new(2.0, 0.0, 0.0))
            .expect("in front");
        assert!(
            right.x > 32.0,
            "+x world → right image column, got {}",
            right.x
        );
    }

    #[test]
    fn a_point_behind_the_camera_does_not_project() {
        // Behind the eye (further +z than the eye) → w <= 0 → None (the behind-camera gate).
        let c = cam();
        assert_eq!(c.project_point(DVec3::new(0.0, 0.0, 20.0)), None);
    }

    #[test]
    fn a_degenerate_viewport_does_not_project() {
        let mut c = cam();
        c.width = 0;
        assert_eq!(c.project_point(DVec3::ZERO), None);
        let mut c = cam();
        c.height = 0;
        assert_eq!(c.project_point(DVec3::ZERO), None);
    }

    #[test]
    fn screen_aabb_contains_is_inclusive_on_every_edge() {
        let r = ScreenAabb {
            min: ScreenPos { x: 10.0, y: 20.0 },
            max: ScreenPos { x: 30.0, y: 40.0 },
        };
        assert!(r.contains(ScreenPos { x: 20.0, y: 30.0 })); // interior
        assert!(r.contains(ScreenPos { x: 10.0, y: 20.0 })); // min corner (inclusive)
        assert!(r.contains(ScreenPos { x: 30.0, y: 40.0 })); // max corner (inclusive)
        assert!(!r.contains(ScreenPos { x: 9.9, y: 30.0 })); // left of min x
        assert!(!r.contains(ScreenPos { x: 30.1, y: 30.0 })); // right of max x
        assert!(!r.contains(ScreenPos { x: 20.0, y: 19.9 })); // above min y
        assert!(!r.contains(ScreenPos { x: 20.0, y: 40.1 })); // below max y
    }

    // ---- fit_camera_to_scene (Slice V3 framing) --------------------------------

    use vd_client::realm_scene::RealmScene;
    use vd_core::geometry::Boundary;
    use vd_core::pose::RealmId;
    use vd_wire::channels::SceneRow;

    /// A `Shell` boundary at `center`, radius `r`.
    fn shell_b(realm: RealmId, center: DVec3, r: f64) -> SceneRow {
        SceneRow {
            realm,
            parent: None,
            pose: vd_core::pose::StampedPose::at_rest(
                vd_core::pose::FrameRef::SystemSpace { system_seed: 0 },
                center,
                vd_core::UniverseTick(1),
            ),
            bag: vd_core::look::look_bag(&Boundary::Shell { r }),
        }
    }

    /// An `Aabb` boundary at `center`, half-extents `half`.
    fn aabb_b(realm: RealmId, center: DVec3, half: DVec3) -> SceneRow {
        SceneRow {
            realm,
            parent: None,
            pose: vd_core::pose::StampedPose::at_rest(
                vd_core::pose::FrameRef::SystemSpace { system_seed: 0 },
                center,
                vd_core::UniverseTick(1),
            ),
            bag: vd_core::look::look_bag(&Boundary::Aabb { half }),
        }
    }

    #[test]
    fn an_empty_scene_frames_to_no_camera() {
        // Nothing to frame ⇒ None (never a nonsense camera at the origin).
        assert_eq!(fit_camera_to_scene(&RealmScene::default(), 64, 48), None);
    }

    #[test]
    fn a_degenerate_viewport_frames_to_no_camera() {
        let scene = RealmScene::from_scene_rows(&[shell_b(RealmId::System(1), DVec3::ZERO, 10.0)])
            .expect("scene");
        assert_eq!(fit_camera_to_scene(&scene, 0, 48), None);
        assert_eq!(fit_camera_to_scene(&scene, 64, 0), None);
    }

    #[test]
    fn the_fitted_camera_frames_every_box_center_and_extent_in_the_viewport() {
        // Two well-separated boxes (a sphere and a box); the fitted camera must place BOTH
        // entirely on-screen — every corner of each box's extent projects inside [0,w]×[0,h].
        let (w, h) = (128usize, 96usize);
        let scene = RealmScene::from_scene_rows(&[
            shell_b(RealmId::System(1), DVec3::new(-200.0, 0.0, 0.0), 60.0),
            aabb_b(
                RealmId::Station(2),
                DVec3::new(200.0, 40.0, 0.0),
                DVec3::new(50.0, 50.0, 50.0),
            ),
        ])
        .expect("scene");
        let cam = fit_camera_to_scene(&scene, w, h).expect("frames");
        assert_eq!(cam.fov_y, FIT_FOV_Y);
        assert_eq!((cam.width, cam.height), (w, h));
        // Every box corner projects, and lands inside the viewport (the framing succeeded).
        for (_realm, rbox) in scene.iter() {
            let ext = match rbox.shape {
                vd_client::realm_scene::BoxShape::Sphere { r } => DVec3::splat(r),
                vd_client::realm_scene::BoxShape::Box { half } => half,
            };
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        let corner = rbox.draw_center() + ext * DVec3::new(sx, sy, sz);
                        let p = cam
                            .project_point(corner)
                            .expect("corner projects (in front)");
                        // Range-`contains` (not `a && b`) so the assert carries no short-circuit
                        // branch of its own (HR5: `assert!(a && b)` leaves an uncoverable arm).
                        assert!(
                            (0.0..=w as f64).contains(&p.x),
                            "corner x {} off-screen (w={w})",
                            p.x
                        );
                        assert!(
                            (0.0..=h as f64).contains(&p.y),
                            "corner y {} off-screen (h={h})",
                            p.y
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_single_tiny_box_uses_the_distance_floor_and_still_frames() {
        // A zero-radius (degenerate) box: `radius == 0` would give a zero fit distance, so the
        // `.max(radius + NEAR_PLANE*2)` FLOOR must kick in — exercises that branch. The box
        // center still projects near the viewport center.
        let (w, h) = (64usize, 64usize);
        let scene = RealmScene::from_scene_rows(&[shell_b(
            RealmId::System(1),
            DVec3::new(5.0, 6.0, 7.0),
            0.0,
        )])
        .expect("scene");
        let cam = fit_camera_to_scene(&scene, w, h).expect("frames");
        // The eye is the floor distance back along the view dir (not AT the center → finite).
        assert!(cam.eye.is_finite());
        assert_ne!(cam.eye, cam.target, "the floor kept the eye off the target");
        let p = cam
            .project_point(DVec3::new(5.0, 6.0, 7.0))
            .expect("center projects");
        // The center of the framed box lands at the viewport center (within a pixel).
        assert!((p.x - w as f64 / 2.0).abs() < 1.0, "center x {}", p.x);
        assert!((p.y - h as f64 / 2.0).abs() < 1.0, "center y {}", p.y);
    }

    #[test]
    fn scene_bounds_covers_both_the_sphere_and_box_extent_arms() {
        // Directly exercise box_extent's two arms via a mixed scene: the union AABB must reach
        // the sphere's far edge (r) AND the box's far corner (half).
        let scene = RealmScene::from_scene_rows(&[
            shell_b(RealmId::System(1), DVec3::new(-100.0, 0.0, 0.0), 10.0),
            aabb_b(
                RealmId::Station(2),
                DVec3::new(100.0, 0.0, 0.0),
                DVec3::new(20.0, 5.0, 5.0),
            ),
        ])
        .expect("scene");
        let (center, radius) = scene_bounds(&scene).expect("bounds");
        // Union spans x from -110 (sphere far edge) to +120 (box far corner) ⇒ center x = 5.
        assert!((center.x - 5.0).abs() < 1e-9, "center x {}", center.x);
        // Radius is half the diagonal of the union AABB (positive, finite).
        assert!(radius > 0.0);
        assert!(radius.is_finite());
        // The empty-scene None arm.
        assert_eq!(scene_bounds(&RealmScene::default()), None);
    }

    // ---- the extracted framing core (fit_camera_to_bounds + bounds_union), directly ------------

    #[test]
    fn fit_camera_to_bounds_frames_the_sphere_and_refuses_a_degenerate_viewport() {
        // DIRECT (not via the scene delegate): the sphere's center lands at the viewport center and
        // its ±radius extremes project on-screen — the fit distance actually fits the sphere.
        let (w, h) = (128usize, 96usize);
        let center = DVec3::new(10.0, -20.0, 30.0);
        let radius = 75.0;
        let cam = fit_camera_to_bounds(center, radius, w, h).expect("frames");
        assert_eq!(cam.target, center);
        assert_eq!(cam.fov_y, FIT_FOV_Y);
        assert_eq!((cam.width, cam.height), (w, h));
        let p = cam.project_point(center).expect("center projects");
        assert!((p.x - w as f64 / 2.0).abs() < 1e-6, "center x {}", p.x);
        assert!((p.y - h as f64 / 2.0).abs() < 1e-6, "center y {}", p.y);
        for offset in [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            -DVec3::X,
            -DVec3::Y,
            -DVec3::Z,
        ] {
            let q = cam
                .project_point(center + offset * radius)
                .expect("extreme projects (in front)");
            assert!((0.0..=w as f64).contains(&q.x), "extreme x {} (w={w})", q.x);
            assert!((0.0..=h as f64).contains(&q.y), "extreme y {} (h={h})", q.y);
        }
        // The MOVED zero-dimension guard (it lives here now, not in the scene delegate).
        assert_eq!(fit_camera_to_bounds(center, radius, 0, h), None);
        assert_eq!(fit_camera_to_bounds(center, radius, w, 0), None);
    }

    #[test]
    fn bounds_union_spans_every_item_and_an_empty_iterator_is_none() {
        // DIRECT (a Vec iterator — its own monomorphization, both arms): two items whose union AABB
        // spans x from -110 (sphere-like splat extent) to +120 (box far corner) ⇒ center x = 5, and
        // the radius is half the union diagonal.
        let items = vec![
            (DVec3::new(-100.0, 0.0, 0.0), DVec3::splat(10.0)),
            (DVec3::new(100.0, 0.0, 0.0), DVec3::new(20.0, 5.0, 5.0)),
        ];
        let (center, radius) = bounds_union(items.into_iter()).expect("bounds");
        assert!((center.x - 5.0).abs() < 1e-9, "center x {}", center.x);
        let expected_radius = DVec3::new(230.0, 20.0, 20.0).length() * 0.5;
        assert!(
            (radius - expected_radius).abs() < 1e-9,
            "radius {radius} vs {expected_radius}"
        );
        // The MOVED empty guard (it lives here now, not in the scene walk).
        assert_eq!(bounds_union(Vec::new().into_iter()), None);
    }
}
