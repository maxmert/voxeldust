//! The follow-own-dot first-person camera math (pure). Parameterized by an `up`-vector
//! — world-Y for P1.5, planet-radial / ship-local later — the ONE seam that must not
//! need reshaping (slice_3 plan §4). Mouse-look is a LOCAL view-only response: this
//! turns the camera immediately; the rendered ENTITY orientation stays server-delivered
//! (NOT prediction).

use glam::DVec3;
use vd_core::kinematics;
use vd_core::pose::{LatticePos, Tier};

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
    /// The camera's ORTHONORMAL BASIS `(right, up, back)` — the same basis
    /// `glam::DMat4::look_at_rh` builds and the same one [`look_rotation`] hands the renderer, so
    /// the pixel model and the drawn frame cannot be built from two conventions.
    #[must_use]
    pub fn basis(&self) -> (DVec3, DVec3, DVec3) {
        look_basis(self.target - self.eye, self.up)
    }

    /// Project a world point to a pixel position, or `None` if it is behind the camera or the
    /// viewport is degenerate (zero-size).
    ///
    /// ★ S5 — CAMERA-RELATIVE (`docs/design/DEFERRED.md` D-LOOK-3). The point is expressed
    /// RELATIVE TO THE EYE first and only then rotated into view axes, which is exactly what the
    /// renderer now does, so the model and the pixels share one arithmetic shape. It also removes
    /// the near/far pair from the pixel map entirely: a symmetric perspective's `x`, `y` and `w`
    /// rows do not contain `near` or `far` at all (only the unused `z` row does), so the old
    /// `perspective_rh(NEAR_PLANE, FAR_PLANE)` was two literals that could never change a pixel —
    /// and a far plane of `1e12` m was a visible lie on a world whose star gap is `2.2e15` m.
    #[must_use]
    pub fn project_point(&self, world_p: DVec3) -> Option<ScreenPos> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let aspect = self.width as f64 / self.height as f64;
        let (right, up, back) = self.basis();
        let rel = world_p - self.eye;
        // View-space coordinates of the eye-relative point; depth `w = -z_view` (a right-handed
        // view looks down -Z), which is what the perspective divide needs.
        let w = -back.dot(rel);
        // A point on or behind the eye plane cannot be projected.
        if w <= 0.0 {
            return None;
        }
        let focal = 1.0 / (self.fov_y * 0.5).tan();
        let ndc_x = (focal / aspect) * right.dot(rel) / w; // [-1, 1] left→right
        let ndc_y = focal * up.dot(rel) / w; // [-1, 1] bottom→top
        // Viewport map: NDC→pixels, flipping y so +ndc_y (up) → smaller pixel row (top).
        let px = (ndc_x * 0.5 + 0.5) * self.width as f64;
        let py = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height as f64;
        Some(ScreenPos { x: px, y: py })
    }
}

/// The framing floor's near-plane reference (m) — the scene-fit standoff is never allowed closer
/// than the bounding radius plus this, so a zero-radius (single point) scene is still viewed from a
/// sane range. NOT a projection parameter: [`CaptureCamera::project_point`] contains no near plane
/// (see its note), and the RENDERER derives its own planes per frame from what is drawn
/// ([`depth_planes`]).
pub const NEAR_PLANE: f64 = 0.1;

/// The camera basis `(right, up, back)` for a look `direction` and an `up` reference — the ONE
/// expression both the pixel model ([`CaptureCamera::basis`]) and the renderer
/// ([`look_rotation`]) build their frame from. Matches `glam::DMat4::look_at_rh` exactly:
/// `back = -normalize(direction)`, `right = normalize(up × back)`, `up' = back × right`.
///
/// Branchless (HR5): the caller guarantees a non-degenerate direction and a non-parallel `up` —
/// every caller in the tree builds `direction` from a normalized quaternion or a fitted standoff.
#[must_use]
pub fn look_basis(direction: DVec3, up: DVec3) -> (DVec3, DVec3, DVec3) {
    let back = -direction.normalize();
    let right = up.cross(back).normalize();
    (right, back.cross(right), back)
}

/// ★ THE RENDER FRAME'S ROTATION (S5, D-LOOK-3). The camera orientation as a QUATERNION, built in
/// `f64` from the look direction and the up reference.
///
/// WHY IT EXISTS — a MEASURED defect, not a preference. The renderer used to orient its camera with
/// Bevy's `Transform::looking_at(target, up)`, which subtracts `translation` from `target` in `f32`.
/// On THE world the eye stands `1e8`–`1e15` m from the composed origin, where one `f32` ulp is
/// `16` m to `1.3e8` m — so a target one metre ahead of the eye rounds to the eye ITSELF, the
/// direction comes out ZERO, and Bevy's `look_to` silently falls back to `Dir3::NEG_Z`. The camera
/// then looked down world `-Z` no matter where the pilot was facing, which is precisely why every
/// parked pixel gate found a correctly-composed subject's rectangle EMPTY in the readback.
/// Building the rotation here, in `f64`, from the direction the caller already holds, makes that
/// cancellation unrepresentable.
#[must_use]
pub fn look_rotation(direction: DVec3, up: DVec3) -> glam::DQuat {
    let (right, up_o, back) = look_basis(direction, up);
    glam::DQuat::from_mat3(&glam::DMat3::from_cols(right, up_o, back))
}

/// ★ THE CAMERA-RELATIVE FLATTEN (S5, D-LOOK-3): a world position expressed in the RENDER FRAME,
/// whose origin is the eye. The subtraction happens in `f64` and only the RESULT is ever narrowed
/// to `f32`, so the drawn error is relative to the DISTANCE to the thing (≈ `6e-8` of it — far
/// under a pixel at any range) instead of relative to the absolute coordinate (which is what put a
/// `1.3e8` m quantum on a `2.2e15` m warp destination).
#[must_use]
pub fn eye_relative(world_p: DVec3, eye: DVec3) -> DVec3 {
    world_p - eye
}

/// ★ THE SAME SUBTRACTION, DONE BEFORE ANYTHING IS FLATTENED (slice S4).
///
/// [`eye_relative`] above is correct about the SUBTRACTION and wrong about the ORDER: by the time it
/// runs, both halves have already been flattened from the frame origin, and each of those flattens
/// rounded independently.
///
/// At the star placement radius a position is about 1.53e18 lattice cells, whose f64 spacing is 256
/// cells — a quarter of a metre. So two ships flying in convoy each round to a quarter of a metre,
/// separately, TODAY. Their drawn separation is wrong by up to half a metre and it flickers as they
/// move, because the two roundings are independent.
///
/// ★ AND THE OBVIOUS TEST DOES NOT CATCH IT. A hundred metres is exactly 102,400 cells, and 102,400 is
/// a whole multiple of 256 — so both endpoints round by the SAME amount, the errors cancel, and the
/// drawn distance is exactly 100.000 m. Every whole-metre separation is exact here. The defect only
/// shows on separations that are not multiples of the rounding step
/// (`vd_core::pose::tests::subtracting_before_flattening_is_exact_where_flattening_first_is_not`).
///
/// Subtracting on the integers first removes it entirely — not reduces it, removes it.
#[must_use]
pub fn eye_relative_lattice(world_p: LatticePos, eye: LatticePos, tier: Tier) -> DVec3 {
    world_p.delta_m(eye, tier)
}

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
    base_radius_m.max(DOT_MIN_APPARENT_RADIUS_PX * one_pixel_world_m(dist_m, fov_y, viewport_h_px))
}

/// The WORLD SIZE OF ONE PIXEL at view distance `dist_m` under a symmetric vertical `fov_y` over
/// `viewport_h_px` rows — the camera model's own resolution limit, and THE one derivation both the
/// apparent-size floor ([`marker_world_radius`]) and the near plane ([`depth_planes`]) are built
/// from. Negative distances clamp to zero (a subject at or behind the eye has no forward extent).
#[must_use]
pub fn one_pixel_world_m(dist_m: f64, fov_y: f64, viewport_h_px: f64) -> f64 {
    2.0 * dist_m.max(0.0) * (fov_y * 0.5).tan() / viewport_h_px
}

/// The perspective near/far pair the drawn picture needs, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthPlanes {
    /// The near plane: nothing the picture can RESOLVE is closer, so nothing resolvable is clipped.
    pub near: f64,
    /// The far plane: the farthest drawn SURFACE. Under the reverse-Z infinite projection the
    /// renderer uses, this clips nothing — it is the honest declaration of the picture's own reach
    /// (and what the CPU-side frustum reports), never a guessed backstop.
    pub far: f64,
}

/// ★ THE DERIVED DEPTH PLANES (S5, D-LOOK-3) — near/far read off WHAT IS ACTUALLY DRAWN, never
/// declared as lengths.
///
/// `subjects` is one `(distance from the eye to the thing's centre, its drawn world radius)` pair
/// per drawn thing — the renderer feeds it straight from the transforms it just wrote, so the
/// planes describe this frame's picture and no other.
///
/// THE DERIVATION.
/// * `near` = the world size of ONE PIXEL at the NEAREST DRAWN SURFACE. Anything thinner than a
///   pixel at that range cannot be resolved by this camera at all, so a near plane there cannot
///   clip anything the picture could have shown. The reference distance never falls below
///   [`DEFAULT_EYE_OFFSET`] (arm's length — the avatar's own marker is always drawn there, and a
///   realm shell the eye is passing THROUGH drives the nearest surface to zero every crossing).
/// * `far` = the FARTHEST DRAWN SURFACE, floored at that same reference so `near < far` holds for
///   an empty picture too (`near` is one pixel's worth of the reference distance, and a viewport
///   with more rows than `2·tan(fov/2)` — every viewport — makes that strictly smaller than the
///   reference itself).
///
/// WHY NO LOGARITHMIC DEPTH. Measured (`crates/bins/tests/render_scale.rs`): under the reverse-Z
/// infinite projection Bevy builds for a `PerspectiveProjection`, the depth code is `near/z`, so
/// ONE `f32` code ulp is a CONSTANT `1.1920929e-7` of the range at every distance — 21.8 m at
/// 1.82e8 m and 2.68e8 m at 2.25e15 m, both orders under the thinnest thing the world draws there.
/// Reverse-Z already spans the world's full dynamic range; a second depth scheme would buy nothing.
#[must_use]
pub fn depth_planes(subjects: &[(f64, f64)], fov_y: f64, viewport_h_px: f64) -> DepthPlanes {
    let mut nearest_surface = f64::INFINITY;
    let mut farthest_surface = 0.0_f64;
    for &(dist_m, radius_m) in subjects {
        nearest_surface = nearest_surface.min((dist_m - radius_m).max(0.0));
        farthest_surface = farthest_surface.max(dist_m + radius_m);
    }
    // An EMPTY picture has no nearest surface at all — arm's length is then the whole reference.
    if !nearest_surface.is_finite() {
        nearest_surface = DEFAULT_EYE_OFFSET;
    }
    let reference = nearest_surface.max(DEFAULT_EYE_OFFSET);
    DepthPlanes {
        near: one_pixel_world_m(reference, fov_y, viewport_h_px),
        far: farthest_surface.max(reference),
    }
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

/// ★ THE FRAMING BOUNDS (S5, D-LOOK-3) — WHAT A DIAGNOSTIC CAPTURE MAY FRAME.
///
/// `items` is one `(drawn centre, per-axis half-extent, states its own outline)` triple per drawn
/// thing. The union is taken over the SELF-AUTHORED OUTLINES only; if the picture holds no outline
/// at all, it falls back to the union of every drawn centre — the pre-S5 behaviour, which is
/// exactly right for a sky made of nothing but points of light.
///
/// WHY THE FILTER EXISTS — a MEASURED failure. A point of light states no outline, so there is
/// nothing OF IT to frame: it is drawn at a floored apparent size wherever it happens to be. Once
/// the world grew to true scale, a galaxy-standing observer's picture held sibling stars
/// `2.25e15` m out; unioning their bare CENTRES pushed the fitted eye to `~6.4e15` m and collapsed
/// every in-system silhouette to a degenerate point (`render_crossing_smoke` measured the home
/// shell at `4e-11` px). Framing what actually has an extent is the whole cure, and it needs no
/// realm kind, no subject list and no literal: the author of the pixels decides.
#[must_use]
pub fn framing_bounds(items: &[(DVec3, DVec3, bool)]) -> Option<(DVec3, f64)> {
    bounds_union(
        items
            .iter()
            .filter(|(_, _, outlined)| *outlined)
            .map(|&(c, e, _)| (c, e)),
    )
    .or_else(|| bounds_union(items.iter().map(|&(c, e, _)| (c, e))))
}

/// The union bounding sphere of the scene's framable content — [`framing_bounds`] over the boxes'
/// DRAWN centres ± extents, tagged by whether the box states its own outline. A straight-line
/// delegate: the renderer draws in reduced space, so bounding raw absolutes would aim the capture
/// camera somewhere nothing is; the per-box extent branch is [`box_extent`]'s.
fn scene_bounds(scene: &RealmScene) -> Option<(DVec3, f64)> {
    let items: Vec<(DVec3, DVec3, bool)> = scene
        .iter()
        .map(|(_realm, rbox)| {
            (
                rbox.draw_center(),
                box_extent(rbox.shape),
                rbox.body == vd_client::realm_scene::BodyKind::Look,
            )
        })
        .collect();
    framing_bounds(&items)
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
    // The third-party reference the camera basis is measured against (the projection itself no
    // longer builds a matrix — see `project_point`).
    use glam::DMat4;

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

    // ---------------------------------------------------------------------------
    // ★ S5 — THE RENDER FRAME (D-LOOK-3). The three laws the camera-relative flatten
    // rests on, each measured rather than argued.
    // ---------------------------------------------------------------------------

    #[test]
    fn look_rotation_reproduces_the_look_at_basis_exactly() {
        // The renderer builds its camera rotation with `look_rotation`; the pixel model builds its
        // basis with the same `look_basis`; `glam::DMat4::look_at_rh` is the third-party reference
        // both must agree with, or a projected rectangle and its pixels describe two frames.
        let eye = DVec3::new(3.0, -4.0, 11.0);
        let target = DVec3::new(-7.0, 2.0, -1.0);
        let up = DVec3::new(0.1, 1.0, 0.2).normalize();
        let (right, up_o, back) = look_basis(target - eye, up);
        let view = DMat4::look_at_rh(eye, target, up);
        // look_at_rh's rows ARE the basis vectors (it maps world → view).
        let row0 = DVec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let row1 = DVec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let row2 = DVec3::new(view.x_axis.z, view.y_axis.z, view.z_axis.z);
        assert!(close(right, row0), "right {right} vs {row0}");
        assert!(close(up_o, row1), "up {up_o} vs {row1}");
        assert!(close(back, row2), "back {back} vs {row2}");
        // And the quaternion carries the same basis in its columns.
        let q = look_rotation(target - eye, up);
        assert!(close(q * DVec3::X, right), "quat right");
        assert!(close(q * DVec3::Y, up_o), "quat up");
        assert!(close(q * DVec3::Z, back), "quat back");
    }

    #[test]
    fn the_render_frame_survives_the_world_scale_that_kills_an_f32_camera() {
        // ★ THE MEASURED DEFECT (D-LOOK-3), pinned so it cannot come back. Bevy's
        // `Transform::looking_at` subtracts `target - translation` in f32 and falls back to facing
        // world -Z when that comes out zero. At THE world's eye magnitudes a unit-ahead target IS
        // the eye in f32 — so the camera silently stopped facing where the pilot faced.
        // A FIXED LADDER, deliberately: these are not readings of today's world (the derived mass
        // cap moved the star gap to 1.4989795871538758e15 m on 2026-08-20 and re-drew every star),
        // they are three magnitudes spanning the range this crate must survive. Pinning them keeps
        // the f32 defect measured at the SAME points across every world re-solve; the gates that
        // must read the live world do so (`render_scale`).
        for magnitude in [1.8248e8_f64, 3.2e11, 2.2485e15] {
            let axis = DVec3::splat(magnitude / 3.0_f64.sqrt());
            let forward = DVec3::new(0.5, 0.5, -0.5).normalize();
            let f32_direction = (axis + forward).as_vec3() - axis.as_vec3();
            assert_eq!(
                f32_direction,
                glam::Vec3::ZERO,
                "at |eye| = {magnitude:e} m an f32 look-at direction must be measured as LOST \
                 (this is the defect the render frame removes)",
            );
            // The f64 basis is unharmed at the same magnitude...
            let q = look_rotation(forward, DVec3::Y);
            assert!(
                close(q * DVec3::Z, -forward),
                "the f64 basis still faces the mark"
            );
            // ...and the eye-relative flatten places a subject one metre ahead of the eye at one
            // metre ahead of the RENDER ORIGIN, to within the f64 quantum of the absolute
            // coordinate — which is MILLIONTHS of the world size of one pixel at that same range,
            // i.e. unobservable in the readback. (The f32 path's quantum, measured above, is the
            // whole marker.)
            let placed = eye_relative(axis + forward, axis);
            let residual = (placed - forward).length();
            let pixel = one_pixel_world_m(magnitude, FIT_FOV_Y, 720.0);
            assert!(
                residual * 1.0e6 < pixel,
                "at |eye| = {magnitude:e} m the flatten residual {residual:e} m must be under a \
                 millionth of one pixel's {pixel:e} m of world",
            );
        }
    }

    #[test]
    fn the_depth_planes_are_read_off_the_drawn_picture_and_never_clip_it() {
        let fov = FIT_FOV_Y;
        let rows = 720.0;
        // A WORLD-SCALE RANGE PAIR: the avatar's own marker at arm's length and a body 2.2485e15 m
        // out with a 7.8e12 m drawn radius (a warp destination's floored point of light). The far
        // magnitude is a FIXED ruler, not a reading — THE world's star gap became
        // 1.4989795871538758e15 m with the derived mass cap, and this test would say nothing new if
        // it tracked it.
        let subjects = [
            (DEFAULT_EYE_OFFSET, f64::from(0.5_f32)),
            (2.2485e15, 7.76e12),
        ];
        let planes = depth_planes(&subjects, fov, rows);
        // NEAR: one pixel's worth of world at the nearest surface (arm's length wins here), so
        // nothing the camera can resolve is clipped — and strictly closer than that surface.
        let nearest_surface = DEFAULT_EYE_OFFSET - 0.5;
        assert!(
            planes.near < nearest_surface,
            "near {} must sit inside the nearest surface {nearest_surface}",
            planes.near
        );
        assert_eq!(
            planes.near,
            one_pixel_world_m(DEFAULT_EYE_OFFSET, fov, rows)
        );
        // FAR: the farthest drawn SURFACE, exactly — no margin, no guess.
        assert_eq!(planes.far, 2.2485e15 + 7.76e12);
        // A FAR-ONLY picture moves the near plane out with it (the reference is the scene's own
        // nearest surface, not a constant): one pixel at 1.0e11 m is 1.15e8 m of world.
        let far_only = depth_planes(&[(1.0e11, 0.0)], fov, rows);
        assert_eq!(far_only.near, one_pixel_world_m(1.0e11, fov, rows));
        assert_eq!(far_only.far, 1.0e11);
        // AN EMPTY PICTURE still yields a usable, ordered pair (arm's length is the whole
        // reference) — the branch a capture takes before the first row arrives.
        let empty = depth_planes(&[], fov, rows);
        assert_eq!(empty.near, one_pixel_world_m(DEFAULT_EYE_OFFSET, fov, rows));
        assert_eq!(empty.far, DEFAULT_EYE_OFFSET);
        assert!(
            empty.near < empty.far,
            "near {} far {}",
            empty.near,
            empty.far
        );
        // A subject the eye is INSIDE drives the nearest surface to zero (a realm shell you are
        // crossing) — the reference floor is what keeps the plane positive.
        let inside = depth_planes(&[(1.0e6, 2.0e6)], fov, rows);
        assert_eq!(
            inside.near,
            one_pixel_world_m(DEFAULT_EYE_OFFSET, fov, rows)
        );
        assert_eq!(inside.far, 3.0e6);
        // ONE PIXEL AT OR BEHIND THE EYE has no forward extent (the clamp arm).
        assert_eq!(one_pixel_world_m(-5.0, fov, rows), 0.0);
    }

    #[test]
    fn the_framing_bounds_frame_outlines_and_fall_back_to_points_when_there_are_none() {
        // ★ THE MEASURED FAILURE (D-LOOK-3): unioning a sibling star's bare CENTRE 2.25e15 m out
        // with the home shell pushed the fitted eye past 6e15 m and collapsed the home silhouette.
        // A point of light states no outline, so there is nothing of it to frame.
        let home = (DVec3::ZERO, DVec3::splat(1.58e11), true);
        let sibling = (DVec3::new(2.2485e15, 0.0, 0.0), DVec3::ZERO, false);
        let (center, radius) = framing_bounds(&[home, sibling]).expect("bounds");
        assert_eq!(center, DVec3::ZERO);
        assert_eq!(radius, DVec3::splat(2.0 * 1.58e11).length() * 0.5);
        // WITHOUT the filter the same picture frames a 1.1e15 m sphere — the collapse, measured.
        let (_, unfiltered) =
            bounds_union([(home.0, home.1), (sibling.0, sibling.1)].into_iter()).expect("bounds");
        assert!(
            unfiltered > radius * 1000.0,
            "the unfiltered union {unfiltered:e} must dwarf the framed {radius:e}"
        );
        // A SKY OF PURE POINTS still frames (the fallback arm) — the pre-S5 behaviour, which is
        // exactly right when nothing states an outline.
        let (c, r) = framing_bounds(&[sibling]).expect("point fallback");
        assert_eq!(c, sibling.0);
        assert_eq!(r, 0.0);
        // NOTHING DRAWN AT ALL is still None (the empty guard, through both arms).
        assert_eq!(framing_bounds(&[]), None);
    }
}

#[cfg(test)]
mod eye_relative_on_the_lattice {
    //! SLICE S4: reduce against the eye BEFORE flattening, so two things drawn near each other are
    //! drawn at the distance they actually are apart.
    use super::*;
    use glam::I64Vec3;

    /// The star placement radius, in fine lattice cells — where one f64 step is 256 cells, a quarter
    /// of a metre.
    const R_CELLS: i64 = 1_534_955_097_245_569_024;

    #[test]
    fn a_convoy_at_the_placement_radius_draws_the_distance_it_actually_is_apart() {
        // TWO SHIPS FLYING TOGETHER, today, on the world as it stands. Each position is flattened from
        // the frame origin before the eye is subtracted, so each rounds independently by up to a
        // quarter of a metre — and the drawn gap between them is wrong, and flickers as they move.
        //
        // The separation swept here is NOT a multiple of the rounding step, deliberately: a whole
        // metre is exactly 102,400 cells and 102,400 is a multiple of 256, so both endpoints round the
        // SAME way and the errors cancel. A gate built on a round number could never have failed.
        let tier = Tier::Fine;
        let eye = LatticePos::at(I64Vec3::new(R_CELLS, 0, 0), DVec3::ZERO);

        let mut worst_old = 0.0_f64;
        let mut worst_new = 0.0_f64;
        for sep_cells in [102_401_i64, 102_501, 102_655, 103_000] {
            let ship = LatticePos::at(I64Vec3::new(R_CELLS + sep_cells, 0, 0), DVec3::ZERO);
            let truth = sep_cells as f64 * tier.cell_edge_m();

            // THE OLD PATH: flatten both from the frame origin, then subtract.
            let old = eye_relative(
                ship.delta_m(LatticePos::ORIGIN, tier),
                eye.delta_m(LatticePos::ORIGIN, tier),
            )
            .x;
            // THE NEW PATH: subtract on the integers, flatten the small result.
            let new = eye_relative_lattice(ship, eye, tier).x;

            worst_old = worst_old.max((old - truth).abs());
            worst_new = worst_new.max((new - truth).abs());
        }

        assert!(
            worst_old > 0.0,
            "the old path must be measurably wrong here, or this gate proves nothing"
        );
        assert!(
            worst_old <= 0.25,
            "bounded by one step of the absolute coordinate: {worst_old} m"
        );
        assert_eq!(
            worst_new, 0.0,
            "reducing on the lattice must be EXACT, not merely closer"
        );
    }

    #[test]
    fn the_two_paths_agree_where_the_old_one_was_already_exact() {
        // Not a regression test — a statement of scope. Near the frame origin, and on separations that
        // are multiples of the rounding step, the old path was already right, and the new one must not
        // move those answers.
        let tier = Tier::Fine;
        let eye = LatticePos::from_metres(DVec3::new(10.0, 0.0, 0.0), tier);
        let ship = LatticePos::from_metres(DVec3::new(110.0, 0.0, 0.0), tier);
        assert_eq!(
            eye_relative_lattice(ship, eye, tier),
            eye_relative(
                ship.delta_m(LatticePos::ORIGIN, tier),
                eye.delta_m(LatticePos::ORIGIN, tier)
            )
        );
    }
}
