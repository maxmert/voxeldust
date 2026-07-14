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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-9
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
}
