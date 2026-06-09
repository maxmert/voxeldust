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
}
