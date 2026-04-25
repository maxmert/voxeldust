//! Shared camera-rotation math used by both server (per-tick player
//! rotation broadcast, shard handoff spawn poses) and client
//! (`consume_spawn_pose_latch` / `update_camera_rotation`).
//!
//! Centralising the math guarantees that server-computed spawn
//! poses round-trip through client read-back without coordinate
//! convention drift — there is ONE canonical `quat_from_yaw_pitch`
//! in the codebase.

use glam::{DQuat, DVec3};

/// Camera forward vector (world-space) for the given yaw / pitch.
///
/// Convention: yaw is rotation about the world +Y axis; pitch is
/// elevation. At `(yaw, pitch) = (0, 0)` the camera looks along +X.
/// Pitch range: [-π/2, π/2]. Yaw wraps.
///
/// ```text
///   fwd.x = cos(yaw) · cos(pitch)
///   fwd.y = sin(pitch)
///   fwd.z = sin(yaw) · cos(pitch)
/// ```
pub fn forward_from_yaw_pitch(yaw: f64, pitch: f64) -> DVec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    DVec3::new(cy * cp, sp, sy * cp)
}

/// Quaternion that orients a camera (default forward = -Z, up = +Y)
/// to look along `forward_from_yaw_pitch(yaw, pitch)`. This is the
/// rotation the client composes into `Transform.rotation` on its
/// `MainCamera`; server-side functions that pack a
/// `handoff::SpawnPose.rotation` call this to produce matching
/// values.
///
/// Impl: build an orthonormal basis with `forward = -Z_local =
/// world_forward`, `up = +Y_world`, then convert the 3×3 rotation
/// to quaternion. Matches `Transform::looking_to(fwd, +Y)` in
/// Bevy's convention.
pub fn quat_from_yaw_pitch(yaw: f64, pitch: f64) -> DQuat {
    let forward = forward_from_yaw_pitch(yaw, pitch);
    // Degenerate guard: looking straight up / down pins forward to
    // ±Y, collinear with the up vector. Fall back to a tiny nudge
    // so `cross(forward, up)` is non-zero.
    let up = if forward.y.abs() > 0.9999 {
        DVec3::Z
    } else {
        DVec3::Y
    };
    // Bevy's `looking_to` convention: local -Z aligns with `forward`,
    // local +Y aligns with `up`. The right axis (local +X) is
    // `forward × up` (right-hand rule: looking along +X with +Y up
    // puts +Z to the right). `up × forward` would give -right and
    // also invert local +Y via the y_local cross product, rendering
    // the camera upside-down — that was a real bug that put the
    // player on the ceiling.
    let z_local = -forward.normalize();
    let x_local = forward.cross(up).normalize();
    let y_local = z_local.cross(x_local);
    DQuat::from_mat3(&glam::DMat3::from_cols(x_local, y_local, z_local))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_at_origin_is_plus_x() {
        let fwd = forward_from_yaw_pitch(0.0, 0.0);
        assert!((fwd - DVec3::X).length() < 1e-6);
    }

    #[test]
    fn forward_at_yaw_90_is_plus_z() {
        let fwd = forward_from_yaw_pitch(std::f64::consts::FRAC_PI_2, 0.0);
        assert!((fwd - DVec3::Z).length() < 1e-6);
    }

    #[test]
    fn camera_up_is_world_up_at_horizontal() {
        // At any yaw with pitch=0, the camera's local +Y must map to
        // world +Y — otherwise the view is upside-down ("walking on
        // the ceiling"). Regression guard for the cross-product sign
        // bug that inverted y_local.
        for yaw in [0.0, 0.5, 1.0, std::f64::consts::PI, -1.7] {
            let q = quat_from_yaw_pitch(yaw, 0.0);
            let world_up = q * DVec3::Y;
            assert!(
                (world_up - DVec3::Y).length() < 1e-6,
                "yaw={yaw} expected up=+Y got {:?}",
                world_up,
            );
        }
    }

    #[test]
    fn quat_round_trips_forward() {
        let cases = [
            (0.0, 0.0),
            (std::f64::consts::FRAC_PI_2, 0.0),
            (-std::f64::consts::FRAC_PI_4, 0.3),
            (1.5, -0.2),
        ];
        for (yaw, pitch) in cases {
            let q = quat_from_yaw_pitch(yaw, pitch);
            let fwd_expected = forward_from_yaw_pitch(yaw, pitch);
            // Camera forward = rotation * (-Z).
            let fwd_actual = q * DVec3::NEG_Z;
            assert!(
                (fwd_actual - fwd_expected).length() < 1e-5,
                "yaw={} pitch={} expected={:?} got={:?}",
                yaw, pitch, fwd_expected, fwd_actual,
            );
        }
    }
}
