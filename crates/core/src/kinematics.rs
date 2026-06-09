//! The input → pose CONVENTION, defined ONCE and shared by the input PRODUCER (the
//! client's `nav`/`camera`) and the input CONSUMER (the stub sim now; real physics
//! later). It is the single source of: the movement-axis map (`InputDatagram.movement`
//! = `[forward, strafe, vertical]`, forward is local `-Z`), the yaw/pitch → orient
//! convention (`from_euler(YXZ)`), and their inverses.
//!
//! Before this module the convention was hand-re-encoded in `stub::integrate`,
//! `client-harness::nav`, and `client-harness::camera` — a silent drift hazard (a change
//! in one would mis-aim the others with no compile error). Now there is one definition;
//! the sim applies it (`orient · axes · speed·dt`) and the client inverts it. Pure +
//! branchless (HR5: all branching, if any, stays in monomorphic callers).

use glam::{DQuat, DVec3, EulerRot};

/// Orientation for a `(yaw, pitch)`: `from_euler(YXZ, yaw, pitch, 0)`. Yaw is around
/// `+Y`; forward is `orient · -Z`.
#[must_use]
pub fn orient_from_yaw_pitch(yaw: f64, pitch: f64) -> DQuat {
    DQuat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)
}

/// The world forward direction for a `(yaw, pitch)`: `(-cosP·sinY, sinP, -cosP·cosY)`.
#[must_use]
pub fn forward_from_yaw_pitch(yaw: f64, pitch: f64) -> DVec3 {
    orient_from_yaw_pitch(yaw, pitch) * DVec3::NEG_Z
}

/// Recover `(yaw, pitch)` from a forward direction — the inverse of
/// [`forward_from_yaw_pitch`] (pitch clamped to a valid `asin` domain).
#[must_use]
pub fn forward_to_yaw_pitch(forward: DVec3) -> (f64, f64) {
    let pitch = forward.y.clamp(-1.0, 1.0).asin();
    let yaw = (-forward.x).atan2(-forward.z);
    (yaw, pitch)
}

/// The LOCAL-frame motion axes for an input `movement = [forward, strafe, vertical]`
/// (each clamped to `[-1, 1]`): `(strafe, vertical, -forward)`. The sim applies
/// `orient · these · speed·dt`.
#[must_use]
pub fn local_axes_from_movement(movement: [f32; 3]) -> DVec3 {
    DVec3::new(
        f64::from(movement[1].clamp(-1.0, 1.0)),
        f64::from(movement[2].clamp(-1.0, 1.0)),
        -f64::from(movement[0].clamp(-1.0, 1.0)),
    )
}

/// The input `movement = [forward, strafe, vertical]` that yields a desired LOCAL
/// motion direction — the inverse of [`local_axes_from_movement`]'s map, clamped to
/// `[-1, 1]` (used by the walk-to controller to aim a step).
#[must_use]
pub fn movement_from_local(local: DVec3) -> [f32; 3] {
    [
        clamp_unit(-local.z),
        clamp_unit(local.x),
        clamp_unit(local.y),
    ]
}

fn clamp_unit(v: f64) -> f32 {
    v.clamp(-1.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_forward_is_minus_z() {
        assert!(forward_from_yaw_pitch(0.0, 0.0).abs_diff_eq(DVec3::NEG_Z, 1e-12));
    }

    #[test]
    fn forward_yaw_pitch_round_trips_over_a_grid() {
        // Skip the ±90° gimbal poles (yaw is ill-defined there).
        for yi in -3..=3 {
            for pi in -3..=3 {
                let yaw = f64::from(yi) * 0.4;
                let pitch = f64::from(pi) * 0.4; // |pitch| <= 1.2 < π/2
                let (ry, rp) = forward_to_yaw_pitch(forward_from_yaw_pitch(yaw, pitch));
                assert!((ry - yaw).abs() < 1e-9, "yaw {yaw} -> {ry}");
                assert!((rp - pitch).abs() < 1e-9, "pitch {pitch} -> {rp}");
            }
        }
    }

    #[test]
    fn movement_axis_map_and_inverse_round_trip() {
        // forward -> local -Z; strafe -> local +X; vertical -> local +Y.
        assert_eq!(local_axes_from_movement([1.0, 0.0, 0.0]), DVec3::NEG_Z);
        assert_eq!(local_axes_from_movement([0.0, 1.0, 0.0]), DVec3::X);
        assert_eq!(local_axes_from_movement([0.0, 0.0, 1.0]), DVec3::Y);
        // inverse recovers the movement triple.
        assert_eq!(movement_from_local(DVec3::NEG_Z), [1.0, 0.0, 0.0]);
        assert_eq!(movement_from_local(DVec3::X), [0.0, 1.0, 0.0]);
        assert_eq!(movement_from_local(DVec3::Y), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn movement_clamps_out_of_range_input() {
        // forward=5 -> local -Z clamped to -1; strafe=-5 -> +1... wait: axes=(strafe,vert,-fwd).
        assert_eq!(
            local_axes_from_movement([5.0, -5.0, 0.0]),
            DVec3::new(-1.0, 0.0, -1.0)
        );
        // local (9, 0, -9) -> movement [-(-9)->1, 9->1, 0] all clamped to unit.
        assert_eq!(
            movement_from_local(DVec3::new(9.0, 0.0, -9.0)),
            [1.0, 1.0, 0.0]
        );
    }
}
