//! The walk-to / look-at closed-loop math (pure). Both are CLOSED LOOPS: call once per
//! tick with the entity's freshly-delivered pose; they emit the `InputAction` that
//! reduces the error, and converge. They INVERT the input→pose convention defined ONCE
//! in [`vd_core::kinematics`] (`movement_from_local` / `forward_to_yaw_pitch`) — see that
//! module for the axis map + Euler order; the convention is NOT restated here (the sim
//! applies the same source, so there is one definition with no prose copy to drift).

use glam::{DQuat, DVec3};
use vd_core::kinematics;
use vd_devproto::InputAction;

/// Max look delta per tick (radians) — caps the turn rate so look-at is smooth.
pub const MAX_LOOK_STEP: f64 = 0.2;

/// One walk-to tick: the `Move` axes to drive toward the target + whether arrived.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalkStep {
    pub movement: [f32; 3],
    pub arrived: bool,
}

impl WalkStep {
    /// The `Move` input to enqueue this tick.
    #[must_use]
    pub fn action(self) -> InputAction {
        InputAction::Move(self.movement)
    }
}

/// Proportional walk-to in the server's LOCAL movement frame. Aims the per-tick step at
/// the target (clamped `[-1, 1]` axes); `arrived` once within `arrive_epsilon`.
#[must_use]
pub fn walk_to(own_pos: DVec3, own_orient: DQuat, target: DVec3, arrive_epsilon: f64) -> WalkStep {
    let to_target = target - own_pos;
    if to_target.length() <= arrive_epsilon {
        return WalkStep {
            movement: [0.0; 3],
            arrived: true,
        };
    }
    // Express the desired world direction in the entity's local frame, then invert the
    // server's axis map via the ONE shared convention (vd_core::kinematics).
    let local = (own_orient.conjugate() * to_target.normalize()).normalize();
    WalkStep {
        movement: kinematics::movement_from_local(local),
        arrived: false,
    }
}

/// One look-to tick: the `Look` (yaw, pitch) delta to turn toward the target + aligned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookStep {
    pub look: [f32; 2],
    pub aligned: bool,
}

impl LookStep {
    #[must_use]
    pub fn action(self) -> InputAction {
        InputAction::Look(self.look)
    }
}

/// Closed-loop look-at: turn the entity to face `target`, emitting a per-tick `Look`
/// delta (clamped to ±[`MAX_LOOK_STEP`]); `aligned` when both angular errors are within
/// `align_epsilon`. Uses the delivered orient as the current facing.
#[must_use]
pub fn look_at(own_pos: DVec3, own_orient: DQuat, target: DVec3, align_epsilon: f64) -> LookStep {
    let to_target = target - own_pos;
    if to_target.length() <= f64::EPSILON {
        return LookStep {
            look: [0.0; 2],
            aligned: true,
        };
    }
    let (desired_yaw, desired_pitch) = kinematics::forward_to_yaw_pitch(to_target.normalize());
    let (current_yaw, current_pitch) = kinematics::forward_to_yaw_pitch(own_orient * DVec3::NEG_Z);
    let yaw_err = wrap_pi(desired_yaw - current_yaw);
    let pitch_err = desired_pitch - current_pitch;
    let aligned = yaw_err.abs() <= align_epsilon && pitch_err.abs() <= align_epsilon;
    LookStep {
        look: [clamp_step(yaw_err), clamp_step(pitch_err)],
        aligned,
    }
}

/// Wrap an angle to `[-π, π)` (shortest turn) — branchless.
fn wrap_pi(angle: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    (angle + PI).rem_euclid(TAU) - PI
}

fn clamp_step(v: f64) -> f32 {
    v.clamp(-MAX_LOOK_STEP, MAX_LOOK_STEP) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a test orient through the SHARED convention (no hand-rolled from_euler).
    fn yawed(yaw: f64, pitch: f64) -> DQuat {
        kinematics::orient_from_yaw_pitch(yaw, pitch)
    }

    #[test]
    fn walk_to_arrives_within_epsilon() {
        let step = walk_to(DVec3::new(0.5, 0.0, 0.0), DQuat::IDENTITY, DVec3::ZERO, 1.0);
        assert!(step.arrived);
        assert_eq!(step.movement, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn walk_to_aims_the_step_at_the_target_through_the_server_frame() {
        // identity orient: forward is local -Z. A target straight ahead (-Z) -> movement
        // forward (+1 on axis 0). Verify the reconstructed step points at the target.
        let step = walk_to(
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::new(0.0, 0.0, -10.0),
            0.1,
        );
        assert!(!step.arrived);
        assert!(
            (step.movement[0] - 1.0).abs() < 1e-6,
            "forward: {:?}",
            step.movement
        );
        assert!(step.movement[1].abs() < 1e-6, "no strafe");
        assert!(step.movement[2].abs() < 1e-6, "no vertical");
        // Replay the SERVER's transform and confirm the world step heads at the target.
        let m = step.movement;
        let axes = kinematics::local_axes_from_movement(m); // replay via the SHARED convention
        let world_step = DQuat::IDENTITY * axes;
        assert!(
            world_step
                .normalize()
                .abs_diff_eq(DVec3::new(0.0, 0.0, -1.0), 1e-6)
        );
    }

    #[test]
    fn walk_to_respects_a_yawed_heading() {
        // Yawed 90° about Y: the entity faces local-forward = world +X-ish. A target to
        // the world side must still produce a step that (after the server transform)
        // points at the target.
        let orient = yawed(std::f64::consts::FRAC_PI_2, 0.0);
        let target = DVec3::new(5.0, 0.0, 0.0);
        let step = walk_to(DVec3::ZERO, orient, target, 0.1);
        let m = step.movement;
        let axes = kinematics::local_axes_from_movement(m); // replay via the SHARED convention
        let world_step = (orient * axes).normalize();
        assert!(
            world_step.abs_diff_eq(DVec3::new(1.0, 0.0, 0.0), 1e-6),
            "world step {world_step:?} should head +X toward the target"
        );
    }

    #[test]
    fn look_at_degenerate_target_is_aligned() {
        let step = look_at(DVec3::ZERO, DQuat::IDENTITY, DVec3::ZERO, 0.01);
        assert!(step.aligned);
        assert_eq!(step.look, [0.0, 0.0]);
    }

    #[test]
    fn look_at_aligned_when_already_facing_target() {
        // identity orient faces -Z; target straight ahead at -Z -> aligned, zero look.
        let step = look_at(
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::new(0.0, 0.0, -5.0),
            0.01,
        );
        assert!(step.aligned);
        assert!(step.look[0].abs() < 1e-6, "no yaw turn");
        assert!(step.look[1].abs() < 1e-6, "no pitch turn");
    }

    #[test]
    fn look_at_yaw_error_drives_a_clamped_turn() {
        // Target to the +X side: needs a yaw turn -> not aligned, yaw look clamped.
        let step = look_at(
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::new(10.0, 0.0, 0.0),
            0.01,
        );
        assert!(!step.aligned, "LHS-false branch (yaw error)");
        assert!(
            (f64::from(step.look[0]).abs() - MAX_LOOK_STEP).abs() < 1e-6,
            "yaw clamped"
        );
    }

    #[test]
    fn look_at_pitch_only_error_is_not_aligned() {
        // Face the target in yaw (still -Z) but the target is ABOVE -> pitch error only:
        // exercises the (yaw aligned) && (pitch not) arm.
        let step = look_at(
            DVec3::ZERO,
            DQuat::IDENTITY,
            DVec3::new(0.0, 10.0, -10.0),
            0.01,
        );
        assert!(!step.aligned, "LHS-true RHS-false branch (pitch error)");
        assert!(step.look[1].abs() > 0.0, "a pitch turn is commanded");
    }

    #[test]
    fn step_actions_wrap_the_values() {
        let w = WalkStep {
            movement: [1.0, 0.0, 0.0],
            arrived: false,
        };
        assert_eq!(w.action(), InputAction::Move([1.0, 0.0, 0.0]));
        let l = LookStep {
            look: [0.1, -0.2],
            aligned: false,
        };
        assert_eq!(l.action(), InputAction::Look([0.1, -0.2]));
    }

    #[test]
    fn wrap_pi_normalizes_to_the_shortest_turn() {
        use std::f64::consts::PI;
        assert!(wrap_pi(0.0).abs() < 1e-9);
        // 3π wraps to -π (the [-π, π) range) — a single deterministic value.
        assert!((wrap_pi(3.0 * PI) + PI).abs() < 1e-9, "3π → -π");
        assert!(
            wrap_pi(1.5 * PI) < 0.0,
            "1.5π wraps to a negative (shorter) turn"
        );
    }
}
