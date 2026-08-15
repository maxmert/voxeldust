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

/// `|forward.y|` at/above which a direction is treated as "at the gimbal pole" (within
/// ~0.8° of straight up/down): there yaw is ill-defined (`atan2` of ~0/~0), so a yaw
/// correction is meaningless. Inside this band `look_at` holds yaw and drives pitch only
/// — reducing pitch pulls the facing off the pole, after which yaw resumes (cleaner +
/// faster than letting yaw chase the degenerate value). The convention itself stays in
/// [`vd_core::kinematics`]; this is monomorphic controller logic (HR5).
const NEAR_VERTICAL: f64 = 0.9999;

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

/// Walk-to in the server's LOCAL movement frame: aims the per-tick step at the target with
/// UNIT-clamped axes; `arrived` once within `arrive_epsilon`.
///
/// `max_step_m` is the BRAKING term (Stage B4): the caller's one-tick full-speed travel
/// (`move_speed_mps · tick_dt_s`). Within that distance the axes scale down proportionally
/// (`distance / max_step_m`), so the last tick's step lands ON the target instead of 10 m past
/// it — the arrival phase, and the reason a sub-step `arrive_epsilon` can settle. The server
/// integrates `axes · speed · dt`, so a smaller magnitude simply moves slower; no new message.
///
/// `max_step_m <= 0` (or non-finite) disables braking — full-magnitude axes throughout, the
/// pre-B4 behaviour, where the ORIGINAL CONTRACT stands: `arrive_epsilon` MUST exceed one sim
/// step, or the fixed step overshoots and oscillates around the target forever (pinned by
/// `walk_to_oscillates_if_arrive_epsilon_is_below_the_step`).
#[must_use]
pub fn walk_to(
    own_pos: DVec3,
    own_orient: DQuat,
    target: DVec3,
    arrive_epsilon: f64,
    max_step_m: f64,
) -> WalkStep {
    let to_target = target - own_pos;
    let distance = to_target.length();
    if distance <= arrive_epsilon {
        return WalkStep {
            movement: [0.0; 3],
            arrived: true,
        };
    }
    // Express the desired world direction in the entity's local frame, then invert the
    // server's axis map via the ONE shared convention (vd_core::kinematics).
    let local = (own_orient.conjugate() * to_target.normalize()).normalize();
    // THE BRAKE: unit direction scaled by how many steps remain, capped at full. Monomorphic
    // two-arm guard (HR5): a non-positive/non-finite max_step_m is the no-brake legacy arm.
    let scale = if max_step_m.is_finite() && max_step_m > 0.0 {
        (distance / max_step_m).min(1.0)
    } else {
        1.0
    };
    WalkStep {
        movement: kinematics::movement_from_local(local * scale),
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
/// delta (clamped to ±[`MAX_LOOK_STEP`]); `aligned` once the facing is within
/// `align_epsilon` of the target (the true 3-D angle, so it stays honest at the pole).
/// Uses the delivered orient as the current facing.
#[must_use]
pub fn look_at(own_pos: DVec3, own_orient: DQuat, target: DVec3, align_epsilon: f64) -> LookStep {
    let to_target = target - own_pos;
    if to_target.length() <= f64::EPSILON {
        return LookStep {
            look: [0.0; 2],
            aligned: true,
        };
    }
    let desired_dir = to_target.normalize();
    let current_dir = own_orient * DVec3::NEG_Z;
    let (desired_yaw, desired_pitch) = kinematics::forward_to_yaw_pitch(desired_dir);
    let (current_yaw, current_pitch) = kinematics::forward_to_yaw_pitch(current_dir);
    // Gimbal pole guard: if EITHER endpoint is near-vertical, its yaw is ill-defined —
    // hold yaw this tick and drive pitch only. `|` (not `||`) evaluates both endpoints
    // unconditionally, so there is no short-circuit arm to leave uncovered (HR5).
    let near_pole = near_vertical(desired_dir) | near_vertical(current_dir);
    let yaw_err = if near_pole {
        0.0
    } else {
        wrap_pi(desired_yaw - current_yaw)
    };
    let pitch_err = desired_pitch - current_pitch;
    // `aligned` is the TRUE 3-D angle between the facing and the target — not a
    // yaw/pitch decomposition. This is honest at the pole (where the guard zeroes
    // yaw_err): it can never report aligned while the facing is still > align_epsilon
    // off the target, even though the per-tick COMMAND drops yaw to avoid spinning.
    let aligned = current_dir.angle_between(desired_dir) <= align_epsilon;
    LookStep {
        look: [clamp_step(yaw_err), clamp_step(pitch_err)],
        aligned,
    }
}

/// Whether a unit direction is within [`NEAR_VERTICAL`] of straight up/down.
fn near_vertical(dir: DVec3) -> bool {
    dir.y.abs() >= NEAR_VERTICAL
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

    /// Drive a `walk_to` CLOSED LOOP exactly as `stub::integrate` does (world step =
    /// `orient · local_axes_from_movement(movement) · step_len`, via the shared
    /// convention), asserting the distance to target never regresses. Returns the tick
    /// it arrived, or `None` if it never did within `max_ticks` (a stall/oscillation).
    fn drive_walk_to(
        mut pos: DVec3,
        orient: DQuat,
        target: DVec3,
        arrive_eps: f64,
        step_len: f64,
        max_ticks: u32,
    ) -> Option<u32> {
        let mut prev = f64::INFINITY;
        for tick in 0..max_ticks {
            let s = walk_to(pos, orient, target, arrive_eps, 0.0);
            if s.arrived {
                return Some(tick);
            }
            let dist = (target - pos).length();
            assert!(dist <= prev + 1e-9, "distance regressed: {dist} > {prev}");
            prev = dist;
            let axes = kinematics::local_axes_from_movement(s.movement);
            pos += orient * axes * step_len;
        }
        None
    }

    /// Drive a `look_at` CLOSED LOOP exactly as `stub::integrate` applies look
    /// (`yaw += look[0]; pitch += look[1]; orient = orient_from_yaw_pitch`). Returns the
    /// tick it aligned, or `None` (a stall/oscillation). When `monotone`, asserts the
    /// controller's own error `max(|yaw_err|, |pitch_err|)` never grows — the clamped
    /// per-axis turn must converge without overshoot (off at the pole, where recovered
    /// yaw is noise; there convergence-to-aligned alone proves no spin).
    fn drive_look_at(
        start_yaw: f64,
        start_pitch: f64,
        target: DVec3,
        align_eps: f64,
        max_ticks: u32,
        monotone: bool,
    ) -> Option<u32> {
        let (mut yaw, mut pitch) = (start_yaw, start_pitch);
        let (dyaw, dpitch) = kinematics::forward_to_yaw_pitch(target.normalize());
        let mut prev = f64::INFINITY;
        for tick in 0..max_ticks {
            let orient = kinematics::orient_from_yaw_pitch(yaw, pitch);
            let s = look_at(DVec3::ZERO, orient, target, align_eps);
            if s.aligned {
                return Some(tick);
            }
            if monotone {
                let (cyaw, cpitch) = kinematics::forward_to_yaw_pitch(orient * DVec3::NEG_Z);
                let err = wrap_pi(dyaw - cyaw).abs().max((dpitch - cpitch).abs());
                assert!(
                    err <= prev + 1e-9,
                    "controller error regressed: {err} > {prev}"
                );
                prev = err;
            }
            yaw += f64::from(s.look[0]);
            pitch += f64::from(s.look[1]);
        }
        None
    }

    #[test]
    fn walk_to_arrives_within_epsilon() {
        let step = walk_to(
            DVec3::new(0.5, 0.0, 0.0),
            DQuat::IDENTITY,
            DVec3::ZERO,
            1.0,
            0.0,
        );
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
            0.0,
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
        let step = walk_to(DVec3::ZERO, orient, target, 0.1, 0.0);
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
        // Face the target in yaw (still -Z) but the target is ABOVE -> a pitch-only
        // angular error keeps it not-aligned and commands a pitch turn.
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
    fn walk_to_converges_monotonically_even_from_a_wrong_heading() {
        // CONVERGENCE (not just one tick): a fixed heading that does NOT face the
        // target must still strafe/back toward it, the distance shrinking every tick
        // until arrival. A bang-bang oscillation or a stall would fail to arrive.
        let orient = yawed(0.9, 0.3);
        let arrived = drive_walk_to(
            DVec3::new(4.0, -1.0, 2.0),
            orient,
            DVec3::new(-2.0, 1.5, -3.0),
            0.25, // arrive_eps > step_len, so it lands inside the band (no overshoot loop)
            0.1,
            1000,
        );
        assert!(arrived.is_some(), "walk_to did not converge");
    }

    #[test]
    fn walk_to_oscillates_if_arrive_epsilon_is_below_the_step() {
        // CONTRACT (pinned): walk_to emits fixed unit-magnitude movement and does NOT
        // know the sim step, so an arrive_epsilon BELOW one step can never settle — the
        // step overshoots and the dot oscillates around the target forever. Documented
        // so callers pick a tolerance ≥ one step; a sub-step epsilon never arrives.
        let step_len = 0.1;
        let arrive_eps = 0.02; // < step_len
        let mut pos = DVec3::new(0.0, 0.0, -0.55);
        let mut min_dist = f64::INFINITY;
        let mut ever_arrived = false;
        for _ in 0..200 {
            let s = walk_to(pos, DQuat::IDENTITY, DVec3::ZERO, arrive_eps, 0.0);
            ever_arrived |= s.arrived; // accumulate with `|=` (no branch to leave uncovered)
            min_dist = min_dist.min(pos.length());
            let axes = kinematics::local_axes_from_movement(s.movement);
            pos += DQuat::IDENTITY * axes * step_len;
        }
        assert!(
            !ever_arrived,
            "a sub-step tolerance cannot settle — it oscillates"
        );
        assert!(
            min_dist <= step_len,
            "it does get within a step of the target, just never inside epsilon"
        );
    }

    /// The no-brake guard's OTHER refusal arm (HR5): a NON-FINITE `max_step_m` disables braking
    /// exactly like a non-positive one — the same full-magnitude step, equality-pinned against
    /// the `0.0` legacy arm the oscillation contract above drives.
    #[test]
    fn walk_to_treats_a_non_finite_brake_as_no_brake() {
        let pos = DVec3::new(0.0, 0.0, -0.55);
        let braked_off = walk_to(pos, DQuat::IDENTITY, DVec3::ZERO, 0.02, 0.0);
        let non_finite = walk_to(pos, DQuat::IDENTITY, DVec3::ZERO, 0.02, f64::NAN);
        assert_eq!(non_finite, braked_off);
        assert!(!non_finite.arrived);
    }

    /// THE BRAKE (Stage B4): with `max_step_m` = the sim step, the SAME sub-step tolerance that
    /// oscillates forever un-braked (the pinned test above) settles — inside the last step the
    /// axes scale down, the final step lands on the target, and `arrived` fires. This is the
    /// arrival phase that lets a flight-speed fixture park at a rendezvous point.
    #[test]
    fn walk_to_with_a_brake_settles_inside_a_sub_step_epsilon() {
        let step_len = 0.1;
        let arrive_eps = 0.02; // < step_len — the exact tolerance the un-braked contract forbids
        let mut pos = DVec3::new(0.0, 0.0, -0.55);
        let mut arrived_at = None;
        for tick in 0..200 {
            let s = walk_to(pos, DQuat::IDENTITY, DVec3::ZERO, arrive_eps, step_len);
            if s.arrived {
                arrived_at = Some(tick);
                break;
            }
            let axes = kinematics::local_axes_from_movement(s.movement);
            pos += DQuat::IDENTITY * axes * step_len;
        }
        assert!(
            arrived_at.is_some(),
            "the braked controller settles inside a sub-step epsilon"
        );
        let parked = pos.length();
        assert!(
            parked <= arrive_eps,
            "parked inside the arrive ball: {parked} > {arrive_eps}"
        );
    }

    #[test]
    fn look_at_converges_monotonically_for_an_off_axis_target() {
        // Yaw AND pitch error together: the clamped per-tick turn must converge with no
        // overshoot in the controller's own error space.
        let arrived = drive_look_at(0.5, -0.2, DVec3::new(3.0, 2.0, -1.0), 0.02, 1000, true);
        assert!(arrived.is_some(), "off-axis look_at did not converge");
    }

    #[test]
    fn look_at_converges_for_a_pure_pitch_target() {
        // Facing -Z, target above and ahead → pitch-only error; converges monotonically.
        let arrived = drive_look_at(0.0, 0.0, DVec3::new(0.0, 5.0, -5.0), 0.02, 1000, true);
        assert!(arrived.is_some(), "pure-pitch look_at did not converge");
    }

    #[test]
    fn look_at_at_the_pole_commands_zero_yaw_so_it_cannot_spin() {
        // PINS the guard directly: target straight up (near-vertical) from a yawed +
        // pitched facing → the yaw command is EXACTLY zero (yaw held), so it can never
        // chase the degenerate atan2; pitch still drives toward vertical. Without the
        // guard, look[0] would be a non-zero (clamped) yaw — this test kills that.
        let orient = kinematics::orient_from_yaw_pitch(1.2, 0.3);
        let step = look_at(DVec3::ZERO, orient, DVec3::Y, 0.01);
        assert_eq!(step.look[0], 0.0, "yaw is held at the pole");
        assert!(!step.aligned, "not yet aligned — pitch must still drive up");
        assert!(step.look[1] > 0.0, "pitch drives toward vertical");
    }

    #[test]
    fn look_at_converges_at_the_vertical_pole_without_spinning() {
        // Target straight up: yaw is ill-defined. The pole guard holds yaw and drives
        // pitch, so it REACHES aligned (and faster than chasing the degenerate yaw).
        let arrived = drive_look_at(0.7, 0.0, DVec3::Y, 0.02, 1000, false);
        assert!(
            arrived.is_some(),
            "vertical-target look_at did not converge via the pole guard"
        );
    }

    #[test]
    fn the_convergence_budget_is_a_real_bound_not_a_hang() {
        // A budget too small to reach the (far) target returns None — proving the
        // controllers don't falsely report arrival/alignment, and exercising the
        // budget-exhausted path of the drivers.
        assert_eq!(
            drive_walk_to(
                DVec3::ZERO,
                DQuat::IDENTITY,
                DVec3::new(0.0, 0.0, -100.0),
                0.25,
                0.1,
                3,
            ),
            None,
            "100 m is unreachable in 3 ticks of 0.1 m"
        );
        assert_eq!(
            drive_look_at(0.0, 0.0, DVec3::new(10.0, 0.0, 0.0), 0.02, 2, true),
            None,
            "a 90° turn is unreachable in 2 ticks of 0.2 rad"
        );
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
