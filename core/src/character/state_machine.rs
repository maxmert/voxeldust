//! Body / head decoupling state machine.
//!
//! Run once per Physics tick on every shard that owns walking players.
//! Reads the camera-driven look angles the client sent in
//! `PlayerInputData`, splits them into a body-frame yaw and a head-yaw
//! relative to the body, and triggers a turn-in-place when the head
//! reaches its swing limit.
//!
//! # Inputs
//!
//! - `cam_yaw`, `cam_pitch` — what the client camera demands (radians).
//! - `horizontal_speed` — magnitude of the character's planar velocity
//!   (m/s); drives the walk → run blend signal and decides whether the
//!   body chases the velocity vector.
//! - `velocity_direction_yaw` — yaw of the planar velocity vector
//!   (radians, in the same tangent frame as `BodyYaw`). Only consulted
//!   when `horizontal_speed > class.stationary_speed_threshold`.
//!
//! # Outputs
//!
//! - Writes `BodyYaw`, `HeadYaw`, `HeadPitch`.
//! - Inserts/updates/removes `TurnInPlace` to drive the per-tick eased
//!   body rotation.
//!
//! # Why a free function (not a Bevy system)
//!
//! Each shard reads its own velocity / tangent-frame source — ship
//! interior uses ship-local Y-up, planet uses radial up + east/north.
//! Hoisting the math into a pure function keeps the state machine
//! reusable while letting each shard stay in charge of its frame
//! conversion.

use super::body_head::{BodyYaw, HeadPitch, HeadYaw, TurnInPlace};
use super::class::CharacterClass;

/// Wrap an angle into `(-π, π]`.
#[inline]
pub fn wrap_pi(a: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let mut x = (a + std::f32::consts::PI) % two_pi;
    if x <= 0.0 {
        x += two_pi;
    }
    x - std::f32::consts::PI
}

/// Smooth-step easing: 3t² - 2t³. Matches the Mixamo TIP clip's
/// authored body rotation curve closely enough that the body-yaw and
/// the foot-step poses stay synchronized for a 0.5 s turn.
#[inline]
fn ease_in_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Per-tick frame-aware exponential decay toward a target angle.
/// Returns the new value. `tau` is the time constant in seconds.
#[inline]
fn exp_decay_angle(current: f32, target: f32, dt: f32, tau: f32) -> f32 {
    if tau <= 0.0 {
        return target;
    }
    let alpha = 1.0 - (-dt / tau).exp();
    let delta = wrap_pi(target - current);
    wrap_pi(current + alpha * delta)
}

/// Output of one state-machine step. The shard applies these by
/// writing the components and inserting / updating / removing
/// [`TurnInPlace`]. Returning a struct (vs. mutating in place) keeps
/// the function trivially testable without a `World`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyHeadUpdate {
    pub body_yaw: f32,
    pub head_yaw: f32,
    pub head_pitch: f32,
    /// `Some` when a turn is active for the next tick; `None` when no
    /// turn (the shard removes the [`TurnInPlace`] component).
    pub turn: Option<TurnInPlace>,
    /// Drives the wire-format `locomotion_speed` field; equal to the
    /// input `horizontal_speed` (mirrored for downstream convenience).
    pub locomotion_speed: f32,
}

/// One tick of the body / head decoupling state machine.
///
/// `velocity_direction_yaw` (in earlier revisions of this function)
/// is intentionally absent: AAA convention is **body chases camera
/// direction when moving**, not the velocity vector. Letting the body
/// chase velocity made pressing S (back-pedal) rotate the body 180°
/// to face the velocity direction (= behind the camera) — exactly
/// what no third-person / first-person hybrid does. With body chasing
/// camera, the player back-pedals / strafes while still facing
/// forward; W/A/S/D + diagonals all keep the body oriented at the
/// camera, and the legs play whichever locomotion clip best matches
/// the speed (Phase D adds backward / strafe variants when authored).
pub fn step_body_head(
    class: &CharacterClass,
    prev_body_yaw: f32,
    prev_turn: Option<TurnInPlace>,
    cam_yaw: f32,
    cam_pitch: f32,
    horizontal_speed: f32,
    dt: f32,
) -> BodyHeadUpdate {
    let head_pitch = cam_pitch.clamp(-class.head_pitch_limit, class.head_pitch_limit);

    // Branch 1: Mid-turn — drive body yaw via eased curve, then pop.
    if let Some(mut turn) = prev_turn {
        turn.t = (turn.t + dt / class.turn_in_place_duration).min(1.0);
        let body_yaw = wrap_pi(
            prev_body_yaw + ease_in_out(turn.t) * wrap_pi(turn.target_body_yaw - prev_body_yaw),
        );
        let head_yaw =
            wrap_pi(cam_yaw - body_yaw).clamp(-class.head_yaw_limit, class.head_yaw_limit);
        let turn_done = turn.t >= 1.0;
        return BodyHeadUpdate {
            body_yaw,
            head_yaw,
            head_pitch,
            turn: if turn_done { None } else { Some(turn) },
            locomotion_speed: horizontal_speed,
        };
    }

    // Branch 2: Moving — body chases CAMERA direction (not velocity).
    // Camera-relative movement: W / S / A / D / diagonals all keep
    // the body facing the camera; the legs animate the gait.
    if horizontal_speed > class.stationary_speed_threshold {
        let body_yaw = exp_decay_angle(prev_body_yaw, cam_yaw, dt, class.body_align_tau);
        let head_yaw =
            wrap_pi(cam_yaw - body_yaw).clamp(-class.head_yaw_limit, class.head_yaw_limit);
        return BodyHeadUpdate {
            body_yaw,
            head_yaw,
            head_pitch,
            turn: None,
            locomotion_speed: horizontal_speed,
        };
    }

    // Branch 3: Stationary — head leads. Trigger turn when head yaw
    // exceeds the per-class limit.
    let head_local = wrap_pi(cam_yaw - prev_body_yaw);
    if head_local.abs() > class.head_yaw_limit {
        // Re-anchor the head to the per-class fraction of the limit
        // (e.g. 70°) so the new neutral leaves headroom and the user
        // doesn't immediately re-trigger another turn.
        let reanchor = class.head_yaw_limit * class.turn_reanchor_fraction;
        let target_body_yaw = wrap_pi(prev_body_yaw + (head_local - head_local.signum() * reanchor));
        // First tick of the turn: body hasn't moved yet.
        let head_yaw =
            wrap_pi(cam_yaw - prev_body_yaw).clamp(-class.head_yaw_limit, class.head_yaw_limit);
        return BodyHeadUpdate {
            body_yaw: prev_body_yaw,
            head_yaw,
            head_pitch,
            turn: Some(TurnInPlace {
                target_body_yaw,
                t: 0.0,
            }),
            locomotion_speed: horizontal_speed,
        };
    }

    // Stationary, head within limit — body and head decoupled.
    BodyHeadUpdate {
        body_yaw: prev_body_yaw,
        head_yaw: head_local,
        head_pitch,
        turn: None,
        locomotion_speed: horizontal_speed,
    }
}

/// Apply a [`BodyHeadUpdate`] to the player entity components.
/// Convenience wrapper used by both shards from their per-tick system
/// to keep the call sites tiny.
#[inline]
pub fn apply_update(
    body_yaw: &mut BodyYaw,
    head_yaw: &mut HeadYaw,
    head_pitch: &mut HeadPitch,
    update: &BodyHeadUpdate,
) {
    body_yaw.0 = update.body_yaw;
    head_yaw.0 = update.head_yaw;
    head_pitch.0 = update.head_pitch;
}

#[cfg(test)]
mod tests {
    use super::super::class::HUMAN_DEFAULT;
    use super::*;

    fn class() -> CharacterClass {
        HUMAN_DEFAULT
    }

    #[test]
    fn stationary_head_exceeds_limit_triggers_turn() {
        let cls = class();
        // Camera demands 95° while body is at 0° — head can't reach it.
        let cam = std::f32::consts::FRAC_PI_2 + 0.1;
        let r = step_body_head(&cls, 0.0, None, cam, 0.0, 0.0, 1.0 / 20.0);
        let turn = r.turn.expect("expected TurnInPlace to fire");
        assert!(turn.target_body_yaw > 0.0);
        assert_eq!(turn.t, 0.0);
        // Head yaw clamps to limit on the firing tick.
        assert!((r.head_yaw - cls.head_yaw_limit).abs() < 1e-5);
    }

    #[test]
    fn turn_progresses_and_completes() {
        let cls = class();
        let mut body = 0.0_f32;
        let mut turn = Some(TurnInPlace {
            target_body_yaw: cls.head_yaw_limit, // 90° turn target
            t: 0.0,
        });
        let dt = 1.0 / 20.0;
        let cam = cls.head_yaw_limit; // camera held at the limit
        let mut steps = 0;
        while turn.is_some() {
            let r = step_body_head(&cls, body, turn, cam, 0.0, 0.0, dt);
            body = r.body_yaw;
            turn = r.turn;
            steps += 1;
            assert!(steps < 100, "turn should finish within the duration");
        }
        // Body should have rotated approximately to the target.
        assert!(
            (body - cls.head_yaw_limit).abs() < 0.05,
            "body landed at {} radians, target {}",
            body,
            cls.head_yaw_limit
        );
    }

    #[test]
    fn moving_body_chases_camera_direction() {
        let cls = class();
        let dt = 1.0 / 20.0;
        let mut body = 0.0_f32;
        // Camera demanded 90° to the left. Body should chase the
        // CAMERA yaw (not velocity), regardless of which key the
        // player is holding (forward, backward, strafe).
        let cam = std::f32::consts::FRAC_PI_2;
        let mut steps = 0;
        loop {
            let r = step_body_head(&cls, body, None, cam, 0.0, 5.0, dt);
            body = r.body_yaw;
            steps += 1;
            if (body - cam).abs() < 0.05 || steps > 200 {
                break;
            }
        }
        assert!(
            (body - cam).abs() < 0.05,
            "body did not chase camera (got {} after {} ticks)",
            body,
            steps
        );
    }

    #[test]
    fn back_pedaling_does_not_rotate_body() {
        // The classic AAA-correctness regression: holding S (back-pedal)
        // moves the player backward, but the body must keep facing the
        // camera. This is the canary for the velocity-vs-camera-chase
        // bug that previously rotated the body 180°.
        let cls = class();
        let dt = 1.0 / 20.0;
        let body = 0.0; // body already aligned with camera (forward).
        let cam = 0.0; // camera unchanged — player only presses S.
        let r = step_body_head(&cls, body, None, cam, 0.0, 5.0, dt);
        assert!(
            r.body_yaw.abs() < 1e-3,
            "body rotated to {} when back-pedaling — should have stayed at 0",
            r.body_yaw
        );
    }

    #[test]
    fn strafing_does_not_rotate_body() {
        // Holding A or D produces sideways velocity; body must remain
        // facing the camera (side-walk), not turn 90° to face the
        // velocity direction.
        let cls = class();
        let dt = 1.0 / 20.0;
        let body = 0.0;
        let cam = 0.0;
        let r = step_body_head(&cls, body, None, cam, 0.0, 5.0, dt);
        assert!(
            r.body_yaw.abs() < 1e-3,
            "body rotated to {} when strafing — should have stayed at 0",
            r.body_yaw
        );
    }

    #[test]
    fn pitch_is_clamped() {
        let cls = class();
        let r = step_body_head(&cls, 0.0, None, 0.0, 100.0, 0.0, 1.0 / 20.0);
        assert!((r.head_pitch - cls.head_pitch_limit).abs() < 1e-5);
    }

    #[test]
    fn stationary_head_within_limit_does_not_trigger_turn() {
        // Re-test the existing case to make sure the signature change
        // didn't regress stationary behavior.
        let cls = class();
        let r = step_body_head(&cls, 0.0, None, 0.5, 0.0, 0.0, 1.0 / 20.0);
        assert_eq!(r.turn, None);
        assert!((r.head_yaw - 0.5).abs() < 1e-5);
        assert_eq!(r.body_yaw, 0.0);
    }
}
