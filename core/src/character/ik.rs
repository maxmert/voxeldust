//! Inverse-kinematics solvers — pure Rust, no Bevy / no Rapier deps.
//!
//! AAA character animation overlays IK on top of the clip-driven
//! pose: the artist-authored animation says where the foot
//! "intends" to plant, and a per-frame solver corrects for actual
//! terrain contact (foot IK, Phase G), aim-target tracking
//! (look-at IK, Phase H), and grab / button-press reach (hand IK,
//! future). All three reduce to "given a chain of two or three
//! bones, rotate it so the end effector reaches a target world
//! position" — exactly what [`solve_two_bone_ik`] computes
//! analytically below.
//!
//! # Why analytical, not CCD / FABRIK
//!
//! For 2-bone chains (hip → knee → ankle, shoulder → elbow → wrist)
//! the analytical closed-form solution from Hugo Elias / Sebastian
//! Lague:
//! - Always converges in O(1) — no iteration loop.
//! - Produces deterministic, frame-coherent output (essential for
//!   network-replicated visuals).
//! - Naturally clamps when the target is unreachable (extends the
//!   chain along the target direction).
//! - Keeps the limb plane on the user-supplied pole side, so the
//!   knee never bends backwards.
//!
//! CCD / FABRIK only earn their keep on chains of 4+ bones (full
//! spine, multi-joint fingers). When character IK lands those
//! later, this module gains a separate `solve_ccd` entry point;
//! the analytical path stays.

use glam::{Quat, Vec3};

/// Inputs for one 2-bone IK solve. All positions in **world space**
/// (or any single coherent frame — caller picks).
#[derive(Clone, Copy, Debug)]
pub struct TwoBoneInput {
    /// World position of the chain's root joint (hip / shoulder).
    pub root: Vec3,
    /// World position of the middle joint (knee / elbow) at the
    /// pre-IK pose. Used for two things only: (1) measuring the
    /// upper-bone length, (2) detecting the bend-side fallback when
    /// the pole is degenerate.
    pub mid: Vec3,
    /// World position of the end effector (ankle / wrist) at the
    /// pre-IK pose. Used for measuring the lower-bone length.
    pub tip: Vec3,
    /// Where the end effector should land in world space.
    pub target: Vec3,
    /// World position whose direction (from `root`) tells the solver
    /// which side of the chain the bend should face. For a leg this
    /// is typically `root + character_forward` (knees forward); for
    /// an arm it's `root + character_back` (elbows back).
    pub pole: Vec3,
}

/// Solved limb pose. Both fields are **world-space rotations** that,
/// when applied to the bones (replacing or composing with their
/// existing world rotations — caller picks), put the tip at
/// [`TwoBoneInput::target`] (or the closest reachable point if the
/// target is past the chain's reach).
#[derive(Clone, Copy, Debug)]
pub struct TwoBoneOutput {
    /// World rotation that takes the original `root → mid` vector to
    /// the solved `root → mid_new` vector. Apply to the root bone.
    pub root_delta_world: Quat,
    /// World rotation that takes the original `mid → tip` vector to
    /// the solved `mid_new → tip_new` vector. Apply to the mid bone.
    pub mid_delta_world: Quat,
    /// True iff the target was within the chain's reach. When false
    /// the chain is fully extended toward the target — the closest
    /// the joints can physically get.
    pub reached: bool,
}

/// Numerical floor — chain segments shorter than this collapse the
/// solve to the identity (avoids divide-by-zero in degenerate rigs).
const MIN_BONE_LENGTH: f32 = 1.0e-5;

/// Cap the target distance at this fraction of the chain's reach to
/// avoid singular fully-locked-out poses. AAA convention: 0.999 keeps
/// 1 mm of bend room at a 1 m chain.
const REACH_CLAMP: f32 = 0.999;

/// Analytical 2-bone IK. AAA-typical implementation: cosine law for
/// joint angles, Quat::from_rotation_arc for the world rotations,
/// pole-aware bend axis to keep the knee on the right side.
///
/// Returns identity rotations + `reached = false` for degenerate
/// inputs (zero-length bones, root == target). Clamps target distance
/// to keep the chain just under fully extended.
pub fn solve_two_bone_ik(input: TwoBoneInput) -> TwoBoneOutput {
    let upper = input.mid - input.root;
    let lower = input.tip - input.mid;
    let upper_len = upper.length();
    let lower_len = lower.length();
    if upper_len < MIN_BONE_LENGTH || lower_len < MIN_BONE_LENGTH {
        return identity_output();
    }

    let to_target = input.target - input.root;
    let mut target_dist = to_target.length();
    let chain_reach = upper_len + lower_len;
    // `reached` reflects whether the analytical solution was a true
    // intersection of the two circles (target inside the full chain
    // reach), independent of the implementation-detail clamp band.
    let reached = target_dist <= chain_reach;
    let max_solver_dist = chain_reach * REACH_CLAMP;
    target_dist = target_dist.min(max_solver_dist).max(MIN_BONE_LENGTH);
    let to_target_dir = to_target.normalize_or_zero();
    if to_target_dir.length_squared() < 0.5 {
        // root and target are coincident → no useful direction. Hold pose.
        return identity_output();
    }

    // Cosine law on the new triangle (root, mid_new, tip = target):
    //   cos(angle_at_root) = (upper² + d² - lower²) / (2 * upper * d)
    // Clamp before acos to absorb floating-point overshoot.
    let cos_root_new = (upper_len * upper_len + target_dist * target_dist
        - lower_len * lower_len)
        / (2.0 * upper_len * target_dist);
    let angle_root_new = cos_root_new.clamp(-1.0, 1.0).acos();

    // Project the pole onto the plane perpendicular to the target
    // direction. Removes the parallel component so the perpendicular
    // axis points purely "side" of the chain — defines which side
    // the knee bends toward.
    let pole_vec = input.pole - input.root;
    let pole_perp_raw = pole_vec - to_target_dir * pole_vec.dot(to_target_dir);
    let pole_perp = if pole_perp_raw.length_squared() > MIN_BONE_LENGTH * MIN_BONE_LENGTH {
        pole_perp_raw.normalize()
    } else {
        // Degenerate: pole is on the target axis. Fall back to the
        // current-pose bend side so the knee doesn't snap.
        let cur_perp_raw = upper - to_target_dir * upper.dot(to_target_dir);
        if cur_perp_raw.length_squared() > MIN_BONE_LENGTH * MIN_BONE_LENGTH {
            cur_perp_raw.normalize()
        } else {
            // Even the current pose is collinear — pick an arbitrary
            // perpendicular axis. Visually arbitrary in this corner.
            let helper = if to_target_dir.x.abs() < 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            (helper - to_target_dir * helper.dot(to_target_dir)).normalize()
        }
    };

    // New mid-joint position: angle_root_new from the target axis,
    // pivoting toward the pole side.
    let mid_new = input.root
        + (to_target_dir * angle_root_new.cos() + pole_perp * angle_root_new.sin())
            * upper_len;

    // World deltas: rotation that takes the old upper vector onto
    // the new one, and same for lower.
    let upper_old_dir = upper / upper_len;
    let upper_new = mid_new - input.root;
    let upper_new_dir = upper_new.normalize_or_zero();
    let root_delta_world = Quat::from_rotation_arc(upper_old_dir, upper_new_dir);

    // For the mid delta: rotate the OLD lower vector (which was in
    // the old limb plane) to point at the target from `mid_new`. The
    // mid bone's parent (root) has already rotated by
    // `root_delta_world`, so the lower vector in the parent-rotated
    // frame is `root_delta_world * lower`. Rotate that to the new
    // direction.
    let lower_old_dir = lower / lower_len;
    let lower_after_root = root_delta_world * lower_old_dir;
    let lower_new_dir = (input.target - mid_new).normalize_or_zero();
    let mid_delta_world = if lower_after_root.length_squared() > 0.5
        && lower_new_dir.length_squared() > 0.5
    {
        Quat::from_rotation_arc(lower_after_root, lower_new_dir)
    } else {
        Quat::IDENTITY
    };

    TwoBoneOutput {
        root_delta_world,
        mid_delta_world,
        reached,
    }
}

#[inline]
fn identity_output() -> TwoBoneOutput {
    TwoBoneOutput {
        root_delta_world: Quat::IDENTITY,
        mid_delta_world: Quat::IDENTITY,
        reached: false,
    }
}

// =============================================================================
// Aim chain (neck + head) IK — Phase H
// =============================================================================

/// Inputs for an aim-chain (neck + head) IK solve. Positions in world
/// space; `forward_local` is the bones' local-frame axis pointing
/// in the direction the bone should be aimed (e.g. Mixamo Y-bot's
/// head + neck bind has `+Z` as the looking axis).
///
/// The aim is split between two bones to mimic real human spinal
/// articulation: the neck does a fraction of the total turn, and the
/// head completes the remainder. Each bone has SEPARATE yaw and
/// pitch limits — humans can yaw the head much further than they
/// can pitch it. This produces the characteristic "the body slightly
/// turns, then the head finishes the look" sequence that makes
/// characters feel alive instead of robotic.
#[derive(Clone, Copy, Debug)]
pub struct AimChainInput {
    /// World position of the neck joint at the current animated pose.
    pub neck_pos: Vec3,
    /// World rotation of the neck joint (current animated pose, BEFORE
    /// IK). Used to compute the neck's forward axis in world space.
    pub neck_world_rot: Quat,
    /// World position of the head joint at the current animated pose.
    pub head_pos: Vec3,
    /// World rotation of the head joint (current animated pose, BEFORE
    /// IK). Used to compute the head's forward axis in world space.
    pub head_world_rot: Quat,
    /// Forward axis in the bones' local frame after bind. Same axis
    /// for both bones (consistent rig). Use the per-class
    /// `look_forward_local` from `CharacterClass`.
    pub forward_local: Vec3,
    /// Up axis in the bones' local frame after bind — the axis
    /// rotation for the YAW component of a turn. The right axis is
    /// derived as `up_local × forward_local`.
    pub up_local: Vec3,
    /// Where the head should look, in world space.
    pub target: Vec3,
    /// Fraction of the total aim handled by the neck (`0.0..1.0`).
    /// Industry convention: `0.3` (neck does 30 %, head does the rest).
    /// `1.0` makes the neck do everything (head stays in animation
    /// pose); `0.0` makes the head do everything (neck stays).
    pub neck_share: f32,
    /// Hard limit on neck YAW (around `up_local`) in radians.
    /// Half-cone — one-sided angle from neutral.
    pub neck_yaw_limit: f32,
    /// Hard limit on neck PITCH (around the right axis) in radians.
    /// Half-cone.
    pub neck_pitch_limit: f32,
    /// Hard limit on head YAW (around `up_local`) in radians, applied
    /// AFTER the neck has rotated. Half-cone.
    pub head_yaw_limit: f32,
    /// Hard limit on head PITCH (around the right axis) in radians,
    /// applied AFTER the neck has rotated. Half-cone.
    pub head_pitch_limit: f32,
}

/// Solved aim-chain pose. Both rotations are world-space deltas to
/// COMPOSE on top of the bones' current world rotations: a Bevy
/// caller does
/// ```ignore
/// let new_neck_world = neck_delta_world * neck_old_world;
/// let new_head_world = head_delta_world * (neck_delta_world * head_old_world);
/// ```
/// then converts back to local via parent inverse.
#[derive(Clone, Copy, Debug)]
pub struct AimChainOutput {
    /// World-space rotation that turns the neck (and everything
    /// below it on the rig, including the head) toward the target
    /// by up to `neck_share * total` (capped at `neck_cone_limit`).
    pub neck_delta_world: Quat,
    /// World-space rotation that turns the head the rest of the way
    /// toward the target, applied AFTER the neck has rotated.
    /// Capped at `head_cone_limit`.
    pub head_delta_world: Quat,
}

/// Closed-form aim-chain solver using **explicit yaw + pitch
/// decomposition** in each bone's local frame. Industry-standard
/// approach for character look-at because it:
/// - Eliminates the "axis is arbitrary" problem of
///   `from_rotation_arc` for near-antiparallel inputs (180° turns
///   were previously pitching the head up instead of yawing).
/// - Lets us clamp yaw and pitch on independent budgets (humans
///   yaw the head far further than they pitch it).
/// - Composes with sub-bone "shares" cleanly: the neck does
///   `neck_share` of yaw and `neck_share` of pitch, the head
///   finishes within its own limits.
///
/// Returns identity rotations for degenerate inputs (zero-length
/// forward axis, target on top of neck/head).
pub fn solve_aim_chain(input: AimChainInput) -> AimChainOutput {
    let right_local = input.up_local.cross(input.forward_local);
    if right_local.length_squared() < MIN_BONE_LENGTH * MIN_BONE_LENGTH {
        return identity_aim_output();
    }
    let right_local = right_local.normalize();

    // -- Neck: figure out how far the head's bind forward would have
    //    to yaw + pitch in the neck's local frame to land on the
    //    target. Apply `neck_share` and clamp.
    let to_target_neck = input.target - input.neck_pos;
    if to_target_neck.length_squared() < MIN_BONE_LENGTH * MIN_BONE_LENGTH {
        return identity_aim_output();
    }
    let to_target_neck_world_dir = to_target_neck.normalize();
    let (neck_full_yaw, neck_full_pitch) = decompose_yaw_pitch(
        to_target_neck_world_dir,
        input.neck_world_rot,
        input.forward_local,
        input.up_local,
        right_local,
    );
    let neck_yaw = (neck_full_yaw * input.neck_share)
        .clamp(-input.neck_yaw_limit, input.neck_yaw_limit);
    let neck_pitch = (neck_full_pitch * input.neck_share)
        .clamp(-input.neck_pitch_limit, input.neck_pitch_limit);
    let neck_delta_local = compose_yaw_pitch(neck_yaw, neck_pitch, input.up_local, right_local);
    let neck_delta_world =
        input.neck_world_rot * neck_delta_local * input.neck_world_rot.inverse();

    // -- Head: residual yaw + pitch from the head's POST-neck local
    //    frame to the target. Clamp at the head's own limits.
    let head_world_rot_after_neck = neck_delta_world * input.head_world_rot;
    let head_pos_after_neck =
        input.neck_pos + neck_delta_world * (input.head_pos - input.neck_pos);
    let to_target_head = input.target - head_pos_after_neck;
    if to_target_head.length_squared() < MIN_BONE_LENGTH * MIN_BONE_LENGTH {
        return AimChainOutput {
            neck_delta_world,
            head_delta_world: Quat::IDENTITY,
        };
    }
    let to_target_head_world_dir = to_target_head.normalize();
    let (head_full_yaw, head_full_pitch) = decompose_yaw_pitch(
        to_target_head_world_dir,
        head_world_rot_after_neck,
        input.forward_local,
        input.up_local,
        right_local,
    );
    let head_yaw = head_full_yaw.clamp(-input.head_yaw_limit, input.head_yaw_limit);
    let head_pitch = head_full_pitch.clamp(-input.head_pitch_limit, input.head_pitch_limit);
    let head_delta_local = compose_yaw_pitch(head_yaw, head_pitch, input.up_local, right_local);
    let head_delta_world =
        head_world_rot_after_neck * head_delta_local * head_world_rot_after_neck.inverse();

    AimChainOutput {
        neck_delta_world,
        head_delta_world,
    }
}

#[inline]
fn identity_aim_output() -> AimChainOutput {
    AimChainOutput {
        neck_delta_world: Quat::IDENTITY,
        head_delta_world: Quat::IDENTITY,
    }
}

/// Project a world-space target direction into a bone's local frame
/// and split the required rotation into a (yaw, pitch) pair.
///
/// `yaw` is rotation around `up_local` taking `forward_local` toward
/// the projection of the target on the (forward, right) plane.
/// `pitch` is rotation around the right axis taking the projection
/// up toward the target's vertical component. Both are signed
/// radians: positive yaw = right-hand rule around `up_local`,
/// positive pitch = nose down (right-hand rule around right axis).
fn decompose_yaw_pitch(
    target_world_dir: Vec3,
    bone_world_rot: Quat,
    forward_local: Vec3,
    up_local: Vec3,
    right_local: Vec3,
) -> (f32, f32) {
    let target_local = bone_world_rot.inverse() * target_world_dir;
    // Components in the bone's local basis. `up_local`/`forward_local`/
    // `right_local` are unit and orthogonal so dot is the projection.
    let t_right = target_local.dot(right_local);
    let t_up = target_local.dot(up_local);
    let t_forward = target_local.dot(forward_local);
    // Yaw: angle in the (forward, right) plane from forward toward
    // the target. atan2 handles all four quadrants including the
    // antiparallel case (target = -forward → yaw = π).
    let yaw = t_right.atan2(t_forward);
    // Pitch: vertical lift of the target above the (forward, right)
    // plane. Positive `t_up` = target above; we encode that as
    // negative pitch (nose up) by convention so `compose_yaw_pitch`
    // applied to `forward_local` lands on `target_local`.
    let pitch = -t_up.clamp(-1.0, 1.0).asin();
    (yaw, pitch)
}

/// Rebuild a local-frame rotation quaternion from (yaw, pitch).
/// Pitch is applied first (around the right axis), then yaw (around
/// the up axis) — so when this rotates `forward_local` we recover
/// the originally-decomposed target direction (modulo clamps).
fn compose_yaw_pitch(yaw: f32, pitch: f32, up_local: Vec3, right_local: Vec3) -> Quat {
    let yaw_quat = Quat::from_axis_angle(up_local, yaw);
    let pitch_quat = Quat::from_axis_angle(right_local, pitch);
    yaw_quat * pitch_quat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() < eps
    }

    /// Apply a solve's world deltas to the original bone positions and
    /// return the new (root_pos, mid_pos, tip_pos). Useful in tests to
    /// verify the tip lands on the target.
    fn applied(input: TwoBoneInput, out: TwoBoneOutput) -> (Vec3, Vec3, Vec3) {
        let upper = input.mid - input.root;
        let lower = input.tip - input.mid;
        let new_mid = input.root + out.root_delta_world * upper;
        let new_tip = new_mid + (out.mid_delta_world * out.root_delta_world) * lower;
        (input.root, new_mid, new_tip)
    }

    #[test]
    fn target_at_full_reach_lands_within_clamp_band() {
        // Target exactly at the tip's bind position. The
        // `REACH_CLAMP = 0.999` deliberately prevents a fully
        // locked-out chain (numerical singularity at the joint
        // angles), so the new tip lands ~0.1 % short of full
        // reach — about 2 mm at a 2 m chain. Verify it's in that
        // band and the solver reports `reached`.
        let input = TwoBoneInput {
            root: Vec3::ZERO,
            mid: Vec3::new(0.0, 1.0, 0.0),
            tip: Vec3::new(0.0, 2.0, 0.0),
            target: Vec3::new(0.0, 2.0, 0.0),
            pole: Vec3::new(0.0, 1.0, 1.0),
        };
        let out = solve_two_bone_ik(input);
        assert!(out.reached);
        let (_, _, tip) = applied(input, out);
        let chain_reach = (input.mid - input.root).length() + (input.tip - input.mid).length();
        let expected_reach_clamp_gap = chain_reach * (1.0 - REACH_CLAMP);
        let actual_gap = (input.target - tip).length();
        assert!(
            actual_gap <= expected_reach_clamp_gap * 2.0,
            "tip gap {actual_gap} exceeded twice the clamp band {expected_reach_clamp_gap}"
        );
    }

    #[test]
    fn target_within_reach_lands_on_target() {
        // Hip at origin, knee straight down 1m, ankle straight down
        // 2m total. Move target 0.3m forward (knee should bend
        // backwards, but pole is forward → knee bends forward). The
        // new ankle should land at the target.
        let input = TwoBoneInput {
            root: Vec3::ZERO,
            mid: Vec3::new(0.0, -1.0, 0.0),
            tip: Vec3::new(0.0, -2.0, 0.0),
            target: Vec3::new(0.3, -1.8, 0.0),
            pole: Vec3::new(1.0, -1.0, 0.0), // forward + down
        };
        let out = solve_two_bone_ik(input);
        assert!(out.reached);
        let (_, _, tip) = applied(input, out);
        assert!(
            approx_eq(tip, input.target, 5.0e-3),
            "tip {tip:?} != target {:?}",
            input.target,
        );
    }

    #[test]
    fn unreachable_target_extends_chain() {
        // Target 5m away, chain reach is 2m → solver should extend
        // the chain toward the target and report `reached = false`.
        let input = TwoBoneInput {
            root: Vec3::ZERO,
            mid: Vec3::new(0.0, -1.0, 0.0),
            tip: Vec3::new(0.0, -2.0, 0.0),
            target: Vec3::new(5.0, -1.0, 0.0),
            pole: Vec3::new(1.0, -1.0, 0.0),
        };
        let out = solve_two_bone_ik(input);
        assert!(!out.reached, "target should be reported unreachable");
        let (_, _, tip) = applied(input, out);
        // Tip lies on the line from root to target, at distance ≈ chain reach.
        let tip_dist = tip.length();
        assert!(
            (tip_dist - 2.0 * REACH_CLAMP).abs() < 0.05,
            "tip dist {tip_dist} should be ~chain reach"
        );
        let target_dir = input.target.normalize();
        let tip_dir = tip.normalize();
        assert!(
            tip_dir.dot(target_dir) > 0.99,
            "tip should be along the line to target, got {tip_dir:?}"
        );
    }

    #[test]
    fn pole_keeps_knee_on_intended_side() {
        // Symmetric setup where the chain could bend either way.
        // Pole on +Z forces +Z bend; the new mid (knee) should have
        // a positive Z component.
        let input = TwoBoneInput {
            root: Vec3::ZERO,
            mid: Vec3::new(0.0, -1.0, 0.0),
            tip: Vec3::new(0.0, -2.0, 0.0),
            target: Vec3::new(0.0, -1.5, 0.0),
            pole: Vec3::new(0.0, 0.0, 1.0),
        };
        let out = solve_two_bone_ik(input);
        let (_, mid, _) = applied(input, out);
        assert!(
            mid.z > 0.05,
            "mid {mid:?} should have bent toward +Z pole"
        );

        // Flip pole to -Z — knee should go the other way.
        let input2 = TwoBoneInput {
            pole: Vec3::new(0.0, 0.0, -1.0),
            ..input
        };
        let out2 = solve_two_bone_ik(input2);
        let (_, mid2, _) = applied(input2, out2);
        assert!(
            mid2.z < -0.05,
            "mid {mid2:?} should have bent toward -Z pole"
        );
    }

    #[test]
    fn degenerate_zero_length_bone_returns_identity() {
        let input = TwoBoneInput {
            root: Vec3::ZERO,
            mid: Vec3::ZERO,
            tip: Vec3::new(0.0, -1.0, 0.0),
            target: Vec3::new(0.0, -0.5, 0.0),
            pole: Vec3::Z,
        };
        let out = solve_two_bone_ik(input);
        assert_eq!(out.root_delta_world, Quat::IDENTITY);
        assert_eq!(out.mid_delta_world, Quat::IDENTITY);
        assert!(!out.reached);
    }

    // -------------------------------------------------------------
    // Aim chain (Phase H) tests
    // -------------------------------------------------------------

    /// Project the solver's deltas onto the chain to recover the new
    /// head forward direction (what the head's "looking axis" points
    /// at after the IK is applied).
    fn applied_head_forward(input: AimChainInput, out: AimChainOutput) -> Vec3 {
        let new_head_world_rot = out.head_delta_world * out.neck_delta_world * input.head_world_rot;
        new_head_world_rot * input.forward_local
    }

    /// Helper: build a simple aim chain with neck at origin pointing
    /// `+Z`, head 0.2 m above pointing `+Z`, no rotation in either.
    fn upright_chain() -> AimChainInput {
        AimChainInput {
            neck_pos: Vec3::ZERO,
            neck_world_rot: Quat::IDENTITY,
            head_pos: Vec3::new(0.0, 0.2, 0.0),
            head_world_rot: Quat::IDENTITY,
            forward_local: Vec3::Z,
            up_local: Vec3::Y,
            target: Vec3::new(0.0, 0.2, 5.0),
            neck_share: 0.3,
            neck_yaw_limit: std::f32::consts::FRAC_PI_3,   // 60°
            neck_pitch_limit: std::f32::consts::FRAC_PI_3, // 60°
            head_yaw_limit: std::f32::consts::FRAC_PI_3,   // 60°
            head_pitch_limit: std::f32::consts::FRAC_PI_3, // 60°
        }
    }

    #[test]
    fn target_straight_ahead_produces_minimal_rotation() {
        let input = upright_chain();
        let out = solve_aim_chain(input);
        let fwd = applied_head_forward(input, out);
        let want = Vec3::Z;
        assert!(
            (fwd - want).length() < 1e-3,
            "forward {fwd:?} should still be ~+Z for straight-ahead target"
        );
    }

    #[test]
    fn target_to_the_side_aims_partway_toward_it() {
        // Target 90° to the right (+X). The chain should rotate
        // partially: neck does 0.3 × 90° = 27°, head does up to 60°
        // more. Total ≈ 87° (1° short of full because head is cone
        // capped at 60°).
        let input = AimChainInput {
            target: Vec3::new(5.0, 0.2, 0.0),
            ..upright_chain()
        };
        let out = solve_aim_chain(input);
        let fwd = applied_head_forward(input, out);
        let want = Vec3::X;
        // Angle between achieved forward and target direction:
        let achieved_dot = fwd.dot(want);
        // 27° + 60° = 87° ≈ cos 3° ≈ 0.998 from target — but the
        // post-neck head pivot moves slightly so allow some slack.
        assert!(
            achieved_dot > 0.99,
            "forward {fwd:?} should be near +X (target), dot={achieved_dot}"
        );
    }

    #[test]
    fn target_behind_caps_at_per_bone_limits() {
        // Target directly behind (-Z). Full turn = 180°. Neck does
        // 0.3 × 180° = 54° (under its 60° limit). Head residual is
        // 180° - 54° = 126°, clamped at 60°. Total turn = 54 + 60 =
        // 114° (cos ≈ -0.407). And it MUST be pure yaw — no random
        // up/down flip — because of the explicit yaw+pitch
        // decomposition in the bone's local frame.
        let input = AimChainInput {
            target: Vec3::new(0.0, 0.2, -5.0),
            ..upright_chain()
        };
        let out = solve_aim_chain(input);
        let fwd = applied_head_forward(input, out);
        let dot_with_initial = fwd.dot(Vec3::Z);
        assert!(
            (dot_with_initial + 0.407).abs() < 0.05,
            "forward {fwd:?} should be ~114° from +Z (cos ≈ -0.407), got dot {dot_with_initial}"
        );
        assert!(
            fwd.y.abs() < 0.05,
            "forward {fwd:?} should be pure yaw (Y ≈ 0)"
        );
    }

    #[test]
    fn neck_share_zero_makes_head_do_everything() {
        let input = AimChainInput {
            target: Vec3::new(2.0, 0.2, 2.0), // 45° to right
            neck_share: 0.0,
            ..upright_chain()
        };
        let out = solve_aim_chain(input);
        // Neck delta should be identity (or near-identity).
        let (_, neck_angle) = out.neck_delta_world.to_axis_angle();
        assert!(
            neck_angle.abs() < 1e-3,
            "neck rotated {neck_angle} rad despite share=0"
        );
    }

    #[test]
    fn target_above_yields_pure_pitch() {
        // Target directly above the head — should be pure pitch
        // rotation (no yaw component). Forward goes from +Z toward
        // +Y. Within neck (0.3 × 90° = 27°) + head (60° clamp) =
        // 87° pitch up.
        let input = AimChainInput {
            target: Vec3::new(0.0, 5.0, 0.0),
            ..upright_chain()
        };
        let out = solve_aim_chain(input);
        let fwd = applied_head_forward(input, out);
        // Pure pitch up: x ≈ 0.
        assert!(fwd.x.abs() < 0.05, "forward {fwd:?} should be pure pitch (x ≈ 0)");
        // 87° tilt up means Y ≈ sin(87°) ≈ 0.998, Z ≈ cos(87°) ≈ 0.052.
        assert!(
            fwd.y > 0.95,
            "forward {fwd:?} should pitch nearly straight up (y > 0.95)"
        );
    }

    #[test]
    fn degenerate_target_at_neck_returns_identity() {
        let input = AimChainInput {
            target: Vec3::ZERO,
            ..upright_chain()
        };
        let out = solve_aim_chain(input);
        assert_eq!(out.neck_delta_world, Quat::IDENTITY);
    }
}
