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
}
