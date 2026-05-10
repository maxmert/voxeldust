//! Look-at IK — character heads + necks track an attention target.
//!
//! Phase H of the character system plan. The server broadcasts a
//! per-character `LookTarget` (the world-space position the
//! character's currently paying attention to — typically the
//! nearest other player inside its peripheral cone, see
//! `update_look_targets` on each shard). The client converts that
//! into the active shard's Bevy camera-relative frame, walks the
//! neck + head bones for their CURRENT animated world pose (not
//! `GlobalTransform` — same staleness trap that bit foot IK in
//! Phase G), and rotates them via the closed-form aim-chain solver
//! in `voxeldust_core::character::ik`.
//!
//! # Composition with the head additive layer
//!
//! `apply_head_local_rotation` (Phase E) already post-multiplies
//! the head bone with `(head_yaw, head_pitch)` for the local
//! player's first-person camera. The look-at IK runs AFTER that
//! pass, then layers ONE MORE rotation that aims the head bone at
//! the world target. For the LOCAL player there's no `LookTarget`
//! broadcast (the server doesn't echo your own attention back to
//! you yet), so look-at IK is a no-op locally — the additive
//! layer drives the head, exactly as before. For REMOTE players,
//! the server-supplied target overrides; the head visibly tracks
//! the world.
//!
//! # Why drive the visible head from a world target instead of
//! body-relative angles?
//!
//! `head_yaw + head_pitch` are body-relative — when the body
//! turns, the head's apparent direction drifts with it. A
//! world-space target stays anchored: a head looking at a fixed
//! point keeps looking at that point even as the character walks
//! past. This is the difference between "AI mannequin" and
//! "character noticed me."
//!
//! # Per-frame flow (per-character, per-bone)
//!
//! 1. Read `RemotePlayerView.look_target_delta`. `None` → skip.
//! 2. Resolve the active shard's `ShardOrigin`; compose the
//!    absolute system-space target = `position + delta`, then
//!    `origin + rotation * target_shard_local`.
//! 3. Convert to Bevy camera-relative by subtracting
//!    `CameraWorldPos`.
//! 4. Walk the neck + head bones (`walk_bone_world_pose`) for
//!    current world position + rotation.
//! 5. Solve the aim chain with the per-class yaw / pitch / share
//!    config.
//! 6. Apply the world-rotation deltas to LOCAL `Transform`s via
//!    the same conjugation pattern as foot IK
//!    (`local_inv * world_delta * local`).

use bevy::app::AnimationSystems;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use glam::{DVec3, Quat as GQuat, Vec3 as GVec3};

use voxeldust_core::character::{solve_aim_chain, AimChainInput};

use crate::remote::RemotePlayers;
use crate::shard::{CameraWorldPos, ShardOrigin};

use super::assets::CharacterAssetRegistry;
use super::camera_attach::CharacterCameraSet;
use super::render::{BoneRegistry, RemoteCharacterTag};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterLookIkSet;

pub struct CharacterLookIkPlugin;

impl Plugin for CharacterLookIkPlugin {
    fn build(&self, app: &mut App) {
        // Run after AnimationSystems (so we layer onto clip pose) AND
        // after CharacterCameraSet (which contains the head additive
        // pass — look-at IK comes on top of that). Before
        // TransformSystems::Propagate so the IK'd local rotations
        // fold into this frame's GlobalTransform.
        app.add_systems(
            PostUpdate,
            apply_look_ik
                .after(AnimationSystems)
                .after(CharacterCameraSet)
                .before(TransformSystems::Propagate)
                .in_set(CharacterLookIkSet),
        );
    }
}

fn apply_look_ik(
    visuals: Query<(Entity, &RemoteCharacterTag, &BoneRegistry)>,
    remote_players: Res<RemotePlayers>,
    asset_registry: Res<CharacterAssetRegistry>,
    camera_world: Res<CameraWorldPos>,
    shard_origins: Query<&ShardOrigin>,
    sources: Res<crate::shard::SourceIndex>,
    parents: Query<&ChildOf>,
    mut transforms: Query<&mut Transform>,
) {
    for (visual_e, tag, bones) in &visuals {
        if !bones.resolved {
            continue;
        }
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        let Some(delta_f32) = remote.look_target_delta else {
            continue;
        };
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;

        let Some(&source_entity) = sources.by_shard.get(&remote.shard) else {
            continue;
        };
        let Ok(shard_origin) = shard_origins.get(source_entity) else {
            continue;
        };

        // Reconstruct the target in Bevy camera-relative space:
        //   target_shard_local (DVec3) = position + delta
        //   target_system = shard.origin + shard.rotation * shard_local
        //   target_world  = target_system - camera_world.pos
        let target_shard_local =
            remote.position + DVec3::new(delta_f32.x as f64, delta_f32.y as f64, delta_f32.z as f64);
        let target_system = shard_origin.origin + shard_origin.rotation * target_shard_local;
        let target_world_dvec = target_system - camera_world.pos;
        let target_world = Vec3::new(
            target_world_dvec.x as f32,
            target_world_dvec.y as f32,
            target_world_dvec.z as f32,
        );

        let (Some(neck_e), Some(head_e)) = (
            bones.get(class.neck_bone),
            bones.get(class.head_bone),
        ) else {
            continue;
        };

        let Some((neck_pos, neck_world_rot)) = walk_bone_world_pose(neck_e, &parents, &transforms)
        else {
            continue;
        };
        let Some((head_pos, head_world_rot)) = walk_bone_world_pose(head_e, &parents, &transforms)
        else {
            continue;
        };

        let solve = solve_aim_chain(AimChainInput {
            neck_pos: bevy_to_g_vec3(neck_pos),
            neck_world_rot: bevy_to_g_quat(neck_world_rot),
            head_pos: bevy_to_g_vec3(head_pos),
            head_world_rot: bevy_to_g_quat(head_world_rot),
            // class.look_*_local are already workspace `glam::Vec3`
            // (= `GVec3`) — pass through without conversion.
            forward_local: class.look_forward_local,
            up_local: class.look_up_local,
            target: bevy_to_g_vec3(target_world),
            neck_share: class.look_neck_share,
            neck_yaw_limit: class.look_neck_yaw_limit,
            neck_pitch_limit: class.look_neck_pitch_limit,
            head_yaw_limit: class.look_head_yaw_limit,
            head_pitch_limit: class.look_head_pitch_limit,
        });

        let neck_delta = g_to_bevy_quat(solve.neck_delta_world);
        let head_delta = g_to_bevy_quat(solve.head_delta_world);

        // Apply NECK first — the head bone's parent rotation
        // changes by `neck_delta`, so we must compute the head's
        // post-neck world rotation to do the head's local
        // conjugation correctly.
        if let Ok(mut neck_tf) = transforms.get_mut(neck_e) {
            let delta_local = neck_world_rot.inverse() * neck_delta * neck_world_rot;
            neck_tf.rotation = neck_tf.rotation * delta_local;
        }
        let head_world_rot_after_neck = neck_delta * head_world_rot;
        if let Ok(mut head_tf) = transforms.get_mut(head_e) {
            let delta_local =
                head_world_rot_after_neck.inverse() * head_delta * head_world_rot_after_neck;
            head_tf.rotation = head_tf.rotation * delta_local;
        }
    }
}

/// Walk the parent chain from a bone up to its root entity, composing
/// **current** local Transforms to recover the bone's world-space pose.
///
/// Same rationale as the matching helper in `foot_ik` — Bevy
/// propagates `GlobalTransform` at the END of `PostUpdate`, so any
/// `GlobalTransform` we read in our PostUpdate IK system is from the
/// previous frame and reflects the previous frame's IK output.
/// Walking current `Transform`s gives the FRESH animated pose
/// (AnimationSystems already wrote each bone's `Transform` this
/// frame).
fn walk_bone_world_pose(
    bone: Entity,
    parents: &Query<&ChildOf>,
    transforms: &Query<&mut Transform>,
) -> Option<(Vec3, Quat)> {
    let mut pos = Vec3::ZERO;
    let mut rot = Quat::IDENTITY;
    let mut e = bone;
    loop {
        let Ok(local) = transforms.get(e) else {
            return None;
        };
        pos = local.translation + local.rotation * (local.scale * pos);
        rot = local.rotation * rot;
        match parents.get(e) {
            Ok(child_of) => e = child_of.parent(),
            Err(_) => return Some((pos, rot)),
        }
    }
}

#[inline]
fn bevy_to_g_vec3(v: Vec3) -> GVec3 {
    GVec3::new(v.x, v.y, v.z)
}

#[inline]
fn bevy_to_g_quat(q: Quat) -> GQuat {
    GQuat::from_xyzw(q.x, q.y, q.z, q.w)
}

#[inline]
fn g_to_bevy_quat(q: GQuat) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}
