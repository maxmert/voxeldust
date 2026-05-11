//! Phase I — client-side ragdoll bone renderer.
//!
//! When a remote character is mid-ragdoll, the server streams per-
//! tick world-space bone transforms in `RemotePlayerView.ragdoll_bones`.
//! This system overrides the corresponding visual bones' local
//! `Transform`s so the avatar's skinned mesh follows the server-
//! authoritative physics. The `AnimationPlayer`'s output is the
//! prior-pass write; we land AFTER it (and after the head-additive
//! pass) so the override wins this frame.
//!
//! # Why an "override on top" instead of pausing AnimationPlayer
//!
//! Pausing the player is the textbook approach but requires per-
//! character state and a clean re-activate path when the ragdoll
//! ends. Overriding each frame is idempotent: the animation pose is
//! computed, then immediately replaced for the ragdoll-driven bones.
//! Bones the ragdoll spec doesn't cover (e.g. Spine1, finger bones,
//! sub-clavicles on rigs with more joints than our 13-body
//! skeleton) stay in their last animation pose, which reads as
//! "frozen mid-fall" — acceptable for a first ship, and easy to
//! improve later by enriching the skeleton spec.
//!
//! # Coordinate conversion
//!
//! Wire bone transforms arrive in the source shard's local frame —
//! the same frame the rapier physics body lives in. To set the
//! visual bone's LOCAL `Transform.rotation`/`.translation`
//! (relative to its parent bone) we:
//!  1. Convert shard-local → system-space (`shard.origin + shard.rotation * .`).
//!  2. Convert system-space → Bevy camera-relative (subtract
//!     `CameraWorldPos`).
//!  3. Read the parent bone's CURRENT world pose by manually walking
//!     up the bone hierarchy from the visual entity (same trick as
//!     foot IK + look-at IK — avoids the stale `GlobalTransform`
//!     trap, see [`walk_bone_world_pose`]).
//!  4. `local = parent_world_inverse * bone_world` for both
//!     translation and rotation.

use bevy::app::AnimationSystems;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use glam::{DQuat, DVec3};

use crate::remote::RemotePlayers;
use crate::shard::{CameraWorldPos, ShardOrigin};

use super::render::{BoneRegistry, RemoteCharacterTag};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterRagdollSet;

pub struct CharacterRagdollPlugin;

impl Plugin for CharacterRagdollPlugin {
    fn build(&self, app: &mut App) {
        // Run AFTER all the other PostUpdate animation passes
        // (animation sample, head additive, foot IK, look-at IK)
        // but BEFORE `TransformSystems::Propagate` so the override
        // folds into this frame's `GlobalTransform`.
        app.add_systems(
            PostUpdate,
            apply_ragdoll_bones
                .after(AnimationSystems)
                .after(super::CharacterCameraSet)
                .after(super::CharacterFootIkSet)
                .after(super::CharacterLookIkSet)
                .before(TransformSystems::Propagate)
                .in_set(CharacterRagdollSet),
        );
    }
}

fn apply_ragdoll_bones(
    visuals: Query<(&RemoteCharacterTag, &BoneRegistry)>,
    remote_players: Res<RemotePlayers>,
    sources: Res<crate::shard::SourceIndex>,
    shard_origins: Query<&ShardOrigin>,
    camera_world: Res<CameraWorldPos>,
    parents: Query<&ChildOf>,
    mut transforms: Query<&mut Transform>,
) {
    for (tag, bones) in &visuals {
        if !bones.resolved {
            continue;
        }
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        if remote.ragdoll_bones.is_empty() {
            continue;
        }
        let Some(&source_entity) = sources.by_shard.get(&remote.shard) else {
            continue;
        };
        let Ok(shard_origin) = shard_origins.get(source_entity) else {
            continue;
        };

        // Pre-compute each ragdoll bone's target Bevy-world pose
        // ONCE, before any of them have been written back. This
        // way the parent-walk in step 3 reads a consistent
        // pre-write skeleton — critical because we're about to
        // overwrite both translation AND rotation on potentially
        // many bones in this loop, and the children's parent-walk
        // would otherwise pick up half-overwritten values.
        struct Pending {
            bone_entity: Entity,
            world_pos: Vec3,
            world_rot: Quat,
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(remote.ragdoll_bones.len());
        for bone in &remote.ragdoll_bones {
            let Some(bone_e) = bones.get(bone.bone_name.as_str()) else {
                continue;
            };
            let pos_shard = DVec3::new(
                bone.translation.x as f64,
                bone.translation.y as f64,
                bone.translation.z as f64,
            );
            let pos_system = shard_origin.origin + shard_origin.rotation * pos_shard;
            let pos_relative = pos_system - camera_world.pos;
            let world_pos = Vec3::new(
                pos_relative.x as f32,
                pos_relative.y as f32,
                pos_relative.z as f32,
            );
            // Rapier sends a `glam::Quat` ↔ workspace `glam::Quat`
            // (both 0.29 here — Phase 0 widened our workspace glam,
            // so `RagdollBoneTransform.rotation` is already the
            // expected type). Compose with shard rotation in f64
            // first to avoid precision loss on far-from-origin
            // shards.
            let rot_shard = DQuat::from_xyzw(
                bone.rotation.x as f64,
                bone.rotation.y as f64,
                bone.rotation.z as f64,
                bone.rotation.w as f64,
            );
            let rot_system = shard_origin.rotation * rot_shard;
            let world_rot = Quat::from_xyzw(
                rot_system.x as f32,
                rot_system.y as f32,
                rot_system.z as f32,
                rot_system.w as f32,
            );
            pending.push(Pending {
                bone_entity: bone_e,
                world_pos,
                world_rot,
            });
        }

        // Apply each override.
        for entry in pending {
            // Parent's world pose via manual walk — read BEFORE
            // we write this bone's local Transform so we don't
            // see our own previous write.
            let parent_world = parents
                .get(entry.bone_entity)
                .ok()
                .map(|child_of| child_of.parent())
                .and_then(|parent_e| walk_bone_world_pose(parent_e, &parents, &transforms))
                .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
            let local_pos = parent_world.1.inverse() * (entry.world_pos - parent_world.0);
            let local_rot = parent_world.1.inverse() * entry.world_rot;
            if let Ok(mut tf) = transforms.get_mut(entry.bone_entity) {
                tf.translation = local_pos;
                tf.rotation = local_rot;
            }
        }
    }
}

/// Walk the parent chain accumulating local Transforms to recover
/// the bone's fresh world pose. Same trick as foot IK + look-at IK
/// uses — `GlobalTransform` is one frame stale here so we'd be
/// reading the previous frame's ragdoll output.
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
