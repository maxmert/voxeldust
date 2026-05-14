//! Per-character bind-pose extraction.
//!
//! Reads the rig's bind matrices (via Bevy's `SkinnedMeshInverseBindposes`
//! asset) once per visual at scene-resolution time and stores the
//! per-bone bind world rotation in the visual's local frame on a
//! [`HandBindData`] component.
//!
//! # Why this exists
//!
//! Tablet IK and finger curl need to know how each bone's local axes
//! map to body-local directions at bind. Hardcoding that ("bone +Y =
//! finger, bone +X = thumb, …") works only if the rig follows a
//! specific convention — and Mixamo's actual axis layout is rig-
//! specific (bone +X is the *bend* axis, not the *thumb* axis, on
//! Y-bot). The right answer is to read the bind matrices at runtime
//! and derive every axis from data instead of guessing per rig.
//!
//! # Where this runs
//!
//! `populate_hand_bind_data` runs in `Update` after
//! [`crate::character::CharacterRenderSet`] (so bones are resolved)
//! and before [`crate::character::tablet_ik::CharacterTabletIkSet`].
//! It is one-shot per visual: the `Without<HandBindData>` query filter
//! makes it a no-op once the component is inserted.

use std::collections::HashMap;

use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::prelude::*;

use voxeldust_core::character::class_by_id;

use super::render::{BoneRegistry, RemoteCharacterTag};

/// Bind-pose orientations for the bones tablet IK and finger curl
/// need, expressed in the visual root's local frame (= body local).
///
/// Each `Quat` is the rotation that maps **bone-local axes → body-
/// local directions** at bind. So `left_hand * Vec3::Y` = the world-
/// space direction the LEFT hand bone's local +Y axis points at bind,
/// expressed in body local. For Mixamo Y-bot LEFT hand in the
/// palms-down T-pose this is approximately body-LEFT (-X mesh).
#[derive(Component, Debug, Clone)]
pub struct HandBindData {
    pub left_hand: Quat,
    pub right_hand: Quat,
    /// Per-finger-bone bind rotation in body local. Keyed by Mixamo
    /// bone name (matches `class.tablet_left_hand_curl_bones`).
    pub left_finger_bones: HashMap<String, Quat>,
    /// Position of the LEFT index fingertip in the LEFT-hand
    /// (wrist) bone's LOCAL frame at bind. Used by tablet IK to
    /// compute the wrist target so the actual fingertip — not the
    /// wrist — lands at the focus cursor's UV. Reads from the
    /// rig's bind matrices via `inverse_bindposes`, so it tracks
    /// the actual rig geometry instead of relying on a 1D length
    /// estimate (which ignores the lateral offset between the
    /// wrist's +Y axis and the index finger).
    pub left_index_tip_offset_in_wrist_local: Vec3,
}

/// SystemSet for bind-pose extraction. Placed between bone resolution
/// and any IK/curl pass that needs `HandBindData`.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterBindPoseSet;

pub struct CharacterBindPosePlugin;

impl Plugin for CharacterBindPosePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            populate_hand_bind_data
                .in_set(CharacterBindPoseSet)
                .after(super::CharacterAssetSet)
                .after(super::render::CharacterRenderSet),
        );
    }
}

/// Walk descendants of `from` until an entity with `SkinnedMesh` is
/// found; return the entity + a borrow of its `SkinnedMesh`.
fn find_skinned_mesh<'a>(
    from: Entity,
    children_q: &'a Query<&Children>,
    skin_q: &'a Query<&SkinnedMesh>,
) -> Option<(Entity, &'a SkinnedMesh)> {
    let Ok(top) = children_q.get(from) else {
        return None;
    };
    let mut stack: Vec<Entity> = top.iter().collect();
    while let Some(e) = stack.pop() {
        if let Ok(sm) = skin_q.get(e) {
            return Some((e, sm));
        }
        if let Ok(c) = children_q.get(e) {
            for ch in c.iter() {
                stack.push(ch);
            }
        }
    }
    None
}

/// Compose all `Transform.rotation`s on the parent chain from `from`
/// up to (but not including) `to`. The result `R` satisfies
/// `R * v_in_from_local = v_in_to_local`.
fn walk_parent_chain_rot(
    from: Entity,
    to: Entity,
    transforms: &Query<&Transform>,
    parents: &Query<&ChildOf>,
) -> Option<Quat> {
    let mut e = from;
    let mut rot = Quat::IDENTITY;
    while e != to {
        let local = transforms.get(e).ok()?;
        // Closer-to-`to` rotations multiply on the LEFT, since each
        // step takes a vector from the current entity's local frame
        // up to its parent's local frame.
        rot = local.rotation * rot;
        e = parents.get(e).ok()?.parent();
    }
    Some(rot)
}

/// One-shot per visual: read the rig's `SkinnedMeshInverseBindposes`
/// asset, extract the bind world rotation for each bone of interest
/// in body local, and insert [`HandBindData`].
fn populate_hand_bind_data(
    mut commands: Commands,
    visuals: Query<(Entity, &RemoteCharacterTag, &BoneRegistry), Without<HandBindData>>,
    children_q: Query<&Children>,
    name_q: Query<&Name>,
    skin_q: Query<&SkinnedMesh>,
    transforms_q: Query<&Transform>,
    parents_q: Query<&ChildOf>,
    bindposes_assets: Res<Assets<SkinnedMeshInverseBindposes>>,
) {
    for (visual_e, tag, bones) in &visuals {
        if !bones.resolved {
            continue;
        }
        let Some(class) = class_by_id(tag.class_id) else {
            continue;
        };

        let Some((sm_e, sm)) = find_skinned_mesh(visual_e, &children_q, &skin_q) else {
            continue;
        };
        let Some(bindposes) = bindposes_assets.get(&sm.inverse_bindposes) else {
            // Asset not yet loaded — try again next tick.
            continue;
        };

        // Build a (bone_name → joint_index) map by walking sm.joints.
        let mut name_to_idx: HashMap<&str, usize> = HashMap::new();
        for (i, &joint_e) in sm.joints.iter().enumerate() {
            if let Ok(name) = name_q.get(joint_e) {
                name_to_idx.insert(name.as_str(), i);
            }
        }

        let Some(&li) = name_to_idx.get(class.left_hand_bone) else {
            continue;
        };
        let Some(&ri) = name_to_idx.get(class.right_hand_bone) else {
            continue;
        };

        let Some(mesh_to_visual_rot) =
            walk_parent_chain_rot(sm_e, visual_e, &transforms_q, &parents_q)
        else {
            continue;
        };

        // `inverse_bindposes[i]` maps mesh-local → joint-local at bind.
        // Its inverse = joint's bind world matrix in mesh space. Rotation
        // part = bone-local → mesh-local at bind. Then mesh-local →
        // body-local via mesh_to_visual_rot.
        let bone_in_visual = |idx: usize| -> Option<Quat> {
            let bind_in_mesh = bindposes.get(idx)?.inverse();
            let (_, rot_in_mesh, _) = bind_in_mesh.to_scale_rotation_translation();
            Some(mesh_to_visual_rot * rot_in_mesh)
        };

        let Some(left_in_visual) = bone_in_visual(li) else {
            continue;
        };
        let Some(right_in_visual) = bone_in_visual(ri) else {
            continue;
        };

        let mut left_finger_bones: HashMap<String, Quat> = HashMap::new();
        for &name in class.tablet_left_hand_curl_bones {
            let Some(&idx) = name_to_idx.get(name) else {
                continue;
            };
            let Some(rot) = bone_in_visual(idx) else {
                continue;
            };
            left_finger_bones.insert(name.to_string(), rot);
        }

        // Find the LEFT index fingertip bone — Mixamo rigs may
        // terminate the index chain at Index4, Index3, or
        // Index_End depending on FBX2glTF export settings. Try the
        // standard candidate names in deepest-first order; fall
        // back to (0, 0.13, 0) along bone +Y if none are present.
        let tip_candidates = [
            "mixamorig:LeftHandIndex4",
            "mixamorig:LeftHandIndex_End",
            "mixamorig:LeftHandIndex3",
            "mixamorig:LeftHandIndex2",
            "mixamorig:LeftHandIndex1",
        ];
        let left_index_tip_offset_in_wrist_local = (|| {
            let wrist_bind_in_mesh = bindposes.get(li)?.inverse();
            for &name in &tip_candidates {
                if let Some(&tip_idx) = name_to_idx.get(name) {
                    let tip_bind_in_mesh = bindposes.get(tip_idx)?.inverse();
                    // Tip pose in wrist's local frame = wrist^-1 *
                    // tip. Translation = offset from wrist origin
                    // to tip origin in wrist-local axes.
                    let tip_in_wrist = wrist_bind_in_mesh.inverse() * tip_bind_in_mesh;
                    let (_, _, trans) = tip_in_wrist.to_scale_rotation_translation();
                    info!(
                        bone = name,
                        offset = ?(trans.x, trans.y, trans.z),
                        "left index fingertip offset (in wrist local) from rig data"
                    );
                    return Some(trans);
                }
            }
            None
        })()
        .unwrap_or(Vec3::new(0.0, 0.13, 0.0));

        let l_y = left_in_visual * Vec3::Y;
        let l_x = left_in_visual * Vec3::X;
        let l_z = left_in_visual * Vec3::Z;
        let r_y = right_in_visual * Vec3::Y;
        let r_x = right_in_visual * Vec3::X;
        let r_z = right_in_visual * Vec3::Z;
        info!(
            player_id = tag.player_id,
            l_quat = ?(left_in_visual.x, left_in_visual.y, left_in_visual.z, left_in_visual.w),
            l_bone_x_in_body = ?(l_x.x, l_x.y, l_x.z),
            l_bone_y_in_body = ?(l_y.x, l_y.y, l_y.z),
            l_bone_z_in_body = ?(l_z.x, l_z.y, l_z.z),
            r_quat = ?(right_in_visual.x, right_in_visual.y, right_in_visual.z, right_in_visual.w),
            r_bone_x_in_body = ?(r_x.x, r_x.y, r_x.z),
            r_bone_y_in_body = ?(r_y.x, r_y.y, r_y.z),
            r_bone_z_in_body = ?(r_z.x, r_z.y, r_z.z),
            mesh_to_visual_y = ?{
                let v = mesh_to_visual_rot * Vec3::Y;
                (v.x, v.y, v.z)
            },
            "hand bind data extracted (bone +Y is along bone, body coords: +Z forward, +Y up, +X right)"
        );

        commands.entity(visual_e).insert(HandBindData {
            left_hand: left_in_visual,
            right_hand: right_in_visual,
            left_finger_bones,
            left_index_tip_offset_in_wrist_local,
        });
    }
}
