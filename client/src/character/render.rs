//! Skinned-mesh rendering of remote players.
//!
//! Phase C of the character plan. For every entry in
//! [`crate::remote::RemotePlayers`], spawn one skinned-mesh visual
//! entity parented under the [`ChunkSource`] for the shard that
//! player lives on. The visual's local `Transform` is the player's
//! shard-local pose, and [`crate::shard::ShardOriginPlugin`]'s
//! per-frame rebase composes it with the shard's system-space origin
//! to produce the final camera-relative Bevy world transform.
//!
//! # Why parent under the ChunkSource?
//!
//! The whole rendering pipeline already knows how to rebase
//! per-shard floating-origin transforms relative to the camera. By
//! parenting visuals under the appropriate `ChunkSource`, we get:
//!  - sub-mm precision at any planet scale (the small-magnitude
//!    `(origin - camera_world)` subtraction lives in f64),
//!  - correct ship interior rendering (visual rotation = ship_rot ×
//!    body_yaw_quat applied via Bevy's hierarchy multiplication),
//!  - free shadow-map / cull / IBL participation via `bevy_pbr`.
//!
//! # Bone resolution
//!
//! Bevy's `SceneRoot(handle)` materializes the glTF hierarchy
//! asynchronously. Phase C polls each visual entity's `Children` and
//! walks the hierarchy once when it appears, populating
//! [`BoneRegistry`] with `bone_name → Entity` mappings. Phases E
//! (camera-bone attachment) and G/H (IK) consume this map without
//! re-walking.
//!
//! # Class metadata
//!
//! Today the wire protocol does not carry a per-player `class_id`,
//! so every remote player is rendered as
//! [`voxeldust_core::character::HUMAN_DEFAULT`]. When a future
//! cosmetic phase adds a wire field, the spawn system reads it from
//! `RemoteEntity` and looks up the class via
//! [`voxeldust_core::character::class_by_id`] — no other change.

use std::collections::{HashMap, HashSet};

use bevy::animation::{AnimatedBy, AnimationPlayer, AnimationTargetId};
use bevy::animation::graph::AnimationGraphHandle;
use bevy::prelude::*;
use bevy::scene::SceneRoot;
use glam::{DVec3, Quat as GQuat};

use voxeldust_core::character::{class_by_id, CharacterClass, HUMAN_DEFAULT};

use crate::remote::{RemoteEntitiesSet, RemoteEntity, RemotePlayers};
use crate::shard::{ShardKey, SourceIndex};

use super::assets::{CharacterAssetRegistry, ClassAssets};

/// SystemSet for character rendering. Runs after
/// [`RemoteEntitiesSet`] (so the latest remote-player snapshot is
/// available) and after the asset registry has had a tick to advance
/// its load state.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterRenderSet;

/// Marker on a remote-player visual entity. Stores the player_id so
/// the per-frame sync system can correlate `RemoteEntity` → visual
/// `Transform`. `class_id` lets bone resolution + animation systems
/// look up class metadata without a separate query.
#[derive(Component, Clone, Copy, Debug)]
pub struct RemoteCharacterTag {
    pub player_id: u64,
    pub class_id: u16,
    /// Which shard this player lives on at spawn time. Visual is
    /// parented under that shard's `ChunkSource`; if the player
    /// migrates to a different shard the despawn → respawn cycle
    /// re-parents.
    pub source_shard: ShardKey,
}

/// Cached bone-name → Entity lookup, populated after Bevy
/// materializes the scene hierarchy. Phase E (camera attachment) and
/// G/H (IK) read this every frame; the resolution itself only walks
/// the hierarchy once per visual.
#[derive(Component, Default, Debug)]
pub struct BoneRegistry {
    pub bones: HashMap<String, Entity>,
    /// Sentinel — flips to `true` after the first hierarchy walk
    /// produces a non-empty map.
    pub resolved: bool,
}

impl BoneRegistry {
    /// Look up a bone by name. Returns `None` if the scene hasn't
    /// materialized yet OR the named bone isn't in this rig.
    #[inline]
    pub fn get(&self, name: &str) -> Option<Entity> {
        self.bones.get(name).copied()
    }
}

pub struct CharacterRenderPlugin;

impl Plugin for CharacterRenderPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Update,
            CharacterRenderSet
                .after(RemoteEntitiesSet)
                .after(super::CharacterAssetSet),
        )
        .add_systems(
            Update,
            (
                spawn_remote_character_visuals,
                despawn_remote_character_visuals,
                bevy::ecs::schedule::ApplyDeferred,
                // Phase T1 (seamless transitions): a player whose
                // shard changed (boarded a ship, EVA-exited, etc.) is
                // still present in `RemotePlayers` but their
                // `RemoteEntity.shard` flipped. Reparent the existing
                // visual under the new ChunkSource INSTEAD of
                // despawning + respawning, so AnimationPlayer,
                // LocomotionAnimState, BoneRegistry, HandBindData,
                // LocalCharacterTag, TabletIkBlendState all persist
                // across the shard boundary — no T-pose blink, no
                // bone-resolution stall, no camera-attachment severing.
                reparent_visuals_on_shard_change,
                resolve_bones_when_ready,
                sync_remote_pose,
            )
                .chain()
                .in_set(CharacterRenderSet),
        );
    }
}

/// Spawn one skinned-mesh visual per remote player not yet visible.
/// Reactive: re-runs each frame, no-ops when nothing new arrived.
///
/// Spawn requires three preconditions per player:
///   1. The player's class assets are in `Ready` state.
///   2. A `ChunkSource` exists for the shard the player lives on
///      (so we can parent under it for the floating-origin rebase).
/// If either is missing this frame we defer — the next tick will
/// retry once the asset / source materializes.
fn spawn_remote_character_visuals(
    mut commands: Commands,
    remote_players: Res<RemotePlayers>,
    registry: Res<CharacterAssetRegistry>,
    sources: Res<SourceIndex>,
    existing: Query<&RemoteCharacterTag>,
) {
    if remote_players.by_id.is_empty() {
        return;
    }

    let already_visualized: HashSet<u64> =
        existing.iter().map(|tag| tag.player_id).collect();

    for (&player_id, remote) in &remote_players.by_id {
        if already_visualized.contains(&player_id) {
            continue;
        }

        // Phase A: every remote player is HUMAN_DEFAULT. When the
        // wire protocol adds a `class_id` field per player, swap in
        // `class_by_id(remote.class_id).unwrap_or(&HUMAN_DEFAULT)`.
        let class: &CharacterClass = &HUMAN_DEFAULT;
        let assets = match registry.ready(class.id) {
            Some(a) => a,
            None => continue,
        };

        let Some(&parent) = sources.by_shard.get(&remote.shard) else {
            // ChunkSource for this player's shard not yet spawned —
            // happens during the brief window between secondary
            // connection and source materialization. Try again next
            // tick.
            continue;
        };

        spawn_one(&mut commands, parent, player_id, class, remote, assets);
    }
}

/// Spawn a single visual entity. Factored out so a future
/// `spawn_local_character_visual` (Phase E) can reuse the same
/// component bundle without copy-paste drift.
fn spawn_one(
    commands: &mut Commands,
    parent: Entity,
    player_id: u64,
    class: &CharacterClass,
    remote: &RemoteEntity,
    assets: &ClassAssets,
) {
    let initial_transform = local_transform_for(remote, class);
    let visual = commands
        .spawn((
            SceneRoot(assets.scene.clone()),
            initial_transform,
            GlobalTransform::default(),
            Visibility::default(),
            AnimationPlayer::default(),
            AnimationGraphHandle(assets.graph.clone()),
            RemoteCharacterTag {
                player_id,
                class_id: class.id,
                source_shard: remote.shard,
            },
            BoneRegistry::default(),
            // `Name` makes the visual easy to find in `bevy-inspector-egui`
            // and tracing logs; not load-bearing.
            Name::new(format!("character/{}/{}", class.name, player_id)),
        ))
        .id();
    commands.entity(parent).add_child(visual);

    info!(
        player_id,
        class = class.name,
        shard = %remote.shard,
        "character render: spawned visual"
    );
}

/// Despawn visuals whose player is no longer present in
/// `RemotePlayers` at all (left AOI, disconnected). Crucially, a
/// player whose `RemoteEntity.shard` *flipped* (= shard migration:
/// boarded a ship, EVA-exited, etc.) is STILL present in
/// `RemotePlayers` and must NOT be despawned here —
/// [`reparent_visuals_on_shard_change`] handles them by reparenting the
/// existing visual under the new shard's `ChunkSource` so all
/// per-visual state (animation, bones, IK, tablet, local tag) persists.
fn despawn_remote_character_visuals(
    mut commands: Commands,
    remote_players: Res<RemotePlayers>,
    visuals: Query<(Entity, &RemoteCharacterTag)>,
) {
    for (entity, tag) in &visuals {
        let still_present = remote_players.by_id.contains_key(&tag.player_id);
        if !still_present {
            commands.entity(entity).despawn();
            info!(
                player_id = tag.player_id,
                "character render: despawned visual (left AOI or disconnected)"
            );
        }
    }
}

/// Phase T1 — reparent a visual under the new ChunkSource when the
/// player's `RemoteEntity.shard` differs from the visual's
/// `tag.source_shard`. Preserves every per-visual component
/// (`AnimationPlayer`, `LocomotionAnimState`, `BoneRegistry`,
/// `HandBindData`, `TabletIkBlendState`, `LocalCharacterTag`, etc.)
/// across the shard boundary — the transition is invisible to the
/// renderer.
///
/// `commands.entity(parent).add_child(visual)` automatically removes
/// `visual` from any previous parent before attaching it to `parent`
/// (Bevy 0.18 hierarchy semantics), so this is a single atomic
/// reparent — no orphaned-frame.
///
/// `Transform.translation` is recomputed via [`local_transform_for`]
/// using the destination shard's `RemoteEntity` snapshot so the visual
/// renders in the correct world position on the very first frame
/// after the reparent (otherwise it would be interpreted in the new
/// parent's local frame with the *old* parent's coordinates and
/// snap on the next [`sync_remote_pose`]).
///
/// Defers to next tick if the destination ChunkSource has not yet
/// materialized (`SourceIndex.by_shard` miss): the visual stays under
/// the old parent — which is in the grace window so still
/// rendering — and reparents on the first tick after the new source
/// appears.
fn reparent_visuals_on_shard_change(
    mut commands: Commands,
    remote_players: Res<RemotePlayers>,
    asset_registry: Res<CharacterAssetRegistry>,
    sources: Res<SourceIndex>,
    mut visuals: Query<(Entity, &mut RemoteCharacterTag, &mut Transform)>,
) {
    for (visual, mut tag, mut transform) in &mut visuals {
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        if remote.shard == tag.source_shard {
            continue;
        }
        let Some(&new_parent) = sources.by_shard.get(&remote.shard) else {
            // Destination ChunkSource not yet spawned — the old parent
            // lives in the grace window so the visual stays visible
            // there. Try again next tick.
            continue;
        };
        let class = asset_registry
            .ready(tag.class_id)
            .map(|a| a.class)
            .or_else(|| voxeldust_core::character::class_by_id(tag.class_id));
        let Some(class) = class else {
            continue;
        };
        commands.entity(new_parent).add_child(visual);
        *transform = local_transform_for(remote, class);
        let old_shard = tag.source_shard;
        tag.source_shard = remote.shard;
        info!(
            player_id = tag.player_id,
            %old_shard,
            new_shard = %remote.shard,
            "character render: reparented visual on shard migration"
        );
    }
}

/// One-shot scene-hierarchy walk: when the spawned `SceneRoot`
/// materializes, find every `Name`d entity in its sub-tree, build the
/// `bone_name → Entity` map, validate against the class definition,
/// **manually attach the AnimationTargetId + AnimatedBy components
/// that bevy_gltf would have added if the rig glTF had its own
/// animations**, and flip `BoneRegistry.resolved`.
///
/// # Why we have to do this manually
///
/// `bevy_gltf` only attaches `AnimationTargetId` (the bone's UUID)
/// AND `AnimatedBy` (the back-pointer to the AnimationPlayer entity)
/// to a bone when it processes a glTF *that contains animations*.
/// Our split asset layout (rig in `y_bot.glb`, clips in separate
/// animation `.glb`s) means y_bot.glb has zero animations, so
/// neither component is added — Bevy's animation system has no UUID
/// to match curves against AND no link from a target to a player, so
/// the rig stays in T-pose despite the `AnimationPlayer` running
/// clips.
///
/// The retarget works because `AnimationTargetId::from_names` is a
/// deterministic UUID derived from the bone's name-path from the
/// scene root. Both y_bot.glb and idle.glb came through FBX2glTF
/// with the same Mixamo bone hierarchy
/// (`mixamorig:Hips/mixamorig:Spine/...`), so our hand-computed IDs
/// match the IDs idle.glb's clips reference — no per-clip remapping
/// code, just one pure function.
///
/// # Idempotence
///
/// Re-runs each frame for visuals whose registry isn't resolved yet
/// (cheap fast-path: skip the `if registry.resolved` arm). After
/// resolution this system is a no-op for that visual forever.
fn resolve_bones_when_ready(
    mut commands: Commands,
    mut visuals: Query<(Entity, &RemoteCharacterTag, &mut BoneRegistry, Option<&Children>)>,
    children_query: Query<&Children>,
    name_query: Query<&Name>,
) {
    for (visual_entity, tag, mut registry, children) in &mut visuals {
        if registry.resolved {
            continue;
        }
        let Some(top_children) = children else { continue };
        if top_children.is_empty() {
            continue;
        }

        let mut bones: HashMap<String, Entity> = HashMap::new();
        // (entity, AnimationTargetId computed from the bone's name
        // path) pairs to insert in a second pass. We collect during
        // the walk to avoid borrowing `commands` mutably while the
        // child-query iterator is live.
        let mut anim_targets: Vec<(Entity, AnimationTargetId)> = Vec::new();
        // BFS with per-entity name path. Each top-level child starts
        // a fresh path (matching bevy_gltf's `paths_recur` which
        // initialises path = [] for each scene-root node).
        let mut stack: Vec<(Entity, Vec<Name>)> = top_children
            .iter()
            .map(|e| (e, Vec::<Name>::new()))
            .collect();
        let mut sample_paths: Vec<String> = Vec::new();
        while let Some((e, parent_path)) = stack.pop() {
            let mut my_path = parent_path.clone();
            if let Ok(name) = name_query.get(e) {
                my_path.push(name.clone());
                bones.insert(name.as_str().to_string(), e);
                let id = AnimationTargetId::from_names(my_path.iter());
                anim_targets.push((e, id));
                if sample_paths.len() < 5 {
                    sample_paths.push(
                        my_path
                            .iter()
                            .map(|n| n.as_str())
                            .collect::<Vec<_>>()
                            .join("/"),
                    );
                }
            }
            if let Ok(c) = children_query.get(e) {
                for child in c.iter() {
                    stack.push((child, my_path.clone()));
                }
            }
        }

        if bones.is_empty() {
            // Hierarchy exists but no Name components — the scene
            // probably hasn't fully materialized yet (Bevy spawns
            // intermediate nodes in stages). Try again next tick.
            continue;
        }

        // Validate required bones per class metadata.
        if let Some(class) = class_by_id(tag.class_id) {
            let mut missing: Vec<&str> = Vec::new();
            for &b in &[
                class.head_bone,
                class.neck_bone,
                class.left_foot_bone,
                class.right_foot_bone,
            ] {
                if !bones.contains_key(b) {
                    missing.push(b);
                }
            }
            for b in class.fp_cull_bones {
                if !bones.contains_key(*b) && !missing.contains(b) {
                    missing.push(*b);
                }
            }
            if missing.is_empty() {
                info!(
                    player_id = tag.player_id,
                    class = class.name,
                    bones = bones.len(),
                    "character render: scene resolved, all expected bones present"
                );
            } else {
                warn!(
                    player_id = tag.player_id,
                    class = class.name,
                    bones = bones.len(),
                    missing = ?missing,
                    "character render: scene resolved but expected bones missing — \
                     camera-bone attachment / IK will fall back. Verify the rig matches \
                     CharacterClass bone-name constants."
                );
            }
        }

        // Insert the synthesised AnimationTargetId + AnimatedBy on
        // every named entity. The `AnimationPlayer` for this visual
        // lives on `visual_entity` (see `spawn_one`), so all bones
        // back-point there.
        let bound_count = anim_targets.len();
        for (bone_entity, target_id) in anim_targets {
            commands
                .entity(bone_entity)
                .insert((target_id, AnimatedBy(visual_entity)));
        }
        info!(
            player_id = tag.player_id,
            class_id = tag.class_id,
            bones_bound = bound_count,
            sample_paths = ?sample_paths,
            "character render: synthesized AnimationTargetId + AnimatedBy on {} bones \
             (sample paths logged so you can confirm they match the clips' targets)",
            bound_count
        );

        registry.bones = bones;
        registry.resolved = true;
    }
}

/// Per-frame: copy the latest `RemoteEntity` pose onto each visual's
/// local `Transform`. The shard's floating-origin rebase
/// (`ShardOriginPlugin`) composes the parent's f64 system-space pose
/// with the camera position; Bevy's hierarchy then multiplies parent
/// × visual.local for the final world transform.
///
/// Smoothing: today this is a direct copy; if 20 Hz network jitter
/// becomes visible we'll fold in interpolation against the existing
/// snapshot buffer (same pattern as the camera-pose smoother).
fn sync_remote_pose(
    remote_players: Res<RemotePlayers>,
    asset_registry: Res<CharacterAssetRegistry>,
    mut visuals: Query<(&RemoteCharacterTag, &mut Transform)>,
) {
    for (tag, mut transform) in &mut visuals {
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        // Class metadata: prefer the registry-resolved class (so
        // future per-class hot-reload can rebind), fall back to the
        // stable `class_by_id` lookup so a racing-load scenario
        // (Ready → !Ready) doesn't snap the transform to default.
        let class = asset_registry
            .ready(tag.class_id)
            .map(|a| a.class)
            .or_else(|| voxeldust_core::character::class_by_id(tag.class_id));
        let Some(class) = class else { continue };
        *transform = local_transform_for(remote, class);
    }
}

/// Compute a visual's shard-local Transform from the latest
/// `RemoteEntity` snapshot. Centralized so spawn + per-frame sync
/// agree byte-for-byte on the formula — a pure function makes the
/// "did spawn use a different rotation than sync?" class of bugs
/// impossible.
///
/// `class` supplies two per-class offsets so the renderer is rig-
/// agnostic:
///  * `visual_y_offset` shifts the mesh down by the capsule
///    half-extent (server's `position` is capsule centre, mesh
///    origin is feet — without the shift, characters float).
///  * `mesh_yaw_offset` rotates the mesh so its bind-pose facing
///    direction aligns with the server's body-yaw convention
///    (Mixamo glTF faces -Z at identity; KCC's yaw=0 = +X — so the
///    offset is -π/2).
///
/// Both fields live on [`voxeldust_core::character::CharacterClass`],
/// so a future race with a different rig pipeline (e.g. Blender
/// export with +Z forward, or a mesh whose origin is at the hips)
/// is one-const-edit away from rendering correctly.
///
/// Translation: `remote.position` is shard-local — the player on a
/// SHIP primary stores ship-local coords, on PLANET primary stores
/// planet-local coords. The y-offset shifts to feet-on-floor; the
/// shard-relative xy is passed through verbatim, with the parent
/// ChunkSource's transform supplying the shard-to-world conversion.
fn local_transform_for(
    remote: &RemoteEntity,
    class: &voxeldust_core::character::CharacterClass,
) -> Transform {
    let mut translation = dvec3_to_bevy(remote.position);
    translation.y += class.visual_y_offset;
    // Body yaw + mesh-bind alignment. Both rotate around the shard-
    // local Y axis (ship interior = ship's local up; planet =
    // flat-Y planet axis pre-tangent-frame rendering, see
    // `client/src/shard_types/planet.rs`).
    //
    // `body_yaw_sign` reconciles two opposing yaw conventions:
    //  * camera yaw / KCC yaw / `body_yaw`: +ve = turn RIGHT (CW
    //    from above; `forward_from_look(y, 0) = (cos y, 0, sin y)`
    //    rotates +X → +Z as yaw grows).
    //  * `Quat::from_rotation_y(angle)`: right-hand rule, +ve = CCW
    //    from above.
    // For Mixamo Y-bot the sign is -1 (= flip), so the visual body
    // rotates the SAME way the camera does. Future rigs override
    // per-class without renderer changes.
    let rotation =
        Quat::from_rotation_y(class.body_yaw_sign * remote.body_yaw + class.mesh_yaw_offset);
    Transform {
        translation,
        rotation,
        scale: Vec3::ONE,
    }
}

#[inline]
fn dvec3_to_bevy(v: DVec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

#[inline]
#[allow(dead_code)] // Reserved for Phase D animation driver.
fn dquat_to_bevy(q: glam::DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

#[allow(dead_code)] // Phase E will use the GQuat alias for camera bone composition.
type _ReservedAlias = GQuat;
