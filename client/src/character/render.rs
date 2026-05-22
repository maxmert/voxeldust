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
use crate::shard::{CameraWorldPos, ShardKey, ShardOrigin, SourceIndex};

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
                .after(super::CharacterAssetSet)
                // CRITICAL: render after the camera-pose has been
                // updated for this frame, otherwise on a SHIP→EVA
                // transition `rebase_root_visuals` uses the previous
                // frame's `CameraWorldPos` (still anchored to ship-
                // local space) to compute the world delta for the
                // newly-root-rebased local visual whose
                // `RemoteEntity.position` just flipped to system-space.
                // The mismatch yields a visual rendered tens of metres
                // off from the camera for one frame — the "Ship→EVA
                // player blinks" symptom. Ordering after both
                // PlayerSyncSet (writes CameraWorldPos) and
                // ShardOriginSet (writes ChunkSource Transforms from
                // CameraWorldPos) guarantees the visual rebase sees a
                // consistent camera state.
                .after(crate::camera::PlayerSyncSet)
                .after(crate::shard::ShardOriginSet),
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
                // CRITICAL: ApplyDeferred BEFORE sync_remote_pose +
                // rebase_root_visuals. Reparent issues Commands
                // (insert/remove `RootRebaseVisual`, attach/detach
                // `ChildOf`) which are buffered. Without flushing
                // here, the next two systems see STALE component
                // state for the just-reparented visual:
                //   * `sync_remote_pose` writes the visual's
                //     Transform from the NEW shard's
                //     `RemoteEntity.position` (e.g. system-space
                //     ~3.5e9 m) while the visual is still parented
                //     under the OLD shard's ChunkSource — Bevy's
                //     hierarchy multiplies parent.world × local =
                //     garbage world position.
                //   * `rebase_root_visuals` queries
                //     `With<RootRebaseVisual>` to apply the f64 rebase
                //     — the marker hasn't been inserted yet so the
                //     just-detached visual is SKIPPED, leaving the
                //     garbage Transform in place for the full frame.
                // Result: a one-frame visible jump to wildly wrong
                // world coordinates on every SHIP→EVA / EVA→SHIP /
                // SHIP→SHIP transition. Flushing here ensures the
                // marker/parent commands take effect before either
                // downstream Transform writer runs.
                bevy::ecs::schedule::ApplyDeferred,
                resolve_bones_when_ready,
                sync_remote_pose,
                // Phase: AAA precision fix for SYSTEM-parented (root-
                // rebased) visuals — see `RootRebaseVisual`. Runs
                // after `sync_remote_pose` to overwrite the f32
                // transform that `local_transform_for` would have
                // written (large value, would cancel in Bevy hierarchy
                // multiplication) with an f64-computed `world − cam`
                // delta, then casts the SMALL result to f32. Visual
                // sits at the right world position with sub-cm
                // precision regardless of distance from the system
                // origin.
                rebase_root_visuals,
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
    shard_origins: Query<&ShardOrigin>,
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
        let Ok(parent_origin) = shard_origins.get(parent).map(|o| o.origin) else {
            // `SourceIndex.by_shard` is populated synchronously when a
            // shard connects, but the bundled `ShardOrigin` component
            // on the ChunkSource entity is queued via `Commands` and
            // only becomes visible to queries after the next
            // `ApplyDeferred`. A fallback to `DVec3::ZERO` here would
            // mis-classify a SHIP/PLANET parent (origin ~1e9–1e10 m)
            // as root-rebased (because `0.length() < 1.0`), pinning
            // the visual into `rebase_root_visuals`'s code path with
            // the wrong arithmetic — visuals end up ~1e9 m off-screen
            // until the player's `RemoteEntity.shard` changes. Defer
            // one tick instead; the component lands deterministically.
            continue;
        };

        spawn_one(&mut commands, parent, parent_origin, player_id, class, remote, assets);
    }
}

/// Marker for visuals whose authoritative shard sits at SYSTEM-space
/// origin (i.e. `ShardOrigin::origin ≈ DVec3::ZERO`). Their server-
/// sent `position` is full system-space (~1e9–1e10 m magnitude for a
/// typical orbital location), so the Bevy hierarchy multiplication
/// `parent.world + visual.local` cancels two large opposite-sign f32
/// values → ~32 m precision quantization at that scale → the visual
/// jitters in/out of the camera frustum every few client frames as
/// the camera moves smoothly under orbital velocity. AAA visual
/// quality requires sub-cm precision in the camera-near render
/// window.
///
/// We avoid the cancellation by detaching these visuals from Bevy's
/// hierarchy entirely (they live as root entities) and writing
/// their `Transform` each tick via [`rebase_root_visuals`] — that
/// computes `world_pos = remote.position − cam` in f64 ONCE, then
/// casts to f32. Single subtraction at the order-of-magnitude of
/// the result (small), not the order-of-magnitude of the inputs
/// (huge) — so the precision is bounded by the SMALL output's f32
/// representation (~1e-7 m at 10 m magnitude).
#[derive(Component)]
struct RootRebaseVisual;

/// Spawn a single visual entity. Factored out so a future
/// `spawn_local_character_visual` (Phase E) can reuse the same
/// component bundle without copy-paste drift.
fn spawn_one(
    commands: &mut Commands,
    parent: Entity,
    parent_origin: DVec3,
    player_id: u64,
    class: &CharacterClass,
    remote: &RemoteEntity,
    assets: &ClassAssets,
) {
    // High-precision rebase decision — see [`RootRebaseVisual`] for
    // why visuals whose authoritative shard has origin ≈ ZERO must
    // bypass the Bevy hierarchy multiplication. ~1 m epsilon picks
    // up SYSTEM (origin = DVec3::ZERO) but excludes any shard whose
    // origin actually tracks a celestial body (planet centre =
    // ~1e10 m, ship hull = ~1e9 m, etc.).
    let needs_root_rebase = parent_origin.length() < 1.0;
    let initial_transform = local_transform_for(remote, class);
    let mut entity_cmds = commands.spawn((
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
    ));
    if needs_root_rebase {
        entity_cmds.insert(RootRebaseVisual);
    }
    let visual = entity_cmds.id();
    if !needs_root_rebase {
        commands.entity(parent).add_child(visual);
    }

    info!(
        player_id,
        class = class.name,
        shard = %remote.shard,
        root_rebase = needs_root_rebase,
        "character render: spawned visual"
    );
}

/// Marker tracking the last frame a visual's player was seen in
/// `RemotePlayers`. Inserted lazily on first absence; cleared when
/// the player reappears. The despawn fires only after `DESPAWN_GRACE`
/// of continuous absence — protects against the 1-tick gap between
/// the OLD primary's last WorldState (which doesn't list the player
/// post-transition) and the NEW primary's first WorldState (which
/// does, but takes ~50 ms to arrive over UDP).
///
/// Without this grace, seamless ShardHandoff transitions despawned
/// the player visual the same frame the old primary's TCP closed,
/// then respawned it ~50 ms later when the new primary's first
/// WorldState landed. The respawn waited for asset+bone resolve,
/// producing a visible 1-3 frame blink + a fresh local-tag insertion
/// race that broke camera bone-following.
#[derive(Component)]
struct AbsentSince(std::time::Instant);

/// 500 ms is comfortably more than two server ticks (50 ms each) AND
/// the longest measured network jitter on the LAN dev cluster
/// (~80 ms p99). Past 500 ms we trust the absence is real
/// (disconnect / left AOI for good) and despawn.
const DESPAWN_GRACE: std::time::Duration = std::time::Duration::from_millis(500);

/// Despawn visuals whose player is no longer present in
/// `RemotePlayers` for longer than [`DESPAWN_GRACE`]. Crucially, a
/// player whose `RemoteEntity.shard` *flipped* (= shard migration:
/// boarded a ship, EVA-exited, etc.) is STILL present in
/// `RemotePlayers` and must NOT be despawned here —
/// [`reparent_visuals_on_shard_change`] handles them by reparenting the
/// existing visual under the new shard's `ChunkSource` so all
/// per-visual state (animation, bones, IK, tablet, local tag) persists.
fn despawn_remote_character_visuals(
    mut commands: Commands,
    remote_players: Res<RemotePlayers>,
    visuals: Query<(Entity, &RemoteCharacterTag, Option<&AbsentSince>)>,
) {
    let now = std::time::Instant::now();
    for (entity, tag, absent_since) in &visuals {
        let still_present = remote_players.by_id.contains_key(&tag.player_id);
        if still_present {
            // Reappeared (e.g., new primary's first WorldState
            // landed). Clear any pending absence marker so the
            // grace window restarts on the NEXT absence.
            if absent_since.is_some() {
                commands.entity(entity).remove::<AbsentSince>();
            }
            continue;
        }
        match absent_since {
            None => {
                // First frame of absence — start the grace timer.
                commands.entity(entity).insert(AbsentSince(now));
            }
            Some(start) => {
                if now.saturating_duration_since(start.0) >= DESPAWN_GRACE {
                    commands.entity(entity).despawn();
                    info!(
                        player_id = tag.player_id,
                        absent_ms = now.saturating_duration_since(start.0).as_millis() as u64,
                        "character render: despawned visual (left AOI or disconnected)"
                    );
                }
            }
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
    shard_origins: Query<&ShardOrigin>,
    mut visuals: Query<(
        Entity,
        &mut RemoteCharacterTag,
        &mut Transform,
        Option<&RootRebaseVisual>,
    )>,
) {
    for (visual, mut tag, mut transform, root_rebase) in &mut visuals {
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        if remote.shard == tag.source_shard {
            continue;
        }
        let Some(&new_parent) = sources.by_shard.get(&remote.shard) else {
            // Destination ChunkSource not yet spawned — the old parent
            // (or the root-rebased visual) stays visible. Try again
            // next tick.
            continue;
        };
        let class = asset_registry
            .ready(tag.class_id)
            .map(|a| a.class)
            .or_else(|| voxeldust_core::character::class_by_id(tag.class_id));
        let Some(class) = class else {
            continue;
        };
        // Pick the rendering mode for the NEW parent: root-rebased
        // (no parent, explicit f64 world rebase per frame) when the
        // new parent's shard sits at origin ≈ ZERO (SYSTEM); normal
        // Bevy-hierarchy parenting otherwise. The transition between
        // the two modes is what makes EVA→ship boarding (and the
        // reverse) precision-correct in both directions.
        let Ok(new_origin) = shard_origins.get(new_parent).map(|o| o.origin) else {
            // Same race as in `spawn_remote_character_visuals`:
            // `SourceIndex.by_shard` for the destination is updated
            // synchronously on shard connect, but the new ChunkSource
            // entity's `ShardOrigin` lands only after `ApplyDeferred`.
            // Defer the reparent one tick; a `DVec3::ZERO` fallback
            // would mis-flag a SHIP/PLANET destination as root-rebased
            // and detach the visual from its (correct) parent.
            continue;
        };
        let new_needs_root_rebase = new_origin.length() < 1.0;
        match (root_rebase.is_some(), new_needs_root_rebase) {
            (false, false) => {
                // Parented → parented: standard reparent.
                commands.entity(new_parent).add_child(visual);
                *transform = local_transform_for(remote, class);
            }
            (true, false) => {
                // Root-rebased → parented (e.g. EVA boards a ship).
                // Remove the marker, add as child of the new parent,
                // reset Transform to shard-local — `sync_remote_pose`
                // will write the shard-local pose next tick.
                commands.entity(visual).remove::<RootRebaseVisual>();
                commands.entity(new_parent).add_child(visual);
                *transform = local_transform_for(remote, class);
            }
            (false, true) => {
                // Parented → root-rebased (e.g. ground player
                // hull-exits into EVA). Detach from parent, add the
                // marker — `rebase_root_visuals` will write the
                // correct world-space transform next tick.
                commands.entity(visual).remove_parent_in_place();
                commands.entity(visual).insert(RootRebaseVisual);
            }
            (true, true) => {
                // Root-rebased → root-rebased (rare: SYSTEM →
                // SYSTEM at different seed). No parent change; the
                // marker stays; `rebase_root_visuals` will write the
                // correct transform next tick from the new shard's
                // `RemoteEntity` snapshot.
            }
        }
        let old_shard = tag.source_shard;
        tag.source_shard = remote.shard;
        debug!(
            player_id = tag.player_id,
            %old_shard,
            new_shard = %remote.shard,
            new_root_rebase = new_needs_root_rebase,
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
    primary_ws: Res<crate::shard::worldstate::PrimaryWorldState>,
    secondary_ws: Res<crate::shard::worldstate::SecondaryWorldStates>,
    mut visuals: Query<(
        &RemoteCharacterTag,
        Option<&crate::character::camera_attach::LocalCharacterTag>,
        &mut Transform,
    )>,
) {
    // Render remote players at a fixed interpolation lag — the 20 Hz
    // server tick step turns into continuous motion on a 60+ fps
    // client. Local player skipped (no client-prediction policy: own
    // avatar feels snappy, even at the cost of one tick of "behind
    // the world" feel).
    //
    // Indexed on server-authoritative `game_time` (not wall-clock).
    // Same temporal reference as `CameraWorldPos::pos_at_game_time`
    // used by `rebase_root_visuals` — keeping ONE temporal paradigm
    // in the render pipeline. For parented visuals (same-shard) this
    // is behaviour-neutral vs the previous wall-clock indexing
    // (snapshots arrive together with `cam.pos` so the lerp window
    // is consistent either way); for root-rebased cross-shard
    // visuals the game-time indexing is load-bearing.
    let target = render_target_game_time(&primary_ws, &secondary_ws);
    for (tag, local, mut transform) in &mut visuals {
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        // Skip pose sync if the visual's parenting / coordinate rebase is not yet
        // aligned with the remote entity's active shard. This prevents coordinate
        // frame leakage / wild teleporting before reparent_visuals_on_shard_change completes.
        if remote.shard != tag.source_shard {
            continue;
        }
        // Class metadata: prefer the registry-resolved class (so
        // future per-class hot-reload can rebind), fall back to the
        // stable `class_by_id` lookup so a racing-load scenario
        // (Ready → !Ready) doesn't snap the transform to default.
        let class = asset_registry
            .ready(tag.class_id)
            .map(|a| a.class)
            .or_else(|| voxeldust_core::character::class_by_id(tag.class_id));
        let Some(class) = class else { continue };
        if local.is_some() {
            *transform = local_transform_for(remote, class);
        } else {
            let pos = match target {
                Some(t) => remote.interpolated_pose_at_game_time(t).0,
                // First-frame fallback before any WS has landed:
                // current snapshot. Identical to a degenerate lerp.
                None => remote.position,
            };
            // body_yaw is still applied via `local_transform_for`'s
            // scalar formula; rotation interpolation isn't currently
            // surfaced because the visual rotation is reconstructed
            // from `body_yaw` + class offset rather than written
            // from `remote.rotation`. Future polish phase can
            // interpolate body_yaw scalar (with wrap-aware lerp)
            // when ships start broadcasting rolled body frames for
            // ground walkers.
            *transform = local_transform_with_position(remote, class, pos);
        }
    }
}

/// Shared render-target game-time used by both `sync_remote_pose`
/// (parented visuals) and `rebase_root_visuals` (root-rebased
/// cross-shard visuals): `resolve_game_time_now(...) −
/// INTERPOLATION_DELAY`. The 50 ms (= one server tick @ 20 Hz)
/// back-lag absorbs same-shard tick jitter and gives a
/// secondary-shard's WS time to land before its entities are
/// rendered against the primary's `cam.pos`. Returns `None` only
/// when no WS — primary or secondary — has been received yet;
/// callers should fall back to the current snapshot in that case.
fn render_target_game_time(
    primary_ws: &crate::shard::worldstate::PrimaryWorldState,
    secondary_ws: &crate::shard::worldstate::SecondaryWorldStates,
) -> Option<f64> {
    crate::shard::worldstate::resolve_game_time_now(primary_ws, secondary_ws)
        .map(|now| now - crate::remote::INTERPOLATION_DELAY.as_secs_f64())
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
/// Phase T7 — same Transform formula as [`local_transform_for`] but
/// with the translation overridden to `position_override`. Used by
/// `sync_remote_pose` for remote players where the position comes
/// from `RemoteEntity::interpolated_pose` instead of the most-recent
/// snapshot's `position`. Body yaw and the per-class mesh-bind /
/// y-offset constants are identical to the non-interpolated path.
/// Phase: AAA precision rebase for [`RootRebaseVisual`] visuals
/// (visuals whose authoritative shard has origin ≈ ZERO).
///
/// `sync_remote_pose` writes `Transform.translation = remote.position`
/// for all visuals; for parented visuals that's the SHARD-LOCAL value
/// and Bevy hierarchy composes it with the parent's f32 world
/// transform correctly. For root-rebased visuals there is no parent,
/// so we must produce the WORLD transform directly — and crucially,
/// compute it from the f64 inputs (`remote.position`, `cam`) so the
/// large-magnitude inputs cancel ONCE in f64 (precise to f64 epsilon
/// at any magnitude), and only the small result is cast to f32
/// (where it's precise to ~1e-7 m at 10 m magnitude).
///
/// Without this system, root-rebased visuals would inherit whatever
/// `sync_remote_pose` wrote — `remote.position` in f32 — which at
/// system-space scales (~3.5e9 m) has ~32 m granularity. The camera
/// is also at that scale; their f32 difference jitters by tens of
/// metres frame-to-frame as the camera moves under orbital velocity,
/// strobing the visual in and out of the camera frustum.
fn rebase_root_visuals(
    remote_players: Res<RemotePlayers>,
    asset_registry: Res<CharacterAssetRegistry>,
    cam: Res<CameraWorldPos>,
    primary_ws: Res<crate::shard::worldstate::PrimaryWorldState>,
    secondary_ws: Res<crate::shard::worldstate::SecondaryWorldStates>,
    mut visuals: Query<(&RemoteCharacterTag, &mut Transform), With<RootRebaseVisual>>,
) {
    // Cross-shard render: `cam.pos` is from PRIMARY WS, `remote.position`
    // is from a SECONDARY WS — independent 20 Hz streams. Naive
    // `remote − cam` reads operands from different simulation instants
    // and oscillates by `velocity × tick_phase_offset` (≈ 1.5–4 km
    // per frame at orbital scales). Game-time-indexed interpolation
    // evaluates BOTH operands at the same `target` server-time so
    // the subtraction lands on a single simulation instant.
    //
    // Additionally, even without cross-shard mismatch, the lagged
    // `interpolated_pose` (50 ms back) vs un-lagged `cam.pos`
    // produced a steady `velocity × 50 ms` offset. Both sides now
    // use the same back-lag via `pos_at_game_time` / `interpolated_pose_at_game_time`.
    let target = render_target_game_time(&primary_ws, &secondary_ws);
    for (tag, mut transform) in &mut visuals {
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        // Skip rebase if the visual is in the middle of a shard migration and
        // hasn't had its rebase marker or parent updated yet.
        if remote.shard != tag.source_shard {
            continue;
        }
        let class = asset_registry
            .ready(tag.class_id)
            .map(|a| a.class)
            .or_else(|| voxeldust_core::character::class_by_id(tag.class_id));
        let Some(class) = class else { continue };
        let (pos_f64, cam_pos) = match target {
            Some(t) => (
                remote.interpolated_pose_at_game_time(t).0,
                cam.pos_at_game_time(t),
            ),
            // First-frame fallback before any WS has landed: current
            // snapshots on both sides. Equivalent to the pre-fix
            // direct subtraction; degenerate cases (single-frame
            // boot) where there's no temporal misalignment to begin
            // with.
            None => (remote.position, cam.pos),
        };
        // Single f64 subtraction at the order of magnitude of the
        // result (small) — not the magnitude of the operands (huge)
        // — so the cast to f32 only loses precision at the small
        // output's scale.
        let delta = pos_f64 - cam_pos;
        let mut translation = Vec3::new(delta.x as f32, delta.y as f32, delta.z as f32);
        translation.y += class.visual_y_offset;
        let rotation = Quat::from_rotation_y(
            class.body_yaw_sign * remote.body_yaw + class.mesh_yaw_offset,
        );
        *transform = Transform {
            translation,
            rotation,
            scale: Vec3::ONE,
        };
    }
}

fn local_transform_with_position(
    remote: &RemoteEntity,
    class: &voxeldust_core::character::CharacterClass,
    position_override: DVec3,
) -> Transform {
    let mut translation = dvec3_to_bevy(position_override);
    translation.y += class.visual_y_offset;
    let rotation =
        Quat::from_rotation_y(class.body_yaw_sign * remote.body_yaw + class.mesh_yaw_offset);
    Transform {
        translation,
        rotation,
        scale: Vec3::ONE,
    }
}

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
