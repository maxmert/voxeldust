//! Mechanical sub-grids (rotors / pistons / hinges / sliders).
//!
//! Shard-agnostic: every shard that broadcasts `SubGridTransform` +
//! `SubGridAssignmentUpdate` creates its own child entities under its
//! `ChunkSource`. Rotors on a ship, rotors on a planet turret, and
//! future debris mechanical parts all share this code.
//!
//! Phase 16 MVP:
//!   * `SubGridAssignmentUpdate` → store the `block_pos → sub_grid_id`
//!     map per shard.
//!   * `SubGridTransform` per tick → spawn / update a sub-grid entity
//!     as a child of the owning shard's `ChunkSource`. The entity
//!     carries a Bevy `Transform` driven from
//!     `(translation, rotation)` so every frame the server's
//!     authoritative pose lands on Bevy's scene graph.
//!
//! Deferred to a later increment (and not gating first-playable):
//!   * Splitting chunk meshes by sub-grid assignment so blocks
//!     belonging to a rotor head re-parent under the SubGrid entity
//!     and therefore rotate with it. Today the chunks still render
//!     under the root `ChunkSource`; the SubGrid entity exists and
//!     its transform is live, but visible rotor animation requires
//!     the mesh-split pass.

use std::collections::HashMap;

use bevy::prelude::*;

use voxeldust_core::client_message::SubGridTransformData;

use crate::net::{GameEvent, NetEvent};
use crate::shard::{
    PrimaryShard, PrimaryWorldState, Secondaries, ShardKey, ShardRegistrySet, SourceIndex,
    WorldStateIngestSet,
};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SubGridSet;

/// Marker component on each sub-grid entity, parented under the shard's
/// `ChunkSource`. Carries the identifying `(shard, sub_grid_id)` so
/// raycast / highlight / future mesh-split can look them up.
#[derive(Component, Debug, Clone, Copy)]
pub struct SubGrid {
    pub shard: ShardKey,
    pub sub_grid_id: u32,
}

/// `(ShardKey, block_pos) → sub_grid_id` lookup. Populated from
/// `SubGridAssignmentUpdate` (primary) and
/// `SecondarySubGridAssignment` events. Used by future mesh-split +
/// raycast refinement to know which blocks belong to which sub-grid.
#[derive(Resource, Default)]
pub struct SubGridAssignments {
    pub by_shard: HashMap<ShardKey, HashMap<IVec3, u32>>,
}

/// Per-sub-grid live transform state, keyed by `(ShardKey, sub_grid_id)`.
/// Updated every frame from WorldState broadcasts; consumed by the
/// transform-apply system.
#[derive(Resource, Default)]
pub struct SubGridTransforms {
    pub by_shard: HashMap<(ShardKey, u32), SubGridTransformData>,
}

/// Entity index: `(ShardKey, sub_grid_id) → Entity` for fast lookup.
#[derive(Resource, Default)]
pub struct SubGridIndex {
    pub entries: HashMap<(ShardKey, u32), Entity>,
}

pub struct SubGridPlugin;

impl Plugin for SubGridPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SubGridAssignments>()
            .init_resource::<SubGridTransforms>()
            .init_resource::<SubGridIndex>()
            .configure_sets(
                Update,
                SubGridSet
                    .after(WorldStateIngestSet)
                    .after(ShardRegistrySet),
            )
            .add_systems(
                Update,
                (
                    ingest_assignments,
                    ingest_transforms,
                    spawn_or_update_sub_grids,
                    gc_despawned_shards,
                )
                    .chain()
                    .in_set(SubGridSet),
            );
    }
}

/// Drain primary + secondary `SubGridAssignmentUpdate` events into
/// the per-shard assignment map. Resolves shard key via
/// `PrimaryShard` / `Secondaries` on the fly.
fn ingest_assignments(
    mut events: MessageReader<GameEvent>,
    primary: Res<PrimaryShard>,
    secondaries: Res<Secondaries>,
    mut assignments: ResMut<SubGridAssignments>,
) {
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::SubGridAssignmentUpdate(data) => {
                let Some(key) = primary.current else { continue };
                apply_assignments(&mut assignments, key, &data.assignments);
            }
            NetEvent::SecondarySubGridAssignment { seed, data } => {
                let Some(key) = secondaries
                    .runtimes
                    .keys()
                    .find(|k| k.seed == *seed)
                    .copied()
                else {
                    continue;
                };
                apply_assignments(&mut assignments, key, &data.assignments);
            }
            _ => {}
        }
    }
}

fn apply_assignments(
    store: &mut SubGridAssignments,
    key: ShardKey,
    incoming: &[(glam::IVec3, u32)],
) {
    let map = store.by_shard.entry(key).or_default();
    // `SubGridAssignmentUpdate` is a full snapshot (server re-sends on
    // every change), so replace wholesale.
    map.clear();
    for &(pos, sub_grid_id) in incoming {
        map.insert(IVec3::new(pos.x, pos.y, pos.z), sub_grid_id);
    }
}

/// Per-tick live pose refresh for every sub-grid on every loaded shard.
/// Primary shard's `WorldState.sub_grids[]` carries the authoritative
/// transform list. (Secondary-shard sub-grid transforms would also
/// flow here once the server wire format exposes them — today SHIP's
/// own sub-grids only appear in the SHIP shard's own WS, so visible
/// sub-grid animation for a secondary ship requires that ship's
/// secondary WS to be received. We already do this via
/// `SecondaryWorldStates`.)
fn ingest_transforms(
    primary: Res<PrimaryShard>,
    primary_ws: Res<PrimaryWorldState>,
    mut transforms: ResMut<SubGridTransforms>,
) {
    transforms.by_shard.clear();
    let (Some(key), Some(ws)) = (primary.current, primary_ws.latest.as_ref()) else {
        return;
    };
    for sg in &ws.sub_grids {
        transforms
            .by_shard
            .insert((key, sg.sub_grid_id), sg.clone());
    }
}

/// Spawn or update a Bevy entity per `(ShardKey, sub_grid_id)` in
/// `SubGridTransforms`. Parented to the shard's `ChunkSource`; Bevy's
/// transform propagation carries the ship's world pose into the
/// sub-grid without extra logic here.
fn spawn_or_update_sub_grids(
    transforms: Res<SubGridTransforms>,
    sources: Res<SourceIndex>,
    mut index: ResMut<SubGridIndex>,
    mut existing: Query<&mut Transform, With<SubGrid>>,
    mut commands: Commands,
) {
    for (&(shard, sub_grid_id), t) in &transforms.by_shard {
        let Some(&parent) = sources.by_shard.get(&shard) else { continue };
        let translation = Vec3::new(t.translation.x, t.translation.y, t.translation.z);
        let rotation = Quat::from_xyzw(
            t.rotation.x,
            t.rotation.y,
            t.rotation.z,
            t.rotation.w,
        );
        if let Some(&entity) = index.entries.get(&(shard, sub_grid_id)) {
            if let Ok(mut tf) = existing.get_mut(entity) {
                tf.translation = translation;
                tf.rotation = rotation;
            }
        } else {
            let entity = commands
                .spawn((
                    SubGrid {
                        shard,
                        sub_grid_id,
                    },
                    Transform {
                        translation,
                        rotation,
                        scale: Vec3::ONE,
                    },
                    GlobalTransform::IDENTITY,
                    Visibility::default(),
                    InheritedVisibility::default(),
                    ViewVisibility::default(),
                    Name::new(format!("sub_grid[{}/{}]", shard, sub_grid_id)),
                    ChildOf(parent),
                ))
                .id();
            index.entries.insert((shard, sub_grid_id), entity);
            tracing::info!(%shard, sub_grid_id, "sub-grid entity spawned");
        }
    }

    // Despawn sub-grids the server stopped broadcasting (e.g. rotor
    // block destroyed). Keyed on presence in `transforms.by_shard`.
    let stale: Vec<(ShardKey, u32)> = index
        .entries
        .keys()
        .copied()
        .filter(|k| !transforms.by_shard.contains_key(k))
        .collect();
    for key in stale {
        if let Some(e) = index.entries.remove(&key) {
            commands.entity(e).despawn();
            tracing::info!(shard = %key.0, sub_grid_id = key.1, "sub-grid despawned (server stopped broadcasting)");
        }
    }
}

/// Clean up sub-grid entries for shards that have been despawned from
/// `SourceIndex` (full shard teardown). Bevy already recursively
/// despawned the entities via `ChildOf`; this just clears the index
/// so downstream code doesn't see stale `(ShardKey, id) → Entity`
/// mappings.
fn gc_despawned_shards(sources: Res<SourceIndex>, mut index: ResMut<SubGridIndex>) {
    index
        .entries
        .retain(|(shard, _), _| sources.by_shard.contains_key(shard));
}
