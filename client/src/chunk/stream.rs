//! Chunk ingest — primary + secondary paths.
//!
//! Decodes `ChunkSnapshot` / `ChunkDelta` (primary and secondary
//! variants) into `ChunkStorage`, runs the greedy mesher with neighbour
//! context, and spawns / replaces a child mesh under the shard's
//! `ChunkSource` root entity. ShardKey resolution goes through
//! `PrimaryShard` + `Secondaries` (phase 3) so the same code path
//! handles both sources.

use std::collections::HashMap;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use voxeldust_core::block::{
    block_id::BlockId,
    chunk_mesher::{mesh_chunk, ChunkQuads, FACE_NORMALS},
    chunk_storage::ChunkStorage,
    palette::CHUNK_SIZE,
    registry::BlockRegistry,
    serialization::deserialize_chunk,
    sub_block_mesher::{mesh_sub_blocks, SubBlockMeshData},
};

use crate::chunk::cache::{ChunkKey, ChunkStorageCache};
use crate::chunk::material::{ensure_chunk_material, ChunkMaterialCache};
use crate::net::{GameEvent, NetEvent};
use crate::shard::{PrimaryShard, Secondaries, ShardKey, SourceIndex};

/// `(ShardKey, chunk_index)` → chunk mesh entity. Snapshot re-arrival
/// for the same key replaces the mesh.
#[derive(Resource, Default)]
pub struct ChunkIndex {
    pub entries: HashMap<ChunkKey, Entity>,
}

/// Registry shared with `mesh_chunk`. Built once at startup; blocks
/// have stable `BlockId` + `color_hint` registered in `voxeldust-core`.
#[derive(Resource)]
pub struct SharedBlockRegistry(pub BlockRegistry);

impl Default for SharedBlockRegistry {
    fn default() -> Self {
        Self(BlockRegistry::new())
    }
}

/// Primary ingest — `ChunkSnapshot` + `ChunkDelta` events arrive
/// without shard info, so we resolve the primary `ShardKey` via the
/// `PrimaryShard` resource. If no primary is set yet, the chunk is
/// dropped (a subsequent snapshot will correct).
pub fn ingest_primary_chunks(
    mut events: MessageReader<GameEvent>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut mat_cache: ResMut<ChunkMaterialCache>,
    mut chunk_index: ResMut<ChunkIndex>,
    mut storage: ResMut<ChunkStorageCache>,
    registry: Res<SharedBlockRegistry>,
    primary: Res<PrimaryShard>,
    sources: Res<SourceIndex>,
    panel_configs: Res<crate::hud::panel_config::HudPanelConfigs>,
) {
    let Some(primary_key) = primary.current else {
        // Nothing to route primary-keyed events to; log sparsely for
        // visibility during the phase-1 empty registry but don't spam.
        for GameEvent(ev) in events.read() {
            if let NetEvent::ChunkSnapshot(_) | NetEvent::ChunkDelta(_) = ev {
                tracing::debug!("primary chunk event with no primary shard — dropping");
            }
        }
        return;
    };

    let material = ensure_chunk_material(&mut mat_cache, &mut materials);
    let Some(&parent) = sources.by_shard.get(&primary_key) else {
        return;
    };

    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::ChunkSnapshot(cs) => {
                let chunk_index_v = IVec3::new(cs.chunk_x, cs.chunk_y, cs.chunk_z);
                spawn_or_replace_chunk(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut images,
                    &material,
                    &registry.0,
                    &mut chunk_index,
                    &mut storage,
                    &panel_configs,
                    primary_key,
                    parent,
                    chunk_index_v,
                    &cs.data,
                );
            }
            NetEvent::ChunkDelta(cd) => {
                let chunk_index_v = IVec3::new(cd.chunk_x, cd.chunk_y, cd.chunk_z);
                apply_chunk_delta(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut images,
                    &material,
                    &registry.0,
                    &mut chunk_index,
                    &mut storage,
                    &panel_configs,
                    primary_key,
                    parent,
                    chunk_index_v,
                    cd,
                );
            }
            _ => {}
        }
    }
}

/// Secondary ingest — `SecondaryChunkSnapshot` / `SecondaryChunkDelta`
/// carry just the seed; we resolve the `ShardKey` by searching
/// `Secondaries`. Missing key → secondary hasn't been registered yet
/// (typical during first-tick race); drop, next snapshot will correct.
pub fn ingest_secondary_chunks(
    mut events: MessageReader<GameEvent>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut mat_cache: ResMut<ChunkMaterialCache>,
    mut chunk_index: ResMut<ChunkIndex>,
    mut storage: ResMut<ChunkStorageCache>,
    registry: Res<SharedBlockRegistry>,
    secondaries: Res<Secondaries>,
    sources: Res<SourceIndex>,
    panel_configs: Res<crate::hud::panel_config::HudPanelConfigs>,
) {
    let material = ensure_chunk_material(&mut mat_cache, &mut materials);
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::SecondaryChunkSnapshot { seed, data } => {
                let Some(key) = resolve_secondary_key(&secondaries, *seed) else {
                    tracing::debug!(%seed, "secondary chunk snapshot: key not yet registered");
                    continue;
                };
                let Some(&parent) = sources.by_shard.get(&key) else { continue };
                let idx = IVec3::new(data.chunk_x, data.chunk_y, data.chunk_z);
                spawn_or_replace_chunk(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut images,
                    &material,
                    &registry.0,
                    &mut chunk_index,
                    &mut storage,
                    &panel_configs,
                    key,
                    parent,
                    idx,
                    &data.data,
                );
            }
            NetEvent::SecondaryChunkDelta { seed, data } => {
                let Some(key) = resolve_secondary_key(&secondaries, *seed) else {
                    continue;
                };
                let Some(&parent) = sources.by_shard.get(&key) else { continue };
                let idx = IVec3::new(data.chunk_x, data.chunk_y, data.chunk_z);
                apply_chunk_delta(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut images,
                    &material,
                    &registry.0,
                    &mut chunk_index,
                    &mut storage,
                    &panel_configs,
                    key,
                    parent,
                    idx,
                    data,
                );
            }
            _ => {}
        }
    }
}

fn resolve_secondary_key(secondaries: &Secondaries, seed: u64) -> Option<ShardKey> {
    secondaries
        .runtimes
        .keys()
        .find(|k| k.seed == seed)
        .copied()
}

#[allow(clippy::too_many_arguments)]
fn spawn_or_replace_chunk(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    material: &Handle<StandardMaterial>,
    registry: &BlockRegistry,
    chunks: &mut ChunkIndex,
    storage: &mut ChunkStorageCache,
    panel_configs: &crate::hud::panel_config::HudPanelConfigs,
    key: ShardKey,
    parent: Entity,
    chunk_index: IVec3,
    data: &[u8],
) {
    let chunk = match deserialize_chunk(data) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(?e, %key, ?chunk_index, "chunk deserialize failed");
            return;
        }
    };

    let cache_key: ChunkKey = (key, chunk_index);
    // Insert into cache first so neighbour lookups for subsequent chunks
    // in the same batch see the just-arrived chunk.
    storage.entries.insert(cache_key, chunk);

    remesh_chunk_from_cache(
        commands, meshes, materials, images, material, registry, chunks, storage,
        panel_configs, key, parent, chunk_index,
    );
}

#[allow(clippy::too_many_arguments)]
fn apply_chunk_delta(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    material: &Handle<StandardMaterial>,
    registry: &BlockRegistry,
    chunks: &mut ChunkIndex,
    storage: &mut ChunkStorageCache,
    panel_configs: &crate::hud::panel_config::HudPanelConfigs,
    key: ShardKey,
    parent: Entity,
    chunk_index: IVec3,
    cd: &voxeldust_core::client_message::ChunkDeltaData,
) {
    let Some(chunk) = storage.get_mut(key, chunk_index) else {
        tracing::warn!(
            %key,
            ?chunk_index,
            "chunk delta arrived before snapshot — dropping; next snapshot will correct",
        );
        return;
    };
    for m in &cd.mods {
        chunk.set_block(m.bx, m.by, m.bz, BlockId::from_u16(m.block_type));
    }
    // Apply sub-block mods so `HudPanel` placements (and wires, rails,
    // ladders, etc.) appear on the client. The mesher + per-tile HUD
    // spawn below run against the updated `chunk.sub_blocks` map.
    //
    // Action codes come straight from `voxeldust_core::client_message::
    // action` constants — the server echoes the request's `action` into
    // `SubBlockModData.action`:
    //   action::PLACE_SUB  (10) → add SubBlockElement
    //   action::REMOVE_SUB (11) → drop the element on that face
    for m in &cd.sub_block_mods {
        use voxeldust_core::block::sub_block::{SubBlockElement, SubBlockType};
        use voxeldust_core::client_message::action;
        match m.action {
            action::REMOVE_SUB => {
                chunk.remove_sub_block(m.bx, m.by, m.bz, m.face);
                tracing::info!(
                    block = ?(m.bx, m.by, m.bz),
                    face = m.face,
                    "sub-block delta: removed",
                );
            }
            action::PLACE_SUB => {
                let Some(element_type) = SubBlockType::from_u8(m.element_type) else {
                    tracing::warn!(
                        element_type = m.element_type,
                        "sub-block delta: unknown element_type",
                    );
                    continue;
                };
                chunk.add_sub_block(
                    m.bx,
                    m.by,
                    m.bz,
                    SubBlockElement {
                        face: m.face,
                        element_type,
                        rotation: m.rotation & 0x03,
                        flags: 0,
                    },
                );
                tracing::info!(
                    block = ?(m.bx, m.by, m.bz),
                    face = m.face,
                    element = element_type.label(),
                    "sub-block delta: placed",
                );
            }
            _ => {
                tracing::warn!(action = m.action, "sub-block delta: unknown action");
            }
        }
    }

    remesh_chunk_from_cache(
        commands, meshes, materials, images, material, registry, chunks, storage,
        panel_configs, key, parent, chunk_index,
    );
}

#[allow(clippy::too_many_arguments)]
fn remesh_chunk_from_cache(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    material: &Handle<StandardMaterial>,
    registry: &BlockRegistry,
    chunks: &mut ChunkIndex,
    storage: &ChunkStorageCache,
    panel_configs: &crate::hud::panel_config::HudPanelConfigs,
    key: ShardKey,
    parent: Entity,
    chunk_index: IVec3,
) {
    let Some(chunk) = storage.get(key, chunk_index) else { return };
    let neighbours = storage.neighbours(key, chunk_index);
    let quads = mesh_chunk(chunk, &neighbours, registry, false);
    let sub_block_mesh = mesh_sub_blocks(chunk);
    let has_hud_panels = chunk.iter_sub_blocks().any(|(_, elems)| {
        elems.iter().any(|e| {
            e.element_type == voxeldust_core::block::sub_block::SubBlockType::HudPanel
        })
    });

    let cache_key: ChunkKey = (key, chunk_index);
    if let Some(old) = chunks.entries.remove(&cache_key) {
        // `despawn` recursively drops the chunk entity + its sub-block
        // mesh child via `ChildOf`.
        commands.entity(old).despawn();
    }
    if quads.quads.is_empty() && sub_block_mesh.is_empty() && !has_hud_panels {
        return;
    }
    let local_offset = Vec3::new(
        (chunk_index.x * CHUNK_SIZE as i32) as f32,
        (chunk_index.y * CHUNK_SIZE as i32) as f32,
        (chunk_index.z * CHUNK_SIZE as i32) as f32,
    );

    // Root chunk entity. Even if the main mesh is empty, we may still
    // need it as a parent for sub-block geometry.
    let mut chunk_entity = commands.spawn((
        Transform::from_translation(local_offset),
        GlobalTransform::IDENTITY,
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        Name::new(format!(
            "chunk[{}/{},{},{}]",
            key, chunk_index.x, chunk_index.y, chunk_index.z
        )),
        ChildOf(parent),
    ));
    if !quads.quads.is_empty() {
        let mesh_handle = meshes.add(quads_to_bevy_mesh(&quads, registry));
        chunk_entity.insert((Mesh3d(mesh_handle), MeshMaterial3d(material.clone())));
    }
    let entity = chunk_entity.id();
    chunks.entries.insert(cache_key, entity);

    // Sub-block child mesh (ladders, wires, cables, rails, pipes, ...).
    // Spawned as a child of the chunk so it inherits local-offset +
    // transform chain (root ChunkSource → chunk → sub-block mesh).
    if !sub_block_mesh.is_empty() {
        let handle = meshes.add(sub_block_mesh_to_bevy(&sub_block_mesh));
        commands.spawn((
            Mesh3d(handle),
            MeshMaterial3d(material.clone()),
            Transform::IDENTITY,
            Name::new("sub_blocks"),
            ChildOf(entity),
        ));
    }

    // HUD panel sub-blocks — one textured quad per `SubBlockType::HudPanel`.
    // Spawned as children of the chunk entity so they move + despawn
    // with it. On each remesh we respawn from scratch: the set of
    // HudPanels rarely changes at tick granularity, so the cost is a
    // one-off texture / material allocation per present panel per
    // delta.
    if has_hud_panels {
        crate::hud::block_tile::spawn_hud_tiles_for_chunk(
            commands,
            meshes,
            materials,
            images,
            panel_configs,
            entity,
            chunk,
            key,
            chunk_index,
        );
    }
}

fn sub_block_mesh_to_bevy(data: &SubBlockMeshData) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(data.vertices.len());
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(data.vertices.len());
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(data.vertices.len());
    for v in &data.vertices {
        positions.push(v.position);
        normals.push(v.normal);
        colors.push([v.color[0], v.color[1], v.color[2], 1.0]);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        VertexAttributeValues::Float32x4(colors),
    );
    mesh.insert_indices(Indices::U32(data.indices.clone()));
    mesh
}

fn quads_to_bevy_mesh(quads: &ChunkQuads, registry: &BlockRegistry) -> Mesh {
    let count = quads.quads.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(count * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(count * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(count * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(count * 6);

    for quad in &quads.quads {
        let normal = FACE_NORMALS[quad.face as usize];
        let def = registry.get(BlockId::from_u16(quad.block_id));
        let color = [
            def.color_hint[0] as f32 / 255.0,
            def.color_hint[1] as f32 / 255.0,
            def.color_hint[2] as f32 / 255.0,
            1.0,
        ];
        let base = positions.len() as u32;
        for vert in &quad.vertices {
            positions.push([vert[0] as f32, vert[1] as f32, vert[2] as f32]);
            normals.push(normal);
            colors.push(color);
        }
        // CCW winding: verified against legacy voxydust render order.
        indices.extend_from_slice(&[base + 2, base, base + 1, base + 1, base + 3, base + 2]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        VertexAttributeValues::Float32x4(colors),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
