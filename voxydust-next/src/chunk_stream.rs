//! Server-streamed chunk renderer.
//!
//! Every chunk source (the player's own primary shard + every secondary
//! pre-connected shard) is represented by one **root entity** carrying
//! `ChunkSource` + a `Transform`; every chunk mesh is spawned as a child of
//! that root at its ship-local `chunk_index × CHUNK_SIZE` offset. The root's
//! transform is driven every frame from the shard's authoritative pose
//! (primary: identity on the shard that *is* its frame of reference;
//! secondary: the matching entity in the primary WorldState), and Bevy's
//! transform propagation carries every chunk child into place automatically.
//!
//! The unified parent-child structure for primary and secondary is the
//! piece that makes `shard_transition::GraceWindow` possible: when the
//! player boards/exits/launches/lands, the old primary's root entity is
//! *handed over* to `GraceWindow` rather than destroyed, and its children
//! keep rendering against a system-space anchor for `GRACE_SECS` while the
//! new primary's chunks stream in. At grace expiry `commands.entity(root)
//! .despawn()` reclaims the entire subtree in one call.
//!
//! Server authority is total. Chunk positions (chunk index, block data),
//! source-to-ship association (via `WorldState.entities[].shard_id`), and
//! primary seed (via `NetEvent::Connected`) all come from the server. The
//! only local computation is the greedy-mesh conversion of the raw chunk
//! storage into a Bevy `Mesh`.

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
};
use voxeldust_core::client_message::EntityKind;

use crate::net_plugin::GameEvent;
use crate::network::NetEvent;
use crate::shard_transition::{GraceWindow, ShardContext, ShardTransitionSet, WorldStates};

pub struct ChunkStreamPlugin;

impl Plugin for ChunkStreamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkIndex>()
            .init_resource::<SourceIndex>()
            .init_resource::<SharedBlockRegistry>()
            .init_resource::<ChunkMaterialHandle>()
            .init_resource::<PendingPrimaryChunks>()
            .init_resource::<ChunkStorageCache>()
            .add_systems(
                Update,
                (sync_source_transforms, ingest_chunk_events)
                    .chain()
                    .after(ShardTransitionSet),
            );
    }
}

/// Retained per-chunk block storage keyed by `(seed, chunk_index)`.
///
/// Necessary for client-side block raycasting (highlight cube, place/break
/// targeting): the renderer needs the full `ChunkStorage` to test block
/// occupancy along a ray, not just the derived greedy-mesh. Legacy client
/// uses `ClientChunkCache` for the same purpose; voxydust-next keeps its
/// own seed-keyed variant because our `SourceIndex` is already seed-keyed
/// and introducing a second id type (`ClientChunkCache::ChunkSourceId`)
/// would force a translation layer across every consumer.
///
/// The cache is written by `ingest_chunk_events` on both full snapshots
/// (replace) and deltas (mutate in place). A missing chunk for a given
/// key means the client has not received it yet — consumers must handle
/// that (raycast returns None / meshing skips).
#[derive(Resource, Default)]
pub struct ChunkStorageCache {
    pub entries: bevy::platform::collections::HashMap<(SourceId, IVec3), ChunkStorage>,
}

impl ChunkStorageCache {
    pub fn get(&self, seed: SourceId, chunk_index: IVec3) -> Option<&ChunkStorage> {
        self.entries.get(&(seed, chunk_index))
    }

    pub fn seeds(&self) -> impl Iterator<Item = SourceId> + '_ {
        self.entries.keys().map(|(s, _)| *s).collect::<std::collections::HashSet<_>>().into_iter()
    }
}

/// Buffer for primary `ChunkSnapshot` payloads that arrive before
/// `ShardContext.primary_seed` is populated. Flushed at the start of the
/// next `ingest_chunk_events` tick that has a seed. Without this buffer
/// the first-frame connection race (Connected + ChunkSnapshots both
/// arriving in the same mpsc batch, but scheduled after ingest) would
/// permanently lose those snapshots — the server only resends on chunk
/// *deltas*, and a fully-static ship emits no deltas, so the cockpit
/// interior would never appear.
#[derive(Resource, Default)]
pub struct PendingPrimaryChunks {
    /// `(chunk_index, raw snapshot bytes)` waiting for a primary seed.
    pub entries: Vec<(IVec3, Vec<u8>)>,
}

/// A chunk source is uniquely identified by its shard seed. On SHIP shards
/// the seed equals the ship's `entity_id`; on PLANET/SYSTEM/GALAXY shards
/// it is the shard's world seed. The legacy client used `0` as a sentinel
/// for "primary," conflating "which source is this" with "which is in the
/// camera frame" — keeping the two orthogonal (`SourceIndex` is keyed by
/// seed; `ShardContext.primary_seed` tracks which one is currently primary)
/// lets the grace window, secondary pre-connect, and primary promotion all
/// share the same storage without special cases.
pub type SourceId = u64;

pub type ChunkKey = (SourceId, IVec3);

/// Marker on the root of each chunk source. Carries the seed so render
/// diagnostics and grace-window driving can correlate the entity back to
/// its shard.
#[derive(Component, Debug, Clone, Copy)]
pub struct ChunkSource {
    pub id: SourceId,
}

/// seed → root entity, for every currently-live chunk source. Grace-window
/// sources are moved *out* of here into `GraceWindow::sources` at the
/// moment of transition so this map always reflects "sources currently
/// receiving snapshots."
#[derive(Resource, Default, Debug)]
pub struct SourceIndex {
    pub entries: HashMap<SourceId, Entity>,
}

impl SourceIndex {
    /// Remove and return the root entity for `seed`, leaving the entity
    /// *alive* in the world so the grace-window driver can keep its
    /// children rendered. The caller owns the entity afterwards.
    pub fn take(&mut self, seed: SourceId) -> Option<Entity> {
        self.entries.remove(&seed)
    }

    /// No-op when the seed is already tracked; otherwise does nothing. The
    /// legacy client treated "promote secondary → primary" as a rekey of a
    /// sentinel `0`, but with seeds as the canonical id both roles share
    /// the same entry — primary vs secondary is tracked by `ShardContext`
    /// not here.
    pub fn rekey_to_primary(&mut self, _seed: &SourceId, _commands: &mut Commands) {}
}

/// `(seed, chunk_index)` → chunk mesh entity. Snapshot re-arrival for the
/// same key replaces the mesh.
#[derive(Resource, Default)]
pub struct ChunkIndex {
    pub entries: HashMap<ChunkKey, Entity>,
}

#[derive(Resource)]
pub struct SharedBlockRegistry(pub BlockRegistry);
impl Default for SharedBlockRegistry {
    fn default() -> Self { Self(BlockRegistry::new()) }
}

#[derive(Resource, Default)]
struct ChunkMaterialHandle(Option<Handle<StandardMaterial>>);

/// Each frame, place every **live** source (i.e. not in the grace window —
/// `drive_grace_transforms` handles those) at its authoritative pose.
///
/// **Frame-of-reference contract (critical).** Bevy's world frame is
/// "system-space-aligned, centered at the primary's `ws.origin`." That
/// means:
///
/// * **Positions** are `entity.position` as broadcast in the primary WS.
///   On a SHIP primary the server already centers AOI entities on its own
///   ship (`ship-shard/main.rs:2548`: `entity.position = external_pos -
///   exterior.position`), so those values are directly usable. The own
///   ship itself has `position = DVec3::ZERO` — it IS the origin.
/// * **Orientations** are `entity.rotation` — world-space (system-space)
///   rotations. On SHIP primaries this means the own-ship entity carries
///   the ship's world rotation; we apply it to the root so chunks render
///   in the ship's actual orientation instead of axis-aligned to the
///   system. Without this rotation, the chunk mesh would appear to
///   tumble as the ship rotates in space while the camera orbits around
///   it rigidly. (This was the visible "ship disappears / camera flies
///   off" symptom: the primary root was stuck at identity rotation so
///   the cockpit interior rendered in a fixed system-space orientation
///   while every other subsystem — camera, bodies, secondaries —
///   tracked the ship's actual world rotation.)
/// * **Celestial bodies + secondary ships** use the same frame, applied
///   identically in their own subsystems (`celestial.rs`, the secondary
///   branch here). All in lockstep.
fn sync_source_transforms(
    sources: Res<SourceIndex>,
    grace: Res<GraceWindow>,
    ctx: Res<ShardContext>,
    worlds: Res<WorldStates>,
    mut q: Query<&mut Transform, With<ChunkSource>>,
) {
    // Multi-shard AOI lookup. A secondary ship's position may be reported
    // by the primary WS (ship-shard publishes nearby ships in its
    // `entities[]` via `external_entities`), by the SYSTEM secondary WS
    // (system shard broadcasts all ships in its AOI), or by the GALAXY
    // secondary WS (very distant ships). Legacy client walks ws +
    // sec_ws + grace_ws each frame for this reason — we do the same via
    // `WorldStates` which accumulates primary + per-type secondary + the
    // one-shot `last_primary` (grace).
    //
    // Reading from the resource (not replaying NetEvent::WorldState)
    // ensures the lookup is frame-consistent regardless of which shard
    // sent a WS this tick; a primary that doesn't see the ship locally
    // can still drive its transform from a SYSTEM secondary broadcast.
    let Some(primary_ws) = worlds.primary.as_ref() else {
        return;
    };

    // Own-ship rotation for the primary root on SHIP shards.
    let own_ship = primary_ws
        .entities
        .iter()
        .find(|e| e.is_own && e.kind == EntityKind::Ship);

    // Helper: does this WS have a ship with entity_id == seed?
    let find_anchor = |seed: u64| {
        let in_primary = primary_ws
            .entities
            .iter()
            .find(|e| e.entity_id == seed && e.kind == EntityKind::Ship);
        if in_primary.is_some() {
            return in_primary;
        }
        for ws in worlds.secondary_by_type.values() {
            if let Some(e) = ws
                .entities
                .iter()
                .find(|e| e.entity_id == seed && e.kind == EntityKind::Ship)
            {
                return Some(e);
            }
        }
        if let Some(grace) = worlds.last_primary.as_ref() {
            return grace
                .entities
                .iter()
                .find(|e| e.entity_id == seed && e.kind == EntityKind::Ship);
        }
        None
    };

    // Rate-limited diag: one line per second so we can see source count,
    // primary seed presence, own-ship anchor, and ws.origin. The
    // `ws_origin` field is how you tell whether the ship is actually
    // moving in system-space: if thrusters fire but origin stays
    // frozen, the server isn't applying force (signal-graph wiring
    // issue); if origin advances but camera looks static, the issue
    // is on the client render side (shard_origin / celestial parallax).
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_LOG: AtomicU64 = AtomicU64::new(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if now_ms.saturating_sub(LAST_LOG.load(Ordering::Relaxed)) > 1000 {
            LAST_LOG.store(now_ms, Ordering::Relaxed);
            let own_vel = own_ship.map(|s| (s.velocity.x, s.velocity.y, s.velocity.z));
            tracing::info!(
                source_count = sources.entries.len(),
                primary_seed = ?ctx.primary_seed,
                own_ship_present = own_ship.is_some(),
                own_ship_rot = ?own_ship.map(|s| (s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w)),
                own_ship_velocity = ?own_vel,
                ws_origin = ?(primary_ws.origin.x, primary_ws.origin.y, primary_ws.origin.z),
                grace_count = grace.sources.len(),
                entity_count = primary_ws.entities.len(),
                "sync_source_transforms diag"
            );
        }
    }

    for (&seed, &entity) in &sources.entries {
        if grace.sources.contains_key(&seed) {
            continue;
        }
        let Ok(mut tf) = q.get_mut(entity) else { continue };

        if ctx.primary_seed == Some(seed) {
            // Primary source: centered at Bevy origin. Rotation from the
            // own-ship entity on SHIP shards; identity on other shard
            // types (PLANET / SYSTEM / GALAXY have no primary "anchor
            // body" — their geometry is already published in the shard's
            // rendering frame).
            tf.translation = Vec3::ZERO;
            tf.rotation = if let Some(ship) = own_ship {
                Quat::from_xyzw(
                    ship.rotation.x as f32,
                    ship.rotation.y as f32,
                    ship.rotation.z as f32,
                    ship.rotation.w as f32,
                )
            } else {
                Quat::IDENTITY
            };
            continue;
        }

        // Secondary: find anchor across all available WSs.
        let Some(anchor) = find_anchor(seed) else {
            continue;
        };
        tf.translation = Vec3::new(
            anchor.position.x as f32,
            anchor.position.y as f32,
            anchor.position.z as f32,
        );
        tf.rotation = Quat::from_xyzw(
            anchor.rotation.x as f32,
            anchor.rotation.y as f32,
            anchor.rotation.z as f32,
            anchor.rotation.w as f32,
        );
    }
}

fn ingest_chunk_events(
    mut events: MessageReader<GameEvent>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mat: ResMut<ChunkMaterialHandle>,
    mut chunks: ResMut<ChunkIndex>,
    mut sources: ResMut<SourceIndex>,
    mut pending: ResMut<PendingPrimaryChunks>,
    mut storage: ResMut<ChunkStorageCache>,
    registry_shared: Res<SharedBlockRegistry>,
    ctx: Res<ShardContext>,
) {
    if mat.0.is_none() {
        mat.0 = Some(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.85,
            metallic: 0.0,
            // `cull_mode: None` is a diagnostic carryover from the initial
            // debugging of the greedy-mesh winding order. Now that quads
            // are verified CCW (see `quads_to_bevy_mesh` index layout),
            // leaving back-face culling off is a net cost we want to
            // reclaim. The one caveat is that interior block faces on a
            // just-built ship visible from outside (open cockpit) would
            // pop between frames during auto-generated mesh updates; if
            // that reappears we clamp cull_mode back off and open a
            // separate bug against the mesher. Keeping `Back` as the
            // default for now since it's correct for the shipped path.
            cull_mode: Some(bevy::render::render_resource::Face::Back),
            ..default()
        }));
    }
    let material = mat.0.clone().unwrap();

    // Flush any primary snapshots that were received before `primary_seed`
    // was populated (e.g. the scheduling race on the first connection
    // frame — see `net_plugin::NetworkPlugin::build` for the ordering
    // fix that makes this empty in practice, but we keep the buffer as
    // a resilience mechanism against future scheduler changes).
    if let Some(seed) = ctx.primary_seed {
        if !pending.entries.is_empty() {
            let drained = std::mem::take(&mut pending.entries);
            tracing::info!(count = drained.len(), seed, "flushing buffered primary chunks");
            for (chunk_index, data) in drained {
                spawn_or_replace_chunk(
                    &mut commands,
                    &mut meshes,
                    &material,
                    &registry_shared.0,
                    &mut chunks,
                    &mut sources,
                    &mut storage,
                    seed,
                    chunk_index,
                    &data,
                );
            }
        }
    }

    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::ChunkSnapshot(cs) => {
                let chunk_index = IVec3::new(cs.chunk_x, cs.chunk_y, cs.chunk_z);
                let Some(seed) = ctx.primary_seed else {
                    pending.entries.push((chunk_index, cs.data.clone()));
                    continue;
                };
                spawn_or_replace_chunk(
                    &mut commands,
                    &mut meshes,
                    &material,
                    &registry_shared.0,
                    &mut chunks,
                    &mut sources,
                    &mut storage,
                    seed,
                    chunk_index,
                    &cs.data,
                );
            }
            NetEvent::SecondaryChunkSnapshot { seed, data } => {
                spawn_or_replace_chunk(
                    &mut commands,
                    &mut meshes,
                    &material,
                    &registry_shared.0,
                    &mut chunks,
                    &mut sources,
                    &mut storage,
                    *seed,
                    IVec3::new(data.chunk_x, data.chunk_y, data.chunk_z),
                    &data.data,
                );
            }
            NetEvent::ChunkDelta(cd) => {
                // Primary-shard chunk delta — apply block mods to retained
                // ChunkStorage, then re-mesh the affected chunk so the
                // break/place action the user just triggered becomes
                // visible within one server tick. Deltas only ever come
                // from the primary TCP (server-authoritative edits).
                let Some(seed) = ctx.primary_seed else { continue };
                let chunk_index = IVec3::new(cd.chunk_x, cd.chunk_y, cd.chunk_z);
                apply_chunk_delta(
                    &mut commands,
                    &mut meshes,
                    &material,
                    &registry_shared.0,
                    &mut chunks,
                    &mut sources,
                    &mut storage,
                    seed,
                    chunk_index,
                    cd,
                );
            }
            NetEvent::SecondaryChunkDelta { seed, data: cd } => {
                let chunk_index = IVec3::new(cd.chunk_x, cd.chunk_y, cd.chunk_z);
                apply_chunk_delta(
                    &mut commands,
                    &mut meshes,
                    &material,
                    &registry_shared.0,
                    &mut chunks,
                    &mut sources,
                    &mut storage,
                    *seed,
                    chunk_index,
                    cd,
                );
            }
            _ => {}
        }
    }
}

/// Apply a `ChunkDeltaData` to retained storage and re-mesh the chunk.
///
/// Deltas carry sparse block edits (`mods: Vec<BlockModData>`) and
/// sub-block edits (`sub_block_mods`). Missing the retained storage
/// means we never received the underlying snapshot — dropping is safe;
/// the next snapshot will rebuild. Stale `seq` (server re-broadcast of
/// an already-applied delta) is rejected here to keep client state
/// monotonic even if TCP duplicates a packet.
fn apply_chunk_delta(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<StandardMaterial>,
    registry: &BlockRegistry,
    chunks: &mut ChunkIndex,
    sources: &mut SourceIndex,
    storage: &mut ChunkStorageCache,
    seed: SourceId,
    chunk_index: IVec3,
    cd: &voxeldust_core::client_message::ChunkDeltaData,
) {
    let Some(chunk) = storage.entries.get_mut(&(seed, chunk_index)) else {
        tracing::warn!(
            seed,
            ?chunk_index,
            "chunk delta arrived before snapshot — dropping; next snapshot will correct"
        );
        return;
    };
    // Apply block mods.
    for m in &cd.mods {
        chunk.set_block(m.bx, m.by, m.bz, BlockId::from_u16(m.block_type));
    }
    // TODO: apply sub_block_mods when sub-block rendering lands (#23).
    let _ = &cd.sub_block_mods;

    // Re-mesh and re-spawn. Snapshot path handles the full mesh rebuild
    // including despawn-old + spawn-new; a delta does the same because
    // even a single block change can flip a face between visible and
    // hidden. For large deltas the server would send a new snapshot
    // instead (ChunkStorage diff threshold), so re-meshing on every
    // delta is not a perf concern at the typical edit rate.
    remesh_chunk(
        commands, meshes, material, registry, chunks, sources, storage, seed, chunk_index,
    );
}

/// Rebuild the Bevy mesh for a single chunk from its current retained
/// `ChunkStorage`. Used after deltas — for snapshots the `spawn_or_replace_chunk`
/// path is used directly since it already has the compressed bytes.
fn remesh_chunk(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<StandardMaterial>,
    registry: &BlockRegistry,
    chunks: &mut ChunkIndex,
    sources: &mut SourceIndex,
    storage: &ChunkStorageCache,
    seed: SourceId,
    chunk_index: IVec3,
) {
    let Some(chunk) = storage.get(seed, chunk_index) else { return };
    let neighbours: [Option<&ChunkStorage>; 6] = [None; 6];
    let quads = mesh_chunk(chunk, &neighbours, registry, false);
    let key = (seed, chunk_index);
    if let Some(old) = chunks.entries.remove(&key) {
        commands.entity(old).despawn();
    }
    if quads.quads.is_empty() {
        return;
    }
    let mesh_handle = meshes.add(quads_to_bevy_mesh(&quads, registry));
    let parent = ensure_source_entity(commands, sources, seed);
    let local_offset = Vec3::new(
        (chunk_index.x * CHUNK_SIZE as i32) as f32,
        (chunk_index.y * CHUNK_SIZE as i32) as f32,
        (chunk_index.z * CHUNK_SIZE as i32) as f32,
    );
    let entity = commands
        .spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(local_offset),
            Name::new(format!(
                "chunk_{}_{}_{}_{}",
                seed, chunk_index.x, chunk_index.y, chunk_index.z
            )),
            ChildOf(parent),
        ))
        .id();
    chunks.entries.insert(key, entity);
}

/// Spawn (or replace) a chunk mesh as a child of the source's root entity,
/// creating the root lazily on first snapshot. Uniform for primary and
/// secondary — the only thing that differs downstream is how
/// `sync_source_transforms` drives the root's Transform.
fn spawn_or_replace_chunk(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<StandardMaterial>,
    registry: &BlockRegistry,
    chunks: &mut ChunkIndex,
    sources: &mut SourceIndex,
    storage: &mut ChunkStorageCache,
    seed: SourceId,
    chunk_index: IVec3,
    data: &[u8],
) {
    let chunk = match deserialize_chunk(data) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(?e, seed, ?chunk_index, "chunk deserialize failed");
            return;
        }
    };

    let key = (seed, chunk_index);
    let neighbours: [Option<&ChunkStorage>; 6] = [None; 6];
    let quads = mesh_chunk(&chunk, &neighbours, registry, false);
    // Retain storage for raycast queries. Moved in after meshing since
    // `ChunkStorage` is not `Clone` (palette-compressed storage would
    // need deep copy). Empty-mesh chunks still matter for raycasting
    // (air blocks are part of the occupancy field) so insert before
    // the early-return path.
    storage.entries.insert(key, chunk);

    if quads.quads.is_empty() {
        if let Some(old) = chunks.entries.remove(&key) {
            commands.entity(old).despawn();
        }
        return;
    }

    let mesh_handle = meshes.add(quads_to_bevy_mesh(&quads, registry));
    let parent = ensure_source_entity(commands, sources, seed);

    if let Some(old) = chunks.entries.remove(&key) {
        commands.entity(old).despawn();
    }
    let local_offset = Vec3::new(
        (chunk_index.x * CHUNK_SIZE as i32) as f32,
        (chunk_index.y * CHUNK_SIZE as i32) as f32,
        (chunk_index.z * CHUNK_SIZE as i32) as f32,
    );
    let entity = commands
        .spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(local_offset),
            Name::new(format!(
                "chunk_{}_{}_{}_{}",
                seed, chunk_index.x, chunk_index.y, chunk_index.z
            )),
            ChildOf(parent),
        ))
        .id();
    chunks.entries.insert(key, entity);
}

/// Fetch or create the root `ChunkSource` entity for `seed`. The root starts
/// at identity transform; `sync_source_transforms` will place it correctly
/// before the next render.
fn ensure_source_entity(
    commands: &mut Commands,
    sources: &mut SourceIndex,
    seed: SourceId,
) -> Entity {
    if let Some(&entity) = sources.entries.get(&seed) {
        return entity;
    }
    let entity = commands
        .spawn((
            Transform::IDENTITY,
            GlobalTransform::default(),
            Visibility::Visible,
            InheritedVisibility::VISIBLE,
            ViewVisibility::default(),
            ChunkSource { id: seed },
            Name::new(format!("chunk_source_{}", seed)),
        ))
        .id();
    sources.entries.insert(seed, entity);
    entity
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
        indices.extend_from_slice(&[base + 2, base, base + 1, base + 1, base + 3, base + 2]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(positions));
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(normals));
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(colors));
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
