use std::collections::HashMap;

use glam::IVec3;

use super::block_id::BlockId;
use super::block_meta::{BlockMeta, BlockOrientation};
use super::chunk_storage::{ChunkStorage, DamageResult};
use super::palette::CHUNK_SIZE;
use super::registry::BlockRegistry;
use super::sub_block;

/// Pre-configuration for a block's power role in a ship/base blueprint.
#[derive(Clone, Debug)]
pub enum PowerConfig {
    /// Reactor: defines named power circuits with allocated fractions.
    Source {
        circuits: Vec<(String, f32)>,
        broadcast_range: f32,
    },
    /// Consumer: subscribes to a reactor's circuit for power.
    Consumer {
        reactor_pos: IVec3,
        circuit: String,
    },
}

/// Persisted seat binding: (label, source_u8, key_name, key_mode_u8, axis_dir_u8, channel_name, property_u8).
pub type SavedSeatBinding = (String, u8, String, u8, u8, String, u8);

/// Persisted seat configuration (bindings + seated channel).
#[derive(Clone, Debug, Default)]
pub struct SavedSeatConfig {
    pub bindings: Vec<SavedSeatBinding>,
    pub seated_channel_name: String,
}

/// Multi-chunk Cartesian block grid for ships and stations.
///
/// Ships use flat Cartesian coordinates (no sphere projection). The grid is
/// subdivided into 62³ chunks indexed by `IVec3` chunk keys. Block positions
/// within a chunk are local `(u8, u8, u8)` in the range `[0, 61]`.
///
/// The grid auto-creates chunks on write and never removes empty chunks
/// (explicit cleanup via `compact()` if needed). This avoids edge cases
/// where a block is placed in a previously-removed chunk.
pub struct ShipGrid {
    chunks: HashMap<IVec3, ChunkStorage>,
    /// Per-block signal channel overrides for pre-configured ships.
    /// Set by ship builders (e.g., `build_starter_ship`), read by `add_default_signal_bindings`.
    /// Key = world block position, value = channel name the block should subscribe to.
    channel_overrides: HashMap<IVec3, String>,
    /// Per-block boost channel for pre-configured ships.
    /// Thrusters subscribing to a boost channel multiply their thrust by the published value.
    boost_channels: HashMap<IVec3, String>,
    /// Per-block wireless power configuration for pre-configured ships.
    /// Sources define circuits; consumers point to a reactor and circuit.
    power_configs: HashMap<IVec3, PowerConfig>,
    /// Per-block sub-grid assignment for mechanical systems (rotors, pistons).
    /// Blocks not in this map belong to SubGridId::ROOT (0).
    /// Reconstructed from grid on startup via BFS from each mechanical mount.
    sub_grid_assignments: HashMap<IVec3, u32>,
    /// Persisted signal bindings per block (subscribe + publish).
    /// Loaded from redb on startup. Used by `add_default_signal_bindings` to restore
    /// player-configured signal bindings after shard restart.
    /// Key = block position, Value = (subscribe_bindings, publish_bindings)
    /// Each binding is (channel_name, property_u8).
    saved_signal_bindings: HashMap<IVec3, (Vec<(String, u8)>, Vec<(String, u8)>)>,
    /// Persisted seat configuration per block (generic seat format).
    /// If present, replaces the preset defaults on load.
    saved_seat_configs: HashMap<IVec3, SavedSeatConfig>,
}

impl ShipGrid {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            channel_overrides: HashMap::new(),
            boost_channels: HashMap::new(),
            power_configs: HashMap::new(),
            sub_grid_assignments: HashMap::new(),
            saved_signal_bindings: HashMap::new(),
            saved_seat_configs: HashMap::new(),
        }
    }

    /// Set a channel override for a block position (used by ship builders).
    pub fn set_channel_override(&mut self, x: i32, y: i32, z: i32, channel: &str) {
        self.channel_overrides.insert(IVec3::new(x, y, z), channel.to_string());
    }

    /// Get the channel override for a block position, if any.
    pub fn channel_override(&self, pos: IVec3) -> Option<&str> {
        self.channel_overrides.get(&pos).map(|s| s.as_str())
    }

    /// Set the boost channel for a block position (used by ship builders).
    pub fn set_boost_channel(&mut self, x: i32, y: i32, z: i32, channel: &str) {
        self.boost_channels.insert(IVec3::new(x, y, z), channel.to_string());
    }

    /// Get the boost channel for a block position, if any.
    pub fn boost_channel(&self, pos: IVec3) -> Option<&str> {
        self.boost_channels.get(&pos).map(|s| s.as_str())
    }

    /// Set power configuration for a block position (used by ship builders).
    pub fn set_power_config(&mut self, x: i32, y: i32, z: i32, config: PowerConfig) {
        self.power_configs.insert(IVec3::new(x, y, z), config);
    }

    /// Get the power configuration for a block position, if any.
    pub fn power_config(&self, pos: IVec3) -> Option<&PowerConfig> {
        self.power_configs.get(&pos)
    }

    /// Get the sub-grid assignment for a block position (0 = root grid).
    pub fn sub_grid_id(&self, pos: IVec3) -> u32 {
        self.sub_grid_assignments.get(&pos).copied().unwrap_or(0)
    }

    /// Assign a block to a sub-grid.
    pub fn set_sub_grid(&mut self, pos: IVec3, sub_grid_id: u32) {
        if sub_grid_id == 0 {
            self.sub_grid_assignments.remove(&pos);
        } else {
            self.sub_grid_assignments.insert(pos, sub_grid_id);
        }
    }

    /// Remove all sub-grid assignments for a given sub-grid ID, returning them to root.
    pub fn clear_sub_grid(&mut self, sub_grid_id: u32) {
        self.sub_grid_assignments.retain(|_, &mut id| id != sub_grid_id);
    }

    /// Iterate all block positions assigned to a specific sub-grid.
    pub fn blocks_in_sub_grid(&self, sub_grid_id: u32) -> impl Iterator<Item = IVec3> + '_ {
        self.sub_grid_assignments.iter()
            .filter(move |(_, id)| **id == sub_grid_id)
            .map(|(pos, _)| *pos)
    }

    /// Iterate all sub-grid assignments (pos → sub_grid_id).
    pub fn iter_sub_grid_assignments(&self) -> impl Iterator<Item = (IVec3, u32)> + '_ {
        self.sub_grid_assignments.iter().map(|(&pos, &id)| (pos, id))
    }

    /// Set saved signal bindings for a block (loaded from persistence).
    pub fn set_saved_signal_bindings(&mut self, pos: IVec3, subscribe: Vec<(String, u8)>, publish: Vec<(String, u8)>) {
        if subscribe.is_empty() && publish.is_empty() {
            self.saved_signal_bindings.remove(&pos);
        } else {
            self.saved_signal_bindings.insert(pos, (subscribe, publish));
        }
    }

    /// Get saved signal bindings for a block position, if any.
    pub fn saved_signal_bindings(&self, pos: IVec3) -> Option<&(Vec<(String, u8)>, Vec<(String, u8)>)> {
        self.saved_signal_bindings.get(&pos)
    }

    /// Set saved seat configuration for a block (loaded from persistence).
    pub fn set_saved_seat_config(&mut self, pos: IVec3, config: SavedSeatConfig) {
        if config.bindings.is_empty() && config.seated_channel_name.is_empty() {
            self.saved_seat_configs.remove(&pos);
        } else {
            self.saved_seat_configs.insert(pos, config);
        }
    }

    /// Get saved seat configuration for a block, if any.
    pub fn saved_seat_config(&self, pos: IVec3) -> Option<&SavedSeatConfig> {
        self.saved_seat_configs.get(&pos)
    }

    /// Iterate all channel overrides (position → channel name).
    pub fn iter_channel_overrides(&self) -> impl Iterator<Item = (IVec3, &str)> {
        self.channel_overrides.iter().map(|(&pos, name)| (pos, name.as_str()))
    }

    /// Iterate all boost channels (position → channel name).
    pub fn iter_boost_channels(&self) -> impl Iterator<Item = (IVec3, &str)> {
        self.boost_channels.iter().map(|(&pos, name)| (pos, name.as_str()))
    }

    /// Iterate all power configs (position → config).
    pub fn iter_power_configs(&self) -> impl Iterator<Item = (IVec3, &PowerConfig)> {
        self.power_configs.iter().map(|(&pos, cfg)| (pos, cfg))
    }

    // -----------------------------------------------------------------------
    // Coordinate helpers
    // -----------------------------------------------------------------------

    /// Convert a world-space block position to (chunk_key, local_x, local_y, local_z).
    ///
    /// World coordinates are signed integers. Chunk keys divide by `CHUNK_SIZE` (62)
    /// with floored division so that negative coordinates work correctly.
    #[inline]
    pub fn world_to_chunk(wx: i32, wy: i32, wz: i32) -> (IVec3, u8, u8, u8) {
        let cs = CHUNK_SIZE as i32;
        let chunk_key = IVec3::new(
            wx.div_euclid(cs),
            wy.div_euclid(cs),
            wz.div_euclid(cs),
        );
        let lx = wx.rem_euclid(cs) as u8;
        let ly = wy.rem_euclid(cs) as u8;
        let lz = wz.rem_euclid(cs) as u8;
        (chunk_key, lx, ly, lz)
    }

    /// Convert (chunk_key, local_x, local_y, local_z) back to world-space.
    #[inline]
    pub fn chunk_to_world(chunk_key: IVec3, lx: u8, ly: u8, lz: u8) -> (i32, i32, i32) {
        let cs = CHUNK_SIZE as i32;
        (
            chunk_key.x * cs + lx as i32,
            chunk_key.y * cs + ly as i32,
            chunk_key.z * cs + lz as i32,
        )
    }

    // -----------------------------------------------------------------------
    // Block access (world coordinates)
    // -----------------------------------------------------------------------

    /// Get the block at world-space position `(wx, wy, wz)`.
    /// Returns `Air` if the chunk doesn't exist.
    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> BlockId {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        match self.chunks.get(&key) {
            Some(chunk) => chunk.get_block(lx, ly, lz),
            None => BlockId::AIR,
        }
    }

    /// Set the block at world-space position `(wx, wy, wz)`.
    /// Creates the chunk if it doesn't exist. Returns the previous block type.
    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, id: BlockId) -> BlockId {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        let chunk = self
            .chunks
            .entry(key)
            .or_insert_with(ChunkStorage::new_empty);
        chunk.set_block(lx, ly, lz, id)
    }

    /// Get metadata at world-space position.
    pub fn get_meta(&self, wx: i32, wy: i32, wz: i32) -> Option<&BlockMeta> {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        self.chunks.get(&key)?.get_meta(lx, ly, lz)
    }

    /// Get or create metadata at world-space position.
    pub fn get_or_create_meta(&mut self, wx: i32, wy: i32, wz: i32) -> &mut BlockMeta {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        let chunk = self
            .chunks
            .entry(key)
            .or_insert_with(ChunkStorage::new_empty);
        chunk.get_or_create_meta(lx, ly, lz)
    }

    /// Remove metadata at world-space position.
    pub fn remove_meta(&mut self, wx: i32, wy: i32, wz: i32) {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        if let Some(chunk) = self.chunks.get_mut(&key) {
            chunk.remove_meta(lx, ly, lz);
        }
    }

    /// Set orientation at world-space position.
    pub fn set_orientation(&mut self, wx: i32, wy: i32, wz: i32, orientation: BlockOrientation) {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        let chunk = self
            .chunks
            .entry(key)
            .or_insert_with(ChunkStorage::new_empty);
        chunk.set_orientation(lx, ly, lz, orientation);
    }

    // -----------------------------------------------------------------------
    // Sub-block elements
    // -----------------------------------------------------------------------

    /// Add a sub-block element at a world-space position.
    pub fn add_sub_block(&mut self, wx: i32, wy: i32, wz: i32, element: sub_block::SubBlockElement) -> bool {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        let chunk = self.chunks.entry(key).or_insert_with(ChunkStorage::new_empty);
        chunk.add_sub_block(lx, ly, lz, element)
    }

    /// Remove a sub-block element from a face at world-space position.
    pub fn remove_sub_block(&mut self, wx: i32, wy: i32, wz: i32, face: u8) -> Option<sub_block::SubBlockElement> {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        self.chunks.get_mut(&key)?.remove_sub_block(lx, ly, lz, face)
    }

    /// Get sub-block elements at a world-space position.
    pub fn get_sub_blocks(&self, wx: i32, wy: i32, wz: i32) -> &[sub_block::SubBlockElement] {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        match self.chunks.get(&key) {
            Some(chunk) => chunk.get_sub_blocks(lx, ly, lz),
            None => &[],
        }
    }

    /// Check if a face at world-space position has a sub-block.
    pub fn has_sub_block(&self, wx: i32, wy: i32, wz: i32, face: u8) -> bool {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        match self.chunks.get(&key) {
            Some(chunk) => chunk.has_sub_block(lx, ly, lz, face),
            None => false,
        }
    }

    /// Apply damage to a block at world-space position.
    pub fn damage_block(
        &mut self,
        wx: i32,
        wy: i32,
        wz: i32,
        amount: u16,
        registry: &BlockRegistry,
    ) -> DamageResult {
        let (key, lx, ly, lz) = Self::world_to_chunk(wx, wy, wz);
        match self.chunks.get_mut(&key) {
            Some(chunk) => chunk.damage_block(lx, ly, lz, amount, registry),
            None => DamageResult::NoEffect,
        }
    }

    // -----------------------------------------------------------------------
    // Chunk-level access (for meshing, collider generation, serialization)
    // -----------------------------------------------------------------------

    /// Get a chunk by its key.
    pub fn get_chunk(&self, key: IVec3) -> Option<&ChunkStorage> {
        self.chunks.get(&key)
    }

    /// Get a mutable chunk by its key.
    pub fn get_chunk_mut(&mut self, key: IVec3) -> Option<&mut ChunkStorage> {
        self.chunks.get_mut(&key)
    }

    /// Iterate all chunks with their keys.
    pub fn iter_chunks(&self) -> impl Iterator<Item = (IVec3, &ChunkStorage)> {
        self.chunks.iter().map(|(&k, v)| (k, v))
    }

    /// Iterate all chunks mutably.
    pub fn iter_chunks_mut(&mut self) -> impl Iterator<Item = (IVec3, &mut ChunkStorage)> {
        self.chunks.iter_mut().map(|(k, v)| (*k, v))
    }

    /// Number of chunks in the grid.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// All chunk keys.
    pub fn chunk_keys(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.chunks.keys().copied()
    }

    /// Get a chunk's 6 face neighbors for meshing.
    /// Returns `[+X, -X, +Y, -Y, +Z, -Z]`.
    pub fn get_chunk_neighbors(&self, key: IVec3) -> [Option<&ChunkStorage>; 6] {
        [
            self.chunks.get(&(key + IVec3::X)),
            self.chunks.get(&(key - IVec3::X)),
            self.chunks.get(&(key + IVec3::Y)),
            self.chunks.get(&(key - IVec3::Y)),
            self.chunks.get(&(key + IVec3::Z)),
            self.chunks.get(&(key - IVec3::Z)),
        ]
    }

    /// Insert a complete chunk (for deserialization or procedural generation).
    pub fn insert_chunk(&mut self, key: IVec3, chunk: ChunkStorage) {
        self.chunks.insert(key, chunk);
    }

    // -----------------------------------------------------------------------
    // Spatial queries
    // -----------------------------------------------------------------------

    /// Compute the axis-aligned bounding box of all non-air blocks in world space.
    /// Returns `None` if the grid is completely empty.
    ///
    /// This is a **tight** block-level bounding box — it scans every non-air block
    /// in every chunk to find the exact min/max. Use `solid_bounding_box_cached()`
    /// to avoid recomputing every frame.
    pub fn bounding_box(&self) -> Option<(IVec3, IVec3)> {
        let mut min = IVec3::MAX;
        let mut max = IVec3::MIN;
        let mut found = false;

        let cs = CHUNK_SIZE as i32;
        for (&key, chunk) in &self.chunks {
            if chunk.is_empty() {
                continue;
            }
            let chunk_origin = key * cs;

            for x in 0..CHUNK_SIZE as u8 {
                for y in 0..CHUNK_SIZE as u8 {
                    for z in 0..CHUNK_SIZE as u8 {
                        if !chunk.get_block(x, y, z).is_air() {
                            let wp = chunk_origin + IVec3::new(x as i32, y as i32, z as i32);
                            if !found {
                                min = wp;
                                max = wp;
                                found = true;
                            } else {
                                min = min.min(wp);
                                max = max.max(wp);
                            }
                        }
                    }
                }
            }
        }

        if found { Some((min, max)) } else { None }
    }

    /// Total number of non-air blocks across all chunks.
    pub fn total_block_count(&self) -> u64 {
        self.chunks.values().map(|c| c.non_air_count() as u64).sum()
    }

    /// Whether the grid has any non-air blocks.
    pub fn is_empty(&self) -> bool {
        self.chunks.values().all(|c| c.is_empty())
    }

    /// Remove chunks that are completely empty (all air, no metadata).
    /// Call periodically to reclaim memory from demolished areas.
    pub fn compact(&mut self) {
        self.chunks.retain(|_, chunk| {
            !chunk.is_empty() || chunk.meta_count() > 0
        });
    }

    // -----------------------------------------------------------------------
    // Collider generation helpers
    // -----------------------------------------------------------------------

    /// Collect all solid block positions in a chunk as `(lx, ly, lz)` triples.
    /// Used to generate per-chunk Rapier colliders.
    pub fn solid_blocks_in_chunk(
        &self,
        key: IVec3,
        registry: &BlockRegistry,
    ) -> Vec<(u8, u8, u8)> {
        let chunk = match self.chunks.get(&key) {
            Some(c) => c,
            None => return Vec::new(),
        };

        if chunk.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        for x in 0..CHUNK_SIZE as u8 {
            for y in 0..CHUNK_SIZE as u8 {
                for z in 0..CHUNK_SIZE as u8 {
                    let id = chunk.get_block(x, y, z);
                    if !id.is_air() && registry.is_solid(id) {
                        result.push((x, y, z));
                    }
                }
            }
        }
        result
    }

    /// Generate a Rapier-compatible compound collider description for a chunk.
    ///
    /// Returns a list of `(position, half_extents)` pairs where each entry is
    /// a unit cube at the block's world position. The caller creates the actual
    /// Rapier `ColliderBuilder::compound()` from these.
    ///
    /// The positions are in **ship-local coordinates** (world-space block coords
    /// offset to center the ship at origin or wherever the ship's physics origin is).
    ///
    /// `origin_offset` is subtracted from each block position to center colliders
    /// around the ship's center of mass or a chosen reference point.
    /// Generate collider shapes for a chunk, optionally filtered by sub-grid membership.
    ///
    /// - `sub_grid_filter = None`: include only blocks NOT in any sub-grid (root grid).
    /// - `sub_grid_filter = Some(id)`: include only blocks in that specific sub-grid.
    pub fn chunk_collider_shapes(
        &self,
        key: IVec3,
        origin_offset: glam::Vec3,
        registry: &BlockRegistry,
    ) -> Vec<(glam::Vec3, glam::Vec3)> {
        self.chunk_collider_shapes_filtered(key, origin_offset, registry, None)
    }

    /// Generate collider shapes for a chunk, filtered by sub-grid membership.
    pub fn chunk_collider_shapes_filtered(
        &self,
        key: IVec3,
        origin_offset: glam::Vec3,
        registry: &BlockRegistry,
        sub_grid_filter: Option<u32>,
    ) -> Vec<(glam::Vec3, glam::Vec3)> {
        let solids = self.solid_blocks_in_chunk(key, registry);
        let cs_i = CHUNK_SIZE as i32;
        let cs = CHUNK_SIZE as f32;
        let chunk_origin = glam::Vec3::new(
            key.x as f32 * cs,
            key.y as f32 * cs,
            key.z as f32 * cs,
        );

        let mut result: Vec<(glam::Vec3, glam::Vec3)> = solids
            .iter()
            .filter(|&&(lx, ly, lz)| {
                let world_pos = IVec3::new(
                    key.x * cs_i + lx as i32,
                    key.y * cs_i + ly as i32,
                    key.z * cs_i + lz as i32,
                );
                let block_sg = self.sub_grid_id(world_pos);
                match sub_grid_filter {
                    None => block_sg == 0,       // root grid: only blocks NOT in any sub-grid
                    Some(id) => block_sg == id,   // specific sub-grid
                }
            })
            .map(|&(lx, ly, lz)| {
                let world_pos = chunk_origin + glam::Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5);
                let pos = world_pos - origin_offset;
                let half_extents = glam::Vec3::splat(0.5);
                (pos, half_extents)
            })
            .collect();

        // Sub-block element colliders: thin boxes on block faces.
        if let Some(chunk) = self.chunks.get(&key) {
            for (flat_idx, elements) in chunk.iter_sub_blocks() {
                let (lx, ly, lz) = super::palette::index_to_xyz(flat_idx as usize);
                let block_center = chunk_origin
                    + glam::Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5)
                    - origin_offset;

                for elem in elements {
                    let (offset, half_extents) =
                        sub_block_collider_shape(elem.face, &elem.element_type);
                    result.push((block_center + offset, half_extents));
                }
            }
        }

        result
    }
}

impl super::block_grid::BlockGridView for ShipGrid {
    fn iter_blocks(&self) -> Box<dyn Iterator<Item = (IVec3, BlockId)> + '_> {
        let cs = CHUNK_SIZE as i32;
        Box::new(self.chunks.iter().flat_map(move |(&key, chunk)| {
            let chunk_origin = key * cs;
            (0..CHUNK_SIZE as u8).flat_map(move |x| {
                (0..CHUNK_SIZE as u8).flat_map(move |y| {
                    (0..CHUNK_SIZE as u8).filter_map(move |z| {
                        let id = chunk.get_block(x, y, z);
                        if id.is_air() {
                            None
                        } else {
                            let wp = chunk_origin + IVec3::new(x as i32, y as i32, z as i32);
                            Some((wp, id))
                        }
                    })
                })
            })
        }))
    }

    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.get_block(x, y, z)
    }

    fn get_meta(&self, x: i32, y: i32, z: i32) -> Option<&BlockMeta> {
        let (key, lx, ly, lz) = Self::world_to_chunk(x, y, z);
        self.chunks.get(&key)?.get_meta(lx, ly, lz)
    }
}

/// Compute the collider offset and half-extents for a sub-block element on a face.
/// Returns (offset_from_block_center, half_extents).
fn sub_block_collider_shape(
    face: u8,
    element_type: &sub_block::SubBlockType,
) -> (glam::Vec3, glam::Vec3) {
    use sub_block::SubBlockType;

    // Element thickness and size on the face, depending on type.
    let (face_half_u, face_half_v, thickness) = match element_type {
        SubBlockType::PowerWire | SubBlockType::SignalWire | SubBlockType::Cable => {
            (0.4, 0.04, 0.04)
        }
        SubBlockType::Rail | SubBlockType::ConveyorBelt => {
            (0.4, 0.15, 0.06)
        }
        SubBlockType::Pipe | SubBlockType::PipeValve | SubBlockType::PipePump => {
            (0.4, 0.1, 0.1)
        }
        SubBlockType::Ladder => {
            (0.35, 0.45, 0.05)
        }
        SubBlockType::RotorMount | SubBlockType::HingeMount => {
            (0.2, 0.2, 0.08)
        }
        SubBlockType::PistonMount | SubBlockType::SliderMount => {
            (0.15, 0.3, 0.08)
        }
        _ => {
            (0.15, 0.15, 0.05)
        }
    };

    // Offset from block center to the face surface + half thickness outward.
    let face_offset = 0.5 + thickness * 0.5;
    let offset = match face {
        0 => glam::Vec3::new(face_offset, 0.0, 0.0),   // +X
        1 => glam::Vec3::new(-face_offset, 0.0, 0.0),  // -X
        2 => glam::Vec3::new(0.0, face_offset, 0.0),   // +Y
        3 => glam::Vec3::new(0.0, -face_offset, 0.0),  // -Y
        4 => glam::Vec3::new(0.0, 0.0, face_offset),   // +Z
        5 => glam::Vec3::new(0.0, 0.0, -face_offset),  // -Z
        _ => glam::Vec3::ZERO,
    };

    // Half-extents: face_half_u and face_half_v are in the plane of the face,
    // thickness/2 is perpendicular. Map to XYZ based on which axis the face is on.
    let half_extents = match face {
        0 | 1 => glam::Vec3::new(thickness * 0.5, face_half_v, face_half_u), // X face: thin in X
        2 | 3 => glam::Vec3::new(face_half_u, thickness * 0.5, face_half_v), // Y face: thin in Y
        4 | 5 => glam::Vec3::new(face_half_u, face_half_v, thickness * 0.5), // Z face: thin in Z
        _ => glam::Vec3::splat(0.1),
    };

    (offset, half_extents)
}

// ---------------------------------------------------------------------------
// Starter ship builder
// ---------------------------------------------------------------------------

/// Ship layout specification — defines the initial ship structure from blocks.
pub struct StarterShipLayout {
    /// Ship dimensions in blocks (width_x, height_y, length_z).
    pub width: u8,
    pub height: u8,
    pub length: u8,
}

impl StarterShipLayout {
    /// The default starter ship: 10×6×16 block hull with interior space.
    /// Interior: 8×4×14 blocks (8m wide, 4m tall, 14m long).
    /// 4m interior height comfortably fits the player (1.8m capsule + headroom).
    pub fn default_starter() -> Self {
        Self {
            width: 10,
            height: 6,
            length: 16,
        }
    }
}

/// Build the default starter ship in a ShipGrid.
///
/// The ship is centered around the origin: X from `-width/2` to `width/2-1`,
/// Y from `0` to `height-1`, Z from `-length/2` to `length/2-1`.
///
/// Structure:
/// - Hull walls on all 6 faces (1 block thick)
/// - Interior is air
/// - Front wall has window blocks (cockpit glass)
/// - Right wall (+X) has a door opening at Z=0
/// - Cockpit block placed at the front interior
/// - Ownership core block placed at center
pub fn build_starter_ship(layout: &StarterShipLayout) -> ShipGrid {
    let mut grid = ShipGrid::new();

    let w = layout.width as i32;
    let h = layout.height as i32;
    let l = layout.length as i32;

    let x_min = -(w / 2);
    let x_max = x_min + w - 1;
    let y_min = 0;
    let y_max = h - 1;
    let z_min = -(l / 2);
    let z_max = z_min + l - 1;

    // Door opening parameters (on the +X wall, centered at Z=0)
    let door_z_min = -1i32;
    let door_z_max = 0i32;
    let door_y_max = (h - 2).min(3); // door is 3 blocks tall (fits player)

    for x in x_min..=x_max {
        for y in y_min..=y_max {
            for z in z_min..=z_max {
                let on_x_min = x == x_min;
                let on_x_max = x == x_max;
                let on_y_min = y == y_min;
                let on_y_max = y == y_max;
                let on_z_min = z == z_min;
                let on_z_max = z == z_max;

                let is_boundary = on_x_min || on_x_max || on_y_min || on_y_max || on_z_min || on_z_max;

                if !is_boundary {
                    // Interior: air
                    continue;
                }

                // Door opening on +X wall — leave as air (no block, no collider).
                if on_x_max && !on_y_min && !on_y_max && !on_z_min && !on_z_max {
                    if z >= door_z_min && z <= door_z_max && y >= 1 && y <= door_y_max {
                        continue; // air gap — player can walk through
                    }
                }

                // Front wall: window blocks for cockpit
                if on_z_min && !on_x_min && !on_x_max && !on_y_min && !on_y_max {
                    grid.set_block(x, y, z, BlockId::WINDOW);
                    continue;
                }

                // Floor, ceiling, and remaining walls: hull
                grid.set_block(x, y, z, BlockId::HULL_STANDARD);
            }
        }
    }

    // Place cockpit at interior, center. Seated 2 blocks back from the
    // front glass so the pilot isn't pressed against the canopy — there
    // is clear reading distance to the HUD panels placed on the glass
    // (see HUD panel loop below).
    let cockpit_z = z_min + 3;
    grid.set_block(0, 1, cockpit_z, BlockId::COCKPIT);

    // Place ownership core at center of ship
    grid.set_block(0, 1, 0, BlockId::OWNERSHIP_CORE);

    // Reactor and battery placed inside the hull for power generation.
    grid.set_block(0, 1, 1, BlockId::REACTOR_SMALL);
    grid.set_block(0, 1, 2, BlockId::BATTERY);

    // Reactor wireless power configuration: two circuits.
    // "main" at full power for linear thrusters, "rcs" at 30% for rotation thrusters.
    let reactor_pos = IVec3::new(0, 1, 1);
    grid.set_power_config(0, 1, 1, PowerConfig::Source {
        circuits: vec![("main".into(), 1.0), ("rcs".into(), 0.3)],
        broadcast_range: 50.0,
    });

    // -----------------------------------------------------------------------
    // Thrusters: 48 linear + 12 RCS = 60 total
    // -----------------------------------------------------------------------
    // Each thruster protrudes 1 block outside the hull.
    // facing_direction = exhaust direction; thrust (reaction) = -facing.
    // Channel overrides assign each thruster to a throttle signal channel.
    // PowerConfig::Consumer links each thruster to the reactor wirelessly.

    // Helper: place thruster + set throttle channel + set power config.
    let place_thruster = |grid: &mut ShipGrid, x: i32, y: i32, z: i32, normal: IVec3, channel: &str, power_circuit: &str| {
        grid.set_block(x, y, z, BlockId::THRUSTER_SMALL_CHEMICAL);
        grid.set_orientation(x, y, z, BlockOrientation::from_face_normal(normal));
        grid.set_channel_override(x, y, z, channel);
        grid.set_power_config(x, y, z, PowerConfig::Consumer {
            reactor_pos,
            circuit: power_circuit.to_string(),
        });
    };

    // Symmetric position pairs around CoM ≈ (0, 3, 0).
    // Y pairs: (y_min=0, y_max=5) → offsets (-2.5, +2.5) ✓
    //          (1, 4) → offsets (-1.5, +1.5) ✓
    // X pairs: (x_min=-5, x_max=4) → centers (-4.5, 4.5) ✓
    //          (-3, 2) → centers (-2.5, 2.5) ✓
    // Z pairs: (z_min=-8, z_max=7) → centers (-7.5, 7.5) ✓
    //          (-4, 3) → centers (-3.5, 3.5) ✓

    // --- LINEAR THRUSTERS (8 per direction = 48 total, fully symmetric) ---

    // Aft (z_max+1): exhaust +Z, thrust -Z (forward). 8 in a 4×2 grid.
    for &x in &[x_min, -3, 2, x_max] {
        for &y in &[y_min, y_max] {
            place_thruster(&mut grid, x, y, z_max + 1, IVec3::new(0, 0, 1), "thrust-forward", "main");
        }
    }

    // Fore: on sides near bow. Exhaust -Z, thrust +Z (reverse). 8 symmetric.
    for &y in &[y_min, 1, 4, y_max] {
        place_thruster(&mut grid, x_min - 1, y, z_min + 1, IVec3::new(0, 0, -1), "thrust-reverse", "main");
        place_thruster(&mut grid, x_max + 1, y, z_min + 1, IVec3::new(0, 0, -1), "thrust-reverse", "main");
    }

    // Port (x_min-1): exhaust -X, thrust +X (right). 8 in a 4×2 grid.
    for &z in &[z_min, -4, 3, z_max] {
        for &y in &[y_min, y_max] {
            place_thruster(&mut grid, x_min - 1, y, z, IVec3::new(-1, 0, 0), "thrust-right", "main");
        }
    }

    // Starboard (x_max+1): exhaust +X, thrust -X (left). 8 in a 4×2 grid.
    for &z in &[z_min, -4, 3, z_max] {
        for &y in &[y_min, y_max] {
            place_thruster(&mut grid, x_max + 1, y, z, IVec3::new(1, 0, 0), "thrust-left", "main");
        }
    }

    // Bottom (y_min-1): exhaust -Y, thrust +Y (up). 8 in a 4×2 grid.
    for &x in &[x_min, -3, 2, x_max] {
        for &z in &[z_min, z_max] {
            place_thruster(&mut grid, x, y_min - 1, z, IVec3::new(0, -1, 0), "thrust-up", "main");
        }
    }

    // Top (y_max+1): exhaust +Y, thrust -Y (down). 8 in a 4×2 grid.
    for &x in &[x_min, -3, 2, x_max] {
        for &z in &[z_min, z_max] {
            place_thruster(&mut grid, x, y_max + 1, z, IVec3::new(0, 1, 0), "thrust-down", "main");
        }
    }

    // --- RCS THRUSTERS (12 total: 2 per rotation direction) ---
    // Each couple pair is placed so cross-axis torque cancels to zero:
    // - Yaw: both at same Y → no roll contamination
    // - Pitch: both at same X → no roll contamination
    // - Roll: both at same Z → no yaw contamination
    // All RCS on "rcs" circuit (30% power → less responsive rotation).

    // Yaw CW (-Y torque): port-bow pushes +X, starboard-stern pushes -X. Both at y=2.
    place_thruster(&mut grid, x_min - 1, 2, z_min + 2, IVec3::new(-1, 0, 0), "torque-yaw-cw", "rcs");
    place_thruster(&mut grid, x_max + 1, 2, z_max - 2, IVec3::new(1, 0, 0), "torque-yaw-cw", "rcs");
    // Yaw CCW (+Y torque): starboard-bow pushes -X, port-stern pushes +X. Both at y=2.
    place_thruster(&mut grid, x_max + 1, 2, z_min + 2, IVec3::new(1, 0, 0), "torque-yaw-ccw", "rcs");
    place_thruster(&mut grid, x_min - 1, 2, z_max - 2, IVec3::new(-1, 0, 0), "torque-yaw-ccw", "rcs");

    // Pitch up (-X torque): top-bow pushes -Y, bottom-stern pushes +Y. Both at x=0.
    place_thruster(&mut grid, 0, y_max + 1, z_min + 2, IVec3::new(0, 1, 0), "torque-pitch-up", "rcs");
    place_thruster(&mut grid, 0, y_min - 1, z_max - 2, IVec3::new(0, -1, 0), "torque-pitch-up", "rcs");
    // Pitch down (+X torque): top-stern pushes -Y, bottom-bow pushes +Y. Both at x=0.
    place_thruster(&mut grid, 0, y_max + 1, z_max - 2, IVec3::new(0, 1, 0), "torque-pitch-down", "rcs");
    place_thruster(&mut grid, 0, y_min - 1, z_min + 2, IVec3::new(0, -1, 0), "torque-pitch-down", "rcs");

    // Roll CW (-Z torque): port-top pushes +X, starboard-bottom pushes -X. Both at z=0.
    place_thruster(&mut grid, x_min - 1, y_max, 0, IVec3::new(-1, 0, 0), "torque-roll-cw", "rcs");
    place_thruster(&mut grid, x_max + 1, y_min, 0, IVec3::new(1, 0, 0), "torque-roll-cw", "rcs");
    // Roll CCW (+Z torque): port-bottom pushes +X, starboard-top pushes -X. Both at z=0.
    place_thruster(&mut grid, x_min - 1, y_min, 0, IVec3::new(-1, 0, 0), "torque-roll-ccw", "rcs");
    place_thruster(&mut grid, x_max + 1, y_max, 0, IVec3::new(1, 0, 0), "torque-roll-ccw", "rcs");

    // --- CRUISE DRIVE ---
    // Boost module: subscribes to "cruise" (C key), publishes boost to "boost-accel".
    grid.set_block(0, 1, 3, BlockId::CRUISE_DRIVE_SMALL);
    grid.set_channel_override(0, 1, 3, "cruise");
    grid.set_boost_channel(0, 1, 3, "boost-accel");
    grid.set_power_config(0, 1, 3, PowerConfig::Consumer {
        reactor_pos,
        circuit: "main".to_string(),
    });

    // Forward thrusters subscribe to "boost-accel" for cruise boost.
    for &x in &[x_min, -3, 2, x_max] {
        for &y in &[y_min, y_max] {
            grid.set_boost_channel(x, y, z_max + 1, "boost-accel");
        }
    }
    // Reverse thrusters subscribe to "boost-brake" (future: separate brake boost).
    for &y in &[y_min, 1, 4, y_max] {
        grid.set_boost_channel(x_min - 1, y, z_min + 1, "boost-brake");
        grid.set_boost_channel(x_max + 1, y, z_min + 1, "boost-brake");
    }

    // --- Ship system blocks ---
    // Each block placed individually with its own power connection.
    // Channel wiring is automatic via seat_presets defaults matching the pilot seat.

    // Flight computer: angular damping. Toggled via F1 key → "flight-computer-toggle" channel.
    grid.set_block(-2, 1, -4, BlockId::FLIGHT_COMPUTER);
    grid.set_power_config(-2, 1, -4, PowerConfig::Consumer {
        reactor_pos,
        circuit: "main".to_string(),
    });

    // Hover module: 6-DOF hover. Toggled via H key → "atmo-comp" channel.
    grid.set_block(-2, 1, -3, BlockId::HOVER_MODULE);
    grid.set_power_config(-2, 1, -3, PowerConfig::Consumer {
        reactor_pos,
        circuit: "main".to_string(),
    });

    // Autopilot: target tracking. Toggled via T key → "autopilot-engage" channel.
    grid.set_block(-2, 1, -2, BlockId::AUTOPILOT);
    grid.set_power_config(-2, 1, -2, PowerConfig::Consumer {
        reactor_pos,
        circuit: "main".to_string(),
    });

    // Warp computer: star selection + warp initiation.
    // G key → "warp-cycle", Enter → "warp-accept", Backspace → "warp-cancel".
    grid.set_block(-2, 1, -1, BlockId::WARP_COMPUTER);
    grid.set_power_config(-2, 1, -1, PowerConfig::Consumer {
        reactor_pos,
        circuit: "main".to_string(),
    });

    // Engine controller: master on/off toggle. X key → "engine-toggle" channel.
    grid.set_block(-2, 1, 4, BlockId::ENGINE_CONTROLLER);
    grid.set_power_config(-2, 1, 4, PowerConfig::Consumer {
        reactor_pos,
        circuit: "main".to_string(),
    });

    // --- Sub-block elements: decorative only (power is wireless now) ---
    use sub_block::{SubBlockElement, SubBlockType};

    // Cockpit HUD canopy: attach a `HudPanel` sub-block to the pilot-
    // facing (+Z, face 4) side of every window block in the front wall.
    // Widget kind stays `None` by default — client-side config
    // (HudPanelConfigs) lets the pilot pick Gauge/Numeric/Toggle/Text/
    // Button + channel later via F on each panel. Placing them now
    // guarantees the glass is pre-canopied with interactive surfaces
    // the player can customise immediately on connect.
    //
    // Window block layout (lines above): x ∈ [-3, 3], y ∈ [1, 4],
    // z = z_min. Skipping the outer edges of the frame (`!on_x_min`,
    // `!on_x_max`, `!on_y_min`, `!on_y_max`) yields a 7×4 grid of 28
    // HUD panels.
    for y in 1..=(y_max - 1) {
        for x in (x_min + 1)..=(x_max - 1) {
            grid.add_sub_block(
                x,
                y,
                z_min,
                SubBlockElement {
                    face: 4, // +Z, inward toward pilot
                    element_type: SubBlockType::HudPanel,
                    rotation: 0,
                    flags: 0,
                },
            );
        }
    }

    // Decorative cable along the interior ceiling center (visual only).
    for z in (z_min + 2)..=(z_max - 2) {
        grid.add_sub_block(0, y_max, z, SubBlockElement {
            face: 3,
            element_type: SubBlockType::Cable,
            rotation: 0,
            flags: 0,
        });
    }

    // Ladder on the interior left wall (face 0 = +X).
    // Face 0 (+X) tangents: u=-Z, v=+Y. Side rails run along v (vertical), rungs along u (horizontal).
    for y in 1..y_max {
        grid.add_sub_block(x_min, y, 0, SubBlockElement {
            face: 0,
            element_type: SubBlockType::Ladder,
            rotation: 0, // rails vertical (v=+Y), rungs horizontal (u=-Z)
            flags: 0,
        });
    }

    // Rail on the floor near the door (face 2 = +Y).
    // Face 2 (+Y) tangents: u=+X, v=+Z. Rail chain runs along Z → rotation=1 (swap to v axis).
    for z in -3..=3 {
        grid.add_sub_block(x_max - 1, y_min, z, SubBlockElement {
            face: 2,
            element_type: SubBlockType::Rail,
            rotation: 1,
            flags: 0,
        });
    }

    // Surface light near the cockpit.
    grid.add_sub_block(0, y_max, z_min + 2, SubBlockElement {
        face: 3, // -Y (ceiling underside)
        element_type: SubBlockType::SurfaceLight,
        rotation: 0,
        flags: 0,
    });

    grid
}

// ---------------------------------------------------------------------------
// Sub-grid membership computation (BFS flood-fill)
// ---------------------------------------------------------------------------

/// Compute which blocks belong to a mechanical mount's child sub-grid.
///
/// Starting from the block on the output side of the mount (mount_pos + face_offset),
/// BFS through all face-adjacent solid blocks. The mount block itself is NOT crossed.
/// Returns the set of block positions in the child sub-grid (may be empty if nothing
/// is attached to the output face).
/// Compute sub-grid members via BFS from a mechanical mount's output face.
///
/// `own_sg_id` is the sub-grid ID being computed (0 for new sub-grids).
/// The BFS stops at blocks that belong to a DIFFERENT sub-grid (sub-grid boundary).
/// This prevents nested mounts from having their members stolen by a parent's re-BFS.
pub fn compute_sub_grid_members(
    grid: &ShipGrid,
    registry: &super::registry::BlockRegistry,
    mount_pos: IVec3,
    output_face: u8,
    own_sg_id: u32,
) -> std::collections::HashSet<IVec3> {
    use std::collections::{HashSet, VecDeque};

    let seed = mount_pos + sub_block::face_to_offset(output_face);
    let seed_block = grid.get_block(seed.x, seed.y, seed.z);
    if seed_block == BlockId::AIR {
        return HashSet::new();
    }

    let offsets = [
        IVec3::X, IVec3::NEG_X,
        IVec3::Y, IVec3::NEG_Y,
        IVec3::Z, IVec3::NEG_Z,
    ];

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(seed);
    queue.push_back(seed);

    while let Some(pos) = queue.pop_front() {
        for &d in &offsets {
            let neighbor = pos + d;
            if neighbor == mount_pos { continue; } // don't cross the joint
            if visited.contains(&neighbor) { continue; }
            let block = grid.get_block(neighbor.x, neighbor.y, neighbor.z);
            if block == BlockId::AIR || !registry.is_solid(block) { continue; }
            // Don't cross into blocks owned by a different sub-grid.
            let neighbor_sg = grid.sub_grid_id(neighbor);
            if neighbor_sg != 0 && neighbor_sg != own_sg_id {
                continue; // belongs to another sub-grid — boundary
            }
            visited.insert(neighbor);
            queue.push_back(neighbor);
        }
    }

    visited
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_chunk_positive() {
        let (key, lx, ly, lz) = ShipGrid::world_to_chunk(0, 0, 0);
        assert_eq!(key, IVec3::ZERO);
        assert_eq!((lx, ly, lz), (0, 0, 0));

        let (key, lx, ly, lz) = ShipGrid::world_to_chunk(61, 61, 61);
        assert_eq!(key, IVec3::ZERO);
        assert_eq!((lx, ly, lz), (61, 61, 61));

        let (key, lx, ly, lz) = ShipGrid::world_to_chunk(62, 0, 0);
        assert_eq!(key, IVec3::new(1, 0, 0));
        assert_eq!((lx, ly, lz), (0, 0, 0));
    }

    #[test]
    fn world_to_chunk_negative() {
        let (key, lx, ly, lz) = ShipGrid::world_to_chunk(-1, 0, 0);
        assert_eq!(key, IVec3::new(-1, 0, 0));
        assert_eq!((lx, ly, lz), (61, 0, 0));

        let (key, lx, ly, lz) = ShipGrid::world_to_chunk(-62, 0, 0);
        assert_eq!(key, IVec3::new(-1, 0, 0));
        assert_eq!((lx, ly, lz), (0, 0, 0));

        let (key, lx, ly, lz) = ShipGrid::world_to_chunk(-63, 0, 0);
        assert_eq!(key, IVec3::new(-2, 0, 0));
        assert_eq!((lx, ly, lz), (61, 0, 0));
    }

    #[test]
    fn chunk_to_world_roundtrip() {
        for wx in [-130, -63, -62, -1, 0, 1, 61, 62, 63, 130] {
            for wy in [-1, 0, 30, 61, 62] {
                for wz in [-1, 0, 61, 62] {
                    let (key, lx, ly, lz) = ShipGrid::world_to_chunk(wx, wy, wz);
                    let (rx, ry, rz) = ShipGrid::chunk_to_world(key, lx, ly, lz);
                    assert_eq!((wx, wy, wz), (rx, ry, rz), "roundtrip failed for ({wx},{wy},{wz})");
                }
            }
        }
    }

    #[test]
    fn get_set_across_chunks() {
        let mut grid = ShipGrid::new();
        assert!(grid.is_empty());

        // Place blocks in different chunks
        grid.set_block(0, 0, 0, BlockId::STONE);
        grid.set_block(62, 0, 0, BlockId::DIRT); // next chunk in +X
        grid.set_block(-1, 0, 0, BlockId::GRASS); // previous chunk in -X

        assert_eq!(grid.get_block(0, 0, 0), BlockId::STONE);
        assert_eq!(grid.get_block(62, 0, 0), BlockId::DIRT);
        assert_eq!(grid.get_block(-1, 0, 0), BlockId::GRASS);
        assert_eq!(grid.get_block(5, 5, 5), BlockId::AIR);

        assert_eq!(grid.chunk_count(), 3);
        assert_eq!(grid.total_block_count(), 3);
    }

    #[test]
    fn bounding_box() {
        let mut grid = ShipGrid::new();
        assert!(grid.bounding_box().is_none());

        grid.set_block(5, 10, 15, BlockId::STONE);
        let (min, max) = grid.bounding_box().unwrap();
        // Tight block-level bounds: single block at (5, 10, 15).
        assert_eq!(min, IVec3::new(5, 10, 15));
        assert_eq!(max, IVec3::new(5, 10, 15));

        grid.set_block(70, 0, 0, BlockId::DIRT); // chunk (1,0,0)
        let (min, max) = grid.bounding_box().unwrap();
        assert_eq!(min, IVec3::new(5, 0, 0));
        assert_eq!(max, IVec3::new(70, 10, 15));
    }

    #[test]
    fn compact_removes_empty_chunks() {
        let mut grid = ShipGrid::new();
        grid.set_block(0, 0, 0, BlockId::STONE);
        grid.set_block(0, 0, 0, BlockId::AIR); // remove it
        assert_eq!(grid.chunk_count(), 1); // chunk still exists

        grid.compact();
        assert_eq!(grid.chunk_count(), 0); // removed
    }

    #[test]
    fn starter_ship_structure() {
        let layout = StarterShipLayout::default_starter();
        let grid = build_starter_ship(&layout);
        let w = layout.width as i32;
        let h = layout.height as i32;
        let l = layout.length as i32;
        let x_min = -(w / 2);
        let z_min = -(l / 2);

        assert!(!grid.is_empty());

        // Floor should be hull
        assert_eq!(grid.get_block(x_min, 0, z_min), BlockId::HULL_STANDARD);
        assert_eq!(grid.get_block(0, 0, 0), BlockId::HULL_STANDARD);

        // Interior should be air (above floor, below ceiling, inside walls)
        assert_eq!(grid.get_block(0, 2, 1), BlockId::AIR);

        // Front wall should be windows (interior face of front wall)
        assert_eq!(grid.get_block(0, 2, z_min), BlockId::WINDOW);

        // Cockpit should be placed 2 blocks back from the front glass
        // so the pilot has clear reading distance to the HUD canopy.
        assert_eq!(grid.get_block(0, 1, z_min + 3), BlockId::COCKPIT);

        // Ownership core at center
        assert_eq!(grid.get_block(0, 1, 0), BlockId::OWNERSHIP_CORE);

        // Outside the ship should be air
        assert_eq!(grid.get_block(50, 50, 50), BlockId::AIR);
    }

    #[test]
    fn chunk_neighbors() {
        let mut grid = ShipGrid::new();
        let center = IVec3::ZERO;
        let right = IVec3::X;

        grid.set_block(0, 0, 0, BlockId::STONE); // chunk (0,0,0)
        grid.set_block(62, 0, 0, BlockId::DIRT); // chunk (1,0,0)

        let neighbors = grid.get_chunk_neighbors(center);
        assert!(neighbors[0].is_some()); // +X neighbor exists
        assert!(neighbors[1].is_none()); // -X doesn't
    }

    #[test]
    fn damage_block_world_coords() {
        let reg = BlockRegistry::new();
        let mut grid = ShipGrid::new();
        grid.set_block(5, 5, 5, BlockId::HULL_STANDARD);

        let result = grid.damage_block(5, 5, 5, 10, &reg);
        assert!(matches!(result, DamageResult::Damaged { .. }));

        // Damage non-existent chunk returns NoEffect
        let result = grid.damage_block(999, 999, 999, 10, &reg);
        assert_eq!(result, DamageResult::NoEffect);
    }

    #[test]
    fn collider_shapes() {
        let reg = BlockRegistry::new();
        let mut grid = ShipGrid::new();
        grid.set_block(0, 0, 0, BlockId::HULL_STANDARD);
        grid.set_block(1, 0, 0, BlockId::HULL_STANDARD);

        let shapes = grid.chunk_collider_shapes(IVec3::ZERO, glam::Vec3::ZERO, &reg);
        assert_eq!(shapes.len(), 2);

        // Each shape should be a unit half-extent cube
        for (_, he) in &shapes {
            assert_eq!(*he, glam::Vec3::splat(0.5));
        }
    }
}
