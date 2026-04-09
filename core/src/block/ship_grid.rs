use std::collections::HashMap;

use glam::IVec3;

use super::block_id::BlockId;
use super::block_meta::{BlockMeta, BlockOrientation};
use super::chunk_storage::{ChunkStorage, DamageResult};
use super::palette::CHUNK_SIZE;
use super::registry::BlockRegistry;
use super::sub_block;

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
}

impl ShipGrid {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            channel_overrides: HashMap::new(),
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
    pub fn chunk_collider_shapes(
        &self,
        key: IVec3,
        origin_offset: glam::Vec3,
        registry: &BlockRegistry,
    ) -> Vec<(glam::Vec3, glam::Vec3)> {
        let solids = self.solid_blocks_in_chunk(key, registry);
        let cs = CHUNK_SIZE as f32;
        let chunk_origin = glam::Vec3::new(
            key.x as f32 * cs,
            key.y as f32 * cs,
            key.z as f32 * cs,
        );

        let mut result: Vec<(glam::Vec3, glam::Vec3)> = solids
            .iter()
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

    // Place cockpit at the front interior, center
    let cockpit_z = z_min + 1;
    grid.set_block(0, 1, cockpit_z, BlockId::COCKPIT);

    // Place ownership core at center of ship
    grid.set_block(0, 1, 0, BlockId::OWNERSHIP_CORE);

    // Reactor and battery placed inside the hull for power generation.
    // No default wiring — the no-wire fallback grants full power until
    // the player places their first PowerWire sub-block.
    grid.set_block(0, 1, 1, BlockId::REACTOR_SMALL);
    grid.set_block(0, 1, 2, BlockId::BATTERY);

    // -----------------------------------------------------------------------
    // Thrusters: 48 linear + 12 RCS = 60 total
    // -----------------------------------------------------------------------
    // Each thruster protrudes 1 block outside the hull.
    // facing_direction = exhaust direction; thrust (reaction) = -facing.
    // Channel overrides assign each thruster to the correct signal channel.
    // Power wiring connects each thruster to the power bus.

    // Helper: place thruster + set channel override.
    let place_thruster = |grid: &mut ShipGrid, x: i32, y: i32, z: i32, normal: IVec3, channel: &str| {
        grid.set_block(x, y, z, BlockId::THRUSTER_SMALL_CHEMICAL);
        grid.set_orientation(x, y, z, BlockOrientation::from_face_normal(normal));
        grid.set_channel_override(x, y, z, channel);
    };

    // Mid-points for 8-per-face layout (corners + mid-edges).
    let x_mid = (x_min + x_max) / 2;
    let y_mid = (y_min + y_max) / 2;
    let z_mid = (z_min + z_max) / 2;

    // --- LINEAR THRUSTERS (8 per direction = 48 total) ---

    // Aft (z_max+1): exhaust +Z, thrust -Z (forward).
    for &x in &[x_min, x_mid, x_max] {
        for &y in &[y_min, y_max] {
            place_thruster(&mut grid, x, y, z_max + 1, IVec3::new(0, 0, 1), "thrust-forward");
        }
    }
    place_thruster(&mut grid, x_min, y_mid, z_max + 1, IVec3::new(0, 0, 1), "thrust-forward");
    place_thruster(&mut grid, x_max, y_mid, z_max + 1, IVec3::new(0, 0, 1), "thrust-forward");

    // Fore: on port/starboard sides near bow. Exhaust -Z, thrust +Z (reverse).
    for &y in &[y_min, y_mid, y_max] {
        place_thruster(&mut grid, x_min - 1, y, z_min + 1, IVec3::new(0, 0, -1), "thrust-reverse");
        place_thruster(&mut grid, x_max + 1, y, z_min + 1, IVec3::new(0, 0, -1), "thrust-reverse");
    }
    place_thruster(&mut grid, x_min - 1, y_min, z_min + 2, IVec3::new(0, 0, -1), "thrust-reverse");
    place_thruster(&mut grid, x_max + 1, y_max, z_min + 2, IVec3::new(0, 0, -1), "thrust-reverse");

    // Port (x_min-1): exhaust -X, thrust +X (right).
    for &z in &[z_min, z_mid, z_max] {
        for &y in &[y_min, y_max] {
            place_thruster(&mut grid, x_min - 1, y, z, IVec3::new(-1, 0, 0), "thrust-right");
        }
    }
    place_thruster(&mut grid, x_min - 1, y_mid, z_min, IVec3::new(-1, 0, 0), "thrust-right");
    place_thruster(&mut grid, x_min - 1, y_mid, z_max, IVec3::new(-1, 0, 0), "thrust-right");

    // Starboard (x_max+1): exhaust +X, thrust -X (left).
    for &z in &[z_min, z_mid, z_max] {
        for &y in &[y_min, y_max] {
            place_thruster(&mut grid, x_max + 1, y, z, IVec3::new(1, 0, 0), "thrust-left");
        }
    }
    place_thruster(&mut grid, x_max + 1, y_mid, z_min, IVec3::new(1, 0, 0), "thrust-left");
    place_thruster(&mut grid, x_max + 1, y_mid, z_max, IVec3::new(1, 0, 0), "thrust-left");

    // Bottom (y_min-1): exhaust -Y, thrust +Y (up).
    for &x in &[x_min, x_mid, x_max] {
        for &z in &[z_min, z_max] {
            place_thruster(&mut grid, x, y_min - 1, z, IVec3::new(0, -1, 0), "thrust-up");
        }
    }
    place_thruster(&mut grid, x_min, y_min - 1, z_mid, IVec3::new(0, -1, 0), "thrust-up");
    place_thruster(&mut grid, x_max, y_min - 1, z_mid, IVec3::new(0, -1, 0), "thrust-up");

    // Top (y_max+1): exhaust +Y, thrust -Y (down).
    for &x in &[x_min, x_mid, x_max] {
        for &z in &[z_min, z_max] {
            place_thruster(&mut grid, x, y_max + 1, z, IVec3::new(0, 1, 0), "thrust-down");
        }
    }
    place_thruster(&mut grid, x_min, y_max + 1, z_mid, IVec3::new(0, 1, 0), "thrust-down");
    place_thruster(&mut grid, x_max, y_max + 1, z_mid, IVec3::new(0, 1, 0), "thrust-down");

    // --- RCS THRUSTERS (12 total: 4 per rotation axis) ---

    // Yaw CW (Y-axis): front-port pushes +X, back-starboard pushes -X.
    place_thruster(&mut grid, x_min - 1, y_mid, z_min + 1, IVec3::new(-1, 0, 0), "torque-yaw-cw");
    place_thruster(&mut grid, x_max + 1, y_mid, z_max - 1, IVec3::new(1, 0, 0), "torque-yaw-cw");
    // Yaw CCW: front-starboard pushes -X, back-port pushes +X.
    place_thruster(&mut grid, x_max + 1, y_mid, z_min + 1, IVec3::new(1, 0, 0), "torque-yaw-ccw");
    place_thruster(&mut grid, x_min - 1, y_mid, z_max - 1, IVec3::new(-1, 0, 0), "torque-yaw-ccw");

    // Pitch up (X-axis): top-front pushes +Z, bottom-back pushes -Z.
    place_thruster(&mut grid, x_mid, y_max + 1, z_min + 1, IVec3::new(0, 1, 0), "torque-pitch-up");
    place_thruster(&mut grid, x_mid, y_min - 1, z_max - 1, IVec3::new(0, -1, 0), "torque-pitch-up");
    // Pitch down: top-back pushes -Z, bottom-front pushes +Z.
    place_thruster(&mut grid, x_mid, y_max + 1, z_max - 1, IVec3::new(0, 1, 0), "torque-pitch-down");
    place_thruster(&mut grid, x_mid, y_min - 1, z_min + 1, IVec3::new(0, -1, 0), "torque-pitch-down");

    // Roll CW (Z-axis): left-top pushes -Y, right-bottom pushes +Y.
    place_thruster(&mut grid, x_min - 1, y_max, z_mid, IVec3::new(-1, 0, 0), "torque-roll-cw");
    place_thruster(&mut grid, x_max + 1, y_min, z_mid, IVec3::new(1, 0, 0), "torque-roll-cw");
    // Roll CCW: left-bottom pushes +Y, right-top pushes -Y.
    place_thruster(&mut grid, x_min - 1, y_min, z_mid, IVec3::new(-1, 0, 0), "torque-roll-ccw");
    place_thruster(&mut grid, x_max + 1, y_max, z_mid, IVec3::new(1, 0, 0), "torque-roll-ccw");

    // --- Sub-block elements: power wiring + decoration ---
    use sub_block::{SubBlockElement, SubBlockType};

    // Helper to add a PowerWire on a face.
    let pw = |grid: &mut ShipGrid, x: i32, y: i32, z: i32, face: u8| {
        grid.add_sub_block(x, y, z, SubBlockElement {
            face, element_type: SubBlockType::PowerWire, rotation: 0, flags: 0,
        });
    };

    // --- Power bus: reactor → floor perimeter → wall corners → ceiling perimeter → thrusters ---

    // Reactor (0,1,1) connects down to floor bus.
    pw(&mut grid, 0, 1, 1, 3); // -Y face (down to floor)
    pw(&mut grid, 0, y_min, 1, 2); // +Y face of floor block below reactor

    // Battery (0,1,2) connects down to floor bus.
    pw(&mut grid, 0, 1, 2, 3); // -Y face (down to floor)
    pw(&mut grid, 0, y_min, 2, 2); // +Y face of floor block below battery

    // Floor bus: +Y face of floor blocks running along Z axis (main bus line).
    for z in z_min..=z_max {
        pw(&mut grid, 0, y_min, z, 2);
    }
    // Floor bus: branch from center (x=0) to both walls along X.
    for x in x_min..=x_max {
        pw(&mut grid, x, y_min, z_min, 2);
        pw(&mut grid, x, y_min, z_max, 2);
    }

    // Wall verticals: wire up all 4 wall corners (interior face).
    // Left wall (x_min): face 0 (+X interior)
    for y in y_min..=y_max {
        pw(&mut grid, x_min, y, z_min, 0);
        pw(&mut grid, x_min, y, z_max, 0);
    }
    // Right wall (x_max): face 1 (-X interior)
    for y in y_min..=y_max {
        pw(&mut grid, x_max, y, z_min, 1);
        pw(&mut grid, x_max, y, z_max, 1);
    }

    // Ceiling bus: -Y face of ceiling blocks along edges.
    for x in x_min..=x_max {
        pw(&mut grid, x, y_max, z_min, 3);
        pw(&mut grid, x, y_max, z_max, 3);
    }

    // Thruster connections: each thruster block gets a PowerWire on the face
    // connecting it back to the hull, and the hull block gets one facing the thruster.
    // Aft thrusters (z_max+1): thruster face 5 (-Z) connects to hull face 4 (+Z).
    for &(tx, ty) in &[(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)] {
        pw(&mut grid, tx, ty, z_max + 1, 5); // thruster: -Z face
        pw(&mut grid, tx, ty, z_max, 4);     // hull: +Z face
    }
    // Fore thrusters (on sides at z_min+1): face 5 (-Z) from thruster side.
    for &(tx, ty) in &[(x_min - 1, y_min), (x_min - 1, y_max), (x_max + 1, y_min), (x_max + 1, y_max)] {
        // These are on port/starboard walls, not front face.
        // Connect via the wall block they're adjacent to.
        pw(&mut grid, tx, ty, z_min + 1, 5); // thruster: -Z
    }
    // Port thrusters (x_min-1): face 0 (+X) connects to hull face 1 (-X).
    for &(ty, tz) in &[(y_min, z_min), (y_max, z_min), (y_min, z_max), (y_max, z_max)] {
        pw(&mut grid, x_min - 1, ty, tz, 0); // thruster: +X
        pw(&mut grid, x_min, ty, tz, 1);     // hull: -X (already has wall wire)
    }
    // Starboard thrusters (x_max+1): face 1 (-X) connects to hull face 0 (+X).
    for &(ty, tz) in &[(y_min, z_min), (y_max, z_min), (y_min, z_max), (y_max, z_max)] {
        pw(&mut grid, x_max + 1, ty, tz, 1); // thruster: -X
        pw(&mut grid, x_max, ty, tz, 0);     // hull: +X
    }
    // Bottom thrusters (y_min-1): face 2 (+Y) connects to hull face 3 (-Y).
    for &(tx, tz) in &[(x_min, z_min), (x_max, z_min), (x_min, z_max), (x_max, z_max)] {
        pw(&mut grid, tx, y_min - 1, tz, 2); // thruster: +Y
        pw(&mut grid, tx, y_min, tz, 3);     // hull: -Y (floor bottom)
    }
    // Top thrusters (y_max+1): face 3 (-Y) connects to hull face 2 (+Y).
    for &(tx, tz) in &[(x_min, z_min), (x_max, z_min), (x_min, z_max), (x_max, z_max)] {
        pw(&mut grid, tx, y_max + 1, tz, 3); // thruster: -Y
        pw(&mut grid, tx, y_max, tz, 2);     // hull: +Y (ceiling top)
    }

    // Decorative cable along the interior ceiling center (visual only, not power).
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

        // Cockpit should be placed
        assert_eq!(grid.get_block(0, 1, z_min + 1), BlockId::COCKPIT);

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
