//! Per-ship interior-volume mask.
//!
//! Marks which air voxels inside a ship qualify as "interior" (i.e., a place
//! a player should be considered aboard the ship). Computed from the ship's
//! block grid by a two-gate algorithm: **flood-fill from semantic seeds**
//! AND **geometric enclosure**. Both gates must pass; together they handle
//! arbitrary ship shapes correctly:
//!
//! * **Protrusions (thrusters, antennas, landing gear)** — voxels in the
//!   gap between a protrusion and the main hull are never reachable from a
//!   seat without crossing solid hull, so the topology gate keeps them out.
//! * **Open hatches / doors** — the flood-fill leaks through them, but the
//!   geometric gate rejects leaked voxels (they sit in mostly open space
//!   outside the hull envelope).
//! * **L-shapes, multi-room, split cabins** — any ship with a seat in each
//!   cabin gets that cabin's interior marked; cabins with no seat are
//!   treated as storage / exterior.
//!
//! The mask is computed on the shard that owns the ship (ship-shard),
//! serialized per-chunk as packed `u64` bit arrays, and shipped to the
//! system / planet shards in `ShipColliderSync`. Every consumer can then
//! answer "is this ship-local point interior?" in O(1).

use std::collections::{HashMap, HashSet, VecDeque};

use glam::{IVec3, Vec3};

use super::palette::{block_index, index_to_xyz, CHUNK_SIZE, CHUNK_VOLUME};
use super::registry::{BlockRegistry, FunctionalBlockKind};
use super::ship_grid::ShipGrid;

/// Number of `u64` words required to pack `CHUNK_VOLUME` bits (one bit per
/// voxel). `ceil(238_328 / 64) = 3_724`.
pub const CHUNK_INTERIOR_WORDS: usize = (CHUNK_VOLUME + 63) / 64;

/// Minimum number of axis rays (of 6) that must hit ship geometry for a
/// voxel to count as "geometrically enclosed." Tuned so a cockpit with a
/// single open hatch passes (5/6) while a voxel in the open space next to a
/// thruster strut fails (1-2/6).
pub const INTERIOR_GEOMETRIC_MIN_HITS: u32 = 5;

/// Packed per-chunk bit array. Layout matches `palette::block_index(x,y,z)`
/// so bit `i` corresponds to the voxel at (x,y,z) where i = x*3844 + y*62 + z.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkInteriorBits {
    words: Box<[u64]>,
}

impl ChunkInteriorBits {
    pub fn zeroed() -> Self {
        Self { words: vec![0u64; CHUNK_INTERIOR_WORDS].into_boxed_slice() }
    }

    /// Build from a `Vec<u64>` of length `CHUNK_INTERIOR_WORDS`.
    /// Panics if the length is wrong (protocol round-trip invariant).
    pub fn from_words(words: Vec<u64>) -> Self {
        assert_eq!(
            words.len(),
            CHUNK_INTERIOR_WORDS,
            "interior bits word count mismatch"
        );
        Self { words: words.into_boxed_slice() }
    }

    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn set(&mut self, lx: u8, ly: u8, lz: u8) {
        let i = block_index(lx, ly, lz);
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    #[inline]
    pub fn get(&self, lx: u8, ly: u8, lz: u8) -> bool {
        let i = block_index(lx, ly, lz);
        (self.words[i / 64] >> (i % 64)) & 1 != 0
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    pub fn popcount(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

impl Default for ChunkInteriorBits {
    fn default() -> Self { Self::zeroed() }
}

/// Sparse per-ship interior mask keyed by chunk. Voxel coordinates are
/// ship-local (signed `i32` relative to ship origin).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InteriorMask {
    chunks: HashMap<IVec3, ChunkInteriorBits>,
}

impl InteriorMask {
    pub fn new() -> Self { Self::default() }

    pub fn is_empty(&self) -> bool { self.chunks.is_empty() }

    pub fn chunks(&self) -> impl Iterator<Item = (IVec3, &ChunkInteriorBits)> {
        self.chunks.iter().map(|(k, v)| (*k, v))
    }

    pub fn insert_chunk(&mut self, key: IVec3, bits: ChunkInteriorBits) {
        if bits.is_empty() {
            self.chunks.remove(&key);
            return;
        }
        self.chunks.insert(key, bits);
    }

    pub fn get_chunk(&self, key: IVec3) -> Option<&ChunkInteriorBits> {
        self.chunks.get(&key)
    }

    /// True if the given ship-local voxel is marked interior.
    #[inline]
    pub fn is_interior_voxel(&self, v: IVec3) -> bool {
        let cs = CHUNK_SIZE as i32;
        let key = IVec3::new(
            v.x.div_euclid(cs),
            v.y.div_euclid(cs),
            v.z.div_euclid(cs),
        );
        let lx = v.x.rem_euclid(cs) as u8;
        let ly = v.y.rem_euclid(cs) as u8;
        let lz = v.z.rem_euclid(cs) as u8;
        match self.chunks.get(&key) {
            Some(bits) => bits.get(lx, ly, lz),
            None => false,
        }
    }

    /// Continuous-coordinate query. Floors to voxel, then O(1) lookup.
    #[inline]
    pub fn is_interior(&self, ship_local: Vec3) -> bool {
        let v = IVec3::new(
            ship_local.x.floor() as i32,
            ship_local.y.floor() as i32,
            ship_local.z.floor() as i32,
        );
        self.is_interior_voxel(v)
    }

    /// Total marked voxels across all chunks. Diagnostic / telemetry.
    pub fn total_voxels(&self) -> u64 {
        self.chunks.values().map(|b| b.popcount() as u64).sum()
    }
}

/// Voxel-lookup adapter. Implemented for any grid; used by `compute` and by
/// external consumers that want to probe "is solid here" the same way.
///
/// Distinguishes between:
/// * **solid**: anything with collision (walls, thrusters, reactors, seats).
///   Terminates flood-fill — the player can't occupy it.
/// * **hull**: structural walls, floors, ceilings, windows. Used *only* by
///   the enclosure-raycast gate: a ray that hits a thruster or seat doesn't
///   mean the voxel is "inside the ship" — only a hit on a hull block does.
///   This prevents starter-ship-style thruster cages from tricking the
///   enclosure test into classifying open exterior space as interior.
pub trait SolidVoxelLookup {
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool;
    /// Default = same as `is_solid`. Override to distinguish structural hull
    /// blocks from functional/equipment blocks.
    fn is_hull(&self, x: i32, y: i32, z: i32) -> bool {
        self.is_solid(x, y, z)
    }
}

struct GridLookup<'a> {
    grid: &'a ShipGrid,
    registry: &'a BlockRegistry,
}
impl SolidVoxelLookup for GridLookup<'_> {
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.registry.is_solid(self.grid.get_block(x, y, z))
    }
    fn is_hull(&self, x: i32, y: i32, z: i32) -> bool {
        let id = self.grid.get_block(x, y, z);
        // Structural hull: solid AND not a functional block. Thrusters,
        // reactors, seats, batteries, etc. are functional — they ride
        // inside or protrude outside the hull but aren't the hull itself.
        self.registry.is_solid(id) && self.registry.functional_kind(id).is_none()
    }
}

/// Compute the interior mask for a ship from its block grid.
///
/// Seeds are the voxels directly above every `FunctionalBlockKind::Seat`
/// block (cockpit / seat / jump seat). BFS from all seeds simultaneously
/// through air voxels bounded by the solid-block AABB + 1-block pad; each
/// candidate voxel must additionally pass the geometric enclosure test
/// (`INTERIOR_GEOMETRIC_MIN_HITS` of 6 axis rays hit ship geometry within
/// the hull extent). Both gates must pass; failing voxels don't propagate,
/// so leaks through open hatches stop at the hull surface.
///
/// Returns an empty mask if the ship has no seats or no solid blocks.
pub fn compute(grid: &ShipGrid, registry: &BlockRegistry) -> InteriorMask {
    let Some((hull_min, hull_max)) = solid_block_aabb(grid, registry) else {
        return InteriorMask::default();
    };
    let pad = 1;
    let bounds_min = hull_min - IVec3::splat(pad);
    let bounds_max = hull_max + IVec3::splat(pad);
    let extent = hull_max - hull_min;
    let max_ray_steps = extent.x.max(extent.y).max(extent.z) + 2;

    let lookup = GridLookup { grid, registry };

    // Find seed voxels. A seat block itself is solid (the chair frame); the
    // player actually sits in the voxel above (headroom). Use that as the
    // seed — it's always interior by definition. If that voxel is also
    // solid for any reason, step up another block.
    let mut seeds: Vec<IVec3> = Vec::new();
    for (chunk_key, chunk) in grid.iter_chunks() {
        for idx in 0..CHUNK_VOLUME {
            let (lx, ly, lz) = index_to_xyz(idx);
            let bid = chunk.get_block(lx, ly, lz);
            if bid.is_air() { continue; }
            let Some(kind) = registry.functional_kind(bid) else { continue; };
            if !matches!(kind, FunctionalBlockKind::Seat) { continue; }
            let (wx, wy, wz) = ShipGrid::chunk_to_world(chunk_key, lx, ly, lz);
            for dy in 1..=2 {
                let s = IVec3::new(wx, wy + dy, wz);
                if !lookup.is_solid(s.x, s.y, s.z) {
                    seeds.push(s);
                    break;
                }
            }
        }
    }
    if seeds.is_empty() {
        return InteriorMask::default();
    }

    // BFS. `queued` prevents re-queuing the same voxel; `interior` collects
    // voxels that passed both gates. Failed-gate voxels never expand.
    let mut queued: HashSet<IVec3> = HashSet::new();
    let mut queue: VecDeque<IVec3> = VecDeque::new();
    let mut interior: HashSet<IVec3> = HashSet::new();
    for s in &seeds {
        if queued.insert(*s) {
            queue.push_back(*s);
        }
    }

    const DELTAS: [IVec3; 6] = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];

    while let Some(v) = queue.pop_front() {
        // Out-of-bounds guard (pad + 1 slack around the solid AABB).
        if v.x < bounds_min.x || v.x > bounds_max.x { continue; }
        if v.y < bounds_min.y || v.y > bounds_max.y { continue; }
        if v.z < bounds_min.z || v.z > bounds_max.z { continue; }
        // Skip solid voxels — walls and furniture don't become interior.
        if lookup.is_solid(v.x, v.y, v.z) { continue; }
        // Geometric enclosure gate.
        if !is_geometrically_enclosed(
            v,
            &lookup,
            hull_min,
            hull_max,
            max_ray_steps,
            INTERIOR_GEOMETRIC_MIN_HITS,
        ) {
            continue;
        }
        interior.insert(v);
        for delta in &DELTAS {
            let nb = v + *delta;
            if queued.insert(nb) {
                queue.push_back(nb);
            }
        }
    }

    // Pack into per-chunk bits.
    let mut mask = InteriorMask::default();
    let cs = CHUNK_SIZE as i32;
    for v in &interior {
        let key = IVec3::new(
            v.x.div_euclid(cs),
            v.y.div_euclid(cs),
            v.z.div_euclid(cs),
        );
        let lx = v.x.rem_euclid(cs) as u8;
        let ly = v.y.rem_euclid(cs) as u8;
        let lz = v.z.rem_euclid(cs) as u8;
        mask.chunks
            .entry(key)
            .or_insert_with(ChunkInteriorBits::zeroed)
            .set(lx, ly, lz);
    }
    mask
}

/// AABB (inclusive) over every solid block in the ship grid.
fn solid_block_aabb(grid: &ShipGrid, registry: &BlockRegistry) -> Option<(IVec3, IVec3)> {
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    let mut any = false;
    for (chunk_key, chunk) in grid.iter_chunks() {
        for idx in 0..CHUNK_VOLUME {
            let (lx, ly, lz) = index_to_xyz(idx);
            let bid = chunk.get_block(lx, ly, lz);
            if bid.is_air() { continue; }
            if !registry.is_solid(bid) { continue; }
            let (wx, wy, wz) = ShipGrid::chunk_to_world(chunk_key, lx, ly, lz);
            min = min.min(IVec3::new(wx, wy, wz));
            max = max.max(IVec3::new(wx, wy, wz));
            any = true;
        }
    }
    if any { Some((min, max)) } else { None }
}

/// 6-axis raycast enclosure test. Counts how many of ±X, ±Y, ±Z rays from
/// `v` hit a solid voxel before exiting the hull AABB + 1-block pad. Returns
/// true iff the count is at least `min_hits`.
fn is_geometrically_enclosed(
    v: IVec3,
    lookup: &impl SolidVoxelLookup,
    hull_min: IVec3,
    hull_max: IVec3,
    max_steps: i32,
    min_hits: u32,
) -> bool {
    const DIRS: [IVec3; 6] = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];
    let mut hits = 0u32;
    for dir in DIRS {
        for step in 1..=max_steps {
            let p = v + dir * step;
            if p.x < hull_min.x - 1
                || p.x > hull_max.x + 1
                || p.y < hull_min.y - 1
                || p.y > hull_max.y + 1
                || p.z < hull_min.z - 1
                || p.z > hull_max.z + 1
            {
                break;
            }
            // Rays only terminate on **hull** blocks (structural walls /
            // floors / ceilings / windows). Functional blocks — reactors,
            // thrusters, seats, cruise drives, battery, etc. — are
            // equipment INSIDE the interior, so rays pass through them as
            // if they were air. This is what makes a voxel standing next
            // to the reactor still count as interior: rays toward the hull
            // walls in every direction reach them, even if a piece of
            // equipment happens to sit between the voxel and a wall.
            if lookup.is_hull(p.x, p.y, p.z) {
                hits += 1;
                break;
            }
            // Non-hull solid (functional block): step through transparently.
        }
    }
    hits >= min_hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{registry::BlockRegistry, BlockId, ShipGrid, build_starter_ship, StarterShipLayout};

    fn registry() -> BlockRegistry {
        BlockRegistry::new()
    }

    #[test]
    fn empty_ship_produces_empty_mask() {
        let grid = ShipGrid::new();
        let mask = compute(&grid, &registry());
        assert!(mask.is_empty());
    }

    #[test]
    fn ship_without_seat_produces_empty_mask() {
        // Hollow box, no seats.
        let mut grid = ShipGrid::new();
        for x in 0..6 {
            for y in 0..6 {
                for z in 0..6 {
                    let on_face = x == 0 || x == 5 || y == 0 || y == 5 || z == 0 || z == 5;
                    if on_face {
                        grid.set_block(x, y, z, BlockId::HULL_STANDARD);
                    }
                }
            }
        }
        let mask = compute(&grid, &registry());
        assert!(mask.is_empty());
    }

    #[test]
    fn hollow_box_with_seat_marks_room_as_interior() {
        let mut grid = ShipGrid::new();
        // 8x8x8 hollow box.
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    let on_face = x == 0 || x == 7 || y == 0 || y == 7 || z == 0 || z == 7;
                    if on_face {
                        grid.set_block(x, y, z, BlockId::HULL_STANDARD);
                    }
                }
            }
        }
        // Seat at (3, 1, 3); flood-fill seed is (3, 2, 3).
        grid.set_block(3, 1, 3, BlockId::COCKPIT);
        let mask = compute(&grid, &registry());
        // Every interior voxel in the 6x6x6 interior cavity should be
        // marked; outside should not.
        assert!(mask.is_interior(Vec3::new(4.0, 4.0, 4.0)));
        assert!(mask.is_interior(Vec3::new(1.1, 1.1, 1.1)));
        // Corner just past the +X face.
        assert!(!mask.is_interior(Vec3::new(8.5, 4.0, 4.0)));
        // Inside a wall block (solid — not interior).
        assert!(!mask.is_interior(Vec3::new(0.5, 4.0, 4.0)));
    }

    #[test]
    fn starter_ship_thrusters_do_not_pull_outside_into_interior() {
        // Use the actual starter ship — this is the exact configuration
        // the user hits in-game.
        let layout = StarterShipLayout::default_starter();
        let grid = build_starter_ship(&layout);
        let mask = compute(&grid, &registry());
        assert!(!mask.is_empty(), "starter ship should have an interior");

        // Ownership-core voxel + 1 (above it) should be interior.
        assert!(
            mask.is_interior(Vec3::new(0.5, 2.5, 0.5)),
            "center of main cabin must be interior"
        );
        // Cockpit headspace.
        // Cockpit is at (0, 1, z_min + 1) = (0, 1, -(length/2) + 1) = (0, 1, -7) for default.
        assert!(
            mask.is_interior(Vec3::new(0.5, 2.5, -6.5)),
            "cockpit headspace must be interior"
        );

        // Space JUST past the +X thruster strut (thrusters are at x_max + 1 = 5).
        // Voxel at (5.5, 2.5, 0.5) is outside the hull next to a thruster;
        // it must NOT be interior.
        assert!(
            !mask.is_interior(Vec3::new(5.5, 2.5, 0.5)),
            "space next to a +X thruster must not be interior"
        );
        // Far outside on the +Z end.
        assert!(
            !mask.is_interior(Vec3::new(0.5, 2.5, 10.0)),
            "far outside the ship must not be interior"
        );
    }

    #[test]
    fn starter_ship_spawn_and_movement_voxels_are_interior() {
        // Player spawns at (cockpit_x + 0.5, cockpit_y + 1.0, cockpit_z + 1.5)
        // which is (0.5, 2.0, -5.5) for the default starter ship. Floor
        // voxel (0, 2, -6). All voxels the player can plausibly step to
        // inside the cabin (±3 blocks around spawn) MUST be interior —
        // otherwise `hull_exit_check` triggers EVA handoff on any move.
        let layout = StarterShipLayout::default_starter();
        let grid = build_starter_ship(&layout);
        let mask = compute(&grid, &registry());

        // Spawn itself.
        assert!(
            mask.is_interior(glam::Vec3::new(0.5, 2.0, -5.5)),
            "spawn position must be interior"
        );

        // Walking around the cockpit — every voxel the player can plausibly
        // stand on inside the main cabin must be interior.
        for dz in -1..=2 {
            for dx in -1..=1 {
                let v = glam::Vec3::new(0.5 + dx as f32, 2.0, -5.5 + dz as f32);
                assert!(
                    mask.is_interior(v),
                    "interior near-spawn voxel ({}, {}, {}) must be interior",
                    v.x,
                    v.y,
                    v.z
                );
            }
        }

        // Post-gravity rest position: 1.2 m capsule falls from spawn (y=2)
        // to stand on the floor block. Center ends at y ≈ 1.9, floor voxel
        // (_, 1, _). This is THE position where `hull_exit_check` actually
        // queries at runtime, and it must be interior along the full
        // walkable corridor from cockpit to aft. Tests that functional
        // blocks along the way (REACTOR, BATTERY, CRUISE_DRIVE) don't
        // break the enclosure gate — those blocks sit in the walkway but
        // rays must still pass through them to the hull walls.
        //
        // Walking at y=1.9 means voxel y=1. Solid blocks on that row at
        // x=0 are COCKPIT (z=-5, moved 2 blocks back from the front
        // glass to give the pilot reading distance to the HUD canopy),
        // OWNERSHIP_CORE (z=0), REACTOR_SMALL (z=1), BATTERY (z=2),
        // CRUISE_DRIVE_SMALL (z=3). All other z values in [-8, 7] are
        // air and must be marked interior.
        let solid_z_at_y1_x0: [i32; 5] = [-5, 0, 1, 2, 3];
        for z in -6..=6 {
            if solid_z_at_y1_x0.contains(&z) { continue; }
            let v = glam::Vec3::new(0.5, 1.9, z as f32 + 0.5);
            assert!(
                mask.is_interior(v),
                "walking-path voxel ({:.1}, {:.1}, {:.1}) must be interior",
                v.x,
                v.y,
                v.z
            );
        }

        // The voxel RIGHT NEXT to a functional block must be interior too
        // — even though a ray in that direction is blocked by the
        // functional block at close range, the same ray passes through it
        // to reach the hull (that's the whole point of the hull-only gate).
        // (0, 1, 4) is adjacent to CRUISE_DRIVE_SMALL at (0, 1, 3).
        assert!(
            mask.is_interior(glam::Vec3::new(0.5, 1.9, 4.5)),
            "voxel next to CRUISE_DRIVE must be interior"
        );
        assert!(
            mask.is_interior(glam::Vec3::new(0.5, 1.9, -1.5)),
            "voxel next to OWNERSHIP_CORE (from -Z side) must be interior"
        );
        // Specific regression: runtime eject fired at ship_local (0.60,
        // 1.90, -3.96) → voxel (0, 1, -4). That voxel sits right next to
        // the FLIGHT_COMPUTER at (-2, 1, -4) and the chain of functional
        // blocks (COCKPIT, OWNERSHIP_CORE, REACTOR, BATTERY, CRUISE_DRIVE)
        // along the centerline. Rays in the -X, +Z and -Z directions must
        // pass through those functional blocks to reach the actual hull
        // walls; if they terminate early the voxel fails the gate and the
        // player is ejected as they walk toward the cockpit.
        assert!(
            mask.is_interior(glam::Vec3::new(0.60, 1.90, -3.96)),
            "voxel along centerline near FLIGHT_COMPUTER must be interior"
        );
    }

    #[test]
    fn just_past_open_hatch_is_exterior_on_starter_ship() {
        // Starter ship has a +X hatch at z ∈ [-1, 0], y ∈ [1, 3].
        // Stepping through it from inside lands at (x_max + 1, y, z) =
        // (5, y, z). That voxel must NOT be interior — the player is now
        // outside the ship.
        let layout = StarterShipLayout::default_starter();
        let grid = build_starter_ship(&layout);
        let mask = compute(&grid, &registry());

        // Voxel just inside the hatch (on the +X wall at x_max = 4, y=2,z=0).
        // The hatch cutout is at x=x_max (wall), so the voxel is at x=4.
        assert!(
            mask.is_interior(Vec3::new(4.5, 2.5, -0.5)),
            "voxel at +X hatch opening (inside wall line) must be interior"
        );
        // One step past the hatch (x=5) — outside the ship.
        assert!(
            !mask.is_interior(Vec3::new(5.5, 2.5, -0.5)),
            "voxel just past the +X hatch must not be interior"
        );
    }
}
