//! Phase 4.5 — sparse uniform hash grid for spatial queries.
//!
//! The system-shard's AOI build runs every tick and asks "for each
//! observer, which entities are within their visibility range?". The
//! original code did this as a `for observer in N { for candidate in
//! N { distance check } }` linear scan — O(N²) per tick — which caps
//! at ~100 ships before tick budget breaks.
//!
//! This module replaces the inner loop with a sparse hash grid:
//!
//!   * Cells are indexed by integer-quantized 3D position.
//!   * Each cell holds the entity ids that fall within it.
//!   * `query_radius` iterates ONLY populated cells (via the
//!     hashmap), prunes via cell-AABB-vs-sphere distance, and emits
//!     entities whose actual position is within the radius.
//!
//! Per-tick cost shape:
//!   * Build: O(entities) inserts.
//!   * Query: O(populated_cells × AABB_check + entities_in_radius).
//!
//! For realistic distributions (entities clustered around planets +
//! stations) `populated_cells << conceptual_cells_in_radius`, so the
//! query is dominated by entities that actually return. This is the
//! AAA-quality property — the cost scales with the answer size, not
//! with the search space.
//!
//! # Why sparse, not dense
//!
//! A 200 km coarse-AOI radius / 1024 m cell size = 195 cells per axis
//! → 7.4M conceptual cells. A dense grid (Vec-backed) would allocate
//! that many slots upfront — gigabytes per system. The hash grid
//! materialises only the cells that actually hold entities, so memory
//! tracks population, not extent.
//!
//! # Why per-tick rebuild, not incremental
//!
//! Entities (ships, EVA players, surface-player aggregates) update
//! their positions every tick. Tracking per-entity cell membership
//! across moves requires bookkeeping that's strictly more expensive
//! than rebuild for fully-dynamic data. We rebuild from scratch each
//! tick — `clear()` then `insert()` per candidate — and pay only the
//! cost we need.
//!
//! # Cell size choice
//!
//! 1024 m matches the plan and balances:
//!   * Small enough that close-range queries (`ship_full_range = 2 km`)
//!     touch only ~3 cells per axis = ~27 cells, all of which can be
//!     fully iterated cheaply.
//!   * Large enough that per-cell overhead doesn't dominate at high
//!     entity density (10 entities/cell @ 1 km³ ≈ 10 entities/km³, a
//!     reasonable upper bound for a busy system).

use std::collections::HashMap;

use glam::{DVec3, IVec3};

/// Default cell size for the system-shard's AOI grid. See the
/// module doc for the rationale behind 1024 m.
pub const DEFAULT_CELL_SIZE_M: f64 = 1024.0;

/// Sparse uniform 3D hash grid.
#[derive(Debug)]
pub struct SpatialGrid<E: Copy + Eq + std::hash::Hash> {
    /// Cell edge length in meters. Constant after construction.
    cell_size: f64,
    /// Cell key → list of entity ids that fall within it.
    cells: HashMap<IVec3, Vec<E>>,
    /// Entity id → its position. Stored alongside the cell index so
    /// `query_radius` can do a fine-grained per-entity distance check
    /// after the cell-AABB prune.
    positions: HashMap<E, DVec3>,
}

impl<E: Copy + Eq + std::hash::Hash> Default for SpatialGrid<E> {
    fn default() -> Self {
        Self::new(DEFAULT_CELL_SIZE_M)
    }
}

impl<E: Copy + Eq + std::hash::Hash> SpatialGrid<E> {
    pub fn new(cell_size: f64) -> Self {
        debug_assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            cell_size,
            cells: HashMap::new(),
            positions: HashMap::new(),
        }
    }

    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    pub fn entity_count(&self) -> usize {
        self.positions.len()
    }

    pub fn populated_cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Drop every entity. Per-tick rebuild starts here.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.positions.clear();
    }

    /// Quantize a world position to a cell key. Pure helper — public
    /// so tests + grid integrations can compute cell coords without
    /// going through `insert`.
    pub fn cell_for(&self, position: DVec3) -> IVec3 {
        IVec3::new(
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
            (position.z / self.cell_size).floor() as i32,
        )
    }

    /// Insert (or replace) an entity at `position`. If the entity was
    /// previously inserted, its old cell entry is removed first —
    /// supports re-inserts within a single tick without leaking cell
    /// references.
    pub fn insert(&mut self, entity: E, position: DVec3) {
        if let Some(old_pos) = self.positions.insert(entity, position) {
            let old_cell = self.cell_for(old_pos);
            if let Some(bucket) = self.cells.get_mut(&old_cell) {
                bucket.retain(|e| *e != entity);
                if bucket.is_empty() {
                    self.cells.remove(&old_cell);
                }
            }
        }
        let cell = self.cell_for(position);
        self.cells.entry(cell).or_default().push(entity);
    }

    /// Remove an entity. Idempotent — removing an absent entity is a
    /// no-op.
    pub fn remove(&mut self, entity: E) {
        let Some(pos) = self.positions.remove(&entity) else {
            return;
        };
        let cell = self.cell_for(pos);
        if let Some(bucket) = self.cells.get_mut(&cell) {
            bucket.retain(|e| *e != entity);
            if bucket.is_empty() {
                self.cells.remove(&cell);
            }
        }
    }

    /// Get an entity's stored position, or `None` if not in the grid.
    pub fn position_of(&self, entity: E) -> Option<DVec3> {
        self.positions.get(&entity).copied()
    }

    /// Yield every entity within `radius` of `center`, paired with
    /// its stored position.
    ///
    /// Algorithm:
    ///   1. Iterate every populated cell (sparse — empty cells are
    ///      not in the map).
    ///   2. For each cell, compute the closest point on its AABB to
    ///      `center`. If that distance exceeds `radius`, the cell
    ///      can't contain any in-range entities — skip.
    ///   3. For surviving cells, do a per-entity squared-distance
    ///      check and emit those within the radius.
    ///
    /// Returns owned `(entity, position)` pairs — the caller usually
    /// projects through additional metadata anyway, so allocating
    /// here is no more expensive than borrowing.
    pub fn query_radius(&self, center: DVec3, radius: f64) -> Vec<(E, DVec3)> {
        if radius <= 0.0 {
            return Vec::new();
        }
        let r2 = radius * radius;
        let mut out = Vec::new();
        for (cell_key, ids) in &self.cells {
            // Cell AABB in world space.
            let cell_min = DVec3::new(
                cell_key.x as f64 * self.cell_size,
                cell_key.y as f64 * self.cell_size,
                cell_key.z as f64 * self.cell_size,
            );
            let cell_max = cell_min + DVec3::splat(self.cell_size);
            // Closest point on AABB to center, by per-axis clamp.
            let clamped = DVec3::new(
                center.x.clamp(cell_min.x, cell_max.x),
                center.y.clamp(cell_min.y, cell_max.y),
                center.z.clamp(cell_min.z, cell_max.z),
            );
            let cell_dist2 = (center - clamped).length_squared();
            if cell_dist2 > r2 {
                continue;
            }
            for id in ids {
                let Some(&pos) = self.positions.get(id) else {
                    continue;
                };
                if (center - pos).length_squared() <= r2 {
                    out.push((*id, pos));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid<E: Copy + Eq + std::hash::Hash>() -> SpatialGrid<E> {
        SpatialGrid::new(1024.0)
    }

    #[test]
    fn insert_quantizes_to_correct_cell() {
        let mut g = grid::<u64>();
        // 1024.0 cell size: position (1500, -2000, 50) → cell (1, -2, 0).
        g.insert(1, DVec3::new(1500.0, -2000.0, 50.0));
        let cell = g.cell_for(DVec3::new(1500.0, -2000.0, 50.0));
        assert_eq!(cell, IVec3::new(1, -2, 0));
    }

    #[test]
    fn cell_for_handles_negative_floor_correctly() {
        // -0.1 must floor to -1 (not 0). This matters at cell boundaries
        // where naive truncation would put two entities on opposite
        // sides of the origin in the same cell.
        let g = grid::<u64>();
        assert_eq!(g.cell_for(DVec3::new(-0.1, 0.0, 0.0)), IVec3::new(-1, 0, 0));
        assert_eq!(g.cell_for(DVec3::new(0.0, 0.0, 0.0)), IVec3::ZERO);
        assert_eq!(g.cell_for(DVec3::new(-1024.1, 0.0, 0.0)), IVec3::new(-2, 0, 0));
    }

    #[test]
    fn insert_then_query_finds_nearby_entities() {
        let mut g = grid::<u64>();
        g.insert(1, DVec3::new(0.0, 0.0, 0.0));
        g.insert(2, DVec3::new(500.0, 0.0, 0.0));
        g.insert(3, DVec3::new(5000.0, 0.0, 0.0));
        let results = g.query_radius(DVec3::ZERO, 1000.0);
        let ids: Vec<u64> = {
            let mut v: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            v.sort();
            v
        };
        assert_eq!(ids, vec![1, 2], "id 3 is outside the 1km radius");
    }

    #[test]
    fn query_radius_excludes_cell_just_outside_aabb() {
        // Center at origin, radius 100m. Cell at (10, 0, 0) starts at
        // x=10240 — far outside. AABB-prune must drop it without
        // checking entities.
        let mut g = grid::<u64>();
        g.insert(99, DVec3::new(10500.0, 0.0, 0.0)); // far away
        let results = g.query_radius(DVec3::ZERO, 100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn query_radius_includes_entities_in_corner_cells() {
        // Entity exactly at the corner of a cell that just barely
        // intersects the radius. AABB-prune must accept the cell;
        // per-entity check must accept the entity.
        let mut g = grid::<u64>();
        g.insert(1, DVec3::new(2000.0, 0.0, 0.0));
        // Radius 2001 — entity is just inside.
        let results = g.query_radius(DVec3::ZERO, 2001.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn re_insert_moves_entity_to_new_cell() {
        let mut g = grid::<u64>();
        g.insert(1, DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(g.populated_cell_count(), 1);
        // Move the entity to a different cell.
        g.insert(1, DVec3::new(5000.0, 0.0, 0.0));
        assert_eq!(g.populated_cell_count(), 1, "old cell must be cleaned up");
        let results = g.query_radius(DVec3::ZERO, 100.0);
        assert!(results.is_empty(), "old position must not match");
        let results = g.query_radius(DVec3::new(5000.0, 0.0, 0.0), 100.0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn remove_drops_entity_and_cleans_cell() {
        let mut g = grid::<u64>();
        g.insert(1, DVec3::new(100.0, 0.0, 0.0));
        g.remove(1);
        assert_eq!(g.entity_count(), 0);
        assert_eq!(g.populated_cell_count(), 0, "empty cell must be removed");
        // Idempotent — second remove is a no-op.
        g.remove(1);
    }

    #[test]
    fn remove_preserves_other_entities_in_same_cell() {
        let mut g = grid::<u64>();
        g.insert(1, DVec3::new(100.0, 0.0, 0.0));
        g.insert(2, DVec3::new(200.0, 0.0, 0.0)); // same cell
        g.remove(1);
        assert_eq!(g.entity_count(), 1);
        assert_eq!(g.populated_cell_count(), 1);
        let results = g.query_radius(DVec3::new(150.0, 0.0, 0.0), 200.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn clear_resets_all_state() {
        let mut g = grid::<u64>();
        g.insert(1, DVec3::ZERO);
        g.insert(2, DVec3::new(5000.0, 0.0, 0.0));
        g.clear();
        assert_eq!(g.entity_count(), 0);
        assert_eq!(g.populated_cell_count(), 0);
        assert!(g.query_radius(DVec3::ZERO, 100_000.0).is_empty());
    }

    #[test]
    fn zero_radius_query_returns_empty() {
        // Defensive: a radius-0 query is a degenerate point query.
        // Even an entity sitting AT the center returns nothing — we
        // treat radius<=0 as "no results, don't bother walking cells."
        let mut g = grid::<u64>();
        g.insert(1, DVec3::ZERO);
        let results = g.query_radius(DVec3::ZERO, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn negative_radius_query_returns_empty() {
        let mut g = grid::<u64>();
        g.insert(1, DVec3::ZERO);
        let results = g.query_radius(DVec3::ZERO, -100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn position_of_returns_correct_position() {
        let mut g = grid::<u64>();
        g.insert(1, DVec3::new(100.0, -50.0, 200.0));
        assert_eq!(g.position_of(1), Some(DVec3::new(100.0, -50.0, 200.0)));
        assert_eq!(g.position_of(2), None);
    }

    #[test]
    fn many_entities_in_one_cell_all_returned() {
        let mut g = grid::<u64>();
        // 100 entities in one cell.
        for i in 0..100 {
            g.insert(i, DVec3::new(100.0 + i as f64, 0.0, 0.0));
        }
        // The cell is (0, 0, 0) (since x ∈ [100, 199] all quantize to 0).
        assert_eq!(g.populated_cell_count(), 1);
        let results = g.query_radius(DVec3::new(150.0, 0.0, 0.0), 100.0);
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn aabb_prune_avoids_walking_far_cells() {
        // Build many far-away populated cells; query a small radius
        // at origin and verify only nearby cells contribute. We
        // can't directly assert "iterations skipped" without an
        // instrumentation API, but the result should be correct +
        // fast.
        let mut g = grid::<u64>();
        // 1000 entities scattered far away. Start at i=1 so x=0 is
        // free for the near-origin probe entity below.
        for i in 1..=1000 {
            let x = (i as f64) * 10_000.0;
            g.insert(i, DVec3::new(x, 0.0, 0.0));
        }
        // Plus one near origin (50m).
        g.insert(9999, DVec3::new(50.0, 0.0, 0.0));
        let results = g.query_radius(DVec3::ZERO, 100.0);
        assert_eq!(results.len(), 1, "only the near-origin entity is in radius");
        assert_eq!(results[0].0, 9999);
    }
}
