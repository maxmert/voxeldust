//! Generic block edit pipeline — source-agnostic block modification queue
//! with a pure apply function.
//!
//! ## Architecture
//!
//! Any system can push edits to `BlockEditQueue`: player input, cellular automata,
//! structural damage, blueprint assembly, Lua scripts, explosions, pistons/rotors.
//!
//! Once per tick, the shard's apply system drains the queue and calls
//! `apply_edits_to_grid()` — a pure function that modifies the grid and returns
//! what changed. The shard then handles its own side effects (physics collider
//! rebuild, hull bounds update, network broadcast, etc.).
//!
//! This separation keeps the core logic reusable across ship-shard, planet-shard,
//! and any future shard type that manages block data.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use glam::IVec3;

use super::block_id::BlockId;
use super::registry::BlockRegistry;
use super::ship_grid::ShipGrid;
use crate::client_message::BlockModData;
use crate::shard_types::SessionToken;

/// Who/what initiated a block edit — for permissions, auditing, rate-limiting.
#[derive(Clone, Debug)]
pub enum EditSource {
    /// Player-initiated edit (break/place via input).
    Player(SessionToken),
    /// Cellular automata (water flow, fire spread, sand falling).
    CellularAutomata,
    /// Structural damage (block destroyed by chain reaction).
    StructuralDamage,
    /// Blueprint assembly (automated ship construction).
    Blueprint,
    /// Lua robot script.
    Script,
    /// Explosion or projectile impact.
    Explosion,
    /// Mechanical system (piston extension, rotor block movement).
    Mechanical,
}

/// A single block modification request (source-agnostic).
#[derive(Clone, Debug)]
pub struct BlockEdit {
    /// World-space block position to modify.
    pub pos: IVec3,
    /// New block type. `BlockId::AIR` = break, anything else = place.
    pub new_block: BlockId,
    /// What initiated this edit.
    pub source: EditSource,
}

/// Queue of pending block edits. Any system can push edits into this queue.
/// Drained once per tick by the shard's apply system.
#[derive(Resource, Default)]
pub struct BlockEditQueue {
    edits: Vec<BlockEdit>,
}

impl BlockEditQueue {
    /// Push a single edit.
    pub fn push(&mut self, edit: BlockEdit) {
        self.edits.push(edit);
    }

    /// Push multiple edits at once (e.g., blueprint assembly).
    pub fn push_batch(&mut self, edits: impl IntoIterator<Item = BlockEdit>) {
        self.edits.extend(edits);
    }

    /// Drain up to `limit` edits, leaving the rest for next tick.
    /// Enables rate-limited processing for large batches (blueprints)
    /// without spiking a single tick's processing budget.
    pub fn drain_up_to(&mut self, limit: usize) -> Vec<BlockEdit> {
        let count = self.edits.len().min(limit);
        self.edits.drain(..count).collect()
    }

    /// Drain all pending edits (for normal per-tick processing).
    pub fn drain_all(&mut self) -> Vec<BlockEdit> {
        std::mem::take(&mut self.edits)
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Number of pending edits.
    pub fn len(&self) -> usize {
        self.edits.len()
    }
}

/// Entity lifecycle operation emitted by the edit pipeline.
/// The shard executes these to spawn/despawn ECS entities for functional blocks.
#[derive(Clone, Debug)]
pub enum EntityOp {
    /// A functional block was placed — spawn an ECS entity for it.
    Spawn {
        pos: IVec3,
        block_id: BlockId,
    },
    /// A functional block was broken — despawn its ECS entity.
    /// `entity_index` comes from `BlockMeta::entity_index` at the old position.
    Despawn {
        pos: IVec3,
        entity_index: u32,
    },
}

/// Result of applying block edits to a grid.
/// Tells the shard what changed so it can handle side effects.
pub struct ApplyResult {
    /// Per-chunk block modifications (batched). Key = chunk IVec3,
    /// value = list of block changes within that chunk.
    /// Used for: collider rebuild, ChunkDelta network broadcast.
    pub dirty_chunks: HashMap<IVec3, Vec<BlockModData>>,
    /// Whether any solid↔air boundary changed (hull shape affected).
    /// If true, the shard should recompute hull bounding box.
    pub solidity_changed: bool,
    /// Whether interactable block types (COCKPIT, etc.) were added or removed.
    /// If true, the shard should rescan interactable blocks.
    pub interactables_changed: bool,
    /// Number of edits that actually changed a block (vs no-ops).
    pub applied_count: usize,
    /// Entity lifecycle operations for functional blocks.
    /// The shard processes these to spawn/despawn ECS entities.
    pub entity_ops: Vec<EntityOp>,
}

/// Apply block edits to a ShipGrid. **Pure function** — no side effects.
///
/// The caller (shard-specific apply system) handles:
/// - Physics collider rebuild for dirty chunks
/// - Hull bounds / sealed rooms recomputation
/// - Network broadcast of ChunkDeltas
/// - Persistence marking
///
/// Edits that don't change anything (same block type) are skipped.
pub fn apply_edits_to_grid(
    grid: &mut ShipGrid,
    edits: &[BlockEdit],
    registry: &BlockRegistry,
) -> ApplyResult {
    let mut dirty_chunks: HashMap<IVec3, Vec<BlockModData>> = HashMap::new();
    let mut solidity_changed = false;
    let mut interactables_changed = false;
    let mut applied_count = 0usize;
    let mut entity_ops = Vec::new();

    for edit in edits {
        let old = grid.get_block(edit.pos.x, edit.pos.y, edit.pos.z);
        if old == edit.new_block {
            continue; // no-op
        }

        // Check for functional block despawn BEFORE overwriting the block
        // (we need the old block's metadata to get the entity_index).
        if registry.is_functional(old) {
            let entity_index = grid
                .get_meta(edit.pos.x, edit.pos.y, edit.pos.z)
                .map(|m| m.entity_index)
                .unwrap_or(0);
            if entity_index != 0 {
                entity_ops.push(EntityOp::Despawn {
                    pos: edit.pos,
                    entity_index,
                });
            }
        }

        grid.set_block(edit.pos.x, edit.pos.y, edit.pos.z, edit.new_block);
        applied_count += 1;

        // Check for functional block spawn AFTER placing the new block.
        if registry.is_functional(edit.new_block) {
            entity_ops.push(EntityOp::Spawn {
                pos: edit.pos,
                block_id: edit.new_block,
            });
        }

        // Track dirty chunk + block modification for this edit.
        let (chunk_key, lx, ly, lz) = ShipGrid::world_to_chunk(
            edit.pos.x, edit.pos.y, edit.pos.z,
        );
        dirty_chunks
            .entry(chunk_key)
            .or_default()
            .push(BlockModData {
                bx: lx,
                by: ly,
                bz: lz,
                block_type: edit.new_block.as_u16(),
            });

        // Track whether hull shape changed (solid ↔ non-solid transition).
        let old_solid = registry.is_solid(old);
        let new_solid = registry.is_solid(edit.new_block);
        if old_solid != new_solid {
            solidity_changed = true;
        }

        // Track whether interactable blocks changed (used by legacy systems).
        if is_interactable(old) || is_interactable(edit.new_block) {
            interactables_changed = true;
        }
    }

    ApplyResult {
        dirty_chunks,
        solidity_changed,
        interactables_changed,
        applied_count,
        entity_ops,
    }
}

/// Check if a block type is an interactable that requires rescanning
/// when placed or removed.
fn is_interactable(id: BlockId) -> bool {
    matches!(
        id,
        BlockId::COCKPIT | BlockId::DOOR | BlockId::OWNERSHIP_CORE
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockRegistry, ShipGrid};

    #[test]
    fn empty_queue() {
        let mut queue = BlockEditQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(queue.drain_all().is_empty());
    }

    #[test]
    fn push_and_drain() {
        let mut queue = BlockEditQueue::default();
        queue.push(BlockEdit {
            pos: IVec3::new(1, 2, 3),
            new_block: BlockId::STONE,
            source: EditSource::Blueprint,
        });
        queue.push(BlockEdit {
            pos: IVec3::new(4, 5, 6),
            new_block: BlockId::AIR,
            source: EditSource::Explosion,
        });
        assert_eq!(queue.len(), 2);

        let drained = queue.drain_all();
        assert_eq!(drained.len(), 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn drain_up_to_rate_limits() {
        let mut queue = BlockEditQueue::default();
        for i in 0..100 {
            queue.push(BlockEdit {
                pos: IVec3::new(i, 0, 0),
                new_block: BlockId::STONE,
                source: EditSource::Blueprint,
            });
        }
        assert_eq!(queue.len(), 100);

        let batch1 = queue.drain_up_to(30);
        assert_eq!(batch1.len(), 30);
        assert_eq!(queue.len(), 70);

        let batch2 = queue.drain_up_to(50);
        assert_eq!(batch2.len(), 50);
        assert_eq!(queue.len(), 20);

        let batch3 = queue.drain_up_to(50); // only 20 left
        assert_eq!(batch3.len(), 20);
        assert!(queue.is_empty());
    }

    #[test]
    fn apply_edits_basic() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();
        grid.set_block(5, 5, 5, BlockId::HULL_STANDARD);

        let edits = vec![BlockEdit {
            pos: IVec3::new(5, 5, 5),
            new_block: BlockId::AIR,
            source: EditSource::Player(SessionToken(0)),
        }];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert_eq!(result.applied_count, 1);
        assert!(result.solidity_changed); // solid → air
        assert_eq!(grid.get_block(5, 5, 5), BlockId::AIR);
        assert_eq!(result.dirty_chunks.len(), 1);
    }

    #[test]
    fn apply_noop_skipped() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();
        grid.set_block(5, 5, 5, BlockId::STONE);

        // Edit to the same block type — should be skipped.
        let edits = vec![BlockEdit {
            pos: IVec3::new(5, 5, 5),
            new_block: BlockId::STONE,
            source: EditSource::Blueprint,
        }];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert_eq!(result.applied_count, 0);
        assert!(result.dirty_chunks.is_empty());
    }

    #[test]
    fn apply_batches_per_chunk() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();

        // Two edits in the same chunk.
        let edits = vec![
            BlockEdit {
                pos: IVec3::new(1, 0, 0),
                new_block: BlockId::STONE,
                source: EditSource::Blueprint,
            },
            BlockEdit {
                pos: IVec3::new(2, 0, 0),
                new_block: BlockId::DIRT,
                source: EditSource::Blueprint,
            },
        ];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert_eq!(result.applied_count, 2);
        // Both edits in the same chunk → one dirty chunk entry with 2 mods.
        assert_eq!(result.dirty_chunks.len(), 1);
        let mods = result.dirty_chunks.values().next().unwrap();
        assert_eq!(mods.len(), 2);
    }

    #[test]
    fn apply_tracks_interactable_changes() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();

        let edits = vec![BlockEdit {
            pos: IVec3::new(0, 0, 0),
            new_block: BlockId::COCKPIT,
            source: EditSource::Player(SessionToken(0)),
        }];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert!(result.interactables_changed);
    }

    #[test]
    fn apply_emits_spawn_for_functional_block() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();

        let edits = vec![BlockEdit {
            pos: IVec3::new(5, 5, 5),
            new_block: BlockId::THRUSTER_SMALL_CHEMICAL,
            source: EditSource::Player(SessionToken(0)),
        }];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert_eq!(result.entity_ops.len(), 1);
        match &result.entity_ops[0] {
            EntityOp::Spawn { pos, block_id } => {
                assert_eq!(*pos, IVec3::new(5, 5, 5));
                assert_eq!(*block_id, BlockId::THRUSTER_SMALL_CHEMICAL);
            }
            _ => panic!("expected Spawn"),
        }
    }

    #[test]
    fn apply_emits_despawn_for_removed_functional_block() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();

        // Place a functional block with a fake entity_index in metadata.
        grid.set_block(5, 5, 5, BlockId::REACTOR_SMALL);
        let meta = grid.get_or_create_meta(5, 5, 5);
        meta.entity_index = 42;

        // Now break it.
        let edits = vec![BlockEdit {
            pos: IVec3::new(5, 5, 5),
            new_block: BlockId::AIR,
            source: EditSource::Explosion,
        }];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert_eq!(result.entity_ops.len(), 1);
        match &result.entity_ops[0] {
            EntityOp::Despawn { pos, entity_index } => {
                assert_eq!(*pos, IVec3::new(5, 5, 5));
                assert_eq!(*entity_index, 42);
            }
            _ => panic!("expected Despawn"),
        }
    }

    #[test]
    fn apply_no_entity_ops_for_structural_blocks() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();

        let edits = vec![BlockEdit {
            pos: IVec3::new(0, 0, 0),
            new_block: BlockId::HULL_STANDARD,
            source: EditSource::Blueprint,
        }];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert!(result.entity_ops.is_empty());
    }

    #[test]
    fn apply_replace_functional_emits_despawn_then_spawn() {
        let registry = BlockRegistry::new();
        let mut grid = ShipGrid::new();

        // Place one functional block.
        grid.set_block(3, 3, 3, BlockId::REACTOR_SMALL);
        let meta = grid.get_or_create_meta(3, 3, 3);
        meta.entity_index = 99;

        // Replace with a different functional block.
        let edits = vec![BlockEdit {
            pos: IVec3::new(3, 3, 3),
            new_block: BlockId::BATTERY,
            source: EditSource::Player(SessionToken(0)),
        }];

        let result = apply_edits_to_grid(&mut grid, &edits, &registry);
        assert_eq!(result.entity_ops.len(), 2);
        assert!(matches!(result.entity_ops[0], EntityOp::Despawn { entity_index: 99, .. }));
        assert!(matches!(result.entity_ops[1], EntityOp::Spawn { block_id, .. } if block_id == BlockId::BATTERY));
    }
}
