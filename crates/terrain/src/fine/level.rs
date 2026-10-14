//! ★ THE FINE LEVELS (ruling B95; ruling B81: ONE generic tile machinery for every level): what makes a level a level —
//! its rung, how many nodes its tile spans, how far its raw raster reaches past the tile, and how many mean levels a
//! finished tile carries — in ONE place, so the record, the finish, the read and the needs serve every level by the same
//! code.
//!
//! - **L1**, the 128 m level (the rung-7 cell on the home planet): a tile is 8 × 8 macro nodes; its raw raster reaches one
//!   node past it (half its two-node buffer); a finished tile carries five mean levels, so it serves rungs 7–12.
//! - **L2**, the 16 m level (the rung-4 cell): a tile is one macro node; its raw raster reaches an eighth of a node past it
//!   (half its quarter-node buffer — 1 km on the home planet); two mean levels, so it serves rungs 4–6, and rungs 0–3 read
//!   its raster by the curve.
//!
//! Both rasters reach 64 cells past their tiles on the home planet, so both finish with the same band (64 less the blend).
//!
//! **Example.** The home planet's 16 m tile under hull 17 is the node (148, 36) of face 3: 512 × 512 cells of 16 m, its raw
//! raster 640 × 640, its finished levels 512², 256² and 128² — read by the chunks of rungs 4, 5 and 6 under the hull.

use super::l1::cells_per_node_shift;
use super::stitch::BLEND;
use crate::body::BodyDefinition;
use crate::macro_lattice::MacroLattice;

/// A fine level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FineLevel {
    /// The 128 m level.
    L1,
    /// The 16 m level.
    L2,
}

impl FineLevel {
    /// Every level, the coarsest first.
    pub const ALL: [FineLevel; 2] = [FineLevel::L1, FineLevel::L2];

    /// The level's number on the wire and in the store (the `level` byte of a fine record).
    #[must_use]
    pub const fn id(self) -> u8 {
        match self {
            FineLevel::L1 => 1,
            FineLevel::L2 => 2,
        }
    }

    /// The level of a record's `level` byte; `None` for a byte no level holds.
    #[must_use]
    pub const fn from_id(id: u8) -> Option<FineLevel> {
        match id {
            1 => Some(FineLevel::L1),
            2 => Some(FineLevel::L2),
            _ => None,
        }
    }

    /// The level's rung: its raster's cell.
    #[must_use]
    pub const fn rung(self) -> u8 {
        match self {
            FineLevel::L1 => 7,
            FineLevel::L2 => 4,
        }
    }

    /// A tile's side in macro nodes, as a shift.
    #[must_use]
    pub const fn tile_nodes_shift(self) -> u32 {
        match self {
            FineLevel::L1 => 3,
            FineLevel::L2 => 0,
        }
    }

    /// The mean levels a finished tile carries over its raster: the rungs over the level's own it serves.
    #[must_use]
    pub const fn mean_levels(self) -> u32 {
        match self {
            FineLevel::L1 => 5,
            FineLevel::L2 => 2,
        }
    }

    /// The coarsest rung the level serves.
    #[must_use]
    pub const fn top_rung(self) -> u8 {
        self.rung() + self.mean_levels() as u8
    }

    /// How far a raw raster reaches past its tile, as a shift under the node: half the level's buffer.
    #[must_use]
    pub const fn margin_shift(self) -> u32 {
        match self {
            FineLevel::L1 => 0,
            FineLevel::L2 => 3,
        }
    }
}

/// ★ A LEVEL'S NUMBERS ON ONE BODY: its cells to a node as a shift, a tile's side in cells as a shift, how far a raw
/// raster reaches past its tile (cells), and the level's cells along a face's edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelGeom {
    pub level: FineLevel,
    pub shift: u32,
    pub ts: u32,
    pub margin: i64,
    pub n: i32,
}

impl LevelGeom {
    /// The level's numbers on `body` over `lattice`; `None` where the level's cells do not nest in the nodes by a power
    /// of two, or so coarsely that the margin is under a cell.
    #[must_use]
    pub fn of(
        level: FineLevel,
        body: &BodyDefinition,
        lattice: &MacroLattice,
    ) -> Option<LevelGeom> {
        LevelGeom::from_shift(
            level,
            cells_per_node_shift(body, lattice, level.rung())?,
            lattice.edge,
        )
    }

    /// The level's numbers from its cells to a node as a shift and the lattice's edge.
    #[must_use]
    pub fn from_shift(level: FineLevel, shift: u32, edge: u32) -> Option<LevelGeom> {
        let margin = (1i64 << shift) >> level.margin_shift();
        (margin > 0).then_some(LevelGeom {
            level,
            shift,
            ts: shift + level.tile_nodes_shift(),
            margin,
            n: i32::try_from(u64::from(edge) << shift).ok()?,
        })
    }

    /// The level's numbers from a record's side (a tile's side in cells as a shift) and the lattice's edge — what a host
    /// that holds no body reads.
    #[must_use]
    pub fn from_side(level: FineLevel, ts: u32, edge: u32) -> Option<LevelGeom> {
        LevelGeom::from_shift(level, ts.checked_sub(level.tile_nodes_shift())?, edge)
    }

    /// A tile's side in cells.
    #[must_use]
    pub const fn side(&self) -> i64 {
        1i64 << self.ts
    }

    /// The cut's band: the margin less the blend, so no blend reads past a raw raster. A level whose band is not over
    /// zero reads tiles but cannot finish them ([`super::finish::finished_square`] refuses it).
    #[must_use]
    pub const fn band(&self) -> i64 {
        self.margin - BLEND
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::home_planet;

    /// Each level states its numbers, a record's byte names it back, and the home planet nests both: the 128 m tile 8 × 8
    /// nodes of 64 cells, the 16 m tile one node of 512 cells, each raster 64 cells past its tile, each band 56.
    #[test]
    fn each_level_states_its_numbers_and_the_home_planet_nests_both() {
        assert_eq!(FineLevel::ALL.map(FineLevel::id), [1, 2]);
        assert_eq!(
            FineLevel::ALL.map(|l| FineLevel::from_id(l.id())),
            [Some(FineLevel::L1), Some(FineLevel::L2)]
        );
        assert_eq!((FineLevel::from_id(0), FineLevel::from_id(3)), (None, None));
        assert_eq!(FineLevel::ALL.map(FineLevel::top_rung), [12, 6]);
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let l1 = LevelGeom::of(FineLevel::L1, &body, &lattice).expect("the 128 m level");
        let l2 = LevelGeom::of(FineLevel::L2, &body, &lattice).expect("the 16 m level");
        assert_eq!(
            (l1.shift, l1.ts, l1.margin, l1.side(), l1.band()),
            (6, 9, 64, 512, 56)
        );
        assert_eq!(
            (l2.shift, l2.ts, l2.margin, l2.side(), l2.band()),
            (9, 9, 64, 512, 56)
        );
        assert_eq!(l2.n, l1.n << 3);
        // A host with no body reads the same numbers from a record's side and the head's edge.
        assert_eq!(
            LevelGeom::from_side(FineLevel::L2, l2.ts, lattice.edge),
            Some(l2)
        );
        // No level where the cells do not nest, where a margin is under a cell, or where a face's cells pass a word.
        let one =
            MacroLattice::with_edge(&body, body.ladder().n).expect("a lattice of one-cell nodes");
        assert_eq!(LevelGeom::of(FineLevel::L1, &body, &one), None);
        assert_eq!(LevelGeom::from_shift(FineLevel::L2, 2, 3), None);
        assert_eq!(LevelGeom::from_shift(FineLevel::L1, 31, 256), None);
        assert_eq!(LevelGeom::from_side(FineLevel::L1, 2, 3), None);
    }
}
