//! ★ THE CELL ADDRESS — the name of one cell of one realm's grid (ruling V6, Format A).
//!
//! `(body, face, rung, i, j, k)`. The body is the realm whose grid it is. The face is one of six on a
//! cube-sphere and always `PosX` on a flat grid. The rung is the detail level: a cell is `2^rung`
//! metres, and rung 0 is the only rung anything is ever written at. `i`, `j`, `k` are SIGNED whole
//! numbers: on a flat grid the origin is the realm's own frame origin, so a thruster one metre aft of
//! it sits at `k = −1`; on a cube-sphere they never go negative, and the sign is simply unused.
//!
//! The address holds no parent frame. A cell never says where its realm is (SL1).
//!
//! **The chunk is a packing, not part of the name.** Cells are stored and streamed in chunks of
//! `62 × 62 × 62`. A chunk's coordinate and a cell's index inside it are DERIVED from the address by
//! one frozen rule ([`CellAddr::chunk`], [`CellAddr::cell_index`]) and never stored in it.

use crate::pose::RealmId;

use super::bend::Face;

/// The detail rung of an address: a cell is `2^rung` metres. Four bits: `0..=15`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rung(u8);

impl Rung {
    /// The finest rung: one-metre cells, the only writable rung.
    pub const ZERO: Rung = Rung(0);
    /// The coarsest rung the address can name.
    pub const MAX: u8 = 15;

    /// A rung from its number; `None` above [`Rung::MAX`] (a decoder REFUSES, never defaults).
    #[must_use]
    pub const fn new(level: u8) -> Option<Rung> {
        if level <= Rung::MAX {
            Some(Rung(level))
        } else {
            None
        }
    }

    /// The rung's number.
    #[must_use]
    pub const fn level(self) -> u8 {
        self.0
    }

    /// Metres per cell at this rung, as a power of two.
    #[must_use]
    pub const fn cell_m(self) -> u32 {
        1 << self.0
    }
}

/// The name of one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellAddr {
    /// The realm whose grid this is.
    pub body: RealmId,
    /// The cube face; always [`Face::PosX`] on a flat grid.
    pub face: Face,
    /// The detail rung.
    pub rung: Rung,
    /// Along the face's `u` axis (signed).
    pub i: i32,
    /// Along the face's `v` axis (signed).
    pub j: i32,
    /// Along the radial, in cells from the body's floor; along the third axis on a flat grid (signed).
    pub k: i32,
}

/// Cells per chunk edge. A PACKING constant, not an address field.
pub const CHUNK_EDGE: i32 = 62;

/// The coordinate of a chunk: the address's `(i, j, k)` each divided by the chunk edge, rounding
/// toward negative infinity, so a flat grid's negative half packs without a special case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkCoord {
    pub body: RealmId,
    pub face: Face,
    pub rung: Rung,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// A cell's index inside its chunk: `(c·62 + b)·62 + a`, with `a` (the `i` remainder) fastest.
/// Eighteen bits hold any chunk edge up to 64. FROZEN with the record (ruling V6, Format A).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellIndex(u32);

impl CellIndex {
    /// The largest index a 62-edge chunk holds.
    pub const MAX: u32 = (CHUNK_EDGE as u32) * (CHUNK_EDGE as u32) * (CHUNK_EDGE as u32) - 1;

    /// The packed index; `None` above [`CellIndex::MAX`] (a decoder REFUSES, never defaults).
    #[must_use]
    pub const fn new(index: u32) -> Option<CellIndex> {
        if index <= CellIndex::MAX {
            Some(CellIndex(index))
        } else {
            None
        }
    }

    /// The packed value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The `(a, b, c)` remainders, `a` fastest.
    #[must_use]
    pub const fn unpack(self) -> (i32, i32, i32) {
        let e = CHUNK_EDGE as u32;
        let a = self.0 % e;
        let b = (self.0 / e) % e;
        let c = self.0 / (e * e);
        (a as i32, b as i32, c as i32)
    }
}

impl CellAddr {
    /// The chunk this cell packs into.
    #[must_use]
    pub const fn chunk(self) -> ChunkCoord {
        ChunkCoord {
            body: self.body,
            face: self.face,
            rung: self.rung,
            x: self.i.div_euclid(CHUNK_EDGE),
            y: self.j.div_euclid(CHUNK_EDGE),
            z: self.k.div_euclid(CHUNK_EDGE),
        }
    }

    /// This cell's index inside its chunk.
    #[must_use]
    pub const fn cell_index(self) -> CellIndex {
        let a = self.i.rem_euclid(CHUNK_EDGE) as u32;
        let b = self.j.rem_euclid(CHUNK_EDGE) as u32;
        let c = self.k.rem_euclid(CHUNK_EDGE) as u32;
        let e = CHUNK_EDGE as u32;
        CellIndex((c * e + b) * e + a)
    }

    /// The address of the cell at `index` inside `chunk` — the inverse of the two above.
    #[must_use]
    pub const fn from_chunk(chunk: ChunkCoord, index: CellIndex) -> CellAddr {
        let (a, b, c) = index.unpack();
        CellAddr {
            body: chunk.body,
            face: chunk.face,
            rung: chunk.rung,
            i: chunk.x * CHUNK_EDGE + a,
            j: chunk.y * CHUNK_EDGE + b,
            k: chunk.z * CHUNK_EDGE + c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: RealmId = RealmId::Planet(7);

    #[test]
    fn a_rung_is_four_bits_and_a_cell_index_is_bounded() {
        assert_eq!(Rung::new(0), Some(Rung::ZERO));
        assert_eq!(Rung::new(15).map(Rung::level), Some(15));
        assert_eq!(Rung::new(16), None, "a rung above fifteen is refused");
        assert_eq!(Rung::new(3).map(Rung::cell_m), Some(8));
        assert_eq!(
            CellIndex::new(CellIndex::MAX).map(CellIndex::get),
            Some(238_327)
        );
        assert_eq!(
            CellIndex::new(CellIndex::MAX + 1),
            None,
            "an index past the chunk is refused"
        );
    }

    #[test]
    fn the_chunk_packing_round_trips_on_both_sides_of_the_origin() {
        // Every cell of a 3-chunk cube around the origin, at rung 0 and rung 5.
        for level in [0u8, 5] {
            let rung = Rung::new(level).expect("rung");
            let mut i = -70;
            while i <= 70 {
                let mut j = -70;
                while j <= 70 {
                    let mut k = -70;
                    while k <= 70 {
                        let addr = CellAddr {
                            body: BODY,
                            face: Face::NegY,
                            rung,
                            i,
                            j,
                            k,
                        };
                        let back = CellAddr::from_chunk(addr.chunk(), addr.cell_index());
                        assert_eq!(back, addr, "the packing is lossless at ({i}, {j}, {k})");
                        assert!(addr.cell_index().get() <= CellIndex::MAX);
                        k += 7;
                    }
                    j += 7;
                }
                i += 7;
            }
        }
    }

    #[test]
    fn the_linearisation_is_frozen_with_i_fastest() {
        let addr = CellAddr {
            body: BODY,
            face: Face::PosX,
            rung: Rung::ZERO,
            i: 1,
            j: 2,
            k: 3,
        };
        assert_eq!(addr.cell_index().get(), (3 * 62 + 2) * 62 + 1);
        assert_eq!(addr.cell_index().unpack(), (1, 2, 3));
        let stern = CellAddr {
            body: BODY,
            face: Face::PosX,
            rung: Rung::ZERO,
            i: 0,
            j: 0,
            k: -1,
        };
        assert_eq!(
            stern.chunk().z,
            -1,
            "one metre aft packs into the chunk below the origin"
        );
        assert_eq!(
            stern.cell_index().unpack(),
            (0, 0, 61),
            "and sits at that chunk's top cell"
        );
    }
}
