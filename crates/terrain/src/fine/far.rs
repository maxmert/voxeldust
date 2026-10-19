//! ★ THE FAR LAKES FROM THE 128 m TILES (ruling B116 part 2): what an 8 km node's lake water is on the finished 128 m
//! ground, folded once by the planet's shard and read by the client from rung 13 up, so a lake keeps its level and its
//! outline at every distance (W16).
//!
//! Each 8 km node under or beside a lake gives one row ([`FarNode`]): its WET SHARE (how much of its 128 m ground stands
//! under water), its WATER LEVEL (the highest level its wet cells hold) and its FLOOR (the mean ground of its wet cells).
//! The client folds each pyramid level from the rows by the majority class, as the fine tiles' coarse copies stand
//! ([`super::tile::class_mean_levels`]): a block at least half wet stands at its wet cells' mean floor and holds the
//! water; a block less than half wet stands at its dry ground's mean, read from the pyramid's own block mean less the wet
//! cells' share ([`fold_level`]). Heights in whole metres from the ladder radius, the artifact's own unit.
//!
//! **Example.** The basin lake's node at the lake's middle: share 255, level −2 721 m over the sea, floor −3 050 m. A
//! 16 km block of it and three land nodes holds 1.2 nodes of water: less than half, so it stands at its land's mean and
//! draws dry; a block of three such lake nodes and one land node stands at −3 050 m and draws the lake.

use super::l1::TileKey;
use super::level::LevelGeom;
use super::stitch::MISSING;
use super::tile::{WET_FULL, WaterLevels};
use crate::artifact::DRY_M;
use crate::macro_lattice::MacroLattice;
use std::collections::BTreeMap;

/// One 8 km node's lake water on the 128 m ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarNode {
    pub node: u32,
    /// Its 128 m cells under water, in 255ths of the node.
    pub share: u8,
    /// The highest level its wet cells hold, or [`DRY_M`].
    pub level_m: i16,
    /// The mean ground of its wet cells, or [`DRY_M`] where none is wet.
    pub floor_m: i16,
}

/// Quarter metres over the sea as whole metres from the ladder radius, rounded half up.
fn metres(quarters: i64, sea_m: i16) -> i16 {
    (((quarters + 2) >> 2) + i64::from(sea_m)) as i16
}

/// ★ THE FAR ROWS OF ONE FINISHED 128 m TILE: each of its nodes `keep` names, from its finest level's cells.
#[must_use]
#[allow(
    clippy::integer_division,
    reason = "the fold runs on the planet's shard's CPU alone (the client reads the list): an exact integer quotient, one \
              answer on every host"
)]
pub fn tile_far_nodes(
    key: TileKey,
    levels: &WaterLevels,
    geom: LevelGeom,
    lattice: &MacroLattice,
    sea_m: i16,
    keep: impl Fn(u32) -> bool,
) -> Vec<FarNode> {
    let side = 1usize << geom.ts;
    let per = 1usize << geom.shift;
    let nodes = side >> geom.shift;
    let (heights, level, wet) = (&levels.heights[0], &levels.level[0], &levels.wet[0]);
    let mut out = Vec::new();
    for b in 0..nodes {
        for a in 0..nodes {
            let node = lattice.index(
                key.face,
                (key.tx << (geom.ts - geom.shift)) + a as i32,
                (key.ty << (geom.ts - geom.shift)) + b as i32,
            );
            if !keep(node) {
                continue;
            }
            let (mut full, mut floor, mut top) = (0i64, 0i64, MISSING);
            for y in 0..per {
                for x in 0..per {
                    let c = (b * per + y) * side + a * per + x;
                    if wet[c] == WET_FULL && level[c] != MISSING && heights[c] != MISSING {
                        full += 1;
                        floor += i64::from(heights[c]);
                        top = top.max(level[c]);
                    }
                }
            }
            let cells = (per * per) as i64;
            out.push(FarNode {
                node,
                share: ((full * 255 + cells / 2) / cells) as u8,
                level_m: if full == 0 {
                    DRY_M
                } else {
                    metres(i64::from(top), sea_m)
                },
                floor_m: if full == 0 {
                    DRY_M
                } else {
                    metres((2 * floor + full).div_euclid(2 * full), sea_m)
                },
            });
        }
    }
    out
}

/// The list's form.
pub const FAR_FORM: u8 = 1;

/// ★ THE LIST ON THE WIRE AND IN THE STORE: the form, the count, then each row sorted by node — the node as the
/// difference from the one before (LEB128), the share, the level and the floor (two bytes each, little end first).
#[must_use]
pub fn encode_far_nodes(rows: &[FarNode]) -> Vec<u8> {
    let mut sorted = rows.to_vec();
    sorted.sort_by_key(|r| r.node);
    sorted.dedup_by_key(|r| r.node);
    let mut out = vec![FAR_FORM];
    out.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
    let mut last = 0u32;
    for r in &sorted {
        let mut d = r.node - last;
        last = r.node;
        loop {
            let byte = (d & 0x7f) as u8;
            d >>= 7;
            if d == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
        out.push(r.share);
        out.extend_from_slice(&r.level_m.to_le_bytes());
        out.extend_from_slice(&r.floor_m.to_le_bytes());
    }
    out
}

/// The list back, sorted by node. `None` for another form or a short record.
#[must_use]
pub fn decode_far_nodes(bytes: &[u8]) -> Option<Vec<FarNode>> {
    let [form, c0, c1, c2, c3, rest @ ..] = bytes else {
        return None;
    };
    if *form != FAR_FORM {
        return None;
    }
    let count = u32::from_le_bytes([*c0, *c1, *c2, *c3]) as usize;
    let mut rest = rest;
    let mut out = Vec::with_capacity(count);
    let mut node = 0u32;
    for _ in 0..count {
        let mut d = 0u32;
        let mut shift = 0u32;
        loop {
            let (&byte, tail) = rest.split_first()?;
            rest = tail;
            d |= u32::from(byte & 0x7f).checked_shl(shift)?;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        node = node.checked_add(d)?;
        let [share, l0, l1, f0, f1, tail @ ..] = rest else {
            return None;
        };
        rest = tail;
        out.push(FarNode {
            node,
            share: *share,
            level_m: i16::from_le_bytes([*l0, *l1]),
            floor_m: i16::from_le_bytes([*f0, *f1]),
        });
    }
    rest.is_empty().then_some(out)
}

/// One folded block of a pyramid level: the height it stands at and the water level it holds ([`DRY_M`] where none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarBlock {
    pub z_m: i16,
    pub level_m: i16,
}

/// ★ ONE PYRAMID LEVEL FOLDED FROM THE LIST (the client's own fold, as it folds the coast counts from the mask): for each
/// block of `2^k × 2^k` nodes that holds a listed node, its majority class — at least half its ground wet: the wet cells'
/// mean floor and the highest level its nodes hold; else its dry ground's mean, which is the level's own block mean
/// `z_m(block)` (the mean of its nodes' rows) less the wet cells' part, and NO water ([`DRY_M`]) — the rule a fine tile's
/// coarse copy follows ([`super::tile::class_mean_levels`]), so rung 13 never wets a place rung 12 draws dry. MEASURED
/// before (2026-10-19, the 3 200 / 4 500 km pair over the basin): a dry block carried its lake's level, and every vertex
/// of it whose smooth ground dipped under that level drew a dot of water — hundreds of dots over land rung 12 draws dry.
/// `edge` the body's lattice edge (a node `face·edge² + j·edge + i`), `k` from 1 (a block `face·(edge/2^k)² +
/// (j/2^k)·(edge/2^k) + i/2^k`); `z_m` the pyramid level's own heights. Nothing where the edge does not halve `k` times.
#[must_use]
#[allow(
    clippy::integer_division,
    reason = "the client's fold of a short list, CPU only: an exact integer quotient, one answer on every host"
)]
pub fn fold_level(
    rows: &[FarNode],
    edge: u32,
    k: u32,
    z_m: impl Fn(u32) -> Option<i16>,
) -> BTreeMap<u32, FarBlock> {
    let coarse = edge >> k;
    if coarse == 0 || coarse << k != edge {
        return BTreeMap::new();
    }
    let per_face = edge * edge;
    // Per block: the wet share's sum (255ths of a node), the wet floor's weighted sum, the highest level.
    let mut sums: BTreeMap<u32, (i64, i64, i16)> = BTreeMap::new();
    for r in rows {
        let face = r.node / per_face;
        let rest = r.node - face * per_face;
        let j = rest / edge;
        let i = rest - j * edge;
        let block = face * coarse * coarse + (j >> k) * coarse + (i >> k);
        let e = sums.entry(block).or_insert((0, 0, DRY_M));
        if r.share > 0 && r.floor_m != DRY_M {
            e.0 += i64::from(r.share);
            e.1 += i64::from(r.share) * i64::from(r.floor_m);
        }
        if r.level_m != DRY_M {
            e.2 = if e.2 == DRY_M {
                r.level_m
            } else {
                e.2.max(r.level_m)
            };
        }
    }
    let nodes = 1i64 << (2 * k);
    let mut out = BTreeMap::new();
    for (block, (wet, floor, level)) in sums {
        let Some(mean) = z_m(block) else {
            continue;
        };
        let whole = nodes * 255;
        let wet_class = wet > 0 && 2 * wet >= whole;
        let z = if wet_class {
            (2 * floor + wet).div_euclid(2 * wet)
        } else {
            let dry = whole - wet;
            (2 * (i64::from(mean) * whole - floor) + dry).div_euclid(2 * dry)
        };
        if let Ok(z_m) = i16::try_from(z) {
            out.insert(
                block,
                FarBlock {
                    z_m,
                    level_m: if wet_class { level } else { DRY_M },
                },
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list comes back exactly, sorted, with node gaps of any size; a short record and another form are refused.
    #[test]
    fn the_far_list_round_trips() {
        let rows = vec![
            FarNode {
                node: 900_000,
                share: 255,
                level_m: -2721,
                floor_m: -3050,
            },
            FarNode {
                node: 3,
                share: 0,
                level_m: DRY_M,
                floor_m: DRY_M,
            },
            FarNode {
                node: 130,
                share: 128,
                level_m: 270,
                floor_m: 263,
            },
        ];
        let bytes = encode_far_nodes(&rows);
        let mut sorted = rows.clone();
        sorted.sort_by_key(|r| r.node);
        assert_eq!(decode_far_nodes(&bytes), Some(sorted));
        assert_eq!(decode_far_nodes(&bytes[..bytes.len() - 1]), None);
        let mut other = bytes.clone();
        other[0] = FAR_FORM + 1;
        assert_eq!(decode_far_nodes(&other), None);
    }

    /// A level-1 block of four nodes (an edge of 4): three lake nodes wet whole stand it at their floor with the lake's
    /// level; one lake node of four leaves it at its dry ground's mean — the block's own mean less the wet node's part —
    /// and dry.
    #[test]
    fn a_block_folds_by_its_majority_class() {
        // Nodes 0, 1, 4 and 5 make block 0 of face 0 (edge 4: rows of four nodes; blocks of two by two).
        let wet = |node| FarNode {
            node,
            share: 255,
            level_m: 100,
            floor_m: 40,
        };
        let three = [wet(0), wet(1), wet(4)];
        let folded = fold_level(&three, 4, 1, |_| Some(70));
        assert_eq!(
            folded.get(&0),
            Some(&FarBlock {
                z_m: 40,
                level_m: 100
            })
        );
        // One wet node of four: the dry three stand at (4·70 − 40)/3 = 80, and the block holds no water.
        let one = [wet(0)];
        let folded = fold_level(&one, 4, 1, |_| Some(70));
        assert_eq!(
            folded.get(&0),
            Some(&FarBlock {
                z_m: 80,
                level_m: DRY_M
            })
        );
        assert!(fold_level(&one, 3, 1, |_| Some(70)).is_empty());
    }
}
