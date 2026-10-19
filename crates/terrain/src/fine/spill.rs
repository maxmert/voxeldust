//! ★ A LAKE STANDS AT ITS 128 m SPILL (ruling B115; W11's law — a still water stands at most at its lowest outlet — read
//! on the finer ground; `docs/investigation/2026-10-18/lake_spill_design.md`).
//!
//! The 8 km world's lake stands at its brim node's height. The 128 m level holds only that node's MEAN to its row (B84),
//! so the brim's own 128 m valleys can cut the rim under the lake's level, and the water map, which spreads each water
//! through every cell under its level that a path joins, would run the lake out through the cut. A lake's FINE LEVEL is
//! therefore the lower of its 8 km level and its lowest 128 m pass to lower ground.
//!
//! The water map spreads inside ONE raster at a time ([`super::water::raster_water`]), so the pass is found per raster
//! ([`raster_passes`]): from the lake's own cells, the path whose highest ground is lowest, to a cell of a node outside the
//! lake whose row stands under the lake's level. The lake's fine level is the lowest such pass over every raster that
//! holds one of its cells ([`fine_level`]); at that level no raster's spread reaches lower ground. Integers in quarter
//! metres over the sea; ties by the cell's index; one answer on every host.
//!
//! **Example.** Lake #16 stands at −163 m. Its brim node's row is −163 m, but in the raster of its brim's tile one 128 m
//! valley runs down to the basin at −936 m, and the lowest path from the lake's cells to the basin's cells crosses ground no
//! higher than −936 m: the raster's pass is −936 m, and the lake's fine level is the lowest pass over its rasters.

use crate::artifact::{Artifact, DRY_M, FACIES_SHIFT};
use crate::macro_lattice::{MacroLattice, NO_NODE};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A node no lake holds.
pub const NO_LAKE: u32 = u32::MAX;

/// Whether a node's standing water is a lake: its word over its row, and not the sea.
fn is_lake(artifact: &Artifact, node: usize) -> bool {
    let r = &artifact.rows[node];
    (r.receiver_facies >> FACIES_SHIFT) & crate::solve::FACIES_SEA == 0
        && r.water_m != DRY_M
        && r.water_m > r.z_m
}

/// ★ EACH NODE'S LAKE: the lowest node index of the connected nodes under one lake word (the same water level, eight
/// neighbours), or [`NO_LAKE`]. The nodes are visited in index order, so a lake's id is its first node.
#[must_use]
pub fn lake_ids(lattice: &MacroLattice, artifact: &Artifact) -> Vec<u32> {
    let n = artifact.rows.len();
    let mut id = vec![NO_LAKE; n];
    for start in 0..n {
        if id[start] != NO_LAKE || !is_lake(artifact, start) {
            continue;
        }
        let word = artifact.rows[start].water_m;
        id[start] = start as u32;
        let mut stack = vec![start as u32];
        while let Some(i) = stack.pop() {
            for m in lattice.neighbours(i) {
                if m != NO_NODE
                    && id[m as usize] == NO_LAKE
                    && artifact.rows[m as usize].water_m == word
                    && is_lake(artifact, m as usize)
                {
                    id[m as usize] = start as u32;
                    stack.push(m);
                }
            }
        }
    }
    id
}

/// ★ THE LAKES' PASSES IN ONE RASTER: for each lake with a cell in the raster (`cell_lake`, [`NO_LAKE`] where none), the
/// lowest path — the highest ground along it — from its own cells to a cell of another node whose row (`cell_row`, quarter
/// metres over the sea) stands under the lake's level (`level`, its 8 km level in quarter metres), where that path stays
/// under the level. `ground` the raster's heights (`side` a row; [`super::water::DRY`] where no face holds a cell, never
/// crossed). Returns `(lake, pass)` sorted by lake; a lake with no such path in this raster is not listed.
#[must_use]
pub fn raster_passes(
    ground: &[i32],
    side: usize,
    cell_lake: &[u32],
    cell_row: &[i32],
    level: impl Fn(u32) -> i32,
) -> Vec<(u32, i32)> {
    let n = ground.len();
    let mut lakes: Vec<u32> = cell_lake
        .iter()
        .copied()
        .filter(|&l| l != NO_LAKE)
        .collect();
    lakes.sort_unstable();
    lakes.dedup();
    let open = |c: usize| ground[c] != super::water::DRY;
    // Each cell's column in its row, without a division (the crate's integer fence).
    let column: Vec<usize> = (0..n).map(|k| k % side).collect();
    let mut out = Vec::new();
    for lake in lakes {
        let lam = level(lake);
        let mut best = vec![i32::MAX; n];
        let mut heap = BinaryHeap::new();
        for (c, b) in best.iter_mut().enumerate() {
            if cell_lake[c] == lake && open(c) {
                *b = ground[c];
                heap.push(Reverse((ground[c], c)));
            }
        }
        while let Some(Reverse((h, c))) = heap.pop() {
            if h > best[c] {
                continue;
            }
            if h >= lam {
                break;
            }
            if cell_lake[c] != lake && cell_row[c] < lam {
                out.push((lake, h));
                break;
            }
            let x = column[c];
            let around = [
                (x > 0).then(|| c - 1),
                (x + 1 < side).then(|| c + 1),
                (c >= side).then(|| c - side),
                (c + side < n).then(|| c + side),
            ];
            for m in around.into_iter().flatten() {
                if !open(m) {
                    continue;
                }
                let hm = h.max(ground[m]);
                if hm < best[m] {
                    best[m] = hm;
                    heap.push(Reverse((hm, m)));
                }
            }
        }
    }
    out
}

/// ★ A LAKE'S FINE LEVEL: the lower of its 8 km level and every pass its rasters found (quarter metres over the sea).
#[must_use]
pub fn fine_level(level_8km: i32, passes: impl IntoIterator<Item = i32>) -> i32 {
    passes.into_iter().fold(level_8km, i32::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row of seven cells: the lake's cells 0..2 (level 40), its brim node's cell 3 (row 40, ground 25), lower ground
    /// beyond (a node whose row stands at 10) — the pass is the brim cell's 25; with the brim's ground at 50 there is no pass
    /// under the level; a second lake at a lower level beyond counts as lower ground too, and it finds no pass of its own.
    #[test]
    fn a_lake_spills_at_its_lowest_rim_cell() {
        let lake = 7;
        let ground = [5, 8, 12, 25, 9, 6, 4];
        let cell_lake = [lake, lake, lake, NO_LAKE, NO_LAKE, NO_LAKE, NO_LAKE];
        let cell_row = [3, 3, 3, 40, 10, 10, 10];
        assert_eq!(
            raster_passes(&ground, 7, &cell_lake, &cell_row, |_| 40),
            [(lake, 25)]
        );
        let high_rim = [5, 8, 12, 50, 9, 6, 4];
        assert!(raster_passes(&high_rim, 7, &cell_lake, &cell_row, |_| 40).is_empty());
        let other = [lake, lake, lake, NO_LAKE, 9, 9, 9];
        assert_eq!(
            raster_passes(&ground, 7, &other, &cell_row, |l| if l == lake {
                40
            } else {
                20
            }),
            [(lake, 25)]
        );
        assert_eq!(fine_level(40, [25, 31]), 25);
        assert_eq!(fine_level(40, []), 40);
    }
}
