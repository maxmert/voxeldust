//! The finish on a small cube: six faces of three by three tiles, a node of sixteen cells (the cut's band eight), each
//! tile's raw raster a smooth ground of the true cell plus the tile's own offset times the squared distance to the
//! nearest of a set of agreement cells — so every tile agrees exactly there and nowhere else, on every face.
#![allow(clippy::integer_division, clippy::modulo_arithmetic, reason = "a fixture's placing; never a kernel's path")]

use super::*;
use crate::fine::points::CellId;
use crate::fine::stitch::{draw_over, finished_tile};
use std::collections::BTreeMap;
use vd_recipe::bend::{direction, inv_n_of};

const SHIFT: u32 = 4;
const SIDE: i64 = 128;
const NODE: i64 = 16;
const N: i32 = 384;
const BAND: i64 = NODE - BLEND;

/// A cell's direction at twenty fraction bits.
fn dir(c: CellId) -> [i64; 3] {
    direction(i32::from(c.face), c.i, c.j, inv_n_of(N as u32)).map(|g| g.raw() >> 20)
}

/// The smooth ground of a true cell (about four quarter metres a cell).
fn ground(c: CellId) -> i32 {
    let d = dir(c);
    ((d[0] + 2 * d[1] - d[2]) >> 10) as i32
}

/// The squared distance in cells (about) between two directions.
fn d2(a: [i64; 3], b: [i64; 3]) -> i64 {
    ((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2) + (a[2] - b[2]).pow(2)) >> 24
}

/// The agreement cells: near every tile corner of every face, moved by a table.
fn agreement() -> Vec<[i64; 3]> {
    let moves = [(3, -2), (-4, 1), (2, 4), (-1, -3)];
    let mut out = Vec::new();
    for face in 0..6u8 {
        for cj in 0..=3i32 {
            for ci in 0..=3i32 {
                let (mx, my) = moves[((ci + 2 * cj) & 3) as usize];
                let (i, j) = ((ci * 128 + mx).clamp(0, N - 1), (cj * 128 + my).clamp(0, N - 1));
                out.push(dir(CellId { face, i, j }));
            }
        }
    }
    out
}

/// Every tile of the cube.
fn every_tile() -> Vec<TileKey> {
    Face::ALL.iter().flat_map(|&face| (0..9).map(move |k| TileKey { face, tx: k % 3, ty: k / 3 })).collect()
}

/// The raw raster of `key` in its own chart (its square and a node beyond): the ground, plus (when `offsets`) the tile's
/// own offset times the squared distance to the nearest agreement cell.
fn raw(key: TileKey, offsets: bool, agree: &[[i64; 3]]) -> Field {
    let (x0, y0) = (i64::from(key.tx) * SIDE - NODE, i64::from(key.ty) * SIDE - NODE);
    let w = (SIDE + 2 * NODE) as usize;
    let off = if offsets { 1 + i32::from(key.face.index()) * 9 + key.ty * 3 + key.tx } else { 0 };
    let centre = dir(CellId { face: key.face.index(), i: key.tx * 128 + 64, j: key.ty * 128 + 64 });
    let near: Vec<[i64; 3]> = agree.iter().copied().filter(|&a| d2(a, centre) < 300 * 300).collect();
    let mut v = Vec::with_capacity(w * w);
    for y in y0..y0 + w as i64 {
        for x in x0..x0 + w as i64 {
            v.push(chart_cell(key.face, x as i32, y as i32, N).map_or(MISSING, |c| {
                let dc = dir(c);
                ground(c) + off * near.iter().map(|&a| d2(a, dc)).min().unwrap_or(0) as i32
            }));
        }
    }
    Field { gx0: x0, gy0: y0, w, h: w, v }
}

fn rasters(offsets: bool) -> BTreeMap<TileKey, Field> {
    let agree = agreement();
    every_tile().into_iter().map(|k| (k, raw(k, offsets, &agree))).collect()
}

fn block(key: TileKey, raws: &BTreeMap<TileKey, Field>) -> [Option<(TileKey, &Field)>; 9] {
    block_keys(key, SHIFT, N).map(|k| k.map(|k| (k, &raws[&k])))
}

#[test]
fn the_nine_tiles_around_a_tile_cross_faces_and_skip_the_place_past_a_cube_corner() {
    let k = |face, tx, ty| Some(TileKey { face, tx, ty });
    // Inside a face: the face's own grid.
    let inner = block_keys(TileKey { face: Face::PosX, tx: 1, ty: 1 }, SHIFT, N);
    assert_eq!(inner[0], k(Face::PosX, 0, 0));
    assert_eq!(inner[8], k(Face::PosX, 2, 2));
    // At a cube corner: eight tiles on three faces, nothing diagonally past the corner.
    let corner = block_keys(TileKey { face: Face::PosX, tx: 0, ty: 0 }, SHIFT, N);
    assert_eq!(corner[0], None);
    assert_eq!(corner.iter().flatten().count(), 8);
    let faces: std::collections::BTreeSet<u8> = corner.iter().flatten().map(|t| t.face.index()).collect();
    assert_eq!(faces.len(), 3);
}

#[test]
fn inside_a_face_the_finish_is_the_b85_finish_to_the_integer() {
    let raws = rasters(true);
    for face in Face::ALL {
        let key = TileKey { face, tx: 1, ty: 1 };
        let b = block(key, &raws);
        let fields: [&Field; 9] = std::array::from_fn(|s| b[s].map(|(_, f)| f).expect("an inner tile's nine"));
        assert_eq!(finished_square(b, SHIFT, N), finished_tile(fields, 1, 1, SIDE, BAND), "{face:?}");
    }
}

#[test]
fn rasters_that_agree_finish_into_their_own_ground_on_every_tile_of_the_cube() {
    let raws = rasters(false);
    for key in every_tile() {
        let square = finished_square(block(key, &raws), SHIFT, N).expect("finishes");
        for y in 0..SIDE {
            for x in 0..SIDE {
                let (i, j) = ((i64::from(key.tx) * SIDE + x) as i32, (i64::from(key.ty) * SIDE + y) as i32);
                let c = CellId { face: key.face.index(), i, j };
                assert_eq!(square.v[(y * SIDE + x) as usize], ground(c), "{key:?} ({x}, {y})");
            }
        }
    }
}

/// ★ THE DECISIVE TEST: every tile's own rule, read across each of its four sides over its neighbour's cells, draws
/// exactly the neighbour's finished heights — inside a face, across every face's edge (turned or not), and along every
/// side of a cube-corner tile away from the three-tile corner itself (whose rule each of the three reads in its own
/// quarter only).
#[test]
fn both_sides_of_every_side_draw_the_same_ground_across_faces() {
    let raws = rasters(true);
    let finished: BTreeMap<TileKey, Field> =
        every_tile().into_iter().map(|k| (k, finished_square(block(k, &raws), SHIFT, N).expect("every tile finishes"))).collect();
    let lv = Level { side: SIDE, band: BAND, n: N };
    let (mut compared, mut across_faces) = (0usize, 0usize);
    for key in every_tile() {
        let p = prepare(block(key, &raws), SHIFT, N).expect("prepares");
        let [x0, y0, x1, y1] = p.square;
        let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
        let near_three = |x: i64, y: i64| {
            (0..4).any(|ci| p.meets[ci] == Meet::Three && (x - corners[ci].0).abs().max((y - corners[ci].1).abs()) < 2 * NODE)
        };
        // The strips one node deep past each side, in this tile's chart.
        let strips = [[x0, y0 - NODE, x1, y0], [x0, y1, x1, y1 + NODE], [x0 - NODE, y0, x0, y1], [x1, y0, x1 + NODE, y1]];
        for rect in strips {
            let drawn = draw_over(p.refs(), &p.cuts, p.square, rect, p.meets);
            for y in rect[1]..rect[3] {
                for x in rect[0]..rect[2] {
                    if near_three(x, y) {
                        continue;
                    }
                    let c = chart_cell(key.face, x as i32, y as i32, N).expect("a side's strip lies on a face");
                    let other = TileKey { face: Face::from_index(c.face).expect("a face"), tx: c.i >> 7, ty: c.j >> 7 };
                    let (ox, oy) = to_chart(lv, key.face, other.face, x, y).expect("the neighbour's chart holds it");
                    let theirs = finished[&other].at(ox, oy);
                    assert_eq!(drawn.at(x, y), theirs, "{key:?} → {other:?} at ({x}, {y})");
                    compared += 1;
                    across_faces += usize::from(other.face != key.face);
                }
            }
        }
    }
    // Every tile's four strips, less the three-tile corners' reach.
    assert!(compared > 54 * 4 * 128 * 16 * 9 / 10, "{compared}");
    assert!(across_faces > 0);
}

#[test]
fn a_block_that_does_not_fit_the_cube_finishes_nothing() {
    let raws = rasters(false);
    let key = TileKey { face: Face::PosX, tx: 1, ty: 1 };
    // No tile in the middle.
    let mut b = block(key, &raws);
    b[4] = None;
    assert!(finished_square(b, SHIFT, N).is_none());
    // A neighbour from the opposite face: no chart of the tile holds it.
    let far = TileKey { face: Face::NegX, tx: 1, ty: 1 };
    let mut b = block(key, &raws);
    b[5] = Some((far, &raws[&far]));
    assert!(finished_square(b, SHIFT, N).is_none());
    // A neighbour that does not touch the tile.
    let apart = TileKey { face: Face::PosX, tx: 1, ty: 1 };
    let corner = TileKey { face: Face::PosX, tx: 0, ty: 0 };
    let mut b = block(apart, &raws);
    b[5] = Some((corner, &raws[&corner]));
    assert!(finished_square(b, SHIFT, N).is_none());
}
