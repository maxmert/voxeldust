//! ★ THE FINISH ON ANY TILE OF THE CUBE (B2's last generator item; rulings B81 — one generic stitch, no seams —, B85 —
//! the corner rule — and B86 item 9 (1): every tile finishes, at a face's edge and at a cube corner too).
//!
//! A tile's nine raw rasters may lie on up to three faces, each raster in its own face's chart. Inside a face the corner
//! rule runs in that face's chart ([`super::stitch`]). Across a face's edge two charts disagree on which way is west and
//! which way the rows run, and a cut's ties and its search order follow the chart. So **every piece of the join is
//! computed in ONE chart chosen by rule**: a corner point in the chart of the lowest of the tiles meeting there, an edge's
//! midpoint and each half of its cut in the chart of the lower of its two tiles (tiles ordered by face, then column, then
//! row). Every tile that reads a piece computes it in that chart from the same rasters and turns the answer into its own
//! chart cell by cell ([`super::points::chart_of`]), so both sides of a face's edge join along the same line. The chart a
//! corner is joined in decides its order too: a tile whose chart turns that chart's columns into rows joins rows first.
//!
//! **A cube corner** joins three tiles (the diagonal one does not exist): its cuts start at the corner itself and the
//! three are joined by [`super::stitch::Meet::Three`], the same rule from each of the three tiles.
//!
//! Inside a face every chosen chart is the face's own, and the finish is the B85 finish to the integer.
//!
//! **Example.** The home planet's tile (face 0, column 151, row 40) touches face 0's `+u` edge: its east neighbours lie on
//! face 2, turned. The cut along that edge is searched in face 0's chart (face 0 is the lower); the tile across reads the
//! same cut from face 0's chart and turns it into its own rows, so both draw the same valley floor up to the edge.

use super::l1::{TILE_NODES, TileKey};
use super::points::{chart_cell, chart_of};
use super::stitch::{
    BLEND, CutLine, Field, MISSING, Meet, TileCuts, col_cut_half, corner_point, draw, joined,
    mid_across_columns, mid_across_rows, row_cut_half,
};
use std::borrow::Cow;
use vd_seed::bend::Face;

/// The level's numbers: a tile's side and the cut's band in cells, the cells along a face's edge.
#[derive(Clone, Copy)]
struct Level {
    side: i64,
    band: i64,
    n: i32,
}

/// The place in `to`'s chart of the cell at `(x, y)` of `from`'s chart (`None` where no cell is there, or `to`'s chart
/// does not hold it).
fn to_chart(lv: Level, from: Face, to: Face, x: i64, y: i64) -> Option<(i64, i64)> {
    if from == to {
        return Some((x, y));
    }
    let c = chart_cell(from, x as i32, y as i32, lv.n)?;
    chart_of(to, c, lv.n).map(|(i, j)| (i64::from(i), i64::from(j)))
}

/// A tile's square on its own face (short where the face ends inside the tile).
fn own_square(lv: Level, key: TileKey) -> [i64; 4] {
    let (x0, y0) = (i64::from(key.tx) * lv.side, i64::from(key.ty) * lv.side);
    let n = i64::from(lv.n);
    [x0, y0, (x0 + lv.side).min(n), (y0 + lv.side).min(n)]
}

/// A tile's square in `chart`'s chart.
fn square_in(lv: Level, key: TileKey, chart: Face) -> Option<[i64; 4]> {
    let [x0, y0, x1, y1] = own_square(lv, key);
    let a = to_chart(lv, key.face, chart, x0, y0)?;
    let b = to_chart(lv, key.face, chart, x1 - 1, y1 - 1)?;
    Some([
        a.0.min(b.0),
        a.1.min(b.1),
        a.0.max(b.0) + 1,
        a.1.max(b.1) + 1,
    ])
}

/// `src` (a raster in `from`'s chart) read over the rectangle `[x0, x1) × [y0, y1)` of `to`'s chart.
fn pull(lv: Level, src: &Field, from: Face, to: Face, [x0, y0, x1, y1]: [i64; 4]) -> Field {
    let (w, h) = ((x1 - x0).max(0) as usize, (y1 - y0).max(0) as usize);
    let mut v = Vec::with_capacity(w * h);
    for y in y0..y1 {
        for x in x0..x1 {
            v.push(to_chart(lv, to, from, x, y).map_or(MISSING, |(a, b)| src.at(a, b)));
        }
    }
    Field {
        gx0: x0,
        gy0: y0,
        w,
        h,
        v,
    }
}

/// A tile corner's place (between cells) in `c`'s chart, from the cells around it in `t`'s chart.
fn corner_in(lv: Level, t: Face, (vx, vy): (i64, i64), c: Face) -> Option<(i64, i64)> {
    let mut at: Option<(i64, i64)> = None;
    for (dx, dy) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
        if chart_cell(t, (vx + dx) as i32, (vy + dy) as i32, lv.n).is_none() {
            continue;
        }
        let p = to_chart(lv, t, c, vx + dx, vy + dy)?;
        at = Some(at.map_or(p, |q| (q.0.max(p.0), q.1.max(p.1))));
    }
    at
}

/// Two squares' shared side: whether it runs between columns, its line, its extent `[lo, hi)` along it, and whether the
/// first square is the west (or north) one.
#[derive(Clone, Copy)]
struct Side {
    cols: bool,
    ge: i64,
    lo: i64,
    hi: i64,
    first_west: bool,
}

fn shared_side(a: [i64; 4], b: [i64; 4]) -> Option<Side> {
    let (ylo, yhi) = (a[1].max(b[1]), a[3].min(b[3]));
    let (xlo, xhi) = (a[0].max(b[0]), a[2].min(b[2]));
    let side = |cols: bool, ge: i64, lo: i64, hi: i64, first_west: bool| {
        (lo < hi).then_some(Side {
            cols,
            ge,
            lo,
            hi,
            first_west,
        })
    };
    if a[2] == b[0] {
        side(true, a[2], ylo, yhi, true)
    } else if b[2] == a[0] {
        side(true, a[0], ylo, yhi, false)
    } else if a[3] == b[1] {
        side(false, a[3], xlo, xhi, true)
    } else if b[3] == a[1] {
        side(false, a[1], xlo, xhi, false)
    } else {
        None
    }
}

/// Where a half of a cut ends: a cell of the tile's chart, or a three-tile corner of the tile (its index), where the cut
/// starts at the corner itself.
#[derive(Clone, Copy)]
enum End {
    At((i64, i64)),
    Corner(usize),
}

/// The finish's working set: the level, the tile's face, its nine slots (key and raw raster, own chart), its corners.
struct Work<'a> {
    lv: Level,
    face: Face,
    slots: [Option<(TileKey, &'a Field)>; 9],
    corners: [(i64, i64); 4],
}

/// The slots meeting at each of the tile's corners (NW, NE, SW, SE).
const AT_CORNER: [[usize; 4]; 4] = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]];

impl Work<'_> {
    fn key(&self, slot: usize) -> Option<TileKey> {
        self.slots[slot].map(|(k, _)| k)
    }

    /// The pair `(p, q)`'s chart (the lower tile's face), the side they share there, and their rasters read over `rect`
    /// of that chart (the west or north one first).
    fn pair(&self, p: usize, q: usize) -> Option<(Face, Side, [TileKey; 2])> {
        let (kp, kq) = (self.key(p)?, self.key(q)?);
        let c = kp.min(kq).face;
        let side = shared_side(square_in(self.lv, kp, c)?, square_in(self.lv, kq, c)?)?;
        let first = if side.first_west { [kp, kq] } else { [kq, kp] };
        Some((c, side, first))
    }

    fn raster(&self, key: TileKey) -> Option<&Field> {
        self.slots
            .iter()
            .flatten()
            .find(|(k, _)| *k == key)
            .map(|(_, f)| *f)
    }

    fn pulled(&self, key: TileKey, c: Face, rect: [i64; 4]) -> Option<Field> {
        Some(pull(self.lv, self.raster(key)?, key.face, c, rect))
    }

    /// ★ A CORNER POINT, in the tile's chart: where the four tiles meeting at corner `ci` agree best, searched in the
    /// lowest one's chart; at a three-tile corner the corner's own cell in the tile.
    fn corner_point(&self, ci: usize) -> Option<(i64, i64)> {
        let (vx, vy) = self.corners[ci];
        let keys: Vec<TileKey> = AT_CORNER[ci].iter().filter_map(|&s| self.key(s)).collect();
        if keys.len() < 4 {
            return Some((vx - (ci & 1) as i64, vy - ((ci >> 1) & 1) as i64));
        }
        let c = keys.iter().min()?.face;
        let (cx, cy) = corner_in(self.lv, self.face, (vx, vy), c)?;
        let b = self.lv.band;
        let rect = [cx - b, cy - b, cx + b + 1, cy + b + 1];
        let f: Vec<Field> = keys
            .iter()
            .map(|&k| self.pulled(k, c, rect))
            .collect::<Option<_>>()?;
        let (kx, ky) = corner_point([&f[0], &f[1], &f[2], &f[3]], cx, cy, b);
        to_chart(self.lv, c, self.face, kx, ky)
    }

    /// How the cuts meet at corner `ci`: three tiles, or four joined in the lowest one's chart (rows first where that
    /// chart's columns are the tile's rows).
    fn meet(&self, ci: usize) -> Option<Meet> {
        let keys: Vec<TileKey> = AT_CORNER[ci].iter().filter_map(|&s| self.key(s)).collect();
        if keys.len() < 4 {
            return Some(Meet::Three);
        }
        let c = keys.iter().min()?.face;
        let (cx, cy) = corner_in(self.lv, self.face, self.corners[ci], c)?;
        let a = to_chart(self.lv, c, self.face, cx - 1, cy)?;
        let b = to_chart(self.lv, c, self.face, cx, cy)?;
        Some(Meet::Four {
            rows_first: a.1 != b.1,
        })
    }

    /// ★ AN EDGE'S MIDPOINT, in the tile's chart: where the pair `(p, q)` agree best on their side's middle line, searched
    /// in the lower one's chart.
    fn midpoint(&self, p: usize, q: usize) -> Option<(i64, i64)> {
        let (c, side, [a, b]) = self.pair(p, q)?;
        let (band, mid) = (self.lv.band, (side.lo + side.hi) >> 1);
        let (x, y) = if side.cols {
            let rect = [side.ge - band, mid, side.ge + band + 1, mid + 1];
            (
                mid_across_columns(
                    &self.pulled(a, c, rect)?,
                    &self.pulled(b, c, rect)?,
                    side.ge,
                    mid,
                    band,
                ),
                mid,
            )
        } else {
            let rect = [mid, side.ge - band, mid + 1, side.ge + band + 1];
            (
                mid,
                mid_across_rows(
                    &self.pulled(a, c, rect)?,
                    &self.pulled(b, c, rect)?,
                    side.ge,
                    mid,
                    band,
                ),
            )
        };
        to_chart(self.lv, c, self.face, x, y)
    }

    /// An end's cell in the pair's chart `c` (its side `side`): a cell of the tile's chart turned into `c`'s, or a
    /// three-tile corner's first cell along the side, on its second tile's side of the line.
    fn end_in(&self, end: End, c: Face, side: Side) -> Option<(i64, i64)> {
        match end {
            End::At((x, y)) => to_chart(self.lv, self.face, c, x, y),
            End::Corner(ci) => {
                let (vx, vy) = corner_in(self.lv, self.face, self.corners[ci], c)?;
                let along = if side.cols { vy } else { vx };
                let a = if along == side.lo {
                    side.lo
                } else {
                    side.hi - 1
                };
                Some(if side.cols {
                    (side.ge, a)
                } else {
                    (a, side.ge)
                })
            }
        }
    }

    /// ★ ONE HALF OF A CUT, in the tile's chart: the pair `(p, q)`'s least-cost line from `from` to `to`, searched in the
    /// lower one's chart and turned cell by cell into the tile's.
    fn half(&self, p: usize, q: usize, from: End, to: End) -> Option<CutLine> {
        let (c, side, [a, b]) = self.pair(p, q)?;
        let band = self.lv.band;
        let (e0, e1) = (self.end_in(from, c, side)?, self.end_in(to, c, side)?);
        let along = |e: (i64, i64)| if side.cols { e.1 } else { e.0 };
        let (e0, e1) = if along(e0) <= along(e1) {
            (e0, e1)
        } else {
            (e1, e0)
        };
        let line = if side.cols {
            let rect = [side.ge - band, e0.1, side.ge + band + 1, e1.1 + 1];
            col_cut_half(
                &self.pulled(a, c, rect)?,
                &self.pulled(b, c, rect)?,
                side.ge,
                band,
                e0,
                e1,
            )?
        } else {
            let rect = [e0.0, side.ge - band, e1.0 + 1, side.ge + band + 1];
            row_cut_half(
                &self.pulled(a, c, rect)?,
                &self.pulled(b, c, rect)?,
                side.ge,
                band,
                e0,
                e1,
            )?
        };
        // The line in the tile's chart: along the side as the tile's chart runs it.
        let t_cols = shared_side(
            square_in(self.lv, self.key(p)?, self.face)?,
            square_in(self.lv, self.key(q)?, self.face)?,
        )?
        .cols;
        let mut pts: Vec<(i64, i64)> = Vec::with_capacity(line.path.len());
        for (k, &across) in line.path.iter().enumerate() {
            let a = line.a0 + k as i64;
            let (x, y) = if side.cols { (across, a) } else { (a, across) };
            let (tx, ty) = to_chart(self.lv, c, self.face, x, y)?;
            pts.push(if t_cols { (ty, tx) } else { (tx, ty) });
        }
        pts.sort_unstable();
        let a0 = pts.first()?.0;
        Some(CutLine {
            a0,
            path: pts.into_iter().map(|(_, across)| across).collect(),
        })
    }
}

/// A cut that no corner reads (a pair with a slot the cube does not hold).
fn unread() -> CutLine {
    CutLine {
        a0: 0,
        path: vec![0],
    }
}

/// The slots' offsets from the tile, row by row NW … SE (a column and a row step each).
const SLOT_STEP: [(i64, i64); 9] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// ★ THE NINE TILES AROUND `key` on any face (row by row NW … SE, `2^shift` cells a node, `n` cells along a face's
/// edge): each slot the tile that holds the cell just past the tile's side or corner in its chart; `None` past a cube
/// corner, where no face holds that cell.
#[must_use]
pub fn block_keys(key: TileKey, shift: u32, n: i32) -> [Option<TileKey>; 9] {
    let lv = Level {
        side: i64::from(TILE_NODES) << shift,
        band: 0,
        n,
    };
    let ts = shift + TILE_NODES.trailing_zeros();
    let [x0, y0, x1, y1] = own_square(lv, key);
    SLOT_STEP.map(|(dx, dy)| {
        let x = [x0 - 1, x0, x1][(dx + 1) as usize];
        let y = [y0 - 1, y0, y1][(dy + 1) as usize];
        let c = chart_cell(key.face, x as i32, y as i32, n)?;
        Some(TileKey {
            face: Face::from_index(c.face)?,
            tx: c.i >> ts,
            ty: c.j >> ts,
        })
    })
}

/// A tile's drawing in its own chart: the nine rasters there, its cuts, how each corner meets, its square and side.
struct Prepared<'a> {
    here: Vec<Cow<'a, Field>>,
    cuts: TileCuts,
    meets: [Meet; 4],
    square: [i64; 4],
    side: i64,
}

impl Prepared<'_> {
    fn refs(&self) -> [&Field; 9] {
        std::array::from_fn(|k| self.here[k].as_ref())
    }
}

/// Everything the tile in slot 4 of `block` draws from (see [`finished_square`]).
fn prepare<'a>(
    block: [Option<(TileKey, &'a Field)>; 9],
    shift: u32,
    n: i32,
) -> Option<Prepared<'a>> {
    let lv = Level {
        side: i64::from(TILE_NODES) << shift,
        band: (1i64 << shift) - BLEND,
        n,
    };
    let (key, _) = block[4]?;
    let square = own_square(lv, key);
    let [x0, y0, x1, y1] = square;
    let work = Work {
        lv,
        face: key.face,
        slots: block,
        corners: [(x0, y0), (x1, y0), (x0, y1), (x1, y1)],
    };
    // The rasters in the tile's chart, each over its own square and one node beyond.
    let node = 1i64 << shift;
    let mut here: Vec<Cow<'a, Field>> = Vec::with_capacity(9);
    for slot in block {
        here.push(match slot {
            Some((k, f)) if k.face == key.face => Cow::Borrowed(f),
            Some((k, f)) => {
                let [a, b, c, d] = square_in(lv, k, key.face)?;
                Cow::Owned(pull(
                    lv,
                    f,
                    k.face,
                    key.face,
                    [a - node, b - node, c + node, d + node],
                ))
            }
            None => Cow::Owned(Field {
                gx0: 0,
                gy0: 0,
                w: 0,
                h: 0,
                v: Vec::new(),
            }),
        });
    }
    let points: Vec<(i64, i64)> = (0..4)
        .map(|ci| work.corner_point(ci))
        .collect::<Option<_>>()?;
    let meets: Vec<Meet> = (0..4).map(|ci| work.meet(ci)).collect::<Option<_>>()?;
    // A corner's end of a half: its point, or the corner itself where three tiles meet.
    let end = |ci: usize| {
        if meets[ci] == Meet::Three {
            End::Corner(ci)
        } else {
            End::At(points[ci])
        }
    };
    let mid = |p: usize, q: usize| work.midpoint(p, q).map(End::At);
    // An outer half (a neighbour pair's, from its midpoint to the tile's corner), or none where a slot is missing.
    let outer = |p: usize, q: usize, ci: usize| -> Option<CutLine> {
        if work.key(p).is_none() | work.key(q).is_none() {
            return Some(unread());
        }
        work.half(p, q, mid(p, q)?, end(ci))
    };
    // The tile's own edge: its two halves joined at the midpoint, in the tile's chart's order along it.
    let whole = |p: usize, q: usize, c0: usize, c1: usize| -> Option<CutLine> {
        let m = mid(p, q)?;
        Some(joined(
            work.half(p, q, end(c0), m)?,
            &work.half(p, q, m, end(c1))?,
        ))
    };
    let cuts = TileCuts {
        corners: [points[0], points[1], points[2], points[3]],
        cols: [
            [outer(0, 1, 0)?, outer(1, 2, 1)?],
            [whole(3, 4, 0, 2)?, whole(4, 5, 1, 3)?],
            [outer(6, 7, 2)?, outer(7, 8, 3)?],
        ],
        rows: [
            [outer(0, 3, 0)?, whole(1, 4, 0, 1)?, outer(2, 5, 1)?],
            [outer(3, 6, 2)?, whole(4, 7, 2, 3)?, outer(5, 8, 3)?],
        ],
    };
    Some(Prepared {
        here,
        cuts,
        meets: [meets[0], meets[1], meets[2], meets[3]],
        square,
        side: lv.side,
    })
}

/// ★ THE FINISHED SQUARE of the tile in slot 4 of `block` (row by row NW … SE, as [`block_keys`] names them; each slot
/// its tile's key and raw raster in that tile's own face chart; `None` past a cube corner), in the tile's own chart:
/// `TILE_NODES` nodes of `2^shift` cells a side, `n` cells along a face's edge. `None` where the tile is missing, a
/// neighbour is not where its key says, or a cut cannot be joined.
#[must_use]
pub fn finished_square(block: [Option<(TileKey, &Field)>; 9], shift: u32, n: i32) -> Option<Field> {
    let p = prepare(block, shift, n)?;
    Some(draw(p.refs(), &p.cuts, p.square, p.side, p.meets))
}

#[cfg(test)]
mod tests;
