//! ★ THE ONE STITCH (B1 (e); ruling B81: one generic treatment for every border of every level; the way-in design
//! revision 4 §3.4) and THE MEAN PYRAMID (§3.3).
//!
//! **The cut** (Efros & Freeman 2001): two neighbour tiles both solve their overlap. Along their shared edge, every row
//! of the overlap takes the column within `±band` of the edge where the two heights differ least along a path that moves
//! at most one column a row — mostly along a divide, where both agree. Each side draws its own heights up to the cut, and
//! a smooth step [`BLEND`] cells each side crosses it. The costs are the squared differences in quarter metres
//! (integers); the path's ties go to the western column, the last row's least cost too. The step is the smoothstep at
//! its exact rational points: with `s = d + 8` (`d` the cell's column less the cut's, held to `±8`) the east side's
//! weight is `2·s²·(24 − s)` out of 4 096, so every host blends with the same integers. A border across rows is the same
//! cut on the swapped axes.
//!
//! **The corner rule** (ruling B85): every edge's cut runs in two halves, from each of its corner points to its
//! midpoint; a corner point is where the four tiles meeting there agree best, a midpoint where the edge's two tiles agree
//! best on the edge's middle line. So every cut is a pure function of the two tiles beside it and its ends, and a drawn
//! cell depends only on its tile and the eight around it: the planet's shard draws the FINISHED tile and ships it. A cell
//! is joined first within its row of tiles across that row's column cuts, then across the nearer row cut. MEASURED (B85,
//! sixteen solved belt tiles): the four finished tiles meet the bench's two-step stitch on every reading (the slope break
//! across the cuts 1.669 against 1.669 at the 99th, at the four-tile corner 1.304 against 1.307), and the step across the
//! finished tiles' own straight edges reads as the stitched ground's on the same lines (9.17 against 9.25 m at the median).
//!
//! **The mean pyramid**: a tile's coarser copies — each level's box means of `2^k × 2^k` cells taken from the raster's
//! exact integer sums (no rounding carried from level to level), rounded half up. A rung reads the level whose cell is
//! its own, so a swap compares two box-mean levels of one surface.
//!
//! **Example.** Two 128 m tiles of the home planet's belt meet along a node edge; the cut runs along the divide north of
//! the river they share, the 8-cell step crosses it, and the far copies are made from that drawn surface — no straight
//! tile edge shows at any distance. Where four belt tiles meet, the four cuts end at the one cell where all four agree,
//! and each of the four tiles finds that same cell from the same four rasters.

/// A missing height (a cell no tile holds).
pub const MISSING: i32 = i32::MIN;
/// The blend's half-width in cells (B0 step 2a: 3 cells left marks where the cuts cross; 8 left none).
pub const BLEND: i64 = 8;
/// The blend's weights' one: 4 096.
const ONE: i64 = 4_096;

/// A raster of quarter metres placed on the level's global cells: column `gx0 ..`, row `gy0 ..`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub gx0: i64,
    pub gy0: i64,
    pub w: usize,
    pub h: usize,
    pub v: Vec<i32>,
}

impl Field {
    /// The height at a global cell, or [`MISSING`] outside the field.
    #[must_use]
    pub fn at(&self, gx: i64, gy: i64) -> i32 {
        let (x, y) = (gx - self.gx0, gy - self.gy0);
        if (x < 0) | (y < 0) | (x >= self.w as i64) | (y >= self.h as i64) {
            MISSING
        } else {
            self.v[y as usize * self.w + x as usize]
        }
    }

    /// The same field with its axes swapped (a cut across rows is a cut across columns of the swapped fields).
    #[must_use]
    pub fn swapped(&self) -> Field {
        let mut v = vec![0i32; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                v[x * self.h + y] = self.v[y * self.w + x];
            }
        }
        Field {
            gx0: self.gy0,
            gy0: self.gx0,
            w: self.h,
            h: self.w,
            v,
        }
    }
}

/// The east side's blend weight out of 4 096 at `d` columns from the cut (east positive).
#[must_use]
pub fn blend_weight(d: i64) -> i64 {
    let s = d.clamp(-BLEND, BLEND) + BLEND;
    2 * s * s * (24 - s)
}

/// The blended height of the two sides' heights at `d` columns from the cut; one side missing draws the other.
#[must_use]
pub fn blend(west: i32, east: i32, d: i64) -> i32 {
    if west == MISSING {
        return east;
    }
    if east == MISSING {
        return west;
    }
    let w = blend_weight(d);
    ((i64::from(west) * (ONE - w) + i64::from(east) * w + (ONE >> 1)) >> 12) as i32
}

/// ★ THE CUT along the global column `ge`: for every row both fields hold, the column within `ge ± band`.
#[must_use]
pub fn cut(west: &Field, east: &Field, ge: i64, band: i64) -> (i64, Vec<i64>) {
    let gy0 = west.gy0.max(east.gy0);
    let gy1 = (west.gy0 + west.h as i64).min(east.gy0 + east.h as i64);
    let h = (gy1 - gy0).max(0) as usize;
    let cols = (2 * band + 1) as usize;
    let mut cost = vec![0i64; h * cols];
    let mut from = vec![0usize; h * cols];
    for y in 0..h {
        for k in 0..cols {
            let x = ge - band + k as i64;
            let (a, b) = (west.at(x, gy0 + y as i64), east.at(x, gy0 + y as i64));
            let e = if (a == MISSING) | (b == MISSING) {
                1i64 << 40
            } else {
                let d = i64::from(a) - i64::from(b);
                d * d
            };
            if y == 0 {
                cost[k] = e;
                continue;
            }
            let (lo, hi) = (k.saturating_sub(1), (k + 1).min(cols - 1));
            let mut best = (i64::MAX, lo);
            for kk in lo..=hi {
                if cost[(y - 1) * cols + kk] < best.0 {
                    best = (cost[(y - 1) * cols + kk], kk);
                }
            }
            cost[y * cols + k] = e + best.0;
            from[y * cols + k] = best.1;
        }
    }
    let mut path = vec![ge; h];
    if h > 0 {
        let mut k = (0..cols)
            .min_by(|&p, &q| {
                cost[(h - 1) * cols + p]
                    .cmp(&cost[(h - 1) * cols + q])
                    .then(p.cmp(&q))
            })
            .unwrap_or(0);
        for y in (0..h).rev() {
            path[y] = ge - band + k as i64;
            k = from[y * cols + k];
        }
    }
    (gy0, path)
}

/// ★ THE STITCH of a west and an east field along the global column `ge`: the cut, then each side up to it with the
/// blend across it; the result covers both fields' columns over the rows they share.
#[must_use]
pub fn stitch_columns(west: &Field, east: &Field, ge: i64, band: i64) -> (Field, Vec<i64>) {
    let (gy0, path) = cut(west, east, ge, band);
    let h = path.len();
    let gx0 = west.gx0.min(east.gx0);
    let gx1 = (west.gx0 + west.w as i64).max(east.gx0 + east.w as i64);
    let w = (gx1 - gx0) as usize;
    let mut v = vec![MISSING; w * h];
    for (y, &c) in path.iter().enumerate() {
        for x in 0..w {
            let (gx, gy) = (gx0 + x as i64, gy0 + y as i64);
            v[y * w + x] = blend(west.at(gx, gy), east.at(gx, gy), gx - c);
        }
    }
    (Field { gx0, gy0, w, h, v }, path)
}

/// The same stitch across the global row `ge` (`north` above, `south` below).
#[must_use]
pub fn stitch_rows(north: &Field, south: &Field, ge: i64, band: i64) -> (Field, Vec<i64>) {
    let (f, path) = stitch_columns(&north.swapped(), &south.swapped(), ge, band);
    (f.swapped(), path)
}

/// ★ THE MEAN PYRAMID of the `side × side` square of `f` from global cell `(gx, gy)`: level `k` (1 ..= `levels`) the
/// box means of `2^k × 2^k` cells, from the raster's own integer sums, rounded half up. `None` where a cell is missing or
/// the side does not halve so often.
#[must_use]
pub fn mean_levels(f: &Field, gx: i64, gy: i64, side: usize, levels: u32) -> Option<Vec<Vec<i32>>> {
    if (side >> levels) << levels != side {
        return None;
    }
    let mut out = Vec::with_capacity(levels as usize);
    for k in 1..=levels {
        let n = side >> k;
        let mut level = Vec::with_capacity(n * n);
        for by in 0..n {
            for bx in 0..n {
                let mut sum = 0i64;
                for y in 0..1usize << k {
                    for x in 0..1usize << k {
                        let z = f.at(gx + ((bx << k) + x) as i64, gy + ((by << k) + y) as i64);
                        if z == MISSING {
                            return None;
                        }
                        sum += i64::from(z);
                    }
                }
                level.push(((sum + (1i64 << (2 * k - 1))) >> (2 * k)) as i32);
            }
        }
        out.push(level);
    }
    Some(out)
}

/// A cut as a function of the coordinate along its edge: `across(a)` for `a` in `a0 ..`, held at its ends beyond them
/// (so the cuts of two rows of tiles meet exactly at their shared corner point).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutLine {
    pub a0: i64,
    pub path: Vec<i64>,
}

impl CutLine {
    /// The cut's across-coordinate at `a`.
    #[must_use]
    pub fn at(&self, a: i64) -> i64 {
        let k = (a - self.a0).clamp(0, self.path.len() as i64 - 1);
        self.path[k as usize]
    }
}

/// The squared difference of two heights, or a cost no cut takes where either is missing.
fn diff2(a: i32, b: i32) -> i64 {
    if (a == MISSING) | (b == MISSING) {
        1i64 << 40
    } else {
        let d = i64::from(a) - i64::from(b);
        d * d
    }
}

/// ★ ONE HALF OF A CUT: the least-cost path from `(a0, c0)` to `(a1, c1)` (`a` along the edge, `c` across it, within
/// `[lo, hi]`), one step across at most per step along, `cost(a, c)` each cell; ties to the lower `c`. `None` where the
/// ends cannot be joined so.
#[must_use]
pub fn half_path(
    cost: &dyn Fn(i64, i64) -> i64,
    (a0, c0): (i64, i64),
    (a1, c1): (i64, i64),
    lo: i64,
    hi: i64,
) -> Option<CutLine> {
    if (a1 < a0) | ((c1 - c0).abs() > a1 - a0) | (c0 < lo) | (c0 > hi) | (c1 < lo) | (c1 > hi) {
        return None;
    }
    let (steps, cols) = ((a1 - a0 + 1) as usize, (hi - lo + 1) as usize);
    let mut best = vec![i64::MAX; steps * cols];
    let mut from = vec![0usize; steps * cols];
    best[(c0 - lo) as usize] = cost(a0, c0);
    for s in 1..steps {
        for k in 0..cols {
            let (kl, kh) = (k.saturating_sub(1), (k + 1).min(cols - 1));
            let mut b = (i64::MAX, kl);
            for kk in kl..=kh {
                let v = best[(s - 1) * cols + kk];
                if v < b.0 {
                    b = (v, kk);
                }
            }
            if b.0 != i64::MAX {
                best[s * cols + k] = b.0 + cost(a0 + s as i64, lo + k as i64);
                from[s * cols + k] = b.1;
            }
        }
    }
    let mut path = vec![0i64; steps];
    let mut k = (c1 - lo) as usize;
    for s in (0..steps).rev() {
        path[s] = lo + k as i64;
        k = from[s * cols + k];
    }
    Some(CutLine { a0, path })
}

/// The two halves of an edge's cut joined at the midpoint into one line from the first corner to the second.
pub(crate) fn joined(first: CutLine, second: &CutLine) -> CutLine {
    let mut path = first.path;
    path.extend_from_slice(&second.path[1..]);
    CutLine { a0: first.a0, path }
}

/// ★ A CORNER POINT: where the four tiles meeting at the corner `(gx, gy)` agree best — the least sum of the six pairs'
/// squared differences within `±band` of the corner, ties to the lower row, then the lower column.
#[must_use]
pub fn corner_point(four: [&Field; 4], gx: i64, gy: i64, band: i64) -> (i64, i64) {
    let mut best = (i64::MAX, gx, gy);
    for y in gy - band..=gy + band {
        for x in gx - band..=gx + band {
            let z = four.map(|f| f.at(x, y));
            let mut c = 0i64;
            for a in 0..4 {
                for b in a + 1..4 {
                    c = c.saturating_add(diff2(z[a], z[b]));
                }
            }
            if c < best.0 {
                best = (c, x, y);
            }
        }
    }
    (best.1, best.2)
}

/// ★ AN EDGE'S MIDPOINT across a column edge `ge` (`west`, `east`) on the row `gy`: the column within `±band` where the two
/// agree best, ties west.
#[must_use]
pub fn mid_across_columns(west: &Field, east: &Field, ge: i64, gy: i64, band: i64) -> i64 {
    (ge - band..=ge + band)
        .min_by_key(|&x| (diff2(west.at(x, gy), east.at(x, gy)), x))
        .unwrap_or(ge)
}

/// The same across a row edge `ge` (`north`, `south`) on the column `gx`: the row where the two agree best, ties north.
#[must_use]
pub fn mid_across_rows(north: &Field, south: &Field, ge: i64, gx: i64, band: i64) -> i64 {
    (ge - band..=ge + band)
        .min_by_key(|&y| (diff2(north.at(gx, y), south.at(gx, y)), y))
        .unwrap_or(ge)
}

/// One half of a column edge's cut between `west` and `east` at the column `ge`, from the cell `from` to the cell `to`
/// (each `(x, y)`, `from` the upper).
pub(crate) fn col_cut_half(
    west: &Field,
    east: &Field,
    ge: i64,
    band: i64,
    from: (i64, i64),
    to: (i64, i64),
) -> Option<CutLine> {
    let cost = |y: i64, x: i64| diff2(west.at(x, y), east.at(x, y));
    half_path(&cost, (from.1, from.0), (to.1, to.0), ge - band, ge + band)
}

/// One half of a row edge's cut between `north` and `south` at the row `ge`, from the cell `from` to the cell `to`
/// (each `(x, y)`, `from` the western).
pub(crate) fn row_cut_half(
    north: &Field,
    south: &Field,
    ge: i64,
    band: i64,
    from: (i64, i64),
    to: (i64, i64),
) -> Option<CutLine> {
    let cost = |x: i64, y: i64| diff2(north.at(x, y), south.at(x, y));
    half_path(&cost, from, to, ge - band, ge + band)
}

/// The cut lines a tile's drawing reads (ruling B85). Each is a pure function of the two tiles beside its edge and of its
/// two ends (a corner point and an edge's midpoint), so a tile and its neighbour compute the same line for their edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileCuts {
    /// The tile's corner points: north-west, north-east, south-west, south-east.
    pub corners: [(i64, i64); 4],
    /// The column cuts west and east of the tile, by row of tiles: the row above (their lower halves, from the edge's
    /// midpoint to this tile's corner), the tile's own row (whole, corner to corner), the row below (their upper halves).
    pub cols: [[CutLine; 2]; 3],
    /// The row cuts north and south of the tile, by column of tiles: the column to the west (its half from the edge's
    /// midpoint to this tile's corner), the tile's own column (whole), the column to the east (its half from the corner).
    pub rows: [[CutLine; 3]; 2],
}

/// One half of a cut to search: the two tiles, the edge's global column (or row), the two ends `(x, y)` (the upper, or the
/// western, first), and whether the edge runs between columns.
type HalfCut<'a> = (&'a Field, &'a Field, i64, (i64, i64), (i64, i64), bool);

/// ★ A TILE'S CUTS (ruling B85) from the nine raw rasters around it (`block` row by row: NW, N, NE, W, the tile, E, SW, S,
/// SE): each edge's cut in two halves, from its corner points to its midpoint, a corner point from its four tiles, a
/// midpoint from its two. `None` where a half cannot join its ends (a band over a sixth of the tile's side).
#[must_use]
pub fn tile_cuts(block: [&Field; 9], tx: i64, ty: i64, side: i64, band: i64) -> Option<TileCuts> {
    let [nw, n, ne, w, t, e, sw, s, se] = block;
    let (x0, y0, x1, y1) = (tx * side, ty * side, (tx + 1) * side, (ty + 1) * side);
    let half = side >> 1;
    let corners = [
        corner_point([nw, n, w, t], x0, y0, band),
        corner_point([n, ne, t, e], x1, y0, band),
        corner_point([w, t, sw, s], x0, y1, band),
        corner_point([t, e, s, se], x1, y1, band),
    ];
    let [c_nw, c_ne, c_sw, c_se] = corners;
    // A column edge's midpoint on its row of tiles' middle row; a row edge's on its column of tiles' middle column.
    let mc = |a: &Field, b: &Field, ge: i64, gy: i64| (mid_across_columns(a, b, ge, gy, band), gy);
    let mr = |a: &Field, b: &Field, ge: i64, gx: i64| (gx, mid_across_rows(a, b, ge, gx, band));
    let (m_wt, m_te) = (mc(w, t, x0, y0 + half), mc(t, e, x1, y0 + half));
    let (m_nt, m_ts) = (mr(n, t, y0, x0 + half), mr(t, s, y1, x0 + half));
    let halves: [HalfCut<'_>; 16] = [
        (nw, n, x0, mc(nw, n, x0, y0 - half), c_nw, true),
        (n, ne, x1, mc(n, ne, x1, y0 - half), c_ne, true),
        (w, t, x0, c_nw, m_wt, true),
        (w, t, x0, m_wt, c_sw, true),
        (t, e, x1, c_ne, m_te, true),
        (t, e, x1, m_te, c_se, true),
        (sw, s, x0, c_sw, mc(sw, s, x0, y1 + half), true),
        (s, se, x1, c_se, mc(s, se, x1, y1 + half), true),
        (nw, w, y0, mr(nw, w, y0, x0 - half), c_nw, false),
        (n, t, y0, c_nw, m_nt, false),
        (n, t, y0, m_nt, c_ne, false),
        (ne, e, y0, c_ne, mr(ne, e, y0, x1 + half), false),
        (w, sw, y1, mr(w, sw, y1, x0 - half), c_sw, false),
        (t, s, y1, c_sw, m_ts, false),
        (t, s, y1, m_ts, c_se, false),
        (e, se, y1, c_se, mr(e, se, y1, x1 + half), false),
    ];
    let l: Vec<CutLine> = halves
        .iter()
        .map(|&(a, b, ge, from, to, cols)| {
            if cols {
                col_cut_half(a, b, ge, band, from, to)
            } else {
                row_cut_half(a, b, ge, band, from, to)
            }
        })
        .collect::<Option<_>>()?;
    let h = |k: usize| l[k].clone();
    Some(TileCuts {
        corners,
        cols: [
            [h(0), h(1)],
            [joined(h(2), &l[3]), joined(h(4), &l[5])],
            [h(6), h(7)],
        ],
        rows: [
            [h(8), joined(h(9), &l[10]), h(11)],
            [h(12), joined(h(13), &l[14]), h(15)],
        ],
    })
}

/// ★ THE FINISHED TILE (ruling B85): the square of tile `(tx, ty)` (`side` cells, cells `[tx·side, (tx + 1)·side)` by
/// rows likewise) drawn from the nine raw rasters around it (as [`tile_cuts`]'s `block`, each over its own tile and at
/// least `band + BLEND` cells beyond) along the tile's cuts, every corner joined columns first ([`draw`]). `None` where a
/// cut cannot be joined.
#[must_use]
pub fn finished_tile(block: [&Field; 9], tx: i64, ty: i64, side: i64, band: i64) -> Option<Field> {
    let cuts = tile_cuts(block, tx, ty, side, band)?;
    let square = [tx * side, ty * side, (tx + 1) * side, (ty + 1) * side];
    Some(draw(
        block,
        &cuts,
        square,
        side,
        [Meet::Four { rows_first: false }; 4],
    ))
}

/// ★ HOW THE CUTS MEET AT ONE OF A TILE'S CORNERS (B2's face-edge cut, ruling B86 item 9 (1)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Meet {
    /// Four tiles: joined columns first in the chart the four share by rule (the face of the lowest of them), so a tile
    /// whose own chart turns that chart's columns into its rows joins rows first — the same heights from every side.
    Four { rows_first: bool },
    /// Three tiles — a cube corner, where the diagonal tile does not exist. The tile's weight is the product of its sides'
    /// weights across its two cuts; each neighbour's, its side's weight across its cut with the tile times its side's
    /// weight against the other neighbour, read across the bisector of the tile's own quarter (the two neighbours' cut
    /// runs along the cube's third edge, which no chart of this tile holds). Each of the three tiles reads the same rule
    /// in its own quarter, and one cut alone gives the two-tile blend to the integer.
    Three,
}

/// One quarter of a tile: the four rasters around its corner (NW, NE, SW, SE of it), the corner point, the four cuts
/// from it (the column cuts north and south of it, the row cuts west and east), the tile's corner (between cells) and how
/// the cuts meet there.
struct Quarter<'a> {
    slots: [&'a Field; 4],
    point: (i64, i64),
    cuts: [&'a CutLine; 4],
    corner: (i64, i64),
    meet: Meet,
    /// The tile's own place among the four (`3 − ` the corner's index).
    me: usize,
}

impl Quarter<'_> {
    fn value(&self, x: i64, y: i64) -> i32 {
        let [a, b, c, d] = self.slots.map(|f| f.at(x, y));
        let [nc, sc, wr, er] = self.cuts;
        match self.meet {
            Meet::Four { rows_first: false } => {
                let row = if x < self.point.0 { wr.at(x) } else { er.at(x) };
                blend(
                    blend(a, b, x - nc.at(y)),
                    blend(c, d, x - sc.at(y)),
                    y - row,
                )
            }
            Meet::Four { rows_first: true } => {
                let col = if y < self.point.1 { nc.at(y) } else { sc.at(y) };
                blend(
                    blend(a, c, y - wr.at(x)),
                    blend(b, d, y - er.at(x)),
                    x - col,
                )
            }
            Meet::Three => self.three([a, b, c, d], x, y),
        }
    }

    /// The three-tile rule (see [`Meet::Three`]): the tile `me`, its neighbour in its row `me ^ 1` and in its column
    /// `me ^ 2`.
    fn three(&self, z: [i32; 4], x: i64, y: i64) -> i32 {
        let me = self.me;
        let (east, south) = (me & 1 == 1, me & 2 == 2);
        let [nc, sc, wr, er] = self.cuts;
        let col_cut = if south { sc } else { nc };
        let row_cut = if east { er } else { wr };
        let side = |d: i64, plus: bool| blend_weight(if plus { d } else { -d });
        // The tile's weight across each of its cuts, and its depths from the two sides that meet at the corner.
        let over_col = side(x - col_cut.at(y), east);
        let over_row = side(y - row_cut.at(x), south);
        let u = if east {
            x - self.corner.0
        } else {
            self.corner.0 - 1 - x
        };
        let v = if south {
            y - self.corner.1
        } else {
            self.corner.1 - 1 - y
        };
        // The column neighbour's against the row neighbour's: the row neighbour's side is the one nearer its edge.
        let row_over_col = blend_weight(u - v);
        let w = [
            (me, over_col * over_row),
            (me ^ 1, (ONE - over_col) * (ONE - row_over_col)),
            (me ^ 2, (ONE - over_row) * row_over_col),
        ];
        let (mut sum, mut total) = (0i64, 0i64);
        for (k, wk) in w {
            if z[k] != MISSING && wk > 0 {
                sum += wk * i64::from(z[k]);
                total += wk;
            }
        }
        if total == 0 {
            return MISSING;
        }
        (sum + (total >> 1)).div_euclid(total) as i32
    }
}

/// ★ THE DRAWING of a tile's square from its nine rasters and its cuts, all in one chart (B85; B2's face-edge cut): the
/// tile's own square is `square` (`[x0, y0, x1, y1)`, short of `side` where a face ends inside the tile); the drawn
/// field is `side` cells square from `(x0, y0)`. Each cell is drawn by its quarter's corner: the four rasters around that
/// corner joined across the corner's four cuts as `meets` says (NW, NE, SW, SE). Where the face ends inside the tile the
/// cells past it are drawn by the same rule (they are never read: the face's partner holds those cells).
#[must_use]
pub fn draw(
    block: [&Field; 9],
    cuts: &TileCuts,
    square: [i64; 4],
    side: i64,
    meets: [Meet; 4],
) -> Field {
    draw_over(
        block,
        cuts,
        square,
        [square[0], square[1], square[0] + side, square[1] + side],
        meets,
    )
}

/// [`draw`] over any rectangle `[x0, y0, x1, y1)` of the chart (a cell outside the tile's square by the quarter its
/// place falls in — the tests read a tile's rule across its own sides).
#[must_use]
pub fn draw_over(
    block: [&Field; 9],
    cuts: &TileCuts,
    square: [i64; 4],
    rect: [i64; 4],
    meets: [Meet; 4],
) -> Field {
    let [x0, y0, x1, y1] = square;
    let (mx, my) = ((x0 + x1) >> 1, (y0 + y1) >> 1);
    let [nw, n, ne, w, t, e, sw, s, se] = block;
    let [[v_nwn, v_nne], [v_wt, v_te], [v_sws, v_sse]] = &cuts.cols;
    let [[h_nww, h_nt, h_nee], [h_wsw, h_ts, h_ese]] = &cuts.rows;
    let [c_nw, c_ne, c_sw, c_se] = cuts.corners;
    let [m_nw, m_ne, m_sw, m_se] = meets;
    let quarters = [
        Quarter {
            slots: [nw, n, w, t],
            point: c_nw,
            cuts: [v_nwn, v_wt, h_nww, h_nt],
            corner: (x0, y0),
            meet: m_nw,
            me: 3,
        },
        Quarter {
            slots: [n, ne, t, e],
            point: c_ne,
            cuts: [v_nne, v_te, h_nt, h_nee],
            corner: (x1, y0),
            meet: m_ne,
            me: 2,
        },
        Quarter {
            slots: [w, t, sw, s],
            point: c_sw,
            cuts: [v_wt, v_sws, h_wsw, h_ts],
            corner: (x0, y1),
            meet: m_sw,
            me: 1,
        },
        Quarter {
            slots: [t, e, s, se],
            point: c_se,
            cuts: [v_te, v_sse, h_ts, h_ese],
            corner: (x1, y1),
            meet: m_se,
            me: 0,
        },
    ];
    let [rx0, ry0, rx1, ry1] = rect;
    let (w, h) = ((rx1 - rx0).max(0) as usize, (ry1 - ry0).max(0) as usize);
    let mut v = Vec::with_capacity(w * h);
    for y in ry0..ry1 {
        for x in rx0..rx1 {
            v.push(quarters[usize::from(y >= my) * 2 + usize::from(x >= mx)].value(x, y));
        }
    }
    Field {
        gx0: rx0,
        gy0: ry0,
        w,
        h,
        v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(gx0: i64, gy0: i64, w: usize, h: usize, f: impl Fn(i64, i64) -> i32) -> Field {
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                v.push(f(gx0 + x, gy0 + y));
            }
        }
        Field { gx0, gy0, w, h, v }
    }

    #[test]
    fn the_blend_runs_from_the_west_to_the_east_by_the_exact_smoothstep() {
        assert_eq!(blend_weight(-8), 0);
        assert_eq!(blend_weight(-100), 0);
        assert_eq!(blend_weight(0), 2_048);
        assert_eq!(blend_weight(8), 4_096);
        assert_eq!(blend(100, 200, 0), 150);
        assert_eq!(blend(100, 200, -9), 100);
        assert_eq!(blend(MISSING, 7, 0), 7);
        assert_eq!(blend(7, MISSING, 0), 7);
    }

    #[test]
    fn two_equal_fields_stitch_into_themselves_and_the_cut_finds_where_they_agree() {
        let surface = |x: i64, y: i64| (3 * x + 5 * y) as i32;
        let west = field(0, 0, 40, 20, surface);
        let east = field(24, 0, 40, 20, surface);
        let (s, path) = stitch_columns(&west, &east, 32, 8);
        assert_eq!(path.len(), 20);
        for y in 0..20 {
            for x in 0..64 {
                assert_eq!(s.at(x, y), surface(x, y));
            }
        }
        // Where the two differ by a step except along column 30, the cut follows column 30.
        let east2 = field(24, 0, 40, 20, |x, y| {
            if x == 30 {
                surface(x, y)
            } else {
                surface(x, y) + 400
            }
        });
        let (_, path2) = cut(&west, &east2, 32, 8);
        assert!(path2.iter().all(|&c| c == 30), "{path2:?}");
    }

    #[test]
    fn a_cut_across_rows_is_the_column_cut_of_the_swapped_fields_and_disjoint_fields_share_no_row()
    {
        let surface = |x: i64, y: i64| (x * x - y) as i32;
        let north = field(0, 0, 10, 30, surface);
        let south = field(0, 20, 10, 30, surface);
        let (s, path) = stitch_rows(&north, &south, 25, 4);
        assert_eq!(path.len(), 10);
        assert_eq!(s.at(3, 24), surface(3, 24));
        assert_eq!(s.swapped().swapped(), s);
        let far = field(0, 100, 10, 5, surface);
        let (none, empty) = stitch_columns(&north, &far, 5, 2);
        assert!(empty.is_empty());
        assert_eq!(none.h, 0);
        // A cell outside both fields is missing.
        assert_eq!(north.at(-1, 0), MISSING);
        assert_eq!(north.at(0, 30), MISSING);
    }

    #[test]
    fn a_missing_cell_costs_the_cut_and_the_pyramid_refuses_it() {
        let west = field(0, 0, 12, 4, |x, _| if x == 5 { MISSING } else { 0 });
        let east = field(0, 0, 12, 4, |_, _| 0);
        let (_, path) = cut(&west, &east, 5, 2);
        assert!(path.iter().all(|&c| c != 5));
        assert_eq!(mean_levels(&west, 0, 0, 8, 2), None);
    }

    #[test]
    fn a_half_cut_joins_its_ends_along_the_cheapest_cells_and_refuses_ends_it_cannot_join() {
        // Column 3 is free: the path walks to it, stays, and leaves it only to reach its end.
        let groove = |_: i64, c: i64| if c == 3 { 0 } else { 10 };
        assert_eq!(
            half_path(&groove, (0, 3), (10, 3), 0, 6)
                .expect("joins")
                .path,
            vec![3; 11]
        );
        assert_eq!(
            half_path(&groove, (0, 0), (6, 5), 0, 6)
                .expect("joins")
                .path,
            vec![0, 1, 2, 3, 3, 4, 5]
        );
        // Equal costs: every tie goes to the lower column.
        let flat = |_: i64, _: i64| 0;
        assert_eq!(
            half_path(&flat, (0, 2), (4, 2), 0, 4).expect("joins").path,
            vec![2, 1, 0, 1, 2]
        );
        // Ends it cannot join: backwards, too far across, outside the band.
        assert_eq!(half_path(&flat, (5, 0), (4, 0), 0, 4), None);
        assert_eq!(half_path(&flat, (0, 0), (2, 3), 0, 4), None);
        assert_eq!(half_path(&flat, (0, -1), (3, 0), 0, 4), None);
        assert_eq!(half_path(&flat, (0, 0), (5, 5), 0, 4), None);
        // A line holds its ends beyond them.
        let line = CutLine {
            a0: 5,
            path: vec![1, 2, 3],
        };
        assert_eq!([line.at(0), line.at(6), line.at(99)], [1, 2, 3]);
    }

    #[test]
    fn the_corner_point_is_where_the_four_agree_and_a_midpoint_where_the_two_agree() {
        let d = |x: i64, y: i64| ((x - 7) * (x - 7) + (y - 9) * (y - 9)) as i32;
        let four: Vec<Field> = (1..=4)
            .map(|k| field(0, 0, 20, 20, move |x, y| k * d(x, y)))
            .collect();
        assert_eq!(
            corner_point([&four[0], &four[1], &four[2], &four[3]], 10, 10, 4),
            (7, 9)
        );
        // Everywhere equal: the lowest row, then the lowest column.
        assert_eq!(
            corner_point([&four[0], &four[0], &four[0], &four[0]], 10, 10, 4),
            (6, 6)
        );
        // A tile that holds no cell west of column 8: the best of the cells all four hold.
        let east_only = field(8, 0, 12, 20, |x, y| 5 * d(x, y));
        assert_eq!(
            corner_point([&four[0], &four[1], &four[2], &east_only], 10, 10, 4),
            (8, 9)
        );
        // The midpoints: on row 9 the two agree at column 7; on column 7 at row 9; equal fields tie west and north.
        assert_eq!(mid_across_columns(&four[0], &four[1], 10, 9, 4), 7);
        assert_eq!(mid_across_rows(&four[0], &four[1], 10, 7, 4), 9);
        assert_eq!(mid_across_columns(&four[0], &four[0], 10, 9, 4), 6);
        assert_eq!(mid_across_rows(&four[0], &four[0], 10, 9, 4), 6);
        // No band: the edge itself.
        assert_eq!(mid_across_columns(&four[0], &four[1], 10, 9, -1), 10);
    }

    /// The ground every test tile shares.
    fn ground(x: i64, y: i64) -> i32 {
        (3 * x - 2 * y + 7) as i32
    }

    /// The squared distance to the nearest of `agree`.
    fn near2(x: i64, y: i64, agree: &[(i64, i64)]) -> i32 {
        agree
            .iter()
            .map(|&(ax, ay)| (x - ax) * (x - ax) + (y - ay) * (y - ay))
            .min()
            .unwrap_or(0) as i32
    }

    /// The raw raster of tile `(i, j)` (a `side` grid, `margin` cells beyond the tile): the shared ground plus the tile's
    /// own offset times the squared distance to the nearest of `agree` — so all tiles agree exactly there and nowhere else.
    fn raw(i: i64, j: i64, side: i64, margin: i64, agree: &[(i64, i64)]) -> Field {
        let off = (i + 3 * j + 1) as i32;
        let w = (side + 2 * margin) as usize;
        field(i * side - margin, j * side - margin, w, w, |x, y| {
            ground(x, y) + off * near2(x, y, agree)
        })
    }

    /// The nine rasters around tile `(tx, ty)` of a grid `n` tiles wide, row by row.
    fn block(grid: &[Field], n: i64, tx: i64, ty: i64) -> [&Field; 9] {
        let at = |dx: i64, dy: i64| &grid[((ty + dy) * n + tx + dx) as usize];
        [
            at(-1, -1),
            at(0, -1),
            at(1, -1),
            at(-1, 0),
            at(0, 0),
            at(1, 0),
            at(-1, 1),
            at(0, 1),
            at(1, 1),
        ]
    }

    #[test]
    fn a_tile_drawn_from_nine_equal_rasters_is_their_own_ground() {
        let (side, band) = (48, 6);
        let grid: Vec<Field> = (0..3)
            .flat_map(|j| (0..3).map(move |i| raw(i, j, side, 16, &[])))
            .collect();
        let tile = finished_tile(block(&grid, 3, 1, 1), 1, 1, side, band).expect("joins");
        assert_eq!(tile, field(side, side, 48, 48, ground));
        // Every tie went to the lowest row and column: the corner points sit `band` cells north-west of the corners.
        let cuts = tile_cuts(block(&grid, 3, 1, 1), 1, 1, side, band).expect("joins");
        assert_eq!(cuts.corners, [(42, 42), (90, 42), (42, 90), (90, 90)]);
    }

    #[test]
    fn two_neighbours_compute_the_same_cut_for_their_edge_and_their_corners() {
        // A 5 × 5 grid of tiles; at each tile corner the tiles agree at one point moved by a table (within the band).
        let (side, band, n) = (48, 6, 5);
        let moves = [(3, -2), (-4, 1), (2, 4), (-1, -3)];
        let agree: Vec<(i64, i64)> = (0..=n)
            .flat_map(|cj| (0..=n).map(move |ci| (ci, cj)))
            .map(|(ci, cj)| {
                let (mx, my) = moves[((ci + 2 * cj) & 3) as usize];
                (ci * side + mx, cj * side + my)
            })
            .collect();
        let grid: Vec<Field> = (0..n)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| raw(i, j, side, 16, &agree))
            .collect();
        let cuts = |tx: i64, ty: i64| {
            tile_cuts(block(&grid, n, tx, ty), tx, ty, side, band).expect("joins")
        };
        let (t, east, south, north, west) =
            (cuts(2, 2), cuts(3, 2), cuts(2, 3), cuts(2, 1), cuts(1, 2));
        // The corner points are the agreement points, and each is the same from every tile that meets there.
        assert_eq!(t.corners, [(98, 100), (143, 93), (99, 142), (140, 145)]);
        assert_eq!(
            [east.corners[0], east.corners[2]],
            [t.corners[1], t.corners[3]]
        );
        assert_eq!(
            [south.corners[0], south.corners[1]],
            [t.corners[2], t.corners[3]]
        );
        assert_eq!(
            [north.corners[2], north.corners[3]],
            [t.corners[0], t.corners[1]]
        );
        // A shared edge's whole cut is the same line from both sides.
        assert_eq!(t.cols[1][1], east.cols[1][0]);
        assert_eq!(t.rows[1][1], south.rows[0][1]);
        assert_eq!(t.rows[0][1], north.rows[1][1]);
        assert_eq!(t.cols[1][0], west.cols[1][1]);
        // A neighbour's half is the matching part of that neighbour's own whole cut.
        let starts = |half: &CutLine, whole: &CutLine| {
            (half.a0, half.path.as_slice()) == (whole.a0, &whole.path[..half.path.len()])
        };
        let ends = |half: &CutLine, whole: &CutLine| {
            let k = whole.path.len() - half.path.len();
            (half.a0, half.path.as_slice()) == (whole.a0 + k as i64, &whole.path[k..])
        };
        assert_eq!(
            [
                starts(&t.cols[2][0], &south.cols[1][0]),
                starts(&t.cols[2][1], &south.cols[1][1])
            ],
            [true, true],
            "the row below's column cuts"
        );
        assert_eq!(
            [
                ends(&t.cols[0][0], &north.cols[1][0]),
                ends(&t.cols[0][1], &north.cols[1][1])
            ],
            [true, true],
            "the row above's"
        );
        assert_eq!(
            [
                starts(&t.rows[0][2], &east.rows[0][1]),
                starts(&t.rows[1][2], &east.rows[1][1])
            ],
            [true, true],
            "the east column's"
        );
        assert_eq!(
            [
                ends(&t.rows[0][0], &west.rows[0][1]),
                ends(&t.rows[1][0], &west.rows[1][1])
            ],
            [true, true],
            "the west column's"
        );
        // The drawn tile: its own raster far from every cut, the shared ground at the agreement points.
        let tile = finished_tile(block(&grid, n, 2, 2), 2, 2, side, band).expect("joins");
        assert_eq!(tile.at(120, 120), grid[12].at(120, 120));
        assert_eq!(
            [tile.at(98, 100), tile.at(99, 142)],
            [ground(98, 100), ground(99, 142)]
        );
    }

    /// THE B85 DRAWING (rows of tiles joined across their nearer column cut, then the nearer row cut), kept as the
    /// reference the drawing by quarters must meet to the integer.
    /// ★ THE FINISHED TILE (ruling B85): the square of tile `(tx, ty)` (`side` cells, cells `[tx·side, (tx + 1)·side)` by
    /// rows likewise) drawn from the nine raw rasters around it (as [`tile_cuts`]'s `block`, each over its own tile and at
    /// least `band + BLEND` cells beyond) along the tile's cuts. A cell's height is joined first within its row of tiles across
    /// that row's column cuts (the nearer one, the blend across it), then across the row cut nearer to it, the row cut over a
    /// column being its own column's line between the tile's corner points and the neighbouring columns' halves beyond them.
    /// `None` where a cut cannot be joined.
    fn by_rows(block: [&Field; 9], tx: i64, ty: i64, side: i64, band: i64) -> Option<Field> {
        let cuts = tile_cuts(block, tx, ty, side, band)?;
        let [nw, n, ne, w, t, e, sw, s, se] = block;
        let (x0, y0, x1, y1) = (tx * side, ty * side, (tx + 1) * side, (ty + 1) * side);
        let [c_nw, c_ne, c_sw, c_se] = cuts.corners;
        let [[v_nwn, v_nne], [v_wt, v_te], [v_sws, v_sse]] = &cuts.cols;
        let [[h_nww, h_nt, h_nee], [h_wsw, h_ts, h_ese]] = &cuts.rows;
        let north_at = |x: i64| {
            if x < c_nw.0 {
                h_nww.at(x)
            } else if x > c_ne.0 {
                h_nee.at(x)
            } else {
                h_nt.at(x)
            }
        };
        let south_at = |x: i64| {
            if x < c_sw.0 {
                h_wsw.at(x)
            } else if x > c_se.0 {
                h_ese.at(x)
            } else {
                h_ts.at(x)
            }
        };
        // A row of tiles joined across its column cuts at `(x, y)`: west, middle and east rasters and the two cuts.
        let row_join =
            |wf: &Field, mf: &Field, ef: &Field, vw: &CutLine, ve: &CutLine, x: i64, y: i64| {
                let (cw, ce) = (vw.at(y), ve.at(y));
                if x - cw < ce - x {
                    blend(wf.at(x, y), mf.at(x, y), x - cw)
                } else {
                    blend(mf.at(x, y), ef.at(x, y), x - ce)
                }
            };
        let mut v = Vec::with_capacity((side * side) as usize);
        for y in y0..y1 {
            for x in x0..x1 {
                let (hn, hs) = (north_at(x), south_at(x));
                let mid = row_join(w, t, e, v_wt, v_te, x, y);
                let z = if y - hn < hs - y {
                    if y - hn > BLEND {
                        mid
                    } else {
                        blend(row_join(nw, n, ne, v_nwn, v_nne, x, y), mid, y - hn)
                    }
                } else if hs - y > BLEND {
                    mid
                } else {
                    blend(mid, row_join(sw, s, se, v_sws, v_sse, x, y), y - hs)
                };
                v.push(z);
            }
        }
        Some(Field {
            gx0: x0,
            gy0: y0,
            w: side as usize,
            h: side as usize,
            v,
        })
    }

    #[test]
    fn the_drawing_by_quarters_is_the_b85_drawing_to_the_integer() {
        // The 5 × 5 grid of moved agreement points: every inner tile, every cell.
        let (side, band, n) = (48, 6, 5);
        let moves = [(3, -2), (-4, 1), (2, 4), (-1, -3)];
        let agree: Vec<(i64, i64)> = (0..=n)
            .flat_map(|cj| (0..=n).map(move |ci| (ci, cj)))
            .map(|(ci, cj)| {
                let (mx, my) = moves[((ci + 2 * cj) & 3) as usize];
                (ci * side + mx, cj * side + my)
            })
            .collect();
        let grid: Vec<Field> = (0..n)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| raw(i, j, side, 16, &agree))
            .collect();
        for ty in 1..n - 1 {
            for tx in 1..n - 1 {
                let b = block(&grid, n, tx, ty);
                assert_eq!(
                    finished_tile(b, tx, ty, side, band),
                    by_rows(b, tx, ty, side, band),
                    "tile ({tx}, {ty})"
                );
            }
        }
        // Rows first is columns first on the swapped rasters and cuts.
        let b = block(&grid, n, 2, 2);
        let cuts = tile_cuts(b, 2, 2, side, band).expect("joins");
        let square = [96, 96, 144, 144];
        let rows_first = draw(b, &cuts, square, side, [Meet::Four { rows_first: true }; 4]);
        let swapped: Vec<Field> = b.iter().map(|f| f.swapped()).collect();
        let sb = [0, 3, 6, 1, 4, 7, 2, 5, 8].map(|k| &swapped[k]);
        let [[v_nwn, v_nne], [v_wt, v_te], [v_sws, v_sse]] = cuts.cols.clone();
        let [[h_nww, h_nt, h_nee], [h_wsw, h_ts, h_ese]] = cuts.rows.clone();
        let sc = TileCuts {
            corners: cuts.corners.map(|(x, y)| (y, x)),
            cols: [[h_nww, h_wsw], [h_nt, h_ts], [h_nee, h_ese]],
            rows: [[v_nwn, v_wt, v_sws], [v_nne, v_te, v_sse]],
        };
        let sc = TileCuts {
            corners: [sc.corners[0], sc.corners[2], sc.corners[1], sc.corners[3]],
            ..sc
        };
        let columns_first = draw(sb, &sc, square, side, [Meet::Four { rows_first: false }; 4]);
        assert_eq!(rows_first, columns_first.swapped());
    }

    #[test]
    fn three_tiles_at_a_cube_corner_blend_as_two_away_from_it_and_stay_between_the_three_near_it() {
        // The north-west tile of the block does not exist: the corner where the tile meets its west and north neighbours
        // joins three tiles.
        let (side, band) = (48, 6);
        let grid: Vec<Field> = (0..3)
            .flat_map(|j| (0..3).map(move |i| (i, j)))
            .map(|(i, j)| raw(i, j, side, 16, &[(50, 47), (93, 95)]))
            .collect();
        let empty = Field {
            gx0: 0,
            gy0: 0,
            w: 0,
            h: 0,
            v: Vec::new(),
        };
        let mut b = block(&grid, 3, 1, 1);
        let cuts = tile_cuts(b, 1, 1, side, band).expect("joins");
        b[0] = &empty;
        let square = [48, 48, 96, 96];
        let four = draw(
            b,
            &cuts,
            square,
            side,
            [Meet::Four { rows_first: false }; 4],
        );
        let three = draw(
            b,
            &cuts,
            square,
            side,
            [
                Meet::Three,
                Meet::Four { rows_first: false },
                Meet::Four { rows_first: false },
                Meet::Four { rows_first: false },
            ],
        );
        let (mut far, mut near) = (0, 0);
        let clear = band + BLEND;
        for y in 48..96 {
            for x in 48..96 {
                let (u, v) = (x - 48, y - 48);
                if u.max(v) >= 24 {
                    // Another quarter: the corner's rule does not reach it.
                    assert_eq!(three.at(x, y), four.at(x, y), "({x}, {y})");
                } else if v - u >= BLEND && v >= clear {
                    // The west cut alone within the blend: the two-tile blend, to the integer.
                    assert_eq!(
                        three.at(x, y),
                        blend(
                            grid[3].at(x, y),
                            grid[4].at(x, y),
                            x - cuts.cols[1][0].at(y)
                        ),
                        "({x}, {y})"
                    );
                    far += 1;
                } else if u - v >= BLEND && u >= clear {
                    // The north cut alone.
                    assert_eq!(
                        three.at(x, y),
                        blend(
                            grid[1].at(x, y),
                            grid[4].at(x, y),
                            y - cuts.rows[0][1].at(x)
                        ),
                        "({x}, {y})"
                    );
                    far += 1;
                } else {
                    let z = [grid[1].at(x, y), grid[3].at(x, y), grid[4].at(x, y)];
                    let (lo, hi) = (
                        z.iter().min().copied().unwrap_or(0),
                        z.iter().max().copied().unwrap_or(0),
                    );
                    assert!((lo..=hi).contains(&three.at(x, y)), "({x}, {y})");
                    near += 1;
                }
            }
        }
        assert_eq!((far > 0, near > 0), (true, true));
    }

    #[test]
    fn a_band_too_wide_for_the_tile_draws_nothing() {
        // The north-west corner point lies 5 cells in, the west edge's midpoint 5 cells out: one row apart, ten columns.
        let (side, band) = (12, 6);
        let agree = [(17, 17), (7, 19)];
        let grid: Vec<Field> = (0..3)
            .flat_map(|j| (0..3).map(move |i| (i, j)))
            .map(|(i, j)| raw(i, j, side, 16, &agree))
            .collect();
        assert_eq!(tile_cuts(block(&grid, 3, 1, 1), 1, 1, side, band), None);
        assert_eq!(finished_tile(block(&grid, 3, 1, 1), 1, 1, side, band), None);
    }

    #[test]
    fn the_pyramid_takes_each_level_from_the_rasters_own_sums() {
        let f = field(0, 0, 8, 8, |x, y| (x + 8 * y) as i32);
        let levels = mean_levels(&f, 0, 0, 8, 3).expect("halves three times");
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].len(), 16);
        // The 2 × 2 mean of 0, 1, 8, 9 is 4.5, rounded half up to 5; the whole square's mean 31.5 to 32.
        assert_eq!(levels[0][0], 5);
        assert_eq!(levels[2], vec![32]);
        assert_eq!(mean_levels(&f, 0, 0, 6, 2), None);
    }
}
