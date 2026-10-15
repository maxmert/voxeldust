//! ★ ONE LEVEL UNDER THE NEXT COARSER LEVEL (rulings B96, B97; ruling B81 item 4: ONE generic tile machinery) — the solve
//! the 16 m level runs under the 128 m flow, written once for every level that refines a coarser one: the points of a
//! square domain at the level's rung, their planar graph, the coarser flow's links that cross the domain's rim (exits and
//! inflows), each point's rate and rain read from the coarser points by a Gaussian of the logarithms, the steady form of
//! the joint law under the threshold with the half step and the settle law, and the raster in B96's form (today's drawn
//! coarser ground plus the law's detail finer than the coarser cell).
//!
//! The 16 m level calls it over the 128 m flow ([`super::l2`]); the 4 m bench level over the 16 m flow record
//! ([`super::l3`]). The arithmetic is the 16 m solve's own, moved here unchanged, so a 16 m tile's bytes do not move.
//!
//! **Example.** A 4 m tile beside hull 19: 768 × 768 cells of 4 m with its buffer, about 560 000 points, the 16 m flow's
//! links crossing its rim as its exits, the steady form, then 640 × 640 heights — today's 16 m curve plus the law's
//! detail under 16 m.

use super::delaunay::{Dir, Graph, NONE, triangulate_on_face};
use super::geometry::{Geometry, geometry, locate};
use super::points::{CellId, Level, chart_cell, chart_of, domain_points};
use super::read::{FineBook, FineZ};
use super::tree::{accumulate, tree};
use crate::gf::Gf;
use crate::solve::{JOINT_K_LP, JOINT_N};
use crate::units::{LENGTH_BITS, STEPS_PER_M};
use vd_seed::bend::Face;

/// The drawn ground of `book` at the centre of `cell` (a cell of `rung`) in metres over the sea; `None` where its tile is
/// not held.
pub(crate) fn drawn(book: &FineBook, cell: &CellId, rung: u8, sea: Gf) -> Option<Gf> {
    let face = Face::from_index(cell.face)?;
    match book.z(face, rung, rung, cell.i, cell.j) {
        FineZ::Held(z) => {
            Some(Gf::from_i64(z.raw()) / Gf::from_i64(STEPS_PER_M << LENGTH_BITS) - sea)
        }
        FineZ::Missing(_) | FineZ::Outside => None,
    }
}

/// A direction as metres on the sphere.
pub(crate) fn metres(d: &Dir, r_m: Gf) -> [Gf; 3] {
    let s = r_m / Gf::from_f64((1u64 << vd_recipe::bend::DIR_BITS) as f64);
    [
        Gf::from_i64(d[0].raw()) * s,
        Gf::from_i64(d[1].raw()) * s,
        Gf::from_i64(d[2].raw()) * s,
    ]
}

pub(crate) fn dist2(a: &[Gf; 3], b: &[Gf; 3]) -> Gf {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// The median of `v` (the lower of the two middles), `None` when empty.
pub(crate) fn median(mut v: Vec<Gf>) -> Option<Gf> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    (!v.is_empty()).then(|| v[(v.len() - 1) >> 1])
}

/// The joint law's slope for the rate `u` and the water `q` (`(U/(K·Q))^(1/n)`); unbounded (a large word) where no water.
pub(crate) fn river_slope(u: Gf, q: Gf) -> Gf {
    let k = Gf::from_f64(JOINT_K_LP);
    if q <= Gf::ZERO || u <= Gf::ZERO {
        return Gf::from_f64(f64::MAX);
    }
    (u / (k * q)).powf_fine(Gf::from_f64(1.0 / JOINT_N))
}

/// A grid of point indices by their cells on a chart rectangle (compressed rows): every point whose cell stands at
/// `(x, y)` of the rectangle `[x0, x0 + w) × [y0, y0 + h)`.
pub(crate) struct CellGrid {
    x0: i32,
    y0: i32,
    w: i32,
    h: i32,
    start: Vec<u32>,
    pts: Vec<u32>,
}

impl CellGrid {
    pub(crate) fn new(at: &[Option<(i32, i32)>], (x0, y0, w, h): (i32, i32, i32, i32)) -> CellGrid {
        let slot = |&(x, y): &(i32, i32)| {
            ((x >= x0) & (y >= y0) & (x < x0 + w) & (y < y0 + h))
                .then(|| ((y - y0) * w + (x - x0)) as usize)
        };
        let mut start = vec![0u32; (w * h) as usize + 1];
        for s in at.iter().flatten().filter_map(slot) {
            start[s + 1] += 1;
        }
        for i in 0..start.len() - 1 {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut pts = vec![0u32; start[start.len() - 1] as usize];
        for (p, a) in at.iter().enumerate() {
            if let Some(s) = a.as_ref().and_then(slot) {
                pts[fill[s] as usize] = p as u32;
                fill[s] += 1;
            }
        }
        CellGrid {
            x0,
            y0,
            w,
            h,
            start,
            pts,
        }
    }

    /// The points of the cells `[xa, xb] × [ya, yb]` (clipped to the rectangle).
    pub(crate) fn near(
        &self,
        (xa, xb): (i32, i32),
        (ya, yb): (i32, i32),
    ) -> impl Iterator<Item = u32> + '_ {
        let (xa, xb) = (xa.max(self.x0), xb.min(self.x0 + self.w - 1));
        let (ya, yb) = (ya.max(self.y0), yb.min(self.y0 + self.h - 1));
        (ya..=yb).flat_map(move |y| {
            (xa..=xb).flat_map(move |x| {
                let s = ((y - self.y0) * self.w + (x - self.x0)) as usize;
                self.pts[self.start[s] as usize..self.start[s + 1] as usize]
                    .iter()
                    .copied()
            })
        })
    }
}

/// ★ THE COARSER LEVEL'S FLOW a finer level reads: each coarser point's cell (at the coarser rung), whether it is an
/// outlet, its drawn height, its receiver, the water through it, its rate and its rain, its place in metres; the chart
/// rectangle its points stand in; its rung, its cells a face edge and its spacing in metres.
pub(crate) struct Coarse<'a> {
    pub cells: &'a [CellId],
    pub outlet: &'a [bool],
    pub h: &'a [Gf],
    pub recv: &'a [u32],
    pub q: &'a [Gf],
    pub rate: &'a [Gf],
    pub rain: &'a [Gf],
    pub pos: &'a [[Gf; 3]],
    /// The coarser points' chart rectangle at the coarser rung: `(x0, y0, side)`.
    pub rect: (i32, i32, i32),
    pub n: i32,
    pub spacing: Gf,
}

/// ★ A FINER LEVEL'S DOMAIN, SET UP: its points, graph and geometry; each point's drawn coarser ground (an exit's the
/// coarser link's height), whether it is an outlet, its chart place; its rate, rain and water read from the coarser flow
/// (the rim's inflows added to the water); the nearest coarser point of each (its CELL), and the read's weights.
pub(crate) struct Setup {
    pub dirs: Vec<Dir>,
    pub cells: Vec<CellId>,
    pub graph: Graph,
    pub geom: Geometry,
    pub pos: Vec<[Gf; 3]>,
    pub base: Vec<Gf>,
    pub outlet: Vec<bool>,
    pub rate: Vec<Gf>,
    pub rain: Vec<Gf>,
    pub water: Vec<Gf>,
    pub cell_of: Vec<u32>,
    pub wstart: Vec<u32>,
    pub wcell: Vec<u32>,
    pub wval: Vec<Gf>,
}

/// The read's Gaussian over the coarser points, in coarser spacings, and its reach in widths (the 128 m level's own read).
pub(crate) const READ_SIGMA: f64 = 0.5;
pub(crate) const READ_REACH: f64 = 2.5;

/// ★ SET UP the finer level's square domain `(i0, j0, side)` (cells of `rung`, `n` a face edge) on `face` under the
/// coarser flow `c`, `down` rungs coarser: the points, the planar graph, the drawn coarser ground by `drawn_at`, the
/// read, the rim. `None` where a domain cannot be built or the drawn ground is not held.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn setup(
    seed: u64,
    r_m: Gf,
    face: Face,
    rung: u8,
    n: i32,
    down: u32,
    (i0, j0, side): (i32, i32, i32),
    c: &Coarse<'_>,
    drawn_at: &dyn Fn(&CellId) -> Option<Gf>,
) -> Option<Setup> {
    let level = Level::new(seed, rung, n as u32)?;
    let pts = domain_points(&level, face, i0, j0, side, side);
    let dirs: Vec<Dir> = pts.iter().map(|p| p.dir).collect();
    let cells: Vec<CellId> = pts.iter().map(|p| p.cell).collect();
    // The face's plane (`triangulate_on_face`): the words shifted so a face's edge spans about 2^28 units — about 8 000
    // between two neighbours.
    let plane_shift = (32 - (n as u32).leading_zeros()).saturating_add(12).min(40);
    let graph = triangulate_on_face(&dirs, face, 40 - plane_shift)?;
    let geom = geometry(&dirs, &graph, r_m);
    let np = dirs.len();
    let pos: Vec<[Gf; 3]> = dirs.iter().map(|d| metres(d, r_m)).collect();
    let mut base = Vec::with_capacity(np);
    for cell in &cells {
        base.push(drawn_at(cell)?);
    }
    let mut outlet: Vec<bool> = base.iter().map(|&b| b <= Gf::ZERO).collect();
    // Each point's place on the chart, and each coarser point's in the same chart (its cell's middle): where a link
    // crosses the domain's rim, and which cells a search reads.
    let at4: Vec<Option<(i32, i32)>> = cells.iter().map(|&cl| chart_of(face, cl, n)).collect();
    let at7: Vec<Option<(i32, i32)>> = c.cells.iter().map(|&cl| chart_of(face, cl, c.n)).collect();
    let mid = 1i32 << (down - 1);
    let in4 = |a: Option<(i32, i32)>| -> Option<(i32, i32)> {
        a.map(|(x, y)| ((x << down) + mid, (y << down) + mid))
    };
    let inside = |a: Option<(i32, i32)>| {
        in4(a).is_some_and(|(x, y)| (x >= i0) & (y >= j0) & (x < i0 + side) & (y < j0 + side))
    };
    let grid4 = CellGrid::new(&at4, (i0, j0, side, side));
    let (fi0, fj0, fside) = c.rect;
    let grid7 = CellGrid::new(&at7, (fi0, fj0, fside, fside));
    // The point nearest a place on the chart (its cell and the eight around; ties to the lower index).
    let nearest4 = |x: i32, y: i32, target: &[Gf; 3]| -> Option<usize> {
        grid4
            .near((x - 1, x + 1), (y - 1, y + 1))
            .map(|p| (dist2(&pos[p as usize], target), p))
            .min_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            })
            .map(|b| b.1 as usize)
    };
    // THE READ'S WEIGHTS AND EACH POINT'S CELL: the coarser points within the reach of each point, by the Gaussian of their
    // distance (σ half a coarser spacing); its cell the nearest (ties to the lower index).
    let sigma = c.spacing * Gf::from_f64(READ_SIGMA);
    let reach2 = (sigma * Gf::from_f64(READ_REACH)) * (sigma * Gf::from_f64(READ_REACH));
    let two_s2 = sigma * sigma * Gf::TWO;
    let (mut wstart, mut wcell, mut wval) = (Vec::with_capacity(np + 1), Vec::new(), Vec::new());
    let mut cell_of = vec![u32::MAX; np];
    for p in 0..np {
        wstart.push(wcell.len() as u32);
        let Some((x, y)) = at4[p] else {
            continue;
        };
        let (cx, cy) = (x >> down, y >> down);
        let first = wcell.len();
        let mut sum = Gf::ZERO;
        let mut best: Option<(Gf, u32)> = None;
        for m in grid7.near((cx - 2, cx + 2), (cy - 2, cy + 2)) {
            let d2 = dist2(&pos[p], &c.pos[m as usize]);
            if best.is_none_or(|b| (d2, m) < b) {
                best = Some((d2, m));
            }
            if d2 <= reach2 {
                let w = (-(d2 / two_s2)).exp_fine();
                wcell.push(m);
                wval.push(w);
                sum += w;
            }
        }
        if let Some(b) = best {
            cell_of[p] = b.1;
            if wcell.len() == first {
                wcell.push(b.1);
                wval.push(Gf::ONE);
                sum = Gf::ONE;
            }
        }
        for w in &mut wval[first..] {
            *w = *w / sum;
        }
    }
    wstart.push(wcell.len() as u32);
    // Each point's rate and rain by the same weights (the logarithm of the rate), its water the rain over its area.
    let ln_rate1: Vec<Gf> = c.rate.iter().map(|r| r.ln_fine()).collect();
    let mut rate = Vec::with_capacity(np);
    let mut rains = Vec::with_capacity(np);
    let mut water = Vec::with_capacity(np);
    for p in 0..np {
        let (mut l, mut rain) = (Gf::ZERO, Gf::ZERO);
        for w in wstart[p] as usize..wstart[p + 1] as usize {
            l += wval[w] * ln_rate1[wcell[w] as usize];
            rain += wval[w] * c.rain[wcell[w] as usize];
        }
        rate.push(if wstart[p + 1] > wstart[p] {
            l.exp_fine()
        } else {
            Gf::ZERO
        });
        rains.push(rain);
        water.push(rain * geom.area[p]);
    }
    // THE RIM: a coarser link leaving the domain is an exit at the point nearest its crossing, on the drawn ground along
    // the link; a link entering hands its water to the point nearest its crossing.
    let lerp = |a: Gf, b: Gf, t: Gf| a + (b - a) * t;
    let crossing = |ci: usize, r: usize| -> Option<(i32, i32, Gf)> {
        let (a, b) = (in4(at7[ci])?, in4(at7[r])?);
        // The parameter where the segment meets the rim, by bisection on the chart (forty halvings, exact in `Gf`).
        let (mut lo, mut hi) = (Gf::ZERO, Gf::ONE);
        let at = |t: Gf| {
            let x = Gf::from_i32(a.0) + Gf::from_i32(b.0 - a.0) * t;
            let y = Gf::from_i32(a.1) + Gf::from_i32(b.1 - a.1) * t;
            (
                x.floor().to_i64_floor() as i32,
                y.floor().to_i64_floor() as i32,
            )
        };
        let inside_xy =
            |(x, y): (i32, i32)| (x >= i0) & (y >= j0) & (x < i0 + side) & (y < j0 + side);
        let a_in = inside_xy(a);
        for _ in 0..40 {
            let m = (lo + hi) * Gf::HALF;
            if inside_xy(at(m)) == a_in {
                lo = m;
            } else {
                hi = m;
            }
        }
        let (x, y) = at(if a_in { lo } else { hi });
        Some((x, y, lo))
    };
    let mut exits = 0usize;
    for ci in 0..c.h.len() {
        let r = c.recv[ci];
        if r == NONE || c.outlet[ci] {
            continue;
        }
        let r = r as usize;
        let (c_in, r_in) = (inside(at7[ci]), inside(at7[r]));
        if c_in == r_in {
            continue;
        }
        let Some((x, y, t)) = crossing(ci, r) else {
            continue;
        };
        let foot = [
            lerp(c.pos[ci][0], c.pos[r][0], t),
            lerp(c.pos[ci][1], c.pos[r][1], t),
            lerp(c.pos[ci][2], c.pos[r][2], t),
        ];
        let Some(p) = nearest4(x, y, &foot) else {
            continue;
        };
        if c_in {
            let z = lerp(c.h[ci], c.h[r], t);
            base[p] = if outlet[p] { base[p].lesser(z) } else { z };
            outlet[p] = true;
            exits += 1;
        } else if !outlet[p] {
            water[p] += c.q[ci];
        }
    }
    if exits == 0 && !outlet.iter().any(|&o| o) {
        // No coarser water leaves and no sea: the rim's lowest point drains the domain.
        let rim = |p: usize| {
            at4[p].is_some_and(|(x, y)| {
                (x == i0) | (y == j0) | (x == i0 + side - 1) | (y == j0 + side - 1)
            })
        };
        if let Some(p) = (0..np).filter(|&p| rim(p)).min_by(|&a, &b| {
            base[a]
                .partial_cmp(&base[b])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        }) {
            outlet[p] = true;
        }
    }
    Some(Setup {
        dirs,
        cells,
        graph,
        geom,
        pos,
        base,
        outlet,
        rate,
        rain: rains,
        water,
        cell_of,
        wstart,
        wcell,
        wval,
    })
}

/// ★ THE STEADY FORM, the 128 m level's own law (ruling B97): each pass builds the drainage from the last pass's heights
/// (the first from the drawn coarser ground), brings each point the water its tree carries, and stands each point over
/// its receiver by the joint law's slope for its rate and that water, never steeper than `tan`; from the second pass each
/// point moves HALF way to that form (the 128 m level's half step, B92). The passes stop when no coarser cell's mean
/// (over its `ncell` coarser points) moves more than `settle` metres, within `bound`. Returns the heights and the passes.
pub(crate) fn steady(
    s: &Setup,
    ncell: usize,
    tan: Gf,
    settle: Gf,
    bound: usize,
) -> (Vec<Gf>, usize) {
    let cell_means = |h: &[Gf]| -> Vec<Gf> {
        let (mut sum, mut area) = (vec![Gf::ZERO; ncell], vec![Gf::ZERO; ncell]);
        for ((&c, &hp), &ap) in s.cell_of.iter().zip(h).zip(&s.geom.area) {
            if c != u32::MAX {
                sum[c as usize] += hp * ap;
                area[c as usize] += ap;
            }
        }
        (0..ncell)
            .map(|c| {
                if area[c] > Gf::ZERO {
                    sum[c] / area[c]
                } else {
                    Gf::ZERO
                }
            })
            .collect()
    };
    let mut h = s.base.clone();
    let mut last = cell_means(&h);
    let mut passes = 0usize;
    for pass in 1..=bound {
        passes = pass;
        let t = tree(&h, &s.outlet, &s.graph, &s.geom.len);
        let q = accumulate(&t, &s.water);
        for &c in &t.order {
            let c = c as usize;
            let r = t.recv[c];
            if s.outlet[c] || r == NONE {
                h[c] = s.base[c];
                continue;
            }
            let full = h[r as usize] + river_slope(s.rate[c], q[c]).lesser(tan) * t.link[c];
            h[c] = if pass == 1 {
                full
            } else {
                (h[c] + full) * Gf::HALF
            };
        }
        let now = cell_means(&h);
        let moved = now
            .iter()
            .zip(&last)
            .map(|(x, y)| (*x - *y).abs())
            .fold(Gf::ZERO, Gf::greater);
        last = now;
        if moved <= settle {
            break;
        }
    }
    (h, passes)
}

/// ★ THE RASTER IN B96's FORM over `(ri0, rj0, rside)` (cells of the level's rung, `n` a face edge, `down` rungs under the
/// coarser cell): the law's heights `h` read at each cell's centre by the walk's weights, beside the drawn coarser ground
/// by `drawn_at`; the law's raster less its own coarser version (the box means over each coarser cell, read back by the
/// same Catmull-Rom between the cells' centres) added to the drawn ground. Returns each cell's height in metres over the
/// sea (`None` off the faces), and the detail's sizes. `None` where a cell cannot be located or its drawn ground is not
/// held.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn raster(
    s: &Setup,
    h: &[Gf],
    face: Face,
    n: i32,
    down: u32,
    (ri0, rj0, rside): (i32, i32, i32),
    drawn_at: &dyn Fn(&CellId) -> Option<Gf>,
) -> Option<(Vec<Option<Gf>>, Vec<Gf>)> {
    let rs = rside as usize;
    let inv_n = vd_recipe::bend::inv_n_of(n as u32);
    let mut law: Vec<Option<Gf>> = Vec::with_capacity(rs * rs);
    let mut curve: Vec<Option<Gf>> = Vec::with_capacity(rs * rs);
    let mut row_from = 0u32;
    for bj in 0..rside {
        let mut from = row_from;
        for ai in 0..rside {
            let Some(c) = chart_cell(face, ri0 + ai, rj0 + bj, n) else {
                law.push(None);
                curve.push(None);
                continue;
            };
            let qd = vd_recipe::bend::direction(i32::from(c.face), c.i, c.j, inv_n);
            let (tri, w) = locate(&s.dirs, &s.graph, from, &qd)?;
            let v = s.graph.tris[tri as usize];
            law.push(Some(
                w[0] * h[v[0] as usize] + w[1] * h[v[1] as usize] + w[2] * h[v[2] as usize],
            ));
            curve.push(Some(drawn_at(&c)?));
            if ai == 0 {
                row_from = tri;
            }
            from = tri;
        }
    }
    // ★ THE DETAIL (ruling B96): today's drawn coarser ground plus the law's detail finer than the coarser cell — the
    // law's raster less its own coarser version, made exactly as today's curve is made from its samples (the box means
    // over each coarser cell, read back by the same Catmull-Rom between the cells' centres).
    let k = 1usize << down;
    let mb = rs >> down;
    let mut means = vec![Gf::ZERO; mb * mb];
    for by in 0..mb {
        for bx in 0..mb {
            let (mut sum, mut cnt) = (Gf::ZERO, 0i64);
            for y in by * k..(by + 1) * k {
                for x in bx * k..(bx + 1) * k {
                    if let Some(z) = law[y * rs + x] {
                        sum += z;
                        cnt += 1;
                    }
                }
            }
            means[by * mb + bx] = if cnt > 0 {
                sum / Gf::from_i64(cnt)
            } else {
                Gf::ZERO
            };
        }
    }
    // The Catmull-Rom weights at the fraction `t` (Catmull & Rom 1974, the kernel the fine read draws with).
    let cr = |t: Gf| -> [Gf; 4] {
        let (t2, t3) = (t * t, t * t * t);
        [
            (Gf::TWO * t2 - t3 - t) * Gf::HALF,
            (Gf::from_i64(3) * t3 - Gf::from_i64(5) * t2 + Gf::TWO) * Gf::HALF,
            (Gf::from_i64(4) * t2 - Gf::from_i64(3) * t3 + t) * Gf::HALF,
            (t3 - t2) * Gf::HALF,
        ]
    };
    // A cell's place among the coarser cells' centres: `(x + ½ − k/2) / k`, its whole part and its fraction.
    let place = |x: usize| -> (i64, Gf) {
        let u = (Gf::from_i64(x as i64) + Gf::HALF - Gf::from_i64((k >> 1) as i64))
            / Gf::from_i64(k as i64);
        let i = u.floor();
        (i.to_i64_floor(), u - i)
    };
    let clampm = |i: i64| i.clamp(0, mb as i64 - 1) as usize;
    let (mut out, mut detail) = (Vec::with_capacity(rs * rs), Vec::new());
    for y in 0..rs {
        let (iy, ty) = place(y);
        let wy = cr(ty);
        for x in 0..rs {
            let (Some(lz), Some(cz)) = (law[y * rs + x], curve[y * rs + x]) else {
                out.push(None);
                continue;
            };
            let (ix, tx) = place(x);
            let wx = cr(tx);
            let mut low = Gf::ZERO;
            for (dy, &wyv) in wy.iter().enumerate() {
                for (dx, &wxv) in wx.iter().enumerate() {
                    low += wyv
                        * wxv
                        * means[clampm(iy + dy as i64 - 1) * mb + clampm(ix + dx as i64 - 1)];
                }
            }
            let d = lz - low;
            detail.push(d.abs());
            out.push(Some(cz + d));
        }
    }
    detail.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    Some((out, detail))
}
