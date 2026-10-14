//! ★ THE 16 m LEVEL'S SOLVE (ruling B95; the way-in design revision 4 §3.2–§3.4, build step B3).
//!
//! A 16 m TILE is one macro node of a face; its DOMAIN adds a buffer of a quarter node each side (2 km on the home planet,
//! the measured seam rule). It REFINES the drawn 128 m ground — the finished 128 m tiles a host holds
//! ([`FineBook`]) — and never contradicts it:
//!
//! 1. **The 128 m flow.** Over the node and [`FLOW_NODES`] each side, the 128 m points stand at the drawn ground; their
//!    flow tree and water (the rain and the 8 km rivers entering, as the 128 m domain's) give each 128 m link its water,
//!    and the same joint law read backwards gives each node its RATE — `U = K·Q·S^n` over the node's links under the
//!    threshold, the median — the rate at which the drawn ground is in balance (ruling B95: on all land, where the
//!    macro world's present rise is nought on still land and would draw it flat).
//! 2. **The 128 m channels kept.** A 128 m link is a CHANNEL where the 16 m law says its water cuts: the joint law's slope
//!    `(U/(K·Q))^(1/n)` under the creep's steady slope at `ℓ = A/width` (Roering, Perron & Kirchner 2007); the 16 m points
//!    within three quarters of a spacing of a channel link are outlets on the drawn ground along it. A 128 m link that
//!    crosses the domain's rim is its exit (outward) or its inflow (inward), so two neighbours' valleys leave the same way.
//! 3. **The steady form** on the 16 m points (§3.2): each point over its receiver by the lower of the joint law's slope and
//!    the creep's (`D` [`CREEP_D_M2_YR`], `S_c` 1.25 — Roering, Kirchner & Dietrich 1999, the Oregon Coast Range), never
//!    steeper than `S_c`; the water and the area shared over the lower neighbours (Freeman 1991, `p` [`FREEMAN_P`]); the
//!    passes stop when no point changes its receiver.
//! 4. **The hold per 128 m cell by its slopes** (§3.3, B84's rule): on the fixed tree each link's slope scaled by a factor
//!    per 128 m point's own Voronoi cell, read with Gaussian weights (σ half a 128 m spacing), half the logarithm a sweep,
//!    until each cell's mean meets the drawn ground's mean over the same points; a cell with fewer than four free points,
//!    or whose own links lift it less than a metre, is LEFT TO THE LAW (the design's owed cure: such a cell drove its
//!    factor without end); a scaled slope stays under `S_c` (the threshold is a law, B23).
//!
//! **Example.** The 16 m tile under hull 17 on the home planet's day-side hills: about 590 000 points over 12.3 km, the
//! 128 m flow over 57 km around it, its channels marked, the steady form and the hold, then 640 × 640 heights in quarter
//! metres — the tile and half its buffer — the same on any host.

use super::delaunay::{Dir, NONE, triangulate_on_face};
use super::geometry::{geometry, locate};
use super::l1::{Domain, L1_RUNG, U0_LAND, cells_per_node_shift, domain_nodes};
use super::points::{CellId, Level, chart_cell, chart_of, domain_points};
use super::read::{FineBook, FineZ};
use super::tree::{accumulate, tree};
use crate::artifact::Artifact;
use crate::body::{BodyDefinition, TAN_REPOSE};
use crate::gf::Gf;
use crate::macro_lattice::MacroLattice;
use crate::solve::{JOINT_K_LP, JOINT_N};
use crate::surface::SOIL_CRITICAL_SLOPE;
use crate::units::{LENGTH_BITS, STEPS_PER_M};
use vd_seed::bend::Face;

/// The level's rung (the rung-4 cell: 16 m on the home planet).
pub const L2_RUNG: u8 = 4;
/// The buffer each side, as a shift of the node: a quarter node (2 km on the home planet; B0 step 2b measured one node's
/// eighth let a cut run along a valley the halves placed apart).
pub const BUFFER_SHIFT: u32 = 2;
/// The 128 m flow's region: the node and this many nodes each side (the Gaussian read of the rates reaches 2.45 nodes, so
/// every 128 m point the domain reads stands in it whole).
pub const FLOW_NODES: i32 = 3;
/// The creep's rate (m²/yr; Roering, Kirchner & Dietrich 1999, the Oregon Coast Range — the design's interim, A14).
pub const CREEP_D_M2_YR: f64 = 0.003;
/// Freeman's exponent: the water shared over the lower neighbours by `(drop/length)^p` (Freeman 1991).
pub const FREEMAN_P: f64 = 1.1;
/// The warm start's passes (the joint law under tan 35°, as the 128 m level) and the steady form's.
pub const WARM_PASSES: usize = 40;
/// ★ THE STEADY FORM'S END (B93's rule one level down): the drainage built again after every second pass, the passes
/// stopped when no 128 m cell's mean moves more than [`STEADY_SETTLE_M`], within the 128 m level's own bound.
pub const STEADY_BOUND: usize = super::l1::FINE_ROUND_BOUND;
pub const STEADY_SETTLE_M: f64 = 1.0;
/// A channel's reach in 16 m spacings; the hold's Gaussian in 128 m spacings and its reach in widths.
pub const CHANNEL_REACH: f64 = 0.75;
pub const HOLD_SIGMA: f64 = 0.5;
pub const HOLD_REACH: f64 = 2.5;

/// A solved 16 m tile: the tile and half its buffer as `side × side` heights in quarter metres over the sea (rows of the
/// face's chart from its north-west corner), and the readings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L2Tile {
    pub side: usize,
    pub quarters: Vec<i32>,
    pub points: usize,
    /// The warm start's passes and the steady form's.
    pub passes: [usize; 2],
    /// The channels' 128 m links and the 16 m points they hold.
    pub channels: [usize; 2],
    /// The channels' beds under the drawn ground: the mean and the largest depth, in centimetres.
    pub bed_cm: [i64; 2],
    /// The law's detail added to today's ground: the median and the largest size, in centimetres.
    pub detail_cm: [i64; 2],
}

/// The drawn 128 m ground at the centre of `cell` (a cell of `rung`) in metres over the sea; `None` where its tile is not
/// held.
fn drawn(book: &FineBook, cell: &CellId, rung: u8, sea: Gf) -> Option<Gf> {
    let face = Face::from_index(cell.face)?;
    match book.z(face, rung, rung, cell.i, cell.j) {
        FineZ::Held(z) => {
            Some(Gf::from_i64(z.raw()) / Gf::from_i64(STEPS_PER_M << LENGTH_BITS) - sea)
        }
        FineZ::Missing(_) | FineZ::Outside => None,
    }
}

/// A direction as metres on the sphere.
fn metres(d: &Dir, r_m: Gf) -> [Gf; 3] {
    let s = r_m / Gf::from_f64((1u64 << vd_recipe::bend::DIR_BITS) as f64);
    [
        Gf::from_i64(d[0].raw()) * s,
        Gf::from_i64(d[1].raw()) * s,
        Gf::from_i64(d[2].raw()) * s,
    ]
}

fn dist2(a: &[Gf; 3], b: &[Gf; 3]) -> Gf {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// The point `p`'s squared distance to the segment `a → b` and the segment's parameter at its foot, in `[0, 1]`.
fn to_segment(p: &[Gf; 3], a: &[Gf; 3], b: &[Gf; 3]) -> (Gf, Gf) {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if len2 > Gf::ZERO {
        ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2).clamp(Gf::ZERO, Gf::ONE)
    } else {
        Gf::ZERO
    };
    let foot = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    (dist2(p, &foot), t)
}

/// The median of `v` (the lower of the two middles), `None` when empty.
fn median(mut v: Vec<Gf>) -> Option<Gf> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    (!v.is_empty()).then(|| v[(v.len() - 1) >> 1])
}

/// ★ THE CREEP'S STEADY SLOPE (Roering, Kirchner & Dietrich 1999): the slope at which the nonlinear creep
/// `q = D·S / (1 − (S/S_c)²)` carries `ul` (the rate times the area over the contour), in its closed form
/// `S = S_c²·(√(D² + (2·ul/S_c)²) − D) / (2·ul)`; nought where nothing is carried.
fn creep_slope(ul: Gf) -> Gf {
    let (d, sc) = (
        Gf::from_f64(CREEP_D_M2_YR),
        Gf::from_f64(SOIL_CRITICAL_SLOPE),
    );
    if ul <= Gf::ZERO {
        return Gf::ZERO;
    }
    let two_ul = ul * Gf::TWO;
    let x = two_ul / sc;
    sc * sc * ((d * d + x * x).sqrt() - d) / two_ul
}

/// ★ THE BANK'S MEAN HEIGHT OVER ITS CHANNEL (Roering, Perron & Kirchner 2007: the steady hillslope of the nonlinear
/// creep): from the divide (`x` = 0) to the channel (`x` = `half`), each step's slope the creep's steady slope for the rate
/// `u` times the distance from the divide ([`creep_slope`], the same form the 16 m solve stands on); the height over the
/// foot `z(x) = ∫ₓ S`, and its mean over the hillslope `(1/L)·∫₀ᴸ s·S(s) ds` — how far a channel's bed lies under the
/// mean of the cell its banks fill. Thirty-two steps by the middle rule.
fn bank_mean_height(u: Gf, half: Gf) -> Gf {
    const STEPS: i64 = 32;
    if half <= Gf::ZERO {
        return Gf::ZERO;
    }
    let dx = half / Gf::from_i64(STEPS);
    let mut sum = Gf::ZERO;
    for k in 0..STEPS {
        let x = dx * (Gf::from_i64(k) + Gf::HALF);
        sum += x * creep_slope(u * x) * dx;
    }
    sum / half
}

/// The joint law's slope for the rate `u` and the water `q` (`(U/(K·Q))^(1/n)`); unbounded (a large word) where no water.
fn river_slope(u: Gf, q: Gf) -> Gf {
    let k = Gf::from_f64(JOINT_K_LP);
    if q <= Gf::ZERO || u <= Gf::ZERO {
        return Gf::from_f64(f64::MAX);
    }
    (u / (k * q)).powf_fine(Gf::from_f64(1.0 / JOINT_N))
}

/// ★ THE 128 m FLOW around the node `(ni, nj)` of `face`: the domain over the node and [`FLOW_NODES`] each side, its
/// points at the drawn ground, the tree, the water through each point, each point's rate and rain. `None` where a
/// finished 128 m tile it reads is not held.
struct Flow {
    d: Domain,
    h: Vec<Gf>,
    recv: Vec<u32>,
    q: Vec<Gf>,
    rate: Vec<Gf>,
    rain: Vec<Gf>,
    pos: Vec<[Gf; 3]>,
}

fn flow(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    book: &FineBook,
    face: Face,
    (ni, nj): (i32, i32),
) -> Option<Flow> {
    let d = domain_nodes(
        body,
        lattice,
        artifact,
        L1_RUNG,
        face,
        (ni - FLOW_NODES, nj - FLOW_NODES),
        2 * FLOW_NODES + 1,
    )?;
    let sea = Gf::from_i32(i32::from(artifact.sea_m));
    let r_m = Gf::from_f64(body.radius_m());
    let mut h = Vec::with_capacity(d.dirs.len());
    for (p, c) in d.cells.iter().enumerate() {
        let z = drawn(book, c, L1_RUNG, sea)?;
        h.push(if d.outlet[p] { z.lesser(d.base[p]) } else { z });
    }
    let t = tree(&h, &d.outlet, &d.graph, &d.geom.len);
    let q = accumulate(&t, &d.water);
    // THE RATE: the joint law read backwards over each node's links under the threshold, the median a node.
    let (k, n, tan) = (
        Gf::from_f64(JOINT_K_LP),
        Gf::from_f64(JOINT_N),
        Gf::from_f64(TAN_REPOSE),
    );
    let mut samples: Vec<Vec<Gf>> = vec![Vec::new(); d.nodes.len()];
    for c in 0..h.len() {
        let r = t.recv[c];
        if d.outlet[c] || r == NONE {
            continue;
        }
        let s = (h[c] - h[r as usize]) / t.link[c];
        if s > Gf::ZERO && s < tan && q[c] > Gf::ZERO {
            samples[d.pnode[c] as usize].push(k * q[c] * s.powf_fine(n));
        }
    }
    let overall =
        median(samples.iter().flatten().copied().collect()).unwrap_or(Gf::from_f64(U0_LAND));
    let node_rate: Vec<Gf> = samples
        .into_iter()
        .map(|v| median(v).unwrap_or(overall))
        .collect();
    // Each read node's rate (a node outside the region the region's own median), then each point's by the Gaussian of
    // the logarithms (the 128 m level's own read).
    let mut ref_rate = vec![overall.ln_fine(); d.refs.len()];
    for (k, &r) in d.node_ref.iter().enumerate() {
        if (r as usize) < ref_rate.len() {
            ref_rate[r as usize] = node_rate[k].ln_fine();
        }
    }
    let mut rate = Vec::with_capacity(h.len());
    let mut rain = Vec::with_capacity(h.len());
    for p in 0..h.len() {
        let mut l = Gf::ZERO;
        for w in d.wstart[p] as usize..d.wstart[p + 1] as usize {
            l += d.wval[w] * ref_rate[d.wnode[w] as usize];
        }
        rate.push(l.exp_fine());
        rain.push(d.water[p] / d.geom.area[p]);
    }
    let pos = d.dirs.iter().map(|x| metres(x, r_m)).collect();
    Some(Flow {
        recv: t.recv,
        d,
        h,
        q,
        rate,
        rain,
        pos,
    })
}

/// A grid of point indices by their cells on a chart rectangle (compressed rows): every point whose cell stands at
/// `(x, y)` of the rectangle `[x0, x0 + w) × [y0, y0 + h)`.
struct CellGrid {
    x0: i32,
    y0: i32,
    w: i32,
    h: i32,
    start: Vec<u32>,
    pts: Vec<u32>,
}

impl CellGrid {
    fn new(at: &[Option<(i32, i32)>], (x0, y0, w, h): (i32, i32, i32, i32)) -> CellGrid {
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
    fn near(&self, (xa, xb): (i32, i32), (ya, yb): (i32, i32)) -> impl Iterator<Item = u32> + '_ {
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

/// The 16 m drainage of the moment: the flow tree, and the water and the area through each point shared over its lower
/// neighbours by Freeman's rule (from the highest point down, ties to the lower index; a point with no lower neighbour
/// hands all to its receiver; an outlet keeps what reaches it).
struct Network<'a> {
    outlet: &'a [bool],
    graph: &'a super::delaunay::Graph,
    len: &'a [Gf],
    water: &'a [Gf],
    area: &'a [Gf],
}

impl Network<'_> {
    fn build(&self, h: &[Gf]) -> (super::tree::Tree, Vec<Gf>, Vec<Gf>) {
        let t = tree(h, self.outlet, self.graph, self.len);
        let n = h.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            h[b as usize]
                .partial_cmp(&h[a as usize])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let p_freeman = Gf::from_f64(FREEMAN_P);
        let (mut q, mut a) = (self.water.to_vec(), self.area.to_vec());
        let mut shares: Vec<(u32, Gf)> = Vec::with_capacity(16);
        for &c in &order {
            let c = c as usize;
            if self.outlet[c] {
                continue;
            }
            shares.clear();
            let mut sum = Gf::ZERO;
            let (lo, hi) = (
                self.graph.start[c] as usize,
                self.graph.start[c + 1] as usize,
            );
            for (&m, &l) in self.graph.adj[lo..hi].iter().zip(&self.len[lo..hi]) {
                if h[m as usize] < h[c] {
                    let w = ((h[c] - h[m as usize]) / l).powf_fine(p_freeman);
                    shares.push((m, w));
                    sum += w;
                }
            }
            if sum > Gf::ZERO {
                for &(m, w) in &shares {
                    let f = w / sum;
                    let (qc, ac) = (q[c] * f, a[c] * f);
                    q[m as usize] += qc;
                    a[m as usize] += ac;
                }
            } else if t.recv[c] != NONE {
                let r = t.recv[c] as usize;
                let (qc, ac) = (q[c], a[c]);
                q[r] += qc;
                a[r] += ac;
            }
        }
        (t, q, a)
    }
}

/// ★ THE STEADY FORM'S HEIGHTS on the tree `t`: each point over its receiver by the lower of the joint law's slope for
/// its rate and water and the creep's for its rate and area over the 16 m contour, never steeper than `S_c`; an outlet
/// (or a point no outlet reaches) on its base.
#[allow(clippy::too_many_arguments)]
fn integrate_form(
    t: &super::tree::Tree,
    outlet: &[bool],
    base: &[Gf],
    rates: &[Gf],
    q: &[Gf],
    a: &[Gf],
    width: Gf,
    sc: Gf,
    h: &mut [Gf],
) {
    for &c in &t.order {
        let c = c as usize;
        let r = t.recv[c];
        if outlet[c] || r == NONE {
            h[c] = base[c];
            continue;
        }
        let ul = rates[c] * a[c] / width;
        let s = river_slope(rates[c], q[c])
            .lesser(creep_slope(ul))
            .lesser(sc);
        h[c] = h[r as usize] + s * t.link[c];
    }
}

/// ★ THE 16 m TILE `key` — one macro node: the node `(key.tx, key.ty)` of `key.face` — refining the finished 128 m tiles of `book`. `None` where the
/// rung's cells do not nest in the nodes by a power of two, a finished 128 m tile it reads is not held, or a domain
/// cannot be built.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn solve_l2(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    book: &FineBook,
    key: super::l1::TileKey,
) -> Option<L2Tile> {
    let (face, ni, nj) = (key.face, key.tx, key.ty);
    let f1 = flow(body, lattice, artifact, book, face, (ni, nj))?;
    let sea = Gf::from_i32(i32::from(artifact.sea_m));
    let r_m = Gf::from_f64(body.radius_m());
    let (n4, n7) = (
        body.ladder().cells_per_edge(L2_RUNG) as i32,
        body.ladder().cells_per_edge(L1_RUNG) as i32,
    );
    let shift4 = cells_per_node_shift(body, lattice, L2_RUNG)?;
    let shift7 = cells_per_node_shift(body, lattice, L1_RUNG)?;
    let down = shift4.checked_sub(shift7)?;
    let node4 = 1i32 << shift4;
    let buf = node4 >> BUFFER_SHIFT;
    let (i0, j0, side) = ((ni << shift4) - buf, (nj << shift4) - buf, node4 + 2 * buf);
    let level = Level::new(body.seed(), L2_RUNG, n4 as u32)?;
    let pts = domain_points(&level, face, i0, j0, side, side);
    let dirs: Vec<Dir> = pts.iter().map(|p| p.dir).collect();
    let cells: Vec<CellId> = pts.iter().map(|p| p.cell).collect();
    // The face's plane (`triangulate_on_face`): the words shifted so a face's edge spans about 2^28 units — about 8 000
    // between two 16 m neighbours on the home planet.
    let plane_shift = (32 - (n4 as u32).leading_zeros())
        .saturating_add(12)
        .min(40);
    let graph = triangulate_on_face(&dirs, face, 40 - plane_shift)?;
    let geom = geometry(&dirs, &graph, r_m);
    let n = dirs.len();
    let pos: Vec<[Gf; 3]> = dirs.iter().map(|d| metres(d, r_m)).collect();
    let mut base = Vec::with_capacity(n);
    for c in &cells {
        base.push(drawn(book, c, L2_RUNG, sea)?);
    }
    let mut outlet: Vec<bool> = base.iter().map(|&b| b <= Gf::ZERO).collect();
    // Each 16 m point's place on the chart (rung 4), and each 128 m point's in the same chart at rung 4 (its cell's
    // middle): where a link crosses the domain's rim, and which cells a search reads.
    let at4: Vec<Option<(i32, i32)>> = cells.iter().map(|&c| chart_of(face, c, n4)).collect();
    let at7: Vec<Option<(i32, i32)>> = f1.d.cells.iter().map(|&c| chart_of(face, c, n7)).collect();
    let mid = 1i32 << (down - 1);
    let in4 = |a: Option<(i32, i32)>| -> Option<(i32, i32)> {
        a.map(|(x, y)| ((x << down) + mid, (y << down) + mid))
    };
    let inside = |a: Option<(i32, i32)>| {
        in4(a).is_some_and(|(x, y)| (x >= i0) & (y >= j0) & (x < i0 + side) & (y < j0 + side))
    };
    let grid4 = CellGrid::new(&at4, (i0, j0, side, side));
    let (fi0, fj0) = ((ni - FLOW_NODES) << shift7, (nj - FLOW_NODES) << shift7);
    let fside = (2 * FLOW_NODES + 1) << shift7;
    let grid7 = CellGrid::new(&at7, (fi0, fj0, fside, fside));
    // The 16 m point nearest a place on the chart at rung 4 (its cell and the eight around; ties to the lower index).
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
    // THE HOLD'S WEIGHTS AND EACH POINT'S CELL: the 128 m points within the reach of each 16 m point, by the Gaussian of
    // their distance (σ half a 128 m spacing); its cell the nearest (ties to the lower index).
    let spacing7 = Gf::from_f64(lattice.node_m() / f64::from(1u32 << shift7));
    let spacing4 = Gf::from_f64(lattice.node_m() / f64::from(1u32 << shift4));
    let sigma = spacing7 * Gf::from_f64(HOLD_SIGMA);
    let reach2 = (sigma * Gf::from_f64(HOLD_REACH)) * (sigma * Gf::from_f64(HOLD_REACH));
    let two_s2 = sigma * sigma * Gf::TWO;
    let (mut wstart, mut wcell, mut wval) = (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    let mut cell_of = vec![u32::MAX; n];
    for p in 0..n {
        wstart.push(wcell.len() as u32);
        let Some((x, y)) = at4[p] else {
            continue;
        };
        let (cx, cy) = (x >> down, y >> down);
        let first = wcell.len();
        let mut sum = Gf::ZERO;
        let mut best: Option<(Gf, u32)> = None;
        for m in grid7.near((cx - 2, cx + 2), (cy - 2, cy + 2)) {
            let d2 = dist2(&pos[p], &f1.pos[m as usize]);
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
    let ln_rate1: Vec<Gf> = f1.rate.iter().map(|r| r.ln_fine()).collect();
    let mut rate = Vec::with_capacity(n);
    let mut water = Vec::with_capacity(n);
    for p in 0..n {
        let (mut l, mut rain) = (Gf::ZERO, Gf::ZERO);
        for w in wstart[p] as usize..wstart[p + 1] as usize {
            l += wval[w] * ln_rate1[wcell[w] as usize];
            rain += wval[w] * f1.rain[wcell[w] as usize];
        }
        rate.push(if wstart[p + 1] > wstart[p] {
            l.exp_fine()
        } else {
            Gf::ZERO
        });
        water.push(rain * geom.area[p]);
    }
    // THE RIM: a 128 m link leaving the domain is an exit at the 16 m point nearest its crossing, on the drawn ground
    // along the link; a link entering hands its water to the point nearest its crossing.
    let lerp = |a: Gf, b: Gf, t: Gf| a + (b - a) * t;
    let crossing = |c: usize, r: usize| -> Option<(i32, i32, Gf)> {
        let (a, b) = (in4(at7[c])?, in4(at7[r])?);
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
    for c in 0..f1.h.len() {
        let r = f1.recv[c];
        if r == NONE || f1.d.outlet[c] {
            continue;
        }
        let r = r as usize;
        let (ci, ri) = (inside(at7[c]), inside(at7[r]));
        if ci == ri {
            continue;
        }
        let Some((x, y, t)) = crossing(c, r) else {
            continue;
        };
        let foot = [
            lerp(f1.pos[c][0], f1.pos[r][0], t),
            lerp(f1.pos[c][1], f1.pos[r][1], t),
            lerp(f1.pos[c][2], f1.pos[r][2], t),
        ];
        let Some(p) = nearest4(x, y, &foot) else {
            continue;
        };
        if ci {
            let z = lerp(f1.h[c], f1.h[r], t);
            base[p] = if outlet[p] { base[p].lesser(z) } else { z };
            outlet[p] = true;
            exits += 1;
        } else if !outlet[p] {
            water[p] += f1.q[c];
        }
    }
    // THE CHANNELS: a 128 m link inside the domain whose water the 16 m law cuts holds the 16 m points along it.
    let (mut channel_links, mut channel_points) = (0usize, 0usize);
    let (mut bed_sum, mut bed_max) = (Gf::ZERO, Gf::ZERO);
    let reach_c = spacing4 * Gf::from_f64(CHANNEL_REACH);
    let reach_c2 = reach_c * reach_c;
    for c in 0..f1.h.len() {
        let r = f1.recv[c];
        if r == NONE || f1.d.outlet[c] || !inside(at7[c]) {
            continue;
        }
        let r = r as usize;
        let (u, q, rain) = (f1.rate[c], f1.q[c], f1.rain[c]);
        if rain <= Gf::ZERO {
            continue;
        }
        // The area over the contour at the 128 m level's own width: a 128 m link drains a strip one 128 m spacing wide.
        let ul = u * (q / rain) / spacing7;
        if river_slope(u, q) >= creep_slope(ul) {
            continue;
        }
        channel_links += 1;
        // ★ THE BED UNDER THE CELL'S MEAN: the drawn 128 m ground is each cell's mean, the bed plus its banks; the bed
        // lies under it by the banks' mean height over half a 128 m spacing at the cell's rate.
        let depth = bank_mean_height(u, spacing7 * Gf::HALF);
        bed_sum += depth;
        bed_max = bed_max.greater(depth);
        let (Some(a), Some(b)) = (in4(at7[c]), in4(at7[r])) else {
            continue;
        };
        let reach = 1i32 << down;
        let near: Vec<u32> = grid4
            .near(
                (a.0.min(b.0) - reach, a.0.max(b.0) + reach),
                (a.1.min(b.1) - reach, a.1.max(b.1) + reach),
            )
            .collect();
        for p in near {
            let p = p as usize;
            let (d2, t) = to_segment(&pos[p], &f1.pos[c], &f1.pos[r]);
            if d2 <= reach_c2 {
                let z = lerp(f1.h[c], f1.h[r], t) - depth;
                if !outlet[p] {
                    channel_points += 1;
                }
                base[p] = if outlet[p] { base[p].lesser(z) } else { z };
                outlet[p] = true;
            }
        }
    }
    if exits == 0 && !outlet.iter().any(|&o| o) {
        // No 128 m water leaves and no sea: the rim's lowest point drains the domain.
        let rim = |p: usize| {
            at4[p].is_some_and(|(x, y)| {
                (x == i0) | (y == j0) | (x == i0 + side - 1) | (y == j0 + side - 1)
            })
        };
        if let Some(p) = (0..n).filter(|&p| rim(p)).min_by(|&a, &b| {
            base[a]
                .partial_cmp(&base[b])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        }) {
            outlet[p] = true;
        }
    }
    // THE WARM START: the joint law under tan 35° (the 128 m level's own form) on the 16 m points.
    let tan = Gf::from_f64(TAN_REPOSE);
    let mut h = base.clone();
    let mut last_recv: Vec<u32> = Vec::new();
    let mut warm = 0usize;
    for _ in 0..WARM_PASSES {
        warm += 1;
        let t = tree(&h, &outlet, &graph, &geom.len);
        let q = accumulate(&t, &water);
        for &c in &t.order {
            let c = c as usize;
            let r = t.recv[c];
            if outlet[c] || r == NONE {
                h[c] = base[c];
                continue;
            }
            h[c] = h[r as usize] + river_slope(rate[c], q[c]).lesser(tan) * t.link[c];
        }
        let same = t.recv == last_recv;
        last_recv = t.recv;
        if same {
            break;
        }
    }
    // THE STEADY FORM, the law alone at each cell's own rate: the drainage and the shared water built again every pass
    // until no point changes its receiver.
    let sc = Gf::from_f64(SOIL_CRITICAL_SLOPE);
    let rates: Vec<Gf> = rate.clone();
    let net = Network {
        outlet: &outlet,
        graph: &graph,
        len: &geom.len,
        water: &water,
        area: &geom.area,
    };
    let (mut t, mut q, mut a) = net.build(&h);
    // Each 128 m cell's mean over its 16 m points (the settle law reads how far the means move).
    let ncell = f1.h.len();
    let cell_means = |h: &[Gf]| -> Vec<Gf> {
        let (mut sum, mut area) = (vec![Gf::ZERO; ncell], vec![Gf::ZERO; ncell]);
        for p in 0..n {
            let c = cell_of[p];
            if c != u32::MAX {
                sum[c as usize] += h[p] * geom.area[p];
                area[c as usize] += geom.area[p];
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
    let settle = Gf::from_f64(STEADY_SETTLE_M);
    let mut last = cell_means(&h);
    let mut steady = 0usize;
    for pass in 1..=STEADY_BOUND {
        steady = pass;
        integrate_form(&t, &outlet, &base, &rates, &q, &a, spacing4, sc, &mut h);
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
        if pass.is_multiple_of(2) {
            (t, q, a) = net.build(&h);
        }
    }
    // THE LAW'S RASTER: the tile and half its buffer, cell centres at rung 4, by the walk's weights; beside it the drawn
    // 128 m ground at the same cells (today's ground, the curve every rung under 7 reads).
    let half = buf >> 1;
    let (ri0, rj0, rside) = (
        (ni << shift4) - half,
        (nj << shift4) - half,
        node4 + 2 * half,
    );
    let rs = rside as usize;
    let inv_n = vd_recipe::bend::inv_n_of(n4 as u32);
    let mut law: Vec<Option<Gf>> = Vec::with_capacity(rs * rs);
    let mut curve: Vec<Option<Gf>> = Vec::with_capacity(rs * rs);
    let mut row_from = 0u32;
    for bj in 0..rside {
        let mut from = row_from;
        for ai in 0..rside {
            let Some(c) = chart_cell(face, ri0 + ai, rj0 + bj, n4) else {
                law.push(None);
                curve.push(None);
                continue;
            };
            let qd = vd_recipe::bend::direction(i32::from(c.face), c.i, c.j, inv_n);
            let (tri, w) = locate(&dirs, &graph, from, &qd)?;
            let v = graph.tris[tri as usize];
            law.push(Some(
                w[0] * h[v[0] as usize] + w[1] * h[v[1] as usize] + w[2] * h[v[2] as usize],
            ));
            curve.push(Some(drawn(book, &c, L2_RUNG, sea)?));
            if ai == 0 {
                row_from = tri;
            }
            from = tri;
        }
    }
    // ★ THE DETAIL (ruling B95, the owner: "adjust the formula and law, so we don't break what we already have today"):
    // the 16 m ground is TODAY'S DRAWN GROUND plus the law's detail finer than 128 m — the law's raster less its own 128 m
    // version, made exactly as today's curve is made from its samples (the box means over each 128 m cell, read back by
    // the same Catmull-Rom between the cells' centres). So each 128 m cell keeps today's mean, the 128 m shapes stay as
    // they are, and the law only adds what lies under 128 m: no hold, no fight between two levels.
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
    // A cell's place among the 128 m cells' centres: `(x + ½ − k/2) / k`, its whole part and its fraction.
    let place = |x: usize| -> (i64, Gf) {
        let u = (Gf::from_i64(x as i64) + Gf::HALF - Gf::from_i64((k >> 1) as i64))
            / Gf::from_i64(k as i64);
        let i = u.floor();
        (i.to_i64_floor(), u - i)
    };
    let clampm = |i: i64| i.clamp(0, mb as i64 - 1) as usize;
    let (mut quarters, mut detail) = (Vec::with_capacity(rs * rs), Vec::new());
    for y in 0..rs {
        let (iy, ty) = place(y);
        let wy = cr(ty);
        for x in 0..rs {
            let (Some(lz), Some(cz)) = (law[y * rs + x], curve[y * rs + x]) else {
                quarters.push(i32::MIN);
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
            let z = cz + d;
            quarters.push((z * Gf::from_i64(4) + Gf::HALF).floor().to_i64_floor() as i32);
        }
    }
    detail.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let cm = |g: Gf| (g * Gf::from_i64(100)).floor().to_i64_floor();
    Some(L2Tile {
        side: rs,
        quarters,
        points: n,
        passes: [warm, steady],
        channels: [channel_links, channel_points],
        bed_cm: [
            cm(bed_sum / Gf::from_i64(channel_links.max(1) as i64)),
            cm(bed_max),
        ],
        detail_cm: [
            detail.get(detail.len() >> 1).map_or(0, |&g| cm(g)),
            detail.last().map_or(0, |&g| cm(g)),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fine::l1::{TILE_NODES, TileKey, solve_l1};
    use crate::fine::level::{FineLevel, LevelGeom};
    use crate::fine::stitch::{Field, mean_levels};
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_solve_words};
    use crate::solve::{Schedule, SolveWords};

    /// ★ ONE 16 m TILE OF THE WET MOON, END TO END (ruling B96). The 16 m node at the corner where four 128 m tiles meet
    /// — (NegX, 6 … 7, 0 … 1), beside the tile the 128 m test's rule picks — whose flow (three nodes each way, and the
    /// 128 m curve's stencil a sample or two past them) reads all four; each 128 m tile held as its own raw square and
    /// that square's mean levels (the finish's stitch is the finish's own tests'; the shard holds the 3 × 3 finished tiles
    /// around a 16 m tile for the same reach). With none held the 16 m tile does not solve; with them it solves: the warm
    /// start and the steady form within their passes, every raster cell given a height, and EVERY cell today's drawn
    /// 128 m ground plus at most the law's largest detail (and the raster's rounding) — the 16 m level refines today's
    /// ground and never departs from it by more than its own detail.
    #[test]
    fn the_wet_moon_16m_tile_solves_on_its_128m_ground() {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact = Artifact::of(&state, &facies, &climate, true);
        let node = TileKey {
            face: Face::NegX,
            tx: 7 * TILE_NODES,
            ty: TILE_NODES,
        };
        let mut book =
            FineBook::new(FineLevel::L1, &moon, &lattice, artifact.sea_m).expect("a 128 m book");
        assert_eq!(
            solve_l2(&moon, &lattice, &artifact, &book, node).map(|t| t.side),
            None
        );
        // The four 128 m tiles around the node's corner, solved side by side, each held as its own square.
        let geom = LevelGeom::of(FineLevel::L1, &moon, &lattice).expect("the 128 m level");
        let keys: Vec<TileKey> = [(6, 0), (7, 0), (6, 1), (7, 1)]
            .into_iter()
            .map(|(tx, ty)| TileKey {
                face: node.face,
                tx,
                ty,
            })
            .collect();
        let squares: Vec<(TileKey, Vec<Vec<i32>>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .iter()
                .map(|&k| {
                    let (moon, lattice, artifact) = (&moon, &lattice, &artifact);
                    scope.spawn(move || {
                        let t = solve_l1(moon, lattice, artifact, k).expect("a 128 m tile solves");
                        let (w, m, n) = (t.side, geom.margin as usize, geom.side() as usize);
                        let v: Vec<i32> = (0..n)
                            .flat_map(|y| t.quarters[(m + y) * w + m..(m + y) * w + m + n].to_vec())
                            .collect();
                        let square = Field {
                            gx0: i64::from(k.tx) * geom.side(),
                            gy0: i64::from(k.ty) * geom.side(),
                            w: n,
                            h: n,
                            v,
                        };
                        let mut levels = vec![square.v.clone()];
                        levels.extend(
                            mean_levels(
                                &square,
                                square.gx0,
                                square.gy0,
                                n,
                                FineLevel::L1.mean_levels(),
                            )
                            .expect("the mean levels"),
                        );
                        (k, levels)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a 128 m solve"))
                .collect()
        });
        for (k, levels) in squares {
            assert!(book.insert(k, levels));
        }
        let tile = solve_l2(&moon, &lattice, &artifact, &book, node).expect("the 16 m tile solves");
        // The tile (one node) and half its buffer each side, every cell given a height.
        let shift4 = cells_per_node_shift(&moon, &lattice, L2_RUNG).expect("nests") as i32;
        let half = (1i32 << shift4) >> (BUFFER_SHIFT + 1);
        let side = (1usize << shift4) + 2 * half as usize;
        assert_eq!((tile.side, tile.quarters.len()), (side, side * side));
        assert_eq!(tile.quarters.iter().filter(|&&q| q == i32::MIN).count(), 0);
        assert!(tile.passes[0] <= WARM_PASSES, "{:?}", tile.passes);
        assert!(
            (1..=STEADY_BOUND).contains(&tile.passes[1]),
            "{:?}",
            tile.passes
        );
        assert!(tile.detail_cm[1] > 0, "{:?}", tile.detail_cm);
        // Every cell today's drawn ground plus at most the largest detail (its centimetre's floor, and the rounding).
        let sea = Gf::from_i32(i32::from(artifact.sea_m));
        let bound = Gf::from_i64((tile.detail_cm[1] + 1) * 4) / Gf::from_i64(100) + Gf::HALF;
        let (i0, j0) = ((node.tx << shift4) - half, (node.ty << shift4) - half);
        let mut beyond = 0usize;
        for b in 0..side {
            for a in 0..side {
                let cell = CellId {
                    face: node.face.index(),
                    i: i0 + a as i32,
                    j: j0 + b as i32,
                };
                let curve = drawn(&book, &cell, L2_RUNG, sea).expect("today's ground is held");
                let q = Gf::from_i64(i64::from(tile.quarters[b * side + a]));
                beyond += usize::from((q - curve * Gf::from_i64(4)).abs() > bound);
            }
        }
        assert_eq!(beyond, 0);
    }
}
