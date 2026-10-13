//! ★ THE 128 m LEVEL'S SOLVE (B1 (d); the way-in design revision 4 §3.2–§3.4; rulings B80, B82, B84).
//!
//! A TILE is 8 × 8 macro nodes of a face; its DOMAIN adds a buffer of two nodes each side (the measured seam rule). On
//! the domain's hashed Poisson-disk points ([`super::points`]) and their spherical Delaunay graph ([`super::delaunay`])
//! the joint law's STEADY STATE is solved: each point stands over its receiver by `min(S, S_c)·L`, `S = (U/(K·Q))^(1/n)`
//! (Adams et al. 2020, `n` = 2.2), `Q` its rain over the area that drains through it, `S_c` = tan 35° (B82: Burbank et
//! al. 1996). Each node's `U/K` is solved by B80's rounds so the law's mean meets the node's own row: the same law first
//! on points four times further apart (the coarse start), then three rounds at this level, each round's tree passes
//! stopped when no node's mean moves more than 5 m, each round taking HALF the logarithm of `(row/mean)^n` (no clamp,
//! B23). No correction follows (B84): the swap to the rows is measured under a pixel at its switch.
//!
//! **The rim.** The domain is closed except where the macro world's own rivers cross it: a link from a node inside to
//! its receiver outside leaves at the domain point nearest the link's midpoint, at the LOWER of the two nodes' rows (a
//! property of the link, the same for the tiles on both sides); a link from a node outside to a receiver inside hands the
//! outside node's whole discharge to the point nearest its midpoint. Every point over the sea's level of the smooth
//! macro surface below 0 m is the sea, an outlet at that surface.
//!
//! **Example.** The tile of nodes (3, 184 … 191, 552 … 559) on the home planet's belt: about 605 000 points, its 8 km
//! rivers' exits and entries found from their links, the coarse start and three rounds, then 640 × 640 heights in
//! quarter metres — the tile and half its buffer — the same on any host.

use super::delaunay::{Dir, Graph, NONE, chord2, triangulate};
use super::geometry::{Geometry, geometry, locate};
use super::points::{CellId, Level, chart_cell, domain_points};
use super::tree::{Tree, accumulate, tree};
use crate::artifact::{Artifact, RECEIVER_NONE, RECEIVER_SLOT_MASK, node_at, sample_z};
use crate::body::{BodyDefinition, TAN_REPOSE};
use crate::gf::Gf;
use crate::macro_lattice::{MacroLattice, NO_NODE};
use crate::solve::{JOINT_K_LP, JOINT_N};
use crate::units::{LENGTH_BITS, STEPS_PER_M};
use std::collections::BTreeMap;
use vd_seed::bend::Face;

/// The level's rung (the rung-7 cell: 128 m on the home planet).
pub const L1_RUNG: u8 = 7;
/// The coarse start's rung: points four times further apart.
pub const COARSE_RUNG: u8 = L1_RUNG + 2;
/// A tile's side in macro nodes.
pub const TILE_NODES: i32 = 8;
/// The buffer each side, in macro nodes (B0 step 2b: one node let a cut run along a valley the halves placed apart).
pub const BUFFER_NODES: i32 = 2;
/// The rounds of the coarse start (each rebuilding its drainage every pass, B0 step 3).
pub const COARSE_ROUNDS: usize = 10;
/// A coarse round's passes at most, and the mean's move under which a round or a pass stops (metres).
pub const MAX_PASSES: usize = 60;
pub const SETTLE_M: f64 = 5.0;
/// ★ THE FINE ROUNDS' BOUND (ruling B92): this level's rounds stop by the settle law ([`SETTLE_M`]); the bound only
/// guarantees an end, and it is the old method's own pass budget for this level (three rounds of [`MAX_PASSES`]), so
/// the rounds are never slower than the method they replaced. MEASURED: three tiles beside the home berth settled at 78,
/// 78 and 132 rounds with the drainage rebuilt every [`REBUILD_ROUNDS`] rounds.
pub const FINE_ROUND_BOUND: usize = 3 * MAX_PASSES;
/// ★ THE DRAINAGE'S REBUILD (ruling B92 amended, 2026-10-13): this level's drainage is built again from the heights after
/// every `REBUILD_ROUNDS` rounds. MEASURED (the bench's census, three tiles beside the berth, the same start): built only
/// ONCE, the two sides of a divide grew apart with no capture to join them — 1.0–1.7 % of the links over 63°, peaks of
/// 11.5–24.4 km (the owner saw the walls and a spire in the window); rebuilt after every round, the rounds settled on one
/// tile of three; rebuilt after every second round they settled on all three, with 0.02–0.05 % of the links over 63°
/// and peaks of 4.7–5.6 km (the method before B92: 0.0003–0.012 %, 5.5–6.5 km, never settled).
pub const REBUILD_ROUNDS: usize = 2;
/// The Gaussian read's width and reach, in nodes (the same weight in every direction; a node enters or leaves the
/// window only at a weight of 4·10⁻⁶).
pub const READ_SIGMA: f64 = 0.7;
pub const READ_REACH: f64 = 3.5 * READ_SIGMA;
/// The rounds' first rise over `K` for land and for the sea (the steady law's scale; the rounds move it).
pub const U0_LAND: f64 = 1.0e-3;
pub const U0_SEA: f64 = 1.0e-9;

/// A 128 m tile: its face and its index there (nodes `[tx·8, tx·8 + 8) × [ty·8, ty·8 + 8)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TileKey {
    pub face: Face,
    pub tx: i32,
    pub ty: i32,
}

/// The integer inverse of the artifact's logarithm class (`log2_class`): the class's lower bound.
#[must_use]
pub fn decode_class(class: u8, frac: u32) -> u64 {
    let bits = u32::from(class) >> frac;
    let below = u64::from(class) & ((1u64 << frac) - 1);
    (((1u64 << frac) | below) << bits >> frac).saturating_sub(1)
}

/// The cells of `rung` to a node, as a shift: `None` unless the rung's cells nest in the nodes by a power of two.
#[must_use]
pub fn cells_per_node_shift(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    rung: u8,
) -> Option<u32> {
    let n_r = body.ladder().cells_per_edge(rung);
    (0..20u32).find(|&s| lattice.edge << s == n_r)
}

/// One solved domain's working set.
pub struct Domain {
    pub dirs: Vec<Dir>,
    pub cells: Vec<CellId>,
    pub graph: Graph,
    pub geom: Geometry,
    /// Each point's node, as its index into `nodes`.
    pub pnode: Vec<u32>,
    /// The domain's nodes (the chart's 12 × 12, global indices) and each one's row over the sea and whether it is held.
    pub nodes: Vec<u32>,
    pub node_z: Vec<Gf>,
    pub held: Vec<bool>,
    /// Every node a weight reads (global indices, sorted), each one's first rise by its own row (land or sea), and
    /// each domain node's place among them.
    pub refs: Vec<u32>,
    pub ref_u0: Vec<Gf>,
    pub node_ref: Vec<u32>,
    /// Each point's Gaussian weights over nearby nodes (indices into `refs`), flat.
    pub wstart: Vec<u32>,
    pub wnode: Vec<u32>,
    pub wval: Vec<Gf>,
    pub base: Vec<Gf>,
    pub water: Vec<Gf>,
    pub outlet: Vec<bool>,
}

fn metres_over_sea(z: vd_recipe::Gi, sea_m: i16) -> Gf {
    Gf::from_i64(z.raw()) / Gf::from_i64(STEPS_PER_M << LENGTH_BITS)
        - Gf::from_i32(i32::from(sea_m))
}

fn chord2_m(a: &Dir, b: &Dir, r_m: Gf) -> Gf {
    let s = r_m / Gf::from_f64((1u64 << vd_recipe::bend::DIR_BITS) as f64);
    let d = |k: usize| Gf::from_i64(a[k].raw() - b[k].raw()) * s;
    d(0) * d(0) + d(1) * d(1) + d(2) * d(2)
}

/// ★ THE DOMAIN of `key` at `rung` (see the module's head). `None` where the rung's cells do not nest in the nodes by
/// a power of two, the level refuses, or the domain holds too few points.
#[must_use]
pub fn domain(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    rung: u8,
    key: TileKey,
) -> Option<Domain> {
    let n_r = body.ladder().cells_per_edge(rung);
    let shift = cells_per_node_shift(body, lattice, rung)?;
    let level = Level::new(body.seed(), rung, n_r)?;
    let r_m = Gf::from_f64(body.radius_m());
    let node_m = Gf::from_f64(lattice.node_m());
    let edge = lattice.edge as i32;
    let (ni0, nj0) = (
        key.tx * TILE_NODES - BUFFER_NODES,
        key.ty * TILE_NODES - BUFFER_NODES,
    );
    let side = TILE_NODES + 2 * BUFFER_NODES;
    // The domain's nodes, in chart order.
    let mut nodes = Vec::new();
    let mut local: BTreeMap<u32, u32> = BTreeMap::new();
    for b in nj0..nj0 + side {
        for a in ni0..ni0 + side {
            if let Some(c) = chart_cell(key.face, a, b, edge) {
                let node = lattice.index(Face::from_index(c.face)?, c.i, c.j);
                local.entry(node).or_insert_with(|| {
                    nodes.push(node);
                    (nodes.len() - 1) as u32
                });
            }
        }
    }
    let sea = artifact.sea_m;
    let z_of =
        |node: u32| Gf::from_i32(i32::from(artifact.rows[node as usize].z_m) - i32::from(sea));
    let node_z: Vec<Gf> = nodes.iter().map(|&n| z_of(n)).collect();
    let held: Vec<bool> = node_z.iter().map(|&z| z > Gf::ZERO).collect();
    let pts = domain_points(
        &level,
        key.face,
        ni0 << shift,
        nj0 << shift,
        side << shift,
        side << shift,
    );
    let dirs: Vec<Dir> = pts.iter().map(|p| p.dir).collect();
    let cells: Vec<CellId> = pts.iter().map(|p| p.cell).collect();
    let graph = triangulate(&dirs)?;
    let geom = geometry(&dirs, &graph, r_m);
    let n = dirs.len();
    let mut pnode = Vec::with_capacity(n);
    let mut base = Vec::with_capacity(n);
    for c in &cells {
        let face = Face::from_index(c.face)?;
        let node = lattice.index(face, c.i >> shift, c.j >> shift);
        pnode.push(*local.get(&node)?);
        base.push(metres_over_sea(
            sample_z(lattice, artifact, face, rung, c.i, c.j)?,
            sea,
        ));
    }
    // The Gaussian weights over the nodes near each point.
    let two_s2 = Gf::from_f64(2.0 * READ_SIGMA * READ_SIGMA);
    let reach2 = Gf::from_f64(READ_REACH * READ_REACH);
    let (mut wstart, mut wnode, mut wval) = (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    // Each domain node's nearby nodes (7 × 7, deduplicated at a cube corner) with their directions, ONCE: every point
    // of a node reads the same list.
    let near_of: Vec<Vec<(u32, Dir)>> = nodes
        .iter()
        .map(|&k| {
            let (face, ni, nj) = lattice.split(k);
            let mut near: Vec<u32> = Vec::with_capacity(49);
            for dj in -3..=3 {
                for di in -3..=3 {
                    near.push(node_at(
                        lattice,
                        face,
                        i64::from(ni) + di,
                        i64::from(nj) + dj,
                    ));
                }
            }
            near.sort_unstable();
            near.dedup();
            near.into_iter()
                .map(|m| (m, lattice.direction(m)))
                .collect()
        })
        .collect();
    for (p, &pn) in pnode.iter().enumerate() {
        wstart.push(wnode.len() as u32);
        let first = wnode.len();
        let mut sum = Gf::ZERO;
        for &(m, ref dm) in &near_of[pn as usize] {
            let d2 = chord2_m(&dirs[p], dm, r_m) / (node_m * node_m);
            if d2 <= reach2 {
                let w = (-(d2 / two_s2)).exp_fine();
                wnode.push(m);
                wval.push(w);
                sum += w;
            }
        }
        for w in &mut wval[first..] {
            *w = *w / sum;
        }
    }
    wstart.push(wnode.len() as u32);
    // The weights' nodes, dense: each read node once, its first rise by its own row.
    let mut refs: Vec<u32> = wnode.clone();
    refs.sort_unstable();
    refs.dedup();
    for w in &mut wnode {
        *w = refs.binary_search(w).expect("a read node") as u32;
    }
    let ref_u0: Vec<Gf> = refs
        .iter()
        .map(|&m| Gf::from_f64(if z_of(m) > Gf::ZERO { U0_LAND } else { U0_SEA }))
        .collect();
    let node_ref: Vec<u32> = nodes
        .iter()
        .map(|m| refs.binary_search(m).unwrap_or(usize::MAX) as u32)
        .collect();
    let rain_m = |node: u32| {
        Gf::from_i64(decode_class(artifact.rows[node as usize].rain, 4) as i64)
            / Gf::from_i64(1_000)
    };
    let ref_rain: Vec<Gf> = refs.iter().map(|&m| rain_m(m)).collect();
    let mut water = Vec::with_capacity(n);
    for p in 0..n {
        let mut r = Gf::ZERO;
        for t in wstart[p] as usize..wstart[p + 1] as usize {
            r += wval[t] * ref_rain[wnode[t] as usize];
        }
        water.push(r * geom.area[p]);
    }
    let mut outlet: Vec<bool> = base.iter().map(|&b| b <= Gf::ZERO).collect();
    // The macro rivers that cross the rim.
    let nearest = |node_local: u32, target: &Dir| -> Option<usize> {
        let mut best: Option<(i128, usize)> = None;
        for (p, &k) in pnode.iter().enumerate() {
            if k == node_local {
                let d = chord2(&dirs[p], target);
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, p));
                }
            }
        }
        best.map(|b| b.1)
    };
    let receiver = |node: u32| {
        let f = artifact.rows[node as usize].receiver_facies;
        if f & RECEIVER_NONE != 0 {
            NO_NODE
        } else {
            lattice.neighbours(node)[(f & RECEIVER_SLOT_MASK) as usize]
        }
    };
    // The link's midpoint: the half-sum of the two directions (about one unit long, so every component stays under one
    // — the recipe's normalise holds only there; the whole sum, two units long, was slow and wrong).
    let midpoint = |a: u32, b: u32| {
        let (da, db) = (lattice.direction(a), lattice.direction(b));
        vd_recipe::bend::normalise([
            (da[0] + db[0]) >> 1,
            (da[1] + db[1]) >> 1,
            (da[2] + db[2]) >> 1,
        ])
    };
    let mut outs = 0usize;
    for (li, &k) in nodes.iter().enumerate() {
        let r = receiver(k);
        if r != NO_NODE
            && !local.contains_key(&r)
            && let Some(p) = nearest(li as u32, &midpoint(k, r))
        {
            outlet[p] = true;
            base[p] = node_z[li].lesser(z_of(r));
            outs += 1;
        }
        for o in lattice.neighbours(k) {
            if o != NO_NODE
                && !local.contains_key(&o)
                && receiver(o) == k
                && let Some(p) = nearest(li as u32, &midpoint(o, k))
                && !outlet[p]
            {
                let q = artifact.rows[o as usize].discharge_log;
                water[p] += Gf::from_i64(decode_class(q, 2) as i64) / Gf::from_i64(1_000);
            }
        }
    }
    if outs == 0 && !outlet.iter().any(|&o| o) {
        // No river leaves and no sea: the rim's lowest point drains the domain.
        let rim = |c: &CellId| -> bool {
            let (a, b) = (c.i - (ni0 << shift), c.j - (nj0 << shift));
            let s = side << shift;
            (c.face == key.face.index()) & ((a == 0) | (b == 0) | (a == s - 1) | (b == s - 1))
        };
        let low = (0..n).filter(|&p| rim(&cells[p])).min_by(|&a, &b| {
            base[a]
                .partial_cmp(&base[b])
                .expect("finite")
                .then(a.cmp(&b))
        });
        if let Some(p) = low {
            outlet[p] = true;
        }
    }
    Some(Domain {
        dirs,
        cells,
        graph,
        geom,
        pnode,
        nodes,
        node_z,
        held,
        refs,
        ref_u0,
        node_ref,
        wstart,
        wnode,
        wval,
        base,
        water,
        outlet,
    })
}

/// Each held-or-not node's mean height over its points on `d`, weighted by the points' areas (nought for a node with none).
fn node_means(d: &Domain, h: &[Gf]) -> Vec<Gf> {
    let nl = d.nodes.len();
    let (mut sum, mut area) = (vec![Gf::ZERO; nl], vec![Gf::ZERO; nl]);
    for ((&k, &hp), &a) in d.pnode.iter().zip(h).zip(&d.geom.area) {
        sum[k as usize] += hp * a;
        area[k as usize] += a;
    }
    (0..nl)
        .map(|k| {
            if area[k] > Gf::ZERO {
                sum[k] / area[k]
            } else {
                Gf::ZERO
            }
        })
        .collect()
}

/// Each point's rise: the Gaussian of the read nodes' logarithms.
fn point_rises(d: &Domain, u: &[Gf]) -> Vec<Gf> {
    let ln_u: Vec<Gf> = u.iter().map(|x| x.ln_fine()).collect();
    (0..d.base.len())
        .map(|p| {
            let mut s = Gf::ZERO;
            for t in d.wstart[p] as usize..d.wstart[p + 1] as usize {
                s += d.wval[t] * ln_u[d.wnode[t] as usize];
            }
            s.exp_fine()
        })
        .collect()
}

/// ★ ONE EXACT PASS of the joint law's steady state on the tree `t` with its flow `q`: from the outlets up the order, each
/// point its receiver's height plus the law's slope over its link (an outlet or a point no outlet reaches its base).
fn steady_pass(d: &Domain, t: &Tree, q: &[Gf], up: &[Gf], h: &mut [Gf]) {
    let (k_lp, inv_n) = (Gf::from_f64(JOINT_K_LP), Gf::from_f64(1.0 / JOINT_N));
    let tan = Gf::from_f64(TAN_REPOSE);
    let tiny = Gf::from_f64(f64::MIN_POSITIVE);
    for &c in &t.order {
        let c = c as usize;
        let r = t.recv[c];
        if d.outlet[c] || r == NONE {
            h[c] = d.base[c];
            continue;
        }
        let s = (up[c] / (k_lp * q[c].greater(tiny)))
            .powf_fine(inv_n)
            .lesser(tan);
        h[c] = h[r as usize] + s * t.link[c];
    }
}

/// ★ A ROUND'S SLOPES ON THE NETWORK, in logarithms (ruling B92): each point's slope is exp((L − ln(K·q)) / n) — L the
/// Gaussian of the read nodes' logarithms of the rise, ln(K·q) the point's own, taken once with each network — held under
/// the 35° repose; one exponential a point a round, where the rise and the power took three.
fn round_slopes(d: &Domain, u: &[Gf], ln_kq: &[Gf]) -> Vec<Gf> {
    let inv_n = Gf::from_f64(1.0 / JOINT_N);
    let tan = Gf::from_f64(TAN_REPOSE);
    let ln_u: Vec<Gf> = u.iter().map(|x| x.ln_fine()).collect();
    (0..d.base.len())
        .map(|p| {
            let mut l = Gf::ZERO;
            for t in d.wstart[p] as usize..d.wstart[p + 1] as usize {
                l += d.wval[t] * ln_u[d.wnode[t] as usize];
            }
            ((l - ln_kq[p]) * inv_n).exp_fine().lesser(tan)
        })
        .collect()
}

/// The heights along the network from its outlets up: each point its receiver's height plus its slope over its link (an
/// outlet or a point no outlet reaches its base).
fn integrate(d: &Domain, t: &Tree, slopes: &[Gf], h: &mut [Gf]) {
    for &c in &t.order {
        let c = c as usize;
        let r = t.recv[c];
        if d.outlet[c] || r == NONE {
            h[c] = d.base[c];
            continue;
        }
        h[c] = h[r as usize] + slopes[c] * t.link[c];
    }
}

/// The largest move of a held node's mean between two passes.
fn moved_max(d: &Domain, now: &[Gf], last: &[Gf]) -> Gf {
    (0..d.nodes.len())
        .filter(|&k| d.held[k])
        .map(|k| (now[k] - last[k]).abs())
        .fold(Gf::ZERO, Gf::greater)
}

/// The clamp-free half step (B23): each held node's rise times (row/mean)^(n/2), a node over the sea only.
fn half_step(d: &Domain, u: &mut [Gf], means: &[Gf]) {
    let half_n = Gf::from_f64(0.5 * JOINT_N);
    for (k, &mean) in means.iter().enumerate() {
        let r = d.node_ref[k] as usize;
        if d.held[k] && mean > Gf::ZERO && r < u.len() {
            u[r] *= (d.node_z[k] / mean).powf_fine(half_n);
        }
    }
}

/// The heights on `d` after `rounds` rounds from the rises `u` (one a read node, `d.refs`' order) and the start `h`, the
/// drainage REBUILT every pass; returns each round's passes. The coarse start's method.
pub fn rounds(d: &Domain, u: &mut [Gf], h: &mut [Gf], rounds: usize) -> Vec<usize> {
    let settle = Gf::from_f64(SETTLE_M);
    let mut log = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let up = point_rises(d, u);
        let mut last = node_means(d, h);
        let mut passes = 0;
        while passes < MAX_PASSES {
            passes += 1;
            let t = tree(h, &d.outlet, &d.graph, &d.geom.len);
            let q = accumulate(&t, &d.water);
            steady_pass(d, &t, &q, &up, h);
            let now = node_means(d, h);
            let moved = moved_max(d, &now, &last);
            last = now;
            if moved <= settle {
                break;
            }
        }
        log.push(passes);
        half_step(d, u, &last);
    }
    log
}

/// The drainage of the heights `h` on `d` and each point's ln(K·q) on it (the flow is fixed until the next rebuild).
fn network(d: &Domain, h: &[Gf]) -> (Tree, Vec<Gf>) {
    let t = tree(h, &d.outlet, &d.graph, &d.geom.len);
    let (k_lp, tiny) = (Gf::from_f64(JOINT_K_LP), Gf::from_f64(f64::MIN_POSITIVE));
    let ln_kq = accumulate(&t, &d.water)
        .iter()
        .map(|&qc| (k_lp * qc.greater(tiny)).ln_fine())
        .collect();
    (t, ln_kq)
}

/// ★ THE FINE ROUNDS (ruling B92, 2026-10-13, amended the same day): each round is ONE exact pass of the joint law's
/// steady state on the drainage (B77's "the stream law's analytic solution on a drainage network built first") and the
/// half step of the rises; the drainage is built again after every [`REBUILD_ROUNDS`] rounds, so a divide's two sides
/// meet again by capture; the rounds stop when no held node's mean moves more than [`SETTLE_M`] (the settle law), within
/// [`FINE_ROUND_BOUND`]. Returns the rounds run. MEASURED before B92 (the bench `fine_settle_bench`): rebuilt every PASS,
/// the drainage never stopped capturing (~1 % of the points changed their receiver in every pass), so every round ended
/// at its pass limit mid-drift.
///
/// **Example.** A 128 m tile beside the home berth: the drainage of its start is built, two rounds run on it, it is built
/// again from their heights, and after 78 rounds no node's mean moves 5 m; a ridge between two valleys that drain apart
/// stays one ridge, because the higher side's points drain over it once the drainage is built again.
pub fn fine_rounds(d: &Domain, u: &mut [Gf], h: &mut [Gf]) -> usize {
    fine_rounds_within(d, u, h, FINE_ROUND_BOUND)
}

/// [`fine_rounds`] within `bound` rounds: the rounds run, `bound` when the settle law never held.
fn fine_rounds_within(d: &Domain, u: &mut [Gf], h: &mut [Gf], bound: usize) -> usize {
    let settle = Gf::from_f64(SETTLE_M);
    let (mut t, mut ln_kq) = network(d, h);
    let mut last = node_means(d, h);
    for round in 1..=bound {
        let slopes = round_slopes(d, u, &ln_kq);
        integrate(d, &t, &slopes, h);
        let now = node_means(d, h);
        let moved = moved_max(d, &now, &last);
        half_step(d, u, &now);
        last = now;
        if moved <= settle {
            return round;
        }
        if round % REBUILD_ROUNDS == 0 {
            (t, ln_kq) = network(d, h);
        }
    }
    bound
}

/// A solved 128 m tile: the tile and half its buffer as `side × side` heights in quarter metres over the sea (rows of
/// the tile's face chart from its north-west corner), and the readings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L1Tile {
    pub side: usize,
    pub quarters: Vec<i32>,
    pub points: usize,
    pub passes: Vec<usize>,
}

/// ★ THE 128 m TILE `key`: the coarse start, the rounds, the raster. `None` where a domain cannot be built.
#[must_use]
pub fn solve_l1(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    key: TileKey,
) -> Option<L1Tile> {
    let coarse = domain(body, lattice, artifact, COARSE_RUNG, key)?;
    let mut uc = coarse.ref_u0.clone();
    let mut hc = coarse.base.clone();
    let mut passes = rounds(&coarse, &mut uc, &mut hc, COARSE_ROUNDS);
    let fine = domain(body, lattice, artifact, L1_RUNG, key)?;
    // The fine rises: the coarse start's where it read the node, else the first rise by the row.
    let mut u: Vec<Gf> = fine
        .refs
        .iter()
        .zip(&fine.ref_u0)
        .map(|(m, &u0)| coarse.refs.binary_search(m).map_or(u0, |i| uc[i]))
        .collect();
    // The fine start: the coarse heights at each fine point (the outlets keep their base).
    let mut h = fine.base.clone();
    let mut from = 0u32;
    for ((hp, dir), &out) in h.iter_mut().zip(&fine.dirs).zip(&fine.outlet) {
        if out {
            continue;
        }
        if let Some((t, w)) = locate(&coarse.dirs, &coarse.graph, from, dir) {
            let v = coarse.graph.tris[t as usize];
            *hp = w[0] * hc[v[0] as usize] + w[1] * hc[v[1] as usize] + w[2] * hc[v[2] as usize];
            from = t;
        }
    }
    passes.push(fine_rounds(&fine, &mut u, &mut h));
    // The raster: the tile and half its buffer, cell centres at this rung, by the walk's weights.
    let n_r = body.ladder().cells_per_edge(L1_RUNG);
    let shift = cells_per_node_shift(body, lattice, L1_RUNG)?;
    let half = BUFFER_NODES >> 1;
    let (i0, j0) = (
        (key.tx * TILE_NODES - half) << shift,
        (key.ty * TILE_NODES - half) << shift,
    );
    let side = ((TILE_NODES + 2 * half) << shift) as usize;
    let inv_n = vd_recipe::bend::inv_n_of(n_r);
    let mut quarters = Vec::with_capacity(side * side);
    let mut row_from = 0u32;
    for b in 0..side as i32 {
        let mut from = row_from;
        for a in 0..side as i32 {
            let Some(c) = chart_cell(key.face, i0 + a, j0 + b, n_r as i32) else {
                quarters.push(i32::MIN);
                continue;
            };
            let q = vd_recipe::bend::direction(i32::from(c.face), c.i, c.j, inv_n);
            let (t, w) = locate(&fine.dirs, &fine.graph, from, &q)?;
            let v = fine.graph.tris[t as usize];
            let z = w[0] * h[v[0] as usize] + w[1] * h[v[1] as usize] + w[2] * h[v[2] as usize];
            quarters.push((z * Gf::from_i64(4) + Gf::HALF).floor().to_i64_floor() as i32);
            if a == 0 {
                row_from = t;
            }
            from = t;
        }
    }
    Some(L1Tile {
        side,
        quarters,
        points: fine.dirs.len(),
        passes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{ZField, log2_class};
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_planet, home_solve_words};
    use crate::solve::{Schedule, SolveWords};

    #[test]
    fn a_class_decodes_to_its_lower_bound_and_a_rung_nests_by_its_shift() {
        // 1 000 falls in the class of 992 … 1 022; the next class starts at 1 023; the first class is nought.
        let c = log2_class(1_000, 4);
        assert_eq!(
            [
                decode_class(c, 4),
                decode_class(c + 1, 4),
                decode_class(0, 4)
            ],
            [991, 1_023, 0]
        );
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        assert_eq!(cells_per_node_shift(&body, &lattice, L1_RUNG), Some(6));
        let one =
            MacroLattice::with_edge(&body, body.ladder().n).expect("a lattice of one-cell nodes");
        assert_eq!(cells_per_node_shift(&body, &one, L1_RUNG), None);
    }

    /// ★ ONE 128 m TILE OF THE WET MOON, END TO END (the bench's premise: `erosion_bench --fine-moon`). The tile is
    /// picked by a rule — the most land nodes, the first of equals, faces then rows then columns — and solves: the coarse
    /// start's ten rounds and this level's three, each within its passes; the raster the tile and half its buffer, every
    /// cell a face holds given a height.
    #[test]
    fn the_wet_moon_tile_with_the_most_land_solves_end_to_end() {
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
        let edge = lattice.edge as i32;
        let mut best = (
            0usize,
            TileKey {
                face: Face::PosX,
                tx: 0,
                ty: 0,
            },
        );
        for f in 0..6u8 {
            let face = Face::from_index(f).expect("a face");
            for ty in 0..=(edge - 1) >> 3 {
                for tx in 0..=(edge - 1) >> 3 {
                    let land = (0..64)
                        .map(|k| (tx * 8 + (k & 7), ty * 8 + (k >> 3)))
                        .filter(|&(i, j)| (i < edge) & (j < edge))
                        .filter(|&(i, j)| {
                            artifact
                                .z_m(lattice.index(face, i, j))
                                .is_some_and(|z| z > artifact.sea_m)
                        })
                        .count();
                    if land > best.0 {
                        best = (land, TileKey { face, tx, ty });
                    }
                }
            }
        }
        let tile = solve_l1(&moon, &lattice, &artifact, best.1).expect("the tile solves");
        // The fine rounds' own end (ruling B92): within a bound of none no round runs and the start stands as it was.
        let fine = domain(&moon, &lattice, &artifact, L1_RUNG, best.1).expect("the fine domain");
        let (mut u, mut h) = (fine.ref_u0.clone(), fine.base.clone());
        assert_eq!(fine_rounds_within(&fine, &mut u, &mut h, 0), 0);
        assert_eq!((u, h), (fine.ref_u0.clone(), fine.base.clone()));
        assert_eq!((tile.side, tile.quarters.len()), (640, 640 * 640));
        // The coarse start's ten rounds each within their passes, and this level's rounds within their bound.
        assert_eq!(tile.passes.len(), COARSE_ROUNDS + 1);
        assert_eq!(
            tile.passes[..COARSE_ROUNDS]
                .iter()
                .filter(|&&p| (p == 0) | (p > MAX_PASSES))
                .count(),
            0,
            "{:?}",
            tile.passes
        );
        assert!(
            (1..=FINE_ROUND_BOUND).contains(&tile.passes[COARSE_ROUNDS]),
            "{:?}",
            tile.passes
        );
        // Every raster cell a face holds has a height; only the cube corner's diagonal region has none.
        let shift = cells_per_node_shift(&moon, &lattice, L1_RUNG).expect("nests") as i32;
        let n7 = edge << shift;
        let (i0, j0) = (
            (best.1.tx * TILE_NODES - 1) << shift,
            (best.1.ty * TILE_NODES - 1) << shift,
        );
        let unheld = (0..640)
            .flat_map(|b| (0..640).map(move |a| (a, b)))
            .filter(|&(a, b)| {
                chart_cell(best.1.face, i0 + a, j0 + b, n7).is_some()
                    & (tile.quarters[(b * 640 + a) as usize] == i32::MIN)
            })
            .count();
        assert_eq!(unheld, 0);
        // MEASURED on the bench (release, 2026-10-11): the rule picks (NegX, 7, 0), all 64 nodes land, 563 134 points,
        // 3.4 s, the heights −0.8 … 80.8 m over the sea.
        assert_eq!(
            best.1,
            TileKey {
                face: Face::NegX,
                tx: 7,
                ty: 0
            }
        );
    }
}
