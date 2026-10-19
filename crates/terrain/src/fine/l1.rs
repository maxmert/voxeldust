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
//! property of the link, the same for the tiles on both sides) — under the game's rule at its BED under that row
//! ([`exits_at_bed`], ruling B94 item 1); a link from a node outside to a receiver inside hands the
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
    /// Each domain node's BASE LEVEL over the sea (the rounds hold its mean's height over it): nought under
    /// [`OutletRule::Today`], the first standing water its 8 km drainage reaches under [`OutletRule::StandingWater`]
    /// ([`base_levels`]).
    pub node_base: Vec<Gf>,
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
    /// The 8 km rivers that leave the domain: each exit's point, its node (an index into `nodes`) and the receiver
    /// outside (a global index) — [`exits_at_bed`] reads them.
    pub exits: Vec<(u32, u32, u32)>,
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
    domain_nodes(
        body,
        lattice,
        artifact,
        rung,
        key.face,
        (
            key.tx * TILE_NODES - BUFFER_NODES,
            key.ty * TILE_NODES - BUFFER_NODES,
        ),
        TILE_NODES + 2 * BUFFER_NODES,
    )
}

/// ★ THE DOMAIN OVER ANY SQUARE OF NODES: the `side × side` macro nodes of `face`'s chart from the node `(ni0, nj0)`, at
/// `rung` — [`domain`]'s rule for a tile, and the 16 m level's view of the 128 m ground's flow
/// ([`super::l2`]). `None` as [`domain`].
#[must_use]
pub fn domain_nodes(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    rung: u8,
    face: Face,
    (ni0, nj0): (i32, i32),
    side: i32,
) -> Option<Domain> {
    let n_r = body.ladder().cells_per_edge(rung);
    let shift = cells_per_node_shift(body, lattice, rung)?;
    let level = Level::new(body.seed(), rung, n_r)?;
    let r_m = Gf::from_f64(body.radius_m());
    let node_m = Gf::from_f64(lattice.node_m());
    let edge = lattice.edge as i32;
    // The domain's nodes, in chart order.
    let mut nodes = Vec::new();
    let mut local: BTreeMap<u32, u32> = BTreeMap::new();
    for b in nj0..nj0 + side {
        for a in ni0..ni0 + side {
            if let Some(c) = chart_cell(face, a, b, edge) {
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
        face,
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
    let mut exits = Vec::new();
    for (li, &k) in nodes.iter().enumerate() {
        let r = receiver(k);
        if r != NO_NODE
            && !local.contains_key(&r)
            && let Some(p) = nearest(li as u32, &midpoint(k, r))
        {
            outlet[p] = true;
            base[p] = node_z[li].lesser(z_of(r));
            exits.push((p as u32, li as u32, r));
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
    if exits.is_empty() && !outlet.iter().any(|&o| o) {
        // No river leaves and no sea: the rim's lowest point drains the domain.
        let rim = |c: &CellId| -> bool {
            let (a, b) = (c.i - (ni0 << shift), c.j - (nj0 << shift));
            let s = side << shift;
            (c.face == face.index()) & ((a == 0) | (b == 0) | (a == s - 1) | (b == s - 1))
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
    let node_base = vec![Gf::ZERO; nodes.len()];
    Some(Domain {
        dirs,
        cells,
        graph,
        geom,
        pnode,
        nodes,
        node_z,
        held,
        node_base,
        refs,
        ref_u0,
        node_ref,
        wstart,
        wnode,
        wval,
        base,
        water,
        outlet,
        exits,
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

/// The clamp-free half step (B23): each held node's rise times (row/mean)^(n/2), both heights over the node's own base
/// level, a node over its base level only. A node whose mean stays over a row it cannot reach (its points already flat on
/// their outlets) has the law's own limit, a rise of nought; the word keeps it at the smallest positive normal number, so
/// its logarithm stays finite — the floor the steady pass gives the flow (`q.greater(tiny)`). MEASURED before (2026-10-18,
/// the basin lake's tiles): such a rise shrank by 0.0016 a round, reached nought after about a hundred rounds, and its
/// logarithm made every height downstream a NaN.
fn half_step(d: &Domain, u: &mut [Gf], means: &[Gf]) {
    let half_n = Gf::from_f64(0.5 * JOINT_N);
    let tiny = Gf::from_f64(f64::MIN_POSITIVE);
    for (k, &mean) in means.iter().enumerate() {
        let r = d.node_ref[k] as usize;
        let b = d.node_base[k];
        if d.held[k] && mean > b && r < u.len() {
            u[r] = (u[r] * ((d.node_z[k] - b) / (mean - b)).powf_fine(half_n)).greater(tiny);
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

/// ★ WHERE A TILE'S WATER ENDS ITS RIVERS (the one-volume revision §3.2; ruling B105 item 1, Q1 (b)): the rule the
/// domain's outlets follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutletRule {
    /// Today's (the world's FINE_VERSION 3): every point whose smooth base stands under the sea's level, the 8 km
    /// rivers' exits at the rim, and the rim's lowest point where neither stands.
    Today,
    /// ★ EVERY STANDING WATER'S POINTS UNDER ITS LEVEL (B105 Q1 (b); bench only until the world move): each water spreads
    /// from its own 8 km nodes through the points whose smooth base stands under its level — the sea at its level from
    /// the ocean's nodes, each lake at its water word from its own nodes — so a shore follows the base's own contour,
    /// never a node square, and a pocket under the sea's level no path joins to the ocean is no sea; a dry end node's
    /// points under its own row are its floor; the 8 km rivers' exits at the rim stay, each at its bed
    /// ([`exits_at_bed`]).
    StandingWater,
    /// ★ THE DROWNED BASIN (the lake floor design, bench only): the solve takes every lake's basin as LAND that drains to the
    /// lake's deepest node (its floor the basin's only outlet inside the lake, a dry end's rule), and the water map then
    /// floods that eroded ground at the lake's level — so a shore follows the drowned valleys, never the smooth 8 km
    /// surface's contour. The sea, the rivers' exits and the dry ends stay as [`OutletRule::StandingWater`] has them.
    Drowned,
}

/// ★ THE STANDING WATERS A SOLVE READS (rulings B105 Q1 (b), B115): each macro node's standing water in quarter metres
/// over the sea — the sea at nought; a lake at its FINE level where `levels` names its lake (its id: the lowest node index
/// of the connected nodes under its word, [`super::spill::lake_ids`]' rule), else at its 8 km word; none on dry ground,
/// and none on a lake node whose own row stands at or over its lake's fine level (its floor dry under the lowered lake).
///
/// **Example.** The basin lake stands at −2 002 m on the 8 km world; with its fine level −2 721 m named, a node of its
/// floor at −2 500 m reads no water (dry land that drains to the lowered lake), and a node at −3 100 m reads −2 721 m.
pub struct Waters<'a> {
    lattice: &'a MacroLattice,
    artifact: &'a Artifact,
    levels: &'a BTreeMap<u32, i32>,
    /// [`OutletRule::Drowned`]'s view for the solve: every lake's basin is land.
    drowned: bool,
    ids: std::cell::RefCell<BTreeMap<u32, u32>>,
    members: std::cell::RefCell<BTreeMap<u32, std::sync::Arc<Vec<u32>>>>,
}

impl<'a> Waters<'a> {
    /// The waters of `artifact` with the lakes' fine `levels` (lake id → quarter metres over the sea; empty: every lake at
    /// its 8 km word).
    #[must_use]
    pub fn new(
        lattice: &'a MacroLattice,
        artifact: &'a Artifact,
        levels: &'a BTreeMap<u32, i32>,
    ) -> Waters<'a> {
        Waters {
            lattice,
            artifact,
            levels,
            drowned: false,
            ids: std::cell::RefCell::new(BTreeMap::new()),
            members: std::cell::RefCell::new(BTreeMap::new()),
        }
    }

    /// [`OutletRule::Drowned`]'s view of the same waters for the solve: every lake's basin is land.
    #[must_use]
    pub fn drowned(mut self) -> Waters<'a> {
        self.drowned = true;
        self
    }

    /// ★ A LAKE'S DEEPEST NODE: the lowest row among its nodes (the lower index on a tie) — the drowned basin's outlet.
    #[must_use]
    pub fn lake_pit(&self, lake: u32) -> u32 {
        let nodes = self.lake_nodes(lake);
        nodes
            .iter()
            .copied()
            .min_by_key(|&k| (self.artifact.rows[k as usize].z_m, k))
            .unwrap_or(lake)
    }

    /// Whether the node is the sea's.
    fn sea(&self, node: u32) -> bool {
        use crate::artifact::FACIES_SHIFT;
        (self.artifact.rows[node as usize].receiver_facies >> FACIES_SHIFT)
            & crate::solve::FACIES_SEA
            != 0
    }

    /// Whether the node lies under a lake on the 8 km world (its word over its row, not the sea).
    pub fn under_lake(&self, node: u32) -> bool {
        let r = &self.artifact.rows[node as usize];
        !self.sea(node) && r.water_m != crate::artifact::DRY_M && r.water_m > r.z_m
    }

    /// ★ THE LAKE of a lake node: the lowest node index of the connected nodes under its word (eight neighbours), found
    /// once by a walk over the lake and kept.
    #[must_use]
    pub fn lake_of(&self, node: u32) -> u32 {
        if let Some(&id) = self.ids.borrow().get(&node) {
            return id;
        }
        let word = self.artifact.rows[node as usize].water_m;
        let mut members = vec![node];
        let mut seen = std::collections::BTreeSet::from([node]);
        let mut i = 0;
        while i < members.len() {
            for m in self.lattice.neighbours(members[i]) {
                if m != NO_NODE
                    && self.artifact.rows[m as usize].water_m == word
                    && self.under_lake(m)
                    && seen.insert(m)
                {
                    members.push(m);
                }
            }
            i += 1;
        }
        members.sort_unstable();
        let id = members.first().copied().unwrap_or(node);
        let mut ids = self.ids.borrow_mut();
        for &m in &members {
            ids.insert(m, id);
        }
        self.members
            .borrow_mut()
            .insert(id, std::sync::Arc::new(members));
        id
    }

    /// A lake's nodes, in index order (`lake` its id).
    #[must_use]
    pub fn lake_nodes(&self, lake: u32) -> std::sync::Arc<Vec<u32>> {
        let id = self.lake_of(lake);
        std::sync::Arc::clone(self.members.borrow().get(&id).expect("a walked lake"))
    }

    /// ★ THE TILES WHOSE RASTER HOLDS A LAKE'S CELLS: each tile of a lake node or of a node beside one (a raster reaches
    /// half a node past its tile, [`BUFFER_NODES`]), so the lake's spill is read over all of them.
    #[must_use]
    pub fn lake_tiles(&self, lake: u32) -> std::collections::BTreeSet<TileKey> {
        let tile = |node: u32| {
            let (face, ni, nj) = self.lattice.split(node);
            TileKey {
                face,
                tx: ni.div_euclid(TILE_NODES),
                ty: nj.div_euclid(TILE_NODES),
            }
        };
        let mut out = std::collections::BTreeSet::new();
        for &k in self.lake_nodes(lake).iter() {
            out.insert(tile(k));
            for m in self.lattice.neighbours(k) {
                if m != NO_NODE {
                    out.insert(tile(m));
                }
            }
        }
        out
    }

    /// ★ THE LAKES IN A TILE'S DOMAIN (the tile and its buffer, [`BUFFER_NODES`] a side): every lake with a node there —
    /// the lakes whose level the tile's solve reads.
    #[must_use]
    pub fn domain_lakes(&self, key: TileKey) -> std::collections::BTreeSet<u32> {
        let edge = self.lattice.edge as i32;
        let side = TILE_NODES + 2 * BUFFER_NODES;
        let (ni0, nj0) = (
            key.tx * TILE_NODES - BUFFER_NODES,
            key.ty * TILE_NODES - BUFFER_NODES,
        );
        let mut out = std::collections::BTreeSet::new();
        for b in nj0..nj0 + side {
            for a in ni0..ni0 + side {
                if let Some(c) = chart_cell(key.face, a, b, edge)
                    && let Some(face) = Face::from_index(c.face)
                {
                    let node = self.lattice.index(face, c.i, c.j);
                    if self.under_lake(node) {
                        out.insert(self.lake_of(node));
                    }
                }
            }
        }
        out
    }

    /// ★ THE NODE'S STANDING WATER, quarter metres over the sea (see the type's head).
    #[must_use]
    pub fn level_q(&self, node: u32) -> Option<i32> {
        if self.sea(node) {
            return Some(0);
        }
        if !self.under_lake(node) || self.drowned {
            return None;
        }
        let r = &self.artifact.rows[node as usize];
        let sea = i32::from(self.artifact.sea_m);
        let word = 4 * (i32::from(r.water_m) - sea);
        if self.levels.is_empty() {
            return Some(word);
        }
        let level = self
            .levels
            .get(&self.lake_of(node))
            .copied()
            .unwrap_or(word);
        (4 * (i32::from(r.z_m) - sea) < level).then_some(level)
    }

    /// ★ A LAKE'S LEVEL by its id: its fine level where `levels` names it, else its 8 km word (quarter metres over the sea).
    #[must_use]
    pub fn lake_level_q(&self, lake: u32) -> i32 {
        let word = 4
            * (i32::from(self.artifact.rows[lake as usize].water_m)
                - i32::from(self.artifact.sea_m));
        self.levels.get(&lake).copied().unwrap_or(word)
    }

    /// ★ THE BASE OF A LOWERED LAKE'S DRY FLOOR (ruling B115; Powell 1875): a lake node that its lake's fine level leaves
    /// dry is land that drains into that lake, so its base level is the lake's fine level (metres over the sea) — never a
    /// dry end's playa, though the 8 km world gives a closed lake's nodes no receiver. MEASURED before (2026-10-18, the
    /// owner's window over the basin lake, *"Doesn't look believable"*): all 255 nodes of the closed basin lake end their
    /// water; the 162 left dry at −2 721 m each stood as a smooth flat plate one node square wide, so straight node edges
    /// crossed the lake.
    fn lowered_base_m(&self, node: u32) -> Option<Gf> {
        if !self.under_lake(node) || self.level_q(node).is_some() {
            return None;
        }
        let lake = self.lake_of(node);
        if self.drowned {
            // The drowned basin drains to its deepest node: that node's floor is the base.
            let pit = self.lake_pit(lake);
            let sea = i32::from(self.artifact.sea_m);
            return Some(Gf::from_i32(
                i32::from(self.artifact.rows[pit as usize].z_m) - sea,
            ));
        }
        Some(Gf::from_i32(self.lake_level_q(lake)) / Gf::from_i64(4))
    }

    /// Whether a node is a DRY END for the solve: no water and no receiver — and on a lake's floor only the drowned
    /// basin's deepest node (a lowered lake's dry floor drains into its lake).
    fn dry_end(&self, node: u32) -> bool {
        if self.level_q(node).is_some() {
            return false;
        }
        if self.under_lake(node) {
            return self.drowned && node == self.lake_pit(self.lake_of(node));
        }
        self.ends_dry(node) && self.dry_pit(node) == node
    }

    /// Whether the 8 km world ends a node's water on dry ground: no water, no receiver.
    fn ends_dry(&self, node: u32) -> bool {
        !self.sea(node)
            && !self.under_lake(node)
            && self.artifact.rows[node as usize].receiver_facies & RECEIVER_NONE != 0
    }

    /// ★ THE PIT A DRY HOLLOW'S NODE DRAINS TO (Powell 1875; ruling B116's bench): the 8 km world ends the water of EVERY
    /// node of a dry hollow, not of its lowest alone, so from a node that ends dry the path goes down to its lowest
    /// neighbour that also ends dry and stands lower (the lower index on a tie), until none does — that node is the
    /// hollow's pit, and only it keeps a dry end's playa; the others are land whose base level is the pit's floor.
    /// MEASURED before (2026-10-19, the home planet): 6 584 nodes end dry, 5 451 of them in 148 clusters of more than one
    /// node (the largest 2 322 and 1 106), and every one stood as a smooth playa a node square wide — the broad smooth
    /// floors with straight edges in every picture of the basins.
    #[must_use]
    pub fn dry_pit(&self, node: u32) -> u32 {
        let z = |k: u32| self.artifact.rows[k as usize].z_m;
        let mut at = node;
        for _ in 0..self.lattice.node_count() {
            let lower = self
                .lattice
                .neighbours(at)
                .into_iter()
                .filter(|&m| m != NO_NODE && self.ends_dry(m) && z(m) < z(at))
                .min_by_key(|&m| (z(m), m));
            match lower {
                Some(m) => at = m,
                None => return at,
            }
        }
        at
    }

    /// The node's standing water in metres over the sea.
    fn level_m(&self, node: u32) -> Option<Gf> {
        self.level_q(node)
            .map(|q| Gf::from_i32(q) / Gf::from_i64(4))
    }
}

/// The outlets of `d` under [`OutletRule::StandingWater`]: the domain's own river exits at any height, each standing
/// water's spread through the points under its level, and each dry end node's floor; a domain none of them reaches drains
/// at its lowest point.
///
/// MEASURED before (2026-10-18, slice W1 item 5's bench on the basin lake's tiles 3,97,20 … 3,98,21): the exits were kept
/// only over the sea's level, as if the sea took every exit under it. In the closed basin east of the belt no sea stands,
/// so a node whose 8 km river left the domain at −3 026 m lost its only outlet; its points filled to their spill at about
/// −2 314 m, the rounds drove its rise toward nought by 0.0016 a round, its logarithm ran to minus infinity, and every
/// height that drains through it became a NaN, stored as 0 m — 433 252 finished cells of nine tiles, a 2 km cliff at the
/// lake.
#[must_use]
pub fn standing_water_outlets(d: &Domain, waters: &Waters<'_>) -> Vec<bool> {
    let n = d.dirs.len();
    // Each domain node's standing water over the sea ([`Waters`]), or none; and whether it is a dry end (no receiver, no
    // water, and no lake's floor: a lowered lake's dry floor drains into its lake).
    let level: Vec<Option<Gf>> = d.nodes.iter().map(|&k| waters.level_m(k)).collect();
    let dry_end: Vec<bool> = d.nodes.iter().map(|&k| waters.dry_end(k)).collect();
    // The river exits, at any height (a closed basin under the sea's level drains through them as any land does); and
    // today's other outlets over the sea's level (the sea's own rule sets the rest).
    let mut out: Vec<bool> = (0..n)
        .map(|p| d.outlet[p] && d.base[p] > Gf::ZERO)
        .collect();
    for &(p, _, _) in &d.exits {
        out[p as usize] = true;
    }
    // Each distinct level spreads from its own nodes' points under it, through the points under it.
    let mut levels: Vec<Gf> = level.iter().flatten().copied().collect();
    levels.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    levels.dedup();
    for lam in levels {
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = (0..n)
            .filter(|&p| level[d.pnode[p] as usize] == Some(lam) && d.base[p] <= lam)
            .collect();
        for &p in &stack {
            seen[p] = true;
        }
        while let Some(p) = stack.pop() {
            out[p] = true;
            for &q in d.graph.neighbours(p) {
                let q = q as usize;
                if !seen[q] && d.base[q] <= lam {
                    seen[q] = true;
                    stack.push(q);
                }
            }
        }
    }
    // A dry end node's floor: its points whose base stands at or under its own row — or at its lowest point, where the
    // smooth base stands over the row at every point (the curve between a lone low node's samples).
    let mut lowest = vec![None::<Gf>; d.nodes.len()];
    for (&k, &b) in d.pnode.iter().zip(&d.base) {
        let l = &mut lowest[k as usize];
        *l = Some(l.map_or(b, |m| m.lesser(b)));
    }
    for (o, (&k, &b)) in out.iter_mut().zip(d.pnode.iter().zip(&d.base)) {
        let k = k as usize;
        if dry_end[k] && lowest[k].is_some_and(|m| b <= d.node_z[k].greater(m)) {
            *o = true;
        }
    }
    // A domain no outlet reaches: its lowest point (the first lowest by index).
    if !out.iter().any(|&o| o)
        && let Some(p) = (0..n).min_by(|&a, &b| {
            d.base[a]
                .partial_cmp(&d.base[b])
                .expect("finite")
                .then(a.cmp(&b))
        })
    {
        out[p] = true;
    }
    out
}

/// ★ THE BASE LEVEL of a macro node (Powell 1875: a river's work ends at the water it runs into — the sea the grand base
/// level, a lake a local one; ruling B94 item 1, the one-volume revision §3.2): along the node's 8 km receiver chain, the
/// first node that holds standing water gives its level ([`Waters`]: the sea at nought, a lake at its fine or 8 km level), and a chain
/// that ends dry gives its end's floor (the end node's row). Metres over the sea. The receiver tree has no loop; the walk is
/// bounded by the lattice's node count only so it always ends.
fn base_level(waters: &Waters<'_>, node: u32) -> Gf {
    let (lattice, artifact) = (waters.lattice, waters.artifact);
    let sea = i32::from(artifact.sea_m);
    let mut k = node;
    for _ in 0..lattice.node_count() {
        let r = &artifact.rows[k as usize];
        if let Some(level) = waters.level_m(k) {
            return level;
        }
        if let Some(base) = waters.lowered_base_m(k) {
            return base;
        }
        if r.receiver_facies & RECEIVER_NONE != 0 {
            // A dry hollow's node drains to its pit: the pit's floor is its base (its own row where it is the pit).
            let pit = if waters.ends_dry(k) {
                waters.dry_pit(k)
            } else {
                k
            };
            return Gf::from_i32(i32::from(artifact.rows[pit as usize].z_m) - sea);
        }
        let next = lattice.neighbours(k)[(r.receiver_facies & RECEIVER_SLOT_MASK) as usize];
        if next == NO_NODE {
            return Gf::from_i32(i32::from(r.z_m) - sea);
        }
        k = next;
    }
    Gf::from_i32(i32::from(artifact.rows[k as usize].z_m) - sea)
}

/// ★ THE BASE LEVELS under [`OutletRule::StandingWater`]: each domain node's [`base_level`]; a node is held where its row
/// stands over it, and every read node's first rise is the land's where its row stands over its own base level (a dry
/// basin's slopes under the sea's level rise as land, never as the sea's floor).
pub fn base_levels(d: &mut Domain, waters: &Waters<'_>) {
    let artifact = waters.artifact;
    let sea = i32::from(artifact.sea_m);
    d.node_base = d.nodes.iter().map(|&k| base_level(waters, k)).collect();
    d.held = d
        .node_z
        .iter()
        .zip(&d.node_base)
        .map(|(&z, &b)| z > b)
        .collect();
    d.ref_u0 = d
        .refs
        .iter()
        .map(|&m| {
            let z = Gf::from_i32(i32::from(artifact.rows[m as usize].z_m) - sea);
            Gf::from_f64(if z > base_level(waters, m) {
                U0_LAND
            } else {
                U0_SEA
            })
        })
        .collect();
}

/// Hack's law in Whipple & Tucker 1999's metric form `A = ka·x^h` (Hack 1957): the coefficient (m^0.33) and the exponent.
pub const HACK_KA: f64 = 6.69;
pub const HACK_H: f64 = 1.67;
/// The channel head's drainage area, m² (Whipple & Tucker 1999; their calibration bodies the King Range and Taiwan's
/// Central Range). Its distance from the divide, `xc = (A_c/ka)^(1/h)` by Hack's law, is about 316 m.
pub const CHANNEL_HEAD_AREA_M2: f64 = 1.0e5;

/// ★ THE CHANNEL'S DEPTH UNDER THE CELL'S MEAN, metres: [`exits_at_bed`]'s `HI·Rf` for a link of `len_m` whose rows drop
/// by `slope` and whose node drains `area_m2`; nought where the law gives no relief (no drop, no area, or a link no longer
/// than the channel head's distance from the divide).
#[must_use]
pub fn bed_depth_m(slope: Gf, area_m2: Gf, len_m: Gf) -> Gf {
    if slope <= Gf::ZERO || area_m2 <= Gf::ZERO {
        return Gf::ZERO;
    }
    let theta = Gf::ONE / Gf::from_f64(JOINT_N);
    let ka = Gf::from_f64(HACK_KA);
    let hack_p = Gf::from_f64(HACK_H) * theta;
    let rest = Gf::ONE - hack_p;
    let hi = rest / (Gf::TWO - hack_p);
    let head = (Gf::from_f64(CHANNEL_HEAD_AREA_M2) / ka)
        .powf_fine(Gf::ONE / Gf::from_f64(HACK_H))
        .powf_fine(rest);
    let ks = slope * area_m2.powf_fine(theta);
    hi * (ks * ka.powf_fine(-theta) / rest * (len_m.powf_fine(rest) - head)).greater(Gf::ZERO)
}

/// ★ A RIVER LEAVES THE TILE AT ITS BED (ruling B94 item 1; the one-volume revision §3.2), under [`OutletRule::
/// StandingWater`]: each 8 km river that leaves the domain leaves under the lower of its link's two rows by the channel's
/// depth under the cell's mean, `D = HI·Rf`:
/// - `Rf`, the channel relief inside the cell — Whipple & Tucker 1999 (JGR 104:17661) eq. 22,
///   `Rf = ks·ka^−θ·(L^(1−p) − xc^(1−p))/(1−p)`, with the steepness `ks = S·A^θ` read off the 8 km world (`S` the rows'
///   drop over the link, `A` the node's discharge over its rain), `θ = 1/n` of the joint law, `p = h·θ` by Hack's law,
///   `L` the link's length;
/// - `HI = (1−p)/(2−p)`, the share of that relief under the channel's mean — the same law's steady profile
///   `z ∝ L^(1−p) − x^(1−p)` averaged along the channel (Pike & Wilson 1971: the depth under the mean is the hypsometric
///   integral of the relief).
///
/// The bed never falls under the node's base level (Powell 1875: a river cuts no lower than the water it runs into): where
/// the lower row stands under it — a river that runs into the sea or a lake over that water's floor — the river leaves at
/// the water's level. Every term is the link's own, so each tile that holds the exit reads the same bed. MEASURED before
/// (2026-10-18, the river mouth's tiles 5,73,79 … 5,75,81): an exit kept at the sea floor's row (−94 m) cut the land's
/// drainage under the sea's level where no sea reaches — 23 464 finished dry cells under it against 494.
///
/// **Example.** A river that drains 300 km² leaves the hill tile over a link 8 km long whose rows drop 40 m: `ks` 36,
/// `Rf` 294 m, so it leaves 57 m under the lower row. A river of the same area on a plain whose rows drop 1 m over the
/// same link leaves 1.4 m under it, as Kansas's rivers lie a few metres under their uplands.
pub fn exits_at_bed(
    d: &mut Domain,
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
) {
    let r_m = Gf::from_f64(body.radius_m());
    let sea = i32::from(artifact.sea_m);
    let thousand = Gf::from_i64(1_000);
    let beds: Vec<(usize, Gf)> = d
        .exits
        .iter()
        .filter(|&&(p, _, _)| d.outlet[p as usize])
        .map(|&(p, li, r)| {
            let (li, k) = (li as usize, d.nodes[li as usize]);
            let row = &artifact.rows[k as usize];
            let zk = d.node_z[li];
            let zr = Gf::from_i32(i32::from(artifact.rows[r as usize].z_m) - sea);
            let len = chord2_m(&lattice.direction(k), &lattice.direction(r), r_m).sqrt();
            let q = Gf::from_i64(decode_class(row.discharge_log, 2) as i64) / thousand;
            let rain = Gf::from_i64(decode_class(row.rain, 4) as i64) / thousand;
            let area = if rain > Gf::ZERO { q / rain } else { Gf::ZERO };
            let depth = bed_depth_m((zk - zr) / len, area, len);
            let lower = zk.lesser(zr);
            (p as usize, (lower - depth).greater(d.node_base[li]))
        })
        .collect();
    for (p, z) in beds {
        d.base[p] = z;
    }
}

/// A solved 128 m tile: the tile and half its buffer as `side × side` heights in quarter metres over the sea (rows of
/// the tile's face chart from its north-west corner), and the readings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L1Tile {
    pub side: usize,
    pub quarters: Vec<i32>,
    pub points: usize,
    pub passes: Vec<usize>,
    /// ★ THE RIVERS' WATER under [`OutletRule::StandingWater`] (the bench, slice W1): each raster cell a carved channel
    /// holds, its bank's height in quarter metres over the sea ([`super::carve`]); `i32::MIN` where none (every cell under
    /// [`OutletRule::Today`]).
    pub river: Vec<i32>,
    /// Each raster cell's share of the narrower rivers through it, in 255ths ([`super::carve::Carve::share`]).
    pub river_share: Vec<u8>,
    /// ★ THE WATER MAP of the raster under [`OutletRule::StandingWater`] ([`super::water::raster_water`]): each cell's
    /// water level in quarter metres over the sea, [`super::water::DRY`] where none (every cell under
    /// [`OutletRule::Today`]).
    pub water: Vec<i32>,
    /// ★ THE LAKES' SPILLS IN THIS RASTER (ruling B115; [`super::spill::raster_passes`]): each lake with a cell here whose
    /// lowest path to lower ground stays under its level, and that path's height (quarter metres over the sea); empty under
    /// [`OutletRule::Today`].
    pub spills: Vec<(u32, i32)>,
}

/// ★ THE 128 m TILE `key`: the coarse start, the rounds, the raster. `None` where a domain cannot be built.
#[must_use]
pub fn solve_l1(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    key: TileKey,
) -> Option<L1Tile> {
    solve_l1_with(body, lattice, artifact, key, OutletRule::Today)
}

/// [`solve_l1`] under the outlet `rule` (the game's is [`OutletRule::StandingWater`]), every lake at its 8 km word.
#[must_use]
pub fn solve_l1_with(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    key: TileKey,
    rule: OutletRule,
) -> Option<L1Tile> {
    solve_l1_levels(body, lattice, artifact, key, rule, &BTreeMap::new())
}

/// ★ [`solve_l1_with`] WITH THE LAKES' FINE LEVELS (ruling B115, form (a′)): each lake `levels` names (its id → quarter
/// metres over the sea) stands at that level in the outlets, the base levels and the water map, so the ground under its
/// old water is solved as land.
#[must_use]
pub fn solve_l1_levels(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    key: TileKey,
    rule: OutletRule,
    levels: &BTreeMap<u32, i32>,
) -> Option<L1Tile> {
    let waters = Waters::new(lattice, artifact, levels);
    // The solve's view: the same waters, or under [`OutletRule::Drowned`] every lake's basin as land.
    let solving = match rule {
        OutletRule::Drowned => Waters::new(lattice, artifact, levels).drowned(),
        _ => Waters::new(lattice, artifact, levels),
    };
    let wet_rule = rule != OutletRule::Today;
    let mut coarse = domain(body, lattice, artifact, COARSE_RUNG, key)?;
    if wet_rule {
        coarse.outlet = standing_water_outlets(&coarse, &solving);
        base_levels(&mut coarse, &solving);
        exits_at_bed(&mut coarse, body, lattice, artifact);
    }
    let mut uc = coarse.ref_u0.clone();
    let mut hc = coarse.base.clone();
    let mut passes = rounds(&coarse, &mut uc, &mut hc, COARSE_ROUNDS);
    let mut fine = domain(body, lattice, artifact, L1_RUNG, key)?;
    if wet_rule {
        fine.outlet = standing_water_outlets(&fine, &solving);
        base_levels(&mut fine, &solving);
        exits_at_bed(&mut fine, body, lattice, artifact);
    }
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
    // THE RIVER CARVE (the bench's rule only until the world move): the channels of the settled drainage.
    let carve = wet_rule.then(|| {
        let (t, _) = network(&fine, &h);
        let q = super::tree::accumulate(&t, &fine.water);
        // The rung's cell: rung 0 is one metre, each rung twice the one under it.
        let cell_m = Gf::from_i64(1i64 << L1_RUNG);
        let ch = super::carve::channels(&t, &fine.graph, &q, body.facts().gravity_mm_s2, cell_m);
        (t, ch, cell_m)
    });
    let r_m = Gf::from_f64(body.radius_m());
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
    let mut river = vec![i32::MIN; side * side];
    // Each raster cell's node: its standing water ([`Waters`], quarter metres over the sea), its lake (where it lies under
    // one) and its row (quarter metres over the sea).
    let mut node_level: Vec<Option<i32>> = Vec::with_capacity(side * side);
    let mut cell_lake: Vec<u32> = Vec::with_capacity(side * side);
    let mut cell_row: Vec<i32> = Vec::with_capacity(side * side);
    let sea = i32::from(artifact.sea_m);
    let mut share = vec![0u8; side * side];
    let mut row_from = 0u32;
    for b in 0..side as i32 {
        let mut from = row_from;
        for a in 0..side as i32 {
            let Some((c, node)) = chart_cell(key.face, i0 + a, j0 + b, n_r as i32).and_then(|c| {
                let f = Face::from_index(c.face)?;
                Some((c, lattice.index(f, c.i >> shift, c.j >> shift)))
            }) else {
                quarters.push(i32::MIN);
                node_level.push(None);
                cell_lake.push(super::spill::NO_LAKE);
                cell_row.push(i32::MAX);
                continue;
            };
            let level = waters.level_q(node);
            node_level.push(level);
            cell_lake.push(if level.is_some() && waters.under_lake(node) {
                waters.lake_of(node)
            } else {
                super::spill::NO_LAKE
            });
            cell_row.push(4 * (i32::from(artifact.rows[node as usize].z_m) - sea));
            let q = vd_recipe::bend::direction(i32::from(c.face), c.i, c.j, inv_n);
            let (t, w) = locate(&fine.dirs, &fine.graph, from, &q)?;
            let v = fine.graph.tris[t as usize];
            let mut z = w[0] * h[v[0] as usize] + w[1] * h[v[1] as usize] + w[2] * h[v[2] as usize];
            if let Some((tree, ch, cell_m)) = &carve {
                let cv = super::carve::carve_cell(
                    ch,
                    &fine.dirs,
                    &fine.graph,
                    tree,
                    &h,
                    r_m,
                    v,
                    &q,
                    z,
                    *cell_m,
                );
                z = cv.z;
                if let Some(bank) = cv.bank {
                    river[quarters.len()] =
                        (bank * Gf::from_i64(4) + Gf::HALF).floor().to_i64_floor() as i32;
                }
                share[quarters.len()] =
                    (cv.share.lesser(Gf::ONE) * Gf::from_i64(255) + Gf::HALF).to_i64_floor() as u8;
            }
            quarters.push((z * Gf::from_i64(4) + Gf::HALF).floor().to_i64_floor() as i32);
            if a == 0 {
                row_from = t;
            }
            from = t;
        }
    }
    let (water, spills) = if carve.is_some() {
        (
            super::water::raster_water(&quarters, side, &node_level, &river),
            super::spill::raster_passes(&quarters, side, &cell_lake, &cell_row, |l| {
                waters.lake_level_q(l)
            }),
        )
    } else {
        (vec![super::water::DRY; side * side], Vec::new())
    };
    Some(L1Tile {
        side,
        water,
        spills,
        quarters,
        points: fine.dirs.len(),
        passes,
        river,
        river_share: share,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{ZField, log2_class};
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_planet, home_solve_words};
    use crate::solve::{Schedule, SolveWords};

    /// The bed's depth law on the doc's two rivers (300 km² over an 8 km link: a 40 m drop on the hills, 1 m on a plain),
    /// and nought with no drop, no area, or a link shorter than the channel head's distance.
    #[test]
    fn a_river_leaves_at_its_bed_by_the_channel_law() {
        let (area, len) = (Gf::from_f64(3.0e8), Gf::from_f64(8_000.0));
        let hill = bed_depth_m(Gf::from_f64(40.0 / 8_000.0), area, len).to_f64();
        assert!((hill - 57.1).abs() < 0.2, "{hill}");
        let plain = bed_depth_m(Gf::from_f64(1.0 / 8_000.0), area, len).to_f64();
        assert!((plain - 1.43).abs() < 0.01, "{plain}");
        assert_eq!(bed_depth_m(Gf::ZERO, area, len), Gf::ZERO);
        assert_eq!(bed_depth_m(Gf::from_f64(0.01), Gf::ZERO, len), Gf::ZERO);
        assert_eq!(
            bed_depth_m(Gf::from_f64(0.01), area, Gf::from_f64(200.0)),
            Gf::ZERO
        );
    }

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
        // 3.4 s, the heights −0.8 … 80.8 m over the sea; since generator 27 (the seasons' climate, 2026-10-17) it picks
        // (NegX, 7, 2).
        assert_eq!(
            best.1,
            TileKey {
                face: Face::NegX,
                tx: 7,
                ty: 2
            }
        );
    }
}
