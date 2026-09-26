//! ★ THE EIGHT STANDS AND THEIR FOUR NESTED WINDOWS (plan §3).
//!
//! Each stand's square comes from its direction, as `shipped_drainage` and `shore_teeth` derive
//! theirs (`cell_of`, `node_under`, `sea_edge`). A stand has TWO centres: the far one (W2 and W3,
//! the 250 km and 1 000 km windows) and the near one (W0 and W1, the 1 km and 16 km windows), so a
//! pilot's close windows stand on a shore even where the far aim stands inland.
//!
//! **Example.** The belt: W2 is centred on the owner's aim; W1 on the nearest lake node to it that
//! has a dry side; W0 on the middle of that node's dry edge, so the 1 km window holds the lake's
//! shore under a pilot's boots.

use std::collections::{BTreeMap, VecDeque};

use vd_bins::drainage_gates::{
    BELT, COAST, GENTLE, HIGH_LAKE, PLAIN, POCKET_1, POCKET_2, cell_of, node_under,
};
use vd_bins::shore_geometry::{NodeEdge, sea_edge, unit_of};
use vd_bins::shore_rules::{Nodes, WORD_LAKE, WORD_LAND, WORD_POCKET, WORD_SEA};
use vd_core::glam::DVec3;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};

use crate::census::{Census, val};

/// The reference stand's node at generator 21 (`step2b_census.txt`, THE POSE).
pub const REFERENCE_NODE: u32 = 8_857_509;
/// The high lake's size and level as the step-2b pictures stated them.
pub const HIGH_LAKE_NODES: usize = 12_562;
pub const HIGH_LAKE_LEVEL_M: i16 = 5_983;
/// The gentle rule's words (design §7.2, made exact in plan §3.1).
pub const GENTLE_SLOPE: f64 = 1.0 / 50.0;
pub const GENTLE_REACH_KM: f64 = 1_000.0;
pub const GENTLE_MIN_PAIRS: usize = 4;
pub const GENTLE_MARGIN: i32 = 3;

/// One stand.
#[derive(Clone, Debug)]
pub struct Stand {
    pub name: &'static str,
    pub face: Face,
    /// The centre of W2 and W3, in rung-0 cells of the face.
    pub far: (i64, i64),
    /// The centre of W1, in rung-0 cells of the face.
    pub w1: (i64, i64),
    /// The centre of W0, in rung-0 cells of the face.
    pub w0: (i64, i64),
    /// The stand's own water level to judge (the high lake), metres.
    pub lake_level: Option<i16>,
}

/// ★ ONE WINDOW: whole cells of its alignment rung, so the windows nest exactly at every rung it
/// reads (plan §3.2, amended: the plan's "whole chunks of the coarsest rung" would quantise W2's
/// place to 254 km and could leave a stand at its edge; whole CELLS of the coarsest rung nest the
/// same and move the window by at most one such cell).
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub name: &'static str,
    pub face: Face,
    /// The finest rung read and the last.
    pub fine: u8,
    pub last: u8,
    /// Columns a side at the finest rung, and the origin there.
    pub n: i32,
    pub i0: i32,
    pub j0: i32,
    /// How far the window was pulled to stay on the face, in finest columns.
    pub pull: (i32, i32),
}

impl Window {
    /// The window at rung `r`: its origin and its columns a side.
    #[must_use]
    pub fn at(&self, r: u8) -> (i32, i32, i32) {
        let k = r - self.fine;
        (self.i0 >> k, self.j0 >> k, self.n >> k)
    }

    /// The rungs the window reads.
    pub fn rungs(&self) -> impl Iterator<Item = u8> {
        self.fine..=self.last
    }
}

/// ★ A WINDOW of `n` columns at `fine` about `centre0` (rung-0 cells), aligned to the cells of rung
/// `align`, pulled inward to stay on the face.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "a window's own words, each stated once"
)]
pub fn window(
    body: &BodyDefinition,
    name: &'static str,
    face: Face,
    fine: u8,
    last: u8,
    n: i32,
    align: u8,
    centre0: (i64, i64),
) -> Window {
    let cells = i64::from(body.ladder().cells_per_edge(fine));
    let a = 1i64 << (align - fine);
    let want = |c: i64| ((c >> fine) - i64::from(n) / 2).div_euclid(a) * a;
    let (wi, wj) = (want(centre0.0), want(centre0.1));
    let clamp = |v: i64| v.clamp(0, cells - i64::from(n));
    let (i0, j0) = (clamp(wi), clamp(wj));
    Window {
        name,
        face,
        fine,
        last,
        n,
        i0: i0 as i32,
        j0: j0 as i32,
        pull: ((i0 - wi) as i32, (j0 - wj) as i32),
    }
}

/// The four windows of a stand (plan §3.2).
#[must_use]
pub fn windows_of(body: &BodyDefinition, s: &Stand) -> [Window; 4] {
    [
        window(body, "W0", s.face, 0, 4, 992, 4, s.w0),
        window(body, "W1", s.face, 3, 8, 1_984, 8, s.w1),
        window(body, "W2", s.face, 7, 13, 1_984, 13, s.far),
        window(body, "W3", s.face, 10, 13, 992, 13, s.far),
    ]
}

/// A node's centre in rung-0 cells.
#[must_use]
pub fn node_centre(fine: &MacroLattice, i: i32, j: i32) -> (i64, i64) {
    let c = i64::from(fine.cells_per_node);
    (i64::from(i) * c + c / 2, i64::from(j) * c + c / 2)
}

/// The unit direction of a node.
#[must_use]
pub fn node_dir(fine: &MacroLattice, node: u32) -> DVec3 {
    unit_of(fine.direction(node))
}

/// Whether `(i, j)` stands `margin` nodes inside the face.
fn inside(fine: &MacroLattice, i: i32, j: i32, margin: i32) -> bool {
    let last = fine.edge as i32 - 1 - margin;
    (margin..=last).contains(&i) && (margin..=last).contains(&j)
}

const SIDES: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Whether a word is dry ground (land or a pocket's floor).
fn dry(word: u8) -> bool {
    (word == WORD_LAND) | (word == WORD_POCKET)
}

/// ★ THE NEAREST LAKE EDGE to a rung-0 cell on its face: the lake node nearest the cell (Euclidean in
/// node units, a tie the smaller index) that has a dry node on one of its four sides, within
/// `reach` nodes. Returns the edge (wet = the lake node).
#[must_use]
pub fn nearest_lake_edge(
    fine: &MacroLattice,
    nodes: &Nodes,
    face: Face,
    cell0: (i64, i64),
    reach: i32,
) -> Option<NodeEdge> {
    let cpn = f64::from(fine.cells_per_node);
    let (x, y) = (cell0.0 as f64 / cpn - 0.5, cell0.1 as f64 / cpn - 0.5);
    let (ci, cj) = (x.round() as i32, y.round() as i32);
    let mut best: Option<(f64, u32, NodeEdge)> = None;
    for dj in -reach..=reach {
        for di in -reach..=reach {
            let (i, j) = (ci + di, cj + dj);
            if !inside(fine, i, j, 1) {
                continue;
            }
            let n = fine.index(face, i, j);
            if nodes.word[n as usize] != WORD_LAKE {
                continue;
            }
            let Some(d) = SIDES
                .iter()
                .map(|&(a, b)| (i + a, j + b))
                .find(|&(a, b)| dry(nodes.word[fine.index(face, a, b) as usize]))
            else {
                continue;
            };
            let dist = (f64::from(i) - x).hypot(f64::from(j) - y);
            if best
                .as_ref()
                .is_none_or(|b| dist < b.0 || (dist == b.0 && n < b.1))
            {
                best = Some((
                    dist,
                    n,
                    NodeEdge {
                        face,
                        dry: d,
                        wet: (i, j),
                        discharge: 0,
                    },
                ));
            }
        }
    }
    best.map(|b| b.2)
}

/// ★ ONE POCKET: the body of nodes at or under the sea's level joined to `start` through the eight
/// neighbours (the cube seams included), as `lake_census` labels it. Returns its nodes and its
/// deepest node.
#[must_use]
pub fn pocket_of(fine: &MacroLattice, nodes: &Nodes, start: u32) -> (Vec<u32>, u32) {
    let sea = nodes.sea_m;
    let mut seen = std::collections::BTreeSet::new();
    let mut queue = VecDeque::from([start]);
    seen.insert(start);
    let mut out = Vec::new();
    let mut deepest = start;
    while let Some(v) = queue.pop_front() {
        out.push(v);
        if nodes.z_m[v as usize] < nodes.z_m[deepest as usize] {
            deepest = v;
        }
        for m in fine.neighbours(v) {
            if m == NO_NODE || seen.contains(&m) || nodes.z_m[m as usize] > sea {
                continue;
            }
            seen.insert(m);
            queue.push_back(m);
        }
    }
    out.sort_unstable();
    (out, deepest)
}

/// ★ THE HOPS FROM THE SEA: a walk over the eight neighbours from every sea node at once — the
/// Chebyshev distance to the nearest sea node on a face.
#[must_use]
pub fn hops_from_sea(fine: &MacroLattice, nodes: &Nodes) -> Vec<u16> {
    let n = nodes.word.len();
    let mut hops = vec![u16::MAX; n];
    let mut queue = VecDeque::new();
    for (k, &w) in nodes.word.iter().enumerate() {
        if w == WORD_SEA {
            hops[k] = 0;
            queue.push_back(k as u32);
        }
    }
    while let Some(v) = queue.pop_front() {
        let h = hops[v as usize];
        for m in fine.neighbours(v) {
            if m != NO_NODE && hops[m as usize] == u16::MAX {
                hops[m as usize] = h.saturating_add(1);
                queue.push_back(m);
            }
        }
    }
    hops
}

/// One gentle-coast candidate: the dry node, its sea neighbour, the coast's slope (the median over
/// the pairs of the 5 × 5 block) and the pairs it read.
#[derive(Clone, Copy, Debug)]
pub struct Gentle {
    pub face: Face,
    pub dry: (i32, i32),
    pub wet: (i32, i32),
    pub slope: f64,
    pub pairs: usize,
}

/// The 1-D Catmull-Rom of four words at `t`, and its slope per node.
fn cr1(p: [f64; 4], t: f64) -> (f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    let w = [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ];
    let dw = vd_bins::shore_rules::dweights(t);
    (
        w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + w[3] * p[3],
        dw[0] * p[0] + dw[1] * p[1] + dw[2] * p[2] + dw[3] * p[3],
    )
}

/// ★ THE GENTLE-COAST RULE (plan §3.1): every land–sea pair of 4-neighbours on one face, three
/// nodes inside it, within `reach_km` of the belt (or the whole globe with `None`); each pair's
/// slope where the 1-D Catmull-Rom through the four nodes on its line crosses the sea (by
/// bisection), over the node; each dry node's coast slope the median over the pairs of its 5 × 5
/// block (at least four). Returns every dry node's coast, least slope first.
#[must_use]
pub fn gentle_coasts(
    body: &BodyDefinition,
    fine: &MacroLattice,
    nodes: &Nodes,
    reach_km: Option<f64>,
) -> Vec<Gentle> {
    let sea = f64::from(nodes.sea_m);
    let belt = DVec3::new(BELT[0], BELT[1], BELT[2]).normalize();
    let cos_reach = reach_km.map(|km| (km * 1_000.0 / body.radius_m()).cos());
    let node_m = fine.node_m();
    let z = |f: Face, i: i32, j: i32| f64::from(nodes.z_m[fine.index(f, i, j) as usize]);
    // The pairs' slopes by dry node, and each dry node's first sea neighbour.
    let mut by_dry: BTreeMap<(u8, i32, i32), Vec<f64>> = BTreeMap::new();
    let mut first_wet: BTreeMap<(u8, i32, i32), (i32, i32)> = BTreeMap::new();
    for node in 0..nodes.word.len() as u32 {
        if nodes.word[node as usize] != WORD_LAND {
            continue;
        }
        let (face, i, j) = fine.split(node);
        if !inside(fine, i, j, GENTLE_MARGIN) {
            continue;
        }
        if let Some(c) = cos_reach
            && node_dir(fine, node).dot(belt) < c
        {
            continue;
        }
        for &(di, dj) in &SIDES {
            let (wi, wj) = (i + di, j + dj);
            if !inside(fine, wi, wj, GENTLE_MARGIN)
                || nodes.word[fine.index(face, wi, wj) as usize] != WORD_SEA
            {
                continue;
            }
            let p = [
                z(face, i - di, j - dj),
                z(face, i, j),
                z(face, wi, wj),
                z(face, wi + di, wj + dj),
            ];
            let f0 = cr1(p, 0.0).0 - sea;
            let f1 = cr1(p, 1.0).0 - sea;
            if f0 * f1 > 0.0 {
                continue;
            }
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if (cr1(p, mid).0 - sea) * f0 > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let slope = cr1(p, 0.5 * (lo + hi)).1.abs() / node_m;
            let key = (face.index(), i, j);
            by_dry.entry(key).or_default().push(slope);
            first_wet.entry(key).or_insert((wi, wj));
        }
    }
    let mut out = Vec::new();
    for &(f, i, j) in by_dry.keys() {
        let mut block = Vec::new();
        for dj in -2..=2 {
            for di in -2..=2 {
                if let Some(v) = by_dry.get(&(f, i + di, j + dj)) {
                    block.extend_from_slice(v);
                }
            }
        }
        if block.len() < GENTLE_MIN_PAIRS {
            continue;
        }
        block.sort_by(f64::total_cmp);
        let median = block[block.len() / 2];
        let face = Face::from_index(f).unwrap_or(Face::PosX);
        out.push(Gentle {
            face,
            dry: (i, j),
            wet: first_wet[&(f, i, j)],
            slope: median,
            pairs: block.len(),
        });
    }
    out.sort_by(|a, b| {
        a.slope.total_cmp(&b.slope).then(
            fine.index(a.face, a.dry.0, a.dry.1)
                .cmp(&fine.index(b.face, b.dry.0, b.dry.1)),
        )
    });
    out
}

/// ★ THE FACE AN AIM'S WINDOWS STAND ON (plan §3.1, amended): the aim's own face, unless its W2
/// would be pulled by more than a quarter of its width to stay on the face (the aim stands near a
/// cube edge or corner); then, among the faces that meet there, the one whose pulled W2 holds the
/// most LAKE nodes (a tie: the aim's own face). The windows are face-local, so a stand at a cube
/// corner must pick one face; the rule picks the one that shows what the stand is for. Returns the
/// face, the aim's rung-0 cell on it, and a note.
///
/// **Example.** The plain's pinned node 1 219 stands at (3, 1) on face `PosX`, three nodes from a
/// cube corner; `PosX` there is sea, and the forty lakes of the step-2 picture stand on a
/// neighbouring face, so the plain's windows stand on that face.
#[must_use]
pub fn face_for_aim(
    body: &BodyDefinition,
    fine: &MacroLattice,
    nodes: &Nodes,
    aim: DVec3,
) -> (Face, (i64, i64), String) {
    let d = [aim.x, aim.y, aim.z];
    let n0 = i64::from(body.ladder().cells_per_edge(0));
    // W2's half-width in rung-0 cells (1 984 columns of rung 7).
    let half = 1_984i64 * 128 / 2;
    let cpn = i64::from(fine.cells_per_node);
    let last = i64::from(fine.edge) - 1;
    // One face's reading: the aim's cell on it, the W2 pull, and the lake nodes W2 holds.
    let read = |face: Face| -> Option<((i64, i64), i64, usize)> {
        if vd_seed::bend::dot_axis(d, face.basis().n) <= 0.0 {
            return None;
        }
        let (t, s) = vd_seed::bend::face_coords(face, d);
        if t.abs() > 1.3 || s.abs() > 1.3 {
            return None;
        }
        let cell = |x: f64| -> i64 {
            (((vd_seed::bend::unbend(x.clamp(-1.0, 1.0)) + 1.0) * 0.5 * n0 as f64) as i64)
                .clamp(0, n0 - 1)
        };
        let c = (cell(t), cell(s));
        let lo = |v: i64| (v - half).clamp(0, n0 - 2 * half);
        let (i0, j0) = (lo(c.0), lo(c.1));
        let pull = (i0 - (c.0 - half)).abs().max((j0 - (c.1 - half)).abs());
        let mut lakes = 0usize;
        for nj in j0 / cpn..=((j0 + 2 * half) / cpn).min(last) {
            for ni in i0 / cpn..=((i0 + 2 * half) / cpn).min(last) {
                let node = fine.index(face, ni as i32, nj as i32);
                lakes += usize::from(nodes.word[node as usize] == WORD_LAKE);
            }
        }
        Some((c, pull, lakes))
    };
    let own = vd_seed::bend::face_of(d);
    let (f_own, i_own, j_own) = cell_of(body, aim, 0);
    let own_read = read(own).unwrap_or(((i64::from(i_own), i64::from(j_own)), 0, 0));
    if own_read.1 <= half / 2 {
        return (
            f_own,
            own_read.0,
            format!(
                "its own face {f_own:?} (W2 pulled {:.0} km)",
                own_read.1 as f64 / 1_000.0
            ),
        );
    }
    let mut notes = Vec::new();
    let mut best = (own, own_read);
    for face in Face::ALL {
        let Some(r) = read(face) else { continue };
        notes.push(format!(
            "{face:?}: {} lake nodes in W2, pulled {:.0} km",
            r.2,
            r.1 as f64 / 1_000.0
        ));
        if r.2 > best.1.2 {
            best = (face, r);
        }
    }
    (
        best.0,
        best.1.0,
        format!(
            "the face {:?} — the aim's own face {own:?} would pull W2 by {:.0} km, over a quarter of it, so the face of most lakes among those that meet there: {}",
            best.0,
            own_read.1 as f64 / 1_000.0,
            notes.join("; ")
        ),
    )
}

/// ★ THE PLAIN'S LAKES (plan §3.1, amended): the plain is the stand of "about forty lakes", and its
/// pinned node stands three nodes from a cube corner where W2 would be pulled onto sea. So the
/// plain's W2 stands on the square of W2's size (31 nodes) that holds the most LAKE nodes among the
/// faces that meet at the aim, its centre within `reach` nodes of the aim — the step-2 picture
/// from 150 km over the aim sees the ground out to its horizon, 1 390 km (a stated rule). Returns
/// the face, the square's centre in rung-0 cells, its lake nodes and a note.
///
/// **Example.** The plain's aim stands on `PosX`'s corner; the densest square of lakes within 600 km
/// stands on a neighbouring face, and W2 is laid on it, so the lake lines read the plain's ponds.
#[must_use]
pub fn densest_lakes(
    fine: &MacroLattice,
    nodes: &Nodes,
    aim: DVec3,
    reach: i64,
    side: i64,
) -> Option<(Face, (i64, i64), usize, String)> {
    let d = [aim.x, aim.y, aim.z];
    let e = i64::from(fine.edge);
    let cpn = i64::from(fine.cells_per_node);
    let mut best: Option<(usize, i64, Face, (i64, i64))> = None;
    let mut notes = Vec::new();
    for face in Face::ALL {
        if vd_seed::bend::dot_axis(d, face.basis().n) <= 0.0 {
            continue;
        }
        let (t, s) = vd_seed::bend::face_coords(face, d);
        if t.abs() > 1.3 || s.abs() > 1.3 {
            continue;
        }
        let node = |x: f64| -> i64 {
            (((vd_seed::bend::unbend(x.clamp(-1.0, 1.0)) + 1.0) * 0.5 * e as f64) as i64)
                .clamp(0, e - 1)
        };
        let (ci, cj) = (node(t), node(s));
        // The face's lake nodes as a summed table.
        let w = (e + 1) as usize;
        let mut sum = vec![0u32; w * w];
        for j in 0..e {
            for i in 0..e {
                let lake = u32::from(
                    nodes.word[fine.index(face, i as i32, j as i32) as usize] == WORD_LAKE,
                );
                let k = (j as usize + 1) * w + i as usize + 1;
                sum[k] = lake + sum[k - 1] + sum[k - w] - sum[k - w - 1];
            }
        }
        let count = |x0: i64, y0: i64| -> usize {
            let (x1, y1) = ((x0 + side) as usize, (y0 + side) as usize);
            let (x0, y0) = (x0 as usize, y0 as usize);
            (sum[y1 * w + x1] + sum[y0 * w + x0] - sum[y0 * w + x1] - sum[y1 * w + x0]) as usize
        };
        let mut face_best: Option<(usize, i64, (i64, i64))> = None;
        for y0 in 0..=(e - side) {
            for x0 in 0..=(e - side) {
                let (mx, my) = (x0 + side / 2, y0 + side / 2);
                let dist = (mx - ci).abs().max((my - cj).abs());
                if dist > reach {
                    continue;
                }
                let n = count(x0, y0);
                if face_best.is_none_or(|b| n > b.0 || (n == b.0 && dist < b.1)) {
                    face_best = Some((n, dist, (mx, my)));
                }
            }
        }
        if let Some((n, dist, (mx, my))) = face_best {
            notes.push(format!(
                "{face:?}: {n} lake nodes, {dist} nodes from the aim"
            ));
            if best.is_none_or(|b| n > b.0 || (n == b.0 && dist < b.1)) {
                best = Some((n, dist, face, (mx * cpn + cpn / 2, my * cpn + cpn / 2)));
            }
        }
    }
    best.map(|(n, _, face, c)| (face, c, n, notes.join("; ")))
}

/// ★ THE EIGHT STANDS (plan §3.1), each printed with its pick; the census carries the stand checks.
#[must_use]
pub fn pick_all(
    body: &BodyDefinition,
    artifact: &Artifact,
    fine: &MacroLattice,
    nodes: &Nodes,
    census: &mut Census,
    only: &[String],
) -> Vec<Stand> {
    let want = |name: &str| only.is_empty() || only.iter().any(|o| o == name);
    let dir3 = |d: [f64; 3]| DVec3::new(d[0], d[1], d[2]).normalize();
    let cpn = fine.cells_per_node;
    let mut out = Vec::new();
    let near_of = |face: Face, cell: (i64, i64), reach: i32| -> Option<NodeEdge> {
        nearest_lake_edge(fine, nodes, face, cell, reach)
    };
    let describe_edge = |e: &NodeEdge| {
        let n = fine.index(e.face, e.wet.0, e.wet.1);
        let d = node_dir(fine, n);
        format!(
            "the wet node ({}, {}) beside the dry node ({}, {}) on face {:?}, node {n}, aim {:.6} {:.6} {:.6}",
            e.wet.0, e.wet.1, e.dry.0, e.dry.1, e.face, d.x, d.y, d.z
        )
    };

    // THE BELT.
    if want("belt") {
        let (face, i, j) = cell_of(body, dir3(BELT), 0);
        let far = (i64::from(i), i64::from(j));
        let (w0, w1) = match near_of(face, far, 40) {
            Some(e) => {
                census.say(format!(
                    "  stand belt: W2/W3 on the aim {BELT:?}; W1 on the nearest lake node's centre, W0 on its dry edge's middle: {}",
                    describe_edge(&e)
                ));
                (e.middle(cpn), node_centre(fine, e.wet.0, e.wet.1))
            }
            None => {
                census.say("  stand belt: no lake within 40 nodes; W0/W1 on the aim");
                (far, far)
            }
        };
        out.push(Stand {
            name: "belt",
            face,
            far,
            w1,
            w0,
            lake_level: None,
        });
    }

    // THE COAST.
    if want("coast") {
        let node = node_under(fine, dir3(COAST));
        match sea_edge(artifact, fine, node) {
            Some(e) => {
                census.say(format!(
                    "  stand coast: node {node} (generator 20's pick 4718858), its sea edge: {}",
                    describe_edge(&e)
                ));
                let m = e.middle(cpn);
                out.push(Stand {
                    name: "coast",
                    face: e.face,
                    far: m,
                    w1: m,
                    w0: m,
                    lake_level: None,
                });
            }
            None => census.say(format!(
                "  stand coast: REFUSED — node {node} has no sea neighbour on its face"
            )),
        }
    }

    // THE PLAIN.
    if want("plain") {
        let node = node_under(fine, dir3(PLAIN));
        let (face0, far0, note) = face_for_aim(body, fine, nodes, node_dir(fine, node));
        census.say(format!(
            "  stand plain: node {node} (pinned: 1219) stands on {note}"
        ));
        let (face, far) = match densest_lakes(fine, nodes, node_dir(fine, node), 75, 31) {
            Some((f, c, n, notes)) => {
                census.say(format!(
                    "  stand plain: W2 on the densest 31-node square of lakes within 75 nodes (600 km) of the aim: {n} lake nodes on face {f:?}, centre ({}, {}) nodes ({notes})",
                    c.0 / i64::from(cpn),
                    c.1 / i64::from(cpn)
                ));
                census.reading(
                    "PLAIN-LAKES-IN-W2",
                    &format!("{n}nodes/{f:?}"),
                    "the-plain's-W2-rule(amended-plan§3.1)",
                );
                (f, c)
            }
            None => (face0, far0),
        };
        let (w0, w1) = match near_of(face, far, 40) {
            Some(e) => {
                census.say(format!(
                    "  stand plain: node {node} (pinned: 1219); W1 on the nearest lake node's centre, W0 on its dry edge's middle: {}",
                    describe_edge(&e)
                ));
                (e.middle(cpn), node_centre(fine, e.wet.0, e.wet.1))
            }
            None => (far, far),
        };
        out.push(Stand {
            name: "plain",
            face,
            far,
            w1,
            w0,
            lake_level: None,
        });
    }

    // THE REFERENCE STAND.
    if want("reference") {
        let (face, i, j) = fine.split(REFERENCE_NODE);
        let far = node_centre(fine, i, j);
        let (w0, w1) = match near_of(face, far, 15) {
            Some(e) => {
                census.say(format!(
                    "  stand reference: node {REFERENCE_NODE}; W1 on the nearest lake node's centre within W2, W0 on its dry edge's middle: {}",
                    describe_edge(&e)
                ));
                (e.middle(cpn), node_centre(fine, e.wet.0, e.wet.1))
            }
            None => {
                census.say(format!(
                    "  stand reference: node {REFERENCE_NODE}; no lake within W2, so W0/W1 on the node centre"
                ));
                (far, far)
            }
        };
        out.push(Stand {
            name: "reference",
            face,
            far,
            w1,
            w0,
            lake_level: None,
        });
    }

    // THE POCKETS need the hops from the sea.
    let hops = if want("pocket1") {
        hops_from_sea(fine, nodes)
    } else {
        Vec::new()
    };

    // POCKET 1.
    if want("pocket1") {
        let (members, deepest) = pocket_of(fine, nodes, POCKET_1);
        let depth = f64::from(nodes.sea_m) - f64::from(nodes.z_m[deepest as usize]);
        let rim = members
            .iter()
            .copied()
            .min_by(|&a, &b| hops[a as usize].cmp(&hops[b as usize]).then(a.cmp(&b)))
            .unwrap_or(POCKET_1);
        let (face, i, j) = fine.split(rim);
        let toward = SIDES
            .iter()
            .map(|&(a, b)| (i + a, j + b))
            .filter(|&(a, b)| inside(fine, a, b, 0))
            .min_by_key(|&(a, b)| {
                let n = fine.index(face, a, b);
                (hops[n as usize], n)
            })
            .unwrap_or((i, j));
        census.say(format!(
            "  stand pocket1: the pocket of node {POCKET_1}: {} nodes (DOC 28 912), deepest node {deepest} at {depth:.0} m under the sea (DOC 11 866); the rim node {rim} ({i}, {j}) on face {face:?}, {} hops from the sea; W0/W1 on its edge toward ({}, {})",
            members.len(),
            hops[rim as usize],
            toward.0,
            toward.1
        ));
        census.reading(
            "POCKET1-SIZE",
            &format!("{}nodes/{}", members.len(), val(depth, 0, "m")),
            "DOC:28912nodes/11866m",
        );
        let e = NodeEdge {
            face,
            dry: (i, j),
            wet: toward,
            discharge: 0,
        };
        out.push(Stand {
            name: "pocket1",
            face,
            far: node_centre(fine, i, j),
            w1: e.middle(cpn),
            w0: e.middle(cpn),
            lake_level: None,
        });
    }

    // THE GENTLE COAST.
    if want("gentle") {
        let near = gentle_coasts(body, fine, nodes, Some(GENTLE_REACH_KM));
        let (list, scope) = match near.first() {
            Some(g) if g.slope <= GENTLE_SLOPE => (near, "within 1000 km of the belt"),
            _ => (
                gentle_coasts(body, fine, nodes, None),
                "the WHOLE GLOBE (none within 1000 km of the belt reads <= 1/50)",
            ),
        };
        match list.first() {
            Some(g) => {
                census.say(format!(
                    "  stand gentle: the rule's least coast slope {scope}: {:.5} over {} pairs, the dry node ({}, {}) beside the sea node ({}, {}) on face {:?}",
                    g.slope, g.pairs, g.dry.0, g.dry.1, g.wet.0, g.wet.1, g.face
                ));
                let (pf, pd, pw) = GENTLE;
                let pinned = list.iter().find(|c| c.face == pf && c.dry == pd);
                census.say(format!(
                    "  stand gentle: PINNED (drainage_gates::GENTLE) the dry node ({}, {}) beside the sea node ({}, {}) on face {pf:?}; its coast slope today {}",
                    pd.0,
                    pd.1,
                    pw.0,
                    pw.1,
                    pinned.map_or("unread".to_string(), |c| format!("{:.5} over {} pairs", c.slope, c.pairs))
                ));
                census.line(
                    "CHECK-GENTLE-PIN",
                    &format!("rule({},{})/{:?}", g.dry.0, g.dry.1, g.face),
                    &format!("=pin({},{})/{pf:?}[drainage_gates::GENTLE]", pd.0, pd.1),
                    if g.face == pf && g.dry == pd {
                        "GREEN"
                    } else {
                        "NO-BAND"
                    },
                );
                let g = pinned.copied().unwrap_or(Gentle {
                    face: pf,
                    dry: pd,
                    wet: pw,
                    slope: f64::NAN,
                    pairs: 0,
                });
                for (k, c) in list.iter().take(5).enumerate() {
                    let n = fine.index(c.face, c.dry.0, c.dry.1);
                    let d = node_dir(fine, n);
                    census.say(format!(
                        "    gentle #{}: slope {:.5} ({} pairs), node {n} ({}, {}) face {:?}, aim {:.6} {:.6} {:.6}",
                        k + 1,
                        c.slope,
                        c.pairs,
                        c.dry.0,
                        c.dry.1,
                        c.face,
                        d.x,
                        d.y,
                        d.z
                    ));
                }
                census.judged(
                    "G-GENTLE-SLOPE",
                    &val(g.slope, 5, ""),
                    "<=0.02[W6;design§7.2]",
                    g.slope <= GENTLE_SLOPE,
                );
                let e = NodeEdge {
                    face: g.face,
                    dry: g.dry,
                    wet: g.wet,
                    discharge: 0,
                };
                let m = e.middle(cpn);
                out.push(Stand {
                    name: "gentle",
                    face: g.face,
                    far: m,
                    w1: m,
                    w0: m,
                    lake_level: None,
                });
            }
            None => census.say("  stand gentle: REFUSED — no coast pair on the globe"),
        }
    }

    // POCKET 2.
    if want("pocket2") {
        let (members, deepest) = pocket_of(fine, nodes, POCKET_2);
        let depth = f64::from(nodes.sea_m) - f64::from(nodes.z_m[deepest as usize]);
        let (face, i, j) = fine.split(deepest);
        census.say(format!(
            "  stand pocket2: the pocket of node {POCKET_2}: {} nodes (DOC 12 542), deepest node {deepest} ({i}, {j}) face {face:?} at {depth:.0} m under the sea (DOC 8 193); every window on the deepest node",
            members.len()
        ));
        census.reading(
            "POCKET2-SIZE",
            &format!("{}nodes/{}", members.len(), val(depth, 0, "m")),
            "DOC:12542nodes/8193m",
        );
        let c = node_centre(fine, i, j);
        out.push(Stand {
            name: "pocket2",
            face,
            far: c,
            w1: c,
            w0: c,
            lake_level: None,
        });
    }

    // THE HIGH LAKE.
    if want("highlake") {
        let aim = dir3(HIGH_LAKE);
        let under = node_under(fine, aim);
        let (face, ui, uj) = fine.split(under);
        let cell = node_centre(fine, ui, uj);
        // The lake patch: the lake nodes joined through the eight neighbours, from the nearest lake
        // node to the aim.
        let start = if nodes.word[under as usize] == WORD_LAKE {
            Some(under)
        } else {
            near_of(face, cell, 8).map(|e| fine.index(face, e.wet.0, e.wet.1))
        };
        match start {
            Some(start) => {
                let mut seen = std::collections::BTreeSet::from([start]);
                let mut queue = VecDeque::from([start]);
                let mut levels: BTreeMap<i16, usize> = BTreeMap::new();
                while let Some(v) = queue.pop_front() {
                    *levels.entry(nodes.level_m[v as usize]).or_default() += 1;
                    for m in fine.neighbours(v) {
                        if m != NO_NODE && nodes.word[m as usize] == WORD_LAKE && seen.insert(m) {
                            queue.push_back(m);
                        }
                    }
                }
                // The edge node nearest the aim among the patch's nodes with a dry side.
                let aim_cell = cell_of(body, aim, 0);
                let (x, y) = (
                    aim_cell.1 as f64 / f64::from(cpn) - 0.5,
                    aim_cell.2 as f64 / f64::from(cpn) - 0.5,
                );
                let mut best: Option<(f64, u32, NodeEdge)> = None;
                for &n in &seen {
                    let (f, i, j) = fine.split(n);
                    if f != face || !inside(fine, i, j, 1) {
                        continue;
                    }
                    let Some(d) = SIDES
                        .iter()
                        .map(|&(a, b)| (i + a, j + b))
                        .find(|&(a, b)| dry(nodes.word[fine.index(face, a, b) as usize]))
                    else {
                        continue;
                    };
                    let dist = (f64::from(i) - x).hypot(f64::from(j) - y);
                    if best
                        .as_ref()
                        .is_none_or(|b| dist < b.0 || (dist == b.0 && n < b.1))
                    {
                        best = Some((
                            dist,
                            n,
                            NodeEdge {
                                face,
                                dry: d,
                                wet: (i, j),
                                discharge: 0,
                            },
                        ));
                    }
                }
                let size = seen.len();
                let main_level = levels
                    .iter()
                    .max_by_key(|(_, c)| **c)
                    .map_or(0, |(l, _)| *l);
                census.say(format!(
                    "  stand highlake: the aim's node {under}; the lake patch {size} nodes (DOC 12 562), levels {levels:?} (DOC 5 983 m)"
                ));
                let ok = size == HIGH_LAKE_NODES && main_level == HIGH_LAKE_LEVEL_M;
                census.judged(
                    "CHECK-HIGHLAKE",
                    &format!("{size}nodes/{main_level}m"),
                    "=12562nodes/5983m[status§8b;DOC]",
                    ok,
                );
                match best {
                    Some((_, n, e)) => {
                        census.say(format!("    the edge node {n}: {}", describe_edge(&e)));
                        out.push(Stand {
                            name: "highlake",
                            face,
                            far: node_centre(fine, e.wet.0, e.wet.1),
                            w1: e.middle(cpn),
                            w0: e.middle(cpn),
                            lake_level: Some(main_level),
                        });
                    }
                    None => {
                        census.say("    REFUSED — the patch has no edge node on the aim's face")
                    }
                }
            }
            None => census.say("  stand highlake: REFUSED — no lake near the aim"),
        }
    }
    out
}
