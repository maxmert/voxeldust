//! ★ THE MACRO LATTICE (the landform arc, slice 8c stage C1; `slice_8c_design.md` §2; the
//! investigation's `03_erosion_rivers.md` §4.1 and `06_laws_integration.md` §4.3).
//!
//! A lattice is a grid of points laid over the whole globe. The solve that gives a planet its
//! continents, valleys and rivers runs on its OWN lattice: a uniform cube-sphere grid with `edge`
//! nodes along each face edge, where `edge` DIVIDES the body's rung-0 cell count `N`. A node is the
//! cell `(face, i, j)` of that grid and its direction is the recipe's own bend of that cell's
//! CENTRE — the same call the column pass makes, with a different count — so a node is a point the
//! chunk pass can name exactly. Because `edge` divides `N`, a chunk finds its node by ONE integer
//! division, with no float comparison and no ragged strip at a face edge.
//!
//! What this module states, once:
//! - [`MACRO_CELL_TARGET_M`], a METRIC resolution of THE world — the drainage skeleton is resolved
//!   at about eight kilometres on every body, like `SHORT_WAVE_M` is thirty metres on every body
//!   (SL5: one world, no scale knob); it stands in for no physical fact the world holds;
//! - [`MIN_MACRO_EDGE`] and [`MACRO_EDGE_CEILING`], the floor and the memory ceiling of the edge,
//!   both COST knobs and named as such;
//! - the divisor rule ([`macro_edge`]): the divisor of `N` in that range whose node is nearest the
//!   target, ties to the smaller.
//!
//! **Example.** The home planet has `N = 9 961 472 = 2¹⁹ · 19` cells along a face edge. The divisor
//! 1 216 gives a node of exactly 8 192 m and 8 871 936 nodes over the globe. A pilot who looks 50 km
//! down a valley sees six nodes span that view: the solve decided which way the valley runs and
//! where its river is; the spectrum and the carve draw the spurs and the gullies inside each node.
//! The home planet's moon (`N = 557 056 = 2¹⁵ · 17`) gets 68 and 27 744 nodes.

// ★ A LATTICE MAY DIVIDE (ruling F7's rule is about the KERNELS a card runs). The divisor search
// and a node's face split run ONCE per body or per node on the CPU, in the solve, which is
// single-threaded, off the tick and never a GPU kernel (03 §5.3); the chunk pass's own lookup is a
// shift by the node's power-of-two size where the size is one, and the general division here is
// the CPU's, integer-exact on every host by definition.
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the solve's lattice runs on the CPU once per body, never in a kernel; integer-exact on every host"
)]

use vd_recipe::Gi;
use vd_recipe::bend::{DIR_BITS, direction, inv_n_of};
use vd_seed::bend::{Face, K1, K2, K3, bend};
use vd_seed::seam::{Edge, across};

use crate::body::BodyDefinition;
use crate::gf::Gf;

/// ★ THE STATED METRIC RESOLUTION of the drainage skeleton: a node of about this many metres, on
/// every body (the design's ask 1, the recommended value taken).
pub const MACRO_CELL_TARGET_M: u64 = 8_192;
/// The fewest nodes along a face edge: a body too small for the target still gets a lattice of
/// this edge, so its solve is total (a 50 km body: 8 per edge, 384 nodes).
pub const MIN_MACRO_EDGE: u32 = 8;
/// ★ THE MEMORY CEILING (a cost knob, named as one): the most nodes along a face edge, whatever the
/// body's size. At 2 048 the lattice holds 25 165 824 nodes; the solve's transient state at the
/// bench's measured bytes per node is what this number bounds. An ice giant of the home system
/// (`N = 59 768 832`) would take 7 296 per edge under the target alone, 319 million nodes.
pub const MACRO_EDGE_CEILING: u32 = 2_048;
/// The stencil's word for a neighbour that does not exist: at a cube corner only three faces meet,
/// so a corner node has SEVEN neighbours, and the eighth slot carries this.
pub const NO_NODE: u32 = u32::MAX;

/// The eight stencil offsets `(di, dj)`, in a FIXED order: the row below, the row, the row above,
/// left to right. Every neighbour walk reads them in this order and nowhere else.
pub const STENCIL: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// ★ THE DIVISOR RULE. Among the divisors `d` of `n` with `MIN_MACRO_EDGE ≤ d ≤ MACRO_EDGE_CEILING`,
/// the one whose node `n / d` is nearest [`MACRO_CELL_TARGET_M`], ties to the SMALLER `d`; `None`
/// when no divisor lies in the range (a body of fewer than eight cells a face, which the ladder
/// admits for a pebble and which no solve is asked for). Trial division to `sqrt(n)` — at most
/// 8 192 steps, exact integer work, identical on every target.
///
/// **Example.** A 5 km body of THE world has `N = 7 854 = 2 · 3 · 7 · 11 · 17`; every divisor gives a
/// node smaller than the target, so the rule takes the smallest divisor at or above eight, which is
/// 11 — a 714 m node and 726 nodes. A stated answer, not a clamp after the fact.
#[must_use]
pub fn macro_edge(n: u32) -> Option<u32> {
    let mut best: Option<(u64, u32)> = None;
    let mut d = 1u32;
    while u64::from(d) * u64::from(d) <= u64::from(n) {
        if n.is_multiple_of(d) {
            best = consider(best, n, d);
            best = consider(best, n, n / d);
        }
        d += 1;
    }
    best.map(|(_, edge)| edge)
}

/// One candidate divisor against the best so far: nearer wins; at the same distance the smaller
/// edge wins. A divisor outside the stated range is not a candidate.
fn consider(best: Option<(u64, u32)>, n: u32, d: u32) -> Option<(u64, u32)> {
    if !(MIN_MACRO_EDGE..=MACRO_EDGE_CEILING).contains(&d) {
        return best;
    }
    let node = u64::from(n / d);
    let distance = node.abs_diff(MACRO_CELL_TARGET_M);
    match best {
        Some((best_distance, best_edge))
            if best_distance < distance || (best_distance == distance && best_edge <= d) =>
        {
            best
        }
        _ => Some((distance, d)),
    }
}

/// The macro lattice of one body: its edge, its node's size in rung-0 cells, the cell-count
/// reciprocal the bend reads, and the ladder radius the areas and chords scale by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacroLattice {
    /// Nodes along a face edge, `n_macro`.
    pub edge: u32,
    /// Rung-0 cells along a node's edge, `N / n_macro` — an exact integer by the divisor rule.
    pub cells_per_node: u32,
    /// `inv_n_of(edge)`: what the bend multiplies by instead of dividing.
    inv_edge: Gi,
    /// The ladder radius in metres, as a fenced float: the one scale the areas and chords carry.
    radius_m: Gf,
}

impl MacroLattice {
    /// The lattice of `body`; `None` where the divisor rule finds no edge.
    #[must_use]
    pub fn of(body: &BodyDefinition) -> Option<MacroLattice> {
        let n = body.ladder().n;
        let edge = macro_edge(n)?;
        Some(MacroLattice {
            edge,
            cells_per_node: n / edge,
            inv_edge: inv_n_of(edge),
            radius_m: Gf::from_f64(body.ladder().radius_m()),
        })
    }

    /// The nodes over the globe: six faces of `edge²`.
    #[must_use]
    pub fn node_count(&self) -> usize {
        6 * (self.edge as usize) * (self.edge as usize)
    }

    /// The global node index of cell `(face, i, j)`: `face · edge² + j · edge + i` — the total
    /// order every tie-break in the solve reads.
    #[must_use]
    pub fn index(&self, face: Face, i: i32, j: i32) -> u32 {
        (u32::from(face.index()) * self.edge + j as u32) * self.edge + i as u32
    }

    /// The cell `(face, i, j)` of a global node index.
    #[must_use]
    pub fn split(&self, node: u32) -> (Face, i32, i32) {
        let per_face = self.edge * self.edge;
        let face = Face::from_index((node / per_face) as u8).unwrap_or(Face::NegZ);
        let rest = node % per_face;
        (face, (rest % self.edge) as i32, (rest / self.edge) as i32)
    }

    /// The node's unit direction at the bend's fraction bits: the recipe's own bend of the cell's
    /// centre, `(2i + 1)/edge − 1`, so a node is exactly a column direction of a coarser count.
    #[must_use]
    pub fn direction(&self, node: u32) -> [Gi; 3] {
        let (face, i, j) = self.split(node);
        direction(i32::from(face.index()), i, j, self.inv_edge)
    }

    /// The eight neighbours of a node in [`STENCIL`] order, through the seam table where the
    /// stencil leaves the face; [`NO_NODE`] where the stencil leaves the cube at a corner.
    ///
    /// A neighbour across a seam is never `i + 1`: the partner face's axes may be swapped, and the
    /// table says which cell is adjacent. Both hosts read one table, so a river that crosses a face
    /// seam is one river.
    #[must_use]
    pub fn neighbours(&self, node: u32) -> [u32; 8] {
        let (face, i, j) = self.split(node);
        let edge = self.edge as i32;
        let mut out = [NO_NODE; 8];
        for (slot, (di, dj)) in STENCIL.iter().enumerate() {
            let (ni, nj) = (i + di, j + dj);
            let i_in = (0..edge).contains(&ni);
            let j_in = (0..edge).contains(&nj);
            out[slot] = if i_in && j_in {
                self.index(face, ni, nj)
            } else if i_in {
                let side = if nj < 0 { Edge::VMinus } else { Edge::VPlus };
                let c = across(face, side, ni, edge);
                self.index(c.face, c.i, c.j)
            } else if j_in {
                let side = if ni < 0 { Edge::UMinus } else { Edge::UPlus };
                let c = across(face, side, nj, edge);
                self.index(c.face, c.i, c.j)
            } else {
                NO_NODE
            };
        }
        out
    }

    /// ★ THE NODE'S AREA in whole square metres: a MIDPOINT QUADRATURE of the bend's own area
    /// density `W'(a)·W'(b) / |n + W(a)u + W(b)v|³` at the node's centre, times the node's
    /// parameter area `(2/edge)²` and the radius squared, FLOORED. The density varies by a quarter
    /// across a face (a corner node is about 24 % smaller than a centre node), so the discharge
    /// must add real square metres or a basin near a face corner carries the wrong river; across
    /// one node the midpoint error is parts per ten thousand. The floor makes every later SUM exact
    /// integer arithmetic with no rounding order at all.
    #[must_use]
    pub fn area_m2(&self, node: u32) -> u64 {
        let (_, i, j) = self.split(node);
        let a = self.param(i);
        let b = self.param(j);
        let wa = Gf::from_f64(bend(a.to_f64()));
        let wb = Gf::from_f64(bend(b.to_f64()));
        let norm2 = Gf::ONE + wa * wa + wb * wb;
        let norm = norm2.sqrt();
        let density = bend_slope(a) * bend_slope(b) / (norm2 * norm);
        let cell = Gf::TWO / Gf::from_i64(i64::from(self.edge));
        (density * cell * cell * self.radius_m * self.radius_m).to_i64_floor() as u64
    }

    /// ★ THE CHORD between two nodes in whole metres, FLOORED: the radius times the length of the
    /// difference of the two unit directions, one square root under the fence. Macro distances are
    /// kilometres, so a metre is a hundredth of a percent of one; the slope comparison then
    /// cross-multiplies integers and two hosts cannot disagree.
    #[must_use]
    pub fn chord_m(&self, a: u32, b: u32) -> u32 {
        chord_between(self.radius_m, self.direction(a), self.direction(b))
    }

    /// The ladder radius the areas and chords scale by, for the solve's own chord reads.
    #[must_use]
    pub(crate) fn radius_m(&self) -> Gf {
        self.radius_m
    }

    /// ★ THE CHUNK'S LOOKUP: the macro cell `(i, j)` of rung-`rung` cell `(i, j)` of a face — the
    /// rung-0 index `i · 2^rung` divided by the node's exact size. The reason the edge is a divisor:
    /// no float, no seam, no ragged strip.
    #[must_use]
    pub fn node_of_cell(&self, rung: u8, i: i32, j: i32) -> (i32, i32) {
        let size = i64::from(self.cells_per_node);
        (
            ((i64::from(i) << rung) / size) as i32,
            ((i64::from(j) << rung) / size) as i32,
        )
    }

    /// The face parameter of a node's centre as a fenced float, `(2i + 1)/edge − 1`.
    fn param(&self, i: i32) -> Gf {
        (Gf::from_i64(2 * i64::from(i) + 1) / Gf::from_i64(i64::from(self.edge))) - Gf::ONE
    }
}

/// The bend's derivative `W'(a) = k₁ + a²·(3k₂ + a²·5k₃)` — the polynomial the inverse bend
/// differentiates, in the same Horner form.
fn bend_slope(a: Gf) -> Gf {
    let a2 = a * a;
    Gf::from_f64(K1) + a2 * (Gf::from_f64(3.0 * K2) + a2 * Gf::from_f64(5.0 * K3))
}

/// The chord between two unit directions at the bend's bits, in whole metres — the one square root
/// of the solve's routing, under the fence, floored once.
#[must_use]
pub(crate) fn chord_between(radius_m: Gf, a: [Gi; 3], b: [Gi; 3]) -> u32 {
    let one = Gf::from_i64(1i64 << DIR_BITS);
    let d0 = Gf::from_i64(a[0].raw() - b[0].raw()) / one;
    let d1 = Gf::from_i64(a[1].raw() - b[1].raw()) / one;
    let d2 = Gf::from_i64(a[2].raw() - b[2].raw()) / one;
    let len = (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
    (radius_m * len).to_i64_floor() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{home_moon, home_planet};
    use std::collections::BTreeSet;

    /// The design's worked rows, on THE world's own numbers: the home planet's `N = 2¹⁹ · 19` takes
    /// 1 216 (a node of exactly 8 192 m), its moon's `N = 2¹⁵ · 17` takes 68 (8 192 m too).
    #[test]
    fn the_divisor_rule_on_the_home_planet_and_its_moon() {
        let home = MacroLattice::of(&home_planet()).expect("a lattice");
        assert_eq!(home.edge, 1_216);
        assert_eq!(home.cells_per_node, 8_192);
        assert_eq!(home.node_count(), 8_871_936);
        let moon = MacroLattice::of(&home_moon()).expect("a lattice");
        assert_eq!(moon.edge, 68);
        assert_eq!(moon.cells_per_node, 8_192);
        assert_eq!(moon.node_count(), 27_744);
    }

    /// The rule's arms, on stated counts. Every divisor under the target takes the smallest at or
    /// above the floor (the design's 5 km body, `N = 7 854 = 2·3·7·11·17`: 11, a 714 m node). An
    /// exact divisor wins (`2¹⁷`: 16; `2¹⁶·3`: 24; `2¹⁵·3`: 12). Without an exact one the nearest
    /// wins (`2¹⁴·3`: 8 at 6 144 m is 2 048 off, 12 at 4 096 m is 4 096 off). The ceiling closes the
    /// exact divisor of `2²⁵` (4 096 stands over it) and 2 048 at 16 384 m is what remains. A
    /// pebble of five cells a face has no edge.
    #[test]
    fn the_divisor_rules_arms() {
        assert_eq!(macro_edge(7_854), Some(11));
        assert_eq!(macro_edge(131_072), Some(16));
        assert_eq!(macro_edge(196_608), Some(24));
        assert_eq!(macro_edge(98_304), Some(12));
        assert_eq!(macro_edge(49_152), Some(8));
        assert_eq!(macro_edge(33_554_432), Some(2_048));
        assert_eq!(macro_edge(5), None);
    }

    /// The tie the rule promises, at the one step where it is decided: two candidates at one
    /// distance keep the SMALLER edge, whichever is considered first, and a nearer candidate
    /// replaces a further one.
    #[test]
    fn a_tie_keeps_the_smaller_edge() {
        // 16 · 8 202 = 131 232: the candidate 16 stands 10 off, like the best; 8 < 16 stays.
        assert_eq!(consider(Some((10, 8)), 131_232, 16), Some((10, 8)));
        // 8 · 8 202 = 65 616: the candidate 8 stands 10 off, like the best 16; 8 replaces it.
        assert_eq!(consider(Some((10, 16)), 65_616, 8), Some((10, 8)));
        // 8 · 8 182 = 65 456: ten off from below is the same distance.
        assert_eq!(consider(Some((10, 16)), 65_456, 8), Some((10, 8)));
        // A nearer candidate replaces the best; a further one does not.
        assert_eq!(consider(Some((10, 16)), 8 * 8_197, 8), Some((5, 8)));
        assert_eq!(consider(Some((5, 16)), 65_616, 8), Some((5, 16)));
    }

    /// A node's index is a total order over `(face, i, j)`, and the split inverts it on every node
    /// of the moon.
    #[test]
    fn index_and_split_are_inverses_over_the_whole_moon() {
        let moon = MacroLattice::of(&home_moon()).expect("a lattice");
        let mut seen = BTreeSet::new();
        for node in 0..moon.node_count() as u32 {
            let (face, i, j) = moon.split(node);
            assert!((0..moon.edge as i32).contains(&i));
            assert!((0..moon.edge as i32).contains(&j));
            assert_eq!(moon.index(face, i, j), node);
            assert!(seen.insert(node));
        }
        // An index past the six faces reads as −Z (the bend's own rule for a face past 5), so the
        // split is total.
        let (face, _, _) = moon.split(moon.node_count() as u32);
        assert_eq!(face, Face::NegZ);
    }

    /// Every neighbour relation is symmetric through the seam table; an interior node has eight
    /// neighbours, an edge node eight, a corner node SEVEN — and the node's own index is never
    /// among them.
    #[test]
    fn neighbours_are_symmetric_and_a_corner_has_seven() {
        let moon = MacroLattice::of(&home_moon()).expect("a lattice");
        let mut corners = 0;
        for node in 0..moon.node_count() as u32 {
            let ring = moon.neighbours(node);
            let mut present = 0;
            for &m in &ring {
                if m == NO_NODE {
                    continue;
                }
                present += 1;
                assert_ne!(m, node);
                assert!(
                    moon.neighbours(m).contains(&node),
                    "node {node} names {m} which does not name it back"
                );
            }
            assert!(
                (7..=8).contains(&present),
                "node {node} has {present} neighbours"
            );
            corners += usize::from(present == 7);
            // No neighbour is named twice.
            let distinct: BTreeSet<u32> = ring.iter().copied().filter(|&m| m != NO_NODE).collect();
            assert_eq!(distinct.len(), present);
        }
        // A cube has eight corners, each a node of exactly one face... no: three faces meet at a
        // corner and each face's corner cell touches it, so 24 cells are corner cells, each with
        // seven neighbours.
        assert_eq!(corners, 24);
    }

    /// The stencil across a seam at a stated cell: +X's +u edge is +Y with the axes swapped.
    #[test]
    fn a_seam_neighbour_is_the_tables_cell_and_never_i_plus_one() {
        let moon = MacroLattice::of(&home_moon()).expect("a lattice");
        let edge = moon.edge as i32;
        let node = moon.index(Face::PosX, edge - 1, 5);
        let ring = moon.neighbours(node);
        // Stencil slot 4 is (+1, 0): across the +u side at along = 5 → +Y at (5, edge − 1).
        assert_eq!(ring[4], moon.index(Face::PosY, 5, edge - 1));
        // Slot 2 is (+1, −1): across the +u side at along = 4.
        assert_eq!(ring[2], moon.index(Face::PosY, 4, edge - 1));
        // Slot 7 is (+1, +1): across the +u side at along = 6.
        assert_eq!(ring[7], moon.index(Face::PosY, 6, edge - 1));
    }

    /// The areas: a centre node stands over a corner node by the bend's own ratio (about 0.76 at
    /// the corner, 0.70 at an edge midpoint), and the six faces' areas sum to the sphere's within
    /// the quadrature's parts per ten thousand.
    #[test]
    fn the_areas_sum_to_the_sphere_and_shrink_toward_a_corner() {
        let moon = MacroLattice::of(&home_moon()).expect("a lattice");
        let total: u64 = (0..moon.node_count() as u32).map(|n| moon.area_m2(n)).sum();
        let r = home_moon().ladder().radius_m();
        let sphere = 4.0 * std::f64::consts::PI * r * r;
        let error = (total as f64 - sphere).abs() / sphere;
        assert!(error < 2.0e-4, "area error {error}");
        let mid = moon.edge as i32 / 2;
        let centre = moon.area_m2(moon.index(Face::PosX, mid, mid)) as f64;
        let corner = moon.area_m2(moon.index(Face::PosX, 0, 0)) as f64;
        let edge_mid = moon.area_m2(moon.index(Face::PosX, moon.edge as i32 - 1, mid)) as f64;
        let corner_ratio = corner / centre;
        let edge_ratio = edge_mid / centre;
        assert!(
            (0.74..0.78).contains(&corner_ratio),
            "corner ratio {corner_ratio}"
        );
        assert!(
            (0.69..0.73).contains(&edge_ratio),
            "edge ratio {edge_ratio}"
        );
    }

    /// A chord between neighbours is about the node's size — 8 192 m at the face centre, under it
    /// toward a corner — and a diagonal is √2 of a side; a node's chord to itself is zero.
    #[test]
    fn chords_between_neighbours_are_about_a_node() {
        let moon = MacroLattice::of(&home_moon()).expect("a lattice");
        let mid = moon.edge as i32 / 2;
        let node = moon.index(Face::PosZ, mid, mid);
        let ring = moon.neighbours(node);
        let side = moon.chord_m(node, ring[3]);
        let diagonal = moon.chord_m(node, ring[0]);
        assert!((7_900..=8_300).contains(&side), "side {side}");
        let ratio = f64::from(diagonal) / f64::from(side);
        assert!(
            (ratio - std::f64::consts::SQRT_2).abs() < 0.02,
            "ratio {ratio}"
        );
        assert_eq!(moon.chord_m(node, node), 0);
        let corner = moon.index(Face::PosZ, 0, 0);
        let corner_side = moon.chord_m(corner, moon.neighbours(corner)[4]);
        assert!(
            corner_side < side,
            "corner side {corner_side} under centre side {side}"
        );
    }

    /// The chunk's lookup: rung-0 cell 8 191 is node 0 and cell 8 192 is node 1; at rung 13 one
    /// cell is one node.
    #[test]
    fn a_chunks_cell_finds_its_node_by_one_division() {
        let home = MacroLattice::of(&home_planet()).expect("a lattice");
        assert_eq!(home.node_of_cell(0, 8_191, 8_192), (0, 1));
        assert_eq!(home.node_of_cell(13, 7, 1_215), (7, 1_215));
        assert_eq!(home.node_of_cell(3, 1_023, 1_024), (0, 1));
    }
}
