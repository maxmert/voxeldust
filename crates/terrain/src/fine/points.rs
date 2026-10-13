//! ★ THE FINE LEVELS' POINTS (B1 (a); the way-in design revision 4 §3.1; ruling B83 item 1).
//!
//! A fine level — a rung's cell, `n_l` cells to a face edge — solves its ground on POINTS, never on a grid (B80: the
//! regular grid was the root of the straight lines). The set is the HASHED POISSON-DISK: Matérn's hard-core process,
//! type II (Matérn 1960). Each cell holds [`CANDIDATES`] candidates, each at a place and with a priority drawn from the
//! leaf's integer hash of the seed, the level, the cell and the candidate. A candidate is KEPT unless a candidate of a
//! higher priority (ties by the cell and the candidate's index) stands closer than THAT candidate's radius `r = cell/√π`
//! — the radius that keeps one point a cell on average: `(1 − e^(−λπr²))/(πr²)` = 0.98 at λ = 4. The cell is the
//! higher-ranked candidate's own cell, MEASURED (its chords to its next cells along `i` and `j`), because the bend's cells
//! differ in size across a face: a nominal radius measured 0.83 points a cell near a face edge. Only the higher-ranked
//! candidate of a pair can drop the other, so the rule reads one radius and every tile decides the same.
//!
//! **Every host, every tile, the same points.** A candidate's place is a sub-cell of a `2^sub_bits` lattice inside its
//! cell, and its direction is the recipe's integer direction of that sub-cell (`vd_recipe::bend::direction` at the finer
//! count), so the words are the same on the shard and on every client. Its fate reads only the candidates of its own
//! cell's eight neighbours, each found by stepping from THE CELL'S OWN FACE (a face edge crossed by the seam's one rule,
//! `vd_seed::seam::across`; the diagonal past a cube corner does not exist), so a tile that holds the cell from another
//! face decides the same. The distance is the exact chord between the integer directions, compared in 128-bit integers.
//!
//! **Example.** On the home planet's belt the 128 m cell (face 3, 23 552, 71 168) draws four candidates; the two whose
//! higher-ranked neighbours stand within 72 m of them are dropped, and the other two are the points that cell gives
//! every 128 m tile that holds it, on the shard and on every client.

use vd_recipe::Gi;
use vd_recipe::bend::{direction, inv_n_of};
use vd_recipe::rng::corner_hash;
use vd_seed::bend::Face;
use vd_seed::seam::{Edge, across, partner_of};

/// The candidates drawn in each cell (Matérn's intensity λ in candidates a cell).
pub const CANDIDATES: u8 = 4;
const _: () = assert!(CANDIDATES == 4, "`Level::candidates` lists four indices");

/// The sub-cell lattice a candidate's place lies on, at most: `2^10` steps a cell edge (1.6 cm in a 16 m cell). A level
/// whose cell count would overflow the recipe's 32-bit index takes fewer (`Level::new`).
pub const SUB_BITS_MAX: u32 = 10;

/// `1/(2π)` at 40 fraction bits: a cell's squared Matérn radius is its area over π, and its area the mean of its two
/// squared chords (`(c_i² + c_j²)/2`). 174 992 710 547.97… rounded.
pub const INV_TWO_PI_Q40: i64 = 174_992_710_548;

/// The salt that separates the fine points' hash from every other use of the seed.
pub const POINT_SALT: u64 = 0x5EED_F1E0_9017_0001;

/// A cell of a fine level: its face and its index on that face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellId {
    pub face: u8,
    pub i: i32,
    pub j: i32,
}

/// One candidate: its cell, its index there, its place (the sub-cell inside the cell), its integer direction at the
/// recipe's 40 fraction bits, its priority, and its cell's squared Matérn radius at 80 fraction bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub cell: CellId,
    pub k: u8,
    pub sub: (i32, i32),
    pub dir: [Gi; 3],
    pub priority: u64,
    pub r2: i128,
}

impl Candidate {
    /// The rank Matérn's rule compares: the priority, then the cell, then the index — a total order every host and
    /// every tile reads the same.
    #[must_use]
    pub fn rank(&self) -> (u64, CellId, u8) {
        (self.priority, self.cell, self.k)
    }

    /// The squared chord to another candidate, at 80 fraction bits (exact).
    #[must_use]
    pub fn chord2(&self, other: &Candidate) -> i128 {
        let d = |c: usize| i128::from(self.dir[c].raw() - other.dir[c].raw());
        d(0) * d(0) + d(1) * d(1) + d(2) * d(2)
    }
}

/// A fine level's point rule: the seed, the level, its cells to a face edge, and the numbers every candidate reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    seed: u64,
    level: u8,
    n_l: u32,
    sub_bits: u32,
    inv_n: Gi,
    inv_sub: Gi,
}

impl Level {
    /// The point rule of the level `level` (its rung) with `n_l` cells to a face edge. `None` where the face has fewer
    /// cells than the recipe's reciprocal holds, or so many that not even one sub-cell bit fits the recipe's 32-bit
    /// index.
    #[must_use]
    pub fn new(seed: u64, level: u8, n_l: u32) -> Option<Level> {
        if n_l < vd_recipe::bend::INV_MIN_CELLS {
            return None;
        }
        // The largest sub-cell lattice whose finest index still fits an i32: n_l · 2^sub < 2^31.
        let room = 31 - (32 - n_l.leading_zeros());
        if room == 0 {
            return None;
        }
        let sub_bits = if room < SUB_BITS_MAX {
            room
        } else {
            SUB_BITS_MAX
        };
        Some(Level {
            seed: seed ^ POINT_SALT,
            level,
            n_l,
            sub_bits,
            inv_n: inv_n_of(n_l),
            inv_sub: inv_n_of(n_l << sub_bits),
        })
    }

    /// The cells to a face edge.
    #[must_use]
    pub fn cells_per_edge(&self) -> u32 {
        self.n_l
    }

    /// The sub-cell bits this level's places use.
    #[must_use]
    pub fn sub_bits(&self) -> u32 {
        self.sub_bits
    }

    /// A cell's squared Matérn radius at 80 fraction bits: its area over π, its area the mean of its squared chords to
    /// its next cells along `i` and `j` (cell centres by the recipe's own bend; the next index past a face's last cell
    /// reads the bend's own continuation, a size and never a cell).
    #[must_use]
    pub fn radius2(&self, cell: CellId) -> i128 {
        let at = |i: i32, j: i32| direction(i32::from(cell.face), i, j, self.inv_n);
        let (c, ci, cj) = (
            at(cell.i, cell.j),
            at(cell.i + 1, cell.j),
            at(cell.i, cell.j + 1),
        );
        let chord2 = |a: [Gi; 3], b: [Gi; 3]| {
            let d = |k: usize| i128::from(a[k].raw() - b[k].raw());
            d(0) * d(0) + d(1) * d(1) + d(2) * d(2)
        };
        ((chord2(c, ci) + chord2(c, cj)) * i128::from(INV_TWO_PI_Q40)) >> 40
    }

    /// Candidate `k` of `cell`: its place and priority from ONE hash of the seed, the level, the cell and `k`, and its
    /// direction from the recipe's integer bend at the sub-cell count.
    #[must_use]
    pub fn candidate(&self, cell: CellId, k: u8) -> Candidate {
        self.candidate_in(cell, k, self.radius2(cell))
    }

    /// Candidate `k` of `cell` whose squared radius is already known.
    fn candidate_in(&self, cell: CellId, k: u8, r2: i128) -> Candidate {
        let h = corner_hash(
            self.seed ^ (u64::from(self.level) << 56),
            i64::from(cell.i),
            i64::from(cell.j),
            (i64::from(cell.face) << 8) | i64::from(k),
        );
        let mask = (1u64 << self.sub_bits) - 1;
        let sub = ((h & mask) as i32, ((h >> self.sub_bits) & mask) as i32);
        let priority = h >> (2 * self.sub_bits);
        let dir = direction(
            i32::from(cell.face),
            (cell.i << self.sub_bits) | sub.0,
            (cell.j << self.sub_bits) | sub.1,
            self.inv_sub,
        );
        Candidate {
            cell,
            k,
            sub,
            dir,
            priority,
            r2,
        }
    }

    /// The candidates of `cell` (its radius measured once).
    #[must_use]
    pub fn candidates(&self, cell: CellId) -> [Candidate; CANDIDATES as usize] {
        let r2 = self.radius2(cell);
        [0, 1, 2, 3].map(|k| self.candidate_in(cell, k, r2))
    }

    /// The cell `(di, dj)` from `cell` (each of `−1`, `0`, `1`), stepping on THE CELL'S OWN FACE: across a face edge by
    /// the seam's one rule; `None` for the diagonal past a cube corner, which no face holds.
    #[must_use]
    pub fn neighbour(&self, cell: CellId, di: i32, dj: i32) -> Option<CellId> {
        let face = Face::from_index(cell.face)?;
        chart_cell(face, cell.i + di, cell.j + dj, self.n_l as i32)
    }

    /// Whether `c` survives Matérn's rule against the candidates of its cell and its eight neighbours, `at` giving a
    /// cell's candidates (a cache or a fresh draw: the same words either way).
    #[must_use]
    pub fn kept_with(
        &self,
        c: &Candidate,
        at: &mut impl FnMut(CellId) -> [Candidate; CANDIDATES as usize],
    ) -> bool {
        for dj in -1..=1 {
            for di in -1..=1 {
                let Some(cell) = self.neighbour(c.cell, di, dj) else {
                    continue;
                };
                for q in at(cell) {
                    // Only a higher-ranked candidate drops `c`, by ITS OWN cell's radius.
                    if q.rank() > c.rank() && q.chord2(c) < q.r2 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// The kept points of one cell, every neighbour drawn afresh.
    #[must_use]
    pub fn kept(&self, cell: CellId) -> Vec<Candidate> {
        let mut fresh = |c: CellId| self.candidates(c);
        self.candidates(cell)
            .into_iter()
            .filter(|c| self.kept_with(c, &mut fresh))
            .collect()
    }
}

/// The cell at `(gi, gj)` of `face`'s chart, which runs past the face's edges: inside the face its own cell; past ONE
/// edge the partner face's cell `over − 1` cells in from the shared edge (the seam's one crossing rule, the same as the
/// box's `site_of`); past two edges at once — a cube corner's diagonal region — no cell (`None`).
#[must_use]
pub fn chart_cell(face: Face, gi: i32, gj: i32, n: i32) -> Option<CellId> {
    let in_i = (gi >= 0) & (gi < n);
    let in_j = (gj >= 0) & (gj < n);
    if in_i & in_j {
        return Some(CellId {
            face: face.index(),
            i: gi,
            j: gj,
        });
    }
    if !(in_i | in_j) {
        return None;
    }
    let (side, along, over) = if !in_i {
        if gi < 0 {
            (Edge::UMinus, gj, -gi)
        } else {
            (Edge::UPlus, gj, gi - n + 1)
        }
    } else if gj < 0 {
        (Edge::VMinus, gi, -gj)
    } else {
        (Edge::VPlus, gi, gj - n + 1)
    };
    let c = across(face, side, along, n);
    let rec = partner_of(face, side);
    let inward = if rec.dst_edge.is_plus() {
        -(over - 1)
    } else {
        over - 1
    };
    let (i, j) = if rec.dst_edge.is_u() {
        (c.i + inward, c.j)
    } else {
        (c.i, c.j + inward)
    };
    // A chart may reach past the partner face too (a domain wider than a face); no cell exists there.
    if (i < 0) | (i >= n) | (j < 0) | (j >= n) {
        return None;
    }
    Some(CellId {
        face: c.face.index(),
        i,
        j,
    })
}

/// ★ WHERE A CELL STANDS IN `face`'s CHART — the inverse of [`chart_cell`]: a cell of `face` at its own index; a cell of
/// a partner face across the side the two share, `over` cells past that side; `None` for a cell of the opposite face,
/// which no chart of `face` holds. A chart's cells and a partner face's cells pair one to one, so every host and every
/// tile reads the same place.
///
/// **Example.** On the home planet's 128 m level, the first cell of face 0 across its `+u` side stands in the chart of
/// face 0 one column past that face's last column.
#[must_use]
pub fn chart_of(face: Face, cell: CellId, n: i32) -> Option<(i32, i32)> {
    if cell.face == face.index() {
        return Some((cell.i, cell.j));
    }
    for side in Edge::ALL {
        let rec = partner_of(face, side);
        if rec.dst_face.index() != cell.face {
            continue;
        }
        // The cell's place along the partner's side and its depth in from that side.
        let (along, at) = if rec.dst_edge.is_u() {
            (cell.j, cell.i)
        } else {
            (cell.i, cell.j)
        };
        let depth = if rec.dst_edge.is_plus() {
            n - 1 - at
        } else {
            at
        };
        // A reversed side runs the other way (the committed table holds none — `vd_seed::seam` asserts it at compile
        // time — so the rule is written without a branch no table takes).
        let along = along + i32::from(rec.reversed) * (n - 1 - 2 * along);
        let over = depth + 1;
        let place = match side {
            Edge::UMinus => (-over, along),
            Edge::UPlus => (n - 1 + over, along),
            Edge::VMinus => (along, -over),
            Edge::VPlus => (along, n - 1 + over),
        };
        return Some(place);
    }
    None
}

/// ★ THE KEPT POINTS OF A TILE'S DOMAIN: every cell of the chart rectangle `[i0, i0 + w) × [j0, j0 + h)` of `face`
/// (cells past the face's edges resolved by [`chart_cell`], the cube corner's diagonal region skipped), each candidate
/// tested against its TRUE neighbours (stepped from its own face). Candidates of the chart and a one-cell ring are drawn
/// once and read from the chart where the chart's neighbour is the true neighbour (a cell on `face` away from its edge);
/// elsewhere the neighbour is drawn afresh — the same words. The points come in chart order (rows, then columns, then
/// the candidate's index), so every host lists them the same.
#[must_use]
pub fn domain_points(
    level: &Level,
    face: Face,
    i0: i32,
    j0: i32,
    w: i32,
    h: i32,
) -> Vec<Candidate> {
    let n = level.n_l as i32;
    let (cw, ch) = (w + 2, h + 2);
    let slot = |gi: i32, gj: i32| ((gj - j0 + 1) * cw + (gi - i0 + 1)) as usize;
    let mut chart: Vec<Option<[Candidate; CANDIDATES as usize]>> =
        Vec::with_capacity((cw * ch) as usize);
    for gj in j0 - 1..j0 + h + 1 {
        for gi in i0 - 1..i0 + w + 1 {
            chart.push(chart_cell(face, gi, gj, n).map(|c| level.candidates(c)));
        }
    }
    let mut out = Vec::new();
    for gj in j0..j0 + h {
        for gi in i0..i0 + w {
            let Some(cands) = chart[slot(gi, gj)] else {
                continue;
            };
            // On `face` and a cell away from its edge, the chart's 3 × 3 is the cell's true neighbourhood.
            let interior = (cands[0].cell.face == face.index())
                & (gi > 0)
                & (gi < n - 1)
                & (gj > 0)
                & (gj < n - 1);
            for c in cands {
                let keep = if interior {
                    let mut from_chart = |cell: CellId| {
                        chart[slot(cell.i, cell.j)]
                            .expect("an interior cell's neighbours lie in the chart")
                    };
                    level.kept_with(&c, &mut from_chart)
                } else {
                    let mut fresh = |cell: CellId| level.candidates(cell);
                    level.kept_with(&c, &mut fresh)
                };
                if keep {
                    out.push(c);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! A test may divide (ruling F7's rule is about the shipped path): a mean or a share is stated as its quotient.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states a mean or a share as its quotient; never a kernel's path"
    )]
    use super::*;

    /// The home planet's 128 m level: about 77 800 cells to a face edge (the ladder's rung 7).
    const N_128: u32 = 77_824;

    fn level() -> Level {
        Level::new(0x1234_5678_9ABC_DEF0, 7, N_128).expect("a valid level")
    }

    #[test]
    fn a_level_refuses_too_few_cells_and_shrinks_its_sub_lattice_for_many() {
        assert_eq!(Level::new(1, 7, 2), None);
        assert_eq!(level().sub_bits(), SUB_BITS_MAX);
        // 2^30 cells leave no bit for a sub-cell; 2^25 leave five.
        assert_eq!(Level::new(1, 0, 1 << 30), None);
        assert_eq!(Level::new(1, 0, 1 << 25).expect("valid").sub_bits(), 5);
        assert_eq!(level().cells_per_edge(), N_128);
    }

    #[test]
    fn a_candidate_is_the_same_words_every_time_and_differs_by_seed_and_level() {
        let l = level();
        let cell = CellId {
            face: 3,
            i: 23_552,
            j: 71_168,
        };
        assert_eq!(l.candidate(cell, 2), l.candidate(cell, 2));
        let other_seed = Level::new(0x1234_5678_9ABC_DEF1, 7, N_128).expect("valid");
        assert_ne!(l.candidate(cell, 2).sub, other_seed.candidate(cell, 2).sub);
        let other_level = Level::new(0x1234_5678_9ABC_DEF0, 6, N_128).expect("valid");
        assert_ne!(
            l.candidate(cell, 2).priority,
            other_level.candidate(cell, 2).priority
        );
    }

    #[test]
    fn kept_points_are_one_a_cell_on_average_and_never_closer_than_the_radius() {
        let l = level();
        // Near a face's middle and near its edge, where the bend's cells are smaller: the same density.
        for j0 in [38_000, 71_000] {
            let pts = domain_points(&l, Face::NegY, 23_000, j0, 64, 64);
            // Matérn's density at four candidates a cell and r = cell/√π: 0.98 a cell, give or take its spread.
            let share = pts.len() as f64 / (64.0 * 64.0);
            assert!((0.93..=1.03).contains(&share), "{share} points a cell");
            for (a, p) in pts.iter().enumerate() {
                for q in &pts[a + 1..] {
                    let higher = if p.rank() > q.rank() { p } else { q };
                    assert!(
                        p.chord2(q) >= higher.r2,
                        "two kept points within the higher one's radius"
                    );
                }
            }
        }
    }

    #[test]
    fn a_domain_lists_each_cells_own_kept_points() {
        let l = level();
        let pts = domain_points(&l, Face::PosX, 500, 700, 8, 8);
        for gj in 700..708 {
            for gi in 500..508 {
                let cell = CellId {
                    face: Face::PosX.index(),
                    i: gi,
                    j: gj,
                };
                let mine: Vec<Candidate> = pts.iter().copied().filter(|p| p.cell == cell).collect();
                assert_eq!(mine, l.kept(cell));
            }
        }
    }

    #[test]
    fn across_a_face_edge_both_faces_keep_the_same_points() {
        // A small level so the domains reach across +X's +u edge into +Y.
        let l = Level::new(77, 7, 64).expect("valid");
        let from_x = domain_points(&l, Face::PosX, 56, 20, 16, 8);
        // Every point the +X chart found on +Y is a point a chart on +Y itself keeps there.
        let on_y: Vec<Candidate> = from_x
            .iter()
            .copied()
            .filter(|p| p.cell.face == Face::PosY.index())
            .collect();
        assert!(!on_y.is_empty(), "the chart reaches +Y");
        for p in &on_y {
            assert!(l.kept(p.cell).contains(p), "the same point on both faces");
        }
        // And the cells the +X chart resolved on +Y keep exactly their own kept set.
        let mut cells: Vec<CellId> = on_y.iter().map(|p| p.cell).collect();
        cells.sort_unstable();
        cells.dedup();
        for cell in cells {
            let here: Vec<Candidate> = from_x.iter().copied().filter(|p| p.cell == cell).collect();
            assert_eq!(here, l.kept(cell));
        }
    }

    #[test]
    fn a_face_corner_cell_has_seven_neighbours_and_the_corner_region_holds_no_cell() {
        let l = Level::new(5, 7, 64).expect("valid");
        let corner = CellId {
            face: Face::PosZ.index(),
            i: 0,
            j: 0,
        };
        let mut count = 0;
        for dj in -1..=1 {
            for di in -1..=1 {
                if (di, dj) != (0, 0) && l.neighbour(corner, di, dj).is_some() {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 7);
        assert_eq!(chart_cell(Face::PosZ, -1, -1, 64), None);
        assert_eq!(chart_cell(Face::PosZ, 64, 70, 64), None);
        // A chart that runs past the partner face too finds no cell there.
        assert_eq!(chart_cell(Face::PosZ, 200, 3, 64), None);
        // A face index past the six refuses.
        assert_eq!(
            l.neighbour(
                CellId {
                    face: 9,
                    i: 0,
                    j: 0
                },
                1,
                0
            ),
            None
        );
    }

    #[test]
    fn a_domain_over_a_cube_corner_skips_the_corner_region() {
        let l = Level::new(5, 7, 64).expect("valid");
        let pts = domain_points(&l, Face::PosZ, -2, -2, 4, 4);
        // The 2 × 2 cells past both edges hold nothing; the rest lie on +Z and its two partners.
        assert!(!pts.is_empty());
        assert!(
            pts.iter()
                .all(|p| p.cell.face != Face::PosZ.index() || (p.cell.i < 2 && p.cell.j < 2))
        );
    }

    #[test]
    fn a_cells_place_in_a_chart_is_the_inverse_of_the_charts_cell() {
        // Every chart cell of every face that holds a cell finds its place back, and every cell of the cube that a face's
        // chart holds stands at a place whose cell it is; the opposite face's cells stand nowhere.
        let n = 8;
        for face in Face::ALL {
            let mut held = 0;
            for gj in -n..2 * n {
                for gi in -n..2 * n {
                    if let Some(c) = chart_cell(face, gi, gj, n) {
                        assert_eq!(chart_of(face, c, n), Some((gi, gj)));
                        held += 1;
                    }
                }
            }
            // The face and its four partners, each whole.
            assert_eq!(held, 5 * n * n);
            for f in 0..6u8 {
                for j in 0..n {
                    for i in 0..n {
                        let c = CellId { face: f, i, j };
                        match chart_of(face, c, n) {
                            Some((gi, gj)) => assert_eq!(chart_cell(face, gi, gj, n), Some(c)),
                            None => {
                                // Neither the face nor one of its partners: the opposite face.
                                assert_ne!(f, face.index());
                                assert!(
                                    Edge::ALL
                                        .iter()
                                        .all(|&s| partner_of(face, s).dst_face.index() != f)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_side_of_a_face_crosses_to_a_cell_of_its_partner() {
        for face in Face::ALL {
            for (gi, gj) in [
                (-1, 10),
                (64, 10),
                (10, -1),
                (10, 64),
                (-3, 10),
                (66, 10),
                (10, -3),
                (10, 66),
            ] {
                let c = chart_cell(face, gi, gj, 64).expect("one edge crossed");
                assert_ne!(c.face, face.index());
                assert!((0..64).contains(&c.i) && (0..64).contains(&c.j));
            }
        }
    }
}
