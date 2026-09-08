//! ★ THE ROUND ARM — a planet's, a moon's and a large round asteroid's grid: six bent faces over a
//! radial band.
//!
//! **The ladder** (ruling V6 A4). A body with `T` rungs has `N = q · 2^(T−1)` cells along a face edge,
//! so every rung tiles the face exactly, and its snapped radius is `R = 2N/π`. `T` comes from `N`: the
//! top rung is the FIRST rung at which a face is at most `64 × 64` chunks. An Earth-sized body lands
//! within 0.01 % of its seed radius (MEASURED against the rule: 646 m on 6 371 km).
//!
//! **The radial band.** `k` counts cells up from the floor radius `R − D_crust`. At rung `L` the band
//! holds `⌊band / 2^L⌋` whole cells and NOTHING above them: a coarse rung loses the partial top slice,
//! so every cell any operation names is a cell every other operation agrees exists. A radial cell is
//! exactly `2^L` m tall at every altitude, so `k` is an integer subtraction and a shift. A chunk is a
//! frustum; only index-space code knows it.
//!
//! **Refusals, never garbage.** A position at the body's centre, a non-finite position, a rung the
//! body does not carry and an index outside the band all answer `None`/`Outside`. Nothing here turns a
//! division by zero into a cell.
//!
//! **Example.** The home planet's seed draws 6 371 000 m. The ladder snaps the cell count to a
//! multiple of 4 096 and the planet states its look at 6 370 354 m. A pilot who watches the planet
//! grow from a dot sees that sphere; the first chunks at rung 12 sit exactly on it, so nothing pops.

use glam::DVec3;

use crate::pose::{LatticePos, Tier};

use super::addr::Rung;
use super::bend::{Face, direction, face_coords, face_of, unbend};
use super::seam::{Edge, SeamRecord, partner_of};
use super::{Dir6, Step, metres_of};

// THE LADDER ARITHMETIC lives in the `vd-seed` leaf since slice 5 (ruling V9 S5-1), so the generator
// and this grid snap a body, name a cell's radius and its face parameter with ONE implementation.
use vd_seed::ladder::{self, Ladder};

/// The radial band a body carries below and above its snapped surface, in whole metres. Seed-derived
/// per body; PROVISIONAL derivation here until the generator crate's body definition owns it (the
/// world identity, ruling V6 Format D).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandParams {
    /// Metres of crust below the surface that cells cover.
    pub crust_m: u32,
    /// Metres above the surface that cells cover (mountains, atmosphere, builds).
    pub above_m: u32,
}

impl BandParams {
    /// A provisional band from the radius alone: `0.13 %` of the radius each way (Earth-like:
    /// 8.3 km of crust, 8.3 km above), never under 64 m so a small body still has room to build, and
    /// never over the 18-bit `k` budget. The generator's body definition replaces this.
    #[must_use]
    pub fn provisional(radius_m: f64) -> BandParams {
        let each = (radius_m * 0.0013).ceil();
        let clamped = each.clamp(64.0, 100_000.0);
        BandParams {
            crust_m: clamped as u32,
            above_m: clamped as u32,
        }
    }
}

/// A cube-sphere grid: the cell count, the band and the rung count of one body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShellGrid {
    ladder: Ladder,
}

impl ShellGrid {
    /// The grid of a body with this seed radius and band, by the ladder rule. `None` when the body has
    /// no shell grid: a non-positive or non-finite radius; a body too large for the address (`N > 2²⁶`,
    /// a radius above 42 723 km — a gas giant); or a crust deeper than the radius, because a band that
    /// reaches the centre is not a shell and the centre has no direction.
    #[must_use]
    pub fn for_body(radius_m: f64, band: BandParams) -> Option<ShellGrid> {
        Ladder::for_radius(radius_m, band.crust_m, band.above_m).map(|ladder| ShellGrid { ladder })
    }

    /// Cells along a face edge at rung 0.
    #[must_use]
    pub const fn n(&self) -> u32 {
        self.ladder.n
    }

    /// The floor radius in whole metres (`k = 0`).
    #[must_use]
    pub const fn floor_m(&self) -> u32 {
        self.ladder.floor_m
    }

    /// The band's thickness in whole metres.
    #[must_use]
    pub const fn band_m(&self) -> u32 {
        self.ladder.band_m
    }

    /// How many rungs this body carries; a valid rung is `< rungs`.
    #[must_use]
    pub const fn rungs(&self) -> u8 {
        self.ladder.rungs
    }

    /// The snapped radius `2N/π`, the sphere the realm draws itself at.
    #[must_use]
    pub fn radius_m(&self) -> f64 {
        self.ladder.radius_m()
    }

    /// Cells along a face edge at `rung` (exact: `N` is a multiple of `2^(rungs−1)`).
    #[must_use]
    pub const fn cells_per_edge(&self, rung: Rung) -> u32 {
        self.ladder.cells_per_edge(rung.level())
    }

    /// Whole radial cells in the band at `rung`. The partial top slice does NOT count: a cell every
    /// operation can name is a cell every operation agrees exists.
    #[must_use]
    pub const fn cells_in_band(&self, rung: Rung) -> u32 {
        self.ladder.cells_in_band(rung.level())
    }

    /// Whether `(i, j, k)` at `rung` is a cell this body has.
    #[must_use]
    pub fn holds(&self, rung: Rung, i: i32, j: i32, k: i32) -> bool {
        self.ladder.holds(rung.level(), i, j, k)
    }

    /// The cell holding `pos` at `rung`: `None` at the centre, for a non-finite position, below the
    /// floor, at or above the rung's ceiling, or at a rung this body does not carry.
    #[must_use]
    pub fn addr_of(&self, pos: LatticePos, rung: Rung) -> Option<(Face, i32, i32, i32)> {
        if rung.level() >= self.ladder.rungs {
            return None;
        }
        let p = metres_of(pos);
        let r = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        // The centre has no direction, and a diverged position has no cell. A NaN fails the first
        // test, so the division below never sees one.
        if !r.is_finite() || r <= 0.0 {
            return None;
        }
        let floor = f64::from(self.ladder.floor_m);
        let cell = f64::from(rung.cell_m());
        let ceiling = floor + f64::from(self.cells_in_band(rung)) * cell;
        if r < floor || r >= ceiling {
            return None;
        }
        let d = [p.x / r, p.y / r, p.z / r];
        let face = face_of(d);
        let (t, s) = face_coords(face, d);
        let n_l = self.cells_per_edge(rung);
        let i = ladder::index_of(unbend(t), n_l);
        let j = ladder::index_of(unbend(s), n_l);
        let k = ((r - floor) / cell).floor() as i32;
        Some((face, i, j, k))
    }

    /// The centre of cell `(face, i, j, k)` at `rung`; `None` for a cell this body does not have.
    #[must_use]
    pub fn cell_center(
        &self,
        face: Face,
        i: i32,
        j: i32,
        k: i32,
        rung: Rung,
    ) -> Option<LatticePos> {
        if !self.holds(rung, i, j, k) {
            return None;
        }
        let n_l = self.cells_per_edge(rung);
        let a = ladder::face_param(i, n_l);
        let b = ladder::face_param(j, n_l);
        let r = self.ladder.cell_radius_m(k, rung.level());
        Some(self.at(face, a, b, r))
    }

    /// The eight corners of cell `(face, i, j, k)` at `rung`, low corner first, `i` fastest; `None`
    /// for a cell this body does not have.
    #[must_use]
    pub fn cell_corners(
        &self,
        face: Face,
        i: i32,
        j: i32,
        k: i32,
        rung: Rung,
    ) -> Option<[LatticePos; 8]> {
        if !self.holds(rung, i, j, k) {
            return None;
        }
        let n_l = self.cells_per_edge(rung);
        let corner = |di: i32, dj: i32, dk: i32| {
            let a = ladder::corner_param(i + di, n_l);
            let b = ladder::corner_param(j + dj, n_l);
            let r = self.ladder.corner_radius_m(k + dk, rung.level());
            self.at(face, a, b, r)
        };
        Some([
            corner(0, 0, 0),
            corner(1, 0, 0),
            corner(0, 1, 0),
            corner(1, 1, 0),
            corner(0, 0, 1),
            corner(1, 0, 1),
            corner(0, 1, 1),
            corner(1, 1, 1),
        ])
    }

    /// The outward unit direction at the centre of cell `(face, i, j, k)` — the grid's radial, which
    /// is NOT gravity (gravity is a function the realm states); `None` for a cell this body does not
    /// have.
    #[must_use]
    pub fn outward(&self, face: Face, i: i32, j: i32, k: i32, rung: Rung) -> Option<[f64; 3]> {
        if !self.holds(rung, i, j, k) {
            return None;
        }
        let n_l = self.cells_per_edge(rung);
        let a = ladder::face_param(i, n_l);
        let b = ladder::face_param(j, n_l);
        Some(direction(face, a, b))
    }

    /// The neighbour of `(face, i, j, k)` in `dir` at `rung`: the next cell on the same face, the
    /// partner cell across a face edge (with whether the along-axis swapped), or `Outside` — past the
    /// band, or from a cell this body does not have.
    #[must_use]
    pub fn neighbor(&self, rung: Rung, face: Face, i: i32, j: i32, k: i32, dir: Dir6) -> Step {
        if !self.holds(rung, i, j, k) {
            return Step::Outside;
        }
        let n_l = self.cells_per_edge(rung) as i32;
        let band = self.cells_in_band(rung) as i32;
        match dir {
            Dir6::KPlus => {
                if k + 1 < band {
                    Step::Same(i, j, k + 1)
                } else {
                    Step::Outside
                }
            }
            Dir6::KMinus => {
                if k > 0 {
                    Step::Same(i, j, k - 1)
                } else {
                    Step::Outside
                }
            }
            Dir6::IPlus => {
                if i + 1 < n_l {
                    Step::Same(i + 1, j, k)
                } else {
                    self.across(face, Edge::UPlus, j, k, n_l)
                }
            }
            Dir6::IMinus => {
                if i > 0 {
                    Step::Same(i - 1, j, k)
                } else {
                    self.across(face, Edge::UMinus, j, k, n_l)
                }
            }
            Dir6::JPlus => {
                if j + 1 < n_l {
                    Step::Same(i, j + 1, k)
                } else {
                    self.across(face, Edge::VPlus, i, k, n_l)
                }
            }
            Dir6::JMinus => {
                if j > 0 {
                    Step::Same(i, j - 1, k)
                } else {
                    self.across(face, Edge::VMinus, i, k, n_l)
                }
            }
        }
    }

    /// The cell across `edge` of `face`, at along-edge position `along`.
    fn across(&self, face: Face, edge: Edge, along: i32, k: i32, n_l: i32) -> Step {
        across_with(partner_of(face, edge), edge, along, k, n_l)
    }

    /// The position at face coordinates `(a, b)` on `face`, at radius `r`.
    fn at(&self, face: Face, a: f64, b: f64, r: f64) -> LatticePos {
        let d = direction(face, a, b);
        LatticePos::from_metres(DVec3::new(d[0] * r, d[1] * r, d[2] * r), Tier::Fine)
    }
}

/// The cell across a seam, given the seam record: the partner face's cell on its own side of the
/// shared edge, at the same (or reversed) position along it. Takes the record so the reversed arm
/// stays exercised even though the frozen basis never reverses a seam (asserted in `seam.rs`).
fn across_with(rec: SeamRecord, edge: Edge, along: i32, k: i32, n_l: i32) -> Step {
    let along = if rec.reversed { n_l - 1 - along } else { along };
    let edge_index = if rec.dst_edge.is_plus() { n_l - 1 } else { 0 };
    let (i, j) = if rec.dst_edge.is_u() {
        (edge_index, along)
    } else {
        (along, edge_index)
    };
    Step::Across {
        face: rec.dst_face,
        i,
        j,
        k,
        axis_swap: edge.is_u() != rec.dst_edge.is_u(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::I64Vec3;
    use std::f64::consts::FRAC_2_PI;
    use vd_seed::ladder::{CHUNK_EDGE, TOP_RUNG_CHUNKS};

    fn earth() -> ShellGrid {
        ShellGrid::for_body(6_371_000.0, BandParams::provisional(6_371_000.0)).expect("earth")
    }

    /// A small body with exactly 62 cells per edge (one chunk per face, one rung): the exhaustive
    /// table body. Radius `2·62/π ≈ 39.5 m`.
    fn tiny() -> ShellGrid {
        ShellGrid {
            ladder: Ladder {
                n: 62,
                floor_m: 20,
                band_m: 40,
                rungs: 1,
            },
        }
    }

    /// A step's parts, total: a same-face step keeps `PosX` and its indices; `Outside` is all
    /// minus one. No test needs an unreachable arm.
    fn parts(step: Step) -> (Face, i32, i32, i32, bool) {
        match step {
            Step::Across {
                face,
                i,
                j,
                k,
                axis_swap,
            } => (face, i, j, k, axis_swap),
            Step::Same(i, j, k) => (Face::PosX, i, j, k, false),
            Step::Outside => (Face::PosX, -1, -1, -1, false),
        }
    }

    #[test]
    fn the_ladder_snaps_known_bodies_within_half_a_ladder_unit() {
        // U-5's rule table, MEASURED against the crate (ruling V6 A4).
        let cases: [(f64, u32, u8, f64); 5] = [
            (500.0, 785, 1, 500.0),
            (200_000.0, 314_112, 8, 199_970.0),
            (1_737_000.0, 2_728_960, 11, 1_737_310.0),
            (6_371_000.0, 10_006_528, 13, 6_370_354.0),
            (12_742_000.0, 20_013_056, 14, 12_740_707.0),
        ];
        for (r, n, rungs, snapped) in cases {
            let g = ShellGrid::for_body(r, BandParams::provisional(r)).expect("body");
            assert_eq!(g.n(), n, "N for radius {r}");
            assert_eq!(g.rungs(), rungs, "rungs for radius {r}");
            let got = g.radius_m();
            assert!((got - snapped).abs() < 1.0, "snapped radius for {r}: {got}");
            // The snap is to the NEAREST multiple of the ladder unit, so the error is at most half a
            // unit: 0.32 m on a one-rung asteroid, 1.3 km on a fourteen-rung super-Earth (0.01 %).
            let unit_m = f64::from(1u32 << (rungs - 1)) * FRAC_2_PI;
            assert!(
                (got - r).abs() <= unit_m / 2.0 + 1e-6,
                "within half a ladder unit for {r}"
            );
            // The band sits around the snapped surface.
            let band = BandParams::provisional(r);
            assert_eq!(
                u64::from(g.floor_m()) + u64::from(band.crust_m),
                got.round() as u64
            );
            assert_eq!(g.band_m(), band.crust_m + band.above_m);
            // Every rung tiles the face exactly, and the top rung is the FIRST at which a face is at
            // most 64 chunks across.
            for level in 0..rungs {
                let rung = Rung::new(level).expect("rung");
                assert_eq!(
                    g.cells_per_edge(rung) << level,
                    n,
                    "rung {level} tiles the face"
                );
            }
            let top = Rung::new(rungs - 1).expect("top");
            assert!(u64::from(g.cells_per_edge(top)) <= CHUNK_EDGE * TOP_RUNG_CHUNKS);
            let below_top = rungs.checked_sub(2).map(|l| Rung::new(l).expect("rung"));
            let over =
                below_top.map(|r| u64::from(g.cells_per_edge(r)) > CHUNK_EDGE * TOP_RUNG_CHUNKS);
            assert_ne!(
                over,
                Some(false),
                "the rung below the top is over 64 chunks for {r}"
            );
        }
    }

    #[test]
    fn at_an_octave_boundary_the_top_rung_is_read_from_the_snapped_count() {
        // Radius 40 427.67 m: the ideal count 63 504 asks for a 32-cell unit, the nearest multiple
        // 63 488 is exactly 16 chunks × 62 × 64 at rung 4, so the top rung is 4 and the body has FIVE
        // rungs, not six (the refuter's F6, at every octave).
        let pinned =
            ShellGrid::for_body(40_427.67, BandParams::provisional(40_427.67)).expect("body");
        assert_eq!((pinned.n(), pinned.rungs()), (63_488, 5));
        // At every octave boundary and just past it, the rule holds for the frozen N: the top rung
        // is at most 64 chunks across, the rung below it is more, and N tiles the top rung.
        let mut r = 40_427.67;
        while r < 4.0e7 {
            for radius in [r, r * 1.000_1, r * 0.999_9] {
                let g = ShellGrid::for_body(radius, BandParams::provisional(radius)).expect("body");
                let top = Rung::new(g.rungs() - 1).expect("top");
                assert!(u64::from(g.cells_per_edge(top)) <= CHUNK_EDGE * TOP_RUNG_CHUNKS);
                let below = Rung::new(g.rungs() - 2).expect("below");
                assert!(
                    u64::from(g.cells_per_edge(below)) > CHUNK_EDGE * TOP_RUNG_CHUNKS,
                    "the rung below the top is over 64 chunks for {radius}"
                );
                assert_eq!(
                    g.n() % (1 << (g.rungs() - 1)),
                    0,
                    "N tiles the top rung for {radius}"
                );
            }
            r *= 2.0;
        }
    }

    #[test]
    fn a_body_too_large_or_ill_formed_has_no_grid() {
        let band = BandParams::provisional(1.0e8);
        assert_eq!(
            ShellGrid::for_body(1.0e8, band),
            None,
            "a gas giant exceeds the address"
        );
        assert_eq!(ShellGrid::for_body(0.0, band), None);
        assert_eq!(ShellGrid::for_body(-5.0, band), None);
        assert_eq!(ShellGrid::for_body(f64::NAN, band), None);
        assert_eq!(ShellGrid::for_body(f64::INFINITY, band), None);
        assert_eq!(
            ShellGrid::for_body(1.0e300, band),
            None,
            "an absurd radius is refused, not overflowed"
        );
        // Sixteen rungs but too many cells: 50 000 km is the super-Earth the address cannot hold.
        assert_eq!(
            ShellGrid::for_body(5.0e7, band),
            None,
            "past the cell-count ceiling"
        );
        // A band whose two halves overflow the width is refused, not wrapped.
        let huge = BandParams {
            crust_m: 1_000,
            above_m: u32::MAX,
        };
        assert_eq!(ShellGrid::for_body(500_000.0, huge), None);
        // A crust as deep as the radius reaches the centre: not a shell.
        let deep = BandParams {
            crust_m: 5_000,
            above_m: 10,
        };
        assert_eq!(ShellGrid::for_body(500.0, deep), None);
        assert_eq!(
            ShellGrid::for_body(4_999.0, deep),
            None,
            "exactly the surface"
        );
        // A pebble: the ideal cell count rounds to zero and the ladder floors it at one cell.
        let pebble = ShellGrid::for_body(
            0.2,
            BandParams {
                crust_m: 0,
                above_m: 3,
            },
        )
        .expect("pebble");
        assert_eq!((pebble.n(), pebble.rungs()), (1, 1));
        // The largest legal body: N = 2^26 exactly at radius 2N/π.
        let r = f64::from(1u32 << 26) * FRAC_2_PI;
        let g = ShellGrid::for_body(r, BandParams::provisional(r)).expect("largest legal body");
        assert_eq!(g.n(), 1 << 26);
        assert_eq!(
            g.rungs(),
            16,
            "rungs 0..=15: sixteen of them, the address's four bits"
        );
        assert_eq!(
            g.cells_per_edge(Rung::new(15).expect("rung")),
            2048,
            "34 chunks across at the top (62 × 33 = 2 046 < 2 048)"
        );
    }

    #[test]
    fn the_provisional_band_is_clamped_at_both_ends() {
        assert_eq!(
            BandParams::provisional(1_000.0),
            BandParams {
                crust_m: 64,
                above_m: 64
            }
        );
        assert_eq!(BandParams::provisional(6_371_000.0).crust_m, 8_283);
        assert_eq!(BandParams::provisional(1.0e9).above_m, 100_000);
    }

    #[test]
    fn the_round_trip_holds_over_the_whole_tiny_body_at_its_one_rung() {
        // G-MAPPING-ROUNDTRIP on the table body: every cell's centre names the cell back, at the
        // floor, in the middle and in the top cell.
        let g = tiny();
        let rung = Rung::ZERO;
        let top = g.cells_in_band(rung) as i32 - 1;
        for face in Face::ALL {
            for i in 0..62 {
                for j in 0..62 {
                    for k in [0, 17, top] {
                        let c = g.cell_center(face, i, j, k, rung).expect("a cell it has");
                        assert_eq!(
                            g.addr_of(c, rung),
                            Some((face, i, j, k)),
                            "{face:?} ({i},{j},{k})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_round_trip_holds_at_the_largest_legal_body_at_every_rung_floor_and_top() {
        // G-MAPPING-ROUNDTRIP at N = 2^26: every rung the body carries, sampled around a ≈ ±0.85
        // (the inverse's worst residual) and at the face edges, at the band's floor and its top cell.
        let r = f64::from(1u32 << 26) * FRAC_2_PI;
        let g = ShellGrid::for_body(r, BandParams::provisional(r)).expect("largest");
        for level in 0..g.rungs() {
            let rung = Rung::new(level).expect("rung");
            let n_l = g.cells_per_edge(rung) as i32;
            let top = g.cells_in_band(rung) as i32 - 1;
            assert!(
                top >= 0,
                "every rung of this body holds at least one radial cell"
            );
            let picks = [
                0,
                1,
                (n_l as f64 * 0.075) as i32,
                (n_l as f64 * 0.925) as i32,
                n_l / 2,
                n_l - 2,
                n_l - 1,
            ];
            for face in Face::ALL {
                for &i in &picks {
                    for &j in &picks {
                        for k in [0, top] {
                            let c = g.cell_center(face, i, j, k, rung).expect("a cell it has");
                            assert_eq!(
                                g.addr_of(c, rung),
                                Some((face, i, j, k)),
                                "{face:?} rung {level} ({i},{j},{k})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_partial_top_slice_of_a_coarse_rung_is_not_a_cell_anywhere() {
        // The refuter's F3: the home planet's band is 16 566 m; at rung 12 (4 096 m cells) it holds
        // FOUR whole cells and the 182 m above them are nobody's. Every operation agrees.
        let g = earth();
        let rung = Rung::new(12).expect("rung");
        assert_eq!(g.cells_in_band(rung), 4);
        assert_eq!(
            g.cell_center(Face::PosZ, 5, 5, 4, rung),
            None,
            "k = 4 is not a cell"
        );
        assert_eq!(g.cell_corners(Face::PosZ, 5, 5, 4, rung), None);
        assert_eq!(g.outward(Face::PosZ, 5, 5, 4, rung), None);
        assert_eq!(
            g.neighbor(rung, Face::PosZ, 5, 5, 3, Dir6::KPlus),
            Step::Outside,
            "the top cell has no cell above it"
        );
        assert_eq!(
            g.neighbor(rung, Face::PosZ, 5, 5, 4, Dir6::KMinus),
            Step::Outside
        );
        // A position in the lost slice is refused at rung 12 and named at rung 0.
        let r = f64::from(g.floor_m()) + 4.0 * 4096.0 + 100.0;
        let p = LatticePos::from_metres(DVec3::new(0.0, 0.0, r), Tier::Fine);
        assert_eq!(g.addr_of(p, rung), None);
        assert!(g.addr_of(p, Rung::ZERO).is_some());
    }

    #[test]
    fn positions_outside_the_band_at_the_centre_or_diverged_are_refused() {
        let g = tiny();
        let below = LatticePos::from_metres(DVec3::new(10.0, 0.0, 0.0), Tier::Fine);
        assert_eq!(g.addr_of(below, Rung::ZERO), None, "below the floor");
        let above = LatticePos::from_metres(DVec3::new(60.0, 0.0, 0.0), Tier::Fine);
        assert_eq!(g.addr_of(above, Rung::ZERO), None, "at the ceiling");
        assert_eq!(
            g.addr_of(LatticePos::ORIGIN, Rung::ZERO),
            None,
            "the body's centre has no direction"
        );
        // The refuter's F1: a grid whose floor reaches the centre is refused at construction, so the
        // centre can never pass a band test; and a diverged position is refused, never filed.
        let nan = LatticePos::at(I64Vec3::ZERO, DVec3::new(f64::NAN, 0.0, 0.0));
        assert_eq!(
            g.addr_of(nan, Rung::ZERO),
            None,
            "a NaN position names no cell"
        );
        let inf = LatticePos::at(I64Vec3::ZERO, DVec3::new(f64::INFINITY, 0.0, 0.0));
        assert_eq!(
            g.addr_of(inf, Rung::ZERO),
            None,
            "an infinite position names no cell"
        );
        let fine = LatticePos::from_metres(DVec3::new(30.0, 1.0, 1.0), Tier::Fine);
        assert_eq!(
            g.addr_of(fine, Rung::new(1).expect("rung")),
            None,
            "a rung this body lacks"
        );
        assert_eq!(
            g.neighbor(
                Rung::new(1).expect("rung"),
                Face::PosX,
                0,
                0,
                0,
                Dir6::IPlus
            ),
            Step::Outside
        );
        assert_eq!(
            g.cell_center(Face::PosX, 0, 0, 0, Rung::new(1).expect("rung")),
            None
        );
        assert_eq!(
            g.cell_center(Face::PosX, 62, 0, 0, Rung::ZERO),
            None,
            "past the face"
        );
        assert_eq!(g.cell_center(Face::PosX, 0, -1, 0, Rung::ZERO), None);
        assert_eq!(
            ladder::index_of(1.0, 62),
            61,
            "the far edge names the last cell"
        );
        assert_eq!(ladder::index_of(-1.0, 62), 0);
    }

    #[test]
    fn the_face_edges_round_trip_across_the_seam_and_the_corners_close_in_three() {
        // G-MAPPING-TABLE on the table body, exhaustive: every edge cell's step across the seam
        // lands on the partner face's edge cell, and stepping back returns (involution); at the
        // eight corners the three corner cells form a closed triangle.
        let g = tiny();
        let rung = Rung::ZERO;
        let n = 62;
        let back_dir = |edge: Edge| match edge {
            Edge::UMinus => Dir6::IMinus,
            Edge::UPlus => Dir6::IPlus,
            Edge::VMinus => Dir6::JMinus,
            Edge::VPlus => Dir6::JPlus,
        };
        let mut crossings = 0;
        for face in Face::ALL {
            for along in 0..n {
                for (edge, dir, i, j) in [
                    (Edge::UMinus, Dir6::IMinus, 0, along),
                    (Edge::UPlus, Dir6::IPlus, n - 1, along),
                    (Edge::VMinus, Dir6::JMinus, along, 0),
                    (Edge::VPlus, Dir6::JPlus, along, n - 1),
                ] {
                    let step = g.neighbor(rung, face, i, j, 5, dir);
                    // The expected partner cell, restated from the record without branches: the
                    // along index mirrors when reversed, the edge index is 0 or n−1, and the axes
                    // swap when the two sides bound different axes.
                    let rec = partner_of(face, edge);
                    let along2 = along + i32::from(rec.reversed) * (n - 1 - 2 * along);
                    let edge_index = i32::from(rec.dst_edge.is_plus()) * (n - 1);
                    let (i2, j2) = if rec.dst_edge.is_u() {
                        (edge_index, along2)
                    } else {
                        (along2, edge_index)
                    };
                    let axis_swap = edge.is_u() != rec.dst_edge.is_u();
                    let f2 = rec.dst_face;
                    assert_eq!(
                        step,
                        Step::Across {
                            face: f2,
                            i: i2,
                            j: j2,
                            k: 5,
                            axis_swap
                        },
                        "{face:?}/{edge:?} at {along} crosses to its partner cell"
                    );
                    // Stepping back across the partner's side returns to the same cell.
                    let back = g.neighbor(rung, f2, i2, j2, 5, back_dir(rec.dst_edge));
                    assert_eq!(
                        back,
                        Step::Across {
                            face,
                            i,
                            j,
                            k: 5,
                            axis_swap
                        },
                        "involution at {face:?}/{edge:?} {along}"
                    );
                    // The two cells share a physical edge: their centres are one cell apart, which
                    // no restatement of the rule can fake.
                    let a = metres_of(g.cell_center(face, i, j, 5, rung).expect("cell"));
                    let b = metres_of(g.cell_center(f2, i2, j2, 5, rung).expect("cell"));
                    let gap = (a - b).length();
                    assert!(gap > 0.3, "seam neighbours are a cell apart, got {gap} m");
                    assert!(gap < 1.6, "seam neighbours are a cell apart, got {gap} m");
                    crossings += 1;
                }
            }
        }
        assert_eq!(
            crossings,
            6 * 4 * 62,
            "every directed edge cell crossed once"
        );
        // Corners: from a face's corner cell, its two seam directions reach two corner cells whose
        // own seam directions close the triangle.
        let corner_dirs = [
            (0, 0, Dir6::IMinus, Dir6::JMinus),
            (0, n - 1, Dir6::IMinus, Dir6::JPlus),
            (n - 1, 0, Dir6::IPlus, Dir6::JMinus),
            (n - 1, n - 1, Dir6::IPlus, Dir6::JPlus),
        ];
        // "On an edge" without a short-circuit: an index hits 0 or n−1.
        let edge_hits = |x: i32| usize::from(x == 0) + usize::from(x == n - 1);
        let is_corner = |i: i32, j: i32| edge_hits(i) * edge_hits(j) == 1;
        for face in Face::ALL {
            for (i, j, d1, d2) in corner_dirs {
                let (fa, ia, ja, _, _) = parts(g.neighbor(rung, face, i, j, 0, d1));
                let (fb, ib, jb, _, _) = parts(g.neighbor(rung, face, i, j, 0, d2));
                assert!(is_corner(ia, ja), "a corner's seam neighbour is a corner");
                assert!(is_corner(ib, jb), "a corner's seam neighbour is a corner");
                assert_ne!(fa, fb, "two different faces meet here");
                // The third edge: one of fa's corner cell's seam steps lands on fb's corner cell.
                let mut closed = false;
                for d in [Dir6::IMinus, Dir6::IPlus, Dir6::JMinus, Dir6::JPlus] {
                    let (f3, i3, j3, _, _) = parts(g.neighbor(rung, fa, ia, ja, 0, d));
                    closed |= (f3, i3, j3) == (fb, ib, jb);
                }
                assert!(
                    closed,
                    "the corner at {face:?} ({i},{j}) closes in three steps"
                );
            }
        }
    }

    #[test]
    fn the_parts_helper_is_total_and_a_reversed_record_mirrors_the_along_index() {
        assert_eq!(parts(Step::Same(1, 2, 3)), (Face::PosX, 1, 2, 3, false));
        assert_eq!(parts(Step::Outside), (Face::PosX, -1, -1, -1, false));
        // The frozen basis never reverses a seam (asserted at compile time in seam.rs), so the
        // reversed arm is driven with a synthetic record: along 3 of 62 becomes 58 on the partner.
        let rec = SeamRecord {
            dst_face: Face::NegZ,
            dst_edge: Edge::VPlus,
            reversed: true,
        };
        assert_eq!(
            across_with(rec, Edge::UMinus, 3, 9, 62),
            Step::Across {
                face: Face::NegZ,
                i: 58,
                j: 61,
                k: 9,
                axis_swap: true
            }
        );
        let straight = SeamRecord {
            dst_face: Face::NegZ,
            dst_edge: Edge::UMinus,
            reversed: false,
        };
        assert_eq!(
            across_with(straight, Edge::UPlus, 3, 9, 62),
            Step::Across {
                face: Face::NegZ,
                i: 0,
                j: 3,
                k: 9,
                axis_swap: false
            }
        );
    }

    #[test]
    fn same_face_steps_and_the_band_walls() {
        let g = tiny();
        let r = Rung::ZERO;
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 5, Dir6::IPlus),
            Step::Same(4, 4, 5)
        );
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 5, Dir6::IMinus),
            Step::Same(2, 4, 5)
        );
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 5, Dir6::JPlus),
            Step::Same(3, 5, 5)
        );
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 5, Dir6::JMinus),
            Step::Same(3, 3, 5)
        );
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 5, Dir6::KPlus),
            Step::Same(3, 4, 6)
        );
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 5, Dir6::KMinus),
            Step::Same(3, 4, 4)
        );
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 0, Dir6::KMinus),
            Step::Outside,
            "the floor"
        );
        assert_eq!(
            g.neighbor(r, Face::PosZ, 3, 4, 39, Dir6::KPlus),
            Step::Outside,
            "the ceiling"
        );
        assert_eq!(g.cells_in_band(r), 40);
        assert!(!g.holds(r, 0, 0, 40));
        assert!(!g.holds(r, 62, 0, 0));
        assert!(!g.holds(r, 0, 62, 0));
        assert!(!g.holds(r, -1, 0, 0));
        assert!(!g.holds(r, 0, 0, -1));
        assert!(g.holds(r, 61, 61, 39));
    }

    #[test]
    fn corners_bracket_the_centre_and_outward_is_the_centre_direction() {
        let g = earth();
        let rung = Rung::new(3).expect("rung");
        let (face, i, j, k) = (Face::NegY, 1000, 2000, 700);
        let c = metres_of(g.cell_center(face, i, j, k, rung).expect("cell"));
        let corners = g.cell_corners(face, i, j, k, rung).expect("cell");
        let mean = corners
            .iter()
            .fold(DVec3::ZERO, |acc, p| acc + metres_of(*p))
            / 8.0;
        assert!(
            (mean - c).length() < 0.01,
            "the corners' mean is the centre to a centimetre"
        );
        let out = g.outward(face, i, j, k, rung).expect("cell");
        let dir = c.normalize();
        assert!((out[0] - dir.x).abs() < 1e-9);
        assert!((out[1] - dir.y).abs() < 1e-9);
        assert!((out[2] - dir.z).abs() < 1e-9);
        // The low and high radial corners are one cell (8 m) apart.
        let dr = metres_of(corners[4]).length() - metres_of(corners[0]).length();
        assert!(
            (dr - 8.0).abs() < 1e-6,
            "a rung-3 cell is 8 m tall, got {dr}"
        );
    }
}
