//! ★ THE FLAT ARM — a hull's, a station's and a small asteroid's grid: exact one-metre cubes, nothing
//! bent, the origin at the realm's own frame origin.
//!
//! Every operation is an integer shift on the position lattice, which already counts in `2⁻¹⁰ m`
//! steps ([`crate::pose::FINE_CELL_EDGE_M`]): a rung-`L` cell index is the lattice cell shifted right by
//! `10 + L` (an ARITHMETIC shift, so the stern keeps its sign), and a cell's centre is that index
//! shifted back with half a cell added. A position is NORMALISED at the door, so a caller that built
//! one with the whole offset in the float half is read the same as one that filled the integer half.
//! A sub-metre site is the same shift with `10 − s` and a mask — planted here for the record slice.
//!
//! The domain is a box in whole cells, `-half ≤ index < half` on each axis at rung 0 — the slot a
//! built realm was sold (ruling V6 A3: a box). The domain is a SERVER-SIDE admission rule; a client
//! decodes an address without it, because the indices are signed.
//!
//! **Example.** An engineer walks to the stern of a hull and places a thruster one metre aft of the
//! hull's own origin. Its lattice cell is `−1024` on that axis; shifted by ten it is `−1`, and the block
//! lands where the engineer stood.

use glam::{DVec3, I64Vec3};

use crate::pose::{LatticePos, Tier};

use super::addr::Rung;
use super::{Dir6, Step};

/// The fine lattice's bits below the whole metre (`2⁻¹⁰ m` steps).
const FINE_BITS: u32 = 10;

/// The largest half-extent a flat grid accepts, in whole cells: `2²⁴` (16 777 km). Keeps
/// `half + cell` inside an `i32` at every rung, and no built realm is a fraction of that size.
pub const MAX_HALF_CELLS: i32 = 1 << 24;

/// A sub-metre scale: a small block is `1/2^s` m, `s ≤ 8`, so a site fits one byte per axis (the
/// lattice itself has ten bits below the metre; the address format reserves four). A decoder that
/// meets a larger scale REFUSES it here instead of shifting by a negative count or truncating a site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubScale(u8);

impl SubScale {
    /// The whole cell.
    pub const WHOLE: SubScale = SubScale(0);
    /// The finest scale a one-byte site can express.
    pub const MAX: u8 = 8;

    /// A scale from its exponent; `None` above [`SubScale::MAX`].
    #[must_use]
    pub const fn new(exponent: u8) -> Option<SubScale> {
        if exponent <= SubScale::MAX {
            Some(SubScale(exponent))
        } else {
            None
        }
    }

    /// The exponent.
    #[must_use]
    pub const fn exponent(self) -> u8 {
        self.0
    }
}

/// A flat grid: its half-extents in whole cells at rung 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentityGrid {
    half_cells: [i32; 3],
}

impl IdentityGrid {
    /// A grid whose box has these half-extents in metres, rounded UP to whole cells and at least one
    /// cell, so every point of the sold slot lies inside the grid's domain. `None` for a half-extent
    /// that is not finite or is above [`MAX_HALF_CELLS`]: a slot nobody could have sold.
    #[must_use]
    pub fn from_half_extent_m(half_m: [f64; 3]) -> Option<IdentityGrid> {
        Some(IdentityGrid {
            half_cells: [
                whole_cells_up(half_m[0])?,
                whole_cells_up(half_m[1])?,
                whole_cells_up(half_m[2])?,
            ],
        })
    }

    /// The cube inscribed in a shell of radius `r_m` — the transition for a built realm whose sold
    /// slot is still stated as a shell (ruling V6 A3 makes new slots boxes; old rows stay readable).
    /// `half = r / √3`, rounded DOWN so the cube never leaves the shell, and at least one cell. `None`
    /// for a radius that is not finite or is absurd.
    #[must_use]
    pub fn inscribed_in_shell(r_m: f64) -> Option<IdentityGrid> {
        let half = r_m * 0.577_350_269_189_625_8; // 1/√3 as a literal
        let cells = whole_cells_down(half)?;
        Some(IdentityGrid {
            half_cells: [cells, cells, cells],
        })
    }

    /// The half-extents in whole cells at rung 0.
    #[must_use]
    pub const fn half_cells(&self) -> [i32; 3] {
        self.half_cells
    }

    /// The index range `[lo, hi)` of one axis at `rung`: the rung-0 box, coarsened so that every rung-0
    /// cell of the box lies inside some rung-`L` cell.
    fn axis_range(&self, axis: usize, rung: Rung) -> (i32, i32) {
        let cell = rung.cell_m() as i32;
        let half = self.half_cells[axis];
        ((-half).div_euclid(cell), (half + cell - 1).div_euclid(cell))
    }

    /// Whether `(i, j, k)` lies inside the domain at `rung`.
    #[must_use]
    pub fn contains(&self, rung: Rung, i: i32, j: i32, k: i32) -> bool {
        let (ilo, ihi) = self.axis_range(0, rung);
        let (jlo, jhi) = self.axis_range(1, rung);
        let (klo, khi) = self.axis_range(2, rung);
        i >= ilo && i < ihi && j >= jlo && j < jhi && k >= klo && k < khi
    }

    /// The cell holding `pos` at `rung`, or `None` outside the domain. `pos` is in the realm's own
    /// frame at the FINE tier (every frame at or below a star system is); it is normalised here, so
    /// the whole position is read, never only its integer half.
    #[must_use]
    pub fn addr_of(&self, pos: LatticePos, rung: Rung) -> Option<(i32, i32, i32)> {
        let shift = FINE_BITS + u32::from(rung.level());
        let c = pos.normalize(Tier::Fine).cell();
        let i = shift_to_i32(c.x, shift)?;
        let j = shift_to_i32(c.y, shift)?;
        let k = shift_to_i32(c.z, shift)?;
        if self.contains(rung, i, j, k) {
            Some((i, j, k))
        } else {
            None
        }
    }

    /// The centre of cell `(i, j, k)` at `rung`: the index shifted back, plus half a cell, offset zero.
    #[must_use]
    pub fn cell_center(i: i32, j: i32, k: i32, rung: Rung) -> LatticePos {
        let shift = FINE_BITS + u32::from(rung.level());
        let half = 1i64 << (shift - 1);
        LatticePos::at(
            I64Vec3::new(
                (i64::from(i) << shift) + half,
                (i64::from(j) << shift) + half,
                (i64::from(k) << shift) + half,
            ),
            DVec3::ZERO,
        )
    }

    /// The eight corners of cell `(i, j, k)` at `rung`, low corner first, `i` fastest.
    #[must_use]
    pub fn cell_corners(i: i32, j: i32, k: i32, rung: Rung) -> [LatticePos; 8] {
        let shift = FINE_BITS + u32::from(rung.level());
        let corner = |di: i64, dj: i64, dk: i64| {
            LatticePos::at(
                I64Vec3::new(
                    (i64::from(i) + di) << shift,
                    (i64::from(j) + dj) << shift,
                    (i64::from(k) + dk) << shift,
                ),
                DVec3::ZERO,
            )
        };
        [
            corner(0, 0, 0),
            corner(1, 0, 0),
            corner(0, 1, 0),
            corner(1, 1, 0),
            corner(0, 0, 1),
            corner(1, 0, 1),
            corner(0, 1, 1),
            corner(1, 1, 1),
        ]
    }

    /// The neighbour of `(i, j, k)` in `dir` at `rung`: the next cell, or `Outside` past the box. A
    /// flat grid has no seams, so this never crosses one.
    #[must_use]
    pub fn neighbor(&self, rung: Rung, i: i32, j: i32, k: i32, dir: Dir6) -> Step {
        let (ni, nj, nk) = match dir {
            Dir6::IPlus => (i + 1, j, k),
            Dir6::IMinus => (i - 1, j, k),
            Dir6::JPlus => (i, j + 1, k),
            Dir6::JMinus => (i, j - 1, k),
            Dir6::KPlus => (i, j, k + 1),
            Dir6::KMinus => (i, j, k - 1),
        };
        if self.contains(rung, ni, nj, nk) {
            Step::Same(ni, nj, nk)
        } else {
            Step::Outside
        }
    }
}

/// The sub-metre site of `pos` inside its whole-metre cell at `scale` (`1/2^s` m): the `s` lattice
/// bits below the metre, per axis, of the NORMALISED position. Scale 0 is the whole cell and answers
/// `[0, 0, 0]`. Planted here for the record slice; the shift form IS the definition, and it equals
/// `⌊frac(metres) · 2^s⌋` on every axis (the sweep test proves it).
#[must_use]
pub fn sub_site(pos: LatticePos, scale: SubScale) -> [u8; 3] {
    let s = u32::from(scale.exponent());
    let mask = (1i64 << s) - 1;
    let c = pos.normalize(Tier::Fine).cell();
    [
        ((c.x >> (FINE_BITS - s)) & mask) as u8,
        ((c.y >> (FINE_BITS - s)) & mask) as u8,
        ((c.z >> (FINE_BITS - s)) & mask) as u8,
    ]
}

/// An arithmetic shift of a lattice cell to a cell index, refused when it does not fit an `i32`.
fn shift_to_i32(cell: i64, shift: u32) -> Option<i32> {
    i32::try_from(cell >> shift).ok()
}

/// Metres to whole cells, rounded up, at least one; `None` when not finite or above the maximum.
fn whole_cells_up(m: f64) -> Option<i32> {
    if !m.is_finite() {
        return None;
    }
    let c = m.ceil();
    if c > f64::from(MAX_HALF_CELLS) {
        return None;
    }
    Some(if c < 1.0 { 1 } else { c as i32 })
}

/// Metres to whole cells, rounded down, at least one; `None` when not finite or above the maximum.
fn whole_cells_down(m: f64) -> Option<i32> {
    if !m.is_finite() {
        return None;
    }
    let c = m.floor();
    if c > f64::from(MAX_HALF_CELLS) {
        return None;
    }
    Some(if c < 1.0 { 1 } else { c as i32 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> IdentityGrid {
        IdentityGrid::from_half_extent_m([8.0, 4.5, 16.0]).expect("a sane slot")
    }

    #[test]
    fn the_box_rounds_up_the_inscribed_cube_rounds_down_and_absurd_slots_are_refused() {
        assert_eq!(
            grid().half_cells(),
            [8, 5, 16],
            "4.5 m rounds up to five cells"
        );
        assert_eq!(
            IdentityGrid::from_half_extent_m([0.2, 0.0, 0.9]).map(|g| g.half_cells()),
            Some([1, 1, 1])
        );
        // r = 10 m → 5.77 m → 5 cells; a tiny shell still gives one cell.
        assert_eq!(
            IdentityGrid::inscribed_in_shell(10.0).map(|g| g.half_cells()),
            Some([5, 5, 5])
        );
        assert_eq!(
            IdentityGrid::inscribed_in_shell(1.0).map(|g| g.half_cells()),
            Some([1, 1, 1])
        );
        // The refuter's F13: a slot that decodes as NaN or infinity, or is wider than any realm,
        // is refused rather than becoming an empty or overflowing grid.
        assert_eq!(IdentityGrid::from_half_extent_m([f64::NAN, 1.0, 1.0]), None);
        assert_eq!(
            IdentityGrid::from_half_extent_m([1.0, f64::INFINITY, 1.0]),
            None
        );
        assert_eq!(IdentityGrid::from_half_extent_m([1.0, 1.0, 3.0e7]), None);
        assert_eq!(IdentityGrid::inscribed_in_shell(f64::NAN), None);
        assert_eq!(IdentityGrid::inscribed_in_shell(1.0e9), None);
        // The largest slot still has room at the coarsest rung.
        let big = IdentityGrid::from_half_extent_m([f64::from(MAX_HALF_CELLS); 3]).expect("max");
        assert!(big.contains(Rung::new(15).expect("rung"), 0, 0, 0));
    }

    #[test]
    fn every_cell_of_the_box_on_both_sides_of_the_origin_round_trips_at_every_rung() {
        // U-6 / the negative-half gate: exhaustive over the box, rungs 0..=4.
        let g = grid();
        for level in 0..=4u8 {
            let rung = Rung::new(level).expect("rung");
            let (ilo, ihi) = g.axis_range(0, rung);
            let (jlo, jhi) = g.axis_range(1, rung);
            let (klo, khi) = g.axis_range(2, rung);
            for i in ilo..ihi {
                for j in jlo..jhi {
                    for k in klo..khi {
                        let centre = IdentityGrid::cell_center(i, j, k, rung);
                        assert_eq!(
                            g.addr_of(centre, rung),
                            Some((i, j, k)),
                            "centre round-trips at rung {level} ({i}, {j}, {k})"
                        );
                        let corners = IdentityGrid::cell_corners(i, j, k, rung);
                        assert_eq!(
                            g.addr_of(corners[0], rung),
                            Some((i, j, k)),
                            "low corner is inside"
                        );
                        // The high corner belongs to the NEXT cell (half-open cells).
                        let next = g.addr_of(corners[7], rung);
                        assert_ne!(next, Some((i, j, k)), "the high corner is the next cell's");
                    }
                }
            }
        }
    }

    #[test]
    fn a_thruster_one_metre_aft_keeps_its_sign_however_the_position_was_built() {
        let g = grid();
        let aft = LatticePos::from_metres(DVec3::new(0.25, 0.25, -0.5), Tier::Fine);
        assert_eq!(
            g.addr_of(aft, Rung::ZERO),
            Some((0, 0, -1)),
            "k = −1, not the far bow"
        );
        // The refuter's F12: a position that rides at cell zero with the metres in its float half
        // reads the same, because the grid normalises at the door.
        let raw = LatticePos::at(I64Vec3::ZERO, DVec3::new(0.25, 0.25, -1.0));
        assert_eq!(g.addr_of(raw, Rung::ZERO), Some((0, 0, -1)));
        let outside = LatticePos::from_metres(DVec3::new(9.0, 0.0, 0.0), Tier::Fine);
        assert_eq!(g.addr_of(outside, Rung::ZERO), None, "past the slot");
        let far = LatticePos::at(I64Vec3::new(i64::MAX / 4, 0, 0), DVec3::ZERO);
        assert_eq!(
            g.addr_of(far, Rung::ZERO),
            None,
            "a cell that does not fit an i32 is refused"
        );
    }

    #[test]
    fn neighbours_step_inside_the_box_and_stop_at_its_six_walls() {
        let g = grid();
        let r = Rung::ZERO;
        assert_eq!(g.neighbor(r, 0, 0, 0, Dir6::IPlus), Step::Same(1, 0, 0));
        assert_eq!(g.neighbor(r, 0, 0, 0, Dir6::IMinus), Step::Same(-1, 0, 0));
        assert_eq!(g.neighbor(r, 0, 0, 0, Dir6::JPlus), Step::Same(0, 1, 0));
        assert_eq!(g.neighbor(r, 0, 0, 0, Dir6::JMinus), Step::Same(0, -1, 0));
        assert_eq!(g.neighbor(r, 0, 0, 0, Dir6::KPlus), Step::Same(0, 0, 1));
        assert_eq!(g.neighbor(r, 0, 0, 0, Dir6::KMinus), Step::Same(0, 0, -1));
        for (i, j, k, dir, wall) in [
            (7, 0, 0, Dir6::IPlus, "+i"),
            (-8, 0, 0, Dir6::IMinus, "−i"),
            (0, 4, 0, Dir6::JPlus, "+j"),
            (0, -5, 0, Dir6::JMinus, "−j"),
            (0, 0, 15, Dir6::KPlus, "+k"),
            (0, 0, -16, Dir6::KMinus, "−k"),
        ] {
            assert_eq!(
                g.neighbor(r, i, j, k, dir),
                Step::Outside,
                "the wall at {wall}"
            );
        }
    }

    #[test]
    fn the_sub_site_equals_the_index_space_form_over_the_lattice_on_both_sides_of_the_origin() {
        // The refuter's F14/F17: a scale above the lattice's bits is refused, and the shift form
        // equals ⌊frac(metres) · 2^s⌋ at every scale over a sweep of positions.
        assert_eq!(
            SubScale::new(9),
            None,
            "finer than a byte per axis is refused"
        );
        assert_eq!(SubScale::new(8).map(SubScale::exponent), Some(8));
        assert_eq!(SubScale::new(0), Some(SubScale::WHOLE));
        let eighth = SubScale::new(3).expect("1/8 m");
        let p = LatticePos::from_metres(DVec3::new(0.375, 2.9, -0.125), Tier::Fine);
        assert_eq!(
            sub_site(p, eighth),
            [3, 7, 7],
            "−0.125 m is the top eighth of cell −1"
        );
        assert_eq!(
            sub_site(p, SubScale::WHOLE),
            [0, 0, 0],
            "scale 0 is the whole cell"
        );
        let mut checked = 0u32;
        let mut n = 0i64;
        while n < 400 {
            // A deterministic walk over ±3 m in irregular steps, on every axis differently.
            let m = DVec3::new(
                (n as f64) * 0.013_7 - 3.0,
                (n as f64) * -0.007_1 + 1.234_5,
                (n as f64) * 0.019_9 - 2.5,
            );
            let pos = LatticePos::from_metres(m, Tier::Fine);
            let mut s = 0u8;
            while s <= SubScale::MAX {
                let scale = SubScale::new(s).expect("in range");
                let want =
                    [m.x, m.y, m.z].map(|v| ((v - v.floor()) * f64::from(1u32 << s)).floor() as u8);
                assert_eq!(sub_site(pos, scale), want, "position {m:?} at scale {s}");
                checked += 1;
                s += 1;
            }
            n += 1;
        }
        assert_eq!(checked, 400 * 9);
    }
}
