//! ★ THE LADDER — the arithmetic that puts a round body's grid on the power-of-two ladder and turns a
//! cell index into a radius and a face parameter (the voxel foundation, slice 2's shell grid, moved
//! here in slice 5 so the generator and the grid share ONE implementation).
//!
//! A round body's grid has `N` cells along a face edge at rung 0, and `N = q · 2^(T−1)` where `T` is
//! the number of rungs, so every rung divides the face evenly. The radius is `R = 2N/π`, which makes a
//! cell at the face centre exactly one metre. The band is a crust below the surface and a height
//! above it, in whole metres, floored to whole cells at every rung.
//!
//! **Every expression here is exactly the one the shell grid used before the move** (slice 2's golden
//! digest pins prove it): the same operations in the same order, so no address moved.
//!
//! **Example.** The home planet's look radius is 3 351 154 m. The ideal edge is `N = R·π/2`, about
//! 5.26 million cells; the ladder snaps it to a multiple of `2^(T−1)` so that the top rung is the first
//! with at most 64 chunks per face edge, and the radius becomes `2N/π` — 3 350 759 m, within a snap
//! unit of the look radius and exact on the ladder (MEASURED by `terrain_cost`).

use std::f64::consts::{FRAC_2_PI, FRAC_PI_2};

/// Cells per chunk edge: a PACKING constant.
pub const CHUNK_EDGE: u64 = 62;
/// The top rung is the first rung with at most this many chunks along a face edge.
pub const TOP_RUNG_CHUNKS: u64 = 64;
/// The largest face edge the address can name.
pub const N_MAX: u64 = 1 << 26;
/// The coarsest rung the address can name (four bits).
pub const RUNG_MAX: u8 = 15;

/// A round body's grid on the ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ladder {
    /// Cells along a face edge at rung 0.
    pub n: u32,
    /// The radius of the band's floor, in whole metres.
    pub floor_m: u32,
    /// The band's height from the floor, in whole metres (crust plus above).
    pub band_m: u32,
    /// How many rungs the body has: `0..rungs`.
    pub rungs: u8,
}

/// Metres per cell at a rung, as a power of two.
#[must_use]
pub const fn cell_m(rung: u8) -> u32 {
    1 << rung
}

/// ★ GAP STEPS PER METRE — the integer recipe's unit of length (ruling F7): one step is 1/128 m,
/// which is the density byte's own step at the one-metre rung. Every radius, every surface and every
/// cave hollow the recipe holds is a whole number of these steps or a fixed-point fraction of one.
/// Pinned against `vd_recipe::height::GAP_STEPS_PER_CELL` by the generator's own test.
pub const STEPS_PER_M: i64 = 128;

/// The first rung with at most [`TOP_RUNG_CHUNKS`] chunks along a face edge of `n` cells.
#[must_use]
pub fn top_rung_for(n: f64) -> u32 {
    let chunks = (n / ((CHUNK_EDGE * TOP_RUNG_CHUNKS) as f64)).ceil() as u64;
    if chunks <= 1 {
        0
    } else {
        64 - (chunks - 1).leading_zeros()
    }
}

/// Snap an ideal edge to a multiple of `2^top`, never to zero.
#[must_use]
pub fn snap(n_ideal: f64, top: u32) -> u64 {
    let unit = 1u64 << top;
    let q = (n_ideal / (unit as f64)).round() as u64;
    let q = if q == 0 { 1 } else { q };
    q * unit
}

impl Ladder {
    /// The ladder for a body of `radius_m` with a crust of `crust_m` below its surface and `above_m`
    /// above it; `None` for a radius that is not a positive finite number, a body larger than the
    /// address can name, a top rung past the address's four bits, or a crust deeper than the body.
    #[must_use]
    pub fn for_radius(radius_m: f64, crust_m: u32, above_m: u32) -> Option<Ladder> {
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return None;
        }
        let n_ideal = radius_m * FRAC_PI_2;
        if n_ideal > 1e15 {
            return None; // absurd; keeps the integer arithmetic below total
        }
        let n = snap(n_ideal, top_rung_for(n_ideal));
        let top = top_rung_for(n as f64);
        if top > u32::from(RUNG_MAX) || n > N_MAX {
            return None;
        }
        let radius = (n as f64) * FRAC_2_PI;
        let surface = radius.round() as u64;
        let crust = u64::from(crust_m);
        if crust >= surface {
            return None;
        }
        let band_m = u32::try_from(crust + u64::from(above_m)).ok()?;
        Some(Ladder {
            n: n as u32,
            floor_m: (surface - crust) as u32,
            band_m,
            rungs: (top + 1) as u8,
        })
    }

    /// The body's radius on the ladder: `2N/π`.
    #[must_use]
    pub fn radius_m(&self) -> f64 {
        f64::from(self.n) * FRAC_2_PI
    }

    /// Cells along a face edge at a rung.
    #[must_use]
    pub const fn cells_per_edge(&self, rung: u8) -> u32 {
        self.n >> rung
    }

    /// Whole cells in the band at a rung: the band's metres over the rung's cell width. A cell is a
    /// POWER OF TWO metres, so the division is the rung's own shift (ruling F7: no `/` on the recipe's
    /// path) and the answer is the same whole number it always was.
    #[must_use]
    pub const fn cells_in_band(&self, rung: u8) -> u32 {
        self.band_m >> rung
    }

    /// Whether the address is inside the body's grid at that rung.
    #[must_use]
    pub fn holds(&self, rung: u8, i: i32, j: i32, k: i32) -> bool {
        if rung >= self.rungs {
            return false;
        }
        let n_l = self.cells_per_edge(rung) as i32;
        let band = self.cells_in_band(rung) as i32;
        i >= 0 && i < n_l && j >= 0 && j < n_l && k >= 0 && k < band
    }

    /// The radius of a cell's centre at a rung: the floor plus `k + ½` cells.
    #[must_use]
    pub fn cell_radius_m(&self, k: i32, rung: u8) -> f64 {
        f64::from(self.floor_m) + (f64::from(k) + 0.5) * f64::from(cell_m(rung))
    }

    /// The radius of a cell's lower corner at a rung: the floor plus `k` cells.
    #[must_use]
    pub fn corner_radius_m(&self, k: i32, rung: u8) -> f64 {
        f64::from(self.floor_m) + f64::from(k) * f64::from(cell_m(rung))
    }

    /// ★ THE INTEGER TWIN of [`Ladder::cell_radius_m`]: a cell centre's radius in GAP STEPS
    /// ([`STEPS_PER_M`], 1/128 m), EXACT. The floor is whole metres and a cell is a power of two
    /// metres, so `floor + (k + ½)·cell` is a half-integer count of metres and 128 times it is a
    /// whole number — the integer recipe's density never rounds the radius (ruling F7).
    #[must_use]
    pub const fn cell_radius_steps(&self, k: i32, rung: u8) -> i64 {
        (self.floor_m as i64) * STEPS_PER_M
            + (2 * (k as i64) + 1) * (cell_m(rung) as i64) * (STEPS_PER_M >> 1)
    }

    /// ★ THE INTEGER TWIN of [`Ladder::corner_radius_m`]: a cell's lower corner in gap steps, exact.
    #[must_use]
    pub const fn corner_radius_steps(&self, k: i32, rung: u8) -> i64 {
        (self.floor_m as i64) * STEPS_PER_M + (k as i64) * (cell_m(rung) as i64) * STEPS_PER_M
    }
}

/// The face parameter of a cell's centre: `(2i + 1)/n_l − 1`, in `(−1, 1)`.
#[must_use]
pub fn face_param(i: i32, n_l: u32) -> f64 {
    (2.0 * f64::from(i) + 1.0) / f64::from(n_l) - 1.0
}

/// The face parameter of a cell's lower corner: `2i/n_l − 1`, in `[−1, 1]`.
#[must_use]
pub fn corner_param(i: i32, n_l: u32) -> f64 {
    2.0 * f64::from(i) / f64::from(n_l) - 1.0
}

/// The cell index a face parameter falls in, the last cell holding the `+1` edge.
#[must_use]
pub fn index_of(a: f64, n_l: u32) -> i32 {
    let u = ((a + 1.0) * 0.5 * f64::from(n_l)).floor() as i32;
    if u >= n_l as i32 { n_l as i32 - 1 } else { u }
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]
    use super::*;

    #[test]
    fn the_ladder_snaps_a_body_and_refuses_what_the_address_cannot_name() {
        let earth = Ladder::for_radius(6_371_000.0, 8_283, 8_283).expect("earth");
        assert_eq!(earth.n % (1 << (earth.rungs - 1)), 0, "N = q·2^(T−1)");
        assert!(
            (earth.radius_m() - 6_371_000.0).abs() < 4_100.0,
            "within half a unit"
        );
        assert_eq!(earth.cells_per_edge(0), earth.n);
        assert_eq!(earth.cells_per_edge(3), earth.n >> 3);
        assert_eq!(earth.cells_in_band(0), earth.band_m);
        assert_eq!(earth.cells_in_band(4), earth.band_m >> 4);
        assert!(earth.holds(0, 0, 0, 0));
        assert!(!earth.holds(earth.rungs, 0, 0, 0), "past the top rung");
        assert!(!earth.holds(0, -1, 0, 0));
        assert!(!earth.holds(0, 0, 0, earth.band_m as i32));
        assert_eq!(earth.cell_radius_m(0, 0), f64::from(earth.floor_m) + 0.5);
        assert_eq!(earth.corner_radius_m(2, 1), f64::from(earth.floor_m) + 4.0);
        // The integer twins: the same radius in gap steps, exactly, at every rung and on both sides
        // of the band's floor (the halo reads a `k` of −1).
        let floor_steps = i64::from(earth.floor_m) * STEPS_PER_M;
        assert_eq!(earth.cell_radius_steps(0, 0), floor_steps + 64);
        assert_eq!(earth.corner_radius_steps(2, 1), floor_steps + 4 * 128);
        assert_eq!(earth.corner_radius_steps(-1, 0), floor_steps - 128);
        for rung in [0u8, 1, 5, 12] {
            for k in [-1i32, 0, 1, 61, 4_097] {
                assert_eq!(
                    earth.cell_radius_steps(k, rung) as f64 / STEPS_PER_M as f64,
                    earth.cell_radius_m(k, rung),
                    "cell ({k}, {rung})"
                );
                assert_eq!(
                    earth.corner_radius_steps(k, rung) as f64 / STEPS_PER_M as f64,
                    earth.corner_radius_m(k, rung),
                    "corner ({k}, {rung})"
                );
            }
        }
        assert_eq!(Ladder::for_radius(f64::NAN, 1, 1), None);
        assert_eq!(Ladder::for_radius(0.0, 1, 1), None);
        assert_eq!(Ladder::for_radius(1e16, 1, 1), None, "absurd");
        assert_eq!(Ladder::for_radius(1e9, 1, 1), None, "past N_MAX");
        assert_eq!(
            Ladder::for_radius(100.0, 200, 1),
            None,
            "a crust deeper than the body"
        );
        assert_eq!(
            Ladder::for_radius(100.0, 10, u32::MAX),
            None,
            "a band that overflows the whole metres"
        );
    }

    #[test]
    fn the_face_parameters_and_the_index_agree_on_both_edges() {
        assert_eq!(face_param(0, 4), -0.75);
        assert_eq!(face_param(3, 4), 0.75);
        assert_eq!(corner_param(0, 4), -1.0);
        assert_eq!(corner_param(4, 4), 1.0);
        assert_eq!(index_of(-1.0, 4), 0);
        assert_eq!(index_of(1.0, 4), 3, "the +1 edge belongs to the last cell");
        assert_eq!(index_of(0.0, 4), 2);
        assert_eq!(cell_m(0), 1);
        assert_eq!(cell_m(3), 8);
        assert_eq!(top_rung_for(100.0), 0);
        assert_eq!(top_rung_for(f64::from(CHUNK_EDGE as u32 * 64 + 1)), 1);
        assert_eq!(snap(0.4, 0), 1, "never zero");
        assert_eq!(snap(1000.0, 3), 1000);
        assert_eq!(snap(1003.0, 3), 1000);
    }
}
