//! ★ THE CLIMATE GRID (the design §3): a regular LATITUDE–LONGITUDE grid on the body's own spin axis — the energy
//! balance, the winds and the vapour live here, and a node reads them by a bilinear reading in its latitude and
//! longitude ([`ClimateGrid::reading`]).
//!
//! **WHY LATITUDE–LONGITUDE, NOT THE CUBE.** A diffusion between cell centres is CONSISTENT only where the line between
//! two centres crosses their shared edge at a right angle. The cube-sphere lattice's cells are skewed away from a face's
//! centre: its two-point flux was wrong by 12–16 %, and by a factor of ten at a cube corner — MEASURED on the sphere's
//! own eigenfunction `∇² sin φ = −2·sin φ/a²` (2026-10-15), an error that does not fall with the cell's size. A
//! latitude–longitude grid is orthogonal everywhere: its east face has length `a·Δφ` across a centre distance
//! `a·cos φ·Δλ`, its north face `a·cos φ_n·Δλ` across `a·Δφ`, so the two-point flux is exact to second order, it
//! conserves energy exactly, its zonal means are its rows, and the flux through a pole is nought by geometry (the face's
//! length is). A rotating planet's climate is zonal and meridional by its physics, so the grid's axes are the physics'
//! axes, never a lean the ground could show: a node reads the grid's fields smoothly.
//!
//! **THE FENCE.** A node's latitude and longitude come from its direction by [`Gf::acos`]: `φ = π/2 − acos z`,
//! `λ = ±acos(x/√(x² + y²))` with the sign of `y`; the rows' cosines and sines by [`cos_series`] (a Taylor polynomial in
//! exact operations, as [`Gf::exp`] is).
//!
//! **Example.** At 90 rows the home planet's climate grid holds 16 200 cells of 2° (221 km at the equator): a continent
//! spans dozens of them, and the node under hull 15's berth reads its summer temperature from the four cells around it.

use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;
use vd_recipe::height::POLE_AXIS;

use crate::gf::Gf;

/// A unit direction in fenced floats.
#[must_use]
pub fn unit(d: [Gi; 3]) -> [Gf; 3] {
    let s = Gf::from_i64(1i64 << DIR_BITS);
    let v = [
        Gf::from_i64(d[0].raw()) / s,
        Gf::from_i64(d[1].raw()) / s,
        Gf::from_i64(d[2].raw()) / s,
    ];
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// `cos θ` for `θ` in `[0, π]`: the Taylor polynomial to `θ²⁴/24!` on `[0, π/2]` (its truncation under `10⁻¹⁷`), and
/// `cos θ = −cos(π − θ)` beyond. Add, multiply, divide by a whole number: one answer on every target.
#[must_use]
pub fn cos_series(theta: Gf) -> Gf {
    let pi = Gf::from_f64(std::f64::consts::PI);
    let (x, sign) = if theta > pi * Gf::HALF {
        (pi - theta, -Gf::ONE)
    } else {
        (theta, Gf::ONE)
    };
    let x2 = x * x;
    // 1 − x²/2(1 − x²/12(1 − x²/30(…))): Horner from the inside, the n-th factor x²/((2n − 1)·2n).
    let mut p = Gf::ONE;
    let mut n = 12i64;
    while n >= 1 {
        p = Gf::ONE - x2 * p / Gf::from_i64((2 * n - 1) * (2 * n));
        n -= 1;
    }
    sign * p
}

/// The latitude and the east longitude of a unit direction, radians (`φ` in `[−π/2, π/2]`, `λ` in `[0, 2π)`), by the
/// fenced arc cosine.
#[must_use]
pub fn lat_lon(d: [Gf; 3]) -> (Gf, Gf) {
    let half_pi = Gf::from_f64(std::f64::consts::FRAC_PI_2);
    let two_pi = Gf::from_f64(std::f64::consts::TAU);
    let z = d[POLE_AXIS].clamp(-Gf::ONE, Gf::ONE);
    let lat = half_pi - z.acos();
    let (x, y) = (d[0], d[1]);
    let r = (x * x + y * y).sqrt();
    let lon = if r > Gf::ZERO {
        let a = (x / r).clamp(-Gf::ONE, Gf::ONE).acos();
        if y < Gf::ZERO { two_pi - a } else { a }
    } else {
        Gf::ZERO
    };
    (lat, lon)
}

/// The climate grid: `rows` latitude bands of `π/rows` from the south pole, `2·rows` longitude columns of `π/rows` from
/// longitude 0; cell `(r, c)` is index `r·cols + c`.
#[derive(Clone, Debug)]
pub struct ClimateGrid {
    pub rows: usize,
    pub cols: usize,
    /// The radius, m.
    pub radius_m: Gf,
    /// Each row's centre latitude's sine and cosine.
    pub sin_lat: Vec<Gf>,
    pub cos_lat: Vec<Gf>,
    /// Each row's cell area, m².
    pub area: Vec<Gf>,
    /// A row's east–west conductance (the face's length over the centres' distance), and the conductance across each
    /// row's northern edge (nought at the north pole).
    pub c_east: Vec<Gf>,
    pub c_north: Vec<Gf>,
}

impl ClimateGrid {
    /// The grid of `rows ≥ 2` latitude bands on a body of radius `radius_m`.
    #[must_use]
    pub fn new(radius_m: Gf, rows: usize) -> ClimateGrid {
        let cols = 2 * rows;
        let pi = Gf::from_f64(std::f64::consts::PI);
        let step = pi / Gf::from_i64(rows as i64);
        // The angle from the south pole at each half row: the edges at 2r, the centres at 2r + 1. Latitude = angle − π/2,
        // so sin φ = −cos(angle) and cos φ = sin(angle) = √(1 − cos²).
        let half = |m: usize| {
            let c = cos_series(pi * Gf::from_i64(m as i64) / Gf::from_i64(2 * rows as i64));
            (-c, (Gf::ONE - c * c).greater(Gf::ZERO).sqrt())
        };
        let mut sin_lat = Vec::with_capacity(rows);
        let mut cos_lat = Vec::with_capacity(rows);
        let mut area = Vec::with_capacity(rows);
        let mut c_east = Vec::with_capacity(rows);
        let mut c_north = Vec::with_capacity(rows);
        for r in 0..rows {
            let (s_south, _) = half(2 * r);
            let (s_mid, c_mid) = half(2 * r + 1);
            let (s_north, c_edge_north) = half(2 * r + 2);
            sin_lat.push(s_mid);
            cos_lat.push(c_mid);
            area.push(radius_m * radius_m * step * (s_north - s_south));
            // The rows' and the columns' steps are one angle, so the east face over the centres' distance is
            // `1/cos φ` and the north face over the rows' distance is `cos φ_north`.
            c_east.push(Gf::ONE / c_mid);
            c_north.push(if r + 1 < rows { c_edge_north } else { Gf::ZERO });
        }
        ClimateGrid {
            rows,
            cols,
            radius_m,
            sin_lat,
            cos_lat,
            area,
            c_east,
            c_north,
        }
    }

    /// The cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Whether the grid is empty (never: two rows at least).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// The cell's row.
    #[must_use]
    pub fn row(&self, cell: usize) -> usize {
        cell / self.cols
    }

    /// The cell's four neighbours with their conductances: west, east (the columns wrap), south, north (`None` past a
    /// pole).
    #[must_use]
    pub fn neighbours(&self, cell: usize) -> [Option<(usize, Gf)>; 4] {
        let (r, c) = (cell / self.cols, cell % self.cols);
        let west = r * self.cols + (c + self.cols - 1) % self.cols;
        let east = r * self.cols + (c + 1) % self.cols;
        let south = (r > 0).then(|| ((r - 1) * self.cols + c, self.c_north[r - 1]));
        let north = (r + 1 < self.rows).then(|| ((r + 1) * self.cols + c, self.c_north[r]));
        [
            Some((west, self.c_east[r])),
            Some((east, self.c_east[r])),
            south,
            north,
        ]
    }

    /// The field's diffusion `(1/A_i)·Σ_j c_ij·(f_j − f_i)`, per square metre — the grid's Laplacian.
    #[must_use]
    pub fn laplacian(&self, f: &[Gf]) -> Vec<Gf> {
        (0..self.len())
            .map(|i| {
                let mut s = Gf::ZERO;
                for (m, c) in self.neighbours(i).into_iter().flatten() {
                    s += c * (f[m] - f[i]);
                }
                s / self.area[self.row(i)]
            })
            .collect()
    }

    /// The row's step in radians, `π/rows`.
    fn step(&self) -> Gf {
        Gf::from_f64(std::f64::consts::PI) / Gf::from_i64(self.rows as i64)
    }

    /// ★ THE BILINEAR READING of a cell field at a direction: the four cells around the point's latitude and longitude
    /// and their weights, which sum to one (beyond the outermost rows' centres, that row alone).
    #[must_use]
    pub fn reading(&self, d: [Gf; 3]) -> [(usize, Gf); 4] {
        let (lat, lon) = lat_lon(d);
        let step = self.step();
        // The centre of row r stands r + ½ steps from the south pole, of column c at c + ½ steps from longitude 0.
        let y = (lat + Gf::from_f64(std::f64::consts::FRAC_PI_2)) / step - Gf::HALF;
        let x = lon / step - Gf::HALF;
        let fy = y.floor();
        let r = fy.to_i64_floor();
        let (r0, r1, wy) = if r < 0 {
            (0, 0, Gf::ZERO)
        } else if r as usize + 1 >= self.rows {
            (self.rows - 1, self.rows - 1, Gf::ZERO)
        } else {
            (r as usize, r as usize + 1, y - fy)
        };
        let fx = x.floor();
        let wx = x - fx;
        let c0 = fx.to_i64_floor().rem_euclid(self.cols as i64) as usize;
        let c1 = (c0 + 1) % self.cols;
        let (ax, ay) = (Gf::ONE - wx, Gf::ONE - wy);
        [
            (r0 * self.cols + c0, ax * ay),
            (r0 * self.cols + c1, wx * ay),
            (r1 * self.cols + c0, ax * wy),
            (r1 * self.cols + c1, wx * wy),
        ]
    }

    /// The cell holding a direction.
    #[must_use]
    pub fn cell_at(&self, d: [Gf; 3]) -> usize {
        let (lat, lon) = lat_lon(d);
        let step = self.step();
        let r = ((lat + Gf::from_f64(std::f64::consts::FRAC_PI_2)) / step)
            .floor()
            .to_i64_floor()
            .clamp(0, self.rows as i64 - 1) as usize;
        let c = (lon / step)
            .floor()
            .to_i64_floor()
            .rem_euclid(self.cols as i64) as usize;
        r * self.cols + c
    }
}

/// The cyclic three-band system `−lo_c·x_{c−1} + di_c·x_c − up_c·x_{c+1} = rhs_c` (indices wrapping), solved by the
/// Thomas algorithm and the Sherman–Morrison correction for the two corner terms.
pub fn cyclic_three_band(lo: &[Gf], di: &[Gf], up: &[Gf], rhs: &[Gf], x: &mut [Gf]) {
    let m = di.len();
    // As a·x_{c−1} + b·x_c + c·x_{c+1} = d with a = −lo, c = −up; corners a_0 (to x_{m−1}) and c_{m−1} (to x_0).
    let a: Vec<Gf> = lo.iter().map(|v| -*v).collect();
    let cc: Vec<Gf> = up.iter().map(|v| -*v).collect();
    let gamma = -di[0];
    let mut b = di.to_vec();
    b[0] = di[0] - gamma;
    b[m - 1] = di[m - 1] - a[0] * cc[m - 1] / gamma;
    let solve = |d: &[Gf], out: &mut [Gf]| {
        let mut cp = vec![Gf::ZERO; m];
        let mut dp = vec![Gf::ZERO; m];
        cp[0] = cc[0] / b[0];
        dp[0] = d[0] / b[0];
        for k in 1..m {
            let den = b[k] - a[k] * cp[k - 1];
            cp[k] = cc[k] / den;
            dp[k] = (d[k] - a[k] * dp[k - 1]) / den;
        }
        out[m - 1] = dp[m - 1];
        for k in (0..m - 1).rev() {
            out[k] = dp[k] - cp[k] * out[k + 1];
        }
    };
    let mut y = vec![Gf::ZERO; m];
    solve(rhs, &mut y);
    let mut u = vec![Gf::ZERO; m];
    u[0] = gamma;
    u[m - 1] = cc[m - 1];
    let mut z = vec![Gf::ZERO; m];
    solve(&u, &mut z);
    // v = (1, 0, …, 0, a_0/γ)
    let vy = y[0] + a[0] / gamma * y[m - 1];
    let vz = z[0] + a[0] / gamma * z[m - 1];
    let k = vy / (Gf::ONE + vz);
    for i in 0..m {
        x[i] = y[i] - k * z[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ THE CYCLIC THREE-BAND SOLVE: a seven-cell ring with unequal couplings is solved, and the solution meets every
    /// row of the system including the two corner terms.
    #[test]
    fn the_cyclic_three_band_solve_meets_its_system() {
        let lo: Vec<Gf> = [0.3, 0.5, 0.2, 0.7, 0.1, 0.4, 0.6]
            .iter()
            .map(|&v| Gf::from_f64(v))
            .collect();
        let up: Vec<Gf> = [0.2, 0.4, 0.6, 0.1, 0.5, 0.3, 0.2]
            .iter()
            .map(|&v| Gf::from_f64(v))
            .collect();
        let di: Vec<Gf> = [2.0, 2.5, 1.9, 3.0, 2.2, 2.1, 2.4]
            .iter()
            .map(|&v| Gf::from_f64(v))
            .collect();
        let rhs: Vec<Gf> = [1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0]
            .iter()
            .map(|&v| Gf::from_f64(v))
            .collect();
        let mut x = vec![Gf::ZERO; 7];
        cyclic_three_band(&lo, &di, &up, &rhs, &mut x);
        for c in 0..7 {
            let prev = x[(c + 6) % 7];
            let next = x[(c + 1) % 7];
            let row = -lo[c] * prev + di[c] * x[c] - up[c] * next;
            assert!((row - rhs[c]).abs().to_f64() < 1e-12, "row {c}");
        }
    }

    /// The fenced cosine at the known angles.
    #[test]
    fn the_cosine_series_meets_the_known_angles() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_6, PI};
        for (theta, want) in [
            (0.0, 1.0),
            (FRAC_PI_6, 0.866_025_403_784_438_6),
            (FRAC_PI_3, 0.5),
            (FRAC_PI_2, 0.0),
            (2.0 * FRAC_PI_3, -0.5),
            (PI, -1.0),
        ] {
            let got = cos_series(Gf::from_f64(theta)).to_f64();
            assert!((got - want).abs() < 4e-16, "cos {theta} = {got}");
        }
    }

    /// ★ THE GRID'S LAPLACIAN against the sphere's own: `∇² sin φ = −2·sin φ/a²` at 90 and 180 rows, the error falling
    /// with the row's height; the operator conserves (its area-weighted sum is nought); the cells' areas sum to the
    /// sphere's.
    #[test]
    fn the_grids_laplacian_meets_the_spheres() {
        let a = Gf::from_f64(6.341_67e6);
        let a2 = a.to_f64() * a.to_f64();
        let mut last = f64::INFINITY;
        for rows in [90usize, 180] {
            let grid = ClimateGrid::new(a, rows);
            assert_eq!(grid.len(), 2 * rows * rows);
            assert!(!grid.is_empty());
            let f: Vec<Gf> = (0..grid.len()).map(|i| grid.sin_lat[grid.row(i)]).collect();
            let lap = grid.laplacian(&f);
            let mut worst = 0.0f64;
            let mut total = Gf::ZERO;
            for (i, l) in lap.iter().enumerate() {
                let err = (l.to_f64() * a2 + 2.0 * f[i].to_f64()).abs();
                worst = if err > worst { err } else { worst };
                total += *l * grid.area[grid.row(i)];
            }
            assert!(worst < 1e-3, "rows {rows}: worst {worst}");
            assert!(worst < last / 3.0, "rows {rows}: {worst} against {last}");
            last = worst;
            assert!(total.abs().to_f64() < 1e-9, "conserved: {}", total.to_f64());
            let sphere: f64 =
                (0..rows).map(|r| grid.area[r].to_f64()).sum::<f64>() * grid.cols as f64;
            assert!((sphere / (4.0 * std::f64::consts::PI * a2) - 1.0).abs() < 1e-12);
        }
    }

    /// ★ THE READING: a direction's latitude and longitude by the fenced arc cosine; the four weights sum to one; a smooth
    /// field reads back within the grid's own step; the north pole reads the top row; the date line wraps.
    #[test]
    fn the_reading_finds_the_four_cells() {
        let grid = ClimateGrid::new(Gf::ONE, 90);
        // 30° N, 60° E: cos φ = √3/2, sin φ = ½, cos λ = ½, sin λ = √3/2 (literal values).
        let s3 = 0.866_025_403_784_438_6;
        let d = [
            Gf::from_f64(s3 * 0.5),
            Gf::from_f64(s3 * s3),
            Gf::from_f64(0.5),
        ];
        let (lat, lon) = lat_lon(d);
        assert!((lat.to_f64() - std::f64::consts::FRAC_PI_6).abs() < 1e-14);
        assert!((lon.to_f64() - std::f64::consts::FRAC_PI_3).abs() < 1e-14);
        let w = grid.reading(d);
        let sum: f64 = w.iter().map(|(_, x)| x.to_f64()).sum();
        assert!((sum - 1.0).abs() < 1e-14);
        let field: Vec<Gf> = (0..grid.len()).map(|i| grid.sin_lat[grid.row(i)]).collect();
        let read: f64 = w.iter().map(|&(c, x)| (field[c] * x).to_f64()).sum();
        assert!((read - 0.5).abs() < 1e-3, "{read}");
        // 30° N lies on a row's edge and 60° E on a column's: the holding cell is one of the four the reading names.
        assert!(w.iter().any(|&(c, _)| c == grid.cell_at(d)));
        let pole = grid.reading([Gf::ZERO, Gf::ZERO, Gf::ONE]);
        for &(c, _) in &pole {
            assert_eq!(grid.row(c), grid.rows - 1);
        }
        let south = grid.reading([Gf::ZERO, Gf::ZERO, -Gf::ONE]);
        for &(c, _) in &south {
            assert_eq!(grid.row(c), 0);
        }
        // Just west of longitude 0: the last column and the first.
        let west = [Gf::from_f64(0.999_999), Gf::from_f64(-0.001_414), Gf::ZERO];
        let ww = grid.reading(west);
        assert_eq!(ww[0].0 % grid.cols, grid.cols - 1);
        assert_eq!(ww[1].0 % grid.cols, 0);
        // A direction on the pole axis has longitude 0.
        assert_eq!(lat_lon([Gf::ZERO, Gf::ZERO, Gf::ONE]).1, Gf::ZERO);
    }
}
