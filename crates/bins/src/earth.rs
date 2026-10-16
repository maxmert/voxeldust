//! ★ EARTH AS A MEASURING STICK ON THE HOME PLANET'S OWN GRID — bench only (ruling B36 item 7 and B79 Q1: Earth's real
//! data, kept OUTSIDE the repository, as READINGS; never a constant turned toward them, ruling B38). The readers and the
//! one binning rule the Earth benches share:
//!
//! - **THE RELIEF**: the raw grid of NOAA NCEI's ETOPO 2022 surface relief at 60″ (21 600 × 10 800 little-endian 16-bit
//!   whole metres, row 0 at the north edge, column 0 at 180° W; DOI 10.25921/fd45-gt74, no copyright restriction),
//!   averaged onto the home planet's macro lattice: each node the mean of the cells whose centres fall in it, by each
//!   cell's area. The home planet's lattice radius is 0.46 % short of Earth's, so a chord or an area is 0.46 % short and a
//!   share does not move (`examples/earth_relief.rs` states the same).
//! - **THE RAIN**: GPCP v2.3's monthly long-term mean 1991–2020 (2.5°, mm/day; NOAA PSL), and **THE WIND**: NCEP/NCAR
//!   Reanalysis 1's monthly long-term mean 10 m wind (the T62 Gaussian grid, m/s; NOAA PSL) — both decoded once from
//!   their netCDF files into NumPy `.npy` arrays (`~/voxeldust_earth/climate/README.txt` states the rows' order).
//!
//! The frame: z to the north pole, x to the prime meridian — the lattice's own pole axis is z
//! (`vd_recipe::height::POLE_AXIS`).
//!
//! **Example.** The rain bench lays Earth's Andes and Amazon on the home planet's 8 km nodes, runs the seasonal climate
//! with Earth's charter, and reads the node under Manaus against GPCP's Manaus: a reading, never a target.

use vd_core::glam::DVec3;
use vd_terrain::macro_lattice::MacroLattice;

use crate::drainage_gates::node_under;

/// The ETOPO raw grid's columns and rows (60 arc-seconds over the globe), and its cells to a degree.
pub const ETOPO_COLUMNS: usize = 21_600;
pub const ETOPO_ROWS: usize = 10_800;
const CELLS_PER_DEGREE: f64 = 60.0;
/// The binning's threads: a fixed count, so the sums add in one order and the means print alike on every machine.
const BIN_THREADS: usize = 8;

/// A node's unit direction (the lattice's own bend of the node's centre).
#[must_use]
pub fn dir_of(lattice: &MacroLattice, node: u32) -> DVec3 {
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let d = lattice.direction(node);
    DVec3::new(
        d[0].raw() as f64 / unit,
        d[1].raw() as f64 / unit,
        d[2].raw() as f64 / unit,
    )
    .normalize()
}

/// The latitude and the east longitude of a unit direction, degrees (the longitude in `[0, 360)`).
#[must_use]
pub fn lat_lon_of(d: DVec3) -> (f64, f64) {
    let lat = d.z.clamp(-1.0, 1.0).asin().to_degrees();
    let lon = d.y.atan2(d.x).to_degrees().rem_euclid(360.0);
    (lat, lon)
}

/// Reads the ETOPO raw grid.
#[must_use]
pub fn read_etopo(path: &str) -> Vec<i16> {
    let bytes = std::fs::read(path).expect("the raw grid reads");
    assert_eq!(
        bytes.len(),
        ETOPO_COLUMNS * ETOPO_ROWS * 2,
        "the raw grid holds 21 600 × 10 800 16-bit words"
    );
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// ★ Each node's height in whole metres: the mean of the ETOPO cells whose centres fall in it, by each cell's area.
#[must_use]
pub fn bin_heights(lattice: &MacroLattice, grid: &[i16]) -> Vec<i32> {
    let n = lattice.node_count();
    let lon: Vec<(f64, f64)> = (0..ETOPO_COLUMNS)
        .map(|c| {
            let l = (-180.0 + (c as f64 + 0.5) / CELLS_PER_DEGREE).to_radians();
            (l.cos(), l.sin())
        })
        .collect();
    let rows_each = ETOPO_ROWS.div_ceil(BIN_THREADS);
    let parts: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..BIN_THREADS)
            .map(|t| {
                let lon = &lon;
                scope.spawn(move || {
                    let mut weight = vec![0.0f64; n];
                    let mut weighted = vec![0.0f64; n];
                    for r in (t * rows_each)..((t + 1) * rows_each).min(ETOPO_ROWS) {
                        let lat = (90.0 - (r as f64 + 0.5) / CELLS_PER_DEGREE).to_radians();
                        let (cos_lat, sin_lat) = (lat.cos(), lat.sin());
                        for (c, &(cos_lon, sin_lon)) in lon.iter().enumerate() {
                            let dir = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
                            let node = node_under(lattice, dir) as usize;
                            weight[node] += cos_lat;
                            weighted[node] += cos_lat * f64::from(grid[r * ETOPO_COLUMNS + c]);
                        }
                    }
                    (weight, weighted)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a binning thread ends"))
            .collect()
    });
    let mut weight = vec![0.0f64; n];
    let mut weighted = vec![0.0f64; n];
    for (w, wz) in &parts {
        for (sum, x) in weight.iter_mut().zip(w) {
            *sum += x;
        }
        for (sum, x) in weighted.iter_mut().zip(wz) {
            *sum += x;
        }
    }
    assert!(
        weight.iter().all(|&w| w > 0.0),
        "every node holds at least one cell's centre"
    );
    weighted
        .iter()
        .zip(&weight)
        .map(|(wz, w)| (wz / w).round() as i32)
        .collect()
}

/// A NumPy `.npy` array of little-endian 32-bit floats: its shape and its values (C order).
#[must_use]
pub fn read_npy_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).expect("the .npy file reads");
    assert_eq!(&b[..6], b"\x93NUMPY", "a NumPy file");
    let header_len = usize::from(u16::from_le_bytes([b[8], b[9]]));
    let header = std::str::from_utf8(&b[10..10 + header_len]).expect("an ASCII header");
    assert!(
        header.contains("'<f4'"),
        "little-endian 32-bit floats: {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "C order: {header}"
    );
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .expect("a shape");
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("a dimension"))
        .collect();
    let data: Vec<f32> = b[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "the data fills the shape"
    );
    (shape, data)
}

/// A monthly global field on a latitude–longitude grid: `[month][row][column]`, the rows' latitudes stated, the
/// columns even from a first east longitude.
pub struct MonthlyGrid {
    /// The rows' latitudes, degrees, in the file's order.
    pub lats: Vec<f64>,
    /// The first column's east longitude and the step, degrees.
    pub lon0: f64,
    pub dlon: f64,
    pub cols: usize,
    pub values: Vec<f32>,
}

impl MonthlyGrid {
    /// GPCP v2.3's monthly rain, mm/day: rows south to north at `−88.75 + 2.5·r`, columns at `1.25 + 2.5·c` east.
    #[must_use]
    pub fn gpcp(path: &str) -> MonthlyGrid {
        let (shape, values) = read_npy_f32(path);
        assert_eq!(shape, vec![12, 72, 144]);
        MonthlyGrid {
            lats: (0..72).map(|r| -88.75 + 2.5 * r as f64).collect(),
            lon0: 1.25,
            dlon: 2.5,
            cols: 144,
            values,
        }
    }

    /// NCEP/NCAR R1's monthly 10 m wind component, m/s: rows north to south on the T62 Gaussian latitudes (94),
    /// columns at `1.875·c` east.
    #[must_use]
    pub fn ncep(path: &str) -> MonthlyGrid {
        let (shape, values) = read_npy_f32(path);
        assert_eq!(shape, vec![12, 94, 192]);
        let mut lats = gaussian_latitudes(94);
        lats.reverse();
        MonthlyGrid {
            lats,
            lon0: 0.0,
            dlon: 1.875,
            cols: 192,
            values,
        }
    }

    /// The field in month `m` (0 = January) at a latitude and east longitude, degrees: bilinear between the four cells
    /// around it (the nearest row past the outermost rows).
    #[must_use]
    pub fn at(&self, m: usize, lat: f64, lon: f64) -> f64 {
        let rows = self.lats.len();
        let ascending = self.lats[1] > self.lats[0];
        // The row pair around `lat` and the share toward the second.
        let (r0, r1, wr) = {
            let mut k = 0usize;
            while k + 1 < rows
                && ((ascending && self.lats[k + 1] <= lat)
                    || (!ascending && self.lats[k + 1] >= lat))
            {
                k += 1;
            }
            if k + 1 >= rows {
                (rows - 1, rows - 1, 0.0)
            } else if (ascending && lat < self.lats[0]) || (!ascending && lat > self.lats[0]) {
                (0, 0, 0.0)
            } else {
                let w = (lat - self.lats[k]) / (self.lats[k + 1] - self.lats[k]);
                (k, k + 1, w)
            }
        };
        let x = (lon - self.lon0).rem_euclid(360.0) / self.dlon;
        let c0 = (x.floor() as usize) % self.cols;
        let c1 = (c0 + 1) % self.cols;
        let wc = x - x.floor();
        let v = |r: usize, c: usize| f64::from(self.values[(m * rows + r) * self.cols + c]);
        let a = v(r0, c0) * (1.0 - wc) + v(r0, c1) * wc;
        let b = v(r1, c0) * (1.0 - wc) + v(r1, c1) * wc;
        a * (1.0 - wr) + b * wr
    }
}

/// The `n` Gaussian latitudes, degrees, south to north: the arcsines of the roots of the Legendre polynomial `P_n`
/// (Newton from the Chebyshev guesses).
#[must_use]
pub fn gaussian_latitudes(n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        out.push(x.asin().to_degrees());
    }
    out.sort_by(f64::total_cmp);
    out
}
