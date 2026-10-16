//! ★ L3′ — THE WINDS FROM THE HEATING (the design §11.3; the research `/tmp/vd-g27/winds_laws.md`, outside the repo): the
//! steady damped Gill/Matsuno balance on the sphere in its weak-temperature-gradient limit (Gill 1980; Bretherton & Sobel
//! 2003; Shamir et al. 2023), applied to the near-surface layer with the bulk drag law. ONE law gives the zonal cells and
//! the land–sea departures — the monsoon's pull to a heated summer continent and the push from a cooled winter one.
//!
//! 1. **THE HEATING THE MEAN FLOW CARRIES** (W/m²): the column's dry-static-energy source after the storms take their part,
//!    `Q = N − ∇·F_e + L·(P − E − C_e)`, less its global mean: `N` the column's net energy input (the balance's kept sunlight
//!    less its outgoing longwave and its heat store), `F_e` the storms' share of the balance's poleward energy flow (beyond
//!    the mean cells' share `w`), `P − E` the season's rain less its evaporation and `C_e` the storms' own vapour convergence
//!    (the vapour model's, the previous round's: the lag the design states).
//! 2. **THE DRY STABILITY BETWEEN THE BRANCHES**: `Δs = h_T − c_p·T`, `h_T = 1.06 ×` the season's zonal moist energy maximum
//!    (Siler, Roe & Armour 2018, Earth, named) — positive everywhere by construction.
//! 3. **THE NEAR-SURFACE LAYER'S MASS BUDGET**: `ε_T·Φ/Δs + ∇·V = −g·Q/(β_s·p_s·Δs)`, `ε_T = B·g/(c_p·p_s)` (the balance's own
//!    longwave slope over the column's heat capacity), `β_s` = 0.2 the near-surface share of the lower branch (Earth: NCEP's
//!    10 m wind against Earth's winter cell; named).
//! 4. **THE MOMENTUM BALANCE**: `r·u − f·v = −∂Φ/∂x`, `r·v + f·u = −∂Φ/∂y`, `r = C_D·|U|/h` — the bulk drag law, `C_D` by the
//!    log law `(κ/ln(z/z₀))²` at 10 m over the surface's roughness (the sea's Charnock value as 1.3 × 10⁻³; land's roughness
//!    0.05 m, Garratt 1992 — named), `h` the lower tenth of the column (Siler et al., named), and `|U|` the wind's own speed
//!    with the convective gust `w_g` = 1 m/s (Godfrey & Beljaars 1991, named: a calm cell still mixes), so the drag is a law
//!    at every speed and no floor is needed.
//! 5. **THE SOLVE**: `u, v` eliminated face by face (each face's normal wind from its own gradient and the Coriolis turn of
//!    the averaged cross gradient), one non-symmetric elliptic system for `Φ` per round, solved EXACTLY by banded
//!    Gaussian elimination with partial pivoting (the Krylov methods stall on its Coriolis terms — MEASURED, see
//!    [`banded_solve`]); [`DRAG_ROUNDS`] rounds of
//!    the drag on the round's own wind.
//! 6. **THE STORMS' WESTERLIES** beyond the cells' edges by the steady angular momentum (the surface torque sums to nought),
//!    applied only when the cells' own torque is EASTERLY (as ExoPlaSim's is in every month on the home planet, MEASURED) —
//!    a cells' torque of the other sign is reported, never spread as storm-belt easterlies.
//!
//! MEASURED by the research on ExoPlaSim's own heating (the home planet, T21): the direction within 45° on 71–84 % of the
//! cells, the correlations 0.76–0.80 (east) and 0.68–0.77 (north); on Earth with GPCP's rain India's July westerlies
//! +6.3 m/s against NCEP's +6.0.
//!
//! **Example.** In the home planet's January the big southern continent takes the summer sun: its column's heating pulls the
//! sea's air over the coast, the spin turns it east, and the vapour it carries rains on the continent; in July the same
//! continent, cold, pushes its air out to sea.

use crate::gf::Gf;

use super::balance::{BalanceWords, LATENT_J_KG, moist_energy};
use super::grid::{ClimateGrid, cos_series};
use super::wind::{
    CELL_SHARE_WIDTH, DRAG_COEFFICIENT, EARTH_EDGE_DEG, EARTH_HALF_WIDTH, EARTH_RADIUS_M,
    EARTH_SPIN_RAD_S, LOWER_BRANCH_SHARE, TOP_ENERGY_RATIO,
};

/// ★ THE NEAR-SURFACE SHARE OF THE LOWER BRANCH (Earth: NCEP's 10 m north wind at 10° N in January against Earth's winter
/// cell; ExoPlaSim's home run reads 0.18–0.37; named).
pub const NEAR_SURFACE_SHARE: f64 = 0.2;
/// ★ THE LAND'S ROUGHNESS LENGTH, m (Garratt 1992's grass and crops; named), the log law's height (10 m) and von Kármán's
/// constant.
pub const LAND_ROUGHNESS_M: f64 = 0.05;
pub const DRAG_HEIGHT_M: f64 = 10.0;
pub const VON_KARMAN: f64 = 0.4;
/// ★ THE CONVECTIVE GUST, m/s (Godfrey & Beljaars 1991; named).
pub const GUST_M_S: f64 = 1.0;
/// The drag's rounds (each solves with the last round's wind).
pub const DRAG_ROUNDS: usize = 3;

/// One season's fields the heated winds read, per climate cell: the temperature (K), the column's net energy input `N`
/// (W/m²), the mean flow's latent heating `L·(P − E − C_e)` (W/m², the previous round's), the sea share and the humidity
/// share.
#[derive(Clone, Copy, Debug)]
pub struct Heating<'a> {
    pub t: &'a [Gf],
    pub net: &'a [Gf],
    pub latent: &'a [Gf],
    pub sea: &'a [Gf],
    pub rh: &'a [Gf],
    /// ★ THE BENCH'S SWITCH (ruling B38): the heating's departure from its row's mean only — the land–sea pull alone, for
    /// adding to a zonal wind law's own winds; no storms' westerlies then.
    pub departures_only: bool,
}

/// One season's heated winds: east and north per cell (m/s), the rising branch's latitude's sine, each row's mean cells'
/// share, whether the storms' westerlies ran (the cells' torque was easterly), and the last solve's residual (the exact
/// solve's rounding; the iteration words stay nought and met for the bench's report).
#[derive(Clone, Debug)]
pub struct HeatedWinds {
    pub east: Vec<Gf>,
    pub north: Vec<Gf>,
    pub rising_sin_lat: Gf,
    pub cell_share: Vec<Gf>,
    pub worst_iterations: usize,
    pub met: bool,
    pub storms_ran: bool,
    /// The last solve's residual over its right side (root mean squares).
    pub residual: Gf,
}

/// A sparse row: up to fifteen (column, coefficient) pairs, merged.
#[derive(Clone, Copy)]
struct Row {
    len: usize,
    col: [u32; 15],
    val: [Gf; 15],
}

impl Row {
    fn new() -> Row {
        Row {
            len: 0,
            col: [0; 15],
            val: [Gf::ZERO; 15],
        }
    }
    fn add(&mut self, c: usize, v: Gf) {
        for k in 0..self.len {
            if self.col[k] as usize == c {
                self.val[k] += v;
                return;
            }
        }
        self.col[self.len] = c as u32;
        self.val[self.len] = v;
        self.len += 1;
    }
}

/// The edges of each row (the angle from the south pole at `r·step`), as latitudes, and each row's centre latitude.
fn latitudes(grid: &ClimateGrid) -> (Vec<Gf>, Vec<Gf>) {
    let pi = Gf::from_f64(std::f64::consts::PI);
    let step = pi / Gf::from_i64(grid.rows as i64);
    let half_pi = pi * Gf::HALF;
    let edges = (0..=grid.rows)
        .map(|r| Gf::from_i64(r as i64) * step - half_pi)
        .collect();
    let centres = (0..grid.rows)
        .map(|r| (Gf::from_i64(r as i64) + Gf::HALF) * step - half_pi)
        .collect();
    (edges, centres)
}

/// ★ THE SEASON'S HEATED WINDS on a body spinning at `omega` rad/s; `speed` the last round's wind speed per cell (m/s) to
/// start the drag from (the previous pass's, or empty for a cold start at the gust).
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "one law in its six steps; split, each step would only hand the next its fields"
)]
pub fn heated_winds(
    grid: &ClimateGrid,
    words: &BalanceWords,
    omega: Gf,
    h: Heating<'_>,
    speed: &[Gf],
) -> HeatedWinds {
    let rows = grid.rows;
    let cols = grid.cols;
    let n = grid.len();
    let a = grid.radius_m;
    let pi = Gf::from_f64(std::f64::consts::PI);
    let dl = pi / Gf::from_i64(rows as i64);
    let g = words.p_surf_pa / words.column_kg_m2;
    let inv_cols = Gf::ONE / Gf::from_i64(cols as i64);
    let (edges, centres) = latitudes(grid);
    let sin_edge: Vec<Gf> = edges
        .iter()
        .map(|&e| -cos_series(e + pi * Gf::HALF))
        .collect();
    // The zonal rows: the moist energy, the net heating (W), the transport at each row's northern edge.
    let mut hz = vec![Gf::ZERO; rows];
    let mut net_row = vec![Gf::ZERO; rows];
    for i in 0..n {
        let r = grid.row(i);
        hz[r] += moist_energy(words, h.t[i], h.rh[i]).0 * inv_cols;
        net_row[r] += h.net[i] * grid.area[r];
    }
    let mut f_edge = vec![Gf::ZERO; rows];
    let mut acc = Gf::ZERO;
    for r in 0..rows {
        acc += net_row[r];
        f_edge[r] = acc;
    }
    let mut hmax_row = 0;
    for r in 1..rows {
        if hz[r] > hz[hmax_row] {
            hmax_row = r;
        }
    }
    // The rising branch: the energy-flux equator nearest the moist energy's maximum (the readings' rule).
    let mut rising = grid.sin_lat[hmax_row];
    let mut best = usize::MAX;
    for r in 0..rows - 1 {
        let below = if r == 0 { Gf::ZERO } else { f_edge[r - 1] };
        if below < Gf::ZERO && f_edge[r] >= Gf::ZERO {
            let d = r.abs_diff(hmax_row);
            if d < best {
                best = d;
                rising = sin_edge[r + 1];
            }
        }
    }
    // The cells' edges (Held's eddy limit) and the cells' share per row, as the readings' wind law states them.
    let edge_rad = Gf::from_f64(EARTH_EDGE_DEG) * pi / Gf::from_i64(180)
        * (Gf::from_f64(EARTH_SPIN_RAD_S * EARTH_RADIUS_M) / (omega * a)).sqrt();
    let s_d = if edge_rad >= pi * Gf::HALF {
        Gf::ONE
    } else {
        cos_series(pi * Gf::HALF - edge_rad)
    };
    let (lo, hi) = (rising.lesser(-s_d), rising.greater(s_d));
    let gauss = |x: Gf, half: Gf| {
        let sigma = Gf::from_f64(CELL_SHARE_WIDTH / EARTH_HALF_WIDTH) * half;
        let u = (x - rising) / sigma;
        (-(u * u)).exp()
    };
    let share = |x: Gf| {
        if x < lo || x > hi {
            return Gf::ZERO;
        }
        let half = if x >= rising {
            hi - rising
        } else {
            rising - lo
        };
        if half <= Gf::ZERO {
            return Gf::ZERO;
        }
        let at_edge = gauss(rising + half, half);
        (gauss(x, half) - at_edge) / (Gf::ONE - at_edge)
    };
    let cell_share: Vec<Gf> = grid.sin_lat.iter().map(|&x| share(x)).collect();
    // The storms' energy flow at each row's northern edge and its divergence per row (W/m²).
    let storm_edge: Vec<Gf> = (0..rows)
        .map(|r| {
            let w = if r + 1 < rows {
                (cell_share[r] + cell_share[r + 1]) * Gf::HALF
            } else {
                cell_share[r]
            };
            (Gf::ONE - w) * f_edge[r]
        })
        .collect();
    let ring = |r: usize| grid.area[r] * Gf::from_i64(cols as i64);
    let storm_div: Vec<Gf> = (0..rows)
        .map(|r| {
            let south = if r == 0 { Gf::ZERO } else { storm_edge[r - 1] };
            (storm_edge[r] - south) / ring(r)
        })
        .collect();
    // THE HEATING and THE STABILITY per cell; the heating less its global mean.
    let h_top = Gf::from_f64(TOP_ENERGY_RATIO) * hz[hmax_row];
    let ds: Vec<Gf> = (0..n).map(|i| h_top - words.cp_j_kg_k * h.t[i]).collect();
    let mut q: Vec<Gf> = (0..n)
        .map(|i| h.net[i] - storm_div[grid.row(i)] + h.latent[i])
        .collect();
    let (mut qs, mut qa) = (Gf::ZERO, Gf::ZERO);
    for (i, qi) in q.iter().enumerate() {
        qs += *qi * grid.area[grid.row(i)];
        qa += grid.area[grid.row(i)];
    }
    let q_mean = qs / qa;
    for v in &mut q {
        *v -= q_mean;
    }
    if h.departures_only {
        for r in 0..rows {
            let mut m = Gf::ZERO;
            for c in 0..cols {
                m += q[r * cols + c];
            }
            m *= inv_cols;
            for c in 0..cols {
                q[r * cols + c] -= m;
            }
        }
    }
    let beta_s = Gf::from_f64(NEAR_SURFACE_SHARE);
    let rhs: Vec<Gf> = (0..n)
        .map(|i| -(q[i] * g / (beta_s * words.p_surf_pa * ds[i])) * grid.area[grid.row(i)])
        .collect();
    let eps_t = words.b_w_m2_k * g / (words.cp_j_kg_k * words.p_surf_pa);
    // The drag per cell.
    let rho = words.p_surf_pa / (words.cp_j_kg_k / Gf::from_f64(3.5) * h.t[n / 2]);
    let depth = Gf::from_f64(LOWER_BRANCH_SHARE) * words.column_kg_m2 / rho;
    let cd_land = {
        let k = Gf::from_f64(VON_KARMAN)
            / (Gf::from_f64(DRAG_HEIGHT_M) / Gf::from_f64(LAND_ROUGHNESS_M)).ln();
        k * k
    };
    let cd_sea = Gf::from_f64(DRAG_COEFFICIENT);
    let cd: Vec<Gf> = (0..n)
        .map(|i| h.sea[i] * cd_sea + (Gf::ONE - h.sea[i]) * cd_land)
        .collect();
    let gust2 = Gf::from_f64(GUST_M_S * GUST_M_S);
    let mut spd: Vec<Gf> = if speed.len() == n {
        speed.to_vec()
    } else {
        vec![Gf::ZERO; n]
    };
    let f_c: Vec<Gf> = grid.sin_lat.iter().map(|&s| Gf::TWO * omega * s).collect();
    let f_e: Vec<Gf> = sin_edge.iter().map(|&s| Gf::TWO * omega * s).collect();
    let mut phi = vec![Gf::ZERO; n];
    let mut east = vec![Gf::ZERO; n];
    let mut north = vec![Gf::ZERO; n];
    let mut worst = 0;
    let mut all_met = true;
    let mut residual = Gf::ZERO;
    let idx = |r: usize, c: usize| r * cols + (c % cols);
    for _ in 0..DRAG_ROUNDS {
        let r_drag: Vec<Gf> = (0..n)
            .map(|i| cd[i] * (spd[i] * spd[i] + gust2).sqrt() / depth)
            .collect();
        // THE OPERATOR: each cell's outflow through its four faces, each face's normal wind eliminated.
        let mut m = vec![Row::new(); n];
        for j in 0..rows {
            let dy_c = |jj: usize| -> (usize, usize, Gf) {
                let (j0, j1) = (jj.saturating_sub(1), (jj + 1).min(rows - 1));
                (j0, j1, a * (centres[j1] - centres[j0]))
            };
            for i in 0..cols {
                let k = j * cols + i;
                // East and west faces.
                for side in [1i64, -1] {
                    let (i0, i1) = if side == 1 {
                        (i, i + 1)
                    } else {
                        (i + cols - 1, i)
                    };
                    let rr = (r_drag[idx(j, i0)] + r_drag[idx(j, i1)]) * Gf::HALF;
                    let ff = f_c[j];
                    let den = rr * rr + ff * ff;
                    let d = a * grid.cos_lat[j] * dl;
                    let length = a * dl;
                    let sgn = Gf::from_i64(side) * length;
                    m[k].add(idx(j, i1), -(rr / d / den) * sgn);
                    m[k].add(idx(j, i0), (rr / d / den) * sgn);
                    for ii in [i0, i1] {
                        let (j0, j1, dd) = dy_c(j);
                        m[k].add(idx(j1, ii), -(ff * Gf::HALF / dd / den) * sgn);
                        m[k].add(idx(j0, ii), (ff * Gf::HALF / dd / den) * sgn);
                    }
                }
                // North and south faces (none past a pole).
                for side in [1i64, -1] {
                    if (side == 1 && j == rows - 1) || (side == -1 && j == 0) {
                        continue;
                    }
                    let (j0, j1) = if side == 1 { (j, j + 1) } else { (j - 1, j) };
                    let rr = (r_drag[idx(j0, i)] + r_drag[idx(j1, i)]) * Gf::HALF;
                    let ff = f_e[j1];
                    let den = rr * rr + ff * ff;
                    let d = a * (centres[j1] - centres[j0]);
                    let length = a * grid.c_north[j0] * dl;
                    let sgn = Gf::from_i64(side) * length;
                    m[k].add(idx(j1, i), -(rr / d / den) * sgn);
                    m[k].add(idx(j0, i), (rr / d / den) * sgn);
                    for jj in [j0, j1] {
                        let dd = Gf::TWO * a * grid.cos_lat[jj] * dl;
                        m[k].add(idx(jj, i + 1), (ff * Gf::HALF / dd / den) * sgn);
                        m[k].add(idx(jj, i + cols - 1), -(ff * Gf::HALF / dd / den) * sgn);
                    }
                }
                m[k].add(k, eps_t / ds[k] * grid.area[j]);
            }
        }
        phi = solve_in_band_order(&m, &rhs, rows, cols);
        let (its, met) = (0usize, true);
        worst = worst.max(its);
        all_met &= met;
        {
            let mut ax = vec![Gf::ZERO; n];
            apply(&m, &phi, &mut ax);
            let rr: Gf = (0..n)
                .map(|i| (rhs[i] - ax[i]) * (rhs[i] - ax[i]))
                .fold(Gf::ZERO, |a, b| a + b);
            let bb: Gf = rhs.iter().map(|&v| v * v).fold(Gf::ZERO, |a, b| a + b);
            residual = (rr / bb.greater(Gf::from_f64(f64::MIN_POSITIVE))).sqrt();
        }
        // The winds at the centres from the solved field.
        for j in 0..rows {
            for i in 0..cols {
                let k = j * cols + i;
                let phx = (phi[idx(j, i + 1)] - phi[idx(j, i + cols - 1)])
                    / (Gf::TWO * a * grid.cos_lat[j] * dl);
                let (j0, j1) = (j.saturating_sub(1), (j + 1).min(rows - 1));
                let phy =
                    (phi[j1 * cols + i] - phi[j0 * cols + i]) / (a * (centres[j1] - centres[j0]));
                let rr = r_drag[k];
                let ff = f_c[j];
                let den = rr * rr + ff * ff;
                east[k] = (-rr * phx - ff * phy) / den;
                north[k] = (-rr * phy + ff * phx) / den;
            }
        }
        for i in 0..n {
            spd[i] = (east[i] * east[i] + north[i] * north[i]).sqrt();
        }
    }
    // THE STORMS' WESTERLIES beyond the edges, when the cells' torque is easterly.
    let mut torque = Gf::ZERO;
    let mut shape_torque = Gf::ZERO;
    let mut shape = vec![Gf::ZERO; rows];
    for r in 0..rows {
        let x = grid.sin_lat[r];
        let mut ring_u = Gf::ZERO;
        for c in 0..cols {
            let i = r * cols + c;
            ring_u += rho * cd[i] * (spd[i] * spd[i] + gust2).sqrt() * east[i];
        }
        let beyond = if x > hi {
            x - hi
        } else if x < lo {
            lo - x
        } else {
            Gf::ZERO
        };
        if beyond > Gf::ZERO {
            let half = if x > hi { hi - rising } else { rising - lo };
            let sigma = Gf::from_f64(CELL_SHARE_WIDTH / EARTH_HALF_WIDTH)
                * half.greater(Gf::from_f64(1e-3));
            let u = beyond / sigma;
            let f_mid = ((if r == 0 { Gf::ZERO } else { f_edge[r - 1] }) + f_edge[r]) * Gf::HALF;
            shape[r] = (Gf::ONE - (-(u * u)).exp()) * f_mid.abs();
            shape_torque += shape[r] * a * grid.cos_lat[r] * ring(r);
        } else {
            torque += ring_u * inv_cols * a * grid.cos_lat[r] * ring(r);
        }
    }
    let storms_ran = !h.departures_only && torque < Gf::ZERO && shape_torque > Gf::ZERO;
    if storms_ran {
        for (r, &sh) in shape.iter().enumerate() {
            if sh <= Gf::ZERO {
                continue;
            }
            let stress = -torque * sh / shape_torque;
            for c in 0..cols {
                let i = r * cols + c;
                let k = rho * cd[i];
                let m_u = (stress.abs() / k).sqrt();
                east[i] += if stress < Gf::ZERO { -m_u } else { m_u };
            }
        }
    }
    HeatedWinds {
        east,
        north,
        rising_sin_lat: rising,
        cell_share,
        worst_iterations: worst,
        met: all_met,
        storms_ran,
        residual,
    }
}

/// `y = M·x`.
fn apply(m: &[Row], x: &[Gf], y: &mut [Gf]) {
    for (k, row) in m.iter().enumerate() {
        let mut s = Gf::ZERO;
        for t in 0..row.len {
            s += row.val[t] * x[row.col[t] as usize];
        }
        y[k] = s;
    }
}

/// ★ THE BAND ORDER: the cells by latitude row and, inside a row, the longitudes FOLDED at the date line — 0, cols − 1, 1,
/// cols − 2, … — so every east or west neighbour (the date line's pair too) sits at most two places away. The operator
/// couples a cell to its own row and the rows beside it, its diagonal neighbours included, so the farthest coupling is
/// `cols + 2` places away: half the row-major order's `2·cols − 1` (its date line's diagonal neighbour), and a quarter of
/// the elimination's work.
#[must_use]
fn band_order(rows: usize, cols: usize) -> Vec<usize> {
    let fold = |c: usize| {
        if 2 * c < cols {
            2 * c
        } else {
            2 * (cols - 1 - c) + 1
        }
    };
    (0..rows * cols)
        .map(|k| (k / cols) * cols + fold(k % cols))
        .collect()
}

/// `M·x = b` solved exactly in [`band_order`]: the rows and the unknowns renamed by the order, the banded solve at the
/// half-band `cols + 2`, the answer named back.
fn solve_in_band_order(m: &[Row], b: &[Gf], rows: usize, cols: usize) -> Vec<Gf> {
    let order = band_order(rows, cols);
    let mut pm = vec![Row::new(); m.len()];
    let mut pb = vec![Gf::ZERO; b.len()];
    for (k, row) in m.iter().enumerate() {
        let to = &mut pm[order[k]];
        for t in 0..row.len {
            to.add(order[row.col[t] as usize], row.val[t]);
        }
        pb[order[k]] = b[k];
    }
    let x = banded_solve(&pm, &pb, cols + 2);
    order.iter().map(|&p| x[p]).collect()
}

/// ★ THE EXACT BANDED SOLVE: Gaussian elimination with partial pivoting inside the band (LAPACK's `gbsv` scheme): every
/// entry within `half` places of the diagonal ([`band_order`] puts the operator's there); pivoting can widen the upper
/// band by as much again. An entry outside the band is a defect the solve refuses (it panics), never a silent drop.
/// Exact to the word's rounding, one answer on every host, and immune to the Coriolis terms' non-normality that stalls a
/// Krylov method (MEASURED: GMRES with ILU(0) left 14 % of the residual after 4 000 iterations on Earth's 2° grid).
fn banded_solve(m: &[Row], b: &[Gf], half: usize) -> Vec<Gf> {
    let n = b.len();
    let kl = half;
    let ku = half;
    let width = 2 * kl + ku + 1;
    // Band storage: row i holds columns i − kl … i + kl + ku at offsets 0 … width − 1.
    let mut a = vec![Gf::ZERO; n * width];
    let at = |i: usize, j: usize| i * width + (j + kl - i);
    for (i, row) in m.iter().enumerate() {
        for t in 0..row.len {
            let j = row.col[t] as usize;
            assert!(
                j + kl >= i && j <= i + ku,
                "an entry outside the band: row {i}, column {j}"
            );
            a[at(i, j)] += row.val[t];
        }
    }
    let mut x = b.to_vec();
    for k in 0..n {
        // The pivot: the largest entry in column k among rows k … k + kl (the first on a tie).
        let last = (k + kl).min(n - 1);
        let mut p = k;
        for i in k + 1..=last {
            if a[at(i, k)].abs() > a[at(p, k)].abs() {
                p = i;
            }
        }
        let jmax = (k + kl + ku).min(n - 1);
        if p != k {
            for j in k..=jmax {
                let (ik, ip) = (at(k, j), at(p, j));
                a.swap(ik, ip);
            }
            x.swap(k, p);
        }
        let pivot = a[at(k, k)];
        for i in k + 1..=last {
            let l = a[at(i, k)] / pivot;
            if l == Gf::ZERO {
                continue;
            }
            a[at(i, k)] = Gf::ZERO;
            for j in k + 1..=jmax {
                let akj = a[at(k, j)];
                a[at(i, j)] -= l * akj;
            }
            let xk = x[k];
            x[i] -= l * xk;
        }
    }
    for k in (0..n).rev() {
        let jmax = (k + kl + ku).min(n - 1);
        let mut sum = x[k];
        for j in k + 1..=jmax {
            sum -= a[at(k, j)] * x[j];
        }
        x[k] = sum / a[at(k, k)];
    }
    x
}

/// The latent heating `L·x` of a water rate `x` (kg/m²/s), W/m².
#[must_use]
pub fn latent_w_m2(x: Gf) -> Gf {
    Gf::from_f64(LATENT_J_KG) * x
}
