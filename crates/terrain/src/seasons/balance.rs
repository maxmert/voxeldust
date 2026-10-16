//! ★ L2 — THE SEASONAL MOIST ENERGY BALANCE on the climate grid (the design §3.2; North, Short & Mengel 1983's land–sea
//! structure, Siler, Roe & Armour 2018's moist transport):
//!
//! `C·∂T/∂t = Q·(1 − α) − (A + B·T) + ∇·((p_s/g)·D·∇h)`, `h = c_p·T + L·RH·q*(T)`,
//!
//! solved by IMPLICIT STEPS through the year (the proven route: spacesim's and climlab's). Each step is one linear solve:
//! the moist diffusivity `∂h/∂T` is taken at the step's start as the mean of each edge's two cells (so the operator is
//! symmetric and positive, and a Jacobi-preconditioned conjugate gradient in a fixed order solves it), and the moist
//! part's curvature over the edge — `h_j − h_i − ∂h/∂T·(T_j − T_i)` — rides the step's right side. The year repeats
//! until two years agree ([`periodic_year`]).
//!
//! The constants and their bodies (B41): `B` = 1.9 W/(m²·K) and the Bond albedo (today's climate law, Earth); `D` =
//! 1.16 × 10⁶ m²/s (Siler et al. 2018, fitted on ERA-Interim, Earth); the sea's 50 m mixed layer (today's, Earth's
//! mid-latitude ocean); the latent heat of water at 0 °C, 2.501 × 10⁶ J/kg (water's own). `A` is no constant: it is
//! solved so the body's annual global mean is the charter's surface temperature, `A = Q̄·(1 − α) − B·T_charter` (the
//! diffusion conserves energy).
//!
//! **Example.** On the home planet the north polar continent stands under a sun that sets for a quarter of the year and
//! shines all day at midsummer; its small heat capacity lets it swing tens of kelvin, while the sea beside it, fifty
//! metres of water deep, swings a few. The energy the balance carries from the summer continent is what the rain belt
//! and the monsoon read next.

use crate::climate::saturation_vapour_pa;
use crate::gf::Gf;

use super::grid::{ClimateGrid, cyclic_three_band};
use super::sun::SunTable;

/// ★ THE LATENT HEAT OF VAPORISATION of water at 0 °C, J/kg (water's own property).
pub const LATENT_J_KG: f64 = 2.501e6;
/// ★ THE MOIST DIFFUSIVITY of the energy balance, m²/s (Siler, Roe & Armour 2018, fitted on ERA-Interim: calibration
/// body Earth, a constant the law cannot compute yet).
pub const DIFFUSIVITY_M2_S: f64 = 1.16e6;
/// The molar mass of water, kg/mol (water's own).
pub const WATER_KG_MOL: f64 = 18.015e-3;
/// ★ THE CONJUGATE GRADIENT'S STOP: every cell's residual, divided by its diagonal, under this many kelvin.
pub const SOLVE_TOL_K: f64 = 1e-7;
/// ★ THE ICE–ALBEDO LAW (Williams & Kasting 1997; North, Short & Mengel 1983; Earth's calibration, named): a surface
/// fully frozen below 263.15 K and free above 273.15 K, linear between, reflects `ICE_CONTRAST` more of the sunlight
/// than the ice-free surface (Earth's ice 0.62 against its ice-free planet's 0.30). The ice-free albedo is SOLVED so the
/// body's sunlight-weighted mean albedo over the year is the charter's Bond albedo — the same closure the outgoing
/// longwave's `A` makes, so the charter's two words still hold exactly.
pub const ICE_FROZEN_K: f64 = 263.15;
pub const ICE_FREE_K: f64 = 273.15;
pub const ICE_CONTRAST: f64 = 0.32;
/// A bound on the conjugate gradient's iterations: never reached by a well-posed step (the bench reports the count);
/// past it the step returns what it holds and says so.
pub const SOLVE_BOUND: usize = 20_000;

/// The body's words the balance reads, computed once from the charter.
#[derive(Clone, Copy, Debug)]
pub struct BalanceWords {
    /// The outgoing longwave's sensitivity `B`, W/(m²·K).
    pub b_w_m2_k: Gf,
    /// `1 − α`, the share of the sunlight the body keeps (the charter's Bond albedo `α`, the year's mean).
    pub kept: Gf,
    /// Whether ice can form: the body holds water (a dry body's surface does not frost into ice sheets).
    pub ice: bool,
    /// The charter's annual global mean surface temperature, K.
    pub t_mean_k: Gf,
    /// The air column's mass, `p_s/g`, kg/m² (zero without air).
    pub column_kg_m2: Gf,
    /// The surface pressure, Pa.
    pub p_surf_pa: Gf,
    /// The air's heat capacity at constant pressure, J/(kg·K).
    pub cp_j_kg_k: Gf,
    /// The ratio of water's molar mass to the air's.
    pub eps: Gf,
    /// The moist diffusivity `D`, m²/s.
    pub diffusivity_m2_s: Gf,
    /// The year, seconds.
    pub year_s: Gf,
}

/// The surface each cell stands on for one year: its heat capacity, J/(m²·K), and its relative humidity's share.
#[derive(Clone, Debug)]
pub struct Surface {
    pub capacity: Vec<Gf>,
    pub rh: Vec<Gf>,
}

/// ★ THE BALANCE'S WORDS from the charter, by law: the air column `p_s/g`, the air's heat capacity `c_p = (7/2)·R/μ` (a
/// diatomic gas, as the lapse rate reads it), water's mass ratio `μ_water/μ`; a body with no air has no column, so its
/// cells carry no heat and no vapour between them.
#[must_use]
pub fn balance_words(
    body: &crate::body::BodyDefinition,
    words: &crate::solve::SolveWords,
) -> BalanceWords {
    use crate::climate::{GAS_CONSTANT_J_MOL_K, OLR_SENSITIVITY_W_M2_K};
    let g = Gf::from_i64(i64::from(body.facts().gravity_mm_s2)) / Gf::from_i64(1_000);
    let p = Gf::from_i64(i64::from(words.p_surf_pa.unwrap_or(0)));
    let mu = Gf::from_i64(i64::from(words.mu_q8.unwrap_or(0))) * Gf::from_f64(1.0e-3 / 256.0);
    let (cp, eps) = if mu > Gf::ZERO {
        (
            Gf::from_f64(3.5) * Gf::from_f64(GAS_CONSTANT_J_MOL_K) / mu,
            Gf::from_f64(WATER_KG_MOL) / mu,
        )
    } else {
        (Gf::ZERO, Gf::ZERO)
    };
    BalanceWords {
        b_w_m2_k: Gf::from_f64(OLR_SENSITIVITY_W_M2_K),
        kept: Gf::ONE - Gf::from_i64(i64::from(words.bond_albedo_q12)) / Gf::from_i64(4_096),
        ice: words.water_km3 > 0,
        t_mean_k: Gf::from_i64(i64::from(words.t_surface_mk.unwrap_or(words.t_eq_mk)))
            / Gf::from_i64(1_000),
        column_kg_m2: p / g,
        p_surf_pa: p,
        cp_j_kg_k: cp,
        eps,
        diffusivity_m2_s: Gf::from_f64(DIFFUSIVITY_M2_S),
        year_s: Gf::from_i64(words.year_s as i64),
    }
}

/// ★ THE SEASONAL HEAT CAPACITY of a cell with sea share `sea`, J/(m²·K) (NSM83: land and sea differ only by their
/// capacity): the sea's 50 m mixed layer, and on land the air column over it plus the ground's annual skin (today's rock
/// law, `crate::climate::seasonal_capacity_j_m2_k`'s dry half).
#[must_use]
pub fn capacity_j_m2_k(words: &BalanceWords, sea: Gf, year_s: u64) -> Gf {
    let water = crate::climate::seasonal_capacity_j_m2_k(Gf::ONE, year_s);
    let rock = crate::climate::seasonal_capacity_j_m2_k(Gf::ZERO, year_s);
    sea * water + (Gf::ONE - sea) * (words.column_kg_m2 * words.cp_j_kg_k + rock)
}

/// The moist static energy `h = c_p·T + L·RH·q*(T)` and its slope `∂h/∂T`, J/kg and J/(kg·K) (no vapour where there is
/// no air).
#[must_use]
pub fn moist_energy(words: &BalanceWords, t_k: Gf, rh: Gf) -> (Gf, Gf) {
    if words.p_surf_pa <= Gf::ZERO {
        return (words.cp_j_kg_k * t_k, words.cp_j_kg_k);
    }
    let l = Gf::from_f64(LATENT_J_KG);
    let e = saturation_vapour_pa(t_k);
    let q = words.eps * e / words.p_surf_pa;
    // Tetens's slope: d ln e_s / dT = 17.27·237.3/(t + 237.3)², t in °C.
    let tc =
        t_k - Gf::from_f64(crate::climate::FREEZING_K) + Gf::from_f64(crate::climate::TETENS_C_C);
    let dlog = Gf::from_f64(crate::climate::TETENS_B * crate::climate::TETENS_C_C) / (tc * tc);
    let h = words.cp_j_kg_k * t_k + l * rh * q;
    let dh = words.cp_j_kg_k + l * rh * q * dlog;
    (h, dh)
}

/// The global annual mean of the sunlight over the grid, W/m², by each cell's area.
#[must_use]
pub fn mean_sun(grid: &ClimateGrid, sun: &SunTable) -> Gf {
    let (mut s, mut a) = (Gf::ZERO, Gf::ZERO);
    for r in 0..grid.rows {
        s += sun.harmonic(grid.sin_lat[r], 0).0 * grid.area[r];
        a += grid.area[r];
    }
    s / a
}

/// One year of the balance: the temperature at the end of each step.
#[derive(Clone, Debug)]
pub struct Year {
    /// `[step][cell]`, K.
    pub t: Vec<Vec<Gf>>,
    /// `[step][cell]`, the sunlight kept over the step, W/m².
    pub kept: Vec<Vec<Gf>>,
    /// The conjugate gradient's largest iteration count in the year, and whether every step met its stop.
    pub worst_iterations: usize,
    pub all_met: bool,
    /// The year's sunlight-weighted mean albedo the ice gave (the closure reads it).
    pub mean_albedo: Gf,
}

/// The frozen share of a surface at `t_k` (one below [`ICE_FROZEN_K`], nought above [`ICE_FREE_K`], linear between).
#[must_use]
pub fn frozen_share(t_k: Gf) -> Gf {
    let lo = Gf::from_f64(ICE_FROZEN_K);
    let hi = Gf::from_f64(ICE_FREE_K);
    if t_k <= lo {
        Gf::ONE
    } else if t_k >= hi {
        Gf::ZERO
    } else {
        (hi - t_k) / (hi - lo)
    }
}

/// ★ ONE YEAR from `start` (the temperature at the year's start, K) in `2^steps_shift` implicit steps, with the ice-free
/// albedo `alpha_free` (each cell's albedo by its frozen share at the step's start).
#[must_use]
pub fn balance_year(
    grid: &ClimateGrid,
    words: &BalanceWords,
    sun: &SunTable,
    steps_shift: u32,
    surface: &Surface,
    start: &[Gf],
    alpha_free: Gf,
) -> Year {
    let n = grid.len();
    let steps = 1usize << steps_shift;
    let dt = words.year_s / Gf::from_i64(steps as i64);
    let a_olr = mean_sun(grid, sun) * words.kept - words.b_w_m2_k * words.t_mean_k;
    let k0 = words.column_kg_m2 * words.diffusivity_m2_s;
    let mut t = start.to_vec();
    let mut out = Year {
        t: Vec::with_capacity(steps),
        kept: Vec::with_capacity(steps),
        worst_iterations: 0,
        all_met: true,
        mean_albedo: Gf::ZERO,
    };
    let contrast = if words.ice {
        Gf::from_f64(ICE_CONTRAST)
    } else {
        Gf::ZERO
    };
    let (mut reflected, mut incoming) = (Gf::ZERO, Gf::ZERO);
    // The step's matrix: the diagonal and each edge's coupling (symmetric).
    let mut diag = vec![Gf::ZERO; n];
    let mut coupling = vec![[Gf::ZERO; 4]; n];
    let mut rhs = vec![Gf::ZERO; n];
    for step in 0..steps {
        let row_sun: Vec<Gf> = (0..grid.rows)
            .map(|r| sun.season_mean(grid.sin_lat[r], step, steps_shift))
            .collect();
        let kept: Vec<Gf> = (0..n)
            .map(|i| {
                let q = row_sun[grid.row(i)];
                let albedo = alpha_free + contrast * frozen_share(t[i]);
                let area = grid.area[grid.row(i)];
                reflected += q * albedo * area;
                incoming += q * area;
                q * (Gf::ONE - albedo)
            })
            .collect();
        let he: Vec<(Gf, Gf)> = (0..n)
            .map(|i| moist_energy(words, t[i], surface.rh[i]))
            .collect();
        for i in 0..n {
            let area = grid.area[grid.row(i)];
            let ca = surface.capacity[i] * area / dt;
            let mut d = ca + words.b_w_m2_k * area;
            let mut b = ca * t[i] + area * (kept[i] - a_olr);
            for (slot, edge) in grid.neighbours(i).into_iter().enumerate() {
                let Some((m, c)) = edge else {
                    coupling[i][slot] = Gf::ZERO;
                    continue;
                };
                let dh = (he[i].1 + he[m].1) * Gf::HALF;
                let kk = k0 * c * dh;
                coupling[i][slot] = kk;
                d += kk;
                // The curvature the linear coupling misses over this edge, at the step's start.
                b += k0 * c * ((he[m].0 - he[i].0) - dh * (t[m] - t[i]));
            }
            diag[i] = d;
            rhs[i] = b;
        }
        let (iterations, met) = conjugate_gradient(grid, &diag, &coupling, &rhs, &mut t);
        out.worst_iterations = out.worst_iterations.max(iterations);
        out.all_met &= met;
        out.t.push(t.clone());
        out.kept.push(kept);
    }
    out.mean_albedo = if incoming > Gf::ZERO {
        reflected / incoming
    } else {
        Gf::ONE - words.kept
    };
    out
}

/// `y = M·x` for the step's matrix: the diagonal less each edge's coupling to its neighbour.
fn apply(grid: &ClimateGrid, diag: &[Gf], coupling: &[[Gf; 4]], x: &[Gf], y: &mut [Gf]) {
    for i in 0..x.len() {
        let mut s = diag[i] * x[i];
        for (slot, edge) in grid.neighbours(i).into_iter().enumerate() {
            if let Some((m, _)) = edge {
                s -= coupling[i][slot] * x[m];
            }
        }
        y[i] = s;
    }
}

/// ★ THE ROW-PRECONDITIONED CONJUGATE GRADIENT on the step's symmetric positive matrix, in the cells' own order, from
/// `x`'s start; its iterations and whether every residual met [`SOLVE_TOL_K`]. The preconditioner solves each latitude
/// row's own east–west coupling EXACTLY (its cyclic three-band block, symmetric and positive as the matrix is): the
/// stiffness of the thin polar rows lives along the rows, so the iterations no longer pay for it.
fn conjugate_gradient(
    grid: &ClimateGrid,
    diag: &[Gf],
    coupling: &[[Gf; 4]],
    rhs: &[Gf],
    x: &mut [Gf],
) -> (usize, bool) {
    let n = x.len();
    let cols = grid.cols;
    let tol = Gf::from_f64(SOLVE_TOL_K);
    let mut ax = vec![Gf::ZERO; n];
    apply(grid, diag, coupling, x, &mut ax);
    let mut r: Vec<Gf> = (0..n).map(|i| rhs[i] - ax[i]).collect();
    let met = |r: &[Gf]| (0..n).all(|i| (r[i] / diag[i]).abs() < tol);
    if met(&r) {
        return (0, true);
    }
    let (mut lo, mut di, mut up, mut rr, mut zz) = (
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
    );
    let mut precondition = |r: &[Gf], z: &mut [Gf]| {
        for row in 0..grid.rows {
            for c in 0..cols {
                let i = row * cols + c;
                lo[c] = coupling[i][0];
                up[c] = coupling[i][1];
                di[c] = diag[i];
                rr[c] = r[i];
            }
            cyclic_three_band(&lo, &di, &up, &rr, &mut zz);
            z[row * cols..(row + 1) * cols].copy_from_slice(&zz);
        }
    };
    let mut z = vec![Gf::ZERO; n];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![Gf::ZERO; n];
    for it in 1..=SOLVE_BOUND {
        apply(grid, diag, coupling, &p, &mut ap);
        let alpha = rz / dot(&p, &ap);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if met(&r) {
            return (it, true);
        }
        precondition(&r, &mut z);
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    (SOLVE_BOUND, false)
}

fn dot(a: &[Gf], b: &[Gf]) -> Gf {
    let mut s = Gf::ZERO;
    for (x, y) in a.iter().zip(b) {
        s += *x * *y;
    }
    s
}

/// A periodic year and how it was reached.
#[derive(Clone, Debug)]
pub struct Periodic {
    pub year: Year,
    /// The years stepped, and the largest change of a cell's year-start temperature over the last of them, K.
    pub years: usize,
    pub change_k: Gf,
    /// The ice-free albedo the closure found.
    pub alpha_free: Gf,
}

/// ★ THE PERIODIC YEAR: years stepped from `start` until no cell's year-start temperature moves more than `stop.0` kelvin
/// over a year, within `stop.1` years (a bound the bench reads, never a law: past it the last year is returned with its
/// change).
#[must_use]
pub fn periodic_year(
    grid: &ClimateGrid,
    words: &BalanceWords,
    sun: &SunTable,
    steps_shift: u32,
    surface: &Surface,
    start: &[Gf],
    stop: (Gf, usize),
) -> Periodic {
    periodic_year_from(
        grid,
        words,
        sun,
        steps_shift,
        surface,
        start,
        Gf::ONE - words.kept,
        stop,
    )
}

/// [`periodic_year`] from a known ice-free albedo (a warm start: the last pass's).
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the balance's grid, words, sun, steps, surface and its two warm-start words"
)]
pub fn periodic_year_from(
    grid: &ClimateGrid,
    words: &BalanceWords,
    sun: &SunTable,
    steps_shift: u32,
    surface: &Surface,
    start: &[Gf],
    alpha_start: Gf,
    stop: (Gf, usize),
) -> Periodic {
    let (tol_k, bound) = stop;
    let mut from = start.to_vec();
    let mut years = 0;
    // The ice-free albedo's closure: move it each year by the year's miss.
    let target = Gf::ONE - words.kept;
    let mut alpha_free = alpha_start;
    let albedo_tol = Gf::from_f64(1e-4);
    loop {
        let year = balance_year(grid, words, sun, steps_shift, surface, &from, alpha_free);
        years += 1;
        let end = year.t.last().expect("a year has steps");
        let mut change = Gf::ZERO;
        for (a, b) in end.iter().zip(&from) {
            change = change.greater((*a - *b).abs());
        }
        let miss = target - year.mean_albedo;
        if (change < tol_k && miss.abs() < albedo_tol) || years >= bound {
            return Periodic {
                year,
                years,
                change_k: change,
                alpha_free,
            };
        }
        alpha_free += miss;
        from.clone_from(end);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::seasons::sun::YEAR_SAMPLES_SHIFT;
    use crate::solve::SolveWords;

    /// Earth's charter on the home planet's body (the G-ICE test's words), with the tilt's cosine `cos_tilt_q1024`.
    pub(crate) fn earth_words(cos_tilt_q1024: i32) -> SolveWords {
        SolveWords {
            insolation_q12: 4_096,
            t_eq_mk: 255_000,
            t_surface_mk: Some(288_000),
            bond_albedo_q12: 1_253,
            mu_q8: Some(7_416),
            scale_height_m: Some(8_500),
            p_surf_pa: Some(101_325),
            tau_ir_q12: Some(2_458),
            day_s: Some(86_400),
            obliquity_cos_q1024: Some(cos_tilt_q1024),
            ecc_q16: 0,
            year_s: 31_557_600,
            water_km3: 1_386_000_000,
            ..crate::home::home_solve_words()
        }
    }

    /// An aquaplanet of `rows` latitude bands on the home planet's radius: its grid, words, sun and surface.
    pub(crate) fn aquaplanet(
        rows: usize,
        cos_tilt_q1024: i32,
    ) -> (ClimateGrid, BalanceWords, SunTable, Surface, SolveWords) {
        let body = crate::home::home_planet();
        let words = earth_words(cos_tilt_q1024);
        let bw = balance_words(&body, &words);
        let grid = ClimateGrid::new(Gf::from_f64(body.ladder().radius_m()), rows);
        let sun = SunTable::new(
            Gf::from_f64(crate::climate::SOLAR_CONSTANT_W_M2),
            Gf::ZERO,
            Gf::from_i64(i64::from(cos_tilt_q1024)) / Gf::from_i64(1_024),
            257,
            YEAR_SAMPLES_SHIFT,
        );
        let surface = Surface {
            capacity: vec![capacity_j_m2_k(&bw, Gf::ONE, words.year_s); grid.len()],
            rh: vec![Gf::from_f64(0.8); grid.len()],
        };
        (grid, bw, sun, surface, words)
    }

    /// ★ THE BALANCE KEEPS THE CHARTER: on an aquaplanet with no tilt, the periodic year's global mean is the charter's
    /// 288 K (the closure), the two hemispheres mirror each other, the equator is warmer than the poles, and the ice-free
    /// albedo's closure meets the charter's mean albedo.
    #[test]
    fn the_balance_keeps_the_charters_mean_and_its_symmetry() {
        let (grid, bw, sun, surface, _) = aquaplanet(18, 1_024);
        let start = vec![bw.t_mean_k; grid.len()];
        let p = periodic_year(
            &grid,
            &bw,
            &sun,
            5,
            &surface,
            &start,
            (Gf::from_f64(0.01), 400),
        );
        assert!(p.year.all_met);
        assert!(p.years < 400);
        let steps = p.year.t.len();
        let annual: Vec<f64> = (0..grid.len())
            .map(|i| (0..steps).map(|s| p.year.t[s][i].to_f64()).sum::<f64>() / steps as f64)
            .collect();
        let (mut sum, mut area) = (0.0, 0.0);
        for (i, a) in annual.iter().enumerate() {
            sum += a * grid.area[grid.row(i)].to_f64();
            area += grid.area[grid.row(i)].to_f64();
        }
        assert!((sum / area - 288.0).abs() < 0.05, "{}", sum / area);
        let row = |r: usize| {
            (0..grid.cols)
                .map(|c| annual[r * grid.cols + c])
                .sum::<f64>()
                / grid.cols as f64
        };
        for r in 0..grid.rows / 2 {
            assert!((row(r) - row(grid.rows - 1 - r)).abs() < 0.02, "row {r}");
        }
        assert!(row(grid.rows / 2) > row(grid.rows - 1) + 20.0);
        assert!((p.year.mean_albedo - (Gf::ONE - bw.kept)).abs().to_f64() < 1e-3);
        // The frozen share's three sides.
        assert_eq!(frozen_share(Gf::from_f64(250.0)), Gf::ONE);
        assert_eq!(frozen_share(Gf::from_f64(280.0)), Gf::ZERO);
        assert!((frozen_share(Gf::from_f64(268.15)).to_f64() - 0.5).abs() < 1e-12);
    }

    /// ★ A BODY WITH NO AIR carries no heat and no vapour: no column, no heat capacity of air, a dry moist energy.
    #[test]
    fn an_airless_body_has_no_column_and_no_vapour() {
        let moon = crate::home::home_moon();
        let words = crate::home::home_moon_solve_words();
        let bw = balance_words(&moon, &words);
        assert_eq!(bw.column_kg_m2, Gf::ZERO);
        assert!(!bw.ice);
        let (h, dh) = moist_energy(&bw, Gf::from_f64(250.0), Gf::from_f64(0.8));
        assert_eq!(h, bw.cp_j_kg_k * Gf::from_f64(250.0));
        assert_eq!(dh, bw.cp_j_kg_k);
    }
}
