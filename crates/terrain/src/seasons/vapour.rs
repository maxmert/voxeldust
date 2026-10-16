//! ★ L4 + L5 — THE EVAPORATION, THE VAPOUR AND THE RAIN of one season on the climate grid (the design §3.4, §3.5).
//!
//! **THE EVAPORATION** (Priestley & Taylor 1972): `E = α_PT·Δ/(Δ + γ)·(R_n − G)/L`, `α_PT` = 1.26 (saturated surfaces,
//! Earth's sea and wet land, named), `Δ` Clausius–Clapeyron's slope at the cell's temperature, `γ = c_p·p_s/(ε·L)`; the
//! net radiation at the surface by Earth's split AS SHARES OF THE BODY'S OWN SUNLIGHT (Trenberth, Fasullo & Kiehl 2009;
//! constants the law cannot compute yet, named): of the 240 W/m² Earth keeps, the surface takes 161 and loses 63 net as
//! longwave, so `R_n = (161/240)·kept − (63/240)·kept̄`, `kept̄` the body's own global annual mean (a body with less
//! sunlight holds its 288 K by a stronger greenhouse, so its surface's net longwave loss is smaller in proportion —
//! MEASURED need: with Earth's fixed 63 W/m² the home planet, at 0.75 of Earth's sunlight, evaporated ~30 % under the
//! ExoPlaSim reading); `G` the sea's heat store from the energy balance. Where the available energy is
//! negative the sea takes vapour back as DEW: a sink of the column's own vapour, `|E|·W/W_sat` — it can take only the
//! vapour there is. A sea under its freezing point evaporates nothing (the phase rule, physics). Over land the same law is
//! the POTENTIAL evaporation.
//!
//! **THE VAPOUR**: the column's water `W` (kg/m²) in a season's steady state,
//! `∇·(W·v) − ∇·((1 − w)·D·∇W) + P(W) = E + E_land`, by finite volumes on the climate grid: the upwind flux across each
//! face with the face's own wind, the storms' eddy diffusion with the energy balance's own `D` weighted by the eddies'
//! share `1 − w` (Siler, Roe & Armour 2018's partition: inside the mean cells the cells' own flow carries the vapour —
//! toward the rain belt, which dries the subtropics), and THE RAIN BY THE COLUMN'S SATURATION (Bretherton, Peters & Back
//! 2004, MEASURED over the tropical oceans; named, its use beyond them an extrapolation the readings judge):
//! `P = exp(a·(r − b))` mm/day, `a` = 15.6, `b` = 0.603, `r = W/W_sat`, the saturated column `W_sat = e_s(T_H)·H₀/(R_v·T_H)`
//! at the air over the cell's ground, `T_H = T − Γ·H`, the vapour's scale height `H₀ = R_v·T_H²/(L·Γ)` (Clausius–Clapeyron
//! with the lapse rate). Converging air rains long before it piles into a line; sinking air far from saturation barely
//! rains (the deserts); high ground's colder air saturates sooner (the windward slope's rain). The vapour's residence
//! time is then a READING (Earth's measured 8.9 days, van der Ent & Tuinenburg 2017).
//!
//! Solved by ROW RELAXATION (the latitude–longitude grid's standard): each row's cyclic three-band system is solved
//! exactly given the rows beside it, sweeping north then south, the rain's steep law linearised at the sweep's column
//! (NEWTON: the rain is convex in the column and the system an M-matrix, so the steps fall monotonically onto the
//! solution from above and every column stays positive — Ortega & Rheinboldt's M-function theorem), until no cell's
//! column moves more than a stated share in a sweep. The rain is `P(W)`: everything the sea and the land give
//! comes down somewhere, so the body's rain is its evaporation.
//!
//! **Example.** In Earth's July the summer sea south of India evaporates by its stored heat; the monsoon's westerlies
//! carry the vapour onto the coast; the Western Ghats lift it, its rain-out time there falls from days to hours, and the
//! coast takes most of it — the plateau behind gets the rest.

use crate::climate::saturation_vapour_pa;
use crate::gf::Gf;

use super::balance::{BalanceWords, LATENT_J_KG};
use super::grid::{ClimateGrid, cyclic_three_band};

/// ★ PRIESTLEY & TAYLOR's coefficient for saturated surfaces (Earth's sea and wet land; named).
pub const PT_ALPHA: f64 = 1.26;
/// ★ EARTH'S SURFACE ENERGY SPLIT (Trenberth, Fasullo & Kiehl 2009; named), as shares of the sunlight the planet keeps:
/// the surface's sunlight (161 of 240 W/m²) and its net longwave loss (63 of 240), the latter of the body's own mean.
pub const SURFACE_SW_SHARE: f64 = 161.0 / 240.0;
pub const SURFACE_LW_SHARE: f64 = 63.0 / 240.0;
/// ★ BRETHERTON, PETERS & BACK 2004's rain by the column's saturation, `P = exp(a·(r − b))` mm/day (the tropical
/// oceans; named).
pub const RAIN_SLOPE: f64 = 15.6;
pub const RAIN_OFFSET: f64 = 0.603;
/// The gas constant of water vapour, J/(kg·K) (water's own).
pub const R_VAPOUR: f64 = 461.5;
/// Sea water's freezing point, K (water's own, at the sea's salt).
pub const SEA_FREEZING_K: f64 = 271.35;
/// The sweeps' stop: no cell's column moves more than this share of it in a sweep.
pub const SWEEP_TOL: f64 = 1e-7;
/// A bound on the sweeps, never reached by a well-posed season (the bench reports the count).
pub const SWEEP_BOUND: usize = 20_000;

/// One season's forcing on the climate grid: per cell the wind (east, north, m/s), the temperature (K), the kept
/// sunlight (W/m²), the heat store (W/m²), the sea share, the mean height of the cell's land over the sea (m), and the
/// land's own evaporation (kg/m²/s, the previous pass's).
#[derive(Clone, Copy, Debug)]
pub struct Forcing<'a> {
    pub east: &'a [Gf],
    pub north: &'a [Gf],
    pub t: &'a [Gf],
    pub kept: &'a [Gf],
    pub g_store: &'a [Gf],
    pub sea: &'a [Gf],
    pub height_m: &'a [Gf],
    pub land_evap: &'a [Gf],
    /// Each cell's mean cells' share of the transport (the eddies carry the rest).
    pub cell_share: &'a [Gf],
    /// The body's global annual mean of the kept sunlight, W/m² (the surface's net longwave loss is its share of it).
    pub kept_mean: Gf,
    /// ★ THE BENCH'S SWITCHES (ruling B38: the laws under measurement, one at a time; the game will read one law):
    /// the rain-out law, and whether the eddy diffusion runs everywhere.
    pub rain_law: RainLaw,
    pub eddy_everywhere: bool,
    /// Each cell's rain-out rate for [`RainLaw::Residence`], 1/s: the mountains' own mean `⟨φ/τ(H)⟩` from the node grid
    /// ([`super::orography`]), or `1/τ_res` on the first round.
    pub rain_rate: &'a [Gf],
    /// The lapse rate, K/m.
    pub lapse_k_m: Gf,
}

/// The rain-out laws under measurement (the bench chooses; the design §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RainLaw {
    /// `P = W/τ`, `τ` the vapour's measured residence time at the sea's level, 8.9 days (van der Ent & Tuinenburg 2017).
    Residence,
    /// `P = exp(a·(r − b))` mm/day by the column's saturation (Bretherton, Peters & Back 2004).
    Saturation,
}

/// ★ THE VAPOUR'S MEASURED RESIDENCE TIME, s (van der Ent & Tuinenburg 2017; named): [`RainLaw::Residence`]'s.
pub const RESIDENCE_S: f64 = 8.9 * 86_400.0;

/// One season's water: per cell the column (kg/m²), the rain and the sea's evaporation (kg/m²/s), the land's potential
/// evaporation (kg/m²/s); and the sweeps it took.
#[derive(Clone, Debug)]
pub struct Water {
    pub column: Vec<Gf>,
    pub rain: Vec<Gf>,
    pub sea_evap: Vec<Gf>,
    pub pet: Vec<Gf>,
    /// The storms' own vapour convergence per cell (the eddy diffusion's), kg/m²/s: the heated winds' next round reads it.
    pub eddy_convergence: Vec<Gf>,
    pub sweeps: usize,
    pub met: bool,
}

/// ★ PRIESTLEY–TAYLOR's evaporation from a wet surface at `t_k` with the available energy `energy_w_m2`, kg/m²/s.
#[must_use]
pub fn priestley_taylor(words: &BalanceWords, t_k: Gf, energy_w_m2: Gf) -> Gf {
    let l = Gf::from_f64(LATENT_J_KG);
    let e = saturation_vapour_pa(t_k);
    let tc =
        t_k - Gf::from_f64(crate::climate::FREEZING_K) + Gf::from_f64(crate::climate::TETENS_C_C);
    let delta = e * Gf::from_f64(crate::climate::TETENS_B * crate::climate::TETENS_C_C) / (tc * tc);
    let gamma = words.cp_j_kg_k * words.p_surf_pa / (words.eps * l);
    Gf::from_f64(PT_ALPHA) * delta / (delta + gamma) * energy_w_m2 / l
}

/// ★ ONE SEASON'S WATER.
#[must_use]
pub fn season_water(grid: &ClimateGrid, words: &BalanceWords, f: Forcing<'_>) -> Water {
    let n = grid.len();
    let cols = grid.cols;
    let l = Gf::from_f64(LATENT_J_KG);
    let rv = Gf::from_f64(R_VAPOUR);
    let rain_a = Gf::from_f64(RAIN_SLOPE);
    let rain_b = Gf::from_f64(RAIN_OFFSET);
    let per_day = Gf::from_i64(86_400);
    let sw_share = Gf::from_f64(SURFACE_SW_SHARE);
    let lw_net = Gf::from_f64(SURFACE_LW_SHARE) * f.kept_mean;
    let freezing = Gf::from_f64(SEA_FREEZING_K);
    let pi = Gf::from_f64(std::f64::consts::PI);
    let step = pi / Gf::from_i64(grid.rows as i64);
    let d = words.diffusivity_m2_s;
    // The sources, the dew, the rain-out times and the saturations, per cell.
    let mut sea_evap = vec![Gf::ZERO; n];
    let mut pet = vec![Gf::ZERO; n];
    let mut source = vec![Gf::ZERO; n];
    let mut dew_rate = vec![Gf::ZERO; n];
    let mut w_sat = vec![Gf::ZERO; n];
    for i in 0..n {
        let t = f.t[i];
        let r_n = sw_share * f.kept[i] - lw_net;
        let t_h = t - f.lapse_k_m * f.height_m[i];
        let h0 = rv * t_h * t_h / (l * f.lapse_k_m);
        w_sat[i] = saturation_vapour_pa(t_h) * h0 / (rv * t_h);
        let e_sea = if t > freezing {
            priestley_taylor(words, t, r_n - f.g_store[i])
        } else {
            Gf::ZERO
        };
        if e_sea >= Gf::ZERO {
            sea_evap[i] = e_sea * f.sea[i];
        } else {
            // Dew: the sea takes back vapour at the rate the law states, scaled by the vapour there is.
            dew_rate[i] = -e_sea * f.sea[i] / w_sat[i];
        }
        pet[i] = priestley_taylor(words, t, r_n).greater(Gf::ZERO);
        source[i] = sea_evap[i] + f.land_evap[i] * (Gf::ONE - f.sea[i]);
    }
    // The rain of a column `w` over its saturation `ws`, kg/m²/s, and its slope in `w`.
    let rain_of = |i: usize, w: Gf, ws: Gf| match f.rain_law {
        RainLaw::Saturation => (rain_a * (w / ws - rain_b)).exp() / per_day,
        RainLaw::Residence => w * f.rain_rate[i],
    };
    let rain_slope = |i: usize, w: Gf, ws: Gf| match f.rain_law {
        RainLaw::Saturation => rain_of(i, w, ws) * rain_a / ws,
        RainLaw::Residence => f.rain_rate[i],
    };
    // Each cell's four faces: west, east, south, north — the neighbour, the coupling INTO i from it (its inflowing
    // wind's flux and the diffusion), and i's own outflow sum.
    let mut out_sum = vec![Gf::ZERO; n];
    let mut coef: Vec<[(usize, Gf); 4]> = vec![[(usize::MAX, Gf::ZERO); 4]; n];
    let mut dif_face: Vec<[(usize, Gf); 4]> = vec![[(usize::MAX, Gf::ZERO); 4]; n];
    for i in 0..n {
        let r = grid.row(i);
        let c = i % cols;
        let east_len = step * grid.radius_m;
        let west = r * cols + (c + cols - 1) % cols;
        let east = r * cols + (c + 1) % cols;
        let mut faces: [(usize, Gf, Gf, Gf); 4] = [
            (
                west,
                east_len,
                -(f.east[i] + f.east[west]) * Gf::HALF,
                grid.c_east[r],
            ),
            (
                east,
                east_len,
                (f.east[i] + f.east[east]) * Gf::HALF,
                grid.c_east[r],
            ),
            (usize::MAX, Gf::ZERO, Gf::ZERO, Gf::ZERO),
            (usize::MAX, Gf::ZERO, Gf::ZERO, Gf::ZERO),
        ];
        if r > 0 {
            let s_cell = (r - 1) * cols + c;
            faces[2] = (
                s_cell,
                grid.radius_m * step * grid.c_north[r - 1],
                -(f.north[i] + f.north[s_cell]) * Gf::HALF,
                grid.c_north[r - 1],
            );
        }
        if r + 1 < grid.rows {
            let n_cell = (r + 1) * cols + c;
            faces[3] = (
                n_cell,
                grid.radius_m * step * grid.c_north[r],
                (f.north[i] + f.north[n_cell]) * Gf::HALF,
                grid.c_north[r],
            );
        }
        for (k, &(m, len, out_wind, cond)) in faces.iter().enumerate() {
            if m == usize::MAX {
                continue;
            }
            let eddy = if f.eddy_everywhere {
                Gf::ONE
            } else {
                Gf::ONE - (f.cell_share[i] + f.cell_share[m]) * Gf::HALF
            };
            let dif = d * eddy * cond;
            let adv_out = if out_wind > Gf::ZERO {
                out_wind * len
            } else {
                Gf::ZERO
            };
            out_sum[i] += adv_out + dif;
            let adv_in = if out_wind < Gf::ZERO {
                -out_wind * len
            } else {
                Gf::ZERO
            };
            coef[i][k] = (m, adv_in + dif);
            dif_face[i][k] = (m, dif);
        }
    }
    // ROW RELAXATION with Newton's linearisation of the rain at the current column.
    // From a saturated column (above the solution): Newton's steps then descend onto it.
    let mut w: Vec<Gf> = w_sat.clone();
    let tol = Gf::from_f64(SWEEP_TOL);
    let mut sweeps = 0;
    let mut met = false;
    let (mut lo, mut di, mut up, mut rhs, mut x) = (
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
        vec![Gf::ZERO; cols],
    );
    while sweeps < SWEEP_BOUND {
        sweeps += 1;
        let mut moved = Gf::ZERO;
        for r in (0..grid.rows).chain((0..grid.rows).rev()) {
            let area = grid.area[r];
            for c in 0..cols {
                let i = r * cols + c;
                let mut b = area * source[i];
                for &(m, a) in &coef[i][2..4] {
                    if m != usize::MAX {
                        b += a * w[m];
                    }
                }
                let p0 = rain_of(i, w[i], w_sat[i]);
                let slope = rain_slope(i, w[i], w_sat[i]);
                di[c] = out_sum[i] + area * (slope + dew_rate[i]);
                lo[c] = coef[i][0].1;
                up[c] = coef[i][1].1;
                rhs[c] = b + area * (slope * w[i] - p0);
            }
            cyclic_three_band(&lo, &di, &up, &rhs, &mut x);
            for (c, &xc) in x.iter().enumerate() {
                let i = r * cols + c;
                let rel = (xc - w[i]).abs() / (xc.abs() + w_sat[i] * Gf::from_f64(1e-6));
                moved = moved.greater(rel);
                w[i] = xc;
            }
        }
        if moved < tol {
            met = true;
            break;
        }
    }
    let rain: Vec<Gf> = (0..n).map(|i| rain_of(i, w[i], w_sat[i])).collect();
    let eddy_convergence: Vec<Gf> = (0..n)
        .map(|i| {
            let mut s = Gf::ZERO;
            for &(m, dif) in &dif_face[i] {
                if m != usize::MAX {
                    s += dif * (w[m] - w[i]);
                }
            }
            s / grid.area[grid.row(i)]
        })
        .collect();
    Water {
        column: w,
        rain,
        sea_evap,
        pet,
        eddy_convergence,
        sweeps,
        met,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seasons::balance::periodic_year;
    use crate::seasons::balance::tests::aquaplanet;

    /// ★ THE WATER IS CONSERVED: on an aquaplanet's season with a steady easterly and a northward wind, the body's rain is
    /// its evaporation (the solve's own stop as the tolerance); every column is positive; Priestley–Taylor's evaporation
    /// is nought with no energy and grows with it.
    #[test]
    fn the_season_rains_what_the_sea_evaporates() {
        let (grid, bw, sun, surface, _) = aquaplanet(18, 940);
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
        let n = grid.len();
        let st = 0;
        let east = vec![Gf::from_f64(-4.0); n];
        let north = vec![Gf::from_f64(1.0); n];
        let zero = vec![Gf::ZERO; n];
        let one = vec![Gf::ONE; n];
        let rate = vec![Gf::ONE / Gf::from_f64(RESIDENCE_S); n];
        for (law, eddy) in [
            (RainLaw::Residence, true),
            (RainLaw::Residence, false),
            (RainLaw::Saturation, true),
        ] {
            let w = season_water(
                &grid,
                &bw,
                Forcing {
                    east: &east,
                    north: &north,
                    t: &p.year.t[st],
                    kept: &p.year.kept[st],
                    g_store: &zero,
                    sea: &one,
                    height_m: &zero,
                    land_evap: &zero,
                    cell_share: &zero,
                    kept_mean: Gf::from_f64(236.0),
                    rain_law: law,
                    eddy_everywhere: eddy,
                    rain_rate: &rate,
                    lapse_k_m: Gf::from_f64(0.0065),
                },
            );
            assert!(w.met, "{law:?}");
            let (mut rain, mut evap) = (0.0, 0.0);
            for i in 0..n {
                let a = grid.area[grid.row(i)].to_f64();
                rain += w.rain[i].to_f64() * a;
                evap += w.sea_evap[i].to_f64() * a;
                assert!(w.column[i] > Gf::ZERO);
            }
            assert!(
                (rain / evap - 1.0).abs() < 1e-3,
                "{law:?}: rain {rain} against the evaporation {evap}"
            );
        }
        assert_eq!(
            priestley_taylor(&bw, Gf::from_f64(288.0), Gf::ZERO),
            Gf::ZERO
        );
        assert!(priestley_taylor(&bw, Gf::from_f64(288.0), Gf::from_f64(100.0)) > Gf::ZERO);
    }
}
