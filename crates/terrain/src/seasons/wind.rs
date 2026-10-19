//! ★ L3 — THE RAIN BELT AND THE SURFACE WINDS of one step of the year (the design §3.3), on the climate grid.
//!
//! 1. **The rain belt** (the rising branch) is the ENERGY-FLUX EQUATOR of the step's zonal transport — the latitude where
//!    the heat the air carries turns from southward to northward (Bischoff & Schneider 2014; Donohoe et al. 2013 measured
//!    it on Earth's seasons) — the crossing nearest the zonal moist energy's maximum where there are several (Privé &
//!    Plumb 2007: the belt stands at that maximum's equatorward side).
//! 2. **The mean cells** span from the rising branch to their sinking edges. The edges stand where the eddies take over
//!    (Held 2000; Kang & Lu 2012): `φ_d ∝ (g·H·Δ_v/(Ω²·a²))^{1/4}`, so `φ_d = 30° × √(Ω_E·a_E/(Ω·a))` — the constant
//!    `g·H·Δ_v` calibrated on Earth's annual 30° edge (a constant the law cannot compute yet, named); an edge past the
//!    pole is a cell over the whole hemisphere (the slow spin's global cell), never a cap. Inside the cells the mass flux
//!    is `ψ = F/g_ms` with the gross moist stability `g_ms = h_T − h(φ)`, `h_T = 1.06 ×` the zonal moist energy's maximum
//!    over THE CONVECTING TROPICS (Siler, Roe & Armour 2018's 1.06, Earth; positive everywhere by construction): the free
//!    troposphere matches the convecting tropics' near-surface air (convective quasi-equilibrium, Emanuel, Neelin &
//!    Bretherton 1994) and the weak temperature gradient spreads it over the equatorial deformation radius
//!    `φ_eq = √(c/(2Ω·a))` (Matsuno 1966; Gill 1980), `c = N·D/π` the gravest baroclinic wave under the tropopause (Gill's
//!    eq. 2.1), `N² = (g/T)·(g/c_p − Γ)` from the body's own lapse rate and `D` the tropopause's height at 0.1 bar
//!    (Robinson & Catling 2014); a column whose own air outdoes that free troposphere convects through it, so its upper
//!    air is at least the ratio over its own `h` (the same law read locally; `g_ms ≥ 0.06·h`). MEASURED (2026-10-18): over the whole globe the maximum stood at the summer pole of the
//!    home planet's 55° tilt in most seasons (g_ms 40–61 kJ/kg at the equator against ExoPlaSim's 13–16). The lower branch carries `ψ` back
//!    toward the rising branch in the lowest tenth of the column's mass (Siler et al.: below 900 hPa, named), so
//!    `v = −ψ·g/(2π·a·cos φ·0.1·p_s)`.
//! 3. **The boundary layer's balance** of the Coriolis turn and the surface drag, `f·v·δ = C_D·|U|·u` (the bulk drag law,
//!    `C_D` = 1.3 × 10⁻³ over the sea, Earth, named; `δ` the lower tenth's depth, `0.1·p_s/(ρ·g)`): in the cells it gives
//!    `u` from `v` in closed form, `u² = (√(v⁴ + 4a²) − v²)/2`, `a = f·v·δ/C_D`; the trades, and a cross-equator
//!    monsoon's westerlies where the flow turns poleward.
//! 4. **The eddy belts** (poleward of the edges): the atmosphere's angular momentum is steady, so the surface's torque
//!    sums to nought over the globe — the eddy belts' westerly stress returns exactly the cells' easterly torque, spread
//!    by the eddies' own transport `|F|`, and only when the cells' torque IS easterly (a cells' torque of the other sign is
//!    the cells' own error, never storm-belt easterlies, as in [`super::gill`]; MEASURED 2026-10-18: −12 m/s polar
//!    easterlies on the home planet); `u` from the stress by the drag law, `v` from the same balance (poleward under
//!    westerlies: the Ferrel cell's lower branch).
//! 5. **The land–sea departure** (Lindzen & Nigam 1987): the boundary layer's pressure falls over a column warmer than its
//!    latitude's mean, `Φ' = −(g·δ/T̄)·T'`, and the wind answers the pressure, the Coriolis turn and the same drag, `r·u' −
//!    f·v' = −∂Φ'/∂x`, `f·u' + r·v' = −∂Φ'/∂y`, `r = C_D·|U|/δ` with the wind's own speed (a fixed point of a few rounds).
//!    This is the monsoon's pull toward a hot summer continent and its push away in winter.
//!
//! **Example.** In Earth's July the energy-flux equator stands near 10° N; the southern trades cross the equator, turn
//! east under the northern Coriolis force and blow onto India as the monsoon's westerlies, while the heat over the
//! Tibetan Plateau's continent pulls the low-level air inland.

use crate::gf::Gf;

use super::balance::{BalanceWords, moist_energy};
use super::grid::ClimateGrid;

/// ★ THE CELLS' EDGE on Earth, degrees (the annual Hadley edge; the eddy limit's calibration, named), and Earth's spin
/// (rad/s) and radius (m) it was measured at.
pub const EARTH_EDGE_DEG: f64 = 30.0;
pub const EARTH_SPIN_RAD_S: f64 = 7.292e-5;
pub const EARTH_RADIUS_M: f64 = 6.371e6;
/// ★ SILER, ROE & ARMOUR 2018's mean cells' share width in `sin φ` (their Earth fit), over Earth's cell half-width in
/// `sin φ` (`sin 30°`) it was fitted on: the share's width is that fraction of each cell's own half-width (named).
pub const CELL_SHARE_WIDTH: f64 = 0.3;
pub const EARTH_HALF_WIDTH: f64 = 0.5;
/// ★ SILER et al.'s top moist energy over the zonal maximum (their Earth fit; named).
pub const TOP_ENERGY_RATIO: f64 = 1.06;
/// ★ THE LOWER BRANCH'S SHARE of the column's mass (Siler et al.: below 900 hPa; named).
pub const LOWER_BRANCH_SHARE: f64 = 0.1;
/// ★ THE SURFACE'S DRAG COEFFICIENT over the sea (the bulk law's Earth value; named).
pub const DRAG_COEFFICIENT: f64 = 1.3e-3;
/// ★ THE TROPOPAUSE'S PRESSURE, Pa (Robinson & Catling 2014: the thick atmospheres of the Solar System hold their
/// tropopause near 0.1 bar, where the air turns thin enough to the infrared; the Solar System's bodies, named).
pub const TROPOPAUSE_PA: f64 = 10_000.0;
/// The rounds of the departure's fixed point between its speed and its drag.
pub const DEPARTURE_ROUNDS: usize = 6;

/// One step's winds: east and north components at each cell, m/s; and the rising branch's latitude's sine.
#[derive(Clone, Debug)]
pub struct Winds {
    pub east: Vec<Gf>,
    pub north: Vec<Gf>,
    pub rising_sin_lat: Gf,
    /// Each ROW's mean cells' share of the transport (nought in the eddy belts): the vapour's eddy diffusion carries the
    /// rest (Siler, Roe & Armour 2018's partition).
    pub cell_share: Vec<Gf>,
    /// The cells' closure by ROW (readings for the bench): the northward energy transport at the row's centre (W), the
    /// gross moist stability `h_T − h` (J/kg), the zonal moist energy `h` (J/kg) and the mean cells' mass flux `ψ` (kg/s).
    pub row_transport: Vec<Gf>,
    pub row_gms: Vec<Gf>,
    pub row_h: Vec<Gf>,
    pub row_psi: Vec<Gf>,
}

/// One step's fields the winds read: the temperature `t` (K), the kept sunlight `kept` (W/m²), the heat store
/// `g_store` (`C·∂T/∂t`, W/m²), each cell's humidity share `rh`, and the outgoing longwave's intercept `a_olr`.
#[derive(Clone, Copy, Debug)]
pub struct StepFields<'a> {
    pub t: &'a [Gf],
    pub kept: &'a [Gf],
    pub g_store: &'a [Gf],
    pub rh: &'a [Gf],
    pub a_olr: Gf,
    /// ★ THE BENCH'S SWITCH (ruling B38): whether the land–sea departures (Lindzen & Nigam 1987) run.
    pub departures: bool,
}

/// ★ THE STEP'S WINDS on a body spinning at `omega` rad/s.
#[must_use]
pub fn winds(
    grid: &ClimateGrid,
    words: &BalanceWords,
    omega: Gf,
    step_fields: StepFields<'_>,
) -> Winds {
    let StepFields {
        t,
        kept,
        g_store,
        rh,
        a_olr,
        departures,
    } = step_fields;
    let rows = grid.rows;
    let cols = grid.cols;
    let n = grid.len();
    let pi = Gf::from_f64(std::f64::consts::PI);
    // The zonal rows: net heating (W), moist energy (J/kg), temperature (K).
    let mut net = vec![Gf::ZERO; rows];
    let mut h = vec![Gf::ZERO; rows];
    let mut tz = vec![Gf::ZERO; rows];
    let inv_cols = Gf::ONE / Gf::from_i64(cols as i64);
    for i in 0..n {
        let r = grid.row(i);
        let olr = a_olr + words.b_w_m2_k * t[i];
        net[r] += (kept[i] - olr - g_store[i]) * grid.area[r];
        h[r] += moist_energy(words, t[i], rh[i]).0 * inv_cols;
        tz[r] += t[i] * inv_cols;
    }
    // The northward transport at each row's northern edge (W): the southern cap's net heating carried north.
    let mut f_edge = vec![Gf::ZERO; rows];
    let mut acc = Gf::ZERO;
    for r in 0..rows {
        acc += net[r];
        f_edge[r] = acc;
    }
    // The rising branch: the upward crossing of the transport nearest the moist energy's maximum.
    let mut hmax_row = 0;
    for r in 1..rows {
        if h[r] > h[hmax_row] {
            hmax_row = r;
        }
    }
    let step = pi / Gf::from_i64(rows as i64);
    let edge_sin = |r: usize| -> Gf {
        // The northern edge of row r: latitude −π/2 + (r + 1)·step; its sine by the grid's own rows (the mean of the two
        // rows' centre sines is within the grid's step of it; the exact sine is not needed for a belt's position).
        if r + 1 < rows {
            (grid.sin_lat[r] + grid.sin_lat[r + 1]) * Gf::HALF
        } else {
            Gf::ONE
        }
    };
    let mut rising = grid.sin_lat[hmax_row];
    let mut best = usize::MAX;
    for r in 0..rows - 1 {
        let below = if r == 0 { Gf::ZERO } else { f_edge[r - 1] };
        if below < Gf::ZERO && f_edge[r] >= Gf::ZERO {
            let d = r.abs_diff(hmax_row);
            if d < best {
                best = d;
                rising = edge_sin(r);
            }
        }
    }
    // The convecting tropics' band (the equatorial deformation radius) and the free troposphere's moist energy over it.
    let s_eq = tropics_band_sin(words, omega, grid.radius_m, tz[rows / 2]);
    let mut h_band = h[rows / 2];
    for (&s, &hr) in grid.sin_lat.iter().zip(&h) {
        if s.abs() <= s_eq && hr > h_band {
            h_band = hr;
        }
    }
    let h_top = Gf::from_f64(TOP_ENERGY_RATIO) * h_band;
    // The cells' edges in sin φ: Held's eddy limit scaled from Earth's; the region holds the rising branch.
    let edge_rad = Gf::from_f64(EARTH_EDGE_DEG) * Gf::from_f64(std::f64::consts::PI)
        / Gf::from_i64(180)
        * (Gf::from_f64(EARTH_SPIN_RAD_S * EARTH_RADIUS_M) / (omega * grid.radius_m)).sqrt();
    let s_d = if edge_rad >= Gf::from_f64(std::f64::consts::FRAC_PI_2) {
        Gf::ONE
    } else {
        super::grid::cos_series(Gf::from_f64(std::f64::consts::FRAC_PI_2) - edge_rad)
    };
    let (lo, hi) = (rising.lesser(-s_d), rising.greater(s_d));
    // The mean cells' share of the transport inside each cell (Siler et al.'s Gaussian about the rising branch, its width
    // the cell's own: Earth's 0.3 over Earth's half-width 0.5 in sin φ), less its own value at the edge and renormalised,
    // so it falls to nought AT the edge: the cells' wind dies away into the calm of the sinking air (the horse latitudes)
    // and no wind jumps at an edge.
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
    // The eddy belts' own weight: nought at the edge, growing over the cell's own Gaussian width beyond it, so the
    // storms' westerlies rise out of the same calm.
    let eddy_weight = |x: Gf| {
        let beyond = if x > hi {
            x - hi
        } else if x < lo {
            lo - x
        } else {
            return Gf::ZERO;
        };
        let half = if x > hi { hi - rising } else { rising - lo };
        let sigma =
            Gf::from_f64(CELL_SHARE_WIDTH / EARTH_HALF_WIDTH) * half.greater(Gf::from_f64(1e-3));
        let u = beyond / sigma;
        Gf::ONE - (-(u * u)).exp()
    };
    // Each row's zonal-mean meridional wind at its centre: the mean cells' lower branch.
    let two_pi_a = Gf::TWO * pi * grid.radius_m;
    let low_mass = Gf::from_f64(LOWER_BRANCH_SHARE) * words.column_kg_m2;
    // The surface air's density at the equator's row (the gas law), and the lower tenth's depth.
    let rho = words.p_surf_pa / rd_t(words, tz[rows / 2]);
    let delta = low_mass / rho;
    let cd = Gf::from_f64(DRAG_COEFFICIENT);
    let mut v_row = vec![Gf::ZERO; rows];
    let mut u_row = vec![Gf::ZERO; rows];
    let mut eddy_shape = vec![Gf::ZERO; rows];
    let mut cell_share = vec![Gf::ZERO; rows];
    let (mut row_transport, mut row_gms, mut row_psi) = (
        vec![Gf::ZERO; rows],
        vec![Gf::ZERO; rows],
        vec![Gf::ZERO; rows],
    );
    let mut torque = Gf::ZERO;
    for r in 0..rows {
        let x = grid.sin_lat[r];
        // The transport at the row's centre: the mean of its two edges.
        let f_mid = ((if r == 0 { Gf::ZERO } else { f_edge[r - 1] }) + f_edge[r]) * Gf::HALF;
        let w = share(x);
        cell_share[r] = w;
        // A column whose own air outdoes the tropics' free troposphere convects through it (the same quasi-equilibrium,
        // read locally): its upper air is at least the ratio over its own moist energy, so g_ms stays positive.
        let g_ms = h_top.greater(Gf::from_f64(TOP_ENERGY_RATIO) * h[r]) - h[r];
        let psi = w * f_mid / g_ms;
        (row_transport[r], row_gms[r], row_psi[r]) = (f_mid, g_ms, psi);
        let cosl = grid.cos_lat[r];
        let v = -psi / (two_pi_a * cosl * low_mass);
        let f = Gf::TWO * omega * x;
        let a = f * v * delta / cd;
        let u_mag = (((v * v * v * v + Gf::from_i64(4) * a * a).sqrt() - v * v) * Gf::HALF).sqrt();
        let u = if a < Gf::ZERO { -u_mag } else { u_mag };
        v_row[r] = v;
        u_row[r] = u;
        // The cells' surface stress and its torque about the axis over the row's ring.
        let speed = (u * u + v * v).sqrt();
        let stress = rho * cd * speed * u;
        let ring = grid.area[r] * Gf::from_i64(cols as i64);
        torque += stress * grid.radius_m * cosl * ring;
        // The eddy belts only: beyond the cells' edges, rising out of the edge's calm.
        eddy_shape[r] = eddy_weight(x) * f_mid.abs();
    }
    // The eddy belts return the torque: their westerly stress spread by the eddies' share.
    let mut shape_torque = Gf::ZERO;
    for ((shape, area), cosl) in eddy_shape.iter().zip(&grid.area).zip(&grid.cos_lat) {
        shape_torque += *shape * grid.radius_m * *cosl * *area * Gf::from_i64(cols as i64);
    }
    if torque < Gf::ZERO && shape_torque > Gf::ZERO {
        for r in 0..rows {
            let stress = -torque * eddy_shape[r] / shape_torque;
            let u = {
                let m = (stress.abs() / (rho * cd)).sqrt();
                if stress < Gf::ZERO { -m } else { m }
            };
            let f = Gf::TWO * omega * grid.sin_lat[r];
            // The same balance gives the eddy belt's meridional wind: v = C_D·|U|·u/(f·δ).
            let v = if f.abs() > Gf::ZERO {
                cd * u.abs() * u / (f * delta)
            } else {
                Gf::ZERO
            };
            u_row[r] += u;
            v_row[r] += v;
        }
    }
    // The land–sea departures (Lindzen & Nigam 1987) on the zonal winds.
    let mut east = vec![Gf::ZERO; n];
    let mut north = vec![Gf::ZERO; n];
    let phi: Vec<Gf> = (0..n)
        .map(|i| {
            let r = grid.row(i);
            -(gravity_from(words) * delta / tz[r]) * (t[i] - tz[r])
        })
        .collect();
    let mut speed: Vec<Gf> = (0..n)
        .map(|i| {
            let r = grid.row(i);
            (u_row[r] * u_row[r] + v_row[r] * v_row[r]).sqrt()
        })
        .collect();
    let rounds = if departures { DEPARTURE_ROUNDS } else { 1 };
    for _ in 0..rounds {
        for i in 0..n {
            let r = grid.row(i);
            let c = i % cols;
            let west = r * cols + (c + cols - 1) % cols;
            let eastc = r * cols + (c + 1) % cols;
            let dx = Gf::TWO * step * grid.radius_m * grid.cos_lat[r];
            let x_force = -(phi[eastc] - phi[west]) / dx;
            let y_force = if r > 0 && r + 1 < rows {
                -(phi[(r + 1) * cols + c] - phi[(r - 1) * cols + c])
                    / (Gf::TWO * step * grid.radius_m)
            } else {
                Gf::ZERO
            };
            let f = Gf::TWO * omega * grid.sin_lat[r];
            let drag = cd * speed[i] / delta;
            let den = drag * drag + f * f;
            let (du, dv) = if departures && den > Gf::ZERO {
                (
                    (drag * x_force + f * y_force) / den,
                    (drag * y_force - f * x_force) / den,
                )
            } else {
                (Gf::ZERO, Gf::ZERO)
            };
            east[i] = u_row[r] + du;
            north[i] = v_row[r] + dv;
        }
        for i in 0..n {
            speed[i] = (east[i] * east[i] + north[i] * north[i]).sqrt();
        }
    }
    Winds {
        east,
        north,
        rising_sin_lat: rising,
        cell_share,
        row_transport,
        row_gms,
        row_h: h,
        row_psi,
    }
}

/// ★ THE CONVECTING TROPICS' BAND, the sine of its edge's latitude: the equatorial deformation radius
/// `φ_eq = √(c/(2Ω·a))`, `c = N·D/π` (Gill 1980, eq. 2.1), `N² = (g/T)·(g/c_p − Γ)` at the equator's temperature `t_k` with
/// the body's own lapse rate `Γ` ([`crate::climate::LAPSE_MOIST_SHARE`] of the dry adiabat), and `D` the height where the
/// pressure falls to [`TROPOPAUSE_PA`] along that lapse, `D = (T/Γ)·(1 − (p_tp/p_s)^{R·Γ/g})`. Air no thicker than the
/// tropopause's pressure has no troposphere to spread a temperature: the band is the equator's row alone (a sine of
/// nought); a band past the pole is the whole globe.
fn tropics_band_sin(words: &BalanceWords, omega: Gf, radius_m: Gf, t_k: Gf) -> Gf {
    let p_tp = Gf::from_f64(TROPOPAUSE_PA);
    if words.p_surf_pa <= p_tp || omega <= Gf::ZERO {
        return Gf::ZERO;
    }
    let g = gravity_from(words);
    let dry = g / words.cp_j_kg_k;
    let lapse = Gf::from_f64(crate::climate::LAPSE_MOIST_SHARE) * dry;
    let r_gas = words.cp_j_kg_k / Gf::from_f64(3.5);
    let depth = t_k / lapse * (Gf::ONE - ((p_tp / words.p_surf_pa).ln() * r_gas * lapse / g).exp());
    let n = (g / t_k * (dry - lapse)).sqrt();
    let c = n * depth / Gf::from_f64(std::f64::consts::PI);
    let phi = (c / (Gf::TWO * omega * radius_m)).sqrt();
    if phi >= Gf::from_f64(std::f64::consts::FRAC_PI_2) {
        Gf::ONE
    } else {
        super::grid::cos_series(Gf::from_f64(std::f64::consts::FRAC_PI_2) - phi)
    }
}

/// The surface gravity, m/s², from the balance's column and pressure (`g = p_s/(p_s/g)`).
fn gravity_from(words: &BalanceWords) -> Gf {
    words.p_surf_pa / words.column_kg_m2
}

/// The gas's `R·T/μ` at temperature `t`, J/kg: `c_p·T/3.5` for the diatomic air the balance reads.
fn rd_t(words: &BalanceWords, t: Gf) -> Gf {
    words.cp_j_kg_k * t / Gf::from_f64(3.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seasons::balance::tests::aquaplanet;
    use crate::seasons::balance::{mean_sun, periodic_year};

    /// ★ THE WINDS OF AN AQUAPLANET WITH NO TILT, at its equinox: the rain belt at the equator, the trades from the east
    /// at 15° and blowing toward the equator, the westerlies at 50°, and the calm at the cells' 30° edge.
    #[test]
    fn the_aquaplanets_winds_stand_where_earths_do() {
        let (grid, bw, sun, surface, words) = aquaplanet(36, 1_024);
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
        let steps = p.year.t.len();
        let st = steps / 4;
        let prev = st - 1;
        let dt = bw.year_s / Gf::from_i64(steps as i64);
        let g: Vec<Gf> = (0..grid.len())
            .map(|i| surface.capacity[i] * (p.year.t[st][i] - p.year.t[prev][i]) / dt)
            .collect();
        let a_olr = mean_sun(&grid, &sun) * bw.kept - bw.b_w_m2_k * bw.t_mean_k;
        let omega = Gf::TWO * Gf::from_f64(std::f64::consts::PI)
            / Gf::from_i64(i64::from(words.day_s.expect("a day")));
        let w = winds(
            &grid,
            &bw,
            omega,
            StepFields {
                t: &p.year.t[st],
                kept: &p.year.kept[st],
                g_store: &g,
                rh: &surface.rh,
                a_olr,
                departures: false,
            },
        );
        assert!(
            w.rising_sin_lat.abs().to_f64() < 0.1,
            "{}",
            w.rising_sin_lat.to_f64()
        );
        let at = |lat: f64| {
            let r = ((lat + 90.0) / 180.0 * grid.rows as f64) as usize;
            let i = r * grid.cols;
            (w.east[i].to_f64(), w.north[i].to_f64())
        };
        let (u15, v15) = at(15.0);
        assert!(u15 < -1.0, "the trades at 15° N: {u15}");
        assert!(v15 < 0.0, "toward the equator: {v15}");
        let (u50, _) = at(50.0);
        assert!(u50 > 1.0, "the westerlies at 50° N: {u50}");
        let (u_s15, v_s15) = at(-15.0);
        assert!(u_s15 < -1.0);
        assert!(v_s15 > 0.0);
        assert_eq!(w.cell_share.len(), grid.rows);
    }

    /// ★ THE CONVECTING TROPICS' BAND by the body's own air and spin: Earth's air at Earth's spin gives a tropopause near
    /// 16 km, a gravest wave near 53 m/s and a band near 14°; a third of the spin widens it by √3; air thinner than the
    /// tropopause's pressure keeps the equator's row alone; a body spinning slowly enough has one band over the globe.
    #[test]
    fn the_tropics_band_follows_the_air_and_the_spin() {
        let (grid, bw, _, _, _) = aquaplanet(36, 1_024);
        let earth = Gf::from_f64(7.292e-5);
        let t = Gf::from_f64(300.0);
        // Earth's band between 12° and 16° (their sines 0.2079 and 0.2756).
        let e = tropics_band_sin(&bw, earth, grid.radius_m, t).to_f64();
        assert!((0.2079..0.2756).contains(&e), "Earth's band's sine {e}");
        // A third of the spin: the angle √3 times as wide (the sines' ratio within 0.1 of it at these small angles).
        let slow = tropics_band_sin(&bw, earth / Gf::from_i64(3), grid.radius_m, t).to_f64();
        assert!(
            (slow / e - 3f64.sqrt()).abs() < 0.1,
            "a third of the spin: {slow} against {e}"
        );
        let thin = BalanceWords {
            p_surf_pa: Gf::from_f64(TROPOPAUSE_PA),
            ..bw
        };
        assert_eq!(tropics_band_sin(&thin, earth, grid.radius_m, t), Gf::ZERO);
        assert_eq!(tropics_band_sin(&bw, Gf::ZERO, grid.radius_m, t), Gf::ZERO);
        assert_eq!(
            tropics_band_sin(&bw, earth / Gf::from_i64(10_000), grid.radius_m, t),
            Gf::ONE
        );
    }
}
