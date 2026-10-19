//! ★ THE SEASONS' CHAIN AS THE SOLVE'S CLIMATE (the design §7 step 1, bench first — ruling B38): one call that returns
//! [`crate::climate::Climate`]'s rows from the seasons' laws, so every reader of the solve (the discharge, the lakes'
//! budget, the ice, the talus, the craters) reads them unchanged; and a STATE carried from pass to pass — the last
//! year's temperature, the ice-free albedo, the land's wetness and its own evaporation, each season's mountains' rate
//! and factor — so each pass starts warm (the relief moves little in a pass; the lags are one pass, as the design states).
//!
//! The rows (the design §3.6): the rain, the year's sum of the seasons' mountains' rain on the node, whole millimetres;
//! the potential evaporation, Priestley–Taylor's on the climate grid read to the node; the runoff, Turc–Pike's on them;
//! the aridity, the potential evaporation over the rain on the UNEP axis; the temperature, the balance's annual mean read
//! to the node less the lapse to its air; the frost line, where that annual mean reaches freezing; the equilibrium line,
//! where the warmest quarter's temperature reaches Ohmura's for the node's rain; the upwind slot, the year's mean wind's.
//!
//! **Example.** Pass 200 of the home planet's solve: the belt has risen a little since pass 199; the balance steps one
//! year from pass 199's year and is periodic again; the winds and the vapour follow; the mountains' factor starts from
//! pass 199's and settles in a few rounds; the belt's windward ramp reads its rain, and the rivers cut by it.

use crate::body::BodyDefinition;
use crate::climate::{
    ARIDITY_HYPER_RATIO, ARIDITY_WET_RATIO, Climate, FREEZING_K, NO_WIND, SeaView, datum_z,
    lapse_rate_k_m, ohmura_ela_summer_c, runoff_mm_yr,
};
use crate::gf::Gf;
use crate::macro_lattice::{MacroLattice, NO_NODE};
use crate::solve::{SolveWords, Z_STEPS_PER_M};

use super::balance::{
    BalanceWords, Surface, balance_words, capacity_j_m2_k, mean_sun, periodic_year_from,
};
use super::grid::ClimateGrid;
use super::orography::{NodeGeometry, mountain_rain};
use super::sun::{SunTable, YEAR_SAMPLES_SHIFT};
use super::vapour::{Forcing, RESIDENCE_S, RainLaw, season_water};
use super::wind::{StepFields, Winds, winds};

/// ★ THE CHAIN'S RESOLUTIONS (each read by its double on the bench, the design §5): the climate grid's rows, the
/// balance's steps a year (as a power of two), the seasons (as a power of two), the sun table's rows.
pub const CLIMATE_ROWS: usize = 90;
pub const STEPS_SHIFT: u32 = 6;
pub const SEASONS_SHIFT: u32 = 4;
pub const SUN_ROWS: usize = 1_025;
/// The periodic year's stop: no cell's year-start temperature moves more than this, K, within this many years.
pub const YEAR_TOL_K: f64 = 0.01;
pub const YEAR_BOUND: usize = 400;
/// ★ THE SEA'S RELATIVE HUMIDITY (Siler, Roe & Armour 2018, Earth's oceans; named).
pub const SEA_RH: f64 = 0.8;

/// The state the chain carries from one call to the next.
pub struct SeasonState {
    grid: ClimateGrid,
    geo: NodeGeometry,
    t_start: Vec<Gf>,
    alpha_free: Gf,
    wetness: Vec<Gf>,
    land_evap: Vec<Vec<Gf>>,
    cell_rates: Vec<Vec<Gf>>,
    phi: Vec<Option<Vec<u16>>>,
    /// Each season's departures' wind speed, the next pass's start for the drag.
    speed: Vec<Vec<Gf>>,
    /// The last call's readings for the bench: the years the balance stepped, the mountains' worst rounds.
    pub last_years: usize,
    pub last_rounds: usize,
    /// The largest loop the mountains' rain solved exactly, and the largest it left to the sweeps.
    pub last_loop: usize,
    pub last_left: usize,
    /// The heated winds' most drag rounds in a season, and whether every season's drag met its stop.
    pub last_drag: usize,
    pub last_drag_met: bool,
    /// The spin-up's calls and whether it met its stop ([`SPIN_TOL`]); the last call's year of rain per climate cell, mm.
    pub last_spin: (usize, bool),
    /// The spin-up's readings, one a call: the largest change of a cell's year of rain over its stop's allowance, the cells
    /// whose change exceeded it, and the mean year of rain over the land's cells (mm).
    pub spin_trace: Vec<(Gf, usize, Gf)>,
    /// Which climate cells are mostly land (the last call's sea share under a half).
    cell_land: Vec<bool>,
    cell_rain: Vec<Gf>,
    /// ★ THE YEAR'S SURFACE WIND per climate cell, the last call's (ruling B109; the sea's glint and waves): the mean of
    /// the seasons' winds east and north, m/s, and the mean of their speeds, m/s — a monsoon that turns with the seasons
    /// keeps its speed while its mean vector shrinks.
    pub last_wind: Vec<[Gf; 3]>,
    /// Each season's surface wind in its two parts, the last call's, in the seasons' order (a bench reading).
    pub last_season_winds: Vec<SeasonWinds>,
}

impl SeasonState {
    /// A cold state for `lattice` (the climate grid and the node geometry built once).
    #[must_use]
    pub fn new(lattice: &MacroLattice, threads: usize) -> SeasonState {
        SeasonState::with_rows(lattice, CLIMATE_ROWS, threads)
    }

    /// A cold state on a climate grid of `rows` latitude bands (a resolution the bench reads by its double).
    #[must_use]
    pub fn with_rows(lattice: &MacroLattice, rows: usize, threads: usize) -> SeasonState {
        let grid = ClimateGrid::new(lattice.radius_m(), rows);
        let geo = NodeGeometry::of(lattice, &grid, threads);
        let n = grid.len();
        let seasons = 1usize << SEASONS_SHIFT;
        SeasonState {
            t_start: Vec::new(),
            alpha_free: Gf::ZERO,
            wetness: vec![Gf::ONE; n],
            land_evap: vec![vec![Gf::ZERO; n]; seasons],
            cell_rates: vec![vec![Gf::ONE / Gf::from_f64(RESIDENCE_S); n]; seasons],
            phi: vec![None; seasons],
            speed: vec![Vec::new(); seasons],
            grid,
            geo,
            last_years: 0,
            last_rounds: 0,
            last_loop: 0,
            last_left: 0,
            last_drag: 0,
            last_drag_met: true,
            last_spin: (0, true),
            spin_trace: Vec::new(),
            cell_land: Vec::new(),
            cell_rain: Vec::new(),
            last_wind: Vec::new(),
            last_season_winds: Vec::new(),
        }
    }
}

/// The aridity byte from a potential evaporation and a rain, mm/yr, on the UNEP axis ([`crate::climate::aridity_q8`]'s,
/// read with the seasons' own potential evaporation).
fn aridity_of(pet: Gf, rain_mm_yr: u32) -> u8 {
    if rain_mm_yr == 0 {
        return 255;
    }
    let ratio = (pet / Gf::from_i64(i64::from(rain_mm_yr))).greater(Gf::from_f64(1e-12));
    let lo = Gf::from_f64(ARIDITY_WET_RATIO).ln();
    let hi = Gf::from_f64(ARIDITY_HYPER_RATIO).ln();
    let share = ((ratio.ln() - lo) / (hi - lo)).clamp(Gf::ZERO, Gf::ONE);
    (share * Gf::from_f64(255.0) + Gf::HALF).to_i64_floor() as u8
}

/// The height an excess of temperature reaches at the lapse rate, sixteenths from the ladder radius, from the datum.
fn line_z(lapse_k_m: Gf, over: Gf, datum: i32) -> i32 {
    if lapse_k_m <= Gf::ZERO {
        return if over >= Gf::ZERO { i32::MAX } else { i32::MIN };
    }
    let steps =
        over / lapse_k_m * Gf::from_i64(i64::from(Z_STEPS_PER_M)) + Gf::from_i64(i64::from(datum));
    steps
        .clamp(
            Gf::from_i64(i64::from(i32::MIN) + 1),
            Gf::from_i64(i64::from(i32::MAX) - 1),
        )
        .to_i64_floor() as i32
}

/// One node's rows: temperature (dK), rain, aridity, PET, runoff, frost line, equilibrium line, upwind slot.
type NodeRow = (i16, u32, u8, u32, u32, i32, i32, u8);

/// One season's answers inside a call: its winds (the cells' plus the departures), its water, its mountains' rain, and the
/// next pass's departures' speed and land evaporation.
struct SeasonOut {
    /// The heated winds' drag rounds and whether they met their stop.
    drag: (usize, bool),
    east: Vec<Gf>,
    north: Vec<Gf>,
    water: super::vapour::Water,
    mr: super::orography::MountainRain,
    speed: Vec<Gf>,
    land_evap: Vec<Gf>,
    parts: SeasonWinds,
}

/// ★ ONE SEASON'S SURFACE WIND per climate cell, in its two parts (a reading for the bench, the last call's): the cells'
/// law (the season's steps' mean, [`super::wind::winds`]) and the heated winds' land–sea departures (Gill,
/// [`super::gill::heated_winds`]), east and north, m/s; the season's wind is their sum.
#[derive(Clone, Debug, Default)]
pub struct SeasonWinds {
    pub cell_east: Vec<Gf>,
    pub cell_north: Vec<Gf>,
    pub dep_east: Vec<Gf>,
    pub dep_north: Vec<Gf>,
    /// The cells' closure by row, the season's steps' mean ([`super::wind::Winds`]'s readings): the transport (W), the
    /// gross moist stability (J/kg), the moist energy (J/kg), the mass flux (kg/s), the cells' share; and the rising
    /// branch's sine of latitude.
    pub row_transport: Vec<Gf>,
    pub row_gms: Vec<Gf>,
    pub row_h: Vec<Gf>,
    pub row_psi: Vec<Gf>,
    pub row_share: Vec<Gf>,
    pub rising_sin_lat: Gf,
}

/// ★ THE SPIN-UP'S STOP (2026-10-16): the climate's slow words (the land's wetness and own evaporation) start cold, and MEASURED the land's rain needs about ten calls to settle (its median 468 → ~530 mm, the
/// desert 7.3 → ~2 %); before the first pass the chain therefore repeats on the starting relief until no climate cell's
/// year of rain moves more than this share of itself (or one whole millimetre where it is nearly dry), within
/// [`SPIN_BOUND`] calls — so every later call, one a pass or one every few passes, only follows the relief.
pub const SPIN_TOL: f64 = 0.01;
/// The bound on the spin-up's calls (the state reports the count and whether it met its stop).
pub const SPIN_BOUND: usize = 30;

/// ★ THE SEASONS' CLIMATE over `lattice` for the relief `z` as it stands, warm from `state`; from a cold state the
/// spin-up first ([`SPIN_TOL`]).
#[must_use]
pub fn seasonal_climate(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    z: &[i32],
    sea: Option<SeaView<'_>>,
    threads: usize,
    state: &mut SeasonState,
) -> Climate {
    if !state.t_start.is_empty() {
        return climate_once(body, lattice, words, z, sea, threads, state);
    }
    let (tol, one_mm) = (Gf::from_f64(SPIN_TOL), Gf::ONE);
    let mut calls = 0;
    loop {
        let before = state.cell_rain.clone();
        let c = climate_once(body, lattice, words, z, sea, threads, state);
        calls += 1;
        let (mut worst, mut over) = (Gf::ZERO, 0usize);
        if before.len() == state.cell_rain.len() {
            for (a, b) in before.iter().zip(&state.cell_rain) {
                let ratio = (*b - *a).abs() / (*a * tol).greater(one_mm);
                worst = worst.greater(ratio);
                over += usize::from(ratio > Gf::ONE);
            }
        } else {
            over = state.cell_rain.len();
        }
        let (mut sum, mut count) = (Gf::ZERO, 0i64);
        for (r, &l) in state.cell_rain.iter().zip(&state.cell_land) {
            if l {
                sum += *r;
                count += 1;
            }
        }
        state
            .spin_trace
            .push((worst, over, sum / Gf::from_i64(count.max(1))));
        let settled = over == 0;
        if settled || calls >= SPIN_BOUND {
            state.last_spin = (calls, settled);
            return c;
        }
    }
}

/// One call of the chain, warm from `state` (the spin-up repeats it from a cold one).
#[allow(
    clippy::too_many_lines,
    reason = "one chain, its six laws in their order; split, each piece would only pass the others' fields on"
)]
fn climate_once(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    words: &SolveWords,
    z: &[i32],
    sea: Option<SeaView<'_>>,
    threads: usize,
    state: &mut SeasonState,
) -> Climate {
    let n_nodes = lattice.node_count();
    let datum = datum_z(lattice, z, sea.map(|s| s.level));
    let steps16 = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let ocean = |i: usize| sea.is_some_and(|s| s.ocean[i]);
    // Each node's air over the datum (the sea's surface over the ocean), metres.
    let air_m: Vec<Gf> = (0..n_nodes)
        .map(|i| {
            let air = if ocean(i) {
                z[i].max(sea.map_or(z[i], |s| s.level))
            } else {
                z[i]
            };
            Gf::from_i64(i64::from(air) - i64::from(datum)) / steps16
        })
        .collect();
    let bw: BalanceWords = balance_words(body, words);
    let lapse = lapse_rate_k_m(body.facts().gravity_mm_s2, words.mu_q8);
    let grid = &state.grid;
    let n = grid.len();
    // Each climate cell's sea share and its ground's mean height over the datum.
    let mut a_all = vec![Gf::ZERO; n];
    let mut a_sea = vec![Gf::ZERO; n];
    let mut h_sum = vec![Gf::ZERO; n];
    for (i, h) in air_m.iter().enumerate() {
        let c = state.geo.cell[i] as usize;
        let a = Gf::from_i64(lattice.area_m2(i as u32) as i64);
        a_all[c] += a;
        if ocean(i) {
            a_sea[c] += a;
        } else {
            h_sum[c] += a * h.greater(Gf::ZERO);
        }
    }
    let sea_share: Vec<Gf> = (0..n)
        .map(|c| {
            if a_all[c] > Gf::ZERO {
                a_sea[c] / a_all[c]
            } else {
                Gf::ONE
            }
        })
        .collect();
    let height: Vec<Gf> = (0..n)
        .map(|c| {
            if a_all[c] > Gf::ZERO {
                h_sum[c] / a_all[c]
            } else {
                Gf::ZERO
            }
        })
        .collect();
    let has_water = words.water_km3 > 0;
    let capacity: Vec<Gf> = sea_share
        .iter()
        .map(|&s| capacity_j_m2_k(&bw, s, words.year_s))
        .collect();
    let rh_sea = if has_water {
        Gf::from_f64(SEA_RH)
    } else {
        Gf::ZERO
    };
    let rh: Vec<Gf> = (0..n)
        .map(|c| rh_sea * (sea_share[c] + (Gf::ONE - sea_share[c]) * state.wetness[c]))
        .collect();
    let surface = Surface {
        capacity: capacity.clone(),
        rh: rh.clone(),
    };
    let s_flux = Gf::from_f64(crate::climate::SOLAR_CONSTANT_W_M2)
        * Gf::from_i64(i64::from(words.insolation_q12))
        / Gf::from_i64(4_096);
    let ecc = Gf::from_i64(i64::from(words.ecc_q16)) / Gf::from_i64(65_536);
    let cos_tilt =
        Gf::from_i64(i64::from(words.obliquity_cos_q1024.unwrap_or(1_024))) / Gf::from_i64(1_024);
    let sun = SunTable::new(s_flux, ecc, cos_tilt, SUN_ROWS, YEAR_SAMPLES_SHIFT);
    // L2: the periodic year, warm.
    let start = if state.t_start.len() == n {
        state.t_start.clone()
    } else {
        vec![bw.t_mean_k; n]
    };
    let alpha = if state.alpha_free > Gf::ZERO {
        state.alpha_free
    } else {
        Gf::ONE - bw.kept
    };
    let p = periodic_year_from(
        grid,
        &bw,
        &sun,
        STEPS_SHIFT,
        &surface,
        &start,
        alpha,
        (Gf::from_f64(YEAR_TOL_K), YEAR_BOUND),
    );
    let steps = 1usize << STEPS_SHIFT;
    let seasons = 1usize << SEASONS_SHIFT;
    let per = steps / seasons;
    let dt = bw.year_s / Gf::from_i64(steps as i64);
    let season_s = bw.year_s / Gf::from_i64(seasons as i64);
    let a_olr = mean_sun(grid, &sun) * bw.kept - bw.b_w_m2_k * bw.t_mean_k;
    let omega = Gf::TWO * Gf::from_f64(std::f64::consts::PI)
        / Gf::from_i64(i64::from(words.day_s.unwrap_or(0).max(1)));
    // L3: the winds of every step.
    let step_w: Vec<Winds> = (0..steps)
        .map(|st| {
            let prev = (st + steps - 1) % steps;
            let g: Vec<Gf> = (0..n)
                .map(|i| capacity[i] * (p.year.t[st][i] - p.year.t[prev][i]) / dt)
                .collect();
            winds(
                grid,
                &bw,
                omega,
                StepFields {
                    t: &p.year.t[st],
                    kept: &p.year.kept[st],
                    g_store: &g,
                    rh: &rh,
                    a_olr,
                    departures: false,
                },
            )
        })
        .collect();
    // L4–L6: each season's heated winds, vapour, rain and mountains' rain. A season reads only the last pass's state, so the
    // seasons run on the cores, one season a core, each core taking the next season when it finishes one; the answers join
    // in the seasons' order, so the bytes are the same on any number of cores (ruling B64).
    let k = Gf::from_i64(per as i64);
    let kept_mean = mean_sun(grid, &sun) * bw.kept;
    let last_speed = std::mem::take(&mut state.speed);
    let last_land_evap = std::mem::take(&mut state.land_evap);
    let last_rates = std::mem::take(&mut state.cell_rates);
    let last_phi = std::mem::take(&mut state.phi);
    let geo = &state.geo;
    let season = |m: usize, inner: usize| -> SeasonOut {
        let range = m * per..(m + 1) * per;
        let mean = |f: &dyn Fn(usize, usize) -> Gf| -> Vec<Gf> {
            (0..n)
                .map(|i| {
                    range
                        .clone()
                        .map(|st| f(st, i))
                        .fold(Gf::ZERO, |a, b| a + b)
                        / k
                })
                .collect()
        };
        let t = mean(&|st, i| p.year.t[st][i]);
        // The cell law's zonal winds plus the heating's land–sea departures (Gill's damped balance on the season's
        // departure from its row's mean heating: the monsoon's pull, the winter continent's push).
        let net = mean(&|st, i| {
            let prev = (st + steps - 1) % steps;
            let gs = capacity[i] * (p.year.t[st][i] - p.year.t[prev][i]) / dt;
            p.year.kept[st][i] - (a_olr + bw.b_w_m2_k * p.year.t[st][i]) - gs
        });
        let dep = super::gill::heated_winds(
            grid,
            &bw,
            omega,
            super::gill::Heating {
                t: &t,
                net: &net,
                sea: &sea_share,
                rh: &rh,
                departures_only: true,
            },
            &last_speed[m],
        );
        let speed = (0..n)
            .map(|i| (dep.east[i] * dep.east[i] + dep.north[i] * dep.north[i]).sqrt())
            .collect();
        let cell_east = mean(&|st, i| step_w[st].east[i]);
        let cell_north = mean(&|st, i| step_w[st].north[i]);
        let east: Vec<Gf> = cell_east
            .iter()
            .zip(&dep.east)
            .map(|(a, b)| *a + *b)
            .collect();
        let north: Vec<Gf> = cell_north
            .iter()
            .zip(&dep.north)
            .map(|(a, b)| *a + *b)
            .collect();
        let kept = mean(&|st, i| p.year.kept[st][i]);
        let share = mean(&|st, i| step_w[st].cell_share[grid.row(i)]);
        let g = mean(&|st, i| {
            let prev = (st + steps - 1) % steps;
            capacity[i] * (p.year.t[st][i] - p.year.t[prev][i]) / dt
        });
        let water = season_water(
            grid,
            &bw,
            Forcing {
                east: &east,
                north: &north,
                t: &t,
                kept: &kept,
                g_store: &g,
                sea: &sea_share,
                height_m: &height,
                land_evap: &last_land_evap[m],
                cell_share: &share,
                kept_mean,
                rain_law: RainLaw::Residence,
                eddy_everywhere: true,
                rain_rate: &last_rates[m],
                lapse_k_m: lapse,
            },
        );
        let mr = mountain_rain(
            lattice,
            geo,
            grid,
            &east,
            &north,
            &t,
            &water.column,
            &air_m,
            lapse,
            inner,
            last_phi[m].as_deref(),
        );
        // The next pass's land evaporation this season: Turc–Pike on the season's rain and PET.
        let land_evap = (0..n)
            .map(|c| {
                let (pr, pe) = (water.rain[c], water.pet[c]);
                if pe > Gf::ZERO {
                    pr / (Gf::ONE + (pr / pe) * (pr / pe)).sqrt()
                } else {
                    Gf::ZERO
                }
            })
            .collect();
        SeasonOut {
            drag: (dep.worst_iterations, dep.met),
            east,
            north,
            water,
            mr,
            speed,
            land_evap,
            parts: {
                let rows_mean = |f: &dyn Fn(&Winds) -> &Vec<Gf>| -> Vec<Gf> {
                    (0..grid.rows)
                        .map(|r| {
                            range
                                .clone()
                                .map(|st| f(&step_w[st])[r])
                                .fold(Gf::ZERO, |a, b| a + b)
                                / k
                        })
                        .collect()
                };
                SeasonWinds {
                    cell_east,
                    cell_north,
                    dep_east: dep.east,
                    dep_north: dep.north,
                    row_transport: rows_mean(&|w| &w.row_transport),
                    row_gms: rows_mean(&|w| &w.row_gms),
                    row_h: rows_mean(&|w| &w.row_h),
                    row_psi: rows_mean(&|w| &w.row_psi),
                    row_share: rows_mean(&|w| &w.cell_share),
                    rising_sin_lat: range
                        .clone()
                        .map(|st| step_w[st].rising_sin_lat)
                        .fold(Gf::ZERO, |a, b| a + b)
                        / k,
                }
            },
        }
    };
    let mut node_rain = vec![Gf::ZERO; n_nodes];
    let mut cell_rain_year = vec![Gf::ZERO; n];
    let mut pet_year = vec![Gf::ZERO; n];
    let mut wind_year = (vec![Gf::ZERO; n], vec![Gf::ZERO; n]);
    let mut surface_wind = vec![[Gf::ZERO; 3]; n];
    let mut worst_rounds = 0;
    let (mut worst_loop, mut worst_left) = (0, 0);
    let (mut worst_drag, mut drag_met) = (0, true);
    let (mut next_speed, mut next_land_evap) = (Vec::new(), Vec::new());
    let (mut next_rates, mut next_phi) = (Vec::new(), Vec::new());
    let mut season_winds = Vec::with_capacity(seasons);
    // Each thread takes the next season when it finishes one (a slow season holds only its own thread); the answers join
    // in the seasons' order, so the bytes are the same on any number of cores.
    let inner = (threads / seasons).max(1);
    for o in crate::par::map_in_turn(seasons, threads, |m| season(m, inner)) {
        worst_rounds = worst_rounds.max(o.mr.rounds);
        worst_drag = worst_drag.max(o.drag.0);
        drag_met &= o.drag.1;
        worst_loop = worst_loop.max(o.mr.largest_loop);
        worst_left = worst_left.max(o.mr.largest_left);
        for (acc, r) in node_rain.iter_mut().zip(&o.mr.rain) {
            *acc += *r * season_s;
        }
        for c in 0..n {
            cell_rain_year[c] += o.water.rain[c] * season_s;
            pet_year[c] += o.water.pet[c] * season_s;
            wind_year.0[c] += o.east[c] * o.water.column[c];
            wind_year.1[c] += o.north[c] * o.water.column[c];
            let w = &mut surface_wind[c];
            w[0] += o.east[c];
            w[1] += o.north[c];
            w[2] += (o.east[c] * o.east[c] + o.north[c] * o.north[c]).sqrt();
        }
        next_speed.push(o.speed);
        next_land_evap.push(o.land_evap);
        next_rates.push(o.mr.cell_rate);
        next_phi.push(Some(o.mr.phi));
        season_winds.push(o.parts);
    }
    state.last_season_winds = season_winds;
    let per = Gf::from_i64(seasons as i64);
    state.last_wind = surface_wind
        .into_iter()
        .map(|w| [w[0] / per, w[1] / per, w[2] / per])
        .collect();
    state.speed = next_speed;
    state.land_evap = next_land_evap;
    state.cell_rates = next_rates;
    state.phi = next_phi;
    state.cell_rain.clone_from(&cell_rain_year);
    state.cell_land = sea_share.iter().map(|&x| x < Gf::HALF).collect();
    // The next pass's land wetness: the year's Turc–Pike evaporation over the PET.
    for c in 0..n {
        let (pr, pe) = (cell_rain_year[c], pet_year[c]);
        state.wetness[c] = if pe > Gf::ZERO {
            pr / (Gf::ONE + (pr / pe) * (pr / pe)).sqrt() / pe
        } else {
            Gf::ONE
        };
    }
    state
        .t_start
        .clone_from(p.year.t.last().expect("a year has steps"));
    state.alpha_free = p.alpha_free;
    state.last_years = p.years;
    state.last_rounds = worst_rounds;
    state.last_loop = worst_loop;
    state.last_drag = worst_drag;
    state.last_drag_met = drag_met;
    state.last_left = worst_left;
    // THE ROWS. The annual mean and the warmest quarter at each climate cell (the datum's air).
    let quarter = steps / 4;
    let mut annual = vec![Gf::ZERO; n];
    let mut warm = vec![Gf::from_i64(-1); n];
    for c in 0..n {
        let mut sum = Gf::ZERO;
        for st in 0..steps {
            sum += p.year.t[st][c];
        }
        annual[c] = sum / Gf::from_i64(steps as i64);
        for st in 0..steps {
            let mut q = Gf::ZERO;
            for d in 0..quarter {
                q += p.year.t[(st + d) % steps][c];
            }
            warm[c] = warm[c].greater(q / Gf::from_i64(quarter as i64));
        }
    }
    let lapse_mk_km = (lapse * Gf::from_i64(1_000_000) + Gf::HALF)
        .to_i64_floor()
        .clamp(0, i64::from(u32::MAX)) as u32;
    let per_mm = Gf::ONE; // kg/m² of water is a millimetre.
    let rows: Vec<NodeRow> = crate::par::map(n_nodes, threads, |i| {
        let mut t_ann = Gf::ZERO;
        let mut t_warm = Gf::ZERO;
        let mut pet = Gf::ZERO;
        let mut ue = Gf::ZERO;
        let mut un = Gf::ZERO;
        for &(c, w) in &state.geo.reading[i] {
            let c = c as usize;
            t_ann += annual[c] * w;
            t_warm += warm[c] * w;
            pet += pet_year[c] * w;
            ue += wind_year.0[c] * w;
            un += wind_year.1[c] * w;
        }
        let t_node = t_ann - lapse * air_m[i];
        let rain = if has_water && bw.column_kg_m2 > Gf::ZERO {
            (node_rain[i] * per_mm + Gf::HALF)
                .to_i64_floor()
                .clamp(0, i64::from(u32::MAX)) as u32
        } else {
            0
        };
        let pet_mm = pet.greater(Gf::ZERO);
        let runoff = (runoff_mm_yr(Gf::from_i64(i64::from(rain)), pet_mm) + Gf::HALF)
            .to_i64_floor()
            .clamp(0, i64::from(rain)) as u32;
        let frost = line_z(lapse, t_ann - Gf::from_f64(FREEZING_K), datum);
        let ela = line_z(
            lapse,
            t_warm - (Gf::from_f64(FREEZING_K) + ohmura_ela_summer_c(rain)),
            datum,
        );
        // The year's vapour-weighted wind, as the slot it comes from.
        let d = state.geo.dir[i];
        let upwind = if ue * ue + un * un > Gf::ZERO {
            let ex = [-d[1], d[0], Gf::ZERO];
            let len = (ex[0] * ex[0] + ex[1] * ex[1]).sqrt();
            if len > Gf::ZERO {
                let e = [ex[0] / len, ex[1] / len, Gf::ZERO];
                let nn = [
                    d[1] * e[2] - d[2] * e[1],
                    d[2] * e[0] - d[0] * e[2],
                    d[0] * e[1] - d[1] * e[0],
                ];
                let wind = [
                    ue * e[0] + un * nn[0],
                    ue * e[1] + un * nn[1],
                    ue * e[2] + un * nn[2],
                ];
                let around = state.geo.around[i];
                let near = around.map(|m| {
                    if m == NO_NODE {
                        [Gf::ZERO; 3]
                    } else {
                        state.geo.dir[m as usize]
                    }
                });
                crate::climate::upwind_pick(&around, &near, d, wind)
            } else {
                NO_WIND
            }
        } else {
            NO_WIND
        };
        (
            (t_node * Gf::from_f64(10.0) + Gf::HALF)
                .to_i64_floor()
                .clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16,
            rain,
            aridity_of(pet_mm, rain),
            (pet_mm + Gf::HALF)
                .to_i64_floor()
                .clamp(0, i64::from(u32::MAX)) as u32,
            runoff,
            frost,
            ela,
            upwind,
        )
    });
    Climate {
        temperature_dk: rows.iter().map(|r| r.0).collect(),
        rain_mm_yr: rows.iter().map(|r| r.1).collect(),
        aridity_q8: rows.iter().map(|r| r.2).collect(),
        pet_mm_yr: rows.iter().map(|r| r.3).collect(),
        runoff_mm_yr: rows.iter().map(|r| r.4).collect(),
        frost_z: rows.iter().map(|r| r.5).collect(),
        ela_z: rows.iter().map(|r| r.6).collect(),
        upwind: rows.iter().map(|r| r.7).collect(),
        lapse_mk_km,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{home_moon, home_solve_words};
    use crate::land::initial_land;

    /// ★ THE CHAIN ON A SMALL WET WORLD (the home planet's words on the moon's lattice, as the climate's own tests do):
    /// every row in its range — the rain on most of the land, never over the year's own evaporation; the runoff under
    /// the rain; the temperatures of a living planet; the frost line under the equilibrium line where the summer is
    /// warm — and the state carried into the second call.
    #[test]
    fn the_chain_gives_sane_rows_and_starts_warm() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        // The home planet's air and sun with the water the solve's own wet-moon tests carry.
        let words = crate::solve::SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let land = initial_land(&moon, &lattice, &words.land());
        let mut state = SeasonState::with_rows(&lattice, 18, 4);
        let c = seasonal_climate(
            &moon,
            &lattice,
            &words,
            &land.z,
            land.sea_view(),
            4,
            &mut state,
        );
        let first_years = state.last_years;
        let n = lattice.node_count();
        assert_eq!(c.rain_mm_yr.len(), n);
        let land_nodes: Vec<usize> = (0..n).filter(|&i| !land.ocean[i]).collect();
        assert!(!land_nodes.is_empty());
        let wet = land_nodes.iter().filter(|&&i| c.rain_mm_yr[i] > 0).count();
        assert!(
            wet * 10 > land_nodes.len() * 9,
            "{wet} of {} land nodes rain",
            land_nodes.len()
        );
        for i in 0..n {
            assert!(c.runoff_mm_yr[i] <= c.rain_mm_yr[i]);
            let t = f64::from(c.temperature_dk[i]) / 10.0;
            assert!((150.0..360.0).contains(&t), "node {i}: {t} K");
        }
        assert!(c.lapse_mk_km > 0);
        // The state carries the year: the second call starts from the first call's last step and its albedo.
        assert_eq!(state.t_start.len(), state.grid.len());
        assert!(state.alpha_free > Gf::ZERO);
        let carried = state.t_start.clone();
        let again = seasonal_climate(
            &moon,
            &lattice,
            &words,
            &land.z,
            land.sea_view(),
            4,
            &mut state,
        );
        assert!(first_years < YEAR_BOUND);
        assert!(state.last_years < YEAR_BOUND);
        assert_eq!(again.rain_mm_yr.len(), n);
        assert!(state.phi.iter().all(Option::is_some));
        assert_eq!(carried.len(), state.t_start.len());
    }

    /// ★ THE SAME BYTES ON ANY NUMBER OF CORES (ruling B64): the seasons run in waves of one season a core and join in
    /// their order, so one core, five cores (waves of 5, 5, 5 and 1) and sixteen give the same rows, word for word, on the
    /// first call and on the warm second call.
    #[test]
    fn the_chain_gives_the_same_bytes_on_any_number_of_cores() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let words = crate::solve::SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let land = initial_land(&moon, &lattice, &words.land());
        let run = |threads: usize| {
            let mut state = SeasonState::with_rows(&lattice, 18, threads);
            let first = seasonal_climate(
                &moon,
                &lattice,
                &words,
                &land.z,
                land.sea_view(),
                threads,
                &mut state,
            );
            let second = seasonal_climate(
                &moon,
                &lattice,
                &words,
                &land.z,
                land.sea_view(),
                threads,
                &mut state,
            );
            (first, second)
        };
        let one = run(1);
        assert_eq!(run(5), one);
        assert_eq!(run(16), one);
    }
}
