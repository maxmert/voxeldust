//! ★ THE ONE COPY OF THE MACRO SOLVE FOR THE BENCHES (rulings B10, B11, B37, B38): a line-for-line copy of
//! `vd_terrain::solve::solve_full` in which a bench may switch the drains (the washboard bench's routing variants,
//! `examples/router_bench.rs`) and the laws of the belts' design (`docs/investigation/2026-10-02/belts_and_time_design.md`
//! §3; `examples/belts_bench.rs`) — IN THE COPY ONLY. Nothing in the game reads this module: the game's solve is
//! `vd_terrain::solve::solve_full`, untouched (the owner, 2026-10-01: no test seams in the game; ruling B38: one bench,
//! the binding rules unbroken). THE CONTROL: with `Router::D8` and the default `Laws`, the copy rebuilds the release
//! gate's pinned artifact bit for bit, or nothing a variant prints can be trusted.
//!
//! **Example.** The washboard bench routes the home planet by D∞ in the copy and reads its furrows; the belts' bench
//! raises the home planet's belts by their influx in the copy and reads their mountains beside Earth's. Neither run
//! changes a byte of the world the game solves.

use crate::router_variants::{
    Dinf, Fans, FillOpts, basin_fill_deposit, basin_fill_shaped, dinf, dinf_accumulate, dinf_cut,
    dinf_cut_skipping, dinf_sweep, dominant, paola_fan_offsets, rho8, single_links, units,
};
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::solve::SolveWords;
use vd_terrain::solve::{
    AGE_GATE_MIN_NODES, GLACIAL_EPOCH_YR, LAKE_SEA_SOLVES, MacroSolve, RouteReport, Schedule,
    coast_band, envelope_steps, settle_lakes_sea, tan_repose_q16,
};

/// ★ THE LAWS THE COPY MAY SWITCH (the belts' design §3; ruling B38: each one a published law with its own calibration
/// body, never a knob). The default is the solve's own: every law as the game runs it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Laws {
    /// ★ L0, THE LIMITS AS LAWS. `false`: the solve's own limits — the crater clip and the uplift re-clamp at the
    /// drawn relief (`solve.rs:1746-1756`) and the envelope after the loop (`:1867`). `true`: none of them; instead THE
    /// PEAK LAW acts on each node's STANDING height over the sea after every pass, at the law's own cap
    /// (`vd_terrain::land::relief_cap_m`: the crust's strength over its density and gravity, Everest's calibration),
    /// and a node over it collapses where it stands, its debris moved to its lowest neighbour (no mass vanishes). The
    /// land's own uplift clamp (`land.rs:1049-1057`) stays inside `initial_land`: it binds at no node on the home planet
    /// (MEASURED, `phase0_relief_cause.txt`).
    pub limits_as_law: bool,
    /// ★ L1, L2, L4, THE BELTS BY THEIR INFLUX. `false`: the belts rise by the room (`land.rs:1004-1016`), evenly over
    /// the age. `true`: a convergent margin on continental crust takes in crust at `F = v_n · T_in` (the world's one
    /// plate speed, `vd_terrain::craters::EARTH_MEAN_PLATE_SPEED_MM_YR`, times the convergence share; the incoming
    /// plate's continental thickness for a collision, the upper plate's shortening share for an ocean under a continent),
    /// spread by today's bump over both sides of a collision or one side of an Andean margin, raised by Airy's
    /// `(ρ_m − ρ_c)/ρ_m` of the world's own densities, ONLY over the pair's own window (the pair's draw times the plates'
    /// turnover `t_m = 2·A_oc/C0`, Parsons 1982). Trenches, arcs, ridges and rifts keep today's profiles: a stated
    /// interim until their own laws (the design's L8).
    pub belts_by_influx: bool,
    /// ★ L5, THE TIME AXIS. `None`: the solve's even passes (`age / passes` each). `Some(t_min)`: the passes keep their
    /// count on a GEOMETRIC axis from the present — the last pass `t_min` years long, each earlier pass longer by one
    /// ratio solved so the passes sum to the age EXACTLY in whole years (the first pass takes the rounding). Each pass's
    /// cut reads its own duration (the gain); every node's uplift and the glacial epoch's cut spread by TIME (the share of
    /// each pass's years), so the ice works only in the last 2.6 Myr. The routing, climate, talus and rebound cadences
    /// stay counted in passes, as the solve counts them (M-TIME-1 reads whether that matters).
    pub geometric_last_yr: Option<u64>,
    /// ★ B40 item 2, THE PLATES' MEAN SPEED: `true` reads the world's constant as its own text states it — a plate's MEAN
    /// speed is Earth's 50 mm/yr — so a unit drift share is 100 mm/yr (the share is uniform on [0, 1): its mean is ½).
    /// `false`: the constant as the craters read it today (a unit share 50 mm/yr, a mean of 25). The craters' own clock in
    /// the copy keeps the shipped reading (a bench interim; the build changes the one constant for both).
    pub mean_speed: bool,
    /// ★ B40 item 1, THE FORCE BALANCE AND THE WIDENING BELT: `true` holds every belt node's height to its own platform
    /// plus the force balance's height (`ΔF = ρ_c·g·h·[L + h·ρ_m/(2(ρ_m − ρ_c))] = F_b`, the force calibrated on Tibet's
    /// measured mean of 5 023 m over Earth's 35 km of crust and carried to a body by its elastic thickness, `L` the node's
    /// own continental crust); what a belt node cannot take is its pair's excess rock, which fills the pair's continental
    /// nodes nearest the boundary first, each up to its ceiling, so the belt WIDENS into a plateau; an excess with no room
    /// left carries into the next pass. Peaks keep the peak law (L0).
    pub force_balance: bool,
    /// ★ THE CRATERS' OWN TIME (ruling B37 item 1, the landscape's time applied to craters): `false` stamps every
    /// crater at the solve's start (the solve's own order, so even a young crater erodes for the whole age); `true`
    /// stamps each crater in the pass where its impact falls (the solve's start plus the age less the crater's age), so it
    /// erodes only since it was made. The population is the crater law's own in both.
    pub craters_at_own_age: bool,
}

/// ★ THE FORCE'S CALIBRATION BODY (B40 item 1): the Tibetan Plateau's measured mean height, metres (Fielding et al.
/// 1994, Geology 22: the arid central and northern interior), over Earth's continental crust of 35 km, under Earth's
/// gravity; the force it implies is carried to a body by the ratio of its elastic thickness to Earth's.
pub const TIBET_MEAN_M: f64 = 5_023.0;
/// Earth's surface gravity for the calibration, m/s².
pub const EARTH_GRAVITY_M_S2: f64 = 9.81;

/// ★ THE PASSES' BOUNDARIES in years from the solve's start, `passes + 1` of them, `0` first and `age` last: even
/// (`age · p / passes`), or geometric from the present with the last pass `t_min` long (see [`Laws::geometric_last_yr`]).
#[must_use]
pub fn pass_times(age: u64, passes: u32, geometric_last_yr: Option<u64>) -> Vec<u64> {
    let p = u64::from(passes.max(1));
    let Some(t_min) = geometric_last_yr else {
        return (0..=p)
            .map(|k| (u128::from(age) * u128::from(k) / u128::from(p)) as u64)
            .collect();
    };
    // The ratio r ≥ 1 with t_min · (r^P − 1)/(r − 1) = age, by bisection (a fixed count: one answer everywhere).
    let total = |r: f64| -> f64 {
        if (r - 1.0).abs() < 1e-12 {
            t_min as f64 * p as f64
        } else {
            t_min as f64 * (r.powi(p as i32) - 1.0) / (r - 1.0)
        }
    };
    let (mut lo, mut hi) = (1.0f64, 2.0f64);
    while total(hi) < age as f64 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if total(mid) < age as f64 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r = 0.5 * (lo + hi);
    // The lengths from the last pass back; the first pass takes the rounding so the sum is the age exactly.
    let mut lengths: Vec<u64> = (0..p)
        .map(|k| (t_min as f64 * r.powi((p - 1 - k) as i32)).round() as u64)
        .collect();
    let rest: u64 = lengths[1..].iter().sum();
    lengths[0] = age.saturating_sub(rest);
    let mut times = Vec::with_capacity(lengths.len() + 1);
    let mut t = 0u64;
    times.push(t);
    for l in lengths {
        t += l;
        times.push(t);
    }
    times
}

/// The sweep's gain for one pass of `dt_yr` years: [`Schedule::gain`]'s own product, `Δt · K₀/√(mm per m) · 2^24`,
/// floored under the fence.
fn gain_for(dt_yr: u64, k0_per_yr: f64) -> u64 {
    use vd_terrain::Gf;
    let k = Gf::from_f64(k0_per_yr) / Gf::from_f64(vd_terrain::solve::MM_PER_M).sqrt();
    (Gf::from_i64(dt_yr as i64) * k * Gf::from_i64(1i64 << vd_terrain::solve::GAIN_BITS))
        .to_i64_floor() as u64
}

/// The years of `[a0, a1)` inside `[t0, t1)`.
fn overlap(a0: u64, a1: u64, t0: u64, t1: u64) -> u64 {
    a1.min(t1).saturating_sub(a0.max(t0))
}

/// ★ THE ANDEAN SHARE: the part of the convergence that shortens the upper plate of an ocean under a continent — Earth's
/// ratio as a STATED CALIBRATION (Norabuena et al. 1998: 10–15 mm/yr of inland shortening against 60–80 mm/yr of
/// convergence; the middle, 12.5/70). The design's Q-B5 asks whether a law (the megathrust's stress) replaces it.
pub const ANDEAN_SHARE: f64 = 12.5 / 70.0;
/// The window draw's salt: the ASCII of "WINDOW".
pub const WINDOW_SALT: u64 = 0x5749_4e44_4f57;

/// What the belts' law found on a body.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeltReport {
    /// The nodes the influx raises: all, in a collision, at an Andean margin.
    pub nodes: usize,
    pub collision_nodes: usize,
    pub andean_nodes: usize,
    /// The surface uplift over each node's window, metres: the largest and the mean over the belt nodes.
    pub max_total_m: f64,
    pub mean_total_m: f64,
    /// The plates' turnover `t_m`, years; its inputs: the oceanic crust's area (m²), the ridges' length (m) and
    /// `C0 = Σ length × opening` (m²/yr).
    pub t_m_yr: f64,
    pub ocean_area_m2: f64,
    pub ridge_length_m: f64,
    pub c0_m2_yr: f64,
    /// The windows over the belt nodes, million years, at 10/25/50/75/90 %.
    pub windows_myr: [f64; 5],
    /// The convergence at the belt nodes, mm/yr, at 10/50/90 %.
    pub vn_mm_yr: [f64; 3],
    /// The surface uplift RATE at the belt nodes, mm/yr, at 50 % and the largest.
    pub rate_mm_yr: [f64; 2],
}

/// The belts by their influx: each node's surface uplift over its window (sixteenths; zero where no belt), the window's
/// start in years from the solve's start, and the report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Belts {
    pub total: Vec<i64>,
    pub start_yr: Vec<u64>,
    pub report: BeltReport,
    /// Each node's convergent pair (an index into `pairs`), `u32::MAX` where none (B40 item 1).
    pub pair: Vec<u32>,
    /// Each node's ceiling: its platform's height plus the force balance's height, sixteenths (`i32::MAX` where none).
    pub ceiling: Vec<i32>,
    /// The convergent pairs: their continental nodes nearest the boundary first, with each node's distance in metres.
    pub pairs: Vec<Vec<(u32, u32)>>,
}

/// The force balance's height, metres: the root of `ρ_c·g·[L·h + h²·ρ_m/(2Δρ)] = F_b`.
fn force_height_m(force_n_m: f64, crust_m: f64, gravity_m_s2: f64) -> f64 {
    use vd_terrain::land::{CONTINENTAL_CRUST_DENSITY_KGM3 as RC, MANTLE_DENSITY_KGM3 as RM};
    let a = RC * gravity_m_s2 * RM / (2.0 * (RM - RC));
    let b = RC * gravity_m_s2 * crust_m;
    (-b + (b * b + 4.0 * a * force_n_m).sqrt()) / (2.0 * a)
}

/// A node's direction as fenced floats: `vd_terrain::land::dir_of`, re-stated (it is crate-private), the integer
/// direction over `2^DIR_BITS`.
fn dir_gf(lattice: &vd_terrain::macro_lattice::MacroLattice, node: u32) -> [vd_terrain::Gf; 3] {
    use vd_terrain::Gf;
    let one = Gf::from_i64(1i64 << vd_recipe::bend::DIR_BITS);
    let d = lattice.direction(node);
    [
        Gf::from_i64(d[0].raw()) / one,
        Gf::from_i64(d[1].raw()) / one,
        Gf::from_i64(d[2].raw()) / one,
    ]
}

/// A unit draw from a hash: the top 53 bits over `2^53`.
fn unit_of(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// The value under which `share` of the sorted values lie.
fn quantile(sorted: &[f64], share: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * share).round() as usize]
}

/// ★ THE BELTS' LAW (the design's L1, L2, L4) on the initial land: see [`Laws::belts_by_influx`].
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn belts_by_influx(
    body: &BodyDefinition,
    words: &SolveWords,
    land: &vd_terrain::land::Land,
    age_yr: u64,
    mean_speed: bool,
) -> Belts {
    use vd_terrain::Gf;
    use vd_terrain::land::{
        BELT_HALF_WIDTHS, CONTINENTAL_CRUST_DENSITY_KGM3, EARTH_CONTINENTAL_CRUST_M,
        EARTH_RELIEF_CAP_M, Kind, MANTLE_DENSITY_KGM3, boundary_of, bump, flexural_parameter_m,
        relief_cap_m,
    };
    let lattice = body
        .macro_lattice()
        .expect("a body with a belts' law has a lattice");
    let n = lattice.node_count();
    let radius = body.ladder().radius_m();
    let gravity = body.facts().gravity_mm_s2;
    let half_width = flexural_parameter_m(words.elastic_thickness_m, gravity) * BELT_HALF_WIDTHS;
    let continental_m = EARTH_CONTINENTAL_CRUST_M * relief_cap_m(body) / EARTH_RELIEF_CAP_M
        * land.freeboard_factor.to_f64();
    let airy = (MANTLE_DENSITY_KGM3 - CONTINENTAL_CRUST_DENSITY_KGM3) / MANTLE_DENSITY_KGM3;
    let speed_m_yr = vd_terrain::craters::EARTH_MEAN_PLATE_SPEED_MM_YR / 1_000.0
        * if mean_speed { 2.0 } else { 1.0 };
    let node_m = lattice.node_m();
    let bounds: Vec<vd_terrain::land::Boundary> = (0..n as u32)
        .map(|k| boundary_of(&land.plates, dir_gf(&lattice, k)))
        .collect();
    // The plates' clock: the ridges' length and opening (the nodes within half a node of a divergent boundary's
    // bisector), the oceanic crust's area, and Parsons's turnover.
    let (mut c0, mut ridge, mut ocean) = (0.0f64, 0.0f64, 0.0f64);
    for (k, (b, &crust)) in bounds.iter().zip(&land.crust).enumerate() {
        let area = lattice.area_m2(k as u32) as f64;
        if crust < 128 {
            ocean += area;
        }
        if b.kind == Kind::Divergent && radius * b.sine.to_f64() < node_m / 2.0 {
            let length = area / node_m;
            ridge += length;
            c0 += length * b.convergence.to_f64().abs() * speed_m_yr;
        }
    }
    let t_m = if c0 > 0.0 { 2.0 * ocean / c0 } else { 0.0 };
    let mut total = vec![0i64; n];
    let mut start_yr = vec![age_yr; n];
    let mut report = BeltReport {
        t_m_yr: t_m,
        ocean_area_m2: ocean,
        ridge_length_m: ridge,
        c0_m2_yr: c0,
        ..BeltReport::default()
    };
    let (mut windows, mut vns, mut rates) = (Vec::new(), Vec::new(), Vec::new());
    let mut sum_total = 0.0f64;
    for k in 0..n {
        let b = bounds[k];
        let Some(o) = b.other else { continue };
        if b.kind != Kind::Convergent || land.crust[k] < 128 || t_m <= 0.0 {
            continue;
        }
        let shape = bump(Gf::from_f64(radius * b.sine.to_f64() / half_width)).to_f64();
        if shape <= 0.0 {
            continue;
        }
        let own = &land.plates[usize::from(b.plate)];
        let other = &land.plates[usize::from(o)];
        let v_n = b.convergence.to_f64() * speed_m_yr;
        let collision = other.affinity >= Gf::ZERO;
        let (influx, spread_m) = if collision {
            (
                v_n * continental_m * other.crust_scatter.to_f64(),
                16.0 / 15.0 * half_width,
            )
        } else {
            (
                ANDEAN_SHARE * v_n * continental_m * own.crust_scatter.to_f64(),
                8.0 / 15.0 * half_width,
            )
        };
        let rate = influx * shape / spread_m * airy;
        let (lo, hi) = (b.plate.min(o), b.plate.max(o));
        let share = unit_of(vd_seed::rng::child_seed(
            body.seed(),
            WINDOW_SALT,
            (u64::from(lo) << 8) | u64::from(hi),
        ));
        let window = share * t_m;
        let up = rate * window;
        total[k] = (up * f64::from(vd_terrain::solve::Z_STEPS_PER_M)) as i64;
        start_yr[k] = age_yr.saturating_sub(window as u64);
        report.nodes += 1;
        if collision {
            report.collision_nodes += 1;
        } else {
            report.andean_nodes += 1;
        }
        report.max_total_m = report.max_total_m.max(up);
        sum_total += up;
        windows.push(window / 1e6);
        vns.push(v_n * 1_000.0);
        rates.push(rate * 1_000.0);
    }
    for list in [&mut windows, &mut vns, &mut rates] {
        list.sort_by(f64::total_cmp);
    }
    report.mean_total_m = sum_total / report.nodes.max(1) as f64;
    // B40 item 1: the convergent pairs, each continental node's ceiling, and the pairs' candidates nearest first.
    let force_n_m = {
        use vd_terrain::land::{CONTINENTAL_CRUST_DENSITY_KGM3 as RC, MANTLE_DENSITY_KGM3 as RM};
        let h = TIBET_MEAN_M;
        RC * EARTH_GRAVITY_M_S2
            * h
            * (EARTH_CONTINENTAL_CRUST_M + h * RM / (2.0 * (RM - RC)))
            * f64::from(words.elastic_thickness_m)
            / vd_terrain::land::EARTH_ELASTIC_THICKNESS_M
    };
    let g = f64::from(gravity) / 1_000.0;
    let mut keys: Vec<(u8, u8)> = Vec::new();
    let mut pair = vec![u32::MAX; n];
    let mut ceiling = vec![i32::MAX; n];
    let mut pairs: Vec<Vec<(u32, u32)>> = Vec::new();
    for k in 0..n {
        let b = bounds[k];
        let Some(o) = b.other else { continue };
        if b.kind != Kind::Convergent || land.crust[k] < 128 {
            continue;
        }
        let key = (b.plate.min(o), b.plate.max(o));
        let id = match keys.iter().position(|&x| x == key) {
            Some(i) => i,
            None => {
                keys.push(key);
                pairs.push(Vec::new());
                keys.len() - 1
            }
        };
        pair[k] = id as u32;
        let crust = continental_m * land.plates[usize::from(b.plate)].crust_scatter.to_f64();
        let h = force_height_m(force_n_m, crust, g);
        ceiling[k] = land.z[k] + (h * f64::from(vd_terrain::solve::Z_STEPS_PER_M)) as i32;
        pairs[id].push(((radius * b.sine.to_f64()) as u32, k as u32));
    }
    for list in &mut pairs {
        list.sort_unstable();
    }
    report.windows_myr = [0.1, 0.25, 0.5, 0.75, 0.9].map(|q| quantile(&windows, q));
    report.vn_mm_yr = [0.1, 0.5, 0.9].map(|q| quantile(&vns, q));
    report.rate_mm_yr = [quantile(&rates, 0.5), rates.last().copied().unwrap_or(0.0)];
    Belts {
        total,
        start_yr,
        report,
        pair,
        ceiling,
        pairs,
    }
}

/// The share of a window `[start, end)` that falls in `[t0, t1)`, as an exact cumulative split of `total`:
/// `cum(t) = ⌊total · clamp(t − start, 0, end − start)/(end − start)⌋`, the step `cum(t1) − cum(t0)`.
fn window_step(total: i64, start: u64, end: u64, t0: u64, t1: u64) -> i64 {
    if end <= start {
        return 0;
    }
    let span = i128::from(end - start);
    let cum = |t: u64| -> i128 {
        let inside = i128::from(t.clamp(start, end) - start);
        (i128::from(total) * inside).div_euclid(span)
    };
    (cum(t1) - cum(t0)) as i64
}

/// The peak law's rounds in one pass: a fixed count (one answer on every host), each a sweep in node order.
const PEAK_ROUNDS: usize = 256;

/// ★ THE PEAK LAW per node (L0): a node whose standing height over the sea passes the law's cap collapses where it
/// stands; its excess volume goes to its lowest neighbour, and the rounds repeat until no node stands over the cap or
/// the fixed count ends — so the debris runs out downhill until it finds room, and no volume vanishes (the volume moved
/// is the excess times the node's area, laid over the neighbour's area). Returns the collapses summed over the rounds
/// and the largest excess still over the cap after the last round, in sixteenths.
fn peak_law(state: &mut MacroSolve, cap: i32) -> (usize, i32) {
    let n = state.node_count();
    let mut collapsed = 0usize;
    for _ in 0..PEAK_ROUNDS {
        let mut moved = 0usize;
        for i in 0..n {
            let over = state.z[i] - state.sea_z;
            if over <= cap {
                continue;
            }
            let excess = over - cap;
            let mut lowest: Option<u32> = None;
            for m in state.lattice.neighbours(i as u32) {
                if m == vd_terrain::macro_lattice::NO_NODE {
                    continue;
                }
                if lowest.is_none_or(|l| state.z[m as usize] < state.z[l as usize]) {
                    lowest = Some(m);
                }
            }
            state.z[i] -= excess;
            if let Some(m) = lowest {
                let volume = i128::from(excess) * i128::from(state.area[i]);
                state.z[m as usize] +=
                    volume.div_euclid(i128::from(state.area[m as usize].max(1))) as i32;
            }
            moved += 1;
        }
        collapsed += moved;
        if moved == 0 {
            break;
        }
    }
    let left = state
        .z
        .iter()
        .map(|&z| (z - state.sea_z - cap).max(0))
        .max()
        .unwrap_or(0);
    (collapsed, left)
}

/// A picture of a stage: the stage's name and the field in sixteenths (the washboard bench's trace).
pub type Shot<'a> = &'a dyn Fn(&str, &[i32]);

/// The routing variant (the washboard bench, ruling B10): `D8` is the solve's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Router {
    D8,
    Rho8,
    Dinf,
    /// D8's routing with the BASIN FILL in place of the solve's deposit.
    Fill,
    /// D∞'s routing with the basin fill.
    DinfFill,
    /// D8's routing with Paola's sloped fans and the skip bit (the deposit's nodes stay uncut).
    Fan,
    /// The same without the skip bit: how much of a fan the next sweep grades away.
    FanNoSkip,
    /// D∞'s routing with Paola's sloped fans and the skip bit — ruling B11's candidate world change.
    DinfFan,
}

impl Router {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Router::D8 => "d8",
            Router::Rho8 => "rho8",
            Router::Dinf => "dinf",
            Router::Fill => "fill",
            Router::DinfFill => "dinf+fill",
            Router::Fan => "fan",
            Router::FanNoSkip => "fan-noskip",
            Router::DinfFan => "dinf+fan",
        }
    }
}

/// What the copy leaves: the final state, the facies, and the routes the lakes' fixed point read.
pub struct Solved {
    pub state: MacroSolve,
    pub facies: Vec<u8>,
    pub rewrites: Vec<String>,
    /// The belts' law's own reading, where the copy raised the belts by their influx.
    pub belts: Option<Belts>,
    /// The craters stamped at the start (each centre and width), for a reading by cause.
    pub craters: Vec<vd_terrain::craters::Crater>,
    /// The nodes the peak law collapsed, summed over the passes and their rounds (L0).
    pub collapsed: usize,
    /// The largest excess over the peak law's cap left after any pass's last round, sixteenths (zero: it settled).
    pub peak_left: i32,
    /// The belts' excess rock that found no room by the end, m³ (B40 item 1; zero: every metre placed).
    pub carry_left_m3: f64,
}

/// ★ THE COPY of `vd_terrain::solve::solve_full` (generator 22), line for line, with the routing variant's
/// three hooks marked `VARIANT` and the laws' hooks marked `LAW`. Every other line is the solve's own; the control
/// run proves it (the routing bench's `--planet d8` must rebuild the release gate's pinned artifact bit for bit).
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn solve_copy(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
    v: Router,
    laws: &Laws,
    trace: Option<Shot<'_>>,
) -> Option<Solved> {
    let lattice = body.macro_lattice()?;
    let mut land = vd_terrain::land::initial_land(body, &lattice, &words.land());
    // LAW (L1, L2, L4): the belts by their influx replace the room's belts; their nodes read the folded belt's rock.
    let belts = laws
        .belts_by_influx
        .then(|| belts_by_influx(body, words, &land, schedule.age_yr, laws.mean_speed));
    let mut even_total = land.uplift.clone();
    if let Some(b) = &belts {
        for (k, &t) in b.total.iter().enumerate() {
            if t > 0 {
                even_total[k] = 0;
                land.province[k] = vd_terrain::strata::Province::FoldedBelt.code();
            }
        }
    }
    if let Some(shot) = trace {
        shot("00_land", &land.z);
    }
    let mut state = MacroSolve::from_land(body, &land)?;
    let gravity = body.facts().gravity_mm_s2;
    let radius_m = body.radius_m();
    let u = if matches!(v, Router::D8 | Router::Fill) {
        // (the fan variants read chords, so they take the units)
        Vec::new()
    } else {
        units(&lattice)
    };
    let mut rewrites = Vec::new();
    let pre_climate = vd_terrain::climate::climate(body, &lattice, words, &land.z, land.sea_view());
    let surface = vd_terrain::craters::Surface {
        crust: &land.crust,
        boundary_m: &land.boundary_m,
        plate: &land.plate,
        plates: &land.plates,
        province: &land.province,
        rain_mm_yr: &pre_climate.rain_mm_yr,
    };
    let craters = vd_terrain::craters::crater_population(body, &lattice, words, &surface);
    // LAW (B37, the craters' own time): today every crater is stamped here; on its own time each waits for its pass.
    if !laws.craters_at_own_age {
        vd_terrain::craters::apply_craters(&mut state.z, &lattice, &craters, gravity);
    }
    let crater_record = craters.clone();
    drop(pre_climate);
    let relief = envelope_steps(body);
    // LAW (L0): the crater clip and the uplift re-clamp are the solve's own limits; as laws they do not exist.
    if !laws.limits_as_law {
        for z in &mut state.z {
            *z = (*z).clamp(-relief, relief);
        }
        for (u, &z) in state.uplift.iter_mut().zip(&state.z) {
            *u = (*u).clamp(-relief - z, relief - z);
        }
    }
    // LAW (L0): the peak law's cap, the law's own relief (never the drawn share of it), in sixteenths.
    let peak_cap =
        (vd_terrain::land::relief_cap_m(body) * f64::from(vd_terrain::solve::Z_STEPS_PER_M)) as i32;
    let mut collapsed = 0usize;
    let mut peak_left = 0i32;
    // LAW (B40 item 1): each pair's excess rock with no room yet, volume in sixteenths × m², carried pass to pass.
    let mut carry: Vec<i128> = Vec::new();
    state.z_flood.clone_from(&state.z);
    if let Some(shot) = trace {
        shot("01_craters", &state.z);
    }
    let gain = schedule.gain();
    let alpha_m = vd_terrain::land::flexural_parameter_m(words.elastic_thickness_m, gravity);
    let mut since_route = schedule.flood_every;
    let mut since_climate = schedule.climate_every;
    let mut since_rebound = 0;
    let mut climate = None;
    let no_ice: Vec<i32> = if words.water_km3 == 0 {
        vec![i32::MAX; state.node_count()]
    } else {
        Vec::new()
    };
    let epoch_step = GLACIAL_EPOCH_YR / u64::from(schedule.passes.max(1));
    let epoch_left = if laws.geometric_last_yr.is_some() {
        0
    } else {
        GLACIAL_EPOCH_YR - epoch_step * u64::from(schedule.passes)
    };
    let mut routes: Vec<RouteReport> = Vec::new();
    // The talus relaxations the loop ran (the solve's own `report.talus.len()`).
    let mut talus_done = 0u32;
    // VARIANT: the two links of the last routing, for D∞'s cut between routings.
    let mut links: Option<Dinf> = None;
    // VARIANT (fans): the nodes the deposit ever raised (the skip bit), the last pass's fans over
    // their spills with their volume, and a pass's own span of time.
    let mut deposited = vec![false; state.node_count()];
    let mut above: Option<(Vec<(u32, i32)>, f64)> = None;
    let dt_yr = schedule.age_yr as f64 / f64::from(schedule.passes.max(1));
    // LAW (L5): the passes' boundaries in years, even or geometric.
    let times = pass_times(schedule.age_yr, schedule.passes, laws.geometric_last_yr);
    let epoch_start = schedule.age_yr.saturating_sub(GLACIAL_EPOCH_YR);
    for pass in 0..schedule.passes {
        let gain = if laws.geometric_last_yr.is_some() {
            gain_for(
                times[pass as usize + 1] - times[pass as usize],
                schedule.k0_per_yr,
            )
        } else {
            gain
        };
        let epoch_step = if laws.geometric_last_yr.is_some() {
            overlap(
                epoch_start,
                schedule.age_yr,
                times[pass as usize],
                times[pass as usize + 1],
            )
        } else {
            epoch_step
        };
        // LAW (B37, the craters' own time): the craters whose impact falls in this pass are stamped now.
        if laws.craters_at_own_age {
            let (t0, t1) = (times[pass as usize], times[pass as usize + 1]);
            let due: Vec<vd_terrain::craters::Crater> = crater_record
                .iter()
                .copied()
                .filter(|c| {
                    let at = schedule.age_yr.saturating_sub(c.age_yr);
                    at >= t0 && (at < t1 || (pass + 1 == schedule.passes && at <= t1))
                })
                .collect();
            if !due.is_empty() {
                vd_terrain::craters::apply_craters(&mut state.z, &lattice, &due, gravity);
            }
        }
        if since_climate >= schedule.climate_every {
            if let Some(sea) = vd_terrain::land::sea_level_connected(
                &lattice,
                &state.z,
                &state.area,
                words.water_km3,
                false,
            ) {
                state.sea_z = sea.level;
                state.ocean = sea.mask.ocean;
            }
            let c = vd_terrain::climate::climate(
                body,
                &lattice,
                words,
                &state.z,
                (words.water_km3 > 0).then_some(state.sea_view()),
            );
            state.rain.clone_from(&c.rain_mm_yr);
            climate = Some(c);
            since_climate = 0;
            since_route = schedule.flood_every;
        }
        since_climate += 1;
        if since_route >= schedule.flood_every {
            routes.push(state.route());
            // VARIANT 1 AND 2: the drains after the routing, and the accumulation over them.
            match v {
                Router::D8 => state.accumulate(),
                Router::Fill | Router::Fan | Router::FanNoSkip => {
                    state.accumulate();
                    links = Some(single_links(&state));
                }
                Router::Rho8 => {
                    let seed = state.seed;
                    let r = rho8(&mut state, &u, radius_m, seed);
                    rewrites.push(format!(
                        "pass {pass}: rho8 read {} slope nodes, moved {}, rings {}",
                        r.nodes, r.changed, r.cyclic
                    ));
                    state.accumulate();
                }
                Router::Dinf | Router::DinfFill | Router::DinfFan => {
                    let d = dinf(&state, &u, radius_m);
                    let rings = dominant(&mut state, &d);
                    rewrites.push(format!(
                        "pass {pass}: dinf {} facet nodes, {} split, rings {} (dominant tree rings {rings})",
                        d.facets, d.split, d.cyclic
                    ));
                    dinf_accumulate(&mut state, &d);
                    links = Some(d);
                }
            }
            since_route = 0;
        }
        since_route += 1;
        if let Some(c) = climate.as_ref() {
            if (pass + 1) * schedule.talus_passes / schedule.passes.max(1)
                > pass * schedule.talus_passes / schedule.passes.max(1)
            {
                let tan: Vec<u32> = c.aridity_q8.iter().map(|&a| tan_repose_q16(a)).collect();
                let _ = state.talus(&tan);
                talus_done += 1;
            }
            let line: &[i32] = if words.water_km3 == 0 {
                &no_ice
            } else {
                &c.ela_z
            };
            let _ = state.ice(line, &c.temperature_dk, c.lapse_mk_km, gravity, epoch_step);
        }
        // LAW (L1, L2, L4): each node's uplift for THIS pass, handed to the shipped sweep through its own word — the
        // sweep adds `uplift_step(total, pass, passes)`, so a total of `step · passes` adds exactly `step` (ARITH: the
        // cumulative floors of an exact multiple). A belt node takes its window's share of the pass; every other node
        // takes the solve's own even share of its total.
        // LAW (L5): on the geometric axis every node's even total spreads by TIME, not by the pass's index.
        let (t0, t1) = (times[pass as usize], times[pass as usize + 1]);
        if belts.is_some() || laws.geometric_last_yr.is_some() {
            let mut steps: Vec<i64> = even_total
                .iter()
                .enumerate()
                .map(|(i, &even)| {
                    let belt_total = belts.as_ref().map_or(0, |b| b.total[i]);
                    if belt_total > 0 {
                        let b = belts.as_ref().expect("a belt total has its belts");
                        window_step(belt_total, b.start_yr[i], schedule.age_yr, t0, t1)
                    } else if laws.geometric_last_yr.is_some() {
                        window_step(i64::from(even), 0, schedule.age_yr, t0, t1)
                    } else {
                        i64::from(MacroSolve::uplift_step(even, pass, schedule.passes))
                    }
                })
                .collect();
            // LAW (B40 item 1): every belt node up to its ceiling; the refused rock widens its pair nearest first.
            if let (true, Some(b)) = (laws.force_balance, belts.as_ref()) {
                if carry.len() != b.pairs.len() {
                    carry = vec![0i128; b.pairs.len()];
                }
                let mut excess = vec![0i128; b.pairs.len()];
                for (i, step) in steps.iter_mut().enumerate() {
                    if b.total[i] <= 0 || b.pair[i] == u32::MAX {
                        continue;
                    }
                    let room = i64::from((b.ceiling[i] - state.z[i]).max(0));
                    if *step > room {
                        excess[b.pair[i] as usize] +=
                            i128::from(*step - room) * i128::from(state.area[i]);
                        *step = room;
                    }
                }
                for (pi, list) in b.pairs.iter().enumerate() {
                    let mut remaining = excess[pi] + carry[pi];
                    for &(_, j) in list {
                        if remaining <= 0 {
                            break;
                        }
                        let j = j as usize;
                        let need = i64::from(b.ceiling[j] - state.z[j]) - steps[j];
                        if need <= 0 {
                            continue;
                        }
                        let area = i128::from(state.area[j].max(1));
                        let raise = i64::try_from((remaining / area).min(i128::from(need)))
                            .expect("a raise fits a 64-bit word");
                        steps[j] += raise;
                        remaining -= i128::from(raise) * area;
                        if raise < need {
                            break;
                        }
                    }
                    carry[pi] = remaining.max(0);
                }
            }
            for (u, step) in state.uplift.iter_mut().zip(&steps) {
                *u = i32::try_from(*step * i64::from(schedule.passes))
                    .expect("a pass's uplift times the pass count fits a 32-bit word");
            }
        }
        // VARIANT 3: D∞'s cut and deposit over both links; the others cut with the solve's own sweep.
        match (v, links.as_ref()) {
            (Router::Dinf, Some(d)) => {
                let _ = dinf_sweep(&mut state, d, gain, pass, schedule.passes);
            }
            (Router::Fan | Router::FanNoSkip | Router::DinfFan, Some(d)) => {
                let skip_on = v != Router::FanNoSkip;
                let _ = dinf_cut_skipping(
                    &mut state,
                    d,
                    gain,
                    pass,
                    schedule.passes,
                    skip_on.then_some(deposited.as_slice()),
                );
                // CHECK-FILL-SURVIVES: the last pass's fans over their spills, after this cut.
                if let Some((list, before)) = above.take() {
                    let after = volume_above(&state, &list);
                    if pass <= 3 || pass + 1 == schedule.passes {
                        rewrites.push(format!(
                            "pass {pass}: basin fill survives the cut: {:.3e} of {:.3e} m³ over the spills ({:.1} %)",
                            after, before, if before > 0.0 { 100.0 * after / before } else { f64::NAN }
                        ));
                    }
                }
                let c = climate
                    .as_ref()
                    .expect("the climate stands from the first pass");
                let talus: Vec<f64> = c
                    .aridity_q8
                    .iter()
                    .map(|&a| f64::from(tan_repose_q16(a)) / 65_536.0)
                    .collect();
                let fans = std::cell::RefCell::new(Fans::default());
                let shape = |st: &MacroSolve, nodes: &[u32], inl: &[(u32, u128, u128)]| {
                    paola_fan_offsets(
                        st,
                        nodes,
                        inl,
                        &u,
                        radius_m,
                        dt_yr,
                        &talus,
                        &mut fans.borrow_mut(),
                    )
                };
                let (f, marks) = basin_fill_shaped(
                    &mut state,
                    d,
                    &FillOpts {
                        shape: Some(&shape),
                        runoff: Some(&c.runoff_mm_yr),
                        cap_at_spill: false,
                    },
                );
                for &r in &marks.raised {
                    deposited[r as usize] = true;
                }
                let over = volume_above(&state, &marks.above);
                above = Some((marks.above, over));
                if pass < 3 || pass + 1 == schedule.passes {
                    let fans = fans.borrow();
                    let mut sl = fans.slopes.clone();
                    sl.sort_by(f64::total_cmp);
                    let q = |p: f64| {
                        sl.get(((sl.len() as f64 - 1.0) * p) as usize)
                            .copied()
                            .unwrap_or(f64::NAN)
                    };
                    let mut tops = fans.top_m.clone();
                    tops.sort_by(f64::total_cmp);
                    rewrites.push(format!(
                        "pass {pass}: basin fill (fans) {} hollows, {} fed, {} full, {} nodes raised, {} steps; {} inlets ({} dry, {} capped at the scree), slope p10 {:.2e} p50 {:.2e} p90 {:.2e}, fan top p50 {:.0} m max {:.0} m; {:.3e} m³ over the spills",
                        f.basins, f.fed, f.full, f.filled, f.total_fill, fans.inlets, fans.dry, fans.capped, q(0.1), q(0.5), q(0.9),
                        tops.get(tops.len() / 2).copied().unwrap_or(f64::NAN), tops.last().copied().unwrap_or(f64::NAN), over
                    ));
                }
            }
            (Router::Fill | Router::DinfFill, Some(d)) => {
                let _ = dinf_cut(&mut state, d, gain, pass, schedule.passes);
                let f = basin_fill_deposit(&mut state, d);
                if pass < 3 || pass + 1 == schedule.passes {
                    rewrites.push(format!(
                        "pass {pass}: basin fill {} hollows, {} fed, {} full, {} nodes raised, {} steps",
                        f.basins, f.fed, f.full, f.filled, f.total_fill
                    ));
                }
            }
            _ => {
                let _ = state.sweep(gain, pass, schedule.passes);
            }
        }
        since_rebound += 1;
        if since_rebound >= schedule.isostasy_every {
            let _ = state.rebound(alpha_m);
            since_rebound = 0;
        }
        // LAW (L0): the peak law after every pass, on the height that stands.
        if laws.limits_as_law {
            let (c, left) = peak_law(&mut state, peak_cap);
            collapsed += c;
            peak_left = peak_left.max(left);
        }
        if let Some(shot) = trace
            && [0u32, 1, 4, 9, 19, schedule.passes - 1].contains(&pass)
        {
            shot(&format!("02_pass{:03}", pass + 1), &state.z);
        }
    }
    let climate = match climate {
        Some(c) => c,
        None => vd_terrain::climate::climate(
            body,
            &lattice,
            words,
            &state.z,
            (words.water_km3 > 0).then_some(state.sea_view()),
        ),
    };
    let tan: Vec<u32> = climate
        .aridity_q8
        .iter()
        .map(|&a| tan_repose_q16(a))
        .collect();
    // The talus relaxations the loop had no passes to hold: the count the schedule states runs.
    while talus_done < schedule.talus_passes {
        let _ = state.talus(&tan);
        talus_done += 1;
    }
    drop(tan);
    // LAW (L0): the envelope after the loop is the solve's own limit; as a law it does not exist.
    if !laws.limits_as_law {
        let _ = state.envelope(relief);
    }
    if let Some(shot) = trace {
        shot("03_envelope", &state.z);
    }
    if let Some(sea) = vd_terrain::land::sea_level_connected(
        &lattice,
        &state.z,
        &state.area,
        words.water_km3,
        false,
    ) {
        state.sea_z = sea.level;
    }
    routes.push(state.route());
    let ice_mask = {
        let line: &[i32] = if words.water_km3 == 0 {
            &no_ice
        } else {
            &climate.ela_z
        };
        let (mask, _, _, _) = state.ice(
            line,
            &climate.temperature_dk,
            climate.lapse_mk_km,
            gravity,
            epoch_left,
        );
        mask
    };
    let mut final_climate = climate;
    let mut budget = None;
    if words.water_km3 > 0 {
        let mut climate =
            vd_terrain::climate::climate(body, &lattice, words, &state.z, Some(state.sea_view()));
        let mut lakes = vd_terrain::lakes::budget(&state, &climate);
        let volume = lakes.report.volume_m3;
        let _ = settle_lakes_sea(
            &mut state,
            words.water_km3,
            LAKE_SEA_SOLVES,
            volume,
            &mut routes,
            &mut |s: &MacroSolve| {
                climate =
                    vd_terrain::climate::climate(body, &lattice, words, &s.z, Some(s.sea_view()));
                lakes = vd_terrain::lakes::budget(s, &climate);
                lakes.report.volume_m3
            },
        );
        state.water_z.clone_from(&lakes.water);
        final_climate = climate;
        budget = Some(lakes);
    }
    let _ = state.hypsometric_integrals(AGE_GATE_MIN_NODES);
    let _ = state.sediment_by_basin();
    let tree = match &budget {
        Some(lakes) => vd_terrain::lakes::water_tree(&state, lakes),
        None => vd_terrain::lakes::dry_tree(&state),
    };
    drop(budget);
    let _ = state.follow_the_water(tree, &final_climate.runoff_mm_yr);
    drop(final_climate);
    let facies = state.facies(&ice_mask, coast_band(words, gravity), words.water_km3 > 0);
    Some(Solved {
        state,
        facies,
        rewrites,
        belts,
        craters: crater_record,
        collapsed,
        peak_left,
        carry_left_m3: carry.iter().sum::<i128>() as f64
            / f64::from(vd_terrain::solve::Z_STEPS_PER_M),
    })
}

/// The volume the listed nodes stand over their spills, in m³ (heights in sixteenths).
fn volume_above(state: &MacroSolve, list: &[(u32, i32)]) -> f64 {
    list.iter()
        .map(|&(v, spill)| {
            let i = v as usize;
            f64::from((state.z[i] - spill).max(0)) / 16.0 * state.area[i] as f64
        })
        .sum()
}

/// The artifact exactly as `vd_bins::artifact_worker::run_solve` makes it from a solved state.
#[must_use]
pub fn artifact_of(body: &BodyDefinition, words: &SolveWords, s: &Solved) -> Artifact {
    let lattice = body.macro_lattice().expect("a lattice");
    let climate =
        vd_terrain::climate::climate(body, &lattice, words, &s.state.z, Some(s.state.sea_view()));
    Artifact::of(&s.state, &s.facies, &climate, words.water_km3 > 0)
}
