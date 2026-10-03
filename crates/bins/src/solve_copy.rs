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
    /// ★ F2, THE JOINT LAW (the design §11; the owner's choice to bench it, 2026-10-03): `false` cuts with the solve's
    /// own stream power (`n = 1`, `m = ½`, `K₀`); `true` cuts with ONE published joint calibration, Adams et al. 2020
    /// (Science Advances 6, eaaz3166, eq. 1b and its regional best fit): `E = K_lp · Q^m · S^n`, `Q` the discharge
    /// (drainage area times rain, m³/yr), `K_lp = 2.2 × 10⁻⁹ m⁻²`, `m = 1`, `n = 2.2` — calibration body: 142 basins of
    /// Bhutan and central-eastern Nepal (¹⁰Be). The rock shares read against the crystalline basement (the calibration
    /// body's gneiss). The implicit step is solved per node by Newton, outlets first, with the shipped bookkeeping, so the
    /// deposit and the rebound read it as they read the solve's own cut. Its authors: "we do not suggest that this
    /// represents a universal n value".
    pub joint_law: bool,
    /// ★ THE MIXED AXIS (ruling B39's gate: short passes everywhere): `Some((t_min, t_max))` — from the present backwards
    /// each pass is [`MIXED_RATIO`] times the next, the last `t_min` years, until a pass reaches `t_max`; then even passes
    /// of `t_max` back to the start, the first taking the remainder, so the passes sum to the age exactly. The pass count
    /// follows from the axis ([`mixed_times`]); the schedule must state it. Overrides `geometric_last_yr`.
    pub mixed_axis: Option<(u64, u64)>,
    /// ★ P2, THE CRUST'S PARTITION AT A COLLISION (the design §12; the owner's choice to research and bench it,
    /// 2026-10-03): `false` keeps every metre of the incoming crust in the belt; `true` keeps only the crust ABOVE THE
    /// DECOUPLING LEVEL, and the crust under it goes down into the mantle with the plate's mantle lithosphere. The level
    /// is India's measured stored crust, [`INDIA_STORED_CRUST_M`] of its 38 km (Replumaz et al. 2010, Tectonophysics
    /// 492: 40–50 % of India's crust recycled, "a decoupling level at about 15–19 km depth"; the whole collision's budget
    /// agrees: about 50 % of the crust is missing, Ingalls et al. 2016, Nature Geoscience 9) — calibration body
    /// India–Asia, carried to a body by its elastic thickness over Earth's (as B40's force). An Andean margin keeps its
    /// own law: its influx is the upper plate's own shortening, which no plate carries down.
    pub crust_partition: bool,
    /// ★ S2, THE COLLISION SLOWS (the design §12): `false` closes a collision at its convergence for its whole window;
    /// `true` closes it at `v_n · e^(−t/τ)`, `t` the time since the collision began, `τ = 1/ε̇` from the constant
    /// contractional strain rate across Tibet, [`TIBET_STRAIN_RATE_PER_S`] (Clark 2012, Nature 483: India's penetration
    /// into Eurasia "has decreased exponentially since their collision"; `τ` = 45.1 Myr) — calibration body Tibet;
    /// `v_n` reads as the speed when the collision began. An Andean margin keeps its own law.
    pub collision_slowing: bool,
    /// ★ A RESOLUTION, NOT A LAW (ruling B41 item 1, the gate's cause): `k > 1` runs the solve's own sweep `k` times in
    /// each pass on the pass's ONE routing, each with a `k`-th of the pass's gain, the uplift in the first only and the
    /// cut summed — so a pass's cut is stepped finer while the drains, the flood, the climate and the rebound keep their
    /// own cadence. If `k` sweeps in a 25 Myr pass cut as the 12.5 Myr axis cuts, the sweep's own step is the cause; if
    /// they cut as one sweep does, the routing every pass is. `0` and `1` are the solve's one sweep.
    pub sub_sweeps: u32,
    /// ★ A RESOLUTION, NOT A LAW (ruling B41 item 1): with `sub_sweeps > 1`, `true` finds the drains again (the flood,
    /// the receivers and the discharge) before every sub-sweep after the first, as a pass of a `k` times finer axis
    /// would, while the climate, the rebound and the talus keep the pass's cadence. If routed sub-sweeps cut as the finer
    /// axis cuts, the routing's cadence is the gate's cause.
    pub sub_route: bool,
}

/// ★ P2's calibration (Replumaz et al. 2010): the thickness of India's crust stored in the collision (in the thickened
/// crust or moved out sideways), metres, of the craton's 38 km — the decoupling level.
pub const INDIA_STORED_CRUST_M: f64 = 19_000.0;
/// ★ S2's calibration (Clark 2012): the constant contractional strain rate across Tibet, per second.
pub const TIBET_STRAIN_RATE_PER_S: f64 = 7.03e-16;
/// Seconds in a Julian year (365.25 days): S2's strain rate against the solve's years.
const SECONDS_PER_YEAR: f64 = 31_557_600.0;

/// The mixed axis's ratio between neighbouring recent passes (a numerical resolution, read by the convergence gate).
pub const MIXED_RATIO: f64 = 1.25;

/// ★ THE MIXED AXIS's boundaries in years from the solve's start (see [`Laws::mixed_axis`]).
#[must_use]
pub fn mixed_times(age: u64, t_min: u64, t_max: u64) -> Vec<u64> {
    let mut lengths_from_present: Vec<u64> = Vec::new();
    let mut length = t_min as f64;
    let mut covered = 0u64;
    while covered < age {
        let l = (length.round() as u64).min(t_max).max(1);
        let l = l.min(age - covered);
        lengths_from_present.push(l);
        covered += l;
        length = (length * MIXED_RATIO).min(t_max as f64);
    }
    // Oldest first: the passes run from the start to the present.
    lengths_from_present.reverse();
    let mut times = Vec::with_capacity(lengths_from_present.len() + 1);
    let mut t = 0u64;
    times.push(t);
    for l in lengths_from_present {
        t += l;
        times.push(t);
    }
    times
}

/// ★ F2's constants (Adams et al. 2020): `K_lp` in m⁻² (for `m = 1`), the discharge exponent, the slope exponent.
pub const JOINT_K_LP: f64 = 2.2e-9;
pub const JOINT_M: f64 = 1.0;
pub const JOINT_N: f64 = 2.2;
/// F2's Newton steps at most (the bench stops a node's iteration when it moves less than a micrometre; the build states a
/// fixed count as its deterministic form).
const JOINT_NEWTON: usize = 60;

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
    /// P2: the share of the incoming crust a collision keeps, at 10/50/90 % of the collision nodes (1 without the law).
    pub kept_share: [f64; 3],
    /// S2: a collision's influx over its window as a share of the constant speed's, at 10/50/90 % of the collision
    /// nodes (1 without the law).
    pub slowing_share: [f64; 3],
    /// The rock the belts raise over their windows (each node's surface uplift times its area), m³: in collisions and
    /// at Andean margins.
    pub volume_m3: [f64; 2],
}

/// A node's belt kind in [`Belts::kind`]: no belt by the influx law (the room's belt or no belt).
pub const BELT_NONE: u8 = 0;
/// A node's belt kind: a collision of two continents.
pub const BELT_COLLISION: u8 = 1;
/// A node's belt kind: an ocean under a continent (an Andean margin).
pub const BELT_ANDEAN: u8 = 2;

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
    /// Each node's belt kind: [`BELT_NONE`], [`BELT_COLLISION`] or [`BELT_ANDEAN`].
    pub kind: Vec<u8>,
    /// S2's e-folding time in years, where the slowing law acts (`None`: every window closes at a constant speed).
    pub tau_yr: Option<f64>,
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
    laws: &Laws,
) -> Belts {
    use vd_terrain::Gf;
    use vd_terrain::land::{
        BELT_HALF_WIDTHS, CONTINENTAL_CRUST_DENSITY_KGM3, EARTH_CONTINENTAL_CRUST_M,
        EARTH_ELASTIC_THICKNESS_M, EARTH_RELIEF_CAP_M, Kind, MANTLE_DENSITY_KGM3, boundary_of,
        bump, flexural_parameter_m, relief_cap_m,
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
        * if laws.mean_speed { 2.0 } else { 1.0 };
    // P2: the decoupling level, carried to the body by its elastic thickness; S2: the e-folding time, years.
    let decoupling_m =
        INDIA_STORED_CRUST_M * f64::from(words.elastic_thickness_m) / EARTH_ELASTIC_THICKNESS_M;
    let tau_yr = laws
        .collision_slowing
        .then(|| 1.0 / (TIBET_STRAIN_RATE_PER_S * SECONDS_PER_YEAR));
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
    let mut kind = vec![BELT_NONE; n];
    let mut report = BeltReport {
        t_m_yr: t_m,
        ocean_area_m2: ocean,
        ridge_length_m: ridge,
        c0_m2_yr: c0,
        ..BeltReport::default()
    };
    let (mut windows, mut vns, mut rates) = (Vec::new(), Vec::new(), Vec::new());
    let (mut kept_shares, mut slow_shares) = (Vec::new(), Vec::new());
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
            // P2: the belt keeps the incoming crust above the decoupling level.
            let incoming_m = continental_m * other.crust_scatter.to_f64();
            let kept_m = if laws.crust_partition {
                incoming_m.min(decoupling_m)
            } else {
                incoming_m
            };
            kept_shares.push(kept_m / incoming_m);
            (v_n * kept_m, 16.0 / 15.0 * half_width)
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
        // S2: a collision's influx falls as `e^(−t/τ)` over its window, so the window closes `τ·(1 − e^(−W/τ))` years'
        // worth of the starting speed's crust.
        let years = match tau_yr {
            Some(tau) if collision => {
                let y = -tau * (-window / tau).exp_m1();
                slow_shares.push(if window > 0.0 { y / window } else { 1.0 });
                y
            }
            _ => {
                if collision {
                    slow_shares.push(1.0);
                }
                window
            }
        };
        let up = rate * years;
        total[k] = (up * f64::from(vd_terrain::solve::Z_STEPS_PER_M)) as i64;
        start_yr[k] = age_yr.saturating_sub(window as u64);
        let area = lattice.area_m2(k as u32) as f64;
        report.nodes += 1;
        if collision {
            report.collision_nodes += 1;
            report.volume_m3[0] += up * area;
            kind[k] = BELT_COLLISION;
        } else {
            report.andean_nodes += 1;
            report.volume_m3[1] += up * area;
            kind[k] = BELT_ANDEAN;
        }
        report.max_total_m = report.max_total_m.max(up);
        sum_total += up;
        windows.push(window / 1e6);
        vns.push(v_n * 1_000.0);
        rates.push(rate * 1_000.0);
    }
    for list in [
        &mut windows,
        &mut vns,
        &mut rates,
        &mut kept_shares,
        &mut slow_shares,
    ] {
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
    report.kept_share = [0.1, 0.5, 0.9].map(|q| quantile(&kept_shares, q));
    report.slowing_share = [0.1, 0.5, 0.9].map(|q| quantile(&slow_shares, q));
    Belts {
        total,
        start_yr,
        report,
        pair,
        ceiling,
        pairs,
        kind,
        tau_yr,
    }
}

/// The share of a SLOWING collision's window `[start, end)` that falls in `[t0, t1)` (S2), the influx falling as
/// `e^(−(t − start)/τ)`: `cum(t) = ⌊total · (1 − e^(−(t − start)/τ)) / (1 − e^(−(end − start)/τ))⌋`, the step
/// `cum(t1) − cum(t0)`; `cum(end)` is `total` exactly, so the window closes its whole budget.
fn slowing_step(total: i64, start: u64, end: u64, tau_yr: f64, t0: u64, t1: u64) -> i64 {
    if end <= start {
        return 0;
    }
    let whole = -(-((end - start) as f64) / tau_yr).exp_m1();
    let cum = |t: u64| -> i64 {
        let inside = (t.clamp(start, end) - start) as f64;
        let part = -(-inside / tau_yr).exp_m1();
        (total as f64 * (part / whole)).floor() as i64
    };
    cum(t1) - cum(t0)
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

/// ★ F2's SWEEP: the shipped sweep's own order and bookkeeping (every node's uplift share first; then the cut, outlets
/// first, a lake node skipped; then the deposit), with the joint law's implicit cut: a node `x` over its base `b` after
/// the uplift, `x₀`, settles where `x − x₀ + Δt · K_lp · share · Q^m · (x/L)^n = 0` (heights in metres), the root found by
/// Newton from `x₀` (the function is convex and rising, so the steps fall to the root without passing it).
fn joint_sweep(state: &mut MacroSolve, dt_yr: f64, pass: u32, passes: u32) {
    let steps_m = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
    let basement = f64::from(vd_terrain::strata::Province::CrystallineBasement.erodibility_q8());
    for i in 0..state.node_count() {
        state.z[i] += MacroSolve::uplift_step(state.uplift[i], pass, passes);
        state.cut_now[i] = 0;
    }
    for k in (0..state.order.len()).rev() {
        let node = state.order[k];
        let i = node as usize;
        let r = state.receiver[i];
        if r == vd_terrain::macro_lattice::NO_NODE || state.pit[i] > 0 {
            continue;
        }
        let base = crate::router_variants::routing_base(state, r);
        let z0 = state.z[i];
        if z0 <= base {
            continue;
        }
        let x0 = f64::from(z0 - base) / steps_m;
        let length = f64::from(state.chord[i].max(1));
        let discharge_m3_yr = state.discharge[i] as f64 / 1_000.0;
        let share = f64::from(state.erodibility_q8[i]) / basement;
        let c = dt_yr * JOINT_K_LP * share * discharge_m3_yr.powf(JOINT_M);
        let mut x = x0;
        for _ in 0..JOINT_NEWTON {
            let s = x / length;
            let g = x - x0 + c * s.powf(JOINT_N);
            let dg = 1.0 + c * JOINT_N * s.powf(JOINT_N - 1.0) / length;
            let next = (x - g / dg).max(0.0);
            let moved = x - next;
            x = next;
            if moved.abs() < 1e-6 {
                break;
            }
        }
        let cut = z0 - (base + (x * steps_m).floor() as i32);
        if cut > 0 {
            state.z[i] -= cut;
            state.removed[i] = state.removed[i].saturating_add(cut as u32);
            state.pending[i] = state.pending[i].saturating_add(cut);
            state.cut_now[i] = cut as u32;
        }
    }
    let _ = state.deposit();
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

/// ★ THE PASS LEDGER (ruling B41 item 1, the gate's cause): what each process did to the heights in one pass, as a change
/// of volume over the land and over the sea floor (m³, a lowering negative; a node's side is read before the process:
/// land where it stands over the sea's level), and the land's readings after the pass. A readout only: no law reads it.
///
/// **Example.** If the land over 1 000 m parts between the 214- and the 414-pass axes only in the last 2.6 Myr, and the
/// ice's column carries the difference, the ice's cut is the cause; if it parts over the whole age in the cut's column,
/// the rivers' time step is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PassLedger {
    /// The pass's start and end, years from the solve's start.
    pub t0_yr: u64,
    pub t1_yr: u64,
    /// Each process's change of volume, `[land, sea floor]`, m³.
    pub craters_m3: [f64; 2],
    pub talus_m3: [f64; 2],
    pub ice_m3: [f64; 2],
    /// The sweep's net change: the uplift, less the cut, plus the deposit's fill.
    pub sweep_m3: [f64; 2],
    /// The sweep's two known parts: the uplift it adds and the cut it removes (a positive volume).
    pub uplift_m3: [f64; 2],
    pub cut_m3: [f64; 2],
    pub rebound_m3: [f64; 2],
    pub peak_m3: [f64; 2],
    /// After the pass: the sea's level (sixteenths), the land's mean height over it (m, by area), the land's share over
    /// 1 000 m (by area), and the land's share of the globe.
    pub sea_z: i32,
    pub land_mean_m: f64,
    pub land_over_1km: f64,
    pub land_share: f64,
}

/// The change of volume from `before` to `after`, m³, `[land, sea floor]` by each node's side before.
fn volume_change(before: &[i32], after: &[i32], area: &[u64], sea_z: i32) -> [f64; 2] {
    let mut v = [0.0f64; 2];
    for ((&b, &a), &ar) in before.iter().zip(after).zip(area) {
        v[usize::from(b <= sea_z)] += f64::from(a - b) * ar as f64;
    }
    let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
    [v[0] / steps, v[1] / steps]
}

/// A per-node volume in sixteenths times the area, m³, `[land, sea floor]` by each node's side.
fn volume_of(per_node: impl Iterator<Item = i64>, z: &[i32], area: &[u64], sea_z: i32) -> [f64; 2] {
    let mut v = [0.0f64; 2];
    for ((d, &zz), &ar) in per_node.zip(z).zip(area) {
        v[usize::from(zz <= sea_z)] += d as f64 * ar as f64;
    }
    let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
    [v[0] / steps, v[1] / steps]
}

/// The land's readings over the sea's level: its mean height (m, by area), its share over 1 000 m, its share of the globe.
fn land_readings(z: &[i32], area: &[u64], sea_z: i32) -> (f64, f64, f64) {
    let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
    let over = i64::from(1_000 * vd_terrain::solve::Z_STEPS_PER_M);
    let (mut land, mut all, mut sum, mut high) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&z, &a) in z.iter().zip(area) {
        let a = a as f64;
        all += a;
        let h = i64::from(z) - i64::from(sea_z);
        if h > 0 {
            land += a;
            sum += h as f64 / steps * a;
            if h > over {
                high += a;
            }
        }
    }
    (
        sum / land.max(1.0),
        high / land.max(1.0),
        land / all.max(1.0),
    )
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
    /// The pass ledger, one entry a pass, where the caller asked for it (empty otherwise).
    pub ledger: Vec<PassLedger>,
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
    ledger_on: bool,
) -> Option<Solved> {
    let mut ledger: Vec<PassLedger> = Vec::new();
    let lattice = body.macro_lattice()?;
    let mut land = vd_terrain::land::initial_land(body, &lattice, &words.land());
    // LAW (L1, L2, L4): the belts by their influx replace the room's belts; their nodes read the folded belt's rock.
    let belts = laws
        .belts_by_influx
        .then(|| belts_by_influx(body, words, &land, schedule.age_yr, laws));
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
    let epoch_left = if laws.geometric_last_yr.is_some() || laws.mixed_axis.is_some() {
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
    // LAW (L5): the passes' boundaries in years: even, geometric, or the mixed axis (ruling B39's gate).
    let times = match laws.mixed_axis {
        Some((t_min, t_max)) => mixed_times(schedule.age_yr, t_min, t_max),
        None => pass_times(schedule.age_yr, schedule.passes, laws.geometric_last_yr),
    };
    assert_eq!(
        times.len(),
        schedule.passes as usize + 1,
        "the schedule states the axis's own pass count"
    );
    let axis_varies = laws.geometric_last_yr.is_some() || laws.mixed_axis.is_some();
    let epoch_start = schedule.age_yr.saturating_sub(GLACIAL_EPOCH_YR);
    for pass in 0..schedule.passes {
        let gain = if axis_varies {
            gain_for(
                times[pass as usize + 1] - times[pass as usize],
                schedule.k0_per_yr,
            )
        } else {
            gain
        };
        let epoch_step = if axis_varies {
            overlap(
                epoch_start,
                schedule.age_yr,
                times[pass as usize],
                times[pass as usize + 1],
            )
        } else {
            epoch_step
        };
        let mut entry = PassLedger {
            t0_yr: times[pass as usize],
            t1_yr: times[pass as usize + 1],
            ..PassLedger::default()
        };
        // LAW (B37, the craters' own time): the craters whose impact falls in this pass are stamped now.
        if laws.craters_at_own_age {
            let before = ledger_on.then(|| state.z.clone());
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
            if let Some(b) = before {
                entry.craters_m3 = volume_change(&b, &state.z, &state.area, state.sea_z);
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
                let before = ledger_on.then(|| state.z.clone());
                let tan: Vec<u32> = c.aridity_q8.iter().map(|&a| tan_repose_q16(a)).collect();
                let _ = state.talus(&tan);
                talus_done += 1;
                if let Some(b) = before {
                    entry.talus_m3 = volume_change(&b, &state.z, &state.area, state.sea_z);
                }
            }
            let line: &[i32] = if words.water_km3 == 0 {
                &no_ice
            } else {
                &c.ela_z
            };
            let before = ledger_on.then(|| state.z.clone());
            let _ = state.ice(line, &c.temperature_dk, c.lapse_mk_km, gravity, epoch_step);
            if let Some(b) = before {
                entry.ice_m3 = volume_change(&b, &state.z, &state.area, state.sea_z);
            }
        }
        // LAW (L1, L2, L4): each node's uplift for THIS pass, handed to the shipped sweep through its own word — the
        // sweep adds `uplift_step(total, pass, passes)`, so a total of `step · passes` adds exactly `step` (ARITH: the
        // cumulative floors of an exact multiple). A belt node takes its window's share of the pass; every other node
        // takes the solve's own even share of its total.
        // LAW (L5): on the geometric axis every node's even total spreads by TIME, not by the pass's index.
        let (t0, t1) = (times[pass as usize], times[pass as usize + 1]);
        if belts.is_some() || axis_varies {
            let mut steps: Vec<i64> = even_total
                .iter()
                .enumerate()
                .map(|(i, &even)| {
                    let belt_total = belts.as_ref().map_or(0, |b| b.total[i]);
                    if belt_total > 0 {
                        let b = belts.as_ref().expect("a belt total has its belts");
                        // LAW (S2): a slowing collision's budget follows its falling speed.
                        match b.tau_yr {
                            Some(tau) if b.kind[i] == BELT_COLLISION => slowing_step(
                                belt_total,
                                b.start_yr[i],
                                schedule.age_yr,
                                tau,
                                t0,
                                t1,
                            ),
                            _ => window_step(belt_total, b.start_yr[i], schedule.age_yr, t0, t1),
                        }
                    } else if axis_varies {
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
        // The ledger: the uplift the sweep adds this pass, on each node's side before the sweep.
        let before = ledger_on.then(|| {
            entry.uplift_m3 = volume_of(
                state
                    .uplift
                    .iter()
                    .map(|&u| i64::from(MacroSolve::uplift_step(u, pass, schedule.passes))),
                &state.z,
                &state.area,
                state.sea_z,
            );
            state.z.clone()
        });
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
                // LAW (F2): the joint law's sweep, or the solve's own.
                if laws.joint_law {
                    joint_sweep(
                        &mut state,
                        (times[pass as usize + 1] - times[pass as usize]) as f64,
                        pass,
                        schedule.passes,
                    );
                } else if laws.sub_sweeps > 1 {
                    // RESOLUTION (the gate's cause): k sweeps on this pass's routing, each a k-th of the gain.
                    let k = laws.sub_sweeps;
                    let uplift = std::mem::take(&mut state.uplift);
                    let mut cut_sum = vec![0u32; state.node_count()];
                    for sub in 0..k {
                        state.uplift = if sub == 0 {
                            uplift.clone()
                        } else {
                            vec![0; uplift.len()]
                        };
                        if sub > 0 && laws.sub_route {
                            routes.push(state.route());
                            state.accumulate();
                        }
                        let _ = state.sweep(gain / u64::from(k), pass, schedule.passes);
                        for (sum, &c) in cut_sum.iter_mut().zip(&state.cut_now) {
                            *sum = sum.saturating_add(c);
                        }
                    }
                    state.uplift = uplift;
                    state.cut_now = cut_sum;
                } else {
                    let _ = state.sweep(gain, pass, schedule.passes);
                }
            }
        }
        if let Some(b) = before {
            entry.sweep_m3 = volume_change(&b, &state.z, &state.area, state.sea_z);
            entry.cut_m3 = volume_of(
                state.cut_now.iter().map(|&c| i64::from(c)),
                &b,
                &state.area,
                state.sea_z,
            );
        }
        since_rebound += 1;
        if since_rebound >= schedule.isostasy_every {
            let before = ledger_on.then(|| state.z.clone());
            let _ = state.rebound(alpha_m);
            since_rebound = 0;
            if let Some(b) = before {
                entry.rebound_m3 = volume_change(&b, &state.z, &state.area, state.sea_z);
            }
        }
        // LAW (L0): the peak law after every pass, on the height that stands.
        if laws.limits_as_law {
            let before = ledger_on.then(|| state.z.clone());
            let (c, left) = peak_law(&mut state, peak_cap);
            collapsed += c;
            peak_left = peak_left.max(left);
            if let Some(b) = before {
                entry.peak_m3 = volume_change(&b, &state.z, &state.area, state.sea_z);
            }
        }
        if ledger_on {
            let (mean, high, share) = land_readings(&state.z, &state.area, state.sea_z);
            entry.sea_z = state.sea_z;
            entry.land_mean_m = mean;
            entry.land_over_1km = high;
            entry.land_share = share;
            ledger.push(entry);
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
        ledger,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The home system's age, years (the axes' own sum).
    const AGE: u64 = 5_000_000_000;

    /// The mixed axis and its twice-as-fine axis sum to the age exactly, start at zero, and run oldest first.
    #[test]
    fn the_mixed_axes_cover_the_age() {
        let mixed = mixed_times(AGE, 500_000, 25_000_000);
        let fine = mixed_times(AGE, 250_000, 12_500_000);
        assert_eq!((mixed[0], mixed[mixed.len() - 1]), (0, AGE));
        assert_eq!((fine[0], fine[fine.len() - 1]), (0, AGE));
        assert_eq!((mixed.len() - 1, fine.len() - 1), (214, 414));
        assert_eq!(mixed[mixed.len() - 1] - mixed[mixed.len() - 2], 500_000);
        assert_eq!(fine[fine.len() - 1] - fine[fine.len() - 2], 250_000);
    }

    /// S2's time constant is Clark's 45.1 Myr, and a slowing collision's passes close its whole budget, the early
    /// passes more than the late ones; a constant-speed window's passes close it too.
    #[test]
    fn a_slowing_collision_closes_its_whole_budget_early_first() {
        let tau = 1.0 / (TIBET_STRAIN_RATE_PER_S * SECONDS_PER_YEAR);
        assert_eq!((tau / 1e5).round(), 451.0);
        let start = AGE - 133_000_000;
        let total = 1_000_000i64;
        let times = mixed_times(AGE, 500_000, 25_000_000);
        let slow: Vec<i64> = times
            .windows(2)
            .map(|w| slowing_step(total, start, AGE, tau, w[0], w[1]))
            .collect();
        let even: Vec<i64> = times
            .windows(2)
            .map(|w| window_step(total, start, AGE, w[0], w[1]))
            .collect();
        assert_eq!(slow.iter().sum::<i64>(), total);
        assert_eq!(even.iter().sum::<i64>(), total);
        // The window's first half closes `(1 − e^(−66.5/45.1))/(1 − e^(−133/45.1))` = 0.81 of the budget (ARITH); a
        // constant speed closes half in each half.
        let mid = start + 66_500_000;
        let early = slowing_step(total, start, AGE, tau, start, mid);
        assert_eq!((early as f64 / total as f64 * 100.0).round(), 81.0);
        assert_eq!(
            window_step(total, start, AGE, start, mid),
            window_step(total, start, AGE, mid, AGE)
        );
        // Before the window nothing; an empty window closes nothing.
        assert_eq!(slowing_step(total, start, AGE, tau, 0, start), 0);
        assert_eq!(slowing_step(total, AGE, AGE, tau, 0, AGE), 0);
    }
}
