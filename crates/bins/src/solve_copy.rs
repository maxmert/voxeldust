//! ★ THE ONE COPY OF THE MACRO SOLVE FOR THE BENCHES (rulings B37, B38; RE-BASED ON PHASE 2 of the ground step, rulings
//! B52–B53): a line-for-line copy of `vd_terrain::solve::solve_full` in which a bench may take ONE ruled law out at a
//! time — IN THE COPY ONLY. Nothing in the game reads this module: the game's solve is `vd_terrain::solve::solve_full`,
//! untouched (the owner, 2026-10-01: no test seams in the game; ruling B38: one bench, the binding rules unbroken).
//! THE CONTROL: with the default [`Laws`] — every law as the game runs it — the copy rebuilds the game's solve bit for
//! bit (the unit test `the_control_is_the_games_own_solve` on the home moon; the belts' bench's `phase2` case against
//! the release gate's pinned home artifact), or nothing a switch prints can be trusted.
//!
//! The copy calls the game's own law functions — the belts ([`vd_terrain::belts::belts`]), the joint law's sweep, the
//! stepped ice, the exact rebound, the landscape's time, the water's own paths, the peak law — so a switch can only take
//! a law OUT, back to the form the game had before it; the bench's own forms of the laws (runs 9–30,
//! `docs/investigation/2026-10-02/phase0_belts_bench.txt`) and the washboard bench's routing variants (rulings B10–B12)
//! ran on the pre-phase-2 erosion and are retired with it (DEFERRED D-TERRAIN-10: the variants return on the joint law
//! when R2 measures D∞, ruling B12).
//!
//! **Example.** The belts' bench solves the home planet in the copy twice — once as the game does, once with the water's
//! own paths taken out — and reads how many of the plateau's rivers end in closed basins each time. Neither run changes
//! a byte of the world the game solves.

use vd_terrain::BodyDefinition;

/// ★ A CANDIDATE UNDER MEASUREMENT, NOT A RULED LAW (2026-10-13, ruling B94 step 2; the macro study's defect 1,
/// `docs/investigation/2026-10-11/macro_rain_rise_study.md`): the rain belt's factor renormalised so its area mean is one —
/// the belt then only moves the supply between latitudes and never removes it. Each band of the circulation is uniform in
/// the sine of the latitude, so the factor's area mean is `sinking + (rising − sinking)·8/15` (0.575) on any body. The
/// bench sets it (`belts_bench … +rain`); off, the copy is the game's. The rain is scaled after the game's climate, the
/// runoff and the aridity follow it; the ice's line does not (a probe, stated).
pub static RAIN_RENORMALISED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn rained(mut c: vd_terrain::climate::Climate) -> vd_terrain::climate::Climate {
    use vd_terrain::climate::{BELT_RISING_SHARE, BELT_SINKING_SHARE, aridity_q8, runoff_mm_yr};
    if !RAIN_RENORMALISED.load(std::sync::atomic::Ordering::Relaxed) {
        return c;
    }
    let mean = BELT_SINKING_SHARE + (BELT_RISING_SHARE - BELT_SINKING_SHARE) * 8.0 / 15.0;
    for i in 0..c.rain_mm_yr.len() {
        let rain = (f64::from(c.rain_mm_yr[i]) / mean).round() as u32;
        c.rain_mm_yr[i] = rain;
        let pet = Gf::from_i64(i64::from(c.pet_mm_yr[i]));
        c.runoff_mm_yr[i] = runoff_mm_yr(Gf::from_i64(i64::from(rain)), pet)
            .to_f64()
            .round()
            .max(0.0) as u32;
        let t_k = Gf::from_f64(f64::from(c.temperature_dk[i]) / 10.0);
        c.aridity_q8[i] = aridity_q8(t_k, rain);
    }
    c
}

/// The seasons' chain on `threads`, warm from this solve's `state`.
fn seasons_on(
    body: &BodyDefinition,
    lattice: &vd_terrain::macro_lattice::MacroLattice,
    words: &vd_terrain::solve::SolveWords,
    z: &[i32],
    sea: Option<vd_terrain::climate::SeaView<'_>>,
    threads: usize,
    state: &mut vd_terrain::seasons::chain::SeasonState,
) -> vd_terrain::climate::Climate {
    let clock = std::time::Instant::now();
    let c =
        vd_terrain::seasons::chain::seasonal_climate(body, lattice, words, z, sea, threads, state);
    if std::env::var_os("VD_BENCH_TIMES").is_some() {
        // The land's rain (the nodes off the connected ocean): its median and its share under 250 mm.
        let mut land: Vec<u32> = (0..c.rain_mm_yr.len())
            .filter(|&i| !sea.is_some_and(|s| s.ocean[i]))
            .map(|i| c.rain_mm_yr[i])
            .collect();
        land.sort_unstable();
        let median = land.get(land.len() / 2).copied().unwrap_or(0);
        let dry = land.iter().filter(|&&r| r < 250).count() as f64 / land.len().max(1) as f64;
        eprintln!(
            "SEASONS: a call in {:.1} s (the balance {} years, the mountains' worst rounds {}, the largest loop solved {}, left {}; the drag's rounds {} met {}; the spin-up {:?}); the land's rain median {median} mm, under 250 mm {:.1} %",
            clock.elapsed().as_secs_f64(),
            state.last_years,
            state.last_rounds,
            state.last_loop,
            state.last_left,
            state.last_drag,
            state.last_drag_met,
            state.last_spin,
            100.0 * dry
        );
    }
    c
}

/// ★ THE COPY'S ONE CLIMATE (the game's `vd_terrain::solve` climate source, ruling B101): with [`Laws::seasons`] in, a
/// body with air that is not locked to its star reads the seasons' chain, warm from call to call within this solve
/// (never shared between two solves); elsewhere — and with the law out, generator 26's climate — the temperature law of
/// `vd_terrain::climate`, then [`rained`].
struct CopyClimate {
    seasons: Option<vd_terrain::seasons::chain::SeasonState>,
}

impl CopyClimate {
    fn of(
        lattice: &vd_terrain::macro_lattice::MacroLattice,
        words: &vd_terrain::solve::SolveWords,
        laws: &Laws,
        cores: vd_terrain::par::Cores,
    ) -> CopyClimate {
        let air = words.p_surf_pa.is_some_and(|p| p > 0);
        let locked = words.flags & vd_terrain::solve::WORD_FLAG_TIDALLY_LOCKED != 0;
        CopyClimate {
            seasons: (laws.seasons && air && !locked)
                .then(|| vd_terrain::seasons::chain::SeasonState::new(lattice, cores.threads())),
        }
    }

    /// The passes between two climates: [`Laws::rain_every`] for the chain, every pass for the temperature law.
    fn every(&self, laws: &Laws) -> u32 {
        if self.seasons.is_some() {
            laws.rain_every
        } else {
            1
        }
    }

    /// The climate over `z` as it stands.
    fn at(
        &mut self,
        body: &BodyDefinition,
        lattice: &vd_terrain::macro_lattice::MacroLattice,
        words: &vd_terrain::solve::SolveWords,
        z: &[i32],
        sea: Option<vd_terrain::climate::SeaView<'_>>,
        cores: vd_terrain::par::Cores,
    ) -> vd_terrain::climate::Climate {
        match self.seasons.as_mut() {
            Some(state) => seasons_on(body, lattice, words, z, sea, cores.threads(), state),
            None => rained(vd_terrain::climate::climate_on(
                body, lattice, words, z, sea, cores,
            )),
        }
    }
}
use vd_terrain::Gf;
use vd_terrain::artifact::Artifact;
use vd_terrain::belts::{BELT_NONE, BeltReport, Belts};
use vd_terrain::solve::{
    AGE_GATE_MIN_NODES, FullReport, GLACIAL_EPOCH_YR, LAKE_SEA_SOLVES, MacroSolve, Schedule,
    SolveWords, SweepReport, Z_STEPS_PER_M, active_share, belt_land, coast_band, erosion_starts,
    pass_uplift, raise_times, settle_lakes_sea, tan_repose_q16,
};

/// ★ THE CRITICAL TAPER's calibration body, TAIWAN (Davis, Suppe & Dahlen 1983, JGR 88): the wedge's internal friction
/// `μ`, its basal friction `μ_b`, the pore-pressure ratio `λ = λ_b`, and the basal dip `β` (rad).
pub const TAPER_MU: f64 = 1.03;
pub const TAPER_MU_B: f64 = 0.85;
pub const TAPER_LAMBDA: f64 = 0.7;
pub const TAPER_BETA_RAD: f64 = 6.0 * std::f64::consts::PI / 180.0;
/// ★ THE BACK (RETRO) FLANK's slope, rad: Taiwan's measured retro flank, 6° (Roe et al. 2008, p. 823).
pub const RETRO_TAPER_RAD: f64 = 6.0 * std::f64::consts::PI / 180.0;

/// ★ THE CRITICAL TAPER's front slope from its frictions, rad (Davis, Suppe & Dahlen 1983; in the form of Mouthereau et
/// al. 2006, Eq. 1): `α + β = [β + μ_b(1 − λ_b)] / [1 + (1 − λ)·2 sin φ/(1 − sin φ)]`, `φ = atan μ`. The game states its
/// value at Taiwan's numbers as a tangent ([`vd_terrain::belts::PRO_TAPER_TAN`]), because its fence offers no tangent;
/// the test `the_critical_taper_at_taiwan` checks that constant against this formula.
#[must_use]
pub fn critical_taper_rad(mu: f64, mu_b: f64, lambda: f64, beta: f64) -> f64 {
    let sin_phi = mu.atan().sin();
    (beta + mu_b * (1.0 - lambda)) / (1.0 + (1.0 - lambda) * 2.0 * sin_phi / (1.0 - sin_phi)) - beta
}

/// ★ THE LAWS THE COPY MAY TAKE OUT, one at a time (rulings B38, B52–B53). The default is the game's own: every law in.
/// A law taken out falls back to the form the game had before it — never to a knob. ★ A RUN WITH A LAW OUT IS A
/// MEASUREMENT OF THAT LAW'S OWN EFFECT, NEVER A CANDIDATE (ruling B38 item 4): the forms it falls back to were retired
/// for cause (the room's belts are a cap, ruling B37 item 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Laws {
    /// ★ THE BELTS AS CRITICAL WEDGES (rulings B49–B51: the wedge, the crust's partition, the collision's slowing, one
    /// level a pair). Out: no wedge rises, and every node keeps the land's own uplift — the belts by the room
    /// (`land.rs`), evenly over the age. The plates' turnover stays the law's, so the landscape's time still reads it.
    pub wedge: bool,
    /// ★ THE LANDSCAPE'S TIME (rulings B45–B46). Out: every plate is as old as the system, so every node erodes for the
    /// whole age and no crater is erased by a raise.
    pub landscape_time: bool,
    /// ★ THE CRATERS AT THEIR OWN AGE (ruling B37). Out: every crater of the record is stamped before the first pass.
    pub craters_at_own_age: bool,
    /// ★ THE WATER'S OWN PATHS (ruling B51 item 2). Out: the sweep reads the routing's own receivers, its discharge is
    /// the rain summed down the routing's tree, and the deposit fills the routing's own hollows.
    pub water_paths: bool,
    /// ★ A CANDIDATE UNDER MEASUREMENT, NOT A RULED LAW (the convergence gate's cause, 2026-10-05; off by default, so the
    /// control stays the game's): a full lake passes the rain of its catchment in the SHARE of its water that leaves it
    /// — the budget's own water passed over its spill over the water it held and passed — where the game passes all of
    /// it (B51 item 2 as built: the sinks cleared). A closed lake passes nothing either way, so at the threshold the
    /// row below a hollow moves smoothly from nothing instead of jumping to the whole catchment's rain.
    pub pass_share: bool,
    /// ★ A CANDIDATE UNDER MEASUREMENT, NOT A RULED LAW (ruling B54: the cure first; off by default): the water's own
    /// tree wherever a hollow stands — its lakes, its closed bodies, its spills — and the ROUTING's own receivers (its
    /// D8 and phase 1's two-field flats) on the land outside every hollow, where the game's tree reads the ground's
    /// descent with the budget's own flat potential. Measures whether the young plateaus' capture speed is the flats'
    /// rule.
    pub routing_flats: bool,
    /// ★ THE CLIMATE'S SPACING — A RESOLUTION, NOT A LAW (ruling B68 measured with it; the bench's `^C`): the climate
    /// and the running sea re-solved every this many passes. The game re-solves both every pass (ruling B69 item 1),
    /// so the default is one; a larger spacing measures the error the game's twenty passes carried until generator 26.
    pub climate_every: u32,
    /// ★ THE CLIMATE'S OWN SPACING — A RESOLUTION, NOT A LAW (ruling B101; the bench's `+seasonsN`): the seasons' chain
    /// re-solved every this many passes, the running sea still every pass (ruling B68 found the sea's stale spacing, not
    /// the climate's, made the jump near the present). The game's [`vd_terrain::solve::SEASONS_EVERY`] by default; the
    /// temperature law is re-solved every pass whatever it states.
    pub rain_every: u32,
    /// ★ THE SEASONS' CLIMATE (rulings B99–B102, generator 27): a body with air that is not locked to its star reads the
    /// seasons' chain every [`Laws::rain_every`] passes, as the game does. Out: generator 26's climate, every pass.
    pub seasons: bool,
    /// ★ THE INITIAL LAND (rulings B58 item 2, B59–B61; [`crate::land_copy`]): the game's own by default (generator
    /// 26), or the bench land copy with a law taken out or a diagnostic in.
    pub land: crate::land_copy::LandLaws,
}

impl Default for Laws {
    fn default() -> Laws {
        Laws {
            wedge: true,
            landscape_time: true,
            craters_at_own_age: true,
            water_paths: true,
            pass_share: false,
            routing_flats: false,
            climate_every: 1,
            rain_every: vd_terrain::solve::SEASONS_EVERY as u32,
            seasons: true,
            land: crate::land_copy::LandLaws::default(),
        }
    }
}

/// The receivers' order, leaves first (each node before its receiver), for a tree the copy re-wires; a node on a
/// ring of receivers is left out.
fn leaves_first(receiver: &[u32]) -> Vec<u32> {
    use vd_terrain::macro_lattice::NO_NODE;
    let n = receiver.len();
    let mut donors = vec![0u32; n];
    for &r in receiver {
        if r != NO_NODE {
            donors[r as usize] += 1;
        }
    }
    let mut order: Vec<u32> = (0..n as u32).filter(|&i| donors[i as usize] == 0).collect();
    let mut k = 0;
    while k < order.len() {
        let r = receiver[order[k] as usize];
        if r != NO_NODE {
            donors[r as usize] -= 1;
            if donors[r as usize] == 0 {
                order.push(r);
            }
        }
        k += 1;
    }
    order
}

/// ★ THE CANDIDATE [`Laws::routing_flats`]: the routing's own receivers and chords on the land outside every hollow
/// (the budget's home the ocean's or none), the order rebuilt, and the discharge the rain on each node's area summed
/// down the tree with no sink, in the game's own units (mm·m²/yr). ★ NO RING: two rules can point at each other (open
/// land the routing leads into a hollow whose water leads back out; two flat rules on one flat), so a node the order
/// leaves out takes the water's own receiver back, until no node is left out. Returns the nodes so repaired.
fn routing_on_open_land(
    state: &mut MacroSolve,
    lakes: &vd_terrain::lakes::Lakes,
    routed: &[u32],
    routed_chord: &[u32],
) -> usize {
    use vd_terrain::lakes::{NO_DEP, OCEAN};
    let water: Vec<u32> = state.receiver.clone();
    let water_chord: Vec<u32> = state.chord.clone();
    // Open land: the budget's home is the ocean's or none, and the node is no sea. An edge from open land INTO a hollow
    // stays the water's own, so the routing can never lead open land into a hollow whose water leads back out to it:
    // the two rules then differ only where they walk the same ground — on the flats.
    let open = |v: u32| {
        let i = v as usize;
        ((lakes.home[i] == NO_DEP) | (lakes.home[i] == OCEAN)) & !state.ocean[i]
    };
    for i in 0..state.z.len() {
        let r = routed[i];
        if open(i as u32) && (r != vd_terrain::macro_lattice::NO_NODE) && open(r) {
            state.receiver[i] = r;
            state.chord[i] = routed_chord[i];
        }
    }
    let mut repaired = 0usize;
    loop {
        state.order = leaves_first(&state.receiver);
        if state.order.len() == state.z.len() {
            break;
        }
        let mut seen = vec![false; state.z.len()];
        for &v in &state.order {
            seen[v as usize] = true;
        }
        let mut moved = 0usize;
        for (i, &kept) in seen.iter().enumerate() {
            if !kept && (state.receiver[i] != water[i]) {
                state.receiver[i] = water[i];
                state.chord[i] = water_chord[i];
                moved += 1;
            }
        }
        repaired += moved;
        if moved == 0 {
            break;
        }
    }
    let mut q: Vec<u128> = state
        .rain
        .iter()
        .zip(&state.area)
        .map(|(&r, &a)| u128::from(r) * u128::from(a))
        .collect();
    for k in 0..state.order.len() {
        let i = state.order[k] as usize;
        let r = state.receiver[i];
        if r != vd_terrain::macro_lattice::NO_NODE {
            q[r as usize] += q[i];
        }
    }
    for (d, &v) in state.discharge.iter_mut().zip(&q) {
        *d = u64::try_from(v).unwrap_or(u64::MAX);
    }
    repaired
}

/// ★ THE CANDIDATE [`Laws::pass_share`]: the discharge rebuilt over the water's own tree the game's
/// `follow_the_budget` left (its receivers and order), each node's rain on its area summed down it, and at every full
/// lake's spill node only the share of the water that leaves (Σ passed over Σ held and passed of the full bodies that
/// spill there, the budget's own counts). Units as the game's row: mm·m²/yr.
fn pass_share_discharge(state: &mut MacroSolve, lakes: &vd_terrain::lakes::Lakes) {
    use std::collections::BTreeMap;
    let mut share: BTreeMap<u32, (u128, u128)> = BTreeMap::new();
    for (d, dep) in lakes.deps.iter().enumerate().skip(1) {
        if (lakes.body_of[d] == d as u32)
            & dep.full
            & (dep.spill_node != vd_terrain::macro_lattice::NO_NODE)
        {
            let e = share.entry(dep.spill_node).or_default();
            e.0 += dep.passed;
            e.1 += dep.held + dep.passed;
        }
    }
    let mut q: Vec<u128> = state
        .rain
        .iter()
        .zip(&state.area)
        .map(|(&r, &a)| u128::from(r) * u128::from(a))
        .collect();
    for k in 0..state.order.len() {
        let i = state.order[k] as usize;
        if let Some(&(passed, inflow)) = share.get(&(i as u32)) {
            q[i] = if inflow == 0 {
                q[i]
            } else {
                q[i] * passed / inflow
            };
        }
        let r = state.receiver[i];
        if r != vd_terrain::macro_lattice::NO_NODE {
            q[r as usize] += q[i];
        }
    }
    for (d, &v) in state.discharge.iter_mut().zip(&q) {
        *d = u64::try_from(v).unwrap_or(u64::MAX);
    }
}

/// ★ ONE PASS OF THE LEDGER (ruling B41 item 1, the gate's cause): what each process did to the land and the sea floor,
/// and the land as the pass leaves it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PassLedger {
    /// The pass's start and end, years from the solve's start.
    pub t0_yr: u64,
    pub t1_yr: u64,
    /// Each process's change of volume, `[land, sea floor]`, m³ (the side by the sea's level before the pass).
    pub craters_m3: [f64; 2],
    pub talus_m3: [f64; 2],
    pub ice_m3: [f64; 2],
    pub uplift_m3: [f64; 2],
    /// The sweeps' net change: the cut less the deposit's fill (a negative volume where the land wears down).
    pub sweep_m3: [f64; 2],
    pub rebound_m3: [f64; 2],
    pub peak_m3: [f64; 2],
    /// The stepped ice's steps in the pass.
    pub ice_steps: u32,
    /// After the pass: the sea's level (sixteenths), the land's mean height over it (m, by area), the land's share over
    /// 1 000 m (by area), and the land's share of the globe.
    pub sea_z: i32,
    pub land_mean_m: f64,
    pub land_over_1km: f64,
    pub land_share: f64,
    /// ★ THE ROCK'S WAY over the pass's sweeps (the gate's cause, ruling B39), m³: the cut; the rises the deposit laid
    /// OUTSIDE the closed hollows (a spilling lake's room) and INSIDE them (a lake's room and the level fill), both read
    /// on the ground; and the game's own readings of the rest ([`vd_terrain::solve::SweepReport::rock_to_sea`]) — the
    /// rock that reached the SEA, the rock a full hollow passed OVER its spill, the rock DROPPED (an inland end no body
    /// holds, a ring of receivers, a body with no way: none is expected), and the level fill's REMAINDER.
    pub rock_cut_m3: f64,
    pub rock_laid_open_m3: f64,
    pub rock_laid_closed_m3: f64,
    pub rock_over_spill_m3: f64,
    pub rock_sea_m3: f64,
    pub rock_dropped_m3: f64,
    pub rock_remainder_m3: f64,
    /// The instrument's own check, sixteenths·m²: the cut less the rises, the sea, the dropped rock and the remainder —
    /// zero, or rock went somewhere the deposit does not say.
    pub rock_check: i128,
    /// ★ THE CLOSED HOLLOWS at the pass's routing: their count, their area (m²), and the area that changed sides since
    /// the routing before — opened (closed then, open now) and closed (open then, closed now).
    pub closed_bodies: u32,
    pub closed_m2: f64,
    pub opened_m2: f64,
    pub newly_closed_m2: f64,
    /// ★ THE LAND'S TEXTURE (the convergence gate's cause): at a pass that crosses a whole [`TEXTURE_EVERY_YR`], the
    /// median of Hammond's local relief over the land (m) and the plains' share of the land (half or more of the window
    /// gentle, under 90 m of relief; `stands::relief_readings`); `-1` at the other passes.
    pub relief_p50_m: f64,
    pub plains_pct: f64,
}

/// The ledger reads the land's texture once every this many years ([`PassLedger::relief_p50_m`]).
pub const TEXTURE_EVERY_YR: u64 = 250_000_000;

/// The land's texture now: the median local relief (m) and the plains' share of the land (%), by the stands' own rule.
fn texture(state: &MacroSolve) -> (f64, f64) {
    let z_m: Vec<i32> = state.z.iter().map(|&z| z / Z_STEPS_PER_M).collect();
    let land: Vec<bool> = state.z.iter().map(|&z| z > state.sea_z).collect();
    let readings = crate::stands::relief_readings(&state.lattice, &z_m, &land);
    let mut spans: Vec<i32> = readings.iter().flatten().map(|r| r.span_m).collect();
    let plains = readings
        .iter()
        .flatten()
        .filter(|r| {
            (u32::from(r.gentle) * 2 >= u32::from(r.window))
                & (r.span_m < crate::stands::LOW_RELIEF_M)
        })
        .count();
    let count = spans.len().max(1);
    spans.sort_unstable();
    let p50 = spans.get(spans.len() / 2).copied().unwrap_or(0);
    (f64::from(p50), 100.0 * plains as f64 / count as f64)
}

/// ★ THE ROCK'S WAY of one sweep (see [`PassLedger::rock_cut_m3`]), read after the game's own sweep from its own state
/// and report: the sweep's cut per node (`cut_now`), the rises since the ground before it (`z_pre`) inside and outside
/// the closed hollows, and the deposit's own readings. Returns `[cut, laid open, laid closed]` in m³ and the check.
fn rock_way(
    state: &MacroSolve,
    z_pre: &[i32],
    slot: Option<&[u32]>,
    r: &SweepReport,
) -> ([f64; 3], i128) {
    let (mut cut, mut laid_open, mut laid_closed) = (0u128, 0u128, 0u128);
    for i in 0..state.z.len() {
        let area = u128::from(state.area[i].max(1));
        cut += u128::from(state.cut_now[i]) * area;
        let rise = i64::from(state.z[i]) - i64::from(z_pre[i]) + i64::from(state.cut_now[i]);
        let laid = u128::from(rise.max(0) as u64) * area;
        if slot.is_some_and(|s| s[i] != u32::MAX) {
            laid_closed += laid;
        } else {
            laid_open += laid;
        }
    }
    let m3 = |x: u128| x as f64 / f64::from(Z_STEPS_PER_M);
    let check = cut as i128
        - (laid_open + laid_closed + r.rock_to_sea + r.rock_dropped + r.rock_remainder) as i128;
    ([m3(cut), m3(laid_open), m3(laid_closed)], check)
}

/// What the copy solved: the state, the facies and the game's own report, and the bench's readings beside them.
#[derive(Clone, Debug)]
pub struct Solved {
    pub state: MacroSolve,
    pub facies: Vec<u8>,
    /// The climate the artifact ships: the solve's own last reading over its final ground (ruling B101).
    pub climate: vd_terrain::climate::Climate,
    pub report: FullReport,
    /// The belts the wedges raised, where the law is in (`None` where [`Laws::wedge`] took it out).
    pub belts: Option<Belts>,
    /// The craters the solve stamped — never one a raise erased.
    pub craters: Vec<vd_terrain::craters::Crater>,
    /// The peak law's collapses over the passes and their rounds, and the largest excess over its cap left after any
    /// pass, sixteenths (zero: it settled every time).
    pub collapsed: usize,
    pub peak_left: i32,
    /// The pass ledger, one entry a pass, where the caller asked for it (empty otherwise).
    pub ledger: Vec<PassLedger>,
    /// The ledger by plate (rows by plate, columns by process, m³ summed over the solve), where the ledger is on.
    pub by_plate: Vec<[f64; 7]>,
    /// The ledger by plate and pass ([`PlatePass`]), where the ledger is on.
    pub plate_passes: Vec<PlatePass>,
    /// The landscape's time: each plate's raise time, years before the present, and each node's plate.
    pub raise_yr: Vec<u64>,
    pub plate_of: Vec<u8>,
    /// Every node that stands on a wedge's plateau at the end (empty where the wedge is out).
    pub widened: Vec<bool>,
    /// The candidate [`Laws::routing_flats`]'s ring repairs summed over the passes (zero where it is off).
    pub flats_repaired: u64,
}

/// ★ THE LEDGER BY PLATE AND PASS (ruling B66 item 1: which process, where, WHEN): one row a plate a pass — each
/// process's volume change on that plate in that pass (m³, the columns of [`Solved::by_plate`]), and the plate's land
/// after the pass (land: no connected sea over it): its area (m²) and mean height over the sea (m); the same of its CALM
/// land — over the sea's level, and never moved by a margin, a ridge, a rift or a belt (the old plains' standing part:
/// the end's causes also take out the craters' reach and the closed basins' walls); and the area of its land at or under
/// the sea's level (the dry closed basins).
#[derive(Clone, Copy, Debug, Default)]
pub struct PlatePass {
    pub t1_yr: u64,
    pub plate: u8,
    pub process_m3: [f64; 7],
    pub land_m2: f64,
    pub land_mean_m: f64,
    pub calm_m2: f64,
    pub calm_mean_m: f64,
    pub under_m2: f64,
}

/// Each plate's land after a pass (see [`PlatePass`]): `[land m², land height·m², calm m², calm height·m², under m²]`.
fn plate_land(state: &MacroSolve, plate: &[u8], calm: &[bool], plates: usize) -> Vec<[f64; 5]> {
    let mut rows = vec![[0.0f64; 5]; plates];
    let sea_m = f64::from(state.sea_z) / f64::from(Z_STEPS_PER_M);
    for k in 0..state.z.len() {
        if state.ocean[k] {
            continue;
        }
        let a = state.area[k] as f64;
        let over = f64::from(state.z[k]) / f64::from(Z_STEPS_PER_M) - sea_m;
        let row = &mut rows[usize::from(plate[k])];
        row[0] += a;
        row[1] += over * a;
        if state.z[k] > state.sea_z {
            if calm[k] {
                row[2] += a;
                row[3] += over * a;
            }
        } else {
            row[4] += a;
        }
    }
    rows
}

/// ★ THE LEDGER BY PLATE (ruling B61 item 2: which process carries a plate's difference between two axes): each node's
/// volume change from `before` to `after`, m³, added to its plate's row in the column `proc` (0 craters, 1 talus, 2 ice,
/// 3 uplift, 4 sweep, 5 rebound, 6 peak).
fn plate_change(
    table: &mut [[f64; 7]],
    proc: usize,
    before: &[i32],
    after: &[i32],
    area: &[u64],
    plate: &[u8],
) {
    for k in 0..after.len() {
        let d = after[k] - before[k];
        if d != 0 {
            table[usize::from(plate[k])][proc] +=
                f64::from(d) * area[k] as f64 / f64::from(Z_STEPS_PER_M);
        }
    }
}

/// The volume each side of the sea's level gained from `before` to `after`, `[land, sea floor]`, m³ (sixteenths × m²
/// over sixteen).
fn volume_change(before: &[i32], after: &[i32], area: &[u64], sea_z: i32) -> [f64; 2] {
    let mut v = [0.0f64; 2];
    for ((&b, &a), &m2) in before.iter().zip(after).zip(area) {
        let side = usize::from(b <= sea_z);
        v[side] += f64::from(a - b) * m2 as f64 / f64::from(Z_STEPS_PER_M);
    }
    v
}

/// The land after a pass: its mean height over the sea (m, by area), its share over 1 000 m, and its share of the globe.
fn land_reading(state: &MacroSolve) -> (f64, f64, f64) {
    let (mut land, mut over_km, mut height, mut all) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&z, &m2) in state.z.iter().zip(&state.area) {
        let a = m2 as f64;
        all += a;
        if z > state.sea_z {
            let over = f64::from(z) / f64::from(Z_STEPS_PER_M)
                - f64::from(state.sea_z) / f64::from(Z_STEPS_PER_M);
            land += a;
            height += over * a;
            if over > 1_000.0 {
                over_km += a;
            }
        }
    }
    let per = |x: f64, of: f64| if of > 0.0 { x / of } else { 0.0 };
    (per(height, land), per(over_km, land), per(land, all))
}

/// One routing of a pass: the game's own (`route_for_pass`, ruling B73 item 1) — the ocean's label alone where the
/// budget's tree routes the rest (a body with water and the water's own paths in) — and the full routing where a law
/// taken out (`water_paths`) or a candidate (`routing_flats`) reads the routing's own receivers.
fn route_copy(
    state: &mut MacroSolve,
    has_water: bool,
    laws: &Laws,
) -> vd_terrain::solve::RouteReport {
    if has_water & laws.water_paths & !laws.routing_flats {
        return state.label_for_budget();
    }
    let report = state.route();
    state.accumulate();
    report
}

/// ★ THE COPY of `vd_terrain::solve::solve_full` (phase 2), line for line, with each law's hook marked `LAW`. Every other
/// line is the solve's own; the control proves it.
#[must_use]
pub fn solve_copy(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
    laws: &Laws,
    ledger_on: bool,
) -> Option<Solved> {
    solve_copy_watched(body, words, schedule, laws, ledger_on, None)
}

/// A bench's look at the copy's state inside every pass: the pass's index, its end in years from the solve's start, the
/// stage just done (`craters`, `talus`, `ice`, `uplift`, `sweep`, `rebound`, `peak` — the last the pass's end; its
/// receivers are the ones the pass's sweeps read), and the state. It reads; it never writes.
pub type Watch<'a> = &'a mut dyn FnMut(usize, u64, &str, &MacroSolve);

/// [`solve_copy`] with a bench's [`Watch`] called at the end of every pass (ruling B58: where a grid mark forms in the
/// solve's history). The watch only reads, so the solve is the copy's own whether a watch is given or not.
#[must_use]
pub fn solve_copy_watched(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
    laws: &Laws,
    ledger_on: bool,
    watch: Option<Watch<'_>>,
) -> Option<Solved> {
    // ★ THE COPY'S CORES (ruling B64): `VD_BENCH_CORES` threads for the game's own steps the copy calls (a bench
    // setting; the same bytes on any number), one when unset.
    let cores = vd_terrain::par::Cores(
        std::env::var("VD_BENCH_CORES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
    );
    solve_copy_inner(body, words, schedule, laws, ledger_on, watch, cores)
}

/// [`solve_copy`] on `cores` threads (the same bytes on any number, ruling B64): for a test that solves the copy as the
/// shared test solves do, on all the host's cores (ruling B74 item 2).
#[must_use]
pub fn solve_copy_on(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
    laws: &Laws,
    ledger_on: bool,
    cores: vd_terrain::par::Cores,
) -> Option<Solved> {
    solve_copy_inner(body, words, schedule, laws, ledger_on, None, cores)
}

#[allow(clippy::too_many_lines)]
fn solve_copy_inner(
    body: &BodyDefinition,
    words: &SolveWords,
    schedule: Schedule,
    laws: &Laws,
    ledger_on: bool,
    mut watch: Option<Watch<'_>>,
    cores: vd_terrain::par::Cores,
) -> Option<Solved> {
    let lattice = body.macro_lattice()?;
    // LAND (B58 item 2, B59–B61): the game's own initial land, or the bench land copy with a law out.
    let mut land = if laws.land == crate::land_copy::LandLaws::default() {
        vd_terrain::land::initial_land_on(body, &lattice, &words.land(), cores)
    } else {
        crate::land_copy::initial_land_copy(body, &lattice, &words.land(), laws.land)
    };
    let n = lattice.node_count();
    let has_water = words.water_km3 > 0;
    let mut report = FullReport::default();
    // LAND (B59 item 2, B61 item 1): the game's belts' law, or the belts' copy with the share or the Andean coast out
    // (the coast goes with the margins).
    let mut belts = if laws.land == crate::land_copy::LandLaws::default() {
        vd_terrain::belts::belts(body, &lattice, words, &land, schedule.age_yr)
    } else {
        crate::belts_copy::belts_copy(
            body,
            &lattice,
            words,
            &land,
            schedule.age_yr,
            laws.land.share,
            laws.land.margins,
        )
    };
    // LAW (B49–B51): out, no wedge rises; the plates' turnover stays the law's.
    if !laws.wedge {
        belts = Belts {
            total: vec![0; n],
            start_yr: vec![schedule.age_yr; n],
            kind: vec![BELT_NONE; n],
            plateau: vec![false; n],
            wedge: Vec::new(),
            report: BeltReport {
                t_m_yr: belts.report.t_m_yr,
                ..BeltReport::default()
            },
        };
    }
    let even_total = belt_land(&belts, &mut land);
    // LAW (B45–B46): out, every plate as old as the system.
    let raise_yr = if laws.landscape_time {
        raise_times(&land.plates, schedule.age_yr, belts.report.t_m_yr > 0)
    } else {
        vec![schedule.age_yr; land.plates.len()]
    };
    let erosion_start = erosion_starts(&belts, &raise_yr, &land.plate, schedule.age_yr);
    let mut state = MacroSolve::from_land(body, &land)?;
    state.cores = cores;
    let gravity = body.facts().gravity_mm_s2;
    let mut source = CopyClimate::of(&lattice, words, laws, cores);
    let rain_every = source.every(laws);
    let pre_climate = source.at(body, &lattice, words, &land.z, land.sea_view(), cores);
    let surface = vd_terrain::craters::Surface {
        crust: &land.crust,
        boundary_m: &land.boundary_m,
        plate: &land.plate,
        plates: &land.plates,
        province: &land.province,
        rain_mm_yr: &pre_climate.rain_mm_yr,
    };
    let craters = vd_terrain::craters::crater_population(body, &lattice, words, &surface);
    drop(pre_climate);
    // LAW (B37): out, every crater of the record stamped before the first pass.
    if !laws.craters_at_own_age {
        vd_terrain::craters::apply_craters(&mut state.z, &lattice, &craters, gravity);
        report.crater_record.extend_from_slice(&craters);
    }
    state.z_flood.clone_from(&state.z);
    let peak_cap = (Gf::from_f64(vd_terrain::land::relief_cap_m(body))
        * Gf::from_i64(i64::from(Z_STEPS_PER_M)))
    .to_i64_floor() as i32;
    let alpha_m = vd_terrain::land::flexural_parameter_m(words.elastic_thickness_m, gravity);
    let times = schedule.times();
    let passes = times.len() - 1;
    let sub_sweeps = schedule.sub_sweeps.max(1);
    let epoch_start = schedule.age_yr.saturating_sub(GLACIAL_EPOCH_YR);
    let no_ice: Vec<i32> = if has_water {
        Vec::new()
    } else {
        vec![i32::MAX; n]
    };
    let mut since_climate = laws.climate_every;
    let mut since_rain = rain_every;
    let mut climate: Option<vd_terrain::climate::Climate> = None;
    let mut talus_done = 0u32;
    let mut ledger: Vec<PassLedger> = Vec::new();
    let mut by_plate: Vec<[f64; 7]> = if ledger_on {
        vec![[0.0; 7]; land.plates.len()]
    } else {
        Vec::new()
    };
    let mut plate_passes: Vec<PlatePass> = Vec::new();
    // The calm land ([`PlatePass`]): never moved by a margin, a ridge, a rift or a belt over the whole age.
    let calm: Vec<bool> = if ledger_on {
        (0..n)
            .map(|k| {
                (belts.total[k] <= 0)
                    & (land.uplift[k] == 0)
                    & (belts.kind[k] == BELT_NONE)
                    & !belts.plateau[k]
            })
            .collect()
    } else {
        Vec::new()
    };
    // The candidate's ring repairs over the passes ([`Laws::routing_flats`]).
    let mut flats_repaired = 0u64;
    // The ledger's closed hollows at the routing before (the gate's cause).
    let mut was_closed: Vec<bool> = if ledger_on {
        vec![false; n]
    } else {
        Vec::new()
    };
    // ★ THE STAGES' TIMES (ruling B62: where a solve's time goes), printed every 25 passes when `VD_BENCH_TIMES` is set —
    // a bench reading on stderr, never part of the solve.
    let timing = std::env::var("VD_BENCH_TIMES").is_ok();
    let started = std::time::Instant::now();
    let mut times_s = [0.0f64; 6];
    for pass in 0..passes {
        let (t0, t1) = (times[pass], times[pass + 1]);
        let last = pass + 1 == passes;
        let mut entry = PassLedger {
            t0_yr: t0,
            t1_yr: t1,
            ..PassLedger::default()
        };
        let mut before = if ledger_on {
            state.z.clone()
        } else {
            Vec::new()
        };
        let sea_before = state.sea_z;
        let plate_before = by_plate.clone();
        let book = |slot: &mut [f64; 2], state: &MacroSolve, before: &mut Vec<i32>| {
            if ledger_on {
                *slot = volume_change(before, &state.z, &state.area, sea_before);
                before.clone_from(&state.z);
            }
        };
        if laws.craters_at_own_age {
            let due: Vec<vd_terrain::craters::Crater> = craters
                .iter()
                .copied()
                .filter(|c| {
                    let at = schedule.age_yr.saturating_sub(c.age_yr);
                    (at >= erosion_start[c.node as usize])
                        & (at >= t0)
                        & ((at < t1) | (last & (at <= t1)))
                })
                .collect();
            vd_terrain::craters::apply_craters(&mut state.z, &lattice, &due, gravity);
            report.crater_record.extend_from_slice(&due);
        }
        if ledger_on {
            plate_change(
                &mut by_plate,
                0,
                &before,
                &state.z,
                &state.area,
                &land.plate,
            );
        }
        book(&mut entry.craters_m3, &state, &mut before);
        if let Some(w) = watch.as_mut() {
            w(pass, t1, "craters", &state);
        }
        if since_climate >= laws.climate_every {
            if let Some(sea) = vd_terrain::land::sea_level_connected(
                &lattice,
                &state.z,
                &state.area,
                words.water_km3,
                false,
            ) {
                state.sea_z = sea.level;
                state.ocean = sea.mask.ocean;
                report.sea_reseeded |= sea.reseeded;
                report.sea_unsettled |= !sea.settled;
            }
            if since_rain >= rain_every {
                let clock = std::time::Instant::now();
                let c = source.at(
                    body,
                    &lattice,
                    words,
                    &state.z,
                    has_water.then_some(state.sea_view()),
                    cores,
                );
                times_s[4] += clock.elapsed().as_secs_f64();
                state.rain.clone_from(&c.rain_mm_yr);
                climate = Some(c);
                since_rain = 0;
            }
            since_climate = 0;
        }
        since_climate += 1;
        since_rain += 1;
        let c = climate
            .as_ref()
            .expect("the climate stands from the first pass");
        let clock = std::time::Instant::now();
        report.routes.push(route_copy(&mut state, has_water, laws));
        times_s[0] += clock.elapsed().as_secs_f64();
        // CANDIDATE (B54): the routing's own receivers, kept for the open land.
        let (routed, routed_chord) = if laws.routing_flats {
            (state.receiver.clone(), state.chord.clone())
        } else {
            (Vec::new(), Vec::new())
        };
        // LAW (B51 item 2): out, the routing's own receivers, discharge and hollows.
        let clock = std::time::Instant::now();
        let mut basins = if laws.water_paths {
            state.follow_the_budget(c, has_water)
        } else {
            None
        };
        times_s[1] += clock.elapsed().as_secs_f64();
        // CANDIDATE (not a ruled law): the full lakes pass their catchment's rain in the share of their water that leaves.
        if laws.water_paths & laws.pass_share & has_water {
            let lakes = vd_terrain::lakes::budget(&state, c);
            pass_share_discharge(&mut state, &lakes);
        }
        // CANDIDATE (B54, not a ruled law): the routing's own receivers on the land outside every hollow.
        if laws.water_paths & laws.routing_flats & has_water {
            let lakes = vd_terrain::lakes::budget(&state, c);
            flats_repaired +=
                routing_on_open_land(&mut state, &lakes, &routed, &routed_chord) as u64;
        }
        if let (true, Some(b)) = (ledger_on, &basins) {
            entry.closed_bodies = b.bodies.len() as u32;
            for (i, (&s, was)) in b.slot.iter().zip(was_closed.iter_mut()).enumerate() {
                let now = s != u32::MAX;
                let a = state.area[i] as f64;
                entry.closed_m2 += f64::from(u8::from(now)) * a;
                entry.opened_m2 += f64::from(u8::from(*was & !now)) * a;
                entry.newly_closed_m2 += f64::from(u8::from(!*was & now)) * a;
                *was = now;
            }
        }
        let talus_due = (pass as u64 + 1) * u64::from(schedule.talus_passes) / passes as u64
            > pass as u64 * u64::from(schedule.talus_passes) / passes as u64;
        if talus_due {
            let tan: Vec<u32> = c.aridity_q8.iter().map(|&a| tan_repose_q16(a)).collect();
            report.talus.push(state.talus(&tan));
            talus_done += 1;
        }
        if ledger_on {
            plate_change(
                &mut by_plate,
                1,
                &before,
                &state.z,
                &state.area,
                &land.plate,
            );
        }
        book(&mut entry.talus_m3, &state, &mut before);
        if let Some(w) = watch.as_mut() {
            w(pass, t1, "talus", &state);
        }
        let epoch = vd_terrain::belts::overlap(epoch_start, schedule.age_yr, t0, t1);
        if epoch > 0 {
            let line: &[i32] = if has_water { &c.ela_z } else { &no_ice };
            let (cut, steps, floored) =
                state.ice_stepped(line, &c.temperature_dk, c.lapse_mk_km, gravity, epoch);
            report.ice.2 += cut;
            report.ice_steps += steps;
            report.ice_floored += floored;
            entry.ice_steps = steps;
        }
        if ledger_on {
            plate_change(
                &mut by_plate,
                2,
                &before,
                &state.z,
                &state.area,
                &land.plate,
            );
        }
        book(&mut entry.ice_m3, &state, &mut before);
        if let Some(w) = watch.as_mut() {
            w(pass, t1, "ice", &state);
        }
        pass_uplift(&mut state.z, &even_total, &belts, schedule.age_yr, t0, t1);
        if ledger_on {
            plate_change(
                &mut by_plate,
                3,
                &before,
                &state.z,
                &state.area,
                &land.plate,
            );
        }
        book(&mut entry.uplift_m3, &state, &mut before);
        if let Some(w) = watch.as_mut() {
            w(pass, t1, "uplift", &state);
        }
        let active: Vec<u32> = erosion_start
            .iter()
            .map(|&st| active_share(st, t0, t1))
            .collect();
        let dt = Gf::from_i64((t1 - t0) as i64) / Gf::from_i64(i64::from(sub_sweeps));
        let mut cut_sum = vec![0u32; n];
        let mut lifted = [0.0f64; 2];
        // LAW (B65 item 1): the water routed again before each later share of the sweeps.
        let mut groups = state.basin_groups();
        for s in 0..sub_sweeps {
            if schedule.routes_before(s) {
                let clock = std::time::Instant::now();
                report.routes.push(route_copy(&mut state, has_water, laws));
                times_s[0] += clock.elapsed().as_secs_f64();
                let clock = std::time::Instant::now();
                basins = if laws.water_paths {
                    state.follow_the_budget(c, has_water)
                } else {
                    None
                };
                groups = state.basin_groups();
                times_s[1] += clock.elapsed().as_secs_f64();
            }
            let clock = std::time::Instant::now();
            report
                .sweeps
                .push(state.sweep_grouped(dt, Some(&active), basins.as_ref(), &groups));
            times_s[2] += clock.elapsed().as_secs_f64();
            for (sum, &cut) in cut_sum.iter_mut().zip(&state.cut_now) {
                *sum = sum.saturating_add(cut);
            }
            if ledger_on {
                // `before` still holds the ground just before this sweep (the last book).
                let r = report.sweeps.last().copied().unwrap_or_default();
                let slot = basins.as_ref().map(|b| b.slot.as_slice());
                let (way, check) = rock_way(&state, &before, slot, &r);
                let m3 = |x: u128| x as f64 / f64::from(Z_STEPS_PER_M);
                entry.rock_cut_m3 += way[0];
                entry.rock_laid_open_m3 += way[1];
                entry.rock_laid_closed_m3 += way[2];
                entry.rock_over_spill_m3 += m3(r.rock_over_spill);
                entry.rock_sea_m3 += m3(r.rock_to_sea);
                entry.rock_dropped_m3 += m3(r.rock_dropped);
                entry.rock_remainder_m3 += m3(r.rock_remainder);
                entry.rock_check += check;
            }
            let mut swept = [0.0f64; 2];
            if ledger_on {
                plate_change(
                    &mut by_plate,
                    4,
                    &before,
                    &state.z,
                    &state.area,
                    &land.plate,
                );
            }
            book(&mut swept, &state, &mut before);
            entry.sweep_m3[0] += swept[0];
            entry.sweep_m3[1] += swept[1];
            if let Some(w) = watch.as_mut() {
                w(pass, t1, "sweep", &state);
            }
            let clock = std::time::Instant::now();
            report.rebounds.push(state.rebound(alpha_m));
            times_s[3] += clock.elapsed().as_secs_f64();
            if ledger_on {
                plate_change(
                    &mut by_plate,
                    5,
                    &before,
                    &state.z,
                    &state.area,
                    &land.plate,
                );
            }
            book(&mut lifted, &state, &mut before);
            entry.rebound_m3[0] += lifted[0];
            entry.rebound_m3[1] += lifted[1];
            if let Some(w) = watch.as_mut() {
                w(pass, t1, "rebound", &state);
            }
        }
        state.cut_now = cut_sum;
        let (collapsed, left) = state.peak_law(peak_cap);
        report.peak.0 += collapsed;
        report.peak.1 = report.peak.1.max(left);
        if ledger_on {
            plate_change(
                &mut by_plate,
                6,
                &before,
                &state.z,
                &state.area,
                &land.plate,
            );
        }
        book(&mut entry.peak_m3, &state, &mut before);
        if let Some(w) = watch.as_mut() {
            w(pass, t1, "peak", &state);
        }
        if timing && (pass % 25 == 0 || pass + 1 == passes) {
            eprintln!(
                "TIMES pass {pass}/{passes}: {:.0} s in all; route {:.0} s, budget {:.0} s, sweeps {:.0} s, rebounds {:.0} s, climate {:.0} s",
                started.elapsed().as_secs_f64(),
                times_s[0],
                times_s[1],
                times_s[2],
                times_s[3],
                times_s[4]
            );
        }
        if ledger_on {
            (entry.relief_p50_m, entry.plains_pct) =
                if t1 / TEXTURE_EVERY_YR > t0 / TEXTURE_EVERY_YR {
                    texture(&state)
                } else {
                    (-1.0, -1.0)
                };
            (entry.land_mean_m, entry.land_over_1km, entry.land_share) = land_reading(&state);
            entry.sea_z = state.sea_z;
            ledger.push(entry);
            let lands = plate_land(&state, &land.plate, &calm, by_plate.len());
            for (p, (row, l)) in by_plate.iter().zip(&lands).enumerate() {
                let mut process_m3 = *row;
                for (d, b) in process_m3.iter_mut().zip(&plate_before[p]) {
                    *d -= b;
                }
                plate_passes.push(PlatePass {
                    t1_yr: t1,
                    plate: p as u8,
                    process_m3,
                    land_m2: l[0],
                    land_mean_m: l[1] / l[0].max(1.0),
                    calm_m2: l[2],
                    calm_mean_m: l[3] / l[2].max(1.0),
                    under_m2: l[4],
                });
            }
        }
    }
    report.craters = report.crater_record.len();
    let climate = match climate {
        Some(c) => c,
        None => source.at(
            body,
            &lattice,
            words,
            &state.z,
            has_water.then_some(state.sea_view()),
            cores,
        ),
    };
    let tan: Vec<u32> = climate
        .aridity_q8
        .iter()
        .map(|&a| tan_repose_q16(a))
        .collect();
    while talus_done < schedule.talus_passes {
        report.talus.push(state.talus(&tan));
        talus_done += 1;
    }
    drop(tan);
    if let Some(sea) = vd_terrain::land::sea_level_connected(
        &lattice,
        &state.z,
        &state.area,
        words.water_km3,
        false,
    ) {
        state.sea_z = sea.level;
        report.sea_reseeded |= sea.reseeded;
        report.sea_unsettled |= !sea.settled;
    }
    report.routes.push(state.route());
    let ice_mask = {
        let line: &[i32] = if has_water { &climate.ela_z } else { &no_ice };
        let (mask, under, thickest) =
            state.ice_mask(line, &climate.temperature_dk, climate.lapse_mk_km, gravity);
        report.ice.0 = under;
        report.ice.1 = thickest.to_f64();
        mask
    };
    let mut final_climate = climate;
    let mut budget = None;
    if has_water {
        let mut climate = source.at(
            body,
            &lattice,
            words,
            &state.z,
            Some(state.sea_view()),
            cores,
        );
        let mut lakes = vd_terrain::lakes::budget(&state, &climate);
        let volume = lakes.report.volume_m3;
        let fix = settle_lakes_sea(
            &mut state,
            words.water_km3,
            LAKE_SEA_SOLVES,
            volume,
            &mut report.routes,
            &mut |s: &MacroSolve| {
                climate = source.at(body, &lattice, words, &s.z, Some(s.sea_view()), cores);
                lakes = vd_terrain::lakes::budget(s, &climate);
                lakes.report.volume_m3
            },
        );
        report.lake_sea_solves = fix.solves;
        report.lake_sea_converged = fix.converged;
        report.sea_reseeded |= fix.reseeded;
        report.sea_unsettled |= fix.unsettled;
        report.sea_before_lakes = (state.sea_z != fix.first).then_some(fix.first);
        report.lakes = lakes.report.clone();
        state.water_z.clone_from(&lakes.water);
        final_climate = climate;
        budget = Some(lakes);
    }
    report.sea_z = has_water.then_some(state.sea_z);
    report.ocean_share = state.ocean_share();
    report.coast_band = coast_band(words, gravity);
    report.integrals = state.hypsometric_integrals(AGE_GATE_MIN_NODES);
    report.deposits = state.sediment_by_basin().len();
    let tree = match &budget {
        Some(lakes) => vd_terrain::lakes::water_tree(&state, lakes),
        None => vd_terrain::lakes::dry_tree(&state),
    };
    drop(budget);
    report.tree = state.follow_the_water(tree, &final_climate.runoff_mm_yr);
    // ★ THE CLIMATE THE ARTIFACT SHIPS, as the game's (`solve_full_shipped_on`): the lakes' last reading on a wet body
    // with the chain, else one more reading over the final ground (a dry body reads no sea).
    let shipped = if has_water && source.seasons.is_some() {
        final_climate
    } else {
        drop(final_climate);
        source.at(
            body,
            &lattice,
            words,
            &state.z,
            has_water.then_some(state.sea_view()),
            cores,
        )
    };
    report.raise_yr.clone_from(&raise_yr);
    let widened = if laws.wedge {
        belts.plateau.clone()
    } else {
        Vec::new()
    };
    let wedge = laws.wedge.then(|| belts.clone());
    report.belts = belts;
    let facies = state.facies(&ice_mask, report.coast_band, has_water);
    Some(Solved {
        state,
        facies,
        climate: shipped,
        belts: wedge,
        craters: report.crater_record.clone(),
        collapsed: report.peak.0,
        peak_left: report.peak.1,
        report,
        ledger,
        by_plate,
        plate_passes,
        raise_yr,
        plate_of: land.plate,
        widened,
        flats_repaired,
    })
}

/// The artifact exactly as `vd_bins::artifact_worker::run_solve` makes it from a solved state.
#[must_use]
pub fn artifact_of(words: &SolveWords, s: &Solved) -> Artifact {
    Artifact::of(&s.state, &s.facies, &s.climate, words.water_km3 > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_terrain::home::{
        HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words, home_solve_words,
    };

    /// ★ THE CONTROL (ruling B38 item 1): with every law in, the copy is the game's own solve, bit for bit — the state's
    /// ground, water, facies and report — on the home moon, dry and with the home planet's air and a sea; and a law
    /// taken out changes the world exactly where the moon's own facts let it act.
    #[test]
    fn the_control_is_the_games_own_solve() {
        let moon = home_moon();
        let wet = SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        // Every solve on all the host's cores, as the shared test solves (ruling B74 item 2): the same bytes on any number.
        let cores =
            vd_terrain::par::Cores(std::thread::available_parallelism().map_or(1, usize::from));
        for words in [home_moon_solve_words(), wet] {
            let schedule = Schedule::standard(HOME_SYSTEM_AGE_YR);
            let (state, facies, report, climate) =
                vd_terrain::solve::solve_full_shipped_on(&moon, &words, schedule, cores)
                    .expect("a solve");
            let copy = solve_copy_on(&moon, &words, schedule, &Laws::default(), true, cores)
                .expect("a copy");
            assert_eq!(copy.state.z, state.z);
            assert_eq!(copy.state.water_z, state.water_z);
            assert_eq!(copy.state.receiver, state.receiver);
            assert_eq!(copy.state.discharge, state.discharge);
            assert_eq!(copy.facies, facies);
            assert_eq!(copy.report, report);
            assert_eq!(copy.ledger.len(), schedule.times().len() - 1);
            // The ledger by plate and pass: a row a plate a pass, whose processes sum to the plate's totals.
            let plates = copy.by_plate.len();
            assert_eq!(copy.plate_passes.len(), copy.ledger.len() * plates);
            for (p, total) in copy.by_plate.iter().enumerate() {
                let mut sum = [0.0f64; 7];
                for r in copy
                    .plate_passes
                    .iter()
                    .filter(|r| usize::from(r.plate) == p)
                {
                    for (s, v) in sum.iter_mut().zip(r.process_m3) {
                        *s += v;
                    }
                }
                for (s, t) in sum.iter().zip(total) {
                    assert!(
                        (s - t).abs() <= 1e-6 * t.abs().max(1.0),
                        "plate {p}: {s} against {t}"
                    );
                }
            }
            assert_eq!(
                artifact_of(&words, &copy).digest(),
                Artifact::of(&state, &facies, &climate, words.water_km3 > 0).digest()
            );
        }
        // The water's paths or the craters' own time taken out moves the wet moon's ground; the landscape's time or the
        // wedge taken out moves nothing, because the moon is a stagnant lid — no ridge opens an ocean, so no belt rises
        // and no plate is reworked (every raise time is already the system's age).
        let schedule = Schedule::standard(HOME_SYSTEM_AGE_YR);
        let all =
            solve_copy_on(&moon, &wet, schedule, &Laws::default(), false, cores).expect("a copy");
        let out =
            |laws: Laws| solve_copy_on(&moon, &wet, schedule, &laws, false, cores).expect("a copy");
        let water = out(Laws {
            water_paths: false,
            ..Laws::default()
        });
        assert_ne!(water.state.z, all.state.z);
        let craters = out(Laws {
            craters_at_own_age: false,
            ..Laws::default()
        });
        assert_ne!(craters.state.z, all.state.z);
        let time = out(Laws {
            landscape_time: false,
            ..Laws::default()
        });
        assert_eq!(time.state.z, all.state.z);
        let wedge = out(Laws {
            wedge: false,
            ..Laws::default()
        });
        assert_eq!(wedge.state.z, all.state.z);
        assert!(all.belts.as_ref().is_some_and(|b| b.wedge.is_empty()));
        assert!(wedge.belts.is_none());
    }

    /// The front flank's constant is the critical taper's value at Taiwan's numbers, as a tangent, and the back flank's
    /// is the tangent of Taiwan's measured 6°.
    #[test]
    fn the_critical_taper_at_taiwan() {
        let a = critical_taper_rad(TAPER_MU, TAPER_MU_B, TAPER_LAMBDA, TAPER_BETA_RAD);
        assert!((a - 0.037_815).abs() < 1e-6, "{a}");
        assert!((a.tan() - vd_terrain::belts::PRO_TAPER_TAN).abs() < 1e-6);
        assert!((RETRO_TAPER_RAD.tan() - vd_terrain::belts::RETRO_TAPER_TAN).abs() < 1e-12);
    }
}
