//! ★ THE SOLVE'S BENCH (the landform arc, slice 8c stage C1; measurement M-L1 of
//! `slice_8c_design.md` §9): the solve's CORE — the lattice, the starting surface, the priority
//! flood, the receivers, the flats, the discharge and the stream-power sweep — on THE world's own
//! bodies, single-threaded, with its WALL TIME per phase and its PEAK MEMORY. The number the
//! ship-or-derive decision (the design's ask 2) is made on, and the number that says whether a
//! planet a pilot approaches can be solved before the pilot arrives.
//!
//! What runs, per body: the state's construction (the directions, the areas, the starting
//! surface), ONE routing, ONE accumulation, ONE sweep — each timed alone — then the WHOLE standard
//! schedule (forty sweeps, four routings) timed as one, and the process's peak resident set
//! after it. The bodies: the home planet, its moon, and the home system's other planets the
//! ladder accepts, smallest first, up to a stated node ceiling (`VD_BENCH_MAX_NODES`, default
//! ten million) — an ice giant at the lattice's memory ceiling holds twenty-five million nodes.
//! `VD_BENCH_BODY=<seed>` runs ONE body alone, so its peak memory is its own (the peak resident
//! set never falls, so a body solved after another reads the larger of the two).
//!
//! Run in release, on a quiet machine:
//!
//! ```text
//! cargo run --release -p vd-bins --example macro_bench
//! ```

use std::process::ExitCode;
use std::time::Instant;

use vd_bins::DEV;
use vd_core::pose::RealmId;
use vd_physics::worldgen::{UniverseConfig, body_facts_in_subtree, shard_boot_world};
use vd_terrain::BodyDefinition;
use vd_terrain::home::home_moon;
use vd_terrain::land::{Kind, LandWords, humps, hypsometry, initial_land};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{
    FACIES_COAST, FACIES_ICE, FACIES_LAKE, FACIES_SEA, MacroSolve, Schedule, SolveWords,
    Z_STEPS_PER_M, age_gate, solve, solve_full,
};

/// The node ceiling a body must stand under to be solved by default.
const DEFAULT_MAX_NODES: usize = 10_000_000;

/// The process's peak resident set in bytes (macOS reports bytes, Linux kilobytes).
fn peak_rss_bytes() -> u64 {
    // SAFETY: `getrusage` writes one plain struct the caller owns; a zeroed `rusage` is a valid
    // value of that struct on every target the game ships on.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    };
    let raw = usage.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    }
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn metres(steps: i32) -> f64 {
    f64::from(steps) / f64::from(Z_STEPS_PER_M)
}

/// ★ THE INITIAL LAND's bench (stage C2): the land timed, its plates, its crust share, its sea,
/// gate G-LAND-HYPSOMETRY's reading (two humps and the valley between), then the schedule from
/// the land.
fn bench_land(label: &str, body: &BodyDefinition, words: &LandWords, age_yr: u64) {
    let Some(lattice) = MacroLattice::of(body) else {
        return;
    };
    let t = Instant::now();
    let land = initial_land(body, &lattice, words);
    let t_land = t.elapsed();
    let n = lattice.node_count();
    let area: Vec<u64> = (0..n as u32).map(|k| lattice.area_m2(k)).collect();
    let total: u128 = area.iter().map(|&a| u128::from(a)).sum();
    let share = |pred: &dyn Fn(usize) -> bool| {
        (0..n)
            .filter(|&k| pred(k))
            .map(|k| u128::from(area[k]))
            .sum::<u128>() as f64
            / total as f64
    };
    let continental = share(&|k| land.crust[k] >= 128);
    let ocean = land.sea_z.map_or(0.0, |sea| share(&|k| land.z[k] <= sea));
    let kinds = [
        Kind::None,
        Kind::Convergent,
        Kind::Divergent,
        Kind::Transform,
    ]
    .map(|kind| share(&|k| land.kind[k] == kind as u8 && land.boundary_m[k] < 250_000));
    let (lo, hi) = land
        .z
        .iter()
        .fold((i32::MAX, i32::MIN), |(lo, hi), &z| (lo.min(z), hi.max(z)));
    println!(
        "\n{label} — THE INITIAL LAND (C2): {:.3} s; {} plates; continental crust {:.1} % of the area; \
         relief {:.0}..{:.0} m; sea {}; ocean share {:.1} %; within 250 km of a boundary: convergent \
         {:.1} %, divergent {:.1} %, transform {:.1} %",
        t_land.as_secs_f64(),
        land.plates.len(),
        continental * 100.0,
        metres(lo),
        metres(hi),
        land.sea_z
            .map_or("none".to_owned(), |s| format!("{:.0} m", metres(s))),
        ocean * 100.0,
        kinds[1] * 100.0,
        kinds[2] * 100.0,
        kinds[3] * 100.0
    );
    let (base, hist) = hypsometry(&land.z, &area);
    match humps(&hist) {
        Some((low, high, valley)) => println!(
            "  G-LAND-HYPSOMETRY: TWO HUMPS at {:.0} m and {:.0} m, the valley between {:.2} of the lower \
             hump ({})",
            metres(base) + f64::from(low as u32) * 250.0,
            metres(base) + f64::from(high as u32) * 250.0,
            valley,
            if valley < 0.5 {
                "GREEN"
            } else {
                "RED: the valley is not a valley"
            }
        ),
        None => println!("  G-LAND-HYPSOMETRY: ONE HUMP — RED"),
    }
    let t = Instant::now();
    let mut state = MacroSolve::from_land(body, &land).expect("a state");
    let schedule = Schedule::standard(age_yr);
    let gain = schedule.gain();
    let mut routes = Vec::new();
    let mut since = schedule.flood_every;
    let mut last = Default::default();
    for pass in 0..schedule.passes {
        if since >= schedule.flood_every {
            routes.push(state.route());
            state.accumulate();
            since = 0;
        }
        since += 1;
        last = state.sweep(gain, pass, schedule.passes);
    }
    let t_solve = t.elapsed();
    let r = routes.last().copied().unwrap_or_default();
    let (lo, hi) = state.range();
    println!(
        "  THE SCHEDULE FROM THE LAND: {:.3} s; last routing: sea seeded {}, outlets {}, lake nodes {}, flat {}, \
         undrained {}, cyclic {}; relief after {:.0}..{:.0} m; last sweep lowered {} nodes",
        t_solve.as_secs_f64(),
        r.sea_seeded,
        r.outlets,
        r.raised,
        r.flat,
        r.undrained,
        r.cyclic,
        metres(lo),
        metres(hi),
        last.lowered
    );
}

/// ★ THE FULL SOLVE's bench (stage C3): the whole schedule — the craters, the climate inside it,
/// the uplift, the rebound, the talus, the ice, the coast, the envelope — timed as one, with the
/// gates' readings: G-AGE (the basins' integrals against Strahler's band), the facies shares, the
/// deposits. `VD_BENCH_K0_SCAN=1` adds the erodibility scan on this body (a tenth, one, ten times
/// the stated `K0_PER_YR`) so the calibration against the age is a MEASURED choice.
fn bench_full(label: &str, body: &BodyDefinition, words: &SolveWords) {
    let schedule = Schedule::standard(words.age_yr);
    let rss_before = peak_rss_bytes();
    let t = Instant::now();
    let Some((state, facies, report)) = solve_full(body, words, schedule) else {
        return;
    };
    let t_full = t.elapsed();
    let rss = peak_rss_bytes();
    let n = state.node_count();
    let total: u128 = state.area.iter().map(|&a| u128::from(a)).sum();
    let share = |bit: u8| {
        (0..n)
            .filter(|&k| facies[k] & bit != 0)
            .map(|k| u128::from(state.area[k]))
            .sum::<u128>() as f64
            / total as f64
            * 100.0
    };
    let (lo, hi) = state.range();
    let last_route = report.routes.last().copied().unwrap_or_default();
    let cut: u64 = report.sweeps.iter().map(|s| s.total_cut).sum();
    let lifted: usize = report.rebounds.iter().map(|r| r.1).sum();
    let talus_first = report.talus.first().map_or(0, |t| t.1);
    let talus_last = report.talus.last().map_or(0, |t| t.1);
    println!(
        "\n{label} — THE FULL SOLVE (C3): {:.3} s wall on one thread; peak RSS {:.0} MB ({:.1} B/node over \
         the start); {} craters; {} routings (last: {} outlets, {} lake nodes, {} flat, undrained {}, \
         cyclic {}); cuts {:.0} m summed over the sweeps; rebounds {} (nodes lifted {}); talus worst \
         excess {:.1} m first, {:.1} m last; ice: {} nodes, thickest {:.0} m, deepest cut {:.0} m; \
         envelope: max |z| {:.0} m, scaled {}; coast band {:.1} m; deposits in {} basins",
        t_full.as_secs_f64(),
        mb(rss),
        rss.saturating_sub(rss_before) as f64 / n as f64,
        report.craters,
        report.routes.len(),
        last_route.outlets,
        last_route.raised,
        last_route.flat,
        last_route.undrained,
        last_route.cyclic,
        cut as f64 / f64::from(Z_STEPS_PER_M),
        report.rebounds.len(),
        lifted,
        talus_first as f64 / f64::from(Z_STEPS_PER_M),
        talus_last as f64 / f64::from(Z_STEPS_PER_M),
        report.ice.0,
        report.ice.1,
        metres(report.ice.2),
        metres(report.envelope.0),
        report.envelope.1,
        metres(report.coast_band),
        report.deposits
    );
    println!(
        "  the shape: relief {:.0}..{:.0} m; facies by area: sea {:.1} %, lake {:.1} %, coast {:.1} %, \
         ice {:.1} %",
        metres(lo),
        metres(hi),
        share(FACIES_SEA),
        share(FACIES_LAKE),
        share(FACIES_COAST),
        share(FACIES_ICE)
    );
    match age_gate(&report.integrals) {
        Some((median, mature)) => println!(
            "  G-AGE: {} basins of 100+ land nodes; the median hypsometric integral {:.2} — {}",
            report.integrals.len(),
            f64::from(median) / 256.0,
            if mature {
                "MATURE (GREEN)"
            } else {
                "outside Strahler's maturity band (RED)"
            }
        ),
        None => println!("  G-AGE: no basin of 100+ land nodes to read"),
    }
    // THE PHASES ALONE (the generator may not read a clock, so the bench times them): the
    // climate once, the crater record once, one talus pass, one rebound, the ice once — on the
    // final state, which holds a routing.
    {
        let lattice = MacroLattice::of(body).expect("a lattice");
        let t = Instant::now();
        let c = vd_terrain::climate::climate(body, &lattice, words, &state.z, Some(state.sea_z));
        let t_climate = t.elapsed();
        // The crater record reads the surface's own retention age, so the bench builds the rows it
        // reads — the initial land and the climate over it — outside the clock.
        let land = initial_land(body, &lattice, &words.land());
        let pre = vd_terrain::climate::climate(body, &lattice, words, &land.z, land.sea_z);
        let surface = vd_terrain::craters::Surface {
            crust: &land.crust,
            boundary_m: &land.boundary_m,
            plate: &land.plate,
            plates: &land.plates,
            province: &land.province,
            rain_mm_yr: &pre.rain_mm_yr,
        };
        let t = Instant::now();
        let craters = vd_terrain::craters::crater_population(body, &lattice, words, &surface);
        let t_population = t.elapsed();
        let mut scratch = state.z.clone();
        let t = Instant::now();
        vd_terrain::craters::apply_craters(
            &mut scratch,
            &lattice,
            &craters,
            body.facts().gravity_mm_s2,
        );
        let t_apply = t.elapsed();
        drop(scratch);
        let mut probe = state.clone();
        let tan: Vec<u32> = c
            .aridity_q8
            .iter()
            .map(|&a| vd_terrain::solve::tan_repose_q16(a))
            .collect();
        let t = Instant::now();
        probe.talus(&tan);
        let t_talus = t.elapsed();
        let t = Instant::now();
        probe.rebound(vd_terrain::land::flexural_parameter_m(
            words.elastic_thickness_m,
            body.facts().gravity_mm_s2,
        ));
        let t_rebound = t.elapsed();
        let t = Instant::now();
        probe.ice(
            &c.ela_z,
            &c.temperature_dk,
            c.lapse_mk_km,
            body.facts().gravity_mm_s2,
            vd_terrain::solve::GLACIAL_EPOCH_YR,
        );
        let t_ice = t.elapsed();
        println!(
            "  phases alone: climate {:.3} s, crater population {:.3} s ({} craters), craters applied {:.3} s, \
             one talus pass {:.3} s, one rebound {:.3} s, the ice {:.3} s",
            t_climate.as_secs_f64(),
            t_population.as_secs_f64(),
            craters.len(),
            t_apply.as_secs_f64(),
            t_talus.as_secs_f64(),
            t_rebound.as_secs_f64(),
            t_ice.as_secs_f64()
        );
    }
    if std::env::var_os("VD_BENCH_K0_SCAN").is_some() {
        for factor in [0.1, 0.01, 0.001, 10.0] {
            let scan = Schedule {
                k0_per_yr: schedule.k0_per_yr * factor,
                ..schedule
            };
            let t = Instant::now();
            let Some((_, _, r)) = solve_full(body, words, scan) else {
                continue;
            };
            match age_gate(&r.integrals) {
                Some((median, mature)) => println!(
                    "  K0 × {factor}: {:.3} s; median integral {:.2} over {} basins — {}",
                    t.elapsed().as_secs_f64(),
                    f64::from(median) / 256.0,
                    r.integrals.len(),
                    if mature { "mature" } else { "not mature" }
                ),
                None => println!("  K0 × {factor}: no basin to read"),
            }
        }
    }
}

/// ★ THE TRACE (`VD_BENCH_TRACE=1`): the full schedule replayed pass by pass in the open, with the
/// field's extremes after every step — the instrument that says WHICH pass runs away when the
/// whole solve does.
fn bench_trace(label: &str, body: &BodyDefinition, words: &SolveWords) {
    let schedule = Schedule::standard(words.age_yr);
    let Some(lattice) = MacroLattice::of(body) else {
        return;
    };
    let land = initial_land(body, &lattice, &words.land());
    let Some(mut state) = MacroSolve::from_land(body, &land) else {
        return;
    };
    let gravity = body.facts().gravity_mm_s2;
    let extremes = |state: &MacroSolve, what: &str| {
        let (lo, hi) = state.range();
        println!(
            "  trace {label}: {what}: z {:.0}..{:.0} m",
            metres(lo),
            metres(hi)
        );
    };
    extremes(&state, "the land");
    let pre = vd_terrain::climate::climate(body, &lattice, words, &land.z, land.sea_z);
    let surface = vd_terrain::craters::Surface {
        crust: &land.crust,
        boundary_m: &land.boundary_m,
        plate: &land.plate,
        plates: &land.plates,
        province: &land.province,
        rain_mm_yr: &pre.rain_mm_yr,
    };
    let craters = vd_terrain::craters::crater_population(body, &lattice, words, &surface);
    drop(pre);
    vd_terrain::craters::apply_craters(&mut state.z, &lattice, &craters, gravity);
    let relief = vd_terrain::solve::envelope_steps(body);
    for z in &mut state.z {
        *z = (*z).clamp(-relief, relief);
    }
    for (u, &z) in state.uplift.iter_mut().zip(&state.z) {
        *u = (*u).clamp(-relief - z, relief - z);
    }
    state.z_flood.clone_from(&state.z);
    extremes(&state, "the craters, clipped");
    let gain = schedule.gain();
    let alpha = vd_terrain::land::flexural_parameter_m(words.elastic_thickness_m, gravity);
    let mut climate = None;
    for pass in 0..schedule.passes {
        if pass % schedule.climate_every == 0 {
            let c = vd_terrain::climate::climate(body, &lattice, words, &state.z, land.sea_z);
            state.rain.clone_from(&c.rain_mm_yr);
            climate = Some(c);
            let r = state.route();
            state.accumulate();
            println!(
                "  trace {label}: pass {pass} climate+routing: outlets {}, lakes {}, flat {}, undrained {}",
                r.outlets, r.raised, r.flat, r.undrained
            );
        }
        let s = state.sweep(gain, pass, schedule.passes);
        let (lo, hi) = state.range();
        println!(
            "  trace {label}: pass {pass} sweep: lowered {}, max cut {:.1} m, z {:.0}..{:.0} m",
            s.lowered,
            metres(s.max_cut),
            metres(lo),
            metres(hi)
        );
        if (pass + 1) % schedule.isostasy_every == 0 {
            let (level, lifted, max_lift) = state.rebound(alpha);
            let (lo, hi) = state.range();
            println!(
                "  trace {label}: pass {pass} rebound level {level}: lifted {lifted}, max lift {:.1} m, z {:.0}..{:.0} m",
                metres(max_lift),
                metres(lo),
                metres(hi)
            );
        }
    }
    let climate = climate.expect("a climate");
    let tan: Vec<u32> = climate
        .aridity_q8
        .iter()
        .map(|&a| vd_terrain::solve::tan_repose_q16(a))
        .collect();
    for k in 0..schedule.talus_passes {
        let (shed, worst) = state.talus(&tan);
        let (lo, hi) = state.range();
        println!(
            "  trace {label}: talus {k}: shed {shed}, worst excess {:.1} m, z {:.0}..{:.0} m",
            worst as f64 / f64::from(Z_STEPS_PER_M),
            metres(lo),
            metres(hi)
        );
    }
    let (_, under, thickest, deepest) = state.ice(
        &climate.ela_z,
        &climate.temperature_dk,
        climate.lapse_mk_km,
        gravity,
        vd_terrain::solve::GLACIAL_EPOCH_YR,
    );
    println!(
        "  trace {label}: ice: {under} nodes, thickest {thickest:.0} m, deepest cut {:.1} m",
        metres(deepest)
    );
    extremes(&state, "after the ice");
}

/// One body's bench: the phases timed, the schedule timed, the memory read.
fn bench(label: &str, body: &BodyDefinition, age_yr: u64) {
    let Some(lattice) = MacroLattice::of(body) else {
        println!("{label}: NO LATTICE (the divisor rule found no edge)");
        return;
    };
    let nodes = lattice.node_count();
    println!(
        "\n{label}: seed {}, ladder radius {:.1} km, N {}, edge {}, node {} m, {nodes} nodes",
        body.seed(),
        body.ladder().radius_m() / 1000.0,
        body.ladder().n,
        lattice.edge,
        lattice.cells_per_node
    );
    let rss_before = peak_rss_bytes();
    let t = Instant::now();
    let mut state = MacroSolve::new(body).expect("a state");
    let t_new = t.elapsed();
    let (lo0, hi0) = state.range();
    let t = Instant::now();
    let route = state.route();
    let t_route = t.elapsed();
    let t = Instant::now();
    state.accumulate();
    let t_acc = t.elapsed();
    let schedule = Schedule::standard(age_yr);
    let t = Instant::now();
    let sweep = state.sweep(schedule.gain(), 0, 1);
    let t_sweep = t.elapsed();
    let rss_phases = peak_rss_bytes();
    drop(state);
    println!(
        "  phases: new {:.3} s, route {:.3} s, accumulate {:.3} s, sweep {:.3} s; peak RSS {:.0} MB \
         ({:.0} MB over the start, {:.1} B/node)",
        t_new.as_secs_f64(),
        t_route.as_secs_f64(),
        t_acc.as_secs_f64(),
        t_sweep.as_secs_f64(),
        mb(rss_phases),
        mb(rss_phases.saturating_sub(rss_before)),
        rss_phases.saturating_sub(rss_before) as f64 / nodes as f64
    );
    println!(
        "  route: sea seeded {}, outlets {}, raised (lake nodes) {}, flat {}, undrained {}, cyclic {}",
        route.sea_seeded, route.outlets, route.raised, route.flat, route.undrained, route.cyclic
    );
    println!(
        "  first sweep: lowered {}, skipped lakes {}, max cut {:.2} m, total cut {:.0} m",
        sweep.lowered,
        sweep.skipped_lake,
        metres(sweep.max_cut),
        sweep.total_cut as f64 / f64::from(Z_STEPS_PER_M)
    );
    let t = Instant::now();
    let (state, report) = solve(body, schedule).expect("a solve");
    let t_solve = t.elapsed();
    let rss_solve = peak_rss_bytes();
    let (lo, hi) = state.range();
    let lakes = report.routes.last().map_or(0, |r| r.raised);
    let flats = report.routes.last().map_or(0, |r| r.flat);
    let undrained: usize = report.routes.iter().map(|r| r.undrained + r.cyclic).sum();
    let last = report.sweeps.last().copied().unwrap_or_default();
    println!(
        "  THE SCHEDULE ({} passes, a routing every {}): {:.3} s wall on one thread; peak RSS {:.0} MB; \
         state {} B/node",
        schedule.passes,
        schedule.flood_every,
        t_solve.as_secs_f64(),
        mb(rss_solve),
        state.bytes_per_node()
    );
    println!(
        "  the shape: relief {:.0}..{:.0} m before, {:.0}..{:.0} m after; last routing: {lakes} lake \
         nodes, {flats} flat nodes; undrained or cyclic over the schedule: {undrained}; last sweep \
         lowered {} nodes by at most {:.2} m",
        metres(lo0),
        metres(hi0),
        metres(lo),
        metres(hi),
        last.lowered,
        metres(last.max_cut)
    );
}

fn main() -> ExitCode {
    let max_nodes = std::env::var("VD_BENCH_MAX_NODES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_NODES);
    let age_yr = (vd_physics::taxonomy::SYSTEM_AGE_GYR * 1.0e9) as u64;
    println!(
        "macro_bench: THE SOLVE'S CORE on THE world's bodies; erosional age {age_yr} yr (the census's \
         system age); bodies up to {max_nodes} nodes (VD_BENCH_MAX_NODES)"
    );
    let Some(home) = vd_bins::home_body(DEV.universe_seed) else {
        println!("macro_bench: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    let only: Option<u64> = std::env::var("VD_BENCH_BODY")
        .ok()
        .and_then(|v| v.parse().ok());
    let wanted = |seed: u64| only.is_none_or(|s| s == seed);
    if wanted(home_moon().seed()) {
        bench("the home moon", &home_moon(), age_yr);
        bench_land(
            "the home moon",
            &home_moon(),
            &vd_terrain::home::home_moon_land_words(),
            age_yr,
        );
        bench_full(
            "the home moon",
            &home_moon(),
            &vd_terrain::home::home_moon_solve_words(),
        );
    }
    // The home system's other planets the ladder accepts, smallest first.
    let config = UniverseConfig::world(DEV.move_speed, DEV.tick_dt);
    let system = vd_core::worldgen::HOME_SYSTEM;
    let held = std::collections::BTreeSet::from([system]);
    let lineage = std::collections::BTreeSet::from([vd_core::worldgen::GALAXY]);
    let (rows, _) = shard_boot_world(DEV.universe_seed, &config, &held, system, &lineage);
    let mut planets: Vec<(u64, f64)> = rows
        .iter()
        .filter(|r| r.parent == Some(system))
        .filter_map(|r| match (r.realm, r.look) {
            (RealmId::Planet(seed), Some(vd_core::geometry::Boundary::Shell { r: look })) => {
                Some((seed, look))
            }
            _ => None,
        })
        .collect();
    planets.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
    for (seed, look) in planets {
        if !wanted(seed) {
            continue;
        }
        let realm = RealmId::Planet(seed);
        let Some(facts) = body_facts_in_subtree(DEV.universe_seed, &config, &held, &lineage, realm)
        else {
            continue;
        };
        let [g, rho] =
            vd_physics::worldgen::relief_words(facts.taxon.mass_kg, facts.taxon.radius_m);
        let Some(body) = g.zip(rho).and_then(|(g, rho)| {
            BodyDefinition::from_seed(seed, look, vd_terrain::BodyFacts::new(g, rho))
        }) else {
            println!("\nPlanet({seed}): the ladder refuses the look {look:.0} m");
            continue;
        };
        let label = if body == home {
            "THE HOME PLANET".to_owned()
        } else {
            format!("Planet({seed}) ({:?})", facts.taxon.class)
        };
        match MacroLattice::of(&body) {
            Some(l) if l.node_count() > max_nodes => println!(
                "\n{label}: {} nodes at edge {} — over the bench's ceiling, skipped",
                l.node_count(),
                l.edge
            ),
            _ => {
                bench(&label, &body, age_yr);
                let charter = vd_physics::worldgen::body_charter_in_subtree(
                    DEV.universe_seed,
                    &config,
                    &held,
                    &lineage,
                    realm,
                );
                match charter.and_then(|c| c.elastic_thickness_m.map(|te| (c, te))) {
                    Some((c, elastic_thickness_m)) => {
                        bench_land(
                            &label,
                            &body,
                            &LandWords {
                                water_km3: c.water_km3.unwrap_or(0),
                                elastic_thickness_m,
                            },
                            age_yr,
                        );
                        let words = vd_bins::solve_words_of(&c);
                        if std::env::var_os("VD_BENCH_TRACE").is_some() {
                            bench_trace(&label, &body, &words);
                        } else {
                            bench_full(&label, &body, &words);
                        }
                    }
                    None => println!(
                        "  no charter words for the land (no elastic thickness): C2 skipped"
                    ),
                }
            }
        }
    }
    ExitCode::SUCCESS
}
