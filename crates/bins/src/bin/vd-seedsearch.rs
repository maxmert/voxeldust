//! ★ THE SEED-SEARCH TOOL (celestial taxonomy arc T4; T4b made it a DECISION INSTRUMENT). The
//! owner's standing search for a yellow-sun home. NOT wired into any boot: a report-only
//! instrument that READS THE world at candidate seeds through the ONE generator (SL5 — no second
//! generator, no variant, no world knob) and prints every Earth-like candidate with its numbers.
//!
//! THE SL5 FIREWALL: the ranking weights below are TOOL policy — they live HERE, never in
//! `UniverseConfig`, and never reach the generator (a seed-selection instrument may have
//! knobs; THE world may not — `the_ranking_weights_are_absent_from_the_one_config` pins it).
//!
//! Usage: `vd-seedsearch [start_seed] [count] [out.md]` — defaults: seed 0, the derived sweep
//! size, `home_candidates.md` in the working directory. The markdown file carries the FULL
//! candidate table; stdout carries the header, the measured rate and the top rows.

use vd_physics::taxonomy::{M_EARTH_KG, R_EARTH_M};
use vd_physics::worldgen::EarthLikeCandidate;

/// ★ THE DERIVED SWEEP (T4b). The T4 ruling-F run MEASURED the rate on THE world: 17 seeds of
/// 3072 hold an Earth-like body, i.e. one candidate every `3072/17 = 180.7` seeds (the design
/// predicted ~1 in 192 — §8.3). The owner reads a TOP-TEN shortlist, and a ranking only means
/// something when the pool it selects from is several times deeper than the shortlist it feeds;
/// FOUR times is the shallowest depth at which a "top ten" is a real decile. So the sweep is sized
/// for `4 × 10 = 40` candidates at the MEASURED rate:
///
/// `ceil(40 × 3072 / 17) = 7229` seeds.
///
/// Every number in that expression is a measurement or the shortlist the owner asked for — there
/// is no rounder number chosen by hand.
const MEASURED_SWEEP: u64 = 3072;
const MEASURED_HITS: u64 = 17;
const SHORTLIST: u64 = 10;
const POOL_DEPTH_PER_SHORTLIST_ROW: u64 = 4;
const TARGET_CANDIDATES: u64 = SHORTLIST * POOL_DEPTH_PER_SHORTLIST_ROW;
const DERIVED_SWEEP: u64 = (TARGET_CANDIDATES * MEASURED_SWEEP).div_ceil(MEASURED_HITS);

/// TOOL-POLICY ranking weights (§8.4 as extended by T4b). Every term is a NATURAL-LOG DISTANCE
/// from the reference home — Earth around the Sun — so the terms are already commensurable and
/// the weights stay at one. What the expression optimizes, stated:
///
/// * `|ln(R/R⊕)|` + `|ln(M/M⊕)|` — an Earth-SIZED and Earth-MASSED world. Together these two also
///   pin surface gravity (`g ∝ M/R²`), so no third term is needed for how heavy it feels.
/// * `|ln(M★/M☉)|` + `|ln(L★/L☉)|` — a Sun-MASSED star with a Sun-BRIGHT sky. Mass fixes the
///   class and the orbit ladder; luminosity is what the sky actually looks like, and the two are
///   not interchangeable (the mass-luminosity relation is steep).
/// * `|ln(T_eq/T⊕)|` — temperate at EARTH's own equilibrium temperature, derived through the very
///   expression the predicate's band uses (`earth_like_t_bound_k(1.0)`), never a transcribed 254 K.
/// * `−ln(1 + planets + moons)` — the ONLY reward term: a system with more worlds in it is more to
///   fly to. It is a log too, so a richer neighbour system can trade against a small Earth-likeness
///   penalty but can never swamp it.
///
/// DELIBERATELY ABSENT. Insolation carries no information (quantised and seed-free — §8.2). The
/// neighbour-star geometry is REPORTED but NOT RANKED: whether a near first warp or a far one is
/// better is the owner's taste, not a measurement, and the table prints both so they can re-rank.
const W_RADIUS: f64 = 1.0;
const W_MASS: f64 = 1.0;
const W_STAR_MASS: f64 = 1.0;
const W_STAR_LUMA: f64 = 1.0;
const W_TEMP: f64 = 1.0;
const W_RICHNESS: f64 = 1.0;

/// The light-year, from the two SI DEFINITIONS it is made of (the metre via `c`, the Julian year):
/// `299 792 458 m/s × 365.25 × 86 400 s`. A unit conversion, not a tuning number — the owner's
/// star gaps are quoted in light-years everywhere else in the arc.
const LIGHT_YEAR_M: f64 = 299_792_458.0 * 365.25 * 86_400.0;

fn main() {
    let mut args = std::env::args().skip(1);
    let start: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    // An EXPLICIT count sweeps exactly that many seeds. The DEFAULT sweeps the derived block and
    // then EXTENDS at the rate this run itself measured, until the target pool is full: the
    // requirement is the POOL DEPTH the owner reads from, and how many seeds that costs is the
    // world's own answer, not a number anyone picks. (The T4 rate sized the first block; a bigger
    // sample measures a slightly different rate, and re-deriving from it is what closes the gap.)
    let explicit: Option<u64> = args.next().and_then(|a| a.parse().ok());
    let out_path = args
        .next()
        .unwrap_or_else(|| "home_candidates.md".to_owned());
    let config = vd_bins::process_world_config(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    // Earth's own equilibrium temperature under THIS world's one law — the rank's reference.
    let t_earth_k = vd_physics::worldgen::earth_like_t_bound_k(1.0);

    let mut hits: Vec<(f64, u64, EarthLikeCandidate)> = Vec::new();
    let mut seeds_with_hits = 0u64;
    let mut count = 0u64;
    let mut block = explicit.unwrap_or(DERIVED_SWEEP);
    // The round cap is the target itself: each round is sized to find at least the shortfall, so
    // in the worst case it finds one candidate per round and still terminates.
    for _round in 0..TARGET_CANDIDATES {
        for seed in (start + count)..(start + count + block) {
            let found = vd_physics::worldgen::earth_like_candidates(seed, &config);
            if !found.is_empty() {
                seeds_with_hits += 1;
            }
            for c in found {
                hits.push((rank_of(&c, t_earth_k), seed, c));
            }
        }
        count += block;
        let have = hits.len() as u64;
        if explicit.is_some() || have >= TARGET_CANDIDATES {
            break;
        }
        // Re-derive the next block from THIS sweep's own measured period, for the shortfall.
        let measured_period = count.max(1) / have.max(1);
        block = (TARGET_CANDIDATES - have).max(1) * measured_period.max(1);
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));

    let header = format!(
        "[seedsearch] swept seeds {start}..{} ({count}): {seeds_with_hits} seeds hold an \
         Earth-like world ({} candidate bodies) — 1 in {:.1}. The sweep is DERIVED: the first \
         block is ceil({TARGET_CANDIDATES} × {MEASURED_SWEEP} / {MEASURED_HITS}) = \
         {DERIVED_SWEEP} seeds (the T4 measured rate 17/3072, for a pool \
         {POOL_DEPTH_PER_SHORTLIST_ROW}× deeper than the {SHORTLIST}-row shortlist), then \
         extended at THIS run's own measured rate until the {TARGET_CANDIDATES}-candidate pool \
         was full.",
        start + count,
        hits.len(),
        if hits.is_empty() {
            f64::INFINITY
        } else {
            count as f64 / hits.len() as f64
        },
    );
    println!("{header}");
    println!(
        "[seedsearch] NOTE (§8.2): insolation and T_eq are structural no-ops given rocky+G \
         (the ladder is quantised and seed-free); the real discriminants are the star's class, \
         its luminosity, the planet's drawn mass and the system's own census."
    );
    println!(
        "[seedsearch] rank (LOWER IS BETTER) = |ln R/R⊕| + |ln M/M⊕| + |ln M★/M☉| + |ln L★/L☉| \
         + |ln T_eq/T⊕| − ln(1+planets+moons), at T⊕ = {t_earth_k:.3} K. The neighbour-star \
         geometry is reported, NOT ranked."
    );
    println!("[seedsearch] THE TOP {SHORTLIST} (the full table is in {out_path}):");
    for (i, (rank, seed, c)) in hits.iter().take(SHORTLIST as usize).enumerate() {
        println!("{}", verbose_row(i + 1, *rank, *seed, c, &config));
    }

    let mut md = String::new();
    md.push_str("# Home-seed candidates — THE world, read at every swept seed\n\n");
    md.push_str(&header.replace("[seedsearch] ", ""));
    md.push_str("\n\n");
    md.push_str(
        "**The rank (lower is better).** `|ln R/R⊕| + |ln M/M⊕| + |ln M★/M☉| + |ln L★/L☉| + \
         |ln T_eq/T⊕| − ln(1 + planets + moons)`. Every term is a natural-log distance from \
         Earth-around-the-Sun, so they are commensurable and the weights are all one. The last \
         term is the only reward: a system with more worlds in it is more to fly to. Insolation \
         is absent (quantised and seed-free). **The neighbour-star geometry is reported but NOT \
         ranked** — near or far is the owner's taste, so re-rank on those columns freely.\n\n",
    );
    md.push_str(&format!(
        "Earth's reference equilibrium temperature under this world's one law: **{t_earth_k:.4} K** \
         (`earth_like_t_bound_k(1.0)`). Nothing here changes the world seed; this is a report.\n\n",
    ));
    md.push_str(&format!(
        "**Provenance — the world this table reads.** The stellar mass draw's upper bound is \
         DERIVED (owner ruling 2026-08-20): **{:.15} M☉**, the largest star this galaxy can host, \
         with a reserved system bound of **{:.1} m** and a placement radius of **{:.1} m**. That \
         bound enters the bounded power law `sample_imf_mass` inverts, so EVERY star in the world \
         moved when it landed and this table supersedes any earlier one. The `boots` column exists \
         for the same reason: the reservation used to be a sample of seed 0's own population, and \
         a seed drawing a heavier star than seed 0 did could not start its galaxy at all.\n\n",
        vd_physics::worldgen::imf_mass_hi_msun(),
        vd_physics::worldgen::target_system_bound_max_m(),
        config.stellar.galaxy_rim_r_m,
    ));
    md.push_str(
        "| # | rank | seed | boots | star class | M★ (M☉) | L★ (L☉) | planet | M (M⊕) | \
         R (R⊕) | R (km) | ρ (kg/m³) | g (m/s²) | class | T_eq (K) | S (S⊕) | air | planets | \
         moons | own moons | siblings | nearest (ly) | farthest (ly) | first warp (s) |\n",
    );
    md.push_str(
        "|--:|--:|--:|:-:|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|--:|--:|:-:|--:|--:|--:|--:|--:|--:|--:|\n",
    );
    for (i, (rank, seed, c)) in hits.iter().enumerate() {
        md.push_str(&md_row(i + 1, *rank, *seed, c, &config));
    }
    md.push_str("\n## What each column is\n\n");
    md.push_str(
        "* **star class / M★ / L★** — the drawn star: its Morgan-Keenan class, its mass in solar \
         masses, its main-sequence luminosity in solar luminosities.\n\
         * **M / R / ρ / g** — the candidate world's drawn mass, its Chen–Kipping radius, the \
         density those two imply (`M / (4/3·π·R³)`) and its surface gravity.\n\
         * **class** — the derived composition verdict (the predicate admits Rocky only).\n\
         * **T_eq / S / air** — equilibrium temperature at its Bond albedo, insolation relative to \
         Earth, and whether the cosmic-shoreline and Jeans verdicts left it an atmosphere.\n\
         * **planets / moons** — how many worlds the candidate's own star system holds in total; \
         **own moons** is how many that planet keeps.\n\
         * **siblings / nearest / farthest** — the other star systems of this galaxy and the 3-D \
         distance to the closest and farthest of them: the first journey out, and the far corner.\n\
         * **boots** — whether that seed's galaxy actually STARTS: the four world-deriving boot \
         fences every process runs, in the same order (the nesting fence, the measured visibility \
         climb against the shipped look carrier, the FINE-lattice storage budget, and the 3-D \
         seeded-system separation). `yes` means the galaxy shard comes up; anything else names the \
         fence that refused. This column exists because a home seed that cannot boot its own \
         galaxy leaves the pilot drifting outside their star system with no star map at all — the \
         defect measured on 2026-08-20.\n\
         * **first warp** — the GOVERNED closed-form time (`vd_core::flight::leg_time_s`) to fly \
         the nearest gap at the galaxy's own speed ceiling, starting and ending at foot speed. It \
         is what the trip costs in seconds, printed so the geometry means something.\n",
    );
    match std::fs::write(&out_path, md) {
        Ok(()) => println!("[seedsearch] wrote the full table to {out_path}"),
        Err(e) => println!("[seedsearch] FAILED to write {out_path}: {e}"),
    }
}

/// THE DESIRABILITY EXPRESSION (documented on the weights above). Lower is better.
fn rank_of(c: &EarthLikeCandidate, t_earth_k: f64) -> f64 {
    W_RADIUS * (c.radius_m / R_EARTH_M).ln().abs()
        + W_MASS * (c.mass_kg / M_EARTH_KG).ln().abs()
        + W_STAR_MASS * c.star_mass_msun.ln().abs()
        + W_STAR_LUMA * c.star_luma_lsun.ln().abs()
        + W_TEMP * (c.t_eq_k / t_earth_k).ln().abs()
        - W_RICHNESS * (1.0 + f64::from(c.system_planets + c.system_moons)).ln()
}

/// ★ DOES THAT SEED'S GALAXY ACTUALLY START? (owner ruling 2026-08-20 — the whole reason this
/// tool now has a `boots` column.) The FOUR world-deriving boot fences every process runs, in the
/// order the gateway and the shard run them, against the SAME world config. `yes`, or the name of
/// the fence that refused with its own words — never a bare `no`.
fn boot_verdict(seed: u64, config: &vd_physics::worldgen::UniverseConfig) -> String {
    if let Err(e) = vd_physics::worldgen::guard_world_nests(seed, config) {
        return format!("NO — nest: {e}");
    }
    if let Err(e) = vd_physics::worldgen::guard_visibility_climb_bounded(
        seed,
        config,
        vd_wire::session_flow::LOOK_CARRIER_ARITY,
    ) {
        return format!("NO — climb: {e}");
    }
    if let Err(e) = vd_physics::worldgen::guard_root_representable(config) {
        return format!("NO — storage: {e}");
    }
    if let Err(e) = vd_physics::worldgen::guard_seeded_systems_disjoint(seed, config) {
        return format!("NO — separation: {e:?}");
    }
    "yes".to_owned()
}

/// The candidate's density, from the two drawn numbers it already carries.
fn density_kg_m3(c: &EarthLikeCandidate) -> f64 {
    c.mass_kg / (4.0 / 3.0 * std::f64::consts::PI * c.radius_m.powi(3))
}

/// The GOVERNED closed-form time to fly the nearest star gap at the galaxy's own ceiling, from
/// foot speed to foot speed — the speed law's own expression, not a second model. `None` when the
/// gap cannot hold both ramps (`leg_time_s`'s own refusal).
fn first_warp_s(
    c: &EarthLikeCandidate,
    config: &vd_physics::worldgen::UniverseConfig,
) -> Option<f64> {
    // The DEV cluster's own cadence, derived exactly as every gate derives it (half the tick rate).
    let aoi_cadence_ticks = (((1.0 / vd_bins::DEV.tick_dt).round() as u64) / 2).max(1);
    let tuning = vd_core::flight::FlightTuning::derive(
        vd_bins::DEV.move_speed,
        vd_bins::DEV.tick_dt,
        aoi_cadence_ticks,
        u32::try_from(vd_bins::DEV.boot_ticks_p99).unwrap_or(u32::MAX),
    );
    let galaxy_cap = vd_core::flight::realm_speed_cap_mps(
        config.scale.galaxy_r_m,
        tuning.v_foot_mps,
        tuning.traverse_s,
    );
    vd_core::flight::leg_time_s(
        c.nearest_sibling_m,
        galaxy_cap,
        tuning.v_foot_mps,
        tuning.v_foot_mps,
        tuning.tau_s,
    )
}

/// One stdout row — the whole picture on two lines, for the owner to read without opening a file.
fn verbose_row(
    n: usize,
    rank: f64,
    seed: u64,
    c: &EarthLikeCandidate,
    config: &vd_physics::worldgen::UniverseConfig,
) -> String {
    let boots = boot_verdict(seed, config);
    format!(
        "  {n:>2}. seed={seed} rank={rank:.4}\n      star {:?} {:.4} M☉, {:.4} L☉  |  world {:?} \
         {:.3} M⊕, {:.3} R⊕ ({:.1} km), ρ {:.0} kg/m³, g {:.2} m/s², {:?}, T_eq {:.2} K, \
         S {:.6} S⊕, air {}\n      system: {} planets, {} moons (this world keeps {})  |  \
         neighbours: {} sibling stars, nearest {:.4} ly, farthest {:.4} ly  |  in {:?}  |  \
         galaxy boots: {}",
        c.star_class,
        c.star_mass_msun,
        c.star_luma_lsun,
        c.body,
        c.mass_kg / M_EARTH_KG,
        c.radius_m / R_EARTH_M,
        c.radius_m / 1.0e3,
        density_kg_m3(c),
        vd_physics::taxonomy::surface_gravity_mps2(c.mass_kg, c.radius_m),
        c.planet_class,
        c.t_eq_k,
        c.insolation_rel,
        if c.has_atmosphere { "yes" } else { "no" },
        c.system_planets,
        c.system_moons,
        c.own_moons,
        c.sibling_count,
        c.nearest_sibling_m / LIGHT_YEAR_M,
        c.farthest_sibling_m / LIGHT_YEAR_M,
        c.system,
        boots,
    )
}

/// One markdown table row.
fn md_row(
    n: usize,
    rank: f64,
    seed: u64,
    c: &EarthLikeCandidate,
    config: &vd_physics::worldgen::UniverseConfig,
) -> String {
    let warp = first_warp_s(c, config).map_or_else(|| "—".to_owned(), |s| format!("{s:.1}"));
    let boots = boot_verdict(seed, config);
    format!(
        "| {n} | {rank:.4} | {seed} | {boots} | {:?} | {:.4} | {:.4} | `{:?}` | {:.3} | {:.3} | \
         {:.1} | {:.0} | {:.2} | {:?} | {:.2} | {:.6} | {} | {} | {} | {} | {} | {:.4} | {:.4} | \
         {warp} |\n",
        c.star_class,
        c.star_mass_msun,
        c.star_luma_lsun,
        c.body,
        c.mass_kg / M_EARTH_KG,
        c.radius_m / R_EARTH_M,
        c.radius_m / 1.0e3,
        density_kg_m3(c),
        vd_physics::taxonomy::surface_gravity_mps2(c.mass_kg, c.radius_m),
        c.planet_class,
        c.t_eq_k,
        c.insolation_rel,
        if c.has_atmosphere { "yes" } else { "no" },
        c.system_planets,
        c.system_moons,
        c.own_moons,
        c.sibling_count,
        c.nearest_sibling_m / LIGHT_YEAR_M,
        c.farthest_sibling_m / LIGHT_YEAR_M,
    )
}
