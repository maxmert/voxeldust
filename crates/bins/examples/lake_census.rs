//! ★ THE LAKE CENSUS (the owner, 2026-09-22, from 1 448 km over the belt: "It's full of patchy lakes,
//! you never would see anything like this in the real world"): the home planet is solved once and
//! its water facies are COUNTED — sea, lake and dry land per node; the lakes' share of the land
//! against Earth's; every lake as a connected patch of lake nodes with its size; and a map around one
//! direction, one character per node, so the picture the owner flew can be read as text. The octave
//! table is printed beside it, because a lake stands in a hollow and the hollows have a wavelength.
//!
//! ★ STEP 1 OF THE PLAN — THE WATER POSITIONS AS NUMBERS (ruling B6, 2026-09-24; the owner, after
//! the flight of that day: *"water positions for lake and rivers is not believable"*). WHY: no gate
//! saw that complaint (`docs/investigation/2026-09-24/terrain_believability_status.md` §4, the blind
//! spots), so the plan measures before it changes the world. The census now also prints these
//! readings. Each one can go red, and each one moves nothing:
//!
//! 1. **G-HYPSO** (ruling B3): the final field's heights over the sea in fixed bins, the two-hump
//!    test of gate G-LAND-HYPSOMETRY, the basins' G-AGE integrals against Strahler's band, and the
//!    land's share over 1 000 m.
//! 2. **THE INLAND POCKETS** (status §6 T7; ruling B7 item 1): the nodes under the sea's level that no
//!    path joins to the ocean — closed basins since plan step 2 — with the census's own labelling
//!    checked against the solve's ocean (CHECK-OCEAN, CHECK-POCKET-NOT-SEA) and each pocket's end.
//! 3. **G-RIVER** (ruling B3): every channel head walked down the shipped receivers to its end, and
//!    every trunk step on which the floor rises. Beside it, THE DISCHARGE ROW AND ITS TREE (status §7
//!    W10): the row, which carries the RUNOFF since plan step 2, against the receivers it ships with.
//! 4. ★ THE LAKES AT THE RUNG SWAPS (G-RUNG-LAKE, G-RUNG-LIP, G-RUNG-HOLD, G-POCKET-HELD) LEFT WITH
//!    RULING B22 (2026-09-28): they read the drawn water and the shore law on the shipped column,
//!    and no water is drawn and no law holds a column any more. The one-volume fill step brings its
//!    own gates. THE STORED LEVEL-1 WATER WORD stays checked (CHECK-LEVEL1-WORD): it reads the
//!    solve's own data, which the fill step reads.
//! 5. **THE LAKE BUDGET** (status §7 W3, W4; plan step 2 parts 1 and 2): each hollow's watershed
//!    supply beside the runoff of its own nodes, the three ends, the cascade's pours and its books
//!    (exact), and G-LAKE-SHARE. Step 1 printed this as a WHAT-IF; since step 2 it is the budget.
//!    ★ And THE ROW AT THE SPILLS (review finding 3 of plan step 2): the discharge row at each
//!    hollow's spill against the water the cascade passes on there.
//!
//! WHAT IT READS: the solve's final state (the heights, the receivers, the routing fill, the ocean,
//! the discharge row), the lake budget's own hierarchy and descent (`lakes::budget`, run again over
//! the same climate), and the shipped artifact's rows. It never re-implements a column. Every gate
//! prints one
//! `CENSUS <gate> <value> <band> <verdict>` line with no space inside a field; every self-check
//! prints a `CENSUS CHECK-…` line of the same shape; and the world's versions head the output, so a
//! reading names its world.
//!
//! **Example.** The solve holds a pond in a hollow of the belt. The census counts the pond's nodes
//! and its budget, and walks the river off the range beside it down the shipped receivers into a
//! hollow the range's whole watershed now feeds, and the hollow spills.
//!
//! `cargo run --release -p vd-bins --example lake_census -- [dx dy dz] [half_nodes]`

use std::collections::VecDeque;

use vd_bins::grid_bias::unit;
use vd_core::glam::DVec3;
use vd_recipe::height::AMP_BITS;
use vd_recipe::noise::NOISE_BITS;
use vd_terrain::artifact::{
    Artifact, DRY_M, FACIES_SHIFT, Side, block_water, coast_side, log2_class,
};
use vd_terrain::climate::Climate;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::lakes::{Lakes, NO_DEP, OCEAN};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{DRY, FACIES_LAKE, FACIES_SEA, MacroSolve, Z_STEPS_PER_M};

/// Earth's lakes: 117 million lakes over 5.0 million km², 3.7 % of the land outside the ice
/// (Verpoorter, Kutser, Seekell, Tranvik 2014, GRL 41). Most of that area is glacial: the Canadian
/// and Fennoscandian shields the ice sheets scoured. Outside the glaciated shields the share is
/// under one percent.
const EARTH_LAKE_SHARE_PCT: f64 = 3.7;

/// ★ A RIVER'S DRAINAGE THRESHOLD, in nodes. G-GRID's valleys and G-RIVER's channels read this one
/// number, so the two gates count one river set: a land node is on a channel where its DRAINAGE
/// AREA over the shipped receivers is at least this many times its own area. ★ Since plan step 2
/// the area itself, never the discharge row: the row carries the RUNOFF (ruling B7 item 2), and a
/// desert river's runoff is a small share of its rain.
const VALLEY_NODES: u64 = 100;

fn n_of(lattice: &MacroLattice) -> usize {
    lattice.node_count()
}

fn main() {
    // ★ THE WORLD THIS READING NAMES (ruling B6, step 1): the recipe's version and the artifact's
    // version head the output, so no number below is read against another world.
    println!(
        "WORLD generator {} artifact {} (vd_terrain::GENERATOR_VERSION, vd_terrain::artifact::ARTIFACT_VERSION)",
        vd_terrain::GENERATOR_VERSION,
        vd_terrain::artifact::ARTIFACT_VERSION
    );
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let aim = if a.len() >= 3 {
        DVec3::new(a[0], a[1], a[2]).normalize()
    } else {
        DVec3::new(0.617_270, -0.437_286, -0.654_033).normalize()
    };
    let half = a.get(3).map_or(60usize, |v| *v as usize);
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("a macro lattice");
    let radius_m = body.radius_m();
    println!("the octave table (wavelength, amplitude, kind), coarsest first:");
    for (k, o) in body.octave_table().iter().enumerate() {
        let f = o.frequency_int.raw() as f64
            + o.frequency_frac.raw() as f64 / (1u64 << NOISE_BITS) as f64;
        if f <= 0.0 {
            continue;
        }
        let amp_m = o.amplitude.raw() as f64 / (128.0 * (1u64 << AMP_BITS) as f64);
        let kind = if o.kind.raw() == 0 {
            "smooth"
        } else {
            "RIDGED"
        };
        let fine = if o.fine.raw() == 0 { "coarse" } else { "fine" };
        println!(
            "  {k:>2}: {:>10.0} m wavelength, {:>8.1} m amplitude, {kind}, {fine}",
            radius_m / f,
            amp_m
        );
    }
    let node_m = lattice.node_m();
    let node_km2 = node_m * node_m / 1.0e6;
    let words = home_solve_words();
    // The pockets the FIRST land holds, read in the block below (plan question 9).
    let first_land;
    // ★ STEP 2 OF RULING W7 — THE BASINS AGAINST THE INVENTORY (2026-09-22): over the INITIAL land
    // (the isostasy, the belts' uplift not yet applied), the continental PLATFORM is the
    // area-weighted median of the shield's and the shelf's nodes; the ocean FLOOR the same over the
    // deep sediment; the basins' volume is the water that fits under the platform; the inventory
    // is what the census gave the body. The sea overtops the platform by the difference.
    {
        use vd_terrain::land::{LandWords, initial_land};
        use vd_terrain::strata::Province;
        let land = initial_land(
            &body,
            &lattice,
            &LandWords {
                water_km3: words.water_km3,
                elastic_thickness_m: words.elastic_thickness_m,
            },
        );
        first_land = (land.pockets, land.pocket_nodes);
        let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
        let median_of = |pick: &dyn Fn(u8) -> bool| -> Option<f64> {
            let mut rows: Vec<(i32, u64)> = (0..n_of(&lattice))
                .filter(|&i| pick(land.province[i]))
                .map(|i| (land.z[i], lattice.area_m2(i as u32)))
                .collect();
            rows.sort_unstable();
            let total: u64 = rows.iter().map(|r| r.1).sum();
            let mut acc = 0u64;
            for (z, a) in rows {
                acc += a;
                if acc * 2 >= total {
                    return Some(f64::from(z) / steps);
                }
            }
            None
        };
        let platform = median_of(&|p| {
            p == Province::CrystallineBasement.code() || p == Province::FlatShelf.code()
        });
        let floor = median_of(&|p| p == Province::DeepSediment.code());
        if let (Some(platform), Some(floor)) = (platform, floor) {
            let total_area: f64 = (0..n_of(&lattice))
                .map(|i| lattice.area_m2(i as u32) as f64)
                .sum();
            let cont_area: f64 = (0..n_of(&lattice))
                .filter(|&i| land.crust[i] >= 128)
                .map(|i| lattice.area_m2(i as u32) as f64)
                .sum();
            let level = (platform * steps) as i32;
            let basins_m3: f64 = (0..n_of(&lattice))
                .filter(|&i| land.z[i] < level)
                .map(|i| lattice.area_m2(i as u32) as f64 * f64::from(level - land.z[i]) / steps)
                .sum();
            let inventory_m3 = words.water_km3 as f64 * 1.0e9;
            let sea_m = land.sea_z.map_or(f64::NAN, |z| f64::from(z) / steps);
            println!(
                "\nstep 2, the basins against the inventory (the initial land, loaded):\n  continental crust {:.1} % of the area; the platform's median {platform:.0} m, the floor's median {floor:.0} m: a step of {:.0} m\n  the basins under the platform hold {:.3e} km³; the inventory is {:.3e} km³ ({:.2} × the basins); Earth: 1.37e9 km³ over basins of ~1.3e9 km³ (0.99 ×)\n  the solved sea stands {sea_m:.0} m: {:.0} m OVER the platform (Earth: the sea at the shelf edge, 29 % land on 40 % crust)",
                cont_area * 100.0 / total_area,
                platform - floor,
                basins_m3 / 1.0e9,
                inventory_m3 / 1.0e9,
                inventory_m3 / basins_m3,
                sea_m - platform
            );
        }
    }
    let env = |name: &str, default: u32| -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let env_yr = |name: &str, default: u64| -> u64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let mut schedule = vd_terrain::solve::Schedule::standard(
        std::env::var("VD_CENSUS_AGE_YR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(vd_terrain::home::HOME_SYSTEM_AGE_YR),
    );
    // The axis's resolution (ruling B39's gate): the last pass, the longest pass, the sweeps a routing.
    schedule.t_min_yr = env_yr("VD_CENSUS_T_MIN_YR", schedule.t_min_yr);
    schedule.t_max_yr = env_yr("VD_CENSUS_T_MAX_YR", schedule.t_max_yr);
    schedule.sub_sweeps = env("VD_CENSUS_SUB_SWEEPS", schedule.sub_sweeps);
    schedule.talus_passes = env("VD_CENSUS_TALUS_PASSES", schedule.talus_passes);
    println!("schedule: {schedule:?}");
    let started = std::time::Instant::now();
    let (state, facies_v, report) =
        vd_terrain::solve::solve_full(&body, &words, schedule).expect("the home planet solves");
    println!(
        "solved in {:.1} s; the peak law: {} collapses, {} steps left over its cap; ice: {} nodes under the line, thickest {:.0} m, its cut summed {} steps over {} steps of time",
        started.elapsed().as_secs_f64(),
        report.peak.0,
        report.peak.1,
        report.ice.0,
        report.ice.1,
        report.ice.2,
        report.ice_steps
    );
    println!(
        "sweeps: first cut {} steps over {} nodes, last cut {} steps; fills first {} last {} steps",
        report.sweeps[0].total_cut,
        report.sweeps[0].lowered,
        report.sweeps[report.sweeps.len() - 1].total_cut,
        report.sweeps[0].total_fill,
        report.sweeps[report.sweeps.len() - 1].total_fill
    );
    // ★ THE LAKES' BUDGET (ruling W11, 2026-09-22): the depression hierarchy over the final field
    // and every hollow's own water balance. The routing fill decides the receivers and nothing
    // else, so the numbers below say how many hollows actually hold water.
    {
        let l = &report.lakes;
        let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
        // ★ THE LAND IS EVERY NODE THAT IS NOT THE OCEAN (review finding 10 of plan step 2): an
        // inland pocket under the sea's level is land, and its lakes are in the numerator — the
        // same land G-LAKE-SHARE divides by.
        let land_m2: f64 = (0..n_of(&lattice))
            .filter(|&i| !state.ocean[i])
            .map(|i| lattice.area_m2(i as u32) as f64)
            .sum();
        println!(
            "\nthe depression hierarchy: {} depressions ({} leaves); {} water bodies — DRY {} ({:.1} %), PARTIAL {} ({:.1} %), SPILLING {} ({:.1} %)",
            l.depressions,
            l.leaves,
            l.bodies,
            l.dry,
            l.dry as f64 * 100.0 / l.bodies.max(1) as f64,
            l.partial,
            l.partial as f64 * 100.0 / l.bodies.max(1) as f64,
            l.spilling,
            l.spilling as f64 * 100.0 / l.bodies.max(1) as f64
        );
        println!(
            "the standing water: {} nodes, {:.3e} km², {:.4e} km³ ({:.3} % of the inventory {:.4e} km³); the lakes' share of the land (every node that is not the ocean) by AREA {:.3} %",
            l.lake_nodes,
            l.lake_area_m2 as f64 / 1.0e6,
            l.volume_m3 as f64 / 1.0e9,
            l.volume_m3 as f64 / 1.0e9 * 100.0 / words.water_km3 as f64,
            words.water_km3 as f64,
            l.lake_area_m2 as f64 * 100.0 / land_m2.max(1.0)
        );
        match report.sea_before_lakes {
            // ★ In SIXTEENTHS as well (review finding 8 of step 2b): the whole metre hides a shift
            //   the climate's classes can see.
            Some(before) => println!(
                "THE DOUBLE COUNT: the sea was re-solved over the inventory less the lakes: {:.0} m → {:.0} m ({before} → {} sixteenths, {:+})",
                f64::from(before) / steps,
                f64::from(state.sea_z) / steps,
                state.sea_z,
                state.sea_z - before
            ),
            None => println!(
                "THE DOUBLE COUNT: the lakes' volume left the sea's level where it stood ({:.0} m) — under the solve's own sixteenth of a metre",
                f64::from(state.sea_z) / steps
            ),
        }
        println!(
            "THE LAKES' SEA FIXED POINT (plan question 8): {} re-solves, the sea stood still: {}; a connected sea solve re-seeded from another body: {}; a connected sea solve whose ocean does not hold its own level's water (review finding 4): {}",
            report.lake_sea_solves,
            report.lake_sea_converged,
            report.sea_reseeded,
            report.sea_unsettled
        );
    }
    println!(
        "\ncraters stamped: {}; routes: {} (one a pass of {}, plus the final ones); pits RAISED by each flood (= lake nodes at that route), and the pass's {} sweeps summed:",
        report.craters,
        report.routes.len(),
        schedule.times().len() - 1,
        schedule.sub_sweeps
    );
    let sub = schedule.sub_sweeps.max(1) as usize;
    for (k, r) in report.routes.iter().enumerate() {
        let pass_sweeps = report.sweeps.get(k * sub..(k + 1) * sub);
        let sw = pass_sweeps.map(|s| {
            s.iter()
                .fold(vd_terrain::solve::SweepReport::default(), |mut a, b| {
                    a.lowered += b.lowered;
                    a.total_cut += b.total_cut;
                    a.max_cut = a.max_cut.max(b.max_cut);
                    a.skipped_lake += b.skipped_lake;
                    a.filled += b.filled;
                    a.total_fill += b.total_fill;
                    a
                })
        });
        println!(
            "  route {k}: pits {} ({} steps), outlets {} | sweep: lowered {} cut {} max {} skipped {} | filled {} fill {}",
            r.raised,
            r.pit_steps,
            r.outlets,
            sw.map_or(0, |s| s.lowered),
            sw.map_or(0, |s| s.total_cut),
            sw.map_or(0, |s| s.max_cut),
            sw.map_or(0, |s| s.skipped_lake),
            sw.map_or(0, |s| s.filled),
            sw.map_or(0, |s| s.total_fill)
        );
    }
    // ★ THE CLIMATE OVER THE CONNECTED OCEAN, the very one the solve's budget and the artifact read.
    let climate =
        vd_terrain::climate::climate(&body, &lattice, &words, &state.z, Some(state.sea_view()));
    // ★ GATE G-BUZZSAW (ruling B2 step 2; Egholm et al. 2009): no peak on Earth stands more than
    // about 1 500 m over its LOCAL snowline, under every tectonic style. A free reading over two
    // fields we already hold — the solved ground and the climate's own equilibrium line.
    {
        let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
        let mut worst = f64::MIN;
        let mut over = 0usize;
        let mut land = 0usize;
        // The land: every node that is not the ocean (review finding 10), a pocket's floor too.
        for i in 0..n_of(&lattice) {
            if state.ocean[i] || climate.ela_z[i] == i32::MAX {
                continue;
            }
            land += 1;
            let h = (f64::from(state.z[i]) - f64::from(climate.ela_z[i])) / steps;
            if h > worst {
                worst = h;
            }
            if h > 1_500.0 {
                over += 1;
            }
        }
        println!(
            "\nG-BUZZSAW: the highest peak stands {:.0} m over its own snowline; {over} of {land} land nodes stand over 1 500 m of it ({:.3} %) — Egholm 2009: about 1 500 m, everywhere on Earth",
            worst,
            over as f64 * 100.0 / land.max(1) as f64
        );
        let mut ela: Vec<i32> = (0..n_of(&lattice))
            .filter(|&i| !state.ocean[i] && climate.ela_z[i] != i32::MAX)
            .map(|i| climate.ela_z[i])
            .collect();
        ela.sort_unstable();
        if !ela.is_empty() {
            println!(
                "  the snowline over the land: lowest {:.0} m, median {:.0} m, highest {:.0} m (the sea stands at {:.0} m)",
                f64::from(ela[0]) / steps,
                f64::from(ela[ela.len() / 2]) / steps,
                f64::from(ela[ela.len() - 1]) / steps,
                f64::from(state.sea_z) / steps
            );
        }
    }
    // ★ GATE G-CRATER (ruling B2 step 2): Earth holds about 190 confirmed impact structures, 43 of
    // them wider than 20 km, the median 8 km, and 45 % of the record younger than 200 Ma because
    // the surface renews itself (Earth Impact Database; Osinski et al. 2022).
    {
        let record = &report.crater_record;
        let mut widths: Vec<u32> = record.iter().map(|c| c.diameter_m).collect();
        let mut ages: Vec<u64> = record.iter().map(|c| c.age_yr).collect();
        widths.sort_unstable();
        ages.sort_unstable();
        let wide = widths.iter().filter(|&&d| d >= 20_000).count();
        let young = ages.iter().filter(|&&a| a < 200_000_000).count();
        println!(
            "G-CRATER: {} craters kept of {} the production function drew; {wide} wider than 20 km; median {:.1} km — Earth: 190 structures, 43 over 20 km, median 8 km",
            record.len(),
            vd_terrain::craters::expected_count(&body, &lattice, &words),
            widths
                .get(widths.len() / 2)
                .map_or(0.0, |&d| f64::from(d) / 1_000.0)
        );
        println!(
            "  the age histogram: {young} younger than 200 Ma ({:.0} %), median {:.0} Ma, oldest {:.0} Ma — Earth: 45 % younger than 200 Ma, which is 4.4 % of its history",
            young as f64 * 100.0 / record.len().max(1) as f64,
            ages.get(ages.len() / 2).map_or(0.0, |&a| a as f64 / 1.0e6),
            ages.last().map_or(0.0, |&a| a as f64 / 1.0e6)
        );
    }
    // ★ THE BUDGET'S OWN COST, measured on its own (ruling W11): the depression hierarchy and the
    // water balance over the whole planet, run once more against the clock, so the price is a
    // number and not an argument. ★ The census KEEPS this hierarchy (ruling B6, step 1): G-RIVER
    // and the watershed's what-if below read its depressions and its water bodies.
    let lakes = {
        let started = std::time::Instant::now();
        let lakes = vd_terrain::lakes::budget(&state, &climate);
        println!(
            "the hierarchy and the budget cost {:.1} s of the solve, and found {} water bodies",
            started.elapsed().as_secs_f64(),
            lakes.report.bodies
        );
        lakes
    };
    let artifact =
        vd_terrain::artifact::Artifact::of(&state, &facies_v, &climate, words.water_km3 > 0);
    let n = artifact.node_count();
    let facies: Vec<u8> = artifact
        .rows
        .iter()
        .map(|r| r.receiver_facies >> FACIES_SHIFT)
        .collect();
    let sea = facies.iter().filter(|&&f| f & FACIES_SEA != 0).count();
    let lake = facies
        .iter()
        .filter(|&&f| f & FACIES_LAKE != 0 && f & FACIES_SEA == 0)
        .count();
    let dry = n - sea - lake;
    // The drainage area of every node over the shipped receivers: the river threshold's word.
    let drain = drainage_area(&state);
    // ★ THE RIVER SET TWO WAYS (plan step 2 part 4): by the drainage AREA (the gates' rule since
    // step 2) and by the STEP-1 RULE — the RAIN accumulated over the shipped receivers at least
    // `VALLEY_NODES` times the node's own rain — so a change of the field and a change of the rule
    // can be told apart. MEASURED: the two rules do not read one river set.
    let by_area: Vec<bool> = (0..n)
        .map(|i| !state.ocean[i] && drain[i] >= state.area[i].saturating_mul(VALLEY_NODES))
        .collect();
    let by_rain: Vec<bool> = {
        let mut rain: Vec<u64> = (0..n)
            .map(|i| u64::from(state.rain[i]) * state.area[i])
            .collect();
        for &node in &state.order {
            let r = state.receiver[node as usize];
            if r != NO_NODE {
                rain[r as usize] += rain[node as usize];
            }
        }
        (0..n)
            .map(|i| {
                let own = u64::from(state.rain[i]) * state.area[i];
                !state.ocean[i] && rain[i] >= own.saturating_mul(VALLEY_NODES)
            })
            .collect()
    };
    // ★ GATE G-ICE (ruling B2 step 2): Earth carries ice on about 10 % of its land today and about
    // 30 % at a glacial maximum (NSIDC; the literature spreads 25–32 %).
    {
        let ice = facies
            .iter()
            .filter(|&&f| f & vd_terrain::solve::FACIES_ICE != 0 && f & FACIES_SEA == 0)
            .count();
        println!(
            "\nG-ICE: {ice} nodes under ice, {:.2} % of the land ({} land nodes) — Earth: about 10 % today, about 30 % at a glacial maximum. The solve's own reading: {} nodes under the line, thickest {:.0} m, deepest cut {} steps",
            ice as f64 * 100.0 / (lake + dry).max(1) as f64,
            lake + dry,
            report.ice.0,
            report.ice.1,
            report.ice.2
        );
    }
    println!(
        "\nnodes {n} of {node_m:.0} m ({node_km2:.1} km²): sea {sea} ({:.2} %), lake {lake} ({:.2} %), dry land {dry} ({:.2} %)",
        sea as f64 * 100.0 / n as f64,
        lake as f64 * 100.0 / n as f64,
        dry as f64 * 100.0 / n as f64
    );
    println!(
        "the lakes' share of the LAND by NODE COUNT (every node that is not the ocean; the share by AREA is G-LAKE-SHARE's, below): {:.2} % — Earth {EARTH_LAKE_SHARE_PCT} % (glacial shields included), under 1 % elsewhere",
        lake as f64 * 100.0 / (lake + dry) as f64
    );

    // Every lake as a connected patch of lake nodes (the lattice's own ring, seams included).
    let mut seen = vec![false; n];
    let mut sizes: Vec<usize> = Vec::new();
    let mut patches: Vec<Vec<u32>> = Vec::new();
    for start in 0..n {
        if seen[start] || facies[start] & FACIES_LAKE == 0 || facies[start] & FACIES_SEA != 0 {
            continue;
        }
        let mut patch = Vec::new();
        let mut queue = VecDeque::from([start as u32]);
        seen[start] = true;
        while let Some(node) = queue.pop_front() {
            patch.push(node);
            for m in lattice.neighbours(node) {
                let mi = m as usize;
                if m == u32::MAX || mi >= n || seen[mi] {
                    continue;
                }
                if facies[mi] & FACIES_LAKE != 0 && facies[mi] & FACIES_SEA == 0 {
                    seen[mi] = true;
                    queue.push_back(m);
                }
            }
        }
        sizes.push(patch.len());
        patches.push(patch);
    }
    sizes.sort_unstable();
    if !sizes.is_empty() {
        let total: usize = sizes.iter().sum();
        println!(
            "lakes: {} patches; median {} nodes ({:.0} km²), p90 {} nodes, largest {} nodes ({:.0} km²); mean {:.1} nodes",
            sizes.len(),
            sizes[sizes.len() / 2],
            sizes[sizes.len() / 2] as f64 * node_km2,
            sizes[sizes.len() * 9 / 10],
            sizes[sizes.len() - 1],
            sizes[sizes.len() - 1] as f64 * node_km2,
            total as f64 / sizes.len() as f64
        );
        let mut hist = [0usize; 8];
        for &s in &sizes {
            let b = (usize::BITS - s.leading_zeros()) as usize;
            hist[b.min(7)] += 1;
        }
        println!(
            "  size classes (nodes): 1:{} 2-3:{} 4-7:{} 8-15:{} 16-31:{} 32-63:{} 64+:{}",
            hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7]
        );
    }

    // ★ GATE G-GRID (ruling B2 step 3; the report's §3.4 and §6.3): the long-axis histogram of the
    // lakes and the valleys against the FACE GRID'S OWN four directions. A router with no direction
    // of its own leaves the four bins equal, so the flatness — the biggest bin over the mean bin —
    // reads one. A spike at 45° names the tie-break: `steeper()` and the flats both break a tie on
    // the smaller node index, which is the stencil's own `(−1, −1)` corner.
    //
    // A VALLEY is a chain of receivers a real river could cut: a land node whose discharge is at
    // least a hundred times the rain on its own node, which is the drainage-area threshold a
    // channel needs, and its long axis is taken over eight receiver steps — one step is one of
    // eight directions by construction and says nothing, eight steps is a line on the ground.
    {
        use vd_bins::grid_bias::{AxisHistogram, patch_axis, trunk_axis};
        const TRUNK_STEPS: usize = 8;
        const PATCH_MIN: usize = 4;
        let mut lake_four = AxisHistogram::new(4);
        let mut lake_twelve = AxisHistogram::new(12);
        for patch in patches.iter().filter(|p| p.len() >= PATCH_MIN) {
            if let Some(axis) = patch_axis(&lattice, patch) {
                lake_four.add(axis.degrees);
                lake_twelve.add(axis.degrees);
            }
        }
        let mut valley_four = AxisHistogram::new(4);
        let mut valley_twelve = AxisHistogram::new(12);
        let mut step1_four = AxisHistogram::new(4);
        // The area rule's trunks OUTSIDE every hollow the routing filled (the node's own pit is
        // zero): a flat's receivers run by the distance field, so this separates the flats' lean
        // from the sloped ground's.
        let mut sloped_four = AxisHistogram::new(4);
        // ★ BY THE WATER'S CLASS (plan step 2b): the area rule's trunks whose start node is DRY
        // ground off every flat, on a FLAT of the ground (a neighbour of its own height, not the
        // root), or in a SPILLING lake's hollow (the flat step across the lake).
        let class = classes(&state, &lakes);
        let on_ground_flat = |i: usize| -> bool {
            !is_root(&lakes, i)
                && lattice.neighbours(i as u32).iter().any(|&m| {
                    m != NO_NODE
                        && state.z[m as usize] == state.z[i]
                        && !is_root(&lakes, m as usize)
                })
        };
        let mut dry_four = AxisHistogram::new(4);
        let mut flat_four = AxisHistogram::new(4);
        let mut lake_steps_four = AxisHistogram::new(4);
        for node in 0..n as u32 {
            let i = node as usize;
            if !by_area[i] && !by_rain[i] {
                continue;
            }
            if let Some(axis) = trunk_axis(&lattice, &state.receiver, node, TRUNK_STEPS) {
                if by_area[i] {
                    valley_four.add(axis.degrees);
                    valley_twelve.add(axis.degrees);
                    if state.pit[i] == 0 {
                        sloped_four.add(axis.degrees);
                    }
                    match class[i] {
                        Class::Spilling => lake_steps_four.add(axis.degrees),
                        Class::Dry if on_ground_flat(i) => flat_four.add(axis.degrees),
                        _ => dry_four.add(axis.degrees),
                    }
                }
                if by_rain[i] {
                    step1_four.add(axis.degrees);
                }
            }
        }
        println!(
            "  G-GRID's valleys by the STEP-1 RULE (the rain over the shipped receivers at least {VALLEY_NODES} times the node's own), for a like-for-like reading against generator 19: {} trunks: {} | FLATNESS {:.3}",
            step1_four.total(),
            step1_four.line(),
            step1_four.flatness()
        );
        println!(
            "  G-GRID's valleys by the area rule OUTSIDE every hollow the routing filled (the node's own pit zero): {} trunks: {} | FLATNESS {:.3}",
            sloped_four.total(),
            sloped_four.line(),
            sloped_four.flatness()
        );
        println!(
            "  G-GRID's valleys by the area rule BY THE WATER'S CLASS (plan step 2b): dry ground off the flats {} trunks: {} | FLATNESS {:.3}; on a flat of the ground {} trunks: {} | FLATNESS {:.3}; across a spilling lake {} trunks: {} | FLATNESS {:.3}",
            dry_four.total(),
            dry_four.line(),
            dry_four.flatness(),
            flat_four.total(),
            flat_four.line(),
            flat_four.flatness(),
            lake_steps_four.total(),
            lake_steps_four.line(),
            lake_steps_four.flatness()
        );
        // ★ A REAL VERDICT (review findings 1 and 5 of step 2b): plan step 2b's law is that the
        //   directions over a filled flat and over a spilling lake do not lean on the lattice, and
        //   G-GRID reads them. The verdict reads those two classes against the aggregate's own
        //   tolerance; the dry class is printed beside them, and the aggregate line below reads it.
        //   The aggregate can read green while each class leans, because the dry ground leans to
        //   the diagonals and the flats to the rows, and the two cancel in one histogram.
        census(
            "G-GRID-VALLEYS-BY-CLASS",
            &format!(
                "{:.3}dry/{:.3}flats/{:.3}spilling-lakes",
                dry_four.flatness(),
                flat_four.flatness(),
                lake_steps_four.flatness()
            ),
            "~1.006(<=1.05-on-the-flats-and-the-spilling-lakes)[W13.3;plan-step-2b:a-flat's-directions-do-not-lean-on-the-lattice]",
            Verdict::of(flat_four.flatness() <= 1.05 && lake_steps_four.flatness() <= 1.05),
        );
        census(
            "G-GRID-VALLEYS",
            &format!(
                "{:.3}area-rule/{:.3}step1-rule/{:.3}outside-filled-hollows",
                valley_four.flatness(),
                step1_four.flatness(),
                sloped_four.flatness()
            ),
            "~1.006[W13.3:a-router-with-no-direction-of-its-own]",
            Verdict::of(valley_four.flatness() <= 1.05),
        );
        println!(
            "\nG-GRID: the long axes against the face grid (0° and 90° are the stencil's rows, 45° and 135° its diagonals)\n  lakes   ({} patches of {PATCH_MIN}+ nodes): {} | FLATNESS {:.3}\n    {} | flatness {:.3}\n  valleys ({} trunks over {VALLEY_NODES} nodes of drainage): {} | FLATNESS {:.3}\n    {} | flatness {:.3}\n  a router with no direction of its own reads 1.000; a spike at 45° or 135° names the tie-break",
            lake_four.total(),
            lake_four.line(),
            lake_four.flatness(),
            lake_twelve.line(),
            lake_twelve.flatness(),
            valley_four.total(),
            valley_four.line(),
            valley_four.flatness(),
            valley_twelve.line(),
            valley_twelve.flatness()
        );
    }

    // ★ STEP 1 OF THE PLAN (ruling B6, 2026-09-24): THE WATER POSITIONS AS NUMBERS. Each block
    // prints its reading, its band with the band's source, and one CENSUS line. Nothing here moves
    // the world.
    hypsometry_lines(&state, &report.integrals);
    let ocean = inland_sea_lines(&state, &lakes, &facies, first_land);
    river_lines(&state, &lakes, &facies, &ocean, &by_area, "");
    river_lines(&state, &lakes, &facies, &ocean, &by_rain, ":step1-rule");
    discharge_tree_lines(&state, &lakes, &climate, &report.tree);
    level_one_word_lines(&state.lattice, &artifact);
    budget_lines(&state, &lakes, &facies);
    step2b_measure_lines(&state, &lakes);
    lake_walk_lines(&state, &lakes, radius_m);
    drop(climate);

    // The node under the aim: the face's node whose direction is nearest.
    let mut best = 0u32;
    let mut best_dot = -2.0;
    for node in 0..n as u32 {
        let d = lattice.direction(node);
        let v = DVec3::new(d[0].raw() as f64, d[1].raw() as f64, d[2].raw() as f64).normalize();
        let dot = v.dot(aim);
        if dot > best_dot {
            best_dot = dot;
            best = node;
        }
    }
    let (face, i0, j0) = lattice.split(best);
    let edge = lattice.edge as i32;
    let per_face = (lattice.edge * lattice.edge) as usize;
    println!(
        "\nthe map around the aim: face {:?} node ({i0}, {j0}); {} nodes a side = {:.0} km; '~' sea, 'L' lake, '#' land, ':' coast band",
        face,
        2 * half + 1,
        (2 * half + 1) as f64 * node_m / 1000.0
    );
    let half = half as i32;
    let mut j = j0 - half;
    while j <= j0 + half {
        let mut line = String::new();
        let mut i = i0 - half;
        while i <= i0 + half {
            let c = if i < 0 || j < 0 || i >= edge || j >= edge {
                ' '
            } else {
                let node =
                    face.index() as usize * per_face + j as usize * edge as usize + i as usize;
                let f = facies[node];
                if f & FACIES_SEA != 0 {
                    '~'
                } else if f & FACIES_LAKE != 0 {
                    'L'
                } else if f & vd_terrain::solve::FACIES_COAST != 0 {
                    ':'
                } else {
                    '#'
                }
            };
            line.push(c);
            i += 1;
        }
        println!("{line}");
        j += 1;
    }
}

/// A gate's verdict word.
#[derive(Clone, Copy)]
enum Verdict {
    Green,
    Red,
    NoBand,
}

impl Verdict {
    /// GREEN where the reading holds its band, RED where it does not.
    fn of(holds: bool) -> Verdict {
        if holds { Verdict::Green } else { Verdict::Red }
    }

    /// The word a script greps.
    fn word(self) -> &'static str {
        match self {
            Verdict::Green => "GREEN",
            Verdict::Red => "RED",
            Verdict::NoBand => "NO-BAND",
        }
    }
}

/// ★ ONE GREP-ABLE LINE A GATE: `CENSUS <gate> <value> <band> <verdict>`. A field never holds a
/// space (the line writes `_` for one), so a script splits the line on white space into five
/// words. The value carries its unit, and the band carries its source in brackets.
fn census(gate: &str, value: &str, band: &str, verdict: Verdict) {
    println!(
        "CENSUS {gate} {} {} {}",
        value.replace(' ', "_"),
        band.replace(' ', "_"),
        verdict.word()
    );
}

/// ★ GATE G-HYPSO (ruling B3) on the FINAL solved field: the heights of the whole surface, the sea
/// floor and the land, each node weighted by its own area. Four readings:
///
/// 1. The histogram in fixed bins of 1 000 m over the sea's level.
/// 2. THE TWO-HUMP TEST. The rule is gate G-LAND-HYPSOMETRY's own (`land::humps` over
///    `land::hypsometry`): bins of 250 m, each bin smoothed 1-2-1 with its two neighbours, a hump
///    is a local maximum at least a tenth of the largest bin, and the two largest humps count. The
///    gate holds where two humps stand and the lower one stands under the sea: Earth's ocean floors
///    and its continents (ruling B3).
/// 3. THE SOLVE'S OWN G-AGE INTEGRALS, which B3 names as the gate's calibration: the median basin's
///    hypsometric integral against Strahler's maturity band (Strahler 1952; `solve::HI_MATURE`).
/// 4. The land's share over 1 000 m. NO BAND YET: ruling B3 names Earth's share, but the repo holds
///    no Earth number for it with a published source.
///
/// **Example.** The census reads the belt's range and the abyss in one histogram. A second hump
/// under the sea says the ocean floors stand where the water's load puts them; one hump says the
/// planet is one smooth ball of noise, as it was before 8c.
fn hypsometry_lines(state: &MacroSolve, integrals: &[vd_terrain::solve::BasinIntegral]) {
    // The fixed bin and the span each side of the sea's level, metres; one more bin each side takes
    // everything past the span. The height over the sea that B3's share reads, metres.
    const BIN_M: i64 = 1_000;
    const SPAN_M: i64 = 12_000;
    const HIGH_M: i64 = 1_000;
    let steps = f64::from(Z_STEPS_PER_M);
    let per_m = i64::from(Z_STEPS_PER_M);
    let sea = i64::from(state.sea_z);
    let sea_m = f64::from(state.sea_z) / steps;
    let inner = (2 * SPAN_M / BIN_M) as usize;
    let mut bins = vec![0u128; inner + 2];
    let (mut all, mut land, mut high, mut under) = (0u128, 0u128, 0u128, 0u128);
    let (mut land_steps, mut under_steps) = (0i128, 0i128);
    for (k, &z) in state.z.iter().enumerate() {
        let a = u128::from(state.area[k]);
        let over = i64::from(z) - sea;
        let slot = (over + SPAN_M * per_m).div_euclid(BIN_M * per_m);
        let at = if slot < 0 {
            0
        } else if slot >= inner as i64 {
            inner + 1
        } else {
            slot as usize + 1
        };
        bins[at] += a;
        all += a;
        if over > 0 {
            land += a;
            land_steps += i128::from(over) * a as i128;
            if over > HIGH_M * per_m {
                high += a;
            }
        } else {
            under += a;
            under_steps += i128::from(over) * a as i128;
        }
    }
    let share = |a: u128| a as f64 * 100.0 / all.max(1) as f64;
    println!(
        "\nG-HYPSO (ruling B3), the FINAL solved field, every node by its own area; heights over the sea's level ({sea_m:.0} m over the ladder radius)"
    );
    println!("  the whole surface in {BIN_M} m bins over the sea, % of the surface:");
    println!("    under {} m: {:.2} %", -SPAN_M, share(bins[0]));
    let mut line = String::new();
    for (k, &a) in bins[1..=inner].iter().enumerate() {
        let from = k as i64 * BIN_M - SPAN_M;
        line.push_str(&format!(
            " {from:>+6}..{:>+6} m {:>5.2} % |",
            from + BIN_M,
            share(a)
        ));
        if k % 6 == 5 {
            println!("   {line}");
            line.clear();
        }
    }
    if !line.is_empty() {
        println!("   {line}");
    }
    println!("    over {SPAN_M} m: {:.2} %", share(bins[inner + 1]));
    // The two-hump test, by G-LAND-HYPSOMETRY's own rule over the final field.
    let (lo, hist) = vd_terrain::land::hypsometry(&state.z, &state.area);
    let hump_bin_m = f64::from(vd_terrain::land::HYPSOMETRY_BIN_M);
    let hump_m = |k: usize| f64::from(lo) / steps + (k as f64 + 0.5) * hump_bin_m - sea_m;
    let band = "2humps,the-lower-under-the-sea[ruling-B3-G-HYPSO;rule:land::humps]";
    match vd_terrain::land::humps(&hist) {
        Some((low, high_bin, valley)) => {
            let (low_m, high_m) = (hump_m(low), hump_m(high_bin));
            println!(
                "  the two-hump test (G-LAND-HYPSOMETRY's own rule: {hump_bin_m:.0} m bins smoothed 1-2-1, a hump is a local maximum at least a tenth of the largest bin, the two largest count): humps at {low_m:+.0} m and {high_m:+.0} m over the sea; the shallowest bin between them stands at {valley:.2} of the lower hump"
            );
            census(
                "G-HYPSO-HUMPS",
                &format!("2humps@{low_m:+.0}m,{high_m:+.0}m"),
                band,
                Verdict::of(low_m < 0.0),
            );
        }
        None => {
            println!(
                "  the two-hump test (G-LAND-HYPSOMETRY's own rule): ONE hump or none, so the field is not two-humped"
            );
            census("G-HYPSO-HUMPS", "under-2humps", band, Verdict::Red);
        }
    }
    // THE SOLVE'S OWN G-AGE INTEGRALS, which ruling B3 names as G-HYPSO's calibration: the median
    // basin's hypsometric integral against Strahler's maturity band (Strahler 1952; `HI_MATURE`).
    // ★ The solve's OWN reading (`FullReport::integrals`, plan step 2b question 7): the basins of
    // the erosion's own routing tree, read before the solve swapped in the water's tree.
    let (mature_lo, mature_hi) = vd_terrain::solve::HI_MATURE;
    let age_band = format!("{mature_lo:.2}-{mature_hi:.2}[Strahler-1952;solve.rs-HI_MATURE]");
    match vd_terrain::solve::age_gate(integrals) {
        Some((median_q8, mature)) => {
            let median = f64::from(median_q8) / 256.0;
            println!(
                "  the basins' hypsometric integrals (gate G-AGE's own reading over {} basins of {} land nodes or more): the median {median:.3}, against Strahler's maturity band {mature_lo:.2} to {mature_hi:.2}",
                integrals.len(),
                vd_terrain::solve::AGE_GATE_MIN_NODES
            );
            census(
                "G-HYPSO-AGE",
                &format!("{median:.3}integral"),
                &age_band,
                Verdict::of(mature),
            );
        }
        None => {
            println!("  the basins' hypsometric integrals: no basin to read");
            census("G-HYPSO-AGE", "no-basin", &age_band, Verdict::Red);
        }
    }
    let high_pct = high as f64 * 100.0 / land.max(1) as f64;
    println!(
        "  the land (over the sea) is {:.2} % of the surface and its mean stands {:.0} m over the sea; the ground under the sea is {:.2} % and its mean stands {:.0} m (01_reference_target.md §6.3 states Earth's +840 m and -3 800 m WITHOUT a source, so neither is a band)",
        share(land),
        land_steps as f64 / land.max(1) as f64 / steps,
        share(under),
        under_steps as f64 / under.max(1) as f64 / steps
    );
    println!(
        "  the land's share over {HIGH_M} m: {high_pct:.2} % of the land. NO BAND YET: ruling B3 names Earth's share over 1 000 m, and the repo holds no Earth number for it with a published source. OWED: a cited Earth hypsometric table in the G-HYPSO row of ruling B3"
    );
    census(
        "G-HYPSO-LAND-OVER-1000M",
        &format!("{high_pct:.2}%of-land"),
        "no-band-yet[owed:a-cited-Earth-table-in-ruling-B3]",
        Verdict::NoBand,
    );
}

/// One connected body of nodes at or under the sea's level.
struct Pocket {
    /// Its nodes.
    nodes: usize,
    /// Its area, m².
    area_m2: u128,
    /// Its deepest node.
    deepest: u32,
    /// Its volume under the sea's level, m³.
    volume_m3: u128,
}

/// ★ THE INLAND POCKETS (status §6 T7; ruling B7 item 1: *"an inland pocket is a closed basin, not
/// the sea"*): the census's OWN labelling of every node at or under the sea's level through the
/// lattice's eight neighbours, the cube seams included. The largest body by area is the OCEAN
/// (ties to the smaller lowest index, the solve's own rule); every other body is a POCKET.
///
/// Since plan step 2 the solve labels the same ocean (`MacroSolve::ocean`), so this is an
/// INDEPENDENT CHECK: CHECK-OCEAN counts the nodes where the two labellings disagree, and
/// CHECK-POCKET-NOT-SEA the pocket nodes that still carry the sea's facies bit — both 0 by
/// construction. It prints how each pocket ends by the budget (the water body over its deepest
/// node: DRY, PARTIAL or SPILLING), the pockets' volume under the sea's level, and the pockets the
/// FIRST land already held (plan question 9).
///
/// Returns, for each node, whether it belongs to the ocean.
///
/// **Example.** A crater floor 300 km inland stands 40 m under the sea's level. No path through the
/// lattice joins it to the ocean, so it is a pocket: the budget decides its water, and the census
/// says whether it ended a salt pan, a closed lake or a lake with an outlet.
fn inland_sea_lines(
    state: &MacroSolve,
    lakes: &Lakes,
    facies: &[u8],
    first_land: (usize, usize),
) -> Vec<bool> {
    const UNSEEN: u32 = u32::MAX;
    let n = state.node_count();
    let steps = f64::from(Z_STEPS_PER_M);
    let mut body = vec![UNSEEN; n];
    let mut pockets: Vec<Pocket> = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    for start in 0..n as u32 {
        let s = start as usize;
        if body[s] != UNSEEN || state.z[s] > state.sea_z {
            continue;
        }
        let id = pockets.len() as u32;
        let mut pocket = Pocket {
            nodes: 0,
            area_m2: 0,
            deepest: start,
            volume_m3: 0,
        };
        body[s] = id;
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            let vi = v as usize;
            pocket.nodes += 1;
            pocket.area_m2 += u128::from(state.area[vi]);
            pocket.volume_m3 += u128::from((state.sea_z - state.z[vi]) as u32)
                * u128::from(state.area[vi])
                / Z_STEPS_PER_M as u128;
            if state.z[vi] < state.z[pocket.deepest as usize] {
                pocket.deepest = v;
            }
            for m in state.lattice.neighbours(v) {
                if m == NO_NODE {
                    continue;
                }
                let mi = m as usize;
                if body[mi] == UNSEEN && state.z[mi] <= state.sea_z {
                    body[mi] = id;
                    queue.push_back(m);
                }
            }
        }
        pockets.push(pocket);
    }
    // The largest by area; a tie keeps the body met first (the solve's own rule).
    let mut ocean = UNSEEN;
    for (k, p) in pockets.iter().enumerate() {
        if ocean == UNSEEN || p.area_m2 > pockets[ocean as usize].area_m2 {
            ocean = k as u32;
        }
    }
    let is_ocean: Vec<bool> = body.iter().map(|&b| b != UNSEEN && b == ocean).collect();
    let ocean_off = (0..n).filter(|&i| is_ocean[i] != state.ocean[i]).count();
    let sea_bit_off = (0..n)
        .filter(|&i| !is_ocean[i] && facies[i] & FACIES_SEA != 0)
        .count();
    let mut inland: Vec<&Pocket> = pockets
        .iter()
        .enumerate()
        .filter(|(k, _)| *k as u32 != ocean)
        .map(|(_, p)| p)
        .collect();
    inland.sort_by_key(|p| std::cmp::Reverse(p.area_m2));
    let nodes: usize = inland.iter().map(|p| p.nodes).sum();
    let area_km2 = inland.iter().map(|p| p.area_m2).sum::<u128>() as f64 / 1.0e6;
    let volume_km3 = inland.iter().map(|p| p.volume_m3).sum::<u128>() as f64 / 1.0e9;
    let ocean_km2 = pockets.get(ocean as usize).map_or(0, |p| p.area_m2) as f64 / 1.0e6;
    // How each pocket ends: the water body the budget stands over its deepest node.
    let verdict = |p: &Pocket| -> usize {
        let h = lakes.home[p.deepest as usize];
        if h == NO_DEP || h == OCEAN {
            return 3;
        }
        let b = lakes.body_of[h as usize];
        if b == NO_DEP || lakes.deps[b as usize].level == DRY {
            0
        } else if lakes.deps[b as usize].full {
            2
        } else {
            1
        }
    };
    let mut ends = [0usize; 4];
    for p in &inland {
        ends[verdict(p)] += 1;
    }
    println!(
        "\nTHE INLAND POCKETS (status §6 T7; ruling B7 item 1): nodes at or under the sea's level ({:.0} m) that no path through the lattice's own neighbours (the cube seams included) joins to the OCEAN, the largest such body ({ocean_km2:.4e} km²) — each a CLOSED BASIN since plan step 2",
        f64::from(state.sea_z) / steps
    );
    println!(
        "  {} pockets, {nodes} nodes, {area_km2:.0} km², {volume_km3:.4e} km³ under the sea's level; by the budget over each deepest node: DRY {} PARTIAL {} SPILLING {} (the ocean's own root {}); the FIRST land held {} pockets of {} nodes (plan question 9). The largest ten:",
        inland.len(),
        ends[0],
        ends[1],
        ends[2],
        ends[3],
        first_land.0,
        first_land.1
    );
    const ENDS: [&str; 4] = ["DRY", "PARTIAL", "SPILLING", "ROOT"];
    for p in inland.iter().take(10) {
        let aim = unit(&state.lattice, p.deepest);
        println!(
            "    {:>7} nodes {:>10.0} km², deepest {:>6.0} m under the sea at node {} — {} — aim {:.6} {:.6} {:.6}",
            p.nodes,
            p.area_m2 as f64 / 1.0e6,
            f64::from(state.sea_z - state.z[p.deepest as usize]) / steps,
            p.deepest,
            ENDS[verdict(p)],
            aim.x,
            aim.y,
            aim.z
        );
    }
    census(
        "INLAND-POCKETS",
        &format!("{area_km2:.0}km2/{nodes}nodes/{}pockets", inland.len()),
        "no-band[ruling-B7:each-a-closed-basin-by-its-budget]",
        Verdict::NoBand,
    );
    census(
        "CHECK-OCEAN",
        &format!("{ocean_off}nodes"),
        "0[the-census's-own-labelling-is-the-solve's]",
        Verdict::of(ocean_off == 0),
    );
    census(
        "CHECK-POCKET-NOT-SEA",
        &format!("{sea_bit_off}nodes"),
        "0[ruling-B7:a-pocket-is-never-the-sea]",
        Verdict::of(sea_bit_off == 0),
    );
    is_ocean
}

/// Where a river ends, or that it runs on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum End {
    /// Not an end: the water runs on to the receiver.
    Runs,
    /// The ocean.
    Ocean,
    /// A node the facies call SEA that is not the ocean: 0 by construction since plan step 2.
    InlandSea,
    /// Under a closed lake's water: the budget holds the river there.
    ClosedLake,
    /// A dry hollow's pit: the budget holds the river there as a salt pan.
    DryBasin,
    /// An end the budget does not name — no receiver inside a spilling lake's hollow, or on dry
    /// ground that has a way down: a defect.
    Defect,
    /// A node on a ring of receivers: a defect.
    Cycle,
}

/// The water class of each node in the water's own tree (plan step 2b): what the shipped receiver
/// of the node is by construction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    /// Dry ground: the budget's own descent.
    Dry,
    /// A node of a spilling lake's hollow: the flat step toward the lake's spill.
    Spilling,
    /// The spill node of a spilling lake: the exit over its saddle.
    Spill,
    /// Under a closed lake's water: an end.
    Closed,
    /// The ocean (or a root): an end.
    Ocean,
}

/// The water body over node `i`, or `NO_DEP`.
fn body_at(lakes: &Lakes, i: usize) -> u32 {
    let h = lakes.home[i];
    if h == NO_DEP || h == OCEAN {
        NO_DEP
    } else {
        lakes.body_of[h as usize]
    }
}

/// ★ THE CLASS OF EVERY NODE in the water's own tree, read from the budget alone (never from the
/// shipped receivers, which the checks compare against it).
fn classes(state: &MacroSolve, lakes: &Lakes) -> Vec<Class> {
    let n = state.node_count();
    let mut out: Vec<Class> = (0..n)
        .map(|i| {
            let b = body_at(lakes, i);
            if state.ocean[i] || is_root(lakes, i) {
                Class::Ocean
            } else if b != NO_DEP && lakes.deps[b as usize].full {
                Class::Spilling
            } else if lakes.water[i] != DRY {
                Class::Closed
            } else {
                Class::Dry
            }
        })
        .collect();
    for (d, dep) in lakes.deps.iter().enumerate().skip(1) {
        if lakes.body_of[d] == d as u32 && dep.full {
            out[dep.spill_node as usize] = Class::Spill;
        }
    }
    out
}

/// ★ THE END AT ONE NODE (plan step 2b; ruling B8 item 2), or [`End::Runs`]. A node with a
/// shipped receiver runs on. A node with none ends the river: at the ocean, under a CLOSED lake's
/// water (the river ends where it meets the lake), or at a DRY hollow's pit (a salt pan). Any
/// other end is a defect: no receiver inside a spilling lake's hollow, where the water must cross to
/// the spill, or on dry ground that has a way down.
fn end_at(state: &MacroSolve, lakes: &Lakes, facies: &[u8], ocean: &[bool], i: usize) -> End {
    if state.receiver[i] != NO_NODE {
        return End::Runs;
    }
    if ocean[i] {
        return End::Ocean;
    }
    if facies[i] & FACIES_SEA != 0 {
        return End::InlandSea;
    }
    let b = body_at(lakes, i);
    if b == NO_DEP || lakes.deps[b as usize].full {
        return End::Defect;
    }
    if lakes.water[i] != DRY {
        End::ClosedLake
    } else if lakes.descent.down[i] == NO_NODE {
        End::DryBasin
    } else {
        End::Defect
    }
}

/// ★ THE DRAINAGE AREA of every node, m²: its own area and every node's upstream of it, down the
/// SHIPPED receivers. The river threshold reads it, because the discharge row is the RUNOFF since
/// plan step 2. ★ The plan said this area rule reads "the same river set as at gen 19"; MEASURED it
/// does not (the step-1 rule weighted each node by its own rain), so the census prints both.
fn drainage_area(state: &MacroSolve) -> Vec<u64> {
    let mut drain = state.area.clone();
    for &node in &state.order {
        let r = state.receiver[node as usize];
        if r != NO_NODE {
            drain[r as usize] += drain[node as usize];
        }
    }
    drain
}

/// ★ GATE G-RIVER (ruling B3: every river reaches the sea or a dry basin; a valley's floor descends
/// monotonically along its trunk). A RIVER is a land node whose DRAINAGE AREA over the shipped
/// receivers is at least [`VALLEY_NODES`] times its own area — G-GRID's valley threshold, so both
/// gates read one river set (the rain-weighted threshold of step 1 read the discharge row, which
/// carries the runoff since plan step 2). A CHANNEL HEAD is a river node no river node feeds. Each
/// head's water is walked down the SHIPPED receivers to its end ([`end_at`]); the ends are counted
/// per head. Each trunk step is read once: the FLOOR is the ground, or the water's surface where
/// water stands (a lake's flat surface is a floor that does not rise), and a step on which the
/// floor rises is a defect. The band is zero on both lines.
///
/// **Example.** A river off the belt's range enters a hollow the routing filled. The budget holds
/// no water there, so the river ends in a dry basin, and that end is lawful. If the shipped receiver
/// instead carries it up out of a filled hollow onto a ridge 40 m higher, the floor rises and
/// G-RIVER-DESCENT counts the step.
fn river_lines(
    state: &MacroSolve,
    lakes: &Lakes,
    facies: &[u8],
    ocean: &[bool],
    river: &[bool],
    label: &str,
) {
    let n = state.node_count();
    let steps = f64::from(Z_STEPS_PER_M);
    let term: Vec<End> = (0..n)
        .map(|i| end_at(state, lakes, facies, ocean, i))
        .collect();
    // Each node's end, outlets first: a node's end is its own, or its receiver's. A node the order
    // never reached stands on a ring of receivers.
    let mut end = vec![End::Cycle; n];
    for &node in state.order.iter().rev() {
        let i = node as usize;
        let r = state.receiver[i];
        end[i] = if term[i] != End::Runs {
            term[i]
        } else if r == NO_NODE {
            End::Defect
        } else {
            end[r as usize]
        };
    }
    // The heads: a head is a river node that no river node feeds.
    let mut fed = vec![false; n];
    for node in 0..n as u32 {
        let i = node as usize;
        let r = state.receiver[i];
        if river[i] && r != NO_NODE {
            fed[r as usize] = true;
        }
    }
    let mut ends = [0usize; 7];
    let mut heads = 0usize;
    let mut on_trunk = vec![false; n];
    for node in 0..n as u32 {
        let i = node as usize;
        if !river[i] || fed[i] {
            continue;
        }
        heads += 1;
        ends[end[i] as usize] += 1;
        // Mark the trunk down to its end; a walk stops where an earlier walk already passed.
        let mut cur = i;
        while !on_trunk[cur] {
            on_trunk[cur] = true;
            let r = state.receiver[cur];
            if term[cur] != End::Runs || r == NO_NODE {
                break;
            }
            cur = r as usize;
        }
    }
    // Every trunk step once: the ground's rise, and the floor's.
    let floor = |i: usize| -> i32 {
        let w = state.water_level(i as u32);
        if w == DRY {
            state.z[i]
        } else {
            state.z[i].max(w)
        }
    };
    let (mut walked, mut ground_rises, mut hidden, mut floor_rises, mut in_fill) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut worst, mut worst_at) = (0i32, NO_NODE);
    for node in 0..n as u32 {
        let i = node as usize;
        let r = state.receiver[i];
        if !on_trunk[i] || term[i] != End::Runs || r == NO_NODE {
            continue;
        }
        let ri = r as usize;
        walked += 1;
        let rise = floor(ri) - floor(i);
        if state.z[ri] > state.z[i] && rise <= 0 {
            hidden += 1;
        }
        if state.z[ri] > state.z[i] {
            ground_rises += 1;
        }
        if rise > 0 {
            floor_rises += 1;
            if state.pit[i] > 0 || state.pit[ri] > 0 {
                in_fill += 1;
            }
            if rise > worst {
                worst = rise;
                worst_at = node;
            }
        }
    }
    let [_, to_ocean, to_pocket, to_lake, to_basin, defect, cycle] = ends;
    let node_km2 = state.lattice.node_m().powi(2) / 1.0e6;
    let rule = if label.is_empty() {
        "whose DRAINAGE AREA over the SHIPPED receivers is at least"
    } else {
        "(THE STEP-1 RULE, for comparison) whose RAIN over the SHIPPED receivers is at least"
    };
    println!(
        "\nG-RIVER{label} (ruling B3): a river is a land node {rule} {VALLEY_NODES} times its own (G-GRID's valley threshold, about {:.0} km²); {heads} channel heads, each walked down the SHIPPED receivers to its end",
        VALLEY_NODES as f64 * node_km2
    );
    println!(
        "  the ends, in heads: the ocean {to_ocean}, a node the facies call sea that is not the ocean {to_pocket} (0 by construction since plan step 2), a closed lake {to_lake}, a dry closed basin {to_basin}; DEFECTS: an end the budget does not name {defect}, a ring of receivers {cycle}"
    );
    println!(
        "  (since plan step 2b the shipped receivers are the water's own: a river ENDS where it meets a closed lake's water or a dry hollow's pit — an inland pocket included — and crosses a spilling lake to its spill)"
    );
    println!(
        "  along the trunks: {walked} steps; the GROUND rises on {ground_rises} of them ({hidden} under a water's flat surface); the FLOOR (the ground, or the water's surface where water stands) rises on {floor_rises} steps, {in_fill} of them in a hollow the ROUTING filled (a reading of the erosion's scratch surface)"
    );
    if worst_at != NO_NODE {
        let aim = unit(&state.lattice, worst_at);
        println!(
            "  the worst floor rise: {:.1} m at node {worst_at} — aim {:.6} {:.6} {:.6}",
            f64::from(worst) / steps,
            aim.x,
            aim.y,
            aim.z
        );
    }
    census(
        &format!("G-RIVER-END{label}"),
        &format!("{}heads-without-an-end", defect + cycle),
        "0heads[ruling-B3-G-RIVER]",
        Verdict::of(defect + cycle == 0),
    );
    census(
        &format!("G-RIVER-DESCENT{label}"),
        &format!("{floor_rises}rising-steps"),
        "0steps[ruling-B3-G-RIVER]",
        Verdict::of(floor_rises == 0),
    );
    census(
        &format!("G-RIVER-ENDS{label}"),
        &format!(
            "ocean{to_ocean}/closed-lake{to_lake}/dry-basin{to_basin}/not-ocean-sea{to_pocket}"
        ),
        "no-band[a-reading;the-plan's-gen-20-prediction:0-pocket-ends]",
        Verdict::NoBand,
    );
}

/// ★ THE DISCHARGE ROW AGAINST THE TREE IT SHIPS WITH (status §7 W10; ruling B7 item 2; since
/// plan step 2b ruling B8 item 2: the water's own tree and the budget's own water). Four readings:
///
/// 1. The shipped receiver against the class the budget names for each node ([`classes`]): every
///    DRY node must take the budget's own descent (CHECK-DRY-FOLLOWS-DESCENT), every spill of a
///    spilling lake its exit (CHECK-EXIT-AGREES), and no receiver is left out of the order
///    (CHECK-TREE-CYCLIC).
/// 2. The climate's RUNOFF accumulated again over the SHIPPED tree by the same rule — less, at
///    each spill node, the water its spilling lakes hold (`MacroSolve::spill_sinks`) — against the
///    shipped row: the nodes that differ, and those whose SHIPPED byte (the log class) differs
///    (W10-DISCHARGE-TREE).
/// 3. The nodes where the shipped row FALLS downstream with no sink to take it, which no tree can
///    hold.
/// 4. The tree's own counts (`FullReport::tree`).
///
/// **Example.** A river off the belt's range crosses a spilling lake: its row at the lake's notch
/// is the range's runoff less what the lake hands the air, and the census's own sum over the same
/// receivers with the same sink reads the same number.
fn discharge_tree_lines(
    state: &MacroSolve,
    lakes: &Lakes,
    climate: &Climate,
    tree: &vd_terrain::lakes::TreeReport,
) {
    let n = state.node_count();
    let class = classes(state, lakes);
    let (mut dry, mut dry_off, mut spills, mut exit_off) = (0usize, 0usize, 0usize, 0usize);
    let (mut spilling, mut closed, mut ocean) = (0usize, 0usize, 0usize);
    for (i, &c) in class.iter().enumerate() {
        match c {
            Class::Dry => {
                dry += 1;
                dry_off += usize::from(state.receiver[i] != lakes.descent.down[i]);
            }
            Class::Spill => spills += 1,
            Class::Spilling => spilling += 1,
            Class::Closed => {
                closed += 1;
                dry_off += usize::from(state.receiver[i] != NO_NODE);
            }
            Class::Ocean => ocean += 1,
        }
    }
    for (d, dep) in lakes.deps.iter().enumerate().skip(1) {
        if lakes.body_of[d] == d as u32 && dep.full {
            exit_off += usize::from(state.receiver[dep.spill_node as usize] != dep.exit);
        }
    }
    let cyclic = n - state.order.len();
    let mut sink = vec![0u64; n];
    for &(node, water) in &state.spill_sinks {
        sink[node as usize] += water;
    }
    let mut again: Vec<u64> = climate
        .runoff_mm_yr
        .iter()
        .zip(&state.area)
        .map(|(&q, &a)| u64::from(q) * a)
        .collect();
    let mut short = 0usize;
    for &node in &state.order {
        let i = node as usize;
        short += usize::from(again[i] < sink[i]);
        again[i] = again[i].saturating_sub(sink[i]);
        let r = state.receiver[i];
        if r != NO_NODE {
            again[r as usize] += again[i];
        }
    }
    let (mut differ, mut class_differ, mut falls) = (0usize, 0usize, 0usize);
    let mut worst = 1.0f64;
    for node in 0..n as u32 {
        let i = node as usize;
        let row = state.discharge[i];
        if again[i] != row {
            differ += 1;
            worst = worst.max(again[i].max(row) as f64 / again[i].min(row).max(1) as f64);
        }
        if log2_class(again[i], 2) != log2_class(row, 2) {
            class_differ += 1;
        }
        let r = state.receiver[i];
        if r != NO_NODE && sink[r as usize] == 0 && state.discharge[r as usize] < row {
            falls += 1;
        }
    }
    println!(
        "\nTHE DISCHARGE ROW AND ITS TREE (status §7 W10, W11; ruling B8 item 2: the budget's own water over the water's own receivers)"
    );
    println!(
        "  the classes the budget names: dry ground {dry}, a spilling lake's hollow {spilling}, a spill {spills}, under a closed lake's water {closed}, the ocean or a root {ocean}; dry nodes and closed-lake nodes whose shipped receiver is not the budget's (the descent, or none) {dry_off}; spills that do not point at their exit {exit_off}; nodes left out of the order {cyclic}"
    );
    println!(
        "  the tree's own counts: {} nodes of spilling lakes' hollows, {} spills, {} closed-lake ends; the flat steps across the lakes: split {}, straight {}, nearest {}; exits that disagree at one saddle {}, rings {}, short sinks {}",
        tree.region_nodes,
        tree.spills,
        tree.lake_ends,
        tree.split_steps,
        tree.straight_steps,
        tree.fallback_steps,
        tree.exit_disagree,
        tree.cyclic,
        tree.shortfall
    );
    println!(
        "  the climate's runoff accumulated again over the shipped tree, less the {} sinks at their spills: {differ} nodes differ (the worst by {worst:.1} times), {class_differ} of them in the SHIPPED byte (the row's log class); {short} sinks short; the shipped row FALLS downstream with no sink at {falls} nodes, which no tree can hold",
        state.spill_sinks.len()
    );
    census(
        "W10-DISCHARGE-TREE",
        &format!("{class_differ}nodes"),
        "0nodes[ruling-B8-item-2:the-budget's-water-over-the-water's-receivers]",
        Verdict::of(class_differ == 0),
    );
    census(
        "CHECK-DRY-FOLLOWS-DESCENT",
        &format!("{dry_off}nodes"),
        "0[plan-step-2b:the-dry-ground-takes-the-budget's-descent]",
        Verdict::of(dry_off == 0),
    );
    census(
        "CHECK-EXIT-AGREES",
        &format!("{exit_off}spills/{}disagree", tree.exit_disagree),
        "0[plan-step-2b:a-spill-points-where-the-cascade-pours]",
        Verdict::of(exit_off == 0 && tree.exit_disagree == 0),
    );
    census(
        "CHECK-TREE-CYCLIC",
        &format!("{cyclic}nodes/{falls}falls/{short}short"),
        "0[plan-step-2b:one-tree,no-ring]",
        Verdict::of(cyclic == 0 && falls == 0 && short == 0),
    );
}

/// ★ THE STORED LEVEL-1 WATER WORD, CHECKED (plan step 2 part 5; kept on 2026-09-28 from the
/// rung-swap block that ruling B22 removed, because it reads the SOLVE's stored data and no drawn
/// water): the shipped word `pyramid_water[0]` against `artifact::block_water` folded again over
/// the rows and the coast mask. Zero is expected, and the one-volume fill step reads this word.
///
/// **Example.** A tarn fills one node of the belt's pass at 2 100 m and its three neighbours in the
/// level-1 block are rock: the fold names 2 100 m, and so must the stored word.
fn level_one_word_lines(lattice: &MacroLattice, artifact: &Artifact) {
    let (Some(coarse), Some(fold)) = (lattice.coarser(1), artifact.pyramid_water.first()) else {
        println!(
            "\nTHE STORED LEVEL-1 WATER WORD: the artifact holds no level 1, nothing to check"
        );
        return;
    };
    let m = coarse.node_count();
    let (mut lake_max, mut has_sea) = (vec![DRY_M; m], vec![false; m]);
    for node in 0..lattice.node_count() as u32 {
        let (face, fi, fj) = lattice.split(node);
        let up = coarse.index(face, fi >> 1, fj >> 1) as usize;
        match coast_side(&artifact.coast, node) {
            Some(Side::Sea) => has_sea[up] = true,
            Some(Side::Lake) => {
                lake_max[up] = lake_max[up].max(artifact.rows[node as usize].water_m);
            }
            _ => {}
        }
    }
    let disagree = (0..m)
        .filter(|&u| block_water(lake_max[u], has_sea[u], artifact.sea_m) != fold[u])
        .count();
    println!(
        "\nTHE STORED LEVEL-1 WATER WORD: the shipped word disagrees with `block_water` over the rows at {disagree} of {m} level-1 nodes (0 expected)"
    );
    census(
        "CHECK-LEVEL1-WORD",
        &format!("{disagree}nodes"),
        "0[plan-step-2-part-5:the-stored-word-is-the-rows-own-fold]",
        Verdict::of(disagree == 0),
    );
}

/// ★ THE LAKE BUDGET'S OWN LINES (plan step 2 parts 1 and 2; ruling B7; status §7 W3 and W4): the
/// census's watershed WHAT-IF of step 1 is now the budget itself. Each leaf's WATERSHED supply
/// beside the runoff of its own nodes (the budget's rule before step 2), the three ends of the
/// water bodies and of the leaves, the cascade's transfers and its books, and the lakes' share of
/// the land.
///
/// **Example.** A hollow at the foot of the belt's wet range took only the runoff under its own
/// brim at generator 19 and stood a dry pan; its watershed takes in the whole range face, and the
/// line of the ten hollows fed most beyond their brim names it with its aim.
fn budget_lines(state: &MacroSolve, lakes: &Lakes, facies: &[u8]) {
    const STATES: [&str; 3] = ["DRY", "PARTIAL", "SPILLING"];
    let n = state.node_count();
    let deps = &lakes.deps;
    let m = deps.len();
    let steps = f64::from(Z_STEPS_PER_M);
    let r = &lakes.report;
    // The budget run here against the solve's own water: they must be one budget.
    let stale = (0..n)
        .filter(|&i| lakes.water[i] != state.water_z[i])
        .count();
    let state_of = |d: usize| -> usize {
        let b = lakes.body_of[d];
        if b == NO_DEP || deps[b as usize].level == DRY {
            0
        } else if deps[b as usize].full {
            2
        } else {
            1
        }
    };
    let mut leaf_ends = [0usize; 3];
    let (mut leaves, mut fed_above) = (0usize, 0usize);
    let mut ratio: Vec<f64> = Vec::new();
    let mut gains: Vec<(u128, usize)> = Vec::new();
    for d in 1..m {
        if deps[d].children[0] != NO_DEP {
            continue;
        }
        leaves += 1;
        leaf_ends[state_of(d)] += 1;
        let (shed, brim) = (deps[d].supply, deps[d].brim_supply);
        if brim > 0 {
            ratio.push(shed as f64 / brim as f64);
        } else if shed > 0 {
            fed_above += 1;
        }
        gains.push((shed.saturating_sub(brim), d));
    }
    ratio.sort_by(f64::total_cmp);
    gains.sort_unstable_by_key(|&(g, d)| (std::cmp::Reverse(g), d));
    let pick = |q: usize| ratio.get(ratio.len() * q / 100).copied().unwrap_or(0.0);
    let (mut land_m2, mut lake_m2) = (0u128, 0u128);
    for ((&ocean, &area), &f) in state.ocean.iter().zip(&state.area).zip(facies) {
        if ocean {
            continue;
        }
        let a = u128::from(area);
        land_m2 += a;
        lake_m2 += a * u128::from(f & FACIES_LAKE != 0);
    }
    let lake_pct = lake_m2 as f64 * 100.0 / land_m2.max(1) as f64;
    let km3_yr = |v: u128| v as f64 / 1.0e12;
    println!(
        "\nTHE LAKE BUDGET (plan step 2; ruling B7; status §7 W3, W4): each hollow fed from its whole WATERSHED down the ground, every full hollow's overflow passed on (Fill–Spill–Merge, Barnes, Callaghan & Wickert 2021)"
    );
    println!(
        "  {leaves} leaf hollows; the watershed's supply over the runoff of the hollow's own nodes (the rule before step 2): median {:.2} times, p90 {:.2} times, largest {:.1} times; {fed_above} hollows with no runoff of their own take water only from above their brim",
        pick(50),
        pick(90),
        ratio.last().copied().unwrap_or(0.0)
    );
    println!(
        "  the water bodies: DRY {} PARTIAL {} SPILLING {}; the leaves by the body over them: DRY {} PARTIAL {} SPILLING {}",
        r.dry, r.partial, r.spilling, leaf_ends[0], leaf_ends[1], leaf_ends[2]
    );
    println!(
        "  the cascade: {} pours into another hollow's side of a saddle, {} into the ocean's side that the ground leads into a lower hollow (plan step 2b, question 1), {} off saddles of three sides or more (question 2); its walk visited {} rings over {} leaves (the skip over full rings, review finding 6); the books, km³/yr: runoff {:.4} = held and evaporated {:.4} + to the ocean {:.4} + stranded {:.4}",
        r.transfers,
        r.ocean_pours,
        r.saddle_pours,
        r.cascade_steps,
        leaves,
        km3_yr(r.runoff_total),
        km3_yr(r.evaporated_total),
        km3_yr(r.to_ocean),
        km3_yr(r.stranded)
    );
    println!(
        "  the standing water: {} nodes, {:.3e} km², {:.4e} km³; the lakes cover {lake_pct:.3} % of the land (every node that is not the ocean) — G-LAKE-SHARE's ceiling 4 % (Verpoorter et al. 2014), W11 names 0.2 to 0.8 % for dry interiors",
        r.lake_nodes,
        r.lake_area_m2 as f64 / 1.0e6,
        r.volume_m3 as f64 / 1.0e9
    );
    println!("  the ten hollows the watershed feeds most beyond their own nodes:");
    for &(_, d) in gains.iter().take(10) {
        let dep = &deps[d];
        let aim = unit(&state.lattice, dep.pit);
        let spill = if dep.spill_z == i32::MAX {
            "it never spills".to_string()
        } else {
            format!(
                "it spills {:.0} m over the sea",
                (i64::from(dep.spill_z) - i64::from(state.sea_z)) as f64 / steps
            )
        };
        println!(
            "    {:>8.0} km², {spill}; its own nodes' runoff {:.4} km³/yr, its watershed's {:.4} km³/yr; {} — aim {:.6} {:.6} {:.6}",
            dep.area_m2 as f64 / 1.0e6,
            km3_yr(dep.brim_supply),
            km3_yr(dep.supply),
            STATES[state_of(d)],
            aim.x,
            aim.y,
            aim.z
        );
    }
    // ★ THE ROW AGAINST THE BUDGET'S OWN WATER (review finding 3 of plan step 2; plan question 3;
    // since plan step 2b, ruling B8 item 2: the row IS the budget's own water, so each reading is
    // a gate with a band of zero):
    // 1. ROW-SEA: the row that reaches the OCEAN from the land (at every land node whose shipped
    //    receiver is an ocean node) against the cascade's own water to the ocean — equal to the
    //    unit;
    // 2. ROW-OUT-OF-CLOSED-HOLLOWS: for every hollow that spills into the sea (its parent is the
    //    ocean) and is NOT full, the row on every shipped receiver edge that leaves the hollow's own
    //    nodes — zero, because the budget holds all the water that reaches such a hollow;
    // 3. ROW-HELD: for every water body that is not full, the rows at its ENDS (its nodes with no
    //    receiver) summed, against the water the body holds — equal, body by body; an end inside a
    //    spilling body is counted apart (a defect).
    let mut row_to_ocean = 0u128;
    for i in 0..n {
        let to = state.receiver[i];
        if !state.ocean[i] && to != NO_NODE && state.ocean[to as usize] {
            row_to_ocean += u128::from(state.discharge[i]);
        }
    }
    // The hollow each depression lies in that spills into the sea: a parent is pushed after its
    // children, so a walk from the last index down meets every parent first.
    let mut top = vec![NO_DEP; m];
    for d in (1..m).rev() {
        let p = deps[d].parent;
        top[d] = if p == OCEAN || p == NO_DEP {
            d as u32
        } else {
            top[p as usize]
        };
    }
    let top_of = |node: usize| -> u32 {
        let h = lakes.home[node];
        if h == OCEAN || h == NO_DEP {
            NO_DEP
        } else {
            top[h as usize]
        }
    };
    let mut out_of = vec![0u128; m];
    for i in 0..n {
        let t = top_of(i);
        let to = state.receiver[i];
        if t == NO_DEP || to == NO_NODE || top_of(to as usize) == t {
            continue;
        }
        out_of[t as usize] += u128::from(state.discharge[i]);
    }
    let (mut closed, mut leaking, mut closed_row, mut worst_row, mut worst_d) =
        (0usize, 0usize, 0u128, 0u128, 0usize);
    for d in 1..m {
        if top[d] != d as u32 || deps[d].parent != OCEAN || deps[d].full {
            continue;
        }
        closed += 1;
        leaking += usize::from(out_of[d] > 0);
        closed_row += out_of[d];
        if out_of[d] > worst_row {
            (worst_row, worst_d) = (out_of[d], d);
        }
    }
    // The ends of every body.
    let mut ends_of = vec![0u128; m];
    let (mut ends_full, mut ends_none) = (0usize, 0usize);
    for i in 0..n {
        if state.receiver[i] != NO_NODE || state.ocean[i] {
            continue;
        }
        let b = body_at(lakes, i);
        if b == NO_DEP {
            ends_none += 1;
        } else if deps[b as usize].full {
            ends_full += 1;
        } else {
            ends_of[b as usize] += u128::from(state.discharge[i]);
        }
    }
    let (mut bodies_closed, mut held_off, mut held_off_water) = (0usize, 0usize, 0u128);
    for d in 1..m {
        if lakes.body_of[d] != d as u32 || deps[d].full {
            continue;
        }
        bodies_closed += 1;
        if ends_of[d] != deps[d].held {
            held_off += 1;
            held_off_water += ends_of[d].abs_diff(deps[d].held);
            if held_off <= 5 {
                let aim = unit(&state.lattice, deps[d].pit);
                println!(
                    "    ROW-HELD off: body {d} (pit {} at {:.0} m, spill {:.0} m, level {}), children {:?}, held {} against ends {}; aim {:.6} {:.6} {:.6}",
                    deps[d].pit,
                    f64::from(deps[d].pit_z) / steps,
                    f64::from(deps[d].spill_z) / steps,
                    if deps[d].level == DRY {
                        "DRY".to_string()
                    } else {
                        format!("{:.1} m", f64::from(deps[d].level) / steps)
                    },
                    deps[d].children,
                    deps[d].held,
                    ends_of[d],
                    aim.x,
                    aim.y,
                    aim.z
                );
            }
        }
    }
    println!(
        "  THE ROW AGAINST THE BUDGET (review finding 3; ruling B8 item 2): the discharge row delivers {:.4} km³/yr from the land to the ocean, the cascade {:.4} km³/yr (a gap of {:.4} km³/yr; {:.4} km³/yr held); of the {closed} hollows that spill into the sea and are NOT full (the budget holds all their water), {leaking} have a row leaving them; the largest row out of one is {:.4} km³/yr, out of a {:.0} km² hollow whose own water is {} (the rows out of them sum to {:.1} km³/yr)",
        km3_yr(row_to_ocean),
        km3_yr(r.to_ocean),
        km3_yr(row_to_ocean.abs_diff(r.to_ocean)),
        km3_yr(r.evaporated_total),
        km3_yr(worst_row),
        deps[worst_d].area_m2 as f64 / 1.0e6,
        STATES[state_of(worst_d)],
        km3_yr(closed_row)
    );
    println!(
        "  THE ROW AT THE ENDS (plan step 2b): of {bodies_closed} water bodies that are not full, {held_off} have ends whose rows do not sum to the water they hold (off by {:.4} km³/yr in all); ends inside a spilling body {ends_full}, ends under no body {ends_none}",
        km3_yr(held_off_water)
    );
    census(
        "ROW-SEA",
        &format!("{}row-to-ocean/{}budget-to-ocean", row_to_ocean, r.to_ocean),
        "equal[ruling-B8-item-2:the-row's-sum-into-the-sea-is-the-cascade's]",
        Verdict::of(row_to_ocean == r.to_ocean),
    );
    census(
        "ROW-OUT-OF-CLOSED-HOLLOWS",
        &format!(
            "{:.1}km3/yr-row-to-ocean/{:.1}km3/yr-budget/{leaking}-of-{closed}-closed-hollows-leak",
            km3_yr(row_to_ocean),
            km3_yr(r.to_ocean)
        ),
        "0-leak[ruling-B8-item-2:a-closed-body-passes-nothing-on]",
        Verdict::of(leaking == 0 && row_to_ocean == r.to_ocean),
    );
    census(
        "ROW-HELD",
        &format!(
            "{held_off}of{bodies_closed}bodies/{ends_full}ends-in-spilling/{ends_none}ends-in-none"
        ),
        "0[plan-step-2b:the-ends-of-a-closed-body-carry-what-it-holds]",
        Verdict::of(held_off == 0 && ends_full == 0 && ends_none == 0),
    );
    census(
        "CHECK-BUDGET-STALE",
        &format!("{stale}nodes"),
        "0[the-budget-run-here-is-the-solve's-own]",
        Verdict::of(stale == 0),
    );
    census(
        "CHECK-BUDGET-CONSERVE",
        &format!(
            "{}runoff-{}held-{}ocean-{}stranded",
            r.runoff_total, r.evaporated_total, r.to_ocean, r.stranded
        ),
        "exact[ruling-B7:nothing-is-lost]",
        Verdict::of(r.runoff_total == r.evaporated_total + r.to_ocean + r.stranded),
    );
    census(
        "G-LAKE-SHARE",
        &format!("{lake_pct:.3}%of-land"),
        "<=4%[ruling-B3;Verpoorter-et-al-2014]",
        Verdict::of(lake_pct <= 4.0),
    );
    census(
        "BUDGET-LEAVES",
        &format!(
            "DRY{}/PARTIAL{}/SPILLING{}/transfers{}",
            leaf_ends[0], leaf_ends[1], leaf_ends[2], r.transfers
        ),
        "no-band[a-reading]",
        Verdict::NoBand,
    );
}

/// Whether a node is a ROOT of the budget's hierarchy: the ocean's own component with no way down
/// (the connected ocean, or the six lowest nodes of a body with no sea).
fn is_root(lakes: &Lakes, i: usize) -> bool {
    lakes.home[i] == OCEAN && lakes.descent.down[i] == NO_NODE
}

/// ★ PLAN STEP 2b, W0 — MEASURE BEFORE THE CASCADE OR THE DESCENT MOVES (ruling B8 item 2). Three
/// readings the owner reads with the plan's questions 1 to 3; none of them moves the world.
///
/// 1. **OCEAN-SIDE-POURS (question 1).** A full hollow whose parent is the ocean hands its overflow
///    to the sea in the cascade. The ground's own descent from the saddle's ocean-side gate can end
///    in a lower closed hollow that met the sea earlier; no single receiver can then carry that
///    overflow through the hollow to the sea. Counted with the water, and split by whether the
///    hollow it lands in is full.
/// 2. **SADDLE-SIDES (question 2).** Saddles by their count of sides, the ocean's side included; at
///    a saddle of three sides or more, the full depressions whose pour today (the sibling the
///    binary merge paired them with, or the sea) is not the steepest side that is not full.
/// 3. **GROUND-PLATEAUS (question 3).** Connected nodes of one exact height (not the root): how
///    many, how many drain and how many are closed; the leaves of zero depth the sweep opens on
///    them and the water those hold; and the trunks of the budget's own DESCENT tree that run over
///    a plateau, with their grid lean beside the lean of the trunks off the plateaus.
///
/// **Example.** A crater lake on the belt's shoulder spills down its seaward rim into a closed
/// valley hollow that met the sea over a lower sill. Line 1 counts that crater's overflow: the
/// cascade books it to the sea, and the ground carries it into the valley.
fn step2b_measure_lines(state: &MacroSolve, lakes: &Lakes) {
    use vd_bins::grid_bias::{AxisHistogram, trunk_axis};
    let n = state.node_count();
    let deps = &lakes.deps;
    let km3_yr = |v: u128| v as f64 / 1.0e12;
    // 1. THE OCEAN SIDE'S POURS.
    let (mut pours, mut pours_water) = (0usize, 0u128);
    let (mut into_hollow, mut into_hollow_water) = (0usize, 0u128);
    let (mut into_closed, mut into_closed_water) = (0usize, 0u128);
    for d in 1..deps.len() {
        let dep = &deps[d];
        if dep.parent != OCEAN || !dep.full || dep.passed == 0 {
            continue;
        }
        pours += 1;
        pours_water += dep.passed;
        let saddle = &lakes.saddles[dep.saddle as usize];
        let ocean_side = saddle
            .sides
            .iter()
            .find(|k| k.root == OCEAN)
            .expect("a saddle into the ocean has the ocean's side");
        if ocean_side.entry == OCEAN {
            continue;
        }
        into_hollow += 1;
        into_hollow_water += dep.passed;
        let body = lakes.body_of[ocean_side.entry as usize];
        if body == NO_DEP || !deps[body as usize].full {
            into_closed += 1;
            into_closed_water += dep.passed;
        }
    }
    println!(
        "\nPLAN STEP 2b, W0 (ruling B8 item 2): the three readings before the cascade or the descent moves"
    );
    println!(
        "  M1 OCEAN-SIDE-POURS (question 1): {pours} full hollows pour {:.4} km³/yr into the sea; the ground's descent from the ocean-side gate of {into_hollow} of them ({:.4} km³/yr) ends in a lower hollow that met the sea earlier, {into_closed} of them ({:.4} km³/yr) in one that is NOT full",
        km3_yr(pours_water),
        km3_yr(into_hollow_water),
        km3_yr(into_closed_water)
    );
    census(
        "M1-OCEAN-SIDE-POURS",
        &format!(
            "{into_hollow}pours/{:.4}km3/yr-into-a-hollow/{into_closed}/{:.4}km3/yr-into-a-hollow-not-full",
            km3_yr(into_hollow_water),
            km3_yr(into_closed_water)
        ),
        "no-band[plan-step-2b-question-1]",
        Verdict::NoBand,
    );
    // 2. THE SADDLES' SIDES.
    let mut by_sides = [0usize; 10];
    let (mut multi_full, mut differ, mut differ_water) = (0usize, 0usize, 0u128);
    for saddle in &lakes.saddles {
        by_sides[saddle.sides.len().min(9)] += 1;
    }
    for d in 1..deps.len() {
        let dep = &deps[d];
        if dep.saddle == vd_terrain::lakes::NO_SADDLE || !dep.full || dep.passed == 0 {
            continue;
        }
        let saddle = &lakes.saddles[dep.saddle as usize];
        if saddle.sides.len() < 3 {
            continue;
        }
        // The steepest side that is the ocean or not full, by the final flags.
        let chosen = saddle
            .sides
            .iter()
            .find(|s| s.root == OCEAN || !deps[s.root as usize].full)
            .map_or(NO_NODE, |s| s.gate);
        {
            multi_full += 1;
            let today = if dep.parent == OCEAN {
                saddle
                    .sides
                    .iter()
                    .find(|s| s.root == OCEAN)
                    .map_or(NO_NODE, |s| s.gate)
            } else {
                let kids = deps[dep.parent as usize].children;
                let sibling = if kids[0] == d as u32 {
                    kids[1]
                } else {
                    kids[0]
                };
                deps[sibling as usize].gate
            };
            if today != chosen {
                differ += 1;
                differ_water += dep.passed;
            }
        }
    }
    println!(
        "  M2 SADDLE-SIDES (question 2): saddles by their sides (the ocean's side counted): 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8+:{}; at the saddles of three sides or more {multi_full} full depressions pour on, and {differ} of them ({:.4} km³/yr) pour today into a side that is not the steepest side that is the ocean or not full",
        by_sides[2],
        by_sides[3],
        by_sides[4],
        by_sides[5],
        by_sides[6],
        by_sides[7],
        by_sides[8] + by_sides[9],
        km3_yr(differ_water)
    );
    census(
        "M2-SADDLE-SIDES",
        &format!(
            "{}multi-way-saddles/{differ}pours/{:.4}km3/yr-not-the-steepest-open-side",
            lakes.saddles.iter().filter(|s| s.sides.len() >= 3).count(),
            km3_yr(differ_water)
        ),
        "no-band[plan-step-2b-question-2]",
        Verdict::NoBand,
    );
    // 3. THE GROUND'S PLATEAUS: connected nodes of one exact height that are not the root.
    const UNSEEN: u32 = u32::MAX;
    let mut plateau = vec![UNSEEN; n];
    let (mut draining, mut draining_nodes, mut closed, mut closed_nodes) =
        (0usize, 0usize, 0usize, 0usize);
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut members: Vec<u32> = Vec::new();
    let mut next_id = 0u32;
    for start in 0..n {
        if plateau[start] != UNSEEN || is_root(lakes, start) {
            continue;
        }
        let z0 = state.z[start];
        let has_equal = state
            .lattice
            .neighbours(start as u32)
            .iter()
            .any(|&m| m != NO_NODE && state.z[m as usize] == z0 && !is_root(lakes, m as usize));
        if !has_equal {
            continue;
        }
        members.clear();
        plateau[start] = next_id;
        queue.push_back(start as u32);
        let mut drains = false;
        while let Some(v) = queue.pop_front() {
            members.push(v);
            for m in state.lattice.neighbours(v) {
                if m == NO_NODE {
                    continue;
                }
                let mi = m as usize;
                if is_root(lakes, mi) || state.z[mi] < z0 {
                    drains = true;
                    continue;
                }
                if state.z[mi] == z0 && plateau[mi] == UNSEEN {
                    plateau[mi] = next_id;
                    queue.push_back(m);
                }
            }
        }
        next_id += 1;
        if drains {
            draining += 1;
            draining_nodes += members.len();
        } else {
            closed += 1;
            closed_nodes += members.len();
        }
    }
    let (mut flat_leaves, mut flat_leaf_water, mut flat_leaf_wet) = (0usize, 0u128, 0usize);
    for dep in deps.iter().skip(1) {
        if dep.children[0] != NO_DEP || dep.pit_z != dep.spill_z {
            continue;
        }
        flat_leaves += 1;
        flat_leaf_water += dep.held;
        flat_leaf_wet += usize::from(dep.held > 0);
    }
    // The descent tree's drainage area, leaves first: every node after every node that descends
    // into it, which is the sweep's own order `(height, index)` taken from the top.
    let mut by_height: Vec<u32> = (0..n as u32).collect();
    by_height.sort_unstable_by_key(|&v| std::cmp::Reverse((state.z[v as usize], v)));
    let mut drain = state.area.clone();
    for &v in &by_height {
        let d = lakes.descent.down[v as usize];
        if d != NO_NODE {
            drain[d as usize] += drain[v as usize];
        }
    }
    drop(by_height);
    let mut on_flat = AxisHistogram::new(4);
    let mut off_flat = AxisHistogram::new(4);
    let (mut trunks, mut trunks_on_flat) = (0usize, 0usize);
    for v in 0..n as u32 {
        let i = v as usize;
        if state.ocean[i] || drain[i] < state.area[i].saturating_mul(VALLEY_NODES) {
            continue;
        }
        trunks += 1;
        let flat = plateau[i] != UNSEEN;
        trunks_on_flat += usize::from(flat);
        if let Some(axis) = trunk_axis(&state.lattice, &lakes.descent.down, v, 8) {
            if flat {
                on_flat.add(axis.degrees);
            } else {
                off_flat.add(axis.degrees);
            }
        }
    }
    println!(
        "  M3 GROUND-PLATEAUS (question 3): {} plateaus of two nodes or more — {draining} drain ({draining_nodes} nodes), {closed} are closed ({closed_nodes} nodes); {flat_leaves} leaves of ZERO depth (pit at the spill's height), {flat_leaf_wet} of them hold water, {:.4} km³/yr in all",
        draining + closed,
        km3_yr(flat_leaf_water)
    );
    println!(
        "  the budget's DESCENT tree ({VALLEY_NODES} nodes of drainage area or more): {trunks} trunk nodes, {trunks_on_flat} on a plateau; the grid lean of their 8-step axes ON a plateau: {} | FLATNESS {:.3}; OFF the plateaus: {} | FLATNESS {:.3}",
        on_flat.line(),
        on_flat.flatness(),
        off_flat.line(),
        off_flat.flatness()
    );
    census(
        "M3-GROUND-PLATEAUS",
        &format!(
            "{draining_nodes}draining-nodes/{closed_nodes}closed-nodes/{flat_leaves}zero-depth-leaves/{trunks_on_flat}of{trunks}descent-trunks-on-a-plateau/flatness{:.3}on/{:.3}off",
            on_flat.flatness(),
            off_flat.flatness()
        ),
        "no-band[plan-step-2b-question-3]",
        Verdict::NoBand,
    );
}

/// ★ THE WALK ACROSS A SPILLING LAKE (plan step 2b, question 4; the planet's own reading of test T9):
/// from every node of a spilling lake's hollow whose water takes SIX steps inside that hollow (the
/// spill may be the sixth), on one face, the angle between the six steps' displacement and the
/// straight line from the start to the lake's spill, and whether all six steps are one lattice
/// offset (an exact run along a row or a diagonal). Beside it, from the same starts, THE CONTROL:
/// the rule before the split (the neighbour nearest the spill by the summed chord plus the step,
/// ties to the smaller index), walked over the census's OWN summed chord from each spill; and the
/// axes of the straight lines themselves, so a lean of the lakes' shapes is told from a lean of the
/// step.
///
/// **Example.** A lake on the plain spills at a notch 20° off the face's row. From its far shore
/// the river's six steps mix rows and diagonals and point at the notch, never along the row.
fn lake_walk_lines(state: &MacroSolve, lakes: &Lakes, radius_m: f64) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use vd_bins::grid_bias::{AxisHistogram, pair_axis};
    const STEPS: usize = 6;
    struct Tally {
        walks: usize,
        runs: usize,
        errors: Vec<f64>,
        axes: AxisHistogram,
    }
    let n = state.node_count();
    let class = classes(state, lakes);
    let lattice = &state.lattice;
    // The census's own summed chord from each spilling lake's spill over its hollow.
    let chord = |a: u32, b: u32| -> u64 {
        ((unit(lattice, a) - unit(lattice, b)).length() * radius_m).floor() as u64
    };
    let mut dist = vec![u64::MAX; n];
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for (d, dep) in lakes.deps.iter().enumerate().skip(1) {
        if lakes.body_of[d] == d as u32 && dep.full {
            dist[dep.spill_node as usize] = 0;
            heap.push(Reverse((0, dep.spill_node)));
        }
    }
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        let from = if class[u as usize] == Class::Spilling {
            body_at(lakes, u as usize)
        } else {
            NO_DEP
        };
        for m in lattice.neighbours(u) {
            if m == NO_NODE || class[m as usize] != Class::Spilling {
                continue;
            }
            let bm = body_at(lakes, m as usize);
            let joins = bm == from || (from == NO_DEP && lakes.deps[bm as usize].spill_node == u);
            let step = d + chord(u, m);
            if joins && step < dist[m as usize] {
                dist[m as usize] = step;
                heap.push(Reverse((step, m)));
            }
        }
    }
    let control_step = |i: u32, b: u32, spill: u32| -> u32 {
        let mut best = (u64::MAX, NO_NODE);
        for m in lattice.neighbours(i) {
            if m == NO_NODE {
                continue;
            }
            let own = m == spill
                || (class[m as usize] == Class::Spilling && body_at(lakes, m as usize) == b);
            if own && dist[m as usize] < dist[i as usize] {
                best = best.min((dist[m as usize] + chord(i, m), m));
            }
        }
        best.1
    };
    // One walk: where it ends and its six offsets, or None where it leaves the hollow or the face.
    let walk = |start: u32,
                b: u32,
                spill: u32,
                step: &dyn Fn(u32) -> u32|
     -> Option<(u32, Vec<(i32, i32)>)> {
        let face = lattice.split(start).0;
        let mut cur = start;
        let mut offsets = Vec::with_capacity(STEPS);
        for k in 0..STEPS {
            let r = step(cur);
            let inside = r != NO_NODE
                && ((r == spill && k == STEPS - 1)
                    || (r != spill
                        && class[r as usize] == Class::Spilling
                        && body_at(lakes, r as usize) == b));
            if !inside {
                return None;
            }
            let (f, a, c) = lattice.split(cur);
            let (fr, ar, cr) = lattice.split(r);
            if f != face || fr != face {
                return None;
            }
            offsets.push((ar - a, cr - c));
            cur = r;
        }
        Some((cur, offsets))
    };
    let new_tally = || Tally {
        walks: 0,
        runs: 0,
        errors: Vec::new(),
        axes: AxisHistogram::new(4),
    };
    let (mut shipped, mut control, mut lines) = (new_tally(), new_tally(), AxisHistogram::new(4));
    let add = |t: &mut Tally, start: u32, spill: u32, end: u32, offsets: &[(i32, i32)]| {
        let (_, i0, j0) = lattice.split(start);
        let (_, i1, j1) = lattice.split(end);
        let (_, si, sj) = lattice.split(spill);
        let walked = (f64::from(i1 - i0), f64::from(j1 - j0));
        let line = (f64::from(si - i0), f64::from(sj - j0));
        let dot = walked.0 * line.0 + walked.1 * line.1;
        let cross = walked.0 * line.1 - walked.1 * line.0;
        t.errors.push(cross.atan2(dot).abs().to_degrees());
        t.walks += 1;
        t.runs += usize::from(offsets.iter().all(|&o| o == offsets[0]));
        if let Some(axis) = pair_axis(lattice, start, end) {
            t.axes.add(axis.degrees);
        }
    };
    for start in 0..n as u32 {
        if class[start as usize] != Class::Spilling {
            continue;
        }
        let b = body_at(lakes, start as usize);
        let spill = lakes.deps[b as usize].spill_node;
        if lattice.split(spill).0 != lattice.split(start).0 {
            continue;
        }
        let shipped_walk = walk(start, b, spill, &|i| state.receiver[i as usize]);
        let control_walk = walk(start, b, spill, &|i| control_step(i, b, spill));
        // Both rules from the same starts only.
        let (Some((e1, o1)), Some((e2, o2))) = (shipped_walk, control_walk) else {
            continue;
        };
        add(&mut shipped, start, spill, e1, &o1);
        add(&mut control, start, spill, e2, &o2);
        if let Some(axis) = pair_axis(lattice, start, spill) {
            lines.add(axis.degrees);
        }
    }
    let summary = |t: &mut Tally| -> (f64, f64, f64) {
        t.errors.sort_by(f64::total_cmp);
        let mean = t.errors.iter().sum::<f64>() / t.errors.len().max(1) as f64;
        let p90 = t
            .errors
            .get(t.errors.len() * 9 / 10)
            .copied()
            .unwrap_or(0.0);
        (mean, p90, t.runs as f64 * 100.0 / t.walks.max(1) as f64)
    };
    let (mean, p90, run_pct) = summary(&mut shipped);
    let (c_mean, c_p90, c_run_pct) = summary(&mut control);
    println!(
        "\nTHE WALK ACROSS A SPILLING LAKE (plan step 2b, question 4): {} six-step walks from the same starts inside their lake's hollow toward its spill; the bearing error against the straight line to the spill, SHIPPED (the split): mean {mean:.2}°, p90 {p90:.2}°, exact lattice runs {run_pct:.1} %, the walks' axes {} | FLATNESS {:.3}; THE CONTROL (the nearest by the summed chord, the rule before the split): mean {c_mean:.2}°, p90 {c_p90:.2}°, exact runs {c_run_pct:.1} %, axes {} | FLATNESS {:.3}; the straight lines to the spills themselves: {} | FLATNESS {:.3}",
        shipped.walks,
        shipped.axes.line(),
        shipped.axes.flatness(),
        control.axes.line(),
        control.axes.flatness(),
        lines.line(),
        lines.flatness()
    );
    census(
        "G-LAKE-WALK",
        &format!(
            "{mean:.2}deg-mean/{run_pct:.1}%runs/flatness{:.3}/control{c_mean:.2}deg/{c_run_pct:.1}%runs/flatness{:.3}/lines-flatness{:.3}/{}walks",
            shipped.axes.flatness(),
            control.axes.flatness(),
            lines.flatness(),
            shipped.walks
        ),
        // ★ A GATE, NOT A READING (review finding 5 of step 2b): test T9's 12° bound on the mean
        //   error stays the gate until the owner rules another bound for lakes that are node
        //   squares (status §11 item 11). A gate is never turned into a reading after it fails.
        "<=12deg-mean[plan-step-2b-question-4:test-T9's-bound;the-owner's-word-owed-for-node-square-lakes]",
        Verdict::of(mean <= 12.0),
    );
}
