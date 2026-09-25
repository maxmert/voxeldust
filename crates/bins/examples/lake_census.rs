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
//! 4. **THE LAKES AT THE RUNG SWAPS** (ruling W16; status §6 T5, §7 W6; plan step 2 part 5): the lake
//!    columns whose drawn water changes at the swaps 9 → 10, 14 → 15 and 15 → 16 (LOST, MIXED,
//!    SHIFTED), the dry columns the shore law lifts over a lake (the lip), and the dry columns whose
//!    INLAND/LAND word changes while their ground stands under the sea (the hold), read on the
//!    SHIPPED column.
//!    ★ And THE POCKET'S FLOOR (review finding 1 of plan step 2): at each rung, the columns over a
//!    pocket's dry floor that the shore law holds over the sea (G-POCKET-HELD).
//! 5. **THE LAKE BUDGET** (status §7 W3, W4; plan step 2 parts 1 and 2): each hollow's watershed
//!    supply beside the runoff of its own nodes, the three ends, the cascade's pours and its books
//!    (exact), and G-LAKE-SHARE. Step 1 printed this as a WHAT-IF; since step 2 it is the budget.
//!    ★ And THE ROW AT THE SPILLS (review finding 3 of plan step 2): the discharge row at each
//!    hollow's spill against the water the cascade passes on there.
//!
//! WHAT IT READS: the solve's final state (the heights, the receivers, the routing fill, the ocean,
//! the discharge row), the lake budget's own hierarchy and descent (`lakes::budget`, run again over
//! the same climate), the shipped artifact's rows, its level-1 and level-2 water words and its coast
//! mask, and THE SHIPPED COLUMN on each side of every rung swap (`vd_bins::shore_geometry::RungReader`:
//! the chunk builder's `sample_side`, `column_water`, `sample_z` and the one kernel, on the body with
//! its solved sea). It never re-implements a column. Every gate prints one
//! `CENSUS <gate> <value> <band> <verdict>` line with no space inside a field; every self-check
//! prints a `CENSUS CHECK-…` line of the same shape; and the world's versions head the output, so a
//! reading names its world.
//!
//! **Example.** A pilot over the belt sees a pond in a hollow from 150 km and, since plan step 2,
//! still sees it from 500 km: the census reads the pond's column on both sides of the rung 9 → 10
//! swap and finds the same water. The same pass walks the river off the range beside it down the
//! shipped receivers into a hollow the range's whole watershed now feeds, and the hollow spills.
//!
//! `cargo run --release -p vd-bins --example lake_census -- [dx dy dz] [half_nodes]`

use std::collections::VecDeque;

use vd_bins::grid_bias::unit;
// ★ THE SHIPPED COLUMN, ONE COPY (plan step 1, ruling B6): the chunk builder's own reads in the
// chunk builder's order, shared with the teeth instrument, so the rung swap is measured on the
// path the client draws and never on a copy of it.
use vd_bins::shore_geometry::{RungReader, ShippedColumn};
use vd_core::glam::DVec3;
use vd_recipe::Gi;
use vd_recipe::height::{AMP_BITS, SIDE_INLAND, SIDE_LAND, SIDE_SEA};
use vd_recipe::noise::NOISE_BITS;
use vd_terrain::artifact::{
    Artifact, DRY_M, FACIES_SHIFT, PyramidField, Side, block_water, coast_level, coast_side,
    log2_class,
};
use vd_terrain::climate::Climate;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::lakes::{Descent, Lakes, NO_DEP, OCEAN};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{DRY, FACIES_LAKE, FACIES_SEA, MacroSolve, Z_STEPS_PER_M};
use vd_terrain::{BodyDefinition, metres_of_q28};

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
    let mut schedule = vd_terrain::solve::Schedule::standard(
        std::env::var("VD_CENSUS_AGE_YR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(vd_terrain::home::HOME_SYSTEM_AGE_YR),
    );
    schedule.passes = env("VD_CENSUS_PASSES", schedule.passes);
    schedule.flood_every = env("VD_CENSUS_FLOOD_EVERY", schedule.flood_every);
    schedule.climate_every = env("VD_CENSUS_CLIMATE_EVERY", schedule.climate_every);
    schedule.isostasy_every = env("VD_CENSUS_ISOSTASY_EVERY", schedule.isostasy_every);
    schedule.talus_passes = env("VD_CENSUS_TALUS_PASSES", schedule.talus_passes);
    println!("schedule: {schedule:?}");
    let started = std::time::Instant::now();
    let (state, facies_v, report) =
        vd_terrain::solve::solve_full(&body, &words, schedule).expect("the home planet solves");
    println!(
        "solved in {:.1} s; envelope worst {} steps, clamped {}; ice: {} nodes under the line, thickest {:.0} m, deepest cut {} steps",
        started.elapsed().as_secs_f64(),
        report.envelope.0,
        report.envelope.1,
        report.ice.0,
        report.ice.1,
        report.ice.2
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
            Some(before) => println!(
                "THE DOUBLE COUNT: the sea was re-solved over the inventory less the lakes: {:.0} m → {:.0} m",
                f64::from(before) / steps,
                f64::from(state.sea_z) / steps
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
        "\ncraters stamped: {}; routes: {} (every {} of {} passes, plus the final); pits RAISED by each flood (= lake nodes at that route):",
        report.craters,
        report.routes.len(),
        vd_terrain::solve::FLOOD_EVERY,
        vd_terrain::solve::PASSES
    );
    for (k, r) in report.routes.iter().enumerate() {
        let sw = report.sweeps.get(k);
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
    hypsometry_lines(&state);
    let ocean = inland_sea_lines(&state, &lakes, &facies, first_land);
    river_lines(&state, &lakes, &facies, &ocean, &by_area, "");
    river_lines(&state, &lakes, &facies, &ocean, &by_rain, ":step1-rule");
    discharge_tree_lines(&state, &lakes.descent, &climate);
    // The body WITH ITS SOLVED SEA, as every host that holds the artifact builds it: the shipped
    // column reads its sea radius, and the census's own `body` has none.
    let sea_body = home_planet().with_sea_m(artifact.sea());
    rung_swap_lines(&sea_body, &state.lattice, &artifact, &patches);
    budget_lines(&state, &lakes, &facies);
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
fn hypsometry_lines(state: &MacroSolve) {
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
    let integrals = state.hypsometric_integrals(vd_terrain::solve::AGE_GATE_MIN_NODES);
    let (mature_lo, mature_hi) = vd_terrain::solve::HI_MATURE;
    let age_band = format!("{mature_lo:.2}-{mature_hi:.2}[Strahler-1952;solve.rs-HI_MATURE]");
    match vd_terrain::solve::age_gate(&integrals) {
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
    /// A lake with no outlet: the budget stands it under its spill.
    ClosedLake,
    /// A closed hollow the budget leaves dry.
    DryBasin,
    /// A land node with no receiver: a defect.
    NoReceiver,
    /// A node on a ring of receivers: a defect.
    Cycle,
}

/// ★ THE END AT ONE NODE, or [`End::Runs`]. The ocean ends a river. A hollow the routing filled ends
/// it where the lake budget holds no water in that hollow (a dry basin) or holds a lake under its
/// spill (a closed lake); where the budget's lake spills, the river runs on over the spill. The
/// hollow is the one the GROUND leads into: the leaf whose pit ends this node's descent
/// (`vd_terrain::lakes::Descent`, the budget's own). A pocket under the sea's level is such a
/// hollow since plan step 2.
fn end_at(state: &MacroSolve, lakes: &Lakes, facies: &[u8], ocean: &[bool], i: usize) -> End {
    if ocean[i] {
        return End::Ocean;
    }
    if facies[i] & FACIES_SEA != 0 {
        return End::InlandSea;
    }
    if state.pit[i] == 0 {
        return End::Runs;
    }
    let pit = lakes.descent.end[i] as usize;
    let leaf = lakes.home[pit];
    if leaf == NO_DEP || leaf == OCEAN {
        return End::Runs;
    }
    let body = lakes.body_of[leaf as usize];
    if body == NO_DEP {
        return End::Runs;
    }
    let water = &lakes.deps[body as usize];
    if water.level == DRY {
        End::DryBasin
    } else if water.full {
        End::Runs
    } else {
        End::ClosedLake
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
            End::NoReceiver
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
    let [
        _,
        to_ocean,
        to_pocket,
        to_lake,
        to_basin,
        no_receiver,
        cycle,
    ] = ends;
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
        "  the ends, in heads: the ocean {to_ocean}, a node the facies call sea that is not the ocean {to_pocket} (0 by construction since plan step 2), a closed lake {to_lake}, a dry closed basin {to_basin}; DEFECTS: no receiver on land {no_receiver}, a ring of receivers {cycle}"
    );
    println!(
        "  (a hollow the routing filled — an inland pocket included — ends a river where the lake budget holds no water in it, a dry basin, or holds a lake under its spill, a closed lake; a spilling lake passes the river on)"
    );
    println!(
        "  along the trunks: {walked} steps; the GROUND rises on {ground_rises} of them ({hidden} under a water's flat surface); the FLOOR (the ground, or the water's surface where water stands) rises on {floor_rises} steps, {in_fill} of them in a hollow the routing filled"
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
        &format!("{}heads-without-an-end", no_receiver + cycle),
        "0heads[ruling-B3-G-RIVER]",
        Verdict::of(no_receiver + cycle == 0),
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

/// ★ THE DISCHARGE ROW AGAINST THE TREE IT SHIPS WITH (status §7 W10; ruling B7 item 2). Since
/// plan step 2 the solve accumulates the RUNOFF over the receivers of its LAST routing
/// (`MacroSolve::accumulate_runoff`), so the shipped row and the shipped receivers are one tree by
/// construction. Three readings:
///
/// 1. The shipped receiver against the steepest lower neighbour on the final GROUND, split by why
///    they may differ: inside a hollow the routing filled (the fill routes over its flat, by
///    design), toward a filled hollow (the fill hides the ground's own way, by design), and
///    elsewhere (a chord that differs by its last metre).
/// 2. The climate's RUNOFF row accumulated again over the SHIPPED tree, against the shipped row:
///    the nodes that differ, and those whose SHIPPED byte (the log class) differs.
/// 3. The nodes where the shipped row FALLS downstream, which no tree can hold.
///
/// **Example.** A river on the coast reaches the sea by a new mouth after the final route. Its row
/// now names that mouth, and the old mouth carries only the water of its own small catchment.
fn discharge_tree_lines(state: &MacroSolve, descent: &Descent, climate: &Climate) {
    let n = state.node_count();
    let (mut compared, mut in_fill, mut into_fill, mut elsewhere) =
        (0usize, 0usize, 0usize, 0usize);
    for node in 0..n as u32 {
        let i = node as usize;
        let d = descent.down[i];
        if d == NO_NODE || state.z[d as usize] >= state.z[i] {
            continue;
        }
        compared += 1;
        if state.receiver[i] == d {
            continue;
        }
        if state.pit[i] > 0 {
            in_fill += 1;
        } else if state.pit[d as usize] > 0 {
            into_fill += 1;
        } else {
            elsewhere += 1;
        }
    }
    let mut again: Vec<u64> = climate
        .runoff_mm_yr
        .iter()
        .zip(&state.area)
        .map(|(&q, &a)| u64::from(q) * a)
        .collect();
    for &node in &state.order {
        let r = state.receiver[node as usize];
        if r != NO_NODE {
            again[r as usize] += again[node as usize];
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
        if r != NO_NODE && state.discharge[r as usize] < row {
            falls += 1;
        }
    }
    println!(
        "\nTHE DISCHARGE ROW AND ITS TREE (status §7 W10; ruling B7 item 2: the row is the RUNOFF over the final receivers)"
    );
    println!(
        "  the shipped receiver against the steepest lower neighbour on the final ground: {compared} land nodes have a lower neighbour; the two differ at {in_fill} nodes inside a hollow the routing filled (by design), {into_fill} nodes whose ground leads into a filled hollow (by design), and {elsewhere} nodes elsewhere (a chord's last metre)"
    );
    println!(
        "  the climate's runoff accumulated again over the shipped tree: {differ} nodes differ (the worst by {worst:.1} times), {class_differ} of them in the SHIPPED byte (the row's log class); the shipped row FALLS downstream at {falls} nodes, which no tree can hold"
    );
    census(
        "W10-DISCHARGE-TREE",
        &format!("{class_differ}nodes"),
        "0nodes[ruling-B7-item-2:the-runoff-over-the-final-receivers]",
        Verdict::of(class_differ == 0),
    );
}

/// What one lake column did at a rung swap.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Change {
    /// The same water at both rungs.
    Kept,
    /// Held under water before, over it after: the lake vanished.
    Lost,
    /// A lake's level before, the sea's after.
    Mixed,
    /// A lake's level before, another lake's after.
    Shifted,
}

/// The change of one lake column `a` (the finer rung) against `b` (the coarser rung) at one swap,
/// or `None` where `a` is not a lake held under its water.
fn lake_change(a: &ShippedColumn, b: &ShippedColumn, sea: Gi) -> Option<Change> {
    if a.side_word != SIDE_SEA || a.water == sea {
        return None;
    }
    Some(if b.side_word != SIDE_SEA {
        Change::Lost
    } else if b.water == a.water {
        Change::Kept
    } else if b.water == sea {
        Change::Mixed
    } else {
        Change::Shifted
    })
}

/// Whether a dry column's INLAND/LAND word changed at a swap while its ground base stood under the
/// sea's level at one of the two rungs: the shore law then lifts it at one rung and leaves it at
/// the other (plan step 2 risk R7; gate G-RUNG-HOLD).
fn hold_changed(a: &ShippedColumn, b: &ShippedColumn) -> bool {
    let dry = |c: &ShippedColumn| (c.side_word == SIDE_LAND) | (c.side_word == SIDE_INLAND);
    dry(a) && dry(b) && a.side_word != b.side_word && (a.base < a.hold || b.base < b.hold)
}

/// The tallies of one swap.
#[derive(Default)]
struct SwapTally {
    /// Lake columns read, and each change.
    lakes: usize,
    lost: usize,
    mixed: usize,
    shifted: usize,
    /// Columns that are no lake before and a lake after.
    gained: usize,
    /// The largest change of level, metres.
    worst_m: f64,
    /// Dry columns read for the hold, and those whose hold changed.
    holds_read: usize,
    holds: usize,
    /// The largest change of surface at such a column, metres.
    hold_step_m: f64,
    /// Columns the field could not read, and columns whose base does not give the kernel's own
    /// surface again (a drift alarm).
    unread: usize,
    base_off: usize,
}

impl SwapTally {
    /// One lake column pair read.
    fn lake(&mut self, a: &ShippedColumn, b: &ShippedColumn, sea: Gi) -> Option<Change> {
        self.base_off += usize::from(!a.base_agrees) + usize::from(!b.base_agrees);
        let c = lake_change(a, b, sea)?;
        self.lakes += 1;
        match c {
            Change::Kept => {}
            Change::Lost => self.lost += 1,
            Change::Mixed => self.mixed += 1,
            Change::Shifted => self.shifted += 1,
        }
        if (c == Change::Mixed) | (c == Change::Shifted) {
            self.worst_m = self.worst_m.max(metres_of_q28(b.water - a.water).abs());
        }
        Some(c)
    }

    /// One dry column pair read for the hold.
    fn hold(&mut self, a: &ShippedColumn, b: &ShippedColumn) {
        self.base_off += usize::from(!a.base_agrees) + usize::from(!b.base_agrees);
        self.holds_read += 1;
        if hold_changed(a, b) {
            self.holds += 1;
            self.hold_step_m = self
                .hold_step_m
                .max(metres_of_q28(b.surface - a.surface).abs());
        }
    }

    /// The CENSUS lines of the swap from rung `before` to rung `after`, trusted or not. The level
    /// steps are stated in pixels at the swap distance too: the swap stands where one cell of
    /// `before` is one pixel high, so a step of `h` metres is `h / cell_m(before)` pixels there.
    fn lines(&self, before: u8, after: u8, trusted: bool) {
        let cell = f64::from(vd_seed::ladder::cell_m(before));
        let px = self.worst_m / cell;
        let tag = format!("@{before}-{after}");
        println!(
            "  the swap {before} → {after}: of {} lake columns, {} turn to LAND, {} take the sea's level, {} take another lake's level; the largest change of level {:.1} m = {px:.3} px at the swap distance (a cell of rung {before} is {cell:.0} m); {} columns are no lake before and a lake after; of {} dry columns read under or beside the sea's level, {} change their INLAND/LAND word, the largest surface step there {:.1} m = {:.3} px; {} unread, {} base alarms",
            self.lakes,
            self.lost,
            self.mixed,
            self.shifted,
            self.worst_m,
            self.gained,
            self.holds_read,
            self.holds,
            self.hold_step_m,
            self.hold_step_m / cell,
            self.unread,
            self.base_off
        );
        census(
            &format!("G-RUNG-LAKE-LOST{tag}"),
            &format!("{}columns", self.lost),
            "0[ruling-W16:a-rung-boundary-is-invisible]",
            Verdict::of(trusted && self.lost == 0),
        );
        census(
            &format!("G-RUNG-LAKE-MIXED{tag}"),
            &format!("{}columns", self.mixed),
            "0[ruling-W16;plan-step-2-part-5:never-a-mean-with-the-sea]",
            Verdict::of(trusted && self.mixed == 0),
        );
        census(
            &format!("G-RUNG-LAKE-SHIFTED{tag}"),
            &format!("{}columns/{:.3}px", self.shifted, px),
            "<=1px[plan-question-6:a-proposed-bound,the-owner's-word-owed]",
            Verdict::of(trusted && px <= 1.0),
        );
        census(
            &format!("G-RUNG-HOLD{tag}"),
            &format!("{}columns/{:.3}px", self.holds, self.hold_step_m / cell),
            "0[ruling-W16;plan-step-2-risk-R7]",
            Verdict::of(trusted && self.holds == 0),
        );
    }
}

/// ★ THE LAKES AT THE RUNG SWAPS (ruling W16: a rung boundary is invisible; status §6 T5, §7 W6;
/// plan step 2 part 5). Three swaps are read, each on THE SHIPPED COLUMN (`RungReader`, the chunk
/// builder's own reads and the one kernel), never on a copy of it:
///
/// 1. **ROWS → LEVEL 1 (rung 9 → 10).** Every lake node at its centre cell on both rungs: LOST
///    (land after), MIXED (the sea's level after), SHIFTED (another lake's level after); the lake
///    patches that change or vanish whole. THE LIP: every cell of every dry node that shares a
///    level-1 block with a lake — a dry column the shore law holds against a LAKE's level (0 by
///    construction since the land rule never reads the level word). THE HOLD (G-RUNG-HOLD): every
///    dry node whose own row or a neighbour's stands under the sea's level, at its centre cell.
/// 2. **LEVEL 1 → 2 (rung 14 → 15) and LEVEL 2 → 3 (rung 15 → 16).** Per cell of the coarser rung
///    whose footprint holds a lake node: each of its four finer cells against it; and per cell
///    whose level node or a neighbour stands under the sea's level: the hold.
///
/// The shipped level-1 word is checked against `artifact::block_water` folded again over the rows
/// and the coast mask; a check that fails turns every gate RED, because a number the census cannot
/// trust is not a green.
///
/// **Example.** A tarn fills one node of the belt's pass. At rung 9 its column stands under its own
/// 2 100 m level; at rung 10 its level-1 block holds one lake node of four and the word is that
/// lake's own 2 100 m, so the tarn is the same water on both sides of the swap. Before plan step 2
/// the word was DRY and the tarn turned to rock.
fn rung_swap_lines(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    patches: &[Vec<u32>],
) {
    let levels = artifact.pyramid.len() as u32;
    let mut read = String::new();
    let mut swap_rungs: Vec<u8> = Vec::new();
    for rung in 6u8..=16 {
        let level = PyramidField::level_for(lattice, levels, rung);
        let side = coast_level(lattice.cells_per_node, rung);
        read.push_str(&format!(" {rung}:level{level}/side{side}"));
        if level == 1 && side == 0 {
            swap_rungs.push(rung);
        }
    }
    println!(
        "\nTHE LAKES AT THE RUNG SWAPS (ruling W16: a rung boundary is invisible; status §6 T5, §7 W6; plan step 2 part 5)"
    );
    println!(
        "  what each rung reads (the pyramid level of its field / the level of its side's footprint, 0 = the fine node's own word):{read}"
    );
    let (Some(&first), Some(&last)) = (swap_rungs.first(), swap_rungs.last()) else {
        println!(
            "  no rung reads its water from level 1 while its side is the fine node's own: nothing to count"
        );
        return;
    };
    let before = first.saturating_sub(1);
    if first == 0 || PyramidField::level_for(lattice, levels, before) != 0 {
        println!(
            "  rung {before} does not read the rows, so rung {first} is not the swap from the rows: nothing to count"
        );
        return;
    }
    let (Some(coarse), Some(fold)) = (lattice.coarser(1), artifact.pyramid_water.first()) else {
        println!("  the artifact holds no level 1: nothing to count");
        return;
    };
    let n = lattice.node_count();
    let m = coarse.node_count();
    let sea = body.plan_charter(0, vd_seed::bend::Face::PosX).sea_radius;
    let sea_m = artifact.sea_m;
    // The fold again, over the rows and the mask, beside the shipped word; and which level-1
    // blocks hold a lake node.
    let (mut lake_max, mut has_sea, mut has_lake) =
        (vec![DRY_M; m], vec![false; m], vec![false; m]);
    for node in 0..n as u32 {
        let (face, fi, fj) = lattice.split(node);
        let up = coarse.index(face, fi >> 1, fj >> 1) as usize;
        match coast_side(&artifact.coast, node) {
            Some(Side::Sea) => has_sea[up] = true,
            Some(Side::Lake) => {
                has_lake[up] = true;
                lake_max[up] = lake_max[up].max(artifact.rows[node as usize].water_m);
            }
            _ => {}
        }
    }
    let disagree = (0..m)
        .filter(|&u| block_water(lake_max[u], has_sea[u], sea_m) != fold[u])
        .count();
    drop(lake_max);
    drop(has_sea);
    // ★ THE CELL AT A NODE'S CENTRE on a rung whose cell is no wider than the node.
    let cpn = i64::from(lattice.cells_per_node);
    let centre = |f: i32, rung: u8| -> i32 { ((i64::from(f) * cpn + cpn / 2) >> rung) as i32 };
    let span = |f: i32, rung: u8| -> (i32, i32) {
        let lo = ((i64::from(f) * cpn) >> rung) as i32;
        let hi = ((i64::from(f + 1) * cpn) >> rung) as i32;
        (lo, hi.max(lo + 1))
    };
    let counts = artifact.coast_counts();
    let per_face = lattice.edge * lattice.edge;
    let mut class = vec![Change::Kept; n];
    let mut tally = SwapTally::default();
    let (mut lip_checked, mut lip_nodes, mut lip_cols) = (0usize, 0usize, 0usize);
    let (mut dry_before, mut wrong_reader) = (0usize, 0usize);
    let z_under = |node: u32| i32::from(artifact.rows[node as usize].z_m) < i32::from(sea_m);
    for f in 0..6u32 {
        let face_first = f * per_face;
        let (face, _, _) = lattice.split(face_first);
        let (Some(near), Some(far)) = (
            RungReader::new(body, artifact, &counts, face, before),
            RungReader::new(body, artifact, &counts, face, first),
        ) else {
            wrong_reader += 1;
            continue;
        };
        if near.fine != *lattice || far.fine != *lattice || near.level != 0 || far.level != 1 {
            wrong_reader += 1;
            continue;
        }
        for node in face_first..face_first + per_face {
            let i = node as usize;
            let (_, fi, fj) = lattice.split(node);
            let up = coarse.index(face, fi >> 1, fj >> 1) as usize;
            let (ci, cj) = (
                (centre(fi, before), centre(fj, before)),
                (centre(fi, first), centre(fj, first)),
            );
            match coast_side(&artifact.coast, node) {
                Some(Side::Lake) => {
                    let (Some(a), Some(b)) = (near.column(ci.0, ci.1), far.column(cj.0, cj.1))
                    else {
                        tally.unread += 1;
                        continue;
                    };
                    dry_before += usize::from(a.side_word != SIDE_SEA);
                    if let Some(c) = tally.lake(&a, &b, sea) {
                        class[i] = c;
                    }
                }
                // A pocket's dry floor is a dry node too (review finding 1 of plan step 2).
                Some(Side::Land | Side::Pocket) => {
                    // THE HOLD: a dry node whose ground or a neighbour's stands under the sea.
                    let low = z_under(node)
                        || lattice
                            .neighbours(node)
                            .into_iter()
                            .any(|nb| nb != NO_NODE && z_under(nb));
                    if low {
                        let (Some(a), Some(b)) = (near.column(ci.0, ci.1), far.column(cj.0, cj.1))
                        else {
                            tally.unread += 1;
                            continue;
                        };
                        tally.hold(&a, &b);
                    }
                    if !has_lake[up] {
                        continue;
                    }
                    // THE LIP: every cell of the dry node at the far rung.
                    lip_checked += 1;
                    let (i_lo, i_hi) = span(fi, first);
                    let (j_lo, j_hi) = span(fj, first);
                    let mut lipped = false;
                    for x in i_lo..i_hi {
                        for y in j_lo..j_hi {
                            let Some(b) = far.column(x, y) else {
                                tally.unread += 1;
                                continue;
                            };
                            tally.base_off += usize::from(!b.base_agrees);
                            if b.side_word != SIDE_SEA && b.hold != sea && b.base < b.hold {
                                lip_cols += 1;
                                lipped = true;
                            }
                        }
                    }
                    lip_nodes += usize::from(lipped);
                }
                _ => {}
            }
        }
    }
    let (mut changed, mut vanished) = (0usize, 0usize);
    for patch in patches {
        if patch.iter().any(|&v| class[v as usize] != Change::Kept) {
            changed += 1;
        }
        if patch.iter().all(|&v| class[v as usize] == Change::Lost) {
            vanished += 1;
        }
    }
    let cells = span(0, first).1;
    let trusted = disagree + tally.unread + tally.base_off + dry_before + wrong_reader == 0;
    println!(
        "  rungs {first} to {last} draw a lake column's water from level 1 while its side stays the fine node's own word; the first swap is rung {before} → {first}"
    );
    tally.lines(before, first, trusted);
    println!(
        "  of {} lake patches: {changed} change their drawn water at the swap {before} → {first}, and {vanished} of them vanish whole",
        patches.len()
    );
    println!(
        "  THE LIP (T5): {lip_checked} dry nodes share a level-1 block with a lake; every one of their {cells}×{cells} cells at rung {first} read: {lip_cols} columns on {lip_nodes} nodes are held against a LAKE's level their ground stands under (0 by construction: a land column never reads the level word)"
    );
    println!(
        "  checks, 0 expected on each: the shipped level-1 word disagrees with `block_water` over the rows at {disagree} of {m} level-1 nodes; {} columns unread; {} columns whose ground base does not give the kernel's own surface again; {dry_before} lake nodes NOT held under water at rung {before}; {wrong_reader} faces whose reader stands on another lattice or level",
        tally.unread, tally.base_off
    );
    census(
        "CHECK-RUNG-SWAP",
        &format!(
            "{disagree}fold+{}unread+{}base+{dry_before}dry+{wrong_reader}faces",
            tally.unread, tally.base_off
        ),
        "0[a-measurement-that-cannot-be-trusted-is-not-a-green]",
        Verdict::of(trusted),
    );
    census(
        "G-RUNG-LAKE",
        &format!(
            "{}nodes/{changed}patches/{vanished}vanished",
            tally.lost + tally.mixed + tally.shifted
        ),
        "0-lost-0-vanished[ruling-W16;SHIFTED-read-by-its-own-line]",
        Verdict::of(trusted && tally.lost == 0 && vanished == 0),
    );
    census(
        "G-RUNG-LIP",
        &format!("{lip_nodes}nodes/{lip_cols}columns"),
        "0[ruling-W16:a-rung-boundary-is-invisible]",
        Verdict::of(trusted && lip_nodes == 0),
    );
    // 2. THE LEVEL SWAPS.
    for (b_rung, a_rung) in [(14u8, 15u8), (15, 16)] {
        far_swap_lines(body, lattice, artifact, &counts, b_rung, a_rung);
    }
    // 3. THE POCKET'S FLOOR AT EACH RUNG.
    pocket_hold_lines(body, lattice, artifact, &counts);
}

/// ★ A POCKET'S FLOOR IS NEVER HELD OVER THE SEA (2026-09-25, review finding 1 of plan step 2;
/// ruling B7 item 1; ruling W16). At each rung the census reads, every column over a pocket's dry
/// floor ([`Side::Pocket`]): at a rung whose cell is no wider than a fine node, each pocket node's
/// centre cell; at a far rung, each cell whose footprint counts a pocket node. A column is HELD
/// where the shore law reads the LAND word while its ground base stands under the sea's level: the
/// law lifts that floor a quarter of its depth over the sea, a mesa.
///
/// The GATE counts the columns whose OWN side is the pocket — the node's own word at a near rung,
/// the footprint's majority at a far one — and 0 is its only green. Beside it, a READING: the far
/// cells whose footprint holds a pocket's floor as a MINORITY, which the footprint rule of W15
/// names LAND, and how many of those the law holds over the sea.
///
/// **Example.** A salt pan lies behind a one-node ridge on the belt's coast, 2 km under the sea's
/// level. Before the pocket word its floor read LAND with the sea in its stencil, and the census
/// counted it held at every rung — a mesa over the sea's level; with the word it reads INLAND and
/// the count is zero.
fn pocket_hold_lines(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    counts: &std::sync::Arc<vd_terrain::artifact::CoastCounts>,
) {
    println!(
        "\nTHE POCKET'S FLOOR AT EACH RUNG (review finding 1 of plan step 2; ruling B7 item 1; ruling W16): the columns over a pocket's dry floor the shore law holds over the sea"
    );
    let cpn = i64::from(lattice.cells_per_node);
    let centre = |f: i32, rung: u8| -> i32 { ((i64::from(f) * cpn + cpn / 2) >> rung) as i32 };
    let per_face = lattice.edge * lattice.edge;
    let lifted = |c: &ShippedColumn| (c.side_word == SIDE_LAND) & (c.base < c.hold);
    for rung in [9u8, 10, 13, 14, 15, 16] {
        let kf = coast_level(lattice.cells_per_node, rung);
        let edge_r = body.ladder().cells_per_edge(rung);
        let exact = kf == 0 || lattice.coarser(kf).is_some_and(|c| c.edge == edge_r);
        if !exact {
            println!(
                "  rung {rung}: its footprint (level {kf}) is not one level node a cell: not counted"
            );
            continue;
        }
        let (mut read, mut held, mut minority, mut minority_held, mut unread) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut deepest_m = 0.0f64;
        for f in 0..6u32 {
            let face_first = f * per_face;
            let (face, _, _) = lattice.split(face_first);
            let Some(reader) = RungReader::new(body, artifact, counts, face, rung) else {
                unread += 1;
                continue;
            };
            let mut take = |c: &ShippedColumn, own: bool| {
                let lift = lifted(c);
                if lift {
                    deepest_m = deepest_m.max(metres_of_q28(c.hold - c.base));
                }
                if own {
                    read += 1;
                    held += usize::from(lift);
                } else {
                    minority += 1;
                    minority_held += usize::from(lift);
                }
            };
            if kf == 0 {
                for node in face_first..face_first + per_face {
                    if coast_side(&artifact.coast, node) != Some(Side::Pocket) {
                        continue;
                    }
                    let (_, fi, fj) = lattice.split(node);
                    match reader.column(centre(fi, rung), centre(fj, rung)) {
                        Some(c) => take(&c, c.side == Some(Side::Pocket)),
                        None => unread += 1,
                    }
                }
                continue;
            }
            for cj in 0..edge_r {
                for ci in 0..edge_r {
                    if counts.pockets(kf, face, ci, cj).unwrap_or(0) == 0 {
                        continue;
                    }
                    match reader.column(ci as i32, cj as i32) {
                        Some(c) => take(&c, c.side == Some(Side::Pocket)),
                        None => unread += 1,
                    }
                }
            }
        }
        println!(
            "  rung {rung} (footprint level {kf}): {read} columns whose own side is a pocket's floor, {held} of them held over the sea; {minority} far cells hold a pocket's floor as a minority of their footprint and read another side, {minority_held} of those held over the sea; the deepest floor held stands {deepest_m:.0} m under the sea; {unread} unread"
        );
        census(
            &format!("G-POCKET-HELD@{rung}"),
            &format!("{held}of{read}columns"),
            "0[review-finding-1;ruling-B7-item-1:a-pocket's-floor-is-never-lifted-over-the-sea]",
            Verdict::of(unread == 0 && held == 0),
        );
        census(
            &format!("POCKET-MINORITY-HELD@{rung}"),
            &format!("{minority_held}of{minority}cells"),
            "no-band[a-reading:the-footprint-rule-of-W15]",
            Verdict::NoBand,
        );
    }
}

/// ★ ONE LEVEL SWAP, per cell of the coarser rung `after` (plan step 2 part 5): where the cell's
/// footprint holds a lake node, each of its four cells of `before` against it; where its level
/// node or a neighbour stands under the sea's level, the hold. The footprint of a cell of these
/// rungs IS one level node (the home planet's node is a power of two in cells), which the reader
/// checks before it counts.
fn far_swap_lines(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    counts: &std::sync::Arc<vd_terrain::artifact::CoastCounts>,
    before: u8,
    after: u8,
) {
    let levels = artifact.pyramid.len() as u32;
    let ka = PyramidField::level_for(lattice, levels, after);
    let kb = PyramidField::level_for(lattice, levels, before);
    let exact = ka == coast_level(lattice.cells_per_node, after)
        && kb == coast_level(lattice.cells_per_node, before)
        && ka == kb + 1
        && ka >= 1;
    let (Some(level_a), true) = (lattice.coarser(ka), exact) else {
        println!(
            "  the swap {before} → {after}: its footprints are not one level node each (level {kb} → {ka}): not counted"
        );
        return;
    };
    let sea = body.plan_charter(0, vd_seed::bend::Face::PosX).sea_radius;
    let sea_m = i32::from(artifact.sea_m);
    let z_level = &artifact.pyramid[ka as usize - 1];
    let edge_a = body.ladder().cells_per_edge(after) as i32;
    let mut tally = SwapTally::default();
    let mut wrong_reader = 0usize;
    for f in 0..6u32 {
        let (face, _, _) = lattice.split(f * lattice.edge * lattice.edge);
        let (Some(near), Some(far)) = (
            RungReader::new(body, artifact, counts, face, before),
            RungReader::new(body, artifact, counts, face, after),
        ) else {
            wrong_reader += 1;
            continue;
        };
        if near.level != kb || far.level != ka || level_a.edge as i32 != edge_a {
            wrong_reader += 1;
            continue;
        }
        for cj in 0..edge_a {
            for ci in 0..edge_a {
                let node = level_a.index(face, ci, cj);
                let lake = counts
                    .count(ka, face, ci as u32, cj as u32)
                    .is_some_and(|(_, l)| l > 0);
                let low = i32::from(z_level[node as usize]) < sea_m
                    || level_a
                        .neighbours(node)
                        .into_iter()
                        .any(|nb| nb != NO_NODE && i32::from(z_level[nb as usize]) < sea_m);
                if !lake && !low {
                    continue;
                }
                let Some(b) = far.column(ci, cj) else {
                    tally.unread += 1;
                    continue;
                };
                for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let Some(a) = near.column(2 * ci + di, 2 * cj + dj) else {
                        tally.unread += 1;
                        continue;
                    };
                    if lake {
                        let was_lake = tally.lake(&a, &b, sea).is_some();
                        let is_lake = b.side_word == SIDE_SEA && b.water != sea;
                        tally.gained += usize::from(!was_lake && is_lake);
                    }
                    if low {
                        tally.hold(&a, &b);
                    }
                }
            }
        }
    }
    let trusted = tally.unread + tally.base_off + wrong_reader == 0;
    tally.lines(before, after, trusted);
    census(
        &format!("CHECK-RUNG-SWAP@{before}-{after}"),
        &format!(
            "{}unread+{}base+{wrong_reader}faces",
            tally.unread, tally.base_off
        ),
        "0[a-measurement-that-cannot-be-trusted-is-not-a-green]",
        Verdict::of(trusted),
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
        "  the cascade: {} pours into a sibling across a spill; its walk visited {} rings over {} leaves (the skip over full rings, review finding 6); the books, km³/yr: runoff {:.4} = held and evaporated {:.4} + to the ocean {:.4} + stranded {:.4}",
        r.transfers,
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
    // ★ THE ROW AGAINST THE BUDGET'S OWN WATER (review finding 3 of plan step 2; plan question 3).
    // The shipped discharge row is the runoff accumulated over the ROUTING receivers (choice A,
    // built), and the routing fill carries every hollow's water over its rim; the budget holds a
    // hollow's water where the hollow is not full. Two readings, no band — the cure is the owner's
    // word (choice B, the budget's own water; or the receivers of §7 W11):
    // 1. the row that reaches the OCEAN from the land (at every land node whose shipped receiver is
    //    an ocean node) against the cascade's own water to the ocean;
    // 2. THE ROW OUT OF A CLOSED HOLLOW: for every hollow that spills into the sea (its parent is
    //    the ocean) and is NOT full, the row on every shipped receiver edge that leaves the
    //    hollow's own nodes. The budget holds all the water that reaches such a hollow, so a row
    //    that is the budget's water carries 0 there.
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
    println!(
        "  THE ROW AGAINST THE BUDGET (review finding 3; plan question 3): the discharge row delivers {:.4} km³/yr from the land to the ocean, the cascade {:.4} km³/yr — the row carries {:.4} km³/yr to the sea that the budget holds in its hollows ({:.4} km³/yr held); of the {closed} hollows that spill into the sea and are NOT full (the budget holds all their water), {leaking} have a row leaving them; the largest row out of one is {:.4} km³/yr, out of a {:.0} km² hollow whose own water is {} (a river through a chain of hollows leaves each of them, so the rows out of them sum to {:.1} km³/yr, more than the runoff)",
        km3_yr(row_to_ocean),
        km3_yr(r.to_ocean),
        km3_yr(row_to_ocean.saturating_sub(r.to_ocean)),
        km3_yr(r.evaporated_total),
        km3_yr(worst_row),
        deps[worst_d].area_m2 as f64 / 1.0e6,
        STATES[state_of(worst_d)],
        km3_yr(closed_row)
    );
    census(
        "ROW-OUT-OF-CLOSED-HOLLOWS",
        &format!(
            "{:.1}km3/yr-row-to-ocean/{:.1}km3/yr-budget/{leaking}-of-{closed}-closed-hollows-leak",
            km3_yr(row_to_ocean),
            km3_yr(r.to_ocean)
        ),
        "no-band[review-finding-3;plan-question-3-B-the-owner's-word-owed]",
        Verdict::NoBand,
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
