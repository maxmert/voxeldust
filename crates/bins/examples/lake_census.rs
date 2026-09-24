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
//! 2. **THE INLAND SEA** (status §6 T7): the nodes under the sea's level that no path joins to the
//!    ocean, and the ten largest pockets with an aim each.
//! 3. **G-RIVER** (ruling B3): every channel head walked down the shipped receivers to its end, and
//!    every trunk step on which the floor rises. Beside it, THE DISCHARGE ROW AND ITS TREE (status §7
//!    W10): the row against the receivers it ships with.
//! 4. **THE LAKES AT THE RUNG SWAP** (ruling W16; status §6 T5, §7 W6): the lake nodes and patches
//!    whose drawn water changes where the far rungs start to read pyramid level 1, and the dry
//!    columns the shore law lifts over a lake there (the lip), read on the SHIPPED column.
//! 5. **THE WATERSHED AGAINST THE BUDGET** (status §7 W3): each hollow's whole watershed supply beside
//!    the supply the budget used, and the budget's own rule run again over the watershed. A WHAT-IF.
//!
//! WHAT IT READS: the solve's final state (the heights, the receivers, the routing fill, the
//! discharge row, the rain row), the lake budget's own hierarchy (`lakes::budget`, run again over
//! the same climate), the shipped artifact's rows and its level-1 water word, and THE SHIPPED
//! COLUMN on each side of the rung swap (`vd_bins::shore_geometry::RungReader`: the chunk builder's
//! `sample_side`, `column_water`, `sample_z` and the one kernel, on the body with its solved sea).
//! It never re-implements a column. Every gate prints one `CENSUS <gate> <value> <band> <verdict>`
//! line with no space inside a field; every self-check prints a `CENSUS CHECK-…` line of the same
//! shape; and the world's versions head the output, so a reading names its world.
//!
//! **Example.** A pilot over the belt sees a pond in a hollow from 150 km and loses it from 500 km.
//! The census names the pond's node: it is the only wet node of its level-1 block, so the fold turns
//! it to land at the rung 9 → 10 swap, and G-RUNG-LAKE counts it. The same pass walks the river off
//! the range beside it down the shipped receivers: the river ends in a dry closed basin, because the
//! budget gives that hollow only the runoff of the ground under its own brim. The what-if gives the
//! hollow the range's runoff too, and the hollow turns from DRY to SPILLING.
//!
//! `cargo run --release -p vd-bins --example lake_census -- [dx dy dz] [half_nodes]`

use std::collections::VecDeque;

use vd_bins::grid_bias::unit;
// ★ THE SHIPPED COLUMN, ONE COPY (plan step 1, ruling B6): the chunk builder's own reads in the
// chunk builder's order, shared with the teeth instrument, so the rung swap is measured on the
// path the client draws and never on a copy of it.
use vd_bins::shore_geometry::RungReader;
use vd_core::glam::DVec3;
use vd_recipe::height::{AMP_BITS, SIDE_LAND, SIDE_SEA};
use vd_recipe::noise::NOISE_BITS;
use vd_terrain::artifact::{
    Artifact, DRY_M, FACIES_SHIFT, PyramidField, Side, coast_level, coast_side, fold_water,
    log2_class,
};
use vd_terrain::climate::Climate;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::lakes::{Lakes, NO_DEP, OCEAN};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{DRY, FACIES_LAKE, FACIES_SEA, MacroSolve, Z_STEPS_PER_M, steeper};
use vd_terrain::{BodyDefinition, metres_of_q28};

/// Earth's lakes: 117 million lakes over 5.0 million km², 3.7 % of the land outside the ice
/// (Verpoorter, Kutser, Seekell, Tranvik 2014, GRL 41). Most of that area is glacial: the Canadian
/// and Fennoscandian shields the ice sheets scoured. Outside the glaciated shields the share is
/// under one percent.
const EARTH_LAKE_SHARE_PCT: f64 = 3.7;

/// ★ A RIVER'S DRAINAGE THRESHOLD, in nodes. G-GRID's valleys and G-RIVER's channels read this one
/// number, so the two gates count one river set: a land node is on a channel where its discharge is
/// at least this many times the rain on its own node. That is the drainage area a channel needs.
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
        let land_m2: f64 = (0..n_of(&lattice))
            .filter(|&i| state.z[i] > state.sea_z)
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
            "the standing water: {} nodes, {:.3e} km², {:.4e} km³ ({:.3} % of the inventory {:.4e} km³); the lakes' share of the land by AREA {:.2} %",
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
    let climate =
        vd_terrain::climate::climate(&body, &lattice, &words, &state.z, Some(state.sea_z));
    // ★ GATE G-BUZZSAW (ruling B2 step 2; Egholm et al. 2009): no peak on Earth stands more than
    // about 1 500 m over its LOCAL snowline, under every tectonic style. A free reading over two
    // fields we already hold — the solved ground and the climate's own equilibrium line.
    {
        let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
        let mut worst = f64::MIN;
        let mut over = 0usize;
        let mut land = 0usize;
        for i in 0..n_of(&lattice) {
            if state.z[i] <= state.sea_z || climate.ela_z[i] == i32::MAX {
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
            .filter(|&i| state.z[i] > state.sea_z && climate.ela_z[i] != i32::MAX)
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
        "the lakes' share of the LAND: {:.2} % — Earth {EARTH_LAKE_SHARE_PCT} % (glacial shields included), under 1 % elsewhere",
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
        for node in 0..n as u32 {
            let i = node as usize;
            if state.z[i] <= state.sea_z {
                continue;
            }
            let own = u64::from(state.rain[i]) * state.area[i];
            if state.discharge[i] < own.saturating_mul(VALLEY_NODES) {
                continue;
            }
            if let Some(axis) = trunk_axis(&lattice, &state.receiver, node, TRUNK_STEPS) {
                valley_four.add(axis.degrees);
                valley_twelve.add(axis.degrees);
            }
        }
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
    let ocean = inland_sea_lines(&state);
    let descent = terrain_descent(&state);
    river_lines(&state, &lakes, &descent, &ocean);
    discharge_tree_lines(&state, &descent);
    // The body WITH ITS SOLVED SEA, as every host that holds the artifact builds it: the shipped
    // column reads its sea radius, and the census's own `body` has none.
    let sea_body = home_planet().with_sea_m(artifact.sea());
    rung_swap_lines(&sea_body, &state.lattice, &artifact, &patches);
    watershed_lines(&state, &climate, &lakes, &descent);
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
}

/// ★ THE INLAND SEA (status §6 T7, §7): the facies byte marks EVERY node at or under the sea's level
/// as sea (`solve.rs` `facies`), and so does the budget's root (`lakes.rs` step 1), with no test of
/// a path to the ocean. This walks the lattice's own eight neighbours, across the cube seams, from
/// every such node. The largest connected body by area is the OCEAN; every other body is a POCKET
/// the client still draws as sea at the sea's own level. No band: whether such a pocket is a lake
/// by its budget or the sea is the owner's open question (status §11 Q3).
///
/// Returns, for each node, whether it belongs to the ocean.
///
/// **Example.** A crater floor 300 km inland stands 40 m under the sea's level. No path through the
/// lattice joins it to the ocean, yet the client paints it as a square of sea at the sea's own
/// level. The census counts it as a pocket and prints its aim, so a pilot can fly to it.
fn inland_sea_lines(state: &MacroSolve) -> Vec<bool> {
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
        };
        body[s] = id;
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            let vi = v as usize;
            pocket.nodes += 1;
            pocket.area_m2 += u128::from(state.area[vi]);
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
    let ocean = pockets
        .iter()
        .enumerate()
        .max_by_key(|(_, p)| p.area_m2)
        .map_or(UNSEEN, |(k, _)| k as u32);
    let is_ocean: Vec<bool> = body.iter().map(|&b| b != UNSEEN && b == ocean).collect();
    let mut inland: Vec<&Pocket> = pockets
        .iter()
        .enumerate()
        .filter(|(k, _)| *k as u32 != ocean)
        .map(|(_, p)| p)
        .collect();
    inland.sort_by_key(|p| std::cmp::Reverse(p.area_m2));
    let nodes: usize = inland.iter().map(|p| p.nodes).sum();
    let area_km2 = inland.iter().map(|p| p.area_m2).sum::<u128>() as f64 / 1.0e6;
    let ocean_km2 = pockets.get(ocean as usize).map_or(0, |p| p.area_m2) as f64 / 1.0e6;
    println!(
        "\nTHE INLAND SEA (status §6 T7): nodes at or under the sea's level ({:.0} m) that no path through the lattice's own neighbours (the cube seams included) joins to the OCEAN, the largest such body ({ocean_km2:.4e} km²)",
        f64::from(state.sea_z) / steps
    );
    println!(
        "  {} pockets, {nodes} nodes, {area_km2:.0} km²; the facies byte marks every one of them SEA, so the client draws each at the sea's own level. The largest ten:",
        inland.len()
    );
    for p in inland.iter().take(10) {
        let aim = unit(&state.lattice, p.deepest);
        println!(
            "    {:>7} nodes {:>10.0} km², deepest {:>6.0} m under the sea at node {} — aim {:.6} {:.6} {:.6}",
            p.nodes,
            p.area_m2 as f64 / 1.0e6,
            f64::from(state.sea_z - state.z[p.deepest as usize]) / steps,
            p.deepest,
            aim.x,
            aim.y,
            aim.z
        );
    }
    census(
        "INLAND-SEA",
        &format!("{area_km2:.0}km2/{nodes}nodes/{}pockets", inland.len()),
        "no-band-yet[owed:the-owner's-word,status-§11-Q3]",
        Verdict::NoBand,
    );
    is_ocean
}

/// ★ THE WATER'S OWN WAY ON THE GROUND: each node's steepest lower neighbour on the TERRAIN — never
/// the routing fill — and the node where that descent stops.
struct Descent {
    /// The steepest neighbour that stands lower by the budget's own order `(height, index)`, so a
    /// flat still descends by its index; [`NO_NODE`] at a pit and at or under the sea.
    down: Vec<u32>,
    /// Where the descent stops: a pit over the sea, which is the pit of one LEAF of the budget's
    /// hierarchy, or the first node at or under the sea.
    end: Vec<u32>,
}

/// ★ THE DESCENT OVER THE FINAL TERRAIN. "Lower" is the budget's own total order `(height, index)`
/// (`lakes.rs` step 2), so a descent always ends at a node with no lower neighbour, and that node is
/// exactly one of the budget's leaf pits; "steepest" is the solve's own comparison (`steeper`) over
/// the lattice's own chord. The ends are read lowest first, so a node takes its lower neighbour's
/// end in one pass.
///
/// **Example.** Rain on a spur of the belt's range runs down the ground, never down the fill. The
/// descent follows it into the hollow at the range's foot and stops at the hollow's pit, even where
/// the routing fill carries the same rain over the hollow's brim to the sea.
fn terrain_descent(state: &MacroSolve) -> Descent {
    let n = state.node_count();
    let lattice = &state.lattice;
    let z = &state.z;
    let mut down = vec![NO_NODE; n];
    for node in 0..n as u32 {
        let i = node as usize;
        if z[i] <= state.sea_z {
            continue;
        }
        let mut best = NO_NODE;
        let (mut dz_best, mut l_best) = (0i64, 0u32);
        for m in lattice.neighbours(node) {
            if m == NO_NODE || (z[m as usize], m) >= (z[i], node) {
                continue;
            }
            let dz = i64::from(z[i]) - i64::from(z[m as usize]);
            let l = lattice.chord_m(node, m);
            if best == NO_NODE || steeper(dz, l, m, dz_best, l_best, best) {
                best = m;
                dz_best = dz;
                l_best = l;
            }
        }
        down[i] = best;
    }
    let mut by_height: Vec<u32> = (0..n as u32).collect();
    by_height.sort_unstable_by_key(|&v| (z[v as usize], v));
    let mut end = vec![NO_NODE; n];
    for &v in &by_height {
        let d = down[v as usize];
        end[v as usize] = if d == NO_NODE { v } else { end[d as usize] };
    }
    Descent { down, end }
}

/// Where a river ends, or that it runs on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum End {
    /// Not an end: the water runs on to the receiver.
    Runs,
    /// The ocean.
    Ocean,
    /// A pocket at or under the sea's level that is not the ocean.
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

/// ★ THE END AT ONE NODE, or [`End::Runs`]. The sea ends a river. A hollow the routing filled ends
/// it where the lake budget holds no water in that hollow (a dry basin) or holds a lake under its
/// spill (a closed lake); where the budget's lake spills, the river runs on over the spill. The
/// hollow is the one the GROUND leads into: the leaf whose pit ends this node's descent.
fn end_at(state: &MacroSolve, lakes: &Lakes, descent: &Descent, ocean: &[bool], i: usize) -> End {
    if state.z[i] <= state.sea_z && ocean[i] {
        return End::Ocean;
    }
    if state.z[i] <= state.sea_z {
        return End::InlandSea;
    }
    if state.pit[i] == 0 {
        return End::Runs;
    }
    let pit = descent.end[i] as usize;
    if state.z[pit] <= state.sea_z {
        return End::Runs;
    }
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

/// ★ GATE G-RIVER (ruling B3: every river reaches the sea or a dry basin; a valley's floor descends
/// monotonically along its trunk). A RIVER is a land node whose SHIPPED discharge row stands at
/// least [`VALLEY_NODES`] times the rain on its own node — G-GRID's valley threshold, so both gates
/// read one river set. A CHANNEL HEAD is a river node no river node feeds. Each head's water is
/// walked down the SHIPPED receivers to its end ([`end_at`]); the ends are counted per head. Each
/// trunk step is read once: the FLOOR is the ground, or the water's surface where water stands (a
/// lake's flat surface is a floor that does not rise), and a step on which the floor rises is a
/// defect. The band is zero on both lines.
///
/// **Example.** A river off the belt's range enters a hollow the routing filled. The budget holds
/// no water there, so the river ends in a dry basin, and that end is lawful. If the shipped receiver
/// instead carries it up out of a filled hollow onto a ridge 40 m higher, the floor rises and
/// G-RIVER-DESCENT counts the step.
fn river_lines(state: &MacroSolve, lakes: &Lakes, descent: &Descent, ocean: &[bool]) {
    let n = state.node_count();
    let steps = f64::from(Z_STEPS_PER_M);
    let term: Vec<End> = (0..n)
        .map(|i| end_at(state, lakes, descent, ocean, i))
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
    // The rivers, and the heads: a head is a river node that no river node feeds.
    let river: Vec<bool> = (0..n)
        .map(|i| {
            let own = u64::from(state.rain[i]) * state.area[i];
            state.z[i] > state.sea_z && state.discharge[i] >= own.saturating_mul(VALLEY_NODES)
        })
        .collect();
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
    println!(
        "\nG-RIVER (ruling B3): a river is a land node whose SHIPPED discharge row is at least {VALLEY_NODES} times the rain on its own node (G-GRID's valley threshold, about {:.0} km² of drainage); {heads} channel heads, each walked down the SHIPPED receivers to its end",
        VALLEY_NODES as f64 * node_km2
    );
    println!(
        "  the ends, in heads: the ocean {to_ocean}, an inland sea pocket {to_pocket}, a closed lake {to_lake}, a dry closed basin {to_basin}; DEFECTS: no receiver on land {no_receiver}, a ring of receivers {cycle}"
    );
    println!(
        "  (a hollow the routing filled ends a river where the lake budget holds no water in it, a dry basin, or holds a lake under its spill, a closed lake; a spilling lake passes the river on)"
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
        "G-RIVER-END",
        &format!("{}heads-without-an-end", no_receiver + cycle),
        "0heads[ruling-B3-G-RIVER]",
        Verdict::of(no_receiver + cycle == 0),
    );
    census(
        "G-RIVER-DESCENT",
        &format!("{floor_rises}rising-steps"),
        "0steps[ruling-B3-G-RIVER]",
        Verdict::of(floor_rises == 0),
    );
}

/// ★ THE DISCHARGE ROW AGAINST THE TREE IT SHIPS WITH (status §7 W10, a latent defect). The solve
/// accumulates the discharge at its last pass's routing (`solve.rs` `accumulate` in the pass loop)
/// and routes again after it (the final route, and again where the lakes re-solve the sea), so the
/// shipped row and the shipped receivers can come from two different trees. R2 seeds its rivers
/// from that row. Three readings:
///
/// 1. The shipped receiver against the steepest lower neighbour on the final GROUND, split by why
///    they may differ: inside a hollow the routing filled (the fill routes over its flat, by
///    design), toward a filled hollow (the fill hides the ground's own way, by design), and
///    elsewhere (a stale receiver, or a chord that differs by its last metre).
/// 2. The discharge row accumulated again over the SHIPPED tree with the SAME rain row, against the
///    shipped row: the nodes that differ, and those whose SHIPPED byte (the log class) differs.
/// 3. The nodes where the shipped row FALLS downstream, which no tree can hold.
///
/// The band is zero: ruling B6 item 1, step 2, re-accumulates the discharge after the final route.
///
/// **Example.** A river on the coast reaches the sea by a new mouth after the final route, but its
/// discharge row still names the old tree. The old mouth's node carries the whole river's water in
/// the shipped byte, and the new mouth carries only its own rain.
fn discharge_tree_lines(state: &MacroSolve, descent: &Descent) {
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
    let mut again: Vec<u64> = state
        .rain
        .iter()
        .zip(&state.area)
        .map(|(&p, &a)| u64::from(p) * a)
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
        "\nTHE DISCHARGE ROW AND ITS TREE (status §7 W10: the row is the last pass's accumulation, the receivers come from the routes after it)"
    );
    println!(
        "  the shipped receiver against the steepest lower neighbour on the final ground: {compared} land nodes have a lower neighbour; the two differ at {in_fill} nodes inside a hollow the routing filled (by design), {into_fill} nodes whose ground leads into a filled hollow (by design), and {elsewhere} nodes elsewhere (a stale receiver, or a chord's last metre)"
    );
    println!(
        "  the discharge row accumulated again over the shipped tree with the same rain row: {differ} nodes differ (the worst by {worst:.1} times), {class_differ} of them in the SHIPPED byte (the row's log class); the shipped row FALLS downstream at {falls} nodes, which no tree can hold"
    );
    census(
        "W10-DISCHARGE-TREE",
        &format!("{class_differ}nodes"),
        "0nodes[ruling-B6-item-1-step-2:re-accumulate-after-the-final-route]",
        Verdict::of(class_differ == 0),
    );
}

/// ★ THE LAKES AT THE RUNG SWAP (ruling W16: a rung boundary is invisible; status §6 T5, §7 W6).
/// The rungs up to the swap read the ROWS, so a lake column takes its own row's level. From the
/// swap on, a rung reads pyramid level 1 while its SIDE is still the fine node's own mask word, so a
/// lake column takes the LEVEL-1 WATER WORD (`column_water`, the `Side::Lake` arm): the fold of its
/// 2 × 2 block (`fold_water`: DRY under two wet children of four, else the mean of EVERY wet child,
/// the sea and a lake alike), and LAND where the word is DRY.
///
/// THE READING IS THE SHIPPED COLUMN, NEVER A COPY OF IT: for every lake node the chunk builder's
/// own reads ([`RungReader::column`]: `sample_side` on the fine lattice, `column_water`, the slope
/// share, `sample_z`, the ONE kernel) run at the node's centre cell on the last rung before the swap
/// and on the first rung after it. Inside one fine node both rungs read one row and one level-1
/// word, so the centre cell speaks for every cell of the node. A lake node CHANGES where the side
/// word after the swap is LAND (it turns to land), or where its water radius moves (a mean with the
/// sea's level, or another lake level). A patch changes where one of its nodes changes.
///
/// THE LIP (T5): a DRY column reads the sea before the swap and the level-1 word after it. Where
/// that word is a lake's level and the column's own ground base stands UNDER it, the shore law
/// (`vd_recipe::height::shore`) holds the column at least a quarter of that depth OVER the lake: a
/// wall of land standing out of the water. Every cell of every dry node that shares a level-1 block
/// with a lake node is read (a sea-only block folds to the sea's own level, and no other block can
/// move a dry column's water).
///
/// The shipped level-1 word is also checked against `fold_water` over the rows, and every column's
/// ground base against the kernel's own surface; a check that fails turns both gates RED, because a
/// number the census cannot trust is not a green.
///
/// **Example.** A tarn fills one node of the belt's pass. At rung 9 its column stands under its own
/// 2 100 m level. At rung 10 its level-1 block holds one wet child of four, the word is DRY, and the
/// tarn turns to rock. The ridge node beside a wider lake reads the lake's level at rung 10, its
/// ground stands 30 m under it, and the shore law raises it 7.5 m over the water: the lip.
fn rung_swap_lines(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    patches: &[Vec<u32>],
) {
    const KEPT: u8 = 0;
    const LOST: u8 = 1;
    const MIXED: u8 = 2;
    const SHIFTED: u8 = 3;
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
        "\nTHE LAKES AT THE RUNG SWAP (ruling W16: a rung boundary is invisible; status §6 T5, §7 W6)"
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
    // The rung before the swap must read the rows themselves, or there is no swap to measure.
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
    // The fold again, over the rows, beside the shipped word; and which level-1 blocks hold a sea
    // node or a lake node by the SHIPPED side word (the coast mask).
    let (mut wet_sum, mut wet_count) = (vec![0i32; m], vec![0i32; m]);
    let (mut has_sea, mut has_lake) = (vec![false; m], vec![false; m]);
    for node in 0..n as u32 {
        let (face, fi, fj) = lattice.split(node);
        let up = coarse.index(face, fi >> 1, fj >> 1) as usize;
        let w = artifact.rows[node as usize].water_m;
        if w != DRY_M {
            wet_sum[up] += i32::from(w);
            wet_count[up] += 1;
        }
        match coast_side(&artifact.coast, node) {
            Some(Side::Sea) => has_sea[up] = true,
            Some(Side::Lake) => has_lake[up] = true,
            _ => {}
        }
    }
    let disagree = (0..m)
        .filter(|&u| fold_water(wet_sum[u], wet_count[u]) != fold[u])
        .count();
    drop(wet_sum);
    drop(wet_count);
    // ★ THE CELL AT A NODE'S CENTRE on a rung whose cell is no wider than the node: the rung-0
    // index of the node's centre, `f · cells_per_node + cells_per_node / 2`, shifted down by the
    // rung. Its column's nearest fine node is `f`, and its nearest level-1 node is `f >> 1`.
    let cpn = i64::from(lattice.cells_per_node);
    let centre = |f: i32, rung: u8| -> i32 { ((i64::from(f) * cpn + cpn / 2) >> rung) as i32 };
    // The cells of one node on a rung, `[lo, hi)` along one axis; at least the one cell.
    let span = |f: i32, rung: u8| -> (i32, i32) {
        let lo = ((i64::from(f) * cpn) >> rung) as i32;
        let hi = ((i64::from(f + 1) * cpn) >> rung) as i32;
        (lo, hi.max(lo + 1))
    };
    let counts = artifact.coast_counts();
    let per_face = lattice.edge * lattice.edge;
    let mut class = vec![KEPT; n];
    let mut tally = [0usize; 4];
    let (mut lake_nodes, mut lost_m2, mut worst_m) = (0usize, 0u128, 0.0f64);
    let (mut lip_checked, mut lip_nodes, mut lip_cols) = (0usize, 0usize, 0usize);
    let (mut lip_depth_m, mut lip_top_m) = (0.0f64, 0.0f64);
    // The checks, 0 expected on each.
    let (mut unread, mut base_off, mut dry_before, mut wrong_reader) =
        (0usize, 0usize, 0usize, 0usize);
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
            match coast_side(&artifact.coast, node) {
                Some(Side::Lake) => {
                    let (Some(a), Some(b)) = (
                        near.column(centre(fi, before), centre(fj, before)),
                        far.column(centre(fi, first), centre(fj, first)),
                    ) else {
                        unread += 1;
                        continue;
                    };
                    base_off += usize::from(!a.base_agrees) + usize::from(!b.base_agrees);
                    dry_before += usize::from(a.side_word != SIDE_SEA);
                    lake_nodes += 1;
                    let c = if b.side_word == SIDE_LAND {
                        LOST
                    } else if b.water == a.water {
                        KEPT
                    } else if has_sea[up] {
                        MIXED
                    } else {
                        SHIFTED
                    };
                    if c == LOST {
                        lost_m2 += u128::from(lattice.area_m2(node));
                    }
                    if c == MIXED || c == SHIFTED {
                        worst_m = worst_m.max(metres_of_q28(b.water - a.water).abs());
                    }
                    class[i] = c;
                    tally[c as usize] += 1;
                }
                Some(Side::Land) if has_lake[up] => {
                    let Some(a) = near.column(centre(fi, before), centre(fj, before)) else {
                        unread += 1;
                        continue;
                    };
                    base_off += usize::from(!a.base_agrees);
                    lip_checked += 1;
                    let (i_lo, i_hi) = span(fi, first);
                    let (j_lo, j_hi) = span(fj, first);
                    let mut lipped = false;
                    for ci in i_lo..i_hi {
                        for cj in j_lo..j_hi {
                            let Some(b) = far.column(ci, cj) else {
                                unread += 1;
                                continue;
                            };
                            base_off += usize::from(!b.base_agrees);
                            if b.side_word == SIDE_LAND && b.water != a.water && b.base < b.water {
                                lip_cols += 1;
                                lipped = true;
                                lip_depth_m = lip_depth_m.max(metres_of_q28(b.water - b.base));
                                lip_top_m = lip_top_m.max(metres_of_q28(b.surface - b.water));
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
        if patch.iter().any(|&v| class[v as usize] != KEPT) {
            changed += 1;
        }
        if patch.iter().all(|&v| class[v as usize] == LOST) {
            vanished += 1;
        }
    }
    let [kept, lost, mixed, shifted] = tally;
    let moved = lost + mixed + shifted;
    let cells = span(0, first).1;
    let trusted = disagree + unread + base_off + dry_before + wrong_reader == 0;
    println!(
        "  rungs {first} to {last} draw a lake column's water from level 1 while its side stays the fine node's own word; the swap is rung {before} → {first}"
    );
    println!(
        "  THE SHIPPED COLUMN (the chunk builder's reads through `RungReader`) at each lake node's centre cell, rung {before} against rung {first}: of {lake_nodes} lake nodes, {lost} turn to LAND ({:.0} km²); {mixed} take a level mixed with the sea's (their block holds a sea node); {shifted} take another lake level; {kept} keep their own; the largest change of level {worst_m:.1} m",
        lost_m2 as f64 / 1.0e6
    );
    println!(
        "  of {} lake patches: {changed} change their drawn water at the swap, and {vanished} of them vanish whole",
        patches.len()
    );
    println!(
        "  THE LIP (T5): {lip_checked} dry nodes share a level-1 block with a lake; every one of their {cells}×{cells} cells at rung {first} read: {lip_cols} columns on {lip_nodes} nodes stand on ground UNDER the lake level they read after the swap, and the shore law holds them OVER it — that ground stands up to {lip_depth_m:.1} m under the level, and the drawn top up to {lip_top_m:.1} m over it"
    );
    println!(
        "  checks, 0 expected on each: the shipped level-1 word disagrees with `fold_water` over the rows at {disagree} of {m} level-1 nodes; {unread} columns unread; {base_off} columns whose ground base does not give the kernel's own surface again; {dry_before} lake nodes NOT held under water at rung {before}; {wrong_reader} faces whose reader stands on another lattice or level"
    );
    if !trusted {
        println!("  ★ A CHECK FAILED: THE TWO GATES BELOW ARE NOT TRUSTED, SO THEY READ RED");
    }
    census(
        "CHECK-RUNG-SWAP",
        &format!(
            "{disagree}fold+{unread}unread+{base_off}base+{dry_before}dry+{wrong_reader}faces"
        ),
        "0[a-measurement-that-cannot-be-trusted-is-not-a-green]",
        Verdict::of(trusted),
    );
    census(
        "G-RUNG-LAKE",
        &format!("{moved}nodes/{changed}patches"),
        "0[ruling-W16:a-rung-boundary-is-invisible]",
        Verdict::of(trusted && moved == 0 && changed == 0),
    );
    census(
        "G-RUNG-LIP",
        &format!("{lip_nodes}nodes/{lip_cols}columns"),
        "0[ruling-W16:a-rung-boundary-is-invisible]",
        Verdict::of(trusted && lip_nodes == 0),
    );
}

/// The water that the lake budget's own rule settles over one stated supply per depression.
struct Fill {
    /// Each depression's verdict: its water reaches its spill.
    full: Vec<bool>,
    /// The water body that covers each depression, or [`NO_DEP`].
    body_of: Vec<u32>,
    /// Each body's level, sixteenths, or [`DRY`].
    level: Vec<i32>,
    /// The standing water at each node, sixteenths, or [`DRY`].
    water: Vec<i32>,
    /// The bodies that end DRY, PARTIAL and SPILLING.
    ends: [usize; 3],
}

/// The names of a hollow's three ends, in [`water_state`]'s order.
const WATER_STATES: [&str; 3] = ["DRY", "PARTIAL", "SPILLING"];

/// A depression's water under a fill, by the body that covers it: 0 DRY, 1 PARTIAL, 2 SPILLING.
fn water_state(fill: &Fill, d: usize) -> usize {
    let b = fill.body_of[d];
    if b == NO_DEP || fill.level[b as usize] == DRY {
        0
    } else if fill.full[b as usize] {
        2
    } else {
        1
    }
}

/// ★ THE BUDGET'S OWN RULE, REPLICATED OVER ANY SUPPLY (`vd_terrain::lakes::budget`, steps 3 to 6,
/// `crates/terrain/src/lakes.rs:249-319` and `:396-488`). A depression is FULL where it closed, its
/// supply is positive, and its supply reaches its deficit; each depression's water body follows by
/// the budget's descending pass; a full body stands at its spill; a partial body fills over its own
/// nodes, lowest first, while the supply pays the evaporation; then one level a node. The hierarchy
/// and every deficit are the budget's own, and only the supply is the caller's. The budget's own
/// helpers are private, so this is a replica, and the census runs it once over the budget's OWN
/// supply and counts every node where it disagrees with the budget's own water.
fn budget_rule(state: &MacroSolve, climate: &Climate, lakes: &Lakes, supply: &[u128]) -> Fill {
    let deps = &lakes.deps;
    let m = deps.len();
    let n = state.node_count();
    let mut full = vec![false; m];
    for d in 1..m {
        full[d] = deps[d].parent != NO_DEP && supply[d] > 0 && supply[d] >= deps[d].deficit;
    }
    let mut body_of = vec![NO_DEP; m];
    for d in (1..m).rev() {
        let parent = deps[d].parent;
        let kids = deps[d].children;
        let kids_full = kids[0] == NO_DEP || (full[kids[0] as usize] && full[kids[1] as usize]);
        let up = if parent == OCEAN || parent == NO_DEP {
            NO_DEP
        } else {
            body_of[parent as usize]
        };
        body_of[d] = if full[d] && up != NO_DEP {
            up
        } else if full[d] || kids_full {
            d as u32
        } else {
            NO_DEP
        };
    }
    let mut level = vec![DRY; m];
    let mut ends = [0usize; 3];
    let mut slot = vec![NO_DEP; m];
    let mut partial: Vec<u32> = Vec::new();
    for d in 1..m {
        if body_of[d] != d as u32 {
            continue;
        }
        if full[d] {
            level[d] = deps[d].spill_z;
            ends[2] += 1;
        } else {
            slot[d] = partial.len() as u32;
            partial.push(d as u32);
        }
    }
    // The nodes of every partial body, by a counting sort: a body's nodes are the nodes whose own
    // depression it covers.
    let body_at = |i: usize| -> u32 {
        let h = lakes.home[i];
        if h == NO_DEP || h == OCEAN {
            NO_DEP
        } else {
            body_of[h as usize]
        }
    };
    let slot_at = |i: usize| -> u32 {
        let b = body_at(i);
        if b == NO_DEP {
            NO_DEP
        } else {
            slot[b as usize]
        }
    };
    let mut counts = vec![0u32; partial.len() + 1];
    for i in 0..n {
        let s = slot_at(i);
        if s != NO_DEP {
            counts[s as usize + 1] += 1;
        }
    }
    for k in 1..counts.len() {
        counts[k] += counts[k - 1];
    }
    let mut cells = vec![0u32; counts[partial.len()] as usize];
    let mut at = counts.clone();
    for i in 0..n {
        let s = slot_at(i);
        if s != NO_DEP {
            cells[at[s as usize] as usize] = i as u32;
            at[s as usize] += 1;
        }
    }
    drop(at);
    for (k, &d) in partial.iter().enumerate() {
        let own = &mut cells[counts[k] as usize..counts[k + 1] as usize];
        own.sort_unstable_by_key(|&v| (state.z[v as usize], v));
        let lv = fill_level(state, climate, own, supply[d as usize]);
        level[d as usize] = lv;
        ends[usize::from(lv != DRY)] += 1;
    }
    let mut water = vec![DRY; n];
    for (i, w) in water.iter_mut().enumerate() {
        let b = body_at(i);
        if b == NO_DEP {
            continue;
        }
        let lv = level[b as usize];
        if lv != DRY && state.z[i] < lv {
            *w = lv;
        }
    }
    Fill {
        full,
        body_of,
        level,
        water,
        ends,
    }
}

/// The budget's closed-lake level (`lakes.rs` `fill_level`), replicated: over nodes sorted lowest
/// first, the highest step of equal heights whose standing water the supply still pays the
/// evaporation of, one sixteenth over that step; [`DRY`] where it cannot pay for the first step.
fn fill_level(state: &MacroSolve, climate: &Climate, own: &[u32], supply: u128) -> i32 {
    if supply == 0 {
        return DRY;
    }
    let mut level = DRY;
    let mut paid = 0u128;
    let mut k = 0usize;
    while k < own.len() {
        let step_z = state.z[own[k] as usize];
        let mut cost = 0u128;
        let mut end = k;
        while end < own.len() && state.z[own[end] as usize] == step_z {
            cost += deficit_rate(state, climate, own[end] as usize);
            end += 1;
        }
        if paid + cost > supply {
            break;
        }
        paid += cost;
        level = step_z.saturating_add(1);
        k = end;
    }
    level
}

/// A node's DEFICIT, mm·m²/yr (`lakes.rs` `rates`, its second word): what covering the node with
/// water costs — the runoff it stops yielding, plus the open surface's evaporation over its rain.
fn deficit_rate(state: &MacroSolve, climate: &Climate, i: usize) -> u128 {
    let p = i64::from(climate.rain_mm_yr[i]);
    let e = i64::from(climate.pet_mm_yr[i]);
    let q = i64::from(climate.runoff_mm_yr[i]);
    ((q + e - p).max(0) as u128) * u128::from(state.area[i])
}

/// ★ THE WATERSHED AGAINST THE BUDGET (status §7 W3) — A WHAT-IF THAT MOVES NOTHING. The budget
/// feeds each hollow only the runoff of the ground under its own spill (`lakes.rs:209-259`,
/// `:384-390`): every node over the brim books its water to the parent or to the ocean.
/// Fill-Spill-Merge, which ruling W11 cites, feeds a hollow from its whole WATERSHED. Here a leaf's
/// watershed is every node whose descent on the ground ([`Descent`]) ends at the leaf's pit, and a
/// parent's is its two children's. The census sets each leaf's watershed supply beside the budget's,
/// then runs the budget's own rule ([`budget_rule`]) over the watershed supply and counts the leaves
/// that move between DRY, PARTIAL and SPILLING, and the lake share that would follow. The overflow
/// of a full hollow whose parent holds no water is still lost (W4), and the sea is held where it
/// stands, so the what-if is a floor under the repaired budget, not the repair.
///
/// **Example.** A hollow at the foot of the belt's wet range gets only the runoff under its own
/// brim, so the budget leaves it a dry pan. Its watershed takes in the whole range face, the supply
/// rises many times, and under the same rule the hollow fills to its spill and runs out through a
/// river.
fn watershed_lines(state: &MacroSolve, climate: &Climate, lakes: &Lakes, descent: &Descent) {
    let n = state.node_count();
    let deps = &lakes.deps;
    let m = deps.len();
    let steps = f64::from(Z_STEPS_PER_M);
    // The budget run here against the solve's own water: they must be one budget.
    let stale = (0..n)
        .filter(|&i| lakes.water[i] != state.water_z[i])
        .count();
    // 1. Each leaf's watershed: the runoff of every node whose descent ends at the leaf's pit.
    let mut shed = vec![0u128; m];
    let mut foreign = 0usize;
    for node in 0..n as u32 {
        let i = node as usize;
        let pit = descent.end[i];
        let p = pit as usize;
        if state.z[p] <= state.sea_z {
            continue;
        }
        let leaf = lakes.home[p];
        if leaf == NO_DEP || leaf == OCEAN || deps[leaf as usize].pit != pit {
            foreign += 1;
            continue;
        }
        shed[leaf as usize] += u128::from(climate.runoff_mm_yr[i]) * u128::from(state.area[i]);
    }
    // 2. A parent's watershed is its two children's: a parent is made after its children, so one
    //    rising pass settles every chain.
    for d in 1..m {
        let kids = deps[d].children;
        if kids[0] != NO_DEP {
            shed[d] = shed[kids[0] as usize] + shed[kids[1] as usize];
        }
    }
    // 3. The budget's own rule twice: over its own supply (the replica's check), then over the
    //    watershed's (the what-if).
    let own: Vec<u128> = deps.iter().map(|d| d.supply).collect();
    let replica = budget_rule(state, climate, lakes, &own);
    let replica_off = (0..n)
        .filter(|&i| replica.water[i] != lakes.water[i])
        .count();
    let full_off = (1..m).filter(|&d| replica.full[d] != deps[d].full).count();
    let whatif = budget_rule(state, climate, lakes, &shed);
    // 4. The leaves: their supply against their watershed's, and their end before and after.
    let mut moves = [[0usize; 3]; 3];
    let (mut leaves, mut fed_above) = (0usize, 0usize);
    let mut ratio: Vec<f64> = Vec::new();
    let mut gains: Vec<(u128, usize)> = Vec::new();
    for d in 1..m {
        if deps[d].children[0] != NO_DEP {
            continue;
        }
        leaves += 1;
        moves[water_state(&replica, d)][water_state(&whatif, d)] += 1;
        let supply = deps[d].supply;
        if supply > 0 {
            ratio.push(shed[d] as f64 / supply as f64);
        } else if shed[d] > 0 {
            fed_above += 1;
        }
        gains.push((shed[d].saturating_sub(supply), d));
    }
    ratio.sort_by(f64::total_cmp);
    gains.sort_unstable_by_key(|&(g, d)| (std::cmp::Reverse(g), d));
    let pick = |q: usize| ratio.get(ratio.len() * q / 100).copied().unwrap_or(0.0);
    // 5. The lake share either way, over the land.
    let (mut land_m2, mut budget_m2, mut whatif_m2, mut whatif_nodes) =
        (0u128, 0u128, 0u128, 0usize);
    for node in 0..n as u32 {
        let i = node as usize;
        if state.z[i] <= state.sea_z {
            continue;
        }
        let a = u128::from(state.area[i]);
        land_m2 += a;
        if lakes.water[i] != DRY {
            budget_m2 += a;
        }
        if whatif.water[i] != DRY {
            whatif_m2 += a;
            whatif_nodes += 1;
        }
    }
    let budget_pct = budget_m2 as f64 * 100.0 / land_m2.max(1) as f64;
    let whatif_pct = whatif_m2 as f64 * 100.0 / land_m2.max(1) as f64;
    println!(
        "\nTHE WATERSHED AGAINST THE BUDGET (status §7 W3) — A WHAT-IF, IT MOVES NOTHING: the budget feeds a hollow only the runoff under its own spill; Fill-Spill-Merge (ruling W11's source) feeds it from its whole watershed, every node whose water runs down the GROUND to the hollow's pit"
    );
    println!(
        "  checks, 0 expected on each: the budget run here and the solve's own water disagree at {stale} nodes; {foreign} nodes descend to a pit that is not its leaf's own; the budget's rule, replicated over its own supply, disagrees with its water at {replica_off} nodes and with its verdicts at {full_off} depressions (the replica's bodies DRY {} PARTIAL {} SPILLING {}, the budget's {} {} {})",
        replica.ends[0],
        replica.ends[1],
        replica.ends[2],
        lakes.report.dry,
        lakes.report.partial,
        lakes.report.spilling
    );
    let trusted = stale + foreign + replica_off + full_off == 0;
    if !trusted {
        println!("  ★ A CHECK FAILED: THE WHAT-IF BELOW IS NOT TRUSTED");
    }
    census(
        "CHECK-WHATIF",
        &format!("{stale}stale+{foreign}foreign+{replica_off}water+{full_off}verdicts"),
        "0[the-replica-must-be-the-budget's-own-rule]",
        Verdict::of(trusted),
    );
    println!(
        "  {leaves} leaf hollows; the watershed's supply over the budget's: median {:.2} times, p90 {:.2} times, largest {:.1} times; {fed_above} hollows with no supply of their own take water only from above their brim",
        pick(50),
        pick(90),
        ratio.last().copied().unwrap_or(0.0)
    );
    let mut line = String::new();
    for (a, row) in moves.iter().enumerate() {
        for (b, &count) in row.iter().enumerate() {
            line.push_str(&format!(
                " {}→{} {count};",
                WATER_STATES[a], WATER_STATES[b]
            ));
        }
    }
    println!(
        "  the budget's own rule over the watershed's supply (a full hollow's overflow into a dry parent is still lost, W4, so this is a floor under the repair):"
    );
    println!("    the leaves, budget → what-if, in hollows:{line}");
    println!(
        "    the water bodies: DRY {} PARTIAL {} SPILLING {} (the budget: {} {} {})",
        whatif.ends[0],
        whatif.ends[1],
        whatif.ends[2],
        lakes.report.dry,
        lakes.report.partial,
        lakes.report.spilling
    );
    println!(
        "    the lakes would cover {whatif_nodes} nodes, {whatif_pct:.3} % of the land (the budget: {budget_pct:.3} %); G-LAKE-SHARE's ceiling is 4 % (ruling B3), W11 names 0.2 to 0.8 % for dry interiors; the sea stays where it stands"
    );
    println!("  the ten hollows the watershed feeds most beyond the budget:");
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
            "    {:>8.0} km², {spill}; the budget's supply {:.4} km³/yr, the watershed's {:.4} km³/yr; {} → {} — aim {:.6} {:.6} {:.6}",
            dep.area_m2 as f64 / 1.0e6,
            dep.supply as f64 / 1.0e12,
            shed[d] as f64 / 1.0e12,
            WATER_STATES[water_state(&replica, d)],
            WATER_STATES[water_state(&whatif, d)],
            aim.x,
            aim.y,
            aim.z
        );
    }
    census(
        "WHATIF-LAKE-SHARE",
        &format!("{whatif_pct:.3}%of-land"),
        "no-band-yet[a-what-if,moves-nothing;beside:G-LAKE-SHARE<=4%,W11:0.2-0.8%-dry-interiors]",
        Verdict::NoBand,
    );
    census(
        "WHATIF-HOLLOWS-FILLED",
        &format!(
            "{}leaves-DRY→PARTIAL/{}leaves-DRY→SPILLING/{}leaves-PARTIAL→SPILLING",
            moves[0][1], moves[0][2], moves[1][2]
        ),
        "no-band-yet[a-what-if,moves-nothing]",
        Verdict::NoBand,
    );
}
