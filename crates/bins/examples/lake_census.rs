//! ★ THE LAKE CENSUS (the owner, 2026-09-22, from 1 448 km over the belt: "It's full of patchy lakes,
//! you never would see anything like this in the real world"): the home planet is solved once and
//! its water facies are COUNTED — sea, lake and dry land per node; the lakes' share of the land
//! against Earth's; every lake as a connected patch of lake nodes with its size; and a map around one
//! direction, one character per node, so the picture the owner flew can be read as text. The octave
//! table is printed beside it, because a lake stands in a hollow and the hollows have a wavelength.
//!
//! `cargo run --release -p vd-bins --example lake_census -- [dx dy dz] [half_nodes]`

use std::collections::VecDeque;

use vd_core::glam::DVec3;
use vd_recipe::height::AMP_BITS;
use vd_recipe::noise::NOISE_BITS;
use vd_terrain::artifact::FACIES_SHIFT;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::{FACIES_LAKE, FACIES_SEA};

/// Earth's lakes: 117 million lakes over 5.0 million km², 3.7 % of the land outside the ice
/// (Verpoorter, Kutser, Seekell, Tranvik 2014, GRL 41). Most of that area is glacial: the Canadian
/// and Fennoscandian shields the ice sheets scoured. Outside the glaciated shields the share is
/// under one percent.
const EARTH_LAKE_SHARE_PCT: f64 = 3.7;

fn n_of(lattice: &MacroLattice) -> usize {
    lattice.node_count()
}

fn main() {
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
    // number and not an argument.
    {
        let started = std::time::Instant::now();
        let lakes = vd_terrain::lakes::budget(&state, &climate);
        println!(
            "the hierarchy and the budget cost {:.1} s of the solve, and found {} water bodies",
            started.elapsed().as_secs_f64(),
            lakes.report.bodies
        );
    }
    let artifact =
        vd_terrain::artifact::Artifact::of(&state, &facies_v, &climate, words.water_km3 > 0);
    drop(climate);
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
        const VALLEY_NODES: u64 = 100;
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
