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
    for start in 0..n {
        if seen[start] || facies[start] & FACIES_LAKE == 0 || facies[start] & FACIES_SEA != 0 {
            continue;
        }
        let mut size = 0;
        let mut queue = VecDeque::from([start as u32]);
        seen[start] = true;
        while let Some(node) = queue.pop_front() {
            size += 1;
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
        sizes.push(size);
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
