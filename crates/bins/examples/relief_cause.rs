//! ★ WHY THE HOME PLANET'S MOUNTAINS ARE LOW — the cause bench of phase 0 (the owner's "Find the cause", 2026-10-02,
//! after the first Earth reading found, on land whose window stays over the sea level, one fifth of Earth's mountains and
//! about half of Earth's local relief: `docs/investigation/2026-10-02/phase0_earth_relief.txt`).
//!
//! It runs the SHIPPED functions only, on the home planet's own words, and keeps what they return. No switch exists in
//! the game for it (the owner, 2026-10-01: no test seams in the game). THE CONTROL: the artifact it rebuilds from the
//! shipped solve must match the release gate's pinned digest, so every reading below is the world's own.
//!
//! **THE CANDIDATES, each with its reading:**
//! 1. **THE FREEBOARD FACTOR** (`vd_terrain::land::freeboard_factor`): the continental thickness the law solved, and
//!    whether it stands at an end of `FREEBOARD_FACTOR_RANGE` (a clamp: the law asked for more than the range holds).
//! 2. **THE BELTS' ROOM** (`initial_land_from`: `room = relief − tallest`): the relief the seed drew, the tallest
//!    magnitude of the dry platform field about its mean (rebuilt here from the land's own public words — the crust
//!    byte, the plate's scatter, the solved factor — and checked against the land's own heights), its side (the
//!    highest platform or the deepest ocean floor), and the room left to the belts.
//! 3. **THE BELTS' SHAPE:** the uplift the belts ask for — its largest total, its share of the land, its steepest slope
//!    between land neighbours — and the relief rule on two fields with NO erosion: the platform alone, and the platform
//!    with the whole uplift. Can the plates and the belts alone make mountains at 8.2 km?
//! 4. **THE CLAMPS' HITS:** the land's uplift clamp (nodes at a bound `±relief − h`), and since phase 2 the peak law (the
//!    one limit left, a law: its collapses and what it left over its cap; the envelope that stood here went, B23).
//! 5. **THE EROSION AGAINST THE UPLIFT:** on the belts (land whose uplift over the age is 1 000 m or more), the mean
//!    uplift, the mean removal and the mean height left over the sea; the relief rule on the final field, the belts apart
//!    from the rest; the rebounds.
//! 6. (RETIRED with phase 2: the erosion's strength scan read the old stream-power law's `K₀`; the joint law's constant
//!    is its calibration's, never scanned — ruling B38.)
//!
//! `cargo run --release -p vd-bins --example relief_cause`
//!
//! **Example.** A folded belt rises over the whole age on a plate boundary of the home planet. If the reading of item 5
//! finds that the rivers removed nearly all of its uplift, the belt is a graded plateau and the cure is in the balance
//! of the uplift and the erosion; if item 2 finds that the belt had little room to begin with, the cure is in the room.

use vd_bins::stands::{LandReading, ReliefReading, read_land, relief_readings};
use vd_terrain::Gf;
use vd_terrain::artifact::Artifact;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::land::{
    FREEBOARD_FACTOR_RANGE, dry_field, initial_land, isostatic_parts_m, ocean_mask, relief_cap_m,
};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{FACIES_SEA, Schedule, Z_STEPS_PER_M, solve_full};

/// The release gate's pinned digest of the home planet's artifact (`crates/bins/tests/home_artifact_pin.rs`).
const HOME_ARTIFACT_DIGEST: [u64; 2] = [0x6ac9_f81d_0b21_0db3, 0xd3b9_c08d_8919_5424];
/// A belt node: land whose uplift over the age is this much or more, metres (MEASURED on the first run: no land node
/// rises 1 000 m, and 3.69 % of the land rises 100 m or more).
const BELT_M: i32 = 100;
/// The rebound's block edge: the rebound lifts every node of a level-4 block by one value (16 × 16 nodes, 131 km).
const REBOUND_BLOCK: i32 = 16;

/// Whole metres of a height in sixteenths, rounded half up: the artifact's own rounding (`metres_i16`).
fn metres(sixteenths: i32) -> i32 {
    (i64::from(sixteenths) + i64::from(Z_STEPS_PER_M / 2)).div_euclid(i64::from(Z_STEPS_PER_M))
        as i32
}

/// ★ THE REBOUND'S BLOCK STEPS: the mean height step between land neighbours on one face, by where the step stands in a
/// block of [`REBOUND_BLOCK`] nodes — position 0 is the block's edge, positions 1 to 15 are inside it. A field with no
/// block steps reads the same at every position; the other fifteen positions are the control.
fn block_steps(lattice: &MacroLattice, z: &[i32], land: &[bool]) -> [f64; REBOUND_BLOCK as usize] {
    let edge = lattice.edge as i32;
    let mut sum = [0.0f64; REBOUND_BLOCK as usize];
    let mut count = [0u64; REBOUND_BLOCK as usize];
    for v in 0..lattice.node_count() as u32 {
        if !land[v as usize] {
            continue;
        }
        let (face, i, j) = lattice.split(v);
        for (ni, nj, at) in [(i + 1, j, i + 1), (i, j + 1, j + 1)] {
            if ni >= edge || nj >= edge {
                continue;
            }
            let m = lattice.index(face, ni, nj);
            if !land[m as usize] {
                continue;
            }
            let k = at.rem_euclid(REBOUND_BLOCK) as usize;
            sum[k] += f64::from((z[v as usize] - z[m as usize]).abs()) / f64::from(Z_STEPS_PER_M);
            count[k] += 1;
        }
    }
    let mut mean = [0.0f64; REBOUND_BLOCK as usize];
    for k in 0..REBOUND_BLOCK as usize {
        mean[k] = sum[k] / count[k].max(1) as f64;
    }
    mean
}

/// One line of the block steps: the edge's mean step against the inside's median.
fn block_line(label: &str, mean: &[f64; REBOUND_BLOCK as usize]) {
    let mut inside: Vec<f64> = mean[1..].to_vec();
    inside.sort_by(f64::total_cmp);
    let median = inside[inside.len() / 2];
    println!(
        "  {label:<46} the step at a block's edge {:.1} m; inside a block {:.1}..{:.1} m (the median {median:.1} m); \
         the edge over the median {:.2}",
        mean[0],
        inside[0],
        inside[inside.len() - 1],
        mean[0] / median.max(1e-9)
    );
}

/// One compact line of a land reading.
fn line(label: &str, r: &LandReading) {
    println!(
        "  {label:<46} land {:>6.2} % | plains {:>5.2} tablelands {:>5.2} hills {:>4.2} mountains {:>5.2} % | \
         relief p50 {:>4} p90 {:>4} p99 {:>5} m | land over 1 000 m {:>5.2} %",
        r.land_pct,
        r.class_pct[0],
        r.class_pct[1],
        r.class_pct[2],
        r.class_pct[3],
        r.span_m[2],
        r.span_m[4],
        r.span_m[5],
        r.high_pct
    );
}

/// The relief rule on one field in sixteenths over a sea level in sixteenths, the land by `land`: all the land, and the
/// land whose window stays over the sea level.
fn read_field(
    lattice: &MacroLattice,
    area: &[u64],
    z: &[i32],
    sea: i32,
    land: &[bool],
) -> (LandReading, LandReading) {
    let sea_m = metres(sea);
    let over: Vec<i32> = z.iter().map(|&s| metres(s) - sea_m).collect();
    let readings = relief_readings(lattice, &over, land);
    let all = read_land(lattice, area, &over, land, &readings, &|_| true, false);
    let dry = |readings: &[Option<ReliefReading>], v: u32| {
        readings[v as usize].is_some_and(|r| r.lo_m > 0)
    };
    let away = read_land(
        lattice,
        area,
        &over,
        land,
        &readings,
        &|v| dry(&readings, v),
        false,
    );
    (all, away)
}

fn main() {
    println!(
        "WHY THE HOME PLANET'S MOUNTAINS ARE LOW (phase 0, the cause bench), generator {}",
        vd_terrain::GENERATOR_VERSION
    );
    let body = home_planet();
    let words = home_solve_words();
    let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
    let n = lattice.node_count();
    let area: Vec<u64> = (0..n as u32).map(|v| lattice.area_m2(v)).collect();
    let steps = f64::from(Z_STEPS_PER_M);
    let started = std::time::Instant::now();
    let land = initial_land(&body, &lattice, &words.land());
    println!(
        "the initial land in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let relief_m = body.relief_m();
    let cap_m = relief_cap_m(&body);
    let sea = land
        .sea_z
        .expect("the home planet has a sea at the initial land");

    // 1. The freeboard factor.
    let factor = land.freeboard_factor.to_f64();
    let (lo, hi) = FREEBOARD_FACTOR_RANGE;
    println!(
        "\n1. THE FREEBOARD FACTOR: {factor:.4} times Earth's continental thickness (the range {lo}..{hi}){}",
        if factor <= lo || factor >= hi {
            " — AT AN END OF THE RANGE: the clamp binds"
        } else {
            " — inside the range: the law's own answer"
        }
    );

    // 2. The belts' room, from the dry platform field rebuilt from the land's own words.
    let mut part_o = Vec::with_capacity(n);
    let mut part_c = Vec::with_capacity(n);
    for k in 0..n {
        let share = Gf::from_f64(f64::from(land.crust[k]) / 255.0);
        let scatter = land.plates[usize::from(land.plate[k])].crust_scatter;
        let (o, c) = isostatic_parts_m(share, cap_m, scatter);
        part_o.push(o);
        part_c.push(c);
    }
    let (dry, tallest) = dry_field(&part_o, &part_c, &area, land.freeboard_factor);
    let sixteen = Gf::from_i64(i64::from(Z_STEPS_PER_M));
    let (mut worst, mut checked) = (0i64, 0usize);
    for ((&h, &z), &ocean) in dry.iter().zip(&land.z).zip(&land.ocean) {
        if ocean {
            continue;
        }
        let rebuilt = (h * sixteen).to_i64_floor();
        worst = worst.max((rebuilt - i64::from(z)).abs());
        checked += 1;
    }
    let highest = dry.iter().map(|h| h.to_f64()).fold(f64::MIN, f64::max);
    let deepest = dry.iter().map(|h| h.to_f64()).fold(f64::MAX, f64::min);
    let room = (relief_m - tallest.to_f64()).max(0.0);
    println!(
        "CONTROL the rebuilt platform field against the land's own heights: the worst difference {:.1} m over {checked} \
         land nodes (the crust share is rebuilt from its byte)",
        worst as f64 / steps
    );
    println!(
        "2. THE BELTS' ROOM: the relief drawn {relief_m:.0} m (the law's cap {cap_m:.0} m); the dry platform field about its \
         mean reaches {highest:.0} m and {deepest:.0} m, so the tallest magnitude is {:.0} m ({}); THE ROOM LEFT TO THE \
         BELTS {room:.0} m",
        tallest.to_f64(),
        if highest >= -deepest {
            "the highest platform sets it"
        } else {
            "the deepest ocean floor sets it"
        }
    );
    println!(
        "   the sea at the initial land stands {:.0} m over the field's mean, so a platform at the mean stands {:.0} m \
         under it",
        f64::from(sea) / steps,
        f64::from(sea) / steps
    );

    // 3. The belts' shape.
    let land_side: Vec<bool> = land.ocean.iter().map(|&o| !o).collect();
    let (mut max_up, mut land_a, mut up100, mut up1000) = (i32::MIN, 0u128, 0u128, 0u128);
    let mut steepest = 0.0f64;
    for k in 0..n {
        if !land_side[k] {
            continue;
        }
        let a = u128::from(area[k]);
        land_a += a;
        max_up = max_up.max(land.uplift[k]);
        if land.uplift[k] >= 100 * Z_STEPS_PER_M {
            up100 += a;
        }
        if land.uplift[k] >= BELT_M * Z_STEPS_PER_M {
            up1000 += a;
        }
        for m in lattice.neighbours(k as u32) {
            if m == NO_NODE || !land_side[m as usize] {
                continue;
            }
            let rise = f64::from((land.uplift[k] - land.uplift[m as usize]).abs()) / steps;
            steepest = steepest.max(rise / f64::from(lattice.chord_m(k as u32, m).max(1)));
        }
    }
    let pct = |x: u128| x as f64 * 100.0 / land_a.max(1) as f64;
    println!(
        "3. THE BELTS' SHAPE: the largest uplift over the age on land {:.0} m ({:.0} % of the room); land with an uplift \
         of 100 m or more {:.2} %, of {BELT_M} m or more {:.2} %; the uplift's steepest slope between land neighbours \
         {:.2} % (the relief rule's gentle limit is 8 %)",
        f64::from(max_up) / steps,
        if room > 0.0 {
            f64::from(max_up) / steps * 100.0 / room
        } else {
            0.0
        },
        pct(up100),
        pct(up1000),
        steepest * 100.0
    );
    let platform_mask = ocean_mask(&lattice, &land.z, &area, sea);
    let platform_land: Vec<bool> = platform_mask.ocean.iter().map(|&o| !o).collect();
    let (platform, platform_away) = read_field(&lattice, &area, &land.z, sea, &platform_land);
    let no_erosion: Vec<i32> = land
        .z
        .iter()
        .zip(&land.uplift)
        .map(|(&z, &u)| z + u)
        .collect();
    let ne_mask = ocean_mask(&lattice, &no_erosion, &area, sea);
    let ne_land: Vec<bool> = ne_mask.ocean.iter().map(|&o| !o).collect();
    let (ne, ne_away) = read_field(&lattice, &area, &no_erosion, sea, &ne_land);
    println!("   the relief rule on fields with NO erosion (the initial sea level):");
    let ne_steps = block_steps(&lattice, &no_erosion, &ne_land);
    line("the platform alone, all land", &platform);
    line(
        "the platform alone, away from closed basins",
        &platform_away,
    );
    line("the platform + the whole uplift, all land", &ne);
    line(
        "the platform + the whole uplift, away from basins",
        &ne_away,
    );

    // 4a. The land's uplift clamp.
    let relief_steps = (Gf::from_f64(relief_m) * sixteen).to_i64_floor() as i32;
    let at_top = (0..n)
        .filter(|&k| land.uplift[k] != 0 && land.uplift[k] == relief_steps - land.z[k])
        .count();
    let at_bottom = (0..n)
        .filter(|&k| land.uplift[k] != 0 && land.uplift[k] == -relief_steps - land.z[k])
        .count();
    println!(
        "4. THE CLAMPS: the land's uplift clamp holds {at_top} nodes at its top bound and {at_bottom} at its bottom bound \
         (of {n})"
    );

    // 5. The shipped solve.
    let started = std::time::Instant::now();
    let (state, facies, report) = solve_full(&body, &words, Schedule::standard(words.age_yr))
        .expect("the home planet solves");
    let climate =
        vd_terrain::climate::climate(&body, &lattice, &words, &state.z, Some(state.sea_view()));
    let artifact = Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
    let digest = artifact.digest();
    println!(
        "\nthe shipped solve in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    println!(
        "CONTROL the artifact rebuilt from the shipped solve: digest [{:#x}, {:#x}] {} the release gate's pin",
        digest[0],
        digest[1],
        if digest == HOME_ARTIFACT_DIGEST {
            "MATCHES"
        } else {
            "DIFFERS FROM"
        }
    );
    println!(
        "4. THE PEAK LAW over the passes (the one limit, a law): {} collapses; the largest excess left over its cap after \
         a pass {:.0} m; the cap {:.0} m over the body's datum",
        report.peak.0,
        f64::from(report.peak.1) / steps,
        vd_terrain::land::relief_cap_m(&body)
    );
    let land_final: Vec<bool> = facies.iter().map(|&f| f & FACIES_SEA == 0).collect();
    let (fin, fin_away) = read_field(&lattice, &area, &state.z, state.sea_z, &land_final);
    let sea_m = metres(state.sea_z);
    let over: Vec<i32> = state.z.iter().map(|&s| metres(s) - sea_m).collect();
    let readings = relief_readings(&lattice, &over, &land_final);
    let belt =
        |v: u32| land_final[v as usize] && state.uplift[v as usize] >= BELT_M * Z_STEPS_PER_M;
    let belts = read_land(&lattice, &area, &over, &land_final, &readings, &belt, false);
    let rest = read_land(
        &lattice,
        &area,
        &over,
        &land_final,
        &readings,
        &|v| land_final[v as usize] && !belt(v),
        false,
    );
    let (mut b_area, mut b_up, mut b_removed, mut b_over) = (0u128, 0i128, 0i128, 0i128);
    for v in 0..n as u32 {
        if !belt(v) {
            continue;
        }
        let a = i128::from(area[v as usize] as i64);
        b_area += area[v as usize] as u128;
        b_up += i128::from(state.uplift[v as usize]) * a;
        b_removed += i128::from(state.removed[v as usize]) * a;
        b_over += i128::from(over[v as usize]) * a;
    }
    let mean = |s: i128| s as f64 / b_area.max(1) as f64;
    println!(
        "5. THE EROSION AGAINST THE UPLIFT on the belts ({:.2} million km² of land with {BELT_M} m of uplift or more): the \
         mean uplift {:.0} m, the mean removal {:.0} m ({:.0} % of the uplift), the mean height left {:.0} m over the sea",
        b_area as f64 / 1e12,
        mean(b_up) / steps,
        mean(b_removed) / steps,
        mean(b_removed) * 100.0 / mean(b_up).max(1.0),
        mean(b_over)
    );
    println!("   the relief rule on the final field:");
    line("all land (the Earth instrument's home column)", &fin);
    line("away from closed basins", &fin_away);
    line("the belts (uplift 100 m or more)", &belts);
    line("the rest of the land", &rest);
    println!(
        "   the rebounds (level, nodes lifted, the largest lift in m): {:?}",
        report
            .rebounds
            .iter()
            .map(|&(l, nodes, lift)| (l, nodes, (f64::from(lift) / steps).round()))
            .collect::<Vec<_>>()
    );
    println!(
        "   THE REBOUND'S BLOCK STEPS (land neighbours on one face; a block is {REBOUND_BLOCK} nodes, 131 km):"
    );
    block_line("the field with NO erosion and no rebound", &ne_steps);
    block_line(
        "the final field (8 rebounds, the last one uneroded)",
        &block_steps(&lattice, &state.z, &land_final),
    );
    let cut: u64 = report.sweeps.iter().map(|s| s.total_cut).sum();
    let fill: u64 = report.sweeps.iter().map(|s| s.total_fill).sum();
    println!(
        "   the sweeps: {} passes, the cut summed {:.3e} m·nodes, the fill summed {:.3e} m·nodes; the craters {}",
        report.sweeps.len(),
        cut as f64 / steps,
        fill as f64 / steps,
        report.craters
    );
}
