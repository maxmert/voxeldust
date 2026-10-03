//! ★ THE BELTS' BENCH (rulings B37 and B38; the design `docs/investigation/2026-10-02/belts_and_time_design.md` §5):
//! the home planet solved by the ONE COPY of the macro solve (`vd_bins::solve_copy`) with the design's laws switched IN
//! THE COPY ONLY — nothing in the game changes. Each case prints Earth's set (the stands' relief rule and land readings,
//! one copy with the Earth instrument) and the mountains split BY CAUSE.
//!
//! **THE CASES** (`cargo run --release -p vd-bins --example belts_bench [-- today limits belts-today-limits belts]`):
//! - `today`: the solve's own laws. ★ THE CONTROL: its artifact must match the release gate's pinned digest.
//! - `belts-today-limits`: the belts by their influx (L1, L2, L4) under today's limits — what the limits cut (Q-B0).
//! - `limits`: the limits as laws (L0) alone: no clip, no uplift clamps, no envelope; the peak law per node.
//! - `belts`: the limits as laws AND the belts by their influx.
//! - `today-geometric`: the solve's own laws on the geometric time axis (L5) — the axis alone.
//! - `belts-geometric`: the limits as laws, the belts by their influx, and the geometric axis.
//! - `b40`: ruling B40 — the limits as laws, the belts by their influx at the plates' corrected mean speed (Earth's
//!   50 mm/yr), held by the force balance and widening into plateaus; `b40-geometric` on the geometric axis.
//! - `today-craters`, `b40-craters`: each crater stamped in the pass where its impact falls (ruling B37's landscape time).
//! - `joint`: today's laws with the erosion on F2, the joint law (Adams et al. 2020: `E = K_lp·Q·S^2.2`); `b40-joint`:
//!   ruling B40's laws, the craters' own time and the joint law together. Every case also prints F3, each node's inner
//!   relief from the 10 km law (Montgomery & Brandon 2002), and the relief rule on the node means it implies.
//! - any case `@N`: on N passes, the pass-counted cadences scaled to keep their spacing in years (M-TIME-1).
//!
//! **THE CAUSES**, exclusive, in this order: a CLOSED BASIN'S WALL (the relief window reaches the sea level or under
//! it); a CRATER (within one diameter of a crater's centre, by the summed chord); a BELT (the folded belt's rock: the
//! nodes a belt raised); the PLATFORM (all other land).
//!
//! **Example.** If the `belts` case reads two per cent of the land as mountains and most of them as BELT, the influx law
//! put the ranges where the plates close; if they read as CRATER, the law did not, and the craters' rims made them.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use vd_bins::solve_copy::{Laws, Router, Solved, artifact_of, solve_copy};
use vd_bins::stands::{LandReading, MOUNTAINS, ReliefReading, merged, read_land, relief_readings};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{FACIES_SEA, Schedule, Z_STEPS_PER_M};
use vd_terrain::strata::Province;

/// ★ F3, THE 10 km RELIEF LAW (Montgomery & Brandon 2002, EPSL 201, eq. 2 and Fig. 4): `E = E0 + K·Rz/[1 − (Rz/Rc)²]`,
/// `E` in mm/yr, `Rz` the relief in a 10 km circle in metres — calibration body: Earth's continents and the Olympics. A
/// node's inner relief is the root of the law at its recent erosion rate; its mean height is its channel height plus
/// Strahler's mature share of that relief (the middle of `vd_terrain::solve::HI_MATURE`).
const MB_E0_MM_YR: f64 = 0.01;
const MB_K_MM_YR_M: f64 = 2.5e-4;
const MB_RC_M: f64 = 1_500.0;

/// The inner relief of a node eroding at `e_mm_yr`, metres: the root of `K·Rz = (E − E0)·(1 − Rz²/Rc²)`.
fn inner_relief_m(e_mm_yr: f64) -> f64 {
    let d = e_mm_yr - MB_E0_MM_YR;
    if d <= 0.0 {
        return 0.0;
    }
    let a = d / (MB_RC_M * MB_RC_M);
    (-MB_K_MM_YR_M + (MB_K_MM_YR_M * MB_K_MM_YR_M + 4.0 * a * d).sqrt()) / (2.0 * a)
}

/// The geometric axis's last pass, years (the design's L5: half a million years; M-TIME-1 varies it).
const GEOMETRIC_LAST_YR: u64 = 500_000;

/// The release gate's pinned digest of the home planet's artifact (`crates/bins/tests/home_artifact_pin.rs`).
const HOME_ARTIFACT_DIGEST: [u64; 2] = [0x1eb5_3c78_2459_e0df, 0xc2cc_7661_57d2_a0e4];

/// Whole metres of a height in sixteenths, rounded half up: the artifact's own rounding.
fn metres(sixteenths: i32) -> i32 {
    (i64::from(sixteenths) + i64::from(Z_STEPS_PER_M / 2)).div_euclid(i64::from(Z_STEPS_PER_M))
        as i32
}

/// One compact line of a land reading.
fn line(label: &str, r: &LandReading) {
    println!(
        "  {label:<34} land {:>6.2} % | plains {:>5.2} tablelands {:>5.2} hills {:>4.2} mountains {:>5.2} % | \
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

/// The nodes within one diameter of a crater's centre, by the summed chord.
fn crater_reach(lattice: &MacroLattice, solved: &Solved) -> Vec<bool> {
    let n = lattice.node_count();
    let mut inside = vec![false; n];
    let mut stamp = vec![u32::MAX; n];
    let mut dist = vec![0u64; n];
    for (round, c) in solved.craters.iter().enumerate() {
        let round = round as u32;
        let reach = u64::from(c.diameter_m);
        let mut heap = BinaryHeap::new();
        stamp[c.node as usize] = round;
        dist[c.node as usize] = 0;
        heap.push(Reverse((0u64, c.node)));
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            inside[u as usize] = true;
            for m in lattice.neighbours(u) {
                if m == NO_NODE {
                    continue;
                }
                let step = d + u64::from(lattice.chord_m(u, m));
                if step > reach {
                    continue;
                }
                if stamp[m as usize] != round || step < dist[m as usize] {
                    stamp[m as usize] = round;
                    dist[m as usize] = step;
                    heap.push(Reverse((step, m)));
                }
            }
        }
    }
    inside
}

/// The readings of one solved case: Earth's set, the belts' heights, and the mountains by cause.
fn read_case(lattice: &MacroLattice, area: &[u64], solved: &Solved, last_dt_yr: u64) {
    let state = &solved.state;
    let n = lattice.node_count();
    let sea_m = metres(state.sea_z);
    let over: Vec<i32> = state.z.iter().map(|&s| metres(s) - sea_m).collect();
    let land: Vec<bool> = solved.facies.iter().map(|&f| f & FACIES_SEA == 0).collect();
    let readings = relief_readings(lattice, &over, &land);
    let dry = |r: &[Option<ReliefReading>], v: u32| r[v as usize].is_some_and(|x| x.lo_m > 0);
    let all = read_land(lattice, area, &over, &land, &readings, &|_| true, false);
    let away = read_land(
        lattice,
        area,
        &over,
        &land,
        &readings,
        &|v| dry(&readings, v),
        false,
    );
    line("all land", &all);
    line("away from closed basins", &away);
    // F3: each land node's inner relief from its last pass's erosion rate, and the node-mean field it implies.
    let hi = 0.5 * (vd_terrain::solve::HI_MATURE.0 + vd_terrain::solve::HI_MATURE.1);
    let rz: Vec<f64> = (0..n)
        .map(|v| {
            let e_mm_yr =
                f64::from(state.cut_now[v]) / f64::from(Z_STEPS_PER_M) / last_dt_yr.max(1) as f64
                    * 1_000.0;
            inner_relief_m(e_mm_yr)
        })
        .collect();
    let mean_over: Vec<i32> = over
        .iter()
        .zip(&rz)
        .map(|(&o, &r)| o + (hi * r).round() as i32)
        .collect();
    let mean_readings = relief_readings(lattice, &mean_over, &land);
    let mean_away = read_land(
        lattice,
        area,
        &mean_over,
        &land,
        &mean_readings,
        &|v| dry(&mean_readings, v),
        false,
    );
    let mut land_rz: Vec<(f64, u64)> = (0..n)
        .filter(|&v| land[v])
        .map(|v| (rz[v], area[v]))
        .collect();
    land_rz.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_a: u128 = land_rz.iter().map(|&(_, a)| u128::from(a)).sum();
    let at = |q: f64| -> f64 {
        let mut acc = 0u128;
        for &(r, a) in &land_rz {
            acc += u128::from(a);
            if acc as f64 >= q * total_a as f64 {
                return r;
            }
        }
        0.0
    };
    let over_share = |limit: f64| -> f64 {
        let a: u128 = land_rz
            .iter()
            .filter(|&&(r, _)| r > limit)
            .map(|&(_, a)| u128::from(a))
            .sum();
        a as f64 * 100.0 / total_a.max(1) as f64
    };
    println!(
        "  F3 THE INNER RELIEF (Montgomery & Brandon 2002, from each node's last-pass erosion over {:.1} Myr): Rz at \
         50 / 90 / 99 % of the land {:.0} / {:.0} / {:.0} m; land with Rz over 1 000 m {:.2} %, over 1 500 m {:.2} % \
         (Earth: under 5 % and under 2 %, Montgomery & Brandon 2002)",
        last_dt_yr as f64 / 1e6,
        at(0.5),
        at(0.9),
        at(0.99),
        over_share(1_000.0),
        over_share(1_499.0)
    );
    line("node means (+ Strahler share of Rz), away", &mean_away);
    // The causes, exclusive, in order.
    let craters = crater_reach(lattice, solved);
    let belt_code = Province::FoldedBelt.code();
    let cause = |v: usize| -> usize {
        match readings[v] {
            Some(r) if r.lo_m <= 0 => 0,
            _ if craters[v] => 1,
            _ if state.province[v] == belt_code => 2,
            _ => 3,
        }
    };
    let names = ["closed basin's wall", "crater", "belt", "platform"];
    let (mut land_a, mut mount_a) = ([0u128; 4], [0u128; 4]);
    let (mut all_land, mut all_mount) = (0u128, 0u128);
    let mut belt_over: Vec<(i32, u64)> = Vec::new();
    for v in 0..n {
        let Some(r) = readings[v] else { continue };
        let a = u128::from(area[v]);
        let c = cause(v);
        land_a[c] += a;
        all_land += a;
        if merged(r.class(false)) == MOUNTAINS {
            mount_a[c] += a;
            all_mount += a;
        }
        if state.province[v] == belt_code {
            belt_over.push((over[v], area[v]));
        }
    }
    let pct = |x: u128, of: u128| x as f64 * 100.0 / of.max(1) as f64;
    println!(
        "  THE MOUNTAINS BY CAUSE (each cause: its share of the land; mountains in its own land; its share of all mountains):"
    );
    for c in 0..4 {
        println!(
            "    {:<20} land {:>6.2} % | mountains {:>6.2} % of it | {:>6.2} % of all mountains",
            names[c],
            pct(land_a[c], all_land),
            pct(mount_a[c], land_a[c]),
            pct(mount_a[c], all_mount)
        );
    }
    // B40 item 1: each pair's plateau — the nodes within 100 m of their ceiling — and its half-width.
    if let Some(b) = &solved.belts
        && solved.carry_left_m3 >= 0.0
        && b.pairs.iter().any(|l| !l.is_empty())
    {
        let near = 100 * Z_STEPS_PER_M;
        let mut widths: Vec<f64> = Vec::new();
        let mut plateau_area = 0u128;
        for list in &b.pairs {
            let mut w = 0u32;
            for &(dist, j) in list {
                let j = j as usize;
                if b.ceiling[j] != i32::MAX && state.z[j] >= b.ceiling[j] - near {
                    w = w.max(dist);
                    plateau_area += u128::from(area[j]);
                }
            }
            widths.push(f64::from(w) / 1_000.0);
        }
        widths.sort_by(f64::total_cmp);
        println!(
            "  THE PLATEAUS (B40): {} convergent pairs; the half-width at 50 % / the largest: {:.0} / {:.0} km; the land \
             within 100 m of its ceiling {:.2} million km²; the rock that found no room {:.3e} m³",
            widths.len(),
            widths.get(widths.len() / 2).copied().unwrap_or(0.0),
            widths.last().copied().unwrap_or(0.0),
            plateau_area as f64 / 1e12,
            solved.carry_left_m3
        );
    }
    belt_over.sort_unstable();
    let belt_area: u64 = belt_over.iter().map(|&(_, a)| a).sum();
    let share_over = |h: i32| -> f64 {
        let a: u64 = belt_over
            .iter()
            .filter(|&&(o, _)| o > h)
            .map(|&(_, a)| a)
            .sum();
        a as f64 * 100.0 / belt_area.max(1) as f64
    };
    println!(
        "  THE BELTS' LAND (the folded belt's rock): {:.2} million km²; height over the sea at 50 / 90 / 100 %: {} / {} / {} m; \
         over 1 000 / 2 000 / 3 000 / 5 000 m: {:.2} / {:.2} / {:.2} / {:.2} %",
        belt_area as f64 / 1e12,
        belt_over.get(belt_over.len() / 2).map_or(0, |x| x.0),
        belt_over.get(belt_over.len() * 9 / 10).map_or(0, |x| x.0),
        belt_over.last().map_or(0, |x| x.0),
        share_over(1_000),
        share_over(2_000),
        share_over(3_000),
        share_over(5_000)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cases: Vec<String> = if args.is_empty() {
        ["today", "belts-today-limits", "limits", "belts"]
            .map(str::to_string)
            .to_vec()
    } else {
        args
    };
    println!(
        "THE BELTS' BENCH (rulings B37, B38), generator {}; the cases: {cases:?}",
        vd_terrain::GENERATOR_VERSION
    );
    let body = home_planet();
    let words = home_solve_words();
    let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
    let area: Vec<u64> = (0..lattice.node_count() as u32)
        .map(|v| lattice.area_m2(v))
        .collect();
    for case in &cases {
        // `name@N`: the case on N passes, every pass-counted cadence (the climate, the rebound) scaled by N/40 so its
        // spacing in YEARS stays the solve's own; the talus keeps its total count (the convergence test, M-TIME-1).
        let (name, passes) = match case.split_once('@') {
            Some((name, n)) => (name, n.parse::<u32>().expect("a pass count after @")),
            None => (case.as_str(), vd_terrain::solve::PASSES),
        };
        let standard = Schedule::standard(words.age_yr);
        let k = passes / standard.passes;
        assert_eq!(
            k * standard.passes,
            passes,
            "the pass count is a multiple of the solve's {}",
            standard.passes
        );
        let schedule = Schedule {
            passes,
            climate_every: standard.climate_every * k,
            isostasy_every: standard.isostasy_every * k,
            ..standard
        };
        let laws = match name {
            "today" => Laws::default(),
            "belts-today-limits" => Laws {
                belts_by_influx: true,
                ..Laws::default()
            },
            "limits" => Laws {
                limits_as_law: true,
                ..Laws::default()
            },
            "belts" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                ..Laws::default()
            },
            "today-geometric" => Laws {
                geometric_last_yr: Some(GEOMETRIC_LAST_YR),
                ..Laws::default()
            },
            "belts-geometric" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                geometric_last_yr: Some(GEOMETRIC_LAST_YR),
                ..Laws::default()
            },
            "b40" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                ..Laws::default()
            },
            "b40-geometric" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                geometric_last_yr: Some(GEOMETRIC_LAST_YR),
                ..Laws::default()
            },
            "today-craters" => Laws {
                craters_at_own_age: true,
                ..Laws::default()
            },
            "joint" => Laws {
                joint_law: true,
                ..Laws::default()
            },
            "b40-joint" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                craters_at_own_age: true,
                joint_law: true,
                ..Laws::default()
            },
            "joint-geometric" => Laws {
                joint_law: true,
                geometric_last_yr: Some(GEOMETRIC_LAST_YR),
                ..Laws::default()
            },
            "b40-joint-geometric" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                craters_at_own_age: true,
                joint_law: true,
                geometric_last_yr: Some(GEOMETRIC_LAST_YR),
            },
            "b40-craters" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                craters_at_own_age: true,
                ..Laws::default()
            },
            other => panic!(
                "a case is today, belts-today-limits, limits, belts, today-geometric, belts-geometric, b40, b40-geometric, today-craters, b40-craters, joint, b40-joint, joint-geometric or b40-joint-geometric, not {other}"
            ),
        };
        let started = Instant::now();
        let solved = solve_copy(&body, &words, schedule, Router::D8, &laws, None)
            .expect("the home planet solves");
        println!(
            "\n== CASE {case} ({laws:?}): solved in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        if case == "today" && passes == standard.passes {
            let digest = artifact_of(&body, &words, &solved).digest();
            println!(
                "CONTROL the copy with the solve's own laws: digest [{:#x}, {:#x}] {} the release gate's pin",
                digest[0],
                digest[1],
                if digest == HOME_ARTIFACT_DIGEST {
                    "MATCHES"
                } else {
                    "DIFFERS FROM"
                }
            );
        }
        if let Some(b) = &solved.belts {
            let r = &b.report;
            println!(
                "  THE BELTS' LAW: {} nodes ({} in collisions, {} at Andean margins); the plates' turnover {:.0} Myr \
                 (oceanic crust {:.1} million km², ridges {:.0} km, C0 {:.2} km²/yr); windows at 10/25/50/75/90 %: \
                 {:.1} / {:.1} / {:.1} / {:.1} / {:.1} Myr; convergence at 10/50/90 %: {:.1} / {:.1} / {:.1} mm/yr; \
                 surface uplift rate at 50 % {:.3} mm/yr, the largest {:.3} mm/yr; the uplift over the window: mean \
                 {:.0} m, the largest {:.0} m",
                r.nodes,
                r.collision_nodes,
                r.andean_nodes,
                r.t_m_yr / 1e6,
                r.ocean_area_m2 / 1e12,
                r.ridge_length_m / 1e3,
                r.c0_m2_yr / 1e6,
                r.windows_myr[0],
                r.windows_myr[1],
                r.windows_myr[2],
                r.windows_myr[3],
                r.windows_myr[4],
                r.vn_mm_yr[0],
                r.vn_mm_yr[1],
                r.vn_mm_yr[2],
                r.rate_mm_yr[0],
                r.rate_mm_yr[1],
                r.mean_total_m,
                r.max_total_m
            );
        }
        if solved.collapsed > 0 || laws.limits_as_law {
            println!(
                "  THE PEAK LAW: {} node collapses over the passes and their rounds; the largest excess left over its \
                 cap after a pass: {:.0} m",
                solved.collapsed,
                f64::from(solved.peak_left) / f64::from(Z_STEPS_PER_M)
            );
        }
        let times = vd_bins::solve_copy::pass_times(words.age_yr, passes, laws.geometric_last_yr);
        let last_dt_yr = times[times.len() - 1] - times[times.len() - 2];
        read_case(&lattice, &area, &solved, last_dt_yr);
    }
}
