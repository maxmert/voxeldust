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
//! - `b40-joint-kept`, `b40-joint-slow`, `b40-joint-kept-slow`: `b40-joint` with P2, the crust's partition at a
//!   collision (Replumaz et al. 2010), with S2, the collision's slowing (Clark 2012), and with both (the design §12);
//!   `b40-craters-kept-slow`: `b40-craters` with both (the solve's own erosion beside the joint law).
//! - any case `~mixed`: on the mixed axis (ruling B39's gate: passes from 0.5 Myr at the present, each 1.25 times
//!   the next, up to 25 Myr, then even 25 Myr passes back to the start; about 210 passes).
//! - any case `~fine`: the convergence gate's twice-as-fine axis (ruling B39 item 1): the mixed axis from 0.25 Myr up
//!   to 12.5 Myr, about twice the passes, the climate and the rebound every twice as many passes so their spacing in
//!   years stays the mixed axis's (as `@N` keeps the even axis's, M-TIME-1).
//! - any case `@N`: on N passes, the pass-counted cadences scaled to keep their spacing in years (M-TIME-1).
//! - any case `%K` (last): K sweeps in each pass on the pass's one routing, each a K-th of the pass's time — a
//!   resolution, not a law (ruling B41 item 1: does the sweep's own step or the routing every pass carry the gate's cause);
//!   `%Kr`: the same with the drains found again before each sub-sweep.
//! - any case `+ice`, `+icereb`, `+reb` (before the axis): ruling B42 item 2's cures — the ice stepped in time with no
//!   per-call cap; the same with its cut booked for the rebound; the rebound after every pass. `+base`: ruling B42 item
//!   1's mechanism cured — the sweep reads a receiver as a lake only where the routing found one. `+rebx`: the rebound
//!   without its floors (each node's fraction carried), so a finer rebound spacing is not read low by rounding. `+lt`:
//!   ruling B45 item 2's landscape's time — each plate's ground erodes only since its raise under the reworking law.
//!   `+lake`: ruling B46 item 2's lakes' cure — a lake's level follows its outlet inside the sweep. `+wedge`: ruling
//!   B49's critical wedge — each margin's cross-section a taper wedge sized by its rock, a plateau past the ceiling.
//!   `+level`: ruling B50's level plateau (one level a pair); `+wrain`: the erosion reads the joint law's own rain on the
//!   WATER'S OWN PATHS (the solve's own lakes' budget and water tree after every routing; a closed lake ends its rivers);
//!   `+water`: the same paths with the budget's runoff less the spilling lakes' evaporation (a reading: not the law's `Q`).
//! - `--ledger=DIR`: each case also writes its PASS LEDGER to `DIR/<case>.csv` — each process's change of volume on the
//!   land and the sea floor, and the land's readings, pass by pass (ruling B41 item 1: the gate's cause).
//!
//! **THE CAUSES**, exclusive, in this order: a CLOSED BASIN'S WALL (the relief window reaches the sea level or under
//! it); a CRATER (within one diameter of a crater's centre, by the summed chord); a BELT (the folded belt's rock: the
//! nodes a belt raised) — by a collision, at an Andean margin, or by the room; a WIDENED PLATEAU (a node the force
//! balance's widening raised, B40); the OLD PLAINS (all other land).
//!
//! **Example.** If the `belts` case reads two per cent of the land as mountains and most of them as BELT, the influx law
//! put the ranges where the plates close; if they read as CRATER, the law did not, and the craters' rims made them.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use vd_bins::solve_copy::{
    BELT_ANDEAN, BELT_COLLISION, Laws, Router, Solved, artifact_of, solve_copy,
};
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

/// The mixed axis (ruling B39's gate): the last pass half a million years, the longest 25 million years.
const MIXED_T_MIN_YR: u64 = 500_000;
const MIXED_T_MAX_YR: u64 = 25_000_000;
/// The convergence gate's finer axis: every pass length halved.
const FINE_T_MIN_YR: u64 = 250_000;
const FINE_T_MAX_YR: u64 = 12_500_000;

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
    // The craters the solve stamped (ruling B3's G-CRATER: about 190 known on Earth), and the land within their reach.
    {
        let mut d: Vec<u32> = solved.craters.iter().map(|c| c.diameter_m).collect();
        d.sort_unstable();
        let q = |p: f64| {
            d.get(((d.len().max(1) - 1) as f64 * p) as usize)
                .copied()
                .unwrap_or(0) as f64
                / 1_000.0
        };
        println!(
            "  THE CRATERS STAMPED: {} (G-CRATER: about 190 known on Earth); diameters at 50 / 90 / 100 %: {:.0} / {:.0} / {:.0} km",
            d.len(),
            q(0.5),
            q(0.9),
            q(1.0)
        );
    }
    // The causes, exclusive, in order.
    let craters = crater_reach(lattice, solved);
    let belt_code = Province::FoldedBelt.code();
    // A belt node's kind: a collision's, an Andean margin's, or (no influx law, or an arc the law leaves) the room's.
    let belt_kind = |v: usize| -> usize {
        match solved.belts.as_ref().map(|b| b.kind[v]) {
            Some(BELT_COLLISION) => 0,
            Some(BELT_ANDEAN) => 1,
            _ => 2,
        }
    };
    // The widened plateaus (B40's widening) read as their own cause, apart from the old plains (B45 item 2).
    let widened = |v: usize| solved.widened.get(v).copied().unwrap_or(false);
    let cause = |v: usize| -> usize {
        match readings[v] {
            Some(r) if r.lo_m <= 0 => 0,
            _ if craters[v] => 1,
            _ if state.province[v] == belt_code => 2 + belt_kind(v),
            _ if widened(v) => 5,
            _ => 6,
        }
    };
    let names = [
        "closed basin's wall",
        "crater",
        "collision belt",
        "Andean belt",
        "belt by the room",
        "widened plateau",
        "old plains",
    ];
    let (mut land_a, mut mount_a) = ([0u128; 7], [0u128; 7]);
    let (mut all_land, mut all_mount) = (0u128, 0u128);
    let mut belt_over: [Vec<(i32, u64)>; 4] = Default::default();
    // The widened plateau split by the margin that built it (a collision's, an Andean margin's, the room's).
    let (mut plat_land, mut plat_mount, mut plat_h) = ([0u128; 3], [0u128; 3], [0.0f64; 3]);
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
        if c == 5 {
            let k = belt_kind(v);
            plat_land[k] += a;
            plat_h[k] += f64::from(over[v]) * a as f64;
            if merged(r.class(false)) == MOUNTAINS {
                plat_mount[k] += a;
            }
        }
        if state.province[v] == belt_code {
            belt_over[0].push((over[v], area[v]));
            belt_over[1 + belt_kind(v)].push((over[v], area[v]));
        }
    }
    let pct = |x: u128, of: u128| x as f64 * 100.0 / of.max(1) as f64;
    println!(
        "  THE MOUNTAINS BY CAUSE (each cause: its share of the land; mountains in its own land; its share of all mountains):"
    );
    for c in 0..7 {
        println!(
            "    {:<20} land {:>6.2} % | mountains {:>6.2} % of it | {:>6.2} % of all mountains",
            names[c],
            pct(land_a[c], all_land),
            pct(mount_a[c], land_a[c]),
            pct(mount_a[c], all_mount)
        );
    }
    println!(
        "    the widened plateau by its margin — collision: land {:.2} %, mountains {:.2} % of it, mean {:.0} m | Andean: \
         land {:.2} %, mountains {:.2} % of it, mean {:.0} m | the room's: land {:.2} %",
        pct(plat_land[0], all_land),
        pct(plat_mount[0], plat_land[0]),
        plat_h[0] / plat_land[0].max(1) as f64,
        pct(plat_land[1], all_land),
        pct(plat_mount[1], plat_land[1]),
        plat_h[1] / plat_land[1].max(1) as f64,
        pct(plat_land[2], all_land)
    );
    // B45 item 2: the land's height by cause; the OLD PLAINS are the land no wall, crater, belt or widened plateau
    // claims (Earth's: a mean of 485 m, a median of 349 m, 13 % over 1 000 m — the belts' design §15, ETOPO 2022 on CRUST
    // 2.0's old types).
    let mut by_cause: [Vec<(i32, u64)>; 7] = Default::default();
    for v in 0..n {
        if readings[v].is_some() {
            by_cause[cause(v)].push((over[v], area[v]));
        }
    }
    println!(
        "  THE LAND'S HEIGHT BY CAUSE (over the sea: mean / median; the share over 1 000 m; Earth's old plains 485 / 349 m, 13 %):"
    );
    for (c, list) in by_cause.iter_mut().enumerate() {
        if list.is_empty() {
            continue;
        }
        list.sort_unstable();
        let a_all: u128 = list.iter().map(|&(_, a)| u128::from(a)).sum();
        let mean = list
            .iter()
            .map(|&(h, a)| f64::from(h) * a as f64)
            .sum::<f64>()
            / a_all as f64;
        let mut acc = 0u128;
        let mut median = 0;
        for &(h, a) in list.iter() {
            acc += u128::from(a);
            if acc * 2 >= a_all {
                median = h;
                break;
            }
        }
        let high: u128 = list
            .iter()
            .filter(|&&(h, _)| h > 1_000)
            .map(|&(_, a)| u128::from(a))
            .sum();
        println!(
            "    {:<20} {:>6.0} / {:>6} m | {:>6.2} % over 1 000 m",
            names[c],
            mean,
            median,
            pct(high, a_all)
        );
    }
    // B48: each cause's steepest slope to a land neighbour (the relief rule's own slope) at 50 / 90 % of its area, and
    // the belts' mountain share by height band — Earth's young belts read 3.2 / 9.4 % and, at 0–1 / 1–2 / 2–3 / 3–4 / over
    // 4 km, mountains 2.7 / 10.8 / 33.1 / 36.0 / 14.4 % (the belts' shape research, the design §17).
    let slope: Vec<f64> = (0..n as u32)
        .map(|v| {
            if !land[v as usize] {
                return 0.0;
            }
            lattice
                .neighbours(v)
                .iter()
                .filter(|&&m| m != NO_NODE && land[m as usize])
                .map(|&m| {
                    f64::from((over[v as usize] - over[m as usize]).abs())
                        / f64::from(lattice.chord_m(v, m).max(1))
                })
                .fold(0.0, f64::max)
        })
        .collect();
    println!(
        "  THE SLOPES BY CAUSE (the steepest slope to a land neighbour at 50 / 90 % of the cause's land; Earth's young belts 3.2 / 9.4 %):"
    );
    for (c, name) in names.iter().enumerate() {
        let mut list: Vec<(f64, u64)> = (0..n)
            .filter(|&v| readings[v].is_some() && cause(v) == c)
            .map(|v| (slope[v], area[v]))
            .collect();
        if list.is_empty() {
            continue;
        }
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: u128 = list.iter().map(|&(_, a)| u128::from(a)).sum();
        let at = |q: f64| -> f64 {
            let mut acc = 0u128;
            for &(sl, a) in &list {
                acc += u128::from(a);
                if acc as f64 >= q * total as f64 {
                    return sl;
                }
            }
            0.0
        };
        println!(
            "    {name:<20} {:>5.1} / {:>5.1} %",
            at(0.5) * 100.0,
            at(0.9) * 100.0
        );
    }
    let bands = [
        (0, 1_000),
        (1_000, 2_000),
        (2_000, 3_000),
        (3_000, 4_000),
        (4_000, i32::MAX),
    ];
    let (mut band_land, mut band_mount) = ([0u128; 5], [0u128; 5]);
    for v in 0..n {
        let Some(r) = readings[v] else { continue };
        if !matches!(cause(v), 2 | 3 | 5) {
            continue;
        }
        let Some(b) = bands
            .iter()
            .position(|&(lo, hi)| over[v] >= lo && over[v] < hi)
        else {
            continue;
        };
        band_land[b] += u128::from(area[v]);
        if merged(r.class(false)) == MOUNTAINS {
            band_mount[b] += u128::from(area[v]);
        }
    }
    println!(
        "  THE BELTS' MOUNTAINS BY HEIGHT (collision, Andean, widened plateau) at 0–1 / 1–2 / 2–3 / 3–4 / over 4 km: \
         {:.1} / {:.1} / {:.1} / {:.1} / {:.1} % of each band (Earth's young belts 2.7 / 10.8 / 33.1 / 36.0 / 14.4 %); \
         the bands' share of the belts' land {:.1} / {:.1} / {:.1} / {:.1} / {:.1} %",
        pct(band_mount[0], band_land[0]),
        pct(band_mount[1], band_land[1]),
        pct(band_mount[2], band_land[2]),
        pct(band_mount[3], band_land[3]),
        pct(band_mount[4], band_land[4]),
        pct(band_land[0], band_land.iter().sum()),
        pct(band_land[1], band_land.iter().sum()),
        pct(band_land[2], band_land.iter().sum()),
        pct(band_land[3], band_land.iter().sum()),
        pct(band_land[4], band_land.iter().sum())
    );
    // B49: where the belts' and the plateaus' water ends — straight to the sea, or through a standing lake (the budget's
    // water) on its way — and their rain; read in the solve's own order, outlets first.
    {
        let lake_bit = vd_terrain::solve::FACIES_LAKE;
        let mut through_lake = vec![false; n];
        let mut to_sea = vec![false; n];
        for k in (0..state.order.len()).rev() {
            let i = state.order[k] as usize;
            let r = state.receiver[i];
            let here = solved.facies[i] & lake_bit != 0;
            if r == NO_NODE {
                through_lake[i] = here;
                to_sea[i] = solved.facies[i] & FACIES_SEA != 0;
            } else {
                through_lake[i] = here || through_lake[r as usize];
                to_sea[i] = to_sea[r as usize];
            }
        }
        for (label, set) in [
            ("the belts (collision and Andean)", [2usize, 3]),
            ("the plateaus", [5usize, 5]),
            ("the old plains", [6usize, 6]),
            ("all the land", [usize::MAX, usize::MAX]),
        ] {
            let (mut a_all, mut a_lake, mut a_sea, mut rain) = (0u128, 0u128, 0u128, 0.0f64);
            for v in 0..n {
                if readings[v].is_none() || (set[0] != usize::MAX && !set.contains(&cause(v))) {
                    continue;
                }
                let a = u128::from(area[v]);
                a_all += a;
                if through_lake[v] {
                    a_lake += a;
                }
                if to_sea[v] {
                    a_sea += a;
                }
                rain += f64::from(state.rain[v]) * a as f64;
            }
            if a_all == 0 {
                continue;
            }
            println!(
                "  THE DRAINAGE of {label}: {:.1} % of the land drains to the sea, {:.1} % passes a standing lake on its way; \
                 the mean rain {:.0} mm/yr",
                pct(a_sea, a_all),
                pct(a_lake, a_all),
                rain / a_all as f64
            );
        }
    }
    // B45 item 2: each plate's raise time and its land's height (the landscape's time).
    if !solved.raise_yr.is_empty() {
        let plates = solved.raise_yr.len();
        let (mut a_land, mut h_sum, mut a_plat, mut hp_sum) = (
            vec![0u128; plates],
            vec![0.0f64; plates],
            vec![0u128; plates],
            vec![0.0f64; plates],
        );
        for v in 0..n {
            if readings[v].is_none() {
                continue;
            }
            let p = usize::from(solved.plate_of[v]);
            a_land[p] += u128::from(area[v]);
            h_sum[p] += f64::from(over[v]) * area[v] as f64;
            if cause(v) == 6 {
                a_plat[p] += u128::from(area[v]);
                hp_sum[p] += f64::from(over[v]) * area[v] as f64;
            }
        }
        let mut order: Vec<usize> = (0..plates).filter(|&p| a_land[p] > 0).collect();
        order.sort_by_key(|&p| solved.raise_yr[p]);
        println!(
            "  THE LANDSCAPE'S TIME (B45): each plate with land — its raise, Myr ago; its land, million km²; the land's mean height; the old plains' mean height:"
        );
        for p in order {
            println!(
                "    plate {:>2}: raised {:>5.0} Myr ago | land {:>6.2} million km² | mean {:>5.0} m | old plains {:>5.0} m on {:>5.2} million km²",
                p,
                solved.raise_yr[p] as f64 / 1e6,
                a_land[p] as f64 / 1e12,
                h_sum[p] / a_land[p].max(1) as f64,
                hp_sum[p] / a_plat[p].max(1) as f64,
                a_plat[p] as f64 / 1e12
            );
        }
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
    let kinds = [
        "THE BELTS' LAND (the folded belt's rock)",
        "  of it, the collisions'",
        "  of it, the Andean margins'",
        "  of it, the room's",
    ];
    for (label, list) in kinds.iter().zip(belt_over.iter_mut()) {
        if list.is_empty() {
            continue;
        }
        list.sort_unstable();
        let belt_area: u64 = list.iter().map(|&(_, a)| a).sum();
        let share_over = |h: i32| -> f64 {
            let a: u64 = list.iter().filter(|&&(o, _)| o > h).map(|&(_, a)| a).sum();
            a as f64 * 100.0 / belt_area.max(1) as f64
        };
        println!(
            "  {label}: {:.2} million km²; height over the sea at 50 / 90 / 100 %: {} / {} / {} m; \
             over 1 000 / 2 000 / 3 000 / 5 000 m: {:.2} / {:.2} / {:.2} / {:.2} %",
            belt_area as f64 / 1e12,
            list.get(list.len() / 2).map_or(0, |x| x.0),
            list.get(list.len() * 9 / 10).map_or(0, |x| x.0),
            list.last().map_or(0, |x| x.0),
            share_over(1_000),
            share_over(2_000),
            share_over(3_000),
            share_over(5_000)
        );
    }
}

/// The pass ledger as CSV, one line a pass (ruling B41 item 1, the gate's cause).
fn write_ledger(path: &str, ledger: &[vd_bins::solve_copy::PassLedger]) {
    use std::fmt::Write as _;
    let mut out = String::from(
        "t0_yr,t1_yr,craters_land,craters_sea,talus_land,talus_sea,ice_land,ice_sea,sweep_land,sweep_sea,\
         uplift_land,uplift_sea,cut_land,cut_sea,rebound_land,rebound_sea,peak_land,peak_sea,sea_z,land_mean_m,\
         land_over_1km,land_share,cut_lake,cut_held,cut_rerouted,cut_rest,nodes_lake,nodes_held,nodes_rerouted,\
         nodes_rest,ice_steps\n",
    );
    for e in ledger {
        let _ = write!(out, "{},{}", e.t0_yr, e.t1_yr);
        for pair in [
            e.craters_m3,
            e.talus_m3,
            e.ice_m3,
            e.sweep_m3,
            e.uplift_m3,
            e.cut_m3,
            e.rebound_m3,
            e.peak_m3,
        ] {
            let _ = write!(out, ",{:.6e},{:.6e}", pair[0], pair[1]);
        }
        let _ = write!(
            out,
            ",{},{:.3},{:.6},{:.6}",
            e.sea_z, e.land_mean_m, e.land_over_1km, e.land_share
        );
        for v in e.cut_by_source_m3 {
            let _ = write!(out, ",{v:.6e}");
        }
        for c in e.nodes_by_source {
            let _ = write!(out, ",{c}");
        }
        let _ = writeln!(out, ",{}", e.ice_steps);
    }
    std::fs::write(path, out).expect("the ledger writes");
    println!(
        "  THE PASS LEDGER: {} passes written to {path}",
        ledger.len()
    );
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--ledger=DIR`: each case writes its pass ledger to `DIR/<case>.csv` (ruling B41 item 1, the gate's cause).
    let ledger_dir: Option<String> = args
        .iter()
        .find_map(|a| a.strip_prefix("--ledger=").map(str::to_string));
    args.retain(|a| !a.starts_with("--ledger="));
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
        // `name~mixed`: the case on the mixed axis (ruling B39's gate): its own pass count, the solve's cadence counts.
        // `name~fine`: the convergence gate's twice-as-fine mixed axis.
        // `name%K`: K sweeps in each pass on the pass's one routing (a resolution; ruling B41 item 1, the gate's cause).
        // `name%Kr`: the same, the drains found again before each sub-sweep.
        let (case, sub_sweeps, sub_route) = match case.split_once('%') {
            Some((c, k)) => {
                let (k, routed) = match k.strip_suffix('r') {
                    Some(k) => (k, true),
                    None => (k, false),
                };
                (
                    c.to_string(),
                    k.parse::<u32>().expect("a sweep count after %"),
                    routed,
                )
            }
            None => (case.clone(), 1, false),
        };
        let case = &case;
        let (case_name, axis) = if let Some(name) = case.strip_suffix("~mixed") {
            (name, Some((MIXED_T_MIN_YR, MIXED_T_MAX_YR, 1)))
        } else if let Some(name) = case.strip_suffix("~fine") {
            (name, Some((FINE_T_MIN_YR, FINE_T_MAX_YR, 2)))
        } else {
            (case.as_str(), None)
        };
        let (name, passes) = match case_name.split_once('@') {
            Some((name, n)) => (name, n.parse::<u32>().expect("a pass count after @")),
            None => (case_name, vd_terrain::solve::PASSES),
        };
        // `name+ice`, `+icereb`, `+reb`: ruling B42 item 2's three cures (the ice stepped in time; the same with its cut
        // booked for the rebound; the rebound after every pass).
        let mut cures = name.split('+');
        let name = cures.next().expect("a case name");
        let (mut ice_stepped, mut ice_rebound, mut rebound_every_pass, mut base_cure) =
            (false, false, false, false);
        let mut rebound_exact = false;
        let mut landscape_time = false;
        let mut lake_cure = false;
        let mut wedge = false;
        let mut level_plateau = false;
        let mut water = vd_bins::solve_copy::ErosionWater::Routing;
        for cure in cures {
            match cure {
                "ice" => ice_stepped = true,
                "icereb" => {
                    ice_stepped = true;
                    ice_rebound = true;
                }
                "reb" => rebound_every_pass = true,
                "base" => base_cure = true,
                "rebx" => rebound_exact = true,
                "lt" => landscape_time = true,
                "lake" => lake_cure = true,
                "wedge" => wedge = true,
                "level" => level_plateau = true,
                "wrain" => water = vd_bins::solve_copy::ErosionWater::RainOnWaterTree,
                "water" => water = vd_bins::solve_copy::ErosionWater::BudgetRunoff,
                other => panic!(
                    "a cure is ice, icereb, reb, rebx, base, lt, lake, wedge, level, wrain or water, not {other}"
                ),
            }
        }
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
                ..Laws::default()
            },
            "b40-craters" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                craters_at_own_age: true,
                ..Laws::default()
            },
            "b40-joint-kept" | "b40-joint-slow" | "b40-joint-kept-slow" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                craters_at_own_age: true,
                joint_law: true,
                crust_partition: name.contains("kept"),
                collision_slowing: name.contains("slow"),
                ..Laws::default()
            },
            "b40-craters-kept-slow" => Laws {
                limits_as_law: true,
                belts_by_influx: true,
                mean_speed: true,
                force_balance: true,
                craters_at_own_age: true,
                crust_partition: true,
                collision_slowing: true,
                ..Laws::default()
            },
            other => panic!(
                "a case is today, belts-today-limits, limits, belts, today-geometric, belts-geometric, b40, b40-geometric, today-craters, b40-craters, joint, b40-joint, joint-geometric, b40-joint-geometric, b40-joint-kept, b40-joint-slow, b40-joint-kept-slow or b40-craters-kept-slow, not {other}"
            ),
        };
        let (laws, schedule) = if let Some((t_min, t_max, cadence)) = axis {
            let times = vd_bins::solve_copy::mixed_times(words.age_yr, t_min, t_max);
            let passes = (times.len() - 1) as u32;
            (
                Laws {
                    mixed_axis: Some((t_min, t_max)),
                    ..laws
                },
                Schedule {
                    passes,
                    climate_every: standard.climate_every * cadence,
                    isostasy_every: standard.isostasy_every * cadence,
                    ..standard
                },
            )
        } else {
            (laws, schedule)
        };
        let laws = Laws {
            sub_sweeps,
            sub_route,
            ice_stepped,
            ice_rebound,
            rebound_every_pass,
            base_cure,
            rebound_exact,
            landscape_time,
            lake_cure,
            wedge,
            level_plateau,
            water,
            ..laws
        };
        let case = &if sub_sweeps > 1 {
            format!("{case}%{sub_sweeps}{}", if sub_route { "r" } else { "" })
        } else {
            case.clone()
        };
        let started = Instant::now();
        let solved = solve_copy(
            &body,
            &words,
            schedule,
            Router::D8,
            &laws,
            None,
            ledger_dir.is_some(),
        )
        .expect("the home planet solves");
        println!(
            "\n== CASE {case} ({laws:?}): solved in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        if let Some(dir) = &ledger_dir {
            write_ledger(&format!("{dir}/{case}.csv"), &solved.ledger);
        }
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
                 {:.0} m, the largest {:.0} m; P2 the crust a collision keeps at 10/50/90 %: {:.2} / {:.2} / {:.2}; S2 a \
                 collision's influx against a constant speed's at 10/50/90 %: {:.2} / {:.2} / {:.2}; the rock raised over \
                 the windows: collisions {:.3e} m³, Andean margins {:.3e} m³",
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
                r.max_total_m,
                r.kept_share[0],
                r.kept_share[1],
                r.kept_share[2],
                r.slowing_share[0],
                r.slowing_share[1],
                r.slowing_share[2],
                r.volume_m3[0],
                r.volume_m3[1]
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
        let times = match laws.mixed_axis {
            Some((t_min, t_max)) => vd_bins::solve_copy::mixed_times(words.age_yr, t_min, t_max),
            None => vd_bins::solve_copy::pass_times(words.age_yr, passes, laws.geometric_last_yr),
        };
        let last_dt_yr = times[times.len() - 1] - times[times.len() - 2];
        read_case(&lattice, &area, &solved, last_dt_yr);
    }
}
