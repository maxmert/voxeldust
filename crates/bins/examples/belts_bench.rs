//! ★ THE BELTS' BENCH (rulings B37 and B38; the design `docs/investigation/2026-10-02/belts_and_time_design.md` §5):
//! the home planet solved by the ONE COPY of the macro solve (`vd_bins::solve_copy`) with the design's laws switched IN
//! THE COPY ONLY — nothing in the game changes. Each case prints Earth's set (the stands' relief rule and land readings,
//! one copy with the Earth instrument) and the mountains split BY CAUSE.
//!
//! ★ RE-BASED ON PHASE 2 (2026-10-04, rulings B52–B53): the game's solve now runs every ruled law, and the copy is that
//! solve with ONE law taken out at a time. The bench's own forms of the laws and their cases (runs 9–30, `today`,
//! `b40-joint-kept-slow+base+icereb+reb+rebx+lt+lake+wedge+level+wrain~mixed%2`, …) are recorded in
//! `docs/investigation/2026-10-02/phase0_belts_bench.txt` and in git; they ran on the pre-phase-2 erosion.
//!
//! **THE CASES** (`cargo run --release -p vd-bins --example belts_bench [-- phase2 phase2-water …]`):
//! - `phase2`: every law as the game runs it. ★ THE CONTROL: its artifact must match the release gate's pinned digest.
//! - `phase2-<law>…`: the same with each named law taken out (`vd_bins::solve_copy::Laws`): `-water` the water's own
//!   paths (B51 item 2), `-lt` the landscape's time (B45–B46), `-craters` the craters at their own age (B37), `-wedge`
//!   the critical wedges (B49–B51; the belts by the room come back); generator 26's land laws in the land copy
//!   (`vd_bins::land_copy`, rulings B59–B61) — `-cell` the cell's mean, `-margins` the margins by kind (and the Andean
//!   coast), `-spread` the sourced spread, `-relief` the older landscape, `-share` the crust share read continuously.
//!   Each is a MEASUREMENT of that law's own effect, never a candidate (ruling B38 item 4: the room is a cap, ruling
//!   B37 item 4).
//! - any case `~fine` / `~finer`: the convergence gate's axis twice / four times as fine (rulings B39, B47): every pass
//!   length of the solve's own mixed axis halved / quartered, so about twice / four times the passes (and the routings),
//!   the climate every pass on either axis (the solve's own, ruling B69 item 1).
//! - any case `%K` (last): K sweeps on each pass's one routing, each a K-th of the pass's time — a resolution (the solve's
//!   own is eight, ruling B62).
//! - any case `@R` (last of all): the water routed R times a pass, the pass's sweeps shared among the routings — a
//!   resolution (ruling B63 item 1 measured with it; the solve's own is four, ruling B65).
//! - `+seasons` / `+seasonsN`: the seasons' chain (the game's climate since generator 27, every 8th pass) re-solved every
//!   pass / every N passes, the running sea every pass (`Laws::rain_every`); `-seasons` takes it out (generator 26's climate).
//! - any case `^C` (just before `@R`): the climate and the running sea re-solved every C passes, counted in passes on
//!   either axis — a resolution (ruling B68: the sea's stale spacing near the present; `Laws::climate_every`); without
//!   it every pass, the solve's own since ruling B69 item 1 (`^20` is the game's spacing until then).
//! - `--ledger=DIR`: each case also writes its PASS LEDGER to `DIR/<case>.csv` — each process's change of volume on the
//!   land and the sea floor, and the land's readings, pass by pass (ruling B41 item 1: the gate's cause) — and its LEDGER
//!   BY PLATE AND PASS to `DIR/<case>.plates.csv` (ruling B66 item 1: which process, where, when). `<case>` is the case's
//!   full name, its axis and routings included.
//! - `--dump=DIR`: each case also writes its final field to `DIR/<case>.<row>.bin` (heights, the relief code, the plate,
//!   the water, the ocean, each node's cause and where its water ends).
//!
//! **THE CAUSES**, exclusive, in this order: a CLOSED BASIN'S WALL (the relief window reaches the sea level or under
//! it); a CRATER (within one diameter of a crater's centre, by the summed chord); a BELT (the folded belt's rock: the
//! nodes a belt raised) — by a collision, at an Andean margin, or by the room; a WEDGE'S PLATEAU (a node that stands on
//! a wedge's level plateau at the end, B50–B51); the OLD PLAINS (all other land).
//!
//! **Example.** If `phase2` reads two per cent of the land as mountains and most of them as BELT, the wedges put the
//! ranges where the plates close; if `phase2-wedge` reads them as CRATER, the room's belts did not, and the craters' rims
//! made them.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use vd_bins::solve_copy::{Laws, Solved, artifact_of, solve_copy};
use vd_bins::stands::{LandReading, MOUNTAINS, ReliefReading, merged, read_land, relief_readings};
use vd_terrain::belts::{BELT_ANDEAN, BELT_COLLISION};
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

/// The release gate's pinned digest of the home planet's artifact (`crates/bins/tests/home_artifact_pin.rs`).
const HOME_ARTIFACT_DIGEST: [u64; 2] = [0xddad_33b9_72ca_7ebb, 0xfcb3_cec0_6fe7_11ba];

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
/// Reads a solved case and prints its readings; returns each node's CAUSE (the exclusive causes' index, 255 off the land)
/// and where its WATER ENDS (bit 1: at the sea; bit 2: through a standing lake), for the field dump (ruling B66 item 1:
/// where two axes differ).
fn read_case(
    lattice: &MacroLattice,
    area: &[u64],
    solved: &Solved,
    last_dt_yr: u64,
) -> (Vec<u8>, Vec<u8>) {
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
    let causes: Vec<u8> = (0..n)
        .map(|v| readings[v].map_or(u8::MAX, |_| cause(v) as u8))
        .collect();
    let drains = {
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
        (0..n)
            .map(|v| u8::from(to_sea[v]) | (u8::from(through_lake[v]) << 1))
            .collect::<Vec<u8>>()
    };
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
    (causes, drains)
}

/// The final field of a case as raw rows (`--dump`): heights, the stands' relief code, the plate, the water, the ocean,
/// and the case reader's own rows — each node's cause and where its water ends ([`read_case`]).
fn dump_case(
    dir: &str,
    case: &str,
    lattice: &MacroLattice,
    solved: &Solved,
    causes: Vec<u8>,
    drains: Vec<u8>,
) {
    let st = &solved.state;
    let z_m: Vec<i32> = st.z.iter().map(|&z| z / Z_STEPS_PER_M).collect();
    let land: Vec<bool> = st.z.iter().map(|&z| z > st.sea_z).collect();
    let inland: Vec<bool> = vec![false; st.z.len()];
    let class = vd_bins::stands::relief_classes(lattice, &z_m, &land, &inland);
    let rows: [(&str, Vec<u8>); 7] = [
        ("z", st.z.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ("class", class),
        ("plate", solved.plate_of.clone()),
        (
            "water",
            st.water_z.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        ("ocean", st.ocean.iter().map(|&o| u8::from(o)).collect()),
        ("cause", causes),
        ("drain", drains),
    ];
    for (name, bytes) in rows {
        let path = format!("{dir}/{case}.{name}.bin");
        match std::fs::write(&path, bytes) {
            Ok(()) => println!("  the dump: {path}"),
            Err(e) => println!("  the dump: could not write {path}: {e}"),
        }
    }
}

/// The pass ledger as CSV, one line a pass (ruling B41 item 1, the gate's cause).
fn write_ledger(path: &str, ledger: &[vd_bins::solve_copy::PassLedger]) {
    use std::fmt::Write as _;
    let mut out = String::from(
        "t0_yr,t1_yr,craters_land,craters_sea,talus_land,talus_sea,ice_land,ice_sea,uplift_land,uplift_sea,\
         sweep_land,sweep_sea,rebound_land,rebound_sea,peak_land,peak_sea,ice_steps,sea_z,land_mean_m,land_over_1km,\
         land_share,rock_cut,rock_laid_open,rock_laid_closed,rock_over_spill,rock_sea,rock_dropped,rock_remainder,\
         rock_check,closed_bodies,closed_m2,opened_m2,newly_closed_m2,relief_p50_m,plains_pct\n",
    );
    for e in ledger {
        let _ = write!(out, "{},{}", e.t0_yr, e.t1_yr);
        for pair in [
            e.craters_m3,
            e.talus_m3,
            e.ice_m3,
            e.uplift_m3,
            e.sweep_m3,
            e.rebound_m3,
            e.peak_m3,
        ] {
            let _ = write!(out, ",{:.6e},{:.6e}", pair[0], pair[1]);
        }
        let _ = write!(
            out,
            ",{},{},{:.3},{:.6},{:.6}",
            e.ice_steps, e.sea_z, e.land_mean_m, e.land_over_1km, e.land_share
        );
        let _ = writeln!(
            out,
            ",{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{},{:.6e},{:.6e},{:.6e},{},{:.3}",
            e.rock_cut_m3,
            e.rock_laid_open_m3,
            e.rock_laid_closed_m3,
            e.rock_over_spill_m3,
            e.rock_sea_m3,
            e.rock_dropped_m3,
            e.rock_remainder_m3,
            e.rock_check,
            e.closed_bodies,
            e.closed_m2,
            e.opened_m2,
            e.newly_closed_m2,
            e.relief_p50_m,
            e.plains_pct
        );
    }
    match std::fs::write(path, out) {
        Ok(()) => println!("  the pass ledger: {path} ({} passes)", ledger.len()),
        Err(e) => println!("  the pass ledger: could not write {path}: {e}"),
    }
}

/// The ledger by plate and pass as CSV, one line a plate a pass (ruling B66 item 1: which process, where, when).
fn write_plate_ledger(path: &str, rows: &[vd_bins::solve_copy::PlatePass]) {
    use std::fmt::Write as _;
    let mut out = String::from(
        "t1_yr,plate,craters,talus,ice,uplift,sweep,rebound,peak,land_m2,land_mean_m,calm_m2,calm_mean_m,under_m2\n",
    );
    for r in rows {
        let _ = write!(out, "{},{}", r.t1_yr, r.plate);
        for v in r.process_m3 {
            let _ = write!(out, ",{v:.6e}");
        }
        let _ = writeln!(
            out,
            ",{:.6e},{:.3},{:.6e},{:.3},{:.6e}",
            r.land_m2, r.land_mean_m, r.calm_m2, r.calm_mean_m, r.under_m2
        );
    }
    match std::fs::write(path, out) {
        Ok(()) => println!(
            "  the ledger by plate and pass: {path} ({} rows)",
            rows.len()
        ),
        Err(e) => println!("  the ledger by plate and pass: could not write {path}: {e}"),
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--ledger=DIR`: each case writes its pass ledger to `DIR/<case>.csv` (ruling B41 item 1, the gate's cause).
    let ledger_dir: Option<String> = args
        .iter()
        .find_map(|a| a.strip_prefix("--ledger=").map(str::to_string));
    args.retain(|a| !a.starts_with("--ledger="));
    // `--dump=DIR`: each case also writes its final field to `DIR/<case>.<row>.bin` (the gate's cause: where two axes
    // differ) — `z` (i32, sixteenths), `class` (u8, the stands' relief code), `plate` (u8), `water` (i32), `ocean` (u8),
    // `cause` (u8, the causes' index, 255 off the land) and `drain` (u8: 1 at the sea, 2 through a standing lake).
    let dump_dir: Option<String> = args
        .iter()
        .find_map(|a| a.strip_prefix("--dump=").map(str::to_string));
    args.retain(|a| !a.starts_with("--dump="));
    // `--artifact=DIR`: each case also writes its artifact, exactly as the shard's worker makes it, to `DIR/<case>.art`
    // (ruling B58 item 2: the readers of the shipped rows run on a candidate's own world).
    let artifact_dir: Option<String> = args
        .iter()
        .find_map(|a| a.strip_prefix("--artifact=").map(str::to_string));
    args.retain(|a| !a.starts_with("--artifact="));
    let cases: Vec<String> = if args.is_empty() {
        vec!["phase2".to_string()]
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
        // The files carry the case's full name (its axis and routings too, so two cases never write one file).
        let full = case.as_str();
        // `case…@R` (last of all): the water routed R times a pass, the sweeps shared among the routings (B63 item 1).
        let (case, routings) = match case.rsplit_once('@') {
            Some((c, r)) => (c, Some(r.parse::<u32>().expect("a routing count after @"))),
            None => (case.as_str(), None),
        };
        // `case…^C` (just before `@R`): the climate and the sea re-solved every C passes, in passes on either axis — a
        // resolution (ruling B68 measured with it); without it every pass, the solve's own (ruling B69 item 1).
        let (case, climate_every) = match case.rsplit_once('^') {
            Some((c, k)) => (
                c,
                Some(k.parse::<u32>().expect("a climate spacing after ^")),
            ),
            None => (case, None),
        };
        // `case%K`: K sweeps a pass (a resolution).
        let (case_axis, sub_sweeps) = match case.split_once('%') {
            Some((c, k)) => (c, k.parse::<u32>().expect("a sweep count after %")),
            None => (case, vd_terrain::solve::SUB_SWEEPS),
        };
        // `case~fine` / `case~finer`: the convergence gate's axis twice / four times as fine, the climate every pass.
        let (name, fineness) = if let Some(name) = case_axis.strip_suffix("~finer") {
            (name, 4u64)
        } else if let Some(name) = case_axis.strip_suffix("~fine") {
            (name, 2u64)
        } else {
            (case_axis, 1u64)
        };
        // `phase2+pass…`: a CANDIDATE under measurement switched in (never a ruled law; `Laws::pass_share`).
        let (name, candidates) = name.split_once('+').unwrap_or((name, ""));
        let mut parts = name.split('-');
        assert_eq!(
            parts.next(),
            Some("phase2"),
            "a case is phase2[-water][-lt][-craters][-wedge][-seasons][-cell][-margins][-spread][-relief][-share][+pass|+flats|+cellK|+airy|+flex|+blendM|+scatterA][~fine][%K]"
        );
        let mut laws = Laws::default();
        vd_bins::solve_copy::RAIN_RENORMALISED.store(false, std::sync::atomic::Ordering::Relaxed);
        if let Some(every) = climate_every {
            laws.climate_every = every;
        }
        for cand in candidates.split('+').filter(|c| !c.is_empty()) {
            match cand {
                "pass" => laws.pass_share = true,
                "flats" => laws.routing_flats = true,
                // Ruling B94 step 2: the rain belt renormalised (`solve_copy::RAIN_RENORMALISED`, a probe).
                "rain" => vd_bins::solve_copy::RAIN_RENORMALISED
                    .store(true, std::sync::atomic::Ordering::Relaxed),
                // Ruling B99 step 1 (generator 27's S3): the seasons' chain every pass (the game's: every 8th).
                "seasons" => laws.rain_every = 1,
                // The seasons' chain every N passes, the sea every pass (`Laws::rain_every`).
                every if every.starts_with("seasons") => {
                    laws.rain_every = every["seasons".len()..]
                        .parse()
                        .expect("a pass count after seasons");
                }
                other => {
                    // The land copy's forms and diagnostics (rulings B58 item 2, B59–B61): cellK, airy, flex; the game's
                    // own margins, spread, spectrum and share (no change on generator 26); the diagnostics blendM
                    // (metres) and scatterA (sixteenths).
                    use vd_bins::land_copy::Relief;
                    if let Some(k) = other.strip_prefix("cell") {
                        laws.land.cell = Some(k.parse().expect("cellK"));
                    } else if let Some(m) = other.strip_prefix("blend") {
                        laws.land.blend_m = Some(m.parse().expect("blendM"));
                    } else if let Some(a) = other.strip_prefix("scatter") {
                        laws.land.scatter = Some(a.parse().expect("scatterA"));
                    } else {
                        match other {
                            "margins" => laws.land.margins = true,
                            "spread" => laws.land.spread = true,
                            "airy" => laws.land.relief = Relief::Airy,
                            "flex" => laws.land.relief = Relief::Flex,
                            "spectrum" => laws.land.relief = Relief::Spectrum,
                            "share" => laws.land.share = true,
                            _ => panic!(
                                "a candidate is pass, flats, cellK, margins, spread, airy, flex, spectrum, share, blendM or scatterA, not {other}"
                            ),
                        }
                    }
                }
            }
        }
        for out in parts {
            match out {
                "water" => laws.water_paths = false,
                "lt" => laws.landscape_time = false,
                "craters" => laws.craters_at_own_age = false,
                "wedge" => laws.wedge = false,
                // Generator 27's climate (rulings B99–B102) out: generator 26's, every pass.
                "seasons" => laws.seasons = false,
                // Generator 26's land laws (rulings B59–B61), each taken out in the land copy only.
                "cell" => laws.land.cell = None,
                "margins" => laws.land.margins = false,
                "spread" => laws.land.spread = false,
                "relief" => laws.land.relief = vd_bins::land_copy::Relief::None,
                "share" => laws.land.share = false,
                other => panic!(
                    "a law taken out is water, lt, craters, wedge, seasons, cell, margins, spread, relief or share, not {other}"
                ),
            }
        }
        let standard = Schedule::standard(words.age_yr);
        // `VD_BENCH_T_MIN_YR`: the axis's last pass in years, in place of the axis's own (the gate's cause: a change of
        // the discretization at one resolution — the chaos test). Unset, the axis is the solve's own.
        let t_min_yr = std::env::var("VD_BENCH_T_MIN_YR")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(standard.t_min_yr / fineness);
        let schedule = Schedule {
            t_min_yr,
            t_max_yr: standard.t_max_yr / fineness,
            sub_sweeps,
            routings: routings.unwrap_or(standard.routings),
            ..standard
        };
        let started = Instant::now();
        let solved = solve_copy(&body, &words, schedule, &laws, ledger_dir.is_some())
            .expect("the home planet solves");
        println!(
            "\n== CASE {case} ({laws:?}, {} passes, {} sweeps and {} routings a pass): solved in {:.1} s",
            schedule.times().len() - 1,
            schedule.sub_sweeps,
            schedule.routings,
            started.elapsed().as_secs_f64()
        );
        if let Some(dir) = &ledger_dir {
            write_ledger(&format!("{dir}/{full}.csv"), &solved.ledger);
            write_plate_ledger(&format!("{dir}/{full}.plates.csv"), &solved.plate_passes);
            // THE LEDGER BY PLATE (ruling B61 item 2): each plate's volume change by process over the solve, km³.
            println!(
                "  THE LEDGER BY PLATE, km³ over the solve: plate | craters | talus | ice | uplift | sweeps | rebounds | peak law | net"
            );
            for (p, row) in solved.by_plate.iter().enumerate() {
                let net: f64 = row.iter().sum();
                println!(
                    "    plate {p:2}: {}",
                    row.iter()
                        .chain(std::iter::once(&net))
                        .map(|v| format!("{:12.0}", v / 1e9))
                        .collect::<Vec<_>>()
                        .join(" |")
                );
            }
        }
        if let Some(dir) = &artifact_dir {
            let path = format!("{dir}/{case}.art");
            match vd_bins::artifact_file::write(&path, &artifact_of(&words, &solved)) {
                Ok(()) => println!("  the artifact: {path}"),
                Err(e) => println!("  the artifact: could not write {path}: {e}"),
            }
        }
        if case == "phase2" {
            let digest = artifact_of(&words, &solved).digest();
            println!(
                "CONTROL the copy with every law as the game runs it: digest [{:#x}, {:#x}] {} the release gate's pin",
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
            let mut windows: Vec<f64> = b
                .wedge
                .iter()
                .map(|w| (words.age_yr - w.start_yr) as f64 / 1e6)
                .collect();
            windows.sort_by(f64::total_cmp);
            let at = |p: f64| {
                windows
                    .get(((windows.len().max(1) - 1) as f64 * p) as usize)
                    .copied()
                    .unwrap_or(0.0)
            };
            println!(
                "  THE BELTS' LAW (the critical wedges): {} nodes rise ({} in collisions, {} at Andean margins, {} on a \
                 plateau at the end); the plates' turnover {:.0} Myr; the windows the risen nodes read at 10/50/90 %: \
                 {:.1} / {:.1} / {:.1} Myr; the rock raised by the end: collisions {:.3e} m³, Andean margins {:.3e} m³",
                r.nodes,
                r.collision_nodes,
                r.andean_nodes,
                r.plateau_nodes,
                r.t_m_yr as f64 / 1e6,
                at(0.1),
                at(0.5),
                at(0.9),
                r.volume_m3[0] as f64,
                r.volume_m3[1] as f64
            );
        }
        println!(
            "  THE PEAK LAW: {} node collapses over the passes and their rounds; the largest excess left over its cap \
             after a pass: {:.0} m",
            solved.collapsed,
            f64::from(solved.peak_left) / f64::from(Z_STEPS_PER_M)
        );
        println!(
            "  THE STEPPED ICE: {} steps over the epoch; the one-year guard acted {} times (DEFERRED D-TERRAIN-10)",
            solved.report.ice_steps, solved.report.ice_floored
        );
        // The rock's ways (the game's deposit, 2026-10-05): a way refused because the water's way from its gate ends in a
        // body the rest already passed, at an inland end or on a ring; the rock with no way left is dropped and counted.
        let refused: usize = solved.report.sweeps.iter().map(|s| s.rock_refused).sum();
        let stuck: u128 = solved.report.sweeps.iter().map(|s| s.rock_stuck).sum();
        println!(
            "  THE ROCK'S WAYS: {refused} ways refused over the sweeps; the rock with no way {:.3e} m³",
            stuck as f64 / f64::from(Z_STEPS_PER_M)
        );
        if laws.routing_flats {
            println!(
                "  THE CANDIDATE (the routing's receivers on open land): {} nodes given the water's own receiver back over the passes (a ring of receivers)",
                solved.flats_repaired
            );
        }
        let times = schedule.times();
        let last_dt_yr = times[times.len() - 1] - times[times.len() - 2];
        let (causes, drains) = read_case(&lattice, &area, &solved, last_dt_yr);
        if let Some(dir) = &dump_dir {
            dump_case(dir, full, &lattice, &solved, causes, drains);
        }
    }
}
