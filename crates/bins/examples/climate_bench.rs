//! ★ THE SEASONS' CLIMATE BENCH (ruling B99 step 1, generator 27; the design
//! `docs/investigation/2026-10-15/rain_seasons_design.md` §5). Read only: no world change, nothing the game reads.
//!
//! **THE MODES.**
//! - `earth-data DIR`: Earth's relief binned onto the home planet's 8 km nodes (`vd_bins::earth`), cached as
//!   `DIR/earth_z.bin` (one little-endian `i32` of whole metres a node), the sea by the solve's own rule; GPCP's measured
//!   rain as the reference maps `DIR/gpcp_{year,jan,jul}.png` (an even longitude–latitude picture, the one rain scale)
//!   and its global and land means.
//! - `earth-s2 DIR [ROWS] [SEASONS]` / `home-s2 DIR FILE.art [ROWS] [SEASONS]`: THE RAIN (S2) — the energy balance, the
//!   winds and the vapour of each season on the climate grid, twice (the second round's land as wet as the first round's
//!   rain makes it); the rain's global and land means, its zonal profile, the land's dry share and named places beside
//!   GPCP (Earth); the maps `DIR/<tag>_rain_{year,jan,jul}.png` on GPCP's own scale.
//! - `export-latlon FILE.art OUT_DIR NLAT NLON`: the home planet's stored world on a latitude–longitude grid for an
//!   OUTSIDE measuring stick (ExoPlaSim, the owner's word 2026-10-15): at NLAT Gaussian latitudes (south to north) and
//!   NLON even longitudes from 0° east, each cell's land share and mean ground height over the sea (m, the sea's cells at
//!   nought), as `OUT_DIR/land_share.npy` and `OUT_DIR/height_m.npy` (NLAT × NLON little-endian f32), the latitudes in
//!   `OUT_DIR/lats.txt`; every node lands in the cell whose centre is nearest in latitude and longitude.
//! - `stack OUT.png A.png B.png …`: pictures stacked top to bottom for a side-by-side look.
//! - `chain-fixed FILE.art [THREADS]`: the seasons' chain from a cold state on a stored world's ground, its spin-up call by
//!   call.
//! - `art-maps FILE.art OUT_DIR TAG [LAT0 LAT1 LON0 LON1]`: a stored world (optionally only the window between the two
//!   latitudes and the two east longitudes, degrees, at the same picture width: a region at node detail) (the home planet's) as three maps on one even longitude–latitude
//!   picture — `OUT_DIR/TAG_relief.png` (the ground by height, shaded by a light from the north-west, the sea dark),
//!   `OUT_DIR/TAG_rain.png` (the rain rows on the one rain scale), `OUT_DIR/TAG_rivers.png` (the discharge rows, the
//!   larger rivers brighter) — and the land's rain readings.
//! - `earth-s1 DIR [ROWS]`: THE ENERGY BALANCE AND THE RAIN BELT on Earth's charter over Earth's land and sea (S1): the
//!   periodic year of `vd_terrain::seasons::balance` on the climate grid (ROWS latitude bands, 90 by default), its cost,
//!   the zonal temperatures by season, a continental swing, and the energy-flux equator by month beside GPCP's zonal
//!   rain maximum; the maps `DIR/earth_t_{jan,jul}.png`.
//!
//! The data (outside the repo): `VD_EARTH_DIR` (default `~/voxeldust_earth/climate`) holds
//! `etopo2022_60s_i16le.raw`, `gpcp.npy`, `u10.npy`, `v10.npy` (`README.txt` there).
//!
//! `cargo run --release -p vd-bins --example climate_bench -- earth-data /tmp/vd-g27/bench`
//!
//! **Example.** The reference map shows the Amazon at ~2 400 mm a year and the Sahara under 50 mm; the seasons' law on
//! Earth's charter is read beside it, node by node.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "a bench reads its cells by index beside several arrays; its readings take many fields"
)]

use std::path::{Path, PathBuf};

use vd_bins::earth::{MonthlyGrid, bin_heights, dir_of, lat_lon_of, read_etopo};
use vd_terrain::Gf;
use vd_terrain::home::home_planet;
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::seasons::balance::{
    BalanceWords, Surface, balance_words, capacity_j_m2_k, moist_energy, periodic_year,
};
use vd_terrain::seasons::grid::{ClimateGrid, unit};
use vd_terrain::seasons::orography::{NodeGeometry, mountain_rain};
use vd_terrain::seasons::sun::{SunTable, YEAR_SAMPLES_SHIFT};
use vd_terrain::seasons::vapour::{Forcing, RESIDENCE_S, RainLaw, season_water};
use vd_terrain::seasons::wind::{StepFields, Winds, winds};
use vd_terrain::solve::SolveWords;

/// The pictures' width; the height is half of it (an even longitude–latitude map).
const MAP_W: u32 = 1_440;
/// GPCP's days a year (mm/day to mm/yr).
const DAYS: f64 = 365.25;

fn earth_dir() -> PathBuf {
    std::env::var_os("VD_EARTH_DIR").map_or_else(
        || {
            let home = std::env::var_os("HOME").expect("HOME");
            Path::new(&home).join("voxeldust_earth/climate")
        },
        PathBuf::from,
    )
}

/// The rain scale every map shares: a logarithmic ramp from 10 mm/yr (pale sand) through 250 and 1 000 to 4 000 mm/yr
/// (deep blue).
fn rain_colour(mm_yr: f64) -> [u8; 3] {
    if mm_yr.is_nan() {
        return [70, 80, 96];
    }
    const STOPS: [(f64, [f64; 3]); 6] = [
        (10.0, [244.0, 236.0, 214.0]),
        (100.0, [222.0, 196.0, 140.0]),
        (250.0, [190.0, 200.0, 120.0]),
        (600.0, [110.0, 170.0, 90.0]),
        (1_500.0, [40.0, 120.0, 150.0]),
        (4_000.0, [20.0, 40.0, 120.0]),
    ];
    let x = mm_yr.max(STOPS[0].0).log10();
    let mut c = STOPS[STOPS.len() - 1].1;
    for w in STOPS.windows(2) {
        let (a, b) = (w[0].0.log10(), w[1].0.log10());
        if x <= b {
            let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
            c = [0, 1, 2].map(|k| w[0].1[k] + (w[1].1[k] - w[0].1[k]) * t);
            break;
        }
    }
    c.map(|v| v.round() as u8)
}

/// An even longitude–latitude picture of `f(lat, lon)` coloured by `colour`, longitude 0 at the left edge.
fn map_png(path: &Path, f: &dyn Fn(f64, f64) -> f64, colour: &dyn Fn(f64) -> [u8; 3]) {
    let h = MAP_W / 2;
    let mut img = image::RgbImage::new(MAP_W, h);
    for y in 0..h {
        let lat = 90.0 - (f64::from(y) + 0.5) * 180.0 / f64::from(h);
        for x in 0..MAP_W {
            let lon = (f64::from(x) + 0.5) * 360.0 / f64::from(MAP_W);
            img.put_pixel(x, y, image::Rgb(colour(f(lat, lon))));
        }
    }
    img.save(path).expect("the picture saves");
}

fn earth_data(out: &Path) {
    std::fs::create_dir_all(out).expect("the output folder");
    let dir = earth_dir();
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("the home planet's lattice");
    let n = lattice.node_count();
    let started = std::time::Instant::now();
    let grid = read_etopo(
        dir.join("etopo2022_60s_i16le.raw")
            .to_str()
            .expect("a path"),
    );
    let z = bin_heights(&lattice, &grid);
    drop(grid);
    let mut bytes = Vec::with_capacity(n * 4);
    for v in &z {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out.join("earth_z.bin"), bytes).expect("the heights save");
    let area: Vec<u64> = (0..n as u32).map(|k| lattice.area_m2(k)).collect();
    let z16: Vec<i32> = z
        .iter()
        .map(|&m| m * vd_terrain::solve::Z_STEPS_PER_M)
        .collect();
    let ocean = vd_terrain::land::ocean_mask(&lattice, &z16, &area, 0).ocean;
    let total: f64 = area.iter().map(|&a| a as f64).sum();
    let sea: f64 = (0..n).filter(|&i| ocean[i]).map(|i| area[i] as f64).sum();
    println!(
        "EARTH ON THE HOME LATTICE: {n} nodes binned in {:.1} s; the connected ocean {:.2} % of the surface (Earth 70.8 %)",
        started.elapsed().as_secs_f64(),
        100.0 * sea / total
    );
    // GPCP's year and its two solstice months, the global and the land means by each node's own area.
    let gpcp = MonthlyGrid::gpcp(dir.join("gpcp.npy").to_str().expect("a path"));
    let year =
        |lat: f64, lon: f64| (0..12).map(|m| gpcp.at(m, lat, lon)).sum::<f64>() / 12.0 * DAYS;
    let (mut all, mut land, mut land_area) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (lat, lon) = lat_lon_of(dir_of(&lattice, i as u32));
        let p = year(lat, lon);
        all += p * area[i] as f64;
        if !ocean[i] {
            land += p * area[i] as f64;
            land_area += area[i] as f64;
        }
    }
    println!(
        "GPCP 1991–2020: the global mean {:.0} mm/yr, the land's mean {:.0} mm/yr (the land as this lattice's sea rule names it)",
        all / total,
        land / land_area
    );
    map_png(&out.join("gpcp_year.png"), &year, &rain_colour);
    map_png(
        &out.join("gpcp_jan.png"),
        &|lat, lon| gpcp.at(0, lat, lon) * DAYS,
        &rain_colour,
    );
    map_png(
        &out.join("gpcp_jul.png"),
        &|lat, lon| gpcp.at(6, lat, lon) * DAYS,
        &rain_colour,
    );
    println!("the maps: {}", out.display());
}

/// ★ EARTH'S CHARTER, word for word (the G-ICE test's words, `vd_terrain::climate`): Earth's flux, air, day, tilt
/// (cos 23.4° = 940/1 024), eccentricity, year and water, on the home planet's radius and gravity.
fn earth_words() -> SolveWords {
    SolveWords {
        insolation_q12: 4_096,
        t_eq_mk: 255_000,
        t_surface_mk: Some(288_000),
        bond_albedo_q12: 1_253,
        mu_q8: Some(7_416),
        scale_height_m: Some(8_500),
        p_surf_pa: Some(101_325),
        tau_ir_q12: Some(2_458),
        day_s: Some(86_400),
        obliquity_cos_q1024: Some(940),
        ecc_q16: 1_097,
        year_s: 31_557_600,
        water_km3: 1_386_000_000,
        ..vd_terrain::home::home_solve_words()
    }
}

/// The sun's table of a charter: its flux, eccentricity and tilt.
fn sun_of(words: &SolveWords) -> SunTable {
    let s = Gf::from_f64(vd_terrain::climate::SOLAR_CONSTANT_W_M2)
        * Gf::from_i64(i64::from(words.insolation_q12))
        / Gf::from_i64(4_096);
    let e = Gf::from_i64(i64::from(words.ecc_q16)) / Gf::from_i64(65_536);
    let c =
        Gf::from_i64(i64::from(words.obliquity_cos_q1024.unwrap_or(1_024))) / Gf::from_i64(1_024);
    SunTable::new(s, e, c, 1_025, YEAR_SAMPLES_SHIFT)
}

/// Each climate cell's sea share by area, from the fine lattice's ocean mask.
fn sea_share(fine: &MacroLattice, grid: &ClimateGrid, ocean: &[bool]) -> Vec<Gf> {
    let mut all = vec![0.0f64; grid.len()];
    let mut sea = vec![0.0f64; grid.len()];
    for (node, &wet) in ocean.iter().enumerate() {
        let c = grid.cell_at(unit(fine.direction(node as u32)));
        let a = fine.area_m2(node as u32) as f64;
        all[c] += a;
        if wet {
            sea[c] += a;
        }
    }
    all.iter()
        .zip(&sea)
        .map(|(a, s)| Gf::from_f64(if *a > 0.0 { s / a } else { 1.0 }))
        .collect()
}

/// The step's winds of a periodic year (the heat store from the step before).
fn step_winds(
    grid: &ClimateGrid,
    bw: &BalanceWords,
    omega: Gf,
    year: &vd_terrain::seasons::balance::Year,
    surface: &Surface,
    a_olr: Gf,
    st: usize,
    departures: bool,
) -> Winds {
    let steps = year.t.len();
    let prev = (st + steps - 1) % steps;
    let dt = bw.year_s / Gf::from_i64(steps as i64);
    let g: Vec<Gf> = (0..grid.len())
        .map(|i| surface.capacity[i] * (year.t[st][i] - year.t[prev][i]) / dt)
        .collect();
    winds(
        grid,
        bw,
        omega,
        StepFields {
            t: &year.t[st],
            kept: &year.kept[st],
            g_store: &g,
            rh: &surface.rh,
            a_olr,
            departures,
        },
    )
}

/// A wind map: the background field coloured, and one short line a few cells apart along the wind (its length the
/// speed, 1 px per m/s), white with a dark head at the downwind end.
fn wind_png(
    path: &Path,
    grid: &ClimateGrid,
    w: &Winds,
    background: &dyn Fn(f64, f64) -> f64,
    colour: &dyn Fn(f64) -> [u8; 3],
) {
    let h = MAP_W / 2;
    let mut img = image::RgbImage::new(MAP_W, h);
    for y in 0..h {
        let lat = 90.0 - (f64::from(y) + 0.5) * 180.0 / f64::from(h);
        for x in 0..MAP_W {
            let lon = (f64::from(x) + 0.5) * 360.0 / f64::from(MAP_W);
            img.put_pixel(x, y, image::Rgb(colour(background(lat, lon))));
        }
    }
    let every = (grid.rows / 30).max(1);
    for r in (every / 2..grid.rows).step_by(every) {
        for c in (every / 2..grid.cols).step_by(every) {
            let i = r * grid.cols + c;
            let lat = -90.0 + (r as f64 + 0.5) * 180.0 / grid.rows as f64;
            let lon = (c as f64 + 0.5) * 360.0 / grid.cols as f64;
            let (px, py) = (
                lon / 360.0 * f64::from(MAP_W),
                (90.0 - lat) / 180.0 * f64::from(h),
            );
            let (u, v) = (w.east[i].to_f64(), w.north[i].to_f64());
            let len = (u * u + v * v).sqrt();
            let k = len.max(1e-9);
            let steps = (len * 2.0).ceil() as i32;
            for s in 0..=steps {
                let t = f64::from(s) / f64::from(steps.max(1));
                let qx = px + u / k * len * 2.0 * t;
                let qy = py - v / k * len * 2.0 * t;
                if qx >= 0.0 && qy >= 0.0 && qx < f64::from(MAP_W) && qy < f64::from(h) {
                    let shade = if t > 0.7 {
                        [20, 20, 20]
                    } else {
                        [255, 255, 255]
                    };
                    img.put_pixel(qx as u32, qy as u32, image::Rgb(shade));
                }
            }
        }
    }
    img.save(path).expect("the picture saves");
}

/// The temperature scale: −40 °C deep blue, 0 °C white, +40 °C deep red.
fn temp_colour(k: f64) -> [u8; 3] {
    let t = ((k - 273.15) / 40.0).clamp(-1.0, 1.0);
    let c = if t < 0.0 {
        [255.0 * (1.0 + t), 255.0 * (1.0 + t), 255.0]
    } else {
        [255.0, 255.0 * (1.0 - t), 255.0 * (1.0 - t)]
    };
    c.map(|v: f64| v.round() as u8)
}

/// ★ THE ENERGY-FLUX EQUATOR of one step: the zonal rows' net heating `Σ (kept − OLR − G)·A`, its northward transport at
/// each row edge (the southern cap's net heating carried north), and the latitudes, degrees, where the transport turns
/// from southward to northward going north.
fn flux_equators(grid: &ClimateGrid, net_rows: &[f64]) -> Vec<f64> {
    let mut f = 0.0;
    let mut out = Vec::new();
    for r in 0..grid.rows - 1 {
        let before = f;
        f += net_rows[r];
        // The edge between row r and r + 1 stands at latitude −90 + (r + 1)·step.
        if before < 0.0 && f >= 0.0 {
            out.push(-90.0 + (r + 1) as f64 * 180.0 / grid.rows as f64);
        }
    }
    out
}

fn earth_s1(out: &Path, rows: usize) {
    std::fs::create_dir_all(out).expect("the output folder");
    let dir = earth_dir();
    let body = home_planet();
    let fine = MacroLattice::of(&body).expect("the home planet's lattice");
    let n = fine.node_count();
    let bytes =
        std::fs::read(out.join("earth_z.bin")).expect("run earth-data first: DIR/earth_z.bin");
    let z: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(z.len(), n);
    let area: Vec<u64> = (0..n as u32).map(|k| fine.area_m2(k)).collect();
    let z16: Vec<i32> = z
        .iter()
        .map(|&m| m * vd_terrain::solve::Z_STEPS_PER_M)
        .collect();
    let ocean = vd_terrain::land::ocean_mask(&fine, &z16, &area, 0).ocean;
    let gpcp = MonthlyGrid::gpcp(dir.join("gpcp.npy").to_str().expect("a path"));
    season_s1(
        out,
        "earth",
        &body,
        &fine,
        &ocean,
        &earth_words(),
        rows,
        Some(&gpcp),
    );
}

/// ★ THE HOME PLANET'S S1: its own charter on its stored world (`FILE.art`, the sea by the rows' facies).
fn home_s1(out: &Path, art: &str, rows: usize) {
    std::fs::create_dir_all(out).expect("the output folder");
    let body = home_planet();
    let fine = MacroLattice::of(&body).expect("the home planet's lattice");
    let artifact = vd_bins::artifact_file::read(art).expect("the world reads");
    let ocean: Vec<bool> = artifact
        .rows
        .iter()
        .map(|r| {
            (r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
                & vd_terrain::solve::FACIES_SEA
                != 0
        })
        .collect();
    assert_eq!(ocean.len(), fine.node_count());
    season_s1(
        out,
        "home",
        &body,
        &fine,
        &ocean,
        &vd_terrain::home::home_solve_words(),
        rows,
        None,
    );
}

/// The energy balance's periodic year on a body's sea and land, its readings and its maps (`DIR/<tag>_t_{jan,jul}.png`;
/// "January" and "July" are the two solstice months of the body's own year, counted from its spring equinox as
/// Earth's); with Earth's GPCP, the rain belt against the measured one.
fn season_s1(
    out: &Path,
    tag: &str,
    body: &vd_terrain::BodyDefinition,
    fine: &MacroLattice,
    ocean: &[bool],
    words: &SolveWords,
    rows: usize,
    gpcp: Option<&MonthlyGrid>,
) {
    let body = *body;
    let words = *words;
    let bw: BalanceWords = balance_words(&body, &words);
    let grid = ClimateGrid::new(Gf::from_f64(body.ladder().radius_m()), rows);
    let sea = sea_share(fine, &grid, ocean);
    let surface = Surface {
        capacity: sea
            .iter()
            .map(|&s| capacity_j_m2_k(&bw, s, words.year_s))
            .collect(),
        // S1's first reading: the land as wet as the sea (the land's wetness by its rain joins with S2, lagged).
        rh: vec![Gf::from_f64(0.8); grid.len()],
    };
    let sun = sun_of(&words);
    let start = vec![bw.t_mean_k; grid.len()];
    let steps_shift = 6;
    let t0 = std::time::Instant::now();
    let p = periodic_year(
        &grid,
        &bw,
        &sun,
        steps_shift,
        &surface,
        &start,
        (Gf::from_f64(0.01), 400),
    );
    let secs = t0.elapsed().as_secs_f64();
    let steps = 1usize << steps_shift;
    println!(
        "THE PERIODIC YEAR: {} years from a uniform {:.0} K start in {:.1} s ({:.2} s a year); the last year's largest change {:.4} K; the solve's worst iterations {} (all met: {})",
        p.years,
        bw.t_mean_k.to_f64(),
        secs,
        secs / p.years as f64,
        p.change_k.to_f64(),
        p.year.worst_iterations,
        p.year.all_met
    );
    println!(
        "  the ice-free albedo {:.3} (the charter's mean {:.3})",
        p.alpha_free.to_f64(),
        1.0 - bw.kept.to_f64()
    );
    // Zonal means: the annual mean, January (step 52 of 64 from the spring equinox) and July (step 20).
    let (jan, jul) = (steps * 52 / 64, steps * 20 / 64);
    let zonal = |t: &[Gf], r: usize| {
        (0..grid.cols)
            .map(|c| t[r * grid.cols + c].to_f64())
            .sum::<f64>()
            / grid.cols as f64
    };
    let mut annual = vec![0.0f64; grid.len()];
    for st in &p.year.t {
        for (a, v) in annual.iter_mut().zip(st) {
            *a += v.to_f64() / steps as f64;
        }
    }
    let annual_g: Vec<Gf> = annual.iter().map(|&v| Gf::from_f64(v)).collect();
    println!("THE ZONAL TEMPERATURE at the sea's level, °C: latitude, the year, January, July");
    for lat in [-80i32, -60, -40, -20, 0, 20, 40, 60, 80] {
        let r = ((f64::from(lat) + 90.0) / 180.0 * rows as f64) as usize;
        println!(
            "  {lat:>4}°  {:>6.1}  {:>6.1}  {:>6.1}",
            zonal(&annual_g, r) - 273.15,
            zonal(&p.year.t[jan], r) - 273.15,
            zonal(&p.year.t[jul], r) - 273.15
        );
    }
    let mut gmean = 0.0;
    let mut garea = 0.0;
    for (i, a) in annual.iter().enumerate() {
        gmean += a * grid.area[grid.row(i)].to_f64();
        garea += grid.area[grid.row(i)].to_f64();
    }
    println!(
        "  the global annual mean {:.2} °C (the charter's 14.85 °C)",
        gmean / garea - 273.15
    );
    // A continent's swing: central Siberia (62° N, 105° E) and the Southern Ocean beside it in latitude (62° S, 105° E).
    let at = |lat: f64, lon: f64| {
        let r = ((lat + 90.0) / 180.0 * rows as f64) as usize;
        let c = (lon / 360.0 * grid.cols as f64) as usize;
        r * grid.cols + c
    };
    let places: &[(&str, f64, f64)] = if tag == "earth" {
        &[
            ("central Siberia 62° N 105° E", 62.0, 105.0),
            ("the Southern Ocean 62° S 105° E", -62.0, 105.0),
            ("the Sahara 23° N 10° E", 23.0, 10.0),
        ]
    } else {
        &[
            ("the north pole", 89.0, 0.0),
            ("the south pole", -89.0, 0.0),
            ("60° N 0°", 60.0, 0.0),
            ("the equator 0° 0°", 0.0, 0.0),
        ]
    };
    for &(name, lat, lon) in places {
        let i = at(lat, lon);
        println!(
            "  {name}: January {:.1} °C, July {:.1} °C, sea share {:.2}",
            p.year.t[jan][i].to_f64() - 273.15,
            p.year.t[jul][i].to_f64() - 273.15,
            sea[i].to_f64()
        );
    }
    // The energy-flux equator by month against GPCP's zonal rain maximum (the tropics' band, 30° S–30° N).
    let a_olr =
        vd_terrain::seasons::balance::mean_sun(&grid, &sun) * bw.kept - bw.b_w_m2_k * bw.t_mean_k;
    let dt = bw.year_s.to_f64() / steps as f64;
    println!(
        "THE RAIN BELT: month, the energy-flux equator(s) (deg), the zonal moist energy's maximum (deg), GPCP's zonal rain maximum in 30° S–30° N (deg)"
    );
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    for (m, name) in months.iter().enumerate() {
        // The step of the month's middle: the spring equinox (step 0) is 20 March; a month is 64/12 steps.
        let day = 15.0 + 30.44 * m as f64;
        let from_equinox = (day - 79.0).rem_euclid(365.25);
        let st = ((from_equinox / 365.25) * steps as f64) as usize % steps;
        let prev = (st + steps - 1) % steps;
        let mut net_rows = vec![0.0f64; rows];
        let mut h_rows = vec![0.0f64; rows];
        for i in 0..grid.len() {
            let r = grid.row(i);
            let t = p.year.t[st][i];
            let g = surface.capacity[i].to_f64() * (t.to_f64() - p.year.t[prev][i].to_f64()) / dt;
            let olr = a_olr.to_f64() + bw.b_w_m2_k.to_f64() * t.to_f64();
            net_rows[r] += (p.year.kept[st][i].to_f64() - olr - g) * grid.area[r].to_f64();
            h_rows[r] += moist_energy(&bw, t, surface.rh[i]).0.to_f64() / grid.cols as f64;
        }
        let efe = flux_equators(&grid, &net_rows);
        let hmax = (0..rows)
            .max_by(|&a, &b| h_rows[a].total_cmp(&h_rows[b]))
            .expect("rows");
        let hlat = -90.0 + (hmax as f64 + 0.5) * 180.0 / rows as f64;
        let mut best = (f64::MIN, f64::NAN);
        if let Some(gpcp) = gpcp {
            for k in 0..=24 {
                let lat = -30.0 + 2.5 * k as f64;
                let mean: f64 = (0..144)
                    .map(|c| gpcp.at(m, lat, 1.25 + 2.5 * c as f64))
                    .sum::<f64>()
                    / 144.0;
                if mean > best.0 {
                    best = (mean, lat);
                }
            }
        }
        println!(
            "  {name}: {:?}  {hlat:>6.1}  {:>6.1}",
            efe.iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>(),
            best.1
        );
    }
    let read = |t: &[Gf], lat: f64, lon: f64| {
        let d = vd_core::glam::DVec3::new(
            lat.to_radians().cos() * lon.to_radians().cos(),
            lat.to_radians().cos() * lon.to_radians().sin(),
            lat.to_radians().sin(),
        );
        let w = grid.reading([Gf::from_f64(d.x), Gf::from_f64(d.y), Gf::from_f64(d.z)]);
        w.iter()
            .map(|&(c, x)| t[c].to_f64() * x.to_f64())
            .sum::<f64>()
    };
    map_png(
        &out.join(format!("{tag}_t_jan.png")),
        &|lat, lon| read(&p.year.t[jan], lat, lon),
        &temp_colour,
    );
    map_png(
        &out.join(format!("{tag}_t_jul.png")),
        &|lat, lon| read(&p.year.t[jul], lat, lon),
        &temp_colour,
    );
    // THE WINDS of January and July: the zonal profile, the agreement with Earth's measured winds, and the maps.
    let omega = Gf::TWO * Gf::from_f64(std::f64::consts::PI)
        / Gf::from_i64(i64::from(words.day_s.unwrap_or(86_400)));
    let t1 = std::time::Instant::now();
    let departures = std::env::var_os("VD_NO_DEPARTURES").is_none();
    let w_jan = step_winds(&grid, &bw, omega, &p.year, &surface, a_olr, jan, departures);
    let w_jul = step_winds(&grid, &bw, omega, &p.year, &surface, a_olr, jul, departures);
    println!(
        "THE WINDS: two steps in {:.2} s; the rising branch at {:.1}° (January) and {:.1}° (July)",
        t1.elapsed().as_secs_f64(),
        w_jan.rising_sin_lat.to_f64().asin().to_degrees(),
        w_jul.rising_sin_lat.to_f64().asin().to_degrees()
    );
    let ncep = if tag == "earth" {
        let dir = earth_dir();
        Some((
            MonthlyGrid::ncep(dir.join("u10.npy").to_str().expect("a path")),
            MonthlyGrid::ncep(dir.join("v10.npy").to_str().expect("a path")),
        ))
    } else {
        None
    };
    println!("  the zonal-mean east wind, m/s: latitude, January (law / NCEP), July (law / NCEP)");
    for lat in [-70i32, -50, -30, -15, 0, 15, 30, 50, 70] {
        let r = ((f64::from(lat) + 90.0) / 180.0 * rows as f64) as usize;
        let zu = |w: &Winds| {
            (0..grid.cols)
                .map(|c| w.east[r * grid.cols + c].to_f64())
                .sum::<f64>()
                / grid.cols as f64
        };
        let nu = |m: usize| {
            ncep.as_ref().map_or(f64::NAN, |(u, _)| {
                (0..192)
                    .map(|c| u.at(m, f64::from(lat), 1.875 * c as f64))
                    .sum::<f64>()
                    / 192.0
            })
        };
        println!(
            "  {lat:>4}°  {:>6.1} / {:>6.1}   {:>6.1} / {:>6.1}",
            zu(&w_jan),
            nu(0),
            zu(&w_jul),
            nu(6)
        );
    }
    if let Some((nu, nv)) = &ncep {
        for (name, w, m) in [("January", &w_jan, 0usize), ("July", &w_jul, 6)] {
            let (mut agree_land, mut all_land, mut agree_sea, mut all_sea) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..grid.len() {
                let r = grid.row(i);
                let lat = -90.0 + (r as f64 + 0.5) * 180.0 / rows as f64;
                if lat.abs() > 80.0 {
                    continue;
                }
                let lon =
                    (i % grid.cols) as f64 * 360.0 / grid.cols as f64 + 180.0 / grid.cols as f64;
                let (u0, v0) = (nu.at(m, lat, lon), nv.at(m, lat, lon));
                let (u1, v1) = (w.east[i].to_f64(), w.north[i].to_f64());
                let (s0, s1) = ((u0 * u0 + v0 * v0).sqrt(), (u1 * u1 + v1 * v1).sqrt());
                if s0 < 1.0 || s1 < 1e-6 {
                    continue;
                }
                let cosang = (u0 * u1 + v0 * v1) / (s0 * s1);
                let a = grid.area[r].to_f64();
                let ok = if cosang > std::f64::consts::FRAC_1_SQRT_2 {
                    a
                } else {
                    0.0
                };
                if sea[i].to_f64() > 0.5 {
                    agree_sea += ok;
                    all_sea += a;
                } else {
                    agree_land += ok;
                    all_land += a;
                }
            }
            println!(
                "  {name}: the wind's direction within 45° of NCEP's (where NCEP blows over 1 m/s, 80° S–80° N): the sea {:.0} %, the land {:.0} %",
                100.0 * agree_sea / all_sea,
                100.0 * agree_land / all_land
            );
        }
    }
    wind_png(
        &out.join(format!("{tag}_wind_jan.png")),
        &grid,
        &w_jan,
        &|lat, lon| read(&p.year.t[jan], lat, lon),
        &temp_colour,
    );
    wind_png(
        &out.join(format!("{tag}_wind_jul.png")),
        &grid,
        &w_jul,
        &|lat, lon| read(&p.year.t[jul], lat, lon),
        &temp_colour,
    );
    println!("the maps: {}", out.display());
}

/// The bench's switches for S2: the rain-out law, the eddy diffusion everywhere, and Earth's measured winds (NCEP's
/// monthly 10 m winds, the seasons' means) in place of the law's.
#[derive(Clone, Copy, Debug)]
struct Switches {
    rain_law: RainLaw,
    eddy_everywhere: bool,
    measured_winds: bool,
    no_departures: bool,
    mountains: bool,
    gill: bool,
    gpcp_heat: bool,
    gill_departures: bool,
}

impl Switches {
    fn parse(arg: Option<&String>) -> Switches {
        let a = arg.map_or("", String::as_str);
        Switches {
            rain_law: if a.contains("sat") {
                RainLaw::Saturation
            } else {
                RainLaw::Residence
            },
            eddy_everywhere: a.contains("eddy"),
            measured_winds: a.contains("ncep"),
            no_departures: a.contains("nodep"),
            mountains: a.contains("mtn"),
            gill: a.contains("gill"),
            gpcp_heat: a.contains("gpcpq"),
            gill_departures: a.contains("gilldep"),
        }
    }
}

/// ★ THE RAIN OF A BODY (S2): the seasons' chain on the climate grid, two rounds; prints its readings and draws its maps.
fn season_s2(
    out: &Path,
    tag: &str,
    body: &vd_terrain::BodyDefinition,
    fine: &MacroLattice,
    ocean: &[bool],
    height_over_sea_m: &[f64],
    words: &SolveWords,
    rows: usize,
    seasons: usize,
    gpcp: Option<&MonthlyGrid>,
    switches: Switches,
) {
    let t_all = std::time::Instant::now();
    println!("THE SWITCHES: {switches:?}");
    let bw = balance_words(body, words);
    let grid = ClimateGrid::new(Gf::from_f64(body.ladder().radius_m()), rows);
    let n = grid.len();
    let sea = sea_share(fine, &grid, ocean);
    // Each climate cell's mean height of its ground over the sea (sea nodes at nought), by area.
    let mut hsum = vec![0.0f64; n];
    let mut asum = vec![0.0f64; n];
    for (node, &h) in height_over_sea_m.iter().enumerate() {
        let c = grid.cell_at(unit(fine.direction(node as u32)));
        let a = fine.area_m2(node as u32) as f64;
        hsum[c] += a * if ocean[node] { 0.0 } else { h.max(0.0) };
        asum[c] += a;
    }
    let height: Vec<Gf> = hsum
        .iter()
        .zip(&asum)
        .map(|(h, a)| Gf::from_f64(if *a > 0.0 { h / a } else { 0.0 }))
        .collect();
    let capacity: Vec<Gf> = sea
        .iter()
        .map(|&s| capacity_j_m2_k(&bw, s, words.year_s))
        .collect();
    let sun = sun_of(words);
    let steps_shift = 6u32;
    let steps = 1usize << steps_shift;
    let per = steps / seasons;
    let omega = Gf::TWO * Gf::from_f64(std::f64::consts::PI)
        / Gf::from_i64(i64::from(words.day_s.unwrap_or(86_400)));
    let lapse = vd_terrain::climate::lapse_rate_k_m(body.facts().gravity_mm_s2, words.mu_q8);
    let a_olr =
        vd_terrain::seasons::balance::mean_sun(&grid, &sun) * bw.kept - bw.b_w_m2_k * bw.t_mean_k;
    let dt = bw.year_s / Gf::from_i64(steps as i64);
    let season_s = bw.year_s / Gf::from_i64(seasons as i64);
    // THE NODE GRID for the mountains' rain: its geometry, each node's ground over the sea, and each season's climate-cell
    // rain-out rate (the first round's at the sea's level).
    let geo = switches
        .mountains
        .then(|| NodeGeometry::of(fine, &grid, 14));
    let node_h: Vec<Gf> = height_over_sea_m
        .iter()
        .zip(ocean)
        .map(|(h, &wet)| Gf::from_f64(if wet { 0.0 } else { h.max(0.0) }))
        .collect();
    let mut cell_rates: Vec<Vec<Gf>> = vec![vec![Gf::ONE / Gf::from_f64(RESIDENCE_S); n]; seasons];
    let mut node_rain = vec![0.0f64; fine.node_count()];
    let mut wetness = vec![Gf::ONE; n];
    let mut land_evap_seasons: Vec<Vec<Gf>> = vec![vec![Gf::ZERO; n]; seasons];
    // THE HEATED WINDS' start: each season's wind speed from the round before (the drag's warm start).
    let mut speed_seasons: Vec<Vec<Gf>> = vec![Vec::new(); seasons];
    let mut gill_report = (0usize, true, 0usize);
    let mut gill_winds: Vec<(Vec<Gf>, Vec<Gf>)> = vec![(Vec::new(), Vec::new()); seasons];
    let mut start = vec![bw.t_mean_k; n];
    let mut year_rain = vec![0.0f64; n];
    let mut season_rain: Vec<Vec<f64>> = vec![vec![0.0; n]; seasons];
    let rounds_n = if switches.gill { 3 } else { 2 };
    for round in 0..rounds_n {
        let rh: Vec<Gf> = (0..n)
            .map(|i| Gf::from_f64(0.8) * (sea[i] + (Gf::ONE - sea[i]) * wetness[i]))
            .collect();
        let surface = Surface {
            capacity: capacity.clone(),
            rh: rh.clone(),
        };
        let t0 = std::time::Instant::now();
        let p = periodic_year(
            &grid,
            &bw,
            &sun,
            steps_shift,
            &surface,
            &start,
            (Gf::from_f64(0.01), 400),
        );
        let t_ebm = t0.elapsed().as_secs_f64();
        start.clone_from(&p.year.t[steps - 1]);
        let t1 = std::time::Instant::now();
        let step_w: Vec<Winds> = (0..steps)
            .map(|st| {
                step_winds(
                    &grid,
                    &bw,
                    omega,
                    &p.year,
                    &surface,
                    a_olr,
                    st,
                    !switches.no_departures,
                )
            })
            .collect();
        let t_wind = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let mut pet_year = vec![0.0f64; n];
        let mut worst_sweeps = 0;
        let mut all_met = true;
        year_rain.iter_mut().for_each(|x| *x = 0.0);
        let mut sea_evap_total = 0.0;
        for (m, season_rain_m) in season_rain.iter_mut().enumerate() {
            let range = m * per..(m + 1) * per;
            let k = Gf::from_i64(per as i64);
            let mean = |f: &dyn Fn(usize, usize) -> Gf| -> Vec<Gf> {
                (0..n)
                    .map(|i| {
                        range
                            .clone()
                            .map(|st| f(st, i))
                            .fold(Gf::ZERO, |a, b| a + b)
                            / k
                    })
                    .collect()
            };
            let (east, north) = if switches.measured_winds {
                let dir = earth_dir();
                let nu = MonthlyGrid::ncep(dir.join("u10.npy").to_str().expect("a path"));
                let nv = MonthlyGrid::ncep(dir.join("v10.npy").to_str().expect("a path"));
                // The season's months: its steps' share of the year from the spring equinox (20 March), as months.
                let months: Vec<usize> = range
                    .clone()
                    .map(|st| {
                        ((79.0 + (st as f64 + 0.5) / steps as f64 * 365.25) / 30.44) as usize % 12
                    })
                    .collect();
                let at = |g: &MonthlyGrid, i: usize| {
                    let lat = -90.0 + (grid.row(i) as f64 + 0.5) * 180.0 / rows as f64;
                    let lon = ((i % grid.cols) as f64 + 0.5) * 360.0 / grid.cols as f64;
                    months.iter().map(|&m| g.at(m, lat, lon)).sum::<f64>() / months.len() as f64
                };
                (
                    (0..n).map(|i| Gf::from_f64(at(&nu, i))).collect::<Vec<_>>(),
                    (0..n).map(|i| Gf::from_f64(at(&nv, i))).collect::<Vec<_>>(),
                )
            } else if switches.gill {
                let t = mean(&|st, i| p.year.t[st][i]);
                let net = mean(&|st, i| {
                    let prev = (st + steps - 1) % steps;
                    let g = capacity[i] * (p.year.t[st][i] - p.year.t[prev][i]) / dt;
                    p.year.kept[st][i] - (a_olr + bw.b_w_m2_k * p.year.t[st][i]) - g
                });
                // The research's control (`gpcpq`): GPCP's measured rain's latent heat as the only heating, the cooling
                // uniform (under the moist stability since 2026-10-16: a control of the heating's pattern only).
                let net = if switches.gpcp_heat {
                    let g = gpcp.expect("gpcpq is Earth's");
                    let months: Vec<usize> = range
                        .clone()
                        .map(|st| {
                            ((79.0 + (st as f64 + 0.5) / steps as f64 * 365.25) / 30.44) as usize
                                % 12
                        })
                        .collect();
                    let lat_lon = |i: usize| {
                        (
                            -90.0 + (grid.row(i) as f64 + 0.5) * 180.0 / rows as f64,
                            ((i % grid.cols) as f64 + 0.5) * 360.0 / grid.cols as f64,
                        )
                    };
                    let q: Vec<Gf> = (0..n)
                        .map(|i| {
                            let (lat, lon) = lat_lon(i);
                            let mmday = months.iter().map(|&mo| g.at(mo, lat, lon)).sum::<f64>()
                                / months.len() as f64;
                            vd_terrain::seasons::gill::latent_w_m2(Gf::from_f64(mmday / 86_400.0))
                        })
                        .collect();
                    q
                } else {
                    net
                };
                let hw = vd_terrain::seasons::gill::heated_winds(
                    &grid,
                    &bw,
                    omega,
                    vd_terrain::seasons::gill::Heating {
                        t: &t,
                        net: &net,
                        sea: &sea,
                        rh: &rh,
                        departures_only: switches.gill_departures,
                    },
                    &speed_seasons[m],
                );
                if m == 0 || m == seasons / 2 {
                    println!(
                        "  heated winds, season {m}: {} iterations, the residual {:.2e}, the storms' westerlies {}",
                        hw.worst_iterations,
                        hw.residual.to_f64(),
                        hw.storms_ran
                    );
                }
                gill_report.0 = gill_report.0.max(hw.worst_iterations);
                gill_report.1 &= hw.met;
                gill_report.2 += usize::from(hw.storms_ran);
                speed_seasons[m] = (0..n)
                    .map(|i| (hw.east[i] * hw.east[i] + hw.north[i] * hw.north[i]).sqrt())
                    .collect();
                if switches.gill_departures {
                    // The cell law's own zonal winds plus the heating's land–sea pull.
                    let ze = mean(&|st, i| step_w[st].east[i]);
                    let zn = mean(&|st, i| step_w[st].north[i]);
                    (
                        (0..n).map(|i| ze[i] + hw.east[i]).collect(),
                        (0..n).map(|i| zn[i] + hw.north[i]).collect(),
                    )
                } else {
                    (hw.east, hw.north)
                }
            } else {
                (
                    mean(&|st, i| step_w[st].east[i]),
                    mean(&|st, i| step_w[st].north[i]),
                )
            };
            gill_winds[m] = (east.clone(), north.clone());
            let t = mean(&|st, i| p.year.t[st][i]);
            let share = mean(&|st, i| step_w[st].cell_share[grid.row(i)]);
            let kept = mean(&|st, i| p.year.kept[st][i]);
            let g = mean(&|st, i| {
                let prev = (st + steps - 1) % steps;
                capacity[i] * (p.year.t[st][i] - p.year.t[prev][i]) / dt
            });
            let water = season_water(
                &grid,
                &bw,
                Forcing {
                    east: &east,
                    north: &north,
                    t: &t,
                    kept: &kept,
                    g_store: &g,
                    sea: &sea,
                    height_m: &height,
                    land_evap: &land_evap_seasons[m],
                    cell_share: &share,
                    kept_mean: vd_terrain::seasons::balance::mean_sun(&grid, &sun) * bw.kept,
                    rain_law: switches.rain_law,
                    eddy_everywhere: switches.eddy_everywhere,
                    rain_rate: &cell_rates[m],
                    lapse_k_m: lapse,
                },
            );
            worst_sweeps = worst_sweeps.max(water.sweeps);
            if let Some(geo) = &geo {
                let tm = std::time::Instant::now();
                let mr = mountain_rain(
                    fine,
                    geo,
                    &grid,
                    &east,
                    &north,
                    &t,
                    &water.column,
                    &node_h,
                    lapse,
                    14,
                    None,
                );
                println!(
                    "  the mountains' rain: season {m} in {:.1} s, {} rounds (met {}), the most still {} nodes, the largest loop solved {}",
                    tm.elapsed().as_secs_f64(),
                    mr.rounds,
                    mr.met,
                    mr.still,
                    mr.largest_loop
                );
                cell_rates[m].clone_from(&mr.cell_rate);
                if round == rounds_n - 1 {
                    for (acc, r) in node_rain.iter_mut().zip(&mr.rain) {
                        *acc += (*r * season_s).to_f64();
                    }
                }
            }
            if std::env::var_os("VD_CLIMATE_PROBE").is_some() && (m == 0 || m == seasons / 2) {
                for (name, lat, lon) in [
                    ("Amazon", -3.0f64, 300.0f64),
                    ("Atlantic 5N 30W", 5.0, 330.0),
                    ("Sahara", 23.0, 10.0),
                    ("Congo", 0.0, 22.0),
                ] {
                    let r = ((lat + 90.0) / 180.0 * rows as f64) as usize;
                    let c = (lon / 360.0 * grid.cols as f64) as usize;
                    let i = r * grid.cols + c;
                    let rv = 461.5;
                    let tk = t[i].to_f64() - lapse.to_f64() * height[i].to_f64();
                    let h0 = rv * tk * tk / (2.501e6 * lapse.to_f64());
                    let es = vd_terrain::climate::saturation_vapour_pa(Gf::from_f64(tk)).to_f64();
                    let ws = es * h0 / (rv * tk);
                    println!(
                        "  PROBE round {round} season {m} {name}: T {:.1} °C, sea {:.2}, W {:.1} of W_sat {ws:.1} (r {:.2}), rain {:.2} mm/d, sea E {:.2} mm/d, land E {:.2} mm/d, wind ({:.1}, {:.1}), wet {:.2}",
                        tk - 273.15,
                        sea[i].to_f64(),
                        water.column[i].to_f64(),
                        water.column[i].to_f64() / ws,
                        water.rain[i].to_f64() * 86_400.0,
                        water.sea_evap[i].to_f64() * 86_400.0,
                        land_evap_seasons[m][i].to_f64() * 86_400.0,
                        east[i].to_f64(),
                        north[i].to_f64(),
                        wetness[i].to_f64()
                    );
                }
            }
            all_met &= water.met;
            for i in 0..n {
                let mm = (water.rain[i] * season_s).to_f64();
                season_rain_m[i] = mm;
                year_rain[i] += mm;
                pet_year[i] += (water.pet[i] * season_s).to_f64();
                sea_evap_total +=
                    (water.sea_evap[i] * season_s).to_f64() * grid.area[grid.row(i)].to_f64();
            }
            // The next round's land evaporation this season: Turc–Pike on the season's rain and PET.
            for i in 0..n {
                let (pr, pe) = ((water.rain[i]).to_f64(), water.pet[i].to_f64());
                let et = if pe > 0.0 {
                    pr / (1.0 + (pr / pe).powi(2)).sqrt()
                } else {
                    0.0
                };
                land_evap_seasons[m][i] = Gf::from_f64(et);
            }
        }
        let t_water = t2.elapsed().as_secs_f64();
        // The land's wetness for the next round: the year's Turc–Pike evaporation over the PET.
        for i in 0..n {
            let (pr, pe) = (year_rain[i], pet_year[i]);
            wetness[i] = Gf::from_f64(if pe > 0.0 {
                (1.0 / (1.0 + (pr / pe).powi(2)).sqrt()) * pr / pe
            } else {
                1.0
            });
        }
        let total_area: f64 = (0..n).map(|i| grid.area[grid.row(i)].to_f64()).sum();
        let global: f64 = (0..n)
            .map(|i| year_rain[i] * grid.area[grid.row(i)].to_f64())
            .sum::<f64>()
            / total_area;
        println!(
            "ROUND {round}: the balance {:.1} s ({} years), the winds {:.2} s, the water {:.2} s (worst sweeps {worst_sweeps}, all met {all_met}); the global rain {global:.0} mm/yr, the sea's evaporation {:.0} mm/yr over the globe",
            t_ebm,
            p.years,
            t_wind,
            t_water,
            sea_evap_total / total_area
        );
    }
    println!(
        "{tag}: the whole S2 in {:.1} s",
        t_all.elapsed().as_secs_f64()
    );
    if switches.gill {
        println!(
            "THE HEATED WINDS: the solve's worst iterations {}, all met {}, the storms' westerlies ran in {} of {} season-rounds",
            gill_report.0,
            gill_report.1,
            gill_report.2,
            seasons * rounds_n
        );
    }
    if tag == "earth" {
        // The seasons' winds against NCEP's (the season holding mid-January and mid-July).
        let dir = earth_dir();
        let nu = MonthlyGrid::ncep(dir.join("u10.npy").to_str().expect("a path"));
        let nv = MonthlyGrid::ncep(dir.join("v10.npy").to_str().expect("a path"));
        for (name, month, st) in [("January", 0usize, 52usize), ("July", 6, 20)] {
            let (ew, nw) = &gill_winds[(st / per).min(seasons - 1)];
            let (mut agree_land, mut all_land, mut agree_sea, mut all_sea) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..n {
                let r = grid.row(i);
                let lat = -90.0 + (r as f64 + 0.5) * 180.0 / rows as f64;
                if lat.abs() > 80.0 {
                    continue;
                }
                let lon = ((i % grid.cols) as f64 + 0.5) * 360.0 / grid.cols as f64;
                let (u0, v0) = (nu.at(month, lat, lon), nv.at(month, lat, lon));
                let (u1, v1) = (ew[i].to_f64(), nw[i].to_f64());
                let (s0, s1) = ((u0 * u0 + v0 * v0).sqrt(), (u1 * u1 + v1 * v1).sqrt());
                if s0 < 1.0 || s1 < 1e-6 {
                    continue;
                }
                let ok = if (u0 * u1 + v0 * v1) / (s0 * s1) > std::f64::consts::FRAC_1_SQRT_2 {
                    grid.area[r].to_f64()
                } else {
                    0.0
                };
                if sea[i].to_f64() > 0.5 {
                    agree_sea += ok;
                    all_sea += grid.area[r].to_f64();
                } else {
                    agree_land += ok;
                    all_land += grid.area[r].to_f64();
                }
            }
            println!(
                "  {name}'s season: the wind's direction within 45° of NCEP's: the sea {:.0} %, the land {:.0} %",
                100.0 * agree_sea / all_sea,
                100.0 * agree_land / all_land
            );
        }
    }
    // THE READINGS.
    let lat_of_row = |r: usize| -90.0 + (r as f64 + 0.5) * 180.0 / rows as f64;
    let lon_of_col = |c: usize| (c as f64 + 0.5) * 360.0 / grid.cols as f64;
    let area = |i: usize| grid.area[grid.row(i)].to_f64();
    let land = |i: usize| sea[i].to_f64() < 0.5;
    let gp = |i: usize| {
        gpcp.map(|g| {
            (0..12)
                .map(|m| g.at(m, lat_of_row(grid.row(i)), lon_of_col(i % grid.cols)))
                .sum::<f64>()
                / 12.0
                * DAYS
        })
    };
    let mean_over = |f: &dyn Fn(usize) -> f64, pick: &dyn Fn(usize) -> bool| {
        let (mut s, mut a) = (0.0, 0.0);
        for i in 0..n {
            if pick(i) {
                s += f(i) * area(i);
                a += area(i);
            }
        }
        s / a
    };
    println!(
        "THE RAIN, mm/yr: the global mean {:.0}, the land's mean {:.0}",
        mean_over(&|i| year_rain[i], &|_| true),
        mean_over(&|i| year_rain[i], &land)
    );
    let dry = |f: &dyn Fn(usize) -> f64| {
        100.0 * mean_over(&|i| if f(i) < 250.0 { 1.0 } else { 0.0 }, &land)
    };
    println!("  the land under 250 mm: {:.1} %", dry(&|i| year_rain[i]));
    if gpcp.is_some() {
        println!(
            "  GPCP: the global mean {:.0}, the land's mean {:.0}, the land under 250 mm {:.1} %",
            mean_over(&|i| gp(i).unwrap_or(0.0), &|_| true),
            mean_over(&|i| gp(i).unwrap_or(0.0), &land),
            dry(&|i| gp(i).unwrap_or(0.0))
        );
        // The pattern correlation, over all cells and over land, by area.
        let corr = |pick: &dyn Fn(usize) -> bool| {
            let mx = mean_over(&|i| year_rain[i], pick);
            let my = mean_over(&|i| gp(i).unwrap_or(0.0), pick);
            let cov = mean_over(&|i| (year_rain[i] - mx) * (gp(i).unwrap_or(0.0) - my), pick);
            let vx = mean_over(&|i| (year_rain[i] - mx).powi(2), pick);
            let vy = mean_over(&|i| (gp(i).unwrap_or(0.0) - my).powi(2), pick);
            cov / (vx * vy).sqrt()
        };
        println!(
            "  the pattern's correlation with GPCP: all {:.2}, the land {:.2}",
            corr(&|_| true),
            corr(&land)
        );
    }
    println!("  the zonal profile, mm/yr: latitude, the law, GPCP");
    for lat in (-80..=80).step_by(10) {
        let r = ((f64::from(lat) + 90.0) / 180.0 * rows as f64) as usize;
        let z: f64 = (0..grid.cols)
            .map(|c| year_rain[r * grid.cols + c])
            .sum::<f64>()
            / grid.cols as f64;
        let zg: f64 = gpcp.map_or(f64::NAN, |_| {
            (0..grid.cols)
                .map(|c| gp(r * grid.cols + c).unwrap_or(0.0))
                .sum::<f64>()
                / grid.cols as f64
        });
        println!("  {lat:>4}°  {z:>6.0}  {zg:>6.0}");
    }
    let cell = |lat: f64, lon: f64| {
        let r = ((lat + 90.0) / 180.0 * rows as f64) as usize;
        let c = (lon.rem_euclid(360.0) / 360.0 * grid.cols as f64) as usize;
        r * grid.cols + c
    };
    if gpcp.is_some() {
        println!("  named places, mm/yr: the law / GPCP (January season, July season of the law)");
        let (jan_s, jul_s) = ((52 / per).min(seasons - 1), (20 / per).min(seasons - 1));
        for (name, lat, lon) in [
            ("the Amazon 3° S 60° W", -3.0, -60.0),
            ("the Congo 0° 22° E", 0.0, 22.0),
            ("the Sahara 23° N 10° E", 23.0, 10.0),
            ("India's west coast 15° N 74° E", 15.0, 74.0),
            ("the Atacama 23° S 70° W", -23.0, -70.0),
            ("western Europe 48° N 2° E", 48.0, 2.0),
            ("central Siberia 62° N 105° E", 62.0, 105.0),
            ("the Tibetan Plateau 33° N 88° E", 33.0, 88.0),
            ("Australia's interior 25° S 133° E", -25.0, 133.0),
            ("the US Great Plains 40° N 100° W", 40.0, -100.0),
        ] {
            let i = cell(lat, lon);
            println!(
                "  {name}: {:.0} / {:.0}  ({:.0}, {:.0})",
                year_rain[i],
                gp(i).unwrap_or(f64::NAN),
                season_rain[jan_s][i] * seasons as f64,
                season_rain[jul_s][i] * seasons as f64
            );
        }
    }
    // THE MAPS on GPCP's scale, read bilinearly from the climate grid.
    let read = |f: &[f64], lat: f64, lon: f64| {
        let d = vd_core::glam::DVec3::new(
            lat.to_radians().cos() * lon.to_radians().cos(),
            lat.to_radians().cos() * lon.to_radians().sin(),
            lat.to_radians().sin(),
        );
        let w = grid.reading([Gf::from_f64(d.x), Gf::from_f64(d.y), Gf::from_f64(d.z)]);
        w.iter().map(|&(c, x)| f[c] * x.to_f64()).sum::<f64>()
    };
    let (jan_s, jul_s) = ((52 / per).min(seasons - 1), (20 / per).min(seasons - 1));
    let jan: Vec<f64> = season_rain[jan_s]
        .iter()
        .map(|x| x * seasons as f64)
        .collect();
    let jul: Vec<f64> = season_rain[jul_s]
        .iter()
        .map(|x| x * seasons as f64)
        .collect();
    map_png(
        &out.join(format!("{tag}_rain_year.png")),
        &|lat, lon| read(&year_rain, lat, lon),
        &rain_colour,
    );
    map_png(
        &out.join(format!("{tag}_rain_jan.png")),
        &|lat, lon| read(&jan, lat, lon),
        &rain_colour,
    );
    map_png(
        &out.join(format!("{tag}_rain_jul.png")),
        &|lat, lon| read(&jul, lat, lon),
        &rain_colour,
    );
    if switches.mountains {
        let node_at = |lat: f64, lon: f64| {
            let d = vd_core::glam::DVec3::new(
                lat.to_radians().cos() * lon.to_radians().cos(),
                lat.to_radians().cos() * lon.to_radians().sin(),
                lat.to_radians().sin(),
            );
            vd_bins::drainage_gates::node_under(fine, d) as usize
        };
        map_png(
            &out.join(format!("{tag}_rain_nodes_year.png")),
            &|lat, lon| node_rain[node_at(lat, lon)],
            &rain_colour,
        );
        // The land's readings on the nodes.
        let (mut land_n, mut dry) = (0usize, 0usize);
        let mut lr: Vec<f64> = Vec::new();
        for (i, &wet) in ocean.iter().enumerate() {
            if !wet {
                land_n += 1;
                if node_rain[i] < 250.0 {
                    dry += 1;
                }
                lr.push(node_rain[i]);
            }
        }
        lr.sort_by(f64::total_cmp);
        println!(
            "THE NODES' RAIN: the land's median {:.0} mm/yr, its 10th and 90th percentiles {:.0} and {:.0}, the land under 250 mm {:.1} % (by nodes)",
            lr[lr.len() / 2],
            lr[lr.len() / 10],
            lr[lr.len() * 9 / 10],
            100.0 * dry as f64 / land_n as f64
        );
        if gpcp.is_some() {
            println!("  named places on the nodes, mm/yr: the law / GPCP");
            for (name, lat, lon) in [
                ("the Amazon 3° S 60° W", -3.0, -60.0),
                ("the Sahara 23° N 10° E", 23.0, 10.0),
                ("India's Western Ghats 15° N 74° E", 15.0, 74.0),
                ("the Deccan behind them 17° N 77° E", 17.0, 77.0),
                ("the Himalaya's south face 28° N 85° E", 28.0, 85.0),
                ("the Tibetan Plateau 33° N 88° E", 33.0, 88.0),
                ("the Andes' east face 13° S 72° W", -13.0, -72.0),
                ("the Atacama 23° S 70° W", -23.0, -70.0),
                ("Norway's west coast 61° N 6° E", 61.0, 6.0),
                ("Sweden behind it 61° N 15° E", 61.0, 15.0),
                ("the Cascades' west 47° N 122° W", 47.0, -122.0),
                ("the Columbia basin behind 47° N 119° W", 47.0, -119.0),
            ] {
                let i = node_at(lat, lon);
                println!(
                    "  {name}: {:.0} / {:.0}",
                    node_rain[i],
                    gp(grid.cell_at([
                        Gf::from_f64(lat.to_radians().cos() * lon.to_radians().cos()),
                        Gf::from_f64(lat.to_radians().cos() * lon.to_radians().sin()),
                        Gf::from_f64(lat.to_radians().sin())
                    ]))
                    .unwrap_or(f64::NAN)
                );
            }
        }
    }
    println!("the maps: {}", out.display());
}

fn earth_s2(out: &Path, rows: usize, seasons: usize, switches: Switches) {
    let dir = earth_dir();
    let body = home_planet();
    let fine = MacroLattice::of(&body).expect("the home planet's lattice");
    let n = fine.node_count();
    let bytes =
        std::fs::read(out.join("earth_z.bin")).expect("run earth-data first: DIR/earth_z.bin");
    let z: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(z.len(), n);
    let area: Vec<u64> = (0..n as u32).map(|k| fine.area_m2(k)).collect();
    let z16: Vec<i32> = z
        .iter()
        .map(|&m| m * vd_terrain::solve::Z_STEPS_PER_M)
        .collect();
    let ocean = vd_terrain::land::ocean_mask(&fine, &z16, &area, 0).ocean;
    let heights: Vec<f64> = z.iter().map(|&m| f64::from(m)).collect();
    let gpcp = MonthlyGrid::gpcp(dir.join("gpcp.npy").to_str().expect("a path"));
    season_s2(
        out,
        "earth",
        &body,
        &fine,
        &ocean,
        &heights,
        &earth_words(),
        rows,
        seasons,
        Some(&gpcp),
        switches,
    );
}

fn home_s2(out: &Path, art: &str, rows: usize, seasons: usize, switches: Switches) {
    std::fs::create_dir_all(out).expect("the output folder");
    let body = home_planet();
    let fine = MacroLattice::of(&body).expect("the home planet's lattice");
    let artifact = vd_bins::artifact_file::read(art).expect("the world reads");
    let ocean: Vec<bool> = artifact
        .rows
        .iter()
        .map(|r| {
            (r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
                & vd_terrain::solve::FACIES_SEA
                != 0
        })
        .collect();
    let sea_m = f64::from(artifact.sea_m);
    let heights: Vec<f64> = artifact
        .rows
        .iter()
        .map(|r| f64::from(r.z_m) - sea_m)
        .collect();
    // TODAY'S RAIN (generator 26's stored rows, each class at its middle) as the same map, read at the node under each
    // pixel, and its land's readings beside the new law's.
    let today: Vec<f64> = artifact
        .rows
        .iter()
        .map(|r| {
            let lo = vd_terrain::fine::l1::decode_class(r.rain, 4) as f64;
            let hi = vd_terrain::fine::l1::decode_class(r.rain.saturating_add(1), 4) as f64;
            0.5 * (lo + hi)
        })
        .collect();
    let node_at = |lat: f64, lon: f64| {
        let d = vd_core::glam::DVec3::new(
            lat.to_radians().cos() * lon.to_radians().cos(),
            lat.to_radians().cos() * lon.to_radians().sin(),
            lat.to_radians().sin(),
        );
        vd_bins::drainage_gates::node_under(&fine, d) as usize
    };
    map_png(
        &out.join("home_rain_today.png"),
        &|lat, lon| {
            let i = node_at(lat, lon);
            if ocean[i] { f64::NAN } else { today[i] }
        },
        &rain_colour,
    );
    let (mut land_n, mut dry) = (0usize, 0usize);
    let mut land_rain: Vec<f64> = Vec::new();
    for (i, &wet) in ocean.iter().enumerate() {
        if !wet {
            land_n += 1;
            if today[i] < 250.0 {
                dry += 1;
            }
            land_rain.push(today[i]);
        }
    }
    land_rain.sort_by(f64::total_cmp);
    println!(
        "TODAY'S RAIN (generator 26): the land's median {:.0} mm/yr, the land under 250 mm {:.1} % (by nodes)",
        land_rain[land_rain.len() / 2],
        100.0 * dry as f64 / land_n as f64
    );
    season_s2(
        out,
        "home",
        &body,
        &fine,
        &ocean,
        &heights,
        &vd_terrain::home::home_solve_words(),
        rows,
        seasons,
        None,
        switches,
    );
}

/// Writes a NumPy `.npy` array of little-endian f32 with the given 2-D shape.
fn write_npy_f32(path: &Path, rows: usize, cols: usize, data: &[f32]) {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut b = Vec::with_capacity(10 + header.len() + data.len() * 4);
    b.extend_from_slice(b"\x93NUMPY\x01\x00");
    b.extend_from_slice(&(header.len() as u16).to_le_bytes());
    b.extend_from_slice(header.as_bytes());
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).expect("the array saves");
}

fn export_latlon(art: &str, out: &Path, nlat: usize, nlon: usize) {
    std::fs::create_dir_all(out).expect("the output folder");
    let body = home_planet();
    let fine = MacroLattice::of(&body).expect("the home planet's lattice");
    let artifact = vd_bins::artifact_file::read(art).expect("the world reads");
    let sea_m = f64::from(artifact.sea_m);
    let lats = vd_bins::earth::gaussian_latitudes(nlat);
    // The edges between the Gaussian rows: midpoints, the poles at the ends.
    let mut edges = vec![-90.0];
    for k in 0..nlat - 1 {
        edges.push(0.5 * (lats[k] + lats[k + 1]));
    }
    edges.push(90.0);
    let mut area = vec![0.0f64; nlat * nlon];
    let mut land = vec![0.0f64; nlat * nlon];
    let mut height = vec![0.0f64; nlat * nlon];
    for (i, r) in artifact.rows.iter().enumerate() {
        let (lat, lon) = lat_lon_of(dir_of(&fine, i as u32));
        let row = edges.partition_point(|&e| e <= lat).clamp(1, nlat) - 1;
        let col = ((lon / 360.0 * nlon as f64) as usize).min(nlon - 1);
        let c = row * nlon + col;
        let a = fine.area_m2(i as u32) as f64;
        let wet = (r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
            & vd_terrain::solve::FACIES_SEA
            != 0;
        area[c] += a;
        if !wet {
            land[c] += a;
            height[c] += a * (f64::from(r.z_m) - sea_m).max(0.0);
        }
    }
    let share: Vec<f32> = (0..nlat * nlon)
        .map(|c| (land[c] / area[c]) as f32)
        .collect();
    let h: Vec<f32> = (0..nlat * nlon)
        .map(|c| (height[c] / area[c]) as f32)
        .collect();
    write_npy_f32(&out.join("land_share.npy"), nlat, nlon, &share);
    write_npy_f32(&out.join("height_m.npy"), nlat, nlon, &h);
    std::fs::write(
        out.join("lats.txt"),
        lats.iter()
            .map(|l| format!("{l:.6}"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .expect("the latitudes save");
    let land_total: f64 = land.iter().sum::<f64>() / area.iter().sum::<f64>();
    println!(
        "exported {nlat} × {nlon}: the land {:.2} % of the surface; the sea level {sea_m} m",
        100.0 * land_total
    );
}

/// ★ THE CHAIN ON A FIXED RELIEF (2026-10-16): the seasons' chain from a cold state on a stored world's ground, its
/// spin-up's readings call by call — whether the climate settles on ground that does not move.
fn chain_fixed(art: &str, threads: usize) {
    let body = home_planet();
    let fine = MacroLattice::of(&body).expect("the home planet's lattice");
    let a = vd_bins::artifact_file::read(art).expect("the world reads");
    let steps = vd_terrain::solve::Z_STEPS_PER_M;
    let z: Vec<i32> = a.rows.iter().map(|r| i32::from(r.z_m) * steps).collect();
    let ocean: Vec<bool> = a
        .rows
        .iter()
        .map(|r| {
            (r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
                & vd_terrain::solve::FACIES_SEA
                != 0
        })
        .collect();
    let sea = vd_terrain::climate::SeaView {
        level: i32::from(a.sea_m) * steps,
        ocean: &ocean,
    };
    let words = vd_terrain::home::home_solve_words();
    let mut state = vd_terrain::seasons::chain::SeasonState::new(&fine, threads);
    let clock = std::time::Instant::now();
    let c = vd_terrain::seasons::chain::seasonal_climate(
        &body,
        &fine,
        &words,
        &z,
        Some(sea),
        threads,
        &mut state,
    );
    println!(
        "the spin-up: {:?} in {:.0} s",
        state.last_spin,
        clock.elapsed().as_secs_f64()
    );
    for (k, (worst, over, land)) in state.spin_trace.iter().enumerate() {
        println!(
            "  call {:2}: the largest change {:.2} × its allowance, {} cells over it; the land's mean rain {:.0} mm",
            k + 1,
            worst.to_f64(),
            over,
            land.to_f64()
        );
    }
    let mut land: Vec<u32> = (0..c.rain_mm_yr.len())
        .filter(|&i| !ocean[i])
        .map(|i| c.rain_mm_yr[i])
        .collect();
    land.sort_unstable();
    println!(
        "the land's rain median {} mm, under 250 mm {:.1} %; the drag's rounds {} met {}; the mountains' rounds {}",
        land[land.len() / 2],
        100.0 * land.iter().filter(|&&r| r < 250).count() as f64 / land.len() as f64,
        state.last_drag,
        state.last_drag_met,
        state.last_rounds
    );
}

/// Pictures of one width stacked top to bottom, a dark band between them (for the owner's side-by-side look).
fn stack(out: &str, inputs: &[String]) {
    let imgs: Vec<image::RgbImage> = inputs
        .iter()
        .map(|p| image::open(p).expect("a picture").to_rgb8())
        .collect();
    let w = imgs
        .iter()
        .map(image::RgbImage::width)
        .max()
        .expect("pictures");
    let band = 12;
    let h: u32 =
        imgs.iter().map(image::RgbImage::height).sum::<u32>() + band * (imgs.len() as u32 - 1);
    let mut canvas = image::RgbImage::from_pixel(w, h, image::Rgb([20, 20, 24]));
    let mut y = 0;
    for img in &imgs {
        image::imageops::overlay(&mut canvas, img, 0, i64::from(y));
        y += img.height() + band;
    }
    canvas.save(out).expect("saves");
}

fn art_maps(art: &str, out: &Path, tag: &str, window: Option<[f64; 4]>) {
    let [lat0, lat1, lon0, lon1] = window.unwrap_or([-90.0, 90.0, 0.0, 360.0]);
    std::fs::create_dir_all(out).expect("the output folder");
    let body = home_planet();
    let fine = MacroLattice::of(&body).expect("the home planet's lattice");
    let a = vd_bins::artifact_file::read(art).expect("the world reads");
    let sea_m = f64::from(a.sea_m);
    let wet = |i: usize| {
        (a.rows[i].receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
            & vd_terrain::solve::FACIES_SEA
            != 0
    };
    let node_at = |lat: f64, lon: f64| {
        let d = vd_core::glam::DVec3::new(
            lat.to_radians().cos() * lon.to_radians().cos(),
            lat.to_radians().cos() * lon.to_radians().sin(),
            lat.to_radians().sin(),
        );
        vd_bins::drainage_gates::node_under(&fine, d) as usize
    };
    let h = ((f64::from(MAP_W) * (lat1 - lat0) / (lon1 - lon0)).round() as u32).max(1);
    let (mut relief, mut rain, mut rivers) = (
        image::RgbImage::new(MAP_W, h),
        image::RgbImage::new(MAP_W, h),
        image::RgbImage::new(MAP_W, h),
    );
    let height = |lat: f64, lon: f64| f64::from(a.rows[node_at(lat, lon)].z_m) - sea_m;
    let step = (lon1 - lon0) / f64::from(MAP_W);
    let km_per_px = 2.0 * std::f64::consts::PI * body.ladder().radius_m() / 1_000.0 * step / 360.0;
    for y in 0..h {
        let lat = lat1 - (f64::from(y) + 0.5) * (lat1 - lat0) / f64::from(h);
        let dlon = (lat.to_radians().cos()).max(0.05);
        for x in 0..MAP_W {
            let lon = lon0 + (f64::from(x) + 0.5) * step;
            let i = node_at(lat, lon);
            let hm = f64::from(a.rows[i].z_m) - sea_m;
            let dzx = (height(lat, lon + step) - height(lat, lon - step))
                / (2.0 * km_per_px * 1_000.0 * dlon);
            let dzy =
                (height(lat + step, lon) - height(lat - step, lon)) / (2.0 * km_per_px * 1_000.0);
            // A light from the north-west, 45° up; the slope exaggerated ten times on the globe (the 8 km ground reads), three
            // times in a window.
            let ex = if window.is_some() { 3.0 } else { 10.0 };
            let (nx, ny, nz) = (-dzx * ex, -dzy * ex, 1.0);
            let l = (nx * nx + ny * ny + nz * nz).sqrt();
            let shade = ((-nx * 0.5 + ny * 0.5 + nz * 0.707) / l).clamp(0.15, 1.0);
            let base: [f64; 3] = if wet(i) {
                [40.0, 60.0, 90.0]
            } else if hm < 300.0 {
                [110.0, 140.0, 90.0]
            } else if hm < 1_000.0 {
                [150.0, 150.0, 100.0]
            } else if hm < 2_500.0 {
                [150.0, 120.0, 90.0]
            } else {
                [200.0, 195.0, 190.0]
            };
            let c = base.map(|v| (v * if wet(i) { 1.0 } else { shade }).round() as u8);
            relief.put_pixel(x, y, image::Rgb(c));
            let r = &a.rows[i];
            let lo = vd_terrain::fine::l1::decode_class(r.rain, 4) as f64;
            let hi = vd_terrain::fine::l1::decode_class(r.rain.saturating_add(1), 4) as f64;
            rain.put_pixel(
                x,
                y,
                image::Rgb(rain_colour(if wet(i) { f64::NAN } else { 0.5 * (lo + hi) })),
            );
            let q = f64::from(r.discharge_log);
            let v = if wet(i) {
                20.0
            } else {
                ((q - 60.0) * 2.0).clamp(0.0, 255.0)
            };
            rivers.put_pixel(
                x,
                y,
                image::Rgb([(v * 0.3) as u8, (v * 0.6) as u8, v as u8]),
            );
        }
    }
    relief
        .save(out.join(format!("{tag}_relief.png")))
        .expect("saves");
    rain.save(out.join(format!("{tag}_rain.png")))
        .expect("saves");
    rivers
        .save(out.join(format!("{tag}_rivers.png")))
        .expect("saves");
    let mut land_rain: Vec<f64> = (0..a.rows.len())
        .filter(|&i| !wet(i))
        .map(|i| {
            let r = &a.rows[i];
            0.5 * (vd_terrain::fine::l1::decode_class(r.rain, 4) as f64
                + vd_terrain::fine::l1::decode_class(r.rain.saturating_add(1), 4) as f64)
        })
        .collect();
    land_rain.sort_by(f64::total_cmp);
    let dry = land_rain.iter().filter(|&&p| p < 250.0).count();
    println!(
        "{tag}: the land {:.2} % of the nodes; its rain median {:.0} mm/yr (10th {:.0}, 90th {:.0}); under 250 mm {:.1} %; the sea level {sea_m} m",
        100.0 * land_rain.len() as f64 / a.rows.len() as f64,
        land_rain[land_rain.len() / 2],
        land_rain[land_rain.len() / 10],
        land_rain[land_rain.len() * 9 / 10],
        100.0 * dry as f64 / land_rain.len() as f64
    );
    // The land's heights over the sea: the median, the 90th and the top, and the shares over 1, 2 and 3 km.
    let mut land_h: Vec<f64> = (0..a.rows.len())
        .filter(|&i| !wet(i))
        .map(|i| f64::from(a.rows[i].z_m) - sea_m)
        .collect();
    land_h.sort_by(f64::total_cmp);
    let over =
        |m: f64| 100.0 * land_h.iter().filter(|&&v| v > m).count() as f64 / land_h.len() as f64;
    println!(
        "{tag}: the land's height median {:.0} m, 90th {:.0} m, top {:.0} m; over 1 / 2 / 3 km {:.1} / {:.1} / {:.1} %",
        land_h[land_h.len() / 2],
        land_h[land_h.len() * 9 / 10],
        land_h[land_h.len() - 1],
        over(1_000.0),
        over(2_000.0),
        over(3_000.0)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("earth-data") => earth_data(Path::new(args.get(1).expect("an output folder"))),
        Some("home-s1") => home_s1(
            Path::new(args.get(1).expect("an output folder")),
            args.get(2).expect("the home planet's FILE.art"),
            args.get(3).map_or(90, |r| r.parse().expect("ROWS")),
        ),
        Some("earth-s2") => earth_s2(
            Path::new(args.get(1).expect("an output folder")),
            args.get(2).map_or(90, |r| r.parse().expect("ROWS")),
            args.get(3).map_or(8, |r| r.parse().expect("SEASONS")),
            Switches::parse(args.get(4)),
        ),
        Some("home-s2") => home_s2(
            Path::new(args.get(1).expect("an output folder")),
            args.get(2).expect("the home planet's FILE.art"),
            args.get(3).map_or(90, |r| r.parse().expect("ROWS")),
            args.get(4).map_or(8, |r| r.parse().expect("SEASONS")),
            Switches::parse(args.get(5)),
        ),
        Some("stack") => stack(args.get(1).expect("OUT.png"), &args[2..]),
        Some("chain-fixed") => chain_fixed(
            args.get(1).expect("FILE.art"),
            args.get(2).map_or(14, |t| t.parse().expect("threads")),
        ),
        Some("art-maps") => art_maps(
            args.get(1).expect("FILE.art"),
            Path::new(args.get(2).expect("an output folder")),
            args.get(3).map_or("world", String::as_str),
            (args.len() >= 8).then(|| [4, 5, 6, 7].map(|k| args[k].parse().expect("a degree"))),
        ),
        Some("export-latlon") => export_latlon(
            args.get(1).expect("FILE.art"),
            Path::new(args.get(2).expect("an output folder")),
            args.get(3).map_or(32, |r| r.parse().expect("NLAT")),
            args.get(4).map_or(64, |r| r.parse().expect("NLON")),
        ),
        Some("earth-s1") => earth_s1(
            Path::new(args.get(1).expect("an output folder")),
            args.get(2).map_or(90, |r| r.parse().expect("ROWS")),
        ),
        _ => eprintln!("modes: earth-data DIR | earth-s1 DIR [ROWS] | home-s1 DIR FILE.art [ROWS]"),
    }
}
