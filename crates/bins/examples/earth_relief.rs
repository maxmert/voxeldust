//! ★ EARTH'S RELIEF ON THE HOME PLANET'S OWN GRID — phase 0 of the ground step, the first reading of bench M0-4
//! (ruling B36 item 7, the owner's Q2: Earth's real data as the measure; the owner's "Earth first", 2026-10-02, after
//! the stands census found no hills class and mountains on 1.8 % of the home planet's land).
//!
//! **WHAT IT DOES.**
//! 1. **EARTH'S HEIGHTS.** It reads a raw grid made from NOAA NCEI's ETOPO 2022 global relief model at 60 arc-seconds,
//!    the SURFACE version: the top of the ice sheets, heights over the EGM2008 geoid (NOAA National Centers for
//!    Environmental Information 2022, ETOPO 2022 Global Relief Model, DOI 10.25921/fd45-gt74; no copyright
//!    restriction). The data stays OUTSIDE the repository and only numbers are recorded (ruling B36 item 7). The raw
//!    grid: 21 600 × 10 800 little-endian 16-bit whole metres, row 0 at the north edge, column 0 at 180° W, each cell's
//!    centre half a cell in (the GeoTIFF's "pixel is area"); made from `ETOPO_2022_v1_60s_N90W180_surface.tif` by
//!    undoing its DEFLATE and its floating-point predictor and rounding each height to the metre.
//! 2. **ONE GRID.** It averages Earth's heights onto the HOME PLANET'S OWN macro lattice: each node is the mean of the
//!    cells whose centres fall in it, by each cell's area (the cosine of its latitude). The home planet's look radius
//!    is Earth's mean radius (6 371 km) and its lattice's own radius is 6 341.7 km (MEASURED: 8 192 m a node at a face's
//!    centre × 1 216 nodes × 2/π), so a node is the same 8.2 km on both bodies. Every chord and every area is the
//!    lattice's own, 0.46 % short of Earth's: a slope reads 0.46 % steep, and a share, a ratio, does not move.
//! 3. **ONE SEA RULE.** Earth's sea is the home solve's own rule (`vd_terrain::land::ocean_mask`, ruling B7 item 1):
//!    the largest body of nodes at or under the sea level, joined through the lattice's eight neighbours. A hollow
//!    under the sea level inside the land is land: a closed basin. A strait narrower than a node joins nothing, on
//!    either body.
//! 4. **ONE RELIEF RULE.** Both bodies are read by the stands' own rule (`vd_bins::stands::relief_readings`). The
//!    closed-basin floor counts with the plains on both bodies: the home planet's receiver tree has no counterpart on
//!    Earth's data here.
//! 5. **THE READINGS**, for Earth's land, for Earth's land without Antarctica (south of 60° S; Greenland's ice, about
//!    1.5 % of the land, stays in), and for the home planet's land (its own solve, run in this process by the shard's
//!    own function):
//!    - each relief class's share of the land's area;
//!    - the local relief's spread (the window's highest less its lowest height), by area;
//!    - THE VARIETY: how many relief classes stand within 60 km (by the summed chord) of each land node of the 32 km
//!      grid — G-VARIETY's reading with the relief alone (the design §2.1); the sea floor counts as a class;
//!    - THE HYPSOMETRY: the land's share over 1 000 m, the land's mean height over the sea, and the ocean's mean
//!      depth (G-HYPSO).
//! 6. **THE CONTROLS** (each can fail): every node's own direction finds the node again (the binning's lookup); named
//!    places print their node's height and side; the land's share at the node scale prints beside the share of the
//!    cells at or over the sea level.
//!
//! `cargo run --release -p vd-bins --example earth_relief -- <the raw grid>`
//!
//! **Example.** A node of the Tibetan Plateau and a node of the home planet's crystalline basement are both read by the
//! same window over nodes of 8.2 km, so "tablelands" on Earth means what "tablelands" means on the home planet, and a
//! difference between the two columns is a difference between the two grounds, never between two instruments.

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_bins::drainage_gates::node_under;
use vd_bins::stands::{
    HIGH_M, LAND_CLASSES, LandReading, PERCENTILES, PLAINS, RELIEF, ReliefReading, SEA_FLOOR,
    VARIETY_GRID, VARIETY_REACH_M, ocean_mean_m, read_land, relief_readings,
};
use vd_core::glam::DVec3;
use vd_terrain::artifact::FACIES_SHIFT;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::solve::FACIES_SEA;

/// The raw grid's columns and rows (60 arc-seconds over the globe), and its cells to a degree.
const COLUMNS: usize = 21_600;
const ROWS: usize = 10_800;
const CELLS_PER_DEGREE: f64 = 60.0;
/// Earth's sea level in the grid's own heights: the EGM2008 geoid.
const EARTH_SEA_M: i32 = 0;
/// South of this latitude Earth's land is Antarctica's ice sheet and its fringing islands.
const ANTARCTIC_LAT_DEG: f64 = -60.0;
/// The binning's threads: a fixed count, so the sums add in one order and the means print alike on every machine.
const BIN_THREADS: usize = 8;

/// A node's unit direction (the lattice's own bend of the node's centre).
fn dir_of(lattice: &MacroLattice, node: u32) -> DVec3 {
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let d = lattice.direction(node);
    DVec3::new(
        d[0].raw() as f64 / unit,
        d[1].raw() as f64 / unit,
        d[2].raw() as f64 / unit,
    )
    .normalize()
}

/// The unit direction of a latitude and a longitude in degrees, in the binning's frame: z to the north pole, x to the
/// prime meridian.
fn dir_at(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
    DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
}

/// The latitude of a unit direction, degrees.
fn lat_of(d: DVec3) -> f64 {
    d.z.clamp(-1.0, 1.0).asin().to_degrees()
}

/// Reads the raw grid.
fn read_grid(path: &str) -> Vec<i16> {
    let bytes = std::fs::read(path).expect("the raw grid reads");
    assert_eq!(
        bytes.len(),
        COLUMNS * ROWS * 2,
        "the raw grid holds 21 600 × 10 800 16-bit words"
    );
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// Each node's height: the mean of the grid's cells whose centres fall in it, by each cell's area.
fn bin(lattice: &MacroLattice, grid: &[i16]) -> Vec<i32> {
    let n = lattice.node_count();
    let lon: Vec<(f64, f64)> = (0..COLUMNS)
        .map(|c| {
            let l = (-180.0 + (c as f64 + 0.5) / CELLS_PER_DEGREE).to_radians();
            (l.cos(), l.sin())
        })
        .collect();
    let rows_each = ROWS.div_ceil(BIN_THREADS);
    let parts: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..BIN_THREADS)
            .map(|t| {
                let lon = &lon;
                scope.spawn(move || {
                    let mut weight = vec![0.0f64; n];
                    let mut weighted = vec![0.0f64; n];
                    for r in (t * rows_each)..((t + 1) * rows_each).min(ROWS) {
                        let lat = (90.0 - (r as f64 + 0.5) / CELLS_PER_DEGREE).to_radians();
                        let (cos_lat, sin_lat) = (lat.cos(), lat.sin());
                        for (c, &(cos_lon, sin_lon)) in lon.iter().enumerate() {
                            let dir = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
                            let node = node_under(lattice, dir) as usize;
                            weight[node] += cos_lat;
                            weighted[node] += cos_lat * f64::from(grid[r * COLUMNS + c]);
                        }
                    }
                    (weight, weighted)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a binning thread ends"))
            .collect()
    });
    let mut weight = vec![0.0f64; n];
    let mut weighted = vec![0.0f64; n];
    for (w, wz) in &parts {
        for (sum, x) in weight.iter_mut().zip(w) {
            *sum += x;
        }
        for (sum, x) in weighted.iter_mut().zip(wz) {
            *sum += x;
        }
    }
    let empty = weight.iter().filter(|&&w| w == 0.0).count();
    assert_eq!(empty, 0, "every node holds at least one cell's centre");
    weighted
        .iter()
        .zip(&weight)
        .map(|(wz, w)| (wz / w).round() as i32)
        .collect()
}

/// Prints one table of three columns; `full` adds the variety and the closed basins' rows.
fn table(cols: &[&LandReading; 3], node_km: f64, full: bool) {
    let row = |label: &str, f: &dyn Fn(&LandReading) -> String| {
        println!(
            "  {label:<44} {:>12} {:>12} {:>12}",
            f(cols[0]),
            f(cols[1]),
            f(cols[2])
        );
    };
    row("the land's area, million km²", &|r| {
        format!("{:.2}", r.land_km2 / 1e6)
    });
    row("the land's share of the surface counted, %", &|r| {
        format!("{:.2}", r.land_pct)
    });
    println!("THE RELIEF CLASSES, % of the land's area:");
    for (k, &c) in LAND_CLASSES.iter().enumerate() {
        let label = if c == PLAINS {
            "plains (with the closed-basin floors)".to_string()
        } else {
            RELIEF[usize::from(c)].to_string()
        };
        row(&label, &|r| format!("{:.2}", r.class_pct[k]));
    }
    row("a window half or more gentle (under 8 %), %", &|r| {
        format!("{:.2}", r.gentle_pct)
    });
    if full {
        row("the land under the sea level (closed basins), %", &|r| {
            format!("{:.2}", r.basin_pct)
        });
        row("mountains at a closed basin's wall, % of land", &|r| {
            format!("{:.2}", r.wall_mountain_pct)
        });
    }
    println!(
        "THE LOCAL RELIEF (the window's highest less its lowest), metres, by the land's area:"
    );
    for (k, p) in PERCENTILES.iter().enumerate() {
        row(&format!("{:.1} % of the land under", p * 100.0), &|r| {
            format!("{}", r.span_m[k])
        });
    }
    if full {
        println!(
            "THE VARIETY: relief classes within {} km of each land node of the {} km grid (the sea floor counts):",
            VARIETY_REACH_M / 1_000,
            f64::from(VARIETY_GRID) * node_km
        );
        row("the mean count", &|r| format!("{:.2}", r.variety_mean));
        for k in 0..5 {
            row(
                &format!(
                    "{} class{}, % of the nodes",
                    k + 1,
                    if k == 0 { "" } else { "es" }
                ),
                &|r| format!("{:.2}", r.variety_pct[k]),
            );
        }
    }
    println!("THE HYPSOMETRY (heights over the sea):");
    row(&format!("the land's share over {HIGH_M} m, %"), &|r| {
        format!("{:.2}", r.high_pct)
    });
    row("the land's mean height, m", &|r| {
        format!("{:.0}", r.land_mean_m)
    });
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("the raw grid's path: earth_relief <etopo2022_60s_i16le.raw>");
    println!(
        "EARTH'S RELIEF ON THE HOME PLANET'S OWN GRID (phase 0, bench M0-4), generator {}",
        vd_terrain::GENERATOR_VERSION
    );
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
    let n = lattice.node_count();
    let area: Vec<u64> = (0..n as u32).map(|v| lattice.area_m2(v)).collect();
    println!(
        "the grid: {n} nodes, {} an edge, {:.0} m at a face's centre",
        lattice.edge,
        lattice.node_m()
    );

    // Control 1: every node's own direction finds the node again.
    let lost = (0..n as u32)
        .filter(|&v| node_under(&lattice, dir_of(&lattice, v)) != v)
        .count();
    println!(
        "CONTROL the lookup: {lost} of {n} nodes do not find themselves from their own direction"
    );
    assert_eq!(lost, 0, "the binning's lookup finds every node");

    // Earth.
    let started = std::time::Instant::now();
    let grid = read_grid(&path);
    let cells_land: f64 = {
        let (mut w_all, mut w_land) = (0.0f64, 0.0f64);
        for r in 0..ROWS {
            let w = (90.0 - (r as f64 + 0.5) / CELLS_PER_DEGREE)
                .to_radians()
                .cos();
            let row = &grid[r * COLUMNS..(r + 1) * COLUMNS];
            w_all += w * COLUMNS as f64;
            w_land += w * row.iter().filter(|&&h| i32::from(h) >= EARTH_SEA_M).count() as f64;
        }
        w_land * 100.0 / w_all
    };
    let z_earth = bin(&lattice, &grid);
    drop(grid);
    println!(
        "Earth binned in {:.1} s ({BIN_THREADS} threads)",
        started.elapsed().as_secs_f64()
    );
    let mask = vd_terrain::land::ocean_mask(&lattice, &z_earth, &area, EARTH_SEA_M);
    let land_earth: Vec<bool> = mask.ocean.iter().map(|&o| !o).collect();
    let over_earth: Vec<i32> = z_earth.iter().map(|&z| z - EARTH_SEA_M).collect();
    let land_area: u128 = (0..n)
        .filter(|&v| land_earth[v])
        .map(|v| u128::from(area[v]))
        .sum();
    let all_area: u128 = area.iter().map(|&a| u128::from(a)).sum();
    println!(
        "CONTROL the land's share: {:.2} % of the surface at the node scale (the ocean rule; {} closed basins under the \
         sea level, {} nodes, count as land) against {cells_land:.2} % of the 1.85 km cells at or over the sea level",
        land_area as f64 * 100.0 / all_area as f64,
        mask.pockets,
        mask.pocket_nodes
    );
    println!("CONTROL named places (the node's mean height; its side by the ocean rule):");
    for (name, lat, lon) in [
        ("Everest", 27.9881, 86.9250),
        ("the Tibetan Plateau", 33.0, 88.0),
        ("the Amazon basin", -3.0, -60.0),
        ("Death Valley", 36.25, -116.83),
        ("the Dead Sea", 31.5, 35.5),
        ("the South Caspian", 39.5, 51.0),
        ("Lake Baikal", 53.5, 108.2),
        ("the Black Sea", 43.0, 34.0),
        ("the Mediterranean", 35.0, 18.0),
        ("the Baltic", 57.5, 19.5),
        ("the Red Sea", 20.0, 38.5),
        ("the Persian Gulf", 27.0, 51.0),
        ("Hudson Bay", 60.0, -85.0),
    ] {
        let v = node_under(&lattice, dir_at(lat, lon)) as usize;
        let side = match (land_earth[v], z_earth[v] <= EARTH_SEA_M) {
            (false, _) => "the ocean",
            (true, true) => "land: a closed basin under the sea level",
            (true, false) => "land",
        };
        println!(
            "  {name} ({lat:.2}, {lon:.2}): node {v}, {} m, {side}",
            z_earth[v]
        );
    }
    let readings_earth = relief_readings(&lattice, &z_earth, &land_earth);
    let dirs: Vec<DVec3> = (0..n as u32).map(|v| dir_of(&lattice, v)).collect();
    let earth = read_land(
        &lattice,
        &area,
        &over_earth,
        &land_earth,
        &readings_earth,
        &|_| true,
        true,
    );
    let earth_no_ice = read_land(
        &lattice,
        &area,
        &over_earth,
        &land_earth,
        &readings_earth,
        &|v| lat_of(dirs[v as usize]) >= ANTARCTIC_LAT_DEG,
        true,
    );
    let earth_ocean = ocean_mean_m(&area, &over_earth, &land_earth);

    // The home planet, by the shard's own solve.
    let started = std::time::Instant::now();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    println!(
        "the home planet solved in {:.1} s; its sea's level {} m over the ladder radius",
        started.elapsed().as_secs_f64(),
        artifact.sea_m
    );
    let over_home: Vec<i32> = artifact
        .rows
        .iter()
        .map(|r| i32::from(r.z_m) - i32::from(artifact.sea_m))
        .collect();
    let land_home: Vec<bool> = artifact
        .rows
        .iter()
        .map(|r| (r.receiver_facies >> FACIES_SHIFT) & FACIES_SEA == 0)
        .collect();
    let readings_home = relief_readings(&lattice, &over_home, &land_home);
    let home = read_land(
        &lattice,
        &area,
        &over_home,
        &land_home,
        &readings_home,
        &|_| true,
        true,
    );
    let home_ocean = ocean_mean_m(&area, &over_home, &land_home);

    // The land whose window stays over the sea level: the closed basins and their walls left out.
    let dry = |readings: &[Option<ReliefReading>], v: u32| {
        readings[v as usize].is_some_and(|r| r.lo_m > 0)
    };
    let earth_dry = read_land(
        &lattice,
        &area,
        &over_earth,
        &land_earth,
        &readings_earth,
        &|v| dry(&readings_earth, v),
        false,
    );
    let earth_no_ice_dry = read_land(
        &lattice,
        &area,
        &over_earth,
        &land_earth,
        &readings_earth,
        &|v| dry(&readings_earth, v) && lat_of(dirs[v as usize]) >= ANTARCTIC_LAT_DEG,
        false,
    );
    let home_dry = read_land(
        &lattice,
        &area,
        &over_home,
        &land_home,
        &readings_home,
        &|v| dry(&readings_home, v),
        false,
    );
    let node_km = lattice.node_m() / 1_000.0;
    println!(
        "\nTHE THREE COLUMNS: Earth's land | Earth's land without Antarctica (south of {ANTARCTIC_LAT_DEG:.0}°) | the home \
         planet's land — every node of {node_km:.1} km, one relief rule, one sea rule"
    );
    table(&[&earth, &earth_no_ice, &home], node_km, true);
    println!(
        "  {:<44} {:>12.0} {:>12} {:>12.0}",
        "the ocean's mean depth, m", earth_ocean, "", home_ocean
    );
    println!(
        "\nTHE SAME COLUMNS, THE LAND WHOSE WINDOW STAYS OVER THE SEA LEVEL (the closed basins and their walls left out):"
    );
    table(&[&earth_dry, &earth_no_ice_dry, &home_dry], node_km, false);
    println!("THE CLASS CODES CHECKED: sea floor {SEA_FLOOR}, the land classes {LAND_CLASSES:?}");
}
