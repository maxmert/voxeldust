//! THE CHUNK BUDGET, AS IT STANDS (L25, ruling T1 of `owner_decisions_2026-09-16_terrain.md`:
//! *"the eight-millisecond budget"*). 8a adds roughness and makes dense chunks common; this bench
//! states the BASE a worker pays today, so 8a's delta is judged against a measurement and not
//! against a memory.
//!
//! It measures, in release, on THE world's home planet:
//!
//! 1. **THE NAMED CHUNK** — the one the walk's flights built in 111–112 ms
//!    (`NegX`, rung 1, 39847/28299/2146): the phases apart (the columns, the core cells with the
//!    carve, the box's halo, the extraction, the geometry), its cells, its triangles, its bytes,
//!    and ITS ANATOMY — how many of its radial columns hold one surface, how many hold a cave
//!    (three sign changes or more), how many stand solid or empty, and a few columns printed as
//!    runs of rock and air.
//! 2. **THE DISTRIBUTION** — every chunk the descent wants at rungs 0 to 3 from the walk's stand
//!    and from the six picture stands (capped per rung, evenly sampled): per rung the median, p90,
//!    p99 and max of the whole build, and the count over the 8 ms budget. The ten slowest chunks
//!    are named with their phase split and their triangle counts.
//! 3. **THE PARENT HIT** — what a warm parent cache saves on the dense chunk against a cold one.
//!
//! The stands are DERIVED exactly as the two flights derive them (`crates/bins/tests/
//! terrain_moving_eye.rs` for the walker, `crates/bins/tests/terrain_pictures.rs` for the six):
//! the day side on the equator, the star 15° over the horizon. Only the EYE POINT matters here —
//! the descent reads no facing — so the nose and its tilt are not restated.
//!
//! Run in release:
//!
//! ```text
//! cargo run --release -p vd-bins --example chunk_budget
//! ```
//!
//! Knobs (all optional): `VD_BUDGET_ROUNDS` (rounds on the named chunk, default 5),
//! `VD_BUDGET_PER_RUNG` (chunks sampled per rung per stand, default 100),
//! `VD_BUDGET_STANDS` (a comma-separated list of stand names; the default is all seven).

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use vd_bins::DEV;
use vd_client::chunks::{ParentCache, ParentMesh, geometry_from, parent_keys};
use vd_client::ladder_view::LadderView;
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_seed::bend::Face;
use vd_terrain::ChunkKey;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkLattice, ColumnField, column_field, generate_in};
use vd_terrain::extract::extract;
use vd_terrain::lattice::sample_box;

/// The budget one chunk's build is judged against: the owner's 8 ms (ruling V10, restated as L25).
const BUDGET_MS: f64 = 8.0;
/// The star's elevation both flights stand under.
const SUN_ELEVATION_DEG: f64 = 15.0;
/// The eye heights the seven stands use, in metres over the surface under them.
const WALK_EYE_M: f64 = 1.8;
const HILL_M: f64 = 300.0;
const ALOFT_M: f64 = 60_000.0;
const ORBIT_M: f64 = 2_000_000.0;
/// THE PARENT CACHE'S BUDGET, restated from `vd_client_render::terrain::PARENT_CACHE_BYTES` (that
/// crate rides the `render` feature and this bench is headless): 256 MB, and a floor of one
/// chunk's eight parents for each of eight workers.
const PARENT_CACHE_BYTES: usize = 256 << 20;
const PARENT_CACHE_FLOOR: usize = 64;
/// The far stand's frame share and the steps the seam search walks (the picture gate's own).
const FAR_FRAME_SHARE: f64 = 0.25;
const SEAM_STEPS: i32 = 400;
const SUN_ELEVATION_BAND_DEG: (f64, f64) = (12.0, 18.0);
const SUN_OFF_NOSE_DEG: f64 = 120.0;

/// One chunk's build, phase by phase, in milliseconds.
///
/// `parents_ms` is the phase the first form of this bench hid inside the geometry: the COARSER
/// meshes the morph targets read. A chunk whose parents the lane already holds pays nothing there;
/// a chunk that raises all eight pays more than every other phase together (MEASURED).
#[derive(Clone, Copy, Debug, Default)]
struct Phases {
    columns_ms: f64,
    core_ms: f64,
    halo_ms: f64,
    parents_ms: f64,
    extract_ms: f64,
    geometry_ms: f64,
    /// The box, the parents, the extraction and the geometry: what a worker pays on a miss.
    total_ms: f64,
    /// The same without the parents: what a worker pays when the lane holds them.
    warm_ms: f64,
    vertices: usize,
    triangles: usize,
    bytes: u64,
    parent_builds: u64,
    parent_hits: u64,
}

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Time one chunk's build with the phases apart. `parents` is the lane's cache: warm it by using
/// the same one across a stand, exactly as a worker does.
fn time_chunk(
    body: &vd_terrain::BodyDefinition,
    realm: RealmId,
    key: ChunkKey,
    parents: &ParentCache,
    rounds: u32,
) -> Option<Phases> {
    let rf = f64::from(rounds);
    // 1. The columns: one height, one direction and one biome per column of the chunk.
    let start = Instant::now();
    let mut column: Option<ColumnField> = None;
    for _ in 0..rounds {
        column = column_field(body, key.face, key.rung, key.x, key.y);
    }
    let columns_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
    let column = column?;
    // 2. The core cells: the carve (the caverns and the tubes) and the strata, or a skip.
    let start = Instant::now();
    for _ in 0..rounds {
        generate_in(body, &column, key.z)?;
    }
    let core_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
    // 3. The whole box: the core again plus the halo and the box's own set-up.
    let start = Instant::now();
    let mut samples = None;
    for _ in 0..rounds {
        samples = sample_box(body, key);
    }
    let box_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
    let samples = samples?;
    // 4. The extraction: the surface nets over the box.
    let start = Instant::now();
    let mut mesh_verts = 0;
    let mut mesh_tris = 0;
    for _ in 0..rounds {
        let mesh = extract(&samples);
        mesh_verts = mesh.vertices.len();
        mesh_tris = mesh.triangles.len();
    }
    let extract_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
    // 5. THE PARENTS: the coarser meshes the morph targets read, raised through the lane's own
    //    cache. A hit costs a lock; a miss costs a whole coarse chunk's build.
    let before = parents.stats();
    let start = Instant::now();
    for p in parent_keys(body, key) {
        let _ = parents.get(realm, body, p);
    }
    let parents_ms = start.elapsed().as_secs_f64() * 1e3;
    let after = parents.stats();
    // 6. The geometry: the positions, the morph targets, the normals, the skirts, with the
    //    parents WARM. The extraction runs inside it, so its own share is the difference.
    let start = Instant::now();
    let mut geometry = None;
    for _ in 0..rounds {
        geometry = geometry_from(body, realm, key, &samples, parents);
    }
    let geometry_all_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
    let geometry = geometry?;
    let warm_ms = box_ms + geometry_all_ms;
    Some(Phases {
        columns_ms,
        core_ms,
        halo_ms: box_ms - columns_ms - core_ms,
        parents_ms,
        extract_ms,
        geometry_ms: geometry_all_ms - extract_ms,
        total_ms: warm_ms + parents_ms,
        warm_ms,
        vertices: mesh_verts,
        triangles: mesh_tris,
        bytes: geometry.upload_bytes(),
        parent_builds: after.builds - before.builds,
        parent_hits: after.hits - before.hits,
    })
}

/// The quantile of a sorted list, by the nearest-rank rule.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// THE NAMED CHUNK'S ANATOMY: what its cells say about why it is dense.
struct Anatomy {
    rock: usize,
    air: usize,
    solid_columns: usize,
    empty_columns: usize,
    one_surface: usize,
    cave_columns: usize,
    max_flips: usize,
}

fn anatomy(lattice: &ChunkLattice) -> Anatomy {
    let mut out = Anatomy {
        rock: 0,
        air: 0,
        solid_columns: 0,
        empty_columns: 0,
        one_surface: 0,
        cave_columns: 0,
        max_flips: 0,
    };
    let mut b = 0;
    while b < CHUNK_EDGE {
        let mut a = 0;
        while a < CHUNK_EDGE {
            let mut flips = 0;
            let mut last = lattice.cell(a, b, 0).gap < 0;
            let mut rock_here = 0;
            let mut c = 0;
            while c < CHUNK_EDGE {
                let solid = lattice.cell(a, b, c).gap < 0;
                rock_here += usize::from(solid);
                if (c > 0) && (solid != last) {
                    flips += 1;
                }
                last = solid;
                c += 1;
            }
            out.rock += rock_here;
            out.air += CHUNK_EDGE - rock_here;
            out.max_flips = out.max_flips.max(flips);
            match flips {
                0 if rock_here == CHUNK_EDGE => out.solid_columns += 1,
                0 => out.empty_columns += 1,
                1 => out.one_surface += 1,
                _ => out.cave_columns += 1,
            }
            a += 1;
        }
        b += 1;
    }
    out
}

/// One radial column printed as runs of rock and air, bottom first.
fn column_runs(lattice: &ChunkLattice, a: usize, b: usize) -> String {
    let mut out = String::new();
    let mut run = 0;
    let mut last = lattice.cell(a, b, 0).gap < 0;
    let mut c = 0;
    while c < CHUNK_EDGE {
        let solid = lattice.cell(a, b, c).gap < 0;
        if solid == last {
            run += 1;
        } else {
            out.push_str(&format!("{}{run} ", if last { "R" } else { "A" }));
            run = 1;
            last = solid;
        }
        c += 1;
    }
    out.push_str(&format!("{}{run}", if last { "R" } else { "A" }));
    out
}

/// The seven stands' eye points in the planet's frame, named.
fn stands(body: &vd_terrain::BodyDefinition) -> Vec<(&'static str, DVec3)> {
    let Some(orbit) = vd_bins::home_orbit(DEV.universe_seed) else {
        return Vec::new();
    };
    let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
        .position
        .normalize();
    let along = sun.cross(DVec3::Z).normalize();
    let zenith = (90.0_f64 - SUN_ELEVATION_DEG).to_radians();
    let d = (sun * zenith.cos() + along * zenith.sin()).normalize();
    let h = vd_terrain::height::height_m(body, [d.x, d.y, d.z], 0);
    // THE SEAM STAND: the point on the cube's twelve edges where the star stands inside the
    // picture gate's band and nearest `SUN_OFF_NOSE_DEG` off the nose along the edge.
    let seam_param = |i: i32| -1.0 + 2.0 * f64::from(i) / f64::from(SEAM_STEPS);
    let seam_at = |face: Face, edge: i32, i: i32| {
        let (a, b) = match edge {
            0 => (1.0, seam_param(i)),
            1 => (-1.0, seam_param(i)),
            2 => (seam_param(i), 1.0),
            _ => (seam_param(i), -1.0),
        };
        DVec3::from_array(vd_seed::bend::direction(face, a, b))
    };
    let seam_nose = |face: Face, edge: i32, i: i32, sign: f64| {
        let p = seam_at(face, edge, i);
        let tangent = (seam_at(face, edge, (i + 1).min(SEAM_STEPS))
            - seam_at(face, edge, (i - 1).max(0)))
        .normalize()
            * sign;
        let level_sun = (sun - p * sun.dot(p)).normalize();
        let elevation = sun.dot(p).clamp(-1.0, 1.0).asin().to_degrees();
        let off_nose = tangent.dot(level_sun).clamp(-1.0, 1.0).acos().to_degrees();
        (p, elevation, off_nose)
    };
    let in_band =
        |elevation: f64| (SUN_ELEVATION_BAND_DEG.0..=SUN_ELEVATION_BAND_DEG.1).contains(&elevation);
    let mut best = (f64::MAX, Face::ALL[0], 0_i32, 0_i32, 1.0_f64);
    for face in Face::ALL {
        let mut edge = 0;
        while edge < 4 {
            let mut i = 0;
            while i <= SEAM_STEPS {
                for sign in [1.0, -1.0] {
                    let (_, elevation, off_nose) = seam_nose(face, edge, i, sign);
                    let off_miss = (off_nose - SUN_OFF_NOSE_DEG).abs();
                    let miss = if in_band(elevation) {
                        off_miss
                    } else {
                        360.0 + (elevation - SUN_ELEVATION_DEG).abs() + off_miss
                    };
                    if miss < best.0 {
                        best = (miss, face, edge, i, sign);
                    }
                }
                i += 1;
            }
            edge += 1;
        }
    }
    let (seam_dir, _, _) = seam_nose(best.1, best.2, best.3, best.4);
    let seam_h = vd_terrain::height::height_m(body, [seam_dir.x, seam_dir.y, seam_dir.z], 0);
    let far_m = body.ladder().radius_m()
        / (vd_core::geometry::REFERENCE_VIEW_FOV_Y_RAD * FAR_FRAME_SHARE * 0.5).sin();
    vec![
        ("walk", d * (h + WALK_EYE_M)),
        ("ground", d * (h + WALK_EYE_M)),
        ("hill", d * (h + HILL_M)),
        ("aloft", d * (h + ALOFT_M)),
        ("orbit", d * (h + ORBIT_M)),
        ("seam", seam_dir * (seam_h + WALK_EYE_M)),
        ("far", d * far_m),
    ]
}

fn named_chunk(body: &vd_terrain::BodyDefinition, realm: RealmId, named: ChunkKey, rounds: u32) {
    println!("\n=== 1. THE NAMED CHUNK {named:?} ===");
    let cold = ParentCache::with_capacity(64);
    match time_chunk(body, realm, named, &cold, rounds) {
        None => println!("chunk_budget: the named chunk is outside the ladder — REFUSED"),
        Some(p) => {
            println!(
                "phases (ms): columns {:.2} | core cells + carve {:.2} | halo + set-up {:.2} | \
                 PARENTS {:.2} | extraction {:.2} | geometry {:.2}",
                p.columns_ms, p.core_ms, p.halo_ms, p.parents_ms, p.extract_ms, p.geometry_ms
            );
            println!(
                "TOTAL with the parents COLD {:.2} ms ({:.1}x the budget); with the parents WARM \
                 {:.2} ms ({:.1}x)",
                p.total_ms,
                p.total_ms / BUDGET_MS,
                p.warm_ms,
                p.warm_ms / BUDGET_MS
            );
            println!(
                "shares of the cold build: columns {:.0}% | core {:.0}% | halo {:.0}% | parents \
                 {:.0}% | extraction {:.0}% | geometry {:.0}%",
                100.0 * p.columns_ms / p.total_ms,
                100.0 * p.core_ms / p.total_ms,
                100.0 * p.halo_ms / p.total_ms,
                100.0 * p.parents_ms / p.total_ms,
                100.0 * p.extract_ms / p.total_ms,
                100.0 * p.geometry_ms / p.total_ms
            );
            println!(
                "size: {} vertices, {} triangles, {} bytes uploaded ({} KB); it raised {} parent \
                 meshes and hit {}",
                p.vertices,
                p.triangles,
                p.bytes,
                p.bytes / 1024,
                p.parent_builds,
                p.parent_hits
            );
        }
    }
    let Some(column) = column_field(body, named.face, named.rung, named.x, named.y) else {
        return;
    };
    let Some(lattice) = generate_in(body, &column, named.z) else {
        return;
    };
    let a = anatomy(&lattice);
    let span = vd_terrain::digest::surface_column(body, named.face, named.rung, named.x, named.y);
    let cells = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;
    let cols = CHUNK_EDGE * CHUNK_EDGE;
    println!(
        "anatomy: how = {:?}; {} of {cells} cells rock ({:.0}%), {} air",
        lattice.how,
        a.rock,
        100.0 * a.rock as f64 / cells as f64,
        a.air
    );
    println!(
        "columns of {cols}: one surface {} ({:.0}%), CAVE (3+ sign changes) {} ({:.0}%), solid \
         {}, empty {}; the deepest column holds {} sign changes",
        a.one_surface,
        100.0 * a.one_surface as f64 / cols as f64,
        a.cave_columns,
        100.0 * a.cave_columns as f64 / cols as f64,
        a.solid_columns,
        a.empty_columns,
        a.max_flips
    );
    println!(
        "the column's surface spans chunks {} to {} ({} chunks along the radial), sampled {:.1} m \
         to {:.1} m (a relief of {:.1} m inside one chunk column), peak {:.1} m",
        span.lo,
        span.hi,
        span.hi - span.lo + 1,
        span.sampled_low_m,
        span.sampled_high_m,
        span.sampled_high_m - span.sampled_low_m,
        span.peak_m
    );
    let n = body.ladder().cells_per_edge(named.rung) as i32;
    let chunks = (n - 1) / CHUNK_EDGE as i32 + 1;
    let on_seam =
        (named.x == 0) | (named.y == 0) | (named.x >= chunks - 1) | (named.y >= chunks - 1);
    println!(
        "the face holds {chunks} chunks per edge at rung {}; this chunk stands at ({}, {}), {} a \
         face seam",
        named.rung,
        named.x,
        named.y,
        if on_seam { "ON" } else { "NOT on" }
    );
    println!(
        "the carve at rung {}: tubes {}, caverns {}",
        named.rung,
        vd_terrain::carve::tubes_carve_at(body, named.rung),
        vd_terrain::carve::caverns_carve_at(body, named.rung)
    );
    println!("a few columns, bottom first (R = rock, A = air, the number is cells):");
    for (a_i, b_i) in [(0, 0), (15, 15), (31, 31), (47, 47), (61, 61), (31, 0)] {
        println!("  ({a_i:>2},{b_i:>2}) {}", column_runs(&lattice, a_i, b_i));
    }
}

fn parent_hit(body: &vd_terrain::BodyDefinition, realm: RealmId, named: ChunkKey, rounds: u32) {
    println!("\n=== 3. THE PARENT HIT ON THE NAMED CHUNK ===");
    let Some(samples) = sample_box(body, named) else {
        return;
    };
    let rf = f64::from(rounds);
    let parents = parent_keys(body, named);
    let warm = ParentCache::with_capacity(64);
    for p in &parents {
        if let Some(m) = ParentMesh::build(body, *p) {
            warm.insert(realm, *p, Arc::new(m));
        }
    }
    let start = Instant::now();
    for _ in 0..rounds {
        let _ = geometry_from(body, realm, named, &samples, &warm);
    }
    let warm_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
    let start = Instant::now();
    for _ in 0..rounds {
        let fresh = ParentCache::with_capacity(64);
        let _ = geometry_from(body, realm, named, &samples, &fresh);
    }
    let cold_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
    let mut parent_ms = 0.0;
    let mut parent_tris = 0;
    for p in &parents {
        let start = Instant::now();
        let mut tris = 0;
        for _ in 0..rounds {
            tris = ParentMesh::build(body, *p).map_or(0, |m| m.triangle_count());
        }
        parent_ms += start.elapsed().as_secs_f64() * 1e3 / rf;
        parent_tris += tris;
    }
    println!(
        "the geometry step with {} parents WARM {warm_ms:.2} ms, COLD {cold_ms:.2} ms; the \
         parents' own builds {parent_ms:.2} ms ({parent_tris} triangles); a hit SAVES {:.2} ms \
         ({:.0}% of the cold build)",
        parents.len(),
        cold_ms - warm_ms,
        100.0 * (cold_ms - warm_ms) / cold_ms.max(f64::MIN_POSITIVE)
    );
}

/// ★ THE PARENT MESH'S OWN PHASES (section 4): a parent is not a cheap object — it is a WHOLE
/// COARSE CHUNK built with `extract_all_edges`, which keeps every edge crossing and not only the
/// surface ones. This states where a parent's own time goes, so the arc can say which phase grows
/// when 8a makes the ground rougher.
fn parent_anatomy(body: &vd_terrain::BodyDefinition, named: ChunkKey, rounds: u32) {
    println!("\n=== 4. WHAT ONE PARENT MESH COSTS, PHASE BY PHASE ===");
    let rf = f64::from(rounds);
    let parents = parent_keys(body, named);
    println!(
        "{:>24} | {:>9} | {:>11} | {:>9} | {:>9} | {:>9} | {:>9}",
        "parent (face x/y/z)", "box", "all-edges", "rest", "total", "verts", "tris"
    );
    let mut box_sum = 0.0;
    let mut edge_sum = 0.0;
    let mut rest_sum = 0.0;
    let mut tri_sum = 0;
    for p in &parents {
        let start = Instant::now();
        let mut samples = None;
        for _ in 0..rounds {
            samples = sample_box(body, *p);
        }
        let box_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
        let Some(samples) = samples else { continue };
        let start = Instant::now();
        let mut verts = 0;
        let mut tris = 0;
        for _ in 0..rounds {
            let m = vd_terrain::extract::extract_all_edges(&samples);
            verts = m.vertices.len();
            tris = m.triangles.len();
        }
        let edge_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
        // The surface-only extraction over the SAME box, for the ratio.
        let start = Instant::now();
        let mut surface_tris = 0;
        for _ in 0..rounds {
            surface_tris = extract(&samples).triangles.len();
        }
        let surface_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
        let start = Instant::now();
        for _ in 0..rounds {
            let _ = ParentMesh::build(body, *p);
        }
        let all_ms = start.elapsed().as_secs_f64() * 1e3 / rf;
        let rest_ms = all_ms - box_ms - edge_ms;
        box_sum += box_ms;
        edge_sum += edge_ms;
        rest_sum += rest_ms;
        tri_sum += tris;
        println!(
            "{:>6?} {:>6}/{:>6}/{:>5} | {box_ms:>6.2} ms | {edge_ms:>8.2} ms | {rest_ms:>6.2} ms \
             | {all_ms:>6.2} ms | {verts:>9} | {tris:>9}  (the surface-only extraction of the same \
             box: {surface_ms:.2} ms, {surface_tris} triangles)",
            p.face, p.x, p.y, p.z
        );
    }
    let all = box_sum + edge_sum + rest_sum;
    println!(
        "the eight parents together: the box {box_sum:.2} ms ({:.0}%), the ALL-EDGES extraction \
         {edge_sum:.2} ms ({:.0}%), the positions + buckets + normals {rest_sum:.2} ms ({:.0}%); \
         {tri_sum} triangles in all",
        100.0 * box_sum / all,
        100.0 * edge_sum / all,
        100.0 * rest_sum / all
    );
}

fn main() -> ExitCode {
    let Some(body) = vd_bins::home_body(DEV.universe_seed) else {
        println!("chunk_budget: REFUSED — the home system holds no planet the recipe accepts");
        return ExitCode::FAILURE;
    };
    let realm = RealmId::Planet(body.seed());
    let rounds = env_usize("VD_BUDGET_ROUNDS", 5).max(1) as u32;
    let per_rung = env_usize("VD_BUDGET_PER_RUNG", 100).max(1);
    let only: Option<Vec<String>> = std::env::var("VD_BUDGET_STANDS")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_owned()).collect());
    println!(
        "chunk_budget: home planet seed {}, ladder radius {:.0} m, {} rungs, {} octaves, relief \
         {} m",
        body.seed(),
        body.ladder().radius_m(),
        body.ladder().rungs,
        body.octave_count(),
        body.relief_bound_m(0) as i64
    );
    println!(
        "chunk_budget: the budget is {BUDGET_MS} ms of worker time per chunk (the box and the \
         geometry together); {rounds} rounds on the named chunk, one round per chunk in the \
         distribution, up to {per_rung} chunks per rung per stand"
    );

    let named = ChunkKey {
        face: Face::NegX,
        rung: 1,
        x: 39847,
        y: 28299,
        z: 2146,
    };
    named_chunk(&body, realm, named, rounds);
    parent_hit(&body, realm, named, rounds);
    parent_anatomy(&body, named, rounds);

    println!("\n=== 2. THE DISTRIBUTION OVER THE SEVEN STANDS (rungs 0 to 3) ===");
    let all = stands(&body);
    if all.is_empty() {
        println!("chunk_budget: REFUSED — the home planet states no orbit");
        return ExitCode::FAILURE;
    }
    // Every measured chunk, so the ten slowest can be named at the end.
    let mut all_rows: Vec<(&'static str, ChunkKey, Phases)> = Vec::new();
    for (name, eye) in &all {
        if let Some(list) = &only
            && !list.iter().any(|s| s == name)
        {
            continue;
        }
        let mut view = LadderView::default();
        let wanted = view.wanted(&body, [eye.x, eye.y, eye.z]);
        let len = eye.length();
        let surface =
            vd_terrain::height::height_m(&body, [eye.x / len, eye.y / len, eye.z / len], 0);
        println!(
            "\n-- the {name} stand: the eye {:.1} m over the recipe's surface, {} chunks wanted, \
             rungs {} to {}",
            len - surface,
            wanted.keys.len(),
            wanted.rung_min,
            wanted.rung_max
        );
        // The lane's own cache, warm across the stand, at THE RENDERER'S OWN BUDGET
        // (`vd_client_render::terrain::PARENT_CACHE_BYTES`, restated here because that crate rides
        // the `render` feature): 256 MB, with a floor of one chunk's parents per worker.
        let parents = ParentCache::default();
        parents.set_budget_bytes(PARENT_CACHE_BYTES, PARENT_CACHE_FLOOR);
        let mut by_rung: BTreeMap<u8, Vec<ChunkKey>> = BTreeMap::new();
        for key in &wanted.keys {
            if key.rung <= 3 {
                by_rung.entry(key.rung).or_default().push(*key);
            }
        }
        if by_rung.is_empty() {
            println!("   (the descent wants nothing at rungs 0 to 3 from this stand)");
            continue;
        }
        println!(
            "{:>4} | {:>6} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>9} | {:>9} | {:>9} | \
             {:>9}",
            "rung",
            "chunks",
            "> 8 ms",
            "median",
            "p90",
            "p99",
            "max",
            "mean box",
            "mean par",
            "mean extr",
            "mean geom"
        );
        for (rung, keys) in &by_rung {
            // An even sample of the ring, never its first `per_rung` keys.
            let step = (keys.len() / per_rung).max(1);
            let picked: Vec<ChunkKey> = keys.iter().copied().step_by(step).take(per_rung).collect();
            let mut times: Vec<f64> = Vec::with_capacity(picked.len());
            let mut box_sum = 0.0;
            let mut par_sum = 0.0;
            let mut extr_sum = 0.0;
            let mut geom_sum = 0.0;
            let mut over = 0;
            for key in picked {
                let Some(p) = time_chunk(&body, realm, key, &parents, 1) else {
                    continue;
                };
                times.push(p.total_ms);
                box_sum += p.columns_ms + p.core_ms + p.halo_ms;
                par_sum += p.parents_ms;
                extr_sum += p.extract_ms;
                geom_sum += p.geometry_ms;
                over += usize::from(p.total_ms > BUDGET_MS);
                all_rows.push((name, key, p));
            }
            if times.is_empty() {
                continue;
            }
            let n = times.len() as f64;
            times.sort_by(f64::total_cmp);
            println!(
                "{rung:>4} | {:>6} | {:>7} | {:>5.2} ms | {:>5.2} ms | {:>5.2} ms | {:>5.2} ms | \
                 {:>6.2} ms | {:>6.2} ms | {:>6.2} ms | {:>6.2} ms",
                times.len(),
                over,
                quantile(&times, 0.5),
                quantile(&times, 0.9),
                quantile(&times, 0.99),
                times[times.len() - 1],
                box_sum / n,
                par_sum / n,
                extr_sum / n,
                geom_sum / n
            );
        }
        let stats = parents.stats();
        println!(
            "   the parent cache over this stand: {} hits, {} builds, {} waits; it holds {} \
             meshes, {} MB",
            stats.hits,
            stats.builds,
            stats.waits,
            parents.len(),
            parents.held_bytes() >> 20
        );
    }

    println!("\n=== THE TEN SLOWEST CHUNKS MEASURED ===");
    all_rows.sort_by(|a, b| b.2.total_ms.total_cmp(&a.2.total_ms));
    println!(
        "{:>7} | {:>8} | {:>4} | {:>24} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>9} | {:>9}",
        "stand",
        "total",
        "rung",
        "chunk (face x/y/z)",
        "columns",
        "core",
        "halo",
        "parents",
        "extract",
        "geometry",
        "triangles"
    );
    for (name, key, p) in all_rows.iter().take(10) {
        println!(
            "{name:>7} | {:>5.1} ms | {:>4} | {:>6?} {:>6}/{:>6}/{:>5} | {:>5.2} ms | {:>5.2} ms \
             | {:>5.2} ms | {:>5.1} ms | {:>5.2} ms | {:>6.2} ms | {:>9}",
            p.total_ms,
            key.rung,
            key.face,
            key.x,
            key.y,
            key.z,
            p.columns_ms,
            p.core_ms,
            p.halo_ms,
            p.parents_ms,
            p.extract_ms,
            p.geometry_ms,
            p.triangles
        );
    }
    let mut totals: Vec<f64> = all_rows.iter().map(|r| r.2.total_ms).collect();
    let mut warms: Vec<f64> = all_rows.iter().map(|r| r.2.warm_ms).collect();
    totals.sort_by(f64::total_cmp);
    warms.sort_by(f64::total_cmp);
    let over = totals.iter().filter(|t| **t > BUDGET_MS).count();
    let over_warm = warms.iter().filter(|t| **t > BUDGET_MS).count();
    let sum_box: f64 = all_rows
        .iter()
        .map(|r| r.2.columns_ms + r.2.core_ms + r.2.halo_ms)
        .sum();
    let sum_par: f64 = all_rows.iter().map(|r| r.2.parents_ms).sum();
    let sum_extract: f64 = all_rows.iter().map(|r| r.2.extract_ms).sum();
    let sum_geom: f64 = all_rows.iter().map(|r| r.2.geometry_ms).sum();
    let sum_all = sum_box + sum_par + sum_extract + sum_geom;
    println!(
        "\nALL {} chunks, the parents COLD: median {:.2} ms, p90 {:.2} ms, p99 {:.2} ms, max \
         {:.2} ms; {over} over the budget ({:.0}%)",
        totals.len(),
        quantile(&totals, 0.5),
        quantile(&totals, 0.9),
        quantile(&totals, 0.99),
        totals.last().copied().unwrap_or(0.0),
        100.0 * over as f64 / totals.len().max(1) as f64
    );
    println!(
        "ALL {} chunks, the parents WARM: median {:.2} ms, p90 {:.2} ms, p99 {:.2} ms, max \
         {:.2} ms; {over_warm} over the budget ({:.0}%)",
        warms.len(),
        quantile(&warms, 0.5),
        quantile(&warms, 0.9),
        quantile(&warms, 0.99),
        warms.last().copied().unwrap_or(0.0),
        100.0 * over_warm as f64 / warms.len().max(1) as f64
    );
    println!(
        "the whole set's time by phase: the box (columns + core + halo) {:.0}%, the parents \
         {:.0}%, the extraction {:.0}%, the geometry {:.0}%",
        100.0 * sum_box / sum_all,
        100.0 * sum_par / sum_all,
        100.0 * sum_extract / sum_all,
        100.0 * sum_geom / sum_all
    );
    // THE CARD'S REACHABLE SHARE (ruling F9 item 2): the card computes the same sample box byte
    // for byte and hands it to the geometry step. Everything else stays on the CPU.
    println!(
        "the card's reachable share (it does the box and nothing else): {:.0}% of the whole set's \
         time; the extraction, the parents and the geometry stay on the CPU ({:.0}%)",
        100.0 * sum_box / sum_all,
        100.0 * (sum_all - sum_box) / sum_all
    );
    ExitCode::SUCCESS
}
