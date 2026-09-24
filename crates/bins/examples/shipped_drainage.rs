//! ★ THE DUNES NUMBER — THE DRAINAGE GATES ON THE SHIPPED GROUND (plan step 1 of
//! `docs/investigation/2026-09-24/terrain_believability_status.md`, "measure first, no world
//! change"; ruling B6, 2026-09-24).
//!
//! **THE OWNER'S WORDS.** *"the land look like dunes everywhere"* (the flight of 2026-09-24).
//!
//! **WHY IT EXISTS.** No gate can see the dunes today (status §4, THE BLIND SPOTS). `drain_census`
//! counts closed hollows at rung 3 (8 m columns) with a 0.1 m floor, over a stand 5.95 km a side. At
//! the floor of a 3 125 m ripple 199 m tall, two columns 8 m apart differ by 0.026 m (status §4,
//! ARITH), so that census reads a dune field as drained ground. This instrument reads the ground at
//! the distance the owner judged it (rung 7: 128 m columns, drawn out to 111 km) over a square many
//! ripples wide, and it measures each closed basin by its SPILL DEPTH, which does not shrink with the
//! column spacing. Its numbers CAN GO RED, and R2 (plan step 3) must turn them.
//!
//! **WHAT IT READS.** The home planet, solved ONCE through `vd_bins::artifact_worker::run_solve`
//! (about two minutes), as `lake_census` and `erosion_bench` do; never a store, because a store
//! holds whatever version somebody last flew. The field is the one the chunk builder reads at the
//! rung (`PyramidField::level_for`: the rows themselves at rungs 0 to 9). Then, on each stand — the
//! belt, the coast and a plain, picked by `vd_bins::drainage_gates::pick_stands` (the rule
//! `drain_census` uses), and one more stand along `--dir` when it is stated (the reference stand) —
//! a square of whole chunks at the rung. Three variants of ONE grid:
//!
//! 1. **(a) SHIPPED** — the chunk builder's OWN function, `vd_terrain::chunk::column_field`, with the
//!    field: the solve plus the fine octaves the rung keeps. Nothing re-implemented.
//! 2. **(b) SOLVE ONLY** — the chunk builder's reads through the ONE shared reader
//!    (`vd_bins::shore_geometry::RungReader`: the side, the water, the slope share and `Z`, in the
//!    chunk builder's order), into the same kernel (`column_surface_at`), with the first octave set
//!    at the rung's octave count, so the sum adds nothing (`relief_parts_from` runs its loop from
//!    `first` to the count; the doc of `relief_of_table_from` states it: "`first` at or past `count`
//!    sums nothing"). The bench and the shore law still run. ★ Before (b) means anything,
//!    the instrument compares the shared reader's own surface (the shipped first octave) with
//!    `column_field`, column by column; one differing column is RED, because then (b) is not "the
//!    shipped ground without its fine octaves".
//! 3. **(c) THE FINE RELIEF** — (a) minus (b), column by column: the lift, the spread, the slope and
//!    the spacing of what the noise adds.
//!
//! On (a) and on (b): G-HACK and G-BREACH (the R1 bench's own two functions, now
//! `vd_bins::drainage_gates`), and THE SPILL CENSUS (`drainage_gates::spill_census`): a priority
//! flood from the square's rim and from every wet column; each closed basin's depth (spill minus
//! pit, metres), area and volume; the count of basins 1, 10 and 100 m deep or deeper on each
//! 100 km² of dry ground; the depth histogram; the deepest ten, with a cell and an aim a picture can
//! be taken at.
//!
//! Each gate prints one line `CENSUS <GATE-NAME> <value> <band> <GREEN|RED|NO-BAND>`, the unit in
//! the gate's name, so a script can grep the run. The band word carries its SOURCE in brackets and
//! holds no space, so a line always splits into five words: `[1,6](Sassolas-Serrayet_…_2018)`, or
//! `none(<why>)`. G-HACK's band is published (Sassolas-Serrayet, Cattin, Ferry & Godard 2018); the
//! two `[0,0]` bands are this instrument's own identity checks; no published band exists for the
//! others in our units, so they print their reading and `NO-BAND`, and no band is invented to pass.
//! A value the instrument could not make (`NaN`) is RED under a band, because it cannot pass.
//!
//! **WHAT A DUNE FIELD PRINTS, AND WHY** (ARITH — an expectation read off the octave table, NOT a
//! band). At rung 7 the kernel sums four fine octaves over the solve (status §5 D8): 25 000 m smooth
//! (496 m), then 12 500 m, 6 250 m and 3 125 m RIDGED (402 m, 320 m, 199 m) — each amplitude at a
//! roughness factor m of one, and every column multiplies them by its own m, 0.06 to 1.
//!
//! - A ridged octave is `1 − 2|n|` (`recipe/src/height.rs:443`). Its crest is a LINE where the noise
//!   crosses zero, and its trough is a ROUND hollow where the noise peaks. So the 3 125 m octave digs
//!   a closed trough about every ripple, with a ring of crests around it.
//! - THE DEPTH. The term runs from −a to +a, so a trough stands up to 2a under its ring, and its
//!   spill over the ring's lowest saddle is of the order of a: about 200 m at m = 1, and
//!   199 × 0.06 = 12 m at the floor.
//! - THE COUNT. Gradient noise has of the order of one peak and one trough per lattice cell, and the
//!   fold turns both into troughs. The 3 125 m lattice cell is 3.125² = 9.8 km², so the ceiling is
//!   of the order of 10 to 20 troughs on each 100 km². The three wider octaves tilt the rings and
//!   open a part of them: their slopes a·2π/λ at m = 1 are 0.125, 0.20 and 0.32 — 0.40 as a root sum
//!   of squares, the same as the 3 125 m octave's own 0.40. So a part of the rings stays closed: not
//!   all, and not none.
//! - SO A DUNE FIELD PRINTS: basins 10 m deep or deeper at of the order of 1 to 10 on each 100 km²;
//!   basins 100 m deep only where m is near one (the belt); a filled share of the dry ground of some
//!   percent; G-BREACH of the order of (λ/2)·(depth/2)·cell = 1 562 m × 100 m × 128 m = 2 × 10⁷ m³ a
//!   pit at m = 1, and 1 562 m × 6 m × 128 m = 1 × 10⁶ m³ at the floor; a hill spacing on (a) of the
//!   order of the table's own Rice spacing along a grid axis, 2π·√(4/3 · Σs² / Σ(s·k)²) = 4.7 km (the
//!   run prints it from the table; the 4/3 is the isotropic walk, see `Table::rice_km`; each octave
//!   is taken as ONE wavenumber, so the number is an order and never a band); and
//!   G-HACK RED — water ends in a trough within half a ripple (1.6 km, 12 columns), and radial flow
//!   into a round trough gives L = √(a/π), which is c = 0.56 at n = 0.5: under the band's floor of 1.
//!
//! **WHAT A DRAINED FIELD PRINTS, AND WHY.** On drained ground every dry column has a downhill path
//! to the rim or to the water. So the flood fills nothing: no basin over the arithmetic's own noise
//! (a word of the length format is 3 × 10⁻¹¹ m), a filled share near 0 % (`G-FILLED-1m-pct`, the
//! share under the basins 1 m deep or deeper; `G-FILLED-pct` also counts a basin one word deep, so
//! it can read over zero on drained ground), and G-BREACH near zero —
//! the only pits left are flat cells, and a flat cell's walk removes nothing. Its streams join into
//! trunks, so G-HACK reads inside c ∈ [1, 6] and n ∈ [0.45, 0.7]. A lake does not count against it:
//! a lake column is WET, so it is an outlet and never a basin.
//!
//! **(b) IS NOT A DRAINED FIELD EITHER.** It is a Catmull-Rom read of 8 192 m nodes, so it holds no
//! valley narrower than about two nodes. The R1 bench read the same kind of bicubic upsample at
//! L = 1.99·a^0.883 (DOC, status §7 W9): n over the band, because water runs down a smooth surface
//! in parallel sheets and never joins. Its hill spacing should read 16 km or more. A closed basin in
//! (b) is a hollow the SOLVE left dry (status §7 W3), or a Catmull-Rom overshoot beside a sharp node.
//!
//! Run in release, ONE job at a time, and only while the `perf` tree is idle (ruling B6.5):
//!
//! ```text
//! cargo run --release -p vd-bins --example shipped_drainage -- [--rung 7] [--side-km 200] [--dir dx dy dz]
//! ```
//!
//! **Example.** A pilot hangs 120 km over the belt. The instrument reads the rung-7 ground under
//! her as the chunk builder makes it, one column every 128 m over a square 206 km a side, and
//! floods it from the rim and from every column the coast mask calls sea or lake. If the fine
//! octaves make dunes, the flood fills a trough every few kilometres, and `G-SPILL-10m` prints a
//! number of order ten on each 100 km², while the same square without the fine octaves prints a
//! few. After R2 ships one eroded level, the same line on the same stand must fall toward zero.

use std::sync::Arc;
use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_bins::drainage_gates::{
    Basin, DEPTH_CLASSES, Heights, Spill, cell_of, g_breach, g_hack, pick_stands, spill_census,
};
use vd_bins::shore_geometry::{RungReader, quantile_sorted, unit_of};
use vd_core::glam::DVec3;
use vd_recipe::Gi;
use vd_recipe::height::{OCTAVE_RIDGED, roughness_factor};
use vd_recipe::noise::NOISE_BITS;
use vd_recipe::plan::{FieldRead, PlanCharter, column_surface_at, site_direction};
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, CoastCounts, PyramidField, sample_z, slope_share};
use vd_terrain::chunk::{CHUNK_EDGE, column_field};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::units::{LENGTH_BITS, STEPS_PER_M};

/// ★ G-HACK's PUBLISHED BANDS — `L = c·a^n`, `L` in km and `a` in km² (Sassolas-Serrayet, Cattin,
/// Ferry & Godard 2018; the R1 bench reads the same two). A field is inside only where BOTH hold.
const HACK_C: (f64, f64) = (1.0, 6.0);
const HACK_N: (f64, f64) = (0.45, 0.7);
/// The source word the two G-HACK bands print with.
const HACK_SOURCE: &str = "Sassolas-Serrayet_Cattin_Ferry_Godard_2018";
/// The source word of a reading with no published band in our units.
const NO_PUBLISHED: &str = "no_published_band";
/// The source word of the shipped-path check: this instrument's own identity, `column_field` and the
/// shared reader must give one word for one column.
const PATH_CHECK: &str = "own_check:column_field_equals_RungReader";

/// ★ A GATE'S BAND AND THE BAND'S SOURCE. The source rides in the band word with no space in it,
/// so a `CENSUS` line always splits into five words (the rule `lake_census` states for its lines).
enum Band {
    /// An inclusive band `[lo, hi]`, and where it comes from.
    Range(f64, f64, &'static str),
    /// No band yet; the word says why.
    NoBand(&'static str),
}

/// The rung the owner judged the dunes at: rung 7 draws out to 111 km (status §5 D8), the middle of
/// the pilot's 100–150 km look.
const DEFAULT_RUNG: u8 = 7;
/// The square's side, in km: about 64 ripples of the 3 125 m crest and 25 solve nodes a side, so
/// one hollow or one node never decides the reading.
const DEFAULT_SIDE_KM: f64 = 200.0;
/// How many of the deepest basins each variant names by cell and aim.
const DEEPEST_SHOWN: usize = 10;
/// A roughness factor at or over this share counts as FULL roughness (a factor of one at the
/// noise's fraction bits, less the kernel's one-unit rounding).
const FULL_ROUGH: f64 = 0.999;

/// What every stand reads: the body, its solved artifact, the coast counts folded once, the rung
/// and the square's size.
struct Ground<'a> {
    body: &'a BodyDefinition,
    artifact: &'a Artifact,
    counts: Arc<CoastCounts>,
    rung: u8,
    /// Whole chunks along each side of the square.
    chunks: i32,
}

/// One stand to measure.
struct Aim {
    name: &'static str,
    /// The stand's short word in a `CENSUS` gate name.
    slug: &'static str,
    dir: DVec3,
}

/// The octave table's own two statements about the fine relief at the rung, printed beside the
/// readings of (c) so the expectation is COMPUTED from the table and never typed.
struct Table {
    /// The fine octaves' slopes `a·2π/λ` at a factor of one, as a root sum of squares.
    slope_rss: f64,
    /// ★ Rice's mean distance between crests along a straight walk down one grid axis, in km:
    /// `2π·√(λ₂ / λ₄)` over the walk's own spectral moments (Rice 1945). The noise is ISOTROPIC, so
    /// the walk meets an octave of wavenumber `k` at an angle `θ`, as `k·cos θ`; over all angles
    /// `⟨cos²θ⟩ = 1/2` and `⟨cos⁴θ⟩ = 3/8` (the directional moments of an isotropic surface,
    /// Longuet-Higgins 1957), so `λ₂/λ₄ = 4/3 · Σs² / Σ(s·k)²`. The one-wave formula without the
    /// 4/3 is the spacing along the wave itself, 13 % shorter than any grid walk reads.
    rice_km: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let after = |name: &str, k: usize| -> Option<f64> {
        let at = args.iter().position(|a| a == name)?;
        args.get(at + k).and_then(|v| v.parse().ok())
    };
    let rung = after("--rung", 1).map_or(DEFAULT_RUNG, |v| v as u8);
    let side_km = after("--side-km", 1).unwrap_or(DEFAULT_SIDE_KM);
    let stated = if args.iter().any(|a| a == "--dir") {
        match [after("--dir", 1), after("--dir", 2), after("--dir", 3)] {
            [Some(x), Some(y), Some(z)] => Some(DVec3::new(x, y, z).normalize()),
            _ => panic!("--dir takes three numbers: dx dy dz"),
        }
    } else {
        None
    };

    println!(
        "shipped_drainage — the dunes number (plan step 1, docs/investigation/2026-09-24 §4 and §8)"
    );
    // ── THE INPUT: the home planet, solved once. ────────────────────────────────────────────────
    let started = Instant::now();
    let artifact = run_solve(&SolveJob {
        body: home_planet(),
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let fine = body
        .macro_lattice()
        .expect("the home planet has a macro lattice");
    println!(
        "  the solve: {:.1} s; GENERATOR_VERSION {}, ARTIFACT_VERSION {}; the sea stands {} m over the ladder radius ({:.0} m)",
        started.elapsed().as_secs_f64(),
        vd_terrain::GENERATOR_VERSION,
        artifact.version,
        artifact.sea().unwrap_or(0),
        body.radius_m()
    );
    assert!(
        rung < body.ladder().rungs,
        "the home planet has rungs 0 to {}",
        body.ladder().rungs - 1
    );

    // ── WHAT THE RUNG SUMS over the solve. ──────────────────────────────────────────────────────
    let level = PyramidField::level_for(&fine, artifact.pyramid.len() as u32, rung);
    let first = body.first_fine();
    let live = body.octaves_at(rung).len();
    println!(
        "  rung {rung}: {} m columns (nominal); the field is pyramid level {level}; the solve stands in for octaves 0 to {}; the fine octaves this rung sums:",
        vd_seed::ladder::cell_m(rung),
        first.saturating_sub(1)
    );
    let (mut s2, mut sk2) = (0.0f64, 0.0f64);
    for k in first..live {
        let o = &body.octave_table()[k];
        let wave_m = body.radius_m() / vd_terrain::body::octave_frequency(o);
        let amp_m = vd_terrain::body::octave_amplitude_m(o);
        let wavenumber = core::f64::consts::TAU / wave_m;
        let s = amp_m * wavenumber;
        s2 += s * s;
        sk2 += (s * wavenumber) * (s * wavenumber);
        println!(
            "    octave {k:>2}: {wave_m:>8.0} m wavelength, {amp_m:>6.1} m amplitude and a slope of {s:.3} at a roughness factor of one, {}",
            if o.kind == OCTAVE_RIDGED {
                "RIDGED (1 − 2|n|: round troughs, rings of crests)"
            } else {
                "smooth"
            }
        );
    }
    if live <= first {
        println!(
            "    none: at this rung the ground is the solve alone, so (a) and (b) must read the same"
        );
    }
    let table = Table {
        slope_rss: s2.sqrt(),
        rice_km: if sk2 > 0.0 {
            core::f64::consts::TAU * (4.0 / 3.0 * s2 / sk2).sqrt() / 1_000.0
        } else {
            f64::INFINITY
        },
    };
    println!(
        "  the table's own statements: the fine slopes' root sum of squares {:.3} at a factor of one (the body's own a·2π/λ);\n    Rice's crest spacing along a grid axis {:.2} km (each octave taken as one wavenumber: an order, not a band)",
        table.slope_rss, table.rice_km
    );

    // ── THE SQUARE: whole chunks, the side rounded up. ──────────────────────────────────────────
    let chunk_m = CHUNK_EDGE as f64 * f64::from(vd_seed::ladder::cell_m(rung));
    let chunks = ((side_km * 1_000.0 / chunk_m).ceil() as i32).max(1);
    let ground = Ground {
        body: &body,
        artifact: &artifact,
        // The body's own coast counts, folded once (the inherent `Artifact::coast_counts`).
        counts: artifact.coast_counts(),
        rung,
        chunks,
    };
    println!(
        "  each square: {chunks} × {chunks} chunks of {CHUNK_EDGE} columns = {} columns a side, about {:.0} km (asked {side_km:.0} km)",
        chunks * CHUNK_EDGE as i32,
        f64::from(chunks) * chunk_m / 1_000.0
    );

    // ── THE STANDS. ─────────────────────────────────────────────────────────────────────────────
    let mut aims: Vec<Aim> = pick_stands(&artifact, &fine)
        .into_iter()
        .map(|s| Aim {
            name: s.name,
            slug: s.name.rsplit(' ').next().unwrap_or(s.name),
            dir: s.dir,
        })
        .collect();
    if let Some(dir) = stated {
        aims.push(Aim {
            name: "the stated direction (the reference stand)",
            slug: "stated",
            dir,
        });
    }
    for aim in &aims {
        measure(&ground, aim, &table);
    }

    println!(
        "\nHOW TO READ (the arithmetic is in this file's header; ARITH, an expectation and not a band):"
    );
    println!(
        "  a DUNE field: basins 10 m deep or deeper at of the order of 1 to 10 a 100 km², a filled share of some percent,"
    );
    println!(
        "    G-BREACH of the order of 1e6 to 2e7 m³ a pit, a hill spacing on (a) of the order of the table's Rice spacing ({:.2} km),",
        table.rice_km
    );
    println!(
        "    and G-HACK RED: radial flow into round troughs gives c = 0.56 at n = 0.5, under the band's floor of 1."
    );
    println!(
        "  a DRAINED field: no basin over the arithmetic's noise, G-FILLED-1m near 0 %, G-BREACH near 0, G-HACK inside both bands."
    );
    println!(
        "  (b) the solve alone: a few basins (hollows the solve left dry), G-HACK RED on n (parallel sheets; the R1 bench read"
    );
    println!("    n = 0.883 on the bicubic upsample), a hill spacing of 16 km or more.");
    println!(
        "  every CENSUS line reads: CENSUS <gate>:<stand>[:<variant>] <value> [lo,hi](source)|none(why) <GREEN|RED|NO-BAND>"
    );
}

/// ★ ONE STAND: the square, the check of the shipped path, the three variants and their gates.
fn measure(g: &Ground, aim: &Aim, table: &Table) {
    let t0 = Instant::now();
    let edge = CHUNK_EDGE as i32;
    let (face, ci, cj) = cell_of(g.body, aim.dir, g.rung);
    let n_cells = g.body.ladder().cells_per_edge(g.rung) as i32;
    let whole = n_cells / edge;
    println!(
        "\n══ {} — aim {:.6} {:.6} {:.6} (feed it to `land_stand -- 2000 dx dy dz`) ══",
        aim.name, aim.dir.x, aim.dir.y, aim.dir.z
    );
    if whole < g.chunks + 2 {
        println!(
            "  REFUSED — a face holds {whole} whole chunks at rung {}, and the square needs {} and one chunk of margin on each side",
            g.rung, g.chunks
        );
        unread_stand(aim.slug);
        return;
    }
    let Some(reader) = RungReader::new(g.body, g.artifact, &g.counts, face, g.rung) else {
        println!(
            "  NO READING — the chunk builder's read of rung {} on face {face:?} does not exist",
            g.rung
        );
        unread_stand(aim.slug);
        return;
    };
    // ★ THE SQUARE STAYS ON ONE FACE, one chunk clear of each edge, so the grid is ONE regular
    // lattice and every column's site is its own face's cell. A stand near an edge pulls the square
    // inward, and the pull is printed.
    let want_x = (ci - g.chunks * edge / 2).div_euclid(edge);
    let want_y = (cj - g.chunks * edge / 2).div_euclid(edge);
    let x0 = want_x.clamp(1, whole - 1 - g.chunks);
    let y0 = want_y.clamp(1, whole - 1 - g.chunks);
    let (i0, j0) = (x0 * edge, y0 * edge);
    let charter = &reader.charter;
    let w = (g.chunks * edge) as usize;

    // The column spacing ON THE GROUND at the square's middle: the chord between two neighbour
    // columns' directions times the ladder radius. The face bend makes it differ from the rung's
    // nominal cell, and the ground's own height (a few km over 6 300 km) is left out.
    let unit = |i: i32, j: i32| unit_of(site_direction(charter, i32::from(face.index()), i, j));
    let (mi, mj) = (i0 + w as i32 / 2, j0 + w as i32 / 2);
    let radius_m = g.body.radius_m();
    let dx = (unit(mi + 1, mj) - unit(mi, mj)).length() * radius_m;
    let dy = (unit(mi, mj + 1) - unit(mi, mj)).length() * radius_m;
    let cell_m = (dx * dy).sqrt();
    let cell_km2 = dx * dy / 1.0e6;
    let pulled = (x0 - want_x).abs().max((y0 - want_y).abs());
    let pulled_km = f64::from(pulled * edge) * cell_m / 1_000.0;
    let holds_stand = (i0..i0 + w as i32).contains(&ci) & (j0..j0 + w as i32).contains(&cj);
    println!(
        "  the square: face {face:?}, cells [{i0}, {}) × [{j0}, {}) at rung {}; {w} × {w} columns of {dx:.1} m × {dy:.1} m on the ground; {:.1} km a side, {:.0} km²",
        i0 + w as i32,
        j0 + w as i32,
        g.rung,
        w as f64 * cell_m / 1_000.0,
        (w * w) as f64 * cell_km2
    );
    if pulled_km > 0.0 {
        println!(
            "  the stand stands near a face edge: the square was pulled {pulled_km:.1} km inward; it {} the stand's own column",
            if holds_stand {
                "still holds"
            } else {
                "NO LONGER holds"
            }
        );
    }

    // ── THE READ: (a) from the chunk builder, (b) and the check through the shared reader. ──────
    let first_none = charter.octave_count.raw() as usize;
    let cells = w * w;
    let mut shipped = vec![0i64; cells];
    let mut solve = vec![0i64; cells];
    let mut dry = vec![true; cells];
    let mut factor = vec![0i64; cells];
    let (mut differ, mut water_differ, mut refused, mut unread) = (0usize, 0usize, 0usize, 0usize);
    let mut cy = 0;
    while cy < g.chunks {
        let mut cx = 0;
        while cx < g.chunks {
            let (x, y) = (x0 + cx, y0 + cy);
            let field = reader.source.field();
            let Some(cf) = column_field(g.body, Some(field), face, g.rung, x, y) else {
                refused += 1;
                cx += 1;
                continue;
            };
            for b in 0..edge {
                for a in 0..edge {
                    let k = (b * edge + a) as usize;
                    let at = (cy * edge + b) as usize * w + (cx * edge + a) as usize;
                    let h = cf.columns[k].1;
                    shipped[at] = (h - charter.radius).raw();
                    let Some(pair) = read_pair(&reader, x * edge + a, y * edge + b, first_none)
                    else {
                        unread += 1;
                        continue;
                    };
                    differ += usize::from(pair.shipped != h);
                    water_differ += usize::from(pair.water != cf.water[k]);
                    solve[at] = (pair.solve - charter.radius).raw();
                    dry[at] = !pair.wet;
                    factor[at] = pair.factor.raw();
                }
            }
            cx += 1;
        }
        cy += 1;
    }
    if refused + unread > 0 {
        println!(
            "  NO READING — the field did not hold {refused} of the square's chunks and {unread} columns; nothing below would be the shipped ground"
        );
        unread_stand(aim.slug);
        return;
    }
    println!(
        "  THE SHIPPED PATH, CHECKED: `column_field` against the shared reader (`shore_geometry::RungReader`) at the shipped first octave ({}):\n    {differ} of {cells} surfaces and {water_differ} water radii differ (both must be 0, or (b) is not the shipped ground less its octaves)",
        reader.first
    );
    census(
        &format!("SHIPPED-PATH-DIFFER:{}", aim.slug),
        (differ + water_differ) as f64,
        &Band::Range(0.0, 0.0, PATH_CHECK),
    );

    // ── WHAT STANDS IN THE SQUARE. ──────────────────────────────────────────────────────────────
    let per_m = (STEPS_PER_M << LENGTH_BITS) as f64;
    let dry_cells = dry.iter().filter(|&&d| d).count();
    println!(
        "  the water: {:.2} % of the columns stand under the sea or a lake (the coast mask's side word); {:.0} km² of dry ground",
        (cells - dry_cells) as f64 * 100.0 / cells as f64,
        dry_cells as f64 * cell_km2
    );
    if dry_cells == 0 {
        // No dry ground, no drainage to read: the published gate prints NaN, which is RED.
        println!("  NO READING — the square holds no dry ground");
        census(
            &format!("G-HACK-c:{}:shipped", aim.slug),
            f64::NAN,
            &Band::Range(HACK_C.0, HACK_C.1, HACK_SOURCE),
        );
        return;
    }
    let mut floor_m: Vec<f64> = (0..cells)
        .filter(|&i| dry[i])
        .map(|i| solve[i] as f64 / per_m)
        .collect();
    floor_m.sort_by(f64::total_cmp);
    println!(
        "  the solved ground under the dry columns ((b), m over the ladder radius): lowest {:.0}, median {:.0}, highest {:.0}",
        quantile_sorted(&floor_m, 0.0),
        quantile_sorted(&floor_m, 0.5),
        quantile_sorted(&floor_m, 1.0)
    );
    let one = f64::from(1u32 << NOISE_BITS);
    let mut m: Vec<f64> = (0..cells)
        .filter(|&i| dry[i])
        .map(|i| factor[i] as f64 / one)
        .collect();
    m.sort_by(f64::total_cmp);
    let m_mean = m.iter().sum::<f64>() / m.len() as f64;
    let full = m.iter().filter(|&&v| v >= FULL_ROUGH).count();
    println!(
        "  the roughness factor m on the dry columns (the greater of the placeholder noise and the solve's slope share):\n    mean {m_mean:.3}, p10 {:.3}, median {:.3}, p90 {:.3}; {:.1} % of the columns at full roughness",
        quantile_sorted(&m, 0.1),
        quantile_sorted(&m, 0.5),
        quantile_sorted(&m, 0.9),
        full as f64 * 100.0 / m.len() as f64
    );

    // ── (a) and (b): the gates. ─────────────────────────────────────────────────────────────────
    let place = Place {
        face,
        i0,
        j0,
        w,
        rung: g.rung,
        charter,
        cell_km2,
    };
    let a = variant(
        &Heights {
            w,
            cell_m,
            h: &shipped,
            per_m,
            dry: Some(dry.as_slice()),
        },
        "(a) SHIPPED — `column_field`: the solve plus the fine octaves the rung keeps",
        aim.slug,
        "shipped",
        &place,
    );
    let b = variant(
        &Heights {
            w,
            cell_m,
            h: &solve,
            per_m,
            dry: Some(dry.as_slice()),
        },
        "(b) SOLVE ONLY — the same reads and kernel, the octave sum empty (the bench and the shore still run)",
        aim.slug,
        "solve",
        &place,
    );
    // No pit in (b) and some in (a) is an unbounded ratio; no pit in either is no ratio at all, and a
    // reading the instrument could not make prints NaN, never a number.
    let ratio = if b.breach_m3 > 0.0 {
        a.breach_m3 / b.breach_m3
    } else if a.breach_m3 > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    println!(
        "  G-BREACH (a) over (b): {ratio:.2} — how many times the fine octaves multiply the channel a pit needs (one grid, one definition;\n    inf = no pit in (b), NaN = no pit in either);\n  basins 10 m deep or deeper: {} in (a) against {} in (b)",
        a.deep10, b.deep10
    );
    census(
        &format!("G-BREACH-RATIO:{}", aim.slug),
        ratio,
        &Band::NoBand(NO_PUBLISHED),
    );

    // ── (c) THE FINE RELIEF, (a) − (b). ─────────────────────────────────────────────────────────
    let diff: Vec<i64> = shipped.iter().zip(&solve).map(|(s, o)| s - o).collect();
    let mut fine_m: Vec<f64> = diff
        .iter()
        .zip(&dry)
        .filter(|&(_, d)| *d)
        .map(|(v, _)| *v as f64 / per_m)
        .collect();
    let count = fine_m.len() as f64;
    let mean = fine_m.iter().sum::<f64>() / count;
    let rms = (fine_m.iter().map(|v| v * v).sum::<f64>() / count).sqrt();
    fine_m.sort_by(f64::total_cmp);
    let (mut sq, mut slopes) = (0.0f64, 0usize);
    for y in 1..w - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            // The column and its four neighbours on dry ground: a central difference across the
            // shore would read the shore law's step to the sea floor, not the fine relief.
            if !(dry[i] & dry[i - 1] & dry[i + 1] & dry[i - w] & dry[i + w]) {
                continue;
            }
            let gx = (diff[i + 1] - diff[i - 1]) as f64 / per_m / (2.0 * dx);
            let gy = (diff[i + w] - diff[i - w]) as f64 / per_m / (2.0 * dy);
            sq += gx * gx + gy * gy;
            slopes += 1;
        }
    }
    let slope_rms = (sq / slopes.max(1) as f64).sqrt();
    println!("  ── (c) THE FINE RELIEF, (a) − (b), on the dry columns ──");
    println!(
        "    the mean {mean:.1} m (the ridged LIFT, status §5 D7: an upper bound of about 960 m); the RMS {rms:.1} m;\n    p1 {:.1} m, median {:.1} m, p99 {:.1} m",
        quantile_sorted(&fine_m, 0.01),
        quantile_sorted(&fine_m, 0.5),
        quantile_sorted(&fine_m, 0.99)
    );
    println!(
        "    the RMS slope {slope_rms:.3} (a gradient's magnitude, metres a metre, over {slopes} columns whose four neighbours are dry);\n    the table states {:.3} at a factor of one and {:.3} at this square's mean factor — the body's own a·2π/λ per octave,\n    not an RMS of the drawn noise, so the two agree in order only",
        table.slope_rss,
        table.slope_rss * m_mean
    );
    let none = Band::NoBand(NO_PUBLISHED);
    census(&format!("G-FINE-MEAN-m:{}", aim.slug), mean, &none);
    census(&format!("G-FINE-RMS-m:{}", aim.slug), rms, &none);
    census(&format!("G-FINE-SLOPE-rms:{}", aim.slug), slope_rms, &none);
    let spacing = [
        ("shipped", hill_spacing_km(&shipped, &dry, w, dx, dy)),
        ("solve", hill_spacing_km(&solve, &dry, w, dx, dy)),
        ("fine", hill_spacing_km(&diff, &dry, w, dx, dy)),
    ];
    println!(
        "    THE HILL SPACING — the mean distance between the crests a straight walk along the grid crosses, in km:\n    (a) {:.2}, (b) {:.2}, (c) {:.2}; the table's Rice spacing {:.2} (a dune field reads ONE spacing of its order; (b) reads 16 km or more)",
        spacing[0].1, spacing[1].1, spacing[2].1, table.rice_km
    );
    for (tag, km) in spacing {
        census(&format!("G-HILL-SPACING-km:{}:{tag}", aim.slug), km, &none);
    }
    println!("  the stand took {:.1} s", t0.elapsed().as_secs_f64());
}

/// What one column reads beside the chunk builder's own answer.
struct Pair {
    /// The shared reader's surface at the SHIPPED first octave — `column_field` must answer the
    /// same word.
    shipped: Gi,
    /// The surface with the octave sum EMPTY: the solve, the bench and the shore alone.
    solve: Gi,
    /// The water radius the column reads (ZERO for none).
    water: Gi,
    /// Whether the column stands under its water.
    wet: bool,
    /// The roughness factor the fine octaves are multiplied by, at the noise's fraction bits.
    factor: Gi,
}

/// ★ ONE COLUMN THROUGH THE SHARED READER (`shore_geometry::RungReader::column`: the chunk
/// builder's reads in the chunk builder's order, then the ONE kernel), plus the two words that
/// reader does not return — `Z` and the slope share, read by the same public functions on the same
/// lattice — so the kernel can run a second time with the octave sum empty. `None` where the field
/// does not hold the column's stencil (then the chunk builder builds no chunk either).
///
/// A column is WET where the side word the shore law obeys says so; where no side is stated, the
/// ground's own sign decides (`ShippedColumn::ground_dry`: the base at or over the water is land),
/// which is the rule the shore law itself runs (`vd_recipe::height::shore`, `g >= 0`). A body with
/// no water (the water word ZERO) is dry everywhere by the same rule.
fn read_pair(reader: &RungReader, i: i32, j: i32, none: usize) -> Option<Pair> {
    let col = reader.column(i, j)?;
    let field = reader.source.field();
    let z = sample_z(&reader.lattice, field, reader.face, reader.rung, i, j)?;
    let share = slope_share(
        &reader.slope,
        &reader.lattice,
        field,
        reader.face,
        reader.rung,
        i,
        j,
    )?;
    // The same read with the octave sum EMPTY: `first` at the rung's count sums nothing.
    let bare = FieldRead {
        z,
        first: none,
        slope_share: share,
        water: col.water,
        side: col.side_word,
    };
    let wet = col.drawn_wet().unwrap_or_else(|| !col.ground_dry());
    Some(Pair {
        shipped: col.surface,
        solve: column_surface_at(&reader.charter, col.dir, &bare).h,
        water: col.water,
        wet,
        factor: roughness_factor(&reader.charter.roughness, col.dir).max(share),
    })
}

/// Where the square stands, so a basin can be named by its cell and its aim.
struct Place<'a> {
    face: Face,
    i0: i32,
    j0: i32,
    w: usize,
    rung: u8,
    charter: &'a PlanCharter,
    cell_km2: f64,
}

impl Place<'_> {
    /// The rung's cell and the unit direction of grid index `at`.
    fn locate(&self, at: usize) -> (i32, i32, DVec3) {
        let i = self.i0 + (at % self.w) as i32;
        let j = self.j0 + (at / self.w) as i32;
        let dir = site_direction(self.charter, i32::from(self.face.index()), i, j);
        (i, j, unit_of(dir))
    }
}

/// What one variant hands back for the comparison of (a) with (b).
struct Reading {
    breach_m3: f64,
    deep10: usize,
}

/// ★ THE GATES ON ONE VARIANT: G-HACK, G-BREACH and the spill census.
fn variant(f: &Heights, title: &str, slug: &str, tag: &str, place: &Place) -> Reading {
    println!("  ── {title} ──");
    let (c, n, points) = g_hack(f);
    println!(
        "    G-HACK   L = {c:.3}·a^{n:.3} (L in km, a in km²) over {points} network cells (a basin of 10 cells or more, {:.3} km²; no wet cell fitted)",
        10.0 * f.cell_m * f.cell_m / 1.0e6
    );
    println!(
        "             the bands: c in [{}, {}], n in [{}, {}] (Sassolas-Serrayet, Cattin, Ferry & Godard 2018); fewer than 2 network cells reads 0 and 0",
        HACK_C.0, HACK_C.1, HACK_N.0, HACK_N.1
    );
    census(
        &format!("G-HACK-c:{slug}:{tag}"),
        c,
        &Band::Range(HACK_C.0, HACK_C.1, HACK_SOURCE),
    );
    census(
        &format!("G-HACK-n:{slug}:{tag}"),
        n,
        &Band::Range(HACK_N.0, HACK_N.1, HACK_SOURCE),
    );
    census(
        &format!("G-HACK-cells:{slug}:{tag}"),
        points as f64,
        &Band::NoBand("a_count"),
    );
    let (breach, pits) = g_breach(f);
    println!(
        "    G-BREACH {breach:.3e} m³ a pit: the mean a one-cell-wide channel must remove, over {pits} pits (a pit has no lower\n             neighbour; a flat cell counts and removes nothing; the unit of Schott et al. 2024 is UNVERIFIED, so no band)"
    );
    census(
        &format!("G-BREACH-m3:{slug}:{tag}"),
        breach,
        &Band::NoBand("Schott_2024_unit_unverified"),
    );
    let spill = spill_census(f);
    report_spill(&spill, slug, tag, place);
    Reading {
        breach_m3: breach,
        deep10: spill.at_least(10.0),
    }
}

/// ★ THE SPILL CENSUS, PRINTED: the counts on each 100 km² of dry ground, the filled share, the
/// histogram, the census's own check, and the deepest basins by cell and aim.
fn report_spill(s: &Spill, slug: &str, tag: &str, place: &Place) {
    let dry_km2 = s.dry as f64 * place.cell_km2;
    let per_100 = |count: usize| count as f64 * 100.0 / dry_km2;
    let filled_pct = s.flooded as f64 * 100.0 / s.dry.max(1) as f64;
    // The same share with the 1 m floor of `G-SPILL-1m`: a basin one word deep is the arithmetic's
    // own step, and a drained field may hold many such steps, so the floored share is the one a
    // drained field must read near zero.
    let filled_1m_pct = s.flooded_at_least(1.0) as f64 * 100.0 / s.dry.max(1) as f64;
    println!(
        "    SPILL    {} closed basins on {dry_km2:.0} km² of dry ground (a flood from the rim and from every wet column);\n             their fill covers {filled_pct:.3} % of the dry ground, and the fills of the basins 1 m deep or deeper {filled_1m_pct:.3} %",
        s.basins.len()
    );
    let none = Band::NoBand(NO_PUBLISHED);
    for (name, depth) in [("1m", 1.0), ("10m", 10.0), ("100m", 100.0)] {
        let count = s.at_least(depth);
        println!(
            "             {count:>7} basins {depth:>3.0} m deep or deeper: {:.3} on each 100 km²",
            per_100(count)
        );
        census(
            &format!("G-SPILL-{name}-per100km2:{slug}:{tag}"),
            per_100(count),
            &none,
        );
    }
    census(&format!("G-FILLED-pct:{slug}:{tag}"), filled_pct, &none);
    census(
        &format!("G-FILLED-1m-pct:{slug}:{tag}"),
        filled_1m_pct,
        &none,
    );
    let labels: [&str; DEPTH_CLASSES] = [
        "under 0.1 m",
        "0.1–1 m",
        "1–10 m",
        "10–100 m",
        "100–1 000 m",
        "1 000 m or more",
    ];
    let line: Vec<String> = labels
        .iter()
        .zip(s.histogram().iter())
        .map(|(label, count)| format!("{label}: {count}"))
        .collect();
    println!(
        "             the depth histogram, in basins: {}",
        line.join(" | ")
    );
    let mut deep: Vec<&Basin> = s.basins.iter().filter(|b| b.depth_m >= 1.0).collect();
    if !deep.is_empty() {
        let mut depths: Vec<f64> = deep.iter().map(|b| b.depth_m).collect();
        let mut areas: Vec<f64> = deep
            .iter()
            .map(|b| b.cells as f64 * place.cell_km2)
            .collect();
        depths.sort_by(f64::total_cmp);
        areas.sort_by(f64::total_cmp);
        println!(
            "             the basins 1 m deep or deeper: median depth {:.1} m, median area {:.3} km², largest {:.1} km²",
            quantile_sorted(&depths, 0.5),
            quantile_sorted(&areas, 0.5),
            quantile_sorted(&areas, 1.0)
        );
    }
    println!(
        "             the census's own check: {} basins whose cells disagree on the fill level (must be 0)",
        s.split
    );
    census(
        &format!("SPILL-SPLIT:{slug}:{tag}"),
        s.split as f64,
        &Band::Range(0.0, 0.0, "own_check:one_fill_level_a_basin"),
    );
    deep.sort_by(|p, q| q.depth_m.total_cmp(&p.depth_m));
    for basin in deep.iter().take(DEEPEST_SHOWN) {
        let (i, j, aim) = place.locate(basin.pit);
        println!(
            "             {:>7.1} m deep, {:>8.3} km², {:.2e} m³ — the pit at cell ({i}, {j}) of face {:?} at rung {}; aim {:.6} {:.6} {:.6}",
            basin.depth_m,
            basin.cells as f64 * place.cell_km2,
            basin.volume_m3,
            place.face,
            place.rung,
            aim.x,
            aim.y,
            aim.z
        );
    }
}

/// ★ THE HILL SPACING: the mean distance, in km, between the crests a straight walk along the grid
/// crosses — a crest being a column higher than the column before it and at least as high as the
/// one after it, along each of the grid's two axes. A step counts only where the column and both
/// its neighbours on that axis stand on dry ground, so a coast's drop to the sea floor never reads
/// as a crest. A dune field has ONE spacing and a real landscape has many; the number says which
/// spacing the eye sees most. `inf` where the walk crosses no crest.
fn hill_spacing_km(h: &[i64], dry: &[bool], w: usize, dx: f64, dy: f64) -> f64 {
    let (mut length_m, mut crests) = (0.0f64, 0usize);
    for y in 1..w - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if dry[i - 1] & dry[i] & dry[i + 1] {
                length_m += dx;
                crests += usize::from((h[i - 1] < h[i]) & (h[i] >= h[i + 1]));
            }
            if dry[i - w] & dry[i] & dry[i + w] {
                length_m += dy;
                crests += usize::from((h[i - w] < h[i]) & (h[i] >= h[i + w]));
            }
        }
    }
    if crests == 0 {
        return f64::INFINITY;
    }
    length_m / crests as f64 / 1_000.0
}

/// ★ A STAND THE INSTRUMENT COULD NOT READ still prints its path check, as `NaN`, which is RED under
/// the `[0,0]` band: a script that greps the run sees the missing stand as a failure, never as a gap.
fn unread_stand(slug: &str) {
    census(
        &format!("SHIPPED-PATH-DIFFER:{slug}"),
        f64::NAN,
        &Band::Range(0.0, 0.0, PATH_CHECK),
    );
}

/// ★ ONE `CENSUS` LINE: the gate's name (its unit in the name), the value, the band word with the
/// band's source in brackets, and the verdict. The verdict is COMPUTED from the value and the band,
/// never written: inside the inclusive band is GREEN, outside is RED, and a value the instrument
/// could not make (`NaN`) is RED, because `contains` holds for no `NaN`. A reading with no band
/// prints `none(<why>) NO-BAND`.
fn census(name: &str, value: f64, band: &Band) {
    match band {
        Band::Range(lo, hi, source) => {
            let verdict = if (*lo..=*hi).contains(&value) {
                "GREEN"
            } else {
                "RED"
            };
            println!("CENSUS {name} {value:.4e} [{lo},{hi}]({source}) {verdict}");
        }
        Band::NoBand(why) => println!("CENSUS {name} {value:.4e} none({why}) NO-BAND"),
    }
}
