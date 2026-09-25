//! ★ THE TEETH NUMBER — HOW FAR THE DRAWN SHORE STANDS FROM THE GROUND'S OWN WATERLINE, HOW TALL THE
//! WALL ON IT IS, AND WHICH WAY IT RUNS (plan step 1 of
//! `docs/investigation/2026-09-24/terrain_believability_status.md`; ruling B6, 2026-09-24).
//!
//! **THE OWNER'S WORDS.** *"shores are squared like teeth"* (the flight of 2026-09-24).
//!
//! **WHY IT EXISTS.** No gate saw the teeth (the status file, §4 "THE BLIND SPOTS"). The causes are
//! read in the code (§6): the side of the water is the NEAREST fine node's word at rungs 0–13 (T1),
//! the shore law obeys that word and not the ground (T2), and a lake's edge has the same fault and
//! one more (T4). So the drawn shoreline is the edge of an 8 km node square, a bank stands on it, and
//! it turns 90° at every node corner. This instrument turns that picture into numbers that CAN GO
//! RED. It changes nothing in the world.
//!
//! **WHAT IT READS — THE SHIPPED PATH, NOT A COPY OF IT.** For every column of a square it reads
//! what the chunk builder reads (`vd_bins::shore_geometry::RungReader`, the steps of
//! `vd_terrain::chunk`'s `ColumnRead::column`): the SIDE (`sample_side` on the fine lattice), the
//! WATER that side names (`column_water`), the GROUND BASE the shore law reads (`radius + z +
//! coarse`, as `vd_recipe::plan::column_surface_at` hands it to `vd_recipe::height::shore`), and
//! the final SURFACE from the ONE kernel. Two checks run on every square:
//! * `G-TEETH-SELFCHECK` — one whole chunk of the square's middle is built by
//!   `vd_terrain::chunk::column_field` itself, and every column's surface and water must equal this
//!   reader's, bit for bit. This is the proof that the reader IS the shipped path.
//! * `G-TEETH-BASECHECK` — a DRIFT ALARM: the base, the fine half, the bench and the shore law must
//!   give the kernel's own surface again in every column. The reader runs the kernel's own last
//!   lines again, so this proves no more than that; it goes RED the day the kernel builds its
//!   surface from another base, and then every offset of the square is void.
//!
//! **WHAT IT PRINTS**, per stand, per rung (default 0, 5, 7, 9, 10, 13; a column is `2^rung` metres
//! at the face's middle, `vd_seed::ladder::cell_m`, and the square's own chord is printed), and
//! SEPARATELY for the SEA's shore and for the LAKES' edges:
//! 1. **THE OFFSET.** For each pair of neighbouring columns where the drawn side changes, the
//!    distance in metres along the same row or column to the nearest place where the ground's own
//!    sign (`base − water`, with the WET column's water) changes. A shore that follows its ground
//!    reads at most half a column. Where the search reaches the square's end (or a column the field
//!    did not hold) before a crossing, the reading is a LOWER BOUND and is counted as one.
//! 2. **THE WALL.** The surface step across that pair against the median step between all other
//!    neighbouring columns of the square (the slope's own step): the median ratio, and how many
//!    pairs stand over ten times it. Beside it, the shore law's FORCED FLOOR for the pair — the
//!    least wall `vd_recipe::height::shore` allows, `(w_dry + |g_dry|/4) − (w_wet − |g_wet|/4)`,
//!    never under zero, each `g` the column's base over its OWN water (T2's prediction). On the
//!    sea's shore both waters are the sea and the floor is `(|g_dry| + |g_wet|)/4`; beside a lake
//!    at rungs 0–9 the dry column holds the SEA (T4), and the floor is mostly zero.
//! 3. **THE DIRECTION.** The drawn shore and the ground's own waterline traced as lines, cut into
//!    windows of a stated length (default 2 km), each window's bearing histogrammed against the face
//!    grid's axes, and the flatness gate G-GRID already reads (`vd_bins::grid_bias::AxisHistogram`).
//!    A square coast reads high; a natural coast reads near one.
//! 4. **THE DISAGREEMENT.** The columns the side word holds on the OTHER side of the water from their
//!    own ground, and how far from THAT water's drawn shore they reach — inland (the water drawn
//!    over dry ground) and offshore (the land drawn over drowned ground). The SEA judges every
//!    column but a lake's against the body's sea. A LAKE judges its own columns against their own
//!    level, and a land column against its own lake word where it holds one (a far rung's level-1
//!    word, T5), else against the nearest drawn lake's level within two fine nodes (T4: at rungs 0–9
//!    its own word is the sea, so nothing holds it over the lake).
//!
//! **THE CENSUS LINES**, one per gate, for a script to grep:
//! `CENSUS <GATE-NAME> <value> <band> <GREEN|RED|NO-BAND>`, the gate name carrying the water, the
//! stand and the rung (`G-TEETH-OFFSET.sea.coast.r07`). The bands and their sources:
//! * OFFSET — the largest offset at most HALF A COLUMN. Source: ruling W6 (the shore law: "a
//!   column's surface is held on its GROUND's side of its water — the ground being the solved field
//!   plus the coarse octaves", which is the base read here; `vd_recipe::height::shore`: "The shore
//!   is where the ground crosses the water's level") and ruling B6 item 1 step (4) ("one shoreline
//!   from one field"), which the status file's plan row 4 states as "from where the ground crosses
//!   the water's level, not from a node square". A side that follows the ground's sign changes
//!   between the same two columns as the ground, so every offset is at most half a column, the
//!   largest included. The verdict: RED where the largest reading (exact, or a lower bound) is over
//!   the band; GREEN where it is inside and no pair is a bound; NO-BAND where it is inside but a pair
//!   is only a bound (undecided).
//! * GRID and GRID-GROUND — the flatness at four bins at most the 95th percentile of a histogram of
//!   the same count with NO direction of its own, by simulation. Source: ruling B3 (G-GRID: "the
//!   long-axis histogram … is flat against the grid's directions"; Hyväluoma 2017) and the method of
//!   ruling W13.3 (a perfect router 1.006, the 95th 1.012 at 96 804 axes — this run prints the same
//!   simulation at that count first, `G-TEETH-NULLCHECK`). Judged only on a square of at least four
//!   fine nodes with at least sixteen windows (stated choices); a smaller reading prints NO-BAND.
//! * WALL, WALL10, DISAGREE, REACH, NULLCHECK — no published band yet: printed with NO-BAND.
//!
//! **WHAT THE TEETH SHOULD PRINT.**
//! * TODAY (predicted from the status file §6 and ruling W13.3, UNMEASURED). At rungs 0–9 the sea's
//!   drawn shore stands up to half a node (about 4 km) from the ground's own waterline, far over the
//!   half-column band: OFFSET RED. The wall is several times the slope's own step, near the shore
//!   law's floor. The drawn shore's windows pile into the 0° and 90° bins (a flatness near two),
//!   while the ground's own waterline in the same square reads near one: GRID RED, GRID-GROUND
//!   near its band. At rungs 10–13 the base is a level-1 mean, so the disagreeing columns reach up
//!   to a whole level-1 square (about 16 km). The lakes read the same, and at rungs 10–13 most small
//!   lakes have no edge at all (T5).
//! * AFTER THE CURE (plan step 4: one shoreline from one field). OFFSET GREEN at every rung (at most
//!   half a column), the wall's median ratio near one, the drawn shore's flatness equal to the
//!   ground's and inside the band, and no column on the other side of the water from its ground.
//!
//! `cargo run --release -p vd-bins --example shore_teeth -- [side_m] [window_m] [max_columns]
//! [dx dy dz]` — the rungs from `VD_TEETH_RUNGS` (default `0,5,7,9,10,13`). With no direction the
//! stands are THE COAST (the pinned one `vd_bins::drainage_gates::pick_stands` states for every
//! instrument) and THE LAKE's edge (pinned: the edge the largest lake had at generator 20, review
//! finding 9 of plan step 2b); each square is centred on the node edge where the side word changes.
//! The run solves the home planet first (about two minutes, as `lake_census` does).
//!
//! **Example.** A pilot at 120 km over the belt's coast draws rung 7, one column every 128 m. The
//! coast mask says LAND for the whole node square beside the sea, so the drawn shore runs along the
//! node's edge for 8 km and turns 90°. The ground's own waterline curves 3 km offshore of it. The
//! teeth are PREDICTED (unmeasured) to print `CENSUS G-TEETH-OFFSET.sea.coast.r07 3000.0m
//! <=0.50col(64.0m) RED` and a drawn flatness near two. After the cure the same stand prints an
//! offset under 64 m, and the drawn line reads what its ground reads.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_bins::drainage_gates::{cell_of, pick_stands};
use vd_bins::grid_bias::AxisHistogram;
use vd_bins::shore_geometry::{
    NodeEdge, RungReader, ShippedColumn, flat_null_quantile, flat_null_sorted, lake_edge,
    quantile_sorted, sea_edge, unit_of, window_bearings, zero_contours,
};
use vd_core::glam::DVec3;
use vd_recipe::Gi;
use vd_recipe::height::SHORE_SHIFT;
use vd_recipe::plan::site_direction;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, CoastCounts, FACIES_SHIFT};
use vd_terrain::chunk::{CHUNK_EDGE, column_field};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::metres_of_q28;
use vd_terrain::solve::{FACIES_LAKE, FACIES_SEA};

/// The rungs read when `VD_TEETH_RUNGS` states none: the boots (0), the near ground (5), the
/// owner's 100–150 km view (7 and 9), the first pyramid rung (10), and the last rung whose side is
/// still one fine node (13).
const DEFAULT_RUNGS: [u8; 6] = [0, 5, 7, 9, 10, 13];

/// The square's stated side in metres when none is given: sixteen nodes of 8 192 m, about the view
/// the owner judged from 120 km.
const SIDE_M: f64 = 131_072.0;

/// The window's stated length in metres when none is given (the task's 2 km).
const WINDOW_M: f64 = 2_000.0;

/// The most columns a square holds on a side when none is given: at the fine rungs the stated side
/// holds more columns than a run can read, so the square shrinks to this many (1 024 columns at
/// rung 0 is 1 km).
const MAX_COLUMNS: usize = 1_024;

/// The fewest columns a square holds on a side: at the far rungs the stated side holds too few
/// columns to trace a line (sixteen at rung 13), so the square grows to this many.
const MIN_COLUMNS: usize = 128;

/// ★ THE SHORTEST WINDOW, in columns. A window of a few columns can only read the stencil's own
/// four directions, which says nothing — the reason gate G-GRID reads a trunk over eight receiver
/// steps (`examples/lake_census.rs`, `TRUNK_STEPS`). So a window is never shorter than eight
/// columns, whatever length is stated.
const MIN_WINDOW_COLUMNS: f64 = 8.0;

/// The fewest windows a direction reading is judged on, and the fewest fine nodes the square must
/// span: a square narrower than four nodes holds too few node corners to show a tooth's turn. Both
/// are STATED choices, not published bands: under them the reading prints NO-BAND.
const MIN_WINDOWS: usize = 16;
const MIN_SPAN_NODES: f64 = 4.0;

/// The draws of the flat null per count.
const TRIALS: usize = 2_000;

/// The wall's stated multiple: a step over ten times the slope's own step is counted.
const WALL_TIMES: f64 = 10.0;

/// The OFFSET band in columns: half a column (see the header's sources), and the float slack of
/// the linear interpolation that places a crossing inside a pair.
const OFFSET_BAND_COLS: f64 = 0.5;
const OFFSET_SLACK_COLS: f64 = 1.0e-9;

/// Ruling W13.3's count of axes and its two readings of a perfect router at four bins.
const W133_AXES: usize = 96_804;
const W133_MEDIAN: f64 = 1.006;
const W133_P95: f64 = 1.012;

/// Which water a shore belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Water {
    Sea,
    Lake,
}

impl Water {
    fn name(self) -> &'static str {
        match self {
            Water::Sea => "sea",
            Water::Lake => "lake",
        }
    }
}

/// One stand: a name for the census lines, the face, and the square's centre in rung-0 cells.
struct Stand {
    name: &'static str,
    face: Face,
    centre0: (i64, i64),
}

fn main() {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let side_m = a.first().copied().unwrap_or(SIDE_M);
    let window_m = a.get(1).copied().unwrap_or(WINDOW_M);
    let cap = a.get(2).map_or(MAX_COLUMNS, |v| *v as usize).max(2);
    let stated = (a.len() >= 6).then(|| DVec3::new(a[3], a[4], a[5]).normalize());
    let rungs: Vec<u8> = std::env::var("VD_TEETH_RUNGS").ok().map_or_else(
        || DEFAULT_RUNGS.to_vec(),
        |v| {
            v.split(',')
                .map(|r| r.trim().parse::<u8>().expect("a rung"))
                .collect()
        },
    );
    println!(
        "shore_teeth: a stated side of {side_m:.0} m, windows of {window_m:.0} m, at most {cap} columns a side, rungs {rungs:?}"
    );

    // ★ THE BAND'S OWN METHOD, FIRST: the flat null at ruling W13.3's own count. It must read what
    // W13.3's simulation read, or every GRID band below is a different band. No published band
    // holds this reading, so it prints NO-BAND beside W13.3's two numbers for a reader to compare.
    let null = flat_null_sorted(W133_AXES, 4, TRIALS);
    let (median, p95) = (quantile_sorted(&null, 0.5), quantile_sorted(&null, 0.95));
    println!(
        "the band's method: {W133_AXES} axes with no direction of their own, at 4 bins, over {TRIALS} draws: the flatness's median {median:.4}x and 95th percentile {p95:.4}x (ruling W13.3, its own simulation: {W133_MEDIAN}x and {W133_P95}x)"
    );
    census(
        "G-TEETH-NULLCHECK",
        &format!("{p95:.4}x"),
        &format!("W13.3={W133_P95}x"),
        "NO-BAND",
    );

    let started = Instant::now();
    let artifact = run_solve(&SolveJob {
        body: home_planet(),
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let fine = body.macro_lattice().expect("a macro lattice");
    let counts = artifact.coast_counts();
    println!(
        "solved in {:.1} s; the sea stands at {} m; the fine node is {:.0} m ({} rung-0 cells)",
        started.elapsed().as_secs_f64(),
        artifact.sea().unwrap_or(0),
        fine.node_m(),
        fine.cells_per_node
    );

    let stands = match stated {
        Some(dir) => {
            let (face, i, j) = cell_of(&body, dir, 0);
            println!(
                "the stand: the stated direction {:.6} {:.6} {:.6}, rung-0 cell ({i}, {j}) of face {face:?}",
                dir.x, dir.y, dir.z
            );
            vec![Stand {
                name: "stated",
                face,
                centre0: (i64::from(i), i64::from(j)),
            }]
        }
        None => node_stands(&body, &artifact, &fine),
    };

    let mut bands: BTreeMap<usize, f64> = BTreeMap::new();
    for stand in &stands {
        for &rung in &rungs {
            measure(
                &body, &artifact, &counts, stand, rung, side_m, window_m, cap, &mut bands,
            );
        }
    }
    legend();
}

/// ★ THE LAKE STAND, PINNED (review finding 9 of plan step 2b; ruling B6: every step is compared on
/// the SAME stands): the edge the largest lake had at generator 20 — its WET node (256, 674) and
/// the DRY node (257, 674) beside it on face `NegX`. The largest lake grew at generator 21 (6 398 →
/// 12 562 nodes, in another place), so a stand that follows "the largest" reads other ground at
/// each step. The largest lake of this run is still found and printed, as a reading.
///
/// **Example.** Step 4 draws a lake's edge where the ground crosses its level. The teeth of this
/// lake's edge are read before and after it, whichever lake is then the largest.
const LAKE_STAND: (Face, (i32, i32), (i32, i32)) = (Face::NegX, (257, 674), (256, 674));

/// ★ THE TWO STANDS, both PINNED: THE COAST, as `pick_stands` states it for every instrument, and
/// THE LAKE's edge ([`LAKE_STAND`]). Each square is centred on the node edge where the side word
/// changes.
fn node_stands(body: &BodyDefinition, artifact: &Artifact, fine: &MacroLattice) -> Vec<Stand> {
    let mut out = Vec::new();
    let picked = pick_stands(artifact, fine);
    match picked.iter().find(|s| s.name == "the coast") {
        Some(coast) => {
            let (face, i, j) = cell_of(body, coast.dir, 0);
            let (ni, nj) = fine.node_of_cell(0, i, j);
            let node = fine.index(face, ni, nj);
            match sea_edge(artifact, fine, node) {
                Some(edge) => {
                    describe(&format!("the coast (node {node})"), &edge, fine);
                    out.push(stand_of("coast", &edge, fine));
                }
                None => println!(
                    "the coast: REFUSED — node {node} has no sea node beside it on its own face"
                ),
            }
        }
        None => println!("the coast: REFUSED — the shared pick names no coast"),
    }
    let (face, dry, wet) = LAKE_STAND;
    let edge = NodeEdge {
        face,
        dry,
        wet,
        discharge: artifact.rows[fine.index(face, dry.0, dry.1) as usize].discharge_log,
    };
    let facies = |(i, j): (i32, i32)| {
        artifact.rows[fine.index(face, i, j) as usize].receiver_facies >> FACIES_SHIFT
    };
    let lake_here = (facies(wet) & FACIES_LAKE != 0) & (facies(wet) & FACIES_SEA == 0);
    let dry_here = facies(dry) & (FACIES_LAKE | FACIES_SEA) == 0;
    describe(
        &format!(
            "the lake (PINNED: generator 20's largest; the wet node is a lake today: {lake_here}, the dry node is dry today: {dry_here})"
        ),
        &edge,
        fine,
    );
    out.push(stand_of("lake", &edge, fine));
    match lake_edge(artifact, fine, 1) {
        Some((edge, size)) => describe(
            &format!("(a reading, not a stand) the largest lake of this run ({size} nodes)"),
            &edge,
            fine,
        ),
        None => println!("(a reading) the solve left no lake with a dry side"),
    }
    out
}

/// A stand from a node edge: centred on the edge's middle.
fn stand_of(name: &'static str, edge: &NodeEdge, fine: &MacroLattice) -> Stand {
    Stand {
        name,
        face: edge.face,
        centre0: edge.middle(fine.cells_per_node),
    }
}

/// Prints a stand's node edge and the direction to feed `land_stand`.
fn describe(what: &str, edge: &NodeEdge, fine: &MacroLattice) {
    let dry = fine.index(edge.face, edge.dry.0, edge.dry.1);
    let d = unit_of(fine.direction(dry));
    println!(
        "  stand {what}: the DRY node ({}, {}) beside the WET node ({}, {}) on face {:?}; discharge class {}; aim {:.6} {:.6} {:.6} (feed it to `land_stand -- 2000 dx dy dz`)",
        edge.dry.0, edge.dry.1, edge.wet.0, edge.wet.1, edge.face, edge.discharge, d.x, d.y, d.z
    );
}

/// One census line: `CENSUS <GATE-NAME> <value> <band> <verdict>`.
fn census(name: &str, value: &str, band: &str, verdict: &str) {
    println!("CENSUS {name} {value} {band} {verdict}");
}

/// A sorted copy's quantile.
fn q_of(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    quantile_sorted(&v, q)
}

/// ★ ONE SQUARE: the columns read, the two self-checks, then the four readings for each water.
#[allow(
    clippy::too_many_arguments,
    reason = "an instrument's knobs, each stated once"
)]
fn measure(
    body: &BodyDefinition,
    artifact: &Artifact,
    counts: &std::sync::Arc<CoastCounts>,
    stand: &Stand,
    rung: u8,
    side_m: f64,
    window_m: f64,
    cap: usize,
    bands: &mut BTreeMap<usize, f64>,
) {
    let tag = format!("{}.r{rung:02}", stand.name);
    if rung >= body.ladder().rungs {
        println!(
            "\n== {tag}: REFUSED — the body has rungs 0 to {} only",
            body.ladder().rungs.saturating_sub(1)
        );
        return;
    }
    let Some(reader) = RungReader::new(body, artifact, counts, stand.face, rung) else {
        println!("\n== {tag}: REFUSED — no field at this rung");
        return;
    };
    let n_cells = i64::from(body.ladder().cells_per_edge(rung));
    // The rung-0 cells one column of this rung covers: `2^rung`, which is also the column's width
    // in metres at the face's middle (a rung-0 cell is one metre there).
    let cells0 = f64::from(vd_seed::ladder::cell_m(rung));
    let want = ((side_m / cells0).round() as usize).max(MIN_COLUMNS);
    let n = want.min(cap).min(n_cells as usize);
    let n = n - n % 2;
    if n < 2 {
        println!("\n== {tag}: REFUSED — the face holds too few columns at this rung");
        return;
    }
    let ni = n as i64;
    let (ci, cj) = (stand.centre0.0 >> rung, stand.centre0.1 >> rung);
    // The square is centred on the stand; where it would leave the face it is SHIFTED inward, so
    // it still holds the stand and every column stands on one face.
    let i0 = (ci - ni / 2).clamp(0, n_cells - ni);
    let j0 = (cj - ni / 2).clamp(0, n_cells - ni);
    let shift = (i0 - (ci - ni / 2), j0 - (cj - ni / 2));

    let t = Instant::now();
    let mut cols: Vec<Option<ShippedColumn>> = Vec::with_capacity(n * n);
    for b in 0..ni {
        for a in 0..ni {
            cols.push(reader.column((i0 + a) as i32, (j0 + b) as i32));
        }
    }
    let read_s = t.elapsed().as_secs_f64();

    // The metres one column spans along each face axis, at the square's middle: the chord between
    // two neighbouring columns' own directions, times the ladder radius.
    let radius = body.radius_m();
    let (cx, cy) = (i0 + ni / 2, j0 + ni / 2);
    let dir_at = |i: i64, j: i64| {
        unit_of(site_direction(
            &reader.charter,
            i32::from(stand.face.index()),
            i as i32,
            j as i32,
        ))
    };
    let m_i = (dir_at(cx + 1, cy) - dir_at(cx, cy)).length() * radius;
    let m_j = (dir_at(cx, cy + 1) - dir_at(cx, cy)).length() * radius;
    let m_col = 0.5 * (m_i + m_j);
    let span_nodes = n as f64 * cells0 / f64::from(reader.fine.cells_per_node);
    println!(
        "\n== {tag}: rung {rung} ({cells0:.0} m cells at the face's middle), pyramid level {}; a square of {n} x {n} columns = {:.1} km a side ({span_nodes:.1} fine nodes), from cell ({i0}, {j0}) of face {:?}, shifted by ({}, {}) columns to stay on the face; a column is {m_i:.1} m along i and {m_j:.1} m along j; read in {read_s:.1} s",
        reader.level,
        n as f64 * m_col / 1_000.0,
        stand.face,
        shift.0,
        shift.1
    );

    self_checks(body, &reader, &cols, &tag, (cx, cy), n_cells);

    // The square's census of the drawn side.
    let (mut land, mut sea, mut lake, mut unknown, mut missing) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for c in &cols {
        match c {
            None => missing += 1,
            Some(c) => match c.drawn_wet() {
                None => unknown += 1,
                Some(false) => land += 1,
                Some(true) if c.lake() => lake += 1,
                Some(true) => sea += 1,
            },
        }
    }
    println!(
        "   the square, by the side word the shore law obeys: {land} land, {sea} sea, {lake} lake, {unknown} with no stated side, {missing} not read (columns)"
    );

    let square = Square {
        n,
        cols: &cols,
        m_i,
        m_j,
        sea_radius: reader.sea_radius(),
    };
    let edges = square.edges();
    let slope_step = q_of(&edges.steps, 0.5);
    let window_cols = (window_m / m_col).max(MIN_WINDOW_COLUMNS);
    let node_cols = (reader.fine.cells_per_node >> rung).max(1) as usize;
    // The lake level every column is judged against: its own lake's, its own lake word, or the
    // nearest drawn lake's within two fine nodes.
    let lake_levels = square.lake_levels(2 * node_cols);
    let t2 = if reader.level == 0 {
        format!("T2: up to half a node, {:.0} m", reader.fine.node_m() / 2.0)
    } else {
        format!(
            "T2: up to a whole level-{} square, {:.0} m",
            reader.level,
            reader.fine.node_m() * f64::from(1u32 << reader.level)
        )
    };
    for water in [Water::Sea, Water::Lake] {
        let gate = |g: &str| format!("G-TEETH-{g}.{}.{tag}", water.name());
        let mine: Vec<&Edge> = edges.list.iter().filter(|e| e.water == water).collect();
        let along_i = mine.iter().filter(|e| !e.along_j).count();
        println!(
            "   the {} {}: {} side-change pairs ({} across i, {} across j)",
            if water == Water::Sea {
                "SEA's"
            } else {
                "LAKES'"
            },
            if water == Water::Sea {
                "shore"
            } else {
                "edges"
            },
            mine.len(),
            along_i,
            mine.len() - along_i
        );

        // 1. THE OFFSET.
        let band_m = OFFSET_BAND_COLS * m_col;
        let band = format!("<={OFFSET_BAND_COLS:.2}col({band_m:.1}m)");
        if mine.is_empty() {
            census(&gate("OFFSET"), "none", &band, "NO-BAND");
        } else {
            let off_m: Vec<f64> = mine.iter().map(|e| e.offset_m).collect();
            let off_cols: Vec<f64> = mine.iter().map(|e| e.offset_cols).collect();
            let censored = mine.iter().filter(|e| e.censored).count();
            let max_m = q_of(&off_m, 1.0);
            let max_cols = q_of(&off_cols, 1.0);
            println!(
                "     1. THE OFFSET from the drawn shore to the ground's own waterline, along the pair's row or column: median {:.1} m ({:.2} columns), p90 {:.1} m, max {max_m:.1} m ({max_cols:.2} columns); {censored} pairs are LOWER BOUNDS (no crossing before the square's end or a column not read)",
                q_of(&off_m, 0.5),
                q_of(&off_cols, 0.5),
                q_of(&off_m, 0.9)
            );
            println!(
                "        band: at most half a column — ruling W6 (the surface held on its GROUND's side of its water, the ground being the solved field plus the coarse octaves) and ruling B6 item 1 step (4) (one shoreline from one field); RED where the largest reading is over it, GREEN where it is inside and no pair is a bound, NO-BAND where a bound leaves it undecided"
            );
            // A lower bound over the band proves RED; only exact readings can prove GREEN.
            let verdict = if max_cols > OFFSET_BAND_COLS + OFFSET_SLACK_COLS {
                "RED"
            } else if censored == 0 {
                "GREEN"
            } else {
                "NO-BAND"
            };
            census(&gate("OFFSET"), &format!("{max_m:.1}m"), &band, verdict);
        }

        // 2. THE WALL.
        if mine.is_empty() {
            census(&gate("WALL"), "none", "no-band", "NO-BAND");
            census(&gate("WALL10"), "none", "no-band", "NO-BAND");
        } else {
            let walls: Vec<f64> = mine.iter().map(|e| e.wall_m).collect();
            let ratios: Vec<f64> = walls.iter().map(|w| w / slope_step.max(1.0e-9)).collect();
            let over = ratios.iter().filter(|&&r| r > WALL_TIMES).count();
            let floors: Vec<f64> = mine.iter().map(|e| e.floor_m).collect();
            let ratio = q_of(&ratios, 0.5);
            println!(
                "     2. THE WALL across the drawn shore: median {:.2} m, p90 {:.2} m, max {:.2} m; the slope's own step between the other neighbouring columns: median {slope_step:.3} m; the ratio median {ratio:.2}x, {over} of {} pairs over {WALL_TIMES:.0}x; the shore law's forced floor (w_dry + |g_dry|/4) - (w_wet - |g_wet|/4), never under zero (T2's prediction): median {:.2} m, p90 {:.2} m",
                q_of(&walls, 0.5),
                q_of(&walls, 0.9),
                q_of(&walls, 1.0),
                mine.len(),
                q_of(&floors, 0.5),
                q_of(&floors, 0.9)
            );
            census(&gate("WALL"), &format!("{ratio:.2}x"), "no-band", "NO-BAND");
            census(
                &gate("WALL10"),
                &format!("{over}of{}pairs", mine.len()),
                "no-band",
                "NO-BAND",
            );
        }

        // 3. THE DIRECTION.
        let drawn = square.drawn_field(water);
        let ground = match water {
            Water::Sea => square.sea_ground_field(),
            Water::Lake => square.lake_ground_field(&lake_levels),
        };
        let judged_span = span_nodes >= MIN_SPAN_NODES;
        let mut direction = |what: &str, field: &[Option<f64>], name: &str| {
            let mut four = AxisHistogram::new(4);
            let mut twelve = AxisHistogram::new(12);
            for line in zero_contours(n, field) {
                for d in window_bearings(&line, window_cols) {
                    four.add(d);
                    twelve.add(d);
                }
            }
            let count = four.total();
            let p95 = if count == 0 {
                f64::NAN
            } else {
                *bands
                    .entry(count)
                    .or_insert_with(|| flat_null_quantile(count, 4, TRIALS, 0.95))
            };
            let flat = four.flatness();
            println!(
                "        {what}: {count} windows | {} | flatness {flat:.3}x | the band at this count {p95:.3}x\n          12 bins: {} | flatness {:.3}x",
                four.line(),
                twelve.line(),
                twelve.flatness()
            );
            let band = format!("<={p95:.3}x");
            if count == 0 {
                census(name, "none", "no-band", "NO-BAND");
            } else if count < MIN_WINDOWS || !judged_span {
                census(name, &format!("{flat:.3}x"), &band, "NO-BAND");
            } else {
                let verdict = if flat <= p95 { "GREEN" } else { "RED" };
                census(name, &format!("{flat:.3}x"), &band, verdict);
            }
        };
        println!(
            "     3. THE DIRECTION over windows of {:.0} m ({window_cols:.1} columns), against the face grid (0° along i, 90° along j, 45° and 135° the diagonals); the band: at most the 95th percentile of a flat histogram of the same count, by simulation (ruling B3 G-GRID; the method of ruling W13.3); judged on {MIN_SPAN_NODES:.0}+ fine nodes and {MIN_WINDOWS}+ windows",
            window_cols * m_col
        );
        direction("the drawn shore", &drawn, &gate("GRID"));
        direction("the ground's own waterline", &ground, &gate("GRID-GROUND"));

        // 4. THE DISAGREEMENT, and how far each such column stands from THIS water's drawn shore.
        let (row_d, col_d) = square.reach(&drawn);
        let mut inland = 0usize;
        let mut offshore = 0usize;
        let mut reach_m: Vec<f64> = Vec::new();
        let mut reach_censored = 0usize;
        let mut depth_m: Vec<f64> = Vec::new();
        for (k, c) in cols.iter().enumerate() {
            let Some(c) = c else {
                continue;
            };
            let Some(wet) = c.drawn_wet() else {
                continue;
            };
            let Some(level) = square.judged_level(water, c, lake_levels[k]) else {
                continue;
            };
            let ground_dry = c.base >= level;
            // Drawn wet over drowned ground, or drawn dry over dry ground: the two agree.
            if wet != ground_dry {
                continue;
            }
            if wet {
                inland += 1;
            } else {
                offshore += 1;
            }
            let r = (row_d[k] * m_i).min(col_d[k] * m_j);
            if r.is_finite() {
                reach_m.push(r);
            } else {
                reach_censored += 1;
            }
            depth_m.push(metres_of_q28(c.base - level).abs());
        }
        let total = inland + offshore;
        println!(
            "     4. THE DISAGREEMENT: {total} columns ({:.3} % of the square) stand on the other side of the {} from their own ground: {inland} drawn wet over dry ground (the water reaches inland), {offshore} drawn dry over drowned ground (the land reaches offshore); how far from this water's drawn shore along a row or column: median {:.0} m, p90 {:.0} m, max {:.0} m, {reach_censored} with no such shore on their row or column ({t2}); how far their ground stands on the other side: median {:.1} m, max {:.1} m",
            total as f64 * 100.0 / (n * n) as f64,
            if water == Water::Sea {
                "sea"
            } else {
                "lake's level"
            },
            q_of(&reach_m, 0.5),
            q_of(&reach_m, 0.9),
            q_of(&reach_m, 1.0),
            q_of(&depth_m, 0.5),
            q_of(&depth_m, 1.0)
        );
        census(
            &gate("DISAGREE"),
            &format!("{total}cols"),
            "no-band",
            "NO-BAND",
        );
        let reach = if reach_m.is_empty() {
            "none".to_string()
        } else {
            format!("{:.1}m", q_of(&reach_m, 1.0))
        };
        census(&gate("REACH"), &reach, "no-band", "NO-BAND");
    }
}

/// ★ THE TWO SELF-CHECKS: one whole chunk from the chunk builder against this reader, and the base
/// against the kernel's own surface in every column of the square.
fn self_checks(
    body: &BodyDefinition,
    reader: &RungReader<'_>,
    cols: &[Option<ShippedColumn>],
    tag: &str,
    centre: (i64, i64),
    n_cells: i64,
) {
    let edge = CHUNK_EDGE as i64;
    let (kx, ky) = (centre.0 / edge, centre.1 / edge);
    if (kx + 1) * edge <= n_cells && (ky + 1) * edge <= n_cells {
        let built = column_field(
            body,
            Some(reader.source.field()),
            reader.face,
            reader.rung,
            kx as i32,
            ky as i32,
        );
        let mut equal = 0usize;
        let total = CHUNK_EDGE * CHUNK_EDGE;
        if let Some(built) = &built {
            for b in 0..CHUNK_EDGE {
                for a in 0..CHUNK_EDGE {
                    let k = b * CHUNK_EDGE + a;
                    let mine =
                        reader.column((kx * edge) as i32 + a as i32, (ky * edge) as i32 + b as i32);
                    if let Some(m) = mine
                        && m.surface == built.columns[k].1
                        && m.water == built.water[k]
                    {
                        equal += 1;
                    }
                }
            }
        }
        println!(
            "   SELF-CHECK: {equal} of {total} columns of chunk ({kx}, {ky}) equal the chunk builder's own surface and water, bit for bit{}",
            if built.is_none() {
                " (the chunk builder built NO chunk here)"
            } else {
                ""
            }
        );
        let verdict = if equal == total { "GREEN" } else { "RED" };
        census(
            &format!("G-TEETH-SELFCHECK.{tag}"),
            &format!("{equal}of{total}cols"),
            "=all",
            verdict,
        );
    } else {
        println!("   SELF-CHECK: skipped — the middle chunk is a partial chunk at the face's edge");
        census(
            &format!("G-TEETH-SELFCHECK.{tag}"),
            "none",
            "=all",
            "NO-BAND",
        );
    }
    let read = cols.iter().flatten().count();
    let agree = cols.iter().flatten().filter(|c| c.base_agrees).count();
    println!(
        "   BASE-CHECK (a drift alarm): in {agree} of {read} columns the base, the fine half, the bench and the shore law give the kernel's own surface again"
    );
    let verdict = if agree == read { "GREEN" } else { "RED" };
    census(
        &format!("G-TEETH-BASECHECK.{tag}"),
        &format!("{agree}of{read}cols"),
        "=all",
        verdict,
    );
}

/// One pair of neighbouring columns where the drawn side changes.
struct Edge {
    /// The pair steps along `j` (the columns `(a, b)` and `(a, b + 1)`), else along `i`.
    along_j: bool,
    water: Water,
    /// The distance to the ground's own crossing, in columns and in metres.
    offset_cols: f64,
    offset_m: f64,
    /// The distance is a LOWER BOUND: the search stopped at the square's end or at a column not
    /// read before it found a crossing.
    censored: bool,
    /// The surface step across the pair, in metres.
    wall_m: f64,
    /// The shore law's forced floor for the pair, in metres (see [`Square::edges`]).
    floor_m: f64,
}

/// The side-change pairs and the steps between every other pair of neighbours.
struct Edges {
    list: Vec<Edge>,
    steps: Vec<f64>,
}

/// One square of columns, `b · n + a`.
struct Square<'a> {
    n: usize,
    cols: &'a [Option<ShippedColumn>],
    m_i: f64,
    m_j: f64,
    sea_radius: Gi,
}

impl Square<'_> {
    /// The column at `(a, b)`.
    fn at(&self, a: usize, b: usize) -> Option<&ShippedColumn> {
        self.cols[b * self.n + a].as_ref()
    }

    /// ★ EVERY PAIR OF NEIGHBOURS: a side-change pair becomes an [`Edge`] with its offset, wall and
    /// forced floor; every other pair gives the slope's own step.
    ///
    /// ★ THE FORCED FLOOR is what `vd_recipe::height::shore` guarantees for the pair: a column the
    /// side word holds DRY stands at least `|g| >> SHORE_SHIFT` over its own water, and one it holds
    /// WET at least that far under its own water, each `g` the column's base over its own water. So
    /// the wall is at least `(w_dry + |g_dry|/4) − (w_wet − |g_wet|/4)`; under zero the law forces no
    /// wall and the floor reads zero.
    fn edges(&self) -> Edges {
        let n = self.n;
        let share = f64::from(1u32 << SHORE_SHIFT);
        let mut list = Vec::new();
        let mut steps = Vec::new();
        for b in 0..n {
            for a in 0..n {
                for along_j in [false, true] {
                    let (na, nb) = if along_j { (a, b + 1) } else { (a + 1, b) };
                    if na >= n || nb >= n {
                        continue;
                    }
                    let (Some(p), Some(q)) = (self.at(a, b), self.at(na, nb)) else {
                        continue;
                    };
                    let step = metres_of_q28(p.surface - q.surface).abs();
                    let (wet, dry) = match (p.drawn_wet(), q.drawn_wet()) {
                        (Some(true), Some(false)) => (p, q),
                        (Some(false), Some(true)) => (q, p),
                        _ => {
                            steps.push(step);
                            continue;
                        }
                    };
                    let water = if wet.lake() { Water::Lake } else { Water::Sea };
                    let w = wet.water;
                    let (pair, fixed) = if along_j { (b, a) } else { (a, b) };
                    let value = |k: usize| -> Option<f64> {
                        let c = if along_j {
                            self.at(fixed, k)
                        } else {
                            self.at(k, fixed)
                        };
                        c.map(|c| metres_of_q28(c.base - w))
                    };
                    let (offset_cols, censored) = nearest_crossing(n, pair, value);
                    let m = if along_j { self.m_j } else { self.m_i };
                    let g_wet = metres_of_q28(wet.base - wet.water).abs();
                    let g_dry = metres_of_q28(dry.base - dry.water).abs();
                    let floor_m =
                        (metres_of_q28(dry.water - wet.water) + (g_dry + g_wet) / share).max(0.0);
                    list.push(Edge {
                        along_j,
                        water,
                        offset_cols,
                        offset_m: offset_cols * m,
                        censored,
                        wall_m: step,
                        floor_m,
                    });
                }
            }
        }
        Edges { list, steps }
    }

    /// ★ HOW FAR EACH COLUMN STANDS FROM THE NEAREST SHORE OF ONE FIELD along its own row and along
    /// its own column, in columns (infinite where its row or column holds no sign change). `field`
    /// is a water's drawn field ([`Square::drawn_field`]), so the distance is to THAT water's shore.
    fn reach(&self, field: &[Option<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n;
        let mut row = vec![f64::INFINITY; n * n];
        let mut col = vec![f64::INFINITY; n * n];
        for along_j in [false, true] {
            let out = if along_j { &mut col } else { &mut row };
            for line in 0..n {
                let idx = |k: usize| if along_j { k * n + line } else { line * n + k };
                // A shore stands between the columns `k` and `k + 1` of this line.
                let change = |k: usize| match (field[idx(k)], field[idx(k + 1)]) {
                    (Some(x), Some(y)) => (x >= 0.0) != (y >= 0.0),
                    _ => false,
                };
                let mut last: Option<f64> = None;
                for k in 0..n {
                    if k >= 1 && change(k - 1) {
                        last = Some(k as f64 - 0.5);
                    }
                    if let Some(l) = last {
                        out[idx(k)] = k as f64 - l;
                    }
                }
                let mut next: Option<f64> = None;
                for k in (0..n).rev() {
                    if k + 1 < n && change(k) {
                        next = Some(k as f64 + 0.5);
                    }
                    if let Some(x) = next {
                        out[idx(k)] = out[idx(k)].min(x - k as f64);
                    }
                }
            }
        }
        (row, col)
    }

    /// The drawn side of one water as a field of ±1: −1 under that water, +1 elsewhere, `None`
    /// where no side is stated.
    fn drawn_field(&self, water: Water) -> Vec<Option<f64>> {
        self.cols
            .iter()
            .map(|c| {
                c.as_ref().and_then(|c| {
                    c.drawn_wet().map(|wet| {
                        let mine = if c.lake() { Water::Lake } else { Water::Sea };
                        if wet && mine == water { -1.0 } else { 1.0 }
                    })
                })
            })
            .collect()
    }

    /// The ground's own height over the body's sea, in metres.
    fn sea_ground_field(&self) -> Vec<Option<f64>> {
        self.cols
            .iter()
            .map(|c| c.as_ref().map(|c| metres_of_q28(c.base - self.sea_radius)))
            .collect()
    }

    /// ★ THE LAKE LEVEL EACH COLUMN IS JUDGED AGAINST, as a radius:
    /// * a column the side word draws under a LAKE: its own level;
    /// * a column drawn under the SEA: none (it is no column of a lake);
    /// * any other column: its OWN water where that is a lake's level (a far rung's level-1 word,
    ///   status T5), else the level of the nearest column drawn under a lake — a walk over the four
    ///   sides from every lake column at once, out to `reach` columns; past that, none.
    ///
    /// A dry column beside a lake holds the SEA as its own water at rungs 0–9 (status T4), so the
    /// lake's own level is the one its ground is judged against.
    fn lake_levels(&self, reach: usize) -> Vec<Option<Gi>> {
        let n = self.n;
        let mut walked: Vec<Option<(Gi, usize)>> = vec![None; n * n];
        let mut queue = VecDeque::new();
        for (k, c) in self.cols.iter().enumerate() {
            if let Some(c) = c
                && c.drawn_wet() == Some(true)
                && c.lake()
            {
                walked[k] = Some((c.water, 0));
                queue.push_back(k);
            }
        }
        while let Some(k) = queue.pop_front() {
            let Some((w, d)) = walked[k] else {
                continue;
            };
            if d >= reach {
                continue;
            }
            for q in sides_of(n, k).into_iter().flatten() {
                if walked[q].is_none() {
                    walked[q] = Some((w, d + 1));
                    queue.push_back(q);
                }
            }
        }
        self.cols
            .iter()
            .zip(walked)
            .map(|(c, walk)| {
                let c = c.as_ref()?;
                match c.drawn_wet() {
                    Some(true) if c.lake() => Some(c.water),
                    Some(true) => None,
                    _ if (c.water != self.sea_radius) & (c.water != Gi::ZERO) => Some(c.water),
                    _ => walk.map(|(w, _)| w),
                }
            })
            .collect()
    }

    /// ★ THE GROUND'S OWN HEIGHT OVER ITS LAKE'S LEVEL, in metres ([`Square::lake_levels`]). A
    /// column one of whose four sides holds ANOTHER lake's level is left out: between two levels the
    /// sign flips on the line where the walk from one lake meets the walk from the other, and that
    /// line runs along the grid — a false waterline that would lean the control toward the grid.
    fn lake_ground_field(&self, levels: &[Option<Gi>]) -> Vec<Option<f64>> {
        let n = self.n;
        (0..n * n)
            .map(|k| {
                let c = self.cols[k].as_ref()?;
                let w = levels[k]?;
                let split = sides_of(n, k)
                    .into_iter()
                    .flatten()
                    .any(|q| matches!(levels[q], Some(v) if v != w));
                (!split).then(|| metres_of_q28(c.base - w))
            })
            .collect()
    }

    /// ★ THE LEVEL A COLUMN IS JUDGED AGAINST FOR ONE WATER, or `None` where the column is not that
    /// water's to judge: the SEA judges every column but a lake's against the body's sea; a LAKE
    /// judges every column but the sea's against its lake level ([`Square::lake_levels`]).
    fn judged_level(&self, water: Water, c: &ShippedColumn, lake: Option<Gi>) -> Option<Gi> {
        match (water, c.drawn_wet() == Some(true), c.lake()) {
            (Water::Sea, true, true) | (Water::Lake, true, false) => None,
            (Water::Sea, _, _) => Some(self.sea_radius),
            (Water::Lake, _, _) => lake,
        }
    }
}

/// The four sides of column `k` of a square of `n` columns a side, where they stand inside it.
fn sides_of(n: usize, k: usize) -> [Option<usize>; 4] {
    let (a, b) = (k % n, k / n);
    [
        (a > 0).then(|| k - 1),
        (a + 1 < n).then(|| k + 1),
        (b > 0).then(|| k - n),
        (b + 1 < n).then(|| k + n),
    ]
}

/// ★ THE NEAREST CROSSING OF THE GROUND along one line of `n` columns, from the pair `pair` (the
/// columns `pair` and `pair + 1`): `value(k)` is the ground over the water at column `k`, in metres,
/// or `None` for a column not read. The crossing inside a pair stands at the linear interpolation of
/// its two values; a value of exactly zero is on the dry side, as in the shore law (`g >= 0`).
///
/// The search runs both ways from the pair. Each way stops at its first crossing (an EXACT
/// distance), or at the line's end or a column not read (a LOWER BOUND: a crossing past that point
/// stands at least that far). The answer is the smaller of the two ways, in columns from the pair's
/// middle, and `true` where the smaller is a bound. So an answer marked exact IS exact — the other
/// way stands at least as far — and a bound says only "at least this far".
fn nearest_crossing(n: usize, pair: usize, value: impl Fn(usize) -> Option<f64>) -> (f64, bool) {
    let middle = pair as f64 + 0.5;
    // Some(Some(x)): the ground crosses at x; Some(None): no crossing in the pair; None: a column of
    // the pair was not read.
    let cross = |k: usize| -> Option<Option<f64>> {
        let (p, q) = (value(k)?, value(k + 1)?);
        Some(((p >= 0.0) != (q >= 0.0)).then(|| k as f64 + p / (p - q)))
    };
    // Down: the pairs `pair`, `pair − 1`, … `0`. A stop at the pair `k` leaves the columns from
    // `k + 1` up clear of a crossing, so the bound is `middle − (k + 1)`; the line's start is `k = −1`.
    let mut down = (middle, true);
    let mut k = pair as i64;
    while k >= 0 {
        match cross(k as usize) {
            Some(Some(x)) => {
                down = ((middle - x).abs(), false);
                break;
            }
            Some(None) => k -= 1,
            None => {
                down = (middle - (k + 1) as f64, true);
                break;
            }
        }
    }
    // Up: the pairs `pair + 1`, … `n − 2`. A stop at the pair `k` leaves the columns up to `k` clear,
    // so the bound is `k − middle`; the line's end is `k = n − 1`.
    let mut up = (n as f64 - 1.0 - middle, true);
    let mut k = pair + 1;
    while k + 1 < n {
        match cross(k) {
            Some(Some(x)) => {
                up = ((x - middle).abs(), false);
                break;
            }
            Some(None) => k += 1,
            None => {
                up = (k as f64 - middle, true);
                break;
            }
        }
    }
    // The smaller way; at a tie the exact one.
    if down.0 < up.0 || (down.0 <= up.0 && !down.1) {
        down
    } else {
        up
    }
}

/// The reading guide, printed once at the end.
fn legend() {
    println!(
        "\nTHE LEGEND (every length in metres; a `col` is one column of the rung; an `x` on a flatness is times the mean bin, on a wall times the slope's own step):\n  OFFSET — for each pair of neighbouring columns where the drawn side changes, the distance along the same row or column to the nearest place where the ground's base crosses THAT water's level. A shore that follows its ground reads at most half a column (the band). A reading that met the square's end first is a lower bound: over the band it proves RED; inside the band it leaves the verdict NO-BAND.\n  WALL — the surface step across that pair over the median step between the square's other neighbouring columns; WALL10 counts the pairs over ten times. Beside it the shore law's forced floor. No band yet.\n  GRID — the flatness of the drawn shore's window bearings at four bins; GRID-GROUND the same for the ground's own waterline in the same square (the control). Band: the 95th percentile of a flat histogram at the same count (ruling B3 G-GRID; ruling W13.3's method). NULLCHECK prints that method at W13.3's own count.\n  DISAGREE / REACH — the columns the side word holds on the other side of the water from their own ground, and how far from that water's drawn shore the farthest one stands. No band yet.\n  SELFCHECK / BASECHECK — the instrument reads the shipped column (SELFCHECK, against the chunk builder) with the kernel's own base (BASECHECK, a drift alarm): a RED here voids every other line of that square.\nWHAT THE TEETH SHOULD PRINT.\n  TODAY (predicted from the status file §6, UNMEASURED): at rungs 0–9 OFFSET RED with a largest offset near half a node (about 4 km); the wall several times the slope's step, near the shore law's floor; GRID RED near 2x while GRID-GROUND reads near its band; at rungs 10–13 a REACH up to a level-1 square (about 16 km), and most small lakes with no edge at all.\n  AFTER THE CURE (plan step 4, one shoreline from one field): OFFSET GREEN at every rung, the wall near 1x, GRID equal to GRID-GROUND and inside its band, DISAGREE near zero."
    );
}
