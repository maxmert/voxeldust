//! ★ THE REFERENCE STAND AND ITS SKYLINE (ruling B6 item 2, 2026-09-24; plan step 1 and §9 of
//! `docs/investigation/2026-09-24/terrain_believability_status.md`).
//!
//! **THE OWNER'S WORDS.** *"we need to have this quality of the terrain and world"* and *"please make
//! sure that visually we are moving towards this target picture"* (ruling B6). For the light:
//! *"Ideally with long shadows"* (ruling B6.5), amended the same day to *"We can go without long
//! shadows for now"*. So the picture is judged as the client draws it, with the caption "no far
//! shadow", and the sun this instrument asks for is the lowest sun the picture light allows.
//!
//! **WHY IT EXISTS.** Ruling B6 item 2 asks each step of the arc for a FOURTH stand, framed like the
//! owner's reference picture, and for the skyline numbers on that stand by the rule of
//! `11_reference_trace.md`. Nobody chose that stand yet. And nobody measured the skyline of the
//! ground the game SHIPS: M1b (`skyline_march`) marched the recipe's own relief before 8c, with no
//! solve under it. This instrument chooses the stand by a stated rule, marches the shipped ground
//! from it, and prints our numbers beside the reference's.
//!
//! **THE SKYLINE IS A REPORT, NOT A GATE.** Ruling T6 ask 5 (2026-09-16; the owner: *"I would also
//! not use any limiting number, but rely on physics and models more and see how that will
//! unfold"*): the march prints the counts and the largest rise beside the reference trace, with no
//! floor, no target and no refusal. So every skyline census line says NO-BAND, prints the
//! reference's 70-run envelope as its band word, and says UNDER, INSIDE or OVER that envelope on
//! the next line. The lines that CAN go red are the instrument's own:
//! - **G-REF-DETAIL-TO-HORIZON**: the marched samples the shipped field cannot state at the rung the
//!   ladder asks for. The client then draws a coarser rung there (ruling F9), and the checklist row
//!   "detail to the horizon" (status §9; `01` §2: "a rung ladder that does not go blank") wants zero.
//! - **G-REF-SHIPPED-PATH**: the column reader against the chunk builder's own `column_field`, on
//!   one chunk for each rung the march reads (SL10: one generator; the reader must BE the shipped
//!   column).
//! - **G-REF-SUN-ELEVATION** and **G-REF-SUN-OFF-NOSE**: a stated sun (`VD_STAND_SUN`) against the
//!   picture light of ruling D8-7 (12°–18° up, 100°–140° off the nose).
//!
//! **WHAT IT READS.**
//! 1. **THE POSE, BY A RULE.** The rule reads the SOLVED field: the artifact's rows, one row for each
//!    8 192 m node. The search holds the nodes inside 300 km of the belt stand's direction. Each land
//!    node whose eight neighbours are land gets a score in metres: its height over the mean of the
//!    ring 20–60 km round it, plus the greatest relief (the highest minus the lowest top) 30–60 km
//!    away inside one 55° sector of bearing. The best score is the stand. The middle of its sector
//!    is the look bearing. The eye stands 400 m over the SHIPPED ground at the node's own direction.
//! 2. **THE SKYLINE.** The ONE march and the ONE break rule (`vd_bins::skyline`) read the SHIPPED
//!    column (`vd_bins::shore_geometry::RungReader`: the coast mask's side, the water that side
//!    names, the slope share, the solved `Z`, then the one kernel `column_surface_at`, in the chunk
//!    builder's order). Each sample reads the rung the client's ladder draws at its distance from
//!    the eye. The visible top of a column is its ground, or the water over it.
//! 3. **THE FRAMING.** The pitch that puts our skyline where the reference's own skyline stands in
//!    its frame (the median row of `11` §7's trace), the crop of the client's 45° frame that holds
//!    the reference's 55°, the sun that is wanted, and the spawn line the picture tools take.
//!
//! `cargo run --release -p vd-bins --example reference_stand -- [search_km] [dx dy dz]`, with an
//! optional `VD_STAND_SUN="x y z"` (the star's unit direction in the planet's frame, the stamp's
//! `star.direction_body`) to judge the sun of one moment.
//!
//! **Example.** A pilot stands on a node near the belt. The node rises 900 m over the ground 20–60 km
//! round it, and the eye stands 400 m over the column the chunk builder draws there. To the
//! north-east, 30–60 km out, the solve raised a range 2 400 m over the valley in front of it. The
//! march steps along 360 bearings. Each sample reads the rung its distance asks for (rung 0 under
//! the boots, rung 7 at 100 km, rung 9 past 222 km), and the coast mask says where the sea stands.
//! Inside the 55° round the range the rule finds one break at 0.25° in 60°. The reference holds
//! 2.45 to 5.45, so the line says UNDER, as a report. The detail line reads zero samples and the
//! shipped-path line reads zero columns, so the number is the shipped ground's own.

use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_bins::drainage_gates::cell_of;
use vd_bins::grid_bias::unit;
use vd_bins::shore_geometry::{RungReader, unit_of};
use vd_bins::skyline::{
    BEARINGS, Eye, Fan, Ground, M1B_STEPS, THRESHOLDS_DEG, bearing_dir, is_break, judge,
    judge_sector, march, rises, tangent_basis,
};
use vd_client::ladder_view::AskBound;
use vd_core::geometry::{REFERENCE_VIEW_FOV_Y_RAD, REFERENCE_VIEW_ROWS_PX};
use vd_core::glam::{DMat3, DQuat, DVec3};
use vd_terrain::artifact::{Artifact, CoastCounts, DRY_M, PyramidField};
use vd_terrain::chunk::{CHUNK_EDGE, column_field};
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_terrain::macro_lattice::NO_NODE;
use vd_terrain::units::metres_of_q28;
use vd_terrain::{BodyDefinition, GENERATOR_VERSION};

/// THE BELT STAND's direction in the planet's frame: the aim `lake_census`, `drain_census` and
/// `rung_swap` use, where the owner flew.
const BELT: [f64; 3] = [0.617_270, -0.437_286, -0.654_033];
/// How far from the aim a stand may stand, metres of arc (the first argument, in km, replaces it).
/// A stated choice: the reference stand stands "near the belt's range" (ruling B6 item 2).
const SEARCH_M: f64 = 300_000.0;
/// The ring a stand must stand high over, metres of arc from it (ruling B6 item 2: "a high point").
const RING_INNER_M: f64 = 20_000.0;
const RING_OUTER_M: f64 = 60_000.0;
/// The band the range is looked for in, metres of arc (ruling B6 item 2 and `01` §2: the range
/// stands 30–60 km away).
const RELIEF_INNER_M: f64 = 30_000.0;
const RELIEF_OUTER_M: f64 = 60_000.0;
/// The reference's span of bearing, degrees (`11` §1: 55° is the base of the 45°–65° sweep, ASSUMED).
const SECTOR_DEG: f64 = 55.0;
/// The same span on the rule's 1° grid: 27 bearings each side of the look, 55 in all.
const SECTOR_HALF: usize = 27;
/// The eye over its local SHIPPED ground, metres (ruling B6 item 2: ~400 m; `01` §2: 373 m on this
/// planet sees 50 km of ground).
const EYE_OVER_GROUND_M: f64 = 400.0;
/// The client's capture frame, columns (client-render's `WINDOW_W`); its rows and its vertical field
/// of view are `vd_core::geometry`'s reference view (720 rows, 45°, client-render's `WINDOW_H`).
const FRAME_COLS_PX: f64 = 1280.0;
/// The reference photograph, pixels (`11` §1: 1 080 × 810, four by three).
const REF_COLS_PX: f64 = 1080.0;
const REF_ROWS_PX: f64 = 810.0;
/// ★ WHERE THE REFERENCE'S SKYLINE STANDS IN ITS OWN FRAME, rows from the top: the median of the
/// 108 rows of `11` §7's finished trace A (ARITH on that MEASURED table). The reference's principal
/// point is ASSUMED to be its centre (`11` §3 point 2).
const REF_SKYLINE_ROW_PX: f64 = 192.0;
/// The reference's own pitch, degrees down (`11` §3 point 3, ASSUMED), and the median of its
/// skyline on the 1° grid at that pitch, degrees (ARITH on `11` §7's 60 values). Printed only as a
/// cross-check of the framing arithmetic, never read by it.
const REF_PITCH_DEG: f64 = 15.0;
const REF_SKYLINE_MEDIAN_DEG: f64 = -3.235;
/// Where the look-at target stands when the view axis does not meet the ground's sphere: the middle
/// of the 30–60 km band, metres along the axis.
const LOOK_FALLBACK_M: f64 = 45_000.0;
/// ★ THE PICTURE LIGHT (ruling D8-7, `owner_decisions_2026-09-07_voxels.md`: "the light stated
/// (12°–18° up, 100°–140° off the nose)"; `terrain_pictures` holds its star to the same bands).
const SUN_ELEVATION_BAND_DEG: (f64, f64) = (12.0, 18.0);
const SUN_OFF_NOSE_BAND_DEG: (f64, f64) = (100.0, 140.0);
/// ★ THE SUN THAT IS WANTED, degrees over the eye's horizontal: the FLOOR of D8-7's band. The lowest
/// sun the ruling allows casts the longest shadow the ruling allows (ruling B6.5: *"Ideally with
/// long shadows"*).
const SUN_ELEVATION_DEG: f64 = 12.0;
/// The sun's bearing off the nose, degrees: the middle of D8-7's band, behind the shoulder, where
/// `terrain_pictures` stands its star (`SUN_OFF_NOSE_DEG`).
const SUN_OFF_NOSE_DEG: f64 = 120.0;
/// The ridge the shadow arithmetic is printed for, metres tall.
const SHADOW_RIDGE_M: f64 = 500.0;

/// ★ THE REFERENCE, TRACED BY OUR OWN RULE (`11_reference_trace.md` §4, "BOTH, 70 runs": the lowest,
/// the median and the highest over both trace families and every setting — the band §0 rounds).
/// Breaks per 60° of bearing at each of [`THRESHOLDS_DEG`].
const REF_BREAKS: [(f64, f64, f64); 4] = [
    (4.00, 6.00, 8.89),
    (2.45, 5.23, 7.89),
    (2.45, 4.00, 5.45),
    (0.00, 2.00, 3.33),
];
/// The reference's largest rise over the ±5° trend, degrees (the same envelope).
const REF_RISE_DEG: (f64, f64, f64) = (0.431, 0.820, 1.067);
/// The reference's BASE run (`11` §4: a 55° span, a 15° pitch): its largest rise in degrees and its
/// breaks per 60° at the four thresholds.
const REF_BASE_RISE_DEG: f64 = 0.716;
const REF_BASE_BREAKS: [f64; 4] = [4.80, 3.60, 3.60, 1.20];
/// ★ THE THRESHOLDS INSIDE THE TRACE'S OWN NOISE (`11` §5): the rise noise is the size of the 0.05°
/// threshold and inside the 0.10° one. Those two readings carry a warning beside them.
const NOISY: [bool; 4] = [true, true, false, false];
/// The gate names' threshold words, one for each threshold.
const TAGS: [&str; 4] = ["B005", "B010", "B025", "B050"];
/// The band word's source for every skyline line (no spaces: a later script splits on them).
const SKYLINE_SOURCE: &str = "ruling-T6-ask5:a-report-no-floor;11s4:the-70-run-envelope";

/// A census line's band.
enum Band {
    /// A band `[lo, hi]` from its named source (no spaces): a value outside it is RED, and so is a
    /// value that is not a number, because a reading the instrument could not make cannot pass.
    Within(f64, f64, &'static str),
    /// The reference's measured envelope `[low, high]` beside a reading the owner ruled a REPORT:
    /// never a verdict.
    Report(f64, f64, &'static str),
    /// No published band yet; the word says what exists in its place (no spaces).
    NoneYet(&'static str),
}

/// ★ ONE CENSUS LINE for a later script to grep: `CENSUS <gate> <value> <band><units>(<source>)
/// <verdict>`. The value is a plain number; its units and its band's source ride on the band word.
fn census(gate: &str, value: f64, band: &Band, units: &str) -> &'static str {
    let (word, verdict) = match band {
        Band::Within(lo, hi, source) => (
            format!("[{lo},{hi}]{units}({source})"),
            if (*lo..=*hi).contains(&value) {
                "GREEN"
            } else {
                "RED"
            },
        ),
        Band::Report(lo, hi, source) => (format!("report[{lo},{hi}]{units}({source})"), "NO-BAND"),
        Band::NoneYet(what) => (format!("none-yet-{units}({what})"), "NO-BAND"),
    };
    println!("CENSUS {gate} {value:.3} {word} {verdict}");
    verdict
}

/// Where a reading stands against an envelope `[lo, hi]`: a report word, never a verdict.
fn position(value: f64, lo: f64, hi: f64) -> &'static str {
    if !value.is_finite() {
        "NO-READING against"
    } else if value < lo {
        "UNDER"
    } else if value > hi {
        "OVER"
    } else {
        "INSIDE"
    }
}

/// ★ THE RUNG RULE: the finest rung whose TERRITORY holds a distance from the eye. A rung's
/// territory is its effective switch distance as the client's ladder states it for a still eye
/// (`AskBound::unbounded().for_body`: one cell one pixel at the reference view, floored by the
/// handover's own step, ruling T7 rule 3 and W17); the top rung's territory reaches everywhere.
/// The crossfade band (−10 % to +10 % of a switch) is not modelled: the finer rung owns a distance
/// up to its switch.
fn rung_at(territory: &[f64], d_m: f64) -> u8 {
    let mut rung = 0usize;
    while rung + 1 < territory.len() && d_m > territory[rung] {
        rung += 1;
    }
    rung as u8
}

/// An angle folded into `[−π, π)`.
fn wrap(x: f64) -> f64 {
    (x + PI).rem_euclid(TAU) - PI
}

/// ★ THE BEST SECTOR of an annulus: over the 360 sector middles on the 1° grid, the one whose nodes
/// (bearing in radians, visible top in metres) inside ±`SECTOR_DEG`/2 span the greatest relief, the
/// highest top minus the lowest. `None` where no sector holds a node.
fn best_sector(annulus: &[(f64, f64)]) -> Option<(usize, f64)> {
    let half = (0.5 * SECTOR_DEG).to_radians();
    let mut best: Option<(usize, f64)> = None;
    for k in 0..BEARINGS {
        let middle = (k as f64) * TAU / (BEARINGS as f64);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &(b, top) in annulus {
            if wrap(b - middle).abs() <= half {
                lo = lo.min(top);
                hi = hi.max(top);
            }
        }
        if lo <= hi && best.is_none_or(|(_, r)| hi - lo > r) {
            best = Some((k, hi - lo));
        }
    }
    best
}

/// One candidate stand and the two terms of its score, all in metres.
struct Candidate {
    node: u32,
    dir: DVec3,
    top_m: f64,
    over_m: f64,
    look_deg: usize,
    relief_m: f64,
    score_m: f64,
    off_aim_km: f64,
}

/// The chunk builder's readers, one for each `(rung, face)` asked for, built when first asked.
type Readers<'a> = BTreeMap<(u8, u8), Option<RungReader<'a>>>;

/// ★ THE SHIPPED COLUMN a direction falls in at a rung: the column's own direction, its ground and
/// its water as radii in metres from the body's centre (the water is ZERO on a body with none).
/// `None` where the rung has no field or the field does not hold the column's stencil.
///
/// UNITS: `RungReader::column` answers radii in gap steps at the length format;
/// `vd_terrain::units::metres_of_q28` divides by `STEPS_PER_M << LENGTH_BITS` (2³⁵), a power of two,
/// so a radius of about 3.35 million metres comes out to a few nanometres.
fn shipped_top<'a>(
    readers: &mut Readers<'a>,
    body: &'a BodyDefinition,
    artifact: &'a Artifact,
    counts: &Arc<CoastCounts>,
    dir: DVec3,
    rung: u8,
) -> Option<(DVec3, f64, f64)> {
    let (face, i, j) = cell_of(body, dir, rung);
    let reader = readers
        .entry((rung, face.index()))
        .or_insert_with(|| RungReader::new(body, artifact, counts, face, rung))
        .as_ref()?;
    let col = reader.column(i, j)?;
    Some((
        unit_of(col.dir),
        metres_of_q28(col.surface),
        metres_of_q28(col.water),
    ))
}

fn main() -> ExitCode {
    let a: Vec<f64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("a number"))
        .collect();
    let search_m = a.first().map_or(SEARCH_M, |km| km * 1_000.0);
    let aim = if a.len() >= 4 {
        DVec3::new(a[1], a[2], a[3]).normalize()
    } else {
        DVec3::from_array(BELT).normalize()
    };
    let sun: Option<DVec3> = std::env::var("VD_STAND_SUN").ok().map(|v| {
        let w: Vec<f64> = v
            .split_whitespace()
            .map(|x| x.parse().expect("a sun component"))
            .collect();
        DVec3::new(w[0], w[1], w[2]).normalize()
    });
    let started = Instant::now();
    let Some(artifact) = run_solve(&SolveJob {
        body: home_planet(),
        words: home_solve_words(),
    }) else {
        println!("reference_stand: REFUSED — the home planet did not solve");
        return ExitCode::FAILURE;
    };
    let solve_s = started.elapsed().as_secs_f64();
    let body = home_planet().with_sea_m(artifact.sea());
    let Some(fine) = body.macro_lattice() else {
        println!("reference_stand: REFUSED — the home planet has no macro lattice");
        return ExitCode::FAILURE;
    };
    let radius_m = body.ladder().radius_m();
    let rungs = body.ladder().rungs;
    let levels = artifact.pyramid.len() as u32;
    let counts = artifact.coast_counts();
    // The rows' heights and the sea are whole metres over the ladder radius (`Row::z_m`,
    // `Artifact::sea`).
    let sea_m = artifact.sea().unwrap_or(0);
    println!(
        "reference_stand: generator {GENERATOR_VERSION}, artifact version {}; the home planet solved in {solve_s:.1} s; \
         the ladder radius {radius_m:.1} m, the sea {sea_m} m over it; {rungs} rungs, {levels} pyramid levels; \
         the macro node {:.0} m",
        artifact.version,
        fine.node_m()
    );
    println!(
        "  every CENSUS line reads: CENSUS <gate> <value> <band><units>(<source>) <verdict>; GREEN or RED against a ruled \
         band; NO-BAND = a report (the skyline, ruling T6 ask 5) or a reading with no band yet"
    );

    // ── 1. THE POSE, BY THE RULE, ON THE SOLVED FIELD ─────────────────────────────────────────────
    let n = fine.node_count();
    // A node's VISIBLE top in metres over the ladder radius: its ground, or the water over it.
    let top_of = |node: u32| -> f64 {
        let r = artifact.rows[node as usize];
        if r.water_m == DRY_M {
            f64::from(r.z_m)
        } else {
            f64::from(r.z_m.max(r.water_m))
        }
    };
    // The `land_stand` rule: dry and over the sea.
    let land = |node: u32| -> bool {
        let r = artifact.rows[node as usize];
        r.water_m == DRY_M && i32::from(r.z_m) > sea_m
    };
    let mut start = 0u32;
    let mut start_dot = -2.0;
    for node in 0..n as u32 {
        let dot = unit(&fine, node).dot(aim);
        if dot > start_dot {
            start_dot = dot;
            start = node;
        }
    }
    // Every node within the search plus the outer ring of the aim, by a walk over the lattice's own
    // eight neighbours (the seams included), so no candidate's ring is cut short.
    let outer_m = RING_OUTER_M.max(RELIEF_OUTER_M);
    let cos_region = ((search_m + outer_m) / radius_m).cos();
    let mut seen = vec![false; n];
    seen[start as usize] = true;
    let mut queue = VecDeque::from([start]);
    let mut region: Vec<(u32, DVec3)> = Vec::new();
    while let Some(node) = queue.pop_front() {
        region.push((node, unit(&fine, node)));
        for m in fine.neighbours(node) {
            if m == NO_NODE || seen[m as usize] {
                continue;
            }
            if unit(&fine, m).dot(aim) >= cos_region {
                seen[m as usize] = true;
                queue.push_back(m);
            }
        }
    }
    let cos_search = (search_m / radius_m).cos();
    let cos_outer = (outer_m / radius_m).cos();
    let mut found: Vec<Candidate> = Vec::new();
    for &(c, dc) in &region {
        if dc.dot(aim) < cos_search
            || !land(c)
            || !fine.neighbours(c).iter().all(|&m| m == NO_NODE || land(m))
        {
            continue;
        }
        let (e1, e2) = tangent_basis(dc.to_array());
        let (e1, e2) = (DVec3::from_array(e1), DVec3::from_array(e2));
        let mut ring_sum = 0.0;
        let mut ring_n = 0usize;
        let mut annulus: Vec<(f64, f64)> = Vec::new();
        for &(m, dm) in &region {
            let cos = dc.dot(dm);
            if cos < cos_outer {
                continue;
            }
            // Metres of arc on the ladder sphere, the same measure the march steps by.
            let s = cos.clamp(-1.0, 1.0).acos() * radius_m;
            let top = top_of(m);
            if (RING_INNER_M..=RING_OUTER_M).contains(&s) {
                ring_sum += top;
                ring_n += 1;
            }
            if (RELIEF_INNER_M..=RELIEF_OUTER_M).contains(&s) {
                let v = dm - dc * cos;
                annulus.push((v.dot(e2).atan2(v.dot(e1)), top));
            }
        }
        let Some((look_deg, relief_m)) = best_sector(&annulus) else {
            continue;
        };
        if ring_n == 0 {
            continue;
        }
        let over_m = top_of(c) - ring_sum / ring_n as f64;
        found.push(Candidate {
            node: c,
            dir: dc,
            top_m: top_of(c),
            over_m,
            look_deg,
            relief_m,
            score_m: over_m + relief_m,
            off_aim_km: dc.dot(aim).clamp(-1.0, 1.0).acos() * radius_m / 1_000.0,
        });
    }
    found.sort_by(|x, y| y.score_m.total_cmp(&x.score_m));
    println!(
        "\nTHE POSE RULE: inside {:.0} km of the aim ({:.6} {:.6} {:.6}), every land node with eight land neighbours scores \
         (its top over the mean top of the ring {:.0}–{:.0} km round it) + (the greatest relief, the highest top minus the lowest, \
         {:.0}–{:.0} km away inside one {SECTOR_DEG:.0}° sector of bearing); metres, the solved rows; {} nodes walked, {} candidates",
        search_m / 1_000.0,
        aim.x,
        aim.y,
        aim.z,
        RING_INNER_M / 1_000.0,
        RING_OUTER_M / 1_000.0,
        RELIEF_INNER_M / 1_000.0,
        RELIEF_OUTER_M / 1_000.0,
        region.len(),
        found.len()
    );
    println!(
        "  rank | node | km off the aim | top m | over its ring m | the sector's middle ° (the basis's bearing) | relief m | score m"
    );
    for (k, c) in found.iter().take(5).enumerate() {
        println!(
            "  {:>4} | {:>8} | {:>14.1} | {:>5.0} | {:>15.0} | {:>43} | {:>8.0} | {:>7.0}",
            k + 1,
            c.node,
            c.off_aim_km,
            c.top_m,
            c.over_m,
            c.look_deg,
            c.relief_m,
            c.score_m
        );
    }
    let Some(stand) = found.first() else {
        println!(
            "reference_stand: REFUSED — no land node with eight land neighbours inside the search"
        );
        return ExitCode::FAILURE;
    };
    let d = stand.dir;
    let (face, ni, nj) = fine.split(stand.node);

    // ── 2. THE EYE OVER THE SHIPPED GROUND ─────────────────────────────────────────────────────────
    let body_ref = &body;
    let artifact_ref = &artifact;
    let counts_ref = &counts;
    let mut readers: Readers<'_> = BTreeMap::new();
    let Some((_, foot_ground, foot_water)) =
        shipped_top(&mut readers, body_ref, artifact_ref, counts_ref, d, 0)
    else {
        println!("reference_stand: REFUSED — the field holds no rung-0 column under the stand");
        return ExitCode::FAILURE;
    };
    // Radii in metres from the body's centre.
    let r_f = foot_ground.max(foot_water);
    let r_c = r_f + EYE_OVER_GROUND_M;
    let look_rad = (stand.look_deg as f64).to_radians();
    let (e1, e2) = tangent_basis(d.to_array());
    let look = DVec3::from_array(bearing_dir(e1, e2, look_rad));
    println!(
        "\nTHE POSE: node {} (face {face:?}, node {ni}, {nj}); direction {:.6} {:.6} {:.6}; the solved node's top {:.0} m over the \
         ladder radius; the SHIPPED ground under the eye (rung 0, the 1 m column) {:.1} m; the eye {EYE_OVER_GROUND_M:.0} m over it, \
         at {:.1} m over the ladder radius",
        stand.node,
        d.x,
        d.y,
        d.z,
        stand.top_m,
        r_f - radius_m,
        r_c - radius_m
    );
    println!(
        "  the look: bearing {}° in the eye's tangent basis (`vd_bins::skyline::tangent_basis`); the look direction {:.6} {:.6} {:.6}",
        stand.look_deg, look.x, look.y, look.z
    );

    // ── 3. THE MARCH THROUGH THE SHIPPED FIELD ─────────────────────────────────────────────────────
    let bound = AskBound::unbounded().for_body(&body, rungs, levels);
    let territory: Vec<f64> = (0..rungs).map(|r| bound.territory_m(r, rungs)).collect();
    let mut hits = vec![0usize; usize::from(rungs)];
    // The first direction each rung read a column at: the cross-check's chunk for that rung.
    let mut first_dir: Vec<Option<DVec3>> = vec![None; usize::from(rungs)];
    // The samples that read a coarser rung than their own, because their own could not state them.
    let mut coarser = 0usize;
    let mut ridge_samples: Vec<(DVec3, f64)> = Vec::new();
    let eye = Eye {
        dir: d.to_array(),
        radius_m: r_c,
    };
    let fan = Fan {
        bearings: BEARINGS,
        first_rad: look_rad,
    };
    let t0 = Instant::now();
    let sky = {
        let mut ground = |dir: [f64; 3], s: f64| -> Ground {
            // The straight distance from the eye to the column's direction on the sphere of the
            // eye's own ground, in the stable form: (r_c − r_f)² + (2·√(r_c·r_f)·sin(θ/2))², with
            // θ = s / R the angle the march itself uses.
            let chord = 2.0 * (r_c * r_f).sqrt() * (0.5 * s / radius_m).sin();
            let d_eye = ((r_c - r_f) * (r_c - r_f) + chord * chord).sqrt();
            let rung = rung_at(&territory, d_eye);
            hits[usize::from(rung)] += 1;
            let dv = DVec3::from_array(dir);
            // ★ A COLUMN THE FIELD CANNOT STATE AT ITS OWN RUNG: the chunk builder builds no chunk
            // there and the client keeps the coarser rung standing (ruling F9). So the march reads
            // the next coarser rung that can state it, and counts the sample; only a column no rung
            // can state is missing.
            let mut read = rung;
            loop {
                if let Some((cd, g, w)) =
                    shipped_top(&mut readers, body_ref, artifact_ref, counts_ref, dv, read)
                {
                    let top = g.max(w);
                    coarser += usize::from(read != rung);
                    if first_dir[usize::from(read)].is_none() {
                        first_dir[usize::from(read)] = Some(dv);
                    }
                    if (RELIEF_INNER_M..=RELIEF_OUTER_M).contains(&s) {
                        ridge_samples.push((dv, top));
                    }
                    return Ground::Column(cd.to_array(), top);
                }
                if read + 1 >= rungs {
                    return Ground::Missing;
                }
                read += 1;
            }
        };
        march(&eye, radius_m, &fan, &M1B_STEPS, &mut ground)
    };
    let march_s = t0.elapsed().as_secs_f64();
    println!(
        "\nTHE MARCH: {BEARINGS} bearings a degree apart from the look, steps of {:.0} m to {:.0} km then {:.0} m to {:.0} km \
         (M1b's, `01` §1.4a); {} samples; {coarser} read a coarser rung than their own, {} no rung could state; {march_s:.1} s",
        M1B_STEPS.near_step_m,
        M1B_STEPS.near_limit_m / 1_000.0,
        M1B_STEPS.far_step_m,
        M1B_STEPS.limit_m / 1_000.0,
        sky.samples,
        sky.missing
    );
    println!(
        "  THE RUNG RULE: each sample reads the finest rung whose territory (the client ladder's switch distance for a still eye, \
         floored by the handover's step) holds its straight distance from the eye; the crossfade band is not modelled"
    );
    println!("  rung | cell m | territory km | pyramid level | fine octaves kept | samples");
    for (r, (t, h)) in territory.iter().zip(&hits).enumerate() {
        if *h == 0 {
            continue;
        }
        let rung = r as u8;
        println!(
            "  {rung:>4} | {:>6} | {:>12.1} | {:>13} | {:>17} | {h:>7}",
            vd_seed::ladder::cell_m(rung),
            t / 1_000.0,
            PyramidField::level_for(&fine, levels, rung),
            body.octaves_at(rung)
                .len()
                .saturating_sub(body.first_fine())
        );
    }

    // ── 3b. THE READER AGAINST THE CHUNK BUILDER, ONE CHUNK A RUNG ─────────────────────────────────
    // The reader mirrors the chunk builder's own column (`ColumnRead`, crate-private). Here it is
    // checked against `column_field`, the chunk builder's public door, over the whole chunk that
    // holds the first column each rung read: every surface and every water radius must be equal.
    let edge = CHUNK_EDGE as i32;
    let (mut checked, mut differ, mut refused) = (0usize, 0usize, 0usize);
    for (r, first) in first_dir.iter().enumerate() {
        let Some(dv) = first else {
            continue;
        };
        let rung = r as u8;
        let (cface, ci, cj) = cell_of(&body, *dv, rung);
        let Some(Some(reader)) = readers.get(&(rung, cface.index())) else {
            continue;
        };
        let (x, y) = (ci.div_euclid(edge), cj.div_euclid(edge));
        let Some(cf) = column_field(&body, Some(reader.source.field()), cface, rung, x, y) else {
            refused += 1;
            continue;
        };
        let n_cells = body.ladder().cells_per_edge(rung) as i32;
        for cb in 0..edge {
            for ca in 0..edge {
                let (i, j) = (x * edge + ca, y * edge + cb);
                // A partial chunk at a face's far edge holds the partner face's columns there; the
                // reader reads this face's own columns only.
                if i >= n_cells || j >= n_cells {
                    continue;
                }
                // `ColumnField::columns` packs the chunk's columns in the order `b · 62 + a`.
                let k = (cb * edge + ca) as usize;
                checked += 1;
                differ += match reader.column(i, j) {
                    Some(col) => {
                        usize::from((col.surface != cf.columns[k].1) | (col.water != cf.water[k]))
                    }
                    None => 1,
                };
            }
        }
    }
    println!(
        "  THE SHIPPED PATH, CHECKED: the reader against `vd_terrain::chunk::column_field` on one chunk for each rung the march \
         read: {checked} columns compared, {differ} differ (surface or water), {refused} chunks the chunk builder refused"
    );

    // ── 4. THE FRAMING, FROM THE REFERENCE'S OWN ROWS ──────────────────────────────────────────────
    let sector_sky: Vec<f64> = (0..=2 * SECTOR_HALF)
        .map(|off| sky.deg[(BEARINGS + off - SECTOR_HALF) % BEARINGS])
        .collect();
    let mut sorted = sector_sky.clone();
    sorted.sort_by(f64::total_cmp);
    let horizon_deg = sorted[SECTOR_HALF];
    // The reference's focal length in its own pixels (`11` §3 point 1: F = (1080/2) / tan(span/2)),
    // and the angle its median skyline row stands over its centre row.
    let ref_focal_px = 0.5 * REF_COLS_PX / (0.5 * SECTOR_DEG).to_radians().tan();
    let delta = ((0.5 * REF_ROWS_PX - REF_SKYLINE_ROW_PX) / ref_focal_px).atan();
    let axis_deg = horizon_deg - delta.to_degrees();
    // The client's frame: square pixels, so one focal length in pixels serves both axes.
    let half_v = REFERENCE_VIEW_FOV_Y_RAD * 0.5;
    let focal_px = 0.5 * REFERENCE_VIEW_ROWS_PX / half_v.tan();
    let half_h = (0.5 * FRAME_COLS_PX / focal_px).atan();
    let crop_cols = 2.0 * focal_px * (0.5 * SECTOR_DEG).to_radians().tan();
    let crop_rows = crop_cols * REF_ROWS_PX / REF_COLS_PX;
    let frame_share = 0.5 - focal_px * delta.tan() / REFERENCE_VIEW_ROWS_PX;
    let dip_deg = (r_f / r_c).clamp(-1.0, 1.0).acos().to_degrees();
    println!("\nTHE FRAMING (the arithmetic, on the frame's centre column):");
    println!(
        "  the client's frame: {FRAME_COLS_PX:.0} × {REFERENCE_VIEW_ROWS_PX:.0} px at {:.1}° of vertical view, so {:.2}° across; \
         the reference's {SECTOR_DEG:.0}° across at 4:3 is its centred crop of {crop_cols:.0} × {crop_rows:.0} px",
        2.0 * half_v.to_degrees(),
        2.0 * half_h.to_degrees()
    );
    println!(
        "  the horizon = the median of the marched skyline over the {SECTOR_DEG:.0}° sector: {horizon_deg:.3}° of elevation \
         (the smooth sphere's own horizon from this eye dips {dip_deg:.3}°)"
    );
    println!(
        "  the reference's skyline stands at row {REF_SKYLINE_ROW_PX:.0} of {REF_ROWS_PX:.0} (the median of `11` §7's trace A), \
         {:.3} of its frame from the top; with its focal length {ref_focal_px:.1} px that row stands δ = {:.3}° over its axis \
         (cross-check: its ASSUMED {REF_PITCH_DEG:.0}° pitch and its skyline's median {REF_SKYLINE_MEDIAN_DEG:.3}° give {:.3}°)",
        REF_SKYLINE_ROW_PX / REF_ROWS_PX,
        delta.to_degrees(),
        REF_PITCH_DEG + REF_SKYLINE_MEDIAN_DEG
    );
    println!(
        "  so the view axis stands at {horizon_deg:.3}° − {:.3}° = {axis_deg:.3}° of elevation; the horizon then stands \
         {frame_share:.3} of the client's frame from its top, and at the reference's own row in the {crop_rows:.0}-row crop",
        delta.to_degrees()
    );

    // THE POSE AS THE PICTURE TOOLS TAKE IT (oblique_stand's construction: the camera looks along −Z).
    let eps = axis_deg.to_radians();
    let forward = (look * eps.cos() + d * eps.sin()).normalize();
    let right = forward.cross(d).normalize();
    let cam_up = right.cross(forward);
    let orient = DQuat::from_mat3(&DMat3::from_cols(right, cam_up, -forward)).normalize();
    let position_m = d * r_c;
    let beta = -eps;
    let theta = FRAC_PI_2 - beta;
    let off_axis = r_c * theta.sin();
    let reach_m = if beta > 0.0 && off_axis < r_f {
        r_c * theta.cos() - (r_f * r_f - off_axis * off_axis).sqrt()
    } else {
        LOOK_FALLBACK_M
    };
    let target = position_m + forward * reach_m;
    println!(
        "\nTHE STAND FOR THE PICTURE TOOLS (the planet's frame, metres from its centre; the facing a unit quaternion):"
    );
    println!(
        "  VD_SPAWN_POSES='1000=Planet({HOME_PLANET_SEED}):{},{},{}@{},{},{},{}'",
        position_m.x, position_m.y, position_m.z, orient.x, orient.y, orient.z, orient.w
    );
    println!(
        "  the look-at target ({reach_m:.0} m along the view axis, where it meets the eye's ground sphere or the band's middle): {} {} {}",
        target.x, target.y, target.z
    );
    println!(
        "  the gateway reads VD_SPAWN_POSES as `account=Planet(seed):x,y,z@qx,qy,qz,qw` entries split by `;`; \
         `vdctl look_at` aims the nose at a target"
    );
    println!("  the older stand tools, aimed at this node (THEIR framing, not this one):");
    println!(
        "    land_stand — a NADIR pose over the land node nearest an aim; args [height_m] [dx dy dz]; VD_STAND_SUN=\"x y z\" keeps \
         land under a star at least 15° up; VD_STAND_HIGHEST=1 takes the highest land instead:"
    );
    println!(
        "      cargo run --release -p vd-bins --example land_stand -- {EYE_OVER_GROUND_M:.0} {:.6} {:.6} {:.6}",
        d.x, d.y, d.z
    );
    println!(
        "    oblique_stand — the nose along the bearing to the NEAREST SEA, the smooth horizon `below_deg` over the axis; args <dx> \
         <dy> <dz> <below_deg> <steps> <step_km> <seq_alt_m> [alt_m ...]; VD_STAND_SUN, VD_STAND_SUN_MIN_DEG (default 10), \
         VD_STAND_AWAY=1 turns the nose from the sea:"
    );
    println!(
        "      cargo run --release -p vd-bins --example oblique_stand -- {:.6} {:.6} {:.6} {:.3} 1 20 150000 {EYE_OVER_GROUND_M:.0}",
        d.x,
        d.y,
        d.z,
        delta.to_degrees()
    );
    println!(
        "    sun_stand — a COAST node whose star stands nearest a target elevation; args <sun_x> <sun_y> <sun_z> [target_deg] \
         [alt_m ...]; it reads no VD_STAND_* variable"
    );

    // ── 5. THE SUN ─────────────────────────────────────────────────────────────────────────────────
    let mut reds = 0usize;
    let sun_elev = SUN_ELEVATION_DEG.to_radians();
    let off_nose = SUN_OFF_NOSE_DEG.to_radians();
    // The sun's horizontal direction `off_nose` round from the look, to one side.
    let across = |side: f64| (look * off_nose.cos() + right * (side * off_nose.sin())).normalize();
    let from_right = (d * sun_elev.sin() + across(1.0) * sun_elev.cos()).normalize();
    let from_left = (d * sun_elev.sin() + across(-1.0) * sun_elev.cos()).normalize();
    println!(
        "\nTHE SUN WANTED (ruling D8-7's picture light, {:.0}°–{:.0}° up and {:.0}°–{:.0}° off the nose): {SUN_ELEVATION_DEG:.0}° over \
         the eye's horizontal, the band's floor, so the longest shadow the ruling allows (B6.5), and {SUN_OFF_NOSE_DEG:.0}° off the \
         nose, behind the shoulder; a ridge {SHADOW_RIDGE_M:.0} m tall then casts {:.2} km of shadow. The star follows the \
         almanac: a flight takes the picture at a tick whose stamp's star.direction_body stands at",
        SUN_ELEVATION_BAND_DEG.0,
        SUN_ELEVATION_BAND_DEG.1,
        SUN_OFF_NOSE_BAND_DEG.0,
        SUN_OFF_NOSE_BAND_DEG.1,
        SHADOW_RIDGE_M / sun_elev.tan() / 1_000.0
    );
    println!(
        "  from the right: VD_STAND_SUN=\"{:.6} {:.6} {:.6}\"",
        from_right.x, from_right.y, from_right.z
    );
    println!(
        "  from the left:  VD_STAND_SUN=\"{:.6} {:.6} {:.6}\"",
        from_left.x, from_left.y, from_left.z
    );
    println!(
        "  the picture's caption: \"no far shadow\" (ruling B6.5 amended; the far shadow is D-TERRAIN-8's, not built)"
    );
    match sun {
        Some(s) => {
            let up = s.dot(d);
            let elev = up.clamp(-1.0, 1.0).asin().to_degrees();
            let h = s - d * up;
            let off = h.dot(right).atan2(h.dot(look)).to_degrees();
            println!(
                "  VD_STAND_SUN given: the star stands {elev:.2}° up and {off:+.2}° off the look (+ to the right)"
            );
            let light = "ruling-D8-7:the-picture-light";
            let elev_band = Band::Within(SUN_ELEVATION_BAND_DEG.0, SUN_ELEVATION_BAND_DEG.1, light);
            let nose_band = Band::Within(SUN_OFF_NOSE_BAND_DEG.0, SUN_OFF_NOSE_BAND_DEG.1, light);
            reds += usize::from(census("G-REF-SUN-ELEVATION", elev, &elev_band, "deg") == "RED");
            reds +=
                usize::from(census("G-REF-SUN-OFF-NOSE", off.abs(), &nose_band, "deg") == "RED");
        }
        None => println!("  no VD_STAND_SUN given: the sun of one moment is not judged"),
    }

    // ── 6. THE SKYLINE NUMBERS, BESIDE THE REFERENCE'S ─────────────────────────────────────────────
    let (full_rise, full_per60) = judge(&sky.deg);
    let sector = judge_sector(&sky.deg, 0, SECTOR_HALF);
    // One bearing of the 1° grid stands for one degree of span.
    let deg_per_bearing = 360.0 / BEARINGS as f64;
    let sector_per60: [f64; 4] = std::array::from_fn(|k| {
        sector.breaks[k] as f64 * 60.0 / (sector.bearings as f64 * deg_per_bearing)
    });
    println!(
        "\nTHE SKYLINE NUMBERS (the rule of `11` §3: a break stands over both neighbours and over the ±5° median by the threshold; \
         per 60° of bearing). Our {SECTOR_DEG:.0}° sector reads the whole ±5° window for every bearing; the reference dropped 5° \
         at each edge of its picture, so its counts are a lower bound (`11` §3 point 4). A REPORT (ruling T6 ask 5): no floor, \
         no target, no refusal."
    );
    println!(
        "  reading                           | the reference, 70 runs (low / median / high) | its base 55° run | ours, the {SECTOR_DEG:.0}° sector | ours, 360°"
    );
    for (k, t) in THRESHOLDS_DEG.iter().enumerate() {
        println!(
            "  breaks/60° at {t:.2}°               | {:>5.2} / {:>5.2} / {:>5.2}{:<22} | {:>16.2} | {:>18.2} | {:>10.2}",
            REF_BREAKS[k].0,
            REF_BREAKS[k].1,
            REF_BREAKS[k].2,
            if NOISY[k] { " (the trace's noise)" } else { "" },
            REF_BASE_BREAKS[k],
            sector_per60[k],
            full_per60[k]
        );
    }
    println!(
        "  largest rise over the ±5° trend, ° | {:>5.3} / {:>5.3} / {:>5.3}                       | {REF_BASE_RISE_DEG:>16.3} | {:>18.3} | {full_rise:>10.3}",
        REF_RISE_DEG.0, REF_RISE_DEG.1, REF_RISE_DEG.2, sector.largest_rise_deg
    );
    println!(
        "  M1b (before 8c, the recipe's own relief, four stations, 360°): largest rise 0.054°–0.064°, zero breaks at 0.10° and over (`11` §0)"
    );
    let sector_range = sorted[sorted.len() - 1] - sorted[0];
    println!(
        "  the skyline's range over the sector {sector_range:.2}° (the reference 4.87°–7.49° over 45°–65°; a report, never a gate, `11` §9)"
    );
    for (k, tag) in TAGS.iter().enumerate() {
        let (lo, _, hi) = REF_BREAKS[k];
        let envelope = Band::Report(lo, hi, SKYLINE_SOURCE);
        for (scope, value) in [("SECTOR", sector_per60[k]), ("FULL", full_per60[k])] {
            census(
                &format!("G-SKYLINE-{scope}-{tag}"),
                value,
                &envelope,
                "breaks/60deg",
            );
            println!(
                "    {scope} at {:.2}°: {value:.2} breaks per 60° stands {} the reference's envelope {lo:.2}–{hi:.2}{}",
                THRESHOLDS_DEG[k],
                position(value, lo, hi),
                if NOISY[k] {
                    " (`11` §5: this threshold is inside the trace's own noise)"
                } else {
                    ""
                }
            );
        }
    }
    let rise_envelope = Band::Report(REF_RISE_DEG.0, REF_RISE_DEG.2, SKYLINE_SOURCE);
    for (scope, value) in [("SECTOR", sector.largest_rise_deg), ("FULL", full_rise)] {
        census(
            &format!("G-SKYLINE-{scope}-RISE"),
            value,
            &rise_envelope,
            "deg",
        );
        println!(
            "    {scope}: the largest rise {value:.3}° stands {} the reference's envelope {:.3}°–{:.3}°",
            position(value, REF_RISE_DEG.0, REF_RISE_DEG.2),
            REF_RISE_DEG.0,
            REF_RISE_DEG.2
        );
    }

    // THE SECTOR'S OWN SKYLINE AND ITS BREAKS, so the picture and the numbers can be read together.
    let values: Vec<String> = sector_sky.iter().map(|v| format!("{v:.2}")).collect();
    println!(
        "\n  the skyline over the sector, degrees of elevation, one value a degree of bearing from −{SECTOR_HALF}° to +{SECTOR_HALF}° of \
         the look (`11` §7 prints the reference's the same way):"
    );
    println!("  {}", values.join(" "));
    let r = rises(&sky.deg);
    for off in 0..=2 * SECTOR_HALF {
        let b = (BEARINGS + off - SECTOR_HALF) % BEARINGS;
        if !is_break(&sky.deg, &r, b, THRESHOLDS_DEG[0]) {
            continue;
        }
        let passed = THRESHOLDS_DEG
            .iter()
            .filter(|t| is_break(&sky.deg, &r, b, **t))
            .count();
        println!(
            "    a break at {:+}° of the look: rise {:.3}°, the skyline {:.3}°, its crest {:.1} km away; over {passed} of the four thresholds",
            off as i64 - SECTOR_HALF as i64,
            r[b],
            sky.deg[b],
            sky.at_m[b] / 1_000.0
        );
    }

    // ── 7. THE RELIEF 30–60 km AWAY, ON THE SHIPPED GROUND ─────────────────────────────────────────
    let (e1v, e2v) = (DVec3::from_array(e1), DVec3::from_array(e2));
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut held = 0usize;
    for &(dv, top) in &ridge_samples {
        let v = dv - d * dv.dot(d);
        let off = wrap(v.dot(e2v).atan2(v.dot(e1v)) - look_rad).to_degrees();
        if off.abs() <= 0.5 * SECTOR_DEG {
            lo = lo.min(top);
            hi = hi.max(top);
            held += 1;
        }
    }
    // Both tops are radii in metres, so their difference is metres of relief.
    let relief_m = if held > 0 { hi - lo } else { f64::NAN };
    let over_eye_m = if held > 0 { hi - r_f } else { f64::NAN };
    println!(
        "\nTHE RELIEF {:.0}–{:.0} km AWAY in the sector, on the SHIPPED ground ({held} samples): the highest top minus the lowest \
         {relief_m:.0} m; the highest top stands {over_eye_m:.0} m over the eye's own ground (the solved rows gave {:.0} m for this sector)",
        RELIEF_INNER_M / 1_000.0,
        RELIEF_OUTER_M / 1_000.0,
        stand.relief_m
    );
    census(
        "G-REF-RELIEF-30-60KM",
        relief_m,
        &Band::NoneYet(
            "01s2:ridges-1500-3000m-tall-unmarked;11-U16e:no-metre-band-rests-on-the-camera-height",
        ),
        "m",
    );

    // ── 8. THE LINES THAT CAN GO RED: the instrument's own ─────────────────────────────────────────
    println!("\nTHE INSTRUMENT'S OWN LINES (these can go red):");
    // The samples not drawn at the rung the ladder asks for: a coarser rung stood in, or none could.
    reds += usize::from(
        census(
            "G-REF-DETAIL-TO-HORIZON",
            (coarser + sky.missing) as f64,
            &Band::Within(
                0.0,
                0.0,
                "01s2+status-s9:detail-to-the-horizon;ruling-F9:the-coarser-rung-stands-in",
            ),
            "samples",
        ) == "RED",
    );
    // No chunk compared is no reading, and a reading the instrument could not make cannot pass.
    let shipped_differ = if checked == 0 {
        f64::NAN
    } else {
        (differ + refused) as f64
    };
    reds += usize::from(
        census(
            "G-REF-SHIPPED-PATH",
            shipped_differ,
            &Band::Within(
                0.0,
                0.0,
                "SL10:one-generator;the-reader-equals-column_field",
            ),
            "columns",
        ) == "RED",
    );
    println!(
        "\nreference_stand: {reds} census lines RED (the lines that can go red: the detail to the horizon, the shipped path, and \
         the sun when VD_STAND_SUN is given; the skyline lines are a report, ruling T6 ask 5)"
    );

    // ── 9. THE CHECKLIST HINT (the status file, §9) ────────────────────────────────────────────────
    println!(
        "\nTHE CHECKLIST (the status file §9) — what this stand can read, and what it cannot:"
    );
    println!(
        "  READS: the SKYLINE NUMBERS rows (breaks at 0.25° and 0.50°, the largest rise) on the shipped ground at this generator; \
         the RELIEF of the ridges 30–60 km away (the metres, not the snow on them); \"detail to the horizon\" as the count of \
         samples the field could not state at their own rung (G-REF-DETAIL-TO-HORIZON)"
    );
    println!(
        "  CANNOT READ: the snow, the ground's colour and the beach, the forest as a mass, the long shadows, the haze, the clouds \
         (the dressing, D-TERRAIN-8); the valleys that join and the lake in a valley (a skyline sees no drainage); the cliffs, \
         spires and canopy on a crest (`11` §6: the reference counts them and a height field cannot hold them) — the picture \
         carries those"
    );
    ExitCode::SUCCESS
}
