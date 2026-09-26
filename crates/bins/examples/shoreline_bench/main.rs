//! ★ THE SHORELINE BENCH — PLAN STEP 4, SLICE 0: A PROTOTYPE BENCH THAT PAINTS THE DRAWN WATER
//! (ruling B8 item 1; `docs/investigation/2026-09-24/shoreline_design.md` §19 "WHAT FOLLOWS";
//! `docs/investigation/2026-09-25/shoreline_papers_survey.md` §6; the living status file
//! `docs/investigation/2026-09-24/terrain_believability_status.md` §8 row 4).
//!
//! **WHY IT EXISTS.** Two paper rounds of the one-shoreline design found 47 faults, and each cure
//! added a mechanism the next round broke. The owner's word is asked on numbers and pictures, not on
//! a third paper. So this bench computes, for each CANDIDATE RULE, the DRAWN SET — the drawn ground
//! `s` and the drawn water `l` per column, where a column is drawn wet by the sheet's own rule
//! (`s < l`, `vd_terrain::position::water_sheet_sided`) — at every rung a pilot flies through, on
//! eight stands, and reads every line on that drawn set.
//!
//! **THE RULES** (survey §6.1; `vd_bins::shore_rules`): R0 today (bit for bit the shipped column,
//! proven by CHECK-R0 against `vd_terrain::chunk::column_field` on every chunk); C0 the pure contour
//! (W6's hold off); N1 the node-scale side as one field; O1 the ground's own crossing moved from the
//! node-scale line by at most one coarser cell (O1q: a quarter cell); N1 and O1 with and without the
//! fold of rung-F point sides; and, for the high lake's far rungs only, R0w and R0b.
//!
//! **THE WORLD DOES NOT CHANGE.** No generator, recipe, artifact or client drawing code changes: the
//! bench is an instrument. A finding that needs a world change is registered in `DEFERRED.md`.
//!
//! **WHAT IT PRINTS.** A header (the world's versions, the digest, every named choice with its
//! source); the controls (no solve); the solve and the client cache (filled through its own
//! receiver, as `skirt_gap` does); the stands; per stand, window and rung: the self-checks, then
//! every rule's lines; the cross-rung lines per window; SHORE-STEP on transects; the far diagnosis;
//! COST; the deciding number and the Choice-A table; the PANEL index; the phase times and the peak
//! resident memory. Every gate is one line `CENSUS <GATE>.<stand>.<window>.<rule>.rNN <value>
//! <band>[source] <GREEN|RED|NO-BAND|VOID>`.
//!
//! ```text
//! cargo run --release -p vd-bins --example shoreline_bench -- \
//!   --census docs/investigation/2026-09-25/shoreline_bench_census.txt \
//!   --panels docs/investigation/2026-09-25/shoreline_bench \
//!   [--stands belt,gentle] [--rules R0,N1,O1] [--choice k=0.5] [--no-panels] [--threads 10]
//!   [--no-cost] [--no-controls]
//! ```
//!
//! **THE PLAN'S FIXES, stated** (the task asked to fix the plan where it is wrong, and say so):
//! 1. The windows are aligned to the CELLS of their coarsest rung, not its chunks: one chunk of
//!    rung 12 is 254 km, so a chunk-aligned W2 could leave a stand at its edge; whole cells nest the
//!    same (`stands::Window`).
//! 2. Gate names carry the window (`.W2.`): W1 and W2 both read rung 7, W2 and W3 rungs 10–13.
//! 3. The column passes run on several threads, one chunk row a task; every column is a pure
//!    function of its address, so the numbers are the same on any count. COST is read on ONE thread.
//! 4. The example is a directory (`examples/shoreline_bench/`), one file per part.
//!
//! **Example.** A pilot descends over the gentle belt coast. At each ring swap the bench reads the
//! drawn coast under the rules and prints how many pixels the coast moves sideways; it also paints
//! the same coast from above and from an oblique eye, one tile per rule, for the owner's look.

mod census;
mod controls;
mod cost;
mod eval;
mod far;
mod far_window;
mod lines;
mod panels;
mod pyramid;
mod seam;
mod stands;
mod transects;

use std::collections::BTreeMap;
use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_bins::shore_geometry::{RungReader, unit_of};
use vd_bins::shore_rules::{NamedChoices, Nodes, RUNGS_MAX, band_held, shore_band, shore_band_by};
use vd_client::artifact_book::{ArtifactCache, ArtifactReceiver, SharedField};
use vd_core::glam::DVec3;
use vd_core::pose::RealmId;
use vd_recipe::plan::site_direction;
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, CoastCounts, PyramidField};
use vd_terrain::chunk::CHUNK_EDGE;
use vd_terrain::home::{HOME_PLANET_SEED, home_planet, home_solve_words};
use vd_wire::channels::BulkMsg;

use census::{Census, val};
use eval::{Ctx, FAMS, Fold, FoldMap, NVAR, RungOut, Var};
use lines::{LineCtx, Pool};
use panels::{Geo, Panels, TileSrc};
use stands::{Stand, Window};

/// The run's arguments.
struct Args {
    census: Option<String>,
    panels: Option<String>,
    stands: Vec<String>,
    rules: Vec<String>,
    choices: Vec<String>,
    threads: usize,
    cost: bool,
    controls: bool,
    /// A DEVELOPMENT AID only: a scratch file the solved artifact is kept in between iteration
    /// runs (the header says so; the census run solves).
    artifact_cache: Option<String>,
    /// DEVELOPMENT PROBES: `FACE:RUNG:I:J`, each column printed with its parts, then the run stops.
    probes: Vec<String>,
    /// The far window W4 (S0 item 5): 0 off (`--no-far`), 1 beside W0–W3, 2 alone (`--far-only`).
    far: u8,
    /// W-SEAM (S0 item 6): the two windows flush against the busiest coastal cube edge.
    seam: bool,
}

fn args() -> Args {
    let mut a = Args {
        census: None,
        panels: None,
        stands: Vec::new(),
        rules: Vec::new(),
        choices: Vec::new(),
        threads: std::thread::available_parallelism()
            .map_or(4, |n| n.get().saturating_sub(4).clamp(1, 10)),
        cost: true,
        controls: true,
        artifact_cache: None,
        probes: Vec::new(),
        far: 1,
        seam: true,
    };
    let list: Vec<String> = std::env::args().skip(1).collect();
    let mut k = 0;
    let mut no_panels = false;
    while k < list.len() {
        let next = |k: usize| list.get(k + 1).cloned().unwrap_or_default();
        match list[k].as_str() {
            "--census" => {
                a.census = Some(next(k));
                k += 1;
            }
            "--panels" => {
                a.panels = Some(next(k));
                k += 1;
            }
            "--stands" => {
                a.stands = next(k).split(',').map(str::to_string).collect();
                k += 1;
            }
            "--rules" => {
                a.rules = next(k).split(',').map(str::to_string).collect();
                k += 1;
            }
            "--choice" => {
                a.choices.push(next(k));
                k += 1;
            }
            "--threads" => {
                a.threads = next(k).parse().unwrap_or(a.threads);
                k += 1;
            }
            "--artifact-cache" => {
                a.artifact_cache = Some(next(k));
                k += 1;
            }
            "--probe" => {
                a.probes.push(next(k));
                k += 1;
            }
            "--no-panels" => no_panels = true,
            "--no-cost" => a.cost = false,
            "--no-controls" => a.controls = false,
            "--no-far" => a.far = 0,
            "--far-only" => a.far = 2,
            "--no-seam" => a.seam = false,
            "--seam-only" => {
                a.seam = true;
                a.far = 3;
            }
            other => eprintln!("shoreline_bench: an unknown argument `{other}` is ignored"),
        }
        k += 1;
    }
    if no_panels {
        a.panels = None;
    }
    a
}

/// The peak resident memory of this process, in megabytes (`getrusage`: bytes on macOS).
fn peak_rss_mb() -> f64 {
    // SAFETY: `getrusage` writes into the zeroed struct we own; `RUSAGE_SELF` is a valid target.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let ok = unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) } == 0;
    if !ok {
        return f64::NAN;
    }
    let raw = usage.ru_maxrss as f64;
    if cfg!(target_os = "macos") {
        raw / 1.0e6
    } else {
        raw / 1.0e3
    }
}

/// ★ THE CLIENT'S CACHE through its own receiver, exactly as `skirt_gap` fills it.
fn fill_cache(artifact: &Artifact, realm: RealmId) -> ArtifactCache {
    let mut rx = ArtifactReceiver::default();
    rx.accept(BulkMsg::ArtifactHead {
        realm,
        world_tag: 0,
        version: artifact.version,
        edge: artifact.edge,
        digest: artifact.digest(),
        tiles_per_edge: artifact.tiles_per_edge(),
        levels: artifact.pyramid.len() as u32,
        sea_m: artifact.sea_m,
        coast_parts: vd_bins::artifact_store::coast_parts_of(artifact),
    });
    for (k, level) in artifact.pyramid.iter().enumerate().rev() {
        rx.accept(BulkMsg::ArtifactPyramid {
            realm,
            level: k as u32 + 1,
            part: 0,
            parts: 1,
            z_m: level.clone(),
            water_m: artifact.pyramid_water[k].clone(),
        });
    }
    for face in Face::ALL {
        for tx in 0..artifact.tiles_per_edge() {
            for ty in 0..artifact.tiles_per_edge() {
                rx.accept(BulkMsg::ArtifactTile {
                    realm,
                    face: face.index(),
                    tx,
                    ty,
                    rows: artifact.tile(face, tx, ty).to_bytes(),
                });
            }
        }
    }
    let parts = vd_bins::artifact_store::coast_parts_of(artifact);
    for (part, bits) in artifact
        .coast
        .chunks(vd_bins::artifact_source::COAST_PART_BYTES)
        .enumerate()
    {
        rx.accept(BulkMsg::ArtifactCoast {
            realm,
            part: part as u32,
            parts,
            bits: bits.to_vec(),
        });
    }
    rx.book().get(realm).cloned().expect("a cache")
}

/// Everything a stand's run reads.
struct World<'a> {
    body: &'a BodyDefinition,
    artifact: &'a Artifact,
    counts: &'a std::sync::Arc<CoastCounts>,
    cache: &'a ArtifactCache,
    nodes: &'a Nodes,
    /// The rows' nodes with every height outside the shore band replaced (`band_held`): what a
    /// host that holds only the mask, the lake levels and the band reads (CHECK-SHORE-BAND).
    banded: &'a Nodes,
    /// The same under S0's first band rule, which left out the wet nodes at `W − 1` and over
    /// (CHECK-SHORE-BAND-CONTROL: it must draw another N2 somewhere).
    banded_control: &'a Nodes,
    ch: NamedChoices,
    r1: u8,
    f_rung: u8,
    octave_counts: [usize; RUNGS_MAX],
    lift_mu: [f64; RUNGS_MAX],
    slope_mu: [f64; RUNGS_MAX],
    octave_reach: [f64; vd_recipe::height::OCTAVES_CAP],
    octave_rms: [f64; vd_recipe::height::OCTAVES_CAP],
    vars: Vec<Var>,
    threads: usize,
    far: u8,
}

impl World<'_> {
    /// ★ THE READ CONTEXT of one rung on one face.
    fn ctx(&self, face: Face, rung: u8, r0_only: bool) -> Option<Ctx<'_>> {
        self.ctx_over(face, rung, r0_only, self.nodes)
    }

    /// The read context of one rung over another node store (the band-held nodes).
    fn ctx_on<'b>(&'b self, face: Face, rung: u8, nodes: &'b Nodes) -> Option<Ctx<'b>> {
        self.ctx_over(face, rung, false, nodes)
    }

    fn ctx_over<'b>(
        &'b self,
        face: Face,
        rung: u8,
        r0_only: bool,
        nodes: &'b Nodes,
    ) -> Option<Ctx<'b>> {
        let reader = RungReader::new(self.body, self.artifact, self.counts, face, rung)?;
        let fine = self.body.macro_lattice()?;
        let level = reader.level;
        let cache_field = self.cache.field_at_rung(&fine, rung);
        let cache_next = self
            .cache
            .level(level + 1)
            .map(|l| l.clone() as SharedField);
        Some(Ctx {
            body: self.body,
            nodes,
            cache_field,
            cache_next,
            rung,
            face,
            ch: self.ch,
            r1: self.r1,
            f_rung: self.f_rung,
            octave_counts: self.octave_counts,
            first: self.body.first_fine(),
            a_r: self.body.relief_bound_m(rung),
            sea: self.artifact.sea().map(f64::from),
            sea_radius: reader.sea_radius(),
            cpn: fine.cells_per_node,
            r0_only,
            lift_mu: self.lift_mu[usize::from(rung)],
            slope_mu: self.slope_mu[usize::from(rung)],
            octave_reach: self.octave_reach,
            octave_rms: self.octave_rms,
            reader,
        })
    }
}

/// The column's metres along `i` and `j` at `(i, j)` of a rung's face.
fn metric(ctx: &Ctx, i: i32, j: i32) -> (f64, f64) {
    let r = ctx.body.radius_m();
    let d = |a: i32, b: i32| {
        unit_of(site_direction(
            &ctx.reader.charter,
            i32::from(ctx.face.index()),
            a,
            b,
        ))
    };
    (
        (d(i + 1, j) - d(i, j)).length() * r,
        (d(i, j + 1) - d(i, j)).length() * r,
    )
}

/// What a stand hands the run's summary.
#[derive(Default)]
struct StandSummary {
    /// Per rule: the largest SHORE-STEP (px) over the sea's swaps 0 → 1 … 9 → 10, its p99, matched.
    decide: BTreeMap<usize, (f64, f64, usize)>,
}

fn main() {
    let a = args();
    let t_all = Instant::now();
    let mut ch = NamedChoices::default();
    let mut c = Census::new(a.census.as_deref());
    let mut overrides = Vec::new();
    for pair in &a.choices {
        match ch.set(pair) {
            Ok(()) => overrides.push(pair.clone()),
            Err(e) => eprintln!("shoreline_bench: --choice {pair} refused: {e}"),
        }
    }
    let vars: Vec<Var> = if a.rules.is_empty() {
        Var::ALL.to_vec()
    } else {
        let mut v: Vec<Var> = Var::ALL
            .iter()
            .copied()
            .filter(|v| a.rules.iter().any(|r| r == v.name()))
            .collect();
        if !v.contains(&Var::R0) {
            v.insert(0, Var::R0);
        }
        v
    };
    c.say("# THE SHORELINE BENCH, plan step 4 slice 0 (ruling B8 item 1; survey §6; design §19 WHAT FOLLOWS)");
    c.say(format!(
        "# written by `cargo run --release -p vd-bins --example shoreline_bench`; {} threads for the column passes (COST on one); one job on the machine",
        a.threads
    ));
    c.say(format!(
        "WORLD generator {} (vd_terrain::GENERATOR_VERSION)",
        vd_terrain::GENERATOR_VERSION
    ));
    c.say("THE NAMED CHOICES (ruling T9: each with its default and its source):");
    for line in ch.describe() {
        c.say(format!("  - {line}"));
    }
    c.say(format!(
        "  overrides this run: {}",
        if overrides.is_empty() {
            "none".to_string()
        } else {
            overrides.join(", ")
        }
    ));
    c.say(format!(
        "THE RULES: {}",
        vars.iter().map(|v| v.name()).collect::<Vec<_>>().join(", ")
    ));
    let mut phases: Vec<(String, f64)> = Vec::new();

    // ── THE CONTROLS FIRST (no solve) ────────────────────────────────────────────────────────
    let t = Instant::now();
    let bands = if a.controls {
        controls::run(&mut c, a.threads, 1_984)
    } else {
        c.say("THE CONTROLS: skipped (--no-controls); the shape gates print NO-BAND for KINK");
        controls::Bands {
            kink: Vec::new(),
            lattice: Vec::new(),
            kink_power: false,
            void: false,
        }
    };
    phases.push(("controls".into(), t.elapsed().as_secs_f64()));

    // ── THE SOLVE AND THE CACHE ──────────────────────────────────────────────────────────────
    let t = Instant::now();
    let artifact = match a
        .artifact_cache
        .as_deref()
        .and_then(vd_bins::artifact_file::read)
    {
        Some(art) => {
            c.say(format!(
                "# DEVELOPMENT RUN: the artifact was READ from the scratch file {:?}, not solved; a census run solves",
                a.artifact_cache
            ));
            art
        }
        None => {
            let art = run_solve(&SolveJob {
                body: home_planet(),
                words: home_solve_words(),
            })
            .expect("the home planet solves");
            if let Some(path) = a.artifact_cache.as_deref() {
                let _ = vd_bins::artifact_file::write(path, &art);
            }
            art
        }
    };
    phases.push(("solve".into(), t.elapsed().as_secs_f64()));
    let d = artifact.digest();
    c.say(format!(
        "ARTIFACT version {} digest {:016x}{:016x}; the sea {} m over the ladder radius",
        artifact.version,
        d[0],
        d[1],
        artifact.sea().unwrap_or(0)
    ));
    let body = home_planet().with_sea_m(artifact.sea());
    let fine = body.macro_lattice().expect("a lattice");
    let counts = artifact.coast_counts();
    let t = Instant::now();
    let realm = RealmId::Planet(HOME_PLANET_SEED);
    let cache = fill_cache(&artifact, realm);
    let levels = artifact.pyramid.len() as u32;
    let mut count_diff = 0usize;
    for k in 1..=levels {
        let same = cache
            .level(k)
            .and_then(|l| l.counts.as_ref())
            .is_some_and(|cc| **cc == *counts);
        count_diff += usize::from(!same);
    }
    c.judged(
        "CHECK-COUNTS",
        &format!("{count_diff}of{levels}levels-differ"),
        "0[plan§2;W15.1-item-5]",
        count_diff == 0,
    );
    let nodes = Nodes::of(&artifact);
    // THE SHORE BAND (ruling B14 item 1) and the nodes a band-holding host reads.
    let band_m = ch.word_band_m.min(f64::from(i16::MAX));
    let in_band = shore_band(&nodes, &fine, band_m);
    let banded = band_held(&nodes, &in_band);
    let first_rule: fn(i32, bool, i32, i32) -> bool = |z, wet, w, band| {
        if wet {
            z > w - band && z < w - 1
        } else {
            z > w - band && z < w + band
        }
    };
    let banded_control = band_held(&nodes, &shore_band_by(&nodes, &fine, band_m, first_rule));
    c.say(format!(
        "the shore band: {} of {} nodes hold their height ({} m; `shore_rules::shore_band`); the rest read the band's edge (CHECK-SHORE-BAND on W4)",
        in_band.iter().filter(|&&b| b).count(),
        in_band.len(),
        ch.word_band_m
    ));
    drop(in_band);
    phases.push(("cache+nodes".into(), t.elapsed().as_secs_f64()));
    let rungs = body.ladder().rungs;
    let r1 = (0..rungs)
        .find(|&r| PyramidField::level_for(&fine, levels, r) >= 1)
        .unwrap_or(rungs - 1);
    let f_rung = r1.saturating_sub(ch.fold_under);
    let mut octave_counts = [0usize; RUNGS_MAX];
    for (r, slot) in octave_counts
        .iter_mut()
        .enumerate()
        .take(usize::from(rungs))
    {
        *slot = body.octaves_at(r as u8).len();
    }
    c.say(format!(
        "the body: {rungs} rungs; the fine lattice {} nodes an edge, {} rung-0 cells a node ({:.0} m); {} pyramid levels; R1 = {r1} (the first rung that reads a level); F = {f_rung}; the first fine octave {}; the octave counts by rung {:?}",
        fine.edge,
        fine.cells_per_node,
        fine.node_m(),
        levels,
        body.first_fine(),
        &octave_counts[..usize::from(rungs)]
    ));
    // N2's lift: the fine octaves' mean at share one, by rung, over 4 096 fine nodes spread over the
    // sphere (the ridged octaves stand on a positive mean; the smooth ones on nought).
    let lift_mu = {
        let charter = body.plan_charter(0, Face::PosX);
        let first = body.first_fine();
        let samples = 4_096usize;
        let stride = (fine.node_count() / samples).max(1);
        let mut sums = [0.0f64; RUNGS_MAX];
        for s in 0..samples {
            let terms = vd_bins::shore_rules::fine_terms(
                &charter,
                first,
                octave_counts[0],
                fine.direction((s * stride) as u32),
            );
            for (r, sum) in sums.iter_mut().enumerate().take(usize::from(rungs)) {
                *sum += vd_bins::shore_rules::metres(vd_bins::shore_rules::term_sum(
                    &terms,
                    first,
                    octave_counts[r],
                ));
            }
        }
        sums.map(|x| x / samples as f64)
    };
    c.say(format!(
        "N2's lift: the fine octaves' mean at share one by rung (4 096 fine nodes): {:?} m",
        lift_mu[..usize::from(rungs)]
            .iter()
            .map(|x| (x * 10.0).round() / 10.0)
            .collect::<Vec<f64>>()
    ));
    // N2's bank: the fine octaves' RMS slope at share one, per rung, along one axis at the rung's
    // own column — the step between two rung-0 columns one rung-r column apart, over 2 048 random
    // rung-0 sites (a fixed generator: the numbers repeat).
    let slope_mu = {
        let charter = body.plan_charter(0, Face::PosX);
        let first = body.first_fine();
        let n0 = i64::from(body.ladder().cells_per_edge(0));
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut sq = [0.0f64; RUNGS_MAX];
        let mut count = [0usize; RUNGS_MAX];
        let cap = vd_recipe::height::OCTAVES_CAP;
        let mut oct_h = vec![0.0f64; cap];
        let mut oct_s = vec![0.0f64; cap];
        let mut oct_n = 0usize;
        let fine_sum = |d: [vd_recipe::Gi; 3], k: usize| {
            vd_bins::shore_rules::metres(vd_bins::shore_rules::term_sum(
                &vd_bins::shore_rules::fine_terms(&charter, first, k, d),
                first,
                k,
            ))
        };
        for _ in 0..2_048 {
            let face = (next() % 6) as i32;
            let span = (n0 - 4_096).max(1) as u64;
            let i = (next() % span) as i32 + 1_024;
            let j = (next() % span) as i32 + 1_024;
            let d0 = site_direction(&charter, face, i, j);
            // Each fine octave's height and its slope over two rung-0 columns (the finest octave's
            // wave is some fifty of them).
            {
                let k = octave_counts[0];
                let t0 = vd_bins::shore_rules::fine_terms(&charter, first, k, d0);
                let ti = vd_bins::shore_rules::fine_terms(
                    &charter,
                    first,
                    k,
                    site_direction(&charter, face, i + 2, j),
                );
                let tj = vd_bins::shore_rules::fine_terms(
                    &charter,
                    first,
                    k,
                    site_direction(&charter, face, i, j + 2),
                );
                let h = 2.0 * f64::from(vd_seed::ladder::cell_m(0));
                for o in first..k.min(cap) {
                    let (a, b, c2) = (
                        vd_bins::shore_rules::metres(t0[o]),
                        vd_bins::shore_rules::metres(ti[o]),
                        vd_bins::shore_rules::metres(tj[o]),
                    );
                    oct_h[o] += a * a;
                    oct_s[o] += ((b - a) / h).powi(2) + ((c2 - a) / h).powi(2);
                }
                oct_n += 1;
            }
            for r in 0..usize::from(rungs) {
                let k = octave_counts[r];
                if k <= first || r > 11 {
                    continue;
                }
                let step = 1i32 << r;
                let di = site_direction(&charter, face, i + step, j);
                let dj = site_direction(&charter, face, i, j + step);
                let h = f64::from(vd_seed::ladder::cell_m(r as u8));
                let s0 = fine_sum(d0, k);
                let gi = (fine_sum(di, k) - s0) / h;
                let gj = (fine_sum(dj, k) - s0) / h;
                sq[r] += gi * gi + gj * gj;
                count[r] += 2;
            }
        }
        let mut out = [0.0f64; RUNGS_MAX];
        for r in 0..RUNGS_MAX {
            if count[r] > 0 {
                out[r] = (sq[r] / count[r] as f64).sqrt();
            }
        }
        let mut reach = [0.0f64; vd_recipe::height::OCTAVES_CAP];
        let mut rms_of = [0.0f64; vd_recipe::height::OCTAVES_CAP];
        let mut lines = Vec::new();
        for o in first..octave_counts[0].min(cap) {
            let rms_h = (oct_h[o] / oct_n.max(1) as f64).sqrt();
            let rms_s = (oct_s[o] / (2 * oct_n.max(1)) as f64).sqrt();
            rms_of[o] = rms_h;
            if rms_s > 0.0 {
                reach[o] = vd_bins::shore_rules::FADE_PEAK_SLOPE * rms_h / rms_s;
            }
            let lambda = body.radius_m() / vd_terrain::body::octave_frequency(&charter.octaves[o]);
            lines.push(format!(
                "octave {o}: wave {:.0} m, RMS height {:.1} m, RMS slope {:.4}, reach {:.0} m",
                lambda, rms_h, rms_s, reach[o]
            ));
        }
        (out, reach, rms_of, lines)
    };
    let (slope_mu, octave_reach, octave_rms, octave_lines) = slope_mu;
    // ★ THE REACH LAW (the build design's S0 item 1, Q2): C_kind = 15/8 x RMS height / (RMS slope x
    // wavelength), measured over UNIT octaves of the recipe's own noise — not over the table — so both
    // hosts can state it as two constants; then each fine octave's law reach is C_kind x its wave.
    let (c_smooth, c_ridged) = (
        unit_octave_c(&body, vd_recipe::height::OCTAVE_SMOOTH, 1_000.0, 2_048),
        unit_octave_c(&body, vd_recipe::height::OCTAVE_RIDGED, 1_000.0, 2_048),
    );
    let law_reach = {
        let charter = body.plan_charter(0, Face::PosX);
        let mut out = [0.0f64; vd_recipe::height::OCTAVES_CAP];
        for o in body.first_fine()..octave_counts[0].min(vd_recipe::height::OCTAVES_CAP) {
            let oct = &charter.octaves[o];
            let lambda = body.radius_m() / vd_terrain::body::octave_frequency(oct);
            let cc = if oct.kind == vd_recipe::height::OCTAVE_RIDGED {
                c_ridged
            } else {
                c_smooth
            };
            out[o] = cc * lambda;
            c.say(format!(
                "N2's reach law: octave {o} ({}, wave {lambda:.0} m): the statistic {:.0} m, the law {:.0} m ({:+.1} %), the spectrum's sine law {:.0} m",
                if oct.kind == vd_recipe::height::OCTAVE_RIDGED { "ridged" } else { "smooth" },
                octave_reach[o],
                out[o],
                100.0 * (out[o] / octave_reach[o].max(1e-9) - 1.0),
                vd_bins::shore_rules::FADE_PEAK_SLOPE * lambda / std::f64::consts::TAU
            ));
        }
        out
    };
    c.say(format!(
        "N2's reach law: C_smooth {c_smooth:.4}, C_ridged {c_ridged:.4} (15/8 x RMS height / (RMS slope per axis x wave), unit octaves of the recipe's noise, 1 000 m wave, 2 048 seeded sites x 4 seeds); in use: {}",
        if ch.n2_reach_law { "the law" } else { "the statistic" }
    ));
    let octave_reach = if ch.n2_reach_law {
        law_reach
    } else {
        octave_reach
    };
    for l in &octave_lines {
        c.say(format!(
            "N2's ramp per octave (share one, 2 048 sites): {l}"
        ));
    }
    c.say(format!(
        "N2's bank: the fine octaves' RMS slope at share one by rung, one axis at the rung's column (2 048 sites): {:?}",
        slope_mu[..usize::from(rungs)]
            .iter()
            .map(|x| (x * 1_000.0).round() / 1_000.0)
            .collect::<Vec<f64>>()
    ));
    let world = World {
        body: &body,
        artifact: &artifact,
        counts: &counts,
        cache: &cache,
        nodes: &nodes,
        banded: &banded,
        banded_control: &banded_control,
        ch,
        r1,
        f_rung,
        octave_counts,
        lift_mu,
        slope_mu,
        octave_reach,
        octave_rms,
        vars: vars.clone(),
        threads: a.threads,
        far: a.far,
    };

    if !a.probes.is_empty() {
        for p in &a.probes {
            let w: Vec<i64> = p.split(':').filter_map(|x| x.parse().ok()).collect();
            if w.len() == 7 {
                // A profile: FACE:RUNG:I:J0:J1:STEP:AXIS (AXIS 0 walks j, 1 walks i).
                let face = Face::from_index(w[0] as u8).unwrap_or(Face::PosX);
                if let Some(ctx) = world.ctx(face, w[1] as u8, false) {
                    let mut caches = eval::Caches::default();
                    let mut ck = eval::Checks::default();
                    let mut t = w[3];
                    while t <= w[4] {
                        let (i, j) = if w[6] == 0 { (w[2], t) } else { (t, w[2]) };
                        if let Some(o) =
                            ctx.eval(i as i32, j as i32, &Fold::none(), &mut caches, &mut ck)
                        {
                            let row: Vec<String> = Var::ALL
                                .iter()
                                .map(|v| {
                                    let d = o.drawn[v.idx()];
                                    format!(
                                        "{}={:.1}{}/g{:.1}/f{:02x}",
                                        v.name(),
                                        d.s,
                                        if d.wet() { "w" } else { "" },
                                        d.g,
                                        d.flags
                                    )
                                })
                                .collect();
                            c.say(format!(
                                "profile ({i}, {j}): B {:.1} f {:.1} raw {:.1} | {}",
                                o.shared.b,
                                o.shared.f,
                                o.shared.h_raw,
                                row.join(" ")
                            ));
                        }
                        t += w[5];
                    }
                }
                continue;
            }
            if w.len() != 4 {
                continue;
            }
            let face = Face::from_index(w[0] as u8).unwrap_or(Face::PosX);
            if let Some(ctx) = world.ctx(face, w[1] as u8, false) {
                for line in eval::probe(&ctx, w[2] as i32, w[3] as i32) {
                    c.say(line);
                }
            }
        }
        return;
    }

    // ── THE STANDS ───────────────────────────────────────────────────────────────────────────
    let t = Instant::now();
    c.say("\n== THE STANDS (plan §3.1) ==");
    let stand_list = stands::pick_all(&body, &artifact, &fine, &nodes, &mut c, &a.stands);
    phases.push(("stands".into(), t.elapsed().as_secs_f64()));
    let mut panels = Panels {
        dir: a.panels.clone(),
        budget: ch.panel_budget_bytes,
        spent: 0,
        files: Vec::new(),
    };
    let mut pool: Pool = BTreeMap::new();
    let mut summaries: BTreeMap<String, StandSummary> = BTreeMap::new();
    for stand in stand_list.iter().filter(|_| a.far != 3) {
        let t = Instant::now();
        let s = run_stand(&mut c, &world, stand, &bands, &mut panels, &mut pool);
        summaries.insert(stand.name.to_string(), s);
        phases.push((format!("stand {}", stand.name), t.elapsed().as_secs_f64()));
        c.say(format!(
            "   (stand {} took {:.0} s; peak resident {:.0} MB)",
            stand.name,
            t.elapsed().as_secs_f64(),
            peak_rss_mb()
        ));
    }
    // W-SEAM (S0 item 6), with its own lake pool so the stands' pooled lines stay the stands'.
    if a.seam && a.far != 2 {
        let t = Instant::now();
        let mut seam_pool: Pool = BTreeMap::new();
        seam::run(&mut c, &world, &mut seam_pool);
        phases.push(("seam".into(), t.elapsed().as_secs_f64()));
    }
    c.say("\n== THE POOLED LAKE LINE (every stand) ==");
    lines::pooled_lines(&mut c, &pool);

    // ── COST ─────────────────────────────────────────────────────────────────────────────────
    if a.cost {
        let t = Instant::now();
        c.say("\n== COST (plan §8.6; one thread) ==");
        let dir3 = |d: [f64; 3]| DVec3::new(d[0], d[1], d[2]).normalize();
        let plain = stands::node_dir(
            &fine,
            vd_bins::drainage_gates::node_under(&fine, dir3(vd_bins::drainage_gates::PLAIN)),
        );
        cost::shipped(
            &mut c,
            &body,
            &cache,
            realm,
            &nodes,
            "plain",
            plain,
            &[7, 9],
        );
        cost::shipped(
            &mut c,
            &body,
            &cache,
            realm,
            &nodes,
            "belt",
            dir3(vd_bins::drainage_gates::BELT),
            &[7, 9],
        );
        whatif_cost(&mut c, &world, &stand_list);
        phases.push(("cost".into(), t.elapsed().as_secs_f64()));
    }

    // ── THE DECIDING NUMBER AND THE CHOICE-A TABLE ───────────────────────────────────────────
    choice_a(&mut c, &summaries);

    // ── THE PANEL INDEX, THE TIMES, THE MEMORY ───────────────────────────────────────────────
    c.say("\n== THE PANELS ==");
    for (name, size, caption) in &panels.files {
        c.say(format!("  {name}  {size} bytes  {caption}"));
    }
    c.say(format!(
        "  {} panels, {} bytes in all (budget {})",
        panels.files.len(),
        panels.spent,
        panels.budget
    ));
    if !c.voided.is_empty() {
        c.say(format!(
            "######## GLOBAL VOID BANNER: {} stand(s) had a RED self-check: {:?} ########",
            c.voided.len(),
            c.voided
        ));
    }
    c.say("\n== THE PHASES ==");
    for (name, s) in &phases {
        c.say(format!("  {name}: {s:.1} s"));
    }
    c.say(format!(
        "  the whole run: {:.1} s; peak resident memory {:.0} MB",
        t_all.elapsed().as_secs_f64(),
        peak_rss_mb()
    ));
}

/// ★ ONE STAND: its windows, every rung's checks and lines, the cross-rung lines, the panels, the
/// transects and the far diagnosis.
fn run_stand(
    c: &mut Census,
    w: &World,
    stand: &Stand,
    bands: &controls::Bands,
    panels: &mut Panels,
    pool: &mut Pool,
) -> StandSummary {
    c.begin_stand(stand.name);
    c.say(format!(
        "\n==================== THE STAND `{}` ====================",
        stand.name
    ));
    if w.far == 2 {
        far_window::run(c, w, stand);
        c.end_stand();
        return StandSummary::default();
    }
    let fine = w.body.macro_lattice().expect("a lattice");
    let wins = stands::windows_of(w.body, stand);
    let mut transects: Vec<transects::Transect> = Vec::new();
    let mut map7_w2: Option<FoldMap> = None;
    let mut w2_metric = (f64::NAN, f64::NAN, f64::NAN);
    let pocket = matches!(stand.name, "pocket1" | "pocket2");
    for (wi, win) in wins.iter().enumerate() {
        let r0_only = wi == 3;
        let vars: Vec<Var> = if r0_only {
            vec![Var::R0]
        } else {
            w.vars.clone()
        };
        c.say(format!(
            "\n-- {} of `{}`: face {:?}, rungs {}..={}, {} columns at rung {} from ({}, {}), pulled ({}, {}) columns to stay on the face",
            win.name, stand.name, win.face, win.fine, win.last, win.n, win.fine, win.i0, win.j0, win.pull.0, win.pull.1
        ));
        let mut fold_map: Option<FoldMap> = None;
        let mut cross: Vec<Vec<Vec<u32>>> = vec![Vec::new(); NVAR];
        let mut dims: Vec<(i32, i32, usize)> = Vec::new();
        for r in win.rungs() {
            let Some(ctx) = w.ctx(win.face, r, r0_only) else {
                c.say(format!("   rung {r}: REFUSED — no read at this rung"));
                continue;
            };
            let (i0, j0, n) = win.at(r);
            let fold = Fold {
                map: fold_map.as_ref(),
                ctx_f: None,
                pyramid: None,
            };
            let out = eval::eval_window(&ctx, i0, j0, n, &fold, w.threads, true);
            let (m_i, m_j) = metric(&ctx, i0 + n / 2, j0 + n / 2);
            let m_col = 0.5 * (m_i + m_j);
            c.say(format!(
                "   {} rung {r}: {n} x {n} columns of {m_i:.2} x {m_j:.2} m ({:.2} km a side), level {}; {} columns read in {:.1} s",
                win.name,
                n as f64 * m_col / 1_000.0,
                ctx.reader.level,
                out.checks.columns,
                out.seconds
            ));
            let sheet = if [0u8, 3, 7, 10, 11].contains(&r) {
                let edge = CHUNK_EDGE as i32;
                eval::sheet_check(&ctx, (i0 + n / 2) / edge, (j0 + n / 2) / edge)
            } else {
                None
            };
            let lx = LineCtx {
                stand: stand.name,
                window: win.name,
                rung: r,
                face: win.face,
                m_i,
                m_j,
                m_col,
                q_r: f64::from(vd_seed::ladder::cell_m(r)) / 128.0,
                sea: w.artifact.sea().map(f64::from),
                pocket_stand: pocket,
                fine: &fine,
                nodes: w.nodes,
                vars: &vars,
                f_rung: w.f_rung,
                wall_rung: wi < 3 && r == win.fine,
                shape_rung: wi == 2 && [7u8, 9, 10].contains(&r),
                choices: w.ch,
                kink_band: &|node| {
                    if bands.kink_power {
                        bands.kink_at(node)
                    } else {
                        None
                    }
                },
                lattice_band: &|cols| bands.lattice_at(cols),
                shape_void: bands.void,
            };
            lines::check_lines(c, &lx, &out, sheet);
            lines::rule_lines(c, &lx, &out, pool);
            // The gentle coast's own slope, measured on W2 at rung 7.
            if stand.name == "gentle" && wi == 2 && r == 7 {
                gentle_measured(c, &out, w.nodes.sea_m, m_i, m_j);
            }
            // The panels at rungs 3 (W1), 7 and 11 (W2).
            if (wi == 1 && r == 3) || (wi == 2 && (r == 7 || r == 11)) {
                let geo = Geo {
                    face: win.face,
                    i0,
                    j0,
                    n_cells: w.body.ladder().cells_per_edge(r),
                    radius: w.body.radius_m(),
                    charter: &ctx.reader.charter,
                    m_col,
                };
                panels.composites(c, stand.name, win.name, &out, &geo, m_i, m_j, "");
            }
            // The second pocket's bump rows (R0 at rungs 3, 7 and 10).
            if stand.name == "pocket2" && ((wi == 1 && r == 3) || (wi == 2 && (r == 7 || r == 10)))
            {
                far::bumps(c, w.body, &ctx.reader, &out, win.name, m_i, m_j, w.nodes);
            }
            // The transects, placed on W2's rung-7 ground.
            if wi == 2 && r == 7 {
                transects = transects::place(&out, &fine, w.nodes, win.face, m_i, m_j, &w.ch);
                w2_metric = (m_i, m_j, m_col);
                c.say(format!(
                    "   the transects: {} over the sea, {} over the lakes",
                    transects.iter().filter(|t| t.kind == 0).count(),
                    transects.iter().filter(|t| t.kind == 1).count()
                ));
            }
            // The fold's samples at rung F.
            if r == w.f_rung && !r0_only {
                fold_map = Some(FoldMap {
                    i0,
                    j0,
                    n,
                    keys: out
                        .cols
                        .iter()
                        .map(|col| col.as_ref().map_or([0; FAMS], |o| o.keys))
                        .collect(),
                });
            }
            // The far panel of pocket 2 (W3 at rung 11, R0).
            if stand.name == "pocket2" && wi == 3 && r == 11 {
                panels.far_row(
                    c,
                    "pocket2_r11far_top_R0.png",
                    &[TileSrc::of(&out, Var::R0)],
                    "R0",
                    m_i,
                    m_j,
                    n as f64 * m_col / 1_000.0,
                    r,
                );
            }
            for v in &vars {
                cross[v.idx()].push(
                    out.cols
                        .iter()
                        .map(|col| col.as_ref().map_or(0, |o| o.drawn[v.idx()].wet_key()))
                        .collect(),
                );
            }
            dims.push((i0, j0, n as usize));
        }
        for v in &vars {
            lines::cross_lines(
                c,
                stand.name,
                win.name,
                *v,
                win.fine,
                w.f_rung,
                &cross[v.idx()],
                &dims,
            );
        }
        if wi == 2 {
            map7_w2 = fold_map;
        }
    }
    // SHORE-STEP.
    let mut summary = StandSummary::default();
    if !transects.is_empty() {
        let t = Instant::now();
        let ctxs: Vec<Option<Ctx>> = (0..=13u8).map(|r| w.ctx(stand.face, r, false)).collect();
        let steps = transects::measure(
            &transects,
            &ctxs,
            map7_w2.as_ref(),
            None,
            w2_metric.0,
            w2_metric.1,
            w2_metric.2,
            &w.ch,
            w.threads,
            false,
        );
        c.say(format!(
            "   SHORE-STEP: {} transects walked in {:.1} s",
            transects.len(),
            t.elapsed().as_secs_f64()
        ));
        summary.decide = transects::print(
            c,
            stand.name,
            &steps,
            &transects,
            &w.vars,
            w.f_rung,
            w.r1,
            0..13,
        );
    } else {
        c.say("   SHORE-STEP: no transect (no water's contour in W2)");
    }
    // The high lake's far diagnosis and its far panel.
    if stand.name == "highlake"
        && let Some(level) = stand.lake_level
    {
        let tiles = far::high_lake(
            c, w.body, w.artifact, w.counts, w.nodes, &wins[3], level, w.threads,
        );
        if !tiles.is_empty() {
            let (i0, j0, n) = wins[3].at(11);
            if let Some(ctx) = w.ctx(stand.face, 11, true) {
                let (m_i, m_j) = metric(&ctx, i0 + n / 2, j0 + n / 2);
                panels.far_row(
                    c,
                    "highlake_r11far_top_R0-R0w-R0b.png",
                    &tiles,
                    "R0,R0w,R0b",
                    m_i,
                    m_j,
                    n as f64 * 0.5 * (m_i + m_j) / 1_000.0,
                    11,
                );
            }
        }
    }
    if w.far == 1 {
        far_window::run(c, w, stand);
    }
    c.end_stand();
    summary
}

/// ★ THE GENTLE COAST'S MEASURED SLOPE on W2 at rung 7: the median `|∇B|` over the columns where
/// `B` crosses the sea (a reading beside the rule's node-scale pick).
fn gentle_measured(c: &mut Census, out: &RungOut, sea_m: i16, m_i: f64, m_j: f64) {
    let n = out.n;
    let sea = f64::from(sea_m);
    let b = |k: usize| out.cols[k].as_ref().map(|o| f64::from(o.shared.b));
    let mut slopes = Vec::new();
    for bb in 1..n.saturating_sub(1) {
        for aa in 1..n - 1 {
            let k = bb * n + aa;
            let (Some(here), Some(right), Some(upv)) = (b(k), b(k + 1), b(k + n)) else {
                continue;
            };
            if ((here - sea) * (right - sea) < 0.0) || ((here - sea) * (upv - sea) < 0.0) {
                let (Some(l), Some(dn)) = (b(k - 1), b(k - n)) else {
                    continue;
                };
                let gx = (right - l) / (2.0 * m_i);
                let gy = (upv - dn) / (2.0 * m_j);
                slopes.push(gx.hypot(gy));
            }
        }
    }
    c.reading(
        "GENTLE-SLOPE-W2.gentle.W2.C0.r07",
        &format!(
            "median{}/{}cols",
            val(vd_bins::shore_shape::quantile(&slopes, 0.5), 5, ""),
            slopes.len()
        ),
        "|grad-B|-along-B=sea;a-reading-beside-the-rule",
    );
}

/// ★ COST-WHATIF (plan §8.6): the what-if's own extra time per column on ONE thread, split by the
/// candidate count, on 20 000 columns of the plain's and the belt's W2 at rung 7.
fn whatif_cost(c: &mut Census, w: &World, stands: &[Stand]) {
    for s in stands.iter().filter(|s| matches!(s.name, "plain" | "belt")) {
        let win = stands::windows_of(w.body, s)[2];
        let Some(ctx) = w.ctx(win.face, 7, false) else {
            continue;
        };
        let (i0, j0, n) = win.at(7);
        let mut caches = eval::Caches::default();
        let mut ck = eval::Checks::default();
        let fold = Fold::none();
        let step = ((n * n) as usize / 20_000).max(1);
        for k in (0..(n * n) as usize).step_by(step) {
            let (a, b) = ((k % n as usize) as i32, (k / n as usize) as i32);
            let _ = ctx.eval(i0 + a, j0 + b, &fold, &mut caches, &mut ck);
        }
        for (slot, name) in ["0cand", "1cand", "2+cand"].iter().enumerate() {
            let (ns, cnt) = ck.whatif_ns[slot];
            c.reading(
                &format!("COST-WHATIF.{}.r07.{name}", s.name),
                &format!("{}ns/col/{cnt}cols", val(ns / cnt.max(1) as f64, 0, "")),
                "one-thread;every-rule-together;instrument-code",
            );
        }
    }
}

/// ★ THE DECIDING NUMBER AND THE CHOICE-A TABLE (survey §6.4). The bench chooses nothing (K9).
fn choice_a(c: &mut Census, summaries: &BTreeMap<String, StandSummary>) {
    c.say("\n== THE DECIDING NUMBER (survey §6.4) ==");
    if let Some(g) = summaries.get("gentle") {
        let get = |v: Var| g.decide.get(&v.idx()).copied();
        let line = |v: Var| {
            get(v).map_or("none".to_string(), |(m, p, n)| {
                format!(
                    "{}:max{}px/p99{}px/{n}pairs",
                    v.name(),
                    val(m, 3, ""),
                    val(p, 3, "")
                )
            })
        };
        let o1 = get(Var::O1);
        let beside: Vec<String> = [
            Var::N2,
            Var::O1nf,
            Var::O1q,
            Var::N1,
            Var::N1nf,
            Var::C0,
            Var::R0,
        ]
        .iter()
        .map(|&v| line(v))
        .collect();
        match o1 {
            Some((m, p, n)) => c.line(
                "DECIDING-NUMBER.gentle.O1",
                &format!(
                    "max{}px/p99{}px/{n}pairs;beside:{}",
                    val(m, 3, ""),
                    val(p, 3, ""),
                    beside.join(";")
                ),
                "<=2px,p99<=1px[T7;W17.1;survey§6.4]",
                if n < 200 {
                    "NO-BAND"
                } else if m <= 2.0 && p <= 1.0 {
                    "GREEN"
                } else {
                    "RED"
                },
            ),
            None => c.say("DECIDING-NUMBER: the gentle stand did not run or has no sea transect"),
        }
        // The walls where O1's bound acts, beside it.
        let walls: Vec<String> = c
            .all
            .iter()
            .filter(|(gate, ..)| gate.starts_with("CLAMP-BINDS.gentle.") && gate.contains(".O1."))
            .map(|(gate, value, _, verdict)| format!("{gate}={value}({verdict})"))
            .collect();
        c.say(format!(
            "   the walls where O1's clamp binds: {}",
            walls.join("  ")
        ));
    }
    c.say("\n== THE CHOICE-A TABLE (survey §6.4; the bench recommends nothing) ==");
    for stand in ["gentle", "plain"] {
        for v in [Var::O1, Var::N1, Var::N2] {
            let mine = |prefix: &str| -> Vec<(String, String, String)> {
                c.all
                    .iter()
                    .filter(|(gate, ..)| {
                        gate.starts_with(prefix)
                            && gate.contains(&format!(".{stand}."))
                            && gate.contains(&format!(".{}.", v.name()))
                    })
                    .map(|(g, value, _, verdict)| (g.clone(), value.clone(), verdict.clone()))
                    .collect()
            };
            // (red, green, no-band) of the lines with this prefix.
            let count = |prefix: &str| -> (usize, usize, usize) {
                let l = mine(prefix);
                (
                    l.iter().filter(|x| x.2 == "RED").count(),
                    l.iter().filter(|x| x.2 == "GREEN").count(),
                    l.iter().filter(|x| x.2 == "NO-BAND").count(),
                )
            };
            let word = |(r, g, n): (usize, usize, usize)| -> String {
                let verdict = if r > 0 {
                    "FAIL"
                } else if g > 0 {
                    "PASS"
                } else {
                    "NO-BAND"
                };
                format!("{verdict}(red{r}/green{g}/no-band{n})")
            };
            let crit = [
                ("1 SHORE-STEP", "SHORE-STEP"),
                ("2 CHECK-DRAWN-IS-SIDE", "CHECK-DRAWN-IS-SIDE"),
                ("2 G-SEA-OVER-DRY", "G-SEA-OVER-DRY"),
                ("2 LAKE-APPEAR", "LAKE-APPEAR"),
                ("2 LAKE-FLICKER", "LAKE-FLICKER"),
                ("3 WALL-on-CLAMP-BINDS", "CLAMP-BINDS"),
                ("4 G-SHORE-LATTICE", "G-SHORE-LATTICE"),
                ("4 G-COAST-KINK", "G-COAST-KINK"),
                ("4 G-LAKE-FRACTAL", "G-LAKE-FRACTAL"),
                ("4 G-LAKE-DBC", "G-LAKE-DBC"),
            ];
            let parts: Vec<String> = crit
                .iter()
                .map(|(label, prefix)| format!("{label} {}", word(count(prefix))))
                .collect();
            c.say(format!("  {stand} {}: {}", v.name(), parts.join("; ")));
        }
    }
    let _ = (Window::at, PyramidField::level_for);
}

/// ★ ONE KIND'S REACH CONSTANT over UNIT octaves of the recipe's own noise (the build design's §4):
/// `15/8 × RMS height / (RMS slope per axis × wavelength)`, the height and the slope read through the
/// recipe's own octave sum at share one, at `sites` seeded rung-0 sites on the six faces, the slope
/// over two rung-0 cells, averaged over four noise seeds. A unit octave's height and slope both scale
/// with its amplitude, and its slope with one over its wave, so the constant needs one wave only.
fn unit_octave_c(body: &BodyDefinition, kind: vd_recipe::Gi, lambda_m: f64, sites: usize) -> f64 {
    use vd_recipe::Gi;
    use vd_recipe::height::{AMP_BITS, OCTAVE_FINE, OCTAVES_CAP, Octave, relief_parts_from};
    use vd_recipe::noise::{NOISE_BITS, NOISE_ONE};
    let charter = body.plan_charter(0, Face::PosX);
    let n0 = i64::from(body.ladder().cells_per_edge(0));
    let freq = body.radius_m() / lambda_m;
    let whole = freq.floor();
    let (f_int, f_frac) = (
        Gi::new(whole as i64),
        Gi::new(((freq - whole) * f64::from(1u32 << NOISE_BITS)).round() as i64),
    );
    let h = 2.0 * f64::from(vd_seed::ladder::cell_m(0));
    let (mut sum_h, mut sum_s, mut n_h, mut n_s) = (0.0f64, 0.0f64, 0usize, 0usize);
    for salt in 0..4u64 {
        let unit = Octave {
            seed: 0x5eed_0f0c_7a5e_0000 ^ salt,
            frequency_int: f_int,
            frequency_frac: f_frac,
            amplitude: Gi::new(1i64 << (7 + 8 + AMP_BITS - 8)),
            kind,
            fine: OCTAVE_FINE,
            pad: [Gi::ZERO; 2],
        };
        let mut table = [Octave::smooth(0, Gi::ZERO, Gi::ZERO, Gi::ZERO); OCTAVES_CAP];
        table[0] = unit;
        let term = |d: [Gi; 3]| {
            vd_bins::shore_rules::metres(
                relief_parts_from(&table, 0, 1, d, &charter.roughness, NOISE_ONE).fine,
            )
        };
        let mut state = 0x243f_6a88_85a3_08d3u64 ^ (salt << 32);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..sites {
            let face = (next() % 6) as i32;
            let span = (n0 - 4_096).max(1) as u64;
            let i = (next() % span) as i32 + 1_024;
            let j = (next() % span) as i32 + 1_024;
            let t0 = term(site_direction(&charter, face, i, j));
            let ti = term(site_direction(&charter, face, i + 2, j));
            let tj = term(site_direction(&charter, face, i, j + 2));
            sum_h += t0 * t0;
            sum_s += ((ti - t0) / h).powi(2) + ((tj - t0) / h).powi(2);
            n_h += 1;
            n_s += 2;
        }
    }
    let rms_h = (sum_h / n_h as f64).sqrt();
    let rms_s = (sum_s / n_s as f64).sqrt();
    vd_bins::shore_rules::FADE_PEAK_SLOPE * rms_h / (rms_s * lambda_m)
}
