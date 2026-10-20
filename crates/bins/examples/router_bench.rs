//! ★ THE WASHBOARD BENCH — THE TWO CURES OF D8's GRID BIAS, MEASURED BEFORE R2 (ruling B10 item 2,
//! 2026-09-26; `docs/investigation/2026-09-24/terrain_believability_status.md` §8c and the plan's order).
//!
//! **WHY IT EXISTS.** The shoreline bench's panels showed parallel 45° ridges one node apart across
//! the high lake and the second pocket's floor, and chains of lakes on exact diagonals: D8's own bias
//! on smooth ground, carved by the solve's erosion into the 8 km field. The owner ruled that two cures
//! are measured first — a seeded spread of the drain direction (Rho8) and the two-neighbour split (D∞)
//! — and this bench measures them WITHOUT changing the world: `vd_bins::router_variants` rewrites the
//! drains of a routed state, and this example runs the solve's own pass loop around it.
//!
//! **TWO STAGES.**
//! 1. `--cone`: the home moon under Tarboton's cone at two poles (a face's middle, near a face's
//!    corner), each router's upslope-area error inward and outward (Tarboton's Table 2 quantity, in
//!    nodes²), Hyväluoma's rotation score at 15° and 45°, and the fourfold amplitude.
//! 2. `--planet d8`: the HOME PLANET solved by the bench's copy of `vd_terrain::solve::solve_full`
//!    (`vd_bins::solve_copy`, every law as the game runs it), and its readings.
//!    ★ THE CONTROL: it must give the artifact the pin records (`crates/bins/tests/home_artifact_pin.rs`)
//!    bit for bit (`--check-stock` also runs the real solve and compares every height and every lake
//!    level). ★ SINCE PHASE 2 (2026-10-04) the variants' own cut and deposit were the old stream-power
//!    law, so the copy no longer routes by them: Rho8, D∞, the fills and the fans return on the joint
//!    law when R2 measures D∞ (ruling B12; DEFERRED D-TERRAIN-10). Stage 1 reads the routers alone and
//!    stands.
//!
//! **WHAT IT PRINTS (stage 2)**, each a `CENSUS <GATE>.<variant> …` line: G-GRID's valleys (the census's
//! own rule — a land node of at least a hundred nodes of drainage area, its long axis over eight
//! receiver steps of the shipped tree) at four and twelve bins; the lakes' long axes (patches of four
//! nodes or more); ★ THE FACING — the ground's own slope direction over the land, its fourfold mode and
//! twelve-bin flatness, a reading no router makes; the lake share, the sea and the ocean share.
//! `--write-artifact <file>` writes the variant's artifact as a plain file (`vd_bins::artifact_file`),
//! which `--artifact-readings` and `equator_stands` read (the shoreline bench that painted it left
//! with the drawn water, ruling B22).
//!
//! ```text
//! cargo run --release -p vd-bins --example router_bench -- --cone
//! cargo run --release -p vd-bins --example router_bench -- --planet d8 --write-artifact d8.art
//! ```
//!
//! **Example.** Over the second pocket's floor, D8's furrows run north-east one node apart; the census
//! line FACING reads a strong fourfold mode at phase 45°. After the cure the same floor's slopes face
//! every way, the mode falls toward the isotropic reading, and a picture of that floor shows hollows
//! where the furrows stood.

use std::time::Instant;

use vd_bins::grid_bias::{
    AZIMUTHS, AxisHistogram, ConeError, cone_error, cone_z, patch_axis, polar_map, trunk_axis, unit,
};
use vd_bins::router_variants::{dinf, dinf_accumulate, facing, paola_braided, rho8, units};
use vd_bins::solve_copy::{Laws, artifact_of, solve_copy};
use vd_terrain::home::{home_moon, home_planet, home_solve_words};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{DRY, MacroSolve, Schedule};

/// The artifact the pin records (`crates/bins/tests/home_artifact_pin.rs`).
const HOME_ARTIFACT_DIGEST: [u64; 2] = [0xddad_33b9_72ca_7ebb, 0xfcb3_cec0_6fe7_11ba];

/// The cone's slope (grid_bias's own).
const CONE_SLOPE: f64 = 0.05;

/// The routers stage 1 compares (the washboard bench's own; ruling B10 item 2).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    D8,
    Rho8,
    Dinf,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::D8 => "d8",
            Variant::Rho8 => "rho8",
            Variant::Dinf => "dinf",
        }
    }
}

// ── STAGE 1: THE CONE ────────────────────────────────────────────────────────────────────────

fn cone_stage() {
    let body = home_moon();
    let lattice = MacroLattice::of(&body).expect("the home moon has a lattice");
    let radius_m = body.radius_m();
    let u = units(&lattice);
    let edge = lattice.edge as i32;
    let face = vd_seed::bend::Face::PosX;
    println!(
        "== STAGE 1: TARBOTON'S CONE on the home moon ({} nodes of {:.0} m) ==",
        lattice.node_count(),
        lattice.node_m()
    );
    for (pole, node) in [
        ("face-middle", lattice.index(face, edge / 2, edge / 2)),
        ("near-corner", lattice.index(face, edge / 6, edge / 6)),
    ] {
        let apex = unit(&lattice, node);
        let reference = unit(&lattice, lattice.neighbours(node)[4]);
        for v in [Variant::D8, Variant::Rho8, Variant::Dinf] {
            let mut state = MacroSolve::new(&body).expect("the moon has a solve state");
            state.z = cone_z(&lattice, apex, radius_m, CONE_SLOPE);
            let _ = state.route();
            match v {
                Variant::D8 => {
                    state.accumulate();
                }
                Variant::Rho8 => {
                    let seed = state.seed;
                    let _ = rho8(&mut state, &u, radius_m, seed);
                    state.accumulate();
                }
                Variant::Dinf => {
                    let d = dinf(&state, &u, radius_m);
                    dinf_accumulate(&mut state, &d);
                }
            }
            let band = |lo: f64, hi: f64| {
                cone_error(
                    &lattice,
                    &state.discharge,
                    apex,
                    radius_m,
                    (lo.to_radians(), hi.to_radians()),
                )
            };
            let inward = band(15.0, 45.0);
            let outward = band(135.0, 165.0);
            let map = polar_map(
                &lattice,
                &state.discharge,
                apex,
                reference,
                (20f64.to_radians(), 70f64.to_radians()),
            );
            let rot = |deg: usize| map.rotation_score(deg * AZIMUTHS / 360);
            let say = |e: ConeError| format!("mse{:.1}/rel{:.2}%", e.mse, e.relative_pct);
            println!(
                "CENSUS CONE.{pole}.{} inward:{} outward:{} rotation15:{:.3} rotation45:{:.3} fourfold:{:.2}% no-band[Tarboton-1997;Hyvaluoma-2017]",
                v.name(),
                say(inward),
                say(outward),
                rot(15),
                rot(45),
                map.fourfold_pct()
            );
        }
    }
}

// ── STAGE 2: THE PLANET, THE SOLVE'S OWN LOOP AROUND THE VARIANT ─────────────────────────────

/// ★ THE READINGS of a solved state: G-GRID's valleys and lakes (the census's own rule), the facing,
/// the lake share, the sea.
fn readings(tag: &str, state: &MacroSolve) {
    let lattice = &state.lattice;
    let n = state.node_count();
    // The drainage area over the shipped tree (one node's area a node), donors first.
    let mut drain: Vec<u64> = state.area.clone();
    for &node in &state.order {
        let r = state.receiver[node as usize];
        if r != NO_NODE {
            drain[r as usize] += drain[node as usize];
        }
    }
    let mut four = AxisHistogram::new(4);
    let mut twelve = AxisHistogram::new(12);
    for node in 0..n as u32 {
        let i = node as usize;
        if state.ocean[i] || drain[i] < state.area[i].saturating_mul(100) {
            continue;
        }
        if let Some(axis) = trunk_axis(lattice, &state.receiver, node, 8) {
            four.add(axis.degrees);
            twelve.add(axis.degrees);
        }
    }
    println!(
        "CENSUS G-GRID-VALLEYS.{tag} {:.3}four/{:.3}twelve/{}trunks ~1.006[W13.3] {}",
        four.flatness(),
        twelve.flatness(),
        four.total(),
        four.line()
    );
    // The lakes: patches of one level, four nodes or more.
    let lake = |i: usize| state.water_z[i] != DRY && !state.ocean[i];
    let mut seen = vec![false; n];
    let (mut l4, mut l12) = (AxisHistogram::new(4), AxisHistogram::new(12));
    let (mut lake_area, mut land_area, mut patches, mut largest) = (0u64, 0u64, 0usize, 0usize);
    for i in 0..n {
        if !state.ocean[i] {
            land_area += state.area[i];
        }
        if lake(i) {
            lake_area += state.area[i];
        }
        if seen[i] || !lake(i) {
            continue;
        }
        let level = state.water_z[i];
        let mut stack = vec![i as u32];
        let mut nodes = Vec::new();
        seen[i] = true;
        while let Some(v) = stack.pop() {
            nodes.push(v);
            for m in lattice.neighbours(v) {
                if m != NO_NODE
                    && !seen[m as usize]
                    && lake(m as usize)
                    && state.water_z[m as usize] == level
                {
                    seen[m as usize] = true;
                    stack.push(m);
                }
            }
        }
        patches += 1;
        largest = largest.max(nodes.len());
        if nodes.len() >= 4
            && let Some(axis) = patch_axis(lattice, &nodes)
        {
            l4.add(axis.degrees);
            l12.add(axis.degrees);
        }
    }
    println!(
        "CENSUS G-GRID-LAKES.{tag} {:.3}four/{:.3}twelve/{}patches-of-4+ no-band[a-reading] {}",
        l4.flatness(),
        l12.flatness(),
        l4.total(),
        l4.line()
    );
    let land = |i: usize| !state.ocean[i];
    let f = facing(lattice, &state.z, &land);
    println!(
        "CENSUS FACING.{tag} fourfold{:.2}%/phase{:.1}deg/twelve-flatness{:.3}/{}nodes no-band[the-ground's-own-slope-direction]",
        f.fourfold_pct,
        f.phase_deg,
        f.flatness(),
        f.nodes
    );
    println!(
        "CENSUS LAKES.{tag} share{:.3}%/{}patches/largest{}nodes; sea {} steps; ocean share {:.3} no-band[ruling-B9:4.828%-at-generator-21]",
        100.0 * lake_area as f64 / land_area.max(1) as f64,
        patches,
        largest,
        state.sea_z,
        state.ocean_share()
    );
}

fn planet_stage(write: Option<&str>, check_stock: bool) {
    let body = home_planet();
    let words = home_solve_words();
    let schedule = Schedule::standard(words.age_yr);
    println!("== STAGE 2: THE HOME PLANET, d8 (the copy, every law as the game runs it) ==");
    println!(
        "  Paola, Heller & Angevine 1992's braided inlet slope over the load-to-water ratio: {:.3}",
        paola_braided()
    );
    let t = Instant::now();
    let solved = solve_copy(&body, &words, schedule, &Laws::default(), false)
        .expect("the home planet solves");
    println!("the copy solved in {:.1} s", t.elapsed().as_secs_f64());
    let artifact = artifact_of(&words, &solved);
    let digest = artifact.digest();
    println!(
        "ARTIFACT d8 digest [{:#018x}, {:#018x}]",
        digest[0], digest[1]
    );
    let ok = digest == HOME_ARTIFACT_DIGEST;
    println!(
        "CENSUS CHECK-COPY-IS-THE-SOLVE.d8 {} =the-pin[home_artifact_pin,generator-{}] {}",
        if ok { "equal" } else { "DIFFERS" },
        vd_terrain::GENERATOR_VERSION,
        if ok { "GREEN" } else { "RED" }
    );
    if check_stock {
        let t = Instant::now();
        let (stock, _, _) =
            vd_terrain::solve::solve_full(&body, &words, schedule).expect("the planet solves");
        let dz = stock
            .z
            .iter()
            .zip(&solved.state.z)
            .filter(|(a, b)| a != b)
            .count();
        let dw = stock
            .water_z
            .iter()
            .zip(&solved.state.water_z)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "CENSUS CHECK-STOCK.d8 {}heights/{}levels-differ-from-solve_full ({:.1} s) {}",
            dz,
            dw,
            t.elapsed().as_secs_f64(),
            if dz == 0 && dw == 0 { "GREEN" } else { "RED" }
        );
    }
    readings("d8", &solved.state);
    if let Some(path) = write {
        match vd_bins::artifact_file::write(path, &artifact) {
            Ok(()) => println!("wrote the d8 artifact to {path}"),
            Err(e) => println!("could not write {path}: {e}"),
        }
    }
}

/// ★ THE READINGS OF AN ARTIFACT FILE (the plain file of `vd_bins::artifact_file`): the facing of
/// the land and — ★ THE WASHBOARD'S OWN READING — the facing of the land's FINE texture, each node's
/// height less the mean of its eight neighbours, so the continents' regional slopes (which the cube's
/// faces orient) leave the reading and the one-node furrows stay; and the lake share. The heights are
/// the rows' whole metres.
fn artifact_readings(path: &str, tag: &str) {
    use vd_terrain::artifact::{DRY_M, FACIES_SHIFT};
    use vd_terrain::solve::FACIES_SEA;
    let Some(a) = vd_bins::artifact_file::read(path) else {
        println!("cannot read the artifact {path}");
        return;
    };
    let body = home_planet();
    let lattice = body.macro_lattice().expect("a lattice");
    let n = lattice.node_count();
    let sea = |i: usize| (a.rows[i].receiver_facies >> FACIES_SHIFT) & FACIES_SEA != 0;
    let z: Vec<i32> = a.rows.iter().map(|r| i32::from(r.z_m)).collect();
    let land = |i: usize| !sea(i);
    let f = facing(&lattice, &z, &land);
    // The fine texture, in decimetres: the height less the mean of the eight neighbours.
    let mut fine = vec![0i32; n];
    let mut has = vec![false; n];
    for i in 0..n {
        let nb = lattice.neighbours(i as u32);
        if nb.contains(&NO_NODE) {
            continue;
        }
        let mean = nb.iter().map(|&m| f64::from(z[m as usize])).sum::<f64>() / 8.0;
        fine[i] = ((f64::from(z[i]) - mean) * 10.0).round() as i32;
        has[i] = true;
    }
    let fine_keep = |i: usize| {
        land(i)
            && has[i]
            && lattice
                .neighbours(i as u32)
                .iter()
                .all(|&m| m != NO_NODE && has[m as usize])
    };
    let g = facing(&lattice, &fine, &fine_keep);
    let (mut lake, mut dry) = (0u64, 0u64);
    for (i, r) in a.rows.iter().enumerate() {
        if sea(i) {
            continue;
        }
        let area = lattice.area_m2(i as u32);
        dry += area;
        if r.water_m != DRY_M {
            lake += area;
        }
    }
    println!(
        "CENSUS FACING-LAND.{tag} fourfold{:.2}%/phase{:.1}deg/twelve-flatness{:.3}/{}nodes no-band[the-ground's-own-slope-direction]",
        f.fourfold_pct,
        f.phase_deg,
        f.flatness(),
        f.nodes
    );
    println!(
        "CENSUS FACING-FINE.{tag} fourfold{:.2}%/phase{:.1}deg/twelve-flatness{:.3}/{}nodes no-band[the-washboard:one-node-furrows;phase-45=diagonals,0=rows]",
        g.fourfold_pct,
        g.phase_deg,
        g.flatness(),
        g.nodes
    );
    println!(
        "CENSUS LAKE-SHARE-ROWS.{tag} {:.3}% no-band[the-rows'-own-water]",
        100.0 * lake as f64 / dry.max(1) as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cone = false;
    let mut planet = false;
    let mut write: Option<String> = None;
    let mut check_stock = false;
    let mut readings_done = false;
    let mut k = 0;
    while k < args.len() {
        match args[k].as_str() {
            "--cone" => cone = true,
            "--planet" => {
                k += 1;
                match args.get(k).map(String::as_str) {
                    Some("d8") => planet = true,
                    other => panic!(
                        "--planet d8 (the variants left with the old erosion, DEFERRED D-TERRAIN-10), not {other:?}"
                    ),
                }
            }
            "--write-artifact" => {
                k += 1;
                write = args.get(k).cloned();
            }
            "--check-stock" => check_stock = true,
            "--artifact-readings" => {
                let path = args
                    .get(k + 1)
                    .cloned()
                    .expect("--artifact-readings FILE TAG");
                let tag = args
                    .get(k + 2)
                    .cloned()
                    .expect("--artifact-readings FILE TAG");
                artifact_readings(&path, &tag);
                readings_done = true;
                k += 2;
            }
            other => panic!("unknown argument {other}"),
        }
        k += 1;
    }
    if cone {
        cone_stage();
    }
    if planet {
        planet_stage(write.as_deref(), check_stock);
    }
    if !cone && !planet && !readings_done {
        println!(
            "usage: router_bench [--cone] [--planet d8] [--write-artifact FILE] [--check-stock]"
        );
    }
}
