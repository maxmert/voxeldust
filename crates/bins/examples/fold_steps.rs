//! ★ THE FOLD STEPS, PLANET-WIDE (ruling B55 item 2, 2026-10-05; the owner: "Agree, go" — the level
//! handover's pops on steep ground measured over the whole planet, headless).
//!
//! The near rungs read the artifact's ROWS and the far rungs read a PYRAMID LEVEL, whose node is the
//! MEAN of four of the level under it. At a FOLD — a rung pair whose two rungs read different levels
//! — the ground a chunk draws changes by the octaves the coarser rung drops AND by the fold's own
//! spread. The ladder places the swap where its STATED step (`handover_step_m`: the octave table's
//! bound plus `field_step_m`, the slope law over the fold) is one pixel. The slope law reads the
//! noise's own slope, and phase 2's solved ranges are steeper (the rung-swap run on one massif: a
//! 16 km node stands up to 4 253 m from one of its children, against the law's 722 m). So this tool
//! reads, at EVERY fold of the home planet, the surface at both rungs at the same directions — four
//! places across each node of the coarser level, through the shipped reader
//! `vd_terrain::height::height_field_m` — and states how often and by how much the real step stands
//! over the stated one, and what that is in PIXELS at the distance the swap happens.
//!
//! `cargo run --release -p vd-bins --example fold_steps -- [artifact_file]`: the home planet's
//! artifact from the file when it is there, else solved here and written to it.
//!
//! **Example.** If the 9 → 10 fold reads a real step of 2 000 m where 1 620 m is stated, the swap
//! that should move the ground by one pixel moves it by 1.2 pixels there, and the table says on how
//! much of the planet that happens.

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_client::ladder_view::{handover_step_m, pixel_rad, switch_m};
use vd_seed::bend::{Face, direction};
use vd_terrain::artifact::{PyramidField, ZField};
use vd_terrain::height::height_field_m;
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;

/// The places across one node of the coarser level the surface is read at, on each axis.
const ACROSS: [f64; 4] = [0.0, 0.25, 0.5, 0.75];

fn main() {
    let path = std::env::args().nth(1);
    let artifact = match path.as_deref().and_then(vd_bins::artifact_file::read) {
        Some(a) => {
            println!(
                "fold_steps: the artifact read from {}",
                path.as_deref().unwrap_or("")
            );
            a
        }
        None => {
            let started = std::time::Instant::now();
            let a = run_solve(&SolveJob {
                body: home_planet(),
                words: home_solve_words(),
            })
            .expect("the home planet solves");
            println!(
                "fold_steps: solved in {:.0} s",
                started.elapsed().as_secs_f64()
            );
            if let Some(p) = path.as_deref() {
                match vd_bins::artifact_file::write(p, &a) {
                    Ok(()) => println!("fold_steps: the artifact written to {p}"),
                    Err(e) => println!("fold_steps: could not write {p}: {e}"),
                }
            }
            a
        }
    };
    let wet = home_planet().with_sea_m(artifact.sea());
    let lattice = MacroLattice::of(&wet).expect("a lattice");
    let drawn = artifact.drawn_range_m(&lattice);
    // The body every host draws (ruling B55): the band the head's drawn range sizes.
    let body = wet.with_band_m(drawn.0, drawn.1).unwrap_or(wet);
    let levels = artifact.pyramid.len() as u32;
    let counts = artifact.coast_counts();
    let top = body.ladder().rungs - 1;
    println!(
        "the home planet: {} rungs, {levels} pyramid levels, the drawn range {} .. {} m; the band's top {} m over the radius",
        body.ladder().rungs,
        drawn.0,
        drawn.1,
        body.ladder().above_m()
    );
    let field_of = |level: u32| -> Box<dyn ZField> {
        if level == 0 {
            Box::new(artifact.clone())
        } else {
            Box::new(PyramidField::of(&artifact, level, &counts).expect("a level"))
        }
    };
    println!(
        "\nL -> L+1 | levels | samples | stated step m | the swap m | median m | p99 m | p99.9 m | max m | over the stated step | max ratio | p99.9 px | max px"
    );
    let mut worst_all: Vec<(f64, u8, [f64; 3], f64, f64)> = Vec::new();
    for rung in 0..top {
        let here = PyramidField::level_for(&lattice, levels, rung);
        let next = PyramidField::level_for(&lattice, levels, rung + 1);
        if here == next {
            continue;
        }
        let (fine, coarse) = (field_of(here), field_of(next));
        let at = lattice.coarser(next).expect("the coarser level");
        let n = f64::from(at.edge);
        let field = vd_terrain::artifact::field_step_m(&body, &lattice, levels, rung);
        let step = handover_step_m(&body, rung, field);
        let s = switch_m(rung).max(vd_client::ladder_view::switch_floor_m(step));
        let mut diffs: Vec<f64> = Vec::new();
        let mut worst: Vec<(f64, u8, [f64; 3], f64, f64)> = Vec::new();
        for face in Face::ALL {
            for j in 0..at.edge {
                for i in 0..at.edge {
                    for fa in ACROSS {
                        for fb in ACROSS {
                            let a = 2.0 * (f64::from(i) + fa) / n - 1.0;
                            let b = 2.0 * (f64::from(j) + fb) / n - 1.0;
                            let d = direction(face, a, b);
                            let (Some(h), Some(c)) = (
                                height_field_m(&body, fine.as_ref(), d, rung),
                                height_field_m(&body, coarse.as_ref(), d, rung + 1),
                            ) else {
                                continue;
                            };
                            let diff = (h - c).abs();
                            diffs.push(diff);
                            if diff > step {
                                worst.push((diff / step, rung, d, h, c));
                            }
                        }
                    }
                }
            }
        }
        diffs.sort_by(f64::total_cmp);
        let q = |p: f64| diffs[((diffs.len() as f64 * p) as usize).min(diffs.len() - 1)];
        let over = diffs.iter().filter(|d| **d > step).count();
        let px = |m: f64| (m / s) / pixel_rad();
        let max = q(1.0);
        println!(
            "{rung:>3} -> {:>3} | {here} -> {next} | {:>8} | {step:>13.1} | {s:>10.0} | {:>8.1} | {:>5.1} | {:>7.1} | {max:>6.1} | {over:>8} ({:.4} %) | {:>9.2} | {:>8.2} | {:>6.2}",
            rung + 1,
            diffs.len(),
            q(0.5),
            q(0.99),
            q(0.999),
            100.0 * over as f64 / diffs.len().max(1) as f64,
            max / step,
            px(q(0.999)),
            px(max)
        );
        worst.sort_by(|x, y| y.0.total_cmp(&x.0));
        worst_all.extend(worst.into_iter().take(5));
    }
    println!(
        "\nTHE WORST PLACES (the ratio of the real step to the stated one; the direction; the two surfaces over the radius, m):"
    );
    worst_all.sort_by(|x, y| y.0.total_cmp(&x.0));
    let r = body.radius_m();
    for (ratio, rung, d, h, c) in worst_all.iter().take(15) {
        println!(
            "  {ratio:>6.2} at {rung} -> {}: ({:.6}, {:.6}, {:.6}) fine {:.0} m, coarse {:.0} m",
            rung + 1,
            d[0],
            d[1],
            d[2],
            h - r,
            c - r
        );
    }
}
