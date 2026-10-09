//! ★ THE RIVERS' GRID LEAN, FROM AN ARTIFACT FILE (ruling B58: "remove the straight or 45° angled rivers (or
//! staircased)"). The shipped receivers ARE the water's own tree, so the macro rivers' geometry is in the rows alone.
//! For every land node whose discharge is in the top share of the land's (the rivers), the walk 8 steps down the shipped
//! receivers on one face gives an AXIS; the reading states:
//!
//! 1. THE ON-LINE SHARE: the axes within 5° of a lattice line (0°, 45°, 90°, 135°). With no lean the share is
//!    10/45 = 22.2 %; a router that walks the stencil's own directions reads near 100 %.
//! 2. THE STRAIGHT RUNS: along every river walk, the steps that keep the previous step's direction, as a share of all
//!    steps, and the longest run. A river that follows the ground's own bearing turns; one that follows the lattice
//!    runs straight.
//!
//! `cargo run --release -p vd-bins --example river_lean -- <artifact_file> [<artifact_file> …]`
//!
//! **Example.** A 60 km river that runs due south over eight nodes has its axis at 90°: it counts as on-line, and its
//! seven repeated steps count as straight. The same river after the cure bends between the row and the diagonal: its
//! axis reads 74°, and only two of its steps repeat.

use vd_seed::bend::Face;
use vd_terrain::artifact::{RECEIVER_NONE, RECEIVER_SLOT_MASK, node_at};
use vd_terrain::home::home_planet;
use vd_terrain::macro_lattice::{MacroLattice, STENCIL};

/// The steps a walk takes for its axis.
const STEPS: usize = 8;
/// The rivers: the land nodes whose discharge class is in this top share of the land's.
const TOP_SHARE: f64 = 0.05;

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "<artifact_file> [<artifact_file> …]");
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let edge = i64::from(lattice.edge);
    println!(
        "river_lean: the home planet's lattice, {} nodes a face edge; a river is a land node in the top {:.0} % of the land's discharge; axes over {STEPS} steps",
        lattice.edge,
        TOP_SHARE * 100.0
    );
    println!(
        "\nthe file | rivers | axes on a lattice line (±5°; none 22.2 %) | the axes' histogram 0/22/45/67/90/112/135/157° | straight steps | longest straight run"
    );
    for f in &files {
        let Some(artifact) = vd_bins::artifact_file::read(f) else {
            println!("{f}: not read");
            continue;
        };
        let sea = artifact.sea_m;
        // The land's discharge classes, for the top share.
        let mut classes: Vec<u8> = artifact
            .rows
            .iter()
            .filter(|r| r.z_m > sea)
            .map(|r| r.discharge_log)
            .collect();
        classes.sort_unstable();
        let cut = classes[((classes.len() - 1) as f64 * (1.0 - TOP_SHARE)) as usize];
        let (mut rivers, mut on_line, mut straight, mut steps_all, mut longest) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut hist = [0usize; 8];
        for face in Face::ALL {
            for j in 0..edge {
                for i in 0..edge {
                    let node = node_at(&lattice, face, i, j);
                    let row = artifact.rows[node as usize];
                    if row.z_m <= sea || row.discharge_log < cut {
                        continue;
                    }
                    // The walk on this face.
                    let (mut ci, mut cj) = (i, j);
                    let (mut prev, mut run, mut ok) = (u8::MAX, 0usize, true);
                    let mut taken = 0usize;
                    for _ in 0..STEPS {
                        let r = artifact.rows[node_at(&lattice, face, ci, cj) as usize];
                        if r.receiver_facies & RECEIVER_NONE != 0 {
                            ok = false;
                            break;
                        }
                        let slot = r.receiver_facies & RECEIVER_SLOT_MASK;
                        let (di, dj) = STENCIL[slot as usize];
                        let (di, dj) = (i64::from(di), i64::from(dj));
                        let (ni, nj) = (ci + di, cj + dj);
                        if ni < 0 || nj < 0 || ni >= edge || nj >= edge {
                            ok = false;
                            break;
                        }
                        steps_all += 1;
                        if slot == prev {
                            straight += 1;
                            run += 1;
                            longest = longest.max(run);
                        } else {
                            run = 1;
                        }
                        prev = slot;
                        ci = ni;
                        cj = nj;
                        taken += 1;
                    }
                    if !ok || taken < STEPS {
                        continue;
                    }
                    rivers += 1;
                    let ang = ((cj - j) as f64)
                        .atan2((ci - i) as f64)
                        .to_degrees()
                        .rem_euclid(180.0);
                    let off = ang.rem_euclid(45.0);
                    if !(5.0..=40.0).contains(&off) {
                        on_line += 1;
                    }
                    hist[((ang + 11.25) / 22.5) as usize % 8] += 1;
                }
            }
        }
        println!(
            "{f}\n  {rivers} rivers | {:.1} % on a lattice line | {:?} | {:.1} % straight | {longest} steps",
            100.0 * on_line as f64 / rivers.max(1) as f64,
            hist,
            100.0 * straight as f64 / steps_all.max(1) as f64
        );
    }
}
