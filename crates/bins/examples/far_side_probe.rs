//! ★ DOES A COARSE CELL CONTRADICT THE GROUND UNDER IT? (2026-09-22; the owner, from 41 000 km:
//! *"the globe shows squares of water on the land"*, and the same ground flips between water and
//! land as the rings sweep under a moving hull; ruling W15.)
//!
//! The defect's own number. For every rung whose CELL is wider than a fine macro node, this walks
//! EVERY cell of the globe and asks one question: does the side the cell is drawn on agree with the
//! wet fraction of the fine nodes under its whole footprint — which is the same thing as the wet
//! majority of its four children at the rung below, because a parent's count is the sum of its
//! children's?
//!
//! * `disagree OLD` counts the cells the ONE fine node nearest the cell's centre puts on the wrong
//!   side — the rule ruling W10 shipped, and the checkerboard's own number.
//! * `disagree NEW` counts them under the footprint's wet fraction (ruling W15). It is ZERO by
//!   construction, and this instrument is the test that could have failed.
//! * `children differ` counts, for information, how many of the four children stand on the other
//!   side from their parent. That is REFINEMENT, not contradiction: a coarse cell shows the side
//!   most of its ground stands on, and the rung below opens the bay inside it.
//!
//! `cargo run --release -p vd-bins --example far_side_probe`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_seed::bend::Face;
use vd_terrain::artifact::{PyramidField, coast_level, node_and_fraction, sample_side};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;

fn main() {
    let body = home_planet();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    let body = home_planet().with_sea_m(artifact.sea());
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let levels = artifact.pyramid.len() as u32;
    // THE FOLD'S OWN COST, once per body on each host: the client pays it on the frame the mask
    // lands, so it is a number and not a hope.
    let began = std::time::Instant::now();
    let counts = artifact.coast_counts();
    let fold_ms = began.elapsed().as_secs_f64() * 1.0e3;
    let top = body.ladder().rungs.min(18);
    // The rung at which a cell first covers more than one fine node: the first rung this question
    // can be asked at, computed from the node's own size and never typed.
    let first = (0..=top)
        .find(|&r| coast_level(lattice.cells_per_node, r) > 0)
        .expect("a rung whose cell is wider than a node");
    println!(
        "far_side_probe: node {} m ({} cells), levels of counts {}, folded in {fold_ms:.1} ms, rungs {first}..{top}, every cell of the globe",
        lattice.node_m().round(),
        lattice.cells_per_node,
        counts.levels(),
    );
    println!(
        "rung | cell m | nodes a side | cells | disagree OLD (the centre node) | disagree NEW (the footprint) | children differ | parent != sum of children"
    );
    for rung in first..=top {
        let k = coast_level(lattice.cells_per_node, rung).min(counts.levels());
        let level = PyramidField::level_for(&lattice, levels, rung);
        let field = PyramidField::of(&artifact, level, &counts).expect("a level");
        let old = PyramidField {
            counts: None,
            ..field.clone()
        };
        let n = body.ladder().cells_per_edge(rung) as i32;
        let child_n = body.ladder().cells_per_edge(rung - 1) as i32;
        let mut cells = 0u64;
        let mut disagree_old = 0u64;
        let mut disagree_new = 0u64;
        let mut children_differ = 0u64;
        let mut sum_wrong = 0u64;
        for face in 0..6u8 {
            let face = Face::from_index(face).expect("a face");
            for y in 0..n {
                for x in 0..n {
                    // THE FOOTPRINT'S OWN ANSWER: the wet fraction under the cell's whole block.
                    let Some(want) = counts.side(
                        k,
                        face,
                        block(&lattice, rung, x, k),
                        block(&lattice, rung, y, k),
                    ) else {
                        continue;
                    };
                    cells += 1;
                    if sample_side(&lattice, &old, face, rung, x, y) != Some(want) {
                        disagree_old += 1;
                    }
                    if sample_side(&lattice, &field, face, rung, x, y) != Some(want) {
                        disagree_new += 1;
                    }
                    // THE FOUR CHILDREN at the rung below: their counts must sum to this cell's,
                    // and their sides refine it.
                    if 2 * x + 1 < child_n && 2 * y + 1 < child_n {
                        let kc = coast_level(lattice.cells_per_node, rung - 1).min(counts.levels());
                        let mut sum = (0u32, 0u32);
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let (cx, cy) = (2 * x + dx, 2 * y + dy);
                            let child = counts
                                .count(
                                    kc,
                                    face,
                                    block(&lattice, rung - 1, cx, kc),
                                    block(&lattice, rung - 1, cy, kc),
                                )
                                .unwrap_or((0, 0));
                            sum.0 += child.0;
                            sum.1 += child.1;
                            if sample_side(&lattice, &field, face, rung - 1, cx, cy) != Some(want) {
                                children_differ += 1;
                            }
                        }
                        let parent = counts
                            .count(
                                k,
                                face,
                                block(&lattice, rung, x, k),
                                block(&lattice, rung, y, k),
                            )
                            .unwrap_or((0, 0));
                        if kc > 0 && sum != parent {
                            sum_wrong += 1;
                        }
                    }
                }
            }
        }
        println!(
            "{rung:>4} | {:>6} | {:>12} | {cells:>9} | {disagree_old:>30} | {disagree_new:>28} | {children_differ:>15} | {sum_wrong:>24}",
            vd_seed::ladder::cell_m(rung),
            1u32 << k,
        );
    }
}

/// The level-`k` coarse node a cell's own footprint sits in: the cell's nearest FINE node along one
/// axis, held inside the face, shifted `k` levels up — the very mapping `sample_side` reads.
fn block(lattice: &MacroLattice, rung: u8, i: i32, k: u32) -> u32 {
    let half = vd_recipe::Gi::new(1i64 << (vd_recipe::noise::NOISE_BITS - 1));
    let (n, t) = node_and_fraction(lattice, rung, i);
    let n = n + i64::from(t >= half);
    (n.clamp(0, i64::from(lattice.edge) - 1) >> k) as u32
}
