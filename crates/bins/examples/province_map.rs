//! ★ THE ROCK MAP OVER ONE FACE (slice 8d step 2; `slice_8d_design.md` §3.5): the home planet is
//! solved once, and the province byte of every node is counted and drawn.
//!
//! The picture the owner flies cannot show this yet — the paint table that gives each substance a
//! colour is slice 8e — so the rock map is read as TEXT: a histogram of the provinces over the whole
//! planet and over one face, and a coarse map of one face, one character per sixteen nodes (the
//! MOST COMMON province of the block, so a belt narrower than the block is not lost to a corner
//! sample).
//!
//! `cargo run --release -p vd-bins --example province_map -- [face] [block]`
//! (`face` is the bend's index: 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z; `block` the nodes a character
//! covers, 16 by default).
//!
//! **Example.** The owner asks which rock a miner hits on the day-side belt. The map says the belt
//! stands in the folded-belt province, so the mine cuts slate and quartzite; two hundred kilometres
//! west the shelf gives limestone.

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::strata::Province;

/// The character each province is drawn with.
fn mark(p: Province) -> char {
    match p {
        Province::CrystallineBasement => 'B',
        Province::FoldedBelt => 'M',
        Province::FlatShelf => 's',
        Province::RiftBasalt => 'r',
        Province::DeepSediment => '.',
    }
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("an integer"))
        .collect();
    let face = args.first().copied().unwrap_or(4).min(5);
    let block = args.get(1).copied().unwrap_or(16).max(1);
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("a macro lattice");
    let started = std::time::Instant::now();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    println!(
        "province_map: solved in {:.1} s, {} nodes, {} bytes, artifact version {}",
        started.elapsed().as_secs_f64(),
        artifact.node_count(),
        artifact.bytes(),
        artifact.version
    );

    // The histogram over the whole planet, and over the one face the map draws.
    let edge = lattice.edge as usize;
    let per_face = edge * edge;
    let mut whole = [0usize; 256];
    let mut one = [0usize; 256];
    for (n, row) in artifact.rows.iter().enumerate() {
        whole[usize::from(row.province)] += 1;
        if n / per_face == face {
            one[usize::from(row.province)] += 1;
        }
    }
    let histogram = |title: &str, tally: &[usize; 256], total: usize| {
        println!("\n{title}");
        for (code, &count) in tally.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let name = Province::from_code(code as u8);
            let share = count as f64 * 100.0 / total as f64;
            println!("  {code} {name:?}: {count} nodes, {share:.2} %");
        }
    };
    histogram(
        "the provinces, over the whole planet:",
        &whole,
        artifact.node_count(),
    );
    histogram(&format!("the same, over face {face}:"), &one, per_face);

    // The map: one character per block of nodes, the block's most common province.
    println!(
        "\nface {face}, one character per {block} nodes (B basement, M folded belt, s shelf, \
         r rift basalt, . deep sediment):"
    );
    let mut j = 0;
    while j < edge {
        let mut line = String::new();
        let mut i = 0;
        while i < edge {
            let mut tally = [0usize; 256];
            let mut dj = 0;
            while (dj < block) && (j + dj < edge) {
                let mut di = 0;
                while (di < block) && (i + di < edge) {
                    let node = face * per_face + (j + dj) * edge + (i + di);
                    tally[usize::from(artifact.rows[node].province)] += 1;
                    di += 1;
                }
                dj += 1;
            }
            let mut best = 0usize;
            for code in 1..256usize {
                if tally[code] > tally[best] {
                    best = code;
                }
            }
            line.push(Province::from_code(best as u8).map_or('?', mark));
            i += block;
        }
        println!("{line}");
        j += block;
    }
}
