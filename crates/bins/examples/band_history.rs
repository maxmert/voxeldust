//! ★ WHERE A GRID MARK FORMS IN THE SOLVE'S HISTORY (ruling B58 item 2: "a cause the routing does not remove … is
//! measured and brought back with its numbers"). Two readings of one tile of the home planet:
//!
//! 1. THE BELT'S MAP at the solve's start (the belts' law over the initial land, no solve): each node's belt kind, its
//!    side of the wedge, whether it stands on the plateau at the end, its window's start and its rise by the end.
//! 2. `--history DIR`: the bench copy of the solve ([`vd_bins::solve_copy::solve_copy_watched`], every law as the game
//!    runs it) with a watch that writes the tile's heights (`i32`, sixteenths) and the receivers' slots (`u8`, the
//!    stencil's order, 255 for none) at the end of every `--every` passes from pass `--from` on and the last, one file
//!    pair a pass, and
//!    `index.txt` (the pass, its end in years from the solve's start). The copy's final artifact digest is printed
//!    beside a given artifact file's (`--against FILE`), so the history is proven to be that file's own solve.
//!
//! `cargo run --release -p vd-bins --example band_history -- [--tile F,I0,J0,W] [--history DIR] [--every K]
//! [--from P] [--passes P,Q,…] [--stages] [--against FILE]`
//!
//! **Example.** The belt tile of the home planet (face 3, nodes 1088.. × 960..) holds an east-west valley 330 km long
//! whose floor falls an even 18 m a node; the history shows the pass where its line first appears — on a plateau that
//! has just risen level, or on a slope — and so which rule drew it.

use std::io::Write;

use vd_seed::bend::Face;
use vd_terrain::artifact::node_at;
use vd_terrain::belts::{BELT_ANDEAN, BELT_COLLISION, SIDE_ANDEAN, SIDE_PRO, SIDE_RETRO};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{MacroSolve, Schedule, Z_STEPS_PER_M};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|k| args.get(k + 1))
            .cloned()
    };
    let tile: Vec<i64> = opt("--tile")
        .unwrap_or_else(|| "3,1088,960,64".to_string())
        .split(',')
        .map(|x| x.parse().expect("--tile F,I0,J0,W"))
        .collect();
    let (face, i0, j0, w) = (Face::ALL[tile[0] as usize], tile[1], tile[2], tile[3]);
    let every: usize = opt("--every").map_or(10, |x| x.parse().expect("--every K"));
    let from: usize = opt("--from").map_or(0, |x| x.parse().expect("--from P"));
    // `--passes P,Q,…`: exactly these passes (and the last), in place of `--every` and `--from`.
    let listed: Option<Vec<usize>> = opt("--passes").map(|x| {
        x.split(',')
            .map(|p| p.parse().expect("--passes P,Q,…"))
            .collect()
    });
    let body = home_planet();
    let words = home_solve_words();
    let lattice = MacroLattice::of(&body).expect("a lattice");
    let nodes: Vec<u32> = (0..w)
        .flat_map(|j| (0..w).map(move |i| (i, j)))
        .map(|(i, j)| node_at(&lattice, face, i0 + i, j0 + j))
        .collect();

    // 1. THE BELT'S MAP.
    let land = vd_terrain::land::initial_land(&body, &lattice, &words.land());
    let belts = vd_terrain::belts::belts(&body, &lattice, &words, &land, words.age_yr);
    let mut side = vec![u8::MAX; lattice.node_count()];
    for wn in &belts.wedge {
        side[wn.node as usize] = wn.side;
    }
    println!(
        "THE BELT'S MAP: face {}, nodes [{i0}..{}) × [{j0}..{}), every 4th node; a cell is the kind and side \
         (P pro, R retro, A Andean, . none), '*' on the plateau at the end, the rise by the end in hundreds of metres, \
         the window's start in Myr before the present",
        tile[0],
        i0 + w,
        j0 + w
    );
    for j in (0..w).step_by(4) {
        let mut line = String::new();
        for i in (0..w).step_by(4) {
            let k = nodes[(j * w + i) as usize] as usize;
            let s = match (belts.kind[k], side[k]) {
                (BELT_COLLISION, SIDE_PRO) => 'P',
                (BELT_COLLISION, SIDE_RETRO) => 'R',
                (BELT_ANDEAN, SIDE_ANDEAN) => 'A',
                _ => '.',
            };
            let star = if belts.plateau[k] { '*' } else { ' ' };
            let rise = belts.total[k] / i64::from(Z_STEPS_PER_M) / 100;
            let ago = (words.age_yr - belts.start_yr[k]) / 1_000_000;
            line.push_str(&format!("{s}{star}{rise:>2}/{ago:<4}"));
        }
        println!("  {line}");
    }
    println!(
        "  the plates under the tile (every 8th node): {}",
        (0..w)
            .step_by(8)
            .map(|j| (0..w)
                .step_by(8)
                .map(|i| format!("{:>3}", land.plate[nodes[(j * w + i) as usize] as usize]))
                .collect::<String>())
            .collect::<Vec<_>>()
            .join(" |")
    );

    // 2. THE HISTORY.
    let Some(dir) = opt("--history") else {
        return;
    };
    std::fs::create_dir_all(&dir).expect("the history's folder");
    let schedule = Schedule::standard(words.age_yr);
    let passes = schedule.times().len() - 1;
    let mut index = String::new();
    // `--stages`: every stage of a wanted pass (the tile after each process), in place of the pass's end alone.
    let stages = args.iter().any(|a| a == "--stages");
    // A stage that runs more than once a pass (the sweeps and their rebounds) is numbered in its order.
    let mut seen: (usize, std::collections::BTreeMap<String, u32>) =
        (usize::MAX, Default::default());
    let mut look = |pass: usize, t1: u64, stage: &str, state: &MacroSolve| {
        if !stages && stage != "peak" {
            return;
        }
        if seen.0 != pass {
            seen = (pass, Default::default());
        }
        let count = seen.1.entry(stage.to_string()).or_insert(0);
        *count += 1;
        let stage = format!("{stage}{count}");
        let wanted = match &listed {
            Some(list) => list.contains(&pass),
            None => pass >= from && pass.is_multiple_of(every),
        };
        if !wanted && pass + 1 != passes {
            return;
        }
        let mut z = Vec::with_capacity(nodes.len() * 4);
        let mut slot = Vec::with_capacity(nodes.len());
        for &v in &nodes {
            z.extend_from_slice(&state.z[v as usize].to_le_bytes());
            let r = state.receiver[v as usize];
            let s = lattice
                .neighbours(v)
                .iter()
                .position(|&m| (m == r) & (r != NO_NODE))
                .map_or(255u8, |s| s as u8);
            slot.push(s);
        }
        let name = if stages {
            format!("pass{pass:04}.{stage}")
        } else {
            format!("pass{pass:04}")
        };
        for (ext, bytes) in [("z.i32", &z), ("slot.u8", &slot)] {
            std::fs::File::create(format!("{dir}/{name}.{ext}"))
                .and_then(|mut f| f.write_all(bytes))
                .expect("a snapshot");
        }
        index.push_str(&format!("{pass} {t1} {} {stage}\n", state.sea_z));
    };
    let started = std::time::Instant::now();
    let solved = vd_bins::solve_copy::solve_copy_watched(
        &body,
        &words,
        schedule,
        &vd_bins::solve_copy::Laws::default(),
        false,
        Some(&mut look),
    )
    .expect("the home planet solves");
    std::fs::write(format!("{dir}/index.txt"), &index).expect("the index");
    let digest = vd_bins::solve_copy::artifact_of(&body, &words, &solved).digest();
    println!(
        "THE HISTORY: {passes} passes in {:.0} s, a snapshot every {every} passes in {dir}; the copy's digest [{:#x}, {:#x}]",
        started.elapsed().as_secs_f64(),
        digest[0],
        digest[1]
    );
    if let Some(file) = opt("--against") {
        let other = vd_bins::artifact_file::read(&file).expect("the artifact file");
        let theirs = other.digest();
        println!(
            "  {file}: digest [{:#x}, {:#x}] — {}",
            theirs[0],
            theirs[1],
            if theirs == digest {
                "THE SAME SOLVE"
            } else {
                "A DIFFERENT SOLVE"
            }
        );
    }
}
