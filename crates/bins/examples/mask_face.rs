//! A diagnostic (not a gate): the coast mask's side word of every fine node of one face, one node a
//! pixel — land black, the sea white, a lake grey, a pocket's floor dark red — so a pattern seen in
//! the skin's pictures can be told apart as the solve's data or N2's samples.
//!
//! ```text
//! cargo run --release -p vd-bins --example mask_face -- --artifact-cache <home.art> --face NegY --out <file.png>
//! cargo run --release -p vd-bins --example mask_face -- --artifact-cache <home.art> --edges
//! ```
//!
//! `--edges` reads, over all six faces, where the water's edges and the ground's steps stand
//! against the solve's 16 × 16-node rebound blocks (the status file's W10): the share of the
//! water/land edges between two nodes that lie on a block line (a node index `k` with
//! `k mod 16 = 0`, against 1/16 by chance), and the mean height step across a block line against
//! the mean step inside the blocks.

use image::{Rgb, RgbImage};
use vd_bins::shore_rules::{Nodes, WORD_LAKE, WORD_POCKET, WORD_SEA};
use vd_seed::bend::Face;
use vd_terrain::artifact::node_at;
use vd_terrain::home::home_planet;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|k| args.get(k + 1).cloned())
    };
    let cache = arg("--artifact-cache").expect("--artifact-cache");
    if args.iter().any(|a| a == "--edges") {
        edges(&cache);
        return;
    }
    let out = arg("--out").expect("--out");
    let face_name = arg("--face").unwrap_or_else(|| "NegY".into());
    let artifact = vd_bins::artifact_file::read(&cache).expect("the artifact");
    let body = home_planet().with_sea_m(artifact.sea());
    let fine = body.macro_lattice().expect("a lattice");
    let nodes = Nodes::of(&artifact);
    let face = Face::ALL
        .into_iter()
        .find(|f| format!("{f:?}") == face_name)
        .expect("a face name");
    let n = fine.edge;
    let mut img = RgbImage::new(n, n);
    for j in 0..n {
        for i in 0..n {
            let node = node_at(&fine, face, i64::from(i), i64::from(j));
            let px = match nodes.word[node as usize] {
                WORD_SEA => Rgb([255, 255, 255]),
                WORD_LAKE => Rgb([150, 150, 150]),
                WORD_POCKET => Rgb([120, 20, 20]),
                _ => Rgb([0, 0, 0]),
            };
            img.put_pixel(i, n - 1 - j, px);
        }
    }
    img.save(&out).expect("the picture");
    println!("wrote {out}: face {face:?}, {n} x {n} nodes");
}

/// The water/land edges and the height steps against the 16-node block lines, for moduli 8, 16, 32.
fn edges(cache: &str) {
    let artifact = vd_bins::artifact_file::read(cache).expect("the artifact");
    let body = home_planet().with_sea_m(artifact.sea());
    let fine = body.macro_lattice().expect("a lattice");
    let nodes = Nodes::of(&artifact);
    let n = i64::from(fine.edge);
    let wet = |w: u8| w == WORD_SEA || w == WORD_LAKE;
    for m in [8i64, 16, 32] {
        // [on a block line, off it]: water/land edges, and Σ|Δz|, pairs.
        let mut coast = [0u64; 2];
        let mut step = [0f64; 2];
        let mut pairs = [0u64; 2];
        let mut by_phase = vec![0u64; m as usize];
        for face in Face::ALL {
            for j in 0..n {
                for i in 0..n {
                    let a = node_at(&fine, face, i, j) as usize;
                    for (di, dj) in [(1i64, 0i64), (0, 1)] {
                        let (i2, j2) = (i + di, j + dj);
                        if i2 >= n || j2 >= n {
                            continue;
                        }
                        let b = node_at(&fine, face, i2, j2) as usize;
                        // The line between the two nodes stands at index i2 (or j2).
                        let k = if di == 1 { i2 } else { j2 };
                        let on = usize::from(k.rem_euclid(m) != 0);
                        let (wa, wb) = (wet(nodes.word[a]), wet(nodes.word[b]));
                        if wa != wb {
                            coast[on] += 1;
                            by_phase[k.rem_euclid(m) as usize] += 1;
                        }
                        if !wa && !wb {
                            step[on] += f64::from(
                                (i32::from(nodes.z_m[a]) - i32::from(nodes.z_m[b])).abs(),
                            );
                            pairs[on] += 1;
                        }
                    }
                }
            }
        }
        let total = coast[0] + coast[1];
        let (peak, peak_n) = by_phase
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map_or((0, 0), |(k, &c)| (k, c));
        println!(
            "MOD {m}: water/land edges on a block line {} of {total} = {:.2} % (by chance {:.2} %); the busiest phase {peak} with {:.2} %; the mean dry height step across a block line {:.1} m against {:.1} m inside the blocks ({} and {} dry pairs)",
            coast[0],
            100.0 * coast[0] as f64 / total.max(1) as f64,
            100.0 / m as f64,
            100.0 * peak_n as f64 / total.max(1) as f64,
            step[0] / pairs[0].max(1) as f64,
            step[1] / pairs[1].max(1) as f64,
            pairs[0],
            pairs[1]
        );
    }
}
