//! ★ N2's FAR DATA, MEASURED (the build design's slice S0 item 2,
//! `docs/investigation/2026-09-26/n2_build_design.md` §2.5, §3.5, Q1 and Q3; ruling B13 item 3).
//!
//! **WHY IT EXISTS.** From rung R₁ up the client holds no fine row, but N2 reads each fine node's
//! height within the 30 m band of a water and each lake node's level. The design asks the owner to
//! ship those words ("the shore band", an SL6 question) and to derive the far fold's sample counts
//! on the client. Both asks need MEASURED sizes and times, not arithmetic. This instrument reads
//! the home planet's artifact and prints them. It changes no world file and ships nothing.
//!
//! **WHAT IT PRINTS** (one `CENSUS` line each):
//! * BAND-NODES — the fine nodes whose height a far host needs (`shore_rules::shore_band`, the
//!   one definition): node `n` stands in the stencil of a node cell whose 6 × 6 block holds a node
//!   wet for water `b`, and its banded word is not its side's edge (wet for `b`: `z > W − 30`; not
//!   wet: `W − 30 < z < W + 30`).
//! * BAND-BYTES — the band as a sorted, delta-coded node list (LEB128 deltas) with a 2-byte height
//!   each, and as a reach bitmap (one bit a node) with the same heights; the smaller wins.
//! * LAKE-NODES — the lake nodes, whose level the mask does not carry (2 bytes each).
//! * CORNER-MIXED-CELLS — the node cells whose four corners stand on both sides of some water of
//!   their block: by the held read's maximum principle only these draw samples on both sides.
//! * FAR-FOLD-COST — the time to key a mixed cell's 64 × 64 rung-7 samples with the bench's f64
//!   side (the words once per cell, the held read per sample), on one thread.
//! * EMPTY-GLOBE — the mixed cells times that cost over a two-worker share: the seconds a far
//!   globe would wait for its derived sample pyramid (ARITH from the two measured numbers).
//! * MIXED-NODE-SQUARES — the node squares (a rung-13 column each) that hold a quarter of a mixed
//!   cell: the unit the shipped far counts need (ruling B14 item 3).
//!
//! **Example.** The gentle coast rises a metre in fifty: its land nodes within 1.5 km of the sea
//! stand within 30 m of it and join the band; a cliff coast crosses the band inside one node, so
//! only its shore nodes join. The census counts both on the whole globe.
//!
//! ```text
//! cargo run --release -p vd-bins --example n2_far_data -- --artifact-cache <home.art> \
//!   [--census <file>] [--sample-cells 400]
//! ```

use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_bins::shore_rules::{
    NamedChoices, NodeSite, Nodes, WORD_LAKE, WORD_SEA, Water, block_index, candidates, dweights,
    edge_coast, held_read, mirrored, point_side, shore_band, weights_f64,
};
use vd_seed::bend::Face;
use vd_terrain::artifact::node_at;
use vd_terrain::home::{home_planet, home_solve_words};

/// The band's half-width, metres (the benched N2's `word_band_m`).
const BAND_M: i32 = 30;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|k| args.get(k + 1).cloned())
    };
    let cache = arg("--artifact-cache");
    let census_path = arg("--census");
    let sample_cells: usize = arg("--sample-cells")
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut out: Vec<String> = Vec::new();

    let t = Instant::now();
    let artifact = match cache.as_deref().and_then(vd_bins::artifact_file::read) {
        Some(a) => {
            say(
                &mut out,
                format!(
                    "# the artifact READ from {cache:?} (a development run; a census run solves)"
                ),
            );
            a
        }
        None => run_solve(&SolveJob {
            body: home_planet(),
            words: home_solve_words(),
        })
        .expect("the home planet solves"),
    };
    let d = artifact.digest();
    say(
        &mut out,
        format!(
            "ARTIFACT version {} digest {:016x}{:016x}; the sea {} m; read in {:.1} s",
            artifact.version,
            d[0],
            d[1],
            artifact.sea_m,
            t.elapsed().as_secs_f64()
        ),
    );
    let body = home_planet().with_sea_m(artifact.sea());
    let fine = body.macro_lattice().expect("a lattice");
    let nodes = Nodes::of(&artifact);
    let n_nodes = nodes.z_m.len();
    let edge = i64::from(fine.edge);
    say(
        &mut out,
        format!(
            "the fine lattice: {} nodes an edge, {} nodes, {} rung-0 cells a node",
            fine.edge, n_nodes, fine.cells_per_node
        ),
    );

    // ── THE BAND (the one definition, `shore_band`) AND THE MIXED CELLS ───────────────────────
    let t = Instant::now();
    let in_band = shore_band(&nodes, &fine, f64::from(BAND_M));
    let mut mixed: Vec<(Face, i64, i64)> = Vec::new();
    let mut cells = 0u64;
    let mut cells_with_water = 0u64;
    for face in Face::ALL {
        for nj in -1..edge {
            for ni in -1..edge {
                cells += 1;
                let mut block = [0u32; 36];
                for (k, slot) in block.iter_mut().enumerate() {
                    let (oi, oj) = ((k % 6) as i64 - 2, (k / 6) as i64 - 2);
                    *slot = node_at(&fine, face, ni + oi, nj + oj);
                }
                let mut waters: Vec<Water> = Vec::new();
                for &n in &block {
                    if let Some(w) = nodes.water_of(n)
                        && !waters.contains(&w)
                    {
                        waters.push(w);
                    }
                }
                if waters.is_empty() {
                    continue;
                }
                cells_with_water += 1;
                let mut is_mixed = false;
                for &b in &waters {
                    // The corners: offsets (0, 0), (1, 0), (0, 1), (1, 1).
                    let wet_corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .filter(|&&(oi, oj)| nodes.wet_for(block[block_index(oi, oj)], b))
                        .count();
                    is_mixed |= wet_corners > 0 && wet_corners < 4;
                }
                if is_mixed {
                    mixed.push((face, ni, nj));
                }
            }
        }
    }
    let band_nodes: Vec<u32> = in_band
        .iter()
        .enumerate()
        .filter_map(|(k, &b)| b.then_some(k as u32))
        .collect();
    let walk_s = t.elapsed().as_secs_f64();
    say(
        &mut out,
        format!(
            "the walk: {cells} node cells ({} a face, the seam cells on both faces), {cells_with_water} with a water in their block, in {walk_s:.1} s",
            (edge + 1) * (edge + 1)
        ),
    );

    // ── THE SIZES ──────────────────────────────────────────────────────────────────────────────
    let mut delta_bytes = 0u64;
    let mut prev = 0u32;
    for (k, &n) in band_nodes.iter().enumerate() {
        let delta = if k == 0 { n } else { n - prev };
        let mut v = delta;
        loop {
            delta_bytes += 1;
            v >>= 7;
            if v == 0 {
                break;
            }
        }
        prev = n;
    }
    let heights = 2 * band_nodes.len() as u64;
    let list_bytes = delta_bytes + heights;
    let bitmap_bytes = (n_nodes as u64).div_ceil(8) + heights;
    let lake_nodes = nodes.word.iter().filter(|&&w| w == WORD_LAKE).count() as u64;
    let sea_nodes = nodes.word.iter().filter(|&&w| w == WORD_SEA).count() as u64;
    let mb = |b: u64| b as f64 / 1_048_576.0;
    say(
        &mut out,
        format!(
            "CENSUS BAND-NODES {} of {} nodes ({:.2} %) (`shore_band`: a wet node joins over W - 30, any other node inside W +- 30; S0's first count, 202 431, left out the wet nodes at W - 1 and over) no-band[S0-item-2;Q1]",
            band_nodes.len(),
            n_nodes,
            100.0 * band_nodes.len() as f64 / n_nodes as f64
        ),
    );
    say(
        &mut out,
        format!(
            "CENSUS BAND-BYTES list{:.3}MB(deltas{:.3}MB+heights{:.3}MB)/bitmap{:.3}MB/smaller{:.3}MB no-band[S0-item-2;Q1]",
            mb(list_bytes),
            mb(delta_bytes),
            mb(heights),
            mb(bitmap_bytes),
            mb(list_bytes.min(bitmap_bytes))
        ),
    );
    say(
        &mut out,
        format!(
            "CENSUS LAKE-NODES {lake_nodes} nodes {:.3}MB (2 bytes a level; the sea's nodes {sea_nodes}) no-band[S0-item-2;Q1]",
            mb(2 * lake_nodes)
        ),
    );
    // POCKET-LAKES-NEAR-SEA (S0 item 3, Q11): lake nodes under the sea's level with a sea node
    // within three nodes (the side hold's reach), and the distinct lake levels among them.
    let mut near_nodes = 0u64;
    let mut near_levels: Vec<i16> = Vec::new();
    for face in Face::ALL {
        for nj in 0..edge {
            for ni in 0..edge {
                let n = node_at(&fine, face, ni, nj);
                if nodes.word[n as usize] != WORD_LAKE {
                    continue;
                }
                let level = nodes.level_m[n as usize];
                if level >= nodes.sea_m {
                    continue;
                }
                let mut sea_near = false;
                for dj in -3..=3i64 {
                    for di in -3..=3i64 {
                        let m = node_at(&fine, face, ni + di, nj + dj);
                        sea_near |= nodes.word[m as usize] == WORD_SEA;
                    }
                }
                if sea_near {
                    near_nodes += 1;
                    if !near_levels.contains(&level) {
                        near_levels.push(level);
                    }
                }
            }
        }
    }
    say(
        &mut out,
        format!(
            "CENSUS POCKET-LAKES-NEAR-SEA {near_nodes} lake nodes under the sea's level within 3 nodes of a sea node, {} distinct levels {:?} no-band[S0-item-3;Q11]",
            near_levels.len(),
            &near_levels[..near_levels.len().min(12)]
        ),
    );
    say(
        &mut out,
        format!(
            "CENSUS CORNER-MIXED-CELLS {} of {cells} cells ({:.2} %) no-band[S0-item-2;Q3]",
            mixed.len(),
            100.0 * mixed.len() as f64 / cells as f64
        ),
    );

    // ── THE FOLD'S COST: key a sample of mixed cells' 64 × 64 rung-7 samples ───────────────────
    let ch = NamedChoices::default();
    let rung_f: u8 = 7;
    let per_node = i64::from(fine.cells_per_node >> rung_f);
    let step = (mixed.len() / sample_cells.max(1)).max(1);
    let t = Instant::now();
    let mut keyed = 0u64;
    let mut wet_samples = 0u64;
    let mut timed = 0usize;
    for &(face, ni, nj) in mixed.iter().step_by(step).take(sample_cells) {
        // The rung-7 columns whose node cell is (ni, nj): i = per_node·ni + per_node/2 + 0..per_node.
        let i0 = per_node * ni + per_node / 2;
        let j0 = per_node * nj + per_node / 2;
        for b in 0..per_node {
            for a in 0..per_node {
                let (i, j) = ((i0 + a) as i32, (j0 + b) as i32);
                let site = NodeSite::of(&fine, face, rung_f, i, j);
                let block = site.block(&fine);
                let (wa, wb) = (weights_f64(site.ta_q), weights_f64(site.tb_q));
                let (dwa, dwb) = (dweights(site.ta), dweights(site.tb));
                let cands = candidates(&nodes, &block, site.ta, site.tb, &wa, &wb, &dwa, &dwb, &ch);
                let mut fields = [0.0f64; 8];
                for (k, c) in cands.as_slice().iter().enumerate() {
                    fields[k] = c.s;
                }
                if let Some((true, _)) = point_side(&cands, &fields, nodes.sea_m) {
                    wet_samples += 1;
                }
                keyed += 1;
            }
        }
        timed += 1;
    }
    let fold_s = t.elapsed().as_secs_f64();
    let per_cell_us = 1e6 * fold_s / timed.max(1) as f64;
    say(
        &mut out,
        format!(
            "CENSUS FAR-FOLD-COST {per_cell_us:.0}us/mixed-cell ({:.2}us/sample; {timed} cells, {keyed} samples, {wet_samples} wet; the bench's f64 side per sample, one thread; an upper bound: the integer form reads the words once a cell) no-band[S0-item-2;Q3]",
            1e6 * fold_s / keyed.max(1) as f64
        ),
    );
    // THE FOLD AS A HOST WOULD RUN IT: each water's sixteen banded words ONCE a cell, then only
    // the held read and the side per sample (the design §3.3's record).
    let t = Instant::now();
    let mut keyed2 = 0u64;
    let mut wet2 = 0u64;
    for &(face, ni, nj) in mixed.iter().step_by(step).take(sample_cells) {
        let mut block = [0u32; 36];
        for (k, slot) in block.iter_mut().enumerate() {
            let (oi, oj) = ((k % 6) as i64 - 2, (k / 6) as i64 - 2);
            *slot = node_at(&fine, face, ni + oi, nj + oj);
        }
        let mut words: Vec<(f64, [f64; 16])> = Vec::new();
        let mut seen: Vec<Water> = Vec::new();
        for &n in &block {
            if let Some(b) = nodes.water_of(n)
                && !seen.contains(&b)
            {
                seen.push(b);
                let w = b.level_m(nodes.sea_m);
                let mut zhat = [0.0f64; 16];
                for (st, slot) in zhat.iter_mut().enumerate() {
                    let (oi, oj) = ((st % 4) as i64 - 1, (st / 4) as i64 - 1);
                    let m = block[block_index(oi, oj)];
                    let z = f64::from(nodes.z_m[m as usize]);
                    *slot = mirrored(z, nodes.wet_for(m, b), w)
                        .clamp(w - ch.word_band_m, w + ch.word_band_m);
                }
                words.push((w, zhat));
            }
        }
        let i0 = per_node * ni + per_node / 2;
        let j0 = per_node * nj + per_node / 2;
        for b in 0..per_node {
            let site_j = NodeSite::of(&fine, face, rung_f, (i0) as i32, (j0 + b) as i32);
            let wb = weights_f64(site_j.tb_q);
            for a in 0..per_node {
                let site = NodeSite::of(&fine, face, rung_f, (i0 + a) as i32, (j0 + b) as i32);
                let wa = weights_f64(site.ta_q);
                let wet = words
                    .iter()
                    .any(|(w, zhat)| held_read(zhat, &wa, &wb, ch.held_both_orders) - w < 0.0);
                wet2 += u64::from(wet);
                keyed2 += 1;
            }
        }
    }
    let fold2_s = t.elapsed().as_secs_f64();
    let per_cell2_us = 1e6 * fold2_s / timed.max(1) as f64;
    say(
        &mut out,
        format!(
            "CENSUS FAR-FOLD-COST-RECORD {per_cell2_us:.0}us/mixed-cell ({:.3}us/sample; {keyed2} samples, {wet2} wet against the per-sample path's {wet_samples}; the words once a cell, the held read per sample, f64, one thread) no-band[S0-item-2;Q3]",
            1e6 * fold2_s / keyed2.max(1) as f64
        ),
    );
    let globe2_s = mixed.len() as f64 * per_cell2_us / 1e6;
    say(
        &mut out,
        format!(
            "CENSUS EMPTY-GLOBE-RECORD {:.0}s-on-one-core/{:.0}s-on-two-workers (the {} mixed cells x {per_cell2_us:.0}us; ARITH from two measured numbers) no-band[S0-item-2;Q3]",
            globe2_s,
            globe2_s / 2.0,
            mixed.len()
        ),
    );
    say(
        &mut out,
        format!(
            "CENSUS FOLD-SHIPPED-BYTES {:.3}MB (three 2-byte words a mixed CELL; the first count, kept for the record: a rung-13 column is a NODE SQUARE, which halves four node cells, so the shipped unit is the square, MIXED-NODE-SQUARES) no-band[S0-item-2;Q3]",
            mb(6 * mixed.len() as u64)
        ),
    );
    // THE NODE SQUARES the shipped counts need: node square (si, sj) is centred on node (si, sj)
    // and holds a quarter of the node cells (si - 1 ..= si) x (sj - 1 ..= sj); a square needs its
    // counts shipped when any of those four cells is mixed (a uniform quarter's counts follow from
    // one point side). Squares with a node on the face: 0 ..= edge - 1 on each axis.
    let mut squares: std::collections::BTreeSet<(u8, i64, i64)> = std::collections::BTreeSet::new();
    for &(face, ni, nj) in &mixed {
        for (si, sj) in [(ni, nj), (ni + 1, nj), (ni, nj + 1), (ni + 1, nj + 1)] {
            if (0..edge).contains(&si) && (0..edge).contains(&sj) {
                squares.insert((face.index(), si, sj));
            }
        }
    }
    say(
        &mut out,
        format!(
            "CENSUS MIXED-NODE-SQUARES {} of {} squares ({:.2} %); FOLD-SHIPPED-BYTES-SQUARES {:.3}MB (three 2-byte words a square: its 4 096 samples' wet count, lake count and highest lake level, the one naming rule's words) no-band[S0-item-2;B14-item-3]",
            squares.len(),
            6 * edge * edge,
            100.0 * squares.len() as f64 / (6 * edge * edge) as f64,
            mb(6 * squares.len() as u64)
        ),
    );
    let globe_s = mixed.len() as f64 * per_cell_us / 1e6;
    say(
        &mut out,
        format!(
            "CENSUS EMPTY-GLOBE {:.0}s-on-one-core/{:.0}s-on-two-workers (the home planet's {} mixed cells x {per_cell_us:.0}us; ARITH from two measured numbers; a hemisphere is about half) no-band[S0-item-2;Q3]",
            globe_s,
            globe_s / 2.0,
            mixed.len()
        ),
    );
    seam_and_corners(&mut out, &nodes, &fine);
    if let Some(p) = census_path {
        let _ = std::fs::write(&p, out.join("\n") + "\n");
    }
}

/// ★ THE SEAM AND THE CORNER (the build design's S0 item 6): EDGE-COAST counts the node pairs that
/// straddle a cube edge with another water on each side (each pair once), per edge, and names the
/// busiest stretch as the seam stand; CORNER-COAST names every face corner whose 4 × 4 corner nodes
/// hold both sides of some water (a coast within three nodes of the cube corner).
fn seam_and_corners(
    out: &mut Vec<String>,
    nodes: &Nodes,
    fine: &vd_terrain::macro_lattice::MacroLattice,
) {
    let edge = i64::from(fine.edge);
    let ec = edge_coast(nodes, fine);
    say(
        out,
        format!(
            "CENSUS EDGE-COAST {} node pairs straddle a cube edge with another water on each side ({}) no-band[S0-item-6]",
            ec.pairs,
            ec.per_edge
                .iter()
                .map(|(f, e, n)| format!("{f:?}/{e:?}:{n}"))
                .collect::<Vec<_>>()
                .join(" ")
        ),
    );
    if let Some((count, face, side, at)) = ec.best {
        say(
            out,
            format!(
                "CENSUS SEAM-STAND face {face:?} side {side:?} at node {at} along the edge: {count} straddling pairs within 32 nodes (the busiest stretch; the bench's W-SEAM stands there) no-band[S0-item-6]"
            ),
        );
    }
    // MIXED-LEVEL-NODES and TOP-BYTES (the far-water design's Q3): the level-1 nodes (2 x 2 fine
    // nodes) that are part wet and part dry, whose dry fine nodes' mean the lid's exact form needs.
    let mut part_wet = 0u64;
    let mut level1 = 0u64;
    for face in Face::ALL {
        for cj in 0..edge / 2 {
            for ci in 0..edge / 2 {
                level1 += 1;
                let mut wet = 0;
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let n = node_at(fine, face, 2 * ci + da, 2 * cj + db);
                    wet += usize::from(nodes.water_of(n).is_some());
                }
                part_wet += u64::from(wet > 0 && wet < 4);
            }
        }
    }
    say(
        out,
        format!(
            "CENSUS MIXED-LEVEL-NODES {part_wet} of {level1} level-1 nodes are part wet; TOP-BYTES {:.1}KB at 2 B each (the lid's dry word, the far-water design's Q3) no-band[design-D-Q3]",
            part_wet as f64 * 2.0 / 1024.0
        ),
    );
    let mut coastal: Vec<String> = Vec::new();
    for face in Face::ALL {
        for (ci, cj) in [
            (0i64, 0i64),
            (edge - 1, 0),
            (0, edge - 1),
            (edge - 1, edge - 1),
        ] {
            let (di, dj) = (if ci == 0 { 1 } else { -1 }, if cj == 0 { 1 } else { -1 });
            let mut waters: Vec<Water> = Vec::new();
            let mut ns: Vec<u32> = Vec::new();
            for b in 0..4i64 {
                for a in 0..4i64 {
                    let n = node_at(fine, face, ci + di * a, cj + dj * b);
                    ns.push(n);
                    if let Some(w) = nodes.water_of(n)
                        && !waters.contains(&w)
                    {
                        waters.push(w);
                    }
                }
            }
            let mixed = waters.iter().any(|&w| {
                let wet = ns.iter().filter(|&&n| nodes.wet_for(n, w)).count();
                wet > 0 && wet < ns.len()
            });
            if mixed {
                coastal.push(format!("{face:?}({ci},{cj})"));
            }
        }
    }
    say(
        out,
        format!(
            "CENSUS CORNER-COAST {} of 24 face corners hold a coast within three nodes of the cube corner: {} no-band[S0-item-6;Q12]",
            coastal.len(),
            if coastal.is_empty() {
                "none".to_string()
            } else {
                coastal.join(" ")
            }
        ),
    );
}

/// One census line: printed, and kept for the census file.
fn say(out: &mut Vec<String>, line: String) {
    println!("{line}");
    out.push(line);
}
