//! ★ BENCH STEP 1, PART 1 — THE DRAINAGE LADDER ON EARTH (ruling B79; the landforms design
//! `docs/investigation/2026-10-11/landforms_by_law_design.md`, revision 2: §2 F2, §3, §5.1).
//!
//! A MEASUREMENT in floats (the build owes the integer form), never a game path. Nothing here touches the game: no
//! game crate changes, nothing is shipped, the Earth data stays outside the repository, and no constant is set from
//! Earth (Earth is a reading, B38 item 4).
//!
//! For each Earth tile (Taiwan's Central Range N23 E120, the Oregon Coast Range N43 W124):
//!
//! 1. **EARTH'S 8 KM FACTS** — each 8 192 m node's MEAN height `Z` (the meaning the game's `Z` has); the river TREE by
//!    the game's kind of routing on `Z` (a priority flood from the outlets, then the steepest descent on the filled
//!    surface, a flat taking the cell the flood came from; outlets the tile's rim and the sea); the law's own water
//!    `Q = R·A` over that tree; the rain `R` a per-tile published mean (stated per tile below).
//! 2. **THE DRAINAGE LADDER** (§3): the trunks on the tree's links (level 0); levels of halving cells down to the law's
//!    own valley spacing `λ = c_λ·l_c` (`c_λ` solved here for `n` = 2.2 from the law's own non-dimensional form);
//!    each finer source ROUTED DOWN the ground the start and the coarser levels make until it reaches a coarser
//!    stream's floor; the stream law's analytic bed (`S = k_s·Q^(−1/n)`, the joint law `E = K_lp·Q·S^n`); Roering's
//!    sides (S1) and rounded divides (S2); the carve into the start `h = min(S + L, P)` (no deposits, no roughness:
//!    part 1 is F2 and the channels only).
//! 3. **THE PICTURES**: the generated ground's shaded relief beside Earth's real ground of the same square at the same
//!    cell — the whole square at 128 m, a 48 km window at 32 m.
//! 4. **THE READINGS** for both grounds at the same cell, and the cost.
//!
//! Run in release, one job at a time on the machine:
//!
//! ```text
//! cargo run --release -p vd-bins --example landforms_bench -- --out DIR
//! ```

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use vd_bins::drainage_gates::{Heights, g_hack, spill_census};
use vd_terrain::solve::{JOINT_K_LP, JOINT_N};
use vd_terrain::surface::SOIL_CRITICAL_SLOPE;

/// The node's side, metres (the game's macro node on the home planet, `macro_lattice.rs:331`).
const NODE_M: f64 = 8_192.0;
/// ★ S1's transport coefficient `K_h`, m²/yr (Roering, Kirchner & Dietrich 1999, the Oregon Coast Range; the same
/// interim the R2 bench's hillslope law reads).
const K_H: f64 = 0.003;
/// ★ S1/S2's rock-to-soil density ratio `ρ_r/ρ_s` (Roering, Perron & Kirchner 2007, the Oregon Coast Range) — an
/// interim of the design's list (§2.0 "the interims"), NOT read at first hand in this bench.
const RHO_RATIO: f64 = 2.0;
/// ★ The channel's width `w = a·Q^b` with `Q` the mean flow in m³/s (Moody & Troutman 2002, a global set: `a` 7.2,
/// `b` 0.5; the downstream hydraulic geometry of Leopold & Maddock 1953) — an interim, NOT read at first hand here.
const WIDTH_A: f64 = 7.2;
const WIDTH_B: f64 = 0.5;
/// Seconds in a year (the width law reads m³/s).
const YEAR_S: f64 = 31_557_600.0;
/// The joint law's area exponent on `Q = R·A` (ruling B51).
const M_EXP: f64 = 1.0;

/// One Earth tile and its stated rain.
struct TileSpec {
    path: &'static str,
    label: &'static str,
    /// The per-tile published mean rain, mm/yr, and its source.
    rain_mm: f64,
    rain_source: &'static str,
}

const TILES: [TileSpec; 2] = [
    TileSpec {
        path: "/Users/maxim/voxeldust_earth/srtm1/N23E120.hgt",
        label: "taiwan",
        rain_mm: 2_502.0,
        rain_source: "Taiwan's Water Resources Agency, the island's mean annual rain 1949-2009 (isohyetal map): 2 502 mm",
    },
    TileSpec {
        path: "/Users/maxim/voxeldust_earth/srtm1/N43W124.hgt",
        label: "oregon",
        rain_mm: 1_778.0,
        rain_source: "the City of Coos Bay's 2006 wastewater master plan: the zone of 60-80 inches a year; its middle, 70 in",
    },
];

/// The eight neighbours, `(dx, dy)`.
const NB: [(i64, i64); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
/// No receiver.
const NONE: usize = usize::MAX;

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE EARTH READER (copied from `erosion_bench`, unchanged in its rule).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ AN EARTH TILE IN SQUARE CELLS: a 1-arc-second HGT tile (3 601 × 3 601 big-endian heights, north row first, its
/// south-west corner in the file's name, voids −32 768 skipped) averaged into square cells of `cell_m` over the
/// largest square the tile holds from its north-west corner (an equirectangular frame at the tile's middle latitude).
/// Returns the heights (metres, row by row, north first) and the side in cells.
fn earth_cells(path: &str, cell_m: f64) -> (Vec<f64>, usize) {
    const N: usize = 3_601;
    let bytes = std::fs::read(path).expect("the HGT tile reads");
    assert_eq!(bytes.len(), N * N * 2, "a 1-arc-second HGT tile");
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("a tile name");
    let lat0: f64 = name[1..3].parse().expect("the latitude");
    let lat0 = if name.starts_with('S') { -lat0 } else { lat0 };
    let mid = (lat0 + 0.5).to_radians();
    let (m_lat, m_lon): (f64, f64) = (110_574.0, 111_320.0 * mid.cos());
    let side_m = m_lat.min(m_lon);
    let w = (side_m / cell_m).floor() as usize;
    let mut sum = vec![0.0f64; w * w];
    let mut count = vec![0u32; w * w];
    for row in 0..N {
        let y_m = row as f64 / (N - 1) as f64 * m_lat;
        for col in 0..N {
            let x_m = col as f64 / (N - 1) as f64 * m_lon;
            let (cx, cy) = ((x_m / cell_m) as usize, (y_m / cell_m) as usize);
            if cx >= w || cy >= w {
                continue;
            }
            let k = 2 * (row * N + col);
            let h = i16::from_be_bytes([bytes[k], bytes[k + 1]]);
            if h == i16::MIN {
                continue;
            }
            sum[cy * w + cx] += f64::from(h);
            count[cy * w + cx] += 1;
        }
    }
    let z = sum
        .iter()
        .zip(&count)
        .map(|(s, &c)| if c > 0 { s / f64::from(c) } else { 0.0 })
        .collect();
    (z, w)
}

/// The top-left `side × side` square of a `w × w` field.
fn crop(f: &[f64], w: usize, x0: usize, y0: usize, side: usize) -> Vec<f64> {
    (0..side)
        .flat_map(|y| (0..side).map(move |x| f[(y + y0) * w + x + x0]))
        .collect()
}

/// The `k × k` block means of a `w × w` field (`w` a multiple of `k`).
fn block_mean(f: &[f32], w: usize, k: usize) -> Vec<f64> {
    let n = w / k;
    let mut out = vec![0.0f64; n * n];
    for y in 0..w {
        for x in 0..w {
            out[(y / k) * n + x / k] += f64::from(f[y * w + x]);
        }
    }
    let inv = 1.0 / (k * k) as f64;
    out.iter_mut().for_each(|v| *v *= inv);
    out
}

/// A site's own chance on `[0, 1)`: splitmix64 of the site's key (level, cell, axis).
fn h01(level: u64, i: u64, j: u64, axis: u64) -> f64 {
    let mut z = (level.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ j.wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ axis.wrapping_mul(0x27D4_EB2F_1656_67C5);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Catmull-Rom through four samples at `t ∈ [0, 1)` between the middle two.
fn catmull_rom(s: [f64; 4], t: f64) -> f64 {
    let (a, b, c, d) = (s[0], s[1], s[2], s[3]);
    0.5 * ((2.0 * b)
        + (c - a) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t * t
        + (3.0 * b - 3.0 * c + d - a) * t * t * t)
}

/// The four Catmull-Rom weights at `t`.
fn cr_weights(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [
        0.5 * (-t + 2.0 * t2 - t3),
        0.5 * (2.0 - 5.0 * t2 + 3.0 * t3),
        0.5 * (t + 4.0 * t2 - 3.0 * t3),
        0.5 * (-t2 + t3),
    ]
}

/// A node field (`n × n`, node centres at `(i + ½)·NODE_M`) read by Catmull-Rom at `(x, y)` metres, the rim held.
fn node_cr(vals: &[f64], n: usize, x: f64, y: f64) -> f64 {
    let (u, v) = (x / NODE_M - 0.5, y / NODE_M - 0.5);
    let (iu, iv) = (u.floor() as i64, v.floor() as i64);
    let (wu, wv) = (cr_weights(u - u.floor()), cr_weights(v - v.floor()));
    let at = |i: i64, j: i64| {
        vals[(j.clamp(0, n as i64 - 1) as usize) * n + i.clamp(0, n as i64 - 1) as usize]
    };
    let mut s = 0.0;
    for (b, wy) in wv.iter().enumerate() {
        let mut r = 0.0;
        for (a, wx) in wu.iter().enumerate() {
            r += wx * at(iu + a as i64 - 1, iv + b as i64 - 1);
        }
        s += wy * r;
    }
    s
}

/// A node field read by Catmull-Rom at every cell centre of a `w × w` raster of `cw` metres (separable).
fn node_cr_raster(vals: &[f64], n: usize, w: usize, cw: f64) -> Vec<f32> {
    // First along x for every node row, then along y.
    let mut rows = vec![0.0f64; n * w];
    for j in 0..n {
        for x in 0..w {
            let u = (x as f64 + 0.5) * cw / NODE_M - 0.5;
            let iu = u.floor() as i64;
            let wu = cr_weights(u - u.floor());
            let mut s = 0.0;
            for (a, wx) in wu.iter().enumerate() {
                let i = (iu + a as i64 - 1).clamp(0, n as i64 - 1) as usize;
                s += wx * vals[j * n + i];
            }
            rows[j * w + x] = s;
        }
    }
    let mut out = vec![0.0f32; w * w];
    for y in 0..w {
        let v = (y as f64 + 0.5) * cw / NODE_M - 0.5;
        let iv = v.floor() as i64;
        let wv = cr_weights(v - v.floor());
        let js: Vec<usize> = (0..4)
            .map(|b| (iv + b as i64 - 1).clamp(0, n as i64 - 1) as usize)
            .collect();
        for x in 0..w {
            let mut s = 0.0;
            for b in 0..4 {
                s += wv[b] * rows[js[b] * w + x];
            }
            out[y * w + x] = s as f32;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// EARTH'S 8 KM FACTS.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The 8 km facts of one tile: the game's inputs, read from Earth.
struct Facts {
    /// Nodes a side.
    n: usize,
    /// The rain, m/yr (one per-tile mean).
    rain_m: f64,
    /// Each node's mean height, metres.
    z: Vec<f64>,
    /// The node is the sea (more than half its 32 m cells at or under nought): an outlet, no ladder.
    sea: Vec<bool>,
    /// The receiver node, or [`NONE`] at an outlet (the rim, the sea).
    recv: Vec<usize>,
    /// The nodes, outlets first (the flood's order: every receiver before its donors).
    order: Vec<usize>,
    /// The tree's area, m², the node's own included.
    area: Vec<f64>,
    /// The node's anchor (uniform over the node), metres.
    anchor: Vec<[f64; 2]>,
    /// `log k_s` per node (`k_s = S·Q^(1/n)`), the non-voting nodes filled by their voting neighbours' mean.
    log_ks: Vec<f64>,
    /// The node votes (a land node with `S > 0` along its link).
    vote: Vec<bool>,
    /// The link's slope `S` (3 × 3 least squares of `Z`, along the link; the steepest at a rim outlet).
    slope: Vec<f64>,
    /// The unit fall line of the 3 × 3 least-squares plane (`−∇Z`), a rim outlet's way out of the tile.
    fall: Vec<[f64; 2]>,
}

impl Facts {
    /// The facts from the node means `z` and the sea shares.
    fn new(z: Vec<f64>, sea_share: &[f64], n: usize, rain_m: f64) -> Facts {
        let sea: Vec<bool> = sea_share.iter().map(|&s| s > 0.5).collect();
        // The routing: a priority flood from the rim and the sea, the steepest descent on the filled surface.
        let cells = n * n;
        let key = |h: f64| -> i64 { (h * 65_536.0).round() as i64 };
        let mut zf = z.clone();
        let mut seen = vec![false; cells];
        let mut parent = vec![NONE; cells];
        let mut order = Vec::with_capacity(cells);
        let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
        for j in 0..n {
            for i in 0..n {
                let k = j * n + i;
                if i == 0 || j == 0 || i == n - 1 || j == n - 1 || sea[k] {
                    seen[k] = true;
                    heap.push(Reverse((key(zf[k]), k)));
                }
            }
        }
        while let Some(Reverse((_, c))) = heap.pop() {
            order.push(c);
            let (ci, cj) = ((c % n) as i64, (c / n) as i64);
            for (dx, dy) in NB {
                let (ni, nj) = (ci + dx, cj + dy);
                if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 {
                    continue;
                }
                let m = nj as usize * n + ni as usize;
                if seen[m] {
                    continue;
                }
                seen[m] = true;
                zf[m] = zf[m].max(zf[c]);
                parent[m] = c;
                heap.push(Reverse((key(zf[m]), m)));
            }
        }
        // ★ A rim node takes a receiver too where a neighbour stands lower on the filled surface: it is an OUTLET only
        // where water truly leaves the tile there (a crest node on the rim drains inward). The sea never drains.
        let mut recv = vec![NONE; cells];
        for k in 0..cells {
            if sea[k] {
                continue;
            }
            let (ci, cj) = ((k % n) as i64, (k / n) as i64);
            let mut best = (0.0f64, NONE);
            for (dx, dy) in NB {
                let (ni, nj) = (ci + dx, cj + dy);
                if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 {
                    continue;
                }
                let m = nj as usize * n + ni as usize;
                let chord = if dx != 0 && dy != 0 {
                    NODE_M * std::f64::consts::SQRT_2
                } else {
                    NODE_M
                };
                let s = (zf[k] - zf[m]) / chord;
                if s > best.0 {
                    best = (s, m);
                }
            }
            recv[k] = if best.1 == NONE { parent[k] } else { best.1 };
        }
        // Receivers before donors: by the filled height, ties by the flood's own order.
        let mut rank = vec![0usize; cells];
        for (i, &c) in order.iter().enumerate() {
            rank[c] = i;
        }
        order.sort_by(|&a, &b| zf[a].total_cmp(&zf[b]).then(rank[a].cmp(&rank[b])));
        // The tree's area.
        let mut area = vec![NODE_M * NODE_M; cells];
        for &c in order.iter().rev() {
            if recv[c] != NONE {
                area[recv[c]] += area[c];
            }
        }
        // The anchors.
        let anchor: Vec<[f64; 2]> = (0..cells)
            .map(|k| {
                let (i, j) = ((k % n) as f64, (k / n) as f64);
                [
                    (i + h01(0, (k % n) as u64, (k / n) as u64, 0)) * NODE_M,
                    (j + h01(0, (k % n) as u64, (k / n) as u64, 1)) * NODE_M,
                ]
            })
            .collect();
        // The slope: the 3 × 3 least-squares plane of Z, read along the link (the steepest at a rim outlet).
        let mut slope = vec![0.0f64; cells];
        let mut fall = vec![[0.0f64; 2]; cells];
        let mut vote = vec![false; cells];
        let mut log_ks = vec![0.0f64; cells];
        for k in 0..cells {
            if sea[k] {
                continue;
            }
            let (ci, cj) = ((k % n) as i64, (k / n) as i64);
            // Normal equations of z = a + b·x + c·y over the available nodes (x, y in nodes).
            let mut m = [[0.0f64; 3]; 3];
            let mut r = [0.0f64; 3];
            for dj in -1..=1i64 {
                for di in -1..=1i64 {
                    let (ni, nj) = (ci + di, cj + dj);
                    if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 {
                        continue;
                    }
                    let v = [1.0, di as f64, dj as f64];
                    let zz = z[nj as usize * n + ni as usize];
                    for a in 0..3 {
                        for b in 0..3 {
                            m[a][b] += v[a] * v[b];
                        }
                        r[a] += v[a] * zz;
                    }
                }
            }
            let g = solve3(m, r);
            let (gx, gy) = (g[1] / NODE_M, g[2] / NODE_M);
            let gl = (gx * gx + gy * gy).sqrt();
            if gl > 0.0 {
                fall[k] = [-gx / gl, -gy / gl];
            }
            let s = if recv[k] == NONE {
                (gx * gx + gy * gy).sqrt()
            } else {
                let rk = recv[k];
                let (dx, dy) = (((rk % n) as f64) - ci as f64, ((rk / n) as f64) - cj as f64);
                let len = (dx * dx + dy * dy).sqrt();
                -(gx * dx + gy * dy) / len
            };
            slope[k] = s;
            if s > 0.0 {
                vote[k] = true;
                let q = rain_m * area[k];
                log_ks[k] = s.ln() + q.ln() / JOINT_N;
            }
        }
        // The non-voting nodes take their voting neighbours' mean, ring by ring (a fill before the read).
        let mut have = vote.clone();
        for _ in 0..2 * n {
            let mut changed = false;
            let snapshot = have.clone();
            for k in 0..cells {
                if snapshot[k] {
                    continue;
                }
                let (ci, cj) = ((k % n) as i64, (k / n) as i64);
                let (mut s, mut c) = (0.0, 0usize);
                for (dx, dy) in NB {
                    let (ni, nj) = (ci + dx, cj + dy);
                    if ni < 0 || nj < 0 || ni >= n as i64 || nj >= n as i64 {
                        continue;
                    }
                    let m = nj as usize * n + ni as usize;
                    if snapshot[m] {
                        s += log_ks[m];
                        c += 1;
                    }
                }
                if c > 0 {
                    log_ks[k] = s / c as f64;
                    have[k] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Facts {
            n,
            rain_m,
            z,
            sea,
            recv,
            order,
            area,
            anchor,
            log_ks,
            vote,
            slope,
            fall,
        }
    }

    /// The node under `(x, y)` metres.
    fn node_at(&self, x: f64, y: f64) -> usize {
        let i = ((x / NODE_M).floor() as i64).clamp(0, self.n as i64 - 1) as usize;
        let j = ((y / NODE_M).floor() as i64).clamp(0, self.n as i64 - 1) as usize;
        j * self.n + i
    }
}

/// A 3 × 3 linear system by Cramer's rule.
fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    let det = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    let mut out = [0.0; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        let mut a = m;
        for row in 0..3 {
            a[row][c] = r[row];
        }
        *slot = det(a) / d;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE LAW AT A POINT (§2.0 S1, S2; §3.3; §3.4).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The law's quantities at one stream point.
#[derive(Clone, Copy, Debug)]
struct PointLaw {
    /// The steepness `k_s` (`S = k_s·Q^(−1/n)`).
    ks: f64,
    /// The steady rate `U = E = K_lp·k_s^n`, m/yr.
    e: f64,
    /// The law's length `l_c = (K_h^n·U^(1−n)/K′)^(1/(n+2m))`, `K′ = K_lp·R` (Theodoratos et al. 2018, eq. A8), m.
    lc: f64,
    /// The valley spacing `λ = c_λ·l_c`, m.
    lambda: f64,
    /// The hillslope length `L_H = λ/2`, m.
    lh: f64,
    /// S1's side slope at the stream: `2q/(K_h + √(K_h² + 4q²/S_c²))`, `q = (ρ_r/ρ_s)·E·L_H`.
    sh: f64,
    /// S2's smooth-minimum width `k = 2·S_h²/|C|`, `|C| = (ρ_r/ρ_s)·E/K_h`, m.
    k2: f64,
}

/// The law at a point from its `log k_s`, the rain and `c_λ`.
fn point_law(log_ks: f64, rain_m: f64, c_lambda: f64) -> PointLaw {
    let n = JOINT_N;
    let ks = log_ks.exp();
    let e = JOINT_K_LP * ks.powf(n);
    let kp = JOINT_K_LP * rain_m;
    let lc = ((n * K_H.ln() + (1.0 - n) * e.ln() - kp.ln()) / (n + 2.0 * M_EXP)).exp();
    let lambda = c_lambda * lc;
    let lh = 0.5 * lambda;
    let q = RHO_RATIO * e * lh;
    let sc = SOIL_CRITICAL_SLOPE;
    let sh = 2.0 * q / (K_H + (K_H * K_H + 4.0 * q * q / (sc * sc)).sqrt());
    let curvature = RHO_RATIO * e / K_H;
    let k2 = 2.0 * sh * sh / curvature;
    PointLaw {
        ks,
        e,
        lc,
        lambda,
        lh,
        sh,
        k2,
    }
}

/// ★ THE BED'S SLOPE at a point (§3.4 with S5): the stream law `k_s·Q^(−1/n)`, never steeper than the soil's runaway
/// slope `S_c` — the bench's form of the slope–area law's colluvial regime (S5; Stock & Dietrich 2003's own
/// debris-flow law is owed).
fn bed_slope(ks: f64, q_m3_yr: f64) -> f64 {
    (ks * q_m3_yr.powf(-1.0 / JOINT_N)).min(SOIL_CRITICAL_SLOPE)
}

/// The channel's half width at `Q` m³/yr.
fn half_width(q_m3_yr: f64) -> f64 {
    0.5 * WIDTH_A * (q_m3_yr / YEAR_S).powf(WIDTH_B)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// c_λ — THE LAW'S OWN NON-DIMENSIONAL FORM (§3.3), SOLVED ONCE.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// What the non-dimensional solve found.
struct CLambda {
    /// The valley spacing at the transect halfway between the main divide and the base level, in `l_c`.
    value: f64,
    /// The spacing at the other transects: (the distance from the base level as a share of the half width, spacing).
    by_row: Vec<(f64, f64)>,
    /// The mean `|∂z/∂t|` over the last step against the uplift (a steady state reads near nought).
    residual: f64,
    seconds: f64,
}

/// The valleys of a transect: local minima whose rise to the higher of the two lower-bounded maxima on each side
/// stands at least `thr` over them (the topographic prominence of a minimum). A side that reaches the transect's end
/// (non-periodic) with no lower point counts its maximum to the end. Only minima with `valid(k)` count.
fn count_valleys(p: &[f64], periodic: bool, thr: f64, valid: &dyn Fn(usize) -> bool) -> usize {
    let len = p.len();
    if len < 3 {
        return 0;
    }
    let at = |k: i64| -> Option<f64> {
        if periodic {
            Some(p[k.rem_euclid(len as i64) as usize])
        } else if k < 0 || k >= len as i64 {
            None
        } else {
            Some(p[k as usize])
        }
    };
    let mut count = 0;
    for k in 0..len as i64 {
        let z = p[k as usize];
        let (Some(l), Some(r)) = (at(k - 1), at(k + 1)) else {
            continue;
        };
        if !(z < l && z <= r && valid(k as usize)) {
            continue;
        }
        let side = |step: i64| -> f64 {
            let mut m = z;
            let mut i = k + step;
            let mut walked = 0usize;
            while let Some(v) = at(i) {
                if v < z || walked >= len {
                    break;
                }
                m = m.max(v);
                i += step;
                walked += 1;
            }
            m
        };
        if side(-1).min(side(1)) - z >= thr {
            count += 1;
        }
    }
    count
}

/// ★ `c_λ` FOR THE JOINT LAW (§3.3): a landscape evolution model in the law's own non-dimensional form — `K` = `D` =
/// `U` = 1, so `l_c` = 1 — with `E = K·A^m·S^n` (implicit, outlets first, the macro sweep's own form) and LINEAR
/// diffusion `D·∇²z` (the form Perron et al. 2009 and Theodoratos et al. 2018 solved), on a strip periodic along `x`
/// with the base level at both long edges, run to a steady state from a small random start. `c_λ` is the mean spacing
/// of the valleys crossing the transect halfway between the main divide and a base level, in units of `l_c`.
fn c_lambda(n_exp: f64, m_exp: f64, dx: f64, t_end: f64) -> CLambda {
    let started = Instant::now();
    let (lx, ly) = (96.0f64, 48.0f64);
    let nx = (lx / dx) as usize;
    let ny = (ly / dx) as usize + 1;
    let cells = nx * ny;
    let mut z = vec![0.0f64; cells];
    for y in 1..ny - 1 {
        for x in 0..nx {
            z[y * nx + x] = 1e-3 * h01(99, x as u64, y as u64, 0);
        }
    }
    let dt = 0.2 * dx * dx;
    let steps = (t_end / dt) as usize;
    let mut recv = vec![NONE; cells];
    let mut dist = vec![dx; cells];
    let mut area = vec![0.0f64; cells];
    let mut order: Vec<usize> = Vec::with_capacity(cells);
    let mut residual = 0.0;
    let neighbour = |c: usize, dxx: i64, dyy: i64| -> Option<usize> {
        let (x, y) = ((c % nx) as i64, (c / nx) as i64);
        let ny2 = y + dyy;
        if ny2 < 0 || ny2 >= ny as i64 {
            return None;
        }
        let nx2 = (x + dxx).rem_euclid(nx as i64);
        Some(ny2 as usize * nx + nx2 as usize)
    };
    for step in 0..steps {
        // The routing: a priority flood from the base rows, the steepest descent on the filled surface.
        let key = |h: f64| -> i64 { (h * 1.0e9).round() as i64 };
        let mut zf = z.clone();
        let mut seen = vec![false; cells];
        let mut parent = vec![NONE; cells];
        order.clear();
        let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
        for x in 0..nx {
            for y in [0, ny - 1] {
                let c = y * nx + x;
                seen[c] = true;
                heap.push(Reverse((key(zf[c]), c)));
            }
        }
        while let Some(Reverse((_, c))) = heap.pop() {
            order.push(c);
            for (ddx, ddy) in NB {
                if let Some(m) = neighbour(c, ddx, ddy)
                    && !seen[m]
                {
                    seen[m] = true;
                    zf[m] = zf[m].max(zf[c]);
                    parent[m] = c;
                    heap.push(Reverse((key(zf[m]), m)));
                }
            }
        }
        for c in 0..cells {
            recv[c] = NONE;
            if parent[c] == NONE {
                continue;
            }
            let mut best = (0.0f64, NONE, dx);
            for (ddx, ddy) in NB {
                if let Some(m) = neighbour(c, ddx, ddy) {
                    let d = if ddx != 0 && ddy != 0 {
                        dx * std::f64::consts::SQRT_2
                    } else {
                        dx
                    };
                    let s = (zf[c] - zf[m]) / d;
                    if s > best.0 {
                        best = (s, m, d);
                    }
                }
            }
            if best.1 == NONE {
                let p = parent[c];
                let (px, py) = ((p % nx) as i64, (p / nx) as i64);
                let (cx, cy) = ((c % nx) as i64, (c / nx) as i64);
                let ddx = (px - cx).rem_euclid(nx as i64);
                let diag = ddx != 0 && py != cy;
                recv[c] = p;
                dist[c] = if diag {
                    dx * std::f64::consts::SQRT_2
                } else {
                    dx
                };
            } else {
                recv[c] = best.1;
                dist[c] = best.2;
            }
        }
        area.iter_mut().for_each(|a| *a = dx * dx);
        for &c in order.iter().rev() {
            if recv[c] != NONE {
                area[recv[c]] += area[c];
            }
        }
        let before = z.clone();
        // The uplift and the stream law, implicit, outlets first.
        for &c in &order {
            let y = c / nx;
            if y == 0 || y == ny - 1 {
                continue;
            }
            let r = recv[c];
            let x0 = z[c] + dt - z[r];
            if x0 <= 0.0 {
                z[c] += dt;
                continue;
            }
            let cc = dt * area[c].powf(m_exp) / dist[c].powf(n_exp);
            let mut x = x0;
            for _ in 0..30 {
                let f = x - x0 + cc * x.powf(n_exp);
                let fp = 1.0 + n_exp * cc * x.powf(n_exp - 1.0);
                let nxt = x - f / fp;
                if (nxt - x).abs() < 1e-12 {
                    x = nxt;
                    break;
                }
                x = nxt.max(0.0);
            }
            z[c] = z[r] + x;
        }
        // The linear diffusion, explicit.
        let snap = z.clone();
        for y in 1..ny - 1 {
            for x in 0..nx {
                let c = y * nx + x;
                let l = snap[y * nx + (x + nx - 1) % nx];
                let r = snap[y * nx + (x + 1) % nx];
                let u = snap[(y - 1) * nx + x];
                let d = snap[(y + 1) * nx + x];
                z[c] += dt * (l + r + u + d - 4.0 * snap[c]) / (dx * dx);
            }
        }
        if step + 1 == steps {
            residual = z
                .iter()
                .zip(&before)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / cells as f64
                / dt;
        }
    }
    let zmax = z.iter().copied().fold(0.0f64, f64::max);
    let thr = 0.02 * zmax;
    let mut by_row = Vec::new();
    let half = (ny - 1) / 2;
    let mut value = 0.0;
    for share in [0.25f64, 0.5, 0.75] {
        let off = ((half as f64) * share).round() as usize;
        let mut total = 0usize;
        for y in [off, ny - 1 - off] {
            let row: Vec<f64> = (0..nx).map(|x| z[y * nx + x]).collect();
            total += count_valleys(&row, true, thr, &|_| true);
        }
        let spacing = if total > 0 {
            2.0 * lx / total as f64
        } else {
            f64::NAN
        };
        by_row.push((share, spacing));
        if (share - 0.5).abs() < 1e-9 {
            value = spacing;
        }
    }
    CLambda {
        value,
        by_row,
        residual,
        seconds: started.elapsed().as_secs_f64(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE DRAINAGE LADDER (§3).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// No segment.
const NO_SEG: u32 = u32::MAX;
/// The ceiling blocks' side, in raster cells.
const BLOCK: usize = 8;
/// ★ The bench's compute bound on one segment's reach, metres (one node). A segment that reaches it is COUNTED and
/// printed: it is a cost bound of this bench, never a law of the world.
const R_MAX: f64 = NODE_M;

/// What a stream is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// A tree link: from the node's anchor to its receiver's (level 0).
    Trunk { from: usize, to: usize },
    /// A rim outlet's run from its anchor out of the tile (level 0).
    Outlet { node: usize },
    /// A routed source of a finer level.
    Fine,
}

/// One stream of the ladder.
struct Stream {
    level: u8,
    kind: Kind,
    /// The path, upstream first; the last point is the join on the parent's path (or the receiver's anchor).
    pts: Vec<[f64; 2]>,
    /// The parent and where on it this stream ends: (stream, segment, t ∈ [0, 1]).
    parent: Option<(usize, usize, f64)>,
    /// `log k_s` at each point (the node field by Catmull-Rom).
    lks: Vec<f64>,
    /// The area at each point, m².
    area: Vec<f64>,
    /// The bed at each point, metres.
    bed: Vec<f64>,
    /// The source cell's area, m² (a fine stream).
    src_area: f64,
}

/// One segment as the splat reads it.
#[derive(Clone, Copy)]
struct Seg {
    a: [f32; 2],
    b: [f32; 2],
    bed: [f32; 2],
    sh: f32,
    hw: f32,
    lh: f32,
    k2: f32,
    r: f32,
    stream: u32,
    local: u32,
}

/// One raster cell's candidates.
#[derive(Clone, Copy)]
struct Cell {
    /// The lowest profile value and its segment.
    best: f32,
    best_seg: u32,
    /// The lowest from another stream (S2's divide).
    sec: f32,
    sec_seg: u32,
    /// The lowest BOUNDED profile (the side flat past `L_H`): the datum's closure reads it.
    bnd: f32,
    /// The nearest floor's segment within the floor's reach (the routes' join test), and its distance.
    floor_seg: u32,
    floor_d: f32,
}

impl Cell {
    const EMPTY: Cell = Cell {
        best: f32::INFINITY,
        best_seg: NO_SEG,
        sec: f32::INFINITY,
        sec_seg: NO_SEG,
        bnd: f32::INFINITY,
        floor_seg: NO_SEG,
        floor_d: f32::INFINITY,
    };
}

/// One level's routing tally.
#[derive(Default, Clone, Debug)]
struct LevelStats {
    cell_m: f64,
    sources: usize,
    skipped_lambda: usize,
    skipped_sea: usize,
    at_floor: usize,
    joined: usize,
    joined_same: usize,
    stuck: usize,
    left: usize,
    steps: usize,
}

/// The ladder over one tile.
struct Ladder<'a> {
    f: &'a Facts,
    c_lambda: f64,
    cw: f64,
    w: usize,
    threads: usize,
    /// The finest level.
    levels: usize,
    /// The start `S` (Catmull-Rom of `Z`) and the lift `L`, per raster cell.
    start: Vec<f32>,
    lift: Vec<f32>,
    lift_nodes: Vec<f64>,
    /// The mean hold `T` (the exact remainder), per raster cell and per node.
    hold: Vec<f32>,
    hold_nodes: Vec<f64>,
    /// The datum `M` per node and the anchors' beds.
    m_nodes: Vec<f64>,
    anchor_bed: Vec<f64>,
    streams: Vec<Stream>,
    /// The node's outgoing level-0 stream.
    out_stream: Vec<usize>,
    cells: Vec<Cell>,
    segs: Vec<Seg>,
    /// The ground the routes read: `min(S + L, best)` of the levels splatted so far.
    comp: Vec<f32>,
    /// The profile envelope `P` (S2's smooth minimum applied) and the drawn ground `h`.
    pen: Vec<f32>,
    h: Vec<f32>,
    stats: Vec<LevelStats>,
    /// Segments whose reach met [`R_MAX`].
    capped: usize,
}

impl<'a> Ladder<'a> {
    fn new(f: &'a Facts, c_lambda: f64, cw: f64, threads: usize) -> Ladder<'a> {
        let w = (f.n as f64 * NODE_M / cw).round() as usize;
        // ★ THE START (a), MEAN-KEEPING: the Catmull-Rom of the node means misses a node's own mean by tens of
        // metres where a plain meets a range (the design's start has a zero-mean fine relief, Q7). So the start is
        // the Catmull-Rom of node values corrected until every node's mean is its `Z` (the same hold, on the start
        // alone, before any carve) — the first form let the lift go negative and dig ovals under the sea level.
        let n = f.n;
        let per = (NODE_M / cw).round() as usize;
        let mut vals = f.z.clone();
        let mut start = node_cr_raster(&vals, n, w, cw);
        for _ in 0..30 {
            let mut sum = vec![0.0f64; n * n];
            for y in 0..w {
                for x in 0..w {
                    sum[(y / per).min(n - 1) * n + (x / per).min(n - 1)] +=
                        f64::from(start[y * w + x]);
                }
            }
            let mut worst = 0.0f64;
            for k in 0..n * n {
                let r = f.z[k] - sum[k] / (per * per) as f64;
                worst = worst.max(r.abs());
                vals[k] += r;
            }
            start = node_cr_raster(&vals, n, w, cw);
            if worst < 0.01 {
                break;
            }
        }
        let start_fix = vals
            .iter()
            .zip(&f.z)
            .map(|(v, z)| (v - z).abs())
            .fold(0.0f64, f64::max);
        println!(
            "  the start (a): the Catmull-Rom of node values held to each node's mean Z; the largest correction {start_fix:.1} m"
        );
        // The finest level: the law's smallest spacing over the land, never under two raster cells.
        let lambda_min = (0..f.n * f.n)
            .filter(|&k| !f.sea[k])
            .map(|k| point_law(f.log_ks[k], f.rain_m, c_lambda).lambda)
            .fold(f64::INFINITY, f64::min);
        let mut levels = 0;
        while levels < 12 {
            let s = NODE_M / f64::from(1u32 << (levels + 1));
            if s < lambda_min || s < 2.0 * cw {
                break;
            }
            levels += 1;
        }
        Ladder {
            f,
            c_lambda,
            cw,
            w,
            threads,
            levels,
            start,
            lift: vec![0.0; w * w],
            lift_nodes: vec![0.0; f.n * f.n],
            hold: vec![0.0; w * w],
            hold_nodes: vec![0.0; f.n * f.n],
            m_nodes: vec![0.0; f.n * f.n],
            anchor_bed: vec![0.0; f.n * f.n],
            streams: Vec::new(),
            out_stream: vec![NONE; f.n * f.n],
            cells: vec![Cell::EMPTY; w * w],
            segs: Vec::new(),
            comp: vec![0.0; w * w],
            pen: vec![f32::INFINITY; w * w],
            h: vec![0.0; w * w],
            stats: Vec::new(),
            capped: 0,
        }
    }

    fn lks_at(&self, p: [f64; 2]) -> f64 {
        node_cr(&self.f.log_ks, self.f.n, p[0], p[1])
    }

    fn law_at(&self, lks: f64) -> PointLaw {
        point_law(lks, self.f.rain_m, self.c_lambda)
    }

    /// ★ THE ANCHORS' BEDS, outlets first: the sea's at the sea level (the base level); a rim outlet's `Z − M`; every
    /// other node `max(Z − M, its receiver's bed)` — a river never cuts under its outlet (the bench's form of the
    /// design's `m = min(M_n, M_r)`, which can step UP at a node whose receiver's `M` is smaller).
    fn set_anchor_beds(&mut self) {
        let f = self.f;
        for &c in &f.order {
            self.anchor_bed[c] = if f.sea[c] {
                0.0
            } else if f.recv[c] == NONE {
                f.z[c] - self.m_nodes[c]
            } else {
                (f.z[c] - self.m_nodes[c]).max(self.anchor_bed[f.recv[c]])
            };
        }
    }

    /// ★ LEVEL 0: a trunk on every land link (a Catmull-Rom spline through the anchors of the main donor, the node,
    /// the receiver and the receiver's receiver) and an outlet run from every rim outlet out of the tile.
    fn build_trunks(&mut self) {
        let f = self.f;
        let n = f.n;
        self.streams.clear();
        self.out_stream = vec![NONE; n * n];
        let mut main_donor = vec![NONE; n * n];
        for k in 0..n * n {
            let r = f.recv[k];
            if r != NONE && (main_donor[r] == NONE || f.area[k] > f.area[main_donor[r]]) {
                main_donor[r] = k;
            }
        }
        let side = n as f64 * NODE_M;
        for (k, &is_sea) in f.sea.iter().enumerate() {
            if is_sea {
                continue;
            }
            let a = f.anchor[k];
            let r = f.recv[k];
            let (pts, kind) = if r == NONE {
                // ★ A rim outlet's river leaves the tile straight out through its nearest edge — along the node's own
                // fall line where that points out of the tile, else along the edge's outward normal. (A first form
                // routed it down the start: a crest node's river then ran 30 km across the tile in a deepening trench.)
                let cw = self.cw;
                let d = [a[0], side - a[0], a[1], side - a[1]];
                let e = (0..4).min_by(|&i, &j| d[i].total_cmp(&d[j])).unwrap_or(0);
                let normal = match e {
                    0 => [-1.0, 0.0],
                    1 => [1.0, 0.0],
                    2 => [0.0, -1.0],
                    _ => [0.0, 1.0],
                };
                let fall = f.fall[k];
                let dir = if fall[0] * normal[0] + fall[1] * normal[1] > 0.0 {
                    fall
                } else {
                    normal
                };
                let inside = |q: [f64; 2]| q[0] >= 0.0 && q[1] >= 0.0 && q[0] < side && q[1] < side;
                let mut pts = vec![a];
                let mut q = a;
                while inside(q) {
                    q = [q[0] + cw * dir[0], q[1] + cw * dir[1]];
                    pts.push(q);
                }
                (pts, Kind::Outlet { node: k })
            } else {
                let b = f.anchor[r];
                let p0 = if main_donor[k] != NONE {
                    f.anchor[main_donor[k]]
                } else {
                    [2.0 * a[0] - b[0], 2.0 * a[1] - b[1]]
                };
                let p3 = if f.recv[r] != NONE {
                    f.anchor[f.recv[r]]
                } else {
                    [2.0 * b[0] - a[0], 2.0 * b[1] - a[1]]
                };
                let chord = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
                let count = (2.0 * chord / self.cw).ceil().max(2.0) as usize;
                let pts: Vec<[f64; 2]> = (0..=count)
                    .map(|i| {
                        let t = i as f64 / count as f64;
                        [
                            catmull_rom([p0[0], a[0], b[0], p3[0]], t),
                            catmull_rom([p0[1], a[1], b[1], p3[1]], t),
                        ]
                    })
                    .collect();
                (pts, Kind::Trunk { from: k, to: r })
            };
            let lks: Vec<f64> = pts.iter().map(|&p| self.lks_at(p)).collect();
            let len = pts.len();
            self.out_stream[k] = self.streams.len();
            self.streams.push(Stream {
                level: 0,
                kind,
                pts,
                parent: None,
                lks,
                area: vec![0.0; len],
                bed: vec![0.0; len],
                src_area: 0.0,
            });
        }
        // The parents: a trunk ends at its receiver's anchor, the first point of the receiver's own level-0 stream.
        for s in 0..self.streams.len() {
            if let Kind::Trunk { to, .. } = self.streams[s].kind {
                let o = self.out_stream[to];
                self.streams[s].parent = (o != NONE).then_some((o, 0, 0.0));
            }
        }
    }

    /// ★ THE BED of stream `s` (§3.4). A trunk: the χ shape between its two anchors' beds, its area linear from the
    /// node's tree area to the receiver's. An outlet run: from its anchor down at the law's slope. A fine stream: from
    /// its join on the parent upward, `b' = min(k_s·Q^(−1/n), S_c)`.
    fn bed_stream(&mut self, s: usize) {
        let f = self.f;
        let rain = f.rain_m;
        let kind = self.streams[s].kind;
        let parent_bed = self.streams[s].parent.map(|(p, seg, t)| {
            let pb = &self.streams[p].bed;
            pb[seg] + t * (pb[(seg + 1).min(pb.len() - 1)] - pb[seg])
        });
        let st = &mut self.streams[s];
        let len = st.pts.len();
        let dist = |st: &Stream, k: usize| -> f64 {
            let (a, b) = (st.pts[k], st.pts[k + 1]);
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
        };
        match kind {
            Kind::Trunk { from, to } => {
                let (b0, b1) = (self.anchor_bed[from], self.anchor_bed[to]);
                let (a0, a1) = (f.area[from], f.area[to]);
                let mut arc = vec![0.0f64; len];
                for k in 1..len {
                    arc[k] = arc[k - 1] + dist(st, k - 1);
                }
                let total = arc[len - 1].max(f64::MIN_POSITIVE);
                let mut chi = vec![0.0f64; len];
                for (slot, &at) in st.area.iter_mut().zip(&arc) {
                    *slot = a0 + (a1 - a0) * at / total;
                }
                for k in 1..len {
                    let q = rain * 0.5 * (st.area[k - 1] + st.area[k]);
                    chi[k] = chi[k - 1] + q.powf(-1.0 / JOINT_N) * (arc[k] - arc[k - 1]);
                }
                let chi_total = chi[len - 1].max(f64::MIN_POSITIVE);
                for (slot, &c) in st.bed.iter_mut().zip(&chi) {
                    *slot = b0 - (b0 - b1) * c / chi_total;
                }
            }
            Kind::Outlet { node } => {
                let a = f.area[node];
                st.bed[0] = self.anchor_bed[node];
                st.area[0] = a;
                for k in 1..len {
                    st.area[k] = a;
                    let slope = bed_slope(st.lks[k - 1].exp(), rain * a);
                    st.bed[k] = st.bed[k - 1] - slope * dist(st, k - 1);
                }
            }
            Kind::Fine => {
                st.bed[len - 1] = parent_bed.unwrap_or(0.0);
                for k in (0..len - 1).rev() {
                    let slope = bed_slope(st.lks[k].exp(), rain * st.area[k]);
                    st.bed[k] = st.bed[k + 1] + slope * dist(st, k);
                }
            }
        }
    }

    /// The expected area along a fine stream (phase A): the source cell, plus a strip one cell wide along its run.
    fn expected_area(st: &mut Stream, cell_m: f64) {
        let mut arc = 0.0;
        for k in 0..st.pts.len() {
            if k > 0 {
                let (a, b) = (st.pts[k - 1], st.pts[k]);
                arc += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            }
            st.area[k] = st.src_area + cell_m * arc;
        }
    }

    /// ★ THE AREAS FROM THE REAL TRIBUTARIES (§3.3, C3), finest level first: a fine stream's area at a point is its
    /// source cell plus every tributary that joins upstream of the point; its total goes to its parent if the parent
    /// is a fine stream (a trunk's area is the tree's, which already holds its whole basin).
    fn real_areas(&mut self) {
        let count = self.streams.len();
        let mut joins: Vec<Vec<(usize, f64)>> = vec![Vec::new(); count];
        let mut by_level: Vec<usize> = (0..count).collect();
        // Finest level first; inside a level the later routes first (a same-level parent is routed before its
        // tributaries).
        by_level.sort_by_key(|&s| (Reverse(self.streams[s].level), Reverse(s)));
        for s in by_level {
            if self.streams[s].kind != Kind::Fine {
                continue;
            }
            let mut add = std::mem::take(&mut joins[s]);
            add.sort_by_key(|j| j.0);
            let st = &mut self.streams[s];
            let mut acc = st.src_area;
            let mut next = 0;
            for k in 0..st.pts.len() {
                while next < add.len() && add[next].0 < k {
                    acc += add[next].1;
                    next += 1;
                }
                st.area[k] = acc;
            }
            let total = st.area[st.pts.len() - 1];
            if let Some((p, seg, _)) = st.parent
                && self.streams[p].kind == Kind::Fine
            {
                joins[p].push((seg, total));
            }
        }
    }

    /// Every bed, coarse to fine (a parent before its tributaries).
    fn bed_all(&mut self) {
        let mut by_level: Vec<usize> = (0..self.streams.len()).collect();
        // The outlet runs before the trunks are not needed (a trunk reads the anchors), but a fine stream reads its
        // parent: sort by level, the level-0 streams first.
        by_level.sort_by_key(|&s| self.streams[s].level);
        for s in by_level {
            self.bed_stream(s);
        }
    }
}

/// The maxima of a raster over square blocks: the segments' ceilings (§3.5's cull radius from the box's own bounds).
struct BlockMax {
    nb: usize,
    side_m: f64,
    vals: Vec<f32>,
    global: f32,
}

impl BlockMax {
    fn of(f: &[f32], w: usize, cw: f64) -> BlockMax {
        let nb = w.div_ceil(BLOCK);
        let mut vals = vec![f32::NEG_INFINITY; nb * nb];
        for y in 0..w {
            for x in 0..w {
                let k = (y / BLOCK) * nb + x / BLOCK;
                vals[k] = vals[k].max(f[y * w + x]);
            }
        }
        let global = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        BlockMax {
            nb,
            side_m: BLOCK as f64 * cw,
            vals,
            global,
        }
    }

    /// The maximum over the blocks touching the box `[x0, x1] × [y0, y1]` metres.
    fn max_in(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        let clampb =
            |v: f64| ((v / self.side_m).floor() as i64).clamp(0, self.nb as i64 - 1) as usize;
        let (i0, i1, j0, j1) = (clampb(x0), clampb(x1), clampb(y0), clampb(y1));
        let mut m = f32::NEG_INFINITY;
        for j in j0..=j1 {
            for i in i0..=i1 {
                m = m.max(self.vals[j * self.nb + i]);
            }
        }
        f64::from(m)
    }
}

/// The candidates of segments `range` into a band of rows starting at `y0` (one thread's share of the raster).
fn band_splat(
    y0: usize,
    cells: &mut [Cell],
    segs: &[Seg],
    range: std::ops::Range<usize>,
    w: usize,
    cw: f64,
) {
    let rows = cells.len() / w;
    let y1 = y0 + rows;
    for si in range {
        let s = segs[si];
        let r = f64::from(s.r);
        let (ax, ay, bx, by) = (
            f64::from(s.a[0]),
            f64::from(s.a[1]),
            f64::from(s.b[0]),
            f64::from(s.b[1]),
        );
        let cell_range = |lo: f64, hi: f64, top: usize| -> Option<(usize, usize)> {
            let c0 = (lo / cw - 0.5).ceil().max(0.0);
            let c1 = (hi / cw - 0.5).floor();
            if c1 < 0.0 || c0 > c1 {
                return None;
            }
            Some((c0 as usize, (c1 as usize).min(top - 1)))
        };
        let Some((cy0, cy1)) = cell_range(ay.min(by) - r, ay.max(by) + r, w) else {
            continue;
        };
        let (cy0, cy1) = (cy0.max(y0), cy1.min(y1 - 1));
        if cy0 > cy1 {
            continue;
        }
        let Some((cx0, cx1)) = cell_range(ax.min(bx) - r, ax.max(bx) + r, w) else {
            continue;
        };
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let (hw, sh, lh) = (f64::from(s.hw), f64::from(s.sh), f64::from(s.lh));
        let (b0, b1) = (f64::from(s.bed[0]), f64::from(s.bed[1]));
        let fr = hw.max(0.75 * cw);
        for cy in cy0..=cy1 {
            let py = (cy as f64 + 0.5) * cw;
            let row = (cy - y0) * w;
            for cx in cx0..=cx1 {
                let px = (cx as f64 + 0.5) * cw;
                let t = if l2 > 0.0 {
                    (((px - ax) * dx + (py - ay) * dy) / l2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let (qx, qy) = (ax + t * dx, ay + t * dy);
                let d = ((px - qx) * (px - qx) + (py - qy) * (py - qy)).sqrt();
                if d > r {
                    continue;
                }
                let bed = b0 + t * (b1 - b0);
                let e = (d - hw).max(0.0);
                let v = (bed + sh * e) as f32;
                let vb = (bed + sh * e.min(lh)) as f32;
                let c = &mut cells[row + cx];
                if vb < c.bnd {
                    c.bnd = vb;
                }
                if d <= fr && (d as f32) < c.floor_d {
                    c.floor_d = d as f32;
                    c.floor_seg = si as u32;
                }
                let same_as_best =
                    c.best_seg != NO_SEG && segs[c.best_seg as usize].stream == s.stream;
                if v < c.best {
                    if c.best_seg != NO_SEG && !same_as_best {
                        c.sec = c.best;
                        c.sec_seg = c.best_seg;
                    }
                    c.best = v;
                    c.best_seg = si as u32;
                } else if v < c.sec && !same_as_best {
                    c.sec = v;
                    c.sec_seg = si as u32;
                }
            }
        }
    }
}

/// The point of segment `s` nearest `p`, and the distance.
fn nearest_on(s: &Seg, p: [f64; 2]) -> ([f64; 2], f64, f64) {
    let (ax, ay, bx, by) = (
        f64::from(s.a[0]),
        f64::from(s.a[1]),
        f64::from(s.b[0]),
        f64::from(s.b[1]),
    );
    let (dx, dy) = (bx - ax, by - ay);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((p[0] - ax) * dx + (p[1] - ay) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = [ax + t * dx, ay + t * dy];
    (q, t, ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt())
}

/// How one route ended.
enum Outcome {
    /// It reached a floor: its path (the last point on the parent) and the parent (stream, segment, `t`).
    Joined(Vec<[f64; 2]>, usize, usize, f64),
    /// The source stood on a floor already.
    AtFloor,
    /// No descent left (a closed hollow of the coarser ground), or the step bound met.
    Stuck(usize),
    /// It left the tile.
    Left(usize),
}

/// A same-level floor mark: no stream.
const NO_MARK: u64 = u64::MAX;

/// ★ THE ROUTE (§3.2): from `p0` DOWN the coarser ground `comp` by the steepest descent of its bilinear gradient,
/// `step` metres a step, until a coarser floor (the `floor_seg` mark) or — the bench's addition, see `route_level` —
/// the floor of a stream of the same level routed before it (the `same` mark), at most `max_steps`.
#[allow(clippy::too_many_arguments)]
fn route_one(
    comp: &[f32],
    cells: &[Cell],
    segs: &[Seg],
    same: &[u64],
    streams: &[Stream],
    w: usize,
    cw: f64,
    p0: [f64; 2],
    step: f64,
    max_steps: usize,
) -> Outcome {
    let side = w as f64 * cw;
    let cell_of = |p: [f64; 2]| -> usize {
        let x = ((p[0] / cw) as usize).min(w - 1);
        let y = ((p[1] / cw) as usize).min(w - 1);
        y * w + x
    };
    let at = |x: i64, y: i64| {
        f64::from(comp[(y.clamp(0, w as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize])
    };
    let sample = |p: [f64; 2]| -> (f64, f64, f64) {
        let (u, v) = (p[0] / cw - 0.5, p[1] / cw - 0.5);
        let (i0, j0) = (u.floor() as i64, v.floor() as i64);
        let (fu, fv) = (u - u.floor(), v - v.floor());
        let (mut gx, mut gy, mut h) = (0.0, 0.0, 0.0);
        for (dj, wy) in [(0i64, 1.0 - fv), (1, fv)] {
            for (di, wx) in [(0i64, 1.0 - fu), (1, fu)] {
                let (i, j) = (i0 + di, j0 + dj);
                let k = wx * wy;
                gx += k * (at(i + 1, j) - at(i - 1, j)) / (2.0 * cw);
                gy += k * (at(i, j + 1) - at(i, j - 1)) / (2.0 * cw);
                h += k * at(i, j);
            }
        }
        (gx, gy, h)
    };
    // The floor under `p`, if any: the coarser first, then the same level's.
    let floor_at = |p: [f64; 2]| -> Option<(usize, usize, f64, [f64; 2])> {
        let i = cell_of(p);
        let c = cells[i];
        if c.floor_seg != NO_SEG {
            let s = &segs[c.floor_seg as usize];
            let (q, t, _) = nearest_on(s, p);
            return Some((s.stream as usize, s.local as usize, t, q));
        }
        if same[i] != NO_MARK {
            let (si, k) = ((same[i] >> 32) as usize, (same[i] & 0xFFFF_FFFF) as usize);
            let (a, b) = (streams[si].pts[k], streams[si].pts[k + 1]);
            let seg = Seg {
                a: [a[0] as f32, a[1] as f32],
                b: [b[0] as f32, b[1] as f32],
                bed: [0.0; 2],
                sh: 0.0,
                hw: 0.0,
                lh: 0.0,
                k2: 0.0,
                r: 0.0,
                stream: si as u32,
                local: k as u32,
            };
            let (q, t, _) = nearest_on(&seg, p);
            return Some((si, k, t, q));
        }
        None
    };
    let mut p = p0;
    let mut pts = vec![p];
    if floor_at(p).is_some() {
        return Outcome::AtFloor;
    }
    let mut lowest = sample(p).2;
    let mut since = 0usize;
    for steps in 0..max_steps {
        if let Some((si, k, t, q)) = floor_at(p) {
            pts.push(q);
            return Outcome::Joined(pts, si, k, t);
        }
        let (gx, gy, _) = sample(p);
        let g = (gx * gx + gy * gy).sqrt();
        if g < 1e-12 {
            return Outcome::Stuck(steps);
        }
        let next = [p[0] - step * gx / g, p[1] - step * gy / g];
        if next[0] < 0.0 || next[1] < 0.0 || next[0] >= side || next[1] >= side {
            return Outcome::Left(steps);
        }
        p = next;
        pts.push(p);
        let h = sample(p).2;
        if h < lowest - 1e-6 {
            lowest = h;
            since = 0;
        } else {
            since += 1;
            if since > 6 {
                return Outcome::Stuck(steps);
            }
        }
    }
    Outcome::Stuck(max_steps)
}

/// Quadratic smooth minimum of width `k` (`min − h²·k/4`, `h = max(k − |a − b|, 0)/k`): a parabola of curvature
/// `−2S²/k` where slopes `±S` meet (S2).
fn smin(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

impl Ladder<'_> {
    /// The segments of `level`'s streams, each with its law and its reach from the ground's ceilings.
    fn segs_of_level(&mut self, level: u8) -> std::ops::Range<usize> {
        let first = self.segs.len();
        let bm = BlockMax::of(&self.comp, self.w, self.cw);
        let rain = self.f.rain_m;
        let mut out = Vec::new();
        let mut capped = 0usize;
        for (si, st) in self.streams.iter().enumerate() {
            if st.level != level {
                continue;
            }
            for k in 0..st.pts.len() - 1 {
                let (a, b) = (st.pts[k], st.pts[k + 1]);
                let law = self.law_at(st.lks[k]);
                let hw = half_width(rain * st.area[k]);
                let bmin = st.bed[k].min(st.bed[k + 1]);
                let (x0, x1) = (a[0].min(b[0]), a[0].max(b[0]));
                let (y0, y1) = (a[1].min(b[1]), a[1].max(b[1]));
                let mut margin = bm.side_m;
                let mut r;
                loop {
                    let ceil = if margin > 16.0 * bm.side_m {
                        f64::from(bm.global)
                    } else {
                        bm.max_in(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
                    };
                    r = hw + (ceil + law.k2 - bmin).max(0.0) / law.sh.max(1e-12);
                    if r <= margin || margin > 16.0 * bm.side_m {
                        break;
                    }
                    margin *= 2.0;
                }
                r = r.max(hw + law.lh);
                if r >= R_MAX {
                    r = R_MAX;
                    capped += 1;
                }
                out.push(Seg {
                    a: [a[0] as f32, a[1] as f32],
                    b: [b[0] as f32, b[1] as f32],
                    bed: [st.bed[k] as f32, st.bed[k + 1] as f32],
                    sh: law.sh as f32,
                    hw: hw as f32,
                    lh: law.lh as f32,
                    k2: law.k2 as f32,
                    r: r as f32,
                    stream: si as u32,
                    local: k as u32,
                });
            }
        }
        self.capped += capped;
        self.segs.extend(out);
        first..self.segs.len()
    }

    /// Splat `range` over the raster, one band of rows a thread.
    fn splat_range(&mut self, range: std::ops::Range<usize>) {
        let (w, cw) = (self.w, self.cw);
        let rows_per = w.div_ceil(self.threads);
        let segs = &self.segs;
        let cells = &mut self.cells;
        std::thread::scope(|sc| {
            for (band, chunk) in cells.chunks_mut(rows_per * w).enumerate() {
                let range = range.clone();
                sc.spawn(move || band_splat(band * rows_per, chunk, segs, range, w, cw));
            }
        });
    }

    /// The coarser ground's shape `min(S + L, best)` (the ceilings read it; the routes read it plus `T`).
    fn update_comp(&mut self) {
        for i in 0..self.w * self.w {
            self.comp[i] = (self.start[i] + self.lift[i]).min(self.cells[i].best);
        }
    }

    /// ★ ONE LEVEL'S ROUTES (§3.2): one source per cell of `NODE_M / 2^j`, uniform over the cell, where the law's
    /// spacing allows the level (`s_j ≥ λ`, §3.3) and the node is land; each routed down the coarser ground.
    fn route_level(&mut self, j: usize, step_factor: f64) {
        let s_j = NODE_M / f64::from(1u32 << j);
        let s_prev = 2.0 * s_j;
        let side = self.f.n as f64 * NODE_M;
        let nc = (side / s_j).round() as usize;
        let mut stats = LevelStats {
            cell_m: s_j,
            ..LevelStats::default()
        };
        let mut sources: Vec<[f64; 2]> = Vec::new();
        for cj in 0..nc {
            for ci in 0..nc {
                stats.sources += 1;
                let p = [
                    (ci as f64 + h01(j as u64, ci as u64, cj as u64, 0)) * s_j,
                    (cj as f64 + h01(j as u64, ci as u64, cj as u64, 1)) * s_j,
                ];
                if self.f.sea[self.f.node_at(p[0], p[1])] {
                    stats.skipped_sea += 1;
                    continue;
                }
                if self.law_at(self.lks_at(p)).lambda > s_j {
                    stats.skipped_lambda += 1;
                    continue;
                }
                sources.push(p);
            }
        }
        let step = self.cw * step_factor;
        let max_steps = (4.0 * s_prev / step).ceil() as usize;
        let (w, cw) = (self.w, self.cw);
        // ★ THE BENCH'S ADDITION (the design joins a route only to a COARSER stream, §3.2): on the first run every
        // pair of same-level routes that drained one slope ran side by side to the coarser floor, a brush of parallel
        // streams. So the routes run one at a time, the HIGHEST source first, and a route also ends where it reaches
        // the floor of a same-level stream routed before it — a confluence. Deterministic (the order is the sources'
        // heights, ties by the cell), but a route now depends on the higher routes near it (a wider window in the
        // build), which the report states.
        let heights: Vec<f64> = sources
            .iter()
            .map(|p| {
                let x = ((p[0] / cw) as usize).min(w - 1);
                let y = ((p[1] / cw) as usize).min(w - 1);
                f64::from(self.comp[y * w + x])
            })
            .collect();
        let mut order: Vec<usize> = (0..sources.len()).collect();
        order.sort_by(|&a, &b| heights[b].total_cmp(&heights[a]).then(a.cmp(&b)));
        let mut same = vec![NO_MARK; w * w];
        let first_new = self.streams.len();
        let ground: Vec<f32> = self
            .comp
            .iter()
            .zip(&self.hold)
            .map(|(c, t)| c + t)
            .collect();
        for idx in order {
            let o = route_one(
                &ground,
                &self.cells,
                &self.segs,
                &same,
                &self.streams,
                w,
                cw,
                sources[idx],
                step,
                max_steps,
            );
            match o {
                Outcome::Joined(pts, ps, pk, t) => {
                    stats.joined += 1;
                    if self.streams[ps].level == j as u8 {
                        stats.joined_same += 1;
                    }
                    stats.steps += pts.len();
                    let lks: Vec<f64> = pts.iter().map(|&p| self.lks_at(p)).collect();
                    let len = pts.len();
                    let mut st = Stream {
                        level: j as u8,
                        kind: Kind::Fine,
                        pts,
                        parent: Some((ps, pk, t)),
                        lks,
                        area: vec![0.0; len],
                        bed: vec![0.0; len],
                        src_area: s_j * s_j,
                    };
                    Self::expected_area(&mut st, s_j);
                    // Its floor, for the same-level routes after it.
                    let si = self.streams.len();
                    for k in 0..len - 1 {
                        let q = st.pts[k];
                        let fr = half_width(self.f.rain_m * st.area[k]).max(cw);
                        let (x0, x1) = (
                            ((q[0] - fr) / cw).floor().max(0.0) as usize,
                            ((q[0] + fr) / cw).ceil() as usize,
                        );
                        let (y0, y1) = (
                            ((q[1] - fr) / cw).floor().max(0.0) as usize,
                            ((q[1] + fr) / cw).ceil() as usize,
                        );
                        for y in y0..=y1.min(w - 1) {
                            for x in x0..=x1.min(w - 1) {
                                let c = [(x as f64 + 0.5) * cw, (y as f64 + 0.5) * cw];
                                let d = ((c[0] - q[0]).powi(2) + (c[1] - q[1]).powi(2)).sqrt();
                                if d <= fr && same[y * w + x] == NO_MARK {
                                    same[y * w + x] = ((si as u64) << 32) | k as u64;
                                }
                            }
                        }
                    }
                    self.streams.push(st);
                }
                Outcome::AtFloor => stats.at_floor += 1,
                Outcome::Stuck(n) => {
                    stats.stuck += 1;
                    stats.steps += n;
                }
                Outcome::Left(n) => {
                    stats.left += 1;
                    stats.steps += n;
                }
            }
        }
        for s in first_new..self.streams.len() {
            self.bed_stream(s);
        }
        self.stats.push(stats);
    }

    /// ★ PHASE A: the trunks, then each level routed on the ground the coarser ones made (beds on the expected areas).
    fn build_routes(&mut self, step_factor: f64) {
        self.build_trunks();
        for s in 0..self.streams.len() {
            self.bed_stream(s);
        }
        self.cells.fill(Cell::EMPTY);
        self.segs.clear();
        self.capped = 0;
        self.stats.clear();
        for i in 0..self.w * self.w {
            self.comp[i] = self.start[i] + self.lift[i];
        }
        let r = self.segs_of_level(0);
        self.splat_range(r);
        self.update_comp();
        for j in 1..=self.levels {
            self.route_level(j, step_factor);
            let r = self.segs_of_level(j as u8);
            self.splat_range(r);
            self.update_comp();
        }
    }

    /// Every level splatted again, coarse first (each level's reach from the ground the coarser ones left), then `P`.
    fn splat_all(&mut self) {
        self.cells.fill(Cell::EMPTY);
        self.segs.clear();
        self.capped = 0;
        for i in 0..self.w * self.w {
            self.comp[i] = self.start[i] + self.lift[i];
        }
        for level in 0..=self.levels as u8 {
            let r = self.segs_of_level(level);
            self.splat_range(r);
            self.update_comp();
        }
        self.compose_pen();
    }

    /// ★ `P` (§2.0): the lowest profile, rounded by S2's smooth minimum with the second lowest of another stream,
    /// weighted `(1 − u_a·u_b)/2` (one at a divide, near nought at a join).
    fn compose_pen(&mut self) {
        let (w, cw) = (self.w, self.cw);
        for y in 0..w {
            for x in 0..w {
                let i = y * w + x;
                let c = self.cells[i];
                if c.best_seg == NO_SEG {
                    self.pen[i] = f32::INFINITY;
                    continue;
                }
                if c.sec_seg == NO_SEG {
                    self.pen[i] = c.best;
                    continue;
                }
                let p = [(x as f64 + 0.5) * cw, (y as f64 + 0.5) * cw];
                let (sa, sb) = (
                    &self.segs[c.best_seg as usize],
                    &self.segs[c.sec_seg as usize],
                );
                let (qa, ta, da) = nearest_on(sa, p);
                let (qb, tb, db) = nearest_on(sb, p);
                let weight = if da > 0.0 && db > 0.0 {
                    let dot = ((p[0] - qa[0]) * (p[0] - qb[0]) + (p[1] - qa[1]) * (p[1] - qb[1]))
                        / (da * db);
                    0.5 * (1.0 - dot)
                } else {
                    0.0
                };
                let k = 0.5 * (f64::from(sa.k2) + f64::from(sb.k2)) * weight;
                // ★ THE BENCH'S BOUND ON S2 (stated): a divide never stands under the higher of its two streams' beds
                // — no closed hollow between two streams. S2's width reaches ~700 m where the steady rate is low, and
                // its rounding (up to k/4) then sank a divide 160 m under both beds.
                let bed_a = f64::from(sa.bed[0]) + ta * f64::from(sa.bed[1] - sa.bed[0]);
                let bed_b = f64::from(sb.bed[0]) + tb * f64::from(sb.bed[1] - sb.bed[0]);
                self.pen[i] = smin(f64::from(c.best), f64::from(c.sec), k)
                    .max(bed_a.max(bed_b))
                    .min(f64::from(c.best)) as f32;
            }
        }
    }

    /// `h = min(S + L, P) + T` (no deposits, no roughness in part 1; `T` the mean hold's exact remainder).
    fn compose_h(&mut self) {
        for i in 0..self.w * self.w {
            self.h[i] = (self.start[i] + self.lift[i]).min(self.pen[i]) + self.hold[i];
        }
    }

    /// Each node's mean of a raster.
    fn node_means(
        &self,
        f: &[f32],
        only_finite: bool,
        minus: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<usize>) {
        let n = self.f.n;
        let per = (NODE_M / self.cw).round() as usize;
        let mut sum = vec![0.0f64; n * n];
        let mut count = vec![0usize; n * n];
        for y in 0..self.w {
            for x in 0..self.w {
                let v = f[y * self.w + x];
                if only_finite && !v.is_finite() {
                    continue;
                }
                let k = (y / per).min(n - 1) * n + (x / per).min(n - 1);
                sum[k] += f64::from(v) - minus.map_or(0.0, |m| m[k]);
                count[k] += 1;
            }
        }
        let means = sum
            .iter()
            .zip(&count)
            .map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();
        (means, count)
    }

    /// ★ THE LIFT AND THE MEAN HOLD (§3.6, and the bench's closure, stated).
    /// 1. THE DESIGN'S LIFT `L` raises the START alone: each land node's `L_n` grows by its shortfall `Z − ⟨h⟩`
    ///    (`h = min(S + L, P)`), read by Catmull-Rom, `iters` times — but never past the lift at which the start stops
    ///    showing in that node (`max(P − S)` over the node): past it a lift changes nothing there, and the first run
    ///    without that rule grew `L` to 4.7 km in fully carved nodes and spilled it into their neighbours.
    /// 2. THE MEAN HOLD `T` (the R2 bench's own hold) takes the exact remainder: a smooth offset of the whole drawn
    ///    ground, the Catmull-Rom of one offset per node, iterated (a shift of the beds alone diverged on Taiwan — a
    ///    node's mean is also pulled down by a NEIGHBOUR's deep valley).
    ///
    /// Returns the largest and the mean `|Z − ⟨h⟩|` over the land nodes after (G-MEAN).
    fn lift_iterate(&mut self, iters: usize) -> (f64, f64) {
        let n = self.f.n;
        let per = (NODE_M / self.cw).round() as usize;
        let mut sat = vec![f64::NEG_INFINITY; n * n];
        for y in 0..self.w {
            for x in 0..self.w {
                let i = y * self.w + x;
                if self.pen[i].is_finite() {
                    let k = (y / per).min(n - 1) * n + (x / per).min(n - 1);
                    sat[k] = sat[k].max(f64::from(self.pen[i] - self.start[i]));
                }
            }
        }
        self.lift_nodes.iter_mut().for_each(|l| *l = 0.0);
        self.lift.iter_mut().for_each(|l| *l = 0.0);
        self.hold_nodes.iter_mut().for_each(|l| *l = 0.0);
        self.hold.iter_mut().for_each(|l| *l = 0.0);
        for _ in 0..iters {
            self.compose_h();
            let (means, _) = self.node_means(&self.h, false, None);
            for k in 0..n * n {
                if !self.f.sea[k] {
                    let top = if sat[k].is_finite() {
                        sat[k]
                    } else {
                        f64::INFINITY
                    };
                    // The lift is a carve's volume: never negative (no deposits in part 1).
                    self.lift_nodes[k] = (self.lift_nodes[k] + self.f.z[k] - means[k])
                        .min(top)
                        .max(0.0);
                }
            }
            self.lift = node_cr_raster(&self.lift_nodes, n, self.w, self.cw);
        }
        for _ in 0..iters {
            self.compose_h();
            let (means, _) = self.node_means(&self.h, false, None);
            for ((t, z), m) in self.hold_nodes.iter_mut().zip(&self.f.z).zip(&means) {
                *t += z - m;
            }
            self.hold = node_cr_raster(&self.hold_nodes, n, self.w, self.cw);
        }
        self.compose_h();
        let (means, _) = self.node_means(&self.h, false, None);
        let land: Vec<f64> = (0..n * n)
            .filter(|&k| !self.f.sea[k])
            .map(|k| (self.f.z[k] - means[k]).abs())
            .collect();
        let worst = land.iter().copied().fold(0.0f64, f64::max);
        (worst, land.iter().sum::<f64>() / land.len().max(1) as f64)
    }

    /// ★ THE DATUM'S CLOSURE (the bench's; the design leaves `M` and `L` one freedom apart): `M_n` = the node's mean of
    /// the law's own steady relief over its anchor's bed — the BOUNDED envelope (each side flat past its `L_H`), with
    /// no start. Returns the mean `|ΔM|` over the land nodes.
    fn update_m(&mut self) -> f64 {
        let n = self.f.n;
        let bnd: Vec<f32> = self.cells.iter().map(|c| c.bnd).collect();
        let (means, count) = self.node_means(&bnd, true, Some(&self.anchor_bed));
        let mut change = 0.0;
        let mut land = 0usize;
        for k in 0..n * n {
            if self.f.sea[k] || count[k] == 0 {
                continue;
            }
            change += (means[k] - self.m_nodes[k]).abs();
            self.m_nodes[k] = means[k];
            land += 1;
        }
        change / land.max(1) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE READINGS — one rule for both grounds at one cell.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// A field's drainage: the priority flood from the rim and the sea (`z ≤ 0`), the steepest descent on the filled
/// surface (a flat taking the cell the flood came from); the order outlets first; the area in cells.
struct Net {
    recv: Vec<u32>,
    order: Vec<u32>,
    area: Vec<f64>,
}

fn net_of(z: &[f64], w: usize) -> Net {
    let cells = w * w;
    let key = |h: f64| -> i64 { (h * 65_536.0).round() as i64 };
    let mut zf = z.to_vec();
    let mut seen = vec![false; cells];
    let mut parent = vec![u32::MAX; cells];
    let mut order: Vec<u32> = Vec::with_capacity(cells);
    let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
    for y in 0..w {
        for x in 0..w {
            let i = y * w + x;
            if x == 0 || y == 0 || x == w - 1 || y == w - 1 || z[i] <= 0.0 {
                seen[i] = true;
                heap.push(Reverse((key(zf[i]), i as u32)));
            }
        }
    }
    while let Some(Reverse((_, c))) = heap.pop() {
        let c = c as usize;
        order.push(c as u32);
        let (cx, cy) = ((c % w) as i64, (c / w) as i64);
        for (dx, dy) in NB {
            let (nx, ny) = (cx + dx, cy + dy);
            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                continue;
            }
            let m = ny as usize * w + nx as usize;
            if seen[m] {
                continue;
            }
            seen[m] = true;
            zf[m] = zf[m].max(zf[c]);
            parent[m] = c as u32;
            heap.push(Reverse((key(zf[m]), m as u32)));
        }
    }
    let mut recv = vec![u32::MAX; cells];
    for i in 0..cells {
        if parent[i] == u32::MAX {
            continue;
        }
        let (cx, cy) = ((i % w) as i64, (i / w) as i64);
        let mut best = (0.0f64, u32::MAX);
        for (dx, dy) in NB {
            let (nx, ny) = (cx + dx, cy + dy);
            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                continue;
            }
            let m = ny as usize * w + nx as usize;
            let chord = if dx != 0 && dy != 0 {
                std::f64::consts::SQRT_2
            } else {
                1.0
            };
            let s = (zf[i] - zf[m]) / chord;
            if s > best.0 {
                best = (s, m as u32);
            }
        }
        recv[i] = if best.1 == u32::MAX {
            parent[i]
        } else {
            best.1
        };
    }
    let mut area = vec![1.0f64; cells];
    for &c in order.iter().rev() {
        let r = recv[c as usize];
        if r != u32::MAX {
            area[r as usize] += area[c as usize];
        }
    }
    Net { recv, order, area }
}

/// ★ THE STRAIGHT LINES (the R2 bench's rule): of the cells whose area is in the top `share` (the valleys), the share
/// whose next `run` cells downstream stay within one cell of the straight line from the first to the last.
fn straight_share(w: usize, net: &Net, share: f64, run: usize) -> (f64, usize) {
    let mut qs = net.area.clone();
    qs.sort_by(f64::total_cmp);
    let threshold = qs[((1.0 - share) * qs.len() as f64) as usize];
    let (mut straight, mut all) = (0usize, 0usize);
    for i in (0..w * w).filter(|&i| net.area[i] >= threshold) {
        let mut path = vec![i];
        let mut c = i;
        while path.len() <= run {
            let r = net.recv[c];
            if r == u32::MAX {
                break;
            }
            c = r as usize;
            path.push(c);
        }
        if path.len() <= run {
            continue;
        }
        let at = |k: usize| ((path[k] % w) as f64, (path[k] / w) as f64);
        let ((x0, y0), (x1, y1)) = (at(0), at(run));
        let (dx, dy) = (x1 - x0, y1 - y0);
        let len = (dx * dx + dy * dy).sqrt();
        if len == 0.0 {
            continue;
        }
        let dev = (0..=run)
            .map(|k| {
                let (x, y) = at(k);
                ((x - x0) * dy - (y - y0) * dx).abs() / len
            })
            .fold(0.0f64, f64::max);
        all += 1;
        straight += usize::from(dev <= 1.0);
    }
    (straight as f64 / all.max(1) as f64, all)
}

/// ★ THE ROUGHNESS BY AXIS (the R2 bench's stripes): along x, y and the two diagonals together, the mean square of the
/// second difference over the mean square of the first.
fn stripes(z: &[f64], w: usize) -> [f64; 3] {
    let w = w as i64;
    let at = |x: i64, y: i64| z[(y * w + x) as usize];
    let mut sums = [[0.0f64; 2]; 3];
    for y in 1..w - 1 {
        for x in 1..w - 1 {
            for (k, (dx, dy)) in [(1i64, 0i64), (0, 1), (1, 1), (1, -1)]
                .into_iter()
                .enumerate()
            {
                let c = at(x, y);
                let (p, m) = (at(x + dx, y + dy), at(x - dx, y - dy));
                let slot = k.min(2);
                sums[slot][0] += (p - 2.0 * c + m) * (p - 2.0 * c + m);
                sums[slot][1] += (p - c) * (p - c);
            }
        }
    }
    sums.map(|[second, first]| second / first.max(f64::MIN_POSITIVE))
}

/// ★ THE GRID'S LEAN (the R2 bench's rule): of the valley cells (area in the top `share`), the share whose Horn
/// gradient bearing lies within 5° of a lattice line. None: 22 %.
fn lean(z: &[f64], w: usize, net: &Net, share: f64) -> (f64, usize) {
    let mut sorted = net.area.clone();
    sorted.sort_by(f64::total_cmp);
    let cut = sorted[((sorted.len() - 1) as f64 * (1.0 - share)) as usize];
    let (mut near, mut all) = (0usize, 0usize);
    for y in 1..w - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if net.area[i] < cut {
                continue;
            }
            let at = |dx: i64, dy: i64| z[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize];
            let gx = (at(1, -1) + 2.0 * at(1, 0) + at(1, 1))
                - (at(-1, -1) + 2.0 * at(-1, 0) + at(-1, 1));
            let gy = (at(-1, 1) + 2.0 * at(0, 1) + at(1, 1))
                - (at(-1, -1) + 2.0 * at(0, -1) + at(1, -1));
            if gx == 0.0 && gy == 0.0 {
                continue;
            }
            let bearing = (-gy).atan2(-gx).to_degrees().rem_euclid(45.0);
            all += 1;
            if !(5.0..=40.0).contains(&bearing) {
                near += 1;
            }
        }
    }
    (near as f64 / all.max(1) as f64, all)
}

/// The network's channel donors: for each cell, its donors with area at least `a_min`, largest first.
fn channel_donors(w: usize, net: &Net, a_min: f64) -> Vec<Vec<u32>> {
    let mut donors: Vec<Vec<u32>> = vec![Vec::new(); w * w];
    for i in 0..w * w {
        let r = net.recv[i];
        if r != u32::MAX && net.area[i] >= a_min {
            donors[r as usize].push(i as u32);
        }
    }
    for d in &mut donors {
        d.sort_by(|a, b| net.area[*b as usize].total_cmp(&net.area[*a as usize]));
    }
    donors
}

/// From `d`, up the largest channel donor for at most `len` cells: the end and the cells walked.
fn walk_up(donors: &[Vec<u32>], d: usize, len: usize) -> (usize, usize) {
    let (mut cur, mut steps) = (d, 1usize);
    while steps < len {
        match donors[cur].first() {
            Some(&u) => {
                cur = u as usize;
                steps += 1;
            }
            None => break,
        }
    }
    (cur, steps)
}

/// The angle between two vectors, degrees.
fn angle_deg(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (la, lb) = (
        (a.0 * a.0 + a.1 * a.1).sqrt(),
        (b.0 * b.0 + b.1 * b.1).sqrt(),
    );
    if la == 0.0 || lb == 0.0 {
        return f64::NAN;
    }
    ((a.0 * b.0 + a.1 * b.1) / (la * lb))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// Mean and standard deviation.
fn mean_sd(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    (m, var.sqrt())
}

/// ★ THE JUNCTION ANGLES AND THE TRIBUTARY BEARINGS. At each channel cell (area ≥ `a_min` cells) with two or more
/// channel donors: the JUNCTION ANGLE between the two largest branches, each read from the junction to the point `len`
/// cells up its main line (Seybold, Rothman & Kirchner 2017 read the branches' directions the same way); and, along
/// each main stem (a stem continues up its largest donor), every tributary's angle to the stem's own arriving
/// direction. Returns (angle mean, angle sd, junctions; the mean over stems with four or more tributaries of the sd of
/// those angles, and the stems).
fn junctions(w: usize, net: &Net, a_min: f64, len: usize) -> (f64, f64, usize, f64, usize) {
    let donors = channel_donors(w, net, a_min);
    let pos = |i: usize| ((i % w) as f64, (i / w) as f64);
    let mut stem = vec![u32::MAX; w * w];
    let mut next_stem = 0u32;
    let mut angles = Vec::new();
    let mut by_stem: std::collections::BTreeMap<u32, Vec<f64>> = std::collections::BTreeMap::new();
    for &c in net.order.iter().rev() {
        let c = c as usize;
        if net.area[c] < a_min {
            continue;
        }
        stem[c] = match donors[c].first() {
            Some(&d) => stem[d as usize],
            None => {
                next_stem += 1;
                next_stem
            }
        };
        if donors[c].len() < 2 {
            continue;
        }
        let pc = pos(c);
        let (main_end, main_steps) = walk_up(&donors, donors[c][0] as usize, len);
        let (second_end, second_steps) = walk_up(&donors, donors[c][1] as usize, len);
        if main_steps * 2 < len || second_steps * 2 < len {
            continue;
        }
        let (pm, ps) = (pos(main_end), pos(second_end));
        let a = angle_deg((pm.0 - pc.0, pm.1 - pc.1), (ps.0 - pc.0, ps.1 - pc.1));
        if a.is_finite() {
            angles.push(a);
        }
        let arriving = (pc.0 - pm.0, pc.1 - pm.1);
        for &d in &donors[c][1..] {
            let (end, steps) = walk_up(&donors, d as usize, len);
            if steps * 2 < len {
                continue;
            }
            let pe = pos(end);
            let t = angle_deg((pc.0 - pe.0, pc.1 - pe.1), arriving);
            if t.is_finite() {
                by_stem.entry(stem[c]).or_default().push(t);
            }
        }
    }
    let (am, asd) = mean_sd(&angles);
    let spreads: Vec<f64> = by_stem
        .values()
        .filter(|v| v.len() >= 4)
        .map(|v| mean_sd(v).1)
        .collect();
    (am, asd, angles.len(), mean_sd(&spreads).0, spreads.len())
}

/// ★ THE VALLEY SPACING: along every `stride`-th row and column, the valleys (minima on land with a prominence of at
/// least `thr` metres); the land length over the count, metres.
fn spacing(z: &[f64], w: usize, cell_m: f64, thr: f64, stride: usize) -> (f64, usize) {
    let mut count = 0usize;
    let mut land = 0usize;
    for k in (0..w).step_by(stride) {
        let row: Vec<f64> = (0..w).map(|x| z[k * w + x]).collect();
        let col: Vec<f64> = (0..w).map(|y| z[y * w + k]).collect();
        for t in [row, col] {
            land += t.iter().filter(|&&v| v > 0.0).count();
            count += count_valleys(&t, false, thr, &|i| t[i] > 0.0);
        }
    }
    (land as f64 * cell_m / count.max(1) as f64, count)
}

/// One ground's readings at one cell.
#[derive(Clone, Debug, Default)]
struct Reading {
    straight: f64,
    stripes: [f64; 3],
    lean: f64,
    hack_c: f64,
    hack_n: f64,
    junction: (f64, f64, usize),
    bearing_spread: (f64, usize),
    spacing_m: f64,
    pits_1m: usize,
    pits_10m: usize,
    land_km2: f64,
}

/// Every reading of a field `z` (`w × w`, `cell_m`), with the channel threshold `a_min_m2`.
fn read_field(z: &[f64], w: usize, cell_m: f64, a_min_m2: f64) -> Reading {
    let net = net_of(z, w);
    let (straight, _) = straight_share(w, &net, 0.05, 16);
    let (le, _) = lean(z, w, &net, 0.05);
    let a_min = (a_min_m2 / (cell_m * cell_m)).max(4.0);
    let (am, asd, jn, spread, stems) = junctions(w, &net, a_min, 6);
    let (sp, _) = spacing(z, w, cell_m, 10.0, 4);
    let hi: Vec<i64> = z.iter().map(|&v| (v * 65_536.0).round() as i64).collect();
    let dry: Vec<bool> = z.iter().map(|&v| v > 0.0).collect();
    let heights = Heights {
        w,
        cell_m,
        h: &hi,
        per_m: 65_536.0,
        dry: Some(&dry),
    };
    let (hack_c, hack_n, _) = g_hack(&heights);
    let spill = spill_census(&heights);
    let land_km2 = dry.iter().filter(|&&d| d).count() as f64 * cell_m * cell_m / 1.0e6;
    Reading {
        straight,
        stripes: stripes(z, w),
        lean: le,
        hack_c,
        hack_n,
        junction: (am, asd, jn),
        bearing_spread: (spread, stems),
        spacing_m: sp,
        pits_1m: spill.at_least(1.0),
        pits_10m: spill.at_least(10.0),
        land_km2,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE PICTURES (the shaded relief of `erosion_bench`, its tint and its sun).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The height tint over the sea: blue by depth under it, green to tan to brown to grey to snow over it.
fn tint(z: f64, sea_m: f64) -> [f64; 3] {
    let ramp = |stops: &[(f64, [f64; 3])], t: f64| -> [f64; 3] {
        let mut prev = stops[0];
        for &(at, c) in stops {
            if t <= at {
                let span = (at - prev.0).max(1e-9);
                let k = ((t - prev.0) / span).clamp(0.0, 1.0);
                return [
                    prev.1[0] + (c[0] - prev.1[0]) * k,
                    prev.1[1] + (c[1] - prev.1[1]) * k,
                    prev.1[2] + (c[2] - prev.1[2]) * k,
                ];
            }
            prev = (at, c);
        }
        prev.1
    };
    if z < sea_m {
        ramp(
            &[
                (0.0, [150.0, 195.0, 225.0]),
                (1_000.0, [70.0, 130.0, 190.0]),
                (6_000.0, [25.0, 55.0, 110.0]),
            ],
            sea_m - z,
        )
    } else {
        ramp(
            &[
                (0.0, [95.0, 140.0, 85.0]),
                (400.0, [160.0, 175.0, 105.0]),
                (1_200.0, [200.0, 175.0, 115.0]),
                (2_500.0, [160.0, 115.0, 80.0]),
                (4_500.0, [150.0, 140.0, 132.0]),
                (6_500.0, [240.0, 240.0, 240.0]),
            ],
            z - sea_m,
        )
    }
}

/// One shaded relief of `h` (`n × n`, `dx` metres a pixel), the sun at 315° and 45° up (Horn's gradient), heights
/// exaggerated `exaggeration` times.
fn render(h: &[f64], n: usize, dx: f64, exaggeration: f64) -> image::RgbImage {
    let mut img = image::RgbImage::new(n as u32, n as u32);
    let zenith = (90.0f64 - 45.0).to_radians();
    let azimuth = (360.0f64 - 315.0 + 90.0).to_radians();
    let at = |x: i64, y: i64| {
        h[(y.clamp(0, n as i64 - 1) as usize) * n + x.clamp(0, n as i64 - 1) as usize]
    };
    for y in 0..n as i64 {
        for x in 0..n as i64 {
            let (a, b, c) = (at(x - 1, y - 1), at(x, y - 1), at(x + 1, y - 1));
            let (d, f) = (at(x - 1, y), at(x + 1, y));
            let (g, hh, i) = (at(x - 1, y + 1), at(x, y + 1), at(x + 1, y + 1));
            let dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * dx) * exaggeration;
            let dzdy = ((g + 2.0 * hh + i) - (a + 2.0 * b + c)) / (8.0 * dx) * exaggeration;
            let slope = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
            let mut aspect = dzdy.atan2(-dzdx);
            if aspect < 0.0 {
                aspect += std::f64::consts::TAU;
            }
            let lit = (zenith.cos() * slope.cos()
                + zenith.sin() * slope.sin() * (azimuth - aspect).cos())
            .max(0.0);
            let base = tint(h[(y as usize) * n + x as usize], 0.0);
            let k = 0.35 + 0.65 * lit;
            img.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([
                    (base[0] * k).clamp(0.0, 255.0) as u8,
                    (base[1] * k).clamp(0.0, 255.0) as u8,
                    (base[2] * k).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    img
}

/// Writes `a` and `b` alone and side by side (a white gap of 8 pixels): the generated ground on the left.
fn write_pair(dir: &str, name: &str, a: &image::RgbImage, b: &image::RgbImage) -> [String; 3] {
    let _ = std::fs::create_dir_all(dir);
    let (w, h) = (a.width(), a.height());
    let mut pair = image::RgbImage::from_pixel(2 * w + 8, h, image::Rgb([255, 255, 255]));
    for y in 0..h {
        for x in 0..w {
            pair.put_pixel(x, y, *a.get_pixel(x, y));
            pair.put_pixel(w + 8 + x, y, *b.get_pixel(x, y));
        }
    }
    let paths = [
        format!("{dir}/{name}_generated.png"),
        format!("{dir}/{name}_earth.png"),
        format!("{dir}/{name}_generated_vs_earth.png"),
    ];
    for (img, p) in [(a, &paths[0]), (b, &paths[1]), (&pair, &paths[2])] {
        if let Err(e) = img.save(p) {
            println!("  the picture could not be written to {p}: {e}");
        }
    }
    paths
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// MAIN.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sarg = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|k| args.get(k + 1))
            .cloned()
    };
    let out = sarg("--out").unwrap_or_else(|| "landforms_bench1".to_string());
    let cell: f64 = sarg("--cell").and_then(|v| v.parse().ok()).unwrap_or(32.0);
    let outer: usize = sarg("--outer").and_then(|v| v.parse().ok()).unwrap_or(2);
    let step_factor: f64 = sarg("--route-step")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let window_nodes: usize = sarg("--window-nodes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let only = sarg("--tiles");
    let threads = sarg("--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(4, |n| n.get().saturating_sub(2).max(1))
        });
    println!("landforms_bench — bench step 1, part 1: the drainage ladder on Earth (ruling B79)");
    println!(
        "  the working cell {cell} m, {outer} outer passes, the route's step {} m, {threads} threads; pictures to {out}",
        cell * step_factor
    );
    println!(
        "  the laws: the joint law K_lp {JOINT_K_LP:e} m^-2, n {JOINT_N}; S1/S2 K_h {K_H} m²/yr, S_c {SOIL_CRITICAL_SLOPE}, ρ_r/ρ_s {RHO_RATIO}; the width {WIDTH_A}·Q^{WIDTH_B} (Q in m³/s)"
    );

    // ── c_λ: the law's own non-dimensional form, solved once. ──
    let cl_dx: f64 = sarg("--cl-dx").and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let cl_t: f64 = sarg("--cl-t").and_then(|v| v.parse().ok()).unwrap_or(240.0);
    println!("  c_λ's solve: the cell {cl_dx} l_c, the time {cl_t} (in l_c and U units)");
    let cl = c_lambda(JOINT_N, M_EXP, cl_dx, cl_t);
    println!(
        "c_λ (n {JOINT_N}, m {M_EXP}): {:.2} l_c at the transect halfway between the divide and the base level; by the distance from the base level as a share of the half width: {}; the steady residual |∂z/∂t|/U {:.4}; {:.1} s",
        cl.value,
        cl.by_row
            .iter()
            .map(|(s, v)| format!("{s:.2} → {v:.2}"))
            .collect::<Vec<_>>()
            .join(", "),
        cl.residual,
        cl.seconds
    );
    let control = c_lambda(1.0, 0.5, cl_dx, cl_t);
    if sarg("--cl-only").is_some() {
        return;
    }
    println!(
        "c_λ CONTROL (n 1, m 0.5; the design quotes Perron et al. 2009's 6.4–12.7 for n = 1, its m UNMEASURED here): {:.2}; {}; residual {:.4}",
        control.value,
        control
            .by_row
            .iter()
            .map(|(s, v)| format!("{s:.2} → {v:.2}"))
            .collect::<Vec<_>>()
            .join(", "),
        control.residual
    );
    let c_l = cl.value;

    for spec in TILES
        .iter()
        .filter(|t| only.as_deref().is_none_or(|o| o.contains(t.label)))
    {
        run_tile(
            spec,
            c_l,
            cell,
            outer,
            step_factor,
            window_nodes,
            threads,
            &out,
        );
    }
}

/// One tile: the facts, the ladder, the pictures, the readings.
#[allow(clippy::too_many_arguments)]
fn run_tile(
    spec: &TileSpec,
    c_l: f64,
    cell: f64,
    outer: usize,
    step_factor: f64,
    window_nodes: usize,
    threads: usize,
    out: &str,
) {
    println!("\n══ {} ({}) ══", spec.label.to_uppercase(), spec.path);
    println!(
        "  the rain {:.0} mm/yr — {} (ONE per-tile mean, no gridded climatology)",
        spec.rain_mm, spec.rain_source
    );
    let (z8, n) = earth_cells(spec.path, NODE_M);
    let per = (NODE_M / cell).round() as usize;
    let w = n * per;
    let (e_fine_all, ew) = earth_cells(spec.path, cell);
    let e_fine = crop(&e_fine_all, ew, 0, 0, w);
    drop(e_fine_all);
    let mut share = vec![0.0f64; n * n];
    for y in 0..w {
        for x in 0..w {
            if e_fine[y * w + x] <= 0.0 {
                share[(y / per) * n + x / per] += 1.0;
            }
        }
    }
    share.iter_mut().for_each(|s| *s /= (per * per) as f64);
    let f = Facts::new(z8, &share, n, spec.rain_mm / 1_000.0);
    let land: Vec<usize> = (0..n * n).filter(|&k| !f.sea[k]).collect();
    let laws: Vec<PointLaw> = land
        .iter()
        .map(|&k| point_law(f.log_ks[k], f.rain_m, c_l))
        .collect();
    let pick = |g: &dyn Fn(&PointLaw) -> f64| -> (f64, f64, f64) {
        let mut v: Vec<f64> = laws.iter().map(g).collect();
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2], v[v.len() - 1])
    };
    let zr = land.iter().map(|&k| f.z[k]);
    println!(
        "  THE 8 KM FACTS: {n} × {n} nodes ({:.1} km a side), {} land, {} sea; Z {:.0}…{:.0} m; voting nodes {}; the tree's largest area {:.0} km²",
        n as f64 * NODE_M / 1_000.0,
        land.len(),
        n * n - land.len(),
        zr.clone().fold(f64::INFINITY, f64::min),
        zr.fold(f64::NEG_INFINITY, f64::max),
        f.vote.iter().filter(|&&v| v).count(),
        f.area.iter().copied().fold(0.0, f64::max) / 1.0e6
    );
    let slopes: Vec<f64> = land
        .iter()
        .filter(|&&k| f.vote[k])
        .map(|&k| f.slope[k])
        .collect();
    let (smin_, smed, smax) = {
        let mut s = slopes.clone();
        s.sort_by(f64::total_cmp);
        (s[0], s[s.len() / 2], s[s.len() - 1])
    };
    println!(
        "  the links' slope S (min/median/max over the voting nodes): {smin_:.4} / {smed:.4} / {smax:.4}"
    );
    for (name, g) in [
        ("k_s", (&|l: &PointLaw| l.ks) as &dyn Fn(&PointLaw) -> f64),
        ("U = E (mm/yr)", &|l: &PointLaw| l.e * 1_000.0),
        ("l_c (m)", &|l: &PointLaw| l.lc),
        ("λ = c_λ·l_c (m)", &|l: &PointLaw| l.lambda),
        ("S_h", &|l: &PointLaw| l.sh),
        ("S2's k (m)", &|l: &PointLaw| l.k2),
    ] {
        let (a, b, c) = pick(g);
        println!("  {name:>16} over the land nodes (min/median/max): {a:.4e} / {b:.4e} / {c:.4e}");
    }

    // ── THE LADDER. ──
    let started = Instant::now();
    let mut lad = Ladder::new(&f, c_l, cell, threads);
    println!(
        "  THE LADDER: {} levels below the trunks (the finest {:.0} m), the raster {w} × {w} cells of {cell} m",
        lad.levels,
        NODE_M / f64::from(1u32 << lad.levels)
    );
    let mut timing = Vec::new();
    for o in 0..outer {
        let t = Instant::now();
        lad.set_anchor_beds();
        lad.build_routes(step_factor);
        let routes_s = t.elapsed().as_secs_f64();
        lad.real_areas();
        lad.bed_all();
        lad.splat_all();
        let dm = lad.update_m();
        lad.set_anchor_beds();
        lad.bed_all();
        lad.splat_all();
        let (worst, mean) = lad.lift_iterate(20);
        println!(
            "  outer pass {}: the routes {routes_s:.1} s, the pass {:.1} s; |ΔM| mean {dm:.1} m; G-MEAN after the lift: worst {worst:.2} m, mean {mean:.2} m; {} streams, {} segments, {} at the reach bound",
            o + 1,
            t.elapsed().as_secs_f64(),
            lad.streams.len(),
            lad.segs.len(),
            lad.capped
        );
        timing.push(t.elapsed().as_secs_f64());
    }
    let t = Instant::now();
    let (worst, mean) = lad.lift_iterate(20);
    let total_s = started.elapsed().as_secs_f64();
    println!(
        "  the final splat and lift {:.1} s; G-MEAN worst {worst:.2} m, mean {mean:.2} m over the land nodes",
        t.elapsed().as_secs_f64()
    );
    for s in &lad.stats {
        println!(
            "    level {:>6.0} m: {} sources, {} too fine for the law's λ, {} on the sea, {} on a floor already, {} joined ({} of them to a same-level stream), {} stuck (G-ORPHAN), {} left the tile; {:.1} steps a route",
            s.cell_m,
            s.sources,
            s.skipped_lambda,
            s.skipped_sea,
            s.at_floor,
            s.joined,
            s.joined_same,
            s.stuck,
            s.left,
            s.steps as f64 / (s.joined + s.stuck + s.left).max(1) as f64
        );
    }
    let clipped = (0..w * w)
        .filter(|&i| lad.start[i] + lad.lift[i] < lad.pen[i])
        .count();
    let m_land: Vec<f64> = land.iter().map(|&k| lad.m_nodes[k]).collect();
    let l_land: Vec<f64> = land.iter().map(|&k| lad.lift_nodes[k]).collect();
    let t_land: Vec<f64> = land.iter().map(|&k| lad.hold_nodes[k]).collect();
    println!(
        "  THE CARVE: the start stands on {:.1} % of the columns (where no profile lies under it); M (the datum) mean {:.0} m, max {:.0} m; L (the start's lift) mean {:.0} m, max {:.0} m; T (the hold's remainder) mean {:.0} m, |T| max {:.0} m",
        100.0 * clipped as f64 / (w * w) as f64,
        mean_sd(&m_land).0,
        m_land.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean_sd(&l_land).0,
        l_land.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean_sd(&t_land).0,
        t_land.iter().map(|t| t.abs()).fold(0.0f64, f64::max)
    );
    // A probe of single columns (`--probe x,y` in 128 m picture pixels), for naming a picture's defect.
    if let Ok(list) = std::env::var("VD_LANDFORMS_PROBE") {
        for pair in list.split(';') {
            let v: Vec<usize> = pair
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            if v.len() != 2 {
                continue;
            }
            let k = (128.0 / cell).round() as usize;
            let (x, y) = (v[0] * k + k / 2, v[1] * k + k / 2);
            let i = y * w + x;
            let c = lad.cells[i];
            let node = f.node_at((x as f64 + 0.5) * cell, (y as f64 + 0.5) * cell);
            let seg = (c.best_seg != NO_SEG).then(|| lad.segs[c.best_seg as usize]);
            println!(
                "  PROBE ({}, {}) node {node} Z {:.1} sea {} M {:.1} anchor bed {:.1}: S {:.1} L {:.1} T {:.1} P {:.1} best {:.1} h {:.1}; best seg: {:?}",
                v[0],
                v[1],
                f.z[node],
                f.sea[node],
                lad.m_nodes[node],
                lad.anchor_bed[node],
                lad.start[i],
                lad.lift[i],
                lad.hold[i],
                lad.pen[i],
                c.best,
                lad.h[i],
                seg.map(|s| (
                    s.bed,
                    s.sh,
                    s.hw,
                    s.r,
                    lad.streams[s.stream as usize].level,
                    lad.streams[s.stream as usize].kind
                ))
            );
        }
    }
    let columns = (w * w) as f64;
    println!(
        "  THE COST: {total_s:.1} s for {} columns of {cell} m = {:.2} µs a column (all passes: routes, beds, {} full splats, the lift); one outer pass {:.2} µs a column; {threads} threads",
        w * w,
        total_s * 1.0e6 / columns,
        3 * outer + 1,
        timing.first().copied().unwrap_or(0.0) * 1.0e6 / columns
    );

    // ── THE PICTURES. ──
    let dir = out.to_string();
    let gen_fine: Vec<f64> = lad.h.iter().map(|&v| f64::from(v)).collect();
    let k128 = (128.0 / cell).round() as usize;
    let gen128 = block_mean(&lad.h, w, k128);
    let w128 = w / k128;
    let (e128_all, e128w) = earth_cells(spec.path, 128.0);
    let e128 = crop(&e128_all, e128w, 0, 0, w128);
    let start128 = block_mean(&lad.start, w, k128);
    let pair128 = write_pair(
        &dir,
        &format!("{}_128m_whole", spec.label),
        &render(&gen128, w128, 128.0, 4.0),
        &render(&e128, w128, 128.0, 4.0),
    );
    let start_path = format!("{dir}/{}_128m_whole_start.png", spec.label);
    let _ = render(&start128, w128, 128.0, 4.0).save(&start_path);
    // The window: the block of `window_nodes` nodes with the highest mean Z.
    let wn = window_nodes.min(n);
    let mut best = (f64::NEG_INFINITY, 0usize, 0usize);
    for j in 0..=n - wn {
        for i in 0..=n - wn {
            let mut s = 0.0;
            for b in 0..wn {
                for a in 0..wn {
                    s += f.z[(j + b) * n + i + a];
                }
            }
            if s > best.0 {
                best = (s, i, j);
            }
        }
    }
    let (wi, wj) = (best.1 * per, best.2 * per);
    let ww = wn * per;
    let gen_win = crop(&gen_fine, w, wi, wj, ww);
    let e_win = crop(&e_fine, w, wi, wj, ww);
    let pair_win = write_pair(
        &dir,
        &format!("{}_{}m_window", spec.label, cell as usize),
        &render(&gen_win, ww, cell, 1.0),
        &render(&e_win, ww, cell, 1.0),
    );
    println!(
        "  THE PICTURES (no far shadow, no cover, no water; the sun at 315°, 45° up; the generated ground LEFT, Earth RIGHT):"
    );
    println!("    the whole square at 128 m (heights ×4): {}", pair128[2]);
    println!("    the start (a) alone at 128 m (×4): {start_path}");
    println!(
        "    the {:.0} km window at {cell} m (heights ×1), nodes ({}, {}) to ({}, {}): {}",
        ww as f64 * cell / 1_000.0,
        best.1,
        best.2,
        best.1 + wn - 1,
        best.2 + wn - 1,
        pair_win[2]
    );

    // ── THE READINGS. ──
    let k256 = (256.0 / cell).round() as usize;
    let gen256 = block_mean(&lad.h, w, k256);
    let w256 = w / k256;
    let (e256_all, e256w) = earth_cells(spec.path, 256.0);
    let e256 = crop(&e256_all, e256w, 0, 0, w256);
    let lam_window: Vec<f64> = {
        let mut v = Vec::new();
        for y in (wj..wj + ww).step_by(8) {
            for x in (wi..wi + ww).step_by(8) {
                let p = [(x as f64 + 0.5) * cell, (y as f64 + 0.5) * cell];
                if !f.sea[f.node_at(p[0], p[1])] {
                    v.push(point_law(node_cr(&f.log_ks, n, p[0], p[1]), f.rain_m, c_l).lambda);
                }
            }
        }
        v.sort_by(f64::total_cmp);
        v
    };
    let law_lambda = lam_window
        .get(lam_window.len() / 2)
        .copied()
        .unwrap_or(f64::NAN);
    println!(
        "  THE READINGS (generated | Earth). The law's own spacing c_λ·l_c in the window: median {law_lambda:.0} m (the levels stop at the first cell under it)."
    );
    {
        let sl: Vec<f32> = (0..w * w).map(|i| lad.start[i] + lad.lift[i]).collect();
        let sl_fine: Vec<f64> = sl.iter().map(|&v| f64::from(v)).collect();
        let sl_win = crop(&sl_fine, w, wi, wj, ww);
        let hi: Vec<i64> = sl_win
            .iter()
            .map(|&v| (v * 65_536.0).round() as i64)
            .collect();
        let dry: Vec<bool> = sl_win.iter().map(|&v| v > 0.0).collect();
        let spill = spill_census(&Heights {
            w: ww,
            cell_m: cell,
            h: &hi,
            per_m: 65_536.0,
            dry: Some(&dry),
        });
        println!(
            "  G-PIT of the start + T ALONE in the window (no ladder): {} basins ≥ 1 m, {} ≥ 10 m",
            spill.at_least(1.0),
            spill.at_least(10.0)
        );
    }
    for (what, g, e, ww_, c, a_min) in [
        (
            "the whole square at 256 m",
            &gen256,
            &e256,
            w256,
            256.0,
            1.0e6,
        ),
        (
            "the whole square at 128 m",
            &gen128,
            &e128,
            w128,
            128.0,
            1.0e6,
        ),
        (
            "the window at the working cell",
            &gen_win,
            &e_win,
            ww,
            cell,
            1.0e5,
        ),
    ] {
        let t = Instant::now();
        let rg = read_field(g, ww_, c, a_min);
        let re = read_field(e, ww_, c, a_min);
        println!(
            "   {what} ({c:.0} m cells, channels from {:.2} km²; read in {:.1} s):",
            a_min / 1.0e6,
            t.elapsed().as_secs_f64()
        );
        print_readings(&rg, &re);
    }
}

/// The readings side by side.
fn print_readings(g: &Reading, e: &Reading) {
    let row = |name: &str, a: String, b: String| println!("      {name:<44} {a:>22} | {b:<22}");
    row(
        "THE STRAIGHT LINES (valleys, 16 cells)",
        format!("{:.1} %", 100.0 * g.straight),
        format!("{:.1} %", 100.0 * e.straight),
    );
    row(
        "roughness x / y / diagonal",
        format!(
            "{:.2}/{:.2}/{:.2}",
            g.stripes[0], g.stripes[1], g.stripes[2]
        ),
        format!(
            "{:.2}/{:.2}/{:.2}",
            e.stripes[0], e.stripes[1], e.stripes[2]
        ),
    );
    row(
        "the lean (none: 22 %)",
        format!("{:.1} %", 100.0 * g.lean),
        format!("{:.1} %", 100.0 * e.lean),
    );
    row(
        "Hack's n (c)",
        format!("{:.3} ({:.2})", g.hack_n, g.hack_c),
        format!("{:.3} ({:.2})", e.hack_n, e.hack_c),
    );
    row(
        "junction angle mean ± sd (junctions)",
        format!(
            "{:.1}° ± {:.1}° ({})",
            g.junction.0, g.junction.1, g.junction.2
        ),
        format!(
            "{:.1}° ± {:.1}° ({})",
            e.junction.0, e.junction.1, e.junction.2
        ),
    );
    row(
        "tributary bearings' sd per trunk (trunks)",
        format!("{:.1}° ({})", g.bearing_spread.0, g.bearing_spread.1),
        format!("{:.1}° ({})", e.bearing_spread.0, e.bearing_spread.1),
    );
    row(
        "valley spacing (minima ≥ 10 m)",
        format!("{:.0} m", g.spacing_m),
        format!("{:.0} m", e.spacing_m),
    );
    row(
        "G-PIT: closed basins ≥ 1 m / ≥ 10 m",
        format!("{} / {}", g.pits_1m, g.pits_10m),
        format!("{} / {}", e.pits_1m, e.pits_10m),
    );
    row(
        "land",
        format!("{:.0} km²", g.land_km2),
        format!("{:.0} km²", e.land_km2),
    );
}
