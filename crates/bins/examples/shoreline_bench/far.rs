//! ★ THE FAR DIAGNOSIS (plan §8.4): why the high lake draws as stripes from 1 000 km, and what the
//! second pocket's bump rows are — R0 only, one change at a time.
//!
//! **THE HIGH LAKE.** At rungs 10–13 R0 reads the pyramid's level-1 means for the ground and the
//! level-1 block's highest lake word for the water. Two one-change variants split the cause:
//! **R0w** reads the nearest FINE row's own level in place of the block word (tests the word);
//! **R0b** reads the ground from the rows at the far rung (tests the means). Each line counts the
//! lake's columns drawn dry, split by cause, their phase on the level-1 lattice, the sheet drawn
//! over dry columns, and the dry land drawn under the lake's level.
//!
//! **THE POCKET'S BUMPS.** The drawn floor's local maxima, their spacing and the lean of their
//! nearest-neighbour vectors on the grid — on `s`, on `B` alone and on `f` alone — so the lattice of
//! bumps is laid at the solve's floor or at the octaves' own lattice.
//!
//! **Example.** If the stripes are the block word, R0w takes STRIPES-DRY-IN-LAKE to zero and R0b
//! leaves it; if they are the means, the reverse; if both leave it at zero, the columns are not the
//! stripes' cause and the drawing is (the census says so).

use vd_bins::shore_geometry::{RungReader, zero_contours};
use vd_bins::shore_rules::{NodeSite, Nodes, WORD_LAKE, WORD_LAND, factor_q, metres, unit_of_q};
use vd_bins::shore_shape::{chord_axes, flat_null_l4, four_fold, quantile};
use vd_recipe::Gi;
use vd_recipe::height::{SIDE_SEA, shore};
use vd_seed::bend::Face;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, CoastCounts, Side};

use crate::census::{Census, val};
use crate::eval::{NVAR, RungOut, Var};
use crate::stands::Window;

/// One far column under R0, R0w and R0b: `(s, l)` metres, `l` NaN for none.
type Three = [(f64, f64); 3];

/// One far column's reading: the three rules, the nearest fine node's word, whether its side is a
/// lake, and the nearest fine node's own lake level.
type FarCol = Option<(Three, u8, u8, i16)>;

/// ★ THE HIGH LAKE'S FAR LINES on W3 (rungs 10–13), and the phase line at rung 11.
#[allow(clippy::too_many_arguments, reason = "the lake's own words")]
pub fn high_lake(
    c: &mut Census,
    body: &BodyDefinition,
    artifact: &Artifact,
    counts: &std::sync::Arc<CoastCounts>,
    nodes: &Nodes,
    w3: &Window,
    level: i16,
    threads: usize,
) -> Vec<crate::panels::TileSrc> {
    let fine = body.macro_lattice().expect("a lattice");
    let mut tiles: Vec<crate::panels::TileSrc> = Vec::new();
    for r in w3.rungs() {
        let (i0, j0, n) = w3.at(r);
        let Some(r1) = RungReader::new(body, artifact, counts, w3.face, r) else {
            continue;
        };
        let Some(r0b) = RungReader::with_level(body, artifact, counts, w3.face, r, 0) else {
            continue;
        };
        let n = n as usize;
        let q = f64::from(vd_seed::ladder::cell_m(r)) / 128.0;
        let rq = r1.charter.radius;
        let sea_radius = r1.sea_radius();
        // Rows over the threads: each column's three reads.
        let rows: Vec<Vec<FarCol>> = {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let out: std::sync::Mutex<Vec<Option<Vec<FarCol>>>> =
                std::sync::Mutex::new(vec![None; n]);
            std::thread::scope(|sc| {
                for _ in 0..threads.max(1) {
                    sc.spawn(|| {
                        loop {
                            let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if b >= n {
                                break;
                            }
                            let j = j0 + b as i32;
                            let mut row = Vec::with_capacity(n);
                            for a in 0..n {
                                let i = i0 + a as i32;
                                let (Some(c1), Some(c0)) = (r1.column(i, j), r0b.column(i, j))
                                else {
                                    row.push(None);
                                    continue;
                                };
                                let site = NodeSite::of(&fine, w3.face, r, i, j);
                                let near = site.nearest(&fine);
                                let word = nodes.word[near as usize];
                                let lvl = nodes.level_m[near as usize];
                                let l_of = |w: Gi| {
                                    if w == Gi::ZERO {
                                        f64::NAN
                                    } else {
                                        metres(w - rq)
                                    }
                                };
                                let r0 = (metres(c1.surface - rq), l_of(c1.water));
                                // R0w: the fine row's own level on a lake-side column.
                                let r0w = if c1.side == Some(Side::Lake)
                                    && c1.water != sea_radius
                                    && lvl != vd_terrain::artifact::DRY_M
                                {
                                    let wq = rq + vd_bins::shore_rules::length_of(f64::from(lvl));
                                    let s = shore(c1.base, wq, c1.raw, SIDE_SEA);
                                    (metres(s - rq), metres(wq - rq))
                                } else {
                                    r0
                                };
                                // R0b: the rows' ground at the far rung, the level's side and water.
                                let s_b = shore(c0.base, c1.hold, c0.raw, c1.side_word);
                                let r0bv = (metres(s_b - rq), l_of(c1.water));
                                row.push(Some((
                                    [r0, r0w, r0bv],
                                    word,
                                    u8::from(c1.side == Some(Side::Lake)),
                                    lvl,
                                )));
                            }
                            if let Ok(mut o) = out.lock() {
                                o[b] = Some(row);
                            }
                        }
                    });
                }
            });
            out.into_inner()
                .unwrap_or_default()
                .into_iter()
                .map(Option::unwrap_or_default)
                .collect()
        };
        let cols: Vec<FarCol> = rows.into_iter().flatten().collect();
        if r == 11 {
            for vi in 0..3 {
                let mut t = crate::panels::TileSrc {
                    n,
                    s: vec![f32::NAN; n * n],
                    l: vec![f32::NAN; n * n],
                    wet: vec![false; n * n],
                };
                for (k, col) in cols.iter().enumerate() {
                    if let Some((three, ..)) = col {
                        let (sv, lv) = three[vi];
                        t.s[k] = sv as f32;
                        t.l[k] = lv as f32;
                        t.wet[k] = !lv.is_nan() && sv < lv;
                    }
                }
                tiles.push(t);
            }
        }
        let names = ["R0", "R0w", "R0b"];
        for (vi, name) in names.iter().enumerate() {
            let (mut dry_a, mut dry_b, mut dry_c, mut lake_cols) = (0usize, 0usize, 0usize, 0usize);
            let mut phase = [0usize; 64];
            let (mut over_dry, mut under_land) = (0usize, 0usize);
            for (k, col) in cols.iter().enumerate() {
                let Some((three, word, _, lvl)) = col else {
                    continue;
                };
                let (s, l) = three[vi];
                let wet = !l.is_nan() && s < l;
                if *word == WORD_LAKE && *lvl == level {
                    lake_cols += 1;
                    if !wet {
                        if l.is_nan() || (l - f64::from(nodes.sea_m)).abs() < 0.5 {
                            dry_a += 1;
                        } else if (l - f64::from(level)).abs() >= 0.5 {
                            dry_b += 1;
                        } else {
                            dry_c += 1;
                        }
                        let (a, b) = ((i0 as usize + k % n) % 8, (j0 as usize + k / n) % 8);
                        phase[b * 8 + a] += 1;
                    }
                }
                // The sheet over the column's 3 × 3 against a dry column.
                let (a, b) = ((k % n) as i64, (k / n) as i64);
                let mut lc = f64::NAN;
                for db in -1..=1i64 {
                    for da in -1..=1i64 {
                        let (x, y) = (a + da, b + db);
                        if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                            continue;
                        }
                        if let Some((t, ..)) = cols[y as usize * n + x as usize] {
                            let l2 = t[vi].1;
                            if !l2.is_nan() && (lc.is_nan() || l2 > lc) {
                                lc = l2;
                            }
                        }
                    }
                }
                if !wet && !lc.is_nan() && s < lc - q {
                    over_dry += 1;
                }
                if *word == WORD_LAND
                    && s < f64::from(level) - q
                    && !lc.is_nan()
                    && (lc - f64::from(level)).abs() < 0.5
                {
                    under_land += 1;
                }
            }
            let dry = dry_a + dry_b + dry_c;
            c.line(
                &format!("STRIPES-DRY-IN-LAKE.highlake.W3.{name}.r{r:02}"),
                &format!("{dry}of{lake_cols}cols(a-none-or-sea:{dry_a}/b-other-lake:{dry_b}/c-other:{dry_c})"),
                "0[survey§6.2;DEFERRED-stripes]",
                if dry == 0 { "GREEN" } else { "RED" },
            );
            if r == 11 {
                let mean = dry as f64 / 64.0;
                let top = phase.iter().copied().max().unwrap_or(0) as f64;
                let ratio = if mean > 0.0 { top / mean } else { f64::NAN };
                let null = phase_null(dry, 2_000);
                c.line(
                    &format!("STRIPES-PHASE.highlake.W3.{name}.r11"),
                    &format!("{}x-mean/{dry}cols", val(ratio, 2, "")),
                    &format!("<={}[uniform-null-q95]", val(null, 2, "")),
                    if dry < 64 {
                        "NO-BAND"
                    } else if ratio <= null {
                        "GREEN"
                    } else {
                        "RED"
                    },
                );
            }
            c.judged(
                &format!("STRIPES-SHEET-OVER-DRY.highlake.W3.{name}.r{r:02}"),
                &format!("{over_dry}cols"),
                "0[finding-2]",
                over_dry == 0,
            );
            c.reading(
                &format!("STRIPES-LAND-UNDER-LEVEL.highlake.W3.{name}.r{r:02}"),
                &format!("{under_land}cols"),
                "land-drawn-under-the-lake's-level-beside-its-sheet",
            );
        }
    }
    tiles
}

/// The uniform null of the phase histogram's biggest bin over its mean, q95 over `trials`.
fn phase_null(count: usize, trials: usize) -> f64 {
    if count == 0 {
        return f64::NAN;
    }
    let mut rng = 0x1357_9BDF_2468_ACE1u64;
    let mut out = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut bins = [0usize; 64];
        for _ in 0..count {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            bins[(rng >> 58) as usize] += 1;
        }
        out.push(*bins.iter().max().unwrap_or(&0) as f64 / (count as f64 / 64.0));
    }
    quantile(&out, 0.95)
}

/// The local maxima of a field over an `n × n` window: higher than all eight neighbours by `q`.
fn maxima(n: usize, h: &dyn Fn(usize) -> Option<f64>, q: f64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for b in 1..n.saturating_sub(1) {
        for a in 1..n - 1 {
            let Some(me) = h(b * n + a) else { continue };
            let mut top = true;
            'nb: for db in [-1i64, 0, 1] {
                for da in [-1i64, 0, 1] {
                    if da == 0 && db == 0 {
                        continue;
                    }
                    let k = ((b as i64 + db) as usize) * n + (a as i64 + da) as usize;
                    match h(k) {
                        Some(v) if me >= v + q => {}
                        _ => {
                            top = false;
                            break 'nb;
                        }
                    }
                }
            }
            if top {
                out.push((a, b));
            }
        }
    }
    out
}

/// The nearest-neighbour vectors of a point set, as axes in degrees, and their lengths.
fn nn_vectors(pts: &[(usize, usize)], m_i: f64, m_j: f64) -> (Vec<f64>, Vec<f64>) {
    let cell = 64usize;
    let mut grid: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (k, p) in pts.iter().enumerate() {
        grid.entry((p.0 / cell, p.1 / cell)).or_default().push(k);
    }
    let (mut axes, mut lens) = (Vec::new(), Vec::new());
    for (k, p) in pts.iter().enumerate() {
        let (gx, gy) = (p.0 / cell, p.1 / cell);
        let mut best: Option<(f64, f64, f64)> = None;
        for ring in 0..4usize {
            for dy in gy.saturating_sub(ring)..=gy + ring {
                for dx in gx.saturating_sub(ring)..=gx + ring {
                    if dx.abs_diff(gx) != ring && dy.abs_diff(gy) != ring {
                        continue;
                    }
                    for &m in grid.get(&(dx, dy)).map_or(&[][..], Vec::as_slice) {
                        if m == k {
                            continue;
                        }
                        let (vx, vy) = (
                            (pts[m].0 as f64 - p.0 as f64) * m_i,
                            (pts[m].1 as f64 - p.1 as f64) * m_j,
                        );
                        let d = vx.hypot(vy);
                        if best.is_none_or(|b| d < b.0) {
                            best = Some((d, vx, vy));
                        }
                    }
                }
            }
            if best.is_some() && ring >= 1 {
                break;
            }
        }
        if let Some((d, vx, vy)) = best {
            let mut ang = vy.atan2(vx).to_degrees();
            while ang < 0.0 {
                ang += 180.0;
            }
            while ang >= 180.0 {
                ang -= 180.0;
            }
            axes.push(ang);
            lens.push(d);
        }
    }
    (axes, lens)
}

/// ★ THE POCKET'S BUMP ROWS on one rung of one window (R0): the maxima of `s`, `B` and `f`, their
/// lattice lean and spacing, the straight walls, and the floor's lakes' lean.
#[allow(clippy::too_many_arguments, reason = "the window's own words")]
pub fn bumps(
    c: &mut Census,
    body: &BodyDefinition,
    reader: &RungReader,
    out: &RungOut,
    window: &str,
    m_i: f64,
    m_j: f64,
    nodes: &Nodes,
) {
    let n = out.n;
    let r = out.rung;
    let q = f64::from(vd_seed::ladder::cell_m(r)) / 128.0;
    let vi = Var::R0.idx();
    let tag = |g: &str| format!("{g}.pocket2.{window}.R0.r{r:02}");
    let waves: Vec<String> = body
        .octaves_at(r)
        .iter()
        .skip(body.first_fine())
        .map(|o| {
            let f =
                o.frequency_int.raw() as f64 + o.frequency_frac.raw() as f64 / (1u64 << 28) as f64;
            format!("{:.0}", body.radius_m() / f.max(1.0e-9))
        })
        .collect();
    let node_m = body.macro_lattice().map_or(f64::NAN, |l| l.node_m());
    for (what, pick) in [("s", 0usize), ("B", 1), ("f", 2)] {
        let h = |k: usize| -> Option<f64> {
            out.cols[k].as_ref().map(|col| match pick {
                0 => f64::from(col.drawn[vi].s),
                1 => f64::from(col.shared.b),
                _ => f64::from(col.shared.f),
            })
        };
        let pts = maxima(n, &h, q);
        let (axes, lens) = nn_vectors(&pts, m_i, m_j);
        let (l4, m4) = four_fold(&axes);
        let null = flat_null_l4(axes.len(), 2_000, 0.95);
        let gate = tag(&format!("BUMP-LATTICE.{what}"));
        let value = format!(
            "L4{}/M4{}/{}bumps",
            val(l4, 3, ""),
            val(m4, 3, ""),
            axes.len()
        );
        let band = format!("|L4|<={}[flat-null-q95]", val(null, 3, ""));
        if axes.len() < 16 {
            c.line(&gate, &value, &band, "NO-BAND");
        } else {
            c.judged(&gate, &value, &band, l4.abs() <= null);
        }
        c.reading(
            &tag(&format!("BUMP-SPACING.{what}")),
            &val(quantile(&lens, 0.5), 0, "m"),
            &format!("octave-waves[{}]m;node{:.0}m", waves.join(","), node_m),
        );
    }
    // FLAT-NODES and the factor on the floor (the pocket's own nodes under the window).
    let fine = body.macro_lattice().expect("a lattice");
    let mut floor_nodes = std::collections::BTreeSet::new();
    let mut factors = Vec::new();
    for (k, col) in out.cols.iter().enumerate() {
        let Some(col) = col else { continue };
        if col.shared.word != vd_bins::shore_rules::WORD_POCKET {
            continue;
        }
        let (i, j) = (out.i0 + (k % n) as i32, out.j0 + (k / n) as i32);
        let site = NodeSite::of(&fine, reader.face, r, i, j);
        floor_nodes.insert(site.nearest(&fine));
        if k % 7 == 0
            && let Some(sc) = reader.column(i, j)
        {
            factors.push(unit_of_q(factor_q(&reader.charter, sc.dir, sc.share)));
        }
    }
    let flat = floor_nodes
        .iter()
        .filter(|&&nd| {
            fine.neighbours(nd).iter().any(|&m| {
                m != vd_terrain::macro_lattice::NO_NODE
                    && nodes.z_m[m as usize] == nodes.z_m[nd as usize]
            })
        })
        .count();
    c.reading(
        &tag("FLAT-NODES"),
        &format!("{flat}of{}nodes", floor_nodes.len()),
        "a-floor-node-whose-z-equals-a-neighbour's",
    );
    c.reading(
        &tag("FLOOR-FACTOR"),
        &format!(
            "p10{}/p50{}/p90{}",
            val(quantile(&factors, 0.1), 3, ""),
            val(quantile(&factors, 0.5), 3, ""),
            val(quantile(&factors, 0.9), 3, "")
        ),
        "the-roughness-factor-m-on-the-floor(D6)",
    );
    // STRAIGHT-WALL: the ten largest neighbour steps of s.
    let mut steps: Vec<(f64, usize, bool)> = Vec::new();
    for b in 0..n {
        for a in 0..n {
            for along_j in [false, true] {
                let (na, nb) = if along_j { (a, b + 1) } else { (a + 1, b) };
                if na >= n || nb >= n {
                    continue;
                }
                let (Some(p), Some(qq)) =
                    (out.cols[b * n + a].as_ref(), out.cols[nb * n + na].as_ref())
                else {
                    continue;
                };
                steps.push((
                    f64::from((p.drawn[vi].s - qq.drawn[vi].s).abs()),
                    (b * n + a) * 2 + usize::from(along_j),
                    f64::from((p.shared.h_raw - qq.shared.h_raw).abs())
                        > 0.5 * f64::from((p.drawn[vi].s - qq.drawn[vi].s).abs()),
                ));
            }
        }
    }
    steps.sort_by(|x, y| y.0.total_cmp(&x.0));
    let cpn = i64::from(fine.cells_per_node);
    let cell = 1i64 << r;
    let line_dist = |pos0: i64, period: i64, phase: i64| -> f64 {
        let m = (pos0 - phase).rem_euclid(period);
        (m.min(period - m)) as f64 / cell as f64
    };
    for (rank, (step, at, raw_too)) in steps.iter().take(10).enumerate() {
        let (k, along_j) = (at / 2, at % 2 == 1);
        let (a, b) = (
            (k % n) as i64 + i64::from(out.i0),
            (k / n) as i64 + i64::from(out.j0),
        );
        // The pair's shared edge in rung-0 cells, across the step's own axis.
        let edge0 = if along_j {
            (b + 1) * cell
        } else {
            (a + 1) * cell
        };
        c.say(format!(
            "   STRAIGHT-WALL #{:>2} {window} r{r:02}: step {step:.1} m across {}; to the nearest fine node square edge {:.1} cols, knot line {:.1} cols; level-1 square edge {:.1} cols; the raw ground steps there too: {raw_too}",
            rank + 1,
            if along_j { "j" } else { "i" },
            line_dist(edge0, cpn, 0),
            line_dist(edge0, cpn, cpn / 2),
            line_dist(edge0, 2 * cpn, 0),
        ));
    }
    // The floor's lakes' lean (G-SHORE-LATTICE on the pocket's drawn lakes).
    let field: Vec<Option<f64>> = out
        .cols
        .iter()
        .map(|col| {
            col.as_ref()
                .map(|c| if c.drawn[vi].wet_key() > 1 { -1.0 } else { 1.0 })
        })
        .collect();
    let lines = zero_contours(n, &field);
    let axes = chord_axes(&lines, 8.0);
    let (l4, _) = four_fold(&axes);
    let null = flat_null_l4(axes.len(), 2_000, 0.95);
    let gate = tag("G-SHORE-LATTICE.floor-lakes");
    let value = format!("L4{}/{}windows", val(l4, 3, ""), axes.len());
    let band = format!("|L4|<={}[flat-null-q95]", val(null, 3, ""));
    if axes.len() < 16 {
        c.line(&gate, &value, &band, "NO-BAND");
    } else {
        c.judged(&gate, &value, &band, l4.abs() <= null);
    }
    let _ = (NVAR, Face::PosX);
}
