//! ★ SKIN-FRONT (slice K0c; the far-water design §5.2 group 2): at every front `L → L + 1` the
//! two rungs draw the same water colour. A pixel exactly at the front's distance `1.1 · s_L` is
//! read twice through ONE filter: from the finer rung's source with the level floor at its end
//! (`t = 1`), and from the coarser rung's source. A rung up to 13 reads its chunk's window; from
//! rung 14 a rung reads the far page. The pixels stand where the water changes (in the window's
//! mixed node cells) and at the finer chunk's edges, at six incidences from the nadir to 86°.
//!
//! The failing controls: (i) the coarser window's finest texel at a whole cell, not half; (ii) a
//! window margin of 12 texels at 84° and 86°; (iii) today's box fold drawn per column (255 or 0)
//! through the same filter.
//!
//! **Example.** At the 12 → 13 front (3 916 km) a pilot looks at the steep coast at 60°. The
//! pixel's footprint is 1.1 × 2.2 texels of the rung-12 window's 2 km level. With the floor at its
//! end the rung-12 chunk reads its 4 km level, the rung-13 chunk's finest, so both read the same
//! texels with the same weights and draw the same value; under the box fold the rung-12 column is
//! water and the rung-13 column land, 255 steps apart.

use std::collections::{BTreeSet, HashMap};

use vd_client::artifact_book::ArtifactCache;
use vd_client::chunks::drawn_surface_m;
use vd_client::ladder_view::LadderView;
use vd_core::glam::DVec3;
use vd_seed::bend::{Face, direction};
use vd_seed::ladder::index_of;
use vd_terrain::BodyDefinition;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};

use crate::census::Census;
use crate::pyramid::Pyramid;
use crate::sampler::{Eye, Src, sample};
use crate::skin::{R8Level, SkinPyramid, Window, cut_window, r8, window_area, window_base_level};

/// The incidences read at every front, degrees (the design's nadir, o30, o60, o80, o84, o86).
pub const VIEWS: [f64; 8] = [0.0, 30.0, 60.0, 80.0, 84.0, 86.0, 88.0, 89.0];
/// Past this incidence the 16× sampler saturates (1 / cos 86.4° = 16): the rim (the design's Q8).
pub const THETA_RIM_DEG: f64 = 86.4;
/// The finer chunks read at every front, and the pixels in each (inside and at its edges).
const CHUNKS_PER_FRONT: usize = 12;
const PIXELS_PER_CHUNK: usize = 24;
const EDGE_PIXELS_PER_CHUNK: usize = 8;

/// The skin's sources over one far window: the pyramid's cells (levels F..12), the quarters' sums
/// (12..18), and the far page.
pub struct Sources<'a> {
    pub body: &'a BodyDefinition,
    pub face: Face,
    pub f_rung: u8,
    pub pyr: &'a Pyramid,
    pub skin: &'a SkinPyramid,
    pub page: Vec<R8Level>,
    /// A window's finest texel is a cell over `2^sub`; its margin in finest texels; its levels.
    pub sub: u8,
    pub margin: i64,
    pub levels: u8,
    windows: HashMap<(ChunkKey, i64, i8, u8), Option<Window>>,
}

impl<'a> Sources<'a> {
    /// The sources of one far window; the far page is the skin's levels from the node square up.
    #[must_use]
    #[allow(clippy::too_many_arguments, reason = "the sources' own words")]
    pub fn new(
        body: &'a BodyDefinition,
        face: Face,
        f_rung: u8,
        pyr: &'a Pyramid,
        skin: &'a SkinPyramid,
        sub: u8,
        margin: i64,
        levels: u8,
    ) -> Sources<'a> {
        let page_lo = source_lo(pyr.q_rung + 2, pyr.q_rung, sub);
        let page = skin
            .levels
            .iter()
            .filter(|l| l.level >= page_lo)
            .map(|l| R8Level {
                level: l.level,
                i0: l.i0,
                j0: l.j0,
                n: l.n,
                texels: l
                    .wet
                    .iter()
                    .map(|&w| r8(u64::from(w), l.per_texel()))
                    .collect(),
            })
            .collect();
        Sources {
            body,
            face,
            f_rung,
            pyr,
            skin,
            page,
            sub,
            margin,
            levels,
            windows: HashMap::new(),
        }
    }

    /// The skin's words of texel `(i, j)` of level `l`, from the cells (F..=12) or the sums.
    #[must_use]
    pub fn count(&self, l: u8, i: i64, j: i64) -> Option<(u64, u64)> {
        if l <= self.pyr.q_rung {
            self.pyr.texel(self.f_rung, l, i, j).map(|(w, _, n)| (w, n))
        } else {
            self.skin.count(l, i, j).map(|(w, _, n)| (w, n))
        }
    }

    /// The window of `key`, cut with `margin` and `levels` levels, its finest level raised by
    /// `raise`; `None` where it leaves the keyed cells.
    pub fn window(&mut self, key: ChunkKey, margin: i64, raise: i8, levels: u8) -> Option<&Window> {
        let k = (key, margin, raise, levels);
        if !self.windows.contains_key(&k) {
            let mut missing = 0usize;
            let count = |l: u8, i: i64, j: i64| self.count(l, i, j);
            let lw = window_base_level(key.rung, self.sub);
            let win = cut_window(key, lw, levels, margin, &count, &mut missing);
            let win = if raise != 0 {
                raise_window(&win, raise, self)
            } else {
                Some(win)
            };
            let win = if missing == 0 { win } else { None };
            self.windows.insert(k, win);
        }
        self.windows.get(&k).and_then(Option::as_ref)
    }

    /// Drops the windows cut with `margin` and `levels` (the rim's search cuts wide ones).
    pub fn forget(&mut self, margin: i64, levels: u8) {
        self.windows.retain(|k, _| k.1 != margin || k.3 != levels);
    }
}

/// A window whose finest level is dropped `raise` times (SKIN-FRONT's control i: the finest texel
/// a whole cell), refilled from the counts at the coarser levels.
fn raise_window(win: &Window, raise: i8, src: &Sources) -> Option<Window> {
    let r = raise as u8;
    let levels: Vec<R8Level> = win
        .levels
        .iter()
        .map(|lv| {
            let l = lv.level + r;
            let (i0, j0, n) = (lv.i0 >> r, lv.j0 >> r, (lv.n >> r).max(1));
            let mut texels = vec![0u8; (n * n) as usize];
            for b in 0..n {
                for a in 0..n {
                    if let Some((w, c)) = src.count(l, i0 + a, j0 + b) {
                        texels[(b * n + a) as usize] = r8(w, c);
                    }
                }
            }
            R8Level {
                level: l,
                i0,
                j0,
                n,
                texels,
            }
        })
        .collect();
    Some(Window {
        key: win.key,
        levels,
    })
}

/// The finest level a rung reads: its window's up to the node square's rung (a cell over
/// `2^sub`), the far page's from the next rung (the node square for `sub = 1`, the quarter
/// square for `sub = 2`).
#[must_use]
pub fn source_lo(rung: u8, q_rung: u8, sub: u8) -> u8 {
    if rung > q_rung + 1 {
        (q_rung + 2).saturating_sub(sub)
    } else {
        window_base_level(rung, sub)
    }
}

/// A deterministic stream of fractions.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// ★ THE FRONT LINES of one far window, every front `lo_rung → lo_rung + 1 … 13 → 14`.
pub fn front_lines(c: &mut Census, tag: &str, src: &mut Sources, sea_r: f64, lo_rung: u8) {
    let body = src.body;
    let q = src.pyr.q_rung;
    let alpha = vd_client::ladder_view::pixel_rad();
    let mixed: Vec<(i32, i32)> = {
        let mut v: Vec<(i32, i32)> = src.pyr.mixed_classes.keys().copied().collect();
        v.sort_unstable();
        v
    };
    if mixed.is_empty() {
        c.say(format!(
            "   SKIN-FRONT.{tag}: no mixed node cell, nothing to read"
        ));
        return;
    }
    let per_node = f64::from(src.pyr.per_node);
    let n0 = f64::from(body.ladder().cells_per_edge(src.f_rung));
    for l in lo_rung..=q + 5 {
        let d_front = 1.1 * vd_client::ladder_view::switch_m(l);
        let (lo_f, lo_c) = (source_lo(l, q, src.sub), source_lo(l + 1, q, src.sub));
        let floor = f64::from(lo_c - lo_f);
        let mut state = 0x5eed_0000_u64 + u64::from(l);
        let mut worst = [[0.0f64; 2]; VIEWS.len()];
        let mut ctl_i = [0.0f64; VIEWS.len()];
        let mut ctl_ii = [0.0f64; VIEWS.len()];
        let mut ctl_iii = [0.0f64; VIEWS.len()];
        let mut read = [0usize; VIEWS.len()];
        let mut rim_hits: Vec<Vec<RimHit>> = vec![Vec::new(); VIEWS.len()];
        for _ in 0..CHUNKS_PER_FRONT {
            // A mixed node cell, and the finer chunk over it.
            let (ni, nj) = mixed[(lcg(&mut state) * mixed.len() as f64) as usize % mixed.len()];
            let (sx, sy) = (
                f64::from(ni) * per_node + 0.5 * per_node + lcg(&mut state) * per_node,
                f64::from(nj) * per_node + 0.5 * per_node + lcg(&mut state) * per_node,
            );
            let (f_rung, face) = (src.f_rung, src.face);
            let at_rung = |r: u8, s: f64| (s * 2f64.powi(i32::from(f_rung) - i32::from(r))) as i32;
            let key_of = |r: u8, x: f64, y: f64| ChunkKey {
                face,
                rung: r,
                x: at_rung(r, x).div_euclid(CHUNK_EDGE as i32),
                y: at_rung(r, y).div_euclid(CHUNK_EDGE as i32),
                z: 0,
            };
            let key_f = key_of(l, sx, sy);
            // The chunk's own span in samples, for its edge pixels.
            let span = CHUNK_EDGE as f64 * 2f64.powi(i32::from(l) - i32::from(f_rung));
            let (cx0, cy0) = (f64::from(key_f.x) * span, f64::from(key_f.y) * span);
            let mut points: Vec<(f64, f64)> =
                Vec::with_capacity(PIXELS_PER_CHUNK + EDGE_PIXELS_PER_CHUNK);
            for _ in 0..PIXELS_PER_CHUNK {
                points.push((
                    sx + (lcg(&mut state) - 0.5) * per_node,
                    sy + (lcg(&mut state) - 0.5) * per_node,
                ));
            }
            for e in 0..EDGE_PIXELS_PER_CHUNK {
                let along = lcg(&mut state) * span;
                let off = (lcg(&mut state) - 0.5) * 2.0;
                points.push(match e % 4 {
                    0 => (cx0 + off, cy0 + along),
                    1 => (cx0 + span + off, cy0 + along),
                    2 => (cx0 + along, cy0 + off),
                    _ => (cx0 + along, cy0 + span + off),
                });
            }
            for &(x, y) in &points {
                let (a, b) = (2.0 * x / n0 - 1.0, 2.0 * y / n0 - 1.0);
                if a.abs() >= 1.0 || b.abs() >= 1.0 {
                    continue;
                }
                let dir = direction(src.face, a, b);
                let p = DVec3::new(dir[0], dir[1], dir[2]) * sea_r;
                for (vi, &deg) in VIEWS.iter().enumerate() {
                    let az = lcg(&mut state) * std::f64::consts::TAU;
                    let eye = Eye::looking_at(p, d_front, deg.to_radians(), az, alpha);
                    let Some(hit) = eye.footprint(0.0, 0.0, sea_r) else {
                        continue;
                    };
                    let (hx, hy) = (index_of(hit.a, n0 as u32), index_of(hit.b, n0 as u32));
                    let (kx, ky) = (f64::from(hx) + 0.5, f64::from(hy) + 0.5);
                    let key_f = key_of(l, kx, ky);
                    let key_c = key_of(l + 1, kx, ky);
                    if deg > THETA_RIM_DEG {
                        rim_hits[vi].push((hit, key_f, key_c));
                    }
                    for (quant, slot) in [(false, 0usize), (true, 1usize)] {
                        let Some(vf) = read_rung(src, l, key_f, &hit, floor, quant, src.margin, 0)
                        else {
                            continue;
                        };
                        let Some(vc) =
                            read_rung(src, l + 1, key_c, &hit, 0.0, quant, src.margin, 0)
                        else {
                            continue;
                        };
                        worst[vi][slot] = worst[vi][slot].max((vf - vc).abs());
                        if slot == 0 {
                            read[vi] += 1;
                            // Control (i): the coarser source's finest texel at a whole cell.
                            if let Some(ci) =
                                read_rung(src, l + 1, key_c, &hit, 0.0, quant, src.margin, 1)
                            {
                                ctl_i[vi] = ctl_i[vi].max((vf - ci).abs());
                            }
                            // Control (ii): a 12-texel margin on both windows.
                            if deg >= 84.0 {
                                let (Some(f12), Some(c12)) = (
                                    read_rung(src, l, key_f, &hit, floor, quant, 12, 0),
                                    read_rung(src, l + 1, key_c, &hit, 0.0, quant, 12, 0),
                                ) else {
                                    continue;
                                };
                                ctl_ii[vi] =
                                    ctl_ii[vi].max((f12 - c12).abs().max((f12 - vf).abs()));
                            }
                            // Control (iii): the box fold's water per column through the filter.
                            let (bf, bc) = (box_read(src, l, &hit), box_read(src, l + 1, &hit));
                            if let (Some(bf), Some(bc)) = (bf, bc) {
                                ctl_iii[vi] = ctl_iii[vi].max((bf - bc).abs());
                            }
                        }
                    }
                }
            }
        }
        for (vi, &deg) in VIEWS.iter().enumerate() {
            let f64_steps = worst[vi][0];
            let q8_steps = worst[vi][1];
            let gate = format!("SKIN-FRONT.{tag}.r{l:02}-r{:02}.o{deg:02.0}", l + 1);
            let value = format!(
                "{f64_steps:.3}steps(f64)/{q8_steps:.3}steps(8-bit)/{}pixels;controls:whole-cell{:.1}/margin12{:.1}/box{:.0}",
                read[vi], ctl_i[vi], ctl_ii[vi], ctl_iii[vi]
            );
            let band = "0-steps-f64,<=1-step-8-bit[design-D§3.1-P1]";
            if read[vi] == 0 {
                // At this front's distance the view's footprint leaves the sphere (the limb).
                c.line(
                    &gate,
                    &format!("{value};the-footprint-leaves-the-limb"),
                    band,
                    "NO-BAND",
                );
            } else if deg > THETA_RIM_DEG {
                // THE RIM (Q8): past θ_rim the sampler saturates; a reading, never a verdict, with
                // the margin that closes it (the design's "the margin that would make it 0").
                let closes = if !RIM_SEARCH.load(std::sync::atomic::Ordering::Relaxed) {
                    "no-search".to_string()
                } else if f64_steps < 0.5 {
                    format!("closed-at-margin{}-{}levels", src.margin, src.levels)
                } else {
                    let base = src.levels;
                    (base..=base + 2)
                        .map(|lv| match rim_margin(src, l, floor, &rim_hits[vi], lv) {
                            (Some(m), _) => format!("{lv}levels:closed-at-margin{m}"),
                            (None, last) => format!(
                                "{lv}levels:open-at-margin{}({last:.1}steps)",
                                src.margin * RIM_MARGIN_HALVES[RIM_MARGIN_HALVES.len() - 1] / 2
                            ),
                        })
                        .collect::<Vec<_>>()
                        .join("/")
                };
                c.reading(
                    &gate.replace("SKIN-FRONT", "SKIN-RIM"),
                    &format!("{value};{closes}"),
                    "a-reading-past-theta-rim-86.4deg[design-D-Q8]",
                );
                // SKIN-RIM-PAGE: past θ_rim BOTH rungs read the one far page (equal by
                // construction); its price is the page's value against the ideal one (the one
                // pyramid, no window), beside the finer window's own distance from the ideal.
                let (mut page_d, mut win_d) = (Vec::new(), 0.0f64);
                for (hit, kf, _) in rim_hits[vi].clone() {
                    let Some(r) = reference_read(src, &hit, false) else {
                        continue;
                    };
                    if let Some(p) = read_rung(src, q + 2, kf, &hit, 0.0, false, src.margin, 0) {
                        page_d.push((p - r).abs());
                    }
                    if let Some(w) = read_rung(src, l, kf, &hit, floor, false, src.margin, 0) {
                        win_d = win_d.max((w - r).abs());
                    }
                }
                if !page_d.is_empty() {
                    page_d.sort_by(f64::total_cmp);
                    let p99 = page_d[((page_d.len() as f64 * 0.99) as usize).min(page_d.len() - 1)];
                    let mean = page_d.iter().sum::<f64>() / page_d.len() as f64;
                    c.reading(
                        &gate.replace("SKIN-FRONT", "SKIN-RIM-PAGE"),
                        &format!(
                            "the-page-against-the-ideal:mean{mean:.1}/p99{p99:.1}/max{:.1}steps;the-finer-window-against-the-ideal:max{win_d:.1}steps;{}pixels",
                            page_d[page_d.len() - 1],
                            page_d.len()
                        ),
                        "the-price-of-reading-the-far-page-past-theta-rim[design-D-Q8]",
                    );
                }
            } else {
                // From 14 -> 15 up both rungs read ONE far page: the equality is structural, and
                // the whole-cell control reads the same coarse levels, so it is only a reading.
                // At 80° and more the filter reads coarse levels only, so the whole-cell control
                // cannot change a value there: its power is required where the finest level is read.
                let control_read = l > q + 1 || deg >= 80.0;
                c.judged(
                    &gate,
                    &value,
                    band,
                    f64_steps < 0.5 && q8_steps <= 1.0 && (control_read || ctl_i[vi] > 0.5),
                );
            }
        }
    }
}

/// One rim pixel: its hit, and the finer and the coarser chunk over it.
type RimHit = (crate::sampler::Hit, ChunkKey, ChunkKey);

/// The margins the rim's search tries, in halves of the run's margin (1×, 1.5×, 2×, 3×, 4×, 6×).
const RIM_MARGIN_HALVES: [i64; 6] = [2, 3, 4, 6, 8, 12];

/// ★ THE MARGIN THAT CLOSES THE RIM (Q8): with windows of `levels` levels, the smallest margin of
/// [`RIM_MARGIN_HALVES`] at which every rim pixel of the front reads under half a step (f64
/// weights), and the largest step at the last margin tried.
fn rim_margin(
    src: &mut Sources,
    l: u8,
    floor: f64,
    hits: &[RimHit],
    levels: u8,
) -> (Option<i64>, f64) {
    let mut last = f64::NAN;
    for k in RIM_MARGIN_HALVES {
        let m = src.margin * k / 2;
        let mut worst = 0.0f64;
        for (hit, kf, kc) in hits {
            let (Some(vf), Some(vc)) = (
                read_rung_with(src, l, *kf, hit, floor, false, m, 0, levels),
                read_rung_with(src, l + 1, *kc, hit, 0.0, false, m, 0, levels),
            ) else {
                continue;
            };
            worst = worst.max((vf - vc).abs());
        }
        if m != src.margin || levels != src.levels {
            src.forget(m, levels);
        }
        last = worst;
        if worst < 0.5 {
            return (Some(m), worst);
        }
    }
    (None, last)
}

/// ★ ONE RUNG'S READ of a hit: its source (its chunk's window up to rung 13, the far page from
/// rung 14), the hit in that source's finest texels, the filter.
#[allow(clippy::too_many_arguments, reason = "the read's own words")]
fn read_rung(
    src: &mut Sources,
    rung: u8,
    key: ChunkKey,
    hit: &crate::sampler::Hit,
    floor: f64,
    quant: bool,
    margin: i64,
    raise: i8,
) -> Option<f64> {
    let levels = src.levels;
    read_rung_with(src, rung, key, hit, floor, quant, margin, raise, levels)
}

/// [`read_rung`] with windows of `levels` levels.
#[allow(clippy::too_many_arguments, reason = "the read's own words")]
fn read_rung_with(
    src: &mut Sources,
    rung: u8,
    key: ChunkKey,
    hit: &crate::sampler::Hit,
    floor: f64,
    quant: bool,
    margin: i64,
    raise: i8,
    levels: u8,
) -> Option<f64> {
    let q = src.pyr.q_rung;
    let body = src.body;
    if rung > q + 1 {
        // The far page; control (i) reads it with its finest level dropped `raise` times.
        let lo = source_lo(rung, q, src.sub) + raise as u8;
        let n_l = f64::from(body.ladder().cells_per_edge(lo));
        let (pos, dx, dy) = hit.in_texels(n_l);
        let page = &src.page;
        let fetch = |l: u8, i: i64, j: i64| -> u8 {
            page.iter()
                .find(|p| p.level == l)
                .map_or(0, |p| p.clamped(i, j))
        };
        let hi = page.last().map_or(lo, |p| p.level);
        let s = Src {
            lo,
            hi,
            fetch: &fetch,
        };
        return Some(sample(&s, pos, dx, dy, floor, quant));
    }
    let win = src.window(key, margin, raise, levels)?.clone();
    let lo = win.levels.first()?.level;
    let hi = win.levels.last()?.level;
    let n_l = f64::from(body.ladder().cells_per_edge(lo));
    let (pos, dx, dy) = hit.in_texels(n_l);
    let fetch = |l: u8, i: i64, j: i64| -> u8 {
        win.levels
            .iter()
            .find(|p| p.level == l)
            .map_or(0, |p| p.clamped(i, j))
    };
    let s = Src {
        lo,
        hi,
        fetch: &fetch,
    };
    let floor = floor - f64::from(raise);
    Some(sample(&s, pos, dx, dy, floor.max(0.0), quant))
}

/// Today's box fold drawn per column: 255 where the rung's column's one-rule fold is wet, else 0,
/// read through the same filter at the column's own grid (control iii).
fn box_read(src: &Sources, rung: u8, hit: &crate::sampler::Hit) -> Option<f64> {
    let n_l = f64::from(src.body.ladder().cells_per_edge(rung));
    let (pos, dx, dy) = hit.in_texels(n_l);
    let fetch = |l: u8, i: i64, j: i64| -> u8 {
        let _ = l;
        match src.count(rung, i, j) {
            Some((w, n)) if 2 * w >= n && w > 0 => 255,
            _ => 0,
        }
    };
    let s = Src {
        lo: rung,
        hi: rung,
        fetch: &fetch,
    };
    Some(sample(&s, pos, dx, dy, 0.0, false))
}

/// ★ ONE EMULATED CLIMB: the eye straight over a point (rung-F sample coordinates) from `h0` to
/// `h1` metres, one frame per half per cent of altitude, a square of pixels about the nadir.
pub struct Climb {
    pub name: &'static str,
    pub at: (f64, f64),
    pub h0: f64,
    pub h1: f64,
    pub half: i32,
    pub step: i32,
}

/// ★ SKIN-TRACK (the design §5.2 group 3): every frame and pixel of a climb, the drawn value (the
/// rung the ladder draws at the pixel's distance, through its own source, with the band's level
/// floor) against the reference (the same filter on the ONE pyramid, with no rung, no window and
/// no floor). The control: today's box fold drawn per column at the same rung, against the same
/// reference. SKIN-LAKE-PIECE: the nadir pixel's value along the climb, beside the box fold's.
pub fn track_lines(c: &mut Census, tag: &str, src: &mut Sources, sea_r: f64, climb: &Climb) {
    let body = src.body;
    let q = src.pyr.q_rung;
    let f = src.f_rung;
    let alpha = vd_client::ladder_view::pixel_rad();
    let n0 = f64::from(body.ladder().cells_per_edge(f));
    let (a, b) = (2.0 * climb.at.0 / n0 - 1.0, 2.0 * climb.at.1 / n0 - 1.0);
    let dir = direction(src.face, a, b);
    let p = DVec3::new(dir[0], dir[1], dir[2]) * sea_r;
    let top = src.skin.levels.last().map_or(q, |l| l.level);
    let (mut worst, mut worst_q8, mut ctl, mut pixels, mut frames) =
        (0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
    let mut rounding = 0.0f64;
    let mut worst_at = String::new();
    let mut piece: Vec<(f64, f64, f64, u8)> = Vec::new();
    let mut h = climb.h0;
    while h <= climb.h1 {
        let eye = Eye::over(p, h, alpha);
        frames += 1;
        for y in (-climb.half..=climb.half).step_by(climb.step as usize) {
            for x in (-climb.half..=climb.half).step_by(climb.step as usize) {
                let Some(hit) = eye.footprint(f64::from(x), f64::from(y), sea_r) else {
                    continue;
                };
                // The drawn rung: the finest whose front the pixel has not passed.
                let mut l = 0u8;
                while l < top && hit.d >= 1.1 * vd_client::ladder_view::switch_m(l) {
                    l += 1;
                }
                let s_l = vd_client::ladder_view::switch_m(l);
                let t = ((hit.d - 0.9 * s_l) / (0.2 * s_l)).clamp(0.0, 1.0);
                let floor = t * f64::from(
                    source_lo(l + 1, q, src.sub).saturating_sub(source_lo(l, q, src.sub)),
                );
                let (hx, hy) = (index_of(hit.a, n0 as u32), index_of(hit.b, n0 as u32));
                let key = ChunkKey {
                    face: src.face,
                    rung: l,
                    x: ((f64::from(hx) + 0.5) * 2f64.powi(i32::from(f) - i32::from(l))) as i32
                        / CHUNK_EDGE as i32,
                    y: ((f64::from(hy) + 0.5) * 2f64.powi(i32::from(f) - i32::from(l))) as i32
                        / CHUNK_EDGE as i32,
                    z: 0,
                };
                let (Some(drawn), Some(drawn_q8)) = (
                    read_rung(src, l, key, &hit, floor, false, src.margin, 0),
                    read_rung(src, l, key, &hit, floor, true, src.margin, 0),
                ) else {
                    continue;
                };
                let (Some(reference), Some(reference_q8)) = (
                    reference_read(src, &hit, false),
                    reference_read(src, &hit, true),
                ) else {
                    continue;
                };
                pixels += 1;
                rounding = rounding.max((drawn_q8 - reference).abs());
                if (drawn - reference).abs() > worst {
                    let n_lo = f64::from(body.ladder().cells_per_edge(source_lo(l, q, src.sub)));
                    let (_, dx, dy) = hit.in_texels(n_lo);
                    let (lx, ly) = (dx.0.hypot(dx.1), dy.0.hypot(dy.1));
                    let (pmax, pmin) = (lx.max(ly), lx.min(ly));
                    let n = (pmax / pmin.max(1e-12)).ceil().clamp(1.0, 16.0);
                    worst_at = format!(
                        "h{:.0}km px({x},{y}) d{:.0}km rung{l} t{t:.2} floor{floor:.2} lambda-own{:.2} (pmax{pmax:.2},pmin{pmin:.2},N{n}) drawn{drawn:.1} ref{reference:.1}",
                        h / 1_000.0,
                        hit.d / 1_000.0,
                        (pmax / n).log2()
                    );
                }
                worst = worst.max((drawn - reference).abs());
                worst_q8 = worst_q8.max((drawn_q8 - reference_q8).abs());
                if let Some(bx) = box_read(src, l.max(f + 1), &hit) {
                    ctl = ctl.max((bx - reference).abs());
                }
                if x == 0 && y == 0 {
                    piece.push((
                        h,
                        drawn,
                        box_read(src, l.max(f + 1), &hit).unwrap_or(f64::NAN),
                        l,
                    ));
                }
            }
        }
        h *= 1.005;
    }
    c.say(format!("   SKIN-TRACK worst pixel: {worst_at}"));
    c.judged(
        &format!("SKIN-TRACK.{tag}.{}", climb.name),
        &format!(
            "{worst:.3}steps(f64)/{worst_q8:.3}steps(8-bit)/{pixels}pixels/{frames}frames;control-box-fold:{ctl:.0}steps;the-8-bit-weights'-own-rounding:{rounding:.3}steps"
        ),
        "0-steps-f64,<=1-step-8-bit;control>0[design-D§3.1-P2]",
        pixels > 0 && worst < 0.5 && worst_q8 <= 1.0 && ctl > 0.5,
    );
    // SKIN-LAKE-PIECE: the nadir's value at a few altitudes, and at every swap it crosses.
    let mut line = String::new();
    let mut last_rung = u8::MAX;
    for (k, &(h, v, bx, l)) in piece.iter().enumerate() {
        if k % 40 == 0 || l != last_rung {
            line.push_str(&format!(
                "{:.0}km:r{l}:{:.0}%/box{:.0}; ",
                h / 1_000.0,
                100.0 * v / 255.0,
                100.0 * bx / 255.0
            ));
        }
        last_rung = l;
    }
    c.reading(
        &format!("SKIN-LAKE-PIECE.{tag}.{}", climb.name),
        line.trim_end(),
        "the-nadir-pixel's-share-along-the-climb,beside-today's-box-fold[design-D-Q1]",
    );
}

/// ★ THE REFERENCE READ: the filter on the ONE pyramid (every level from F up to the top, no rung,
/// no window, no floor), at the hit.
fn reference_read(src: &Sources, hit: &crate::sampler::Hit, quant: bool) -> Option<f64> {
    let lo = src.f_rung;
    let hi = src.skin.levels.last()?.level;
    let n_l = f64::from(src.body.ladder().cells_per_edge(lo));
    let (pos, dx, dy) = hit.in_texels(n_l);
    // Never read past the pyramid: the value at a texel it does not hold is refused.
    let missing = std::cell::Cell::new(false);
    let fetch = |l: u8, i: i64, j: i64| -> u8 {
        match src.count(l, i, j) {
            Some((w, n)) => r8(w, n),
            None => {
                missing.set(true);
                0
            }
        }
    };
    let s = Src {
        lo,
        hi,
        fetch: &fetch,
    };
    let v = sample(&s, pos, dx, dy, 0.0, quant);
    (!missing.get()).then_some(v)
}

/// ★ PHI-HANDOVER (the design §5.2 group 4, Q5): between 55.6 and 111.2 km the crisp sheet hands
/// the coast over to the skin by the pixel's footprint. Per pixel of a nadir view over `at`, the
/// sheet's coverage (four sub-pixel samples, as 4× MSAA, each the class of its rung-F sample: the
/// sheet drawn as the samples' squares) against the skin's value, by altitude: the mean and the
/// p99 of their difference, and φ, the skin's weight there.
pub fn handover_lines(c: &mut Census, tag: &str, src: &mut Sources, sea_r: f64, at: (f64, f64)) {
    let body = src.body;
    let q = src.pyr.q_rung;
    let f = src.f_rung;
    let alpha = vd_client::ladder_view::pixel_rad();
    let n0 = f64::from(body.ladder().cells_per_edge(f));
    let dir = direction(src.face, 2.0 * at.0 / n0 - 1.0, 2.0 * at.1 / n0 - 1.0);
    let p = DVec3::new(dir[0], dir[1], dir[2]) * sea_r;
    let t0 = f64::from(1u32 << f);
    for h_km in [50.0f64, 56.0, 65.0, 75.0, 85.0, 95.0, 105.0, 111.0, 120.0] {
        let eye = Eye::over(p, h_km * 1e3, alpha);
        let mut diffs: Vec<f64> = Vec::new();
        let mut phi_sum = 0.0;
        for y in (-16..=16).step_by(2) {
            for x in (-16..=16).step_by(2) {
                let (fx, fy) = (f64::from(x), f64::from(y));
                let Some(hit) = eye.footprint(fx, fy, sea_r) else {
                    continue;
                };
                let mut l = 0u8;
                while l < q + 6 && hit.d >= 1.1 * vd_client::ladder_view::switch_m(l) {
                    l += 1;
                }
                let s_l = vd_client::ladder_view::switch_m(l);
                let t = ((hit.d - 0.9 * s_l) / (0.2 * s_l)).clamp(0.0, 1.0);
                let floor = t * f64::from(
                    source_lo(l + 1, q, src.sub).saturating_sub(source_lo(l, q, src.sub)),
                );
                let (hx, hy) = (index_of(hit.a, n0 as u32), index_of(hit.b, n0 as u32));
                let key = ChunkKey {
                    face: src.face,
                    rung: l,
                    x: ((f64::from(hx) + 0.5) * 2f64.powi(i32::from(f) - i32::from(l))) as i32
                        / CHUNK_EDGE as i32,
                    y: ((f64::from(hy) + 0.5) * 2f64.powi(i32::from(f) - i32::from(l))) as i32
                        / CHUNK_EDGE as i32,
                    z: 0,
                };
                let margin = src.margin;
                let Some(skin) = read_rung(src, l, key, &hit, floor, false, margin, 0) else {
                    continue;
                };
                // The sheet: four sub-pixel samples, each its rung-F sample's class.
                let mut wet = 0.0;
                let mut n = 0.0;
                for (ox, oy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                    if let Some(sp) = eye.footprint(fx + ox, fy + oy, sea_r) {
                        let (sx, sy) = (index_of(sp.a, n0 as u32), index_of(sp.b, n0 as u32));
                        if let Some(cl) = src.pyr.class_at(i64::from(sx), i64::from(sy)) {
                            wet += f64::from(u8::from(cl > 0));
                            n += 1.0;
                        }
                    }
                }
                if n == 0.0 {
                    continue;
                }
                let sheet = 255.0 * wet / n;
                diffs.push((sheet - skin).abs() / 255.0);
                let px = hit.d * alpha;
                phi_sum += (2.0 * px / t0 - 1.0).clamp(0.0, 1.0);
            }
        }
        if diffs.is_empty() {
            continue;
        }
        let count = diffs.len() as f64;
        let mean = diffs.iter().sum::<f64>() / count;
        diffs.sort_by(f64::total_cmp);
        let p99 = diffs[((count * 0.99) as usize).min(diffs.len() - 1)];
        c.reading(
            &format!("PHI-HANDOVER.{tag}.h{h_km:03.0}km"),
            &format!(
                "mean{mean:.3}/p99{p99:.3}(|sheet-skin|,share);phi{:.2};{}pixels",
                phi_sum / count,
                diffs.len()
            ),
            "the-crisp-sheet-against-the-skin-in-the-hand-over-band[design-D-Q5]",
        );
    }
}

/// ★ SKIN-MB (the design §5.2 group 7): a window's bytes at each rung (R8 with its mips, a chunk
/// away from a face edge) at this run's resolution and margin, the window's mixed chunks in the
/// window's area, and the far page's bytes a face.
pub fn memory_lines(c: &mut Census, tag: &str, src: &mut Sources) {
    let q = src.pyr.q_rung;
    let f = src.f_rung;
    let per_node = i64::from(src.pyr.per_node);
    for l in 6..=q + 1 {
        let lw = window_base_level(l, src.sub);
        // A chunk's side in finest texels, the margin, the alignment to the coarsest level.
        let side = (CHUNK_EDGE as i64) << (l.saturating_sub(lw));
        let side = if l < lw {
            (CHUNK_EDGE as i64) >> (lw - l)
        } else {
            side
        };
        let align = 1i64 << (src.levels - 1);
        let n = ((side + 2 * src.margin + align - 1) / align + 1) * align;
        let bytes: f64 = (0..src.levels).map(|k| ((n >> k) * (n >> k)) as f64).sum();
        // The mixed chunks of this rung inside the window: the chunks that hold a mixed cell.
        let mut chunks = std::collections::BTreeSet::new();
        for &(ni, nj) in src.pyr.mixed_classes.keys() {
            let (sx, sy) = (
                i64::from(ni) * per_node + per_node,
                i64::from(nj) * per_node + per_node,
            );
            let shift = i32::from(l) - i32::from(f);
            let (cx, cy) = if shift >= 0 {
                (sx >> shift, sy >> shift)
            } else {
                (sx << -shift, sy << -shift)
            };
            chunks.insert((
                cx.div_euclid(CHUNK_EDGE as i64),
                cy.div_euclid(CHUNK_EDGE as i64),
            ));
        }
        c.reading(
            &format!("SKIN-MB.{tag}.r{l:02}"),
            &format!(
                "{:.0}KB-a-window({n}x{n}-texels-of-level-{lw})/{}mixed-chunks-in-the-window/{:.1}MB-for-them",
                bytes / 1024.0,
                chunks.len(),
                bytes * chunks.len() as f64 / 1_048_576.0
            ),
            "the-window's-memory-at-this-resolution[design-D-Q8]",
        );
    }
    let page_lo = source_lo(q + 2, q, src.sub);
    let edge = src.body.ladder().cells_per_edge(page_lo) as f64;
    c.reading(
        &format!("SKIN-MB.{tag}.far-page"),
        &format!(
            "{:.1}MB-a-face/{:.1}MB-the-planet(levels-{page_lo}-up,R8-with-mips,no-border)",
            edge * edge * 4.0 / 3.0 / 1_048_576.0,
            6.0 * edge * edge * 4.0 / 3.0 / 1_048_576.0
        ),
        "the-far-page's-memory[design-D-Q8]",
    );
}

/// The heights over a stand at which SKIN-MB-LADDER counts the windows, metres.
const LADDER_HEIGHTS_M: [f64; 7] = [2.0e3, 10.0e3, 50.0e3, 150.0e3, 500.0e3, 2.0e6, 8.0e6];
/// The window forms SKIN-MB-LADDER prices: (a finest texel of a cell over `2^sub`, the margin in
/// finest texels, the levels) — the design's form (seamless to 86.4° under the Vulkan level rule),
/// its two wider forms for the rim (88° and 89°), and the quarter-cell form the GL rule needs.
const LADDER_FORMS: [(u8, i64, u8); 4] = [(1, 22, 5), (1, 44, 5), (1, 88, 5), (2, 52, 5)];

/// Whether the rim's margin search runs (`--no-rim-search` turns it off for a run that only
/// draws the panels or prices the memory).
pub static RIM_SEARCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// ★ SKIN-MB-LADDER (the design §5.2 group 7, Q8): the windows a pilot's own chunk set needs. At each
/// height over `at` (rung-F sample coordinates), the client's own descent over the drawn ground
/// gives the wanted set; every wanted column of rungs 6..=13 on this face whose window area holds
/// both water and land needs a window (an all-water or all-land window is a constant), and its
/// bytes are its five levels' texels at one byte each. Priced for both [`LADDER_FORMS`].
pub fn ladder_memory_lines(
    c: &mut Census,
    tag: &str,
    src: &Sources,
    cache: &ArtifactCache,
    at: (f64, f64),
) {
    let body = src.body;
    let q = src.pyr.q_rung;
    let n0 = f64::from(body.ladder().cells_per_edge(src.f_rung));
    let d = direction(src.face, 2.0 * at.0 / n0 - 1.0, 2.0 * at.1 / n0 - 1.0);
    let dir = DVec3::new(d[0], d[1], d[2]).normalize();
    let ground = drawn_surface_m(body, Some(cache), dir.to_array());
    for h in LADDER_HEIGHTS_M {
        let eye = dir * (ground + h);
        let mut view = LadderView::default();
        let wanted = view.wanted(body, eye.to_array(), Some(cache));
        let window_rungs = 6..=q + 1;
        let cols: BTreeSet<(u8, i32, i32)> = wanted
            .keys
            .iter()
            .filter(|k| k.face == src.face && window_rungs.contains(&k.rung))
            .map(|k| (k.rung, k.x, k.y))
            .collect();
        let other_face = wanted
            .keys
            .iter()
            .filter(|k| k.face != src.face && window_rungs.contains(&k.rung))
            .map(|k| (k.face, k.rung, k.x, k.y))
            .collect::<BTreeSet<_>>()
            .len();
        let mut parts = Vec::new();
        for (sub, margin, n_levels) in LADDER_FORMS {
            let top = n_levels - 1;
            let (mut mixed, mut uniform, mut unread, mut bytes) = (0usize, 0usize, 0usize, 0u64);
            for &(rung, x, y) in &cols {
                let key = ChunkKey {
                    face: src.face,
                    rung,
                    x,
                    y,
                    z: 0,
                };
                let lw = window_base_level(rung, sub);
                let (i0, j0, n) = window_area(key, lw, n_levels, margin);
                let (ti0, tj0, tn) = (i0 >> top, j0 >> top, (n >> top).max(1));
                let (mut wet, mut all, mut read) = (0u64, 0u64, true);
                for b in 0..tn {
                    for a in 0..tn {
                        match src.count(lw + top, ti0 + a, tj0 + b) {
                            Some((w, s)) => {
                                wet += w;
                                all += s;
                            }
                            None => read = false,
                        }
                    }
                }
                if !read {
                    unread += 1;
                } else if wet == 0 || wet == all {
                    uniform += 1;
                } else {
                    mixed += 1;
                    bytes += (0..n_levels)
                        .map(|k| {
                            let side = (n >> k) as u64;
                            side * side
                        })
                        .sum::<u64>();
                }
            }
            parts.push(format!(
                "cell/{}+margin{margin}+{n_levels}levels:{mixed}windows={:.1}MB({uniform}constant,{unread}unread)",
                1u32 << sub,
                bytes as f64 / 1_048_576.0
            ));
        }
        c.reading(
            &format!("SKIN-MB-LADDER.{tag}.h{:05.0}km", h / 1e3),
            &format!(
                "{}columns-r6..r{}-on-the-face:{};{other_face}on-other-faces",
                cols.len(),
                q + 1,
                parts.join("/")
            ),
            "the-windows-a-pilot's-chunk-set-needs[design-D-Q8]",
        );
    }
}
