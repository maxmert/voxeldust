//! ★ THE SKIN'S EXACTNESS LINES (slice K0, group 1 of the far-water design §5.2), each with its
//! failing control:
//! * SKIN-NEST — every texel's words against the sums of its four children, counted again here;
//!   control: one child's wet count plus one.
//! * SKIN-GRID — every texel of every window a client cuts (its five levels and its margins),
//!   filled from the base counts directly, against the one pyramid's texel at the same level and
//!   place; control: a window cut one texel off its alignment.
//! * SKIN-QUANT — a window's coarser levels rounded once from their own counts, against mips
//!   averaged from rounded children (what a GPU's mip generator makes); the control IS that
//!   average, which must differ.
//! * AREA-NEST.skin — the wet samples of every coarser texel against its four children's: exactly
//!   nought, beside the box fold's measured areas.
//!
//! **Example.** On the high lake's far window the rung-13 chunk over the 5 882 m lake cuts a window
//! of level-12 texels (4 km) with 22 texels of margin on each side, aligned to 64 km. Each of its
//! 128 × 128 finest texels reads the same 8-bit value as the one pyramid's texel at the same
//! place, and a window cut one texel off reads another value at the lake's edge.

use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};

use crate::census::Census;
use crate::skin::{
    Level, SkinPyramid, WINDOW_LEVELS, Window, cut_window, direct_count, r8, window_base_level,
};

/// The margin on each side of a window, in its finest texels (the design §2.3 item 7: at
/// θ_rim = 86.4° the 16× sampler's half footprint at a front plus one coarsest texel).
pub const WINDOW_MARGIN: i64 = 22;

/// ★ THE EXACTNESS LINES of one pyramid and the windows of rungs `rungs` cut inside it.
pub fn exactness(
    c: &mut Census,
    tag: &str,
    pyr: &SkinPyramid,
    rungs: std::ops::RangeInclusive<u8>,
) {
    let Some(base) = pyr.levels.first() else {
        return;
    };
    // SKIN-NEST, and its control.
    let mut nest_differ = 0u64;
    let mut texels = 0u64;
    for pair in pyr.levels.windows(2) {
        let (lo, hi) = (&pair[0], &pair[1]);
        for b in 0..hi.n {
            for a in 0..hi.n {
                let (i, j) = (hi.i0 + a, hi.j0 + b);
                let (pw, pl) = hi.get(i, j).unwrap_or_default();
                let (mut sw, mut sl) = (0u32, 0u32);
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (w, l) = lo.get(2 * i + da, 2 * j + db).unwrap_or_default();
                    sw += w;
                    sl += l;
                }
                texels += 1;
                nest_differ += u64::from(pw != sw || pl != sl);
            }
        }
    }
    let control = nest_control(pyr);
    c.judged(
        &format!("SKIN-NEST.{tag}"),
        &format!("{nest_differ}of{texels}texels(control:{control})"),
        "0,control>0[design-D§3.1-P0]",
        nest_differ == 0 && texels > 0 && control > 0,
    );
    // SKIN-GRID and SKIN-QUANT on every chunk whose window lies inside the pyramid.
    for rung in rungs {
        let windows = windows_inside(pyr, base, rung);
        if windows.is_empty() {
            c.say(format!(
                "   SKIN-GRID.{tag}.r{rung:02}: no chunk's window lies inside the pyramid"
            ));
            continue;
        }
        let (mut grid_differ, mut grid_texels, mut shifted_differ) = (0u64, 0u64, 0u64);
        let (mut quant_differ, mut quant_texels) = (0u64, 0u64);
        for win in &windows {
            for lv in &win.levels {
                for b in 0..lv.n {
                    for a in 0..lv.n {
                        let (i, j) = (lv.i0 + a, lv.j0 + b);
                        let got = lv.texels[(b * lv.n + a) as usize];
                        let want = pyr.count(lv.level, i, j).map(|(w, _, n)| r8(w, n));
                        grid_texels += 1;
                        grid_differ += u64::from(want != Some(got));
                    }
                }
            }
            // The control: the same chunk's window cut one texel off its alignment, compared at
            // the places the aligned window names.
            let shifted = cut_shifted(base, win.key);
            for (lv, sv) in win.levels.iter().zip(&shifted.levels) {
                for b in 0..lv.n.min(sv.n) {
                    for a in 0..lv.n.min(sv.n) {
                        let got = sv.texels[(b * sv.n + a) as usize];
                        let want = lv.texels[(b * lv.n + a) as usize];
                        shifted_differ += u64::from(got != want);
                    }
                }
            }
            // SKIN-QUANT: each coarser level against the mean of its finer level's rounded texels.
            for pair in win.levels.windows(2) {
                let (fine, coarse) = (&pair[0], &pair[1]);
                for b in 0..coarse.n {
                    for a in 0..coarse.n {
                        let (i, j) = (coarse.i0 + a, coarse.j0 + b);
                        let mut sum = 0u32;
                        for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            sum += u32::from(fine.clamped(2 * i + da, 2 * j + db));
                        }
                        let averaged = ((sum + 2) / 4) as u8;
                        quant_texels += 1;
                        quant_differ +=
                            u64::from(averaged != coarse.texels[(b * coarse.n + a) as usize]);
                    }
                }
            }
        }
        c.judged(
            &format!("SKIN-GRID.{tag}.r{rung:02}"),
            &format!(
                "{grid_differ}of{grid_texels}texels/{}windows(control-one-texel-off:{shifted_differ})",
                windows.len()
            ),
            "0,control>0[design-D§3.1-P1]",
            grid_differ == 0 && grid_texels > 0 && shifted_differ > 0,
        );
        c.reading(
            &format!("SKIN-QUANT-CONTROL.{tag}.r{rung:02}"),
            &format!("{quant_differ}of{quant_texels}coarser-texels-differ-from-averaged-mips"),
            "the-rule-rounds-each-texel-once-from-its-count;averaged-mips-would-drift[design-D§2.1]",
        );
    }
    // AREA-NEST.skin: every coarser texel's wet samples against its four children's.
    for pair in pyr.levels.windows(2) {
        let (lo, hi) = (&pair[0], &pair[1]);
        let (mut coarse_wet, mut fine_wet) = (0u64, 0u64);
        for b in 0..hi.n {
            for a in 0..hi.n {
                let (i, j) = (hi.i0 + a, hi.j0 + b);
                coarse_wet += u64::from(hi.get(i, j).unwrap_or_default().0);
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    fine_wet += u64::from(lo.get(2 * i + da, 2 * j + db).unwrap_or_default().0);
                }
            }
        }
        let pct = if fine_wet == 0 {
            0.0
        } else {
            100.0 * (coarse_wet as f64 - fine_wet as f64) / fine_wet as f64
        };
        c.judged(
            &format!("AREA-NEST.skin.{tag}.r{:02}-r{:02}", lo.level, hi.level),
            &format!("{pct:.6}%({coarse_wet}vs{fine_wet}samples)"),
            "exactly0[design-D§3.1-P3;the-box-fold:+1.8/-6.9%]",
            coarse_wet == fine_wet,
        );
    }
}

/// SKIN-NEST's control: the count of texels the check reads as wrong after one child's wet count
/// is raised by one.
fn nest_control(pyr: &SkinPyramid) -> u64 {
    let (Some(lo), Some(hi)) = (pyr.levels.first(), pyr.levels.get(1)) else {
        return 0;
    };
    let mut bad = lo.clone();
    if let Some(w) = bad.wet.first_mut() {
        *w += 1;
    }
    let mut differ = 0u64;
    for b in 0..hi.n {
        for a in 0..hi.n {
            let (i, j) = (hi.i0 + a, hi.j0 + b);
            let (pw, _) = hi.get(i, j).unwrap_or_default();
            let mut sw = 0u32;
            for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                sw += bad.get(2 * i + da, 2 * j + db).unwrap_or_default().0;
            }
            differ += u64::from(pw != sw);
        }
    }
    differ
}

/// The windows of every rung-`rung` chunk of the pyramid's face whose window lies inside the
/// pyramid's base, filled from the base counts directly.
fn windows_inside(pyr: &SkinPyramid, base: &Level, rung: u8) -> Vec<Window> {
    let lw = window_base_level(rung, 1);
    if lw < base.level || pyr.level(lw + WINDOW_LEVELS - 1).is_none() {
        return Vec::new();
    }
    let edge = CHUNK_EDGE as i64;
    // The pyramid's area in rung-`rung` columns, along each axis.
    let (lo_i, hi_i) = rung_span(base.i0, base.n, base.level, rung);
    let (lo_j, hi_j) = rung_span(base.j0, base.n, base.level, rung);
    let mut out = Vec::new();
    let count = |l: u8, i: i64, j: i64| direct_count(base, l, i, j);
    for y in lo_j.div_euclid(edge)..=(hi_j - 1).div_euclid(edge) {
        for x in lo_i.div_euclid(edge)..=(hi_i - 1).div_euclid(edge) {
            let key = ChunkKey {
                face: pyr.face,
                rung,
                x: x as i32,
                y: y as i32,
                z: 0,
            };
            let mut missing = 0usize;
            let win = cut_window(key, lw, WINDOW_LEVELS, WINDOW_MARGIN, &count, &mut missing);
            if missing == 0 {
                out.push(win);
            }
        }
    }
    out
}

/// The window of `key` cut one texel off at every level (SKIN-GRID's control): the same cut, with
/// every texel read one texel along `i`.
fn cut_shifted(base: &Level, key: ChunkKey) -> Window {
    let count = |l: u8, i: i64, j: i64| direct_count(base, l, i + 1, j);
    let mut missing = 0usize;
    cut_window(
        key,
        window_base_level(key.rung, 1),
        WINDOW_LEVELS,
        WINDOW_MARGIN,
        &count,
        &mut missing,
    )
}

/// A level's span `[o, o + n)` of level-`level` texels along one axis, in rung-`rung` columns
/// that lie wholly inside it.
fn rung_span(o: i64, n: i64, level: u8, rung: u8) -> (i64, i64) {
    if rung >= level {
        let s = rung - level;
        ((o + (1 << s) - 1) >> s, (o + n) >> s)
    } else {
        let s = level - rung;
        (o << s, (o + n) << s)
    }
}
