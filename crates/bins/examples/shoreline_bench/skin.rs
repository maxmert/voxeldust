//! ★ THE SKIN (the far-water design D, ruling B16; `docs/investigation/2026-09-26/
//! far_water_seamless_design.md` §2.1–§2.3, slice K0): the share of N2's rung-F point samples
//! under every texel of ONE global per-face pyramid, the window a client cuts from it for one
//! chunk, and the 8-bit value of a texel, rounded once from its own exact count.
//!
//! A level-`l` texel is one rung-`l` cell: 128 m at level 7 (one sample), 8 192 m at level 13 (one
//! node square). Its words are the wet samples under it and the lake samples among them. A texel
//! of level `l + 1` holds exactly four of level `l`, so its words are their sums: the skin NESTS.
//! No count is compared with a threshold anywhere on the skin's path, so no piece of water can
//! show, hide and show again from one rung to the next.
//!
//! **Example.** A rung-16 texel over the gentle coast is 65 km a side and holds 2²⁰ samples:
//! 530 000 wet. Its value is round(255 · 530 000 / 1 048 576) = 129, and its four rung-15 children
//! hold wet counts whose sum is the same 530 000.

use vd_seed::bend::Face;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};

/// The rung whose point samples the skin counts (F on the home planet, ruling B15 Q6).
pub const SKIN_BASE: u8 = 7;
/// The levels a window holds in the design: its finest texel and four coarser (§2.2; the named
/// choice `skin_levels` reads others).
pub const WINDOW_LEVELS: u8 = 5;

/// ★ ONE LEVEL of a count pyramid: texels `[i0, i0 + n) × [j0, j0 + n)` of level `level`, each with
/// its wet and lake sample counts.
#[derive(Clone, Debug)]
pub struct Level {
    pub level: u8,
    pub i0: i64,
    pub j0: i64,
    pub n: i64,
    pub wet: Vec<u32>,
    pub lake: Vec<u32>,
}

impl Level {
    /// The words of texel `(i, j)`, `None` outside the level.
    #[must_use]
    pub fn get(&self, i: i64, j: i64) -> Option<(u32, u32)> {
        let (a, b) = (i - self.i0, j - self.j0);
        ((0..self.n).contains(&a) && (0..self.n).contains(&b)).then(|| {
            let k = (b * self.n + a) as usize;
            (self.wet[k], self.lake[k])
        })
    }

    /// The samples one texel of this level holds.
    #[must_use]
    pub fn per_texel(&self) -> u64 {
        samples_per_texel(self.level)
    }

    /// The next level up: each texel the sum of its four children. The origin and the side must
    /// be even.
    #[must_use]
    pub fn up(&self) -> Level {
        let n = self.n / 2;
        let mut wet = vec![0u32; (n * n) as usize];
        let mut lake = vec![0u32; (n * n) as usize];
        for b in 0..n {
            for a in 0..n {
                let k = (b * n + a) as usize;
                for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = ((2 * b + db) * self.n + 2 * a + da) as usize;
                    wet[k] += self.wet[c];
                    lake[k] += self.lake[c];
                }
            }
        }
        Level {
            level: self.level + 1,
            i0: self.i0 / 2,
            j0: self.j0 / 2,
            n,
            wet,
            lake,
        }
    }
}

/// The samples a texel of `level` holds.
#[must_use]
pub fn samples_per_texel(level: u8) -> u64 {
    1u64 << (2 * u32::from(level - SKIN_BASE))
}

/// ★ THE ONE PYRAMID of a region of one face: levels `base ..= top`, each over the same area.
pub struct SkinPyramid {
    pub face: Face,
    pub levels: Vec<Level>,
}

impl SkinPyramid {
    /// The pyramid over `base` up to `top`. `base`'s origin and side must be multiples of
    /// `2^(top − base.level)`, so every level covers the same area.
    #[must_use]
    pub fn from_base(face: Face, base: Level, top: u8) -> SkinPyramid {
        let mut levels = vec![base];
        while levels.last().is_some_and(|l| l.level < top && l.n >= 2) {
            let next = levels.last().map(Level::up);
            levels.extend(next);
        }
        SkinPyramid { face, levels }
    }

    /// Level `l`, if the pyramid holds it.
    #[must_use]
    pub fn level(&self, l: u8) -> Option<&Level> {
        self.levels.iter().find(|x| x.level == l)
    }

    /// The words of texel `(i, j)` of level `l`: `(wet, lake, samples)`.
    #[must_use]
    pub fn count(&self, l: u8, i: i64, j: i64) -> Option<(u64, u64, u64)> {
        let lv = self.level(l)?;
        let (w, k) = lv.get(i, j)?;
        Some((u64::from(w), u64::from(k), lv.per_texel()))
    }
}

/// ★ THE 8-BIT VALUE of a count: round(255 · wet / n), in integers, half up — rounded ONCE from the
/// texel's own exact count, never averaged from rounded children (SKIN-QUANT).
#[must_use]
pub fn r8(wet: u64, n: u64) -> u8 {
    ((510 * wet + n) / (2 * n)) as u8
}

/// One level of a window: 8-bit texels `[i0, i0 + n) × [j0, j0 + n)` of level `level`.
#[derive(Clone, Debug)]
pub struct R8Level {
    pub level: u8,
    pub i0: i64,
    pub j0: i64,
    pub n: i64,
    pub texels: Vec<u8>,
}

impl R8Level {
    /// Texel `(i, j)`, clamped to the level's edge (the sampler's clamp-to-edge).
    #[must_use]
    pub fn clamped(&self, i: i64, j: i64) -> u8 {
        let a = (i - self.i0).clamp(0, self.n - 1);
        let b = (j - self.j0).clamp(0, self.n - 1);
        self.texels[(b * self.n + a) as usize]
    }
}

/// ★ A WINDOW: what a client cuts from the skin for one chunk.
#[derive(Clone, Debug)]
pub struct Window {
    pub key: ChunkKey,
    pub levels: Vec<R8Level>,
}

/// The finest level of rung `rung`'s window: the cell over `2^sub` (the design: `sub = 1`, half a
/// cell), never finer than one sample (`T_L = max(T₀, cell_L / 2^sub)`).
#[must_use]
pub fn window_base_level(rung: u8, sub: u8) -> u8 {
    rung.saturating_sub(sub).max(SKIN_BASE)
}

/// ★ THE WINDOW OF CHUNK `key`, as a client cuts it (the design §2.3 item 7): the chunk's footprint
/// in its finest texels, `margin` texels more on each side, aligned outward to whole texels of its
/// coarsest level; its five levels, each texel's value rounded once from `count(level, i, j)` =
/// `(wet, samples)`. A texel `count` does not hold is written 0 and reported in `missing`.
pub fn cut_window(
    key: ChunkKey,
    lw: u8,
    n_levels: u8,
    margin: i64,
    count: &dyn Fn(u8, i64, i64) -> Option<(u64, u64)>,
    missing: &mut usize,
) -> Window {
    let (i0, j0, n0) = window_area(key, lw, n_levels, margin);
    let mut levels = Vec::with_capacity(usize::from(n_levels));
    for k in 0..n_levels {
        let (li0, lj0, ln) = (i0 >> k, j0 >> k, n0 >> k);
        let l = lw + k;
        let mut texels = vec![0u8; (ln * ln) as usize];
        for b in 0..ln {
            for a in 0..ln {
                texels[(b * ln + a) as usize] = match count(l, li0 + a, lj0 + b) {
                    Some((w, n)) => r8(w, n),
                    None => {
                        *missing += 1;
                        0
                    }
                };
            }
        }
        levels.push(R8Level {
            level: l,
            i0: li0,
            j0: lj0,
            n: ln,
            texels,
        });
    }
    Window { key, levels }
}

/// ★ THE AREA chunk `key`'s window covers, `(i0, j0, n0)` in level-`lw` texels: the chunk's
/// footprint, `margin` texels more on each side, aligned outward to whole texels of the window's
/// coarsest level (level `k` of the window holds `(i0 >> k, j0 >> k, n0 >> k)`).
#[must_use]
pub fn window_area(key: ChunkKey, lw: u8, n_levels: u8, margin: i64) -> (i64, i64, i64) {
    let edge = CHUNK_EDGE as i64;
    let (lo_i, hi_i) = footprint(key.x, key.rung, lw, edge);
    let (lo_j, hi_j) = footprint(key.y, key.rung, lw, edge);
    let align = 1i64 << (n_levels - 1);
    let down = |v: i64| v.div_euclid(align) * align;
    let up = |v: i64| (v + align - 1).div_euclid(align) * align;
    let (i0, i1) = (down(lo_i - margin), up(hi_i + margin));
    let (j0, j1) = (down(lo_j - margin), up(hi_j + margin));
    (i0, j0, (i1 - i0).max(j1 - j0))
}

/// A chunk's footprint along one axis, `[lo, hi)`, in level-`lw` texels: chunk index `x` of rung
/// `rung` holds columns `[62x, 62x + 62)`.
fn footprint(x: i32, rung: u8, lw: u8, edge: i64) -> (i64, i64) {
    let (lo, hi) = (i64::from(x) * edge, i64::from(x) * edge + edge);
    if rung >= lw {
        let s = rung - lw;
        (lo << s, hi << s)
    } else {
        let s = lw - rung;
        (lo >> s, (hi + (1 << s) - 1) >> s)
    }
}

/// ★ THE DIRECT COUNT of texel `(i, j)` of level `l` from the level-`base` counts under it — the
/// second path a window's values come from (the client builds a window from its records, never
/// from the pyramid's sums), so SKIN-GRID holds two computations equal.
#[must_use]
pub fn direct_count(base: &Level, l: u8, i: i64, j: i64) -> Option<(u64, u64)> {
    if l < base.level {
        return None;
    }
    let s = l - base.level;
    let side = 1i64 << s;
    let (ci, cj) = (i << s, j << s);
    let mut wet = 0u64;
    for b in 0..side {
        for a in 0..side {
            let (w, _) = base.get(ci + a, cj + b)?;
            wet += u64::from(w);
        }
    }
    Some((wet, samples_per_texel(l)))
}
