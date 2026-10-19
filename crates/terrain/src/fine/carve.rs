//! ★ THE RIVER CARVE AT A FINE LEVEL (the one-volume revision §3.5; rulings B105 Q3, B114) — ONE LAW AT EVERY LEVEL.
//!
//! Each link of the level's own drainage takes a width `w` and a mean depth `d̄` from its discharge by Leopold & Maddock
//! 1953's downstream hydraulic geometry with the body's gravity factor ([`crate::river`]). A channel WIDER THAN THE LEVEL'S
//! CELL is carved into the level's raster across its width in a PARABOLIC section — the mean depth of a parabola is two
//! thirds of its greatest, so the greatest is `1.5·d̄` and the section's area stays the law's `w·d̄` — and its water stands
//! at its BANK, the steady state's own height along the link. A narrower river is no geometry at this level: its share
//! belongs to the water map (slice W2). The bank is the drawn top, so where the carve appears the water's top does not
//! step (§3.5 item 2). No channel-head knob: a small discharge gives a narrow channel, and only the cell's own size
//! decides whether this level shows it as geometry.
//!
//! **Example.** The belt's trunk carries 9 500 m³/s: 391 m wide and 16 m deep at Earth's gravity. On a 128 m raster about
//! three cells across hold its bed, 24 m under the bank at the middle, and its water stands at the bank; on the 16 m
//! raster its banks are drawn twenty-four cells apart, and a side creek of 30 m³/s (21 m wide) is carved there too.

use super::delaunay::{Dir, Graph, NONE};
use super::tree::Tree;
use crate::artifact::log2_class;
use crate::gf::Gf;
use crate::river::{channel_depth_mm, channel_width_mm};
use vd_recipe::bend::DIR_BITS;

/// The parabolic section's greatest depth over its mean depth (a parabola's mean is two thirds of its greatest).
const PARABOLA_PEAK: f64 = 1.5;

/// The level's channels: each point whose link to its receiver is wider than the level's cell, with its half width and
/// greatest depth (metres), each point within [`Channels::ring`] graph steps of a channel point, and that ring.
#[derive(Clone, Debug)]
pub struct Channels {
    pub on: Vec<bool>,
    /// Each point's link width (metres; nought with no receiver), wide or narrow.
    pub width: Vec<Gf>,
    pub half_w: Vec<Gf>,
    pub depth_max: Vec<Gf>,
    pub near: Vec<bool>,
    pub ring: usize,
}

/// ★ THE LEVEL'S CHANNELS on the drainage `tree` with each point's discharge `q_m3_yr` (cubic metres a year of the
/// body's own year, the unit the rows' discharge class is written in), on a body of `gravity_mm_s2`, for a raster of
/// `cell_m` metres; `graph` gives the ring's steps.
#[must_use]
pub fn channels(
    tree: &Tree,
    graph: &Graph,
    q_m3_yr: &[Gf],
    gravity_mm_s2: u32,
    cell_m: Gf,
) -> Channels {
    let n = q_m3_yr.len();
    let (mut on, mut width, mut half_w, mut depth_max) = (
        vec![false; n],
        vec![Gf::ZERO; n],
        vec![Gf::ZERO; n],
        vec![Gf::ZERO; n],
    );
    let thousand = Gf::from_i64(1_000);
    let mut widest = Gf::ZERO;
    for p in 0..n {
        if tree.recv[p] == NONE || q_m3_yr[p] <= Gf::ZERO {
            continue;
        }
        // The row's own class rule: mm·m² a year, two fraction bits.
        let mm = (q_m3_yr[p] * thousand).to_i64_floor().max(0) as u64;
        let class = log2_class(mm, 2);
        let w = Gf::from_i64(channel_width_mm(class, gravity_mm_s2)) / thousand;
        width[p] = w;
        if w >= cell_m {
            on[p] = true;
            half_w[p] = w * Gf::HALF;
            depth_max[p] = Gf::from_f64(PARABOLA_PEAK)
                * Gf::from_i64(channel_depth_mm(class, gravity_mm_s2))
                / thousand;
            widest = widest.greater(w);
        }
    }
    // The ring a cell's search walks from its triangle: the widest half width in cells, rounded up, and one more step.
    let ring = (widest * Gf::HALF / cell_m).to_i64_floor().max(0) as usize + 2;
    let mut near = on.clone();
    let mut front: Vec<u32> = (0..n as u32).filter(|&p| on[p as usize]).collect();
    for _ in 0..ring {
        let mut next = Vec::new();
        for &p in &front {
            for &m in graph.neighbours(p as usize) {
                if !near[m as usize] {
                    near[m as usize] = true;
                    next.push(m);
                }
            }
        }
        front = next;
    }
    Channels {
        on,
        width,
        half_w,
        depth_max,
        near,
        ring,
    }
}

/// A direction in metres on a sphere of radius `r_m`.
fn metres(d: &Dir, r_m: Gf) -> [Gf; 3] {
    let s = r_m / Gf::from_f64((1u64 << DIR_BITS) as f64);
    [
        Gf::from_i64(d[0].raw()) * s,
        Gf::from_i64(d[1].raw()) * s,
        Gf::from_i64(d[2].raw()) * s,
    ]
}

/// One raster cell's carve: its height, the bank of the carved channel that holds it (the river's water level), and the
/// SHARE of the cell the narrower rivers through it cover (their width over the cell's, the widest one; a tint, never
/// geometry, §3.5 item 3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Carve {
    pub z: Gf,
    pub bank: Option<Gf>,
    pub share: Gf,
}

/// ★ ONE RASTER CELL'S CARVE: the cell at `cell` (its direction) whose height `z` came from the triangle `tri` on a raster
/// of `cell_m` metres; the channel links within the ring of the triangle's corners are read, and the nearest one (by its
/// distance over its half width) whose section holds the cell lowers it to its bed `bank − depth_max·(1 − (s/half_w)²)`
/// (never raising it) and gives its water the bank's height; the triangle's own corners' narrower links that pass within
/// half a cell of its centre give the cell their share.
#[allow(
    clippy::too_many_arguments,
    reason = "a cell's carve reads the level's channels, its points, graph, tree and heights, and the cell itself"
)]
#[must_use]
pub fn carve_cell(
    ch: &Channels,
    dirs: &[Dir],
    graph: &Graph,
    tree: &Tree,
    h: &[Gf],
    r_m: Gf,
    tri: [u32; 3],
    cell: &Dir,
    z: Gf,
    cell_m: Gf,
) -> Carve {
    let c = metres(cell, r_m);
    // The narrower rivers' share, from the triangle's own corners.
    let mut share = Gf::ZERO;
    for &v in &tri {
        let p = v as usize;
        if ch.on[p] || ch.width[p] <= Gf::ZERO {
            continue;
        }
        let r = tree.recv[p] as usize;
        let (s, _) = to_link(&c, &metres(&dirs[p], r_m), &metres(&dirs[r], r_m));
        if s <= cell_m * Gf::HALF {
            share = share.greater(ch.width[p] / cell_m);
        }
    }
    if !tri.iter().any(|&v| ch.near[v as usize]) {
        return Carve {
            z,
            bank: None,
            share,
        };
    }
    // The channel points within the ring of the triangle's corners.
    let mut seen: Vec<u32> = tri.to_vec();
    let mut front: Vec<u32> = tri.to_vec();
    for _ in 0..ch.ring {
        let mut next = Vec::new();
        for &p in &front {
            for &m in graph.neighbours(p as usize) {
                if ch.near[m as usize] && !seen.contains(&m) {
                    seen.push(m);
                    next.push(m);
                }
            }
        }
        front = next;
    }
    let mut best: Option<(Gf, Gf, Gf)> = None;
    for &p in &seen {
        let p = p as usize;
        if !ch.on[p] {
            continue;
        }
        let r = tree.recv[p] as usize;
        let (s, t) = to_link(&c, &metres(&dirs[p], r_m), &metres(&dirs[r], r_m));
        if s > ch.half_w[p] {
            continue;
        }
        let ratio = s / ch.half_w[p];
        let bank = h[p] + (h[r] - h[p]) * t;
        let bed = bank - ch.depth_max[p] * (Gf::ONE - ratio * ratio);
        if best.is_none_or(|(k, _, _)| ratio < k) {
            best = Some((ratio, bank, bed));
        }
    }
    match best {
        Some((_, bank, bed)) => Carve {
            z: z.lesser(bed),
            bank: Some(bank),
            share,
        },
        None => Carve {
            z,
            bank: None,
            share,
        },
    }
}

/// A point's distance (metres) to the segment `a → b`, and where along it the nearest place stands (0 at `a`, 1 at `b`).
fn to_link(c: &[Gf; 3], a: &[Gf; 3], b: &[Gf; 3]) -> (Gf, Gf) {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if len2 > Gf::ZERO {
        ((ac[0] * ab[0] + ac[1] * ab[1] + ac[2] * ab[2]) / len2).clamp(Gf::ZERO, Gf::ONE)
    } else {
        Gf::ZERO
    };
    let off = [ac[0] - ab[0] * t, ac[1] - ab[1] * t, ac[2] - ab[2] * t];
    (
        (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt(),
        t,
    )
}
