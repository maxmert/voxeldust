//! ★ THE 4 m LEVEL ON THE BENCH (build step B5; `docs/investigation/2026-10-14/ground_2to4m_design.md` revision 2) — no
//! game path reads it: the shard, the gateway and the client know two levels until the owner's look and word.
//!
//! The level is the 16 m level one rung band down (ruling B97's law, B96's form), through the same solve
//! ([`super::refine`]):
//!
//! 1. **The 16 m flow record** ([`flow_record`]), one per 16 m tile: the 16 m domain of the node (the node and its 2 km
//!    buffer, set up under the 128 m flow exactly as the 16 m tile is — so the water at its rim is the 128 m flow's, the
//!    8 km rivers' included), its tree and water on the DRAWN 16 m ground (the finished 16 m tiles), and each 128 m cell's
//!    rate read back by the joint law over its 16 m links under the threshold (the median a cell), read to each 16 m point
//!    by the Gaussian of the logarithms (σ half a 128 m spacing — the 16 m level's own read, one level down).
//! 2. **The 4 m tile** ([`solve_l3`]): 512 × 512 cells of the rung-2 cell (2 048 m on the home planet, a quarter node),
//!    its domain 128 cells each side, set up under the record; the steady form under tan 35° with the half step; the
//!    passes stop when no 16 m cell's mean moves more than [`STEADY_SETTLE_M`] (one stored quarter metre), within the
//!    bound; the raster (the tile and 64 cells each side) is today's drawn 16 m ground plus the law's detail finer than
//!    16 m, in metres — the bench chooses the stored unit.
//!
//! **Example.** The 4 m tiles in the middle of the node beside hull 19: four tiles of 2 048 m read one 16 m flow record of
//! the node (about 560 000 points), each tile about 560 000 points of its own; their rasters show the small chutes and
//! the sharper crests the 16 m curve smooths away, on the same valleys.

use super::l1::{TileKey, cells_per_node_shift};
use super::l2::{L2_RUNG, l2_domain};
use super::points::CellId;
use super::read::FineBook;
use super::refine::{Coarse, drawn, median, raster, setup, steady};
use super::tree::{accumulate, tree};
use crate::artifact::Artifact;
use crate::body::{BodyDefinition, TAN_REPOSE};
use crate::gf::Gf;
use crate::macro_lattice::MacroLattice;
use crate::solve::{JOINT_K_LP, JOINT_N};
use vd_seed::bend::Face;

/// The level's rung (the rung-2 cell: 4 m on the home planet).
pub const L3_RUNG: u8 = 2;
/// A tile's side in cells, as a shift: 512 cells, as every level's tile.
pub const TILE_CELLS_SHIFT: u32 = 9;
/// The buffer each side in cells (the same cell count as the 16 m level's: 128).
pub const BUFFER_CELLS: i32 = 128;
/// ★ THE STEADY FORM'S END: no 16 m cell's mean moves more than one stored quarter metre (1 m would be about the whole 4 m
/// signal: Earth's 2–16 m detail is 0.7–1.2 m RMS), within the 128 m level's own bound.
pub const STEADY_SETTLE_M: f64 = 0.25;
pub const STEADY_BOUND: usize = super::l1::FINE_ROUND_BOUND;

/// ★ THE 16 m FLOW RECORD of one node: each 16 m point's cell, outlet word, drawn 16 m height, receiver, water, rate,
/// rain and place; the chart rectangle of the node's 16 m domain; the face's 16 m cells an edge; the 16 m spacing.
pub struct FlowRecord {
    pub face: Face,
    pub node: (i32, i32),
    cells: Vec<CellId>,
    outlet: Vec<bool>,
    h: Vec<Gf>,
    recv: Vec<u32>,
    q: Vec<Gf>,
    rate: Vec<Gf>,
    rain: Vec<Gf>,
    pos: Vec<[Gf; 3]>,
    rect: (i32, i32, i32),
    n: i32,
    spacing: Gf,
}

impl FlowRecord {
    /// The 16 m points the record holds.
    #[must_use]
    pub fn points(&self) -> usize {
        self.h.len()
    }
}

/// ★ THE 16 m FLOW RECORD of the node `key` (a 16 m tile): its domain set up under the 128 m flow of `book1` (the
/// finished 128 m tiles), its tree and water on the drawn 16 m ground of `book2` (the finished 16 m tiles), each 128 m
/// cell's rate read back by the joint law. `None` where a tile it reads is not held or a domain cannot be built.
#[must_use]
pub fn flow_record(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    book1: &FineBook,
    book2: &FineBook,
    key: TileKey,
) -> Option<FlowRecord> {
    let d = l2_domain(body, lattice, artifact, book1, key)?;
    let s = &d.s;
    // The drawn 16 m ground at each 16 m point (an outlet the lower of it and its exit's height, as the 128 m flow reads).
    let mut h = Vec::with_capacity(s.cells.len());
    for (p, c) in s.cells.iter().enumerate() {
        let z = drawn(book2, c, L2_RUNG, d.sea)?;
        h.push(if s.outlet[p] { z.lesser(s.base[p]) } else { z });
    }
    let t = tree(&h, &s.outlet, &s.graph, &s.geom.len);
    let q = accumulate(&t, &s.water);
    // THE RATE: the joint law read backwards over each 128 m cell's 16 m links under the threshold, the median a cell,
    // then each 16 m point's by the read's own weights (the Gaussian of the logarithms over the 128 m points).
    let (k, n, tan) = (
        Gf::from_f64(JOINT_K_LP),
        Gf::from_f64(JOINT_N),
        Gf::from_f64(TAN_REPOSE),
    );
    let ncell = d.f1.h.len();
    let mut samples: Vec<Vec<Gf>> = vec![Vec::new(); ncell];
    for c in 0..h.len() {
        let r = t.recv[c];
        let cell = s.cell_of[c];
        if s.outlet[c] || r == super::delaunay::NONE || cell == u32::MAX {
            continue;
        }
        let sl = (h[c] - h[r as usize]) / t.link[c];
        if sl > Gf::ZERO && sl < tan && q[c] > Gf::ZERO {
            samples[cell as usize].push(k * q[c] * sl.powf_fine(n));
        }
    }
    let overall = median(samples.iter().flatten().copied().collect())
        .unwrap_or(Gf::from_f64(super::l1::U0_LAND));
    let ln_cell: Vec<Gf> = samples
        .into_iter()
        .map(|v| median(v).unwrap_or(overall).ln_fine())
        .collect();
    let mut rate = Vec::with_capacity(h.len());
    for p in 0..h.len() {
        let mut l = Gf::ZERO;
        for w in s.wstart[p] as usize..s.wstart[p + 1] as usize {
            l += s.wval[w] * ln_cell[s.wcell[w] as usize];
        }
        rate.push(if s.wstart[p + 1] > s.wstart[p] {
            l.exp_fine()
        } else {
            overall
        });
    }
    let node4 = 1i32 << d.shift4;
    Some(FlowRecord {
        face: key.face,
        node: (key.tx, key.ty),
        cells: s.cells.clone(),
        outlet: s.outlet.clone(),
        h,
        recv: t.recv,
        q,
        rate,
        rain: s.rain.clone(),
        pos: s.pos.clone(),
        rect: (
            (key.tx << d.shift4) - d.buf,
            (key.ty << d.shift4) - d.buf,
            node4 + 2 * d.buf,
        ),
        n: d.n4,
        spacing: Gf::from_f64(lattice.node_m() / f64::from(1u32 << d.shift4)),
    })
}

/// A solved 4 m tile: the tile and 64 cells each side as `side × side` heights in metres over the sea (rows of the face's
/// chart from its north-west corner; `None` off the faces), and the readings.
#[derive(Clone, Debug)]
pub struct L3Tile {
    pub side: usize,
    pub metres: Vec<Option<f64>>,
    pub points: usize,
    pub passes: usize,
    /// The points whose slope the soil law set in the last pass (nought with no soil law).
    pub on_soil: usize,
    /// The law's detail added to today's 16 m ground: the median and the largest size, in centimetres.
    pub detail_cm: [i64; 2],
}

/// ★ THE 4 m TILE `key` (tiles of [`TILE_CELLS_SHIFT`] cells at [`L3_RUNG`]) under the 16 m flow `record` and the finished
/// 16 m tiles of `book2`. `None` where the rung's cells do not nest, the tile does not lie in the record's node, a tile it
/// reads is not held, or a domain cannot be built.
#[must_use]
pub fn solve_l3(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    sea_m: i16,
    record: &FlowRecord,
    book2: &FineBook,
    key: TileKey,
    soil: Option<&crate::fine::refine::SoilLaw>,
) -> Option<L3Tile> {
    let shift2 = cells_per_node_shift(body, lattice, L3_RUNG)?;
    let shift4 = cells_per_node_shift(body, lattice, L2_RUNG)?;
    let down = shift2.checked_sub(shift4)?;
    let tiles_shift = shift2.checked_sub(TILE_CELLS_SHIFT)?;
    if key.face != record.face || (key.tx >> tiles_shift, key.ty >> tiles_shift) != record.node {
        return None;
    }
    let n2 = body.ladder().cells_per_edge(L3_RUNG) as i32;
    let sea = Gf::from_i32(i32::from(sea_m));
    let r_m = Gf::from_f64(body.radius_m());
    let ts = 1i32 << TILE_CELLS_SHIFT;
    let (ti0, tj0) = (key.tx << TILE_CELLS_SHIFT, key.ty << TILE_CELLS_SHIFT);
    let coarse = Coarse {
        cells: &record.cells,
        outlet: &record.outlet,
        h: &record.h,
        recv: &record.recv,
        q: &record.q,
        rate: &record.rate,
        rain: &record.rain,
        pos: &record.pos,
        rect: record.rect,
        n: record.n,
        spacing: record.spacing,
    };
    let drawn_at = |c: &CellId| drawn(book2, c, L3_RUNG, sea);
    let s = setup(
        body.seed(),
        r_m,
        key.face,
        L3_RUNG,
        n2,
        down,
        (
            ti0 - BUFFER_CELLS,
            tj0 - BUFFER_CELLS,
            ts + 2 * BUFFER_CELLS,
        ),
        &coarse,
        &drawn_at,
    )?;
    let (h, passes, on_soil) = steady(
        &s,
        record.h.len(),
        Gf::from_f64(TAN_REPOSE),
        Gf::from_f64(STEADY_SETTLE_M),
        STEADY_BOUND,
        soil,
    );
    let half = BUFFER_CELLS >> 1;
    let rside = ts + 2 * half;
    let (out, detail) = raster(
        &s,
        &h,
        key.face,
        n2,
        down,
        (ti0 - half, tj0 - half, rside),
        &drawn_at,
    )?;
    let cm = |g: Gf| (g * Gf::from_i64(100)).floor().to_i64_floor();
    Some(L3Tile {
        side: rside as usize,
        metres: out.iter().map(|z| z.map(Gf::to_f64)).collect(),
        points: s.dirs.len(),
        passes,
        on_soil,
        detail_cm: [
            detail.get(detail.len() >> 1).map_or(0, |&g| cm(g)),
            detail.last().map_or(0, |&g| cm(g)),
        ],
    })
}
