//! ★ THE FINE READ (B1 (g); the way-in design rev 4 §2 and §4.5; rulings B83 item 2 and B85): a column's height from the
//! finished 128 m tiles a host holds — ONE read for the shard and every client (SL10), integers only.
//!
//! **What a host holds.** A finished tile (B85): its own square after its cuts, `2^ts` cells of rung 7 a side (512 on the
//! home planet), and its mean levels — level `k` the box means of `2^k × 2^k` cells, which rung `7 + k` reads (B83
//! item 2: rungs 7–12). Every height is in quarter metres over the body's sea, rows from the tile's north-west corner.
//!
//! **The read.** The surface a chunk draws at `draw_rung` reads level `k = max(draw_rung − 7, 0)`. A level's samples are
//! the centres of its cells, as the macro nodes' are (`artifact::node_and_fraction`): in half rung-0 cells the centre
//! of cell `i` at rung `c` is `i·2^(c+1) + 2^c`, a level sample's `X·2^(8+k) + 2^(7+k)`, so the sample before a point and
//! the fraction past it come from one shift and one mask. The sixteen samples around it go through the macro field's own
//! Catmull-Rom kernel (`vd_recipe::macro_field::catmull_rom_16`) and the sea is added back. A column at rung `7 + k`
//! stands on its own sample (fraction zero, the kernel's weights 0, 1, 0, 0), so it reads the mean exactly; a column at
//! rungs 0–6 and a float direction's read (the eye, the morph) read the curve between them. A sample off the face is the
//! partner face's cell by the chart's one crossing rule (`points::chart_cell`); a cube corner's diagonal region reads
//! the face's nearest cell, as the macro read does (`artifact::node_at`).
//!
//! **The three answers.** [`FineZ::Held`] — the height. [`FineZ::Missing`] — the level serves the column but a tile of
//! its stencil is not held: the chunk is not built and the coarser rung stands (F9, B81 Q5). [`FineZ::Outside`] — the
//! level does not serve the column (a rung over 12, or a field with no book): today's read stands.
//!
//! **Example.** A hull flies 2 km over the home planet's belt. Its rung-9 chunks read level 2 of the belt's finished
//! tiles (512 m means) at their columns' own samples; the rung-3 chunks under its landing gear read the 128 m raster's
//! curve at their 8 m cells; both stand on the same finished ground the planet's shard drew.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::l1::TileKey;
use super::level::{FineLevel, LevelGeom};
use super::points::chart_cell;
use crate::artifact::{DRY_M, Side, ZField};
use crate::body::BodyDefinition;
use crate::chunk::CHUNK_EDGE;
use crate::macro_lattice::MacroLattice;
use crate::units::{LENGTH_BITS, STEPS_PER_M};
use vd_recipe::Gi;
use vd_recipe::macro_field::catmull_rom_16;
use vd_recipe::noise::NOISE_BITS;
use vd_seed::bend::Face;

/// The coarsest rung L1 serves: rung 7's raster and its five mean levels (B83 item 2); rung 13 reads the node's row.
pub const L1_TOP_RUNG: u8 = FineLevel::L1.top_rung();
/// The mean levels a finished 128 m tile carries over its raster.
pub const MEAN_LEVELS: u32 = FineLevel::L1.mean_levels();
/// Gap steps a quarter metre: the raster's unit in the height's.
const QUARTER_STEPS: i64 = STEPS_PER_M >> 2;

/// What the fine read answers for one column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FineZ {
    /// The fine ground's height, in the height's unit (gap steps at [`LENGTH_BITS`]) over the body's radius, as the
    /// macro field's `Z`.
    Held(Gi),
    /// The level serves the column, but this tile of its stencil is not held: the chunk waits.
    Missing(TileKey),
    /// The level does not serve the column: today's read stands.
    Outside,
}

/// ★ THE BOOK of the finished tiles of ONE level a host holds, and the one read over them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FineBook {
    /// The body's sea, in the height's unit: the rasters stand over it.
    sea: Gi,
    /// The level's numbers on the body (its rung, a tile's side, the cells along a face's edge).
    geom: LevelGeom,
    /// Each held tile's levels, shared by pointer: a book handed on (a copy-on-write cache, a field a builder holds)
    /// never copies a tile's heights.
    tiles: BTreeMap<TileKey, Arc<HeldLevels>>,
}

/// ★ A HELD TILE'S LEVELS (ruling B86 item 3, far copies first): from its finest held level `lo` to the coarsest, the
/// level-`k` heights at `levels[k − lo]` — a far tile holds only its coarse copies, and a finer level, when the eye
/// comes near, joins them below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldLevels {
    pub lo: u32,
    pub levels: Vec<Vec<i32>>,
    /// ★ THE TILE'S WATER (slice W2), at the same levels as `levels`: each cell's water level (quarter metres over the
    /// sea, [`super::stitch::MISSING`] where no water holds it whole) and wet share (0 ..= [`super::tile::WET_FULL`]);
    /// `None` for a tile held without its water (a record of the older form).
    pub water: Option<HeldWater>,
}

/// A held tile's water planes ([`HeldLevels::water`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldWater {
    pub level: Vec<Vec<i32>>,
    pub wet: Vec<Vec<i32>>,
}

/// ★ THE WATER AT A PLACE ([`FineBook::water`]): the level and share of the level's cell that holds the place's centre
/// (the nearest sample — a water's level never blends, the one-volume revision §3.4's column rule), or why there is none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FineWater {
    /// The cell's water level (quarter metres over the sea, [`super::stitch::MISSING`] where no water holds the cell whole)
    /// and its wet share.
    Held { level: i32, wet: i32 },
    /// The level serves the place, but the tile that holds it is not held (or holds no water).
    Missing(TileKey),
    /// The level does not serve the place.
    Outside,
}

/// ★ WHERE THE 128 m LEVEL APPLIES (the way-in design §3.6, a law and never a flag on a body): where the macro solve's
/// water exists — a body whose artifact states a sea. A body with no water (an airless moon) keeps today's ground until
/// its own families exist.
#[must_use]
pub fn applies(sea_m: i16) -> bool {
    sea_m != DRY_M
}

impl FineBook {
    /// An empty book of `level` for `body`, its sea at `sea_m` metres (the artifact's). `None` where the level's cells
    /// do not nest in the macro nodes by a power of two, or a tile is too small to carry its mean levels.
    #[must_use]
    pub fn new(
        level: FineLevel,
        body: &BodyDefinition,
        lattice: &MacroLattice,
        sea_m: i16,
    ) -> Option<FineBook> {
        FineBook::of_geom(LevelGeom::of(level, body, lattice)?, sea_m)
    }

    /// An empty book of `level` from the numbers a host holds without the body: the macro lattice's edge (the artifact's
    /// head), a tile's side in the level's cells as a shift (a tile record states it), and the sea. `None` where a tile
    /// is too small to carry the level's mean levels or smaller than its nodes.
    #[must_use]
    pub fn with_side(level: FineLevel, edge: u32, ts: u32, sea_m: i16) -> Option<FineBook> {
        FineBook::of_geom(LevelGeom::from_side(level, ts, edge)?, sea_m)
    }

    fn of_geom(geom: LevelGeom, sea_m: i16) -> Option<FineBook> {
        (geom.ts >= geom.level.mean_levels()).then(|| FineBook {
            sea: Gi::new(i64::from(sea_m) * STEPS_PER_M) << LENGTH_BITS,
            geom,
            tiles: BTreeMap::new(),
        })
    }

    /// The book's level.
    #[must_use]
    pub fn level(&self) -> FineLevel {
        self.geom.level
    }

    /// A tile's side in the level's cells, as a shift.
    #[must_use]
    pub fn side_shift(&self) -> u32 {
        self.geom.ts
    }

    /// How many tiles are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether no tile is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// The held tiles' keys, in order.
    pub fn keys(&self) -> impl Iterator<Item = TileKey> + '_ {
        self.tiles.keys().copied()
    }

    /// A finished tile's side in the level's cells.
    #[must_use]
    pub fn side(&self) -> usize {
        1usize << self.geom.ts
    }

    /// Holds the finished tile `key`: `levels[0]` its square, `levels[k]` its `2^k` means, each `side >> k` squared.
    /// `false` (and nothing held) where a level has another size or a level is missing.
    pub fn insert(&mut self, key: TileKey, levels: Vec<Vec<i32>>) -> bool {
        if levels.len() != self.geom.level.mean_levels() as usize + 1 {
            return false;
        }
        self.insert_from(key, 0, levels)
    }

    /// ★ HOLDS THE LEVELS `lo ..` OF THE TILE `key` (ruling B86 item 3): the levels from `lo` to the coarsest join what
    /// the book holds of the tile; levels the book already holds stand as they are (they are the same tile's). `false`
    /// (and nothing changed) where a level has another size, the levels stop short of the coarsest the book needs, or a
    /// gap would stand between them and the levels held.
    pub fn insert_from(&mut self, key: TileKey, lo: u32, levels: Vec<Vec<i32>>) -> bool {
        self.insert_planes(key, lo, levels, None)
    }

    /// ★ HOLDS THE LEVELS `lo ..` OF THE TILE `key` WITH THEIR WATER (slice W2): as [`FineBook::insert_from`], the water's
    /// planes joining the heights level by level. `false` (and nothing changed) as there, or where a water plane has
    /// another size than its level.
    pub fn insert_water_from(
        &mut self,
        key: TileKey,
        lo: u32,
        levels: super::tile::WaterLevels,
    ) -> bool {
        let n = levels.heights.len();
        let sized = (levels.level.len() == n)
            && (levels.wet.len() == n)
            && (0..n).all(|k| {
                (levels.level[k].len() == levels.heights[k].len())
                    && (levels.wet[k].len() == levels.heights[k].len())
            });
        sized
            && self.insert_planes(
                key,
                lo,
                levels.heights,
                Some(HeldWater {
                    level: levels.level,
                    wet: levels.wet,
                }),
            )
    }

    /// The one insert under [`FineBook::insert_from`] and [`FineBook::insert_water_from`]: a tile's levels from `lo`, with
    /// their water or without; a merge keeps the water only where both parts carry it.
    fn insert_planes(
        &mut self,
        key: TileKey,
        lo: u32,
        levels: Vec<Vec<i32>>,
        water: Option<HeldWater>,
    ) -> bool {
        let top = self.geom.level.mean_levels() + 1;
        let hi = lo + levels.len() as u32;
        let sized = (lo < top)
            && (hi <= top)
            && levels
                .iter()
                .zip(lo..)
                .all(|(l, k)| l.len() == (self.side() >> k) * (self.side() >> k));
        let held = self.tiles.get(&key);
        // The levels must reach the coarsest, or the finest level the book holds of the tile.
        let joins = held.map_or(hi == top, |h| hi >= h.lo);
        if !sized || !joins {
            return false;
        }
        let (merged, water) = match held {
            Some(h) if h.lo <= lo => return true,
            Some(h) => {
                let cut = (h.lo - lo) as usize;
                let mut all = levels;
                all.truncate(cut);
                all.extend(h.levels.iter().cloned());
                let water = match (water, &h.water) {
                    (Some(mut w), Some(hw)) => {
                        w.level.truncate(cut);
                        w.wet.truncate(cut);
                        w.level.extend(hw.level.iter().cloned());
                        w.wet.extend(hw.wet.iter().cloned());
                        Some(w)
                    }
                    _ => None,
                };
                (all, water)
            }
            None => (levels, water),
        };
        self.tiles.insert(
            key,
            Arc::new(HeldLevels {
                lo,
                levels: merged,
                water,
            }),
        );
        true
    }

    /// ★ THE WATER AT THE SAMPLE NEAREST the centre of the cell `(i, j)` of `cell_rung` on `face`, for a read at
    /// `draw_rung`: that sample's level and share, never a blend — the IDENTITY a finer level's solve takes (the one-volume
    /// revision §3.4 item 5: a 16 m cell takes the 128 m water at its place).
    #[must_use]
    pub fn water(&self, face: Face, draw_rung: u8, cell_rung: u8, i: i32, j: i32) -> FineWater {
        self.water_read(face, draw_rung, cell_rung, (i, j), false)
    }

    /// ★ THE WATER A DRAWN PLACE READS — THE COLUMN RULE (the one-volume revision §3.4): the place takes the level of the
    /// water that holds the level's samples around it whole (the highest of the four samples bracketing its centre), and
    /// its share from the sample nearest it. The drawing then wets the place where its OWN smooth ground stands under that
    /// level, so the shore follows the ground's crossing of the level and never a sample's square. MEASURED before
    /// (2026-10-18, the owner's window at the basin lake: *"The shores squared"*): the nearest sample alone held a level
    /// only where its cell lay wholly under water, so every shore stepped along the cells — 128 m squares from afar, 16 m
    /// ones near. A solve never reads this (it takes [`Self::water`]), so no stored tile changes.
    #[must_use]
    pub fn shore_water(
        &self,
        face: Face,
        draw_rung: u8,
        cell_rung: u8,
        i: i32,
        j: i32,
    ) -> FineWater {
        self.water_read(face, draw_rung, cell_rung, (i, j), true)
    }

    /// [`Self::water`] (`around` false) and [`Self::shore_water`] (`around` true).
    fn water_read(
        &self,
        face: Face,
        draw_rung: u8,
        cell_rung: u8,
        (i, j): (i32, i32),
        around: bool,
    ) -> FineWater {
        let level = self.geom.level;
        if draw_rung > level.top_rung() {
            return FineWater::Outside;
        }
        let k = u32::from(draw_rung.saturating_sub(level.rung()));
        let half = Gi::new(1i64 << (NOISE_BITS - 1));
        let (x, tx) = sample_and_fraction(level.rung(), k, cell_rung, i);
        let (y, ty) = sample_and_fraction(level.rung(), k, cell_rung, j);
        // One sample's water: its level and share, or the key of the tile that holds it when that tile (or its water) is
        // not held.
        let at = |sx: i64, sy: i64| -> Result<(i32, i32), TileKey> {
            let (key, index) = self.locate(face, k, sx, sy);
            self.tiles
                .get(&key)
                .and_then(|h| {
                    let at = k.checked_sub(h.lo)? as usize;
                    let w = h.water.as_ref()?;
                    Some((w.level[at][index], w.wet[at][index]))
                })
                .ok_or(key)
        };
        let nearest = match at(x + i64::from(tx >= half), y + i64::from(ty >= half)) {
            Ok(w) => w,
            Err(key) => return FineWater::Missing(key),
        };
        if !around {
            return FineWater::Held {
                level: nearest.0,
                wet: nearest.1,
            };
        }
        let mut top = super::stitch::MISSING;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            match at(x + dx, y + dy) {
                Ok((l, w)) if w == super::tile::WET_FULL => top = top.max(l),
                Ok(_) => {}
                Err(key) => return FineWater::Missing(key),
            }
        }
        FineWater::Held {
            level: top,
            wet: nearest.1,
        }
    }

    /// The finest level the book holds of the tile `key`, if any.
    #[must_use]
    pub fn held_from(&self, key: TileKey) -> Option<u32> {
        self.tiles.get(&key).map(|h| h.lo)
    }

    /// Lets the tile `key` go (a host bounded in bytes); whether it was held.
    pub fn remove(&mut self, key: TileKey) -> bool {
        self.tiles.remove(&key).is_some()
    }

    /// Whether the tile `key` is held.
    #[must_use]
    pub fn holds(&self, key: TileKey) -> bool {
        self.tiles.contains_key(&key)
    }

    /// ★ THE READ: the fine ground a chunk at `draw_rung` draws, at the centre of the cell `(i, j)` of `cell_rung` on
    /// `face` (a column passes its own rung twice; a float direction passes rung 0's cell).
    #[must_use]
    pub fn z(&self, face: Face, draw_rung: u8, cell_rung: u8, i: i32, j: i32) -> FineZ {
        let level = self.geom.level;
        if draw_rung > level.top_rung() {
            return FineZ::Outside;
        }
        let k = u32::from(draw_rung.saturating_sub(level.rung()));
        let (x, ta) = sample_and_fraction(level.rung(), k, cell_rung, i);
        let (y, tb) = sample_and_fraction(level.rung(), k, cell_rung, j);
        let mut words = [Gi::ZERO; 16];
        for row in 0..4i64 {
            for col in 0..4i64 {
                match self.sample(face, k, x + col - 1, y + row - 1) {
                    Ok(q) => {
                        words[(row * 4 + col) as usize] =
                            Gi::new(i64::from(q) * QUARTER_STEPS) << LENGTH_BITS
                    }
                    Err(key) => return FineZ::Missing(key),
                }
            }
        }
        FineZ::Held(catmull_rom_16(&words, ta, tb) + self.sea)
    }

    /// The level-`k` sample `(x, y)` of `face`'s chart in quarter metres over the sea, or the key of the tile that holds it
    /// when that tile is not held.
    fn sample(&self, face: Face, k: u32, x: i64, y: i64) -> Result<i32, TileKey> {
        let (key, index) = self.locate(face, k, x, y);
        self.tiles
            .get(&key)
            .and_then(|h| k.checked_sub(h.lo).map(|at| h.levels[at as usize][index]))
            .ok_or(key)
    }

    /// The tile that holds the level-`k` sample `(x, y)` of `face`'s chart, and the sample's index in that level: a
    /// sample off the face by the chart's one crossing rule, a cube corner's diagonal region the face's nearest cell.
    fn locate(&self, face: Face, k: u32, x: i64, y: i64) -> (TileKey, usize) {
        let n = self.geom.n >> k;
        let (x, y) = (
            x.clamp(-i64::from(n), 2 * i64::from(n)) as i32,
            y.clamp(-i64::from(n), 2 * i64::from(n)) as i32,
        );
        let (f, ci, cj) = match chart_cell(face, x, y, n) {
            Some(c) => (Face::from_index(c.face).unwrap_or(face), c.i, c.j),
            None => (face, x.clamp(0, n - 1), y.clamp(0, n - 1)),
        };
        let s = self.geom.ts - k;
        let mask = (1i32 << s) - 1;
        (
            TileKey {
                face: f,
                tx: ci >> s,
                ty: cj >> s,
            },
            (((cj & mask) << s) | (ci & mask)) as usize,
        )
    }

    /// ★ THE TILES A CHUNK READS: every finished tile the stencils of the chunk `(face, rung, x, y)`'s columns can read —
    /// its cells and its halo, the stencil's one sample before and two after, and one more each side (a halo column across
    /// a face's edge reads in its own face's chart). What a waiting chunk is named with; empty at a rung no fine level
    /// serves.
    #[must_use]
    pub fn tiles_of_chunk(&self, face: Face, rung: u8, x: i32, y: i32) -> Vec<TileKey> {
        let level = self.geom.level;
        if rung > level.top_rung() {
            return Vec::new();
        }
        let k = u32::from(rung.saturating_sub(level.rung()));
        let e = CHUNK_EDGE as i32;
        let lo = |c: i32| sample_and_fraction(level.rung(), k, rung, c * e - 1).0 - 2;
        let hi = |c: i32| sample_and_fraction(level.rung(), k, rung, c * e + e).0 + 3;
        let (x0, x1, y0, y1) = (lo(x), hi(x), lo(y), hi(y));
        let n = i64::from(self.geom.n >> k);
        let s = self.geom.ts - k;
        let mut keys = BTreeSet::new();
        // On the face: the tiles over the rectangle clipped to it.
        for ty in (y0.max(0) >> s)..=(y1.min(n - 1) >> s) {
            for tx in (x0.max(0) >> s)..=(x1.min(n - 1) >> s) {
                keys.insert(TileKey {
                    face,
                    tx: tx as i32,
                    ty: ty as i32,
                });
            }
        }
        // Past the face's edges: each sample by the chart's crossing rule — walked only for a chunk whose samples reach
        // past an edge (the lane asks again every frame while a tile is missing, for the chunk and each parent).
        if (x0 < 0) | (x1 >= n) | (y0 < 0) | (y1 >= n) {
            for sy in y0..=y1 {
                for sx in x0..=x1 {
                    if (sx < 0) | (sx >= n) | (sy < 0) | (sy >= n) {
                        keys.insert(self.locate(face, k, sx, sy).0);
                    }
                }
            }
        }
        keys.into_iter().collect()
    }

    /// Whether every tile of `keys` is held at the level a chunk at `rung` reads (its level or a finer one).
    #[must_use]
    pub fn holds_all(&self, keys: &[TileKey], rung: u8) -> bool {
        let k = u32::from(rung.saturating_sub(self.geom.level.rung()));
        keys.iter()
            .all(|key| self.tiles.get(key).is_some_and(|h| h.lo <= k))
    }
}

/// ★ A MACRO FIELD WITH ITS FINE GROUND (B2; ruling B95): the one field a chunk and the shard's ground read on a body
/// where the fine levels apply — every macro word from `inner` (the whole artifact, its tiles, or a pyramid level), the
/// fine ground from the 128 m book, and under the 16 m level's top rung from the 16 m book when the field carries one.
///
/// **Example.** The client's chunk at rung 9 over the belt reads the artifact's tiles for the rock province and the 128 m
/// book's 512 m means for its height; the chunk at rung 2 under a walker's feet reads the 16 m book's raster by the curve.
#[derive(Clone)]
pub struct FineField {
    pub inner: Arc<dyn ZField + Send + Sync>,
    pub book: Arc<FineBook>,
    /// The finer level's book (the 16 m level), read for the rungs it serves; `None` reads the 128 m book alone.
    pub finer: Option<Arc<FineBook>>,
}

impl ZField for FineField {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.inner.z_m(node)
    }
    fn level(&self) -> u32 {
        self.inner.level()
    }
    fn water_facies(&self, node: u32) -> Option<(i16, u8)> {
        self.inner.water_facies(node)
    }
    fn province(&self, node: u32) -> Option<u8> {
        self.inner.province(node)
    }
    fn water_side(&self, node: u32) -> Option<Side> {
        self.inner.water_side(node)
    }
    /// ★ THE FINEST LEVEL THAT HOLDS THE READ (ruling B95): the 16 m ground where the 16 m level serves the rung and its
    /// tiles are here, else the 128 m ground — so a reader of a rung's field (the eye's ground, a vertex's coarser
    /// surface) never falls to the 8 km ground where a 16 m tile is still on its way. A chunk is only built on the 16 m
    /// ground once every 16 m tile its stencils read is here, so this read never mixes the levels inside a chunk.
    fn fine_z(&self, face: Face, draw_rung: u8, cell_rung: u8, i: i32, j: i32) -> FineZ {
        if let Some(f) = self
            .finer
            .as_ref()
            .filter(|f| draw_rung <= f.level().top_rung())
            && let held @ FineZ::Held(_) = f.z(face, draw_rung, cell_rung, i, j)
        {
            return held;
        }
        self.book.z(face, draw_rung, cell_rung, i, j)
    }
    /// ★ THE FINEST LEVEL'S WATER (slice W2): the 16 m book's where it serves the rung and holds the place, else the 128 m
    /// book's — the same level the ground's read takes, so the water and the ground it lies on come from one tile.
    fn fine_water(&self, face: Face, draw_rung: u8, cell_rung: u8, i: i32, j: i32) -> FineWater {
        if let Some(f) = self
            .finer
            .as_ref()
            .filter(|f| draw_rung <= f.level().top_rung())
            && let held @ FineWater::Held { .. } = f.shore_water(face, draw_rung, cell_rung, i, j)
        {
            return held;
        }
        self.book.shore_water(face, draw_rung, cell_rung, i, j)
    }
}

/// The level-`k` sample of a level at `rung` at or before the centre of cell `i` of `cell_rung` along one axis, and the
/// fraction past it at [`NOISE_BITS`]: the centre in half rung-0 cells less the first sample's, by the samples' spacing
/// as a shift.
#[must_use]
pub fn sample_and_fraction(rung: u8, k: u32, cell_rung: u8, i: i32) -> (i64, Gi) {
    let bits = u32::from(rung) + 1 + k;
    let centre = (i64::from(i) << (u32::from(cell_rung) + 1)) + (1i64 << cell_rung);
    let rel = centre - (1i64 << (bits - 1));
    (
        rel >> bits,
        Gi::new((rel & ((1i64 << bits) - 1)) << (NOISE_BITS - bits)),
    )
}

/// A field that answers ONE fine read everywhere over a level macro field: the column's and the float read's tests
/// stand a chunk on it.
#[cfg(test)]
pub(crate) mod test_field {
    use super::FineZ;
    use crate::artifact::{Side, ZField};
    use vd_seed::bend::Face;

    pub(crate) struct OneFineRead(pub FineZ);

    impl ZField for OneFineRead {
        fn z_m(&self, _node: u32) -> Option<i16> {
            Some(0)
        }
        fn water_facies(&self, _node: u32) -> Option<(i16, u8)> {
            None
        }
        fn province(&self, _node: u32) -> Option<u8> {
            None
        }
        fn water_side(&self, _node: u32) -> Option<Side> {
            None
        }
        fn fine_z(&self, _face: Face, _draw_rung: u8, _cell_rung: u8, _i: i32, _j: i32) -> FineZ {
            self.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::home_planet;

    /// A quarter-metre word in the height's unit.
    fn gap(q: i64) -> Gi {
        Gi::new(q * QUARTER_STEPS) << LENGTH_BITS
    }

    #[test]
    fn a_cell_centre_finds_the_sample_before_it_and_the_fraction_past_it() {
        // A rung-7 cell and a rung-12 cell stand on their own samples.
        assert_eq!(sample_and_fraction(7, 0, 7, 5), (5, Gi::ZERO));
        assert_eq!(sample_and_fraction(7, 5, 12, 3), (3, Gi::ZERO));
        // Rung 0's cell 64 stands half a metre past sample 0's centre (64 m): 1/256 of a 128 m cell.
        assert_eq!(sample_and_fraction(7, 0, 0, 64), (0, Gi::new(1 << 20)));
        // The face's first metre lies before sample 0: sample −1, 129/256 past it; at level 2, 513/1024.
        assert_eq!(sample_and_fraction(7, 0, 0, 0), (-1, Gi::new(129 << 20)));
        assert_eq!(sample_and_fraction(7, 2, 0, 0), (-1, Gi::new(513 << 18)));
    }

    #[test]
    fn the_book_refuses_cells_that_do_not_nest_and_tiles_too_small_for_their_levels() {
        let body = home_planet();
        let n = body.ladder().n;
        // A node of one cell: rung 7's cells do not nest in it.
        let one = MacroLattice::with_edge(&body, n).expect("a lattice");
        assert_eq!(FineBook::new(FineLevel::L1, &body, &one, 0), None);
        // A node of 128 cells: one rung-7 cell a node, an 8-cell tile, too small for five mean levels.
        let small = MacroLattice::with_edge(&body, n >> 7).expect("a lattice");
        assert_eq!(FineBook::new(FineLevel::L1, &body, &small, 0), None);
        // The home planet's own lattice: 512-cell tiles.
        let lattice = MacroLattice::of(&body).expect("a lattice");
        assert_eq!(
            FineBook::new(FineLevel::L1, &body, &lattice, 0)
                .expect("a book")
                .side(),
            512
        );
    }

    /// The levels of tile `(tx, ty)` of face 0 over a global plane: the raster `4·(X + 2Y)` quarters at rung-7 cell
    /// `(X, Y)`, level `k` the stated plane `1000·k + 3·X_k + 5·Y_k` at its own cell `(X_k, Y_k)`.
    fn plane(tx: i32, ty: i32) -> Vec<Vec<i32>> {
        (0..=MEAN_LEVELS)
            .map(|k| {
                let s = 512i32 >> k;
                (0..s * s)
                    .map(|c| {
                        let (x, y) = (tx * s + (c & (s - 1)), ty * s + (c >> (9 - k)));
                        if k == 0 {
                            4 * (x + 2 * y)
                        } else {
                            1_000 * k as i32 + 3 * x + 5 * y
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_held_tile_reads_its_samples_its_means_and_the_curve_between_and_names_a_missing_neighbour()
    {
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let sea_m = -100i16;
        let mut book = FineBook::new(FineLevel::L1, &body, &lattice, sea_m).expect("a book");
        let face = Face::from_index(0).expect("a face");
        for ty in 3..=5 {
            for tx in 2..=4 {
                assert!(book.insert(TileKey { face, tx, ty }, plane(tx, ty)));
            }
        }
        let sea = Gi::new(i64::from(sea_m) * STEPS_PER_M) << LENGTH_BITS;
        // A rung-7 column on tile (3, 4) reads its own sample.
        let (i, j) = (3 * 512 + 10, 4 * 512 + 20);
        assert_eq!(
            book.z(face, 7, 7, i, j),
            FineZ::Held(gap(i64::from(4 * (i + 2 * j))) + sea)
        );
        // A rung-12 column reads level 5's mean at its own cell.
        let (i, j) = (3 * 16 + 2, 4 * 16 + 5);
        assert_eq!(
            book.z(face, 12, 12, i, j),
            FineZ::Held(gap(i64::from(5_000 + 3 * i + 5 * j)) + sea)
        );
        // A rung-3 column reads the curve between the raster's samples: a plane, within a millimetre.
        let (i, j) = (24_576 + 1_003, 32_768 + 2_007);
        let (x256, y256) = (i64::from(i) * 16 + 8 - 128, i64::from(j) * 16 + 8 - 128);
        let want = Gi::new((x256 + 2 * y256) << 27) + sea;
        let FineZ::Held(got) = book.z(face, 3, 3, i, j) else {
            panic!("held")
        };
        assert!(
            (got - want).raw().abs() < 1 << 25,
            "{got:?} against {want:?}"
        );
        // A column at tile (4, 4)'s east edge needs tile (5, 4), which the book does not hold.
        assert_eq!(
            book.z(face, 7, 7, 5 * 512 - 1, 4 * 512 + 100),
            FineZ::Missing(TileKey { face, tx: 5, ty: 4 })
        );
        // Rung 13 reads the node's row: no fine level serves it.
        assert_eq!(book.z(face, 13, 13, 0, 0), FineZ::Outside);
    }

    #[test]
    fn a_tile_with_a_level_missing_or_missized_is_not_held() {
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let mut book = FineBook::new(FineLevel::L1, &body, &lattice, 0).expect("a book");
        let key = TileKey {
            face: Face::from_index(1).expect("a face"),
            tx: 0,
            ty: 0,
        };
        let mut short = plane(0, 0);
        short.pop();
        assert!(!book.insert(key, short));
        let mut missized = plane(0, 0);
        missized[2].pop();
        assert!(!book.insert(key, missized));
        assert!(!book.holds(key));
        assert!(book.insert(key, plane(0, 0)));
        assert!(book.holds(key));
    }

    #[test]
    fn a_book_from_the_head_numbers_is_the_body_book_and_refuses_a_side_too_small() {
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        assert_eq!(
            FineBook::with_side(FineLevel::L1, lattice.edge, 9, -100),
            FineBook::new(FineLevel::L1, &body, &lattice, -100)
        );
        // Smaller than its eight nodes, or too small for five mean levels.
        assert_eq!(FineBook::with_side(FineLevel::L1, lattice.edge, 2, 0), None);
        assert_eq!(FineBook::with_side(FineLevel::L1, lattice.edge, 4, 0), None);
        // The level applies where the solve's water exists.
        assert_eq!(
            [applies(DRY_M), applies(0), applies(-3_000)],
            [false, true, true]
        );
    }

    #[test]
    fn a_chunk_names_the_tiles_its_stencils_read_inside_a_face_and_across_its_edge() {
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let book = FineBook::new(FineLevel::L1, &body, &lattice, 0).expect("a book");
        let face = Face::from_index(0).expect("a face");
        // A rung-12 chunk reads level 5 (16 samples a tile): samples 121 … 189 by 183 … 251, tiles 7 … 11 by 11 … 15.
        let keys = book.tiles_of_chunk(face, 12, 2, 3);
        assert_eq!(keys.len(), 25);
        assert_eq!(
            (keys[0], keys[24]),
            (
                TileKey {
                    face,
                    tx: 7,
                    ty: 11
                },
                TileKey {
                    face,
                    tx: 11,
                    ty: 15
                }
            )
        );
        // A rung-0 chunk at the face's first column: its own tile and the partner face's tile across the edge.
        let n7 = (lattice.edge << 6) as i32;
        let c = chart_cell(face, -1, 5, n7).expect("the partner face's cell");
        let partner = TileKey {
            face: Face::from_index(c.face).expect("a face"),
            tx: c.i >> 9,
            ty: c.j >> 9,
        };
        let mut want = vec![TileKey { face, tx: 0, ty: 0 }, partner];
        want.sort();
        assert_eq!(book.tiles_of_chunk(face, 0, 0, 10), want);
        // Rung 13 reads the node's row.
        assert_eq!(book.tiles_of_chunk(face, 13, 0, 0), Vec::new());
    }

    #[test]
    fn a_far_tile_holds_its_coarse_copies_and_a_finer_level_joins_them_below() {
        // B86 item 3: the tile (3, 4) arrives with its levels 3 … 5 only; a rung-10 chunk reads it, a rung-9 one waits.
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let mut book = FineBook::new(FineLevel::L1, &body, &lattice, 0).expect("a book");
        let face = Face::from_index(0).expect("a face");
        let key = TileKey { face, tx: 3, ty: 4 };
        let all = plane(3, 4);
        // Levels that stop short of the coarsest, or have another size, are refused.
        assert!(!book.insert_from(key, 3, all[3..5].to_vec()));
        assert!(!book.insert_from(key, 3, vec![all[2].clone(), all[4].clone(), all[5].clone()]));
        assert!(!book.insert_from(key, 6, Vec::new()));
        assert!(book.insert_from(key, 3, all[3..].to_vec()));
        assert_eq!(book.held_from(key), Some(3));
        let mut whole = FineBook::new(FineLevel::L1, &body, &lattice, 0).expect("a book");
        assert!(whole.insert(key, all.clone()));
        let (i, j) = ((3 * 64 + 10) << 3, (4 * 64 + 20) << 3);
        assert_eq!(
            book.z(face, 10, 10, 3 * 64 + 10, 4 * 64 + 20),
            whole.z(face, 10, 10, 3 * 64 + 10, 4 * 64 + 20)
        );
        assert_ne!(
            whole.z(face, 10, 10, 3 * 64 + 10, 4 * 64 + 20),
            FineZ::Missing(key)
        );
        assert_eq!(
            book.z(face, 9, 9, 3 * 128 + 20, 4 * 128 + 40),
            FineZ::Missing(key)
        );
        assert!(book.holds_all(&[key], 10));
        assert!(!book.holds_all(&[key], 9));
        // A finer level with a gap under the held ones is refused; one that reaches them joins below them.
        assert!(!book.insert_from(key, 0, all[0..2].to_vec()));
        assert!(book.insert_from(key, 1, all[1..3].to_vec()));
        assert_eq!(book.held_from(key), Some(1));
        // Levels the book already holds stand as they are.
        assert!(book.insert_from(key, 2, all[2..].to_vec()));
        assert_eq!(book.held_from(key), Some(1));
        // The whole tile now reads as the full insert reads it, at every rung it holds.
        assert_eq!(
            book.z(face, 8, 0, i + 77, j + 3),
            whole.z(face, 8, 0, i + 77, j + 3)
        );
        assert_eq!(
            book.z(face, 12, 12, 3 * 16 + 5, 4 * 16 + 6),
            whole.z(face, 12, 12, 3 * 16 + 5, 4 * 16 + 6)
        );
        assert!(book.insert_from(key, 0, all[0..1].to_vec()));
        assert_eq!(book.held_from(key), Some(0));
        assert_eq!(book, whole);
    }

    #[test]
    fn a_held_tile_is_shared_counted_and_let_go_and_a_fine_field_reads_its_book() {
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let mut book = FineBook::new(FineLevel::L1, &body, &lattice, 0).expect("a book");
        let face = Face::from_index(0).expect("a face");
        assert_eq!(
            (book.len(), book.is_empty(), book.side_shift()),
            (0, true, 9)
        );
        for (tx, ty) in [(3, 4), (2, 4)] {
            assert!(book.insert(TileKey { face, tx, ty }, plane(tx, ty)));
        }
        assert_eq!(
            book.keys().collect::<Vec<_>>(),
            vec![
                TileKey { face, tx: 2, ty: 4 },
                TileKey { face, tx: 3, ty: 4 }
            ]
        );
        assert!(book.holds_all(&[TileKey { face, tx: 2, ty: 4 }], 7));
        assert!(!book.holds_all(&[TileKey { face, tx: 9, ty: 9 }], 7));
        // A field over the book: the macro words from its inner field, the fine ground from the book.
        let field = FineField {
            inner: Arc::new(test_field::OneFineRead(FineZ::Outside)),
            book: Arc::new(book.clone()),
            finer: None,
        };
        assert_eq!(
            (
                field.z_m(7),
                field.level(),
                field.water_facies(7),
                field.province(7),
                field.water_side(7)
            ),
            (Some(0), 0, None, None, None)
        );
        assert_eq!(
            field.fine_z(face, 7, 7, 3 * 512 + 10, 4 * 512 + 20),
            book.z(face, 7, 7, 3 * 512 + 10, 4 * 512 + 20)
        );
        assert_eq!(field.fine_z(face, 13, 13, 0, 0), FineZ::Outside);
        // ★ THE 16 m BOOK UNDER IT (ruling B95): the field reads the finer book for the rungs it serves (rungs 0–6) and the
        // 128 m book above them; the 16 m book's own levels are the rung-4 raster and its means at rungs 5 and 6.
        let mut fine = FineBook::new(FineLevel::L2, &body, &lattice, 0).expect("a 16 m book");
        assert_eq!(
            (fine.level(), fine.side_shift(), fine.side()),
            (FineLevel::L2, 9, 512)
        );
        let node = TileKey {
            face,
            tx: 40,
            ty: 33,
        };
        let levels: Vec<Vec<i32>> = (0..=2u32)
            .map(|k| vec![100 * (k as i32 + 1); 1 << (2 * (9 - k))])
            .collect();
        assert!(!fine.insert(node, levels[..2].to_vec()));
        assert!(fine.insert(node, levels));
        let both = FineField {
            inner: Arc::new(test_field::OneFineRead(FineZ::Outside)),
            book: Arc::new(book.clone()),
            finer: Some(Arc::new(fine.clone())),
        };
        let (i, j) = (40 * 512 + 100, 33 * 512 + 100);
        assert_eq!(both.fine_z(face, 4, 4, i, j), FineZ::Held(gap(100)));
        assert_eq!(
            both.fine_z(face, 6, 6, i >> 2, j >> 2),
            FineZ::Held(gap(300))
        );
        assert_eq!(
            both.fine_z(face, 7, 7, 3 * 512 + 10, 4 * 512 + 20),
            book.z(face, 7, 7, 3 * 512 + 10, 4 * 512 + 20)
        );
        assert_eq!(fine.z(face, 7, 7, i >> 3, j >> 3), FineZ::Outside);
        // Where the 16 m tile is not here yet the field reads the 128 m ground under it, never nothing.
        let (oi, oj) = ((3 * 512 + 10) << 3, (4 * 512 + 20) << 3);
        assert_eq!(
            fine.z(face, 4, 4, oi, oj),
            FineZ::Missing(TileKey {
                face,
                tx: 24,
                ty: 32
            })
        );
        assert_eq!(both.fine_z(face, 4, 4, oi, oj), book.z(face, 4, 4, oi, oj));
        assert!(matches!(both.fine_z(face, 4, 4, oi, oj), FineZ::Held(_)));
        // The rung-4 chunk (332, 274) and its stencil lie inside the node's tile.
        assert_eq!(fine.tiles_of_chunk(face, 4, 332, 274), vec![node]);
        assert!(fine.holds_all(&[node], 4));
        assert_eq!(fine.tiles_of_chunk(face, 7, 0, 0), Vec::new());
        // Letting a tile go.
        assert!(book.remove(TileKey { face, tx: 2, ty: 4 }));
        assert!(!book.remove(TileKey { face, tx: 2, ty: 4 }));
        assert_eq!(book.len(), 1);
    }

    #[test]
    fn a_read_past_the_face_edge_names_the_partner_face_tile_and_a_cube_corner_the_face_own() {
        let body = home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let book = FineBook::new(FineLevel::L1, &body, &lattice, 0).expect("a book");
        let face = Face::from_index(0).expect("a face");
        let n7 = (lattice.edge << 6) as i32;
        // The stencil's first sample of the face's first column lies on the partner face.
        let c = chart_cell(face, -1, 99, n7).expect("the partner face's cell");
        let partner = TileKey {
            face: Face::from_index(c.face).expect("a face"),
            tx: c.i >> 9,
            ty: c.j >> 9,
        };
        assert_eq!(book.z(face, 7, 7, 0, 100), FineZ::Missing(partner));
        // At the face's corner the first sample lies in the cube corner's diagonal region: the face's own corner cell.
        assert_eq!(
            book.z(face, 7, 7, 0, 0),
            FineZ::Missing(TileKey { face, tx: 0, ty: 0 })
        );
    }
}
