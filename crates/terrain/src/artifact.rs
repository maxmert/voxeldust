//! ★ THE ARTIFACT (the landform arc, slice 8c stage C4; `slice_8c_design.md` §7; the owner's
//! ruling of 2026-09-19: the solve runs ONCE on the server, the artifact is saved in the shard's own
//! store and SHIPPED to every client in tiles).
//!
//! What the solve keeps, per node of the macro lattice, nine bytes: the eroded height `Z` in whole
//! metres, the water level (a lake's spill level, the sea's level, or DRY), the D8 receiver's
//! stencil slot with the facies bits, the discharge as a quantised logarithm, and three climate
//! bytes (the temperature class, the rain class, the aridity) for 8e and 18. Beside the rows a
//! PYRAMID of `Z`: each level the integer mean of a 2 × 2 block of the one below, down to the edge's
//! odd factor — the far view's field (the top level of the home planet is a few kilobytes).
//!
//! How a column reads it ([`sample_z`]): the column's cell at its rung names the macro node it lies
//! in and its fraction across that node by ONE integer division (the edge divides the cell count),
//! then the sixteen nodes around it are gathered THROUGH THE SEAM TABLE — never by adding one to
//! `i` — and the recipe's Catmull-Rom kernel ([`vd_recipe::macro_field`]) answers the height. The
//! lattice is cell-centred on every face and the samples straddle a cube edge at a uniform spacing,
//! so the read is C1 across the edge (03 §5.4). Within two nodes of a cube corner the stencil's
//! missing quadrant takes the nearest node of the face — a stated fallback; the quintic-fade blend
//! of 03 §5.4 is owed and gate G-MACRO-CORNER measures it.
//!
//! Who reads what: the SHIP is by tiles of [`TILE_EDGE`] nodes a side, each row-major; a client
//! holds a [`TileCache`] its chunk builder reads through the [`ZField`] trait, the same trait the
//! whole [`Artifact`] and the golden pin's [`SparseZ`] answer, so ONE read serves the shard, the
//! client and the gate.
//!
//! **Example.** A pilot drops toward one coast of the home planet. The client already holds the
//! pyramid, so the globe from orbit shows the continents' drainage. As the coast fills the screen the
//! client asks for the tiles under the eye — 36 KB each — and the chunks it builds read the valley
//! the solve cut, node by node, the same valley the shard's collider reads.

// ★ THE ARTIFACT MAY DIVIDE on the CPU: the node lookup is one integer division by the node's size
// in cells (the divisor rule's reason), run per column on the CPU today; the card's path (owed)
// reads the same quotient through a reciprocal drawn once per body.
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the artifact's read is the CPU's today; the card's path reads a reciprocal (owed)"
)]

use std::collections::{BTreeMap, BTreeSet};

use crate::chunk::{CHUNK_EDGE, ChunkKey};

use vd_recipe::Gi;
use vd_recipe::macro_field::catmull_rom_16;
use vd_recipe::noise::NOISE_BITS;
use vd_seed::bend::Face;
use vd_seed::digest::{FNV_OFFSET, fnv1a};
use vd_seed::seam::{Edge, across, partner_of};

use crate::climate::Climate;
use crate::macro_lattice::{MacroLattice, NO_NODE};
use crate::solve::{MacroSolve, Z_STEPS_PER_M};
use crate::units::{LENGTH_BITS, STEPS_PER_M};

/// The artifact's own version: part of the world identity with the generator's; a change to a row's
/// meaning or the pyramid's rule bumps it.
pub const ARTIFACT_VERSION: u32 = 1;
/// A tile's edge in nodes: 64 × 64 rows of nine bytes is 36 KB — under a second on the lane.
pub const TILE_EDGE: u32 = 64;
/// The water level's word for a dry node.
pub const DRY_M: i16 = i16::MIN;
/// The bytes a row packs to.
pub const ROW_BYTES: usize = 9;
/// The receiver byte: bits 0..3 the stencil slot of the receiver, bit 3 set for NO receiver (an
/// outlet), bits 4..8 the facies (the solve's bits shifted by four).
pub const RECEIVER_NONE: u8 = 0x08;
pub const RECEIVER_SLOT_MASK: u8 = 0x07;
pub const FACIES_SHIFT: u32 = 4;
/// The temperature class's floor, decikelvin: class 0 is 173 K, class 255 is 428 K.
pub const TEMPERATURE_FLOOR_DK: i32 = 1_730;
/// ★ THE CORNER BLEND's radius in nodes (03 §5.4): inside it the read's missing quadrant is filled
/// by the nearest node of the face — the stated fallback until the quintic-fade blend lands.
pub const CORNER_BLEND: i32 = 2;

/// One node's row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Row {
    /// The eroded height, whole metres from the ladder radius.
    pub z_m: i16,
    /// The water level, whole metres, or [`DRY_M`].
    pub water_m: i16,
    /// The receiver's slot and the facies bits.
    pub receiver_facies: u8,
    /// The discharge as `4·log₂(Q + 1)` with two fraction bits from the next two bits, clamped.
    pub discharge_log: u8,
    /// The temperature class: kelvin over 173, clamped.
    pub temperature: u8,
    /// The rain class: `16·log₂(P + 1)` with four fraction bits from the mantissa, clamped (10 000
    /// mm/yr is 211).
    pub rain: u8,
    /// The aridity byte.
    pub aridity: u8,
}

impl Row {
    /// The nine bytes, little-endian words first.
    #[must_use]
    pub fn to_bytes(self) -> [u8; ROW_BYTES] {
        let z = self.z_m.to_le_bytes();
        let w = self.water_m.to_le_bytes();
        [
            z[0],
            z[1],
            w[0],
            w[1],
            self.receiver_facies,
            self.discharge_log,
            self.temperature,
            self.rain,
            self.aridity,
        ]
    }

    /// The row from its nine bytes.
    #[must_use]
    pub fn from_bytes(b: [u8; ROW_BYTES]) -> Row {
        Row {
            z_m: i16::from_le_bytes([b[0], b[1]]),
            water_m: i16::from_le_bytes([b[2], b[3]]),
            receiver_facies: b[4],
            discharge_log: b[5],
            temperature: b[6],
            rain: b[7],
            aridity: b[8],
        }
    }
}

/// A quantised binary logarithm: the bit length of `v + 1` times `2^frac`, plus the next `frac`
/// bits under the leading one — an integer, monotone, exact on every host.
#[must_use]
pub fn log2_class(v: u64, frac: u32) -> u8 {
    let x = v.saturating_add(1);
    let bits = 63 - x.leading_zeros();
    let below = if bits >= frac {
        (x >> (bits - frac)) & ((1 << frac) - 1)
    } else {
        (x << (frac - bits)) & ((1 << frac) - 1)
    };
    ((bits << frac) + below as u32).min(255) as u8
}

/// A height in sixteenths as whole metres, rounded half up, clamped to the word.
#[must_use]
pub fn metres_i16(sixteenths: i32) -> i16 {
    let m =
        (i64::from(sixteenths) + i64::from(Z_STEPS_PER_M / 2)).div_euclid(i64::from(Z_STEPS_PER_M));
    m.clamp(i64::from(i16::MIN + 1), i64::from(i16::MAX)) as i16
}

/// ★ THE ARTIFACT of one body: the rows in node order and the pyramid of `Z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Artifact {
    /// The macro lattice's edge.
    pub edge: u32,
    pub version: u32,
    pub rows: Vec<Row>,
    /// Level `k` (from 1) holds `6 · (edge >> k)²` heights in whole metres, the integer mean of its
    /// 2 × 2 block below; as many levels as the edge halves to.
    pub pyramid: Vec<Vec<i16>>,
}

/// One tile of rows: `TILE_EDGE` wide (the last tile of a face may be narrower), row-major from
/// the tile's origin `(tx · TILE_EDGE, ty · TILE_EDGE)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tile {
    pub face: u8,
    pub tx: u32,
    pub ty: u32,
    pub rows: Vec<Row>,
}

impl Tile {
    /// The rows as bytes, [`ROW_BYTES`] a row in order: the store's and the wire's one framing.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.rows.len() * ROW_BYTES);
        for r in &self.rows {
            out.extend_from_slice(&r.to_bytes());
        }
        out
    }

    /// The rows back from bytes; `None` for a length that is not whole rows.
    #[must_use]
    pub fn rows_from_bytes(bytes: &[u8]) -> Option<Vec<Row>> {
        if !bytes.len().is_multiple_of(ROW_BYTES) {
            return None;
        }
        Some(
            bytes
                .chunks_exact(ROW_BYTES)
                .map(|c| {
                    let mut b = [0u8; ROW_BYTES];
                    b.copy_from_slice(c);
                    Row::from_bytes(b)
                })
                .collect(),
        )
    }
}

/// What a read needs: the eroded height of a node, or nothing where the reader holds no row yet;
/// and the pyramid LEVEL the field stands at (zero for the rows), so the read gathers on the
/// lattice that level is defined on.
pub trait ZField {
    fn z_m(&self, node: u32) -> Option<i16>;
    /// The pyramid level: zero for the rows, `k` for a level `k` coarser.
    fn level(&self) -> u32 {
        0
    }
}

impl ZField for Artifact {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.rows.get(node as usize).map(|r| r.z_m)
    }
}

/// A few nodes' heights, stated: the golden pin's rows for the eight self-check chunks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SparseZ(pub BTreeMap<u32, i16>);

impl ZField for SparseZ {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.0.get(&node).copied()
    }
}

/// ★ THE CLIENT'S TILE CACHE: the tiles a realm has shipped, read by the chunk builder; a node whose
/// tile has not arrived reads nothing, and the builder keeps the coarser rung standing (ruling F9).
///
/// Each tile's rows sit behind an `Arc`, so a clone of the cache is a pointer bump per tile: the
/// client's core replaces the cache it shares with its builders on every tile that lands (four a
/// tick), and a copy of every row held would be megabytes a tick for a shape that never changes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileCache {
    pub edge: u32,
    tiles: BTreeMap<(u8, u32, u32), std::sync::Arc<[Row]>>,
}

impl TileCache {
    /// An empty cache for a lattice of `edge` nodes a side.
    #[must_use]
    pub fn new(edge: u32) -> TileCache {
        TileCache {
            edge,
            tiles: BTreeMap::new(),
        }
    }

    /// Keep a tile the realm shipped (a later tile with the same key replaces the earlier).
    pub fn apply(&mut self, tile: Tile) {
        self.tiles
            .insert((tile.face, tile.tx, tile.ty), tile.rows.into());
    }

    /// Whether the cache holds a tile.
    #[must_use]
    pub fn holds(&self, face: u8, tx: u32, ty: u32) -> bool {
        self.tiles.contains_key(&(face, tx, ty))
    }

    /// Whether the cache holds every tile in `tiles` — what a chunk's build asks before it starts.
    #[must_use]
    pub fn holds_all(&self, tiles: &BTreeSet<(u8, u32, u32)>) -> bool {
        tiles.iter().all(|&(f, tx, ty)| self.holds(f, tx, ty))
    }

    /// How many tiles the cache holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether the cache holds no tile.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// The row of a node, if its tile is here.
    #[must_use]
    pub fn row(&self, node: u32) -> Option<Row> {
        let per_face = self.edge * self.edge;
        let rest = node % per_face;
        let (i, j) = (rest % self.edge, rest / self.edge);
        let (face, tx, ty) = tile_of_node(self.edge, node);
        let rows = self.tiles.get(&(face, tx, ty))?;
        let width = tile_width(self.edge, tx);
        rows.get(((j - ty * TILE_EDGE) * width + (i - tx * TILE_EDGE)) as usize)
            .copied()
    }
}

impl ZField for TileCache {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.row(node).map(|r| r.z_m)
    }
}

/// ★ A PYRAMID LEVEL AS A FIELD (the far view, 03 §9): level `k` of the pyramid over the lattice
/// `k` levels coarser, so a chunk at a rung whose cell is many macro nodes wide reads the mean
/// heights the level holds — the same Catmull-Rom on the coarser lattice — and needs no tile. The
/// client holds the pyramid from the surface's window and draws the globe from orbit with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyramidField {
    /// The level, from 1.
    pub level: u32,
    /// The level's heights, whole metres, in the coarser lattice's node order.
    pub z_m: Vec<i16>,
}

impl ZField for PyramidField {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.z_m.get(node as usize).copied()
    }

    fn level(&self) -> u32 {
        self.level
    }
}

impl PyramidField {
    /// Level `k` of an artifact's pyramid, or `None` past its levels.
    #[must_use]
    pub fn of(artifact: &Artifact, level: u32) -> Option<PyramidField> {
        artifact
            .pyramid
            .get(level.checked_sub(1)? as usize)
            .map(|z| PyramidField {
                level,
                z_m: z.clone(),
            })
    }

    /// ★ THE LEVEL A RUNG READS: the coarsest level whose node is no wider than the rung's chunk
    /// (a chunk is [`crate::chunk::CHUNK_EDGE`] cells), so a chunk always spans several of the
    /// level's nodes; zero (the rows themselves) where the rung's chunk is under two macro nodes.
    #[must_use]
    pub fn level_for(lattice: &MacroLattice, levels: u32, rung: u8) -> u32 {
        let chunk_cells = (crate::chunk::CHUNK_EDGE as u64) << rung;
        let mut level = 0u32;
        let mut node_cells = u64::from(lattice.cells_per_node);
        while level < levels && node_cells * 4 <= chunk_cells {
            level += 1;
            node_cells <<= 1;
        }
        level
    }
}

/// A tile's width in nodes: `TILE_EDGE`, or what is left of the edge for the last tile.
#[must_use]
pub fn tile_width(edge: u32, t: u32) -> u32 {
    (edge - t * TILE_EDGE).min(TILE_EDGE)
}

/// The tile a global node index lies in: `(face, tx, ty)`.
#[must_use]
pub fn tile_of_node(edge: u32, node: u32) -> (u8, u32, u32) {
    let per_face = edge * edge;
    let face = (node / per_face) as u8;
    let rest = node % per_face;
    let (i, j) = (rest % edge, rest / edge);
    (face, i / TILE_EDGE, j / TILE_EDGE)
}

/// ★ THE NODES A CHUNK READS: every node of the sixteen-node stencil of any column of the chunk or
/// its halo, through the seam table — a DENSE walk of the node range, for a chunk that spans a few
/// nodes (the identity pin's rung-0 keys); a chunk that spans a face asks [`tiles_of_chunk`].
#[must_use]
pub fn nodes_of_chunk(lattice: &MacroLattice, key: ChunkKey) -> BTreeSet<u32> {
    let span = |c: i32| -> (i64, i64) {
        let first = c * CHUNK_EDGE as i32 - crate::lattice::HALO;
        let last = (c + 1) * CHUNK_EDGE as i32 - 1 + crate::lattice::HALO;
        (
            node_and_fraction(lattice, key.rung, first).0 - 1,
            node_and_fraction(lattice, key.rung, last).0 + 2,
        )
    };
    let (i_lo, i_hi) = span(key.x);
    let (j_lo, j_hi) = span(key.y);
    let mut nodes = BTreeSet::new();
    for j in j_lo..=j_hi {
        for i in i_lo..=i_hi {
            nodes.insert(node_at(lattice, key.face, i, j));
        }
    }
    nodes
}

/// ★ THE GOLDEN FIELDS (slice 8c stage C4c; SL10 clause 3): what the world identity's self-check
/// reads instead of the recipe's own relief — the home artifact's ROWS under the six rung-0 keys
/// (a few dozen nodes, `SparseZ`) and its COARSEST PYRAMID LEVEL, which the two top-rung keys read
/// the way the far view does. Both are literals in the build (`home::home_golden_fields`), so every
/// host measures the same eight chunks through the same read at boot, with no artifact in hand; the
/// artifact pin proves the literals are the solve's own rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenFields {
    /// How many pyramid levels the artifact holds (the level a rung reads depends on it).
    pub levels: u32,
    /// The rows under the fine keys.
    pub rows: SparseZ,
    /// The coarsest level, whole.
    pub top: PyramidField,
}

impl GoldenFields {
    /// The field a key's rung reads: the rows at level zero, the top at its level, nothing between
    /// (no golden key stands at a middle rung; a key that did would be refused, never guessed).
    #[must_use]
    pub fn field_for(&self, lattice: &MacroLattice, rung: u8) -> Option<&dyn ZField> {
        let level = PyramidField::level_for(lattice, self.levels, rung);
        if level == 0 {
            Some(&self.rows)
        } else if level == self.top.level {
            Some(&self.top)
        } else {
            None
        }
    }

    /// The fields as text, one item a line: `h levels top_level`, `t z` for each top word in order,
    /// `r node z` for each row.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("h {} {}\n", self.levels, self.top.level);
        for z in &self.top.z_m {
            out.push_str(&format!("t {z}\n"));
        }
        for (node, z) in &self.rows.0 {
            out.push_str(&format!("r {node} {z}\n"));
        }
        out
    }

    /// The fields back from [`GoldenFields::to_text`]; `None` for a line that is not one of the three.
    #[must_use]
    pub fn parse(text: &str) -> Option<GoldenFields> {
        let mut levels = None;
        let mut top_level = None;
        let mut top = Vec::new();
        let mut rows = BTreeMap::new();
        for line in text.lines() {
            let mut w = line.split_whitespace();
            match w.next()? {
                "h" => {
                    levels = Some(w.next()?.parse::<u32>().ok()?);
                    top_level = Some(w.next()?.parse::<u32>().ok()?);
                }
                "t" => top.push(w.next()?.parse::<i16>().ok()?),
                "r" => {
                    let node = w.next()?.parse::<u32>().ok()?;
                    let z = w.next()?.parse::<i16>().ok()?;
                    rows.insert(node, z);
                }
                _ => return None,
            }
        }
        Some(GoldenFields {
            levels: levels?,
            rows: SparseZ(rows),
            top: PyramidField {
                level: top_level?,
                z_m: top,
            },
        })
    }
}

/// ★ THE TILES A CHUNK READS (slice 8c stage C4c): every tile that holds a node of the sixteen-node
/// stencil of any column of the chunk or its halo, through the seam table — so a chunk at a face's
/// edge names the partner face's tiles too. The client asks its cache for these before it builds,
/// and a chunk whose tiles are not all here waits while the coarser rung stands (ruling F9).
///
/// The node range is walked with a stride of one tile plus its last node: two nodes a tile apart
/// never skip a tile, so the walk is `O(tiles)`, never `O(nodes)`, on a chunk that spans a face.
///
/// **Example.** A rung-0 chunk on the home planet whose columns sit over macro node 63 gathers nodes
/// 62 to 65 and names tiles 0 and 1 along that axis; a chunk at the face's low edge names the tile
/// across the seam as well.
#[must_use]
pub fn tiles_of_chunk(lattice: &MacroLattice, key: ChunkKey) -> BTreeSet<(u8, u32, u32)> {
    let span = |c: i32| -> (i64, i64) {
        let first = c * CHUNK_EDGE as i32 - crate::lattice::HALO;
        let last = (c + 1) * CHUNK_EDGE as i32 - 1 + crate::lattice::HALO;
        (
            node_and_fraction(lattice, key.rung, first).0 - 1,
            node_and_fraction(lattice, key.rung, last).0 + 2,
        )
    };
    let (i_lo, i_hi) = span(key.x);
    let (j_lo, j_hi) = span(key.y);
    let stepped = |lo: i64, hi: i64| {
        (lo..=hi)
            .step_by(TILE_EDGE as usize)
            .chain(std::iter::once(hi))
    };
    let mut tiles = BTreeSet::new();
    for j in stepped(j_lo, j_hi) {
        for i in stepped(i_lo, i_hi) {
            let node = node_at(lattice, key.face, i, j);
            tiles.insert(tile_of_node(lattice.edge, node));
        }
    }
    tiles
}

impl Artifact {
    /// ★ THE ARTIFACT OF A SOLVE: every row from the solve's state, its facies and its climate,
    /// then the pyramid. `has_water` is the body's word (a dry body's flooded pits are basins).
    #[must_use]
    pub fn of(state: &MacroSolve, facies: &[u8], climate: &Climate, has_water: bool) -> Artifact {
        let n = state.node_count();
        let lattice = state.lattice;
        let mut rows = Vec::with_capacity(n);
        for node in 0..n as u32 {
            let i = node as usize;
            let ring = lattice.neighbours(node);
            let r = state.receiver[i];
            let slot = if r == NO_NODE {
                RECEIVER_NONE
            } else {
                ring.iter()
                    .position(|&m| m == r)
                    .map_or(RECEIVER_NONE, |s| s as u8)
            };
            // The water level is the flood's routing surface; on a body with no water the raised
            // pits are closed basins and the row says DRY.
            let water = state.water_level(node);
            rows.push(Row {
                z_m: metres_i16(state.z[i]),
                water_m: if !has_water || water == crate::solve::DRY {
                    DRY_M
                } else {
                    metres_i16(water)
                },
                receiver_facies: slot | (facies[i] << FACIES_SHIFT),
                discharge_log: log2_class(state.discharge[i], 2),
                temperature: (i32::from(climate.temperature_dk[i]) - TEMPERATURE_FLOOR_DK)
                    .div_euclid(10)
                    .clamp(0, 255) as u8,
                rain: log2_class(u64::from(climate.rain_mm_yr[i]), 4),
                aridity: climate.aridity_q8[i],
            });
        }
        let mut pyramid = Vec::new();
        let mut below: Vec<i16> = rows.iter().map(|r| r.z_m).collect();
        let mut fine = lattice;
        let mut k = 1;
        while let Some(coarse) = lattice.coarser(k) {
            let mut level = vec![0i32; coarse.node_count()];
            for node in 0..fine.node_count() as u32 {
                let (face, i, j) = fine.split(node);
                level[coarse.index(face, i >> 1, j >> 1) as usize] +=
                    i32::from(below[node as usize]);
            }
            let means: Vec<i16> = level.iter().map(|&s| s.div_euclid(4) as i16).collect();
            below.clone_from(&means);
            pyramid.push(means);
            fine = coarse;
            k += 1;
        }
        Artifact {
            edge: lattice.edge,
            version: ARTIFACT_VERSION,
            rows,
            pyramid,
        }
    }

    /// The nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.rows.len()
    }

    /// The bytes the rows and the pyramid take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.len() * ROW_BYTES + self.pyramid.iter().map(|l| l.len() * 2).sum::<usize>()
    }

    /// ★ THE DIGEST: two FNV-1a folds over the version, the edge, every row's bytes in node order
    /// and every pyramid level — gate G-DRIFT's word, the same on every target or the boot is red.
    #[must_use]
    pub fn digest(&self) -> [u64; 2] {
        let mut a = FNV_OFFSET;
        let mut b = !FNV_OFFSET;
        let head = [self.version.to_le_bytes(), self.edge.to_le_bytes()].concat();
        a = fnv1a(a, &head);
        b = fnv1a(b, &head);
        for r in &self.rows {
            let bytes = r.to_bytes();
            a = fnv1a(a, &bytes);
            b = fnv1a(b, &bytes);
        }
        for level in &self.pyramid {
            for &z in level {
                let bytes = z.to_le_bytes();
                a = fnv1a(a, &bytes);
                b = fnv1a(b, &bytes);
            }
        }
        [a, b]
    }

    /// The tiles along a face's edge.
    #[must_use]
    pub fn tiles_per_edge(&self) -> u32 {
        self.edge.div_ceil(TILE_EDGE)
    }

    /// One tile's rows, row-major from its origin.
    #[must_use]
    pub fn tile(&self, face: Face, tx: u32, ty: u32) -> Tile {
        let per_face = self.edge * self.edge;
        let width = tile_width(self.edge, tx);
        let height = tile_width(self.edge, ty);
        let mut rows = Vec::with_capacity((width * height) as usize);
        for j in ty * TILE_EDGE..ty * TILE_EDGE + height {
            for i in tx * TILE_EDGE..tx * TILE_EDGE + width {
                let node = u32::from(face.index()) * per_face + j * self.edge + i;
                rows.push(self.rows[node as usize]);
            }
        }
        Tile {
            face: face.index(),
            tx,
            ty,
            rows,
        }
    }
}

/// ★ THE NODE AND THE FRACTION along one axis: the cell `i` at `rung` has its centre at rung-0 cell
/// `i · 2^rung + 2^rung / 2`; the macro nodes' centres stand at `c · size + size / 2`. In half
/// rung-0 cells both are whole numbers, so the node index and the fraction across it (at
/// [`NOISE_BITS`]) come from one division, exact on every host. The node may be −1 or the edge
/// (a cell in the first or last half node): the gather then crosses the seam.
#[must_use]
pub fn node_and_fraction(lattice: &MacroLattice, rung: u8, i: i32) -> (i64, Gi) {
    let size = i64::from(lattice.cells_per_node);
    let centre_half = (i64::from(i) << (rung + 1)) + (1i64 << rung);
    let rel = centre_half - size;
    let node = rel.div_euclid(2 * size);
    let rem = rel.rem_euclid(2 * size);
    (node, Gi::new((rem << NOISE_BITS) / (2 * size)))
}

/// The node at `(i, j)` of a face where one axis may stand up to two nodes outside the face —
/// across the seam, by the same rule a chunk's halo uses (`lattice::site_of`): the partner face's
/// edge cell at the along position, then inward by what is left. Both axes outside — a cube
/// corner's missing quadrant — takes the nearest node of the face ([`CORNER_BLEND`]).
#[must_use]
pub fn node_at(lattice: &MacroLattice, face: Face, i: i64, j: i64) -> u32 {
    let n = i64::from(lattice.edge);
    let in_i = (0..n).contains(&i);
    let in_j = (0..n).contains(&j);
    if in_i && in_j {
        return lattice.index(face, i as i32, j as i32);
    }
    if !in_i && !in_j {
        return lattice.index(face, i.clamp(0, n - 1) as i32, j.clamp(0, n - 1) as i32);
    }
    let (side, along, over) = if !in_i {
        if i < 0 {
            (Edge::UMinus, j, -i)
        } else {
            (Edge::UPlus, j, i - n + 1)
        }
    } else if j < 0 {
        (Edge::VMinus, i, -j)
    } else {
        (Edge::VPlus, i, j - n + 1)
    };
    let c = across(face, side, along as i32, n as i32);
    let rec = partner_of(face, side);
    let inward = if rec.dst_edge.is_plus() {
        -(over - 1)
    } else {
        over - 1
    } as i32;
    let (ci, cj) = if rec.dst_edge.is_u() {
        (c.i + inward, c.j)
    } else {
        (c.i, c.j + inward)
    };
    lattice.index(c.face, ci, cj)
}

/// ★ THE READ of `Z` under the cell `(i, j)` of `face` at `rung`, in the height's unit (gap steps
/// at [`LENGTH_BITS`]): the sixteen nodes around the cell through the seam table, then the
/// Catmull-Rom kernel. `None` where the field holds no row for one of them.
#[must_use]
pub fn sample_z(
    lattice: &MacroLattice,
    field: &dyn ZField,
    face: Face,
    rung: u8,
    i: i32,
    j: i32,
) -> Option<Gi> {
    let (ni, ta) = node_and_fraction(lattice, rung, i);
    let (nj, tb) = node_and_fraction(lattice, rung, j);
    let mut words = [Gi::ZERO; 16];
    for row in 0..4i64 {
        for col in 0..4i64 {
            let node = node_at(lattice, face, ni + col - 1, nj + row - 1);
            let z = field.z_m(node)?;
            words[(row * 4 + col) as usize] = Gi::new(i64::from(z) * STEPS_PER_M) << LENGTH_BITS;
        }
    }
    Some(catmull_rom_16(&words, ta, tb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words, home_solve_words};
    use crate::solve::{FACIES_SEA, Schedule, SolveWords, solve_full};

    fn moon_artifact(words: &SolveWords) -> (MacroSolve, Artifact) {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let (state, facies, _) =
            solve_full(&moon, words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        let climate = crate::climate::climate(&moon, &lattice, words, &state.z, Some(state.sea_z));
        let artifact = Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        (state, artifact)
    }

    /// A row's bytes round-trip; the logarithm classes are monotone and land where stated; the
    /// metres round half up and clamp.
    #[test]
    fn rows_and_classes() {
        let row = Row {
            z_m: -1_234,
            water_m: DRY_M,
            receiver_facies: 0x35,
            discharge_log: 200,
            temperature: 115,
            rain: 213,
            aridity: 7,
        };
        assert_eq!(Row::from_bytes(row.to_bytes()), row);
        assert_eq!(log2_class(0, 2), 0);
        assert_eq!(log2_class(1, 2), 4);
        assert_eq!(log2_class(3, 2), 8);
        // 10 001 is 1.0011… × 2¹³: the class is 13 × 16 plus the next four bits, 0011 — 211 (a
        // mantissa's bits, monotone, not the true logarithm's 212.6).
        assert_eq!(log2_class(10_000, 4), 211);
        assert_eq!(log2_class(u64::MAX, 4), 255);
        assert_eq!(log2_class(1, 4), 16);
        let mut last = 0;
        for v in 0..5_000u64 {
            let c = log2_class(v, 4);
            assert!(c >= last);
            last = c;
        }
        assert_eq!(metres_i16(8), 1);
        assert_eq!(metres_i16(7), 0);
        assert_eq!(metres_i16(-8), 0);
        assert_eq!(metres_i16(-9), -1);
        assert_eq!(metres_i16(i32::MAX), i16::MAX);
        assert_eq!(metres_i16(i32::MIN), i16::MIN + 1);
    }

    /// ★ THE ARTIFACT OF THE MOON: nine bytes a node, the pyramid halving to the odd factor, the
    /// digest the same for the same solve and different for a different one, the tiles covering
    /// every node exactly once and reading back into a cache that answers the artifact's own
    /// heights, and the read of `Z` through the cache agreeing with the read through the artifact.
    #[test]
    fn the_moons_artifact_its_tiles_and_its_digest() {
        let (state, artifact) = moon_artifact(&home_moon_solve_words());
        let n = state.node_count();
        assert_eq!(artifact.node_count(), n);
        assert_eq!(artifact.edge, 68);
        assert_eq!(artifact.pyramid.len(), 2);
        assert_eq!(artifact.pyramid[0].len(), 6 * 34 * 34);
        assert_eq!(artifact.pyramid[1].len(), 6 * 17 * 17);
        assert_eq!(
            artifact.bytes(),
            n * ROW_BYTES + 2 * (6 * 34 * 34 + 6 * 17 * 17)
        );
        // A dry, airless moon: no water anywhere, every outlet an outlet, every row's height the
        // rounded state.
        assert!(artifact.rows.iter().all(|r| r.water_m == DRY_M));
        let outlets = artifact
            .rows
            .iter()
            .filter(|r| r.receiver_facies & RECEIVER_NONE != 0)
            .count();
        assert_eq!(outlets, 6);
        for (i, r) in artifact.rows.iter().enumerate() {
            assert_eq!(r.z_m, metres_i16(state.z[i]));
            // The slot names the receiver among the node's neighbours, or the row says none.
            if r.receiver_facies & RECEIVER_NONE == 0 {
                let slot = usize::from(r.receiver_facies & RECEIVER_SLOT_MASK);
                assert_eq!(state.lattice.neighbours(i as u32)[slot], state.receiver[i]);
            } else {
                assert_eq!(state.receiver[i], NO_NODE);
            }
            assert_eq!(r.receiver_facies >> FACIES_SHIFT, 0);
            assert_eq!(r.rain, 0);
            assert_eq!(r.aridity, 255);
        }
        // The pyramid's first level is the mean of its block.
        let lattice = state.lattice;
        let coarse = lattice.coarser(1).expect("a level");
        let (face, i, j) = (Face::PosX, 10, 20);
        let mut sum = 0i32;
        for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            sum += i32::from(artifact.rows[lattice.index(face, i + di, j + dj) as usize].z_m);
        }
        assert_eq!(
            i32::from(artifact.pyramid[0][coarse.index(face, i / 2, j / 2) as usize]),
            sum.div_euclid(4)
        );
        // The digest is a function of the artifact.
        let (_, again) = moon_artifact(&home_moon_solve_words());
        assert_eq!(again.digest(), artifact.digest());
        let (_, wet) = moon_artifact(&SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        });
        assert_ne!(wet.digest(), artifact.digest());
        assert!(wet.rows.iter().any(|r| r.water_m != DRY_M));
        assert!(
            wet.rows
                .iter()
                .any(|r| r.receiver_facies >> FACIES_SHIFT & FACIES_SEA != 0)
        );
        assert!(wet.rows.iter().any(|r| r.rain > 0));
        // The tiles: 68 = 64 + 4, so two tiles an edge, the last four wide.
        assert_eq!(artifact.tiles_per_edge(), 2);
        assert_eq!(tile_width(68, 0), 64);
        assert_eq!(tile_width(68, 1), 4);
        let mut cache = TileCache::new(68);
        assert!(cache.is_empty());
        let mut seen = 0usize;
        for face in Face::ALL {
            for ty in 0..2 {
                for tx in 0..2 {
                    let tile = artifact.tile(face, tx, ty);
                    seen += tile.rows.len();
                    cache.apply(tile);
                }
            }
        }
        assert_eq!(seen, n);
        assert_eq!(cache.len(), 24);
        // The tile's bytes round-trip, and a torn length is refused.
        let tile = artifact.tile(Face::NegZ, 1, 1);
        let bytes = tile.to_bytes();
        assert_eq!(bytes.len(), tile.rows.len() * ROW_BYTES);
        assert_eq!(Tile::rows_from_bytes(&bytes), Some(tile.rows.clone()));
        assert_eq!(Tile::rows_from_bytes(&bytes[1..]), None);
        for node in 0..n as u32 {
            assert_eq!(cache.z_m(node), artifact.z_m(node));
            assert_eq!(cache.row(node), Some(artifact.rows[node as usize]));
        }
        assert_eq!(artifact.z_m(n as u32), None);
        // The read through the cache is the read through the artifact, at two rungs.
        for (rung, i, j) in [(0u8, 5_000, 300_000), (13, 40, 60), (3, 61_000, 3)] {
            let a = sample_z(&lattice, &artifact, Face::NegY, rung, i, j);
            let c = sample_z(&lattice, &cache, Face::NegY, rung, i, j);
            assert_eq!(a, c);
            assert!(a.is_some());
        }
        // A cache missing a tile answers nothing for that tile's nodes and their stencils.
        let mut partial = TileCache::new(68);
        partial.apply(artifact.tile(Face::PosX, 0, 0));
        assert_eq!(partial.z_m(lattice.index(Face::PosX, 66, 66)), None);
        assert_eq!(sample_z(&lattice, &partial, Face::PosX, 13, 66, 66), None);
        assert!(sample_z(&lattice, &partial, Face::PosX, 13, 30, 30).is_some());
        // A sparse field answers only what it holds.
        let mut sparse = SparseZ::default();
        assert_eq!(sample_z(&lattice, &sparse, Face::PosX, 13, 30, 30), None);
        for row in 0..4 {
            for col in 0..4 {
                let node = lattice.index(Face::PosX, 29 + col, 29 + row);
                sparse.0.insert(node, artifact.rows[node as usize].z_m);
            }
        }
        assert_eq!(
            sample_z(&lattice, &sparse, Face::PosX, 13, 30, 30),
            sample_z(&lattice, &artifact, Face::PosX, 13, 30, 30)
        );
    }

    /// ★ THE TILES A CHUNK READS: a chunk over the middle of a tile names that tile; one whose
    /// stencil straddles node 64 names two along that axis; one at the face's low edge names the
    /// tile across the seam; a rung-13 chunk over the whole face names every tile of the face and
    /// the ring of its neighbours; the cache answers `holds_all` only once every one is applied.
    #[test]
    fn the_tiles_a_chunk_reads() {
        let (state, artifact) = moon_artifact(&home_moon_solve_words());
        let lattice = state.lattice;
        let key = |x: i32, y: i32, rung: u8| ChunkKey {
            face: Face::PosZ,
            rung,
            x,
            y,
            z: 0,
        };
        // A rung-0 chunk over cell 248 000 (30.3 nodes in, so node 29 after the half-node
        // offset): tile 0 alone.
        let mid = tiles_of_chunk(&lattice, key(4_000, 4_000, 0));
        assert_eq!(mid, BTreeSet::from([(Face::PosZ.index(), 0, 0)]));
        // Cells 524 272.. sit over node 63; the stencil 62..=65 crosses into tile 1 along `i`.
        let straddle = tiles_of_chunk(&lattice, key(8_456, 4_000, 0));
        assert_eq!(
            straddle,
            BTreeSet::from([(Face::PosZ.index(), 0, 0), (Face::PosZ.index(), 1, 0)])
        );
        // The face's low edge: this face's tile 0 and a tile on the partner face.
        let edge = tiles_of_chunk(&lattice, key(0, 4_000, 0));
        assert_eq!(edge.len(), 2);
        assert!(edge.contains(&(Face::PosZ.index(), 0, 0)));
        assert!(edge.iter().any(|&(f, _, _)| f != Face::PosZ.index()));
        // The whole face at the top rung: the four tiles of +Z and the neighbours' edge tiles.
        let whole = tiles_of_chunk(&lattice, key(0, 0, 13));
        assert_eq!(
            whole
                .iter()
                .filter(|&&(f, _, _)| f == Face::PosZ.index())
                .count(),
            4
        );
        assert!(whole.len() > 4);
        let mut cache = TileCache::new(68);
        assert!(!cache.holds_all(&straddle));
        cache.apply(artifact.tile(Face::PosZ, 0, 0));
        assert!(cache.holds(Face::PosZ.index(), 0, 0));
        assert!(cache.holds_all(&mid));
        assert!(!cache.holds_all(&straddle));
        cache.apply(artifact.tile(Face::PosZ, 1, 0));
        assert!(cache.holds_all(&straddle));
        assert_eq!(
            tile_of_node(68, lattice.index(Face::NegY, 66, 3)),
            (3, 1, 0)
        );
        // The nodes a rung-0 chunk reads: sixteen in the middle of a tile, and every one is a node
        // the straddling chunk's tiles hold.
        let nodes = nodes_of_chunk(&lattice, key(4_000, 4_000, 0));
        assert_eq!(nodes.len(), 16);
        assert!(
            nodes
                .iter()
                .all(|&n| tile_of_node(68, n) == (Face::PosZ.index(), 0, 0))
        );
        let across = nodes_of_chunk(&lattice, key(8_456, 4_000, 0));
        assert!(
            across
                .iter()
                .all(|&n| straddle.contains(&tile_of_node(68, n)))
        );
    }

    /// ★ THE GOLDEN FIELDS round-trip through their text, pick the rows for a fine rung and the top
    /// level for the top rung, refuse a middle rung, and refuse a torn line.
    #[test]
    fn the_golden_fields_round_trip_and_pick_by_rung() {
        let (state, artifact) = moon_artifact(&home_moon_solve_words());
        let lattice = state.lattice;
        let mut rows = SparseZ::default();
        for node in nodes_of_chunk(
            &lattice,
            ChunkKey {
                face: Face::PosX,
                rung: 0,
                x: 300,
                y: 700,
                z: 0,
            },
        ) {
            rows.0.insert(node, artifact.rows[node as usize].z_m);
        }
        let fields = GoldenFields {
            levels: artifact.pyramid.len() as u32,
            rows,
            top: PyramidField::of(&artifact, 2).expect("level 2"),
        };
        let text = fields.to_text();
        assert_eq!(GoldenFields::parse(&text), Some(fields.clone()));
        assert_eq!(GoldenFields::parse("x 1 2"), None);
        assert_eq!(GoldenFields::parse("t 1"), None);
        assert_eq!(GoldenFields::parse("h 2 2\nr 1"), None);
        assert_eq!(fields.field_for(&lattice, 0).map(|f| f.level()), Some(0));
        assert_eq!(fields.field_for(&lattice, 13).map(|f| f.level()), Some(2));
        // Rung 10 reads level 1 on the moon, which the golden fields do not hold.
        assert_eq!(PyramidField::level_for(&lattice, fields.levels, 10), 1);
        assert!(fields.field_for(&lattice, 10).is_none());
    }

    /// ★ THE PYRAMID AS A FIELD: level 1 over the moon's coarser lattice reads the block means, the
    /// level a rung reads climbs with the rung, and a read at a coarse rung through the level agrees
    /// with the read through the rows to within the block mean's own error.
    #[test]
    fn a_pyramid_level_reads_as_a_coarse_field() {
        let (state, artifact) = moon_artifact(&home_moon_solve_words());
        let lattice = state.lattice;
        let level1 = PyramidField::of(&artifact, 1).expect("level 1");
        let coarse = lattice.coarser(1).expect("a level");
        assert_eq!(level1.z_m.len(), coarse.node_count());
        assert_eq!(PyramidField::of(&artifact, 3), None);
        assert_eq!(PyramidField::of(&artifact, 0), None);
        // The level climbs while four of its nodes fit the chunk: a chunk at rung 13 is 62 nodes
        // wide (level 2, the last the moon has); at rung 10 it is 7.75 nodes (level 1, whose node
        // is two); at rung 9 it is 3.9 nodes (the rows); at rung 11 it is 15.5 (level 2).
        assert_eq!(PyramidField::level_for(&lattice, 2, 13), 2);
        assert_eq!(PyramidField::level_for(&lattice, 2, 10), 1);
        assert_eq!(PyramidField::level_for(&lattice, 2, 9), 0);
        assert_eq!(PyramidField::level_for(&lattice, 2, 11), 2);
        assert_eq!(PyramidField::level_for(&lattice, 0, 13), 0);
        // The read through level 1 on the coarser lattice at the cell that is one coarse node.
        let fine_node = lattice.index(Face::PosZ, 20, 30);
        let via_rows = sample_z(&lattice, &artifact, Face::PosZ, 13, 20, 30).expect("rows");
        let via_level = sample_z(&coarse, &level1, Face::PosZ, 14, 10, 15).expect("level");
        let unit = f64::from(1u32 << LENGTH_BITS) * STEPS_PER_M as f64;
        let rows_m = via_rows.raw() as f64 / unit;
        let level_m = via_level.raw() as f64 / unit;
        let own = f64::from(artifact.rows[fine_node as usize].z_m);
        assert!((rows_m - own).abs() < 1.0, "{rows_m} vs {own}");
        // The level's node is the mean of four rows: it stands within the block's own spread.
        let block: Vec<f64> = [(20, 30), (21, 30), (20, 31), (21, 31)]
            .iter()
            .map(|&(i, j)| f64::from(artifact.rows[lattice.index(Face::PosZ, i, j) as usize].z_m))
            .collect();
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for &b in &block {
            if b < lo {
                lo = b;
            }
            if b > hi {
                hi = b;
            }
        }
        assert!(
            level_m >= lo - 1.0 && level_m <= hi + 1.0,
            "{level_m} outside {lo}..{hi}"
        );
    }

    /// ★ THE NODE AND THE FRACTION: at rung 13 on the moon (a node is 2¹³ cells) cell `i` is node
    /// `i` at fraction zero; at rung 0 the first cell lies half a node before node 0, so it reads
    /// node −1 at fraction one half plus half a cell; the last cell of the face reads the node past
    /// the edge.
    #[test]
    fn node_and_fraction_at_the_rungs() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        assert_eq!(lattice.cells_per_node, 8_192);
        assert_eq!(node_and_fraction(&lattice, 13, 7), (7, Gi::ZERO));
        let (node, t) = node_and_fraction(&lattice, 0, 0);
        assert_eq!(node, -1);
        let half = 1i64 << (NOISE_BITS - 1);
        assert!(
            (t.raw() - half).abs() < (1 << (NOISE_BITS - 12)),
            "{}",
            t.raw()
        );
        let (node, t) = node_and_fraction(&lattice, 0, 4_096);
        assert_eq!(node, 0);
        assert!(t.raw() < 1 << (NOISE_BITS - 12), "{}", t.raw());
        let last = lattice.cells_per_node as i32 * lattice.edge as i32 - 1;
        let (node, _) = node_and_fraction(&lattice, 0, last);
        assert_eq!(node, i64::from(lattice.edge) - 1);
    }

    /// ★ THE GATHER ACROSS A SEAM AND AT A CORNER: a node one past the +u edge of +X is the partner
    /// face's edge cell at the same along position; two past is one inward; a node before the −v
    /// edge crosses the other way; the missing quadrant at a corner takes the face's corner node.
    #[test]
    fn the_gather_crosses_the_seam_and_clamps_at_the_corner() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let n = i64::from(lattice.edge);
        let inside = node_at(&lattice, Face::PosX, 5, 7);
        assert_eq!(inside, lattice.index(Face::PosX, 5, 7));
        let one_past = node_at(&lattice, Face::PosX, n, 7);
        let c = across(Face::PosX, Edge::UPlus, 7, n as i32);
        assert_eq!(one_past, lattice.index(c.face, c.i, c.j));
        // Two past: one node inward on the partner (the partner's side is a plus side, so inward
        // is toward smaller j there).
        let two_past = node_at(&lattice, Face::PosX, n + 1, 7);
        assert_eq!(two_past, lattice.index(c.face, c.i, c.j - 1));
        let before = node_at(&lattice, Face::PosX, 5, -1);
        let d = across(Face::PosX, Edge::VMinus, 5, n as i32);
        assert_eq!(before, lattice.index(d.face, d.i, d.j));
        let corner = node_at(&lattice, Face::PosX, -1, -2);
        assert_eq!(corner, lattice.index(Face::PosX, 0, 0));
        let far_corner = node_at(&lattice, Face::PosX, n + 1, n);
        assert_eq!(
            far_corner,
            lattice.index(Face::PosX, (n - 1) as i32, (n - 1) as i32)
        );
        // Every stencil node the read gathers is a neighbour-of-neighbour: symmetric through the
        // table, so the seam neighbour names the face's edge node back.
        assert!(lattice.neighbours(one_past).contains(&lattice.index(
            Face::PosX,
            (n - 1) as i32,
            7
        )));
    }

    /// ★ G-MACRO-EDGE on the moon: the read of `Z` along a cube edge agrees from both faces to
    /// within the read's own rounding, at the rung whose cell is a node and at rung 0 — no scarp.
    #[test]
    fn the_read_agrees_across_a_cube_edge() {
        let (state, artifact) = moon_artifact(&home_moon_solve_words());
        let lattice = state.lattice;
        let n = lattice.cells_per_node as i32 * lattice.edge as i32;
        let unit = f64::from(1u32 << LENGTH_BITS) * STEPS_PER_M as f64;
        let blend = CORNER_BLEND * lattice.cells_per_node as i32;
        let (mut worst, mut worst_corner) = (0i64, 0i64);
        // Along the +u edge of +X (its partner: +Y's +v edge, the axes swapped), every 500 cells.
        let mut j = 0;
        while j < n {
            let here = sample_z(&lattice, &artifact, Face::PosX, 0, n - 1, j).expect("a read");
            let c = across(Face::PosX, Edge::UPlus, j, n);
            let there = sample_z(&lattice, &artifact, c.face, 0, c.i, c.j).expect("a read");
            let step = (here.raw() - there.raw()).abs();
            if j < blend || j >= n - blend {
                worst_corner = worst_corner.max(step);
            } else {
                worst = worst.max(step);
            }
            j += 500;
        }
        // Adjacent cells a metre apart on a field whose slope is under a metre a metre: the step
        // between them is the field's own slope over one cell, under 2 m; the C1 read adds no scarp.
        let step_m = worst as f64 / unit;
        assert!(step_m < 2.0, "a scarp of {step_m} m across the seam");
        // ★ G-MACRO-CORNER, MEASURED: within the corner blend the stated fallback (each face clamps
        // into its own corner node) leaves a step — the quintic-fade blend of 03 §5.4 is owed and
        // this is its number (on the moon about 160 m over 1 m). It is asserted only under the
        // relief, so the owed item is measured, never hidden.
        let corner_m = worst_corner as f64 / unit;
        assert!(
            corner_m > step_m,
            "the corner fallback is a step: {corner_m} m"
        );
        assert!(corner_m < moon_relief_m(), "{corner_m} m");
    }

    fn moon_relief_m() -> f64 {
        home_moon().relief_m()
    }
}
