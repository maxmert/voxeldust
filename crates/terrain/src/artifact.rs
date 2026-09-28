//! ★ THE ARTIFACT (the landform arc, slice 8c stage C4; `slice_8c_design.md` §7; the owner's
//! ruling of 2026-09-19: the solve runs ONCE on the server, the artifact is saved in the shard's own
//! store and SHIPPED to every client in tiles).
//!
//! What the solve keeps, per node of the macro lattice, TEN bytes: the eroded height `Z` in whole
//! metres, the water level (a lake's spill level, the sea's level, or DRY), the D8 receiver's
//! stencil slot with the facies bits, the discharge as a quantised logarithm, three climate
//! bytes (the temperature class, the rain class, the aridity) for 8e and 18, and THE ROCK PROVINCE
//! (slice 8d step 2), which says which rocks the node's beds are drawn from. Beside the rows a
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
//! client asks for the tiles under the eye — 40 KB each — and the chunks it builds read the valley
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
/// meaning, the pyramid's rule or the stored rows' cut bumps it. Version 3: the pyramid is stored
/// in part rows (version 2's one row broke the store's field cap on every big planet, measured in
/// the far-view ship's first flight, 2026-09-20); a version-2 store re-solves at boot. Version 4
/// (2026-09-21) adds the pyramid's water word beside every level; a version-3 store re-solves.
/// ★ Version 5 (2026-09-21, slice 8d step 2; ruling W4 item 4, the owner's SL6 approval) adds THE
/// ROCK PROVINCE to every node row — nine bytes to ten; a version-4 store re-solves. The PYRAMID
/// takes no province word: the far view draws no rock, so a coarse node has nothing to say about
/// which bed a miner cuts.
/// ★ Version 6 (2026-09-22, the coast mask; ruling W10) adds THE COAST MASK — one bit per FINE
/// node, set where the row stands at or under the sea — beside the rows, so a column at ANY rung
/// reads the water's side from the fine row's own word and never from a pyramid level's mean; a
/// version-5 store re-solves.
/// ★ Version 7 (2026-09-23, the lake's side; ruling W16; the owner: *"at some point water is not
/// visible at all"*) widens the coast mask from ONE bit to a TWO-BIT SIDE WORD per fine node —
/// land, sea or LAKE. The mask's bytes double (four nodes a byte), and a version-6 store re-solves.
/// MEASURED before it (`water_edge_step lake`): NOT ONE column of 300 000 stood under a lake's own
/// surface at any rung, because a lake node's bit was clear, the shore law read `SIDE_LAND` and
/// held every lake column at least a quarter of its ground's height ABOVE its own water. The shore
/// law drained every lake on the planet.
/// ★ Version 8 (2026-09-24, plan step 2; ruling B7) keeps every byte's FORMAT and changes four
/// meanings, so a version-7 store re-solves: (1) the DISCHARGE byte is the Turc–Pike RUNOFF
/// accumulated over the final receivers, no longer the last pass's rain over an older tree (status
/// §7 W10); (2) the PYRAMID'S WATER WORD holds the HIGHEST LAKE LEVEL of a block wherever the block
/// holds a lake node, else the sea where it holds a sea node, else DRY — never a two-of-four
/// majority, never a mean ([`block_water`]; status §6 T5, §7 W6); (3) the SEA is the CONNECTED
/// OCEAN only, so an inland pocket's rows carry its own budget's water and never the sea's bit;
/// (4) the coast mask's word 2, unused before, is A POCKET'S DRY FLOOR ([`side_word`],
/// [`Side::Pocket`]; 2026-09-25) — the mask keeps its two bits a node and its size.
/// ★ Version 9 (2026-09-25, plan step 2b; ruling B8 item 2, amending B7 item 2) keeps every byte's
/// FORMAT and changes two meanings, so a version-8 store re-solves: (1) the RECEIVER SLOT is THE
/// WATER'S OWN WAY ([`crate::lakes::water_tree`]) — the ground's descent, across a spilling lake to
/// its spill, over the spill into the side the budget pours into — and "no receiver" is where the
/// water ENDS: the ocean, a closed lake's water, a dry hollow's pit; (2) the DISCHARGE byte is THE
/// BUDGET'S OWN WATER — the runoff over those receivers, less at each spill what its spilling lake
/// hands the air — so a closed body passes nothing on and the row into the sea is the cascade's.
/// Before it the slot was the routing fill's D8 receiver and the row carried 1 607 km³/yr to the sea
/// out of hollows the budget held (MEASURED at generator 20).
pub const ARTIFACT_VERSION: u32 = 9;
/// A tile's edge in nodes: 64 × 64 rows of ten bytes is 40 KB — under a second on the lane.
pub const TILE_EDGE: u32 = 64;
/// The water level's word for a dry node.
pub const DRY_M: i16 = i16::MIN;
/// The bytes a row packs to.
pub const ROW_BYTES: usize = 10;
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
    /// ★ THE ROCK PROVINCE's code (slice 8d step 2; `crate::strata::Province`). A client cannot
    /// compute it: the solve's crust and belt fields are its own working state and are thrown away,
    /// so recomputing it means re-running the solve. It therefore rides the tile that already
    /// crosses, on the row the owner approved (ruling W4 item 4) — no new lane, no new payload kind.
    pub province: u8,
}

impl Row {
    /// The ten bytes, little-endian words first.
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
            self.province,
        ]
    }

    /// The row from its ten bytes.
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
            province: b[9],
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
    /// ★ THE SEA'S LEVEL (slice 8c stage C5), whole metres over the ladder radius, RE-SOLVED over
    /// the eroded field so the water inventory fits under it; [`DRY_M`] on a body with no water.
    /// Folded into the digest; a host that holds the artifact gives its body this sea.
    pub sea_m: i16,
    pub rows: Vec<Row>,
    /// Level `k` (from 1) holds `6 · (edge >> k)²` heights in whole metres, the integer mean of its
    /// 2 × 2 block below; as many levels as the edge halves to.
    pub pyramid: Vec<Vec<i16>>,
    /// ★ THE PYRAMID'S WATER WORD (2026-09-21, the owner's order after the coast flight): level
    /// `k` holds one water level per coarse node, the same shape as `pyramid`. ★ SINCE VERSION 8
    /// (2026-09-24, plan step 2; ruling W16: a rung boundary is invisible) it is [`block_water`]:
    /// the HIGHEST lake level among every lake fine node under the block, else the sea's level where
    /// a sea node stands under it, else [`DRY_M`]. The word before it was the MEAN of the wet
    /// children where at least two of four were wet, so a lake node alone in its block turned to
    /// land at the rung 9 → 10 swap — MEASURED at generator 19: 345 of 506 lakes vanished whole
    /// there — and a block half sea and half lake read a level no water has. ★ Since ruling B22
    /// (2026-09-28) no column reads the word: it is stored and folded into the digest for the
    /// one-volume fill step.
    pub pyramid_water: Vec<Vec<i16>>,
    /// ★ THE COAST MASK (2026-09-22; the owner, from 1 400 km: "during flight the shores changes
    /// again all the time"; ruling W10; widened 2026-09-23 by ruling W16): A TWO-BIT SIDE WORD PER
    /// FINE NODE in node order — bits `2·(node % 4)` and `2·(node % 4) + 1` of byte `node / 4`.
    /// The low bit says the node stands AT OR UNDER ITS OWN WATER (the solve's
    /// [`crate::solve::FACIES_SEA`] or [`crate::solve::FACIES_LAKE`]); the high bit says that water
    /// is a LAKE and not the sea. Until ruling B22 every rung read the SIDE of the water from this
    /// mask, so a coarse rung's ground could be a mean and the shoreline still stood where the fine
    /// row put it. MEASURED before the mask (`vd-bins/examples/shore_step`): the crossing moved a
    /// median of 11.5 km at the swap from the rows to level 1 and about 20 km at each level swap
    /// above. MEASURED before the LAKE bit (`water_edge_step lake`): not one column of 300 000 stood
    /// under a lake's own surface at any rung. The home planet's mask is 8.9 million side words —
    /// 2.2 MB, one part in forty of the rows.
    ///
    /// ★ SINCE RULING B22 (2026-09-28) NO COLUMN READS THE MASK and the shard ships none: the shore
    /// law is removed. The mask is still built, stored and folded into the digest, so the one-volume
    /// fill step finds it in every store.
    pub coast: Vec<u8>,
}

/// The bytes a coast mask of `nodes` fine nodes takes: a two-bit side word each, rounded up.
#[must_use]
pub fn coast_bytes(nodes: usize) -> usize {
    nodes.div_ceil(4)
}

/// ★ WHICH SIDE OF WHICH WATER A NODE STANDS ON (2026-09-23, ruling W16). Three words for water:
/// a node over its water is LAND; one under the SEA takes the body's own sea level; one under a
/// LAKE takes that lake's own level, which stands anywhere between the sea and a mountain pass.
///
/// ★ THE FOURTH WORD, THE POCKET (2026-09-25, ruling B7 item 1: *"an inland pocket is a closed
/// basin, not the sea"*). A DRY node whose ground stands UNDER the sea's level is the floor of a
/// closed basin that no path joins to the ocean. It is its own word, so a reader names it from the
/// node itself — never from a stencil whose width follows the pyramid level.
///
/// ★ SINCE RULING B22 (2026-09-28) NO COLUMN READS THE WORD: the shore law that read it is
/// removed. The word stays the mask's stored side, for the one-volume fill step.
///
/// **Example.** A pilot flies north from the belt. The shore under her is [`Side::Sea`]; the tarn
/// in the pass she crosses is [`Side::Lake`] at 2 100 m, and its water stands two kilometres over
/// the sea she just left. Behind a one-node ridge on the coast, a salt pan 2 km under the sea's
/// level is [`Side::Pocket`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The node stands OVER its own water.
    Land,
    /// The node stands at or under THE SEA.
    Sea,
    /// The node stands at or under A LAKE, at that lake's own level.
    Lake,
    /// ★ The node stands DRY and UNDER THE SEA'S LEVEL: the floor of an inland pocket, where no
    /// water stands (2026-09-25, ruling B7 item 1).
    Pocket,
}

impl Side {
    /// Whether the node stands at or under water at all — the sea and a lake alike. A pocket's
    /// floor is dry.
    #[must_use]
    pub fn wet(self) -> bool {
        matches!(self, Side::Sea | Side::Lake)
    }
}

/// ★ ONE NODE'S SIDE WORD out of a coast mask: `None` where the mask is too short to say (a mask
/// that never arrived, a node past the lattice). A reader NEVER guesses a side.
#[must_use]
pub fn coast_side(mask: &[u8], node: u32) -> Option<Side> {
    let byte = *mask.get(node as usize / 4)?;
    let shift = 2 * (node % 4);
    Some(side_of_word(byte >> shift & 3))
}

/// The side a two-bit mask word names: 0 land, 1 the sea, 2 a pocket's dry floor, 3 a lake.
fn side_of_word(word: u8) -> Side {
    match word {
        0 => Side::Land,
        1 => Side::Sea,
        2 => Side::Pocket,
        _ => Side::Lake,
    }
}

/// ★ THE ONE RULE A NODE'S SIDE WORD IS WRITTEN WITH (2026-09-25, ruling B7 item 1): the facies'
/// own word ([`coast_word`]: land 0, the sea 1, a lake 3), and 2 — A POCKET'S FLOOR — for a DRY
/// node whose row stands UNDER the sea's level. The mask is written with it and every row reader
/// reads with it ([`side_of_row`]), so a host that holds the mask and a host that holds the tile
/// name one side for one node.
///
/// The compare is on the ROW'S OWN WHOLE METRES against the artifact's own sea
/// ([`Artifact::sea_m`]), strict: a dry node whose metre is under the sea's metre stands under the
/// level, and since the sea's facies is the CONNECTED ocean, no path joins it to the ocean — it is
/// a pocket (`land::ocean_mask`). A dry node within the metre of the sea's level reads LAND, which
/// is the word it had before. A dry body's sea is [`DRY_M`], under every row, so a dry body has no
/// pocket.
///
/// **Example.** The salt pan behind the belt's coast stands at −1 400 m against a sea at 3 591 m,
/// with no water in it: word 2. The ridge beside it at 3 640 m: word 0.
#[must_use]
pub fn side_word(facies: u8, z_m: i16, sea_m: i16) -> u8 {
    let word = coast_word(facies);
    if (word == 0) & (z_m < sea_m) { 2 } else { word }
}

/// The two bits the FACIES alone give a node's side: the wet bit, then the lake bit. The mask's
/// own word is [`side_word`], which adds the pocket.
#[must_use]
pub fn coast_word(facies: u8) -> u8 {
    if facies & crate::solve::FACIES_LAKE != 0 {
        3
    } else if facies & crate::solve::FACIES_SEA != 0 {
        1
    } else {
        0
    }
}

/// ★ THE COAST COUNTS — THE WET FRACTION UNDER A CELL, AT EVERY RUNG (2026-09-22; the owner, from
/// 41 000 km: *"the globe shows squares of water on the land"*; ruling W15). The mask states one
/// bit per FINE node. A cell at a far rung covers MANY fine nodes — 262 km against 8 192 m at rung
/// 18, about a thousand of them — and reading the ONE node nearest the cell's centre lets one node
/// in a thousand paint the whole cell: a checkerboard of sea over the land, and a different centre
/// node at each rung, so the same ground flipped side at every ring swap.
///
/// ★ SINCE RULING B22 (2026-09-28) NO COLUMN READS THE COUNTS: the side read that asked for them
/// is removed with the shore law. They stay the mask's own fold — the client still folds them when
/// a mask arrives — for the one-volume fill step.
///
/// Level `k` (from 1) holds, per coarse node of `2^k × 2^k` fine nodes on each face, HOW MANY of
/// those fine nodes are wet. A parent's count is the SUM of its four children's, so a coarse cell
/// shows the side most of its ground stands on and a finer rung refines that edge without ever
/// contradicting it.
///
/// ★ THE WORD IS `u32`, AND THAT IS THE BOUND: a level-`k` count is at most `4^k`, and a lattice
/// halves at most eleven times ([`crate::macro_lattice::MACRO_EDGE_CEILING`] is 2 048 = 2¹¹), so
/// the largest count a body can hold is 4¹¹ = 4 194 304 — past a `u16`. On the home planet: six
/// levels, 2 957 613 coarse nodes (MEASURED); three words each since the pocket's count joined
/// (2026-09-25), 35.5 MB (ARITH: 2 957 613 × 12 bytes).
///
/// ★ THE POCKET IS COUNTED TOO (2026-09-25, ruling B7 item 1): how many of the block's fine nodes
/// are a pocket's dry floor ([`Side::Pocket`]), so a far cell names a pocket by the SAME footprint
/// rule a near cell names its own node — the side is a fact of the ground under the cell at every
/// rung, never of a stencil whose width follows the level.
///
/// The counts are DERIVED, never shipped: the server folds them from its artifact and the client
/// folds them from the mask when the mask lands, so no byte crosses the wire and the artifact's
/// digest does not move.
///
/// **Example.** A hull at 41 000 km draws the belt at rung 18. One of its cells covers 1 024 fine
/// nodes, 300 of them sea: the count says 300, and `300 · 2 < 1 024`, so the cell is LAND — as the
/// ground under it mostly is. The rung below splits that cell in four; the quarter that holds the
/// bay counts 200 of its 256 nodes wet and stands SEA, so the bay opens as the ring sweeps in and
/// the land around it never flips.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoastCounts {
    /// The FINE lattice's edge, the one the mask is indexed on.
    edge: u32,
    /// `levels[k − 1]` holds `6 · (edge >> k)²` count WORDS in the coarse lattice's node order:
    /// how many of the block's fine nodes are WET, how many of those wet ones are a LAKE, and how
    /// many of the DRY ones are a POCKET's floor.
    levels: Vec<Vec<[u32; 3]>>,
}

impl CoastCounts {
    /// ★ THE FOLD, once per body, from the mask alone: level 1 counts the mask's own words four at
    /// a time, and each level above sums its four children. The walk is `O(nodes)` over the whole
    /// pyramid (11.8 million adds on the home planet), integer, and the same on every host.
    #[must_use]
    pub fn of(mask: &[u8], edge: u32) -> CoastCounts {
        let mut levels: Vec<Vec<[u32; 3]>> = Vec::new();
        let mut below_edge = edge;
        let mut k = 1u32;
        while let Some(coarse) = halved(edge, k) {
            let mut level = vec![[0u32; 3]; 6 * (coarse as usize) * (coarse as usize)];
            for face in 0..6u32 {
                for j in 0..below_edge {
                    for i in 0..below_edge {
                        let up = ((face * coarse + (j >> 1)) * coarse + (i >> 1)) as usize;
                        let node = ((face * below_edge + j) * below_edge + i) as usize;
                        let child = match levels.last() {
                            Some(below) => below[node],
                            None => match coast_side(mask, node as u32) {
                                Some(Side::Sea) => [1, 0, 0],
                                Some(Side::Lake) => [1, 1, 0],
                                Some(Side::Pocket) => [0, 0, 1],
                                _ => [0, 0, 0],
                            },
                        };
                        level[up][0] += child[0];
                        level[up][1] += child[1];
                        level[up][2] += child[2];
                    }
                }
            }
            levels.push(level);
            below_edge = coarse;
            k += 1;
        }
        CoastCounts { edge, levels }
    }

    /// How many levels the counts hold: how many times the lattice's edge halves.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.levels.len() as u32
    }

    /// The three counts under the level-`k` coarse node `(face, ci, cj)`; `None` past the levels or
    /// past the level's own lattice.
    fn words(&self, k: u32, face: Face, ci: u32, cj: u32) -> Option<[u32; 3]> {
        let level = self.levels.get(k.checked_sub(1)? as usize)?;
        let edge = halved(self.edge, k)?;
        if ci >= edge || cj >= edge {
            return None;
        }
        level
            .get((((u32::from(face.index()) * edge) + cj) * edge + ci) as usize)
            .copied()
    }

    /// The wet fine nodes under the level-`k` coarse node `(face, ci, cj)`, and how many of those
    /// are a LAKE; `None` past the levels or past the level's own lattice.
    #[must_use]
    pub fn count(&self, k: u32, face: Face, ci: u32, cj: u32) -> Option<(u32, u32)> {
        self.words(k, face, ci, cj).map(|w| (w[0], w[1]))
    }

    /// ★ The fine nodes under the level-`k` coarse node `(face, ci, cj)` that are a POCKET's dry
    /// floor (2026-09-25, ruling B7 item 1); `None` where [`CoastCounts::count`] is `None`.
    #[must_use]
    pub fn pockets(&self, k: u32, face: Face, ci: u32, cj: u32) -> Option<u32> {
        self.words(k, face, ci, cj).map(|w| w[2])
    }

    /// ★ THE SIDE OF ONE COARSE NODE: WET WHERE AT LEAST HALF ITS FINE NODES ARE WET, and then a
    /// LAKE where at least half of the WET ones are a lake. The compare is on whole counts —
    /// `wet · 2 ≥ 4^k`, then `lake · 2 ≥ wet` — so no host divides and no host rounds.
    ///
    /// ★ A TIE IS WET, stated: half a cell of sea reads as sea, which keeps a strait open from
    /// orbit instead of closing it into a land bridge that the next rung cuts again. ★ AND A TIE
    /// BETWEEN THE TWO WATERS IS A LAKE, stated (2026-09-23, ruling W16): a cell half lake and half
    /// sea stands at a river's mouth, and a lake's level is the higher of the two, so reading it as
    /// a lake leaves the ground UNDER water at the cell's own edge instead of over it.
    ///
    /// ★ A DRY CELL IS A POCKET where at least half of its DRY nodes are a pocket's floor
    /// (2026-09-25, ruling B7 item 1): `pocket · 2 ≥ 4^k − wet`. A TIE IS A POCKET, stated: a cell
    /// half basin floor and half rim is left where the solve put it, because a pocket has no water
    /// line to hold and a lift there raises the half that is a basin into a mesa.
    #[must_use]
    pub fn side(&self, k: u32, face: Face, ci: u32, cj: u32) -> Option<Side> {
        let [wet, lake, pocket] = self.words(k, face, ci, cj)?;
        let all = 1u64 << (2 * k);
        if u64::from(wet) * 2 < all {
            return Some(if u64::from(pocket) * 2 >= all - u64::from(wet) {
                Side::Pocket
            } else {
                Side::Land
            });
        }
        Some(if lake * 2 >= wet {
            Side::Lake
        } else {
            Side::Sea
        })
    }
}

/// The edge `k` levels up, or `None` where the edge does not halve that many times (the rule
/// [`crate::macro_lattice::MacroLattice::coarser`] states, on the number alone).
fn halved(edge: u32, k: u32) -> Option<u32> {
    let coarse = edge >> k;
    (coarse > 0 && (coarse << k) == edge).then_some(coarse)
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
    /// ★ THE WATER AND THE FACIES of a node (slice 8c stage C5): the row's water word (the sea's
    /// level, a lake's spill level, or [`DRY_M`]) and its facies bits where the field holds rows;
    /// `None` where it holds no such words (a level built with no water words). Every field states
    /// its own (2026-09-21: the pyramid level carries a word too, so no default is left). ★ Since
    /// ruling B22 (2026-09-28) no column reads it: it is the stored water's reader, for the
    /// one-volume fill step.
    fn water_facies(&self, node: u32) -> Option<(i16, u8)>;
    /// ★ THE ROCK PROVINCE of a node (slice 8d step 2): the row's province byte where the field
    /// holds rows; `None` where it holds none — a PYRAMID LEVEL answers `None`, because the far
    /// view draws no rock and the pyramid carries no province word. A column that reads `None`
    /// takes `crate::strata::DEFAULT_PROVINCE`.
    fn province(&self, node: u32) -> Option<u8>;
    /// ★ WHICH SIDE OF WHICH WATER A FINE NODE STANDS ON (2026-09-22, the coast mask; ruling W10;
    /// widened to the lake 2026-09-23, ruling W16): [`Side`], or `None` where this field cannot
    /// say. `node` is always a FINE (level-0) lattice node, whatever level the field itself stands
    /// at: a pyramid level answers from the coast mask it was given, never from its own means.
    /// ★ Since ruling B22 (2026-09-28) no column reads it: it is the stored mask's reader, for the
    /// one-volume fill step.
    ///
    /// **Example.** The headland on the belt's coast is LAND by its own fine word, whether the host
    /// holds the headland's tile or only the far view's level and the mask.
    fn water_side(&self, node: u32) -> Option<Side>;
}

impl ZField for Artifact {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.rows.get(node as usize).map(|r| r.z_m)
    }
    fn water_facies(&self, node: u32) -> Option<(i16, u8)> {
        self.rows
            .get(node as usize)
            .map(|r| (r.water_m, r.receiver_facies >> FACIES_SHIFT))
    }
    fn province(&self, node: u32) -> Option<u8> {
        self.rows.get(node as usize).map(|r| r.province)
    }
    /// The coast mask's own side word.
    fn water_side(&self, node: u32) -> Option<Side> {
        coast_side(&self.coast, node)
    }
}

/// A few nodes' rows, stated: the golden pin's `(z_m, water_m, facies, province)` for the fine
/// self-check chunks, and ★ THE BODY'S SEA (`Artifact::sea_m`, [`DRY_M`] for none; 2026-09-25,
/// ruling B7 item 1), which a row's side word reads to name a pocket's floor ([`side_of_row`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseRows(pub BTreeMap<u32, (i16, i16, u8, u8)>, pub i16);

impl Default for SparseRows {
    /// No rows, and no sea: a dry body's rows name no pocket.
    fn default() -> SparseRows {
        SparseRows(BTreeMap::new(), DRY_M)
    }
}

impl ZField for SparseRows {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.0.get(&node).map(|r| r.0)
    }
    fn water_facies(&self, node: u32) -> Option<(i16, u8)> {
        self.0.get(&node).map(|r| (r.1, r.2))
    }
    fn province(&self, node: u32) -> Option<u8> {
        self.0.get(&node).map(|r| r.3)
    }
    /// The row's own side word, for a row this field holds; nothing for one it does not.
    fn water_side(&self, node: u32) -> Option<Side> {
        self.0.get(&node).map(|r| side_of_row(r.2, r.0, self.1))
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
    /// ★ THE BODY'S SEA (`Artifact::sea_m`, the head's `sea_m`; [`DRY_M`] for none; 2026-09-25,
    /// ruling B7 item 1): a held row's side word reads it to name a pocket's floor
    /// ([`side_of_row`]), exactly as the mask was written.
    pub sea_m: i16,
    tiles: BTreeMap<(u8, u32, u32), std::sync::Arc<[Row]>>,
}

impl TileCache {
    /// An empty cache for a lattice of `edge` nodes a side, on a body with no sea (see
    /// [`TileCache::with_sea`]).
    #[must_use]
    pub fn new(edge: u32) -> TileCache {
        TileCache {
            edge,
            sea_m: DRY_M,
            tiles: BTreeMap::new(),
        }
    }

    /// The same cache on a body whose sea stands at `sea_m` whole metres ([`DRY_M`] for none): the
    /// head states it, and a row's side word reads it.
    #[must_use]
    pub fn with_sea(mut self, sea_m: i16) -> TileCache {
        self.sea_m = sea_m;
        self
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

    /// Every tile the cache holds, by its `(face, tx, ty)` (2026-09-22: the dev state lists them, so
    /// a chunk's missing tile can be held against what the book really has).
    pub fn tile_ids(&self) -> impl Iterator<Item = (u8, u32, u32)> + '_ {
        self.tiles.keys().copied()
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
    fn water_facies(&self, node: u32) -> Option<(i16, u8)> {
        self.row(node)
            .map(|r| (r.water_m, r.receiver_facies >> FACIES_SHIFT))
    }
    fn province(&self, node: u32) -> Option<u8> {
        self.row(node).map(|r| r.province)
    }
    /// The held row's own side word — the very word the mask carries, so a host reading tiles
    /// and a host reading the mask name one side for one node.
    fn water_side(&self, node: u32) -> Option<Side> {
        self.row(node)
            .map(|r| side_of_row(r.receiver_facies >> FACIES_SHIFT, r.z_m, self.sea_m))
    }
}

/// ★ A ROW'S OWN SIDE, from its facies bits, its own height and the body's sea ([`side_word`]):
/// the ONE rule the mask is written with, so a host reading tiles and a host reading the mask can
/// never name two sides for one node.
#[must_use]
pub fn side_of_row(facies: u8, z_m: i16, sea_m: i16) -> Side {
    side_of_word(side_word(facies, z_m, sea_m))
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
    /// The level's water words (`Artifact::pyramid_water`), the same order; EMPTY for a level
    /// that carries none, whose columns read the body's sea alone.
    pub water_m: Vec<i16>,
    /// ★ THE COAST MASK (2026-09-22, ruling W10): the artifact's OWN fine-node mask
    /// ([`Artifact::coast`]), shared by pointer with every other level, or `None` for a level
    /// assembled with no mask — every level a client assembles since ruling B22, because the shard
    /// ships no mask. It is indexed by FINE node, never by this level's own node.
    pub coast: Option<std::sync::Arc<[u8]>>,
    /// ★ THE COAST COUNTS (2026-09-22, ruling W15): the body's own wet-node counts, folded from
    /// the mask once and shared by pointer with every other level, or `None` for a level with no
    /// mask. No column reads them since ruling B22 (2026-09-28); they stay for the one-volume fill
    /// step.
    pub counts: Option<std::sync::Arc<CoastCounts>>,
}

impl ZField for PyramidField {
    fn z_m(&self, node: u32) -> Option<i16> {
        self.z_m.get(node as usize).copied()
    }

    fn level(&self) -> u32 {
        self.level
    }

    /// The level's water word, with no facies (a coarse node marks no coast).
    fn water_facies(&self, node: u32) -> Option<(i16, u8)> {
        self.water_m.get(node as usize).map(|w| (*w, 0))
    }

    /// ★ NO PROVINCE ON A PYRAMID LEVEL (slice 8d step 2). The far view draws no rock: a chunk
    /// whose cell is many macro nodes wide never shows a bed, so folding a province word up the
    /// pyramid would carry bytes nobody reads. A column at such a rung takes the stated default.
    fn province(&self, _node: u32) -> Option<u8> {
        None
    }

    /// The coast mask's side word where this level holds the mask, nothing where it does not.
    fn water_side(&self, node: u32) -> Option<Side> {
        coast_side(self.coast.as_deref()?, node)
    }
}

impl PyramidField {
    /// Level `k` of an artifact's pyramid, or `None` past its levels. `counts` is the body's own
    /// coast counts ([`Artifact::coast_counts`]), folded ONCE per body and shared by pointer with
    /// every other level: a host that built six levels holds one fold, not six.
    #[must_use]
    pub fn of(
        artifact: &Artifact,
        level: u32,
        counts: &std::sync::Arc<CoastCounts>,
    ) -> Option<PyramidField> {
        artifact
            .pyramid
            .get(level.checked_sub(1)? as usize)
            .map(|z| PyramidField {
                level,
                z_m: z.clone(),
                water_m: artifact
                    .pyramid_water
                    .get(level as usize - 1)
                    .cloned()
                    .unwrap_or_default(),
                // The artifact's own fine-node mask, so this level reads the side the rows state.
                coast: Some(artifact.coast.clone().into()),
                counts: Some(std::sync::Arc::clone(counts)),
            })
    }

    /// ★ THE LEVEL A RUNG READS: the coarsest level whose node is no wider than the rung's chunk
    /// (a chunk is [`crate::chunk::CHUNK_EDGE`] cells), so a chunk always spans several of the
    /// level's nodes; zero (the rows themselves) where the rung's chunk is under two macro nodes.
    #[must_use]
    pub fn level_for(lattice: &MacroLattice, levels: u32, rung: u8) -> u32 {
        let chunk_cells = (crate::chunk::CHUNK_EDGE as u64) << rung;
        let node_cells = u64::from(lattice.cells_per_node);
        // The rows, through the tiles, for a chunk narrower than four nodes: the occupant's own
        // rungs, where the tiles under the boots are shipped.
        if levels == 0 || node_cells * 4 > chunk_cells {
            return 0;
        }
        // ★ THE FINEST LEVEL WHOSE NODE IS AT LEAST THE CELL (the far-view ship's first flights,
        // 2026-09-20). The rule before this took the COARSEST level with four nodes across the
        // chunk, so a rung-14 chunk of 16 km cells read 262 km nodes, every rung read another
        // level, and the owner saw the coast redraw itself at each rung swap and a globe of blobs
        // from orbit. A cell can show a node no finer than itself; a coarser node than the cell
        // throws away what the pyramid holds. Rungs 10 to 14 on the home planet read level 1
        // (16 km nodes) alike, so a swap between them moves no coast; each rung above climbs one.
        let cell = 1u64 << rung;
        let mut level = 1u32;
        let mut node = node_cells << 1;
        while level < levels && node < cell {
            level += 1;
            node <<= 1;
        }
        level
    }
}

/// ★ THE RING RULES, ONE HOME (2026-09-20): the distance at which one cell of `rung` stands one
/// pixel high in the reference view (`cell_m(rung) / pixel`), which is where the client's ladder
/// hands a rung to the next; and the smooth-sphere horizon from `altitude_m` over a body of
/// `radius_m`. The client's ladder read these from its own module and the shard had no copy, so
/// the shard shipped tiles by the occupant's INTEREST SIDE — 76 m for a standing dot, ONE tile —
/// while the client asked the fine rungs out to 445 km. MEASURED on a 40 s leg from 20 km: one
/// tile received, 730 chunks wanted under drawn ground and missing, 2.2 million requests refused
/// for a field not whole, 598 chunks revealed on the way down. Both hosts now read one rule.
/// `pixel_rad` is the reference view's pixel (the drawable floor), which lives beside the view in
/// `vd_core` and is passed in: this crate names nothing above the recipe and the leaf.
#[must_use]
pub fn switch_m(rung: u8, pixel_rad: f64) -> f64 {
    f64::from(vd_seed::ladder::cell_m(rung)) / pixel_rad
}

/// The smooth-sphere horizon of an eye `altitude_m` over a sphere of `radius_m`, in metres.
#[must_use]
pub fn horizon_m(radius_m: f64, altitude_m: f64) -> f64 {
    let h = if altitude_m > 0.0 { altitude_m } else { 0.0 };
    (2.0 * radius_m * h + h * h).sqrt()
}

/// ★ THE LADDER'S OUTER EDGE, ONE HOME (2026-09-21, the far ring's tiles): the client's ladder asks a
/// rung out to `HYSTERESIS_OUT` of its switch distance (the crossfade band's outer edge, ruling
/// D8-2) widened by `ASK_SLACK` where the bounded ask slides. The shard and the gateway ship tiles
/// to the same edge, so no chunk the ladder asks for ever waits on a tile nobody sends. MEASURED
/// before this: about 23 rung-9 chunks past the horizon asked for one tile for the whole coast leg,
/// because the reach stopped at the horizon while the ladder's skyline admits the peaks past it.
/// The client reads these constants from here (`vd_client::ladder_view`).
pub const ASK_HYSTERESIS_OUT: f64 = 1.1;
/// The bounded ask's slack over the outer edge (its derivation and assertions stand in the ladder).
pub const ASK_SLACK: f64 = 0.17;

/// ★ HOW FAR THE GROUND CAN BE SEEN from `altitude_m` over a body of `radius_m` whose ground rises
/// up to `relief_m` over the radius: the smooth horizon plus the horizon a peak of the relief stands
/// over — the same reach the ladder's skyline admits columns by. One home for the ladder, the shard
/// and the gateway.
#[must_use]
pub fn reach_m(radius_m: f64, altitude_m: f64, relief_m: f64) -> f64 {
    horizon_m(radius_m, altitude_m) + horizon_m(radius_m, relief_m)
}

/// ★ THE SWITCH DISTANCE FLOOR, ONE HOME (ruling T7 rule 3; ONE home since ruling W17): a handover
/// whose step is `step_m` may not happen nearer than `step_m / (2 · pixel_rad)`, because the
/// ladder's own tolerance at a handover distance `D` is ONE CELL of the rung that takes over,
/// which is `2 · D · pixel_rad` metres. The client's ladder read this from its own module and the
/// shard had no copy, so a client that pushed a rung's ring out for its step waited for tiles the
/// shard did not ship. Both hosts now read one rule, exactly as they do for [`switch_m`].
#[must_use]
pub fn switch_floor_m(step_m: f64, pixel_rad: f64) -> f64 {
    if step_m > 0.0 {
        step_m / (2.0 * pixel_rad)
    } else {
        0.0
    }
}

/// ★ THE FINEST RUNG THAT READS THE ROWS — the rung whose TILES a shard ships, named so that both
/// hosts can ask the same rung what its handover costs (ruling W17).
#[must_use]
pub fn tile_rung(lattice: &MacroLattice, levels: u32, top_rung: u8) -> u8 {
    let mut rung = 0u8;
    while rung < top_rung && PyramidField::level_for(lattice, levels, rung + 1) == 0 {
        rung += 1;
    }
    rung
}

/// ★ THE FIELD'S OWN STEP AT A HANDOVER, in metres (2026-09-23, ruling W17; the owner, after W16:
/// *"for some of the far-view rungs the change is still visible — not for close or very far
/// view"*).
///
/// **THE DEFECT THIS ANSWERS, with its number.** A rung reads the artifact's ROWS or a PYRAMID
/// LEVEL, and a level's node is the MEAN OF FOUR of the level under it. On the home planet rungs 0
/// to 9 read the rows and rungs 10 to 14 read level 1, so at the 9 → 10 swap — and at no other
/// swap a hull pilot can see — the ground is read from a field twice as coarse. MEASURED through
/// the shipped reader (`vd-bins/examples/rung_swap`, 9 600 directions a pair): the surface moves
/// by up to **1 442.5 m, which is 1.41 CELLS of the rung that takes over and 2.82 PIXELS at the
/// distance the swap happens**, against the ladder's own tolerance of ONE cell (two pixels). Every
/// other pair of the home planet stands inside it — the worst of the rest is 0.62 cells at 8 → 9.
/// And the ladder did not know: [`crate::BodyDefinition::step_bound_m`] reads the OCTAVE TABLE
/// alone, which is exact where both rungs read one field (319.4 m stated against 318.5 m measured
/// at 8 → 9) and silent about the level's own fold (897.7 m stated against 1 442.5 m measured at
/// 9 → 10). So ruling T7's rule 2 (the band's widening) and rule 3 (the switch's floor) both
/// answered ONE and slept at the only swap that needed them.
///
/// **THE LAW, and NOTHING NEW CROSSES A REALM BOUNDARY (SL6).** A fold takes FOUR nodes to ONE, so
/// the coarse level states a height for the CENTRE of a square one node wide, while each child's
/// own height stands at `node / (2·√2)` from that centre — the distance from a parent node's
/// centre to a child node's, a quarter of a node in each axis. The two therefore differ by the
/// FIELD'S OWN SLOPE over that distance. The body already states that slope —
/// [`crate::BodyDefinition::slope_ref`], the first fine octave's own RMS slope, which every host
/// holds in the charter — and the lattice states the node. So the step is
/// `slope_ref · node_m / (2·√2)` for each level the swap crosses, and zero where both rungs read
/// one level. No number is drawn (ruling T9) and no host is told anything it did not already hold.
///
/// **AND IT IS A BOUND, MEASURED.** On the home planet the law states 722.0 m at the rows →
/// level-1 swap, against a measured field contribution of 544.8 m (the whole step 1 442.5 m less
/// the octave table's own 897.7 m): the law stands over the measurement by a third, which is what
/// a bound must do.
///
/// **Example.** The pilot's hull stands 150 km over the belt. The ring where metre-and-a-half
/// cells hand the ground to three-metre cells stood 445 km out, and the ground slid 1.4 km as she
/// flew through it. Now the rows keep that ring out to 627 km, where the same slide is one cell of
/// the rung that takes over, and the crossfade band that pays it off is 1.41 times as wide.
#[must_use]
pub fn field_step_m(
    body: &crate::body::BodyDefinition,
    lattice: &MacroLattice,
    levels: u32,
    rung: u8,
) -> f64 {
    let here = PyramidField::level_for(lattice, levels, rung);
    let next = PyramidField::level_for(lattice, levels, rung.saturating_add(1));
    let slope = crate::units::share_of_q28(body.slope_ref());
    let mut step = 0.0;
    let mut level = here + 1;
    while level <= next {
        if let Some(coarse) = lattice.coarser(level) {
            step += slope * coarse.node_m() / (2.0 * std::f64::consts::SQRT_2);
        }
        level += 1;
    }
    step
}

/// ★ HOW FAR THE TILES MUST REACH under an occupant standing `radial_m` from the body's centre:
/// the finest rung that reads the rows (the tiles) is wanted out to the ladder's OUTER EDGE for it
/// (`ASK_HYSTERESIS_OUT` and `ASK_SLACK` over its switch distance), bounded by how far the ground
/// can be seen ([`reach_m`]: the horizon plus the relief's own), and the sixteen-node stencil of the
/// last chunk reaches two nodes past that. A shard ships the tiles within this of the occupant's
/// direction, so the client's fine rungs never wait on a tile the shard did not send.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "six of the eight name the body and its lattice, and the last two are the reference \
              view's pixel and the handover step the ring must cover; a struct here would be a \
              second home for numbers the callers already hold apart"
)]
pub fn tile_reach_m(
    lattice: &MacroLattice,
    levels: u32,
    top_rung: u8,
    body_radius_m: f64,
    relief_m: f64,
    radial_m: f64,
    pixel_rad: f64,
    step_m: f64,
) -> f64 {
    let rung = tile_rung(lattice, levels, top_rung);
    let altitude_m = radial_m - body_radius_m;
    // ★ THE RING IS THE ONE THE CLIENT ASKS AT (ruling W17): the tier rule's own switch distance,
    // never nearer than the floor the handover's own step sets. `step_m` is that step — the
    // octaves the coarser rung drops plus [`field_step_m`] where the field changes under them —
    // and a caller that states zero gets the ladder every earlier flight flew.
    let tier = switch_m(rung, pixel_rad);
    let floor = switch_floor_m(step_m, pixel_rad);
    // The comparison is written out: `f64::max` is disallowed here (SL10 clause 4).
    let at = if floor > tier { floor } else { tier };
    let asked = at * ASK_HYSTERESIS_OUT * (1.0 + ASK_SLACK);
    let seen = reach_m(body_radius_m, altitude_m, relief_m);
    let ring_m = if seen < asked { seen } else { asked };
    ring_m + 2.0 * lattice.node_m()
}

/// ★ THE TILES WITHIN `radius_m` OF THE UNIT DIRECTION `dir`, the nearest first (2026-09-20, the
/// one tile path): the face the direction falls on, the node under it, and every tile a node
/// within the reach along that face's axes lies in — ACROSS A CUBE EDGE TOO (2026-09-21): the
/// node square is walked through the seam table ([`node_at`]), so a reach that crosses the face's
/// edge names the partner face's tiles, which a chunk at the edge reads (`tiles_of_chunk` walks
/// the same seam). MEASURED before this: a chunk across a cube edge waited for the partner face's
/// tile for as long as the eye stood there. The shard answers a want with these; the gateway
/// serves a session those of them within its own reach — the same function on both hosts (SL10),
/// so what is asked and what is served can never disagree. A direction with no length names
/// nothing. The nearest first, by the angle between the tile's centre node and the direction.
///
/// **Example.** A pilot standing forty nodes from the +Z face's edge with a reach of a hundred
/// nodes is served the +Z tiles and the first tiles of the +X face beyond the edge, so the ground
/// under the seam builds at the fine rungs instead of standing at the coarser one.
#[must_use]
pub fn tiles_within(lattice: &MacroLattice, dir: [f64; 3], radius_m: f64) -> Vec<(u8, u32, u32)> {
    use vd_seed::bend::{BASIS, face_of, unbend};
    use vd_seed::ladder::index_of;
    let face = face_of(dir);
    let basis = BASIS[face.index() as usize];
    let axis =
        |a: [i8; 3]| dir[0] * f64::from(a[0]) + dir[1] * f64::from(a[1]) + dir[2] * f64::from(a[2]);
    let n = axis(basis.n);
    if n <= 0.0 {
        return Vec::new();
    }
    let (a, b) = (unbend(axis(basis.u) / n), unbend(axis(basis.v) / n));
    let edge = lattice.edge;
    let (i, j) = (index_of(a, edge), index_of(b, edge));
    let reach = radius_m / lattice.node_m();
    let reach_nodes = if reach > 1.0 { reach.ceil() as i64 } else { 1 };
    // Past a whole face the square wraps onto the far side of the cube: the walk stops one face
    // short of it, which is every tile of the six faces the direction can see.
    let reach_nodes = reach_nodes.min(i64::from(edge));
    let (i_lo, i_hi) = (i64::from(i) - reach_nodes, i64::from(i) + reach_nodes);
    let (j_lo, j_hi) = (i64::from(j) - reach_nodes, i64::from(j) + reach_nodes);
    // The walk along an axis: INSIDE the face a stride of one tile plus the last node never
    // skips a tile; PAST the face's edge the seam table runs the partner's coordinate the other
    // way, from its far tile inward, so every node past the edge is visited (at most the reach,
    // a few dozen) — MEASURED before this: a stride that landed seven nodes past the edge read
    // the partner's tile 0 and skipped its tile 1, the very tile the chunk at the edge reads.
    let walk = |lo: i64, hi: i64| -> Vec<i64> {
        let n = i64::from(edge);
        let mut out: Vec<i64> = (lo..0).collect();
        // The range always holds the node under the direction, which is inside the face, so the
        // inside part is never empty.
        let (in_lo, in_hi) = (lo.max(0), hi.min(n - 1));
        out.extend((in_lo..=in_hi).step_by(TILE_EDGE as usize));
        out.push(in_hi);
        out.extend(n.max(lo)..=hi);
        out
    };
    let mut set = BTreeSet::new();
    for jj in walk(j_lo, j_hi) {
        for ii in walk(i_lo, i_hi) {
            set.insert(tile_of_node(edge, node_at(lattice, face, ii, jj)));
        }
    }
    let mut out: Vec<(u8, u32, u32)> = set.into_iter().collect();
    // The nearest first: by the angle between the tile's centre node and the direction — one
    // measure that serves a partner face's tile as well as this face's.
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let away = |&(f, tx, ty): &(u8, u32, u32)| -> f64 {
        let (cx, cy) = (
            (tx * TILE_EDGE + tile_width(edge, tx) / 2) as i32,
            (ty * TILE_EDGE + tile_width(edge, ty) / 2) as i32,
        );
        let face = Face::from_index(f).unwrap_or(face);
        let d = lattice.direction(lattice.index(face, cx, cy));
        let len = (0..3)
            .map(|k| {
                let c = d[k].raw() as f64 / unit;
                c * c
            })
            .sum::<f64>()
            .sqrt();
        -(0..3)
            .map(|k| dir[k] * d[k].raw() as f64 / unit)
            .sum::<f64>()
            / len
    };
    out.sort_by(|p, q| away(p).total_cmp(&away(q)));
    out
}

/// A tile's width in nodes: `TILE_EDGE`, or what is left of the edge for the last tile.
#[must_use]
pub fn tile_width(edge: u32, t: u32) -> u32 {
    (edge - t * TILE_EDGE).min(TILE_EDGE)
}

/// ★ THE ROCK PROVINCE UNDER A COLUMN (slice 8d step 2): the NEAREST node's own word — the node
/// whose centre stands nearest the cell's centre. A province is a REGION, not a surface:
/// blending two provinces would make a rock that is half basalt, which no wall is. So the column
/// takes the nearest node's word whole, and the change stands at the line between two nodes.
/// `None` where the field holds no province (a pyramid level, an unarrived tile), and the column
/// then reads [`crate::strata::DEFAULT_PROVINCE`].
///
/// **Example.** A pilot walks east over the line where the shelf meets the basement. The wall of
/// her mine is limestone on one side of the line and granite on the other, and the line stands
/// four kilometres from either node.
#[must_use]
pub fn sample_province(
    lattice: &MacroLattice,
    field: &dyn ZField,
    face: Face,
    rung: u8,
    i: i32,
    j: i32,
) -> Option<u8> {
    let half = Gi::new(1i64 << (NOISE_BITS - 1));
    let (ni, ta) = node_and_fraction(lattice, rung, i);
    let (nj, tb) = node_and_fraction(lattice, rung, j);
    let ni = ni + i64::from(ta >= half);
    let nj = nj + i64::from(tb >= half);
    field.province(node_at(lattice, face, ni, nj))
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
///
/// ★ ONE NODE WIDER EITHER WAY (2026-09-21): a column also reads the field ONE MACRO NODE to each
/// side for its slope share ([`slope_share`]), so the stencil of the neighbour columns reaches one
/// node further than the column's own. A span that stopped where it did before would leave the
/// identity's golden fields a row short and every self-check chunk would refuse to build.
#[must_use]
pub fn nodes_of_chunk(lattice: &MacroLattice, key: ChunkKey) -> BTreeSet<u32> {
    let span = |c: i32| -> (i64, i64) {
        let first = c * CHUNK_EDGE as i32 - crate::lattice::HALO;
        let last = (c + 1) * CHUNK_EDGE as i32 - 1 + crate::lattice::HALO;
        (
            node_and_fraction(lattice, key.rung, first).0 - 2,
            node_and_fraction(lattice, key.rung, last).0 + 3,
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
/// (a few dozen nodes, `SparseZ`) and THE PYRAMID LEVEL THE TOP-RUNG KEYS READ, which the two top-rung keys read
/// the way the far view does. Both are literals in the build (`home::home_golden_fields`), so every
/// host measures the same eight chunks through the same read at boot, with no artifact in hand; the
/// artifact pin proves the literals are the solve's own rows.
///
/// ★ THE GOLDEN FIELDS CARRY NO COAST MASK (2026-09-22, ruling W10), and that is a decision. The
/// mask is two bits per fine node — 2.2 megabytes on the home planet — and these fields are a TEXT
/// literal in the build; megabytes of bits in the source would state what the rows already state.
/// Since ruling B22 (2026-09-28) no column reads a side at all, so the mask's absence moves no
/// byte of the eight chunks. Both hosts read the SAME text, so the identity stays one number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenFields {
    /// How many pyramid levels the artifact holds (the level a rung reads depends on it).
    pub levels: u32,
    /// The artifact's sea (`Artifact::sea_m`): the self-check's body reads it as its biome datum.
    pub sea_m: i16,
    /// The rows under the fine keys: the height, the water level, the facies and the province.
    pub rows: SparseRows,
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

    /// The fields as text, one item a line: `h levels top_level sea_m`, `t z` for each top word in
    /// order, `r node z water facies province` for each row (the province joined in slice 8d
    /// step 2, so the self-check reads the rock the shipped row states).
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("h {} {} {}\n", self.levels, self.top.level, self.sea_m);
        for z in &self.top.z_m {
            out.push_str(&format!("t {z}\n"));
        }
        for (node, (z, w, f, p)) in &self.rows.0 {
            out.push_str(&format!("r {node} {z} {w} {f} {p}\n"));
        }
        out
    }

    /// The body's sea as the fields state it: `None` for a dry body.
    #[must_use]
    pub fn sea(&self) -> Option<i32> {
        (self.sea_m != DRY_M).then_some(i32::from(self.sea_m))
    }

    /// The fields back from [`GoldenFields::to_text`]; `None` for a line that is not one of the three.
    #[must_use]
    pub fn parse(text: &str) -> Option<GoldenFields> {
        let mut header = None;
        let mut top = Vec::new();
        let mut rows = BTreeMap::new();
        for line in text.lines() {
            let mut w = line.split_whitespace();
            match w.next()? {
                "h" => {
                    let levels = w.next()?.parse::<u32>().ok()?;
                    let top_level = w.next()?.parse::<u32>().ok()?;
                    let sea_m = w.next()?.parse::<i16>().ok()?;
                    header = Some((levels, top_level, sea_m));
                }
                "t" => top.push(w.next()?.parse::<i16>().ok()?),
                "r" => {
                    let node = w.next()?.parse::<u32>().ok()?;
                    let z = w.next()?.parse::<i16>().ok()?;
                    let water = w.next()?.parse::<i16>().ok()?;
                    let facies = w.next()?.parse::<u8>().ok()?;
                    let province = w.next()?.parse::<u8>().ok()?;
                    rows.insert(node, (z, water, facies, province));
                }
                _ => return None,
            }
        }
        let (levels, top_level, sea_m) = header?;
        Some(GoldenFields {
            levels,
            sea_m,
            // The rows read the sea the header states, so a golden row names a pocket's floor
            // by the rule the mask was written with.
            rows: SparseRows(rows, sea_m),
            top: PyramidField {
                level: top_level,
                z_m: top,
                // The golden top carries no water word and no coast mask, for the reason the
                // type's own note states: no column reads either since ruling B22.
                water_m: Vec::new(),
                coast: None,
                counts: None,
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
/// ★ ONE NODE WIDER EITHER WAY (2026-09-21), for the slope share the column also reads
/// ([`slope_share`]): the same widening [`nodes_of_chunk`] states.
///
/// **Example.** A rung-0 chunk on the home planet whose columns sit over macro node 63 gathers nodes
/// 61 to 66 and names tiles 0 and 1 along that axis; a chunk at the face's low edge names the tile
/// across the seam as well.
#[must_use]
pub fn tiles_of_chunk(lattice: &MacroLattice, key: ChunkKey) -> BTreeSet<(u8, u32, u32)> {
    let span = |c: i32| -> (i64, i64) {
        let first = c * CHUNK_EDGE as i32 - crate::lattice::HALO;
        let last = (c + 1) * CHUNK_EDGE as i32 - 1 + crate::lattice::HALO;
        (
            node_and_fraction(lattice, key.rung, first).0 - 2,
            node_and_fraction(lattice, key.rung, last).0 + 3,
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

/// ★ THE WATER WORD OF ONE PYRAMID BLOCK (2026-09-24, plan step 2 part 5; ruling W16: a rung
/// boundary is invisible): the HIGHEST lake level among the block's lake nodes where it holds any,
/// else the sea's level where it holds a sea node, else [`DRY_M`]. `lake_max` is that highest
/// level ([`DRY_M`] for none), `has_sea` whether a sea node stands under the block. No majority and
/// no mean: a lake node alone in its block keeps its level at the far rungs, and a block of sea and
/// lake never reads a level no water has.
///
/// ★ WHY THE HIGHEST. Until ruling B22 only a LAKE column read this word, and the shore law held
/// it UNDER its water by a quarter of its ground's depth. Under a LOWER word the higher lake's
/// floor was pushed down under a surface its own water never had; under the higher word no ground
/// was pushed lower than its own lake puts it. ★ Since ruling B22 (2026-09-28) no column reads the
/// word; it is stored for the one-volume fill step.
///
/// **Example.** A tarn fills one node of the belt's pass at 2 100 m, and its three neighbours in
/// the level-1 block are rock. The word is 2 100 m, so the far view's level still names the tarn at
/// its own level; before this word the fold turned it to rock.
#[must_use]
pub fn block_water(lake_max: i16, has_sea: bool, sea_m: i16) -> i16 {
    if lake_max != DRY_M {
        lake_max
    } else if has_sea {
        sea_m
    } else {
        DRY_M
    }
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
                // The rock map the initial land read, carried through the solve untouched.
                province: state.province[i],
            });
        }
        let mut pyramid = Vec::new();
        let mut pyramid_water = Vec::new();
        let sea_m = if has_water && state.sea_z != i32::MIN {
            metres_i16(state.sea_z)
        } else {
            DRY_M
        };
        let mut below: Vec<i16> = rows.iter().map(|r| r.z_m).collect();
        // ★ THE WATER FOLD (plan step 2 part 5): each fine node's LAKE level (its row's water where
        // its side is a lake, else DRY) and whether it is the SEA, carried up every level as the
        // highest lake and "any sea"; each level's word is [`block_water`] of the two.
        let mut below_lake: Vec<i16> = rows
            .iter()
            .zip(facies)
            .map(|(r, &f)| if coast_word(f) == 3 { r.water_m } else { DRY_M })
            .collect();
        let mut below_sea: Vec<bool> = facies.iter().map(|&f| coast_word(f) == 1).collect();
        let mut fine = lattice;
        let mut k = 1;
        while let Some(coarse) = lattice.coarser(k) {
            let mut level = vec![0i32; coarse.node_count()];
            let mut lake_max = vec![DRY_M; coarse.node_count()];
            let mut has_sea = vec![false; coarse.node_count()];
            for node in 0..fine.node_count() as u32 {
                let (face, i, j) = fine.split(node);
                let up = coarse.index(face, i >> 1, j >> 1) as usize;
                level[up] += i32::from(below[node as usize]);
                lake_max[up] = lake_max[up].max(below_lake[node as usize]);
                has_sea[up] |= below_sea[node as usize];
            }
            let means: Vec<i16> = level.iter().map(|&s| s.div_euclid(4) as i16).collect();
            let waters: Vec<i16> = lake_max
                .iter()
                .zip(&has_sea)
                .map(|(&l, &s)| block_water(l, s, sea_m))
                .collect();
            below.clone_from(&means);
            below_lake = lake_max;
            below_sea = has_sea;
            pyramid.push(means);
            pyramid_water.push(waters);
            fine = coarse;
            k += 1;
        }
        // ★ THE COAST MASK (ruling W10, widened by W16 and by B7): a TWO-BIT SIDE WORD per FINE
        // node — land, sea, lake or a pocket's dry floor. It is the rows' own facies bits, heights
        // and the artifact's own sea ([`side_word`]) and nothing else, so a host that holds the
        // mask and a host that holds the tile name the same side for the same node.
        let mut coast = vec![0u8; coast_bytes(n)];
        for (i, &f) in facies.iter().enumerate().take(n) {
            coast[i / 4] |= side_word(f, rows[i].z_m, sea_m) << (2 * (i % 4));
        }
        Artifact {
            edge: lattice.edge,
            version: ARTIFACT_VERSION,
            sea_m,
            rows,
            pyramid,
            pyramid_water,
            coast,
        }
    }

    /// ★ THE COAST COUNTS of this artifact (2026-09-22, ruling W15), folded from its own mask.
    /// Build this ONCE per body and hand the same pointer to every pyramid level: the fold walks
    /// every node of every level, and six folds would hold six copies of 11.8 MB.
    #[must_use]
    pub fn coast_counts(&self) -> std::sync::Arc<CoastCounts> {
        std::sync::Arc::new(CoastCounts::of(&self.coast, self.edge))
    }

    /// The sea as the artifact states it: `None` for a dry body.
    #[must_use]
    pub fn sea(&self) -> Option<i32> {
        (self.sea_m != DRY_M).then_some(i32::from(self.sea_m))
    }

    /// The nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.rows.len()
    }

    /// The bytes the rows and the pyramid take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.len() * ROW_BYTES
            + self.pyramid.iter().map(|l| l.len() * 2).sum::<usize>()
            + self.coast.len()
    }

    /// ★ THE DIGEST: two FNV-1a folds over the version, the edge, every row's bytes in node order,
    /// every pyramid level and THE COAST MASK, folded last after the pyramid words, so one flipped
    /// bit of the mask moves the digest and a store that lost it is refused — gate G-DRIFT's word, the same on every target or the boot is red.
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
        for level in self.pyramid.iter().chain(&self.pyramid_water) {
            for &z in level {
                let bytes = z.to_le_bytes();
                a = fnv1a(a, &bytes);
                b = fnv1a(b, &bytes);
            }
        }
        a = fnv1a(a, &self.coast);
        b = fnv1a(b, &self.coast);
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

/// ★ THE SLOPE CHARTER — every word a column's slope share reads that is not the column's own
/// address, drawn ONCE per chunk (2026-09-21). The shape the crate uses for every kernel's words: a
/// per-column read pays for no divide and crosses no float door.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlopeCharter {
    /// The cells of this rung the central difference steps, each way: one macro node.
    pub step: i32,
    /// The reciprocal of TWICE that step in gap steps, at [`SLOPE_INV_BITS`].
    pub inv: Gi,
    /// The body's slope reference at [`NOISE_BITS`] ([`crate::BodyDefinition::slope_ref`]).
    pub reference: Gi,
    /// The reference's own reciprocal at [`crate::body::SLOPE_RECIP_BITS`].
    pub recip: Gi,
}

impl SlopeCharter {
    /// ★ THE CHARTER OF A CHUNK WITH NO FIELD: one cell, and reciprocals of ZERO. No column reads it
    /// — a slope is only measured where a field stands — and a column that did would read a share of
    /// zero, which is the factor the recipe's own relief already uses.
    pub const NONE: SlopeCharter = SlopeCharter {
        step: 1,
        inv: Gi::ZERO,
        reference: Gi::ZERO,
        recip: Gi::ZERO,
    };
}

/// ★ THE STEP THE MACRO SLOPE IS MEASURED OVER, and the two reciprocals with it: how many cells of
/// `rung` stand on one macro node, and the reciprocal of TWICE that step in gap steps.
///
/// The step is `cells_per_node >> rung`, which is exactly one node: the node index of `i + step` is
/// the node index of `i` plus one and the fraction across the node is the same, so a central
/// difference over it reads the field one node either way and never a fraction of one. A rung whose
/// own cell is wider than a node clamps the step to ONE CELL, and the reciprocal below is taken over
/// that real step, so the slope is a slope on every rung.
///
/// The reciprocal is a FENCED FLOAT rounded once into an integer word ([`SLOPE_INV_BITS`]), the way
/// the body's own draw rounds its shares: `node_m` is the lattice's one float, and both hosts fold
/// the same bits of it, so the word is the same on the shard and on the client.
///
/// **Example.** The home planet at rung 0: a node is 8 192 rung-0 cells and 8 192 m, so the step is
/// 8 192 cells and the reciprocal of 2 × 8 192 m in gap steps is 524 288 — one multiply per column.
#[must_use]
pub fn slope_charter(
    body: &crate::BodyDefinition,
    lattice: &MacroLattice,
    rung: u8,
) -> SlopeCharter {
    let step = (lattice.cells_per_node >> rung).max(1);
    // The step in metres: the cells it covers at this rung, times a rung-0 cell, which is a node
    // over the cells a node holds.
    let node_m = crate::gf::Gf::from_f64(lattice.node_m());
    let cells_at_rung = crate::gf::Gf::from_i64(i64::from(step) << rung);
    let two_step_m = crate::gf::Gf::TWO * cells_at_rung * node_m
        / crate::gf::Gf::from_i64(i64::from(lattice.cells_per_node));
    let unit = crate::gf::Gf::from_i64(1i64 << SLOPE_INV_BITS);
    let divisor = two_step_m * crate::gf::Gf::from_i64(STEPS_PER_M);
    let inv = (unit / divisor + crate::gf::Gf::HALF).to_i64_floor();
    SlopeCharter {
        step: step as i32,
        inv: Gi::new(inv),
        reference: body.slope_ref(),
        recip: body.slope_ref_recip(),
    }
}

/// The fraction bits of [`SlopeCharter::inv`]. Twice the smallest macro node in gap steps is
/// about 2 × 8 192 × 128 = 2²¹, so the word is about 2¹⁹ and a two-word product with a height
/// difference cannot wrap.
pub const SLOPE_INV_BITS: u32 = 40;

/// ★ THE MACRO SLOPE'S SHARE at one column (2026-09-21, the owner's stand over the belt): how steep
/// the SOLVED field `Z` is here, as a share of the body's own slope reference
/// ([`crate::BodyDefinition::slope_ref`]), at [`NOISE_BITS`] — ONE where the field is as steep as its
/// first fine octave, ZERO where it is flat. The column's roughness factor is the GREATER of this
/// and the placeholder noise's own (`vd_recipe::height::relief_of_table_from`).
///
/// **The law, in one line:** `share = min(1, |∇Z| / s(first fine))`, and `|∇Z|` is the CENTRAL
/// DIFFERENCE of [`sample_z`] one macro node either way along each face axis, over twice the step.
///
/// **Why each axis is clamped to the reference before the root.** The magnitude is never smaller
/// than either axis, so a column with one axis already at the reference has a share of ONE whatever
/// the other axis says: the clamp changes no share the `min` above would not have clamped anyway,
/// and it keeps both squares under 2⁵⁷ (the reference is at most the angle of repose). The root is
/// the recipe's own [`vd_recipe::root::isqrt`], which is exact on the square of a Q28 word.
///
/// `None` where the field holds no row for one of the four stencils — the same rule the column's own
/// `Z` read follows: the chunk is not built and the coarser rung stands (ruling F9).
///
/// **Example.** The pilot's stand sits over the home planet's highest belt. The solve lifted the
/// column 8 081 m and the node 8 km away a kilometre less, so the share reads ONE, the fine octaves
/// stand whole, and the belt is a range rather than the rolling sheet the owner photographed.
#[must_use]
pub fn slope_share(
    charter: &SlopeCharter,
    lattice: &MacroLattice,
    field: &dyn ZField,
    face: Face,
    rung: u8,
    i: i32,
    j: i32,
) -> Option<Gi> {
    let d = charter.step;
    let axis = |a: i32, b: i32, c: i32, e: i32| -> Option<Gi> {
        let far = sample_z(lattice, field, face, rung, a, b)?;
        let near = sample_z(lattice, field, face, rung, c, e)?;
        let slope = (far - near).mul_shr(charter.inv, SLOPE_INV_BITS);
        Some(vd_recipe::cell::lesser(
            Gi::new(slope.unsigned_abs() as i64),
            charter.reference,
        ))
    };
    let si = axis(i + d, j, i - d, j)?;
    let sj = axis(i, j + d, i, j - d)?;
    let squares = si.unsigned_abs() * si.unsigned_abs() + sj.unsigned_abs() * sj.unsigned_abs();
    let magnitude = Gi::new(vd_recipe::root::isqrt(squares) as i64);
    Some(vd_recipe::cell::lesser(
        Gi::new(1 << NOISE_BITS),
        magnitude.mul_shr(charter.recip, crate::body::SLOPE_RECIP_BITS - NOISE_BITS),
    ))
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
        let climate =
            crate::climate::climate(&moon, &lattice, words, &state.z, Some(state.sea_view()));
        let artifact = Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        (state, artifact)
    }

    /// ★ A STATED WET MOON (plan step 2): the moon's lattice flat at 500 m with the sea at 0 over
    /// every node of face `PosX` (the connected ocean), and the marks each test states: `(face, i,
    /// j, z_m, level_m)`, a node's height and, where `level_m` is `Some`, the lake the budget
    /// would put over it. Returns the state and its artifact.
    fn stated_moon(marks: &[(Face, i32, i32, i32, Option<i32>)]) -> (MacroSolve, Artifact) {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let m = Z_STEPS_PER_M;
        state.z = vec![500 * m; n];
        for k in 0..n {
            if state.lattice.split(k as u32).0 == Face::PosX {
                state.z[k] = -1_000 * m;
            }
        }
        for &(face, i, j, z, _) in marks {
            state.z[state.lattice.index(face, i, j) as usize] = z * m;
        }
        state.sea_z = 0;
        state.route();
        for &(face, i, j, _, level) in marks {
            if let Some(level) = level {
                state.water_z[state.lattice.index(face, i, j) as usize] = level * m;
            }
        }
        let facies = state.facies(&vec![false; n], 0, true);
        let lattice = state.lattice;
        let climate = crate::climate::climate(
            &moon,
            &lattice,
            &home_solve_words(),
            &state.z,
            Some(state.sea_view()),
        );
        let artifact = Artifact::of(&state, &facies, &climate, true);
        (state, artifact)
    }

    /// ★ TESTS 22 AND 23 (plan step 2 part 5; status §6 T5, §7 W6; ruling W16): THE LEVEL-1 WATER
    /// WORD. A lake node ALONE in its block keeps its level (the two-of-four fold turned it to
    /// land); two lakes in one block give the HIGHER level; a lake among three sea nodes is the
    /// lake's level, never the mean of the two; a block of sea holds the sea; a dry block is DRY;
    /// and the lone lake's level climbs to level 2 as well.
    #[test]
    fn the_level_one_word_keeps_a_lone_lake_and_takes_the_highest() {
        assert_eq!(block_water(120, true, 0), 120);
        assert_eq!(block_water(DRY_M, true, 7), 7);
        assert_eq!(block_water(DRY_M, false, 7), DRY_M);
        let (state, artifact) = stated_moon(&[
            (Face::PosZ, 20, 20, 100, Some(150)),
            (Face::PosZ, 24, 20, 100, Some(150)),
            (Face::PosZ, 25, 21, 300, Some(350)),
            (Face::PosX, 40, 40, 100, Some(150)),
        ]);
        let c1 = state.lattice.coarser(1).expect("level 1");
        let word = |f: Face, i: i32, j: i32| artifact.pyramid_water[0][c1.index(f, i, j) as usize];
        assert_eq!(word(Face::PosZ, 10, 10), 150, "a lone lake keeps its level");
        assert_eq!(word(Face::PosZ, 12, 10), 350, "two lakes: the higher");
        assert_eq!(word(Face::PosZ, 20, 20), DRY_M, "a dry block");
        assert_eq!(word(Face::PosX, 5, 5), artifact.sea_m, "a block of sea");
        assert_eq!(
            word(Face::PosX, 20, 20),
            150,
            "a lake among the sea: never a mean"
        );
        let c2 = state.lattice.coarser(2).expect("level 2");
        assert_eq!(
            artifact.pyramid_water[1][c2.index(Face::PosZ, 5, 5) as usize],
            150,
            "the lone lake climbs to level 2"
        );
    }

    /// ★ THE COAST MASK IS THE ROWS' OWN SIDE WORD (2026-09-22, ruling W10; the LAKE 2026-09-23,
    /// ruling W16). Five statements, each of which could fail, on a WET moon (the dry moon's own
    /// words draw no water, so its mask has no word set and would prove nothing):
    ///
    /// 1. the mask holds one two-bit word per fine node and no more;
    /// 2. every word equals the row's own facies — sea, lake or land — so the mask is a COPY of the
    ///    rows, never a second opinion;
    /// 3. both sides are present, so the body really has a shore;
    /// 4. the `ZField` readers agree: the artifact, a tile cache holding the node's tile and a
    ///    pyramid level built from the artifact all answer ONE side for one fine node, and a
    ///    level with no mask answers nothing;
    /// 5. ONE FLIPPED BIT MOVES THE DIGEST, so a store that lost or mangled the mask is refused.
    ///
    /// **Example.** A pilot walks off a beach into the water. The node under her boots has its bit
    /// set; the chunk she stands in reads that bit through its tile, and the globe she saw from
    /// orbit read the same bit through the mask.
    #[test]
    fn the_coast_mask_is_the_rows_own_sea_bit_and_the_digest_holds_it() {
        let (state, artifact) = moon_artifact(&SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        });
        let n = state.node_count();
        assert_eq!(artifact.coast.len(), coast_bytes(n));
        let mut sea = 0usize;
        let mut land = 0usize;
        for node in 0..n as u32 {
            let row = artifact.rows[node as usize];
            let want = side_of_row(row.receiver_facies >> FACIES_SHIFT, row.z_m, artifact.sea_m);
            assert_eq!(coast_side(&artifact.coast, node), Some(want), "node {node}");
            assert_eq!(artifact.water_side(node), Some(want));
            sea += usize::from(want.wet());
            land += usize::from(!want.wet());
        }
        assert!(sea > 0, "{sea} sea nodes");
        assert!(land > 0, "{land} land nodes");
        // A node past the lattice has no bit.
        assert_eq!(artifact.water_side(n as u32 * 8), None);
        // The readers agree on one node: the artifact, the tile that holds it, and a level.
        let node = (0..n as u32)
            .find(|&node| artifact.water_side(node) == Some(Side::Sea))
            .expect("a sea node");
        let (face, tx, ty) = tile_of_node(artifact.edge, node);
        let mut cache = TileCache::new(artifact.edge).with_sea(artifact.sea_m);
        cache.apply(artifact.tile(Face::from_index(face).expect("a face"), tx, ty));
        assert_eq!(cache.water_side(node), Some(Side::Sea));
        // A node whose tile is not here says nothing, and so does a level with no mask.
        let elsewhere = (0..n as u32)
            .find(|&m| tile_of_node(artifact.edge, m) != (face, tx, ty))
            .expect("a node in another tile");
        assert_eq!(cache.water_side(elsewhere), None);
        let level = PyramidField::of(&artifact, 1, &artifact.coast_counts()).expect("level 1");
        assert_eq!(level.water_side(node), Some(Side::Sea));
        assert_eq!(
            PyramidField {
                coast: None,
                ..level.clone()
            }
            .water_side(node),
            None
        );
        // ★ ONE FLIPPED BIT MOVES THE DIGEST.
        let mut flipped = artifact.clone();
        flipped.coast[node as usize / 4] ^= 1 << (2 * (node % 4));
        assert_ne!(flipped.digest(), artifact.digest());
        // A sparse row answers from its own facies byte, and answers nothing for a row it lacks.
        let rows = SparseRows(
            BTreeMap::from([
                (7, (0i16, DRY_M, FACIES_SEA, 0u8)),
                (8, (0, DRY_M, crate::solve::FACIES_COAST, 0)),
            ]),
            DRY_M,
        );
        assert_eq!(rows.water_side(7), Some(Side::Sea));
        assert_eq!(rows.water_side(8), Some(Side::Land));
        assert_eq!(rows.water_side(9), None);
        // ★ A LAKE ROW NAMES ITS OWN WATER (ruling W16): before it, the row's word was the sea bit
        // alone and a lake node read LAND.
        let lakes = SparseRows(
            BTreeMap::from([(7, (0i16, 100i16, crate::solve::FACIES_LAKE, 0u8))]),
            DRY_M,
        );
        assert_eq!(lakes.water_side(7), Some(Side::Lake));
        assert!(Side::Lake.wet());
        assert!(Side::Sea.wet());
        assert!(!Side::Land.wet());
        assert!(!Side::Pocket.wet());
    }

    /// ★ THE POCKET'S WORD (2026-09-25, ruling B7 item 1; review finding 1 of plan step 2): a DRY
    /// node whose row stands under the sea's level is word 2, [`Side::Pocket`], and every reader
    /// names it alike — the mask, a tile's row, a golden row and a pyramid level's footprint. Six
    /// statements, each of which could fail:
    ///
    /// 1. the rule on its own words: dry under the sea is 2; dry at the sea's own metre is 0; a
    ///    lake and the sea keep their words under the level; a dry body names no pocket;
    /// 2. the mask a stated wet moon writes holds the pocket at the pocket's own floor node, and a
    ///    tile and a sparse row read the same side there;
    /// 3. a sparse row with NO sea (the default) names the same floor LAND;
    /// 4. the counts hold the pocket, and a parent's pocket count is its children's sum;
    /// 5. a far cell whose DRY nodes are at least half pocket reads POCKET — a tie too;
    /// 6. a far cell with fewer reads LAND.
    ///
    /// **Example.** The salt pan behind the belt's coast stands 2 km under the sea's level: word 2
    /// in the mask the pilot's globe reads from orbit, and word 2 in the tile under her boots.
    #[test]
    fn a_dry_floor_under_the_sea_is_a_pocket_in_every_reader() {
        use crate::solve::{FACIES_COAST, FACIES_LAKE};
        assert_eq!(side_word(0, -1_400, 3_591), 2);
        assert_eq!(side_word(FACIES_COAST, 3_590, 3_591), 2);
        assert_eq!(side_word(0, 3_591, 3_591), 0, "the sea's own metre is land");
        assert_eq!(side_word(FACIES_LAKE, -1_400, 3_591), 3);
        assert_eq!(side_word(FACIES_SEA, -1_400, 3_591), 1);
        assert_eq!(
            side_word(0, i16::MIN + 1, DRY_M),
            0,
            "a dry body has no pocket"
        );
        assert_eq!(side_of_row(0, -1_400, 3_591), Side::Pocket);
        // 2. A stated moon: the sea over face PosX at 0, a pocket floor at −200 on face PosZ.
        let (state, artifact) = stated_moon(&[(Face::PosZ, 40, 40, -200, None)]);
        let floor = state.lattice.index(Face::PosZ, 40, 40);
        assert_eq!(coast_side(&artifact.coast, floor), Some(Side::Pocket));
        assert_eq!(artifact.water_side(floor), Some(Side::Pocket));
        let (face, tx, ty) = tile_of_node(artifact.edge, floor);
        let face = Face::from_index(face).expect("a face");
        let mut cache = TileCache::new(artifact.edge).with_sea(artifact.sea_m);
        cache.apply(artifact.tile(face, tx, ty));
        assert_eq!(cache.water_side(floor), Some(Side::Pocket));
        let row = artifact.rows[floor as usize];
        let fields = [(
            floor,
            (
                row.z_m,
                row.water_m,
                row.receiver_facies >> FACIES_SHIFT,
                row.province,
            ),
        )];
        let golden = SparseRows(BTreeMap::from(fields), artifact.sea_m);
        assert_eq!(golden.water_side(floor), Some(Side::Pocket));
        // 3. The same row with no sea stated is land.
        let bare = SparseRows(BTreeMap::from(fields), DRY_M);
        assert_eq!(bare.water_side(floor), Some(Side::Land));
        assert_eq!(TileCache::new(artifact.edge).sea_m, DRY_M);
        // 4. The counts: the floor's level-1 block holds one pocket node, its level-2 block too.
        let counts = artifact.coast_counts();
        assert_eq!(counts.pockets(1, Face::PosZ, 20, 20), Some(1));
        assert_eq!(counts.count(1, Face::PosZ, 20, 20), Some((0, 0)));
        assert_eq!(counts.pockets(2, Face::PosZ, 10, 10), Some(1));
        assert_eq!(counts.pockets(1, Face::PosZ, 21, 21), Some(0));
        assert_eq!(counts.pockets(3, Face::PosZ, 0, 0), None);
        // One pocket node of four dry ones is not half: the block reads LAND.
        assert_eq!(counts.side(1, Face::PosZ, 20, 20), Some(Side::Land));
        // 5 and 6. Paint a level-1 block: two pocket floors and two rim nodes is a TIE — POCKET;
        // three pockets — POCKET; one pocket and one sea of four — LAND (one of three dry).
        let lattice = state.lattice;
        let block = [(10, 10), (11, 10), (10, 11), (11, 11)];
        let paint = |words: [u8; 4]| -> CoastCounts {
            let mut mask = artifact.coast.clone();
            for (k, &(i, j)) in block.iter().enumerate() {
                let node = lattice.index(Face::NegZ, i, j) as usize;
                let shift = 2 * (node % 4);
                mask[node / 4] &= !(3u8 << shift);
                mask[node / 4] |= words[k] << shift;
            }
            CoastCounts::of(&mask, artifact.edge)
        };
        assert_eq!(
            paint([2, 2, 0, 0]).side(1, Face::NegZ, 5, 5),
            Some(Side::Pocket)
        );
        assert_eq!(
            paint([2, 2, 2, 0]).side(1, Face::NegZ, 5, 5),
            Some(Side::Pocket)
        );
        assert_eq!(
            paint([2, 1, 0, 0]).side(1, Face::NegZ, 5, 5),
            Some(Side::Land)
        );
        assert_eq!(
            paint([2, 1, 1, 0]).side(1, Face::NegZ, 5, 5),
            Some(Side::Sea),
            "a wet majority stays water"
        );
    }

    /// ★ A COARSE NODE'S SIDE IS THE ONE MOST OF ITS GROUND STANDS ON (2026-09-22, ruling W15): the
    /// counts' own answer, which the mask's fold states for the one-volume fill step (since ruling
    /// B22 no column reads it). The measured defect it cured: at rung 18 a cell covers about a
    /// thousand fine nodes, and the ONE node at the cell's centre painted the whole cell.
    ///
    /// Four statements, each of which could fail: a block whose centre node is land and whose other
    /// three are sea reads SEA; the reverse reads LAND; a tie is WET; and a wet block of mostly lake
    /// reads LAKE, a tie between the two waters too.
    ///
    /// **Example.** One level-1 node over the moon's coast holds a headland and three nodes of bay:
    /// its counts name it SEA, although the headland is its centre node.
    #[test]
    fn a_coarse_node_takes_the_wet_majority_of_its_block() {
        let (_, mut artifact) = moon_artifact(&SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        });
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let face = Face::PosX;
        // The level-1 node (3, 5) covers the fine nodes (6, 10) to (7, 11); (7, 11) is the one
        // nearest the centre of a cell two nodes wide.
        let block = [(6, 10), (7, 10), (6, 11), (7, 11)];
        let centre = lattice.index(face, 7, 11);
        let paint = |artifact: &mut Artifact, words: [u8; 4]| {
            for (n, &(i, j)) in block.iter().enumerate() {
                let node = lattice.index(face, i, j) as usize;
                let shift = 2 * (node % 4);
                artifact.coast[node / 4] &= !(3u8 << shift);
                artifact.coast[node / 4] |= words[n] << shift;
            }
        };
        // The centre node LAND, the other three SEA: the node is SEA.
        paint(&mut artifact, [1, 1, 1, 0]);
        assert_eq!(artifact.water_side(centre), Some(Side::Land));
        let counts = artifact.coast_counts();
        assert_eq!(counts.count(1, face, 3, 5), Some((3, 0)));
        assert_eq!(counts.side(1, face, 3, 5), Some(Side::Sea));
        // The reverse: the centre node SEA, the other three LAND — the node is LAND.
        paint(&mut artifact, [0, 0, 0, 1]);
        let counts = artifact.coast_counts();
        assert_eq!(counts.count(1, face, 3, 5), Some((1, 0)));
        assert_eq!(artifact.water_side(centre), Some(Side::Sea));
        assert_eq!(counts.side(1, face, 3, 5), Some(Side::Land));
        // A TIE IS WET: two of the four.
        paint(&mut artifact, [1, 0, 1, 0]);
        let counts = artifact.coast_counts();
        assert_eq!(counts.count(1, face, 3, 5), Some((2, 0)));
        assert_eq!(counts.side(1, face, 3, 5), Some(Side::Sea));
        // ★ THE WATER'S OWN KIND FOLDS THE SAME WAY (ruling W16): three of the four wet nodes are
        // a LAKE, so the node is a LAKE; a tie between the two waters is a LAKE, stated; a sea
        // majority among the wet ones is the SEA.
        paint(&mut artifact, [3, 3, 3, 0]);
        let counts = artifact.coast_counts();
        assert_eq!(counts.count(1, face, 3, 5), Some((3, 3)));
        assert_eq!(counts.side(1, face, 3, 5), Some(Side::Lake));
        paint(&mut artifact, [3, 1, 0, 0]);
        let counts = artifact.coast_counts();
        assert_eq!(counts.count(1, face, 3, 5), Some((2, 1)));
        assert_eq!(counts.side(1, face, 3, 5), Some(Side::Lake));
        paint(&mut artifact, [1, 1, 3, 0]);
        let counts = artifact.coast_counts();
        assert_eq!(counts.side(1, face, 3, 5), Some(Side::Sea));
    }

    /// ★ A PARENT'S COUNT IS THE SUM OF ITS FOUR CHILDREN'S, ON EVERY LEVEL (2026-09-22, ruling
    /// W15) — which is WHY a coarse rung never contradicts the rung under it: the coarse cell
    /// states the side of the very ground its four children divide between them.
    ///
    /// **Example.** A rung-15 cell over the moon's bay counts 12 wet nodes of 16; its four rung-14
    /// children count 4, 4, 3 and 1, which is 12.
    #[test]
    fn a_parent_count_is_the_sum_of_its_four_children() {
        let (_, artifact) = moon_artifact(&SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        });
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let counts = CoastCounts::of(&artifact.coast, artifact.edge);
        assert_eq!(counts.levels(), 2, "the moon's edge 68 halves twice");
        // Level 1 against the mask itself.
        let mut level_1_total = 0u64;
        for face in 0..6u8 {
            let face = Face::from_index(face).expect("a face");
            let edge = (lattice.edge >> 1) as i32;
            for cj in 0..edge {
                for ci in 0..edge {
                    let want: (u32, u32) = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|(di, dj)| {
                            let node = lattice.index(face, 2 * ci + di, 2 * cj + dj);
                            match artifact.water_side(node) {
                                Some(Side::Sea) => (1, 0),
                                Some(Side::Lake) => (1, 1),
                                _ => (0, 0),
                            }
                        })
                        .fold((0u32, 0u32), |a, b| (a.0 + b.0, a.1 + b.1));
                    assert_eq!(
                        counts.count(1, face, ci as u32, cj as u32),
                        Some(want),
                        "level 1 at {face:?} ({ci}, {cj})"
                    );
                    level_1_total += u64::from(want.0);
                }
            }
        }
        assert!(level_1_total > 0, "the moon has a sea to count");
        // Every level above: the parent is the sum of its four children.
        let edge_2 = (lattice.edge >> 2) as u32;
        for face in 0..6u8 {
            let face = Face::from_index(face).expect("a face");
            for cj in 0..edge_2 {
                for ci in 0..edge_2 {
                    let want: (u32, u32) = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|(di, dj)| {
                            counts
                                .count(1, face, 2 * ci + di, 2 * cj + dj)
                                .expect("a child")
                        })
                        .fold((0u32, 0u32), |a, b| (a.0 + b.0, a.1 + b.1));
                    assert_eq!(counts.count(2, face, ci, cj), Some(want));
                }
            }
        }
        // Past the levels and past a level's own lattice: nothing, never a guess.
        assert_eq!(counts.count(3, Face::PosX, 0, 0), None);
        assert_eq!(counts.count(0, Face::PosX, 0, 0), None);
        assert_eq!(counts.count(1, Face::PosX, lattice.edge, 0), None);
        assert_eq!(counts.count(1, Face::PosX, 0, lattice.edge), None);
        assert_eq!(counts.side(3, Face::PosX, 0, 0), None);
        // A mask no host holds counts nothing, and a lattice that does not halve holds no level.
        assert_eq!(CoastCounts::of(&[], 0).levels(), 0);
        assert_eq!(CoastCounts::of(&[0u8; 6], 1).levels(), 0);
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
            province: crate::strata::Province::FoldedBelt.code(),
        };
        // ★ THE ROW IS TEN BYTES AND EVERY ONE COMES BACK (slice 8d step 2): the round trip is what
        // says the province byte sits in its own place and steals nobody's.
        assert_eq!(row.to_bytes().len(), ROW_BYTES);
        assert_eq!(ROW_BYTES, 10);
        assert_eq!(Row::from_bytes(row.to_bytes()), row);
        assert_eq!(row.to_bytes()[9], 1);
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

    /// ★ THE ARTIFACT OF THE MOON: ten bytes a node, the pyramid halving to the odd factor, the
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
            n * ROW_BYTES + 2 * (6 * 34 * 34 + 6 * 17 * 17) + coast_bytes(n)
        );
        // ★ THE COAST MASK is one bit a node and the moon is dry, so no bit is set (2026-09-22).
        assert_eq!(artifact.coast.len(), coast_bytes(n));
        assert!(artifact.coast.iter().all(|&b| b == 0));
        // A dry, airless moon: no water anywhere, every row's height the rounded state. ★ THE
        // WATER ENDS AT EVERY ROOT AND EVERY PIT (plan step 2b, question 8): the six lowest nodes,
        // every node with no lower neighbour off the flats, and one middle of every CLOSED flat —
        // counted here from the heights alone, never from the tree.
        assert!(artifact.rows.iter().all(|r| r.water_m == DRY_M));
        let outlets = artifact
            .rows
            .iter()
            .filter(|r| r.receiver_facies & RECEIVER_NONE != 0)
            .count();
        let mut lowest: Vec<u32> = (0..n as u32).collect();
        lowest.sort_unstable_by_key(|&v| (state.z[v as usize], v));
        let mut root = vec![false; n];
        for &v in lowest.iter().take(crate::solve::OUTLETS_WITHOUT_SEA) {
            root[v as usize] = true;
        }
        let mut seen = vec![false; n];
        let mut ends = crate::solve::OUTLETS_WITHOUT_SEA;
        for start in 0..n {
            if root[start] || seen[start] {
                continue;
            }
            // The node's flat (itself alone where no neighbour shares its height), and whether any
            // node of it has a lower or a root neighbour.
            let z0 = state.z[start];
            let mut stack = vec![start as u32];
            seen[start] = true;
            let mut drains = false;
            while let Some(v) = stack.pop() {
                for m in state.lattice.neighbours(v) {
                    if m == NO_NODE {
                        continue;
                    }
                    let mi = m as usize;
                    drains |= root[mi] || state.z[mi] < z0;
                    if !root[mi] && state.z[mi] == z0 && !seen[mi] {
                        seen[mi] = true;
                        stack.push(m);
                    }
                }
            }
            ends += usize::from(!drains);
        }
        assert_eq!(outlets, ends, "the roots and the pits end the water");
        assert!(outlets > 6, "the moon's craters hold pits");
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
        let (wet_state, wet) = moon_artifact(&SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        });
        assert_ne!(wet.digest(), artifact.digest());
        assert!(wet.rows.iter().any(|r| r.water_m != DRY_M));
        // ★ THE ROWS FOLLOW THE WATER, NEVER THE ROUTING FILL (ruling W11): a row states water
        // exactly where the state states a level, and a node the flood raised whose budget put no
        // water in it states DRY. Before W11 every raised node carried the fill's own level.
        let mut raised_and_dry = 0usize;
        for (i, r) in wet.rows.iter().enumerate() {
            let level = wet_state.water_level(i as u32);
            assert_eq!(r.water_m == DRY_M, level == crate::solve::DRY);
            if level != crate::solve::DRY {
                assert_eq!(r.water_m, metres_i16(level));
            }
            if wet_state.z_flood[i] > wet_state.z[i] && level == crate::solve::DRY {
                raised_and_dry += 1;
            }
        }
        assert!(raised_and_dry > 0, "a hollow the rain never filled");
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
        // A sparse field answers only what it holds — the height and the water alike; a pyramid
        // level holds its WATER WORD (2026-09-21) with no facies: the moon is dry, so DRY, and a
        // level built with no words answers nothing.
        let mut sparse = SparseRows::default();
        assert_eq!(sample_z(&lattice, &sparse, Face::PosX, 13, 30, 30), None);
        assert_eq!(sparse.water_facies(0), None);
        let level_1 = PyramidField::of(&artifact, 1, &artifact.coast_counts()).expect("level 1");
        assert_eq!(level_1.water_m.len(), level_1.z_m.len());
        assert_eq!(
            PyramidField {
                water_m: Vec::new(),
                ..level_1.clone()
            }
            .water_facies(0),
            None
        );
        assert_eq!(level_1.water_facies(0), Some((DRY_M, 0)));
        let words = (
            artifact.rows[7].water_m,
            artifact.rows[7].receiver_facies >> FACIES_SHIFT,
        );
        assert_eq!(artifact.water_facies(7), Some(words));
        assert_eq!(cache.water_facies(7), Some(words));
        assert_eq!(artifact.water_facies(n as u32), None);
        for row in 0..4 {
            for col in 0..4 {
                let node = lattice.index(Face::PosX, 29 + col, 29 + row);
                let r = artifact.rows[node as usize];
                sparse.0.insert(
                    node,
                    (
                        r.z_m,
                        r.water_m,
                        r.receiver_facies >> FACIES_SHIFT,
                        r.province,
                    ),
                );
            }
        }
        assert_eq!(
            sample_z(&lattice, &sparse, Face::PosX, 13, 30, 30),
            sample_z(&lattice, &artifact, Face::PosX, 13, 30, 30)
        );
        // A row the sparse field holds answers the row's own water and facies.
        let held = lattice.index(Face::PosX, 29, 29);
        assert_eq!(sparse.water_facies(held), artifact.water_facies(held));
        assert!(sparse.water_facies(held).is_some());
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
        // Cells 524 272.. sit over node 63; the stencil 61..=66 crosses into tile 1 along `i`.
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
        // The nodes a rung-0 chunk reads: THIRTY-SIX in the middle of a tile — the sixteen of the
        // column's own stencil and the ring the slope share's neighbour stencils reach (2026-09-21;
        // it was sixteen before the share existed) — and every one is a node the straddling
        // chunk's tiles hold.
        let nodes = nodes_of_chunk(&lattice, key(4_000, 4_000, 0));
        assert_eq!(nodes.len(), 36);
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
        let mut rows = SparseRows::default();
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
            let r = artifact.rows[node as usize];
            rows.0.insert(
                node,
                (
                    r.z_m,
                    r.water_m,
                    r.receiver_facies >> FACIES_SHIFT,
                    r.province,
                ),
            );
        }
        let fields = GoldenFields {
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            rows,
            // The golden top carries no water word and no coast mask (the text has no line for
            // either).
            top: PyramidField {
                water_m: Vec::new(),
                coast: None,
                counts: None,
                ..PyramidField::of(&artifact, 2, &artifact.coast_counts()).expect("level 2")
            },
        };
        let text = fields.to_text();
        assert_eq!(GoldenFields::parse(&text), Some(fields.clone()));
        assert_eq!(GoldenFields::parse("x 1 2"), None);
        assert_eq!(GoldenFields::parse("t 1"), None);
        for torn in [
            "h",
            "h 2",
            "h 2 2",
            "h x 2 0",
            "h 2 x 0",
            "h 2 2 x",
            "t",
            "t x",
            "r",
            "r 1",
            "r 1 2",
            "r 1 2 3",
            "r x 2 3 4",
            "r 1 x 3 4",
            "r 1 2 x 4",
            "r 1 2 3 x",
            "h 2 2 0\nr 1",
            "h 2 2 0\nr 1 2 3",
            "r 1 2 3 4",
            "t 5",
            "",
            "h 2 2 0\n\nt 1",
        ] {
            assert_eq!(GoldenFields::parse(torn), None, "{torn:?}");
        }
        // The airless moon states no sea; a wet one states its level.
        assert_eq!(fields.sea(), None);
        assert_eq!(artifact.sea(), None);
        let wet = GoldenFields {
            sea_m: -300,
            ..fields.clone()
        };
        assert_eq!(wet.sea(), Some(-300));
        assert_eq!(fields.field_for(&lattice, 0).map(|f| f.level()), Some(0));
        assert_eq!(fields.field_for(&lattice, 15).map(|f| f.level()), Some(2));
        // Rungs 10 to 14 read level 1 on the moon, which the golden fields do not hold.
        assert_eq!(PyramidField::level_for(&lattice, fields.levels, 10), 1);
        assert_eq!(PyramidField::level_for(&lattice, fields.levels, 14), 1);
        assert!(fields.field_for(&lattice, 10).is_none());
    }

    /// ★ THE TILE REACH (2026-09-20): the rung-9 switch at the reference pixel is 445 km; a stand
    /// 20 km over the moon sees a 121 km horizon, so the reach is that horizon plus two nodes; a
    /// stand 200 km up sees 426 km, still under the switch; a stand at the surface reaches two
    /// nodes; a negative altitude reads as the surface; with no pyramid every rung reads the rows
    /// and the top rung's switch bounds the reach. The pixel is the reference view's, as a
    /// literal (`2 · tan(22.5°) / 720`): this crate's tests keep the float fence too.
    #[test]
    fn the_tile_reach_is_the_finest_rows_rung_bounded_by_the_horizon() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let r = moon.radius_m();
        let top = moon.ladder.rungs - 1;
        let pixel = 0.828_427_124_746_190_1 / 720.0;
        let switch9 = switch_m(9, pixel);
        assert!((switch9 - 445_000.0).abs() < 1_000.0, "{switch9}");
        assert_eq!(horizon_m(r, -5.0), 0.0);
        let h20 = horizon_m(r, 20_000.0);
        assert!((h20 - 121_000.0).abs() < 1_000.0, "{h20}");
        let margin = 2.0 * lattice.node_m();
        // ★ The reach is bounded by how far the ground can be SEEN: the horizon plus the relief's
        // own horizon (the ladder's skyline admits the peaks past the smooth horizon), never by
        // the smooth horizon alone.
        let relief = 0.0;
        assert_eq!(
            tile_reach_m(&lattice, 2, top, r, relief, r + 20_000.0, pixel, 0.0),
            reach_m(r, 20_000.0, relief) + margin
        );
        let peaks = 3_000.0;
        let with_peaks = tile_reach_m(&lattice, 2, top, r, peaks, r + 20_000.0, pixel, 0.0);
        assert!(with_peaks > h20 + margin, "{with_peaks}");
        assert!(
            (with_peaks - (h20 + horizon_m(r, peaks) + margin)).abs() < 1.0e-6,
            "{with_peaks}"
        );
        let reach200 = tile_reach_m(&lattice, 2, top, r, relief, r + 200_000.0, pixel, 0.0);
        assert!(
            (reach200 - (horizon_m(r, 200_000.0) + margin)).abs() < 1.0e-6,
            "{reach200}"
        );
        assert_eq!(
            tile_reach_m(&lattice, 2, top, r, relief, r - 1.0, pixel, 0.0),
            margin
        );
        // Far enough out the ladder's OUTER EDGE bounds the reach: the switch distance times the
        // band's hysteresis and the ask's slack, a stand 5 000 km up.
        let asked9 = switch_m(9, pixel) * ASK_HYSTERESIS_OUT * (1.0 + ASK_SLACK);
        assert_eq!(
            tile_reach_m(&lattice, 2, top, r, relief, r + 5.0e6, pixel, 0.0),
            asked9 + margin
        );
        // No pyramid: every rung reads the rows; the top rung's asked edge (over 7 100 km on the
        // moon) stands past the 5 300 km horizon, which bounds instead.
        assert!(switch_m(top, pixel) > horizon_m(r, 5.0e6));
        assert_eq!(
            tile_reach_m(&lattice, 0, top, r, relief, r + 5.0e6, pixel, 0.0),
            horizon_m(r, 5.0e6) + margin
        );
    }

    /// ★ THE FIELD'S OWN STEP AT A HANDOVER (2026-09-23, ruling W17): ZERO where both rungs read
    /// ONE field, and the field's own slope over HALF the node it folds into where the level
    /// changes under them. On the home planet the one swap a hull pilot can see is rungs 9 → 10,
    /// where the artifact's ROWS hand the ground to pyramid level 1.
    ///
    /// ★ RED BEFORE W17: `handover_step_m` read the octave table alone, so this step was ZERO
    /// everywhere and ruling T7's rules 2 and 3 slept through the only swap that needed them —
    /// MEASURED at up to 1 442.5 m of moved ground, 1.41 cells of the rung that takes over.
    #[test]
    fn the_field_states_its_own_step_only_where_the_level_changes() {
        let body = crate::home::home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let mut levels = 0u32;
        while lattice.coarser(levels + 1).is_some() {
            levels += 1;
        }
        assert!(levels >= 2, "the home planet folds a pyramid: {levels}");
        // Every rung under the swap reads the rows, so the field costs nothing.
        for rung in 0..9u8 {
            assert_eq!(
                field_step_m(&body, &lattice, levels, rung),
                0.0,
                "rung {rung}"
            );
        }
        // The swap itself: the slope over half level 1's node, and nothing else.
        let slope = crate::units::share_of_q28(body.slope_ref());
        let level1 = lattice.coarser(1).expect("a level");
        assert_eq!(
            field_step_m(&body, &lattice, levels, 9),
            slope * level1.node_m() / (2.0 * std::f64::consts::SQRT_2)
        );
        assert!(field_step_m(&body, &lattice, levels, 9) > 0.0);
        // Rungs 10 to 13 read level 1 alike, so nothing is paid between them.
        for rung in 10..14u8 {
            assert_eq!(
                field_step_m(&body, &lattice, levels, rung),
                0.0,
                "rung {rung}"
            );
        }
        // And a holder with no pyramid at all pays nothing anywhere: the ladder before W17.
        assert_eq!(field_step_m(&body, &lattice, 0, 9), 0.0);
    }

    /// ★ THE TILE RING IS FLOORED BY THE HANDOVER IT MUST COVER (2026-09-23, ruling W17): the
    /// shard ships the tiles out to the very distance the client's ladder asks at, so a rung whose
    /// step pushes its own switch out never waits on a tile nobody sent.
    #[test]
    fn the_tile_ring_is_floored_by_the_handover_it_must_cover() {
        let body = crate::home::home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let top = body.ladder.rungs - 1;
        let pixel = 0.828_427_124_746_190_1 / 720.0;
        let mut levels = 0u32;
        while lattice.coarser(levels + 1).is_some() {
            levels += 1;
        }
        // The finest rung that reads the rows on the home planet is rung 9.
        let rung = tile_rung(&lattice, levels, top);
        assert_eq!(rung, 9);
        let step = body.step_bound_m(rung) + field_step_m(&body, &lattice, levels, rung);
        // The floor stands PAST the tier rule's own switch distance: that is the defect's cure.
        assert!(
            switch_floor_m(step, pixel) > switch_m(rung, pixel),
            "the step {step} floors at {}",
            switch_floor_m(step, pixel)
        );
        // A zero step floors nothing, and the reach is the one every flight before W17 flew.
        assert_eq!(switch_floor_m(0.0, pixel), 0.0);
        assert_eq!(switch_floor_m(-1.0, pixel), 0.0);
        let r = body.radius_m();
        let high = r + 5.0e6;
        let before = tile_reach_m(&lattice, levels, top, r, 0.0, high, pixel, 0.0);
        let after = tile_reach_m(&lattice, levels, top, r, 0.0, high, pixel, step);
        assert!(after > before, "{after} against {before}");
        let owed = (switch_floor_m(step, pixel) - switch_m(rung, pixel))
            * ASK_HYSTERESIS_OUT
            * (1.0 + ASK_SLACK);
        assert!(
            (after - before - owed).abs() < 1.0e-6,
            "{after} - {before} against {owed}"
        );
    }

    /// ★ THE TILES WITHIN A REACH (2026-09-21): a direction with no length falls behind every
    /// face and names no tile; a reach under one node names the one tile the direction stands on;
    /// a wide reach names a block of tiles on that face, the nearest first.
    ///
    /// **Example.** A session over the moon's +Z pole asks the shard what it can see: the tile
    /// under its feet arrives first, and the ring of tiles around it follows.
    #[test]
    fn the_tiles_within_a_reach_come_nearest_first() {
        use vd_seed::ladder::index_of;
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        // A direction with no length: every face axis reads zero, so the face is behind the eye.
        let nothing = tiles_within(&lattice, [0.0, 0.0, 0.0], 100_000.0);
        assert_eq!(nothing, Vec::new());
        // A reach under one node still names the one tile the direction stands on.
        let dir = [0.0, 0.0, 1.0];
        let mid = index_of(0.0, lattice.edge);
        let under = tile_of_node(lattice.edge, lattice.index(Face::PosZ, mid, mid));
        assert_eq!(
            tiles_within(&lattice, dir, lattice.node_m() / 4.0),
            vec![under]
        );
        // A reach of half a face less a node names a block of tiles on the face the eye looks at,
        // and no other: the square stays inside the face (the moon's face is 68 nodes, two tiles).
        let quarter = f64::from(lattice.edge / 2 - 1) * lattice.node_m();
        let wide = tiles_within(&lattice, dir, quarter);
        let count = wide.len();
        assert!(count > 1, "{count}");
        assert_eq!(wide[0], under);
        assert_eq!(
            wide.iter()
                .filter(|&&(f, _, _)| f != Face::PosZ.index())
                .count(),
            0
        );
        // The nearest first: the angle from each tile's centre node to the direction never falls
        // along the list.
        let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
        let angle = |&(f, tx, ty): &(u8, u32, u32)| -> f64 {
            let face = Face::from_index(f).expect("a face");
            let (cx, cy) = (
                (tx * TILE_EDGE + tile_width(lattice.edge, tx) / 2) as i32,
                (ty * TILE_EDGE + tile_width(lattice.edge, ty) / 2) as i32,
            );
            let d = lattice.direction(lattice.index(face, cx, cy));
            let v: Vec<f64> = d.iter().map(|g| g.raw() as f64 / unit).collect();
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            -(v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2]) / len
        };
        let keys: Vec<f64> = wide.iter().map(angle).collect();
        let mut sorted = keys.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(keys, sorted);
        // ★ ACROSS A CUBE EDGE: an eye near the +Z face's +u edge with a reach past it is served
        // the partner face's first tiles too — the tiles a chunk across the seam reads.
        let near_edge = lattice.index(Face::PosZ, lattice.edge as i32 - 2, mid);
        let d = lattice.direction(near_edge);
        let v: Vec<f64> = d.iter().map(|g| g.raw() as f64 / unit).collect();
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let edge_dir = [v[0] / len, v[1] / len, v[2] / len];
        let across = tiles_within(&lattice, edge_dir, 8.0 * lattice.node_m());
        let faces: BTreeSet<u8> = across.iter().map(|&(f, _, _)| f).collect();
        assert!(faces.contains(&Face::PosZ.index()), "{across:?}");
        assert_eq!(faces.len(), 2, "the partner face too: {across:?}");
        let key = ChunkKey {
            face: Face::PosZ,
            rung: 0,
            x: (lattice.edge as i32 * lattice.cells_per_node as i32 - 1) / CHUNK_EDGE as i32,
            y: (mid * lattice.cells_per_node as i32) / CHUNK_EDGE as i32,
            z: 0,
        };
        for tile in tiles_of_chunk(&lattice, key) {
            assert!(across.contains(&tile), "{tile:?} not in {across:?}");
        }
        // A reach wider than a face wraps no further than the whole cube.
        let all = tiles_within(
            &lattice,
            dir,
            10.0 * f64::from(lattice.edge) * lattice.node_m(),
        );
        let per_edge = lattice.edge.div_ceil(TILE_EDGE) as usize;
        assert!(all.len() <= 6 * per_edge * per_edge, "{}", all.len());
    }

    /// ★ A BODY THAT HOLDS WATER WHOSE SOLVE SETTLED NO LEVEL states a DRY artifact (2026-09-21):
    /// the body's own word is not enough — the state must name a sea as well.
    ///
    /// **Example.** The moon's charter names water, but a fresh state holds no settled sea, so the
    /// artifact the shard ships says DRY and every column reads the body's own level instead.
    #[test]
    fn a_water_body_with_no_settled_sea_states_a_dry_artifact() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let words = home_moon_solve_words();
        let mut state = MacroSolve::new(&moon).expect("a state");
        state.sea_z = i32::MIN;
        let climate = crate::climate::climate(&moon, &lattice, &words, &state.z, None);
        let facies = vec![0u8; state.node_count()];
        let artifact = Artifact::of(&state, &facies, &climate, true);
        assert_eq!(artifact.sea(), None);
    }

    /// ★ THE PYRAMID AS A FIELD: level 1 over the moon's coarser lattice reads the block means, the
    /// level a rung reads climbs with the rung, and a read at a coarse rung through the level agrees
    /// with the read through the rows to within the block mean's own error.
    #[test]
    fn a_pyramid_level_reads_as_a_coarse_field() {
        let (state, artifact) = moon_artifact(&home_moon_solve_words());
        let lattice = state.lattice;
        let level1 = PyramidField::of(&artifact, 1, &artifact.coast_counts()).expect("level 1");
        let coarse = lattice.coarser(1).expect("a level");
        assert_eq!(level1.z_m.len(), coarse.node_count());
        assert_eq!(
            PyramidField::of(&artifact, 3, &artifact.coast_counts()),
            None
        );
        assert_eq!(
            PyramidField::of(&artifact, 0, &artifact.coast_counts()),
            None
        );
        // The rows for a chunk narrower than four nodes (rung 9: 3.9 nodes); from rung 10 the
        // finest level whose node is at least the cell: level 1 (16 km nodes) from rung 10 to
        // rung 14 (16 km cells), level 2 at rung 15 (32 km cells), the last the moon has from
        // there up; no pyramid, no level.
        assert_eq!(PyramidField::level_for(&lattice, 2, 9), 0);
        assert_eq!(PyramidField::level_for(&lattice, 2, 10), 1);
        assert_eq!(PyramidField::level_for(&lattice, 2, 11), 1);
        assert_eq!(PyramidField::level_for(&lattice, 2, 13), 1);
        assert_eq!(PyramidField::level_for(&lattice, 2, 14), 1);
        assert_eq!(PyramidField::level_for(&lattice, 2, 15), 2);
        assert_eq!(PyramidField::level_for(&lattice, 2, 20), 2);
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
        assert!(level_m >= lo - 1.0, "{level_m} under {lo}");
        assert!(level_m <= hi + 1.0, "{level_m} over {hi}");
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
        let raw = t.raw();
        assert!((raw - half).abs() < (1 << (NOISE_BITS - 12)), "{raw}");
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

    /// A SYNTHETIC SOLVED FIELD that rises by `rise_m` metres per macro node along `i` and stands
    /// level along `j`. Catmull-Rom reproduces a straight line exactly, so the central difference
    /// over one node either way is `2 × rise_m` by construction, and the test knows the slope it
    /// asked for. `i0` keeps the heights inside a row's own word near the column the test reads.
    struct Ramp {
        lattice: MacroLattice,
        i0: i32,
        rise_m: i32,
    }

    impl ZField for Ramp {
        fn z_m(&self, node: u32) -> Option<i16> {
            let (_, i, _) = self.lattice.split(node);
            Some(((i - self.i0) * self.rise_m).clamp(-30_000, 30_000) as i16)
        }
        fn water_facies(&self, _node: u32) -> Option<(i16, u8)> {
            None
        }
        fn province(&self, _node: u32) -> Option<u8> {
            None
        }
        /// A ramp names no water and therefore no side.
        fn water_side(&self, _node: u32) -> Option<Side> {
            None
        }
    }

    /// ★ THE MACRO SLOPE'S SHARE, ON A FIELD THE TEST AUTHORS (2026-09-21, the owner's stand over
    /// the belt). Four statements, each of which could fail:
    ///
    /// 1. A FLAT field gives a share of ZERO — the noise placeholder's factor stands alone, so
    ///    nothing the owner has already seen moves where the solve left the ground level.
    /// 2. A STEEP field — steeper than the body's slope reference — gives a share of ONE.
    /// 3. A field BETWEEN them gives the ratio, within a thousandth: the law is a share and not a
    ///    switch.
    /// 4. The column's relief at a share of ONE is the RAW octave sum, and at a share of ZERO it is
    ///    the noise factor's own — so the factor really is the GREATER of the two.
    ///
    /// RED before 2026-09-21: `slope_share` did not exist.
    ///
    /// **Example.** The column under the pilot's boots sits where the solve climbs 2 000 m across one
    /// 8 192 m macro node. That is twice the home planet's slope reference, so the share reads ONE
    /// and the hillside under the boots is a hillside.
    #[test]
    fn the_macro_slopes_share_reads_zero_on_a_plain_and_one_on_a_ramp() {
        let body = crate::home::home_planet();
        let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
        let charter = slope_charter(&body, &lattice, 0);
        // A column in the middle of a face, so every node of the widened stencil stands on it.
        let cell = (lattice.edge as i32 >> 1) * lattice.cells_per_node as i32;
        let (node_i, _) = node_and_fraction(&lattice, 0, cell);
        let face = Face::PosZ;
        let share_of = |rise_m: i32| {
            let ramp = Ramp {
                lattice,
                i0: node_i as i32,
                rise_m,
            };
            slope_share(&charter, &lattice, &ramp, face, 0, cell, cell).expect("a full field")
        };
        // (1) A FLAT field: no slope, no share. The ramp states heights alone: no water, no
        // province and no side, so the share reads nothing else of it.
        assert_eq!(share_of(0), Gi::ZERO, "a plain keeps the noise factor");
        let flat = Ramp {
            lattice,
            i0: node_i as i32,
            rise_m: 0,
        };
        assert_eq!(flat.water_facies(0), None);
        assert_eq!(flat.province(0), None);
        assert_eq!(flat.water_side(0), None);
        // (2) A STEEP field: 2 000 m over one 8 192 m node is 0.244, well over the reference.
        let one = Gi::new(1 << NOISE_BITS);
        assert_eq!(share_of(2_000), one, "a range is fully rough");
        // (3) A field BETWEEN: the share is the ratio of the two slopes.
        let node_m = lattice.node_m();
        let reference = crate::units::share_of_q28(body.slope_ref());
        let middle = share_of(512);
        let want = 512.0 / node_m / reference;
        assert!(want > 0.0, "the middle field slopes");
        assert!(want < 1.0, "the middle field is a middle one");
        let got = crate::units::share_of_q28(middle);
        assert!(
            (got - want).abs() < 1.0e-3,
            "the share {got} against the ratio {want}"
        );
        // (4) THE FACTOR IS THE GREATER. At a share of ONE the column's relief is the RAW sum of the
        // octaves the field did not replace; at ZERO it is the noise factor's own, and the two are
        // not the same column.
        let plan = body.plan_charter(0, face);
        let first = body.first_fine();
        let count = body.octaves_at(0).len();
        let surface = vd_recipe::plan::column_surface_from(
            &plan,
            i32::from(face.index()),
            cell,
            cell,
            &vd_recipe::plan::FieldRead {
                z: Gi::ZERO,
                first,
                slope_share: one,
            },
        );
        let raw = vd_recipe::height::relief(&body.octave_table()[first..count], surface.dir);
        assert_eq!(
            vd_recipe::height::relief_of_table_from(
                body.octave_table(),
                first,
                count,
                surface.dir,
                body.roughness(),
                one,
            ),
            raw,
            "a share of one keeps every fine octave whole"
        );
        assert_ne!(
            vd_recipe::height::relief_of_table_from(
                body.octave_table(),
                first,
                count,
                surface.dir,
                body.roughness(),
                Gi::ZERO,
            ),
            raw,
            "and a share of zero does not"
        );
    }

    /// ★ A COLUMN'S SLOPE SHARE NEEDS THE WIDER STENCIL, and refuses without it (2026-09-21): a
    /// field that holds the column's own sixteen nodes but not the ring around them answers `None`,
    /// so the chunk is not built and the coarser rung stands (ruling F9) — never a column built at
    /// half a rule. Each of the FOUR reads is driven on its own: one node is taken out of the block
    /// that only that read reaches, so a share that forgot a side would go green here.
    #[test]
    fn a_column_without_the_wider_stencil_refuses_its_share() {
        let body = crate::home::home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        let charter = slope_charter(&body, &lattice, 0);
        let cell = (lattice.edge as i32 >> 1) * lattice.cells_per_node as i32;
        let (ni, _) = node_and_fraction(&lattice, 0, cell);
        let (nj, _) = node_and_fraction(&lattice, 0, cell);
        // The whole block the column and its four neighbours read: six nodes a side.
        let block = |skip: Option<(i64, i64)>| {
            let mut rows = SparseRows::default();
            for row in -2..=3i64 {
                for col in -2..=3i64 {
                    if skip == Some((col, row)) {
                        continue;
                    }
                    rows.0.insert(
                        node_at(&lattice, Face::PosZ, ni + col, nj + row),
                        (0, 0, 0, 0),
                    );
                }
            }
            rows
        };
        let share =
            |rows: &SparseRows| slope_share(&charter, &lattice, rows, Face::PosZ, 0, cell, cell);
        // The whole block answers, and a level field is a share of zero.
        assert_eq!(share(&block(None)), Some(Gi::ZERO));
        // Only the column's own sixteen nodes: the ring is gone and the share refuses.
        let mut own = SparseRows::default();
        for row in -1..=2i64 {
            for col in -1..=2i64 {
                own.0.insert(
                    node_at(&lattice, Face::PosZ, ni + col, nj + row),
                    (0, 0, 0, 0),
                );
            }
        }
        assert!(
            sample_z(&lattice, &own, Face::PosZ, 0, cell, cell).is_some(),
            "the column's own read stands"
        );
        assert_eq!(share(&own), None, "the share refuses without the ring");
        // Each side on its own: a node only one of the four reads reaches.
        assert_eq!(share(&block(Some((3, 0)))), None, "the +i read");
        assert_eq!(share(&block(Some((-2, 0)))), None, "the −i read");
        assert_eq!(share(&block(Some((0, 3)))), None, "the +j read");
        assert_eq!(share(&block(Some((0, -2)))), None, "the −j read");
    }

    /// ★ THE SLOPE STEP IS ONE MACRO NODE, AND ONE CELL WHERE A CELL IS WIDER (2026-09-21). On the
    /// home planet a node is 8 192 rung-0 cells, so rung 13's own cell is exactly one node and
    /// rung 14's is two: the step clamps to ONE CELL there, and the reciprocal is taken over the
    /// step the read really makes — halved, because the step doubled.
    ///
    /// RED before 2026-09-21: a reciprocal taken over the node rather than the step would read a
    /// slope twice the real one on every rung above 13, and the far view would be rougher than the
    /// ground under the boots.
    #[test]
    fn the_slope_step_is_one_node_and_clamps_to_one_cell() {
        let body = crate::home::home_planet();
        let lattice = MacroLattice::of(&body).expect("a lattice");
        assert_eq!(lattice.cells_per_node, 8_192);
        // Rung 0: a whole node of cells, and the reciprocal of twice 8 192 m in gap steps.
        let at0 = slope_charter(&body, &lattice, 0);
        assert_eq!(at0.step, 8_192);
        assert_eq!(at0.inv, Gi::new(1 << 19), "2⁴⁰ / (2 × 8 192 × 128)");
        // The body's own two words ride the charter, so no column asks the body for them.
        assert_eq!(
            (at0.reference, at0.recip),
            (body.slope_ref(), body.slope_ref_recip())
        );
        // Rung 13: one cell, which IS one node — the same reciprocal.
        let at13 = slope_charter(&body, &lattice, 13);
        assert_eq!((at13.step, at13.inv), (1, at0.inv));
        // Rung 14: one cell, which is TWO nodes — half the reciprocal.
        let at14 = slope_charter(&body, &lattice, 14);
        assert_eq!((at14.step, at14.inv), (1, Gi::new(1 << 18)));
        // A chunk with no field carries the empty charter and measures nothing.
        assert_eq!(SlopeCharter::NONE.step, 1);
        assert_eq!(SlopeCharter::NONE.inv, Gi::ZERO);
    }
}
