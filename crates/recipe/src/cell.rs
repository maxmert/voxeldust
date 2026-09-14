//! ★ THE CELL — what one cell of the world IS, on words alone (ruling F7; the GPU step G1 of
//! `slice_08_integer_recipe_design.md` §2). Given a column's direction and surface, a cell's radius,
//! the cavern field's value there and the tube carvers that can reach it, this module answers with
//! ONE WORD: the cell's substance code in the low byte and its gap byte above it. The server's
//! collision, the client's CPU worker and the client's GPU all call the SAME function, so a hill a
//! player walks on and a hill the card draws are the same hill by construction, not by agreement.
//!
//! **The kernel names no substance.** A substance's code arrives in the charter ([`CellCharter`]),
//! drawn once per body by the host that owns the naming (`vd_terrain::strata`). So the recipe never
//! learns what "granite" is, and a host may renumber its own substances without moving a single bit
//! of this arithmetic.
//!
//! **The inputs are plain words.** No heap, no CONST table read at run time, no `/` and no `%`, and
//! no loop whose value is read after it ends (the GPU compiler's rules, §1b of the design, two of
//! them MEASURED in this module's own kernels on 2026-09-13): a charter of words, an array of tube
//! carvers, and one cell's own numbers. That is what lets the same source compile for a shader,
//! where there is no allocator and a division is a compiler's argument.
//!
//! **The two byte conventions** (Format B, unchanged by this module):
//! - the GAP is the signed distance to the surface in 1/128 of a cell — negative inside the rock,
//!   positive in the air, `0` exactly on the surface (which is air), clamped to `[−128, 127]`;
//! - the SUBSTANCE is the host's own code, and a hollow cell reads the charter's air code.
//!
//! **Example.** The pilot's boots stand in cell (face 2, 1181, 77, radial 88 of the metre rung). The
//! column says the ground is at 815 406 723 gap steps; the cell's centre is at 815 406 720; the
//! cavern field reads under its threshold and no tube passes. The kernel answers "dirt, gap −3": the
//! boots are three 128ths of a metre inside the ground, and the shard and the card agree on it.

use crate::bend::DIR_BITS;
use crate::gi::Gi;
use crate::height::GAP_STEPS_PER_CELL;
use crate::noise::NOISE_BITS;
use crate::root::isqrt;

/// The fraction bits a LENGTH carries on the recipe's path: the noise's own, so a surface and an
/// octave sum add without a shift.
pub const LENGTH_BITS: u32 = NOISE_BITS;

/// The shift from gap steps to whole metres: 128 steps a metre, so the shift is seven.
pub const STEP_SHIFT: u32 = GAP_STEPS_PER_CELL.trailing_zeros();

/// How many biomes a column can hold. A POWER OF TWO, so the charter's row for a biome is a mask
/// and never a bounds branch.
pub const BIOMES: usize = 4;
/// The mask that picks a biome's row.
pub const BIOME_MASK: i64 = BIOMES as i64 - 1;
const _: () = assert!(BIOMES.is_power_of_two());

/// One byte, as a word.
const BYTE: Gi = Gi::new(0xFF);

/// The shift of the TOPSOIL's code inside a charter row.
pub const ROW_TOPSOIL: u32 = 0;
/// The shift of the SUBSOIL's code inside a charter row.
pub const ROW_SUBSOIL: u32 = 8;
/// The shift of the SEDIMENT's code inside a charter row.
pub const ROW_SEDIMENT: u32 = 16;

/// The gap byte's ends: a full cell of air clamps to the top code, one step short of a whole cell,
/// which the extractor never needs; a full cell of rock clamps to the bottom code.
pub const GAP_TOP: Gi = Gi::new(127);
/// The gap byte's bottom code.
pub const GAP_BOTTOM: Gi = Gi::new(-128);

/// A radial layer that the cell rule EVALUATES, cell by cell.
pub const LAYER_EVALUATED: i64 = 0;
/// A radial layer under the body's band: bedrock at the bottom code, without a lookup.
pub const LAYER_BELOW: i64 = 1;
/// A radial layer over the body's band: the fluid at its radius, at the top code.
pub const LAYER_ABOVE: i64 = 2;

/// The fraction bits a tube's projection parameter carries.
pub const TUBE_T_BITS: u32 = 30;
/// The fraction bits of a tube's squared-length reciprocal.
pub const TUBE_RECIP_BITS: u32 = 62;

/// The greater of two words, by the comparison — the recipe has no `max` (the float fence's rule,
/// kept for the integer path so one reading order holds everywhere). A tie keeps the LEFT word.
#[must_use]
pub fn greater(a: Gi, b: Gi) -> Gi {
    if b > a { b } else { a }
}

/// The lesser of two words, by the comparison. A tie keeps the LEFT word.
#[must_use]
pub fn lesser(a: Gi, b: Gi) -> Gi {
    if b < a { b } else { a }
}

/// A point in the body's frame, in WHOLE gap steps: a direction at the bend's fraction bits times a
/// radius in whole gap steps, through the two-word product (the product passes 63 bits at a planet's
/// radius, so a plain multiply would wrap). What the carvers read.
#[must_use]
pub fn point_at(dir: [Gi; 3], radius_steps: Gi) -> [Gi; 3] {
    [
        dir[0].mul_shr(radius_steps, DIR_BITS),
        dir[1].mul_shr(radius_steps, DIR_BITS),
        dir[2].mul_shr(radius_steps, DIR_BITS),
    ]
}

/// The trilinear blend of eight node values `v[(c·2 + b)·2 + a]` at weights `t`, at
/// [`LENGTH_BITS`]: the one arithmetic the cavern lattice's cell pass, the halo and the GPU share.
#[must_use]
pub fn trilinear8(v: [Gi; 8], t: [Gi; 3]) -> Gi {
    let x00 = blend(v[0], v[1], t[0]);
    let x10 = blend(v[2], v[3], t[0]);
    let x01 = blend(v[4], v[5], t[0]);
    let x11 = blend(v[6], v[7], t[0]);
    blend(blend(x00, x10, t[1]), blend(x01, x11, t[1]), t[2])
}

/// `p + w(q − p)` at [`LENGTH_BITS`]: one multiply, one shift, one add, in that order.
fn blend(p: Gi, q: Gi, w: Gi) -> Gi {
    p + ((w * (q - p)) >> LENGTH_BITS)
}

/// A gap already counted in GAP STEPS (1/128 of a cell) clamped to the signed byte's range, still a
/// word. The floor happened in the one shift that made the steps, so nothing rounds twice.
#[must_use]
pub fn gap_code(gap_steps: Gi) -> Gi {
    if gap_steps < GAP_BOTTOM {
        GAP_BOTTOM
    } else if gap_steps > GAP_TOP {
        GAP_TOP
    } else {
        gap_steps
    }
}

/// ★ THE CELL'S ONE WORD: the substance code in the low byte, the gap byte in the next. A shader
/// writes this word and a host reads it back with [`stratum_of_word`] and [`gap_of_word`], so the
/// two hosts move one `u32` per cell and never a struct whose layout they must agree on.
#[must_use]
pub fn pack(stratum: Gi, gap: Gi) -> u32 {
    let s = (stratum.raw() as u32) & 0xFF;
    let g = (gap.raw() as u32) & 0xFF;
    s | (g << 8)
}

/// The substance code of a cell word.
#[must_use]
pub const fn stratum_of_word(word: u32) -> u8 {
    (word & 0xFF) as u8
}

/// The gap byte of a cell word.
#[must_use]
pub const fn gap_of_word(word: u32) -> i8 {
    ((word >> 8) & 0xFF) as u8 as i8
}

/// A TUBE CARVER: a segment in the body's frame, in GAP STEPS, with a radius and the reciprocal of
/// its own squared length (drawn once with the tube by the host, so no cell divides). `repr(C)`:
/// eight words in this order, so a GPU reads a list of them in place from a buffer the CPU filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Tube {
    pub start: [Gi; 3],
    pub end: [Gi; 3],
    pub radius_steps: Gi,
    /// `floor(2^TUBE_RECIP_BITS / |end − start|²)`. A segment of no length reads one, and then the
    /// projection lands on the start, which is the right answer for a point.
    pub inv_len2: Gi,
}

impl Tube {
    /// The distance from a point to the segment, in gap steps: the SQUARED distance, then one
    /// integer square root.
    #[must_use]
    pub fn distance_steps(&self, point: [Gi; 3]) -> Gi {
        Gi::new(isqrt(self.distance2_steps(point)) as i64)
    }

    /// The SQUARED distance from a point to the segment, in squared gap steps: the projection
    /// clamped to the segment by the stored reciprocal, and no root at all. A caller that only
    /// asks "is this point inside the carver?" compares this with the radius SQUARED and pays no
    /// root (MEASURED, 2026-09-13, `terrain-cost`: the root written out ADDED 73 % to the
    /// cave-dense chunk at the eight-metre rung — 42.7 ms of cell pass to 73.7 — because a box
    /// there is 496 m across, holds many carvers, and almost every cell stands outside every one
    /// of them. With this guard that chunk is 7.8 ms, and no byte of the world moved).
    ///
    /// ★ THE THREE AXES ARE WRITTEN OUT, not walked by an index loop (step G1, 2026-09-13). Three
    /// terms are three terms; a loop over them buys nothing, and on the card it makes the two
    /// offset triples LOCAL ARRAYS the shader must index at run time — the shape the kernel rules
    /// name (§1b) and the shape whose loop tail the SPIR-V back end mis-spelled in [`isqrt`].
    #[must_use]
    pub fn distance2_steps(&self, point: [Gi; 3]) -> u64 {
        let (ab0, ab1, ab2) = (
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
            self.end[2] - self.start[2],
        );
        let (ap0, ap1, ap2) = (
            point[0] - self.start[0],
            point[1] - self.start[1],
            point[2] - self.start[2],
        );
        let dot = ap0 * ab0 + ap1 * ab1 + ap2 * ab2;
        // t = dot / |ab|², at TUBE_T_BITS, clamped to the segment.
        let t = dot.mul_shr(self.inv_len2, TUBE_RECIP_BITS - TUBE_T_BITS);
        let one = Gi::ONE << TUBE_T_BITS;
        let t = if t < Gi::ZERO {
            Gi::ZERO
        } else if t > one {
            one
        } else {
            t
        };
        let d0 = (ap0 - ((ab0 * t) >> TUBE_T_BITS)).unsigned_abs();
        let d1 = (ap1 - ((ab1 * t) >> TUBE_T_BITS)).unsigned_abs();
        let d2 = (ap2 - ((ab2 * t) >> TUBE_T_BITS)).unsigned_abs();
        d0.wrapping_mul(d0)
            .wrapping_add(d1.wrapping_mul(d1))
            .wrapping_add(d2.wrapping_mul(d2))
    }
}

/// The hollow a list of tubes opens at a point, in GAP STEPS: the greatest of `radius − distance`
/// over the list, never below zero. An INDEX LOOP, never an iterator: the GPU compiler refuses an
/// iterator's pointer arithmetic. An EMPTY list opens nothing, which is what most cells meet.
#[must_use]
pub fn tube_hollow_steps(tubes: &[Tube], point_steps: [Gi; 3]) -> Gi {
    let mut best = Gi::ZERO;
    let mut t = 0;
    while t < tubes.len() {
        // ★ THE ROOT IS PAID ONLY INSIDE A CARVER (MEASURED, 2026-09-13): a point is inside one
        // exactly where its SQUARED distance is under the radius SQUARED (the root floors, so
        // `isqrt(d²) < r` and `d² < r²` are the same statement), and a carver the point stands
        // outside opens nothing. A box at the eight-metre rung holds many carvers and almost every
        // cell is outside every one, so the guard takes 73 % off that chunk and changes no byte.
        let radius = tubes[t].radius_steps;
        let d2 = tubes[t].distance2_steps(point_steps);
        let open = if d2 < (radius * radius).unsigned_abs() {
            radius - Gi::new(isqrt(d2) as i64)
        } else {
            Gi::ZERO
        };
        // ★ THE GREATEST, WRITTEN AS AN ADDITION (MEASURED, 2026-09-13, step G1): add only what
        // this carver opens ABOVE what the list has opened so far. Written as
        // `best = greater(best, open)` the card read the hollow back as ZERO — 87 of the probe's
        // 200 points — because rust-gpu carries an accumulator that a BRANCH assigns out of a loop
        // in a spilled word written before the step (`crate::root::isqrt` names the same fault).
        // An accumulator the loop only ADDS to leaves in a phi, which is the shape the octave sum
        // already proves on 3 936 256 columns.
        best += if open > best { open - best } else { Gi::ZERO };
        t += 1;
    }
    best
}

/// ★ THE CHARTER OF A BODY AT A RUNG — everything the cell kernel reads that is not the cell's own
/// numbers, as words. The host draws it ONCE per body per rung and hands it to every cell, on the
/// CPU by reference and on the GPU as one small buffer. `repr(C)`, so the two hosts read the same
/// bytes in the same order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CellCharter {
    /// The sea's radius, in gap steps at [`LENGTH_BITS`]: a cell that is not rock and stands under
    /// it is water, over it air.
    pub sea_radius: Gi,
    /// The depth band caves live in, in gap steps at [`LENGTH_BITS`] under the surface.
    pub cave_min: Gi,
    pub cave_max: Gi,
    /// The cavern field's threshold at the noise's fraction bits, and how many GAP STEPS of hollow
    /// one unit of the field above it opens.
    pub cavern_threshold: Gi,
    pub cavern_scale_steps: Gi,
    /// ONE where anything carves at this rung, ZERO where nothing does (a coarse rung draws the
    /// hill without its caves).
    pub carve_any: Gi,
    /// The rung: a cell is `2^rung` metres across, so the rung is the shift from metres of gap step
    /// to cells of gap step.
    pub rung: Gi,
    /// The topsoil's last metre, the subsoil's last metre and the deepest metre the strata change
    /// at, each a WHOLE number of metres under the surface.
    pub topsoil_m: Gi,
    pub subsoil_end_m: Gi,
    pub strata_end_m: Gi,
    /// The codes the HOST names its substances by: the atmosphere, the sea, and this body's
    /// bedrock. The recipe never learns what they mean.
    pub air_code: Gi,
    pub water_code: Gi,
    pub bedrock_code: Gi,
    /// How many cells a BOX is across — a chunk's own 62, or 64 with its halo. A GPU shell reads it
    /// to find a cell's column and its radial layer; the CPU's own cell pass walks its loops and
    /// never reads it.
    pub box_edge: Gi,
    /// One packed row per biome: the topsoil's code at [`ROW_TOPSOIL`], the subsoil's at
    /// [`ROW_SUBSOIL`], the sediment's at [`ROW_SEDIMENT`] ([`strata_row`] packs one).
    pub strata: [Gi; BIOMES],
}

/// One biome's row of the charter's strata table: three substance codes in three bytes.
#[must_use]
pub fn strata_row(topsoil: Gi, subsoil: Gi, sediment: Gi) -> Gi {
    ((topsoil & BYTE) << ROW_TOPSOIL)
        | ((subsoil & BYTE) << ROW_SUBSOIL)
        | ((sediment & BYTE) << ROW_SEDIMENT)
}

impl CellCharter {
    /// Whether a cell at `depth` under the surface can hold a cave at this rung.
    #[must_use]
    pub fn in_band(&self, depth: Gi) -> bool {
        (self.carve_any != Gi::ZERO) & (depth >= self.cave_min) & (depth <= self.cave_max)
    }

    /// THE FLUID at a radius: the sea under its own surface, the atmosphere over it.
    #[must_use]
    pub fn fluid_code(&self, r: Gi) -> Gi {
        if r < self.sea_radius {
            self.water_code
        } else {
            self.air_code
        }
    }

    /// The substance at `depth_m` WHOLE metres under the surface (0 is the surface cell) in a
    /// biome: the topsoil, then the subsoil, then the sediment, then the bedrock. The biome picks
    /// its row by a MASK, so no bound is tested and no arm can be missed.
    #[must_use]
    pub fn stratum_code(&self, biome: Gi, depth_m: Gi) -> Gi {
        let row = self.strata[(biome & Gi::new(BIOME_MASK)).raw() as usize];
        if depth_m < self.topsoil_m {
            (row >> ROW_TOPSOIL) & BYTE
        } else if depth_m < self.subsoil_end_m {
            (row >> ROW_SUBSOIL) & BYTE
        } else if depth_m < self.strata_end_m {
            (row >> ROW_SEDIMENT) & BYTE
        } else {
            self.bedrock_code
        }
    }

    /// The hollow the cavern field opens for a field value, in GAP STEPS of METRE: zero outside a
    /// room, positive inside.
    #[must_use]
    pub fn cavern_hollow_steps(&self, value: Gi) -> Gi {
        let over = value - self.cavern_threshold;
        if over > Gi::ZERO {
            (over * self.cavern_scale_steps) >> NOISE_BITS
        } else {
            Gi::ZERO
        }
    }
}

/// What a cell's column and radial layer state about it: the inputs of the per-cell tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CellAt {
    /// The column's direction, at the bend's fraction bits.
    pub dir: [Gi; 3],
    /// The column's surface radius, in gap steps at [`LENGTH_BITS`].
    pub h: Gi,
    /// The column's biome, as an index the charter's rows are ordered by.
    pub biome: Gi,
    /// The cell centre's radius in WHOLE gap steps — exact.
    pub r_steps: Gi,
}

/// ★ THE CELL KERNEL — one cell's substance and gap, as one word.
///
/// The whole tail is integers (ruling F7): one subtraction for the gap, ONE shift by
/// `rung + LENGTH_BITS` that divides by the cell's width and floors in the same step, and the clamp
/// to the byte. A hollow comes in as gap steps of METRE and shifts by the rung alone to become gap
/// steps of CELL. `value` is the cavern field at this cell (ZERO where nothing carves) and `tubes`
/// the carvers that can reach it (empty for most cells).
///
/// **Example.** Forty metres under a cliff the cavern field reads over its threshold: the kernel
/// turns that into a metre of hollow, the greater of the rock's gap and the hollow wins, and the
/// cell reads "air, gap +128 steps of metre" — a room in the rock, the same room the shard's
/// collider finds.
#[must_use]
pub fn cell_word(charter: &CellCharter, at: &CellAt, value: Gi, tubes: &[Tube]) -> u32 {
    let rung = charter.rung.raw() as u32;
    let r = at.r_steps << LENGTH_BITS;
    let depth = at.h - r;
    // The rock's gap in gap steps OF A CELL: `(r − h) / cell`, floored once. A cell whose centre is
    // exactly on the surface reads 0, which is air.
    let mut gap_steps = (r - at.h) >> (rung + LENGTH_BITS);
    let mut stratum = if depth <= Gi::ZERO {
        charter.fluid_code(r)
    } else {
        charter.stratum_code(at.biome, depth >> (LENGTH_BITS + STEP_SHIFT))
    };
    if charter.in_band(depth) {
        let p = point_at(at.dir, at.r_steps);
        let hollow_steps = greater(
            charter.cavern_hollow_steps(value),
            tube_hollow_steps(tubes, p),
        );
        if hollow_steps > Gi::ZERO {
            // A hollow is never negative, so the greater is in air: the cell is hollow. The hollow
            // is metres of gap step; the rung's shift makes it cells of gap step.
            gap_steps = greater(gap_steps, hollow_steps >> rung);
            stratum = charter.air_code;
        }
    }
    pack(stratum, gap_code(gap_steps))
}

/// The cell of a layer more than a cell ABOVE every surface of its column: the fluid at its radius,
/// at the top code. `r` is the cell centre's radius in gap steps at [`LENGTH_BITS`].
#[must_use]
pub fn above_cell_word(charter: &CellCharter, r: Gi) -> u32 {
    pack(charter.fluid_code(r), GAP_TOP)
}

/// The cell of a layer more than a cell BELOW every surface, stratum and cave: bedrock at the
/// bottom code.
#[must_use]
pub fn below_cell_word(charter: &CellCharter) -> u32 {
    pack(charter.bedrock_code, GAP_BOTTOM)
}

/// ONE RADIAL LAYER of a box, as a GPU shell reads it: the layer's cell radius, which rule it
/// follows, and where the cavern lattice's radial node sits for it. `repr(C)`, four words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Layer {
    /// The cell centre's radius in WHOLE gap steps — exact.
    pub r_steps: Gi,
    /// [`LAYER_EVALUATED`], [`LAYER_BELOW`] or [`LAYER_ABOVE`].
    pub rule: Gi,
    /// The radial node index inside the column's own cavern lattice.
    pub node: Gi,
    /// The weight between that node and the next, at [`LENGTH_BITS`].
    pub weight: Gi,
}

/// ONE COLUMN of a box, as a GPU shell reads it: what the column pass said about it, and where its
/// cavern lattice sits in the flat node buffer. `repr(C)`, thirteen words.
///
/// The host resolves the lattice's TOPOLOGY — which face a column belongs to across a seam, and
/// whether it is a corner phantom with no lattice at all — and states the answer here as a base, two
/// node indices, two strides and two weights. The kernel then does arithmetic only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Column {
    /// The column's direction, at the bend's fraction bits.
    pub dir: [Gi; 3],
    /// The column's surface radius, in gap steps at [`LENGTH_BITS`].
    pub h: Gi,
    /// The column's biome.
    pub biome: Gi,
    /// Where this column's cavern lattice starts in the node buffer.
    pub base: Gi,
    /// The node indices along the lattice's two face axes.
    pub na: Gi,
    pub nb: Gi,
    /// The lattice's two face strides.
    pub d0: Gi,
    pub d1: Gi,
    /// The weights inside the node cube along the two face axes, at [`LENGTH_BITS`].
    pub wa: Gi,
    pub wb: Gi,
    /// ONE where this column has a cavern lattice at all; ZERO for a corner phantom, and ZERO at a
    /// rung where the caverns do not carve.
    pub has: Gi,
}

/// The cavern field at one cell, read from the flat node buffer: eight nodes around it, blended.
/// A column with no lattice reads ZERO, which opens no room — the same answer the host's own cell
/// pass writes for a corner phantom.
#[must_use]
pub fn cavern_at(column: &Column, layer: &Layer, nodes: &[Gi]) -> Gi {
    if column.has == Gi::ZERO {
        return Gi::ZERO;
    }
    let base = column.base.raw() as usize;
    let d0 = column.d0.raw() as usize;
    let d1 = column.d1.raw() as usize;
    let na = column.na.raw() as usize;
    let nb = column.nb.raw() as usize;
    let nc = layer.node.raw() as usize;
    trilinear8(
        [
            node_at(nodes, base, d0, d1, na, nb, nc),
            node_at(nodes, base, d0, d1, na + 1, nb, nc),
            node_at(nodes, base, d0, d1, na, nb + 1, nc),
            node_at(nodes, base, d0, d1, na + 1, nb + 1, nc),
            node_at(nodes, base, d0, d1, na, nb, nc + 1),
            node_at(nodes, base, d0, d1, na + 1, nb, nc + 1),
            node_at(nodes, base, d0, d1, na, nb + 1, nc + 1),
            node_at(nodes, base, d0, d1, na + 1, nb + 1, nc + 1),
        ],
        [column.wa, column.wb, layer.weight],
    )
}

/// One node of a flat lattice buffer. Written out rather than closed over: a closure's capture is
/// one more shape the GPU compiler has to agree with, and this one buys nothing.
fn node_at(nodes: &[Gi], base: usize, d0: usize, d1: usize, x: usize, y: usize, z: usize) -> Gi {
    nodes[base + (z * d1 + y) * d0 + x]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A charter a test can read: the sea at 1 000 steps, caves between 10 and 100 steps of depth,
    /// four biomes whose rows are easy to name apart.
    fn charter() -> CellCharter {
        CellCharter {
            sea_radius: Gi::new(1_000) << LENGTH_BITS,
            cave_min: Gi::new(10) << LENGTH_BITS,
            cave_max: Gi::new(100) << LENGTH_BITS,
            cavern_threshold: Gi::new(1 << (NOISE_BITS - 1)),
            cavern_scale_steps: Gi::new(20 * GAP_STEPS_PER_CELL),
            carve_any: Gi::ONE,
            rung: Gi::ZERO,
            topsoil_m: Gi::new(2),
            subsoil_end_m: Gi::new(7),
            strata_end_m: Gi::new(47),
            air_code: Gi::ZERO,
            water_code: Gi::ONE,
            bedrock_code: Gi::new(13),
            box_edge: Gi::new(64),
            strata: [
                strata_row(Gi::new(4), Gi::new(9), Gi::new(10)),
                strata_row(Gi::new(5), Gi::new(6), Gi::new(10)),
                strata_row(Gi::new(2), Gi::new(8), Gi::new(10)),
                strata_row(Gi::new(7), Gi::new(13), Gi::new(10)),
            ],
        }
    }

    #[test]
    fn the_greater_and_the_lesser_are_comparisons_and_keep_the_left_word_on_a_tie() {
        assert_eq!(greater(Gi::new(3), Gi::new(7)), Gi::new(7));
        assert_eq!(greater(Gi::new(7), Gi::new(3)), Gi::new(7));
        assert_eq!(greater(Gi::new(3), Gi::new(3)), Gi::new(3));
        assert_eq!(lesser(Gi::new(3), Gi::new(7)), Gi::new(3));
        assert_eq!(lesser(Gi::new(7), Gi::new(3)), Gi::new(3));
        assert_eq!(lesser(Gi::new(3), Gi::new(3)), Gi::new(3));
    }

    #[test]
    fn a_point_is_the_direction_times_the_radius_through_the_two_word_product() {
        let one = Gi::ONE << DIR_BITS;
        let r = Gi::new(815_405_260);
        assert_eq!(
            point_at([one, Gi::ZERO, Gi::ZERO], r),
            [r, Gi::ZERO, Gi::ZERO]
        );
        // Half the direction reads half the radius, and the product passes 63 bits on the way.
        let half = one >> 1;
        assert_eq!(point_at([half, half, Gi::ZERO], r)[0], Gi::new(407_702_630));
        // A negative component keeps its sign.
        assert_eq!(
            point_at([Gi::ZERO - one, Gi::ZERO, Gi::ZERO], r)[0],
            Gi::ZERO - r
        );
    }

    #[test]
    fn the_trilinear_blend_reads_its_corners_and_its_centre() {
        let one = Gi::ONE << LENGTH_BITS;
        let v = [
            Gi::new(0),
            Gi::new(100),
            Gi::new(200),
            Gi::new(300),
            Gi::new(400),
            Gi::new(500),
            Gi::new(600),
            Gi::new(700),
        ];
        assert_eq!(trilinear8(v, [Gi::ZERO; 3]), Gi::ZERO);
        assert_eq!(trilinear8(v, [one, one, one]), Gi::new(700));
        assert_eq!(trilinear8(v, [one, Gi::ZERO, Gi::ZERO]), Gi::new(100));
        assert_eq!(trilinear8(v, [Gi::ZERO, one, Gi::ZERO]), Gi::new(200));
        assert_eq!(trilinear8(v, [Gi::ZERO, Gi::ZERO, one]), Gi::new(400));
        // The centre is the mean of the eight.
        let half = one >> 1;
        assert_eq!(trilinear8(v, [half, half, half]), Gi::new(350));
    }

    #[test]
    fn the_gap_clamps_to_the_signed_byte_and_the_word_carries_both_halves() {
        assert_eq!(gap_code(Gi::ZERO), Gi::ZERO);
        assert_eq!(gap_code(Gi::new(-39)), Gi::new(-39));
        assert_eq!(gap_code(Gi::new(64)), Gi::new(64));
        assert_eq!(
            gap_code(Gi::new(128)),
            GAP_TOP,
            "a full cell is one step short"
        );
        assert_eq!(gap_code(Gi::new(-128)), GAP_BOTTOM);
        assert_eq!(gap_code(Gi::new(-5_000)), GAP_BOTTOM);
        assert_eq!(gap_code(Gi::new(9 * 128)), GAP_TOP);
        // The word: the substance below, the gap above, and back again.
        let w = pack(Gi::new(12), Gi::new(-39));
        assert_eq!(stratum_of_word(w), 12);
        assert_eq!(gap_of_word(w), -39);
        let w = pack(Gi::new(255), GAP_TOP);
        assert_eq!(stratum_of_word(w), 255);
        assert_eq!(gap_of_word(w), 127);
        assert_eq!(gap_of_word(pack(Gi::ZERO, GAP_BOTTOM)), -128);
    }

    #[test]
    fn a_tube_measures_its_distance_and_clamps_the_projection_to_its_ends() {
        // A tube along +x from 0 to 100 steps, three steps wide.
        let tube = Tube {
            start: [Gi::ZERO; 3],
            end: [Gi::new(100), Gi::ZERO, Gi::ZERO],
            radius_steps: Gi::new(3),
            inv_len2: Gi::new(crate::root::recip_pow2(100 * 100, TUBE_RECIP_BITS) as i64),
        };
        // Beside the middle: the perpendicular distance.
        assert_eq!(
            tube.distance_steps([Gi::new(50), Gi::new(7), Gi::ZERO]),
            Gi::new(7)
        );
        // Past the far end: the distance to the end, because the projection clamps to one.
        assert_eq!(
            tube.distance_steps([Gi::new(140), Gi::ZERO, Gi::ZERO]),
            Gi::new(40)
        );
        // Before the start: the distance to the start, because the projection clamps to zero.
        assert_eq!(
            tube.distance_steps([Gi::new(-9), Gi::ZERO, Gi::ZERO]),
            Gi::new(9)
        );
        // A tube of NO length is a point: the projection lands on the start.
        let point = Tube {
            start: [Gi::new(5), Gi::ZERO, Gi::ZERO],
            end: [Gi::new(5), Gi::ZERO, Gi::ZERO],
            radius_steps: Gi::new(1),
            inv_len2: Gi::new(crate::root::recip_pow2(0, TUBE_RECIP_BITS) as i64),
        };
        assert_eq!(
            point.distance_steps([Gi::new(9), Gi::ZERO, Gi::ZERO]),
            Gi::new(4)
        );
        // The hollow: the greatest of `radius − distance`, never below zero, and nothing at all
        // from an empty list.
        assert_eq!(
            tube_hollow_steps(&[], [Gi::ZERO, Gi::ZERO, Gi::ZERO]),
            Gi::ZERO
        );
        assert_eq!(
            tube_hollow_steps(&[tube], [Gi::new(50), Gi::new(1), Gi::ZERO]),
            Gi::new(2)
        );
        assert_eq!(
            tube_hollow_steps(&[tube], [Gi::new(50), Gi::new(90), Gi::ZERO]),
            Gi::ZERO,
            "far from the tube nothing is hollow"
        );
        // THE GREATEST over the list, whichever way round the list is read. (The long tube's own
        // hollow at the middle reads 2, not 3: the stored reciprocal floors, so the projection lands
        // one step short of the centre — the same one step on every host, which is the point.)
        let fat = Tube {
            start: [Gi::new(50), Gi::ZERO, Gi::ZERO],
            end: [Gi::new(50), Gi::ZERO, Gi::ZERO],
            radius_steps: Gi::new(9),
            inv_len2: Gi::new(crate::root::recip_pow2(0, TUBE_RECIP_BITS) as i64),
        };
        let middle = [Gi::new(50), Gi::ZERO, Gi::ZERO];
        assert_eq!(fat.distance_steps(middle), Gi::ZERO);
        assert_eq!(tube_hollow_steps(&[tube], middle), Gi::new(2));
        assert_eq!(tube_hollow_steps(&[point, tube], middle), Gi::new(2));
        assert_eq!(tube_hollow_steps(&[tube, fat], middle), Gi::new(9));
        assert_eq!(tube_hollow_steps(&[fat, tube], middle), Gi::new(9));
        assert_eq!(Tube::default().radius_steps, Gi::ZERO);
    }

    #[test]
    fn the_charter_reads_its_band_its_fluid_its_strata_and_its_cavern() {
        let c = charter();
        // A whole number of gap steps, at the length format's fraction bits.
        let q = |v: i64| Gi::new(v) << LENGTH_BITS;
        // The band: inside, on both ends, and outside.
        assert!(c.in_band(q(10)));
        assert!(c.in_band(q(100)));
        assert!(c.in_band(q(55)));
        assert!(!c.in_band(q(10) - Gi::ONE));
        assert!(!c.in_band(q(100) + Gi::ONE));
        let quiet = CellCharter {
            carve_any: Gi::ZERO,
            ..c
        };
        assert!(!quiet.in_band(q(55)), "a rung with no caves carves none");
        // The fluid.
        assert_eq!(c.fluid_code(q(1_000) - Gi::ONE), c.water_code);
        assert_eq!(c.fluid_code(q(1_000)), c.air_code);
        assert_eq!(c.fluid_code(q(1_000) + Gi::ONE), c.air_code);
        // The strata, in every biome, at every depth the table covers.
        for (biome, topsoil, subsoil) in [(0, 4, 9), (1, 5, 6), (2, 2, 8), (3, 7, 13)] {
            let b = Gi::new(biome);
            assert_eq!(c.stratum_code(b, Gi::ZERO), Gi::new(topsoil));
            assert_eq!(c.stratum_code(b, Gi::ONE), Gi::new(topsoil));
            assert_eq!(c.stratum_code(b, Gi::new(2)), Gi::new(subsoil));
            assert_eq!(c.stratum_code(b, Gi::new(6)), Gi::new(subsoil));
            assert_eq!(c.stratum_code(b, Gi::new(7)), Gi::new(10));
            assert_eq!(c.stratum_code(b, Gi::new(46)), Gi::new(10));
            assert_eq!(c.stratum_code(b, Gi::new(47)), c.bedrock_code);
            assert_eq!(c.stratum_code(b, Gi::new(5_000)), c.bedrock_code);
        }
        // A biome word past the table wraps by the mask, never out of the rows.
        assert_eq!(
            c.stratum_code(Gi::new(4), Gi::ZERO),
            c.stratum_code(Gi::ZERO, Gi::ZERO)
        );
        // The cavern's hollow: nothing under the threshold, and the scale above it.
        assert_eq!(c.cavern_hollow_steps(Gi::ZERO), Gi::ZERO);
        assert_eq!(c.cavern_hollow_steps(c.cavern_threshold), Gi::ZERO);
        let over = c.cavern_threshold + (Gi::ONE << (NOISE_BITS - 2));
        assert_eq!(c.cavern_hollow_steps(over), Gi::new(5 * GAP_STEPS_PER_CELL));
    }

    #[test]
    fn the_strata_row_packs_three_codes_into_three_bytes() {
        let row = strata_row(Gi::new(4), Gi::new(9), Gi::new(10));
        assert_eq!((row >> ROW_TOPSOIL) & BYTE, Gi::new(4));
        assert_eq!((row >> ROW_SUBSOIL) & BYTE, Gi::new(9));
        assert_eq!((row >> ROW_SEDIMENT) & BYTE, Gi::new(10));
        // A code past a byte keeps only its byte, so a row can never bleed into its neighbour.
        assert_eq!(
            (strata_row(Gi::new(0x1FF), Gi::ZERO, Gi::ZERO) >> ROW_SUBSOIL) & BYTE,
            Gi::ZERO
        );
    }

    /// One cell of rock, one of air, one of water, one hollowed by a cavern and one by a tube.
    #[test]
    fn the_cell_kernel_reads_the_rock_the_fluid_and_the_two_carvers() {
        let c = charter();
        let one = Gi::ONE << DIR_BITS;
        let step = Gi::ONE << LENGTH_BITS;
        let at = |r_steps: i64, h_steps: i64, biome: i64| CellAt {
            dir: [one, Gi::ZERO, Gi::ZERO],
            h: Gi::new(h_steps) * step,
            biome: Gi::new(biome),
            r_steps: Gi::new(r_steps),
        };
        // A cell whose centre is 39 steps under a surface at 2 000: rock, gap −39, and at 2 metres
        // of depth (39 steps is 0 whole metres) the topsoil of biome 1.
        let w = cell_word(&c, &at(2_000 - 39, 2_000, 1), Gi::ZERO, &[]);
        assert_eq!(gap_of_word(w), -39);
        assert_eq!(stratum_of_word(w), 5);
        // A cell 300 steps under the surface: two metres down, the subsoil.
        let w = cell_word(&c, &at(2_000 - 300, 2_000, 1), Gi::ZERO, &[]);
        assert_eq!(stratum_of_word(w), 6);
        assert_eq!(gap_of_word(w), -128, "past a cell of rock the byte clamps");
        // A cell exactly ON the surface reads 0 and is air (the fluid rule, over the sea).
        let w = cell_word(&c, &at(2_000, 2_000, 1), Gi::ZERO, &[]);
        assert_eq!(gap_of_word(w), 0);
        assert_eq!(stratum_of_word(w), 0);
        // A cell over the surface but UNDER the sea is water.
        let w = cell_word(&c, &at(900, 800, 1), Gi::ZERO, &[]);
        assert_eq!(stratum_of_word(w), 1);
        assert!(gap_of_word(w) > 0);
        // A CAVERN hollows a cell inside the band: the field over the threshold turns the cell to
        // air and lifts the gap.
        let deep = at(2_000 - 50, 2_000, 1);
        let value = c.cavern_threshold + (Gi::ONE << (NOISE_BITS - 2));
        let w = cell_word(&c, &deep, value, &[]);
        assert_eq!(stratum_of_word(w), 0, "a room is air");
        assert_eq!(
            gap_of_word(w),
            127,
            "five metres of hollow clamps to the top"
        );
        // The same cell with the field UNDER the threshold stays rock.
        let w = cell_word(&c, &deep, Gi::ZERO, &[]);
        assert_eq!(stratum_of_word(w), 5);
        // A TUBE through the cell hollows it just as a cavern does.
        let p = point_at(deep.dir, deep.r_steps);
        let tube = Tube {
            start: [p[0], p[1], p[2]],
            end: [p[0], p[1] + Gi::new(1_000), p[2]],
            radius_steps: Gi::new(4),
            inv_len2: Gi::new(crate::root::recip_pow2(1_000 * 1_000, TUBE_RECIP_BITS) as i64),
        };
        let w = cell_word(&c, &deep, Gi::ZERO, &[tube]);
        assert_eq!(stratum_of_word(w), 0);
        assert_eq!(gap_of_word(w), 4);
        // OUTSIDE the band neither carver is asked: the same tube leaves the cell alone.
        let shallow = at(2_000 - 5, 2_000, 1);
        let w = cell_word(&c, &shallow, value, &[tube]);
        assert_eq!(stratum_of_word(w), 5);
        // A COARSER RUNG: the rung shifts the gap and the hollow into cells of gap step.
        let coarse = CellCharter {
            rung: Gi::new(2),
            ..c
        };
        let w = cell_word(&coarse, &at(2_000 - 40, 2_000, 1), Gi::ZERO, &[]);
        assert_eq!(
            gap_of_word(w),
            -10,
            "a four-metre cell reads a quarter of the steps"
        );
        // The skips write the bytes the kernel would.
        let high = Gi::new(2_000) << LENGTH_BITS;
        let under_the_sea = Gi::new(500) << LENGTH_BITS;
        assert_eq!(gap_of_word(above_cell_word(&c, high)), 127);
        assert_eq!(stratum_of_word(above_cell_word(&c, high)), 0);
        assert_eq!(stratum_of_word(above_cell_word(&c, under_the_sea)), 1);
        assert_eq!(gap_of_word(below_cell_word(&c)), -128);
        assert_eq!(stratum_of_word(below_cell_word(&c)), 13);
    }

    #[test]
    fn a_column_reads_its_cavern_from_the_flat_node_buffer_and_a_phantom_reads_nothing() {
        // A 2 × 2 × 2 lattice whose values are 0, 100, 200 … 700, laid out after two spare words.
        let nodes: Vec<Gi> = (0..10).map(|i| Gi::new((i - 2) * 100)).collect();
        let column = Column {
            base: Gi::new(2),
            na: Gi::ZERO,
            nb: Gi::ZERO,
            d0: Gi::new(2),
            d1: Gi::new(2),
            wa: Gi::ZERO,
            wb: Gi::ZERO,
            has: Gi::ONE,
            ..Default::default()
        };
        let layer = Layer {
            node: Gi::ZERO,
            weight: Gi::ZERO,
            ..Default::default()
        };
        assert_eq!(cavern_at(&column, &layer, &nodes), Gi::ZERO);
        let one = Gi::ONE << LENGTH_BITS;
        let far = Column {
            wa: one,
            wb: one,
            ..column
        };
        let top = Layer {
            weight: one,
            ..layer
        };
        assert_eq!(cavern_at(&far, &top, &nodes), Gi::new(700));
        // A corner phantom has no lattice and opens no room.
        let phantom = Column {
            has: Gi::ZERO,
            ..far
        };
        assert_eq!(cavern_at(&phantom, &top, &nodes), Gi::ZERO);
        assert_eq!(Layer::default().rule, Gi::ZERO);
        assert_eq!(LAYER_EVALUATED, 0);
        assert_eq!(LAYER_BELOW, 1);
        assert_eq!(LAYER_ABOVE, 2);
        assert_eq!(Gi::new(BIOME_MASK), Gi::new(3));
    }
}
