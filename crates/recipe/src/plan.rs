//! ★ THE BOX'S PLAN, ON THE CARD (step G2-A of `slice_08_integer_recipe_design.md` §2): the two
//! passes that stand in FRONT of the cell field — the COLUMN PASS (a column's direction, its
//! surface and its biome) and the NODE PASS (the cavern field at one lattice node).
//!
//! **Why they moved.** Step G1 put the cell field on the card and MEASURED that it did not pay: the
//! card took 1.79 ms of a box and the host still took 2.14 ms preparing the box's plan, against
//! 1.24 ms for the whole box on the terrain's share of this machine's cores (ruling F6). Almost all
//! of that plan was arithmetic, not topology: four thousand columns of octave sum and seven thousand
//! cavern nodes. This module is those two kernels, so a box's request carries its KEY and its
//! CHARTER and the host keeps only the lookups — which face a column belongs to across a seam, and
//! which carvers reach the box.
//!
//! **One source, as always.** `vd_terrain` calls the very functions below for its own column pass
//! and its own node lattice, so the shard's collision, the client's CPU fallback and the card read
//! ONE arithmetic. A second copy would be a port, and SL10 forbids one.
//!
//! **Example.** The client asks for chunk (face 2, rung 0, 19, 1, 4). The host says: these 4 096
//! columns sit at these sites, these 6 859 nodes sit on these lattices, these three carvers reach
//! the box. The card then draws every column's hill and every node's cave itself, and the cell
//! field reads what the card just wrote — no megabyte of plan crosses the bus.

use crate::bend::{DIR_ONE, basis_of, direction, normalise};
use crate::cell::{Column, LENGTH_BITS, point_at};
use crate::gi::Gi;
use crate::height::{BiomeCharter, OCTAVES_CAP, Octave, Roughness, biome_of, relief_of_table};
use crate::noise::value3;
use crate::terrace::{Terrace, terrace};

/// The face byte of a CORNER PHANTOM column: no face at all. Three faces meet at a cube corner and
/// the fourth cell of a two-by-two ring does not exist, so a box holds a placeholder there on the
/// corner's own direction. The extractor never reads a phantom's cells.
pub const CORNER_FACE: i32 = 255;

/// ★ THE PLAN CHARTER — every number the two passes read that is not the column's or the node's own
/// address. Drawn ONCE per box by the host and handed to the card as one row. `repr(C)`, so the
/// buffer the host writes is the row the kernel reads.
///
/// **Example.** The home planet at rung 0: the body's seed, the cell-count reciprocal of that rung,
/// the ladder radius, fourteen live octaves, the biome field's two slow noises, and the cavern
/// field's wavelength reciprocal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PlanCharter {
    /// The body's own seed — the cavern field's.
    pub seed: u64,
    /// The cavern wavelength's reciprocal, and the shift that takes its own bits and the gap step's
    /// back out of the product.
    pub cavern_recip: Gi,
    pub cavern_shift: Gi,
    /// The cell-count reciprocal at this rung ([`crate::bend::inv_n_of`]), so a direction divides
    /// nothing.
    pub inv_n: Gi,
    /// The body's ladder radius, in gap steps at [`crate::cell::LENGTH_BITS`].
    pub radius: Gi,
    /// How many of the octaves below are live at this rung.
    pub octave_count: Gi,
    /// The CHUNK's own face index — the basis a corner phantom's direction stands on.
    pub key_face: Gi,
    /// The biome field.
    pub biome: BiomeCharter,
    /// ★ THE PER-COLUMN ROUGHNESS FACTOR's words (slice 8a stage 3): the slow placeholder octave,
    /// the factor's floor, and the index the fine band begins at. The column pass multiplies its
    /// fine half by the factor ONCE.
    pub roughness: Roughness,
    /// ★ THE CAP-ROCK BENCH's words (slice 8a stage 4): the datum radius, the bed spacing and its
    /// two reciprocals, and the strength ALREADY FADED for this rung. The column pass pulls its
    /// surface toward the nearest bed top ONCE, after the octave sum.
    pub terrace: Terrace,
    /// The octaves, coarsest first.
    pub octaves: [Octave; OCTAVES_CAP],
}

/// What the column pass says about one column of a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnSurface {
    /// The column's unit direction, at the bend's fraction bits.
    pub dir: [Gi; 3],
    /// The surface's radius along it, in gap steps at [`crate::cell::LENGTH_BITS`].
    pub h: Gi,
    /// The biome's code.
    pub biome: Gi,
}

/// THE DIRECTION OF A BOX COLUMN: a face cell's own direction, or a corner phantom's. A phantom's
/// axis sum is the chunk face's normal plus the two corner signs along its own axes, and
/// [`normalise`] puts it on the sphere — the same kernel the bend itself ends with, so the three
/// faces that meet at a corner read the same word triple.
#[must_use]
pub fn site_direction(charter: &PlanCharter, face: i32, i: i32, j: i32) -> [Gi; 3] {
    site_direction_of(charter.inv_n, charter.key_face.raw() as i32, face, i, j)
}

/// [`site_direction`] from the two words it actually reads — the rung's cell-count reciprocal and
/// the CHUNK's own face — so a host asking about one column pays no charter.
#[must_use]
pub fn site_direction_of(inv_n: Gi, key_face: i32, face: i32, i: i32, j: i32) -> [Gi; 3] {
    if face == CORNER_FACE {
        let b = basis_of(key_face);
        normalise([
            corner_axis(b.n[0], b.u[0], b.v[0], i, j),
            corner_axis(b.n[1], b.u[1], b.v[1], i, j),
            corner_axis(b.n[2], b.u[2], b.v[2], i, j),
        ])
    } else {
        direction(face, i, j, inv_n)
    }
}

/// One component of a corner phantom's axis sum, at the bend's fraction bits.
fn corner_axis(n: i32, u: i32, v: i32, i: i32, j: i32) -> Gi {
    Gi::new(i64::from(n)) * DIR_ONE
        + Gi::new(i64::from(i) * i64::from(u)) * DIR_ONE
        + Gi::new(i64::from(j) * i64::from(v)) * DIR_ONE
}

/// ★ THE COLUMN PASS, one column: its direction, the surface's radius along it, and its biome. The
/// kernel the card runs once per column of a box, and the function `vd_terrain`'s own column pass
/// calls.
#[must_use]
pub fn column_surface(charter: &PlanCharter, face: i32, i: i32, j: i32) -> ColumnSurface {
    let dir = site_direction(charter, face, i, j);
    // ★ THE BENCH IS LAST (slice 8a stage 4): the octave sum answers the raw surface, and the
    // terrace then pulls it toward the nearest bed top. The biome below reads the TERRACED surface,
    // because the biome reads where the ground actually stands.
    let h = terrace(
        &charter.terrace,
        charter.radius
            + relief_of_table(
                &charter.octaves,
                charter.octave_count.raw() as usize,
                dir,
                &charter.roughness,
            ),
    );
    ColumnSurface {
        dir,
        h,
        biome: biome_of(&charter.biome, dir, h),
    }
}

/// THE CAVERN FIELD at a point of the body's frame, in `[0, 1)` at the noise's fraction bits: the
/// point in gap steps times the wavelength's reciprocal, through the value noise. The three words it
/// reads are named one by one rather than as a charter, so a host that holds only them — the
/// shard's own cave test — pays no charter to ask.
#[must_use]
pub fn cavern_value(seed: u64, recip: Gi, shift: u32, point_steps: [Gi; 3]) -> Gi {
    value3(
        seed,
        [
            point_steps[0].mul_shr(recip, shift),
            point_steps[1].mul_shr(recip, shift),
            point_steps[2].mul_shr(recip, shift),
        ],
    )
}

impl PlanCharter {
    /// The cavern field at a point, through this charter's own three words.
    #[must_use]
    pub fn cavern_of(&self, point_steps: [Gi; 3]) -> Gi {
        cavern_value(
            self.seed,
            self.cavern_recip,
            self.cavern_shift.raw() as u32,
            point_steps,
        )
    }
}

/// ★ THE NODE PASS, one node: the cavern field at the node's own direction and corner radius. The
/// kernel the card runs once per node of a box's lattices, and the function `vd_terrain`'s own
/// `NodeLattice` calls.
#[must_use]
pub fn node_value(charter: &PlanCharter, face: i32, i: i32, j: i32, r_steps: Gi) -> Gi {
    charter.cavern_of(point_at(direction(face, i, j, charter.inv_n), r_steps))
}

/// Cells between two nodes of the cavern lattice, as a SHIFT: the node a cell sits in is an
/// arithmetic shift and the weight inside the node pair is a mask, so no kernel divides and both
/// floor on either side of zero. `vd_terrain::carve` holds the same two words and asserts they agree.
pub const CAVERN_STRIDE_LOG2: u32 = 2;
/// The same stride as a count.
pub const CAVERN_STRIDE: i32 = 1 << CAVERN_STRIDE_LOG2;

/// ★ ONE CAVERN LATTICE of a box, as the card reads it: which face it is drawn on, where it starts
/// in the flat node buffer, its first node along each axis, and how many nodes it holds along each.
/// `repr(C)`, eight words.
///
/// **Example.** The chunk at the far corner of face `+X` lays TWO lattices down: its own face's, and
/// face `+Y`'s over the halo columns across the seam. A column of the box names its lattice by its
/// site's face, and a corner phantom names none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct NodeBlock {
    /// The face this lattice is drawn on.
    pub face: Gi,
    /// Where its values start in the flat node buffer.
    pub base: Gi,
    /// Its first node along each of the three axes.
    pub node0: [Gi; 3],
    /// How many nodes it holds along each of the three axes.
    pub dims: [Gi; 3],
}

/// The node a cell index sits in: the stride's own SHIFT, which floors on both sides of zero (a halo
/// cell at minus one reads the node pair `(-1, 0)`, never the truncating divide's `(0, 1)`).
#[must_use]
pub fn node_of(v: i32) -> Gi {
    Gi::new(i64::from(v)) >> CAVERN_STRIDE_LOG2
}

/// The weight of a cell index inside its node pair, at the length format: the stride is a power of
/// two, so the weight is a MASK and the word is exact.
#[must_use]
pub fn weight_of(v: i32) -> Gi {
    (Gi::new(i64::from(v)) & Gi::new(i64::from(CAVERN_STRIDE - 1)))
        << (LENGTH_BITS - CAVERN_STRIDE_LOG2)
}

/// ★ ONE COLUMN ROW, whole: what the column pass said about the column AND where its cavern lattice
/// sits. The card builds the row itself from the site and the lattice list, so no host uploads four
/// thousand rows of topology.
///
/// ★ THE SCAN IS ADDITIVE (the design's §1b rule, MEASURED in step G1): every word the loop carries
/// out is only ever ADDED to. An accumulator a branch ASSIGNS comes back one step stale on the card,
/// which is how the carvers' hollow read zero on a quarter of a billion cells.
///
/// **Example.** A halo column of a `+X` chunk that belongs to face `+Y` finds face `+Y`'s lattice in
/// the list, states its base and its two node indices, and the cell kernel then blends eight of that
/// lattice's nodes. A corner phantom finds none, states `has` as zero, and carves no cave.
#[must_use]
pub fn column_row(
    charter: &PlanCharter,
    lattices: &[NodeBlock],
    face: i32,
    i: i32,
    j: i32,
) -> Column {
    let s = column_surface(charter, face, i, j);
    let want = Gi::new(i64::from(face));
    let mut base = Gi::ZERO;
    let mut na = Gi::ZERO;
    let mut nb = Gi::ZERO;
    let mut d0 = Gi::ZERO;
    let mut d1 = Gi::ZERO;
    let mut has = Gi::ZERO;
    let mut l = 0;
    while l < lattices.len() {
        let block = &lattices[l];
        let hit = (block.face == want) & (has == Gi::ZERO);
        base += if hit { block.base - base } else { Gi::ZERO };
        na += if hit {
            node_of(i) - block.node0[0] - na
        } else {
            Gi::ZERO
        };
        nb += if hit {
            node_of(j) - block.node0[1] - nb
        } else {
            Gi::ZERO
        };
        d0 += if hit { block.dims[0] - d0 } else { Gi::ZERO };
        d1 += if hit { block.dims[1] - d1 } else { Gi::ZERO };
        has += if hit { Gi::ONE } else { Gi::ZERO };
        l += 1;
    }
    // ★ A COLUMN WITH NO LATTICE CARRIES NO WEIGHT EITHER. `cavern_at` never reads `wa` or `wb`
    // where `has` is zero, so filling them would change no cell — but it would make the row differ
    // from the row the host wrote before this step, and a row that is byte-identical is one fewer
    // thing to argue about when a card and a shard are compared.
    let (wa, wb) = if has == Gi::ZERO {
        (Gi::ZERO, Gi::ZERO)
    } else {
        (weight_of(i), weight_of(j))
    };
    Column {
        dir: s.dir,
        h: s.h,
        biome: s.biome,
        base,
        na,
        nb,
        d0,
        d1,
        wa,
        wb,
        has,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bend::{DIR_BITS, inv_n_of};
    use crate::cell::LENGTH_BITS;
    use crate::height::{AMP_BITS, GAP_STEPS_PER_CELL};
    use crate::noise::{NOISE_BITS, NOISE_ONE};

    fn octave(seed: u64, frequency_int: i64, amplitude: i64) -> Octave {
        Octave::smooth(seed, Gi::new(frequency_int), Gi::ZERO, Gi::new(amplitude))
    }

    fn charter() -> PlanCharter {
        let mut octaves = [octave(0, 0, 0); OCTAVES_CAP];
        octaves[0] = octave(7, 12, (4 * GAP_STEPS_PER_CELL) << AMP_BITS);
        octaves[1] = octave(9, 31, GAP_STEPS_PER_CELL << AMP_BITS);
        PlanCharter {
            seed: 0x5EED_1234_5678_9ABC,
            cavern_recip: Gi::new(1 << 40),
            cavern_shift: Gi::new(i64::from(48 - NOISE_BITS + 7)),
            inv_n: inv_n_of(1024),
            radius: Gi::new(1_000_000) << LENGTH_BITS,
            octave_count: Gi::new(2),
            key_face: Gi::new(4),
            biome: BiomeCharter {
                sea_radius: Gi::new(1_000_000) << LENGTH_BITS,
                highland_above: Gi::new(400) << LENGTH_BITS,
                highland_recip: Gi::new(1 << 30),
                highland_shift: Gi::new(2),
                temperature: octave(11, 1, 1 << AMP_BITS),
                humidity: octave(13, 1, 1 << AMP_BITS),
            },
            // A roughness field of its own: a slow octave of unit amplitude, a floor of 0.06 at the
            // noise's bits, and the fine band beginning at the second octave.
            roughness: Roughness {
                octave: octave(17, 1, 1 << AMP_BITS),
                m_min: Gi::new(16_106_127),
                first_fine: Gi::ONE,
            },
            // NO BENCH: the charter's own column tests state the octave sum and the biome, and a
            // bench would move every one of their numbers. `column_surface` reads the row all the
            // same, so the test below measures that a zero strength is the identity on the whole
            // path and not only inside the terrace's own kernel.
            terrace: Terrace::NONE,
            octaves,
        }
    }

    /// ★ THE COLUMN PASS IS THE BEND, THE OCTAVE SUM AND THE BIOME, in that order — the same three
    /// answers a host gets by calling them one at a time. A column of the +Z face's own cell and a
    /// corner phantom both answer.
    #[test]
    fn a_column_reads_its_direction_its_surface_and_its_biome() {
        let c = charter();
        let got = column_surface(&c, 4, 300, 700);
        let dir = direction(4, 300, 700, c.inv_n);
        assert_eq!(got.dir, dir);
        assert_eq!(
            got.h,
            c.radius + relief_of_table(&c.octaves, 2, dir, &c.roughness)
        );
        assert_eq!(got.biome, biome_of(&c.biome, dir, got.h));
        // The phantom: the chunk's own face normal with the corner's two signs.
        let corner = column_surface(&c, CORNER_FACE, -1, 1);
        let b = basis_of(4);
        assert_eq!(
            corner.dir,
            normalise([
                corner_axis(b.n[0], b.u[0], b.v[0], -1, 1),
                corner_axis(b.n[1], b.u[1], b.v[1], -1, 1),
                corner_axis(b.n[2], b.u[2], b.v[2], -1, 1),
            ])
        );
        // A corner direction is a unit vector: its square sum is one at the bend's bits.
        let sq = |v: [Gi; 3]| {
            v[0].mul_shr(v[0], DIR_BITS)
                + v[1].mul_shr(v[1], DIR_BITS)
                + v[2].mul_shr(v[2], DIR_BITS)
        };
        let one = DIR_ONE;
        assert!(sq(corner.dir) > one - (one >> 20));
        assert!(sq(corner.dir) < one + (one >> 20));
        assert_ne!(corner.dir, dir);
        assert_eq!(
            corner.h,
            c.radius + relief_of_table(&c.octaves, 2, corner.dir, &c.roughness)
        );
    }

    /// ★ THE NODE PASS IS THE CAVERN FIELD AT THE NODE'S POINT, and a point away from it reads a
    /// different value: the node address actually reaches the noise.
    #[test]
    fn a_node_reads_the_cavern_field_at_its_own_point() {
        let c = charter();
        let r = Gi::new(1_000_000);
        let got = node_value(&c, 4, 64, 128, r);
        assert_eq!(
            got,
            c.cavern_of(point_at(direction(4, 64, 128, c.inv_n), r))
        );
        assert_eq!(
            c.cavern_of([r, r, r]),
            cavern_value(
                c.seed,
                c.cavern_recip,
                c.cavern_shift.raw() as u32,
                [r, r, r]
            )
        );
        assert!(got >= Gi::ZERO);
        assert!(got < NOISE_ONE);
        assert_ne!(got, node_value(&c, 4, 68, 128, r));
        assert_ne!(got, node_value(&c, 0, 64, 128, r));
        assert_ne!(got, node_value(&c, 4, 64, 128, r + Gi::new(4096)));
    }

    /// ★ ONE COLUMN ROW, WHOLE: the surface the column pass names AND the lattice it reads. A
    /// column of the lattice's own face names it; a column of another face names the SECOND lattice;
    /// a corner phantom names none; and where the list holds TWO blocks of one face, the FIRST wins
    /// — the host lays the chunk's own face down first, so a column of that face must never read a
    /// partner's block.
    #[test]
    fn a_column_row_carries_its_surface_and_its_lattice() {
        let c = charter();
        let own = NodeBlock {
            face: Gi::new(4),
            base: Gi::new(100),
            node0: [Gi::new(2), Gi::new(3), Gi::new(0)],
            dims: [Gi::new(19), Gi::new(19), Gi::new(19)],
        };
        let partner = NodeBlock {
            face: Gi::new(0),
            base: Gi::new(7_000),
            node0: [Gi::new(-1), Gi::new(-2), Gi::new(0)],
            dims: [Gi::new(3), Gi::new(19), Gi::new(19)],
        };
        let lattices = [own, partner];
        // The surface half is the column pass's own answer.
        let row = column_row(&c, &lattices, 4, 300, 701);
        let s = column_surface(&c, 4, 300, 701);
        assert_eq!(row.dir, s.dir);
        assert_eq!(row.h, s.h);
        assert_eq!(row.biome, s.biome);
        // The lattice half: the FIRST block of this face.
        assert_eq!(row.has, Gi::ONE);
        assert_eq!(row.base, own.base);
        assert_eq!(row.na, node_of(300) - own.node0[0]);
        assert_eq!(row.nb, node_of(701) - own.node0[1]);
        assert_eq!(row.d0, own.dims[0]);
        assert_eq!(row.d1, own.dims[1]);
        assert_eq!(row.wa, weight_of(300));
        assert_eq!(row.wb, weight_of(701));
        // A column of the PARTNER face reads the second block.
        let across = column_row(&c, &lattices, 0, -1, 5);
        assert_eq!(across.base, partner.base);
        assert_eq!(across.d0, partner.dims[0]);
        assert_eq!(across.has, Gi::ONE);
        // The node and the weight FLOOR on both sides of zero: a halo cell at minus one reads the
        // node pair below it, never the truncating divide's pair above.
        assert_eq!(across.na, node_of(-1) - partner.node0[0]);
        assert_eq!(node_of(-1), Gi::new(-1));
        assert_eq!(
            weight_of(-1),
            Gi::new(3) << (LENGTH_BITS - CAVERN_STRIDE_LOG2)
        );
        assert_eq!(node_of(4), Gi::ONE);
        assert_eq!(weight_of(4), Gi::ZERO);
        // A CORNER PHANTOM names no lattice at all, and no lattice at all names none either.
        let phantom = column_row(&c, &lattices, CORNER_FACE, -1, 1);
        assert_eq!(phantom.has, Gi::ZERO);
        assert_eq!(phantom.base, Gi::ZERO);
        assert_eq!(phantom.d0, Gi::ZERO);
        assert_eq!(phantom.dir, site_direction(&c, CORNER_FACE, -1, 1));
        // ★ AND NO WEIGHT EITHER: a row with no lattice is ZERO in every lattice word, which is
        // what the host wrote before the topology moved to the card. `cavern_at` reads none of
        // them, so this changes no cell; it keeps the ROW byte-identical.
        assert_eq!(phantom.wa, Gi::ZERO);
        assert_eq!(phantom.wb, Gi::ZERO);
        // A box with NO lattice at all leaves every lattice word zero too — the whole row is the
        // surface and nothing else.
        let none = column_row(&c, &[], 4, 300, 701);
        assert_eq!(none.has, Gi::ZERO);
        assert_eq!(
            [
                none.base, none.na, none.nb, none.d0, none.d1, none.wa, none.wb
            ],
            [Gi::ZERO; 7]
        );
        // ★ THE FIRST BLOCK OF A FACE WINS: a second block of the same face changes nothing.
        let twice = [own, partner, own];
        assert_eq!(column_row(&c, &twice, 4, 300, 701), row);
        let shadow = NodeBlock {
            base: Gi::new(9_999),
            ..own
        };
        assert_eq!(column_row(&c, &[own, shadow], 4, 300, 701).base, own.base);
        assert_eq!(CAVERN_STRIDE, 4);
    }

    /// ★ EVERY FACE INDEX ANSWERS, and a phantom on each face is a unit direction — so no arm of
    /// the basis match is left unwalked by the plan's own path.
    #[test]
    fn every_face_and_every_corner_sign_answers() {
        let mut c = charter();
        let mut face = 0i32;
        while face < 6 {
            c.key_face = Gi::new(i64::from(face));
            let own = column_surface(&c, face, 5, 9);
            assert_eq!(own.dir, direction(face, 5, 9, c.inv_n));
            for (i, j) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let p = site_direction(&c, CORNER_FACE, i, j);
                let sq = p[0].mul_shr(p[0], DIR_BITS)
                    + p[1].mul_shr(p[1], DIR_BITS)
                    + p[2].mul_shr(p[2], DIR_BITS);
                assert!(
                    sq > DIR_ONE - (DIR_ONE >> 20),
                    "face {face} corner ({i}, {j})"
                );
                assert!(sq < DIR_ONE + (DIR_ONE >> 20));
            }
            face += 1;
        }
        // A face index past the table reads the last row, so the kernel is total.
        assert_eq!(
            site_direction(&c, 9, 5, 9),
            direction(9, 5, 9, c.inv_n),
            "an index past the table is total"
        );
    }
}
