//! ★ THE BOX'S GPU PLAN (ruling F7 item 2; step G1 of `slice_08_integer_recipe_design.md` §2):
//! everything one box of cells needs, laid out as WORDS a card can read, so the client's GPU runs
//! `vd_recipe::cell::cell_word` on every cell of a chunk and its halo.
//!
//! **What this module is, and what it is not.** It is a LAYOUT, not a second recipe: it holds no
//! arithmetic of the world's shape. The plan states the body's charter, one row per radial layer,
//! one row per column, the cavern lattices' node values laid end to end, and the carvers that reach
//! the box. Every number in it is computed by the same functions the CPU's own cell pass calls
//! (`lattice::box_setup`), and the kernel that reads it is the same kernel the shard's collider
//! calls. The card and the shard therefore cannot disagree; there is only one arithmetic.
//!
//! **Why the host still resolves the lattice.** A column of a box may belong to the chunk's own
//! face, to a PARTNER face across a seam, or to no face at all (a corner phantom). That is topology,
//! not arithmetic, and it costs a few hundred lookups per box against a quarter of a million cells.
//! So the host answers it once — a base, two node indices, two strides and two weights per column —
//! and the kernel does nothing but read and blend. Step G2 moves the columns themselves to the card.
//!
//! **The one word per cell.** The kernel writes the substance code in a byte and the gap byte above
//! it ([`vd_recipe::cell::pack`]). [`BoxPlan::box_of`] turns a readback into the same
//! [`SampleBox`] the CPU builds, which the extractor then reads without knowing which host made it.
//!
//! **Example.** The client wants chunk (face 2, rung 0, 19, 1, 4) of the home planet. The worker
//! builds its plan — 64 layers, 4 096 columns, 6 859 cavern nodes, the three tube carvers that
//! reach it — uploads about half a megabyte, and reads back 262 144 words: the same 262 144 words
//! the shard's CPU writes for the same key, which is why the ground the player walks on is the
//! ground the card draws.

use crate::body::BodyDefinition;
use crate::carve::{CAVERN_STRIDE, CAVERN_STRIDE_LOG2};
use crate::chunk::{CHUNK_EDGE, ChunkKey, ColumnField, cell_of_word, column_field, in_ladder};
use crate::chunk::NodeLattice;
use crate::lattice::{BOX_EDGE, HALO, SampleBox, Site, box_setup};
use crate::units::LENGTH_BITS;
use vd_recipe::Gi;
use vd_recipe::cell::{
    CellAt, CellCharter, Column, LAYER_ABOVE, LAYER_BELOW, LAYER_EVALUATED, Layer, Tube,
    above_cell_word, below_cell_word, cavern_at, cell_word,
};

/// How many words each `repr(C)` row of the plan holds. A test measures each against the type's
/// own size, so a field added to the recipe without a word added here is a red test, never a
/// silently misread buffer.
pub const CHARTER_WORDS: usize = 18;
/// The words of one radial layer.
pub const LAYER_WORDS: usize = 4;
/// The words of one column.
pub const COLUMN_WORDS: usize = 13;
/// The words of one tube carver.
pub const TUBE_WORDS: usize = 8;

/// ONE BOX, as words: what a compute pass reads and what it writes back into.
pub struct BoxPlan {
    pub key: ChunkKey,
    /// The body's charter at this rung, with the box's own edge.
    pub charter: CellCharter,
    /// One row per radial layer of the box, in the box's own order (`c` from the halo up).
    pub layers: Vec<Layer>,
    /// One row per column of the box, in packing order `(b + 1) · 64 + (a + 1)`.
    pub columns: Vec<Column>,
    /// Every cavern lattice's values, laid end to end; a column's row says where its own begins.
    /// Never empty: a card refuses a buffer of no bytes, so one spare word stands in.
    pub nodes: Vec<Gi>,
    /// The carvers that can reach the box. Never empty for the same reason: a carver of NO RADIUS
    /// stands in, and `radius − distance` is never above zero for it, so it hollows nothing.
    pub tubes: Vec<Tube>,
    /// The columns' sites and directions — what the extractor reads beside the cells.
    pub sites: Vec<Site>,
    pub dirs: Vec<[Gi; 3]>,
}

/// The node a cell index sits in: the stride's own SHIFT, which floors on both sides of zero (a
/// halo cell at −1 reads the node pair `(−1, 0)`, never the truncating divide's `(0, 1)`).
fn node_of(v: i32) -> i32 {
    v >> CAVERN_STRIDE_LOG2
}

/// The weight of a cell index inside its node pair, at [`LENGTH_BITS`]: the stride is a power of
/// two, so the weight is a MASK and the word is exact.
fn weight_of(v: i32) -> Gi {
    Gi::new(i64::from(v & (CAVERN_STRIDE as i32 - 1))) << (LENGTH_BITS - CAVERN_STRIDE_LOG2)
}

/// The plan of the box at `key`; `None` for a key outside the body's ladder.
#[must_use]
pub fn plan(body: &BodyDefinition, key: ChunkKey) -> Option<BoxPlan> {
    if !in_ladder(body, key) {
        return None;
    }
    let column = column_field(body, key.face, key.rung, key.x, key.y)?;
    Some(plan_in(body, key, &column))
}

/// The plan of the box at `key`, whose core columns come from `column` — what a worker that already
/// holds the column pass calls.
#[must_use]
pub fn plan_in(body: &BodyDefinition, key: ChunkKey, column: &ColumnField) -> BoxPlan {
    let setup = box_setup(body, key, column);
    let edge = CHUNK_EDGE as i32;
    let rung = key.rung;

    // The lattices, laid end to end: the chunk's own face first, then each partner face present.
    let mut nodes: Vec<Gi> = Vec::new();
    let mut bases: Vec<(u8, usize, &NodeLattice)> = Vec::new();
    if setup.carve_caverns {
        bases.push((setup.own.face.index(), nodes.len(), &setup.own));
        nodes.extend_from_slice(&setup.own.values);
        for lattice in &setup.foreign {
            bases.push((lattice.face.index(), nodes.len(), lattice));
            nodes.extend_from_slice(&lattice.values);
        }
    }
    if nodes.is_empty() {
        // A card refuses a buffer of no bytes; no column reads this word, because none has a lattice.
        nodes.push(Gi::ZERO);
    }

    // One row per column: what the column pass said, and where its lattice sits.
    let mut columns = Vec::with_capacity(BOX_EDGE * BOX_EDGE);
    let mut col = 0;
    while col < setup.sites.len() {
        let site = setup.sites[col];
        let (h, biome) = setup.surfaces[col];
        let mut row = Column {
            dir: setup.dirs[col],
            h,
            biome: Gi::new(biome as i64),
            ..Default::default()
        };
        let mut b = 0;
        while b < bases.len() {
            let (face, base, lattice) = bases[b];
            if face == site.face {
                row.base = Gi::new(base as i64);
                row.na = Gi::new(i64::from(node_of(site.i) - lattice.node0[0]));
                row.nb = Gi::new(i64::from(node_of(site.j) - lattice.node0[1]));
                row.d0 = Gi::new(lattice.dims[0] as i64);
                row.d1 = Gi::new(lattice.dims[1] as i64);
                row.wa = weight_of(site.i);
                row.wb = weight_of(site.j);
                row.has = Gi::ONE;
                break;
            }
            b += 1;
        }
        columns.push(row);
        col += 1;
    }

    // One row per radial layer: its cell radius, its rule, and its own node and weight.
    let mut layers = Vec::with_capacity(BOX_EDGE);
    let mut c = -HALO;
    while c <= edge {
        let k = setup.k0 + c;
        let rule = if k < 0 {
            LAYER_BELOW
        } else if k >= setup.band {
            LAYER_ABOVE
        } else {
            LAYER_EVALUATED
        };
        let evaluated = rule == LAYER_EVALUATED;
        layers.push(Layer {
            r_steps: Gi::new(body.ladder().cell_radius_steps(k, rung)),
            rule: Gi::new(rule),
            // A skipped layer never reads a node, and its own would sit outside the lattice.
            node: if evaluated {
                Gi::new(i64::from(node_of(k) - setup.node0[2]))
            } else {
                Gi::ZERO
            },
            weight: if evaluated { weight_of(k) } else { Gi::ZERO },
        });
        c += 1;
    }

    let mut tubes = setup.tubes;
    if tubes.is_empty() {
        tubes.push(Tube::default());
    }
    BoxPlan {
        key,
        charter: setup.charter,
        layers,
        columns,
        nodes,
        tubes,
        sites: setup.sites,
        dirs: setup.dirs,
    }
}

impl BoxPlan {
    /// THE PLAN RUN ON THIS HOST: every cell's word, in the box's packing order, through the same
    /// kernels the card runs. The CPU's reference for a GPU's answer, and the fallback path on a
    /// card the self-check refuses.
    #[must_use]
    pub fn cells(&self) -> Vec<u32> {
        let edge = self.charter.box_edge.raw() as usize;
        let mut out = Vec::with_capacity(edge * edge * edge);
        let mut c = 0;
        while c < edge {
            let mut b = 0;
            while b < edge {
                let mut a = 0;
                while a < edge {
                    out.push(self.cell(a, b, c));
                    a += 1;
                }
                b += 1;
            }
            c += 1;
        }
        out
    }

    /// One cell's word, by its place in the box — the body of the GPU's entry point, on this host.
    #[must_use]
    pub fn cell(&self, a: usize, b: usize, c: usize) -> u32 {
        let edge = self.charter.box_edge.raw() as usize;
        let layer = &self.layers[c];
        let column = &self.columns[b * edge + a];
        match layer.rule.raw() {
            LAYER_BELOW => below_cell_word(&self.charter),
            LAYER_ABOVE => above_cell_word(&self.charter, layer.r_steps << LENGTH_BITS),
            _ => cell_word(
                &self.charter,
                &CellAt {
                    dir: column.dir,
                    h: column.h,
                    biome: column.biome,
                    r_steps: layer.r_steps,
                },
                cavern_at(column, layer, &self.nodes),
                &self.tubes,
            ),
        }
    }

    /// The box a readback names: the same [`SampleBox`] the CPU's own cell pass builds, which the
    /// extractor reads without ever learning which host wrote the cells.
    #[must_use]
    pub fn box_of(&self, words: &[u32]) -> SampleBox {
        let mut cells = Vec::with_capacity(words.len());
        let mut i = 0;
        while i < words.len() {
            cells.push(cell_of_word(words[i]));
            i += 1;
        }
        SampleBox {
            key: self.key,
            cells,
            sites: self.sites.clone(),
            dirs: self.dirs.clone(),
        }
    }

    /// The charter as [`CHARTER_WORDS`] words, in the `repr(C)` order the kernel reads.
    #[must_use]
    pub fn charter_words(&self) -> Vec<i64> {
        let c = &self.charter;
        let mut w = vec![
            c.sea_radius.raw(),
            c.cave_min.raw(),
            c.cave_max.raw(),
            c.cavern_threshold.raw(),
            c.cavern_scale_steps.raw(),
            c.carve_any.raw(),
            c.rung.raw(),
            c.topsoil_m.raw(),
            c.subsoil_end_m.raw(),
            c.strata_end_m.raw(),
            c.air_code.raw(),
            c.water_code.raw(),
            c.bedrock_code.raw(),
            c.box_edge.raw(),
        ];
        let mut i = 0;
        while i < c.strata.len() {
            w.push(c.strata[i].raw());
            i += 1;
        }
        w
    }

    /// The layers as [`LAYER_WORDS`] words each.
    #[must_use]
    pub fn layer_words(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.layers.len() * LAYER_WORDS);
        let mut i = 0;
        while i < self.layers.len() {
            let l = &self.layers[i];
            w.extend([l.r_steps.raw(), l.rule.raw(), l.node.raw(), l.weight.raw()]);
            i += 1;
        }
        w
    }

    /// The columns as [`COLUMN_WORDS`] words each.
    #[must_use]
    pub fn column_words(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.columns.len() * COLUMN_WORDS);
        let mut i = 0;
        while i < self.columns.len() {
            let c = &self.columns[i];
            w.extend([
                c.dir[0].raw(),
                c.dir[1].raw(),
                c.dir[2].raw(),
                c.h.raw(),
                c.biome.raw(),
                c.base.raw(),
                c.na.raw(),
                c.nb.raw(),
                c.d0.raw(),
                c.d1.raw(),
                c.wa.raw(),
                c.wb.raw(),
                c.has.raw(),
            ]);
            i += 1;
        }
        w
    }

    /// The cavern nodes as one word each.
    #[must_use]
    pub fn node_words(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.nodes.len());
        let mut i = 0;
        while i < self.nodes.len() {
            w.push(self.nodes[i].raw());
            i += 1;
        }
        w
    }

    /// The tube carvers as [`TUBE_WORDS`] words each.
    #[must_use]
    pub fn tube_words(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.tubes.len() * TUBE_WORDS);
        let mut i = 0;
        while i < self.tubes.len() {
            let t = &self.tubes[i];
            w.extend([
                t.start[0].raw(),
                t.start[1].raw(),
                t.start[2].raw(),
                t.end[0].raw(),
                t.end[1].raw(),
                t.end[2].raw(),
                t.radius_steps.raw(),
                t.inv_len2.raw(),
            ]);
            i += 1;
        }
        w
    }
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample chunk with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample chunk; never a kernel's path"
    )]
    use super::*;
    use crate::digest::surface_chunk_z;
    use crate::home::home_planet;
    use crate::lattice::{BOX_CELLS, sample_box};
    use vd_seed::bend::Face;

    fn key(face: Face, rung: u8, x: i32, y: i32, z: i32) -> ChunkKey {
        ChunkKey {
            face,
            rung,
            x,
            y,
            z,
        }
    }

    /// ★ THE PLAN IS THE BOX, CELL FOR CELL. The words a plan produces on this host are the bytes
    /// the CPU's own cell pass writes — a surface chunk, a chunk at a face's far edge (partner
    /// columns and corner phantoms), a chunk under the band's floor and one over its top.
    #[test]
    fn the_plan_names_the_same_cells_the_cell_pass_writes() {
        let m = home_planet();
        // THE LAST chunk of the face, whose halo (and, where the face's cell count is not a whole
        // number of chunks, whose own far columns) crosses the seam into a PARTNER face — the box
        // that makes the plan lay a second lattice down and a column read it.
        let last = (m.ladder().cells_per_edge(0) as i32 - 1) / CHUNK_EDGE as i32;
        for k in [
            key(Face::PosX, 0, 300, 700, surface_chunk_z(&m, Face::PosX, 0, 300, 700)),
            key(
                Face::PosX,
                0,
                last,
                last,
                surface_chunk_z(&m, Face::PosX, 0, last, last),
            ),
            key(Face::PosZ, 3, 5, 7, surface_chunk_z(&m, Face::PosZ, 3, 5, 7)),
            key(Face::PosX, 0, 300, 700, 0),
            key(
                Face::PosX,
                0,
                300,
                700,
                (m.ladder().cells_in_band(0) as i32 / CHUNK_EDGE as i32) - 1,
            ),
        ] {
            let plan = plan(&m, k).expect("the key is on the ladder");
            let words = plan.cells();
            assert_eq!(words.len(), BOX_CELLS);
            let want = sample_box(&m, k).expect("the box");
            let got = plan.box_of(&words);
            assert_eq!(got.cells, want.cells, "{k:?}: the cells");
            assert_eq!(got.sites, want.sites, "{k:?}: the sites");
            assert_eq!(got.dirs, want.dirs, "{k:?}: the directions");
            assert_eq!(got.key, want.key);
        }
        assert!(plan(&m, key(Face::PosX, 0, -1, 0, 0)).is_none(), "off the ladder");
        // ★ THE SEAM IS ACTUALLY CROSSED: the last chunk's box holds a PARTNER face's columns, the
        // plan lays that face's lattice down after its own, and those columns read it. Stated as a
        // measurement, because a box that never crosses would pass every assertion above and leave
        // the partner's whole path unwalked.
        let edge_key = key(
            Face::PosX,
            0,
            last,
            last,
            surface_chunk_z(&m, Face::PosX, 0, last, last),
        );
        let edge_plan = plan(&m, edge_key).expect("the edge chunk");
        assert!(
            edge_plan.columns.iter().any(|c| c.base > Gi::ZERO),
            "a column reads a partner face's lattice"
        );
        assert!(
            edge_plan.columns.iter().any(|c| c.has == Gi::ZERO),
            "a corner phantom has no lattice"
        );
    }

    /// ★ THE WORD WRITERS MATCH THE TYPES THEY WRITE. A field added to a recipe row without a word
    /// added here would be read as the next row's first word on the card; this measures the sizes
    /// so that can never happen silently.
    #[test]
    fn every_row_writes_exactly_as_many_words_as_its_type_holds() {
        use core::mem::size_of;
        assert_eq!(size_of::<CellCharter>(), CHARTER_WORDS * 8);
        assert_eq!(size_of::<Layer>(), LAYER_WORDS * 8);
        assert_eq!(size_of::<Column>(), COLUMN_WORDS * 8);
        assert_eq!(size_of::<Tube>(), TUBE_WORDS * 8);
        let m = home_planet();
        let k = key(Face::PosX, 0, 300, 700, surface_chunk_z(&m, Face::PosX, 0, 300, 700));
        let plan = plan(&m, k).expect("the key is on the ladder");
        assert_eq!(plan.charter_words().len(), CHARTER_WORDS);
        assert_eq!(plan.layer_words().len(), BOX_EDGE * LAYER_WORDS);
        assert_eq!(
            plan.column_words().len(),
            BOX_EDGE * BOX_EDGE * COLUMN_WORDS
        );
        assert_eq!(plan.tube_words().len(), plan.tubes.len() * TUBE_WORDS);
        assert_eq!(plan.node_words().len(), plan.nodes.len());
        assert!(!plan.nodes.is_empty(), "a card refuses an empty buffer");
        assert!(!plan.tubes.is_empty());
        // The charter's own words, read back in the order the kernel reads them.
        let w = plan.charter_words();
        assert_eq!(w[0], plan.charter.sea_radius.raw());
        assert_eq!(w[13], i64::from(BOX_EDGE as u32));
        assert_eq!(w[CHARTER_WORDS - 1], plan.charter.strata[3].raw());
    }

    /// ★ A COARSE RUNG CARVES NO CAVES, so no column has a lattice and the spare word stands in —
    /// and the cells are still the cell pass's own.
    #[test]
    fn a_rung_without_caves_plans_no_lattice_and_still_names_the_same_cells() {
        let m = home_planet();
        let rung = 8u8;
        let k = key(Face::PosY, rung, 1, 1, surface_chunk_z(&m, Face::PosY, rung, 1, 1));
        let plan = plan(&m, k).expect("the key is on the ladder");
        assert_eq!(plan.nodes.len(), 1, "one spare word, no lattice");
        assert!(plan.columns.iter().all(|c| c.has == Gi::ZERO));
        assert_eq!(plan.tubes.len(), 1, "one carver of no radius");
        assert_eq!(plan.tubes[0], Tube::default());
        let want = sample_box(&m, k).expect("the box");
        assert_eq!(plan.box_of(&plan.cells()).cells, want.cells);
    }
}
