//! ★ THE BOX'S GPU PLAN (ruling F7 item 2; step G1 of `slice_08_integer_recipe_design.md` §2):
//! everything one box of cells needs, laid out as WORDS a card can read, so the client's GPU runs
//! `vd_recipe::cell::cell_word` on every cell of a chunk and its halo.
//!
//! **What this module is, and what it is not.** It is a LAYOUT, not a second recipe: it holds no
//! arithmetic of the world's shape. ★ Since step G2-A it holds no arithmetic of the world's PLAN
//! either: the plan states the body's two charters, one site per column, the cavern lattices'
//! EXTENTS, one row per radial layer, and the carvers that reach the box — and the CARD computes
//! every direction, every surface, every biome and every node value from them. The kernels it runs
//! are the kernels the shard's collider calls, so the card and the shard cannot disagree; there is
//! only one arithmetic.
//!
//! **Why the host still resolves the topology.** A column of a box may belong to the chunk's own
//! face, to a PARTNER face across a seam, or to no face at all (a corner phantom). That is a
//! LOOKUP, not arithmetic, and it costs a few thousand integer comparisons per box against a
//! quarter of a million cells. So the host answers it once and the kernel does arithmetic only.
//! MEASURED: 0.056 ms of a 2.01 ms box, against 2.14 ms before step G2-A.
//!
//! **The one word per cell.** The kernel writes the substance code in a byte and the gap byte above
//! it ([`vd_recipe::cell::pack`]). [`BoxPlan::box_of`] turns a readback into the same
//! [`SampleBox`] the CPU builds, which the extractor then reads without knowing which host made it.
//! [`BoxPlan::run`] is the same two passes ON THIS HOST — the reference a card is measured against,
//! and the fallback path on a card the self-check refuses.
//!
//! **Example.** The client wants chunk (face 2, rung 0, 19, 1, 4) of the home planet. The worker
//! builds its plan — 64 layers, 4 096 sites, two lattice extents, the three tube carvers that reach
//! it — uploads about 80 kB, and reads back 262 144 words: the same 262 144 words the shard's CPU
//! writes for the same key, which is why the ground the player walks on is the ground the card
//! draws.

use crate::body::BodyDefinition;
use crate::carve::CAVERN_STRIDE;
use crate::chunk::{CHUNK_EDGE, ChunkKey, cell_of_word, in_ladder};
use crate::lattice::{BOX_EDGE, HALO, SampleBox, Site, box_topology};
use crate::units::LENGTH_BITS;
use vd_recipe::Gi;
use vd_recipe::cell::{
    CellAt, CellCharter, Column, LAYER_ABOVE, LAYER_BELOW, LAYER_EVALUATED, Layer, Tube,
    above_cell_word, below_cell_word, cavern_at, cell_word,
};
use vd_recipe::height::{OCTAVES_CAP, Octave};
use vd_recipe::plan::{NodeBlock, PlanCharter, column_row, node_of, node_value, weight_of};

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
/// The 32-bit words of one column's SITE: the face, the two cell indices and one of padding.
pub const SITE_WORDS: usize = 4;
/// The words of the PLAN charter: eleven of its own, then the biome's two octaves and the body's
/// octave table, [`OCTAVE_WORDS`] words each.
pub const PLAN_CHARTER_WORDS: usize =
    11 + (2 + OCTAVES_CAP) * OCTAVE_WORDS + ROUGHNESS_WORDS + TERRACE_WORDS;
/// The words of one octave: SIX of its own (the `fine` mask joined at slice 8a stage 3) and two of
/// explicit padding, so the row's stride stays 64 bytes. The charter's octave block holds
/// 1 024 bytes, not 512.
pub const OCTAVE_WORDS: usize = 8;

/// The words of the ROUGHNESS row: one octave and two of its own (slice 8a stage 3).
pub const ROUGHNESS_WORDS: usize = OCTAVE_WORDS + 2;

/// The words of the CAP-ROCK BENCH's row: five of its own and three of explicit padding, so the
/// row's stride is 64 bytes (slice 8a stage 4).
pub const TERRACE_WORDS: usize = 8;

/// The bench's row as [`TERRACE_WORDS`] words, in the `repr(C)` order the column pass reads.
/// ★ ONE WRITER, like [`octave_words`] and [`roughness_words`]: a second packer of this row would
/// read half a row the day the row grows, which is exactly what happened when the `kind` word
/// landed.
#[must_use]
pub fn terrace_words(t: &vd_recipe::terrace::Terrace) -> [i64; TERRACE_WORDS] {
    [
        t.datum.raw(),
        t.spacing.raw(),
        t.spacing_recip.raw(),
        t.half_recip.raw(),
        t.strength.raw(),
        t.pad[0].raw(),
        t.pad[1].raw(),
        t.pad[2].raw(),
    ]
}

/// The roughness row as [`ROUGHNESS_WORDS`] words, in the `repr(C)` order the column pass reads.
/// ★ ONE WRITER, like [`octave_words`]: a second packer of this row would read half a row the day
/// the row grows, which is exactly what happened when the `kind` word landed.
#[must_use]
pub fn roughness_words(r: &vd_recipe::height::Roughness) -> [i64; ROUGHNESS_WORDS] {
    let o = octave_words(&r.octave);
    [
        o[0],
        o[1],
        o[2],
        o[3],
        o[4],
        o[5],
        o[6],
        o[7],
        r.m_min.raw(),
        r.first_fine.raw(),
    ]
}

/// One octave as [`OCTAVE_WORDS`] words, in the `repr(C)` order the kernel reads. ★ PUBLIC, because
/// the client's GPU self-check hands the very same rows to the shell's own relief kernel: two
/// writers of one row would drift the moment the row grew, which is exactly what happened when the
/// `kind` word landed.
#[must_use]
pub fn octave_words(o: &Octave) -> [i64; OCTAVE_WORDS] {
    [
        o.seed as i64,
        o.frequency_int.raw(),
        o.frequency_frac.raw(),
        o.amplitude.raw(),
        o.kind.raw(),
        o.fine.raw(),
        o.pad[0].raw(),
        o.pad[1].raw(),
    ]
}

/// The words of one cavern lattice's row.
pub const BLOCK_WORDS: usize = 8;

/// ONE BOX, as words: what the three compute passes read and what they write back into. Step G2-A
/// made this TOPOLOGY ONLY — no direction, no surface, no biome, no node value. The card computes
/// those itself from [`BoxPlan::plan`] and the sites below.
pub struct BoxPlan {
    pub key: ChunkKey,
    /// The body's charter at this rung, with the box's own edge — what the CELL pass reads.
    pub charter: CellCharter,
    /// The body's charter for the COLUMN pass and the NODE pass.
    pub plan: PlanCharter,
    /// One row per radial layer of the box, in the box's own order (`c` from the halo up).
    pub layers: Vec<Layer>,
    /// One site per column of the box, in packing order `(b + 1) · 64 + (a + 1)`: which face the
    /// column belongs to and its cell there, or a corner phantom.
    pub sites: Vec<Site>,
    /// The box's cavern lattices, the chunk's own face first. Never empty: where a rung carves no
    /// cavern one block of NO FACE stands in, which no column can name, so no column has a lattice.
    pub lattices: Vec<NodeBlock>,
    /// One row per slice of the node dispatch: which lattice it belongs to and its radial node.
    pub node_z: Vec<[Gi; 2]>,
    /// The corner radius of each radial node index, in whole gap steps. Every lattice of a box
    /// shares one radial range, so one row serves them all.
    pub node_radii: Vec<Gi>,
    /// How many words the node buffer holds; never zero, because a card refuses a buffer of no bytes.
    pub node_count: usize,
    /// The node dispatch's extent along the two face axes: the widest lattice of the box.
    pub node_extent: [u32; 2],
    /// The carvers that can reach the box. Never empty for the same reason: a carver of NO RADIUS
    /// stands in, and `radius − distance` is never above zero for it, so it hollows nothing.
    pub tubes: Vec<Tube>,
}

/// THE TWO PASSES' ANSWER: what a host or a card writes into the column buffer and the node buffer
/// before the cell pass reads them.
pub struct BoxRun {
    /// One row per column, whole: the direction, the surface, the biome and the lattice indices.
    pub columns: Vec<Column>,
    /// Every lattice's node values, laid end to end.
    pub nodes: Vec<Gi>,
}

/// The plan of the box at `key`; `None` for a key outside the body's ladder.
#[must_use]
pub fn plan(body: &BodyDefinition, key: ChunkKey) -> Option<BoxPlan> {
    if !in_ladder(body, key) {
        return None;
    }
    let edge = CHUNK_EDGE as i32;
    let rung = key.rung;
    let topology = box_topology(body, key);

    // The lattices, laid end to end: the chunk's own face first, then each partner face present.
    let mut lattices = Vec::with_capacity(topology.lattices.len());
    let mut node_z: Vec<[Gi; 2]> = Vec::new();
    let mut node_count = 0usize;
    let mut node_extent = [0u32; 2];
    for (index, e) in topology.lattices.iter().enumerate() {
        lattices.push(NodeBlock {
            face: Gi::new(i64::from(e.face.index())),
            base: Gi::new(node_count as i64),
            node0: [
                Gi::new(i64::from(e.node0[0])),
                Gi::new(i64::from(e.node0[1])),
                Gi::new(i64::from(e.node0[2])),
            ],
            dims: [
                Gi::new(e.dims[0] as i64),
                Gi::new(e.dims[1] as i64),
                Gi::new(e.dims[2] as i64),
            ],
        });
        let mut nc = 0;
        while nc < e.dims[2] {
            node_z.push([Gi::new(index as i64), Gi::new(nc as i64)]);
            nc += 1;
        }
        node_count += e.dims[0] * e.dims[1] * e.dims[2];
        node_extent[0] = node_extent[0].max(e.dims[0] as u32);
        node_extent[1] = node_extent[1].max(e.dims[1] as u32);
    }
    if lattices.is_empty() {
        // NO FACE: a block no site can name, so every column reads `has` as zero and carves no cave.
        // One spare word stands in for the node buffer, which a card refuses to create empty.
        lattices.push(NodeBlock {
            face: Gi::new(NO_FACE),
            ..Default::default()
        });
        node_z.push([Gi::ZERO; 2]);
        node_count = 1;
        node_extent = [1, 1];
    }
    // The corner radius of each radial node index — one row for every lattice of the box, because
    // every lattice of a box stands on the same radial range.
    let k_dims = topology.lattices.first().map_or(1, |e| e.dims[2]);
    let mut node_radii = Vec::with_capacity(k_dims);
    let mut nc = 0;
    while nc < k_dims {
        let k = (topology.node0[2] + nc as i32) * CAVERN_STRIDE as i32;
        node_radii.push(Gi::new(body.ladder().corner_radius_steps(k, rung)));
        nc += 1;
    }

    // One row per radial layer: its cell radius, its rule, and its own node and weight.
    let mut layers = Vec::with_capacity(BOX_EDGE);
    let mut c = -HALO;
    while c <= edge {
        let k = topology.k0 + c;
        let rule = if k < 0 {
            LAYER_BELOW
        } else if k >= topology.band {
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
                node_of(k) - Gi::new(i64::from(topology.node0[2]))
            } else {
                Gi::ZERO
            },
            weight: if evaluated { weight_of(k) } else { Gi::ZERO },
        });
        c += 1;
    }

    let mut tubes = topology.tubes;
    if tubes.is_empty() {
        tubes.push(Tube::default());
    }
    Some(BoxPlan {
        key,
        charter: topology.charter,
        plan: body.plan_charter(rung, key.face),
        layers,
        sites: topology.sites,
        lattices,
        node_z,
        node_radii,
        node_count,
        node_extent,
        tubes,
    })
}

/// The face word of the stand-in lattice block: no face at all, so no column can name it.
const NO_FACE: i64 = -1;

impl BoxPlan {
    /// ★ THE TWO PASSES RUN ON THIS HOST — the column pass and the node pass, through the very
    /// kernels the card runs. The CPU's reference for a card's answer, and the fallback path on a
    /// card the self-check refuses.
    #[must_use]
    pub fn run(&self) -> BoxRun {
        let mut columns = Vec::with_capacity(self.sites.len());
        let mut i = 0;
        while i < self.sites.len() {
            let s = self.sites[i];
            columns.push(column_row(
                &self.plan,
                &self.lattices,
                i32::from(s.face),
                s.i,
                s.j,
            ));
            i += 1;
        }
        let mut nodes = Vec::with_capacity(self.node_count);
        for block in &self.lattices {
            if block.face.raw() == NO_FACE {
                nodes.push(Gi::ZERO);
                continue;
            }
            let mut nc = 0;
            while nc < block.dims[2].raw() {
                let r = self.node_radii[nc as usize];
                let mut nb = 0;
                while nb < block.dims[1].raw() {
                    let mut na = 0;
                    while na < block.dims[0].raw() {
                        nodes.push(node_value(
                            &self.plan,
                            block.face.raw() as i32,
                            ((block.node0[0].raw() + na) * i64::from(CAVERN_STRIDE as i32)) as i32,
                            ((block.node0[1].raw() + nb) * i64::from(CAVERN_STRIDE as i32)) as i32,
                            r,
                        ));
                        na += 1;
                    }
                    nb += 1;
                }
                nc += 1;
            }
        }
        BoxRun { columns, nodes }
    }

    /// THE CELL PASS on this host: every cell's word, in the box's packing order.
    #[must_use]
    pub fn cells_of(&self, run: &BoxRun) -> Vec<u32> {
        let edge = self.charter.box_edge.raw() as usize;
        let mut out = Vec::with_capacity(edge * edge * edge);
        let mut c = 0;
        while c < edge {
            let mut b = 0;
            while b < edge {
                let mut a = 0;
                while a < edge {
                    out.push(self.cell(run, a, b, c));
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
    pub fn cell(&self, run: &BoxRun, a: usize, b: usize, c: usize) -> u32 {
        let edge = self.charter.box_edge.raw() as usize;
        let layer = &self.layers[c];
        let column = &run.columns[b * edge + a];
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
                cavern_at(column, layer, &run.nodes),
                &self.tubes,
            ),
        }
    }

    /// The box a readback names: the same [`SampleBox`] the CPU's own cell pass builds, which the
    /// extractor reads without ever learning which host wrote the cells. `dirs` is the column pass's
    /// own answer — three words a column, from this host's run or from the card's.
    #[must_use]
    pub fn box_of(&self, words: &[u32], dirs: &[[Gi; 3]]) -> SampleBox {
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
            dirs: dirs.to_vec(),
        }
    }

    /// The column pass's directions, as [`BoxPlan::box_of`] wants them.
    #[must_use]
    pub fn dirs_of(run: &BoxRun) -> Vec<[Gi; 3]> {
        run.columns.iter().map(|c| c.dir).collect()
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

    /// The PLAN charter as [`PLAN_CHARTER_WORDS`] words, in the `repr(C)` order the two passes read.
    #[must_use]
    pub fn plan_charter_words(&self) -> Vec<i64> {
        let p = &self.plan;
        let b = &p.biome;
        let mut w = vec![
            p.seed as i64,
            p.cavern_recip.raw(),
            p.cavern_shift.raw(),
            p.inv_n.raw(),
            p.radius.raw(),
            p.octave_count.raw(),
            p.key_face.raw(),
            b.sea_radius.raw(),
            b.highland_above.raw(),
            b.highland_recip.raw(),
            b.highland_shift.raw(),
        ];
        for o in [&b.temperature, &b.humidity] {
            w.extend(octave_words(o));
        }
        w.extend(roughness_words(&p.roughness));
        w.extend(terrace_words(&p.terrace));
        let mut i = 0;
        while i < p.octaves.len() {
            w.extend(octave_words(&p.octaves[i]));
            i += 1;
        }
        w
    }

    /// The columns' SITES as [`SITE_WORDS`] 32-bit words each: the face, the cell's two indices and
    /// one word of padding, so a row is four words wide and the kernel indexes it by a shift.
    #[must_use]
    pub fn site_words(&self) -> Vec<i32> {
        let mut w = Vec::with_capacity(self.sites.len() * SITE_WORDS);
        let mut i = 0;
        while i < self.sites.len() {
            let s = self.sites[i];
            w.extend([i32::from(s.face), s.i, s.j, 0]);
            i += 1;
        }
        w
    }

    /// The cavern lattices as [`BLOCK_WORDS`] words each.
    #[must_use]
    pub fn lattice_words(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.lattices.len() * BLOCK_WORDS);
        let mut i = 0;
        while i < self.lattices.len() {
            let b = &self.lattices[i];
            w.extend([
                b.face.raw(),
                b.base.raw(),
                b.node0[0].raw(),
                b.node0[1].raw(),
                b.node0[2].raw(),
                b.dims[0].raw(),
                b.dims[1].raw(),
                b.dims[2].raw(),
            ]);
            i += 1;
        }
        w
    }

    /// The node dispatch's slice table as two words each: the lattice and the radial node index.
    #[must_use]
    pub fn node_z_words(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.node_z.len() * 2);
        let mut i = 0;
        while i < self.node_z.len() {
            w.extend([self.node_z[i][0].raw(), self.node_z[i][1].raw()]);
            i += 1;
        }
        w
    }

    /// The radial nodes' corner radii as one word each.
    #[must_use]
    pub fn node_radius_words(&self) -> Vec<i64> {
        let mut w = Vec::with_capacity(self.node_radii.len());
        let mut i = 0;
        while i < self.node_radii.len() {
            w.push(self.node_radii[i].raw());
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
            key(
                Face::PosX,
                0,
                300,
                700,
                surface_chunk_z(&m, Face::PosX, 0, 300, 700),
            ),
            key(
                Face::PosX,
                0,
                last,
                last,
                surface_chunk_z(&m, Face::PosX, 0, last, last),
            ),
            key(
                Face::PosZ,
                3,
                5,
                7,
                surface_chunk_z(&m, Face::PosZ, 3, 5, 7),
            ),
            key(Face::PosX, 0, 300, 700, 0),
            // ★ THE TOP CHUNK OF THE BAND, read from the ladder itself: the LAST chunk index the
            // band holds, `(cells_in_band − 1) / CHUNK_EDGE`, so the box's halo reaches OVER the
            // band's top and the plan lays a LAYER_ABOVE row down. One less than that — which is
            // what this fixture asked for until the ladder grew at Step 17 — sits wholly inside the
            // band, and the whole `LAYER_ABOVE` path then goes unwalked on this host and on the card.
            key(
                Face::PosX,
                0,
                300,
                700,
                (m.ladder().cells_in_band(0) as i32 - 1) / CHUNK_EDGE as i32,
            ),
        ] {
            let plan = plan(&m, k).expect("the key is on the ladder");
            let run = plan.run();
            let words = plan.cells_of(&run);
            assert_eq!(words.len(), BOX_CELLS);
            assert_eq!(run.nodes.len(), plan.node_count, "{k:?}: the node count");
            let want = sample_box(&m, None, k).expect("the box");
            let got = plan.box_of(&words, &BoxPlan::dirs_of(&run));
            assert_eq!(got.cells, want.cells, "{k:?}: the cells");
            assert_eq!(got.sites, want.sites, "{k:?}: the sites");
            assert_eq!(got.dirs, want.dirs, "{k:?}: the directions");
            assert_eq!(got.key, want.key);
        }
        assert!(
            plan(&m, key(Face::PosX, 0, -1, 0, 0)).is_none(),
            "off the ladder"
        );
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
        let edge_run = edge_plan.run();
        assert!(
            edge_plan.lattices.len() > 1,
            "the box lays a partner face's lattice down"
        );
        assert!(
            edge_run.columns.iter().any(|c| c.base > Gi::ZERO),
            "a column reads a partner face's lattice"
        );
        assert!(
            edge_run.columns.iter().any(|c| c.has == Gi::ZERO),
            "a corner phantom has no lattice"
        );
        // Every slice of the node dispatch names a lattice the plan holds, and the radii cover the
        // radial range every lattice of the box shares.
        for z in &edge_plan.node_z {
            assert!((z[0].raw() as usize) < edge_plan.lattices.len());
            assert!((z[1].raw() as usize) < edge_plan.node_radii.len());
        }
        assert_eq!(
            edge_plan.node_z.len(),
            edge_plan
                .lattices
                .iter()
                .map(|b| b.dims[2].raw() as usize)
                .sum::<usize>()
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
        assert_eq!(size_of::<PlanCharter>(), PLAN_CHARTER_WORDS * 8);
        assert_eq!(size_of::<vd_recipe::terrace::Terrace>(), TERRACE_WORDS * 8);
        assert_eq!(size_of::<NodeBlock>(), BLOCK_WORDS * 8);
        assert_eq!(size_of::<Octave>(), OCTAVE_WORDS * 8);
        let m = home_planet();
        let k = key(
            Face::PosX,
            0,
            300,
            700,
            surface_chunk_z(&m, Face::PosX, 0, 300, 700),
        );
        let plan = plan(&m, k).expect("the key is on the ladder");
        assert_eq!(plan.charter_words().len(), CHARTER_WORDS);
        assert_eq!(plan.plan_charter_words().len(), PLAN_CHARTER_WORDS);
        assert_eq!(plan.layer_words().len(), BOX_EDGE * LAYER_WORDS);
        assert_eq!(plan.site_words().len(), BOX_EDGE * BOX_EDGE * SITE_WORDS);
        assert_eq!(plan.tube_words().len(), plan.tubes.len() * TUBE_WORDS);
        assert_eq!(
            plan.lattice_words().len(),
            plan.lattices.len() * BLOCK_WORDS
        );
        assert_eq!(plan.node_z_words().len(), plan.node_z.len() * 2);
        assert_eq!(plan.node_radius_words().len(), plan.node_radii.len());
        assert!(plan.node_count > 0, "a card refuses an empty buffer");
        assert!(!plan.tubes.is_empty());
        assert!(!plan.lattices.is_empty());
        // The charters' own words, read back in the order the kernels read them.
        let w = plan.charter_words();
        assert_eq!(w[0], plan.charter.sea_radius.raw());
        assert_eq!(w[13], i64::from(BOX_EDGE as u32));
        assert_eq!(w[CHARTER_WORDS - 1], plan.charter.strata[3].raw());
        let p = plan.plan_charter_words();
        assert_eq!(p[0], plan.plan.seed as i64);
        assert_eq!(p[6], plan.plan.key_face.raw());
        assert_eq!(p[10], plan.plan.biome.highland_shift.raw());
        assert_eq!(p[11], plan.plan.biome.temperature.seed as i64);
        // ★ THE ROUGHNESS ROW at its own place: the head's eleven words, the biome's two octaves,
        // then the placeholder octave and the factor's two words (slice 8a stage 3). A row whose
        // stride slipped would read the first octave of the table here.
        let rough_at = 11 + 2 * OCTAVE_WORDS;
        assert_eq!(p[rough_at], plan.plan.roughness.octave.seed as i64);
        assert_eq!(
            p[rough_at + 3],
            plan.plan.roughness.octave.amplitude.raw(),
            "the placeholder's amplitude is one noise unit"
        );
        assert_eq!(p[rough_at + OCTAVE_WORDS], plan.plan.roughness.m_min.raw());
        assert_eq!(
            p[rough_at + OCTAVE_WORDS + 1],
            plan.plan.roughness.first_fine.raw()
        );
        // ★ THE CAP-ROCK BENCH's row next (slice 8a stage 4), then the octave table. A row whose
        // stride slipped would read the first octave of the table at the bench's datum.
        let bench_at = rough_at + ROUGHNESS_WORDS;
        assert_eq!(p[bench_at], plan.plan.terrace.datum.raw());
        assert_eq!(p[bench_at + 1], plan.plan.terrace.spacing.raw());
        assert_eq!(p[bench_at + 2], plan.plan.terrace.spacing_recip.raw());
        assert_eq!(p[bench_at + 3], plan.plan.terrace.half_recip.raw());
        assert_eq!(
            p[bench_at + 4],
            m.terrace_at(k.rung).strength.raw(),
            "the bench's strength is the RUNG's own"
        );
        assert_eq!(p[bench_at + 5], 0, "the bench's padding is zero");
        assert_eq!(
            p[bench_at + TERRACE_WORDS],
            plan.plan.octaves[0].seed as i64,
            "the octave table begins right after the bench's row"
        );
        // The LAST octave's own five words, at their places in the block, and the three of padding
        // that follow them: a row whose stride slipped would read the next octave's seed here.
        let last = PLAN_CHARTER_WORDS - OCTAVE_WORDS;
        let tail = &plan.plan.octaves[OCTAVES_CAP - 1];
        assert_eq!(p[last], tail.seed as i64);
        assert_eq!(p[last + 3], tail.amplitude.raw());
        assert_eq!(p[last + 4], tail.kind.raw());
        assert_eq!(p[PLAN_CHARTER_WORDS - 1], 0, "the row's padding is zero");
        // The site rows, read back as the kernel reads them.
        let s = plan.site_words();
        assert_eq!(s[0], i32::from(plan.sites[0].face));
        assert_eq!(s[1], plan.sites[0].i);
        assert_eq!(s[2], plan.sites[0].j);
    }

    /// ★ A COARSE RUNG CARVES NO CAVES, so no column has a lattice and the spare word stands in —
    /// and the cells are still the cell pass's own.
    #[test]
    fn a_rung_without_caves_plans_no_lattice_and_still_names_the_same_cells() {
        let m = home_planet();
        let rung = 8u8;
        let k = key(
            Face::PosY,
            rung,
            1,
            1,
            surface_chunk_z(&m, Face::PosY, rung, 1, 1),
        );
        let plan = plan(&m, k).expect("the key is on the ladder");
        let run = plan.run();
        assert_eq!(plan.node_count, 1, "one spare word, no lattice");
        assert_eq!(run.nodes.len(), 1);
        assert_eq!(plan.lattices.len(), 1, "one block of NO FACE");
        assert_eq!(plan.lattices[0].face.raw(), NO_FACE);
        assert!(run.columns.iter().all(|c| c.has == Gi::ZERO));
        assert_eq!(plan.tubes.len(), 1, "one carver of no radius");
        assert_eq!(plan.tubes[0], Tube::default());
        let want = sample_box(&m, None, k).expect("the box");
        assert_eq!(
            plan.box_of(&plan.cells_of(&run), &BoxPlan::dirs_of(&run))
                .cells,
            want.cells
        );
    }
}
