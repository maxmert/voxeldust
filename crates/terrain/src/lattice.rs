//! ★ THE SAMPLE BOX — a chunk plus its HALO: the `64 × 64 × 64` cells the extractor reads, the
//! chunk's own `62³` in the middle and one cell of every neighbour around it (ruling V10 S6-2).
//!
//! **The halo is GENERATED with the chunk, never fetched.** A neighbour cell is computed by the same
//! per-cell rule the neighbour's own cell pass runs — the same column function, the same global
//! cavern nodes, the same tubes — so it is byte-identical to the neighbour's copy, and the extraction
//! of one key is a function of `(seed, key)` and nothing else. A test compares a halo strip against
//! the neighbour chunk's core, cell for cell, on the same face, across a face edge and at a cube
//! corner.
//!
//! **Across a face edge** a halo cell is the partner face's own cell, found by THE crossing rule of
//! the leaf (`vd_seed::seam::across`): the same rule the grid uses to step across a seam.
//!
//! **At a cube corner** three faces meet and the fourth cell of a `2 × 2` ring does not exist. The box
//! holds a PHANTOM column there — a placeholder on the corner direction (`(±1, ±1, ±1)/√3`) filled
//! by the surface rule with no cave — so the box stays a full `64³` array. The extractor never
//! reads a phantom: the corner group is a PRISM of the three real columns (`extract::prism_vertex`),
//! and no chunk owns an edge that touches the phantom. Each face's box lays the same three real
//! columns out in its own local axes; the three layouts are affine images of one another, which is
//! what lets the three chunks compute one corner vertex (`position`).
//!
//! **Below the band's floor** the halo is bedrock at the bottom code; **above the band's top** it is
//! air (or water under the sea) at the top code — what the band was derived to contain.
//!
//! **Example.** The chunk at the +u edge of face `+X` on the home planet, rung 0: its halo column at
//! local `a = 62` is face `+Y`'s column `(j, n − 1)` — the same hill face `+Y`'s own chunk computes,
//! so the two meshes meet on the cube edge without a crack.

use crate::body::BodyDefinition;
use crate::carve::{CAVERN_STRIDE_LOG2, Tube, caverns_carve_at, tubes_carve_at};
pub use crate::chunk::in_ladder;
use crate::chunk::{
    CAVERN_NODES, CHUNK_EDGE, Cell, CellSite, ChunkKey, ColumnField, NodeLattice,
    above_surface_cell, below_surface_cell, cavern_of, charter_of, column_field, finish_cell,
    foreign_extents, generate_in, point_at, tubes_reaching,
};
use crate::strata::Biome;
use crate::units::LENGTH_BITS;
use vd_recipe::Gi;
use vd_recipe::cell::CellCharter;
use vd_recipe::plan::site_direction_of;
use vd_seed::bend::Face;
use vd_seed::seam::{Edge, across};

/// The halo's depth in cells on every side.
pub const HALO: i32 = 1;
/// Cells per box edge: the chunk and its halo.
pub const BOX_EDGE: usize = CHUNK_EDGE + 2;
/// Cells per box.
pub const BOX_CELLS: usize = BOX_EDGE * BOX_EDGE * BOX_EDGE;
/// The face byte of a corner phantom's site: no face. The recipe's column kernel reads the same
/// byte, and this assertion makes a change on one side a red BUILD rather than a phantom the card
/// draws as a face cell.
pub const CORNER_FACE: u8 = 255;
const _: () = assert!(CORNER_FACE as i32 == vd_recipe::plan::CORNER_FACE);

/// Where a box column lives: a cell `(i, j)` of a face, or a corner phantom (`face` is
/// [`CORNER_FACE`], and `i`, `j` hold the corner's signs along the chunk's own `u` and `v`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Site {
    pub face: u8,
    pub i: i32,
    pub j: i32,
}

/// The chunk and its halo, with each column's site and direction.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleBox {
    pub key: ChunkKey,
    /// `BOX_CELLS` cells; local `(a, b, c)` in `−1..=62` at [`SampleBox::index`].
    pub cells: Vec<Cell>,
    /// `BOX_EDGE²` column sites, `(b + 1) · 64 + (a + 1)`.
    pub sites: Vec<Site>,
    /// The same columns' unit directions, at the bend's fraction bits.
    pub dirs: Vec<[Gi; 3]>,
    /// ★ The same columns' WATER surface radius (slice 8c stage C5) at [`LENGTH_BITS`], or ZERO
    /// for a dry column — what the client's water sheet stands on.
    pub water: Vec<Gi>,
    /// ★ THE SAME COLUMNS' GROUND radius at [`LENGTH_BITS`] (2026-09-21, the water sheet's cut):
    /// what the sheet's builder reads to leave out a quad buried under the land in every state the
    /// ladder can draw. ZERO where the host holds no height for the column (the card's readback
    /// carries the directions alone), which keeps the quad.
    pub surfaces: Vec<Gi>,
    /// ★ THE BODY'S SEA as a radius at [`LENGTH_BITS`], ZERO for a body with none (2026-09-20, the
    /// owner: "the form of the shores is changing all the time"): the sea is ONE surface under
    /// every cell of the body, so the shore is the land's own crossing of it and follows the land's
    /// morph through every rung; a sheet drawn only where a column was wet grew and shrank with
    /// the rung's cell, and its edge stepped at every ring swap.
    pub sea: Gi,
}

impl SampleBox {
    /// The index of local `(a, b, c)`, each in `−1..=62`.
    #[must_use]
    pub fn index(a: i32, b: i32, c: i32) -> usize {
        let (a, b, c) = (
            (a + HALO) as usize,
            (b + HALO) as usize,
            (c + HALO) as usize,
        );
        (c * BOX_EDGE + b) * BOX_EDGE + a
    }

    /// The index of local column `(a, b)`.
    #[must_use]
    pub fn column_index(a: i32, b: i32) -> usize {
        ((b + HALO) as usize) * BOX_EDGE + (a + HALO) as usize
    }

    /// The cell at local `(a, b, c)`.
    #[must_use]
    pub fn cell(&self, a: i32, b: i32, c: i32) -> Cell {
        self.cells[Self::index(a, b, c)]
    }

    /// The site of local column `(a, b)`.
    #[must_use]
    pub fn site(&self, a: i32, b: i32) -> Site {
        self.sites[Self::column_index(a, b)]
    }

    /// The direction of local column `(a, b)`.
    #[must_use]
    pub fn dir(&self, a: i32, b: i32) -> [Gi; 3] {
        self.dirs[Self::column_index(a, b)]
    }

    /// Whether local `(a, b, c)` is one of the chunk's own cells.
    #[must_use]
    pub fn is_core(a: i32, b: i32, c: i32) -> bool {
        let edge = CHUNK_EDGE as i32;
        (a >= 0) & (a < edge) & (b >= 0) & (b < edge) & (c >= 0) & (c < edge)
    }
}

/// The site of the local column `(a, b)` of a chunk: its own face's cell, the partner face's cell
/// across the seam, or a corner phantom.
#[must_use]
pub fn site_of(body: &BodyDefinition, key: ChunkKey, a: i32, b: i32) -> Site {
    let n = body.ladder.cells_per_edge(key.rung) as i32;
    let edge = CHUNK_EDGE as i32;
    let (gi, gj) = (key.x * edge + a, key.y * edge + b);
    let in_i = (gi >= 0) & (gi < n);
    let in_j = (gj >= 0) & (gj < n);
    if in_i & in_j {
        return Site {
            face: key.face.index(),
            i: gi,
            j: gj,
        };
    }
    if in_i | in_j {
        // The side crossed, the along-edge position, and how far past the edge (1 for the halo,
        // more for a PARTIAL chunk's cells beyond its face).
        let (side, along, over) = if !in_i {
            if gi < 0 {
                (Edge::UMinus, gj, -gi)
            } else {
                (Edge::UPlus, gj, gi - n + 1)
            }
        } else if gj < 0 {
            (Edge::VMinus, gi, -gj)
        } else {
            (Edge::VPlus, gi, gj - n + 1)
        };
        let c = across(key.face, side, along, n);
        // The partner's edge cell, then `over − 1` cells inward from its edge.
        let rec = vd_seed::seam::partner_of(key.face, side);
        let inward = if rec.dst_edge.is_plus() {
            -(over - 1)
        } else {
            over - 1
        };
        let (i, j) = if rec.dst_edge.is_u() {
            (c.i + inward, c.j)
        } else {
            (c.i, c.j + inward)
        };
        return Site {
            face: c.face.index(),
            i,
            j,
        };
    }
    Site {
        face: CORNER_FACE,
        i: if gi < 0 { -1 } else { 1 },
        j: if gj < 0 { -1 } else { 1 },
    }
}

/// The local box column `(a, b)` of a chunk that holds `site`, if the box holds it — the inverse of
/// [`site_of`] over the box (`−1..=62` per axis), so a host that holds a neighbour chunk's edits
/// can lay them onto this chunk's halo (`compose`). A corner phantom is nobody's cell: `None`.
#[must_use]
pub fn local_of_site(body: &BodyDefinition, key: ChunkKey, site: Site) -> Option<(i32, i32)> {
    let n = body.ladder.cells_per_edge(key.rung) as i32;
    let edge = CHUNK_EDGE as i32;
    let (i0, j0) = (key.x * edge, key.y * edge);
    let inside = |a: i32, b: i32| (a >= -HALO) & (a <= edge) & (b >= -HALO) & (b <= edge);
    if site.face == key.face.index() {
        let (a, b) = (site.i - i0, site.j - j0);
        return if inside(a, b) { Some((a, b)) } else { None };
    }
    // Across each of the four sides: undo the crossing rule, then confirm by the forward map.
    for side in Edge::ALL {
        let rec = vd_seed::seam::partner_of(key.face, side);
        if rec.dst_face.index() != site.face {
            continue;
        }
        // The partner's index across its edge gives how far past our edge the cell is; the other
        // index is the along-edge position.
        let (across_index, along) = if rec.dst_edge.is_u() {
            (site.i, site.j)
        } else {
            (site.j, site.i)
        };
        let over = if rec.dst_edge.is_plus() {
            n - across_index
        } else {
            across_index + 1
        };
        let (a, b) = match side {
            Edge::UMinus => (-over, along - j0),
            Edge::UPlus => (n - i0 + over - 1, along - j0),
            Edge::VMinus => (along - i0, -over),
            Edge::VPlus => (along - i0, n - j0 + over - 1),
        };
        if inside(a, b) & (site_of(body, key, a, b) == site) {
            return Some((a, b));
        }
    }
    None
}

/// The unit direction of a site, at the bend's fraction bits: a face cell's own direction, or the
/// corner's. A corner phantom's axis sum is `(±1, ±1, ±1)` at the bend's own One, and the recipe's
/// [`normalise`] puts it on the sphere — the same kernel the bend itself ends with, so the three faces
/// that meet at the corner read the same word triple.
///
/// ★ ONE SOURCE (step G2-A): the arithmetic is `vd_recipe::plan::site_direction_of`, the kernel the
/// card's COLUMN PASS runs for every column of a box — its own face's cells and the phantoms alike.
#[must_use]
pub fn site_dir(body: &BodyDefinition, key: ChunkKey, site: Site) -> [Gi; 3] {
    site_direction_of(
        body.inv_n(key.rung),
        i32::from(key.face.index()),
        i32::from(site.face),
        site.i,
        site.j,
    )
}

/// ★ EVERYTHING A BOX NEEDS BEFORE ITS CELLS: each column's site, direction, surface and biome, the
/// tube carvers that can reach the box, the cavern lattices its columns read, and the body's charter
/// at this rung. The CPU's own cell pass ([`sample_box`]) and the GPU's plan (`crate::gpu`) both
/// start here, so the card and the shard read ONE box, never two that happen to agree.
pub(crate) struct BoxSetup {
    /// `BOX_EDGE²` columns in packing order `(b + 1) · 64 + (a + 1)`.
    pub sites: Vec<Site>,
    pub dirs: Vec<[Gi; 3]>,
    pub surfaces: Vec<(Gi, Biome)>,
    /// Each column's water surface radius, or ZERO (C5).
    pub water: Vec<Gi>,
    /// ★ Each column's ROCK PROVINCE (slice 8d step 2), the halo's included.
    pub province: Vec<Gi>,
    /// The carvers that can reach the box: a SUPERSET of what any one cell's owner keeps, and a
    /// hollow is the exact greatest over the list, so the superset changes no byte.
    pub tubes: Vec<Tube>,
    /// The chunk's own face's node lattice, and one per partner face present among the columns.
    pub own: NodeLattice,
    pub foreign: Vec<NodeLattice>,
    pub charter: CellCharter,
    pub carve_caverns: bool,
    /// The box's radial range.
    pub k0: i32,
    pub band: i32,
}

/// ★ THE TOPOLOGY OF ONE BOX — everything about it that is LOOKUP rather than arithmetic: which
/// face each column belongs to, which carvers reach it, which cavern lattices its columns read, and
/// the body's charter at this rung. This is what a HOST still does when the card runs the column
/// pass and the node pass (step G2-A): it costs a few thousand integer comparisons against a
/// quarter of a million cells of arithmetic.
///
/// **Example.** The chunk at the far corner of face `+X`: 4 096 sites, of which some belong to face
/// `+Y` and four are corner phantoms; two lattices; the three tube carvers that reach the box.
pub(crate) struct BoxTopology {
    pub sites: Vec<Site>,
    /// The carvers that can reach the box: a SUPERSET of what any one cell's owner keeps, and a
    /// hollow is the exact greatest over the list, so the superset changes no byte.
    pub tubes: Vec<Tube>,
    /// The box's cavern lattices, the chunk's own face first: EMPTY at a rung that carves no cavern.
    pub lattices: Vec<LatticeExtent>,
    pub charter: CellCharter,
    pub carve_caverns: bool,
    pub node0: [i32; 3],
    pub k0: i32,
    pub band: i32,
}

/// WHERE one cavern lattice of a box stands: its face, its first node along each axis, and how many
/// nodes it holds along each. The values themselves are the node pass's answer, on either host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LatticeExtent {
    pub face: Face,
    pub node0: [i32; 3],
    pub dims: [usize; 3],
}

/// The topology of the box at `key`.
pub(crate) fn box_topology(body: &BodyDefinition, key: ChunkKey) -> BoxTopology {
    let rung = key.rung;
    let edge = CHUNK_EDGE as i32;
    let band = body.ladder.cells_in_band(rung) as i32;
    let k0 = key.z * edge;
    let mut sites = Vec::with_capacity(BOX_EDGE * BOX_EDGE);
    let mut b = -HALO;
    while b <= edge {
        let mut a = -HALO;
        while a <= edge {
            sites.push(site_of(body, key, a, b));
            a += 1;
        }
        b += 1;
    }
    // The tubes that can reach the box: the regions around its eight corners, then only the tubes
    // within their radius of the box's bounding sphere. NINE directions, never four thousand — the
    // corners and the centre are the only columns the carver list reads.
    let carve_tubes = tubes_carve_at(body, rung);
    let carve_caverns = caverns_carve_at(body, rung);
    let r_low = body.ladder.corner_radius_steps(k0 - HALO, rung);
    let r_high = body.ladder.corner_radius_steps(k0 + edge + HALO, rung);
    let mut tubes: Vec<Tube> = Vec::new();
    if carve_tubes {
        let dir_at = |a: i32, b: i32| site_dir(body, key, sites[SampleBox::column_index(a, b)]);
        let centre = point_at(dir_at(edge >> 1, edge >> 1), Gi::new((r_low + r_high) >> 1));
        let mut corners = [[Gi::ZERO; 3]; 8];
        let mut corner = 0;
        while corner < 8 {
            let a = if corner & 1 == 0 { -HALO } else { edge };
            let b = if corner & 2 == 0 { -HALO } else { edge };
            let r = Gi::new(if corner & 4 == 0 { r_low } else { r_high });
            corners[corner] = point_at(dir_at(a, b), r);
            corner += 1;
        }
        tubes = tubes_reaching(body, centre, corners);
    }
    // The chunk's own node lattice, EXTENDED by one node on every side so the same-face halo reads
    // it too; the partner faces' lattices over the halo columns across a seam. The node a cell sits
    // in is the stride's SHIFT (ruling F7: no `/` on the recipe's path).
    let node = |v: i32| v >> CAVERN_STRIDE_LOG2;
    let node0 = [
        node(key.x * edge) - 1,
        node(key.y * edge) - 1,
        (node(k0) - 1).max(0),
    ];
    let k_dims = CAVERN_NODES + 2;
    let mut lattices = Vec::new();
    if carve_caverns {
        lattices.push(LatticeExtent {
            face: key.face,
            node0,
            dims: [CAVERN_NODES + 2; 3],
        });
        lattices.extend(foreign_extents(key.face, &sites, node0[2], k_dims));
    }
    BoxTopology {
        sites,
        tubes,
        lattices,
        charter: charter_of(body, rung, BOX_EDGE),
        carve_caverns,
        node0,
        k0,
        band,
    }
}

/// The prologue of one box: [`BoxSetup`] for `key`, whose core columns come from `column`. The
/// topology above, plus the two passes RUN ON THIS HOST — the columns' surfaces and the lattices'
/// node values, through the very kernels the card runs.
pub(crate) fn box_setup(
    body: &BodyDefinition,
    field: Option<&dyn crate::artifact::ZField>,
    key: ChunkKey,
    column: &ColumnField,
) -> Option<BoxSetup> {
    let rung = key.rung;
    let edge = CHUNK_EDGE as i32;
    let topology = box_topology(body, key);
    // Every column: its direction, its surface and its biome. A core column's surface comes from
    // the column pass already run, so no core work is repeated.
    let mut dirs = Vec::with_capacity(BOX_EDGE * BOX_EDGE);
    let mut surfaces: Vec<(Gi, Biome)> = Vec::with_capacity(BOX_EDGE * BOX_EDGE);
    let mut water: Vec<Gi> = Vec::with_capacity(BOX_EDGE * BOX_EDGE);
    // ★ Every column's ROCK PROVINCE (slice 8d step 2), the halo's included: a halo cell must read
    // the province its owning chunk reads, or the wall of a mine would change rock at every chunk
    // edge.
    let mut province: Vec<Gi> = Vec::with_capacity(BOX_EDGE * BOX_EDGE);
    // ★ ONE COLUMN PASS (step G2-A): the charter once per box, the column rule per halo column —
    // the core's own rule (2026-09-20: a halo that read the recipe while the core read the
    // artifact cracked every chunk edge).
    let plan_charter = body.plan_charter(rung, key.face);
    // The body's own lattice beside the level's: the halo reads the water's side the core reads
    // (2026-09-22, the coast mask), and a side comes off the FINE node's own bit.
    let fine = match field {
        Some(_) => Some(body.macro_lattice()?),
        None => None,
    };
    let lattice = match (field, fine.as_ref()) {
        (Some(f), Some(l)) => Some(l.coarser(f.level())?),
        _ => None,
    };
    let read = crate::chunk::ColumnRead {
        body,
        field,
        lattice: lattice.as_ref(),
        fine_lattice: fine.as_ref(),
        charter: &plan_charter,
        first: if field.is_some() {
            body.first_fine()
        } else {
            0
        },
        // The halo reads the core's rule, the slope share included (2026-09-20's cracked edges).
        slope_charter: match lattice.as_ref() {
            Some(l) => crate::artifact::slope_charter(body, l, rung),
            None => crate::artifact::SlopeCharter::NONE,
        },
        key,
        n_cells: body.ladder.cells_per_edge(rung) as i32,
    };
    let mut b = -HALO;
    while b <= edge {
        let mut a = -HALO;
        while a <= edge {
            let core_column = (a >= 0) & (a < edge) & (b >= 0) & (b < edge);
            let (dir, h, biome, w, p) = if core_column {
                let index = (b as usize) * CHUNK_EDGE + a as usize;
                let (dir, h, biome) = column.columns[index];
                (dir, h, biome, column.water[index], column.province[index])
            } else {
                // A halo column is the neighbour's own column: it reads what the neighbour reads.
                let got = read.column(a, b)?;
                (got.dir, got.h, got.biome, got.water, got.province)
            };
            dirs.push(dir);
            surfaces.push((h, biome));
            water.push(w);
            province.push(p);
            a += 1;
        }
        b += 1;
    }
    let mut built = topology
        .lattices
        .iter()
        .map(|e| NodeLattice::build(body, rung, e.face, e.node0, e.dims));
    let own = built.next().unwrap_or_else(|| NodeLattice::empty(key.face));
    let foreign: Vec<NodeLattice> = built.collect();
    let BoxTopology {
        sites,
        tubes,
        charter,
        carve_caverns,
        k0,
        band,
        ..
    } = topology;
    Some(BoxSetup {
        sites,
        dirs,
        surfaces,
        water,
        province,
        tubes,
        own,
        foreign,
        charter,
        carve_caverns,
        k0,
        band,
    })
}

/// The chunk with its halo; `None` for a key outside the body.
#[must_use]
pub fn sample_box(
    body: &BodyDefinition,
    field: Option<&dyn crate::artifact::ZField>,
    key: ChunkKey,
) -> Option<SampleBox> {
    if !in_ladder(body, key) {
        return None;
    }
    let column = column_field(body, field, key.face, key.rung, key.x, key.y)?;
    let core = generate_in(body, &column, key.z)?;
    let rung = key.rung;
    let edge = CHUNK_EDGE as i32;
    let BoxSetup {
        sites,
        dirs,
        surfaces,
        water,
        province,
        tubes,
        own,
        foreign,
        charter,
        carve_caverns,
        k0,
        band,
        ..
    } = box_setup(body, field, key, &column)?;
    let mut cells = Vec::with_capacity(BOX_CELLS);
    let mut c = -HALO;
    while c <= edge {
        let k = k0 + c;
        let r_steps = Gi::new(body.ladder.cell_radius_steps(k, rung));
        let r = r_steps << LENGTH_BITS;
        let mut b = -HALO;
        while b <= edge {
            let mut a = -HALO;
            while a <= edge {
                let cell = if SampleBox::is_core(a, b, c) {
                    core.cell(a as usize, b as usize, c as usize)
                } else if k < 0 {
                    // Below the band's floor: what the below-surface skip writes.
                    below_surface_cell(&charter)
                } else if k >= band {
                    // Above the band's top: what the above-surface skip writes. The top stands
                    // above every surface and above the sea (the room above is derived from the
                    // relief, and the sea lies within it), so this is air in practice; the rule is
                    // shared so it cannot drift from the skip's.
                    above_surface_cell(&charter, r, water[SampleBox::column_index(a, b)])
                } else {
                    let col = SampleBox::column_index(a, b);
                    let site = sites[col];
                    let dir = dirs[col];
                    let (h, biome) = surfaces[col];
                    let column_water = water[col];
                    let value = if carve_caverns & charter.in_band(h - r) {
                        cavern_of(&own, &foreign, site, k)
                    } else {
                        Gi::ZERO
                    };
                    finish_cell(
                        &charter,
                        &CellSite {
                            dir,
                            h,
                            biome,
                            r_steps,
                            water: column_water,
                            province: province[col],
                        },
                        value,
                        &tubes,
                    )
                };
                cells.push(cell);
                a += 1;
            }
            b += 1;
        }
        c += 1;
    }
    Some(SampleBox {
        key,
        cells,
        sites,
        dirs,
        water,
        surfaces: surfaces.iter().map(|s| s.0).collect(),
        sea: body.sea_radius,
    })
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]
    use super::*;
    use crate::digest::surface_chunk_z;
    use crate::home::home_planet;
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

    /// The last chunk index along a face edge at a rung: the partial chunk when the edge is not a
    /// whole number of chunks.
    fn edge_chunk(m: &BodyDefinition, rung: u8) -> i32 {
        (m.ladder.cells_per_edge(rung) as i32 - 1) / CHUNK_EDGE as i32
    }

    /// Every halo cell of `key` at local `(a, b, c)` equals the cell the owning chunk computes,
    /// for the strip of local columns `a_range × b_range`.
    fn halo_strip_matches(
        m: &BodyDefinition,
        k: ChunkKey,
        a_range: (i32, i32),
        b_range: (i32, i32),
    ) -> usize {
        let sb = sample_box(m, None, k).expect("in the band");
        let mut compared = 0;
        let mut b = b_range.0;
        while b <= b_range.1 {
            let mut a = a_range.0;
            while a <= a_range.1 {
                let site = sb.site(a, b);
                let face = Face::from_index(site.face).expect("a face cell");
                let edge = CHUNK_EDGE as i32;
                let owner = key(face, k.rung, site.i / edge, site.j / edge, k.z);
                let own = crate::chunk::generate(m, None, owner).expect("the owner's chunk");
                let (la, lb) = ((site.i % edge) as usize, (site.j % edge) as usize);
                let mut c = 0;
                while c < edge {
                    assert_eq!(
                        sb.cell(a, b, c),
                        own.cell(la, lb, c as usize),
                        "halo ({a}, {b}, {c}) of {k:?} = cell ({la}, {lb}) of {owner:?}"
                    );
                    compared += 1;
                    c += 1;
                }
                a += 1;
            }
            b += 1;
        }
        compared
    }

    #[test]
    fn the_box_holds_the_core_bytes_and_a_halo_on_every_side() {
        let m = home_planet();
        let z = surface_chunk_z(&m, Face::PosX, 0, 300, 700);
        let k = key(Face::PosX, 0, 300, 700, z);
        let sb = sample_box(&m, None, k).expect("in the band");
        let core = crate::chunk::generate(&m, None, k).expect("in the band");
        assert_eq!(sb.cells.len(), BOX_CELLS);
        assert_eq!(sb.sites.len(), BOX_EDGE * BOX_EDGE);
        let mut c = 0;
        while c < CHUNK_EDGE {
            let mut b = 0;
            while b < CHUNK_EDGE {
                let mut a = 0;
                while a < CHUNK_EDGE {
                    assert_eq!(sb.cell(a as i32, b as i32, c as i32), core.cell(a, b, c));
                    a += 1;
                }
                b += 1;
            }
            c += 1;
        }
        assert!(SampleBox::is_core(0, 0, 0));
        assert!(!SampleBox::is_core(-1, 0, 0));
        assert!(!SampleBox::is_core(0, 62, 0));
        assert!(!SampleBox::is_core(0, 0, -1));
        assert_eq!(sample_box(&m, None, key(Face::PosX, 0, 300, 700, -1)), None);
        // The same-face halo on the −a and +b sides, and the radial layers, match their owners.
        let edge = CHUNK_EDGE as i32;
        let n = halo_strip_matches(&m, k, (-1, -1), (0, edge - 1))
            + halo_strip_matches(&m, k, (0, edge - 1), (edge, edge));
        assert_eq!(n, 2 * CHUNK_EDGE * CHUNK_EDGE);
        let below =
            crate::chunk::generate(&m, None, key(Face::PosX, 0, 300, 700, z - 1)).expect("below");
        let above =
            crate::chunk::generate(&m, None, key(Face::PosX, 0, 300, 700, z + 1)).expect("above");
        let mut b = 0;
        while b < edge {
            let mut a = 0;
            while a < edge {
                assert_eq!(
                    sb.cell(a, b, -1),
                    below.cell(a as usize, b as usize, CHUNK_EDGE - 1)
                );
                assert_eq!(sb.cell(a, b, edge), above.cell(a as usize, b as usize, 0));
                a += 1;
            }
            b += 1;
        }
    }

    /// The refuter's standing question for slice 6: across a face edge, the halo is the partner
    /// face's own cell — the crossing rule of the leaf, and the same bytes. On the home planet a
    /// face edge is not a whole number of chunks, so the last chunk is PARTIAL: its own cells beyond
    /// the face are the partner's too, and every one of them matches.
    #[test]
    fn across_a_face_edge_the_halo_is_the_partner_faces_own_cells() {
        let m = home_planet();
        let rung = 4;
        let last = edge_chunk(&m, rung);
        let n_l = m.ladder.cells_per_edge(rung) as i32;
        let edge = CHUNK_EDGE as i32;
        // The last chunk at the +u edge of +X (its +a side crosses to +Y), away from the corners.
        let z = surface_chunk_z(&m, Face::PosX, rung, last, 7);
        let k = key(Face::PosX, rung, last, 7, z);
        let sb = sample_box(&m, None, k).expect("in the band");
        let first_beyond = n_l - last * edge;
        assert!(first_beyond <= edge, "the chunk reaches the face edge");
        assert_eq!(sb.site(first_beyond - 1, 3).face, Face::PosX.index());
        let s0 = sb.site(first_beyond, 3);
        assert_eq!(s0.face, Face::PosY.index(), "+X's +u side is +Y");
        // Beyond the edge the site walks inward on the partner, one cell per column, along the
        // same partner row.
        let s1 = sb.site(first_beyond + 1, 3);
        assert_eq!(s1.i, s0.i, "the along-edge index is the same");
        assert_eq!((s1.j - s0.j).abs(), 1, "one cell further in on the partner");
        // Every column beyond the face, the halo included, equals the partner's own cell.
        let n = halo_strip_matches(&m, k, (first_beyond, edge), (0, edge - 1));
        assert_eq!(
            n,
            (edge + 1 - first_beyond) as usize * CHUNK_EDGE * CHUNK_EDGE
        );
        // And the −v side of +X crosses to −Z: the halo row b = −1 of a chunk at y = 0.
        let z2 = surface_chunk_z(&m, Face::PosX, rung, 5, 0);
        let k2 = key(Face::PosX, rung, 5, 0, z2);
        let sb2 = sample_box(&m, None, k2).expect("in the band");
        assert_eq!(sb2.site(3, -1).face, Face::NegZ.index());
        let n2 = halo_strip_matches(&m, k2, (0, edge - 1), (-1, -1));
        assert_eq!(n2, CHUNK_EDGE * CHUNK_EDGE);
    }

    /// At a cube corner the fourth cell is a phantom on the corner direction, and the three faces'
    /// chunks compute the same phantom byte for byte.
    #[test]
    fn at_a_cube_corner_the_three_chunks_share_one_phantom() {
        let m = home_planet();
        let rung = 6;
        let last = edge_chunk(&m, rung);
        let n_l = m.ladder.cells_per_edge(rung) as i32;
        let edge = CHUNK_EDGE as i32;
        // The first local column past the face in the last chunk.
        let beyond = n_l - last * edge;
        // The corner where +X's (+u, +v) meets +Y and +Z: the direction (1, 1, 1)/√3, at local
        // column (beyond, beyond) of the last chunk of +X.
        let zx = surface_chunk_z(&m, Face::PosX, rung, last, last);
        let kx = key(Face::PosX, rung, last, last, zx);
        let bx = sample_box(&m, None, kx).expect("in the band");
        let phantom = bx.site(beyond, beyond);
        assert_eq!(phantom.face, CORNER_FACE);
        assert_eq!((phantom.i, phantom.j), (1, 1));
        let d = bx.dir(beyond, beyond);
        // `round(2⁴⁰ / √3)`, within the reciprocal square root's own two units.
        let third = 634_803_334_274i64;
        let mut c = 0;
        while c < 3 {
            assert!((d[c].raw() - third).abs() <= 2, "[{c}]: {:?}", d[c]);
            c += 1;
        }
        // Faces +Y and +Z hold the same corner in their own last chunks, at the same local column.
        let ky = key(Face::PosY, rung, last, last, zx);
        let by = sample_box(&m, None, ky).expect("in the band");
        assert_eq!(by.site(beyond, beyond).face, CORNER_FACE);
        assert_eq!(by.dir(beyond, beyond), d);
        let kz = key(Face::PosZ, rung, last, last, zx);
        let bz = sample_box(&m, None, kz).expect("in the band");
        assert_eq!(bz.dir(beyond, beyond), d);
        let mut c = -1;
        while c <= edge {
            assert_eq!(
                bx.cell(beyond, beyond, c),
                by.cell(beyond, beyond, c),
                "layer {c}"
            );
            assert_eq!(
                bx.cell(beyond, beyond, c),
                bz.cell(beyond, beyond, c),
                "layer {c}"
            );
            c += 1;
        }
        // The two strips beside the phantom belong to the two other faces.
        assert_eq!(bx.site(beyond, 3).face, Face::PosY.index());
        assert_eq!(bx.site(3, beyond).face, Face::PosZ.index());
        // A phantom at the (−u, −v) corner of a minus face.
        let k0 = key(Face::NegY, rung, 0, 0, zx);
        let b0 = sample_box(&m, None, k0).expect("in the band");
        assert_eq!(
            b0.site(-1, -1),
            Site {
                face: CORNER_FACE,
                i: -1,
                j: -1
            }
        );
        let d0 = b0.dir(-1, -1);
        let mut c = 0;
        while c < 3 {
            assert!((d0[c].raw() + third).abs() <= 2, "[{c}]: {:?}", d0[c]);
            c += 1;
        }
    }

    /// The band's floor edge is owned by nobody (no chunk lies below it), so it must never cross:
    /// the crust is derived to hold every stratum and cave with 64 m to spare, so layer 0 is bedrock
    /// under every column the golden set walks. Stated as a test, not as an argument.
    #[test]
    fn the_bands_floor_layer_is_bedrock_under_every_golden_column() {
        let m = home_planet();
        let bedrock = m.strata.bedrock.stratum();
        for (face, x, y) in [
            (Face::PosX, 300, 700),
            (Face::NegX, 1181, 77),
            (Face::PosY, 5, 5),
            (Face::NegZ, 40, 4000),
        ] {
            let sb = sample_box(&m, None, key(face, 0, x, y, 0)).expect("the floor chunk");
            let mut n = 0;
            while n < BOX_EDGE * BOX_EDGE {
                let (a, b) = ((n % BOX_EDGE) as i32 - 1, (n / BOX_EDGE) as i32 - 1);
                assert_eq!(sb.cell(a, b, 0).stratum, bedrock, "{face:?} ({a}, {b})");
                assert_eq!(sb.cell(a, b, 0).gap, i8::MIN);
                n += 1;
            }
        }
    }

    /// The inverse of `site_of`: every column of a box, its own, its halo across each side and a
    /// partial chunk's columns beyond the face, maps back to itself; a phantom maps to nothing.
    #[test]
    fn local_of_site_inverts_site_of_over_the_whole_box() {
        let m = home_planet();
        let rung = 3;
        let last = edge_chunk(&m, rung);
        let edge = CHUNK_EDGE as i32;
        for k in [
            key(Face::PosX, rung, 0, 0, 3),
            key(Face::PosX, rung, last, last, 3),
            key(Face::NegY, rung, 0, last, 3),
            key(Face::NegZ, rung, last, 0, 3),
            key(Face::PosZ, rung, 7, 9, 3),
        ] {
            let mut b = -1;
            while b <= edge {
                let mut a = -1;
                while a <= edge {
                    let site = site_of(&m, k, a, b);
                    let back = local_of_site(&m, k, site);
                    if site.face == CORNER_FACE {
                        assert_eq!(back, None, "{k:?} ({a}, {b}) is a phantom");
                    } else {
                        assert_eq!(back, Some((a, b)), "{k:?} ({a}, {b}) → {site:?}");
                    }
                    a += 1;
                }
                b += 1;
            }
        }
        // A site the box does not hold: none.
        let k = key(Face::PosX, rung, 7, 9, 3);
        assert_eq!(
            local_of_site(
                &m,
                k,
                Site {
                    face: Face::PosX.index(),
                    i: 0,
                    j: 0
                }
            ),
            None
        );
        assert_eq!(
            local_of_site(
                &m,
                k,
                Site {
                    face: Face::NegX.index(),
                    i: 5,
                    j: 5
                }
            ),
            None
        );
        // A cell of an ADJACENT face (+Y is across +X's +u side) that this interior chunk's halo
        // does not reach: the side matches, the inverse lands outside the box, none.
        let n = m.ladder.cells_per_edge(rung) as i32;
        assert_eq!(
            local_of_site(
                &m,
                k,
                Site {
                    face: Face::PosY.index(),
                    i: 5_000,
                    j: n - 1
                }
            ),
            None
        );
    }

    #[test]
    fn below_the_floor_is_bedrock_and_above_the_band_is_air_or_water() {
        let m = home_planet();
        let top = (m.ladder.cells_in_band(0) as i32 - 1) / CHUNK_EDGE as i32;
        let floor = sample_box(&m, None, key(Face::NegZ, 0, 10, 10, 0)).expect("the floor chunk");
        assert_eq!(
            floor.cell(5, 5, -1),
            Cell {
                stratum: m.strata.bedrock.stratum(),
                gap: i8::MIN
            }
        );
        let ceiling = sample_box(&m, None, key(Face::NegZ, 0, 10, 10, top)).expect("the top chunk");
        let above = ceiling.cell(5, 5, CHUNK_EDGE as i32);
        assert_eq!(above.gap, i8::MAX);
        assert!(!above.stratum.is_solid());
    }
}
