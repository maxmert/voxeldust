//! ★ THE CHUNK — the recipe's unit of work: one chunk of `62 × 62 × 62` cells at one rung, each cell a
//! substance and a gap. The evaluation order is FIXED: every column's surface first (one height per
//! column, with the rung's octaves), then every cell in the packing order of the address
//! (`(c·62 + b)·62 + a`, `a` fastest), so two hosts fold the same bytes in the same order.
//!
//! **Two steps, the first shared.** A [`ColumnField`] is the column pass of one chunk column — the
//! direction, the surface and the biome of each of the `62 × 62` columns, plus the least and the
//! greatest surface among them. Every chunk of that column along the radial reuses it. The cell pass
//! then decides, from the column extremes, whether the chunk is wholly air and water, wholly bedrock,
//! or crossed by the surface or a cave; only the crossed chunk pays per cell.
//!
//! **A skip writes exactly the bytes the cell pass would.** A chunk is skipped as "above" only when
//! its lowest cell centre is more than one cell above every surface of the column, so every cell's gap
//! clamps to the top code; it is skipped as "below" only when its highest cell centre is more than one
//! cell under every surface, every stratum and every cave, so every cell's gap clamps to the bottom
//! code and every cell is bedrock. A test generates a skipped chunk both ways and compares cell for
//! cell.
//!
//! **The gap byte's conventions** (Format B): `(r − h) / cell` for the rock, clamped to one cell, then
//! the greater of that and a cave's hollow, quantised to a signed byte in 1/128 of a cell: negative
//! inside rock, positive in air; a cell whose centre is exactly ON the surface reads 0 and is AIR;
//! `+1` cell clamps to the top code 127, one step short of a full cell, which the extractor never needs.
//!
//! ★ **THE PER-CELL ARITHMETIC IS THE RECIPE'S OWN KERNEL** (ruling F7, step G1): this module runs
//! the ORCHESTRATION — the column pass, the cavern lattice's nodes, the carvers a chunk can reach —
//! and every cell's substance and gap come from `vd_recipe::cell::cell_word`, the same function the
//! client's card runs (`crate::gpu`). One arithmetic, three hosts; the substance CODES travel to it
//! in the body's charter ([`charter_of`]), so the recipe names no substance and this crate keeps
//! the naming.
//!
//! ★ **THE DENSITY IS ALL INTEGERS** (ruling F7). A cell centre's radius is an EXACT whole number of
//! gap steps (the floor is whole metres and a cell is a power of two metres, so `floor + (k + ½)·cell`
//! times 128 is a whole number — `vd_seed::ladder::cell_radius_steps`); the surface is the same unit at
//! the noise's fraction bits; the gap is ONE subtraction, ONE shift by `rung + the noise's bits` (which
//! divides by the cell's width and floors in one step) and the clamp to the byte. Nothing rounds twice
//! and nothing is a float.
//!
//! **Example.** Chunk (face 2, rung 0, 19, 1, 4) of the home planet spans 62 m of radius; its columns'
//! surfaces run through it, so it pays: 3 844 heights once for the column, then 238 328 cells for this
//! chunk. The chunk two above it is skipped: every cell is air at the top code, exactly as the cell pass
//! would write it.

use crate::body::BodyDefinition;
use crate::carve::{
    CAVERN_STRIDE, CAVERN_STRIDE_LOG2, caverns_carve_at, cell_steps, tube_region, tubes_carve_at,
    tubes_near,
};
use crate::height::biome_of_code;
use crate::strata::{Biome, Stratum};
use crate::units::{LENGTH_BITS, STEPS_PER_M, greater, lesser};
use vd_recipe::Gi;
use vd_recipe::cell::{
    CellAt, CellCharter, Tube, above_cell_word, below_cell_word, cell_word, gap_of_word,
    strata_row, stratum_of_word,
};
use vd_recipe::plan::{PlanCharter, column_surface_from};
use vd_recipe::root::isqrt;

/// ★ THE ARITHMETIC IS THE RECIPE'S (ruling F7, step G1). A point in the body's frame and the
/// cavern lattice's blend are kernels of `vd_recipe::cell`, re-read here under their own names, so
/// the generator, the collider and the shader run ONE body of code and never a copy of it.
pub(crate) use vd_recipe::cell::{point_at, trilinear8};

use vd_seed::bend::Face;

/// Cells per chunk edge.
pub const CHUNK_EDGE: usize = 62;
/// Cells per chunk.
pub const CHUNK_CELLS: usize = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;
/// Gap steps per cell: the registry's density convention (1/128 of a cell). Cross-pinned against
/// `vd_core::registry::GAP_STEPS_PER_CELL` in `tests/tests/voxel_pins.rs` and against the recipe's own
/// constant by this module's test.
pub const GAP_STEPS_PER_CELL: i64 = 128;
const _: () = assert!(GAP_STEPS_PER_CELL == vd_recipe::height::GAP_STEPS_PER_CELL);
const _: () = assert!(GAP_STEPS_PER_CELL == STEPS_PER_M);
/// The most nodes per axis a chunk's cavern lattice needs: the global nodes that cover its 62 cells.
pub const CAVERN_NODES: usize = (CHUNK_EDGE >> CAVERN_STRIDE_LOG2) + 2;

/// Which chunk: the face, the rung, and the chunk's coordinates in cells-per-62 along the face's
/// two axes and the radial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkKey {
    pub face: Face,
    pub rung: u8,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// One cell: what it is and where the surface is relative to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub stratum: Stratum,
    /// The gap in 1/128 of a cell: negative inside solid, positive in air, 0 exactly on the surface
    /// (air), clamped to `[−128, 127]`.
    pub gap: i8,
}

/// A generated chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkLattice {
    pub key: ChunkKey,
    /// `CHUNK_CELLS` cells in the packing order `(c·62 + b)·62 + a`.
    pub cells: Vec<Cell>,
    /// How the chunk was made: by a skip, or cell by cell. The bytes are the same either way.
    pub how: How,
}

/// How a chunk was produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    /// More than a cell above every surface of its column: air and water at the top code.
    AboveSurface,
    /// More than a cell below every surface, stratum and cave of its column: bedrock at the bottom
    /// code.
    BelowSurface,
    /// The surface or the caves cross it: every cell evaluated.
    Evaluated,
}

/// The column pass of one chunk column: what every chunk along the radial shares.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnField {
    pub face: Face,
    pub rung: u8,
    pub x: i32,
    pub y: i32,
    /// `62 × 62` entries in packing order `b·62 + a`: the direction at the bend's fraction bits, the
    /// surface radius in gap steps at [`LENGTH_BITS`], the biome.
    pub columns: Vec<([Gi; 3], Gi, Biome)>,
    /// ★ The same columns' WATER surface radius (slice 8c stage C5) in gap steps at
    /// [`LENGTH_BITS`], or ZERO for a dry column: the artifact's row under the column (the sea,
    /// a lake), the body's sea where the field holds no water word, nothing without a sea.
    pub water: Vec<Gi>,
    /// The same columns' sites: a cell of this face, or — in a PARTIAL chunk at a face's far edge —
    /// the partner face's cell or a corner phantom (slice 6, `lattice::site_of`).
    pub sites: Vec<crate::lattice::Site>,
    /// The least and the greatest surface radius among the columns, in gap steps at [`LENGTH_BITS`].
    pub lowest: Gi,
    pub highest: Gi,
}

impl ChunkLattice {
    /// The cell at `(a, b, c)` inside the chunk, `a` along the face's first axis, `c` radial.
    #[must_use]
    pub fn cell(&self, a: usize, b: usize, c: usize) -> Cell {
        self.cells[(c * CHUNK_EDGE + b) * CHUNK_EDGE + a]
    }
}

/// Quantise a gap already counted in GAP STEPS (1/128 of a cell) to the signed byte: clamped to
/// `[−128, 127]`. The floor happened in the one shift that made the steps, so nothing rounds twice;
/// `+1` cell (128 steps) clamps to the top code 127, one step short of a full cell.
#[must_use]
pub fn quantise_gap(gap_steps: Gi) -> i8 {
    vd_recipe::cell::gap_code(gap_steps).raw() as i8
}

/// Whether a chunk key names a chunk of the body at all.
pub fn in_ladder(body: &BodyDefinition, key: ChunkKey) -> bool {
    if key.rung >= body.ladder.rungs || key.x < 0 || key.y < 0 || key.z < 0 {
        return false;
    }
    let n_l = body.ladder.cells_per_edge(key.rung) as i64;
    let band = i64::from(body.ladder.cells_in_band(key.rung));
    let edge = CHUNK_EDGE as i64;
    i64::from(key.x) * edge < n_l && i64::from(key.y) * edge < n_l && i64::from(key.z) * edge < band
}

/// The column pass for the column of `(face, rung, x, y)`; `None` for a column outside the body.
#[must_use]
pub fn column_field(
    body: &BodyDefinition,
    field: Option<&dyn crate::artifact::ZField>,
    face: Face,
    rung: u8,
    x: i32,
    y: i32,
) -> Option<ColumnField> {
    if !in_ladder(
        body,
        ChunkKey {
            face,
            rung,
            x,
            y,
            z: 0,
        },
    ) {
        return None;
    }
    let key = ChunkKey {
        face,
        rung,
        x,
        y,
        z: 0,
    };
    let mut columns = Vec::with_capacity(CHUNK_EDGE * CHUNK_EDGE);
    let mut water = Vec::with_capacity(CHUNK_EDGE * CHUNK_EDGE);
    let mut sites = Vec::with_capacity(CHUNK_EDGE * CHUNK_EDGE);
    // The extremes start at the FIRST column's own surface, never at a sentinel (the refuter's
    // finding: a sentinel the arithmetic absorbs reads a column's floor kilometres out).
    let mut lowest = Gi::ZERO;
    let mut highest = Gi::ZERO;
    // ★ ONE COLUMN PASS (step G2-A): the charter once per chunk, then the recipe's own kernel per
    // column — the very kernel the card runs, so the shard's hill is the card's hill.
    let charter = body.plan_charter(rung, face);
    // ★ THE MACRO FIELD (slice 8c stage C4): with an artifact every column reads `Z` under its
    // own site and sums the FINE octaves only; a column whose stencil the field does not hold
    // makes no chunk (the client keeps the coarser rung standing, ruling F9). Without a field the
    // column is the recipe's own coarse relief — the crate's kernel tests and the pre-artifact
    // measurements, never a shipped path.
    // The field stands at a pyramid level (zero for the rows): the read gathers on the lattice of
    // that level, so a coarse rung reads the means the client holds from orbit.
    let lattice = match field {
        Some(f) => Some(body.macro_lattice()?.coarser(f.level())?),
        None => None,
    };
    let first = if field.is_some() {
        body.first_fine()
    } else {
        0
    };
    let n_cells = body.ladder.cells_per_edge(rung) as i32;
    let mut b = 0;
    while b < CHUNK_EDGE {
        let mut a = 0;
        while a < CHUNK_EDGE {
            let site = crate::lattice::site_of(body, key, a as i32, b as i32);
            let (z, w, coast) = match (field, lattice) {
                (Some(f), Some(l)) => {
                    // A corner phantom names no cell: its `Z` is the key face's corner node,
                    // which the read reaches by the cell the phantom would be on that face.
                    let (zf, zi, zj) = if site.face == crate::lattice::CORNER_FACE {
                        let gi = key.x * CHUNK_EDGE as i32 + a as i32;
                        let gj = key.y * CHUNK_EDGE as i32 + b as i32;
                        (face, gi.clamp(-1, n_cells), gj.clamp(-1, n_cells))
                    } else {
                        (Face::from_index(site.face).unwrap_or(face), site.i, site.j)
                    };
                    // ★ THE WATER AND THE COAST (C5): the nearest row's level as a radius (the
                    // body's sea where the field holds no water word; ZERO — none — for a dry
                    // row), and the row's coast bit.
                    let (w, coast) = match crate::artifact::sample_row(&l, f, zf, rung, zi, zj) {
                        Some((crate::artifact::DRY_M, facies)) => {
                            (Gi::ZERO, facies & crate::solve::FACIES_COAST != 0)
                        }
                        Some((level, facies)) => (
                            body.radius + (Gi::new(i64::from(level) * STEPS_PER_M) << LENGTH_BITS),
                            facies & crate::solve::FACIES_COAST != 0,
                        ),
                        None => (body.sea_radius, false),
                    };
                    (
                        crate::artifact::sample_z(&l, f, zf, rung, zi, zj)?,
                        w,
                        coast,
                    )
                }
                _ => (Gi::ZERO, body.sea_radius, false),
            };
            let surface =
                column_surface_from(&charter, i32::from(site.face), site.i, site.j, z, first);
            let (dir, h) = (surface.dir, surface.h);
            let biome = biome_of_code(surface.biome);
            if (a == 0) & (b == 0) {
                lowest = h;
                highest = h;
            } else {
                lowest = lesser(lowest, h);
                highest = greater(highest, h);
            }
            // ★ THE COAST MARKED (C5): a column in the coast band that stands above its water is
            // a beach — sand over sandstone, the desert's own strata — and the water below it is
            // where the sheet meets the ground.
            let biome = if coast & (h >= w) {
                Biome::Desert
            } else {
                biome
            };
            columns.push((dir, h, biome));
            water.push(w);
            sites.push(site);
            a += 1;
        }
        b += 1;
    }
    Some(ColumnField {
        face,
        rung,
        x,
        y,
        columns,
        water,
        sites,
        lowest,
        highest,
    })
}

/// ★ THE CHARTER OF THIS BODY AT THIS RUNG — every number the recipe's cell kernel reads that is
/// not the cell's own, drawn ONCE per chunk and handed to every cell (`vd_recipe::cell`). The
/// substance CODES travel in it, so the recipe names no substance and this crate keeps the naming:
/// a renumbering here can never move a bit of that arithmetic. `box_edge` is what a GPU shell reads
/// to find a cell's column; the CPU's own loops never read it.
#[must_use]
pub fn charter_of(body: &BodyDefinition, rung: u8, box_edge: usize) -> CellCharter {
    let code = |s: Stratum| Gi::new(i64::from(s.code()));
    let bedrock = body.strata.bedrock.stratum();
    let row = |topsoil: Stratum, subsoil: Stratum| {
        strata_row(code(topsoil), code(subsoil), code(body.strata.sediment))
    };
    let carve_any = tubes_carve_at(body, rung) | caverns_carve_at(body, rung);
    CellCharter {
        sea_radius: body.sea_radius,
        cave_min: Gi::new(i64::from(body.caves.min_depth_m) * STEPS_PER_M) << LENGTH_BITS,
        cave_max: Gi::new(i64::from(body.caves.max_depth_m) * STEPS_PER_M) << LENGTH_BITS,
        cavern_threshold: body.caves.cavern_threshold,
        cavern_scale_steps: body.caves.cavern_scale_steps,
        carve_any: if carve_any { Gi::ONE } else { Gi::ZERO },
        rung: Gi::new(i64::from(rung)),
        topsoil_m: Gi::new(i64::from(body.strata.topsoil_m)),
        subsoil_end_m: Gi::new(i64::from(body.strata.topsoil_m + body.strata.subsoil_m)),
        strata_end_m: Gi::new(i64::from(body.strata.max_depth_m())),
        air_code: code(Stratum::Air),
        water_code: code(Stratum::Water),
        bedrock_code: code(bedrock),
        box_edge: Gi::new(box_edge as i64),
        // The rows in the biomes' own order (`Biome as u8`), each the topsoil, the subsoil and the
        // body's sediment. A highland's subsoil is the body's bedrock: bare rock under the gravel.
        strata: [
            row(Stratum::Sand, Stratum::Sandstone),
            row(Stratum::Dirt, Stratum::Clay),
            row(Stratum::Snow, Stratum::Permafrost),
            row(Stratum::Gravel, bedrock),
        ],
    }
}

/// THE REFUSED SUBSTANCE — what a cell word names when its low byte names no stratum at all.
///
/// A word this crate's own charter made always names one, so this is reached ONLY by a garbled
/// readback: a card that wrote nonsense, a buffer read short, a driver that lost a dispatch. Two
/// things must hold then. The client must NOT PANIC — a wrong hill is a fault the GPU self-check
/// counts and the log names, while a crash throws the player out of the world for a byte. And the
/// cell must NOT BE A HOLE — a hole under a pilot's boots drops them through the ground, where a
/// wrong rock only looks wrong. So the refusal is SOLID ROCK, and the self-check's differing count
/// is what says a card is not to be trusted.
pub const REFUSED_STRATUM: Stratum = Stratum::Granite;

/// The cell a recipe word names: the substance by its code, the gap byte as it stands. The decode
/// is TOTAL — a code no stratum owns reads [`REFUSED_STRATUM`], never a panic.
#[must_use]
pub fn cell_of_word(word: u32) -> Cell {
    Cell {
        stratum: match Stratum::from_code(stratum_of_word(word)) {
            Some(stratum) => stratum,
            None => REFUSED_STRATUM,
        },
        gap: gap_of_word(word),
    }
}

/// The cell more than a cell above every surface: the fluid at its radius, at the top code.
#[must_use]
pub fn above_surface_cell(charter: &CellCharter, r: Gi, water: Gi) -> Cell {
    cell_of_word(above_cell_word(charter, r, water))
}

/// The cell more than a cell below every surface, stratum and cave: bedrock at the bottom code.
#[must_use]
pub fn below_surface_cell(charter: &CellCharter) -> Cell {
    cell_of_word(below_cell_word(charter))
}

/// A chunk filled from one rule per radial layer AND column: the skip's fill (the layer's radius
/// and the column's water decide a cell above every surface; a cell below every surface is one
/// rule for all).
fn filled(
    key: ChunkKey,
    mut cell: impl FnMut(usize, usize) -> (Stratum, i8),
    how: How,
) -> ChunkLattice {
    let mut cells = Vec::with_capacity(CHUNK_CELLS);
    let mut c = 0;
    while c < CHUNK_EDGE {
        let mut n = 0;
        while n < CHUNK_EDGE * CHUNK_EDGE {
            let (stratum, gap) = cell(n, c);
            cells.push(Cell { stratum, gap });
            n += 1;
        }
        c += 1;
    }
    ChunkLattice { key, cells, how }
}

/// Generate one chunk from its column pass; `None` for a `z` outside the body or a column pass of
/// another column. A chunk more than a cell above every surface, or more than a cell below every
/// surface, stratum and cave, is filled by rule with the bytes the cell pass would write.
#[must_use]
pub fn generate_in(body: &BodyDefinition, column: &ColumnField, z: i32) -> Option<ChunkLattice> {
    let key = ChunkKey {
        face: column.face,
        rung: column.rung,
        x: column.x,
        y: column.y,
        z,
    };
    if !in_ladder(body, key) || column.columns.len() != CHUNK_EDGE * CHUNK_EDGE {
        return None;
    }
    let rung = key.rung;
    let edge = CHUNK_EDGE as i32;
    let k0 = z * edge;
    let charter = charter_of(body, rung, CHUNK_EDGE);
    // A half cell, in gap steps at LENGTH_BITS: the distance from a cell's lower corner to its centre.
    let half_cell = (cell_steps(rung) << LENGTH_BITS) >> 1;
    let r_low = corner_radius(body, k0, rung);
    let r_high = corner_radius(body, k0 + edge, rung);
    let reach = reach_steps(body, rung);
    // Above: the lowest cell CENTRE (r_low + ½ cell) is more than a cell over the highest surface,
    // so every gap is ≥ 1 and clamps to the top code; air or water by the cell's radius.
    if r_low - half_cell > column.highest {
        return Some(filled(
            key,
            |col, c| {
                let r = cell_radius(body, k0 + c as i32, rung);
                let cell = above_surface_cell(&charter, r, column.water[col]);
                (cell.stratum, cell.gap)
            },
            How::AboveSurface,
        ));
    }
    // Below: the highest cell CENTRE (r_high − ½ cell) is more than a cell under the lowest surface
    // less every reach, so every gap is ≤ −1 and clamps to the bottom code, and every cell is past
    // the deepest stratum and the deepest cave: bedrock.
    if r_high + half_cell < column.lowest - reach {
        let cell = below_surface_cell(&charter);
        return Some(filled(
            key,
            |_, _| (cell.stratum, cell.gap),
            How::BelowSurface,
        ));
    }
    Some(cell_pass(body, column, key))
}

/// A cell centre's radius in gap steps at [`LENGTH_BITS`] — exact (the leaf's own integer twin).
#[must_use]
pub(crate) fn cell_radius(body: &BodyDefinition, k: i32, rung: u8) -> Gi {
    Gi::new(body.ladder.cell_radius_steps(k, rung)) << LENGTH_BITS
}

/// A cell's lower corner radius in gap steps at [`LENGTH_BITS`] — exact.
#[must_use]
pub(crate) fn corner_radius(body: &BodyDefinition, k: i32, rung: u8) -> Gi {
    Gi::new(body.ladder.corner_radius_steps(k, rung)) << LENGTH_BITS
}

/// How far under a surface the strata and the caves can still change a cell, in gap steps at
/// [`LENGTH_BITS`], at a rung.
pub(crate) fn reach_steps(body: &BodyDefinition, rung: u8) -> Gi {
    let carve_any = tubes_carve_at(body, rung) | caverns_carve_at(body, rung);
    let cave_reach = if carve_any {
        (Gi::new(i64::from(body.caves.max_depth_m) * STEPS_PER_M) + body.caves.cavern_scale_steps)
            << LENGTH_BITS
    } else {
        Gi::ZERO
    };
    let strata_reach =
        Gi::new((i64::from(body.strata.max_depth_m()) + 1) * STEPS_PER_M) << LENGTH_BITS;
    greater(strata_reach, cave_reach)
}

/// The tubes that can reach a box of cells whose centre and eight corners are given in gap steps: the
/// regions around the corners, then only the tubes within their radius of the box's bounding sphere. A
/// SUPERSET of what any one cell's owner keeps, and a hollow is the exact greatest over the list, so
/// the superset changes no byte. The cell pass and the halo share it.
#[must_use]
pub(crate) fn tubes_reaching(
    body: &BodyDefinition,
    centre: [Gi; 3],
    corners: [[Gi; 3]; 8],
) -> Vec<crate::carve::Tube> {
    let mut reach = Gi::ZERO;
    let mut lo = tube_region(body, centre);
    let mut hi = lo;
    let mut corner = 0;
    while corner < 8 {
        let p = corners[corner];
        let mut sum = 0u64;
        let mut axis = 0;
        while axis < 3 {
            let d = (p[axis] - centre[axis]).unsigned_abs();
            sum = sum.wrapping_add(d.wrapping_mul(d));
            axis += 1;
        }
        reach = greater(reach, Gi::new(isqrt(sum) as i64));
        let region = tube_region(body, p);
        let mut axis = 0;
        while axis < 3 {
            lo[axis] = lo[axis].min(region[axis]);
            hi[axis] = hi[axis].max(region[axis]);
            axis += 1;
        }
        corner += 1;
    }
    let mut tubes = Vec::new();
    for tube in tubes_near(body, lo, hi) {
        if tube.distance_steps(centre) <= reach + tube.radius_steps {
            tubes.push(tube);
        }
    }
    tubes
}

/// The cell pass: every cell of the chunk, in packing order. Public so a test can compare a skipped
/// chunk against it cell for cell.
#[must_use]
pub fn cell_pass(body: &BodyDefinition, column: &ColumnField, key: ChunkKey) -> ChunkLattice {
    let rung = key.rung;
    let edge = CHUNK_EDGE as i32;
    let (i0, j0, k0) = (key.x * edge, key.y * edge, key.z * edge);
    let r_low = body.ladder.corner_radius_steps(k0, rung);
    let r_high = body.ladder.corner_radius_steps(k0 + edge, rung);
    let carve_tubes = tubes_carve_at(body, rung);
    let carve_caverns = caverns_carve_at(body, rung);
    // The tubes that can reach this chunk: every tube of the regions around the chunk's own, then
    // only those that come within their radius of the chunk's bounding sphere (most chunks keep
    // none, and then no cell pays a segment distance).
    let mut tubes = Vec::new();
    if carve_tubes {
        let half = CHUNK_EDGE >> 1;
        let centre_col = column.columns[half * CHUNK_EDGE + half].0;
        let r_mid = Gi::new((r_low + r_high) >> 1);
        let centre = point_at(centre_col, r_mid);
        let mut corners = [[Gi::ZERO; 3]; 8];
        let mut corner = 0;
        while corner < 8 {
            let a = if corner & 1 == 0 { 0 } else { CHUNK_EDGE - 1 };
            let b = if corner & 2 == 0 { 0 } else { CHUNK_EDGE - 1 };
            let (dir, _, _) = column.columns[b * CHUNK_EDGE + a];
            let r = Gi::new(if corner & 4 == 0 { r_low } else { r_high });
            corners[corner] = point_at(dir, r);
            corner += 1;
        }
        tubes = tubes_reaching(body, centre, corners);
    }
    // The cavern lattice on the GLOBAL node grid (one node per `CAVERN_STRIDE` cells of the face and
    // of the radial, counted from the face's and the band's origin), so two neighbouring chunks share
    // their boundary nodes exactly and the field is continuous across every chunk edge. Each node's
    // direction comes from its own face parameter, never from a clamped column. A PARTIAL chunk at a
    // face's far edge holds the partner face's columns as well: those read a lattice of the
    // partner's own nodes, built over exactly the columns present.
    // The node a cell sits in is the stride's own SHIFT, never a divide (ruling F7): the stride is a
    // power of two, and the shift floors on both sides of zero, which is what the halo's `−1` wants.
    let node = |v: i32| v >> CAVERN_STRIDE_LOG2;
    let node0 = [node(i0), node(j0), node(k0)];
    let lattice = if carve_caverns {
        NodeLattice::build(body, rung, key.face, node0, [CAVERN_NODES; 3])
    } else {
        NodeLattice::empty(key.face)
    };
    let foreign: Vec<NodeLattice> = if carve_caverns {
        foreign_extents(key.face, &column.sites, node0[2], CAVERN_NODES)
            .into_iter()
            .map(|e| NodeLattice::build(body, rung, e.face, e.node0, e.dims))
            .collect()
    } else {
        Vec::new()
    };
    let charter = charter_of(body, rung, CHUNK_EDGE);
    let mut cells = Vec::with_capacity(CHUNK_CELLS);
    let mut c = 0;
    while c < CHUNK_EDGE {
        let r_steps = Gi::new(body.ladder.cell_radius_steps(k0 + c as i32, rung));
        let r = r_steps << LENGTH_BITS;
        let mut b = 0;
        while b < CHUNK_EDGE {
            let mut a = 0;
            while a < CHUNK_EDGE {
                let (dir, h, biome) = column.columns[b * CHUNK_EDGE + a];
                let water = column.water[b * CHUNK_EDGE + a];
                let site = column.sites[b * CHUNK_EDGE + a];
                let k = k0 + c as i32;
                // A column of this face reads the chunk's node lattice; a partner face's column (a
                // partial chunk at the face's far edge) reads that face's own global nodes, and a
                // corner phantom reads the field at its own centre — the same rule the halo uses,
                // so a partial chunk's cell beyond the face IS the partner's cell, byte for byte.
                // The cavern value is read only where a cave can be — inside the depth band — so a
                // cell far above or far below the surface pays no interpolation.
                let value = if carve_caverns & charter.in_band(h - r) {
                    cavern_of(&lattice, &foreign, site, k)
                } else {
                    Gi::ZERO
                };
                cells.push(finish_cell(
                    &charter,
                    &CellSite {
                        dir,
                        h,
                        biome,
                        r_steps,
                        water,
                    },
                    value,
                    &tubes,
                ));
                a += 1;
            }
            b += 1;
        }
        c += 1;
    }
    ChunkLattice {
        key,
        cells,
        how: How::Evaluated,
    }
}

/// A box of cavern nodes on ONE face's global node grid: the nodes from `node0` over `dims`, each
/// the field at the node's own direction and corner radius. The chunk's own lattice, the halo's
/// extension of it, and the small lattices a partial chunk builds over a partner face's columns are
/// all this one type, so every cell reads its nodes through one trilinear rule.
pub(crate) struct NodeLattice {
    pub face: Face,
    pub node0: [i32; 3],
    pub dims: [usize; 3],
    /// The field at each node, at the noise's fraction bits.
    pub values: Vec<Gi>,
}

impl NodeLattice {
    /// Build the lattice: every node evaluated, in `(c, b, a)` order.
    pub(crate) fn build(
        body: &BodyDefinition,
        rung: u8,
        face: Face,
        node0: [i32; 3],
        dims: [usize; 3],
    ) -> NodeLattice {
        // The plan charter once per lattice, never once per node: the card's node pass reads the
        // same row for the whole dispatch.
        let charter = body.plan_charter(rung, face);
        let s = CAVERN_STRIDE as i32;
        let mut values = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        let mut nc = 0;
        while nc < dims[2] {
            let k = (node0[2] + nc as i32) * s;
            let r = Gi::new(body.ladder.corner_radius_steps(k, rung));
            let mut nb = 0;
            while nb < dims[1] {
                let j = (node0[1] + nb as i32) * s;
                let mut na = 0;
                while na < dims[0] {
                    let i = (node0[0] + na as i32) * s;
                    values.push(node_value(&charter, face, i, j, r));
                    na += 1;
                }
                nb += 1;
            }
            nc += 1;
        }
        NodeLattice {
            face,
            node0,
            dims,
            values,
        }
    }

    /// A lattice with no nodes, for a rung where caverns do not carve (never read).
    pub(crate) fn empty(face: Face) -> NodeLattice {
        NodeLattice {
            face,
            node0: [0; 3],
            dims: [0; 3],
            values: Vec::new(),
        }
    }

    /// The field at a global cell of this face, interpolated from the eight nodes around it with
    /// weights that are multiples of `1/CAVERN_STRIDE`: the stride is a power of two, so each weight
    /// is an exact word at the noise's fraction bits — the same number on every target, and the same
    /// number from whichever lattice holds the nodes.
    ///
    /// ★ NO `/` AND NO `%` (ruling F7): the stride is a power of two, so the node a cell sits in is an
    /// arithmetic SHIFT and the weight inside the node pair is a MASK. The shift and the mask carry
    /// FLOOR semantics on both sides of zero, which is what a halo wants: a cell at `−1` reads the node
    /// pair `(−1, 0)` with the weight three quarters, instead of the truncating divide's pair `(0, 1)`
    /// with the weight minus one quarter — an EXTRAPOLATION past the field's own node. (A cell index is
    /// non-negative on every path the generator walks today, so this changes no byte of the world; it
    /// is the arithmetic a GPU kernel can run, and it is right where the float recipe was wrong.)
    #[inline]
    pub(crate) fn value_at(&self, cell: [i32; 3]) -> Gi {
        let node = |v: i32| v >> CAVERN_STRIDE_LOG2;
        let (na, nb, nc) = (
            (node(cell[0]) - self.node0[0]) as usize,
            (node(cell[1]) - self.node0[1]) as usize,
            (node(cell[2]) - self.node0[2]) as usize,
        );
        let at =
            |x: usize, y: usize, z: usize| self.values[(z * self.dims[1] + y) * self.dims[0] + x];
        let mask = CAVERN_STRIDE as i32 - 1;
        let weight = |v: i32| Gi::new(i64::from(v & mask)) << (LENGTH_BITS - CAVERN_STRIDE_LOG2);
        trilinear8(
            [
                at(na, nb, nc),
                at(na + 1, nb, nc),
                at(na, nb + 1, nc),
                at(na + 1, nb + 1, nc),
                at(na, nb, nc + 1),
                at(na + 1, nb, nc + 1),
                at(na, nb + 1, nc + 1),
                at(na + 1, nb + 1, nc + 1),
            ],
            [weight(cell[0]), weight(cell[1]), weight(cell[2])],
        )
    }
}

/// The lattices of every PARTNER face present among `sites` (a partial chunk's columns beyond its
/// face, or a halo across a seam): one per face, over the node range those columns need, on the
/// radial node range `[k_node0, k_node0 + k_dims)`.
pub(crate) fn foreign_extents(
    my_face: Face,
    sites: &[crate::lattice::Site],
    k_node0: i32,
    k_dims: usize,
) -> Vec<crate::lattice::LatticeExtent> {
    // The node a face index sits in: the stride's shift, as `NodeLattice::value_at` reads it.
    let node = |v: i32| v >> CAVERN_STRIDE_LOG2;
    let mut out = Vec::new();
    for face in Face::ALL {
        if face == my_face {
            continue;
        }
        let mut lo = [i32::MAX; 2];
        let mut hi = [i32::MIN; 2];
        let mut any = false;
        for site in sites {
            if site.face == face.index() {
                lo = [lo[0].min(site.i), lo[1].min(site.j)];
                hi = [hi[0].max(site.i), hi[1].max(site.j)];
                any = true;
            }
        }
        if any {
            let node0 = [node(lo[0]), node(lo[1]), k_node0];
            out.push(crate::lattice::LatticeExtent {
                face,
                node0,
                dims: [
                    (node(hi[0]) - node0[0] + 2) as usize,
                    (node(hi[1]) - node0[1] + 2) as usize,
                    k_dims,
                ],
            });
        }
    }
    out
}

/// The cavern value of a cell by its SITE: the chunk's own lattice for a cell of its face, the
/// partner's lattice for a cell across a seam. A corner PHANTOM has no lattice and no cave: the
/// extractor never reads a phantom's bytes (its corner group is a prism of the three real columns),
/// so its value is zero by rule. The same rule from a partial chunk's cell pass and from the halo,
/// so a cell beyond the face IS the partner's cell, byte for byte.
#[inline]
pub(crate) fn cavern_of(
    own: &NodeLattice,
    foreign: &[NodeLattice],
    site: crate::lattice::Site,
    k: i32,
) -> Gi {
    if site.face == own.face.index() {
        return own.value_at([site.i, site.j, k]);
    }
    for lattice in foreign {
        if site.face == lattice.face.index() {
            return lattice.value_at([site.i, site.j, k]);
        }
    }
    Gi::ZERO
}

/// The cavern field at one global node: the node's own direction on its face, at the corner
/// radius `r` (in whole gap steps) of its radial index. Shared by the cell pass and the halo.
pub(crate) fn node_value(charter: &PlanCharter, face: Face, i: i32, j: i32, r: Gi) -> Gi {
    vd_recipe::plan::node_value(charter, i32::from(face.index()), i, j, r)
}

/// What a cell's column and radial layer state about it: the inputs of the per-cell tail.
pub(crate) struct CellSite {
    /// The column's direction, at the bend's fraction bits.
    pub dir: [Gi; 3],
    /// The column's surface radius, in gap steps at [`LENGTH_BITS`].
    pub h: Gi,
    pub biome: Biome,
    /// The cell centre's radius in WHOLE gap steps — exact.
    pub r_steps: Gi,
    /// The column's water surface radius at [`LENGTH_BITS`], or ZERO for a dry column.
    pub water: Gi,
}

/// ★ THE PER-CELL TAIL IS THE RECIPE'S KERNEL (ruling F7, step G1). This function only names the
/// crate's own types on either side of `vd_recipe::cell::cell_word`: the charter and the cell's
/// numbers go in as words, one word comes back, and the substance code becomes a [`Stratum`] again.
/// So the cell pass, the halo (slice 6) and the shader all run ONE arithmetic, and a halo cell one
/// chunk computes is byte-identical to the same cell in the chunk that owns it.
///
/// **Example.** The cell under the pilot's boots is asked by the shard for collision and by the
/// card for the picture. Both calls land here, and here they land on the same function.
#[inline]
pub(crate) fn finish_cell(
    charter: &CellCharter,
    site: &CellSite,
    value: Gi,
    tubes: &[Tube],
) -> Cell {
    cell_of_word(cell_word(
        charter,
        &CellAt {
            dir: site.dir,
            h: site.h,
            biome: Gi::new(site.biome as i64),
            r_steps: site.r_steps,
            water: site.water,
        },
        value,
        tubes,
    ))
}

/// Generate one chunk; `None` for a key outside the body's ladder. The column pass and the cell
/// pass in one call, for a host that needs one chunk.
#[must_use]
pub fn generate(
    body: &BodyDefinition,
    field: Option<&dyn crate::artifact::ZField>,
    key: ChunkKey,
) -> Option<ChunkLattice> {
    let column = column_field(body, field, key.face, key.rung, key.x, key.y)?;
    generate_in(body, &column, key.z)
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
    use crate::home::home_planet;
    use vd_seed::ladder::cell_m;

    fn key(face: Face, rung: u8, x: i32, y: i32, z: i32) -> ChunkKey {
        ChunkKey {
            face,
            rung,
            x,
            y,
            z,
        }
    }

    fn surface_z(m: &BodyDefinition, face: Face, rung: u8, x: i32, y: i32) -> i32 {
        crate::digest::surface_chunk_z(m, face, rung, x, y)
    }

    /// The column of the first `count` sampled columns whose surface stands deepest under the sea,
    /// by the column pass alone (cheap), with how deep.
    fn deepest_sea_column(m: &BodyDefinition, rung: u8, count: i32) -> (Face, i32, i32, Gi) {
        let chunks = m.ladder.cells_per_edge(rung) as i32 / CHUNK_EDGE as i32;
        let mut best = (Face::PosX, 0, 0, Gi::ZERO);
        let mut i = 0;
        while i < count {
            let face = Face::ALL[(i % 6) as usize];
            let x = (i * 7919) % chunks;
            let y = (i * 104_729) % chunks;
            let column = column_field(m, None, face, rung, x, y).expect("a column");
            let depth = m.sea_radius - column.highest;
            if (i == 0) | (depth > best.3) {
                best = (face, x, y, depth);
            }
            i += 1;
        }
        best
    }

    /// A whole number of metres as gap steps at the length format's fraction bits.
    fn s(metres: i64) -> Gi {
        Gi::new(metres * STEPS_PER_M) << LENGTH_BITS
    }

    /// ★ THE CHARTER'S PACKED ROWS ARE THE DEPTH TABLE — every biome, every whole metre the table
    /// covers and one past its end, on three bodies whose seeds draw different sediments and
    /// different bedrock. The recipe's kernel reads the rows and never this crate's table, so this
    /// is the measurement that the two say the same thing; without it a renumbering of the
    /// substances would move the world's bytes and nothing would go red.
    #[test]
    fn the_charters_packed_rows_read_what_the_depth_table_reads() {
        for seed in [0x5EEDu64, 0xA11CE, 0xF00D] {
            let m = BodyDefinition::from_seed(seed, 6_371_000.0, crate::home::home_facts())
                .expect("a body");
            let charter = charter_of(&m, 0, CHUNK_EDGE);
            let deepest = m.strata.max_depth_m() + 5;
            for biome in Biome::ALL {
                let mut depth = 0u32;
                while depth <= deepest {
                    let code = charter
                        .stratum_code(Gi::new(biome as i64), Gi::new(i64::from(depth)))
                        .raw() as u8;
                    assert_eq!(
                        Stratum::from_code(code),
                        Some(m.strata.at(biome, depth)),
                        "seed {seed:x}, {biome:?} at {depth} m"
                    );
                    depth += 1;
                }
            }
            // The fluid rule, on either side of this body's own sea.
            // The fluid rule, on either side of a stated sea (a seed-built body has none).
            let wet = m.with_sea_m(Some(0));
            let charter_wet = charter_of(&wet, 0, CHUNK_EDGE);
            assert_eq!(
                charter_wet.fluid_code(wet.sea_radius - Gi::ONE).raw() as u8,
                Stratum::Water.code()
            );
            assert_eq!(
                charter_wet.fluid_code(wet.sea_radius).raw() as u8,
                Stratum::Air.code()
            );
            assert_eq!(charter.sea_radius, Gi::ZERO);
            assert_eq!(charter.fluid_code(Gi::ONE).raw() as u8, Stratum::Air.code());
            // The bedrock the skip writes is the table's own deepest answer.
            assert_eq!(
                below_surface_cell(&charter).stratum,
                m.strata.at(Biome::Grassland, deepest)
            );
            // ★ EVERY BIOME READS ITS OWN ROW, AND NO TWO SHARE ONE. The kernel picks a row by a
            // MASK over the biome's own discriminant, so a row that folded onto another's would
            // read the wrong substance for a whole climate and nothing would go red. Two
            // measurements say it cannot: the row a biome picks is the row AT ITS DISCRIMINANT,
            // and the four topsoils are four different substances.
            let mut topsoils = std::collections::BTreeSet::new();
            for biome in Biome::ALL {
                let row = charter.strata[biome as usize];
                let picked = charter.stratum_code(Gi::new(biome as i64), Gi::ZERO);
                assert_eq!(
                    picked,
                    (row >> vd_recipe::cell::ROW_TOPSOIL) & Gi::new(0xFF),
                    "seed {seed:x}, {biome:?} reads the row at its own discriminant"
                );
                assert!(
                    topsoils.insert(picked),
                    "seed {seed:x}, {biome:?} shares its row with another biome"
                );
            }
            assert_eq!(topsoils.len(), Biome::ALL.len());
        }
    }

    /// A GARBLED WORD IS REFUSED, NEVER A PANIC (the decode is total): a low byte no stratum owns
    /// reads solid rock, and every byte a charter can write reads its own stratum.
    #[test]
    fn a_cell_word_whose_code_names_no_stratum_reads_the_refused_rock() {
        let word = |code: i64, gap: i64| vd_recipe::cell::pack(Gi::new(code), Gi::new(gap));
        // Every code this crate's own charter can write reads back as itself.
        for stratum in Stratum::ALL {
            let cell = cell_of_word(word(i64::from(stratum.code()), -7));
            assert_eq!(cell.stratum, stratum);
            assert_eq!(cell.gap, -7);
        }
        // A byte past the last stratum, and the widest byte a word can carry: refused, not a panic,
        // and the gap still reads what the word says.
        let past = Stratum::ALL.len() as i64;
        assert_eq!(cell_of_word(word(past, 42)).stratum, REFUSED_STRATUM);
        assert_eq!(cell_of_word(word(past, 42)).gap, 42);
        assert_eq!(cell_of_word(word(255, 0)).stratum, REFUSED_STRATUM);
        assert!(REFUSED_STRATUM.is_solid(), "a refusal is never a hole");
    }

    #[test]
    fn the_gap_quantises_to_a_signed_byte_and_clamps() {
        // The steps arrive already floored by the one shift, so this is the clamp and nothing else.
        assert_eq!(quantise_gap(Gi::ZERO), 0, "on the surface: zero, and air");
        assert_eq!(quantise_gap(Gi::new(-39)), -39);
        assert_eq!(quantise_gap(Gi::new(64)), 64);
        assert_eq!(
            quantise_gap(Gi::new(128)),
            127,
            "one cell clamps to the top code, one step short"
        );
        assert_eq!(quantise_gap(Gi::new(-128)), -128);
        assert_eq!(quantise_gap(Gi::new(-5_000)), -128);
        assert_eq!(quantise_gap(Gi::new(9 * 128)), 127);
    }

    #[test]
    fn a_surface_chunk_is_evaluated_and_its_cells_are_rock_below_and_air_above() {
        let m = home_planet();
        let z = surface_z(&m, Face::PosX, 0, 300, 700);
        let chunk = generate(&m, None, key(Face::PosX, 0, 300, 700, z)).expect("in the ladder");
        assert_eq!(chunk.how, How::Evaluated);
        assert_eq!(chunk.cells.len(), CHUNK_CELLS);
        let mut air = 0;
        let mut rock = 0;
        let mut a = 0;
        while a < CHUNK_EDGE {
            let mut c = 0;
            while c < CHUNK_EDGE {
                let cell = chunk.cell(a, 3, c);
                if cell.stratum.is_solid() {
                    assert!(cell.gap < 0, "solid is under the surface");
                    rock += 1;
                } else {
                    assert!(cell.gap >= 0, "air is over the surface");
                    air += 1;
                }
                c += 1;
            }
            a += 1;
        }
        assert!(
            air > 0,
            "the surface passes through: {air} air, {rock} rock"
        );
        assert!(
            rock > 0,
            "the surface passes through: {air} air, {rock} rock"
        );
        assert_eq!(
            chunk,
            generate(&m, None, chunk.key).expect("again"),
            "the same bytes"
        );
        // The column pass is shared: the chunk is the same whether built alone or in the column.
        let column = column_field(&m, None, Face::PosX, 0, 300, 700).expect("a column");
        assert_eq!(generate_in(&m, &column, z), Some(chunk));
        assert!(column.lowest <= column.highest);
        assert_eq!(column.columns.len(), CHUNK_EDGE * CHUNK_EDGE);
        assert_eq!(generate_in(&m, &column, -1), None, "below the band");
        let other = column_field(&m, None, Face::PosX, 0, 301, 700).expect("a column");
        let mut short = other.clone();
        short.columns.truncate(5);
        assert_eq!(
            generate_in(&m, &short, z),
            None,
            "a broken column pass is refused"
        );
    }

    /// The refuter's finding 1: a skip must write exactly the bytes the cell pass would. Both skips,
    /// compared cell for cell against the forced cell pass, at a fine rung and at the coarsest.
    #[test]
    fn a_skipped_chunk_is_byte_identical_to_its_cell_pass() {
        let m = home_planet();
        let mut checked = 0;
        // A fine rung, a middle rung and the coarsest: the last holds only a chunk or two per column.
        for rung in [0u8, 5, m.ladder.rungs - 1] {
            // The coarsest rung has ONE column a face, so the sample column is clamped to what the
            // rung actually holds.
            let columns = (m.ladder.cells_per_edge(rung) as i32 - 1) / CHUNK_EDGE as i32;
            let (x, y) = (10.min(columns), 10.min(columns));
            let column = column_field(&m, None, Face::NegZ, rung, x, y).expect("a column");
            let top = (m.ladder.cells_in_band(rung) as i32 - 1) / CHUNK_EDGE as i32;
            let zs = surface_z(&m, Face::NegZ, rung, x, y);
            for z in [0, 1, zs - 3, zs - 1, zs + 1, zs + 3, top - 1, top] {
                if z < 0 || z > top {
                    continue;
                }
                let made = generate_in(&m, &column, z).expect("in the band");
                let forced = cell_pass(&m, &column, made.key);
                assert_eq!(
                    made.cells, forced.cells,
                    "rung {rung} z {z} ({:?}): a skip must equal the cell pass",
                    made.how
                );
                checked += 1;
            }
        }
        assert!(checked >= 16, "enough chunks compared: {checked}");
    }

    #[test]
    fn chunks_wholly_above_and_below_the_surface_skip_every_cell() {
        // A sea at the ladder radius, stated (C5: the seed draws none); the relief stands on both
        // sides of it, so some columns are under water.
        let m = home_planet().with_sea_m(Some(0));
        let z = surface_z(&m, Face::NegZ, 0, 10, 10);
        let top = (m.ladder.cells_in_band(0) as i32 - 1) / CHUNK_EDGE as i32;
        let above = generate(&m, None, key(Face::NegZ, 0, 10, 10, top)).expect("in the band");
        assert_eq!(above.how, How::AboveSurface);
        assert!(above.cells.iter().all(|c| !c.stratum.is_solid()));
        assert!(above.cells.iter().all(|c| c.gap == i8::MAX));
        let below = generate(&m, None, key(Face::NegZ, 0, 10, 10, 0)).expect("in the band");
        assert_eq!(below.how, How::BelowSurface);
        assert!(below.cells.iter().all(|c| c.gap == i8::MIN));
        assert!(
            below
                .cells
                .iter()
                .all(|c| c.stratum == m.strata.bedrock.stratum())
        );
        assert!(top > z, "the band reaches above the surface");
        // Outside the ladder: refused, never a default chunk.
        assert_eq!(
            generate(&m, None, key(Face::PosX, m.ladder.rungs, 0, 0, 0)),
            None,
            "past the top rung"
        );
        assert_eq!(generate(&m, None, key(Face::PosX, 0, -1, 0, 0)), None);
        assert_eq!(generate(&m, None, key(Face::PosX, 0, 0, -1, 0)), None);
        assert_eq!(generate(&m, None, key(Face::PosX, 0, 0, 0, -1)), None);
        assert_eq!(
            generate(&m, None, key(Face::PosX, 0, 0, 0, top + 1)),
            None,
            "past the band"
        );
        assert_eq!(
            generate(&m, None, key(Face::PosX, 0, 1 << 20, 0, 0)),
            None,
            "past the face"
        );
        assert_eq!(column_field(&m, None, Face::PosX, 0, 0, 1 << 20), None);
        // A chunk above the SEABED but under the SEA: skipped as above every surface, and filled with
        // water where the cell is under the sea. RE-MEASURED 2026-09-16 on slice 8a stage 2 (the
        // ridged band): the deepest of 1 000 sampled columns at rung 2 (4 m cells, 248 m chunks)
        // stands 1 382 m under the sea, more than a chunk and two cells, so the first chunk above its
        // highest surface that is skipped as "above" still starts under the sea and must hold water.
        //
        // ★ WHY THE SCAN WIDENED FROM 300 COLUMNS TO 1 000, and it is a MEASUREMENT. Ridged noise is
        // ONE-SIDED, so the ground ROSE where the crests stand while the sea's own draw did not move
        // (the sea is solved against the lifted field in stage 6, `slice_8a_design.md` §1.6). The wet
        // share of the surface therefore fell, and the SAME 300 scattered columns now meet nothing
        // deeper than 126 m — a third of a chunk. One thousand columns meet 1 382 m, and ten thousand
        // meet 2 389 m, so the deep sea is still there and the scan was simply too thin to find it.
        // (The wet share itself is UNMEASURED at this stage; stage 6 solves the sea and states it.)
        let rung = 2;
        let (face, x, y, depth) = deepest_sea_column(&m, rung, 1_000);
        let chunk_m = s(i64::from(cell_m(rung)) * CHUNK_EDGE as i64);
        let two_cells = s(2 * i64::from(cell_m(rung)));
        assert!(
            depth > chunk_m + two_cells,
            "the home planet's sea stands more than a chunk deep somewhere: {depth:?}"
        );
        let column = column_field(&m, None, face, rung, x, y).expect("a column");
        let highest_m = (column.highest >> (LENGTH_BITS + 7)).raw();
        let z_high = ((highest_m - i64::from(m.ladder.floor_m))
            / (i64::from(cell_m(rung)) * CHUNK_EDGE as i64)) as i32;
        // The chunk right above the one that holds the highest surface: on THE world (SL5, one world,
        // so this is a pinned fact, not luck) its lowest cell centre stands more than a cell over that
        // surface, and its floor is still under the sea.
        let z = z_high + 1;
        assert!(
            corner_radius(&m, z * CHUNK_EDGE as i32, rung) < m.sea_radius,
            "the chunk's floor is under the sea"
        );
        let under_sea = generate_in(&m, &column, z).expect("in the band");
        assert_eq!(under_sea.how, How::AboveSurface);
        assert!(under_sea.cells.iter().any(|c| c.stratum == Stratum::Water));
        assert!(under_sea.cells.iter().all(|c| c.gap == i8::MAX));
    }

    /// ★ THE WATER PER COLUMN (C5): with a field whose nearest rows say "water at this level" and
    /// "coast", a column's water is that level as a radius and its biome is the beach's sand; a
    /// dry row leaves the column dry whatever the body's sea; a field with no water word (a
    /// pyramid level) reads the body's sea. A chunk over such water holds water cells above its
    /// rock, and the box's water column carries the same levels to the sheet.
    #[test]
    fn a_fields_rows_give_each_column_its_water_and_mark_the_coast() {
        use crate::artifact::{DRY_M, SparseRows, ZField, nodes_of_chunk};
        let moon = crate::home::home_moon().with_sea_m(Some(-500));
        let lattice = moon.macro_lattice().expect("a lattice");
        let key = |x: i32| ChunkKey {
            face: Face::PosZ,
            rung: 0,
            x,
            y: 4_000,
            z: 0,
        };
        // Three fields over the same nodes: the sea at +200 m with the coast bit, dry rows, and
        // the pyramid (no water word).
        let mut sea = SparseRows::default();
        let mut dry = SparseRows::default();
        for node in nodes_of_chunk(&lattice, key(4_000)) {
            sea.0.insert(node, (0, 200, crate::solve::FACIES_COAST));
            dry.0.insert(node, (0, DRY_M, 0));
        }
        let wet = column_field(&moon, Some(&sea), Face::PosZ, 0, 4_000, 4_000).expect("columns");
        let level = moon.radius + (Gi::new(200 * STEPS_PER_M) << LENGTH_BITS);
        assert!(wet.water.iter().all(|&w| w == level));
        // The columns stand at the field's zero plus the fine octaves: under 200 m, so the coast
        // bit marks no beach where the water covers the ground, and sand where it stands above.
        let beaches = wet
            .columns
            .iter()
            .zip(&wet.water)
            .filter(|(c, w)| c.1 >= **w && c.2 == Biome::Desert)
            .count();
        let drowned = wet
            .columns
            .iter()
            .zip(&wet.water)
            .filter(|(c, w)| c.1 < **w)
            .count();
        assert_eq!(beaches + drowned, CHUNK_EDGE * CHUNK_EDGE);
        let arid = column_field(&moon, Some(&dry), Face::PosZ, 0, 4_000, 4_000).expect("columns");
        assert!(arid.water.iter().all(|&w| w == Gi::ZERO));
        // The coast bit over a water that stands UNDER every column: every column is a beach.
        let mut shore = SparseRows::default();
        for node in nodes_of_chunk(&lattice, key(4_000)) {
            shore
                .0
                .insert(node, (0, -3_000, crate::solve::FACIES_COAST));
        }
        let beach =
            column_field(&moon, Some(&shore), Face::PosZ, 0, 4_000, 4_000).expect("columns");
        assert!(beach.columns.iter().all(|(_, _, b)| *b == Biome::Desert));
        // A column field of the wrong width makes no chunk.
        let torn_field = ColumnField {
            columns: Vec::new(),
            water: Vec::new(),
            sites: Vec::new(),
            ..wet.clone()
        };
        assert!(generate_in(&moon, &torn_field, 0).is_none());
        let level_1 = crate::artifact::PyramidField {
            level: 1,
            z_m: vec![0; lattice.coarser(1).expect("a level").node_count()],
        };
        let coarse =
            column_field(&moon, Some(&level_1), Face::PosZ, 0, 4_000, 4_000).expect("columns");
        assert!(coarse.water.iter().all(|&w| w == moon.sea_radius));
        assert_eq!(level_1.water_facies(0), None);
        // A level the lattice cannot coarsen to, a cache missing the tile, and a body with no
        // macro lattice all make no column field.
        let level_9 = crate::artifact::PyramidField {
            level: 9,
            z_m: vec![],
        };
        assert!(column_field(&moon, Some(&level_9), Face::PosZ, 0, 4_000, 4_000).is_none());
        let torn = crate::artifact::TileCache::new(lattice.edge);
        assert!(column_field(&moon, Some(&torn), Face::PosZ, 0, 4_000, 4_000).is_none());
        let rock = moon.without_macro_lattice();
        assert!(rock.macro_lattice().is_none());
        assert!(column_field(&rock, Some(&torn), Face::PosZ, 0, 0, 0).is_none());
        // Under a sea three kilometres up, the chunk right above the highest surface is skipped
        // as above every surface and filled with WATER from the columns' own level; the box
        // carries the levels to the sheet.
        let mut ocean = SparseRows::default();
        for node in nodes_of_chunk(&lattice, key(4_000)) {
            ocean.0.insert(node, (0, 3_000, 0));
        }
        let drowned_field =
            column_field(&moon, Some(&ocean), Face::PosZ, 0, 4_000, 4_000).expect("columns");
        let highest_m = (drowned_field.highest >> (LENGTH_BITS + 7)).raw();
        let z_high = ((highest_m - i64::from(moon.ladder.floor_m))
            / (i64::from(cell_m(0)) * CHUNK_EDGE as i64)) as i32;
        let under = generate_in(&moon, &drowned_field, z_high + 1).expect("in the band");
        assert_eq!(under.how, How::AboveSurface);
        assert!(under.cells.iter().all(|c| c.stratum == Stratum::Water));
        let bx = crate::lattice::sample_box(&moon, Some(&sea), key(4_000)).expect("a box");
        assert_eq!(
            bx.water.len(),
            crate::lattice::BOX_EDGE * crate::lattice::BOX_EDGE
        );
        assert_eq!(
            bx.water[crate::lattice::SampleBox::column_index(5, 5)],
            level
        );
        // The halo past the core reads the body's sea.
        assert_eq!(
            bx.water[crate::lattice::SampleBox::column_index(-1, 5)],
            moon.sea_radius
        );
    }

    #[test]
    fn a_coarse_rung_evaluates_the_same_surface_with_fewer_octaves_and_water_stands_under_the_sea()
    {
        let m = home_planet().with_sea_m(Some(0));
        let rung = 3;
        let z = surface_z(&m, Face::PosY, rung, 20, 20);
        let chunk = generate(&m, None, key(Face::PosY, rung, 20, 20, z)).expect("in the ladder");
        assert_eq!(chunk.how, How::Evaluated);
        assert_eq!(chunk.cells.len(), CHUNK_CELLS);
        // The sea stands where the surface dips under it: the deepest sampled column's surface chunk
        // holds water and, below the water, rock.
        let (face, x, y, depth) = deepest_sea_column(&m, rung, 300);
        assert!(depth > Gi::ZERO, "the home planet has a sea");
        let zs = surface_z(&m, face, rung, x, y);
        let sea = generate(&m, None, key(face, rung, x, y, zs)).expect("in the ladder");
        assert!(sea.cells.iter().any(|c| c.stratum == Stratum::Water));
        assert!(
            sea.cells.iter().any(|c| c.stratum.is_solid()),
            "the sea floor is rock in the same chunk"
        );
    }

    #[test]
    fn caves_hollow_some_cells_under_a_fine_surface_and_none_at_a_coarse_rung() {
        let m = home_planet();
        let z = surface_z(&m, Face::PosZ, 0, 40, 41);
        let column = column_field(&m, None, Face::PosZ, 0, 40, 41).expect("a column");
        assert!(z >= 6, "the surface chunk sits well above the floor");
        let mut hollow = 0;
        let mut dz = 1;
        while dz <= 6 {
            let ch = generate_in(&m, &column, z - dz).expect("in the band");
            // A cave cell is air with a non-negative gap; a bedrock chunk holds none.
            hollow += ch
                .cells
                .iter()
                .filter(|c| c.stratum == Stratum::Air)
                .filter(|c| c.gap >= 0)
                .count();
            dz += 1;
        }
        assert!(hollow > 0, "some cave cells under the fine surface");
        // At a coarse rung the same ground carries no cave: the cell is wider than the detail.
        let rung = m.ladder.rungs - 1;
        assert!(!tubes_carve_at(&m, rung));
        assert!(!caverns_carve_at(&m, rung));
        let zc = surface_z(&m, Face::PosZ, rung, 0, 0);
        let coarse = generate(&m, None, key(Face::PosZ, rung, 0, 0, zc)).expect("in the ladder");
        assert_eq!(coarse.how, How::Evaluated);
    }

    /// The refuter's finding 2: the cavern field is one global field. Two neighbouring chunks sample
    /// the same global nodes, so a cell reads the same value whichever chunk holds it, and the
    /// interpolation is exact on every node.
    #[test]
    fn the_cavern_lattice_is_continuous_across_a_chunk_edge() {
        let blank = |node0: [i32; 3]| NodeLattice {
            face: Face::PosX,
            node0,
            dims: [CAVERN_NODES; 3],
            values: vec![Gi::ZERO; CAVERN_NODES * CAVERN_NODES * CAVERN_NODES],
        };
        let one = Gi::ONE << LENGTH_BITS;
        let mut lattice_a = blank([0, 0, 0]);
        let mut lattice_b = blank([15, 0, 0]);
        // Chunk A covers cells 0..62 (global nodes 0..=16); chunk B covers 62..124 (node0 = 15,
        // global nodes 15..=31). Give global node 16 the value one in both.
        lattice_a.values[16] = one;
        lattice_b.values[1] = one;
        // Global cell 63 sits in chunk B, three quarters of the way from node 15 (cell 60) to node
        // 16 (cell 64); chunk A holds cell 61, a quarter of the way. The weights are exact words.
        assert_eq!(lattice_b.value_at([63, 0, 0]), (one >> 2) * Gi::new(3));
        assert_eq!(lattice_a.value_at([61, 0, 0]), one >> 2);
        assert_eq!(lattice_b.value_at([64, 0, 0]), one, "exact on the node");
        assert_eq!(lattice_a.value_at([0, 0, 0]), Gi::ZERO);
        // An empty lattice is what a rung without caverns holds; it is never read.
        let none = NodeLattice::empty(Face::NegY);
        assert_eq!(none.values.len(), 0);
        assert_eq!(none.face, Face::NegY);
        // On the home planet, two neighbouring evaluated chunks under the surface both carve.
        let m = home_planet();
        let z = surface_z(&m, Face::PosZ, 0, 40, 41) - 2;
        let left = generate(&m, None, key(Face::PosZ, 0, 40, 41, z)).expect("in the band");
        let right = generate(&m, None, key(Face::PosZ, 0, 41, 41, z)).expect("in the band");
        assert_eq!(left.how, How::Evaluated);
        assert_eq!(right.how, How::Evaluated);
    }
}
