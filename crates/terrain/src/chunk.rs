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
//! **Example.** Chunk (face 2, rung 0, 19, 1, 4) of the home planet spans 62 m of radius; its columns'
//! surfaces run through it, so it pays: 3 844 heights once for the column, then 238 328 cells for this
//! chunk. The chunk two above it is skipped: every cell is air at the top code, exactly as the cell pass
//! would write it.

use crate::body::BodyDefinition;
use crate::carve::{
    CAVERN_STRIDE, cavern_hollow_m, cavern_value, caverns_carve_at, tube_hollow_m, tube_region,
    tubes_carve_at, tubes_near,
};
use crate::gf::Gf;
use crate::height::{biome_at, height_m};
use crate::strata::{Biome, Stratum};

use vd_seed::bend::{Face, direction};
use vd_seed::ladder::{self, cell_m};

/// Cells per chunk edge.
pub const CHUNK_EDGE: usize = 62;
/// Cells per chunk.
pub const CHUNK_CELLS: usize = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;
/// Gap steps per cell: the registry's density convention (1/128 of a cell). Cross-pinned against
/// `vd_core::registry::GAP_STEPS_PER_CELL` in `tests/tests/voxel_pins.rs`.
pub const GAP_STEPS_PER_CELL: i64 = 128;
/// The most nodes per axis a chunk's cavern lattice needs: the global nodes that cover its 62 cells.
pub const CAVERN_NODES: usize = CHUNK_EDGE / CAVERN_STRIDE + 2;

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
    /// `62 × 62` entries in packing order `b·62 + a`: the direction, the surface radius, the biome.
    pub columns: Vec<([Gf; 3], Gf, Biome)>,
    /// The same columns' sites: a cell of this face, or — in a PARTIAL chunk at a face's far edge —
    /// the partner face's cell or a corner phantom (slice 6, `lattice::site_of`).
    pub sites: Vec<crate::lattice::Site>,
    /// The least and the greatest surface radius among the columns.
    pub lowest_m: Gf,
    pub highest_m: Gf,
}

impl ChunkLattice {
    /// The cell at `(a, b, c)` inside the chunk, `a` along the face's first axis, `c` radial.
    #[must_use]
    pub fn cell(&self, a: usize, b: usize, c: usize) -> Cell {
        self.cells[(c * CHUNK_EDGE + b) * CHUNK_EDGE + a]
    }
}

/// Quantise a gap in cells to the signed byte: floor to 1/128, clamped to `[−128, 127]`.
#[must_use]
pub fn quantise_gap(gap_cells: Gf) -> i8 {
    let steps = (gap_cells * Gf::from_i64(GAP_STEPS_PER_CELL))
        .floor()
        .to_i64_floor();
    steps.clamp(i64::from(i8::MIN), i64::from(i8::MAX)) as i8
}

/// Whether a chunk key names a chunk of the body at all.
pub(crate) fn in_ladder(body: &BodyDefinition, key: ChunkKey) -> bool {
    if key.rung >= body.ladder.rungs || key.x < 0 || key.y < 0 || key.z < 0 {
        return false;
    }
    let n_l = body.ladder.cells_per_edge(key.rung) as i64;
    let band = i64::from(body.ladder.cells_in_band(key.rung));
    let edge = CHUNK_EDGE as i64;
    i64::from(key.x) * edge < n_l && i64::from(key.y) * edge < n_l && i64::from(key.z) * edge < band
}

/// The direction of the cell `(i, j)` of a face at a rung with `n_l` cells per edge.
pub(crate) fn dir_of(face: Face, n_l: u32, i: i32, j: i32) -> [Gf; 3] {
    let d = direction(face, ladder::face_param(i, n_l), ladder::face_param(j, n_l));
    [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])]
}

/// The column pass for the column of `(face, rung, x, y)`; `None` for a column outside the body.
#[must_use]
pub fn column_field(
    body: &BodyDefinition,
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
    let mut sites = Vec::with_capacity(CHUNK_EDGE * CHUNK_EDGE);
    let mut lowest = Gf::from_f64(f64::INFINITY);
    let mut highest = Gf::from_f64(f64::NEG_INFINITY);
    let mut b = 0;
    while b < CHUNK_EDGE {
        let mut a = 0;
        while a < CHUNK_EDGE {
            let site = crate::lattice::site_of(body, key, a as i32, b as i32);
            let dir = crate::lattice::site_dir(body, key, site);
            let h = height_m(body, dir, rung);
            let biome = biome_at(body, dir, h);
            lowest = lowest.lesser(h);
            highest = highest.greater(h);
            columns.push((dir, h, biome));
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
        sites,
        lowest_m: lowest,
        highest_m: highest,
    })
}

/// THE FLUID at a radius: water under the sea, air over it. The one rule the above-surface skip,
/// the cell pass and the halo share.
#[must_use]
pub fn fluid_at(body: &BodyDefinition, r: Gf) -> Stratum {
    if r < body.sea_radius_m {
        Stratum::Water
    } else {
        Stratum::Air
    }
}

/// The cell more than a cell above every surface: the fluid at its radius, at the top code.
#[must_use]
pub fn above_surface_cell(body: &BodyDefinition, r: Gf) -> Cell {
    Cell {
        stratum: fluid_at(body, r),
        gap: i8::MAX,
    }
}

/// The cell more than a cell below every surface, stratum and cave: bedrock at the bottom code.
#[must_use]
pub fn below_surface_cell(body: &BodyDefinition) -> Cell {
    Cell {
        stratum: body.strata.bedrock.stratum(),
        gap: i8::MIN,
    }
}

/// A chunk filled from one rule per radial layer: the skip's fill.
fn filled(key: ChunkKey, mut layer: impl FnMut(usize) -> (Stratum, i8), how: How) -> ChunkLattice {
    let mut cells = Vec::with_capacity(CHUNK_CELLS);
    let mut c = 0;
    while c < CHUNK_EDGE {
        let (stratum, gap) = layer(c);
        let mut n = 0;
        while n < CHUNK_EDGE * CHUNK_EDGE {
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
    let cell_m_f = Gf::from_i64(i64::from(cell_m(rung)));
    let r_low = Gf::from_f64(body.ladder.corner_radius_m(k0, rung));
    let r_high = Gf::from_f64(body.ladder.corner_radius_m(k0 + edge, rung));
    let reach = reach_m(body, cell_m_f);
    // Above: the lowest cell CENTRE (r_low + ½ cell) is more than a cell over the highest surface,
    // so every gap is ≥ 1 and clamps to the top code; air or water by the cell's radius.
    if r_low - Gf::HALF * cell_m_f > column.highest_m {
        return Some(filled(
            key,
            |c| {
                let r = Gf::from_f64(body.ladder.cell_radius_m(k0 + c as i32, rung));
                let cell = above_surface_cell(body, r);
                (cell.stratum, cell.gap)
            },
            How::AboveSurface,
        ));
    }
    // Below: the highest cell CENTRE (r_high − ½ cell) is more than a cell under the lowest surface
    // less every reach, so every gap is ≤ −1 and clamps to the bottom code, and every cell is past
    // the deepest stratum and the deepest cave: bedrock.
    if r_high + Gf::HALF * cell_m_f < column.lowest_m - reach {
        let cell = below_surface_cell(body);
        return Some(filled(key, |_| (cell.stratum, cell.gap), How::BelowSurface));
    }
    Some(cell_pass(body, column, key))
}

/// How far under a surface the strata and the caves can still change a cell, in metres, at a rung.
pub(crate) fn reach_m(body: &BodyDefinition, cell_m_f: Gf) -> Gf {
    let carve_any = tubes_carve_at(body, cell_m_f) | caverns_carve_at(body, cell_m_f);
    let cave_reach = if carve_any {
        Gf::from_i64(i64::from(body.caves.max_depth_m)) + body.caves.cavern_scale_m
    } else {
        Gf::ZERO
    };
    let strata_reach = Gf::from_i64(i64::from(body.strata.max_depth_m())) + Gf::ONE;
    strata_reach.greater(cave_reach)
}

/// The cell pass: every cell of the chunk, in packing order. Public so a test can compare a skipped
/// chunk against it cell for cell.
#[must_use]
pub fn cell_pass(body: &BodyDefinition, column: &ColumnField, key: ChunkKey) -> ChunkLattice {
    let rung = key.rung;
    let edge = CHUNK_EDGE as i32;
    let (i0, j0, k0) = (key.x * edge, key.y * edge, key.z * edge);
    let cell_m_f = Gf::from_i64(i64::from(cell_m(rung)));
    let r_low = Gf::from_f64(body.ladder.corner_radius_m(k0, rung));
    let r_high = Gf::from_f64(body.ladder.corner_radius_m(k0 + edge, rung));
    let carve_tubes = tubes_carve_at(body, cell_m_f);
    let carve_caverns = caverns_carve_at(body, cell_m_f);
    let carve_any = carve_tubes | carve_caverns;
    // The tubes that can reach this chunk: every tube of the regions around the chunk's own, then
    // only those that come within their radius of the chunk's bounding sphere (most chunks keep
    // none, and then no cell pays a segment distance).
    let mut tubes = Vec::new();
    if carve_tubes {
        let centre_col = column.columns[(CHUNK_EDGE / 2) * CHUNK_EDGE + CHUNK_EDGE / 2].0;
        let r_mid = (r_low + r_high) * Gf::HALF;
        let centre = [
            centre_col[0] * r_mid,
            centre_col[1] * r_mid,
            centre_col[2] * r_mid,
        ];
        let mut reach = Gf::ZERO;
        let mut lo = tube_region(body, centre);
        let mut hi = lo;
        let mut corner = 0;
        while corner < 8 {
            let a = if corner & 1 == 0 { 0 } else { CHUNK_EDGE - 1 };
            let b = if corner & 2 == 0 { 0 } else { CHUNK_EDGE - 1 };
            let (dir, _, _) = column.columns[b * CHUNK_EDGE + a];
            let r = if corner & 4 == 0 { r_low } else { r_high };
            let p = [dir[0] * r, dir[1] * r, dir[2] * r];
            let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
            reach = reach.greater((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            let region = tube_region(body, p);
            let mut axis = 0;
            while axis < 3 {
                lo[axis] = lo[axis].min(region[axis]);
                hi[axis] = hi[axis].max(region[axis]);
                axis += 1;
            }
            corner += 1;
        }
        for tube in tubes_near(body, lo, hi) {
            let d = crate::carve::segment_distance_m(tube.start, tube.end, centre);
            if d <= reach + tube.radius_m {
                tubes.push(tube);
            }
        }
    }
    // The cavern lattice on the GLOBAL node grid (one node per `CAVERN_STRIDE` cells of the face and
    // of the radial, counted from the face's and the band's origin), so two neighbouring chunks share
    // their boundary nodes exactly and the field is continuous across every chunk edge. Each node's
    // direction comes from its own face parameter, never from a clamped column. A PARTIAL chunk at a
    // face's far edge holds the partner face's columns as well: those read a lattice of the
    // partner's own nodes, built over exactly the columns present.
    let stride = CAVERN_STRIDE as i32;
    let node0 = [i0 / stride, j0 / stride, k0 / stride];
    let lattice = if carve_caverns {
        NodeLattice::build(body, rung, key.face, node0, [CAVERN_NODES; 3])
    } else {
        NodeLattice::empty(key.face)
    };
    let foreign = if carve_caverns {
        foreign_lattices(body, rung, key.face, &column.sites, node0[2], CAVERN_NODES)
    } else {
        Vec::new()
    };
    let cave_min = Gf::from_i64(i64::from(body.caves.min_depth_m));
    let cave_max = Gf::from_i64(i64::from(body.caves.max_depth_m));
    let rule = CaveRule {
        carve_any,
        cave_min,
        cave_max,
    };
    let mut cells = Vec::with_capacity(CHUNK_CELLS);
    let mut c = 0;
    while c < CHUNK_EDGE {
        let r = Gf::from_f64(body.ladder.cell_radius_m(k0 + c as i32, rung));
        let mut b = 0;
        while b < CHUNK_EDGE {
            let mut a = 0;
            while a < CHUNK_EDGE {
                let (dir, h, biome) = column.columns[b * CHUNK_EDGE + a];
                let site = column.sites[b * CHUNK_EDGE + a];
                let k = k0 + c as i32;
                // A column of this face reads the chunk's node lattice; a partner face's column (a
                // partial chunk at the face's far edge) reads that face's own global nodes, and a
                // corner phantom reads the field at its own centre — the same rule the halo uses,
                // so a partial chunk's cell beyond the face IS the partner's cell, byte for byte.
                // The cavern value is read only where a cave can be — inside the depth band — so a
                // cell far above or far below the surface pays no interpolation.
                let value = if carve_caverns & rule.in_band(h - r) {
                    cavern_of(&lattice, &foreign, site, k)
                } else {
                    Gf::ZERO
                };
                cells.push(finish_cell(
                    body,
                    &CellSite {
                        dir,
                        h,
                        biome,
                        r,
                        cell_m_f,
                    },
                    &rule,
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
    pub values: Vec<Gf>,
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
        let n_l = body.ladder.cells_per_edge(rung);
        let s = CAVERN_STRIDE as i32;
        let mut values = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        let mut nc = 0;
        while nc < dims[2] {
            let k = (node0[2] + nc as i32) * s;
            let r = Gf::from_f64(body.ladder.corner_radius_m(k, rung));
            let mut nb = 0;
            while nb < dims[1] {
                let j = (node0[1] + nb as i32) * s;
                let mut na = 0;
                while na < dims[0] {
                    let i = (node0[0] + na as i32) * s;
                    values.push(node_value(body, face, n_l, i, j, r));
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
    /// weights that are multiples of `1/CAVERN_STRIDE`: exact on every target, and the same number
    /// from whichever lattice holds the nodes.
    #[inline]
    pub(crate) fn value_at(&self, cell: [i32; 3]) -> Gf {
        let s = CAVERN_STRIDE as i32;
        let (na, nb, nc) = (
            (cell[0] / s - self.node0[0]) as usize,
            (cell[1] / s - self.node0[1]) as usize,
            (cell[2] / s - self.node0[2]) as usize,
        );
        let at =
            |x: usize, y: usize, z: usize| self.values[(z * self.dims[1] + y) * self.dims[0] + x];
        let stride = Gf::from_i64(CAVERN_STRIDE as i64);
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
            [
                Gf::from_i32(cell[0] % s) / stride,
                Gf::from_i32(cell[1] % s) / stride,
                Gf::from_i32(cell[2] % s) / stride,
            ],
        )
    }
}

/// The lattices of every PARTNER face present among `sites` (a partial chunk's columns beyond its
/// face, or a halo across a seam): one per face, over the node range those columns need, on the
/// radial node range `[k_node0, k_node0 + k_dims)`.
pub(crate) fn foreign_lattices(
    body: &BodyDefinition,
    rung: u8,
    my_face: Face,
    sites: &[crate::lattice::Site],
    k_node0: i32,
    k_dims: usize,
) -> Vec<NodeLattice> {
    let s = CAVERN_STRIDE as i32;
    let mut out: Vec<NodeLattice> = Vec::new();
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
            let node0 = [lo[0] / s, lo[1] / s, k_node0];
            let dims = [
                (hi[0] / s - node0[0] + 2) as usize,
                (hi[1] / s - node0[1] + 2) as usize,
                k_dims,
            ];
            out.push(NodeLattice::build(body, rung, face, node0, dims));
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
) -> Gf {
    if site.face == own.face.index() {
        return own.value_at([site.i, site.j, k]);
    }
    for lattice in foreign {
        if site.face == lattice.face.index() {
            return lattice.value_at([site.i, site.j, k]);
        }
    }
    Gf::ZERO
}

/// The cavern field at one global node: the node's own direction on its face, at the corner
/// radius `r` of its radial index. Shared by the cell pass and the halo.
pub(crate) fn node_value(body: &BodyDefinition, face: Face, n_l: u32, i: i32, j: i32, r: Gf) -> Gf {
    let dir = dir_of(face, n_l, i, j);
    cavern_value(body, [dir[0] * r, dir[1] * r, dir[2] * r])
}

/// What a cell's column and radial layer state about it: the inputs of the per-cell tail.
pub(crate) struct CellSite {
    pub dir: [Gf; 3],
    pub h: Gf,
    pub biome: Biome,
    pub r: Gf,
    pub cell_m_f: Gf,
}

/// The body's cave band at a rung: whether anything carves, and between which depths.
pub(crate) struct CaveRule {
    pub carve_any: bool,
    pub cave_min: Gf,
    pub cave_max: Gf,
}

impl CaveRule {
    /// Whether a cell at `depth` under the surface can hold a cave at this rung.
    pub(crate) fn in_band(&self, depth: Gf) -> bool {
        self.carve_any & (depth >= self.cave_min) & (depth <= self.cave_max)
    }
}

/// ★ THE PER-CELL TAIL, shared by the cell pass and the halo (slice 6): from a column's surface,
/// a cell's radius, the interpolated cavern value and the tubes that can reach it, the cell's
/// substance and gap. One function, so a halo cell computed by one chunk is byte-identical to the
/// same cell computed by the chunk that owns it.
#[inline]
pub(crate) fn finish_cell(
    body: &BodyDefinition,
    site: &CellSite,
    rule: &CaveRule,
    value: Gf,
    tubes: &[crate::carve::Tube],
) -> Cell {
    let (dir, h, r, cell_m_f) = (site.dir, site.h, site.r, site.cell_m_f);
    let rock_gap_cells = ((r - h) / cell_m_f).clamp(-Gf::ONE, Gf::ONE);
    let depth = h - r;
    let mut gap_cells = rock_gap_cells;
    let mut stratum = if rock_gap_cells >= Gf::ZERO {
        fluid_at(body, r)
    } else {
        body.strata
            .at(site.biome, depth.floor().to_i64_floor().max(0) as u32)
    };
    // The cavern lattice is all zero where caverns do not carve, and the tube list is empty where
    // tubes do not: both contribute nothing there, with no branch.
    if rule.in_band(depth) {
        let p = [dir[0] * r, dir[1] * r, dir[2] * r];
        let hollow_m = cavern_hollow_m(body, value).greater(tube_hollow_m(tubes, p));
        if hollow_m > Gf::ZERO {
            // A hollow is never negative, so the greater is in air: the cell is hollow.
            gap_cells = gap_cells.greater((hollow_m / cell_m_f).clamp(Gf::ZERO, Gf::ONE));
            stratum = Stratum::Air;
        }
    }
    Cell {
        stratum,
        gap: quantise_gap(gap_cells),
    }
}

/// The trilinear blend of eight node values `v[(c·2 + b)·2 + a]` at weights `t`: the one arithmetic
/// the cell pass and the halo share.
#[inline]
pub(crate) fn trilinear8(v: [Gf; 8], t: [Gf; 3]) -> Gf {
    let l = |p: Gf, q: Gf, w: Gf| p + w * (q - p);
    let x00 = l(v[0], v[1], t[0]);
    let x10 = l(v[2], v[3], t[0]);
    let x01 = l(v[4], v[5], t[0]);
    let x11 = l(v[6], v[7], t[0]);
    l(l(x00, x10, t[1]), l(x01, x11, t[1]), t[2])
}

/// Generate one chunk; `None` for a key outside the body's ladder. The column pass and the cell
/// pass in one call, for a host that needs one chunk.
#[must_use]
pub fn generate(body: &BodyDefinition, key: ChunkKey) -> Option<ChunkLattice> {
    let column = column_field(body, key.face, key.rung, key.x, key.y)?;
    generate_in(body, &column, key.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::home_planet;

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
    fn deepest_sea_column(m: &BodyDefinition, rung: u8, count: i32) -> (Face, i32, i32, Gf) {
        let chunks = m.ladder.cells_per_edge(rung) as i32 / CHUNK_EDGE as i32;
        let mut best = (Face::PosX, 0, 0, Gf::from_f64(f64::NEG_INFINITY));
        let mut i = 0;
        while i < count {
            let face = Face::ALL[(i % 6) as usize];
            let x = (i * 7919) % chunks;
            let y = (i * 104_729) % chunks;
            let column = column_field(m, face, rung, x, y).expect("a column");
            let depth = m.sea_radius_m - column.highest_m;
            if depth > best.3 {
                best = (face, x, y, depth);
            }
            i += 1;
        }
        best
    }

    #[test]
    fn the_gap_quantises_to_a_signed_byte_and_clamps() {
        assert_eq!(quantise_gap(Gf::ZERO), 0, "on the surface: zero, and air");
        assert_eq!(quantise_gap(Gf::from_f64(-0.3)), -39, "floor(-38.4)");
        assert_eq!(quantise_gap(Gf::from_f64(0.5)), 64);
        assert_eq!(
            quantise_gap(Gf::ONE),
            127,
            "one cell clamps to the top code, one step short"
        );
        assert_eq!(quantise_gap(-Gf::ONE), -128);
        assert_eq!(quantise_gap(Gf::from_i64(9)), 127);
    }

    #[test]
    fn a_surface_chunk_is_evaluated_and_its_cells_are_rock_below_and_air_above() {
        let m = home_planet();
        let z = surface_z(&m, Face::PosX, 0, 300, 700);
        let chunk = generate(&m, key(Face::PosX, 0, 300, 700, z)).expect("in the ladder");
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
            generate(&m, chunk.key).expect("again"),
            "the same bytes"
        );
        // The column pass is shared: the chunk is the same whether built alone or in the column.
        let column = column_field(&m, Face::PosX, 0, 300, 700).expect("a column");
        assert_eq!(generate_in(&m, &column, z), Some(chunk));
        assert!(column.lowest_m <= column.highest_m);
        assert_eq!(column.columns.len(), CHUNK_EDGE * CHUNK_EDGE);
        assert_eq!(generate_in(&m, &column, -1), None, "below the band");
        let other = column_field(&m, Face::PosX, 0, 301, 700).expect("a column");
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
            let (x, y) = (10, 10);
            let column = column_field(&m, Face::NegZ, rung, x, y).expect("a column");
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
        let m = home_planet();
        let z = surface_z(&m, Face::NegZ, 0, 10, 10);
        let top = (m.ladder.cells_in_band(0) as i32 - 1) / CHUNK_EDGE as i32;
        let above = generate(&m, key(Face::NegZ, 0, 10, 10, top)).expect("in the band");
        assert_eq!(above.how, How::AboveSurface);
        assert!(above.cells.iter().all(|c| !c.stratum.is_solid()));
        assert!(above.cells.iter().all(|c| c.gap == i8::MAX));
        let below = generate(&m, key(Face::NegZ, 0, 10, 10, 0)).expect("in the band");
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
            generate(&m, key(Face::PosX, m.ladder.rungs, 0, 0, 0)),
            None,
            "past the top rung"
        );
        assert_eq!(generate(&m, key(Face::PosX, 0, -1, 0, 0)), None);
        assert_eq!(generate(&m, key(Face::PosX, 0, 0, -1, 0)), None);
        assert_eq!(generate(&m, key(Face::PosX, 0, 0, 0, -1)), None);
        assert_eq!(
            generate(&m, key(Face::PosX, 0, 0, 0, top + 1)),
            None,
            "past the band"
        );
        assert_eq!(
            generate(&m, key(Face::PosX, 0, 1 << 20, 0, 0)),
            None,
            "past the face"
        );
        assert_eq!(column_field(&m, Face::PosX, 0, 0, 1 << 20), None);
        // A chunk above the SEABED but under the SEA: skipped as above every surface, and filled with
        // water where the cell is under the sea. MEASURED on the home planet (the sea stands 5 297 m
        // under the ladder radius and covers about one column in a hundred): the deepest of 300
        // sampled columns at rung 2 (4 m cells, 248 m chunks) stands 489 m under the sea, more than a
        // chunk and two cells, so the first chunk above its highest surface that is skipped as
        // "above" still starts under the sea and must hold water.
        let rung = 2;
        let (face, x, y, depth) = deepest_sea_column(&m, rung, 300);
        let chunk_m = Gf::from_i64(i64::from(cell_m(rung)) * CHUNK_EDGE as i64);
        let two_cells = Gf::from_i64(2 * i64::from(cell_m(rung)));
        assert!(
            depth > chunk_m + two_cells,
            "the home planet's sea stands more than a chunk deep somewhere: {depth:?}"
        );
        let column = column_field(&m, face, rung, x, y).expect("a column");
        let z_high = ((column.highest_m.to_i64_floor() - i64::from(m.ladder.floor_m))
            / (i64::from(cell_m(rung)) * CHUNK_EDGE as i64)) as i32;
        // The chunk right above the one that holds the highest surface: on THE world (SL5, one world,
        // so this is a pinned fact, not luck) its lowest cell centre stands more than a cell over that
        // surface, and its floor is still under the sea.
        let z = z_high + 1;
        assert!(
            Gf::from_f64(m.ladder.corner_radius_m(z * CHUNK_EDGE as i32, rung)) < m.sea_radius_m,
            "the chunk's floor is under the sea"
        );
        let under_sea = generate_in(&m, &column, z).expect("in the band");
        assert_eq!(under_sea.how, How::AboveSurface);
        assert!(under_sea.cells.iter().any(|c| c.stratum == Stratum::Water));
        assert!(under_sea.cells.iter().all(|c| c.gap == i8::MAX));
    }

    #[test]
    fn a_coarse_rung_evaluates_the_same_surface_with_fewer_octaves_and_water_stands_under_the_sea()
    {
        let m = home_planet();
        let rung = 3;
        let z = surface_z(&m, Face::PosY, rung, 20, 20);
        let chunk = generate(&m, key(Face::PosY, rung, 20, 20, z)).expect("in the ladder");
        assert_eq!(chunk.how, How::Evaluated);
        assert_eq!(chunk.cells.len(), CHUNK_CELLS);
        // The sea stands where the surface dips under it: the deepest sampled column's surface chunk
        // holds water and, below the water, rock.
        let (face, x, y, depth) = deepest_sea_column(&m, rung, 300);
        assert!(depth > Gf::ZERO, "the home planet has a sea");
        let zs = surface_z(&m, face, rung, x, y);
        let sea = generate(&m, key(face, rung, x, y, zs)).expect("in the ladder");
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
        let column = column_field(&m, Face::PosZ, 0, 40, 41).expect("a column");
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
        let cell_m_f = Gf::from_i64(i64::from(cell_m(rung)));
        assert!(!tubes_carve_at(&m, cell_m_f));
        assert!(!caverns_carve_at(&m, cell_m_f));
        let zc = surface_z(&m, Face::PosZ, rung, 0, 0);
        let coarse = generate(&m, key(Face::PosZ, rung, 0, 0, zc)).expect("in the ladder");
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
            values: vec![Gf::ZERO; CAVERN_NODES * CAVERN_NODES * CAVERN_NODES],
        };
        let mut lattice_a = blank([0, 0, 0]);
        let mut lattice_b = blank([15, 0, 0]);
        // Chunk A covers cells 0..62 (global nodes 0..=16); chunk B covers 62..124 (node0 = 15,
        // global nodes 15..=31). Give global node 16 the value one in both.
        lattice_a.values[16] = Gf::ONE;
        lattice_b.values[1] = Gf::ONE;
        // Global cell 63 sits in chunk B, three quarters of the way from node 15 (cell 60) to node
        // 16 (cell 64); chunk A holds cell 61, a quarter of the way.
        assert_eq!(lattice_b.value_at([63, 0, 0]), Gf::from_f64(0.75));
        assert_eq!(lattice_a.value_at([61, 0, 0]), Gf::from_f64(0.25));
        assert_eq!(lattice_b.value_at([64, 0, 0]), Gf::ONE, "exact on the node");
        assert_eq!(lattice_a.value_at([0, 0, 0]), Gf::ZERO);
        // An empty lattice is what a rung without caverns holds; it is never read.
        let none = NodeLattice::empty(Face::NegY);
        assert_eq!(none.values.len(), 0);
        assert_eq!(none.face, Face::NegY);
        // On the home planet, two neighbouring evaluated chunks under the surface both carve.
        let m = home_planet();
        let z = surface_z(&m, Face::PosZ, 0, 40, 41) - 2;
        let left = generate(&m, key(Face::PosZ, 0, 40, 41, z)).expect("in the band");
        let right = generate(&m, key(Face::PosZ, 0, 41, 41, z)).expect("in the band");
        assert_eq!(left.how, How::Evaluated);
        assert_eq!(right.how, How::Evaluated);
    }
}
