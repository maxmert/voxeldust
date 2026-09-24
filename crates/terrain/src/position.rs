//! ★ WHERE A VERTEX IS — the one mapping from a mesh vertex in cell space to the body's own frame,
//! shared by the collider (slice 11) and the client (slice 7).
//!
//! A vertex lies inside one group of eight cells. Its position is the trilinear blend of those
//! eight CELL CENTRES (each centre is its column's own direction times its layer's radius), with
//! weights that are exact multiples of `1/256`. The eight terms are summed in ONE canonical order —
//! by the cell's global site, never by the chunk's local axes — so the chunk on either side of a
//! seam, whose local axes differ, computes the same bytes for a shared vertex.
//!
//! ★ **The sum is integers** (ruling F7). A cell centre is its direction at the bend's 40 fraction
//! bits times its radius in whole gap steps, through the two-word product; a weight is the
//! extractor's own quantum, an exact word at [`WEIGHT_BITS`] fraction bits; the eight terms are
//! summed AT FULL PRECISION and shifted ONCE, so no term rounds on its own. Metres are the last
//! step, at the render seam, and only a host outside the recipe asks for them.
//!
//! **At a cube corner** the group is a prism of three real columns (the fourth column is the box's
//! phantom, which no vertex ever weights): the blend is BARYCENTRIC over the three columns' centres,
//! linear along the radial. Barycentric weights are preserved by the affine map between two faces'
//! local layouts of the same three columns, so the three faces that meet at the corner map the one
//! corner vertex to the same metres.
//!
//! Why cell centres and not the face bend continued past the edge: the bend's lines of constant `j`
//! on one face are not the lines of the neighbouring face, so a vertex mapped by one face's bend
//! and by the other's would land in two places (up to most of a cell apart near a corner), and the
//! meshes would crack along every cube edge. Between two real cell centres the interpolation is the
//! same from both sides.
//!
//! **Example.** A vertex on the seam between faces `+X` and `+Y` at rung 0 blends four centres of
//! `+X` and four of `+Y`. Both faces' chunks hold all eight and sum them in the same order, so the
//! client draws one ridge and the shard collides the same one.

use crate::body::BodyDefinition;
use crate::chunk::{CHUNK_EDGE, point_at};
use crate::extract::{ChunkMesh, VERTEX_QUANTUM, phantom_corner};
use crate::lattice::{SampleBox, Site};
use crate::units::{STEPS_PER_M, metres_of_fixed, metres_of_q28};
use std::collections::BTreeMap;
use vd_recipe::Gi;

/// The fraction bits a blend weight carries along ONE axis: the extractor's own quantum is 1/256 of a
/// cell, so eight bits hold a weight exactly.
pub const WEIGHT_BITS: u32 = 8;
const _: () = assert!(1 << WEIGHT_BITS == VERTEX_QUANTUM);

/// The fraction bits the summed position carries: three axes of weight, so the sum of the eight terms
/// is exact and ONE shift at the end reads whole gap steps.
pub const POSITION_BITS: u32 = 3 * WEIGHT_BITS;

/// One at a single axis's weight.
const WEIGHT_ONE: Gi = Gi::new(1 << WEIGHT_BITS);

/// The position of a mesh vertex in the body's frame, in GAP STEPS at [`POSITION_BITS`] fraction bits:
/// the weighted sum of the eight (or six) cell centres, in site order, at full precision.
#[must_use]
pub fn vertex_position(body: &BodyDefinition, samples: &SampleBox, v: [i16; 3]) -> [Gi; 3] {
    let key = samples.key;
    let edge = CHUNK_EDGE as i32;
    let k0 = key.z * edge;
    // The group that holds the vertex, and the vertex's fraction inside it (exact 1/256 steps).
    let mut base = [0i32; 3];
    let mut frac = [Gi::ZERO; 3];
    let mut k = 0;
    while k < 3 {
        let q = i32::from(v[k]);
        // The group is the quantum's own SHIFT and the fraction inside it the quantum's MASK (ruling
        // F7: no `/` and no `%` on a path a GPU kernel will run — this one is step G3). Both floor
        // toward −∞, which is what `div_euclid` did, so a vertex in the halo's `−1` group reads the
        // same group and the same positive fraction it always read.
        let mut cell = q >> WEIGHT_BITS;
        // A vertex exactly on the box's last centre belongs to the last group. No vertex lies below
        // the first centre: every group's origin is at least the halo's `−1`, and a vertex never
        // leaves its group.
        if cell > edge - 1 {
            cell = edge - 1;
        }
        base[k] = cell;
        // The fraction is read from the CLAMPED group, not masked off the quantum: a vertex ON the
        // box's last centre belongs to the last group with the fraction ONE, and a mask would read it
        // as the group's origin instead. (MEASURED: masking moved a halo group's vertex by 38 cm and
        // the composed mesh stopped meeting its neighbour's.)
        frac[k] = Gi::new(i64::from(q - (cell << WEIGHT_BITS)));
        k += 1;
    }
    let centre = |a: i32, b: i32, c: i32| -> ((Site, i32), [Gi; 3]) {
        let dir = samples.dir(a, b);
        let r = Gi::new(body.ladder.cell_radius_steps(k0 + c, key.rung));
        ((samples.site(a, b), k0 + c), point_at(dir, r))
    };
    // The cells with their global sites, centres and weights: eight for a cube group, six for a
    // corner prism. A weight carries three axes' worth of fraction bits.
    let mut terms: Vec<((Site, i32), [Gi; 3], Gi)> = Vec::with_capacity(8);
    match phantom_corner(samples, base) {
        Some((pa, pb)) => {
            // Barycentric over r[0] (opposite the phantom), r[1] (beside it along a), r[2] (beside
            // it along b): the fraction toward r[1] along a is `fa` when r[1] sits at a = 1.
            let u = if pa == 1 {
                frac[0]
            } else {
                WEIGHT_ONE - frac[0]
            };
            let w = if pb == 1 {
                frac[1]
            } else {
                WEIGHT_ONE - frac[1]
            };
            let pos = [[1 - pa, 1 - pb], [pa, 1 - pb], [1 - pa, pb]];
            let weights = [WEIGHT_ONE - u - w, u, w];
            let mut layer = 0;
            while layer < 2 {
                let wc = if layer == 0 {
                    WEIGHT_ONE - frac[2]
                } else {
                    frac[2]
                };
                let mut i = 0;
                while i < 3 {
                    let (site, p) =
                        centre(base[0] + pos[i][0], base[1] + pos[i][1], base[2] + layer);
                    // The prism's weights span two axes only, so the third axis's One makes up the
                    // position's own fraction bits.
                    terms.push((site, p, weights[i] * wc * WEIGHT_ONE));
                    i += 1;
                }
                layer += 1;
            }
        }
        None => {
            let mut corner = 0;
            while corner < 8 {
                let da = corner & 1;
                let db = (corner >> 1) & 1;
                let dc = corner >> 2;
                let wa = if da == 0 {
                    WEIGHT_ONE - frac[0]
                } else {
                    frac[0]
                };
                let wb = if db == 0 {
                    WEIGHT_ONE - frac[1]
                } else {
                    frac[1]
                };
                let wc = if dc == 0 {
                    WEIGHT_ONE - frac[2]
                } else {
                    frac[2]
                };
                let (site, p) = centre(base[0] + da, base[1] + db, base[2] + dc);
                terms.push((site, p, wa * wb * wc));
                corner += 1;
            }
        }
    }
    terms.sort_by(|x, y| x.0.cmp(&y.0));
    let mut p = [Gi::ZERO; 3];
    for (_, centre, w) in &terms {
        p = [
            p[0] + *w * centre[0],
            p[1] + *w * centre[1],
            p[2] + *w * centre[2],
        ];
    }
    p
}

/// THE RENDER SEAM: the same position in METRES, for a host outside the recipe — a client's mesh
/// buffer, a collider's query. The integer sum is the shape; this is the last step out of it.
#[must_use]
pub fn vertex_position_m(body: &BodyDefinition, samples: &SampleBox, v: [i16; 3]) -> [f64; 3] {
    let p = vertex_position(body, samples, v);
    // The sum carries POSITION_BITS of fraction over a whole gap step, and the exit divides by the
    // steps in a metre times those bits — a power of two, so the division is exact.
    [
        metres_of_fixed(p[0], POSITION_BITS),
        metres_of_fixed(p[1], POSITION_BITS),
        metres_of_fixed(p[2], POSITION_BITS),
    ]
}

const _: () = assert!(STEPS_PER_M == 128);

/// The water sheet's points in metres, their unit radials, and its triangles.
pub type WaterSheet = (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[u32; 3]>);

/// ★ THE WATER SHEET ON THE GROUND'S OWN TRIANGLES (2026-09-23, ruling W18; the owner, from
/// 11 500 km: a speckled coast on the coarser rung, from 17 000 km a staircase of 400 km teeth at
/// rung 16, and *"It changes still when I'm leaving farer"* — the third report after W16 and W17).
///
/// Every triangle of the chunk's GROUND mesh that reaches down to the water — some vertex of it
/// stands no higher than the water plus `hide` — gets ONE water triangle: its three points are the
/// ground vertices' OWN directions at the water's radius. The water's level is the HIGHEST among
/// the three vertices' levels, and a vertex's level is the highest water word among the four
/// columns of its group, or the body's sea (★ THE SEA UNDER EVERY CELL, 2026-09-20: on a body with a
/// sea every column holds the sea's level at least, so the sea is one surface at every rung and the
/// shore is where the land crosses it — continuous under the land's morph, never a sheet edge that
/// steps at a ring swap; a lake stands above the sea by its own columns' word). Points in METRES
/// of the body's frame with their unit radials; a water vertex is shared by every triangle at its
/// level. Empty for a dry chunk. The client's mesh is the last step out of it (the floating
/// origin, the single-precision cast).
///
/// ★ **WHY THE GROUND'S TRIANGLES AND NOT A QUAD PER CELL.** The sheet was a quad per CELL on the
/// corner COLUMNS' directions (and one quad per block of eight cells over one level, ruling W16),
/// while the ground is a surface-nets mesh whose vertices lie INSIDE the cells, off the column
/// directions. Two lattices are two chords: a flat quad is a chord of the water's sphere and dips
/// under it by `w² / 4R` at its middle — 10 m at rung 14, 167 m at rung 16, 2.7 km at rung 18 on
/// the home planet — and the ground's vertex stands where the sheet's quad dips, so a sea floor a
/// few metres under the water stood OVER the drawn water. MEASURED (`vd-bins/examples/sheet_poke`,
/// the owner's stands over the berth): of the sea-floor vertices under wet cells, 0.4 % stood above
/// the sheet's chord at rung 13, 8.7 % at rung 14, 8.1–13.7 % at rung 15, 5.5 % at rung 17 and
/// 7.4 % at rung 18, by up to 18 m at rung 14, 137 m at rung 15, 1 958 m at rung 17 and 3 765 m at
/// rung 18 — the speckle, the teeth, and a coast that changed with the rung because each rung's
/// quads dip differently.
///
/// THE LAW: **the water is drawn on the ground's own lattice.** A water point is the ground vertex's
/// direction at the water's radius, so wherever the ground stands under the water the water point
/// is farther from the centre — at the vertices exactly, and along a triangle by the same convex
/// sum of the same three directions — and the ground can cross the water only where it stands over
/// it, which is the shore: a line inside a triangle, the ground's own crossing of one surface, and
/// the morph carries it. No chord is measured because there are not two of them; no rung is named.
/// The block of ruling W16 is retired with the quad.
///
/// ★ THE SHEET UNDER THE LAND IS CUT WHERE IT CAN NEVER SHOW (2026-09-21, the owner: *"when we
/// draw just the planet it is performant, but if water is drawn it becomes slower"* — MEASURED on
/// the coast leg, two flights of one binary: the frame 33.8 ms with the sheet against 24.2 ms
/// without, because the sea under EVERY cell doubled the triangles of every land chunk). A triangle
/// is built only where one of its vertices' ground stands no higher than the water plus `hide`:
/// the land's own morph toward the coarser rung, its sink under the finer one and a cell of
/// placement, which is as far as the drawn land can ever move off its surface. Under that bound the
/// sheet still reaches under the shore; past it the sheet is buried in every state the ladder can
/// draw, and it is not built. A zero `hide` keeps the triangles that touch a wet vertex; the largest
/// word keeps every triangle. The cost is stated: a wet chunk's sheet holds as many triangles as
/// its ground, which is the count the quad per cell had before the block, and the block's saving
/// is given back for a picture with one lattice.
///
/// A water point is the ground vertex's OWN position scaled to the water's radius (a square root
/// and a division at the render seam, both IEEE-754 exact), so the two share their direction to
/// the last bit.
///
/// ★ THE SHEET STANDS AT THE DRAWN GROUND'S OWN RESOLUTION, ON THE COLUMNS' SIDE. The extractor
/// places a vertex at its gap bytes' zero crossing, and a gap byte is 1/128 of a cell
/// ([`crate::chunk::GAP_STEPS_PER_CELL`]): 64 m at rung 13, 2 km at rung 18. The shore law (W6)
/// holds a coastal column as close to the water as its own ground allows — a sea-floor column
/// on a shelf stands under the water by a quarter of its depth, metres — so near a shore the
/// column's depth is SMALLER than the step the drawn ground can show, and the drawn floor lands
/// on either side of the water. MEASURED (`vd-bins/examples/mask_at_pixel`, the patch of dots
/// the owner saw at 11 497 km after the sheet moved onto the ground's triangles: *"some land
/// dissolves"*): in a 200 km window around it every column stands UNDER its water by the shore law
/// and the mask says sea, yet 1 823 of 2 178 drawn vertices at rung 13 and 515 of 587 at rung 14
/// stand more than a metre OVER the water. (A depth tie-break in the shader and a nudge by the
/// vertex buffer's single-precision step were both tried first and refuted: the one lifted the
/// water over every low plain, the other was a thousand times too small.)
///
/// So a vertex's side is its COLUMNS' word — the shore law's surfaces against their own water,
/// never the drawn radius: UNDER where every column of its group stands under, OVER where every
/// one stands over, MIXED at the shore — and the water point is held one extractor's step off the
/// drawn ground on that side wherever the water's own radius would put it nearer: at least
/// `r + q` over a wet vertex, at most `r − q` under a dry one, at the water's radius at a mixed
/// vertex, whose triangle is the shore's own. The water then bulges over a shallow floor by at
/// most one gap step, 1/128 of a cell, under what the rung can draw; the step is the extractor's,
/// the side the recipe's, and nothing is drawn. A host with no heights in its box (the card's
/// readback, surfaces ZERO) reads every vertex as wet and lifts the water one step over the land
/// at the shore, which is stated.
///
/// **Example.** A hull at 17 000 km draws the belt's coast at rung 16. A sea-floor vertex 40 m
/// under the water on a shelf 65 km wide used to stand 127 m over the sheet's quad and drew as a
/// tooth of land; now its water point is that vertex's own direction at the sea's radius, 40 m
/// over it, and the tooth is sea. A beach chunk at rung 4 keeps its sheet under the strand and
/// forty metres inland, where the dunes can still morph down to it; a chunk 400 m up a hillside
/// builds none.
#[must_use]
pub fn water_sheet(
    body: &BodyDefinition,
    samples: &SampleBox,
    mesh: &ChunkMesh,
    hide: Gi,
) -> WaterSheet {
    water_sheet_sided(body, samples, mesh, hide).0
}

/// One GROUND vertex's water word: the side its columns gave it and its water's radius in
/// metres — what a host needs to cut the sea floor that no ray can reach under the water.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundWord {
    pub side: WaterSide,
    pub level_m: f64,
}

/// Where one water vertex comes from: its ground vertex, the side its columns gave it, and its
/// water's level — what a host that MORPHS the ground needs to morph the water with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSource {
    /// The ground vertex's index in the chunk's mesh.
    pub ground: u32,
    /// The side the columns gave the vertex, as [`water_radius`] reads it.
    pub side: WaterSide,
    /// The water's radius at this vertex, in metres.
    pub level_m: f64,
}

/// A vertex's side, from its group's four columns: their surfaces against their own water.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaterSide {
    /// Every column's surface stands under its water.
    Under,
    /// Every column's surface stands over its water.
    Over,
    /// The columns disagree: the shore's own vertex.
    Mixed,
}

/// ★ THE DEEP WATER'S BLOCK, in cells a side (2026-09-23, the performance measurements): where
/// the sea floor under the water is not drawn (a client's floor cut), nothing can poke through
/// the sheet, so ruling W16's flat block is lawful again exactly there — one quad per block of
/// this many cells, on the corner columns at the water's radius, and the ground's own triangles
/// only where the floor is drawn (the shore, a shelf). The chord's dip under the level is real
/// and harmless: it stands over nothing.
pub const SHEET_BLOCK: i32 = 8;

/// ★ HOW WIDE A DEEP-WATER BLOCK MAY BE at a rung, in cells: the widest block up to
/// [`SHEET_BLOCK`] whose chord dips under the sea by no more than the extractor's own step, the
/// depth the rung can tell from zero — `(n · cell)² / 8R ≤ cell / 128`, so `n² ≤ R / (16 · cell)`.
/// ONE where no block passes, and the fine sheet stands alone. MEASURED before the bound (the
/// owner, from 17 900 km): eight-cell blocks at rung 16 dipped 5 km, and the atmosphere read them
/// as lighter squares on the far sea. On the home planet: six cells at rung 13, four at 14, two at
/// 15, two at 16, one from 17.
#[must_use]
pub fn sheet_block_cells(radius_m: f64, cell_m: f64) -> i32 {
    let mut n = 1i32;
    while n < SHEET_BLOCK && f64::from((n + 1) * (n + 1)) * 16.0 * cell_m <= radius_m {
        n += 1;
    }
    n
}

/// A column's unit direction in the body's frame, for a host outside the recipe (the deep
/// water's block corners stand on it).
#[must_use]
pub fn column_dir(samples: &SampleBox, a: i32, b: i32) -> [f64; 3] {
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let d = samples.dirs[SampleBox::column_index(a, b)];
    [
        d[0].raw() as f64 / unit,
        d[1].raw() as f64 / unit,
        d[2].raw() as f64 / unit,
    ]
}

/// ★ THE EXTRACTOR'S OWN STEP at a rung, in metres: a gap byte is 1/128 of a cell.
#[must_use]
pub fn water_quantum_m(rung: u8) -> f64 {
    f64::from(vd_seed::ladder::cell_m(rung)) / crate::chunk::GAP_STEPS_PER_CELL as f64
}

/// ★ THE WATER'S RADIUS OVER ONE GROUND VERTEX: the water's own level, held one extractor's step
/// off the drawn ground on the columns' side wherever the level would put it nearer — at least
/// `ground + q` over a vertex under the water, at most `ground − q` under a vertex over it, the
/// level at a mixed one. ONE rule for the built sheet and for the morphed sheet: a host that moves
/// the ground toward a coarser rung's mesh reads the water's target from the MOVED ground through
/// this same function, so the water rides with the land through every crossfade (ruling W18).
#[must_use]
pub fn water_radius(side: WaterSide, level_m: f64, ground_r: f64, quantum_m: f64) -> f64 {
    match side {
        WaterSide::Under => {
            if ground_r + quantum_m > level_m {
                ground_r + quantum_m
            } else {
                level_m
            }
        }
        WaterSide::Over => {
            if ground_r - quantum_m < level_m {
                ground_r - quantum_m
            } else {
                level_m
            }
        }
        WaterSide::Mixed => level_m,
    }
}

/// [`water_sheet`] with each water vertex's source beside it, in the sheet's vertex order, and
/// every GROUND vertex's own water word, in the mesh's vertex order.
#[must_use]
pub fn water_sheet_sided(
    body: &BodyDefinition,
    samples: &SampleBox,
    mesh: &ChunkMesh,
    hide: Gi,
) -> (WaterSheet, Vec<WaterSource>, Vec<GroundWord>) {
    let mut sheet: WaterSheet = (Vec::new(), Vec::new(), Vec::new());
    let mut sources: Vec<WaterSource> = Vec::new();
    let edge = CHUNK_EDGE as i32;
    let hide_m = metres_of_q28(hide);
    let quantum_m = water_quantum_m(samples.key.rung);
    // A column's level: its own water word, or the body's sea under every cell.
    let level_at = |col: usize| -> Gi {
        let w = samples.water[col];
        if w > samples.sea { w } else { samples.sea }
    };
    // Each ground vertex: its metres, its radius, its own level — the highest word among its
    // group's four columns (the group as `vertex_position` reads it: the quantum's shift, the last
    // centre's group clamped) — and its side, the columns' word.
    let mut points: Vec<[f64; 3]> = Vec::with_capacity(mesh.vertices.len());
    let mut radii_m: Vec<f64> = Vec::with_capacity(mesh.vertices.len());
    let mut levels: Vec<Gi> = Vec::with_capacity(mesh.vertices.len());
    let mut sides: Vec<WaterSide> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let p = vertex_position_m(body, samples, *v);
        radii_m.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
        points.push(p);
        let mut group = [0i32; 2];
        let mut axis = 0;
        while axis < 2 {
            let mut cell = i32::from(v[axis]) >> WEIGHT_BITS;
            if cell > edge - 1 {
                cell = edge - 1;
            }
            group[axis] = cell;
            axis += 1;
        }
        let mut level = Gi::ZERO;
        let (mut under, mut over) = (0u8, 0u8);
        for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let col = SampleBox::column_index(group[0] + da, group[1] + db);
            let l = level_at(col);
            if l > level {
                level = l;
            }
            let s = samples.surfaces[col];
            under += u8::from(s < l);
            over += u8::from(s > l);
        }
        levels.push(level);
        sides.push(if under == 4 {
            WaterSide::Under
        } else if over == 4 {
            WaterSide::Over
        } else {
            WaterSide::Mixed
        });
    }
    // One water vertex per ground vertex and level, shared by the triangles at that level.
    let mut index: BTreeMap<(u32, i64), u32> = BTreeMap::new();
    for t in &mesh.triangles {
        let mut level = Gi::ZERO;
        let mut lowest_m = f64::MAX;
        for i in t {
            let l = levels[*i as usize];
            if l > level {
                level = l;
            }
            let r = radii_m[*i as usize];
            if r < lowest_m {
                lowest_m = r;
            }
        }
        if level == Gi::ZERO {
            continue;
        }
        let level_m = metres_of_q28(level);
        if lowest_m > level_m + hide_m {
            continue;
        }
        let mut ids = [0u32; 3];
        for (k, i) in t.iter().enumerate() {
            let key = (*i, level.raw());
            ids[k] = match index.get(&key) {
                Some(id) => *id,
                None => {
                    let id = sheet.0.len() as u32;
                    let p = points[*i as usize];
                    let r = radii_m[*i as usize];
                    let side = sides[*i as usize];
                    let scale = water_radius(side, level_m, r, quantum_m) / r;
                    sheet.0.push([p[0] * scale, p[1] * scale, p[2] * scale]);
                    // The radial is the ground vertex's own unit direction.
                    sheet.1.push([p[0] / r, p[1] / r, p[2] / r]);
                    sources.push(WaterSource {
                        ground: *i,
                        side,
                        level_m,
                    });
                    index.insert(key, id);
                    id
                }
            };
        }
        sheet.2.push(ids);
    }
    let words = sides
        .iter()
        .zip(&levels)
        .map(|(side, level)| GroundWord {
            side: *side,
            level_m: metres_of_q28(*level),
        })
        .collect();
    (sheet, sources, words)
}

#[cfg(test)]
mod water_sheet_tests {
    use super::*;

    /// A hide bound past every ground: keeps every triangle, as the sheet did before the bound.
    const KEEP_ALL: Gi = Gi::new(i64::MAX >> 2);
    use crate::chunk::ChunkKey;
    use crate::extract::extract_all_edges;
    use vd_seed::bend::Face;

    fn radius(p: [f64; 3]) -> f64 {
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    }

    /// ★ THE SHEET ON THE GROUND'S TRIANGLES (ruling W18): a chunk whose ground stands wholly under
    /// a stated sea yields ONE water triangle per ground triangle, each point at the sea's radius
    /// (within a gap step) along ITS ground vertex's own direction, with that direction as its
    /// unit radial; a dry chunk yields nothing; a chunk where one column is wet yields the
    /// triangles that touch that column's vertices, and only those, under a zero hide.
    #[test]
    fn the_sheet_stands_flat_at_the_water_on_the_ground_s_own_triangles() {
        let moon = crate::home::home_moon();
        let key = ChunkKey {
            face: Face::PosY,
            rung: 4,
            x: 3,
            y: 3,
            z: 0,
        };
        let wet = moon.with_sea_m(Some(3_000));
        let bx = crate::lattice::sample_box(&wet, None, key).expect("a box");
        let mesh = extract_all_edges(&bx);
        let (points, radials, triangles) = water_sheet(&wet, &bx, &mesh, KEEP_ALL);
        assert_eq!(triangles.len(), mesh.triangles.len());
        assert_eq!(points.len(), mesh.vertices.len());
        assert_eq!(radials.len(), points.len());
        let sea_r = wet.sea_radius_m();
        for (t, g) in triangles.iter().zip(&mesh.triangles) {
            for (w, v) in t.iter().zip(g) {
                let p = points[*w as usize];
                let len = radius(p);
                assert!((len - sea_r).abs() < 0.01, "{len} vs {sea_r}");
                let ground = vertex_position_m(&wet, &bx, mesh.vertices[*v as usize]);
                let gl = radius(ground);
                // The same direction, to the last bit of the scaled integer position.
                let dot = (p[0] * ground[0] + p[1] * ground[1] + p[2] * ground[2]) / (len * gl);
                assert!(dot > 1.0 - 1e-12, "{dot}");
                let r = radials[*w as usize];
                assert!((radius(r) - 1.0).abs() < 1e-9);
                assert!((p[0] / len - r[0]).abs() < 1e-9);
            }
        }
        // The dry moon's own surface chunk (the wet one stands under its sea at `z: 0`).
        let key = ChunkKey {
            z: crate::digest::surface_chunk_z(&moon, key.face, key.rung, key.x, key.y),
            ..key
        };
        let dry = crate::lattice::sample_box(&moon, None, key).expect("a box");
        let dry_mesh = extract_all_edges(&dry);
        assert!(!dry_mesh.triangles.is_empty());
        assert_eq!(water_sheet(&moon, &dry, &dry_mesh, KEEP_ALL).0.len(), 0);
        // One wet column on a dry body: only the triangles that touch a vertex of that column's
        // groups carry a level, whatever the hide.
        let mut one = dry;
        one.water[SampleBox::column_index(0, 0)] = wet.sea_radius;
        let (points, _, triangles) = water_sheet(&moon, &one, &dry_mesh, KEEP_ALL);
        assert!(!triangles.is_empty());
        assert!(triangles.len() < dry_mesh.triangles.len());
        assert!(points.len() < dry_mesh.vertices.len());
    }

    /// ★ THE SEA UNDER EVERY CELL, THE LAKE ABOVE IT, AND THE CUT UNDER THE LAND: columns whose
    /// rows are dry still carry the sea's surface under the land; a lake word on one column raises
    /// the triangles that touch it to the lake and leaves the rest at the sea; with no hide only
    /// the triangles a wet vertex touches survive, with a hundred metres those within a hundred
    /// metres over the water, and a sea 20 km under every column keeps nothing with no bound and
    /// every triangle with the largest one. MEASURED here on the box: the counts fall as the
    /// bound tightens and never rise.
    #[test]
    fn the_sea_stands_under_every_cell_and_the_sheet_is_cut_under_the_land() {
        let moon = crate::home::home_moon();
        let key = ChunkKey {
            face: Face::PosY,
            rung: 4,
            x: 3,
            y: 3,
            z: 0,
        };
        // A sea low in this chunk's own relief: a sixteenth of its ground vertices under it, so
        // the hide bound has ground to cut and ground to keep.
        let key = ChunkKey {
            z: crate::digest::surface_chunk_z(&moon, key.face, key.rung, key.x, key.y),
            ..key
        };
        let dry = crate::lattice::sample_box(&moon, None, key).expect("a box");
        let dry_mesh = extract_all_edges(&dry);
        assert!(!dry_mesh.triangles.is_empty());
        let mut radii: Vec<f64> = dry_mesh
            .vertices
            .iter()
            .map(|v| radius(vertex_position_m(&moon, &dry, *v)))
            .collect();
        radii.sort_by(f64::total_cmp);
        let low = radii[radii.len() >> 4];
        let wet = moon.with_sea_m(Some((low - moon.radius_m()) as i32));
        let mut land = crate::lattice::sample_box(&wet, None, key).expect("a box");
        for w in &mut land.water {
            *w = Gi::ZERO;
        }
        let mesh = extract_all_edges(&land);
        let (_, _, triangles) = water_sheet(&wet, &land, &mesh, KEEP_ALL);
        assert_eq!(triangles.len(), mesh.triangles.len());
        let hundred = Gi::new(100 * STEPS_PER_M) << vd_recipe::cell::LENGTH_BITS;
        let at_hundred = water_sheet(&wet, &land, &mesh, hundred).2.len();
        let at_zero = water_sheet(&wet, &land, &mesh, Gi::ZERO).2.len();
        assert!(at_hundred <= mesh.triangles.len(), "{at_hundred}");
        assert!(at_zero < at_hundred, "{at_zero} {at_hundred}");
        assert!(at_zero > 0, "{at_zero}");
        assert!(at_zero < mesh.triangles.len(), "{at_zero}");
        let deep = moon.with_sea_m(Some(-20_000));
        let mut under = crate::lattice::sample_box(&deep, None, key).expect("a box");
        for w in &mut under.water {
            *w = Gi::ZERO;
        }
        let under_mesh = extract_all_edges(&under);
        assert_eq!(water_sheet(&deep, &under, &under_mesh, Gi::ZERO).2.len(), 0);
        assert_eq!(
            water_sheet(&deep, &under, &under_mesh, KEEP_ALL).2.len(),
            under_mesh.triangles.len()
        );
        let lake = wet.sea_radius + (Gi::new(50 * STEPS_PER_M) << vd_recipe::cell::LENGTH_BITS);
        land.water[SampleBox::column_index(0, 0)] = lake;
        let (points, _, triangles) = water_sheet(&wet, &land, &mesh, KEEP_ALL);
        assert_eq!(triangles.len(), mesh.triangles.len());
        // Two levels on the sheet: the lake's few points and the sea's many, and a vertex shared by
        // a lake triangle and a sea triangle stands twice.
        let sea_r = wet.sea_radius_m();
        let (mut at_lake, mut at_sea) = (0usize, 0usize);
        for p in &points {
            let len = radius(*p);
            if (len - (sea_r + 50.0)).abs() < 0.01 {
                at_lake += 1;
            } else {
                assert!((len - sea_r).abs() < 0.01, "{len}");
                at_sea += 1;
            }
        }
        assert!(at_lake > 0 && at_lake < at_sea, "{at_lake} {at_sea}");
        assert!(points.len() > mesh.vertices.len(), "{}", points.len());
    }

    /// ★ THE GROUND NEVER STANDS OVER THE WATER IT IS UNDER (ruling W18, the statement that could
    /// fail): on a real box at a COARSE rung — where the chord of a cell dips metres and the quad
    /// per cell drew the sea floor as teeth — every water triangle whose three ground vertices all
    /// stand under the water has its MIDDLE over the ground's middle, and every water point stands
    /// over its own ground vertex. The same box, read by the retired rule (a quad on the corner
    /// columns at the water's radius), puts ground vertices OVER the water: the fault this law cures,
    /// measured in the same test so the cure is a difference and not an argument.
    #[test]
    fn the_ground_never_stands_over_the_water_it_is_under() {
        let moon = crate::home::home_moon();
        let key = ChunkKey {
            face: Face::PosY,
            rung: 13,
            x: 0,
            y: 0,
            z: 0,
        };
        let cell_m = f64::from(vd_seed::ladder::cell_m(key.rung));
        let dip = cell_m * cell_m / (4.0 * moon.radius_m());
        assert!(dip > 1.0, "a cell's chord dips {dip} m at this rung");
        // A sea through the moon's own relief: some ground under it, some over.
        let wet = moon.with_sea_m(Some(200));
        let bx = crate::lattice::sample_box(&wet, None, key).expect("a box");
        let mesh = extract_all_edges(&bx);
        let ground: Vec<[f64; 3]> = mesh
            .vertices
            .iter()
            .map(|v| vertex_position_m(&wet, &bx, *v))
            .collect();
        let sea_r = wet.sea_radius_m();
        let wet_count = ground.iter().filter(|g| radius(**g) < sea_r).count();
        assert!(
            wet_count > 0 && wet_count < ground.len(),
            "{wet_count} of {}",
            ground.len()
        );
        // Every triangle built (the largest hide), so the water and the ground align one to one.
        let (points, _, triangles) = water_sheet(&wet, &bx, &mesh, KEEP_ALL);
        assert_eq!(triangles.len(), mesh.triangles.len());
        // A vertex's side as the builder reads it: its group's four columns against their water.
        let side = |v: &[i16; 3]| -> (bool, bool) {
            let group =
                [0, 1].map(|axis| (i32::from(v[axis]) >> WEIGHT_BITS).min(CHUNK_EDGE as i32 - 1));
            let (mut under, mut over) = (0, 0);
            for (da, db) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let col = SampleBox::column_index(group[0] + da, group[1] + db);
                let l = bx.water[col].max(bx.sea);
                under += u8::from(bx.surfaces[col] < l);
                over += u8::from(bx.surfaces[col] > l);
            }
            (under == 4, over == 4)
        };
        let q =
            f64::from(vd_seed::ladder::cell_m(key.rung)) / crate::chunk::GAP_STEPS_PER_CELL as f64;
        let mut all_wet = 0usize;
        let (mut wet_v, mut dry_v, mut mixed_v) = (0usize, 0usize, 0usize);
        for (t, g) in triangles.iter().zip(&mesh.triangles) {
            for (w, v) in t.iter().zip(g) {
                let (pw, pg) = (radius(points[*w as usize]), radius(ground[*v as usize]));
                match side(&mesh.vertices[*v as usize]) {
                    (true, _) => {
                        wet_v += 1;
                        assert!(pw >= pg + q - 1e-6, "{pw} under {pg} + {q}");
                    }
                    (_, true) => {
                        dry_v += 1;
                        assert!(pw <= pg - q + 1e-6, "{pw} over {pg} - {q}");
                    }
                    _ => {
                        mixed_v += 1;
                        assert!((pw - sea_r).abs() < 0.01, "{pw} vs {sea_r}");
                    }
                }
            }
            if g.iter().all(|i| side(&mesh.vertices[*i as usize]).0) {
                all_wet += 1;
                let mid =
                    |ps: [[f64; 3]; 3]| [0, 1, 2].map(|k| (ps[0][k] + ps[1][k] + ps[2][k]) / 3.0);
                let mw = mid([
                    points[t[0] as usize],
                    points[t[1] as usize],
                    points[t[2] as usize],
                ]);
                let mg = mid([
                    ground[g[0] as usize],
                    ground[g[1] as usize],
                    ground[g[2] as usize],
                ]);
                assert!(
                    radius(mw) >= radius(mg),
                    "{} under {}",
                    radius(mw),
                    radius(mg)
                );
            }
        }
        assert!(all_wet > 0);
        assert!(wet_v > 0 && mixed_v > 0, "{wet_v} {dry_v} {mixed_v}");
        // The retired rule on the same box: a quad on the four corner columns at the sea's radius;
        // a wet ground vertex inside it against the quad's bilinear point at the vertex's fraction.
        let edge = CHUNK_EDGE as i32;
        let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
        let mut poked = 0usize;
        for (v, g) in mesh.vertices.iter().zip(&ground) {
            if radius(*g) >= sea_r {
                continue;
            }
            let mut group = [0i32; 2];
            let mut frac = [0f64; 2];
            for axis in 0..2 {
                let q = i32::from(v[axis]);
                let cell = (q >> WEIGHT_BITS).min(edge - 1);
                group[axis] = cell;
                frac[axis] = f64::from(q - (cell << WEIGHT_BITS)) / f64::from(1 << WEIGHT_BITS);
            }
            let (u, w) = (frac[0], frac[1]);
            let weights = [(1.0 - u) * (1.0 - w), u * (1.0 - w), (1.0 - u) * w, u * w];
            let mut quad = [0.0f64; 3];
            for (k, (da, db)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                let d = bx.dirs[SampleBox::column_index(group[0] + da, group[1] + db)];
                for axis in 0..3 {
                    quad[axis] += weights[k] * d[axis].raw() as f64 / unit * sea_r;
                }
            }
            poked += usize::from(radius(*g) > radius(quad));
        }
        assert!(
            poked > 0,
            "the retired rule put no wet vertex over its quad on this box"
        );
    }

    /// The group of a vertex ON the box's last centre is the last group (the clamp
    /// `vertex_position` makes), and a triangle is not built twice for one level: a hand-made
    /// mesh of two triangles on the box's far corner, over a wet sea, shares its two common
    /// vertices and reads the halo's columns for the clamped group.
    #[test]
    fn a_deep_water_block_dips_no_more_than_the_extractor_s_step() {
        let planet = crate::home::home_planet();
        let r = planet.radius_m();
        let cell = |rung: u8| f64::from(vd_seed::ladder::cell_m(rung));
        // The home planet: six cells at rung 13, four at 14, three at 15, two at 16, one from 17.
        assert_eq!(sheet_block_cells(r, cell(13)), 6);
        assert_eq!(sheet_block_cells(r, cell(14)), 4);
        assert_eq!(sheet_block_cells(r, cell(15)), 3);
        assert_eq!(sheet_block_cells(r, cell(16)), 2);
        assert_eq!(sheet_block_cells(r, cell(17)), 1);
        assert_eq!(sheet_block_cells(r, cell(18)), 1);
        // A fine rung reaches the cap, and every width passed holds the bound itself.
        assert_eq!(sheet_block_cells(r, cell(4)), SHEET_BLOCK);
        // Every block of two cells or more holds the bound itself; a width of one is no block, and
        // the fine sheet's own cell dips what it dips (5.4 km at rung 19, past the step there).
        for rung in 0..20u8 {
            let cells = sheet_block_cells(r, cell(rung));
            if cells < 2 {
                continue;
            }
            let w = f64::from(cells) * cell(rung);
            let dip = w * w / (8.0 * r);
            assert!(dip <= cell(rung) / 128.0 + 1e-9, "rung {rung}: {dip}");
        }
    }

    #[test]
    fn a_vertex_on_the_last_centre_reads_the_last_group_and_shares_its_water_vertex() {
        let moon = crate::home::home_moon();
        let key = ChunkKey {
            face: Face::PosY,
            rung: 4,
            x: 3,
            y: 3,
            z: 0,
        };
        let wet = moon.with_sea_m(Some(3_000));
        let bx = crate::lattice::sample_box(&wet, None, key).expect("a box");
        let last = (CHUNK_EDGE as i16) << WEIGHT_BITS;
        let mesh = ChunkMesh {
            key,
            vertices: vec![
                [last, last, 0],
                [last - 256, last, 0],
                [last, last - 256, 0],
                [0, 0, 0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let (points, _, triangles) = water_sheet(&wet, &bx, &mesh, KEEP_ALL);
        assert_eq!(triangles.len(), 2);
        assert_eq!(points.len(), 4, "two vertices shared, none built twice");
        assert_eq!(triangles[0], [0, 1, 2]);
        assert_eq!(triangles[1], [0, 2, 3]);
        let sea_r = wet.sea_radius_m();
        for p in &points {
            assert!((radius(*p) - sea_r).abs() < 0.01);
        }
        // An empty mesh over the same water builds nothing, with no branch on the count.
        let none = ChunkMesh {
            key,
            vertices: Vec::new(),
            triangles: Vec::new(),
        };
        assert_eq!(water_sheet(&wet, &bx, &none, KEEP_ALL).2.len(), 0);
        // ★ ONE EXTRACTOR'S STEP OFF THE DRAWN GROUND, ON THE COLUMNS' SIDE: a sea set at the
        // corner vertex's own radius (to the metre) over columns whose surfaces stand OVER it
        // (this hand-made mesh stands deep in the ladder's first band, 338 km from the centre,
        // under the moon's real ground) puts every water point one gap step UNDER its ground;
        // the same columns' surfaces set under the water put every water point one gap step
        // OVER it; a column set to disagree leaves its vertex's water at the water.
        let corner = vertex_position_m(&wet, &bx, mesh.vertices[3]);
        let corner_r = radius(corner);
        let tied = moon.with_sea_m(Some((corner_r - moon.radius_m()) as i32));
        let mut bx_tied = crate::lattice::sample_box(&tied, None, key).expect("a box");
        let q =
            f64::from(vd_seed::ladder::cell_m(key.rung)) / crate::chunk::GAP_STEPS_PER_CELL as f64;
        let level_r = tied.sea_radius_m();
        let ground =
            |bx: &SampleBox, i: usize| radius(vertex_position_m(&tied, bx, mesh.vertices[i]));
        assert!(
            bx_tied.surfaces.iter().all(|s| *s > bx_tied.sea),
            "the real ground stands over"
        );
        let (points, _, _) = water_sheet(&tied, &bx_tied, &mesh, KEEP_ALL);
        for (k, w) in points.iter().enumerate() {
            let g = ground(&bx_tied, k);
            let expected = if g - q < level_r { g - q } else { level_r };
            assert!(
                (radius(*w) - expected).abs() < 1e-6,
                "{} vs {expected}",
                radius(*w)
            );
        }
        for s in &mut bx_tied.surfaces {
            *s = Gi::ZERO;
        }
        let (points, _, _) = water_sheet(&tied, &bx_tied, &mesh, KEEP_ALL);
        for (k, w) in points.iter().enumerate() {
            let g = ground(&bx_tied, k);
            let expected = if g + q > level_r { g + q } else { level_r };
            assert!(
                (radius(*w) - expected).abs() < 1e-6,
                "{} vs {expected}",
                radius(*w)
            );
        }
        // The far corner's group is the last group, columns 61..62 each way: one of them over.
        let last_col = SampleBox::column_index(CHUNK_EDGE as i32, CHUNK_EDGE as i32);
        bx_tied.surfaces[last_col] = bx_tied.sea + Gi::new(1 << vd_recipe::cell::LENGTH_BITS);
        let (points, _, _) = water_sheet(&tied, &bx_tied, &mesh, KEEP_ALL);
        assert!(
            (radius(points[0]) - level_r).abs() < 1e-6,
            "a mixed vertex reads the water"
        );
        assert!(
            (radius(points[3]) - {
                let g = ground(&bx_tied, 3) + q;
                if g > level_r { g } else { level_r }
            })
            .abs()
                < 1e-6
        );
    }
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
    use crate::lattice::sample_box;
    use vd_seed::bend::Face;

    #[test]
    fn a_vertex_on_a_cell_centre_is_that_centre_and_halfway_is_the_mean() {
        let m = home_planet();
        let rung = 4;
        let z = surface_chunk_z(&m, Face::PosX, rung, 9, 7);
        let kx = crate::chunk::ChunkKey {
            face: Face::PosX,
            rung,
            x: 9,
            y: 7,
            z,
        };
        let bx = sample_box(&m, None, kx).expect("in the band");
        // The centre of local cell (3, 4, 5): weight one on that cell.
        let p = vertex_position(&m, &bx, [3 * 256, 4 * 256, 5 * 256]);
        let dir = bx.dir(3, 4);
        let r = Gi::new(m.ladder.cell_radius_steps(z * CHUNK_EDGE as i32 + 5, rung));
        let want = point_at(dir, r);
        let mut k = 0;
        while k < 3 {
            assert_eq!(p[k] >> POSITION_BITS, want[k], "[{k}]");
            k += 1;
        }
        // Halfway between two centres along a: the mean of the two, exactly (the sum carries the
        // weights' own bits, so a half is a half).
        let mid = vertex_position(&m, &bx, [3 * 256 + 128, 4 * 256, 5 * 256]);
        let d2 = point_at(bx.dir(4, 4), r);
        let mut k = 0;
        while k < 3 {
            let mean = (want[k] + d2[k]) >> 1;
            let got = mid[k] >> POSITION_BITS;
            assert!((got - mean).raw().abs() <= 1, "[{k}]: {got:?} vs {mean:?}");
            k += 1;
        }
        // The render seam reads the same position in metres.
        let pm = vertex_position_m(&m, &bx, [3 * 256, 4 * 256, 5 * 256]);
        assert_eq!(pm[0], crate::units::metres_of_steps(want[0].raw()));
        // The clamp arm: a vertex on the box's last centre.
        let _ = vertex_position_m(&m, &bx, [62 * 256, 62 * 256, 62 * 256]);
    }

    /// A corner prism vertex at a column's centre maps to that column's centre, and the three
    /// faces' boxes map the prism's centroid to the same metres.
    #[test]
    fn a_corner_vertex_is_barycentric_over_the_three_real_columns() {
        let m = home_planet();
        let rung = 6;
        let n_l = m.ladder().cells_per_edge(rung) as i32;
        let last = (n_l - 1) / CHUNK_EDGE as i32;
        let beyond = n_l - last * CHUNK_EDGE as i32;
        let z = surface_chunk_z(&m, Face::PosX, rung, last, last);
        let key = |face: Face| crate::chunk::ChunkKey {
            face,
            rung,
            x: last,
            y: last,
            z,
        };
        let boxes = [Face::PosX, Face::PosY, Face::PosZ]
            .map(|f| sample_box(&m, None, key(f)).expect("in the band"));
        // In every box the phantom is at (beyond, beyond) and the own corner cell at (beyond − 1,
        // beyond − 1); the prism group's origin is (beyond − 1, beyond − 1, c).
        let base = ((beyond - 1) * 256) as i16;
        // The vertex at the own corner column's centre, layer c = 5: exactly that centre.
        let p = vertex_position(&m, &boxes[0], [base, base, 5 * 256]);
        let r = Gi::new(m.ladder.cell_radius_steps(z * CHUNK_EDGE as i32 + 5, rung));
        let want = point_at(boxes[0].dir(beyond - 1, beyond - 1), r);
        let mut k = 0;
        while k < 3 {
            assert_eq!(p[k] >> POSITION_BITS, want[k], "[{k}]");
            k += 1;
        }
        // One point of the prism, in the three local layouts: weights (wX, wY, wZ) = (86, 85, 85)
        // quanta. In +X's box (r[0] = X, r[1] = Y along a, r[2] = Z along b) that is local
        // (85, 85); in +Y's (r[0] = Y, r[1] = Z along +Y's a, r[2] = X) it is (85, 86); in +Z's
        // (r[0] = Z, r[1] = X, r[2] = Y) it is (86, 85). One point in metres from all three.
        let px = vertex_position(&m, &boxes[0], [base + 85, base + 85, 5 * 256 + 128]);
        let py = vertex_position(&m, &boxes[1], [base + 85, base + 86, 5 * 256 + 128]);
        let pz = vertex_position(&m, &boxes[2], [base + 86, base + 85, 5 * 256 + 128]);
        assert_eq!(px, py);
        assert_eq!(px, pz);
        // Weight one on the neighbour along a: that neighbour's centre.
        let p1 = vertex_position(&m, &boxes[0], [base + 256, base, 5 * 256]);
        let d1 = point_at(boxes[0].dir(beyond, beyond - 1), r);
        let mut k = 0;
        while k < 3 {
            assert_eq!(p1[k] >> POSITION_BITS, d1[k], "[{k}]");
            k += 1;
        }
        // The (−u, −v) corner of a chunk at (0, 0): the phantom sits at local (0, 0) of its group
        // (origin (−1, −1)), so the fractions run the other way. Weight one on the own corner cell
        // (0, 0) is that cell's centre; a quarter toward each neighbour blends three real centres.
        let k0 = crate::chunk::ChunkKey {
            face: Face::NegY,
            rung,
            x: 0,
            y: 0,
            z,
        };
        let b0 = sample_box(&m, None, k0).expect("in the band");
        let p0 = vertex_position(&m, &b0, [0, 0, 5 * 256]);
        let d0 = point_at(b0.dir(0, 0), r);
        let mut k = 0;
        while k < 3 {
            assert_eq!(p0[k] >> POSITION_BITS, d0[k], "[{k}]");
            k += 1;
        }
        let pq = vertex_position(&m, &b0, [-64, -64, 5 * 256]);
        let dq1 = point_at(b0.dir(-1, 0), r);
        let dq2 = point_at(b0.dir(0, -1), r);
        let mut k = 0;
        while k < 3 {
            // A half on the own cell and a quarter on each neighbour.
            let expect = (d0[k] >> 1) + (dq1[k] >> 2) + (dq2[k] >> 2);
            let got = pq[k] >> POSITION_BITS;
            assert!(
                (got - expect).raw().abs() <= 2,
                "[{k}]: {got:?} vs {expect:?}"
            );
            k += 1;
        }
    }
}
