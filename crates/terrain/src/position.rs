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
use crate::extract::{VERTEX_QUANTUM, phantom_corner};
use crate::lattice::{SampleBox, Site};
use crate::units::{STEPS_PER_M, metres_of_fixed};
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

/// ★ THE WATER SHEET's points (slice 8c stage C5): over the chunk's 62 × 62 core columns and the
/// halo column past each edge, one quad per cell of the face grid whose four corners hold ANY
/// water; the quad stands flat at the HIGHEST water level among its wet corners (a shore quad
/// reaches under the land, which hides it), each corner at that radius along its own column.
/// ★ THE SEA UNDER EVERY CELL (2026-09-20): on a body with a sea every cell holds the sea's level
/// at least, so the sea is one surface at every rung and the shore is where the land crosses it —
/// continuous under the land's morph, never a sheet edge that steps at a ring swap. A lake stands
/// above the sea by its own corners' word. Four
/// points a quad (a level is a quad's, not a column's), in METRES of the body's frame, with each
/// point's unit radial, and two triangles a quad. Empty for a dry chunk. The client's mesh is the
/// last step out of it (the floating origin, the single-precision cast).
/// ★ THE SHEET UNDER THE LAND IS CUT WHERE IT CAN NEVER SHOW (2026-09-21, the owner: *"when we
/// draw just the planet it is performant, but if water is drawn it becomes slower"* — MEASURED on
/// the coast leg, two flights of one binary: the frame 33.8 ms with the sheet against 24.2 ms
/// without, the opaque pass 22.5 ms against 16.3 ms, because the sea under EVERY cell doubled the
/// triangles of every land chunk). A quad stays only where a corner's ground stands no higher
/// than the water plus `hide`: the land's own morph toward the coarser rung, its sink under the
/// finer one and a cell of placement, which is as far as the drawn land can ever move off its
/// column's surface. Under that bound the sheet still reaches under the shore, so the shore stays
/// the land's own crossing of one surface (the sea under every cell); past it the sheet is buried
/// in every state the ladder can draw, and it is not built. A zero `hide` keeps the quads whose
/// corners are wet; the largest word keeps every quad.
///
/// **Example.** A beach chunk at rung 4 keeps its sheet under the strand and forty metres inland,
/// where the dunes can still morph down to it; a chunk 400 m up a hillside builds none.
/// ★ THE SHEET IS ONE QUAD PER BLOCK WHERE THE WATER IS ONE LEVEL (2026-09-21, the second half of
/// the cost): a flat sea needs no quad per cell. MEASURED on the coast eye's wanted set after the
/// hide cut: 33 million sheet triangles against 40 million of land — an ocean chunk at rung 7 drew
/// 7 688 triangles for a flat surface. Over each block of [`SHEET_BLOCK`] × [`SHEET_BLOCK`] cells
/// whose corner columns all hold ONE level (the sea, or one lake) the sheet is ONE quad at that
/// level, its four points the block's corners; a block whose corners disagree (a lake's edge, a
/// pond) keeps a quad per cell, so nothing a cell decides is lost. The flat quad stands under the
/// sphere by its sagitta — 0.3 m at rung 9 for an eight-cell block, 5 m at rung 11 where a cell is
/// two kilometres — a shore step under a twentieth of a cell at the rungs it reaches.
pub const SHEET_BLOCK: i32 = 8;

#[must_use]
pub fn water_sheet(samples: &SampleBox, hide: Gi) -> WaterSheet {
    let mut sheet: WaterSheet = (Vec::new(), Vec::new(), Vec::new());
    let edge = CHUNK_EDGE as i32;
    // A corner's level: its own water word, or the body's sea under every cell.
    let level_at = |col: usize| {
        if samples.water[col] > samples.sea {
            samples.water[col]
        } else {
            samples.sea
        }
    };
    let mut b0 = 0i32;
    while b0 < edge {
        let bh = SHEET_BLOCK.min(edge - b0);
        let mut a0 = 0i32;
        while a0 < edge {
            let bw = SHEET_BLOCK.min(edge - a0);
            // The block's corner columns: one level everywhere, and the lowest ground.
            let first = level_at(SampleBox::column_index(a0, b0));
            let mut uniform = true;
            let mut lowest = samples.surfaces[SampleBox::column_index(a0, b0)];
            let mut b = b0;
            while b <= b0 + bh {
                let mut a = a0;
                while a <= a0 + bw {
                    let col = SampleBox::column_index(a, b);
                    uniform &= level_at(col) == first;
                    if samples.surfaces[col] < lowest {
                        lowest = samples.surfaces[col];
                    }
                    a += 1;
                }
                b += 1;
            }
            if uniform {
                if (first > Gi::ZERO) & (lowest <= first + hide) {
                    push_quad(
                        samples,
                        &mut sheet,
                        [(a0, b0), (a0 + bw, b0), (a0, b0 + bh), (a0 + bw, b0 + bh)],
                        first,
                    );
                }
            } else {
                let mut b = b0;
                while b < b0 + bh {
                    let mut a = a0;
                    while a < a0 + bw {
                        let corners = [(a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1)];
                        let mut level = samples.sea;
                        let mut low = samples.surfaces[SampleBox::column_index(a, b)];
                        for (ca, cb) in corners {
                            let col = SampleBox::column_index(ca, cb);
                            if samples.water[col] > level {
                                level = samples.water[col];
                            }
                            if samples.surfaces[col] < low {
                                low = samples.surfaces[col];
                            }
                        }
                        if (level > Gi::ZERO) & (low <= level + hide) {
                            push_quad(samples, &mut sheet, corners, level);
                        }
                        a += 1;
                    }
                    b += 1;
                }
            }
            a0 += bw;
        }
        b0 += bh;
    }
    sheet
}

/// One quad of the sheet: four points at `level` along the named corner columns, their unit
/// radials, and two triangles.
fn push_quad(samples: &SampleBox, sheet: &mut WaterSheet, corners: [(i32, i32); 4], level: Gi) {
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let steps = STEPS_PER_M as f64;
    let base = sheet.0.len() as u32;
    for (a, b) in corners {
        let dir = samples.dirs[SampleBox::column_index(a, b)];
        let p = point_at(dir, level >> vd_recipe::cell::LENGTH_BITS);
        sheet.0.push([
            p[0].raw() as f64 / steps,
            p[1].raw() as f64 / steps,
            p[2].raw() as f64 / steps,
        ]);
        sheet.1.push([
            dir[0].raw() as f64 / unit,
            dir[1].raw() as f64 / unit,
            dir[2].raw() as f64 / unit,
        ]);
    }
    sheet.2.push([base, base + 1, base + 3]);
    sheet.2.push([base, base + 3, base + 2]);
}

#[cfg(test)]
mod water_sheet_tests {
    use super::*;
    /// A hide bound past every ground: keeps every quad, as the sheet did before the bound.
    const KEEP_ALL: Gi = Gi::new(i64::MAX >> 2);
    use crate::chunk::ChunkKey;
    use vd_seed::bend::Face;

    /// ★ THE SHEET (C5): a chunk under a stated sea yields a quad per cell, four points a quad at
    /// the sea's radius along each corner's column with the column's unit radial; a dry chunk
    /// yields nothing; a chunk where one corner column is wet yields that quad alone.
    #[test]
    fn the_sheet_stands_flat_at_the_water_and_only_where_water_is() {
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
        let (points, radials, triangles) = water_sheet(&bx, KEEP_ALL);
        // ★ ONE QUAD PER BLOCK over one flat sea: eight blocks an edge (seven of eight cells and
        // one of six), never a quad per cell.
        let blocks = (CHUNK_EDGE.div_ceil(SHEET_BLOCK as usize)).pow(2);
        assert_eq!(blocks, 64);
        assert_eq!(points.len(), blocks * 4);
        assert_eq!(radials.len(), blocks * 4);
        assert_eq!(triangles.len(), blocks * 2);
        let radius = wet.sea_radius_m();
        for (p, r) in points.iter().zip(&radials) {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((len - radius).abs() < 0.02, "{len} vs {radius}");
            let rl = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            assert!((rl - 1.0).abs() < 1e-9);
            assert!((p[0] / len - r[0]).abs() < 1e-6);
        }
        assert_eq!(triangles[0], [0, 1, 3]);
        assert_eq!(triangles[1], [0, 3, 2]);
        let dry = crate::lattice::sample_box(&moon, None, key).expect("a box");
        assert_eq!(water_sheet(&dry, KEEP_ALL).0.len(), 0);
        let mut one = dry;
        one.water[SampleBox::column_index(0, 0)] = wet.sea_radius;
        let (points, _, triangles) = water_sheet(&one, KEEP_ALL);
        assert_eq!(points.len(), 4);
        assert_eq!(triangles.len(), 2);
        // ★ THE SEA UNDER EVERY CELL: on a body with a sea, columns whose rows are dry (the land)
        // still carry the sea's surface, under the land; a lake corner above the sea raises its
        // quad to the lake.
        let mut land = crate::lattice::sample_box(&wet, None, key).expect("a box");
        for w in &mut land.water {
            *w = Gi::ZERO;
        }
        let (points, _, triangles) = water_sheet(&land, KEEP_ALL);
        assert_eq!(points.len(), blocks * 4);
        assert_eq!(triangles.len(), blocks * 2);
        // ★ THE SHEET IS CUT UNDER THE LAND: with no hide bound only the quads a corner's ground
        // reaches down to the water survive; with a bound of a hundred metres those within a
        // hundred metres over it; the land 3 km up the moon keeps none at all. MEASURED here on
        // the box: the counts fall as the bound tightens and never rise.
        let hundred = Gi::new(100 * STEPS_PER_M) << vd_recipe::cell::LENGTH_BITS;
        let at_hundred = water_sheet(&land, hundred).2.len();
        let at_zero = water_sheet(&land, Gi::ZERO).2.len();
        assert!(at_hundred <= blocks * 2, "{at_hundred}");
        assert!(at_zero <= at_hundred, "{at_zero} {at_hundred}");
        // Some of this box's ground stands under the 3 km sea, so a corner reaches the water and a
        // quad survives with no bound; a body whose sea stands 20 km under every column keeps
        // nothing with no bound and every block with the largest one.
        let sea = wet.sea_radius;
        let lowest = land
            .surfaces
            .iter()
            .copied()
            .fold(KEEP_ALL, |m, s| if s < m { s } else { m });
        assert!(lowest <= sea, "a corner of this box reaches the sea");
        assert!(at_zero > 0, "{at_zero}");
        let deep = moon.with_sea_m(Some(-20_000));
        let mut under = crate::lattice::sample_box(&deep, None, key).expect("a box");
        for w in &mut under.water {
            *w = Gi::ZERO;
        }
        assert_eq!(water_sheet(&under, Gi::ZERO).2.len(), 0);
        assert_eq!(water_sheet(&under, KEEP_ALL).2.len(), blocks * 2);
        let lake = wet.sea_radius + (Gi::new(50 * STEPS_PER_M) << vd_recipe::cell::LENGTH_BITS);
        land.water[SampleBox::column_index(0, 0)] = lake;
        let (points, _, triangles) = water_sheet(&land, KEEP_ALL);
        // The lake's block keeps a quad per cell (64), the other 63 blocks one each.
        assert_eq!(triangles.len(), 2 * (64 + 63));
        let len = (points[0][0] * points[0][0]
            + points[0][1] * points[0][1]
            + points[0][2] * points[0][2])
            .sqrt();
        assert!((len - (wet.sea_radius_m() + 50.0)).abs() < 0.02, "{len}");
        let last = points[points.len() - 1];
        let len = (last[0] * last[0] + last[1] * last[1] + last[2] * last[2]).sqrt();
        assert!((len - wet.sea_radius_m()).abs() < 0.02, "{len}");
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
