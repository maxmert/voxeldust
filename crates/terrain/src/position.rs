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
        let bx = sample_box(&m, kx).expect("in the band");
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
            .map(|f| sample_box(&m, key(f)).expect("in the band"));
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
        let b0 = sample_box(&m, k0).expect("in the band");
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
