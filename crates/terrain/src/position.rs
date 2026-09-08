//! ★ WHERE A VERTEX IS, IN METRES — the one mapping from a mesh vertex in cell space to the body's
//! own frame, shared by the collider (slice 11) and the client (slice 7).
//!
//! A vertex lies inside one group of eight cells. Its position is the trilinear blend of those
//! eight CELL CENTRES (each centre is its column's own direction times its layer's radius), with
//! weights that are exact multiples of `1/256`. The eight terms are summed in ONE canonical order —
//! by the cell's global site, never by the chunk's local axes — so the chunk on either side of a
//! seam, whose local axes differ, computes the same bytes for a shared vertex.
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
use crate::chunk::CHUNK_EDGE;
use crate::extract::{VERTEX_QUANTUM, phantom_corner};
use crate::gf::Gf;
use crate::lattice::{SampleBox, Site};

/// The position of a mesh vertex in the body's frame, in metres.
#[must_use]
pub fn vertex_position_m(body: &BodyDefinition, samples: &SampleBox, v: [i16; 3]) -> [Gf; 3] {
    let key = samples.key;
    let edge = CHUNK_EDGE as i32;
    let k0 = key.z * edge;
    // The group that holds the vertex, and the vertex's fraction inside it (exact 1/256 steps).
    let mut base = [0i32; 3];
    let mut frac = [Gf::ZERO; 3];
    let mut k = 0;
    while k < 3 {
        let q = i32::from(v[k]);
        let mut cell = q.div_euclid(VERTEX_QUANTUM);
        // A vertex exactly on the box's last centre belongs to the last group. No vertex lies below
        // the first centre: every group's origin is at least the halo's `−1`, and a vertex never
        // leaves its group.
        if cell > edge - 1 {
            cell = edge - 1;
        }
        base[k] = cell;
        frac[k] = Gf::from_i32(q - cell * VERTEX_QUANTUM) / Gf::from_i32(VERTEX_QUANTUM);
        k += 1;
    }
    let centre = |a: i32, b: i32, c: i32| -> ((Site, i32), [Gf; 3]) {
        let dir = samples.dir(a, b);
        let r = Gf::from_f64(body.ladder.cell_radius_m(k0 + c, key.rung));
        (
            (samples.site(a, b), k0 + c),
            [dir[0] * r, dir[1] * r, dir[2] * r],
        )
    };
    // The cells with their global sites, centres and weights: eight for a cube group, six for a
    // corner prism.
    let mut terms: Vec<((Site, i32), [Gf; 3], Gf)> = Vec::with_capacity(8);
    match phantom_corner(samples, base) {
        Some((pa, pb)) => {
            // Barycentric over r[0] (opposite the phantom), r[1] (beside it along a), r[2] (beside
            // it along b): the fraction toward r[1] along a is `fa` when r[1] sits at a = 1.
            let u = if pa == 1 { frac[0] } else { Gf::ONE - frac[0] };
            let w = if pb == 1 { frac[1] } else { Gf::ONE - frac[1] };
            let pos = [[1 - pa, 1 - pb], [pa, 1 - pb], [1 - pa, pb]];
            let weights = [Gf::ONE - u - w, u, w];
            let mut layer = 0;
            while layer < 2 {
                let wc = if layer == 0 {
                    Gf::ONE - frac[2]
                } else {
                    frac[2]
                };
                let mut i = 0;
                while i < 3 {
                    let (site, p) =
                        centre(base[0] + pos[i][0], base[1] + pos[i][1], base[2] + layer);
                    terms.push((site, p, weights[i] * wc));
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
                let wa = if da == 0 { Gf::ONE - frac[0] } else { frac[0] };
                let wb = if db == 0 { Gf::ONE - frac[1] } else { frac[1] };
                let wc = if dc == 0 { Gf::ONE - frac[2] } else { frac[2] };
                let (site, p) = centre(base[0] + da, base[1] + db, base[2] + dc);
                terms.push((site, p, wa * wb * wc));
                corner += 1;
            }
        }
    }
    terms.sort_by(|x, y| x.0.cmp(&y.0));
    let mut p = [Gf::ZERO; 3];
    for (_, centre, w) in &terms {
        p = [
            p[0] + *w * centre[0],
            p[1] + *w * centre[1],
            p[2] + *w * centre[2],
        ];
    }
    p
}

#[cfg(test)]
mod tests {
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
        let p = vertex_position_m(&m, &bx, [3 * 256, 4 * 256, 5 * 256]);
        let dir = bx.dir(3, 4);
        let r = Gf::from_f64(m.ladder.cell_radius_m(z * CHUNK_EDGE as i32 + 5, rung));
        assert_eq!(p, [dir[0] * r, dir[1] * r, dir[2] * r]);
        // Halfway between two centres along a: the mean of the two.
        let p = vertex_position_m(&m, &bx, [3 * 256 + 128, 4 * 256, 5 * 256]);
        let d2 = bx.dir(4, 4);
        let mid = [
            (dir[0] * r) * Gf::HALF + (d2[0] * r) * Gf::HALF,
            (dir[1] * r) * Gf::HALF + (d2[1] * r) * Gf::HALF,
            (dir[2] * r) * Gf::HALF + (d2[2] * r) * Gf::HALF,
        ];
        let mut k = 0;
        while k < 3 {
            assert!(
                (p[k] - mid[k]).abs() < Gf::from_f64(1e-6),
                "{p:?} vs {mid:?}"
            );
            k += 1;
        }
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
        let p = vertex_position_m(&m, &boxes[0], [base, base, 5 * 256]);
        let dir = boxes[0].dir(beyond - 1, beyond - 1);
        let r = Gf::from_f64(m.ladder.cell_radius_m(z * CHUNK_EDGE as i32 + 5, rung));
        assert_eq!(p, [dir[0] * r, dir[1] * r, dir[2] * r]);
        // One point of the prism, in the three local layouts: weights (wX, wY, wZ) = (86, 85, 85)
        // quanta. In +X's box (r[0] = X, r[1] = Y along a, r[2] = Z along b) that is local
        // (85, 85); in +Y's (r[0] = Y, r[1] = Z along +Y's a, r[2] = X) it is (85, 86); in +Z's
        // (r[0] = Z, r[1] = X, r[2] = Y) it is (86, 85). One point in metres from all three.
        let px = vertex_position_m(&m, &boxes[0], [base + 85, base + 85, 5 * 256 + 128]);
        let py = vertex_position_m(&m, &boxes[1], [base + 85, base + 86, 5 * 256 + 128]);
        let pz = vertex_position_m(&m, &boxes[2], [base + 86, base + 85, 5 * 256 + 128]);
        assert_eq!(px, py);
        assert_eq!(px, pz);
        // Weight one on the neighbour along a: that neighbour's centre.
        let p1 = vertex_position_m(&m, &boxes[0], [base + 256, base, 5 * 256]);
        let d1 = boxes[0].dir(beyond, beyond - 1);
        assert_eq!(p1, [d1[0] * r, d1[1] * r, d1[2] * r]);
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
        let p0 = vertex_position_m(&m, &b0, [0, 0, 5 * 256]);
        let d0 = b0.dir(0, 0);
        assert_eq!(p0, [d0[0] * r, d0[1] * r, d0[2] * r]);
        let pq = vertex_position_m(&m, &b0, [-64, -64, 5 * 256]);
        let dq1 = b0.dir(-1, 0);
        let dq2 = b0.dir(0, -1);
        let q = Gf::from_f64(0.25);
        let expect = [
            d0[0] * r * Gf::HALF + dq1[0] * r * q + dq2[0] * r * q,
            d0[1] * r * Gf::HALF + dq1[1] * r * q + dq2[1] * r * q,
            d0[2] * r * Gf::HALF + dq1[2] * r * q + dq2[2] * r * q,
        ];
        let mut k = 0;
        while k < 3 {
            assert!(
                (pq[k] - expect[k]).abs() < Gf::from_f64(1e-6),
                "{pq:?} vs {expect:?}"
            );
            k += 1;
        }
    }
}
