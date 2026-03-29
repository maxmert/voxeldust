//! Icosphere mesh generation for sphere rendering.
//! Uses a well-tested icosahedron base with consistent CCW winding,
//! then subdivides by splitting each triangle into 4.

use std::collections::HashMap;

pub struct IcoSphere {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl IcoSphere {
    /// Generate a unit icosphere with the given number of subdivisions.
    /// subdivisions=0: 20 tris, 1: 80, 2: 320, 3: 1280, 4: 5120.
    pub fn generate(subdivisions: u32) -> Self {
        let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

        let mut vertices: Vec<[f32; 3]> = vec![
            normalize([-1.0,  t,  0.0]),  // 0
            normalize([ 1.0,  t,  0.0]),  // 1
            normalize([-1.0, -t,  0.0]),  // 2
            normalize([ 1.0, -t,  0.0]),  // 3
            normalize([ 0.0, -1.0,  t]),  // 4
            normalize([ 0.0,  1.0,  t]),  // 5
            normalize([ 0.0, -1.0, -t]),  // 6
            normalize([ 0.0,  1.0, -t]),  // 7
            normalize([ t,  0.0, -1.0]),  // 8
            normalize([ t,  0.0,  1.0]),  // 9
            normalize([-t,  0.0, -1.0]),  // 10
            normalize([-t,  0.0,  1.0]),  // 11
        ];

        // 20 triangles of icosahedron — all CCW when viewed from outside.
        // 5 faces around vertex 0 (top)
        let mut indices: Vec<u32> = vec![
            0, 11, 5,
            0, 5, 1,
            0, 1, 7,
            0, 7, 10,
            0, 10, 11,
            // 5 adjacent faces
            1, 5, 9,
            5, 11, 4,
            11, 10, 2,
            10, 7, 6,
            7, 1, 8,
            // 5 faces around vertex 3 (bottom)
            3, 9, 4,
            3, 4, 2,
            3, 2, 6,
            3, 6, 8,
            3, 8, 9,
            // 5 adjacent faces
            4, 9, 5,
            2, 4, 11,
            6, 2, 10,
            8, 6, 7,
            9, 8, 1,
        ];

        let mut midpoint_cache = HashMap::new();

        for _ in 0..subdivisions {
            let mut new_indices = Vec::with_capacity(indices.len() * 4);

            for tri in indices.chunks(3) {
                let v0 = tri[0];
                let v1 = tri[1];
                let v2 = tri[2];

                let a = get_midpoint(v0, v1, &mut vertices, &mut midpoint_cache);
                let b = get_midpoint(v1, v2, &mut vertices, &mut midpoint_cache);
                let c = get_midpoint(v2, v0, &mut vertices, &mut midpoint_cache);

                // 4 sub-triangles preserving CCW winding:
                //       v0
                //      / \
                //     a---c
                //    / \ / \
                //   v1--b--v2
                new_indices.extend_from_slice(&[v0, a, c]);
                new_indices.extend_from_slice(&[a, v1, b]);
                new_indices.extend_from_slice(&[c, b, v2]);
                new_indices.extend_from_slice(&[a, b, c]);
            }

            indices = new_indices;
            midpoint_cache.clear();
        }

        Self { vertices, indices }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn get_midpoint(
    a: u32,
    b: u32,
    vertices: &mut Vec<[f32; 3]>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let va = vertices[a as usize];
    let vb = vertices[b as usize];
    let mid = normalize([
        (va[0] + vb[0]) * 0.5,
        (va[1] + vb[1]) * 0.5,
        (va[2] + vb[2]) * 0.5,
    ]);

    let idx = vertices.len() as u32;
    vertices.push(mid);
    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_vertex_count() {
        let s0 = IcoSphere::generate(0);
        assert_eq!(s0.vertices.len(), 12);
        assert_eq!(s0.indices.len(), 60);

        let s4 = IcoSphere::generate(4);
        assert_eq!(s4.indices.len(), 20 * 3 * 4_usize.pow(4));
    }

    #[test]
    fn all_vertices_on_unit_sphere() {
        let sphere = IcoSphere::generate(3);
        for v in &sphere.vertices {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "vertex not on unit sphere: len = {len}");
        }
    }

    #[test]
    fn consistent_winding_order() {
        // For a convex mesh viewed from outside, all face normals should point outward.
        // Cross product of two edges should point away from origin.
        let sphere = IcoSphere::generate(2);
        let mut bad = 0;
        for tri in sphere.indices.chunks(3) {
            let v0 = sphere.vertices[tri[0] as usize];
            let v1 = sphere.vertices[tri[1] as usize];
            let v2 = sphere.vertices[tri[2] as usize];

            // Edges.
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

            // Cross product (face normal).
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];

            // Center of triangle.
            let cx = (v0[0] + v1[0] + v2[0]) / 3.0;
            let cy = (v0[1] + v1[1] + v2[1]) / 3.0;
            let cz = (v0[2] + v1[2] + v2[2]) / 3.0;

            // Dot(normal, center) should be positive (outward facing).
            let dot = n[0] * cx + n[1] * cy + n[2] * cz;
            if dot < 0.0 {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{bad} triangles have inward-facing normals (bad winding)");
    }
}
