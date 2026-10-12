//! ★ THE FINE GRAPH'S GEOMETRY (B1 (d); the way-in design revision 4 §3.1): each point's AREA on the sphere, each
//! link's LENGTH, and the WALK that finds the triangle under a direction with its barycentric weights — everything the
//! law reads about the graph's shape, in the fenced float ([`Gf`]: IEEE-exact `+ − × ÷ √`, one answer on every target).
//!
//! **The area** is the mixed Voronoi area (Meyer, Desbrun, Schröder & Barr 2003): on a triangle with no obtuse angle each
//! corner takes its Voronoi part, `(|ab|²·cot C + |ac|²·cot B)/8`; on an obtuse triangle the obtuse corner takes half the
//! triangle and the others a quarter each — so the areas tile the surface with no negative part. Each triangle is read
//! in its own plane (a 128 m triangle on a 6 000 km sphere departs from its plane by under a millimetre).
//!
//! **The weights** come from the exact orientations: `q = λ_a·a + λ_b·b + λ_c·c` by Cramer's rule, `λ_a = det[q b c] /
//! det[a b c]`, so the ray from the centre through `q` meets the triangle's plane at the weights `λ / Σλ` — integers
//! until the one division, the same on every host.
//!
//! **Example.** A 128 m raster cell's centre on the home planet's belt falls in the triangle of three points; its height
//! is their heights by these weights, the same quarter metre on the planet's shard and on every client.

use super::delaunay::{Dir, Graph, NONE, orient};
use crate::gf::Gf;
use vd_recipe::bend::DIR_BITS;

/// A direction's words as a vector of metres on a sphere of radius `r_m`.
fn metres(d: &Dir, r_m: Gf) -> [Gf; 3] {
    let scale = r_m / Gf::from_f64((1u64 << DIR_BITS) as f64);
    [
        Gf::from_i64(d[0].raw()) * scale,
        Gf::from_i64(d[1].raw()) * scale,
        Gf::from_i64(d[2].raw()) * scale,
    ]
}

fn sub(a: [Gf; 3], b: [Gf; 3]) -> [Gf; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [Gf; 3], b: [Gf; 3]) -> Gf {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [Gf; 3], b: [Gf; 3]) -> [Gf; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Each point's mixed Voronoi area (m²) and each link's length (m), the lengths in `graph.adj`'s order.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub area: Vec<Gf>,
    pub len: Vec<Gf>,
}

/// ★ THE GEOMETRY of `graph` over `dirs` on a sphere of radius `r_m`.
#[must_use]
pub fn geometry(dirs: &[Dir], graph: &Graph, r_m: Gf) -> Geometry {
    let p: Vec<[Gf; 3]> = dirs.iter().map(|d| metres(d, r_m)).collect();
    let mut area = vec![Gf::ZERO; dirs.len()];
    let (half, quarter, eighth) = (Gf::HALF, Gf::from_f64(0.25), Gf::from_f64(0.125));
    for t in &graph.tris {
        let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
        let (ab, ac, bc) = (sub(b, a), sub(c, a), sub(c, b));
        let n = cross(ab, ac);
        let twice = dot(n, n).sqrt();
        if twice <= Gf::ZERO {
            continue;
        }
        let whole = twice * half;
        // The dot products at each corner: a negative one is an obtuse corner.
        let (da, db, dc) = (dot(ab, ac), dot(sub(a, b), bc), dot(sub(a, c), sub(b, c)));
        let corners = [t[0] as usize, t[1] as usize, t[2] as usize];
        if da < Gf::ZERO || db < Gf::ZERO || dc < Gf::ZERO {
            let obtuse = if da < Gf::ZERO { 0 } else if db < Gf::ZERO { 1 } else { 2 };
            for (k, &v) in corners.iter().enumerate() {
                area[v] += if k == obtuse { whole * half } else { whole * quarter };
            }
        } else {
            // cot at a corner = its dot over twice the area.
            let (cot_a, cot_b, cot_c) = (da / twice, db / twice, dc / twice);
            let (lab, lac, lbc) = (dot(ab, ab), dot(ac, ac), dot(bc, bc));
            area[corners[0]] += (lab * cot_c + lac * cot_b) * eighth;
            area[corners[1]] += (lab * cot_c + lbc * cot_a) * eighth;
            area[corners[2]] += (lac * cot_b + lbc * cot_a) * eighth;
        }
    }
    let mut len = Vec::with_capacity(graph.adj.len());
    for (i, pi) in p.iter().enumerate() {
        for &m in &graph.adj[graph.start[i] as usize..graph.start[i + 1] as usize] {
            let d = sub(*pi, p[m as usize]);
            len.push(dot(d, d).sqrt());
        }
    }
    Geometry { area, len }
}

/// ★ THE WALK: the triangle of `graph` under `q`, starting at triangle `from`, and its barycentric weights; `None`
/// where `q` lies outside the graph's hull.
#[must_use]
pub fn locate(dirs: &[Dir], graph: &Graph, from: u32, q: &Dir) -> Option<(u32, [Gf; 3])> {
    let mut t = from;
    'walk: loop {
        let v = graph.tris[t as usize];
        for e in 0..3 {
            let (x, y) = (v[NEXT[e]], v[PREV[e]]);
            if orient(&dirs[x as usize], &dirs[y as usize], q) < 0 {
                let n = graph.tri_n[t as usize][e];
                if n == NONE {
                    return None;
                }
                t = n;
                continue 'walk;
            }
        }
        let (a, b, c) = (&dirs[v[0] as usize], &dirs[v[1] as usize], &dirs[v[2] as usize]);
        let la = Gf::from_f64(orient(q, b, c) as f64);
        let lb = Gf::from_f64(orient(a, q, c) as f64);
        let lc = Gf::from_f64(orient(a, b, q) as f64);
        let sum = la + lb + lc;
        return Some((t, [la / sum, lb / sum, lc / sum]));
    }
}

/// A triangle's next and previous corner (the crate has no `%`).
const NEXT: [usize; 3] = [1, 2, 0];
const PREV: [usize; 3] = [2, 0, 1];

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::delaunay::triangulate;
    use vd_recipe::Gi;

    /// A direction's words: `(x, y)` units across a plane `2^40` units out (the test's sphere has that radius in metres,
    /// so a unit is a metre).
    fn at(x: i64, y: i64) -> Dir {
        [Gi::new(x), Gi::new(y), Gi::new(1 << DIR_BITS)]
    }

    fn near(a: Gf, b: f64) -> bool {
        (a - Gf::from_f64(b)).abs() < Gf::from_f64(1e-12)
    }

    #[test]
    fn the_mixed_area_tiles_each_triangle_and_the_lengths_are_the_links() {
        // An acute triangle (0, 1, 2); one obtuse triangle three times, its obtuse corner (5) last, first and second;
        // and a flat one that adds nothing.
        let dirs = [at(0, 0), at(4, 0), at(2, 3), at(10, 0), at(20, 0), at(15, 1)];
        let graph = Graph {
            tris: vec![[0, 1, 2], [3, 4, 5], [5, 3, 4], [3, 5, 4], [0, 0, 1]],
            tri_n: Vec::new(),
            start: vec![0, 1, 1, 1, 1, 1, 2],
            adj: vec![1, 3],
        };
        let g = geometry(&dirs, &graph, Gf::from_f64((1u64 << DIR_BITS) as f64));
        // The acute triangle's area 6 by its corners' Voronoi parts: 184/96, 184/96, 208/96.
        assert!(near(g.area[0], 184.0 / 96.0));
        assert!(near(g.area[1], 184.0 / 96.0));
        assert!(near(g.area[2], 208.0 / 96.0));
        // The obtuse triangle's area 5, three times: the obtuse corner half each time, the others a quarter.
        assert_eq!([g.area[3], g.area[4], g.area[5]], [Gf::from_f64(3.75), Gf::from_f64(3.75), Gf::from_f64(7.5)]);
        assert_eq!(g.len, vec![Gf::from_f64(4.0), Gf::from_f64(26.0).sqrt()]);
    }

    #[test]
    fn the_walk_finds_the_triangle_under_a_direction_and_none_outside_the_hull() {
        let inv_n = vd_recipe::bend::inv_n_of(1_024);
        let dirs: Vec<Dir> = [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .map(|&(i, j)| vd_recipe::bend::direction(0, i, j, inv_n))
            .collect();
        let graph = triangulate(&dirs).expect("a graph");
        assert_eq!(graph.tris.len(), 2);
        // Each triangle's centroid, walked to from the first triangle, is that triangle at weights near a third each.
        for (t, v) in graph.tris.iter().enumerate() {
            let q: Dir = [0, 1, 2].map(|k| dirs[v[0] as usize][k] + dirs[v[1] as usize][k] + dirs[v[2] as usize][k]);
            let (found, w) = locate(&dirs, &graph, 0, &q).expect("inside");
            assert_eq!(found, t as u32);
            for x in w {
                assert!((x - Gf::from_f64(1.0 / 3.0)).abs() < Gf::from_f64(1e-3), "{w:?}");
            }
        }
        // Beyond the first corner, away from the opposite one: outside the hull.
        let q: Dir = [0, 1, 2].map(|k| dirs[0][k] + dirs[0][k] - dirs[3][k]);
        assert_eq!(locate(&dirs, &graph, 0, &q), None);
    }
}
