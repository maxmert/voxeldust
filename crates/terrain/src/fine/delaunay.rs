//! ★ THE FINE LEVELS' POINT GRAPH: THE DELAUNAY TRIANGULATION ON THE SPHERE (B1 (b); the way-in design revision 4
//! §3.1). Our own code: the bench's library stays a bench dev-dependency.
//!
//! The points are the fine level's integer directions at the recipe's 40 fraction bits ([`super::points`]). Two
//! predicates decide everything, both EXACT in 128-bit integers, so the same points give the same edges on every host
//! and in every tile (B64, SL10):
//! - **the orientation** of three directions, `det[a b c]` — positive when `c` lies to the left of the great circle from
//!   `a` to `b`, seen from outside the sphere. The products of three 40-bit words stay under 2¹²³.
//! - **the in-circle test**, which on the sphere is the orientation of four directions: `d` lies inside the circle
//!   through `a`, `b`, `c` (counter-clockwise from outside) exactly when `det[b − a, c − a, d − a] > 0` — the circle is
//!   the plane through the three cut from the sphere, and the small cap lies beyond the plane. No projection, so a face
//!   edge and a cube corner are nothing to the triangulation.
//!
//! The triangulation is incremental (Lawson 1977): each point, in a space-filling order, is found by a walk from the
//! last triangle, splits its triangle (or the edge it lies on), and the edges around it are flipped until each is
//! Delaunay. Three far vertices 30° from the points' mean direction hold the first triangle and leave with their
//! triangles at the end; near the domain's rim the graph may then lack a few hull edges, which the tile's buffer absorbs.
//! A point that repeats another's direction exactly is skipped (the hashed set never repeats one). A zero in-circle
//! value (four points on one circle — never seen on hashed directions) keeps the edge it finds.
//!
//! **Example.** The 128 m tile on the home planet's belt holds about 590 000 points with its buffer; each finds its
//! neighbours — about six — the same on the planet's shard and on every client, and a point in the overlap of two tiles
//! has the same neighbours in both.

use vd_recipe::Gi;
use vd_recipe::bend::{DIR_BITS, normalise};

/// No triangle or point.
pub const NONE: u32 = u32::MAX;

/// `√3/2` at 40 fraction bits (the far vertices' 30° and 120° steps): 952 233 214 474.85… rounded.
const SQRT3_HALF_Q40: i64 = 952_233_214_475;

/// A direction as three 40-fraction-bit words.
pub type Dir = [Gi; 3];

/// A triangle's next and previous corner (counter-clockwise): `NEXT[k]` and `PREV[k]` (the crate has no `%`).
const NEXT: [usize; 3] = [1, 2, 0];
const PREV: [usize; 3] = [2, 0, 1];

fn raw(d: &Dir) -> [i128; 3] {
    [
        i128::from(d[0].raw()),
        i128::from(d[1].raw()),
        i128::from(d[2].raw()),
    ]
}

/// The squared chord between two directions at 80 fraction bits (exact).
#[must_use]
pub fn chord2(a: &Dir, b: &Dir) -> i128 {
    let (a, b) = (raw(a), raw(b));
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// `det[a b c]`: positive when `c` lies to the left of the great circle from `a` to `b`, seen from outside.
#[must_use]
pub fn orient(a: &Dir, b: &Dir, c: &Dir) -> i128 {
    let (a, b, c) = (raw(a), raw(b), raw(c));
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// `det[b − a, c − a, d − a]`: positive when `d` lies inside the circle through `a`, `b`, `c` (counter-clockwise from
/// outside).
#[must_use]
pub fn in_circle(a: &Dir, b: &Dir, c: &Dir, d: &Dir) -> i128 {
    let (a, b, c, d) = (raw(a), raw(b), raw(c), raw(d));
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    w[0] * (u[1] * v[2] - u[2] * v[1]) - w[1] * (u[0] * v[2] - u[2] * v[0])
        + w[2] * (u[0] * v[1] - u[1] * v[0])
}

/// `orient` on the plane: positive when `(a, b, c)` turn counter-clockwise (the third word is not read).
fn orient_plane(a: &Dir, b: &Dir, c: &Dir) -> i128 {
    let (a, b, c) = (raw(a), raw(b), raw(c));
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// `in_circle` on the plane: positive when `d` lies inside the circle through the counter-clockwise `a`, `b`, `c`.
fn in_circle_plane(a: &Dir, b: &Dir, c: &Dir, d: &Dir) -> i128 {
    let (a, b, c, d) = (raw(a), raw(b), raw(c), raw(d));
    let row = |p: [i128; 3]| {
        let (x, y) = (p[0] - d[0], p[1] - d[1]);
        [x, y, x * x + y * y]
    };
    let (u, v, w) = (row(a), row(b), row(c));
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}

/// Where a triangulation's predicates are read: on the sphere (the directions themselves) or on a plane (two words of
/// each point, the third nought).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Space {
    Sphere,
    Plane,
}

fn orient_in(space: Space, a: &Dir, b: &Dir, c: &Dir) -> i128 {
    match space {
        Space::Sphere => orient(a, b, c),
        Space::Plane => orient_plane(a, b, c),
    }
}

fn in_circle_in(space: Space, a: &Dir, b: &Dir, c: &Dir, d: &Dir) -> i128 {
    match space {
        Space::Sphere => in_circle(a, b, c, d),
        Space::Plane => in_circle_plane(a, b, c, d),
    }
}

/// A triangle: its three vertices counter-clockwise from outside, and across each vertex's opposite edge the
/// neighbouring triangle ([`NONE`] at the hull).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tri {
    pub v: [u32; 3],
    pub n: [u32; 3],
}

/// The triangulation of a point set: its triangles over the points (the far vertices removed), across each
/// triangle's vertex's opposite edge its neighbour ([`NONE`] at the set's hull), and each point's neighbours in one
/// flat table (`start[p]..start[p + 1]` into `adj`), each list in increasing index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    pub tris: Vec<[u32; 3]>,
    pub tri_n: Vec<[u32; 3]>,
    pub start: Vec<u32>,
    pub adj: Vec<u32>,
}

impl Graph {
    /// The neighbours of point `p`.
    #[must_use]
    pub fn neighbours(&self, p: usize) -> &[u32] {
        &self.adj[self.start[p] as usize..self.start[p + 1] as usize]
    }
}

/// The 2-D Hilbert index of `(x, y)` on a `2^16` grid (Skilling's form): the insertion order, so each walk is short.
fn hilbert(mut x: u32, mut y: u32) -> u64 {
    let mut d = 0u64;
    let mut s = 1u32 << 15;
    while s > 0 {
        let rx = u32::from(x & s > 0);
        let ry = u32::from(y & s > 0);
        d += u64::from(s) * u64::from(s) * u64::from((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = 0xFFFF - x;
                y = 0xFFFF - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

/// The unit direction of `v` (a sum or a combination at 40 fraction bits, at least about half a unit long).
fn unit(v: [i128; 3]) -> Dir {
    // Bring the vector UNDER one unit at 40 bits before the recipe's normalise, whose reciprocal root holds only for
    // components under one (a square sum under 3): a 2.3-unit sum came back 17 units long.
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let bits = (128 - len2.leading_zeros()) >> 1;
    let shift = bits.saturating_sub(DIR_BITS - 1);
    normalise([
        Gi::new((v[0] >> shift) as i64),
        Gi::new((v[1] >> shift) as i64),
        Gi::new((v[2] >> shift) as i64),
    ])
}

/// ★ THE TRIANGULATION of `pts` (each a direction): `None` for fewer than three points.
#[must_use]
pub fn triangulate(pts: &[Dir]) -> Option<Graph> {
    build(pts, Space::Sphere)
}

/// ★ THE TRIANGULATION OF A FINE LEVEL TOO FINE FOR THE SPHERE'S TEST (ruling B95, the 16 m level): the points seen on
/// the plane of the cube's `face` — each direction less its component along the face's normal (the other two words in
/// the face's own turn, so a triangle counter-clockwise on the plane is counter-clockwise from outside), each word
/// shifted down by `shift` bits (a floor, the same for every tile of the level). The sphere's in-circle test reads how
/// far a fourth point stands off the plane of three; at 16 m that is about three units of the directions' 40-bit
/// words, the size of their rounding, so the points are not on one sphere and the flips can turn for ever. On the
/// plane the test is exact with room to spare, and every tile of a face sees the same points the same way, so two
/// tiles build the same edges where they overlap. The lengths and areas the law reads stay the sphere's
/// ([`super::geometry`]). `None` for fewer than three points.
///
/// **Example.** The 16 m tile under hull 17 on the home planet: about 590 000 points over 12.3 km of face 5 (−Z), each
/// seen as its `(y, x)` words over 2⁸, about 8 000 units between neighbours; about six neighbours a point.
#[must_use]
pub fn triangulate_on_face(pts: &[Dir], face: vd_seed::bend::Face, shift: u32) -> Option<Graph> {
    let f = face.index();
    let axis = usize::from(f >> 1);
    let (mut u, mut v) = (NEXT[axis], PREV[axis]);
    if f & 1 == 1 {
        core::mem::swap(&mut u, &mut v);
    }
    let plane: Vec<Dir> = pts
        .iter()
        .map(|p| {
            [
                Gi::new(p[u].raw() >> shift),
                Gi::new(p[v].raw() >> shift),
                Gi::ZERO,
            ]
        })
        .collect();
    build(&plane, Space::Plane)
}

/// The mean direction of `pts` on the sphere and a tangent frame there (integer cross products, normalised): the index of
/// its least-aligned coordinate, and the three far vertices 30° from it, 120° apart.
fn sphere_frame(pts: &[Dir]) -> (usize, [Dir; 3]) {
    let mut sum = [0i128; 3];
    for p in pts {
        let r = raw(p);
        sum = [sum[0] + r[0], sum[1] + r[1], sum[2] + r[2]];
    }
    let c0 = raw(&unit(sum));
    let least = if c0[0].abs() <= c0[1].abs() && c0[0].abs() <= c0[2].abs() {
        0
    } else if c0[1].abs() <= c0[2].abs() {
        1
    } else {
        2
    };
    let mut axis = [0i128; 3];
    axis[least] = 1i128 << DIR_BITS;
    let cross = |a: [i128; 3], b: [i128; 3]| {
        [
            (a[1] * b[2] - a[2] * b[1]) >> DIR_BITS,
            (a[2] * b[0] - a[0] * b[2]) >> DIR_BITS,
            (a[0] * b[1] - a[1] * b[0]) >> DIR_BITS,
        ]
    };
    let e1 = raw(&unit(cross(c0, axis)));
    let e2 = raw(&unit(cross(c0, e1)));
    // The three far vertices: 30° from the mean direction, 120° apart (cos 30° = √3/2, sin 30° = ½; cos/sin of 120° steps
    // ∓½ and ±√3/2).
    let q = i128::from(SQRT3_HALF_Q40);
    let half = 1i128 << (DIR_BITS - 1);
    let far = |cu: i128, su: i128| {
        let t = [
            (cu * e1[0] + su * e2[0]) >> DIR_BITS,
            (cu * e1[1] + su * e2[1]) >> DIR_BITS,
            (cu * e1[2] + su * e2[2]) >> DIR_BITS,
        ];
        unit([
            ((q * c0[0]) >> DIR_BITS) + ((half * t[0]) >> DIR_BITS),
            ((q * c0[1]) >> DIR_BITS) + ((half * t[1]) >> DIR_BITS),
            ((q * c0[2]) >> DIR_BITS) + ((half * t[2]) >> DIR_BITS),
        ])
    };
    let one = 1i128 << DIR_BITS;
    (least, [far(one, 0), far(-half, q), far(-half, -q)])
}

/// The triangulation of `pts` read in `space` (see [`triangulate`] and [`triangulate_on_face`]).
fn build(pts: &[Dir], space: Space) -> Option<Graph> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let mut all: Vec<Dir> = pts.to_vec();
    // ★ The insertion order reads the points' two least-aligned coordinates. On the sphere they come from the mean
    // direction, which also places the three far vertices; on the plane the third word is nought, so it is the least
    // aligned — and the mean direction is never taken there (a tile at a face's centre sums to nearly nothing, and its
    // normalising never ends).
    let least = match space {
        Space::Sphere => {
            let (least, far) = sphere_frame(pts);
            all.extend(far);
            least
        }
        Space::Plane => {
            // A triangle around the points' box: twenty half-sides out each way below, twenty above.
            let (mut lo, mut hi) = ([i64::MAX; 2], [i64::MIN; 2]);
            for p in pts {
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k].raw());
                    hi[k] = hi[k].max(p[k].raw());
                }
            }
            let h = ((hi[0] - lo[0]).max(hi[1] - lo[1]) >> 1).max(1);
            let (cx, cy) = (
                lo[0] + ((hi[0] - lo[0]) >> 1),
                lo[1] + ((hi[1] - lo[1]) >> 1),
            );
            let at = |x: i64, y: i64| [Gi::new(x), Gi::new(y), Gi::ZERO];
            all.extend([
                at(cx - 20 * h, cy - 10 * h),
                at(cx + 20 * h, cy - 10 * h),
                at(cx, cy + 20 * h),
            ]);
            2
        }
    };
    let (a, b, c) = (n as u32, n as u32 + 1, n as u32 + 2);
    let first = if orient_in(space, &all[a as usize], &all[b as usize], &all[c as usize]) > 0 {
        [a, b, c]
    } else {
        [a, c, b]
    };
    let mut tris = vec![Tri {
        v: first,
        n: [NONE; 3],
    }];
    // The insertion order: the Hilbert index of each point's two least-aligned coordinates over the set's box.
    let (ka, kb) = match least {
        0 => (1usize, 2usize),
        1 => (0, 2),
        _ => (0, 1),
    };
    let key = |p: &Dir, k: usize| p[k].raw();
    let (mut lo_a, mut hi_a, mut lo_b, mut hi_b) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
    for p in pts {
        lo_a = lo_a.min(key(p, ka));
        hi_a = hi_a.max(key(p, ka));
        lo_b = lo_b.min(key(p, kb));
        hi_b = hi_b.max(key(p, kb));
    }
    // The box's larger side brought under 2^16 by a shift (the crate has no `/`).
    let span = (hi_a - lo_a).max(hi_b - lo_b).max(1) as u64;
    let shift = (64 - span.leading_zeros()).saturating_sub(16);
    let grid = |v: i64, lo: i64| (((v - lo) as u64) >> shift) as u32;
    let mut order: Vec<(u64, u32)> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                hilbert(grid(key(p, ka), lo_a), grid(key(p, kb), lo_b)),
                i as u32,
            )
        })
        .collect();
    order.sort_unstable();
    let mut last = 0u32;
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for &(_, p) in &order {
        let pd = all[p as usize];
        // The walk: cross any edge the point lies beyond.
        let mut t = last;
        let mut on_edge;
        'walk: loop {
            let tv = tris[t as usize].v;
            on_edge = usize::MAX;
            for e in 0..3 {
                let (x, y) = (tv[NEXT[e]], tv[PREV[e]]);
                let o = orient_in(space, &all[x as usize], &all[y as usize], &pd);
                if o < 0 {
                    t = tris[t as usize].n[e];
                    continue 'walk;
                }
                if o == 0 {
                    on_edge = e;
                }
            }
            break;
        }
        // A repeated direction: skip it.
        let tv = tris[t as usize].v;
        if tv.iter().any(|&v| all[v as usize] == pd) {
            last = t;
            continue;
        }
        stack.clear();
        if on_edge == usize::MAX {
            // Split the triangle into three around `p`.
            let [x, y, z] = tv;
            let [nx, ny, nz] = tris[t as usize].n;
            let (t1, t2) = (tris.len() as u32, tris.len() as u32 + 1);
            tris[t as usize] = Tri {
                v: [p, y, z],
                n: [nx, t1, t2],
            };
            tris.push(Tri {
                v: [x, p, z],
                n: [t, ny, t2],
            });
            tris.push(Tri {
                v: [x, y, p],
                n: [t, t1, nz],
            });
            relink(&mut tris, ny, t, t1);
            relink(&mut tris, nz, t, t2);
            stack.extend([(t, 0usize), (t1, 1), (t2, 2)]);
            last = t;
        } else {
            // On the edge opposite vertex `on_edge` of `t`: split both triangles beside it into two.
            let e = on_edge;
            let u = tris[t as usize].n[e];
            let (x, y, z) = (tv[e], tv[NEXT[e]], tv[PREV[e]]);
            let tn = tris[t as usize].n;
            let (n_y, n_z) = (tn[NEXT[e]], tn[PREV[e]]);
            let t_new = tris.len() as u32;
            // t: (x, y, p) and t_new: (x, p, z).
            tris[t as usize] = Tri {
                v: [x, y, p],
                n: [NONE, t_new, n_z],
            };
            tris.push(Tri {
                v: [x, p, z],
                n: [NONE, n_y, t],
            });
            relink(&mut tris, n_y, t, t_new);
            if u == NONE {
                stack.extend([(t, 2usize), (t_new, 1)]);
            } else {
                let uv = tris[u as usize].v;
                let f = (0..3)
                    .find(|&k| uv[k] != y && uv[k] != z)
                    .expect("the far vertex of the edge's other triangle");
                let w = uv[f];
                let un = tris[u as usize].n;
                // In `u` the edge runs z → y; its other two neighbours sit opposite y and opposite z.
                let ky = (0..3).find(|&k| uv[k] == y).expect("y in u");
                let kz = (0..3).find(|&k| uv[k] == z).expect("z in u");
                let (n_uy, n_uz) = (un[ky], un[kz]);
                let u_new = tris.len() as u32;
                // u: (w, z, p) and u_new: (w, p, y).
                tris[u as usize] = Tri {
                    v: [w, z, p],
                    n: [t_new, u_new, n_uy],
                };
                tris.push(Tri {
                    v: [w, p, y],
                    n: [t, n_uz, u],
                });
                relink(&mut tris, n_uz, u, u_new);
                tris[t as usize].n[0] = u_new;
                tris[t_new as usize].n[0] = u;
                stack.extend([(t, 2usize), (t_new, 1), (u, 2), (u_new, 1)]);
            }
            last = t;
        }
        // Lawson's flips: the edge of `(tri, k)` opposite vertex `k` (the new point) must be Delaunay.
        while let Some((ti, k)) = stack.pop() {
            let tv = tris[ti as usize].v;
            let nb = tris[ti as usize].n[k];
            if nb == NONE {
                continue;
            }
            let (pv, x, y) = (tv[k], tv[NEXT[k]], tv[PREV[k]]);
            let nv = tris[nb as usize].v;
            let m = (0..3)
                .find(|&j| nv[j] != x && nv[j] != y)
                .expect("the opposite vertex");
            let d = nv[m];
            if in_circle_in(
                space,
                &all[pv as usize],
                &all[x as usize],
                &all[y as usize],
                &all[d as usize],
            ) <= 0
            {
                continue;
            }
            // Flip x–y to p–d: ti = (p, x, d), nb = (p, d, y).
            let tn = tris[ti as usize].n;
            let nn = tris[nb as usize].n;
            let a_px = tn[PREV[k]]; // across p–x (opposite y)
            let a_py = tn[NEXT[k]]; // across p–y (opposite x)
            let kx = (0..3).find(|&j| nv[j] == x).expect("x in nb");
            let ky = (0..3).find(|&j| nv[j] == y).expect("y in nb");
            let a_dy = nn[kx]; // across d–y (opposite x)
            let a_xd = nn[ky]; // across x–d (opposite y)
            tris[ti as usize] = Tri {
                v: [pv, x, d],
                n: [a_xd, nb, a_px],
            };
            tris[nb as usize] = Tri {
                v: [pv, d, y],
                n: [a_dy, a_py, ti],
            };
            relink(&mut tris, a_xd, nb, ti);
            relink(&mut tris, a_py, ti, nb);
            stack.extend([(ti, 0usize), (nb, 0usize)]);
        }
    }
    // The far vertices leave with their triangles (a link to one becomes the hull's NONE); each point's neighbours
    // from the rest.
    let mut renum = vec![NONE; tris.len()];
    let mut real: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for (ti, t) in tris.iter().enumerate() {
        if t.v.iter().all(|&v| (v as usize) < n) {
            renum[ti] = real.len() as u32;
            real.push(t.v);
        }
    }
    let tri_n: Vec<[u32; 3]> = tris
        .iter()
        .filter(|t| t.v.iter().all(|&v| (v as usize) < n))
        .map(|t| {
            t.n.map(|k| if k == NONE { NONE } else { renum[k as usize] })
        })
        .collect();
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(real.len() * 6);
    for t in &real {
        for e in 0..3 {
            let (x, y) = (t[e], t[NEXT[e]]);
            pairs.push((x, y));
            pairs.push((y, x));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    let mut start = vec![0u32; n + 1];
    for &(x, _) in &pairs {
        start[x as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let adj = pairs.into_iter().map(|(_, y)| y).collect();
    Some(Graph {
        tris: real,
        tri_n,
        start,
        adj,
    })
}

/// In triangle `t`, the neighbour link that pointed at `from` now points at `to`.
fn relink(tris: &mut [Tri], t: u32, from: u32, to: u32) {
    if t == NONE {
        return;
    }
    for k in 0..3 {
        if tris[t as usize].n[k] == from {
            tris[t as usize].n[k] = to;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_seed::bend::Face;

    /// A jittered 12 × 12 grid of points near the centre of `face`, every point's normal word the face's own.
    fn grid_on(face: Face) -> Vec<Dir> {
        let f = face.index();
        let axis = usize::from(f >> 1);
        let normal = if f & 1 == 1 {
            -(1i64 << 40)
        } else {
            1i64 << 40
        };
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut jitter = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 38) as i64) & ((1 << 26) - 1)
        };
        let mut pts = Vec::with_capacity(144);
        for j in -6..6i64 {
            for i in -6..6i64 {
                let mut d = [Gi::ZERO; 3];
                d[axis] = Gi::new(normal);
                d[NEXT[axis]] = Gi::new((i << 28) + jitter());
                d[PREV[axis]] = Gi::new((j << 28) + jitter());
                pts.push(d);
            }
        }
        pts
    }

    /// ★ THE TRIANGULATION ON A FACE'S PLANE (the 16 m level): on an even and an odd face every triangle turns
    /// counter-clockwise seen from outside the sphere and its circle holds no other point; a shift reads the words as the
    /// same points shifted; fewer than three points have no graph. The words are read over 2⁸ as the 16 m tile reads them:
    /// the in-circle test multiplies four coordinate differences and the far triangle stands 20 half-sides out, so a
    /// tile's coordinates must stay under about 2²⁷ — a 16 m tile's reach is 2²³ over 2⁸ (MEASURED: this grid read
    /// unshifted, its coordinates 2³¹, overflows the 128-bit product).
    #[test]
    fn the_plane_triangulation_is_delaunay_and_turns_as_the_sphere_on_every_face() {
        for face in [Face::PosX, Face::NegY] {
            let pts = grid_on(face);
            let graph = triangulate_on_face(&pts, face, 8).expect("a graph");
            assert!(graph.tris.len() > 200, "{}", graph.tris.len());
            let f = face.index();
            let axis = usize::from(f >> 1);
            let (mut u, mut v) = (NEXT[axis], PREV[axis]);
            if f & 1 == 1 {
                core::mem::swap(&mut u, &mut v);
            }
            let plane: Vec<Dir> = pts
                .iter()
                .map(|p| [Gi::new(p[u].raw() >> 8), Gi::new(p[v].raw() >> 8), Gi::ZERO])
                .collect();
            for t in &graph.tris {
                let [a, b, c] = t.map(|k| k as usize);
                assert!(orient(&pts[a], &pts[b], &pts[c]) > 0, "{face:?} {t:?}");
                let inside = (0..pts.len())
                    .filter(|&d| (d != a) & (d != b) & (d != c))
                    .filter(|&d| in_circle_plane(&plane[a], &plane[b], &plane[c], &plane[d]) > 0)
                    .count();
                assert_eq!(inside, 0, "{face:?} {t:?}");
            }
            assert_eq!(
                (0..pts.len())
                    .filter(|&p| graph.neighbours(p).len() < 2)
                    .count(),
                0
            );
            // The shift: the words over 2⁸ are the points shifted down by 8 bits.
            let down: Vec<Dir> = pts
                .iter()
                .map(|p| p.map(|w| Gi::new(w.raw() >> 8)))
                .collect();
            assert_eq!(Some(graph), triangulate_on_face(&down, face, 0));
        }
        assert_eq!(
            triangulate_on_face(&grid_on(Face::PosX)[..2], Face::PosX, 8),
            None
        );
    }
}
