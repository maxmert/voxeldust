//! ★ THE EXTRACTOR — naive surface nets over the sample box (ruling V10 S6-1), in EXACT INTEGER
//! ARITHMETIC: no float touches a vertex.
//!
//! A `2 × 2 × 2` group of cells whose eight signs differ gets ONE vertex at the average of its edge
//! crossings. A crossing on an edge between gaps `d0` (rock, negative) and `d1` (air) sits at
//! `t = |d0| / (|d0| + |d1|)` from the rock end — a ratio of two small integers, so the vertex is a
//! rational number (per axis, a denominator no larger than `255⁴`, which a 64-bit integer holds).
//! The average is taken exactly and rounded ONCE to the vertex quantum (`1/256` cell, S6-4), half to
//! EVEN, which is the one rounding that gives the same quantum from a chunk whose local axis runs
//! the other way across a seam. Two hosts, or two chunks that share a vertex, cannot differ by an
//! ulp because there is no ulp.
//!
//! **At a cube corner** three faces meet and a ring of cells has THREE members, not four: the group
//! there is a triangular PRISM (three real columns, two layers, nine edges), and its vertex is the
//! average of the prism's crossings. The box's corner phantom column is a placeholder the extractor
//! never reads; the three faces' chunks see the same three real columns in three local layouts that
//! are affine images of one another, and an exact average is preserved by an affine map, so all
//! three compute one corner vertex.
//!
//! Every crossed edge the chunk OWNS gives one quad joining the four vertices around it, facing from
//! rock to air, cut into two triangles along the SHORTER diagonal in integer cell space (S6-3, the tie
//! taking the same geometric diagonal whichever way the quad faces). Ownership: an edge between two
//! cells of one face belongs to the chunk whose own cell is the lower end; an edge across a face seam
//! belongs to the face with the lower index; an edge that touches a phantom, or joins two halo cells,
//! belongs to nobody here — the three seam edges and three radial edges that meet at a corner are
//! owned by the faces they lie on, and their quads close the corner on the prism vertex.
//!
//! **Example.** On the home planet a chunk of rolling hills gives about 5 600 vertices and 10 800
//! triangles. The chunk beside it computes the vertices along their shared edge from the same eight
//! cells to the same 1/256-cell integers, so the two meshes meet without a crack and without a
//! duplicated face.

// ★ THE EXTRACTOR IS THE ONE NAMED EXCEPTION to "no `/` and no `%` on the recipe's path" (ruling F7;
// the rule lives in `lib.rs`). Its arithmetic is already pure integer, but it divides: a group's vertex
// is the AVERAGE of its crossings, and the average's denominator is the crossing count times the
// crossings' own denominators — numbers the cell bytes decide, which no shift divides. Every such
// divide is EXACT on every host (whole numbers, one rounding, `round_half_even`), and every remainder
// is `rem_euclid` or an index wrap over three axes. The GPU's step G2 ports this file; the port is
// where those divides become the prefix sum's own arithmetic, and this comment is the ledger entry
// until then.
use crate::chunk::{CHUNK_EDGE, ChunkKey};
use crate::lattice::{CORNER_FACE, HALO, SampleBox, Site};

/// Vertex steps per cell: the vertex quantum.
pub const VERTEX_QUANTUM: i32 = 256;
/// Groups per box axis: the `2 × 2 × 2` groups start at local `−1..=61`.
pub const GROUPS_PER_AXIS: usize = CHUNK_EDGE + 1;
/// The most edges a group crosses.
const EDGES_PER_GROUP: usize = 12;
/// The most fractional crossings one axis can carry (four parallel edges of a cube; a prism's
/// two ring pairs plus its two diagonals).
const FRACS_PER_AXIS: usize = 4;

/// The twelve edges of a group as corner pairs `(p, q)` with `q = p | axis bit` and the axis.
const GROUP_EDGES: [(u8, u8, usize); EDGES_PER_GROUP] = [
    (0, 1, 0),
    (2, 3, 0),
    (4, 5, 0),
    (6, 7, 0),
    (0, 2, 1),
    (1, 3, 1),
    (4, 6, 1),
    (5, 7, 1),
    (0, 4, 2),
    (1, 5, 2),
    (2, 6, 2),
    (3, 7, 2),
];

/// A chunk's surface: integer vertices in the chunk's own cell space (1/256 cell, local cell `−1`
/// at `−256`) and triangles as vertex indices, wound rock-to-air.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkMesh {
    pub key: ChunkKey,
    pub vertices: Vec<[i16; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// Whether a gap byte is rock (air is `>= 0`, so a cell exactly on the surface is air).
#[must_use]
pub const fn is_rock(gap: i8) -> bool {
    gap < 0
}

/// `num / den` rounded to the nearest integer, HALF TO EVEN; `den > 0`.
#[must_use]
pub fn round_half_even(num: i64, den: i64) -> i64 {
    let twice = 2 * den;
    let up = (2 * num + den).div_euclid(twice);
    let exact_half = (2 * num + den).rem_euclid(twice) == 0;
    if exact_half & (up.rem_euclid(2) == 1) {
        up - 1
    } else {
        up
    }
}

/// The exact average of crossings, per axis: whole parts summed, fractions kept as
/// `(numerator, denominator)`, at most four per axis.
struct Average {
    whole: [i64; 3],
    fracs: [[(i64, i64); FRACS_PER_AXIS]; 3],
    frac_count: [usize; 3],
    count: i64,
}

impl Average {
    const fn new() -> Average {
        Average {
            whole: [0; 3],
            fracs: [[(0, 1); FRACS_PER_AXIS]; 3],
            frac_count: [0; 3],
            count: 0,
        }
    }

    /// Add a crossing: per axis a whole part and a fraction `num / den` (`num = 0` for an axis the
    /// crossing does not move along).
    fn add(&mut self, whole: [i64; 3], frac: [(i64, i64); 3]) {
        let mut k = 0;
        while k < 3 {
            self.whole[k] += whole[k];
            if frac[k].0 != 0 {
                self.fracs[k][self.frac_count[k]] = frac[k];
                self.frac_count[k] += 1;
            }
            k += 1;
        }
        self.count += 1;
    }

    /// The average in quanta, per axis, rounded half to even; `None` with no crossing.
    fn quantised(&self) -> Option<[i16; 3]> {
        if self.count == 0 {
            return None;
        }
        let mut v = [0i16; 3];
        let mut k = 0;
        while k < 3 {
            let mut d = 1i64;
            let mut i = 0;
            while i < self.frac_count[k] {
                d *= self.fracs[k][i].1;
                i += 1;
            }
            let mut n = self.whole[k] * d;
            let mut i = 0;
            while i < self.frac_count[k] {
                #[allow(
                    clippy::integer_division,
                    reason = "THE EXTRACTOR (the GPU's step G2, not yet ported): the common \
                              denominator divides by each crossing's own, exactly, on the CPU"
                )]
                let share = d / self.fracs[k][i].1;
                n += self.fracs[k][i].0 * share;
                i += 1;
            }
            // The average in cells is `n / (count·d)`; in quanta, `n·256 / (count·d)`.
            let q = round_half_even(n * i64::from(VERTEX_QUANTUM), self.count * d);
            v[k] = i16::try_from(q).expect("a vertex lies inside the box");
            k += 1;
        }
        Some(v)
    }
}

/// The exact vertex of a CUBE group whose eight gaps are `g[(dc·2 + db)·2 + da]`, at group origin
/// `(ga, gb, gc)`; `None` when no edge crosses.
#[must_use]
pub fn group_vertex(g: [i8; 8], origin: [i32; 3]) -> Option<[i16; 3]> {
    let mut avg = Average::new();
    let mut e = 0;
    while e < EDGES_PER_GROUP {
        let (p, q, axis) = GROUP_EDGES[e];
        let (dp, dq) = (g[p as usize], g[q as usize]);
        if is_rock(dp) != is_rock(dq) {
            let (mp, mq) = (i64::from(dp).abs(), i64::from(dq).abs());
            let mut whole = [0i64; 3];
            let mut k = 0;
            while k < 3 {
                whole[k] = i64::from(origin[k]) + i64::from((p >> k) & 1);
                k += 1;
            }
            // Along the edge's axis the crossing sits `mp / (mp + mq)` past the `p` end.
            let mut frac = [(0i64, 1i64); 3];
            frac[axis] = (mp, mp + mq);
            avg.add(whole, frac);
        }
        e += 1;
    }
    avg.quantised()
}

/// The crossing on an edge from a cell of gap `d0` to one of gap `d1`, as `(numerator,
/// denominator)` of the fraction past the `d0` end; `None` when both are rock or both air.
fn crossing(d0: i8, d1: i8) -> Option<(i64, i64)> {
    if is_rock(d0) == is_rock(d1) {
        return None;
    }
    let (m0, m1) = (i64::from(d0).abs(), i64::from(d1).abs());
    Some((m0, m0 + m1))
}

/// The exact vertex of a CORNER PRISM group: the three real columns `r[0]` (opposite the phantom),
/// `r[1]` (beside it along `a`) and `r[2]` (beside it along `b`), each with its lower and upper
/// gap, the phantom at local corner `(pa, pb)` of the group at `origin`, and the three columns'
/// global sites. Nine edges: three radials, and per layer the two ring edges to `r[0]` and the
/// diagonal `r[1]–r[2]`, which is a real adjacency across the third seam.
///
/// The vertex is placed in BARYCENTRIC terms — a weight per column, exact — and the three weights
/// are quantised to 256 together (floors, then the leftover quanta to the largest remainders, a tie
/// going to the smaller global site), so the three faces that meet at the corner, each with its own
/// local layout of the same three columns, quantise to ONE point.
#[must_use]
pub fn prism_vertex(
    r: [[i8; 2]; 3],
    phantom: (i32, i32),
    origin: [i32; 3],
    sites: [Site; 3],
) -> Option<[i16; 3]> {
    let (pa, pb) = phantom;
    // Local positions of the three columns inside the group.
    let pos = [[1 - pa, 1 - pb], [pa, 1 - pb], [1 - pa, pb]];
    // Per column, the sum of its barycentric weight over the crossings, as (num, den) fractions
    // (at most four fractional contributions per column), and the radial average.
    let mut whole = [0i64; 3];
    let mut fracs = [[(0i64, 1i64); FRACS_PER_AXIS]; 3];
    let mut frac_count = [0usize; 3];
    let mut radial = Average::new();
    let mut count = 0i64;
    let mut add_frac = |col: usize, f: (i64, i64)| {
        fracs[col][frac_count[col]] = f;
        frac_count[col] += 1;
    };
    // The radial edge of each column: weight one on that column, the crossing along c.
    let mut i = 0;
    while i < 3 {
        if let Some(f) = crossing(r[i][0], r[i][1]) {
            whole[i] += 1;
            radial.add([0, 0, i64::from(origin[2])], [(0, 1), (0, 1), f]);
            count += 1;
        }
        i += 1;
    }
    // The ring edges of each layer: a crossing `n / d` from column i toward column j weighs
    // `(d − n) / d` on i and `n / d` on j.
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    let mut layer = 0;
    while layer < 2 {
        let c = i64::from(origin[2]) + layer as i64;
        for (i, j) in pairs {
            if let Some((n, d)) = crossing(r[i][layer], r[j][layer]) {
                add_frac(i, (d - n, d));
                add_frac(j, (n, d));
                radial.add([0, 0, c], [(0, 1), (0, 1), (0, 1)]);
                count += 1;
            }
        }
        layer += 1;
    }
    if count == 0 {
        return None;
    }
    // Each column's weight `W_i = (whole_i + Σ fracs_i) / count`, as one fraction `N_i / D_i`.
    let mut num = [0i64; 3];
    let mut den = [1i64; 3];
    let mut i = 0;
    while i < 3 {
        let mut d = 1i64;
        let mut k = 0;
        while k < frac_count[i] {
            d *= fracs[i][k].1;
            k += 1;
        }
        let mut n = whole[i] * d;
        let mut k = 0;
        while k < frac_count[i] {
            #[allow(
                clippy::integer_division,
                reason = "THE EXTRACTOR (the GPU's step G2, not yet ported): the common denominator \
                          divides by each crossing's own, exactly, on the CPU"
            )]
            let share = d / fracs[i][k].1;
            n += fracs[i][k].0 * share;
            k += 1;
        }
        num[i] = n;
        den[i] = d * count;
        i += 1;
    }
    // Quantise the three weights to a common 256: floors first, then the leftover quanta to the
    // largest remainders, a tie to the smaller site. The remainders are compared as exact
    // fractions by cross-multiplication.
    let mut q = [0i64; 3];
    let mut rem = [(0i64, 1i64); 3];
    let mut i = 0;
    while i < 3 {
        let scaled = num[i] * i64::from(VERTEX_QUANTUM);
        q[i] = scaled.div_euclid(den[i]);
        rem[i] = (scaled.rem_euclid(den[i]), den[i]);
        i += 1;
    }
    let mut left = i64::from(VERTEX_QUANTUM) - q[0] - q[1] - q[2];
    while left > 0 {
        // The column with the largest remainder not yet raised; ties by site.
        let mut best = 0;
        let mut i = 1;
        while i < 3 {
            let bigger = rem[i].0 * rem[best].1 > rem[best].0 * rem[i].1;
            let tie = rem[i].0 * rem[best].1 == rem[best].0 * rem[i].1;
            if bigger | (tie & (sites[i] < sites[best])) {
                best = i;
            }
            i += 1;
        }
        q[best] += 1;
        rem[best] = (-1, 1);
        left -= 1;
    }
    // The local a and b of the vertex are the weighted positions of the columns; c is the radial
    // average, rounded half to even like every other vertex.
    let a = q[0] * i64::from(pos[0][0]) + q[1] * i64::from(pos[1][0]) + q[2] * i64::from(pos[2][0]);
    let b = q[0] * i64::from(pos[0][1]) + q[1] * i64::from(pos[1][1]) + q[2] * i64::from(pos[2][1]);
    let c = radial.quantised().expect("at least one crossing")[2];
    Some([
        i16::try_from(a + i64::from(origin[0]) * i64::from(VERTEX_QUANTUM))
            .expect("inside the box"),
        i16::try_from(b + i64::from(origin[1]) * i64::from(VERTEX_QUANTUM))
            .expect("inside the box"),
        c,
    ])
}

/// The gaps of the group at `origin`, in `(dc·2 + db)·2 + da` order.
fn group_gaps(samples: &SampleBox, origin: [i32; 3]) -> [i8; 8] {
    let mut g = [0i8; 8];
    let mut corner = 0;
    while corner < 8 {
        g[corner] = samples
            .cell(
                origin[0] + (corner & 1) as i32,
                origin[1] + ((corner >> 1) & 1) as i32,
                origin[2] + (corner >> 2) as i32,
            )
            .gap;
        corner += 1;
    }
    g
}

/// The local corner `(pa, pb)` of the group at `origin` whose column is a phantom, if any.
#[must_use]
pub fn phantom_corner(samples: &SampleBox, origin: [i32; 3]) -> Option<(i32, i32)> {
    let mut corner = 0;
    while corner < 4 {
        let (pa, pb) = (corner & 1, corner >> 1);
        if samples.site(origin[0] + pa, origin[1] + pb).face == CORNER_FACE {
            return Some((pa, pb));
        }
        corner += 1;
    }
    None
}

/// The vertex of the group at `origin`: a prism at a cube corner, a cube everywhere else.
#[must_use]
pub fn vertex_of_group(samples: &SampleBox, origin: [i32; 3]) -> Option<[i16; 3]> {
    match phantom_corner(samples, origin) {
        Some((pa, pb)) => {
            let pos = [[1 - pa, 1 - pb], [pa, 1 - pb], [1 - pa, pb]];
            let mut r = [[0i8; 2]; 3];
            let mut i = 0;
            while i < 3 {
                r[i][0] = samples
                    .cell(origin[0] + pos[i][0], origin[1] + pos[i][1], origin[2])
                    .gap;
                r[i][1] = samples
                    .cell(origin[0] + pos[i][0], origin[1] + pos[i][1], origin[2] + 1)
                    .gap;
                i += 1;
            }
            let sites = [
                samples.site(origin[0] + pos[0][0], origin[1] + pos[0][1]),
                samples.site(origin[0] + pos[1][0], origin[1] + pos[1][1]),
                samples.site(origin[0] + pos[2][0], origin[1] + pos[2][1]),
            ];
            prism_vertex(r, (pa, pb), origin, sites)
        }
        None => group_vertex(group_gaps(samples, origin), origin),
    }
}

/// Whether the chunk OWNS a cell: one of its own `62³` that lies on its own face (a partial chunk's
/// cells beyond the face are the partner's, and the partner's chunk owns them).
fn owns_cell(samples: &SampleBox, p: [i32; 3]) -> bool {
    SampleBox::is_core(p[0], p[1], p[2])
        & (samples.site(p[0], p[1]).face == samples.key.face.index())
}

/// Whether the chunk owns the edge from cell `p` to cell `q` (`q = p + axis`): on one face, the
/// chunk owning the lower end; across a seam, the lower face index; an edge that touches a phantom
/// (face 255) is nobody's, because no chunk's own cell is edge-adjacent to a phantom.
fn owned(samples: &SampleBox, p: [i32; 3], q: [i32; 3]) -> bool {
    let fp = samples.site(p[0], p[1]).face;
    let fq = samples.site(q[0], q[1]).face;
    if fp == fq {
        return owns_cell(samples, p);
    }
    let mine = samples.key.face.index();
    let other = if fp == mine { fq } else { fp };
    (owns_cell(samples, p) | owns_cell(samples, q)) & (mine < other)
}

/// Whether the four groups around the edge from `p` along `axis` all exist in the box: the
/// groups run `−1..=61`, and the ring takes `p` and `p − 1` across the edge, `p` along it.
fn ring_fits(p: [i32; 3], axis: usize) -> bool {
    let edge = CHUNK_EDGE as i32;
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    (p[axis] < edge) & (p[u] >= 0) & (p[u] < edge) & (p[w] >= 0) & (p[w] < edge)
}

/// The index of the group at `origin` in the vertex map.
fn group_index(origin: [i32; 3]) -> usize {
    let g = GROUPS_PER_AXIS;
    (((origin[2] + HALO) as usize) * g + (origin[1] + HALO) as usize) * g
        + (origin[0] + HALO) as usize
}

/// The squared length of a diagonal in quanta.
fn len2(a: [i16; 3], b: [i16; 3]) -> i64 {
    let mut s = 0i64;
    let mut k = 0;
    while k < 3 {
        let d = i64::from(a[k]) - i64::from(b[k]);
        s += d * d;
        k += 1;
    }
    s
}

/// Extract the chunk's surface from its sample box: the quads of the edges the chunk OWNS.
#[must_use]
pub fn extract(samples: &SampleBox) -> ChunkMesh {
    extract_with(samples, false)
}

/// Extract the surface of EVERY crossed edge in the box, the chunk's own and its neighbours'
/// alike (the halo's edges included): the surface as a whole around this chunk, with nothing
/// given to a neighbour. What the client's geomorph reads a coarser rung's mesh from (slice 8
/// step 3): a finer chunk's halo vertices stand over the coarser chunk's halo, whose crossings
/// the coarser chunk does not own, and two finer neighbours must read one and the same coarser
/// surface for the vertex they share. Never a drawn mesh: its quads double a neighbour's.
#[must_use]
pub fn extract_all_edges(samples: &SampleBox) -> ChunkMesh {
    extract_with(samples, true)
}

fn extract_with(samples: &SampleBox, every_edge: bool) -> ChunkMesh {
    let edge = CHUNK_EDGE as i32;
    let mut vertex_of = vec![u32::MAX; GROUPS_PER_AXIS * GROUPS_PER_AXIS * GROUPS_PER_AXIS];
    let mut vertices: Vec<[i16; 3]> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    // The vertex of a group, made on first use, in first-use order.
    let vertex = |origin: [i32; 3], vertex_of: &mut Vec<u32>, vertices: &mut Vec<[i16; 3]>| {
        let gi = group_index(origin);
        if vertex_of[gi] == u32::MAX {
            let v =
                vertex_of_group(samples, origin).expect("a group around a crossed edge crosses");
            vertex_of[gi] = u32::try_from(vertices.len()).expect("fewer than 2^32 vertices");
            vertices.push(v);
        }
        vertex_of[gi]
    };
    let mut c = -HALO;
    while c <= edge {
        let mut b = -HALO;
        while b <= edge {
            let mut a = -HALO;
            while a <= edge {
                let p = [a, b, c];
                let gp = samples.cell(a, b, c).gap;
                let mut axis = 0;
                while axis < 3 {
                    let mut q = p;
                    q[axis] += 1;
                    if q[axis] <= edge {
                        let gq = samples.cell(q[0], q[1], q[2]).gap;
                        if (is_rock(gp) != is_rock(gq))
                            && (owned(samples, p, q) || (every_edge && ring_fits(p, axis)))
                        {
                            // The four groups around the edge, in the cycle that faces +axis.
                            let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
                            let mut origins = [p; 4];
                            let cycle = [(1, 1), (0, 1), (0, 0), (1, 0)];
                            let mut i = 0;
                            while i < 4 {
                                origins[i][u] -= cycle[i].0;
                                origins[i][w] -= cycle[i].1;
                                i += 1;
                            }
                            let mut quad = [0u32; 4];
                            let mut i = 0;
                            while i < 4 {
                                quad[i] = vertex(origins[i], &mut vertex_of, &mut vertices);
                                i += 1;
                            }
                            // Rock at `q` means the surface faces −axis: the cycle is reversed.
                            push_quad(&vertices, quad, is_rock(gq), &mut triangles);
                        }
                    }
                    axis += 1;
                }
                a += 1;
            }
            b += 1;
        }
        c += 1;
    }
    ChunkMesh {
        key: samples.key,
        vertices,
        triangles,
    }
}

/// Whether a triangle has zero area (three collinear or coincident vertices) — which surface nets
/// produce where a cell sits exactly on the surface (gap 0) or two groups quantise together. A
/// zero-area triangle is dropped: it draws nothing and a collider refuses it, and the other half
/// of its quad still covers the quad.
#[must_use]
pub fn is_degenerate(vertices: &[[i16; 3]], t: [u32; 3]) -> bool {
    let p = |i: u32| vertices[i as usize].map(i64::from);
    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    (n[0] == 0) & (n[1] == 0) & (n[2] == 0)
}

/// Cut a quad along its shorter diagonal into two triangles, reversed when `flip` (the surface
/// faces the other way). The diagonal is chosen on the quad as given, the tie taking `0–2`, BEFORE
/// the flip, so the same geometric diagonal is cut whichever way the quad faces. A zero-area
/// triangle is not emitted.
pub fn push_quad(vertices: &[[i16; 3]], q: [u32; 4], flip: bool, triangles: &mut Vec<[u32; 3]>) {
    let at = |i: u32| vertices[i as usize];
    let d02 = len2(at(q[0]), at(q[2]));
    let d13 = len2(at(q[1]), at(q[3]));
    let (t0, t1) = if d02 <= d13 {
        ([q[0], q[1], q[2]], [q[0], q[2], q[3]])
    } else {
        ([q[1], q[2], q[3]], [q[1], q[3], q[0]])
    };
    for t in [t0, t1] {
        if !is_degenerate(vertices, t) {
            if flip {
                triangles.push([t[0], t[2], t[1]]);
            } else {
                triangles.push(t);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement): a test
    //! states the exact quotient a reciprocal stands for, and a fixture picks its sample columns with a
    //! remainder. Neither runs in a kernel.
    #![allow(
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or picks a sample column; never a kernel's path"
    )]

    use super::*;
    use crate::body::BodyDefinition;
    use crate::chunk::Cell;
    use crate::digest::surface_chunk_z;
    use crate::home::home_planet;
    use crate::lattice::{BOX_CELLS, BOX_EDGE, Site, sample_box};
    use crate::position::vertex_position_m;
    use crate::strata::Stratum;
    use std::collections::{BTreeMap, BTreeSet};
    use vd_recipe::Gi;
    use vd_seed::bend::Face;

    /// A synthetic box (a BOUND, never a world): every cell's gap from a rule.
    pub(crate) fn synthetic(mut gap: impl FnMut(i32, i32, i32) -> i8) -> SampleBox {
        let key = ChunkKey {
            face: Face::PosX,
            rung: 0,
            x: 0,
            y: 0,
            z: 0,
        };
        let mut cells = Vec::with_capacity(BOX_CELLS);
        let mut c = -HALO;
        while c <= CHUNK_EDGE as i32 {
            let mut b = -HALO;
            while b <= CHUNK_EDGE as i32 {
                let mut a = -HALO;
                while a <= CHUNK_EDGE as i32 {
                    let g = gap(a, b, c);
                    cells.push(Cell {
                        stratum: if is_rock(g) {
                            Stratum::Granite
                        } else {
                            Stratum::Air
                        },
                        gap: g,
                    });
                    a += 1;
                }
                b += 1;
            }
            c += 1;
        }
        let sites = vec![
            Site {
                face: 0,
                i: 0,
                j: 0
            };
            BOX_EDGE * BOX_EDGE
        ];
        let dirs = vec![[vd_recipe::bend::DIR_ONE, Gi::ZERO, Gi::ZERO]; BOX_EDGE * BOX_EDGE];
        SampleBox {
            key,
            cells,
            sites,
            dirs,
        }
    }

    /// A triangle's normal component along `axis` (twice the signed area's projection).
    fn normal_component(mesh: &ChunkMesh, t: [u32; 3], axis: usize) -> i64 {
        let p = |i: u32| mesh.vertices[i as usize].map(i64::from);
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let (x, y) = ((axis + 1) % 3, (axis + 2) % 3);
        u[x] * w[y] - u[y] * w[x]
    }

    /// The vertices of a mesh in metres, as bit patterns.
    fn metres(body: &BodyDefinition, sb: &SampleBox, mesh: &ChunkMesh) -> Vec<[u64; 3]> {
        mesh.vertices
            .iter()
            .map(|v| {
                let p = vertex_position_m(body, sb, *v);
                [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
            })
            .collect()
    }

    /// A mesh's triangles as sorted triples of vertex metres: the same triangle from two chunks is
    /// the same triple.
    fn triangles_in_metres(
        body: &BodyDefinition,
        sb: &SampleBox,
        mesh: &ChunkMesh,
    ) -> BTreeSet<[[u64; 3]; 3]> {
        let m = metres(body, sb, mesh);
        mesh.triangles
            .iter()
            .map(|t| {
                let mut tri = [m[t[0] as usize], m[t[1] as usize], m[t[2] as usize]];
                tri.sort();
                tri
            })
            .collect()
    }

    #[test]
    fn every_edge_holds_the_owned_surface_and_the_neighbours_share_of_it() {
        let m = home_planet();
        let key = ChunkKey {
            face: Face::PosX,
            rung: 0,
            x: 300,
            y: 700,
            z: surface_chunk_z(&m, Face::PosX, 0, 300, 700),
        };
        let samples = sample_box(&m, key).expect("in the band");
        let owned = extract(&samples);
        let all = extract_all_edges(&samples);
        // The whole holds more than the owned part: the halo's crossings are a neighbour's.
        assert!(all.triangles.len() > owned.triangles.len());
        assert!(all.vertices.len() >= owned.vertices.len());
        // Every owned vertex is a vertex of the whole, at the same quanta (the same groups).
        let whole: std::collections::BTreeSet<[i16; 3]> = all.vertices.iter().copied().collect();
        for v in &owned.vertices {
            assert!(whole.contains(v), "{v:?} owned but not in the whole");
        }
        // Every owned triangle is a triangle of the whole, by its vertices' quanta.
        let tri = |mesh: &ChunkMesh, t: [u32; 3]| -> [[i16; 3]; 3] {
            let mut c = [
                mesh.vertices[t[0] as usize],
                mesh.vertices[t[1] as usize],
                mesh.vertices[t[2] as usize],
            ];
            c.sort_unstable();
            c
        };
        let whole_tris: std::collections::BTreeSet<[[i16; 3]; 3]> =
            all.triangles.iter().map(|t| tri(&all, *t)).collect();
        for t in &owned.triangles {
            assert!(whole_tris.contains(&tri(&owned, *t)));
        }
    }

    #[test]
    fn rounding_is_half_to_even_and_symmetric_under_a_flipped_axis() {
        assert_eq!(round_half_even(7, 2), 4, "3.5 → 4");
        assert_eq!(round_half_even(5, 2), 2, "2.5 → 2");
        assert_eq!(round_half_even(-5, 2), -2, "−2.5 → −2");
        assert_eq!(round_half_even(-7, 2), -4, "−3.5 → −4");
        assert_eq!(round_half_even(85, 1), 85);
        assert_eq!(round_half_even(256, 3), 85, "85.33 → 85");
        assert_eq!(round_half_even(512, 3), 171, "170.67 → 171");
        // The seam property: a fraction F seen as 1 − F from the other side rounds to quanta that
        // sum to exactly 256, for every F in 1/512 steps.
        let mut n = 0;
        while n <= 512 {
            let here = round_half_even(n * 256, 512);
            let there = round_half_even((512 - n) * 256, 512);
            assert_eq!(here + there, 256, "F = {n}/512");
            n += 1;
        }
    }

    #[test]
    fn a_group_vertex_is_the_exact_average_of_its_crossings() {
        // Rock below (c = 0), air above (c = 1), the surface a quarter cell above the rock centres:
        // gaps −0.75 cell below (−96) and +0.25 cell above (+32); t = 96 / 128 = 0.75 from the
        // rock end, so the vertex sits at c = 0.75 → 192 quanta, centred in a and b (+128).
        let g = [-96, -96, -96, -96, 32, 32, 32, 32];
        assert_eq!(group_vertex(g, [0, 0, 0]), Some([128, 128, 192]));
        assert_eq!(
            group_vertex(g, [-1, 3, 5]),
            Some([-128, 896, 192 + 5 * 256])
        );
        // All rock, or all air (zero is air): no vertex.
        assert_eq!(group_vertex([-1; 8], [0, 0, 0]), None);
        assert_eq!(group_vertex([0; 8], [0, 0, 0]), None);
        // A cell exactly on the surface (gap 0) is air, and the crossing sits on it: t = |−50| /
        // (50 + 0) = 1, at the air end.
        let g = [-50, -50, -50, -50, 0, 0, 0, 0];
        assert_eq!(group_vertex(g, [0, 0, 0]), Some([128, 128, 256]));
        // 1/3 cell → 85.33 → 85 (t = 1/3 with gaps −1 and +2).
        let g = [-1, -1, -1, -1, 2, 2, 2, 2];
        assert_eq!(group_vertex(g, [0, 0, 0]), Some([128, 128, 85]));
        // One air corner (corner 1) among rock: three crossings, halfway along the three edges
        // that leave it — (0.5, 0, 0), (1, 0.5, 0), (1, 0, 0.5) — averaged to (5/6, 1/6, 1/6):
        // 213.33 → 213, 42.67 → 43.
        let g = [-1, 1, -1, -1, -1, -1, -1, -1];
        assert_eq!(group_vertex(g, [0, 0, 0]), Some([213, 43, 43]));
    }

    #[test]
    fn a_prism_vertex_averages_the_nine_edges_of_a_corner_group() {
        // Rock below, air above, in all three columns: three radial crossings at t = 0.5 → the
        // vertex at the columns' centroid. With the phantom at (0, 0) the columns sit at (1,1),
        // (0,1), (1,0): a = 2/3, b = 2/3, c = 0.5.
        // The three weights are a third each: 85, 85, 85 leave one quantum, which goes to the
        // smallest site — r[0] here — so r[0] weighs 86.
        let sites = [
            Site {
                face: 0,
                i: 0,
                j: 0,
            },
            Site {
                face: 1,
                i: 0,
                j: 0,
            },
            Site {
                face: 2,
                i: 0,
                j: 0,
            },
        ];
        let r = [[-64, 64]; 3];
        assert_eq!(
            prism_vertex(r, (0, 0), [0, 0, 0], sites),
            Some([171, 171, 128])
        );
        // The phantom at (1, 1): columns (0,0), (1,0), (0,1): a = w1 = 85, b = w2 = 85.
        assert_eq!(
            prism_vertex(r, (1, 1), [0, 0, 0], sites),
            Some([85, 85, 128])
        );
        // The leftover quantum follows the SITE, not the local role: with r[2] the smallest site,
        // r[2] weighs 86 and the vertex's b moves by one quantum.
        let sites_z = [
            Site {
                face: 2,
                i: 0,
                j: 0,
            },
            Site {
                face: 1,
                i: 0,
                j: 0,
            },
            Site {
                face: 0,
                i: 0,
                j: 0,
            },
        ];
        assert_eq!(
            prism_vertex(r, (1, 1), [0, 0, 0], sites_z),
            Some([85, 86, 128])
        );
        // No crossing at all.
        assert_eq!(prism_vertex([[-1, -1]; 3], (1, 1), [0, 0, 0], sites), None);
        // r[0] and r[1] rock, r[2] air, both layers. Phantom at (1, 1): r[0] = (0,0), r[1] = (1,0),
        // r[2] = (0,1). Crossings: r[0]–r[2] along b at t = 0.5 → (0, 0.5, c); r[1]–r[2] on the
        // diagonal at t = 0.5 → (0.5, 0.5, c); both layers: four points → (0.25, 0.5, 0.5).
        let r = [[-1, -1], [-1, -1], [1, 1]];
        assert_eq!(
            prism_vertex(r, (1, 1), [0, 0, 0], sites),
            Some([64, 128, 128])
        );
        // The same columns with the phantom at (0, 0): r[0] = (1,1), r[1] = (0,1), r[2] = (1,0):
        // r[0]–r[2] from (1,1) toward (1,0) at 0.5 → (1, 0.5); the diagonal from (0,1) toward
        // (1,0) at 0.5 → (0.5, 0.5). Average: (0.75, 0.5, 0.5).
        assert_eq!(
            prism_vertex(r, (0, 0), [0, 0, 0], sites),
            Some([192, 128, 128])
        );
        // Phantom at (0, 1): r[0] = (1,0), r[1] = (0,0), r[2] = (1,1). r[0] rock −3, r[1] air +1,
        // r[2] rock −1: r[0]–r[1] along a at t = 3/4 from r[0] (a = 1) toward a = 0 → a = 0.25;
        // r[1]–r[2] on the diagonal at t = 1/2 from r[1] (0,0) toward (1,1) → (0.5, 0.5). Both
        // layers: four points → a = 0.375 (96), b = 0.25 (64), c = 0.5.
        let r = [[-3, -3], [1, 1], [-1, -1]];
        assert_eq!(
            prism_vertex(r, (0, 1), [0, 0, 0], sites),
            Some([96, 64, 128])
        );
        // Weights, for the record: r[0] gets 1/4 per layer from the r[0]–r[1] crossing (the far
        // end weighs 3/4 = 0.75 on r[1]... from r[0] toward r[1] at t = 3/4: r[0] weighs 1/4),
        // r[1] gets 3/4 + 1/2, r[2] gets 1/2, over four crossings: (1/8, 5/8, 1/4) → 32, 160, 64.
        // Phantom at (0, 1): r[0] at (1, 0), r[1] at (0, 0), r[2] at (1, 1): a = w0 + w2 = 96,
        // b = w2 = 64. The same numbers as above, by the same rule.
    }

    #[test]
    fn the_diagonal_is_the_shorter_one_and_a_tie_takes_the_same_geometric_diagonal_when_flipped() {
        // A square: both diagonals equal → the tie takes 0–2.
        let square = [[0, 0, 0], [256, 0, 0], [256, 256, 0], [0, 256, 0]];
        let mut t = Vec::new();
        push_quad(&square, [0, 1, 2, 3], false, &mut t);
        assert_eq!(t, vec![[0, 1, 2], [0, 2, 3]]);
        // Flipped: the same diagonal 0–2, the winding reversed.
        let mut t = Vec::new();
        push_quad(&square, [0, 1, 2, 3], true, &mut t);
        assert_eq!(t, vec![[0, 2, 1], [0, 3, 2]]);
        // Vertex 2 lifted far: 0–2 is longer, so 1–3 is cut.
        let lifted = [[0, 0, 0], [256, 0, 0], [256, 256, 900], [0, 256, 0]];
        let mut t = Vec::new();
        push_quad(&lifted, [0, 1, 2, 3], false, &mut t);
        assert_eq!(t, vec![[1, 2, 3], [1, 3, 0]]);
        let mut t = Vec::new();
        push_quad(&lifted, [0, 1, 2, 3], true, &mut t);
        assert_eq!(t, vec![[1, 3, 2], [1, 0, 3]]);
        // Three collinear vertices: the zero-area half is dropped, the other half stays.
        let collinear = [[0, 0, 0], [128, 0, 0], [256, 0, 0], [0, 256, 0]];
        let mut t = Vec::new();
        push_quad(&collinear, [0, 1, 2, 3], false, &mut t);
        assert_eq!(t, vec![[0, 2, 3]]);
        assert!(is_degenerate(&collinear, [0, 1, 2]));
        assert!(!is_degenerate(&collinear, [0, 2, 3]));
        // Two coincident vertices.
        // Two coincident vertices: 1–3 is the shorter diagonal, and (1, 3, 0) has zero area.
        let coincident = [[0, 0, 0], [0, 0, 0], [256, 256, 0], [0, 256, 0]];
        let mut t = Vec::new();
        push_quad(&coincident, [0, 1, 2, 3], false, &mut t);
        assert_eq!(t, vec![[1, 2, 3]]);
    }

    #[test]
    fn a_flat_floor_gives_one_vertex_per_column_and_faces_up_and_walls_face_outward() {
        // The surface at c = 10.5: rock at c <= 10 (gap −0.5 at c = 10), air above (gap +0.5 at
        // c = 11).
        let sb = synthetic(|_, _, c| ((c - 10) * 128 - 64).clamp(-128, 127) as i8);
        let mesh = extract(&sb);
        // One vertex per group column that the chunk's quads touch: the 63 × 63 groups of −1..=61.
        assert_eq!(mesh.vertices.len(), GROUPS_PER_AXIS * GROUPS_PER_AXIS);
        // One quad per own column (62 × 62 radial edges): two triangles each.
        assert_eq!(mesh.triangles.len(), 2 * CHUNK_EDGE * CHUNK_EDGE);
        for v in &mesh.vertices {
            assert_eq!(v[2], 10 * 256 + 128, "the crossing sits at c = 10.5");
        }
        for t in &mesh.triangles {
            assert!(
                normal_component(&mesh, *t, 2) > 0,
                "a floor faces up: {t:?}"
            );
        }
        // Rock ABOVE air (a ceiling): every face points down.
        let sb = synthetic(|_, _, c| ((10 - c) * 128 - 64).clamp(-128, 127) as i8);
        let mesh = extract(&sb);
        assert_eq!(mesh.triangles.len(), 2 * CHUNK_EDGE * CHUNK_EDGE);
        for t in &mesh.triangles {
            assert!(normal_component(&mesh, *t, 2) < 0, "a ceiling faces down");
        }
        // A wall across a: rock at a <= 10, air beyond; the face points +a. And its mirror.
        let sb = synthetic(|a, _, _| ((a - 10) * 128 - 64).clamp(-128, 127) as i8);
        let mesh = extract(&sb);
        assert_eq!(mesh.triangles.len(), 2 * CHUNK_EDGE * CHUNK_EDGE);
        for t in &mesh.triangles {
            assert!(normal_component(&mesh, *t, 0) > 0, "a wall faces +a");
        }
        let sb = synthetic(|a, _, _| ((10 - a) * 128 - 64).clamp(-128, 127) as i8);
        let mesh = extract(&sb);
        for t in &mesh.triangles {
            assert!(normal_component(&mesh, *t, 0) < 0, "a wall faces −a");
        }
        // A wall across b.
        let sb = synthetic(|_, b, _| ((b - 10) * 128 - 64).clamp(-128, 127) as i8);
        let mesh = extract(&sb);
        assert_eq!(mesh.triangles.len(), 2 * CHUNK_EDGE * CHUNK_EDGE);
        for t in &mesh.triangles {
            assert!(normal_component(&mesh, *t, 1) > 0, "a wall faces +b");
        }
    }

    /// On the home planet: every index is in range, every vertex inside the box, no zero-area
    /// triangle, and every mesh edge (by vertex index) used once or twice — no quad emitted twice.
    fn well_formed(sb: &SampleBox, mesh: &ChunkMesh) {
        let edge = CHUNK_EDGE as i32;
        assert_eq!(mesh.key, sb.key);
        for v in &mesh.vertices {
            let mut k = 0;
            while k < 3 {
                assert!(i32::from(v[k]) >= -VERTEX_QUANTUM, "{v:?} inside the box");
                assert!(
                    i32::from(v[k]) <= edge * VERTEX_QUANTUM,
                    "{v:?} inside the box"
                );
                k += 1;
            }
        }
        let mut uses: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for t in &mesh.triangles {
            assert!(t.iter().all(|i| (*i as usize) < mesh.vertices.len()));
            assert!(
                !is_degenerate(&mesh.vertices, *t),
                "a zero-area triangle: {t:?}"
            );
            let mut k = 0;
            while k < 3 {
                let (p, q) = (t[k], t[(k + 1) % 3]);
                let e = if p <= q { (p, q) } else { (q, p) };
                *uses.entry(e).or_insert(0) += 1;
                k += 1;
            }
        }
        // Naive surface nets are not a manifold at an AMBIGUOUS face: where the four cells of the
        // face between two groups alternate rock and air, all four cell edges cross and up to four
        // quads share the two groups' vertices. The mesh is still closed there (every quad has its
        // four vertices), and no quad is emitted twice — which the triangle-set comparisons prove.
        // Dual contouring's manifold variant is the reserved upgrade (S6-1).
        for n in uses.values() {
            assert!(*n <= 4, "an edge used {n} times");
        }
    }

    /// The ambiguous face, measured: the 3-D checkerboard (a synthetic BOUND) crosses every cell
    /// edge, and an edge between two groups is shared by four quads.
    #[test]
    fn an_ambiguous_face_shares_an_edge_between_four_quads() {
        let sb = synthetic(|a, b, c| {
            if (a + b + c).rem_euclid(2) == 0 {
                -64
            } else {
                64
            }
        });
        let mesh = extract(&sb);
        well_formed(&sb, &mesh);
        let mut uses: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for t in &mesh.triangles {
            let mut k = 0;
            while k < 3 {
                let (p, q) = (t[k], t[(k + 1) % 3]);
                *uses
                    .entry(if p <= q { (p, q) } else { (q, p) })
                    .or_insert(0) += 1;
                k += 1;
            }
        }
        assert_eq!(*uses.values().max().expect("edges"), 4);
    }

    /// The vertex of every group in a slab of one box against the vertex of the same group in
    /// another box that holds those cells too (found through the sites), as metres: a TOTAL
    /// comparison of the shared geometry, independent of which chunk owns which edge.
    fn same_groups_same_vertices(
        body: &BodyDefinition,
        a: &SampleBox,
        b: &SampleBox,
        origins: impl Iterator<Item = [i32; 3]>,
    ) -> usize {
        use crate::lattice::local_of_site;
        let mut compared = 0;
        for oa in origins {
            let Some(va) = vertex_of_group(a, oa) else {
                continue;
            };
            // The group's four columns in b's box, by their sites; a phantom column is found by
            // its neighbours (b holds the same corner at its own local corner).
            let mut la = i32::MAX;
            let mut lb = i32::MAX;
            let mut corner = 0;
            while corner < 4 {
                let (da, db) = (corner & 1, corner >> 1);
                let site = a.site(oa[0] + da, oa[1] + db);
                let (x, y) = local_of_site(body, b.key, site).expect("b holds the group's columns");
                la = la.min(x);
                lb = lb.min(y);
                corner += 1;
            }
            let ob = [la, lb, oa[2]];
            let vb = vertex_of_group(b, ob).expect("the same cells cross in b");
            let pa = vertex_position_m(body, a, va);
            let pb = vertex_position_m(body, b, vb);
            assert_eq!(
                pa, pb,
                "group {oa:?} of {:?} is group {ob:?} of {:?}",
                a.key, b.key
            );
            compared += 1;
        }
        compared
    }

    #[test]
    fn a_home_planet_surface_is_well_formed_and_neighbours_share_no_triangle() {
        let m = home_planet();
        // Two neighbouring columns whose SURFACE lies in the same chunk along the radial, so the
        // shared face carries the surface's own crossings: the first such pair from column 40 on
        // (on the earth-like home planet the relief moves the surface a chunk between some
        // neighbours, which is the extractor's business, not this fixture's).
        let mut x0 = 40;
        while surface_chunk_z(&m, Face::NegZ, 0, x0, 41)
            != surface_chunk_z(&m, Face::NegZ, 0, x0 + 1, 41)
        {
            x0 += 1;
        }
        let z = surface_chunk_z(&m, Face::NegZ, 0, x0, 41);
        let key = |x: i32| ChunkKey {
            face: Face::NegZ,
            rung: 0,
            x,
            y: 41,
            z,
        };
        let a = sample_box(&m, key(x0)).expect("in the band");
        let b = sample_box(&m, key(x0 + 1)).expect("in the band");
        let ma = extract(&a);
        let mb = extract(&b);
        assert!(ma.triangles.len() > 1000);
        well_formed(&a, &ma);
        well_formed(&b, &mb);
        // Every group across the shared face gives the same vertex from both boxes.
        let edge = CHUNK_EDGE as i32;
        let slab = (-1..edge).flat_map(|c| (-1..edge).map(move |bb| [edge - 1, bb, c]));
        let compared = same_groups_same_vertices(&m, &a, &b, slab);
        assert!(
            compared > CHUNK_EDGE,
            "vertices along the shared face: {compared}"
        );
        // No triangle in both meshes (the ownership rule is a partition), and the two meshes share
        // vertices along their common edge.
        let ta = triangles_in_metres(&m, &a, &ma);
        let tb = triangles_in_metres(&m, &b, &mb);
        assert_eq!(
            ta.intersection(&tb).count(),
            0,
            "no triangle in both chunks"
        );
        let va: BTreeSet<[u64; 3]> = metres(&m, &a, &ma).into_iter().collect();
        let vb: BTreeSet<[u64; 3]> = metres(&m, &b, &mb).into_iter().collect();
        assert!(
            va.intersection(&vb).count() >= CHUNK_EDGE,
            "shared edge vertices"
        );
    }

    /// Across a face seam: every group that straddles the seam gives the same vertex in metres
    /// from the two faces' chunks, and no triangle is in both.
    #[test]
    fn across_a_seam_every_vertex_has_a_twin_and_the_meshes_close() {
        let m = home_planet();
        let rung = 2;
        let n_l = m.ladder().cells_per_edge(rung) as i32;
        let last = (n_l - 1) / CHUNK_EDGE as i32;
        let beyond = n_l - last * CHUNK_EDGE as i32;
        let z = surface_chunk_z(&m, Face::PosX, rung, last, 7);
        let kx = ChunkKey {
            face: Face::PosX,
            rung,
            x: last,
            y: 7,
            z,
        };
        let bx = sample_box(&m, kx).expect("in the band");
        let mx = extract(&bx);
        let site = bx.site(beyond, 0);
        assert_eq!(site.face, Face::PosY.index());
        let ky = ChunkKey {
            face: Face::PosY,
            rung,
            x: site.i / CHUNK_EDGE as i32,
            y: site.j / CHUNK_EDGE as i32,
            z,
        };
        let by = sample_box(&m, ky).expect("in the band");
        let my = extract(&by);
        well_formed(&bx, &mx);
        well_formed(&by, &my);
        let edge = CHUNK_EDGE as i32;
        let slab = (-1..edge).flat_map(|c| (0..edge - 1).map(move |bb| [beyond - 1, bb, c]));
        let compared = same_groups_same_vertices(&m, &bx, &by, slab);
        assert!(
            compared > 0,
            "the surface crosses the seam somewhere: {compared}"
        );
        let tx = triangles_in_metres(&m, &bx, &mx);
        let ty = triangles_in_metres(&m, &by, &my);
        assert_eq!(
            tx.intersection(&ty).count(),
            0,
            "no triangle in both chunks"
        );
    }

    /// At a cube corner: the three faces' chunks compute ONE corner vertex for every layer the
    /// surface crosses, every mesh is well formed, no triangle is in two meshes, and the corner
    /// vertex is a vertex of triangles from all three faces.
    #[test]
    fn at_a_cube_corner_the_three_meshes_close_on_one_prism_vertex() {
        let m = home_planet();
        let rung = 2;
        let n_l = m.ladder().cells_per_edge(rung) as i32;
        let last = (n_l - 1) / CHUNK_EDGE as i32;
        let beyond = n_l - last * CHUNK_EDGE as i32;
        let z = surface_chunk_z(&m, Face::PosX, rung, last, last);
        let key = |face: Face| ChunkKey {
            face,
            rung,
            x: last,
            y: last,
            z,
        };
        let boxes = [Face::PosX, Face::PosY, Face::PosZ]
            .map(|f| sample_box(&m, key(f)).expect("in the band"));
        let meshes = boxes.each_ref().map(extract);
        // The corner group of every box: the phantom at local (beyond, beyond), the prism at
        // origin (beyond − 1, beyond − 1, c). For every layer where the surface crosses it, the
        // three boxes place the corner vertex at the same metres.
        let mut corner_vertices: Vec<[u64; 3]> = Vec::new();
        let mut c = -1;
        while c < CHUNK_EDGE as i32 {
            let origin = [beyond - 1, beyond - 1, c];
            assert_eq!(phantom_corner(&boxes[0], origin), Some((1, 1)));
            let vs = boxes.each_ref().map(|sb| vertex_of_group(sb, origin));
            assert_eq!(vs[0].is_some(), vs[1].is_some());
            assert_eq!(vs[0].is_some(), vs[2].is_some());
            if let Some(v) = vs[0] {
                let p = vertex_position_m(&m, &boxes[0], v);
                let py = vertex_position_m(&m, &boxes[1], vs[1].expect("crosses"));
                let pz = vertex_position_m(&m, &boxes[2], vs[2].expect("crosses"));
                assert_eq!(p, py, "layer {c}: +X and +Y agree on the corner vertex");
                assert_eq!(p, pz, "layer {c}: +X and +Z agree on the corner vertex");
                corner_vertices.push([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]);
            }
            c += 1;
        }
        assert!(
            !corner_vertices.is_empty(),
            "the surface crosses the corner column"
        );
        // Every mesh is well formed, and the corner vertex is used by triangles of all three
        // faces — the fan around the corner has all three sides.
        let mut sides = 0;
        let mut all: Vec<BTreeSet<[[u64; 3]; 3]>> = Vec::new();
        for (sb, mesh) in boxes.iter().zip(meshes.iter()) {
            well_formed(sb, mesh);
            let tris = triangles_in_metres(&m, sb, mesh);
            let touching = tris
                .iter()
                .filter(|t| t.iter().any(|v| corner_vertices.contains(v)))
                .count();
            assert!(
                touching >= 2,
                "{:?} has triangles on the corner: {touching}",
                sb.key.face
            );
            sides += 1;
            all.push(tris);
        }
        assert_eq!(sides, 3);
        assert_eq!(all[0].intersection(&all[1]).count(), 0, "no triangle twice");
        assert_eq!(all[1].intersection(&all[2]).count(), 0, "no triangle twice");
        assert_eq!(all[0].intersection(&all[2]).count(), 0, "no triangle twice");
    }
}
