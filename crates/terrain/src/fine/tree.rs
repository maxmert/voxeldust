//! ★ THE FINE FLOW TREE (B1 (d); the bench's linear-time tree, B0 step 3): each point drains to its STEEPEST LOWER
//! neighbour (ties to the smaller index: each neighbour list runs in increasing index and the first of equal slopes is
//! kept) and the order is a walk up the donors from the outlets — on ground with no pit the same receiver the priority
//! flood picks, since every lower neighbour is reached first, in linear time. Every receiver stands strictly lower, so
//! every chain ends at an outlet and the walk reaches every point. A point with no lower neighbour (a pit: the first
//! pass over a smooth start) hands the whole tree to the PRIORITY FLOOD (Barnes, Lehman & Mulla 2014): from the outlets, each point drains to the steepest of its
//! neighbours already reached, on the flooded heights, so every pit spills over its pass. The flood's keys are the
//! heights in steps of 1/4 096 m, ties by the point's index — one order on every host.
//!
//! **Example.** On a 128 m tile of the home planet's belt the first pass starts from the smooth 8 km surface, whose
//! hollows the flood spills; every later pass reads heights computed along the last tree, which hold no pit, and takes
//! the linear-time path.

use super::delaunay::{Graph, NONE};
use crate::gf::Gf;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A flow tree: the order outlets first, each point's receiver ([`NONE`] at an outlet) and the length of its link.
#[derive(Clone, Debug, PartialEq)]
pub struct Tree {
    /// The order, outlets first: every point a walk or the flood reached (the flood leaves out a point no outlet
    /// reaches, which then keeps no receiver).
    pub order: Vec<u32>,
    pub recv: Vec<u32>,
    pub link: Vec<Gf>,
    /// Whether the flood decided it (a pit or an unreached point).
    pub flooded: bool,
}

/// ★ THE TREE over `graph` (link lengths `len` in `graph.adj`'s order) on heights `h`, the `outlet` points its roots.
#[must_use]
pub fn tree(h: &[Gf], outlet: &[bool], graph: &Graph, len: &[Gf]) -> Tree {
    let n = h.len();
    let (mut recv, mut link) = (vec![NONE; n], vec![Gf::ZERO; n]);
    for c in 0..n {
        if outlet[c] {
            continue;
        }
        let (a, b) = (graph.start[c] as usize, graph.start[c + 1] as usize);
        let (mut bs, mut bm, mut bl) = (Gf::ZERO, NONE, Gf::ZERO);
        for (&m, &l) in graph.adj[a..b].iter().zip(&len[a..b]) {
            if h[m as usize] < h[c] {
                let s = (h[c] - h[m as usize]) / l;
                if s > bs {
                    (bs, bm, bl) = (s, m, l);
                }
            }
        }
        if bm == NONE {
            return flood(h, outlet, graph, len);
        }
        recv[c] = bm;
        link[c] = bl;
    }
    let mut start = vec![0u32; n + 1];
    for &r in &recv {
        if r != NONE {
            start[r as usize + 1] += 1;
        }
    }
    for c in 0..n {
        start[c + 1] += start[c];
    }
    let mut fill = start.clone();
    let mut donors = vec![0u32; start[n] as usize];
    for (c, &r) in recv.iter().enumerate() {
        if r != NONE {
            donors[fill[r as usize] as usize] = c as u32;
            fill[r as usize] += 1;
        }
    }
    let mut order: Vec<u32> = (0..n as u32).filter(|&c| outlet[c as usize]).collect();
    let mut k = 0;
    while k < order.len() {
        let c = order[k] as usize;
        order.extend_from_slice(&donors[start[c] as usize..start[c + 1] as usize]);
        k += 1;
    }
    Tree { order, recv, link, flooded: false }
}

/// The priority flood (see the module's head).
fn flood(h: &[Gf], outlet: &[bool], graph: &Graph, len: &[Gf]) -> Tree {
    let n = h.len();
    let key = |z: Gf| (z * Gf::from_i64(4_096) + Gf::HALF).floor().to_i64_floor();
    let mut zf = h.to_vec();
    let (mut done, mut queued) = (vec![false; n], vec![false; n]);
    let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
    for c in 0..n {
        if outlet[c] {
            queued[c] = true;
            heap.push(Reverse((key(zf[c]), c as u32)));
        }
    }
    let (mut order, mut recv, mut link) = (Vec::with_capacity(n), vec![NONE; n], vec![Gf::ZERO; n]);
    let neg_inf = Gf::from_f64(f64::NEG_INFINITY);
    // Each point enters the queue once (`queued`), so each pop is a point not yet done.
    while let Some(Reverse((_, c))) = heap.pop() {
        let c = c as usize;
        done[c] = true;
        order.push(c as u32);
        let (a, b) = (graph.start[c] as usize, graph.start[c + 1] as usize);
        if !outlet[c] {
            let (mut bs, mut bm, mut bl) = (neg_inf, NONE, Gf::ZERO);
            for (&m, &l) in graph.adj[a..b].iter().zip(&len[a..b]) {
                if done[m as usize] {
                    let s = (zf[c] - zf[m as usize]) / l;
                    if s > bs {
                        (bs, bm, bl) = (s, m, l);
                    }
                }
            }
            recv[c] = bm;
            link[c] = bl;
        }
        for &m in &graph.adj[a..b] {
            let m = m as usize;
            if !queued[m] {
                queued[m] = true;
                zf[m] = zf[m].greater(zf[c]);
                heap.push(Reverse((key(zf[m]), m as u32)));
            }
        }
    }
    Tree { order, recv, link, flooded: true }
}

/// The water through each point: its own `water` and every donor's, summed down the tree (the order reversed).
#[must_use]
pub fn accumulate(t: &Tree, water: &[Gf]) -> Vec<Gf> {
    let mut q = water.to_vec();
    for &c in t.order.iter().rev() {
        let r = t.recv[c as usize];
        if r != NONE {
            let add = q[c as usize];
            q[r as usize] += add;
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph from its points' neighbour lists (each in increasing index); no triangles (the tree reads none).
    fn graph(lists: &[&[u32]]) -> Graph {
        let mut start = vec![0u32];
        let mut adj = Vec::new();
        for l in lists {
            adj.extend_from_slice(l);
            start.push(adj.len() as u32);
        }
        Graph { tris: Vec::new(), tri_n: Vec::new(), start, adj }
    }

    fn gf(v: &[f64]) -> Vec<Gf> {
        v.iter().map(|&x| Gf::from_f64(x)).collect()
    }

    #[test]
    fn a_slope_drains_each_point_to_its_steepest_lower_neighbour_in_linear_time() {
        // 0 — 1 — 2 — 3, and 3 also touches 1 by a long link: 3 drains to 2 (slope 1) over 1 (slope 2/4).
        let g = graph(&[&[1], &[0, 2, 3], &[1, 3], &[1, 2]]);
        let len = gf(&[1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0]);
        let t = tree(&gf(&[0.0, 1.0, 2.0, 3.0]), &[true, false, false, false], &g, &len);
        assert_eq!(t.recv, vec![NONE, 0, 1, 2]);
        assert_eq!(t.order, vec![0, 1, 2, 3]);
        assert_eq!(t.link, gf(&[0.0, 1.0, 1.0, 1.0]));
        assert!(!t.flooded);
        // Equal slopes: the smaller index. Point 2 sees 0 and 1 at one height and one length.
        let g = graph(&[&[2], &[2], &[0, 1]]);
        let t = tree(&gf(&[0.0, 0.0, 1.0]), &[true, true, false], &g, &gf(&[1.0, 1.0, 1.0, 1.0]));
        assert_eq!(t.recv, vec![NONE, NONE, 0]);
        // The water down the tree: each point's own and every donor's.
        assert_eq!(accumulate(&t, &gf(&[1.0, 1.0, 1.0])), gf(&[2.0, 1.0, 1.0]));
    }

    #[test]
    fn a_pit_hands_the_tree_to_the_flood_which_spills_it_and_leaves_out_what_no_outlet_reaches() {
        // 0 — 1 — 2 — 3 with a pit at 2 (under both neighbours); point 4 touches nothing.
        let g = graph(&[&[1], &[0, 2], &[1, 3], &[2], &[]]);
        let len = gf(&[1.0; 6]);
        let t = tree(&gf(&[0.0, 2.0, 1.0, 3.0, 5.0]), &[true, false, false, false, false], &g, &len);
        assert!(t.flooded);
        // The pit drains over its pass at 1, and 3 drains through the pit's flooded level.
        assert_eq!(t.recv, vec![NONE, 0, 1, 2, NONE]);
        assert_eq!(t.order, vec![0, 1, 2, 3]);
        // A point reached after both its outlets takes the steeper of the two (point 3, a pit, calls the flood).
        let g = graph(&[&[2], &[2], &[0, 1, 3], &[2]]);
        let t = tree(&gf(&[0.0, 0.5, 2.0, -1.0]), &[true, true, false, false], &g, &gf(&[1.0; 6]));
        assert!(t.flooded);
        assert_eq!(t.recv, vec![NONE, NONE, 0, 2]);
    }
}

