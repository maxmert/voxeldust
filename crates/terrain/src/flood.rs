//! ★ THE PRIORITY FLOOD ON THE SOLVE'S CORES (ruling B72 item 1: the flood, the lakes' hierarchy and the deposit on all
//! cores, the same bytes; Barnes 2017, *Parallel Priority-Flood depression filling for trillion cell digital elevation
//! models on desktops or clusters*, Computers & Geosciences 96, 56–68, <https://doi.org/10.1016/j.cageo.2016.07.001>).
//! The routing's flood and the running sea's reach ask one question: for every node, the least, over the paths from a
//! seed, of the highest node on the path (a seed: its own height). The answer is UNIQUE — it depends on no visit order
//! — so the cores may compute it in any order and give the bytes one core gives.
//!
//! The nodes are cut into the cores' contiguous ranges, the TILES ([`crate::par::ranges`]: a face's rows, in order).
//! Each tile floods itself from its own seeds and from its PERIMETER — every node with a neighbour in another tile — and
//! each perimeter node starts a WATERSHED of its own; the tile writes down where two watersheds touch and the height of
//! the higher side there. One small flood over the watersheds alone then gives each watershed its LEVEL, and every node
//! stands at its tile's fill or its watershed's level, whichever is higher. One tile has no perimeter, so on one core
//! the flood is the one-core priority flood itself, seeds first, lowest first, ties to the smaller index.
//!
//! **Example.** The home planet's shard routes its water on seven threads. Thread 3 floods the rows in the middle of the
//! third face; the rows it shares with threads 2 and 4 each start a watershed of their own. A crater lake cut in half
//! by the line between two tiles is two watersheds in the tiles' floods, and the small flood over the watersheds joins
//! them again: the lake fills to the sill it fills to on one core.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;

use crate::macro_lattice::{MacroLattice, NO_NODE};
use crate::par::Cores;

/// One tile's own flood ([`fill`]): its nodes' fill and watershed, in the tile's order, and every pair of watersheds
/// that touch in it or across its perimeter, with the least height at which they touch.
struct Tile {
    fill: Vec<i32>,
    shed: Vec<u32>,
    touch: Vec<(u32, u32, i32)>,
}

/// ★ THE FILL from the `seed` nodes over the heights `z`: for every node the least, over the paths from a seed through
/// the lattice's own eight neighbours, of the highest node on the path; a seed stands at its own height, and a node no
/// path reaches at `i32::MAX`. On the solve's cores, the same bytes on any number of them (see the module).
///
/// **Example.** A pass of the home planet's solve routes its water: the ocean's nodes are the seeds, a crater 300 km
/// inland fills to its lowest rim node, and every node of the crater's floor reads that rim's height.
#[must_use]
pub fn fill(lattice: &MacroLattice, z: &[i32], seed: &[bool], cores: Cores) -> Vec<i32> {
    let n = z.len();
    let threads = cores.threads();
    let tiles = crate::par::ranges(n, threads);
    // The watersheds: a perimeter node's own index, the seeds' water `outlet`, a node no seed reaches `outlet + 1`.
    let outlet = n as u32;
    let parts: Vec<Tile> = crate::par::per_range(tiles.len(), threads, |r| {
        r.map(|t| tile_flood(lattice, z, seed, tiles[t].clone(), outlet))
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    let touch: Vec<(u32, u32, i32)> = parts.iter().flat_map(|p| p.touch.iter().copied()).collect();
    let level = shed_levels(n, outlet, touch);
    crate::par::join(crate::par::join(crate::par::per_range(
        parts.len(),
        threads,
        |r| {
            r.map(|t| {
                let p = &parts[t];
                p.fill
                    .iter()
                    .zip(&p.shed)
                    .map(|(&f, &s)| f.max(level[s as usize]))
                    .collect::<Vec<i32>>()
            })
            .collect::<Vec<_>>()
        },
    )))
}

/// One tile's flood ([`Tile`]): seeded by the tile's seeds (the watershed `outlet`) and by its perimeter (each node its
/// own watershed), lowest first, ties to the smaller index; then every neighbouring pair of nodes in two watersheds,
/// inside the tile or across its perimeter, at the higher of the two nodes' fills. A node across the perimeter is a
/// perimeter node of its own tile, so it stands at its own height in its own watershed, or in the seeds'.
fn tile_flood(
    lattice: &MacroLattice,
    z: &[i32],
    seed: &[bool],
    tile: Range<usize>,
    outlet: u32,
) -> Tile {
    let none = outlet + 1;
    let start = tile.start;
    let mut fill = vec![i32::MAX; tile.len()];
    let mut shed = vec![none; tile.len()];
    let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();
    for v in tile.clone() {
        let rim = lattice
            .neighbours(v as u32)
            .iter()
            .any(|&m| (m != NO_NODE) && !tile.contains(&(m as usize)));
        if seed[v] | rim {
            fill[v - start] = z[v];
            shed[v - start] = if seed[v] { outlet } else { v as u32 };
            heap.push(Reverse((z[v], v as u32)));
        }
    }
    while let Some(Reverse((level, v))) = heap.pop() {
        let s = shed[v as usize - start];
        for m in lattice.neighbours(v) {
            if (m == NO_NODE) || !tile.contains(&(m as usize)) || (shed[m as usize - start] != none)
            {
                continue;
            }
            let raised = z[m as usize].max(level);
            fill[m as usize - start] = raised;
            shed[m as usize - start] = s;
            heap.push(Reverse((raised, m)));
        }
    }
    let mut touch = Vec::new();
    for v in tile.clone() {
        let (fv, sv) = (fill[v - start], shed[v - start]);
        for m in lattice.neighbours(v as u32) {
            if m == NO_NODE {
                continue;
            }
            let mi = m as usize;
            let (fm, sm) = if tile.contains(&mi) {
                (fill[mi - start], shed[mi - start])
            } else {
                (z[mi], if seed[mi] { outlet } else { m })
            };
            if sm != sv {
                touch.push((sv.min(sm), sv.max(sm), fv.max(fm)));
            }
        }
    }
    least_touch(&mut touch);
    Tile { fill, shed, touch }
}

/// The touching pairs sorted, each pair once at its least height.
fn least_touch(touch: &mut Vec<(u32, u32, i32)>) {
    touch.sort_unstable();
    touch.dedup_by_key(|e| (e.0, e.1));
}

/// ★ THE WATERSHEDS' LEVELS: a flood over the watersheds alone, from the seeds' water `outlet` (which stands under every
/// height), each watershed at the least, over the chains of touching watersheds from the seeds, of the highest touch on
/// the chain. Indexed by watershed (a node index, `outlet`, or `outlet + 1` for no seed's water, which stays at
/// `i32::MAX`).
fn shed_levels(n: usize, outlet: u32, mut touch: Vec<(u32, u32, i32)>) -> Vec<i32> {
    least_touch(&mut touch);
    let mut arcs: Vec<(u32, u32, i32)> = touch
        .iter()
        .flat_map(|&(a, b, h)| [(a, b, h), (b, a, h)])
        .collect();
    arcs.sort_unstable();
    let mut level = vec![i32::MAX; n + 2];
    level[outlet as usize] = i32::MIN;
    let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();
    heap.push(Reverse((i32::MIN, outlet)));
    while let Some(Reverse((w, a))) = heap.pop() {
        if w > level[a as usize] {
            continue;
        }
        let from = arcs.partition_point(|e| e.0 < a);
        for &(_, b, h) in arcs[from..].iter().take_while(|e| e.0 == a) {
            let raised = w.max(h);
            if raised < level[b as usize] {
                level[b as usize] = raised;
                heap.push(Reverse((raised, b)));
            }
        }
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::home_moon;

    /// The one-core priority flood, written out as the solve wrote it before the cores: the reference every core count
    /// must equal.
    fn one_core(lattice: &MacroLattice, z: &[i32], seed: &[bool]) -> Vec<i32> {
        let mut out = vec![i32::MAX; z.len()];
        let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();
        for (i, (&h, _)) in z.iter().zip(seed).enumerate().filter(|(_, (_, s))| **s) {
            out[i] = h;
            heap.push(Reverse((h, i as u32)));
        }
        while let Some(Reverse((level, v))) = heap.pop() {
            for m in lattice.neighbours(v).into_iter().filter(|&m| m != NO_NODE) {
                let mi = m as usize;
                let raised = z[mi].max(level);
                let fresh = out[mi] == i32::MAX;
                out[mi] = [out[mi], raised][usize::from(fresh)];
                heap.extend(fresh.then_some(Reverse((raised, m))));
            }
        }
        out
    }

    /// A field of hills and hollows drawn from a hash, `amp` sixteenths tall, and a seed set of about one node in
    /// `every`: rugged enough that every tile holds closed hollows, watersheds that touch at many heights, and lakes cut
    /// by the tiles' lines.
    fn field(n: usize, amp: u64, every: u64, salt: u64) -> (Vec<i32>, Vec<bool>) {
        let hash = |i: u64| {
            let mut x = (i ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            x ^= x >> 31;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^ (x >> 29)
        };
        // A hash taken into `0..k` by its high word (no division on the recipe's crate).
        let below = |x: u64, k: u64| ((u128::from(x) * u128::from(k)) >> 64) as u64;
        let z = (0..n as u64)
            .map(|i| below(hash(i), amp) as i32 - 50)
            .collect();
        let seed = (0..n as u64)
            .map(|i| below(hash(i + 7), every) == 0)
            .collect();
        (z, seed)
    }

    /// ★ THE SAME BYTES ON ANY NUMBER OF CORES: on the moon's lattice, over rugged fields with few and many seeds, the
    /// fill on one core is the one-core priority flood, and every other core count gives the same vector.
    #[test]
    fn the_fill_on_any_number_of_cores_is_the_one_core_flood() {
        let lattice = home_moon().macro_lattice().expect("the moon has a lattice");
        let n = lattice.node_count();
        for (amp, every, salt) in [(400, 997, 1), (37, 61, 2), (5, 4_001, 3), (3_000, 7, 4)] {
            let (z, seed) = field(n, amp, every, salt);
            let reference = one_core(&lattice, &z, &seed);
            assert!(
                reference.iter().zip(&z).filter(|(f, h)| f > h).count() * 50 > n,
                "the field holds hollows to fill"
            );
            for cores in [0, 1, 2, 3, 5, 8, 13] {
                assert_eq!(
                    fill(&lattice, &z, &seed, Cores(cores)),
                    reference,
                    "{cores} cores, field {salt}"
                );
            }
        }
    }

    /// No seed: no node is reached, on one core and on many.
    #[test]
    fn with_no_seed_no_node_is_reached() {
        let lattice = home_moon().macro_lattice().expect("the moon has a lattice");
        let n = lattice.node_count();
        let (z, _) = field(n, 400, 1, 5);
        let none = vec![false; n];
        assert_eq!(fill(&lattice, &z, &none, Cores(1)), vec![i32::MAX; n]);
        assert_eq!(fill(&lattice, &z, &none, Cores(4)), vec![i32::MAX; n]);
    }

    /// The touching pairs keep each pair once, at its least height.
    #[test]
    fn a_touch_keeps_its_least_height() {
        let mut touch = vec![(3, 9, 40), (1, 2, 7), (3, 9, 12), (1, 2, 7), (3, 9, 25)];
        least_touch(&mut touch);
        assert_eq!(touch, vec![(1, 2, 7), (3, 9, 12)]);
    }
}
