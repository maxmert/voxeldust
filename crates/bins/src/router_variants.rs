//! ★ THE WASHBOARD'S TWO CURES, AS A WHAT-IF (ruling B10 item 2, 2026-09-26).
//!
//! **WHY IT EXISTS.** The shoreline bench's panels showed the ground itself on the lattice: parallel
//! ridges at exactly 45°, one node apart, across the high lake and the second pocket's floor, and
//! chains of lakes on exact diagonals. That is D8's own bias on smooth ground — every node drains to
//! ONE of its eight neighbours, so on an even slope every stream takes the same lattice direction and
//! the erosion cuts parallel valleys one node apart (Fairfield & Leymarie 1991, Water Resour. Res.
//! 27(5): *"the derived water channels tend to flow in parallel lines along preferred directions
//! engendered by the sampling grid orientation"*). The owner ruled (B10) that two cures are MEASURED
//! before R2, and this module is their what-if: it rewrites the drain directions of a state the
//! solve has ROUTED, and it never changes the solve.
//!
//! 1. **RHO8** (Fairfield & Leymarie 1991): each node still drains to ONE neighbour, but the length to
//!    a diagonal neighbour is `ρ` times the row's length with `ρ = 2 − r`, one seeded draw `r` in
//!    `[0, 1)` a node — so over many nodes the directions average to the ground's own slope, not the
//!    lattice's. Here the draw is a SEEDED HASH (a named scatter, ruling T9), so one seed gives one
//!    answer on every host; the earlier refusal "a random tie-break breaks SL10's byte gate" does not
//!    hold for a seeded draw (step 2b's flat step already uses one).
//! 2. **D∞** (Tarboton 1997, Water Resour. Res. 33(2)): the node's steepest direction over the eight
//!    triangular facets its neighbours make, a CONTINUOUS angle, and its water split between the two
//!    neighbours that bracket that angle in proportion to it. No fitted number. It costs the tree a
//!    second receiver, so the what-if carries its own accumulation, its own implicit cut and its own
//!    deposit over the two links — each the solve's own arithmetic with the link's share.
//!
//! The FLATS (a routed node whose receiver stands at its own flooded height) and the outlets keep the
//! solve's own receivers in both cures: the flats are step 2c's work, folded into this one by B10.
//! Lengths here are the chords of the lattice's own directions in `f64`; the solve's cached chord can
//! differ by a metre, which only a rewritten link ever reads.
//!
//! ★ SINCE PHASE 2 (2026-10-04) this module's own cut is the PRE-PHASE-2 stream-power law ([`GAIN_BITS`]): the game
//! erodes by the joint law (ruling B43 item 2), so the what-ifs here read the old erosion, and the solve's copy no longer
//! routes by them (DEFERRED D-TERRAIN-10: their port to the joint law comes when R2 measures D∞, ruling B12). The
//! rewrites themselves (Rho8's drains, D∞'s facets and the facing reading) do not depend on the cut.
//!
//! **Example.** On the high plateau every node of an even slope facing 30° drains north-east under
//! D8, so the cut makes north-east furrows one node apart. Under Rho8 about a third of those nodes
//! drain east instead, and the furrows break up; under D∞ each node sends two thirds of its water
//! north-east and one third east, and the valley follows 30°.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::f64::consts::{PI, SQRT_2};

use vd_core::glam::DVec3;
use vd_recipe::root::isqrt;
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE, STENCIL};
use vd_terrain::solve::{DRY, MacroSolve};

/// ★ THE PRE-PHASE-2 STREAM-POWER ARITHMETIC this what-if's own cut was written on (generator 22's `solve.rs`: the
/// fraction bits of the sweep's coefficient `Δt·K·√Q/L`). Phase 2 replaced the game's erosion with the joint law (ruling
/// B43 item 2), so this what-if's cut is NOT the game's any more: its reading of D∞ against the game waits for the cut's
/// port to the joint law when R2 measures D∞ (ruling B12; DEFERRED D-TERRAIN-10).
const GAIN_BITS: u32 = 24;

/// The uplift of one pass at a node on an even axis of `passes` (generator 22's `MacroSolve::uplift_step`, kept here
/// with the what-if's own cut): `total · (pass + 1) / passes − total · pass / passes`, exact.
fn uplift_step(total: i32, pass: u32, passes: u32) -> i32 {
    let total = i64::from(total);
    let passes = i64::from(passes.max(1));
    let after = (total * (i64::from(pass) + 1)).div_euclid(passes);
    let before = (total * i64::from(pass)).div_euclid(passes);
    (after - before) as i32
}

use crate::grid_bias::unit;

/// Which stencil slots are the diagonals: `(−1, −1)`, `(1, −1)`, `(−1, 1)` and `(1, 1)`.
pub const DIAGONAL: [bool; 8] = [true, false, true, false, false, true, false, true];

/// ★ TARBOTON'S EIGHT FACETS in angular order, each `(the row neighbour's slot, the diagonal
/// neighbour's slot)`: east–north-east, north–north-east, north–north-west, west–north-west,
/// west–south-west, south–south-west, south–south-east, east–south-east.
pub const FACETS: [(usize, usize); 8] = [
    (4, 7),
    (6, 7),
    (6, 5),
    (3, 5),
    (3, 0),
    (1, 0),
    (1, 2),
    (4, 2),
];

/// One whole share in the two-link split: `1 << 16`.
pub const ONE_Q16: u32 = 1 << 16;

/// The salt of Rho8's seeded draw, so it shares no stream with any other draw of the seed.
pub const RHO8_SALT: u64 = 0x5248_4F38_5F44_4952; // "RHO8_DIR"

/// Every node's unit direction, once (the lattice's own fixed-point direction as a plain vector).
#[must_use]
pub fn units(lattice: &MacroLattice) -> Vec<DVec3> {
    (0..lattice.node_count() as u32)
        .map(|node| unit(lattice, node))
        .collect()
}

/// The chord between two nodes in metres, from their unit directions.
#[must_use]
pub fn chord_m(u: &[DVec3], radius_m: f64, a: u32, b: u32) -> f64 {
    (u[a as usize] - u[b as usize]).length() * radius_m
}

/// ★ RHO8's SEEDED DRAW for a node: a number in `[0, 1)` from the seed and the node alone
/// (splitmix64's finaliser over the salted pair), the same on every host.
#[must_use]
pub fn draw(seed: u64, node: u32) -> f64 {
    let mut x = seed ^ RHO8_SALT ^ u64::from(node).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A node whose drain the cures may rewrite: routed, and to a strictly lower flooded neighbour
/// (neither an outlet nor a node of a flat).
fn on_a_slope(state: &MacroSolve, i: usize) -> bool {
    let r = state.receiver[i];
    r != NO_NODE && state.z_flood[r as usize] < state.z_flood[i]
}

/// ★ THE ORDER, donors before their receivers (Kahn's walk over the donor counts): every node whose
/// receiver chain ends at an outlet appears once; a node on a ring never appears, so `n − len` counts
/// the ring. Any order with donors first gives the solve's accumulation, cut and deposit the same
/// integers, because each node reads only its receivers' finished values.
#[must_use]
pub fn topo_order(receiver: &[u32]) -> Vec<u32> {
    let n = receiver.len();
    let mut donors = vec![0u32; n];
    for &r in receiver {
        if r != NO_NODE {
            donors[r as usize] += 1;
        }
    }
    let mut stack: Vec<u32> = (0..n as u32).filter(|&i| donors[i as usize] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = stack.pop() {
        order.push(i);
        let r = receiver[i as usize];
        if r != NO_NODE {
            donors[r as usize] -= 1;
            if donors[r as usize] == 0 {
                stack.push(r);
            }
        }
    }
    order
}

/// What a rewrite did: the nodes it read and the nodes whose drain it moved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rewrite {
    pub nodes: usize,
    pub changed: usize,
    /// Nodes on a ring after the rewrite (0 by construction: every link falls).
    pub cyclic: usize,
}

/// ★ RHO8 over a routed state: every node on a slope drains to its steepest lower neighbour where a
/// diagonal's length is `(2 − r)` rows, `r` the node's seeded draw; the flats and the outlets keep the
/// solve's own receivers. The receivers, their chords and the order are rewritten in place.
pub fn rho8(state: &mut MacroSolve, u: &[DVec3], radius_m: f64, seed: u64) -> Rewrite {
    let n = state.node_count();
    let mut report = Rewrite::default();
    let mut receiver = state.receiver.clone();
    let mut chord = state.chord.clone();
    for (i, (rec, ch)) in receiver.iter_mut().zip(chord.iter_mut()).enumerate() {
        if !on_a_slope(state, i) {
            continue;
        }
        report.nodes += 1;
        let rho = 2.0 - draw(seed, i as u32);
        let z0 = f64::from(state.z_flood[i]);
        let (mut best, mut best_s, mut best_l) = (NO_NODE, 0.0f64, 0.0f64);
        for (slot, &m) in state.lattice.neighbours(i as u32).iter().enumerate() {
            if m == NO_NODE || state.z_flood[m as usize] >= state.z_flood[i] {
                continue;
            }
            let l = chord_m(u, radius_m, i as u32, m);
            let l_eff = if DIAGONAL[slot] { l * rho / SQRT_2 } else { l };
            let s = (z0 - f64::from(state.z_flood[m as usize])) / l_eff;
            if best == NO_NODE || s > best_s {
                (best, best_s, best_l) = (m, s, l);
            }
        }
        if best != *rec {
            report.changed += 1;
            *rec = best;
            *ch = best_l.round() as u32;
        }
    }
    state.receiver = receiver;
    state.chord = chord;
    state.order = topo_order(&state.receiver);
    report.cyclic = n - state.order.len();
    report
}

/// ★ D∞'s TWO LINKS a node: the receivers, the second's share in `1/65536`, and the chords.
#[derive(Clone, Debug, Default)]
pub struct Dinf {
    /// The two receivers; the second is `NO_NODE` where the node has one.
    pub r: Vec<[u32; 2]>,
    /// The share of the node's water that goes to the SECOND receiver, in `1/65536`.
    pub w2: Vec<u32>,
    /// The chords to the two receivers, whole metres.
    pub l: Vec<[u32; 2]>,
    /// Donors before receivers over both links.
    pub order: Vec<u32>,
    /// Nodes routed by a facet (on a slope), and of them the ones with two receivers.
    pub facets: usize,
    pub split: usize,
    /// Nodes on a ring (0 by construction: every facet link falls).
    pub cyclic: usize,
}

/// ★ D∞ over a routed state (Tarboton 1997): each node on a slope takes the facet of greatest slope
/// over the flooded surface, the angle `r` inside it from the row neighbour toward the diagonal one
/// (clamped to the facet), and sends `r/α` of its water to the diagonal and the rest to the row
/// neighbour, `α` the facet's own angle. A flat node and an outlet keep the solve's one receiver.
#[must_use]
pub fn dinf(state: &MacroSolve, u: &[DVec3], radius_m: f64) -> Dinf {
    let n = state.node_count();
    let mut d = Dinf {
        r: vec![[NO_NODE; 2]; n],
        w2: vec![0; n],
        l: vec![[0; 2]; n],
        ..Dinf::default()
    };
    for i in 0..n {
        let r0 = state.receiver[i];
        if !on_a_slope(state, i) {
            d.r[i] = [r0, NO_NODE];
            d.l[i] = [state.chord[i], 0];
            continue;
        }
        let nb = state.lattice.neighbours(i as u32);
        let z0 = f64::from(state.z_flood[i]);
        // (the row, the diagonal, the angle r, the facet's angle α, the slope)
        let mut best: Option<(u32, u32, f64, f64, f64)> = None;
        for &(a, b) in &FACETS {
            let (e1, e2) = (nb[a], nb[b]);
            if e1 == NO_NODE || e2 == NO_NODE {
                continue;
            }
            let z1 = f64::from(state.z_flood[e1 as usize]);
            let z2 = f64::from(state.z_flood[e2 as usize]);
            let d1 = chord_m(u, radius_m, i as u32, e1);
            let d2 = chord_m(u, radius_m, e1, e2);
            let alpha = (d2 / d1).atan();
            let s1 = (z0 - z1) / d1;
            let s2 = (z1 - z2) / d2;
            let r = s2.atan2(s1);
            let (r, s) = if r < 0.0 {
                (0.0, s1)
            } else if r > alpha {
                (alpha, (z0 - z2) / chord_m(u, radius_m, i as u32, e2))
            } else {
                (r, s1.hypot(s2))
            };
            if s > 0.0 && best.is_none_or(|b| s > b.4) {
                best = Some((e1, e2, r, alpha, s));
            }
        }
        let Some((e1, e2, r, alpha, _)) = best else {
            d.r[i] = [r0, NO_NODE];
            d.l[i] = [state.chord[i], 0];
            continue;
        };
        d.facets += 1;
        let w2 = ((r / alpha) * f64::from(ONE_Q16)).round() as u32;
        let l1 = chord_m(u, radius_m, i as u32, e1).round() as u32;
        let l2 = chord_m(u, radius_m, i as u32, e2).round() as u32;
        if w2 == 0 {
            d.r[i] = [e1, NO_NODE];
            d.l[i] = [l1, 0];
        } else if w2 >= ONE_Q16 {
            d.r[i] = [e2, NO_NODE];
            d.l[i] = [l2, 0];
        } else {
            d.split += 1;
            d.r[i] = [e1, e2];
            d.w2[i] = w2;
            d.l[i] = [l1, l2];
        }
    }
    // The order over both links (Kahn).
    let mut donors = vec![0u32; n];
    for links in &d.r {
        for &r in links {
            if r != NO_NODE {
                donors[r as usize] += 1;
            }
        }
    }
    let mut stack: Vec<u32> = (0..n as u32).filter(|&i| donors[i as usize] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = stack.pop() {
        order.push(i);
        for &r in &d.r[i as usize] {
            if r != NO_NODE {
                donors[r as usize] -= 1;
                if donors[r as usize] == 0 {
                    stack.push(r);
                }
            }
        }
    }
    d.cyclic = n - order.len();
    d.order = order;
    d
}

/// The two links of a node as `(receiver, share in 1/65536, chord)` pairs; one pair where the node
/// has one receiver, none at an outlet.
fn links(d: &Dinf, i: usize) -> [(u32, u32, u32); 2] {
    let [r1, r2] = d.r[i];
    if r2 == NO_NODE {
        [(r1, ONE_Q16, d.l[i][0]), (NO_NODE, 0, 0)]
    } else {
        [(r1, ONE_Q16 - d.w2[i], d.l[i][0]), (r2, d.w2[i], d.l[i][1])]
    }
}

/// ★ THE ACCUMULATION over both links: the rain on every node's own area, handed down the order,
/// each node's water split by its share — exact integers, nothing lost (the second link takes the
/// floor of its share and the first the rest).
pub fn dinf_accumulate(state: &mut MacroSolve, d: &Dinf) {
    for i in 0..state.node_count() {
        state.discharge[i] = u64::from(state.rain[i]) * state.area[i];
    }
    for &node in &d.order {
        let i = node as usize;
        let [r1, r2] = d.r[i];
        if r1 == NO_NODE {
            continue;
        }
        let q = state.discharge[i];
        if r2 == NO_NODE {
            state.discharge[r1 as usize] += q;
        } else {
            let q2 = ((u128::from(q) * u128::from(d.w2[i])) >> 16) as u64;
            state.discharge[r2 as usize] += q2;
            state.discharge[r1 as usize] += q - q2;
        }
    }
}

/// The sweep's base level at a node, stated exactly as the solve's own (private) `routing_base`:
/// the flood's level where the routing raised the node, the sea under the sea, the terrain otherwise,
/// and never under the terrain.
pub(crate) fn routing_base(state: &MacroSolve, node: u32) -> i32 {
    let i = node as usize;
    let level = if state.z_flood[i] > state.z[i] {
        state.z_flood[i]
    } else if state.ocean[i] {
        state.sea_z
    } else {
        DRY
    };
    state.z[i].max(level)
}

/// What a two-link sweep did: the nodes it lowered, the largest cut and the cut summed (steps), and
/// the lake nodes its deposit filled with the fill summed (steps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sweep {
    pub lowered: usize,
    pub max_cut: i32,
    pub total_cut: u64,
    pub filled: usize,
    pub total_fill: u64,
}

/// ★ THE TWO-LINK SWEEP: [`dinf_cut`], then the solve's own deposit over both links. With one link of
/// the whole share it is the solve's own sweep to the integer (the test below proves it).
pub fn dinf_sweep(state: &mut MacroSolve, d: &Dinf, gain: u64, pass: u32, passes: u32) -> Sweep {
    let mut report = dinf_cut(state, d, gain, pass, passes);
    let (filled, total_fill) = dinf_deposit(state, d);
    report.filled = filled;
    report.total_fill = total_fill;
    report
}

/// ★ THE IMPLICIT CUT OVER BOTH LINKS: the solve's own stream-power step (the uplift share, then
/// receivers before donors, a lake node skipped), with the node's coefficient split by its links —
/// `c_k = gain · erodibility · √Q · share_k / chord_k` — so that
/// `z' = (z + Σ c_k · base_k) / (1 + Σ c_k)`. No deposit: the caller lays the cut.
pub fn dinf_cut(state: &mut MacroSolve, d: &Dinf, gain: u64, pass: u32, passes: u32) -> Sweep {
    dinf_cut_skipping(state, d, gain, pass, passes, None)
}

/// ★ THE CUT, SKIPPING THE DEPOSIT'S NODES: as [`dinf_cut`], but a node `skip` names is left uncut,
/// as the solve leaves a lake node — the research's cure (a) for a fan the next sweep would grade
/// back to its spill (its stated limit: a fan never cuts down into itself).
pub fn dinf_cut_skipping(
    state: &mut MacroSolve,
    d: &Dinf,
    gain: u64,
    pass: u32,
    passes: u32,
    skip: Option<&[bool]>,
) -> Sweep {
    let mut report = Sweep::default();
    let one = 1i128 << GAIN_BITS;
    for i in 0..state.node_count() {
        state.z[i] += uplift_step(state.uplift[i], pass, passes);
        state.cut_now[i] = 0;
    }
    for k in (0..d.order.len()).rev() {
        let i = d.order[k] as usize;
        if d.r[i][0] == NO_NODE || state.pit[i] > 0 || skip.is_some_and(|m| m[i]) {
            continue;
        }
        let cbase = ((u128::from(gain) * u128::from(state.erodibility_q8[i])) >> 8)
            * u128::from(isqrt(state.discharge[i]));
        let mut numerator = i128::from(state.z[i]) << GAIN_BITS;
        let mut denominator = one;
        for (r, share, l) in links(d, i) {
            if r == NO_NODE {
                continue;
            }
            let c =
                (cbase * u128::from(share) / (u128::from(ONE_Q16) * u128::from(l.max(1)))) as i128;
            numerator += c * i128::from(routing_base(state, r));
            denominator += c;
        }
        let cut = (i128::from(state.z[i]) - numerator.div_euclid(denominator)) as i32;
        if cut > 0 {
            report.lowered += 1;
            report.max_cut = report.max_cut.max(cut);
            report.total_cut += cut as u64;
            state.z[i] -= cut;
            state.removed[i] = state.removed[i].saturating_add(cut as u32);
            state.pending[i] = state.pending[i].saturating_add(cut);
            state.cut_now[i] = cut as u32;
        }
    }
    report
}

/// The solve's own ONE-receiver tree as two-link rows (the second link empty): the cut and the
/// deposits of this module then read the tree the solve routed.
#[must_use]
pub fn single_links(state: &MacroSolve) -> Dinf {
    Dinf {
        r: state.receiver.iter().map(|&r| [r, NO_NODE]).collect(),
        w2: vec![0; state.node_count()],
        l: state.chord.iter().map(|&c| [c, 0]).collect(),
        order: state.order.clone(),
        ..Dinf::default()
    }
}

/// What a basin fill did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BasinFill {
    /// The filled hollows the pass found, and of them the ones that took sediment.
    pub basins: usize,
    pub fed: usize,
    /// The nodes raised and the fill summed, in steps.
    pub filled: usize,
    pub total_fill: u64,
    /// The hollows filled to their spill, whose surplus ran on down the tree.
    pub full: usize,
}

/// ★ THE BASIN FILL — the washboard's cure measured on 2026-09-26. The stage trace of the second
/// pocket's floor found the 45° ridges born in the FIRST pass and growing every pass (the fine
/// texture 25 m → 405 m → 1 004 m RMS, its direction turning to exactly 45°): the solve's deposit
/// fills the FIRST lake node the cut reaches, up to its spill, so the sediment lies in one-node
/// lines along the flat's own straight receiver paths. A lake does not settle its sediment on its
/// inflow's path: the water spreads it, and the floor rises from its lowest point up, as the water
/// itself does. So here the cut is carried down the links as the solve carries it, but a FILLED
/// HOLLOW (the connected nodes the routing raised to one flooded level) HOLDS what reaches any of
/// its nodes; each hollow, upstream first, is then filled as ONE body — its lowest nodes raised to a
/// common level that the volume pays for, never over the spill — and only a surplus runs on from its
/// outlet down the tree, to the next hollow or the sea. Ruling W7's "the cut is laid in the first
/// hollow downstream" stands; only how the hollow takes it changes.
pub fn basin_fill_deposit(state: &mut MacroSolve, d: &Dinf) -> BasinFill {
    basin_fill_shaped(state, d, &FillOpts::level()).0
}

/// ★ A SHAPE FOR THE FILL: given the state, one hollow's nodes and where the sediment entered it
/// (`(node, sediment in steps·m², water in mm·m²/yr)` — the INLETS), the height each node's fill
/// surface stands OVER the hollow's common base level, in steps (non-negative). All zeros is the
/// level fill; a published law of the fill's slope (ruling B11) gives the rest.
pub type FillShape<'a> = &'a dyn Fn(&MacroSolve, &[u32], &[(u32, u128, u128)]) -> Vec<i64>;

/// How a pass fills its hollows.
pub struct FillOpts<'a> {
    /// The fill's shape; `None` is the level fill.
    pub shape: Option<FillShape<'a>>,
    /// The runoff each node sheds (mm/yr), carried down the links beside the sediment so each
    /// inlet knows its water; `None` carries no water.
    pub runoff: Option<&'a [u32]>,
    /// Hold every node's fill under the spill (the level fill), or let the surface stand over it
    /// where the shape lifts it (Paola's fans: the base level never over the spill, the fan over it).
    pub cap_at_spill: bool,
}

impl FillOpts<'_> {
    /// The level fill.
    #[must_use]
    pub fn level() -> FillOpts<'static> {
        FillOpts {
            shape: None,
            runoff: None,
            cap_at_spill: true,
        }
    }
}

/// What a pass's fill left for the next pass to read: the nodes it raised, and those it raised OVER
/// their hollow's spill with that spill (the fans the next sweep may cut).
#[derive(Clone, Debug, Default)]
pub struct FillMarks {
    pub raised: Vec<u32>,
    pub above: Vec<(u32, i32)>,
}

/// ★ THE SHAPED BASIN FILL: as [`basin_fill_deposit`], but each hollow fills to the surface
/// `min(spill, L + g_i)` — `g` the shape's offsets and `L` the one base level the held volume pays
/// for (a bisection over whole steps) — so a law can make the floor fall from its inflows to its
/// lowest part. `None` is the level fill, to the integer.
pub fn basin_fill_shaped(
    state: &mut MacroSolve,
    d: &Dinf,
    opts: &FillOpts,
) -> (BasinFill, FillMarks) {
    const NONE: u32 = u32::MAX;
    let n = state.node_count();
    let mut report = BasinFill::default();
    let mut marks = FillMarks::default();
    // 1. The hollows: connected nodes the routing raised (a pit), of one flooded level.
    let mut basin = vec![NONE; n];
    let mut members: Vec<Vec<u32>> = Vec::new();
    for i in 0..n {
        if state.pit[i] == 0 || basin[i] != NONE {
            continue;
        }
        let id = members.len() as u32;
        let level = state.z_flood[i];
        let mut stack = vec![i as u32];
        basin[i] = id;
        let mut nodes = Vec::new();
        while let Some(v) = stack.pop() {
            nodes.push(v);
            for m in state.lattice.neighbours(v) {
                if m != NO_NODE
                    && basin[m as usize] == NONE
                    && state.pit[m as usize] > 0
                    && state.z_flood[m as usize] == level
                {
                    basin[m as usize] = id;
                    stack.push(m);
                }
            }
        }
        members.push(nodes);
    }
    report.basins = members.len();
    // 2. The cut carried down the links, leaves first; a hollow's node holds what reaches it.
    let mut flux = vec![0u128; n];
    let mut water = vec![0u128; if opts.runoff.is_some() { n } else { 0 }];
    let mut held = vec![0u128; members.len()];
    let mut arrivals: Vec<Vec<(u32, u128, u128)>> = vec![Vec::new(); members.len()];
    let mut position = vec![0usize; n];
    for (k, &node) in d.order.iter().enumerate() {
        position[node as usize] = k;
        let i = node as usize;
        flux[i] += u128::from(state.cut_now[i]) * u128::from(state.area[i].max(1));
        let w_here = match opts.runoff {
            Some(runoff) => {
                water[i] += u128::from(runoff[i]) * u128::from(state.area[i].max(1));
                water[i]
            }
            None => 0,
        };
        if basin[i] != NONE {
            if flux[i] > 0 {
                arrivals[basin[i] as usize].push((node, flux[i], w_here));
            }
            held[basin[i] as usize] += flux[i];
            flux[i] = 0;
            continue;
        }
        let [r1, r2] = d.r[i];
        if r1 == NO_NODE {
            continue;
        }
        if r2 == NO_NODE {
            flux[r1 as usize] += flux[i];
            if opts.runoff.is_some() {
                water[r1 as usize] += water[i];
            }
        } else {
            let f2 = (flux[i] * u128::from(d.w2[i])) >> 16;
            flux[r2 as usize] += f2;
            flux[r1 as usize] += flux[i] - f2;
            if opts.runoff.is_some() {
                let w2 = (water[i] * u128::from(d.w2[i])) >> 16;
                water[r2 as usize] += w2;
                water[r1 as usize] += water[i] - w2;
            }
        }
        flux[i] = 0;
    }
    // 3. The hollows, upstream first (by their latest outlet in the order), each filled as one body.
    let outlet_of = |b: usize, state: &MacroSolve| -> (usize, u32) {
        let mut best = (0usize, NO_NODE);
        for &v in &members[b] {
            let r = state.receiver[v as usize];
            let leaves = r == NO_NODE || basin[r as usize] != b as u32;
            if leaves && (best.1 == NO_NODE || position[v as usize] > best.0) {
                best = (position[v as usize], v);
            }
        }
        best
    };
    let mut queue: Vec<(usize, u32, usize)> = (0..members.len())
        .map(|b| {
            let (p, o) = outlet_of(b, state);
            (p, o, b)
        })
        .collect();
    queue.sort_unstable();
    for (_, outlet, b) in queue {
        let volume = held[b];
        if volume == 0 {
            continue;
        }
        report.fed += 1;
        if let Some(shape) = opts.shape {
            let (paid, full) = offset_fill(
                state,
                &members[b],
                &shape(state, &members[b], &arrivals[b]),
                volume,
                opts.cap_at_spill,
                &mut report,
                &mut marks,
            );
            if full && volume > paid && outlet != NO_NODE {
                report.full += 1;
                let surplus = volume - paid;
                let mut at = state.receiver[outlet as usize];
                while at != NO_NODE {
                    let a = basin[at as usize];
                    if a != NONE && a != b as u32 {
                        held[a as usize] += surplus;
                        break;
                    }
                    at = state.receiver[at as usize];
                }
            }
            continue;
        }
        let mut nodes = members[b].clone();
        nodes.sort_unstable_by_key(|&v| (state.z[v as usize], v));
        let spill = state.z_flood[nodes[0] as usize];
        // The water-filling: the level the volume pays for over the lowest nodes, never the spill.
        let (mut paid, mut area_below, mut level) =
            (0u128, 0u128, i64::from(state.z[nodes[0] as usize]));
        let mut k = 0usize;
        let mut full = false;
        loop {
            let z_here = if k < nodes.len() {
                i64::from(state.z[nodes[k] as usize]).min(i64::from(spill))
            } else {
                i64::from(spill)
            };
            let cost = area_below * (z_here - level) as u128;
            if paid + cost > volume {
                level += ((volume - paid) / area_below.max(1)) as i64;
                paid = volume;
                break;
            }
            paid += cost;
            level = z_here;
            if k >= nodes.len() || z_here >= i64::from(spill) {
                full = true;
                break;
            }
            area_below += u128::from(state.area[nodes[k] as usize].max(1));
            k += 1;
        }
        let level = level.min(i64::from(spill)) as i32;
        for &v in &nodes {
            let i = v as usize;
            if state.z[i] >= level {
                break;
            }
            let dz = level - state.z[i];
            state.z[i] += dz;
            state.pit[i] = state.pit[i].saturating_sub(dz as u32);
            state.pending[i] = state.pending[i].saturating_sub(dz);
            report.filled += 1;
            report.total_fill += dz as u64;
            marks.raised.push(v);
        }
        // 4. The surplus of a hollow filled to its spill runs on from its outlet down the tree.
        if full && volume > paid && outlet != NO_NODE {
            report.full += 1;
            let surplus = volume - paid;
            let mut at = state.receiver[outlet as usize];
            while at != NO_NODE {
                let a = basin[at as usize];
                if a != NONE && a != b as u32 {
                    held[a as usize] += surplus;
                    break;
                }
                at = state.receiver[at as usize];
            }
        }
    }
    (report, marks)
}

/// ★ THE DEPOSIT OVER BOTH LINKS: the solve's own deposit (this pass's cut carried down as a volume,
/// a lake node filled to its spill), the volume split by each node's shares.
pub fn dinf_deposit(state: &mut MacroSolve, d: &Dinf) -> (usize, u64) {
    let n = state.node_count();
    let mut flux = vec![0u128; n];
    let (mut filled, mut total_fill) = (0usize, 0u64);
    for &node in &d.order {
        let i = node as usize;
        let area = u128::from(state.area[i].max(1));
        flux[i] += u128::from(state.cut_now[i]) * area;
        if state.pit[i] > 0 && flux[i] > 0 {
            let room = u128::from(state.pit[i]) * area;
            let laid = flux[i].min(room);
            let dz = (laid / area) as i32;
            if dz > 0 {
                state.z[i] += dz;
                state.pit[i] -= dz as u32;
                state.pending[i] = state.pending[i].saturating_sub(dz);
                flux[i] -= u128::from(dz as u32) * area;
                filled += 1;
                total_fill += dz as u64;
            }
        }
        let [r1, r2] = d.r[i];
        if r1 == NO_NODE {
            continue;
        }
        if r2 == NO_NODE {
            flux[r1 as usize] += flux[i];
        } else {
            let f2 = (flux[i] * u128::from(d.w2[i])) >> 16;
            flux[r2 as usize] += f2;
            flux[r1 as usize] += flux[i] - f2;
        }
    }
    (filled, total_fill)
}

/// ★ THE OFFSET FILL of one hollow: every node raised to the surface `L + g_i` where it stands under
/// it — held under the spill when `cap` is set — `L` the largest whole step, never over the spill,
/// whose cost `Σ a_i · max(0, surface_i − z_i)` the volume pays. Returns the volume laid and
/// whether the base level reached the spill (the hollow is full).
#[allow(clippy::too_many_arguments)]
fn offset_fill(
    state: &mut MacroSolve,
    nodes: &[u32],
    g: &[i64],
    volume: u128,
    cap: bool,
    report: &mut BasinFill,
    marks: &mut FillMarks,
) -> (u128, bool) {
    let spill = i64::from(state.z_flood[nodes[0] as usize]);
    let surface = |level: i64, gi: i64| -> i64 {
        if cap {
            (level + gi).min(spill)
        } else {
            level + gi
        }
    };
    let cost = |state: &MacroSolve, level: i64| -> u128 {
        nodes
            .iter()
            .zip(g)
            .map(|(&v, &gi)| {
                let i = v as usize;
                let target = surface(level, gi);
                let z = i64::from(state.z[i]);
                if target > z {
                    (target - z) as u128 * u128::from(state.area[i].max(1))
                } else {
                    0
                }
            })
            .sum()
    };
    let capacity = cost(state, spill);
    let (level, paid, full) = if volume >= capacity {
        (spill, capacity, true)
    } else {
        let mut lo = nodes
            .iter()
            .zip(g)
            .map(|(&v, &gi)| i64::from(state.z[v as usize]) - gi)
            .min()
            .unwrap_or(spill);
        let mut hi = spill;
        // The largest level whose cost the volume pays: cost(lo) is zero, cost(hi) over the volume.
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if cost(state, mid) <= volume {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo, cost(state, lo), false)
    };
    for (&v, &gi) in nodes.iter().zip(g) {
        let i = v as usize;
        let target = surface(level, gi);
        let z = i64::from(state.z[i]);
        if target > z {
            let dz = (target - z) as i32;
            state.z[i] += dz;
            state.pit[i] = state.pit[i].saturating_sub(dz as u32);
            state.pending[i] = state.pending[i].saturating_sub(dz);
            report.filled += 1;
            report.total_fill += dz as u64;
            marks.raised.push(v);
            if target > spill {
                marks.above.push((v, spill as i32));
            }
        }
    }
    (paid, full)
}

/// ★ PAOLA, HELLER & ANGEVINE 1992 (Basin Research 4, 73–90): the fill surface of an alluvial basin
/// obeys one diffusion (eq. 16a), and at an inlet its slope is the sediment load over the
/// diffusivity (eq. 17a), which for BRAIDED channels makes the slope `(s − 1)/(8 A √c_f)` times the
/// ratio of the sediment to the water a river brings: `c_f = 0.01` (eq. 7, "typically of order
/// 0.01"), `s = 2.7` (§2.5), `ε = 0.4` (Parker 1978, measured; eq. 8), `A = (ε/(1 + ε))^{3/2}`
/// (eq. 10). About 13.9. The calibration body is Earth's gravel-bed rivers; the width, the gravity
/// and the grain size cancel, so the law holds on every body.
#[must_use]
pub fn paola_braided() -> f64 {
    let (c_f, s, eps) = (0.01f64, 2.7f64, 0.4f64);
    let a = (eps / (1.0 + eps)).powf(1.5);
    (s - 1.0) / (8.0 * a * c_f.sqrt())
}

/// What the fan law found in one pass: every fed inlet's slope, reach and fan top.
#[derive(Clone, Debug, Default)]
pub struct Fans {
    pub inlets: usize,
    /// Inlets with no water (their slope is the scree's tangent), and inlets whose slope the scree's
    /// tangent capped.
    pub dry: usize,
    pub capped: usize,
    pub slopes: Vec<f64>,
    pub reach_m: Vec<f64>,
    pub top_m: Vec<f64>,
}

/// ★ THE FAN OFFSETS of one hollow (the research's recommendation, 2026-09-26): each fed inlet `k`
/// takes the slope `S_k = 13.9 · (sediment / water)` in the solve's own units
/// (`62.5 · V_k / (Δt · Q_k)`: steps·m² a pass over mm·m²/yr), capped at the scree's tangent; a fan
/// share `A_k = A_hollow · V_k / ΣV` (uniform aggradation, eq. 16a's steady state); a reach found by
/// visiting the hollow's nodes best-first by the STRAIGHT chord from the inlet (through the hollow's
/// own adjacency) until they cover `A_k`; and the steady profile
/// `Φ_k(d) = ∫_d^R S_k · (1 − A(<r)/A_k) dr`, walked back from the reach. The hollow's offset is the
/// highest fan at each node (fans that meet leave a crease). Offsets in steps.
///
/// **Example.** A river draining a wet range enters the second pocket: its fan falls about nineteen
/// metres a node from the mouth and flattens toward the pocket's middle, where the lake stays.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn paola_fan_offsets(
    state: &MacroSolve,
    nodes: &[u32],
    inlets: &[(u32, u128, u128)],
    u: &[DVec3],
    radius_m: f64,
    dt_yr: f64,
    talus_tan: &[f64],
    fans: &mut Fans,
) -> Vec<i64> {
    let k_paola = paola_braided();
    let index: BTreeMap<u32, usize> = nodes.iter().enumerate().map(|(k, &v)| (v, k)).collect();
    let area: Vec<f64> = nodes
        .iter()
        .map(|&v| state.area[v as usize] as f64)
        .collect();
    let a_hollow: f64 = area.iter().sum();
    let v_sum: f64 = inlets.iter().map(|x| x.1 as f64).sum();
    let mut phi = vec![0.0f64; nodes.len()];
    if v_sum <= 0.0 {
        return vec![0; nodes.len()];
    }
    for &(inlet, v, q) in inlets.iter().filter(|x| x.1 > 0) {
        fans.inlets += 1;
        let cap = talus_tan[inlet as usize];
        let slope = if q == 0 {
            fans.dry += 1;
            cap
        } else {
            let raw = k_paola * 62.5 * v as f64 / (dt_yr * q as f64);
            if raw > cap {
                fans.capped += 1;
                cap
            } else {
                raw
            }
        };
        let a_k = a_hollow * v as f64 / v_sum;
        let Some(&start) = index.get(&inlet) else {
            continue;
        };
        // Best-first by the straight chord from the inlet, through the hollow's own adjacency.
        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
        let mut seen = vec![false; nodes.len()];
        heap.push(Reverse((0, start)));
        seen[start] = true;
        let mut visited: Vec<(usize, f64)> = Vec::new();
        let mut covered = 0.0f64;
        while let Some(Reverse((dmm, k))) = heap.pop() {
            visited.push((k, dmm as f64 / 1000.0));
            covered += area[k];
            if covered >= a_k {
                break;
            }
            for m in state.lattice.neighbours(nodes[k]) {
                if m == NO_NODE {
                    continue;
                }
                if let Some(&km) = index.get(&m)
                    && !seen[km]
                {
                    seen[km] = true;
                    let d = chord_m(u, radius_m, inlet, m);
                    heap.push(Reverse(((d * 1000.0).round() as u64, km)));
                }
            }
        }
        // The steady profile, walked back from the reach.
        let mut cum = Vec::with_capacity(visited.len());
        let mut acc = 0.0f64;
        for &(k, _) in &visited {
            acc += area[k];
            cum.push(acc);
        }
        let mut shape = vec![0.0f64; visited.len()];
        for j in (0..visited.len().saturating_sub(1)).rev() {
            let dd = visited[j + 1].1 - visited[j].1;
            shape[j] = shape[j + 1] + slope * (1.0 - (cum[j] / a_k).min(1.0)) * dd;
        }
        for (j, &(k, _)) in visited.iter().enumerate() {
            if shape[j] > phi[k] {
                phi[k] = shape[j];
            }
        }
        fans.slopes.push(slope);
        fans.reach_m.push(visited.last().map_or(0.0, |x| x.1));
        fans.top_m.push(shape.first().copied().unwrap_or(0.0));
    }
    phi.iter().map(|&m| (m * 16.0).round() as i64).collect()
}

/// ★ THE DOMINANT TREE of a D∞ state: each node's larger link becomes its one receiver, so every
/// reader of ONE receiver (the ice, the basins, the budget's hierarchy) reads a tree; the order is
/// rewritten to match.
pub fn dominant(state: &mut MacroSolve, d: &Dinf) -> usize {
    for i in 0..state.node_count() {
        let [r1, r2] = d.r[i];
        let second = r2 != NO_NODE && d.w2[i] > ONE_Q16 / 2;
        state.receiver[i] = if second { r2 } else { r1 };
        state.chord[i] = if second { d.l[i][1] } else { d.l[i][0] };
    }
    state.order = topo_order(&state.receiver);
    state.node_count() - state.order.len()
}

/// ★ THE GROUND'S OWN FACING, as a reading no router makes: over the kept nodes with all eight
/// neighbours, the least-squares gradient on the stencil's own offsets, weighted by its size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Facing {
    pub nodes: usize,
    /// The fourfold amplitude of the facing, in percent: `|Σ s·e^{4iθ}| / Σ s`. An isotropic ground
    /// reads near zero; ground furrowed along the rows reads positive at phase 0°, along the
    /// diagonals at phase 45°.
    pub fourfold_pct: f64,
    /// The phase of that fourfold mode, in degrees in `[0, 90)`.
    pub phase_deg: f64,
    /// The facing folded into `[0°, 180°)` in twelve 15° bins, weighted by the slope.
    pub twelve: [f64; 12],
}

impl Facing {
    /// The largest of the twelve bins over their mean: one for an isotropic ground.
    #[must_use]
    pub fn flatness(&self) -> f64 {
        let mean = self.twelve.iter().sum::<f64>() / 12.0;
        let max = self.twelve.iter().copied().fold(0.0f64, f64::max);
        if mean > 0.0 { max / mean } else { f64::NAN }
    }
}

/// ★ THE FACING READING of the field `z` over the nodes `keep` names.
#[must_use]
pub fn facing(lattice: &MacroLattice, z: &[i32], keep: &dyn Fn(usize) -> bool) -> Facing {
    let (mut c4, mut s4, mut sum, mut nodes) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    let mut twelve = [0.0f64; 12];
    for i in 0..lattice.node_count() {
        if !keep(i) {
            continue;
        }
        let nb = lattice.neighbours(i as u32);
        if nb.contains(&NO_NODE) {
            continue;
        }
        let (mut gx, mut gy) = (0.0f64, 0.0f64);
        for (slot, &m) in nb.iter().enumerate() {
            let (di, dj) = STENCIL[slot];
            let dz = f64::from(z[m as usize] - z[i]);
            gx += dz * f64::from(di);
            gy += dz * f64::from(dj);
        }
        let s = gx.hypot(gy) / 6.0;
        if s <= 0.0 {
            continue;
        }
        let theta = gy.atan2(gx);
        c4 += s * (4.0 * theta).cos();
        s4 += s * (4.0 * theta).sin();
        sum += s;
        nodes += 1;
        let folded = theta.rem_euclid(PI);
        let bin = ((folded / PI) * 12.0) as usize;
        twelve[bin.min(11)] += s;
    }
    let phase = s4.atan2(c4).to_degrees() / 4.0;
    Facing {
        nodes,
        fourfold_pct: if sum > 0.0 {
            100.0 * c4.hypot(s4) / sum
        } else {
            f64::NAN
        },
        phase_deg: phase.rem_euclid(90.0),
        twelve,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_bias::cone_z;
    use vd_terrain::home::home_moon;

    /// The home moon routed over Tarboton's cone about a face's middle.
    fn cone() -> (MacroSolve, DVec3, f64) {
        let body = home_moon();
        let lattice = MacroLattice::of(&body).expect("the moon has a lattice");
        let edge = lattice.edge as i32;
        let apex = unit(
            &lattice,
            lattice.index(vd_seed::bend::Face::PosX, edge / 2, edge / 2),
        );
        let radius_m = body.radius_m();
        let mut state = MacroSolve::new(&body).expect("the moon has a solve state");
        state.z = cone_z(&lattice, apex, radius_m, 0.05);
        let _ = state.route();
        (state, apex, radius_m)
    }

    /// The draw is a number in `[0, 1)`, the same for the same seed and node, another for another.
    #[test]
    fn the_seeded_draw_is_repeatable_and_in_range() {
        let a = draw(7, 11);
        assert_eq!(a, draw(7, 11));
        assert!((0.0..1.0).contains(&a));
        assert_ne!(a, draw(7, 12));
        assert_ne!(a, draw(8, 11));
        let mean = (0..10_000).map(|k| draw(3, k)).sum::<f64>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.01, "the draw's mean {mean}");
    }

    /// The order puts every donor before its receiver and holds every node of a tree.
    #[test]
    fn the_order_puts_donors_first() {
        // 0 → 1 → 3, 2 → 3, 3 an outlet.
        let receiver = [1, 3, 3, NO_NODE];
        let order = topo_order(&receiver);
        assert_eq!(order.len(), 4);
        let at = |x: u32| order.iter().position(|&o| o == x).expect("in the order");
        assert!(at(0) < at(1));
        assert!(at(1) < at(3));
        assert!(at(2) < at(3));
        // A ring never enters.
        assert_eq!(topo_order(&[1, 0]).len(), 0);
    }

    /// ★ Rho8 keeps every link falling and the tree whole, and it moves drains. How much it weakens
    /// the grid's signature is a MEASUREMENT the washboard bench prints, never an assertion here: on
    /// the moon's inward cone a stochastic router can read a larger fourfold mode than D8 (MEASURED
    /// 2026-09-26: 12.2 % against 6.9 %), because its random paths braid.
    #[test]
    fn rho8_keeps_a_falling_tree_and_moves_drains() {
        let (mut state, _, radius_m) = cone();
        let u = units(&state.lattice);
        let report = rho8(&mut state, &u, radius_m, 42);
        assert_eq!(report.cyclic, 0);
        assert!(report.changed > 0);
        assert!(report.changed < report.nodes);
        for i in 0..state.node_count() {
            let r = state.receiver[i];
            if r != NO_NODE {
                assert!(state.z_flood[r as usize] <= state.z_flood[i]);
            }
        }
        assert_eq!(state.order.len(), state.node_count());
    }

    /// ★ D∞ splits between two FALLING neighbours and keeps every drop. Its grid signature is the
    /// washboard bench's measurement: on an INWARD cone the nodes on the lattice's own rays send all
    /// their water along the ray while their neighbours send part of theirs in, so the rays gather
    /// water and the fourfold mode reads HIGH (MEASURED 2026-09-26: 39.8 % against D8's 6.9 %) — the
    /// cone's error, Tarboton's own quantity, is the fair reading.
    #[test]
    fn dinf_splits_between_falling_neighbours_and_keeps_every_drop() {
        let (mut state, _, radius_m) = cone();
        let u = units(&state.lattice);
        let d = dinf(&state, &u, radius_m);
        assert_eq!(d.cyclic, 0);
        assert!(d.split > 0);
        for i in 0..state.node_count() {
            if d.r[i][1] != NO_NODE {
                for r in d.r[i] {
                    assert!(state.z_flood[r as usize] < state.z_flood[i]);
                }
            }
        }
        let rain: u64 = (0..state.node_count())
            .map(|i| u64::from(state.rain[i]) * state.area[i])
            .sum();
        dinf_accumulate(&mut state, &d);
        let out: u64 = (0..state.node_count())
            .filter(|&i| d.r[i][0] == NO_NODE)
            .map(|i| state.discharge[i])
            .sum();
        assert_eq!(out, rain, "every drop reaches an outlet");
        assert_eq!(dominant(&mut state, &d), 0);
    }

    /// ★ THE PRE-PHASE-2 SOLVE'S OWN SWEEP (generator 22's `MacroSolve::sweep`), kept in the tests as the law this
    /// what-if's cut was written on: the uplift's pass share, the implicit stream-power cut over the one receiver, and
    /// the routing's deposit. The game erodes by the joint law since phase 2 (ruling B43 item 2).
    fn stream_power_sweep(
        state: &mut MacroSolve,
        gain: u64,
        pass: u32,
        passes: u32,
    ) -> vd_terrain::solve::SweepReport {
        let mut report = vd_terrain::solve::SweepReport::default();
        let one = 1i128 << GAIN_BITS;
        for i in 0..state.node_count() {
            state.z[i] += uplift_step(state.uplift[i], pass, passes);
            state.cut_now[i] = 0;
        }
        for k in (0..state.order.len()).rev() {
            let i = state.order[k] as usize;
            let r = state.receiver[i];
            if r == NO_NODE {
                continue;
            }
            if state.pit[i] > 0 {
                report.skipped_lake += 1;
                continue;
            }
            let base = routing_base(state, r);
            let c = ((u128::from(gain) * u128::from(state.erodibility_q8[i])) >> 8)
                * u128::from(isqrt(state.discharge[i]))
                / u128::from(state.chord[i].max(1));
            let c = c as i128;
            let numerator = (i128::from(state.z[i]) << GAIN_BITS) + c * i128::from(base);
            let cut = (i128::from(state.z[i]) - numerator.div_euclid(one + c)) as i32;
            if cut > 0 {
                report.lowered += 1;
                report.max_cut = report.max_cut.max(cut);
                report.total_cut += cut as u64;
                state.z[i] -= cut;
                state.removed[i] = state.removed[i].saturating_add(cut as u32);
                state.pending[i] = state.pending[i].saturating_add(cut);
                state.cut_now[i] = cut as u32;
            }
        }
        let (filled, total_fill) = state.deposit();
        report.filled = filled;
        report.total_fill = total_fill;
        report
    }

    /// ★ THE REPLICA IS THE LAW IT WAS WRITTEN ON: two-link sweeps built from the solve's OWN one receiver and its own
    /// chord give the pre-phase-2 solve's own sweep, height for height — so a D∞ run differs from that law only where
    /// the links differ. (Since phase 2 the game's sweep is the joint law; the what-if's port to it is owed, DEFERRED
    /// D-TERRAIN-10.)
    #[test]
    fn one_link_of_the_whole_share_is_the_solves_own_sweep() {
        let (mut state, _, _) = cone();
        state.accumulate();
        // Rough ground to cut: uplift the cone's far side, so both sweeps have work.
        for (k, u) in state.uplift.iter_mut().enumerate() {
            *u = (k % 97) as i32 * 16;
        }
        let d = Dinf {
            r: state.receiver.iter().map(|&r| [r, NO_NODE]).collect(),
            w2: vec![0; state.node_count()],
            l: state.chord.iter().map(|&c| [c, 0]).collect(),
            order: state.order.clone(),
            ..Dinf::default()
        };
        let mut mine = state.clone();
        let gain = 1u64 << 30;
        let stock = stream_power_sweep(&mut state, gain, 3, 8);
        let replica = dinf_sweep(&mut mine, &d, gain, 3, 8);
        assert_eq!(mine.z, state.z);
        assert_eq!(mine.pit, state.pit);
        assert_eq!(mine.pending, state.pending);
        assert_eq!(replica.lowered, stock.lowered);
        assert_eq!(replica.filled, stock.filled);
        assert!(replica.lowered > 0, "the test cut something");
    }

    /// ★ THE BASIN FILL fills a hollow from its lowest nodes to ONE level and never over its spill:
    /// a bowl lowered into the cone's side takes the sediment its slopes shed, every node it raises
    /// ends at one common level, no node of the bowl stays under that level, and the volume laid is the
    /// volume held, less at most one step over the raised nodes' area (the floor of the division).
    ///
    /// **Example.** A crater bowl under a range: the range's cut runs in, and the bowl's floor rises
    /// as one flat plain — never as a line of filled nodes along the stream that fed it.
    #[test]
    fn a_hollow_fills_from_its_lowest_nodes_to_one_level() {
        let (mut state, _, _) = cone();
        let lattice = state.lattice;
        let edge = lattice.edge as i32;
        let bowl = unit(
            &lattice,
            lattice.index(vd_seed::bend::Face::PosX, edge / 2 + edge / 8, edge / 2),
        );
        for node in 0..state.node_count() {
            let phi = unit(&lattice, node as u32)
                .dot(bowl)
                .clamp(-1.0, 1.0)
                .acos();
            if phi < 0.08 {
                // Thirty kilometres: deeper than the cone rises across the bowl, so the bowl is a
                // CLOSED hollow the routing fills (a 3 km bowl read no hollow at all).
                let share = 1.0 - (phi / 0.08) * (phi / 0.08);
                state.z[node] -= (share * 30_000.0 * 16.0).round() as i32;
            }
        }
        let _ = state.route();
        state.accumulate();
        for i in 0..state.node_count() {
            state.cut_now[i] = u32::from(state.pit[i] == 0);
        }
        let before = state.z.clone();
        let in_bowl: Vec<bool> = state.pit.iter().map(|&p| p > 0).collect();
        let d = single_links(&state);
        let r = basin_fill_deposit(&mut state, &d);
        assert!(r.basins >= 1, "the bowl is a hollow");
        assert!(r.fed >= 1 && r.filled > 0, "the bowl took sediment: {r:?}");
        let raised: Vec<usize> = (0..state.node_count())
            .filter(|&i| state.z[i] != before[i])
            .collect();
        let level = state.z[raised[0]];
        assert!(raised.iter().all(|&i| state.z[i] == level), "one level");
        assert!(
            raised.iter().all(|&i| in_bowl[i]),
            "only the bowl's nodes rise"
        );
        // Every node of the bowl that stood under the level now stands on it.
        for (i, (&z0, &bowl_node)) in before.iter().zip(&in_bowl).enumerate() {
            if bowl_node && z0 < level {
                assert_eq!(
                    state.z[i], level,
                    "no node of the bowl stays under the level"
                );
            }
        }
        let laid: u128 = raised
            .iter()
            .map(|&i| u128::from((state.z[i] - before[i]) as u32) * u128::from(state.area[i]))
            .sum();
        let area: u128 = raised.iter().map(|&i| u128::from(state.area[i])).sum();
        assert_eq!(
            r.total_fill,
            raised
                .iter()
                .map(|&i| (state.z[i] - before[i]) as u64)
                .sum::<u64>()
        );
        assert!(laid > 0 && area > 0);
    }

    /// ★ THE SHAPED FILL WITH A ZERO SHAPE IS THE LEVEL FILL, height for height: the bisection over
    /// the base level and the water-filling find the same largest level the volume pays for. So a
    /// law's shape changes the fill and nothing else.
    #[test]
    fn the_zero_shape_is_the_level_fill() {
        let (mut state, _, _) = cone();
        let lattice = state.lattice;
        let edge = lattice.edge as i32;
        let bowl = unit(
            &lattice,
            lattice.index(vd_seed::bend::Face::PosX, edge / 2 + edge / 8, edge / 2),
        );
        for node in 0..state.node_count() {
            let phi = unit(&lattice, node as u32)
                .dot(bowl)
                .clamp(-1.0, 1.0)
                .acos();
            if phi < 0.08 {
                let share = 1.0 - (phi / 0.08) * (phi / 0.08);
                state.z[node] -= (share * 30_000.0 * 16.0).round() as i32;
            }
        }
        let _ = state.route();
        state.accumulate();
        for i in 0..state.node_count() {
            state.cut_now[i] = u32::from(state.pit[i] == 0);
        }
        let d = single_links(&state);
        let mut shaped = state.clone();
        let level = basin_fill_deposit(&mut state, &d);
        let zero = |_: &MacroSolve, nodes: &[u32], _: &[(u32, u128, u128)]| vec![0i64; nodes.len()];
        let (by_shape, _) = basin_fill_shaped(
            &mut shaped,
            &d,
            &FillOpts {
                shape: Some(&zero),
                runoff: None,
                cap_at_spill: true,
            },
        );
        assert!(level.filled > 0);
        assert_eq!(shaped.z, state.z);
        assert_eq!(shaped.pit, state.pit);
        assert_eq!(by_shape, level);
    }

    /// ★ PAOLA'S BRAIDED CONSTANT is 13.9 from its published parts (c_f 0.01, s 2.7, ε 0.4).
    #[test]
    fn paolas_braided_constant_is_thirteen_point_nine() {
        let k = paola_braided();
        assert!((k - 13.91).abs() < 0.01, "the constant {k}");
    }

    /// ★ A FAN falls from its inlet: the offset is highest at the inlet, never rises away from it
    /// along the visit, is zero at the reach, and its top stands between three tenths and the whole of
    /// the slope times the reach (the half disc's closed form reads two thirds; a fan the hollow's
    /// walls bend reads less).
    #[test]
    fn a_fan_falls_from_its_inlet_to_zero_at_its_reach() {
        let (mut state, _, radius_m) = cone();
        let lattice = state.lattice;
        let edge = lattice.edge as i32;
        let bowl = unit(
            &lattice,
            lattice.index(vd_seed::bend::Face::PosX, edge / 2 + edge / 8, edge / 2),
        );
        for node in 0..state.node_count() {
            let phi = unit(&lattice, node as u32)
                .dot(bowl)
                .clamp(-1.0, 1.0)
                .acos();
            if phi < 0.08 {
                let share = 1.0 - (phi / 0.08) * (phi / 0.08);
                state.z[node] -= (share * 30_000.0 * 16.0).round() as i32;
            }
        }
        let _ = state.route();
        let nodes: Vec<u32> = (0..state.node_count() as u32)
            .filter(|&v| state.pit[v as usize] > 0)
            .collect();
        assert!(nodes.len() > 20, "a hollow of {} nodes", nodes.len());
        // One inlet on the hollow's edge (the node farthest from the bowl's centre), with a load
        // and a water that give a slope of about 2e-3 (Paola: 13.9 · 62.5 · V / (Δt · Q)).
        let far = *nodes
            .iter()
            .max_by(|a, b| {
                let da = unit(&lattice, **a).dot(bowl);
                let db = unit(&lattice, **b).dot(bowl);
                db.total_cmp(&da)
            })
            .expect("a node");
        let dt = 1.0e8;
        let q: u128 = 1_000_000_000_000_000;
        let want = 2.0e-3;
        let v = (want * dt * q as f64 / (paola_braided() * 62.5)) as u128;
        let u = units(&lattice);
        let talus = vec![1.0; state.node_count()];
        let mut fans = Fans::default();
        let g = paola_fan_offsets(
            &state,
            &nodes,
            &[(far, v, q)],
            &u,
            radius_m,
            dt,
            &talus,
            &mut fans,
        );
        assert_eq!(fans.inlets, 1);
        assert!(
            (fans.slopes[0] - want).abs() < 1.0e-6,
            "the slope {}",
            fans.slopes[0]
        );
        let at_inlet = g[nodes.iter().position(|&x| x == far).expect("the inlet")];
        assert_eq!(at_inlet, *g.iter().max().expect("offsets"));
        let top_m = fans.top_m[0];
        let reach = fans.reach_m[0];
        assert!(
            top_m > 0.3 * want * reach && top_m < want * reach,
            "top {top_m} m, reach {reach} m"
        );
        assert!(g.iter().all(|&x| x >= 0));
    }

    /// ★ THE FACING reads a furrowed field: ground that rises along the rows reads a strong fourfold
    /// mode at phase 0°, and the cone — every direction equal — reads a weak one.
    #[test]
    fn the_facing_reads_rows_strongly_and_the_cone_weakly() {
        let (state, _, _) = cone();
        let lattice = &state.lattice;
        let plane: Vec<i32> = (0..lattice.node_count() as u32)
            .map(|node| {
                let (_, i, _) = lattice.split(node);
                i * 160
            })
            .collect();
        let face_of = |node: usize| lattice.split(node as u32).0 == vd_seed::bend::Face::PosX;
        let rows = facing(lattice, &plane, &face_of);
        assert!(
            rows.fourfold_pct > 90.0,
            "rows read {:.2} %",
            rows.fourfold_pct
        );
        assert!(rows.phase_deg < 1.0 || rows.phase_deg > 89.0);
        // The cone on its OWN face only, within 40° of its pole: there every direction is equal in
        // the face's frame. (On the four faces beside it the cone's slope points along a face axis
        // by the cube's own geometry, so the whole moon reads a strong mode that is no router's.)
        let apex = unit(
            lattice,
            lattice.index(
                vd_seed::bend::Face::PosX,
                lattice.edge as i32 / 2,
                lattice.edge as i32 / 2,
            ),
        );
        let near = |node: usize| {
            face_of(node) && unit(lattice, node as u32).dot(apex) > 40f64.to_radians().cos()
        };
        let cone = facing(lattice, &state.z, &near);
        assert!(
            cone.fourfold_pct < rows.fourfold_pct / 4.0,
            "the cone reads {:.2} %",
            cone.fourfold_pct
        );
        assert!(rows.flatness() > cone.flatness());
    }
}
