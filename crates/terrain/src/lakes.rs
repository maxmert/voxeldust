//! ★ THE DEPRESSION HIERARCHY AND THE WATER BUDGET (slice 8d step 5; ruling W11; the watershed and
//! the cascade by ruling B7, 2026-09-24; the water's own tree by ruling B8 item 2, 2026-09-25).
//!
//! The routing flood raises every closed hollow to its spill so the water can be ROUTED across it.
//! Every published landscape model computes that surface and THROWS IT AWAY — Barnes calls it "an
//! important preconditioning step", Landlab writes it "to a scratch surface, never to
//! `topographic__elevation`", Cordonnier calls filling and carving "metaphors … without altering
//! elevation values". We drew it, so every hollow on the planet was a lake: MEASURED on the home
//! planet, 188 848 nodes in 51 209 patches over 9.69 % of the land, against Earth's 3.7 % of the
//! non-glaciated land and about 2 000 lakes bigger than one of our nodes.
//!
//! Here a hollow is a lake only where a FINITE amount of water actually stands in it.
//!
//! **THE HIERARCHY** (Barnes, Callaghan & Wickert 2021, *Fill–Spill–Merge*, Earth Surf. Dynam. 9,
//! 105–121, <https://doi.org/10.5194/esurf-9-105-2021>; the reference code is MIT and none of it is
//! copied — this is the merge tree of the field's own sublevel sets, built here in integers). The
//! nodes are swept in one stated order, lowest first — ★ on a FLAT of one height by the flat's
//! combined potential ([`flats`]; plan step 2b; phase 1 of the ground step), then by the smaller
//! index. A node whose lower
//! neighbours are all unvisited is a PIT and opens a new depression; a node with one lower
//! component joins it; a node with two or more MERGES them — each child closes with that node as
//! its SPILL, and a parent depression carries both. The CONNECTED OCEAN is the root and never
//! fills (2026-09-24, ruling B7 item 1: an inland pocket under the sea's level is a hollow like
//! any other, and its water is its own budget's).
//!
//! **THE WATER** (Langbein 1961, *Salinity and hydrology of closed lakes*, USGS Professional Paper
//! 412, <https://pubs.usgs.gov/pp/0412/report.pdf>): a closed lake stands at the level where the
//! standing water's evaporation balances the inflow. Every node states three whole numbers the
//! climate already computes — its rain `P`, its potential evaporation `PET`
//! ([`crate::climate::pet_mm_yr`]) and its RUNOFF `Q` ([`crate::climate::runoff_mm_yr`], the
//! Turc–Pike partition of the rain). A node UNDER the lake costs `Q + PET − P` over its area,
//! because open water evaporates at the potential rate and no longer yields a river: that is the
//! DEFICIT.
//!
//! ★ **THE WATERSHED** (2026-09-24, ruling B7; status §7 W3, the defect this cures). The runoff of a
//! node runs DOWN THE GROUND — each node to its steepest lower neighbour ([`descent`]), never down
//! the routing fill — and it is booked to the leaf hollow whose pit that descent ends at, or to the
//! ocean. So a hollow is fed by its WHOLE WATERSHED, the range above its brim included, which is
//! Fill–Spill–Merge's own rule. Before, each hollow took only the runoff of the ground under its
//! own spill: MEASURED at generator 19, a hollow's watershed supplied a median 12 times what the
//! budget gave it.
//!
//! ★ **THE CASCADE** (status §7 W4, the second defect this cures). A hollow whose inflow passes its
//! deficit is FULL, and its OVERFLOW runs on over its spill ([`cascade`]): ★ into the STEEPEST OPEN
//! SIDE of its saddle (plan step 2b) — a hollow that is not full, down the ground to the pit its
//! gate's descent ends at, or the ocean's side, down the ground into the sea or into the lower hollow
//! that met the sea first; where every side is full, into the parent's own ring. Nothing is lost:
//! `runoff = evaporated + to the ocean + stranded` to the unit ([`LakeReport`]). Before, a full
//! hollow's overflow into a dry parent vanished.
//!
//! ★ **THE WATER'S OWN TREE** (plan step 2b; ruling B8 item 2; [`water_tree`]). The shipped receivers
//! carry the water the way this budget carries it — the descent on dry ground, across a spilling
//! lake to its spill, over the spill into the side the cascade pours into — and ENDS it where the
//! budget holds it: under a closed lake's water, at a dry hollow's pit. A node at a hollow's spill
//! height stands at the full lake's edge and costs that hollow nothing (THE BRIM).
//!
//! Three ends, decided by the water and never by the shape of the hole: DRY (no inflow, or the first
//! node's own evaporation takes it all — a playa), PARTIAL (a closed lake at its balance level), or
//! SPILLING (the inflow still passes the deficit at the spill: the water leaves through an outlet,
//! and the depression is a lake with a river out of it).
//!
//! **Example.** A hollow lies at the foot of the belt's wet range. The ground of the whole range face
//! runs down into its pit, so it fills to its sill. Its overflow crosses the saddle into the dry pan
//! next to it, which becomes a salt lake half-way up its walls. Nothing is lost, and the metadepression
//! that holds both is never a lake while the pan stands under its own spill. The river that carries
//! the overflow crosses the full lake to the notch, runs down into the pan and ENDS at the salt lake's
//! shore; nothing of it reaches the sea.

// ★ THE BUDGET MAY DIVIDE: it runs once per body on the CPU, off the tick, never in a kernel, and
// every quotient here is an integer one.
#![allow(
    clippy::integer_division,
    reason = "the budget runs on the CPU once per body, never in a kernel"
)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vd_recipe::root::isqrt;
use vd_seed::rng::child_seed;

use crate::climate::Climate;
use crate::macro_lattice::NO_NODE;
use crate::solve::{DRY, MacroSolve, Z_STEPS_PER_M, steeper};

/// No depression.
pub const NO_DEP: u32 = u32::MAX;
/// The ocean's own component: the root of the hierarchy, which never fills.
pub const OCEAN: u32 = 0;

/// One depression of the hierarchy: a hollow, closed at the level where it meets its neighbour.
/// Every sum is over the depression's WHOLE subtree, as it stood when the depression closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Depression {
    /// The lowest node in it.
    pub pit: u32,
    /// That node's height, sixteenths.
    pub pit_z: i32,
    /// The level at which it meets another depression or the sea, sixteenths; `i32::MAX` where it
    /// never met one (a body with no outlet at all).
    pub spill_z: i32,
    /// The node the spill runs over; [`NO_NODE`] where it never spilled.
    pub spill_node: u32,
    /// The depression it spills into: [`OCEAN`], a parent's index, or [`NO_DEP`].
    pub parent: u32,
    /// The two depressions a parent merged; `[NO_DEP, NO_DEP]` for a leaf.
    pub children: [u32; 2],
    /// The area of its nodes, m² — the lake's own surface when it stands at the spill.
    pub area_m2: u128,
    /// The sum of every node's height times its area, sixteenths·m².
    pub z_area: i128,
    /// What covering the whole hollow with water would cost, mm·m²/yr: `Σ (Q + PET − P) · area`
    /// over its nodes UNDER its spill height — the runoff they stop yielding plus what the open
    /// surface hands the air. ★ A node at exactly the spill height stands at the full lake's edge
    /// and costs nothing (plan step 2b).
    pub deficit: u128,
    /// ★ The part of [`Depression::deficit`] its OWN ring holds: the whole of it for a leaf, and a
    /// parent's less its two children's (zero where three hollows meet at one node).
    pub own_deficit: u128,
    /// ★ THE WATERSHED'S SUPPLY, mm·m²/yr (2026-09-24, ruling B7): `Σ Q · area` over every node
    /// whose descent on the ground ends at a pit of this subtree.
    pub supply: u128,
    /// The runoff of the subtree's OWN nodes, mm·m²/yr — what the budget read as the supply before
    /// ruling B7, kept so an instrument can set the two side by side.
    pub brim_supply: u128,
    /// ★ THE GATE: the node where the water from the other side of the spill enters this depression
    /// — its steepest lower neighbour of the spill node inside it, ★ or, on the side the spill node's
    /// own water descends into, that very neighbour (plan step 2b: on a flat every side ties);
    /// [`NO_NODE`] where it never met another.
    pub gate: u32,
    /// ★ THE WATER THE SUBTREE HOLDS after the cascade, mm·m²/yr: its own ring's share plus its
    /// children's, never more than [`Depression::deficit`].
    pub held: u128,
    /// Whether its inflow reached its deficit and the water ran on over its spill.
    pub full: bool,
    /// The water's level, sixteenths, or [`DRY`].
    pub level: i32,
    /// ★ THE SADDLE it closed at: an index into [`Lakes::saddles`], or [`NO_SADDLE`] where it
    /// never closed (plan step 2b).
    pub saddle: u32,
    /// ★ THE WATER IT PASSED OVER ITS SPILL after the cascade, mm·m²/yr: into a side of its
    /// saddle or into the sea (plan step 2b). Zero where it is not full. This is the CASCADE'S
    /// count, never the row: where a full hollow pours into a sibling at the same saddle and the
    /// sibling fills later, the sibling passes that water on again, so both hollows count it. The
    /// row at a spill node is [`MacroSolve::discharge`], which counts each drop once.
    ///
    /// **Example.** Three hollows meet at one notch on the belt's shoulder. The first fills and
    /// pours into the second; later the second fills and passes the first one's water on into the
    /// third with its own. The first counts its water once, the second counts both waters, and
    /// the river at the notch carries both waters once.
    pub passed: u128,
    /// ★ THE EXIT of a full depression (plan step 2b): the gate of the side its water leaves by
    /// ([`open_side`] over the final flags); [`NO_NODE`] where it is not full or every side of its
    /// saddle is full and the water rises into the ring above.
    pub exit: u32,
}

/// No saddle: the depression never closed.
pub const NO_SADDLE: u32 = u32::MAX;

/// ★ ONE SIDE OF A SADDLE (plan step 2b): a component the saddle node met when the sweep reached
/// it — a hollow, or the ocean's own component — and the saddle node's steepest lower neighbour in
/// it, which is where water from the other sides ENTERS it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Side {
    /// The component's root when the saddle closed it: a depression, or [`OCEAN`].
    pub root: u32,
    /// The saddle node's steepest lower neighbour inside the component.
    pub gate: u32,
    /// The drop to the gate, sixteenths, and the chord to it, whole metres: the steepness the
    /// sides are sorted by.
    pub dz: i64,
    pub chord: u32,
    /// Where the water that enters by the gate ends on the ground: the leaf whose pit the gate's
    /// descent ends at, or [`OCEAN`] where it ends at a root.
    pub entry: u32,
}

/// ★ A SADDLE (plan step 2b): a node where two or more components met, with every side it met,
/// STEEPEST FIRST by [`steeper`] — the order the water at the node chooses its way by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Saddle {
    pub node: u32,
    pub sides: Vec<Side>,
}

impl Depression {
    /// The hollow's own volume under its spill, m³ — what it would hold if it were brim full.
    #[must_use]
    pub fn capacity_m3(&self) -> u128 {
        let brim = i128::from(self.spill_z) * (self.area_m2 as i128) - self.z_area;
        (brim.max(0) as u128) / (Z_STEPS_PER_M as u128)
    }
}

/// What the budget found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LakeReport {
    /// The depressions in the hierarchy, the ocean's root left out.
    pub depressions: usize,
    /// Of those, the leaves — one per pit.
    pub leaves: usize,
    /// The water bodies: a depression whose water no ancestor covers.
    pub bodies: usize,
    /// Of the bodies, those that hold no water at all (a playa).
    pub dry: usize,
    /// Of the bodies, those standing under their spill (a closed lake).
    pub partial: usize,
    /// Of the bodies, those standing AT their spill (a lake with an outlet).
    pub spilling: usize,
    /// The nodes under standing fresh water.
    pub lake_nodes: usize,
    /// Their area, m².
    pub lake_area_m2: u128,
    /// The standing water's volume, m³ — the number the sea's inventory must not count twice.
    pub volume_m3: u128,
    /// ★ THE CASCADE'S BOOKS (2026-09-24, ruling B7), mm·m²/yr, EXACT:
    /// `runoff_total = evaporated_total + to_ocean + stranded`. The runoff of every node that is
    /// not the ocean.
    pub runoff_total: u128,
    /// The water the hollows hold, which their open surface hands to the air.
    pub evaporated_total: u128,
    /// The water that reached the ocean: down the ground, or over a full hollow's spill.
    pub to_ocean: u128,
    /// The water that filled a hollow that never met another and had nowhere to go.
    pub stranded: u128,
    /// How many times a full hollow's overflow poured into another hollow's side of its saddle.
    pub transfers: usize,
    /// ★ How many times an overflow poured into the ocean's side of its saddle and the ground led
    /// it into a lower hollow that met the sea earlier (plan step 2b, question 1).
    pub ocean_pours: usize,
    /// ★ How many pours left a saddle of three sides or more (plan step 2b, question 2).
    pub saddle_pours: usize,
    /// ★ How many rings the cascade's walk visited over every arrival (review finding 6 of plan
    /// step 2): the measure of the skip [`cascade`] keeps over full rings.
    pub cascade_steps: usize,
}

/// ★ THE WATER'S OWN WAY ON THE GROUND (2026-09-24, ruling B7; Fill–Spill–Merge's flow
/// directions on the unfilled field): each node's steepest lower neighbour and where that descent
/// ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descent {
    /// The steepest neighbour lower in the budget's own sweep order, ★ or on a FLAT the flat step
    /// toward its drain or its middle (plan step 2b); [`NO_NODE`] at a pit and at a root node.
    pub down: Vec<u32>,
    /// Where the descent stops: a ROOT node (the ocean, or the six lowest nodes of a body with no
    /// sea), or a pit over it — which is the pit of exactly one LEAF of the hierarchy.
    pub end: Vec<u32>,
}

/// The budget's answer: one water level a node, and the hierarchy that decided it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lakes {
    /// The standing water's level at each node, sixteenths, or [`DRY`]. The sea is NOT in here:
    /// [`MacroSolve::water_level`] reads the sea from the ocean's own mask.
    pub water: Vec<i32>,
    /// The hierarchy, index 0 the ocean's root.
    pub deps: Vec<Depression>,
    /// The depression each node was added to.
    pub home: Vec<u32>,
    /// For each depression, the water body that covers it (itself, or an ancestor that floods over
    /// it); [`NO_DEP`] for the ocean.
    pub body_of: Vec<u32>,
    /// The water's way down the ground, which decided each hollow's watershed.
    pub descent: Descent,
    /// ★ Every saddle of the sweep, in the order the sweep met them (plan step 2b).
    pub saddles: Vec<Saddle>,
    pub report: LakeReport,
}

/// The open component a depression belongs to now, with path halving.
fn find(link: &mut [u32], mut d: u32) -> u32 {
    while link[d as usize] != d {
        let up = link[d as usize];
        link[d as usize] = link[up as usize];
        d = link[d as usize];
    }
    d
}

/// A leaf depression opened at node `v`.
fn leaf(v: u32, z_v: i32) -> Depression {
    Depression {
        pit: v,
        pit_z: z_v,
        spill_z: i32::MAX,
        spill_node: NO_NODE,
        parent: NO_DEP,
        children: [NO_DEP, NO_DEP],
        area_m2: 0,
        z_area: 0,
        deficit: 0,
        own_deficit: 0,
        supply: 0,
        brim_supply: 0,
        gate: NO_NODE,
        held: 0,
        full: false,
        level: DRY,
        saddle: NO_SADDLE,
        passed: 0,
        exit: NO_NODE,
    }
}

/// ★ THE ROOT: every node of the CONNECTED OCEAN ([`MacroSolve::ocean`]; 2026-09-24, ruling B7 item
/// 1), or — on a body whose water settled no ocean — the same six lowest nodes the routing takes
/// ([`crate::solve::OUTLETS_WITHOUT_SEA`]), so the hierarchy and the receivers agree about where the water
/// leaves. Returns the root mask, THE FLATS of the ground ([`flats`]), and every other node in the
/// SWEEP'S ORDER: lowest first by `(height, potential on its flat, index)`.
///
/// ★ THE FLAT'S DISTANCE IN THE ORDER (plan step 2b, question 3). By `(height, index)` alone a flat
/// was swept in index order, so a node of a flat that drains could be reached before any neighbour
/// of it was, and it opened a leaf of ZERO depth that ate the water running over it. MEASURED at
/// generator 20: 8 795 such leaves, 462.7 km³/yr held on flats that drain.
fn root_and_order(state: &MacroSolve) -> (Vec<bool>, Flats, Vec<u32>) {
    let n = state.node_count();
    let z = &state.z;
    let threads = state.cores.threads();
    let mut root = state.ocean.clone();
    if !root.contains(&true) {
        for v in crate::solve::outlets_without_sea(z) {
            root[v as usize] = true;
        }
    }
    let flats = flats(state, &root);
    // ★ THE SWEEP'S ORDER ON THE SOLVE'S CORES (ruling B72 item 1): every node's key is its height (its sign bit
    // flipped, so the words order as the heights do), its potential and its index in one word — unique, so the cores'
    // sort is the one order one core's sort gives ([`crate::par::sort_keys`]).
    let keys: Vec<u128> = crate::par::join(crate::par::per_range(n, threads, |r| {
        r.filter(|&v| !root[v])
            .map(|v| {
                (u128::from((z[v] as u32) ^ 0x8000_0000) << 96)
                    | (u128::from(flats.phi[v]) << 32)
                    | v as u128
            })
            .collect::<Vec<u128>>()
    }));
    let order = crate::par::sort_keys(keys, threads)
        .into_iter()
        .map(|k| k as u32)
        .collect();
    (root, flats, order)
}

/// ★ THE FLATS OF THE GROUND (plan step 2b, question 3; Barnes, Lehman & Mulla 2014, *An efficient
/// assignment of drainage direction over flat surfaces in raster digital elevation models*, Computers
/// & Geosciences 62, 128–135, <https://doi.org/10.1016/j.cageo.2013.01.009> — its "toward lower
/// ground" field, measured here in the lattice's own chord). A FLAT is a connected set of nodes of
/// ONE exact height that are not the root; the deposit leaves one wherever it fills a pit to its
/// spill, and the envelope where it clamps.
///
/// - A flat that DRAINS has nodes with a lower neighbour or a root neighbour: its DRAINS. Every node
///   of the flat takes the summed chord to its nearest drain, and heads for that drain.
/// - A CLOSED flat (every neighbour off it stands higher) is the floor of one pit: its MIDDLE — the
///   node farthest from its rim by the summed chord, ties to the smaller index — is the pit, and
///   every node takes the summed chord to it.
///
/// A node off every flat reads zero and heads nowhere.
///
/// ★ BARNES' TWO FIELDS AS ONE POTENTIAL (phase 1 of the ground step, design §5.1; rulings B9 item 2,
/// B52 item 2). The flat's water no longer aims at ONE target — its nearest drain — whose straight
/// line on a flat of node squares runs along the lattice's rows (MEASURED, generator 22: the flats'
/// trunks read 1.470 against ~1.006). Each node of the flat takes two summed chords: `T`, to its
/// nearest drain (the middle on a closed flat), and `A`, to the nearest node of the flat that touches
/// HIGHER ground; the potential is `Φ = 2·T + (A_max − A)`, `A_max` the flat's largest `A` — Barnes'
/// "toward lower" field twice and "away from higher" once, in metres of chord. Its water steps down
/// `Φ`'s own gradient ([`downhill`], [`step_along`]). Every node of the flat off a drain has a
/// neighbour of smaller `Φ`: the step toward its nearest drain shortens `T` by its chord `c` and
/// moves `A` by at most `c`, so `Φ` falls by at least `c` — so the water always reaches a drain, and
/// the flat holds no false pit. A flat that touches no higher ground has `A_max − A = 0`.
///
/// **Example.** A crater bowl the sediment filled to its sill is a flat 40 km across at the sill's
/// height. Its water no longer runs to the bowl's south-west corner by the node index, nor along the
/// rim's straight edge to the nearest point of the sill: it leaves the walls first, gathers along the
/// bowl's middle and runs out through the sill.
struct Flats {
    /// `T`: the summed chord to the node's drain or middle, whole metres; zero off the flats and at a
    /// drain or a middle.
    dist: Vec<u64>,
    /// The drain or the middle the node's water heads for; [`NO_NODE`] off the flats.
    toward: Vec<u32>,
    /// `Φ = 2·T + (A_max − A)`, whole metres; zero off the flats and at a closed flat's middle.
    phi: Vec<u64>,
}

/// The flats of `state` over the stated root ([`Flats`]).
fn flats(state: &MacroSolve, root: &[bool]) -> Flats {
    let n = state.node_count();
    let z = &state.z;
    let mut out = Flats {
        dist: vec![0; n],
        toward: vec![NO_NODE; n],
        phi: vec![0; n],
    };
    // `A` and its scratch sources, over every flat in turn.
    let mut away = vec![0u64; n];
    let mut away_from = vec![NO_NODE; n];
    let mut seen = vec![false; n];
    let mut members: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    let on_flat = |m: u32, z0: i32| (m != NO_NODE) && !root[m as usize] && z[m as usize] == z0;
    // ★ THE NODES THAT START A FLAT, found ON THE SOLVE'S CORES (ruling B72 item 1): a node off the root with a
    // neighbour of its own height; each range's finds join in index order, so the flats are met in the order one core
    // meets them.
    let starts: Vec<u32> = crate::par::per_range(n, state.cores.threads(), |r| {
        r.filter(|&v| {
            !root[v]
                && state
                    .lattice
                    .neighbours(v as u32)
                    .iter()
                    .any(|&m| on_flat(m, z[v]))
        })
        .map(|v| v as u32)
        .collect::<Vec<u32>>()
    })
    .into_iter()
    .flatten()
    .collect();
    for start in starts {
        let z0 = z[start as usize];
        if seen[start as usize] {
            continue;
        }
        members.clear();
        seen[start as usize] = true;
        stack.push(start);
        while let Some(v) = stack.pop() {
            members.push(v);
            for m in state.lattice.neighbours(v) {
                if on_flat(m, z0) && !seen[m as usize] {
                    seen[m as usize] = true;
                    stack.push(m);
                }
            }
        }
        // The higher edge: a node with a higher neighbour; `A` is the summed chord from it.
        let rim: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&v| {
                state
                    .lattice
                    .neighbours(v)
                    .iter()
                    .any(|&m| (m != NO_NODE) && (z[m as usize] > z0))
            })
            .collect();
        spread(state, &mut away, &mut away_from, &members, &rim, z0, root);
        // The drains: a node with a lower neighbour or a root neighbour.
        let mut sources: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&v| {
                state
                    .lattice
                    .neighbours(v)
                    .iter()
                    .any(|&m| (m != NO_NODE) && (root[m as usize] | (z[m as usize] < z0)))
            })
            .collect();
        if sources.is_empty() {
            // A closed flat: its middle, the node farthest from its rim (a closed flat's every
            // neighbour off it stands higher, so its rim is never empty).
            let middle = members
                .iter()
                .copied()
                .max_by_key(|&v| (away[v as usize], std::cmp::Reverse(v)))
                .expect("a flat has members");
            sources = vec![middle];
        }
        spread(
            state,
            &mut out.dist,
            &mut out.toward,
            &members,
            &sources,
            z0,
            root,
        );
        // Barnes' two fields as one potential; a flat that touches no higher ground reads `A` as
        // unreached everywhere and adds nothing.
        let a_max = members
            .iter()
            .map(|&v| away[v as usize])
            .filter(|&a| a != u64::MAX)
            .max()
            .unwrap_or(0);
        for &v in &members {
            let a = away[v as usize];
            let lift = if a == u64::MAX { 0 } else { a_max - a };
            out.phi[v as usize] = 2 * out.dist[v as usize] + lift;
        }
    }
    out
}

/// The summed chord from `sources` over the flat `members` at height `z0` (a Dijkstra keyed by
/// `(distance, node)`, so no visit order decides it) into `dist`, and the source each node heads
/// for into `toward`; a member no source reaches reads `u64::MAX` (an empty `sources`).
fn spread(
    state: &MacroSolve,
    dist: &mut [u64],
    toward: &mut [u32],
    members: &[u32],
    sources: &[u32],
    z0: i32,
    root: &[bool],
) {
    let z = &state.z;
    for &v in members {
        dist[v as usize] = u64::MAX;
    }
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for &s in sources {
        dist[s as usize] = 0;
        toward[s as usize] = s;
        heap.push(Reverse((0, s)));
    }
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        for m in state.lattice.neighbours(u) {
            if (m == NO_NODE) || root[m as usize] || (z[m as usize] != z0) {
                continue;
            }
            let step = d + u64::from(state.chord_m(u, m));
            if step < dist[m as usize] {
                dist[m as usize] = step;
                toward[m as usize] = toward[u as usize];
                heap.push(Reverse((step, m)));
            }
        }
    }
}

/// ★ THE DOWNHILL OF A FIELD at node `i` (phase 1 of the ground step; ruling B58 for the ground's own height): the
/// central difference over `ring`, the node's neighbours with their values — a flat's potential, or the ground's height
/// in sixteenths — each neighbour's offset weighted by the field's drop to it over the offset's squared length. On
/// the lattice's symmetric stencil the sum points EXACTLY down a potential that falls evenly in any
/// one direction (each row and each diagonal pair adds twice its share), so no direction of the
/// lattice is preferred. Scaled so its largest component stays under 2²⁴, where [`step_along`]'s
/// squared offsets fit their words; zero where the ring is flat.
///
/// **Example.** On a basin floor whose potential falls 3 m a kilometre toward the north-north-east,
/// the eight neighbours' drops weigh their offsets so the sum points north-north-east, between the
/// row and the diagonal, and the split then takes each about as often as that bearing asks.
fn downhill(state: &MacroSolve, i: u32, phi_i: i64, ring: &[(u32, i64)]) -> [i128; 3] {
    let at = state.dir_of(i);
    let mut t = [0i128; 3];
    for &(m, phi_m) in ring {
        let d = state.dir_of(m);
        let e = [
            i128::from(d[0]) - i128::from(at[0]),
            i128::from(d[1]) - i128::from(at[1]),
            i128::from(d[2]) - i128::from(at[2]),
        ];
        // The weight at 64 fraction bits: a drop under 2²⁵ m stays under 2⁸⁹, and over the
        // smallest squared offset of THE world's lattices (2³⁵, the largest body's 8 km node at
        // 30 bits) the eight terms stay under 2⁷⁴.
        let e2 = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).max(1);
        let w = ((i128::from(phi_i) - i128::from(phi_m)) << 64) / e2;
        t[0] += w * e[0];
        t[1] += w * e[1];
        t[2] += w * e[2];
    }
    let big = t[0]
        .unsigned_abs()
        .max(t[1].unsigned_abs())
        .max(t[2].unsigned_abs());
    let shift = (128 - big.leading_zeros()).saturating_sub(24);
    [t[0] >> shift, t[1] >> shift, t[2] >> shift]
}

/// ★ THE FLAT STEP (plan step 2b, question 4): from node `i` toward `target`, among `cands` — every
/// neighbour nearer the target by the flat's summed chord, with that neighbour's own distance plus
/// the chord to it. Returns the step and how it was decided: 0 no neighbour lies ahead (the nearest
/// by distance plus chord, ties to the smaller index — the rule before this step), 1 one lies ahead,
/// 2 THE SPLIT between the two that lie nearest the straight line to the target.
///
/// THE SPLIT is Rho8's (Fairfield & Leymarie 1991, *Drainage networks from grid digital elevation
/// models*, Water Resources Research 27, 709–717, <https://doi.org/10.1029/90WR02658>): the two
/// neighbours bracket the line, their offsets from it are `p1 ≤ p2`, and the step takes the FARTHER
/// one with the chance `p1 / (p1 + p2)`. On a row and its diagonal at a bearing `θ` that chance is
/// `tan θ`, so the expected step points EXACTLY along the line and a flat's water leaves no grid
/// direction behind (T9: a NAMED SCATTER around a computed direction, never a drawn number; the
/// chance is the node's own hash under [`crate::body::salt`]'s `FLOW` stream). The nearest-angle rule
/// with no chance runs straight along a row while the bearing stays under 22.5°.
///
/// **Example.** A lake fills a round hollow on the plain and spills at a notch 20° off the face's
/// row. From the far shore every step goes a row or a diagonal; about one step in three takes the
/// diagonal, so the river across the lake bends neither along the row nor along the diagonal but
/// runs straight to the notch.
fn flat_step(state: &MacroSolve, i: u32, target: u32, cands: &[(u32, u64)]) -> (u32, u8) {
    let at = state.dir_of(i);
    let to = state.dir_of(target);
    let t = [
        i128::from(to[0]) - i128::from(at[0]),
        i128::from(to[1]) - i128::from(at[1]),
        i128::from(to[2]) - i128::from(at[2]),
    ];
    step_along(state, i, t, cands)
}

/// ★ THE STEP ALONG A BEARING (the flat step's own rule, [`flat_step`], for any line `t` from node
/// `i`, in the cached directions' units): among `cands`, the nearest by its cost when none lies
/// ahead (0), the one ahead (1), or Rho8's split between the two ahead nearest the line (2).
fn step_along(state: &MacroSolve, i: u32, t: [i128; 3], cands: &[(u32, u64)]) -> (u32, u8) {
    let at = state.dir_of(i);
    let mut near = (u64::MAX, NO_NODE);
    let mut ahead = [(u128::MAX, NO_NODE); 2];
    let mut count = 0usize;
    for &(m, cost) in cands {
        near = near.min((cost, m));
        let d = state.dir_of(m);
        let e = [
            i128::from(d[0]) - i128::from(at[0]),
            i128::from(d[1]) - i128::from(at[1]),
            i128::from(d[2]) - i128::from(at[2]),
        ];
        if e[0] * t[0] + e[1] * t[1] + e[2] * t[2] <= 0 {
            continue;
        }
        let c = [
            e[1] * t[2] - e[2] * t[1],
            e[2] * t[0] - e[0] * t[2],
            e[0] * t[1] - e[1] * t[0],
        ];
        let off = (c[0] * c[0]) as u128 + (c[1] * c[1]) as u128 + (c[2] * c[2]) as u128;
        let key = (off, m);
        count += 1;
        if key < ahead[0] {
            ahead[1] = ahead[0];
            ahead[0] = key;
        } else if key < ahead[1] {
            ahead[1] = key;
        }
    }
    match count {
        0 => (near.1, 0),
        1 => (ahead[0].1, 1),
        _ => (split(state.seed, i, ahead[0], ahead[1]), 2),
    }
}

/// ★ THE SPLIT between the nearer and the farther neighbour ahead, by their squared offsets from
/// the line: the offsets are the roots of the squares after one stated even shift that keeps the
/// larger square inside a 64-bit word, and the node's own hash (32 bits) takes the farther with the
/// chance `p1 / (p1 + p2)`.
fn split(seed: u64, i: u32, near: (u128, u32), far: (u128, u32)) -> u32 {
    let bits = 128 - far.0.leading_zeros();
    let shift = (bits.saturating_sub(62) + 1) & !1;
    let p1 = u128::from(isqrt((near.0 >> shift) as u64));
    let p2 = u128::from(isqrt((far.0 >> shift) as u64));
    let h = u128::from(child_seed(seed, crate::body::salt::FLOW, u64::from(i)) >> 32);
    if h * (p1 + p2) < p1 << 32 {
        far.1
    } else {
        near.1
    }
}

/// ★ THE DESCENT OVER THE GROUND for every node, with the root stated: ★ each node that is not a root
/// takes the neighbour lower in the sweep's order that the ground's own downhill bearing reaches by
/// Rho8's seeded split (ruling B58: no straight, 45° or staircased rivers; [`downhill`] over the
/// heights, [`step_along`]), and its steepest lower neighbour (the solve's own [`steeper`] over the
/// lattice's chord) where none lies ahead of that bearing; ★ a node of a FLAT steps down the flat's potential
/// ([`Flats`], [`downhill`], [`step_along`]; plan step 2b, question 3; phase 1 of the ground step);
/// and the descent ends at a root or at a node
/// with no lower neighbour — a PIT, which opened exactly one leaf of the hierarchy.
///
/// **Example.** Rain on a spur of the belt's range runs down the ground, never down the fill. The
/// descent follows it into the hollow at the range's foot and stops at the hollow's pit, even where
/// the routing fill carries the same rain over the hollow's brim to the sea.
#[must_use]
pub fn descent(state: &MacroSolve) -> Descent {
    let (_, flats, order) = root_and_order(state);
    descent_over(state, &order, &flats)
}

/// The descent over `order` (every node that is not a root, in the sweep's order): a root keeps
/// [`NO_NODE`] as its way down and itself as its end. A node lower in the sweep's order is one the
/// sweep reached first, so the descent and the hierarchy name the same pits.
fn descent_over(state: &MacroSolve, order: &[u32], flats: &Flats) -> Descent {
    let n = state.node_count();
    // ★ EACH NODE'S WAY DOWN ON THE SOLVE'S CORES (ruling B64): it reads only the ground and the flats, never another
    // node's way, so the nodes of `order` are cut into ranges, each computed by its own thread, and joined in order.
    let picks: Vec<u32> = crate::par::join(crate::par::per_range(
        order.len(),
        state.cores.threads(),
        |r| {
            let mut cands: Vec<(u32, u64)> = Vec::with_capacity(8);
            let mut ring: Vec<(u32, i64)> = Vec::with_capacity(8);
            order[r]
                .iter()
                .map(|&v| way_down(state, flats, v, &mut cands, &mut ring))
                .collect::<Vec<u32>>()
        },
    ));
    // Where each descent ends, in the sweep's order (a node's way down comes before it).
    let mut down = vec![NO_NODE; n];
    let mut end: Vec<u32> = (0..n as u32).collect();
    for (&v, &best) in order.iter().zip(&picks) {
        let iv = v as usize;
        down[iv] = best;
        if best != NO_NODE {
            end[iv] = end[best as usize];
        }
    }
    Descent { down, end }
}

/// One node's way down (see [`descent_over`]): the neighbour its descent takes, or [`NO_NODE`] at a pit; `cands` and
/// `ring` are the caller's scratch.
fn way_down(
    state: &MacroSolve,
    flats: &Flats,
    v: u32,
    cands: &mut Vec<(u32, u64)>,
    ring: &mut Vec<(u32, i64)>,
) -> u32 {
    let z = &state.z;
    let iv = v as usize;
    let key = (z[iv], flats.phi[iv], v);
    let mut best = NO_NODE;
    if flats.dist[iv] > 0 {
        cands.clear();
        ring.clear();
        for m in state.lattice.neighbours(v) {
            if m == NO_NODE {
                continue;
            }
            let mi = m as usize;
            // The same flat (a flat node of one height, never a root); a candidate is lower in
            // the flat's potential.
            if (flats.toward[mi] != NO_NODE) & (z[mi] == z[iv]) {
                ring.push((m, flats.phi[mi] as i64));
                if flats.phi[mi] < flats.phi[iv] {
                    cands.push((m, flats.phi[mi] + u64::from(state.chord_m(v, m))));
                }
            }
        }
        let t = downhill(state, v, flats.phi[iv] as i64, ring);
        best = step_along(state, v, t, cands).0;
    } else {
        // ★ RHO8 ON THE GROUND'S OWN SLOPE (ruling B58: no straight, 45° or staircased rivers): the bearing down the
        // ground from all its neighbours ([`downhill`] over the heights), and the lower neighbour that bearing
        // reaches by Rho8's seeded split between the two that bracket it ([`step_along`]); where no lower
        // neighbour lies ahead of the bearing (a ridge's crest, a saddle), the steepest lower neighbour.
        cands.clear();
        ring.clear();
        let (mut dz_best, mut l_best) = (0i64, 0u32);
        for m in state.lattice.neighbours(v) {
            if m == NO_NODE {
                continue;
            }
            ring.push((m, i64::from(z[m as usize])));
            if (z[m as usize], flats.phi[m as usize], m) >= key {
                continue;
            }
            cands.push((m, 0));
            let dz = i64::from(z[iv]) - i64::from(z[m as usize]);
            let l = state.chord_m(v, m);
            if best == NO_NODE || steeper(dz, l, m, dz_best, l_best, best) {
                best = m;
                dz_best = dz;
                l_best = l;
            }
        }
        if best != NO_NODE {
            let t = downhill(state, v, i64::from(z[iv]), ring);
            let (pick, ahead) = step_along(state, v, t, cands);
            if ahead > 0 {
                best = pick;
            }
        }
    }
    best
}

/// ★ WHAT THE SWEEP MEETS AT EACH NODE (ruling B72 item 1): for every node of the sweep's order, the
/// neighbours done before it — a root, or a node earlier in the order — in the stencil's order, each
/// with the node's drop to it and the chord between them, and the node's two rates ([`rates`]). None
/// of it reads what the sweep makes, so the solve's cores compute it first, each a range of the
/// order, and the sweep, which must go one node after another, reads it in order.
///
/// **Example.** A node on the belt's flank has five neighbours lower than itself and three higher:
/// the cores list the five with their drops and chords before the sweep starts, and the sweep only
/// asks which hollow each of the five belongs to now.
struct Before {
    /// Every node's place in the order — a root's the order's length, where the sweep keeps the
    /// root's water.
    slot: Vec<u32>,
    /// Where each node's neighbours start in `met`, and one more: the end.
    at: Vec<usize>,
    /// The neighbours met: the node, its place, the drop to it (sixteenths) and the chord (whole
    /// metres).
    met: Vec<(u32, u32, i32, u32)>,
    /// The place of each node's own way down ([`Descent::down`]; `u32::MAX` at a pit).
    down: Vec<u32>,
    /// Each node's supply and deficit.
    rate: Vec<(u128, u128)>,
}

impl Before {
    /// The lists for `order` over the `root` and the `descent` (see [`Before`]).
    fn of(
        state: &MacroSolve,
        climate: &Climate,
        root: &[bool],
        order: &[u32],
        descent: &Descent,
    ) -> Before {
        let z = &state.z;
        let threads = state.cores.threads();
        let mut slot = vec![order.len() as u32; state.node_count()];
        for (k, &v) in order.iter().enumerate() {
            slot[v as usize] = k as u32;
        }
        let parts = crate::par::per_range(order.len(), threads, |r| {
            let mut count = Vec::with_capacity(r.len());
            let mut met = Vec::new();
            let mut down = Vec::with_capacity(r.len());
            let mut rate = Vec::with_capacity(r.len());
            for k in r {
                let v = order[k];
                let iv = v as usize;
                let first = met.len();
                for m in state.lattice.neighbours(v) {
                    if (m != NO_NODE) && (root[m as usize] | (slot[m as usize] < k as u32)) {
                        let mi = m as usize;
                        met.push((m, slot[mi], z[iv] - z[mi], state.chord_m(v, m)));
                    }
                }
                count.push(met.len() - first);
                let way = descent.down[iv];
                down.push(if way == NO_NODE {
                    u32::MAX
                } else {
                    slot[way as usize]
                });
                rate.push(rates(state, climate, iv));
            }
            (count, met, down, rate)
        });
        let mut at = Vec::with_capacity(order.len() + 1);
        at.push(0);
        for (count, _, _, _) in &parts {
            for &c in count {
                at.push(at[at.len() - 1] + c);
            }
        }
        let (mut met, mut down, mut rate) = (Vec::new(), Vec::new(), Vec::new());
        for (_, mut m, mut d, mut r) in parts {
            met.append(&mut m);
            down.append(&mut d);
            rate.append(&mut r);
        }
        Before {
            slot,
            at,
            met,
            down,
            rate,
        }
    }

    /// The neighbours met at the `k`-th node of the order.
    fn met(&self, k: usize) -> &[(u32, u32, i32, u32)] {
        &self.met[self.at[k]..self.at[k + 1]]
    }
}

/// One root met at a node of the sweep, and its steepest lower neighbour there: the gate the
/// water from the other side enters it by.
#[derive(Clone, Copy)]
struct Met {
    root: u32,
    gate: u32,
    dz: i64,
    chord: u32,
}

/// ★ THE BUDGET over a solved state and the climate that stands over it. The state's OWN field
/// decides the hierarchy — never the routing flood, which is a scratch surface. A body with no
/// water gets no lake: every hollow on it is a closed basin.
///
/// **Example.** The home planet's solve ends, the sea is settled, and the budget walks 8.87 million
/// nodes once: it finds the hollows, hands each one the runoff of its whole watershed, passes every
/// full hollow's overflow on, and says which of them hold water. What it writes is one level a
/// node, and the artifact's water word is that level.
#[must_use]
pub fn budget(state: &MacroSolve, climate: &Climate) -> Lakes {
    let n = state.node_count();
    let z = &state.z;
    let (root, flats, order) = root_and_order(state);
    let mut deps = vec![leaf(NO_NODE, i32::MAX)];
    deps[0].spill_z = i32::MAX;
    let mut link = vec![OCEAN];
    let mut saddles: Vec<Saddle> = Vec::new();
    // 0. THE WATER'S WAY DOWN THE GROUND, first: it reads the sweep's order and the flats and
    //    nothing the sweep makes, and the sweep's saddles read it.
    let descent = descent_over(state, &order, &flats);
    drop(flats);
    let before = Before::of(state, climate, &root, &order, &descent);
    // 1. THE SWEEP, lowest first: the hierarchy, and at every merge the gate of each side. ★ Each
    //    node's depression is kept BY ITS PLACE IN THE ORDER (ruling B72 item 1): the sweep writes
    //    them one after another and reads its neighbours' from one compact list — the root's water
    //    at the list's end — and every node's own is gathered from it once the sweep ends.
    let mut held = vec![NO_DEP; order.len() + 1];
    held[order.len()] = OCEAN;
    let mut met: Vec<Met> = Vec::with_capacity(8);
    for (k, &v) in order.iter().enumerate() {
        let iv = v as usize;
        let z_v = z[iv];
        met.clear();
        for &(m, slot, dz, chord) in before.met(k) {
            let r = find(&mut link, held[slot as usize]);
            let dz = i64::from(dz);
            match met.iter_mut().find(|k| k.root == r) {
                Some(k) => {
                    if steeper(dz, chord, m, k.dz, k.chord, k.gate) {
                        (k.gate, k.dz, k.chord) = (m, dz, chord);
                    }
                }
                None => met.push(Met {
                    root: r,
                    gate: m,
                    dz,
                    chord,
                }),
            }
        }
        met.sort_unstable_by_key(|k| k.root);
        let holder = match met.len() {
            0 => {
                let d = deps.len() as u32;
                deps.push(leaf(v, z_v));
                link.push(d);
                d
            }
            1 => met[0].root,
            _ => {
                // ★ THE SADDLE (plan step 2b): every side the node met, the side the node's own
                //   water descends into FIRST — through the very neighbour it descends to — and
                //   the rest steepest first. Off a flat the descent IS the steepest way; on a flat
                //   every side ties at no drop, and the node's water takes the flat step.
                let way = descent.down[iv];
                let r = find(&mut link, held[before.down[k] as usize]);
                for k in met.iter_mut().filter(|k| k.root == r) {
                    (k.gate, k.dz, k.chord) = (
                        way,
                        i64::from(z_v) - i64::from(z[way as usize]),
                        state.chord_m(v, way),
                    );
                }
                let at = saddles.len() as u32;
                saddles.push(saddle_of(v, &met, way));
                let mut acc = met[0];
                for &other in met.iter().skip(1) {
                    let p = merge(&mut deps, &mut link, acc, other, v, z_v, at);
                    // The gate of the merged parent at THIS node: the steeper of the two, for a
                    // third component that meets them here.
                    let side = if steeper(
                        other.dz,
                        other.chord,
                        other.gate,
                        acc.dz,
                        acc.chord,
                        acc.gate,
                    ) {
                        other
                    } else {
                        acc
                    };
                    acc = Met { root: p, ..side };
                }
                acc.root
            }
        };
        add_cell(&mut deps, holder, z_v, state.area[iv], before.rate[k]);
        held[k] = holder;
    }
    let home: Vec<u32> =
        crate::par::map(n, state.cores.threads(), |v| held[before.slot[v] as usize]);
    drop(held);
    drop(before);
    // ★ THE BRIM (plan step 2b): a node that stands at EXACTLY its depression's spill height is at
    //   the water's edge when the depression is full, never under it (the level word covers
    //   `z < level` only), so it costs the depression nothing: its deficit is the ring above's,
    //   where a higher level covers it. Before the flats had an order, few nodes stood at a spill's
    //   own height; a flat swept from its drains leaves every node of it that drains into a hollow
    //   at that hollow's spill height, and charged the hollow for water that never stands there.
    //   `ties` holds, per depression, the deficit of every node of its subtree at its spill height:
    //   its own ring's, and a child's that closed at the same height.
    let mut ties = vec![0u128; deps.len()];
    for i in 0..n {
        let h = home[i];
        if (h != OCEAN) && (z[i] == deps[h as usize].spill_z) {
            ties[h as usize] += rates(state, climate, i).1;
        }
    }
    for d in 1..deps.len() {
        for c in deps[d].children {
            if (c != NO_DEP) && (deps[c as usize].spill_z == deps[d].spill_z) {
                ties[d] += ties[c as usize];
            }
        }
        deps[d].deficit -= ties[d];
    }
    drop(ties);
    for d in 1..deps.len() {
        let kids = deps[d].children;
        deps[d].own_deficit = deps[d].deficit
            - if kids[0] == NO_DEP {
                0
            } else {
                deps[kids[0] as usize].deficit + deps[kids[1] as usize].deficit
            };
    }
    // 2. THE WATERSHED: every node that is not the ocean books its runoff to the leaf its descent
    //    on the ground ends in, or to the ocean where it ends at a root.
    let mut shed = vec![0u128; deps.len()];
    let mut report = LakeReport::default();
    for i in 0..n {
        if state.ocean[i] {
            continue;
        }
        let q = u128::from(climate.runoff_mm_yr[i]) * u128::from(state.area[i]);
        report.runoff_total += q;
        let end = descent.end[i] as usize;
        if root[end] {
            report.to_ocean += q;
        } else {
            shed[home[end] as usize] += q;
        }
    }
    for d in 1..deps.len() {
        let kids = deps[d].children;
        deps[d].supply = if kids[0] == NO_DEP {
            shed[d]
        } else {
            deps[kids[0] as usize].supply + deps[kids[1] as usize].supply
        };
    }
    // ★ Each saddle side's ENTRY: the leaf the water that enters by its gate runs down to, or the
    //   ocean where the gate's descent ends at a root (plan step 2b).
    for saddle in &mut saddles {
        for side in &mut saddle.sides {
            let end = descent.end[side.gate as usize] as usize;
            side.entry = if root[end] { OCEAN } else { home[end] };
        }
    }
    // 3. THE CASCADE (Fill–Spill–Merge's overflow, with rates in place of volumes).
    let flow = cascade(&deps, &saddles, &shed);
    report.to_ocean += flow.to_ocean;
    report.stranded = flow.stranded;
    report.transfers = flow.transfers;
    report.ocean_pours = flow.ocean_pours;
    report.saddle_pours = flow.saddle_pours;
    report.cascade_steps = flow.steps;
    report.evaporated_total = flow.held_own.iter().sum();
    for d in 1..deps.len() {
        let kids = deps[d].children;
        let below = if kids[0] == NO_DEP {
            0
        } else {
            deps[kids[0] as usize].held + deps[kids[1] as usize].held
        };
        deps[d].held = flow.held_own[d] + below;
        deps[d].full = flow.full[d];
        deps[d].passed = flow.passed[d];
        // ★ THE EXIT (plan step 2b): the gate of the side the water of a full depression leaves
        //   by, from the final flags — the way the shipped receiver at its spill points.
        deps[d].exit = if flow.full[d] {
            open_side(&saddles[deps[d].saddle as usize], &flow.full).map_or(NO_NODE, |k| k.gate)
        } else {
            NO_NODE
        };
    }
    // 4. THE WATER BODY of every depression: the water that actually stands over it. A depression
    //    that fills hands its water up, so its body is its parent's — unless the parent holds no
    //    water of its own, and then the full depression IS a lake at its spill with a river out of
    //    it. A depression that does NOT fill is a lake only when every child of it fills: while one
    //    child still stands under its own spill the water sits in the children, never in one flat
    //    sheet over both. A parent is always made after its children, so one descending pass
    //    settles every chain.
    let mut body_of = vec![NO_DEP; deps.len()];
    for d in (1..deps.len()).rev() {
        let p = deps[d].parent;
        let kids = deps[d].children;
        let kids_full =
            kids[0] == NO_DEP || (deps[kids[0] as usize].full && deps[kids[1] as usize].full);
        let up = if (p == OCEAN) | (p == NO_DEP) {
            NO_DEP
        } else {
            body_of[p as usize]
        };
        body_of[d] = if deps[d].full && up != NO_DEP {
            up
        } else if deps[d].full || kids_full {
            d as u32
        } else {
            NO_DEP
        };
    }
    // 5. THE LEVELS. A full body stands at its spill; a body under its spill is solved over its own
    //    nodes, lowest first, until the standing water's evaporation eats the water it holds.
    report.depressions = deps.len() - 1;
    report.leaves = deps[1..].iter().filter(|d| d.children[0] == NO_DEP).count();
    for d in 1..deps.len() {
        if body_of[d] != d as u32 {
            continue;
        }
        report.bodies += 1;
        if deps[d].full {
            deps[d].level = deps[d].spill_z;
            report.spilling += 1;
        }
    }
    solve_partial_levels(&mut deps, &body_of, &home, state, climate, &mut report);
    // 6. ONE LEVEL A NODE.
    let mut water = vec![DRY; n];
    for i in 0..n {
        let h = home[i];
        if (h == NO_DEP) | (h == OCEAN) {
            continue;
        }
        let body = body_of[h as usize];
        if body == NO_DEP {
            continue;
        }
        let level = deps[body as usize].level;
        if level != DRY && z[i] < level {
            water[i] = level;
            report.lake_nodes += 1;
            report.lake_area_m2 += u128::from(state.area[i]);
            report.volume_m3 += u128::from((level - z[i]) as u32) * u128::from(state.area[i])
                / (Z_STEPS_PER_M as u128);
        }
    }
    Lakes {
        water,
        deps,
        home,
        body_of,
        descent,
        saddles,
        report,
    }
}

/// ★ THE TWO RATES A NODE STATES, mm·m²/yr: its SUPPLY, the Turc–Pike runoff `Q` over its area;
/// and its DEFICIT, what covering it with water would cost — `Q + PET − P`, the runoff it stops
/// yielding plus the water the open surface hands the air over the rain it catches. Both come from
/// ONE climate, so a node's river and its lake's evaporation cannot disagree.
#[must_use]
fn rates(state: &MacroSolve, climate: &Climate, i: usize) -> (u128, u128) {
    let p = i64::from(climate.rain_mm_yr[i]);
    let e = i64::from(climate.pet_mm_yr[i]);
    let q = i64::from(climate.runoff_mm_yr[i]);
    let a = u128::from(state.area[i]);
    ((q as u128) * a, ((q + e - p).max(0) as u128) * a)
}

/// One node's area, height and two rates into the depression that holds it.
fn add_cell(deps: &mut [Depression], d: u32, z_v: i32, area: u64, rate: (u128, u128)) {
    let dep = &mut deps[d as usize];
    dep.area_m2 += u128::from(area);
    dep.z_area += i128::from(z_v) * i128::from(area);
    dep.brim_supply += rate.0;
    dep.deficit += rate.1;
}

/// Two components meeting at node `v`: the sea takes the other one as a child and stays the root;
/// two hollows make a new parent that holds them both. A closing depression states its spill, its
/// parent and its GATE — where the water from the other side enters it.
fn merge(
    deps: &mut Vec<Depression>,
    link: &mut Vec<u32>,
    a: Met,
    b: Met,
    v: u32,
    z_v: i32,
    saddle: u32,
) -> u32 {
    if a.root == OCEAN {
        close(deps, b, v, z_v, OCEAN, saddle);
        link[b.root as usize] = OCEAN;
        return OCEAN;
    }
    let (ra, rb) = (a.root as usize, b.root as usize);
    let p = deps.len() as u32;
    let (pit, pit_z) = if (deps[ra].pit_z, deps[ra].pit) <= (deps[rb].pit_z, deps[rb].pit) {
        (deps[ra].pit, deps[ra].pit_z)
    } else {
        (deps[rb].pit, deps[rb].pit_z)
    };
    let mut parent = leaf(pit, pit_z);
    parent.children = [a.root, b.root];
    parent.area_m2 = deps[ra].area_m2 + deps[rb].area_m2;
    parent.z_area = deps[ra].z_area + deps[rb].z_area;
    parent.brim_supply = deps[ra].brim_supply + deps[rb].brim_supply;
    parent.deficit = deps[ra].deficit + deps[rb].deficit;
    deps.push(parent);
    link.push(p);
    close(deps, a, v, z_v, p, saddle);
    close(deps, b, v, z_v, p, saddle);
    link[ra] = p;
    link[rb] = p;
    p
}

/// A depression closes at `v`: its spill, its parent, its gate and its saddle.
fn close(deps: &mut [Depression], met: Met, v: u32, z_v: i32, parent: u32, saddle: u32) {
    let dep = &mut deps[met.root as usize];
    dep.spill_z = z_v;
    dep.spill_node = v;
    dep.parent = parent;
    dep.gate = met.gate;
    dep.saddle = saddle;
}

/// ★ THE SADDLE AT `v` (plan step 2b): every component the node met, with its gate, sorted
/// STEEPEST FIRST by [`steeper`] — a stated order over integers, ties to the smaller gate — and
/// then the side whose gate is `way`, the node's own descent, moved to the FRONT. Off a flat that
/// side is already the steepest; on a flat every side ties at no drop and the node's water takes
/// the flat step, so the order the saddle's water is poured by and the way the node's own water
/// runs are ONE way (MEASURED before this: two closed bodies on a flat divide traded 1.1 km³/yr
/// between the budget and the row). The entries are filled once the whole hierarchy stands.
fn saddle_of(v: u32, met: &[Met], way: u32) -> Saddle {
    let mut sides: Vec<Side> = met
        .iter()
        .map(|k| Side {
            root: k.root,
            gate: k.gate,
            dz: k.dz,
            chord: k.chord,
            entry: NO_DEP,
        })
        .collect();
    for k in 1..sides.len() {
        let mut j = k;
        while j > 0 && side_steeper(&sides[j], &sides[j - 1]) {
            sides.swap(j, j - 1);
            j -= 1;
        }
    }
    // The side the node's own water descends into leads.
    let first = sides
        .iter()
        .position(|k| k.gate == way)
        .expect("the node descends into a side");
    sides[..=first].rotate_right(1);
    Saddle { node: v, sides }
}

/// Whether side `a` is a steeper way off the saddle than side `b` ([`steeper`] on their gates).
fn side_steeper(a: &Side, b: &Side) -> bool {
    steeper(a.dz, a.chord, a.gate, b.dz, b.chord, b.gate)
}

/// What the cascade settled.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cascade {
    /// The water each depression's OWN ring holds, mm·m²/yr.
    pub held_own: Vec<u128>,
    /// Whether each depression filled and ran over its spill.
    pub full: Vec<bool>,
    /// The overflow that reached the ocean.
    pub to_ocean: u128,
    /// The overflow of a full hollow that never met another.
    pub stranded: u128,
    /// How many times an overflow poured into another hollow's side of its saddle.
    pub transfers: usize,
    /// ★ How many times an overflow poured into the OCEAN'S side of its saddle and the ground led
    /// it into a lower hollow that met the sea earlier (plan step 2b, question 1).
    pub ocean_pours: usize,
    /// ★ How many pours left a saddle of three sides or more (plan step 2b, question 2).
    pub saddle_pours: usize,
    /// How many rings the walk visited over every arrival.
    pub steps: usize,
    /// The water each depression passed over its spill: into a side of its saddle or the sea.
    pub passed: Vec<u128>,
}

/// ★ THE WAY OFF A SADDLE (plan step 2b, questions 1 and 2): the STEEPEST side that is the ocean's
/// or a hollow that is not full, or `None` where every side is full and the water rises into the
/// ring above. It never depends on which full side the water came from: a full depression's own
/// sides under it are full too, so every full side at one saddle takes the SAME way — and that way
/// is the one the saddle node's own descent takes, or the way a full side hands that descent on.
#[must_use]
pub fn open_side<'a>(saddle: &'a Saddle, full: &[bool]) -> Option<&'a Side> {
    saddle
        .sides
        .iter()
        .find(|k| k.root == OCEAN || !full[k.root as usize])
}

/// ★ THE CASCADE (Fill–Spill–Merge's `OverflowInto`, Barnes, Callaghan & Wickert 2021 §2.4, with
/// yearly rates in place of volumes; 2026-09-24, ruling B7; status §7 W4). Each leaf's watershed
/// arrives at the leaf, in index order. Water reaching a depression fills its OWN ring first; a
/// ring whose water reaches its own deficit is FULL, and the rest runs on over its saddle:
///
/// 1. nowhere (STRANDED, counted) where the depression never met another — it never spills, so it
///    is never marked full;
/// 2. ★ into the STEEPEST OPEN SIDE of its saddle ([`open_side`]; plan step 2b, ruling B8 item 2):
///    a hollow side that is not full, at the leaf the water's descent from that side's gate ends
///    in; or the OCEAN'S side, into the sea where the gate's descent ends at a root, and else into
///    the lower hollow it ends in — a hollow that met the sea earlier (Fill–Spill–Merge's own
///    link from an ocean-bound depression to the depression its overflow first reaches);
/// 3. into the PARENT's own ring where every side of the saddle is full.
///
/// Before plan step 2b the water went to the SIBLING the binary merge paired it with (by root
/// index) and to the sea whenever the parent was the ocean. At a saddle of three sides the paired
/// sibling need not be the way the ground leads, and the ocean's side can lead into a lower closed
/// hollow; no receiver tree could then carry the water the way the budget booked it. Now the
/// saddle node's own water and every full side's overflow leave by ONE way, which is the way the
/// shipped receivers take.
///
/// Fullness only grows, so the final holdings do not depend on the order the leaves are taken in.
///
/// ★ THE SKIP (review finding 6 of plan step 2). A ring whose saddle has no open side sends every
/// later arrival on to its parent, FOR GOOD, because fullness only grows. The walk records that edge
/// (`up`) and jumps a chain of such rings at once, with path halving — the union–find the hierarchy
/// itself is built with. [`Cascade::steps`] counts what the walk visited.
///
/// **Example.** A leaf at the range's foot takes the whole range face and passes its deficit. Its
/// overflow crosses the saddle and runs down to the pit of the dry pan beside it, which holds it as a
/// salt lake; nothing reaches the pair's own ring, because the pan is not full. Where a third hollow
/// meets the two at the same notch and the ground falls steeper into it, the overflow runs there.
#[must_use]
pub fn cascade(deps: &[Depression], saddles: &[Saddle], shed: &[u128]) -> Cascade {
    let leaves: Vec<u32> = (1..shed.len() as u32).collect();
    cascade_in(deps, saddles, shed, &leaves)
}

/// The cascade with the watersheds arriving in the stated order ([`cascade`] takes the index
/// order; a test states others and reads the same holdings).
fn cascade_in(deps: &[Depression], saddles: &[Saddle], shed: &[u128], arrivals: &[u32]) -> Cascade {
    let m = deps.len();
    let mut flow = Cascade {
        held_own: vec![0; m],
        full: vec![false; m],
        passed: vec![0; m],
        ..Cascade::default()
    };
    let mut up: Vec<u32> = (0..m as u32).collect();
    for &d in arrivals {
        let water = shed[d as usize];
        if water > 0 {
            arrive(deps, saddles, &mut flow, &mut up, d, water);
        }
    }
    // ★ THE EMPTY RINGS (plan step 2b): where three hollows or more meet at one node, the binary
    // merge makes a ring with no node of its own between them. It is full exactly when both its
    // children are, and the water that rises past it never needs to visit it, so it is settled
    // here, children first.
    for d in 1..m {
        let kids = deps[d].children;
        let empty = kids[0] != NO_DEP && deps[kids[0] as usize].saddle == deps[d].saddle;
        if empty && flow.full[kids[0] as usize] && flow.full[kids[1] as usize] {
            flow.full[d] = true;
        }
    }
    flow
}

/// Water `water` reaching depression `x`, carried on until it stands. `up` is the skip over rings
/// whose saddle has no open side ([`cascade`]).
fn arrive(
    deps: &[Depression],
    saddles: &[Saddle],
    flow: &mut Cascade,
    up: &mut [u32],
    mut x: u32,
    mut water: u128,
) {
    loop {
        x = find(up, x);
        flow.steps += 1;
        let xi = x as usize;
        if !flow.full[xi] {
            let room = deps[xi].own_deficit - flow.held_own[xi];
            let take = water.min(room);
            flow.held_own[xi] += take;
            water -= take;
            if take < room {
                return;
            }
            // A ring whose water reached its deficit is full — unless it never met another, and
            // then it has no spill to run over.
            flow.full[xi] = deps[xi].parent != NO_DEP;
        }
        if water == 0 {
            return;
        }
        if deps[xi].parent == NO_DEP {
            flow.stranded += water;
            return;
        }
        let saddle = &saddles[deps[xi].saddle as usize];
        let Some(side) = open_side(saddle, &flow.full) else {
            // Every side full: every later arrival here runs on to the parent, for good.
            up[xi] = deps[xi].parent;
            x = deps[xi].parent;
            continue;
        };
        flow.passed[xi] += water;
        flow.saddle_pours += usize::from(saddle.sides.len() > 2);
        if side.entry == OCEAN {
            flow.to_ocean += water;
            return;
        }
        if side.root == OCEAN {
            flow.ocean_pours += 1;
        } else {
            flow.transfers += 1;
        }
        x = side.entry;
    }
}

/// ★ THE CLOSED LAKE'S LEVEL: the body's own nodes lowest first, wetted a step of equal heights at
/// a time while the standing water's evaporation stays inside the water the body HOLDS (its own
/// ring's share and its full children's). The first step the water cannot pay for stops it, so a
/// hollow with a small river holds a small lake and a hollow with none holds a pan.
fn solve_partial_levels(
    deps: &mut [Depression],
    body_of: &[u32],
    home: &[u32],
    state: &MacroSolve,
    climate: &Climate,
    report: &mut LakeReport,
) {
    let z = &state.z;
    // The nodes of every body that is not full, gathered by a counting sort so nothing is scanned
    // twice: a body's nodes are the nodes whose own depression it covers.
    let mut slot = vec![NO_DEP; deps.len()];
    let mut partial: Vec<u32> = Vec::new();
    for d in 1..deps.len() {
        if body_of[d] == d as u32 && !deps[d].full {
            slot[d] = partial.len() as u32;
            partial.push(d as u32);
        }
    }
    let mut counts = vec![0u32; partial.len() + 1];
    let body_slot = |home: &[u32], i: usize| -> u32 {
        let h = home[i];
        if (h == NO_DEP) | (h == OCEAN) {
            return NO_DEP;
        }
        let body = body_of[h as usize];
        if body == NO_DEP {
            return NO_DEP;
        }
        slot[body as usize]
    };
    for i in 0..z.len() {
        let s = body_slot(home, i);
        if s != NO_DEP {
            counts[s as usize + 1] += 1;
        }
    }
    for k in 1..counts.len() {
        counts[k] += counts[k - 1];
    }
    let mut cells = vec![0u32; counts[partial.len()] as usize];
    let mut at = counts.clone();
    for i in 0..z.len() {
        let s = body_slot(home, i);
        if s != NO_DEP {
            cells[at[s as usize] as usize] = i as u32;
            at[s as usize] += 1;
        }
    }
    drop(at);
    for (k, &d) in partial.iter().enumerate() {
        let from = counts[k] as usize;
        let to = counts[k + 1] as usize;
        let own = &mut cells[from..to];
        own.sort_unstable_by_key(|&v| (z[v as usize], v));
        let level = fill_level(own, state, climate, deps[d as usize].held);
        deps[d as usize].level = level;
        if level == DRY {
            report.dry += 1;
        } else {
            report.partial += 1;
        }
    }
}

/// The level a supply holds over nodes already sorted lowest first: the highest step of equal
/// heights whose standing water the supply can still pay the evaporation of, one sixteenth over
/// that step; [`DRY`] where it cannot pay for the first step.
fn fill_level(own: &[u32], state: &MacroSolve, climate: &Climate, supply: u128) -> i32 {
    if supply == 0 {
        return DRY;
    }
    let z = &state.z;
    let mut level = DRY;
    let mut paid = 0u128;
    let mut k = 0usize;
    while k < own.len() {
        let step_z = z[own[k] as usize];
        let mut cost = 0u128;
        let mut end = k;
        while end < own.len() && z[own[end] as usize] == step_z {
            cost += rates(state, climate, own[end] as usize).1;
            end += 1;
        }
        if paid + cost > supply {
            break;
        }
        paid += cost;
        level = step_z.saturating_add(1);
        k = end;
    }
    level
}

/// ★ WHAT THE WATER'S OWN TREE FOUND (plan step 2b), counted as it was built and as it was
/// followed ([`MacroSolve::follow_the_water`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeReport {
    /// The nodes of every spilling lake's hollow, which run across the lake to its spill.
    pub region_nodes: usize,
    /// The spill nodes of the spilling lakes, each pointing over its saddle.
    pub spills: usize,
    /// The nodes under a closed lake's water: each one ENDS a river.
    pub lake_ends: usize,
    /// The flat steps across a spilling lake decided by the split between two neighbours ahead,
    /// by the one neighbour ahead, and by the nearest neighbour where none lay ahead.
    pub split_steps: usize,
    pub straight_steps: usize,
    pub fallback_steps: usize,
    /// Spilling lakes at one saddle that named DIFFERENT exits: 0 by construction ([`open_side`]).
    pub exit_disagree: usize,
    /// Nodes the followed tree left out of its order: a ring of receivers, a defect.
    pub cyclic: usize,
    /// Spill nodes whose inflow was less than the water their lakes hold: 0 by construction.
    pub shortfall: usize,
}

/// ★ THE WATER'S OWN TREE (plan step 2b; ruling B8 item 2: *"the shipped receivers and the
/// discharge row follow the budget's own water"*): one receiver a node, and the water each spilling
/// lake keeps at its spill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaterTree {
    /// Each node's receiver; [`NO_NODE`] where the water ENDS — the ocean, a root, a node under a
    /// closed lake's water, or a dry hollow's pit.
    pub receiver: Vec<u32>,
    /// ★ THE SINKS: at each spill node of a spilling lake, the water its lake hands the air (the
    /// held water of every spilling lake that leaves there), sorted by node.
    pub sinks: Vec<(u32, u64)>,
    pub report: TreeReport,
}

/// ★ THE WATER'S OWN TREE over a solved state and its budget (plan step 2b; ruling B8 item 2;
/// Fill–Spill–Merge's flow directions, Barnes, Callaghan & Wickert 2021). The routing fill is a
/// scratch surface (ruling W11) and its receivers carried a full inner hollow's river across its
/// parent's DRY ring toward the parent's own outlet, while the budget poured that water over the
/// inner saddle: MEASURED at generator 20, 3 430 trunk steps rose. Here every node takes the way
/// the budget's own water takes:
///
/// 1. the ocean and every root: no receiver;
/// 2. a node of a SPILLING LAKE's hollow (the whole hollow under its spill, the lake and its brim):
///    the flat step toward the lake's SPILL ([`flat_step`], the summed chord to the spill for
///    "which way", the split for "which neighbour"), so the river crosses the lake to its outlet;
/// 3. the SPILL node of a spilling lake: the gate of the side its water leaves by
///    ([`Depression::exit`]) — over the saddle into the next hollow, or into the sea;
/// 4. a node under a CLOSED lake's water: no receiver — the river ENDS where it meets the lake;
/// 5. every other node: its descent on the ground ([`Descent::down`]), which ends at a dry
///    hollow's pit where no lake stands.
///
/// Each spill node carries a SINK: the water its lakes hand the air, which the discharge row takes
/// off there ([`MacroSolve::follow_the_water`]).
///
/// **Example.** A hollow at the foot of the belt's range fills to its sill and spills east. The
/// range's river runs down the ground to the lake's shore, straight across the lake to the notch in
/// the east rim, over the notch into the dry sibling hollow and down to that hollow's salt lake,
/// where it ends. The row at the notch is the range's runoff less what the full lake hands the air.
#[must_use]
pub fn water_tree(state: &MacroSolve, lakes: &Lakes) -> WaterTree {
    let n = state.node_count();
    let deps = &lakes.deps;
    let mut report = TreeReport::default();
    let threads = state.cores.threads();
    let mut receiver = lakes.descent.down.clone();
    // The spilling body each node lies in, or NO_DEP — ★ on the solve's cores (ruling B72 item 1).
    let region: Vec<u32> = crate::par::map(n, threads, |i| {
        let h = lakes.home[i];
        let b = if (h == NO_DEP) | (h == OCEAN) {
            NO_DEP
        } else {
            lakes.body_of[h as usize]
        };
        if (b != NO_DEP) && deps[b as usize].full {
            b
        } else {
            NO_DEP
        }
    });
    // 4. A closed lake's water ends every river that meets it.
    for i in 0..n {
        if (region[i] == NO_DEP) & (lakes.water[i] != DRY) {
            receiver[i] = NO_NODE;
            report.lake_ends += 1;
        }
    }
    // 3. The spills, their exits and their sinks: every spilling lake at one saddle names one exit.
    let mut spills: Vec<(u32, u32, u64)> = (1..deps.len())
        .filter(|&b| (lakes.body_of[b] == b as u32) & deps[b].full)
        .map(|b| (deps[b].spill_node, deps[b].exit, deps[b].held as u64))
        .collect();
    spills.sort_unstable();
    let mut sinks: Vec<(u32, u64)> = Vec::with_capacity(spills.len());
    for &(s, exit, held) in &spills {
        match sinks.last_mut() {
            Some(last) if last.0 == s => {
                last.1 += held;
                report.exit_disagree += usize::from(receiver[s as usize] != exit);
            }
            _ => {
                sinks.push((s, held));
                receiver[s as usize] = exit;
                report.spills += 1;
            }
        }
    }
    // 2. Every spilling lake's hollow: the summed chord from its spill, over its own nodes.
    let mut dist = vec![u64::MAX; n];
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for &(s, _) in &sinks {
        dist[s as usize] = 0;
        heap.push(Reverse((0, s)));
    }
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        let from = region[u as usize];
        for m in state.lattice.neighbours(u) {
            if m == NO_NODE {
                continue;
            }
            let rm = region[m as usize];
            if rm == NO_DEP {
                continue;
            }
            // From a spill node, into the hollows that spill there; from a hollow's node, on over
            // the same hollow.
            let joins = (rm == from) | ((from == NO_DEP) & (deps[rm as usize].spill_node == u));
            let step = d + u64::from(state.chord_m(u, m));
            if joins & (step < dist[m as usize]) {
                dist[m as usize] = step;
                heap.push(Reverse((step, m)));
            }
        }
    }
    drop(heap);
    // Each spilling hollow's node steps toward its spill — ★ on the solve's cores (ruling B72 item 1): a node reads only
    // the finished distances; the ranges' steps join in index order.
    let steps: Vec<(u32, u32, u8)> = crate::par::join(crate::par::per_range(n, threads, |r| {
        let mut cands: Vec<(u32, u64)> = Vec::with_capacity(8);
        let mut out = Vec::new();
        for i in r {
            let b = region[i];
            if b == NO_DEP {
                continue;
            }
            let spill = deps[b as usize].spill_node;
            cands.clear();
            for m in state.lattice.neighbours(i as u32) {
                if m == NO_NODE {
                    continue;
                }
                let mi = m as usize;
                if ((region[mi] == b) | (m == spill)) & (dist[mi] < dist[i]) {
                    cands.push((m, dist[mi] + u64::from(state.chord_m(i as u32, m))));
                }
            }
            let (step, how) = flat_step(state, i as u32, spill, &cands);
            out.push((i as u32, step, how));
        }
        out
    }));
    for (i, step, how) in steps {
        receiver[i as usize] = step;
        report.region_nodes += 1;
        report.fallback_steps += usize::from(how == 0);
        report.straight_steps += usize::from(how == 1);
        report.split_steps += usize::from(how == 2);
    }
    WaterTree {
        receiver,
        sinks,
        report,
    }
}

/// ★ THE TREE OF A BODY WITH NO WATER (plan step 2b, question 8): no budget runs, so the water's
/// way is the ground's own descent and every pit ENDS it — one meaning for the receiver byte on
/// every body.
///
/// **Example.** On the airless moon a crater bowl has no lake and no outlet: every slope of the
/// bowl points down into it, and its lowest node points nowhere.
#[must_use]
pub fn dry_tree(state: &MacroSolve) -> WaterTree {
    WaterTree {
        receiver: descent(state).down,
        sinks: Vec::new(),
        report: TreeReport::default(),
    }
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE: it states an exact quotient over a stated field; it never runs in a
    //! kernel.
    #![allow(
        clippy::integer_division,
        reason = "a test states an exact quotient over a stated field"
    )]
    use super::*;
    use crate::climate::{Climate, NO_WIND};
    use crate::home::{home_moon, home_moon_solve_words, home_solve_words};
    use crate::solve::{FACIES_LAKE, Schedule, steeper};
    use vd_seed::bend::Face;

    /// ★ THE STATED FIELD: the home moon's own lattice, a plateau one hundred sixteenths high with
    /// ONE node at the sea, and a trough of five nodes across one face — `10, 30, 50, 40, 20` — so
    /// the ends are two pits, the middle is the saddle they share, and the whole trough leaves only
    /// over the plateau. Returns the state and `(A, B, saddle)`.
    fn stated_field() -> (MacroSolve, u32, u32, u32) {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        let (a, b, s) = (
            state.lattice.index(Face::NegZ, 20, 20),
            state.lattice.index(Face::NegZ, 24, 20),
            state.lattice.index(Face::NegZ, 22, 20),
        );
        state.z[a as usize] = 10;
        state.z[state.lattice.index(Face::NegZ, 21, 20) as usize] = 30;
        state.z[s as usize] = 50;
        state.z[state.lattice.index(Face::NegZ, 23, 20) as usize] = 40;
        state.z[b as usize] = 20;
        state.route();
        (state, a, b, s)
    }

    /// A climate over `n` nodes with one stated rain and one stated potential evaporation
    /// everywhere, and the runoff the published partition gives them: the three rows the budget
    /// reads.
    fn flat_climate(n: usize, rain: u32, pet: u32) -> Climate {
        let mut out = Climate {
            temperature_dk: vec![2_880; n],
            rain_mm_yr: vec![rain; n],
            aridity_q8: vec![0; n],
            pet_mm_yr: vec![pet; n],
            runoff_mm_yr: vec![0; n],
            frost_z: vec![i32::MAX; n],
            ela_z: vec![i32::MAX; n],
            upwind: vec![NO_WIND; n],
            lapse_mk_km: 6_500,
        };
        for i in 0..n {
            restate(&mut out, i, rain, pet);
        }
        out
    }

    /// One node's rain and potential evaporation restated, with the runoff the law gives them.
    fn restate(climate: &mut Climate, i: usize, rain: u32, pet: u32) {
        climate.rain_mm_yr[i] = rain;
        climate.pet_mm_yr[i] = pet;
        climate.runoff_mm_yr[i] = (crate::climate::runoff_mm_yr(
            crate::gf::Gf::from_i64(i64::from(rain)),
            crate::gf::Gf::from_i64(i64::from(pet)),
        ) + crate::gf::Gf::HALF)
            .to_i64_floor()
            .clamp(0, i64::from(rain)) as u32;
    }

    /// ★ THE HIERARCHY: two pits, each its own leaf, merging at the saddle into one parent that
    /// spills onward at the plateau; every sum is the subtree's own.
    #[test]
    fn two_pits_merge_at_their_saddle_into_one_depression() {
        let (state, a, b, s) = stated_field();
        let lakes = budget(&state, &flat_climate(state.node_count(), 0, 1_000));
        let (da, db, ds) = (
            lakes.home[a as usize],
            lakes.home[b as usize],
            lakes.home[s as usize],
        );
        assert_ne!(da, db);
        assert_eq!(lakes.deps[da as usize].pit, a);
        assert_eq!(lakes.deps[da as usize].pit_z, 10);
        assert_eq!(lakes.deps[db as usize].pit, b);
        assert_eq!(lakes.deps[da as usize].children, [NO_DEP, NO_DEP]);
        // The saddle closes both leaves and opens their parent.
        assert_eq!(lakes.deps[da as usize].spill_z, 50);
        assert_eq!(lakes.deps[da as usize].spill_node, s);
        assert_eq!(lakes.deps[db as usize].spill_z, 50);
        assert_eq!(lakes.deps[da as usize].parent, ds);
        assert_eq!(lakes.deps[db as usize].parent, ds);
        assert_eq!(lakes.deps[ds as usize].children, [da.min(db), da.max(db)]);
        assert_eq!(
            lakes.deps[ds as usize].pit, a,
            "the deeper pit is the parent's"
        );
        // The parent leaves over the plateau, and it carries both children's own nodes.
        assert_eq!(lakes.deps[ds as usize].spill_z, 100);
        let two = lakes.deps[da as usize].area_m2 + lakes.deps[db as usize].area_m2;
        assert!(
            lakes.deps[ds as usize].area_m2 > two,
            "the saddle is its own"
        );
        // A hollow's brim volume is its spill over its own floor.
        assert_eq!(
            lakes.deps[da as usize].capacity_m3(),
            ((50 - 10) * u128::from(state.area[a as usize])
                + (50 - 30)
                    * u128::from(state.area[state.lattice.index(Face::NegZ, 21, 20) as usize]))
                / 16
        );
        // A node at or under the sea is the root's, never a hollow's.
        let sea = state.lattice.index(Face::PosX, 3, 3);
        assert_eq!(lakes.home[sea as usize], OCEAN);
    }

    /// ★ THE THREE ENDS, on the same field, decided by the water and never by the shape of the
    /// hole: no supply at all leaves every hollow DRY; a supply under the hollow's own evaporation
    /// leaves a lake standing under its spill; a supply over it fills the hollow and it SPILLS.
    #[test]
    fn a_hollow_ends_dry_partial_or_spilling_by_its_water_alone() {
        let (mut state, a, b, _) = stated_field();
        let n = state.node_count();
        // 1. DRY: every node asks for a thousand millimetres and gets none.
        let dry = budget(&state, &flat_climate(n, 0, 1_000));
        assert_eq!(dry.report.lake_nodes, 0);
        assert_eq!(dry.report.volume_m3, 0);
        assert!(dry.report.dry > 0);
        assert_eq!(dry.report.spilling, 0);
        assert_eq!(dry.water[a as usize], DRY);
        assert_eq!(dry.deps[dry.home[a as usize] as usize].level, DRY);
        // 2. SPILLING: the rain beats the evaporation everywhere, so every hollow fills to its
        //    brim and runs out of it.
        let wet = budget(&state, &flat_climate(n, 1_000, 0));
        assert!(wet.deps[wet.home[a as usize] as usize].full);
        assert!(wet.report.spilling > 0);
        assert_eq!(wet.report.dry, 0);
        assert!(wet.water[a as usize] > state.z[a as usize]);
        assert!(wet.report.volume_m3 > 0);
        // 3. PARTIAL, beside a SPILLING neighbour, on ONE field: the first hollow's floor is fed
        //    by its own wet node and walled by a node so thirsty that the water cannot climb past
        //    it, so it stands one sixteenth over the floor; the second hollow's pit is wet enough
        //    to fill it to the saddle the two share. ★ The rest of the ground yields NOTHING
        //    (2026-09-24, ruling B7): every hollow now takes its whole watershed, and a plateau
        //    that yielded water would fill the first hollow too.
        let mut some = flat_climate(n, 0, 1_000);
        let feeder = state.lattice.index(Face::NegZ, 21, 20) as usize;
        restate(&mut some, a as usize, 2_000, 500);
        restate(&mut some, feeder, 500, 9_000);
        restate(&mut some, b as usize, 5_000, 0);
        let part = budget(&state, &some);
        let (da, db) = (part.home[a as usize], part.home[b as usize]);
        assert!(!part.deps[da as usize].full);
        assert_eq!(
            part.deps[da as usize].level, 11,
            "one sixteenth over the floor"
        );
        assert_eq!(part.water[a as usize], 11);
        assert_eq!(part.water[feeder], DRY, "the wall stands over the water");
        assert!(part.deps[db as usize].full, "the second hollow fills");
        assert_eq!(part.deps[db as usize].level, 50, "and stands at the saddle");
        assert_eq!(part.water[b as usize], 50);
        assert_eq!(part.report.partial, 1);
        assert!(part.report.spilling > 0);
        // The facies follow the budget, never the flood: the flood raised both pits.
        state.water_z = part.water;
        assert!(state.z_flood[b as usize] > state.z[b as usize]);
        let facies = state.facies(&vec![false; n], 0, true);
        assert_eq!(facies[a as usize], FACIES_LAKE);
        assert_eq!(facies[feeder], 0, "the wall is dry land over the lake");
    }

    /// ★ THE FILL LEVEL over nodes sorted lowest first, on the stated trough: a supply that cannot
    /// pay the first step's evaporation stands DRY; one that pays exactly the first step stands one
    /// sixteenth over it; a supply that pays every step stands one sixteenth over the highest — the
    /// loop's own end.
    #[test]
    fn the_fill_level_pays_its_steps_lowest_first() {
        let (state, _, _, _) = stated_field();
        let climate = flat_climate(state.node_count(), 0, 1_000);
        let mut own: Vec<u32> = trough(&state).iter().map(|&i| i as u32).collect();
        own.sort_unstable_by_key(|&v| (state.z[v as usize], v));
        let first = rates(&state, &climate, own[0] as usize).1;
        assert!(first > 0, "the floor's evaporation costs water");
        assert_eq!(fill_level(&own, &state, &climate, 0), DRY);
        assert_eq!(fill_level(&own, &state, &climate, first - 1), DRY);
        assert_eq!(fill_level(&own, &state, &climate, first), 11);
        assert_eq!(fill_level(&own, &state, &climate, u128::MAX), 51);
    }

    /// The stated field's five trough nodes, west to east: the pit `A`, its wall, the saddle, the
    /// second hollow's wall and its pit `B`.
    fn trough(state: &MacroSolve) -> [usize; 5] {
        [20, 21, 22, 23, 24].map(|i| state.lattice.index(Face::NegZ, i, 20) as usize)
    }

    /// ★ THE FLAT'S DISTANCE, for the tests alone (plan step 2b, question 3): on every node of one
    /// height with a neighbour of that height, the summed chord over the flat to its nearest node
    /// with a LOWER or an ocean neighbour; zero at such a node; `u64::MAX` off every draining flat.
    fn flat_distance(state: &MacroSolve) -> Vec<u64> {
        let n = state.node_count();
        let mut dist = vec![u64::MAX; n];
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
        for (i, at) in dist.iter_mut().enumerate() {
            let lower = state.lattice.neighbours(i as u32).iter().any(|&m| {
                (m != NO_NODE) && (state.ocean[m as usize] | (state.z[m as usize] < state.z[i]))
            });
            if !state.ocean[i] && lower {
                *at = 0;
                heap.push(Reverse((0, i as u32)));
            }
        }
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            for m in state.lattice.neighbours(u) {
                if (m == NO_NODE)
                    || state.ocean[m as usize]
                    || (state.z[m as usize] != state.z[u as usize])
                {
                    continue;
                }
                let step = d + u64::from(state.chord_m(u, m));
                if step < dist[m as usize] {
                    dist[m as usize] = step;
                    heap.push(Reverse((step, m)));
                }
            }
        }
        dist
    }

    /// ★ AN INDEPENDENT WALK DOWN THE GROUND, for the tests alone: from `i`, step to the steepest
    /// STRICTLY lower neighbour while one exists (ties to the smaller index); on a flat, where none
    /// is lower, the library's own step is taken only after it is checked to go to a neighbour of
    /// the same height NEARER the flat's drain by [`flat_distance`] (the split's chance is the
    /// library's, and this walk does not draw it again); it ends at the ocean or where no step is.
    fn walk_down(state: &MacroSolve, flat: &[u64], down: &[u32], mut i: usize) -> usize {
        loop {
            if state.ocean[i] {
                return i;
            }
            let mut best: Option<(usize, i64, u32)> = None;
            for m in state.lattice.neighbours(i as u32) {
                if m == NO_NODE || state.z[m as usize] >= state.z[i] {
                    continue;
                }
                let dz = i64::from(state.z[i]) - i64::from(state.z[m as usize]);
                let l = state.chord_m(i as u32, m);
                let take = match best {
                    None => true,
                    Some((b, bdz, bl)) => steeper(dz, l, m, bdz, bl, b as u32),
                };
                if take {
                    best = Some((m as usize, dz, l));
                }
            }
            match (best, down[i]) {
                (Some((m, _, _)), _) => i = m,
                (None, NO_NODE) => return i,
                (None, d) => {
                    assert_eq!(
                        state.z[d as usize], state.z[i],
                        "a flat step keeps its height"
                    );
                    assert!(flat[d as usize] < flat[i], "a flat step nears the drain");
                    i = d as usize;
                }
            }
        }
    }

    /// ★ TEST 1 (plan step 2): A HOLLOW'S SUPPLY IS ITS WHOLE WATERSHED. Every node yields water;
    /// the pit `A`'s supply equals the runoff summed over every node whose independent walk down the
    /// ground ends at `A` — and that watershed reaches far past `A`'s own two nodes, which were all
    /// the budget booked before ruling B7 (`brim_supply`). ★ The plateau round the trough is ONE
    /// FLAT that drains into the trough and into the sea (plan step 2b, question 3): its nodes run to
    /// their nearest drain, so the trough's watershed reaches over the plateau.
    #[test]
    fn a_hollows_supply_is_the_runoff_of_its_whole_watershed() {
        let (state, a, _, _) = stated_field();
        let n = state.node_count();
        let climate = flat_climate(n, 1_000, 1_000);
        let lakes = budget(&state, &climate);
        let flat = flat_distance(&state);
        let down = &lakes.descent.down;
        let leaf = lakes.home[a as usize] as usize;
        let mut hand = 0u128;
        let mut shed_nodes = 0usize;
        for i in 0..n {
            if !state.ocean[i] && walk_down(&state, &flat, down, i) == a as usize {
                hand += u128::from(climate.runoff_mm_yr[i]) * u128::from(state.area[i]);
                shed_nodes += 1;
            }
        }
        assert_eq!(lakes.deps[leaf].supply, hand);
        assert!(
            shed_nodes > 2,
            "the watershed is more than the brim: {shed_nodes}"
        );
        assert!(lakes.deps[leaf].supply > lakes.deps[leaf].brim_supply);
        // The library's own descent agrees with the walk at every node.
        for i in 0..n {
            assert_eq!(
                lakes.descent.end[i] as usize,
                walk_down(&state, &flat, down, i),
                "node {i}"
            );
        }
        assert_eq!(descent(&state), lakes.descent);
    }

    /// ★ TESTS 2 AND 5 (plan step 2; status §7 W4): A FULL HOLLOW POURS INTO A DRY SIBLING. Nothing
    /// yields water but `B`'s own pit. With `B` dry too, `A` — which yields nothing of its own —
    /// is DRY. With `B`'s pit wet, `B` fills, and its overflow runs over the saddle down to `A`'s
    /// pit: `A` holds EXACTLY `B`'s supply less `B`'s deficit, and stands PARTIAL one sixteenth over
    /// its floor. Before the cascade that overflow was lost and `A` stayed dry.
    #[test]
    fn a_full_hollow_pours_into_a_dry_sibling() {
        let (state, a, b, _) = stated_field();
        let n = state.node_count();
        let [_, wall_a, _, wall_b, _] = trough(&state);
        let dry = budget(&state, &flat_climate(n, 0, 1_000));
        assert_eq!(dry.deps[dry.home[a as usize] as usize].level, DRY);
        assert_eq!(dry.report.transfers, 0);
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, b as usize, 2_500, 0);
        let lakes = budget(&state, &climate);
        let (da, db) = (
            lakes.home[a as usize] as usize,
            lakes.home[b as usize] as usize,
        );
        assert!(lakes.deps[db].full);
        assert!(!lakes.deps[da].full);
        assert_eq!(lakes.report.transfers, 1);
        let excess =
            2_500 * u128::from(state.area[b as usize]) - 1_000 * u128::from(state.area[wall_b]);
        assert_eq!(lakes.deps[da].held, excess);
        assert_eq!(lakes.deps[da].supply, 0, "A yields nothing of its own");
        assert_eq!(lakes.deps[da].level, 11, "one sixteenth over A's floor");
        assert_eq!(lakes.water[a as usize], 11);
        assert_eq!(lakes.water[wall_a], DRY);
        assert_eq!(lakes.water[b as usize], 50, "B stands at the saddle");
        assert_eq!(
            lakes.deps[da].gate, wall_a as u32,
            "B's overflow enters A by the steepest step down from the saddle"
        );
        assert_eq!(lakes.report.to_ocean, 0);
    }

    /// ★ TESTS 3 AND 4 (plan step 2): WHEN BOTH CHILDREN ARE FULL THE PARENT'S OWN RING TAKES THE
    /// WATER, and a top-level hollow hands the ocean exactly its supply less its deficit. `B`'s pit
    /// feeds both hollows: at 3 500 mm the pair's ring (the saddle) holds a part and the pair
    /// stands as ONE lake at 41 over both floors, the saddle dry; at 10 000 mm the ring fills too,
    /// the pair spills at the plateau, and what reaches the ocean is `B`'s supply less the pair's
    /// whole deficit, to the unit.
    #[test]
    fn a_parent_ring_takes_the_water_when_both_children_are_full() {
        let (state, a, b, s) = stated_field();
        let n = state.node_count();
        let [_, wall_a, _, wall_b, _] = trough(&state);
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, b as usize, 3_500, 0);
        let lakes = budget(&state, &climate);
        let (da, db, dp) = (
            lakes.home[a as usize] as usize,
            lakes.home[b as usize] as usize,
            lakes.home[s as usize] as usize,
        );
        assert!(lakes.deps[da].full);
        assert!(lakes.deps[db].full);
        assert!(!lakes.deps[dp].full);
        assert_eq!(
            lakes.deps[dp].children,
            [da.min(db) as u32, da.max(db) as u32]
        );
        assert_eq!(lakes.body_of[da], dp as u32);
        assert_eq!(lakes.deps[dp].level, 41);
        for node in [a as usize, wall_a, wall_b, b as usize] {
            assert_eq!(lakes.water[node], 41, "node {node}");
        }
        assert_eq!(
            lakes.water[s as usize], DRY,
            "the saddle stands over the lake"
        );
        assert!(lakes.deps[dp].held > lakes.deps[da].deficit + lakes.deps[db].deficit);
        let mut flood = flat_climate(n, 0, 1_000);
        restate(&mut flood, b as usize, 10_000, 0);
        let full = budget(&state, &flood);
        let dp = full.home[s as usize] as usize;
        assert!(full.deps[dp].full);
        assert_eq!(full.deps[dp].parent, OCEAN);
        assert_eq!(full.deps[dp].level, 100, "the pair spills at the plateau");
        assert_eq!(
            full.report.to_ocean,
            10_000 * u128::from(state.area[b as usize]) - full.deps[dp].deficit
        );
        assert_eq!(
            full.report.runoff_total,
            full.report.evaporated_total + full.report.to_ocean + full.report.stranded
        );
    }

    /// ★ TEST 6 (plan step 2): THREE HOLLOWS MEETING AT ONE SADDLE NODE. The first two close into a
    /// parent whose own ring is EMPTY (its deficit is zero), and that parent closes at the same node
    /// into the one that holds the third: the empty ring is full as soon as water reaches it, so
    /// the water passes through it. The three pits yield a little and cost nothing; the saddle
    /// costs more than all three yield, so one lake stands over the three pits under the saddle.
    #[test]
    fn three_hollows_meeting_at_one_node_pass_the_water_through_an_empty_ring() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        let node = |i: i32, j: i32| state.lattice.index(Face::NegZ, i, j) as usize;
        let (v, p1, p2, p3) = (node(40, 40), node(39, 39), node(41, 39), node(40, 41));
        state.z[v] = 60;
        state.z[p1] = 10;
        state.z[p2] = 20;
        state.z[p3] = 30;
        state.route();
        let mut climate = flat_climate(n, 0, 1_000);
        for p in [p1, p2, p3] {
            restate(&mut climate, p, 200, 0);
        }
        let lakes = budget(&state, &climate);
        let (l1, l2, l3) = (lakes.home[p1], lakes.home[p2], lakes.home[p3]);
        let pair = lakes.deps[l1 as usize].parent;
        let top = lakes.home[v];
        assert_eq!(lakes.deps[pair as usize].children, [l1, l2]);
        assert_eq!(lakes.deps[pair as usize].own_deficit, 0);
        assert_eq!(lakes.deps[pair as usize].spill_node, v as u32);
        assert_eq!(lakes.deps[pair as usize].parent, top);
        assert_eq!(lakes.deps[top as usize].children, [pair, l3]);
        assert!(
            lakes.deps[pair as usize].full,
            "the empty ring passes the water"
        );
        assert!(!lakes.deps[top as usize].full);
        assert_eq!(lakes.report.transfers, 2);
        assert_eq!(lakes.deps[top as usize].level, 31);
        for p in [p1, p2, p3] {
            assert_eq!(lakes.water[p], 31);
        }
        assert_eq!(lakes.water[v], DRY);
    }

    /// ★ TEST 7 (plan step 2): A SPILL ACROSS A FACE SEAM. The saddle stands on the edge column of
    /// one face, one pit beside it on that face and the other straight across the seam on the next
    /// face. The first pit's overflow crosses the saddle and runs down to the pit on the OTHER face,
    /// through the gate the seam table names.
    #[test]
    fn a_spill_across_a_face_seam_enters_the_leaf_on_the_other_face() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let edge = state.lattice.edge as i32;
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        let s = state.lattice.index(Face::PosX, edge - 1, 30) as usize;
        let a = state.lattice.index(Face::PosX, edge - 2, 30) as usize;
        let across: Vec<u32> = state
            .lattice
            .neighbours(s as u32)
            .into_iter()
            .filter(|&m| m != NO_NODE)
            .filter(|&m| state.lattice.split(m).0 != Face::PosX)
            .collect();
        let b = across[across.len() / 2] as usize;
        state.z[s] = 50;
        state.z[a] = 10;
        state.z[b] = 20;
        state.route();
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, a, 1_500, 0);
        let lakes = budget(&state, &climate);
        let (da, db) = (lakes.home[a] as usize, lakes.home[b] as usize);
        assert_ne!(state.lattice.split(lakes.deps[db].pit).0, Face::PosX);
        assert_eq!(
            lakes.deps[db].gate, b as u32,
            "the water enters across the seam"
        );
        assert_eq!(lakes.deps[da].spill_node, s as u32);
        assert!(lakes.deps[da].full);
        assert!(
            lakes.deps[db].full,
            "the other face's hollow filled from this one"
        );
        assert_eq!(lakes.deps[db].supply, 0, "it yields nothing of its own");
        assert_eq!(lakes.report.transfers, 1);
    }

    /// Nine nodes of sea round `PosX (3, 3)`: an ocean larger than any pocket the tests state, so
    /// the largest-body rule names it the ocean.
    fn widen_the_sea(state: &mut MacroSolve) {
        for i in 2..=4 {
            for j in 2..=4 {
                state.z[state.lattice.index(Face::PosX, i, j) as usize] = 0;
            }
        }
    }

    /// A plateau with a sea of nine nodes, and a two-node pocket far inland UNDER the sea's level:
    /// its pit at −50 and its wall at −20. Returns the state (not routed) and the pocket's nodes.
    fn pocket_field() -> (MacroSolve, usize, usize) {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        widen_the_sea(&mut state);
        let q = state.lattice.index(Face::NegZ, 40, 40) as usize;
        let wall = state.lattice.index(Face::NegZ, 41, 40) as usize;
        state.z[q] = -50;
        state.z[wall] = -20;
        (state, q, wall)
    }

    /// ★ TEST 8 (plan step 2; ruling B7 item 1): A POCKET UNDER THE SEA'S LEVEL BEHIND A RIDGE IS A
    /// CLOSED BASIN. The routing leaves it out of the ocean and raises it like any hollow; the
    /// budget's root is the ocean alone, so the pocket is a hollow of its own; its water is its
    /// budget's — a lake at its own level, one sixteenth over its floor and 49 under the sea — and
    /// never the sea's facies. With no rain it is a dry basin under the sea's level.
    #[test]
    fn a_pocket_under_the_sea_is_a_closed_basin_by_its_budget() {
        let (mut state, q, wall) = pocket_field();
        let n = state.node_count();
        let route = state.route();
        assert_eq!((route.pockets, route.pocket_nodes), (1, 2));
        assert!(!state.ocean[q]);
        assert_eq!(state.z_flood[q], 100, "the flood raises the pocket");
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, q, 500, 0);
        let lakes = budget(&state, &climate);
        assert_ne!(lakes.home[q], OCEAN);
        assert_eq!(lakes.water[q], -49);
        assert_eq!(lakes.water[wall], DRY);
        state.water_z = lakes.water;
        assert_eq!(
            state.water_level(q as u32),
            -49,
            "its own level, not the sea's"
        );
        let facies = state.facies(&vec![false; n], 0, true);
        assert_eq!(facies[q], FACIES_LAKE);
        assert_eq!(facies[wall], 0);
        let sea = state.lattice.index(Face::PosX, 3, 3) as usize;
        assert_eq!(
            facies[sea] & crate::solve::FACIES_SEA,
            crate::solve::FACIES_SEA
        );
        // No rain: a dry basin under the sea's level.
        let dry = budget(&state, &flat_climate(n, 0, 1_000));
        state.water_z = dry.water;
        assert_eq!(state.water_level(q as u32), DRY);
        assert_eq!(state.facies(&vec![false; n], 0, true)[q], 0);
    }

    /// ★ TEST 9 (plan step 2): A POCKET INSIDE A POCKET. The stated trough stands wholly under the
    /// sea's level, so the pocket holds two hollows under their shared saddle: the inner one fills
    /// first and pours into the outer one, and both stand at their own levels under the sea's.
    #[test]
    fn a_pocket_inside_a_pocket_fills_the_inner_hollow_first() {
        let (mut state, a, b, s) = stated_field();
        let n = state.node_count();
        let [_, wall_a, _, wall_b, _] = trough(&state);
        for (node, z) in [
            (a as usize, -90),
            (wall_a, -70),
            (s as usize, -50),
            (wall_b, -60),
        ] {
            state.z[node] = z;
        }
        state.z[b as usize] = -80;
        widen_the_sea(&mut state);
        let route = state.route();
        assert_eq!((route.pockets, route.pocket_nodes), (1, 5));
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, b as usize, 2_500, 0);
        let lakes = budget(&state, &climate);
        let (da, db) = (
            lakes.home[a as usize] as usize,
            lakes.home[b as usize] as usize,
        );
        assert_eq!(lakes.deps[da].parent, lakes.deps[db].parent);
        assert!(lakes.deps[db].full);
        assert_eq!(
            lakes.water[b as usize], -50,
            "the inner hollow at its saddle"
        );
        assert_eq!(lakes.water[a as usize], -89, "the outer at its own level");
        assert_eq!(lakes.report.transfers, 1);
    }

    /// ★ TESTS 10 AND 11 (plan step 2) on the full solve over the moon's lattice under the home
    /// planet's words: THE BOOKS BALANCE to the unit — every millimetre of runoff is held, reaches
    /// the ocean, or is stranded — and EVERY NODE'S DESCENT ENDS at a root or at a leaf's own pit.
    #[test]
    fn the_books_balance_and_every_descent_ends_at_a_pit_or_the_root() {
        let moon = home_moon();
        let words = crate::solve::SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let (state, _, report, climate) =
            crate::test_solves::solved_shipped(&moon, &words, Schedule::standard(words.age_yr));
        let l = &report.lakes;
        assert!(l.runoff_total > 0);
        assert_eq!(l.runoff_total, l.evaporated_total + l.to_ocean + l.stranded);
        assert_eq!(
            l.stranded, 0,
            "every hollow of a connected lattice meets another"
        );
        // The solve's own climate over its final field (ruling B101: the seasons' chain on a body with air).
        let lakes = budget(&state, &climate);
        assert_eq!(lakes.report, report.lakes, "the budget is the solve's own");
        for i in 0..state.node_count() {
            let end = lakes.descent.end[i] as usize;
            let home = lakes.home[end];
            if home == OCEAN {
                assert_eq!(lakes.descent.down[end], NO_NODE);
                continue;
            }
            let dep = &lakes.deps[home as usize];
            assert_eq!(dep.children, [NO_DEP, NO_DEP], "node {i} ends in a parent");
            assert_eq!(dep.pit as usize, end, "node {i} ends at its leaf's own pit");
        }
    }

    /// ★ TEST 12 (plan step 2): A HOLLOW THAT NEVER MET ANOTHER has no spill: the cascade fills it,
    /// strands the rest and counts it, and never calls it full. A connected lattice has none, so
    /// the hierarchy is stated.
    #[test]
    fn a_hollow_that_never_met_another_strands_its_overflow() {
        let mut alone = leaf(7, 0);
        alone.deficit = 100;
        alone.own_deficit = 100;
        let deps = vec![leaf(NO_NODE, i32::MAX), alone];
        let flow = cascade(&deps, &[], &[0, 150]);
        assert_eq!(flow.held_own[1], 100);
        assert_eq!(flow.stranded, 50);
        assert!(!flow.full[1]);
        let short = cascade(&deps, &[], &[0, 60]);
        assert_eq!((short.held_own[1], short.stranded), (60, 0));
        // Exactly its deficit: brim full, nothing left over to strand.
        let exact = cascade(&deps, &[], &[0, 100]);
        assert_eq!((exact.held_own[1], exact.stranded), (100, 0));
        assert!(!exact.full[1]);
    }

    /// ★ THE SKIP OVER FULL RINGS (review finding 6 of plan step 2): a stated nested lake — leaves 1
    /// and 2 under ring 4, ring 4 and leaf 3 under ring 5, ring 5 into the sea — each ring holding
    /// 10. Leaf 1 takes 100: it fills itself, pours into leaf 2, fills ring 4, pours into leaf 3,
    /// fills ring 5 and sends 50 to the sea. Leaves 2 and 3 then take 5 each, and both run to the
    /// sea. The holdings are the plain walk's; the walk visits 8 rings, where the plain walk
    /// climbed every full ring again and visited 10 (ARITH: 5 + 3 + 2).
    #[test]
    fn the_cascade_skips_full_rings_and_keeps_the_walks_holdings() {
        let ring = |pit: u32, children: [u32; 2], deficit: u128, parent: u32, saddle: u32| {
            let mut d = leaf(pit, 0);
            d.children = children;
            d.own_deficit = 10;
            d.deficit = deficit;
            d.parent = parent;
            d.saddle = saddle;
            d
        };
        let deps = vec![
            leaf(NO_NODE, i32::MAX),
            ring(1, [NO_DEP, NO_DEP], 10, 4, 0),
            ring(2, [NO_DEP, NO_DEP], 10, 4, 0),
            ring(3, [NO_DEP, NO_DEP], 10, 5, 1),
            ring(1, [1, 2], 30, 5, 1),
            ring(1, [4, 3], 50, OCEAN, 2),
        ];
        // The three saddles, each side with its gate and the leaf its water enters.
        let side = |root: u32, gate: u32, entry: u32| Side {
            root,
            gate,
            dz: 1,
            chord: 1,
            entry,
        };
        let saddles = vec![
            Saddle {
                node: 100,
                sides: vec![side(1, 101, 1), side(2, 102, 2)],
            },
            Saddle {
                node: 200,
                sides: vec![side(4, 201, 1), side(3, 203, 3)],
            },
            Saddle {
                node: 300,
                sides: vec![side(OCEAN, 301, OCEAN), side(5, 305, 1)],
            },
        ];
        let flow = cascade(&deps, &saddles, &[0, 100, 5, 5, 0, 0]);
        assert_eq!(flow.held_own, vec![0, 10, 10, 10, 10, 10]);
        assert_eq!(flow.full, vec![false, true, true, true, true, true]);
        assert_eq!((flow.to_ocean, flow.stranded), (60, 0));
        assert_eq!(flow.transfers, 2);
        assert_eq!(flow.steps, 8);
    }

    /// ★ A BODY WITH NO OUTLET UNDER ITS SEA seeds the hierarchy's root with the same six lowest
    /// nodes the routing seeds, so the two agree about where the water leaves.
    #[test]
    fn a_body_with_no_sea_seeds_the_root_with_its_lowest_nodes() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        state.sea_z = i32::MIN;
        state.route();
        let n = state.node_count();
        let lakes = budget(&state, &flat_climate(n, 0, 1_000));
        // The six lowest nodes by height, ties to the smaller index, are the root's own — the very
        // nodes `route` takes for its outlets.
        let mut lowest: Vec<u32> = (0..n as u32).collect();
        lowest.sort_unstable_by_key(|&v| (state.z[v as usize], v));
        for &v in lowest.iter().take(crate::solve::OUTLETS_WITHOUT_SEA) {
            assert_eq!(lakes.home[v as usize], OCEAN);
        }
        assert_eq!(lakes.report.lake_nodes, 0);
    }

    /// ★ THE SOLVE'S OWN LAKES on the moon's lattice under the home planet's words: the budget runs,
    /// every body ends dry, partial or spilling, and every node the facies call a lake stands under
    /// the level the budget settled — never under the routing fill, which raised far more. ★ The
    /// solved field holds few lakes (MEASURED: none in phase 2 before the deposit's cure, 24 hollows
    /// after it — a hollow full of rock passes the rest on, ruling B51 item 2; 98 on generator 26's
    /// moon), so the facies are read over the same field with two hollows dug into it
    /// (`crate::solve::dug`) and the budget over that field.
    #[test]
    fn the_solve_states_a_lake_only_where_the_budget_put_water() {
        let moon = home_moon();
        // The home planet's own air and climate with a sea that covers PART of the moon — the
        // stand `solve.rs` states, so the two tests read one body.
        let words = crate::solve::SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let (mut state, _, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(words.age_yr));
        let (_, lakes) = crate::solve::dug::two_hollows(&moon, &words, &mut state);
        let report = &lakes;
        let facies = state.facies(&vec![false; state.node_count()], 0, true);
        // The dug bowl holds standing water (the helper chooses it by the budget's own verdict). The depressions' count
        // is no measure of the dig: on generator 26's moon the pit dug on the trunk lowered a saddle and opened older
        // hollows (MEASURED: 98 depressions before the dig, 97 after).
        assert!(report.report.partial > 0, "the dug bowl holds a lake");
        assert_eq!(
            report.report.bodies,
            report.report.dry + report.report.partial + report.report.spilling
        );
        let mut lake_nodes = 0usize;
        let mut filled = 0usize;
        for (i, &f) in facies.iter().enumerate() {
            if state.z_flood[i] > state.z[i] {
                filled += 1;
            }
            if f & FACIES_LAKE != 0 {
                lake_nodes += 1;
                assert!(state.water_z[i] > state.z[i]);
                assert_ne!(state.water_level(i as u32), DRY);
            } else {
                assert!(state.water_z[i] <= state.z[i]);
            }
        }
        assert_eq!(lake_nodes, report.report.lake_nodes);
        assert!(lake_nodes > 0, "a lake stands");
        assert!(filled > 0, "the routing fill raised hollows");
        assert!(lake_nodes < filled, "the fill is not the water");
    }

    /// ★ A DRY BODY GETS NO BUDGET AND NO LAKE: every hollow on the airless moon is a closed basin,
    /// and the report's lake rows stay at nothing.
    #[test]
    fn a_dry_body_holds_no_lake_at_all() {
        let moon = home_moon();
        let words = home_moon_solve_words();
        let (state, facies, report) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(words.age_yr));
        assert_eq!(report.lakes, LakeReport::default());
        assert!(state.water_z.iter().all(|&w| w == DRY));
        assert!(facies.iter().all(|&f| f & FACIES_LAKE == 0));
    }

    /// The brim volume's own arithmetic: nothing while the hollow has met nobody, and the spill
    /// over its own floor once it has.
    #[test]
    fn the_brim_volume_of_an_unclosed_hollow_is_nothing() {
        let mut d = leaf(7, 0);
        assert_eq!(d.capacity_m3(), 0);
        d.spill_z = 32;
        d.area_m2 = 1_000;
        d.z_area = 16_000;
        assert_eq!(d.capacity_m3(), (32 * 1_000 - 16_000) / 16);
    }

    // ★★ PLAN STEP 2b (ruling B8 item 2): THE WATER'S OWN TREE AND THE BUDGET'S OWN WATER.

    /// ★ THE WATER'S TREE FOLLOWED over a routed stated field and a climate (plan step 2b): the
    /// budget, and a copy of the state whose receivers, order and row are the water's own; the ring
    /// count, the short sinks and the disagreeing exits are nothing.
    fn follow(state: &MacroSolve, climate: &Climate) -> (Lakes, MacroSolve) {
        let lakes = budget(state, climate);
        let mut out = state.clone();
        out.water_z.clone_from(&lakes.water);
        let tree = water_tree(&out, &lakes);
        let report = out.follow_the_water(tree, &climate.runoff_mm_yr);
        assert_eq!(
            (report.cyclic, report.shortfall, report.exit_disagree),
            (0, 0, 0)
        );
        (lakes, out)
    }

    /// ★ THE TREE'S THREE LAWS on any followed field (plan step 2b): the land's row into the ocean
    /// is the cascade's `to_ocean` to the unit; the rows at the land's ends and the sinks hold what
    /// the budget holds; along every receiver the FLOOR (the ground, or the water's surface where
    /// water stands) never rises.
    fn assert_tree_laws(s: &MacroSolve, lakes: &Lakes) {
        let floor = |i: usize| -> i32 {
            let w = s.water_level(i as u32);
            if w == DRY { s.z[i] } else { s.z[i].max(w) }
        };
        let (mut to_sea, mut ends) = (0u128, 0u128);
        for i in 0..s.node_count() {
            let r = s.receiver[i];
            if s.ocean[i] {
                continue;
            }
            if r == NO_NODE {
                ends += u128::from(s.discharge[i]);
                continue;
            }
            assert!(
                floor(r as usize) <= floor(i),
                "the floor rises from {i} to {r}"
            );
            if s.ocean[r as usize] {
                to_sea += u128::from(s.discharge[i]);
            }
        }
        let sinks: u128 = s.spill_sinks.iter().map(|&(_, w)| u128::from(w)).sum();
        assert_eq!(to_sea, lakes.report.to_ocean, "the row into the sea");
        assert_eq!(
            ends + sinks,
            lakes.report.evaporated_total,
            "the held water"
        );
    }

    /// ★ T1 AND T2 (plan step 2b; ruling B8 item 2): A FULL HOLLOW'S RIVER CROSSES ITS LAKE TO ITS
    /// SPILL AND POURS INTO ITS SIBLING, WHERE IT ENDS. On the stated trough only `B`'s pit is wet:
    /// `B` fills, its hollow runs to the saddle, the saddle points down `A`'s wall, `A`'s lake ENDS
    /// the river, the row at the saddle is `B`'s outflow (its inflow less its lake's evaporation,
    /// taken off at the saddle as the tree's one sink) and the row at `A`'s end is what `A` holds.
    /// Before step 2b the routing fill sent `B`'s water over the plateau toward the pair's own
    /// outlet.
    #[test]
    fn a_full_hollows_river_crosses_to_its_spill_and_ends_in_its_sibling() {
        let (state, a, b, s) = stated_field();
        let n = state.node_count();
        let [_, wall_a, _, wall_b, _] = trough(&state);
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, b as usize, 2_500, 0);
        let (lakes, t) = follow(&state, &climate);
        let (da, db) = (
            lakes.home[a as usize] as usize,
            lakes.home[b as usize] as usize,
        );
        assert!(lakes.deps[db].full);
        assert!(!lakes.deps[da].full);
        assert_eq!(lakes.body_of[db], db as u32);
        assert_eq!(lakes.deps[db].exit, wall_a as u32);
        assert_eq!(t.receiver[b as usize], wall_b as u32);
        assert_eq!(t.receiver[wall_b], s);
        assert_eq!(t.receiver[s as usize], wall_a as u32);
        assert_eq!(t.receiver[wall_a], a);
        assert_eq!(
            t.receiver[a as usize], NO_NODE,
            "the river ends in A's lake"
        );
        // At a saddle of two sides the row at the spill is what `B` passed on: its inflow, the
        // rain on its pit, less what its lake holds.
        let passed = 2_500 * u128::from(state.area[b as usize]) - lakes.deps[db].held;
        assert!(passed > 0);
        assert_eq!(lakes.deps[db].passed, passed);
        assert_eq!(u128::from(t.discharge[s as usize]), passed);
        assert_eq!(u128::from(t.discharge[a as usize]), lakes.deps[da].held);
        assert_eq!(t.spill_sinks, vec![(s, lakes.deps[db].held as u64)]);
        assert_tree_laws(&t, &lakes);
    }

    /// ★ T3 (plan step 2b): A DRY HOLLOW ENDS ITS WATER AT ITS PIT AND PASSES NOTHING ON. The
    /// plateau yields a little everywhere and both pits are far thirstier than all of it: each
    /// hollow is a salt pan, its pit the only end in it, the row at each pit what the hollow holds,
    /// and no receiver leaves the trough.
    #[test]
    fn a_dry_hollow_ends_its_water_at_its_pit() {
        let (state, a, b, _) = stated_field();
        let n = state.node_count();
        let trough = trough(&state);
        let mut climate = flat_climate(n, 300, 1_000);
        restate(&mut climate, a as usize, 0, 1_000_000);
        restate(&mut climate, b as usize, 0, 1_000_000);
        let (lakes, t) = follow(&state, &climate);
        for pit in [a as usize, b as usize] {
            let d = lakes.home[pit] as usize;
            assert_eq!(lakes.deps[d].level, DRY);
            assert!(
                lakes.deps[d].held > 0,
                "the pan holds the water that reaches it"
            );
            assert_eq!(t.receiver[pit], NO_NODE);
            assert_eq!(u128::from(t.discharge[pit]), lakes.deps[d].held);
        }
        for &i in &trough {
            let r = t.receiver[i];
            assert!(
                (r == NO_NODE) | trough.contains(&(r as usize)),
                "node {i} leaves the trough"
            );
        }
        assert_tree_laws(&t, &lakes);
    }

    /// ★ T4 (plan step 2b): A CLOSED LAKE ENDS EVERY RIVER THAT MEETS IT. At 3 500 mm on `B`'s pit
    /// both hollows fill and the pair stands as ONE closed lake at 41 over both floors: every node
    /// under it is an end, the dry saddle over it descends into it, and the rows at its ends sum to
    /// the water the pair holds.
    #[test]
    fn every_node_of_a_closed_lake_ends_the_river() {
        let (state, a, b, s) = stated_field();
        let n = state.node_count();
        let [_, wall_a, _, wall_b, _] = trough(&state);
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, b as usize, 3_500, 0);
        let (lakes, t) = follow(&state, &climate);
        let dp = lakes.home[s as usize] as usize;
        assert!(!lakes.deps[dp].full);
        assert_eq!(lakes.deps[dp].level, 41);
        let lake = [a as usize, wall_a, wall_b, b as usize];
        for &i in &lake {
            assert_eq!(t.receiver[i], NO_NODE, "node {i}");
        }
        assert_eq!(t.receiver[s as usize], lakes.descent.down[s as usize]);
        assert!(lake.contains(&(t.receiver[s as usize] as usize)));
        let ends: u128 = lake.iter().map(|&i| u128::from(t.discharge[i])).sum();
        assert_eq!(ends, lakes.deps[dp].held);
        assert!(t.spill_sinks.is_empty(), "no lake spills");
        assert_tree_laws(&t, &lakes);
    }

    /// ★ T5 (plan step 2b): A PAIR THAT SPILLS INTO THE SEA hands the sea its row exactly. At
    /// 10 000 mm the pair fills and spills over the plateau, and every node of its hollow runs to
    /// its spill.
    #[test]
    fn a_pair_that_spills_hands_the_sea_the_cascades_water() {
        let (state, a, b, s) = stated_field();
        let n = state.node_count();
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, b as usize, 10_000, 0);
        let (lakes, t) = follow(&state, &climate);
        let dp = lakes.home[s as usize] as usize;
        assert!(lakes.deps[dp].full);
        assert_eq!(lakes.deps[dp].parent, OCEAN);
        assert!(lakes.report.to_ocean > 0);
        for i in [a as usize, b as usize, s as usize] {
            assert_ne!(t.receiver[i], NO_NODE, "node {i} runs to the spill");
        }
        assert_tree_laws(&t, &lakes);
    }

    /// The line field of question 1: a sea of nine nodes at `PosX (2..=4, 9..=11)` and, running
    /// east from it along row 10, `r1` 50 (the sill of `D1` to the sea), `D1`'s pit 10, `g` 40,
    /// the saddle `s2` 60, and `D2`'s pit 20; the plateau at 100. Returns the state (routed) and
    /// `(d1, g, s2, d2)`.
    fn ocean_side_field() -> (MacroSolve, usize, usize, usize, usize) {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        let node = |i: i32| state.lattice.index(Face::PosX, i, 10) as usize;
        for i in 2..=4 {
            for j in 9..=11 {
                state.z[state.lattice.index(Face::PosX, i, j) as usize] = 0;
            }
        }
        let (r1, d1, g, s2, d2) = (node(5), node(6), node(7), node(8), node(9));
        for (k, z) in [(r1, 50), (d1, 10), (g, 40), (s2, 60), (d2, 20)] {
            state.z[k] = z;
        }
        state.route();
        (state, d1, g, s2, d2)
    }

    /// ★ T6 (plan step 2b, question 1; Fill–Spill–Merge's link from an ocean-bound depression to
    /// the depression its overflow first reaches): A FULL HOLLOW WHOSE SADDLE FALLS TO THE SEA'S SIDE
    /// POURS DOWN THE GROUND INTO THE LOWER HOLLOW THERE. `D1` met the sea over its sill at 50;
    /// `D2` meets the sea's component at 60, and the ground from that saddle runs down `g` into
    /// `D1`'s pit. `D2` fills and its overflow ENTERS `D1`, which holds it all: nothing reaches the
    /// sea, and the row into the sea is the cascade's nothing. Before step 2b the cascade booked the
    /// overflow straight to the sea while the ground carried it into `D1`.
    #[test]
    fn an_overflow_toward_the_sea_enters_the_lower_hollow_the_ground_leads_into() {
        let (state, d1, g, s2, d2) = ocean_side_field();
        let n = state.node_count();
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, d2, 5_000, 0);
        restate(&mut climate, d1, 0, 20_000);
        let (lakes, t) = follow(&state, &climate);
        let (h1, h2) = (lakes.home[d1] as usize, lakes.home[d2] as usize);
        assert_eq!(lakes.deps[h1].parent, OCEAN, "D1 met the sea first");
        assert_eq!(lakes.deps[h2].parent, OCEAN);
        assert_eq!(lakes.deps[h2].spill_node, s2 as u32);
        assert!(lakes.deps[h2].full);
        assert_eq!(lakes.report.ocean_pours, 1);
        assert_eq!(lakes.report.to_ocean, 0, "the overflow stays in D1");
        assert_eq!(
            lakes.deps[h1].held,
            5_000 * u128::from(state.area[d2]),
            "D1 holds D2's whole overflow"
        );
        assert_eq!(lakes.deps[h2].exit, g as u32);
        assert_eq!(t.receiver[s2], g as u32);
        assert_eq!(t.receiver[g], d1 as u32);
        assert_eq!(t.receiver[d1], NO_NODE);
        assert_tree_laws(&t, &lakes);
    }

    /// The field of three hollows meeting at one node (test 6): the saddle `v` at 60 and the pits
    /// `p1` 10 (a diagonal), `p2` 20 (a diagonal) and `p3` 30 (a row) round it. By the chord the
    /// row to `p3` (30 over one node) is steeper than the diagonal to `p2` (40 over √2 nodes).
    fn three_hollows() -> (MacroSolve, usize, usize, usize, usize) {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        let node = |i: i32, j: i32| state.lattice.index(Face::NegZ, i, j) as usize;
        let (v, p1, p2, p3) = (node(40, 40), node(39, 39), node(41, 39), node(40, 41));
        for (k, z) in [(v, 60), (p1, 10), (p2, 20), (p3, 30)] {
            state.z[k] = z;
        }
        state.route();
        (state, v, p1, p2, p3)
    }

    /// ★ T7 (plan step 2b, question 2): AT A SADDLE OF THREE SIDES THE OVERFLOW TAKES THE STEEPEST
    /// OPEN SIDE, which is where the saddle node's own water runs. `p1` alone is wet: it fills and
    /// its overflow runs to `p3` (the steepest side that is not full), never to `p2` (the side the
    /// binary merge paired it with, which is where it went before step 2b); the saddle's receiver is
    /// `p3`. With `p3` wet as well and `p2` thirsty, both spilling hollows leave by ONE exit into
    /// `p2`, and their two sinks stand at the one saddle. With `p3` alone wet, its overflow runs into
    /// `p1`, and the empty ring between `p1` and `p2` stays open because `p1` is not full.
    #[test]
    fn a_saddle_of_three_sides_pours_into_its_steepest_open_side() {
        let (state, v, p1, p2, p3) = three_hollows();
        let n = state.node_count();
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, p1, 800, 0);
        let (lakes, t) = follow(&state, &climate);
        let (l1, l2, l3) = (
            lakes.home[p1] as usize,
            lakes.home[p2] as usize,
            lakes.home[p3] as usize,
        );
        let saddle = &lakes.saddles[lakes.deps[l1].saddle as usize];
        assert_eq!(saddle.node, v as u32);
        assert_eq!(
            saddle.sides.iter().map(|k| k.root).collect::<Vec<_>>(),
            vec![l1 as u32, l3 as u32, l2 as u32],
            "steepest first: the diagonal of 50, the row of 30, the diagonal of 40"
        );
        assert!(lakes.deps[l1].full);
        assert!(!lakes.deps[l3].full);
        assert_eq!(lakes.deps[l3].held, 800 * u128::from(state.area[p1]));
        assert_eq!(lakes.deps[l2].held, 0, "the paired sibling gets nothing");
        assert_eq!(lakes.report.saddle_pours, 1);
        assert_eq!(t.receiver[v], p3 as u32);
        assert_tree_laws(&t, &lakes);
        // Two spilling hollows at one saddle: one exit, two sinks at one node.
        let mut both = flat_climate(n, 0, 1_000);
        restate(&mut both, p1, 800, 0);
        restate(&mut both, p3, 500, 0);
        restate(&mut both, p2, 0, 5_000);
        let (lakes, t) = follow(&state, &both);
        let (l1, l2, l3) = (
            lakes.home[p1] as usize,
            lakes.home[p2] as usize,
            lakes.home[p3] as usize,
        );
        assert!(lakes.deps[l1].full);
        assert!(lakes.deps[l3].full);
        assert!(!lakes.deps[l2].full);
        assert_eq!(lakes.body_of[l1], l1 as u32);
        assert_eq!(lakes.body_of[l3], l3 as u32);
        assert_eq!(lakes.deps[l1].exit, p2 as u32);
        assert_eq!(lakes.deps[l3].exit, p2 as u32);
        assert_eq!(t.receiver[v], p2 as u32);
        assert_eq!(t.spill_sinks.len(), 1, "one sink at the one saddle");
        assert_eq!(t.spill_sinks[0].0, v as u32);
        // ★ THE ROW AT THE SHARED SADDLE counts each drop once: the rain on `p1` and on `p3`
        //   (neither lake holds any, its pit's own rain pays its own evaporation). The cascade's
        //   `passed` counts `p1`'s water at `p1`, which poured it into `p3` while `p3` was still
        //   open, and again at `p3`, which passed it on when it filled — so no sum of `passed`
        //   is the row (review finding 2 of step 2b).
        let (w1, w3) = (
            800 * u128::from(state.area[p1]),
            500 * u128::from(state.area[p3]),
        );
        assert_eq!((lakes.deps[l1].held, lakes.deps[l3].held), (0, 0));
        assert_eq!(u128::from(t.discharge[v]), w1 + w3);
        assert_eq!(
            (lakes.deps[l1].passed, lakes.deps[l3].passed),
            (w1, w1 + w3)
        );
        assert_tree_laws(&t, &lakes);
        // `p3` alone wet: its overflow runs into `p1`, the steepest side that is not full, and
        // `p1` holds it. The EMPTY RING between `p1` and `p2` (the binary merge's pair, with no
        // node of its own) is full only when both its hollows are, so it stays open here.
        let mut third = flat_climate(n, 0, 1_000);
        restate(&mut third, p3, 500, 0);
        let (lakes, t) = follow(&state, &third);
        let (l1, l3) = (lakes.home[p1] as usize, lakes.home[p3] as usize);
        let pair = lakes.deps[l1].parent as usize;
        assert_eq!(
            lakes.deps[pair].saddle, lakes.deps[l1].saddle,
            "the pair closes at the saddle its hollows close at"
        );
        assert!(lakes.deps[l3].full);
        assert!(!lakes.deps[l1].full);
        assert!(!lakes.deps[pair].full, "an empty ring over an open hollow");
        assert_eq!(lakes.deps[l1].held, 500 * u128::from(state.area[p3]));
        assert_eq!(lakes.deps[l3].exit, p1 as u32);
        assert_eq!(t.receiver[v], p1 as u32);
        assert_tree_laws(&t, &lakes);
    }

    /// ★ T8 (plan step 2b, question 2): THE HOLDINGS DO NOT DEPEND ON THE ORDER THE WATERSHEDS
    /// ARRIVE IN. On the moon's full solve under the home planet's words, and on the three hollows
    /// with every pit wet, the cascade run in the index order, in the reverse order and in an order
    /// shuffled by a hash holds the same water in every ring, fills the same rings and hands the sea
    /// the same water.
    #[test]
    fn the_cascade_holds_the_same_water_in_any_arrival_order() {
        let moon = home_moon();
        let words = crate::solve::SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let (state, _, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(words.age_yr));
        let lattice = moon.macro_lattice().expect("a lattice");
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let solved = budget(&state, &climate);
        let (three, _, p1, p2, p3) = three_hollows();
        let mut wet = flat_climate(three.node_count(), 0, 1_000);
        for (p, rain) in [(p1, 800), (p2, 300), (p3, 500)] {
            restate(&mut wet, p, rain, 0);
        }
        let stated = budget(&three, &wet);
        for lakes in [&solved, &stated] {
            let shed: Vec<u128> = lakes
                .deps
                .iter()
                .enumerate()
                .map(|(d, dep)| {
                    if (d == 0) | (dep.children[0] != NO_DEP) {
                        0
                    } else {
                        dep.supply
                    }
                })
                .collect();
            let natural = cascade(&lakes.deps, &lakes.saddles, &shed);
            let m = shed.len() as u32;
            let reversed: Vec<u32> = (1..m).rev().collect();
            let mut shuffled: Vec<u32> = (1..m).collect();
            shuffled.sort_by_key(|&d| child_seed(7, 11, u64::from(d)));
            for order in [reversed, shuffled] {
                let flow = cascade_in(&lakes.deps, &lakes.saddles, &shed, &order);
                assert_eq!(flow.held_own, natural.held_own);
                assert_eq!(flow.full, natural.full);
                assert_eq!(
                    (flow.to_ocean, flow.stranded),
                    (natural.to_ocean, natural.stranded)
                );
            }
        }
    }

    /// ★ T9 (plan step 2b, question 4): THE FLAT STEP RUNS STRAIGHT TO THE SPILL AND NOT ALONG THE
    /// GRID. A round bowl of radius 12 nodes on `NegY`, walled at 100 and floored at 10 in the land
    /// at 30, spills through a notch cut 20° off the face's row. Wet, it fills and spills. From
    /// every node of its hollow six or more steps from the notch, six steps of the shipped tree
    /// point at the notch with a mean of `1 − cos(error)` under that of 12° — which by Jensen
    /// (`1 − cos` is convex there) bounds the MEAN ERROR under 12° and is read with no
    /// transcendental (the float fence) — and at most half as many of them run straight along one
    /// lattice offset as with the rule before the split (the nearest by the summed chord).
    #[test]
    fn the_flat_step_runs_straight_to_the_spill_and_not_along_the_grid() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![30; n];
        state.sea_z = 0;
        widen_the_sea(&mut state);
        let face = Face::NegY;
        let (ci, cj, r) = (34i32, 34i32, 12i32);
        for i in 0..68 {
            for j in 0..68 {
                let d2 = (i - ci) * (i - ci) + (j - cj) * (j - cj);
                let z = if d2 <= r * r {
                    10
                } else if d2 <= (r + 2) * (r + 2) {
                    100
                } else {
                    continue;
                };
                state.z[state.lattice.index(face, i, j) as usize] = z;
            }
        }
        // The notch: the wall's two cells on the ray at 20°, by the ray's own integer steps
        // (tan 20° ≈ 0.364 ≈ 4/11).
        for (di, dj) in [(12, 5), (13, 5), (14, 5)] {
            state.z[state.lattice.index(face, ci + di, cj + dj) as usize] = 50;
        }
        state.route();
        let mut climate = flat_climate(n, 0, 1_000);
        for i in 0..68 {
            for j in 0..68 {
                if (i - ci) * (i - ci) + (j - cj) * (j - cj) <= r * r {
                    restate(
                        &mut climate,
                        state.lattice.index(face, i, j) as usize,
                        2_000,
                        0,
                    );
                }
            }
        }
        let (lakes, t) = follow(&state, &climate);
        let pit = state.lattice.index(face, ci, cj) as usize;
        let body = lakes.body_of[lakes.home[pit] as usize] as usize;
        assert!(lakes.deps[body].full, "the bowl spills");
        let spill = lakes.deps[body].spill_node;
        let (_, si, sj) = state.lattice.split(spill);
        // The control: the rule before the split, over the same summed chord from the spill.
        let region: Vec<bool> = (0..n)
            .map(|k| {
                let h = lakes.home[k];
                (h != OCEAN) && (lakes.body_of[h as usize] == body as u32)
            })
            .collect();
        let mut dist = vec![u64::MAX; n];
        dist[spill as usize] = 0;
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::from([Reverse((0, spill))]);
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            // The bowl lies inside one face, so no neighbour here is missing.
            for m in state.lattice.neighbours(u) {
                let step = d + u64::from(state.chord_m(u, m));
                if region[m as usize] & (step < dist[m as usize]) {
                    dist[m as usize] = step;
                    heap.push(Reverse((step, m)));
                }
            }
        }
        let control = |i: u32| -> u32 {
            let mut best = (u64::MAX, NO_NODE);
            for m in state.lattice.neighbours(i) {
                let near =
                    (region[m as usize] | (m == spill)) & (dist[m as usize] < dist[i as usize]);
                if near {
                    best = best.min((dist[m as usize] + u64::from(state.chord_m(i, m)), m));
                }
            }
            best.1
        };
        // Six steps from a start, as face offsets (whole numbers). The bearing error is read below
        // through its COSINE in f64: the dot of the walk and the line over the product of their
        // lengths, one f64 square root. The float fence permits a square root (IEEE-754 rounds it
        // the same on every chip); no transcendental and no angle is computed.
        let walk = |start: u32, step: &dyn Fn(u32) -> u32| -> (i32, i32, bool) {
            let (_, i0, j0) = state.lattice.split(start);
            let mut cur = start;
            let mut first = (0, 0);
            let mut straight = true;
            for k in 0..6 {
                let next = step(cur);
                let (_, a, b) = state.lattice.split(cur);
                let (_, c, d) = state.lattice.split(next);
                if k == 0 {
                    first = (c - a, d - b);
                }
                straight &= (c - a, d - b) == first;
                cur = next;
            }
            let (_, i1, j1) = state.lattice.split(cur);
            (i1 - i0, j1 - j0, straight)
        };
        let (mut starts, mut sum_err, mut runs, mut control_runs) = (0i64, 0f64, 0usize, 0usize);
        // A start whose six steps stay in the bowl: a step shortens the summed chord by at most a
        // diagonal's chord (about 11.6 km on the moon).
        for k in 0..n {
            if !region[k] | (dist[k] < 6 * 11_700) {
                continue;
            }
            let (_, i0, j0) = state.lattice.split(k as u32);
            let (li, lj) = (f64::from(si - i0), f64::from(sj - j0));
            let (wi, wj, straight) = walk(k as u32, &|i| t.receiver[i as usize]);
            let (_, _, control_straight) = walk(k as u32, &control);
            let (wi, wj) = (f64::from(wi), f64::from(wj));
            // cos of the error = dot / (|w| |l|), in f64; the error is compared through its
            // cosine (cos 12° = 0.97815), so no angle is read.
            let cos = (wi * li + wj * lj) / ((wi * wi + wj * wj) * (li * li + lj * lj)).sqrt();
            sum_err += 1.0 - cos;
            starts += 1;
            runs += usize::from(straight);
            control_runs += usize::from(control_straight);
        }
        assert!(starts > 100, "{starts} starts");
        // The mean of (1 − cos error) under that of a 12° error everywhere.
        let mean = sum_err / starts as f64;
        assert!(mean <= 1.0 - 0.978_147_6, "mean 1 − cos {mean}");
        assert!(
            runs * 2 <= control_runs,
            "exact runs {runs} against the control's {control_runs} of {starts}"
        );
    }

    /// ★ T13 (plan step 2b): WITH NO EVAPORATION EVERY HOLLOW THAT GETS WATER SPILLS, and the whole
    /// runoff reaches the sea through the tree.
    #[test]
    fn with_no_evaporation_the_whole_runoff_reaches_the_sea() {
        let (state, _, _, _) = stated_field();
        let (lakes, t) = follow(&state, &flat_climate(state.node_count(), 1_000, 0));
        assert_eq!(lakes.report.evaporated_total, 0);
        assert!(lakes.report.runoff_total > 0);
        assert_eq!(lakes.report.to_ocean, lakes.report.runoff_total);
        assert_tree_laws(&t, &lakes);
    }

    /// ★ T14 (plan step 2b): WITH NO RUNOFF EVERY ROW IS NOTHING, and the tree still stands.
    #[test]
    fn with_no_runoff_every_row_is_nothing() {
        let (state, a, b, _) = stated_field();
        let (lakes, t) = follow(&state, &flat_climate(state.node_count(), 0, 1_000));
        assert!(t.discharge.iter().all(|&d| d == 0));
        assert!(t.spill_sinks.is_empty());
        assert_eq!(t.receiver[a as usize], NO_NODE, "a dry pit ends");
        assert_eq!(t.receiver[b as usize], NO_NODE);
        assert_eq!(t.order.len(), t.node_count());
        assert_tree_laws(&t, &lakes);
    }

    /// ★ T15 (plan step 2b): A SPILL ACROSS A FACE SEAM POINTS ACROSS THE SEAM, and the artifact's
    /// receiver slot names that neighbour (the slot is found by position in the node's own ring,
    /// through the seam table).
    #[test]
    fn a_spill_across_a_seam_points_across_it_and_the_slot_names_it() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let edge = state.lattice.edge as i32;
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        let s = state.lattice.index(Face::PosX, edge - 1, 30) as usize;
        let a = state.lattice.index(Face::PosX, edge - 2, 30) as usize;
        let across: Vec<u32> = state
            .lattice
            .neighbours(s as u32)
            .into_iter()
            .filter(|&m| m != NO_NODE)
            .filter(|&m| state.lattice.split(m).0 != Face::PosX)
            .collect();
        let b = across[across.len() / 2] as usize;
        for (k, z) in [(s, 50), (a, 10), (b, 20)] {
            state.z[k] = z;
        }
        state.route();
        // A little water on `A` alone: `A` fills, and its overflow is less than `B` can hold.
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, a, 500, 0);
        let (lakes, t) = follow(&state, &climate);
        assert!(lakes.deps[lakes.home[a] as usize].full);
        assert!(!lakes.deps[lakes.home[b] as usize].full);
        assert_eq!(t.receiver[s], b as u32, "the spill points across the seam");
        let facies = t.facies(&vec![false; n], 0, true);
        let artifact = crate::artifact::Artifact::of(&t, &facies, &climate, true);
        let slot = artifact.rows[s].receiver_facies & crate::artifact::RECEIVER_SLOT_MASK;
        assert_eq!(
            artifact.rows[s].receiver_facies & crate::artifact::RECEIVER_NONE,
            0
        );
        assert_eq!(t.lattice.neighbours(s as u32)[usize::from(slot)], b as u32);
        assert_tree_laws(&t, &lakes);
    }

    /// ★ T16 (plan step 2b): A POCKET UNDER THE SEA'S LEVEL THAT FILLS runs to its own spill like
    /// any spilling hollow, and its water reaches the sea by the ground.
    #[test]
    fn a_pocket_that_fills_runs_to_its_spill() {
        let (mut state, q, wall) = pocket_field();
        let n = state.node_count();
        state.route();
        let mut climate = flat_climate(n, 0, 1_000);
        restate(&mut climate, q, 100_000, 0);
        let (lakes, t) = follow(&state, &climate);
        let body = lakes.body_of[lakes.home[q] as usize] as usize;
        assert!(lakes.deps[body].full, "the pocket fills");
        assert_ne!(t.receiver[q], NO_NODE);
        assert_ne!(t.receiver[wall], NO_NODE);
        assert!(lakes.report.to_ocean > 0);
        assert_tree_laws(&t, &lakes);
    }

    /// ★ QUESTION 3 (plan step 2b): A FLAT THAT DRAINS OPENS NO HOLLOW. The whole moon stands at
    /// one height but one node of sea: by the index alone the sweep opened a leaf of zero depth
    /// wherever a node was reached before any neighbour; swept from its drain, no node of the flat
    /// opens one, and every node's water runs to the sea.
    #[test]
    fn a_flat_that_drains_opens_no_hollow() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        let sea = state.lattice.index(Face::PosX, 3, 3);
        state.z[sea as usize] = 0;
        state.route();
        let lakes = budget(&state, &flat_climate(n, 1_000, 1_000));
        assert_eq!(lakes.report.leaves, 0);
        assert_eq!(lakes.report.depressions, 0);
        assert!(lakes.descent.end.iter().all(|&e| e == sea));
        // Every step on the flat nears its drain.
        let flat = flat_distance(&state);
        for i in 0..n {
            let d = lakes.descent.down[i];
            if (d != NO_NODE) && (d != sea) {
                assert!(flat[d as usize] < flat[i], "node {i}");
            }
        }
    }

    /// ★ QUESTION 3 (plan step 2b): A CLOSED FLAT IS ONE HOLLOW WITH ITS PIT IN ITS MIDDLE. A
    /// square floor of five by five nodes at 50 in the plateau at 100: one leaf, its pit the middle
    /// node (the farthest from the rim), and every node of the floor descends to it.
    #[test]
    fn a_closed_flat_is_one_hollow_with_its_pit_in_its_middle() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        let floor: Vec<usize> = (20..25)
            .flat_map(|i| (20..25).map(move |j| (i, j)))
            .map(|(i, j)| state.lattice.index(Face::NegZ, i, j) as usize)
            .collect();
        for &k in &floor {
            state.z[k] = 50;
        }
        state.route();
        let lakes = budget(&state, &flat_climate(n, 0, 1_000));
        assert_eq!(lakes.report.leaves, 1);
        let middle = state.lattice.index(Face::NegZ, 22, 22);
        let leaf = lakes.home[middle as usize] as usize;
        assert_eq!(lakes.deps[leaf].pit, middle);
        for &k in &floor {
            assert_eq!(lakes.descent.end[k], middle, "node {k}");
        }
    }

    /// ★ BARNES' TWO FIELDS AS ONE POTENTIAL (phase 1 of the ground step): on a flat that drains to
    /// the sea and touches a block of higher ground, `Φ − 2·T` is `A_max − A` — largest beside the
    /// block, zero at the flat's node farthest from it — every node off a drain has a neighbour of
    /// smaller `Φ` on its flat (no false pit), and the descent steps strictly down `Φ`.
    #[test]
    fn the_flats_potential_leaves_the_higher_ground_first() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        let sea = state.lattice.index(Face::PosX, 3, 3);
        state.z[sea as usize] = 0;
        let block: Vec<usize> = (30..33)
            .flat_map(|i| (30..33).map(move |j| (i, j)))
            .map(|(i, j)| state.lattice.index(Face::PosX, i, j) as usize)
            .collect();
        for &k in &block {
            state.z[k] = 200;
        }
        state.route();
        let (root, flats, order) = root_and_order(&state);
        let on_flat = |k: usize| (state.z[k] == 100) & !root[k];
        let lift = |k: usize| flats.phi[k] - 2 * flats.dist[k];
        let beside = state.lattice.index(Face::PosX, 29, 31) as usize;
        let a_max = (0..n)
            .filter(|&k| on_flat(k))
            .map(lift)
            .max()
            .expect("a flat");
        assert_eq!(
            lift(beside),
            a_max,
            "beside the block the lift is the flat's largest"
        );
        assert_eq!((0..n).filter(|&k| on_flat(k)).map(lift).min(), Some(0));
        for &v in &order {
            let k = v as usize;
            if !on_flat(k) || flats.dist[k] == 0 {
                continue;
            }
            let lower = state.lattice.neighbours(v).iter().any(|&m| {
                (m != NO_NODE) && on_flat(m as usize) && (flats.phi[m as usize] < flats.phi[k])
            });
            assert!(lower, "node {k} has a neighbour of smaller potential");
        }
        let descent = descent_over(&state, &order, &flats);
        for k in 0..n {
            let d = descent.down[k];
            if on_flat(k) && (flats.dist[k] > 0) {
                assert!(
                    flats.phi[d as usize] < flats.phi[k],
                    "node {k} steps down the potential"
                );
            }
        }
    }

    /// A flat that touches no higher ground adds nothing to its potential: `Φ = 2·T` everywhere.
    #[test]
    fn a_flat_with_no_higher_ground_is_twice_its_distance() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        state.route();
        let (_, flats, _) = root_and_order(&state);
        assert!(flats.dist.iter().any(|&d| d > 0));
        for k in 0..n {
            assert_eq!(flats.phi[k], 2 * flats.dist[k], "node {k}");
        }
    }

    /// A closed flat's middle stands at the bottom of its potential, and the floor's rim carries
    /// the flat's largest lift.
    #[test]
    fn a_closed_flats_middle_is_the_bottom_of_its_potential() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 3, 3) as usize] = 0;
        let floor: Vec<usize> = (20..25)
            .flat_map(|i| (20..25).map(move |j| (i, j)))
            .map(|(i, j)| state.lattice.index(Face::NegZ, i, j) as usize)
            .collect();
        for &k in &floor {
            state.z[k] = 50;
        }
        state.route();
        let (_, flats, _) = root_and_order(&state);
        let middle = state.lattice.index(Face::NegZ, 22, 22) as usize;
        assert_eq!(flats.phi[middle], 0);
        for &k in &floor {
            if k != middle {
                assert!(flats.phi[k] > 0, "node {k}");
            }
        }
        let corner = state.lattice.index(Face::NegZ, 20, 20) as usize;
        let lift = |k: usize| flats.phi[k] - 2 * flats.dist[k];
        let a_max = floor.iter().map(|&k| lift(k)).max().expect("a floor");
        assert_eq!(lift(corner), a_max);
    }

    /// ★ THE DOWNHILL PREFERS NO DIRECTION OF THE LATTICE: at a face's middle, a potential that
    /// falls evenly along the row, along the diagonal and at 30° and 120° off the row gives a
    /// downhill parallel to the fall within about 1.8° (the sine squared under 10⁻³, read with no
    /// transcendental; MEASURED 0.24–0.94° over the four falls — the coarse moon's own stencil,
    /// whose face middle lies between two nodes, not a preferred direction), and a ring with no fall
    /// gives none.
    #[test]
    fn the_downhill_points_down_the_potential_in_every_direction() {
        #![allow(
            clippy::cast_precision_loss,
            reason = "a test reads a direction's sine"
        )]
        let moon = home_moon();
        let state = MacroSolve::new(&moon).expect("a state");
        let (face, ci, cj) = (Face::PosX, 34i32, 34i32);
        let c = state.lattice.index(face, ci, cj);
        let at = state.dir_of(c);
        let off = |i: i32, j: i32| -> [f64; 3] {
            let d = state.dir_of(state.lattice.index(face, i, j));
            [
                f64::from(d[0] - at[0]),
                f64::from(d[1] - at[1]),
                f64::from(d[2] - at[2]),
            ]
        };
        let (ex, ey) = (off(ci + 1, cj), off(ci, cj + 1));
        for (gx, gy) in [(1_000i64, 0i64), (707, 707), (866, 500), (-500, 866)] {
            let base = 1_000_000i64;
            let ring: Vec<(u32, i64)> = state
                .lattice
                .neighbours(c)
                .iter()
                .map(|&m| {
                    let (_, i, j) = state.lattice.split(m);
                    let phi = base + gx * i64::from(i - ci) + gy * i64::from(j - cj);
                    (m, phi)
                })
                .collect();
            let t = downhill(&state, c, base, &ring);
            let t = [t[0] as f64, t[1] as f64, t[2] as f64];
            let want = [
                -(gx as f64) * ex[0] - (gy as f64) * ey[0],
                -(gx as f64) * ex[1] - (gy as f64) * ey[1],
                -(gx as f64) * ex[2] - (gy as f64) * ey[2],
            ];
            let dot = t[0] * want[0] + t[1] * want[1] + t[2] * want[2];
            let cross = [
                t[1] * want[2] - t[2] * want[1],
                t[2] * want[0] - t[0] * want[2],
                t[0] * want[1] - t[1] * want[0],
            ];
            let c2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
            let n2 = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
                * (want[0] * want[0] + want[1] * want[1] + want[2] * want[2]);
            assert!(dot > 0.0, "the fall ({gx}, {gy}) points down");
            assert!(c2 < 1.0e-3 * n2, "the fall ({gx}, {gy}): sine² {}", c2 / n2);
        }
        let level: Vec<(u32, i64)> = state
            .lattice
            .neighbours(c)
            .iter()
            .map(|&m| (m, 5_000))
            .collect();
        assert_eq!(downhill(&state, c, 5_000, &level), [0, 0, 0]);
    }

    /// ★ QUESTION 3 (plan step 2b): A CLOSED FLAT ROUND A CUBE CORNER is one hollow as well. Only
    /// three faces meet at a corner, so the corner cell of `PosX` has SEVEN neighbours and its
    /// eighth slot is [`NO_NODE`]. Every node within two steps of it stands at 50 in the plateau at
    /// 100: the flat's rim is its outer ring, the corner is the node farthest from it, and every
    /// node of the flat descends to the corner.
    ///
    /// **Example.** A crater bowl the sediment filled flat sits where three faces of the lattice
    /// meet; its floor still drains to one pit in its middle, never toward the missing eighth
    /// neighbour.
    #[test]
    fn a_closed_flat_round_a_cube_corner_is_one_hollow_with_its_pit_at_the_corner() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![100; n];
        state.sea_z = 0;
        state.z[state.lattice.index(Face::PosX, 30, 30) as usize] = 0;
        let corner = state.lattice.index(Face::PosX, 0, 0);
        assert!(state.lattice.neighbours(corner).contains(&NO_NODE));
        let mut floor: Vec<u32> = vec![corner];
        for _ in 0..2 {
            let ring: Vec<u32> = floor
                .iter()
                .flat_map(|&v| state.lattice.neighbours(v))
                .filter(|&m| m != NO_NODE)
                .collect();
            floor.extend(ring);
            floor.sort_unstable();
            floor.dedup();
        }
        for &k in &floor {
            state.z[k as usize] = 50;
        }
        state.route();
        let lakes = budget(&state, &flat_climate(n, 0, 1_000));
        assert_eq!(lakes.report.leaves, 1);
        let leaf = lakes.home[corner as usize] as usize;
        assert_eq!(lakes.deps[leaf].pit, corner);
        for &k in &floor {
            assert_eq!(lakes.descent.end[k as usize], corner, "node {k}");
        }
    }

    /// ★ THE FLAT STEP'S THREE WAYS (plan step 2b, question 4), each stated: with no neighbour
    /// ahead of the node toward its target it takes the nearest by distance plus chord; with one
    /// ahead it takes that one; with two or more it splits between the two nearest the line.
    #[test]
    fn the_flat_step_takes_the_nearest_the_one_ahead_or_the_split() {
        let moon = home_moon();
        let state = MacroSolve::new(&moon).expect("a state");
        let at = |i: i32, j: i32| state.lattice.index(Face::PosZ, i, j);
        let i = at(30, 30);
        let east = at(40, 30);
        // None ahead: the two stated neighbours lie west of the node.
        let behind = [(at(29, 30), 20), (at(29, 31), 10)];
        assert_eq!(flat_step(&state, i, east, &behind), (at(29, 31), 0));
        // One ahead.
        let one = [(at(31, 30), 50), (at(29, 30), 10)];
        assert_eq!(flat_step(&state, i, east, &one), (at(31, 30), 1));
        // Three ahead: the row and the two diagonals; the target on the row, so the row and one
        // diagonal bracket it and the split takes one of them, never the far diagonal.
        let three = [(at(31, 29), 5), (at(31, 30), 5), (at(31, 31), 5)];
        // The split's chance: a target exactly on the row gives the row's offset zero, so p1 = 0
        // and the farther neighbour is never taken.
        assert_eq!(flat_step(&state, i, east, &three), (at(31, 30), 2));
    }

    /// ★ THE SPLIT'S CHANCE, read directly (review finding 6 of step 2b): over 20 000 nodes under
    /// the `FLOW` salt, the split takes the FARTHER neighbour with the chance `p1 / (p1 + p2)`.
    ///
    /// 1. A bearing of about 20° off the row (the line `(11, 4)`, `tan = 4/11`): the row's squared
    ///    offset is `4² = 16`, the diagonal's `(4 − 11)² = 49`, so `p1 = 4`, `p2 = 7` and the chance
    ///    is `4/11`. A swap of `p1` and `p2` reads `7/11` and fails.
    /// 2. The same law past the shift: squared offsets of 101 and 104 bits (`1 · 2¹⁰⁰` and
    ///    `9 · 2¹⁰⁰`), wider than a 64-bit word, so the even shift runs (42 bits); it leaves
    ///    `2⁵⁸` and `9 · 2⁵⁸`, whose roots are `2²⁹` and `3 · 2²⁹`, and the chance `1/4`.
    ///
    /// The bound is five standard deviations of a binomial count of 20 000 draws (68 draws at
    /// `4/11`, 61 at `1/4`): a fair hash leaves one such bound about once in 1.7 million seeds.
    ///
    /// **Example.** A lake on the plain spills at a notch 20° off the face's row. Of every eleven
    /// nodes on the lake, about four take the diagonal step and seven the row step, so the water
    /// runs toward the notch and not along the row.
    #[test]
    fn the_split_takes_the_farther_neighbour_with_the_chance_p1_over_p1_plus_p2() {
        let seed = MacroSolve::new(&home_moon()).expect("a state").seed;
        let (near, far) = (7u32, 9u32);
        let draws = 20_000u32;
        let count = |a: u128, b: u128| -> i64 {
            (0..draws)
                .filter(|&i| split(seed, i, (a, near), (b, far)) == far)
                .count() as i64
        };
        // 1. About 20° off the row: the chance 4/11.
        let farther = count(16, 49);
        let n = i64::from(draws);
        let expected = 4 * n / 11;
        assert!(
            (11 * farther - 4 * n).abs() <= 11 * 340,
            "{farther} of {draws} took the farther neighbour; 4/11 expects {expected}"
        );
        // 2. Past the shift: the chance 1/4.
        let wide = count(1 << 100, 9 << 100);
        let quarter = n / 4;
        assert!(
            (4 * wide - n).abs() <= 4 * 306,
            "{wide} of {draws} took the farther neighbour; 1/4 expects {quarter}"
        );
    }
}
