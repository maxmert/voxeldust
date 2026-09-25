//! ★ THE DEPRESSION HIERARCHY AND THE WATER BUDGET (slice 8d step 5; ruling W11; the watershed and
//! the cascade by ruling B7, 2026-09-24).
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
//! nodes are swept in one stated order, lowest first, ties to the smaller index. A node whose lower
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
//! deficit is FULL, and its OVERFLOW runs on over its spill: into the sibling hollow on the other
//! side of the saddle, down the ground to that sibling's own pit; where the sibling is full too,
//! into the parent's own ring; where the parent is the ocean, into the sea. Nothing is lost:
//! `runoff = evaporated + to the ocean + stranded` to the unit ([`LakeReport`]). Before, a full
//! hollow's overflow into a dry parent vanished.
//!
//! Three ends, decided by the water and never by the shape of the hole: DRY (no inflow, or the first
//! node's own evaporation takes it all — a playa), PARTIAL (a closed lake at its balance level), or
//! SPILLING (the inflow still passes the deficit at the spill: the water leaves through an outlet,
//! and the depression is a lake with a river out of it).
//!
//! **Example.** A hollow lies at the foot of the belt's wet range. The ground of the whole range face
//! runs down into its pit, so it fills to its sill. Its overflow crosses the saddle into the dry pan
//! next to it, which becomes a salt lake half-way up its walls. Nothing is lost, and the metadepression
//! that holds both is never a lake while the pan stands under its own spill.

// ★ THE BUDGET MAY DIVIDE: it runs once per body on the CPU, off the tick, never in a kernel, and
// every quotient here is an integer one.
#![allow(
    clippy::integer_division,
    reason = "the budget runs on the CPU once per body, never in a kernel"
)]

use crate::climate::Climate;
use crate::macro_lattice::NO_NODE;
use crate::solve::{DRY, MacroSolve, OUTLETS_WITHOUT_SEA, Z_STEPS_PER_M, steeper};

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
    /// over its nodes — the runoff they stop yielding plus what the open surface hands the air.
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
    /// — its steepest lower neighbour of the spill node inside it; [`NO_NODE`] where it never met
    /// another.
    pub gate: u32,
    /// ★ THE WATER THE SUBTREE HOLDS after the cascade, mm·m²/yr: its own ring's share plus its
    /// children's, never more than [`Depression::deficit`].
    pub held: u128,
    /// Whether its inflow reached its deficit and the water ran on over its spill.
    pub full: bool,
    /// The water's level, sixteenths, or [`DRY`].
    pub level: i32,
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
    /// How many times a full hollow's overflow poured into the sibling across its spill.
    pub transfers: usize,
    /// ★ How many rings the cascade's walk visited over every arrival (review finding 6 of plan
    /// step 2): the measure of the skip [`cascade`] keeps over full rings.
    pub cascade_steps: usize,
}

/// ★ THE WATER'S OWN WAY ON THE GROUND (2026-09-24, ruling B7; Fill–Spill–Merge's flow
/// directions on the unfilled field): each node's steepest lower neighbour and where that descent
/// ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descent {
    /// The steepest neighbour lower by the budget's own order `(height, index)`, so a flat still
    /// descends by its index; [`NO_NODE`] at a pit and at a root node.
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
    }
}

/// ★ THE ROOT: every node of the CONNECTED OCEAN ([`MacroSolve::ocean`]; 2026-09-24, ruling B7 item
/// 1), or — on a body whose water settled no ocean — the same six lowest nodes the routing takes
/// ([`OUTLETS_WITHOUT_SEA`]), so the hierarchy and the receivers agree about where the water
/// leaves. Returns the root mask and every other node, lowest first by `(height, index)`.
fn root_and_order(state: &MacroSolve) -> (Vec<bool>, Vec<u32>) {
    let n = state.node_count();
    let z = &state.z;
    let mut root = state.ocean.clone();
    let mut order: Vec<u32> = (0..n as u32).filter(|&i| !root[i as usize]).collect();
    order.sort_unstable_by_key(|&v| (z[v as usize], v));
    if order.len() == n {
        let seeds = OUTLETS_WITHOUT_SEA.min(order.len());
        for &v in order.iter().take(seeds) {
            root[v as usize] = true;
        }
        order.drain(0..seeds);
    }
    (root, order)
}

/// ★ THE DESCENT OVER THE GROUND for every node, with the root stated: each node that is not a root
/// takes its steepest neighbour lower by `(height, index)` ("steepest" is the solve's own
/// [`steeper`] over the lattice's chord), and the descent ends at a root or at a node with no lower
/// neighbour — a PIT, which opened exactly one leaf of the hierarchy.
///
/// **Example.** Rain on a spur of the belt's range runs down the ground, never down the fill. The
/// descent follows it into the hollow at the range's foot and stops at the hollow's pit, even where
/// the routing fill carries the same rain over the hollow's brim to the sea.
#[must_use]
pub fn descent(state: &MacroSolve) -> Descent {
    let (_, order) = root_and_order(state);
    descent_over(state, &order)
}

/// The descent over `order` (every node that is not a root, lowest first): a root keeps
/// [`NO_NODE`] as its way down and itself as its end.
fn descent_over(state: &MacroSolve, order: &[u32]) -> Descent {
    let n = state.node_count();
    let z = &state.z;
    let mut down = vec![NO_NODE; n];
    let mut end: Vec<u32> = (0..n as u32).collect();
    for &v in order {
        let iv = v as usize;
        let mut best = NO_NODE;
        let (mut dz_best, mut l_best) = (0i64, 0u32);
        for m in state.lattice.neighbours(v) {
            if m == NO_NODE || (z[m as usize], m) >= (z[iv], v) {
                continue;
            }
            let dz = i64::from(z[iv]) - i64::from(z[m as usize]);
            let l = state.chord_m(v, m);
            if best == NO_NODE || steeper(dz, l, m, dz_best, l_best, best) {
                best = m;
                dz_best = dz;
                l_best = l;
            }
        }
        down[iv] = best;
        if best != NO_NODE {
            end[iv] = end[best as usize];
        }
    }
    Descent { down, end }
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
    let (root, order) = root_and_order(state);
    let mut home = vec![NO_DEP; n];
    let mut done = vec![false; n];
    let mut deps = vec![leaf(NO_NODE, i32::MAX)];
    deps[0].spill_z = i32::MAX;
    let mut link = vec![OCEAN];
    for i in 0..n {
        if root[i] {
            home[i] = OCEAN;
            done[i] = true;
        }
    }
    // 1. THE SWEEP, lowest first: the hierarchy, and at every merge the gate of each side.
    let mut met: Vec<Met> = Vec::with_capacity(8);
    for &v in &order {
        let iv = v as usize;
        let z_v = z[iv];
        met.clear();
        for m in state.lattice.neighbours(v) {
            if m == NO_NODE || !done[m as usize] {
                continue;
            }
            let r = find(&mut link, home[m as usize]);
            let dz = i64::from(z_v) - i64::from(z[m as usize]);
            let chord = state.chord_m(v, m);
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
                let mut acc = met[0];
                for &other in met.iter().skip(1) {
                    let p = merge(&mut deps, &mut link, acc, other, v, z_v);
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
        add_cell(
            &mut deps,
            holder,
            z_v,
            state.area[iv],
            rates(state, climate, iv),
        );
        home[iv] = holder;
        done[iv] = true;
    }
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
    let descent = descent_over(state, &order);
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
    // 3. THE CASCADE (Fill–Spill–Merge's overflow, with rates in place of volumes).
    let entry: Vec<u32> = deps
        .iter()
        .map(|d| {
            if d.gate == NO_NODE {
                NO_DEP
            } else {
                home[descent.end[d.gate as usize] as usize]
            }
        })
        .collect();
    let flow = cascade(&deps, &entry, &shed);
    report.to_ocean += flow.to_ocean;
    report.stranded = flow.stranded;
    report.transfers = flow.transfers;
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
        let up = if p == OCEAN || p == NO_DEP {
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
        if h == NO_DEP || h == OCEAN {
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
fn merge(deps: &mut Vec<Depression>, link: &mut Vec<u32>, a: Met, b: Met, v: u32, z_v: i32) -> u32 {
    if a.root == OCEAN {
        close(deps, b, v, z_v, OCEAN);
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
    close(deps, a, v, z_v, p);
    close(deps, b, v, z_v, p);
    link[ra] = p;
    link[rb] = p;
    p
}

/// A depression closes at `v`: its spill, its parent, and its gate.
fn close(deps: &mut [Depression], met: Met, v: u32, z_v: i32, parent: u32) {
    let dep = &mut deps[met.root as usize];
    dep.spill_z = z_v;
    dep.spill_node = v;
    dep.parent = parent;
    dep.gate = met.gate;
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
    /// How many times an overflow poured into the sibling across its spill.
    pub transfers: usize,
    /// How many rings the walk visited over every arrival.
    pub steps: usize,
}

/// ★ THE CASCADE (Fill–Spill–Merge's `OverflowInto`, Barnes, Callaghan & Wickert 2021 §2.4, with
/// yearly rates in place of volumes; 2026-09-24, ruling B7; status §7 W4). Each leaf's watershed
/// arrives at the leaf, in index order. Water reaching a depression fills its OWN ring first; a
/// ring whose water reaches its own deficit is FULL, and the rest runs on:
///
/// 1. into the SEA where the parent is the ocean;
/// 2. nowhere (STRANDED, counted) where the depression never met another — it never spills, so it
///    is never marked full;
/// 3. into the SIBLING across the spill where the sibling is not full — at the leaf the water's
///    descent from the sibling's gate ends in (`entry`);
/// 4. into the PARENT's own ring where the sibling is full too.
///
/// Fullness only grows, so the final holdings do not depend on the order the leaves are taken in.
///
/// ★ THE SKIP (review finding 6 of plan step 2). A ring that is full and whose sibling is full too
/// sends every later arrival on to its parent, FOR GOOD, because fullness only grows. The walk
/// records that edge (`up`) and jumps a chain of such rings at once, with path halving — the
/// union–find the hierarchy itself is built with. Without it every arrival at a deep leaf of a
/// long nested lake climbed every full ring again: `O(leaves × depth)`, about 10¹⁰ steps on a
/// lattice at the edge ceiling with one basin of 10⁵ nested leaves. The holdings are the same
/// walk's, ring for ring; [`Cascade::steps`] counts what the walk visited.
///
/// **Example.** A leaf at the range's foot takes the whole range face and passes its deficit. Its
/// overflow crosses the saddle and runs down to the pit of the dry pan beside it, which holds it as a
/// salt lake; nothing reaches the pair's own ring, because the pan is not full.
#[must_use]
pub fn cascade(deps: &[Depression], entry: &[u32], shed: &[u128]) -> Cascade {
    let m = deps.len();
    let mut flow = Cascade {
        held_own: vec![0; m],
        full: vec![false; m],
        ..Cascade::default()
    };
    let mut up: Vec<u32> = (0..m as u32).collect();
    for (d, &water) in shed.iter().enumerate().skip(1) {
        if water > 0 {
            arrive(deps, entry, &mut flow, &mut up, d as u32, water);
        }
    }
    flow
}

/// Water `water` reaching depression `x`, carried on until it stands. `up` is the skip over rings
/// that are full with a full sibling ([`cascade`]).
fn arrive(
    deps: &[Depression],
    entry: &[u32],
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
        let p = deps[xi].parent;
        if p == OCEAN {
            flow.to_ocean += water;
            return;
        }
        if p == NO_DEP {
            flow.stranded += water;
            return;
        }
        let kids = deps[p as usize].children;
        let sibling = if kids[0] == x { kids[1] } else { kids[0] };
        if flow.full[sibling as usize] {
            // Both rings full: every later arrival here runs on to the parent, for good.
            up[xi] = p;
            x = p;
        } else {
            flow.transfers += 1;
            x = entry[sibling as usize];
        }
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
        if h == NO_DEP || h == OCEAN {
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
    use crate::solve::{FACIES_LAKE, Schedule, solve_full, steeper};
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

    /// The stated field's five trough nodes, west to east: the pit `A`, its wall, the saddle, the
    /// second hollow's wall and its pit `B`.
    fn trough(state: &MacroSolve) -> [usize; 5] {
        [20, 21, 22, 23, 24].map(|i| state.lattice.index(Face::NegZ, i, 20) as usize)
    }

    /// ★ AN INDEPENDENT WALK DOWN THE GROUND, for the tests alone: from `i`, step to the steepest
    /// neighbour lower by `(height, index)` until no neighbour is lower or the node is the ocean.
    fn walk_down(state: &MacroSolve, mut i: usize) -> usize {
        loop {
            if state.ocean[i] {
                return i;
            }
            let mut best: Option<(usize, i64, u32)> = None;
            for m in state.lattice.neighbours(i as u32) {
                if m == NO_NODE || (state.z[m as usize], m) >= (state.z[i], i as u32) {
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
            match best {
                Some((m, _, _)) => i = m,
                None => return i,
            }
        }
    }

    /// ★ TEST 1 (plan step 2): A HOLLOW'S SUPPLY IS ITS WHOLE WATERSHED. Every node yields water;
    /// the pit `A`'s supply equals the runoff summed over every node whose independent walk down the
    /// ground ends at `A` — and that watershed reaches far past `A`'s own two nodes, which were all
    /// the budget booked before ruling B7 (`brim_supply`).
    #[test]
    fn a_hollows_supply_is_the_runoff_of_its_whole_watershed() {
        let (state, a, _, _) = stated_field();
        let n = state.node_count();
        let climate = flat_climate(n, 1_000, 1_000);
        let lakes = budget(&state, &climate);
        let leaf = lakes.home[a as usize] as usize;
        let mut hand = 0u128;
        let mut shed_nodes = 0usize;
        for i in 0..n {
            if !state.ocean[i] && walk_down(&state, i) == a as usize {
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
                walk_down(&state, i),
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
        let (state, _, report) =
            solve_full(&moon, &words, Schedule::standard(words.age_yr)).expect("a solve");
        let l = &report.lakes;
        assert!(l.runoff_total > 0);
        assert_eq!(l.runoff_total, l.evaporated_total + l.to_ocean + l.stranded);
        assert_eq!(
            l.stranded, 0,
            "every hollow of a connected lattice meets another"
        );
        let lattice = moon.macro_lattice().expect("a lattice");
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
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
        let flow = cascade(&deps, &[NO_DEP, NO_DEP], &[0, 150]);
        assert_eq!(flow.held_own[1], 100);
        assert_eq!(flow.stranded, 50);
        assert!(!flow.full[1]);
        let short = cascade(&deps, &[NO_DEP, NO_DEP], &[0, 60]);
        assert_eq!((short.held_own[1], short.stranded), (60, 0));
        // Exactly its deficit: brim full, nothing left over to strand.
        let exact = cascade(&deps, &[NO_DEP, NO_DEP], &[0, 100]);
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
        let ring = |pit: u32, children: [u32; 2], deficit: u128, parent: u32| {
            let mut d = leaf(pit, 0);
            d.children = children;
            d.own_deficit = 10;
            d.deficit = deficit;
            d.parent = parent;
            d
        };
        let deps = vec![
            leaf(NO_NODE, i32::MAX),
            ring(1, [NO_DEP, NO_DEP], 10, 4),
            ring(2, [NO_DEP, NO_DEP], 10, 4),
            ring(3, [NO_DEP, NO_DEP], 10, 5),
            ring(1, [1, 2], 30, 5),
            ring(1, [4, 3], 50, OCEAN),
        ];
        let entry = [NO_DEP, 1, 2, 3, 1, NO_DEP];
        let flow = cascade(&deps, &entry, &[0, 100, 5, 5, 0, 0]);
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

    /// ★ THE FULL SOLVE'S OWN LAKES on the moon's lattice under the home planet's words: the budget
    /// runs, every body ends dry, partial or spilling, and every node the facies call a lake stands
    /// under the level the budget settled — never under the routing fill, which raised far more.
    #[test]
    fn the_solve_states_a_lake_only_where_the_budget_put_water() {
        let moon = home_moon();
        // The home planet's own air and climate with a sea that covers PART of the moon — the
        // stand `solve.rs` states, so the two tests read one body.
        let words = crate::solve::SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let (state, facies, report) =
            solve_full(&moon, &words, Schedule::standard(words.age_yr)).expect("a solve");
        assert!(report.lakes.depressions > 0);
        assert_eq!(
            report.lakes.bodies,
            report.lakes.dry + report.lakes.partial + report.lakes.spilling
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
        assert_eq!(lake_nodes, report.lakes.lake_nodes);
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
            solve_full(&moon, &words, Schedule::standard(words.age_yr)).expect("a solve");
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
}
