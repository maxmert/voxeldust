//! ★ THE DEPRESSION HIERARCHY AND THE WATER BUDGET (slice 8d step 5; ruling W11).
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
//! its SPILL, and a parent depression carries both. The sea is the root and never fills.
//!
//! **THE WATER** (Langbein 1961, *Salinity and hydrology of closed lakes*, USGS Professional Paper
//! 412, <https://pubs.usgs.gov/pp/0412/report.pdf>): a closed lake stands at the level where the
//! standing water's evaporation balances the inflow. Every node states three whole numbers the
//! climate already computes — its rain `P`, its potential evaporation `PET`
//! ([`crate::climate::pet_mm_yr`]) and its RUNOFF `Q` ([`crate::climate::runoff_mm_yr`], the
//! Turc–Pike partition of the rain). A node's SUPPLY is `Q` over its area. A node UNDER the lake
//! instead costs `Q + PET − P` over its area, because open water evaporates at the potential rate
//! and no longer yields a river: that is the DEFICIT. So the balance at a level `L` is
//!
//! ```text
//! B(L) = Σ supply(every node of the depression) − Σ deficit(every node under L)
//! ```
//!
//! and the lake rises until `B(L) = 0`, capped at the spill. Three ends, decided by the water and
//! never by the shape of the hole: DRY (no supply, or the first node's own evaporation takes it
//! all — a playa), PARTIAL (a closed lake at its balance level), or SPILLING (`B` still positive at
//! the spill: the water leaves through an outlet, and the depression is a lake with a river out of
//! it). ★ A depression that spills hands its overflow to its parent, which is the MERGE.
//!
//! **Example.** Two hollows in the belt's dry interior share a saddle. The northern one takes a
//! river off the range and stands as a salt lake half-way up its own walls; the southern one takes
//! nothing and is a white pan a pilot can land on. Neither reaches the saddle, so the metadepression
//! that holds them both is never a lake and the pilot flies over dry ground between them.

// ★ THE BUDGET MAY DIVIDE: it runs once per body on the CPU, off the tick, never in a kernel, and
// every quotient here is an integer one.
#![allow(
    clippy::integer_division,
    reason = "the budget runs on the CPU once per body, never in a kernel"
)]

use crate::climate::Climate;
use crate::macro_lattice::NO_NODE;
use crate::solve::{DRY, MacroSolve, OUTLETS_WITHOUT_SEA, Z_STEPS_PER_M};

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
    /// The water the subtree delivers, mm·m²/yr: `Σ Q · area`, the Turc–Pike runoff.
    pub supply: u128,
    /// Whether the supply reaches the deficit at the spill: the lake fills and runs out.
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
}

/// The budget's answer: one water level a node, and the hierarchy that decided it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lakes {
    /// The standing water's level at each node, sixteenths, or [`DRY`]. The sea is NOT in here:
    /// [`MacroSolve::water_level`] reads the sea from the field itself.
    pub water: Vec<i32>,
    /// The hierarchy, index 0 the ocean's root.
    pub deps: Vec<Depression>,
    /// The depression each node was added to.
    pub home: Vec<u32>,
    /// For each depression, the water body that covers it (itself, or an ancestor that floods over
    /// it); [`NO_DEP`] for the ocean.
    pub body_of: Vec<u32>,
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
        supply: 0,
        full: false,
        level: DRY,
    }
}

/// ★ THE BUDGET over a solved state and the climate that stands over it. The state's OWN field
/// decides the hierarchy — never the routing flood, which is a scratch surface. A body with no
/// water gets no lake: every hollow on it is a closed basin.
///
/// **Example.** The home planet's solve ends, the sea is settled, and the budget walks 8.87 million
/// nodes once: it finds the hollows, hands each one its own rivers, and says which of them hold
/// water. What it writes is one level a node, and the artifact's water word is that level.
#[must_use]
pub fn budget(state: &MacroSolve, climate: &Climate) -> Lakes {
    let n = state.node_count();
    let z = &state.z;
    let sea_z = state.sea_z;
    let mut home = vec![NO_DEP; n];
    let mut done = vec![false; n];
    let mut deps = vec![leaf(NO_NODE, i32::MAX)];
    deps[0].spill_z = i32::MAX;
    let mut link = vec![OCEAN];
    // 1. THE ROOT: every node at or under the sea. A body whose water settled no level takes the
    //    same six lowest nodes the routing takes ([`OUTLETS_WITHOUT_SEA`]), so the hierarchy and
    //    the receivers agree about where the water leaves.
    let mut order: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        if z[i] <= sea_z {
            home[i] = OCEAN;
            done[i] = true;
        } else {
            order.push(i as u32);
        }
    }
    order.sort_unstable_by_key(|&v| (z[v as usize], v));
    if order.len() == n {
        let seeds = OUTLETS_WITHOUT_SEA.min(order.len());
        for &v in order.iter().take(seeds) {
            home[v as usize] = OCEAN;
            done[v as usize] = true;
        }
        order.drain(0..seeds);
    }
    // 2. THE SWEEP, lowest first.
    let mut roots: Vec<u32> = Vec::with_capacity(8);
    for &v in &order {
        let iv = v as usize;
        let z_v = z[iv];
        roots.clear();
        for m in state.lattice.neighbours(v) {
            if m == NO_NODE || !done[m as usize] {
                continue;
            }
            let r = find(&mut link, home[m as usize]);
            if !roots.contains(&r) {
                roots.push(r);
            }
        }
        roots.sort_unstable();
        let holder = match roots.len() {
            0 => {
                let d = deps.len() as u32;
                deps.push(leaf(v, z_v));
                link.push(d);
                d
            }
            1 => roots[0],
            _ => {
                let mut acc = roots[0];
                for &other in roots.iter().skip(1) {
                    acc = merge(&mut deps, &mut link, acc, other, v, z_v);
                }
                acc
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
    // 3. A COMPONENT THAT NEVER MET ANOTHER can never spill: it is closed at no level and holds
    //    whatever its own supply keeps.
    for d in 1..deps.len() {
        if link[d] == d as u32 {
            deps[d].full = false;
        }
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
    //    nodes, lowest first, until the standing water's evaporation eats the whole supply.
    let mut report = LakeReport {
        depressions: deps.len() - 1,
        leaves: deps[1..].iter().filter(|d| d.children[0] == NO_DEP).count(),
        ..LakeReport::default()
    };
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
    dep.supply += rate.0;
    dep.deficit += rate.1;
}

/// Two components meeting at node `v`: the sea takes the other one as a child and stays the root;
/// two hollows make a new parent that holds them both. A closing depression states its spill, its
/// parent and whether its own water reaches the brim.
fn merge(deps: &mut Vec<Depression>, link: &mut Vec<u32>, a: u32, b: u32, v: u32, z_v: i32) -> u32 {
    if a == OCEAN {
        close(deps, b, v, z_v, OCEAN);
        link[b as usize] = OCEAN;
        return OCEAN;
    }
    let p = deps.len() as u32;
    let (pit, pit_z) = if (deps[a as usize].pit_z, deps[a as usize].pit)
        <= (deps[b as usize].pit_z, deps[b as usize].pit)
    {
        (deps[a as usize].pit, deps[a as usize].pit_z)
    } else {
        (deps[b as usize].pit, deps[b as usize].pit_z)
    };
    let mut parent = leaf(pit, pit_z);
    parent.children = [a, b];
    parent.area_m2 = deps[a as usize].area_m2 + deps[b as usize].area_m2;
    parent.z_area = deps[a as usize].z_area + deps[b as usize].z_area;
    parent.supply = deps[a as usize].supply + deps[b as usize].supply;
    parent.deficit = deps[a as usize].deficit + deps[b as usize].deficit;
    deps.push(parent);
    link.push(p);
    close(deps, a, v, z_v, p);
    close(deps, b, v, z_v, p);
    link[a as usize] = p;
    link[b as usize] = p;
    p
}

/// A depression closes at `v`: its spill, its parent, and the verdict of its own water budget.
fn close(deps: &mut [Depression], d: u32, v: u32, z_v: i32, parent: u32) {
    let dep = &mut deps[d as usize];
    dep.spill_z = z_v;
    dep.spill_node = v;
    dep.parent = parent;
    dep.full = dep.supply > 0 && dep.supply >= dep.deficit;
}

/// ★ THE CLOSED LAKE'S LEVEL: the body's own nodes lowest first, wetted a step of equal heights at
/// a time while the standing water's evaporation stays inside the supply. The first step the supply
/// cannot pay for stops the water, so a hollow with a small river holds a small lake and a hollow
/// with none holds a pan.
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
        let level = fill_level(own, state, climate, deps[d as usize].supply);
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
    use crate::solve::{FACIES_LAKE, Schedule, solve_full};
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
        //    it, so it stands one sixteenth over the floor; the second hollow keeps the ordinary
        //    climate and fills to the saddle the two share.
        let mut some = flat_climate(n, 500, 500);
        let feeder = state.lattice.index(Face::NegZ, 21, 20) as usize;
        restate(&mut some, a as usize, 2_000, 500);
        restate(&mut some, feeder, 500, 9_000);
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
