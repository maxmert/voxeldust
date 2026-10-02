//! ★ THE STANDS RULE AND THE BODIES' CENSUS — phase 0 of the ground step (ruling B35 item 3; ruling B36 item 8,
//! the owner's Q16; the design `docs/investigation/2026-10-02/ground_step_design.md` §2.2; benches M0-5 and M0-6).
//!
//! **WHAT IT DOES.**
//! 1. **THE BODIES (M0-5).** Every round body of the home system — each planet as the system's shard boots it, and
//!    each moon as its planet's own shard boots it — with its charter's facts, as the body's own realm states them
//!    (the tool's path, `vd_bins::body_charter`), and THE THIRD BODY by a stated rule: the solid body, other than the
//!    home planet and its moon, whose lattice node the 8 192 m TARGET sets as on the home planet (the edge ceiling does
//!    not bind, so its checks read the same scales; the owner's choice of 2026-10-02, ruling B36 item 9), with the
//!    largest charter distance from the home planet over the decimal logarithms of its water, its surface pressure,
//!    its surface temperature and its gravity. A DRY world WITH AIR (less water than the home planet, and air) is
//!    preferred; ties go to the smaller seed.
//! 2. **THE STANDS (M0-6)** on the home planet, its moon and the third body, each solved in this process by the
//!    shard's own function on the shard's own words (`vd_bins::artifact_worker::run_solve`):
//!    - **CLASS STANDS.** A node's class is its RELIEF and its ROCK. The relief is Hammond's (Hammond 1964;
//!      Karagulle et al. 2017; ESTIMATE sources), read on the node and its eight neighbours (about 24.6 km a side):
//!      the share of gentle nodes (the steepest slope to a neighbour under 8 %) and the local relief (the highest
//!      less the lowest top) — PLAINS (half or more gentle, under 90 m), TABLELANDS (half or more gentle, 90 m or
//!      more), HILLS (under half gentle, under 300 m), MOUNTAINS (under half gentle, 300 m or more) — plus the SEA
//!      FLOOR (the solve's sea bit) and the CLOSED-BASIN FLOOR (a plain whose water ends inland, by the receiver
//!      tree). A LAND node reads its LAND neighbours only, in its slope and in its window. The relief is ONE rule,
//!      `vd_bins::stands::relief_classes`, which the Earth instrument reads too (`earth_relief.rs`). The rock is the
//!      node's province. A class stands when its largest patch covers at least one view's area (π × 30² km² in the body's
//!      own nodes: 42 on the home planet), at the node of that patch farthest from the patch's edge by the summed
//!      chord (the flats' own middle rule, `vd_terrain::lakes`); among nodes equally far, the most central (the
//!      smallest summed chord to the patch's members), then the smaller index.
//!      ★ MEASURED, first run: NO HILLS CLASS stands on any body. On nodes of 8.2 km, a slope of 8 % across one node
//!      is already 655 m of relief, so a window cannot be steep and under 300 m at once; the eroded 1 km level of
//!      phase 2 resolves hills, and the rule then ADDS their stands.
//!    - **BORDER STANDS.** The three nodes with the most classes within 60 km by the summed chord, at least 60 km
//!      apart, read from the nodes whose two face indices are multiples of four (a 32 km grid; ties to the smaller
//!      index).
//!    - **RANDOM STANDS.** Directions uniform over the sphere from the body's seed through the seed crate's own hash,
//!      split by the land and the sea floor: on the home planet 20 on land and 4 on the sea floor, on the moon 8, on
//!      the third body 12 split by its land share.
//!    - **TODAY'S STANDS** (the home planet): the belt, the coast, the plain, the two pockets and the high lake the
//!      earlier steps pinned; the old reference pose kept as the SUMMIT stand (status §9: it stands on a summit and
//!      looks down); the ground stand and the dusk stand where the owner judged D5 and D6 (ruling B34). The moving-eye
//!      gate's 15° stand is NOT pinned: it is where the star stands 15° up at the run's tick, so it follows the clock.
//! 3. **THE PINS.** With no pins yet, the picks print as the table for `vd_bins::stands::PINNED`. With pins, each
//!    pin prints its node and class today beside the rule's picks, and nothing moves (ruling B6).
//!
//! **WHAT WAITS, STATED SO IT IS NOT FORGOTTEN.** The CLIMATE stands — the climate zone in the class, the tree line,
//! the snow line, the desert scree slope and the two sides of a rain shadow — wait for the corrected climate
//! (ruling B36 item 1: no stand rests on the climate until it runs; phase 2). The FAMILY stands wait for each family's
//! facts (phase 4), the forest stands for the cover (phase 6). Each is ADDED then; none moves a pin.
//!
//! `cargo run --release -p vd-bins --example stands_rule [bodies | home-moon | <seed>]`: `bodies` prints the census
//! alone, `home-moon` stands on the home planet and its moon alone, a seed names the third body to solve beside the
//! rule's own pick.
//!
//! **Example (MEASURED, the first run, `docs/investigation/2026-10-02/phase0_stands_census.txt`).** On the home planet
//! the rule reads 8 871 936 nodes, 77.7 % of them sea floor, and pins 24 class stands; the largest patch of plains on
//! crystalline basement holds 36 158 nodes, and its stand stands 229 km from that patch's edge. On the moon no sea
//! floor exists, its 6 classes stand on its basins, tablelands and mountains, and its 8 random stands are all on
//! land. On the desert world no sea exists either, and closed basins on deep sediment cover 47 % of its nodes.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_bins::drainage_gates::{BELT, COAST, HIGH_LAKE, PLAIN, POCKET_1, POCKET_2, node_under};
use vd_bins::stands::{PINNED, RELIEF, StandBody, StandLayer, relief_classes};
use vd_core::geometry::{Boundary, RealmRegion};
use vd_core::glam::DVec3;
use vd_core::look::{BodyCharter, CHARTER_FLAG_HAS_AIR, CHARTER_FLAG_SOLID_SURFACE};
use vd_core::pose::RealmId;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, FACIES_SHIFT, RECEIVER_NONE, RECEIVER_SLOT_MASK};
use vd_terrain::home::{
    HOME_MOON_SEED, HOME_PLANET_SEED, HOME_UNIVERSE_SEED, home_moon, home_moon_solve_words,
    home_planet, home_solve_words,
};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{FACIES_SEA, SolveWords};
use vd_terrain::strata::Province;

/// One view's reach, metres: a class stands when its largest patch covers π × 30² km² (the design §2.2) — 42 nodes of
/// 8 192 m (67.1 km² each, ARITH), and fewer on a body whose nodes are larger.
const VIEW_REACH_M: f64 = 30_000.0;
/// The border stands' reach and their spacing, metres (the design §2.2: "the most classes within 60 km").
const BORDER_REACH_M: u64 = 60_000;
/// The border stands read the nodes whose two face indices are multiples of this (a 32 km grid).
const BORDER_GRID: i32 = 4;
/// How many border stands each body holds.
const BORDER_STANDS: usize = 3;
/// The random stands' salt in the seed crate's hash: the ASCII of "STANDS".
const RANDOM_SALT: u64 = 0x5354_414e_4453;
/// The random stands each body holds: (land, sea floor) on the home planet; all on the moon; split by the land share
/// on the third body (the design §2.2).
const HOME_RANDOM: (usize, usize) = (20, 4);
const MOON_RANDOM: usize = 8;
const THIRD_RANDOM: usize = 12;

/// The old reference pose (status §9, generator 19): the summit stand.
const SUMMIT: [f64; 3] = [0.631_321, -0.422_643, -0.650_236];
/// The ground stand and the dusk stand of the owner's window session (ruling B34; the spawn poses of accounts 1000
/// and 1001), as positions in the home planet's frame: only their directions are read.
const GROUND_STAND_POS: [f64; 3] = [
    -5_938_058.346_628_912,
    -1_425_733.848_185_895_2,
    -1_723_149.034_800_371_9,
];
const DUSK_STAND_POS: [f64; 3] = [
    -6_122_789.629_295_482,
    442_015.686_202_653_33,
    -1_606_257.696_818_383_4,
];

fn province_name(code: u8) -> &'static str {
    match code {
        c if c == Province::CrystallineBasement as u8 => "crystalline basement",
        c if c == Province::FoldedBelt as u8 => "folded belt",
        c if c == Province::FlatShelf as u8 => "flat shelf",
        c if c == Province::RiftBasalt as u8 => "rift basalt",
        c if c == Province::DeepSediment as u8 => "deep sediment",
        _ => "unknown rock",
    }
}

/// A class's code: the relief times eight plus the province (under 64, so a class set is one `u64`).
fn class_name(class: u8) -> String {
    format!(
        "{} / {}",
        RELIEF[usize::from(class >> 3)],
        province_name(class & 7)
    )
}

/// One round body of the home system.
struct BodyRow {
    realm: RealmId,
    seed: u64,
    parent: Option<RealmId>,
    radius_m: f64,
    charter: Option<BodyCharter>,
}

/// The four decades the third body's distance reads: water, surface pressure, surface temperature, gravity.
fn decades(c: &BodyCharter) -> [f64; 4] {
    [
        (c.water_km3.unwrap_or(0) as f64 + 1.0).log10(),
        (f64::from(c.p_surf_pa.unwrap_or(0)) + 1.0).log10(),
        (f64::from(c.t_surface_mk.unwrap_or(c.t_eq_mk)) / 1_000.0).log10(),
        (f64::from(c.gravity_mm_s2) / 1_000.0).log10(),
    ]
}

fn distance(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// A round body's seed and look radius, from its realm row.
fn round(r: &RealmRegion) -> Option<(u64, f64)> {
    match (r.realm, r.look) {
        (RealmId::Planet(seed), Some(Boundary::Shell { r: radius_m })) => Some((seed, radius_m)),
        _ => None,
    }
}

/// M0-5: every round body of the home system — each planet the system's shard boots, and each moon its planet's own
/// shard boots (the planet's subtree, held by the planet, as `home_body_pin.rs` reads the home moon) — sorted by seed.
fn bodies() -> Vec<BodyRow> {
    let config =
        vd_physics::worldgen::UniverseConfig::world(vd_bins::DEV.move_speed, vd_bins::DEV.tick_dt);
    let system = vd_core::worldgen::HOME_SYSTEM;
    let galaxy = vd_core::worldgen::GALAXY;
    let (rows, _) = vd_bins::home_system_world(HOME_UNIVERSE_SEED);
    let mut out = Vec::new();
    for p in rows.iter().filter(|r| r.parent == Some(system)) {
        let Some((seed, radius_m)) = round(p) else {
            continue;
        };
        out.push(BodyRow {
            realm: p.realm,
            seed,
            parent: p.parent,
            radius_m,
            charter: vd_bins::body_charter(
                HOME_UNIVERSE_SEED,
                &BTreeSet::from([system]),
                &BTreeSet::from([galaxy]),
                p.realm,
            ),
        });
        let held = BTreeSet::from([p.realm]);
        let lineage = BTreeSet::from([galaxy, system]);
        let (moons, _) = vd_physics::worldgen::shard_boot_world(
            HOME_UNIVERSE_SEED,
            &config,
            &held,
            p.realm,
            &lineage,
        );
        for m in moons.iter().filter(|r| r.parent == Some(p.realm)) {
            let Some((seed, radius_m)) = round(m) else {
                continue;
            };
            out.push(BodyRow {
                realm: m.realm,
                seed,
                parent: m.parent,
                radius_m,
                charter: vd_physics::worldgen::body_charter_in_subtree(
                    HOME_UNIVERSE_SEED,
                    &config,
                    &held,
                    &lineage,
                    m.realm,
                ),
            });
        }
    }
    out.sort_by_key(|b| b.seed);
    out
}

/// A body the ladder and the solve accept: its definition from its own charter, and its macro lattice.
fn definition(row: &BodyRow) -> Option<(BodyDefinition, SolveWords)> {
    let charter = row.charter?;
    let body =
        BodyDefinition::from_seed(row.seed, row.radius_m, vd_bins::facts_of_charter(&charter))?;
    body.macro_lattice()?;
    Some((body, vd_bins::solve_words_of(&charter)))
}

/// The divisor rule's answer with NO memory ceiling: the divisor `d ≥ MIN_MACRO_EDGE` of `n` whose node `n / d` is
/// nearest the target, ties to the smaller `d` — `vd_terrain::macro_lattice::macro_edge`'s own rule, read without
/// `MACRO_EDGE_CEILING`. Where it equals a body's edge, the TARGET set the body's node, as it did the home planet's.
fn edge_without_ceiling(n: u32) -> Option<u32> {
    let target = vd_terrain::macro_lattice::MACRO_CELL_TARGET_M;
    let floor = vd_terrain::macro_lattice::MIN_MACRO_EDGE;
    let mut best: Option<(u64, u32)> = None;
    let mut d = 1u32;
    while u64::from(d) * u64::from(d) <= u64::from(n) {
        if n.is_multiple_of(d) {
            for e in [d, n / d] {
                if e < floor {
                    continue;
                }
                let key = (u64::from(n / e).abs_diff(target), e);
                if best.is_none_or(|b| key < b) {
                    best = Some(key);
                }
            }
        }
        d += 1;
    }
    best.map(|(_, e)| e)
}

/// The third body by the rule; prints every candidate.
///
/// ★ THE SCALE CONDITION (the owner, 2026-10-02: "The desert world"): a candidate's node is the one the 8 192 m TARGET
/// sets — its lattice's edge is the divisor rule's answer with no memory ceiling — as on the home planet (1 216 nodes an
/// edge, 8 192 m at a face's centre), so every check reads the third body at the same scales. MEASURED by the first
/// census: without it the rule picked a 25 407 km world whose edge the ceiling cut to 2 048, with nodes of 19 456 m. (A first form of this
/// condition compared the node sizes as floats and refused every body: each body's whole-node edge leaves its node a
/// little off the home planet's.)
fn third_body(rows: &[BodyRow]) -> Option<u64> {
    let home_row = rows.iter().find(|b| b.seed == HOME_PLANET_SEED)?;
    let home = home_row.charter?;
    let home_d = decades(&home);
    let home_water = home.water_km3.unwrap_or(0);
    let mut best: Option<(bool, f64, u64)> = None;
    println!(
        "THE THIRD BODY'S CANDIDATES (solid, with a grid whose node the {} m target sets; the home planet and its moon \
         left out):",
        vd_terrain::macro_lattice::MACRO_CELL_TARGET_M
    );
    for b in rows {
        if b.seed == HOME_PLANET_SEED || b.seed == HOME_MOON_SEED {
            continue;
        }
        let Some(c) = b.charter else { continue };
        let solid = c.flags & CHARTER_FLAG_SOLID_SURFACE != 0;
        let Some((body, _)) = definition(b) else {
            continue;
        };
        let Some(lattice) = body.macro_lattice() else {
            continue;
        };
        if !solid {
            continue;
        }
        let air = c.flags & CHARTER_FLAG_HAS_AIR != 0 && c.p_surf_pa.unwrap_or(0) > 0;
        let dry = c.water_km3.unwrap_or(0) < home_water;
        let preferred = air && dry;
        let same_scale = edge_without_ceiling(body.ladder().n) == Some(lattice.edge);
        let d = distance(&decades(&c), &home_d);
        println!(
            "  {:?}: radius {:.0} km, distance {d:.3} decades, dry {dry}, air {air}, preferred {preferred}, {} nodes \
             of {:.0} m{}",
            b.realm,
            b.radius_m / 1_000.0,
            lattice.node_count(),
            lattice.node_m(),
            if same_scale {
                ""
            } else {
                " — the edge ceiling set its node, not the target: not a candidate"
            }
        );
        if !same_scale {
            continue;
        }
        let key = (preferred, d, b.seed);
        let better = match best {
            None => true,
            Some((p, bd, bs)) => {
                (preferred, d) > (p, bd) || ((preferred, d) == (p, bd) && b.seed < bs)
            }
        };
        if better {
            best = Some(key);
        }
    }
    best.map(|(_, _, seed)| seed)
}

/// The row's facies bits.
fn facies(a: &Artifact, v: u32) -> u8 {
    a.rows[v as usize].receiver_facies >> FACIES_SHIFT
}

/// Whether each node's water ends inland: the receiver tree followed to its end, which is the sea's or a hollow's.
fn drains_inland(lattice: &MacroLattice, a: &Artifact) -> Vec<bool> {
    let n = lattice.node_count();
    // 0 unknown, 1 the sea, 2 inland.
    let mut end = vec![0u8; n];
    let mut path: Vec<u32> = Vec::new();
    for start in 0..n as u32 {
        if end[start as usize] != 0 {
            continue;
        }
        path.clear();
        let mut v = start;
        let result = loop {
            if end[v as usize] != 0 {
                break end[v as usize];
            }
            path.push(v);
            let rf = a.rows[v as usize].receiver_facies;
            if rf & RECEIVER_NONE != 0 {
                break if facies(a, v) & FACIES_SEA != 0 { 1 } else { 2 };
            }
            let next = lattice.neighbours(v)[usize::from(rf & RECEIVER_SLOT_MASK)];
            if next == NO_NODE || path.len() > n {
                break 2;
            }
            v = next;
        };
        for &p in &path {
            end[p as usize] = result;
        }
    }
    end.iter().map(|&e| e == 2).collect()
}

/// Every node's class: the relief by the shared rule (`vd_bins::stands::relief_classes`, the land reading land only)
/// times eight, plus the province. The land is every node the solve did not make the sea; the inland bit is the
/// receiver tree's end.
fn classes(lattice: &MacroLattice, a: &Artifact) -> Vec<u8> {
    let z: Vec<i32> = a.rows.iter().map(|r| i32::from(r.z_m)).collect();
    let land: Vec<bool> = (0..lattice.node_count() as u32)
        .map(|v| facies(a, v) & FACIES_SEA == 0)
        .collect();
    let inland = drains_inland(lattice, a);
    relief_classes(lattice, &z, &land, &inland)
        .iter()
        .zip(&a.rows)
        .map(|(&relief, row)| (relief << 3) | (row.province & 7))
        .collect()
}

/// A node's unit direction.
fn dir_of(lattice: &MacroLattice, node: u32) -> DVec3 {
    let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
    let d = lattice.direction(node);
    DVec3::new(
        d[0].raw() as f64 / unit,
        d[1].raw() as f64 / unit,
        d[2].raw() as f64 / unit,
    )
    .normalize()
}

/// One pick of the rule.
struct Pick {
    layer: StandLayer,
    name: String,
    node: u32,
    note: String,
}

/// The most central of the `tied` nodes inside patch `id`: the smallest summed chord to the patch's members, then the
/// smaller index. One walk of the patch for each tied node; past `CENTRAL_WORK` node visits the smaller index decides
/// alone, and the line says so.
fn central(lattice: &MacroLattice, comp: &[u32], id: u32, tied: &[u32], size: usize) -> u32 {
    const CENTRAL_WORK: usize = 200_000_000;
    if tied.len() == 1 {
        return tied[0];
    }
    if tied.len().saturating_mul(size) > CENTRAL_WORK {
        println!(
            "  (the tie-break of {} tied nodes over a patch of {size} would visit past {CENTRAL_WORK} nodes: the smaller index decides)",
            tied.len()
        );
        return *tied.iter().min().expect("tied nodes");
    }
    let mut best: Option<(u64, u32)> = None;
    let mut dist: BTreeMap<u32, u64> = BTreeMap::new();
    for &t in tied {
        dist.clear();
        let mut heap = BinaryHeap::new();
        dist.insert(t, 0);
        heap.push(Reverse((0u64, t)));
        while let Some(Reverse((d, u))) = heap.pop() {
            if dist.get(&u).is_some_and(|&k| d > k) {
                continue;
            }
            for m in lattice.neighbours(u) {
                if m == NO_NODE || comp[m as usize] != id {
                    continue;
                }
                let step = d + u64::from(lattice.chord_m(u, m));
                if dist.get(&m).is_none_or(|&k| step < k) {
                    dist.insert(m, step);
                    heap.push(Reverse((step, m)));
                }
            }
        }
        let sum: u64 = dist.values().sum();
        if best.is_none_or(|(s, v)| (sum, t) < (s, v)) {
            best = Some((sum, t));
        }
    }
    best.expect("tied nodes").1
}

/// The class stands: one for each class whose largest patch covers a view.
fn class_stands(lattice: &MacroLattice, class: &[u8]) -> Vec<Pick> {
    let n = lattice.node_count();
    let mut comp = vec![u32::MAX; n];
    // For each class: (the largest patch's size, its first node), and the class's whole count.
    let mut largest: BTreeMap<u8, (usize, u32)> = BTreeMap::new();
    let mut count: BTreeMap<u8, usize> = BTreeMap::new();
    let mut queue = VecDeque::new();
    let mut label = 0u32;
    for start in 0..n as u32 {
        *count.entry(class[start as usize]).or_default() += 1;
        if comp[start as usize] != u32::MAX {
            continue;
        }
        let c = class[start as usize];
        comp[start as usize] = label;
        queue.push_back(start);
        let mut size = 0usize;
        while let Some(v) = queue.pop_front() {
            size += 1;
            for m in lattice.neighbours(v) {
                if m != NO_NODE && comp[m as usize] == u32::MAX && class[m as usize] == c {
                    comp[m as usize] = label;
                    queue.push_back(m);
                }
            }
        }
        let e = largest.entry(c).or_insert((0, start));
        if size > e.0 {
            *e = (size, start);
        }
        label += 1;
    }
    let mut out = Vec::new();
    // One view's area in this body's own nodes.
    let view_nodes = (std::f64::consts::PI * VIEW_REACH_M * VIEW_REACH_M
        / (lattice.node_m() * lattice.node_m()))
    .round() as usize;
    // The summed chord from a patch's edge, one word a node, reset to "unreached" for each patch.
    let mut dist = vec![u64::MAX; n];
    for (&c, &(size, first)) in &largest {
        if size < view_nodes {
            continue;
        }
        let id = comp[first as usize];
        // The patch's members (the nodes of its label), and its edge: a member with a neighbour outside it.
        let members: Vec<u32> = (0..n as u32).filter(|&v| comp[v as usize] == id).collect();
        let edge: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&v| {
                lattice
                    .neighbours(v)
                    .iter()
                    .any(|&m| m != NO_NODE && comp[m as usize] != id)
            })
            .collect();
        for &v in &members {
            dist[v as usize] = u64::MAX;
        }
        let mut heap = BinaryHeap::new();
        for &e in &edge {
            dist[e as usize] = 0;
            heap.push(Reverse((0u64, e)));
        }
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            for m in lattice.neighbours(u) {
                if m == NO_NODE || comp[m as usize] != id {
                    continue;
                }
                let step = d + u64::from(lattice.chord_m(u, m));
                if step < dist[m as usize] {
                    dist[m as usize] = step;
                    heap.push(Reverse((step, m)));
                }
            }
        }
        // The farthest from the edge (a patch with no edge, a class that covers the whole body, reads nought
        // everywhere). ★ THE TIE-BREAK: among the nodes equally far — every node of a thin range, where each member
        // touches another class (MEASURED on the first run: two ranges read 0.0 km and fell to an end) — the most
        // central, by the smallest summed chord to the patch's members, then the smaller index.
        let far = members
            .iter()
            .map(|&v| {
                if dist[v as usize] == u64::MAX {
                    0
                } else {
                    dist[v as usize]
                }
            })
            .max()
            .expect("a patch has members");
        let tied: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&v| {
                (if dist[v as usize] == u64::MAX {
                    0
                } else {
                    dist[v as usize]
                }) == far
            })
            .collect();
        let node = central(lattice, &comp, id, &tied, members.len());
        out.push(Pick {
            layer: StandLayer::Class,
            name: class_name(c),
            node,
            note: format!(
                "class {} nodes, largest patch {size}, {:.1} km from the patch's edge, {} tied",
                count[&c],
                far as f64 / 1_000.0,
                tied.len()
            ),
        });
    }
    out
}

/// The border stands: the most classes within the reach, read on the 32 km grid.
fn border_stands(lattice: &MacroLattice, class: &[u8], radius_m: f64) -> Vec<Pick> {
    let n = lattice.node_count();
    let mut stamp = vec![0u32; n];
    let mut dist = vec![0u64; n];
    let mut round = 0u32;
    let mut scored: Vec<(u32, u32)> = Vec::new();
    for v in 0..n as u32 {
        let (_, i, j) = lattice.split(v);
        if i % BORDER_GRID != 0 || j % BORDER_GRID != 0 {
            continue;
        }
        round += 1;
        let mut set = 0u64;
        let mut heap = BinaryHeap::new();
        stamp[v as usize] = round;
        dist[v as usize] = 0;
        heap.push(Reverse((0u64, v)));
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            set |= 1u64 << class[u as usize];
            for m in lattice.neighbours(u) {
                if m == NO_NODE {
                    continue;
                }
                let step = d + u64::from(lattice.chord_m(u, m));
                if step > BORDER_REACH_M {
                    continue;
                }
                if stamp[m as usize] != round || step < dist[m as usize] {
                    stamp[m as usize] = round;
                    dist[m as usize] = step;
                    heap.push(Reverse((step, m)));
                }
            }
        }
        scored.push((set.count_ones(), v));
    }
    scored.sort_by_key(|&(c, v)| (Reverse(c), v));
    let mut out: Vec<Pick> = Vec::new();
    let mut taken: Vec<DVec3> = Vec::new();
    for (c, v) in scored {
        let d = dir_of(lattice, v);
        if taken
            .iter()
            .any(|t| t.angle_between(d) * radius_m < BORDER_REACH_M as f64)
        {
            continue;
        }
        taken.push(d);
        out.push(Pick {
            layer: StandLayer::Border,
            name: format!("border {}", out.len() + 1),
            node: v,
            note: format!("{c} classes within 60 km"),
        });
        if out.len() == BORDER_STANDS {
            break;
        }
    }
    out
}

/// A direction uniform over the sphere from the seed crate's own hash.
fn uniform_dir(seed: u64, i: u64) -> DVec3 {
    let unit = |h: u64| (h >> 11) as f64 / (1u64 << 53) as f64;
    let u = unit(vd_seed::rng::child_seed(seed, RANDOM_SALT, 2 * i));
    let w = unit(vd_seed::rng::child_seed(seed, RANDOM_SALT, 2 * i + 1));
    let z = 1.0 - 2.0 * u;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * w;
    DVec3::new(r * phi.cos(), r * phi.sin(), z)
}

/// The random stands: `land` on land and `sea` on the sea floor, drawn in order until both are full.
fn random_stands(
    lattice: &MacroLattice,
    a: &Artifact,
    seed: u64,
    land: usize,
    sea: usize,
) -> Vec<Pick> {
    let mut out = Vec::new();
    let (mut got_land, mut got_sea) = (0usize, 0usize);
    let mut i = 0u64;
    while (got_land < land || got_sea < sea) && i < 100_000 {
        let node = node_under(lattice, uniform_dir(seed, i));
        i += 1;
        let wet = facies(a, node) & FACIES_SEA != 0;
        let (want, got, word) = if wet {
            (sea, &mut got_sea, "sea floor")
        } else {
            (land, &mut got_land, "land")
        };
        if *got >= want {
            continue;
        }
        *got += 1;
        out.push(Pick {
            layer: StandLayer::Random,
            name: format!("random {word} {}", *got),
            node,
            note: format!("draw {}", i - 1),
        });
    }
    out
}

/// Today's stands on the home planet.
fn today_stands(lattice: &MacroLattice) -> Vec<Pick> {
    let dir = |d: [f64; 3]| DVec3::new(d[0], d[1], d[2]).normalize();
    let at = |name: &str, node: u32, note: &str| Pick {
        layer: StandLayer::Today,
        name: name.to_owned(),
        node,
        note: note.to_owned(),
    };
    vec![
        at("the belt", node_under(lattice, dir(BELT)), "ruling B2"),
        at(
            "the coast",
            node_under(lattice, dir(COAST)),
            "pinned at generator 20",
        ),
        at(
            "the plain",
            node_under(lattice, dir(PLAIN)),
            "pinned at generator 20",
        ),
        at("pocket 1", POCKET_1, "its deepest node at generator 21"),
        at("pocket 2", POCKET_2, "its deepest node at generator 21"),
        at(
            "the high lake",
            node_under(lattice, dir(HIGH_LAKE)),
            "the step-2b aim",
        ),
        at(
            "the summit",
            node_under(lattice, dir(SUMMIT)),
            "the old reference pose (status §9)",
        ),
        at(
            "the ground stand",
            node_under(lattice, dir(GROUND_STAND_POS)),
            "D5's stand (ruling B34)",
        ),
        at(
            "the dusk stand",
            node_under(lattice, dir(DUSK_STAND_POS)),
            "D6's stand (ruling B34)",
        ),
    ]
}

/// Solve one body and pick its stands; print the census lines and return the pins' rows.
fn stands_of(
    which: StandBody,
    body: BodyDefinition,
    words: SolveWords,
    random: (usize, usize),
) -> Vec<(StandBody, StandLayer, String, DVec3)> {
    let seed = body.seed();
    let radius_m = body.radius_m();
    let lattice = MacroLattice::of(&body).expect("the body has a macro lattice");
    let started = std::time::Instant::now();
    let artifact = run_solve(&SolveJob { body, words }).expect("the body solves");
    let n = lattice.node_count();
    let wet = (0..n as u32)
        .filter(|&v| facies(&artifact, v) & FACIES_SEA != 0)
        .count();
    println!(
        "\n== {which:?}: Planet({seed}), {n} nodes, the solve {:.1} s, the sea floor {:.2} % of the nodes",
        started.elapsed().as_secs_f64(),
        100.0 * wet as f64 / n as f64
    );
    let class = classes(&lattice, &artifact);
    let mut picks = class_stands(&lattice, &class);
    picks.extend(border_stands(&lattice, &class, radius_m));
    let (land, sea) = if which == StandBody::Third {
        let land_share = 1.0 - wet as f64 / n as f64;
        let land = (THIRD_RANDOM as f64 * land_share).round() as usize;
        (land, THIRD_RANDOM - land)
    } else {
        random
    };
    picks.extend(random_stands(&lattice, &artifact, seed, land, sea));
    if which == StandBody::Home {
        picks.extend(today_stands(&lattice));
    }
    let mut rows = Vec::new();
    for p in &picks {
        let d = dir_of(&lattice, p.node);
        println!(
            "STAND {which:?} {:?} \"{}\": node {} ({:?}), {:.6} {:.6} {:.6}, z {} m, class {}; {}",
            p.layer,
            p.name,
            p.node,
            lattice.split(p.node),
            d.x,
            d.y,
            d.z,
            artifact.rows[p.node as usize].z_m,
            class_name(class[p.node as usize]),
            p.note
        );
        rows.push((which, p.layer, p.name.clone(), d));
    }
    // With pins: each pin's node and class today, beside the picks (ruling B6: nothing moves).
    for pin in PINNED.iter().filter(|p| p.body == which) {
        let node = node_under(&lattice, DVec3::from_array(pin.dir));
        println!(
            "PIN {which:?} {:?} \"{}\": node {node} today, class {}",
            pin.layer,
            pin.name,
            class_name(class[node as usize])
        );
    }
    rows
}

fn main() {
    println!(
        "THE STANDS RULE (phase 0, M0-5 and M0-6), generator {}",
        vd_terrain::GENERATOR_VERSION
    );
    let rows = bodies();
    println!(
        "\nTHE BODIES OF THE HOME SYSTEM (M0-5): {} round bodies",
        rows.len()
    );
    for b in &rows {
        match b.charter {
            Some(c) => println!(
                "  {:?} under {:?}: radius {:.0} km, g {:.3} m/s², water {} km³, pressure {} Pa, surface {} K, \
                 solid {}, air {}, grid {}",
                b.realm,
                b.parent,
                b.radius_m / 1_000.0,
                f64::from(c.gravity_mm_s2) / 1_000.0,
                c.water_km3.unwrap_or(0),
                c.p_surf_pa.unwrap_or(0),
                f64::from(c.t_surface_mk.unwrap_or(c.t_eq_mk)) / 1_000.0,
                c.flags & CHARTER_FLAG_SOLID_SURFACE != 0,
                c.flags & CHARTER_FLAG_HAS_AIR != 0,
                definition(b).is_some()
            ),
            None => println!("  {:?} under {:?}: NO CHARTER", b.realm, b.parent),
        }
    }
    let ruled = third_body(&rows).expect("the home system holds a third solid body");
    println!("THE THIRD BODY (the rule): Planet({ruled})");
    // `stands_rule bodies`: the census alone, with no solve. `stands_rule home-moon`: the home planet and its moon
    // alone. `stands_rule <seed>`: the third body the owner named, printed beside the rule's own pick.
    let mode = std::env::args().nth(1);
    if mode.as_deref() == Some("bodies") {
        return;
    }
    let mut pins = stands_of(
        StandBody::Home,
        home_planet(),
        home_solve_words(),
        HOME_RANDOM,
    );
    pins.extend(stands_of(
        StandBody::Moon,
        home_moon(),
        home_moon_solve_words(),
        (MOON_RANDOM, 0),
    ));
    let third = match mode.as_deref() {
        Some("home-moon") => None,
        Some(seed) => Some(seed.parse::<u64>().expect("the third body's seed")),
        None => Some(ruled),
    };
    if let Some(third) = third {
        println!("THE THIRD BODY SOLVED: Planet({third}) (the rule's own pick: Planet({ruled}))");
        let third_row = rows
            .iter()
            .find(|b| b.seed == third)
            .expect("the third body is a row");
        let (third_body, third_words) = definition(third_row).expect("the third body is accepted");
        pins.extend(stands_of(StandBody::Third, third_body, third_words, (0, 0)));
    }

    if PINNED.is_empty() {
        println!("\nTHE PINS (no pins yet; this table is `vd_bins::stands::PINNED`):");
        if let Some(third) = third {
            println!("pub const THIRD_BODY_SEED: u64 = {third};");
        }
        println!("pub const PINNED: &[PinnedStand] = &[");
        for (body, layer, name, d) in &pins {
            println!(
                "    PinnedStand {{ body: StandBody::{body:?}, layer: StandLayer::{layer:?}, name: \"{name}\", dir: [{:.6}, {:.6}, {:.6}] }},",
                d.x, d.y, d.z
            );
        }
        println!("];");
    }
}
