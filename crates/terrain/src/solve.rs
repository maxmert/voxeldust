//! ★ THE SOLVE — its CORE (the landform arc, slice 8c stage C1; `slice_8c_design.md` §4; the
//! investigation's `03_erosion_rivers.md` §4.3–4.6, §5).
//!
//! Once per body, on the macro lattice, a short list of passes turns a starting surface into an
//! eroded one: the water is routed downhill (D8), the pits are filled to their spill level (the
//! priority flood), the flats are resolved by a distance field, the rain is accumulated downstream
//! into a DISCHARGE, and the stream power law cuts every node toward its receiver's base level —
//! one implicit sweep, one division per node, unconditionally stable. Integers between passes;
//! the one float is the chord's square root under the fence, floored to a metre before it is
//! compared. Every order is a stated order over integers, so two hosts cannot disagree, and the
//! solve runs on ONE thread, off the tick (03 §5.3).
//!
//! **What C1 leaves out, by name** (the design's C2–C5): the initial land from plates and isostasy
//! (the starting surface here is the recipe's own coarse relief at the node's rung), the climate
//! (the rain is [`P_MIN_MM_YR`] everywhere — 03 §4.5's stated fallback: the rivers are right in
//! shape and wrong in size), the uplift, the sediment budget, the flexural rebound, the talus, the
//! ice, the craters, the coast and the artifact. C1 exists to MEASURE the core's wall time and
//! memory on THE world's bodies before those are built.
//!
//! **Example.** On the home planet's moon (27 744 nodes) the flood seeds the heap with every node
//! under the drawn sea, pops them lowest first, and raises each pit to the level at which its water
//! could leave. The rain that falls on a highland node then runs node to node down the receivers
//! to the sea, and the sweep lowers each node toward the level of the node below it, harder where
//! more water passes. A lake node is skipped, so the hole the flood found stays a lake.

// ★ A SOLVE MAY DIVIDE (ruling F7's rule is about the KERNELS a card runs). The implicit update is
// one division per node on the CPU, single-threaded, off the tick, never a GPU kernel (03 §5.3);
// the quotient is computed with `div_euclid` on 128-bit words, which is integer-exact on every host.
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the solve runs on the CPU once per body, never in a kernel; integer-exact on every host"
)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vd_recipe::Gi;
use vd_recipe::bend::DIR_BITS;
use vd_recipe::root::isqrt;

use crate::body::BodyDefinition;
use crate::gf::Gf;
use crate::height::height;
use crate::macro_lattice::{MacroLattice, NO_NODE, chord_between};
use crate::units::LENGTH_BITS;

/// The macro height's unit: sixteenths of a metre, from the ladder radius. A planet's relief is at
/// most 18 000 m — 288 000 steps, four decimal orders under the word.
pub const Z_STEPS_PER_M: i32 = 16;
/// The shift from the recipe's length word (gap steps of 1/128 m at [`LENGTH_BITS`] fraction bits)
/// to sixteenths: `128 / 16 = 2³`.
const Z_SHIFT: u32 = LENGTH_BITS + 3;
/// The water level's word for a node that holds no standing water.
pub const DRY: i32 = i32::MIN;
/// The rain's floor, in millimetres a year: a hyper-arid basin still carries a river's SHAPE, and a
/// logarithm cannot hold zero (03 §4.5). Until the climate lands (C3), this is the rain everywhere.
pub const P_MIN_MM_YR: u32 = 1;
/// The rain's ceiling, stated ONCE: Earth's wettest station, Mawsynram, is about 11 900 mm/yr.
pub const P_MAX_MM_YR: u32 = 10_000;
/// A body with no node under its sea has no outlet; the flood then seeds the heap with this many
/// lowest nodes by `(height, index)` — one per face, a stated world constant (03 §4.4).
pub const OUTLETS_WITHOUT_SEA: usize = 6;

// ★ THE SCHEDULE's COST KNOBS (the design's ask 5, the recommended values taken; part of the world
// tag, so a change is a new world; 03 §5.5). They are NUMERICAL RESOLUTION — how finely the age is
// stepped — never a dial: the physical dial is the age, and it is the system's own.
/// The stream-power sweeps the age is divided into.
pub const PASSES: u32 = 40;
/// The climate is recomputed over the relief as it then stands every this many passes (C3).
pub const CLIMATE_EVERY: u32 = 10;
/// The flood, the flats and the receivers are recomputed every this many passes.
pub const FLOOD_EVERY: u32 = 10;
/// The flexural rebound runs every this many passes (C3).
pub const ISOSTASY_EVERY: u32 = 5;
/// The talus relaxations after the last sweep (C3).
pub const TALUS_PASSES: u32 = 8;

/// ★ THE ERODIBILITY, calibration body EARTH: `K₀ = 2 × 10⁻⁶ /yr` for the stream power law at
/// `m = ½, n = 1` with the drainage AREA in square metres — the middle of the published range
/// (Stock & Montgomery 1999, 10⁻⁶ to 10⁻⁴; Whipple & Tucker 1999). The law here reads the
/// DISCHARGE in mm·m²/yr, so the constant carries `1/√(mm per m)` with it ([`Schedule::gain`]).
/// C3 ties it down against the per-basin hypsometric integral (gate G-AGE); C1 states it.
pub const K0_PER_YR: f64 = 2.0e-6;
/// Millimetres in a metre: the discharge's unit against the law's.
pub const MM_PER_M: f64 = 1_000.0;
/// The fraction bits of the sweep's coefficient `Δt·K·√Q/L`.
pub const GAIN_BITS: u32 = 24;
/// The bits a cached direction keeps: `[i32; 3]` at 30 fraction bits, so the direction table costs
/// twelve bytes a node; the chord it feeds is floored to a whole metre, and a step of 2⁻³⁰ is six
/// millimetres on the home planet.
const CACHE_BITS: u32 = 30;

/// The schedule of one solve: the age the passes step through, and the two counts C1 reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// ★ THE EROSIONAL AGE in years: the SYSTEM'S OWN age from the census (the design's ask 3).
    pub age_yr: u64,
    pub passes: u32,
    pub flood_every: u32,
}

impl Schedule {
    /// The standard schedule for a body of a system `age_yr` old.
    #[must_use]
    pub const fn standard(age_yr: u64) -> Schedule {
        Schedule {
            age_yr,
            passes: PASSES,
            flood_every: FLOOD_EVERY,
        }
    }

    /// ★ THE SWEEP'S GAIN: `Δt · K₀/√(mm per m) · 2^GAIN_BITS`, floored — the one product of the
    /// law's constant and the age's step, computed ONCE under the fence. `Δt = age / passes`, so a
    /// finer schedule steps a smaller `Δt` and the answer converges to the same landscape (03 §4.6:
    /// the passes are a resolution, not a dial).
    #[must_use]
    pub fn gain(&self) -> u64 {
        let dt = Gf::from_i64(self.age_yr as i64) / Gf::from_i64(i64::from(self.passes));
        let k = Gf::from_f64(K0_PER_YR) / Gf::from_f64(MM_PER_M).sqrt();
        (dt * k * Gf::from_i64(1 << GAIN_BITS)).to_i64_floor() as u64
    }
}

/// What one routing pass found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteReport {
    /// Whether the sea seeded the flood (false: the lowest nodes did).
    pub sea_seeded: bool,
    /// The outlets the flood started from.
    pub outlets: usize,
    /// Nodes the flood RAISED — the lake nodes.
    pub raised: usize,
    /// Nodes routed by the flat's distance field rather than by a lower neighbour.
    pub flat: usize,
    /// Nodes the distance field never reached: a defect (gate G-DRAINAGE).
    pub undrained: usize,
    /// Nodes left out of the topological order: a cycle, a defect.
    pub cyclic: usize,
}

/// What one stream-power sweep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepReport {
    /// Nodes cut by at least one step.
    pub lowered: usize,
    /// Lake nodes skipped, so the hole stays a lake.
    pub skipped_lake: usize,
    /// The deepest single cut, in steps.
    pub max_cut: i32,
    /// Every cut summed, in steps.
    pub total_cut: u64,
}

/// The whole solve's record, pass by pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SolveReport {
    pub routes: Vec<RouteReport>,
    pub sweeps: Vec<SweepReport>,
}

/// The solve's state over one body's macro lattice. Every row is one word a node; the words are
/// public so an instrument can read them, and only the passes below write them.
#[derive(Clone, Debug, PartialEq)]
pub struct MacroSolve {
    pub lattice: MacroLattice,
    /// The sea's level in sixteenths from the ladder radius: a node at or under it is an outlet.
    pub sea_z: i32,
    /// The terrain, sixteenths from the ladder radius.
    pub z: Vec<i32>,
    /// The routing surface: the terrain with every pit raised to its spill level.
    pub z_flood: Vec<i32>,
    /// The D8 receiver; [`NO_NODE`] for an outlet.
    pub receiver: Vec<u32>,
    /// The chord to the receiver in whole metres; zero for an outlet.
    pub chord: Vec<u32>,
    /// The node's true area in whole square metres.
    pub area: Vec<u64>,
    /// The rain at the node, mm/yr, in `[P_MIN_MM_YR, P_MAX_MM_YR]`.
    pub rain: Vec<u32>,
    /// The discharge: the rain accumulated downstream, mm·m²/yr.
    pub discharge: Vec<u64>,
    /// The topological order of the receiver tree, LEAVES FIRST, outlets last.
    pub order: Vec<u32>,
    /// The node directions at [`CACHE_BITS`], read by every chord.
    dir: Vec<[i32; 3]>,
}

/// The sea's level of `body` in sixteenths from the ladder radius.
#[must_use]
pub fn sea_level(body: &BodyDefinition) -> i32 {
    ((body.sea_radius - body.radius).raw() >> Z_SHIFT) as i32
}

/// The rung whose cell is about a node: `log₂` of the node's size in rung-0 cells, at most the
/// body's top rung. The starting surface is the recipe's relief AT THAT RUNG — the octaves the
/// survival rule keeps there, which is exactly the relief a node can resolve.
#[must_use]
pub fn node_rung(body: &BodyDefinition, lattice: &MacroLattice) -> u8 {
    let log2 = (31 - lattice.cells_per_node.leading_zeros()) as u8;
    log2.min(body.ladder().rungs - 1)
}

/// ★ THE STARTING SURFACE: the recipe's own relief at every node's direction, at the node's rung,
/// in sixteenths from the ladder radius (C2 replaces it with the initial land).
#[must_use]
pub fn initial_surface(body: &BodyDefinition, lattice: &MacroLattice) -> Vec<i32> {
    let rung = node_rung(body, lattice);
    (0..lattice.node_count() as u32)
        .map(|node| {
            ((height(body, lattice.direction(node), rung) - body.radius).raw() >> Z_SHIFT) as i32
        })
        .collect()
}

/// Whether neighbour `m` (drop `dz_m` over chord `l_m`) is a STEEPER descent than the best so far
/// (`dz_best` over `l_best`, node `best`): cross-multiplied integers, and on an exact tie the
/// smaller global index. A total order; no float ratio is ever compared.
#[must_use]
pub fn steeper(dz_m: i64, l_m: u32, m: u32, dz_best: i64, l_best: u32, best: u32) -> bool {
    let lhs = dz_m * i64::from(l_best);
    let rhs = dz_best * i64::from(l_m);
    lhs > rhs || (lhs == rhs && m < best)
}

impl MacroSolve {
    /// The solve's state for `body`, at its starting surface; `None` where the body has no
    /// lattice.
    #[must_use]
    pub fn new(body: &BodyDefinition) -> Option<MacroSolve> {
        let lattice = MacroLattice::of(body)?;
        let n = lattice.node_count();
        let z = initial_surface(body, &lattice);
        let dir: Vec<[i32; 3]> = (0..n as u32)
            .map(|node| {
                let d = lattice.direction(node);
                [cache(d[0]), cache(d[1]), cache(d[2])]
            })
            .collect();
        let area = (0..n as u32).map(|node| lattice.area_m2(node)).collect();
        Some(MacroSolve {
            lattice,
            sea_z: sea_level(body),
            z_flood: z.clone(),
            z,
            receiver: vec![NO_NODE; n],
            chord: vec![0; n],
            area,
            rain: vec![P_MIN_MM_YR; n],
            discharge: vec![0; n],
            order: Vec::with_capacity(n),
            dir,
        })
    }

    /// ★ THE SOLVE'S STATE FROM THE INITIAL LAND (stage C2): the land's heights and its sea level
    /// as the start, in place of the recipe's noise. A body with no sea gets a level under every
    /// node, so the flood seeds from its lowest nodes.
    #[must_use]
    pub fn from_land(body: &BodyDefinition, land: &crate::land::Land) -> Option<MacroSolve> {
        let mut state = MacroSolve::new(body)?;
        state.z.clone_from(&land.z);
        state.z_flood.clone_from(&land.z);
        state.sea_z = land.sea_z.unwrap_or(i32::MIN);
        Some(state)
    }

    /// The nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.z.len()
    }

    /// The bytes the state holds per node, summed over its rows — what the design's memory model
    /// is measured against.
    #[must_use]
    pub fn bytes_per_node(&self) -> usize {
        4 + 4 + 4 + 4 + 8 + 4 + 8 + 4 + 12
    }

    /// The water level at a node: a lake's spill level where the flood raised it, the sea's level
    /// under the sea, [`DRY`] elsewhere — one expression, no branch on a landform kind.
    #[must_use]
    pub fn water_level(&self, node: u32) -> i32 {
        let i = node as usize;
        if self.z_flood[i] > self.z[i] {
            self.z_flood[i]
        } else if self.z[i] <= self.sea_z {
            self.sea_z
        } else {
            DRY
        }
    }

    /// The chord between two nodes from the cached directions, whole metres.
    fn chord_m(&self, a: u32, b: u32) -> u32 {
        chord_between(
            self.lattice.radius_m(),
            uncache(self.dir[a as usize]),
            uncache(self.dir[b as usize]),
        )
    }

    /// ★ THE ROUTING: the priority flood, the D8 receivers on the flooded surface, the flats by a
    /// distance field, and the topological order of the receiver tree.
    pub fn route(&mut self) -> RouteReport {
        let n = self.node_count();
        let mut report = RouteReport::default();
        // 1. THE PRIORITY FLOOD. The outlets seed a heap keyed by `(height, index)`; popping the
        //    lowest first, every unvisited neighbour is raised to at least the popped level.
        let mut visited = vec![false; n];
        let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::new();
        let mut dist = vec![u32::MAX; n];
        for i in 0..n {
            self.receiver[i] = NO_NODE;
            self.chord[i] = 0;
            if self.z[i] <= self.sea_z {
                visited[i] = true;
                self.z_flood[i] = self.z[i];
                dist[i] = 0;
                heap.push(Reverse((self.z[i], i as u32)));
            }
        }
        report.sea_seeded = !heap.is_empty();
        if heap.is_empty() {
            let mut lowest: BinaryHeap<(i32, u32)> = BinaryHeap::new();
            for i in 0..n {
                lowest.push((self.z[i], i as u32));
                if lowest.len() > OUTLETS_WITHOUT_SEA {
                    lowest.pop();
                }
            }
            for (z, node) in lowest {
                let i = node as usize;
                visited[i] = true;
                self.z_flood[i] = z;
                dist[i] = 0;
                heap.push(Reverse((z, node)));
            }
        }
        report.outlets = heap.len();
        while let Some(Reverse((level, node))) = heap.pop() {
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE || visited[m as usize] {
                    continue;
                }
                visited[m as usize] = true;
                let raised = self.z[m as usize].max(level);
                self.z_flood[m as usize] = raised;
                heap.push(Reverse((raised, m)));
            }
        }
        drop(heap);
        drop(visited);
        report.raised = (0..n).filter(|&i| self.z_flood[i] > self.z[i]).count();
        // 2. THE RECEIVERS: the steepest lower neighbour on the flooded surface.
        for node in 0..n as u32 {
            let i = node as usize;
            if dist[i] == 0 {
                continue;
            }
            let mut best = NO_NODE;
            let (mut dz_best, mut l_best) = (0i64, 0u32);
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE || self.z_flood[m as usize] >= self.z_flood[i] {
                    continue;
                }
                let dz = i64::from(self.z_flood[i]) - i64::from(self.z_flood[m as usize]);
                let l = self.chord_m(node, m);
                if best == NO_NODE || steeper(dz, l, m, dz_best, l_best, best) {
                    best = m;
                    dz_best = dz;
                    l_best = l;
                }
            }
            if best != NO_NODE {
                self.receiver[i] = best;
                self.chord[i] = l_best;
                dist[i] = 0;
            }
        }
        // 3. THE FLATS: a breadth-first distance to the nearest routed node over equal flooded
        //    heights, level by level (a distance depends on no visit order), then the receiver of a
        //    flat node is its neighbour with the smaller distance, ties to the smaller index.
        let mut frontier: Vec<u32> = (0..n as u32).filter(|&i| dist[i as usize] == 0).collect();
        let mut level = 0u32;
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &node in &frontier {
                for m in self.lattice.neighbours(node) {
                    if m == NO_NODE
                        || dist[m as usize] != u32::MAX
                        || self.z_flood[m as usize] != self.z_flood[node as usize]
                    {
                        continue;
                    }
                    dist[m as usize] = level + 1;
                    next.push(m);
                }
            }
            frontier = next;
            level += 1;
        }
        let (flat, undrained) = self.assign_flat_receivers(&dist);
        report.flat = flat;
        report.undrained = undrained;
        drop(dist);
        // 4. THE ORDER: leaves first. A node is emitted once every donor is; the receiver whose
        //    last donor was emitted is pushed. A stack, so the order is a stated one.
        let mut donors = vec![0u8; n];
        for i in 0..n {
            let r = self.receiver[i];
            if r != NO_NODE {
                donors[r as usize] += 1;
            }
        }
        let mut stack: Vec<u32> = (0..n as u32).filter(|&i| donors[i as usize] == 0).collect();
        self.order.clear();
        while let Some(node) = stack.pop() {
            self.order.push(node);
            let r = self.receiver[node as usize];
            if r != NO_NODE {
                donors[r as usize] -= 1;
                if donors[r as usize] == 0 {
                    stack.push(r);
                }
            }
        }
        report.cyclic = n - self.order.len();
        report
    }

    /// ★ THE FLAT'S RECEIVERS from the distance field: a node the distance field reached (a distance
    /// over zero) takes its neighbour of equal flooded height with the SMALLER distance, ties to the
    /// smaller index; a node it never reached is counted UNDRAINED (a defect, gate G-DRAINAGE) and
    /// keeps no receiver. Returns `(flat, undrained)`.
    fn assign_flat_receivers(&mut self, dist: &[u32]) -> (usize, usize) {
        let (mut flat, mut undrained) = (0usize, 0usize);
        for i in 0..self.node_count() {
            if dist[i] == 0 {
                continue;
            }
            if dist[i] == u32::MAX {
                undrained += 1;
                continue;
            }
            flat += 1;
            let node = i as u32;
            let mut best = NO_NODE;
            for m in self.lattice.neighbours(node) {
                if m == NO_NODE
                    || dist[m as usize] == u32::MAX
                    || self.z_flood[m as usize] != self.z_flood[i]
                {
                    continue;
                }
                if best == NO_NODE
                    || dist[m as usize] < dist[best as usize]
                    || (dist[m as usize] == dist[best as usize] && m < best)
                {
                    best = m;
                }
            }
            self.receiver[i] = best;
            self.chord[i] = self.chord_m(node, best);
        }
        (flat, undrained)
    }

    /// ★ THE DISCHARGE: the rain on every node's own area, accumulated down the receiver tree in
    /// the order, leaves first. Exact integer sums, whatever the order.
    pub fn accumulate(&mut self) {
        for i in 0..self.node_count() {
            self.discharge[i] = u64::from(self.rain[i]) * self.area[i];
        }
        for k in 0..self.order.len() {
            let node = self.order[k] as usize;
            let r = self.receiver[node];
            if r != NO_NODE {
                self.discharge[r as usize] += self.discharge[node];
            }
        }
    }

    /// ★ THE STREAM POWER SWEEP, implicit (Braun & Willett 2013), outlets first:
    /// `z' = (z + c·b) / (1 + c)` with `c = Δt·K·√Q / L` and `b` the receiver's BASE LEVEL —
    /// its water level where it stands under water, its terrain height otherwise, one expression.
    /// An outlet is left alone; a lake node is skipped, so the hole the flood found stays a lake.
    pub fn sweep(&mut self, gain: u64) -> SweepReport {
        let mut report = SweepReport::default();
        let one = 1i128 << GAIN_BITS;
        for k in (0..self.order.len()).rev() {
            let node = self.order[k];
            let i = node as usize;
            let r = self.receiver[i];
            if r == NO_NODE {
                continue;
            }
            if self.z_flood[i] > self.z[i] {
                report.skipped_lake += 1;
                continue;
            }
            let base = self.z[r as usize].max(self.water_level(r));
            let c = (u128::from(gain) * u128::from(isqrt(self.discharge[i])))
                / u128::from(self.chord[i].max(1));
            let c = c as i128;
            let numerator = (i128::from(self.z[i]) << GAIN_BITS) + c * i128::from(base);
            let cut = i128::from(self.z[i]) - numerator.div_euclid(one + c);
            let cut = cut as i32;
            if cut > 0 {
                report.lowered += 1;
                report.max_cut = report.max_cut.max(cut);
                report.total_cut += cut as u64;
                self.z[i] -= cut;
            }
        }
        report
    }

    /// The lowest and the highest terrain, in sixteenths from the ladder radius.
    #[must_use]
    pub fn range(&self) -> (i32, i32) {
        self.z
            .iter()
            .fold((i32::MAX, i32::MIN), |(lo, hi), &z| (lo.min(z), hi.max(z)))
    }
}

/// A direction word at the bend's bits, kept at [`CACHE_BITS`].
fn cache(g: Gi) -> i32 {
    (g.raw() >> (DIR_BITS - CACHE_BITS)) as i32
}

/// A cached direction back at the bend's bits (the dropped bits are zero: a stated rounding).
fn uncache(d: [i32; 3]) -> [Gi; 3] {
    [
        Gi::new(i64::from(d[0]) << (DIR_BITS - CACHE_BITS)),
        Gi::new(i64::from(d[1]) << (DIR_BITS - CACHE_BITS)),
        Gi::new(i64::from(d[2]) << (DIR_BITS - CACHE_BITS)),
    ]
}

/// ★ THE DRIVER: the schedule over `body` from the recipe's own relief (the C1 start) — a routing
/// and an accumulation every `flood_every` passes, a sweep every pass. `None` where the body has
/// no lattice.
#[must_use]
pub fn solve(body: &BodyDefinition, schedule: Schedule) -> Option<(MacroSolve, SolveReport)> {
    run(MacroSolve::new(body)?, schedule)
}

/// ★ THE DRIVER FROM THE INITIAL LAND (stage C2): the same schedule from the land the plates,
/// the isostasy and the belts made under the charter's words.
#[must_use]
pub fn solve_land(
    body: &BodyDefinition,
    words: &crate::land::LandWords,
    schedule: Schedule,
) -> Option<(MacroSolve, SolveReport)> {
    let lattice = MacroLattice::of(body)?;
    let land = crate::land::initial_land(body, &lattice, words);
    run(MacroSolve::from_land(body, &land)?, schedule)
}

/// The schedule over a state.
fn run(mut state: MacroSolve, schedule: Schedule) -> Option<(MacroSolve, SolveReport)> {
    let gain = schedule.gain();
    let mut report = SolveReport::default();
    let mut since_route = schedule.flood_every;
    for _ in 0..schedule.passes {
        if since_route >= schedule.flood_every {
            report.routes.push(state.route());
            state.accumulate();
            since_route = 0;
        }
        since_route += 1;
        report.sweeps.push(state.sweep(gain));
    }
    Some((state, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_planet};
    use crate::macro_lattice::STENCIL;
    use vd_seed::bend::Face;

    /// The gain: five billion years over forty passes, times the Earth-calibrated erodibility over
    /// the root of a thousand, at twenty-four fraction bits — about 132 million.
    #[test]
    fn the_gain_is_the_ages_step_times_the_erodibility() {
        let gain = Schedule::standard(HOME_SYSTEM_AGE_YR).gain();
        let expect = 5.0e9 / 40.0 * 2.0e-6 / 1000f64.sqrt() * f64::from(1u32 << GAIN_BITS);
        assert_eq!(gain, expect.floor() as u64);
        assert!((132_000_000..133_000_000).contains(&gain), "{gain}");
    }

    /// The steeper test is a total order over integers: a steeper drop wins, an equal slope with a
    /// longer chord loses, and an exact tie takes the smaller index.
    #[test]
    fn steeper_is_a_total_order_with_the_smaller_index_on_a_tie() {
        assert!(steeper(20, 100, 7, 10, 100, 3));
        assert!(!steeper(10, 200, 7, 10, 100, 3));
        assert!(steeper(20, 200, 2, 10, 100, 3));
        assert!(!steeper(20, 200, 4, 10, 100, 3));
    }

    /// The node's rung on the home planet is 13 (a node is 2¹³ cells) and on the moon 13 too, and
    /// the starting surface holds the recipe's relief at that rung within the body's own band.
    #[test]
    fn the_starting_surface_is_the_recipes_relief_at_the_nodes_rung() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        assert_eq!(node_rung(&moon, &lattice), 13.min(moon.ladder().rungs - 1));
        let z = initial_surface(&moon, &lattice);
        assert_eq!(z.len(), lattice.node_count());
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(z.iter().all(|&s| s.abs() <= band), "outside the band");
        assert!(z.iter().any(|&s| s != 0), "a flat moon");
        let home = home_planet();
        assert_eq!(
            node_rung(&home, &MacroLattice::of(&home).expect("a lattice")),
            13
        );
    }

    /// ★ THE DRIVER ON A REAL SMALL BODY OF THE WORLD (06 §3.3): the home moon's whole schedule —
    /// four routings, forty sweeps; every routing drains every node to an outlet (gate
    /// G-DRAINAGE) with no cycle; the sweeps only lower; the relief stays inside the band.
    #[test]
    fn the_home_moon_solves_and_every_node_drains() {
        let moon = home_moon();
        let before = MacroSolve::new(&moon).expect("a state");
        let (state, report) =
            solve(&moon, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        assert_eq!(report.sweeps.len(), 40);
        assert_eq!(report.routes.len(), 4);
        for r in &report.routes {
            assert_eq!((r.undrained, r.cyclic), (0, 0), "{r:?}");
            assert!(r.outlets > 0);
        }
        assert!(report.sweeps[0].lowered > 0);
        for (i, (&a, &b)) in before.z.iter().zip(state.z.iter()).enumerate() {
            assert!(b <= a, "node {i} rose from {a} to {b}");
        }
        let (lo, hi) = state.range();
        let (lo0, hi0) = before.range();
        assert!(lo >= lo0, "the floor fell: {lo0} to {lo}");
        assert!(hi <= hi0, "the ceiling rose: {hi0} to {hi}");
        assert_eq!(state.bytes_per_node(), 52);
        assert_eq!(state.node_count(), 27_744);
        // The discharge is conserved: what leaves at the outlets is every node's own rain.
        let total: u64 = (0..state.node_count())
            .map(|i| u64::from(state.rain[i]) * state.area[i])
            .sum();
        let out: u64 = (0..state.node_count())
            .filter(|&i| state.receiver[i] == NO_NODE)
            .map(|i| state.discharge[i])
            .sum();
        assert_eq!(out, total);
    }

    /// ★ THE DRIVER FROM THE LAND on the moon: a stagnant lid with no sea, so the flood seeds from
    /// the six lowest nodes, every node drains, and the sweeps only lower.
    #[test]
    fn the_moon_solves_from_its_initial_land() {
        let moon = home_moon();
        let (state, report) = solve_land(
            &moon,
            &crate::home::home_moon_land_words(),
            Schedule::standard(HOME_SYSTEM_AGE_YR),
        )
        .expect("a solve");
        assert_eq!(state.sea_z, i32::MIN);
        assert_eq!(report.routes.len(), 4);
        for r in &report.routes {
            assert!(!r.sea_seeded);
            assert_eq!(
                (r.outlets, r.undrained, r.cyclic),
                (OUTLETS_WITHOUT_SEA, 0, 0)
            );
        }
        let band = (moon.relief_bound_m(0) * f64::from(Z_STEPS_PER_M)) as i32;
        assert!(state.z.iter().all(|&z| z.abs() <= band));
    }

    /// The flood's two seedings on the moon: with the sea under every node the six lowest nodes
    /// are the outlets; with the sea over every node every node is an outlet and nothing routes.
    #[test]
    fn the_flood_seeds_from_the_sea_or_from_the_lowest_nodes() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        state.sea_z = i32::MIN + 1;
        let r = state.route();
        assert!(!r.sea_seeded);
        assert_eq!(r.outlets, OUTLETS_WITHOUT_SEA);
        assert_eq!((r.undrained, r.cyclic), (0, 0));
        let lowest = state.range().0;
        let outlets: Vec<usize> = (0..state.node_count())
            .filter(|&i| state.receiver[i] == NO_NODE)
            .collect();
        assert_eq!(outlets.len(), OUTLETS_WITHOUT_SEA);
        assert!(outlets.iter().any(|&i| state.z[i] == lowest));
        for &i in &outlets {
            assert_eq!(state.water_level(i as u32), DRY);
        }
        state.sea_z = i32::MAX;
        let r = state.route();
        assert!(r.sea_seeded);
        assert_eq!(r.outlets, state.node_count());
        assert_eq!((r.raised, r.flat, r.undrained, r.cyclic), (0, 0, 0, 0));
        assert_eq!(state.order.len(), state.node_count());
        assert_eq!(state.sweep(1 << 30), SweepReport::default());
    }

    /// ★ THE KERNELS ON A STATED NEIGHBOURHOOD (06 §3.3): a plateau one metre over the sea with
    /// one outlet and one pit. The flood raises the pit to the plateau (a lake with the plateau's
    /// spill level), the plateau is a flat that drains to the outlet by distance, the lake node is
    /// skipped by the sweep, its neighbours' base level is the lake's level, and the nodes at the
    /// outlet are cut.
    #[test]
    fn a_pit_becomes_a_lake_and_a_plateau_drains_by_distance() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        let plateau = Z_STEPS_PER_M;
        state.z = vec![plateau; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 3, 3);
        state.z[outlet as usize] = 0;
        let mid = state.lattice.edge as i32 / 2;
        let pit = state.lattice.index(Face::NegZ, mid, mid);
        state.z[pit as usize] = plateau / 2;
        let r = state.route();
        assert!(r.sea_seeded);
        assert_eq!(r.outlets, 1);
        assert_eq!(r.raised, 1);
        assert_eq!((r.undrained, r.cyclic), (0, 0));
        // Eight nodes touch the outlet and take it by slope; every other node is flat.
        assert_eq!(r.flat, n - 1 - 8);
        assert_eq!(state.water_level(pit), plateau);
        assert_eq!(state.water_level(outlet), 0);
        let dry = state.lattice.index(Face::PosY, 1, 1);
        assert_eq!(state.water_level(dry), DRY);
        for m in state.lattice.neighbours(outlet) {
            assert_eq!(state.receiver[m as usize], outlet);
            assert!(state.chord[m as usize] > 0);
        }
        state.accumulate();
        let total: u64 = state.area.iter().sum::<u64>() * u64::from(P_MIN_MM_YR);
        assert_eq!(state.discharge[outlet as usize], total);
        let s = state.sweep(Schedule::standard(HOME_SYSTEM_AGE_YR).gain());
        assert_eq!(s.skipped_lake, 1);
        assert!(s.lowered >= 8, "{s:?}");
        assert!(s.max_cut > 0);
        assert_eq!(
            state.z[pit as usize],
            plateau / 2,
            "the lake keeps its hole"
        );
        // A node whose receiver is the lake reads the lake's LEVEL as its base, so it is not cut
        // below the plateau toward the hole.
        let donors: Vec<usize> = (0..n).filter(|&i| state.receiver[i] == pit).collect();
        assert!(!donors.is_empty());
        for i in donors {
            assert_eq!(state.z[i], plateau);
        }
        // The stencil order is the fixed one, and a routed node's receiver is a stencil neighbour.
        assert_eq!(STENCIL[0], (-1, -1));
        let some = state.lattice.index(Face::PosX, 3, 4);
        assert!(
            state
                .lattice
                .neighbours(some)
                .contains(&state.receiver[some as usize])
        );
    }

    /// ★ THE UNDRAINED COUNTER on a stated distance field: a node the field never reached is counted
    /// and keeps no receiver; a reached node skips an unreached neighbour and takes the reached one.
    #[test]
    fn an_unreached_node_is_counted_undrained_and_skipped_as_a_neighbour() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z_flood = vec![0; n];
        let mut dist = vec![u32::MAX; n];
        let a = state.lattice.index(Face::PosX, 20, 20);
        let ring = state.lattice.neighbours(a);
        // One routed neighbour at distance 0, the rest unreached; `a` itself at distance 1.
        dist[ring[7] as usize] = 0;
        dist[a as usize] = 1;
        let (flat, undrained) = state.assign_flat_receivers(&dist);
        assert_eq!(flat, 1);
        assert_eq!(undrained, n - 2);
        assert_eq!(state.receiver[a as usize], ring[7]);
        assert_eq!(state.receiver[ring[0] as usize], NO_NODE);
    }

    /// The tie in the flat's receiver choice: two neighbours at one distance keep the smaller index.
    /// On the plateau every routed neighbour of the outlet is at distance zero, so a flat node
    /// adjacent to two of them takes the smaller.
    #[test]
    fn a_flat_node_between_two_routed_nodes_takes_the_smaller_index() {
        let moon = home_moon();
        let mut state = MacroSolve::new(&moon).expect("a state");
        let n = state.node_count();
        state.z = vec![Z_STEPS_PER_M; n];
        state.sea_z = 0;
        let outlet = state.lattice.index(Face::PosX, 10, 10);
        state.z[outlet as usize] = 0;
        state.route();
        // (12, 10) touches (11, 9), (11, 10) and (11, 11), all routed to the outlet at distance 0.
        let node = state.lattice.index(Face::PosX, 12, 10);
        let candidates = [
            state.lattice.index(Face::PosX, 11, 9),
            state.lattice.index(Face::PosX, 11, 10),
            state.lattice.index(Face::PosX, 11, 11),
        ];
        assert_eq!(
            state.receiver[node as usize],
            *candidates.iter().min().expect("three")
        );
    }
}
