//! ★ R1 — THE BENCH (the owner's go, 2026-09-22;
//! `docs/investigation/2026-09-22/fine_terrain_models_and_crates.md` §5 R1). Ruling F7's own order
//! is *"the bench, the discussion, the build"*, and §2.6's two anchors for the fine ground differ by
//! **370×** — 0.99 ns a cell-iteration on a card, 367 ns a node-pass in our macro sweep. No plan
//! should be written across that gap, so this example measures the gap shut.
//!
//! It runs the FOUR OPERATORS of Schott, Galin, Guérin, Peytavie & Paris 2024 (`vd_recipe::amplify`,
//! which states what we took from the reference and where we depart from it) over ONE macro tile of
//! THE home planet — 64 × 64 macro nodes, 524 km a side — upsampled to a working resolution and
//! seeded with the artifact row's own discharge, and prints:
//!
//! 1. **NANOSECONDS PER CELL PER ITERATION** on ONE core, at 1 024 m cells and at 64 m cells.
//! 2. **THE HALO WIDTH** at which a patch's interior stops changing — the halo that recommendation
//!    R3's derivation on both hosts must ship.
//! 3. **THE ITERATION COUNT** at which the routed drainage stops moving, seeded from the macro
//!    discharge against seeded from zero.
//! 4. **G-HACK** — the Hack's-law fit `L = c·a^n` on the drawn river network, before and after the
//!    loop, against the published bands `c ∈ [1, 6]`, `n ∈ [0.45, 0.7]` (Sassolas-Serrayet et al.
//!    2018; Schott et al. 2023 §7.3 validate exactly this way).
//! 5. **G-BREACH** — a breaching volume, before and after, on the definition this file states.
//! 6. **THE DIFF AGAINST THE REFERENCE** — a float transcription of the four GLSL shaders'
//!    arithmetic, run beside our integer operators on one stated field.
//! 7. **THE WHOLE PLANET's cost**, extrapolated from (1), so R2 has a number.
//!
//! ★ **WHAT THIS BENCH DOES NOT DO.** It never touches the solve, the recipe's kernels, the
//! artifact, the wire or the store. Nothing it computes is shipped. The amplified field is thrown
//! away when the process exits.
//!
//! ★ **WHERE THE INPUT COMES FROM.** The home planet is SOLVED ONCE at start-up, through
//! `vd_bins::artifact_worker::run_solve`, exactly as `lake_census` does. We do NOT read the
//! dev-cluster's store: a store holds whatever `ARTIFACT_VERSION` it was written at, and a bench
//! whose input depends on when somebody last flew is not a measurement.
//!
//! Run in release, ONE job at a time on the machine:
//!
//! ```text
//! cargo run --release -p vd-bins --example erosion_bench
//! cargo run --release -p vd-bins --example erosion_bench -- --fine-span 4096 --iters 300
//! ```

use std::collections::BinaryHeap;
use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_recipe::Gi;
use vd_recipe::amplify::{
    A_BITS, D_BITS, H_BITS, K_BITS, NEIGHBOURS, NO_SLOT, Routing, S_BITS, cut, deposit, gather,
    incise, route, stream_power, thermal,
};
use vd_recipe::root::recip_pow2;
use vd_seed::bend::Face;
use vd_terrain::artifact::{
    Artifact, RECEIVER_NONE, RECEIVER_SLOT_MASK, TILE_EDGE, node_at, tiles_within,
};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;
use vd_terrain::strata::Province;

/// The stencil's eight offsets, in `MacroLattice::STENCIL` order — the same order the operators
/// index their slots by, so a slot means one thing everywhere.
const OFF: [(i32, i32); NEIGHBOURS] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// ★ THE STATED EROSION PARAMETERS. The three the reference ships verbatim (`tanThresholdAngle`,
/// `deposition_strength`, the two `0.1` shares) are its own numbers. The two RATES — the
/// erodibility and the thermal rate — cannot be taken verbatim, because the reference's heights
/// live in a normalised domain and ours are metres over the ladder radius; they are STATED CHOICES
/// here, chosen once so the loop removes a few metres of a kilometre of relief over 300 iterations,
/// and THE OWNER'S NUMBER IS OWED. **The timing in measurement 1 does not depend on either**: the
/// operators do the same multiplies whatever the rate says.
struct Params {
    /// `k`, the erodibility, at `K_BITS`.
    k: Gi,
    /// `k_γ`, the thermal rate, at `K_BITS`.
    thermal_rate: Gi,
    /// `tan γ₀`, the talus, at `S_BITS` — the shader's `tanThresholdAngle = 0.57`.
    talus: Gi,
    /// `deposition_strength = 1.0`, at `K_BITS`.
    deposition_strength: Gi,
    /// The shader's two `0.1`, at `K_BITS`.
    share: Gi,
    /// The drainage clamp, at `A_BITS`, in cells.
    a_max: Gi,
    /// The slope clamp, at `S_BITS`.
    s_max: Gi,
    /// The reference's outer clamp `max_spe = 10 000`, at `A_BITS`.
    spe_max: Gi,
}

impl Params {
    /// The reference's own three, and two rates that are still placeholders until [`calibrate`]
    /// replaces them against the tile itself.
    fn stated() -> Params {
        Params {
            k: Gi::new(838),
            thermal_rate: Gi::new(33_554),
            // tan 30° — the shader's `tanThresholdAngle = 0.57`.
            talus: Gi::new(37_356),
            deposition_strength: Gi::new(1 << K_BITS),
            // floor(2²⁴/10), the shader's two 0.1.
            share: Gi::new(1_677_721),
            a_max: Gi::new(1_000_000 << A_BITS),
            s_max: Gi::new(1 << S_BITS),
            spe_max: Gi::new(10_000 << A_BITS),
        }
    }
}

/// The share of the tile's relief the calibration asks the loop to remove from a river cell.
const TARGET_SHARE: f64 = 0.05;
/// Which cell the target speaks about: the 99th percentile of the initial stream power — a river,
/// not a hillside.
const TARGET_PERCENTILE: f64 = 0.99;
/// Where the reference's outer clamp lands on THIS field: the 99.9th percentile.
const CLAMP_PERCENTILE: f64 = 0.999;

/// ★ THE CALIBRATION, AND WHY IT IS NOT A FUDGE (MEASURED, R1's first run).
///
/// The reference ships `k = 5 × 10⁻⁴` and `max_spe = 10 000`, and **those two numbers cannot be
/// carried over**: its heightfields live in a normalised domain where a slope is of order one, and
/// ours are METRES over the ladder radius where a macro-upsampled slope is of order a hundredth.
/// The stream power goes as the slope SQUARED, so the same `k` reads four orders of magnitude too
/// small here — measured, twice: with the reference's own `k` the loop moved the tile's ground by
/// *"0.000 m on average, 0.0 m at the worst"* over 300 iterations.
///
/// So the two rates are DERIVED from a stated target on this very tile, the way ruling T9 asks
/// (*"every physical fact is COMPUTED by a published law with a stated calibration body"*), and
/// both are printed:
///
/// 1. `k` is set so that the cell at the [`TARGET_PERCENTILE`] of the initial stream power — a
///    river, not a hillside — removes [`TARGET_SHARE`] of the TILE'S OWN RELIEF over the run.
/// 2. `max_spe`, the reference's outer clamp, is the [`CLAMP_PERCENTILE`] of that same initial
///    stream power: the same clamp the paper wants, expressed as a percentile of this field
///    instead of as a number from another unit system.
/// 3. `k_γ`, the thermal rate, is set so a slope one talus-width over the talus moves the same
///    height per iteration as `k` gives the river. One rate, one target, no second knob.
///
/// ★ **THE OWNER'S NUMBER IS STILL OWED.** The believable calibration is Hack's law on Earth,
/// which is recommendation R2's own gate; this one only makes R1's before/after comparison mean
/// something. It is a stated rule, not a number somebody liked.
fn calibrate(g: &mut Grid, p: &mut Params, iters: usize) {
    g.pass_route();
    let mut spe: Vec<i64> = Vec::with_capacity(g.cells());
    // One gather is enough to give every cell a drainage of the right order for the calibration.
    g.pass_drainage();
    for i in 0..g.cells() {
        let r = g.routing[i];
        spe.push(stream_power(g.a[i], r.steepest, p.a_max, p.s_max, p.spe_max).raw());
    }
    spe.sort_unstable();
    let at = |q: f64| -> i64 { spe[((g.cells() - 1) as f64 * q) as usize].max(1) };
    let (lo, hi) = (
        g.h.iter().map(|v| v.raw()).min().unwrap_or(0),
        g.h.iter().map(|v| v.raw()).max().unwrap_or(0),
    );
    let relief = (hi - lo).max(1) as f64;
    let want_per_iteration = relief * TARGET_SHARE / iters as f64;
    // cut = spe · k >> (K_BITS + A_BITS − H_BITS), so k = cut · 2^shift / spe.
    let shift = f64::from(1u64.wrapping_shl(K_BITS + A_BITS - H_BITS) as u32);
    let k = (want_per_iteration * shift / at(TARGET_PERCENTILE) as f64).max(1.0);
    p.k = Gi::new(k as i64);
    p.spe_max = Gi::new(at(CLAMP_PERCENTILE));
    // The thermal rate: one talus-width of excess slope moves `want_per_iteration` of height.
    let rate = want_per_iteration * f64::from(1u32 << K_BITS) / f64::from(p.talus.raw() as u32);
    p.thermal_rate = Gi::new(rate.max(1.0) as i64);
    println!(
        "  the calibration, on this tile: relief {:.0} m; the {:.0}th-percentile stream power {} (raw),\n    the {:.1}th {} — so k = {} at 2^{K_BITS} and max_spe = {} (raw), k_γ = {}.\n    The target: that river cell removes {:.0} % of the relief over {iters} iterations, which is {:.3} m an iteration.",
        relief / f64::from(1u32 << H_BITS),
        TARGET_PERCENTILE * 100.0,
        at(TARGET_PERCENTILE),
        CLAMP_PERCENTILE * 100.0,
        at(CLAMP_PERCENTILE),
        p.k.raw(),
        p.spe_max.raw(),
        p.thermal_rate.raw(),
        TARGET_SHARE * 100.0,
        want_per_iteration / f64::from(1u32 << H_BITS),
    );
}

/// The working grid: a square of `w` fine cells, holding one tile plus `halo` cells of the
/// neighbouring tiles' data on every side.
struct Grid {
    w: usize,
    halo: usize,
    cell_m: f64,
    /// The height at `H_BITS`.
    h: Vec<Gi>,
    h_next: Vec<Gi>,
    /// The routed drainage at `A_BITS`, in cells.
    a: Vec<Gi>,
    a_next: Vec<Gi>,
    /// What each cell catches itself, at `A_BITS` — one cell, always.
    a_seed: Vec<Gi>,
    /// The suspended sediment at `H_BITS`.
    sed: Vec<Gi>,
    sed_next: Vec<Gi>,
    /// The routing pre-pass's answers.
    routing: Vec<Routing>,
    /// The rock's own erodibility share, 1/256 (`Province::erodibility_q8`).
    hard: Vec<u8>,
    /// The two chord reciprocals at `D_BITS`: the axial cell and the diagonal.
    inv_d: [Gi; NEIGHBOURS],
}

impl Grid {
    fn cells(&self) -> usize {
        self.w * self.w
    }

    /// One cell's eight neighbour indices and the mask of the ones that exist.
    #[inline]
    fn stencil(&self, x: usize, y: usize) -> ([usize; NEIGHBOURS], u32) {
        let mut idx = [y * self.w + x; NEIGHBOURS];
        let mut present = 0u32;
        for (slot, (dx, dy)) in OFF.iter().enumerate() {
            let nx = x as i64 + i64::from(*dx);
            let ny = y as i64 + i64::from(*dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < self.w && (ny as usize) < self.w {
                idx[slot] = ny as usize * self.w + nx as usize;
                present |= 1 << slot;
            }
        }
        (idx, present)
    }

    /// ★ ONE ITERATION OF THE LOOP — the four operators, as five parallel stencil passes over the
    /// whole grid. Every pass reads the PREVIOUS buffer and writes its own, so no cell ever sees
    /// another cell's half-finished work and the answer does not depend on the order a host walks
    /// the tile in. That is the C1 determinism constraint, and it is why a grid model was chosen
    /// over a droplet model (§2.3).
    fn step(&mut self, p: &Params) {
        for pass in 0..5 {
            self.one_pass(p, pass);
        }
    }

    /// One of the five passes, by index — so the bench can time them apart.
    fn one_pass(&mut self, p: &Params, pass: usize) {
        match pass {
            0 => self.pass_route(),
            1 => self.pass_drainage(),
            2 => self.pass_incise(p),
            3 => self.pass_thermal(p),
            _ => self.pass_deposit(p),
        }
    }

    /// PASS 1 — the routing pre-pass: every cell's weight sum, its reciprocal, its steepest drop
    /// and the slot that drop points at.
    fn pass_route(&mut self) {
        for y in 0..self.w {
            for x in 0..self.w {
                let i = y * self.w + x;
                let (idx, present) = self.stencil(x, y);
                let mut nb = [Gi::ZERO; NEIGHBOURS];
                for (slot, j) in idx.iter().copied().enumerate() {
                    nb[slot] = self.h[j];
                }
                self.routing[i] = route(self.h[i], &nb, &self.inv_d, present);
            }
        }
    }

    /// PASS 2 — operator ①, the drainage: one parallel iteration of the multiple-flow gather.
    fn pass_drainage(&mut self) {
        for y in 0..self.w {
            for x in 0..self.w {
                let i = y * self.w + x;
                let (idx, present) = self.stencil(x, y);
                let mut nb = [Gi::ZERO; NEIGHBOURS];
                let mut vals = [Gi::ZERO; NEIGHBOURS];
                let mut inv = [Gi::ZERO; NEIGHBOURS];
                for (slot, j) in idx.iter().copied().enumerate() {
                    nb[slot] = self.h[j];
                    vals[slot] = self.a[j];
                    inv[slot] = self.routing[j].inv_sum;
                }
                self.a_next[i] = gather(
                    self.a_seed[i],
                    self.h[i],
                    &nb,
                    &vals,
                    &self.inv_d,
                    &inv,
                    present,
                );
            }
        }
        core::mem::swap(&mut self.a, &mut self.a_next);
    }

    /// PASS 3 — operator ②, the clamped stream power and the incision, floored at the receiver.
    fn pass_incise(&mut self, p: &Params) {
        for y in 0..self.w {
            for x in 0..self.w {
                let i = y * self.w + x;
                let (idx, _) = self.stencil(x, y);
                let r = self.routing[i];
                let spe = stream_power(self.a[i], r.steepest, p.a_max, p.s_max, p.spe_max);
                let c = cut(spe, p.k, Gi::new(i64::from(self.hard[i])));
                let floor = if r.steepest_slot == NO_SLOT {
                    self.h[i] - c
                } else {
                    self.h[idx[r.steepest_slot as usize]]
                };
                self.h_next[i] = incise(self.h[i], c, floor);
                // The sediment the incision put into the water, kept for pass 5.
                self.sed_next[i] = c;
            }
        }
        core::mem::swap(&mut self.h, &mut self.h_next);
    }

    /// PASS 4 — operator ③, thermal stabilisation.
    fn pass_thermal(&mut self, p: &Params) {
        for y in 0..self.w {
            for x in 0..self.w {
                let i = y * self.w + x;
                let (idx, present) = self.stencil(x, y);
                let mut nb = [Gi::ZERO; NEIGHBOURS];
                for (slot, j) in idx.iter().copied().enumerate() {
                    nb[slot] = self.h[j];
                }
                self.h_next[i] = thermal(
                    self.h[i],
                    &nb,
                    &self.inv_d,
                    present,
                    p.talus,
                    p.thermal_rate,
                );
            }
        }
        core::mem::swap(&mut self.h, &mut self.h_next);
    }

    /// PASS 5 — operator ④, deposition: the sediment rides the SAME routing weights, and settles
    /// where the transport capacity falls under what is carried.
    fn pass_deposit(&mut self, p: &Params) {
        for y in 0..self.w {
            for x in 0..self.w {
                let i = y * self.w + x;
                let (idx, present) = self.stencil(x, y);
                let mut nb = [Gi::ZERO; NEIGHBOURS];
                let mut vals = [Gi::ZERO; NEIGHBOURS];
                let mut inv = [Gi::ZERO; NEIGHBOURS];
                for (slot, j) in idx.iter().copied().enumerate() {
                    nb[slot] = self.h[j];
                    vals[slot] = self.sed[j];
                    inv[slot] = self.routing[j].inv_sum;
                }
                let carried = gather(Gi::ZERO, self.h[i], &nb, &vals, &self.inv_d, &inv, present);
                let (d, kept) = deposit(carried, self.sed_next[i], p.deposition_strength, p.share);
                self.h_next[i] = self.h[i] + d;
                self.sed_next[i] = kept;
            }
        }
        core::mem::swap(&mut self.h, &mut self.h_next);
        core::mem::swap(&mut self.sed, &mut self.sed_next);
    }
}

/// How the drainage field starts, for measurement 3.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Seeding {
    /// Nothing: every cell starts with no water and the accumulation builds it, as the reference does.
    Zero,
    /// Every fine cell starts at its macro node's own discharge — §2.5's sentence taken literally.
    Everywhere,
    /// Only the rim cells whose outside macro node drains inward carry that node's discharge, and
    /// they carry it in `a_seed` (what a cell CATCHES), which is what sets the gather's fixed point.
    Inflow,
}

/// The macro tile this bench runs on, and where it sits on the cube.
#[derive(Clone, Copy)]
struct TilePick {
    face: Face,
    /// The tile's first node on each axis.
    i0: i64,
    j0: i64,
    /// The tile's width in macro nodes.
    nodes: i64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str, fallback: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|k| args.get(k + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(fallback)
    };
    // The fine measurement's span in cells. 8 192 would be the WHOLE tile at 64 m — 67 million
    // cells and about 4 GB of buffers. The default is one QUARTER of the tile (4 096 × 4 096),
    // which the report says plainly.
    let fine_span = arg("--fine-span", 4_096);
    let fine_iters = arg("--fine-iters", 3);
    let iters = arg("--iters", 300);
    let halo_iters = arg("--halo-iters", 64);

    println!("erosion_bench — R1, the bench (docs/investigation/2026-09-22, §5 R1)");
    println!("  the machine: Apple M4 Pro, one core, release, one job at a time");

    // ── THE INPUT: the home planet, solved once. ────────────────────────────────────────────────
    let body = home_planet();
    let words = home_solve_words();
    let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
    let node_m = lattice.node_m();
    println!(
        "  the home planet: {} nodes a face edge, {} nodes, node {node_m:.0} m, radius {:.0} m",
        lattice.edge,
        lattice.node_count(),
        body.radius_m()
    );
    let t0 = Instant::now();
    let artifact = run_solve(&SolveJob { body, words }).expect("the solve answers");
    println!(
        "  the solve: {:.1} s, ARTIFACT_VERSION {}, sea {} m",
        t0.elapsed().as_secs_f64(),
        artifact.version,
        artifact.sea_m
    );

    // ── THE TILE: the one under the belt direction the owner flew, held off the face's seam. ────
    let belt = [0.617_270_f64, -0.437_286, -0.654_033];
    let len = (belt[0] * belt[0] + belt[1] * belt[1] + belt[2] * belt[2]).sqrt();
    let aim = [belt[0] / len, belt[1] / len, belt[2] / len];
    let tiles = tiles_within(&lattice, aim, node_m);
    let (face_i, tx, ty) = tiles.first().copied().expect("a tile under the direction");
    let tiles_per_edge = lattice.edge / TILE_EDGE;
    // ★ THE HALO MUST NOT CROSS A FACE SEAM. `node_at` carries a read at most TWO nodes past a
    // face edge; a 64-cell halo at 1 024 m is EIGHT macro nodes. So the tile is pulled one tile
    // inward wherever it sits on the face's rim, and the pull is printed.
    let clamp = |t: u32| t.clamp(1, tiles_per_edge - 2);
    let (ctx, cty) = (clamp(tx), clamp(ty));
    if (ctx, cty) != (tx, ty) {
        println!(
            "  the tile under the belt sits on the face's rim; pulled inward to keep the halo off the seam"
        );
    }
    let face = Face::from_index(face_i).expect("a face");
    let pick = TilePick {
        face,
        i0: i64::from(ctx * TILE_EDGE),
        j0: i64::from(cty * TILE_EDGE),
        nodes: i64::from(TILE_EDGE),
    };
    println!(
        "  the tile: face {face_i}, tile ({ctx}, {cty}), nodes [{}..{}) × [{}..{}), {:.0} km a side",
        pick.i0,
        pick.i0 + pick.nodes,
        pick.j0,
        pick.j0 + pick.nodes,
        pick.nodes as f64 * node_m / 1_000.0
    );

    let coarse_cells_per_node = (node_m / 1_024.0).round() as usize;
    let mut p = Params::stated();
    println!("\n  the reference's own constants, carried over verbatim: talus tan 30° (0.57),");
    println!(
        "  deposition strength 1.0, the two shares 0.1, the routing exponent's clamps s_max 1.0."
    );
    println!(
        "  ★ THE TWO RATES ARE DERIVED, NOT CARRIED — the reference's `k` is four orders of magnitude"
    );
    println!(
        "  too small in metres (MEASURED: with it the loop moved the ground 0.000 m over 300 iterations)."
    );
    {
        let mut probe = build(&lattice, &artifact, pick, coarse_cells_per_node, 0);
        calibrate(&mut probe, &mut p, iters);
    }
    let p = p;

    // ── 1. NANOSECONDS PER CELL PER ITERATION. ──────────────────────────────────────────────────
    println!("\n── 1. NANOSECONDS PER CELL PER ITERATION, one core ─────────────────────────────");
    let coarse_span = pick.nodes as usize * coarse_cells_per_node;
    let ns_coarse = time_loop(
        &lattice,
        &artifact,
        pick,
        coarse_cells_per_node,
        coarse_span,
        0,
        9,
        &p,
    );
    println!(
        "  1 024 m cells, the WHOLE tile ({coarse_span} × {coarse_span} = {} cells): {ns_coarse:.1} ns a cell an iteration (median of 3)",
        coarse_span * coarse_span
    );
    let fine_cells_per_node = (node_m / 64.0).round() as usize;
    let fine_nodes = fine_span / fine_cells_per_node;
    let share = fine_span as f64 / (pick.nodes as f64 * fine_cells_per_node as f64);
    let ns_fine = time_loop(
        &lattice,
        &artifact,
        TilePick {
            nodes: fine_nodes as i64,
            ..pick
        },
        fine_cells_per_node,
        fine_span,
        0,
        fine_iters,
        &p,
    );
    println!(
        "  64 m cells, {fine_nodes} × {fine_nodes} nodes ({fine_span} × {fine_span} = {} cells, {:.0} % of the tile's side,\n    {:.1} % of its area — the whole tile's 67 M cells would take about {:.1} GB of buffers): {ns_fine:.1} ns a cell an iteration (median of 3)",
        fine_span * fine_span,
        share * 100.0,
        share * share * 100.0,
        67.1e6 * 72.0 / 1.0e9,
    );
    println!(
        "  the two anchors §2.6 set: 0.99 ns on a card (Schott's Table 2, DERIVED) and 367 ns a node-pass in our macro sweep.\n  ours is {:.0}× the card and {:.2}× the sweep's ceiling.",
        ns_coarse / 0.99,
        ns_coarse / 367.0
    );
    println!(
        "\n  ★ WHERE THE TIME GOES — the five passes timed apart at 1 024 m, one core, median of 3."
    );
    println!(
        "  The host loop here is the STRAIGHTFORWARD one: five separate walks of the tile, each rebuilding"
    );
    println!(
        "  the eight-neighbour stencil with its bounds tests, every buffer read from memory again. An"
    );
    println!(
        "  interior fast path (no bounds tests, `present` a constant) and a fused walk are UNMEASURED."
    );
    pass_breakdown(&lattice, &artifact, pick, coarse_cells_per_node, &p);

    // ── 2. THE HALO WIDTH. ──────────────────────────────────────────────────────────────────────
    println!("\n── 2. THE HALO WIDTH at which the interior stops changing ──────────────────────");
    let reference_halo = 128usize;
    println!(
        "  the loop runs {halo_iters} iterations at 1 024 m. The compared region is THE TILE ITSELF"
    );
    println!(
        "  ({coarse_span} × {coarse_span} cells, the halo excluded), byte for byte against a SEPARATE run at a {reference_halo}-cell halo."
    );
    println!(
        "  ★ A TRIMMED CENTRE WOULD BE A VACUOUS TEST: a stencil carries news one cell an iteration, so"
    );
    println!(
        "  any centre {halo_iters} cells in from the rim is untouched whatever the halo says. The tile's OWN RIM is"
    );
    println!(
        "  where a missing halo shows, which is exactly the seam a neighbouring patch would meet."
    );
    // ★ THE REFERENCE IS ITS OWN, WIDER RUN. A list whose widest entry is also the yardstick
    // scores that entry zero by construction, which is not a measurement. So the yardstick is a
    // SEPARATE run at twice the widest halo under test, and every row below — 64 included — is
    // compared against something it is not.
    let widths = [0usize, 4, 8, 16, 32, 64];
    let reference = {
        let mut g = build(
            &lattice,
            &artifact,
            pick,
            coarse_cells_per_node,
            reference_halo,
        );
        for _ in 0..halo_iters {
            g.step(&p);
        }
        interior(&g, 0)
    };
    let mut runs: Vec<(usize, Vec<Gi>)> = Vec::new();
    for w in widths {
        let mut g = build(&lattice, &artifact, pick, coarse_cells_per_node, w);
        for _ in 0..halo_iters {
            g.step(&p);
        }
        runs.push((w, interior(&g, 0)));
    }
    let mut first_identical = None;
    let mut under_a_centimetre = None;
    for (w, field) in &runs {
        let differ = field
            .iter()
            .zip(reference.iter())
            .filter(|(a, b)| a != b)
            .count();
        let worst = field
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a.raw() - b.raw()).abs())
            .max()
            .unwrap_or(0);
        println!(
            "  halo {w:>3} cells ({:>3} km): {differ:>8} of {} interior cells differ; the worst by {:.3} m",
            (*w as f64 * 1.024) as i64,
            reference.len(),
            worst as f64 / f64::from(1 << H_BITS)
        );
        if differ == 0 && first_identical.is_none() {
            first_identical = Some(*w);
        }
        if worst <= i64::from(1 << H_BITS) / 100 && under_a_centimetre.is_none() {
            under_a_centimetre = Some(*w);
        }
    }
    match first_identical {
        Some(w) => println!(
            "  ★ THE HALO, BYTE FOR BYTE: {w} cells — the narrowest halo whose tile matches the {reference_halo}-cell run exactly."
        ),
        None => println!(
            "  ★ THE HALO, BYTE FOR BYTE: WIDER THAN 64 CELLS at {halo_iters} iterations — no run matched the {reference_halo}-cell one."
        ),
    }
    match under_a_centimetre {
        Some(w) => println!(
            "  ★ THE HALO, UNDER ONE CENTIMETRE: {w} cells — a ground nobody can see the difference of."
        ),
        None => println!(
            "  ★ THE HALO, UNDER ONE CENTIMETRE: WIDER THAN 64 CELLS at {halo_iters} iterations."
        ),
    }

    // ── 3. THE ITERATION COUNT the drainage takes to settle. ────────────────────────────────────
    println!("\n── 3. THE ITERATIONS the routed drainage takes to stop moving ──────────────────");
    println!(
        "  the stopping rule, STATED (the paper gives none; Schott et al. 2023 §4.2 only report"
    );
    println!(
        "  convergence 'after several iterations ranging between n and 4n, with an average of 1.5n'):"
    );
    println!(
        "  a cell has MOVED when its drainage changed by more than one part in 256 between two"
    );
    println!("  iterations; the loop has stopped when fewer than 0.1 % of the cells moved.");
    println!(
        "  ★ THREE SEEDINGS, not two. §2.5's sentence reads 'seed the drainage area of every fine cell"
    );
    println!(
        "  from the macro row's DISCHARGE'. Taken literally that is the SECOND row below, and the"
    );
    println!(
        "  measurement says it is SLOWER. The reason is the operator's own algebra: the gather's fixed"
    );
    println!(
        "  point is set by what each cell CATCHES (`a_seed`), never by what `a` starts at, so a large"
    );
    println!(
        "  start is a transient that has to drain off the tile before the answer appears. The CORRECTED"
    );
    println!(
        "  form — the third row — injects the outside catchment where it actually enters: on the rim"
    );
    println!(
        "  cells whose macro node just outside drains INWARD, its discharge shared over its edge cells."
    );
    println!(
        "  ★ AND TWO REGIMES, because the question is about the DRAINAGE and the erosion moves the"
    );
    println!(
        "  ground under it. THE FIELD HELD STILL isolates the routing operator — which is what the"
    );
    println!(
        "  question asks. THE FIELD EVOLVING is what a real run does, and it is reported beside it."
    );
    // The erosion switched off: the routing operator alone, on a field that cannot move.
    let still = Params {
        k: Gi::ZERO,
        thermal_rate: Gi::ZERO,
        ..p
    };
    for (regime, rp) in [
        ("the field HELD STILL ", &still),
        ("the field EVOLVING   ", &p),
    ] {
        for seeded in [Seeding::Zero, Seeding::Everywhere, Seeding::Inflow] {
            let mut g = build(&lattice, &artifact, pick, coarse_cells_per_node, 0);
            match seeded {
                Seeding::Zero => {}
                Seeding::Everywhere => {
                    seed_from_discharge(&mut g, &lattice, &artifact, pick, coarse_cells_per_node);
                }
                Seeding::Inflow => {
                    seed_inflow_boundary(&mut g, &lattice, &artifact, pick, coarse_cells_per_node);
                }
            }
            let cells = g.cells();
            let mut settled = None;
            let mut previous = g.a.clone();
            for it in 1..=iters {
                g.step(rp);
                let moved =
                    g.a.iter()
                        .zip(previous.iter())
                        .filter(|(now, was)| {
                            let d = (now.raw() - was.raw()).abs();
                            d * 256 > was.raw().abs().max(1)
                        })
                        .count();
                previous.copy_from_slice(&g.a);
                if moved * 1_000 < cells && settled.is_none() {
                    settled = Some(it);
                    break;
                }
            }
            let name = match seeded {
                Seeding::Zero => "seeded from ZERO (the reference's own start)                    ",
                Seeding::Everywhere => {
                    "seeded EVERYWHERE from the macro row's DISCHARGE (§2.5 literally)"
                }
                Seeding::Inflow => {
                    "seeded at the INFLOW BOUNDARY from the macro rows (corrected)   "
                }
            };
            match settled {
                Some(it) => println!(
                    "  {regime} {name}: settled at iteration {it} (the grid is {coarse_span} cells wide; 1.5 n would be {})",
                    (coarse_span * 3) / 2
                ),
                None => println!("  {regime} {name}: STILL MOVING after {iters} iterations"),
            }
        }
    }

    // ── 4 and 5. THE TWO BELIEVABILITY NUMBERS, before and after. ───────────────────────────────
    println!("\n── 4 & 5. G-HACK and G-BREACH at 1 024 m, before and after the loop ────────────");
    let mut g = build(&lattice, &artifact, pick, coarse_cells_per_node, 0);
    seed_from_discharge(&mut g, &lattice, &artifact, pick, coarse_cells_per_node);
    let before = g.h.clone();
    report_gates(
        &g,
        "BEFORE the loop (the bicubic upsample of the macro rows)",
    );
    let t = Instant::now();
    for _ in 0..iters {
        g.step(&p);
    }
    let wall = t.elapsed().as_secs_f64();
    // ★ WHAT THE LOOP ACTUALLY MOVED — printed so a gate that did not change cannot be read as a
    // gate that held. A believability number computed on a field the loop never touched is not a
    // measurement.
    let mut sum = 0.0f64;
    let mut worst = 0i64;
    for (now, was) in g.h.iter().zip(before.iter()) {
        let d = (now.raw() - was.raw()).abs();
        sum += d as f64;
        if d > worst {
            worst = d;
        }
    }
    let step = f64::from(1 << H_BITS);
    println!(
        "  the loop moved the ground: {:.3} m on average, {:.1} m at the worst, over {} cells",
        sum / g.cells() as f64 / step,
        worst as f64 / step,
        g.cells()
    );
    report_gates(&g, &format!("AFTER {iters} iterations"));

    // ── 6. THE DIFF AGAINST THE REFERENCE. ──────────────────────────────────────────────────────
    println!("\n── 6. THE DIFF AGAINST THE REFERENCE IMPLEMENTATION ────────────────────────────");
    reference_diff();

    // ── 7. THE WHOLE PLANET, EXTRAPOLATED. ──────────────────────────────────────────────────────
    println!("\n── 7. THE WHOLE PLANET's cost, extrapolated from (1) ───────────────────────────");
    let tiles_total = 6.0 * f64::from(tiles_per_edge) * f64::from(tiles_per_edge);
    println!(
        "  one tile at 1 024 m, {iters} iterations, MEASURED wall time: {wall:.1} s\n  the planet is {tiles_total:.0} tiles of this size, so the whole surface at 1 024 m is {:.1} core-hours\n  ({:.1} h on one core; on eight cores, about {:.1} h — ruling F6 gives the terrain workers a SHARE of the cores, never all)",
        tiles_total * wall / 3_600.0,
        tiles_total * wall / 3_600.0,
        tiles_total * wall / 3_600.0 / 8.0
    );
    println!(
        "  §2.6's own row for comparison: 1.70 × 10¹¹ cell-iterations at 1 024 m over 300 iterations,\n  which at {ns_coarse:.1} ns is {:.1} core-hours — the same number by a second road.",
        1.70e11 * ns_coarse / 1.0e9 / 3_600.0
    );
    println!(
        "  one artifact tile (524 km) to 64 m, 300 iterations: 2.01 × 10¹⁰ cell-iterations at {ns_fine:.1} ns = {:.1} min on one core.",
        2.01e10 * ns_fine / 1.0e9 / 60.0
    );
    println!("\n  coverage is UNMEASURED (this bench never runs llvm-cov).");
}

/// ★ THE FIVE PASSES TIMED APART, so the 300-odd nanoseconds can be ATTRIBUTED. The routing
/// pre-pass carries the one restoring-division loop in the whole kernel (`recip_pow2` at 40 bits,
/// 41 shift-compare-subtract steps a cell), and this is the measurement that says whether that loop
/// is the cost or a footnote.
fn pass_breakdown(
    lattice: &MacroLattice,
    artifact: &Artifact,
    pick: TilePick,
    cells_per_node: usize,
    p: &Params,
) {
    let names = [
        "① routing pre-pass (the 8 slopes, the squares, and the one reciprocal)",
        "① the drainage gather (8 weights, 8 products)",
        "② clamped stream power and the incision (two roots, the clamps, the floor)",
        "③ thermal stabilisation (16 excesses)",
        "④ the sediment gather and the deposit (8 weights, 8 products)",
    ];
    let mut medians = [0.0f64; 5];
    for (pass, slot) in medians.iter_mut().enumerate() {
        let mut samples = Vec::new();
        for _ in 0..3 {
            let mut g = build(lattice, artifact, pick, cells_per_node, 0);
            let cells = g.cells();
            // Three whole iterations first, so the pass reads a field a real run would hand it.
            for _ in 0..3 {
                g.step(p);
            }
            let t = Instant::now();
            for _ in 0..3 {
                g.one_pass(p, pass);
            }
            samples.push(t.elapsed().as_secs_f64() * 1.0e9 / (cells as f64 * 3.0));
            std::hint::black_box(g.h[cells / 2]);
        }
        samples.sort_by(f64::total_cmp);
        *slot = samples[1];
    }
    let total: f64 = medians.iter().sum();
    for (name, ns) in names.iter().zip(medians.iter()) {
        println!("    {ns:>6.1} ns  ({:>4.1} %)  {name}", ns / total * 100.0);
    }
    println!("    {total:>6.1} ns  (100.0 %)  the five passes added");
}

/// Time the loop: three runs of `iters` iterations, the MEDIAN reported, in nanoseconds a cell an
/// iteration. The grid is rebuilt for each run so no run starts from another's answer.
#[allow(clippy::too_many_arguments)]
fn time_loop(
    lattice: &MacroLattice,
    artifact: &Artifact,
    pick: TilePick,
    cells_per_node: usize,
    span: usize,
    halo: usize,
    iters: usize,
    p: &Params,
) -> f64 {
    let mut samples = Vec::new();
    for _ in 0..3 {
        let mut g = build(lattice, artifact, pick, cells_per_node, halo);
        assert_eq!(g.w, span + 2 * halo, "the grid is the span plus its halo");
        let cells = g.cells();
        let t = Instant::now();
        for _ in 0..iters {
            g.step(p);
        }
        let ns = t.elapsed().as_secs_f64() * 1.0e9 / (cells as f64 * iters as f64);
        // Read one cell so no optimiser can drop the work.
        std::hint::black_box(g.h[cells / 2]);
        samples.push(ns);
    }
    samples.sort_by(f64::total_cmp);
    samples[1]
}

/// ★ THE WORKING GRID: the tile's macro rows, upsampled BICUBICALLY (Catmull-Rom) to `cells_per_node`
/// fine cells a node, with `halo` cells of the NEIGHBOURING tiles' data on every side, read from the
/// same artifact through `node_at`.
///
/// The upsample is written in floats here because it is the bench's INPUT, not an operator: it adds
/// no detail (a bicubic never does), and the build (R2/R3) owes an integer one. The operators
/// themselves name no float anywhere.
fn build(
    lattice: &MacroLattice,
    artifact: &Artifact,
    pick: TilePick,
    cells_per_node: usize,
    halo: usize,
) -> Grid {
    let span = pick.nodes as usize * cells_per_node;
    let w = span + 2 * halo;
    let cell_m = lattice.node_m() / cells_per_node as f64;
    let node_z = |i: i64, j: i64| -> f64 {
        let node = node_at(lattice, pick.face, pick.i0 + i, pick.j0 + j);
        f64::from(artifact.rows[node as usize].z_m)
    };
    let mut h = vec![Gi::ZERO; w * w];
    let mut hard = vec![0u8; w * w];
    for y in 0..w {
        for x in 0..w {
            // The fine cell's centre, in macro-node units relative to the tile's first node.
            let u = (x as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5;
            let v = (y as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5;
            let (iu, fu) = (u.floor(), u - u.floor());
            let (iv, fv) = (v.floor(), v - v.floor());
            let mut rows = [0.0f64; 4];
            for (r, slot) in rows.iter_mut().enumerate() {
                let jj = iv as i64 + r as i64 - 1;
                let s = [
                    node_z(iu as i64 - 1, jj),
                    node_z(iu as i64, jj),
                    node_z(iu as i64 + 1, jj),
                    node_z(iu as i64 + 2, jj),
                ];
                *slot = catmull_rom(s, fu);
            }
            let z = catmull_rom(rows, fv);
            h[y * w + x] = Gi::new((z * f64::from(1 << H_BITS)) as i64);
            // The hardness: the NEAREST node's rock province, never an interpolation — a province
            // is a region, and half a granite is no rock at all (`sample_province`'s own rule).
            let node = node_at(
                lattice,
                pick.face,
                pick.i0 + u.round() as i64,
                pick.j0 + v.round() as i64,
            );
            let code = artifact.rows[node as usize].province;
            hard[y * w + x] = Province::ALL
                .iter()
                .find(|q| q.code() == code)
                .map_or(255, |q| q.erodibility_q8().min(255) as u8);
        }
    }
    let inv_axial = Gi::new(recip_pow2(cell_m.round().max(1.0) as u64, D_BITS) as i64);
    let diag = (cell_m * core::f64::consts::SQRT_2).round().max(1.0) as u64;
    let inv_diag = Gi::new(recip_pow2(diag, D_BITS) as i64);
    let inv_d = [
        inv_diag, inv_axial, inv_diag, inv_axial, inv_axial, inv_diag, inv_axial, inv_diag,
    ];
    let cells = w * w;
    Grid {
        w,
        halo,
        cell_m,
        h,
        h_next: vec![Gi::ZERO; cells],
        a: vec![Gi::ZERO; cells],
        a_next: vec![Gi::ZERO; cells],
        // Every cell catches itself: the reference's `stream = rain · cellArea + incoming`.
        a_seed: vec![Gi::new(1 << A_BITS); cells],
        sed: vec![Gi::ZERO; cells],
        sed_next: vec![Gi::ZERO; cells],
        routing: vec![Routing::default(); cells],
        hard,
        inv_d,
    }
}

/// Catmull-Rom through four samples at `t ∈ [0, 1)` between the middle two.
fn catmull_rom(s: [f64; 4], t: f64) -> f64 {
    let (a, b, c, d) = (s[0], s[1], s[2], s[3]);
    0.5 * ((2.0 * b)
        + (c - a) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t * t
        + (3.0 * b - 3.0 * c + d - a) * t * t * t)
}

/// ★ THE SEED §2.5 NAMES: every fine cell of a macro node starts with that node's OWN DISCHARGE,
/// divided down by the node's cell count, so the patch begins with the GLOBALLY correct water and
/// the iterations only correct it locally. That byte is already on the row the owner approved.
fn seed_from_discharge(
    g: &mut Grid,
    lattice: &MacroLattice,
    artifact: &Artifact,
    pick: TilePick,
    cells_per_node: usize,
) {
    let per_node = (cells_per_node * cells_per_node) as f64;
    for y in 0..g.w {
        for x in 0..g.w {
            let u = (x as f64 + 0.5 - g.halo as f64) / cells_per_node as f64 - 0.5;
            let v = (y as f64 + 0.5 - g.halo as f64) / cells_per_node as f64 - 0.5;
            let node = node_at(
                lattice,
                pick.face,
                pick.i0 + u.round() as i64,
                pick.j0 + v.round() as i64,
            );
            let row = artifact.rows[node as usize];
            let q = unlog(row.discharge_log, 2);
            let rain = unlog(row.rain, 4).max(1.0);
            // The discharge in units of one fine cell's own catch: the node's water over the water
            // one fine cell of the same rain makes.
            let cells = (q / (rain * lattice.area_m2(node) as f64 / per_node)).min(1.0e9);
            g.a[y * g.w + x] = Gi::new((cells * f64::from(1 << A_BITS)) as i64);
        }
    }
}

/// ★ THE CORRECTED SEED: the outside catchment injected WHERE IT ENTERS. For every fine cell on the
/// grid's rim we read the macro node one step further out; if that node's own D8 receiver points
/// back into the tile, its discharge — shared over the `cells_per_node` fine cells facing us — is
/// added to the rim cell's `a_seed`. Because `a_seed` is what a cell CATCHES, it moves the gather's
/// FIXED POINT, which is the thing that decides the answer; an initial `a` only decides how long the
/// transient takes.
///
/// **Example.** A river crosses onto the tile from the highland two nodes west. The rim cells it
/// crosses now catch the highland's whole flow every iteration, so the valley inside the tile is cut
/// by a river that knows where it came from — and the patch next door, computing its own rim from
/// the same shipped rows, cuts the same river.
fn seed_inflow_boundary(
    g: &mut Grid,
    lattice: &MacroLattice,
    artifact: &Artifact,
    pick: TilePick,
    cells_per_node: usize,
) {
    let per_cell = cells_per_node as f64;
    for y in 0..g.w {
        for x in 0..g.w {
            let rim_x = x == 0 || x == g.w - 1;
            let rim_y = y == 0 || y == g.w - 1;
            if !rim_x && !rim_y {
                continue;
            }
            // The outward step, in macro-node units: one node past the tile's own edge.
            let (ox, oy) = (
                if x == 0 { -1 } else { i64::from(rim_x) },
                if y == 0 { -1 } else { i64::from(rim_y) },
            );
            let ni = pick.i0 + (x / cells_per_node) as i64 + ox;
            let nj = pick.j0 + (y / cells_per_node) as i64 + oy;
            let outside = node_at(lattice, pick.face, ni, nj);
            let row = artifact.rows[outside as usize];
            let slot = row.receiver_facies & RECEIVER_SLOT_MASK;
            if row.receiver_facies & RECEIVER_NONE != 0 {
                continue;
            }
            // Does the outside node drain back towards us? Its receiver's offset must point
            // opposite to the outward step we just took.
            let (di, dj) = OFF[slot as usize];
            let inward = (ox == 0 || i64::from(di) == -ox) && (oy == 0 || i64::from(dj) == -oy);
            if !inward {
                continue;
            }
            let q = unlog(row.discharge_log, 2);
            let rain = unlog(row.rain, 4).max(1.0);
            let cells_of_water = q
                / (rain * lattice.area_m2(outside) as f64
                    / (cells_per_node * cells_per_node) as f64);
            let share = (cells_of_water / per_cell).min(1.0e9);
            g.a_seed[y * g.w + x] += Gi::new((share * f64::from(1 << A_BITS)) as i64);
        }
    }
}

/// The value behind one of the row's quantised logarithms (`artifact::log2_class` run backwards).
fn unlog(class: u8, frac: u32) -> f64 {
    let bits = u32::from(class) >> frac;
    let below = f64::from(u32::from(class) & ((1 << frac) - 1)) / f64::from(1u32 << frac);
    // `log2_class` stores `63 − leading_zeros(v + 1)`, which is floor(log2(v + 1)).
    let exponent = f64::from(bits);
    let mut x = 1.0f64;
    let mut k = 0.0;
    while k < exponent {
        x *= 2.0;
        k += 1.0;
    }
    (x * (1.0 + below) - 1.0).max(0.0)
}

/// The grid's interior: the tile minus `trim` cells on every side, in row order.
fn interior(g: &Grid, trim: usize) -> Vec<Gi> {
    let lo = g.halo + trim;
    let hi = g.w - g.halo - trim;
    let mut out = Vec::with_capacity((hi - lo) * (hi - lo));
    for y in lo..hi {
        out.extend_from_slice(&g.h[y * g.w + lo..y * g.w + hi]);
    }
    out
}

/// ★ THE TWO BELIEVABILITY NUMBERS on one field.
fn report_gates(g: &Grid, when: &str) {
    println!("  {when}:");
    let (c, n, points) = g_hack(g);
    let band_c = (1.0..=6.0).contains(&c);
    let band_n = (0.45..=0.7).contains(&n);
    println!(
        "    G-HACK   L = {c:.2}·a^{n:.3} over {points} network cells — c {} [1, 6], n {} [0.45, 0.7]{}",
        if band_c { "INSIDE" } else { "OUTSIDE" },
        if band_n { "INSIDE" } else { "OUTSIDE" },
        if band_c && band_n { "  ✓" } else { "  ✗" }
    );
    let (breach, pits) = g_breach(g);
    println!(
        "    G-BREACH {breach:.3e} m³ a depression, over {pits} depressions (this file's definition; the\n             reference's own units are UNVERIFIED, so the BEFORE/AFTER ratio is the number that can fail)"
    );
}

/// ★ G-HACK — the Hack's-law fit `L = c·a^n` (Sassolas-Serrayet, Cattin, Ferry & Godard 2018;
/// Schott et al. 2023 §7.3 validate a terrain exactly this way).
///
/// The network is drawn with the STEEPEST-descent receiver (the operators' own tie rule, the lowest
/// stencil slot), the drainage area `a` accumulated down it in km², and `L` the LONGEST upstream
/// flow path in km. Both are accumulated in one pass over the cells sorted by height, highest
/// first, which is the topological order of a descent tree. The fit is an ordinary least squares on
/// the logarithms over the cells whose basin passes ten cells — below that a "river" is a pixel.
fn g_hack(g: &Grid) -> (f64, f64, usize) {
    let cells = g.cells();
    let cell_km2 = g.cell_m * g.cell_m / 1.0e6;
    let mut recv = vec![usize::MAX; cells];
    let mut dist_km = vec![0.0f64; cells];
    for y in 0..g.w {
        for x in 0..g.w {
            let i = y * g.w + x;
            let (idx, present) = g.stencil(x, y);
            let mut best = Gi::ZERO;
            for slot in 0..NEIGHBOURS {
                if present >> slot & 1 == 0 {
                    continue;
                }
                let drop = g.h[i] - g.h[idx[slot]];
                if drop > best {
                    best = drop;
                    recv[i] = idx[slot];
                    let diagonal = OFF[slot].0 != 0 && OFF[slot].1 != 0;
                    dist_km[i] = g.cell_m
                        * if diagonal {
                            core::f64::consts::SQRT_2
                        } else {
                            1.0
                        }
                        / 1_000.0;
                }
            }
        }
    }
    let mut order: Vec<u32> = (0..cells as u32).collect();
    order.sort_unstable_by(|a, b| g.h[*b as usize].cmp(&g.h[*a as usize]));
    let mut area = vec![cell_km2; cells];
    let mut length = vec![0.0f64; cells];
    for i in order {
        let i = i as usize;
        let r = recv[i];
        if r == usize::MAX {
            continue;
        }
        area[r] += area[i];
        let candidate = length[i] + dist_km[i];
        if candidate > length[r] {
            length[r] = candidate;
        }
    }
    // The fit, over the network only.
    let floor = 10.0 * cell_km2;
    let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0, 0.0, 0.0, 0.0, 0usize);
    for i in 0..cells {
        if area[i] < floor || length[i] <= 0.0 {
            continue;
        }
        let x = ln(area[i]);
        let y = ln(length[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        n += 1;
    }
    if n < 2 {
        return (0.0, 0.0, n);
    }
    let count = n as f64;
    let slope = (count * sxy - sx * sy) / (count * sxx - sx * sx);
    let intercept = (sy - slope * sx) / count;
    (exp(intercept), slope, n)
}

/// A natural logarithm written out, because the bench's own crate is fenced against `f64::ln` in
/// the recipe and this file keeps the same discipline for the numbers it reports: the argument is
/// split into a power of two and a mantissa, and the mantissa's logarithm comes from the
/// `atanh` series, which converges on `[1, 2)` in a dozen terms.
fn ln(x: f64) -> f64 {
    let mut e = 0i32;
    let mut m = x;
    while m >= 2.0 {
        m *= 0.5;
        e += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        e -= 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut term = z;
    let mut sum = 0.0;
    let mut k = 0;
    while k < 20 {
        sum += term / f64::from(2 * k + 1);
        term *= z2;
        k += 1;
    }
    2.0 * sum + f64::from(e) * core::f64::consts::LN_2
}

/// `e^x`, by the same discipline: the argument is split into whole powers of two and a remainder
/// the Taylor series takes.
fn exp(x: f64) -> f64 {
    let k = (x / core::f64::consts::LN_2).round();
    let r = x - k * core::f64::consts::LN_2;
    let mut term = 1.0f64;
    let mut sum = 0.0f64;
    let mut i = 0;
    while i < 20 {
        sum += term;
        i += 1;
        term *= r / f64::from(i);
    }
    let mut out = sum;
    let mut n = 0.0;
    while n < k.abs() {
        out *= if k > 0.0 { 2.0 } else { 0.5 };
        n += 1.0;
    }
    out
}

/// ★ G-BREACH — a breaching volume, on THE DEFINITION THIS FILE STATES, because Schott et al. 2024
/// publish the metric's VALUES (their erosion 0.02–4.8 ×10³, procedural noise 1.7–41) but not the
/// code that computes them, and we could not run theirs.
///
/// Ours: a priority flood from the grid's rim (Barnes, Lehman & Mulla 2014) records, for every
/// cell, the neighbour it was reached from — so every cell holds a path to the rim whose highest
/// point is the lowest such highest point of any path. A cell is a PIT where every neighbour stands
/// at or above it. For each pit we walk that recorded path outward and add `max(0, h − h_pit)`
/// times the cell area: the material a ONE-CELL-WIDE channel at the pit's own level must still
/// remove for the pit to drain. The answer is the mean over the pits, in cubic metres.
///
/// **What this can and cannot say.** The ABSOLUTE number cannot be set beside the paper's, because
/// their unit is not published — that comparison is UNVERIFIED. The BEFORE/AFTER ratio on the same
/// field with the same definition is a measurement that can fail, and it is the one reported.
fn g_breach(g: &Grid) -> (f64, usize) {
    let cells = g.cells();
    let area = g.cell_m * g.cell_m;
    let mut done = vec![false; cells];
    let mut from = vec![usize::MAX; cells];
    // The priority flood: the rim first, lowest first, each cell carrying the highest point of the
    // path that reached it.
    let mut heap: BinaryHeap<(core::cmp::Reverse<i64>, usize)> = BinaryHeap::new();
    for y in 0..g.w {
        for x in 0..g.w {
            if x == 0 || y == 0 || x == g.w - 1 || y == g.w - 1 {
                let i = y * g.w + x;
                done[i] = true;
                heap.push((core::cmp::Reverse(g.h[i].raw()), i));
            }
        }
    }
    while let Some((core::cmp::Reverse(spill), i)) = heap.pop() {
        let (x, y) = (i % g.w, i / g.w);
        let (idx, present) = g.stencil(x, y);
        for (slot, j) in idx.iter().copied().enumerate() {
            if present >> slot & 1 == 0 || done[j] {
                continue;
            }
            done[j] = true;
            from[j] = i;
            heap.push((core::cmp::Reverse(g.h[j].raw().max(spill)), j));
        }
    }
    // The pits, and the channel each needs.
    let mut total = 0.0f64;
    let mut pits = 0usize;
    for y in 1..g.w - 1 {
        for x in 1..g.w - 1 {
            let i = y * g.w + x;
            let (idx, present) = g.stencil(x, y);
            let lowest = (0..NEIGHBOURS)
                .filter(|slot| present >> slot & 1 == 1)
                .all(|slot| g.h[idx[slot]] >= g.h[i]);
            if !lowest {
                continue;
            }
            pits += 1;
            let pit = g.h[i].raw();
            let mut c = from[i];
            let mut volume = 0.0f64;
            while c != usize::MAX {
                let over = g.h[c].raw() - pit;
                if over <= 0 {
                    break;
                }
                volume += f64::from(u32::try_from(over).unwrap_or(u32::MAX))
                    / f64::from(1 << H_BITS)
                    * area;
                c = from[c];
            }
            total += volume;
        }
    }
    if pits == 0 {
        return (0.0, 0);
    }
    (total / pits as f64, pits)
}

/// ★ THE DIFF AGAINST THE REFERENCE. We could NOT run `H-Schott/MultiScaleErosion`: it is a C++
/// OpenGL application with GLFW, GLEW and ImGui, and this machine has no such harness — so the
/// comparison against its RUNNING output is UNVERIFIED and is said so.
///
/// What CAN be diffed, and is: a float transcription of the four shaders' own arithmetic, written
/// from the source read at their raw URLs, run beside our integer operators on ONE STATED FIELD.
/// That is a measurement that can fail — a wrong exponent, a wrong clamp or a wrong sign shows up
/// at once — and it is the honest half of the diff.
fn reference_diff() {
    use vd_recipe::amplify::{drop_to, weight_in};
    // THE STATED FIELD: a cell at 100 m with two lower neighbours and six level ones, cells 64 m
    // across, drainage 10 000 cells.
    let cell = 64.0f64;
    let hp = 100.0f64;
    let neighbours = [100.0, 68.0, 100.0, 100.0, 84.0, 100.0, 100.0, 100.0];
    let inv_axial = Gi::new(recip_pow2(64, D_BITS) as i64);
    let inv_diag = Gi::new(recip_pow2(90, D_BITS) as i64);
    let inv_d = [
        inv_diag, inv_axial, inv_diag, inv_axial, inv_axial, inv_diag, inv_axial, inv_diag,
    ];
    let mut nb = [Gi::ZERO; NEIGHBOURS];
    for slot in 0..NEIGHBOURS {
        nb[slot] = Gi::new((neighbours[slot] * f64::from(1 << H_BITS)) as i64);
    }
    let h = Gi::new((hp * f64::from(1 << H_BITS)) as i64);
    let r = route(h, &nb, &inv_d, 0xFF);

    // ① THE REFERENCE'S `GetFlowWeighted`, in floats: `weight[i] = pow(|slope_i|, flow_p) / Σ`,
    // with `flow_p = 1.3` — the shipped value. Ours squares instead (the module states why).
    let mut ref_w = [0.0f64; NEIGHBOURS];
    let mut ref_w2 = [0.0f64; NEIGHBOURS];
    let (mut sum13, mut sum2) = (0.0f64, 0.0f64);
    for slot in 0..NEIGHBOURS {
        let diagonal = OFF[slot].0 != 0 && OFF[slot].1 != 0;
        let d = cell
            * if diagonal {
                core::f64::consts::SQRT_2
            } else {
                1.0
            };
        let s = (hp - neighbours[slot]) / d;
        if s > 0.0 {
            ref_w[slot] = exp(1.3 * ln(s));
            ref_w2[slot] = s * s;
            sum13 += ref_w[slot];
            sum2 += ref_w2[slot];
        }
    }
    let mut worst13 = 0.0f64;
    let mut worst2 = 0.0f64;
    for slot in 0..NEIGHBOURS {
        let ours = weight_in(nb[slot], h, inv_d[slot], r.inv_sum, 0xFF, slot).raw() as f64
            / f64::from(1 << 20);
        if sum13 > 0.0 {
            let d = (ours - ref_w[slot] / sum13).abs();
            if d > worst13 {
                worst13 = d;
            }
            let d2 = (ours - ref_w2[slot] / sum2).abs();
            if d2 > worst2 {
                worst2 = d2;
            }
        }
    }
    println!(
        "  ① ROUTING. Against the reference's OWN exponent (`flow_p = 1.3`): the widest weight differs by {worst13:.4} of one."
    );
    println!(
        "     Against the SAME operator at our integer exponent (p = 2, the module's stated departure): {worst2:.2e} of one —"
    );
    println!(
        "     which is the integer arithmetic's own rounding, and says the transcription and the kernel agree."
    );

    // ② THE REFERENCE'S stream power: `spe = pow(stream, p_sa) · clamp(pow(slope, p_sl), 0, 1)`,
    // then `clamp(spe, 0, max_spe) · k`. `p_sa = 0.8`; ours is 3/4.
    let a = 10_000.0f64;
    let slope = r.steepest.raw() as f64 / f64::from(1 << S_BITS);
    let held = if slope * slope > 1.0 {
        1.0
    } else {
        slope * slope
    };
    let ref_08 = exp(0.8 * ln(a)) * held;
    let ref_075 = exp(0.75 * ln(a)) * held;
    let p = Params::stated();
    let ours = stream_power(
        Gi::new((a * f64::from(1 << A_BITS)) as i64),
        r.steepest,
        p.a_max,
        p.s_max,
        p.spe_max,
    )
    .raw() as f64
        / f64::from(1 << A_BITS);
    println!(
        "  ② STREAM POWER. The reference at its own `p_sa = 0.8`: {ref_08:.3}. The same operator at our m = 3/4: {ref_075:.3}.\n     Ours: {ours:.3} — {:.2e} relative against the m = 3/4 transcription, {:.1} % against the reference's 0.8.",
        ((ours - ref_075) / ref_075).abs(),
        ((ours - ref_08) / ref_08).abs() * 100.0
    );

    // ③ THERMAL. The reference's `tanThresholdAngle = 0.57`, the excess above it moved downhill.
    // ★ THE FIELD ABOVE IS TOO FLAT FOR THIS OPERATOR — its steepest drop is 0.5, UNDER the talus,
    // so both sides would read zero and the diff would prove nothing. A STEEPER stated field: the
    // cell at 100 m, slot 1 at 20 m (a drop of 1.25) and slot 4 at 60 m (a drop of 0.625).
    let steep = [100.0f64, 20.0, 100.0, 100.0, 60.0, 100.0, 100.0, 100.0];
    let mut nb_steep = [Gi::ZERO; NEIGHBOURS];
    for slot in 0..NEIGHBOURS {
        nb_steep[slot] = Gi::new((steep[slot] * f64::from(1 << H_BITS)) as i64);
    }
    let talus = 0.57f64;
    let mut ref_net = 0.0f64;
    for slot in 0..NEIGHBOURS {
        let diagonal = OFF[slot].0 != 0 && OFF[slot].1 != 0;
        let d = cell
            * if diagonal {
                core::f64::consts::SQRT_2
            } else {
                1.0
            };
        let down = (hp - steep[slot]) / d - talus;
        let up = (steep[slot] - hp) / d - talus;
        if down > 0.0 {
            ref_net -= down;
        }
        if up > 0.0 {
            ref_net += up;
        }
    }
    let mut ours_net = 0.0f64;
    for slot in 0..NEIGHBOURS {
        let down = drop_to(h, nb_steep[slot], inv_d[slot], 0xFF, slot).raw();
        let up = drop_to(nb_steep[slot], h, inv_d[slot], 0xFF, slot).raw();
        let t = (talus * f64::from(1 << S_BITS)) as i64;
        if down > t {
            ours_net -= f64::from(u32::try_from(down - t).unwrap_or(0)) / f64::from(1 << S_BITS);
        }
        if up > t {
            ours_net += f64::from(u32::try_from(up - t).unwrap_or(0)) / f64::from(1 << S_BITS);
        }
    }
    println!(
        "  ③ THERMAL. The reference's excess net: {ref_net:.6}. Ours: {ours_net:.6} — a difference of {:.2e} of a unit slope.",
        (ref_net - ours_net).abs()
    );

    // ④ DEPOSITION. The reference's three lines, verbatim from `deposition.glsl`.
    let (sed_in, spe_h) = (100.0f64, 10.0f64);
    let mut sed = sed_in + 0.1 * spe_h;
    let ref_d = if 1.0 * sed > spe_h {
        let d = (1.0 * sed - spe_h) * 0.1;
        if d < sed { d } else { sed }
    } else {
        0.0
    };
    sed -= ref_d;
    let (d_ours, kept) = deposit(
        Gi::new((sed_in * f64::from(1 << H_BITS)) as i64),
        Gi::new((spe_h * f64::from(1 << H_BITS)) as i64),
        p.deposition_strength,
        p.share,
    );
    let d_ours_f = f64::from(u32::try_from(d_ours.raw()).unwrap_or(0)) / f64::from(1 << H_BITS);
    let kept_f = f64::from(u32::try_from(kept.raw()).unwrap_or(0)) / f64::from(1 << H_BITS);
    println!(
        "  ④ DEPOSITION. The reference deposits {ref_d:.4} and keeps {sed:.4}. Ours: {d_ours_f:.4} and {kept_f:.4} —\n     {:.2e} and {:.2e} relative. The tolerance the formats set is one part in 65 536 of a metre (15 micrometres).",
        ((d_ours_f - ref_d) / ref_d).abs(),
        ((kept_f - sed) / sed).abs()
    );
    println!(
        "  ★ UNVERIFIED: we could not RUN the reference (C++/OpenGL/GLFW/ImGui, no harness on this machine)."
    );
    println!(
        "    The diff above is against a FLOAT TRANSCRIPTION of its four shaders, read from their raw sources."
    );
}
