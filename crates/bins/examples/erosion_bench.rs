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

// A measurement tool, never shipped: its index loops mirror the solvers' own arrays, and its report functions take
// every reading they print.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "bench code: the loops mirror the solvers' arrays; the reports take every reading"
)]

use std::time::Instant;

use vd_bins::artifact_worker::{SolveJob, run_solve};
// ★ G-HACK and G-BREACH live in ONE place (plan step 1, ruling B6): the shipped-ground instrument
// reads the very same two functions, so the bench's numbers and the shipped ground's can be set
// side by side. `OFF`, `ln` and `exp` came with them, unchanged.
use vd_bins::drainage_gates::{Heights, OFF, exp, g_breach, g_hack, ln};
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
    // ★ R2's SLICE 0 (rulings B8 item 3, B57): R1's loop is THE CONTROL on today's world, so the bench reads a stored
    // artifact (`--artifact`, written by `fold_steps`) instead of a 35-minute solve, picks the tile under a pinned
    // stand's eye (`--aim x,y,z`), may skip the timing studies (`--gates-only`) and draws the tile's shaded relief
    // before and after the loop (`--hillshade DIR`, `--label NAME`). Nothing here touches the game.
    let sarg = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|k| args.get(k + 1))
            .cloned()
    };
    let gates_only = args.iter().any(|a| a == "--gates-only");
    lem::ROCK_SCATTER.store(
        args.iter().any(|a| a == "--lem-rock"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::RELAX_STEPS.store(
        arg("--lem-relax-steps", 400),
        std::sync::atomic::Ordering::Relaxed,
    );
    WINDOW_M.store(
        arg("--lem-window-km", 8) * 1_000,
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::STEADY_FORM.store(
        args.iter().any(|a| a == "--lem-steady"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::HOLD.store(
        args.iter().any(|a| a == "--lem-hold"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::L1_LINEAR.store(
        args.iter().any(|a| a == "--l1-linear"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::SETTLE.store(
        args.iter().any(|a| a == "--l1-settle"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::TRACE.store(
        args.iter().any(|a| a == "--l1-trace"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::JOINT_LOOP.store(
        args.iter().any(|a| a == "--l1-joint"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::FAST_TREE.store(
        args.iter().any(|a| a == "--l1-fast-tree"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::MEANS_SETTLE.store(
        args.iter().any(|a| a == "--l1-means-settle"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::COARSE_START.store(
        args.iter().any(|a| a == "--l1-coarse-start"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::L1_ROUNDS.store(arg("--l1-rounds", 10), std::sync::atomic::Ordering::Relaxed);
    lem::WARP.store(
        args.iter().any(|a| a == "--l1-warp"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::SETTLE_M.store(
        arg("--l1-settle-m", 1),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::CYCLE_STOP.store(
        args.iter().any(|a| a == "--l1-cycle-stop"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::HYSTERESIS_PPM.store(
        arg("--l1-hysteresis-ppm", 0),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::POWER_MODE.store(arg("--l1-power", 0), std::sync::atomic::Ordering::Relaxed);
    lem::POINT_SET.store(arg("--point-set", 0), std::sync::atomic::Ordering::Relaxed);
    lem::SLOPE_HOLD.store(
        args.iter().any(|a| a == "--l1-slope-hold"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::FILL.store(
        args.iter().any(|a| a == "--l1-fill"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::CARVE.store(
        args.iter().any(|a| a == "--l1-carve"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::L2_CHANNELS.store(
        args.iter().any(|a| a == "--l2-channels"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::L2_HOLD_CELLS.store(
        args.iter().any(|a| a == "--l2-hold-cells"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::DAMPED.store(
        args.iter().any(|a| a == "--l1-damped"),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::SLOPE_SWEEPS.store(
        arg("--l1-slope-sweeps", 80),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::HOLD_INNER.store(
        args.iter().any(|a| a == "--l1-hold-inner"),
        std::sync::atomic::Ordering::Relaxed,
    );
    if args.iter().any(|a| a == "--power-error") {
        // The fenced powers against the platform's over the law's range (x from 1e-14 to 1e4, log-spaced).
        let (mut worst_gf, mut worst_fine) = (0.0f64, 0.0f64);
        let (mut t_plat, mut t_gf, mut t_fine) = (0.0f64, 0.0f64, 0.0f64);
        let xs: Vec<f64> = (0..2_000_000)
            .map(|i| 10f64.powf(-14.0 + 18.0 * (i as f64 + 0.5) / 2.0e6))
            .collect();
        let t = Instant::now();
        let plat: Vec<f64> = xs.iter().map(|x| x.powf(1.0 / lem::JOINT_N_PUB)).collect();
        t_plat += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let gf: Vec<f64> = xs
            .iter()
            .map(|x| {
                vd_terrain::Gf::from_f64(*x)
                    .powf(vd_terrain::Gf::from_f64(1.0 / lem::JOINT_N_PUB))
                    .to_f64()
            })
            .collect();
        t_gf += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let fine: Vec<f64> = xs
            .iter()
            .map(|x| {
                lem::fine_exp(
                    lem::fine_ln(vd_terrain::Gf::from_f64(*x))
                        * vd_terrain::Gf::from_f64(1.0 / lem::JOINT_N_PUB),
                )
                .to_f64()
            })
            .collect();
        t_fine += t.elapsed().as_secs_f64();
        for i in 0..xs.len() {
            worst_gf = worst_gf.max(((gf[i] - plat[i]) / plat[i]).abs());
            worst_fine = worst_fine.max(((fine[i] - plat[i]) / plat[i]).abs());
        }
        println!(
            "THE POWER x^(1/2.2) on 2 M values: the platform {:.1} ns each; the fence's powf {:.1} ns, worst relative error {worst_gf:.2e}; the fine power {:.1} ns, worst {worst_fine:.2e}",
            t_plat * 500.0,
            t_gf * 500.0,
            t_fine * 500.0
        );
        return;
    }
    if let Some(v) = sarg("--l1-sc") {
        lem::THRESHOLD_MILLI.store(
            (v.parse::<f64>().expect("a slope") * 1_000.0).round() as usize,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    lem::L1_SIGMA_C.store(arg("--l1-sigma", 70), std::sync::atomic::Ordering::Relaxed);
    lem::HOLD_CELL_M.store(
        arg("--lem-hold-m", 128),
        std::sync::atomic::Ordering::Relaxed,
    );
    lem::SHARED_AREA.store(
        args.iter().any(|a| a == "--lem-mfd"),
        std::sync::atomic::Ordering::Relaxed,
    );
    if let Some(um) = sarg("--uplift-um").and_then(|v| v.parse::<usize>().ok()) {
        lem::RISE_NM.store(um * 1_000, std::sync::atomic::Ordering::Relaxed);
        lem::RISE_SET.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let hillshade_dir = sarg("--hillshade");
    let label = sarg("--label").unwrap_or_else(|| "tile".to_string());

    // ★ THE EARTH REFERENCE (ruling B76; the owner: "An Earth reference"): `--earth FILE.hgt` reads a public 1-arc-second
    // tile (outside the repository, ruling B36), averages it into square cells of `--r2-cell-m` (each cell the mean of
    // the heights inside it — the far view's own rule), and prints the same readings and pictures as R2, with no
    // erosion. A reading, never a target a constant is turned toward (B38).
    // ★ THE GRID EROSION ON EARTH'S OWN FACTS (ruling B79, the owner: "Test grid erosion on Earth"): `--earth-erode
    // FILE.hgt --rain-mm R` builds Earth's 8 km node means from the tile, starts from their smooth Catmull-Rom at
    // `--r2-cell-m`, erodes with R2's laws (the routing switches, `--r2-creep`) for `--r2-steps` × `--r2-dt-yr` while
    // the hold keeps every node's mean, runs the same dose again (G-DOSE), and prints the readings and the pictures
    // beside Earth's real ground of the same square at the same cell.
    // ★ A STORED SCENE RENDERED AGAIN (`--view SCENE --out PNG [--cam EAST_KM,SOUTH_KM,BEARING_DEG,PITCH_DEG,OVER_M]`):
    // the coarse ground with its fine window, no solve; the scene's own camera unless one is stated.
    if let Some(scene) = sarg("--view") {
        let s = view::load_scene(&scene);
        let mut cam = s.cam;
        if let Some(c) = sarg("--cam") {
            let v: Vec<f64> = c.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            assert_eq!(
                v.len(),
                5,
                "--cam EAST_KM,SOUTH_KM,BEARING_DEG,PITCH_DEG,OVER_M"
            );
            cam = view::Camera {
                x: v[0] * 1_000.0,
                n: -v[1] * 1_000.0,
                z: 0.0,
                yaw: v[2].to_radians(),
                pitch: v[3].to_radians(),
                over: v[4],
            };
        }
        let out = sarg("--out").expect("--out PNG");
        let sample = view::layered(&s.coarse, s.w, s.cell_m, &s.fine, s.fw, s.x0, s.y0, s.fcell);
        let detail = view::layered_cell(s.fw as f64 * s.fcell, s.x0, s.y0, s.fcell, s.cell_m);
        view::render_with(
            &sample,
            &detail,
            2.0 * s.w as f64 * s.cell_m,
            &cam,
            s.dry,
            &out,
        );
        println!(
            "the view: {out} (the camera {:.2} km east, {:.2} km south, bearing {:.0}°, pitch {:.1}°, {:.0} m over the ground)",
            cam.x / 1_000.0,
            -cam.n / 1_000.0,
            cam.yaw.to_degrees(),
            cam.pitch.to_degrees(),
            cam.over
        );
        return;
    }

    // ★ THE STREAM LAW ON IRREGULAR POINTS ON EARTH'S OWN 8 KM FACTS (`--lem-earth FILE.hgt --rain-mm R`): the tile's
    // node means and one published rain; the fine ground by `lem::solve`; the readings and the pictures beside Earth's
    // real ground of the same square at the same cell.
    if let Some(path) = sarg("--lem-earth") {
        let cell_m = arg("--r2-cell-m", 128) as f64;
        let spacing_m = arg("--lem-spacing-m", 128) as f64;
        let rain_m = arg("--rain-mm", 1_000) as f64 / 1_000.0;
        let (nodes, nw) = earth_cells(&path, 8_192.0);
        println!("LEM EARTH {path}: {nw} × {nw} nodes of 8 192 m, rain {rain_m:.3} m/yr");
        let rain = vec![rain_m; nw * nw];
        let out = lem::solve(&nodes, &rain, nw, 8_192.0, spacing_m, cell_m);
        let w = out.w;
        let (real_all, rw) = earth_cells(&path, cell_m);
        let real: Vec<f64> = (0..w * w)
            .map(|k| real_all[(k / w).min(rw - 1) * rw + (k % w).min(rw - 1)])
            .collect();
        lem_report(
            &out.raster,
            &real,
            "earth",
            w,
            cell_m,
            hillshade_dir.as_deref(),
            &label,
        );
        if let Some(dir) = hillshade_dir.as_deref() {
            let fine_m = arg("--lem-fine-m", 16) as f64;
            lem_fine_views(
                &out,
                &rain,
                nw,
                8_192.0,
                cell_m,
                &real,
                Some(&path),
                dir,
                &label,
                fine_m,
                None,
            );
        }
        return;
    }

    if let Some(path) = sarg("--earth-erode") {
        let cell_m = arg("--r2-cell-m", 128) as f64;
        let rain_m = arg("--rain-mm", 1_000) as f64 / 1_000.0;
        let steps = arg("--r2-steps", 400);
        let dt_yr = arg("--r2-dt-yr", 250_000) as f64;
        let cpn = (8_192.0 / cell_m).round() as usize;
        let (nodes, nw) = earth_cells(&path, 8_192.0);
        let (mut reg, mut start) = r2::earth_region(&nodes, nw, cpn, cell_m, rain_m);
        reg.threshold = args.iter().any(|a| a == "--threshold");
        // ★ THE START'S OWN ROUGHNESS (`--start-relief`, a diagnostic): the game's older-landscape law (β 2.09, sized
        // 131 m at 110 km; ruling B60) carried below the node, so the water can choose its paths; a smooth start
        // sends every stream down the same slope side by side.
        if args.iter().any(|a| a == "--start-relief") {
            let added = r2::older_relief(&reg, cell_m);
            for (i, h) in start.iter_mut().enumerate() {
                if !reg.sea[i] {
                    *h += added[i];
                }
            }
        }
        reg.rho8 = args.iter().any(|a| a == "--r2-rho8");
        reg.mfd = args.iter().any(|a| a == "--r2-mfd");
        reg.dinf = reg.rho8 || reg.mfd || args.iter().any(|a| a == "--r2-dinf");
        reg.creep = args.iter().any(|a| a == "--r2-creep");
        // ★ THE MEASURED UPLIFT as the forcing (`--uplift-um`, micrometres a year; a published rate of the place, a
        // measurement input only): the hold goes off, the sea and the rim stay the base level.
        reg.uplift_m_yr = arg("--uplift-um", 0) as f64 * 1.0e-6;
        reg.hold_on = reg.uplift_m_yr <= 0.0;
        if !reg.hold_on {
            println!(
                "  the forcing: the rock rises {:.3} mm/yr (a published rate, a measurement input); the mean hold off",
                reg.uplift_m_yr * 1_000.0
            );
        }
        let w = reg.w;
        println!(
            "EARTH EROSION {path}: {nw} × {nw} nodes of 8 192 m ({:.0} km a side) as {w} × {w} cells of {cell_m:.0} m; rain \
             {rain_m:.3} m/yr; {steps} steps of {dt_yr:.0} years ({:.0} Myr); routing {}; hillslope law {}; {}; start {}",
            nw as f64 * 8.192,
            steps as f64 * dt_yr / 1.0e6,
            if reg.mfd {
                "MFD"
            } else if reg.rho8 {
                "Rho8"
            } else if reg.dinf {
                "D∞"
            } else {
                "D8"
            },
            if reg.creep { "on" } else { "off" },
            if reg.threshold {
                "the threshold law at 35°"
            } else {
                "the stepped talus"
            },
            if args.iter().any(|a| a == "--start-relief") {
                "smooth + the older relief"
            } else {
                "smooth"
            }
        );
        let a = r2::run(&reg, &start, steps, dt_yr);
        let b = r2::run(&reg, &a.z, steps, dt_yr);
        let change =
            |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum::<f64>();
        let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        println!(
            "  the mean height: start {:.0} m, after A {:.0} m, after B {:.0} m",
            mean(&start),
            mean(&a.z),
            mean(&b.z)
        );
        println!(
            "  {:.3} s a step; A moved the ground {:.1} m on average; G-DOSE (2T against T) {:.1} % of A's change; the rise \
             (the hold's) {:.4} mm/yr",
            a.step_s,
            change(&a.z, &start) / (w * w) as f64,
            100.0 * change(&b.z, &a.z) / change(&a.z, &start).max(f64::MIN_POSITIVE),
            b.rise_m_yr * 1_000.0
        );
        // EARTH'S REAL GROUND of the same square at the same cell (the tile's NW corner is both fields' origin).
        let (real_all, rw) = earth_cells(&path, cell_m);
        let real: Vec<f64> = (0..w * w).map(|k| real_all[(k / w) * rw + k % w]).collect();
        let real_reg = r2::plain_region(w, cell_m, &real);
        for (who, rg, f) in [("GENERATED", &reg, &a.z), ("EARTH", &real_reg, &real)] {
            let (straight, n) = r2::straight_share(rg, f, 0.05, 16);
            let [x, y, d] = r2::stripes(rg, f);
            let (lean, nl) = r2::lean(rg, f, 0.05);
            println!(
                "  {who}: THE STRAIGHT LINES {:.1} % of {n}; THE STRIPES {x:.3}/{y:.3}/{d:.3}; THE LEAN {:.1} % of {nl}",
                straight * 100.0,
                lean * 100.0
            );
        }
        if let Some(dir) = hillshade_dir.as_deref() {
            for (when, f) in [("generated", &a.z), ("earth", &real), ("start", &start)] {
                for (what, file) in shade::write_views(f, w, 0, cell_m, 0.0, dir, &label, when) {
                    println!("  the shaded relief, {when}, {what}: {file}");
                }
            }
        }
        return;
    }

    if let Some(path) = sarg("--earth") {
        let cell_m = arg("--r2-cell-m", 1_024) as f64;
        let (z, w) = earth_cells(&path, cell_m);
        let reg = r2::plain_region(w, cell_m, &z);
        println!(
            "EARTH {path}: {w} × {w} cells of {cell_m:.0} m ({:.0} km a side), heights {:.0} m to {:.0} m",
            w as f64 * cell_m / 1_000.0,
            z.iter().copied().fold(f64::MAX, f64::min),
            z.iter().copied().fold(f64::MIN, f64::max)
        );
        let (straight, n) = r2::straight_share(&reg, &z, 0.05, 16);
        println!(
            "  THE STRAIGHT LINES EARTH: {:.1} % of {n} valley cells keep their next 16 cells downstream within one cell of a straight line",
            straight * 100.0
        );
        let [x, y, d] = r2::stripes(&reg, &z);
        println!("  THE STRIPES EARTH: along x {x:.3}, along y {y:.3}, along the diagonals {d:.3}");
        let (lean, nl) = r2::lean(&reg, &z, 0.05);
        println!(
            "  THE GRID'S LEAN EARTH: {:.1} % over {nl} cells (none: 22 %)",
            lean * 100.0
        );
        if let Some(dir) = hillshade_dir.as_deref() {
            for (what, file) in shade::write_views(&z, w, 0, cell_m, 0.0, dir, &label, "earth") {
                println!("  the shaded relief, {what}: {file}");
            }
        }
        return;
    }

    println!("erosion_bench — R1, the bench (docs/investigation/2026-09-22, §5 R1)");
    println!("  the machine: Apple M4 Pro, one core, release, one job at a time");

    // ★ B1's 128 m SOLVE ON THE WET MOON (`--fine-moon`): the unit test's premise — the tile its rule picks (the most
    // land nodes, the first of equals, faces then rows then columns) solves; its time, side, passes and range.
    if args.iter().any(|a| a == "--fine-moon") {
        use vd_terrain::artifact::ZField;
        let moon = vd_terrain::home::home_moon();
        let words = vd_terrain::solve::SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let t = Instant::now();
        let art = run_solve(&SolveJob { body: moon, words }).expect("the solve");
        println!(
            "B1 THE WET MOON: solved in {:.1} s, sea {} m",
            t.elapsed().as_secs_f64(),
            art.sea_m
        );
        let lat = MacroLattice::of(&moon).expect("a lattice");
        let edge = lat.edge as i32;
        let mut best = (0usize, None);
        for f in 0..6u8 {
            let face = Face::from_index(f).expect("a face");
            for ty in 0..=(edge - 1) >> 3 {
                for tx in 0..=(edge - 1) >> 3 {
                    let land = (0..64)
                        .map(|k| (tx * 8 + (k & 7), ty * 8 + (k >> 3)))
                        .filter(|&(i, j)| (i < edge) & (j < edge))
                        .filter(|&(i, j)| {
                            art.z_m(lat.index(face, i, j))
                                .is_some_and(|z| z > art.sea_m)
                        })
                        .count();
                    if land > best.0 {
                        best = (land, Some(vd_terrain::fine::l1::TileKey { face, tx, ty }));
                    }
                }
            }
        }
        let key = best.1.expect("a tile with land");
        let t = Instant::now();
        let tile = vd_terrain::fine::l1::solve_l1(&moon, &lat, &art, key);
        let dt = t.elapsed().as_secs_f64();
        match tile {
            Some(tile) => {
                let held: Vec<i32> = tile
                    .quarters
                    .iter()
                    .copied()
                    .filter(|&q| q != i32::MIN)
                    .collect();
                println!(
                    "B1 THE WET MOON's tile {key:?} ({} land nodes of 64): solved in {dt:.1} s, side {}, {} points, passes {:?}, {} cells missing, {:.1} … {:.1} m over the sea",
                    best.0,
                    tile.side,
                    tile.points,
                    tile.passes,
                    tile.quarters.len() - held.len(),
                    f64::from(*held.iter().min().unwrap_or(&0)) / 4.0,
                    f64::from(*held.iter().max().unwrap_or(&0)) / 4.0
                );
            }
            None => println!("B1 THE WET MOON's tile {key:?}: NO SOLVE ({dt:.1} s)"),
        }
        return;
    }

    // ── THE INPUT: the home planet, solved once (or its stored artifact). ───────────────────────
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
    let artifact = match sarg("--artifact")
        .as_deref()
        .and_then(vd_bins::artifact_file::read)
    {
        Some(a) => {
            println!(
                "  the artifact: read from the file (generator {}), ARTIFACT_VERSION {}, sea {} m",
                vd_terrain::GENERATOR_VERSION,
                a.version,
                a.sea_m
            );
            a
        }
        None => {
            let t0 = Instant::now();
            let a = run_solve(&SolveJob { body, words }).expect("the solve answers");
            println!(
                "  the solve: {:.1} s, ARTIFACT_VERSION {}, sea {} m",
                t0.elapsed().as_secs_f64(),
                a.version,
                a.sea_m
            );
            a
        }
    };

    // ★ THE MACRO'S PRESENT RATE, PLANET-WIDE (ruling B76, the balance): `--macro-rates` reads the joint law at every
    // draining dry node, prints its distribution over the land, and the fastest 64 km regions with their aim.
    if args.iter().any(|a| a == "--macro-rates") {
        macro_rates_planet(&lattice, &artifact);
        return;
    }
    // ★ B85 CHECKED ON THE BENCH (`--fine-corner F,TX,TY`): the four finished tiles TX..TX+1 × TY..TY+1 by the corner
    // rule (each from its 3 × 3 raw tiles: sixteen solves) against the bench's stitch of the same four raw tiles (each row
    // of tiles along its column cut, then the rows along the row cut). The readings: the slope break across the inner
    // edges (each line's largest within the cut's band) against the same inside a tile, the break on the finished tiles'
    // own straight edges against the inside, and the two surfaces' difference; then the maps.
    // ★ B87 THE NEEDS OF A VIEW (`--fine-needs X,Y,Z`, metres in the planet's frame): the one call the shard and the
    // gateway serve a viewer by — the tiles by their finest level, and the records one level a record makes.
    if let Some(v) = sarg("--fine-needs") {
        let p: Vec<f64> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let radial = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let dir = [p[0] / radial, p[1] / radial, p[2] / radial];
        let drawn = artifact.drawn_range_m(&lattice);
        let body = body.with_sea_m(artifact.sea());
        let needs = vd_terrain::artifact::fine_needs_of_head(
            &body,
            (drawn.0, drawn.1, artifact.sea_m),
            dir,
            radial,
            vd_core::geometry::drawable_theta_min_rad(),
        );
        let mut by_level = [0usize; 6];
        for (_, need) in &needs {
            by_level[usize::from(*need)] += 1;
        }
        let records: usize = needs.iter().map(|(_, n)| 6 - usize::from(*n)).sum();
        println!(
            "B87 THE NEEDS: {} tiles, by the finest level {by_level:?}, {records} records one level a record (the band {drawn:?}, the sea {})",
            needs.len(),
            artifact.sea_m
        );
        return;
    }
    // ★ B2's FACE-EDGE CUT (ruling B86 item 9 (1); `--fine-edge F,C` — face F's corner C: 0 NW, 1 NE, 2 SW, 3 SE —
    // or `--fine-edge auto`, the cube corner with the most land by rule): the tiles within two of a cube corner on its
    // three faces, finished by the generator's own `finish`; the ground laid out in face F's chart (the quarter past the
    // cube corner empty). The step across the tiles' straight edges on a face's edge against an edge inside a face and a
    // line inside a tile, the slope break at the cube corner against a tile's middle, and the map.
    if let Some(v) = sarg("--fine-edge") {
        use std::collections::{BTreeMap, BTreeSet};
        use vd_terrain::artifact::ZField;
        use vd_terrain::fine::finish::block_keys;
        use vd_terrain::fine::l1::{TILE_NODES, TileKey, cells_per_node_shift, solve_l1};
        use vd_terrain::fine::points::chart_cell;
        use vd_terrain::fine::stitch::{BLEND, Field, MISSING};
        use vd_terrain::fine::tile::{finish, raw_field};
        let shift = cells_per_node_shift(&body, &lattice, 7).expect("nests");
        let n7 = (lattice.edge << shift) as i32;
        let side = i64::from(TILE_NODES) << shift;
        let ts = shift + TILE_NODES.trailing_zeros();
        let edge = lattice.edge as i32;
        // The land nodes in the 2 × 2 tiles at corner `c` of `face`.
        let land = |face: Face, c: usize| -> usize {
            let (bi, bj) = (
                if c & 1 == 1 { edge - 16 } else { 0 },
                if c & 2 == 2 { edge - 16 } else { 0 },
            );
            (0..256)
                .filter(|q| {
                    artifact
                        .z_m(lattice.index(face, bi + (q & 15), bj + (q >> 4)))
                        .is_some_and(|z| z > artifact.sea_m)
                })
                .count()
        };
        let (face, c) = if v == "auto" {
            let mut best = (0usize, Face::PosX, 0usize);
            for f in Face::ALL {
                for c in 0..4 {
                    // The corner's three faces' land together, seen from this face.
                    let l = land(f, c);
                    if l > best.0 {
                        best = (l, f, c);
                    }
                }
            }
            (best.1, best.2)
        } else {
            let k: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (Face::from_index(k[0] as u8).expect("a face"), k[1])
        };
        let (vx, vy) = (
            if c & 1 == 1 { i64::from(n7) } else { 0 },
            if c & 2 == 2 { i64::from(n7) } else { 0 },
        );
        let r = 2 * side;
        // The finished tiles over the region (two tiles each way from the corner), and the raw tiles their blocks read.
        let mut fin_keys: BTreeSet<TileKey> = BTreeSet::new();
        for by in -2..2i64 {
            for bx in -2..2i64 {
                let (x, y) = (vx + bx * side + side / 2, vy + by * side + side / 2);
                if let Some(cell) = chart_cell(face, x as i32, y as i32, n7) {
                    fin_keys.insert(TileKey {
                        face: Face::from_index(cell.face).expect("a face"),
                        tx: cell.i >> ts,
                        ty: cell.j >> ts,
                    });
                }
            }
        }
        let raw_keys: Vec<TileKey> = fin_keys
            .iter()
            .flat_map(|&k| block_keys(k, shift, n7).into_iter().flatten())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        println!(
            "B2 THE FACE'S EDGE: face {} corner {c} ({} land nodes in its 2 × 2 tiles); {} finished tiles over {} raw tiles",
            face.index(),
            land(face, c),
            fin_keys.len(),
            raw_keys.len()
        );
        let t = Instant::now();
        let threads = std::thread::available_parallelism().map_or(4, |p| p.get());
        let mut raw: BTreeMap<TileKey, Field> = BTreeMap::new();
        for batch in raw_keys.chunks(threads) {
            let got: Vec<(TileKey, Field)> = std::thread::scope(|scope| {
                let hs: Vec<_> = batch
                    .iter()
                    .map(|&key| {
                        let (body, lattice, artifact) = (&body, &lattice, &artifact);
                        scope.spawn(move || {
                            (
                                key,
                                raw_field(
                                    key,
                                    solve_l1(body, lattice, artifact, key).expect("solves"),
                                    shift,
                                ),
                            )
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("a tile")).collect()
            });
            raw.extend(got);
        }
        println!(
            "B2 THE FACE'S EDGE: {} raw tiles in {:.1} s",
            raw.len(),
            t.elapsed().as_secs_f64()
        );
        let mut fin: BTreeMap<TileKey, Vec<i32>> = BTreeMap::new();
        for &k in &fin_keys {
            let block = block_keys(k, shift, n7).map(|b| b.map(|b| (b, &raw[&b])));
            let kind = match block
                .iter()
                .flatten()
                .map(|(b, _)| b.face)
                .collect::<BTreeSet<_>>()
                .len()
            {
                1 => "inside a face",
                2 => "on a face's edge",
                _ => "at the cube corner",
            };
            let t = Instant::now();
            let levels = finish(block, shift, n7).unwrap_or_else(|| panic!("{k:?} finishes"));
            println!(
                "B2 THE FACE'S EDGE: {k:?} {kind} finished in {:.3} s",
                t.elapsed().as_secs_f64()
            );
            fin.insert(k, levels.into_iter().next().expect("the square"));
        }
        // The ground in face F's chart over the region: each cell from its own tile's finished square.
        let w = 2 * r;
        let (gx0, gy0) = (vx - r, vy - r);
        let mut z = vec![MISSING; (w * w) as usize];
        for y in 0..w {
            for x in 0..w {
                if let Some(cell) = chart_cell(face, (gx0 + x) as i32, (gy0 + y) as i32, n7) {
                    let k = TileKey {
                        face: Face::from_index(cell.face).expect("a face"),
                        tx: cell.i >> ts,
                        ty: cell.j >> ts,
                    };
                    let (ox, oy) = (
                        i64::from(cell.i) - i64::from(k.tx) * side,
                        i64::from(cell.j) - i64::from(k.ty) * side,
                    );
                    z[(y * w + x) as usize] = fin[&k][(oy * side + ox) as usize];
                }
            }
        }
        let at = |x: i64, y: i64| -> Option<f64> {
            ((x >= 0) & (y >= 0) & (x < w) & (y < w))
                .then(|| z[(y * w + x) as usize])
                .filter(|&q| q != MISSING)
                .map(|q| f64::from(q) / 4.0)
        };
        // The step between the cells `a − 1` and `a` on line `l` against the mean slope of the three pairs around it.
        let jump = |a: i64, l: i64, across_x: bool| -> Option<f64> {
            let zz = |d: i64| if across_x { at(a + d, l) } else { at(l, a + d) };
            Some(((zz(0)? - zz(-1)?) - (zz(1)? - zz(-2)?) / 3.0).abs())
        };
        let reach = (1i64 << shift) - BLEND + BLEND;
        // A boundary's lines: every line crossing it, away from the cube corner's reach.
        let lines = |a: i64, across_x: bool| -> Vec<f64> {
            (0..w)
                .filter(|&l| {
                    let (x, y) = if across_x { (a, l) } else { (l, a) };
                    (x - r).abs().max((y - r).abs()) > 2 * reach
                })
                .filter_map(|l| jump(a, l, across_x))
                .collect()
        };
        let pct = |mut v: Vec<f64>, q: usize| -> f64 {
            v.sort_by(f64::total_cmp);
            v.get(v.len() * q / 100).copied().unwrap_or(f64::NAN)
        };
        let mut on_face_edge = lines(r, true);
        on_face_edge.extend(lines(r, false));
        let mut inside_face: Vec<f64> = Vec::new();
        for a in [r - side, r + side] {
            inside_face.extend(lines(a, true));
            inside_face.extend(lines(a, false));
        }
        let mut in_tile: Vec<f64> = Vec::new();
        for a in [side / 2, r - side / 2, r + side / 2, w - side / 2] {
            in_tile.extend(lines(a, true));
            in_tile.extend(lines(a, false));
        }
        let say = |name: &str, v: &Vec<f64>| {
            println!(
                "B2 THE FACE'S EDGE: the step across {name}: 50th / 99th / largest {:.2} / {:.2} / {:.2} m ({} lines)",
                pct(v.clone(), 50),
                pct(v.clone(), 99),
                v.iter().copied().fold(0.0, f64::max),
                v.len()
            )
        };
        say("the face's edge (tiles of two faces)", &on_face_edge);
        say("a tile edge inside a face", &inside_face);
        say("a straight line inside a tile", &in_tile);
        // The slope break (the second difference over 4 cells per cell length) near the cube corner and in a tile's middle.
        let brk = |x: i64, y: i64, across_x: bool| -> Option<f64> {
            let zz = |d: i64| if across_x { at(x + d, y) } else { at(x, y + d) };
            Some((((zz(3)? - zz(1)?) - (zz(-1)? - zz(-3)?)) / 256.0).abs())
        };
        let around = |cx: i64, cy: i64| -> Vec<f64> {
            (cy - reach..=cy + reach)
                .flat_map(|y| (cx - reach..=cx + reach).map(move |x| (x, y)))
                .flat_map(|(x, y)| [brk(x, y, true), brk(x, y, false)])
                .flatten()
                .collect()
        };
        let corner = around(r, r);
        let middle = around(r + side / 2, r + side / 2);
        println!(
            "B2 THE CUBE CORNER: the slope break within {reach} cells, median / 99th / largest {:.3} / {:.3} / {:.3}; a tile's middle {:.3} / {:.3} / {:.3}",
            pct(corner.clone(), 50),
            pct(corner.clone(), 99),
            corner.iter().copied().fold(0.0, f64::max),
            pct(middle.clone(), 50),
            pct(middle.clone(), 99),
            middle.iter().copied().fold(0.0, f64::max)
        );
        if let Some(dir) = hillshade_dir.as_deref() {
            let _ = std::fs::create_dir_all(dir);
            let lo = z
                .iter()
                .filter(|&&q| q != MISSING)
                .copied()
                .min()
                .unwrap_or(0);
            let zf: Vec<f64> = z
                .iter()
                .map(|&q| f64::from(if q == MISSING { lo } else { q }) / 4.0)
                .collect();
            let path_png = format!("{dir}/fine_edge_{}_{c}.png", face.index());
            hillshade_rect(&zf, w as usize, w as usize, 128.0, &path_png);
            println!(
                "B2 THE FACE'S EDGE: the map {path_png} (the quarter past the cube corner drawn flat at the lowest height)"
            );
        }
        return;
    }
    if let Some(v) = sarg("--fine-corner") {
        use vd_terrain::fine::l1::{TileKey, cells_per_node_shift, solve_l1};
        use vd_terrain::fine::stitch::{BLEND, Field, finished_tile, stitch_columns, stitch_rows};
        let k: Vec<i32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let face = Face::from_index(k[0] as u8).expect("a face");
        let shift = cells_per_node_shift(&body, &lattice, 7).expect("nests") as i64;
        let node = 1i64 << shift;
        let side = 8 * node;
        let band = node - BLEND;
        let (tx0, ty0) = (k[1], k[2]);
        let keys: Vec<TileKey> = (-1..3)
            .flat_map(|dy| {
                (-1..3).map(move |dx| TileKey {
                    face,
                    tx: tx0 + dx,
                    ty: ty0 + dy,
                })
            })
            .collect();
        let t = Instant::now();
        let raw: Vec<Field> = std::thread::scope(|scope| {
            let hs: Vec<_> = keys
                .iter()
                .map(|&key| {
                    let (body, lattice, artifact) = (&body, &lattice, &artifact);
                    scope.spawn(move || {
                        let t = solve_l1(body, lattice, artifact, key).expect("solves");
                        // Each raster covers its tile and half its buffer: one node beyond each edge.
                        Field {
                            gx0: (i64::from(key.tx) * 8 - 1) * node,
                            gy0: (i64::from(key.ty) * 8 - 1) * node,
                            w: t.side,
                            h: t.side,
                            v: t.quarters,
                        }
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("a tile")).collect()
        });
        println!(
            "B85: sixteen raw tiles in {:.1} s; the band ±{band} cells, the blend ±{BLEND}",
            t.elapsed().as_secs_f64()
        );
        let at = |dx: i32, dy: i32| &raw[((dy + 1) * 4 + dx + 1) as usize];
        let t = Instant::now();
        let fin: Vec<Field> = [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .map(|&(dx, dy)| {
                let block = [
                    at(dx - 1, dy - 1),
                    at(dx, dy - 1),
                    at(dx + 1, dy - 1),
                    at(dx - 1, dy),
                    at(dx, dy),
                    at(dx + 1, dy),
                    at(dx - 1, dy + 1),
                    at(dx, dy + 1),
                    at(dx + 1, dy + 1),
                ];
                finished_tile(
                    block,
                    i64::from(k[1] + dx),
                    i64::from(k[2] + dy),
                    side,
                    band,
                )
                .expect("joins")
            })
            .collect();
        println!(
            "B85: a finished tile in {:.3} s",
            t.elapsed().as_secs_f64() / 4.0
        );
        // B2: the generator's own forms — `finish` gives the same square, and the record decodes back exactly.
        {
            use vd_terrain::fine::tile::{decode_levels, encode_levels, finish};
            let n7 = (lattice.edge << shift) as i32;
            let block: [Option<(TileKey, &Field)>; 9] = std::array::from_fn(|s| {
                let (dx, dy) = (s as i32 % 3 - 1, s as i32 / 3 - 1);
                Some((
                    TileKey {
                        face,
                        tx: k[1] + dx,
                        ty: k[2] + dy,
                    },
                    at(dx, dy),
                ))
            });
            let levels = finish(block, shift as u32, n7).expect("finishes");
            let ts = side.trailing_zeros();
            let record = encode_levels(&levels, ts).expect("codes");
            println!(
                "B2 THE FORMS: `finish` gives the bench's square {}; the record {:.0} KB decodes back {}",
                if levels[0] == fin[0].v {
                    "exactly"
                } else {
                    "NOT"
                },
                record.len() as f64 / 1_000.0,
                if decode_levels(&record) == Some((ts, levels)) {
                    "exactly"
                } else {
                    "NOT"
                }
            );
        }
        // What the shard ships (B85): the finished tile and its mean levels (rungs 8–12), each through the coder, decoded back.
        {
            use vd_terrain::fine::{coder, stitch::mean_levels};
            let f = &fin[0];
            let near = coder::encode(&f.v, f.w, f.h).expect("encodes");
            assert_eq!(
                coder::decode(&near, f.w, f.h).as_deref(),
                Some(&f.v[..]),
                "the finished tile decodes back"
            );
            let levels = mean_levels(f, f.gx0, f.gy0, f.w, 5).expect("the mean levels");
            let mut far = 0usize;
            for (i, l) in levels.iter().enumerate() {
                let n = f.w >> (i + 1);
                let b = coder::encode(l, n, n).expect("encodes");
                assert_eq!(
                    coder::decode(&b, n, n).as_deref(),
                    Some(&l[..]),
                    "a mean level decodes back"
                );
                far += b.len();
            }
            println!(
                "B85: the finished tile {} × {}: near {:.0} KB ({:.2} bytes a sample), its five mean levels {:.0} KB, every level decoded back exactly",
                f.w,
                f.h,
                near.len() as f64 / 1_000.0,
                near.len() as f64 / (f.w * f.h) as f64,
                far as f64 / 1_000.0
            );
        }
        // ★ B1 (g) THE READ on the four finished tiles: rung 7 gives the raster back exactly, rung 12 the 16 m … means
        // exactly, rung 0 runs across the tiles' edge with no step; the time a read.
        {
            use vd_terrain::fine::read::{FineBook, FineZ, MEAN_LEVELS};
            use vd_terrain::fine::stitch::mean_levels;
            let mut book = FineBook::new(&body, &lattice, artifact.sea_m).expect("a book");
            for (f, (dx, dy)) in fin.iter().zip([(0, 0), (1, 0), (0, 1), (1, 1)]) {
                let mut levels = vec![f.v.clone()];
                levels.extend(mean_levels(f, f.gx0, f.gy0, f.w, MEAN_LEVELS).expect("the means"));
                assert!(book.insert(
                    TileKey {
                        face,
                        tx: k[1] + dx,
                        ty: k[2] + dy
                    },
                    levels
                ));
            }
            let sea_q = i64::from(artifact.sea_m) * 4;
            let as_q = |z: FineZ| match z {
                FineZ::Held(g) => (g.raw() >> 28) / 32 - sea_q,
                other => panic!("{other:?}"),
            };
            // Rung 7 over the four tiles' inner square (the stencil stays on held tiles), against the rasters.
            let mut off7 = 0usize;
            for y in 2..2 * side - 2 {
                for x in 2..2 * side - 2 {
                    let (gx, gy) = (i64::from(k[1]) * side + x, i64::from(k[2]) * side + y);
                    let f = &fin[((y >= side) as usize) * 2 + (x >= side) as usize];
                    off7 += usize::from(
                        as_q(book.z(face, 7, 7, gx as i32, gy as i32)) != i64::from(f.at(gx, gy)),
                    );
                }
            }
            // Rung 12: level 5's 16 × 16 means of each tile, inner cells.
            let mut off12 = 0usize;
            for (f, (dx, dy)) in fin.iter().zip([(0i64, 0i64), (1, 0), (0, 1), (1, 1)]) {
                let lv = &mean_levels(f, f.gx0, f.gy0, f.w, MEAN_LEVELS).expect("the means")[4];
                // The cells whose stencil (one before, two after) stays on the four held tiles.
                let (xs, ys) = (
                    if dx == 0 { 1..16 } else { 0..14 },
                    if dy == 0 { 1..16 } else { 0..14 },
                );
                for y in ys {
                    for x in xs.clone() {
                        let (i, j) = (
                            (i64::from(k[1]) + dx) * 16 + x,
                            (i64::from(k[2]) + dy) * 16 + y,
                        );
                        off12 += usize::from(
                            as_q(book.z(face, 12, 12, i as i32, j as i32))
                                != i64::from(lv[(y * 16 + x) as usize]),
                        );
                    }
                }
            }
            // Rung 0 along a row across the T|E edge (1 m cells): the largest step between neighbours there and away.
            let row_j = ((i64::from(k[2]) * side + side / 2) << 7) as i32;
            let edge_i = ((i64::from(k[1]) + 1) * side) << 7;
            let t = Instant::now();
            let z0: Vec<f64> = (edge_i - 4_000..edge_i + 4_000)
                .map(|i| match book.z(face, 0, 0, i as i32, row_j) {
                    FineZ::Held(g) => g.raw() as f64 / (128.0 * (1u64 << 28) as f64),
                    other => panic!("{other:?}"),
                })
                .collect();
            let per_read = t.elapsed().as_secs_f64() / 8_000.0;
            let steps: Vec<f64> = z0.windows(2).map(|p| (p[1] - p[0]).abs()).collect();
            let at_edge = steps[3_990..4_010].iter().copied().fold(0.0, f64::max);
            let away = steps[..3_000].iter().copied().fold(0.0, f64::max);
            println!(
                "B1 (g) THE READ: rung 7 against the rasters {off7} cells differ; rung 12 against the means {off12} differ; at rung 0 the largest 1 m step at the tiles' edge {at_edge:.3} m, away from it {away:.3} m; {:.2} µs a read",
                per_read * 1e6
            );
        }
        let (gx0, gy0, w) = (i64::from(k[1]) * side, i64::from(k[2]) * side, 2 * side);
        let mut corner = vec![0i32; (w * w) as usize];
        for f in &fin {
            for y in 0..f.h {
                for x in 0..f.w {
                    corner[((f.gy0 - gy0 + y as i64) * w + f.gx0 - gx0 + x as i64) as usize] =
                        f.v[y * f.w + x];
                }
            }
        }
        let (top, _) = stitch_columns(at(0, 0), at(1, 0), gx0 + side, band);
        let (bottom, _) = stitch_columns(at(0, 1), at(1, 1), gx0 + side, band);
        let (all, _) = stitch_rows(&top, &bottom, gy0 + side, band);
        let bench: Vec<i32> = (0..w * w)
            .map(|i| all.at(gx0 + i % w, gy0 + i / w))
            .collect();
        // The slope break at a cell across x (or y): the second difference over 4 cells, per cell length, as one number.
        let brk = |f: &[i32], x: i64, y: i64, across_x: bool| -> f64 {
            let z = |d: i64| {
                let (xx, yy) = if across_x { (x + d, y) } else { (x, y + d) };
                f64::from(f[(yy * w + xx) as usize]) / 4.0
            };
            (((z(3) - z(1)) - (z(-1) - z(-3))) / 256.0).abs()
        };
        let pct = |mut v: Vec<f64>, q: usize| {
            v.sort_by(f64::total_cmp);
            v[v.len() * q / 100]
        };
        // Each line's largest break within `±reach` of `c` (a column when `across_x`, else a row), over the lines `lines`.
        let line_max = |f: &[i32],
                        c: i64,
                        reach: i64,
                        across_x: bool,
                        lines: std::ops::Range<i64>|
         -> Vec<f64> {
            lines
                .map(|l| {
                    (c - reach..=c + reach)
                        .map(|a| {
                            if across_x {
                                brk(f, a, l, true)
                            } else {
                                brk(f, l, a, false)
                            }
                        })
                        .fold(0.0, f64::max)
                })
                .collect()
        };
        let reach = band + BLEND;
        // The lines away from the square's outer border and from the other inner edge.
        let lines = || (reach + 4)..(w - reach - 4);
        for (name, f) in [("THE CORNER RULE", &corner), ("THE BENCH'S STITCH", &bench)] {
            let mut cut_v = line_max(f, side, reach, true, lines());
            cut_v.extend(line_max(f, side, reach, false, lines()));
            let mut in_v = line_max(f, side / 2, reach, true, lines());
            in_v.extend(line_max(f, side / 2, reach, false, lines()));
            in_v.extend(line_max(f, side + side / 2, reach, true, lines()));
            in_v.extend(line_max(f, side + side / 2, reach, false, lines()));
            // The four-tile corner itself: the cells within the band of both inner edges.
            let near: Vec<f64> = (side - reach..=side + reach)
                .flat_map(|y| (side - reach..=side + reach).map(move |x| (x, y)))
                .flat_map(|(x, y)| [brk(f, x, y, true), brk(f, x, y, false)])
                .collect();
            let mid: Vec<f64> = (side / 2 - reach..=side / 2 + reach)
                .flat_map(|y| (side / 2 - reach..=side / 2 + reach).map(move |x| (x, y)))
                .flat_map(|(x, y)| [brk(f, x, y, true), brk(f, x, y, false)])
                .collect();
            println!(
                "B85 {name}: a line's largest break, median / 90th / 99th — across the cuts {:.3} / {:.3} / {:.3}, inside a tile {:.3} / {:.3} / {:.3}; at the four-tile corner 99th {:.3} against a tile's middle {:.3}",
                pct(cut_v.clone(), 50),
                pct(cut_v.clone(), 90),
                pct(cut_v, 99),
                pct(in_v.clone(), 50),
                pct(in_v.clone(), 90),
                pct(in_v, 99),
                pct(near, 99),
                pct(mid, 99)
            );
        }
        // The finished tiles' own straight edges (two tiles computed apart meet there): the step between the cells `a − 1`
        // and `a` against the mean slope of the three pairs around it, in metres, on the edge and on a straight line inside.
        let jump = |f: &[i32], a: i64, l: i64, across_x: bool| -> f64 {
            let z = |d: i64| {
                let (xx, yy) = if across_x { (a + d, l) } else { (l, a + d) };
                f64::from(f[(yy * w + xx) as usize]) / 4.0
            };
            ((z(0) - z(-1)) - (z(1) - z(-2)) / 3.0).abs()
        };
        let ls = || (0..w).filter(|&l| (l > 3) & (l < w - 4));
        let edge: Vec<f64> = ls()
            .flat_map(|l| [jump(&corner, side, l, true), jump(&corner, side, l, false)])
            .collect();
        let inner: Vec<f64> = ls()
            .flat_map(|l| {
                [
                    jump(&corner, side / 2, l, true),
                    jump(&corner, side / 2, l, false),
                ]
            })
            .collect();
        // The control: the same lines on the bench's one stitched surface (no tile computed apart meets there).
        let control: Vec<f64> = ls()
            .flat_map(|l| [jump(&bench, side, l, true), jump(&bench, side, l, false)])
            .collect();
        println!(
            "B85: the control, the bench's stitch on the same lines 50th / 99th / largest {:.2} / {:.2} / {:.2} m",
            pct(control.clone(), 50),
            pct(control.clone(), 99),
            control.iter().copied().fold(0.0, f64::max)
        );
        println!(
            "B85: the step across the finished tiles' straight edges 50th / 99th / largest {:.2} / {:.2} / {:.2} m, across a straight line inside {:.2} / {:.2} / {:.2} m",
            pct(edge.clone(), 50),
            pct(edge.clone(), 99),
            edge.iter().copied().fold(0.0, f64::max),
            pct(inner.clone(), 50),
            pct(inner.clone(), 99),
            inner.iter().copied().fold(0.0, f64::max)
        );
        let diff: Vec<f64> = (0..w * w)
            .filter(|i| {
                let (x, y) = (i % w, i / w);
                (x >= reach) & (y >= reach) & (x < w - reach) & (y < w - reach)
            })
            .map(|i| f64::from(corner[i as usize] - bench[i as usize]).abs() / 4.0)
            .collect();
        let moved = diff.iter().filter(|&&d| d > 0.0).count();
        println!(
            "B85: the two surfaces away from the square's border: {:.1} % of the cells differ, by {:.2} m on average there, 99th {:.1} m, largest {:.1} m",
            100.0 * moved as f64 / diff.len() as f64,
            diff.iter().sum::<f64>() / moved.max(1) as f64,
            pct(diff.clone(), 99),
            diff.iter().copied().fold(0.0, f64::max)
        );
        if let Some(dir) = hillshade_dir.as_deref() {
            let _ = std::fs::create_dir_all(dir);
            for (what, f) in [("corner_rule", &corner), ("bench_stitch", &bench)] {
                let z: Vec<f64> = f.iter().map(|&q| f64::from(q) / 4.0).collect();
                let path_png = format!("{dir}/fine_corner_{what}_{}_{}_{}.png", k[0], k[1], k[2]);
                hillshade_rect(&z, w as usize, w as usize, 128.0, &path_png);
                println!("B85: the map, {what}: {path_png}");
            }
        }
        return;
    }
    // ★ THE CLIENT'S OWN CHEAP GROUND (the owner: "use the same algorithm on the client ... real one, but not fine one";
    // `--client-standin F,TX,TY`): the server's own first stage alone (the law on points four rung-7 cells apart, its
    // ten rounds) against the full 128 m solve — the time of each, and how far each stand-in stands from the finished
    // ground over the tile's own square: the first stage's ground and today's 8 km ground.
    if let Some(v) = sarg("--client-standin") {
        use vd_terrain::Gf;
        use vd_terrain::fine::l1::{
            BUFFER_NODES, COARSE_ROUNDS, COARSE_RUNG, L1_RUNG, TILE_NODES, TileKey,
            cells_per_node_shift, domain, rounds, solve_l1,
        };
        use vd_terrain::fine::points::chart_cell;
        let k: Vec<i32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let key = TileKey {
            face: Face::from_index(k[0] as u8).expect("a face"),
            tx: k[1],
            ty: k[2],
        };
        let t = Instant::now();
        let full = solve_l1(&body, &lattice, &artifact, key).expect("solves");
        let t_full = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let d = domain(&body, &lattice, &artifact, COARSE_RUNG, key).expect("a coarse domain");
        let mut u = d.ref_u0.clone();
        let mut h = d.base.clone();
        let passes = rounds(&d, &mut u, &mut h, COARSE_ROUNDS);
        let t_coarse = t.elapsed().as_secs_f64();
        // The same raster as the full solve's, read on the first stage's points.
        let n_r = body.ladder().cells_per_edge(L1_RUNG);
        let shift = cells_per_node_shift(&body, &lattice, L1_RUNG).expect("nests");
        let half = BUFFER_NODES >> 1;
        let (i0, j0) = (
            (key.tx * TILE_NODES - half) << shift,
            (key.ty * TILE_NODES - half) << shift,
        );
        let side = ((TILE_NODES + 2 * half) << shift) as usize;
        let inv_n = vd_recipe::bend::inv_n_of(n_r);
        let unit = (vd_terrain::units::STEPS_PER_M << 28) as f64;
        let (mut d_coarse, mut d_macro) = (Vec::new(), Vec::new());
        let mut from = 0u32;
        let edge = (TILE_NODES << shift) as usize;
        let lo = (half << shift) as usize;
        let t = Instant::now();
        for b in lo..lo + edge {
            for a in lo..lo + edge {
                let Some(c) = chart_cell(key.face, i0 + a as i32, j0 + b as i32, n_r as i32) else {
                    continue;
                };
                let q = vd_recipe::bend::direction(i32::from(c.face), c.i, c.j, inv_n);
                let Some((tri, w)) =
                    vd_terrain::fine::geometry::locate(&d.dirs, &d.graph, from, &q)
                else {
                    continue;
                };
                from = tri;
                let vtx = d.graph.tris[tri as usize];
                let zc: Gf = w[0] * h[vtx[0] as usize]
                    + w[1] * h[vtx[1] as usize]
                    + w[2] * h[vtx[2] as usize];
                let zf = f64::from(full.quarters[b * side + a]) / 4.0;
                let face = Face::from_index(c.face).expect("a face");
                let zm =
                    vd_terrain::artifact::sample_z(&lattice, &artifact, face, L1_RUNG, c.i, c.j)
                        .expect("z");
                let zm = zm.raw() as f64 / unit - f64::from(artifact.sea_m);
                d_coarse.push((zc.to_f64() - zf).abs());
                d_macro.push((zm - zf).abs());
            }
        }
        let t_raster = t.elapsed().as_secs_f64();
        let stats = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            let n = v.len();
            (
                v.iter().sum::<f64>() / n as f64,
                v[n / 2],
                v[n * 9 / 10],
                v[n * 99 / 100],
                v[n - 1],
            )
        };
        let (cm, c50, c90, c99, cmax) = stats(d_coarse);
        let (mm, m50, m90, m99, mmax) = stats(d_macro);
        println!(
            "STANDIN {key:?}: the full solve {t_full:.2} s ({} points); the first stage alone {t_coarse:.2} s ({} points, passes {passes:?}), its raster {t_raster:.2} s",
            full.points,
            d.dirs.len()
        );
        println!(
            "STANDIN {key:?}: off the finished ground — the first stage mean {cm:.0} m, median {c50:.0}, 90th {c90:.0}, 99th {c99:.0}, worst {cmax:.0}; today's 8 km ground mean {mm:.0} m, median {m50:.0}, 90th {m90:.0}, 99th {m99:.0}, worst {mmax:.0}"
        );
        return;
    }

    // ★ B2's WAIT, EXPLAINED (`--why-wait F,R,X,Y,Z,N`): the tiles the shard names for the stand's eye (nearest first),
    // the first N of them held, and for the chunk (F, R, X, Y, Z) and each of its parents the fine tiles it reads that
    // are not held.
    if let Some(v) = sarg("--why-wait") {
        use vd_terrain::chunk::ChunkKey;
        use vd_terrain::fine::l1::TileKey;
        let k: Vec<i64> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let body = body.with_sea_m(artifact.sea());
        let drawn = artifact.drawn_range_m(&lattice);
        let body = body
            .with_fine_band_m(drawn.0, drawn.1, artifact.sea_m)
            .unwrap_or(body);
        let at: Vec<f64> = sarg("--eye")
            .unwrap_or_else(|| {
                "4142667.5882631033,-2985007.539965574,-3770764.624794889".to_owned()
            })
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        let r = (at[0] * at[0] + at[1] * at[1] + at[2] * at[2]).sqrt();
        let dir = [at[0] / r, at[1] / r, at[2] / r];
        let ground = vd_terrain::height::height_field_m(&body, &artifact, dir, 0).expect("ground");
        let relief = body.ground_range_m().1;
        let pixel = vd_core::geometry::drawable_theta_min_rad();
        let chunk_m = 62.0 * f64::from(vd_seed::ladder::cell_m(12));
        let reach = vd_terrain::artifact::fine_reach_m(
            12,
            (ground, ground),
            relief,
            r + 1.6,
            pixel,
            (chunk_m, lattice.node_m() * 8.0),
        )
        .unwrap_or(0.0);
        let named = vd_terrain::artifact::tiles_within_of(&lattice, dir, reach, 8);
        let held: std::collections::BTreeSet<(u8, u32, u32)> =
            named.iter().copied().take(k[5] as usize).collect();
        println!(
            "B2 WHY: reach {:.0} km, {} named, {} held; the first named {:?}",
            reach / 1e3,
            named.len(),
            held.len(),
            &named[..3.min(named.len())]
        );
        let book =
            vd_terrain::fine::read::FineBook::new(&body, &lattice, artifact.sea_m).expect("a book");
        let key = ChunkKey {
            face: Face::from_index(k[0] as u8).expect("a face"),
            rung: k[1] as u8,
            x: k[2] as i32,
            y: k[3] as i32,
            z: k[4] as i32,
        };
        for c in std::iter::once(key).chain(vd_client::chunks::parent_keys(&body, key)) {
            let needs = book.tiles_of_chunk(c.face, c.rung, c.x, c.y);
            let missing: Vec<TileKey> = needs
                .iter()
                .copied()
                .filter(|t| !held.contains(&(t.face.index(), t.tx as u32, t.ty as u32)))
                .collect();
            println!(
                "  {c:?}: reads {} fine tiles, {} not held {:?}",
                needs.len(),
                missing.len(),
                &missing[..missing.len().min(6)]
            );
        }
        return;
    }

    // ★ B2's WINDOW STAND (`--stand-pose`): by a stated rule — among the land tiles where the star stands 15–40° up at
    // the cluster's boot (tick 0), the one with the most relief over its 8 × 8 nodes — its 5 × 5 raw block solved, the
    // centre finished, and the spawn pose standing 0.5 m over the FINE ground at the tile's centre, facing the tile's
    // highest node.
    if args.iter().any(|a| a == "--stand-pose") {
        use vd_core::glam::{DMat3, DQuat, DVec3};
        use vd_terrain::artifact::ZField;
        use vd_terrain::fine::l1::TileKey;
        use vd_terrain::fine::stitch::Field;
        let seed = vd_bins::DEV.universe_seed;
        let orbit = vd_bins::home_orbit(seed).expect("the home planet's orbit");
        let sun = -vd_physics::celestial::orbital_state(&orbit, 0.0)
            .position
            .normalize();
        let planet_seed = vd_bins::home_body(seed).expect("the home planet").seed();
        let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
        let dir_of = |node: u32| {
            let d = lattice.direction(node);
            DVec3::new(
                d[0].raw() as f64 / unit,
                d[1].raw() as f64 / unit,
                d[2].raw() as f64 / unit,
            )
            .normalize()
        };
        let edge = lattice.edge as i32;
        let mut best: Option<(i32, TileKey)> = None;
        for f in 0..6u8 {
            let face = Face::from_index(f).expect("a face");
            // At least ten tiles from every face edge: the reach (about eight tiles) stays on the face until the one-chart
            // cut across a face's edge is built (B2, owed).
            for ty in 10..(edge >> 3) - 10 {
                for tx in 10..(edge >> 3) - 10 {
                    let centre = dir_of(lattice.index(face, tx * 8 + 4, ty * 8 + 4));
                    let elevation = centre.dot(sun).asin().to_degrees();
                    if !(15.0..=40.0).contains(&elevation) {
                        continue;
                    }
                    let zs: Vec<i32> = (0..64)
                        .filter_map(|q| {
                            artifact.z_m(lattice.index(face, tx * 8 + (q & 7), ty * 8 + (q >> 3)))
                        })
                        .map(i32::from)
                        .collect();
                    let (lo, hi) = (
                        *zs.iter().min().unwrap_or(&0),
                        *zs.iter().max().unwrap_or(&0),
                    );
                    if lo <= i32::from(artifact.sea_m) {
                        continue;
                    }
                    if best.is_none_or(|(r, _)| hi - lo > r) {
                        best = Some((hi - lo, TileKey { face, tx, ty }));
                    }
                }
            }
        }
        let (relief, key) = best.expect("a lit land tile");
        let centre = dir_of(lattice.index(key.face, key.tx * 8 + 4, key.ty * 8 + 4));
        println!(
            "B2 THE STAND: tile {key:?}, relief {relief} m over its nodes, the star {:.1}° up at boot",
            centre.dot(sun).asin().to_degrees()
        );
        // The 5 × 5 raw block, then the 3 × 3 finished around the centre (the centre's own stencil reads one more).
        let shift = vd_terrain::fine::l1::cells_per_node_shift(&body, &lattice, 7).expect("nests");
        let keys: Vec<TileKey> = (-2..=2)
            .flat_map(|dy| {
                (-2..=2).map(move |dx| TileKey {
                    face: key.face,
                    tx: key.tx + dx,
                    ty: key.ty + dy,
                })
            })
            .collect();
        let t = Instant::now();
        let raw: Vec<Field> = std::thread::scope(|scope| {
            let hs: Vec<_> = keys
                .iter()
                .map(|&k| {
                    let (body, lattice, artifact) = (&body, &lattice, &artifact);
                    scope.spawn(move || {
                        let t = vd_terrain::fine::l1::solve_l1(body, lattice, artifact, k)
                            .expect("solves");
                        vd_terrain::fine::tile::raw_field(k, t, shift)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("a tile")).collect()
        });
        let mut book =
            vd_terrain::fine::read::FineBook::new(&body, &lattice, artifact.sea_m).expect("a book");
        let at = |dx: i32, dy: i32| &raw[((dy + 2) * 5 + dx + 2) as usize];
        for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
            let k = TileKey {
                face: key.face,
                tx: key.tx + dx,
                ty: key.ty + dy,
            };
            let n7 = (lattice.edge << shift) as i32;
            let block: [Option<(TileKey, &Field)>; 9] = std::array::from_fn(|s| {
                let (sx, sy) = (s as i32 % 3 - 1, s as i32 / 3 - 1);
                Some((
                    TileKey {
                        face: key.face,
                        tx: k.tx + sx,
                        ty: k.ty + sy,
                    },
                    at(dx + sx, dy + sy),
                ))
            });
            let levels = vd_terrain::fine::tile::finish(block, shift, n7).expect("finishes");
            assert!(book.insert(k, levels));
        }
        println!(
            "B2 THE STAND: 25 raw tiles and 9 finished in {:.1} s",
            t.elapsed().as_secs_f64()
        );
        // ★ THE FLATS (the owner in the window: "Not sure what flat surface here is"): over the 3 × 3 finished tiles, the
        // share of 128 m cells at or under the sea, the share that is flat (the steepest step to a neighbour under 1 %),
        // and how many of the flat ones stand at or under the sea; and a map with the under-sea cells marked.
        {
            let side = 1usize << book.side_shift();
            let mut cells = 0usize;
            let (mut under, mut flat, mut flat_under) = (0usize, 0usize, 0usize);
            let mut z_all = vec![0f64; 9 * side * side];
            let w3 = 3 * side;
            for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
                let k = TileKey {
                    face: key.face,
                    tx: key.tx + dx,
                    ty: key.ty + dy,
                };
                for y in 0..side {
                    for x in 0..side {
                        let gx = (i64::from(k.tx) << book.side_shift()) + x as i64;
                        let gy = (i64::from(k.ty) << book.side_shift()) + y as i64;
                        let q = |gx: i64, gy: i64| match book.z(k.face, 7, 7, gx as i32, gy as i32)
                        {
                            vd_terrain::fine::read::FineZ::Held(g) => Some(
                                g.raw() as f64 / (128.0 * (1u64 << 28) as f64)
                                    - f64::from(artifact.sea_m),
                            ),
                            _ => None,
                        };
                        let Some(h) = q(gx, gy) else { continue };
                        cells += 1;
                        let steep = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                            .iter()
                            .filter_map(|(a, b)| q(gx + a, gy + b))
                            .map(|n| (n - h).abs() / 128.0)
                            .fold(0.0f64, f64::max);
                        under += usize::from(h <= 0.5);
                        flat += usize::from(steep < 0.01);
                        flat_under += usize::from((steep < 0.01) & (h <= 0.5));
                        z_all[((dy + 1) as usize * side + y) * w3 + (dx + 1) as usize * side + x] =
                            h;
                    }
                }
            }
            println!(
                "B2 THE FLATS: {cells} cells of 128 m; at or under the sea {:.1} %; flat (under a 1 % step) {:.1} %, of which at or under the sea {:.1} %",
                100.0 * under as f64 / cells as f64,
                100.0 * flat as f64 / cells as f64,
                100.0 * flat_under as f64 / flat.max(1) as f64
            );
            if let Some(dir) = hillshade_dir.as_deref() {
                let _ = std::fs::create_dir_all(dir);
                let path_png = format!("{dir}/stand_flats.png");
                hillshade_rect(&z_all, w3, w3, 128.0, &path_png);
                // Mark the cells at or under the sea in blue over the shading.
                let mut img = image::open(&path_png).expect("the map").to_rgb8();
                for (k, h) in z_all.iter().enumerate() {
                    if *h <= 0.5 {
                        let p = img.get_pixel_mut((k % w3) as u32, (k / w3) as u32);
                        *p = image::Rgb([p[0] / 3, p[1] / 2, 200]);
                    }
                }
                img.save(&path_png).expect("the map writes");
                println!("B2 THE FLATS: the map {path_png}");
            }
        }
        // The fine ground under the centre, and the tile's highest node to face.
        let face_of = vd_seed::bend::face_of(centre.to_array());
        let (u, v) = vd_seed::bend::face_coords(face_of, centre.to_array());
        let n0 = body.ladder().cells_per_edge(0);
        let (i, j) = (
            vd_seed::ladder::index_of(vd_seed::bend::unbend(u), n0),
            vd_seed::ladder::index_of(vd_seed::bend::unbend(v), n0),
        );
        let z = match book.z(face_of, 0, 0, i, j) {
            vd_terrain::fine::read::FineZ::Held(g) => {
                g.raw() as f64 / (128.0 * (1u64 << 28) as f64)
            }
            other => panic!("{other:?}"),
        };
        let surface = body.radius_m() + z;
        let high = (0..64)
            .max_by_key(|q| {
                artifact
                    .z_m(lattice.index(key.face, key.tx * 8 + (q & 7), key.ty * 8 + (q >> 3)))
                    .unwrap_or(i16::MIN)
            })
            .expect("a node");
        let toward =
            dir_of(lattice.index(key.face, key.tx * 8 + (high & 7), key.ty * 8 + (high >> 3)));
        let forward = (toward - centre * toward.dot(centre)).normalize();
        let up = (centre - forward * forward.dot(centre)).normalize();
        let right = forward.cross(up).normalize();
        let orient = DQuat::from_mat3(&DMat3::from_cols(right, up, -forward)).normalize();
        let at_m = centre * (surface + 0.5);
        println!(
            "B2 THE STAND: the fine ground {:.1} m over the radius (the macro's {:.1} m); the spawn:\n1000=Planet({planet_seed}):{},{},{}@{},{},{},{}",
            z,
            vd_terrain::height::height_field_m(&body, &artifact, centre.to_array(), 0)
                .map_or(f64::NAN, |g| g - body.radius_m()),
            at_m.x,
            at_m.y,
            at_m.z,
            orient.x,
            orient.y,
            orient.z,
            orient.w
        );
        // ★ A HULL'S BERTH ON THE FINE GROUND (the owner: "No hull" — a berth 4 m over the start's height stood inside
        // the slope ahead): around the stand, the flattest 40 × 12 m footprint at 30, 45 or 60 m (sixteen bearings),
        // its highest fine ground under the footprint, the hull's centre half its height (3 m) and half a metre over it,
        // its nose pointing away from the stand; and the stand turned to face it.
        let ground_at = |d: DVec3| -> f64 {
            let d = d.normalize();
            let f = vd_seed::bend::face_of(d.to_array());
            let (u, v) = vd_seed::bend::face_coords(f, d.to_array());
            let (i, j) = (
                vd_seed::ladder::index_of(vd_seed::bend::unbend(u), n0),
                vd_seed::ladder::index_of(vd_seed::bend::unbend(v), n0),
            );
            match book.z(f, 0, 0, i, j) {
                vd_terrain::fine::read::FineZ::Held(g) => {
                    g.raw() as f64 / (128.0 * (1u64 << 28) as f64)
                }
                other => panic!("{other:?}"),
            }
        };
        let r_m = body.radius_m();
        let mut best: Option<(f64, f64, f64, f64, f64)> = None; // (spread, dist, bearing, highest, lowest)
        for b in 0..16 {
            let bearing = f64::from(b) * std::f64::consts::TAU / 16.0;
            let along = forward * bearing.cos() + right * bearing.sin();
            let side_v = up.cross(along).normalize();
            for dist in [30.0, 45.0, 60.0] {
                let (mut hi, mut lo) = (f64::MIN, f64::MAX);
                for a in (-20..=20).step_by(4) {
                    for c in (-6..=6).step_by(3) {
                        let p = centre * surface
                            + along * (dist + f64::from(a))
                            + side_v * f64::from(c);
                        let z = ground_at(p);
                        hi = hi.max(z);
                        lo = lo.min(z);
                    }
                }
                let spread = hi - lo;
                if best.is_none_or(|bb| spread < bb.0) {
                    best = Some((spread, dist, bearing, hi, lo));
                }
            }
        }
        let (spread, dist, bearing, hi, _lo) = best.expect("a berth");
        let along = forward * bearing.cos() + right * bearing.sin();
        let at = (centre * surface + along * dist).normalize();
        let berth = at * (r_m + hi + 3.5);
        let b_up = at;
        let b_fwd = (along - b_up * along.dot(b_up)).normalize();
        let b_right = b_fwd.cross(b_up).normalize();
        let b_orient = DQuat::from_mat3(&DMat3::from_cols(b_right, b_up, -b_fwd)).normalize();
        let s_fwd = (along - up * along.dot(up)).normalize();
        let s_right = s_fwd.cross(up).normalize();
        let s_orient = DQuat::from_mat3(&DMat3::from_cols(s_right, up, -s_fwd)).normalize();
        println!(
            "B2 THE BERTH: bearing {:.1}° off the stand's facing, {dist:.0} m out, the ground under the footprint spreads {spread:.1} m, its top {:.1} m over the stand's ground\n  --berth-x-m {} --berth-y-m {} --berth-z-m {} --berth-facing-x {} --berth-facing-y {} --berth-facing-z {} --berth-facing-w {}\n  the stand facing the hull: 1000=Planet({planet_seed}):{},{},{}@{},{},{},{}",
            bearing.to_degrees(),
            hi - z,
            berth.x,
            berth.y,
            berth.z,
            b_orient.x,
            b_orient.y,
            b_orient.z,
            b_orient.w,
            at_m.x,
            at_m.y,
            at_m.z,
            s_orient.x,
            s_orient.y,
            s_orient.z,
            s_orient.w
        );
        return;
    }

    // ★ B2's PROVIDER HEADLESS (`--fine-work F,TX,TY`): the shard's own fine provider over the stored artifact, seven
    // workers and a store in memory, an occupant standing 2 m over the tile's centre: how fast the tile under the eye
    // and its neighbours are finished, and what one tick of the provider costs the shard.
    if let Some(v) = sarg("--fine-work") {
        use vd_sim::stub::fine_ship::FineTiles;
        let k: Vec<i32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let face = Face::from_index(k[0] as u8).expect("a face");
        let body = body.with_sea_m(artifact.sea());
        let threads = std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1));
        let artifact = std::sync::Arc::new(artifact);
        let work = vd_bins::fine_source::FineWork::new(
            vd_core::pose::RealmId::Planet(1),
            body,
            artifact.clone(),
            threads,
        )
        .expect("the level applies");
        let node = lattice.index(face, k[1] * 8 + 4, k[2] * 8 + 4);
        let d = lattice.direction(node);
        let unit = (1u64 << vd_recipe::bend::DIR_BITS) as f64;
        let dir = [
            d[0].raw() as f64 / unit,
            d[1].raw() as f64 / unit,
            d[2].raw() as f64 / unit,
        ];
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let dir = [dir[0] / len, dir[1] / len, dir[2] / len];
        let ground = vd_terrain::height::height_field_m(&body, &*artifact, dir, 0).expect("ground");
        let mut store = vd_sim::io::mem::MemStore::default();
        let t0 = Instant::now();
        let (mut worst_tick, mut ticks, mut named) = (0.0f64, 0usize, 0usize);
        let mut first_ring_at = None;
        let mut last_print = 0u64;
        while t0.elapsed().as_secs() < 240 {
            named = work.want(dir, ground + 2.0).len();
            let t = Instant::now();
            work.tick(Some(&mut store));
            worst_tick = worst_tick.max(t.elapsed().as_secs_f64());
            ticks += 1;
            let ring = (-1..=1)
                .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                .all(|(dx, dy)| {
                    work.parts(
                        (k[0] as u8, (k[1] + dx) as u32, (k[2] + dy) as u32),
                        0,
                        vd_sim::stub::fine_ship::NONE_HELD,
                    )
                    .is_some()
                });
            if ring && first_ring_at.is_none() {
                first_ring_at = Some(t0.elapsed().as_secs_f64());
                let (raw, fin, fly) = work.progress();
                println!(
                    "B2 THE PROVIDER: the 3 × 3 ring under the eye finished at {:.1} s (raw {raw}, finished {fin}, in flight {fly})",
                    t0.elapsed().as_secs_f64()
                );
            }
            if t0.elapsed().as_secs() >= last_print + 20 {
                last_print = t0.elapsed().as_secs();
                let (raw, fin, fly) = work.progress();
                println!(
                    "  {:.0} s: {named} named, raw {raw}, finished {fin}, in flight {fly}, the worst tick {:.1} ms",
                    t0.elapsed().as_secs_f64(),
                    worst_tick * 1e3
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let (raw, fin, fly) = work.progress();
        let parts = work
            .parts(
                (k[0] as u8, k[1] as u32, k[2] as u32),
                0,
                vd_sim::stub::fine_ship::NONE_HELD,
            )
            .map_or(0, |p| p.len());
        // ★ FAR COPIES FIRST (ruling B86 item 3): the bytes a session logging in at the stand receives — each named tile
        // from the finest level its view reads — against every tile whole; and how many tiles each level.
        {
            let want = work.want(dir, ground + 2.0);
            let bytes = |t: vd_sim::stub::fine_ship::FineName, lo: u8| {
                work.parts(t, lo, vd_sim::stub::fine_ship::NONE_HELD)
                    .map_or(0, |p| p.iter().map(|b| b.len()).sum::<usize>())
            };
            let (mut far, mut whole) = (0usize, 0usize);
            let mut by_level = [0usize; 6];
            for &(t, need) in &want {
                far += bytes(t, need);
                whole += bytes(t, 0);
                by_level[usize::from(need)] += 1;
            }
            println!(
                "B86 FAR COPIES FIRST: {} tiles named, by the finest level read {by_level:?}; a login receives {:.1} MB where every tile whole is {:.1} MB",
                want.len(),
                far as f64 / 1e6,
                whole as f64 / 1e6
            );
        }
        println!(
            "B2 THE PROVIDER: {threads} workers, {named} tiles named at the stand; after {:.0} s raw {raw}, finished {fin}, in flight {fly}; the stand's tile in {parts} parts; {ticks} ticks, the worst {:.1} ms; the 3 × 3 ring at {:?} s; the store {:.1} MB in {} rows",
            t0.elapsed().as_secs_f64(),
            worst_tick * 1e3,
            first_ring_at,
            vd_sim::io::Store::scan(&store, &[])
                .iter()
                .map(|(_, v)| v.len())
                .sum::<usize>() as f64
                / 1e6,
            vd_sim::io::Store::scan(&store, &[]).len()
        );
        return;
    }
    // ★ B2's SEA RULE, ITS PRICE (`--fine-sea F,TY`): the first tile of row TY on face F whose 12 × 12 domain nodes are
    // all under the sea, solved by the general path (every point an outlet), timed.
    if let Some(v) = sarg("--fine-sea") {
        use vd_terrain::artifact::ZField;
        let k: Vec<i32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let face = Face::from_index(k[0] as u8).expect("a face");
        let edge = lattice.edge as i32;
        let sea_tile = (1..(edge >> 3) - 1).find(|&tx| {
            (0..144).all(|q| {
                let (i, j) = (tx * 8 - 2 + (q % 12), k[1] * 8 - 2 + (q / 12));
                artifact
                    .z_m(lattice.index(face, i.clamp(0, edge - 1), j.clamp(0, edge - 1)))
                    .is_some_and(|z| z <= artifact.sea_m)
            })
        });
        let Some(tx) = sea_tile else {
            println!(
                "B2 THE SEA RULE: no all-sea tile in row {} of face {}",
                k[1], k[0]
            );
            return;
        };
        let key = vd_terrain::fine::l1::TileKey { face, tx, ty: k[1] };
        let t = Instant::now();
        let tile = vd_terrain::fine::l1::solve_l1(&body, &lattice, &artifact, key).expect("solves");
        println!(
            "B2 THE SEA RULE: the all-sea tile {key:?} by the general path in {:.2} s ({} points, passes {:?})",
            t.elapsed().as_secs_f64(),
            tile.points,
            tile.passes
        );
        return;
    }
    // ★ B1 (e) CHECKED ON THE BENCH (`--fine-pair F,TX,TY`): the generator's tile and its east neighbour solved at once,
    // stitched by the generator's own cut, the slope break across the cut against the tiles' inside, and the map.
    if let Some(v) = sarg("--fine-pair") {
        use vd_terrain::fine::stitch::{Field, stitch_columns};
        let k: Vec<i32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let face = Face::from_index(k[0] as u8).expect("a face");
        let keys = [
            vd_terrain::fine::l1::TileKey {
                face,
                tx: k[1],
                ty: k[2],
            },
            vd_terrain::fine::l1::TileKey {
                face,
                tx: k[1] + 1,
                ty: k[2],
            },
        ];
        let t = Instant::now();
        let tiles: Vec<_> = std::thread::scope(|scope| {
            let hs: Vec<_> = keys
                .iter()
                .map(|&key| {
                    let (body, lattice, artifact) = (&body, &lattice, &artifact);
                    scope.spawn(move || {
                        vd_terrain::fine::l1::solve_l1(body, lattice, artifact, key)
                            .expect("solves")
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("a tile")).collect()
        });
        let shift =
            vd_terrain::fine::l1::cells_per_node_shift(&body, &lattice, 7).expect("nests") as i64;
        let node = 1i64 << shift;
        // Each raster covers its tile and half its buffer: one node beyond each edge.
        let field = |key: &vd_terrain::fine::l1::TileKey, t: &vd_terrain::fine::l1::L1Tile| Field {
            gx0: (i64::from(key.tx) * 8 - 1) * node,
            gy0: (i64::from(key.ty) * 8 - 1) * node,
            w: t.side,
            h: t.side,
            v: t.quarters.clone(),
        };
        let (fa, fb) = (field(&keys[0], &tiles[0]), field(&keys[1], &tiles[1]));
        let ge = (i64::from(k[1]) + 1) * 8 * node;
        let (s, path) = stitch_columns(&fa, &fb, ge, node);
        println!(
            "B1 (e): two tiles at once in {:.1} s; the cut wanders {}–{} columns from the edge",
            t.elapsed().as_secs_f64(),
            path.iter().min().unwrap_or(&0) - ge,
            path.iter().max().unwrap_or(&0) - ge
        );
        // The slope break (the second difference across 4 cells, per cell length) across the cut and inside the west tile.
        let brk = |gx: i64, gy: i64| -> f64 {
            let z = |x: i64| f64::from(s.at(x, gy)) / 4.0;
            (((z(gx + 3) - z(gx + 1)) - (z(gx - 1) - z(gx - 3))) / 256.0).abs()
        };
        let pct = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() * 99 / 100]
        };
        let rows: Vec<i64> = (0..path.len() as i64).collect();
        let on_cut = pct(rows
            .iter()
            .map(|&y| brk(path[y as usize], s.gy0 + y))
            .collect());
        let inside = pct(rows
            .iter()
            .map(|&y| brk(ge - 3 * node, s.gy0 + y))
            .collect());
        println!(
            "B1 (e): the slope break, 99th: across the cut {on_cut:.3}, inside the west tile {inside:.3}"
        );
        if let Some(dir) = hillshade_dir.as_deref() {
            let _ = std::fs::create_dir_all(dir);
            let z: Vec<f64> = s.v.iter().map(|&q| f64::from(q) / 4.0).collect();
            let path_png = format!("{dir}/fine_pair_{}_{}_{}.png", k[0], k[1], k[2]);
            hillshade_rect(&z, s.w, s.h, 128.0, &path_png);
            println!("B1 (e): the map: {path_png}");
        }
        return;
    }
    // ★ B1 (d) CHECKED ON THE BENCH (`--fine-l1 F,TX,TY`): the generator's own 128 m solve of one tile — the time, the
    // passes, the straight-line and lean readings, and its map.
    if let Some(v) = sarg("--fine-l1") {
        let k: Vec<i32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        let key = vd_terrain::fine::l1::TileKey {
            face: Face::from_index(k[0] as u8).expect("a face"),
            tx: k[1],
            ty: k[2],
        };
        if args.iter().any(|a| a == "--fine-steps") {
            use vd_terrain::fine::points::{Level, domain_points};
            for rung in [9u8, 7] {
                let n_r = body.ladder().cells_per_edge(rung);
                let shift = vd_terrain::fine::l1::cells_per_node_shift(&body, &lattice, rung)
                    .expect("nests");
                let level = Level::new(body.seed(), rung, n_r).expect("level");
                let (ni0, nj0) = (k[1] * 8 - 2, k[2] * 8 - 2);
                let t = Instant::now();
                let pts = domain_points(
                    &level,
                    key.face,
                    ni0 << shift,
                    nj0 << shift,
                    12 << shift,
                    12 << shift,
                );
                let t_p = t.elapsed().as_secs_f64();
                let dirs: Vec<_> = pts.iter().map(|p| p.dir).collect();
                let t = Instant::now();
                let g = vd_terrain::fine::delaunay::triangulate(&dirs).expect("graph");
                let t_g = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let _ = vd_terrain::fine::geometry::geometry(
                    &dirs,
                    &g,
                    vd_terrain::Gf::from_f64(body.radius_m()),
                );
                let t_m = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mut acc = 0i64;
                for p in &pts {
                    let face = Face::from_index(p.cell.face).expect("face");
                    acc ^= vd_terrain::artifact::sample_z(
                        &lattice, &artifact, face, rung, p.cell.i, p.cell.j,
                    )
                    .expect("z")
                    .raw();
                }
                let t_z = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mut acc2 = 0i64;
                for p in pts.iter().take(2_000) {
                    let face = Face::from_index(p.cell.face).expect("face");
                    for dj in -3..=3i64 {
                        for di in -3..=3i64 {
                            let m = vd_terrain::artifact::node_at(
                                &lattice,
                                face,
                                i64::from(p.cell.i >> shift) + di,
                                i64::from(p.cell.j >> shift) + dj,
                            );
                            acc2 ^= lattice.direction(m)[0].raw();
                        }
                    }
                }
                let t_w = t.elapsed().as_secs_f64() * pts.len() as f64 / 2_000.0;
                println!(
                    "steps rung {rung}: {} points {t_p:.2} s, the graph {t_g:.2} s, the geometry {t_m:.2} s, the smooth surface {t_z:.2} s, the 49 node directions a point {t_w:.2} s (scaled) [{acc} {acc2}]",
                    pts.len()
                );
            }
            return;
        }
        if args.iter().any(|a| a == "--fine-debug") {
            use vd_terrain::fine::l1::{
                COARSE_ROUNDS, COARSE_RUNG, L1_RUNG, domain, fine_rounds, rounds,
            };
            let t = Instant::now();
            let coarse = domain(&body, &lattice, &artifact, COARSE_RUNG, key).expect("coarse");
            let t_cd = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut uc = coarse.ref_u0.clone();
            let mut hc = coarse.base.clone();
            let pc = rounds(&coarse, &mut uc, &mut hc, COARSE_ROUNDS);
            let t_cr = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let fine = domain(&body, &lattice, &artifact, L1_RUNG, key).expect("fine");
            let t_fd = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut u: Vec<_> = fine
                .refs
                .iter()
                .zip(&fine.ref_u0)
                .map(|(m, &u0)| coarse.refs.binary_search(m).map_or(u0, |i| uc[i]))
                .collect();
            let mut h = fine.base.clone();
            // The game's method for this level since ruling B92: the drainage built once, the rounds by the settle law.
            let pf = vec![fine_rounds(&fine, &mut u, &mut h)];
            let t_fr = t.elapsed().as_secs_f64();
            println!(
                "debug: the coarse domain {t_cd:.2} s ({} points), its rounds {t_cr:.2} s {pc:?}; the fine domain {t_fd:.2} s ({} points), its rounds (cold start here) {t_fr:.2} s {pf:?}",
                coarse.dirs.len(),
                fine.dirs.len()
            );
            return;
        }
        let t = Instant::now();
        let tile = vd_terrain::fine::l1::solve_l1(&body, &lattice, &artifact, key)
            .expect("the tile solves");
        let dt = t.elapsed().as_secs_f64();
        let side = tile.side;
        let z: Vec<f64> = tile.quarters.iter().map(|&q| f64::from(q) / 4.0).collect();
        let reg = r2::plain_region(side, 128.0, &z);
        let (straight, ns) = r2::straight_share(&reg, &z, 0.05, 16);
        let (lean, nl) = r2::lean(&reg, &z, 0.05);
        let (lo, hi) = z
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        println!(
            "B1 (d): the generator's 128 m tile {k:?}: {} points, passes {:?}, {dt:.1} s; {side} × {side} heights {lo:.0}–{hi:.0} m; the straight share {:.1} % of {ns}, the lean {:.1} % of {nl}",
            tile.points,
            tile.passes,
            100.0 * straight,
            100.0 * lean
        );
        {
            let t = Instant::now();
            let bytes =
                vd_terrain::fine::coder::encode(&tile.quarters, side, side).expect("encodes");
            let back = vd_terrain::fine::coder::decode(&bytes, side, side).expect("decodes");
            println!(
                "B1 (f): the generator's coder: {} KB ({:.2} bytes a sample), decoded back exactly: {} ({:.2} s)",
                bytes.len() / 1024,
                bytes.len() as f64 / (side * side) as f64,
                back == tile.quarters,
                t.elapsed().as_secs_f64()
            );
        }
        if let Some(dir) = hillshade_dir.as_deref() {
            let _ = std::fs::create_dir_all(dir);
            let path = format!("{dir}/fine_l1_{}_{}_{}.png", k[0], k[1], k[2]);
            hillshade_rect(&z, side, side, 128.0, &path);
            println!("B1 (d): the map: {path}");
        }
        return;
    }
    // ★ B1 (a) CHECKED ON THE BENCH (`--fine-points`): the generator's own hashed Poisson-disk set over a 128 m tile's
    // domain on the home planet — the count, the density, the closest pair against the radius, the time — and a domain
    // that crosses a face edge, against each crossed cell's own kept set.
    if args.iter().any(|a| a == "--fine-points") {
        use vd_terrain::fine::points::{CellId, Level, domain_points};
        let n7 = body.ladder().cells_per_edge(7);
        let level = Level::new(body.seed(), 7, n7).expect("the 128 m level");
        let t = Instant::now();
        let pts = domain_points(&level, Face::NegY, 23_000, 70_000, 768, 768);
        let dt = t.elapsed().as_secs_f64();
        let r2 = level.radius2(vd_terrain::fine::points::CellId {
            face: Face::NegY.index(),
            i: 23_384,
            j: 70_384,
        });
        // The closest pair, by a bucket of cells (pairs only within the 3 × 3 cells around).
        let mut by_cell: std::collections::BTreeMap<(i32, i32), Vec<usize>> =
            std::collections::BTreeMap::new();
        for (k, p) in pts.iter().enumerate() {
            by_cell.entry((p.cell.i, p.cell.j)).or_default().push(k);
        }
        let mut closest = i128::MAX;
        for (k, p) in pts.iter().enumerate() {
            for dj in -1..=1 {
                for di in -1..=1 {
                    if let Some(list) = by_cell.get(&(p.cell.i + di, p.cell.j + dj)) {
                        for &m in list {
                            if m != k {
                                closest = closest.min(p.chord2(&pts[m]));
                            }
                        }
                    }
                }
            }
        }
        println!(
            "B1 (a): {} points over 768 × 768 cells of the 128 m level ({:.4} a cell) in {dt:.2} s; the closest pair {:.3} of the radius",
            pts.len(),
            pts.len() as f64 / (768.0 * 768.0),
            ((closest as f64) / (r2 as f64)).sqrt()
        );
        // A domain across the face edge: every crossed cell's points equal its own kept set.
        let t = Instant::now();
        let edge = domain_points(&level, Face::NegY, n7 as i32 - 64, 70_000, 128, 32);
        let mut crossed: Vec<CellId> = edge
            .iter()
            .filter(|p| p.cell.face != Face::NegY.index())
            .map(|p| p.cell)
            .collect();
        crossed.sort_unstable();
        crossed.dedup();
        let same = crossed.iter().all(|&c| {
            edge.iter()
                .copied()
                .filter(|p| p.cell == c)
                .collect::<Vec<_>>()
                == level.kept(c)
        });
        println!(
            "B1 (a): across the face edge {} cells on the next face, each its own kept set: {same} ({:.2} s)",
            crossed.len(),
            t.elapsed().as_secs_f64()
        );
        // ★ B1 (b): the spherical Delaunay of the same points.
        use vd_terrain::fine::delaunay::{in_circle, triangulate};
        let dirs: Vec<_> = pts.iter().map(|p| p.dir).collect();
        let t = Instant::now();
        let g = triangulate(&dirs).expect("a graph");
        let dt = t.elapsed().as_secs_f64();
        let degs: Vec<usize> = (0..dirs.len()).map(|i| g.neighbours(i).len()).collect();
        let (dmin, dmax) = (
            degs.iter().copied().min().unwrap_or(0),
            degs.iter().copied().max().unwrap_or(0),
        );
        let dmean = degs.iter().sum::<usize>() as f64 / degs.len() as f64;
        // The Delaunay property: for each triangle, no vertex of a neighbouring triangle lies inside its circle.
        let mut edge_tri: std::collections::BTreeMap<(u32, u32), u32> =
            std::collections::BTreeMap::new();
        for (ti, tr) in g.tris.iter().enumerate() {
            for e in 0..3 {
                edge_tri.insert((tr[e], tr[(e + 1) % 3]), ti as u32);
            }
        }
        let mut bad = 0usize;
        for tr in &g.tris {
            for e in 0..3 {
                let (x, y) = (tr[e], tr[(e + 1) % 3]);
                if let Some(&o) = edge_tri.get(&(y, x)) {
                    let ov = g.tris[o as usize];
                    let d = ov
                        .iter()
                        .copied()
                        .find(|&v| v != x && v != y)
                        .expect("the opposite vertex");
                    if in_circle(
                        &dirs[tr[0] as usize],
                        &dirs[tr[1] as usize],
                        &dirs[tr[2] as usize],
                        &dirs[d as usize],
                    ) > 0
                    {
                        bad += 1;
                    }
                }
            }
        }
        println!(
            "B1 (b): the Delaunay of {} points in {dt:.2} s: {} triangles ({:.3} a point), neighbours {dmin}–{dmax} (mean {dmean:.3}); {bad} edges fail the circle test",
            dirs.len(),
            g.tris.len(),
            g.tris.len() as f64 / dirs.len() as f64
        );
        // Two overlapping domains: the points well inside both have the same neighbours.
        let a = domain_points(&level, Face::NegY, 23_000, 70_000, 256, 256);
        let b = domain_points(&level, Face::NegY, 23_128, 70_000, 256, 256);
        let ga = triangulate(&a.iter().map(|p| p.dir).collect::<Vec<_>>()).expect("a");
        let gb = triangulate(&b.iter().map(|p| p.dir).collect::<Vec<_>>()).expect("b");
        let key = |p: &vd_terrain::fine::points::Candidate| (p.cell, p.k);
        let index_b: std::collections::BTreeMap<_, usize> =
            b.iter().enumerate().map(|(i, p)| (key(p), i)).collect();
        let (mut checked, mut differ) = (0usize, 0usize);
        for (i, p) in a.iter().enumerate() {
            // Well inside both: 16 cells from every edge of the overlap (cells 23 144..23 240, rows 70 016..70 240).
            if p.cell.i < 23_144 || p.cell.i >= 23_240 || p.cell.j < 70_016 || p.cell.j >= 70_240 {
                continue;
            }
            let Some(&j) = index_b.get(&key(p)) else {
                continue;
            };
            checked += 1;
            let mut na: Vec<_> = ga
                .neighbours(i)
                .iter()
                .map(|&m| key(&a[m as usize]))
                .collect();
            let mut nb: Vec<_> = gb
                .neighbours(j)
                .iter()
                .map(|&m| key(&b[m as usize]))
                .collect();
            na.sort();
            nb.sort();
            differ += usize::from(na != nb);
        }
        println!(
            "B1 (b): two overlapping domains: {checked} points well inside both, {differ} with other neighbours"
        );
        return;
    }
    // ★ THE PLAIN STAND (B0 step 4, a stated rule on the stored world): among 20 × 20-node windows all of land with a mean
    // height of 200–800 m over the sea and no node rising faster than 0.05 mm a year (the present-rise map, `--rise-cache`),
    // the window of the least relief (its nodes' 90th less their 10th percentile); candidates every 8th node of each face.
    if args.iter().any(|a| a == "--find-plain") {
        let cache = sarg("--rise-cache").expect("--rise-cache FILE (the present-rise map)");
        let bytes = std::fs::read(&cache).expect("the rise map reads");
        let rise: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|w| f64::from_le_bytes(w.try_into().expect("eight bytes")))
            .collect();
        let sea = f64::from(artifact.sea_m);
        let edge = lattice.edge as i64;
        let mut all: Vec<(f64, u8, i64, i64, f64)> = Vec::new();
        for face_i in 0..6u8 {
            let face = Face::from_index(face_i).expect("a face");
            for j0 in (0..edge - 20).step_by(8) {
                for i0 in (0..edge - 20).step_by(8) {
                    let mut zs = Vec::with_capacity(400);
                    let mut ok = true;
                    for dj in 0..20 {
                        for di in 0..20 {
                            let k = node_at(&lattice, face, i0 + di, j0 + dj) as usize;
                            let z = f64::from(artifact.rows[k].z_m) - sea;
                            if z <= 0.0 || rise[k] > 0.05e-3 {
                                ok = false;
                                break;
                            }
                            zs.push(z);
                        }
                        if !ok {
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let mean = zs.iter().sum::<f64>() / 400.0;
                    if !(200.0..=800.0).contains(&mean) {
                        continue;
                    }
                    zs.sort_by(f64::total_cmp);
                    let relief = zs[359] - zs[39];
                    all.push((relief, face_i, i0 + 10, j0 + 10, mean));
                }
            }
        }
        // The TYPICAL plain: the window of the median relief (ties by face, then place).
        all.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.3.cmp(&b.3))
                .then(a.2.cmp(&b.2))
        });
        if all.is_empty() {
            println!("THE PLAIN STAND: no window passes the rule");
        } else {
            let pick = |q: usize| all[(all.len() - 1) * q / 100].0;
            let (relief, fc, i, j, mean) = all[all.len() / 2];
            println!(
                "THE PLAIN STAND: {} windows pass, their relief 10th / 50th / 90th {:.0} / {:.0} / {:.0} m; the median window: face {fc}, node ({i}, {j}), mean {mean:.0} m over the sea, relief {relief:.0} m → --lem-at {fc},{i},{j}",
                all.len(),
                pick(10),
                pick(50),
                pick(90)
            );
        }
        return;
    }
    if args.iter().any(|a| a == "--rise-map") {
        let started = Instant::now();
        let rise = present_rise(&body, &lattice, &words);
        println!(
            "  the present rise from the first land and the belts' law ({:.1} s)",
            started.elapsed().as_secs_f64()
        );
        rise_map(&lattice, &artifact, &rise);
        return;
    }

    // ── THE TILE: the one under the belt direction the owner flew (or `--aim`), held off the face's seam. ──
    let aim_arg: Option<[f64; 3]> = sarg("--aim").and_then(|s| {
        let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (v.len() == 3).then(|| [v[0], v[1], v[2]])
    });
    let belt = aim_arg.unwrap_or([0.617_270_f64, -0.437_286, -0.654_033]);
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

    // ★ THE STREAM LAW ON IRREGULAR POINTS ON THE HOME PLANET (`--lem`, with `--artifact`): `--r2-nodes N` nodes a side,
    // centred on the tile; each node's height over the sea and its rain from the row; the fine ground by `lem::solve`,
    // beside today's 8 km field (the smooth Catmull-Rom the window draws under its octaves).
    if args.iter().any(|a| a == "--lem") {
        let cell_m = arg("--r2-cell-m", 128) as f64;
        let spacing_m = arg("--lem-spacing-m", 128) as f64;
        let nw = (arg("--r2-nodes", 12) as i64).min(pick.nodes);
        let inset = (pick.nodes - nw).div_euclid(2);
        // `--lem-at F,I,J`: the region centred on that node (face F, place I, J) instead of the belt tile's middle.
        let at: Option<Vec<i64>> =
            sarg("--lem-at").map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        let (pick, inset) = match at.as_deref() {
            Some([fc, ci, cj]) => (
                TilePick {
                    face: Face::from_index(*fc as u8).expect("a face"),
                    i0: ci - nw / 2,
                    j0: cj - nw / 2,
                    nodes: nw,
                },
                0,
            ),
            _ => (pick, inset),
        };
        let sea = f64::from(artifact.sea_m);
        let (mut node_z, mut node_rain) = (
            vec![0.0f64; (nw * nw) as usize],
            vec![0.0f64; (nw * nw) as usize],
        );
        for j in 0..nw {
            for i in 0..nw {
                let row = artifact.rows[node_at(
                    &lattice,
                    pick.face,
                    pick.i0 + inset + i,
                    pick.j0 + inset + j,
                ) as usize];
                node_z[(j * nw + i) as usize] = f64::from(row.z_m) - sea;
                node_rain[(j * nw + i) as usize] = unlog(row.rain, 4) / 1_000.0;
            }
        }
        let land = node_z.iter().filter(|z| **z > 0.0).count();
        println!(
            "LEM HOME: {nw} × {nw} nodes from ({}, {}), {land} on land, heights {:.0} to {:.0} m over the sea, rain {:.3} to {:.3} m/yr",
            pick.i0 + inset,
            pick.j0 + inset,
            node_z.iter().copied().fold(f64::MAX, f64::min),
            node_z.iter().copied().fold(f64::MIN, f64::max),
            node_rain.iter().copied().fold(f64::MAX, f64::min),
            node_rain.iter().copied().fold(f64::MIN, f64::max)
        );
        let nw = nw as usize;
        let out = lem::solve(&node_z, &node_rain, nw, node_m, spacing_m, cell_m);
        let w = out.w;
        let today: Vec<f64> = (0..w * w)
            .map(|k| {
                lem::smooth_at(
                    &node_z,
                    nw,
                    node_m,
                    ((k % w) as f64 + 0.5) * cell_m,
                    ((k / w) as f64 + 0.5) * cell_m,
                )
            })
            .collect();
        lem_report(
            &out.raster,
            &today,
            "today8km",
            w,
            cell_m,
            hillshade_dir.as_deref(),
            &label,
        );
        // THE REGION'S PRESENT RISE from the solve's own laws (no world change): the fine window's rise is its nodes'.
        let started = Instant::now();
        let rise_all = present_rise(&body, &lattice, &words);
        let node_rise: Vec<f64> = (0..nw * nw)
            .map(|k| {
                let (i, j) = ((k % nw) as i64, (k / nw) as i64);
                rise_all[node_at(
                    &lattice,
                    pick.face,
                    pick.i0 + inset + i,
                    pick.j0 + inset + j,
                ) as usize]
            })
            .collect();
        println!(
            "  the region's present rise ({:.1} s): mean {:.4} mm/yr, the fastest node {:.4} mm/yr, {} of {} nodes rising",
            started.elapsed().as_secs_f64(),
            node_rise.iter().sum::<f64>() / node_rise.len() as f64 * 1_000.0,
            node_rise.iter().copied().fold(0.0, f64::max) * 1_000.0,
            node_rise.iter().filter(|&&r| r > 0.0).count(),
            node_rise.len()
        );
        if let Some(dir) = hillshade_dir.as_deref() {
            let fine_m = arg("--lem-fine-m", 16) as f64;
            lem_fine_views(
                &out,
                &node_rain,
                nw,
                node_m,
                cell_m,
                &today,
                None,
                dir,
                &label,
                fine_m,
                Some(&node_rise),
            );
        }
        return;
    }

    // ★ THE SEAM TEST (B81, G-SEAM — blocking; `--seam-test --lem-at F,I,J`, with `--artifact` and `--hillshade DIR`):
    // the region of `--r2-nodes` nodes centred on the node, its coarse solve at the rung-7 cell, the present rise (cached
    // beside the pictures), two neighbour L2 tiles at the middle solved apart with a 1 km buffer and ONE joint solve of
    // both, then `seam_report`.
    // ★ THE SEAM AT THE 128 m LEVEL (B0 step 2b; `--seam-l1 --lem-at F,I,J`): a region of 20 × 20 nodes centred on the
    // node; two neighbour 128 m tiles of 8 × 8 nodes with a 2-node buffer solved apart and ONE solve of both, stitched by
    // the one stitch (the cut free over one node, `--cut-blend` cells), then `seam_report`.
    if args.iter().any(|a| a == "--seam-l1") {
        lem::RETURN_LISTS.store(false, std::sync::atomic::Ordering::Relaxed);
        let at: Vec<i64> = sarg("--lem-at")
            .expect("--lem-at F,I,J")
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        // The region: 22 × 22 nodes, every tile one node in from its edge (`pad`), so each tile's rim is crossed by the
        // 8 km rivers' real links (a region that ends at a tile's rim dropped them: its water ponded at the spill).
        let nwu = 22usize;
        let nw = nwu as i64;
        let face = Face::from_index(at[0] as u8).expect("a face");
        let (i0, j0) = (at[1] - nw / 2, at[2] - nw / 2);
        let sea = f64::from(artifact.sea_m);
        let idx = |i: i64, j: i64| node_at(&lattice, face, i0 + i, j0 + j) as usize;
        let (mut node_z, mut node_rain, mut node_recv) = (
            vec![0.0f64; nwu * nwu],
            vec![0.0f64; nwu * nwu],
            vec![usize::MAX; nwu * nwu],
        );
        for k in 0..nwu * nwu {
            let (i, j) = ((k % nwu) as i64, (k / nwu) as i64);
            let row = artifact.rows[idx(i, j)];
            node_z[k] = f64::from(row.z_m) - sea;
            node_rain[k] = unlog(row.rain, 4) / 1_000.0;
            if row.receiver_facies & RECEIVER_NONE == 0 {
                let (di, dj) = OFF[(row.receiver_facies & RECEIVER_SLOT_MASK) as usize];
                let (ri, rj) = (i + i64::from(di), j + i64::from(dj));
                if ri >= 0 && rj >= 0 && ri < nw && rj < nw {
                    node_recv[k] = (rj * nw + ri) as usize;
                }
            }
        }
        {
            let water: Vec<f64> = (0..nwu * nwu)
                .map(|k| {
                    let row = artifact.rows[idx((k % nwu) as i64, (k / nwu) as i64)];
                    if row.water_m == vd_terrain::artifact::DRY_M || row.water_m <= row.z_m {
                        f64::NAN
                    } else {
                        f64::from(row.water_m) - sea
                    }
                })
                .collect();
            lem::FLOOR_BOUND.with(|fb| *fb.borrow_mut() = water);
        }
        // Each node's water: the 8 km world's own discharge through it (its whole catchment, inside the region or not; the
        // row's mm·m²/yr, a thousand to the cubic metre).
        let node_q: Vec<f64> = (0..nwu * nwu)
            .map(|k| {
                unlog(
                    artifact.rows[idx((k % nwu) as i64, (k / nwu) as i64)].discharge_log,
                    2,
                ) / 1_000.0
            })
            .collect();
        let dir = hillshade_dir.clone().expect("--hillshade DIR");
        let _ = std::fs::create_dir_all(&dir);
        let spacing = 128.0;
        let (tile, buf) = (8.0 * node_m, 2.0 * node_m);
        CUT_BAND_M.store(node_m as usize, std::sync::atomic::Ordering::Relaxed);
        CUT_BLEND_CELLS.store(arg("--cut-blend", 8), std::sync::atomic::Ordering::Relaxed);
        let pad = node_m;
        let (ax0, ay0) = (pad + 2.0 * node_m, pad + 6.0 * node_m);
        // `--l1-origin X,Y`: the tile's domain (its buffer included) from that node of the region, in place of tile A's.
        let origin: Option<(f64, f64)> = sarg("--l1-origin").map(|v| {
            let o: Vec<f64> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (pad + o[0] * node_m, pad + o[1] * node_m)
        });
        let (dx0, dy0) = origin.unwrap_or((ax0 - buf, ay0 - buf));
        let a = lem::solve_l1_domain(
            &node_z,
            &node_rain,
            nwu,
            node_m,
            &node_recv,
            &node_q,
            (dx0, dy0, tile + 2.0 * buf, tile + 2.0 * buf),
            spacing,
        );
        // ★ STEP 5 (`--step5`, the owner's look; B83): four 128 m tiles (2 × 2, each 8 × 8 nodes with its 2-node buffer)
        // solved at once, one a core, stitched by the one rule, and B80's report on the 16 × 16-node square against
        // today's 8 km ground — the maps, the readings and the two eye-height views.
        if args.iter().any(|a| a == "--step5") {
            let t_all = Instant::now();
            let side = tile + 2.0 * buf;
            let origins = [
                (pad, pad),
                (pad + tile, pad),
                (pad, pad + tile),
                (pad + tile, pad + tile),
            ];
            let tiles: Vec<lem::DomainSolve> = std::thread::scope(|scope| {
                let handles: Vec<_> = origins
                    .iter()
                    .map(|&(ox, oy)| {
                        let (node_z, node_rain, node_recv, node_q) =
                            (&node_z, &node_rain, &node_recv, &node_q);
                        scope.spawn(move || {
                            lem::solve_l1_domain(
                                node_z,
                                node_rain,
                                nwu,
                                node_m,
                                node_recv,
                                node_q,
                                (ox, oy, side, side),
                                spacing,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a tile solves"))
                    .collect()
            });
            println!(
                "STEP 5 {label}: four 128 m tiles solved at once in {:.1} s",
                t_all.elapsed().as_secs_f64()
            );
            let c = tiles[0].cell;
            let band =
                (CUT_BAND_M.load(std::sync::atomic::Ordering::Relaxed) as f64 / c).round() as i64;
            let blend = CUT_BLEND_CELLS.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let (ax, ay) = (pad + buf, pad + buf);
            let (ge_x, ge_y) = (
                ((ax + tile) / c).round() as i64,
                ((ay + tile) / c).round() as i64,
            );
            let fl: Vec<Field> = tiles.iter().map(Field::of).collect();
            let (north, _) = stitch_v(&fl[0], &fl[1], ge_x, band, blend);
            let (south, _) = stitch_v(&fl[2], &fl[3], ge_x, band, blend);
            let (all, _) = stitch_h(&north, &south, ge_y, band, blend);
            let n = (2.0 * tile / c).round() as usize;
            let (cx0, cy0) = ((ax / c).round() as i64, (ay / c).round() as i64);
            let mut st = vec![0.0f64; n * n];
            let mut today = vec![0.0f64; n * n];
            for y in 0..n {
                for x in 0..n {
                    st[y * n + x] = all.at(cx0 + x as i64, cy0 + y as i64);
                    today[y * n + x] = lem::smooth_at(
                        &node_z,
                        nwu,
                        node_m,
                        ax + (x as f64 + 0.5) * c,
                        ay + (y as f64 + 0.5) * c,
                    );
                }
            }
            println!(
                "  the square: {:.1} km a side, {n} × {n} cells of {c:.0} m (the four tiles stitched, their buffers cut away)",
                n as f64 * c / 1_000.0
            );
            lem_report(&st, &today, "today8km", n, c, Some(&dir), &label);
            return;
        }
        if let Some(n_par) = sarg("--l1-parallel").and_then(|v| v.parse::<usize>().ok()) {
            // ONE TILE A CORE: `n_par` tiles at different places solved at once, each on its own thread.
            let t_all = Instant::now();
            let times: Vec<f64> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..n_par)
                    .map(|t| {
                        let (node_z, node_rain, node_recv, node_q) =
                            (&node_z, &node_rain, &node_recv, &node_q);
                        scope.spawn(move || {
                            let t0 = Instant::now();
                            let ox = pad + (t % 5) as f64 * 2.0 * node_m;
                            let oy = pad + (t / 5) as f64 * 2.0 * node_m + 2.0 * node_m;
                            let _ = lem::solve_l1_domain(
                                node_z,
                                node_rain,
                                nwu,
                                node_m,
                                node_recv,
                                node_q,
                                (ox, oy, tile + 2.0 * buf, tile + 2.0 * buf),
                                spacing,
                            );
                            t0.elapsed().as_secs_f64()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a tile solves"))
                    .collect()
            });
            println!(
                "ONE TILE A CORE: {n_par} tiles at once in {:.1} s of wall time; each tile {:?} s",
                t_all.elapsed().as_secs_f64(),
                times
                    .iter()
                    .map(|t| (t * 10.0).round() / 10.0)
                    .collect::<Vec<_>>()
            );
            return;
        }
        if args.iter().any(|a| a == "--seam-l1-one") {
            if let Some(col) = sarg("--probe-col").and_then(|v| v.parse::<usize>().ok()) {
                // The heights across a column of the tile's picture, at a few rows (metres; the column's own cell starred).
                for row in [300usize, 360, 420, 480] {
                    let line: Vec<String> = (col - 8..=col + 8)
                        .map(|x| {
                            let z = a.raster[row * a.w + x];
                            if x == col {
                                format!("*{z:.0}")
                            } else {
                                format!("{z:.0}")
                            }
                        })
                        .collect();
                    println!("  row {row}: {}", line.join(" "));
                }
            }
            {
                let reg = r2::plain_region(a.w, a.cell, &a.raster);
                let (straight, ns) = r2::straight_share(&reg, &a.raster, 0.05, 16);
                let (lean, nl) = r2::lean(&reg, &a.raster, 0.05);
                println!(
                    "  THE POINT SET {}: the straight share {:.1} % of {ns} valley cells (16 cells down), the lean {:.1} % of {nl} (none: 22 %)",
                    lem::POINT_SET.load(std::sync::atomic::Ordering::Relaxed),
                    100.0 * straight,
                    100.0 * lean
                );
            }
            // The tile inside the domain: its buffer `buf` each side; half the buffer shipped with it.
            let (bx, tc) = (
                (buf / spacing).round() as usize,
                (tile / spacing).round() as usize,
            );
            p_size(
                &a.raster,
                a.w,
                bx,
                bx,
                tc,
                bx / 2,
                &format!("{label} L1 (128 m)"),
            );
            {
                let node_cells = (node_m / spacing).round() as usize;
                let (n0x, n0y) = (
                    (dx0 / node_m).round() as usize,
                    (dy0 / node_m).round() as usize,
                );
                p_switch_band(
                    &a.raster,
                    a.w,
                    a.cell,
                    bx,
                    bx,
                    tc,
                    node_cells,
                    |i, j| node_z[(n0y + j) * nwu + n0x + i],
                    lem::threshold_pub(),
                    &format!("{label} L1"),
                );
            }
            hillshade_rect(
                &a.raster,
                a.w,
                a.h,
                a.cell,
                &format!("{dir}/{label}_tile_a.png"),
            );
            return;
        }
        let b = lem::solve_l1_domain(
            &node_z,
            &node_rain,
            nwu,
            node_m,
            &node_recv,
            &node_q,
            (
                ax0 + tile - buf,
                ay0 - buf,
                tile + 2.0 * buf,
                tile + 2.0 * buf,
            ),
            spacing,
        );
        let j = lem::solve_l1_domain(
            &node_z,
            &node_rain,
            nwu,
            node_m,
            &node_recv,
            &node_q,
            (
                ax0 - buf,
                ay0 - buf,
                2.0 * tile + 2.0 * buf,
                tile + 2.0 * buf,
            ),
            spacing,
        );
        seam_report(&a, &b, &j, ax0, ay0, tile, &dir, &label);
        return;
    }

    if args.iter().any(|a| a == "--seam-test") {
        let at: Vec<i64> = sarg("--lem-at")
            .expect("--lem-at F,I,J")
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        let nw = arg("--r2-nodes", 12) as i64;
        let face = Face::from_index(at[0] as u8).expect("a face");
        let (i0, j0) = (at[1] - nw / 2, at[2] - nw / 2);
        let nwu = nw as usize;
        let sea = f64::from(artifact.sea_m);
        let (mut node_z, mut node_rain) = (vec![0.0f64; nwu * nwu], vec![0.0f64; nwu * nwu]);
        for k in 0..nwu * nwu {
            let row = artifact.rows
                [node_at(&lattice, face, i0 + (k % nwu) as i64, j0 + (k / nwu) as i64) as usize];
            node_z[k] = f64::from(row.z_m) - sea;
            node_rain[k] = unlog(row.rain, 4) / 1_000.0;
        }
        let dir = hillshade_dir.clone().expect("--hillshade DIR");
        let _ = std::fs::create_dir_all(&dir);
        let out = lem::solve(&node_z, &node_rain, nwu, node_m, 128.0, 128.0);
        let out_cell = 128.0;
        let cache = format!("{dir}/present_rise.bin");
        let rise_all: Vec<f64> = match std::fs::read(&cache) {
            Ok(bytes) if bytes.len() == 8 * lattice.node_count() => bytes
                .chunks_exact(8)
                .map(|w| f64::from_le_bytes(w.try_into().expect("eight bytes")))
                .collect(),
            _ => {
                let r = present_rise(&body, &lattice, &words);
                let mut bytes = Vec::with_capacity(r.len() * 8);
                for v in &r {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(&cache, bytes).expect("the rise caches");
                r
            }
        };
        let node_rise: Vec<f64> = (0..nwu * nwu)
            .map(|k| {
                rise_all
                    [node_at(&lattice, face, i0 + (k % nwu) as i64, j0 + (k / nwu) as i64) as usize]
            })
            .collect();
        let spacing = arg("--lem-fine-m", 16) as f64;
        let buf = arg("--lem-buffer-m", 1_024) as f64;
        CUT_BAND_M.store((buf / 2.0) as usize, std::sync::atomic::Ordering::Relaxed);
        CUT_BLEND_CELLS.store(arg("--cut-blend", 3), std::sync::atomic::Ordering::Relaxed);
        let (ia, ja) = ((nwu / 2 - 1) as f64, (nwu / 2) as f64);
        let (ax0, ay0) = (ia * node_m, ja * node_m);
        // The span: four response times of the pair (its coarse mean over its coarse lowest, over its mean rise).
        let mut hs = Vec::new();
        for y in 0..64 {
            for x in 0..128 {
                let (px, py) = (
                    ax0 + (x as f64 + 0.5) * node_m / 64.0,
                    ay0 + (y as f64 + 0.5) * node_m / 64.0,
                );
                if let Some(z) = out
                    .terrain
                    .get_elevation(&fastlem::models::surface::sites::Site2D { x: px, y: py })
                {
                    hs.push(z);
                }
            }
        }
        let mean = hs.iter().sum::<f64>() / hs.len().max(1) as f64;
        let low = hs.iter().copied().fold(f64::MAX, f64::min);
        let ka = ja as usize * nwu + ia as usize;
        let rise = 0.5 * (node_rise[ka] + node_rise[ka + 1]);
        let span = 4.0 * ((mean - low).max(1.0) / rise.max(1.0e-9)).clamp(1.0e4, 5.0e7);
        let steps = lem::RELAX_STEPS.load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "  the pair: nodes ({ia}, {ja}) and ({}, {ja}); rise {:.3} mm/yr; the span {:.2} Myr in {steps} steps; points {spacing} m",
            ia + 1.0,
            rise * 1_000.0,
            span / 1.0e6
        );
        if args.iter().any(|a| a == "--seam-corner") {
            // The four tiles NW, NE, SW, SE and ONE solve of all four.
            let side = node_m + 2.0 * buf;
            let tile = |dx: f64, dy: f64| {
                lem::solve_domain(
                    &out,
                    &node_rain,
                    nwu,
                    node_m,
                    (ax0 + dx * node_m - buf, ay0 + dy * node_m - buf, side, side),
                    spacing,
                    &node_rise,
                    span,
                    steps,
                )
            };
            let (nw_t, ne_t, sw_t, se_t) = (
                tile(0.0, 0.0),
                tile(1.0, 0.0),
                tile(0.0, 1.0),
                tile(1.0, 1.0),
            );
            let joint = lem::solve_domain(
                &out,
                &node_rain,
                nwu,
                node_m,
                (
                    ax0 - buf,
                    ay0 - buf,
                    2.0 * node_m + 2.0 * buf,
                    2.0 * node_m + 2.0 * buf,
                ),
                spacing,
                &node_rise,
                span,
                steps,
            );
            corner_report(
                [&nw_t, &ne_t, &sw_t, &se_t],
                &joint,
                ax0,
                ay0,
                node_m,
                &dir,
                &label,
            );
            return;
        }
        let a = lem::solve_domain(
            &out,
            &node_rain,
            nwu,
            node_m,
            (ax0 - buf, ay0 - buf, node_m + 2.0 * buf, node_m + 2.0 * buf),
            spacing,
            &node_rise,
            span,
            steps,
        );
        {
            let (bx, tc) = (
                (buf / spacing).round() as usize,
                (node_m / spacing).round() as usize,
            );
            p_size(
                &a.raster,
                a.w,
                bx,
                bx,
                tc,
                bx / 2,
                &format!("{label} L2 ({spacing:.0} m)"),
            );
            // P-SWITCH at the 16 m level: its own rungs 4 → 5 → 6, then its 64 m means against the 128 m ground (rung 7)
            // read bilinearly — the swap where the 16 m level hands over to the 128 m one.
            let pixel = vd_core::geometry::REFERENCE_VIEW_FOV_Y_RAD / 720.0;
            let mut lv = (a.w, a.raster.clone());
            let s_c = 1.25;
            println!(
                "  P-SWITCH {label} L2: the step between a rung and the next coarser one over the tile (S_c {s_c:.2})"
            );
            for r in 0..3usize {
                let (fw, fv) = (lv.0, lv.1.clone());
                let fcell = spacing * (1 << r) as f64;
                let ccell = 2.0 * fcell;
                let (cw, cv): (usize, Vec<f64>) = if r < 2 {
                    let nw2 = fw / 2;
                    let mut next = vec![0.0f64; nw2 * nw2];
                    for y in 0..nw2 {
                        for x in 0..nw2 {
                            next[y * nw2 + x] = 0.25
                                * (fv[2 * y * fw + 2 * x]
                                    + fv[2 * y * fw + 2 * x + 1]
                                    + fv[(2 * y + 1) * fw + 2 * x]
                                    + fv[(2 * y + 1) * fw + 2 * x + 1]);
                        }
                    }
                    (nw2, next)
                } else {
                    (0, Vec::new())
                };
                let coarse_at = |px: f64, py: f64| -> f64 {
                    if r < 2 {
                        let (u, v) = (px / ccell - 0.5, py / ccell - 0.5);
                        let (i, j) = (
                            u.floor().clamp(0.0, (cw - 2) as f64) as usize,
                            v.floor().clamp(0.0, (cw - 2) as f64) as usize,
                        );
                        let (fu, fv2) = (
                            (u - i as f64).clamp(0.0, 1.0),
                            (v - j as f64).clamp(0.0, 1.0),
                        );
                        let at = |p: usize, q: usize| cv[q * cw + p];
                        (at(i, j) * (1.0 - fu) + at(i + 1, j) * fu) * (1.0 - fv2)
                            + (at(i, j + 1) * (1.0 - fu) + at(i + 1, j + 1) * fu) * fv2
                    } else {
                        // The 128 m ground (the region's own solve), bilinear on its cells, in region metres.
                        let (gx, gy) = (a.x0 + px, a.y0 + py);
                        let (u, v) = (gx / out_cell - 0.5, gy / out_cell - 0.5);
                        let (i, j) = (u.floor() as usize, v.floor() as usize);
                        let (fu, fv2) = (u - i as f64, v - j as f64);
                        let at = |p: usize, q: usize| out.raster[q * out.w + p];
                        (at(i, j) * (1.0 - fu) + at(i + 1, j) * fu) * (1.0 - fv2)
                            + (at(i, j + 1) * (1.0 - fu) + at(i + 1, j + 1) * fu) * fv2
                    }
                };
                let f = 1usize << r;
                let mut steps = Vec::new();
                for y in bx / f..(bx + tc) / f {
                    for x in bx / f..(bx + tc) / f {
                        let (px, py) = ((x as f64 + 0.5) * fcell, (y as f64 + 0.5) * fcell);
                        steps.push((fv[y * fw + x] - coarse_at(px, py)).abs());
                    }
                }
                steps.sort_by(f64::total_cmp);
                let (max, p99) = (steps[steps.len() - 1], steps[steps.len() * 99 / 100]);
                let bound = s_c * 0.71 * (fcell + ccell);
                println!(
                    "    rung {} → {} ({fcell:.0} m → {ccell:.0} m{}): the step max {max:.1} m, 99th {p99:.1} m; the bound {bound:.0} m; one pixel at {:.1} km (measured max) / {:.1} km (bound) / {:.1} km (the coarser cell)",
                    4 + r,
                    5 + r,
                    if r == 2 { ", the 128 m ground" } else { "" },
                    max / pixel / 1_000.0,
                    bound / pixel / 1_000.0,
                    ccell / pixel / 1_000.0
                );
                lv = (cw, cv);
            }
        }
        if args.iter().any(|a| a == "--seam-one") {
            return;
        }
        let b = lem::solve_domain(
            &out,
            &node_rain,
            nwu,
            node_m,
            (
                ax0 + node_m - buf,
                ay0 - buf,
                node_m + 2.0 * buf,
                node_m + 2.0 * buf,
            ),
            spacing,
            &node_rise,
            span,
            steps,
        );
        let j = lem::solve_domain(
            &out,
            &node_rain,
            nwu,
            node_m,
            (
                ax0 - buf,
                ay0 - buf,
                2.0 * node_m + 2.0 * buf,
                node_m + 2.0 * buf,
            ),
            spacing,
            &node_rise,
            span,
            steps,
        );
        seam_report(&a, &b, &j, ax0, ay0, node_m, &dir, &label);
        return;
    }

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
    let coarse_span = pick.nodes as usize * coarse_cells_per_node;
    let timing: Option<(f64, f64)> = if gates_only {
        println!(
            "\n  --gates-only: the timing, the halo and the iteration studies (1–3, 6, 7) are skipped"
        );
        None
    } else {
        println!(
            "\n── 1. NANOSECONDS PER CELL PER ITERATION, one core ─────────────────────────────"
        );
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
        println!(
            "\n── 2. THE HALO WIDTH at which the interior stops changing ──────────────────────"
        );
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
        println!(
            "\n── 3. THE ITERATIONS the routed drainage takes to stop moving ──────────────────"
        );
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
                        seed_from_discharge(
                            &mut g,
                            &lattice,
                            &artifact,
                            pick,
                            coarse_cells_per_node,
                        );
                    }
                    Seeding::Inflow => {
                        seed_inflow_boundary(
                            &mut g,
                            &lattice,
                            &artifact,
                            pick,
                            coarse_cells_per_node,
                        );
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
                    Seeding::Zero => {
                        "seeded from ZERO (the reference's own start)                    "
                    }
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
        Some((ns_coarse, ns_fine))
    };

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
    if let Some(dir) = hillshade_dir.as_deref() {
        let sea_m = f64::from(artifact.sea_m);
        let before_m: Vec<f64> = before
            .iter()
            .map(|v| v.raw() as f64 / f64::from(1 << H_BITS))
            .collect();
        let after_m: Vec<f64> =
            g.h.iter()
                .map(|v| v.raw() as f64 / f64::from(1 << H_BITS))
                .collect();
        for (when, field) in [("before", &before_m), ("after", &after_m)] {
            for (view, path) in
                shade::write_views(field, g.w, g.halo, g.cell_m, sea_m, dir, &label, when)
            {
                println!("  the shaded relief, {when}, {view}: {path}");
            }
        }
    }

    // ★ THE ×2 STEPS (ruling B76; the owner: "Steps of ×2"; Schott et al. 2024's alternation): `--r2-cascade FINEST_M`
    // erodes the belt's tile at 1 024 m, then a centred region of half the side at half the cell — the same cell count —
    // from the coarser level's eroded field enlarged twice (Catmull-Rom) plus, with `--r2-start-drawn`, the game's own
    // relief between the two cells (the drawn ground at the finer rung less the coarser); each 2 × 2 block keeps its
    // coarse cell's mean ([`r2::hold_to`]). Each level: the dose (`--r2-cascade-myr` million years below 1 km, 500 at
    // it), the same run again for G-DOSE, the Earth reference's readings and the pictures.
    if let Some(finest) = sarg("--r2-cascade").and_then(|v| v.parse::<f64>().ok()) {
        let halo = 64usize;
        let drawn = args.iter().any(|a| a == "--r2-start-drawn");
        let below_myr = arg("--r2-cascade-myr", 100) as f64;
        let node_hold = args.iter().any(|a| a == "--r2-cascade-node-hold");
        println!(
            "  the ×2 steps' hold: {}",
            if node_hold {
                "the 8 km nodes' means at every level"
            } else {
                "each 2 × 2 block at its coarse cell's mean"
            }
        );
        let (mut nodes, mut cell) = (pick.nodes, 1_024.0f64);
        let mut parent: Option<(Vec<f64>, usize)> = None;
        loop {
            let cpn = (node_m / cell).round() as usize;
            let inset = (pick.nodes - nodes).div_euclid(2);
            let lp = TilePick {
                i0: pick.i0 + inset,
                j0: pick.j0 + inset,
                nodes,
                ..pick
            };
            let mut reg = r2::region(&lattice, &artifact, lp, cpn, halo);
            reg.rho8 = args.iter().any(|a| a == "--r2-rho8");
            reg.mfd = args.iter().any(|a| a == "--r2-mfd");
            reg.dinf = reg.rho8 || reg.mfd || args.iter().any(|a| a == "--r2-dinf");
            reg.creep = args.iter().any(|a| a == "--r2-creep");
            let w = reg.w;
            let rung = cell.log2().round() as u8;
            let start: Vec<f64> = match &parent {
                None if drawn => drawn_at(&body, &artifact, &lattice, lp, cpn, halo, rung),
                None => build(&lattice, &artifact, lp, cpn, halo)
                    .h
                    .iter()
                    .map(|v| v.raw() as f64 / f64::from(1 << H_BITS))
                    .collect(),
                Some((pf, pw)) => {
                    let o = (pw - w / 2).div_euclid(2);
                    let at = |i: i64, j: i64| {
                        let (i, j) = (i.clamp(0, *pw as i64 - 1), j.clamp(0, *pw as i64 - 1));
                        pf[j as usize * pw + i as usize]
                    };
                    let mut up: Vec<f64> = (0..w * w)
                        .map(|k| {
                            let u = o as f64 + ((k % w) as f64 + 0.5) / 2.0 - 0.5;
                            let v = o as f64 + ((k / w) as f64 + 0.5) / 2.0 - 0.5;
                            let (iu, fu) = (u.floor() as i64, u - u.floor());
                            let (iv, fv) = (v.floor() as i64, v - v.floor());
                            let mut rows = [0.0f64; 4];
                            for (r, slot) in rows.iter_mut().enumerate() {
                                let j = iv + r as i64 - 1;
                                *slot = catmull_rom(
                                    [at(iu - 1, j), at(iu, j), at(iu + 1, j), at(iu + 2, j)],
                                    fu,
                                );
                            }
                            catmull_rom(rows, fv)
                        })
                        .collect();
                    if drawn {
                        let fine = drawn_at(&body, &artifact, &lattice, lp, cpn, halo, rung);
                        let coarse = drawn_at(&body, &artifact, &lattice, lp, cpn, halo, rung + 1);
                        for ((u, f), c) in up.iter_mut().zip(&fine).zip(&coarse) {
                            *u += f - c;
                        }
                    }
                    // `--r2-cascade-node-hold`: every level keeps only the 8 km nodes' means (R2's own hold); the
                    // block hold (each 2 × 2 block at its coarse cell's mean) otherwise.
                    if !node_hold {
                        r2::hold_to(&mut reg, pf, *pw, o);
                    }
                    up
                }
            };
            let dt = 1.0e6f64.min(cell * cell / (20.0 * r2::CREEP_K_M2_YR));
            let myr = if parent.is_none() { 500.0 } else { below_myr };
            let steps = (myr * 1.0e6 / dt).round() as usize;
            println!(
                "\n── THE ×2 STEP at {cell:.0} m: {nodes} nodes ({:.0} km), {steps} steps of {:.0} years ({myr:.0} Myr), {} ──",
                nodes as f64 * node_m / 1_000.0,
                dt,
                if parent.is_none() {
                    "the first level"
                } else {
                    "from the coarser level"
                }
            );
            let a = r2::run(&reg, &start, steps, dt);
            let b = r2::run(&reg, &a.z, steps, dt);
            let interior = |f: &[f64]| -> Vec<f64> {
                (halo..w - halo)
                    .flat_map(|y| (halo..w - halo).map(move |x| f[y * w + x]))
                    .collect()
            };
            let cut: f64 = interior(&a.z)
                .iter()
                .zip(interior(&start))
                .map(|(x, s)| (x - s).abs())
                .sum::<f64>();
            let dose: f64 = interior(&b.z)
                .iter()
                .zip(interior(&a.z))
                .map(|(x, s)| (x - s).abs())
                .sum::<f64>();
            println!(
                "  {:.3} s a step; G-DOSE (2T against T) {:.1} % of A's change",
                a.step_s,
                100.0 * dose / cut.max(f64::MIN_POSITIVE)
            );
            if parent.is_none() {
                let (mean, median, p90, n) = macro_rate(&lattice, &artifact, lp);
                let (_, _, lc_macro) = r2::balance(&reg, median);
                println!(
                    "  THE MACRO'S OWN RATE on this tile (the joint law at its {n} draining dry nodes): mean {:.4}, median \
                     {:.4}, 90th percentile {:.4} mm/yr; the law's l_c at the median {lc_macro:.0} m",
                    mean * 1_000.0,
                    median * 1_000.0,
                    p90 * 1_000.0
                );
            }
            let (rain, rock, lc) = r2::balance(&reg, b.rise_m_yr);
            println!(
                "  THE BALANCE: rain {rain:.3} m/yr, rock factor {rock:.2}, the rise (the hold's, B's second half) {:.4} mm/yr; \
                 the law's l_c {lc:.0} m (Theodoratos et al. 2018) = {:.1} cells; valleys 6.4–12.7 l_c apart for n = 1 \
                 (STATED there; n = {} UNMEASURED): {:.0}–{:.0} m",
                b.rise_m_yr * 1_000.0,
                lc / cell,
                vd_terrain::solve::JOINT_N,
                6.4 * lc,
                12.7 * lc
            );
            let (straight, n) = r2::straight_share(&reg, &a.z, 0.05, 16);
            let [x, y, d] = r2::stripes(&reg, &a.z);
            let (lean, nl) = r2::lean(&reg, &a.z, 0.05);
            println!(
                "  THE STRAIGHT LINES {:.1} % of {n}; THE STRIPES {x:.3}/{y:.3}/{d:.3}; THE LEAN {:.1} % of {nl}",
                straight * 100.0,
                lean * 100.0
            );
            if let Some(dir) = hillshade_dir.as_deref() {
                let sea_m = f64::from(artifact.sea_m);
                let label = format!("{label}_c{cell:.0}");
                for (what, file) in
                    shade::write_views(&a.z, w, halo, cell, sea_m, dir, &label, "cascade")
                {
                    println!("  the shaded relief, {what}: {file}");
                }
            }
            parent = Some((a.z, w));
            if cell / 2.0 < finest - 0.5 || nodes < 2 {
                break;
            }
            cell /= 2.0;
            nodes /= 2;
        }
        return;
    }

    // ★ R2's FIRST CANDIDATE (`--r2`): the game's joint law at 1 024 m from the same start, A = (Δt, N), B = A run on
    // to 2N (G-DOSE), C = (Δt/2, 2N) from the start (G-STEP); the same gates and pictures as the control.
    if args.iter().any(|a| a == "--r2") {
        let halo = arg("--r2-halo", 64);
        let steps = arg("--r2-steps", 200);
        let dt_yr = arg("--r2-dt-yr", 50_000) as f64;
        println!(
            "\n── R2's FIRST CANDIDATE: the joint law at 1 024 m (K {:e}, n {}), Q = A·R, the mean hold, the macro's talus ──",
            vd_terrain::solve::JOINT_K_LP,
            vd_terrain::solve::JOINT_N
        );
        println!(
            "  the dose, STATED: {steps} steps of {dt_yr:.0} years = {:.1} Myr; a {halo}-cell halo",
            steps as f64 * dt_yr / 1.0e6
        );
        // ★ THE CELL AND THE TILE (the scale bench, ruling B75's R2 step): `--r2-cell-m` the cell (1 024 m unless
        // stated), `--r2-nodes` a smaller tile of that many nodes, centred on the belt's tile, so a finer cell costs
        // about the same cells.
        let cells_per_node = (node_m / arg("--r2-cell-m", 1_024) as f64).round() as usize;
        let nodes = (arg("--r2-nodes", pick.nodes as usize) as i64).min(pick.nodes);
        let inset = (pick.nodes - nodes).div_euclid(2);
        let pick = TilePick {
            i0: pick.i0 + inset,
            j0: pick.j0 + inset,
            nodes,
            ..pick
        };
        println!(
            "  the cell {:.0} m ({cells_per_node} a node), the tile {nodes} nodes ({:.0} km a side)",
            node_m / cells_per_node as f64,
            nodes as f64 * node_m / 1_000.0
        );
        let g0 = build(&lattice, &artifact, pick, cells_per_node, halo);
        let s: Vec<f64> =
            g0.h.iter()
                .map(|v| v.raw() as f64 / f64::from(1 << H_BITS))
                .collect();
        // ★ THE START FROM THE GAME'S OWN GROUND (ruling B76, cause 1): `--r2-start-drawn` reads the DRAWN ground at
        // each cell's centre at the rung of its cell — the field and the recipe's octaves down to that cell, what the
        // window draws today — in place of the 8 km field's smooth Catmull-Rom.
        let s = if args.iter().any(|a| a == "--r2-start-drawn") {
            let started = Instant::now();
            let drawn = drawn_start(&body, &artifact, &lattice, pick, cells_per_node, halo);
            println!(
                "  the start: the game's own drawn ground at the cell's rung ({:.1} s)",
                started.elapsed().as_secs_f64()
            );
            drawn
        } else {
            println!("  the start: the 8 km field's Catmull-Rom (smooth under the nodes)");
            s
        };
        let mut reg = r2::region(&lattice, &artifact, pick, cells_per_node, halo);
        reg.rho8 = args.iter().any(|a| a == "--r2-rho8");
        reg.dinf = reg.rho8 || args.iter().any(|a| a == "--r2-dinf");
        reg.creep = args.iter().any(|a| a == "--r2-creep");
        reg.mfd = args.iter().any(|a| a == "--r2-mfd");
        reg.dinf = reg.dinf || reg.mfd;
        println!(
            "  the hillslope law: {}",
            if reg.creep {
                "Roering, Kirchner & Dietrich 1999, q = K·S/(1 − (S/S_c)²), K 0.003 m²/yr, S_c 1.25 (the Oregon Coast Range)"
            } else {
                "none (the talus at repose only)"
            }
        );
        println!(
            "  the routing: {}",
            if reg.mfd {
                "MFD (Freeman 1991): every lower neighbour, shares by slope^1.1 (p 1.1, a numerical constant)"
            } else if reg.rho8 {
                "Rho8 on D∞'s facet (Fairfield & Leymarie 1991), one seeded receiver a cell"
            } else if reg.dinf {
                "D∞ (Tarboton 1997), two receivers a cell"
            } else {
                "the steepest single receiver (D8)"
            }
        );
        let a = r2::run(&reg, &s, steps, dt_yr);
        let b = r2::run(&reg, &a.z, steps, dt_yr);
        let c = r2::run(&reg, &s, steps * 2, dt_yr / 2.0);
        let interior = |f: &[f64]| -> Vec<f64> {
            let span = reg.w - 2 * halo;
            (0..span)
                .flat_map(|y| (0..span).map(move |x| f[(y + halo) * reg.w + x + halo]))
                .collect()
        };
        let (si, ai, bi, ci) = (interior(&s), interior(&a.z), interior(&b.z), interior(&c.z));
        let cut_a: f64 = interior(&a.cut).iter().sum();
        let stats = |x: &[f64], y: &[f64]| -> (f64, f64, f64, f64) {
            let mut d: Vec<f64> = x.iter().zip(y).map(|(p, q)| (p - q).abs()).collect();
            let sum: f64 = d.iter().sum();
            d.sort_by(f64::total_cmp);
            let q = |p: f64| d[((d.len() - 1) as f64 * p) as usize];
            (sum / d.len() as f64, q(0.5), q(0.99), sum)
        };
        let (mean_moved, med_moved, p99_moved, _) = stats(&ai, &si);
        println!(
            "  A moved the ground from S: {mean_moved:.1} m on average, median {med_moved:.1} m, p99 {p99_moved:.1} m; the rock cut over the tile {:.3e} m·cells; {:.3} s a step; the hold's last residue {:.2} m",
            cut_a, a.step_s, a.hold_residue_m
        );
        let (dose_mean, _, dose_p99, dose_sum) = stats(&bi, &ai);
        println!(
            "  G-DOSE (2T against T): mean {dose_mean:.2} m, p99 {dose_p99:.2} m; the change over A's cut {:.1} % (the design's band: under 5 %)",
            100.0 * dose_sum / cut_a.max(1e-9)
        );
        let (step_mean, _, step_p99, step_sum) = stats(&ci, &ai);
        println!(
            "  G-STEP (Δt/2 against Δt, the same T): mean {step_mean:.2} m, p99 {step_p99:.2} m; the change over A's cut {:.1} %",
            100.0 * step_sum / cut_a.max(1e-9)
        );
        let span = reg.w - 2 * halo;
        for (when, f) in [
            ("S, the macro ground read at 1 024 m", &si),
            ("A", &ai),
            ("B (2T)", &bi),
            ("C (Δt/2)", &ci),
        ] {
            report_heights(f, span, reg.cell_m, when);
        }
        let (lean_s, n_s) = r2::lean(&reg, &s, 0.05);
        let (lean_a, n_a) = r2::lean(&reg, &a.z, 0.05);
        println!(
            "  THE GRID'S LEAN (the valleys' bearings within 5° of a lattice line; none: 22 %): S {:.1} % over {n_s} cells, A {:.1} % over {n_a}",
            lean_s * 100.0,
            lean_a * 100.0
        );
        // ★ THE STRAIGHT LINES (ruling B76): the valleys that run straight.
        for (when, f) in [("S", &s), ("A", &a.z)] {
            let (straight, n) = r2::straight_share(&reg, f, 0.05, 16);
            println!(
                "  THE STRAIGHT LINES {when}: {:.1} % of {n} valley cells keep their next 16 cells downstream within one \
                 cell of a straight line",
                straight * 100.0
            );
        }
        // ★ THE STRIPES (ruling B75's R2 step): the grid's two-cell wave along each axis.
        for (when, f) in [("S", &s), ("A", &a.z)] {
            let [x, y, d] = r2::stripes(&reg, f);
            println!(
                "  THE STRIPES {when} (the second difference's energy over the first's, interior; a smooth field reads near 0, \
                 a two-cell wave 4): along x {x:.3}, along y {y:.3}, along the diagonals {d:.3}"
            );
        }
        if args.iter().any(|x| x == "--r2-cone") {
            let (cone_reg, cone_z) = r2::cone(&reg, 3_000.0, 0.5);
            let cone = r2::run(&cone_reg, &cone_z, steps, dt_yr);
            let (lean_c0, _) = r2::lean(&cone_reg, &cone_z, 0.05);
            let (lean_c, n_c) = r2::lean(&cone_reg, &cone.z, 0.05);
            println!(
                "  G-GRID-OP's CONE (3 000 m, 0.5 m of rain a year, the same dose): the lean before {:.1} %, after {:.1} % over {n_c} valley cells",
                lean_c0 * 100.0,
                lean_c * 100.0
            );
            report_heights(&interior(&cone.z), span, reg.cell_m, "the cone after");
            if let Some(dir) = hillshade_dir.as_deref() {
                for (view, path) in shade::write_views(
                    &cone.z, reg.w, halo, reg.cell_m, -1.0e9, dir, &label, "cone",
                ) {
                    println!("  the shaded relief, cone, {view}: {path}");
                }
            }
        }
        if let Some(dir) = hillshade_dir.as_deref() {
            let sea_m = f64::from(artifact.sea_m);
            for (when, field) in [("r2", &a.z), ("r2_2T", &b.z)] {
                for (view, path) in
                    shade::write_views(field, reg.w, halo, reg.cell_m, sea_m, dir, &label, when)
                {
                    println!("  the shaded relief, {when}, {view}: {path}");
                }
            }
        }
    }

    let Some((ns_coarse, ns_fine)) = timing else {
        return;
    };

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
/// ★ AN EARTH TILE IN SQUARE CELLS (ruling B76): a 1-arc-second HGT tile (3 601 × 3 601 big-endian heights, north row
/// first, its south-west corner in the file's name, voids −32 768 skipped) averaged into square cells of `cell_m` over the
/// largest square the tile holds (an equirectangular frame at the tile's middle latitude — a degree is 110 574 m north and
/// 111 320·cos φ m east). Returns the heights (metres, row by row, north first) and the side in cells.
fn earth_cells(path: &str, cell_m: f64) -> (Vec<f64>, usize) {
    const N: usize = 3_601;
    let bytes = std::fs::read(path).expect("the HGT tile reads");
    assert_eq!(bytes.len(), N * N * 2, "a 1-arc-second HGT tile");
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("a tile name");
    let lat0: f64 = name[1..3].parse().expect("the latitude");
    let lat0 = if name.starts_with('S') { -lat0 } else { lat0 };
    let mid = (lat0 + 0.5).to_radians();
    let (m_lat, m_lon): (f64, f64) = (110_574.0, 111_320.0 * mid.cos());
    let side_m = m_lat.min(m_lon);
    let w = (side_m / cell_m).floor() as usize;
    let mut sum = vec![0.0f64; w * w];
    let mut count = vec![0u32; w * w];
    for row in 0..N {
        let y_m = row as f64 / (N - 1) as f64 * m_lat;
        for col in 0..N {
            let x_m = col as f64 / (N - 1) as f64 * m_lon;
            let (cx, cy) = ((x_m / cell_m) as usize, (y_m / cell_m) as usize);
            if cx >= w || cy >= w {
                continue;
            }
            let k = 2 * (row * N + col);
            let h = i16::from_be_bytes([bytes[k], bytes[k + 1]]);
            if h == i16::MIN {
                continue;
            }
            sum[cy * w + cx] += f64::from(h);
            count[cy * w + cx] += 1;
        }
    }
    let z = sum
        .iter()
        .zip(&count)
        .map(|(s, &c)| if c > 0 { s / f64::from(c) } else { 0.0 })
        .collect();
    (z, w)
}

/// ★ THE GAME'S OWN DRAWN GROUND as a start (ruling B76, cause 1): at each cell's centre of `pick` the ground the client
/// draws at the rung of the cell (`vd_terrain::height::height_field_m` on the body with the artifact's sea and band), in
/// metres over the ladder radius — the field and the recipe's octaves down to that cell.
fn drawn_start(
    body: &vd_terrain::BodyDefinition,
    artifact: &Artifact,
    lattice: &MacroLattice,
    pick: TilePick,
    cells_per_node: usize,
    halo: usize,
) -> Vec<f64> {
    let cell_m = lattice.node_m() / cells_per_node as f64;
    drawn_at(
        body,
        artifact,
        lattice,
        pick,
        cells_per_node,
        halo,
        cell_m.log2().round() as u8,
    )
}

/// [`drawn_start`] at a stated rung.
fn drawn_at(
    body: &vd_terrain::BodyDefinition,
    artifact: &Artifact,
    lattice: &MacroLattice,
    pick: TilePick,
    cells_per_node: usize,
    halo: usize,
    rung: u8,
) -> Vec<f64> {
    let wet = body.with_sea_m(artifact.sea());
    let band = artifact.drawn_range_m(&wet.macro_lattice().expect("a lattice"));
    let drawn = wet.with_band_m(band.0, band.1).unwrap_or(wet);
    let radius = drawn.ladder().radius_m();
    let edge = f64::from(lattice.edge);
    let w = pick.nodes as usize * cells_per_node + 2 * halo;
    (0..w * w)
        .map(|k| {
            let (x, y) = (k % w, k / w);
            let u = (x as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5;
            let v = (y as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5;
            let a = (2.0 * (pick.i0 as f64 + u) + 1.0) / edge - 1.0;
            let b = (2.0 * (pick.j0 as f64 + v) + 1.0) / edge - 1.0;
            let d = vd_seed::bend::direction(pick.face, a, b);
            vd_terrain::height::height_field_m(&drawn, artifact, d, rung).expect("the field covers")
                - radius
        })
        .collect()
}

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

/// ★ THE MACRO'S PRESENT EROSION RATE on a tile (ruling B76, the balance): at every dry node of `pick` that drains, the
/// joint law the solve itself runs — `E = K_lp·rock·Q·S^n`, `Q` the row's discharge, `S` the fall to its receiver over
/// the chord, the rock the province's erodibility over the basement's — in metres a year. At the solve's steady state it
/// is the node's rise. Returns the mean, the median and the 90th percentile, and the nodes read.
fn macro_rate(
    lattice: &MacroLattice,
    artifact: &Artifact,
    pick: TilePick,
) -> (f64, f64, f64, usize) {
    use vd_terrain::solve::{FACIES_SEA, JOINT_K_LP, JOINT_N};
    let basement = f64::from(Province::CrystallineBasement.erodibility_q8());
    let mut rates: Vec<f64> = Vec::new();
    for j in 0..pick.nodes {
        for i in 0..pick.nodes {
            let node = node_at(lattice, pick.face, pick.i0 + i, pick.j0 + j);
            let row = artifact.rows[node as usize];
            let facies = row.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT;
            if row.receiver_facies & RECEIVER_NONE != 0 || facies & FACIES_SEA != 0 {
                continue;
            }
            let (di, dj) = OFF[(row.receiver_facies & RECEIVER_SLOT_MASK) as usize];
            let to = node_at(
                lattice,
                pick.face,
                pick.i0 + i + i64::from(di),
                pick.j0 + j + i64::from(dj),
            );
            let fall = f64::from(row.z_m) - f64::from(artifact.rows[to as usize].z_m);
            if fall <= 0.0 {
                continue;
            }
            let chord = lattice.node_m()
                * if di != 0 && dj != 0 {
                    std::f64::consts::SQRT_2
                } else {
                    1.0
                };
            let q = unlog(row.discharge_log, 2) / 1_000.0;
            let rock = Province::ALL
                .iter()
                .find(|p| p.code() == row.province)
                .map_or(1.0, |p| f64::from(p.erodibility_q8()) / basement);
            rates.push(JOINT_K_LP * rock * q * (fall / chord).powf(JOINT_N));
        }
    }
    rates.sort_by(f64::total_cmp);
    let n = rates.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0);
    }
    let mean = rates.iter().sum::<f64>() / n as f64;
    (mean, rates[n / 2], rates[(n * 9) / 10], n)
}

/// ★ THE MACRO'S PRESENT RATE OVER THE WHOLE PLANET (ruling B76): [`macro_rate`]'s law at every draining dry node
/// (receivers through the lattice's own neighbours, across the seams), the land's distribution, and the eight fastest
/// regions of 8 × 8 nodes (64 km) by their mean, each with its direction (for `--aim`), latitude, rain and height.
fn macro_rates_planet(lattice: &MacroLattice, artifact: &Artifact) {
    use vd_terrain::solve::{FACIES_SEA, JOINT_K_LP, JOINT_N};
    let basement = f64::from(Province::CrystallineBasement.erodibility_q8());
    let n = lattice.node_count();
    let mut rate = vec![f64::NAN; n];
    for node in 0..n as u32 {
        let row = artifact.rows[node as usize];
        let facies = row.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT;
        if facies & FACIES_SEA != 0 {
            continue;
        }
        rate[node as usize] = 0.0;
        if row.receiver_facies & RECEIVER_NONE != 0 {
            continue;
        }
        let slot = (row.receiver_facies & RECEIVER_SLOT_MASK) as usize;
        let to = lattice.neighbours(node)[slot];
        if to == vd_terrain::macro_lattice::NO_NODE {
            continue;
        }
        let fall = f64::from(row.z_m) - f64::from(artifact.rows[to as usize].z_m);
        if fall <= 0.0 {
            continue;
        }
        let chord = f64::from(lattice.chord_m(node, to));
        let q = unlog(row.discharge_log, 2) / 1_000.0;
        let rock = Province::ALL
            .iter()
            .find(|p| p.code() == row.province)
            .map_or(1.0, |p| f64::from(p.erodibility_q8()) / basement);
        rate[node as usize] = JOINT_K_LP * rock * q * (fall / chord).powf(JOINT_N);
    }
    let mut land: Vec<f64> = rate.iter().copied().filter(|r| !r.is_nan()).collect();
    land.sort_by(f64::total_cmp);
    let m = land.len();
    let at = |p: f64| land[((m as f64 - 1.0) * p) as usize] * 1_000.0;
    let share =
        |t: f64| land.iter().filter(|&&r| r * 1_000.0 > t).count() as f64 / m.max(1) as f64 * 100.0;
    println!(
        "THE MACRO'S PRESENT RATE over {m} dry nodes (the joint law, mm/yr): median {:.5}, 90th {:.5}, 99th {:.5}, \
         99.9th {:.5}, the largest {:.5}; over 0.01 mm/yr {:.2} %, over 0.1 {:.3} %, over 1 {:.4} % of the land",
        at(0.5),
        at(0.9),
        at(0.99),
        at(0.999),
        at(1.0),
        share(0.01),
        share(0.1),
        share(1.0)
    );
    // THE LAND'S RAIN (the rows' own, the solve's climate law): what the cover and the rivers can have.
    let mut rain: Vec<f64> = (0..n)
        .filter(|&k| !rate[k].is_nan())
        .map(|k| unlog(artifact.rows[k].rain, 4))
        .collect();
    rain.sort_by(f64::total_cmp);
    let r = rain.len().max(1);
    let wet = |t: f64| rain.iter().filter(|&&x| x >= t).count() as f64 / r as f64 * 100.0;
    println!(
        "THE LAND'S RAIN (mm/yr): median {:.0}, 10th {:.0}, 90th {:.0}; at least 250 mm {:.1} %, 500 mm {:.1} %, 1 000 mm {:.1} %, \
         2 000 mm {:.1} % of the land (Earth: about 70 / 50 / 25 / 8 %)",
        rain[r / 2],
        rain[r / 10],
        rain[(r * 9) / 10],
        wet(250.0),
        wet(500.0),
        wet(1_000.0),
        wet(2_000.0)
    );
    let edge = lattice.edge as i64;
    let block = 8i64;
    let mut regions: Vec<(f64, u32)> = Vec::new();
    for face in 0..6u8 {
        let face_e = vd_seed::bend::Face::from_index(face).expect("a face");
        for bj in 0..edge / block {
            for bi in 0..edge / block {
                let (mut sum, mut k) = (0.0f64, 0usize);
                for j in 0..block {
                    for i in 0..block {
                        let node = node_at(lattice, face_e, bi * block + i, bj * block + j);
                        let r = rate[node as usize];
                        if !r.is_nan() {
                            sum += r;
                            k += 1;
                        }
                    }
                }
                if k * 2 >= (block * block) as usize {
                    let centre = node_at(
                        lattice,
                        face_e,
                        bi * block + block / 2,
                        bj * block + block / 2,
                    );
                    regions.push((sum / k as f64, centre));
                }
            }
        }
    }
    regions.sort_by(|a, b| b.0.total_cmp(&a.0));
    let one = f64::from(1u32 << 20) * f64::from(1u32 << 20);
    for (mean, centre) in regions.iter().take(8) {
        let d = lattice.direction(*centre);
        let v = [
            d[0].raw() as f64 / one,
            d[1].raw() as f64 / one,
            d[2].raw() as f64 / one,
        ];
        let row = artifact.rows[*centre as usize];
        println!(
            "  a fast region: {:.4} mm/yr over 64 km; --aim {:.6},{:.6},{:.6}; latitude {:.1}°; rain {:.0} mm/yr; height {} m",
            mean * 1_000.0,
            v[0],
            v[1],
            v[2],
            v[2].asin().to_degrees(),
            unlog(row.rain, 4),
            row.z_m
        );
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

/// ★ THE TWO BELIEVABILITY NUMBERS on one field, through the shared gates
/// (`vd_bins::drainage_gates`): the heights handed over as plain words at `H_BITS`, a metre being
/// `1 << H_BITS` of them, and no dry mask — every cell of the tile is ground, as it always was here.
fn report_gates(g: &Grid, when: &str) {
    println!("  {when}:");
    let words: Vec<i64> = g.h.iter().map(|v| v.raw()).collect();
    let field = Heights {
        w: g.w,
        cell_m: g.cell_m,
        h: &words,
        per_m: f64::from(1 << H_BITS),
        dry: None,
    };
    let (c, n, points) = g_hack(&field);
    let band_c = (1.0..=6.0).contains(&c);
    let band_n = (0.45..=0.7).contains(&n);
    println!(
        "    G-HACK   L = {c:.2}·a^{n:.3} over {points} network cells — c {} [1, 6], n {} [0.45, 0.7]{}",
        if band_c { "INSIDE" } else { "OUTSIDE" },
        if band_n { "INSIDE" } else { "OUTSIDE" },
        if band_c && band_n { "  ✓" } else { "  ✗" }
    );
    let (breach, pits) = g_breach(&field);
    println!(
        "    G-BREACH {breach:.3e} m³ a depression, over {pits} depressions (this file's definition; the\n             reference's own units are UNVERIFIED, so the BEFORE/AFTER ratio is the number that can fail)"
    );
}

/// The two believability numbers on a field of metres (`w` × `w`, no rim), through the same shared gates.
fn report_heights(f: &[f64], w: usize, cell_m: f64, when: &str) {
    let words: Vec<i64> = f
        .iter()
        .map(|m| (m * f64::from(1 << H_BITS)).round() as i64)
        .collect();
    let field = Heights {
        w,
        cell_m,
        h: &words,
        per_m: f64::from(1 << H_BITS),
        dry: None,
    };
    let (c, n, points) = g_hack(&field);
    let (breach, pits) = g_breach(&field);
    println!(
        "  {when}: G-HACK L = {c:.2}·a^{n:.3} over {points} network cells (c in [1, 6]: {}, n in [0.45, 0.7]: {}); G-BREACH {breach:.3e} over {pits} depressions",
        if (1.0..=6.0).contains(&c) {
            "yes"
        } else {
            "NO"
        },
        if (0.45..=0.7).contains(&n) {
            "yes"
        } else {
            "NO"
        },
    );
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

/// ★ THE SHADED RELIEF (R2's slice 0; rulings B8 item 3 and B57: "a hillshade for the owner's look"). Horn's (1981)
/// slope and aspect on the 3 × 3 window, the sun from the north-west 45° over the horizon (the cartographic
/// convention), a height tint over the body's own sea, and a STATED vertical exaggeration of four, the same on every
/// picture, so a "before" and an "after" compare. Two views: the whole tile, one pixel a 1 024 m cell; and the tile's
/// centre, 128 cells read through Catmull-Rom at 128 m a pixel — the pixel a pilot 100–150 km up sees (the ground step
/// design §5.2). A measurement's picture, never a game path.
mod shade {
    /// The vertical exaggeration every picture states.
    pub const EXAGGERATION: f64 = 4.0;
    /// The centre window's side in 1 024 m cells, and its pixels a cell.
    pub const WINDOW_CELLS: usize = 128;
    pub const WINDOW_SCALE: usize = 8;

    /// Writes the tile's two views of `field` (metres over the ladder radius, `w` × `w`, a `halo` rim left out) and
    /// returns each view's name and path.
    #[allow(clippy::too_many_arguments)]
    pub fn write_views(
        field: &[f64],
        w: usize,
        halo: usize,
        cell_m: f64,
        sea_m: f64,
        dir: &str,
        label: &str,
        when: &str,
    ) -> Vec<(&'static str, String)> {
        let _ = std::fs::create_dir_all(dir);
        let span = w - 2 * halo;
        let tile: Vec<f64> = (0..span)
            .flat_map(|y| (0..span).map(move |x| field[(y + halo) * w + x + halo]))
            .collect();
        let mut out = Vec::new();
        let whole = format!("{dir}/{label}_{when}_tile.png");
        render(&tile, span, cell_m, sea_m, &whole);
        out.push(("the whole tile, 1 024 m a pixel", whole));
        let c0 = halo + span / 2 - WINDOW_CELLS / 2;
        let n = WINDOW_CELLS * WINDOW_SCALE;
        let mut win = vec![0.0f64; n * n];
        for py in 0..n {
            for px in 0..n {
                let u = c0 as f64 + (px as f64 + 0.5) / WINDOW_SCALE as f64 - 0.5;
                let v = c0 as f64 + (py as f64 + 0.5) / WINDOW_SCALE as f64 - 0.5;
                win[py * n + px] = bicubic(field, w, u, v);
            }
        }
        let window = format!("{dir}/{label}_{when}_window.png");
        render(&win, n, cell_m / WINDOW_SCALE as f64, sea_m, &window);
        out.push(("the tile's centre 131 km, 128 m a pixel", window));
        out
    }

    /// Catmull-Rom of `f` (`w` × `w`, cell centres at integer coordinates) at `(u, v)`, the rim held.
    fn bicubic(f: &[f64], w: usize, u: f64, v: f64) -> f64 {
        let (iu, fu) = (u.floor() as i64, u - u.floor());
        let (iv, fv) = (v.floor() as i64, v - v.floor());
        let at = |x: i64, y: i64| {
            f[(y.clamp(0, w as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize]
        };
        let mut rows = [0.0f64; 4];
        for (r, slot) in rows.iter_mut().enumerate() {
            let y = iv + r as i64 - 1;
            *slot =
                super::catmull_rom([at(iu - 1, y), at(iu, y), at(iu + 1, y), at(iu + 2, y)], fu);
        }
        super::catmull_rom(rows, fv)
    }

    /// The height tint over the sea: blue by depth under it, green to tan to brown to grey to snow over it.
    fn tint(z: f64, sea_m: f64) -> [f64; 3] {
        let ramp = |stops: &[(f64, [f64; 3])], t: f64| -> [f64; 3] {
            let mut prev = stops[0];
            for &(at, c) in stops {
                if t <= at {
                    let span = (at - prev.0).max(1e-9);
                    let k = ((t - prev.0) / span).clamp(0.0, 1.0);
                    return [
                        prev.1[0] + (c[0] - prev.1[0]) * k,
                        prev.1[1] + (c[1] - prev.1[1]) * k,
                        prev.1[2] + (c[2] - prev.1[2]) * k,
                    ];
                }
                prev = (at, c);
            }
            prev.1
        };
        if z < sea_m {
            ramp(
                &[
                    (0.0, [150.0, 195.0, 225.0]),
                    (1_000.0, [70.0, 130.0, 190.0]),
                    (6_000.0, [25.0, 55.0, 110.0]),
                ],
                sea_m - z,
            )
        } else {
            ramp(
                &[
                    (0.0, [95.0, 140.0, 85.0]),
                    (400.0, [160.0, 175.0, 105.0]),
                    (1_200.0, [200.0, 175.0, 115.0]),
                    (2_500.0, [160.0, 115.0, 80.0]),
                    (4_500.0, [150.0, 140.0, 132.0]),
                    (6_500.0, [240.0, 240.0, 240.0]),
                ],
                z - sea_m,
            )
        }
    }

    /// One shaded relief of `h` (`n` × `n`, `dx` metres a pixel) to `path`.
    fn render(h: &[f64], n: usize, dx: f64, sea_m: f64, path: &str) {
        let mut img = image::RgbImage::new(n as u32, n as u32);
        let zenith = (90.0f64 - 45.0).to_radians();
        // The sun's compass azimuth 315° (north-west) in the mathematical sense Horn's aspect is measured in.
        let azimuth = (360.0f64 - 315.0 + 90.0).to_radians();
        let at = |x: i64, y: i64| {
            h[(y.clamp(0, n as i64 - 1) as usize) * n + x.clamp(0, n as i64 - 1) as usize]
        };
        for y in 0..n as i64 {
            for x in 0..n as i64 {
                let (a, b, c) = (at(x - 1, y - 1), at(x, y - 1), at(x + 1, y - 1));
                let (d, f) = (at(x - 1, y), at(x + 1, y));
                let (g, hh, i) = (at(x - 1, y + 1), at(x, y + 1), at(x + 1, y + 1));
                let dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * dx) * EXAGGERATION;
                let dzdy = ((g + 2.0 * hh + i) - (a + 2.0 * b + c)) / (8.0 * dx) * EXAGGERATION;
                let slope = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
                let mut aspect = dzdy.atan2(-dzdx);
                if aspect < 0.0 {
                    aspect += std::f64::consts::TAU;
                }
                let lit = (zenith.cos() * slope.cos()
                    + zenith.sin() * slope.sin() * (azimuth - aspect).cos())
                .max(0.0);
                let base = tint(h[(y as usize) * n + x as usize], sea_m);
                let k = 0.35 + 0.65 * lit;
                img.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgb([
                        (base[0] * k).clamp(0.0, 255.0) as u8,
                        (base[1] * k).clamp(0.0, 255.0) as u8,
                        (base[2] * k).clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
        if let Err(e) = img.save(path) {
            println!("  the shaded relief could not be written to {path}: {e}");
        }
    }
}

/// ★ R2's FIRST CANDIDATE (slice 0, ruling B57): THE GAME'S OWN JOINT LAW AT 1 024 m, no caps and no calibration knob.
/// A MEASUREMENT in floats (the build owes the generator's integer form, ruling B52), never a game path. One step:
///
/// 1. the depressions filled by a priority flood from the region's rim and the sea (Barnes, Lehman & Mulla 2014); each
///    cell's receiver its steepest lower neighbour on the filled surface, or on a flat the cell the flood reached it from;
/// 2. the water: every cell's rain on its own area, summed down the receivers — `Q = A·R` (ruling B51) — and the macro
///    rivers that enter the region at its rim (the row's discharge where the outside node drains inward);
/// 3. the joint law, implicit and outlets first, exactly the macro sweep's: `x − x₀ + c·(x/L)^n = 0`, `c = Δt·K·rock·Q`
///    (`JOINT_K_LP`, `JOINT_N`; the rock the province's erodibility over the crystalline basement's); a filled hollow
///    and the sea do not cut;
/// 4. the talus by the macro's own repose law (`tan_repose_q16` from the node's aridity), half the excess a step, in
///    pairs, so the rock moved is the rock received;
/// 5. THE MEAN HOLD: each 8 km node's mean back to its row's height, by the Catmull-Rom of the nodes' residues — the
///    uplift that keeps the far view exact (the R2 design §4.5).
///
/// THE DOSE IS STATED: `steps` steps of `dt_yr` years, and the same field run twice as long is read beside it (G-DOSE).
mod r2 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    use vd_terrain::artifact::{
        Artifact, FACIES_SHIFT, RECEIVER_NONE, RECEIVER_SLOT_MASK, node_at,
    };
    use vd_terrain::macro_lattice::MacroLattice;
    use vd_terrain::solve::{FACIES_SEA, JOINT_K_LP, JOINT_N, tan_repose_q16};
    use vd_terrain::strata::Province;

    use super::{OFF, TilePick, catmull_rom, unlog};

    /// The region's per-cell facts, fixed for the run.
    pub struct Region {
        pub w: usize,
        pub halo: usize,
        pub cell_m: f64,
        pub cells_per_node: usize,
        /// The rain on each cell, metres a year.
        rain_m: Vec<f64>,
        /// The rock's share over the crystalline basement's.
        rock: Vec<f64>,
        /// The repose tangent.
        repose: Vec<f64>,
        /// A sea cell: an outlet that does not cut.
        pub sea: Vec<bool>,
        /// The macro rivers entering at the rim, m³/yr.
        inflow: Vec<f64>,
        /// The node grid the mean hold reads: its side in nodes, and each node's row height.
        nodes: usize,
        node_z: Vec<f64>,
        /// ★ The routing: D∞ (Tarboton 1997, two receivers a cell) when set, the steepest single receiver (D8)
        /// otherwise.
        pub dinf: bool,
        /// With `dinf`: Rho8's one seeded receiver of the D∞ facet instead of the two shares.
        pub rho8: bool,
        /// ★ THE HILLSLOPE LAW (the ground step design's K2; ruling B75's R2 step): Roering, Kirchner & Dietrich 1999's
        /// nonlinear creep between every pair of neighbours after the cut.
        pub creep: bool,
        /// ★ THE WATER SPLIT OVER EVERY LOWER NEIGHBOUR (ruling B76, cause 2): Freeman 1991's multiple-flow routing on
        /// the D∞ path's flood and flats.
        pub mfd: bool,
        /// ★ THE ROCK'S UPLIFT, m/yr (the forcing every landscape model is driven by; the Earth test's measured rate):
        /// every cell that is neither sea nor rim rises by it each step, before the routing. Nought: no forcing.
        pub uplift_m_yr: f64,
        /// THE MEAN HOLD on (each node's mean back to its row each step); off when an uplift drives the region.
        pub hold_on: bool,
        /// ★ THE THRESHOLD HILLSLOPE (Burbank et al. 1996, Nature 379:505; Schmidt & Montgomery 1995, Science 270:617):
        /// no slope stands steeper than the rock mass's repose, `vd_terrain::body::TAN_REPOSE` (35°); the landslides carry
        /// the excess into the channels. Replaces the stepped talus when set.
        pub threshold: bool,
    }

    /// ★ THE HILLSLOPE LAW's transport coefficient `K`, m²/yr (Roering, Kirchner & Dietrich 1999, the Oregon Coast
    /// Range, beside `S_c` 1.25 — `vd_terrain::surface::SOIL_CRITICAL_SLOPE`, the one the surface's rock reads).
    pub const CREEP_K_M2_YR: f64 = 0.003;

    /// A cell's own chance on `[0, 1)` (splitmix64 of its index; the bench's region index, so a tile's seam is not
    /// tested here — the build owes the global cell index).
    pub fn hash01(i: u64) -> f64 {
        let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// ★ THE OLDER RELIEF BELOW THE NODE (a diagnostic start): octaves of smooth value noise from one node's wave down
    /// to two cells, each octave's height `2^(−H)` of the one above (`H = (β − 1)/2`, the game's continental spectrum,
    /// `vd_terrain::land::SPECTRUM_BETA`), the whole sized so its root-mean-square height difference at one kilometre is
    /// the law's own `131 m × (1 km / 110 km)^H` (`SPECTRUM_RMS_M` at `SPECTRUM_LAG_M`). About 10 m at 1 km.
    pub fn older_relief(reg: &Region, cell_m: f64) -> Vec<f64> {
        use vd_terrain::land::{SPECTRUM_BETA, SPECTRUM_LAG_M, SPECTRUM_RMS_M};
        let w = reg.w;
        let hurst = (SPECTRUM_BETA - 1.0) / 2.0;
        let smooth = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let mut field = vec![0.0f64; w * w];
        let (mut wave, mut amp, mut octave) = (8_192.0 / cell_m, 1.0f64, 0u64);
        while wave >= 2.0 {
            let lattice = |a: i64, b: i64| -> f64 {
                let key = (octave << 48) ^ ((a as u64 & 0xFF_FFFF) << 24) ^ (b as u64 & 0xFF_FFFF);
                2.0 * hash01(key) - 1.0
            };
            for y in 0..w {
                for x in 0..w {
                    let (u, v) = (x as f64 / wave, y as f64 / wave);
                    let (a, b) = (u.floor(), v.floor());
                    let (fu, fv) = (smooth(u - a), smooth(v - b));
                    let (a, b) = (a as i64, b as i64);
                    let top = lattice(a, b) * (1.0 - fu) + lattice(a + 1, b) * fu;
                    let bottom = lattice(a, b + 1) * (1.0 - fu) + lattice(a + 1, b + 1) * fu;
                    field[y * w + x] += amp * (top * (1.0 - fv) + bottom * fv);
                }
            }
            wave /= 2.0;
            amp *= 2.0f64.powf(-hurst);
            octave += 1;
        }
        let lag = (1_024.0 / cell_m).round().max(1.0) as usize;
        let (mut sum, mut n) = (0.0f64, 0usize);
        for y in 0..w - lag {
            for x in 0..w - lag {
                let h = field[y * w + x];
                sum +=
                    (field[y * w + x + lag] - h).powi(2) + (field[(y + lag) * w + x] - h).powi(2);
                n += 2;
            }
        }
        let want = SPECTRUM_RMS_M * (lag as f64 * cell_m / SPECTRUM_LAG_M).powf(hurst);
        let scale = want / (sum / n.max(1) as f64).sqrt();
        println!(
            "  the start's older relief: {want:.1} m rms at {:.0} m, the field scaled by {scale:.2}",
            lag as f64 * cell_m
        );
        field.iter().map(|f| f * scale).collect()
    }

    /// The region of `pick` with `halo` cells, `cells_per_node` cells a node, from the stored rows.
    pub fn region(
        lattice: &MacroLattice,
        artifact: &Artifact,
        pick: TilePick,
        cells_per_node: usize,
        halo: usize,
    ) -> Region {
        let span = pick.nodes as usize * cells_per_node;
        let w = span + 2 * halo;
        let cell_m = lattice.node_m() / cells_per_node as f64;
        let basement = f64::from(Province::CrystallineBasement.erodibility_q8());
        let sea_m = f64::from(artifact.sea_m);
        let node_of = |x: usize, y: usize| -> u32 {
            let u = (x as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5;
            let v = (y as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5;
            node_at(
                lattice,
                pick.face,
                pick.i0 + u.round() as i64,
                pick.j0 + v.round() as i64,
            )
        };
        let cells = w * w;
        let (mut rain_m, mut rock, mut repose, mut sea) = (
            vec![0.0; cells],
            vec![1.0; cells],
            vec![0.5; cells],
            vec![false; cells],
        );
        let mut inflow = vec![0.0; cells];
        for y in 0..w {
            for x in 0..w {
                let i = y * w + x;
                let row = artifact.rows[node_of(x, y) as usize];
                rain_m[i] = unlog(row.rain, 4) / 1_000.0;
                let q8 = Province::ALL
                    .iter()
                    .find(|q| q.code() == row.province)
                    .map_or(basement, |q| f64::from(q.erodibility_q8()));
                rock[i] = q8 / basement;
                repose[i] = f64::from(tan_repose_q16(row.aridity)) / 65_536.0;
                sea[i] = ((row.receiver_facies >> FACIES_SHIFT) & FACIES_SEA != 0)
                    && f64::from(row.z_m) < sea_m;
                // The rim inflow: a rim cell whose outside node drains inward takes that node's discharge, shared over
                // the node's cells along the edge.
                let rim_x = x == 0 || x == w - 1;
                let rim_y = y == 0 || y == w - 1;
                if rim_x || rim_y {
                    let ox = if x == 0 { -1 } else { i64::from(rim_x) };
                    let oy = if y == 0 { -1 } else { i64::from(rim_y) };
                    let u = ((x as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5).round()
                        as i64
                        + ox;
                    let v = ((y as f64 + 0.5 - halo as f64) / cells_per_node as f64 - 0.5).round()
                        as i64
                        + oy;
                    let outside = artifact.rows
                        [node_at(lattice, pick.face, pick.i0 + u, pick.j0 + v) as usize];
                    if outside.receiver_facies & RECEIVER_NONE == 0 {
                        let (di, dj) = OFF[(outside.receiver_facies & RECEIVER_SLOT_MASK) as usize];
                        let inward =
                            (ox == 0 || i64::from(di) == -ox) && (oy == 0 || i64::from(dj) == -oy);
                        if inward {
                            // The row's discharge is mm·m²/yr; a cubic metre is a thousand of them.
                            inflow[i] =
                                unlog(outside.discharge_log, 2) / 1_000.0 / cells_per_node as f64;
                        }
                    }
                }
            }
        }
        // The node grid: every node whose square lies inside the region, its row's height.
        let nodes = w / cells_per_node;
        let first = halo as f64 / cells_per_node as f64;
        let mut node_z = vec![0.0; nodes * nodes];
        for nj in 0..nodes {
            for ni in 0..nodes {
                let u = (ni as f64 - first).floor() as i64;
                let v = (nj as f64 - first).floor() as i64;
                let node = node_at(lattice, pick.face, pick.i0 + u, pick.j0 + v);
                node_z[nj * nodes + ni] = f64::from(artifact.rows[node as usize].z_m);
            }
        }
        Region {
            w,
            halo,
            cell_m,
            cells_per_node,
            rain_m,
            rock,
            repose,
            sea,
            inflow,
            nodes,
            node_z,
            dinf: false,
            rho8: false,
            creep: false,
            mfd: false,
            uplift_m_yr: 0.0,
            hold_on: true,
            threshold: false,
        }
    }

    /// ★ G-GRID-OP's CONE (the R2 design §10.4): the region's own size, a cone `relief_m` high over its centre and
    /// sloping to its rim, uniform rain and rock, no sea and no inflow; the mean hold keeps the cone's own node means.
    /// A method with no grid lean cuts it into valleys that fan out radially at every angle.
    pub fn cone(reg: &Region, relief_m: f64, rain_m_yr: f64) -> (Region, Vec<f64>) {
        let w = reg.w;
        let cells = w * w;
        let c = (w as f64 - 1.0) / 2.0;
        let z: Vec<f64> = (0..cells)
            .map(|i| {
                let (x, y) = ((i % w) as f64 - c, (i / w) as f64 - c);
                relief_m * (1.0 - (x * x + y * y).sqrt() / c).max(0.0)
            })
            .collect();
        let (n, s) = (reg.nodes, reg.cells_per_node);
        let mut node_z = vec![0.0; n * n];
        for nj in 0..n {
            for ni in 0..n {
                let mut sum = 0.0;
                for y in nj * s..(nj + 1) * s {
                    for x in ni * s..(ni + 1) * s {
                        sum += z[y * w + x];
                    }
                }
                node_z[nj * n + ni] = sum / (s * s) as f64;
            }
        }
        (
            Region {
                w,
                halo: reg.halo,
                cell_m: reg.cell_m,
                cells_per_node: s,
                rain_m: vec![rain_m_yr; cells],
                rock: vec![1.0; cells],
                repose: vec![0.6; cells],
                sea: vec![false; cells],
                inflow: vec![0.0; cells],
                nodes: n,
                node_z,
                dinf: reg.dinf,
                rho8: reg.rho8,
                creep: reg.creep,
                mfd: reg.mfd,
                uplift_m_yr: reg.uplift_m_yr,
                hold_on: reg.hold_on,
                threshold: reg.threshold,
            },
            z,
        )
    }

    /// ★ THE HOLD TO A COARSER LEVEL (ruling B76's ×2 steps): each 2 × 2 block of this region keeps the mean of the
    /// coarser level's cell over it — the parent field `parent` (`pw × pw`), whose cell `(o + i, o + j)` covers this
    /// region's block `(i, j)`. The halo must be even, so the blocks start at the region's edge.
    pub fn hold_to(reg: &mut Region, parent: &[f64], pw: usize, o: usize) {
        assert_eq!(reg.halo % 2, 0, "an even halo");
        let n = reg.w / 2;
        reg.nodes = n;
        reg.cells_per_node = 2;
        reg.node_z = (0..n * n)
            .map(|k| parent[(o + k / n) * pw + o + k % n])
            .collect();
    }

    /// ★ THE BALANCE (ruling B76): the region's mean rain (m/yr) and rock factor on its interior, and the law's own
    /// characteristic length at a rise `u` (m/yr) — Theodoratos, Seybold & Kirchner 2018, eq. A8, for
    /// `E = K′·A^m·S^n` with `K′ = K_lp·R·rock`, `m = 1` (the joint law reads `Q = R·A`), `n` the joint law's:
    /// `l_c = (D^n·U^(1−n)/K′)^(1/(n + 2m))`, metres. A length the bench's cell must resolve for valleys to form.
    pub fn balance(reg: &Region, u: f64) -> (f64, f64, f64) {
        let (w, h) = (reg.w, reg.halo);
        let mut n = 0usize;
        let (mut rain, mut rock) = (0.0f64, 0.0f64);
        for y in h..w - h {
            for x in h..w - h {
                rain += reg.rain_m[y * w + x];
                rock += reg.rock[y * w + x];
                n += 1;
            }
        }
        let (rain, rock) = (rain / n.max(1) as f64, rock / n.max(1) as f64);
        let k = JOINT_K_LP * rain * rock;
        let nn = JOINT_N;
        let lc = (CREEP_K_M2_YR.powf(nn) * u.abs().max(f64::MIN_POSITIVE).powf(1.0 - nn) / k)
            .powf(1.0 / (nn + 2.0));
        (rain, rock, lc)
    }

    /// ★ AN EARTH REGION FOR THE GRID EROSION (ruling B76/B79, the method against Earth): a field of `nw × nw` Earth
    /// nodes of 8 192 m (their MEAN heights, `node_z`), each `cpn × cpn` cells of `cell_m`; the start is the nodes'
    /// Catmull-Rom (the same smooth start the home planet's bench uses), the hold keeps every node's mean, the rain is
    /// one published value for the tile, the rock the basement's, the repose the wet end of the solve's law (tan 0.47 —
    /// the tile is wet), the sea where the start stands at or under nought, the rim the outlets. A MEASUREMENT input,
    /// never a constant set from Earth (B79 item 1).
    pub fn earth_region(
        node_z: &[f64],
        nw: usize,
        cpn: usize,
        cell_m: f64,
        rain_m: f64,
    ) -> (Region, Vec<f64>) {
        let w = nw * cpn;
        let at = |i: i64, j: i64| {
            let (i, j) = (i.clamp(0, nw as i64 - 1), j.clamp(0, nw as i64 - 1));
            node_z[j as usize * nw + i as usize]
        };
        let start: Vec<f64> = (0..w * w)
            .map(|k| {
                let u = ((k % w) as f64 + 0.5) / cpn as f64 - 0.5;
                let v = ((k / w) as f64 + 0.5) / cpn as f64 - 0.5;
                let (iu, fu) = (u.floor() as i64, u - u.floor());
                let (iv, fv) = (v.floor() as i64, v - v.floor());
                let mut rows = [0.0f64; 4];
                for (r, slot) in rows.iter_mut().enumerate() {
                    let j = iv + r as i64 - 1;
                    *slot =
                        catmull_rom([at(iu - 1, j), at(iu, j), at(iu + 1, j), at(iu + 2, j)], fu);
                }
                catmull_rom(rows, fv)
            })
            .collect();
        let cells = w * w;
        (
            Region {
                w,
                halo: 0,
                cell_m,
                cells_per_node: cpn,
                rain_m: vec![rain_m; cells],
                rock: vec![1.0; cells],
                repose: vec![0.47; cells],
                sea: start.iter().map(|&h| h <= 0.0).collect(),
                inflow: vec![0.0; cells],
                nodes: nw,
                node_z: node_z.to_vec(),
                dinf: false,
                rho8: false,
                creep: false,
                mfd: false,
                uplift_m_yr: 0.0,
                hold_on: true,
                threshold: false,
            },
            start,
        )
    }

    /// ★ A PLAIN REGION over a field of `w × w` cells of `cell_m` with no halo (the Earth reference, ruling B76): one
    /// metre of rain a year everywhere, the basement's rock, a repose of 0.6, and the sea wherever the ground stands at
    /// or under nought — only what the readings' routing needs.
    pub fn plain_region(w: usize, cell_m: f64, z: &[f64]) -> Region {
        let cells = w * w;
        Region {
            w,
            halo: 0,
            cell_m,
            cells_per_node: 1,
            rain_m: vec![1.0; cells],
            rock: vec![1.0; cells],
            repose: vec![0.6; cells],
            sea: z.iter().map(|&h| h <= 0.0).collect(),
            inflow: vec![0.0; cells],
            nodes: w,
            node_z: z.to_vec(),
            dinf: false,
            rho8: false,
            creep: false,
            mfd: false,
            uplift_m_yr: 0.0,
            hold_on: true,
            threshold: false,
        }
    }

    /// ★ THE STRAIGHT LINES on a field (ruling B76): of the interior cells whose water is in the top `share` (the
    /// valleys), the share whose next `run` cells down the steepest path (on the filled surface) stay within one cell of
    /// the straight line from the first to the last. A digitised straight line at any angle stays within half a cell; a
    /// valley that bends leaves it. Returns the share and the valley cells read.
    pub fn straight_share(reg: &Region, z: &[f64], share: f64, run: usize) -> (f64, usize) {
        let (recv, order, _) = route(reg, z);
        let q = water(reg, &recv, &order);
        let (w, h) = (reg.w, reg.halo);
        let inside = |i: usize| {
            let (x, y) = (i % w, i / w);
            x >= h && y >= h && x < w - h && y < w - h
        };
        let mut qs: Vec<f64> = (0..w * w).filter(|&i| inside(i)).map(|i| q[i]).collect();
        qs.sort_by(f64::total_cmp);
        let threshold = qs[((1.0 - share) * qs.len() as f64) as usize];
        let (mut straight, mut all) = (0usize, 0usize);
        for i in (0..w * w).filter(|&i| inside(i) && q[i] >= threshold) {
            let mut path = vec![i];
            let mut c = i;
            while path.len() <= run {
                let r = recv[c];
                if r == u32::MAX {
                    break;
                }
                c = r as usize;
                path.push(c);
            }
            if path.len() <= run {
                continue;
            }
            let at = |k: usize| ((path[k] % w) as f64, (path[k] / w) as f64);
            let ((x0, y0), (x1, y1)) = (at(0), at(run));
            let (dx, dy) = (x1 - x0, y1 - y0);
            let len = (dx * dx + dy * dy).sqrt();
            if len == 0.0 {
                continue;
            }
            let dev = (0..=run)
                .map(|k| {
                    let (x, y) = at(k);
                    ((x - x0) * dy - (y - y0) * dx).abs() / len
                })
                .fold(0.0f64, f64::max);
            all += 1;
            straight += usize::from(dev <= 1.0);
        }
        (straight as f64 / all.max(1) as f64, all)
    }

    /// ★ THE STRIPES on a field (ruling B75's R2 step): along each axis (x, y, and the two diagonals together), the mean
    /// square of the second difference `z₋ − 2z + z₊` over the mean square of the first `z₊ − z`, on the interior. A
    /// field smooth at the grid's scale reads near nought; a wave two cells long reads 4 (its second difference is twice
    /// its first). Stripes along one axis show as that axis's ratio standing over the others'.
    pub fn stripes(reg: &Region, z: &[f64]) -> [f64; 3] {
        // One cell in from the edge at least, so every neighbour read stands inside the field.
        let (w, h) = (reg.w as i64, (reg.halo as i64).max(1));
        let at = |x: i64, y: i64| z[(y * w + x) as usize];
        let mut sums = [[0.0f64; 2]; 3];
        for y in h..w - h {
            for x in h..w - h {
                for (k, (dx, dy)) in [(1i64, 0i64), (0, 1), (1, 1), (1, -1)]
                    .into_iter()
                    .enumerate()
                {
                    let c = at(x, y);
                    let (p, m) = (at(x + dx, y + dy), at(x - dx, y - dy));
                    let slot = k.min(2);
                    sums[slot][0] += (p - 2.0 * c + m) * (p - 2.0 * c + m);
                    sums[slot][1] += (p - c) * (p - c);
                }
            }
        }
        sums.map(|[second, first]| second / first.max(f64::MIN_POSITIVE))
    }

    /// ★ THE GRID'S LEAN on a field: of the cells whose catchment is in the top `share` (the valleys), the share
    /// whose steepest downhill bearing (Horn's gradient) lies within 5° of a lattice line (0°, 45°, 90°, …). With no
    /// lean the share is 10/45 = 22 %.
    pub fn lean(reg: &Region, z: &[f64], share: f64) -> (f64, usize) {
        let w = reg.w;
        // The valleys: the catchments of a fresh routing, by D8 on the field.
        let (recv, order, _) = route(reg, z);
        let q = water(reg, &recv, &order);
        let mut sorted: Vec<f64> = q.clone();
        sorted.sort_by(f64::total_cmp);
        let cut = sorted[((sorted.len() - 1) as f64 * (1.0 - share)) as usize];
        let (mut near, mut all) = (0usize, 0usize);
        for y in 1..w - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                if q[i] < cut {
                    continue;
                }
                let at =
                    |dx: i64, dy: i64| z[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize];
                let gx = (at(1, -1) + 2.0 * at(1, 0) + at(1, 1))
                    - (at(-1, -1) + 2.0 * at(-1, 0) + at(-1, 1));
                let gy = (at(-1, 1) + 2.0 * at(0, 1) + at(1, 1))
                    - (at(-1, -1) + 2.0 * at(0, -1) + at(1, -1));
                if gx == 0.0 && gy == 0.0 {
                    continue;
                }
                let bearing = (-gy).atan2(-gx).to_degrees().rem_euclid(45.0);
                all += 1;
                if !(5.0..=40.0).contains(&bearing) {
                    near += 1;
                }
            }
        }
        (near as f64 / all.max(1) as f64, all)
    }

    /// What one run did.
    pub struct Run {
        pub z: Vec<f64>,
        /// The cut summed over the run, metres a cell.
        pub cut: Vec<f64>,
        /// Wall seconds a step.
        pub step_s: f64,
        /// The mean hold's largest residue at the end, metres.
        pub hold_residue_m: f64,
        /// ★ THE RISE (ruling B76, the balance): the mean height the hold added a year over the run's second half —
        /// at a steady state the region's own uplift, which equals its erosion.
        pub rise_m_yr: f64,
    }

    /// `steps` steps of `dt_yr` from `z`.
    pub fn run(reg: &Region, z0: &[f64], steps: usize, dt_yr: f64) -> Run {
        let w = reg.w;
        let cells = w * w;
        let mut z = z0.to_vec();
        let mut cut = vec![0.0f64; cells];
        let started = std::time::Instant::now();
        let mut residue = 0.0;
        let mut steep_links = 0usize;
        let mut lifted = 0.0f64;
        let mut slid = 0.0f64;
        for step in 0..steps {
            if reg.uplift_m_yr > 0.0 {
                let rise = reg.uplift_m_yr * dt_yr;
                for y in 1..w - 1 {
                    for x in 1..w - 1 {
                        let i = y * w + x;
                        if !reg.sea[i] {
                            z[i] += rise;
                        }
                    }
                }
            }
            if reg.dinf {
                let fl = dinf::route(reg, &z);
                let q = dinf::water(reg, &fl);
                dinf::erode(reg, &mut z, &mut cut, &fl, &q, dt_yr);
            } else {
                let (recv, order, filled) = route(reg, &z);
                let q = water(reg, &recv, &order);
                erode(reg, &mut z, &mut cut, &recv, &order, &filled, &q, dt_yr);
            }
            // The threshold law before the creep: every slope then stands at or under 35°, where the creep's own
            // diffusivity is at most 1.46 K, so its explicit step is stable at every dose this bench runs.
            if reg.threshold {
                slid += threshold(reg, &mut z);
            }
            if reg.creep {
                steep_links = steep_links.max(creep(reg, &mut z, dt_yr));
            }
            if !reg.threshold {
                talus(reg, &mut z);
            }
            if reg.hold_on {
                let (worst, added) = hold(reg, &mut z);
                residue = worst;
                if 2 * step >= steps {
                    lifted += added;
                }
            }
        }
        if reg.creep {
            println!(
                "  the hillslope law: at most {steep_links} links a step past nine tenths of S_c"
            );
        }
        if reg.threshold {
            println!(
                "  the threshold law: the landslides carried {:.1} m a cell on average over the run",
                slid / (w * w) as f64
            );
        }
        let late = steps - steps.div_euclid(2);
        Run {
            z,
            cut,
            step_s: started.elapsed().as_secs_f64() / steps.max(1) as f64,
            hold_residue_m: residue,
            rise_m_yr: lifted / (late.max(1) as f64 * dt_yr),
        }
    }

    fn chord(reg: &Region, slot: usize) -> f64 {
        let (dx, dy) = OFF[slot];
        if dx != 0 && dy != 0 {
            reg.cell_m * std::f64::consts::SQRT_2
        } else {
            reg.cell_m
        }
    }

    /// The priority flood from the rim and the sea; each cell's receiver and the outlets-first order; whether the
    /// flood raised the cell (a filled hollow).
    fn route(reg: &Region, z: &[f64]) -> (Vec<u32>, Vec<u32>, Vec<bool>) {
        let w = reg.w;
        let cells = w * w;
        let key = |h: f64| -> i64 { (h * 65_536.0).round() as i64 };
        let mut zf = z.to_vec();
        let mut seen = vec![false; cells];
        let mut parent = vec![u32::MAX; cells];
        let mut order: Vec<u32> = Vec::with_capacity(cells);
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
        for y in 0..w {
            for x in 0..w {
                let i = y * w + x;
                if x == 0 || y == 0 || x == w - 1 || y == w - 1 || reg.sea[i] {
                    seen[i] = true;
                    heap.push(Reverse((key(zf[i]), i as u32)));
                }
            }
        }
        while let Some(Reverse((_, c))) = heap.pop() {
            let c = c as usize;
            order.push(c as u32);
            let (cx, cy) = ((c % w) as i64, (c / w) as i64);
            for (dx, dy) in OFF {
                let (nx, ny) = (cx + i64::from(dx), cy + i64::from(dy));
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                    continue;
                }
                let m = ny as usize * w + nx as usize;
                if seen[m] {
                    continue;
                }
                seen[m] = true;
                zf[m] = zf[m].max(zf[c]);
                parent[m] = c as u32;
                heap.push(Reverse((key(zf[m]), m as u32)));
            }
        }
        let mut recv = vec![u32::MAX; cells];
        let mut filled = vec![false; cells];
        for i in 0..cells {
            filled[i] = zf[i] > z[i];
            if parent[i] == u32::MAX {
                continue;
            }
            let (cx, cy) = ((i % w) as i64, (i / w) as i64);
            let mut best = (0.0f64, u32::MAX);
            for (slot, (dx, dy)) in OFF.iter().enumerate() {
                let (nx, ny) = (cx + i64::from(*dx), cy + i64::from(*dy));
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                    continue;
                }
                let m = ny as usize * w + nx as usize;
                let s = (zf[i] - zf[m]) / chord(reg, slot);
                if s > best.0 {
                    best = (s, m as u32);
                }
            }
            recv[i] = if best.1 == u32::MAX {
                parent[i]
            } else {
                best.1
            };
        }
        (recv, order, filled)
    }

    /// `Q = A·R`, m³/yr, summed leaves first, with the rim's inflow.
    fn water(reg: &Region, recv: &[u32], order: &[u32]) -> Vec<f64> {
        let area = reg.cell_m * reg.cell_m;
        let mut q: Vec<f64> = (0..recv.len())
            .map(|i| reg.rain_m[i] * area + reg.inflow[i])
            .collect();
        for &c in order.iter().rev() {
            let r = recv[c as usize];
            if r != u32::MAX {
                q[r as usize] += q[c as usize];
            }
        }
        q
    }

    /// The joint law's implicit step, outlets first (the macro sweep's own form).
    #[allow(clippy::too_many_arguments)]
    fn erode(
        reg: &Region,
        z: &mut [f64],
        cut: &mut [f64],
        recv: &[u32],
        order: &[u32],
        filled: &[bool],
        q: &[f64],
        dt_yr: f64,
    ) {
        let w = reg.w;
        for &c in order {
            let i = c as usize;
            let r = recv[i];
            if r == u32::MAX || filled[i] || reg.sea[i] {
                continue;
            }
            let r = r as usize;
            let x0 = z[i] - z[r];
            if x0 <= 0.0 {
                continue;
            }
            let (dx, dy) = (
                (r % w) as i64 - (i % w) as i64,
                (r / w) as i64 - (i / w) as i64,
            );
            let length = if dx != 0 && dy != 0 {
                reg.cell_m * std::f64::consts::SQRT_2
            } else {
                reg.cell_m
            };
            let k = dt_yr * JOINT_K_LP * reg.rock[i] * q[i];
            let mut x = x0;
            for _ in 0..60 {
                let s = x / length;
                let sn = s.powf(JOINT_N);
                let g = x - x0 + k * sn;
                let dg = 1.0 + k * JOINT_N * sn / s / length;
                let next = (x - g / dg).max(0.0);
                let moved = (x - next).abs();
                x = next;
                if moved < 1.0e-6 {
                    break;
                }
            }
            let new = z[r] + x;
            cut[i] += z[i] - new;
            z[i] = new;
        }
    }

    /// ★ THE HILLSLOPE LAW, one explicit step (Roering, Kirchner & Dietrich 1999): between every pair of neighbours the
    /// flux `K·S/(1 − (S/S_c)²)`, weighted by the isotropic nine-point stencil (two thirds on an edge, one sixth on a
    /// corner, so no direction of the grid is favoured), the rock moved the rock received. A link at or past `S_c` holds
    /// no soil, so it creeps nothing: the talus carries it (its slope is past every repose). Sea cells hold. Returns the
    /// links past nine tenths of `S_c` (a reading: where the explicit step would need a smaller one).
    fn creep(reg: &Region, z: &mut [f64], dt_yr: f64) -> usize {
        let w = reg.w;
        let sc = vd_terrain::surface::SOIL_CRITICAL_SLOPE;
        let h2 = reg.cell_m * reg.cell_m;
        let mut delta = vec![0.0f64; z.len()];
        let mut steep = 0usize;
        // Each link once: east, south-east, south, south-west.
        for (dx, dy, weight, len) in [
            (1i64, 0i64, 2.0 / 3.0, reg.cell_m),
            (1, 1, 1.0 / 6.0, reg.cell_m * std::f64::consts::SQRT_2),
            (0, 1, 2.0 / 3.0, reg.cell_m),
            (-1, 1, 1.0 / 6.0, reg.cell_m * std::f64::consts::SQRT_2),
        ] {
            for y in 0..w {
                for x in 0..w {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                        continue;
                    }
                    let (i, m) = (y * w + x, ny as usize * w + nx as usize);
                    if reg.sea[i] || reg.sea[m] {
                        continue;
                    }
                    let dz = z[i] - z[m];
                    let ratio = (dz.abs() / len) / sc;
                    steep += usize::from(ratio > 0.9);
                    if ratio >= 1.0 {
                        continue;
                    }
                    let moved = dt_yr * CREEP_K_M2_YR * weight * dz / h2 / (1.0 - ratio * ratio);
                    delta[i] -= moved;
                    delta[m] += moved;
                }
            }
        }
        for (h, d) in z.iter_mut().zip(&delta) {
            *h += d;
        }
        steep
    }

    /// ★ THE THRESHOLD HILLSLOPE, exactly: each cell's height becomes the lowest of its own and every neighbour's new
    /// height plus the repose's rise over their chord — the lower envelope of the 35° cones under every cell, found from
    /// the lowest cell up (a priority sweep, so the answer does not depend on the order of the cells and no step can
    /// overshoot). Sea and rim cells are the base level and keep their heights. Returns the rock the landslides carried
    /// into the channels, metres summed over the cells.
    fn threshold(reg: &Region, z: &mut [f64]) -> f64 {
        let w = reg.w;
        let tan = vd_terrain::body::TAN_REPOSE;
        let key = |h: f64| -> i64 { (h * 65_536.0).round() as i64 };
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> =
            (0..w * w).map(|i| Reverse((key(z[i]), i as u32))).collect();
        let mut done = vec![false; w * w];
        let mut slid = 0.0f64;
        while let Some(Reverse((_, c))) = heap.pop() {
            let c = c as usize;
            if done[c] {
                continue;
            }
            done[c] = true;
            let (cx, cy) = ((c % w) as i64, (c / w) as i64);
            for (slot, (dx, dy)) in OFF.iter().enumerate() {
                let (nx, ny) = (cx + i64::from(*dx), cy + i64::from(*dy));
                if nx <= 0 || ny <= 0 || nx >= w as i64 - 1 || ny >= w as i64 - 1 {
                    continue;
                }
                let m = ny as usize * w + nx as usize;
                if done[m] || reg.sea[m] {
                    continue;
                }
                let cap = z[c] + tan * chord(reg, slot);
                if z[m] > cap {
                    slid += z[m] - cap;
                    z[m] = cap;
                    heap.push(Reverse((key(cap), m as u32)));
                }
            }
        }
        slid
    }

    /// The talus: half of each cell's excess over its repose toward its steepest lower neighbour, in pairs.
    fn talus(reg: &Region, z: &mut [f64]) {
        let w = reg.w;
        let mut delta = vec![0.0f64; z.len()];
        for y in 0..w {
            for x in 0..w {
                let i = y * w + x;
                let mut best = (0.0f64, usize::MAX, 0.0f64);
                for (slot, (dx, dy)) in OFF.iter().enumerate() {
                    let (nx, ny) = (x as i64 + i64::from(*dx), y as i64 + i64::from(*dy));
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                        continue;
                    }
                    let m = ny as usize * w + nx as usize;
                    let l = chord(reg, slot);
                    let excess = (z[i] - z[m]) - reg.repose[i] * l;
                    if excess > best.0 {
                        best = (excess, m, l);
                    }
                }
                if best.1 != usize::MAX {
                    let moved = best.0 / 2.0 / 2.0;
                    delta[i] -= moved;
                    delta[best.1] += moved;
                }
            }
        }
        for (h, d) in z.iter_mut().zip(&delta) {
            *h += d;
        }
    }

    /// The mean hold: each whole node's mean back to its row, by the Catmull-Rom of the residues; returns the largest
    /// residue it corrected, metres.
    fn hold(reg: &Region, z: &mut [f64]) -> (f64, f64) {
        let (w, n, s) = (reg.w, reg.nodes, reg.cells_per_node);
        let first = reg.halo % s;
        let mut residue = vec![0.0f64; n * n];
        let mut worst = 0.0f64;
        for nj in 0..n {
            for ni in 0..n {
                let (x0, y0) = (first + ni * s, first + nj * s);
                if x0 + s > w || y0 + s > w {
                    continue;
                }
                let mut sum = 0.0;
                for y in y0..y0 + s {
                    for x in x0..x0 + s {
                        sum += z[y * w + x];
                    }
                }
                let d = reg.node_z[nj * n + ni] - sum / (s * s) as f64;
                residue[nj * n + ni] = d;
                worst = worst.max(d.abs());
            }
        }
        let at = |i: i64, j: i64| {
            residue[(j.clamp(0, n as i64 - 1) as usize) * n + i.clamp(0, n as i64 - 1) as usize]
        };
        let mut lifted = 0.0f64;
        for y in 0..w {
            for x in 0..w {
                let u = (x as f64 + 0.5 - first as f64) / s as f64 - 0.5;
                let v = (y as f64 + 0.5 - first as f64) / s as f64 - 0.5;
                let (iu, fu) = (u.floor() as i64, u - u.floor());
                let (iv, fv) = (v.floor() as i64, v - v.floor());
                let mut rows = [0.0f64; 4];
                for (r, slot) in rows.iter_mut().enumerate() {
                    let j = iv + r as i64 - 1;
                    *slot =
                        catmull_rom([at(iu - 1, j), at(iu, j), at(iu + 1, j), at(iu + 2, j)], fu);
                }
                let added = catmull_rom(rows, fv);
                z[y * w + x] += added;
                lifted += added;
            }
        }
        (worst, lifted / (w * w) as f64)
    }

    /// ★ D∞ (Tarboton 1997, *A new method for the determination of flow directions and upslope areas in grid digital
    /// elevation models*, Water Resources Research 33, 309–319): on the filled surface, each cell's water leaves down the
    /// steepest of its eight triangular facets, shared between the facet's two neighbours by the angle — `1 − r/(π/4)`
    /// to the facet's side neighbour, `r/(π/4)` to its diagonal one. A flat cell (no facet falls) follows the flood.
    /// The joint law's implicit step then reads the WEIGHTED slope `S(z) = p₁(z − z₁)/d₁ + p₂(z − z₂)/d₂` over the two
    /// receivers' new heights, its root by Newton as the macro's.
    mod dinf {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        use vd_terrain::solve::{JOINT_K_LP, JOINT_N};

        use super::Region;

        /// The facets: the side neighbour and the diagonal neighbour beside it (x right, y down).
        const FACETS: [((i64, i64), (i64, i64)); 8] = [
            ((1, 0), (1, -1)),
            ((0, -1), (1, -1)),
            ((0, -1), (-1, -1)),
            ((-1, 0), (-1, -1)),
            ((-1, 0), (-1, 1)),
            ((0, 1), (-1, 1)),
            ((0, 1), (1, 1)),
            ((1, 0), (1, 1)),
        ];

        /// The flood and the two receivers of every cell.
        pub struct Flow {
            pub order: Vec<u32>,
            pub filled: Vec<bool>,
            /// The side and the diagonal receiver (`u32::MAX` none) and their shares.
            pub e1: Vec<u32>,
            pub e2: Vec<u32>,
            pub p1: Vec<f64>,
            pub p2: Vec<f64>,
            /// ★ With `mfd`: every lower neighbour of the tilted filled surface and its share (Freeman 1991,
            /// `max(0, S)^1.1` over the sum), and the order is the tilted surface's, lowest first. Empty otherwise.
            pub many: Vec<Vec<(u32, f64)>>,
        }

        /// Freeman 1991's exponent on the slope: a numerical constant of the routing, not a physical calibration.
        const FREEMAN_P: f64 = 1.1;

        pub fn route(reg: &Region, z: &[f64]) -> Flow {
            let w = reg.w;
            let cells = w * w;
            let key = |h: f64| -> i64 { (h * 65_536.0).round() as i64 };
            let mut zf = z.to_vec();
            let mut seen = vec![false; cells];
            let mut parent = vec![u32::MAX; cells];
            let mut order: Vec<u32> = Vec::with_capacity(cells);
            let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
            for y in 0..w {
                for x in 0..w {
                    let i = y * w + x;
                    if x == 0 || y == 0 || x == w - 1 || y == w - 1 || reg.sea[i] {
                        seen[i] = true;
                        heap.push(Reverse((key(zf[i]), i as u32)));
                    }
                }
            }
            while let Some(Reverse((_, c))) = heap.pop() {
                let c = c as usize;
                order.push(c as u32);
                let (cx, cy) = ((c % w) as i64, (c / w) as i64);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (cx + dx, cy + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                            continue;
                        }
                        let m = ny as usize * w + nx as usize;
                        if seen[m] {
                            continue;
                        }
                        seen[m] = true;
                        zf[m] = zf[m].max(zf[c]);
                        parent[m] = c as u32;
                        heap.push(Reverse((key(zf[m]), m as u32)));
                    }
                }
            }
            // ★ THE FLATS (Barnes, Lehman & Mulla 2014, the macro's phase-1 rule at this level): on every flat of the filled
            // surface (no lower neighbour), the summed chord TOWARD its drains (a same-height cell that falls) twice, and
            // AWAY from its higher rim once; the surface is tilted by that potential at a tenth of a micrometre a metre of
            // chord (a centimetre over a 100 km flat), so D∞ finds a direction on it that is no grid line.
            let zf = tilt_flats(reg, &zf);
            let d = reg.cell_m;
            let quarter = std::f64::consts::FRAC_PI_4;
            let mut flow = Flow {
                order,
                filled: (0..cells).map(|i| zf[i] > z[i]).collect(),
                e1: vec![u32::MAX; cells],
                e2: vec![u32::MAX; cells],
                p1: vec![0.0; cells],
                p2: vec![0.0; cells],
                many: Vec::new(),
            };
            for i in 0..cells {
                if parent[i] == u32::MAX {
                    continue;
                }
                let (cx, cy) = ((i % w) as i64, (i / w) as i64);
                let mut best = (0.0f64, u32::MAX, u32::MAX, 0.0f64);
                for ((ax, ay), (bx, by)) in FACETS {
                    let (x1, y1, x2, y2) = (cx + ax, cy + ay, cx + bx, cy + by);
                    if x1 < 0
                        || y1 < 0
                        || x1 >= w as i64
                        || y1 >= w as i64
                        || x2 < 0
                        || y2 < 0
                        || x2 >= w as i64
                        || y2 >= w as i64
                    {
                        continue;
                    }
                    let (j1, j2) = (y1 as usize * w + x1 as usize, y2 as usize * w + x2 as usize);
                    let s1 = (zf[i] - zf[j1]) / d;
                    let s2 = (zf[j1] - zf[j2]) / d;
                    let (mut r, mut s) = (s2.atan2(s1), (s1 * s1 + s2 * s2).sqrt());
                    if r < 0.0 {
                        r = 0.0;
                        s = s1;
                    } else if r > quarter {
                        r = quarter;
                        s = (zf[i] - zf[j2]) / (d * std::f64::consts::SQRT_2);
                    }
                    if s > best.0 {
                        best = (s, j1 as u32, j2 as u32, r);
                    }
                }
                if best.1 == u32::MAX {
                    // A flat: the flood's own way.
                    flow.e1[i] = parent[i];
                    flow.p1[i] = 1.0;
                } else {
                    let share2 = best.3 / quarter;
                    if reg.rho8 {
                        // ★ Rho8 (Fairfield & Leymarie 1991): ONE receiver, the diagonal with the chance `r/(π/4)`
                        // by the cell's own hash, else the side; the expected bearing is the facet's, and no water
                        // spreads.
                        let h = super::hash01(i as u64);
                        flow.e1[i] = if h < share2 { best.2 } else { best.1 };
                        flow.p1[i] = 1.0;
                    } else {
                        flow.e1[i] = best.1;
                        flow.p1[i] = 1.0 - share2;
                        if share2 > 0.0 {
                            flow.e2[i] = best.2;
                            flow.p2[i] = share2;
                        }
                    }
                }
            }
            if reg.mfd {
                // ★ FREEMAN 1991 on the tilted filled surface: every strictly lower neighbour takes `S^p` of the water;
                // the order is the surface's own, lowest first, so every receiver is read before its giver.
                let mut many = vec![Vec::new(); cells];
                for (i, out) in many.iter_mut().enumerate() {
                    if parent[i] == u32::MAX {
                        continue;
                    }
                    let (cx, cy) = ((i % w) as i64, (i / w) as i64);
                    let mut total = 0.0;
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let (nx, ny) = (cx + dx, cy + dy);
                            if (dx == 0 && dy == 0)
                                || nx < 0
                                || ny < 0
                                || nx >= w as i64
                                || ny >= w as i64
                            {
                                continue;
                            }
                            let m = ny as usize * w + nx as usize;
                            let len = if dx != 0 && dy != 0 {
                                d * std::f64::consts::SQRT_2
                            } else {
                                d
                            };
                            let slope = (zf[i] - zf[m]) / len;
                            if slope > 0.0 {
                                let share = slope.powf(FREEMAN_P);
                                total += share;
                                out.push((m as u32, share));
                            }
                        }
                    }
                    if total > 0.0 {
                        for (_, share) in out.iter_mut() {
                            *share /= total;
                        }
                    } else {
                        out.push((parent[i], 1.0));
                    }
                }
                let mut order: Vec<u32> = (0..cells as u32).collect();
                order.sort_by(|&a, &b| zf[a as usize].total_cmp(&zf[b as usize]));
                flow.order = order;
                flow.many = many;
            }
            flow
        }

        /// The filled surface tilted on its flats by Barnes' potential `Φ = 2·T + (A_max − A)` (summed chords, metres),
        /// `T` toward the flat's drains and `A` away from its higher rim, at [`TILT_PER_M`] of height a metre of `Φ`.
        fn tilt_flats(reg: &Region, zf: &[f64]) -> Vec<f64> {
            let w = reg.w;
            let cells = w * w;
            let near = |i: usize| -> Vec<(usize, f64)> {
                let (cx, cy) = ((i % w) as i64, (i / w) as i64);
                let mut v = Vec::with_capacity(8);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (cx + dx, cy + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= w as i64 {
                            continue;
                        }
                        let l = if dx != 0 && dy != 0 {
                            reg.cell_m * std::f64::consts::SQRT_2
                        } else {
                            reg.cell_m
                        };
                        v.push((ny as usize * w + nx as usize, l));
                    }
                }
                v
            };
            let rim = |i: usize| -> bool {
                let (x, y) = (i % w, i / w);
                x == 0 || y == 0 || x == w - 1 || y == w - 1
            };
            // A flat cell: not an outlet, and no neighbour falls from it.
            let flat: Vec<bool> = (0..cells)
                .map(|i| !rim(i) && !reg.sea[i] && near(i).iter().all(|&(m, _)| zf[m] >= zf[i]))
                .collect();
            // Dijkstra over the flat cells from a set of sources at distance 0, through cells of the same height.
            let spread = |sources: &[usize]| -> Vec<f64> {
                let mut dist = vec![f64::INFINITY; cells];
                let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
                for &s in sources {
                    dist[s] = 0.0;
                    heap.push(Reverse((0, s as u32)));
                }
                while let Some(Reverse((dk, u))) = heap.pop() {
                    let u = u as usize;
                    let du = dk as f64 / 1_000.0;
                    if du > dist[u] {
                        continue;
                    }
                    for (m, l) in near(u) {
                        if !flat[m] || zf[m] != zf[u] {
                            continue;
                        }
                        let nd = du + l;
                        if nd < dist[m] {
                            dist[m] = nd;
                            heap.push(Reverse(((nd * 1_000.0) as u64, m as u32)));
                        }
                    }
                }
                dist
            };
            // The drains: a flat cell beside a cell of its own height that falls (or is an outlet).
            let drains: Vec<usize> = (0..cells)
                .filter(|&i| flat[i] && near(i).iter().any(|&(m, _)| zf[m] == zf[i] && (!flat[m])))
                .collect();
            // The higher rim: a flat cell beside a higher cell.
            let highs: Vec<usize> = (0..cells)
                .filter(|&i| flat[i] && near(i).iter().any(|&(m, _)| zf[m] > zf[i]))
                .collect();
            let toward = spread(&drains);
            let away = spread(&highs);
            // A_max a flat: the largest reached `away` among the cells of each height-connected flat — approximated by the
            // largest over the cells of the same height reached from the same drains (one flat each, as their spread).
            let a_max = away
                .iter()
                .copied()
                .filter(|a| a.is_finite())
                .fold(0.0f64, f64::max);
            let mut out = zf.to_vec();
            for i in 0..cells {
                if !flat[i] || !toward[i].is_finite() {
                    continue;
                }
                let lift = if away[i].is_finite() {
                    a_max - away[i]
                } else {
                    0.0
                };
                out[i] = zf[i] + TILT_PER_M * (2.0 * toward[i] + lift);
            }
            out
        }

        /// The flats' tilt: height a metre of Barnes' potential (a tenth of a micrometre).
        const TILT_PER_M: f64 = 1.0e-7;

        /// `Q = A·R` shared down both receivers, leaves first, with the rim's inflow.
        pub fn water(reg: &Region, f: &Flow) -> Vec<f64> {
            let area = reg.cell_m * reg.cell_m;
            let mut q: Vec<f64> = (0..f.e1.len())
                .map(|i| reg.rain_m[i] * area + reg.inflow[i])
                .collect();
            for &c in f.order.iter().rev() {
                let i = c as usize;
                if !f.many.is_empty() {
                    for &(m, share) in &f.many[i] {
                        q[m as usize] += q[i] * share;
                    }
                    continue;
                }
                if f.e1[i] != u32::MAX {
                    q[f.e1[i] as usize] += q[i] * f.p1[i];
                }
                if f.e2[i] != u32::MAX {
                    q[f.e2[i] as usize] += q[i] * f.p2[i];
                }
            }
            q
        }

        /// The implicit step on the weighted slope, outlets first.
        pub fn erode(
            reg: &Region,
            z: &mut [f64],
            cut: &mut [f64],
            f: &Flow,
            q: &[f64],
            dt_yr: f64,
        ) {
            let w = reg.w;
            let chord = |a: usize, b: usize| -> f64 {
                let (dx, dy) = (
                    (a % w) as i64 - (b % w) as i64,
                    (a / w) as i64 - (b / w) as i64,
                );
                if dx != 0 && dy != 0 {
                    reg.cell_m * std::f64::consts::SQRT_2
                } else {
                    reg.cell_m
                }
            };
            for &c in &f.order {
                let i = c as usize;
                if f.e1[i] == u32::MAX || f.filled[i] || reg.sea[i] {
                    continue;
                }
                // S(z) = a·z − b over the receivers' new heights.
                let (mut a, mut b) = (0.0f64, 0.0f64);
                let pairs: Vec<(u32, f64)> = if f.many.is_empty() {
                    vec![(f.e1[i], f.p1[i]), (f.e2[i], f.p2[i])]
                } else {
                    f.many[i].clone()
                };
                for (e, p) in pairs {
                    if e != u32::MAX && p > 0.0 {
                        let l = chord(i, e as usize);
                        a += p / l;
                        b += p * z[e as usize] / l;
                    }
                }
                if a <= 0.0 {
                    continue;
                }
                let floor = b / a;
                if z[i] <= floor {
                    continue;
                }
                let k = dt_yr * JOINT_K_LP * reg.rock[i] * q[i];
                let mut x = z[i];
                for _ in 0..60 {
                    let s = (a * x - b).max(0.0);
                    let sn = s.powf(JOINT_N);
                    let g = x - z[i] + k * sn;
                    let dg = 1.0 + k * JOINT_N * a * if s > 0.0 { sn / s } else { 0.0 };
                    let next = (x - g / dg).max(floor);
                    let moved = (x - next).abs();
                    x = next;
                    if moved < 1.0e-6 {
                        break;
                    }
                }
                cut[i] += z[i] - x;
                z[i] = x;
            }
        }
    }
}

/// ★ THE STREAM LAW'S STEADY STATE ON IRREGULAR POINTS (the owner, 2026-10-11: "use whatever you can, show me the
/// pictures"). The point graph is fastlem's (T. Tada, MPL-2.0, linked unmodified: random sites relaxed once by Lloyd,
/// their Voronoi areas, the Delaunay edges, the frame's edge sites) — no grid, so no grid direction. The heights are the
/// joint law's own steady state (`U = K_lp·Q·S^2.2`, so `S = (U/(K_lp·Q))^(1/2.2)`, Q = rain × the area that drains
/// through the point), summed outward from the outlets along the flow tree (Steer 2021; Cordonnier et al. 2016). No
/// slope stands steeper than the rock mass's repose (35°, the threshold hillslope law). The tree is re-found from the
/// new heights until it stops changing. THE MACRO WORLD SAYS HOW HIGH: each node's uplift is solved so the fine
/// ground's mean over the node equals the node's own mean (the 8 km fact), so no rise rate is assumed.
mod lem {
    use fastlem::core::traits::Model;
    use fastlem::models::surface::builder::TerrainModel2DBulider;
    use fastlem::models::surface::sites::Site2D;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::time::Instant;
    use vd_terrain::solve::{JOINT_K_LP, JOINT_N};

    /// The solved ground rastered at `cell_m`, `w` cells a side, and what a finer window reads from it: the points,
    /// their heights, the water through each, each point's receiver, each node's ln U, and the point graph's sampler.
    pub struct Out {
        pub raster: Vec<f64>,
        pub w: usize,
        pub sites: Vec<Site2D>,
        pub q: Vec<f64>,
        pub recv: Vec<usize>,
        pub ln_u: Vec<f64>,
        pub terrain: fastlem::models::surface::terrain::Terrain2D,
    }

    /// One fine window solved inside a coarse solve: its raster (`w` × `w` cells of `cell_m`) from `(x0, y0)`.
    pub struct Window {
        pub raster: Vec<f64>,
        pub w: usize,
        pub x0: f64,
        pub y0: f64,
        pub cell_m: f64,
    }

    /// ★ THE ROCK'S OWN SCATTER (B80 step 2, a NAMED scatter around the law's erodibility — rulings T9, B23): rock
    /// strength varies inside every formation, and the joint law's erodibility follows the tensile strength's inverse
    /// square (Sklar & Dietrich 2001), so the erodibility is the law's value times `exp(σ·g)`, `g` a smooth field of unit
    /// spread in every direction from 16 km down to 32 m (octaves of value noise, each a half of the one above), the SAME
    /// function of the position at every point spacing. `σ = ln 4 / 2` (a factor of four between strong and weak rock of
    /// one formation, inside the 1–20 MPa span Sklar & Dietrich measured across rocks): ASSUMED, its source owed.
    pub fn rock_scatter(x: f64, y: f64) -> f64 {
        let sigma = 4.0f64.ln() / 2.0;
        let smooth = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (mut sum, mut norm, mut wave, mut amp, mut octave) =
            (0.0f64, 0.0f64, 16_000.0f64, 1.0f64, 0u64);
        while wave >= 32.0 {
            let lattice = |a: i64, b: i64| -> f64 {
                let key = (octave << 52)
                    ^ ((a as u64 & 0x3FF_FFFF) << 26)
                    ^ (b as u64 & 0x3FF_FFFF)
                    ^ 0x0005_EED0_F0C4;
                2.0 * super::r2::hash01(key) - 1.0
            };
            // Each octave turned by its own angle so no two share the lattice's directions.
            let turn = 0.618_034 * octave as f64 * std::f64::consts::TAU;
            let (c, s) = (turn.cos(), turn.sin());
            let (u, v) = ((c * x - s * y) / wave, (s * x + c * y) / wave);
            let (a, b) = (u.floor(), v.floor());
            let (fu, fv) = (smooth(u - a), smooth(v - b));
            let (a, b) = (a as i64, b as i64);
            let top = lattice(a, b) * (1.0 - fu) + lattice(a + 1, b) * fu;
            let bottom = lattice(a, b + 1) * (1.0 - fu) + lattice(a + 1, b + 1) * fu;
            sum += amp * (top * (1.0 - fv) + bottom * fv);
            norm += amp * amp;
            wave /= 2.0;
            amp *= 0.7;
            octave += 1;
        }
        // Value noise of unit lattice spread has a standard deviation near 0.44 a unit octave.
        (sigma * sum / (norm.sqrt() * 0.44)).exp()
    }

    /// The rock's factor at a point: one rock unless the bench asks for the stated scatter (`--lem-rock`).
    fn rock(x: f64, y: f64) -> f64 {
        if ROCK_SCATTER.load(std::sync::atomic::Ordering::Relaxed) {
            rock_scatter(x, y)
        } else {
            1.0
        }
    }

    /// The bench's switch for the rock's stated scatter (`--lem-rock`; off: one rock, the lawful default).
    pub static ROCK_SCATTER: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// The place's stated rise, nanometres a year (`--uplift-um` × 1 000, or the solve's present rise). Valley spacing reads
    /// the absolute rate (Perron's ℓc), while the heights fix only U over K; the relaxation scales both together.
    pub static RISE_NM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    /// Whether a rise is stated at all (a stated zero is a still place: no rise, no cut, the creep alone).
    pub static RISE_SET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// The mean hold on each domain against the coarser cells (`--lem-hold`).
    pub static HOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// The cell the hold reads, metres (`--lem-hold-m`; the coarse solve's spacing, 128 m, by default).
    pub static HOLD_CELL_M: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(128);
    /// The area for the slope laws spread over every lower neighbour (`--lem-mfd`): Freeman 1991, the cure of the comb.
    pub static SHARED_AREA: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// The threshold slope the steady solve reads, thousandths (`--l1-sc`; tan 35° = 0.700 until the design's S_c = 1.25,
    /// review A6, is benched beside it).
    pub static THRESHOLD_MILLI: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(700);
    fn threshold() -> f64 {
        THRESHOLD_MILLI.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000.0
    }
    pub fn threshold_pub() -> f64 {
        threshold()
    }
    /// The rise and the tree in ONE loop (`--l1-joint`).
    pub static JOINT_LOOP: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    /// Each tree pass's moves printed (`--l1-trace`).
    pub static TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// The tree passes stop when no height moves more than a quarter metre (`--l1-settle`).
    pub static SETTLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// The 128 m mean hold as ONE LINEAR SOLVE on a fixed tree (`--l1-linear`; the design §3.3, review B-B4).
    pub static L1_LINEAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// The partition of unity's width for the linear hold, hundredths of a node (`--l1-sigma`).
    pub static L1_SIGMA_C: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(70);

    /// The steady form in one tree solve in place of the relaxation (`--lem-steady`).
    pub static STEADY_FORM: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// The fine window's relaxation steps (`--lem-relax-steps`, 0: the tree solve alone).
    pub static RELAX_STEPS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(400);

    /// THE STEADY STATE on a point graph, the tree re-found from the heights until it settles (or `max_passes`): each
    /// outlet holds its `base`; every other point stands over its receiver by the joint law's slope for its water
    /// (`water` its own inflow, m³/yr, summed down the tree) and uplift `u`, never steeper than the threshold. Returns the
    /// passes, the receivers still changing, the water through each point and each receiver.
    #[allow(clippy::too_many_arguments)]
    fn steady(
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
        outlet: &[bool],
        base: &[f64],
        water: &[f64],
        u: &[f64],
        h: &mut [f64],
        max_passes: usize,
    ) -> (usize, usize, Vec<f64>, Vec<usize>) {
        let n = h.len();
        let tan = threshold();
        let ns = |t: Instant, slot: &std::sync::atomic::AtomicU64| {
            slot.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        };
        SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc = &mut *guard;
            sc.last.clear();
            sc.last.resize(n, usize::MAX);
            sc.prev.clear();
            sc.prev.extend_from_slice(h);
            sc.q.clear();
            sc.q.extend_from_slice(water);
            let (mut passes, mut still) = (0usize, 0usize);
            for _ in 0..max_passes {
                passes += 1;
                let t = Instant::now();
                tree_into(h, outlet, graph, sc);
                ns(t, &T_TREE);
                let t = Instant::now();
                sc.q.copy_from_slice(water);
                for &c in sc.order.iter().rev() {
                    let r = sc.recv[c];
                    if r != usize::MAX {
                        sc.q[r] += sc.q[c];
                    }
                }
                ns(t, &T_WATER);
                let t = Instant::now();
                for &c in &sc.order {
                    let r = sc.recv[c];
                    if outlet[c] || r == usize::MAX {
                        h[c] = base[c];
                        continue;
                    }
                    let s = law_power(u[c] / (JOINT_K_LP * sc.q[c].max(f64::MIN_POSITIVE))).min(tan);
                    h[c] = h[r] + s * sc.len[c];
                }
                ns(t, &T_HEIGHT);
                still = sc.recv.iter().zip(&sc.last).filter(|(a, b)| a != b).count();
                {
                    let mut acc = 0xcbf2_9ce4_8422_2325u64;
                    for &r in &sc.recv {
                        acc = (acc ^ r as u64).wrapping_mul(0x100_0000_01b3);
                    }
                    LAST_TREE_HASH.store(acc, std::sync::atomic::Ordering::Relaxed);
                }
                std::mem::swap(&mut sc.last, &mut sc.recv);
                // Settled: the tree repeats, or no height moved more than the hold's quarter metre (receivers that flip on
                // a tie move nothing).
                let moved = h.iter().zip(&sc.prev).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut d: Vec<f64> = h.iter().zip(&sc.prev).map(|(a, b)| (a - b).abs()).collect();
                    d.sort_by(f64::total_cmp);
                    let big = d.iter().filter(|&&x| x > 0.25).count();
                    println!(
                        "      pass {passes}: {still} receivers changed; heights moved: max {moved:.2} m, 99.9th {:.2} m, 99th {:.3} m; {big} points over 0.25 m",
                        d[d.len() * 999 / 1000],
                        d[d.len() * 99 / 100]
                    );
                }
                sc.prev.copy_from_slice(h);
                if still == 0 || (SETTLE.load(std::sync::atomic::Ordering::Relaxed) && moved <= 0.25) {
                    break;
                }
            }
            if RETURN_LISTS.load(std::sync::atomic::Ordering::Relaxed) {
                (passes, still, sc.q.clone(), sc.last.clone())
            } else {
                (passes, still, Vec::new(), Vec::new())
            }
        })
    }

    /// The 8 km field's Catmull-Rom at a point, metres from the region's north-west corner.
    fn smooth(grid: &[f64], nw: usize, node_m: f64, x: f64, y: f64) -> f64 {
        let (u, v) = (x / node_m - 0.5, y / node_m - 0.5);
        let (iu, fu) = (u.floor() as i64, u - u.floor());
        let (iv, fv) = (v.floor() as i64, v - v.floor());
        let at = |i: i64, j: i64| {
            grid[(j.clamp(0, nw as i64 - 1) as usize) * nw + i.clamp(0, nw as i64 - 1) as usize]
        };
        let mut rows = [0.0f64; 4];
        for (r, slot) in rows.iter_mut().enumerate() {
            let j = iv + r as i64 - 1;
            *slot =
                super::catmull_rom([at(iu - 1, j), at(iu, j), at(iu + 1, j), at(iu + 2, j)], fu);
        }
        super::catmull_rom(rows, fv)
    }

    /// The 8 km field's Catmull-Rom at a point (today's smooth field, for the side-by-side picture).
    pub fn smooth_at(grid: &[f64], nw: usize, node_m: f64, x: f64, y: f64) -> f64 {
        smooth(grid, nw, node_m, x, y)
    }

    /// A node grid read with the SAME WEIGHT IN EVERY DIRECTION: each node centre weighted by a Gaussian of its distance
    /// (σ = 0.7 node), so one low node reads as a round hollow, never a square aligned to the lattice.
    fn radial(grid: &[f64], nw: usize, node_m: f64, x: f64, y: f64) -> f64 {
        let (u, v) = (x / node_m - 0.5, y / node_m - 0.5);
        let sigma2 = 2.0 * 0.7 * 0.7;
        // The window is SYMMETRIC about the point and 3.5 nodes wide each way, so a node enters or leaves it only at a
        // weight of e^(−3.5²/(2·0.7²)) ≈ 4·10⁻⁶ (a window from two nodes behind to three ahead dropped a node at weight
        // 0.017 each time a point crossed a node centre: straight lines along the centres' lines). The Gaussian is
        // separable: one weight a node a direction.
        let (u0, u1) = (
            (u - KERNEL_REACH).ceil() as i64,
            (u + KERNEL_REACH).floor() as i64,
        );
        let (v0, v1) = (
            (v - KERNEL_REACH).ceil() as i64,
            (v + KERNEL_REACH).floor() as i64,
        );
        let mut wu = [0.0f64; 8];
        let mut wv = [0.0f64; 8];
        for (t, i) in (u0..=u1).enumerate() {
            wu[t] = (-(u - i as f64).powi(2) / sigma2).exp();
        }
        for (t, j) in (v0..=v1).enumerate() {
            wv[t] = (-(v - j as f64).powi(2) / sigma2).exp();
        }
        let (mut sum, mut weight) = (0.0f64, 0.0f64);
        for (b, j) in (v0..=v1).enumerate() {
            let row = (j.clamp(0, nw as i64 - 1) as usize) * nw;
            for (a, i) in (u0..=u1).enumerate() {
                let wgt = wu[a] * wv[b];
                sum += wgt * grid[row + i.clamp(0, nw as i64 - 1) as usize];
                weight += wgt;
            }
        }
        sum / weight
    }

    /// ★ WHERE INSIDE ITS NODE A MACRO FEATURE RUNS (`--l1-warp`): a node is a mean over its square, so the 8 km world
    /// cannot say where inside the square a valley's axis or a ridge's crest lies; reading its fields at the node centres
    /// puts every such axis on the straight line through the centres. The reads go instead through a smooth displacement
    /// drawn from the position alone (the same field for every tile: no seam), up to about half a node with a wavelength
    /// of one node — the node's own footprint. The node means stay exact (they are taken over the true squares and held).
    pub fn warp(x: f64, y: f64, node_m: f64) -> (f64, f64) {
        if !WARP.load(std::sync::atomic::Ordering::Relaxed) {
            return (x, y);
        }
        let field = |salt: u64| -> f64 {
            let smooth = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            let (mut sum, mut norm, mut wave, mut amp) = (0.0f64, 0.0f64, 2.0 * node_m, 1.0f64);
            for octave in 0..2u64 {
                let lattice = |a: i64, b: i64| -> f64 {
                    let key = (octave << 52)
                        ^ ((a as u64 & 0x3FF_FFFF) << 26)
                        ^ (b as u64 & 0x3FF_FFFF)
                        ^ salt;
                    2.0 * super::r2::hash01(key) - 1.0
                };
                // Each octave turned by its own angle so the field shares no direction with the node lattice.
                let turn = 0.618_034 * (octave as f64 + 0.5) * std::f64::consts::TAU;
                let (c, sn) = (turn.cos(), turn.sin());
                let (u, v) = ((c * x - sn * y) / wave, (sn * x + c * y) / wave);
                let (a, b) = (u.floor(), v.floor());
                let (fu, fv) = (smooth(u - a), smooth(v - b));
                let (a, b) = (a as i64, b as i64);
                let top = lattice(a, b) * (1.0 - fu) + lattice(a + 1, b) * fu;
                let bottom = lattice(a, b + 1) * (1.0 - fu) + lattice(a + 1, b + 1) * fu;
                sum += amp * (top * (1.0 - fv) + bottom * fv);
                norm += amp;
                wave /= 2.0;
                amp *= 0.5;
            }
            sum / norm
        };
        (
            x + 0.5 * node_m * field(0xA110_F0E5),
            y + 0.5 * node_m * field(0xA110_F1E5),
        )
    }
    /// The reads of the macro fields go through `warp` (`--l1-warp`).
    pub static WARP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// THE ADDITIVE HOLD'S KERNEL, built once: each point's Gaussian weights over the held cells (σ half a cell, reaching
    /// 2.5 cells each way so a cell enters or leaves only at a weight of e^(−2.5²/(2·0.5²)) ≈ 4·10⁻⁶), normalised. `at(i)`
    /// gives the point's place in cells (centres at whole numbers). Returns each point's start in the flat lists, the
    /// cells and the weights.
    fn hold_kernel(
        n: usize,
        at: impl Fn(usize) -> (f64, f64),
        ncx: usize,
        ncy: usize,
        held: &[bool],
    ) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
        const REACH: f64 = 2.5;
        let (mut start, mut cells, mut weights) =
            (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
        for i in 0..n {
            start.push(cells.len());
            let (u, v) = at(i);
            let first = cells.len();
            let mut wsum = 0.0f64;
            for cy in ((v - REACH).ceil() as i64)..=((v + REACH).floor() as i64) {
                for cx in ((u - REACH).ceil() as i64)..=((u + REACH).floor() as i64) {
                    if cx < 0
                        || cy < 0
                        || cx >= ncx as i64
                        || cy >= ncy as i64
                        || !held[cy as usize * ncx + cx as usize]
                    {
                        continue;
                    }
                    let wgt = (-((u - cx as f64).powi(2) + (v - cy as f64).powi(2))
                        / (2.0 * 0.5 * 0.5))
                        .exp();
                    cells.push((cy as usize * ncx + cx as usize) as u32);
                    weights.push(wgt as f32);
                    wsum += wgt;
                }
            }
            for w in &mut weights[first..] {
                *w = (f64::from(*w) / wsum.max(f64::MIN_POSITIVE)) as f32;
            }
        }
        start.push(cells.len());
        (start, cells, weights)
    }

    /// How far a Gaussian read reaches each way, in cells: a cell enters or leaves the window only where its weight is
    /// negligible, so the read has no step where a point crosses a cell's centre.
    const KERNEL_REACH: f64 = 3.5;

    /// The bilinear read of a node grid at a point (the uplift and the rain: no overshoot, never negative).
    fn linear(grid: &[f64], nw: usize, node_m: f64, x: f64, y: f64) -> f64 {
        let (u, v) = (x / node_m - 0.5, y / node_m - 0.5);
        let (iu, fu) = (u.floor() as i64, u - u.floor());
        let (iv, fv) = (v.floor() as i64, v - v.floor());
        let at = |i: i64, j: i64| {
            grid[(j.clamp(0, nw as i64 - 1) as usize) * nw + i.clamp(0, nw as i64 - 1) as usize]
        };
        let top = at(iu, iv) * (1.0 - fu) + at(iu + 1, iv) * fu;
        let bottom = at(iu, iv + 1) * (1.0 - fu) + at(iu + 1, iv + 1) * fu;
        top * (1.0 - fv) + bottom * fv
    }

    /// THE FLOW TREE IN LINEAR TIME where the ground has no pit (every height computed along the last tree has none): each
    /// point drains to its steepest LOWER neighbour — on ground with no pit the same receiver the priority flood picks,
    /// since every lower neighbour is reached first — and the order is a walk up the donors from the outlets. Any pit (the
    /// first pass's smooth start) falls back to the flood (`tree`).
    fn tree_fast(
        h: &[f64],
        outlet: &[bool],
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut sc = Scratch::default();
        tree_into(h, outlet, graph, &mut sc);
        (sc.order, sc.recv, sc.len)
    }

    /// The per-pass working lists, kept between passes so a pass reuses them (a fresh 28 MB a pass cost more than the
    /// tree's own work).
    #[derive(Default)]
    pub struct Scratch {
        order: Vec<usize>,
        recv: Vec<usize>,
        len: Vec<f64>,
        start: Vec<usize>,
        fill: Vec<usize>,
        donors: Vec<usize>,
        q: Vec<f64>,
        prev: Vec<f64>,
        last: Vec<usize>,
        /// The last tree's receivers on the graph `kept_key` (the tie tolerance's memory).
        kept: Vec<usize>,
        kept_key: usize,
    }
    thread_local! {
        static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
    }

    /// `tree_fast` into reused lists: `sc.order`, `sc.recv`, `sc.len`.
    fn tree_into(
        h: &[f64],
        outlet: &[bool],
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
        sc: &mut Scratch,
    ) {
        let flood = |sc: &mut Scratch| {
            let (order, recv, len) = tree(h, outlet, graph);
            (sc.order, sc.recv, sc.len) = (order, recv, len);
        };
        if !FAST_TREE.load(std::sync::atomic::Ordering::Relaxed) {
            flood(sc);
            return;
        }
        let n = h.len();
        let t_scan = Instant::now();
        let flat = flat_of(graph, n);
        // The last tree on THIS graph (the tie tolerance keeps a receiver unless another is steeper by the margin).
        let key = std::ptr::from_ref(graph) as usize;
        if sc.kept_key != key || sc.kept.len() != n {
            sc.kept_key = key;
            sc.kept.clear();
            sc.kept.resize(n, usize::MAX);
        }
        let margin = 1.0 + HYSTERESIS_PPM.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e6;
        sc.recv.clear();
        sc.recv.resize(n, usize::MAX);
        sc.len.clear();
        sc.len.resize(n, 0.0);
        for c in 0..n {
            if outlet[c] {
                continue;
            }
            let hc = h[c];
            let (a, b) = (flat.start[c] as usize, flat.start[c + 1] as usize);
            // The slope to every neighbour in one pass (a lower neighbour is exactly a positive slope), then the steepest:
            // the same arithmetic as before, with the comparisons as selects.
            let (mut bs, mut bm, mut bl) = (0.0f64, usize::MAX, 0.0f64);
            for e in a..b {
                let m = flat.to[e] as usize;
                let l = flat.len[e];
                let s = (hc - h[m]) / l;
                let take = (s > bs) | ((s == bs) & (s > 0.0) & (m < bm));
                bs = if take { s } else { bs };
                bm = if take { m } else { bm };
                bl = if take { l } else { bl };
            }
            let mut best = (bs, bm, bl);
            if margin > 1.0 {
                let was = sc.kept[c];
                for e in a..b {
                    let m = flat.to[e] as usize;
                    if m == was && h[m] < hc {
                        let l = flat.len[e];
                        let s = (hc - h[m]) / l;
                        if s * margin >= best.0 {
                            best = (s, m, l);
                        }
                    }
                }
            }
            if best.1 == usize::MAX {
                FALLBACK_PIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                flood(sc);
                return;
            }
            sc.recv[c] = best.1;
            sc.len[c] = best.2;
        }
        T_SCAN.fetch_add(
            t_scan.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let t_donors = Instant::now();
        // The donors in one flat table, then the order outlets first.
        sc.start.clear();
        sc.start.resize(n + 1, 0);
        for c in 0..n {
            if sc.recv[c] != usize::MAX {
                sc.start[sc.recv[c] + 1] += 1;
            }
        }
        for c in 0..n {
            sc.start[c + 1] += sc.start[c];
        }
        sc.fill.clear();
        sc.fill.extend_from_slice(&sc.start);
        sc.donors.clear();
        sc.donors.resize(sc.start[n], 0);
        for c in 0..n {
            let r = sc.recv[c];
            if r != usize::MAX {
                sc.donors[sc.fill[r]] = c;
                sc.fill[r] += 1;
            }
        }
        T_DONORS.fetch_add(
            t_donors.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let t_order = Instant::now();
        sc.order.clear();
        for c in 0..n {
            if outlet[c] {
                sc.order.push(c);
            }
        }
        let mut k = 0;
        while k < sc.order.len() {
            let c = sc.order[k];
            let (a, b) = (sc.start[c], sc.start[c + 1]);
            for t in a..b {
                let d = sc.donors[t];
                sc.order.push(d);
            }
            k += 1;
        }
        T_ORDER.fetch_add(
            t_order.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        sc.kept.copy_from_slice(&sc.recv);
        if sc.order.len() < n {
            // A cycle cannot form (each link falls), so a short order means a point reached no outlet: the flood decides.
            FALLBACK_SHORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            flood(sc);
            return;
        }
        FAST_DONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub static FALLBACK_PIT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    pub static FALLBACK_SHORT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    pub static FAST_DONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    /// THE TIE TOLERANCE, millionths (`--l1-hysteresis-ppm`; 0: off): a point keeps its receiver unless another lower
    /// neighbour is steeper by more than this share.
    pub static HYSTERESIS_PPM: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    thread_local! {
        /// The region's 8 km water levels over the sea (NaN where dry), for the floor-bound print.
        pub static FLOOR_BOUND: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    /// The start carved along the coarser level's rivers (`--l1-carve`).
    pub static CARVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// The holds keep the domain's outer node ring free (`--l1-hold-inner`).
    pub static HOLD_INNER: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    /// The slope hold's sweeps at most (`--l1-slope-sweeps`).
    pub static SLOPE_SWEEPS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(80);
    /// The 128 m rounds' damped step in place of the ¼–4 clamp (`--l1-damped`).
    pub static DAMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// The 16 m level held per 128 m cell by its slopes (`--l2-hold-cells`).
    pub static L2_HOLD_CELLS: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    /// The 16 m level keeps the 128 m level's channels as outlets (`--l2-channels`).
    pub static L2_CHANNELS: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    /// The fill loop around the additive hold (`--l1-fill`).
    pub static FILL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    /// The slope hold on the rounds' last tree before the additive one (`--l1-slope-hold`).
    pub static SLOPE_HOLD: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    /// A round stops when its flow tree repeats (`--l1-cycle-stop`).
    pub static CYCLE_STOP: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    /// The rounds after a coarse start (`--l1-rounds`).
    pub static L1_ROUNDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(10);
    /// Whether `steady` hands back its water and receivers (the 128 m tiles read neither; `lem::solve` does).
    pub static RETURN_LISTS: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(true);
    /// The last pass's flow tree, hashed (a round watches for a repeat).
    pub static LAST_TREE_HASH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static T_TREE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static T_SCAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static T_DONORS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static T_ORDER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static T_WATER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static T_HEIGHT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    /// The 128 m level starts from a coarse solve's rises and heights (`--l1-coarse-start`).
    pub static COARSE_START: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    thread_local! {
        /// The last domain solve's node rises (the coarse start hands them on).
        pub static LAST_U: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    /// How far a node mean may still move between passes when a round stops, metres (`--l1-settle-m`).
    pub static SETTLE_M: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
    /// A round's passes stop when the node means settle (`--l1-means-settle`).
    pub static MEANS_SETTLE: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    /// ★ THE FINE POWER (the design's §5.2 lever, P-COST): `ln` and `exp` under the fence with every division by a
    /// whole number replaced by a multiplication by its reciprocal written once as a constant — add, multiply, ONE
    /// division (`t = (m − 1)/(m + 1)`), a floor and bit patterns, so one answer on every target. `ln x = ln m + e·ln 2`
    /// (`x = m·2^e`, `m` in `[√½, √2)`, `ln m = 2·atanh t` to `t¹⁵`); `e^y = 2^k·e^f` (`|f| ≤ ln2/2`, Taylor to `f¹³`).
    pub fn fine_ln(x: vd_terrain::Gf) -> vd_terrain::Gf {
        use vd_terrain::Gf;
        const ODD: [f64; 8] = [
            1.0,
            1.0 / 3.0,
            1.0 / 5.0,
            1.0 / 7.0,
            1.0 / 9.0,
            1.0 / 11.0,
            1.0 / 13.0,
            1.0 / 15.0,
        ];
        let bits = x.to_bits();
        let mut e = ((bits >> 52) & 0x7ff) as i64 - 1_023;
        let mut m = Gf::from_bits((bits & ((1u64 << 52) - 1)) | (1_023u64 << 52));
        if m.to_bits() > Gf::from_f64(std::f64::consts::SQRT_2).to_bits() {
            m *= Gf::HALF;
            e += 1;
        }
        let t = (m - Gf::ONE) / (m + Gf::ONE);
        let t2 = t * t;
        let mut p = Gf::from_f64(ODD[7]);
        for c in ODD[..7].iter().rev() {
            p = Gf::from_f64(*c) + t2 * p;
        }
        Gf::TWO * t * p + Gf::from_i64(e) * Gf::from_f64(std::f64::consts::LN_2)
    }

    /// The fine exponential (see `fine_ln`): the reciprocal factorials as constants, Horner from the inside.
    pub fn fine_exp(y: vd_terrain::Gf) -> vd_terrain::Gf {
        use vd_terrain::Gf;
        const INV_FACT: [f64; 14] = [
            1.0,
            1.0,
            1.0 / 2.0,
            1.0 / 6.0,
            1.0 / 24.0,
            1.0 / 120.0,
            1.0 / 720.0,
            1.0 / 5_040.0,
            1.0 / 40_320.0,
            1.0 / 362_880.0,
            1.0 / 3_628_800.0,
            1.0 / 39_916_800.0,
            1.0 / 479_001_600.0,
            1.0 / 6_227_020_800.0,
        ];
        let k = (y * Gf::from_f64(std::f64::consts::LOG2_E) + Gf::HALF).floor();
        let f = y - k * Gf::from_f64(std::f64::consts::LN_2);
        let mut p = Gf::from_f64(INV_FACT[13]);
        for c in INV_FACT[..13].iter().rev() {
            p = Gf::from_f64(*c) + f * p;
        }
        let ki = k.to_i64_floor().clamp(-1_022, 1_022);
        p * Gf::from_bits(((ki + 1_023) as u64) << 52)
    }

    /// The heights' power `x^(1/n)` by the stated mode (`--l1-power`): 0 the platform's, 1 the fence's own `powf`, 2 the
    /// fine power.
    fn law_power(x: f64) -> f64 {
        use vd_terrain::Gf;
        match POWER_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => x.powf(1.0 / JOINT_N),
            1 => Gf::from_f64(x).powf(Gf::from_f64(1.0 / JOINT_N)).to_f64(),
            _ => fine_exp(fine_ln(Gf::from_f64(x)) * Gf::from_f64(1.0 / JOINT_N)).to_f64(),
        }
    }
    pub const JOINT_N_PUB: f64 = JOINT_N;
    /// The power's mode (`--l1-power`).
    pub static POWER_MODE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// THE POINT GRAPH IN FLAT ARRAYS (each point's neighbours side by side), built once per graph: the tree's scans
    /// read memory in order in place of one heap list a point.
    pub struct Flat {
        start: Vec<u32>,
        to: Vec<u32>,
        len: Vec<f64>,
    }
    thread_local! {
        static FLAT: std::cell::RefCell<Option<(usize, usize, std::rc::Rc<Flat>)>> = const { std::cell::RefCell::new(None) };
    }
    fn flat_of(
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
        n: usize,
    ) -> std::rc::Rc<Flat> {
        let key = std::ptr::from_ref(graph) as usize;
        FLAT.with(|cell| {
            if let Some((k, kn, f)) = cell.borrow().as_ref()
                && *k == key
                && *kn == n
            {
                return f.clone();
            }
            let mut start = Vec::with_capacity(n + 1);
            let (mut to, mut len) = (Vec::new(), Vec::new());
            for c in 0..n {
                start.push(to.len() as u32);
                for &(m, l) in graph.neighbors_of(c) {
                    to.push(m as u32);
                    len.push(l);
                }
            }
            start.push(to.len() as u32);
            let f = std::rc::Rc::new(Flat { start, to, len });
            *cell.borrow_mut() = Some((key, n, f.clone()));
            f
        })
    }

    /// The linear-time tree where the ground allows it (`--l1-fast-tree`).
    pub static FAST_TREE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// The flow tree: a priority flood from the outlets over the point graph; each point drains to the steepest of its
    /// neighbours already reached (on the flooded heights), so the tree has no cycle and every pit spills over its pass.
    /// Returns the outlets-first order, each point's receiver and the length of its link.
    fn tree(
        h: &[f64],
        outlet: &[bool],
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let n = h.len();
        let key = |z: f64| -> i64 { (z * 4_096.0).round() as i64 };
        let mut zf = h.to_vec();
        let mut done = vec![false; n];
        let mut queued = vec![false; n];
        let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
        for i in 0..n {
            if outlet[i] {
                queued[i] = true;
                heap.push(Reverse((key(zf[i]), i)));
            }
        }
        let (mut order, mut recv, mut len) =
            (Vec::with_capacity(n), vec![usize::MAX; n], vec![0.0f64; n]);
        while let Some(Reverse((_, c))) = heap.pop() {
            if done[c] {
                continue;
            }
            done[c] = true;
            order.push(c);
            if !outlet[c] {
                let mut best = (f64::MIN, usize::MAX, 0.0f64);
                for &(m, l) in graph.neighbors_of(c) {
                    if done[m] && m != c {
                        let s = (zf[c] - zf[m]) / l;
                        if s > best.0 {
                            best = (s, m, l);
                        }
                    }
                }
                recv[c] = best.1;
                len[c] = best.2;
            }
            for &(m, _) in graph.neighbors_of(c) {
                if !queued[m] {
                    queued[m] = true;
                    zf[m] = zf[m].max(zf[c]);
                    heap.push(Reverse((key(zf[m]), m)));
                }
            }
        }
        (order, recv, len)
    }

    /// The fine ground of a square of `nw × nw` nodes of `node_m`, from each node's mean height over the sea (`node_z`)
    /// and rain (`node_rain_m`, m/yr), on points `spacing_m` apart, rastered at `cell_m`.
    pub fn solve(
        node_z: &[f64],
        node_rain_m: &[f64],
        nw: usize,
        node_m: f64,
        spacing_m: f64,
        cell_m: f64,
    ) -> Out {
        let side = nw as f64 * node_m;
        let num = ((side / spacing_m) * (side / spacing_m)) as usize;
        let started = Instant::now();
        let model = TerrainModel2DBulider::from_random_sites(
            num,
            Site2D { x: 0.0, y: 0.0 },
            Site2D { x: side, y: side },
        )
        .relaxate_sites(1)
        .expect("the sites relax")
        .add_edge_sites(None, None)
        .expect("the edge sites")
        .build()
        .expect("the point graph builds");
        let (sites, areas, graph) = (
            model.sites().to_vec(),
            model.areas().to_vec(),
            model.graph(),
        );
        let n = sites.len();
        println!(
            "  the points: {n} ({spacing_m:.0} m apart), the graph in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        let base: Vec<f64> = sites
            .iter()
            .map(|s| smooth(node_z, nw, node_m, s.x, s.y))
            .collect();
        let water: Vec<f64> = sites
            .iter()
            .zip(&areas)
            .map(|(s, a)| linear(node_rain_m, nw, node_m, s.x, s.y) * a)
            .collect();
        let mut outlet = vec![false; n];
        for &e in model.default_outlets() {
            outlet[e] = true;
        }
        for i in 0..n {
            outlet[i] |= base[i] <= 0.0;
        }
        let node_of = |s: &Site2D| -> usize {
            ((s.y / node_m) as usize).min(nw - 1) * nw + ((s.x / node_m) as usize).min(nw - 1)
        };
        let mut u_node: Vec<f64> = node_z
            .iter()
            .map(|&z| if z > 0.0 { 1.0e-3 } else { 0.0 })
            .collect();
        let mut h = base.clone();
        let (mut q, mut recv) = (Vec::new(), Vec::new());
        let mut ln_u = Vec::new();
        for round in 0..10 {
            // The uplift read in its logarithm with the same weight in every direction (`radial`): neighbouring nodes
            // differ up to a hundredfold, and a read row by row and column by column drew a low node as a square.
            let land_min = u_node
                .iter()
                .zip(node_z)
                .filter(|(_, z)| **z > 0.0)
                .map(|(u, _)| *u)
                .fold(f64::MAX, f64::min);
            ln_u = u_node
                .iter()
                .zip(node_z)
                .map(|(u, z)| if *z > 0.0 { u.ln() } else { land_min.ln() })
                .collect();
            let u: Vec<f64> = sites
                .iter()
                .map(|s| radial(&ln_u, nw, node_m, s.x, s.y).exp() / rock(s.x, s.y))
                .collect();
            let (passes, still, q_now, recv_now) =
                steady(graph, &outlet, &base, &water, &u, &mut h, 60);
            (q, recv) = (q_now, recv_now);
            // Each node's area-weighted mean against its own: the uplift follows the ratio to the law's power.
            let (mut sum, mut area) = (vec![0.0f64; nw * nw], vec![0.0f64; nw * nw]);
            for (i, s) in sites.iter().enumerate() {
                let k = node_of(s);
                sum[k] += h[i] * areas[i];
                area[k] += areas[i];
            }
            let mut worst = 0.0f64;
            for k in 0..nw * nw {
                let mean = sum[k] / area[k].max(f64::MIN_POSITIVE);
                if node_z[k] > 0.0 {
                    worst = worst.max((mean - node_z[k]).abs());
                    if mean > 1.0 {
                        u_node[k] *= (node_z[k] / mean).clamp(0.25, 4.0).powf(JOINT_N);
                    } else {
                        u_node[k] *= 4.0f64.powf(JOINT_N);
                    }
                }
            }
            println!(
                "  round {round}: {passes} tree passes ({still} receivers still changing), the worst node's mean off by \
                 {worst:.0} m ({:.1} s)",
                started.elapsed().as_secs_f64()
            );
        }
        let w = (side / cell_m).round() as usize;
        let terrain = model.create_terrain_from_result(&h);
        let raster = rasterise(&terrain, 0.0, 0.0, w, cell_m, |x, y| {
            smooth(node_z, nw, node_m, x, y)
        });
        println!(
            "  the raster: {w} × {w} cells of {cell_m:.0} m ({:.1} s)",
            started.elapsed().as_secs_f64()
        );
        Out {
            raster,
            w,
            sites,
            q,
            recv,
            ln_u,
            terrain,
        }
    }

    /// The point graph's natural-neighbour heights at the centres of `w` × `w` cells of `cell_m` from `(x0, y0)`.
    fn rasterise(
        terrain: &fastlem::models::surface::terrain::Terrain2D,
        x0: f64,
        y0: f64,
        w: usize,
        cell_m: f64,
        fallback: impl Fn(f64, f64) -> f64 + Sync,
    ) -> Vec<f64> {
        let mut raster = vec![0.0f64; w * w];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let rows = w.div_ceil(threads);
        std::thread::scope(|scope| {
            for (t, chunk) in raster.chunks_mut(rows * w).enumerate() {
                let fallback = &fallback;
                scope.spawn(move || {
                    for (k, cell) in chunk.iter_mut().enumerate() {
                        let (x, y) = (k % w, t * rows + k / w);
                        let s = Site2D {
                            x: x0 + (x as f64 + 0.5) * cell_m,
                            y: y0 + (y as f64 + 0.5) * cell_m,
                        };
                        *cell = terrain
                            .get_elevation(&s)
                            .unwrap_or_else(|| fallback(s.x, s.y));
                    }
                });
            }
        });
        raster
    }

    /// ★ THE STEADY GROUND IN ONE TREE SOLVE (B81's speed lever, to be benched beside `relax`): each point stands over its
    /// receiver by the LOWER of two steady slopes — the joint law's `(U/(K·Q))^(1/2.2)` where water gathers, and the
    /// hillslope law's own steady slope near the divides (Roering et al. 2007: at steady state the creep carries the rise
    /// of all the ground above the point, `q = U·ℓ` with `ℓ` that ground's area over the contour's width, and
    /// `D·S/(1 − (S/S_c)²) = q` gives `S = S_c²(√(D² + (2Uℓ/S_c)²) − D)/(2Uℓ)`) — never steeper than `S_c`; the tree found
    /// again from the heights until it repeats or `max_passes`. Where nothing rises the law's steady ground is its outlet's
    /// level. Returns the passes and the receivers still changing.
    #[allow(clippy::too_many_arguments)]
    fn steady_creep(
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
        outlet: &[bool],
        base: &[f64],
        water: &[f64],
        areas: &[f64],
        u: &[f64],
        k: &[f64],
        h: &mut [f64],
        width: f64,
        max_passes: usize,
    ) -> (usize, usize) {
        let n = h.len();
        let d = super::r2::CREEP_K_M2_YR;
        let sc = vd_terrain::surface::SOIL_CRITICAL_SLOPE;
        let (mut last, mut passes, mut still) = (vec![usize::MAX; n], 0usize, 0usize);
        for _ in 0..max_passes {
            passes += 1;
            let (order, recv, len) = tree_fast(h, outlet, graph);
            let (mut q, mut a) = (water.to_vec(), areas.to_vec());
            if SHARED_AREA.load(std::sync::atomic::Ordering::Relaxed) {
                // Highest first (ties by index): each point shares its water and area with every lower neighbour by the
                // slope to the power 1.1; a point with no lower neighbour hands all to its tree receiver.
                let mut by_height: Vec<usize> = (0..n).collect();
                by_height.sort_by(|&x, &y| h[y].total_cmp(&h[x]).then(x.cmp(&y)));
                let mut share: Vec<(usize, f64)> = Vec::with_capacity(16);
                for &c in &by_height {
                    if outlet[c] {
                        continue;
                    }
                    share.clear();
                    let mut total = 0.0f64;
                    for &(m, l) in graph.neighbors_of(c) {
                        let drop = h[c] - h[m];
                        if drop > 0.0 {
                            let wgt = (drop / l).powf(1.1);
                            share.push((m, wgt));
                            total += wgt;
                        }
                    }
                    if total <= 0.0 {
                        if recv[c] != usize::MAX {
                            share.push((recv[c], 1.0));
                            total = 1.0;
                        } else {
                            continue;
                        }
                    }
                    for &(m, wgt) in &share {
                        q[m] += q[c] * wgt / total;
                        a[m] += a[c] * wgt / total;
                    }
                }
            } else {
                for &c in order.iter().rev() {
                    if recv[c] != usize::MAX {
                        q[recv[c]] += q[c];
                        a[recv[c]] += a[c];
                    }
                }
            }
            for &c in &order {
                if outlet[c] || recv[c] == usize::MAX {
                    h[c] = base[c];
                    continue;
                }
                let s_river = if u[c] > 0.0 && k[c] > 0.0 {
                    (u[c] / (k[c] * q[c].max(f64::MIN_POSITIVE))).powf(1.0 / JOINT_N)
                } else {
                    f64::INFINITY
                };
                let ul = u[c] * a[c] / width;
                let s_creep = if ul > 0.0 {
                    sc * sc * ((d * d + (2.0 * ul / sc).powi(2)).sqrt() - d) / (2.0 * ul)
                } else {
                    0.0
                };
                let s = s_river.min(s_creep).min(sc);
                h[c] = h[recv[c]] + s * len[c];
            }
            still = recv.iter().zip(&last).filter(|(x, y)| x != y).count();
            last = recv;
            if still == 0 {
                break;
            }
        }
        (passes, still)
    }

    /// The cotangent weight `ℓ*/d = (cot α + cot β)/2` of each graph edge (the Voronoi face over the Delaunay edge),
    /// aligned with `graph.neighbors_of(i)`: the apexes are the two common neighbours, one on each side, each the one
    /// with the widest angle (the true face's).
    fn cot_weights(
        sites: &[Site2D],
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
    ) -> Vec<Vec<f64>> {
        (0..sites.len())
            .map(|i| {
                graph
                    .neighbors_of(i)
                    .iter()
                    .map(|&(j, _)| {
                        let (mut left, mut right) = (f64::MAX, f64::MAX);
                        for &(k, _) in graph.neighbors_of(i) {
                            if k == j || !graph.neighbors_of(j).iter().any(|&(m, _)| m == k) {
                                continue;
                            }
                            let (ax, ay) = (sites[i].x - sites[k].x, sites[i].y - sites[k].y);
                            let (bx, by) = (sites[j].x - sites[k].x, sites[j].y - sites[k].y);
                            let cross = ax * by - ay * bx;
                            if cross == 0.0 {
                                continue;
                            }
                            let cot = (ax * bx + ay * by) / cross.abs();
                            if cross > 0.0 {
                                left = left.min(cot);
                            } else {
                                right = right.min(cot);
                            }
                        }
                        let side = |c: f64| if c == f64::MAX { 0.0 } else { c };
                        (0.5 * (side(left) + side(right))).max(0.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// ★ THE RELAXATION TO THE STEADY STATE WITH THE HILLSLOPE LAW (B80 step 2; the proven scheme: FastScape's operator
    /// splitting, Braun & Willett 2013; the law Perron, Kirchner & Dietrich 2009 name the valley spacing's): each step
    /// the rock rises by `U·dt`, the flow tree is found again, the joint law cuts each point implicitly toward its
    /// receiver's new height (Newton, outlets first), and soil creeps implicitly over the Voronoi faces —
    /// `V_i·dz_i/dt = Σ_j k_ij·(ℓ*/d)_ij·(z_j − z_i)`, `k_ij = D·(1 + r² + … + r^18)`, `r = S_ij/S_c` (Roering, Kirchner &
    /// Dietrich 1999's law in Ganti et al. 2012's finite series; D 0.003 m²/yr and S_c 1.25 the game's own) — two Picard
    /// sweeps, each a Jacobi-preconditioned conjugate gradient. No slope clamp: the creep law makes the slopes. Every 25
    /// steps one factor on U keeps the window's mean on the coarse ground's (`target`). The span is four response times
    /// of the window (its mean height over its lowest outlet over its mean U).
    #[allow(clippy::too_many_arguments)]
    fn relax(
        sites: &[Site2D],
        areas: &[f64],
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
        outlet: &[bool],
        base: &[f64],
        water: &[f64],
        u: &mut [f64],
        h: &mut [f64],
        steps: usize,
        target: f64,
        fixed: Option<(&[f64], f64)>,
    ) {
        // `fixed`: the erodibility at each point and the span in years, from ONE field over the whole region (the seam
        // test): no per-window rise scaling and no per-window mean nudge, so two windows never hold two values for one
        // place.
        let started = Instant::now();
        let n = h.len();
        let d_creep = super::r2::CREEP_K_M2_YR;
        let sc = vd_terrain::surface::SOIL_CRITICAL_SLOPE;
        let cot = cot_weights(sites, graph);
        let total_area: f64 = areas.iter().sum();
        let floor = (0..n)
            .filter(|&i| outlet[i])
            .map(|i| base[i])
            .fold(f64::MAX, f64::min);
        let mean_of = |h: &[f64]| h.iter().zip(areas).map(|(z, a)| z * a).sum::<f64>() / total_area;
        let mut mean_u = u.iter().zip(areas).map(|(v, a)| v * a).sum::<f64>() / total_area;
        let mut k_rock: Vec<f64> = sites.iter().map(|s| JOINT_K_LP * rock(s.x, s.y)).collect();
        let rise = RISE_NM.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1.0e-9;
        if let Some((k, _)) = fixed {
            k_rock = k.to_vec();
        } else if RISE_SET.load(std::sync::atomic::Ordering::Relaxed) {
            let f = rise / mean_u.max(1.0e-12);
            for v in u.iter_mut() {
                *v *= f;
            }
            for k in &mut k_rock {
                *k *= f;
            }
            println!(
                "  the place's measured rise {:.3} mm/yr: U and the erodibility both × {f:.1} (every height keeps)",
                rise * 1e3
            );
            mean_u = rise;
        }
        let tau = ((mean_of(h) - floor).max(1.0) / mean_u.max(1.0e-9)).clamp(1.0e4, 5.0e7);
        let dt = match fixed {
            Some((_, span)) => span / steps as f64,
            None => 4.0 * tau / steps as f64,
        };
        // Perron's length for the joint law (Theodoratos 2018): ℓc = (D^n · U^(1−n) / (K·R))^(1/(n+2)).
        let mut per_area: Vec<f64> = water
            .iter()
            .zip(areas)
            .map(|(w, a)| w / a.max(f64::MIN_POSITIVE))
            .collect();
        per_area.sort_by(f64::total_cmp);
        let rain_mean = per_area[per_area.len() / 2];
        let k_mean = k_rock.iter().sum::<f64>() / k_rock.len() as f64;
        let lc = (d_creep.powf(JOINT_N) * mean_u.powf(1.0 - JOINT_N)
            / (k_mean * rain_mean.max(1e-6)))
        .powf(1.0 / (JOINT_N + 2.0));
        println!(
            "  the relaxation: {steps} steps of {:.0} years ({:.2} Myr), the window's U {:.3} mm/yr, Perron's length ℓc ≈ \
             {lc:.0} m",
            dt,
            4.0 * tau / 1.0e6,
            mean_u * 1_000.0
        );
        let mut w_edge: Vec<Vec<f64>> = cot.iter().map(|c| vec![0.0; c.len()]).collect();
        let mut moved_last = 0.0;
        for step in 0..steps {
            let before = h.to_vec();
            for i in 0..n {
                if !outlet[i] {
                    h[i] += u[i] * dt;
                }
            }
            // The joint law, implicitly, outlets first.
            let (order, recv, len) = tree(h, outlet, graph);
            let mut q = water.to_vec();
            for &c in order.iter().rev() {
                if recv[c] != usize::MAX {
                    q[recv[c]] += q[c];
                }
            }
            for &c in &order {
                if outlet[c] {
                    h[c] = base[c];
                    continue;
                }
                let r = recv[c];
                if r == usize::MAX {
                    continue;
                }
                let x0 = h[c] - h[r];
                if x0 <= 0.0 {
                    continue;
                }
                let k = dt * k_rock[c] * q[c];
                let mut x = x0;
                for _ in 0..40 {
                    let s = x / len[c];
                    let sn = s.powf(JOINT_N);
                    let g = x - x0 + k * sn;
                    let dg = 1.0 + k * JOINT_N * sn / s.max(f64::MIN_POSITIVE) / len[c];
                    let next = (x - g / dg).max(0.0);
                    let moved = (x - next).abs();
                    x = next;
                    if moved < 1.0e-6 {
                        break;
                    }
                }
                h[c] = h[r] + x;
            }
            // The threshold law at S_c (Burbank et al. 1996): no link stands steeper; the landslides carry the excess to
            // the channels. The exact lower envelope, from the lowest point up; the creep's series then stays finite.
            {
                let key = |z: f64| -> i64 { (z * 4_096.0).round() as i64 };
                let mut heap: BinaryHeap<Reverse<(i64, usize)>> =
                    (0..n).map(|i| Reverse((key(h[i]), i))).collect();
                let mut done = vec![false; n];
                while let Some(Reverse((_, c))) = heap.pop() {
                    if done[c] {
                        continue;
                    }
                    done[c] = true;
                    for &(m, l) in graph.neighbors_of(c) {
                        if done[m] || outlet[m] {
                            continue;
                        }
                        let cap = h[c] + sc * l;
                        if h[m] > cap {
                            h[m] = cap;
                            heap.push(Reverse((key(cap), m)));
                        }
                    }
                }
            }
            // The creep, implicitly: two Picard sweeps, each a conjugate gradient on the Voronoi faces.
            let old = h.to_vec();
            for _ in 0..2 {
                for i in 0..n {
                    for (e, &(j, l)) in graph.neighbors_of(i).iter().enumerate() {
                        // Past S_c the threshold law moves the rock; the creep flows there at its own rate at S_c
                        // (a link under a fixed edge point can stand steeper than the landslides can reach).
                        let r2 = ((h[i] - h[j]) / l / sc).powi(2).min(1.0);
                        let mut series = 0.0;
                        let mut term = 1.0;
                        for _ in 0..10 {
                            series += term;
                            term *= r2;
                        }
                        w_edge[i][e] = d_creep * series * cot[i][e];
                    }
                }
                let diag: Vec<f64> = (0..n)
                    .map(|i| {
                        if outlet[i] {
                            1.0
                        } else {
                            areas[i] / dt + w_edge[i].iter().sum::<f64>()
                        }
                    })
                    .collect();
                let apply = |x: &[f64], y: &mut [f64]| {
                    for i in 0..n {
                        if outlet[i] {
                            y[i] = x[i];
                            continue;
                        }
                        let mut acc = diag[i] * x[i];
                        for (e, &(j, _)) in graph.neighbors_of(i).iter().enumerate() {
                            if !outlet[j] {
                                acc -= w_edge[i][e] * x[j];
                            }
                        }
                        y[i] = acc;
                    }
                };
                let rhs: Vec<f64> = (0..n)
                    .map(|i| {
                        if outlet[i] {
                            return base[i];
                        }
                        let mut b = areas[i] / dt * old[i];
                        for (e, &(j, _)) in graph.neighbors_of(i).iter().enumerate() {
                            if outlet[j] {
                                b += w_edge[i][e] * base[j];
                            }
                        }
                        b
                    })
                    .collect();
                let mut x = h.to_vec();
                let mut ax = vec![0.0; n];
                apply(&x, &mut ax);
                let mut r: Vec<f64> = (0..n).map(|i| rhs[i] - ax[i]).collect();
                let mut z: Vec<f64> = (0..n).map(|i| r[i] / diag[i]).collect();
                let mut p = z.clone();
                let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
                let norm0 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
                for _ in 0..200 {
                    apply(&p, &mut ax);
                    let pap: f64 = p.iter().zip(&ax).map(|(a, b)| a * b).sum();
                    if pap <= 0.0 {
                        break;
                    }
                    let alpha = rz / pap;
                    for i in 0..n {
                        x[i] += alpha * p[i];
                        r[i] -= alpha * ax[i];
                    }
                    if r.iter().map(|v| v * v).sum::<f64>().sqrt() < 1.0e-9 * norm0 {
                        break;
                    }
                    for i in 0..n {
                        z[i] = r[i] / diag[i];
                    }
                    let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
                    if !(rz_new.is_finite() && rz > 0.0) {
                        break;
                    }
                    let beta = rz_new / rz;
                    rz = rz_new;
                    for i in 0..n {
                        p[i] = z[i] + beta * p[i];
                    }
                }
                h.copy_from_slice(&x);
            }
            moved_last = h
                .iter()
                .zip(&before)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n as f64;
            if fixed.is_none() && (step + 1) % 25 == 0 {
                let mean = mean_of(h);
                let ratio = ((target - floor).max(1.0) / (mean - floor).max(1.0)).clamp(0.5, 2.0);
                for v in u.iter_mut() {
                    *v *= ratio.powf(JOINT_N);
                }
                if (step + 1) % 100 == 0 {
                    println!(
                        "    step {}: the mean {mean:.0} m (the coarse {target:.0} m), {moved_last:.3} m moved a point a step \
                         ({:.1} s)",
                        step + 1,
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        }
        println!(
            "  the relaxation done: {moved_last:.4} m moved a point in the last step ({:.1} s)",
            started.elapsed().as_secs_f64()
        );
    }

    /// One hashed point a cell of the global lattice at `spacing` (cells counted from the region's north-west corner): a
    /// place drawn from the cell's own hash, up to 0.45 of a cell from its centre. A point depends only on its cell, so
    /// two domains that both hold a cell hold the same point there (B81's G-SEAM; the design's §3.1).
    pub fn cell_point(cx: i64, cy: i64, spacing: f64) -> Site2D {
        let key =
            ((cx as u64 & 0xFFFF_FFFF) << 32) ^ (cy as u64 & 0xFFFF_FFFF) ^ 0x51EA_D0C1_4A11_0000;
        let (a, b) = (
            super::r2::hash01(key),
            super::r2::hash01(key ^ 0x9E37_79B9_7F4A_7C15),
        );
        Site2D {
            x: (cx as f64 + 0.5 + 0.9 * (a - 0.5)) * spacing,
            y: (cy as f64 + 0.5 + 0.9 * (b - 0.5)) * spacing,
        }
    }

    /// The coarser solve's spacing: its region's side over the square root of its point count.
    fn coarse_spacing(coarse: &Out) -> f64 {
        (coarse.w as f64 * coarse.w as f64 * 128.0 * 128.0 / coarse.sites.len() as f64).sqrt()
    }

    /// THE POINT SET a domain is solved on (`--point-set`): 0 today's hashed point a cell (0.45 of a cell from its
    /// centre at most), 1 the full jitter, 2 the hashed Poisson-disk, 3 B80's relaxed random set (a reference only).
    pub static POINT_SET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// THE FULL JITTER: one hashed point a cell at a uniform place anywhere inside it.
    pub fn cell_point_full(cx: i64, cy: i64, spacing: f64) -> Site2D {
        let key =
            ((cx as u64 & 0xFFFF_FFFF) << 32) ^ (cy as u64 & 0xFFFF_FFFF) ^ 0x51EA_D0C1_4A11_0000;
        let (a, b) = (
            super::r2::hash01(key),
            super::r2::hash01(key ^ 0x9E37_79B9_7F4A_7C15),
        );
        Site2D {
            x: (cx as f64 + a) * spacing,
            y: (cy as f64 + b) * spacing,
        }
    }

    /// THE HASHED POISSON-DISK (Matérn's hard-core process, type II): four hashed candidates a cell, each with a hashed
    /// priority; a candidate is kept unless a candidate of higher priority stands within `r = cell/√π` — the radius that
    /// keeps one point a cell on average (`(1 − e^(−λπr²))/(πr²)` at λ = 4 is 0.98). A candidate's fate reads only the
    /// candidates within `r` (all in the eight cells around it), so every domain that holds a cell keeps the same points.
    pub fn matern_points(cx: i64, cy: i64, spacing: f64) -> Vec<Site2D> {
        const K: u64 = 4;
        let r2 = spacing * spacing / std::f64::consts::PI;
        let cand = |cx: i64, cy: i64, t: u64| -> (f64, f64, u64) {
            let key = ((cx as u64 & 0xFFFF_FFFF) << 32)
                ^ (cy as u64 & 0xFFFF_FFFF)
                ^ (0x3A7E_4D15_C000_0000 + (t << 56));
            let (a, b) = (
                super::r2::hash01(key),
                super::r2::hash01(key ^ 0x9E37_79B9_7F4A_7C15),
            );
            let pri = (super::r2::hash01(key ^ 0xC2B2_AE3D_27D4_EB4F) * (1u64 << 52) as f64) as u64;
            ((cx as f64 + a) * spacing, (cy as f64 + b) * spacing, pri)
        };
        let mut out = Vec::new();
        for t in 0..K {
            let (x, y, pri) = cand(cx, cy, t);
            let mut kept = true;
            'scan: for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    for u in 0..K {
                        if dx == 0 && dy == 0 && u == t {
                            continue;
                        }
                        let (qx, qy, qp) = cand(cx + dx, cy + dy, u);
                        let higher = (qp, cx + dx, cy + dy, u) > (pri, cx, cy, t);
                        if higher && (qx - x).powi(2) + (qy - y).powi(2) < r2 {
                            kept = false;
                            break 'scan;
                        }
                    }
                }
            }
            if kept {
                out.push(Site2D { x, y });
            }
        }
        out
    }

    /// A solved domain's raster: `w` × `h` cells of `cell` metres from `(x0, y0)`.
    pub struct DomainSolve {
        pub x0: f64,
        pub y0: f64,
        pub w: usize,
        pub h: usize,
        pub cell: f64,
        pub raster: Vec<f64>,
    }

    /// The point graph's natural-neighbour heights over a rectangle of `w` × `h` cells.
    fn rasterise_rect(
        terrain: &fastlem::models::surface::terrain::Terrain2D,
        x0: f64,
        y0: f64,
        w: usize,
        h: usize,
        cell_m: f64,
    ) -> Vec<f64> {
        let mut raster = vec![0.0f64; w * h];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let rows = h.div_ceil(threads);
        std::thread::scope(|scope| {
            for (t, chunk) in raster.chunks_mut(rows * w).enumerate() {
                scope.spawn(move || {
                    for (k, cell) in chunk.iter_mut().enumerate() {
                        let (x, y) = (k % w, t * rows + k / w);
                        let s = Site2D {
                            x: x0 + (x as f64 + 0.5) * cell_m,
                            y: y0 + (y as f64 + 0.5) * cell_m,
                        };
                        *cell = terrain.get_elevation(&s).unwrap_or(f64::NAN);
                    }
                });
            }
        });
        raster
    }

    /// NON-NEGATIVE LEAST SQUARES (Lawson & Hanson 1974, ch. 23): the `x ≥ 0` that minimises `|A·x − b|`, `A` `m × n`
    /// row-major; started from the passive set `warm` (the last solve's positive unknowns).
    fn nnls(a: &[f64], m: usize, n: usize, b: &[f64], warm: &[bool]) -> Vec<f64> {
        let ls = |pass: &[usize]| -> Vec<f64> {
            // The normal equations on the passive columns, by Gaussian elimination with partial pivoting.
            let p = pass.len();
            let mut g = vec![0.0f64; p * (p + 1)];
            for (r, &cr) in pass.iter().enumerate() {
                for (c, &cc) in pass.iter().enumerate() {
                    g[r * (p + 1) + c] = (0..m).map(|i| a[i * n + cr] * a[i * n + cc]).sum();
                }
                g[r * (p + 1) + p] = (0..m).map(|i| a[i * n + cr] * b[i]).sum();
            }
            for col in 0..p {
                let piv = (col..p)
                    .max_by(|&x, &y| {
                        g[x * (p + 1) + col]
                            .abs()
                            .total_cmp(&g[y * (p + 1) + col].abs())
                    })
                    .unwrap_or(col);
                if piv != col {
                    for c in 0..=p {
                        g.swap(col * (p + 1) + c, piv * (p + 1) + c);
                    }
                }
                let d = g[col * (p + 1) + col];
                if d.abs() < 1.0e-300 {
                    continue;
                }
                for r in (col + 1)..p {
                    let f = g[r * (p + 1) + col] / d;
                    if f != 0.0 {
                        for c in col..=p {
                            g[r * (p + 1) + c] -= f * g[col * (p + 1) + c];
                        }
                    }
                }
            }
            let mut z = vec![0.0f64; p];
            for r in (0..p).rev() {
                let d = g[r * (p + 1) + r];
                let acc: f64 = ((r + 1)..p).map(|c| g[r * (p + 1) + c] * z[c]).sum();
                z[r] = if d.abs() < 1.0e-300 {
                    0.0
                } else {
                    (g[r * (p + 1) + p] - acc) / d
                };
            }
            z
        };
        let grad = |x: &[f64]| -> Vec<f64> {
            let r: Vec<f64> = (0..m)
                .map(|i| b[i] - (0..n).map(|j| a[i * n + j] * x[j]).sum::<f64>())
                .collect();
            (0..n)
                .map(|j| (0..m).map(|i| a[i * n + j] * r[i]).sum())
                .collect()
        };
        let mut x = vec![0.0f64; n];
        let mut passive: Vec<bool> = warm.to_vec();
        let mut first = passive.iter().any(|&p| p);
        let tol = 1.0e-12
            * (0..m * n)
                .map(|k| a[k].abs())
                .fold(0.0f64, f64::max)
                .max(1.0e-300)
            * b.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0);
        for _ in 0..3 * n + 10 {
            if !first {
                let w = grad(&x);
                let pick = (0..n)
                    .filter(|&j| !passive[j] && w[j] > tol)
                    .max_by(|&p, &q| w[p].total_cmp(&w[q]).then(q.cmp(&p)));
                let Some(j) = pick else { break };
                passive[j] = true;
            }
            first = false;
            loop {
                let pass: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
                let zp = ls(&pass);
                let mut z = vec![0.0f64; n];
                for (k, &j) in pass.iter().enumerate() {
                    z[j] = zp[k];
                }
                if pass.iter().all(|&j| z[j] > 0.0) {
                    x = z;
                    break;
                }
                let alpha = pass
                    .iter()
                    .filter(|&&j| z[j] <= 0.0)
                    .map(|&j| {
                        if x[j] - z[j] > 0.0 {
                            x[j] / (x[j] - z[j])
                        } else {
                            0.0
                        }
                    })
                    .fold(1.0f64, f64::min);
                for j in 0..n {
                    x[j] += alpha * (z[j] - x[j]);
                    if passive[j] && x[j] <= 1.0e-15 {
                        passive[j] = false;
                        x[j] = 0.0;
                    }
                }
                if !passive.iter().any(|&p| p) {
                    break;
                }
            }
        }
        x
    }

    /// ★ THE LINEAR MEAN HOLD (the design §3.3, review B-B4): on a FIXED flow tree a point stands over its receiver by
    /// `min(S_c, v·Q^(−1/n))·L`, `v` the held nodes' `V = (U/K)^(1/n)` read by a radial partition of unity (σ
    /// `L1_SIGMA_C`) — LINEAR in the nodes' `V`; the held nodes' means are `c + G·V`, and ONE non-negative least-squares
    /// solve gives the `V` that holds them; the links on the threshold form an active set, found again until it repeats;
    /// then the tree is found again from the new heights until it repeats (a cycle keeps its best tree). Returns the trees,
    /// the worst held node's residual and the share of the tree's links on the threshold.
    #[allow(clippy::too_many_arguments)]
    fn linear_hold(
        graph: &terrain_graph::edge_attributed_undirected::EdgeAttributedUndirectedGraph<f64>,
        sites: &[Site2D],
        areas: &[f64],
        outlet: &[bool],
        base: &[f64],
        water: &[f64],
        node_z: &[f64],
        held: &[bool],
        nw: usize,
        node_m: f64,
        h: &mut [f64],
    ) -> (usize, f64, f64) {
        let n = h.len();
        let sc = threshold();
        let sigma = L1_SIGMA_C.load(std::sync::atomic::Ordering::Relaxed) as f64 / 100.0;
        let two_s2 = 2.0 * sigma * sigma;
        let reach = (3.0 * sigma).ceil() as i64;
        let hn: Vec<usize> = (0..nw * nw).filter(|&k| held[k]).collect();
        let nh = hn.len();
        let node_of: Vec<usize> = sites
            .iter()
            .map(|s| {
                ((s.y / node_m) as usize).min(nw - 1) * nw + ((s.x / node_m) as usize).min(nw - 1)
            })
            .collect();
        let row_of: Vec<usize> = (0..nw * nw)
            .map(|k| hn.iter().position(|&m| m == k).unwrap_or(usize::MAX))
            .collect();
        let mut node_area = vec![0.0f64; nw * nw];
        for i in 0..n {
            node_area[node_of[i]] += areas[i];
        }
        // Each point's partition of unity over the held nodes near it (zero where none is near: no rise, flat).
        let weight = |i: usize, k: usize| -> f64 {
            let (u, v) = (sites[i].x / node_m - 0.5, sites[i].y / node_m - 0.5);
            let (ku, kv) = ((k % nw) as f64, (k / nw) as f64);
            (-((u - ku).powi(2) + (v - kv).powi(2)) / two_s2).exp()
        };
        let near = |i: usize| -> Vec<usize> {
            let (u, v) = (sites[i].x / node_m - 0.5, sites[i].y / node_m - 0.5);
            let (iu, iv) = (u.round() as i64, v.round() as i64);
            let mut out = Vec::new();
            for j in (iv - reach)..=(iv + reach) {
                for ii in (iu - reach)..=(iu + reach) {
                    if ii >= 0
                        && j >= 0
                        && ii < nw as i64
                        && j < nw as i64
                        && held[j as usize * nw + ii as usize]
                    {
                        out.push(j as usize * nw + ii as usize);
                    }
                }
            }
            out
        };
        let norm: Vec<f64> = (0..n)
            .map(|i| near(i).iter().map(|&k| weight(i, k)).sum())
            .collect();
        let v_at = |i: usize, vn: &[f64]| -> f64 {
            if norm[i] <= 0.0 {
                return 0.0;
            }
            near(i)
                .iter()
                .map(|&k| weight(i, k) * vn[row_of[k]])
                .sum::<f64>()
                / norm[i]
                * rock(sites[i].x, sites[i].y).powf(-1.0 / JOINT_N)
        };
        let means = |hh: &[f64]| -> Vec<f64> {
            let mut sum = vec![0.0f64; nw * nw];
            for i in 0..n {
                sum[node_of[i]] += hh[i] * areas[i];
            }
            hn.iter()
                .map(|&k| sum[k] / node_area[k].max(f64::MIN_POSITIVE))
                .collect()
        };
        let mut vn = vec![0.0f64; nh];
        let mut warm = vec![false; nh];
        let mut seen: Vec<(u64, f64, Vec<f64>)> = Vec::new();
        let (mut trees, mut worst, mut on_threshold) = (0usize, f64::MAX, 0.0f64);
        let mut last_recv: Vec<usize> = Vec::new();
        for _ in 0..30 {
            trees += 1;
            let (order, recv, len) = tree(h, outlet, graph);
            let mut q = water.to_vec();
            for &c in order.iter().rev() {
                if recv[c] != usize::MAX {
                    q[recv[c]] += q[c];
                }
            }
            // The coefficient of `v` on each link: `Q^(−1/n)·L` (zero at an outlet).
            let coef: Vec<f64> = (0..n)
                .map(|i| {
                    if outlet[i] || recv[i] == usize::MAX {
                        0.0
                    } else {
                        q[i].max(f64::MIN_POSITIVE).powf(-1.0 / JOINT_N) * len[i]
                    }
                })
                .collect();
            // The active set from the heights the last solve left (the first tree: none).
            let mut active: Vec<bool> = (0..n)
                .map(|i| coef[i] > 0.0 && (h[i] - h[recv[i]]) >= sc * len[i] * (1.0 - 1.0e-9))
                .collect();
            if trees == 1 {
                active.iter_mut().for_each(|a| *a = false);
            }
            for _ in 0..10 {
                // `c`: the outlets' bases and the threshold links alone.
                let mut hc = vec![0.0f64; n];
                for &i in &order {
                    hc[i] = if coef[i] == 0.0 {
                        base[i]
                    } else {
                        hc[recv[i]] + if active[i] { sc * len[i] } else { 0.0 }
                    };
                }
                let c = means(&hc);
                // `G`: one column a held node, its `V` alone on the free links.
                let mut g = vec![0.0f64; nh * nh];
                let mut col = vec![0.0f64; n];
                for (m, &km) in hn.iter().enumerate() {
                    let (ku, kv) = ((km % nw) as f64, (km / nw) as f64);
                    for &i in &order {
                        if coef[i] == 0.0 {
                            col[i] = 0.0;
                            continue;
                        }
                        let mut add = 0.0;
                        if !active[i] && norm[i] > 0.0 {
                            let (u, v) = (sites[i].x / node_m - 0.5, sites[i].y / node_m - 0.5);
                            if (u - ku).abs() <= reach as f64 + 0.5
                                && (v - kv).abs() <= reach as f64 + 0.5
                            {
                                add = weight(i, km) / norm[i]
                                    * coef[i]
                                    * rock(sites[i].x, sites[i].y).powf(-1.0 / JOINT_N);
                            }
                        }
                        col[i] = col[recv[i]] + add;
                    }
                    let cm = means(&col);
                    for r in 0..nh {
                        g[r * nh + m] = cm[r];
                    }
                }
                let rhs: Vec<f64> = (0..nh).map(|r| node_z[hn[r]] - c[r]).collect();
                vn = nnls(&g, nh, nh, &rhs, &warm);
                for r in 0..nh {
                    warm[r] = vn[r] > 0.0;
                }
                // The threshold links again from the solved `V`.
                let next: Vec<bool> = (0..n)
                    .map(|i| coef[i] > 0.0 && v_at(i, &vn) * coef[i] > sc * len[i])
                    .collect();
                if next == active {
                    break;
                }
                active = next;
            }
            // The heights on this tree.
            for &i in &order {
                h[i] = if coef[i] == 0.0 {
                    base[i]
                } else {
                    h[recv[i]] + (v_at(i, &vn) * coef[i]).min(sc * len[i])
                };
            }
            let mm = means(h);
            worst = (0..nh)
                .map(|r| (mm[r] - node_z[hn[r]]).abs())
                .fold(0.0f64, f64::max);
            on_threshold = (0..n)
                .filter(|&i| coef[i] > 0.0 && v_at(i, &vn) * coef[i] >= sc * len[i])
                .count() as f64
                / n as f64;
            if recv == last_recv {
                break;
            }
            // A cycle of trees: keep the best of the cycle's heights.
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&recv, &mut hasher);
            let key = std::hash::Hasher::finish(&hasher);
            if seen.iter().any(|(k2, _, _)| *k2 == key) {
                if let Some((_, w2, h2)) = seen
                    .iter()
                    .filter(|(k2, _, _)| *k2 == key)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    && *w2 < worst
                {
                    h.copy_from_slice(h2);
                    worst = *w2;
                }
                break;
            }
            seen.push((key, worst, h.to_vec()));
            last_recv = recv;
        }
        (trees, worst, on_threshold)
    }

    /// ★ ONE 128 m DOMAIN (B0 step 2b, the seam at the 128 m level): the rectangle (metres from the region's north-west
    /// corner) solved on the global cell points at `spacing`, CLOSED at its rim except where a macro river crosses it (an
    /// outgoing link an outlet at the 8 km field's height there, an incoming one its water handed in), the joint law's
    /// steady state with the threshold (B80's 128 m law), and EACH NODE'S OWN MEAN held by its `U/K` (the rounds of
    /// `solve`), only the nodes whose square lies inside the rectangle held. `node_recv` holds each region node's
    /// receiver (an index, or `usize::MAX`), `node_q` the water each node carries (m³/yr).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_l1_domain(
        node_z: &[f64],
        node_rain_m: &[f64],
        nw: usize,
        node_m: f64,
        node_recv: &[usize],
        node_q: &[f64],
        (x0, y0, wm, hm): (f64, f64, f64, f64),
        spacing: f64,
    ) -> DomainSolve {
        let start = COARSE_START.load(std::sync::atomic::Ordering::Relaxed);
        solve_l1_inner(
            node_z,
            node_rain_m,
            nw,
            node_m,
            node_recv,
            node_q,
            (x0, y0, wm, hm),
            spacing,
            start,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_l1_inner(
        node_z: &[f64],
        node_rain_m: &[f64],
        nw: usize,
        node_m: f64,
        node_recv: &[usize],
        node_q: &[f64],
        (x0, y0, wm, hm): (f64, f64, f64, f64),
        spacing: f64,
        coarse_start: bool,
    ) -> DomainSolve {
        // ★ THE COARSE START (`--l1-coarse-start`): the same rounds first on points four times further apart (sixteen times
        // fewer), then this level starts from their rises and heights.
        let coarse = if coarse_start {
            let t = Instant::now();
            let c = solve_l1_inner(
                node_z,
                node_rain_m,
                nw,
                node_m,
                node_recv,
                node_q,
                (x0, y0, wm, hm),
                4.0 * spacing,
                false,
            );
            let u = LAST_U.with(|l| l.borrow().clone());
            println!(
                "  the coarse start at {:.0} m: {:.1} s",
                4.0 * spacing,
                t.elapsed().as_secs_f64()
            );
            Some((c, u))
        } else {
            None
        };
        let started = Instant::now();
        let inside = |x: f64, y: f64| x > x0 && y > y0 && x < x0 + wm && y < y0 + hm;
        let (cx0, cy0) = ((x0 / spacing).floor() as i64, (y0 / spacing).floor() as i64);
        let (cx1, cy1) = (
            ((x0 + wm) / spacing).ceil() as i64,
            ((y0 + hm) / spacing).ceil() as i64,
        );
        let model = match POINT_SET.load(std::sync::atomic::Ordering::Relaxed) {
            3 => {
                // THE REFERENCE (B80's set, not tile-safe): random sites relaxed once by Lloyd, as many as the cells.
                let num = (((cx1 - cx0) * (cy1 - cy0)) as f64 * (wm * hm)
                    / (((cx1 - cx0) * (cy1 - cy0)) as f64 * spacing * spacing))
                    as usize;
                TerrainModel2DBulider::from_random_sites(
                    num,
                    Site2D { x: x0, y: y0 },
                    Site2D {
                        x: x0 + wm,
                        y: y0 + hm,
                    },
                )
                .relaxate_sites(1)
                .expect("the sites relax")
                .build()
                .expect("the domain's graph builds")
            }
            set => {
                let mut pts = Vec::new();
                for cy in cy0..cy1 {
                    for cx in cx0..cx1 {
                        match set {
                            2 => pts.extend(
                                matern_points(cx, cy, spacing)
                                    .into_iter()
                                    .filter(|p| inside(p.x, p.y)),
                            ),
                            _ => {
                                let p = if set == 1 {
                                    cell_point_full(cx, cy, spacing)
                                } else {
                                    cell_point(cx, cy, spacing)
                                };
                                if inside(p.x, p.y) {
                                    pts.push(p);
                                }
                            }
                        }
                    }
                }
                TerrainModel2DBulider::default()
                    .set_sites(pts)
                    .set_bounding_box(
                        Some(Site2D { x: x0, y: y0 }),
                        Some(Site2D {
                            x: x0 + wm,
                            y: y0 + hm,
                        }),
                    )
                    .build()
                    .expect("the domain's graph builds")
            }
        };
        let (sites, areas, graph) = (
            model.sites().to_vec(),
            model.areas().to_vec(),
            model.graph(),
        );
        let n = sites.len();
        let warped: Vec<(f64, f64)> = sites.iter().map(|s| warp(s.x, s.y, node_m)).collect();
        if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
            let d: Vec<f64> = sites
                .iter()
                .zip(&warped)
                .map(|(s, &(x, y))| ((x - s.x).powi(2) + (y - s.y).powi(2)).sqrt() / node_m)
                .collect();
            let mean = d.iter().sum::<f64>() / d.len().max(1) as f64;
            let max = d.iter().copied().fold(0.0f64, f64::max);
            println!("      the displacement: mean {mean:.3} node, max {max:.3} node");
        }
        let mut base: Vec<f64> = warped
            .iter()
            .map(|&(x, y)| smooth(node_z, nw, node_m, x, y))
            .collect();
        let mut water: Vec<f64> = warped
            .iter()
            .zip(&areas)
            .map(|(&(x, y), a)| linear(node_rain_m, nw, node_m, x, y) * a)
            .collect();
        let mut outlet: Vec<bool> = base.iter().map(|&b| b <= 0.0).collect();
        let bucket = 4.0 * spacing;
        let (bw, bh) = (
            (wm / bucket).ceil() as usize + 1,
            (hm / bucket).ceil() as usize + 1,
        );
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); bw * bh];
        let cell_of = |x: f64, y: f64| {
            (((x - x0) / bucket).max(0.0) as usize).min(bw - 1)
                + bw * (((y - y0) / bucket).max(0.0) as usize).min(bh - 1)
        };
        for (i, st) in sites.iter().enumerate() {
            buckets[cell_of(st.x, st.y)].push(i);
        }
        let nearest = |x: f64, y: f64| -> usize {
            let c = cell_of(x, y);
            let (bx, by) = (c % bw, c / bw);
            let mut best = (f64::MAX, 0usize);
            for j in by.saturating_sub(1)..=(by + 1).min(bh - 1) {
                for i in bx.saturating_sub(1)..=(bx + 1).min(bw - 1) {
                    for &k in &buckets[j * bw + i] {
                        let d = (sites[k].x - x).powi(2) + (sites[k].y - y).powi(2);
                        if d < best.0 || (d == best.0 && k < best.1) {
                            best = (d, k);
                        }
                    }
                }
            }
            best.1
        };
        let centre = |k: usize| {
            (
                ((k % nw) as f64 + 0.5) * node_m,
                ((k / nw) as f64 + 0.5) * node_m,
            )
        };
        let (mut outs, mut ins) = (0usize, 0usize);
        for k in 0..nw * nw {
            let r = node_recv[k];
            if r == usize::MAX {
                continue;
            }
            let ((ax, ay), (bx, by)) = (centre(k), centre(r));
            let (a_in, b_in) = (inside(ax, ay), inside(bx, by));
            if a_in == b_in {
                continue;
            }
            let ((px, py), (qx, qy)) = if a_in {
                ((ax, ay), (bx, by))
            } else {
                ((bx, by), (ax, ay))
            };
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..40 {
                let m = 0.5 * (lo + hi);
                if inside(px + (qx - px) * m, py + (qy - py) * m) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let (cx, cy) = (px + (qx - px) * lo, py + (qy - py) * lo);
            let i = nearest(cx, cy);
            if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                println!(
                    "      a macro river {} at ({:.1}, {:.1}) km, the tile's ({:.2}, {:.2}) nodes from its corner, {:.2e} m³/yr",
                    if a_in { "leaves" } else { "enters" },
                    cx / 1_000.0,
                    cy / 1_000.0,
                    (cx - x0) / node_m,
                    (cy - y0) / node_m,
                    node_q[k]
                );
            }
            if a_in {
                // The river leaves at the lower of its link's two node means: a property of the link, the same for the
                // tiles on both sides (the smooth field at a corner mixes a valley's walls into its river).
                outlet[i] = true;
                base[i] = node_z[k].min(node_z[r]);
                outs += 1;
            } else if !outlet[i] {
                water[i] += node_q[k];
                ins += 1;
            }
        }
        if outs == 0 {
            let rim = |s: &Site2D| {
                s.x < x0 + spacing
                    || s.y < y0 + spacing
                    || s.x > x0 + wm - spacing
                    || s.y > y0 + hm - spacing
            };
            if let Some(k) = (0..n)
                .filter(|&k| rim(&sites[k]))
                .min_by(|&a, &b| base[a].total_cmp(&base[b]).then(a.cmp(&b)))
            {
                outlet[k] = true;
            }
        }
        // Each node's own mean held by its U/K (only the nodes whose square lies inside).
        // The held nodes: those inside the domain, less its OUTER ring when `--l1-hold-inner` (that ring is scaffolding — the
        // cut keeps at most one node of buffer beyond the tile's edge — and against the rim its rivers may find no lawful
        // way down to its row).
        let inner = if HOLD_INNER.load(std::sync::atomic::Ordering::Relaxed) {
            node_m
        } else {
            0.0
        };
        let held: Vec<bool> = (0..nw * nw)
            .map(|k| {
                let (nx0, ny0) = ((k % nw) as f64 * node_m, (k / nw) as f64 * node_m);
                nx0 >= x0 + inner - 1.0
                    && ny0 >= y0 + inner - 1.0
                    && nx0 + node_m <= x0 + wm - inner + 1.0
                    && ny0 + node_m <= y0 + hm - inner + 1.0
                    && node_z[k] > 0.0
            })
            .collect();
        let node_of = |s: &Site2D| -> usize {
            ((s.y / node_m) as usize).min(nw - 1) * nw + ((s.x / node_m) as usize).min(nw - 1)
        };
        let mut u_node: Vec<f64> = node_z
            .iter()
            .map(|&z| if z > 0.0 { 1.0e-3 } else { 1.0e-9 })
            .collect();
        let mut h = base.clone();
        // ★ THE START CARRIES THE COARSER LEVEL'S DRAINAGE (`--l1-carve`; B83's generic rule, here 128 m under 8 km): the
        // smooth 8 km field holds no river, and where an 8 km river crosses a node corner between two high nodes it puts
        // a high saddle the fine flood never crosses. So the start surface gets a broad trough (a parabola a third of a
        // node wide each side) along each 8 km river link, its floor falling from the node's row to its receiver's (never
        // rising); only the first tree reads it — every height after comes from the law.
        if CARVE.load(std::sync::atomic::Ordering::Relaxed) && coarse.is_none() {
            let half = node_m / 3.0;
            let centre = |k: usize| {
                (
                    ((k % nw) as f64 + 0.5) * node_m,
                    ((k / nw) as f64 + 0.5) * node_m,
                )
            };
            let mut carved = 0usize;
            for k in 0..nw * nw {
                let r = node_recv[k];
                if r == usize::MAX {
                    continue;
                }
                let ((ax, ay), (bx, by)) = (centre(k), centre(r));
                let (za, zb) = (node_z[k], node_z[k].min(node_z[r]));
                let (dx, dy) = (bx - ax, by - ay);
                let len2 = dx * dx + dy * dy;
                for (idx, st) in sites.iter().enumerate() {
                    if outlet[idx] {
                        continue;
                    }
                    let t = (((st.x - ax) * dx + (st.y - ay) * dy) / len2).clamp(0.0, 1.0);
                    let (px, py) = (ax + t * dx, ay + t * dy);
                    let d = ((st.x - px).powi(2) + (st.y - py).powi(2)).sqrt();
                    if d >= half {
                        continue;
                    }
                    let floor = za + t * (zb - za);
                    let trough = floor + (h[idx] - floor).max(0.0) * (d / half).powi(2);
                    if trough < h[idx] {
                        h[idx] = trough;
                        carved += 1;
                    }
                }
            }
            if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                println!("      the start carved along the 8 km rivers: {carved} point lowerings");
            }
        }
        if let Some((c, u)) = &coarse {
            u_node.clone_from(u);
            for (i, st) in sites.iter().enumerate() {
                if outlet[i] {
                    continue;
                }
                // The coarse raster read bilinearly at the point (cells centred).
                let (fx, fy) = ((st.x - c.x0) / c.cell - 0.5, (st.y - c.y0) / c.cell - 0.5);
                let (ix, iy) = (
                    fx.floor().clamp(0.0, (c.w - 2) as f64) as usize,
                    fy.floor().clamp(0.0, (c.h - 2) as f64) as usize,
                );
                let (tx, ty) = (
                    (fx - ix as f64).clamp(0.0, 1.0),
                    (fy - iy as f64).clamp(0.0, 1.0),
                );
                let at = |x: usize, y: usize| c.raster[y * c.w + x];
                h[i] = (at(ix, iy) * (1.0 - tx) + at(ix + 1, iy) * tx) * (1.0 - ty)
                    + (at(ix, iy + 1) * (1.0 - tx) + at(ix + 1, iy + 1) * tx) * ty;
            }
        }
        let mut worst = 0.0f64;
        let mut worst_in = 0.0f64;
        let ring = |k: usize| {
            let (nx0, ny0) = ((k % nw) as f64 * node_m, (k / nw) as f64 * node_m);
            nx0 >= x0 + node_m - 1.0
                && ny0 >= y0 + node_m - 1.0
                && nx0 + node_m <= x0 + wm - node_m + 1.0
                && ny0 + node_m <= y0 + hm - node_m + 1.0
        };
        let t_graph = started.elapsed().as_secs_f64();
        let t_law = Instant::now();
        if L1_LINEAR.load(std::sync::atomic::Ordering::Relaxed) {
            let (trees, w_lin, share) = linear_hold(
                graph, &sites, &areas, &outlet, &base, &water, node_z, &held, nw, node_m, &mut h,
            );
            println!(
                "  the linear hold: {trees} trees, the worst node {w_lin:.1} m, {:.1} % of the links on the threshold {:.2} ({:.1} s; the graph {t_graph:.1} s)",
                100.0 * share,
                threshold(),
                t_law.elapsed().as_secs_f64()
            );
        }
        // ★ THE ONE LOOP (`--l1-joint`): each node's rise follows its mean after EVERY tree pass, so the rise and the tree
        // settle together; the loop stops when no height moves more than the hold's quarter metre.
        if JOINT_LOOP.load(std::sync::atomic::Ordering::Relaxed) {
            let mut prev = h.clone();
            let mut passes = 0usize;
            let mut moved = f64::MAX;
            while passes < 400 && moved > 0.25 {
                passes += 1;
                let ln_u: Vec<f64> = u_node.iter().map(|u| u.ln()).collect();
                let u: Vec<f64> = sites
                    .iter()
                    .map(|s| radial(&ln_u, nw, node_m, s.x, s.y).exp() / rock(s.x, s.y))
                    .collect();
                let _ = steady(graph, &outlet, &base, &water, &u, &mut h, 1);
                let (mut sum, mut area) = (vec![0.0f64; nw * nw], vec![0.0f64; nw * nw]);
                for (i, st) in sites.iter().enumerate() {
                    let k = node_of(st);
                    sum[k] += h[i] * areas[i];
                    area[k] += areas[i];
                }
                worst = 0.0;
                worst_in = 0.0;
                for k in 0..nw * nw {
                    if !held[k] || area[k] <= 0.0 {
                        continue;
                    }
                    let mean = sum[k] / area[k];
                    worst = worst.max((mean - node_z[k]).abs());
                    if ring(k) {
                        worst_in = worst_in.max((mean - node_z[k]).abs());
                    }
                    u_node[k] *= if mean > 1.0 {
                        (node_z[k] / mean).clamp(0.25, 4.0).powf(JOINT_N)
                    } else {
                        4.0f64.powf(JOINT_N)
                    };
                }
                moved = h
                    .iter()
                    .zip(&prev)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                prev.copy_from_slice(&h);
                if TRACE.load(std::sync::atomic::Ordering::Relaxed)
                    && (passes <= 10 || passes.is_multiple_of(10))
                {
                    println!(
                        "      pass {passes}: the worst node {worst:.0} m (inside the ring {worst_in:.0} m), heights moved up to {moved:.1} m"
                    );
                }
            }
            println!(
                "  the one loop: {passes} passes, the worst node {worst:.0} m, inside the ring {worst_in:.0} m, the last move {moved:.2} m ({:.1} s; the graph {t_graph:.1} s)",
                t_law.elapsed().as_secs_f64()
            );
        }
        let mut pass_log: Vec<usize> = Vec::new();
        let rounds = if L1_LINEAR.load(std::sync::atomic::Ordering::Relaxed)
            || JOINT_LOOP.load(std::sync::atomic::Ordering::Relaxed)
        {
            0
        } else if coarse.is_some() {
            L1_ROUNDS.load(std::sync::atomic::Ordering::Relaxed)
        } else {
            10
        };
        for _ in 0..rounds {
            let ln_u: Vec<f64> = u_node.iter().map(|u| u.ln()).collect();
            let u: Vec<f64> = sites
                .iter()
                .zip(&warped)
                .map(|(s, &(x, y))| radial(&ln_u, nw, node_m, x, y).exp() / rock(s.x, s.y))
                .collect();
            let node_means = |h: &[f64]| -> (Vec<f64>, Vec<f64>) {
                let (mut sum, mut area) = (vec![0.0f64; nw * nw], vec![0.0f64; nw * nw]);
                for (i, st) in sites.iter().enumerate() {
                    let k = node_of(st);
                    sum[k] += h[i] * areas[i];
                    area[k] += areas[i];
                }
                (sum, area)
            };
            if MEANS_SETTLE.load(std::sync::atomic::Ordering::Relaxed) {
                // The round's passes stop when no held node's mean moves more than a metre between passes.
                let mut last = node_means(&h).0;
                let mut passes = 0usize;
                let mut seen: Vec<u64> = Vec::new();
                while passes < 60 {
                    passes += 1;
                    let _ = steady(graph, &outlet, &base, &water, &u, &mut h, 1);
                    let (sum, area) = node_means(&h);
                    let change = (0..nw * nw)
                        .filter(|&k| held[k] && area[k] > 0.0)
                        .map(|k| ((sum[k] - last[k]) / area[k]).abs())
                        .fold(0.0f64, f64::max);
                    last = sum;
                    let tree_hash = LAST_TREE_HASH.load(std::sync::atomic::Ordering::Relaxed);
                    if let Some(at) = seen.iter().position(|&x| x == tree_hash) {
                        if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                            println!(
                                "      round {}: pass {passes} repeats the tree of pass {} (a cycle of {}), the means moving {change:.0} m",
                                pass_log.len() + 1,
                                at + 1,
                                passes - 1 - at
                            );
                        }
                        if CYCLE_STOP.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                    }
                    seen.push(tree_hash);
                    if change <= SETTLE_M.load(std::sync::atomic::Ordering::Relaxed) as f64 {
                        break;
                    }
                }
                pass_log.push(passes);
            } else {
                let (passes, _, _, _) = steady(graph, &outlet, &base, &water, &u, &mut h, 60);
                pass_log.push(passes);
            }
            let (sum, area) = node_means(&h);
            worst = 0.0;
            worst_in = 0.0;
            for k in 0..nw * nw {
                if !held[k] || area[k] <= 0.0 {
                    continue;
                }
                let mean = sum[k] / area[k];
                worst = worst.max((mean - node_z[k]).abs());
                if ring(k) {
                    worst_in = worst_in.max((mean - node_z[k]).abs());
                }
                if DAMPED.load(std::sync::atomic::Ordering::Relaxed) {
                    // ★ THE DAMPED STEP (B23: no clamp): the rounds' own ratio — the row over the node's mean, to the
                    // law's power — with half its logarithm a round in place of the ¼–4 clamp. A node whose mean or row
                    // is not over the sea keeps its rise (its ratio has no meaning there).
                    if mean > 0.0 && node_z[k] > 0.0 {
                        u_node[k] *= (node_z[k] / mean).powf(0.5 * JOINT_N);
                    }
                } else {
                    u_node[k] *= if mean > 1.0 {
                        (node_z[k] / mean).clamp(0.25, 4.0).powf(JOINT_N)
                    } else {
                        4.0f64.powf(JOINT_N)
                    };
                }
            }
            if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                let mut r: Vec<f64> = (0..nw * nw)
                    .filter(|&k| held[k] && area[k] > 0.0)
                    .map(|k| (sum[k] / area[k] - node_z[k]).abs())
                    .collect();
                r.sort_by(f64::total_cmp);
                println!(
                    "      round {}: {} passes; the residuals' median {:.0} m, 90th {:.0} m, worst {:.0} m ({:.1} s)",
                    pass_log.len(),
                    pass_log.last().copied().unwrap_or(0),
                    r[r.len() / 2],
                    r[r.len() * 9 / 10],
                    r[r.len() - 1],
                    t_law.elapsed().as_secs_f64()
                );
            }
        }
        LAST_U.with(|l| l.borrow_mut().clone_from(&u_node));
        if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
            println!(
                "      the rise ×1e-3 and the target (m) by node, region columns 2..9, rows 6..12:"
            );
            for j in 6..12usize {
                let mut row = String::new();
                for i in 2..9usize {
                    let k = j * nw + i;
                    row += &format!("  {:>9.2e}/{:>5.0}", u_node[k] / 1.0e-3, node_z[k]);
                }
                println!("      row {j:>2}:{row}");
            }
        }
        {
            let mut acc = 0u64;
            for (i, v) in h.iter().enumerate() {
                acc = acc
                    .wrapping_mul(0x100000001B3)
                    .wrapping_add(v.to_bits() ^ i as u64);
            }
            println!(
                "  the heights' digest after the law: {acc:016x}; the fast tree {} times, the flood for a pit {} times, for a point cut off {} times",
                FAST_DONE.load(std::sync::atomic::Ordering::Relaxed),
                FALLBACK_PIT.load(std::sync::atomic::Ordering::Relaxed),
                FALLBACK_SHORT.load(std::sync::atomic::Ordering::Relaxed)
            );
        }
        {
            let (mut sum, mut area, mut low) = (
                vec![0.0f64; nw * nw],
                vec![0.0f64; nw * nw],
                vec![f64::MAX; nw * nw],
            );
            for (i, st) in sites.iter().enumerate() {
                let k = node_of(st);
                sum[k] += h[i] * areas[i];
                area[k] += areas[i];
                low[k] = low[k].min(h[i]);
            }
            let floor_bound = (0..nw * nw)
                .filter(|&k| held[k] && area[k] > 0.0 && low[k] > node_z[k])
                .count();
            let held_n = (0..nw * nw).filter(|&k| held[k]).count();
            let pits_law = (0..n)
                .filter(|&i| !outlet[i] && graph.neighbors_of(i).iter().all(|&(m, _)| h[m] > h[i]))
                .count();
            println!(
                "  after the law: {floor_bound} of {held_n} held nodes have their lowest point over their row; {pits_law} pits"
            );
            if TRACE.load(std::sync::atomic::Ordering::Relaxed) {
                FLOOR_BOUND.with(|fb| {
                    let water = fb.borrow();
                    for k in (0..nw * nw).filter(|&k| held[k] && area[k] > 0.0 && low[k] > node_z[k]) {
                        let w = water.get(k).copied().unwrap_or(f64::NAN);
                        println!(
                            "      node ({}, {}): row {:.0} m, the fine floor {:.0} m, the fine mean {:.0} m; the 8 km world's water {}",
                            k % nw,
                            k / nw,
                            node_z[k],
                            low[k],
                            sum[k] / area[k],
                            if w.is_nan() { "none (dry)".to_string() } else { format!("at {w:.0} m ({:.0} m deep)", w - node_z[k]) }
                        );
                    }
                });
            }
        }
        if !pass_log.is_empty() {
            let get = |a: &std::sync::atomic::AtomicU64| {
                a.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e9
            };
            println!(
                "  the passes' parts: the tree {:.2} s (the scan {:.2}, the donors {:.2}, the order {:.2}), the water {:.2} s, the heights {:.2} s",
                get(&T_TREE),
                get(&T_SCAN),
                get(&T_DONORS),
                get(&T_ORDER),
                get(&T_WATER),
                get(&T_HEIGHT)
            );
            println!(
                "  the rounds: tree passes {pass_log:?}, the worst node {worst:.0} m ({:.1} s; the graph {t_graph:.1} s)",
                t_law.elapsed().as_secs_f64()
            );
        }
        // ★ THE SLOPE HOLD (`--l1-slope-hold`): on the rounds' last tree, FIXED, each link's slope is scaled by a smooth
        // factor (each node's own, blended over the nodes in its logarithm by the hold's kernel) until each node's mean is
        // its row; every link still falls, so no pit can form, and the tree does not change, so nothing is captured. The
        // update reads the node's relief over its own lowest point; a row under that floor is left to the additive hold.
        if SLOPE_HOLD.load(std::sync::atomic::Ordering::Relaxed) {
            let t_sh = Instant::now();
            let (order, recv, len) = tree_fast(&h, &outlet, graph);
            let slope0: Vec<f64> = (0..n)
                .map(|i| {
                    if recv[i] == usize::MAX || outlet[i] {
                        0.0
                    } else {
                        (h[i] - h[recv[i]]) / len[i]
                    }
                })
                .collect();
            let inside_node: Vec<bool> = (0..nw * nw)
                .map(|k| {
                    let (nx0, ny0) = ((k % nw) as f64 * node_m, (k / nw) as f64 * node_m);
                    nx0 >= x0 + inner - 1.0
                        && ny0 >= y0 + inner - 1.0
                        && nx0 + node_m <= x0 + wm - inner + 1.0
                        && ny0 + node_m <= y0 + hm - inner + 1.0
                })
                .collect();
            let (k_start, k_cells, k_w) = hold_kernel(
                n,
                |i| (sites[i].x / node_m - 0.5, sites[i].y / node_m - 0.5),
                nw,
                nw,
                &inside_node,
            );
            let mut ln_g = vec![0.0f64; nw * nw];
            let (mut iters, mut worst_sh, mut floor_bound) = (0usize, f64::MAX, 0usize);
            while iters < SLOPE_SWEEPS.load(std::sync::atomic::Ordering::Relaxed) && worst_sh > 0.25
            {
                iters += 1;
                for &i in &order {
                    if recv[i] == usize::MAX || outlet[i] {
                        h[i] = base[i];
                        continue;
                    }
                    let lg: f64 = (k_start[i]..k_start[i + 1])
                        .map(|t| f64::from(k_w[t]) * ln_g[k_cells[t] as usize])
                        .sum();
                    h[i] = h[recv[i]] + slope0[i] * lg.exp() * len[i];
                }
                let (mut sum, mut area, mut low) = (
                    vec![0.0f64; nw * nw],
                    vec![0.0f64; nw * nw],
                    vec![f64::MAX; nw * nw],
                );
                for (i, st) in sites.iter().enumerate() {
                    let k = node_of(st);
                    sum[k] += h[i] * areas[i];
                    area[k] += areas[i];
                    low[k] = low[k].min(h[i]);
                }
                worst_sh = 0.0;
                floor_bound = 0;
                for k in 0..nw * nw {
                    if !held[k] || area[k] <= 0.0 {
                        continue;
                    }
                    let mean = sum[k] / area[k];
                    worst_sh = worst_sh.max((mean - node_z[k]).abs());
                    let (relief, want) = (mean - low[k], node_z[k] - low[k]);
                    if want <= 0.0 {
                        floor_bound += 1;
                        ln_g[k] -= 1.0;
                    } else if relief > 0.0 {
                        ln_g[k] += (want / relief).ln();
                    }
                }
            }
            println!(
                "  the slope hold: {iters} sweeps on one tree, the worst node {worst_sh:.2} m, {floor_bound} nodes whose row lies under their own floor ({:.1} s)",
                t_sh.elapsed().as_secs_f64()
            );
        }
        // THE ADDITIVE HOLD (the 16 m tiles' own machinery, its cells the 8 km nodes): where the law cannot reach a node's
        // mean (its walls already on the threshold), the residual is taken off by a Gaussian over the nodes (σ half a node)
        // until every node inside stands within 0.25 m of its own mean.
        if HOLD.load(std::sync::atomic::Ordering::Relaxed) {
            let t_hold = Instant::now();
            let inside_node: Vec<bool> = (0..nw * nw)
                .map(|k| {
                    let (nx0, ny0) = ((k % nw) as f64 * node_m, (k / nw) as f64 * node_m);
                    nx0 >= x0 + inner - 1.0
                        && ny0 >= y0 + inner - 1.0
                        && nx0 + node_m <= x0 + wm - inner + 1.0
                        && ny0 + node_m <= y0 + hm - inner + 1.0
                })
                .collect();
            let (k_start, k_cells, k_w) = hold_kernel(
                n,
                |i| (sites[i].x / node_m - 0.5, sites[i].y / node_m - 0.5),
                nw,
                nw,
                &inside_node,
            );
            let (mut rounds, mut worst_h) = (0usize, f64::MAX);
            // ★ THE FILL LOOP (`--l1-fill`; the design's G-PIT): after the hold, every pit is filled to its spill height by
            // a priority flood from the outlets (a flat floor falling 1 mm a link, as a valley fills with sediment or
            // water), then the hold again — until the fill raises nothing over a centimetre.
            let (mut loops, mut raised_last, mut raised_pts) = (0usize, 0.0f64, 0usize);
            loop {
                while rounds < 60 * (loops + 1) && worst_h > 0.25 {
                    rounds += 1;
                    let (mut sum, mut area) = (vec![0.0f64; nw * nw], vec![0.0f64; nw * nw]);
                    for (i, st) in sites.iter().enumerate() {
                        let k = node_of(st);
                        sum[k] += h[i] * areas[i];
                        area[k] += areas[i];
                    }
                    let r: Vec<f64> = (0..nw * nw)
                        .map(|k| {
                            if inside_node[k] && area[k] > 0.0 {
                                sum[k] / area[k] - node_z[k]
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    worst_h = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    for (i, hi) in h.iter_mut().enumerate() {
                        let acc: f64 = (k_start[i]..k_start[i + 1])
                            .map(|t| f64::from(k_w[t]) * r[k_cells[t] as usize])
                            .sum();
                        *hi -= acc;
                    }
                }
                if !FILL.load(std::sync::atomic::Ordering::Relaxed) || loops >= 40 {
                    break;
                }
                loops += 1;
                // The priority flood from the outlets: each point no lower than its way out, plus a millimetre a link.
                let key = |z: f64| -> i64 { (z * 4_096.0).round() as i64 };
                let mut zf = h.clone();
                let mut done = vec![false; n];
                let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
                for i in 0..n {
                    if outlet[i] {
                        heap.push(Reverse((key(zf[i]), i)));
                    }
                }
                while let Some(Reverse((_, c))) = heap.pop() {
                    if done[c] {
                        continue;
                    }
                    done[c] = true;
                    for &(m, _) in graph.neighbors_of(c) {
                        if !done[m] && !outlet[m] {
                            let floor = zf[c] + 1.0e-3;
                            if zf[m] < floor {
                                zf[m] = floor;
                            }
                            heap.push(Reverse((key(zf[m]), m)));
                        }
                    }
                }
                raised_last = 0.0;
                raised_pts = 0;
                for i in 0..n {
                    let d = zf[i] - h[i];
                    if d > 0.01 {
                        raised_pts += 1;
                    }
                    raised_last = raised_last.max(d);
                }
                h.copy_from_slice(&zf);
                worst_h = f64::MAX;
                if raised_last <= 0.01 {
                    break;
                }
            }
            if FILL.load(std::sync::atomic::Ordering::Relaxed) {
                println!(
                    "  the fill loop: {loops} fills, the last raised {raised_pts} points by up to {raised_last:.2} m"
                );
            }
            // What the hold costs: the points steeper than the 35° threshold allows, and the pits it opens.
            let (_, recv, len) = tree(&h, &outlet, graph);
            let over = (0..n)
                .filter(|&i| {
                    recv[i] != usize::MAX && len[i] > 0.0 && (h[i] - h[recv[i]]) / len[i] > 0.8391
                })
                .count();
            let steep = (0..n)
                .filter(|&i| {
                    recv[i] != usize::MAX && len[i] > 0.0 && (h[i] - h[recv[i]]) / len[i] > 1.0
                })
                .count();
            let pits = (0..n)
                .filter(|&i| !outlet[i] && graph.neighbors_of(i).iter().all(|&(m, _)| h[m] > h[i]))
                .count();
            // The hollows' depths: each point's fill to its spill by a priority flood from the outlets.
            let depth = {
                let key = |z: f64| -> i64 { (z * 4_096.0).round() as i64 };
                let mut zf = h.clone();
                let mut done = vec![false; n];
                let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
                for i in 0..n {
                    if outlet[i] {
                        heap.push(Reverse((key(zf[i]), i)));
                    }
                }
                while let Some(Reverse((_, c))) = heap.pop() {
                    if done[c] {
                        continue;
                    }
                    done[c] = true;
                    for &(m, _) in graph.neighbors_of(c) {
                        if !done[m] && !outlet[m] {
                            if zf[m] < zf[c] {
                                zf[m] = zf[c];
                            }
                            heap.push(Reverse((key(zf[m]), m)));
                        }
                    }
                }
                (0..n).map(|i| zf[i] - h[i]).collect::<Vec<f64>>()
            };
            let deepest = depth.iter().copied().fold(0.0f64, f64::max);
            let (d1, d10, d100) = (
                depth.iter().filter(|&&d| d > 1.0).count(),
                depth.iter().filter(|&&d| d > 10.0).count(),
                depth.iter().filter(|&&d| d > 100.0).count(),
            );
            println!(
                "  the additive hold: {rounds} rounds, the worst node {worst_h:.2} m; over 40° {:.2} % of the points, over 45° {:.2} %; {pits} pits; the hollows: {d1} points under water over 1 m, {d10} over 10 m, {d100} over 100 m, the deepest {deepest:.1} m ({:.1} s)",
                100.0 * over as f64 / n as f64,
                100.0 * steep as f64 / n as f64,
                t_hold.elapsed().as_secs_f64()
            );
        }
        // The six worst held nodes: the target, the mean, the share of their points on the threshold slope, the rise.
        {
            let tan = threshold();
            let (_, recv, len) = tree(&h, &outlet, graph);
            let (mut sum, mut area, mut steep, mut cnt) = (
                vec![0.0f64; nw * nw],
                vec![0.0f64; nw * nw],
                vec![0usize; nw * nw],
                vec![0usize; nw * nw],
            );
            for (i, st) in sites.iter().enumerate() {
                let k = node_of(st);
                sum[k] += h[i] * areas[i];
                area[k] += areas[i];
                cnt[k] += 1;
                if recv[i] != usize::MAX
                    && len[i] > 0.0
                    && (h[i] - h[recv[i]]) / len[i] >= 0.999 * tan
                {
                    steep[k] += 1;
                }
            }
            let mut rows: Vec<(f64, usize)> = (0..nw * nw)
                .filter(|&k| held[k] && area[k] > 0.0)
                .map(|k| ((sum[k] / area[k] - node_z[k]).abs(), k))
                .collect();
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut low_pt = vec![(f64::MAX, usize::MAX); nw * nw];
            for (i, st) in sites.iter().enumerate() {
                let k = node_of(st);
                if h[i] < low_pt[k].0 {
                    low_pt[k] = (h[i], i);
                }
            }
            for &(_, k) in rows.iter().take(3) {
                let (ki, kj) = ((k % nw) as i64, (k / nw) as i64);
                let mut grid = String::new();
                for dj in -1..=1i64 {
                    for di in -1..=1i64 {
                        let (i, j) = (ki + di, kj + dj);
                        if i >= 0 && j >= 0 && i < nw as i64 && j < nw as i64 {
                            let m = (j as usize) * nw + i as usize;
                            grid +=
                                &format!(" {:>5.0}/{:>5.0}", node_z[m], sum[m] / area[m].max(1.0));
                        }
                    }
                    grid += " |";
                }
                // The water's path from the node's lowest point, node by node, to its outlet.
                let mut path = String::new();
                let (mut c, mut last_node) = (low_pt[k].1, usize::MAX);
                let mut climb = 0.0f64;
                while c != usize::MAX {
                    let m = node_of(&sites[c]);
                    if m != last_node {
                        path += &format!(" ({}, {}) {:.0}", m % nw, m / nw, h[c]);
                        last_node = m;
                    }
                    if recv[c] != usize::MAX {
                        climb = climb.max(h[recv[c]] - h[c]);
                    }
                    c = recv[c];
                }
                println!("    around ({ki}, {kj}) target/mean:{grid}");
                println!(
                    "      its lowest point {:.0} m; the water's path:{path}; the largest climb on it {climb:.0} m",
                    low_pt[k].0
                );
            }
            for &(_, k) in rows.iter().take(6) {
                let lower = (0..nw * nw)
                    .filter(|&m| {
                        let (di, dj) = (
                            (m % nw) as i64 - (k % nw) as i64,
                            (m / nw) as i64 - (k / nw) as i64,
                        );
                        di.abs() <= 1 && dj.abs() <= 1 && m != k && node_z[m] < node_z[k]
                    })
                    .count();
                println!(
                    "    node ({}, {}): target {:.0} m, mean {:.0} m, {:.0} % of points on the threshold, rise ×{:.2e}, {lower} lower neighbours, receiver {}",
                    k % nw,
                    k / nw,
                    node_z[k],
                    sum[k] / area[k],
                    100.0 * steep[k] as f64 / cnt[k].max(1) as f64,
                    u_node[k] / 1.0e-3,
                    if node_recv[k] == usize::MAX {
                        "none".to_string()
                    } else {
                        format!("({}, {})", node_recv[k] % nw, node_recv[k] / nw)
                    }
                );
            }
        }
        let (w, hh) = (
            (wm / spacing).round() as usize,
            (hm / spacing).round() as usize,
        );
        let t_raster = Instant::now();
        let terrain = model.create_terrain_from_result(&h);
        let raster = rasterise_rect(&terrain, x0, y0, w, hh, spacing);
        println!(
            "  the raster {:.1} s; the law before it {:.1} s",
            t_raster.elapsed().as_secs_f64(),
            t_law.elapsed().as_secs_f64()
        );
        println!(
            "  the 128 m domain ({:.1}, {:.1}) km, {:.1} × {:.1} km: {n} points, {outs} macro rivers leave, {ins} enter, the worst node's mean off by {worst:.0} m, inside the rim ring {worst_in:.0} m ({:.1} s)",
            x0 / 1_000.0,
            y0 / 1_000.0,
            wm / 1_000.0,
            hm / 1_000.0,
            started.elapsed().as_secs_f64()
        );
        DomainSolve {
            x0,
            y0,
            w,
            h: hh,
            cell: spacing,
            raster,
        }
    }

    /// ★ ONE DOMAIN OF THE SEAM TEST (B81, G-SEAM): the rectangle `(x0, y0, width, height)` (metres from the region's
    /// north-west corner) solved on the global cell points at `spacing`, CLOSED at its rim except where the coarser
    /// level's tree crosses it — an outgoing crossing an outlet at the coarser ground's height there, an incoming one the
    /// coarser water handed to the nearest point (the design's §3.4 rule 2); the rise and the erodibility read from the
    /// region's ONE field (the present rise read by the same weights in every direction, the erodibility so that the
    /// coarser `U/K` holds), relaxed over `span_yr` in `steps` with no per-window nudge.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_domain(
        coarse: &Out,
        node_rain_m: &[f64],
        nw: usize,
        node_m: f64,
        (x0, y0, wm, hm): (f64, f64, f64, f64),
        spacing: f64,
        rise_nodes: &[f64],
        span_yr: f64,
        steps: usize,
    ) -> DomainSolve {
        let started = Instant::now();
        let inside = |s: &Site2D| s.x > x0 && s.y > y0 && s.x < x0 + wm && s.y < y0 + hm;
        let (cx0, cy0) = ((x0 / spacing).floor() as i64, (y0 / spacing).floor() as i64);
        let (cx1, cy1) = (
            ((x0 + wm) / spacing).ceil() as i64,
            ((y0 + hm) / spacing).ceil() as i64,
        );
        let mut pts = Vec::new();
        for cy in cy0..cy1 {
            for cx in cx0..cx1 {
                let p = cell_point(cx, cy, spacing);
                if inside(&p) {
                    pts.push(p);
                }
            }
        }
        let model = TerrainModel2DBulider::default()
            .set_sites(pts)
            .set_bounding_box(
                Some(Site2D { x: x0, y: y0 }),
                Some(Site2D {
                    x: x0 + wm,
                    y: y0 + hm,
                }),
            )
            .build()
            .expect("the domain's graph builds");
        let (sites, areas, graph) = (
            model.sites().to_vec(),
            model.areas().to_vec(),
            model.graph(),
        );
        let n = sites.len();
        let coarse_at = |x: f64, y: f64| {
            coarse
                .terrain
                .get_elevation(&Site2D { x, y })
                .unwrap_or(0.0)
        };
        let mut base: Vec<f64> = sites.iter().map(|s| coarse_at(s.x, s.y)).collect();
        let mut water: Vec<f64> = sites
            .iter()
            .zip(&areas)
            .map(|(s, a)| linear(node_rain_m, nw, node_m, s.x, s.y) * a)
            .collect();
        let mut outlet: Vec<bool> = base.iter().map(|&b| b <= 0.0).collect();
        // The nearest point by a bucket grid four spacings wide.
        let bucket = 4.0 * spacing;
        let (bw, bh) = (
            (wm / bucket).ceil() as usize + 1,
            (hm / bucket).ceil() as usize + 1,
        );
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); bw * bh];
        let cell_of = |x: f64, y: f64| {
            (((x - x0) / bucket).max(0.0) as usize).min(bw - 1)
                + bw * (((y - y0) / bucket).max(0.0) as usize).min(bh - 1)
        };
        for (i, st) in sites.iter().enumerate() {
            buckets[cell_of(st.x, st.y)].push(i);
        }
        let nearest = |x: f64, y: f64| -> usize {
            let c = cell_of(x, y);
            let (bx, by) = (c % bw, c / bw);
            let mut best = (f64::MAX, 0usize);
            for j in by.saturating_sub(1)..=(by + 1).min(bh - 1) {
                for i in bx.saturating_sub(1)..=(bx + 1).min(bw - 1) {
                    for &k in &buckets[j * bw + i] {
                        let d = (sites[k].x - x).powi(2) + (sites[k].y - y).powi(2);
                        if d < best.0 || (d == best.0 && k < best.1) {
                            best = (d, k);
                        }
                    }
                }
            }
            best.1
        };
        let (mut outs, mut ins) = (0usize, 0usize);
        for (i, st) in coarse.sites.iter().enumerate() {
            let r = coarse.recv[i];
            if r == usize::MAX {
                continue;
            }
            let t = &coarse.sites[r];
            let (a_in, b_in) = (inside(st), inside(t));
            if a_in == b_in {
                continue;
            }
            let (pin, pout) = if a_in { (st, t) } else { (t, st) };
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..40 {
                let m = 0.5 * (lo + hi);
                let q = Site2D {
                    x: pin.x + (pout.x - pin.x) * m,
                    y: pin.y + (pout.y - pin.y) * m,
                };
                if inside(&q) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let c = Site2D {
                x: pin.x + (pout.x - pin.x) * lo,
                y: pin.y + (pout.y - pin.y) * lo,
            };
            let k = nearest(c.x, c.y);
            if a_in {
                outlet[k] = true;
                base[k] = coarse_at(c.x, c.y);
                outs += 1;
            } else if !outlet[k] {
                water[k] += coarse.q[i];
                ins += 1;
            }
        }
        if outs == 0 {
            // No coarser river leaves: the rim's lowest point drains it (the coarser ground's own low).
            let rim = |s: &Site2D| {
                s.x < x0 + spacing
                    || s.y < y0 + spacing
                    || s.x > x0 + wm - spacing
                    || s.y > y0 + hm - spacing
            };
            let k = (0..n)
                .filter(|&k| rim(&sites[k]))
                .min_by(|&a, &b| base[a].total_cmp(&base[b]).then(a.cmp(&b)));
            if let Some(k) = k {
                outlet[k] = true;
            }
        }
        // The one field: U = the present rise; K so the coarser U/K holds (the coarse solve's own ratio).
        let u_coarse: Vec<f64> = sites
            .iter()
            .map(|s| radial(&coarse.ln_u, nw, node_m, s.x, s.y).exp() / rock(s.x, s.y))
            .collect();
        let mut u: Vec<f64> = sites
            .iter()
            .map(|s| radial(rise_nodes, nw, node_m, s.x, s.y).max(0.0))
            .collect();
        let k: Vec<f64> = u
            .iter()
            .zip(&u_coarse)
            .map(|(r, uc)| JOINT_K_LP * r / uc.max(1.0e-30))
            .collect();
        // ★ THE COARSER LEVEL'S CHANNELS (`--l2-channels`; B83 item 5): this level refines the coarser ground, so it keeps
        // the coarser ground's rivers. A coarser link is a channel where THIS level's own law says its water cuts — the
        // river's steady slope `(U/(K·Q))^(1/n)` under the creep's steady slope at `ℓ = A/width` (Roering et al. 2007),
        // the same comparison the steady form makes at every point. Every point of this level within three quarters of a
        // spacing of a channel link becomes an outlet at the coarser ground's height there (linear along the link), so the
        // law builds only the hillslopes and gullies BETWEEN the coarser rivers.
        if L2_CHANNELS.load(std::sync::atomic::Ordering::Relaxed) {
            let d = super::r2::CREEP_K_M2_YR;
            let sc = vd_terrain::surface::SOIL_CRITICAL_SLOPE;
            let reach = 0.75 * spacing;
            let near_box = |s: &Site2D| {
                s.x > x0 - node_m * 0.25
                    && s.y > y0 - node_m * 0.25
                    && s.x < x0 + wm + node_m * 0.25
                    && s.y < y0 + hm + node_m * 0.25
            };
            let (mut links, mut channels, mut marked) = (0usize, 0usize, 0usize);
            for (c, st) in coarse.sites.iter().enumerate() {
                let r = coarse.recv[c];
                if r == usize::MAX || !near_box(st) {
                    continue;
                }
                links += 1;
                let rain = linear(node_rain_m, nw, node_m, st.x, st.y).max(f64::MIN_POSITIVE);
                let uc = radial(rise_nodes, nw, node_m, st.x, st.y).max(0.0);
                let ucoarse = radial(&coarse.ln_u, nw, node_m, st.x, st.y).exp() / rock(st.x, st.y);
                if uc <= 0.0 {
                    continue;
                }
                let kc = JOINT_K_LP * uc / ucoarse.max(1.0e-30);
                let qc = coarse.q[c].max(f64::MIN_POSITIVE);
                let s_river = (uc / (kc * qc)).powf(1.0 / JOINT_N);
                let ul = uc * (qc / rain) / spacing;
                let s_creep = sc * sc * ((d * d + (2.0 * ul / sc).powi(2)).sqrt() - d) / (2.0 * ul);
                if s_river >= s_creep {
                    continue;
                }
                channels += 1;
                let t = &coarse.sites[r];
                let (ha, hb) = (coarse_at(st.x, st.y), coarse_at(t.x, t.y));
                let (dx, dy) = (t.x - st.x, t.y - st.y);
                let len2 = (dx * dx + dy * dy).max(f64::MIN_POSITIVE);
                // The points near the segment: the buckets its box covers.
                let (lo_x, hi_x) = (st.x.min(t.x) - reach, st.x.max(t.x) + reach);
                let (lo_y, hi_y) = (st.y.min(t.y) - reach, st.y.max(t.y) + reach);
                if hi_x < x0 || hi_y < y0 || lo_x > x0 + wm || lo_y > y0 + hm {
                    continue;
                }
                let (b0, b1) = (cell_of(lo_x, lo_y), cell_of(hi_x, hi_y));
                for by in (b0 / bw)..=(b1 / bw) {
                    for bx in (b0 % bw)..=(b1 % bw) {
                        for &kk in &buckets[by * bw + bx] {
                            let p = &sites[kk];
                            let tt =
                                (((p.x - st.x) * dx + (p.y - st.y) * dy) / len2).clamp(0.0, 1.0);
                            let (px, py) = (st.x + tt * dx, st.y + tt * dy);
                            if (p.x - px).powi(2) + (p.y - py).powi(2) <= reach * reach {
                                let hz = ha + tt * (hb - ha);
                                if !outlet[kk] || hz < base[kk] {
                                    base[kk] = hz;
                                }
                                if !outlet[kk] {
                                    marked += 1;
                                }
                                outlet[kk] = true;
                            }
                        }
                    }
                }
            }
            println!(
                "  the coarser channels: {channels} of {links} coarser links cut by this level's law; {marked} points made outlets ({:.1} % of {n})",
                100.0 * marked as f64 / n as f64
            );
        }
        let mut h = base.clone();
        let _ = steady(graph, &outlet, &base, &water, &u_coarse, &mut h, 40);
        let t_solve = Instant::now();
        if STEADY_FORM.load(std::sync::atomic::Ordering::Relaxed) {
            let (passes, still) = steady_creep(
                graph, &outlet, &base, &water, &areas, &u, &k, &mut h, spacing, 60,
            );
            println!(
                "  the steady form: {passes} tree passes ({still} receivers still changing), {:.1} s",
                t_solve.elapsed().as_secs_f64()
            );
        } else {
            relax(
                &sites,
                &areas,
                graph,
                &outlet,
                &base,
                &water,
                &mut u,
                &mut h,
                steps,
                0.0,
                Some((&k, span_yr)),
            );
        }
        // ★ THE HOLD PER COARSER CELL, BY THE SLOPES (`--l2-hold-cells`; B83 item 5, B84's rule): this level must refine
        // the coarser ground, so over each coarser point's own Voronoi cell (irregular: no grid can print) its mean is
        // brought to the coarser surface's mean over the same points — by scaling the slopes along the FIXED flow tree
        // (each link still falls and the channels stay where they are, so no hollow can form), the factor each cell's own,
        // read at a point with Gaussian weights (σ half a coarser spacing) over the coarser cells near it.
        if L2_HOLD_CELLS.load(std::sync::atomic::Ordering::Relaxed) {
            let t_hc = Instant::now();
            let cs = coarse_spacing(coarse);
            let sigma = 0.5 * cs;
            let reach = 2.5 * sigma;
            // The coarser points near the domain, bucketed.
            let cb = cs;
            let (gx0, gy0) = (x0 - 2.0 * reach, y0 - 2.0 * reach);
            let (gw, gh) = (
                ((wm + 4.0 * reach) / cb).ceil() as usize + 1,
                ((hm + 4.0 * reach) / cb).ceil() as usize + 1,
            );
            let mut grid: Vec<Vec<usize>> = vec![Vec::new(); gw * gh];
            let gcell = |x: f64, y: f64| -> Option<usize> {
                let (a, b) = (((x - gx0) / cb).floor(), ((y - gy0) / cb).floor());
                (a >= 0.0 && b >= 0.0 && (a as usize) < gw && (b as usize) < gh)
                    .then(|| b as usize * gw + a as usize)
            };
            for (c, st) in coarse.sites.iter().enumerate() {
                if let Some(g) = gcell(st.x, st.y) {
                    grid[g].push(c);
                }
            }
            let around = |x: f64, y: f64, r: f64| -> Vec<usize> {
                let (a0, b0) = (
                    ((x - r - gx0) / cb).floor().max(0.0) as usize,
                    ((y - r - gy0) / cb).floor().max(0.0) as usize,
                );
                let (a1, b1) = (
                    (((x + r - gx0) / cb).floor() as usize).min(gw - 1),
                    (((y + r - gy0) / cb).floor() as usize).min(gh - 1),
                );
                let mut out = Vec::new();
                for b in b0..=b1 {
                    for a in a0..=a1 {
                        out.extend_from_slice(&grid[b * gw + a]);
                    }
                }
                out
            };
            // Each point's cell (its nearest coarser point) and its weights over the cells near it.
            let mut cell_of_pt = vec![usize::MAX; n];
            let (mut wstart, mut wcell, mut wval) =
                (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
            for (i, st) in sites.iter().enumerate() {
                wstart.push(wcell.len());
                let cands = around(st.x, st.y, reach);
                let mut best = (f64::MAX, usize::MAX);
                let first = wcell.len();
                let mut wsum = 0.0f64;
                for &c in &cands {
                    let cp = &coarse.sites[c];
                    let d2 = (cp.x - st.x).powi(2) + (cp.y - st.y).powi(2);
                    if d2 < best.0 || (d2 == best.0 && c < best.1) {
                        best = (d2, c);
                    }
                    if d2 <= reach * reach {
                        let wgt = (-d2 / (2.0 * sigma * sigma)).exp();
                        wcell.push(c);
                        wval.push(wgt);
                        wsum += wgt;
                    }
                }
                for w in &mut wval[first..] {
                    *w /= wsum.max(f64::MIN_POSITIVE);
                }
                cell_of_pt[i] = best.1;
            }
            wstart.push(wcell.len());
            // The cells this domain holds: those whose every point lies inside it (a cell cut by the rim is not held).
            let mut cells: Vec<usize> = cell_of_pt.clone();
            cells.sort_unstable();
            cells.dedup();
            let slot: std::collections::BTreeMap<usize, usize> =
                cells.iter().enumerate().map(|(s, &c)| (c, s)).collect();
            let nc = cells.len();
            let mut target = vec![0.0f64; nc];
            let mut carea = vec![0.0f64; nc];
            for i in 0..n {
                let sl = slot[&cell_of_pt[i]];
                target[sl] += coarse_at(sites[i].x, sites[i].y) * areas[i];
                carea[sl] += areas[i];
            }
            for sl in 0..nc {
                target[sl] /= carea[sl].max(f64::MIN_POSITIVE);
            }
            let rim_cell: Vec<bool> = {
                let mut r = vec![false; nc];
                for (i, st) in sites.iter().enumerate() {
                    if st.x < x0 + spacing
                        || st.y < y0 + spacing
                        || st.x > x0 + wm - spacing
                        || st.y > y0 + hm - spacing
                    {
                        r[slot[&cell_of_pt[i]]] = true;
                    }
                }
                r
            };
            let (order, recv, len) = tree_fast(&h, &outlet, graph);
            let slope0: Vec<f64> = (0..n)
                .map(|i| {
                    if outlet[i] || recv[i] == usize::MAX {
                        0.0
                    } else {
                        (h[i] - h[recv[i]]) / len[i]
                    }
                })
                .collect();
            // A cell joins the hold only with free points to move (four or more not outlets); the others keep the law.
            let mut free_pts = vec![0usize; nc];
            for i in 0..n {
                if !outlet[i] {
                    free_pts[slot[&cell_of_pt[i]]] += 1;
                }
            }
            let mut ln_g = vec![0.0f64; coarse.sites.len()];
            let (mut sweeps, mut worst, mut left) = (0usize, f64::MAX, 0usize);
            while sweeps < 200 && worst > 1.0 {
                sweeps += 1;
                for &i in &order {
                    if outlet[i] || recv[i] == usize::MAX {
                        h[i] = base[i];
                        continue;
                    }
                    let lg: f64 = (wstart[i]..wstart[i + 1])
                        .map(|t| wval[t] * ln_g[wcell[t]])
                        .sum();
                    h[i] = h[recv[i]] + slope0[i] * lg.exp() * len[i];
                }
                let (mut sum, mut low) = (vec![0.0f64; nc], vec![f64::MAX; nc]);
                for i in 0..n {
                    let sl = slot[&cell_of_pt[i]];
                    sum[sl] += h[i] * areas[i];
                    low[sl] = low[sl].min(h[i]);
                }
                worst = 0.0;
                left = 0;
                for sl in 0..nc {
                    if rim_cell[sl] {
                        continue;
                    }
                    let mean = sum[sl] / carea[sl].max(f64::MIN_POSITIVE);
                    let (relief, want) = (mean - low[sl], target[sl] - low[sl]);
                    if free_pts[sl] < 4 || relief <= 1.0 || want <= 1.0 {
                        left += 1;
                        continue;
                    }
                    worst = worst.max((mean - target[sl]).abs());
                    // Half the logarithmic step a sweep: neighbouring cells share points, so a full step overshoots.
                    ln_g[cells[sl]] += 0.5 * (want / relief).ln();
                }
            }
            let (gmin, gmax) = ln_g
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            println!(
                "  the hold per coarser cell: {nc} cells ({left} left to the law), {sweeps} sweeps on one tree, the worst held cell {worst:.2} m; the slope factors {:.2}–{:.2} ({:.1} s)",
                gmin.exp(),
                gmax.exp(),
                t_hc.elapsed().as_secs_f64()
            );
        }
        // ★ THE MEAN HOLD (the design's §3.3; B81: two tiles must stand at one level): the domain's mean over each coarser
        // cell (the coarse spacing, 128 m) is brought to the coarser ground's value there by subtracting the residuals,
        // read at each point with the same weights in every direction; rounds until the largest residual is under a
        // quarter metre, the rounds' count and the last residual printed (the design's one linear solve replaces it).
        if HOLD.load(std::sync::atomic::Ordering::Relaxed) {
            let t_hold = Instant::now();
            let cc = HOLD_CELL_M.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let (ncx, ncy) = ((wm / cc).round() as usize, (hm / cc).round() as usize);
            let cell_of = |s: &Site2D| -> Option<usize> {
                let (cx, cy) = (((s.x - x0) / cc).floor(), ((s.y - y0) / cc).floor());
                (cx >= 0.0 && cy >= 0.0 && (cx as usize) < ncx && (cy as usize) < ncy)
                    .then(|| cy as usize * ncx + cx as usize)
            };
            let target: Vec<f64> = (0..ncx * ncy)
                .map(|k| {
                    coarse_at(
                        x0 + ((k % ncx) as f64 + 0.5) * cc,
                        y0 + ((k / ncx) as f64 + 0.5) * cc,
                    )
                })
                .collect();
            let all_cells = vec![true; ncx * ncy];
            let (k_start, k_cells, k_w) = hold_kernel(
                sites.len(),
                |i| ((sites[i].x - x0) / cc - 0.5, (sites[i].y - y0) / cc - 0.5),
                ncx,
                ncy,
                &all_cells,
            );
            let (mut rounds, mut worst) = (0usize, f64::MAX);
            while rounds < 60 && worst > 0.25 {
                rounds += 1;
                let (mut sum, mut area) = (vec![0.0f64; ncx * ncy], vec![0.0f64; ncx * ncy]);
                for (i, s) in sites.iter().enumerate() {
                    if let Some(k) = cell_of(s) {
                        sum[k] += h[i] * areas[i];
                        area[k] += areas[i];
                    }
                }
                let r: Vec<f64> = (0..ncx * ncy)
                    .map(|k| sum[k] / area[k].max(f64::MIN_POSITIVE) - target[k])
                    .collect();
                worst = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for (i, hi) in h.iter_mut().enumerate() {
                    let acc: f64 = (k_start[i]..k_start[i + 1])
                        .map(|t| f64::from(k_w[t]) * r[k_cells[t] as usize])
                        .sum();
                    *hi -= acc;
                }
            }
            println!(
                "  the mean hold: {rounds} rounds, the largest cell residual {worst:.2} m ({:.1} s)",
                t_hold.elapsed().as_secs_f64()
            );
        }
        let (w, hh) = (
            (wm / spacing).round() as usize,
            (hm / spacing).round() as usize,
        );
        let terrain = model.create_terrain_from_result(&h);
        let raster = rasterise_rect(&terrain, x0, y0, w, hh, spacing);
        println!(
            "  the domain ({:.1}, {:.1}) km, {:.1} × {:.1} km: {n} points, {outs} coarse rivers leave, {ins} enter ({:.1} s)",
            x0 / 1_000.0,
            y0 / 1_000.0,
            wm / 1_000.0,
            hm / 1_000.0,
            started.elapsed().as_secs_f64()
        );
        DomainSolve {
            x0,
            y0,
            w,
            h: hh,
            cell: spacing,
            raster,
        }
    }

    /// ★ THE FINER WINDOW (B80 step 2): the same law on points `spacing_m` apart inside the square `(x0, y0, side)` of a
    /// coarse solve. The window's edge points are outlets at the coarse ground's height; every coarse point outside whose
    /// water flows in (its receiver inside) hands that water to the fine point nearest its receiver, so the big rivers
    /// enter with their whole catchment; the uplift is the coarse solve's own (its ln U read the same way); the rain the
    /// node's. Below about 32 m the law's own steady slope stands at the threshold, so the finer ground is made of the
    /// channel network the law cuts at that spacing.
    #[allow(clippy::too_many_arguments)]
    pub fn refine(
        coarse: &Out,
        node_rain_m: &[f64],
        nw: usize,
        node_m: f64,
        (x0, y0, side): (f64, f64, f64),
        spacing_m: f64,
        cell_m: f64,
    ) -> Window {
        let started = Instant::now();
        let num = ((side / spacing_m) * (side / spacing_m)) as usize;
        let model = TerrainModel2DBulider::from_random_sites(
            num,
            Site2D { x: x0, y: y0 },
            Site2D {
                x: x0 + side,
                y: y0 + side,
            },
        )
        .relaxate_sites(1)
        .expect("the sites relax")
        .add_edge_sites(None, None)
        .expect("the edge sites")
        .build()
        .expect("the point graph builds");
        let (sites, areas, graph) = (
            model.sites().to_vec(),
            model.areas().to_vec(),
            model.graph(),
        );
        let n = sites.len();
        let coarse_at = |x: f64, y: f64| {
            coarse
                .terrain
                .get_elevation(&Site2D { x, y })
                .unwrap_or(0.0)
        };
        let base: Vec<f64> = sites.iter().map(|s| coarse_at(s.x, s.y)).collect();
        let mut water: Vec<f64> = sites
            .iter()
            .zip(&areas)
            .map(|(s, a)| linear(node_rain_m, nw, node_m, s.x, s.y) * a)
            .collect();
        let mut outlet = vec![false; n];
        for &e in model.default_outlets() {
            outlet[e] = true;
        }
        for i in 0..n {
            outlet[i] |= base[i] <= 0.0;
        }
        // The nearest fine point by a bucket grid four spacings wide.
        let bucket = 4.0 * spacing_m;
        let bw = (side / bucket).ceil() as usize + 1;
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); bw * bw];
        for (i, s) in sites.iter().enumerate() {
            let (bx, by) = (
                (((s.x - x0) / bucket) as usize).min(bw - 1),
                (((s.y - y0) / bucket) as usize).min(bw - 1),
            );
            buckets[by * bw + bx].push(i);
        }
        let nearest = |x: f64, y: f64| -> usize {
            let (bx, by) = (
                (((x - x0) / bucket).max(0.0) as usize).min(bw - 1),
                (((y - y0) / bucket).max(0.0) as usize).min(bw - 1),
            );
            let mut best = (f64::MAX, 0usize);
            for j in by.saturating_sub(1)..=(by + 1).min(bw - 1) {
                for i in bx.saturating_sub(1)..=(bx + 1).min(bw - 1) {
                    for &k in &buckets[j * bw + i] {
                        let d = (sites[k].x - x).powi(2) + (sites[k].y - y).powi(2);
                        if d < best.0 {
                            best = (d, k);
                        }
                    }
                }
            }
            best.1
        };
        let inside = |s: &Site2D| s.x > x0 && s.y > y0 && s.x < x0 + side && s.y < y0 + side;
        let (mut entering, mut inflow) = (0usize, 0.0f64);
        for (i, s) in coarse.sites.iter().enumerate() {
            let r = coarse.recv[i];
            if r != usize::MAX && !inside(s) && inside(&coarse.sites[r]) {
                let k = nearest(coarse.sites[r].x, coarse.sites[r].y);
                if outlet[k] {
                    continue;
                }
                water[k] += coarse.q[i];
                entering += 1;
                inflow += coarse.q[i];
            }
        }
        let u: Vec<f64> = sites
            .iter()
            .map(|s| radial(&coarse.ln_u, nw, node_m, s.x, s.y).exp() / rock(s.x, s.y))
            .collect();
        let mut h = base.clone();
        let (passes, still, _, _) = steady(graph, &outlet, &base, &water, &u, &mut h, 40);
        let steps = RELAX_STEPS.load(std::sync::atomic::Ordering::Relaxed);
        if steps > 0 {
            // The window's own mean of the coarse ground: the creep's steady state keeps it (one factor on U).
            let target = sites
                .iter()
                .zip(&areas)
                .map(|(s, a)| coarse_at(s.x, s.y) * a)
                .sum::<f64>()
                / areas.iter().sum::<f64>();
            let mut u = u;
            relax(
                &sites, &areas, graph, &outlet, &base, &water, &mut u, &mut h, steps, target, None,
            );
        }
        let w = (side / cell_m).round() as usize;
        let terrain = model.create_terrain_from_result(&h);
        let raster = rasterise(&terrain, x0, y0, w, cell_m, coarse_at);
        println!(
            "  the window {:.1} km at ({:.1}, {:.1}) km: {n} points {spacing_m:.0} m apart, {entering} coarse rivers flow in \
             ({:.2e} m³/yr), {passes} tree passes ({still} still changing), {w} × {w} cells of {cell_m:.0} m ({:.1} s)",
            side / 1_000.0,
            x0 / 1_000.0,
            y0 / 1_000.0,
            inflow,
            started.elapsed().as_secs_f64()
        );
        Window {
            raster,
            w,
            x0,
            y0,
            cell_m,
        }
    }
}

/// The readings and the pictures of a generated raster, beside a reference raster of the same square (Earth's real
/// ground) when one is given.
fn lem_report(
    made: &[f64],
    reference: &[f64],
    ref_label: &str,
    w: usize,
    cell_m: f64,
    dir: Option<&str>,
    label: &str,
) {
    let mut views: Vec<(&str, &[f64])> = vec![("generated", made)];
    if !reference.is_empty() {
        views.push((ref_label, reference));
    }
    for (who, f) in &views {
        let rg = r2::plain_region(w, cell_m, f);
        let (straight, n) = r2::straight_share(&rg, f, 0.05, 16);
        let [x, y, d] = r2::stripes(&rg, f);
        let (lean, nl) = r2::lean(&rg, f, 0.05);
        let land: Vec<f64> = f.iter().copied().filter(|h| *h > 0.0).collect();
        println!(
            "  {}: THE STRAIGHT LINES {:.1} % of {n}; THE STRIPES {x:.3}/{y:.3}/{d:.3}; THE LEAN {:.1} % of {nl}; the land's \
             mean {:.0} m, highest {:.0} m",
            who.to_uppercase(),
            straight * 100.0,
            lean * 100.0,
            land.iter().sum::<f64>() / land.len().max(1) as f64,
            land.iter().copied().fold(0.0, f64::max)
        );
        if let Some(dir) = dir {
            for (what, file) in shade::write_views(f, w, 0, cell_m, 0.0, dir, label, who) {
                println!("  the shaded relief, {who}, {what}: {file}");
            }
        }
    }
    // THE VIEWS AT EYE HEIGHT: the made ground's two cameras, the same for the reference, so the pair compares.
    if let Some(dir) = dir {
        let dry = label.starts_with("home");
        for (name, cam) in view::cameras(made, w, cell_m) {
            println!(
                "  the camera '{name}': {:.1} km east, {:.1} km south, {:.0} m over the sea, bearing {:.0}°, pitch {:.1}°",
                cam.x / 1_000.0,
                -cam.n / 1_000.0,
                cam.z,
                cam.yaw.to_degrees(),
                cam.pitch.to_degrees()
            );
            for (who, f) in &views {
                let path = format!("{dir}/{label}_{who}_view_{name}.png");
                view::render(f, w, cell_m, &cam, dry, &path);
                println!("  the view, {who}, {name}: {path}");
            }
        }
    }
}

/// ★ THE VIEW AT EYE HEIGHT (B80 item 3, the owner's look): a small ray-marched picture of a raster (metres over the sea,
/// `w` × `w` cells of `cell_m`, x east, rows south), true scale, lit by one sun with its cast shadow, the sky's light,
/// aerial haze, and a paint by slope (rock past the threshold, soil or plants under it) — a measurement's picture, never
/// a game path. Two cameras chosen by a stated rule from a raster's highest land: in the valley 60 m over the ground
/// 12 km away, and over the range 1.5 km up 30 km away, each looking at the summit.
mod view {
    pub struct Camera {
        pub x: f64,
        pub n: f64,
        pub z: f64,
        pub yaw: f64,
        pub pitch: f64,
        /// The eye over each picture's own ground, metres (each raster stands the camera over its own ground).
        pub over: f64,
    }

    /// The raster's ground at east `x`, north `n` (metres; rows run south), read by Catmull-Rom over 4 × 4 cells so a
    /// coarse layer shows no flat facet per cell; `None` off the raster.
    pub fn ground(f: &[f64], w: usize, cell_m: f64, x: f64, n: f64) -> Option<f64> {
        let (u, v) = (x / cell_m - 0.5, -n / cell_m - 0.5);
        if u < 0.0 || v < 0.0 || u >= (w - 1) as f64 || v >= (w - 1) as f64 {
            return None;
        }
        let (iu, iv) = (u.floor() as i64, v.floor() as i64);
        let (fu, fv) = (u - iu as f64, v - iv as f64);
        let at = |a: i64, b: i64| {
            f[(b.clamp(0, w as i64 - 1) as usize) * w + a.clamp(0, w as i64 - 1) as usize]
        };
        let mut rows = [0.0f64; 4];
        for (r, slot) in rows.iter_mut().enumerate() {
            let b = iv + r as i64 - 1;
            *slot =
                super::catmull_rom([at(iu - 1, b), at(iu, b), at(iu + 1, b), at(iu + 2, b)], fu);
        }
        Some(super::catmull_rom(rows, fv))
    }

    /// A coarse raster with a finer window inside it (`fine`, `fw` cells of `fcell` from `(x0, y0)`, metres east and
    /// south), blended over the window's outer 500 m.
    #[allow(clippy::too_many_arguments)]
    pub fn layered<'a>(
        coarse: &'a [f64],
        w: usize,
        cell_m: f64,
        fine: &'a [f64],
        fw: usize,
        x0: f64,
        y0: f64,
        fcell: f64,
    ) -> impl Fn(f64, f64) -> Option<f64> + Sync + 'a {
        move |x, n| {
            let c = ground(coarse, w, cell_m, x, n)?;
            let side = fw as f64 * fcell;
            let (dx, dy) = (x - x0, -n - y0);
            if dx <= 0.0 || dy <= 0.0 || dx >= side || dy >= side {
                return Some(c);
            }
            let edge = dx.min(dy).min(side - dx).min(side - dy);
            let t = (edge / 500.0).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            let fv = ground(fine, fw, fcell, dx, -dy).unwrap_or(c);
            Some(c + (fv - c) * t)
        }
    }

    /// The cell a [`layered`] read takes its detail from at a point: the window's inside its blend, the coarse elsewhere.
    pub fn layered_cell(
        w_side_m: f64,
        x0: f64,
        y0: f64,
        fcell: f64,
        cell_m: f64,
    ) -> impl Fn(f64, f64) -> f64 + Sync {
        move |x, n| {
            let (dx, dy) = (x - x0, -n - y0);
            if dx > 250.0 && dy > 250.0 && dx < w_side_m - 250.0 && dy < w_side_m - 250.0 {
                fcell
            } else {
                cell_m
            }
        }
    }

    /// Writes a scene: the coarse raster, the fine window, the camera and the paint, as little-endian words.
    #[allow(clippy::too_many_arguments)]
    pub fn store_scene(
        path: &str,
        coarse: &[f64],
        w: usize,
        cell_m: f64,
        fine: &[f64],
        fw: usize,
        x0: f64,
        y0: f64,
        fcell: f64,
        cam: &Camera,
        dry: bool,
    ) {
        let mut b: Vec<u8> = Vec::with_capacity(8 * (coarse.len() + fine.len() + 16));
        for v in [
            w as f64,
            cell_m,
            fw as f64,
            x0,
            y0,
            fcell,
            cam.x,
            cam.n,
            cam.yaw,
            cam.pitch,
            cam.over,
            f64::from(u8::from(dry)),
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in coarse.iter().chain(fine) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, b).expect("the scene writes");
    }

    /// A stored scene: the coarse raster and its side and cell, the window and its side, origin and cell, the camera,
    /// the paint.
    pub struct Scene {
        pub coarse: Vec<f64>,
        pub w: usize,
        pub cell_m: f64,
        pub fine: Vec<f64>,
        pub fw: usize,
        pub x0: f64,
        pub y0: f64,
        pub fcell: f64,
        pub cam: Camera,
        pub dry: bool,
    }

    pub fn load_scene(path: &str) -> Scene {
        let b = std::fs::read(path).expect("the scene reads");
        let word =
            |k: usize| f64::from_le_bytes(b[8 * k..8 * k + 8].try_into().expect("eight bytes"));
        let (w, fw) = (word(0) as usize, word(2) as usize);
        let coarse: Vec<f64> = (0..w * w).map(|k| word(12 + k)).collect();
        let fine: Vec<f64> = (0..fw * fw).map(|k| word(12 + w * w + k)).collect();
        Scene {
            coarse,
            w,
            cell_m: word(1),
            fine,
            fw,
            x0: word(3),
            y0: word(4),
            fcell: word(5),
            cam: Camera {
                x: word(6),
                n: word(7),
                z: 0.0,
                yaw: word(8),
                pitch: word(9),
                over: word(10),
            },
            dry: word(11) > 0.5,
        }
    }

    /// The two cameras of the stated rule.
    pub fn cameras(f: &[f64], w: usize, cell_m: f64) -> Vec<(&'static str, Camera)> {
        let (mut best, mut top) = (0usize, f64::MIN);
        let margin = (2_000.0 / cell_m) as usize;
        for y in margin..w - margin {
            for x in margin..w - margin {
                if f[y * w + x] > top {
                    (best, top) = (y * w + x, f[y * w + x]);
                }
            }
        }
        let (px, pn) = (
            ((best % w) as f64 + 0.5) * cell_m,
            -(((best / w) as f64 + 0.5) * cell_m),
        );
        let mut out = Vec::new();
        for (name, dist, over) in [("valley", 12_000.0, 60.0), ("range", 30_000.0, 1_500.0)] {
            // Of sixteen bearings, the lowest ground at that distance (or at 0.8 or 0.6 of it where the raster ends).
            let mut pick: Option<(f64, f64, f64)> = None;
            for k in 0..16 {
                let a = f64::from(k) * std::f64::consts::TAU / 16.0;
                for d in [dist, dist * 0.8, dist * 0.6] {
                    let (x, n) = (px + d * a.cos(), pn + d * a.sin());
                    if let Some(g) = ground(f, w, cell_m, x, n) {
                        if pick.is_none_or(|p| g.max(0.0) < p.2) {
                            pick = Some((x, n, g.max(0.0)));
                        }
                        break;
                    }
                }
            }
            if let Some((x, n, g)) = pick {
                let z = g + over;
                let d = ((px - x).powi(2) + (pn - n).powi(2)).sqrt();
                out.push((
                    name,
                    Camera {
                        x,
                        n,
                        z,
                        yaw: (pn - n).atan2(px - x),
                        pitch: ((top - z) / d).atan() - 0.06,
                        over,
                    },
                ));
            }
        }
        out
    }

    /// Renders `f` from `cam` to `path`, a 60° field across.
    pub fn render(f: &[f64], w: usize, cell_m: f64, cam: &Camera, dry: bool, path: &str) {
        let sample = |x: f64, n: f64| ground(f, w, cell_m, x, n);
        let detail = |_: f64, _: f64| cell_m;
        render_with(&sample, &detail, 2.0 * w as f64 * cell_m, cam, dry, path);
    }

    /// Renders the ground `sample` gives (metres over the sea at east `x`, north `n`; `None` off the region) from `cam`
    /// to `path`, 1 280 × 720, a 60° field across; `detail_m` the finest spacing (the normals' step), `far` the march's end.
    pub fn render_with(
        sample: &(dyn Fn(f64, f64) -> Option<f64> + Sync),
        detail: &(dyn Fn(f64, f64) -> f64 + Sync),
        far: f64,
        cam: &Camera,
        dry: bool,
        path: &str,
    ) {
        let (width, height) = (1_280usize, 720usize);
        let eye_z = sample(cam.x, cam.n).unwrap_or(0.0).max(0.0) + cam.over;
        let cam = &Camera { z: eye_z, ..*cam };
        let tan_half = (30.0f64).to_radians().tan();
        let aspect = width as f64 / height as f64;
        let fwd = [
            cam.pitch.cos() * cam.yaw.cos(),
            cam.pitch.cos() * cam.yaw.sin(),
            cam.pitch.sin(),
        ];
        let right = [cam.yaw.sin(), -cam.yaw.cos(), 0.0];
        let up = [
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        ];
        // The sun from the camera's left, a little behind, 28° up.
        let sun_az = cam.yaw + 1.75;
        let sun = [
            22f64.to_radians().cos() * sun_az.cos(),
            22f64.to_radians().cos() * sun_az.sin(),
            22f64.to_radians().sin(),
        ];
        let visibility = 45_000.0;
        let mut pixels = vec![[0u8; 3]; width * height];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let rows_per = height.div_ceil(threads);
        std::thread::scope(|scope| {
            for (t, chunk) in pixels.chunks_mut(rows_per * width).enumerate() {
                let (fwd, right, up, sun) = (fwd, right, up, sun);
                scope.spawn(move || {
                    for (k, px) in chunk.iter_mut().enumerate() {
                        let (ix, iy) = (k % width, t * rows_per + k / width);
                        let sx = (2.0 * (ix as f64 + 0.5) / width as f64 - 1.0) * tan_half * aspect;
                        let sy = (1.0 - 2.0 * (iy as f64 + 0.5) / height as f64) * tan_half;
                        let mut d = [
                            fwd[0] + sx * right[0] + sy * up[0],
                            fwd[1] + sx * right[1] + sy * up[1],
                            fwd[2] + sx * right[2] + sy * up[2],
                        ];
                        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        d = [d[0] / l, d[1] / l, d[2] / l];
                        let sky = {
                            let e = d[2].max(0.0);
                            [0.62 - 0.30 * e, 0.72 - 0.22 * e, 0.86 - 0.06 * e]
                        };
                        let mut colour = sky;
                        let mut t_ray = 1.0;
                        let mut prev = 0.0;
                        while t_ray < far {
                            let (x, n, z) = (
                                cam.x + d[0] * t_ray,
                                cam.n + d[1] * t_ray,
                                cam.z + d[2] * t_ray,
                            );
                            let Some(g) = sample(x, n) else { break };
                            if z <= g {
                                // Refine the hit between the last two samples.
                                let (mut a, mut b) = (prev, t_ray);
                                for _ in 0..10 {
                                    let m = 0.5 * (a + b);
                                    let (x, n, z) =
                                        (cam.x + d[0] * m, cam.n + d[1] * m, cam.z + d[2] * m);
                                    if sample(x, n).is_some_and(|g| z <= g) {
                                        b = m;
                                    } else {
                                        a = m;
                                    }
                                }
                                let (x, n) = (cam.x + d[0] * b, cam.n + d[1] * b);
                                let h = detail(x, n) * 0.5;
                                let gx = (sample(x + h, n).unwrap_or(g)
                                    - sample(x - h, n).unwrap_or(g))
                                    / (2.0 * h);
                                let gn = (sample(x, n + h).unwrap_or(g)
                                    - sample(x, n - h).unwrap_or(g))
                                    / (2.0 * h);
                                let nl = (gx * gx + gn * gn + 1.0).sqrt();
                                let normal = [-gx / nl, -gn / nl, 1.0 / nl];
                                let slope = (gx * gx + gn * gn).sqrt();
                                let gz = sample(x, n).unwrap_or(g);
                                let rock = ((slope - 0.45) / 0.3).clamp(0.0, 1.0);
                                let soil = if dry {
                                    [0.55, 0.43, 0.30]
                                } else {
                                    [0.24, 0.32, 0.16]
                                };
                                let stone = if dry {
                                    [0.50, 0.40, 0.33]
                                } else {
                                    [0.36, 0.34, 0.31]
                                };
                                let mut albedo = [
                                    soil[0] + (stone[0] - soil[0]) * rock,
                                    soil[1] + (stone[1] - soil[1]) * rock,
                                    soil[2] + (stone[2] - soil[2]) * rock,
                                ];
                                if gz <= 0.0 {
                                    albedo = [0.08, 0.16, 0.24];
                                }
                                // The sun's cast shadow: march toward the sun.
                                let mut lit = 1.0;
                                let mut s = 4.0;
                                while s < 20_000.0 {
                                    let (sx, sn, sz) =
                                        (x + sun[0] * s, n + sun[1] * s, gz + 1.0 + sun[2] * s);
                                    match sample(sx, sn) {
                                        Some(sg) if sz < sg => {
                                            lit = 0.0;
                                            break;
                                        }
                                        Some(_) => {}
                                        None => break,
                                    }
                                    s += (s * 0.03).max(detail(x, n) * 0.5);
                                }
                                let ndl =
                                    (normal[0] * sun[0] + normal[1] * sun[1] + normal[2] * sun[2])
                                        .max(0.0);
                                let ambient = 0.30 * (0.6 + 0.4 * normal[2]);
                                let light = [
                                    1.00 * ndl * lit * 1.1 + ambient * 0.9,
                                    0.95 * ndl * lit * 1.1 + ambient * 0.95,
                                    0.85 * ndl * lit * 1.1 + ambient * 1.1,
                                ];
                                let surface = [
                                    albedo[0] * light[0],
                                    albedo[1] * light[1],
                                    albedo[2] * light[2],
                                ];
                                let haze = 1.0 - (-b / visibility).exp();
                                colour = [
                                    surface[0] + (sky[0] - surface[0]) * haze,
                                    surface[1] + (sky[1] - surface[1]) * haze,
                                    surface[2] + (sky[2] - surface[2]) * haze,
                                ];
                                break;
                            }
                            prev = t_ray;
                            t_ray += ((z - g) * 0.35).max(t_ray * 0.0015).max(0.5);
                        }
                        let enc = |c: f64| (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                        *px = [enc(colour[0]), enc(colour[1]), enc(colour[2])];
                    }
                });
            }
        });
        let mut img = image::RgbImage::new(width as u32, height as u32);
        for (k, p) in pixels.iter().enumerate() {
            img.put_pixel((k % width) as u32, (k / width) as u32, image::Rgb(*p));
        }
        img.save(path).expect("the view writes");
    }
}

/// ★ THE FINE WINDOW'S VIEW (B80 step 2): the valley camera's 8 km window ahead of the eye solved again at `fine_m`
/// (`lem::refine`), drawn inside the coarse ground with a 500 m blend at its edge; beside it, when an Earth tile is given,
/// Earth's own ground in the same window at the tile's 32 m. The fine window's shaded relief is written too.
#[allow(clippy::too_many_arguments)]
/// The fine window's side, metres (`--lem-window-km`).
static WINDOW_M: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(8_000);

fn lem_fine_views(
    out: &lem::Out,
    rain: &[f64],
    nw: usize,
    node_m: f64,
    cell_m: f64,
    reference: &[f64],
    earth_path: Option<&str>,
    dir: &str,
    label: &str,
    fine_m: f64,
    node_rise: Option<&[f64]>,
) {
    let w = out.w;
    let Some((_, cam)) = view::cameras(&out.raster, w, cell_m)
        .into_iter()
        .find(|(n, _)| *n == "valley")
    else {
        return;
    };
    let side = WINDOW_M.load(std::sync::atomic::Ordering::Relaxed) as f64;
    let region = w as f64 * cell_m;
    let ahead = 0.45 * side;
    let (cx, cy) = (
        cam.x + ahead * cam.yaw.cos(),
        -(cam.n + ahead * cam.yaw.sin()),
    );
    let x0 = ((cx - side / 2.0).clamp(0.0, region - side) / 32.0).floor() * 32.0;
    let y0 = ((cy - side / 2.0).clamp(0.0, region - side) / 32.0).floor() * 32.0;
    if let Some(nr) = node_rise {
        // The nodes whose square meets the window, each by its overlap.
        let (mut sum, mut area) = (0.0f64, 0.0f64);
        for k in 0..nw * nw {
            let (nx0, ny0) = ((k % nw) as f64 * node_m, (k / nw) as f64 * node_m);
            let ox = ((nx0 + node_m).min(x0 + side) - nx0.max(x0)).max(0.0);
            let oy = ((ny0 + node_m).min(y0 + side) - ny0.max(y0)).max(0.0);
            sum += nr[k] * ox * oy;
            area += ox * oy;
        }
        let rise = sum / area.max(1.0);
        println!(
            "  the fine window's present rise from the solve's laws: {:.4} mm/yr",
            rise * 1_000.0
        );
        lem::RISE_NM.store(
            (rise * 1.0e9).round() as usize,
            std::sync::atomic::Ordering::Relaxed,
        );
        lem::RISE_SET.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let win = lem::refine(out, rain, nw, node_m, (x0, y0, side), fine_m, fine_m / 2.0);
    let dry = label.starts_with("home");
    let made = view::layered(
        &out.raster,
        w,
        cell_m,
        &win.raster,
        win.w,
        win.x0,
        win.y0,
        win.cell_m,
    );
    // The close camera: 40 m over the highest ground inside the window (its outer 15 % left out), looking across the
    // window toward its centre (the valley camera's bearing when the summit stands at the centre), 11° down — a view
    // from a summit over the valleys, as the references frame theirs.
    let (wx, wy) = (x0 + side / 2.0, y0 + side / 2.0);
    let edge = (0.15 * win.w as f64) as usize;
    let (mut top, mut at) = (f64::MIN, (wx, wy));
    for y in edge..win.w - edge {
        for x in edge..win.w - edge {
            let z = win.raster[y * win.w + x];
            if z > top {
                top = z;
                at = (
                    win.x0 + (x as f64 + 0.5) * win.cell_m,
                    win.y0 + (y as f64 + 0.5) * win.cell_m,
                );
            }
        }
    }
    let (dx, dy) = (wx - at.0, wy - at.1);
    let yaw = if dx.hypot(dy) < 0.15 * side {
        cam.yaw
    } else {
        (-dy).atan2(dx)
    };
    let cam = view::Camera {
        x: at.0,
        n: -at.1,
        z: 0.0,
        yaw,
        pitch: -0.19,
        over: 40.0,
    };
    let path = format!("{dir}/{label}_generated_view_valley_fine.png");
    let detail = view::layered_cell(
        win.w as f64 * win.cell_m,
        win.x0,
        win.y0,
        win.cell_m,
        cell_m,
    );
    view::render_with(&made, &detail, 2.0 * region, &cam, dry, &path);
    // THE SCENE STORED (`--view` renders it again with any camera, no new solve).
    let scene = format!("{dir}/{label}_scene.bin");
    view::store_scene(
        &scene,
        &out.raster,
        w,
        cell_m,
        &win.raster,
        win.w,
        win.x0,
        win.y0,
        win.cell_m,
        &cam,
        dry,
    );
    println!("  the scene stored: {scene}");
    println!("  the view, generated, the valley through the fine window: {path}");
    for (what, file) in shade::write_views(
        &win.raster,
        win.w,
        0,
        win.cell_m,
        0.0,
        dir,
        label,
        "generated_fine",
    ) {
        println!("  the shaded relief, the fine window, generated, {what}: {file}");
    }
    if let Some(p) = earth_path {
        let (real, rw) = earth_cells(p, 32.0);
        let ew = (side / 32.0) as usize;
        let (ex, ey) = ((x0 / 32.0) as usize, (y0 / 32.0) as usize);
        let crop: Vec<f64> = (0..ew * ew)
            .map(|k| real[(ey + k / ew).min(rw - 1) * rw + (ex + k % ew).min(rw - 1)])
            .collect();
        let earth = view::layered(reference, w, cell_m, &crop, ew, x0, y0, 32.0);
        let path = format!("{dir}/{label}_earth_view_valley_fine.png");
        let detail = view::layered_cell(side, x0, y0, 32.0, cell_m);
        view::render_with(&earth, &detail, 2.0 * region, &cam, dry, &path);
        println!("  the view, earth, the valley through the fine window (Earth's 32 m): {path}");
        for (what, file) in shade::write_views(&crop, ew, 0, 32.0, 0.0, dir, label, "earth_fine") {
            println!("  the shaded relief, the fine window, earth, {what}: {file}");
        }
    }
}

/// ★ THE PRESENT RISE (B80; the macro study's recommendation 2a, no world change): each node's rock uplift rate today,
/// m/yr, from the solve's own laws on the seed's first land — the even share of the node's raise over the whole age, and
/// each belt wedge's rise over the last million years before the present (`wedge_height_m` at the age and a million
/// years before it). The rebound of the eroded rock (the solve's own, from its history) is NOT counted.
fn present_rise(
    body: &vd_terrain::body::BodyDefinition,
    lattice: &MacroLattice,
    words: &vd_terrain::solve::SolveWords,
) -> Vec<f64> {
    let mut land = vd_terrain::land::initial_land_on(
        body,
        lattice,
        &words.land(),
        vd_terrain::par::Cores::default(),
    );
    let age = words.age_yr;
    let belts = vd_terrain::belts::belts(body, lattice, words, &land, age);
    let even = vd_terrain::solve::belt_land(&belts, &mut land);
    let steps = f64::from(vd_terrain::solve::Z_STEPS_PER_M);
    let mut rise: Vec<f64> = even
        .iter()
        .map(|&e| e as f64 / steps / age.max(1) as f64)
        .collect();
    let span = 1_000_000u64.min(age);
    for w in &belts.wedge {
        let h = |t: u64| {
            vd_terrain::belts::wedge_height_m(w, vd_terrain::belts::wedge_area_m2(w, t))
                .0
                .to_f64()
        };
        rise[w.node as usize] += (h(age) - h(age - span)) / span as f64;
    }
    rise
}

/// The present rise's map over the land (`--rise-map`): the share of the land that rises, its spread, and the fastest
/// land nodes with their face and place (for `--lem-at`).
fn rise_map(lattice: &MacroLattice, artifact: &Artifact, rise: &[f64]) {
    let sea = f64::from(artifact.sea_m);
    let land: Vec<usize> = (0..rise.len())
        .filter(|&k| f64::from(artifact.rows[k].z_m) > sea)
        .collect();
    let mut on_land: Vec<f64> = land.iter().map(|&k| rise[k]).collect();
    on_land.sort_by(f64::total_cmp);
    let share = |t: f64| {
        100.0 * on_land.iter().filter(|&&r| r > t).count() as f64 / on_land.len().max(1) as f64
    };
    let q = |p: f64| on_land[((on_land.len() - 1) as f64 * p) as usize] * 1_000.0;
    println!(
        "PRESENT RISE over {} land nodes: rising {:.2} %, over 0.01 mm/yr {:.2} %, over 0.1 {:.2} %, over 0.5 {:.2} %; \
         median {:.5} mm/yr, 99th {:.3}, the fastest {:.3}",
        on_land.len(),
        share(0.0),
        share(1.0e-5),
        share(1.0e-4),
        share(5.0e-4),
        q(0.5),
        q(0.99),
        q(1.0)
    );
    let mut fast = land.clone();
    fast.sort_by(|a, b| rise[*b].total_cmp(&rise[*a]));
    let one = f64::from(1u32 << 20) * f64::from(1u32 << 20);
    for &k in fast.iter().take(12) {
        let (face, i, j) = lattice.split(k as u32);
        let d = lattice.direction(k as u32);
        let v = [
            d[0].raw() as f64 / one,
            d[1].raw() as f64 / one,
            d[2].raw() as f64 / one,
        ];
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        println!(
            "  node {k}: --lem-at {},{i},{j}  rise {:.3} mm/yr, {:.0} m over the sea, rain {:.0} mm/yr, {:.2}° N {:.2}° E",
            face.index(),
            rise[k] * 1_000.0,
            f64::from(artifact.rows[k].z_m) - sea,
            unlog(artifact.rows[k].rain, 4),
            (v[2] / r).asin().to_degrees(),
            v[1].atan2(v[0]).to_degrees()
        );
    }
}

/// The cut's band each side of a shared edge, metres: half the buffer (`--lem-buffer-m`).
static CUT_BAND_M: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(512);
/// The blend's width across a cut, cells (`--cut-blend`).
static CUT_BLEND_CELLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(3);

/// A rectangle's shaded relief (a measurement's picture): Horn's slope on the 3 × 3 window, the sun from the north-west
/// 45° up, a vertical exaggeration of two, a tan tint by height; `w` × `h` cells of `cell` metres.
fn hillshade_rect(f: &[f64], w: usize, h: usize, cell: f64, path: &str) {
    let (lo, hi) = f
        .iter()
        .filter(|v| v.is_finite())
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let at = |x: i64, y: i64| {
        f[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize]
    };
    let sun = {
        let (az, el) = (315f64.to_radians(), 45f64.to_radians());
        [el.cos() * az.sin(), el.cos() * az.cos(), el.sin()]
    };
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let dzdx = ((at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1)))
                / (8.0 * cell)
                * 2.0;
            let dzdn = ((at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1))
                - (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)))
                / (8.0 * cell)
                * 2.0;
            let l = (dzdx * dzdx + dzdn * dzdn + 1.0).sqrt();
            let shade = ((-dzdx * sun[0] - dzdn * sun[1] + sun[2]) / l).max(0.0);
            let t = ((at(x, y) - lo) / (hi - lo).max(1.0)).clamp(0.0, 1.0);
            let base = [0.55 + 0.35 * t, 0.45 + 0.30 * t, 0.32 + 0.25 * t];
            let px = |c: f64| ((c * (0.25 + 0.85 * shade)).clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([px(base[0]), px(base[1]), px(base[2])]),
            );
        }
    }
    img.save(path).expect("the relief writes");
}

/// ★ THE SEAM TEST'S READINGS AND PICTURES (B81, G-SEAM — blocking): the two tiles A (west) and B (east) solved apart,
/// stitched by the minimum-error cut (Efros & Freeman 2001: one column a row, the least squared difference of the two
/// solutions, moving at most one column a row, inside half the buffer of the shared edge; ties to the western column),
/// beside ONE joint solve `j` of both: the step the cut hides, the slope break across the cut against the joint surface
/// at the same places, the pits near the cut, the two surfaces' difference, the maps and two views.
#[allow(clippy::too_many_arguments)]
/// ★ THE TILE CODER (the design §4.4, P-SIZE): each sample in quarter metres predicted from its decoded left `a`, upper
/// `b` and upper-left `c` neighbours by LOCO-I's median rule (`min(a,b)` if `c ≥ max(a,b)`, `max(a,b)` if `c ≤ min(a,b)`,
/// else `a + b − c`; the first row from the left, the first column from above), the residual zigzagged and written as a
/// variable-length word, a run of zero residuals as one zero and its length. A tile decodes alone. Returns the bytes.
fn tile_encode(q: &[i64], w: usize, h: usize) -> Vec<u8> {
    let pred = |q: &[i64], x: usize, y: usize| -> i64 {
        match (x, y) {
            (0, 0) => 0,
            (_, 0) => q[x - 1],
            (0, _) => q[(y - 1) * w],
            _ => {
                let (a, b, c) = (q[y * w + x - 1], q[(y - 1) * w + x], q[(y - 1) * w + x - 1]);
                if c >= a.max(b) {
                    a.min(b)
                } else if c <= a.min(b) {
                    a.max(b)
                } else {
                    a + b - c
                }
            }
        }
    };
    let put = |out: &mut Vec<u8>, mut v: u64| loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    };
    let mut out = Vec::new();
    let mut zeros = 0u64;
    for y in 0..h {
        for x in 0..w {
            let r = q[y * w + x] - pred(q, x, y);
            if r == 0 {
                zeros += 1;
                continue;
            }
            if zeros > 0 {
                put(&mut out, 0);
                put(&mut out, zeros - 1);
                zeros = 0;
            }
            put(&mut out, ((r << 1) ^ (r >> 63)) as u64);
        }
    }
    if zeros > 0 {
        put(&mut out, 0);
        put(&mut out, zeros - 1);
    }
    out
}

/// LOCO-I's median predictor at sample `k` of a raster `w` wide, from the samples already decoded.
fn loco_predict(q: &[i64], w: usize, k: usize) -> i64 {
    let (x, y) = (k % w, k / w);
    match (x, y) {
        (0, 0) => 0,
        (_, 0) => q[k - 1],
        (0, _) => q[k - w],
        _ => {
            let (a, b, c) = (q[k - 1], q[k - w], q[k - w - 1]);
            if c >= a.max(b) {
                a.min(b)
            } else if c <= a.min(b) {
                a.max(b)
            } else {
                a + b - c
            }
        }
    }
}

/// The Rice code's escape: a quotient of this many ones or more writes the residual as a plain 24-bit word, so no sample
/// costs more than `RICE_ESCAPE + 24` bits = 4 bytes (the design's stated worst case).
const RICE_ESCAPE: u64 = 8;

/// The Rice code's parameter from its running statistics (JPEG-LS: the least `k` with `n·2^k ≥ a`).
fn rice_k(a: u64, n: u64) -> u32 {
    let mut k = 0u32;
    while (n << k) < a {
        k += 1;
    }
    k
}

/// ★ THE TILE CODER, ADAPTIVE RICE (P-SIZE; JPEG-LS's code, Weinberger, Seroussi & Sapiro 2000, our own code): LOCO-I's
/// prediction, the residual zigzagged, written as its quotient by `2^k` in ones ended by a zero and its `k` low bits; `k`
/// follows the running mean of the residuals' size (halved every 64 samples). A tile decodes alone.
fn rice_encode(q: &[i64], w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut acc, mut nbits) = (0u64, 0u32);
    let mut put = |out: &mut Vec<u8>, v: u64, bits: u32| {
        for i in (0..bits).rev() {
            acc = (acc << 1) | ((v >> i) & 1);
            nbits += 1;
            if nbits == 8 {
                out.push(acc as u8);
                acc = 0;
                nbits = 0;
            }
        }
    };
    let (mut a, mut n) = (4u64, 1u64);
    for k in 0..w * h {
        let r = q[k] - loco_predict(q, w, k);
        let z = ((r << 1) ^ (r >> 63)) as u64;
        let kk = rice_k(a, n);
        let quot = z >> kk;
        if quot >= RICE_ESCAPE {
            put(&mut out, (1u64 << RICE_ESCAPE) - 1, RICE_ESCAPE as u32);
            put(&mut out, z, 24);
        } else {
            put(&mut out, ((1u64 << quot) - 1) << 1, quot as u32 + 1);
            put(&mut out, z & ((1u64 << kk) - 1), kk);
        }
        a += z;
        n += 1;
        if n == 64 {
            a >>= 1;
            n >>= 1;
        }
    }
    put(&mut out, 0, 7);
    out
}

/// The Rice coder's inverse.
fn rice_decode(bytes: &[u8], w: usize, h: usize) -> Vec<i64> {
    let mut q = vec![0i64; w * h];
    let mut bit = 0usize;
    let mut get = |bits: u32| -> u64 {
        let mut v = 0u64;
        for _ in 0..bits {
            v = (v << 1) | u64::from((bytes[bit / 8] >> (7 - bit % 8)) & 1);
            bit += 1;
        }
        v
    };
    let (mut a, mut n) = (4u64, 1u64);
    for k in 0..w * h {
        let kk = rice_k(a, n);
        let mut quot = 0u64;
        while quot < RICE_ESCAPE && get(1) == 1 {
            quot += 1;
        }
        let z = if quot == RICE_ESCAPE {
            get(24)
        } else {
            (quot << kk) | get(kk)
        };
        let r = ((z >> 1) as i64) ^ -((z & 1) as i64);
        q[k] = loco_predict(&q, w, k) + r;
        a += z;
        n += 1;
        if n == 64 {
            a >>= 1;
            n >>= 1;
        }
    }
    q
}

/// What a better code could reach on the same residuals: their zeroth-order entropy, and the length of an ADAPTIVE RICE
/// code (JPEG-LS's: `k` from the running mean of the residuals' size, reset by halving every 64 samples), in bytes.
fn coder_bounds(q: &[i64], w: usize, h: usize) -> (f64, f64) {
    let mut res = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let p = match (x, y) {
                (0, 0) => 0,
                (_, 0) => q[x - 1],
                (0, _) => q[(y - 1) * w],
                _ => {
                    let (a, b, c) = (q[y * w + x - 1], q[(y - 1) * w + x], q[(y - 1) * w + x - 1]);
                    if c >= a.max(b) {
                        a.min(b)
                    } else if c <= a.min(b) {
                        a.max(b)
                    } else {
                        a + b - c
                    }
                }
            };
            res.push(q[y * w + x] - p);
        }
    }
    let mut counts: std::collections::BTreeMap<i64, u64> = std::collections::BTreeMap::new();
    for &r in &res {
        *counts.entry(r).or_insert(0) += 1;
    }
    let n = res.len() as f64;
    let entropy_bits: f64 = counts
        .values()
        .map(|&c| -(c as f64) * (c as f64 / n).log2())
        .sum();
    let (mut a, mut cnt, mut bits) = (4u64, 1u64, 0u64);
    for &r in &res {
        let z = ((r << 1) ^ (r >> 63)) as u64;
        let mut k = 0u32;
        while (cnt << k) < a {
            k += 1;
        }
        bits += (z >> k) + 1 + u64::from(k);
        a += z;
        cnt += 1;
        if cnt == 64 {
            a >>= 1;
            cnt >>= 1;
        }
    }
    (entropy_bits / 8.0, bits as f64 / 8.0)
}

/// The coder's inverse (`tile_encode`): the samples from the bytes alone.
fn tile_decode(bytes: &[u8], w: usize, h: usize) -> Vec<i64> {
    let mut q = vec![0i64; w * h];
    let mut at = 0usize;
    let mut get = || -> u64 {
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let byte = bytes[at];
            at += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return v;
            }
            shift += 7;
        }
    };
    let mut k = 0usize;
    let mut zeros = 0u64;
    while k < w * h {
        let (x, y) = (k % w, k / w);
        let p = match (x, y) {
            (0, 0) => 0,
            (_, 0) => q[k - 1],
            (0, _) => q[k - w],
            _ => {
                let (a, b, c) = (q[k - 1], q[k - w], q[k - w - 1]);
                if c >= a.max(b) {
                    a.min(b)
                } else if c <= a.min(b) {
                    a.max(b)
                } else {
                    a + b - c
                }
            }
        };
        let r = if zeros > 0 {
            zeros -= 1;
            0
        } else {
            let z = get();
            if z == 0 {
                zeros = get();
                0
            } else {
                ((z >> 1) as i64) ^ -((z & 1) as i64)
            }
        };
        q[k] = p + r;
        k += 1;
    }
    q
}

/// ★ P-SIZE on a solved domain: the near tile (the tile and half its buffer, `keep` cells each side of the tile) and the
/// far levels (the tile's 2 × 2 and 4 × 4 means), each coded and decoded back exactly; the bytes a sample and the parts
/// of 32 KB.
fn p_size(
    raster: &[f64],
    w: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile: usize,
    keep: usize,
    label: &str,
) {
    let crop = |x0: usize, y0: usize, side: usize| -> Vec<i64> {
        let mut q = Vec::with_capacity(side * side);
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                q.push((raster[y * w + x] * 4.0).round() as i64);
            }
        }
        q
    };
    let near_side = tile + 2 * keep;
    let near = crop(tile_x0 - keep, tile_y0 - keep, near_side);
    let near_bytes = tile_encode(&near, near_side, near_side);
    assert_eq!(
        tile_decode(&near_bytes, near_side, near_side),
        near,
        "the near tile decodes exactly"
    );
    let means = |f: usize| -> Vec<i64> {
        let side = tile / f;
        let mut q = Vec::with_capacity(side * side);
        for y in 0..side {
            for x in 0..side {
                let mut sum = 0.0;
                for dy in 0..f {
                    for dx in 0..f {
                        sum += raster[(tile_y0 + y * f + dy) * w + tile_x0 + x * f + dx];
                    }
                }
                q.push((sum / (f * f) as f64 * 4.0).round() as i64);
            }
        }
        q
    };
    let (m2, m4) = (means(2), means(4));
    let (b2, b4) = (
        tile_encode(&m2, tile / 2, tile / 2),
        tile_encode(&m4, tile / 4, tile / 4),
    );
    assert_eq!(
        tile_decode(&b2, tile / 2, tile / 2),
        m2,
        "the 2 × 2 level decodes exactly"
    );
    assert_eq!(
        tile_decode(&b4, tile / 4, tile / 4),
        m4,
        "the 4 × 4 level decodes exactly"
    );
    let far = b2.len() + b4.len();
    let rn = rice_encode(&near, near_side, near_side);
    assert_eq!(
        rice_decode(&rn, near_side, near_side),
        near,
        "the Rice near tile decodes exactly"
    );
    let (r2e, r4e) = (
        rice_encode(&m2, tile / 2, tile / 2),
        rice_encode(&m4, tile / 4, tile / 4),
    );
    assert_eq!(
        rice_decode(&r2e, tile / 2, tile / 2),
        m2,
        "the Rice 2 × 2 level decodes exactly"
    );
    assert_eq!(
        rice_decode(&r4e, tile / 4, tile / 4),
        m4,
        "the Rice 4 × 4 level decodes exactly"
    );
    println!(
        "  P-SIZE {label}, THE ADAPTIVE RICE CODER (written and decoded back exactly): the near tile {:.0} KB ({:.2} bytes a sample, {} parts of 32 KB); the far levels {:.0} KB",
        rn.len() as f64 / 1024.0,
        rn.len() as f64 / (near_side * near_side) as f64,
        rn.len().div_ceil(32 * 1024),
        (r2e.len() + r4e.len()) as f64 / 1024.0
    );
    let (ent_near, rice_near) = coder_bounds(&near, near_side, near_side);
    let (e2, r2b) = coder_bounds(&m2, tile / 2, tile / 2);
    let (e4, r4b) = coder_bounds(&m4, tile / 4, tile / 4);
    println!(
        "  P-SIZE {label}, a better code: the near tile's entropy {:.0} KB, an adaptive Rice code {:.0} KB; the far levels' entropy {:.0} KB, Rice {:.0} KB",
        ent_near / 1024.0,
        rice_near / 1024.0,
        (e2 + e4) / 1024.0,
        (r2b + r4b) / 1024.0
    );
    println!(
        "  P-SIZE {label}: the near tile {near_side} × {near_side} samples {:.0} KB ({:.2} bytes a sample, {} parts of 32 KB); the far levels {} × {} + {} × {} {:.0} KB ({:.2} bytes a sample); every level decodes exactly",
        near_bytes.len() as f64 / 1024.0,
        near_bytes.len() as f64 / (near_side * near_side) as f64,
        near_bytes.len().div_ceil(32 * 1024),
        tile / 2,
        tile / 2,
        tile / 4,
        tile / 4,
        far as f64 / 1024.0,
        far as f64 / (m2.len() + m4.len()) as f64
    );
}

/// ★ P-SWITCH and P-BAND on a solved domain (the design §4.5, Q7). THE LEVELS: the raster (rung 7 at 128 m) and its box
/// means of 2 × 2 cells, again and again, up to the node (rung 13, whose mean the hold makes the node's own). THE STEP of a
/// rung pair at a point: the finer level's sample against the coarser level read bilinearly at its place (the coarser
/// mesh's own surface), over the tile's samples; beside it the design's bound `S_c · 0.71 · (cell_r + cell_r+1)` and the
/// distance at which the measured step and the bound subtend one pixel of the reference view (45° over 720 rows).
/// THE BAND: the fine ground over and under its own node's mean (`node_z`), over the tile.
#[allow(clippy::too_many_arguments)]
fn p_switch_band(
    raster: &[f64],
    w: usize,
    cell: f64,
    tile_x0: usize,
    tile_y0: usize,
    tile: usize,
    node_cells: usize,
    node_z_at: impl Fn(usize, usize) -> f64,
    s_c: f64,
    label: &str,
) {
    let pixel = vd_core::geometry::REFERENCE_VIEW_FOV_Y_RAD / 720.0;
    // The levels over the whole domain (its side a multiple of the node).
    let mut levels: Vec<(usize, Vec<f64>)> = vec![(w, raster.to_vec())];
    while levels.last().map(|l| l.0).unwrap_or(0) > w / node_cells {
        let (lw, lv) = levels.last().expect("a level");
        let nw2 = lw / 2;
        let mut next = vec![0.0f64; nw2 * nw2];
        for y in 0..nw2 {
            for x in 0..nw2 {
                next[y * nw2 + x] = 0.25
                    * (lv[2 * y * lw + 2 * x]
                        + lv[2 * y * lw + 2 * x + 1]
                        + lv[(2 * y + 1) * lw + 2 * x]
                        + lv[(2 * y + 1) * lw + 2 * x + 1]);
            }
        }
        levels.push((nw2, next));
    }
    let bilinear = |lw: usize, lv: &[f64], c: f64, x: f64, y: f64| -> f64 {
        let (u, v) = (x / c - 0.5, y / c - 0.5);
        let (i, j) = (
            u.floor().clamp(0.0, (lw - 2) as f64) as usize,
            v.floor().clamp(0.0, (lw - 2) as f64) as usize,
        );
        let (fu, fv) = (
            (u - i as f64).clamp(0.0, 1.0),
            (v - j as f64).clamp(0.0, 1.0),
        );
        let at = |a: usize, b: usize| lv[b * lw + a];
        (at(i, j) * (1.0 - fu) + at(i + 1, j) * fu) * (1.0 - fv)
            + (at(i, j + 1) * (1.0 - fu) + at(i + 1, j + 1) * fu) * fv
    };
    println!(
        "  P-SWITCH {label}: the step between a rung and the next coarser one over the tile (S_c {s_c:.2})"
    );
    for r in 0..levels.len() - 1 {
        let (fw, fv) = (&levels[r].0, &levels[r].1);
        let (cw, cv) = (&levels[r + 1].0, &levels[r + 1].1);
        let (fc, cc) = (cell * (1 << r) as f64, cell * (1 << (r + 1)) as f64);
        let f = 1usize << r;
        let mut steps = Vec::new();
        for y in tile_y0 / f..(tile_y0 + tile) / f {
            for x in tile_x0 / f..(tile_x0 + tile) / f {
                let (px, py) = ((x as f64 + 0.5) * fc, (y as f64 + 0.5) * fc);
                steps.push((fv[y * fw + x] - bilinear(*cw, cv, cc, px, py)).abs());
            }
        }
        steps.sort_by(f64::total_cmp);
        let max = steps[steps.len() - 1];
        let p99 = steps[steps.len() * 99 / 100];
        let bound = s_c * 0.71 * (fc + cc);
        println!(
            "    rung {} → {} ({:.0} m → {:.0} m): the step max {max:.0} m, 99th {p99:.0} m; the bound {bound:.0} m; one pixel at {:.0} km (measured max) / {:.0} km (bound) / {:.0} km (the coarser cell)",
            7 + r,
            8 + r,
            fc,
            cc,
            max / pixel / 1_000.0,
            bound / pixel / 1_000.0,
            cc / pixel / 1_000.0
        );
    }
    // THE LAST SWAP, to the rows: each node's box mean over the tile against its row, in pixels where an 8 km cell is one.
    {
        let (lw, lv) = levels.last().expect("the node level");
        let c_node = cell * node_cells as f64;
        let mut worst = 0.0f64;
        for j in tile_y0 / node_cells..(tile_y0 + tile) / node_cells {
            for i in tile_x0 / node_cells..(tile_x0 + tile) / node_cells {
                worst = worst.max((lv[j * lw + i] - node_z_at(i, j)).abs());
            }
        }
        println!(
            "    the node's mean against its row: worst {worst:.0} m — {:.3} of a pixel at the rows' switch ({:.0} km)",
            worst / (c_node / pixel) / pixel,
            c_node / pixel / 1_000.0
        );
    }
    let (mut over, mut under) = (f64::MIN, f64::MAX);
    for y in tile_y0..tile_y0 + tile {
        for x in tile_x0..tile_x0 + tile {
            let d = raster[y * w + x] - node_z_at(x / node_cells, y / node_cells);
            over = over.max(d);
            under = under.min(d);
        }
    }
    println!(
        "  P-BAND {label}: the fine ground over its node's mean up to {over:.0} m, under it down to {:.0} m",
        -under
    );
}

fn seam_report(
    a: &lem::DomainSolve,
    b: &lem::DomainSolve,
    j: &lem::DomainSolve,
    ax0: f64,
    ay0: f64,
    node_m: f64,
    dir: &str,
    label: &str,
) {
    let c = a.cell;
    let g = |ds: &lem::DomainSolve, gx: i64, gy: i64| -> f64 {
        let (lx, ly) = (
            gx - (ds.x0 / c).round() as i64,
            gy - (ds.y0 / c).round() as i64,
        );
        if lx < 0 || ly < 0 || lx >= ds.w as i64 || ly >= ds.h as i64 {
            f64::NAN
        } else {
            ds.raster[ly as usize * ds.w + lx as usize]
        }
    };
    let (gx0, gy0) = ((ax0 / c).round() as i64, (ay0 / c).round() as i64);
    let (wc, hc) = (
        (2.0 * node_m / c).round() as usize,
        (node_m / c).round() as usize,
    );
    let ge = gx0 + (node_m / c).round() as i64;
    let band = (CUT_BAND_M.load(std::sync::atomic::Ordering::Relaxed) as f64 / c).round() as i64;
    let cols: Vec<i64> = (ge - band..=ge + band).collect();
    let mut cost = vec![vec![0.0f64; cols.len()]; hc];
    let mut from = vec![vec![0usize; cols.len()]; hc];
    for y in 0..hc {
        for (k, &x) in cols.iter().enumerate() {
            let d = g(a, x, gy0 + y as i64) - g(b, x, gy0 + y as i64);
            let e = if d.is_finite() { d * d } else { 1.0e12 };
            if y == 0 {
                cost[0][k] = e;
                continue;
            }
            let mut best = (f64::MAX, k);
            for kk in k.saturating_sub(1)..=(k + 1).min(cols.len() - 1) {
                if cost[y - 1][kk] < best.0 {
                    best = (cost[y - 1][kk], kk);
                }
            }
            cost[y][k] = e + best.0;
            from[y][k] = best.1;
        }
    }
    let mut cut = vec![0i64; hc];
    let mut k = (0..cols.len())
        .min_by(|&p, &q| cost[hc - 1][p].total_cmp(&cost[hc - 1][q]).then(p.cmp(&q)))
        .expect("a column");
    for y in (0..hc).rev() {
        cut[y] = cols[k];
        k = from[y][k];
    }
    let blend = CUT_BLEND_CELLS.load(std::sync::atomic::Ordering::Relaxed) as f64;
    let (mut st, mut jo) = (vec![0.0f64; wc * hc], vec![0.0f64; wc * hc]);
    for y in 0..hc {
        for x in 0..wc {
            let (gx, gy) = (gx0 + x as i64, gy0 + y as i64);
            let t = (((gx - cut[y]) as f64 / blend).clamp(-1.0, 1.0) * 0.5) + 0.5;
            let t = t * t * (3.0 - 2.0 * t);
            let (va, vb) = (g(a, gx, gy), g(b, gx, gy));
            st[y * wc + x] = if !va.is_finite() {
                vb
            } else if !vb.is_finite() {
                va
            } else {
                va * (1.0 - t) + vb * t
            };
            jo[y * wc + x] = g(j, gx, gy);
        }
    }
    let pct = |mut v: Vec<f64>, p: f64| -> f64 {
        v.retain(|x| x.is_finite());
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            f64::NAN
        } else {
            v[((v.len() - 1) as f64 * p) as usize]
        }
    };
    // The step the cut hides, and the slope break across it, beside the joint surface at the same places and inside A.
    let steps: Vec<f64> = (0..hc)
        .map(|y| (g(a, cut[y], gy0 + y as i64) - g(b, cut[y], gy0 + y as i64)).abs())
        .collect();
    let brk = |f: &[f64], col: &dyn Fn(usize) -> i64| -> Vec<f64> {
        (0..hc)
            .map(|y| {
                let x = (col(y) - gx0) as usize;
                if x < 3 || x + 3 >= wc {
                    return f64::NAN;
                }
                let r = y * wc;
                let west = (f[r + x - 1] - f[r + x - 3]) / (2.0 * c);
                let east = (f[r + x + 3] - f[r + x + 1]) / (2.0 * c);
                (east - west).abs()
            })
            .collect()
    };
    let at_cut = |y: usize| cut[y];
    let inner = |y: usize| cut[y] - (node_m / c / 3.0).round() as i64;
    let pits = |f: &[f64], near: bool| -> f64 {
        let (mut n, mut cells) = (0usize, 0usize);
        for y in 1..hc - 1 {
            for x in 1..wc - 1 {
                let gx = gx0 + x as i64;
                let in_band = (gx - cut[y]).abs() <= band;
                if in_band != near {
                    continue;
                }
                cells += 1;
                let v = f[y * wc + x];
                let mut low = true;
                for (dx, dy) in [
                    (-1i64, -1i64),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    let w2 = f[((y as i64 + dy) as usize) * wc + (x as i64 + dx) as usize];
                    if w2 <= v {
                        low = false;
                        break;
                    }
                }
                n += usize::from(low);
            }
        }
        n as f64 / (cells as f64 * c * c / 1.0e6).max(1.0e-9)
    };
    let diff: Vec<f64> = st.iter().zip(&jo).map(|(p, q)| (p - q).abs()).collect();
    println!(
        "SEAM TEST — two tiles of {:.1} km stitched by the cut, against one joint solve",
        node_m / 1_000.0
    );
    println!(
        "  the step the cut hides |A − B|: median {:.2} m, 99th {:.2} m; the cut wanders {} to {} columns from the edge",
        pct(steps.clone(), 0.5),
        pct(steps, 0.99),
        cut.iter().map(|x| x - ge).min().unwrap_or(0),
        cut.iter().map(|x| x - ge).max().unwrap_or(0)
    );
    println!(
        "  the slope break across the cut, 99th: stitched {:.3}, joint at the same places {:.3}, stitched inside A {:.3}",
        pct(brk(&st, &at_cut), 0.99),
        pct(brk(&jo, &at_cut), 0.99),
        pct(brk(&st, &inner), 0.99)
    );
    println!(
        "  the pits a km²: near the cut stitched {:.1}, joint {:.1}; away from it stitched {:.1}, joint {:.1}",
        pits(&st, true),
        pits(&jo, true),
        pits(&st, false),
        pits(&jo, false)
    );
    println!(
        "  stitched against joint: the mean difference {:.1} m, the 99th {:.1} m",
        diff.iter().filter(|v| v.is_finite()).sum::<f64>() / diff.len() as f64,
        pct(diff.clone(), 0.99)
    );
    let _ = std::fs::create_dir_all(dir);
    hillshade_rect(&st, wc, hc, c, &format!("{dir}/{label}_stitched.png"));
    hillshade_rect(&jo, wc, hc, c, &format!("{dir}/{label}_joint.png"));
    // The cut drawn on the stitched relief (a diagnosis only).
    let mut over = image::open(format!("{dir}/{label}_stitched.png"))
        .expect("the relief reads")
        .to_rgb8();
    for (y, &x) in cut.iter().enumerate() {
        over.put_pixel((x - gx0) as u32, y as u32, image::Rgb([220, 30, 30]));
    }
    over.save(format!("{dir}/{label}_stitched_cut.png"))
        .expect("the cut writes");
    // Two views across the seam, the same camera over both surfaces.
    let sample = |f: &[f64]| {
        let f = f.to_vec();
        move |x: f64, n: f64| -> Option<f64> {
            let (u, v) = ((x - ax0) / c - 0.5, (-n - ay0) / c - 0.5);
            if u < 1.0 || v < 1.0 || u >= (wc - 2) as f64 || v >= (hc - 2) as f64 {
                return None;
            }
            let (iu, iv) = (u.floor() as i64, v.floor() as i64);
            let at = |p: i64, q: i64| f[q as usize * wc + p as usize];
            let mut rows = [0.0f64; 4];
            for (r, slot) in rows.iter_mut().enumerate() {
                let q = (iv + r as i64 - 1).clamp(0, hc as i64 - 1);
                let px = |p: i64| at(p.clamp(0, wc as i64 - 1), q);
                *slot = catmull_rom([px(iu - 1), px(iu), px(iu + 1), px(iu + 2)], u - iu as f64);
            }
            Some(catmull_rom(rows, v - iv as f64))
        }
    };
    let detail = |_: f64, _: f64| c;
    let edge_x = ax0 + node_m;
    for (name, cam) in [
        (
            "from_west",
            view::Camera {
                x: edge_x - 3_000.0,
                n: -(ay0 + 0.8 * node_m),
                z: 0.0,
                yaw: 0.6,
                pitch: -0.2,
                over: 250.0,
            },
        ),
        (
            "from_east",
            view::Camera {
                x: edge_x + 3_000.0,
                n: -(ay0 + 0.8 * node_m),
                z: 0.0,
                yaw: std::f64::consts::PI - 0.6,
                pitch: -0.2,
                over: 250.0,
            },
        ),
    ] {
        for (who, f) in [("stitched", &st), ("joint", &jo)] {
            let s = sample(f);
            let path = format!("{dir}/{label}_{who}_view_{name}.png");
            view::render_with(&s, &detail, 3.0 * node_m, &cam, true, &path);
            println!("  the view, {who}, {name}: {path}");
        }
    }
}

/// ★ A FIELD ON THE GLOBAL CELL LATTICE (the seam tests): `w` × `h` cells of `cell` metres from the global cell
/// `(gx0, gy0)`; NaN where nothing is held.
#[derive(Clone)]
struct Field {
    gx0: i64,
    gy0: i64,
    w: usize,
    h: usize,
    cell: f64,
    v: Vec<f64>,
}

impl Field {
    fn of(ds: &lem::DomainSolve) -> Field {
        Field {
            gx0: (ds.x0 / ds.cell).round() as i64,
            gy0: (ds.y0 / ds.cell).round() as i64,
            w: ds.w,
            h: ds.h,
            cell: ds.cell,
            v: ds.raster.clone(),
        }
    }
    fn at(&self, gx: i64, gy: i64) -> f64 {
        let (x, y) = (gx - self.gx0, gy - self.gy0);
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            f64::NAN
        } else {
            self.v[y as usize * self.w + x as usize]
        }
    }
    /// The same field with its axes swapped (a horizontal cut is a vertical cut of the swapped fields).
    fn swapped(&self) -> Field {
        let mut v = vec![0.0f64; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                v[x * self.h + y] = self.v[y * self.w + x];
            }
        }
        Field {
            gx0: self.gy0,
            gy0: self.gx0,
            w: self.h,
            h: self.w,
            cell: self.cell,
            v,
        }
    }
}

/// ★ THE ONE STITCH (B81; Efros & Freeman 2001): two overlapping fields `a` (west) and `b` (east) meeting at the
/// global column `ge`; for every row the field holds, the cut is the column inside `ge ± band` where the two differ
/// least along a path that moves at most one column a row (ties to the western column, the first row's least-cost
/// start ties to the western column); each side draws its own field up to the cut, a smooth step `blend` cells wide
/// across it. The result covers both fields' columns over the rows they share. Returns the stitched field and the cut.
fn stitch_v(a: &Field, b: &Field, ge: i64, band: i64, blend: f64) -> (Field, Vec<i64>) {
    let gy0 = a.gy0.max(b.gy0);
    let gy1 = (a.gy0 + a.h as i64).min(b.gy0 + b.h as i64);
    let h = (gy1 - gy0).max(0) as usize;
    let cols: Vec<i64> = (ge - band..=ge + band).collect();
    let mut cost = vec![vec![0.0f64; cols.len()]; h];
    let mut from = vec![vec![0usize; cols.len()]; h];
    for y in 0..h {
        for (k, &x) in cols.iter().enumerate() {
            let d = a.at(x, gy0 + y as i64) - b.at(x, gy0 + y as i64);
            let e = if d.is_finite() { d * d } else { 1.0e12 };
            if y == 0 {
                cost[0][k] = e;
                continue;
            }
            let mut best = (f64::MAX, k);
            for kk in k.saturating_sub(1)..=(k + 1).min(cols.len() - 1) {
                if cost[y - 1][kk] < best.0 {
                    best = (cost[y - 1][kk], kk);
                }
            }
            cost[y][k] = e + best.0;
            from[y][k] = best.1;
        }
    }
    let mut cut = vec![ge; h];
    if h > 0 {
        let mut k = (0..cols.len())
            .min_by(|&p, &q| cost[h - 1][p].total_cmp(&cost[h - 1][q]).then(p.cmp(&q)))
            .expect("a column");
        for y in (0..h).rev() {
            cut[y] = cols[k];
            k = from[y][k];
        }
    }
    let gx0 = a.gx0.min(b.gx0);
    let gx1 = (a.gx0 + a.w as i64).max(b.gx0 + b.w as i64);
    let w = (gx1 - gx0) as usize;
    let mut v = vec![f64::NAN; w * h];
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = (gx0 + x as i64, gy0 + y as i64);
            let t = (((gx - cut[y]) as f64 / blend).clamp(-1.0, 1.0) * 0.5) + 0.5;
            let t = t * t * (3.0 - 2.0 * t);
            let (va, vb) = (a.at(gx, gy), b.at(gx, gy));
            v[y * w + x] = if !va.is_finite() {
                vb
            } else if !vb.is_finite() {
                va
            } else {
                va * (1.0 - t) + vb * t
            };
        }
    }
    (
        Field {
            gx0,
            gy0,
            w,
            h,
            cell: a.cell,
            v,
        },
        cut,
    )
}

/// The same stitch across a horizontal edge at the global row `ge` (`a` north, `b` south).
fn stitch_h(a: &Field, b: &Field, ge: i64, band: i64, blend: f64) -> (Field, Vec<i64>) {
    let (f, cut) = stitch_v(&a.swapped(), &b.swapped(), ge, band, blend);
    (f.swapped(), cut)
}

/// ★ THE CORNER TEST'S READINGS AND PICTURES (B81, G-SEAM): four tiles stitched by the one rule (each row along its
/// vertical cut, then the rows along the horizontal cut) against ONE joint solve of all four: the slope break along both
/// cuts against the joint at the same places, the pits within a kilometre of the corner, the difference, the maps.
#[allow(clippy::too_many_arguments)]
fn corner_report(
    tiles: [&lem::DomainSolve; 4],
    joint: &lem::DomainSolve,
    ax0: f64,
    ay0: f64,
    node_m: f64,
    dir: &str,
    label: &str,
) {
    let c = tiles[0].cell;
    let band = (CUT_BAND_M.load(std::sync::atomic::Ordering::Relaxed) as f64 / c).round() as i64;
    let blend = CUT_BLEND_CELLS.load(std::sync::atomic::Ordering::Relaxed) as f64;
    let (ge_x, ge_y) = (
        ((ax0 + node_m) / c).round() as i64,
        ((ay0 + node_m) / c).round() as i64,
    );
    let f: Vec<Field> = tiles.iter().map(|t| Field::of(t)).collect();
    let (north, cut_n) = stitch_v(&f[0], &f[1], ge_x, band, blend);
    let (south, cut_s) = stitch_v(&f[2], &f[3], ge_x, band, blend);
    let (all, cut_h) = stitch_h(&north, &south, ge_y, band, blend);
    let j = Field::of(joint);
    // The core: the four tiles without their buffers.
    let (cx0, cy0) = ((ax0 / c).round() as i64, (ay0 / c).round() as i64);
    let n = (2.0 * node_m / c).round() as usize;
    let mut st = vec![0.0f64; n * n];
    let mut jo = vec![0.0f64; n * n];
    for y in 0..n {
        for x in 0..n {
            st[y * n + x] = all.at(cx0 + x as i64, cy0 + y as i64);
            jo[y * n + x] = j.at(cx0 + x as i64, cy0 + y as i64);
        }
    }
    let pct = |mut v: Vec<f64>, p: f64| -> f64 {
        v.retain(|x| x.is_finite());
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            f64::NAN
        } else {
            v[((v.len() - 1) as f64 * p) as usize]
        }
    };
    // The slope break across a vertical cut at row y (column x) and across the horizontal cut at column x (row y).
    let brk_v = |g: &[f64], x: i64, y: i64| -> f64 {
        let (x, y) = ((x - cx0) as usize, (y - cy0) as usize);
        if x < 3 || x + 3 >= n || y >= n {
            return f64::NAN;
        }
        let r = y * n;
        (((g[r + x + 3] - g[r + x + 1]) - (g[r + x - 1] - g[r + x - 3])) / (2.0 * c)).abs()
    };
    let brk_h = |g: &[f64], x: i64, y: i64| -> f64 {
        let (x, y) = ((x - cx0) as usize, (y - cy0) as usize);
        if y < 3 || y + 3 >= n || x >= n {
            return f64::NAN;
        }
        (((g[(y + 3) * n + x] - g[(y + 1) * n + x]) - (g[(y - 1) * n + x] - g[(y - 3) * n + x]))
            / (2.0 * c))
            .abs()
    };
    let mut on_cuts = (Vec::new(), Vec::new());
    for (k, &x) in cut_n.iter().enumerate() {
        let y = north.gy0 + k as i64;
        on_cuts.0.push(brk_v(&st, x, y));
        on_cuts.1.push(brk_v(&jo, x, y));
    }
    for (k, &x) in cut_s.iter().enumerate() {
        let y = south.gy0 + k as i64;
        on_cuts.0.push(brk_v(&st, x, y));
        on_cuts.1.push(brk_v(&jo, x, y));
    }
    for (k, &y) in cut_h.iter().enumerate() {
        let x = all.gx0 + k as i64;
        on_cuts.0.push(brk_h(&st, x, y));
        on_cuts.1.push(brk_h(&jo, x, y));
    }
    // Pits within a kilometre of the corner, and in the tiles' far corners.
    let pits_near = |g: &[f64], px: i64, py: i64| -> f64 {
        let r = (1_000.0 / c) as i64;
        let (mut count, mut cells) = (0usize, 0usize);
        for y in (py - r).max(cy0 + 1)..(py + r).min(cy0 + n as i64 - 1) {
            for x in (px - r).max(cx0 + 1)..(px + r).min(cx0 + n as i64 - 1) {
                let (lx, ly) = ((x - cx0) as usize, (y - cy0) as usize);
                let v = g[ly * n + lx];
                cells += 1;
                let mut low = true;
                for (dx, dy) in [
                    (-1i64, -1i64),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    if g[((ly as i64 + dy) as usize) * n + (lx as i64 + dx) as usize] <= v {
                        low = false;
                        break;
                    }
                }
                count += usize::from(low);
            }
        }
        count as f64 / (cells as f64 * c * c / 1.0e6).max(1.0e-9)
    };
    let diff: Vec<f64> = st.iter().zip(&jo).map(|(p, q)| (p - q).abs()).collect();
    println!(
        "CORNER TEST — four tiles of {:.1} km stitched by the one rule, against one joint solve",
        node_m / 1_000.0
    );
    println!(
        "  the slope break along the three cuts, 99th: stitched {:.3}, joint at the same places {:.3}",
        pct(on_cuts.0, 0.99),
        pct(on_cuts.1, 0.99)
    );
    println!(
        "  the pits a km² within 1 km of the corner: stitched {:.1}, joint {:.1}; of a far corner: stitched {:.1}, joint {:.1}",
        pits_near(&st, ge_x, ge_y),
        pits_near(&jo, ge_x, ge_y),
        pits_near(&st, cx0 + (n / 6) as i64, cy0 + (n / 6) as i64),
        pits_near(&jo, cx0 + (n / 6) as i64, cy0 + (n / 6) as i64)
    );
    println!(
        "  stitched against joint: the mean difference {:.1} m, the 99th {:.1} m",
        diff.iter().filter(|v| v.is_finite()).sum::<f64>() / diff.len() as f64,
        pct(diff.clone(), 0.99)
    );
    let _ = std::fs::create_dir_all(dir);
    hillshade_rect(&st, n, n, c, &format!("{dir}/{label}_corner_stitched.png"));
    hillshade_rect(&jo, n, n, c, &format!("{dir}/{label}_corner_joint.png"));
    let mut over = image::open(format!("{dir}/{label}_corner_stitched.png"))
        .expect("the relief reads")
        .to_rgb8();
    let mut mark = |x: i64, y: i64| {
        let (lx, ly) = (x - cx0, y - cy0);
        if lx >= 0 && ly >= 0 && (lx as usize) < n && (ly as usize) < n {
            over.put_pixel(lx as u32, ly as u32, image::Rgb([220, 30, 30]));
        }
    };
    for (k, &x) in cut_n.iter().enumerate() {
        mark(x, north.gy0 + k as i64);
    }
    for (k, &x) in cut_s.iter().enumerate() {
        mark(x, south.gy0 + k as i64);
    }
    for (k, &y) in cut_h.iter().enumerate() {
        mark(all.gx0 + k as i64, y);
    }
    over.save(format!("{dir}/{label}_corner_stitched_cut.png"))
        .expect("the cut writes");
    println!(
        "  the maps: {dir}/{label}_corner_stitched.png, …_corner_joint.png, …_corner_stitched_cut.png"
    );
}
