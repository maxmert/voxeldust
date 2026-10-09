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
    let hillshade_dir = sarg("--hillshade");
    let label = sarg("--label").unwrap_or_else(|| "tile".to_string());

    println!("erosion_bench — R1, the bench (docs/investigation/2026-09-22, §5 R1)");
    println!("  the machine: Apple M4 Pro, one core, release, one job at a time");

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
        let g0 = build(&lattice, &artifact, pick, coarse_cells_per_node, halo);
        let s: Vec<f64> =
            g0.h.iter()
                .map(|v| v.raw() as f64 / f64::from(1 << H_BITS))
                .collect();
        let mut reg = r2::region(&lattice, &artifact, pick, coarse_cells_per_node, halo);
        reg.rho8 = args.iter().any(|a| a == "--r2-rho8");
        reg.dinf = reg.rho8 || args.iter().any(|a| a == "--r2-dinf");
        println!(
            "  the routing: {}",
            if reg.rho8 {
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
        sea: Vec<bool>,
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
    }

    /// A cell's own chance on `[0, 1)` (splitmix64 of its index; the bench's region index, so a tile's seam is not
    /// tested here — the build owes the global cell index).
    pub fn hash01(i: u64) -> f64 {
        let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
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
            },
            z,
        )
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
    }

    /// `steps` steps of `dt_yr` from `z`.
    pub fn run(reg: &Region, z0: &[f64], steps: usize, dt_yr: f64) -> Run {
        let w = reg.w;
        let cells = w * w;
        let mut z = z0.to_vec();
        let mut cut = vec![0.0f64; cells];
        let started = std::time::Instant::now();
        let mut residue = 0.0;
        for _ in 0..steps {
            if reg.dinf {
                let fl = dinf::route(reg, &z);
                let q = dinf::water(reg, &fl);
                dinf::erode(reg, &mut z, &mut cut, &fl, &q, dt_yr);
            } else {
                let (recv, order, filled) = route(reg, &z);
                let q = water(reg, &recv, &order);
                erode(reg, &mut z, &mut cut, &recv, &order, &filled, &q, dt_yr);
            }
            talus(reg, &mut z);
            residue = hold(reg, &mut z);
        }
        Run {
            z,
            cut,
            step_s: started.elapsed().as_secs_f64() / steps.max(1) as f64,
            hold_residue_m: residue,
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
    fn hold(reg: &Region, z: &mut [f64]) -> f64 {
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
                z[y * w + x] += catmull_rom(rows, fv);
            }
        }
        worst
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
        }

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
                for (e, p) in [(f.e1[i], f.p1[i]), (f.e2[i], f.p2[i])] {
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
