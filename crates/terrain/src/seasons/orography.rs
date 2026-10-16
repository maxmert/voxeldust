//! ★ L6 — THE MOUNTAINS' RAIN on the 8 km nodes (the design §3.6): the climate grid's vapour read to each node, and the
//! node's own share by the ground under it along the season's wind.
//!
//! **THE LAW.** Air over high ground is colder and rains its vapour out faster: the slow mode of the Linear Feedback
//! Precipitation Model (Hergarten & Robl 2022) in its HEIGHT form, `τ(H) = (β₀·e^{−H/H₀} + 2)·τ_c`, `H₀ = R_v·T²/(L·Γ)`
//! (Clausius–Clapeyron with the lapse rate), `τ_c` = 1 000 s (Smith & Barstad 2004's field value, named), `β₀` such that
//! the sea-level time is the vapour's measured residence time, 8.9 days (van der Ent & Tuinenburg 2017, named) — the
//! climate grid's own rain-out at the sea's level, so the two scales read ONE law.
//!
//! **THE DEPLETION ALONG THE WIND.** The node's vapour is the climate grid's column `W_c` times a factor `φ` that follows
//! the season's wind: `v·∇φ = 1/τ₀ − φ/τ(H)` — on low ground `φ` returns to one over the residence time, over high ground it
//! falls as the ground takes the vapour out. So the windward ramp rains and the lee lies in the depleted column. The
//! node's rain is `W_c·φ/τ(H)`.
//!
//! **THE DISCRETE FORM, LEAN-FREE IN DIRECTION.** The wind's upwind direction is split EXACTLY on the two neighbours of the
//! eight-neighbour ring that bracket it (`u = a·t̂_j + b·t̂_k`, `a, b ≥ 0`, each `t̂` the unit tangent toward the neighbour),
//! and the derivative along it is `a·(φ_i − φ_j)/L_j + b·(φ_i − φ_k)/L_k` with each neighbour's own chord — first-order
//! upwind on two directions, its direction exact for any wind, its numerical spread across the wind under one node.
//! Solved by Gauss–Seidel sweeps over the nodes in the four orders of each face's two axes (a node depends only on its
//! upwind neighbours, and one of the four orders runs with any wind), until no node moves more than a stated amount in
//! a round of four. WHERE TWO WINDS MEET, two neighbours each read the other — a pair that sweeps would relax by a
//! hundredth a sweep — so such a pair is solved EXACTLY together (its 2 × 2 system) whenever the sweep reaches it.
//! WHERE THE WIND TURNS ROUND A CENTRE (the heated winds' lows and highs), the upwind links close into small LOOPS of a few
//! nodes, and the vapour keeps all but a five-hundredth of itself a node, so a sweep relaxes such a loop by a few
//! thousandths a round: after each round the nodes still moving are taken, their loops found (the strongly connected parts
//! of their upwind links, Tarjan 1972) and each loop up to [`LOOP_EXACT`] nodes solved EXACTLY as one system, the loops
//! upwind first — the same answer the sweeps converge to, reached in fewer rounds. (MEASURED, 2026-10-15: sweeping the
//! whole planet in the search's own order instead is four times slower — the home planet's upwind links form ONE
//! connected part of 6–9 million nodes, and inside it the search's order follows the wind worse than the grid's.)
//! The cube's corner nodes have seven neighbours; their ring bridges the missing slot.
//!
//! **Example.** In the home planet's southern summer the wind blows up the belt's windward ramp at −41°: node by node up
//! the ramp the ground rises, `τ(H)` falls from 8.9 days toward one, `φ` falls as the ramp takes its rain, and the plateau
//! behind the crest receives a column a third as wet — the rain shadow.

use crate::gf::Gf;
use crate::macro_lattice::{MacroLattice, NO_NODE};

use super::accel::Anderson;
use super::balance::LATENT_J_KG;
use super::grid::{ClimateGrid, unit};
use super::vapour::{R_VAPOUR, RESIDENCE_S};

/// ★ THE CONDENSATION TIME, s (Smith & Barstad 2004's field value; named): LFPM's `τ_c`.
pub const CONDENSATION_S: f64 = 1_000.0;
/// The ring order of the eight stencil slots around a node (below-left, below, below-right, right, above-right, above,
/// above-left, left): angularly adjacent slots are adjacent here.
const RING: [usize; 8] = [0, 1, 2, 4, 7, 6, 5, 3];
/// The sweeps' stop: no node's factor moves more than this in a round of four sweeps — a ten-thousandth of the rain,
/// under the stored rain's whole millimetre at any rain under ten metres a year.
pub const SWEEP_TOL: f64 = 1e-4;
/// A bound on the rounds, never reached by a well-posed season (the bench reports the count).
pub const ROUND_BOUND: usize = 400;
/// The largest loop of still-moving nodes solved exactly as one system (its elimination's cost grows as the cube); a larger
/// one — a whole latitude's ring of westerlies, which the sweeps settle quickly — is left to the sweeps. A bound on the
/// work, never on the answer: the stop is the sweeps' own.
pub const LOOP_EXACT: usize = 512;

/// ★ THE RAIN-OUT TIME at a height, s: `τ(H) = (β₀·e^{−H/H₀} + 2)·τ_c`, `β₀ = τ_res/τ_c − 2`.
#[must_use]
pub fn rain_out_s(height_m: Gf, t_k: Gf, lapse_k_m: Gf) -> Gf {
    let tau_c = Gf::from_f64(CONDENSATION_S);
    let beta0 = Gf::from_f64(RESIDENCE_S / CONDENSATION_S) - Gf::TWO;
    let h0 = Gf::from_f64(R_VAPOUR) * t_k * t_k / (Gf::from_f64(LATENT_J_KG) * lapse_k_m);
    let h = height_m.greater(Gf::ZERO);
    (beta0 * (-(h / h0)).exp() + Gf::TWO) * tau_c
}

/// The node grid's fixed geometry: each node's unit direction, its eight neighbours, each neighbour's chord (m), its
/// climate cell, and its bilinear reading of the climate grid (computed once a body).
#[derive(Clone, Debug)]
pub struct NodeGeometry {
    pub dir: Vec<[Gf; 3]>,
    pub around: Vec<[u32; 8]>,
    pub chord: Vec<[Gf; 8]>,
    pub cell: Vec<u32>,
    pub reading: Vec<[(u32, Gf); 4]>,
}

impl NodeGeometry {
    /// The geometry of `fine` under the climate grid `grid`, on `threads` cores (each node's own, the same bytes on any).
    #[must_use]
    pub fn of(fine: &MacroLattice, grid: &ClimateGrid, threads: usize) -> NodeGeometry {
        let n = fine.node_count();
        let dir: Vec<[Gf; 3]> = crate::par::map(n, threads, |i| unit(fine.direction(i as u32)));
        let cell: Vec<u32> = crate::par::map(n, threads, |i| grid.cell_at(dir[i]) as u32);
        let reading: Vec<[(u32, Gf); 4]> = crate::par::map(n, threads, |i| {
            grid.reading(dir[i]).map(|(c, w)| (c as u32, w))
        });
        let around: Vec<[u32; 8]> = crate::par::map(n, threads, |i| fine.neighbours(i as u32));
        let chord: Vec<[Gf; 8]> = crate::par::map(n, threads, |i| {
            around[i].map(|m| {
                if m == NO_NODE {
                    Gf::ZERO
                } else {
                    Gf::from_i64(i64::from(fine.chord_m(i as u32, m)))
                }
            })
        });
        NodeGeometry {
            dir,
            around,
            chord,
            cell,
            reading,
        }
    }
}

/// A node's setup: its two upwind neighbours with their couplings, its rain-out rate, and its column.
type NodeSetup = ((u32, Gf, u32, Gf), Gf, Gf);

/// One season's mountains' rain: each node's rain, kg/m²/s; each climate cell's mean rain-out rate `⟨φ/τ(H)⟩` (1/s) for the
/// climate grid's next solve; the rounds the sweeps took and whether they met their stop.
#[derive(Clone, Debug)]
pub struct MountainRain {
    pub rain: Vec<Gf>,
    pub cell_rate: Vec<Gf>,
    pub rounds: usize,
    pub met: bool,
    /// The factor, in sixty-five-thousandths, for the next pass's warm start.
    pub phi: Vec<u16>,
    /// The bench's readings: the most nodes still moving after a round, the largest loop solved exactly, and the largest
    /// left to the sweeps (over [`LOOP_EXACT`]).
    pub still: usize,
    pub largest_loop: usize,
    pub largest_left: usize,
}

/// The warm start's unit: the factor `φ ∈ [0, 1]` kept in sixty-five-thousandths (its rain's change under a
/// ten-thousandth, under the stop).
pub const PHI_UNIT: i64 = 65_535;

/// ★ THE SEASON'S MOUNTAINS' RAIN on `fine`: the climate grid's season wind (`east`, `north`, m/s), temperature (K) and
/// column (`column`, kg/m²) read to each node; each node's ground over the sea, m (the sea at nought).
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "one season's fields on two grids; a struct would only rename them"
)]
pub fn mountain_rain(
    fine: &MacroLattice,
    geo: &NodeGeometry,
    grid: &ClimateGrid,
    east: &[Gf],
    north: &[Gf],
    t: &[Gf],
    column: &[Gf],
    height_m: &[Gf],
    lapse_k_m: Gf,
    threads: usize,
    phi_start: Option<&[u16]>,
) -> MountainRain {
    let n = fine.node_count();
    let tau0 = Gf::from_f64(RESIDENCE_S);
    let inv_tau0 = Gf::ONE / tau0;
    // Each node's two upwind neighbours and their couplings |v|·a/L, its own rain-out rate, and its column — each node's
    // own, so on all cores with the same bytes.
    let setup: Vec<NodeSetup> = crate::par::map(n, threads, |i| {
        let d = geo.dir[i];
        let mut ue = Gf::ZERO;
        let mut un = Gf::ZERO;
        let mut tk = Gf::ZERO;
        let mut wc = Gf::ZERO;
        for &(c, w) in &geo.reading[i] {
            let c = c as usize;
            ue += east[c] * w;
            un += north[c] * w;
            tk += t[c] * w;
            wc += column[c] * w;
        }
        let rate = Gf::ONE / rain_out_s(height_m[i], tk, lapse_k_m);
        let none = (NO_NODE, Gf::ZERO, NO_NODE, Gf::ZERO);
        // The node's own east and north (a node on the pole axis takes the face's).
        let ex = [-d[1], d[0], Gf::ZERO];
        let len = (ex[0] * ex[0] + ex[1] * ex[1]).sqrt();
        let (e, nn) = if len > Gf::ZERO {
            let e = [ex[0] / len, ex[1] / len, Gf::ZERO];
            let nn = [
                d[1] * e[2] - d[2] * e[1],
                d[2] * e[0] - d[0] * e[2],
                d[0] * e[1] - d[1] * e[0],
            ];
            (e, nn)
        } else {
            ([Gf::ONE, Gf::ZERO, Gf::ZERO], [Gf::ZERO, Gf::ONE, Gf::ZERO])
        };
        let speed = (ue * ue + un * un).sqrt();
        if speed <= Gf::ZERO {
            return (none, rate, wc);
        }
        // The upwind direction (where the air comes from), in (east, north).
        let (ux, uy) = (-ue / speed, -un / speed);
        let tangent = |m: u32| -> (Gf, Gf) {
            let dm = geo.dir[m as usize];
            let (x, y) = (
                (dm[0] - d[0]) * e[0] + (dm[1] - d[1]) * e[1] + (dm[2] - d[2]) * e[2],
                (dm[0] - d[0]) * nn[0] + (dm[1] - d[1]) * nn[1] + (dm[2] - d[2]) * nn[2],
            );
            let l = (x * x + y * y).sqrt();
            (x / l, y / l)
        };
        let around = geo.around[i];
        let mut live = [0usize; 8];
        let mut count = 0;
        for &s in &RING {
            if around[s] != NO_NODE {
                live[count] = s;
                count += 1;
            }
        }
        for k in 0..count {
            let (sj, sk) = (live[k], live[(k + 1) % count]);
            let (jx, jy) = tangent(around[sj]);
            let (kx, ky) = tangent(around[sk]);
            let det = jx * ky - jy * kx;
            if det.abs() <= Gf::ZERO {
                continue;
            }
            let a = (ux * ky - uy * kx) / det;
            let b = (jx * uy - jy * ux) / det;
            if a >= Gf::ZERO && b >= Gf::ZERO {
                return (
                    (
                        around[sj],
                        speed * a / geo.chord[i][sj],
                        around[sk],
                        speed * b / geo.chord[i][sk],
                    ),
                    rate,
                    wc,
                );
            }
        }
        (none, rate, wc)
    });
    let pair: Vec<(u32, Gf, u32, Gf)> = setup.iter().map(|s| s.0).collect();
    let rate: Vec<Gf> = setup.iter().map(|s| s.1).collect();
    let w_c: Vec<Gf> = setup.iter().map(|s| s.2).collect();
    drop(setup);
    // The sweeps, each node by its formula, a meeting pair by its 2 × 2 system; from the last pass's factor when there
    // is one (a warm start: the relief moves little in a pass).
    let mut phi: Vec<Gf> = match phi_start {
        Some(q) if q.len() == n => q
            .iter()
            .map(|&v| Gf::from_i64(i64::from(v)) / Gf::from_i64(PHI_UNIT))
            .collect(),
        _ => vec![Gf::ONE; n],
    };
    let edge = fine.edge as usize;
    let per_face = edge * edge;
    let tol = Gf::from_f64(SWEEP_TOL);
    let mut rounds = 0;
    let mut met = false;
    // A node's value as `A + B·φ_partner` given every other dependency's current value (`partner` excluded).
    let affine = |i: usize, partner: u32, phi: &[Gf]| -> (Gf, Gf) {
        let (j, cj, k, ck) = pair[i];
        let mut num = inv_tau0;
        let mut den = rate[i];
        let mut b = Gf::ZERO;
        for (m, c) in [(j, cj), (k, ck)] {
            if m == NO_NODE {
                continue;
            }
            den += c;
            if m == partner {
                b += c;
            } else {
                num += c * phi[m as usize];
            }
        }
        (num / den, b / den)
    };
    let reads = |a: usize, b: u32| {
        let (j, _, k, _) = pair[a];
        j == b || k == b
    };
    // The round in which each node last moved more than the stop, and the loops' working words.
    let mut stamp = vec![0u32; n];
    let mut loops = Loops::new(n);
    // ★ ANDERSON'S ACCELERATION (Walker & Ni 2011) over the rounds: a round `G` (the four sweeps and the loops' exact
    // solve) maps the factor `x` to `G(x)`; the next start is `G(x)` corrected by the last few rounds' differences so that
    // their combined change is least — GMRES's answer for a linear `G`, so the slow modes a round leaves settle in a few
    // rounds. The answer is still `G`'s own fixed point: the stop is the round's change, as before.
    let mut accel = Anderson::new();
    let mut g = phi.clone();
    while rounds < ROUND_BOUND {
        rounds += 1;
        let now = rounds as u32;
        g.copy_from_slice(&phi);
        let mut moved = Gf::ZERO;
        for order in 0..4 {
            for f in 0..6 {
                for jj in 0..edge {
                    let row = if order & 2 == 0 { jj } else { edge - 1 - jj };
                    for ii in 0..edge {
                        let col = if order & 1 == 0 { ii } else { edge - 1 - ii };
                        let i = f * per_face + row * edge + col;
                        let (j, _, k, _) = pair[i];
                        let partner = if j != NO_NODE && reads(j as usize, i as u32) {
                            j
                        } else if k != NO_NODE && reads(k as usize, i as u32) {
                            k
                        } else {
                            NO_NODE
                        };
                        if partner == NO_NODE {
                            let (x, _) = affine(i, NO_NODE, &g);
                            let d = (x - g[i]).abs();
                            if d >= tol {
                                stamp[i] = now;
                            }
                            moved = moved.greater(d);
                            g[i] = x;
                        } else {
                            let p = partner as usize;
                            let (ai, bi) = affine(i, partner, &g);
                            let (ap, bp) = affine(p, i as u32, &g);
                            let xi = (ai + bi * ap) / (Gf::ONE - bi * bp);
                            let xp = ap + bp * xi;
                            let (di, dp) = ((xi - g[i]).abs(), (xp - g[p]).abs());
                            if di.greater(dp) >= tol {
                                stamp[i] = now;
                                stamp[p] = now;
                            }
                            moved = moved.greater(di).greater(dp);
                            g[i] = xi;
                            g[p] = xp;
                        }
                    }
                }
            }
        }
        // The nodes still moving: their loops solved exactly, upwind first.
        let still: Vec<u32> = (0..n)
            .filter(|&i| stamp[i] == now)
            .map(|i| i as u32)
            .collect();
        loops.still = loops.still.max(still.len());
        loops.solve(&still, &pair, &rate, inv_tau0, &mut g);
        let change = phi
            .iter()
            .zip(&g)
            .fold(Gf::ZERO, |m, (a, b)| m.greater((*b - *a).abs()));
        if moved.greater(change) < tol {
            met = true;
            break;
        }
        accel.step(&mut phi, &g, change, (Gf::ZERO, Gf::ONE));
    }
    phi.copy_from_slice(&g);
    let rain: Vec<Gf> = (0..n).map(|i| w_c[i] * phi[i] * rate[i]).collect();
    // Each climate cell's mean rain-out rate ⟨φ/τ⟩ by its nodes' areas.
    let mut sum = vec![Gf::ZERO; grid.len()];
    let mut area = vec![Gf::ZERO; grid.len()];
    for i in 0..n {
        let c = geo.cell[i] as usize;
        let a = Gf::from_i64(fine.area_m2(i as u32) as i64);
        sum[c] += phi[i] * rate[i] * a;
        area[c] += a;
    }
    let cell_rate = (0..grid.len())
        .map(|c| {
            if area[c] > Gf::ZERO {
                sum[c] / area[c]
            } else {
                inv_tau0
            }
        })
        .collect();
    let phi_q = phi
        .iter()
        .map(|&v| {
            (v * Gf::from_i64(PHI_UNIT) + Gf::HALF)
                .to_i64_floor()
                .clamp(0, PHI_UNIT) as u16
        })
        .collect();
    MountainRain {
        rain,
        cell_rate,
        rounds,
        met,
        phi: phi_q,
        still: loops.still,
        largest_loop: loops.largest,
        largest_left: loops.left,
    }
}

/// Each node's two upwind neighbours and their couplings `|v|·a/L` (1/s).
type Upwind = (u32, Gf, u32, Gf);

/// The loops' working words, kept over the rounds: each node's place in the round's still set, Tarjan's words, and the
/// readings.
struct Loops {
    place: Vec<u32>,
    still: usize,
    largest: usize,
    left: usize,
}

const NOT_STILL: u32 = u32::MAX;

impl Loops {
    fn new(n: usize) -> Loops {
        Loops {
            place: vec![NOT_STILL; n],
            still: 0,
            largest: 0,
            left: 0,
        }
    }

    /// ★ THE STILL NODES' LOOPS SOLVED EXACTLY: the strongly connected parts of the still nodes' upwind links (each node
    /// reads at most two), found by Tarjan's algorithm, which closes a part only after every part it reads — so each is
    /// solved with its upwind parts already solved. A single node takes its formula; a loop up to [`LOOP_EXACT`] nodes
    /// its system `(1/τ + Σc)·φ_i − Σ_loop c·φ_m = 1/τ₀ + Σ_outside c·φ_m` by elimination with partial pivoting (the
    /// system is diagonally dominant: its rows' sums are the rain-out rates).
    fn solve(&mut self, still: &[u32], pair: &[Upwind], rate: &[Gf], inv_tau0: Gf, phi: &mut [Gf]) {
        for (k, &v) in still.iter().enumerate() {
            self.place[v as usize] = k as u32;
        }
        let m = still.len();
        // Tarjan's algorithm, iterative: the visit order, the lowest reachable order, the open stack.
        let mut order = vec![NOT_STILL; m];
        let mut low = vec![0u32; m];
        let mut open = vec![false; m];
        let mut stack: Vec<u32> = Vec::new();
        let mut calls: Vec<(u32, u8)> = Vec::new();
        let mut next = 0u32;
        let mut part: Vec<u32> = Vec::new();
        let mut largest = self.largest;
        let mut left = self.left;
        let place = &self.place;
        let reads = |k: usize| -> [u32; 2] {
            let (j, _, l, _) = pair[still[k] as usize];
            [j, l].map(|x| {
                if x == NO_NODE {
                    NOT_STILL
                } else {
                    place[x as usize]
                }
            })
        };
        for root in 0..m {
            if order[root] != NOT_STILL {
                continue;
            }
            calls.push((root as u32, 0));
            while let Some(&(v, e)) = calls.last() {
                let v = v as usize;
                if e == 0 {
                    order[v] = next;
                    low[v] = next;
                    next += 1;
                    stack.push(v as u32);
                    open[v] = true;
                }
                let r = reads(v);
                if (e as usize) < 2 {
                    calls.last_mut().expect("a call").1 = e + 1;
                    let w = r[e as usize];
                    if w == NOT_STILL {
                        continue;
                    }
                    let w = w as usize;
                    if order[w] == NOT_STILL {
                        calls.push((w as u32, 0));
                    } else if open[w] {
                        low[v] = low[v].min(order[w]);
                    }
                    continue;
                }
                calls.pop();
                if let Some(&(u, _)) = calls.last() {
                    let u = u as usize;
                    low[u] = low[u].min(low[v]);
                }
                if low[v] == order[v] {
                    part.clear();
                    loop {
                        let w = stack.pop().expect("the part's nodes") as usize;
                        open[w] = false;
                        part.push(still[w]);
                        if w == v {
                            break;
                        }
                    }
                    if part.len() > 1 {
                        if solve_part(&part, pair, rate, inv_tau0, phi) {
                            largest = largest.max(part.len());
                        } else {
                            left = left.max(part.len());
                        }
                    } else {
                        solve_part(&part, pair, rate, inv_tau0, phi);
                    }
                }
            }
        }
        self.largest = largest;
        self.left = left;
        for &v in still {
            self.place[v as usize] = NOT_STILL;
        }
    }
}

/// One strongly connected part solved with everything upwind of it held; false (left to the sweeps) over [`LOOP_EXACT`].
fn solve_part(part: &[u32], pair: &[Upwind], rate: &[Gf], inv_tau0: Gf, phi: &mut [Gf]) -> bool {
    let s = part.len();
    if s > LOOP_EXACT {
        return false;
    }
    let at = |v: u32| part.iter().position(|&x| x == v);
    let mut a = vec![Gf::ZERO; s * s];
    let mut b = vec![inv_tau0; s];
    for (r, &v) in part.iter().enumerate() {
        let (j, cj, k, ck) = pair[v as usize];
        let mut diag = rate[v as usize];
        for (w, c) in [(j, cj), (k, ck)] {
            if w == NO_NODE {
                continue;
            }
            diag += c;
            match at(w) {
                Some(q) => a[r * s + q] -= c,
                None => b[r] += c * phi[w as usize],
            }
        }
        a[r * s + r] += diag;
    }
    for col in 0..s {
        let p = (col..s)
            .max_by(|&x, &y| {
                a[x * s + col]
                    .abs()
                    .partial_cmp(&a[y * s + col].abs())
                    .expect("a number")
            })
            .expect("a row");
        if p != col {
            for q in 0..s {
                a.swap(col * s + q, p * s + q);
            }
            b.swap(col, p);
        }
        let piv = a[col * s + col];
        for r in col + 1..s {
            let l = a[r * s + col] / piv;
            if l == Gf::ZERO {
                continue;
            }
            for q in col..s {
                let v = a[col * s + q];
                a[r * s + q] -= l * v;
            }
            let v = b[col];
            b[r] -= l * v;
        }
    }
    for r in (0..s).rev() {
        let mut sum = b[r];
        for q in r + 1..s {
            sum -= a[r * s + q] * phi[part[q] as usize];
        }
        phi[part[r] as usize] = sum / a[r * s + r];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::home_moon;

    /// ★ THE RAIN-OUT TIME: the vapour's residence time at the sea's level, shorter on high ground.
    #[test]
    fn the_rain_out_time_falls_with_height() {
        let t = Gf::from_f64(288.0);
        let g = Gf::from_f64(0.0065);
        let sea = rain_out_s(Gf::ZERO, t, g).to_f64();
        assert!((sea / RESIDENCE_S - 1.0).abs() < 1e-12);
        let k2 = rain_out_s(Gf::from_f64(2_000.0), t, g).to_f64();
        let k4 = rain_out_s(Gf::from_f64(4_000.0), t, g).to_f64();
        assert!(k4 < k2);
        assert!(k2 < sea);
        // A pocket under the sea's level rains as the sea's level does.
        assert_eq!(rain_out_s(Gf::from_f64(-500.0), t, g).to_f64(), sea);
    }

    /// ★ THE WINDWARD SIDE RAINS, THE LEE LIES IN ITS SHADOW: the moon's lattice under a steady wind from the east, a
    /// ridge three kilometres high across it; flat ground far upwind rains the column over the residence time.
    #[test]
    fn a_ridge_takes_the_rain_on_its_windward_side() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let grid = ClimateGrid::new(lattice.radius_m(), 18);
        let geo = NodeGeometry::of(&lattice, &grid, 2);
        let n = lattice.node_count();
        let cells = grid.len();
        let east = vec![Gf::from_f64(-6.0); cells];
        let north = vec![Gf::ZERO; cells];
        let t = vec![Gf::from_f64(288.0); cells];
        let column = vec![Gf::from_f64(30.0); cells];
        // The ridge: the nodes whose direction's x component lies in a band (a north–south wall on the +X face's side).
        let mut height = vec![Gf::ZERO; n];
        for (i, d) in geo.dir.iter().enumerate() {
            let x = d[1].to_f64();
            if x.abs() < 0.03 && d[0].to_f64() > 0.5 {
                height[i] = Gf::from_f64(3_000.0);
            }
        }
        let r = mountain_rain(
            &lattice,
            &geo,
            &grid,
            &east,
            &north,
            &t,
            &column,
            &height,
            Gf::from_f64(0.0065),
            2,
            None,
        );
        assert!(r.met);
        // The wind from the east (−X·y… on the +X face the east is +y): windward is y > 0, the lee y < 0.
        let mean_at = |lo: f64, hi: f64| {
            let (mut s, mut k) = (0.0, 0.0);
            for (i, d) in geo.dir.iter().enumerate() {
                let y = d[1].to_f64();
                if d[0].to_f64() > 0.5 && d[2].to_f64().abs() < 0.2 && y >= lo && y < hi {
                    s += r.rain[i].to_f64();
                    k += 1.0;
                }
            }
            s / k
        };
        let ridge = mean_at(-0.03, 0.03);
        let lee = mean_at(-0.12, -0.05);
        let flat_far = mean_at(0.3, 0.4);
        let expected_flat = 30.0 / RESIDENCE_S;
        assert!(
            (flat_far / expected_flat - 1.0).abs() < 0.05,
            "{flat_far} against {expected_flat}"
        );
        assert!(
            ridge > 2.0 * flat_far,
            "the ridge {ridge} against flat ground {flat_far}"
        );
        assert!(
            lee < flat_far,
            "the lee {lee} against flat ground {flat_far}"
        );
        assert_eq!(r.phi.len(), n);
        // A warm start from the factor reaches the same rain.
        let again = mountain_rain(
            &lattice,
            &geo,
            &grid,
            &east,
            &north,
            &t,
            &column,
            &height,
            Gf::from_f64(0.0065),
            2,
            Some(&r.phi),
        );
        assert!(again.rounds <= r.rounds);
    }

    /// ★ A WIND THAT TURNS ROUND A CENTRE (a low's or a high's): a vortex on the moon's equator, its speed rising from
    /// the centre to 6 m/s ten degrees out and fading beyond. Its upwind links close into loops; from an empty column the
    /// rounds meet their stop, the loops are solved exactly, and the answer is the sweeps' own: a warm start from it keeps
    /// it. MEASURED (the control, the loops' solve taken out for one run): on this vortex 30 rounds against 28 with it
    /// (its loops are not the slow ones); on the home planet's sixteen seasons with the heated winds (`climate_bench
    /// home-s2 … res-eddy-gilldep-mtn`, three rounds) 2 300 rounds in all against 962, the worst seasons 202–250 rounds
    /// against 6–17.
    #[test]
    fn a_wind_round_a_centre_settles_by_its_loops() {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let grid = ClimateGrid::new(lattice.radius_m(), 18);
        let geo = NodeGeometry::of(&lattice, &grid, 2);
        let n = lattice.node_count();
        let cells = grid.len();
        let pi = Gf::from_f64(std::f64::consts::PI);
        let step = pi / Gf::from_i64(18);
        let sin_of = |x: Gf| super::super::grid::cos_series(x - pi * Gf::HALF);
        let s0 = sin_of(pi / Gf::from_i64(18));
        let (mut east, mut north) = (vec![Gf::ZERO; cells], vec![Gf::ZERO; cells]);
        for c in 0..cells {
            let (r, k) = (c / grid.cols, c % grid.cols);
            let (sl, cl) = (grid.sin_lat[r], grid.cos_lat[r]);
            let lon = (Gf::from_i64(k as i64) + Gf::HALF) * step;
            let (so, co) = (sin_of(lon), super::super::grid::cos_series(lon));
            let p = [cl * co, cl * so, sl];
            // The centre at latitude 0, longitude 90°: the axis (0, 1, 0); the turn a × p.
            if p[1] <= Gf::ZERO {
                continue;
            }
            let v = [p[2], Gf::ZERO, -p[0]];
            let s = (v[0] * v[0] + v[2] * v[2]).sqrt();
            if s <= Gf::ZERO {
                continue;
            }
            let speed = Gf::from_f64(6.0) * (s / s0) * (Gf::ONE - s / s0).exp();
            let e = [-so, co, Gf::ZERO];
            let no = [-sl * co, -sl * so, cl];
            let dot = |a: [Gf; 3], b: [Gf; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            east[c] = speed * dot(v, e) / s;
            north[c] = speed * dot(v, no) / s;
        }
        let t = vec![Gf::from_f64(288.0); cells];
        let column = vec![Gf::from_f64(30.0); cells];
        let height = vec![Gf::ZERO; n];
        let empty = vec![0u16; n];
        let run = |start: &[u16]| {
            mountain_rain(
                &lattice,
                &geo,
                &grid,
                &east,
                &north,
                &t,
                &column,
                &height,
                Gf::from_f64(0.0065),
                2,
                Some(start),
            )
        };
        let r = run(&empty);
        assert!(r.met, "{} rounds", r.rounds);
        assert!(r.largest_loop > 1, "the loops solved: {}", r.largest_loop);
        let again = run(&r.phi);
        assert!(again.met);
        for i in 0..n {
            let (a, b) = (r.rain[i].to_f64(), again.rain[i].to_f64());
            assert!(
                (a - b).abs() <= 1e-3 * a.abs() + 1e-12,
                "node {i}: {a} against {b}"
            );
        }
    }
}
