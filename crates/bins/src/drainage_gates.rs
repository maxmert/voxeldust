//! ★ THE DRAINAGE GATES — ONE COPY of G-HACK, G-BREACH, THE SPILL CENSUS and THE THREE STANDS, for
//! every instrument that reads a grid of ground heights (plan step 1 of
//! `docs/investigation/2026-09-24/terrain_believability_status.md`; ruling B6, 2026-09-24).
//!
//! **WHY THIS MODULE EXISTS.** The R1 bench (`examples/erosion_bench.rs`) measured G-HACK and
//! G-BREACH on one tile of the bicubic upsample. The shipped-ground instrument
//! (`examples/shipped_drainage.rs`) must read THE SAME two gates on the ground the chunk builder
//! makes. Two copies of one gate can drift apart, and then two readings of "the same" gate disagree
//! for a reason nobody can see. So each gate lives here ONCE, and every instrument calls it. The
//! same holds for the three stands (`examples/drain_census.rs` picked them first): an instrument
//! that picks "the coast" by another rule measures another coast.
//!
//! **WHAT A GATE READS.** A [`Heights`] grid: `w × w` heights in ANY fixed-point unit, how many units
//! make one metre, the width of one cell on the ground, and an optional mask of the DRY cells.
//! Without a mask every cell is ground, which is the bench's own case and gives the bench's own
//! numbers byte for byte. With a mask, a WET cell (under the sea or under a lake) is an OUTLET:
//! water that reaches it has drained, and no gate counts a hollow of the sea floor as a hollow of
//! the land.
//!
//! **Example.** A pilot flies over the belt at 120 km. The shipped-ground instrument samples the
//! ground the chunk builder makes at rung 7, one column every 128 m over a square 200 km a side, and
//! hands the heights here. The spill census floods that square from its rim and from the coast. Each
//! hollow the flood fills before the water spills is one CLOSED BASIN: a dune trough the fine octaves
//! dug, or a lake bed the solve left dry.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vd_core::glam::DVec3;
use vd_recipe::amplify::NEIGHBOURS;
use vd_seed::bend::Face;
use vd_terrain::artifact::Artifact;
use vd_terrain::macro_lattice::{MacroLattice, STENCIL};
use vd_terrain::solve::{FACIES_LAKE, FACIES_SEA};

/// The stencil's eight offsets, in `MacroLattice::STENCIL` order — the same order the erosion
/// operators index their slots by (`vd_recipe::amplify`), so a slot means one thing everywhere.
pub const OFF: [(i32, i32); NEIGHBOURS] = STENCIL;

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE THREE STANDS (moved here from `drain_census`, unchanged).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// One stand: its name and the unit direction under it.
pub struct Stand {
    pub name: &'static str,
    pub dir: DVec3,
}

/// The `(face, i, j)` cell of a direction at a rung.
#[must_use]
pub fn cell_of(body: &vd_terrain::BodyDefinition, dir: DVec3, rung: u8) -> (Face, i32, i32) {
    let d = [dir.x, dir.y, dir.z];
    let face = vd_seed::bend::face_of(d);
    let (t, s) = vd_seed::bend::face_coords(face, d);
    let n = body.ladder().cells_per_edge(rung);
    (
        face,
        vd_seed::ladder::index_of(vd_seed::bend::unbend(t), n),
        vd_seed::ladder::index_of(vd_seed::bend::unbend(s), n),
    )
}

/// ★ THE THREE STANDS (ruling B2): the belt the owner flew, a COAST and a PLAIN, both picked off
/// the census's own map rather than typed in — the coast is a dry node beside a sea node with the
/// greatest discharge near it, and the plain is the land node of the flattest neighbourhood far
/// from any water.
#[must_use]
pub fn pick_stands(artifact: &Artifact, lattice: &MacroLattice) -> Vec<Stand> {
    let n = artifact.node_count();
    let facies: Vec<u8> = artifact
        .rows
        .iter()
        .map(|r| r.receiver_facies >> vd_terrain::artifact::FACIES_SHIFT)
        .collect();
    let mut coast = (0u32, 0u8);
    let mut plain = (0u32, i32::MAX);
    for node in 0..n as u32 {
        let f = facies[node as usize];
        if (f & FACIES_SEA != 0) | (f & FACIES_LAKE != 0) {
            continue;
        }
        let here = artifact.rows[node as usize].z_m;
        let neigh = lattice.neighbours(node);
        let wet = neigh
            .iter()
            .any(|&m| m != u32::MAX && facies[m as usize] & FACIES_SEA != 0);
        if wet {
            let q = artifact.rows[node as usize].discharge_log;
            if q > coast.1 {
                coast = (node, q);
            }
        } else {
            let mut spread = 0i32;
            let mut dry = true;
            for &m in &neigh {
                if m == u32::MAX {
                    dry = false;
                    break;
                }
                let f = facies[m as usize];
                if (f & FACIES_SEA != 0) | (f & FACIES_LAKE != 0) {
                    dry = false;
                    break;
                }
                spread += (i32::from(artifact.rows[m as usize].z_m) - i32::from(here)).abs();
            }
            // A plain is flat AND standing well over the sea, so it is ground and not a shelf.
            if dry && here > 200 && spread < plain.1 {
                plain = (node, spread);
            }
        }
    }
    let of = |node: u32| -> DVec3 {
        let d = lattice.direction(node);
        DVec3::new(d[0].raw() as f64, d[1].raw() as f64, d[2].raw() as f64).normalize()
    };
    println!(
        "the stands: the belt, the coast at node {} (discharge class {}), the plain at node {} (its ring spreads {} steps)",
        coast.0, coast.1, plain.0, plain.1
    );
    vec![
        Stand {
            name: "the belt",
            dir: DVec3::new(0.617_270, -0.437_286, -0.654_033).normalize(),
        },
        Stand {
            name: "the coast",
            dir: of(coast.0),
        },
        Stand {
            name: "a plain",
            dir: of(plain.0),
        },
    ]
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE GRID EVERY GATE READS.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ A SQUARE GRID OF GROUND HEIGHTS, as every gate below reads it.
///
/// The heights are plain words in ONE fixed-point unit, and [`Heights::per_m`] says how many of
/// them make a metre. The R1 bench hands its grid at `H_BITS` (`1 << 16` a metre); the
/// shipped-ground instrument hands the chunk builder's own surface radius less the ladder radius, in
/// gap steps at the length format (`128 << 28` a metre). A gate never names either unit: an order
/// reads the words as they stand, and a volume divides by `per_m` once.
#[derive(Clone, Copy)]
pub struct Heights<'a> {
    /// The grid's width: `w × w` cells, row-major, cell `(x, y)` at `y · w + x`.
    pub w: usize,
    /// One cell's width on the ground, in metres.
    pub cell_m: f64,
    /// The heights, `w × w` words.
    pub h: &'a [i64],
    /// How many units of [`Heights::h`] make one metre.
    pub per_m: f64,
    /// ★ THE DRY CELLS, or `None` where every cell is ground (the bench's own case). A wet cell is
    /// an OUTLET: the floods below start there as they start on the rim, no pit is counted on it,
    /// and G-HACK fits no wet cell.
    pub dry: Option<&'a [bool]>,
}

impl Heights<'_> {
    /// The cells of the grid.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.w * self.w
    }

    /// Whether cell `i` is ground a gate counts: always, without a mask.
    #[must_use]
    pub fn dry_at(&self, i: usize) -> bool {
        self.dry.is_none_or(|m| m[i])
    }
}

/// One cell's eight neighbour indices in [`OFF`] order, and the mask of the ones inside the grid.
#[inline]
#[must_use]
pub fn stencil(w: usize, x: usize, y: usize) -> ([usize; NEIGHBOURS], u32) {
    let mut idx = [y * w + x; NEIGHBOURS];
    let mut present = 0u32;
    for (slot, (dx, dy)) in OFF.iter().enumerate() {
        let nx = x as i64 + i64::from(*dx);
        let ny = y as i64 + i64::from(*dy);
        if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < w {
            idx[slot] = ny as usize * w + nx as usize;
            present |= 1 << slot;
        }
    }
    (idx, present)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// G-HACK (moved here from `erosion_bench`, unchanged on a grid with no mask).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ G-HACK — the Hack's-law fit `L = c·a^n` (Sassolas-Serrayet, Cattin, Ferry & Godard 2018;
/// Schott et al. 2023 §7.3 validate a terrain exactly this way). The published bands are
/// `c ∈ [1, 6]` and `n ∈ [0.45, 0.7]`, with `L` in km and `a` in km².
///
/// The network is drawn with the STEEPEST-descent receiver (the operators' own tie rule, the lowest
/// stencil slot), the drainage area `a` accumulated down it in km², and `L` the LONGEST upstream
/// flow path in km. Both are accumulated in one pass over the cells sorted by height, highest
/// first, which is the topological order of a descent tree. The fit is an ordinary least squares on
/// the logarithms over the cells whose basin passes ten cells — below that a "river" is a pixel.
/// With a dry mask, a wet cell keeps its receiver (a lake's outflow still feeds the river below
/// it) but the fit reads no wet cell.
///
/// Returns `(c, n, the network cells the fit read)`; `(0, 0, count)` where fewer than two cells
/// qualify.
#[must_use]
pub fn g_hack(f: &Heights) -> (f64, f64, usize) {
    let cells = f.cells();
    let cell_km2 = f.cell_m * f.cell_m / 1.0e6;
    let mut recv = vec![usize::MAX; cells];
    let mut dist_km = vec![0.0f64; cells];
    for y in 0..f.w {
        for x in 0..f.w {
            let i = y * f.w + x;
            let (idx, present) = stencil(f.w, x, y);
            let mut best = 0i64;
            for slot in 0..NEIGHBOURS {
                if present >> slot & 1 == 0 {
                    continue;
                }
                let drop = f.h[i] - f.h[idx[slot]];
                if drop > best {
                    best = drop;
                    recv[i] = idx[slot];
                    let diagonal = OFF[slot].0 != 0 && OFF[slot].1 != 0;
                    dist_km[i] = f.cell_m
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
    order.sort_unstable_by(|a, b| f.h[*b as usize].cmp(&f.h[*a as usize]));
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
        if area[i] < floor || length[i] <= 0.0 || !f.dry_at(i) {
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

/// A natural logarithm written out, because the recipe's crates are fenced against `f64::ln` and
/// the numbers these gates report keep the same discipline: the argument is split into a power of
/// two and a mantissa, and the mantissa's logarithm comes from the `atanh` series, which converges
/// on `[1, 2)` in a dozen terms.
#[must_use]
pub fn ln(x: f64) -> f64 {
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
#[must_use]
pub fn exp(x: f64) -> f64 {
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

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// G-BREACH (moved here from `erosion_bench`, unchanged on a grid with no mask).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// ★ G-BREACH — a breaching volume, on THE DEFINITION THIS FILE STATES, because Schott et al. 2024
/// publish the metric's VALUES (their erosion 0.02–4.8 ×10³, procedural noise 1.7–41) but not the
/// code that computes them, and we could not run theirs.
///
/// Ours: a priority flood from the grid's rim (Barnes, Lehman & Mulla 2014) records, for every
/// cell, the neighbour it was reached from — so every cell holds a path to the rim whose highest
/// point is the lowest such highest point of any path. A cell is a PIT where every neighbour stands
/// at or above it. For each pit we walk that recorded path outward and add `max(0, h − h_pit)`
/// times the cell area: the material a ONE-CELL-WIDE channel at the pit's own level must still
/// remove for the pit to drain. The answer is the mean over the pits, in cubic metres. With a dry
/// mask the flood also starts at every wet cell, and no wet cell is a pit.
///
/// **What this can and cannot say.** The ABSOLUTE number cannot be set beside the paper's, because
/// their unit is not published — that comparison is UNVERIFIED. The ratio of two fields read by the
/// same definition (the bench's before and after; the shipped ground and the solve alone) is a
/// measurement that can fail, and it is the one reported. The volume scales with the CELL's width
/// (the channel is one cell wide), so two grids compare only at one spacing.
///
/// ★ **ONE CHANGE OF FORM, NO CHANGE OF NUMBER.** The bench wrote each step's height as
/// `f64::from(u32::try_from(over).unwrap_or(u32::MAX))`, which clamps at `u32::MAX` units. At the
/// bench's `H_BITS` that clamp is 65 535 m and a planet's whole relief is under ten kilometres, so
/// it never bound, and `over as f64` gives the same float. The shipped unit is `2^35` a metre, where
/// the same clamp would bind at an eighth of a metre, so the clamp is gone.
///
/// Returns `(the mean volume in m³, the pits)`; `(0, 0)` where there is no pit.
#[must_use]
pub fn g_breach(f: &Heights) -> (f64, usize) {
    let cells = f.cells();
    let area = f.cell_m * f.cell_m;
    let mut done = vec![false; cells];
    let mut from = vec![usize::MAX; cells];
    // The priority flood: the rim (and every outlet) first, lowest first, each cell carrying the
    // highest point of the path that reached it.
    let mut heap: BinaryHeap<(Reverse<i64>, usize)> = BinaryHeap::new();
    for y in 0..f.w {
        for x in 0..f.w {
            let i = y * f.w + x;
            if x == 0 || y == 0 || x == f.w - 1 || y == f.w - 1 || !f.dry_at(i) {
                done[i] = true;
                heap.push((Reverse(f.h[i]), i));
            }
        }
    }
    while let Some((Reverse(spill), i)) = heap.pop() {
        let (x, y) = (i % f.w, i / f.w);
        let (idx, present) = stencil(f.w, x, y);
        for (slot, j) in idx.iter().copied().enumerate() {
            if present >> slot & 1 == 0 || done[j] {
                continue;
            }
            done[j] = true;
            from[j] = i;
            heap.push((Reverse(f.h[j].max(spill)), j));
        }
    }
    // The pits, and the channel each needs.
    let mut total = 0.0f64;
    let mut pits = 0usize;
    for y in 1..f.w - 1 {
        for x in 1..f.w - 1 {
            let i = y * f.w + x;
            if !f.dry_at(i) {
                continue;
            }
            let (idx, present) = stencil(f.w, x, y);
            let lowest = (0..NEIGHBOURS)
                .filter(|slot| present >> slot & 1 == 1)
                .all(|slot| f.h[idx[slot]] >= f.h[i]);
            if !lowest {
                continue;
            }
            pits += 1;
            let pit = f.h[i];
            let mut c = from[i];
            let mut volume = 0.0f64;
            while c != usize::MAX {
                let over = f.h[c] - pit;
                if over <= 0 {
                    break;
                }
                volume += over as f64 / f.per_m * area;
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

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE SPILL CENSUS (plan step 1: the number `drain_census` cannot see).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The depth classes of the spill census's histogram, in metres: under 0.1, 0.1–1, 1–10, 10–100,
/// 100–1 000, and 1 000 or deeper.
pub const DEPTH_EDGES_M: [f64; 5] = [0.1, 1.0, 10.0, 100.0, 1_000.0];
/// The histogram's classes: one under the first edge, and one at or over each edge.
pub const DEPTH_CLASSES: usize = DEPTH_EDGES_M.len() + 1;

/// One closed basin: the cells a flood from the outlets fills before its water spills.
#[derive(Clone, Copy, Debug)]
pub struct Basin {
    /// The basin's lowest cell, as a grid index (`y · w + x`).
    pub pit: usize,
    /// ★ THE SPILL DEPTH: the spill level minus the pit, in metres.
    pub depth_m: f64,
    /// How many cells the fill covers.
    pub cells: usize,
    /// The water the fill holds, in cubic metres.
    pub volume_m3: f64,
}

/// What the spill census found on one grid.
#[derive(Clone, Debug, Default)]
pub struct Spill {
    /// Every closed basin, in the order the census met them (row-major by their first cell).
    pub basins: Vec<Basin>,
    /// The cells some basin fills.
    pub flooded: usize,
    /// The dry cells of the grid (all of them, without a mask).
    pub dry: usize,
    /// ★ THE CENSUS'S OWN CHECK: basins whose cells disagree on the fill level. An eight-connected
    /// flood gives every cell of one connected fill the same level, so this must read ZERO; any
    /// other number is a defect of this instrument, not of the ground.
    pub split: usize,
}

impl Spill {
    /// How many basins stand at least `m` metres deep.
    #[must_use]
    pub fn at_least(&self, m: f64) -> usize {
        self.basins.iter().filter(|b| b.depth_m >= m).count()
    }

    /// How many cells the fills of the basins at least `m` metres deep cover. [`Spill::flooded`]
    /// counts every basin, and a basin one word deep (3 × 10⁻¹¹ m at the length format) is the
    /// arithmetic's own step, not a hollow; this count holds the same floor as [`Spill::at_least`].
    #[must_use]
    pub fn flooded_at_least(&self, m: f64) -> usize {
        self.basins
            .iter()
            .filter(|b| b.depth_m >= m)
            .map(|b| b.cells)
            .sum()
    }

    /// The basins by depth class ([`DEPTH_EDGES_M`]).
    #[must_use]
    pub fn histogram(&self) -> [usize; DEPTH_CLASSES] {
        let mut out = [0usize; DEPTH_CLASSES];
        for b in &self.basins {
            let class = DEPTH_EDGES_M.iter().filter(|&&e| b.depth_m >= e).count();
            out[class] += 1;
        }
        out
    }
}

/// ★ THE SPILL CENSUS — every closed basin of a grid, by its SPILL DEPTH.
///
/// **The law.** A priority flood (Barnes, Lehman & Mulla 2014, "Priority-Flood") starts at every
/// OUTLET — the grid's rim and, with a dry mask, every wet cell — and raises each cell it reaches to
/// the higher of its own height and the level of the cell it came from. The result is the FILLED
/// surface: the lowest level water can stand at and still leave the grid. Where the filled surface
/// stands over the ground, the ground holds water it cannot drain. One EIGHT-CONNECTED patch of
/// such cells is ONE CLOSED BASIN; its depth is the fill level minus its lowest cell, its area is
/// its cells, its volume is the sum of the fill over the ground.
///
/// **Why a depth and not a count of low columns.** A count of columns lower than their eight
/// neighbours reads the slope across ONE column spacing, so its floor shrinks with the spacing: at
/// 8 m columns the floor of a 3 125 m ripple 199 m tall rises 0.026 m a column, and a 0.1 m floor
/// reads it as drained ground (status §4). A spill depth reads the whole ring around the hollow,
/// whatever the spacing: the same ripple reads about 200 m deep at 8 m columns and at 128 m columns.
///
/// **What it cannot see.** A basin the rim cuts drains through the rim and is not counted; a basin
/// wider than the grid is not a basin here. Both only LOWER the count.
///
/// **Example.** A pilot at 120 km over a dune sea: every trough of the 3 125 m crest is ringed by
/// crests, the flood fills it to its lowest saddle, and the census counts one basin per trough, each
/// tens to hundreds of metres deep. Over a drained valley system the flood reaches every column by a
/// downhill path, fills nothing, and the census reads zero.
#[must_use]
pub fn spill_census(f: &Heights) -> Spill {
    let cells = f.cells();
    let mut fill = vec![0i64; cells];
    let mut done = vec![false; cells];
    let mut heap: BinaryHeap<(Reverse<i64>, usize)> = BinaryHeap::new();
    for y in 0..f.w {
        for x in 0..f.w {
            let i = y * f.w + x;
            if x == 0 || y == 0 || x == f.w - 1 || y == f.w - 1 || !f.dry_at(i) {
                done[i] = true;
                fill[i] = f.h[i];
                heap.push((Reverse(f.h[i]), i));
            }
        }
    }
    while let Some((Reverse(level), i)) = heap.pop() {
        let (idx, present) = stencil(f.w, i % f.w, i / f.w);
        for (slot, j) in idx.iter().copied().enumerate() {
            if present >> slot & 1 == 0 || done[j] {
                continue;
            }
            done[j] = true;
            let v = f.h[j].max(level);
            fill[j] = v;
            heap.push((Reverse(v), j));
        }
    }
    let area = f.cell_m * f.cell_m;
    let mut out = Spill {
        dry: (0..cells).filter(|&i| f.dry_at(i)).count(),
        ..Spill::default()
    };
    let mut seen = vec![false; cells];
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..cells {
        if seen[start] || fill[start] <= f.h[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let spill = fill[start];
        let mut pit = start;
        let mut count = 0usize;
        let mut units = 0.0f64;
        let mut split = false;
        while let Some(i) = stack.pop() {
            count += 1;
            units += (fill[i] - f.h[i]) as f64;
            split |= fill[i] != spill;
            if f.h[i] < f.h[pit] {
                pit = i;
            }
            let (idx, present) = stencil(f.w, i % f.w, i / f.w);
            for (slot, j) in idx.iter().copied().enumerate() {
                if present >> slot & 1 == 0 || seen[j] || fill[j] <= f.h[j] {
                    continue;
                }
                seen[j] = true;
                stack.push(j);
            }
        }
        out.flooded += count;
        out.split += usize::from(split);
        out.basins.push(Basin {
            pit,
            depth_m: (spill - f.h[pit]) as f64 / f.per_m,
            cells: count,
            volume_m3: units / f.per_m * area,
        });
    }
    out
}
