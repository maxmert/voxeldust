//! ★ THE 16 m LEVEL'S SOLVE (rulings B95, B96, B97; the way-in design revision 4 §3.2–§3.4, build step B3).
//!
//! A 16 m TILE is one macro node of a face; its DOMAIN adds a buffer of a quarter node each side (2 km on the home planet,
//! the measured seam rule). It REFINES the drawn 128 m ground — the finished 128 m tiles a host holds
//! ([`FineBook`]) — and never contradicts it:
//!
//! 1. **The 128 m flow.** Over the node and [`FLOW_NODES`] each side, the 128 m points stand at the drawn ground; their
//!    flow tree and water (the rain and the 8 km rivers entering, as the 128 m domain's) give each 128 m link its water,
//!    and the same joint law read backwards gives each node its RATE — `U = K·Q·S^n` over the node's links under the
//!    threshold, the median — the rate at which the drawn ground is in balance (ruling B95: on all land, where the
//!    macro world's present rise is nought on still land and would draw it flat).
//! 2. **The rim.** A 128 m link that crosses the domain's rim is its exit (outward, an outlet on the drawn ground along
//!    the link) or its inflow (inward, its water handed to the point nearest the crossing), so two neighbours' valleys
//!    leave the same way; the sea's points are outlets on their own ground.
//! 3. **The steady form is the 128 m level's own law** (ruling B97, B80's method): each 16 m point over its one steepest
//!    lower neighbour by the joint law's slope `(U/(K·Q))^(1/n)` for its rate and the water the tree brings it, never
//!    steeper than the threshold tan 35° ([`TAN_REPOSE`], Burbank et al. 1996); the drainage built again from each pass's
//!    heights, each point moving half way to the new form from the second pass (the 128 m level's half step, B92); the
//!    passes stop when no 128 m cell's mean moves more than [`STEADY_SETTLE_M`] (the settle law, B39), within the 128 m
//!    level's own bound.
//! 4. **Today's ground plus the law's detail** (ruling B96): the 16 m ground is today's drawn 128 m ground plus the law's
//!    raster less its own 128 m version — the box means over each 128 m cell read back by the same Catmull-Rom — so the law
//!    adds only what lies under 128 m.
//!
//! MEASURED (B97, the bench against Earth's 1 m lidar read at 16 m): the mountain tile beside hull 19 18.4 / 1.61 /
//! 0.046 % of its 16 m links over 35° / 45° / 63° (the San Gabriel Mountains 17.6 / 2.80 / 0.069 %), hull 17's hills
//! 5.47 / 0.096 / 0.0003 % (the Oregon Coast Range 6.6 / 0.065 / 0 %). The form B96 first built — the creep (Roering et
//! al. 2007) at the dry world's slow rate, `S_c` 1.25, channel points pinned under the 128 m channels — sank the law's
//! ground hundreds of metres under today's on steep ground, and the pinned channels stood as walls (7.09 % over 63°).
//!
//! **Example.** The 16 m tile under hull 17 on the home planet's day-side hills: about 560 000 points over 12.3 km, the
//! 128 m flow over 57 km around it, the steady form on the 16 m points, then 640 × 640 heights in quarter metres — the
//! tile and half its buffer, today's ground plus the law's small valleys — the same on any host.

use super::delaunay::NONE;
use super::l1::{Domain, L1_RUNG, U0_LAND, cells_per_node_shift, domain_nodes};
use super::points::CellId;
use super::read::FineBook;
use super::refine::{Coarse, Setup, drawn, median, metres, raster_carved, setup, steady};
use super::tree::{accumulate, tree};
use crate::artifact::Artifact;
use crate::body::{BodyDefinition, TAN_REPOSE};
use crate::gf::Gf;
use crate::macro_lattice::MacroLattice;
use crate::solve::{JOINT_K_LP, JOINT_N};
use vd_seed::bend::Face;

/// The level's rung (the rung-4 cell: 16 m on the home planet).
pub const L2_RUNG: u8 = 4;
/// The buffer each side, as a shift of the node: a quarter node (2 km on the home planet; B0 step 2b measured one node's
/// eighth let a cut run along a valley the halves placed apart).
pub const BUFFER_SHIFT: u32 = 2;
/// The 128 m flow's region: the node and this many nodes each side (the Gaussian read of the rates reaches 2.45 nodes, so
/// every 128 m point the domain reads stands in it whole).
pub const FLOW_NODES: i32 = 3;
/// ★ THE STEADY FORM'S END (ruling B97, the settle law of B92/B93 one level down): the passes stop when no 128 m cell's mean
/// moves more than [`STEADY_SETTLE_M`] metres, within the 128 m level's own bound — a bound that only guarantees an end.
pub const STEADY_BOUND: usize = super::l1::FINE_ROUND_BOUND;
pub const STEADY_SETTLE_M: f64 = 1.0;
/// The Gaussian that reads each 16 m point's rate and rain from the 128 m points around it, in 128 m spacings, and its
/// reach in widths (the 128 m level's own read; [`super::refine`] holds the one value).
pub const READ_SIGMA: f64 = super::refine::READ_SIGMA;
pub const READ_REACH: f64 = super::refine::READ_REACH;

/// A solved 16 m tile: the tile and half its buffer as `side × side` heights in quarter metres over the sea (rows of the
/// face's chart from its north-west corner), and the readings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L2Tile {
    pub side: usize,
    pub quarters: Vec<i32>,
    pub points: usize,
    /// The steady form's passes.
    pub passes: usize,
    /// The law's detail added to today's ground: the median and the largest size, in centimetres.
    pub detail_cm: [i64; 2],
    /// ★ THE WATER MAP under [`super::l1::OutletRule::StandingWater`] (slice W2): each raster cell's water level in quarter
    /// metres over the sea, [`super::water::DRY`] where none (every cell under today's rule).
    pub water: Vec<i32>,
    /// Each raster cell's share of the narrower rivers through it, in 255ths ([`super::carve::Carve::share`]).
    pub river_share: Vec<u8>,
}

/// ★ THE 128 m FLOW around the node `(ni, nj)` of `face`: the domain over the node and [`FLOW_NODES`] each side, its
/// points at the drawn ground, the tree, the water through each point, each point's rate and rain. `None` where a
/// finished 128 m tile it reads is not held.
pub(crate) struct Flow {
    pub d: Domain,
    pub h: Vec<Gf>,
    pub recv: Vec<u32>,
    pub q: Vec<Gf>,
    pub rate: Vec<Gf>,
    pub rain: Vec<Gf>,
    pub pos: Vec<[Gf; 3]>,
}

fn flow(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    book: &FineBook,
    face: Face,
    (ni, nj): (i32, i32),
) -> Option<Flow> {
    let d = domain_nodes(
        body,
        lattice,
        artifact,
        L1_RUNG,
        face,
        (ni - FLOW_NODES, nj - FLOW_NODES),
        2 * FLOW_NODES + 1,
    )?;
    let sea = Gf::from_i32(i32::from(artifact.sea_m));
    let r_m = Gf::from_f64(body.radius_m());
    let mut h = Vec::with_capacity(d.dirs.len());
    for (p, c) in d.cells.iter().enumerate() {
        let z = drawn(book, c, L1_RUNG, sea)?;
        h.push(if d.outlet[p] { z.lesser(d.base[p]) } else { z });
    }
    let t = tree(&h, &d.outlet, &d.graph, &d.geom.len);
    let q = accumulate(&t, &d.water);
    // THE RATE: the joint law read backwards over each node's links under the threshold, the median a node.
    let (k, n, tan) = (
        Gf::from_f64(JOINT_K_LP),
        Gf::from_f64(JOINT_N),
        Gf::from_f64(TAN_REPOSE),
    );
    let mut samples: Vec<Vec<Gf>> = vec![Vec::new(); d.nodes.len()];
    for c in 0..h.len() {
        let r = t.recv[c];
        if d.outlet[c] || r == NONE {
            continue;
        }
        let s = (h[c] - h[r as usize]) / t.link[c];
        if s > Gf::ZERO && s < tan && q[c] > Gf::ZERO {
            samples[d.pnode[c] as usize].push(k * q[c] * s.powf_fine(n));
        }
    }
    let overall =
        median(samples.iter().flatten().copied().collect()).unwrap_or(Gf::from_f64(U0_LAND));
    let node_rate: Vec<Gf> = samples
        .into_iter()
        .map(|v| median(v).unwrap_or(overall))
        .collect();
    // Each read node's rate (a node outside the region the region's own median), then each point's by the Gaussian of
    // the logarithms (the 128 m level's own read).
    let mut ref_rate = vec![overall.ln_fine(); d.refs.len()];
    for (k, &r) in d.node_ref.iter().enumerate() {
        if (r as usize) < ref_rate.len() {
            ref_rate[r as usize] = node_rate[k].ln_fine();
        }
    }
    let mut rate = Vec::with_capacity(h.len());
    let mut rain = Vec::with_capacity(h.len());
    for p in 0..h.len() {
        let mut l = Gf::ZERO;
        for w in d.wstart[p] as usize..d.wstart[p + 1] as usize {
            l += d.wval[w] * ref_rate[d.wnode[w] as usize];
        }
        rate.push(l.exp_fine());
        rain.push(d.water[p] / d.geom.area[p]);
    }
    let pos = d.dirs.iter().map(|x| metres(x, r_m)).collect();
    Some(Flow {
        recv: t.recv,
        d,
        h,
        q,
        rate,
        rain,
        pos,
    })
}

/// ★ THE 16 m DOMAIN of the node `key` under the 128 m flow: the flow over the node and [`FLOW_NODES`] each side, and the
/// 16 m points of the node and its buffer set up by [`super::refine::setup`] (the graph, the drawn 128 m ground, the read,
/// the rim). The 16 m tile solves on it; the 4 m level's 16 m flow record ([`super::l3`]) is built on it too.
pub(crate) struct L2Domain {
    pub f1: Flow,
    pub s: Setup,
    pub n4: i32,
    pub down: u32,
    pub shift4: u32,
    pub buf: i32,
    pub sea: Gf,
}

/// The 16 m domain of `key`; `None` where the rung's cells do not nest in the nodes by a power of two, a finished 128 m
/// tile it reads is not held, or a domain cannot be built.
pub(crate) fn l2_domain(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    book: &FineBook,
    key: super::l1::TileKey,
) -> Option<L2Domain> {
    let (face, ni, nj) = (key.face, key.tx, key.ty);
    let f1 = flow(body, lattice, artifact, book, face, (ni, nj))?;
    let sea = Gf::from_i32(i32::from(artifact.sea_m));
    let r_m = Gf::from_f64(body.radius_m());
    let (n4, n7) = (
        body.ladder().cells_per_edge(L2_RUNG) as i32,
        body.ladder().cells_per_edge(L1_RUNG) as i32,
    );
    let shift4 = cells_per_node_shift(body, lattice, L2_RUNG)?;
    let shift7 = cells_per_node_shift(body, lattice, L1_RUNG)?;
    let down = shift4.checked_sub(shift7)?;
    let node4 = 1i32 << shift4;
    let buf = node4 >> BUFFER_SHIFT;
    let (fi0, fj0) = ((ni - FLOW_NODES) << shift7, (nj - FLOW_NODES) << shift7);
    let fside = (2 * FLOW_NODES + 1) << shift7;
    let coarse = Coarse {
        cells: &f1.d.cells,
        outlet: &f1.d.outlet,
        h: &f1.h,
        recv: &f1.recv,
        q: &f1.q,
        rate: &f1.rate,
        rain: &f1.rain,
        pos: &f1.pos,
        rect: (fi0, fj0, fside),
        n: n7,
        spacing: Gf::from_f64(lattice.node_m() / f64::from(1u32 << shift7)),
    };
    let drawn_at = |c: &CellId| drawn(book, c, L2_RUNG, sea);
    let s = setup(
        body.seed(),
        r_m,
        face,
        L2_RUNG,
        n4,
        down,
        ((ni << shift4) - buf, (nj << shift4) - buf, node4 + 2 * buf),
        &coarse,
        &drawn_at,
    )?;
    Some(L2Domain {
        f1,
        s,
        n4,
        down,
        shift4,
        buf,
        sea,
    })
}

/// ★ THE 16 m TILE `key` — one macro node: the node `(key.tx, key.ty)` of `key.face` — refining the finished 128 m tiles
/// of `book` by [`super::refine`]: the domain under the 128 m flow, the steady form, the raster in B96's form. `None` where
/// the rung's cells do not nest in the nodes by a power of two, a finished 128 m tile it reads is not held, or a domain
/// cannot be built.
#[must_use]
pub fn solve_l2(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    book: &FineBook,
    key: super::l1::TileKey,
) -> Option<L2Tile> {
    solve_l2_with(
        body,
        lattice,
        artifact,
        book,
        key,
        super::l1::OutletRule::Today,
    )
}

/// [`solve_l2`] under the outlet `rule` (slice W2): under [`super::l1::OutletRule::StandingWater`] (the bench until the
/// world move) the settled drainage's channels are carved in the law's raster ([`super::carve`]) and the raster's water
/// map made ([`super::water::raster_water`]) — each 16 m cell's water identity the 128 m water at its place (the book's
/// held water, [`FineBook::water`]), the connection searched again on the 16 m ground, the 16 m channels' banks beside it.
#[must_use]
pub fn solve_l2_with(
    body: &BodyDefinition,
    lattice: &MacroLattice,
    artifact: &Artifact,
    book: &FineBook,
    key: super::l1::TileKey,
    rule: super::l1::OutletRule,
) -> Option<L2Tile> {
    let d = l2_domain(body, lattice, artifact, book, key)?;
    let (h, passes, _) = steady(
        &d.s,
        d.f1.h.len(),
        Gf::from_f64(TAN_REPOSE),
        Gf::from_f64(STEADY_SETTLE_M),
        STEADY_BOUND,
        None,
    );
    // THE RASTER: the tile and half its buffer, cell centres at rung 4.
    let node4 = 1i32 << d.shift4;
    let half = d.buf >> 1;
    let rside = node4 + 2 * half;
    let sea = d.sea;
    let drawn_at = |c: &CellId| drawn(book, c, L2_RUNG, sea);
    let wet_rule = rule != super::l1::OutletRule::Today;
    // THE CARVE (the standing-water rule only): the channels of the settled drainage, read in the law's raster.
    let t = tree(&h, &d.s.outlet, &d.s.graph, &d.s.geom.len);
    let q = accumulate(&t, &d.s.water);
    let cell_m = Gf::from_i64(1i64 << L2_RUNG);
    let ch = super::carve::channels(&t, &d.s.graph, &q, body.facts().gravity_mm_s2, cell_m);
    let r_m = Gf::from_f64(body.radius_m());
    let carve_at = |tri: [u32; 3], dir: &super::delaunay::Dir, z: Gf| {
        super::carve::carve_cell(&ch, &d.s.dirs, &d.s.graph, &t, &h, r_m, tri, dir, z, cell_m)
    };
    let (ri0, rj0) = ((key.tx << d.shift4) - half, (key.ty << d.shift4) - half);
    let (out, detail, banks, shares) = raster_carved(
        &d.s,
        &h,
        key.face,
        d.n4,
        d.down,
        (ri0, rj0, rside),
        &drawn_at,
        wet_rule.then_some(&carve_at as super::refine::CarveAt<'_>),
    )?;
    let quart = |z: Gf| (z * Gf::from_i64(4) + Gf::HALF).floor().to_i64_floor() as i32;
    let mut quarters: Vec<i32> = out.iter().map(|z| z.map_or(i32::MIN, quart)).collect();
    let rs = rside as usize;
    let (water, river_share) = if wet_rule {
        // Each cell's identity: the 128 m water whole at its place.
        let node_level: Vec<Option<i32>> = (0..rs)
            .flat_map(|b| (0..rs).map(move |a| (a, b)))
            .map(|(a, b)| {
                let c = super::points::chart_cell(key.face, ri0 + a as i32, rj0 + b as i32, d.n4)?;
                let f = Face::from_index(c.face)?;
                match book.water(f, L1_RUNG, L2_RUNG, c.i, c.j) {
                    super::read::FineWater::Held { level, wet } if wet == super::tile::WET_FULL => {
                        Some(level)
                    }
                    _ => None,
                }
            })
            .collect();
        // Each cell's REACH (W16, ruling B115): the 128 m water at its place by the column rule — a 16 m spread enters
        // only where the 128 m level holds the same water, so the finer shore refines the coarser one and never runs past it.
        let reach: Vec<i32> = (0..rs)
            .flat_map(|b| (0..rs).map(move |a| (a, b)))
            .map(|(a, b)| {
                super::points::chart_cell(key.face, ri0 + a as i32, rj0 + b as i32, d.n4)
                    .and_then(|c| {
                        let f = Face::from_index(c.face)?;
                        match book.shore_water(f, L1_RUNG, L2_RUNG, c.i, c.j) {
                            super::read::FineWater::Held { level, .. } => Some(level),
                            _ => None,
                        }
                    })
                    .unwrap_or(super::water::DRY)
            })
            .collect();
        // ★ THE COARSER GROUND KEEPS ITS SIDE OF THE WATER (rulings W6 and W16; the owner at the old lake's deltas,
        // 2026-10-20: *"Shores are not believable"* — walls of water along 128 m cell edges, rectangles and blue specks on
        // the delta plains): where the 128 m drawn ground at a cell stands at or over the water beside it (the column rule
        // names a level), the law's finer detail never carries the cell under that water's level — it is held at that
        // coarser ground, so the 16 m shore follows the 128 m ground's own crossing and never a sample's square. MEASURED before: a delta's top stands at its lake's level, so the
        // detail's dimples put half its cells under the water, the spread wet them up to the column rule's band and
        // stopped there beside dry ground under the same level.
        for b in 0..rs {
            for a in 0..rs {
                let k = b * rs + a;
                if quarters[k] == i32::MIN || reach[k] == super::water::DRY || quarters[k] >= reach[k] {
                    continue;
                }
                // The 128 m drawn ground at the cell: where it stands at or over that water, the finer detail may not
                // carry the cell under it (its own crossing of the level, never the 128 m samples' blocks).
                // Held at the coarser ground itself (never at the level, where a centimetre decides wet or dry).
                let coarse = super::points::chart_cell(key.face, ri0 + a as i32, rj0 + b as i32, d.n4)
                    .and_then(|c| drawn_at(&c))
                    .map(quart)
                    .filter(|&z| z >= reach[k]);
                if let Some(z) = coarse {
                    quarters[k] = z;
                }
            }
        }
        let river: Vec<i32> = banks
            .iter()
            .map(|b| b.map_or(super::water::DRY, quart))
            .collect();
        let share: Vec<u8> = shares
            .iter()
            .map(|&g| (g.lesser(Gf::ONE) * Gf::from_i64(255) + Gf::HALF).to_i64_floor() as u8)
            .collect();
        (
            super::water::raster_water_within(&quarters, rs, &node_level, &river, Some(&reach)),
            share,
        )
    } else {
        (vec![super::water::DRY; rs * rs], vec![0u8; rs * rs])
    };
    let cm = |g: Gf| (g * Gf::from_i64(100)).floor().to_i64_floor();
    Some(L2Tile {
        side: rside as usize,
        water,
        river_share,
        quarters,
        points: d.s.dirs.len(),
        passes,
        detail_cm: [
            detail.get(detail.len() >> 1).map_or(0, |&g| cm(g)),
            detail.last().map_or(0, |&g| cm(g)),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fine::l1::{TILE_NODES, TileKey, solve_l1};
    use crate::fine::level::{FineLevel, LevelGeom};
    use crate::fine::stitch::{Field, mean_levels};
    use crate::home::{HOME_SYSTEM_AGE_YR, home_moon, home_solve_words};
    use crate::solve::{Schedule, SolveWords};

    /// ★ ONE 16 m TILE OF THE WET MOON, END TO END (rulings B96, B97). The 16 m node at the corner where four 128 m tiles meet
    /// — (NegX, 6 … 7, 0 … 1), beside the tile the 128 m test's rule picks — whose flow (three nodes each way, and the
    /// 128 m curve's stencil a sample or two past them) reads all four; each 128 m tile held as its own raw square and
    /// that square's mean levels (the finish's stitch is the finish's own tests'; the shard holds the 3 × 3 finished tiles
    /// around a 16 m tile for the same reach). With none held the 16 m tile does not solve; with them it solves: the
    /// steady form within its passes, every raster cell given a height, and EVERY cell today's drawn 128 m ground plus at
    /// most the law's largest detail (and the raster's rounding) — the 16 m level refines today's ground and never departs
    /// from it by more than its own detail.
    #[test]
    fn the_wet_moon_16m_tile_solves_on_its_128m_ground() {
        let moon = home_moon();
        let words = SolveWords {
            water_km3: 3_000_000,
            ..home_solve_words()
        };
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let (state, facies, _) =
            crate::test_solves::solved(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR));
        let climate =
            crate::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_view()));
        let artifact = Artifact::of(&state, &facies, &climate, true);
        let node = TileKey {
            face: Face::NegX,
            tx: 7 * TILE_NODES,
            ty: TILE_NODES,
        };
        let mut book =
            FineBook::new(FineLevel::L1, &moon, &lattice, artifact.sea_m).expect("a 128 m book");
        assert_eq!(
            solve_l2(&moon, &lattice, &artifact, &book, node).map(|t| t.side),
            None
        );
        // The four 128 m tiles around the node's corner, solved side by side, each held as its own square.
        let geom = LevelGeom::of(FineLevel::L1, &moon, &lattice).expect("the 128 m level");
        let keys: Vec<TileKey> = [(6, 0), (7, 0), (6, 1), (7, 1)]
            .into_iter()
            .map(|(tx, ty)| TileKey {
                face: node.face,
                tx,
                ty,
            })
            .collect();
        let squares: Vec<(TileKey, Vec<Vec<i32>>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .iter()
                .map(|&k| {
                    let (moon, lattice, artifact) = (&moon, &lattice, &artifact);
                    scope.spawn(move || {
                        let t = solve_l1(moon, lattice, artifact, k).expect("a 128 m tile solves");
                        let (w, m, n) = (t.side, geom.margin as usize, geom.side() as usize);
                        let v: Vec<i32> = (0..n)
                            .flat_map(|y| t.quarters[(m + y) * w + m..(m + y) * w + m + n].to_vec())
                            .collect();
                        let square = Field {
                            gx0: i64::from(k.tx) * geom.side(),
                            gy0: i64::from(k.ty) * geom.side(),
                            w: n,
                            h: n,
                            v,
                        };
                        let mut levels = vec![square.v.clone()];
                        levels.extend(
                            mean_levels(
                                &square,
                                square.gx0,
                                square.gy0,
                                n,
                                FineLevel::L1.mean_levels(),
                            )
                            .expect("the mean levels"),
                        );
                        (k, levels)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a 128 m solve"))
                .collect()
        });
        for (k, levels) in squares {
            assert!(book.insert(k, levels));
        }
        let tile = solve_l2(&moon, &lattice, &artifact, &book, node).expect("the 16 m tile solves");
        // The tile (one node) and half its buffer each side, every cell given a height.
        let shift4 = cells_per_node_shift(&moon, &lattice, L2_RUNG).expect("nests") as i32;
        let half = (1i32 << shift4) >> (BUFFER_SHIFT + 1);
        let side = (1usize << shift4) + 2 * half as usize;
        assert_eq!((tile.side, tile.quarters.len()), (side, side * side));
        assert_eq!(tile.quarters.iter().filter(|&&q| q == i32::MIN).count(), 0);
        assert!((1..=STEADY_BOUND).contains(&tile.passes), "{}", tile.passes);
        assert!(tile.detail_cm[1] > 0, "{:?}", tile.detail_cm);
        // Every cell today's drawn ground plus at most the largest detail (its centimetre's floor, and the rounding).
        let sea = Gf::from_i32(i32::from(artifact.sea_m));
        let bound = Gf::from_i64((tile.detail_cm[1] + 1) * 4) / Gf::from_i64(100) + Gf::HALF;
        let (i0, j0) = ((node.tx << shift4) - half, (node.ty << shift4) - half);
        let mut beyond = 0usize;
        for b in 0..side {
            for a in 0..side {
                let cell = CellId {
                    face: node.face.index(),
                    i: i0 + a as i32,
                    j: j0 + b as i32,
                };
                let curve = drawn(&book, &cell, L2_RUNG, sea).expect("today's ground is held");
                let q = Gf::from_i64(i64::from(tile.quarters[b * side + a]));
                beyond += usize::from((q - curve * Gf::from_i64(4)).abs() > bound);
            }
        }
        assert_eq!(beyond, 0);
    }
}
