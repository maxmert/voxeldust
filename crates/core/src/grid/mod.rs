//! ★ THE GEOMETRY SEAM — one value per realm behind which a planet's bent grid and a hull's flat
//! grid look the same to everything above it (HR4; ruling V6 A1/A2).
//!
//! Everything above this seam — the chunk container, the mesher, the smooth extractor, the edit
//! path, the collider derivation, the diff codec, the pyramid — is written ONCE and never sees which
//! arm it runs on. This module is the only place geometry is matched on (six arms here, one in the
//! sim's derivation of the value), and every match is on a VALUE the realm carries, never on a shard
//! kind (HR3). The module-dependency fence that makes "only here" structural is owed with the
//! mesher, the first consumer.
//!
//! Six operations: the address of a position, the centre of a cell, its eight corners, its neighbour
//! in one of six directions, the grid's outward direction, and the domain. On the flat arm every
//! operation is an integer shift. On the round arm the address runs the inverse face bend and one
//! floor per axis, and the centre runs the forward bend and one normalisation.
//!
//! **Gravity is NOT a grid operation.** `outward` is the grid's radial for terrain: it exists on the
//! round arm only. Gravity is a function the realm states — toward a centre for a moon, away from an
//! axis for a cylinder, nothing for a station, and later a field from a graviblock (ruling V5).
//!
//! **Example.** A miner digs the same trench with the same tool on a moon and in a station's soil
//! bay. One fixture, two arms, two stores compared byte for byte. That is HR4's gate.

pub mod addr;
pub mod bend;
pub mod identity;
pub mod seam;
pub mod shell;

pub use addr::{CHUNK_EDGE, CellAddr, CellIndex, ChunkCoord, Rung};
pub use bend::Face;
pub use identity::{IdentityGrid, MAX_HALF_CELLS, SubScale, sub_site};
pub use seam::{Edge, SeamRecord};
pub use shell::{BandParams, ShellGrid};

use crate::pose::{LatticePos, RealmId, Tier};

/// A position's metres from the realm's own origin at the fine tier — the one flatten the grid
/// uses. Every frame at or below a star system is a fine-tier frame.
#[must_use]
pub fn metres_of(pos: LatticePos) -> glam::DVec3 {
    pos.delta_m(LatticePos::ORIGIN, Tier::Fine)
}

/// One of the six directions a cell has a neighbour in: along the face's `u` axis (`i`), its `v`
/// axis (`j`), and the radial or third axis (`k`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dir6 {
    IPlus,
    IMinus,
    JPlus,
    JMinus,
    KPlus,
    KMinus,
}

impl Dir6 {
    /// Every direction.
    pub const ALL: [Dir6; 6] = [
        Dir6::IPlus,
        Dir6::IMinus,
        Dir6::JPlus,
        Dir6::JMinus,
        Dir6::KPlus,
        Dir6::KMinus,
    ];
}

/// An arm's answer to a neighbour query, in that arm's own indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// The next cell on the same face.
    Same(i32, i32, i32),
    /// The partner cell across a face edge; `axis_swap` says the along-edge index changed axis.
    Across {
        face: Face,
        i: i32,
        j: i32,
        k: i32,
        axis_swap: bool,
    },
    /// No cell: past the box, below the floor, at the ceiling, or at a rung the body lacks.
    Outside,
}

/// The seam's answer to a neighbour query, as a full address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Neighbor {
    /// The next cell on the same face.
    Same(CellAddr),
    /// The partner cell across a face edge; `axis_swap` says the along-edge index changed axis.
    AcrossSeam { addr: CellAddr, axis_swap: bool },
    /// No cell there.
    Outside,
}

/// What a grid covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridDomain {
    /// A flat grid's box, in whole cells at rung 0.
    Box { half_cells: [i32; 3] },
    /// A round grid's band: cells per face edge, the floor radius, the band thickness and the rungs.
    Band {
        cells_per_edge: u32,
        floor_m: u32,
        band_m: u32,
        rungs: u8,
    },
}

/// THE SEAM VALUE: a realm's grid, one of two arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridMapping {
    /// A hull, a station, a small or irregular asteroid.
    Identity(IdentityGrid),
    /// A planet, a moon, a large round asteroid.
    CubeSphere(ShellGrid),
}

impl GridMapping {
    /// The cell of `body`'s grid holding `pos` (in the realm's own frame) at `rung`, or `None`.
    #[must_use]
    pub fn addr_of(&self, body: RealmId, pos: LatticePos, rung: Rung) -> Option<CellAddr> {
        match self {
            GridMapping::Identity(g) => g.addr_of(pos, rung).map(|(i, j, k)| CellAddr {
                body,
                face: Face::PosX,
                rung,
                i,
                j,
                k,
            }),
            GridMapping::CubeSphere(g) => g.addr_of(pos, rung).map(|(face, i, j, k)| CellAddr {
                body,
                face,
                rung,
                i,
                j,
                k,
            }),
        }
    }

    /// The centre of `addr`; `None` for a cell the grid does not have (a rung the body lacks, an index
    /// off the face or outside the band). A flat grid names a centre for every address.
    #[must_use]
    pub fn cell_center(&self, addr: &CellAddr) -> Option<LatticePos> {
        match self {
            GridMapping::Identity(_) => {
                Some(IdentityGrid::cell_center(addr.i, addr.j, addr.k, addr.rung))
            }
            GridMapping::CubeSphere(g) => {
                g.cell_center(addr.face, addr.i, addr.j, addr.k, addr.rung)
            }
        }
    }

    /// The eight corners of `addr`, low corner first, `i` fastest; `None` as for [`Self::cell_center`].
    #[must_use]
    pub fn cell_corners(&self, addr: &CellAddr) -> Option<[LatticePos; 8]> {
        match self {
            GridMapping::Identity(_) => Some(IdentityGrid::cell_corners(
                addr.i, addr.j, addr.k, addr.rung,
            )),
            GridMapping::CubeSphere(g) => {
                g.cell_corners(addr.face, addr.i, addr.j, addr.k, addr.rung)
            }
        }
    }

    /// The neighbour of `addr` in `dir`.
    #[must_use]
    pub fn neighbor(&self, addr: &CellAddr, dir: Dir6) -> Neighbor {
        let step = match self {
            GridMapping::Identity(g) => g.neighbor(addr.rung, addr.i, addr.j, addr.k, dir),
            GridMapping::CubeSphere(g) => {
                g.neighbor(addr.rung, addr.face, addr.i, addr.j, addr.k, dir)
            }
        };
        match step {
            Step::Same(i, j, k) => Neighbor::Same(CellAddr { i, j, k, ..*addr }),
            Step::Across {
                face,
                i,
                j,
                k,
                axis_swap,
            } => Neighbor::AcrossSeam {
                addr: CellAddr {
                    face,
                    i,
                    j,
                    k,
                    ..*addr
                },
                axis_swap,
            },
            Step::Outside => Neighbor::Outside,
        }
    }

    /// The grid's outward unit direction at `addr`: the radial on a round grid; `None` on a flat
    /// grid, which has no outward direction (gravity is the realm's function, not the grid's), and
    /// `None` for a cell a round grid does not have.
    #[must_use]
    pub fn outward(&self, addr: &CellAddr) -> Option<[f64; 3]> {
        match self {
            GridMapping::Identity(_) => None,
            GridMapping::CubeSphere(g) => g.outward(addr.face, addr.i, addr.j, addr.k, addr.rung),
        }
    }

    /// What this grid covers.
    #[must_use]
    pub fn domain(&self) -> GridDomain {
        match self {
            GridMapping::Identity(g) => GridDomain::Box {
                half_cells: g.half_cells(),
            },
            GridMapping::CubeSphere(g) => GridDomain::Band {
                cells_per_edge: g.n(),
                floor_m: g.floor_m(),
                band_m: g.band_m(),
                rungs: g.rungs(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::Tier;
    use glam::DVec3;

    const HULL: RealmId = RealmId::Ship(crate::ids::EntityId(44));
    const MOON: RealmId = RealmId::Planet(7);

    fn hull() -> GridMapping {
        GridMapping::Identity(IdentityGrid::from_half_extent_m([16.0, 8.0, 32.0]).expect("slot"))
    }

    fn moon() -> GridMapping {
        GridMapping::CubeSphere(
            ShellGrid::for_body(200_000.0, BandParams::provisional(200_000.0)).expect("moon"),
        )
    }

    /// A band's numbers, or zeros for a box — total, so no test needs an unreachable arm.
    fn band(d: GridDomain) -> (u32, u32, u32, u8) {
        match d {
            GridDomain::Band {
                cells_per_edge,
                floor_m,
                band_m,
                rungs,
            } => (cells_per_edge, floor_m, band_m, rungs),
            GridDomain::Box { .. } => (0, 0, 0, 0),
        }
    }

    /// The cell a seam step landed on, if it crossed one.
    fn crossed(n: Neighbor) -> Option<CellAddr> {
        match n {
            Neighbor::AcrossSeam { addr, .. } => Some(addr),
            Neighbor::Same(_) | Neighbor::Outside => None,
        }
    }

    #[test]
    fn the_same_six_operations_answer_on_both_arms() {
        let rung = Rung::ZERO;
        // A thruster one metre aft of a hull's origin.
        let aft = LatticePos::from_metres(DVec3::new(0.5, 0.5, -0.5), Tier::Fine);
        let a = hull().addr_of(HULL, aft, rung).expect("inside the slot");
        assert_eq!(
            (a.body, a.face, a.i, a.j, a.k),
            (HULL, Face::PosX, 0, 0, -1)
        );
        let centre = hull()
            .cell_center(&a)
            .expect("a flat grid names every centre");
        assert_eq!(hull().addr_of(HULL, centre, rung), Some(a));
        assert_eq!(
            hull().cell_corners(&a).expect("corners")[0].cell().z,
            -1024,
            "the low corner is one metre aft"
        );
        assert_eq!(
            hull().outward(&a),
            None,
            "a flat grid has no outward direction"
        );
        assert_eq!(
            hull().domain(),
            GridDomain::Box {
                half_cells: [16, 8, 32]
            }
        );
        assert_eq!(
            hull().neighbor(&a, Dir6::KMinus),
            Neighbor::Same(CellAddr { k: -2, ..a })
        );
        assert_eq!(
            hull().neighbor(&CellAddr { i: 15, ..a }, Dir6::IPlus),
            Neighbor::Outside,
            "the slot's wall"
        );

        // A landing pad on a moon: a point 30 m above the snapped surface on the +Y face.
        let g = moon();
        let (cells_per_edge, floor_m, band_m, rungs) = band(g.domain());
        assert_eq!((cells_per_edge, rungs), (314_112, 8));
        assert_eq!(band(hull().domain()), (0, 0, 0, 0), "a box is not a band");
        let r = f64::from(floor_m) + f64::from(band_m) * 0.5;
        let pad = LatticePos::from_metres(DVec3::new(1.0, r, 2.0), Tier::Fine);
        let p = g.addr_of(MOON, pad, rung).expect("inside the band");
        assert_eq!((p.body, p.face), (MOON, Face::PosY));
        assert_eq!(
            g.addr_of(MOON, g.cell_center(&p).expect("cell"), rung),
            Some(p),
            "the centre names the cell back"
        );
        let out = g
            .outward(&p)
            .expect("a round grid has an outward direction");
        assert!(out[1] > 0.999, "outward on +Y points up: {out:?}");
        let corners = g.cell_corners(&p).expect("cell");
        // A cell the moon does not have answers None on every operation.
        let absent = CellAddr {
            rung: Rung::new(15).expect("rung"),
            ..p
        };
        assert_eq!(g.cell_center(&absent), None);
        assert_eq!(g.cell_corners(&absent), None);
        assert_eq!(g.outward(&absent), None);
        assert!(
            metres_of(corners[4]).length() > metres_of(corners[0]).length(),
            "the k+1 corner is higher"
        );
        assert_eq!(
            g.neighbor(&p, Dir6::IPlus),
            Neighbor::Same(CellAddr { i: p.i + 1, ..p })
        );
        // The face edge: a cell at i = 0 crosses the seam, and the body and the rung ride along.
        let edge = CellAddr { i: 0, ..p };
        let c = crossed(g.neighbor(&edge, Dir6::IMinus)).expect("the −i side of i = 0 is a seam");
        assert_ne!(c.face, Face::PosY, "the partner cell is on another face");
        assert_eq!((c.body, c.rung, c.k), (MOON, rung, p.k));
        assert_eq!(crossed(Neighbor::Outside), None);
        assert_eq!(
            g.neighbor(&CellAddr { k: 0, ..p }, Dir6::KMinus),
            Neighbor::Outside,
            "the floor"
        );
        // Below the floor there is no cell.
        let deep = LatticePos::from_metres(DVec3::new(0.0, 1000.0, 0.0), Tier::Fine);
        assert_eq!(g.addr_of(MOON, deep, rung), None);
    }

    /// THE NO-DRIFT GOLDEN DIGEST (SL10 V1.3, U-7/U-8): the bit patterns of `cell_center` and the
    /// answers of `addr_of` over a fixed sample of addresses on both arms, folded into one number and
    /// PINNED. Every build — debug or release, aarch64 or x86-64 — must produce this exact value, or
    /// the client and the shard would name different cells for the same position.
    #[test]
    fn the_golden_digest_of_centres_and_addresses_is_pinned() {
        let mut acc = crate::digest::FNV_OFFSET;
        let fold = |acc: u64, p: LatticePos| {
            let m = metres_of(p);
            let acc = crate::digest::fnv1a_u64(acc, m.x.to_bits());
            let acc = crate::digest::fnv1a_u64(acc, m.y.to_bits());
            crate::digest::fnv1a_u64(acc, m.z.to_bits())
        };
        // The round arm: Earth-sized, every face, a lattice of cells at three rungs, plus the address
        // of each centre folded back in.
        let earth = GridMapping::CubeSphere(
            ShellGrid::for_body(6_371_000.0, BandParams::provisional(6_371_000.0)).expect("earth"),
        );
        for level in [0u8, 6, 12] {
            let rung = Rung::new(level).expect("rung");
            let (cells_per_edge, _, _, _) = band(earth.domain());
            let n_l = (cells_per_edge >> level) as i32;
            for face in Face::ALL {
                for step in 0..8 {
                    let i = (n_l as i64 * step / 8) as i32;
                    let j = (n_l as i64 * (7 - step) / 8) as i32;
                    let addr = CellAddr {
                        body: MOON,
                        face,
                        rung,
                        i,
                        j,
                        k: 2,
                    };
                    let centre = earth.cell_center(&addr).expect("a cell the body has");
                    acc = fold(acc, centre);
                    let back = earth.addr_of(MOON, centre, rung).expect("inside");
                    acc = crate::digest::fnv1a(acc, &[back.face.index(), back.rung.level()]);
                    acc = crate::digest::fnv1a_u64(acc, back.i as u64);
                    acc = crate::digest::fnv1a_u64(acc, back.j as u64);
                    acc = crate::digest::fnv1a_u64(acc, back.k as u64);
                }
            }
        }
        // The flat arm: a hull's box on both sides of the origin at two rungs.
        for level in [0u8, 3] {
            let rung = Rung::new(level).expect("rung");
            for (i, j, k) in [
                (-16, -8, -32),
                (-1, 0, -1),
                (0, 0, 0),
                (3, -2, 7),
                (15, 7, 31),
            ] {
                let addr = CellAddr {
                    body: HULL,
                    face: Face::PosX,
                    rung,
                    i: i >> level,
                    j: j >> level,
                    k: k >> level,
                };
                acc = fold(acc, hull().cell_center(&addr).expect("cell"));
                for c in hull().cell_corners(&addr).expect("cell") {
                    acc = fold(acc, c);
                }
            }
        }
        assert_eq!(
            acc, GOLDEN_DIGEST,
            "the grid's centres or addresses changed bit pattern: a drift between builds or targets, \
             or a changed face bend — either re-addresses every saved cell"
        );
    }

    /// The pin. Computed once on aarch64-apple-darwin in debug, confirmed in release and on the
    /// x86-64 target (the x86-64 leg under emulation is a smoke test, not the law's proof — V6 D-5).
    const GOLDEN_DIGEST: u64 = 2_969_074_991_149_515_478;

    #[test]
    fn every_direction_is_listed_once() {
        assert_eq!(Dir6::ALL.len(), 6);
        let mut seen = std::collections::BTreeSet::new();
        for d in Dir6::ALL {
            assert!(seen.insert(format!("{d:?}")));
        }
    }
}
