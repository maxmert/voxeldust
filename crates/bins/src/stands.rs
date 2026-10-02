//! ★ THE PINNED STANDS (ruling B35 item 3; ruling B36 item 8, the owner's Q16; the ground step's design
//! `docs/investigation/2026-10-02/ground_step_design.md` §2.2, bench M0-6): the places every phase of the ground
//! step measures and pictures. The stands rule (`examples/stands_rule.rs`) chose them ONCE in phase 0 and they are
//! pinned here by direction. A later phase prints the rule's new picks beside these pins and ADDS stands; it never
//! moves one, so every step is compared on the same ground (ruling B6).
//!
//! The module also holds THE RELIEF RULE ([`relief_classes`]): one copy, read by the stands rule on the world's bodies
//! and by the Earth instrument on Earth's real heights (`examples/earth_relief.rs`; bench M0-4), so both are read alike.
//!
//! **Example (MEASURED, the first run).** The rule found the largest patch of plains on the crystalline basement of the
//! home planet (36 158 nodes) and pinned the node 229 km from that patch's edge. When the eroded 1 km ground lands in
//! phase 2, the census reads the same direction, so its "before" and "after" lines describe the same plain.

use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};

/// The body a stand stands on: the three bodies of every phase's pictures (ruling B36 item 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StandBody {
    /// The home planet.
    Home,
    /// The home planet's moon.
    Moon,
    /// The third body, chosen by the rule over the bodies' charters (bench M0-5).
    Third,
}

/// The layer of the stands rule that picked a stand (the design §2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StandLayer {
    /// One stand for each class of relief and rock that covers at least one view's area.
    Class,
    /// The places with the most classes within 60 km.
    Border,
    /// Uniform over the sphere from the seed's own hash, split by land and sea floor.
    Random,
    /// The stands that earlier steps already hold.
    Today,
}

/// One pinned stand: its body, its layer, its name, and the unit direction under it in the body's own frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinnedStand {
    pub body: StandBody,
    pub layer: StandLayer,
    pub name: &'static str,
    pub dir: [f64; 3],
}

/// ★ THE THIRD BODY (bench M0-5; ruling B36 item 9, the owner's "The desert world"): the solid body whose node the
/// 8 192 m target sets, with the largest charter distance from the home planet, dry and with air — 7 393 km, 1.3 g,
/// 1.68 bar, 350 K, no water.
pub const THIRD_BODY_SEED: u64 = 17782101822166660494;

/// ★ THE PINNED STANDS: the first run of the stands rule on generator 22 (2026-10-02;
/// `docs/investigation/2026-10-02/phase0_stands_census.txt`), every pick at its node's centre direction. The climate
/// stands, the family stands and the forest stands are ADDED by their phases (the rule's header says why each waits).
#[rustfmt::skip]
pub const PINNED: &[PinnedStand] = &[
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "sea floor / crystalline basement", dir: [0.482152, -0.840879, 0.245870] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "sea floor / folded belt", dir: [0.648706, -0.760688, -0.023099] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "sea floor / flat shelf", dir: [0.258977, 0.894252, 0.365027] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "sea floor / rift basalt", dir: [0.792717, 0.609588, -0.001536] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "sea floor / deep sediment", dir: [0.923817, 0.073536, 0.375706] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "basin floor / crystalline basement", dir: [-0.144312, -0.770120, -0.621361] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "basin floor / folded belt", dir: [0.439432, 0.243198, -0.864728] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "basin floor / flat shelf", dir: [-0.237197, -0.654849, 0.717573] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "basin floor / rift basalt", dir: [-0.207662, -0.446537, -0.870334] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "basin floor / deep sediment", dir: [-0.247585, -0.633436, 0.733117] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "plains / crystalline basement", dir: [-0.489087, 0.758909, 0.429944] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "plains / folded belt", dir: [-0.085843, -0.940174, -0.329704] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "plains / flat shelf", dir: [-0.510105, 0.053847, -0.858425] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "plains / rift basalt", dir: [-0.258313, -0.887679, 0.381183] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "plains / deep sediment", dir: [0.000593, -0.395760, 0.918354] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "tablelands / crystalline basement", dir: [-0.565896, -0.772604, -0.287826] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "tablelands / folded belt", dir: [0.640425, -0.455174, -0.618605] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "tablelands / flat shelf", dir: [-0.163727, 0.297525, 0.940570] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "tablelands / rift basalt", dir: [0.417430, -0.310638, -0.853965] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "tablelands / deep sediment", dir: [0.291628, -0.744437, -0.600639] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "mountains / crystalline basement", dir: [-0.063058, -0.502028, -0.862549] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "mountains / flat shelf", dir: [0.243108, -0.782541, -0.573174] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "mountains / rift basalt", dir: [-0.555430, 0.037046, -0.830738] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Class, name: "mountains / deep sediment", dir: [0.295672, -0.683675, -0.667208] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Border, name: "border 1", dir: [0.073721, 0.328812, 0.941514] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Border, name: "border 2", dir: [0.650699, -0.215919, -0.727990] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Border, name: "border 3", dir: [0.353731, -0.661814, -0.660966] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random sea floor 1", dir: [0.740065, -0.508828, -0.439772] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random sea floor 2", dir: [-0.646311, 0.503711, 0.573199] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 1", dir: [-0.160690, 0.349328, 0.923119] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random sea floor 3", dir: [-0.817660, -0.339930, 0.464628] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random sea floor 4", dir: [-0.648427, 0.695975, -0.308482] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 2", dir: [-0.074360, -0.688197, -0.721703] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 3", dir: [-0.897207, 0.351474, -0.267366] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 4", dir: [-0.047979, -0.339032, 0.939551] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 5", dir: [-0.583148, -0.800510, -0.138283] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 6", dir: [-0.920224, 0.068469, -0.385356] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 7", dir: [-0.963751, -0.024281, -0.265697] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 8", dir: [0.145709, -0.413692, -0.898681] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 9", dir: [-0.810467, 0.175799, 0.558783] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 10", dir: [-0.442406, -0.771560, -0.457135] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 11", dir: [-0.775439, 0.182909, 0.604349] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 12", dir: [-0.465535, -0.599113, -0.651415] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 13", dir: [0.216438, -0.533122, -0.817885] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 14", dir: [0.022478, -0.773163, -0.633809] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 15", dir: [-0.630802, 0.060572, -0.773576] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 16", dir: [-0.200175, -0.940385, -0.274965] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 17", dir: [0.223689, -0.723319, 0.653279] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 18", dir: [-0.391508, -0.865144, -0.313443] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 19", dir: [-0.354982, -0.866449, -0.351074] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Random, name: "random land 20", dir: [-0.209737, 0.499484, 0.840551] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "the belt", dir: [0.617270, -0.437286, -0.654033] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "the coast", dir: [0.194169, -0.868118, -0.456804] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "the plain", dir: [0.579815, -0.574634, -0.577590] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "pocket 1", dir: [-0.202087, -0.550667, -0.809893] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "pocket 2", dir: [0.290360, -0.700976, -0.651401] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "the high lake", dir: [-0.395463, -0.825174, -0.403357] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "the summit", dir: [0.631321, -0.422643, -0.650236] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "the ground stand", dir: [-0.935990, -0.224603, -0.271064] },
    PinnedStand { body: StandBody::Home, layer: StandLayer::Today, name: "the dusk stand", dir: [-0.965053, 0.069298, -0.252725] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Class, name: "basin floor / crystalline basement", dir: [-0.248624, 0.317935, 0.914934] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Class, name: "basin floor / deep sediment", dir: [0.726636, -0.678179, -0.109878] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Class, name: "tablelands / crystalline basement", dir: [0.356809, 0.621467, 0.697471] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Class, name: "tablelands / deep sediment", dir: [-0.009136, 0.611770, -0.790983] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Class, name: "mountains / crystalline basement", dir: [-0.965852, -0.238709, 0.100738] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Class, name: "mountains / deep sediment", dir: [0.839461, 0.532743, -0.107192] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Border, name: "border 1", dir: [0.683984, -0.292086, -0.668469] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Border, name: "border 2", dir: [0.714952, -0.024782, -0.698734] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Border, name: "border 3", dir: [0.825169, 0.124777, -0.550933] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 1", dir: [-0.149707, -0.640506, -0.753220] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 2", dir: [-0.342185, 0.388857, -0.855394] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 3", dir: [-0.624248, 0.134588, -0.769546] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 4", dir: [0.647490, 0.693754, -0.315377] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 5", dir: [0.507655, 0.319997, 0.799930] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 6", dir: [0.146708, 0.192834, 0.970202] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 7", dir: [0.773352, -0.210150, 0.598133] },
    PinnedStand { body: StandBody::Moon, layer: StandLayer::Random, name: "random land 8", dir: [0.689442, 0.460313, -0.559269] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "basin floor / crystalline basement", dir: [0.102946, -0.028293, 0.994284] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "basin floor / folded belt", dir: [-0.035636, 0.377220, 0.925438] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "basin floor / rift basalt", dir: [0.541908, -0.573137, 0.614694] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "basin floor / deep sediment", dir: [0.437942, 0.502121, -0.745709] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "tablelands / crystalline basement", dir: [-0.990221, 0.139494, 0.001657] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "tablelands / folded belt", dir: [-0.995306, 0.086255, 0.043887] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "tablelands / rift basalt", dir: [0.147016, -0.352871, 0.924050] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "tablelands / deep sediment", dir: [0.862840, 0.246242, 0.441443] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "mountains / crystalline basement", dir: [-0.268716, 0.737977, 0.619016] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "mountains / folded belt", dir: [-0.268455, -0.948666, -0.167229] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "mountains / rift basalt", dir: [-0.739738, -0.488518, 0.462750] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Class, name: "mountains / deep sediment", dir: [0.370456, 0.649111, 0.664393] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Border, name: "border 1", dir: [-0.645417, 0.332637, 0.687597] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Border, name: "border 2", dir: [0.385529, -0.393463, -0.834598] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Border, name: "border 3", dir: [-0.954185, -0.293097, 0.060219] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 1", dir: [0.099603, -0.775776, -0.623098] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 2", dir: [-0.716543, 0.452718, -0.530671] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 3", dir: [-0.310379, -0.867378, -0.388999] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 4", dir: [0.406496, -0.874857, -0.263413] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 5", dir: [0.364870, 0.557473, -0.745717] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 6", dir: [-0.234282, -0.053655, 0.970687] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 7", dir: [0.415149, -0.351822, -0.838971] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 8", dir: [-0.291944, 0.950433, -0.106981] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 9", dir: [0.524442, 0.189619, -0.830064] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 10", dir: [0.883710, 0.170024, 0.436060] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 11", dir: [-0.690853, -0.421481, 0.587432] },
    PinnedStand { body: StandBody::Third, layer: StandLayer::Random, name: "random land 12", dir: [-0.822645, 0.283062, 0.493083] },
];

/// Hammond's gentle slope: under 8 % (Hammond 1964; ESTIMATE source).
pub const GENTLE_SLOPE: f64 = 0.08;
/// Hammond's relief classes as Karagulle et al. 2017 group them: the local relief under 90 m, and under 300 m
/// (ESTIMATE sources).
pub const LOW_RELIEF_M: i32 = 90;
pub const HILL_RELIEF_M: i32 = 300;

/// The relief codes [`relief_classes`] gives, in the order of [`RELIEF`].
pub const SEA_FLOOR: u8 = 0;
pub const BASIN_FLOOR: u8 = 1;
pub const PLAINS: u8 = 2;
pub const TABLELANDS: u8 = 3;
pub const HILLS: u8 = 4;
pub const MOUNTAINS: u8 = 5;

/// The relief classes' names, in the order of their codes.
pub const RELIEF: [&str; 6] = [
    "sea floor",
    "basin floor",
    "plains",
    "tablelands",
    "hills",
    "mountains",
];

/// Hammond's two readings over one LAND node's window (the node and its land neighbours).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReliefReading {
    /// The window's nodes whose steepest slope to a land neighbour is under [`GENTLE_SLOPE`].
    pub gentle: u8,
    /// The window's nodes: the node and its land neighbours.
    pub window: u8,
    /// The local relief: the highest less the lowest height in the window, metres.
    pub span_m: i32,
    /// The window's lowest height, metres: where it stands under the sea level, the window reaches into a closed basin.
    pub lo_m: i32,
}

impl ReliefReading {
    /// The relief code of these readings: [`BASIN_FLOOR`] or [`PLAINS`] (by `inland`), [`TABLELANDS`], [`HILLS`] or
    /// [`MOUNTAINS`].
    #[must_use]
    pub fn class(self, inland: bool) -> u8 {
        if u32::from(self.gentle) * 2 >= u32::from(self.window) {
            match (self.span_m < LOW_RELIEF_M, inland) {
                (true, true) => BASIN_FLOOR,
                (true, false) => PLAINS,
                (false, _) => TABLELANDS,
            }
        } else if self.span_m < HILL_RELIEF_M {
            HILLS
        } else {
            MOUNTAINS
        }
    }
}

/// ★ HAMMOND'S TWO READINGS of every node (ruling B35 item 3; the design §2.2 layer 1), ONE COPY for every body and for
/// Earth's data: over the node and its eight neighbours, the GENTLE share (the steepest slope to a neighbour, by the
/// lattice's own chord, under [`GENTLE_SLOPE`]) and the LOCAL RELIEF (the highest less the lowest height). `None` for a
/// node that is not land.
///
/// ★ A LAND node reads LAND only: Hammond's relief and gentle share measure the land, so a sea-floor neighbour is left
/// out of both the slope and the window (MEASURED on the stands rule's first run: with it, the pinned coast read
/// "mountains" for the drop to its shelf).
///
/// `z_m` is each node's height in whole metres and `land` whether it is land.
#[must_use]
pub fn relief_readings(
    lattice: &MacroLattice,
    z_m: &[i32],
    land: &[bool],
) -> Vec<Option<ReliefReading>> {
    let n = lattice.node_count();
    let is_land = |m: u32| m != NO_NODE && land[m as usize];
    let slope: Vec<f64> = (0..n as u32)
        .map(|v| {
            lattice
                .neighbours(v)
                .iter()
                .filter(|&&m| is_land(m))
                .map(|&m| {
                    f64::from((z_m[v as usize] - z_m[m as usize]).abs())
                        / f64::from(lattice.chord_m(v, m).max(1))
                })
                .fold(0.0, f64::max)
        })
        .collect();
    (0..n as u32)
        .map(|v| {
            if !land[v as usize] {
                return None;
            }
            let mut window = vec![v];
            window.extend(lattice.neighbours(v).iter().filter(|&&m| is_land(m)));
            let gentle = window
                .iter()
                .filter(|&&w| slope[w as usize] < GENTLE_SLOPE)
                .count();
            let (lo, hi) = window.iter().fold((i32::MAX, i32::MIN), |(lo, hi), &w| {
                (lo.min(z_m[w as usize]), hi.max(z_m[w as usize]))
            });
            Some(ReliefReading {
                gentle: gentle as u8,
                window: window.len() as u8,
                span_m: hi - lo,
                lo_m: lo,
            })
        })
        .collect()
}

/// ★ THE RELIEF RULE: every node's relief code from its two readings ([`relief_readings`]):
/// - a node that is not land: [`SEA_FLOOR`];
/// - half or more gentle, under [`LOW_RELIEF_M`]: [`BASIN_FLOOR`] where its water ends inland (`inland`), else
///   [`PLAINS`];
/// - half or more gentle, [`LOW_RELIEF_M`] or more: [`TABLELANDS`];
/// - under half gentle, under [`HILL_RELIEF_M`]: [`HILLS`]; else [`MOUNTAINS`].
///
/// **Example.** On the home planet a node of the crystalline basement stands 120 m over its lowest land neighbour and
/// every slope around it is under 8 %: half or more of its window is gentle and its relief is 90 m or more, so it reads
/// "tablelands". The same node on a body of nodes 1 km wide could read "hills": the rule reads the grid it is given.
#[must_use]
pub fn relief_classes(
    lattice: &MacroLattice,
    z_m: &[i32],
    land: &[bool],
    inland: &[bool],
) -> Vec<u8> {
    relief_readings(lattice, z_m, land)
        .iter()
        .zip(inland)
        .map(|(reading, &inland)| reading.map_or(SEA_FLOOR, |r| r.class(inland)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_seed::bend::Face;

    /// The home moon's lattice: a real body's grid, small enough for a unit test.
    fn moon_lattice() -> MacroLattice {
        MacroLattice::of(&vd_terrain::home::home_moon()).expect("the home moon has a macro lattice")
    }

    #[test]
    fn flat_land_reads_plains_or_basin_floor_and_water_reads_sea_floor() {
        let lattice = moon_lattice();
        let n = lattice.node_count();
        let z = vec![0; n];
        assert_eq!(
            relief_classes(&lattice, &z, &vec![true; n], &vec![false; n]),
            vec![PLAINS; n]
        );
        assert_eq!(
            relief_classes(&lattice, &z, &vec![true; n], &vec![true; n]),
            vec![BASIN_FLOOR; n]
        );
        assert_eq!(
            relief_classes(&lattice, &z, &vec![false; n], &vec![false; n]),
            vec![SEA_FLOOR; n]
        );
    }

    /// One node 1 000 m over a flat field, in a face's middle: every node that touches it is steep (12 % across a node
    /// of the moon's 8.2 km), so the node and its four edge neighbours (six of nine steep) read mountains, its four
    /// corner neighbours (four of nine steep, the relief 1 000 m) read tablelands, and two nodes away the window holds
    /// no relief: plains.
    #[test]
    fn a_single_tall_node_makes_mountains_ringed_by_tablelands() {
        let lattice = moon_lattice();
        let n = lattice.node_count();
        let mid = (lattice.edge / 2) as i32;
        let at = |i: i32, j: i32| lattice.index(Face::PosZ, mid + i, mid + j);
        let mut z = vec![0; n];
        z[at(0, 0) as usize] = 1_000;
        let relief = relief_classes(&lattice, &z, &vec![true; n], &vec![false; n]);
        assert_eq!(relief[at(0, 0) as usize], MOUNTAINS);
        for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            assert_eq!(
                relief[at(i, j) as usize],
                MOUNTAINS,
                "edge neighbour ({i}, {j})"
            );
        }
        for (i, j) in [(1, 1), (-1, 1), (1, -1), (-1, -1)] {
            assert_eq!(
                relief[at(i, j) as usize],
                TABLELANDS,
                "corner neighbour ({i}, {j})"
            );
        }
        for (i, j) in [(2, 0), (0, -2), (2, 2), (3, 1)] {
            assert_eq!(
                relief[at(i, j) as usize],
                PLAINS,
                "two nodes away ({i}, {j})"
            );
        }
    }

    /// A deep sea node beside flat land: the land around it reads its land neighbours only, so the drop to the sea floor
    /// makes no relief. The control: the same node as land makes mountains, so the rule is what keeps the coast plain.
    #[test]
    fn land_reads_land_only() {
        let lattice = moon_lattice();
        let n = lattice.node_count();
        let mid = (lattice.edge / 2) as i32;
        let at = |i: i32, j: i32| lattice.index(Face::NegY, mid + i, mid + j);
        let mut z = vec![0; n];
        z[at(0, 0) as usize] = -5_000;
        let mut land = vec![true; n];
        land[at(0, 0) as usize] = false;
        let relief = relief_classes(&lattice, &z, &land, &vec![false; n]);
        assert_eq!(relief[at(0, 0) as usize], SEA_FLOOR);
        assert_eq!(relief[at(1, 0) as usize], PLAINS);
        assert_eq!(relief[at(1, 1) as usize], PLAINS);
        let all_land = relief_classes(&lattice, &z, &vec![true; n], &vec![false; n]);
        assert_eq!(all_land[at(1, 0) as usize], MOUNTAINS);
    }
}
