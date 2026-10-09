//! ★ THE PICTURE STANDS OF PHASE 2 (ruling B53: *"I didn't see the mountain region or anything else"* — the owner's one
//! look at the end result needs stands that hold a MOUNTAIN REGION and the other kinds of country in frame; ruling B36
//! item 8: three bodies, two suns; the stands rule's own promise: the FAMILY stands are ADDED when a family's facts
//! exist, and phase 2's belts are such facts).
//!
//! **WHAT IT PICKS, BY A STATED RULE** (printed with every pick, so a later phase reads the same places):
//! 1. On each body — the home planet, its moon, the third body — the solve's own families over the solved field
//!    (`vd_terrain::solve::solve_full`, the shard's own laws): THE COLLISION WEDGE (the wedge nodes of a collision off a
//!    plateau), THE ANDEAN PLATEAU (the wedge nodes of an Andean margin on a plateau), THE MOUNTAINS (land nodes of the
//!    relief rule's mountain class, `vd_bins::stands::relief_readings`), THE OLD PLAINS (land plains no wedge raised),
//!    and THE COAST (a land node beside the connected ocean).
//! 2. For each family and each sun band — LOW (the star 12°–18° over the horizon at the clock's genesis, the picture
//!    gate's raking light) and HIGH (40°–60°) — the LARGEST patch of the family (eight-neighbour connected) whose MIDDLE
//!    stands under that star; the middle is the patch's node farthest from its edge by the summed chord (the stands
//!    rule's own middle), ties to the smaller index. A family with no patch in a band prints so.
//! 3. Two views of each pick, the star over the shoulder (120° off the nose, the picture gate's own rule): NEAR — the
//!    eye 1 500 m over the ground 40 km from the patch's highest node, the nose on that node; ALOFT — the eye 150 km up,
//!    400 km from the patch's middle on the same bearing, the nose on the middle.
//!
//! The star's direction at the clock's genesis is minus the body's place on its orbit at tick 0 in its own frame (the
//! picture gate's own reading: a fresh demand cluster boots its clock at genesis and no body spins); the moon reads its
//! planet's, because the moon's distance from its planet is a few millionths of the planet's from the star.
//!
//! It also prints THE BAND (ruling B23; the belts' design L0: the ladder band must hold what the law allows): each
//! body's relief band at rung 0 beside the solved field's highest and lowest node over the body's radius.
//!
//! `cargo run --release -p vd-bins --example family_stands [-- home moon third]`

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use vd_bins::stands::{MOUNTAINS, PLAINS, THIRD_BODY_SEED, merged, relief_readings};
use vd_core::glam::{DMat3, DQuat, DVec3};
use vd_core::pose::RealmId;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::belts::{BELT_ANDEAN, BELT_COLLISION};
use vd_terrain::home::{home_moon, home_moon_solve_words, home_planet, home_solve_words};
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};
use vd_terrain::solve::{FACIES_SEA, Schedule, SolveWords, Z_STEPS_PER_M, solve_full};

/// The universe the dev cluster boots (`vd_bins::DEV`).
const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
/// The star's bands over the horizon, degrees: the raking light and the high light (ruling B36 item 8).
const BANDS: [(&str, f64, f64); 2] = [("low", 12.0, 18.0), ("high", 40.0, 60.0)];
/// The star's azimuth off the nose, degrees (the picture gate's `SUN_OFF_NOSE_DEG`).
const SUN_OFF_NOSE_DEG: f64 = 120.0;
/// The near view: the eye's height over its ground and its distance from the patch's highest node, metres.
const NEAR_HEIGHT_M: f64 = 1_500.0;
const NEAR_BACK_M: f64 = 40_000.0;
/// The aloft view: the eye's height and its distance from the patch's middle, metres.
const ALOFT_HEIGHT_M: f64 = 150_000.0;
const ALOFT_BACK_M: f64 = 400_000.0;

/// One body to stand on: its name, its realm, its definition, its words, and its star at genesis.
struct Body {
    name: &'static str,
    realm: RealmId,
    def: BodyDefinition,
    words: SolveWords,
    sun: DVec3,
}

/// The star's direction at genesis for a body on `elements` (minus its place on its orbit at tick 0).
fn sun_of(elements: &vd_physics::celestial::OrbitalElements) -> DVec3 {
    let p = vd_physics::celestial::orbital_state(elements, 0.0).position;
    -DVec3::new(p.x, p.y, p.z).normalize()
}

fn bodies(wanted: &BTreeSet<String>) -> Vec<Body> {
    let home_sun = sun_of(&vd_bins::home_orbit(UNIVERSE).expect("the home planet's orbit"));
    let mut out = Vec::new();
    if wanted.contains("home") {
        let def = home_planet();
        out.push(Body {
            name: "the home planet",
            realm: RealmId::Planet(def.seed()),
            def,
            words: home_solve_words(),
            sun: home_sun,
        });
    }
    if wanted.contains("moon") {
        let def = home_moon();
        out.push(Body {
            name: "the home moon",
            realm: RealmId::Planet(def.seed()),
            def,
            words: home_moon_solve_words(),
            sun: home_sun,
        });
    }
    if wanted.contains("third") {
        let system = vd_core::worldgen::HOME_SYSTEM;
        let galaxy = vd_core::worldgen::GALAXY;
        let (rows, movers) = vd_bins::home_system_world(UNIVERSE);
        let realm = RealmId::Planet(THIRD_BODY_SEED);
        let radius_m = rows
            .iter()
            .find(|r| r.realm == realm)
            .and_then(|r| match r.look {
                Some(vd_core::geometry::Boundary::Shell { r }) => Some(r),
                _ => None,
            })
            .expect("the third body is a round planet of the home system");
        let charter = vd_bins::body_charter(
            UNIVERSE,
            &BTreeSet::from([system]),
            &BTreeSet::from([galaxy]),
            realm,
        )
        .expect("the third body states a charter");
        let def = BodyDefinition::from_seed(
            THIRD_BODY_SEED,
            radius_m,
            vd_bins::facts_of_charter(&charter),
        )
        .expect("the third body has a ladder");
        let elements = movers
            .iter()
            .find_map(|(r, e)| (*r == realm).then_some(*e))
            .expect("the third body's orbit");
        out.push(Body {
            name: "the third body (the desert world)",
            realm,
            def,
            words: vd_bins::solve_words_of(&charter),
            sun: sun_of(&elements),
        });
    }
    out
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

/// The patches of a family: its nodes' eight-neighbour components, each listed, largest first (ties to the smaller
/// first node).
fn patches(lattice: &MacroLattice, member: &[bool]) -> Vec<Vec<u32>> {
    let n = member.len();
    let mut seen = vec![false; n];
    let mut out: Vec<Vec<u32>> = Vec::new();
    for start in 0..n {
        if !member[start] || seen[start] {
            continue;
        }
        let mut comp = vec![start as u32];
        seen[start] = true;
        let mut k = 0;
        while k < comp.len() {
            let v = comp[k];
            k += 1;
            for m in lattice.neighbours(v) {
                if m != NO_NODE && member[m as usize] && !seen[m as usize] {
                    seen[m as usize] = true;
                    comp.push(m);
                }
            }
        }
        out.push(comp);
    }
    out.sort_by_key(|c| (Reverse(c.len()), c[0]));
    out
}

/// A patch's middle: its node farthest from its edge by the summed chord (Dijkstra from the nodes with a neighbour
/// outside the patch), ties to the smaller index; and that distance, metres.
fn middle(lattice: &MacroLattice, radius_m: f64, member: &[bool], comp: &[u32]) -> (u32, f64) {
    let chord = |a: u32, b: u32| (dir_of(lattice, a) - dir_of(lattice, b)).length() * radius_m;
    let mut dist: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for &v in comp {
        let edge = lattice
            .neighbours(v)
            .iter()
            .any(|&m| m == NO_NODE || !member[m as usize]);
        if edge {
            dist.insert(v, 0.0);
            heap.push(Reverse((0, v)));
        }
    }
    while let Some(Reverse((d_mm, v))) = heap.pop() {
        let d = d_mm as f64 / 1_000.0;
        if dist.get(&v).is_some_and(|&best| d > best) {
            continue;
        }
        for m in lattice.neighbours(v) {
            if m == NO_NODE || !member[m as usize] {
                continue;
            }
            let next = d + chord(v, m);
            if dist.get(&m).is_none_or(|&best| next < best) {
                dist.insert(m, next);
                heap.push(Reverse(((next * 1_000.0) as u64, m)));
            }
        }
    }
    comp.iter()
        .map(|&v| (v, dist.get(&v).copied().unwrap_or(0.0)))
        .fold((comp[0], -1.0), |best, (v, d)| {
            if d > best.1 || (d == best.1 && v < best.0) {
                (v, d)
            } else {
                best
            }
        })
}

/// A spawn entry: the eye at `eye` (the body's frame, metres) with its nose on `target`, its up the radial.
fn pose(realm: RealmId, account: u64, eye: DVec3, target: DVec3) -> String {
    let forward = (target - eye).normalize();
    let up_hint = eye.normalize();
    let right = forward.cross(up_hint).normalize();
    let up = right.cross(forward).normalize();
    let q = DQuat::from_mat3(&DMat3::from_cols(right, up, -forward)).normalize();
    let RealmId::Planet(seed) = realm else {
        unreachable!("every stand body is a planet realm")
    };
    format!(
        "{account}=Planet({seed}):{},{},{}@{},{},{},{}",
        eye.x, eye.y, eye.z, q.x, q.y, q.z, q.w
    )
}

/// The tangent direction at `d` turned `deg` about the radial from the star's azimuth.
fn off_sun(d: DVec3, sun: DVec3, deg: f64) -> DVec3 {
    let toward = (sun - d * sun.dot(d)).normalize();
    (DQuat::from_axis_angle(d, deg.to_radians()) * toward).normalize()
}

/// A direction `back_m` along the surface from `d` against the tangent `t`.
fn back_from(d: DVec3, t: DVec3, back_m: f64, radius_m: f64) -> DVec3 {
    let axis = d.cross(t).normalize();
    (DQuat::from_axis_angle(axis, -back_m / radius_m) * d).normalize()
}

fn main() {
    let args: BTreeSet<String> = std::env::args().skip(1).collect();
    let wanted: BTreeSet<String> = if args.is_empty() {
        ["home", "moon", "third"].map(str::to_string).into()
    } else {
        args
    };
    println!(
        "family_stands: the picture stands of phase 2 (ruling B53), generator {}; bodies {wanted:?}",
        vd_terrain::GENERATOR_VERSION
    );
    let mut account = 1001u64;
    for body in bodies(&wanted) {
        let started = std::time::Instant::now();
        let (state, facies, report) = solve_full(
            &body.def,
            &body.words,
            Schedule::standard(body.words.age_yr),
        )
        .expect("the body solves");
        let lattice = body.def.macro_lattice().expect("a lattice");
        let climate = vd_terrain::climate::climate(
            &body.def,
            &lattice,
            &body.words,
            &state.z,
            (body.words.water_km3 > 0).then_some(state.sea_view()),
        );
        let artifact = Artifact::of(&state, &facies, &climate, body.words.water_km3 > 0);
        let def = body.def.with_sea_m(artifact.sea());
        let radius_m = def.radius_m();
        let n = lattice.node_count();
        let sea_z = state.datum_z();
        let land: Vec<bool> = (0..n)
            .map(|i| (facies[i] & FACIES_SEA == 0) & (state.z[i] > sea_z))
            .collect();
        let steps = f64::from(Z_STEPS_PER_M);
        let over: Vec<i32> = state
            .z
            .iter()
            .map(|&z| ((f64::from(z) - f64::from(sea_z)) / steps).round() as i32)
            .collect();
        let readings = relief_readings(&lattice, &over, &land);
        let (lo, hi) = state.range();
        println!(
            "\n== {} ({:?}): solved in {:.1} s; the star at genesis {:.6} {:.6} {:.6}; THE BAND at rung 0 ±{:.0} m over the \
             body's radius, the solved field {:.0}..{:.0} m (the datum {:.0} m); the peak law: {} collapses, {} m left",
            body.name,
            body.realm,
            started.elapsed().as_secs_f64(),
            body.sun.x,
            body.sun.y,
            body.sun.z,
            def.relief_bound_m(0),
            f64::from(lo) / steps,
            f64::from(hi) / steps,
            f64::from(sea_z) / steps,
            report.peak.0,
            f64::from(report.peak.1) / steps
        );
        let belts = &report.belts;
        let ocean_beside = |i: usize| {
            lattice
                .neighbours(i as u32)
                .iter()
                .any(|&m| m != NO_NODE && state.ocean[m as usize])
        };
        let families: [(&str, Vec<bool>); 5] = [
            (
                "the collision wedge",
                (0..n)
                    .map(|i| land[i] && belts.kind[i] == BELT_COLLISION && !belts.plateau[i])
                    .collect(),
            ),
            (
                "the Andean plateau",
                (0..n)
                    .map(|i| land[i] && belts.kind[i] == BELT_ANDEAN && belts.plateau[i])
                    .collect(),
            ),
            (
                "the mountains",
                (0..n)
                    .map(|i| {
                        land[i] && readings[i].is_some_and(|r| merged(r.class(false)) == MOUNTAINS)
                    })
                    .collect(),
            ),
            (
                "the old plains",
                (0..n)
                    .map(|i| {
                        land[i]
                            && belts.total[i] == 0
                            && readings[i].is_some_and(|r| merged(r.class(false)) == PLAINS)
                    })
                    .collect(),
            ),
            (
                "the coast",
                (0..n).map(|i| land[i] && ocean_beside(i)).collect(),
            ),
        ];
        let mut entries: Vec<String> = Vec::new();
        for (family, member) in &families {
            let comps = patches(&lattice, member);
            println!(
                "  {family}: {} nodes in {} patches (the largest {})",
                member.iter().filter(|&&m| m).count(),
                comps.len(),
                comps.first().map_or(0, Vec::len)
            );
            for (band, lo_deg, hi_deg) in BANDS {
                let pick = comps.iter().find_map(|comp| {
                    let (mid, edge_m) = middle(&lattice, radius_m, member, comp);
                    let elevation = body.sun.dot(dir_of(&lattice, mid)).asin().to_degrees();
                    ((lo_deg..=hi_deg).contains(&elevation))
                        .then_some((comp, mid, edge_m, elevation))
                });
                let Some((comp, mid, edge_m, elevation)) = pick else {
                    println!(
                        "    {band} star ({lo_deg}°–{hi_deg}°): no patch's middle stands under it"
                    );
                    continue;
                };
                let top = comp
                    .iter()
                    .copied()
                    .max_by_key(|&v| (state.z[v as usize], Reverse(v)))
                    .expect("a patch has a node");
                let (dm, dt) = (dir_of(&lattice, mid), dir_of(&lattice, top));
                let ground = |d: DVec3| {
                    vd_terrain::height::height_field_m(&def, &artifact, d.to_array(), 0)
                        .expect("the solved field covers every direction")
                };
                // NEAR: the nose on the patch's highest node, the star over the shoulder.
                let nose_t = off_sun(dt, body.sun, SUN_OFF_NOSE_DEG);
                let de = back_from(dt, nose_t, NEAR_BACK_M, radius_m);
                let near_eye = de * (ground(de) + NEAR_HEIGHT_M);
                let near = pose(body.realm, account, near_eye, dt * ground(dt));
                // ALOFT: the nose on the patch's middle from 150 km up, 400 km back on the same rule's bearing.
                let nose_m = off_sun(dm, body.sun, SUN_OFF_NOSE_DEG);
                let da = back_from(dm, nose_m, ALOFT_BACK_M, radius_m);
                let aloft_eye = da * (ground(da) + ALOFT_HEIGHT_M);
                let aloft = pose(body.realm, account + 1, aloft_eye, dm * ground(dm));
                println!(
                    "    {band} star: the patch of {} nodes; its middle node {mid} ({:.6} {:.6} {:.6}) {:.0} km from \
                     its edge, the star {elevation:.1}° up; its highest node {top}, {:.0} m over the sea; NEAR account \
                     {account}, ALOFT account {}",
                    comp.len(),
                    dm.x,
                    dm.y,
                    dm.z,
                    edge_m / 1_000.0,
                    f64::from(over[top as usize]),
                    account + 1
                );
                println!("      {near}");
                println!("      {aloft}");
                entries.push(near);
                entries.push(aloft);
                account += 2;
            }
        }
        println!(
            "  VD_SPAWN_POSES for {} ({} stands): {}",
            body.name,
            entries.len(),
            entries.join(";")
        );
    }
}
