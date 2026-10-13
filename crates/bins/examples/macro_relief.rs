//! ★ THE 8 km WORLD'S RELIEF (2026-10-13, ruling B94 step 2): each land node with its eight neighbours (about 25 km
//! across) — the window's highest less its lowest row — over the whole planet's land and over the land within 1 000 km
//! of hull 15's berth, as the 10th, 50th and 90th percentiles and the shares under 20, 50 and 100 m. Read for the game's
//! stored world and for a bench world (`belts_bench --artifact=DIR`), so a candidate law's effect on the plains is one
//! number beside the other. Bench only.
//! `VD_REALM_STORE=<home planet store> cargo run --release -p vd-bins --example macro_relief -- [store] [FILE.art]…`

use std::collections::BTreeSet;

use vd_core::pose::RealmId;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::Artifact;
use vd_terrain::macro_lattice::{MacroLattice, NO_NODE};

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;
const HOME: u64 = 4_030_111_653_607_004_909;
/// Hull 15's standing centre (the dev cluster's spawn), metres in the planet's frame.
const BERTH: [f64; 3] = [
    3_273_359.478_594_958_3,
    -2_850_257.825_591_502_7,
    -4_631_027.492_918_959,
];

fn unit(g: [vd_recipe::Gi; 3]) -> [f64; 3] {
    let s = (1u64 << 40) as f64;
    let v = [
        g[0].raw() as f64 / s,
        g[1].raw() as f64 / s,
        g[2].raw() as f64 / s,
    ];
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn line(name: &str, mut r: Vec<f32>) {
    if r.is_empty() {
        println!("  {name:<44}: no land window");
        return;
    }
    r.sort_by(f32::total_cmp);
    let q = |f: f64| r[((r.len() - 1) as f64 * f).round() as usize];
    let under = |m: f32| 100.0 * r.iter().filter(|v| **v < m).count() as f64 / r.len() as f64;
    println!(
        "  {name:<44}: p10 {:>5.0} p50 {:>5.0} p90 {:>5.0} m | under 20 m {:>5.1} %, 50 m {:>5.1} %, 100 m {:>5.1} % | {} nodes",
        q(0.1),
        q(0.5),
        q(0.9),
        under(20.0),
        under(50.0),
        under(100.0),
        r.len()
    );
}

fn read(name: &str, artifact: &Artifact, lattice: &MacroLattice, radius_m: f64) {
    let sea = i32::from(artifact.sea_m);
    let z = |n: u32| i32::from(artifact.rows[n as usize].z_m);
    let berth = {
        let l = (BERTH[0] * BERTH[0] + BERTH[1] * BERTH[1] + BERTH[2] * BERTH[2]).sqrt();
        [BERTH[0] / l, BERTH[1] / l, BERTH[2] / l]
    };
    let (mut all, mut near) = (Vec::new(), Vec::new());
    for n in 0..lattice.node_count() as u32 {
        if z(n) <= sea {
            continue;
        }
        let (mut lo, mut hi, mut land) = (z(n), z(n), true);
        for m in lattice.neighbours(n) {
            if m == NO_NODE {
                continue;
            }
            land &= z(m) > sea;
            lo = lo.min(z(m));
            hi = hi.max(z(m));
        }
        if !land {
            continue;
        }
        let r = (hi - lo) as f32;
        all.push(r);
        let d = unit(lattice.direction(n));
        let cos = d[0] * berth[0] + d[1] * berth[1] + d[2] * berth[2];
        if cos.clamp(-1.0, 1.0).acos() * radius_m <= 1_000_000.0 {
            near.push(r);
        }
    }
    println!("{name} (the sea {} m)", artifact.sea_m);
    line("the whole land, 3 × 3 nodes (~25 km)", all);
    line("the land within 1 000 km of the berth", near);
}

fn main() {
    let realm = RealmId::Planet(HOME);
    let (rows, _) = vd_bins::home_system_world(UNIVERSE);
    let radius_m = rows
        .iter()
        .find(|r| r.realm == realm)
        .and_then(|r| match r.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => Some(r),
            _ => None,
        })
        .expect("the home planet is round");
    let charter = vd_bins::body_charter(
        UNIVERSE,
        &BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]),
        &BTreeSet::from([vd_core::worldgen::GALAXY]),
        realm,
    )
    .expect("a charter");
    let body = BodyDefinition::from_seed(HOME, radius_m, vd_bins::facts_of_charter(&charter))
        .expect("a ladder");
    let lattice = body.macro_lattice().expect("a lattice");
    println!(
        "THE 8 km WORLD'S RELIEF: each land node's window of itself and its eight neighbours, land only"
    );
    for source in std::env::args().skip(1) {
        if source == "store" {
            let env = vd_bins::process_env();
            let store = vd_bins::open_realm_store(&env, realm)
                .expect("the store opens")
                .expect("VD_REALM_STORE names a file");
            let artifact = vd_bins::artifact_store::read_artifact(
                &store,
                vd_terrain::artifact_world_tag(UNIVERSE),
                &lattice,
            )
            .expect("the stored rows agree with their head")
            .expect("an artifact is stored");
            read(
                &format!("the game's world (digest {:x?})", artifact.digest()),
                &artifact,
                &lattice,
                radius_m,
            );
        } else {
            let artifact = vd_bins::artifact_file::read(&source).expect("the artifact file reads");
            read(
                &format!("{source} (digest {:x?})", artifact.digest()),
                &artifact,
                &lattice,
                radius_m,
            );
        }
    }
}
