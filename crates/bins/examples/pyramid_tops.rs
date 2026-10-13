//! Diagnosis probe (2026-10-13, after Planet 17313999383825979787's first solve landed and its shard panicked in
//! `Artifact::drawn_range_m`): every round body of the home star system with its macro lattice's edge and the edge of the
//! COARSEST pyramid level its artifact folds (the halvings of the edge while they stay exact). A top edge of ONE is a level
//! of one node per face, where a sixteen-node stencil reaches past the whole partner face — beyond `node_at`'s stated reach
//! of two nodes outside a face. `cargo run --release -p vd-bins --example pyramid_tops`.

use std::collections::BTreeSet;

use vd_core::pose::RealmId;
use vd_terrain::BodyDefinition;

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;

fn main() {
    let system = vd_core::worldgen::HOME_SYSTEM;
    let galaxy = vd_core::worldgen::GALAXY;
    let (rows, _) = vd_bins::home_system_world(UNIVERSE);
    for row in &rows {
        let RealmId::Planet(seed) = row.realm else {
            continue;
        };
        let Some(vd_core::geometry::Boundary::Shell { r: radius_m }) = row.look else {
            continue;
        };
        let Some(charter) = vd_bins::body_charter(
            UNIVERSE,
            &BTreeSet::from([system]),
            &BTreeSet::from([galaxy]),
            row.realm,
        ) else {
            println!("planet {seed}: no charter");
            continue;
        };
        let Some(def) =
            BodyDefinition::from_seed(seed, radius_m, vd_bins::facts_of_charter(&charter))
        else {
            println!("planet {seed}: no ladder");
            continue;
        };
        let Some(lattice) = def.macro_lattice() else {
            println!("planet {seed}: no macro lattice (radius {radius_m:.0} m)");
            continue;
        };
        let mut levels = 0u32;
        let mut top = lattice.edge;
        while let Some(coarse) = lattice.coarser(levels + 1) {
            levels += 1;
            top = coarse.edge;
        }
        println!(
            "planet {seed}: radius {:.0} km, edge {}, pyramid levels {levels}, top edge {top}{}",
            radius_m / 1000.0,
            lattice.edge,
            if top == 1 {
                "  ← ONE NODE PER FACE"
            } else {
                ""
            }
        );
    }
}
