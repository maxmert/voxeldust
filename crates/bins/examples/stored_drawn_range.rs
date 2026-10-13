//! Diagnosis probe (2026-10-13): read a planet's STORED artifact and compute what its shard computes the tick the solve
//! lands and at every later boot — the drawn range over the rows and every pyramid level — plus one Catmull-Rom read
//! of the coarsest level at every node of it. Planet 17313999383825979787's shard panicked here (its pyramid folds down
//! to one node per face) until `node_at` walked on across the partner face.
//! `VD_REALM_STORE=<file> VD_PLANET=<seed> cargo run --release -p vd-bins --example stored_drawn_range`

use std::collections::BTreeSet;

use vd_core::pose::RealmId;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::node_at;

const UNIVERSE: u64 = vd_bins::DEV.universe_seed;

fn main() {
    let seed: u64 = std::env::var("VD_PLANET")
        .expect("VD_PLANET names the planet's seed")
        .parse()
        .expect("a seed");
    let realm = RealmId::Planet(seed);
    let (rows, _) = vd_bins::home_system_world(UNIVERSE);
    let radius_m = rows
        .iter()
        .find(|r| r.realm == realm)
        .and_then(|r| match r.look {
            Some(vd_core::geometry::Boundary::Shell { r }) => Some(r),
            _ => None,
        })
        .expect("a round planet of the home system");
    let charter = vd_bins::body_charter(
        UNIVERSE,
        &BTreeSet::from([vd_core::worldgen::HOME_SYSTEM]),
        &BTreeSet::from([vd_core::worldgen::GALAXY]),
        realm,
    )
    .expect("a charter");
    let body = BodyDefinition::from_seed(seed, radius_m, vd_bins::facts_of_charter(&charter))
        .expect("a ladder");
    let lattice = body.macro_lattice().expect("a lattice");
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
    let levels = artifact.pyramid.len() as u32;
    let top = lattice.coarser(levels).expect("the top level is a halving");
    println!(
        "planet {seed}: edge {}, {levels} pyramid levels, top edge {}, digest {:?}",
        lattice.edge,
        top.edge,
        artifact.digest()
    );
    let (lo, hi) = artifact.drawn_range_m(&lattice);
    println!("drawn range {lo} m .. {hi} m");
    // Every node of the top level, each with its sixteen-node stencil through the seam rule.
    let top_z = &artifact.pyramid[levels as usize - 1];
    let mut read = 0usize;
    for node in 0..top.node_count() as u32 {
        let (face, i, j) = top.split(node);
        for row in -1..=2i64 {
            for col in -1..=2i64 {
                let n = node_at(&top, face, i64::from(i) + col, i64::from(j) + row);
                assert!((n as usize) < top_z.len(), "node {n} of {}", top_z.len());
                read += 1;
            }
        }
    }
    println!(
        "the top level's stencils read {read} nodes, every one of its {} nodes real",
        top_z.len()
    );
}
