//! ★ THE GOLDEN FIELDS' RECORDER (slice 8c stage C4c): solves the home planet once, takes the
//! artifact's rows under the six rung-0 self-check keys and its coarsest pyramid level, writes them
//! as `crates/terrain/tests/golden_home_z.txt`, and prints the identity's measured half to pin as
//! `vd_terrain::home::HOME_IDENTITY_MEASURED`. Run after a deliberate change of the solve, then
//! commit the file and the constant together with a `GENERATOR_VERSION` bump.
//!
//! `cargo run --release -p vd-bins --example golden_z_record`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_terrain::artifact::{GoldenFields, PyramidField, SparseZ, nodes_of_chunk};
use vd_terrain::digest::{GOLDEN_SELF_CHECK_KEYS, golden_self_check, self_check_key};
use vd_terrain::home::{home_planet, home_solve_words};
use vd_terrain::macro_lattice::MacroLattice;

fn main() {
    let body = home_planet();
    let lattice = MacroLattice::of(&body).expect("the home planet has a macro lattice");
    let started = std::time::Instant::now();
    let artifact = run_solve(&SolveJob {
        body,
        words: home_solve_words(),
    })
    .expect("the home planet solves");
    println!(
        "golden_z_record: solved in {:.1} s, digest {:?}",
        started.elapsed().as_secs_f64(),
        artifact.digest()
    );
    let body = home_planet();
    let levels = artifact.pyramid.len() as u32;
    let mut rows = SparseZ::default();
    let mut fine_keys = 0;
    for entry in GOLDEN_SELF_CHECK_KEYS {
        let key = self_check_key(&body, entry);
        if PyramidField::level_for(&lattice, levels, key.rung) != 0 {
            continue;
        }
        fine_keys += 1;
        for node in nodes_of_chunk(&lattice, key) {
            rows.0.insert(node, artifact.rows[node as usize].z_m);
        }
    }
    let top = PyramidField::of(&artifact, levels).expect("the coarsest level");
    let fields = GoldenFields { levels, rows, top };
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../terrain/tests/golden_home_z.txt"
    );
    std::fs::write(path, fields.to_text()).expect("the golden fields' file is written");
    let measured = golden_self_check(&body, Some(&fields)).expect("the home planet self-checks");
    println!(
        "golden_z_record: {} fine keys, {} rows, top level {} of {} words, written to {path}",
        fine_keys,
        fields.rows.0.len(),
        fields.top.level,
        fields.top.z_m.len()
    );
    println!(
        "golden_z_record: HOME_IDENTITY_MEASURED = {measured:#x}_u64 — commit it in vd_terrain::home"
    );
}
