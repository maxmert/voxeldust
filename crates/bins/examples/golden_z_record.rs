//! ★ THE GOLDEN FIELDS' RECORDER (slice 8c stage C4c): solves the home planet once, takes the
//! artifact's rows under the six rung-0 self-check keys and its coarsest pyramid level, writes them
//! as `crates/terrain/tests/golden_home_z.txt`, and prints the identity's measured half to pin as
//! `vd_terrain::home::HOME_IDENTITY_MEASURED`. Run after a deliberate change of the solve, then
//! commit the file and the constant together with a `GENERATOR_VERSION` bump.
//!
//! `cargo run --release -p vd-bins --example golden_z_record`

use vd_bins::artifact_worker::{SolveJob, run_solve};
use vd_terrain::artifact::{FACIES_SHIFT, GoldenFields, PyramidField, SparseRows, nodes_of_chunk};
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
    let mut rows = SparseRows::default();
    let mut fine_keys = 0;
    for entry in GOLDEN_SELF_CHECK_KEYS {
        let key = self_check_key(&body, entry);
        if PyramidField::level_for(&lattice, levels, key.rung) != 0 {
            continue;
        }
        fine_keys += 1;
        for node in nodes_of_chunk(&lattice, key) {
            let r = artifact.rows[node as usize];
            rows.0
                .insert(node, (r.z_m, r.water_m, r.receiver_facies >> FACIES_SHIFT));
        }
    }
    // ★ THE LEVEL THE TOP-RUNG KEYS READ (2026-09-20): the runtime's own pick for the top rung,
    // which is the coarsest level only when the top rung's cell reaches it (it does not on the
    // home planet: rung 18 reads level 5 of 6). A golden field the far view never reads would
    // prove nothing about the picture.
    let top_level = GOLDEN_SELF_CHECK_KEYS
        .iter()
        .map(|entry| PyramidField::level_for(&lattice, levels, self_check_key(&body, *entry).rung))
        .max()
        .expect("eight keys");
    let top = PyramidField::of(&artifact, top_level).expect("the top-rung keys' level");
    let fields = GoldenFields {
        levels,
        sea_m: artifact.sea_m,
        rows,
        top,
    };
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
    // ★ THE SEA AND THE OCEAN SHARE (stage C5, gate G-SEA), the same reading the artifact pin makes,
    // so one solve records every pin.
    let (mut wet, mut all) = (0.0f64, 0.0f64);
    for (node, row) in artifact.rows.iter().enumerate() {
        let area = lattice.area_m2(node as u32) as f64;
        all += area;
        if artifact.sea().is_some_and(|sea| i32::from(row.z_m) <= sea) {
            wet += area;
        }
    }
    println!(
        "golden_z_record: HOME_PLANET_SEA_M = {:?}; HOME_PLANET_OCEAN_SHARE_Q4 = {}; HOME_ARTIFACT_DIGEST = [{:#x}, {:#x}]",
        artifact.sea(),
        (wet / all * 10_000.0).round() as u32,
        artifact.digest()[0],
        artifact.digest()[1]
    );
}
