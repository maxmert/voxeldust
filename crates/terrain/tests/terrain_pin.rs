//! ★ THE GOLDEN GATE (the voxel foundation, slice 5; SL10 clause 3; ruling V9 S5-4): the digests of
//! seventy-two columns of THE world's home planet at EVERY rung — the surface chunk of each column and
//! the chunks two below and two above it, so the skipped fills are pinned as well as the evaluated
//! ones — byte for byte, against a committed table. Red on one differing byte.
//!
//! The same test runs in debug and in release, with and without the chip's native flags (`just
//! terrain-pin`, legs G1 and G2), and on every other target the game ships on (the x86-64 leg, G4:
//! emulation as a smoke test, real hardware before "no drift" is called satisfied). A differing table
//! on any leg is the drift SL10 exists to stop.
//!
//! **The home body is two literals** (`vd_terrain::home`) — its seed and the exact bits of its look
//! radius — and `crates/bins/tests/home_body_pin.rs` proves the forest still produces exactly those,
//! so this crate needs no motion crate and the two never drift apart (SL5: one world).
//!
//! **The gate has a control**: `a_tampered_table_is_refused` changes one committed digest and asserts
//! the comparison goes red, so a comparison that could pass vacuously is itself caught.
//!
//! **To record a new table** (only after a deliberate recipe change, with `GENERATOR_VERSION` bumped;
//! before the first saved world a recipe change needs no bump, and the table is re-recorded beside it):
//! `VD_TERRAIN_RECORD=1 cargo test -p vd-terrain --test terrain_pin`, then commit
//! `tests/golden_home.txt`.

use vd_seed::bend::Face;
use vd_terrain::chunk::{CHUNK_EDGE, column_field, generate_in};
use vd_terrain::digest::{digest_of, surface_chunk_z};
use vd_terrain::home::home_planet;

/// The committed table: one line per digest, `face rung x y z d0 d1`.
const GOLDEN: &str = include_str!("golden_home.txt");

/// Columns per face: ten spread by two odd strides, plus the face's corner column and its far edge
/// column — the seams the bend must agree on.
const COLUMNS_PER_FACE: usize = 12;
/// The rows per column: the surface chunk and the chunks two below and two above it.
const ROWS_PER_COLUMN: usize = 3;

/// Seventy-two columns at a rung, distinct: the strides start at one so no stride key repeats the
/// corner.
fn column_keys(chunks: i32) -> Vec<(Face, i32, i32)> {
    let mut keys = Vec::new();
    for (f, face) in Face::ALL.iter().enumerate() {
        let f = f as i32;
        let mut i = 1;
        while i <= 10 {
            let x = (i * 7_919 + f * 104_729) % chunks;
            let y = (i * 15_485_863 + f * 32_452_843) % chunks;
            keys.push((*face, x, y));
            i += 1;
        }
        keys.push((*face, 0, 0));
        keys.push((*face, chunks - 1, 0));
    }
    keys
}

fn table() -> Vec<String> {
    let body = home_planet();
    let mut lines = Vec::new();
    let mut rung = 0u8;
    while rung < body.ladder().rungs {
        let chunks = (body.ladder().cells_per_edge(rung) as i32 / CHUNK_EDGE as i32).max(1);
        let top = (body.ladder().cells_in_band(rung) as i32 - 1) / CHUNK_EDGE as i32;
        let keys = column_keys(chunks);
        let distinct: std::collections::BTreeSet<(Face, i32, i32)> = keys.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            keys.len(),
            "rung {rung}: every column is distinct"
        );
        for (face, x, y) in keys {
            let column = column_field(&body, face, rung, x, y).expect("a home planet column");
            let zs = surface_chunk_z(&body, face, rung, x, y);
            for z in [(zs - 2).max(0), zs, (zs + 2).min(top)] {
                let chunk = generate_in(&body, &column, z).expect("a home planet chunk");
                let d = digest_of(&chunk);
                lines.push(format!(
                    "{} {rung} {x} {y} {z} {:016x} {:016x}",
                    face as u8, d.0[0], d.0[1]
                ));
            }
        }
        rung += 1;
    }
    lines
}

/// The comparison: every computed line against the committed one, in order; `Err` names the first
/// differing lines.
fn compare(committed: &str, computed: &[String]) -> Result<(), String> {
    let committed: Vec<&str> = committed.lines().filter(|l| !l.is_empty()).collect();
    if committed.len() != computed.len() {
        return Err(format!(
            "the table's size moved: {} committed, {} computed",
            committed.len(),
            computed.len()
        ));
    }
    let differing: Vec<String> = committed
        .iter()
        .zip(computed.iter())
        .filter(|(c, l)| **c != l.as_str())
        .map(|(c, l)| format!("committed `{c}` computed `{l}`"))
        .collect();
    if differing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} of {} digests differ:\n{}",
            differing.len(),
            computed.len(),
            differing.join("\n")
        ))
    }
}

#[test]
fn every_golden_chunk_of_the_home_planet_digests_to_its_committed_bytes() {
    let lines = table();
    if std::env::var_os("VD_TERRAIN_RECORD").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden_home.txt");
        std::fs::write(path, lines.join("\n") + "\n").expect("the table is written");
        println!("terrain_pin: RECORDED {} digests to {path}", lines.len());
        return;
    }
    let body = home_planet();
    assert_eq!(body.ladder().rungs, 12, "the home planet has twelve rungs");
    assert_eq!(
        lines.len(),
        6 * COLUMNS_PER_FACE * ROWS_PER_COLUMN * 12,
        "seventy-two columns, three rows each, at every rung"
    );
    if let Err(report) = compare(GOLDEN, &lines) {
        panic!(
            "THE GOLDEN GATE IS RED — the recipe moved a byte (bump GENERATOR_VERSION and re-record \
             on purpose) or this build drifted: {report}"
        );
    }
}

/// The control: a comparison that could not go red would pin nothing.
#[test]
fn a_tampered_table_is_refused() {
    let lines = vec![
        "0 0 0 0 1 aaaa bbbb".to_owned(),
        "0 0 1 0 1 cccc dddd".to_owned(),
    ];
    assert_eq!(
        compare("0 0 0 0 1 aaaa bbbb\n0 0 1 0 1 cccc dddd\n", &lines),
        Ok(())
    );
    let short = compare("0 0 0 0 1 aaaa bbbb\n", &lines).expect_err("a short table is refused");
    assert!(short.contains("size moved"), "{short}");
    let moved = compare("0 0 0 0 1 aaaa bbbb\n0 0 1 0 1 cccc dddx\n", &lines)
        .expect_err("one differing digest is refused");
    assert!(moved.contains("1 of 2 digests differ"), "{moved}");
}
