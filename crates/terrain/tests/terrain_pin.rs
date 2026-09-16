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

/// Columns per face: ten spread by two odd strides, the face's corner column, its last FULL edge
/// column, and the three LAST chunks (partial on the home planet) that hold the +u and +v seams and
/// the (+u, +v) corner — the seams the bend must agree on. A COARSE RUNG HOLDS FEWER COLUMNS THAN
/// THAT and gets every column it has (the top rung of the home planet is one column a face).
const COLUMNS_PER_FACE: usize = 15;
/// The home planet's own line count, PINNED: fifteen columns a face at the thirteen finest rungs and
/// every column the rung has above them (89, 82, 51, 36, 24 and 6 columns at rungs 13 to 18), three
/// rows each. A silent shrink of the table is a red gate.
const GOLDEN_LINES: usize = 1_458 * ROWS_PER_COLUMN;
/// The rows per column: the surface chunk and the chunks two below and two above it.
const ROWS_PER_COLUMN: usize = 3;

/// Seventy-two columns at a rung, distinct: the strides start at one so no stride key repeats the
/// corner.
fn column_keys(chunks: i32, last: i32) -> Vec<(Face, i32, i32)> {
    let mut keys = Vec::new();
    for (f, face) in Face::ALL.iter().enumerate() {
        let f = f as i32;
        let mut candidates = Vec::new();
        let mut i = 1;
        while i <= 10 {
            let x = (i * 7_919 + f * 104_729) % chunks;
            let y = (i * 15_485_863 + f * 32_452_843) % chunks;
            candidates.push((x, y));
            i += 1;
        }
        candidates.push((0, 0));
        candidates.push((chunks - 1, 0));
        // The LAST chunk along each edge - partial on the home planet (a face edge is not a whole
        // number of chunks) - and the (+u, +v) corner: the partner's cells beyond the face, both
        // plus seams and the corner prism.
        candidates.push((last, 0));
        candidates.push((0, last));
        candidates.push((last, last));
        // A COARSE RUNG HAS FEWER COLUMNS THAN THE LIST ASKS FOR (the extended ladder, 2026-09-15):
        // the top rung is ONE CHUNK a face edge, so a face holds one column and the strides all name
        // it. Take each column ONCE and take every one the rung has.
        let mut seen = std::collections::BTreeSet::new();
        for (x, y) in candidates {
            if seen.insert((x, y)) {
                keys.push((*face, x, y));
            }
        }
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
        let last = (body.ladder().cells_per_edge(rung) as i32 - 1) / CHUNK_EDGE as i32;
        let keys = column_keys(chunks, last);
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
    assert_eq!(body.ladder().rungs, 19, "the home planet's ladder");
    assert_eq!(
        lines.len(),
        GOLDEN_LINES,
        "every column the rung has, up to ninety a rung, three rows each"
    );
    assert_eq!(
        column_keys(160_668, 160_668).len(),
        6 * COLUMNS_PER_FACE,
        "a rung with columns to spare still takes fifteen a face"
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
