//! ★ THE TRIANGLE GATE (the voxel foundation, slice 6; ruling V10 S6-9): the digests of the
//! EXTRACTED surface of the same seventy-two columns of THE world's home planet at EVERY rung — the
//! surface chunk of each column — byte for byte against a committed table, plus a COMPOSED row (the
//! generated shape with one mined cell, one placed dirt cell, one block and one sub-metre block,
//! digested after composition and extraction) and a SEATING row (the seat of a sub-metre block on a
//! slope). A quad list could not go red on the diagonal that matters; a triangle list does.
//!
//! The same test runs on every leg the cell table runs on (`just terrain-pin`, `just terrain-legs`).
//!
//! **To record a new table** (only after a deliberate extractor change, with `GENERATOR_VERSION`
//! bumped once a world is saved): `VD_TERRAIN_RECORD=1 cargo test -p vd-terrain --test mesh_pin`,
//! then commit `tests/golden_home_mesh.txt`.

use vd_seed::bend::Face;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::compose::{EditRow, compose};
use vd_terrain::digest::{mesh_digest, surface_chunk_z};
use vd_terrain::extract::extract;
use vd_terrain::home::home_planet;
use vd_terrain::lattice::sample_box;
use vd_terrain::seat::seat_eighths;
use vd_terrain::strata::Stratum;

/// The committed table: one line per row, `face rung x y z d0 d1`, then the composed row and the
/// seating row.
const GOLDEN: &str = include_str!("golden_home_mesh.txt");

const COLUMNS_PER_FACE: usize = 15;

/// The same seventy-two columns the cell table pins.
fn column_keys(chunks: i32, last: i32) -> Vec<(Face, i32, i32)> {
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
        // The LAST chunk along each edge — partial on the home planet (a face edge is not a whole
        // number of chunks) — and the (+u, +v) corner: the partner's cells beyond the face, both
        // plus seams and the corner prism.
        keys.push((*face, last, 0));
        keys.push((*face, 0, last));
        keys.push((*face, last, last));
    }
    keys
}

/// The composed fixture: the surface chunk of face +X, column (300, 700), rung 0, with one mined
/// cell, one placed dirt cell, one block and one sub-metre block, in list order that the
/// composition must re-rank.
fn composed_key() -> ChunkKey {
    let body = home_planet();
    ChunkKey {
        face: Face::PosX,
        rung: 0,
        x: 300,
        y: 700,
        z: surface_chunk_z(&body, Face::PosX, 0, 300, 700),
    }
}

fn composed_rows() -> [EditRow; 5] {
    [
        EditRow::SubMetre { cell: [20, 20, 30] },
        EditRow::TerrainCell {
            cell: [10, 10, 30],
            code: Stratum::Empty.code(),
            gap: i8::MAX,
        },
        EditRow::Block { cell: [15, 15, 31] },
        EditRow::TerrainCell {
            cell: [12, 10, 31],
            code: Stratum::Dirt.code(),
            gap: i8::MIN,
        },
        EditRow::Attachment,
    ]
}

fn table() -> Vec<String> {
    let body = home_planet();
    let mut lines = Vec::new();
    let mut rung = 0u8;
    while rung < body.ladder().rungs {
        let chunks = (body.ladder().cells_per_edge(rung) as i32 / CHUNK_EDGE as i32).max(1);
        let last = (body.ladder().cells_per_edge(rung) as i32 - 1) / CHUNK_EDGE as i32;
        for (face, x, y) in column_keys(chunks, last) {
            let z = surface_chunk_z(&body, face, rung, x, y);
            let key = ChunkKey {
                face,
                rung,
                x,
                y,
                z,
            };
            let samples = sample_box(&body, key).expect("a home planet chunk");
            let d = mesh_digest(&extract(&samples));
            lines.push(format!(
                "{} {rung} {x} {y} {z} {:016x} {:016x}",
                face as u8, d.0[0], d.0[1]
            ));
        }
        rung += 1;
    }
    // The composed row.
    let key = composed_key();
    let mut samples = sample_box(&body, key).expect("the composed chunk");
    assert_eq!(compose(&mut samples, &composed_rows()), 0);
    let d = mesh_digest(&extract(&samples));
    lines.push(format!(
        "composed {} {} {} {} {} {:016x} {:016x}",
        key.face as u8, key.rung, key.x, key.y, key.z, d.0[0], d.0[1]
    ));
    // The seating row: the seats of every cell of one column of the composed chunk, as a string of
    // eighths (`-` for a refusal), from the floor up.
    let mut seats = String::new();
    let mut c = 0u8;
    while c < CHUNK_EDGE as u8 {
        match seat_eighths(&samples, [20, 20, c]) {
            Some(e) => seats.push(char::from(b'0' + e)),
            None => seats.push('-'),
        }
        c += 1;
    }
    lines.push(format!("seats 20 20 {seats}"));
    lines
}

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
            "{} of {} rows differ:\n{}",
            differing.len(),
            computed.len(),
            differing.join("\n")
        ))
    }
}

#[test]
fn every_golden_surface_of_the_home_planet_digests_to_its_committed_triangles() {
    let lines = table();
    if std::env::var_os("VD_TERRAIN_RECORD").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden_home_mesh.txt");
        std::fs::write(path, lines.join("\n") + "\n").expect("the table is written");
        println!("mesh_pin: RECORDED {} rows to {path}", lines.len());
        return;
    }
    assert_eq!(
        lines.len(),
        6 * COLUMNS_PER_FACE * 12 + 2,
        "seventy-two surfaces at every rung, the composed row and the seating row"
    );
    if let Err(report) = compare(GOLDEN, &lines) {
        panic!(
            "THE TRIANGLE GATE IS RED — the extractor, the composition or the seating moved a byte \
             (bump GENERATOR_VERSION and re-record on purpose) or this build drifted: {report}"
        );
    }
}

/// The composed row is not the plain row: the edits move the digest, and the seating row shows
/// the mined cell's seat.
#[test]
fn the_composed_surface_differs_from_the_plain_one_and_the_rows_are_re_ranked() {
    let body = home_planet();
    let key = composed_key();
    let plain = sample_box(&body, key).expect("the chunk");
    let mut composed = plain.clone();
    assert_eq!(compose(&mut composed, &composed_rows()), 0);
    assert_ne!(
        mesh_digest(&extract(&plain)),
        mesh_digest(&extract(&composed))
    );
    // The rows in a different list order compose to the same bytes: the rank orders them.
    let mut rows = composed_rows();
    rows.reverse();
    let mut again = plain.clone();
    assert_eq!(compose(&mut again, &rows), 0);
    assert_eq!(again, composed);
}

#[test]
fn a_tampered_mesh_table_is_refused() {
    let lines = vec![
        "0 0 0 0 1 aaaa bbbb".to_owned(),
        "seats 20 20 000".to_owned(),
    ];
    assert_eq!(
        compare("0 0 0 0 1 aaaa bbbb\nseats 20 20 000\n", &lines),
        Ok(())
    );
    let short = compare("0 0 0 0 1 aaaa bbbb\n", &lines).expect_err("a short table is refused");
    assert!(short.contains("size moved"), "{short}");
    let moved = compare("0 0 0 0 1 aaaa bbbb\nseats 20 20 001\n", &lines)
        .expect_err("one differing row is refused");
    assert!(moved.contains("1 of 2 rows differ"), "{moved}");
}
