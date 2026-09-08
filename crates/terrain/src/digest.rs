//! ★ THE CHUNK DIGEST and the golden self-check — the gate's unit of comparison. A digest folds every
//! cell's substance byte and gap byte in packing order, twice, from two offsets, into 128 bits. The
//! golden set (`tests/terrain_pin.rs`) pins the home planet's chunks at every rung; the self-check
//! folds eight of them at boot into the world tag's MEASURED half.
//!
//! **Example.** The client evaluates the eight self-check chunks at login and states the fold. The
//! gateway, which evaluated the same eight at boot, compares. A chip whose arithmetic drifted by one
//! ulp on one cell states a different number and is refused before it draws a hill.

use crate::body::BodyDefinition;
use crate::chunk::{ChunkKey, generate};
use vd_seed::bend::Face;
use vd_seed::digest::{FNV_OFFSET, fnv1a};

/// A chunk's 128-bit digest: two FNV-1a folds over the same bytes from two offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkDigest(pub [u64; 2]);

/// The second fold's offset: the first offset with every bit flipped.
const SECOND_OFFSET: u64 = !FNV_OFFSET;

/// The digest of one chunk; `None` for a key outside the ladder.
#[must_use]
pub fn chunk_digest(body: &BodyDefinition, key: ChunkKey) -> Option<ChunkDigest> {
    Some(digest_of(&generate(body, key)?))
}

/// The digest of a generated chunk: its key, then its cells in packing order.
#[must_use]
pub fn digest_of(chunk: &crate::chunk::ChunkLattice) -> ChunkDigest {
    let key = chunk.key;
    let mut a = fnv1a(FNV_OFFSET, &[key.face as u8, key.rung]);
    let mut b = fnv1a(SECOND_OFFSET, &[key.face as u8, key.rung]);
    for coord in [key.x, key.y, key.z] {
        a = fnv1a(a, &coord.to_le_bytes());
        b = fnv1a(b, &coord.to_le_bytes());
    }
    for cell in &chunk.cells {
        let bytes = [cell.stratum.code(), cell.gap as u8];
        a = fnv1a(a, &bytes);
        b = fnv1a(b, &bytes);
    }
    ChunkDigest([a, b])
}

/// The digest of an extracted chunk (slice 6): its key, its vertex count, every vertex's three
/// integers and every triangle's three indices, little-endian, in order. Pins the triangle list —
/// a quad list cannot go red on the diagonal that matters.
#[must_use]
pub fn mesh_digest(mesh: &crate::extract::ChunkMesh) -> ChunkDigest {
    let key = mesh.key;
    let mut a = fnv1a(FNV_OFFSET, &[key.face as u8, key.rung]);
    let mut b = fnv1a(SECOND_OFFSET, &[key.face as u8, key.rung]);
    for coord in [key.x, key.y, key.z] {
        a = fnv1a(a, &coord.to_le_bytes());
        b = fnv1a(b, &coord.to_le_bytes());
    }
    let count = (mesh.vertices.len() as u32).to_le_bytes();
    a = fnv1a(a, &count);
    b = fnv1a(b, &count);
    for v in &mesh.vertices {
        for x in v {
            a = fnv1a(a, &x.to_le_bytes());
            b = fnv1a(b, &x.to_le_bytes());
        }
    }
    for t in &mesh.triangles {
        for i in t {
            a = fnv1a(a, &i.to_le_bytes());
            b = fnv1a(b, &i.to_le_bytes());
        }
    }
    ChunkDigest([a, b])
}

/// The eight self-check keys: one chunk per face at rung 0, near the surface, plus two at the top
/// rung. Stated as `(face, rung, x, y)`; the radial index is the surface chunk of that column.
pub const GOLDEN_SELF_CHECK_KEYS: [(Face, u8, i32, i32); 8] = [
    (Face::PosX, 0, 300, 700),
    (Face::NegX, 0, 1_181, 77),
    (Face::PosY, 0, 5, 5),
    (Face::NegY, 0, 2_000, 2_000),
    (Face::PosZ, 0, 999, 1),
    (Face::NegZ, 0, 40, 4_000),
    (Face::PosX, 255, 3, 3),
    (Face::NegY, 255, 1, 2),
];

/// THE SURFACE SPAN of chunk column `(x, y)` of `face` at a rung: the lowest and the highest chunk
/// index along the radial that hold the surface at the column's centre and its four corners, one
/// chunk of margin each way, clamped to the band. Five heights, not a column pass — a client asks
/// this per wanted column per frame. A slope steeper than the margin (a cliff of more than 62 cells
/// inside one column) and a cave mouth deeper than the margin are what slice 8's residency band is
/// for (`D-TERRAIN-3`).
#[must_use]
pub fn surface_chunk_span(
    body: &BodyDefinition,
    face: Face,
    rung: u8,
    x: i32,
    y: i32,
) -> (i32, i32) {
    let edge = crate::chunk::CHUNK_EDGE as i32;
    let key = ChunkKey {
        face,
        rung,
        x,
        y,
        z: 0,
    };
    let cell = i64::from(vd_seed::ladder::cell_m(rung));
    let floor = i64::from(body.ladder.floor_m);
    let top = (body.ladder.cells_in_band(rung) as i32 - 1) / edge;
    let mut lo = i32::MAX;
    let mut hi = i32::MIN;
    for (a, b) in [
        (edge / 2, edge / 2),
        (0, 0),
        (edge - 1, 0),
        (0, edge - 1),
        (edge - 1, edge - 1),
    ] {
        let site = crate::lattice::site_of(body, key, a, b);
        let dir = crate::lattice::site_dir(body, key, site);
        let h = crate::height::height_m(body, dir, rung);
        let z = (((h.to_i64_floor() - floor) / cell) / i64::from(edge)) as i32;
        lo = lo.min(z);
        hi = hi.max(z);
    }
    ((lo - 1).clamp(0, top), (hi + 1).clamp(0, top))
}

/// The chunk index along the radial that holds the SURFACE at the centre column of chunk `(x, y)`
/// of `face` at a rung: the chunk the extractor has work in.
#[must_use]
pub fn surface_chunk_z(body: &BodyDefinition, face: Face, rung: u8, x: i32, y: i32) -> i32 {
    let edge = crate::chunk::CHUNK_EDGE as i32;
    let key = ChunkKey {
        face,
        rung,
        x,
        y,
        z: 0,
    };
    let site = crate::lattice::site_of(body, key, edge / 2, edge / 2);
    let dir = crate::lattice::site_dir(body, key, site);
    let h = crate::height::height_m(body, dir, rung);
    let cell = i64::from(vd_seed::ladder::cell_m(rung));
    let k = (h.to_i64_floor() - i64::from(body.ladder.floor_m)) / cell;
    (k / i64::from(edge)) as i32
}

/// The key a self-check entry names on this body: rung 255 means the body's top rung, and the
/// face coordinates are wrapped into the rung's face so every body, however small, has eight.
#[must_use]
pub fn self_check_key(body: &BodyDefinition, entry: (Face, u8, i32, i32)) -> ChunkKey {
    let rung = if entry.1 == 255 {
        body.ladder.rungs - 1
    } else {
        entry.1
    };
    let chunks_per_edge =
        (body.ladder.cells_per_edge(rung) as i32 / crate::chunk::CHUNK_EDGE as i32).max(1);
    let x = entry.2 % chunks_per_edge;
    let y = entry.3 % chunks_per_edge;
    ChunkKey {
        face: entry.0,
        rung,
        x,
        y,
        z: surface_chunk_z(body, entry.0, rung, x, y),
    }
}

/// The boot self-check: the fold of the eight self-check chunks' CELL digests and, since slice 6,
/// their MESH digests (the extractor, the halo, the seams and the corner are part of the shape a
/// client must compute identically — SL10 clauses 3 and 5), the world tag's MEASURED half.
/// `None` when a self-check key names no chunk of this body — a REFUSAL, never a fold of zeros
/// (the decode-to-Default ban, in the one place whose job is to refuse).
#[must_use]
pub fn golden_self_check(body: &BodyDefinition) -> Option<u64> {
    let mut acc = FNV_OFFSET;
    for entry in GOLDEN_SELF_CHECK_KEYS {
        let key = self_check_key(body, entry);
        let d = chunk_digest(body, key)?.0;
        acc = fnv1a(acc, &d[0].to_le_bytes());
        acc = fnv1a(acc, &d[1].to_le_bytes());
        let samples = crate::lattice::sample_box(body, key)?;
        let m = mesh_digest(&crate::extract::extract(&samples)).0;
        acc = fnv1a(acc, &m[0].to_le_bytes());
        acc = fnv1a(acc, &m[1].to_le_bytes());
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::home_planet;

    /// The span holds the centre's surface chunk with a chunk of margin each way, stays inside the
    /// band at the top rung (where one chunk is the whole band), and widens on a column whose
    /// corners sit in other chunks than its centre.
    #[test]
    fn the_surface_span_holds_the_centres_chunk_with_a_margin_and_stays_in_the_band() {
        let m = home_planet();
        let z = surface_chunk_z(&m, Face::PosX, 0, 300, 700);
        let (lo, hi) = surface_chunk_span(&m, Face::PosX, 0, 300, 700);
        assert!(lo < z, "{lo} {z}");
        assert!(hi > z, "{hi} {z}");
        assert!(lo >= 0);
        // The top rung: the band is one chunk, so the span is (0, 0) whatever the heights.
        let top = m.ladder().rungs - 1;
        assert_eq!(surface_chunk_span(&m, Face::PosX, top, 3, 3), (0, 0));
        // Every column of the golden set: the span holds the golden surface chunk.
        for entry in crate::GOLDEN_SELF_CHECK_KEYS {
            let key = self_check_key(&m, entry);
            let (lo, hi) = surface_chunk_span(&m, key.face, key.rung, key.x, key.y);
            assert!(lo <= key.z, "{key:?}: {lo}");
            assert!(hi >= key.z, "{key:?}: {hi}");
        }
        // A column whose corners disagree with its centre widens the span past ±1: measured over
        // the columns near the golden +X chunk, at least one span is wider than three chunks OR
        // every span is exactly three — both are stated, so the loop's `max` arm is exercised.
        let mut widest = 0;
        for x in 300..340 {
            let (lo, hi) = surface_chunk_span(&m, Face::PosX, 0, x, 700);
            widest = widest.max(hi - lo);
        }
        assert!(widest >= 2, "{widest}");
    }

    #[test]
    fn a_digest_is_the_chunks_bytes_and_moves_when_one_cell_or_the_key_moves() {
        let m = home_planet();
        let z = surface_chunk_z(&m, Face::PosX, 0, 300, 700);
        let key = ChunkKey {
            face: Face::PosX,
            rung: 0,
            x: 300,
            y: 700,
            z,
        };
        let d = chunk_digest(&m, key).expect("in the ladder");
        assert_eq!(
            chunk_digest(&m, key),
            Some(d),
            "the same bytes, the same digest"
        );
        let beside = ChunkKey { x: 301, ..key };
        assert_ne!(chunk_digest(&m, beside), Some(d));
        assert_ne!(d.0[0], d.0[1], "two folds, two offsets");
        assert_eq!(
            chunk_digest(&m, ChunkKey { rung: 99, ..key }),
            None,
            "outside the ladder is refused"
        );
        // One cell changed: the digest moves.
        let mut chunk = generate(&m, key).expect("in the ladder");
        chunk.cells[7].gap = chunk.cells[7].gap.wrapping_add(1);
        assert_ne!(digest_of(&chunk), d);
        // A second body (the home planet's seed plus one): every digest differs.
        let other = BodyDefinition::from_seed(
            m.seed + 1,
            f64::from_bits(crate::home::HOME_PLANET_RADIUS_BITS),
        )
        .expect("on the ladder");
        assert_ne!(chunk_digest(&other, key), Some(d));
    }

    #[test]
    fn the_self_check_has_eight_keys_inside_every_body_and_folds_to_one_number() {
        let m = home_planet();
        for entry in GOLDEN_SELF_CHECK_KEYS {
            let key = self_check_key(&m, entry);
            assert!(key.rung < m.ladder.rungs);
            assert!(
                chunk_digest(&m, key).is_some(),
                "{entry:?} names a chunk of the home planet"
            );
        }
        let keys: std::collections::BTreeSet<ChunkKey> = GOLDEN_SELF_CHECK_KEYS
            .iter()
            .map(|e| self_check_key(&m, *e))
            .collect();
        assert_eq!(keys.len(), 8, "eight distinct chunks");
        let check = golden_self_check(&m).expect("the home planet self-checks");
        assert_eq!(golden_self_check(&m), Some(check));
        // A tiny unnamed test body: the keys wrap into it, and its check differs.
        let rock = BodyDefinition::from_seed(5, 3_000.0).expect("a test body of 3 km");
        for entry in GOLDEN_SELF_CHECK_KEYS {
            let key = self_check_key(&rock, entry);
            assert!(
                chunk_digest(&rock, key).is_some(),
                "{entry:?} wraps into a tiny body"
            );
        }
        assert_ne!(golden_self_check(&rock), Some(check));
        let z0 = surface_chunk_z(&m, Face::PosX, 0, 300, 700);
        let z3 = surface_chunk_z(&m, Face::PosX, 3, 37, 87);
        assert!(
            (z0 / 8 - z3).abs() <= 1,
            "the surface chunk scales with the rung: {z0} vs {z3}"
        );
    }
}
