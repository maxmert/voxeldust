//! ★ AN ARTIFACT AS A PLAIN FILE, FOR THE BENCHES (2026-09-26; the washboard bench, ruling B10 item 2).
//!
//! **WHY IT EXISTS.** The washboard bench (`router_bench`) makes a variant artifact for each routing
//! cure it measures, and a slow solve must not run twice for one reading. So one example writes an
//! artifact that another can read: `router_bench --artifact-readings` and `equator_stands` read it
//! today (the shoreline bench that painted it left with the drawn water, ruling B22). This is the one
//! reader and the one writer. It is a development aid: never a store, never on the wire, and the
//! digest a reader checks is the artifact's own.
//!
//! **THE FORMAT** (little-endian words): the edge and the version (`u32` each), the sea (`i16`), the
//! row count (`u64`) and every row's bytes; then the pyramid's heights and its water words (each a
//! level count, then per level a length and the `i16` words); then the coast mask's length and its
//! bytes.
//!
//! **Example.** The washboard bench solves the home planet with the seeded spread of the drain
//! direction and writes `rho8.art`; `router_bench --artifact-readings rho8.art rho8` reads it back
//! and prints the land's facing and its fine texture's facing, so the owner reads the 45° ridges
//! before and after the cure.

use vd_terrain::artifact::{Artifact, ROW_BYTES, Row};

/// Write `artifact` to `path` in the format above.
///
/// # Errors
/// The file system's own error when the file cannot be written.
pub fn write(path: &str, artifact: &Artifact) -> std::io::Result<()> {
    std::fs::write(path, encode(artifact))
}

/// The artifact read from `path`; `None` for no file, a short file or a malformed row.
#[must_use]
pub fn read(path: &str) -> Option<Artifact> {
    decode(&std::fs::read(path).ok()?)
}

/// The artifact's bytes in the format above.
#[must_use]
pub fn encode(a: &Artifact) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::new();
    for w in [a.edge, a.version] {
        b.extend_from_slice(&w.to_le_bytes());
    }
    b.extend_from_slice(&a.sea_m.to_le_bytes());
    let put_len = |b: &mut Vec<u8>, n: usize| b.extend_from_slice(&(n as u64).to_le_bytes());
    put_len(&mut b, a.rows.len());
    for r in &a.rows {
        b.extend_from_slice(&r.to_bytes());
    }
    for set in [&a.pyramid, &a.pyramid_water] {
        put_len(&mut b, set.len());
        for level in set {
            put_len(&mut b, level.len());
            for v in level {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    put_len(&mut b, a.coast.len());
    b.extend_from_slice(&a.coast);
    // ★ The lakes' record last (ruling B121; a file written before it ends at the coast and reads no lakes).
    let lakes = vd_terrain::artifact::encode_lake_records(&a.lakes);
    put_len(&mut b, lakes.len());
    b.extend_from_slice(&lakes);
    b
}

/// The artifact decoded from `b`; `None` where the bytes end early or a row does not decode.
#[must_use]
pub fn decode(b: &[u8]) -> Option<Artifact> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = b.get(at..at + n)?;
        at += n;
        Some(s)
    };
    let u32_of = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    let edge = u32_of(take(4)?);
    let version = u32_of(take(4)?);
    let sea = take(2)?;
    let sea_m = i16::from_le_bytes([sea[0], sea[1]]);
    let len = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap_or([0; 8])) as usize;
    let rows_n = len(take(8)?);
    let mut rows = Vec::with_capacity(rows_n);
    for _ in 0..rows_n {
        let r = take(ROW_BYTES)?;
        rows.push(Row::from_bytes(r.try_into().ok()?));
    }
    let mut sets: Vec<Vec<Vec<i16>>> = Vec::new();
    for _ in 0..2 {
        let levels = len(take(8)?);
        let mut set = Vec::with_capacity(levels);
        for _ in 0..levels {
            let n = len(take(8)?);
            let raw = take(2 * n)?;
            set.push(
                raw.chunks(2)
                    .map(|p| i16::from_le_bytes([p[0], p[1]]))
                    .collect(),
            );
        }
        sets.push(set);
    }
    let coast_n = len(take(8)?);
    let coast = take(coast_n)?.to_vec();
    // The lakes' record where the file holds one (written from ARTIFACT_VERSION 10 on), else none.
    let lakes = match take(8) {
        Some(n) => {
            let n = len(n);
            vd_terrain::artifact::decode_lake_records(take(n)?)?
        }
        None => Vec::new(),
    };
    let pyramid_water = sets.pop()?;
    let pyramid = sets.pop()?;
    Some(Artifact {
        edge,
        version,
        sea_m,
        rows,
        pyramid,
        pyramid_water,
        coast,
        lakes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Artifact {
        let mut row = Row::from_bytes([0u8; ROW_BYTES]);
        row.z_m = 1234;
        row.water_m = -7;
        Artifact {
            edge: 2,
            version: 9,
            sea_m: 3591,
            rows: vec![row; 24],
            pyramid: vec![vec![5, -6, 7]],
            pyramid_water: vec![vec![i16::MIN, 8, 9]],
            coast: vec![0b1010_0101, 3],
            lakes: vec![vd_terrain::artifact::LakeRecord {
                node: 5,
                age_kyr: 12_600,
                laid_q: 164,
            }],
        }
    }

    /// ★ The round trip: an artifact written and read back is the same artifact, digest and all.
    ///
    /// **Example.** The washboard bench writes a variant; the shoreline bench reads the very same
    /// bytes, so the picture shows the variant and nothing else.
    #[test]
    fn an_artifact_reads_back_as_itself() {
        let a = small();
        let back = decode(&encode(&a)).expect("the bytes decode");
        assert_eq!(back.digest(), a.digest());
        assert_eq!(back.rows.len(), a.rows.len());
        assert_eq!(back.pyramid, a.pyramid);
        assert_eq!(back.pyramid_water, a.pyramid_water);
        assert_eq!(back.coast, a.coast);
        assert_eq!(back.lakes, a.lakes);
        assert_eq!(back.sea_m, a.sea_m);
    }

    /// A file that ends early is refused, never read as a shorter artifact.
    #[test]
    fn a_short_file_is_refused() {
        let bytes = encode(&small());
        assert_eq!(decode(&bytes[..bytes.len() - 1]).map(|a| a.edge), None);
        assert_eq!(decode(&[]).map(|a| a.edge), None);
    }
}
