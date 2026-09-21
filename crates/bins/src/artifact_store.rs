//! ★ THE ARTIFACT IN THE SHARD'S STORE (the landform arc, slice 8c stage C4; the owner's ruling
//! of 2026-09-19: the solve runs once on the server and the artifact is saved in the shard's db).
//!
//! The composition root is the one place that links both the generator (which knows what an
//! artifact is) and the store's row families (which the sim frames as bytes), so the bridge lives
//! here: an [`Artifact`] becomes one HEAD row, one row per TILE, one PYRAMID INDEX row and one row
//! per PYRAMID PART through the [`Store`] seam, and comes back the same way — REFUSED by name when
//! the head names another world tag, another version or another lattice, or when a tile or a part
//! is missing: a store that holds half a planet is not a planet.
//!
//! The pyramid is cut into parts of [`PYRAMID_PART_WORDS`] heights, the wire's own cut, because a
//! store row is one TLV field and a field caps at one mebibyte: the home planet's pyramid is 5.9 MB
//! (MEASURED in the far-view ship's first flight, 2026-09-20, when the one-row pyramid of artifact
//! version 2 panicked every big planet's shard as its solve landed).
//!
//! **Example.** The home planet's shard boots on a fresh pod. It opens its store, reads the head,
//! checks the world tag it was solved under against its own, reads 2 166 tiles and the pyramid's
//! 181 parts, and states its surface within the boot — no solve.

use std::collections::BTreeSet;

use vd_sim::io::Store;
use vd_sim::stub::built_store::{
    ArtifactHead, ArtifactPyramidIndex, ArtifactPyramidPart, ArtifactTile, PYRAMID_PART_WORDS,
    artifact_head_key, artifact_pyramid_key, artifact_pyramid_part_key,
    artifact_pyramid_part_prefix, artifact_tile_key, artifact_tile_prefix, decode_artifact_head,
    decode_artifact_pyramid_index, decode_artifact_pyramid_part, decode_artifact_tile,
    encode_artifact_head, encode_artifact_pyramid_index, encode_artifact_pyramid_part,
    encode_artifact_tile,
};
use vd_terrain::artifact::{Artifact, Row, Tile, TileCache, tile_width};
use vd_terrain::macro_lattice::MacroLattice;

/// The head an artifact writes, under `world_tag`.
#[must_use]
pub fn head_of(artifact: &Artifact, world_tag: u64) -> ArtifactHead {
    ArtifactHead {
        world_tag,
        version: artifact.version,
        edge: artifact.edge,
        digest: artifact.digest(),
        tiles_per_edge: artifact.tiles_per_edge(),
        sea_m: artifact.sea_m,
    }
}

/// A tile's rows as the store's bytes, nine a row — the wire's framing too (`Tile::to_bytes`).
#[must_use]
pub fn tile_bytes(tile: &Tile) -> Vec<u8> {
    tile.to_bytes()
}

/// A tile's rows back from the store's bytes; `None` for a length that is not whole rows.
#[must_use]
pub fn rows_of(bytes: &[u8]) -> Option<Vec<Row>> {
    Tile::rows_from_bytes(bytes)
}

/// A pyramid level's heights as little-endian bytes.
#[must_use]
pub fn level_bytes(level: &[i16]) -> Vec<u8> {
    level.iter().flat_map(|z| z.to_le_bytes()).collect()
}

/// A pyramid level's heights back; `None` for an odd length.
#[must_use]
pub fn level_of(bytes: &[u8]) -> Option<Vec<i16>> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

/// ★ WRITE the artifact: the head, every tile of every face, the pyramid's parts and its index —
/// staged, then ONE commit. A part row of an earlier solve that this one does not rewrite is
/// dropped in the same commit, so the store never holds a level and a half.
pub fn write_artifact(store: &mut dyn Store, world_tag: u64, artifact: &Artifact) {
    let head = head_of(artifact, world_tag);
    store.put(&artifact_head_key(), &encode_artifact_head(&head).into());
    let per = artifact.tiles_per_edge();
    for face in vd_seed::bend::Face::ALL {
        for ty in 0..per {
            for tx in 0..per {
                let tile = artifact.tile(face, tx, ty);
                let row = ArtifactTile {
                    face: face.index(),
                    tx,
                    ty,
                    bytes: tile_bytes(&tile),
                };
                store.put(
                    &artifact_tile_key(face.index(), tx, ty),
                    &encode_artifact_tile(&row).into(),
                );
            }
        }
    }
    let mut parts_per_level = Vec::with_capacity(artifact.pyramid.len());
    let mut written = BTreeSet::new();
    for (k, level) in artifact.pyramid.iter().enumerate() {
        let level_no = k as u32 + 1;
        let bytes = level_bytes(level);
        let mut parts = 0u32;
        for (part, chunk) in bytes.chunks(PYRAMID_PART_WORDS * 2).enumerate() {
            let key = artifact_pyramid_part_key(level_no, part as u32);
            let row = ArtifactPyramidPart {
                level: level_no,
                part: part as u32,
                bytes: chunk.to_vec(),
            };
            store.put(&key, &encode_artifact_pyramid_part(&row).into());
            written.insert(key);
            parts += 1;
        }
        parts_per_level.push(parts);
    }
    for (key, _) in store.scan(&artifact_pyramid_part_prefix()) {
        if !written.contains(&key) {
            store.delete(&key);
        }
    }
    let index = ArtifactPyramidIndex {
        part_words: PYRAMID_PART_WORDS as u32,
        parts_per_level,
    };
    store.put(
        &artifact_pyramid_key(),
        &encode_artifact_pyramid_index(&index).into(),
    );
    store.commit();
}

/// ★ READ the artifact back under `world_tag` for a lattice of `edge`: `Ok(None)` with no head
/// (never solved here), the artifact when every row is present and agrees, or a REFUSAL by name.
///
/// # Errors
/// The head names another world tag, version or edge; a tile is missing or malformed; the
/// pyramid's index or a part is missing, malformed or under the wrong key; the digest of what was
/// read is not the head's.
pub fn read_artifact(
    store: &dyn Store,
    world_tag: u64,
    lattice: &MacroLattice,
) -> Result<Option<Artifact>, String> {
    let Some(head_bytes) = store.get(&artifact_head_key()) else {
        return Ok(None);
    };
    let head = decode_artifact_head(&head_bytes)?;
    if head.world_tag != world_tag {
        return Err(format!(
            "the stored artifact was solved under world tag {:#x}, this build is {:#x}",
            head.world_tag, world_tag
        ));
    }
    if head.version != vd_terrain::artifact::ARTIFACT_VERSION {
        return Err(format!(
            "the stored artifact is version {}, this build reads {}",
            head.version,
            vd_terrain::artifact::ARTIFACT_VERSION
        ));
    }
    if head.edge != lattice.edge {
        return Err(format!(
            "the stored artifact has {} nodes an edge, the body's lattice {}",
            head.edge, lattice.edge
        ));
    }
    let mut cache = TileCache::new(head.edge);
    for (_, bytes) in store.scan(&artifact_tile_prefix()) {
        let tile = decode_artifact_tile(&bytes)?;
        let rows = rows_of(&tile.bytes).ok_or("a stored tile's bytes are not whole rows")?;
        let want = (tile_width(head.edge, tile.tx) * tile_width(head.edge, tile.ty)) as usize;
        if rows.len() != want {
            return Err(format!(
                "a stored tile at ({}, {}, {}) holds {} rows, not {want}",
                tile.face,
                tile.tx,
                tile.ty,
                rows.len()
            ));
        }
        cache.apply(Tile {
            face: tile.face,
            tx: tile.tx,
            ty: tile.ty,
            rows,
        });
    }
    let n = lattice.node_count();
    let mut rows = Vec::with_capacity(n);
    for node in 0..n as u32 {
        rows.push(
            cache
                .row(node)
                .ok_or_else(|| format!("the stored artifact is missing the tile of node {node}"))?,
        );
    }
    let index_bytes = store
        .get(&artifact_pyramid_key())
        .ok_or("the stored artifact has no pyramid")?;
    let index = decode_artifact_pyramid_index(&index_bytes)?;
    let mut pyramid = Vec::with_capacity(index.parts_per_level.len());
    for (k, &parts) in index.parts_per_level.iter().enumerate() {
        let level_no = k as u32 + 1;
        let mut bytes = Vec::with_capacity(parts as usize * index.part_words as usize * 2);
        for part in 0..parts {
            let row = store
                .get(&artifact_pyramid_part_key(level_no, part))
                .ok_or_else(|| {
                    format!(
                        "the stored artifact is missing part {part} of pyramid level {level_no}"
                    )
                })?;
            let row = decode_artifact_pyramid_part(&row)?;
            if (row.level, row.part) != (level_no, part) {
                return Err(format!(
                    "a stored pyramid part says level {} part {}, its key says level {level_no} part {part}",
                    row.level, row.part
                ));
            }
            bytes.extend_from_slice(&row.bytes);
        }
        pyramid.push(level_of(&bytes).ok_or("a stored pyramid level is not whole words")?);
    }
    let artifact = Artifact {
        edge: head.edge,
        version: head.version,
        sea_m: head.sea_m,
        rows,
        pyramid,
    };
    if artifact.digest() != head.digest {
        return Err("the stored artifact's rows do not digest to its head".to_owned());
    }
    Ok(Some(artifact))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_sim::io::mem::MemStore;
    use vd_terrain::home::{HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words};
    use vd_terrain::solve::{Schedule, solve_full};

    fn moon_artifact() -> (MacroLattice, Artifact) {
        let moon = home_moon();
        let lattice = MacroLattice::of(&moon).expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            solve_full(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_z));
        (
            lattice,
            Artifact::of(&state, &facies, &climate, words.water_km3 > 0),
        )
    }

    /// The moon's artifact round-trips through an in-memory store byte for byte; an empty store
    /// reads as none; a head under another world tag, version or edge is refused by name; a
    /// missing tile, a missing pyramid, a missing or misplaced part and a tampered row are
    /// refused; a level over the field cap is cut into part rows, and a stale part row is dropped
    /// by the next write.
    #[test]
    fn the_artifact_round_trips_through_the_store_and_refusals_are_named() {
        let (lattice, artifact) = moon_artifact();
        let mut store = MemStore::default();
        assert_eq!(read_artifact(&store, 7, &lattice), Ok(None));
        write_artifact(&mut store, 7, &artifact);
        assert_eq!(
            read_artifact(&store, 7, &lattice),
            Ok(Some(artifact.clone()))
        );
        assert!(
            read_artifact(&store, 8, &lattice)
                .expect_err("refused")
                .contains("world tag")
        );
        let coarser = lattice.coarser(1).expect("a level");
        assert!(
            read_artifact(&store, 7, &coarser)
                .expect_err("refused")
                .contains("nodes an edge")
        );
        // A missing tile.
        let mut short = MemStore::default();
        write_artifact(&mut short, 7, &artifact);
        short.delete(&artifact_tile_key(2, 1, 1));
        short.commit();
        assert!(
            read_artifact(&short, 7, &lattice)
                .expect_err("refused")
                .contains("missing the tile")
        );
        // No pyramid.
        let mut flat = MemStore::default();
        write_artifact(&mut flat, 7, &artifact);
        flat.delete(&artifact_pyramid_key());
        flat.commit();
        assert!(
            read_artifact(&flat, 7, &lattice)
                .expect_err("refused")
                .contains("no pyramid")
        );
        // A tampered row: the digest refuses.
        let mut bad = MemStore::default();
        write_artifact(&mut bad, 7, &artifact);
        let key = artifact_tile_key(0, 0, 0);
        let mut tile = decode_artifact_tile(&bad.get(&key).expect("a tile")).expect("decodes");
        let tile_ok = tile.clone();
        tile.bytes[0] ^= 1;
        bad.put(&key, &encode_artifact_tile(&tile).into());
        bad.commit();
        assert!(
            read_artifact(&bad, 7, &lattice)
                .expect_err("refused")
                .contains("digest")
        );
        // A tile of the wrong length, a tile of whole rows but too few, a tile that does not
        // decode, a malformed head, a version from another build.
        let mut odd = MemStore::default();
        write_artifact(&mut odd, 7, &artifact);
        tile.bytes.push(0);
        odd.put(&key, &encode_artifact_tile(&tile).into());
        odd.commit();
        assert!(
            read_artifact(&odd, 7, &lattice)
                .expect_err("refused")
                .contains("whole rows")
        );
        tile.bytes.truncate(vd_terrain::artifact::ROW_BYTES);
        odd.put(&key, &encode_artifact_tile(&tile).into());
        odd.commit();
        assert!(
            read_artifact(&odd, 7, &lattice)
                .expect_err("refused")
                .contains("holds 1 rows")
        );
        odd.put(&key, &b"garbage".to_vec().into());
        odd.commit();
        assert!(
            read_artifact(&odd, 7, &lattice)
                .expect_err("refused")
                .contains("tile does not decode")
        );
        odd.put(&key, &encode_artifact_tile(&tile_ok).into());
        odd.commit();
        let mut head = head_of(&artifact, 7);
        head.version += 1;
        odd.put(&artifact_head_key(), &encode_artifact_head(&head).into());
        odd.commit();
        assert!(
            read_artifact(&odd, 7, &lattice)
                .expect_err("refused")
                .contains("version")
        );
        odd.put(&artifact_head_key(), &b"garbage".to_vec().into());
        odd.commit();
        assert!(read_artifact(&odd, 7, &lattice).is_err());
        // The helpers' refusals.
        assert_eq!(rows_of(&[1, 2, 3]), None);
        assert_eq!(level_of(&[1]), None);
        assert_eq!(level_of(&[1, 0, 255, 255]), Some(vec![1, -1]));
        // A missing part.
        let mut torn = MemStore::default();
        write_artifact(&mut torn, 7, &artifact);
        let key = artifact_pyramid_part_key(1, 0);
        let mut part =
            decode_artifact_pyramid_part(&torn.get(&key).expect("a part")).expect("decodes");
        torn.delete(&key);
        torn.commit();
        assert!(
            read_artifact(&torn, 7, &lattice)
                .expect_err("refused")
                .contains("missing part 0 of pyramid level 1")
        );
        // A part under the wrong key, then a part of odd length.
        part.level = 2;
        torn.put(&key, &encode_artifact_pyramid_part(&part).into());
        torn.commit();
        assert!(
            read_artifact(&torn, 7, &lattice)
                .expect_err("refused")
                .contains("says level 2 part 0")
        );
        part.level = 1;
        part.bytes.push(0);
        torn.put(&key, &encode_artifact_pyramid_part(&part).into());
        torn.commit();
        assert!(
            read_artifact(&torn, 7, &lattice)
                .expect_err("refused")
                .contains("whole words")
        );
        torn.put(&key, &b"garbage".to_vec().into());
        torn.commit();
        assert!(
            read_artifact(&torn, 7, &lattice)
                .expect_err("refused")
                .contains("pyramid part does not decode")
        );
        torn.put(&artifact_pyramid_key(), &b"garbage".to_vec().into());
        torn.commit();
        assert!(
            read_artifact(&torn, 7, &lattice)
                .expect_err("refused")
                .contains("pyramid index does not decode")
        );
    }

    /// ★ A level over the field cap: 600 000 heights is 1.2 MB, over the one-mebibyte TLV field
    /// that the one-row pyramid of artifact version 2 broke on the home planet. It is cut into 37
    /// part rows and round-trips; the next write of a smaller pyramid drops the parts it does not
    /// rewrite, and the store reads whole again.
    #[test]
    fn a_level_over_the_field_cap_is_cut_into_parts_and_stale_parts_are_dropped() {
        let (lattice, artifact) = moon_artifact();
        let mut big = artifact.clone();
        big.pyramid[0] = (0..600_000).map(|i| (i % 1_000) as i16).collect();
        let mut wide = MemStore::default();
        write_artifact(&mut wide, 7, &big);
        let index =
            decode_artifact_pyramid_index(&wide.get(&artifact_pyramid_key()).expect("an index"))
                .expect("decodes");
        assert_eq!(index.part_words as usize, PYRAMID_PART_WORDS);
        assert_eq!(index.parts_per_level[0], 37);
        let parts_total: u32 = index.parts_per_level.iter().sum();
        assert_eq!(
            wide.scan(&artifact_pyramid_part_prefix()).len(),
            parts_total as usize
        );
        assert_eq!(read_artifact(&wide, 7, &lattice), Ok(Some(big)));
        // The moon's own pyramid written over it: level 1 shrinks to one part, the other 36 go.
        write_artifact(&mut wide, 7, &artifact);
        let index =
            decode_artifact_pyramid_index(&wide.get(&artifact_pyramid_key()).expect("an index"))
                .expect("decodes");
        assert_eq!(index.parts_per_level[0], 1);
        let parts_total: u32 = index.parts_per_level.iter().sum();
        assert_eq!(
            wide.scan(&artifact_pyramid_part_prefix()).len(),
            parts_total as usize
        );
        assert_eq!(read_artifact(&wide, 7, &lattice), Ok(Some(artifact)));
    }
}
