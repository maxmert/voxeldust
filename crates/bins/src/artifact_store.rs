//! ★ THE ARTIFACT IN THE SHARD'S STORE (the landform arc, slice 8c stage C4; the owner's ruling
//! of 2026-09-19: the solve runs once on the server and the artifact is saved in the shard's db).
//!
//! The composition root is the one place that links both the generator (which knows what an
//! artifact is) and the store's row families (which the sim frames as bytes), so the bridge lives
//! here: an [`Artifact`] becomes one HEAD row, one row per TILE and one PYRAMID row through the
//! [`Store`] seam, and comes back the same way — REFUSED by name when the head names another world
//! tag, another version or another lattice, or when a tile is missing: a store that holds half a
//! planet is not a planet.
//!
//! **Example.** The home planet's shard boots on a fresh pod. It opens its store, reads the head,
//! checks the world tag it was solved under against its own, reads 2 166 tiles and the pyramid,
//! and states its surface within the boot — no solve.

use vd_sim::io::Store;
use vd_sim::stub::built_store::{
    ArtifactHead, ArtifactTile, artifact_head_key, artifact_pyramid_key, artifact_tile_key,
    artifact_tile_prefix, decode_artifact_head, decode_artifact_pyramid, decode_artifact_tile,
    encode_artifact_head, encode_artifact_pyramid, encode_artifact_tile,
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

/// ★ WRITE the artifact: the head, every tile of every face, the pyramid — staged, then ONE commit.
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
    let levels: Vec<Vec<u8>> = artifact.pyramid.iter().map(|l| level_bytes(l)).collect();
    store.put(
        &artifact_pyramid_key(),
        &encode_artifact_pyramid(&levels).into(),
    );
    store.commit();
}

/// ★ READ the artifact back under `world_tag` for a lattice of `edge`: `Ok(None)` with no head
/// (never solved here), the artifact when every row is present and agrees, or a REFUSAL by name.
///
/// # Errors
/// The head names another world tag, version or edge; a tile is missing or malformed; the
/// pyramid is missing or malformed; the digest of what was read is not the head's.
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
    let pyramid_bytes = store
        .get(&artifact_pyramid_key())
        .ok_or("the stored artifact has no pyramid")?;
    let levels = decode_artifact_pyramid(&pyramid_bytes)?;
    let mut pyramid = Vec::with_capacity(levels.len());
    for l in &levels {
        pyramid.push(level_of(l).ok_or("a stored pyramid level is not whole words")?);
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
    /// missing tile, a missing pyramid and a tampered row are refused.
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
        tile.bytes[0] ^= 1;
        bad.put(&key, &encode_artifact_tile(&tile).into());
        bad.commit();
        assert!(
            read_artifact(&bad, 7, &lattice)
                .expect_err("refused")
                .contains("digest")
        );
        // A tile of the wrong length, a malformed head, a version from another build.
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
    }
}
