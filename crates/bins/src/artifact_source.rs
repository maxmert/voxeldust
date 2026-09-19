//! ★ THE ARTIFACT AS A TILE SOURCE (the landform arc, slice 8c stage C4c): the composition root's
//! answer to the sim's [`TileSource`] seam — the artifact the shard holds, encoded into the wire's
//! bulk shapes on demand. The sim ships bytes; this is where the bytes come from.
//!
//! - the HEAD: one `BulkMsg::ArtifactHead` (the world tag, the version, the edge, the digest);
//! - the PYRAMID: every level cut into parts of at most [`PYRAMID_PART_WORDS`] heights, each a
//!   `BulkMsg::ArtifactPyramid`, coarsest level LAST so a client draws from the top down;
//! - the TILES under a direction: the face and the macro cell the direction falls in through the
//!   bend's inverse, then every tile whose rows lie within the radius on that face (a stated
//!   limit: tiles across a cube edge from the occupant are owed — the next face's tiles arrive as
//!   the occupant crosses onto it).
//!
//! **Example.** A pilot stands on the home planet at a coast. Her direction falls on face +Y at
//! macro cell (700, 300); with an interest radius of 200 km (24 nodes) the source names the tiles
//! (10..=11, 4..=5) of that face, and the shard ships them four a tick.

use std::sync::Arc;

use vd_seed::bend::{Face, face_of, unbend};
use vd_seed::ladder::index_of;
use vd_sim::io::Bytes;
use vd_sim::stub::artifact_ship::TileSource;
use vd_terrain::artifact::{Artifact, TILE_EDGE};
use vd_terrain::macro_lattice::MacroLattice;
use vd_wire::channels::BulkMsg;

use crate::artifact_store::tile_bytes;

/// A pyramid part's size in heights: 16 384 words is 32 KB, a tile's weight.
pub const PYRAMID_PART_WORDS: usize = 16_384;

/// The source over one realm's artifact.
pub struct ArtifactTiles {
    pub realm: vd_core::pose::RealmId,
    pub world_tag: u64,
    pub artifact: Arc<Artifact>,
    pub lattice: MacroLattice,
}

impl ArtifactTiles {
    /// The tile that holds macro cell `(i, j)` of a face.
    #[must_use]
    pub fn tile_of(i: u32, j: u32) -> (u32, u32) {
        (i / TILE_EDGE, j / TILE_EDGE)
    }
}

/// ★ THE SOURCE THE SHARD INSTALLS once its artifact is here: over the realm's own body, or
/// `None` for a body too small for a macro lattice (it ships no artifact).
#[must_use]
pub fn artifact_source_of(
    realm: vd_core::pose::RealmId,
    world_tag: u64,
    body: &vd_terrain::BodyDefinition,
    artifact: &Arc<Artifact>,
) -> Option<Box<dyn TileSource>> {
    let lattice = body.macro_lattice()?;
    Some(Box::new(ArtifactTiles {
        realm,
        world_tag,
        artifact: Arc::clone(artifact),
        lattice,
    }))
}

fn encode(msg: &BulkMsg) -> Bytes {
    vd_sim::io::bytes(postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly"))
}

impl TileSource for ArtifactTiles {
    fn head(&self) -> Bytes {
        encode(&BulkMsg::ArtifactHead {
            realm: self.realm,
            world_tag: self.world_tag,
            version: self.artifact.version,
            edge: self.artifact.edge,
            digest: self.artifact.digest(),
            tiles_per_edge: self.artifact.tiles_per_edge(),
            levels: self.artifact.pyramid.len() as u32,
        })
    }

    fn pyramid_parts(&self) -> Vec<Bytes> {
        let mut out = Vec::new();
        for (k, level) in self.artifact.pyramid.iter().enumerate().rev() {
            let parts = level.len().div_ceil(PYRAMID_PART_WORDS).max(1) as u32;
            for (part, words) in level.chunks(PYRAMID_PART_WORDS).enumerate() {
                out.push(encode(&BulkMsg::ArtifactPyramid {
                    realm: self.realm,
                    level: k as u32 + 1,
                    part: part as u32,
                    parts,
                    z_m: words.to_vec(),
                }));
            }
        }
        out
    }

    fn tiles_under(&self, dir: [f64; 3], radius_m: f64) -> Vec<(u8, u32, u32)> {
        let face = face_of(dir);
        let basis = vd_seed::bend::BASIS[face.index() as usize];
        let axis = |a: [i8; 3]| {
            dir[0] * f64::from(a[0]) + dir[1] * f64::from(a[1]) + dir[2] * f64::from(a[2])
        };
        let n = axis(basis.n);
        if n <= 0.0 {
            return Vec::new();
        }
        let (a, b) = (unbend(axis(basis.u) / n), unbend(axis(basis.v) / n));
        let edge = self.lattice.edge;
        let (i, j) = (index_of(a, edge), index_of(b, edge));
        let reach_nodes = (radius_m / self.lattice.node_m()).ceil().max(1.0) as i64;
        let lo = |c: i32| (i64::from(c) - reach_nodes).max(0) as u32 / TILE_EDGE;
        let hi =
            |c: i32| ((i64::from(c) + reach_nodes).min(i64::from(edge) - 1)) as u32 / TILE_EDGE;
        let mut out = Vec::new();
        for ty in lo(j)..=hi(j) {
            for tx in lo(i)..=hi(i) {
                out.push((face.index(), tx, ty));
            }
        }
        // The nearest first: by the square of the tile-centre distance to the occupant's cell.
        let (ci, cj) = (i64::from(i), i64::from(j));
        out.sort_by_key(|&(_, tx, ty)| {
            let (mx, my) = (
                i64::from(tx * TILE_EDGE + TILE_EDGE / 2),
                i64::from(ty * TILE_EDGE + TILE_EDGE / 2),
            );
            (mx - ci) * (mx - ci) + (my - cj) * (my - cj)
        });
        out
    }

    fn tile(&self, face: u8, tx: u32, ty: u32) -> Option<Bytes> {
        let face = Face::from_index(face)?;
        let per = self.artifact.tiles_per_edge();
        if tx >= per || ty >= per {
            return None;
        }
        let tile = self.artifact.tile(face, tx, ty);
        Some(encode(&BulkMsg::ArtifactTile {
            realm: self.realm,
            face: face.index(),
            tx,
            ty,
            rows: tile_bytes(&tile),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact_worker::{SolveJob, run_solve};
    use vd_terrain::home::{home_moon, home_moon_solve_words};

    /// The moon as a source: the head names the artifact, the pyramid comes coarsest first in
    /// parts, the tiles under a direction are that face's tiles nearest first and every one
    /// decodes back to its rows, and a tile past the lattice is none.
    #[test]
    fn the_moons_artifact_as_a_source() {
        let moon = home_moon();
        let artifact = Arc::new(
            run_solve(&SolveJob {
                body: moon,
                words: home_moon_solve_words(),
            })
            .expect("the moon solves"),
        );
        let source = ArtifactTiles {
            realm: vd_core::pose::RealmId::Planet(moon.seed()),
            world_tag: 5,
            artifact: artifact.clone(),
            lattice: moon.macro_lattice().expect("a lattice"),
        };
        let head = postcard::from_bytes::<BulkMsg>(&source.head()).expect("decodes");
        assert_eq!(
            head,
            BulkMsg::ArtifactHead {
                realm: vd_core::pose::RealmId::Planet(moon.seed()),
                world_tag: 5,
                version: artifact.version,
                edge: 68,
                digest: artifact.digest(),
                tiles_per_edge: 2,
                levels: 2,
            }
        );
        let parts = source.pyramid_parts();
        // Level 2 (6 · 17² = 1 734 words) is one part; level 1 (6 · 34² = 6 936) is one part.
        assert_eq!(parts.len(), 2);
        match postcard::from_bytes::<BulkMsg>(&parts[0]).expect("decodes") {
            BulkMsg::ArtifactPyramid {
                level,
                part,
                parts,
                z_m,
                ..
            } => {
                assert_eq!((level, part, parts), (2, 0, 1));
                assert_eq!(z_m, artifact.pyramid[1]);
            }
            other => panic!("{other:?}"),
        }
        // Under +Z at the face centre (node 34 of 68), 300 km is 37 nodes (8.15 km each): the
        // four tiles of face +Z, the nearest first; 100 km (13 nodes) stays inside tile 0.
        assert_eq!(source.tiles_under([0.0, 0.0, 1.0], 100_000.0).len(), 1);
        let under = source.tiles_under([0.0, 0.0, 1.0], 300_000.0);
        assert_eq!(under.len(), 4);
        assert!(under.iter().all(|&(f, _, _)| f == Face::PosZ.index()));
        assert_eq!(under[0], (Face::PosZ.index(), 0, 0));
        // A small radius near the corner of the face keeps one tile.
        let corner = source.tiles_under([0.0, 0.0, 1.0], 10.0);
        assert_eq!(corner.len(), 1);
        // A direction under the face (the far side) names nothing on this face's basis... it names
        // the face it points at, never this one: −Z is its own face.
        let far = source.tiles_under([0.0, 0.0, -1.0], 10.0);
        assert_eq!(far[0].0, Face::NegZ.index());
        let tile = source.tile(Face::PosZ.index(), 1, 1).expect("a tile");
        match postcard::from_bytes::<BulkMsg>(&tile).expect("decodes") {
            BulkMsg::ArtifactTile {
                face, tx, ty, rows, ..
            } => {
                assert_eq!((face, tx, ty), (Face::PosZ.index(), 1, 1));
                assert_eq!(rows.len(), 4 * 4 * vd_terrain::artifact::ROW_BYTES);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(source.tile(Face::PosZ.index(), 2, 0), None);
        assert_eq!(source.tile(9, 0, 0), None);
        assert_eq!(ArtifactTiles::tile_of(130, 5), (2, 0));
        // The shard's helper: a source over the moon; none over a body with no macro lattice.
        let installed = artifact_source_of(
            vd_core::pose::RealmId::Planet(moon.seed()),
            5,
            &moon,
            &artifact,
        )
        .expect("a source");
        assert_eq!(installed.head(), source.head());
        let pebble = vd_terrain::BodyDefinition::from_seed(
            moon.seed(),
            1_000.0,
            vd_terrain::BodyFacts::new(1, 1),
        );
        if let Some(pebble) = pebble {
            assert!(
                pebble.macro_lattice().is_some()
                    || artifact_source_of(vd_core::pose::RealmId::Planet(1), 5, &pebble, &artifact)
                        .is_none()
            );
        }
    }
}
