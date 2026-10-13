//! ★ THE REALM'S GROUND, AS ITS SHARD HOLDS IT (ruling B81 Q6, built 2026-10-13): the [`vd_sim::stub::stand::Ground`] a
//! body's shard injects, so the realm stands its parked children on the ground every client draws there — the finished
//! 128 m tiles the shard holds (`fine_source`'s own book), and the coarser ground (the artifact's rows and their octaves)
//! where it holds none. The read is the one the client's chunks make (`vd_terrain::height::height_field_m` at rung 0),
//! so a hull stands on the surface the player sees under it.
//!
//! **Example.** Hull 15's berth over the home planet: at boot the shard holds no fine tile, so the hull is stood on the
//! 8 km ground; when the pilot's view names the tile under it and the shard reads it back from the store, the tile joins
//! the book, its patch changes, and the hull follows the fine ground up over half a second.

use std::sync::Arc;

use vd_core::glam::DVec3;
use vd_seed::bend::{face_coords, face_of, unbend};
use vd_seed::ladder::index_of;
use vd_terrain::BodyDefinition;
use vd_terrain::artifact::{Artifact, ZField};
use vd_terrain::chunk::CHUNK_EDGE;
use vd_terrain::fine::l1::TileKey;
use vd_terrain::fine::read::FineField;
use vd_terrain::height::height_field_m;

use crate::fine_source::GroundBook;

/// ★ ONE BODY'S GROUND: its artifact, and the fine provider's book where the 128 m level applies.
pub struct ShardGround {
    body: BodyDefinition,
    artifact: Arc<Artifact>,
    fine: Option<GroundBook>,
}

impl ShardGround {
    /// The ground of `body` over its `artifact`, with the fine provider's book where the level applies (`None` on a dry
    /// body: its artifact's ground alone).
    #[must_use]
    pub fn new(
        body: BodyDefinition,
        artifact: Arc<Artifact>,
        fine: Option<GroundBook>,
    ) -> ShardGround {
        ShardGround {
            body,
            artifact,
            fine,
        }
    }

    /// The rung-0 cell under a direction: its face and its indices along the face's two axes.
    fn cell0(&self, dir: [f64; 3]) -> (vd_seed::bend::Face, i32, i32) {
        let face = face_of(dir);
        let (t, s) = face_coords(face, dir);
        let n0 = self.body.ladder().cells_per_edge(0);
        (face, index_of(unbend(t), n0), index_of(unbend(s), n0))
    }
}

/// A tile as an opaque patch id: its face, then its two indices.
fn patch_id(key: TileKey) -> u64 {
    (u64::from(key.face.index()) << 58)
        | ((u64::from(key.tx as u32) & 0x1FFF_FFFF) << 29)
        | (u64::from(key.ty as u32) & 0x1FFF_FFFF)
}

impl vd_sim::stub::stand::Ground for ShardGround {
    fn radius_m(&self, dir: DVec3) -> Option<f64> {
        let d = dir.to_array();
        if let Some(fine) = &self.fine {
            let field = FineField {
                inner: Arc::clone(&self.artifact) as Arc<dyn ZField + Send + Sync>,
                book: fine.book(),
            };
            // The fine ground where its tiles are held; a missing tile answers nothing, and the coarser ground stands.
            if let Some(r) = height_field_m(&self.body, &field, d, 0) {
                return Some(r);
            }
        }
        height_field_m(&self.body, &*self.artifact, d, 0)
    }

    fn patches(&self, dir: DVec3) -> Vec<u64> {
        let Some(fine) = &self.fine else {
            return Vec::new();
        };
        let (face, i, j) = self.cell0(dir.to_array());
        let edge = CHUNK_EDGE as i32;
        fine.book()
            .tiles_of_chunk(face, 0, i.div_euclid(edge), j.div_euclid(edge))
            .into_iter()
            .map(patch_id)
            .collect()
    }

    fn changed(&self) -> Vec<u64> {
        self.fine
            .as_ref()
            .map(|fine| fine.take_changed().into_iter().map(patch_id).collect())
            .unwrap_or_default()
    }
}
