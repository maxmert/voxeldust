//! ★ THE ARTIFACT AS A TILE SOURCE (the landform arc, slice 8c stage C4c): the composition root's
//! answer to the sim's [`TileSource`] seam — the artifact the shard holds, encoded into the wire's
//! bulk shapes on demand. The sim ships bytes; this is where the bytes come from.
//!
//! - the HEAD: one `BulkMsg::ArtifactHead` (the world tag, the version, the edge, the digest);
//! - the PYRAMID: every level cut into parts of at most [`PYRAMID_PART_WORDS`] heights, each a
//!   `BulkMsg::ArtifactPyramid`, coarsest level LAST so a client draws from the top down;
//! - the COAST MASK (2026-09-22, ruling W10): the body's sea bits cut into parts of at most
//!   [`COAST_PART_BYTES`], each a `BulkMsg::ArtifactCoast`, AFTER the pyramid's parts in the one
//!   list the sim pages through — the far view stands first, then every rung reads one shoreline;
//! - the TILES under a direction: the face and the macro cell the direction falls in through the
//!   bend's inverse, then every tile whose rows lie within the radius on that face (a stated
//!   limit: tiles across a cube edge from the occupant are owed — the next face's tiles arrive as
//!   the occupant crosses onto it).
//!
//! **Example.** A pilot stands on the home planet at a coast. Her direction falls on face +Y at
//! macro cell (700, 300); with an interest radius of 200 km (24 nodes) the source names the tiles
//! (10..=11, 4..=5) of that face, and the shard ships them four a tick.

use std::sync::Arc;

use vd_seed::bend::Face;
use vd_sim::io::Bytes;
use vd_sim::stub::artifact_ship::TileSource;
use vd_terrain::artifact::{Artifact, TILE_EDGE};
use vd_terrain::macro_lattice::MacroLattice;
use vd_wire::channels::BulkMsg;

use crate::artifact_store::tile_bytes;

/// A coast part's size in BYTES: the store's own cut (32 KiB of bits), so a shard ships what it
/// stores.
pub use vd_sim::stub::built_store::COAST_PART_BYTES;
/// A pyramid part's size in heights: the store's own cut (16 384 words, 32 KB, a tile's weight),
/// so a shard ships what it stores.
pub use vd_sim::stub::built_store::PYRAMID_PART_WORDS;

/// The source over one realm's artifact.
pub struct ArtifactTiles {
    pub realm: vd_core::pose::RealmId,
    pub world_tag: u64,
    pub artifact: Arc<Artifact>,
    pub lattice: MacroLattice,
    /// The body's radius and its ladder's top rung, for the tile reach under an occupant's altitude.
    body_radius_m: f64,
    /// The tallest ground the recipe raises over the radius: the reach is bounded by how far the
    /// ground can be seen, peaks included (2026-09-21).
    relief_m: f64,
    top_rung: u8,
    /// ★ THE HANDOVER STEP OF THE FINEST TILE-READING RUNG (2026-09-23, ruling W17), taken ONCE:
    /// the octaves that rung's coarser neighbour drops plus the FIELD's own fold where the two
    /// read different pyramid levels. The tile ring is floored by it, so the shard ships the
    /// tiles out to the very distance the client's ladder asks at.
    step_m: f64,
    /// ★ THE DIGEST TAKEN ONCE (the second flight, 2026-09-20): the emitter asks for it every
    /// tick, and hashing the home planet's 85 MB of rows on every ask cost the shard 90 ms a tick
    /// (the water world 47 ms, the moon 0.3 ms — the cost of the artifact's size, five times the
    /// budget on every big planet from the tick its solve landed).
    digest: [u64; 2],
    /// ★ THE PARTS ENCODED ONCE (the same flight): the artifact never changes under a source, so
    /// its 181 pyramid parts (5.9 MB on the home planet) and its 34 coast parts (1.1 MB) are the
    /// same bytes on every ask. Filled on the first ask.
    parts: std::sync::OnceLock<Vec<Bytes>>,
}

impl ArtifactTiles {
    /// A source over `artifact`, the parts not yet encoded.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "seven of the eight are the realm's own address and the body's facts the reach \
                  reads; the eighth is the handover step the tile ring must cover (ruling W17), \
                  and each is a fact the caller already holds apart"
    )]
    pub fn new(
        realm: vd_core::pose::RealmId,
        world_tag: u64,
        artifact: Arc<Artifact>,
        lattice: MacroLattice,
        body_radius_m: f64,
        relief_m: f64,
        top_rung: u8,
        step_m: f64,
    ) -> ArtifactTiles {
        let digest = artifact.digest();
        ArtifactTiles {
            realm,
            world_tag,
            artifact,
            lattice,
            body_radius_m,
            relief_m,
            top_rung,
            step_m,
            digest,
            parts: std::sync::OnceLock::new(),
        }
    }

    /// The tiles whose nodes lie within `radius_m` of the unit direction `dir`, the nearest first:
    /// the terrain crate's own geometry, the gateway's too.
    fn tiles_within(&self, dir: [f64; 3], radius_m: f64) -> Vec<(u8, u32, u32)> {
        vd_terrain::artifact::tiles_within(&self.lattice, dir, radius_m)
    }

    fn encode_parts(&self) -> Vec<Bytes> {
        let mut out = Vec::new();
        for (k, level) in self.artifact.pyramid.iter().enumerate().rev() {
            let parts = level.len().div_ceil(PYRAMID_PART_WORDS).max(1) as u32;
            let water = &self.artifact.pyramid_water[k];
            for (part, (words, water_words)) in level
                .chunks(PYRAMID_PART_WORDS)
                .zip(water.chunks(PYRAMID_PART_WORDS))
                .enumerate()
            {
                out.push(encode(&BulkMsg::ArtifactPyramid {
                    realm: self.realm,
                    level: k as u32 + 1,
                    part: part as u32,
                    parts,
                    z_m: words.to_vec(),
                    water_m: water_words.to_vec(),
                }));
            }
        }
        // ★ THE COAST MASK, AFTER THE PYRAMID (2026-09-22, ruling W10): the same list, so the sim
        // pages through one queue and the gateway caches one arrival order. A client that holds
        // the mask reads the water's side at every rung from the fine row's own bit.
        let parts = self.coast_parts();
        for (part, bits) in self.artifact.coast.chunks(COAST_PART_BYTES).enumerate() {
            out.push(encode(&BulkMsg::ArtifactCoast {
                realm: self.realm,
                part: part as u32,
                parts,
                bits: bits.to_vec(),
            }));
        }
        out
    }

    /// How many coast parts this artifact's mask is cut into; ZERO for a body with no mask at all
    /// (a lattice of no nodes), which no shipped body is.
    #[must_use]
    pub fn coast_parts(&self) -> u32 {
        self.artifact.coast.len().div_ceil(COAST_PART_BYTES) as u32
    }
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
    // ★ THE STEP THE TILE RING IS FLOORED BY (ruling W17), taken once with the source.
    let levels = artifact.pyramid.len() as u32;
    let top_rung = body.ladder().rungs - 1;
    let rung = vd_terrain::artifact::tile_rung(&lattice, levels, top_rung);
    let step_m =
        body.step_bound_m(rung) + vd_terrain::artifact::field_step_m(body, &lattice, levels, rung);
    Some(Box::new(ArtifactTiles::new(
        realm,
        world_tag,
        Arc::clone(artifact),
        lattice,
        body.radius_m(),
        body.relief_bound_m(0),
        top_rung,
        step_m,
    )))
}

fn encode(msg: &BulkMsg) -> Bytes {
    vd_sim::io::bytes(postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly"))
}

impl TileSource for ArtifactTiles {
    fn digest(&self) -> [u64; 2] {
        self.digest
    }

    fn head(&self) -> Bytes {
        encode(&BulkMsg::ArtifactHead {
            realm: self.realm,
            world_tag: self.world_tag,
            version: self.artifact.version,
            edge: self.artifact.edge,
            digest: self.digest,
            tiles_per_edge: self.artifact.tiles_per_edge(),
            levels: self.artifact.pyramid.len() as u32,
            sea_m: self.artifact.sea_m,
            coast_parts: self.coast_parts(),
        })
    }

    fn artifact_parts(&self) -> Vec<Bytes> {
        self.parts.get_or_init(|| self.encode_parts()).clone()
    }

    fn tiles_under(&self, dir: [f64; 3], radial_m: f64, interest_m: f64) -> Vec<(u8, u32, u32)> {
        // ★ THE TILES REACH WHERE THE CLIENT ASKS (ruling W17): the finest tile-reading rung's own
        // handover step floors the ring, and that step carries the FIELD's fold as well as the
        // octaves the coarser rung drops. Both hosts read one rule.
        let reach_m = vd_terrain::artifact::tile_reach_m(
            &self.lattice,
            self.artifact.pyramid.len() as u32,
            self.top_rung,
            self.body_radius_m,
            self.relief_m,
            radial_m,
            vd_core::geometry::drawable_theta_min_rad(),
            self.step_m,
        );
        self.tiles_within(dir, interest_m.max(reach_m))
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
        let source = ArtifactTiles::new(
            vd_core::pose::RealmId::Planet(moon.seed()),
            5,
            artifact.clone(),
            moon.macro_lattice().expect("a lattice"),
            moon.radius_m(),
            moon.relief_bound_m(0),
            moon.ladder().rungs - 1,
            // ★ THE HANDOVER STEP THE TILE RING IS FLOORED BY (ruling W17), the shipped line's own.
            {
                let lattice = moon.macro_lattice().expect("a lattice");
                let levels = artifact.pyramid.len() as u32;
                let rung =
                    vd_terrain::artifact::tile_rung(&lattice, levels, moon.ladder().rungs - 1);
                moon.step_bound_m(rung)
                    + vd_terrain::artifact::field_step_m(&moon, &lattice, levels, rung)
            },
        );
        assert_eq!(source.digest(), artifact.digest());
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
                sea_m: artifact.sea_m,
                coast_parts: source.coast_parts(),
            }
        );
        let parts = source.artifact_parts();
        // The parts are encoded once: a second ask hands back the same bytes.
        assert!(
            source
                .artifact_parts()
                .iter()
                .zip(&parts)
                .all(|(a, b)| Arc::ptr_eq(a, b))
        );
        // Level 2 (6 · 17² = 1 734 words) is one part; level 1 (6 · 34² = 6 936) is one part.
        // ★ THE COAST MASK rides after them (2026-09-22): the moon's 27 744 nodes are 3 468 bytes
        // of bits, one part, and it is the LAST of the list.
        assert_eq!(source.coast_parts(), 1);
        assert_eq!(parts.len(), 3);
        assert_eq!(
            postcard::from_bytes::<BulkMsg>(&parts[2]).expect("decodes"),
            BulkMsg::ArtifactCoast {
                realm: vd_core::pose::RealmId::Planet(moon.seed()),
                part: 0,
                parts: 1,
                bits: artifact.coast.clone(),
            }
        );
        assert_eq!(
            postcard::from_bytes::<BulkMsg>(&parts[0]).expect("decodes"),
            BulkMsg::ArtifactPyramid {
                realm: vd_core::pose::RealmId::Planet(moon.seed()),
                level: 2,
                part: 0,
                parts: 1,
                z_m: artifact.pyramid[1].clone(),
                water_m: artifact.pyramid_water[1].clone(),
            }
        );
        // Under +Z at the face centre (node 34 of 68), 250 km is 31 nodes (8.15 km each): the
        // four tiles of face +Z, the nearest first; 100 km (13 nodes) stays inside tile 0. At
        // 300 km (37 nodes) the reach crosses the face's edges, and the partner faces' first
        // tiles come after the four (2026-09-21, the tiles across a cube edge).
        assert_eq!(source.tiles_within([0.0, 0.0, 1.0], 100_000.0).len(), 1);
        let under = source.tiles_within([0.0, 0.0, 1.0], 250_000.0);
        assert_eq!(under.len(), 4);
        assert!(under.iter().all(|&(f, _, _)| f == Face::PosZ.index()));
        assert_eq!(under[0], (Face::PosZ.index(), 0, 0));
        let across = source.tiles_within([0.0, 0.0, 1.0], 300_000.0);
        assert!(across.len() > 4, "{across:?}");
        assert_eq!(across[0], (Face::PosZ.index(), 0, 0));
        for tile in &under {
            assert!(across.contains(tile), "{tile:?} not in {across:?}");
        }
        // A small radius near the corner of the face keeps one tile.
        let corner = source.tiles_within([0.0, 0.0, 1.0], 10.0);
        assert_eq!(corner.len(), 1);
        // ★ THE TILES REACH AS FAR AS THE FINE RUNGS READ (2026-09-20): an occupant 20 km up sees
        // to a 121 km horizon, one tile; 200 km up the horizon is 426 km and the rung-9 ring's
        // 445 km bounds nothing, so all four tiles of the face come, and the interest side wins
        // when it is the larger.
        let r = moon.radius_m();
        assert_eq!(
            source
                .tiles_under([0.0, 0.0, 1.0], r + 20_000.0, 10.0)
                .len(),
            1
        );
        // ★ Past the face's edges the reach names the partner faces' tiles too (2026-09-21): the
        // four tiles of +Z are all there, and the reach is bounded by the ladder's outer edge, so
        // the count is more than four and less than the whole cube.
        let per_face = usize::try_from(artifact.tiles_per_edge().pow(2)).expect("small");
        for (name, tiles) in [
            (
                "200 km up",
                source.tiles_under([0.0, 0.0, 1.0], r + 200_000.0, 10.0),
            ),
            (
                "300 km of interest",
                source.tiles_under([0.0, 0.0, 1.0], r, 300_000.0),
            ),
        ] {
            assert!(tiles.len() > 4, "{name}: {tiles:?}");
            assert!(tiles.len() < 6 * per_face, "{name}: {tiles:?}");
            for tile in &under {
                assert!(tiles.contains(tile), "{name}: {tile:?} not in {tiles:?}");
            }
        }
        // A direction under the face (the far side) names nothing on this face's basis... it names
        // the face it points at, never this one: −Z is its own face.
        let far = source.tiles_within([0.0, 0.0, -1.0], 10.0);
        assert_eq!(far[0].0, Face::NegZ.index());
        // No direction at all stands under no tile.
        assert!(source.tiles_under([0.0, 0.0, 0.0], r, 10.0).is_empty());
        let tile = source.tile(Face::PosZ.index(), 1, 1).expect("a tile");
        assert_eq!(
            postcard::from_bytes::<BulkMsg>(&tile).expect("decodes"),
            BulkMsg::ArtifactTile {
                realm: vd_core::pose::RealmId::Planet(moon.seed()),
                face: Face::PosZ.index(),
                tx: 1,
                ty: 1,
                rows: tile_bytes(&artifact.tile(Face::PosZ, 1, 1)),
            }
        );
        assert_eq!(source.tile(Face::PosZ.index(), 2, 0), None);
        assert_eq!(source.tile(Face::PosZ.index(), 0, 2), None);
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
        let pebble = moon.without_macro_lattice();
        assert!(pebble.macro_lattice().is_none());
        assert!(
            artifact_source_of(vd_core::pose::RealmId::Planet(1), 5, &pebble, &artifact).is_none()
        );
    }
}
