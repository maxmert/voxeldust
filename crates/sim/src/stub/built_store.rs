//! ★ WHAT THIS REALM REMEMBERS — the two families a realm's own store holds (D-MOVE-2; owner rulings
//! 2026-09-01).
//!
//! **A REALM HAD NO STORE AT ALL BEFORE THIS.** The orchestrator has one, and the outbox has one; a
//! shard has never opened a file. The shard binary's own note calls it later work: *"the P7 durable
//! per-realm store fills this map, whose contents are realm-local by construction."* This is that
//! store's first two families.
//!
//! **WHY A REALM NEEDS TO REMEMBER ANYTHING.** Everything the seed makes is computed — the same number
//! gives the same stars in every process, so nothing is stored and nothing can drift. What a PLAYER
//! made cannot be computed. A ship exists because somebody built it, and if nobody writes that down it
//! stops existing the moment the process stops.
//!
//! **TWO FAMILIES, EACH HELD BY THE PARTY THE LAWS ALREADY MAKE RESPONSIBLE:**
//! - a parent holds the BERTHS it authored for its built children (SL1 — a parent authors its
//!   children's placements and is their only writer; this is its own past authorship, kept);
//! - a realm holds its own BODY (SL3 — a realm authors how it looks).
//!
//! The keyspace is the established shape: one flat table, one prefix byte per family, `scan(prefix)`
//! reads a family whole. Tags are APPEND-ONLY, and tag 0 belongs to the file's own label.
use vd_core::built::{Berth, BlueprintId, BuiltBody, BuiltFacts};
use vd_core::entity_kind::SchemaId;
use vd_core::fence::Fence;
use vd_core::geometry::Boundary;
use vd_core::ids::AccountId;
use vd_core::pose::RealmId;
use vd_core::tlv::{TlvReader, TlvWriter};

/// ★ A BODY'S SHAPE, so a reader can tell what it is holding. Taken at 30 — clear of every block in
/// use (1-3, 7-8, 10-13, 20), so neither list has to be renumbered.
const BODY_SCHEMA: SchemaId = SchemaId(30);
/// A berth's shape. 31, for the same reason.
const BERTH_SCHEMA: SchemaId = SchemaId(31);
/// ★ THE BLOCK STORE's two row families (the voxel foundation, slice 4 plant; storage report §5.2 row
/// 8; owner S4-7, 2026-09-07): the chunk diff rows and the pyramid rows of a realm, beside its body and
/// its berths in the SAME realm store. PLANTED here so no later family can take these ids; the rows
/// themselves land with slice 9 (the store), which is the first writer.
pub const CHUNK_DELTA_SCHEMA: SchemaId = SchemaId(32);
pub const CHUNK_PYRAMID_SCHEMA: SchemaId = SchemaId(33);
/// ★ THE ARTIFACT's THREE ROW FAMILIES (the landform arc, slice 8c stage C4; the owner's ruling of
/// 2026-09-19: the solve runs once on the server and the artifact is saved in the shard's db): one
/// HEAD row (the version, the lattice edge, the digest), one row per TILE of node rows, and one
/// PYRAMID row. The sim never names the generator: a tile is bytes it stores and ships, and the
/// composition root, which links the generator, is the one reader that decodes them.
pub const ARTIFACT_HEAD_SCHEMA: SchemaId = SchemaId(34);
pub const ARTIFACT_TILE_SCHEMA: SchemaId = SchemaId(35);
pub const ARTIFACT_PYRAMID_SCHEMA: SchemaId = SchemaId(36);

/// ★ WHY THESE ROWS ARE FRAMED AND THE MOVEMENT LANE IS NOT.
///
/// Our codec writes values in order and no names, so an OLDER reader cannot skip a field a NEWER
/// writer appended — it reads the new bytes as its own fields, with no error at all. This codebase has
/// paid for that once: its own note records that decode-to-default *"silently zeroed players on
/// rolling deploys."*
///
/// Framing costs about nine bytes of header and six per field. That is the right trade HERE and the
/// wrong one on a hot lane, and the difference is how often the bytes move:
///
/// | | written | read | framed? |
/// |---|---|---|---|
/// | a body | when built, stored, or changed | once, at boot | YES — it will gain fields for years |
/// | a berth | when built or moved home | once, at boot | YES |
/// | a per-tick drive | fifty times a second, per ship | every tick | NO — six numbers, nothing else |
///
/// **THE STORE'S SHAPE AND THE WIRE'S SHAPE STAY SEPARATE.** A ship reads its own body from its file,
/// then states what it IS in the wire's own message. The two evolve on different clocks: the wire is a
/// frozen reviewed contract, and the store must gain blocks, damage and ownership history. One shape
/// for both would freeze each by the other.
///
/// Tags are APPEND-ONLY. A new field takes the next number; nothing is ever renumbered or reused.
mod tags {
    /// Which realm this body is.
    pub const REALM: u16 = 1;
    /// Whose it is.
    pub const OWNER: u16 = 2;
    /// What it was built from.
    pub const BLUEPRINT: u16 = 3;
    /// The box that decides who is inside it.
    pub const BOUND: u16 = 4;
    /// How big it draws.
    pub const LOOK: u16 = 5;
    /// What it is made of.
    pub const FACTS: u16 = 6;
    /// Which commit last moved the row.
    pub const FENCE: u16 = 7;
    /// A berth's child.
    pub const CHILD: u16 = 1;
    /// Where the parent put it.
    pub const OFFSET: u16 = 2;
}

/// The berth family: one row per built child this realm authored a place for.
const BERTH: u8 = 1;
/// The body family: this realm's own row. Exactly one, so it needs no key beyond its tag.
const BODY: u8 = 2;
/// The key family of the realm store's OWNER-FENCE row (slice 4 plant; storage report A-9): the
/// realm fence of the last shard that wrote this store. A store REFUSES to open when the row is not
/// strictly below the opener's fence, so a second writer on a cloud volume is caught by the fence and
/// never by a file lock, which a network-backed volume cannot promise. The row's writer and its
/// refusal land with slice 9; the key is fixed here so it can never collide with a family added later.
const OWNER_FENCE: u8 = 3;

/// ★ THE ARTIFACT's families (slice 8c stage C4): the head, the tiles, the pyramid.
const ARTIFACT_HEAD: u8 = 4;
const ARTIFACT_TILE: u8 = 5;
const ARTIFACT_PYRAMID: u8 = 6;

/// The artifact rows' tags — their own numbering, one schema each, append-only.
mod art {
    pub const WORLD_TAG: u16 = 1;
    pub const VERSION: u16 = 2;
    pub const EDGE: u16 = 3;
    pub const DIGEST: u16 = 4;
    pub const TILES_PER_EDGE: u16 = 5;
    pub const FACE: u16 = 1;
    pub const TX: u16 = 2;
    pub const TY: u16 = 3;
    pub const BYTES: u16 = 4;
    pub const LEVELS: u16 = 1;
}

/// The key of the owner-fence row (one per realm store).
#[must_use]
pub fn owner_fence_key() -> Vec<u8> {
    vec![OWNER_FENCE]
}

/// ★ THE BLOCK STORE's OPERATIONAL NUMBERS (slice 4 plant; storage report §5.2 row 9; owner S4-6,
/// 2026-09-07: the field NAMES now, the VALUES with slice 9's benches). Every field is a number a
/// bench decides; none has a default here, because a number typed before its bench is a magic number.
/// The first constructor is slice 9, which lands each value beside the measurement that chose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockStoreTuning {
    /// Ticks between two checkpoints of the pyramid tail (bench M-6: replay ≤ one interval of edits).
    pub checkpoint_interval_ticks: u64,
    /// The per-realm byte budget of the diff rows, sized to the scattered case (bench M-3: 100 M
    /// marks 57 m apart).
    pub realm_byte_budget: u64,
    /// The most chunk rows one session receives per second (bench M-2: a city's catch-up under the
    /// 20 Hz datagram's p99).
    pub session_rows_per_second: u32,
    /// The farthest a session may edit from its own occupant, in metres (the reach rule of slice 10).
    pub max_reach_m: u32,
    /// The cells per pyramid storage block (bench M-4: write amplification per changed entry).
    pub pyramid_storage_block_cells: u32,
    /// The most ticks an edit may park behind a stalled disk before the refusal is typed (bench M-10:
    /// poses keep shipping, only the edit lane parks).
    pub parked_ticks_max: u32,
}

/// The artifact's head: what a reader checks before it trusts a tile.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactHead {
    /// The generator's declared world tag the artifact was solved under.
    pub world_tag: u64,
    /// The artifact's own version.
    pub version: u32,
    /// The macro lattice's edge.
    pub edge: u32,
    /// The digest, two words.
    pub digest: [u64; 2],
    /// The tiles along a face's edge.
    pub tiles_per_edge: u32,
}

/// One stored tile: its place and its rows' bytes, opaque to the sim.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactTile {
    pub face: u8,
    pub tx: u32,
    pub ty: u32,
    /// The rows, nine bytes each, row-major from the tile's origin.
    pub bytes: Vec<u8>,
}

/// The key of the artifact's head: the family tag alone.
#[must_use]
pub fn artifact_head_key() -> Vec<u8> {
    vec![ARTIFACT_HEAD]
}

/// The key of one tile: the family tag, the face, then the tile's column and row, big-endian so a
/// prefix scan walks a face in order.
#[must_use]
pub fn artifact_tile_key(face: u8, tx: u32, ty: u32) -> Vec<u8> {
    let mut k = vec![ARTIFACT_TILE, face];
    k.extend(tx.to_be_bytes());
    k.extend(ty.to_be_bytes());
    k
}

/// The prefix that reads every tile in one scan.
#[must_use]
pub fn artifact_tile_prefix() -> Vec<u8> {
    vec![ARTIFACT_TILE]
}

/// The key of the pyramid: the family tag alone.
#[must_use]
pub fn artifact_pyramid_key() -> Vec<u8> {
    vec![ARTIFACT_PYRAMID]
}

/// The head, framed.
#[must_use]
pub fn encode_artifact_head(head: &ArtifactHead) -> Vec<u8> {
    TlvWriter::new(ARTIFACT_HEAD_SCHEMA)
        .required(art::WORLD_TAG, &head.world_tag)
        .and_then(|w| w.required(art::VERSION, &head.version))
        .and_then(|w| w.required(art::EDGE, &head.edge))
        .and_then(|w| w.required(art::DIGEST, &head.digest))
        .and_then(|w| w.required(art::TILES_PER_EDGE, &head.tiles_per_edge))
        .expect("a head's fields are small and encode infallibly")
        .finish()
}

/// A tile, framed.
#[must_use]
pub fn encode_artifact_tile(tile: &ArtifactTile) -> Vec<u8> {
    TlvWriter::new(ARTIFACT_TILE_SCHEMA)
        .required(art::FACE, &tile.face)
        .and_then(|w| w.required(art::TX, &tile.tx))
        .and_then(|w| w.required(art::TY, &tile.ty))
        .and_then(|w| w.required(art::BYTES, &tile.bytes))
        .expect("a tile's bytes encode infallibly")
        .finish()
}

/// The pyramid, framed: every level's heights as little-endian words, one blob per level.
#[must_use]
pub fn encode_artifact_pyramid(levels: &[Vec<u8>]) -> Vec<u8> {
    TlvWriter::new(ARTIFACT_PYRAMID_SCHEMA)
        .required(art::LEVELS, &levels)
        .expect("a pyramid's bytes encode infallibly")
        .finish()
}

/// A head read back, or a named refusal (a row that does not decode is refused, never defaulted).
///
/// # Errors
/// The bytes are not a head this build can read.
pub fn decode_artifact_head(bytes: &[u8]) -> Result<ArtifactHead, String> {
    let r = TlvReader::parse(ARTIFACT_HEAD_SCHEMA, bytes)
        .map_err(|e| format!("a stored artifact head does not decode: {e}"))?;
    Ok(ArtifactHead {
        world_tag: field(&r, art::WORLD_TAG, "world tag")?,
        version: field(&r, art::VERSION, "version")?,
        edge: field(&r, art::EDGE, "edge")?,
        digest: field(&r, art::DIGEST, "digest")?,
        tiles_per_edge: field(&r, art::TILES_PER_EDGE, "tiles per edge")?,
    })
}

/// A tile read back, or a named refusal.
///
/// # Errors
/// The bytes are not a tile this build can read.
pub fn decode_artifact_tile(bytes: &[u8]) -> Result<ArtifactTile, String> {
    let r = TlvReader::parse(ARTIFACT_TILE_SCHEMA, bytes)
        .map_err(|e| format!("a stored artifact tile does not decode: {e}"))?;
    Ok(ArtifactTile {
        face: field(&r, art::FACE, "face")?,
        tx: field(&r, art::TX, "tx")?,
        ty: field(&r, art::TY, "ty")?,
        bytes: field(&r, art::BYTES, "bytes")?,
    })
}

/// A pyramid read back, or a named refusal.
///
/// # Errors
/// The bytes are not a pyramid this build can read.
pub fn decode_artifact_pyramid(bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let r = TlvReader::parse(ARTIFACT_PYRAMID_SCHEMA, bytes)
        .map_err(|e| format!("a stored artifact pyramid does not decode: {e}"))?;
    field(&r, art::LEVELS, "levels")
}

/// The key for one built child's berth: the family tag, then the child's name.
///
/// The name is encoded rather than raw so a 128-bit built name and a 64-bit generated one both key
/// correctly — the encoding is what makes the keyspace indifferent to how wide a name is.
#[must_use]
pub fn berth_key(child: RealmId) -> Vec<u8> {
    let mut k = vec![BERTH];
    k.extend(postcard::to_allocvec(&child).expect("a realm name encodes infallibly"));
    k
}

/// The prefix that reads every berth this realm authored, in one scan.
#[must_use]
pub fn berth_prefix() -> Vec<u8> {
    vec![BERTH]
}

/// The key for this realm's own body. One realm, one body, so the tag IS the key.
#[must_use]
pub fn body_key() -> Vec<u8> {
    vec![BODY]
}

/// A berth, framed.
#[must_use]
pub fn encode_berth(berth: &Berth) -> Vec<u8> {
    TlvWriter::new(BERTH_SCHEMA)
        .required(tags::CHILD, &berth.child)
        .and_then(|w| w.required(tags::OFFSET, &berth.offset_m))
        .and_then(|w| w.required(tags::BOUND, &berth.bound))
        .and_then(|w| w.required(tags::LOOK, &berth.look))
        .and_then(|w| w.required(tags::FENCE, &berth.fence))
        .expect("a berth's fields are small and encode infallibly")
        .finish()
}

/// A body, framed.
#[must_use]
pub fn encode_body(body: &BuiltBody) -> Vec<u8> {
    TlvWriter::new(BODY_SCHEMA)
        .required(tags::REALM, &body.realm)
        .and_then(|w| w.required(tags::OWNER, &body.owner))
        .and_then(|w| w.required(tags::BLUEPRINT, &body.blueprint))
        .and_then(|w| w.required(tags::BOUND, &body.bound))
        .and_then(|w| w.required(tags::LOOK, &body.look))
        .and_then(|w| w.required(tags::FACTS, &body.facts))
        .and_then(|w| w.required(tags::FENCE, &body.fence))
        .expect("a body's fields are small and encode infallibly")
        .finish()
}

/// A berth read back, or a named refusal.
///
/// ★ **A ROW THAT DOES NOT DECODE IS REFUSED, NEVER DEFAULTED.** The project bans decoding a durable
/// record to a default, and a berth is exactly why: a defaulted berth is a hull at the origin, which
/// is inside the star, and nothing would have reported a problem.
///
/// # Errors
/// The bytes are not a berth this build can read.
pub fn decode_berth(bytes: &[u8]) -> Result<Berth, String> {
    let r = TlvReader::parse(BERTH_SCHEMA, bytes)
        .map_err(|e| format!("a stored berth does not decode: {e}"))?;
    Ok(Berth {
        child: field(&r, tags::CHILD, "child")?,
        offset_m: field(&r, tags::OFFSET, "offset")?,
        bound: field(&r, tags::BOUND, "bound")?,
        look: field(&r, tags::LOOK, "look")?,
        fence: field(&r, tags::FENCE, "fence")?,
    })
}

/// One field, or a refusal naming which one — monomorphic over the error path so the generic reader
/// above stays a straight line (HR5: all branching in one helper, never inside a generic body).
fn field<T: serde::de::DeserializeOwned>(
    r: &TlvReader<'_>,
    tag: u16,
    name: &str,
) -> Result<T, String> {
    r.required(tag)
        .map_err(|e| format!("a stored row is missing or malformed at {name}: {e}"))
}

/// A body read back, or a named refusal. Refused rather than defaulted, for the same reason a berth is.
///
/// # Errors
/// The bytes are not a body this build can read.
pub fn decode_body(bytes: &[u8]) -> Result<BuiltBody, String> {
    let r = TlvReader::parse(BODY_SCHEMA, bytes)
        .map_err(|e| format!("a stored body does not decode: {e}"))?;
    Ok(BuiltBody {
        realm: field::<RealmId>(&r, tags::REALM, "realm")?,
        owner: field::<AccountId>(&r, tags::OWNER, "owner")?,
        blueprint: field::<BlueprintId>(&r, tags::BLUEPRINT, "blueprint")?,
        bound: field::<Boundary>(&r, tags::BOUND, "bound")?,
        look: field::<Boundary>(&r, tags::LOOK, "look")?,
        facts: field::<BuiltFacts>(&r, tags::FACTS, "facts")?,
        fence: field::<Fence>(&r, tags::FENCE, "fence")?,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ArtifactHead, ArtifactTile, artifact_head_key, artifact_pyramid_key, artifact_tile_key,
        artifact_tile_prefix, decode_artifact_head, decode_artifact_pyramid, decode_artifact_tile,
        encode_artifact_head, encode_artifact_pyramid, encode_artifact_tile,
    };
    use super::{
        BERTH_SCHEMA, BODY_SCHEMA, BlockStoreTuning, CHUNK_DELTA_SCHEMA, CHUNK_PYRAMID_SCHEMA,
        berth_key, berth_prefix, body_key, decode_berth, decode_body, encode_berth, encode_body,
        owner_fence_key,
    };
    use vd_core::built::{Berth, BlueprintId, BuiltBody, BuiltFacts};
    use vd_core::entity_kind::EntityKind;
    use vd_core::entity_kind::SchemaId;
    use vd_core::fence::Fence;
    use vd_core::geometry::Boundary;
    use vd_core::glam::DVec3;
    use vd_core::ids::{AccountId, EntityId};
    use vd_core::pose::RealmId;

    /// ★ THE ARTIFACT's ROWS (slice 8c stage C4): the head, a tile and a pyramid round-trip through
    /// their frames; the keys order a face's tiles by column then row; a row that does not decode is
    /// refused by name, never defaulted.
    #[test]
    fn artifact_rows_round_trip_and_refuse_garbage() {
        let head = ArtifactHead {
            world_tag: 0xdead_beef,
            version: 1,
            edge: 1_216,
            digest: [7, 11],
            tiles_per_edge: 19,
        };
        assert_eq!(
            decode_artifact_head(&encode_artifact_head(&head)),
            Ok(head.clone())
        );
        let tile = ArtifactTile {
            face: 3,
            tx: 2,
            ty: 18,
            bytes: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        };
        assert_eq!(
            decode_artifact_tile(&encode_artifact_tile(&tile)),
            Ok(tile.clone())
        );
        let levels = vec![vec![1u8, 0, 2, 0], vec![3, 0]];
        assert_eq!(
            decode_artifact_pyramid(&encode_artifact_pyramid(&levels)),
            Ok(levels)
        );
        assert_eq!(artifact_head_key(), vec![4]);
        assert_eq!(artifact_pyramid_key(), vec![6]);
        assert_eq!(artifact_tile_prefix(), vec![5]);
        assert_eq!(
            artifact_tile_key(3, 2, 18),
            vec![5, 3, 0, 0, 0, 2, 0, 0, 0, 18]
        );
        assert!(artifact_tile_key(3, 2, 18) < artifact_tile_key(3, 3, 0));
        assert!(artifact_tile_key(3, 2, 18) > artifact_tile_key(3, 2, 17));
        assert!(decode_artifact_head(b"nonsense").is_err());
        assert!(decode_artifact_tile(&encode_artifact_head(&head)).is_err());
        assert!(decode_artifact_pyramid(&encode_artifact_tile(&tile)).is_err());
    }

    fn ship(seq: u64) -> RealmId {
        RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, seq, 0))
    }
    fn a_berth(child: RealmId) -> Berth {
        Berth {
            child,
            offset_m: DVec3::new(1000.0, 0.0, 0.0),
            bound: Boundary::Shell { r: 20.0 },
            look: Boundary::Shell { r: 20.0 },
            fence: Fence(1),
        }
    }
    fn a_body() -> BuiltBody {
        BuiltBody {
            realm: ship(1),
            owner: AccountId(1000),
            blueprint: BlueprintId(7),
            bound: Boundary::Shell { r: 20.0 },
            look: Boundary::Shell { r: 20.0 },
            facts: BuiltFacts {
                mass_g: 50_000_000,
                cross_section_mm2: 12_000_000,
                drag_micro: 820_000,
                max_push_micro_mps2: 98_100_000,
                max_turn_micro_radps2: 800_000,
            },
            fence: Fence(1),
        }
    }

    #[test]
    fn a_berth_round_trips_through_its_own_key() {
        let berth = a_berth(ship(1));
        let read = decode_berth(&encode_berth(&berth)).expect("a written berth reads back");
        assert_eq!(read.child, berth.child);
        assert_eq!(read.offset_m, berth.offset_m);
    }

    #[test]
    fn a_body_round_trips_with_its_whole_minted_name() {
        let body = a_body();
        let read = decode_body(&encode_body(&body)).expect("a written body reads back");
        assert_eq!(
            read.realm, body.realm,
            "the whole 128-bit name survives the store"
        );
        assert_eq!(read.owner, body.owner);
    }

    #[test]
    fn every_berth_shares_one_prefix_so_a_parent_reads_them_in_one_scan() {
        // ★ SL9: a parent's built children are unbounded, so reading them must be ONE scan of one
        // prefix — never a lookup per child, and never a walk of every key in the file.
        let prefix = berth_prefix();
        for seq in 1..=64 {
            assert!(
                berth_key(ship(seq)).starts_with(&prefix),
                "every berth is under one prefix"
            );
        }
    }

    #[test]
    fn two_children_never_share_a_key() {
        // A shared key would silently lose one hull: the second write overwrites the first, and
        // nothing reports it.
        assert_ne!(berth_key(ship(1)), berth_key(ship(2)));
        // A generated child and a built one key differently too, even though their names differ in
        // width — the encoding is what makes the keyspace indifferent to that.
        assert_ne!(berth_key(ship(1)), berth_key(RealmId::Planet(1)));
    }

    #[test]
    fn a_berth_and_a_body_can_never_be_read_as_each_other() {
        // ★ THE FAMILIES ARE SEPARATE TAGS, and that is load-bearing: the encoding is positional and
        // carries no field names, so two different row shapes can decode from one set of bytes with no
        // error at all. The tag is what stops a body being read as a berth.
        assert_ne!(berth_key(ship(1))[0], body_key()[0]);
    }

    #[test]
    fn a_row_that_does_not_decode_is_refused_and_not_defaulted() {
        // A defaulted berth is a hull at the origin — which is inside the star — and nothing would
        // have reported a problem. The project bans decoding a durable record to a default, and this
        // is exactly the case the ban exists for.
        let refusal = decode_berth(&[0xFF, 0xFF, 0xFF]).expect_err("garbage is refused");
        assert!(
            refusal.contains("does not decode"),
            "and it says so: {refusal}"
        );
        assert!(decode_body(&[0xFF, 0xFF, 0xFF]).is_err());
    }

    #[test]
    fn a_reader_survives_a_field_it_has_never_heard_of() {
        // ★ THIS IS WHY THE ROWS ARE FRAMED, and it is the only test that proves it.
        //
        // Our codec writes values in order and NO NAMES. Without framing, a row written by a newer
        // build is read by an older one as its own fields — silently, with no error. This codebase has
        // paid for that once: its note records that decode-to-default "silently zeroed players on
        // rolling deploys".
        //
        // Here a FUTURE writer appends a block table at tag 9. Today's reader has never heard of tag 9.
        // It must step over it by its stated length and read everything it does know, correctly.
        let body = a_body();
        let future = vd_core::tlv::TlvWriter::new(super::BODY_SCHEMA)
            .required(super::tags::REALM, &body.realm)
            .and_then(|w| w.required(super::tags::OWNER, &body.owner))
            .and_then(|w| w.required(super::tags::BLUEPRINT, &body.blueprint))
            .and_then(|w| w.required(super::tags::BOUND, &body.bound))
            .and_then(|w| w.required(super::tags::LOOK, &body.look))
            .and_then(|w| w.required(super::tags::FACTS, &body.facts))
            .and_then(|w| w.required(super::tags::FENCE, &body.fence))
            // A field from a build that does not exist yet — blocks, when they land.
            .and_then(|w| w.optional(9, &vec![1_u8; 64]))
            .expect("a future body encodes")
            .finish();

        let read = decode_body(&future).expect("today's reader reads tomorrow's row");
        assert_eq!(read.realm, body.realm, "the name survived an unknown field");
        assert_eq!(read.owner, body.owner);
        assert_eq!(read.facts, body.facts, "and so did everything after it");
    }

    #[test]
    fn a_row_cut_in_half_is_an_error_and_not_a_shorter_row() {
        // Without the declared field count, a row cut at a field boundary parses as a VALID smaller
        // row — a property a test in this codebase found the hard way. A hull whose mass simply
        // vanished would fly wrong for ever and nothing would say so.
        let whole = encode_body(&a_body());
        let bytes = whole.len();
        for cut in [4, bytes / 3, bytes / 2, bytes - 1] {
            let refused = decode_body(&whole[..cut]).is_err();
            assert!(refused, "a row cut at {cut} of {bytes} bytes is refused");
        }
    }

    #[test]
    fn a_berth_read_as_a_body_is_refused_by_its_own_shape() {
        // The shapes are separate ids, so one can never be read as the other — which matters because
        // the bytes carry no names and a wrong reading would look like data.
        let berth_bytes = encode_berth(&a_berth(ship(1)));
        assert!(decode_body(&berth_bytes).is_err(), "a berth is not a body");
        let body_bytes = encode_body(&a_body());
        assert!(decode_berth(&body_bytes).is_err(), "a body is not a berth");
    }

    #[test]
    fn the_framing_costs_what_was_claimed_and_not_more() {
        // MEASURED, so the trade stated in the module note is a number rather than an argument.
        let body = encode_body(&a_body());
        let berth = encode_berth(&a_berth(ship(1)));
        println!(
            "[framing] a body is {} bytes, a berth is {} bytes",
            body.len(),
            berth.len()
        );
        // A body is written a handful of times per ship, ever, and read once at boot. A hundred bytes
        // there is free. The same hundred bytes on the per-tick movement lane would be fifty times a
        // second, per ship — which is why that lane is NOT framed.
        let (body_b, berth_b) = (body.len(), berth.len());
        assert!(body_b < 200, "a body stays small: {body_b} bytes");
        assert!(berth_b < 200, "a berth stays small: {berth_b} bytes");
    }

    /// The slice 4 plant: the two block families and the owner-fence key never collide with what the
    /// store already holds, and the tuning struct names every number without stating one.
    #[test]
    fn the_block_store_plant_takes_free_ids_and_a_free_key_family() {
        let schemas = [
            BODY_SCHEMA,
            BERTH_SCHEMA,
            CHUNK_DELTA_SCHEMA,
            CHUNK_PYRAMID_SCHEMA,
        ];
        let distinct: std::collections::BTreeSet<u16> = schemas.iter().map(|s| s.0).collect();
        assert_eq!(distinct.len(), schemas.len(), "a schema id is taken twice");
        assert_eq!(
            (CHUNK_DELTA_SCHEMA, CHUNK_PYRAMID_SCHEMA),
            (SchemaId(32), SchemaId(33))
        );
        let families = [berth_prefix()[0], body_key()[0], owner_fence_key()[0]];
        let distinct: std::collections::BTreeSet<_> = families.iter().collect();
        assert_eq!(
            distinct.len(),
            families.len(),
            "a key family is taken twice"
        );
        assert_eq!(owner_fence_key(), vec![3]);
        assert_ne!(owner_fence_key()[0], 0, "key 0 is the store's label row");
        let tuning = BlockStoreTuning {
            checkpoint_interval_ticks: 1,
            realm_byte_budget: 1,
            session_rows_per_second: 1,
            max_reach_m: 1,
            pyramid_storage_block_cells: 1,
            parked_ticks_max: 1,
        };
        assert_eq!(tuning, tuning.clone(), "six named numbers, none shipped");
    }
}
