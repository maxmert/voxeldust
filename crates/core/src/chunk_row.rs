//! THE CHUNK ROW's bag — the one unit of the diff lane (the voxel foundation, slice 4; SL6 rows R-3
//! and R-13, owner YES, ruling V6; the row's shape is the storage report's,
//! `06_storage_diff_lane.md` §4.2).
//!
//! A chunk row is what the owning realm ships to a session that holds a chunk: everything the seed
//! did not decide about that chunk at one rung. It is a skip-unknown TLV bag under its own schema, so
//! a slice that learns a new kind of content APPENDS a tag and an older reader skips it (the rule that
//! covers TAGS, never key bytes or enum discriminants). The row's KEY (which chunk, which rung) and its
//! INSTANT (the tick the owner stated it) ride on the wire message itself
//! (`vd_wire::channels::ChunkRow`), never inside the bag.
//!
//! **The seven tags are the storage report's seven, numbered here today; their payloads land with
//! their slices.** A number is the one thing a later slice cannot add without care; a payload behind
//! a known number is free. Every payload is a postcard-encoded value the owning slice names; until
//! that slice lands, the tag carries opaque bytes and no reader in this build asks for it. What is NOT
//! a tag, and why: a sub-grid's small blocks, a landscape object and a body-owned block are each a CELL
//! RECORD of Format B (a sub-grid cell, an object cell), so they ride inside tags 1 and 2; the block
//! record domain's own growth and damage ride tag 7, which is reserved here for it.
//!
//! **Example.** A miner breaks one cell on a moon. The moon's shard builds one row for that chunk at
//! rung 0: the tick, and one sparse cell record. It ships the row to the three sessions that hold the
//! chunk. A client from last month's build reads the cells and skips the attachment tag it has never
//! heard of.

use crate::entity_kind::SchemaId;
use crate::tlv::{TlvError, TlvReader, TlvWriter};

/// The chunk row's schema. Beside the window body (20) and clear of the entity-kind blob block
/// (1–3, 10–13) and the realm store's rows (30–33); a test pins the disjointness.
pub const CHUNK_ROW_SCHEMA: SchemaId = SchemaId(21);

/// The seven tags of a chunk row (`06_storage_diff_lane.md` §4.2). NUMBERED ONCE, NEVER REUSED. The
/// slice that owns each payload is named beside it.
pub mod tags {
    /// Authored cells, SPARSE: `(CellIndex, record)` pairs at tier 0 (Format B; slice 9 stores them, slice 10
    /// ships them).
    pub const CELLS_SPARSE: u16 = 1;
    /// Authored cells, MASKED-DENSE: a cell mask and the records in mask order, at tier 0, for a
    /// chunk more edited than sparse pays for (slice 10 states the crossover).
    pub const CELLS_DENSE: u16 = 2;
    /// Pyramid entries of a coarse rung (Format C, slice 9 — the canopy fold among them).
    pub const PYRAMID: u16 = 3;
    /// Cells returned to the seed: a revert list, so a client can drop an edit it holds (slice 10).
    pub const REVERTED: u16 = 4;
    /// Attachment presence and static configuration, taking no volume (slice 13).
    pub const ATTACHMENTS: u16 = 5;
    /// Codec: reserved, always none in this build.
    pub const CODEC: u16 = 6;
    /// Feature state — growth stage, damage — the block record domain's tags, reserved here
    /// (slice 14).
    pub const FEATURE_STATE: u16 = 7;
    /// Every tag, for the drift test.
    pub const ALL: [u16; 7] = [
        CELLS_SPARSE,
        CELLS_DENSE,
        PYRAMID,
        REVERTED,
        ATTACHMENTS,
        CODEC,
        FEATURE_STATE,
    ];
}

/// Encode the one payload this build already names: the sparse cell records' bytes. Slice 9's
/// producer is the first caller; until then the size example measures it.
#[must_use]
pub fn chunk_row_bag(cells_sparse: &[u8]) -> Vec<u8> {
    TlvWriter::new(CHUNK_ROW_SCHEMA)
        .required(tags::CELLS_SPARSE, &cells_sparse)
        .expect("one tag; a chunk's records are far under the field cap")
        .finish()
}

/// The sparse cell records' bytes of a row; `None` for a row that carries none (a dense row, a
/// revert-only row); a typed refusal for a bag that is not a chunk row.
pub fn sparse_cells_of(bag: &[u8]) -> Result<Option<Vec<u8>>, TlvError> {
    TlvReader::parse(CHUNK_ROW_SCHEMA, bag)?.optional(tags::CELLS_SPARSE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity_kind::EntityKind;
    use crate::look::WINDOW_BODY_SCHEMA;

    #[test]
    fn the_seven_tags_are_distinct_and_the_schema_id_is_free() {
        let mut seen = std::collections::BTreeSet::new();
        for t in tags::ALL {
            assert!(seen.insert(t), "tag {t} is numbered twice");
            assert!(t >= 1, "tag 0 is never used");
        }
        assert_eq!(tags::ALL, [1, 2, 3, 4, 5, 6, 7], "the numbering moved");
        assert_eq!(CHUNK_ROW_SCHEMA, SchemaId(21));
        // Disjoint from every schema id another table spends: the entity kinds' blobs and the window
        // body. (The realm store's 30..=33 are pinned in their own crate.)
        for kind in EntityKind::ALL {
            assert_ne!(
                kind.def().blob_schema,
                CHUNK_ROW_SCHEMA,
                "{kind:?} spends 21"
            );
        }
        assert_ne!(WINDOW_BODY_SCHEMA, CHUNK_ROW_SCHEMA);
    }

    #[test]
    fn a_row_roundtrips_and_an_older_reader_skips_a_tag_it_never_heard_of() {
        let bag = chunk_row_bag(&[1, 2, 3]);
        assert_eq!(sparse_cells_of(&bag), Ok(Some(vec![1, 2, 3])));
        // A future row carries an attachment list this build never asks for, and no sparse cells.
        let future = TlvWriter::new(CHUNK_ROW_SCHEMA)
            .required(tags::ATTACHMENTS, &vec![9u8, 9])
            .expect("one tag")
            .finish();
        assert_eq!(
            sparse_cells_of(&future),
            Ok(None),
            "a row with no sparse cells"
        );
        // A bag under another schema is refused by name, never read as a row.
        let other = TlvWriter::new(WINDOW_BODY_SCHEMA)
            .required(tags::CELLS_SPARSE, &vec![1u8])
            .expect("one tag")
            .finish();
        assert_eq!(
            sparse_cells_of(&other),
            Err(TlvError::SchemaMismatch {
                expected: CHUNK_ROW_SCHEMA,
                got: WINDOW_BODY_SCHEMA
            })
        );
    }
}
