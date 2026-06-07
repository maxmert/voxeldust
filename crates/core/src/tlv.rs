//! TLV-framed entity-state blobs (`docs/design/generic_transfer.md` §A1).
//!
//! postcard is positional and non-self-describing: an OLDER reader cannot skip
//! trailing fields a NEWER writer appended — the old repo's `handoff_blob.rs`
//! "decode-to-Default-on-any-error" silently zeroed players on rolling deploys.
//! This module reintroduces forward skip at the framing level:
//!
//! ```text
//! TlvHeader { schema_id: u16, writer_max_tag: u16, required_max_tag: u16,
//!             field_count: u16, codec_flags: u8 }
//! repeated TlvField { tag: u16, len: u32, bytes: [u8; len] }   // bytes = postcard of the field
//! ```
//!
//! `field_count` exists because pure TLV cannot detect truncation at a field
//! boundary (a proptest found this): a blob cut after field 1 of 2 would parse as a
//! structurally-valid 1-field blob. With the declared count, truncation at ANY byte
//! of a non-empty blob is a hard error.
//!
//! - A reader consumes the tags it knows and SKIPS unknown tags by `len`.
//! - `required_max_tag` powers the version-floor handshake: a dest whose
//!   `max_known_tag < required_max_tag` cannot represent the state and the transfer is
//!   REFUSED at PREPARE (the entity stays alive on source) — never decoded to Default.
//! - A reader asking for a required tag that is absent gets a hard error.
//! - Writers emit fields in ascending tag order regardless of insertion order, so blob
//!   bytes are canonical (content-hash dedup safe).
//!
//! All integers little-endian, fixed width (no varints: skipping must never need to
//! understand a field to step over it).

use serde::{Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;

use crate::entity_kind::SchemaId;

/// Header byte length: schema u16 + writer_max u16 + required_max u16 +
/// field_count u16 + flags u8.
const HEADER_LEN: usize = 9;
/// Per-field prelude: tag u16 + len u32.
const FIELD_PRELUDE_LEN: usize = 6;
/// Hard cap on a single field payload. Kind blobs are `KindDef::max_state_bytes`
/// (u16) by policy; this framing-level cap matches the transport frame limit so an
/// oversize field is rejected at construction, not mid-wire.
pub const MAX_FIELD_BYTES: usize = 1 << 20;
/// v1 codec flags: postcard, no compression. Any other value is from the future and
/// must be rejected, not guessed at.
const CODEC_FLAGS_V1: u8 = 0;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TlvError {
    #[error("blob truncated at byte {0}")]
    Truncated(usize),
    #[error("duplicate tag {0}")]
    DuplicateTag(u16),
    #[error("schema mismatch: expected {expected:?}, got {got:?}")]
    SchemaMismatch { expected: SchemaId, got: SchemaId },
    #[error("unsupported codec flags {0:#04x}")]
    UnsupportedCodec(u8),
    #[error("required tag {0} missing — refuse the transfer, never decode to Default")]
    MissingRequiredTag(u16),
    #[error("field {tag} failed to decode")]
    FieldDecode { tag: u16 },
    #[error("field {tag} payload exceeds the {MAX_FIELD_BYTES}-byte cap")]
    FieldTooLarge { tag: u16 },
    #[error("writer declared required_max_tag {required} above writer_max_tag {max}")]
    IncoherentHeader { required: u16, max: u16 },
    #[error("header declares {declared} fields but {found} were present (truncation?)")]
    FieldCountMismatch { declared: u16, found: u16 },
}

/// Builds a canonical TLV blob. Fields may be added in any order; output is sorted.
#[derive(Debug)]
pub struct TlvWriter {
    schema: SchemaId,
    fields: BTreeMap<u16, (bool, Vec<u8>)>, // tag -> (required, payload)
}

impl TlvWriter {
    #[must_use]
    pub fn new(schema: SchemaId) -> TlvWriter {
        TlvWriter {
            schema,
            fields: BTreeMap::new(),
        }
    }

    // COVERAGE PATTERN (codified in CLAUDE.md): generic fns are BRANCHLESS shims —
    // llvm counts regions per monomorphization, so a branch inside a generic fn must
    // be exercised for EVERY instantiated type. All branching lives in the
    // monomorphic `push_serialized`, covered once for all types.

    /// Add a REQUIRED field: a reader that does not know this tag cannot represent
    /// the state, and the version-floor handshake will refuse routing to it.
    pub fn required<T: Serialize>(self, tag: u16, value: &T) -> Result<TlvWriter, TlvError> {
        let serialized = postcard::to_allocvec(value);
        self.push_serialized(tag, true, serialized)
    }

    /// Add an OPTIONAL (additive, skip-tolerant) field.
    pub fn optional<T: Serialize>(self, tag: u16, value: &T) -> Result<TlvWriter, TlvError> {
        let serialized = postcard::to_allocvec(value);
        self.push_serialized(tag, false, serialized)
    }

    fn push_serialized(
        mut self,
        tag: u16,
        required: bool,
        serialized: Result<Vec<u8>, postcard::Error>,
    ) -> Result<TlvWriter, TlvError> {
        if self.fields.contains_key(&tag) {
            return Err(TlvError::DuplicateTag(tag));
        }
        let bytes = serialized.map_err(|_| TlvError::FieldDecode { tag })?;
        check_field_len(bytes.len(), tag)?;
        self.fields.insert(tag, (required, bytes));
        Ok(self)
    }

    /// Serialize to canonical bytes.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        let writer_max_tag = self.fields.keys().max().copied().unwrap_or(0);
        let required_max_tag = self
            .fields
            .iter()
            .filter(|(_, (required, _))| *required)
            .map(|(tag, _)| *tag)
            .max()
            .unwrap_or(0);
        let mut out = Vec::with_capacity(
            HEADER_LEN
                + self
                    .fields
                    .values()
                    .map(|(_, b)| FIELD_PRELUDE_LEN + b.len())
                    .sum::<usize>(),
        );
        out.extend_from_slice(&self.schema.0.to_le_bytes());
        out.extend_from_slice(&writer_max_tag.to_le_bytes());
        out.extend_from_slice(&required_max_tag.to_le_bytes());
        // Field count is bounded by distinct u16 tags, so the cast is total.
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(self.fields.len() as u16).to_le_bytes());
        out.push(CODEC_FLAGS_V1);
        for (tag, (_, bytes)) in &self.fields {
            out.extend_from_slice(&tag.to_le_bytes());
            #[allow(clippy::cast_possible_truncation)] // length-checked in push()
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        out
    }
}

/// A field payload must fit [`MAX_FIELD_BYTES`] (which trivially fits the u32 length
/// prefix). Length-based so the guard is testable end-to-end with a ~1 MiB vector.
fn check_field_len(len: usize, tag: u16) -> Result<(), TlvError> {
    if len > MAX_FIELD_BYTES {
        return Err(TlvError::FieldTooLarge { tag });
    }
    Ok(())
}

/// A parsed blob: header + tag-indexed field payloads. Unknown tags were retained as
/// opaque bytes (a forwarding node re-emits them losslessly); readers simply never
/// ask for them.
#[derive(Debug)]
pub struct TlvReader<'a> {
    schema: SchemaId,
    writer_max_tag: u16,
    required_max_tag: u16,
    fields: BTreeMap<u16, &'a [u8]>,
}

impl<'a> TlvReader<'a> {
    /// Parse a blob, validating the header against the expected schema. Total: never
    /// panics on arbitrary input.
    pub fn parse(expected_schema: SchemaId, blob: &'a [u8]) -> Result<TlvReader<'a>, TlvError> {
        if blob.len() < HEADER_LEN {
            return Err(TlvError::Truncated(blob.len()));
        }
        let schema = SchemaId(u16::from_le_bytes([blob[0], blob[1]]));
        let writer_max_tag = u16::from_le_bytes([blob[2], blob[3]]);
        let required_max_tag = u16::from_le_bytes([blob[4], blob[5]]);
        let declared_count = u16::from_le_bytes([blob[6], blob[7]]);
        let codec_flags = blob[8];
        if schema != expected_schema {
            return Err(TlvError::SchemaMismatch {
                expected: expected_schema,
                got: schema,
            });
        }
        if codec_flags != CODEC_FLAGS_V1 {
            return Err(TlvError::UnsupportedCodec(codec_flags));
        }
        if required_max_tag > writer_max_tag {
            return Err(TlvError::IncoherentHeader {
                required: required_max_tag,
                max: writer_max_tag,
            });
        }

        let mut fields = BTreeMap::new();
        let mut at = HEADER_LEN;
        while at < blob.len() {
            if blob.len() - at < FIELD_PRELUDE_LEN {
                return Err(TlvError::Truncated(at));
            }
            let tag = u16::from_le_bytes([blob[at], blob[at + 1]]);
            let len = u32::from_le_bytes([blob[at + 2], blob[at + 3], blob[at + 4], blob[at + 5]])
                as usize;
            at += FIELD_PRELUDE_LEN;
            if blob.len() - at < len {
                return Err(TlvError::Truncated(at));
            }
            if fields.insert(tag, &blob[at..at + len]).is_some() {
                return Err(TlvError::DuplicateTag(tag));
            }
            at += len;
        }
        // Truncation at a field boundary is otherwise undetectable in TLV: the
        // declared count closes that hole.
        let found = fields.len();
        if usize::from(declared_count) != found {
            #[allow(clippy::cast_possible_truncation)] // found <= distinct u16 tags
            return Err(TlvError::FieldCountMismatch {
                declared: declared_count,
                found: found as u16,
            });
        }
        Ok(TlvReader {
            schema,
            writer_max_tag,
            required_max_tag,
            fields,
        })
    }

    #[must_use]
    pub fn schema(&self) -> SchemaId {
        self.schema
    }

    #[must_use]
    pub fn writer_max_tag(&self) -> u16 {
        self.writer_max_tag
    }

    /// The version-floor check (orchestrator-side at PREPARE, reader-side before
    /// decode): can a consumer that knows tags up to `max_known_tag` represent this
    /// blob's required state?
    #[must_use]
    pub fn floor_ok(&self, max_known_tag: u16) -> bool {
        max_known_tag >= self.required_max_tag
    }

    /// Monomorphic field lookup carrying the required/optional distinction; the
    /// generic decoders below stay branchless (see the coverage pattern note).
    fn bytes_required(&self, tag: u16) -> Result<&'a [u8], TlvError> {
        self.fields
            .get(&tag)
            .copied()
            .ok_or(TlvError::MissingRequiredTag(tag))
    }

    /// Read a REQUIRED field. Absence is a hard error — the decode-to-Default ban.
    pub fn required<T: DeserializeOwned>(&self, tag: u16) -> Result<T, TlvError> {
        self.bytes_required(tag)
            .and_then(|bytes| postcard::from_bytes(bytes).map_err(field_decode_err(tag)))
    }

    /// Read an OPTIONAL field; `None` when the writer (an older binary) omitted it.
    pub fn optional<T: DeserializeOwned>(&self, tag: u16) -> Result<Option<T>, TlvError> {
        self.fields
            .get(&tag)
            .map(|bytes| postcard::from_bytes(bytes).map_err(field_decode_err(tag)))
            .transpose()
    }
}

/// Monomorphic error constructor so generic decode shims carry no per-type closure.
fn field_decode_err(tag: u16) -> impl Fn(postcard::Error) -> TlvError {
    move |_| TlvError::FieldDecode { tag }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const SCHEMA: SchemaId = SchemaId(7);

    // The reader/writer pair for "version N" of a player blob.
    const TAG_POS: u16 = 1;
    const TAG_HEALTH: u16 = 2;
    // "Version N+1" adds these.
    const TAG_STAMINA_OPTIONAL: u16 = 3;
    const TAG_FACTION_REQUIRED: u16 = 4;

    fn v_n_blob() -> Vec<u8> {
        TlvWriter::new(SCHEMA)
            .required(TAG_POS, &[1.0f64, 2.0, 3.0])
            .expect("pos")
            .required(TAG_HEALTH, &100.0f32)
            .expect("health")
            .finish()
    }

    #[test]
    fn roundtrip_required_and_optional() {
        let blob = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &[1.0f64, 2.0, 3.0])
            .expect("pos")
            .optional(TAG_STAMINA_OPTIONAL, &0.5f32)
            .expect("stamina")
            .finish();
        let reader = TlvReader::parse(SCHEMA, &blob).expect("parse");
        assert_eq!(reader.schema(), SCHEMA);
        assert_eq!(
            reader.required::<[f64; 3]>(TAG_POS).expect("pos"),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            reader
                .optional::<f32>(TAG_STAMINA_OPTIONAL)
                .expect("stamina"),
            Some(0.5)
        );
        assert_eq!(reader.optional::<f32>(999).expect("absent optional"), None);
    }

    /// THE kind_blob_evolution gate (HR2): older reader, newer writer.
    #[test]
    fn older_reader_skips_newer_optional_tags_and_recovers_known_fields() {
        // Writer at version N+1 adds an OPTIONAL tag the version-N reader never heard of.
        let blob = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &[9.0f64, 8.0, 7.0])
            .expect("pos")
            .required(TAG_HEALTH, &55.0f32)
            .expect("health")
            .optional(TAG_STAMINA_OPTIONAL, &0.25f32)
            .expect("stamina")
            .finish();

        // The version-N reader: knows tags up to TAG_HEALTH only.
        let reader = TlvReader::parse(SCHEMA, &blob).expect("parse");
        assert!(
            reader.floor_ok(TAG_HEALTH),
            "optional additive evolution never refuses"
        );
        assert_eq!(
            reader.required::<[f64; 3]>(TAG_POS).expect("pos"),
            [9.0, 8.0, 7.0]
        );
        assert_eq!(reader.required::<f32>(TAG_HEALTH).expect("health"), 55.0);
        // The unknown tag was skipped, not misparsed: nothing else to assert — the
        // reads above would have failed had framing drifted.
    }

    #[test]
    fn older_reader_refuses_newer_required_tags_via_the_floor() {
        // Writer at version N+1 adds a REQUIRED tag.
        let blob = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &[0.0f64, 0.0, 0.0])
            .expect("pos")
            .required(TAG_FACTION_REQUIRED, &42u32)
            .expect("faction")
            .finish();
        let reader = TlvReader::parse(SCHEMA, &blob).expect("parse");
        // The version-N consumer (max known tag = TAG_HEALTH) cannot represent this
        // state: the floor check refuses BEFORE any decode — the transfer aborts and
        // the player stays alive on source.
        assert!(!reader.floor_ok(TAG_HEALTH));
        // And a reader that ignores the floor and asks for a missing required tag
        // gets a hard error, never a Default.
        assert_eq!(
            reader.required::<u32>(99).expect_err("missing required"),
            TlvError::MissingRequiredTag(99)
        );
    }

    #[test]
    fn canonical_bytes_regardless_of_insertion_order() {
        let a = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &1u8)
            .expect("a")
            .optional(TAG_STAMINA_OPTIONAL, &2u8)
            .expect("b")
            .finish();
        let b = TlvWriter::new(SCHEMA)
            .optional(TAG_STAMINA_OPTIONAL, &2u8)
            .expect("b")
            .required(TAG_POS, &1u8)
            .expect("a")
            .finish();
        assert_eq!(a, b, "blob bytes must be canonical for content-hash dedup");
    }

    #[test]
    fn duplicate_tags_rejected_on_write_and_parse() {
        let dup = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &1u8)
            .expect("first")
            .required(TAG_POS, &2u8)
            .map(|_| ())
            .expect_err("duplicate write");
        assert_eq!(dup, TlvError::DuplicateTag(TAG_POS));

        // Hand-craft a blob with a duplicated tag to exercise the parse-side check.
        let mut blob = v_n_blob();
        let dup_field: Vec<u8> = blob[HEADER_LEN..].to_vec();
        blob.extend_from_slice(&dup_field);
        let err = TlvReader::parse(SCHEMA, &blob)
            .map(|_| ())
            .expect_err("duplicate parse");
        assert_eq!(err, TlvError::DuplicateTag(TAG_POS));
    }

    #[test]
    fn duplicate_optional_tag_also_rejected() {
        let err = TlvWriter::new(SCHEMA)
            .optional(TAG_STAMINA_OPTIONAL, &1u8)
            .expect("first")
            .optional(TAG_STAMINA_OPTIONAL, &2u8)
            .map(|_| ())
            .expect_err("duplicate optional");
        assert_eq!(err, TlvError::DuplicateTag(TAG_STAMINA_OPTIONAL));
    }

    #[test]
    fn truncation_at_a_field_boundary_is_caught_by_the_count() {
        // Cut EXACTLY at the end of the first field: structurally valid TLV — only
        // the declared field count exposes the loss (the proptest's original find).
        let blob = v_n_blob();
        let reader = TlvReader::parse(SCHEMA, &blob).expect("full blob parses");
        assert_eq!(reader.writer_max_tag(), TAG_HEALTH);

        // Recompute the first field's full length to cut after it.
        let first_len = u32::from_le_bytes([
            blob[HEADER_LEN + 2],
            blob[HEADER_LEN + 3],
            blob[HEADER_LEN + 4],
            blob[HEADER_LEN + 5],
        ]) as usize;
        let cut = HEADER_LEN + FIELD_PRELUDE_LEN + first_len;
        let err = TlvReader::parse(SCHEMA, &blob[..cut])
            .map(|_| ())
            .expect_err("boundary truncation");
        assert_eq!(
            err,
            TlvError::FieldCountMismatch {
                declared: 2,
                found: 1
            }
        );
    }

    #[test]
    fn oversize_field_guard_is_exact() {
        assert_eq!(check_field_len(MAX_FIELD_BYTES, 5), Ok(()));
        assert_eq!(
            check_field_len(MAX_FIELD_BYTES + 1, 5),
            Err(TlvError::FieldTooLarge { tag: 5 })
        );
    }

    #[test]
    fn oversize_field_rejected_end_to_end() {
        // A Vec serializes with a length prefix, so this payload exceeds the cap.
        let big = vec![0u8; MAX_FIELD_BYTES];
        let err = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &big)
            .map(|_| ())
            .expect_err("oversize field");
        assert_eq!(err, TlvError::FieldTooLarge { tag: TAG_POS });
    }

    #[test]
    fn serialize_failure_is_an_error_not_a_panic() {
        // A type whose Serialize always fails: the writer surfaces FieldDecode.
        struct FailSer;
        impl serde::Serialize for FailSer {
            fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("deliberate"))
            }
        }
        let err = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &FailSer)
            .map(|_| ())
            .expect_err("serialize failure");
        assert_eq!(err, TlvError::FieldDecode { tag: TAG_POS });
        let err_opt = TlvWriter::new(SCHEMA)
            .optional(TAG_POS, &FailSer)
            .map(|_| ())
            .expect_err("serialize failure via optional");
        assert_eq!(err_opt, TlvError::FieldDecode { tag: TAG_POS });
    }

    #[test]
    fn schema_mismatch_and_bad_codec_rejected() {
        let blob = v_n_blob();
        assert_eq!(
            TlvReader::parse(SchemaId(8), &blob).expect_err("schema"),
            TlvError::SchemaMismatch {
                expected: SchemaId(8),
                got: SCHEMA
            }
        );
        let mut bad_codec = v_n_blob();
        bad_codec[8] = 0xFF;
        assert_eq!(
            TlvReader::parse(SCHEMA, &bad_codec).expect_err("codec"),
            TlvError::UnsupportedCodec(0xFF)
        );
    }

    #[test]
    fn incoherent_header_rejected() {
        // required_max_tag (bytes 4..6) above writer_max_tag (bytes 2..4).
        let mut blob = v_n_blob();
        blob[4] = 0xFF;
        blob[5] = 0xFF;
        let err = TlvReader::parse(SCHEMA, &blob)
            .map(|_| ())
            .expect_err("incoherent header");
        assert_eq!(
            err,
            TlvError::IncoherentHeader {
                required: 0xFFFF,
                max: TAG_HEALTH
            }
        );
    }

    #[test]
    fn field_decode_failure_is_an_error_not_a_default() {
        let blob = TlvWriter::new(SCHEMA)
            .required(TAG_POS, &1u8)
            .expect("write")
            .finish();
        let reader = TlvReader::parse(SCHEMA, &blob).expect("parse");
        // Asking for a [f64; 3] where a u8 was written: hard error.
        assert_eq!(
            reader
                .required::<[f64; 3]>(TAG_POS)
                .expect_err("type clash"),
            TlvError::FieldDecode { tag: TAG_POS }
        );
        assert_eq!(
            reader
                .optional::<[f64; 3]>(TAG_POS)
                .expect_err("type clash optional"),
            TlvError::FieldDecode { tag: TAG_POS }
        );
    }

    #[test]
    fn empty_blob_has_zero_tags_and_parses() {
        let blob = TlvWriter::new(SCHEMA).finish();
        let reader = TlvReader::parse(SCHEMA, &blob).expect("parse");
        assert_eq!(reader.writer_max_tag(), 0);
        assert!(reader.floor_ok(0));
    }

    proptest! {
        /// Parsing is TOTAL: any truncation of a valid blob is an error, never a panic
        /// and never a silent partial decode.
        #[test]
        fn truncation_at_any_point_errors_cleanly(cut in 0usize..64) {
            let blob = v_n_blob();
            prop_assume!(cut < blob.len());
            let truncated = &blob[..cut];
            prop_assert!(TlvReader::parse(SCHEMA, truncated).is_err());
        }

        /// Arbitrary bytes never panic the parser.
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
            let _ = TlvReader::parse(SCHEMA, &bytes);
        }

        /// Values roundtrip through required fields for arbitrary payloads.
        #[test]
        fn arbitrary_payload_roundtrips(v in proptest::collection::vec(any::<u64>(), 0..16)) {
            let blob = TlvWriter::new(SCHEMA)
                .required(TAG_POS, &v)
                .expect("write")
                .finish();
            let reader = TlvReader::parse(SCHEMA, &blob).expect("parse");
            prop_assert_eq!(reader.required::<Vec<u64>>(TAG_POS).expect("read"), v);
        }
    }
}
