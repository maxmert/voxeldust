//! THE WINDOW LANE's body-bag codec (`docs/design/window_lane.md` §2.2/§2.10; Slice A).
//!
//! A window body statement carries a canonical [`crate::tlv`] blob — the "bag of signals" the
//! VU streaming contract promises: tags grow additively forever, unknown tags are skipped, and
//! the client never branches on realm kind. This module names the ONE schema and the two tags
//! the lane speaks today, and the two encode/decode pairs every producer and consumer shares,
//! so a bag can never be framed two ways:
//!
//! - **`TAG_LOOK`** — the subject's OWN outline ([`Boundary`]), stated only by the realm itself
//!   (SL3: a realm draws itself). It carries NO position, by its very shape — a look bag has no
//!   field a position could ride in.
//! - **`TAG_LUMA`** — the parent-authored photometric point-of-light datum for a SLEEPING direct
//!   child (owner R4 ruling): a spectral-class code plus a luminosity scalar, drawn from the same
//!   seed stream that generated the child (`vd-physics` worldgen's per-system draw). NO look
//!   field exists here — a marker can never carry an outline.

use crate::entity_kind::SchemaId;
use crate::geometry::Boundary;
use crate::tlv::{TlvError, TlvReader, TlvWriter};

/// The window-body bag schema (one schema for both statement kinds — the KIND rides the wire
/// enum [`BodyStmt`](https://docs.rs/vd-wire) discriminant, never a bag field). Allocated OUTSIDE
/// the entity-kind blob block (kinds hold 1–3 and 10–13); the registry of taken ids is the
/// `entity_kind` kind table plus this constant.
pub const WINDOW_BODY_SCHEMA: SchemaId = SchemaId(20);

/// The self-look outline tag: payload = one postcard [`Boundary`].
pub const TAG_LOOK: u16 = 1;

/// The marker photometrics tag: payload = one postcard `(class_code: u8, luma_lsun: f64)` pair —
/// brightness and color, exactly the two scalars the owner-ruled marker datum names (§1.1 item 3b).
pub const TAG_LUMA: u16 = 2;

/// Encode a realm's OWN outline as a `TAG_LOOK` bag. The outline is the realm's one geometric
/// fact about itself (its boot-config extent) — never its position, which no field here can hold.
#[must_use]
pub fn look_bag(outline: &Boundary) -> Vec<u8> {
    TlvWriter::new(WINDOW_BODY_SCHEMA)
        .required(TAG_LOOK, outline)
        .expect("a fresh writer holds no duplicate tag and a Boundary is far under the field cap")
        .finish()
}

/// Encode a sleeping child's photometric marker datum as a `TAG_LUMA` bag.
#[must_use]
pub fn luma_bag(class_code: u8, luma_lsun: f64) -> Vec<u8> {
    TlvWriter::new(WINDOW_BODY_SCHEMA)
        .required(TAG_LUMA, &(class_code, luma_lsun))
        .expect("a fresh writer holds no duplicate tag and two scalars are far under the field cap")
        .finish()
}

/// Decode a `TAG_LOOK` bag back to its outline.
///
/// # Errors
/// [`TlvError`] if the blob is not a well-formed window-body bag carrying `TAG_LOOK`.
pub fn look_of(bag: &[u8]) -> Result<Boundary, TlvError> {
    TlvReader::parse(WINDOW_BODY_SCHEMA, bag)?.required(TAG_LOOK)
}

/// Decode a `TAG_LUMA` bag back to its `(class_code, luma_lsun)` pair.
///
/// # Errors
/// [`TlvError`] if the blob is not a well-formed window-body bag carrying `TAG_LUMA`.
pub fn luma_of(bag: &[u8]) -> Result<(u8, f64), TlvError> {
    TlvReader::parse(WINDOW_BODY_SCHEMA, bag)?.required(TAG_LUMA)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn a_look_bag_roundtrips_its_outline_and_nothing_else() {
        // Both boundary kinds a boot roster states today — a dropped or re-tagged field cannot
        // pass as a lucky default because the values are non-zero and shape-distinct.
        let shell = Boundary::Shell { r: 1234.5 };
        let aabb = Boundary::Aabb {
            half: DVec3::new(3.0, 4.0, 5.0),
        };
        assert_eq!(look_of(&look_bag(&shell)), Ok(shell));
        assert_eq!(look_of(&look_bag(&aabb)), Ok(aabb));
        // A look bag is NOT a luma bag: asking it for TAG_LUMA is the missing-required-tag error,
        // never a silent default (the decode-to-Default ban).
        assert_eq!(
            luma_of(&look_bag(&shell)),
            Err(TlvError::MissingRequiredTag(TAG_LUMA))
        );
    }

    #[test]
    fn a_luma_bag_roundtrips_its_scalars_and_never_carries_a_look() {
        let bag = luma_bag(6, 0.0009726074241780799);
        assert_eq!(luma_of(&bag), Ok((6, 0.0009726074241780799)));
        assert_eq!(
            look_of(&bag),
            Err(TlvError::MissingRequiredTag(TAG_LOOK)),
            "a marker can never carry an outline — the third pixel source is unrepresentable"
        );
    }

    #[test]
    fn a_foreign_schema_blob_is_refused_not_guessed_at() {
        // A blob framed under another schema (an entity kind's blob, say) is refused loudly.
        let foreign = TlvWriter::new(SchemaId(1))
            .required(TAG_LOOK, &Boundary::Shell { r: 1.0 })
            .expect("fresh writer")
            .finish();
        assert_eq!(
            look_of(&foreign),
            Err(TlvError::SchemaMismatch {
                expected: WINDOW_BODY_SCHEMA,
                got: SchemaId(1)
            })
        );
        assert_eq!(
            luma_of(&foreign),
            Err(TlvError::SchemaMismatch {
                expected: WINDOW_BODY_SCHEMA,
                got: SchemaId(1)
            })
        );
    }

    #[test]
    fn unknown_future_tags_are_skipped_the_bag_grows_forever() {
        // The additive-forever promise: a NEWER writer appends a tag this reader has never heard
        // of; the reader still finds its own tag and skips the stranger by length.
        let grown = TlvWriter::new(WINDOW_BODY_SCHEMA)
            .required(TAG_LOOK, &Boundary::Shell { r: 9.0 })
            .expect("fresh writer")
            .optional(999, &42u32)
            .expect("distinct tag")
            .finish();
        assert_eq!(look_of(&grown), Ok(Boundary::Shell { r: 9.0 }));
    }
}
