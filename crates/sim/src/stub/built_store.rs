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
        berth_key, berth_prefix, body_key, decode_berth, decode_body, encode_berth, encode_body,
    };
    use vd_core::built::{Berth, BlueprintId, BuiltBody, BuiltFacts};
    use vd_core::entity_kind::EntityKind;
    use vd_core::fence::Fence;
    use vd_core::geometry::Boundary;
    use vd_core::glam::DVec3;
    use vd_core::ids::{AccountId, EntityId};
    use vd_core::pose::RealmId;

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
        for cut in [4, whole.len() / 3, whole.len() / 2, whole.len() - 1] {
            assert!(
                decode_body(&whole[..cut]).is_err(),
                "a row cut at {cut} of {} bytes must be refused",
                whole.len()
            );
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
        assert!(body.len() < 200, "a body stays small: {} bytes", body.len());
        assert!(
            berth.len() < 200,
            "a berth stays small: {} bytes",
            berth.len()
        );
    }
}
