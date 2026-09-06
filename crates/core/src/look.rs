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
use crate::pose::RealmId;
use crate::tlv::{TlvError, TlvReader, TlvWriter};
use glam::I64Vec3;
use serde::{Deserialize, Serialize};

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

/// The marker EXTENT tag (look_horizon.md slice 1; owner Q2 APPROVED 2026-08-17): payload = one
/// postcard `f64` — the child's CIRCUMSCRIBED radius in metres, the number the parent already
/// holds for containment and already reads for its own proxy logic.
///
/// THE ONE-RADIUS LAW (owner's ruling, verbatim): **a bound is a promise about space; a look is a
/// statement about appearance.** The parent's point-of-light bag may never carry more than this
/// single radius — no surface, no detail, no mesh, no second number — and the realm's own picture
/// supersedes it the instant the realm runs (THE DRAW LAW's presence gate: a self-look beats a
/// marker by data presence, never by a flag). The radius exists so a sleeping or lapsed thing
/// still draws as a CORRECTLY-SIZED point of light instead of vanishing (a station, a city, a
/// ship — the non-glowing subjects that had no lawful bag content at all before this tag).
pub const TAG_EXTENT: u16 = 3;

/// ★ THE LIMITING MAGNITUDE (the reach, owner ruling 2026-09-02 R3, built 2026-09-04): the faintest
/// apparent visual magnitude a dark-adapted eye still sees under a dark sky, 6.5 — the perception
/// constant beside the dot angle ([`crate::geometry::VISIBILITY_THETA_MIN_RAD`]). A body whose light
/// is fainter than this at the observer is not seen, whatever its size makes of it.
pub const LIMITING_MAGNITUDE: f64 = 6.5;
/// The Sun's absolute visual magnitude, the zero of the luminosity scale the world's `luma_lsun`
/// datum is stated in.
pub const SUN_ABSOLUTE_MAGNITUDE: f64 = 4.83;
/// One parsec in metres — the unit the magnitude–distance relation is written in.
pub const PARSEC_M: f64 = 3.085_677_581_491_367e16;

/// ★ THE OCCUPANT'S FIGURE (foundation slice 2, 2026-09-05): the CIRCUMSCRIBED extent — the radius,
/// as every look's extent in this crate is — of the one figure every occupant draws as today: a point
/// of light one metre across. ONE home for the two readers: the renderer's point radius IS it, and a
/// shard's interest reach reads it through the same formula a realm's reach uses (`visibility_reach_m`,
/// at the drawable angle). A character with a body of its own will state its extent per entity, as a
/// hull states its rating; until then every dot is born with this one. Example: at this extent and the
/// drawable angle (one pixel at the reference view), a lone occupant is shipped to an observer inside
/// about 870 m, plus the lead for the closing speed.
pub const OCCUPANT_FIGURE_EXTENT_M: f64 = 0.5;

/// ★ THE REACH BY LIGHT: how far a body of luminosity `luma_lsun` (Suns; a planet's is its REFLECTED
/// light, as the world states it) is still visible at `limiting_magnitude`. The magnitude–distance
/// relation: `M = M_sun − 2.5·log10(L)`, `d = 10 pc · 10^((m_lim − M)/5)`. Zero for no light. Example:
/// a Sun-like star reaches about 70 light-years; an Earth-like planet about 50 astronomical units.
#[must_use]
pub fn light_reach_m(luma_lsun: f64, limiting_magnitude: f64) -> f64 {
    if luma_lsun <= 0.0 {
        return 0.0;
    }
    let absolute = SUN_ABSOLUTE_MAGNITUDE - 2.5 * luma_lsun.log10();
    10.0 * 10.0_f64.powf((limiting_magnitude - absolute) / 5.0) * PARSEC_M
}

/// Encode a realm's OWN outline as a `TAG_LOOK` bag. The outline is the realm's one geometric
/// fact about itself (its boot-config extent) — never its position, which no field here can hold.
#[must_use]
pub fn look_bag(outline: &Boundary) -> Vec<u8> {
    TlvWriter::new(WINDOW_BODY_SCHEMA)
        .required(TAG_LOOK, outline)
        .expect("a fresh writer holds no duplicate tag and a Boundary is far under the field cap")
        .finish()
}

/// ★ THE STAR-LOOK EXTENSION SEAM (celestial taxonomy arc T2; owner ruling C, 2026-08-19): a
/// RUNNING realm's OWN look bag may carry `TAG_LUMA` beside its outline — colour and brightness
/// — so a star KEEPS its colour through the marker→body wake handover instead of turning grey
/// the instant it runs. The owner explicitly opens FUTURE star parameters (radiation, flare
/// state, corona…) as FUTURE TAGS on this same bag: the TLV codec's skip-unknown law makes
/// every such append free — no `PROTO_MINOR`, no client change, ever. ONE builder so a
/// luma-bearing and a plain self-look can never be framed two ways (tags ride ascending:
/// `TAG_LOOK` then `TAG_LUMA`).
#[must_use]
pub fn self_look_bag(outline: &Boundary, luma: Option<(u8, f64)>) -> Vec<u8> {
    let writer = TlvWriter::new(WINDOW_BODY_SCHEMA)
        .required(TAG_LOOK, outline)
        .expect("a fresh writer holds no duplicate tag and a Boundary is far under the field cap");
    let writer = match luma {
        Some(datum) => writer
            .required(TAG_LUMA, &datum)
            .expect("distinct tag, two scalars"),
        None => writer,
    };
    writer.finish()
}

/// Encode a sleeping child's photometric marker datum as a `TAG_LUMA` bag.
#[must_use]
pub fn luma_bag(class_code: u8, luma_lsun: f64) -> Vec<u8> {
    TlvWriter::new(WINDOW_BODY_SCHEMA)
        .required(TAG_LUMA, &(class_code, luma_lsun))
        .expect("a fresh writer holds no duplicate tag and two scalars are far under the field cap")
        .finish()
}

/// Encode THE point-of-light bag (look_horizon.md slice 1): the child's circumscribed extent —
/// always, the presence floor's whole point — plus its photometric datum when the child glows.
/// One builder for every marker producer, so a glowing and a non-glowing child's bags can never
/// be framed two ways. Tags ride in ascending order (`TAG_LUMA` then `TAG_EXTENT`), the codec's
/// canonical shape.
#[must_use]
pub fn marker_bag(luma: Option<(u8, f64)>, extent_m: f64) -> Vec<u8> {
    let writer = TlvWriter::new(WINDOW_BODY_SCHEMA);
    let writer = match luma {
        Some(datum) => writer
            .required(TAG_LUMA, &datum)
            .expect("a fresh writer holds no duplicate tag and two scalars are under the cap"),
        None => writer,
    };
    writer
        .required(TAG_EXTENT, &extent_m)
        .expect("distinct tag, one scalar")
        .finish()
}

/// Decode a marker bag's `TAG_EXTENT` back to the circumscribed radius (metres).
///
/// # Errors
/// [`TlvError`] if the blob is not a well-formed window-body bag carrying `TAG_EXTENT` — an
/// old luma-only bag decodes to `MissingRequiredTag` here, which callers treat as "no stated
/// extent" (additive-forever: absence of the tag is absence of the datum, never a default).
pub fn extent_of(bag: &[u8]) -> Result<f64, TlvError> {
    TlvReader::parse(WINDOW_BODY_SCHEMA, bag)?.required(TAG_EXTENT)
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
    fn the_reach_by_light_is_seventy_light_years_for_a_sun_and_fifty_au_for_an_earth() {
        const LIGHT_YEAR_M: f64 = 9.460_730_472_580_8e15;
        const AU_M: f64 = 1.495_978_707e11;
        let sun = light_reach_m(1.0, LIMITING_MAGNITUDE) / LIGHT_YEAR_M;
        assert!((sun - 70.5).abs() < 1.0, "a Sun-like star: {sun} ly");
        // An Earth-like planet as the world states it: albedo 0.3, radius 6.4e6 m, at 1 AU.
        let earth_luma = 0.3 * 6.4e6_f64.powi(2) / (4.0 * AU_M * AU_M);
        let earth = light_reach_m(earth_luma, LIMITING_MAGNITUDE) / AU_M;
        assert!(
            (earth - 51.0).abs() < 3.0,
            "an Earth-like planet: {earth} AU"
        );
        assert_eq!(
            light_reach_m(0.0, LIMITING_MAGNITUDE),
            0.0,
            "no light, no reach"
        );
        assert_eq!(light_reach_m(-1.0, LIMITING_MAGNITUDE), 0.0);
    }

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
    fn a_marker_bag_carries_the_extent_always_and_the_datum_only_when_glowing() {
        // The glowing shape: datum + extent, both decodable, still no look (the third pixel
        // source stays unrepresentable).
        let glowing = marker_bag(Some((4, 1.0)), 150.0);
        assert_eq!(luma_of(&glowing), Ok((4, 1.0)));
        assert_eq!(extent_of(&glowing), Ok(150.0));
        assert_eq!(
            look_of(&glowing),
            Err(TlvError::MissingRequiredTag(TAG_LOOK))
        );
        // The non-glowing shape (the presence floor): ONE radius and nothing else — the
        // one-radius law is the codec's own shape, not a discipline.
        let quiet = marker_bag(None, 40.0);
        assert_eq!(extent_of(&quiet), Ok(40.0));
        assert_eq!(luma_of(&quiet), Err(TlvError::MissingRequiredTag(TAG_LUMA)));
        assert_eq!(look_of(&quiet), Err(TlvError::MissingRequiredTag(TAG_LOOK)));
        // An OLD luma-only bag states no extent (absence of the tag is absence of the datum,
        // never a default) — additive-forever holds in both directions.
        assert_eq!(
            extent_of(&luma_bag(6, 0.25)),
            Err(TlvError::MissingRequiredTag(TAG_EXTENT))
        );
        // And a look bag is not a marker bag.
        assert_eq!(
            extent_of(&look_bag(&Boundary::Shell { r: 9.0 })),
            Err(TlvError::MissingRequiredTag(TAG_EXTENT))
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

    /// THE STAR-LOOK EXTENSION SEAM (owner ruling C, 2026-08-19): a RUNNING realm's own look bag
    /// may carry `TAG_LUMA` beside its outline, and a reader that only knows `TAG_LOOK` still
    /// reads the outline — skip-unknown, so future star parameters are future tags and no
    /// version moves. Both arms of the one builder are measured here.
    #[test]
    fn a_self_look_bag_carries_its_own_luma_when_the_realm_states_one() {
        let outline = Boundary::Shell { r: 7.0 };
        let plain = self_look_bag(&outline, None);
        assert_eq!(plain, look_bag(&outline));
        assert_eq!(look_of(&plain), Ok(outline));
        let lit = self_look_bag(&outline, Some((6, 0.5)));
        assert_eq!(look_of(&lit), Ok(outline));
        assert_eq!(luma_of(&lit), Ok((6, 0.5)));
        assert_eq!(luma_of(&plain), Err(TlvError::MissingRequiredTag(TAG_LUMA)));
        assert_ne!(lit, plain);
    }
}

/// ONE STAR IN THE CATALOGUE (S11; owner-approved 2026-08-24 Q4, the compact-message reversal).
///
/// ★ WHY THIS IS NOT A a picture row. The owner's own argument, and it overturned the design's
/// reuse-the-existing-shape recommendation: the picture row is composed PER OBSERVER, PER INSTANT,
/// ready to draw. A catalogue is neither — it is the same for every player and true until the world
/// is regenerated. Reusing the picture row would mean BENDING it, which is more work and more risk
/// than a small purpose-built one.
///
/// **MEASURED before it was written:** ~47 bytes against a `RealmSnap`'s 95 — 7.0 MB against 14.2 MB
/// at the target census of 150,000 systems.
///
/// **NO SUB-CELL PART, and that is a measurement not an assumption:** every generated system's centre
/// carries `offset = (0,0,0)` at the galaxy's own two-metre step, so the integer cell IS the position.
/// A row that carried a residual would be carrying three zeroes 150,000 times.
///
/// **THE IDENTITY IS CARRIED BECAUSE OF A SEAM (SL8), not for tidiness.** Fly toward a star and it
/// wakes: the client then holds a catalogue POINT and a LIVE REALM for the same star. Unable to tell
/// they are one thing it either draws both — a double image — or swaps them, which is a POP at exactly
/// the moment of arrival. The identity is what lets the point be suppressed as the real thing arrives.
///
/// ★ WHY IT LIVES IN THE CORE AND NOT THE WIRE. The generator builds it and the generator crate may
/// not see the wire crate (the dependency rule runs one way: physics → core, sim → wire → core).
/// Putting it here is what lets ONE fold produce it and the wire carry it without a second shape —
/// exactly how the marker datum already crosses that boundary, as plain scalars.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StarRow {
    /// Which star system this is — the key that lets a live realm replace its own point silently.
    pub realm: RealmId,
    /// Its centre in the GALAXY's frame, as whole cells of that frame's own step. The galaxy is the
    /// only frame a catalogue is ever stated in, so the frame is not repeated 150,000 times.
    pub cell: I64Vec3,
    /// The spectral class as its stable code — the colour to draw it (`marker_datum`'s first scalar).
    pub class_code: u8,
    /// Main-sequence luminosity in solar units — the brightness (`marker_datum`'s second scalar).
    pub luma_lsun: f64,
}

/// THE CATALOGUE'S GENERATION — **derived from the content, never hand-incremented** (S11; the owner's
/// THIRD condition on this message, 2026-08-24 Q4).
///
/// ★ WHY A PERSON MAY NOT TYPE THIS NUMBER. The client caches the sky on disk and asks for it by
/// version: *"I hold generation X"*. If a person increments it, a person eventually forgets to — and a
/// stale cache then looks current forever. The player draws last week's galaxy, flies at a star that has
/// moved, and NOTHING reports a fault, because every part believes it is in agreement. Deriving it from
/// the bytes makes "the content changed but the version did not" unrepresentable rather than unlikely.
///
/// Folded over the ENCODED rows, not the values: the encoding is what the client stores and compares, so
/// a field added or reordered must move this number even when every value still compares equal.
///
/// A collision would let a stale cache pass as current. That is a content digest's known cost, and it is
/// stated here rather than assumed away; see [`crate::digest`].
#[must_use]
pub fn catalogue_generation(encoded: &[u8]) -> u64 {
    crate::digest::fnv1a(crate::digest::FNV_OFFSET, encoded)
}
