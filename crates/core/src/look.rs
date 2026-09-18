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

/// ★ THE SURFACE TAG (the voxel foundation, slice 4; SL6 row R-8, owner YES, ruling V6): the realm's
/// OWN statement that it has a seed-shaped surface, and which generator shapes it. Payload:
/// [`SurfaceStmt`]. Stated by the realm about ITSELF (it rides `BodyStmt::SelfLook`, which only a
/// RUNNING realm may send — so a dormant realm is still never drawn), ONCE PER REALM ON CHANGE, carried
/// retained, never on a keep-alive. A hull states no surface: it has none. A reader that knows only
/// `TAG_LOOK` skips it (the skip-unknown rule). The first producer is slice 5, the generator.
pub const TAG_SURFACE: u16 = 4;

/// ★ THE BODIES TAG (slice 4 plant; SL6 row R-11, owner YES): the poses of a realm's own rigid bodies —
/// a turret, a door, a piston carriage — INSIDE the realm's own row, under that row's one stamp, so
/// the one-stamp rule never falls back into care. The payload is slice 13's (the bodies); until then
/// no producer writes it and no reader asks for it. The number is reserved here so it can never be
/// taken by another meaning.
pub const TAG_BODIES: u16 = 5;

/// ★ THE CHARTER TAG (the landform arc, slice 8b stage 1; ruling V13 L12, crossing A1 approved):
/// the realm's OWN statement of its BODY'S PHYSICAL FACTS as whole numbers — [`BodyCharter`].
/// Payload: one postcard [`BodyCharter`].
///
/// **Why it exists.** The client holds a planet's seed and its shell radius and nothing else, and it
/// may not link the motion crate that holds the planet's mass (SL4). So it cannot derive the gravity
/// the relief law reads, nor the insolation the biome reads. The realm states the integers instead,
/// on the bag that already carries its surface tag, and both hosts then read THE SAME INTEGERS
/// (SL10). It is a realm's statement ABOUT ITSELF, to the clients the gateway composes for — never a
/// parent's per-child message, which carries a placement and nothing else (SL3).
///
/// **It opens no wire arm.** A new tag in this bag is not a new arm: the codec skips what it does not
/// know, so a reader that knows only `TAG_LOOK` still reads the outline (the additive-forever rule).
///
/// Example: the home planet states its gravity in whole mm/s², once, on change; every client that
/// sees the planet then holds the same integer the planet's own shard holds.
pub const TAG_CHARTER: u16 = 6;

/// ★ THE BODY CHARTER (ruling V13 L12: about twenty quantised integers, authored once and stored;
/// the landform arc's `slice_8b_design.md` §4.1) — a round body's physical facts as WHOLE NUMBERS.
///
/// **THE INTEGER IS THE FACT.** The body's own realm computes each number once, with full precision,
/// and floors it into the unit stated below. Nothing downstream ever re-derives it, and no float of
/// it ever crosses. That is what lets the server's CPU, the client's CPU and the client's card agree
/// byte for byte on the shape they build from it (SL10).
///
/// **A WORD WITH NO AUTHOR YET IS `None`, NEVER ZERO.** Nine of the twenty words are drawn by later
/// stages of the arc (the spin, the tilt, the two optical depths, the surface pressure, the surface
/// temperature, the elastic thickness, the water inventory and the sea). A zero in their place would
/// be read as a fact nobody authored — a sea at the ladder radius nobody solved. They are stated
/// ABSENT, exactly as the bag states an absent tag, and the decode-to-Default ban says the same thing
/// one level up.
///
/// **The whole record is carried NOW, before any kernel reads most of it** — the design's ask 7
/// recommendation, taken as ASSUMED pending the owner's answer. Appending a word is free while the
/// ground may still move (ruling T1's free window) and impossible after the freeze (slice 14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyCharter {
    /// Surface gravity, whole mm/s². One step is 1 part in 9 821 on the home planet.
    pub gravity_mm_s2: u32,
    /// Bulk density `M / ((4/3)πR³)`, whole kg/m³.
    pub bulk_density_kgm3: u32,
    /// Escape velocity, whole m/s.
    pub escape_velocity_mps: u32,
    /// Insolation relative to Earth's, in 1/4096 S⊕.
    pub insolation_q12: u32,
    /// Equilibrium temperature, whole millikelvin.
    pub t_eq_mk: u32,
    /// Mean SURFACE temperature, whole millikelvin. Drawn by stage 4's greenhouse (the thermostat,
    /// the design's ask 2 taken as ASSUMED); read by no kernel until 8e.
    pub t_surface_mk: Option<u32>,
    /// Bond albedo, in 1/4096.
    pub bond_albedo_q12: u32,
    /// The atmosphere's mean molecular weight, in 1/256 atomic units. ABSENT on an airless body.
    pub mu_q8: Option<u32>,
    /// The atmosphere's isothermal scale height, whole metres. ABSENT on an airless body.
    pub scale_height_m: Option<u32>,
    /// Surface pressure, whole pascals. DERIVED (ruling T9): Earth's, scaled by the mass, the
    /// gravity and the inverse square of the radius, where the shoreline keeps the air.
    pub p_surf_pa: Option<u32>,
    /// The Rayleigh optical depth at 550 nm, in 1/4096. Drawn by stage 4; the sky (8s) reads it.
    pub tau_vis_q12: Option<u32>,
    /// The grey greenhouse optical depth, in 1/4096. Drawn by stage 4; the climate (8c) reads it.
    pub tau_ir_q12: Option<u32>,
    /// The rotation period, whole seconds. Drawn by stage 4 under tidal locking (a locked body's
    /// day is its year, and `CHARTER_FLAG_TIDALLY_LOCKED` says so).
    pub day_s: Option<u32>,
    /// The COSINE of the obliquity, in 1/1024 — never an angle, because the fence bans `sin` and
    /// `cos`. Drawn and damped by stage 4; its prior is the design's ask 4, taken as ASSUMED.
    pub obliquity_cos_q1024: Option<i32>,
    /// The body's whole water inventory, whole cubic kilometres. ABSENT until stage 5 computes it
    /// from the formation zone and the two retention verdicts.
    pub water_km3: Option<u64>,
    /// The sea's offset from the ladder radius, whole millimetres. Solved by the owning realm at boot
    /// (stage 6) where the water is liquid at the surface; ABSENT where it is not. Until 8c gives the
    /// ground its second hump the recipe does not read it (ruling T8: the pictures are judged dry).
    /// Before stage 6 the doc read: ABSENT until stage 6 solves it by
    /// bisection of the water volume over the shape; absent FOREVER on a body whose condensable is
    /// not liquid at the surface (an ice world has water and no coast).
    pub sea_offset_mm: Option<i32>,
    /// The lithosphere's effective elastic thickness, whole metres. Drawn by stage 4; the solve
    /// (8c) reads it. ABSENT on a body with no solid surface.
    pub elastic_thickness_m: Option<u32>,
    /// The orbit's eccentricity, in 1/65536.
    pub ecc_q16: u32,
    /// The orbital period, whole seconds.
    pub year_s: u64,
    /// The flag word — see [`CHARTER_FLAG_HAS_AIR`], [`CHARTER_FLAG_SOLID_SURFACE`],
    /// [`CHARTER_FLAG_TIDALLY_LOCKED`], [`CHARTER_FLAG_HAS_SEA`] and [`charter_star_class_code`].
    /// Only the bits whose author exists today are defined; a bit a later stage authors (the
    /// condensable) is NOT a bit of this word yet, because a clear bit would be read as a fact
    /// nobody set.
    pub flags: u32,
}

/// The charter flag: the body holds an atmosphere (the census's own shoreline-and-envelope verdict).
pub const CHARTER_FLAG_HAS_AIR: u32 = 1;
/// The charter flag: the body has a surface you can stand on (it is not a giant).
pub const CHARTER_FLAG_SOLID_SURFACE: u32 = 2;
/// The charter flag: the body is tidally locked — its day is its year, one face to its star
/// (authored by the spin law of slice 8b stage 4; a permanent day side and a permanent night side).
pub const CHARTER_FLAG_TIDALLY_LOCKED: u32 = 4;
/// The charter flag: the body HAS A SEA — its water is liquid at its surface and the owning realm
/// solved the level it stands at (`sea_offset_mm`; slice 8b stage 6). Clear on a dry world, an ice
/// world (water, but frozen) and a steam world.
pub const CHARTER_FLAG_HAS_SEA: u32 = 8;
/// Where the illuminating star's class code sits in [`BodyCharter::flags`] — the same code the
/// photometric datum states (`SpectralClass as u8`), so one meaning has one encoding.
pub const CHARTER_STAR_CLASS_SHIFT: u32 = 8;
/// The mask of [`CHARTER_STAR_CLASS_SHIFT`]'s field (seven classes fit in three bits; four bits are
/// kept so a later class table cannot overflow it).
pub const CHARTER_STAR_CLASS_MASK: u32 = 0xF << CHARTER_STAR_CLASS_SHIFT;

/// The illuminating star's class code, read out of a charter's flag word.
#[must_use]
pub fn charter_star_class_code(flags: u32) -> u8 {
    let code = (flags & CHARTER_STAR_CLASS_MASK) >> CHARTER_STAR_CLASS_SHIFT;
    u8::try_from(code).unwrap_or(u8::MAX)
}

/// What [`TAG_SURFACE`] carries: the realm's own frame (the seed is inside it for a seed-shaped
/// realm) and the DECLARED generator tag of the build that runs it. A client compares the tag with its
/// own before it derives one chunk; a mismatch refuses THIS realm's surface and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceStmt {
    pub frame: crate::pose::FrameRef,
    /// The declared half of the world identity (Format D): the generator crate's tag. Never the
    /// measured half — a chip difference refuses a lane, never a look.
    pub generator: u64,
}

/// The budget a realm's whole self-look bag must fit, with its outline, its luma and its surface
/// beside each other: one conservative datagram (`vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET`,
/// stated here as the same number because core cannot name wire).
pub const SELF_LOOK_BUDGET_BYTES: usize = 1200;

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
    self_look_writer(outline, luma).finish()
}

/// THE ONE PROLOGUE of every self-look: the outline, then the luma if any. Both self-look builders
/// go through here, so a self-look can never be framed two ways.
fn self_look_writer(outline: &Boundary, luma: Option<(u8, f64)>) -> TlvWriter {
    let writer = TlvWriter::new(WINDOW_BODY_SCHEMA)
        .required(TAG_LOOK, outline)
        .expect("a fresh writer holds no duplicate tag and a Boundary is far under the field cap");
    match luma {
        Some(datum) => writer
            .required(TAG_LUMA, &datum)
            .expect("distinct tag, two scalars"),
        None => writer,
    }
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

/// A realm's self-look with its surface beside the outline and the optional luma, and its CHARTER
/// beside the surface: the bag a seed-shaped realm states once on change (R-8; slice 8b A1). ONE
/// builder, so a charter-bearing and a plain surface bag can never be framed two ways. Tags ride
/// ascending — `TAG_LOOK`, `TAG_LUMA`, `TAG_SURFACE`, `TAG_CHARTER`.
///
/// A realm that cannot derive its charter states NO surface (slice 8b §4.2 rule 2), so in the shipped
/// path the two arrive together; the `None` arm is what a rig that states a bare surface produces,
/// and the client refuses it and counts the refusal rather than guessing a body's gravity.
#[must_use]
pub fn surface_look_bag(
    outline: &Boundary,
    luma: Option<(u8, f64)>,
    surface: &SurfaceStmt,
    charter: Option<&BodyCharter>,
) -> Vec<u8> {
    let writer = self_look_writer(outline, luma)
        .required(TAG_SURFACE, surface)
        .expect("distinct tag, a frame and one scalar");
    match charter {
        Some(charter) => writer
            .required(TAG_CHARTER, charter)
            .expect("distinct tag, twenty whole numbers"),
        None => writer,
    }
    .finish()
}

/// The surface a self-look bag states; `None` for a realm that states none (a hull, a station
/// without terrain), a typed refusal for a bag that is not a window body.
pub fn surface_of(bag: &[u8]) -> Result<Option<SurfaceStmt>, TlvError> {
    TlvReader::parse(WINDOW_BODY_SCHEMA, bag)?.optional(TAG_SURFACE)
}

/// The CHARTER a self-look bag states; `None` for a realm that states none, a typed refusal for a
/// bag that is not a window body or whose charter word is malformed.
///
/// A malformed charter is an ERROR, never a `Default`: a body whose gravity decoded to zero would
/// build a mountain of unbounded height, and the decode-to-Default ban exists for exactly that.
///
/// # Errors
/// [`TlvError`] when the blob is not a well-formed window-body bag, or when its `TAG_CHARTER`
/// payload does not decode as a [`BodyCharter`].
pub fn charter_of_bag(bag: &[u8]) -> Result<Option<BodyCharter>, TlvError> {
    TlvReader::parse(WINDOW_BODY_SCHEMA, bag)?.optional(TAG_CHARTER)
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

#[cfg(test)]
mod surface_tests {
    use super::*;
    use crate::glam::{DQuat, DVec3};
    use crate::pose::FrameRef;

    /// The five window-body tags are distinct and never move: a reorder or a reuse re-means every
    /// bag ever stated.
    #[test]
    fn the_window_body_tags_are_distinct_and_pinned() {
        let tags = [
            TAG_LOOK,
            TAG_LUMA,
            TAG_EXTENT,
            TAG_SURFACE,
            TAG_BODIES,
            TAG_CHARTER,
        ];
        let distinct: std::collections::BTreeSet<u16> = tags.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            tags.len(),
            "a window-body tag is numbered twice"
        );
        assert_eq!(tags, [1, 2, 3, 4, 5, 6], "the numbering moved");
    }

    fn moon() -> SurfaceStmt {
        SurfaceStmt {
            frame: FrameRef::PlanetCentered { planet_seed: 2298 },
            generator: 0xcbf2_9ce4_8422_2325,
        }
    }

    /// A charter shaped like the one a stage-2 author states: the words the census draws are
    /// present, the words a later stage draws are ABSENT.
    fn charter() -> BodyCharter {
        BodyCharter {
            gravity_mm_s2: 9_821,
            bulk_density_kgm3: 5_514,
            escape_velocity_mps: 11_190,
            insolation_q12: 3_065,
            t_eq_mk: 236_795,
            t_surface_mk: None,
            bond_albedo_q12: 1_228,
            mu_q8: Some(7_168),
            scale_height_m: Some(7_161),
            p_surf_pa: None,
            tau_vis_q12: None,
            tau_ir_q12: None,
            day_s: None,
            obliquity_cos_q1024: None,
            water_km3: None,
            sea_offset_mm: None,
            elastic_thickness_m: None,
            ecc_q16: 1_130,
            year_s: 34_766_100,
            flags: CHARTER_FLAG_HAS_AIR
                | CHARTER_FLAG_SOLID_SURFACE
                | (4 << CHARTER_STAR_CLASS_SHIFT),
        }
    }

    /// ★ THE CHARTER RIDES THE SAME BAG, AND ABSENCE IS ABSENCE (slice 8b stage 1). A charter-bearing
    /// bag still reads as an outline to a reader that knows only `TAG_LOOK`; a bag without a charter
    /// says so and is never read as a body of zero gravity.
    #[test]
    fn a_charter_rides_the_surface_bag_and_a_bag_without_one_states_none() {
        let outline = Boundary::Shell {
            r: 6_370_747.312_696_504,
        };
        let bag = surface_look_bag(&outline, Some((4, 0.823)), &moon(), Some(&charter()));
        assert_eq!(charter_of_bag(&bag), Ok(Some(charter())));
        assert_eq!(surface_of(&bag), Ok(Some(moon())));
        assert_eq!(look_of(&bag), Ok(outline), "the outline is untouched");
        assert_eq!(luma_of(&bag), Ok((4, 0.823)));
        // The words a later stage authors are ABSENT, not zero — a sea nobody solved is not a sea
        // at the ladder radius.
        assert_eq!(
            charter_of_bag(&bag).map(|c| c.and_then(|c| c.sea_offset_mm)),
            Ok(None)
        );
        assert_eq!(
            charter_of_bag(&bag).map(|c| c.and_then(|c| c.water_km3)),
            Ok(None)
        );
        // A surface bag that states no charter says so; a plain self-look states neither.
        let bare = surface_look_bag(&outline, None, &moon(), None);
        assert_eq!(charter_of_bag(&bare), Ok(None));
        assert_eq!(surface_of(&bare), Ok(Some(moon())));
        assert_eq!(charter_of_bag(&self_look_bag(&outline, None)), Ok(None));
        assert_ne!(bag, bare);
        // A blob that is not a window body is refused, never guessed at.
        assert_eq!(charter_of_bag(&[0, 0, 0]), Err(TlvError::Truncated(3)));
        assert_eq!(TAG_CHARTER, 6, "the tag number never moves");
    }

    /// A MALFORMED charter word is an error, never a `Default`: the decode-to-Default ban, measured.
    #[test]
    fn a_malformed_charter_word_is_refused_and_never_defaults() {
        let outline = Boundary::Shell { r: 1.0 };
        let broken = TlvWriter::new(WINDOW_BODY_SCHEMA)
            .required(TAG_LOOK, &outline)
            .expect("fresh writer")
            .required(TAG_CHARTER, &(1u8, 2u8))
            .expect("distinct tag")
            .finish();
        // The outline still reads — the bag is well formed; only the charter word is not a charter.
        assert_eq!(look_of(&broken), Ok(outline));
        let refused = charter_of_bag(&broken);
        assert!(
            refused.is_err(),
            "a short charter word decoded: {refused:?}"
        );
    }

    /// The flag word's three readers, each driven both ways.
    #[test]
    fn the_charter_flags_state_the_air_the_ground_and_the_star() {
        let c = charter();
        assert_eq!(c.flags & CHARTER_FLAG_HAS_AIR, CHARTER_FLAG_HAS_AIR);
        assert_eq!(
            c.flags & CHARTER_FLAG_SOLID_SURFACE,
            CHARTER_FLAG_SOLID_SURFACE
        );
        assert_eq!(charter_star_class_code(c.flags), 4, "a G-class sun");
        let airless_giant = 6 << CHARTER_STAR_CLASS_SHIFT;
        assert_eq!(airless_giant & CHARTER_FLAG_HAS_AIR, 0);
        assert_eq!(airless_giant & CHARTER_FLAG_SOLID_SURFACE, 0);
        assert_eq!(charter_star_class_code(airless_giant), 6);
        assert_eq!(charter_star_class_code(0), 0);
        assert_eq!(
            charter_star_class_code(u32::MAX),
            15,
            "the field is four bits"
        );
    }

    #[test]
    fn a_surface_bag_roundtrips_and_a_reader_that_knows_only_the_look_still_reads_it() {
        let outline = Boundary::Shell { r: 1_737_400.0 };
        let bag = surface_look_bag(&outline, Some((5, 0.0)), &moon(), None);
        assert_eq!(surface_of(&bag), Ok(Some(moon())));
        assert_eq!(
            look_of(&bag),
            Ok(outline),
            "the outline is untouched beside the surface"
        );
        assert_eq!(luma_of(&bag), Ok((5, 0.0)));
        // The skip-unknown rule: a bag with no surface says so, and a plain look bag is still a bag.
        assert_eq!(
            surface_of(&self_look_bag(&outline, None)),
            Ok(None),
            "a hull states no surface"
        );
        let no_luma = surface_look_bag(&outline, None, &moon(), None);
        assert_eq!(surface_of(&no_luma), Ok(Some(moon())));
        assert_eq!(
            luma_of(&no_luma),
            Err(TlvError::MissingRequiredTag(TAG_LUMA))
        );
        assert_eq!(
            surface_of(&[0, 0, 0]),
            Err(TlvError::Truncated(3)),
            "a bag that is not a window body is refused, never read as a look"
        );
        assert_eq!(
            (TAG_SURFACE, TAG_BODIES),
            (4, 5),
            "the tag numbers never move"
        );
    }

    #[test]
    fn the_largest_self_look_fits_one_conservative_datagram() {
        // MEASURED: the condition on R-8. The widest frame is the area's (two seeds); the outline is a
        // full box; the luma rides beside.
        let widest = SurfaceStmt {
            frame: FrameRef::AreaLocal {
                planet_seed: u64::MAX,
                area_seed: u64::MAX,
            },
            generator: u64::MAX,
        };
        let outline = Boundary::Obb {
            half: DVec3::new(f64::MAX, f64::MAX, f64::MAX),
            orient: DQuat::IDENTITY,
        };
        // ★ THE WIDEST CHARTER (slice 8b stage 1): every word present and every word at its
        // widest varint, so the pin below bounds every charter any body can ever state.
        let charter = BodyCharter {
            gravity_mm_s2: u32::MAX,
            bulk_density_kgm3: u32::MAX,
            escape_velocity_mps: u32::MAX,
            insolation_q12: u32::MAX,
            t_eq_mk: u32::MAX,
            t_surface_mk: Some(u32::MAX),
            bond_albedo_q12: u32::MAX,
            mu_q8: Some(u32::MAX),
            scale_height_m: Some(u32::MAX),
            p_surf_pa: Some(u32::MAX),
            tau_vis_q12: Some(u32::MAX),
            tau_ir_q12: Some(u32::MAX),
            day_s: Some(u32::MAX),
            obliquity_cos_q1024: Some(i32::MIN),
            water_km3: Some(u64::MAX),
            sea_offset_mm: Some(i32::MIN),
            elastic_thickness_m: Some(u32::MAX),
            ecc_q16: u32::MAX,
            year_s: u64::MAX,
            flags: u32::MAX,
        };
        let bag = surface_look_bag(&outline, Some((u8::MAX, f64::MAX)), &widest, Some(&charter));
        println!("[surface] the widest self-look bag is {} bytes", bag.len());
        println!(
            "[surface] the widest charter-less self-look bag is {} bytes",
            surface_look_bag(&outline, Some((u8::MAX, f64::MAX)), &widest, None).len()
        );
        assert_eq!(
            charter_of_bag(&bag),
            Ok(Some(charter)),
            "the widest charter roundtrips"
        );
        // MEASURED and pinned tight, so a growth of the bag is a red test and a recorded decision,
        // never a silent drift toward the budget (the refuter's finding 12).
        let len = bag.len();
        // ★ MEASURED 2026-09-18 (slice 8b stage 1): 124 → 251 bytes. The charter's whole field —
        // its TLV tag, its length and twenty widest words — costs 127 bytes, so 949 of the 1 200
        // stay free. The pin was red at 124 before the charter landed, which is what makes this a
        // measurement and not an estimate. A bag with NO charter is still 124 bytes, asserted
        // beside it: nothing that was on the lane yesterday moved a byte.
        assert_eq!(len, 251, "the widest self-look bag moved");
        assert_eq!(
            surface_look_bag(&outline, Some((u8::MAX, f64::MAX)), &widest, None).len(),
            124,
            "the widest CHARTER-LESS self-look bag moved"
        );
        assert!(
            len <= SELF_LOOK_BUDGET_BYTES,
            "the self-look bag exceeds the datagram budget"
        );
    }
}
