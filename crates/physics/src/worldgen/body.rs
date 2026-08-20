//! A GENERATED BODY: what the generator emits, and how it lowers to a containment region.
//!
//! Owns: the pre-lowering body record (realm, parent, shape, placement), a star's drawn photometric
//! identity and a reflector's derived one, the placement kinds, and the lowering that turns a forest
//! of bodies into the frozen region forest a shard boots with.
//!
//! Does NOT own: motion. A placement is either a stored offset or a set of orbital elements, and the
//! lowering treats both as the same kind of fact — where the body is. Nothing downstream may ask
//! which arm produced it (SL4).

use super::{UniverseConfig, interior_band, interior_reach_m};
use crate::celestial::{OrbitalElements, orbital_state};
use crate::motion::Motion;
use crate::taxonomy::SpectralClass;
use glam::DVec3;
use vd_core::frame::FramePlacement;
use vd_core::geometry::{Boundary, RealmRegion};
use vd_core::pose::{LatticePos, RealmId, Tier, frame_for_realm};

/// The inner (acquire) edge of the P3 static containment band, metres inside a surface.
pub(crate) const CONTAINMENT_INSET_M: f64 = 1.0;
/// The outer (release) edge of the P3 static containment band, metres outside a surface.
pub(crate) const CONTAINMENT_OUTSET_M: f64 = 2.0;

/// A body the generator emits before lowering — its realm, parent, shape, and placement. The
/// walk roster uses `StaticOffset` placements (the byte-identity source); THE world's planets use
/// `Orbital`, whose static tick-0 anchor is baked at boot. [`to_regions`]
/// lowers a slice of these to the frozen [`RealmRegion`] forest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GeneratedBody {
    pub(crate) realm: RealmId,
    pub(crate) parent: Option<RealmId>,
    pub(crate) shape: Boundary,
    pub(crate) placement: Placement,
    /// The body's photometric identity, drawn from the SAME per-realm seed stream that generated
    /// it (the window lane's marker datum — owner-approved 2026-08-15/16,
    /// `docs/design/window_lane.md` §2.2/§2.8: a sleeping child's point of light is authored by
    /// its parent from the child's own generation stream). `Some` on every SYSTEM the seed
    /// generator emits (the star's mass → class → luminosity); `None` on the ambient
    /// Universe/Galaxy shells, on planets (their photometric ladder is an owed later draw), and
    /// on every hand-placed walk body (a player-built station has no seed stream). Stored on the
    /// body row exactly as the sibling derived field (`placement`'s orbital elements) is —
    /// NEVER lowered onto `RealmRegion` and never on the wire: the Slice-A marker emit reads it
    /// off the booted forest and ships TLV scalars, not this struct.
    pub(crate) photometrics: Option<StarPhotometrics>,
    /// The taxonomy row (T1): `Some` on every generated planet — computed, never drawn from
    /// the wire; `None` on ambient/hand-placed/fixture bodies, exactly as `photometrics` is.
    /// Never lowered onto `RealmRegion`, never on the wire: every process derives the whole
    /// forest from the seed at boot (the complete SL6 answer — a planet shard computes its own
    /// mass, gravity, temperature and atmosphere locally, from the seed, with no message).
    pub(crate) taxon: Option<crate::taxonomy::BodyTaxon>,
    /// THE LOOK (real-scale design §3.0 — the BOUND/LOOK split): the outline this body DRAWS.
    /// `None` on the ambient Universe/Galaxy of THE world (never drawn, structurally); `Some`
    /// on every drawable body — a system's look is its STAR's photosphere (`star_radius_m` of
    /// the drawn mass), a planet's its Chen–Kipping radius at its drawn mass, a walk/plant
    /// body's its own bound (bound == look at human scale). Lowered onto `RealmRegion.look`.
    pub(crate) look: Option<Boundary>,
}

/// A star system's drawn photometric identity — pure `f(seed)` through the taxonomy layer
/// (`sample_imf_mass` → `classify_spectral` → `main_sequence_luminosity`), drawn ONCE per system
/// at generation from the system's own [`realm_stream`]. Off-wire DATA (HR1): the window lane's
/// marker (`BodyStmt::Marker`) will carry TLV-framed scalars derived from this at Slice A, and
/// the pinned-values test freezes each system's draw on THE world so any stream drift is loud.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarPhotometrics {
    /// The star's drawn mass (solar masses) — the ONE u01 draw everything below derives from.
    /// On a REFLECTOR's datum (a planet, Slice C1) this is the ILLUMINATING star's mass: a
    /// reflector shines by its star, so its color class and its mass provenance are the star's.
    pub mass_msun: f64,
    /// Morgan-Keenan class from the drawn mass (the marker's color class). A reflector carries
    /// its ILLUMINATOR's class — reflected light keeps the star's color.
    pub class: SpectralClass,
    /// Main-sequence luminosity `L/Lsun` from the drawn mass (the marker's luma scalar). A
    /// reflector carries the star's luminosity geometrically diluted at its own orbit and scaled
    /// by its cross-section × its seed-drawn albedo (see [`reflected_photometrics`]).
    pub luma_lsun: f64,
}

/// The canonical GEOMETRIC-ALBEDO table (lo, hi) a reflector's seed-drawn albedo spans — a
/// physical passable table like [`SpectralClass::MASS_BOUNDS`], not a tuning knob: solar-system
/// geometric albedos run from ~0.1 (dark rock — the Moon, Mercury) to ~0.7 (full cloud decks —
/// Venus). Scale-independent; at near-real scale the same bounds hold unchanged.
pub const GEOMETRIC_ALBEDO_BOUNDS: (f64, f64) = (0.1, 0.7);

/// A sleeping REFLECTOR's marker datum (Slice C1 — `docs/design/window_lane.md` §1.1 item 3b: "a
/// point-of-light datum per DIRECT child"; the planets' half of the per-system draw, owed since
/// Slice 0 and landed with the flag day that made it load-bearing): the star's light reflected.
/// Class = the illuminator's (reflected light keeps the star's color); luma = `L★ · albedo ·
/// r² / (4d²)` — the closed-form geometric dilution of starlight over the orbit radius `d`,
/// intercepted by the body's cross-section `r²`, scaled by ONE seed-drawn albedo over the
/// canonical table. Pure f(seed, config), scale-free, straight-line (HR5).
pub(crate) fn reflected_photometrics(
    star: &StarPhotometrics,
    albedo_u01: f64,
    radius_m: f64,
    orbit_m: f64,
) -> StarPhotometrics {
    let (lo, hi) = GEOMETRIC_ALBEDO_BOUNDS;
    let albedo = lo + albedo_u01 * (hi - lo);
    StarPhotometrics {
        mass_msun: star.mass_msun,
        class: star.class,
        luma_lsun: star.luma_lsun * albedo * (radius_m * radius_m) / (4.0 * orbit_m * orbit_m),
    }
}

/// Where a body sits in its parent inertial frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Placement {
    /// A fixed frame-local offset (the walk roster — the byte-identity source).
    StaticOffset(DVec3),
    /// A Keplerian orbit. PRODUCED IN PRODUCTION: [`generate_system_forest`] emits one per planet of
    /// THE world (`UniverseConfig::world` → `visual_demand` → 5 planets per system), and every shipped
    /// shard boots that forest. This doc used to claim "no production producer at P3" under an
    /// `#[allow(dead_code)]` — both false, and the false claim sat on the exact discriminator SL4
    /// governs (audit finding 28); the producer test
    /// `the_world_produces_an_orbital_planet_in_every_system` makes the claim a measurement now.
    Orbital(OrbitalElements),
}

/// The region CENTER a body's boundary sits at IN ITS OWN FRAME, as a `cell == ZERO` [`LatticePos`]:
/// - a **moving** (`Orbital`) body authors its position LIVE through the placement book its parent's
///   shard writes each tick (`author_placements` over the injected `MotionFn`s), so its boundary is
///   at the frame ORIGIN — center **ZERO**, NEVER the epoch. (This is the moving-realm crossing-flap fix: a
///   nonzero epoch center would be DOUBLE-COUNTED against the live frame placement in
///   [`region_signed_distance`](vd_core::geometry::region_signed_distance) — shifting the SOI ~one orbit off
///   the body, so the parent shard and the body's own shard disagree on containment and a crossing flaps.)
/// - a **`StaticOffset`** body's frame is the identity, so its fixed offset IS the boundary center.
///
/// Walk scale is ALL `StaticOffset` ⇒ unchanged ⇒ byte-identical; only the visual/canonical movers flip to
/// ZERO (whose live pose every other consumer — demand, AoI, feed, render — already reads from the frame).
///
/// ⚠ OWED (the placement arc, DEFERRED): this field's ZERO-for-a-mover is the residual two-meanings
/// leak (finding 27) — the boot fence no longer reads it (it takes the boot's worst-instant
/// [`ChildReach`](vd_core::geometry::ChildReach)), but the FIELD's presence still stores "where I sit"
/// on the region record; its deletion (positions living only in authored placement books) lands with
/// the motion-roster split.
fn region_center_of(placement: Placement) -> LatticePos {
    // THE PRODUCER SWITCH (real-scale addendum §A4.8 row 7): the generator emits NORMALIZED
    // centres — the integer half carries the whole-quantum part, so a static child's placement
    // enters the lattice at boot instead of riding a bare f64 offset. `from_metres` on BOTH arms
    // (the Orbital arm's ZERO normalizes to ZERO — value-identical, one rule).
    match placement {
        Placement::Orbital(_) => LatticePos::from_metres(DVec3::ZERO, Tier::Fine),
        Placement::StaticOffset(_) => {
            LatticePos::from_metres(placement_offset(placement), Tier::Fine)
        }
    }
}

/// The frame-local offset of a placement. `Orbital` evaluates [`orbital_state`] ONCE at tick 0
/// (Tier-2 libm is boot-time here, not a per-tick oracle — the cross-host gate is SPIKE-6a).
pub(crate) fn placement_offset(placement: Placement) -> DVec3 {
    match placement {
        Placement::StaticOffset(v) => v,
        Placement::Orbital(elements) => orbital_state(&elements, 0.0).position,
    }
}

/// Lower generated bodies to the frozen `RealmRegion` forest under `config`. Every region shares
/// the one static containment band; the frame is the realm's canonical authority frame
/// (`frame_for_realm`) with parent-provenance from the body (so the Area `.expect` cannot fire),
/// keeping the input-side containment seam and the output-side `transfer_frame` conversions in
/// agreement.
pub(crate) fn to_regions(bodies: &[GeneratedBody], config: &UniverseConfig) -> Vec<RealmRegion> {
    let band = config
        .band
        .build()
        .expect("containment band edges are valid by construction");
    bodies
        .iter()
        .map(|b| {
            // RLM Step 2: per-realm AoI = factor × the body's OWN finite extent, the dead-zone widened
            // by the occupant speed + THIS child's own closing speed. THE scalar comes from the ONE
            // motion discriminant ([`Motion::closing_speed_mps`]) — the fifth rival has-orbit test
            // (`orbital_of(..).map_or(0.0, v_peri)`) collapsed into that accessor (D-PLACE-1; the
            // batch review caught the accessor dead beside the still-live test).
            let v_child = motion_of(b.placement).closing_speed_mps();
            let aoi = config
                .interest
                .build(b.shape.finite_extent(), v_child)
                .expect("aoi band edges are valid by construction");
            RealmRegion {
                realm: b.realm,
                center: region_center_of(b.placement),
                frame: frame_for_realm(b.realm, b.parent)
                    .expect("roster realms have a canonical frame"),
                shape: b.shape,
                band,
                aoi,
                look: b.look,
                parent: b.parent,
                // Look horizon slice 4 (§3.4.4): the interior band is stamped HERE, from the
                // FULL forest this map runs over — before any scope filter drops the
                // grandchildren it is derived from. That is what settles the §3.4.4 CLAIM: a
                // galaxy shard's boot roster row for a star system carries the reach with no
                // message ever crossing a boundary (Ask D stays deferred).
                interior_band: interior_band(
                    interior_reach_m(bodies, b.realm, config.planet.ecc_cap),
                    &config.interest,
                    v_child,
                ),
            }
        })
        .collect()
}

/// The `OrbitalElements` of a placement, or `None` for a static one — the MONOMORPHIC discriminator that
/// keeps [`moving_children`]'s closure branchless (HR5: the `match` is covered once, here).
pub(crate) fn orbital_of(placement: Placement) -> Option<OrbitalElements> {
    match placement {
        Placement::Orbital(elements) => Some(elements),
        Placement::StaticOffset(_) => None,
    }
}

/// The [`Motion`] a generated placement lowers to — Kepler for an orbital body, [`Motion::Fixed`] at
/// the stored offset otherwise. THE one lowering from "what the generator drew" onto the motion
/// discriminant, so every scalar motion property the lowering needs (`closing_speed_mps` today) is
/// read off the accessor instead of re-running a has-orbit test beside it (D-PLACE-1's fifth rival).
/// Monomorphic; both arms covered by the `to_regions` AoI tests.
fn motion_of(placement: Placement) -> Motion {
    match placement {
        Placement::Orbital(elements) => Motion::Kepler(elements),
        Placement::StaticOffset(at) => Motion::Fixed(FramePlacement::moving(at, DVec3::ZERO)),
    }
}
