//! THE seed universe generator (D-45(a); the placement arc S5) — the single source of truth for
//! realm→region geometry, now living where SL4 puts it: in the motion crate, beside the closed-form
//! celestial math it draws on. Closed-form `f(seed)`: every shard computes the IDENTICAL containment
//! forest at boot from the shared universe seed, so the geometry is REPLICATED BY CONSTRUCTION — no
//! shared mutable state, no inter-shard bytes (HR1).
//!
//! The split rule (SL5's guard, stated once): anything that READS THE SEED STREAM or MINTS A BODY
//! lives here; anything that only reads an already-built `&[RealmRegion]` stays in
//! `vd_core::worldgen` (topology utilities over this generator's output — `level_of`,
//! `coord_of_realm`, `ancestor_realms`, `pin_realm_of`, the neighbourhood scope). ONE generator; the
//! crossing path cannot name it (the `crate_isolation` SL4 law).
//!
//! The P3 forest models the mandate hierarchy **Universe ⊃ Galaxy ⊃ StarSystem ⊃ Planet** (walk scale:
//! Station + Area as first-class hand-placed realms — task #133); the visual/demand presets generate
//! the compressed-real Kepler systems every shipped shard boots (SL5: THE world, one of it).

use glam::DVec3;
use serde::{Deserialize, Serialize};

use core::f64::consts::TAU;

use crate::celestial::{G, OrbitalElements, orbital_state};
use crate::motion::Motion;
use crate::taxonomy::{
    FrostThresholds, GalaxyType, SpectralClass, classify_spectral, main_sequence_luminosity,
    orbital_axis_au, sample_imf_mass, sample_rayleigh,
};
use vd_core::frame::FramePlacement;
use vd_core::geometry::{AoiConfig, BandError, Boundary, ContainmentBand, RealmRegion};
use vd_core::pose::{LatticePos, RealmId, frame_for_realm};
use vd_core::realm_path::RealmLevel;
use vd_core::rng::{SplitMix64, child_seed, realm_stream};
use vd_core::worldgen::{
    AREA_A, GALAXY, GALAXY_SEED, MAX_RENDERABLE_EXTENT_M, PLANET_A, STATION_A, SYSTEM_A,
    SYSTEM_A_SEED, SYSTEM_B, UNIVERSE, UNIVERSE_SEED, WALK_DEMAND_AOI_GRACE_S, ancestor_realms,
    grace_ticks_from_seconds, level_of, neighbourhood_scope,
};

/// The inner (acquire) edge of the P3 static containment band, metres inside a surface.
const CONTAINMENT_INSET_M: f64 = 1.0;
/// The outer (release) edge of the P3 static containment band, metres outside a surface.
const CONTAINMENT_OUTSET_M: f64 = 2.0;

// --- P3 WALK-SCALE geometry (placeholders; P4/P5 makes every radius/center `f(seed[, tick])`, D-44). ---
/// The ambient-root (Universe) radius — effectively unbounded; an entity beyond it still resolves to the
/// Universe by the container fold IDENTITY. Non-renderable (far above the render-extent threshold).
const UNIVERSE_R_M: f64 = 1.0e9;
/// The galaxy radius — FINITE (it encloses the star systems), and it is the between-systems space an
/// entity occupies after leaving one system SOI and before entering the next. RENDERABLE (below the extent
/// threshold) so the client draws it as the CONTAINING box around the two systems — an entity in the gap is
/// visibly still inside the Galaxy realm, never orphaned. Contains System B's far face (130 + 40 = 170).
const GALAXY_R_M: f64 = 180.0;
/// A star-system SOI radius (walk scale).
const SYSTEM_SOI_R_M: f64 = 40.0;
/// A planet SOI radius (walk scale), nested inside a system.
const PLANET_SOI_R_M: f64 = 10.0;
/// System B's center on +X — a disjoint sibling of System A with a WALKABLE gap of galaxy between them
/// (System A far-face 40, System B near-face 90 ⇒ a ~50 m pure-Galaxy gap: leaving A you are IN the Galaxy
/// realm until you enter B). The round-trip probe points 0/50/100 still resolve System A / Galaxy / System B.
const SYSTEM_B_OFFSET_M: f64 = 130.0;
/// Planet A's center inside System A (offset from the star at the origin).
const PLANET_A_OFFSET_M: f64 = 20.0;
/// Station A's center inside System A, on the -X side (opposite Planet A on +X), clear of the origin
/// crowd + the round-trip legs at x = 0/50/100. A Cartesian box, not an SOI shell.
const STATION_A_OFFSET_M: f64 = -25.0;
/// Station A's box half-extent (a small docked-station volume). `|-25| + 5 = 30 < 40` ⇒ fully inside
/// System A's r=40 SOI.
const STATION_HALF_M: f64 = 5.0;
/// Area A's center inside Planet A, MEASURED FROM PLANET A — every placement is an offset from its own
/// parent, and Area A's parent is the planet, not the system. +5 puts it at +25 in the system's frame
/// (Planet A is at +20, r=10), so its box still spans x∈[22,28] within the planet's sphere and is still
/// OFFSET from the (20,0,0) escape-SOI probe, which must resolve to Planet 7 and not the Area.
///
/// This was written as +25 — the absolute, in the SYSTEM's frame, on a field that means "offset from my
/// parent". Nothing caught it because the login descent carried one unconverted point all the way down
/// and so compared every level's boundary in the system's frame, where the two agree. Once the descent
/// converts as it steps — the parent doing the downward arithmetic, per the ground rule — an area sitting
/// +25 from a planet that is itself only 10 wide is nowhere near it, and the world stops being coherent.
const AREA_OFFSET_M: f64 = 5.0;
/// Area A's box half-extent (a small sub-planet district volume).
const AREA_HALF_M: f64 = 3.0;

// --- FA-5 (D-45(a)) VISUAL-scale single-system generator — the COMPRESSED-REAL game scale (the ONE
// geometry; `visual_scale` is its static-render expression, `visual_demand` its live demand-cluster
// expression). Every visual geometry number is DERIVED (parameterized helpers below), not a literal. ---
/// `child_seed` salt distinguishing PLANET-kind children under a system (a fixed kind discriminant;
/// `child_seed` avalanches `(parent, salt, index)`, so a distinct salt keeps planet ids off other kinds).
const PLANET_SALT: u64 = 0x504c_414e_4554; // "PLANET"
/// `child_seed` salt distinguishing SYSTEM-kind children under a galaxy — the sibling-kind discriminant
/// for stars, exactly as [`PLANET_SALT`] is for planets. A system's identity is `f(galaxy, index)`, so two
/// galaxies never mint the same system id and a system's planets never collide with another system's.
const SYSTEM_SALT: u64 = 0x5359_5354_454d; // "SYSTEM"
/// SYSTEM_A's RNG lineage root→leaf `[Universe, Galaxy, System]` — MUST equal
/// `realm_path::system_path(SYSTEM_A_SEED).lineage_seeds()` so every shard hosting System A draws the
/// IDENTICAL per-system stream by construction (HR1); consumed once by [`generate_system_forest`].
#[cfg(test)]
const SYSTEM_A_LINEAGE: [u64; 3] = [UNIVERSE_SEED, GALAXY_SEED, SYSTEM_A_SEED];
/// Compressed-real planet count — a 5-planet Kepler system: the inner 3 subtend ≥ the visibility angle
/// from the star (drawn) while the outer 2 fall below it (culled) until an occupant closes in, so the
/// one `cot(θ/2)` rule visibly culls by angular size. Exercises the geometric spacing for n=0..4.
const VISUAL_N_PLANETS: u32 = 5;
/// The compressed-real System SOI radius (render m): System 150 ⊂ Galaxy 180 < cull 200. FLAG: only
/// 20 m of headroom below the cull — no one raises this past ~180 without also moving
/// [`MAX_RENDERABLE_EXTENT_M`] in the same change.
const VISUAL_SYSTEM_SOI_R_M: f64 = 150.0;
/// Headroom (render m) between the OUTER planet's WORST-INSTANT face (apoapsis at the eccentricity
/// cap + its SOI) and the System SOI surface — [`visual_au_to_render_m`] solves the compression so the
/// outer body renders strictly inside its System box at EVERY point of its orbit, not just the epoch.
const VISUAL_SYSTEM_MARGIN_M: f64 = 4.0;
/// THROWAWAY (tiny world): how many stars the demo galaxy holds. Enough to fly between and to watch one
/// wake ahead while another sleeps behind — the system-level expression of the rule already watched
/// working on planets. A real galaxy draws this from its census range, not a constant.
const VISUAL_N_SYSTEMS: u32 = 3;
/// THROWAWAY (tiny world): how much wider than the wake radius the ring is. Above 1.0 a system is asleep
/// when you set off and wakes as you close — which is the whole point of flying there.
const VISUAL_RING_SLACK: f64 = 1.05;
/// A planet's SOI radius as a fraction of the SMALLEST inter-orbit gap. TWO constraints size it, and
/// both are pinned tests, never a hand-tuned coincidence:
/// - `< 0.5` guarantees adjacent SOIs never overlap;
/// - LARGE ENOUGH that a planet stays visible from anywhere inside its own system under the
///   apoapsis-solved compression (the placement arc S4): the visibility reach is `soi · cot(θ/2)` and
///   must cross `2 · system_soi = 300 m`, so `soi ≥ 300 / 76.390 = 3.927 m`; at this fraction the
///   solved SOI is 3.954 m (reach 302.1 m — pinned by
///   `a_planet_is_visible_from_anywhere_inside_its_own_system`). The apoapsis re-solve shrank the
///   compression by the `(1 + ecc_cap)` factor, and 0.35 left the SOI at 3.726 m — visible-from-
///   anywhere broken; this is the smallest two-decimal fraction that restores it with headroom.
const VISUAL_SOI_GAP_FRACTION: f64 = 0.372;
/// The OUTER (slowest) planet's orbital period in seconds — a MAJESTIC-but-visible pace for the human
/// window view (the inner planets are faster by Kepler-3: `T ∝ a^1.5`). Feeds the synthetic central
/// mass via the Kepler-3 inversion. NOT tuned to a frantic few-second orbit: the automated 2-capture
/// render gate samples universe ticks FAR ENOUGH apart to see the sweep, so the period is free to be
/// leisurely for a human watching.
const VISUAL_TARGET_OUTER_PERIOD_S: f64 = 300.0;
/// The minimum angular size (radians) a realm must subtend to enter Area of Interest — a realm is visible
/// (streams in) out to `extent · cot(θ/2) + velocity-lead`. ONE config constant, no kind-branch: this
/// single number sets how far away EVERY realm kind appears, from a planet to a galaxy.
///
/// 1.5° (was 8°). Two independent measurements forced it down, and both are about the same thing — at 8°
/// a body only exists once it is ~14 of its own radii away, which is close enough to be already large:
///
/// 1. A planet was invisible from inside its own star system. Its wake radius was 59.5 m against a system
///    150 m in radius, so you could cross a system's boundary and find it apparently empty — measured, and
///    now pinned by `a_planet_is_visible_from_anywhere_inside_its_own_system`.
/// 2. Bodies POPPED IN at full size instead of growing from a dot, which is the opposite of the owner's
///    stated arrival/departure behaviour (a system shrinks to a dot as you leave, grows from one as you
///    arrive). Angular size IS that behaviour: the smaller this angle, the smaller a body is when it first
///    appears. At 1.5° it enters the scene ~76 radii out — a few pixels — and grows all the way in.
///
/// COST, stated because it is real and pays at every scale: every realm's wake radius grows by the same
/// 5.3×, so more realms are awake at once and the demand loop carries more of them. That is the ONE knob's
/// nature — it cannot be widened for planets alone without branching on kind, which is forbidden. Note the
/// inter-system ring is DERIVED from this same factor ([`UniverseConfig::visual_geometry`]), so a system
/// still sleeps until you approach it no matter what this is set to — that behaviour is invariant here.
const VISIBILITY_THETA_MIN_RAD: f64 = 0.026_180;
/// One solar mass (kg) — the walk preset's INERT central mass (walk emits no `Orbital` body, so it
/// is never read there; [`UniverseConfig::visual_scale`] overrides it with a synthetic mass).
const CANONICAL_STAR_MASS_KG: f64 = 1.989e30;

/// A body the generator emits before lowering — its realm, parent, shape, and placement. The
/// walk roster uses `StaticOffset` placements (the byte-identity source); THE world's planets use
/// `Orbital`, whose static tick-0 anchor is baked at boot. [`to_regions`]
/// lowers a slice of these to the frozen [`RealmRegion`] forest.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GeneratedBody {
    realm: RealmId,
    parent: Option<RealmId>,
    shape: Boundary,
    placement: Placement,
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
    photometrics: Option<StarPhotometrics>,
}

/// A star system's drawn photometric identity — pure `f(seed)` through the taxonomy layer
/// (`sample_imf_mass` → `classify_spectral` → `main_sequence_luminosity`), drawn ONCE per system
/// at generation from the system's own [`realm_stream`]. Off-wire DATA (HR1): the window lane's
/// marker (`BodyStmt::Marker`) will carry TLV-framed scalars derived from this at Slice A, and
/// the pinned-values test freezes each system's draw on THE world so any stream drift is loud.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarPhotometrics {
    /// The star's drawn mass (solar masses) — the ONE u01 draw everything below derives from.
    pub mass_msun: f64,
    /// Morgan-Keenan class from the drawn mass (the marker's color class).
    pub class: SpectralClass,
    /// Main-sequence luminosity `L/Lsun` from the drawn mass (the marker's luma scalar).
    pub luma_lsun: f64,
}

/// Where a body sits in its parent inertial frame.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Placement {
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
    match placement {
        Placement::Orbital(_) => LatticePos::local(DVec3::ZERO),
        Placement::StaticOffset(_) => LatticePos::local(placement_offset(placement)),
    }
}

/// The frame-local offset of a placement. `Orbital` evaluates [`orbital_state`] ONCE at tick 0
/// (Tier-2 libm is boot-time here, not a per-tick oracle — the cross-host gate is SPIKE-6a).
fn placement_offset(placement: Placement) -> DVec3 {
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
fn to_regions(bodies: &[GeneratedBody], config: &UniverseConfig) -> Vec<RealmRegion> {
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
                parent: b.parent,
            }
        })
        .collect()
}

/// The DIRECT MOVING children a shard hosting `hosted_realm` AUTHORS (D-45(a) realm-unification FA-2b):
/// each direct child (`parent == hosted_realm`) whose placement is a live `Orbital`. Under LAW-1 a
/// passive orbiting body is the ZERO-SIGNAL case — the parent shard re-authors its live pose each tick
/// from these `OrbitalElements` (the boot wraps them as injected `MotionFn`s for the placement
/// writer), never a static region `center`.
/// Returned `(realm, elements)` so the sim keys it against each region by realm. A branchless-shim (HR5):
/// the `Orbital`/`StaticOffset` match lives in the monomorphic [`orbital_of`] helper, not this closure.
/// The walk roster is ALL `StaticOffset`, so this is EMPTY at walk scale (byte-identity); the canonical
/// seed generation (P4/FA-5) is what populates it.
fn moving_children(
    bodies: &[GeneratedBody],
    hosted_realm: RealmId,
) -> Vec<(RealmId, OrbitalElements)> {
    bodies
        .iter()
        .filter(|b| b.parent == Some(hosted_realm))
        .filter_map(|b| orbital_of(b.placement).map(|e| (b.realm, e)))
        .collect()
}

/// The `OrbitalElements` of a placement, or `None` for a static one — the MONOMORPHIC discriminator that
/// keeps [`moving_children`]'s closure branchless (HR5: the `match` is covered once, here).
fn orbital_of(placement: Placement) -> Option<OrbitalElements> {
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

/// The direct-child `RealmLevel` roster for `hosted` — from the SAME `(seed, config)` forest
/// [`to_regions`]/[`moving_children`] consume (`b.parent == Some(hosted)`), so the AoI child roster and
/// the containment region roster never diverge (L1). Closed-form `f(seed, config, hosted)`. `filter_map`
/// drops any non-seed-lineage child (a ship — never present in a seed forest). Each level builds the
/// child's `RealmCoord` as `own_coord.child(level)` (§3).
#[must_use]
pub fn direct_child_levels(
    seed_universe: u64,
    config: &UniverseConfig,
    hosted: RealmId,
) -> Vec<RealmLevel> {
    generate_system_forest(seed_universe, config)
        .iter()
        .filter(|b| b.parent == Some(hosted))
        .filter_map(|b| level_of(b.realm))
        .collect()
}

/// [`moving_children`] over the seed forest a shard boots — the AUTHORED moving-child roster for
/// `hosted_realm` (its direct `Orbital` children, each `(realm, elements)`). Closed-form
/// `f(seed, hosted_realm)`; EMPTY at walk scale (byte-identity), populated at canonical scale (P4/FA-5).
#[must_use]
pub fn moving_children_for(
    _seed_universe: u64,
    hosted_realm: RealmId,
) -> Vec<(RealmId, OrbitalElements)> {
    let config = UniverseConfig::walk_scale();
    moving_children(&generate_walk_forest(&config), hosted_realm)
}

// ===== FA-5 (D-45(a)) the config-driven VISUAL/canonical-scale single-system generator ==========
// The SAME generator serves the VISUAL synthetic-mass preset (window-friendly orbiting boxes) AND the
// canonical real-mass preset (P4 real proportions) with ZERO kind-match — only the config differs.

/// The Area-of-Interest visibility factor `cot(θ/2)` for a minimum angular size `theta_rad`: a realm of
/// finite extent `e` subtends `≥ theta_rad` (is visible) out to `e · cot(θ/2)`. Straight-line, branchless
/// (one region, HR5) — ALL AoI branching stays in [`AoiConfig::for_velocity_safe`]. At θ_min = 8° this is
/// ≈ 14.301.
fn visibility_factor(theta_rad: f64) -> f64 {
    1.0 / (theta_rad / 2.0).tan()
}

/// The TWO-LEVEL VISIBILITY CLEARANCE (owner ruling 2026-08-15, items 5/10 + the 2026-08-15
/// addendum) — how far an ancestor's shell must extend BEYOND a child's placement radius so that no
/// descendant two or more levels down subtends the visibility threshold from just outside the
/// ancestor. The GENERAL constraint, solved rather than tuned: for every ancestor `p` and every
/// descendant `g` two-or-more levels below,
///
/// > `R_p ≥ worst_instant_dist(g in p) + r_g + r_g · cot(θ_min/2)`
///
/// With the child placed at some radius and its worst descendant's centre reaching
/// `worst_descendant_reach_m` inside that child (its worst-instant excursion BOUND — a mover judged
/// at the apoapsis of the eccentricity CAP, never at a sampled epoch), the shell owes the child's
/// placement radius plus this clearance. `margin_m` keeps the bound STRICT: the Rayleigh
/// eccentricity draw is CLAMPED to the cap, so a seed can land a planet exactly ON the bound
/// (probability `e^{-ECC_CAP_SIGMAS²/2}` per planet), and the guard refuses `d_min ≤ required` —
/// equality included. Every term is an extent, a threshold, or the world's one containment-headroom
/// parameter — no literal enters here (owner addendum (B)). Straight-line f64 (HR5).
fn two_level_clearance_m(
    worst_descendant_reach_m: f64,
    descendant_extent_m: f64,
    theta_min_rad: f64,
    margin_m: f64,
) -> f64 {
    worst_descendant_reach_m
        + descendant_extent_m * (1.0 + visibility_factor(theta_min_rad))
        + margin_m
}

/// The GALAXY SHELL radius SOLVED from what it must hold (the placement derivation, owner ruling
/// 2026-08-15): the ring of star systems, plus the LARGER of
/// - the pre-existing CONTAINMENT headroom (two system SOIs — every system nests inside the shell
///   with a full system of clearance), and
/// - the TWO-LEVEL VISIBILITY clearance of the worst descendant ([`two_level_clearance_m`]) — so no
///   grandchild is ever visible from just outside the shell.
///
/// `.max` is a branchless clamp (HR5), and the algebra is SCALE-INDEPENDENT: at near-real scale a
/// system SOI dwarfs a planet's visibility reach (`r_g · cot(θ/2)` ≪ `system_soi`), so the
/// containment arm dominates and the visibility bound is trivially slack; at the interim tiny scale
/// the visibility arm binds (the 2026-08-15 measured failure). The solve GROWS THE SHELL rather
/// than pulling the ring inward, and that direction is forced, not chosen: the ring radius is
/// already the LOWER bound the wake law states (`system_soi · cot(θ/2) · slack` — a star must be
/// ASLEEP at departure and wake on approach), so an inward pull breaks the inter-system spacing law
/// and the shell is the only free direction — exactly the ruling's stated fallback.
fn galaxy_shell_r_m(
    ring_r_m: f64,
    system_soi_m: f64,
    worst_descendant_reach_m: f64,
    descendant_extent_m: f64,
    theta_min_rad: f64,
    margin_m: f64,
) -> f64 {
    ring_r_m
        + (2.0 * system_soi_m).max(two_level_clearance_m(
            worst_descendant_reach_m,
            descendant_extent_m,
            theta_min_rad,
            margin_m,
        ))
}

/// Closed-form inversion of Kepler's third law `T = 2π·√(a³/μ)`, `μ = G·M` → the central mass (kg)
/// that yields orbital period `target_period_s` at semi-major axis `sma_ref_m`. The SYNTHETIC-mass crux
/// for the visual scale: a real star mass at tens-of-metres `sma` gives a sub-µs (invisible) period, so
/// the visual system uses a synthetic mass tuned to a seconds-scale period instead. Straight-line f64.
fn synthetic_central_mass(sma_ref_m: f64, target_period_s: f64) -> f64 {
    TAU * TAU * sma_ref_m.powi(3) / (G * target_period_s * target_period_s)
}

/// The AU→render-metre compression solved so the OUTER planet's APOAPSIS at the eccentricity cap +
/// its SOI + `margin` sit EXACTLY at the System SOI surface (the placement arc S4). The denominator
/// used to carry the semi-major axis alone, which reserved margin against a circle the world never
/// promised: any eccentric outer planet crossed its own system's surface at apoapsis (MEASURED — the
/// S0 tripwire), and the boot fence, reading a mover's zeroed centre, could not see it. Solving
/// against `a·(1+ecc_cap)` makes the fence pass BY CONSTRUCTION, for every seed, exactly. Branchless.
/// Parameterized (RLM realistic-demo Slice 0) so `visual_scale` (static) and `visual_demand` (live)
/// derive the SAME geometry from the same compressed-real numbers.
fn visual_au_to_render_m(
    system_soi: f64,
    margin: f64,
    n: u32,
    gap_fraction: f64,
    a0: f64,
    ratio: f64,
    ecc_cap: f64,
) -> f64 {
    let outer_axis_au = orbital_axis_au(n - 1, a0, ratio);
    let soi_au = gap_fraction * a0 * (ratio - 1.0);
    (system_soi - margin) / (outer_axis_au * (1.0 + ecc_cap) + soi_au)
}

/// The planet SOI radius (render m) = the gap-fraction × the SMALLEST inter-orbit gap → adjacent SOIs
/// never overlap by construction. Straight-line f64.
fn visual_planet_soi_r_m(
    system_soi: f64,
    margin: f64,
    n: u32,
    gap_fraction: f64,
    a0: f64,
    ratio: f64,
    ecc_cap: f64,
) -> f64 {
    gap_fraction
        * a0
        * (ratio - 1.0)
        * visual_au_to_render_m(system_soi, margin, n, gap_fraction, a0, ratio, ecc_cap)
}

/// The OUTER (slowest) planet's semi-major axis in render metres — the period-tuning reference.
fn visual_outer_sma_render_m(
    system_soi: f64,
    margin: f64,
    n: u32,
    gap_fraction: f64,
    a0: f64,
    ratio: f64,
    ecc_cap: f64,
) -> f64 {
    orbital_axis_au(n - 1, a0, ratio)
        * visual_au_to_render_m(system_soi, margin, n, gap_fraction, a0, ratio, ecc_cap)
}

/// The synthetic central mass (kg) placing the OUTER planet's period at `outer_period`.
#[allow(clippy::too_many_arguments)] // the compressed-real derive threads one named knob per argument
fn visual_central_mass_kg(
    system_soi: f64,
    margin: f64,
    n: u32,
    gap_fraction: f64,
    a0: f64,
    ratio: f64,
    ecc_cap: f64,
    outer_period: f64,
) -> f64 {
    synthetic_central_mass(
        visual_outer_sma_render_m(system_soi, margin, n, gap_fraction, a0, ratio, ecc_cap),
        outer_period,
    )
}

/// Build one planet's [`OrbitalElements`] from the per-system `stream`, drawn in a FIXED order (ecc-u,
/// incl-u, Ω, ω, M₀) so the forest is pure `f(seed)`. ALL clamps are BRANCHLESS method calls (HR5): the
/// ecc cap is `.min(ecc_cap)` (an `if ecc > cap` would leave an UNREACHABLE true-arm at `ecc_sigma`≈0.03);
/// inclination is UN-clamped (no convergence domain, and `sample_rayleigh` is `≥ 0` — a `.max()`/`.abs()`
/// sign-normalize would inject an uncoverable branch). The star mass is DATA (`stellar.central_mass_kg`).
fn planet_elements(config: &UniverseConfig, stream: &mut SplitMix64, n: u32) -> OrbitalElements {
    let sma = orbital_axis_au(n, config.planet.orbital_a0_au, config.planet.orbital_ratio)
        * config.scale.au_to_render_m;
    let ecc =
        sample_rayleigh(stream.next_f64(), config.planet.ecc_sigma).min(config.planet.ecc_cap);
    let inclination = sample_rayleigh(stream.next_f64(), config.planet.incl_sigma);
    let raan = stream.next_f64() * TAU;
    let arg_periapsis = stream.next_f64() * TAU;
    let mean_anomaly_epoch = stream.next_f64() * TAU;
    OrbitalElements {
        sma,
        ecc,
        inclination,
        raan,
        arg_periapsis,
        mean_anomaly_epoch,
        central_mass: config.stellar.central_mass_kg,
    }
}

/// Draw one star system's photometric identity from ITS OWN per-system `stream` (the window
/// lane's per-system draw — owner-approved 2026-08-15/16, `docs/design/window_lane.md` §2.2:
/// "luma drawn from the same seed stream the parent generated the child from"). ONE u01 draw:
/// mass through the bounded-IMF inverse-CDF, then class + luminosity as closed-form derivations
/// of that mass — the taxonomy discipline (no rejection loop, bit-reproducible). Straight-line,
/// monomorphic (HR5). Parameters are DATA off [`StellarConfig`] (no magic numbers); the MK mass
/// bounds are the taxonomy's canonical passable table, the MLR segments the config's.
fn draw_star_photometrics(st: &StellarConfig, stream: &mut SplitMix64) -> StarPhotometrics {
    let mass_msun = sample_imf_mass(
        stream.next_f64(),
        st.imf_slope,
        st.mass_lo_msun,
        st.mass_hi_msun,
    );
    StarPhotometrics {
        mass_msun,
        class: classify_spectral(mass_msun, &SpectralClass::MASS_BOUNDS),
        luma_lsun: main_sequence_luminosity(mass_msun, &st.mlr_segments),
    }
}

/// The seed of the `n`-th star system in a galaxy. System 0 keeps [`SYSTEM_A_SEED`] — it is the identity
/// every existing fixture, label and gate already names — and the rest avalanche off the galaxy through
/// the same [`child_seed`] every other sibling set uses, so a system's identity is a pure function of
/// (galaxy, index) and no two galaxies ever collide.
#[must_use]
fn system_seed_at(n: u32) -> u64 {
    // Branchless in the HR5 sense: ONE covered comparison, no nested control flow.
    if n == 0 {
        SYSTEM_A_SEED
    } else {
        child_seed(GALAXY_SEED, SYSTEM_SALT, u64::from(n))
    }
}

/// How many star systems a galaxy holds — its CENSUS, drawn from the galaxy's own stream inside
/// `[system_count_lo, system_count_hi]`. Pure `f(universe_seed)` against the galaxy lineage, so every
/// shard agrees on the population without exchanging a byte, and two galaxies from one universe differ
/// without anyone authoring either. `lo == hi` pins the count exactly (the walk roster's shape).
///
/// One draw, no rejection loop — the taxonomy discipline: a sampler is a closed-form map from one uniform.
#[must_use]
fn galaxy_system_count(seed_universe: u64, config: &UniverseConfig) -> u32 {
    let lo = config.galaxy.system_count_lo;
    let hi = config.galaxy.system_count_hi.max(lo);
    let span = u64::from(hi - lo) + 1;
    let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED]);
    // `next_f64` is [0,1); scaling by the inclusive span and truncating lands in [lo, hi] with the last
    // bucket reachable only at exactly 1.0, which the generator never produces — hence the `min`.
    let draw = (stream.next_f64() * span as f64) as u64;
    lo + u32::try_from(draw.min(span - 1)).unwrap_or(0)
}

/// Where the `n`-th system sits in its galaxy. System 0 is the galactic origin; the rest are spaced
/// evenly around a ring of [`StellarConfig::system_ring_r_m`].
///
/// A RING, not random placement, and deliberately: two star systems whose boundaries overlap make
/// containment ambiguous — a position would be inside two authorities at once, and the deepest-containing
/// rule that decides which shard owns you would have no answer. A ring gives a closed-form minimum
/// separation (`2·r·sin(π/(n-1))` between neighbours, `r` from the origin system) that a boot fence can
/// check, where rejection-sampled positions would need a search. Seeded jitter belongs on top of this
/// later; it changes nothing about the separation guarantee.
#[must_use]
fn system_center_at(config: &UniverseConfig, n_systems: u32, n: u32) -> DVec3 {
    let ring_r = config.stellar.system_ring_r_m;
    let others = n_systems.saturating_sub(1).max(1);
    let theta = TAU * f64::from(n.saturating_sub(1)) / f64::from(others);
    // `n == 0` yields cos/sin of the SAME angle as `n == 1` but is multiplied by zero, so the origin
    // system needs no branch of its own — the multiplier is the whole decision.
    let on_ring = f64::from(u32::from(n != 0));
    DVec3::new(theta.cos() * ring_r, 0.0, theta.sin() * ring_r) * on_ring
}

/// Two sibling realms whose boundaries INTERSECT — the authoring mistake that makes "which realm contains
/// this position" have two answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiblingsOverlap {
    /// The two siblings, ordered as authored.
    pub a: RealmId,
    pub b: RealmId,
    /// Their shared parent.
    pub parent: RealmId,
}

/// Refuse a forest in which two STATICALLY-placed siblings intersect.
///
/// WHY THIS IS A CORRECTNESS FENCE, not tidiness. Authority is decided by descending into the deepest
/// child whose boundary contains you. If two siblings overlap, a position inside both has two equally
/// valid answers, and which shard owns you depends on iteration order — a coin flip that decides where
/// your input is applied and who simulates your collisions. It cannot be repaired downstream, because by
/// then the ambiguity is already a routing decision.
///
/// STATIC SIBLINGS ONLY, and that limit is real rather than convenient: an orbiting body's lowered region
/// sits at its frame ORIGIN (center zero — its position is authored live through its frame each tick), so
/// two planets are indistinguishable from co-located by any static comparison. Judging orbits needs their
/// SHELLS compared — two orbits are disjoint iff their radii differ by more than the sum of their
/// boundaries, at every eccentricity — which is a separate check over the moving roster. Recorded rather
/// than silently skipped: an unchecked orbital overlap is the same defect one level down.
///
/// Conservative on both sides: it compares FARTHEST-surface-point distances, so a doubtful placement is
/// refused rather than waved through. Cross-frame siblings are not judged (their numbers are not
/// comparable) — the same honest decline the parent-fit check makes.
/// RUN AT TEST TIME, not at boot, and that is a decision rather than an omission: the generator is
/// deterministic, so proving it over the shipped presets across a sweep of seeds proves every world that can
/// actually be booted, while a per-boot pass would be quadratic in a galaxy's population for an answer that
/// cannot change between runs. If worlds ever stop being purely seed-derived — the moment players place
/// structures the generator did not — this moves to the placement path, where the new body is the only thing
/// that needs judging.
#[cfg(test)]
fn siblings_disjoint(bodies: &[GeneratedBody]) -> Result<(), SiblingsOverlap> {
    for (i, a) in bodies.iter().enumerate() {
        let Placement::StaticOffset(a_at) = a.placement else {
            continue; // an orbit is judged on its shell, not its epoch — see above
        };
        let Some(parent) = a.parent else {
            continue; // the ambient root has no siblings
        };
        for b in bodies.iter().skip(i + 1).filter(|b| b.parent == a.parent) {
            let Placement::StaticOffset(b_at) = b.placement else {
                continue;
            };
            let reach = a.shape.circumscribed_extent() + b.shape.circumscribed_extent();
            if (b_at - a_at).length() < reach {
                return Err(SiblingsOverlap {
                    a: a.realm,
                    b: b.realm,
                    parent,
                });
            }
        }
    }
    Ok(())
}

/// A grandchild-or-deeper body that would be VISIBLE from just outside one of its ancestors — the
/// two-level bound broken by geometry. Carries every number of the verdict so the failure names
/// itself: the worst-instant distance of the body's centre from the ancestor's centre, the body's
/// extent, the resulting minimum viewer distance, and the distance the visibility rule requires.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error(
    "{body:?} subtends >= the visibility threshold from just outside its ancestor {ancestor:?}: \
     worst-instant centre distance {worst_dist_m} m, extent {extent_m} m, minimum viewer distance \
     {d_min_m} m, but the interest band keeps it visible out to {required_m} m — a not-running \
     realm two levels down would owe pixels its parent's placement marker cannot author"
)]
pub struct GrandchildVisibleOutside {
    pub body: RealmId,
    pub ancestor: RealmId,
    /// Worst-instant distance of the body's centre from the ancestor's centre (triangle bound —
    /// each hop contributes its own worst-instant offset magnitude).
    pub worst_dist_m: f64,
    /// The body's finite extent (the same extent the interest band judges visibility on).
    pub extent_m: f64,
    /// `R_ancestor − worst_dist − extent`: how close a viewer just outside the ancestor can get.
    pub d_min_m: f64,
    /// `extent · cot(θ_min/2)`: the distance out to which the interest band keeps the body visible.
    pub required_m: f64,
}

/// One hop's WORST-INSTANT offset magnitude — the same machinery the boot's `ChildReach` roster
/// states: a static child's authored offset, a mover's closed-form worst-instant excursion
/// ([`Motion::max_excursion_m`], the apoapsis — never a re-derived `a·(1+e)` beside it).
fn worst_hop_excursion_m(placement: &Placement) -> f64 {
    match placement {
        Placement::StaticOffset(at) => at.length(),
        Placement::Orbital(elements) => Motion::Kepler(*elements).max_excursion_m(),
    }
}

/// The two-level VERDICT NUMBERS for EVERY `(body, ancestor)` pair — ancestor two or more levels
/// up — with the body at its worst-instant position and the viewer just outside the ancestor's
/// boundary at closest approach. PURE GEOMETRY over the roster: no realm kinds, no motion kinds (a
/// hop's excursion is a magnitude whichever way it is produced). The threshold enters as the SAME
/// `cot(θ/2)` the interest band uses ([`visibility_factor`]) — the condition
/// `angular_size(extent, d_min) < θ_min` is exactly `d_min > extent · cot(θ_min/2)`. A pair is an
/// OFFENCE iff `d_min ≤ required` ([`grandchild_visibility_offences`] filters); a green pair's
/// margin `d_min − required` is the measured headroom the re-solve pins.
fn grandchild_visibility_pairs(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
) -> Vec<GrandchildVisibleOutside> {
    let by_id: std::collections::BTreeMap<RealmId, &GeneratedBody> =
        bodies.iter().map(|b| (b.realm, b)).collect();
    let factor = visibility_factor(theta_min_rad);
    let mut pairs = Vec::new();
    for body in bodies {
        let extent_m = body.shape.finite_extent();
        // Walk the ancestor chain, accumulating the worst-instant centre distance hop by hop.
        let mut worst_dist_m = worst_hop_excursion_m(&body.placement);
        let mut hops = 1_usize;
        let mut cursor = body.parent;
        while let Some(ancestor_id) = cursor {
            let ancestor = by_id
                .get(&ancestor_id)
                .expect("the generated forests resolve every parent (guarded at boot)");
            if hops >= 2 {
                let d_min_m = ancestor.shape.finite_extent() - worst_dist_m - extent_m;
                let required_m = extent_m * factor;
                pairs.push(GrandchildVisibleOutside {
                    body: body.realm,
                    ancestor: ancestor_id,
                    worst_dist_m,
                    extent_m,
                    d_min_m,
                    required_m,
                });
            }
            worst_dist_m += worst_hop_excursion_m(&ancestor.placement);
            hops += 1;
            cursor = ancestor.parent;
        }
    }
    pairs
}

/// Every pair of [`grandchild_visibility_pairs`] that IS an offence — the body would still be
/// VISIBLE (subtend ≥ `theta_min_rad`) from just outside its ancestor, equality included (the
/// margin the shell solve reserves is what keeps the worst lawful seed strictly clear).
fn grandchild_visibility_offences(
    bodies: &[GeneratedBody],
    theta_min_rad: f64,
) -> Vec<GrandchildVisibleOutside> {
    grandchild_visibility_pairs(bodies, theta_min_rad)
        .into_iter()
        .filter(|p| p.d_min_m <= p.required_m)
        .collect()
}

/// The FAIL-LOUD shape of the offence list — split out so both arms are driven by named examples
/// (a green world alone would leave the `Err` arm unrun, HR5).
fn first_offence(offences: Vec<GrandchildVisibleOutside>) -> Result<(), GrandchildVisibleOutside> {
    match offences.first() {
        Some(offence) => Err(*offence),
        None => Ok(()),
    }
}

/// THE GENERATOR VISIBILITY CHECK (owner ruling 2026-08-15, docs/design/owner_decisions_2026-08-15.md
/// item 5) — the two-level bound turned into a PROOF over the generated world: no body two or more
/// levels deep may ever be visible from outside its ancestor. SL7/SL3 make a not-running realm
/// drawable only as its parent's placement marker, one level down; a grandchild that subtends the
/// visibility threshold from outside its grandparent would owe pixels nothing is entitled to author.
/// Worst-instant positions (the same apoapsis machinery the `ChildReach` nest fence consumes), the
/// viewer at the ancestor's boundary at closest approach, and the ONE visibility threshold the
/// interest band uses — never a second literal.
///
/// RESOLVED (owner ruling 2026-08-15, the addendum to items 5/10): the 2026-08-15 measurement — every
/// ring-placed system's planets visible from just outside the galaxy shell — was reported, and the
/// owner ruled the generator SOLVES the margin as a general constraint ([`galaxy_shell_r_m`] /
/// [`two_level_clearance_m`]), never a hand-tuned number. THE world now passes (the green pin
/// `the_two_level_bound_re_solved_on_the_world_…` keeps the old failing numbers as history), and
/// this guard is WIRED into the shard boot fence beside `guard_regions_nest` (the same fail-loud
/// pattern): a world that violates it refuses to boot.
///
/// # Errors
/// [`GrandchildVisibleOutside`] naming the first offending `(body, ancestor)` pair with its numbers.
pub fn guard_grandchildren_invisible_outside(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Result<(), GrandchildVisibleOutside> {
    first_offence(grandchild_visibility_offences(
        &generate_system_forest(seed_universe, config),
        VISIBILITY_THETA_MIN_RAD,
    ))
}

/// The config-driven star-system forest: Universe → Galaxy → `stellar.n_systems` star systems, each with
/// `planet.n_planets` `Orbital` planets (D-45(a) FA-5). A System shell IS its star's frame — the planets
/// orbit its center and the star is DATA (`stellar.central_mass_kg`), never a `RealmId::Star` (HR3).
///
/// EVERY system is built by the SAME loop from its own seed — there is no "system A" special case beyond
/// which seed index 0 carries. That is what makes a galaxy expressible at all: the previous shape named
/// one system in code and hung the planets off a constant, so a second star could not exist at any scale,
/// which is why the login side and the shard side ended up with two different worlds.
///
/// Pure `f(seed_universe, config)`: each system draws its planets from its OWN [`realm_stream`], keyed on
/// its own lineage, in a fixed order — so a shard hosting system 4 generates byte-identical planets to
/// every other shard's view of system 4, without any shared state (HR1). `n_planets == 0` emits no
/// planet; `n_systems == 0` emits the ambient forest alone.
fn generate_system_forest(seed_universe: u64, config: &UniverseConfig) -> Vec<GeneratedBody> {
    let sc = &config.scale;
    let st = &config.stellar;
    let pl = &config.planet;
    let shell = |r: f64| Boundary::Shell { r };
    let origin = Placement::StaticOffset(DVec3::ZERO);
    let mut bodies = vec![
        // Universe: the ambient ROOT (parent None) — the container-fold identity.
        GeneratedBody {
            realm: UNIVERSE,
            parent: None,
            shape: shell(sc.universe_r_m),
            placement: origin,
            photometrics: None,
        },
        // Galaxy: the finite between-systems space, nested in the Universe.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
            photometrics: None,
        },
    ];
    // HOW MANY STARS THIS GALAXY HOLDS — drawn from the galaxy's OWN stream against its census, so two
    // galaxies in one universe differ without anyone choosing. `lo == hi` pins it exactly.
    let n_systems = galaxy_system_count(seed_universe, config);
    for s in 0..n_systems {
        let seed = system_seed_at(s);
        let system = RealmId::System(seed);
        let system_ix = bodies.len();
        bodies.push(GeneratedBody {
            realm: system,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: Placement::StaticOffset(system_center_at(config, n_systems, s)),
            // Filled below, AFTER the planet draws — see the stream-order note there.
            photometrics: None,
        });
        let mut stream = realm_stream(seed_universe, &[UNIVERSE_SEED, GALAXY_SEED, seed]);
        for n in 0..pl.n_planets {
            bodies.push(GeneratedBody {
                realm: RealmId::Planet(child_seed(seed, PLANET_SALT, u64::from(n))),
                parent: Some(system),
                shape: shell(pl.planet_soi_r_m),
                placement: Placement::Orbital(planet_elements(config, &mut stream, n)),
                photometrics: None,
            });
        }
        // The per-system photometric draw (the window lane's marker datum, Slice 0), from the
        // SAME per-system stream the planets drew from — APPENDED after the planet draws,
        // deliberately: a stream is positional exactly like the wire, so drawing the star FIRST
        // would shift every planet's (ecc, incl, Ω, ω, M₀) and re-roll every orbit of THE world.
        // Appending keeps every existing draw byte-identical (the additive discipline), and the
        // draw stays pure f(seed, config) like every sibling derived value.
        bodies[system_ix].photometrics = Some(draw_star_photometrics(st, &mut stream));
    }
    bodies
}

/// Every system's drawn [`StarPhotometrics`] over the config-driven forest — `(realm, draw)`
/// pairs, forest order; the config twin of [`moving_children_for_config`] (the SAME
/// `(seed, config)` builds the SAME forest, so the marker roster and the regions can never
/// disagree). The Slice-A marker emit reads THIS; Slice 0 lands it consumer-less (nothing moves),
/// pinned by the THE-world goldens below. The closure is a branchless shim (HR5): the
/// `Some`/`None` split lives in `Option::map`'s monomorphic body over `photometrics`.
#[must_use]
pub fn system_photometrics_for_config(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<(RealmId, StarPhotometrics)> {
    generate_system_forest(seed_universe, config)
        .iter()
        .filter_map(|b| b.photometrics.map(|p| (b.realm, p)))
        .collect()
}

/// One sleeping child's marker datum as the window lane's `TAG_LUMA` bag (Slice A,
/// `docs/design/window_lane.md` §2.2/§2.8): the spectral class as its stable code (color) plus the
/// main-sequence luminosity (brightness) — exactly the two scalars the owner-ruled R4 datum names,
/// framed by the ONE shared codec (`vd_core::look`) so the parent's emit and every later reader
/// can never frame the bag two ways. The boot plumbs these bags onto the shard's roster
/// (`vd-sim` `ChildLuma`) — boot/config path, never a wire-struct change.
#[must_use]
pub fn marker_luma_bag(p: &StarPhotometrics) -> Vec<u8> {
    vd_core::look::luma_bag(p.class as u8, p.luma_lsun)
}

/// [`to_regions`] over the config-driven system forest — the config-parameterised twin of
/// [`realm_regions_for`] (S2 wraps it with [`UniverseConfig::visual_scale`]). Reuses `to_regions` verbatim.
#[must_use]
pub fn realm_regions_for_config(seed_universe: u64, config: &UniverseConfig) -> Vec<RealmRegion> {
    to_regions(&generate_system_forest(seed_universe, config), config)
}

/// [`moving_children`] over the config-driven system forest — the config twin of [`moving_children_for`].
/// The SAME `(seed, config)` builds the SAME forest as [`realm_regions_for_config`], so the authored
/// moving roster and the regions can NEVER disagree (all-shard-seams-same-config). Reuses `moving_children`.
#[must_use]
pub fn moving_children_for_config(
    seed_universe: u64,
    config: &UniverseConfig,
    hosted: RealmId,
) -> Vec<(RealmId, OrbitalElements)> {
    moving_children(&generate_system_forest(seed_universe, config), hosted)
}

/// A WORLD, materialised once: the bodies that exist, and the containment regions they lower to.
///
/// WHY THIS TYPE EXISTS. The world used to be re-derived from `(seed, config)` at every question — where a
/// login lands, which regions a shard evaluates, how a realm's origin folds. Three problems followed from
/// that, and this type is the answer to all three at once:
///
/// 1. **Different questions could get different worlds.** The gateway resolved a player's home against the
///    GENERATED world and then validated that answer against the HAND-PLACED one. With a single star the two
///    agreed by accident; with several stars a login beside any other star resolves to a realm the check has
///    never heard of, and the gateway panics on its own defence. Holding ONE world makes that class of bug
///    unstateable rather than fixed.
/// 2. **The world could not contain anything a seed does not produce.** Stations are built by players and
///    areas mostly are; no seed ever emits one, so anything needing them had to reach for a second world
///    builder — which is precisely the seam that let (1) happen. A world is now something you HOLD, so it
///    can be generated content, generated content plus what players have built, or (in a test) content
///    placed by hand. Same code path, different contents.
/// 3. **It regenerated the entire forest per call.** A login rebuilt every star and planet to answer one
///    question about one position.
///
/// HONEST SCOPE: this materialises the whole forest once. A universe too large to hold in memory needs the
/// per-subtree lazy generator that is the P4 owe; this type is where that laziness will live, and moving it
/// here first means no caller has to change when it lands.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldView {
    bodies: Vec<GeneratedBody>,
    regions: Vec<RealmRegion>,
}

impl WorldView {
    /// The world a SEED produces: stars and their planets, and nothing else. This is the production world —
    /// what a player logs into today, before anyone has built anything.
    #[must_use]
    pub fn generated(seed_universe: u64, config: &UniverseConfig) -> WorldView {
        WorldView::of(generate_system_forest(seed_universe, config), config)
    }

    /// A world with structures PLACED BY HAND — a station, an area, a planet at a known spot.
    ///
    /// TEST WORLD. The generator emits none of these on purpose: stations are built by players, and so are
    /// areas in all but a few cases, so a seed-derived world contains no station to stand in and no area to
    /// spin up. A test that needs one places it. This is the same shape player-built content will take when
    /// it lands — a world is bodies, and where they came from is not something anything downstream asks.
    ///
    /// Everything below the placement is IDENTICAL to the generated path: the same lowering, the same bands,
    /// the same descend. Nothing here is a second implementation of anything.
    #[must_use]
    pub fn hand_placed(config: &UniverseConfig) -> WorldView {
        WorldView::of(generate_walk_forest(config), config)
    }

    /// Lower a body list to its regions once, and keep both — the regions answer containment questions, the
    /// bodies answer placement ones (an orbit's elements do not survive lowering).
    fn of(bodies: Vec<GeneratedBody>, config: &UniverseConfig) -> WorldView {
        let regions = to_regions(&bodies, config);
        WorldView { bodies, regions }
    }

    /// The containment forest — what a shard evaluates membership against.
    #[must_use]
    pub fn regions(&self) -> &[RealmRegion] {
        &self.regions
    }

    /// The regions a shard holding `held` evaluates: ancestors ∪ direct children, never siblings.
    #[must_use]
    pub fn neighbourhood(&self, held: &std::collections::BTreeSet<RealmId>) -> Vec<RealmRegion> {
        neighbourhood_scope(&self.regions, held)
    }

    /// Is this realm part of this world? The honest form of the check that used to consult a DIFFERENT world
    /// than the one that produced the answer being checked.
    #[must_use]
    pub fn contains_realm(&self, realm: RealmId) -> bool {
        self.regions.iter().any(|r| r.realm == realm)
    }

    /// This world LOWERED for the connection plane: the region forest alone, held as
    /// [`vd_core::worldgen::WorldRealms`]. The bodies — and every orbit element — deliberately do
    /// NOT survive the lowering, so the gateway can hold a world it queries without a dependency
    /// edge to this crate (SL4: the batch review found the crate fence carved open for
    /// vd-connection-plane precisely so it could hold a `WorldView`; it now receives this instead).
    #[must_use]
    pub fn lowered(&self) -> vd_core::worldgen::WorldRealms {
        vd_core::worldgen::WorldRealms::new(self.regions.clone())
    }
}

/// The walk-scale mandate forest as config-driven bodies, in forest order (Universe → Galaxy →
/// System A → Planet A → System B → Station A → Area A). All placements are `StaticOffset`, so the
/// lowering is byte-identical to the pre-generator forest. The GENERIC seed-driven child
/// enumeration (canonical scale) is deferred to P4 — its bodies are not live containment regions
/// until the D-41 non-zero-cell re-quantization.
fn generate_walk_forest(config: &UniverseConfig) -> Vec<GeneratedBody> {
    let sc = &config.scale;
    let st = &config.stellar;
    let pl = &config.planet;
    let sa = &config.satellite;
    let shell = |r: f64| Boundary::Shell { r };
    let boxed = |half: f64| Boundary::Aabb {
        half: DVec3::splat(half),
    };
    let at_x = |off: f64| Placement::StaticOffset(DVec3::new(off, 0.0, 0.0));
    let origin = Placement::StaticOffset(DVec3::ZERO);
    vec![
        // Universe: the ambient ROOT (parent None) — contains all reachable space (fold identity).
        GeneratedBody {
            realm: UNIVERSE,
            parent: None,
            shape: shell(sc.universe_r_m),
            placement: origin,
            photometrics: None,
        },
        // Galaxy: the finite between-systems space, nested in the Universe.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
            photometrics: None,
        },
        // Star system A: nested in the Galaxy at the origin.
        GeneratedBody {
            realm: SYSTEM_A,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: origin,
            photometrics: None,
        },
        // Planet A: nested in system A, offset from the star.
        GeneratedBody {
            realm: PLANET_A,
            parent: Some(SYSTEM_A),
            shape: shell(pl.planet_soi_r_m),
            placement: at_x(sa.planet_offset_m),
            photometrics: None,
        },
        // Star system B: a DISJOINT sibling of system A under the Galaxy (a walkable galaxy gap between).
        GeneratedBody {
            realm: SYSTEM_B,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: at_x(sa.system_b_offset_m),
            photometrics: None,
        },
        // Station A: a first-class Station BOX under System A (depth 3), on the -X side opposite Planet A.
        GeneratedBody {
            realm: STATION_A,
            parent: Some(SYSTEM_A),
            shape: boxed(sa.station_half_m),
            placement: at_x(sa.station_offset_m),
            photometrics: None,
        },
        // Area A: a first-class sub-planet Area BOX under Planet A (depth 4) — the DEEPEST region.
        GeneratedBody {
            realm: AREA_A,
            parent: Some(PLANET_A),
            shape: boxed(sa.area_half_m),
            placement: at_x(sa.area_offset_m),
            photometrics: None,
        },
    ]
}

/// A FIXTURE forest containing PLAYER-BUILT structures — a station and an area — hand-placed beside the
/// generated bodies.
///
/// NOT A SECOND WORLD, and the distinction is the whole point. The seed generates what NATURE puts
/// there: stars and their planets. Stations and areas are built by PLAYERS, so no seeded world contains
/// one, at any scale. Tests that exercise crossing into a station therefore have to place it themselves —
/// exactly as a player would — and that is what this is for. It is never called by a running game.
///
/// It previously masqueraded as world generation, which is how the login side ended up descending a
/// roster with hand-placed bodies while the shards built the real thing: the two disagreed about what
/// exists because one of them was a test fixture.
///
/// The single source of truth for realm→region geometry (see the module docs). At P3 returns the
/// static WALK-scale mandate forest, now GENERATED from [`UniverseConfig::walk_scale`] via
/// [`generate_walk_forest`] + [`to_regions`] (byte-identical to the pre-generator forest);
/// `_seed_universe` is threaded for the frozen P4/P5 `f(seed)` signature (unused while static —
/// the seed-driven canonical generation lands at P4).
#[must_use]
pub fn realm_regions_for(seed_universe: u64) -> Vec<RealmRegion> {
    realm_regions_for_walk_config(seed_universe, &UniverseConfig::walk_scale())
}

/// The walk mandate forest for an EXPLICIT `config` (RLM 5f-4) — the walk topology of
/// [`generate_walk_forest`] lowered by [`to_regions`], carrying `config.interest` into each region's AoI
/// band. `realm_regions_for` is this with [`walk_scale`](UniverseConfig::walk_scale) (AoI inert,
/// byte-identical); a demand cluster passes [`walk_demand`](UniverseConfig::walk_demand) for the LIVE band
/// over the SAME geometry. Uses `generate_walk_forest` (NOT `generate_system_forest`, whose `n_planets = 0`
/// at walk yields only Universe+Galaxy+System A) so the full mandate chain (Planet/Station/Area/System B) is
/// present — the same forest [`container_coord_at`] descends. `_seed_universe` threads the frozen P4/P5
/// `f(seed)` signature (unused while static).
#[must_use]
pub fn realm_regions_for_walk_config(
    _seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<RealmRegion> {
    to_regions(&generate_walk_forest(config), config)
}

/// The regions a shard hosting `hosted_realm` evaluates CONTAINMENT against: its own realm + its ancestor
/// chain to the ambient root + the children it hosts authority INTO — **never siblings**. A sibling
/// crossing routes through the shared PARENT (leaving a system lands you in the galaxy ANCESTOR, and the
/// GALAXY shard — which hosts the systems as its children — detects the entry into the next one). This
/// keeps the per-shard region set O(depth + owned-children), bounded by `MAX_REGIONS`; a galaxy of
/// THOUSANDS of sibling systems never loads them all (an ambient shard scanning many children is the P6
/// spatial index, DEFERRED D-45). Closed-form `f(seed, hosted_realm)`, replicated by construction (HR1).
#[must_use]
pub fn realm_neighbourhood_for(seed_universe: u64, hosted_realm: RealmId) -> Vec<RealmRegion> {
    let all = realm_regions_for(seed_universe);
    let ancestry = ancestor_realms(&all, hosted_realm);
    all.iter()
        .copied()
        .filter(|r| ancestry.contains(&r.realm) || r.parent == Some(hosted_realm))
        .collect()
}

/// The regions a CO-HOSTING shard evaluates containment against: the UNION of the per-realm
/// neighbourhoods over every realm the shard HOLDS (the un-hosted-child cure). A shard that hosts its
/// system AND that system's Planet/Station/Area children must evaluate the deeper regions (a Planet's
/// child Area is a GRANDCHILD of the system, absent from the system's own neighbourhood), so the shard
/// scans the union — deduped by realm, order-stable (region forest order), so the boot depth-key/guard
/// results are deterministic. For a SINGLE-realm shard (`held == {hosted}`) this equals
/// [`realm_neighbourhood_for`] exactly (byte-identical). Closed-form `f(seed, held)`, replicated by
/// construction (HR1) — the held-set is itself seed-derivable topology, not shared mutable state.
#[must_use]
// FIXTURE path — scopes the hand-placed roster (with its player-built station and area), NOT a generated
// world. The `_config` twin is the real one.
pub fn realm_neighbourhood_for_held(
    seed_universe: u64,
    held: &std::collections::BTreeSet<RealmId>,
) -> Vec<RealmRegion> {
    // Scoped over the FIXTURE roster, matching its single-realm twin `realm_neighbourhood_for`. Both
    // exist so a test can hold a station or an area — realms a player builds and no seed ever produces.
    // Routing this at the generated world instead would silently disagree with its own twin.
    neighbourhood_scope(&realm_regions_for(seed_universe), held)
}

/// The co-hosting neighbourhood for an EXPLICIT `config` (RLM 5f-4) — the config twin of
/// [`realm_neighbourhood_for_held`], preserving the ancestors-union-direct-children ("never siblings")
/// scope. A demand shard passes [`walk_demand`](UniverseConfig::walk_demand) so its evaluated regions carry
/// the LIVE AoI band; `realm_neighbourhood_for_held` is this with `walk_scale` (inert, byte-identical). One
/// implementation (HR3).
#[must_use]
pub fn realm_neighbourhood_for_held_config(
    seed_universe: u64,
    held: &std::collections::BTreeSet<RealmId>,
    config: &UniverseConfig,
) -> Vec<RealmRegion> {
    // THE SEAM, CLOSED. This read the WALK roster while the shards built the SYSTEM forest, so the login
    // side and the simulating side described different worlds from the same seed — the client was told a
    // star system was 40 m across while the shards flew planets to 152. Two identical functions differing
    // only in which world they consult is the same defect as branching on shard kind (HR3), wearing
    // different clothes. There is now ONE forest, and this is a thin alias kept only so the call sites
    // read naturally; it will collapse into its twin when the walk world goes.
    realm_neighbourhood_for_config(seed_universe, held, config)
}

/// The config twin of [`realm_neighbourhood_for_held_config`] over the SYSTEM forest (orbiting planets) — the
/// scope a `Visual`/`VisualDemand` shard boots. A planet shard thus evaluates containment against its OWN
/// realm + its ancestor chain + the children IT authors, and NEVER its sibling planets — the direct cure for
/// the origin-stacking re-home flap (a shard cannot place a realm it does not author, so it must not fold it).
/// Same ancestors-union-direct-children scope, over [`realm_regions_for_config`] instead of the walk forest.
/// One filter (HR3). Closed-form `f(seed, held, config)`, replicated by construction (HR1).
#[must_use]
pub fn realm_neighbourhood_for_config(
    seed_universe: u64,
    held: &std::collections::BTreeSet<RealmId>,
    config: &UniverseConfig,
) -> Vec<RealmRegion> {
    neighbourhood_scope(&realm_regions_for_config(seed_universe, config), held)
}

// --- Stellar/orbital PHYSICS (scale-independent; walk + canonical share these) ---
/// Salpeter IMF slope α (Salpeter 1955): `dN/dM ∝ M^-2.35`.
const IMF_SLOPE: f64 = 2.35;
/// Stellar mass sampling bounds (solar masses): the hydrogen-burning limit to a massive-O cap.
const IMF_MASS_LO_MSUN: f64 = 0.08;
const IMF_MASS_HI_MSUN: f64 = 120.0;
/// Titius-Bode orbital spacing seed (AU) + geometric ratio (Chambers 1996).
const ORBITAL_A0_AU: f64 = 0.4;
const ORBITAL_RATIO: f64 = 1.7;
/// Rayleigh scale for orbital eccentricity / inclination (Fabrycky 2014) — small so sampled
/// values stay well inside `KEPLER_ECC_MAX` (the generator also hard-caps at `ecc_cap`).
const ECC_SIGMA: f64 = 0.03;
/// The eccentricity cap in RAYLEIGH SIGMAS (the placement arc S4, owner-gated lever 1): `ecc_cap =
/// ECC_SIGMA · ECC_CAP_SIGMAS = 0.12`, and the AU compression is solved against the APOAPSIS at that
/// cap — so every planet's worst instant lands inside its system shell BY CONSTRUCTION, for every
/// seed, exactly. The cap replaces `KEPLER_ECC_MAX` doing geometry duty (a solver-convergence bound
/// has no business sizing a world); the clamp truncates the physical Rayleigh distribution with
/// probability `e^-(4²/2) = 3.35e-4` per planet — named, derived, never a magic number. MEASURED
/// before this lever: 2 of 3 systems' outer planets crossed their own shell at apoapsis (152.57 m and
/// 155.05 m against 150 m) — the S0 tripwire this lever turns green.
const ECC_CAP_SIGMAS: f64 = 4.0;
const INCL_SIGMA: f64 = 0.02;
/// Per-system occurrence probability of a station / a sub-planet area district.
const STATION_OCCURRENCE_PROB: f64 = 0.3;
const AREA_OCCURRENCE_PROB: f64 = 0.3;

// (The `canonical()`/`seed_derived()` real-scale presets and their CANONICAL_* geometry constants
// are DELETED — SL5, Stage-C audit :866: two complete world variants with zero production callers,
// whose own doc conceded they were not even the starting point for the true-scale work ("its
// astronomical-unit conversion is 2.5x the real value and its bodies do not nest correctly"). The
// true-astronomical-scale numbers are an owed change to THE ONE world, never a parallel preset.
// `CANONICAL_STAR_MASS_KG` and the taxonomy's `CANONICAL_*` census/threshold data survive — they are
// physical constants THE world reads, not preset geometry.)

/// The walk forest is exactly two systems (A + its disjoint sibling B).
const WALK_SYSTEM_COUNT: u32 = 2;

/// Ambient-root + galaxy + client-render scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleConfig {
    pub universe_r_m: f64,
    pub galaxy_r_m: f64,
    pub render_extent_m: f64,
    /// AU→render-metre compression (FA-5). VISUAL scale shrinks AU orbits into the render window
    /// (true scale would carry the real 1-AU metre factor — an owed change to THE world, SL5). Read
    /// ONLY on the `Orbital` generator path (inert on the walk/StaticOffset path), so its value is
    /// self-consistent-but-unused on walk_scale().
    pub au_to_render_m: f64,
}

/// Galaxy population + morphology census.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GalaxyConfig {
    /// Cumulative galaxy-type thresholds for `taxonomy::sample_galaxy_type`.
    pub type_cumulative: [f64; 2],
    pub system_count_lo: u32,
    pub system_count_hi: u32,
}

/// Star physics: SOI scale + the IMF + the mass-luminosity fit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StellarConfig {
    pub system_soi_r_m: f64,
    pub imf_slope: f64,
    pub mass_lo_msun: f64,
    pub mass_hi_msun: f64,
    pub mlr_segments: [(f64, f64, f64); 3],
    /// The parent star mass (kg) fed into every planet's [`OrbitalElements::central_mass`] — DATA, never
    /// a kind (HR3). SYNTHETIC on the visual preset (Kepler-3-tuned to a seconds-scale period so orbits
    /// are visible). Read ONLY on the `Orbital` path (inert on walk).
    pub central_mass_kg: f64,
    /// The radius of the ring the non-origin systems are spaced around (system 0 sits at the galactic
    /// origin). Must leave every system's boundary disjoint from every other's AND inside the galaxy —
    /// overlapping systems would make "which realm contains this position" ambiguous, which is the one
    /// question the whole authority model rests on.
    ///
    /// HOW MANY systems is NOT here: it is drawn from the galaxy's own census
    /// ([`GalaxyConfig::system_count_lo`]..=[`GalaxyConfig::system_count_hi`]) against the galaxy's seed,
    /// so two galaxies from one universe differ. A second count field here would be a second source of
    /// truth for the same fact.
    pub system_ring_r_m: f64,
}

/// Planet physics: SOI scale, orbital spacing, eccentricity/inclination, and the frost/mass
/// thresholds (kept as raw f64 so the config stays serde-clean; `frost_thresholds()` builds the
/// `taxonomy::FrostThresholds` view).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanetConfig {
    pub planet_soi_r_m: f64,
    pub orbital_a0_au: f64,
    pub orbital_ratio: f64,
    pub ecc_sigma: f64,
    pub incl_sigma: f64,
    /// Hard eccentricity cap the generator clamps to — a fail-loud cross-slice invariant: it
    /// MUST stay `<= KEPLER_ECC_MAX` (the fixed Kepler solver's convergence domain).
    pub ecc_cap: f64,
    pub frost_coeff_au: f64,
    pub m_ocean_lo_mearth: f64,
    pub m_gas_mearth: f64,
    pub m_core_crit_mearth: f64,
    /// The number of `Orbital` planets [`generate_system_forest`] emits for THIS system. `0` on
    /// walk_scale() (no planet body ⇒ ambient-only forest, byte-identity); the visual
    /// preset sets [`VISUAL_N_PLANETS`]. The `0..n_planets` range is the generator's only control flow.
    pub n_planets: u32,
}

impl PlanetConfig {
    /// The `taxonomy::FrostThresholds` view over the raw config fields (fed to `classify_planet`).
    #[must_use]
    pub fn frost_thresholds(&self) -> FrostThresholds {
        FrostThresholds {
            m_ocean_lo_mearth: self.m_ocean_lo_mearth,
            m_gas_mearth: self.m_gas_mearth,
            m_core_crit_mearth: self.m_core_crit_mearth,
        }
    }
}

/// Satellite bodies: station/area occurrence + the walk-scale placement geometry.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SatelliteConfig {
    pub station_prob: f64,
    pub area_prob: f64,
    pub station_half_m: f64,
    pub area_half_m: f64,
    pub station_offset_m: f64,
    pub area_offset_m: f64,
    pub planet_offset_m: f64,
    pub system_b_offset_m: f64,
}

/// Containment-band edges (the acquire/release hysteresis).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandConfig {
    pub inset_m: f64,
    pub outset_m: f64,
    pub k_safety_extra: f64,
}

impl BandConfig {
    /// Build the P3 static containment band (v_rel = 0, dt = 1 — the widening is inert). Fallible
    /// (the ctor validates edges); walk-scale edges are valid by construction.
    pub fn build(&self) -> Result<ContainmentBand, BandError> {
        ContainmentBand::for_containment_velocity_safe(
            self.inset_m,
            self.outset_m,
            0.0,
            1.0,
            self.k_safety_extra,
        )
    }
}

// ---- RLM Step 2: per-realm AoI radii (all seed-relative factors, no magic numbers) ----------------
/// The universe seconds-per-tick the AoI widening is measured against — MUST equal `StubConfig.tick_dt_s`
/// (a boot `debug_assert!` cross-checks it, M-2). Named, not inline. `pub` so THE-world pins in
/// sibling crates state the shipped posture from THIS one place instead of copying the literal.
pub const AOI_TICK_DT_S: f64 = 0.05;
/// Grace ticks a would-be release is held (1 s at the visual 20 Hz).
const VISUAL_AOI_GRACE_TICKS: u32 = 20;
/// Extra velocity-safety margin folded into the dead-zone widening (beyond `K_SAFETY`).
const VISUAL_AOI_K_SAFETY_EXTRA: f64 = 0.5;
/// The visual occupant's max speed (m/s) — MUST equal `StubConfig.move_speed_mps · time_multiplier`
/// (boot `debug_assert!`, M-2), so the anti-thrash pad is measured against the speed the sim integrates.
/// Under the single visibility factor (`spin_up_factor == tear_down_factor`) the geometric dead-zone
/// collapses, so THIS non-zero occupant speed is what keeps `tear_down > spin_up` (band validity requires
/// `occupant_v_max + v_child > 0`; see [`UniverseConfig::visual_demand`]). `pub` for the same
/// one-place reason as [`AOI_TICK_DT_S`].
pub const VISUAL_OCCUPANT_V_MAX_MPS: f64 = 2.0;

/// Walk-demand-scale AoI (RLM 5f-4): the LIVE band for the WALK forest, so a WALKING occupant's AoI
/// crosses each separated child's band. Tighter than visual (a walking player over metres, not AU): spin a
/// child up within this multiple of its extent, release past the larger tear-down multiple — HR3
/// proportional, no kind-match. The two DYNAMICS inputs (occupant speed, tick dt) are NOT consts — they are
/// supplied at the composer boot from the LIVE cluster values, which is what closes the M-2 two-home owe (a
/// hardcoded `AOI_TICK_DT_S = 0.05` is wrong at the dev cluster's 50 Hz = 0.02).
const WALK_DEMAND_AOI_SPIN_UP_FACTOR: f64 = 1.2;
const WALK_DEMAND_AOI_TEAR_DOWN_FACTOR: f64 = 1.8;
const WALK_DEMAND_AOI_K_SAFETY_EXTRA: f64 = 0.5;

/// Per-realm AoI radii as UNIFORM FACTORS of the realm's own finite extent (HR3: no match-on-kind — a
/// bigger realm reaches proportionally farther), plus the widening inputs (`occupant_v_max_mps`,
/// `tick_dt_s`) so [`to_regions`] stays a pure `f(UniverseConfig)` (a boot `debug_assert!` cross-checks
/// them against the live `StubConfig`, M-2 — no two-home drift). `walk_scale`/`canonical` set
/// `spin_up_factor = 0` ⇒ AoI inert ⇒ behaviour byte-identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterestConfig {
    pub spin_up_factor: f64,
    pub tear_down_factor: f64,
    pub grace_ticks: u32,
    pub k_safety_extra: f64,
    pub occupant_v_max_mps: f64,
    pub tick_dt_s: f64,
}

impl InterestConfig {
    /// Build the per-realm AoI band from the realm's own `finite_extent` + the child's own orbital
    /// closing speed `v_child` (its `v_peri`; 0 for a static child). At walk-scale (`spin_up_factor <=
    /// 0`) returns the inert band BRANCHLESSLY — NEVER through the fallible ctor (whose reject arm the
    /// byte-identity path must not touch). Fallible only for the LIVE case.
    ///
    /// # Errors
    /// [`BandError::InvalidEdges`] from [`AoiConfig::for_velocity_safe`] if the live factors are degenerate.
    pub fn build(&self, finite_extent: f64, v_child: f64) -> Result<AoiConfig, BandError> {
        if self.spin_up_factor <= 0.0 {
            Ok(AoiConfig::inert())
        } else {
            AoiConfig::for_velocity_safe(
                finite_extent,
                self.spin_up_factor,
                self.tear_down_factor,
                self.occupant_v_max_mps + v_child,
                self.tick_dt_s,
                self.grace_ticks,
                self.k_safety_extra,
            )
        }
    }

    /// The inert preset (walk/canonical): zero factor ⇒ AoI never fires ⇒ byte-identity.
    #[must_use]
    pub fn inert() -> InterestConfig {
        InterestConfig {
            spin_up_factor: 0.0,
            tear_down_factor: 0.0,
            grace_ticks: 0,
            k_safety_extra: 0.0,
            occupant_v_max_mps: 0.0,
            tick_dt_s: AOI_TICK_DT_S,
        }
    }

    /// Whether AoI is LIVE (a positive spin-up factor) — the M-2 boot cross-check applies only here.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.spin_up_factor > 0.0
    }
}

/// THE one config home for the seed universe generator — named sub-structs (no god-struct),
/// every field seed-derivable and doc-cited (no magic numbers).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UniverseConfig {
    pub scale: ScaleConfig,
    pub galaxy: GalaxyConfig,
    pub stellar: StellarConfig,
    pub planet: PlanetConfig,
    pub satellite: SatelliteConfig,
    pub band: BandConfig,
    /// Per-realm AoI radii (RLM Step 2). Inert at walk/canonical (byte-identity); live at visual.
    pub interest: InterestConfig,
}

impl UniverseConfig {
    /// The walk-scale preset — its fields ARE today's named geometry consts, so the generator
    /// reproduces the EXACT current forest (the byte-identity source). The galaxy is renderable
    /// (`galaxy_r_m < render_extent_m`), so the client draws it as the containing box.
    #[must_use]
    pub fn walk_scale() -> UniverseConfig {
        UniverseConfig {
            scale: ScaleConfig {
                universe_r_m: UNIVERSE_R_M,
                galaxy_r_m: GALAXY_R_M,
                render_extent_m: MAX_RENDERABLE_EXTENT_M,
                // Self-consistent with the walk Planet-A StaticOffset (20 m = 0.4 AU * 50), but INERT:
                // walk emits StaticOffset, never Orbital, so this factor is never read.
                au_to_render_m: PLANET_A_OFFSET_M / ORBITAL_A0_AU,
            },
            galaxy: GalaxyConfig {
                type_cumulative: GalaxyType::CANONICAL_CUMULATIVE,
                system_count_lo: WALK_SYSTEM_COUNT,
                system_count_hi: WALK_SYSTEM_COUNT,
            },
            stellar: StellarConfig {
                system_soi_r_m: SYSTEM_SOI_R_M,
                imf_slope: IMF_SLOPE,
                mass_lo_msun: IMF_MASS_LO_MSUN,
                mass_hi_msun: IMF_MASS_HI_MSUN,
                mlr_segments: SpectralClass::MLR_SEGMENTS,
                central_mass_kg: CANONICAL_STAR_MASS_KG, // INERT (walk emits no Orbital body).
                // Where the walk roster's second star already sat. It is no longer inert: with ONE
                // world, this preset drives the same generator as everything else, and a zero ring
                // would stack both stars on the origin — two authorities over one point.
                system_ring_r_m: SYSTEM_B_OFFSET_M,
            },
            planet: PlanetConfig {
                planet_soi_r_m: PLANET_SOI_R_M,
                orbital_a0_au: ORBITAL_A0_AU,
                orbital_ratio: ORBITAL_RATIO,
                ecc_sigma: ECC_SIGMA,
                incl_sigma: INCL_SIGMA,
                // The GEOMETRY cap (4σ of the Rayleigh draw), NOT the solver bound: the compression
                // below solves apoapsis-at-this-cap exactly onto the system shell (see ECC_CAP_SIGMAS).
                ecc_cap: ECC_SIGMA * ECC_CAP_SIGMAS,
                frost_coeff_au: crate::taxonomy::FROST_COEFF_AU,
                m_ocean_lo_mearth: FrostThresholds::CANONICAL.m_ocean_lo_mearth,
                m_gas_mearth: FrostThresholds::CANONICAL.m_gas_mearth,
                m_core_crit_mearth: FrostThresholds::CANONICAL.m_core_crit_mearth,
                n_planets: 0, // ambient-only forest (no Orbital body) — visual_scale() sets N.
            },
            satellite: SatelliteConfig {
                station_prob: STATION_OCCURRENCE_PROB,
                area_prob: AREA_OCCURRENCE_PROB,
                station_half_m: STATION_HALF_M,
                area_half_m: AREA_HALF_M,
                station_offset_m: STATION_A_OFFSET_M,
                area_offset_m: AREA_OFFSET_M,
                planet_offset_m: PLANET_A_OFFSET_M,
                system_b_offset_m: SYSTEM_B_OFFSET_M,
            },
            band: BandConfig {
                inset_m: CONTAINMENT_INSET_M,
                outset_m: CONTAINMENT_OUTSET_M,
                k_safety_extra: 0.0,
            },
            interest: InterestConfig::inert(), // walk: AoI OFF ⇒ byte-identity.
        }
    }

    /// The COMPRESSED-REAL visual geometry on a `walk_scale()` clone (system SOI 150, 5 Kepler planets,
    /// outer period 300 s) WITHOUT an interest band — the ONE game geometry that `visual_scale` (static
    /// render) and `visual_demand` (live demand cluster) both drive, so the two can NEVER disagree on
    /// geometry (they call the SAME parameterized derive helpers with the SAME compressed-real numbers).
    /// Mutates a `walk_scale()` CLONE (byte-identity: walk never calls these helpers), overriding
    /// only the System SOI + the four moving-planet fields (AU→render + planet SOI both DERIVED so the outer
    /// orbit + SOI + margin land exactly at the System surface — provable non-overlap + strict containment,
    /// the SYNTHETIC central mass Kepler-3-tuned to a seconds-scale period, and `n_planets`).
    fn visual_geometry() -> UniverseConfig {
        let mut cfg = UniverseConfig::walk_scale();
        cfg.stellar.system_soi_r_m = VISUAL_SYSTEM_SOI_R_M;
        cfg.scale.au_to_render_m = visual_au_to_render_m(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            cfg.planet.ecc_cap,
        );
        cfg.stellar.central_mass_kg = visual_central_mass_kg(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            cfg.planet.ecc_cap,
            VISUAL_TARGET_OUTER_PERIOD_S,
        );
        cfg.planet.planet_soi_r_m = visual_planet_soi_r_m(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            cfg.planet.ecc_cap,
        );
        cfg.planet.n_planets = VISUAL_N_PLANETS;
        // ╔══════════════════════════════════════════════════════════════════════════════════════════╗
        // ║ THROWAWAY — THE TINY-WORLD NUMBERS. DELETE WHOLESALE WITH THIS PRESET.                    ║
        // ║                                                                                          ║
        // ║ These exist for ONE purpose: to fit several star systems into a metre-scale galaxy so the ║
        // ║ realm wake/sleep rule can be watched working at SYSTEM level before the world goes true-  ║
        // ║ scale. They are proportions of a toy and mean nothing in a real galaxy, where the count   ║
        // ║ comes from the census, the spacing from real astronomy, and a system is ~1e13 m not 150.  ║
        // ║                                                                                          ║
        // ║ NOTHING above this block is throwaway — the N-system loop, the seed-drawn census, the     ║
        // ║ per-system streams and the overlap fence are the final world's machinery.                 ║
        // ║                                                                                          ║
        // ║ SIZED SO THE RULE IS ACTUALLY EXERCISED: a system wakes once it subtends the visibility   ║
        // ║ angle, i.e. from `soi * cot(theta/2)` away. At 150 m and 8 degrees that is ~2145 m, so the ║
        // ║ ring must be WIDER than that or every system is awake from login and the rule never       ║
        // ║ changes its answer — you would prove crossing but never observe waking.                   ║
        // ╚══════════════════════════════════════════════════════════════════════════════════════════╝
        cfg.galaxy.system_count_lo = VISUAL_N_SYSTEMS;
        cfg.galaxy.system_count_hi = VISUAL_N_SYSTEMS;
        cfg.stellar.system_ring_r_m =
            VISUAL_SYSTEM_SOI_R_M * visibility_factor(VISIBILITY_THETA_MIN_RAD) * VISUAL_RING_SLACK;
        // The galaxy must CONTAIN the ring, or a star sits outside its own galaxy and the space between
        // stars belongs to nothing. It therefore exceeds the client's box-cull — which is CORRECT, not a
        // regression: the owner's ruling is that a containment boundary is never drawn as an object. The
        // galaxy stops being scenery and goes back to being what it is, an authority volume.
        //
        // AND it must clear the TWO-LEVEL BOUND (owner ruling 2026-08-15): no planet — a grandchild of
        // the galaxy — may subtend the visibility threshold from just outside the shell. The shell is
        // SOLVED from both constraints ([`galaxy_shell_r_m`]), never picked: the worst descendant is the
        // OUTER planet judged at the apoapsis of the eccentricity CAP (the same worst-instant bound the
        // `ChildReach` fence and the compression solve use), reaching
        // `outer_sma · (1 + ecc_cap) + planet_soi` from its star at the worst instant of the worst seed.
        let worst_descendant_reach_m = visual_outer_sma_render_m(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            cfg.planet.ecc_cap,
        ) * (1.0 + cfg.planet.ecc_cap);
        cfg.scale.galaxy_r_m = galaxy_shell_r_m(
            cfg.stellar.system_ring_r_m,
            VISUAL_SYSTEM_SOI_R_M,
            worst_descendant_reach_m,
            cfg.planet.planet_soi_r_m,
            VISIBILITY_THETA_MIN_RAD,
            VISUAL_SYSTEM_MARGIN_M,
        );
        cfg
    }

    /// The VISUAL-scale preset (D-45(a) FA-5): the STATIC-render expression of the one compressed-real game
    /// geometry (via [`visual_geometry`](UniverseConfig::visual_geometry)) — a 5-planet Kepler system whose
    /// planets ORBIT visibly. Its interest band is the ONE visibility rule: `spin_up_factor ==
    /// tear_down_factor == cot(θ/2)` (a realm streams in once it subtends ≥ θ_min), authored against the
    /// static occupant speed [`VISUAL_OCCUPANT_V_MAX_MPS`]. Under a single visibility factor the geometric
    /// dead-zone collapses, so band validity rests on `occupant_v_max + v_child > 0` (that non-zero speed);
    /// [`visual_demand`](UniverseConfig::visual_demand) is the live demand-cluster expression of the SAME
    /// geometry.
    #[must_use]
    pub fn visual_scale() -> UniverseConfig {
        let vis_factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let mut cfg = UniverseConfig::visual_geometry();
        cfg.interest = InterestConfig {
            spin_up_factor: vis_factor,
            tear_down_factor: vis_factor,
            grace_ticks: VISUAL_AOI_GRACE_TICKS,
            k_safety_extra: VISUAL_AOI_K_SAFETY_EXTRA,
            occupant_v_max_mps: VISUAL_OCCUPANT_V_MAX_MPS,
            tick_dt_s: AOI_TICK_DT_S,
        };
        cfg
    }

    /// The VISUAL-DEMAND preset (RLM realistic-demo Slice 0): the LIVE demand-cluster expression of the
    /// SAME compressed-real geometry as [`visual_scale`](UniverseConfig::visual_scale) (via
    /// [`visual_geometry`](UniverseConfig::visual_geometry)), with the interest band built the `walk_demand`
    /// way — the two DYNAMICS inputs are ARGUMENTS from the live cluster (`occupant_v_max_mps` = the
    /// occupant's max speed `move_speed · time_multiplier`, `tick_dt_s` = the cluster's seconds-per-tick),
    /// the loiter grace converted from seconds ([`grace_ticks_from_seconds`]), and the SAME `cot(θ/2)`
    /// visibility factor for both edges. It reuses [`generate_system_forest`]/[`planet_elements`]/
    /// [`moving_children_for_config`] verbatim (zero new generator control flow).
    ///
    /// PRECONDITION (the equal-factor tripwire): under a single visibility factor
    /// (`spin_up_factor == tear_down_factor`) the geometric dead-zone collapses, so band validity requires
    /// `occupant_v_max_mps + v_child > 0` — a LIVE occupant. A zero-relative-velocity band is
    /// [`BandError::InvalidEdges`], NEVER a valid inert band (and `to_regions`'s `.expect` would panic at
    /// boot on one). The shipped callers thread a positive speed (the demand cluster's 15 m/s ship); an
    /// idle-occupant (`v_rel = 0`) preset under one factor has no valid band (a deferred design choice, not
    /// papered over with an epsilon — see `equal_visibility_factor_with_zero_v_rel_is_err_not_panic`).
    #[must_use]
    pub fn visual_demand(occupant_v_max_mps: f64, tick_dt_s: f64) -> UniverseConfig {
        let vis_factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let mut cfg = UniverseConfig::visual_geometry();
        cfg.interest = InterestConfig {
            spin_up_factor: vis_factor,
            tear_down_factor: vis_factor,
            grace_ticks: grace_ticks_from_seconds(WALK_DEMAND_AOI_GRACE_S, tick_dt_s),
            k_safety_extra: WALK_DEMAND_AOI_K_SAFETY_EXTRA,
            occupant_v_max_mps,
            tick_dt_s,
        };
        cfg
    }

    /// **THE WORLD.** Not a preset, not a scale, not a variant — the only universe this program has.
    ///
    /// WHY THIS EXISTS AND WHY EVERYTHING ELSE HERE IS GOING. There used to be a knob, and on a live
    /// cluster it was MEASURED holding two different values at once: the orchestrator booted one world
    /// and the gateway another, in the same launch, from the same script. Logins were placed by one
    /// universe's rules and simulated by another's. That is not a bug in either half — it is what a knob
    /// IS, and no amount of care downstream removes it. The owner's ruling, twice now: one world, all the
    /// time, with nothing to select.
    ///
    /// The two arguments are NOT a choice of world. They are facts about the cluster running it — how
    /// fast an occupant may travel and how long a tick lasts — which the interest band has to be measured
    /// against or it is measured against a number nobody uses.
    ///
    /// ⚠ ITS SIZES ARE NOT FINAL, and that is a separate, sequenced piece of work rather than a variant
    /// hiding here: the geometry below still carries the tiny-world numbers, stations and areas are still
    /// hand-placed rather than generated, and the galaxy is not yet a lattice of cells. Those land as
    /// changes to THIS world's numbers. Never as a second one.
    #[must_use]
    pub fn world(occupant_v_max_mps: f64, tick_dt_s: f64) -> UniverseConfig {
        UniverseConfig::visual_demand(occupant_v_max_mps, tick_dt_s)
    }

    /// The WALK-demand preset (RLM 5f-4): the EXACT walk-scale geometry with the AoI band turned LIVE, so a
    /// WALKING occupant drives demand-driven spin-up/down over the static walk forest. Differs from
    /// [`walk_scale`](UniverseConfig::walk_scale) in the `interest` field ONLY (geometry byte-identical). The
    /// two DYNAMICS inputs are ARGUMENTS from the live cluster: `occupant_v_max_mps` (the occupant's max
    /// speed `move_speed · time_multiplier`) and `tick_dt_s` (the cluster's seconds-per-tick) — so the
    /// anti-thrash pad + the loiter grace are measured against the speed the sim integrates and the rate it
    /// ticks at (closing the M-2 two-home owe; no hardcoded `AOI_TICK_DT_S`).
    #[must_use]
    pub fn walk_demand(occupant_v_max_mps: f64, tick_dt_s: f64) -> UniverseConfig {
        let mut cfg = UniverseConfig::walk_scale();
        cfg.interest = InterestConfig {
            spin_up_factor: WALK_DEMAND_AOI_SPIN_UP_FACTOR,
            tear_down_factor: WALK_DEMAND_AOI_TEAR_DOWN_FACTOR,
            grace_ticks: grace_ticks_from_seconds(WALK_DEMAND_AOI_GRACE_S, tick_dt_s),
            k_safety_extra: WALK_DEMAND_AOI_K_SAFETY_EXTRA,
            occupant_v_max_mps,
            tick_dt_s,
        };
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celestial::KEPLER_ECC_MAX;
    use vd_core::geometry::{region_depth, region_signed_distance};
    use vd_core::realm_coord::RealmCoord;
    use vd_core::realm_path::{RealmKindTag, RealmPath};
    use vd_core::worldgen::{coord_of_realm, default_home_realm};

    fn regions() -> Vec<RealmRegion> {
        realm_regions_for(0)
    }

    /// THE world at a nominal occupant speed — the config-driven twin the bins boot uses.
    fn boot_world_for_tests() -> WorldView {
        WorldView::generated(0, &UniverseConfig::world(15.0, 0.05))
    }

    #[test]
    fn the_world_preset_is_the_visual_demand_geometry_and_the_alias_is_its_twin() {
        // `UniverseConfig::world` IS `visual_demand` (SL5: one world, the name states the law), and
        // the held-config alias builds the SAME neighbourhood as the fn it delegates to (HR3: the
        // seam stays closed — two identical functions consulting different worlds is how the login
        // side and the simulating side once described different universes from one seed).
        let world = UniverseConfig::world(15.0, 0.05);
        let demand = UniverseConfig::visual_demand(15.0, 0.05);
        assert_eq!(
            generate_system_forest(0, &world).len(),
            generate_system_forest(0, &demand).len(),
            "one geometry"
        );
        let held = std::collections::BTreeSet::from([RealmId::System(7)]);
        assert_eq!(
            realm_neighbourhood_for_held_config(0, &held, &world),
            realm_neighbourhood_for_config(0, &held, &world),
            "the alias is byte-equal to its twin"
        );
    }

    #[test]
    fn the_sibling_fence_judges_an_orbit_on_its_shell_never_its_epoch() {
        // The overlap fence skips a KEPLER sibling on BOTH sides of the pair loop: an orbit is
        // judged on its shell (the SOI nesting fence), not on where its epoch anchor happens to sit
        // — an epoch-position overlap between an orbiting body and a static one is not a defect.
        let orbital = Placement::Orbital(OrbitalElements {
            sma: 1.0e11,
            ecc: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: 1.989e30,
        });
        let body = |realm: RealmId, placement: Placement| GeneratedBody {
            realm,
            parent: Some(RealmId::System(1)),
            shape: Boundary::Shell { r: 10.0 },
            placement,
            photometrics: None,
        };
        // A static + an ORBITING sibling at the "same place": no overlap verdict — the orbiter is
        // skipped by the inner arm.
        let mixed = [
            body(RealmId::Planet(1), Placement::StaticOffset(DVec3::ZERO)),
            body(RealmId::Planet(2), orbital),
        ];
        assert!(siblings_disjoint(&mixed).is_ok());
        // Two STATIC siblings genuinely overlapping: the fence still fires (non-vacuity).
        let clash = [
            body(RealmId::Planet(1), Placement::StaticOffset(DVec3::ZERO)),
            body(
                RealmId::Planet(2),
                Placement::StaticOffset(DVec3::new(5.0, 0.0, 0.0)),
            ),
        ];
        assert!(siblings_disjoint(&clash).is_err());
    }

    #[test]
    fn coord_of_realm_resolves_the_full_root_rooted_lineage_and_is_none_for_a_ship() {
        // The un-lossy RealmId→RealmCoord a source shard uses to KeepAlive-demand a crossing DEST's whole
        // ancestor chain (the Symptom-B freeze fix). A System resolves to [Universe, Galaxy, System]; a
        // Planet one level deeper; a ship (entity-backed, no seed level) resolves to None.
        let regions = realm_regions_for(0);
        let want_sys = RealmCoord::from_path(RealmPath::from_levels(vec![
            level_of(UNIVERSE).expect("Universe is a seed realm"),
            level_of(GALAXY).expect("Galaxy is a seed realm"),
            level_of(SYSTEM_A).expect("System A is a seed realm"),
        ]))
        .expect("a 3-level path has a leaf");
        let want_planet = want_sys.child(level_of(PLANET_A).expect("Planet A is a seed realm"));
        assert_eq!(coord_of_realm(&regions, SYSTEM_A), Some(want_sys));
        assert_eq!(want_planet.path().levels().len(), 4); // full lineage, never lowered()
        assert_eq!(coord_of_realm(&regions, PLANET_A), Some(want_planet));
        // A ship is entity-backed (no seed RealmLevel) ⇒ None — the `?` early-return arm.
        let ship = RealmId::Ship(vd_core::ids::EntityId::pack(
            vd_core::entity_kind::EntityKind::Ship,
            1,
            1,
            1,
        ));
        assert_eq!(coord_of_realm(&regions, ship), None);
    }

    #[test]
    fn region_signed_distance_is_frame_aware_and_identity_at_p3() {
        let rs = regions();
        let system_a = rs
            .iter()
            .find(|r| r.realm == SYSTEM_A)
            .expect("system A is in the forest");
        // At the system center: signed distance = -r_soi (fully inside). Identity frame ⇒ pos unchanged.
        let at_centre = vd_core::pose::StampedPose::at_rest(
            vd_core::pose::FrameRef::SystemSpace { system_seed: 0 },
            DVec3::ZERO,
            vd_core::ids::UniverseTick(0),
        );
        // The book: anchored on the pose's own frame, with the system's frame at the identity (the
        // P3 shipping shape — every placement is the identity, so the reframe moves nothing).
        let book = vd_core::placement::PlacementBook::new(
            at_centre.frame,
            at_centre.universe_tick,
            vec![(system_a.frame, vd_core::frame::FramePlacement::identity())],
        );
        let sd = region_signed_distance(&at_centre, system_a, &book).expect("ok");
        assert!(
            (sd - (-SYSTEM_SOI_R_M)).abs() < 1e-9,
            "center is r_soi inside: {sd}"
        );
    }

    #[test]
    fn realm_neighbourhood_scopes_to_own_ancestors_and_children_never_siblings() {
        // System 7's shard: own + ancestors (Galaxy, Universe) + children Planet 7 AND Station 7 (a
        // first-class child under System 7) — NOT sibling System 8, NOT the grandchild Area 7.
        let n7: Vec<RealmId> = realm_neighbourhood_for(0, SYSTEM_A)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(n7.contains(&SYSTEM_A));
        assert!(n7.contains(&GALAXY));
        assert!(n7.contains(&UNIVERSE));
        assert!(n7.contains(&PLANET_A));
        assert!(
            n7.contains(&STATION_A),
            "the Station is an OWNED child of System 7 — the shard scans it",
        );
        assert!(
            !n7.contains(&SYSTEM_B),
            "a shard NEVER loads a sibling — the scale-bounded rule",
        );
        assert!(
            !n7.contains(&AREA_A),
            "Area 7 is a grandchild (under Planet 7), not a direct child of System 7",
        );
        assert_eq!(n7.len(), 5);
        // The GALAXY shard: own + ancestor Universe + children System 7 & 8 (the between-space owner that
        // routes a sibling crossing) — NOT Planet 7 (a grandchild, not a direct child).
        let ng: Vec<RealmId> = realm_neighbourhood_for(0, GALAXY)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(ng.contains(&GALAXY));
        assert!(ng.contains(&UNIVERSE));
        assert!(ng.contains(&SYSTEM_A));
        assert!(ng.contains(&SYSTEM_B));
        assert!(
            !ng.contains(&PLANET_A),
            "a grandchild is not a direct child"
        );
        assert_eq!(ng.len(), 4);
        // The STATION 7 shard: its own realm + its ANCESTOR CHAIN (System 7, Galaxy, Universe) and NO
        // children (a leaf) — it never pulls its sibling Planet 7 (they share the System 7 parent).
        let nst: Vec<RealmId> = realm_neighbourhood_for(0, STATION_A)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(nst.contains(&STATION_A));
        assert!(nst.contains(&SYSTEM_A));
        assert!(nst.contains(&GALAXY));
        assert!(nst.contains(&UNIVERSE));
        assert!(
            !nst.contains(&PLANET_A),
            "the Station never loads its sibling Planet 7",
        );
        assert_eq!(nst.len(), 4);
        // The AREA 7 shard: its own realm + its ANCESTOR CHAIN (Planet 7, System 7, Galaxy, Universe) and
        // NO children — it never pulls its sibling Station 7 (they share the System 7 ancestor, not a parent).
        let nar: Vec<RealmId> = realm_neighbourhood_for(0, AREA_A)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(nar.contains(&AREA_A));
        assert!(nar.contains(&PLANET_A));
        assert!(nar.contains(&SYSTEM_A));
        assert!(nar.contains(&GALAXY));
        assert!(nar.contains(&UNIVERSE));
        assert!(
            !nar.contains(&STATION_A),
            "the Area never loads the Station (they are not parent/child)",
        );
        assert_eq!(nar.len(), 5);
        // A shard hosting an unknown realm ⇒ empty neighbourhood ⇒ the detector is inert (safe degrade).
        // Both appended kinds (Station/Area) at an ABSENT seed (99) degrade to empty — the seed-7 plant
        // above does NOT make every Station/Area live.
        assert!(realm_neighbourhood_for(0, RealmId::Station(99)).is_empty());
        assert!(realm_neighbourhood_for(0, RealmId::Area(99)).is_empty());
    }

    #[test]
    fn a_single_held_realm_neighbourhood_union_equals_the_single_neighbourhood() {
        // Co-hosting DEGENERATE case: a held-set of exactly one realm is byte-identical to
        // `realm_neighbourhood_for` — the single-realm shard path is untouched.
        for r in [SYSTEM_A, GALAXY, PLANET_A, STATION_A] {
            let single = realm_neighbourhood_for(0, r);
            let held = realm_neighbourhood_for_held(0, &std::collections::BTreeSet::from([r]));
            assert_eq!(
                single, held,
                "the held-set union for {{{r}}} equals its single neighbourhood",
            );
        }
    }

    #[test]
    fn a_cohosted_system_plus_children_union_reaches_the_deepest_grandchild_area() {
        // The un-hosted-child cure: a shard co-hosting System 7 + its children (Planet/Station/Area) must
        // evaluate the DEEPEST region (Area 7, a GRANDCHILD of System 7 absent from System 7's OWN
        // neighbourhood). The union reaches it via Planet 7 being held.
        let held = std::collections::BTreeSet::from([SYSTEM_A, PLANET_A, STATION_A, AREA_A]);
        let realms: Vec<RealmId> = realm_neighbourhood_for_held(0, &held)
            .iter()
            .map(|r| r.realm)
            .collect();
        for expected in [UNIVERSE, GALAXY, SYSTEM_A, PLANET_A, STATION_A, AREA_A] {
            assert!(realms.contains(&expected), "the union includes {expected}");
        }
        // System B (a SIBLING of System 7 — never a child/ancestor of any held realm) is EXCLUDED.
        assert!(
            !realms.contains(&SYSTEM_B),
            "a sibling system is never in the co-hosting union",
        );
        // The union deduplicates (Universe/Galaxy/System 7 appear once even though several held realms
        // share them as ancestors) — the region set is a valid single-root forest.
        assert_eq!(
            realms.len(),
            6,
            "6 distinct regions (7-forest minus the sibling System B)"
        );
    }

    #[test]
    fn realm_neighbourhood_for_config_excludes_sibling_planets_over_the_visual_forest() {
        // The flap cure at VISUAL scale: a planet shard's neighbourhood is its own realm + ancestors + the
        // children it authors — NEVER its sibling planets. A shard cannot place a realm it does not author, so
        // folding a sibling collapses it to the origin and a hosted occupant reads as inside all of them at
        // once (the production hot-potato). Unlike the walk forest, the visual system forest has MULTIPLE
        // orbiting planets, so this is where the exclusion actually bites.
        let cfg = UniverseConfig::visual_scale();
        let forest = realm_regions_for_config(0, &cfg);
        let planets: Vec<RealmId> = forest
            .iter()
            .filter(|r| matches!(r.realm, RealmId::Planet(_)))
            .map(|r| r.realm)
            .collect();
        assert!(
            planets.len() >= 2,
            "the visual forest must have sibling planets to distinguish (got {planets:?})",
        );
        let target = planets[0];
        let sibling = planets[1];
        let parent = forest
            .iter()
            .find(|r| r.realm == target)
            .and_then(|r| r.parent)
            .expect("a visual-forest planet has a parent system");
        let scope: Vec<RealmId> =
            realm_neighbourhood_for_config(0, &std::collections::BTreeSet::from([target]), &cfg)
                .iter()
                .map(|r| r.realm)
                .collect();
        assert!(scope.contains(&target), "own realm is in scope");
        assert!(
            scope.contains(&parent),
            "the parent system (an ancestor) is in scope",
        );
        assert!(
            !scope.contains(&sibling),
            "a SIBLING planet is NEVER in scope — the origin-stacking flap cure",
        );
    }

    // ---- D-45(a) Slice 3b: UniverseConfig -------------------------------------------

    #[test]
    fn walk_scale_equals_the_named_geometry_consts() {
        let c = UniverseConfig::walk_scale();
        assert_eq!(c.scale.universe_r_m, UNIVERSE_R_M);
        // The galaxy is DERIVED to contain the ring of stars, no longer the walk constant: it must hold
        // every system with its reach, or a star sits outside its own galaxy.
        assert!(c.scale.galaxy_r_m > c.stellar.system_ring_r_m + c.stellar.system_soi_r_m);
        assert_eq!(c.scale.render_extent_m, MAX_RENDERABLE_EXTENT_M);
        assert_eq!(c.stellar.system_soi_r_m, SYSTEM_SOI_R_M);
        assert_eq!(c.planet.planet_soi_r_m, PLANET_SOI_R_M);
        assert_eq!(c.satellite.planet_offset_m, PLANET_A_OFFSET_M);
        assert_eq!(c.satellite.system_b_offset_m, SYSTEM_B_OFFSET_M);
        assert_eq!(c.satellite.station_offset_m, STATION_A_OFFSET_M);
        assert_eq!(c.satellite.station_half_m, STATION_HALF_M);
        assert_eq!(c.satellite.area_offset_m, AREA_OFFSET_M);
        assert_eq!(c.satellite.area_half_m, AREA_HALF_M);
        assert_eq!(c.band.inset_m, CONTAINMENT_INSET_M);
        assert_eq!(c.band.outset_m, CONTAINMENT_OUTSET_M);
        // The galaxy is renderable (drawn as the containing box); the Universe is not.
        assert!(c.scale.galaxy_r_m < c.scale.render_extent_m);
        // The 3 FA-5 fields are INERT on walk (self-consistent, never read on the StaticOffset path).
        assert_eq!(c.scale.au_to_render_m, PLANET_A_OFFSET_M / ORBITAL_A0_AU);
        assert_eq!(c.stellar.central_mass_kg, CANONICAL_STAR_MASS_KG);
        assert_eq!(c.planet.n_planets, 0);
    }

    #[test]
    fn universe_config_presets_serde_round_trip() {
        for c in [UniverseConfig::walk_scale(), UniverseConfig::visual_scale()] {
            let bytes = postcard::to_allocvec(&c).expect("encode");
            let back: UniverseConfig = postcard::from_bytes(&bytes).expect("decode");
            assert_eq!(c, back);
        }
    }

    #[test]
    fn every_preset_ecc_cap_is_within_the_kepler_domain() {
        // Fail-loud cross-slice invariant: no preset may cap eccentricity above the fixed Kepler
        // solver's convergence domain (KEPLER_ECC_MAX).
        assert!(UniverseConfig::walk_scale().planet.ecc_cap <= KEPLER_ECC_MAX);
        assert!(UniverseConfig::visual_scale().planet.ecc_cap <= KEPLER_ECC_MAX);
    }

    #[test]
    fn walk_band_builds_the_static_band() {
        UniverseConfig::walk_scale()
            .band
            .build()
            .expect("walk band is valid by construction");
    }

    #[test]
    fn frost_thresholds_reads_the_planet_config() {
        let ft = UniverseConfig::walk_scale().planet.frost_thresholds();
        assert_eq!(ft, FrostThresholds::CANONICAL);
    }

    // ---- D-45(a) Slice 3c: generate -> to_regions lowering (byte-identity) -----------

    #[test]
    fn realm_regions_for_matches_the_frozen_pre_generator_golden() {
        // A HAND-AUTHORED frozen golden (literal values, independent of the generator path): if
        // to_regions drifts any coordinate / shape / parent, this fails against the literals — NOT
        // a self-referential capture of the (rewritten) realm_regions_for output.
        let rs = realm_regions_for(0);
        let expected: [(RealmId, DVec3, Boundary, Option<RealmId>); 7] = [
            (UNIVERSE, DVec3::ZERO, Boundary::Shell { r: 1.0e9 }, None),
            (
                GALAXY,
                DVec3::ZERO,
                Boundary::Shell { r: 180.0 },
                Some(UNIVERSE),
            ),
            (
                SYSTEM_A,
                DVec3::ZERO,
                Boundary::Shell { r: 40.0 },
                Some(GALAXY),
            ),
            (
                PLANET_A,
                DVec3::new(20.0, 0.0, 0.0),
                Boundary::Shell { r: 10.0 },
                Some(SYSTEM_A),
            ),
            (
                SYSTEM_B,
                DVec3::new(130.0, 0.0, 0.0),
                Boundary::Shell { r: 40.0 },
                Some(GALAXY),
            ),
            (
                STATION_A,
                DVec3::new(-25.0, 0.0, 0.0),
                Boundary::Aabb {
                    half: DVec3::splat(5.0),
                },
                Some(SYSTEM_A),
            ),
            (
                AREA_A,
                // +5 FROM ITS PARENT, the planet at +20 — so still +25 in the system's frame, the same
                // place it has always occupied. The golden used to read 25 here, an absolute on a field
                // that means "offset from my parent"; it went unnoticed while nothing converted between
                // levels. The number changed; the world did not.
                DVec3::new(5.0, 0.0, 0.0),
                Boundary::Aabb {
                    half: DVec3::splat(3.0),
                },
                Some(PLANET_A),
            ),
        ];
        assert_eq!(rs.len(), 7);
        for (r, (realm, offset, shape, parent)) in rs.iter().zip(expected) {
            assert_eq!(r.realm, realm);
            assert_eq!(r.center.offset(), offset);
            assert_eq!(
                r.center.cell(),
                glam::I64Vec3::ZERO,
                "step-1 keeps cell == ZERO"
            );
            assert_eq!(r.shape, shape);
            assert_eq!(r.parent, parent);
            assert_eq!(
                r.frame,
                frame_for_realm(realm, parent).expect("canonical frame")
            );
        }
        // The whole forest shares the one static walk band.
        let band = UniverseConfig::walk_scale()
            .band
            .build()
            .expect("walk band");
        for r in &rs {
            assert_eq!(r.band, band);
        }
    }

    #[test]
    fn to_regions_gives_an_orbital_body_a_zero_center_position_authored_by_the_frame() {
        // The Orbital placement arm: a moving body carries NO baked position — its boundary sits at the
        // ZERO origin of its own frame; its live pose is authored per tick into the placement book.
        let elements = OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        };
        let body = GeneratedBody {
            realm: RealmId::Planet(42),
            parent: Some(RealmId::System(42)),
            shape: Boundary::Shell { r: 9.0e8 },
            placement: Placement::Orbital(elements),
            photometrics: None,
        };
        let regions = to_regions(&[body], &UniverseConfig::visual_scale());
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].center.cell(), glam::I64Vec3::ZERO);
        assert_eq!(regions[0].center.offset(), DVec3::ZERO);
    }

    #[test]
    fn placement_offset_reads_static_verbatim_and_the_orbital_tick_zero_position() {
        // The frame-local offset utility. `StaticOffset` returns its fixed vector verbatim; `Orbital`
        // returns the tick-0 ephemeris position. `region_center_of` only ever feeds it `StaticOffset`
        // (movers are frame-AUTHORED ⇒ ZERO center, the moving-frame fix), so its `Orbital` arm — the
        // orbital-position fold the S2 own-absolute path reuses — is exercised directly here.
        let v = DVec3::new(3.0, -4.0, 5.0);
        assert_eq!(placement_offset(Placement::StaticOffset(v)), v);
        let elements = OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        };
        assert_eq!(
            placement_offset(Placement::Orbital(elements)),
            orbital_state(&elements, 0.0).position
        );
    }

    #[test]
    fn moving_children_selects_only_direct_orbital_children() {
        // FA-2b: the AUTHORED moving-child roster keeps ONLY a hosted realm's DIRECT children whose
        // placement is `Orbital` — a STATIC child, an orbital NON-child (someone else's), and an
        // orbital GRANDchild are all excluded. Exercises both `orbital_of` arms + the parent filter.
        let elements = OrbitalElements {
            sma: 1.5e11,
            ecc: 0.1,
            inclination: 0.4,
            raan: 0.3,
            arg_periapsis: 0.9,
            mean_anomaly_epoch: 0.2,
            central_mass: 1.989e30,
        };
        let orbital_child = GeneratedBody {
            realm: RealmId::Planet(1),
            parent: Some(RealmId::System(7)),
            shape: Boundary::Shell { r: 9.0e8 },
            placement: Placement::Orbital(elements),
            photometrics: None,
        };
        let static_child = GeneratedBody {
            realm: RealmId::Station(2),
            parent: Some(RealmId::System(7)),
            shape: Boundary::Shell { r: 1.0e6 },
            placement: Placement::StaticOffset(DVec3::new(5.0, 0.0, 0.0)),
            photometrics: None,
        };
        let orbital_non_child = GeneratedBody {
            realm: RealmId::Planet(3),
            parent: Some(RealmId::System(99)),
            shape: Boundary::Shell { r: 9.0e8 },
            placement: Placement::Orbital(elements),
            photometrics: None,
        };
        let bodies = [orbital_child, static_child, orbital_non_child];
        assert_eq!(
            moving_children(&bodies, RealmId::System(7)),
            vec![(RealmId::Planet(1), elements)],
        );
    }

    #[test]
    fn moving_children_for_is_empty_at_walk_scale() {
        // The byte-identity guarantee: the walk roster is ALL `StaticOffset`, so a walk-scale shard
        // authors NO moving child — `frame_context` registers every region at identity, unchanged.
        assert!(moving_children_for(0, RealmId::System(7)).is_empty());
        assert!(moving_children_for(0, RealmId::System(8)).is_empty());
    }

    // ===== RLM Step 2: per-realm AoI config + generator accessors ========================

    #[test]
    fn interest_config_build_inert_at_zero_factor() {
        let inert = InterestConfig::inert();
        assert!(!inert.is_live());
        assert_eq!(
            inert.build(100.0, 5.0).expect("inert always ok"),
            AoiConfig::inert()
        );
    }

    #[test]
    fn one_world_answers_and_checks_with_the_same_contents() {
        // THE DEFECT THIS TYPE RETIRES, measured. A home used to be RESOLVED against the generated world and
        // then VALIDATED against the hand-placed one. With one star the two happened to agree; with several,
        // the stars a seed draws are simply absent from the hand-placed world, so a perfectly valid home
        // beside any other star fails its own defence — and the gateway panics on the spot.
        let cfg = UniverseConfig::visual_scale();
        let generated = WorldView::generated(0, &cfg);
        let placed = WorldView::hand_placed(&cfg);
        // The two worlds really do hold different things — otherwise the rest of this proves nothing.
        let only_generated: Vec<RealmId> = generated
            .regions()
            .iter()
            .map(|r| r.realm)
            .filter(|realm| !placed.contains_realm(*realm))
            .collect();
        assert!(
            !only_generated.is_empty(),
            "the generated world holds stars the hand-placed one does not"
        );
        // …and EVERY one of them passes the check when the check reads the world that produced it.
        for realm in only_generated {
            assert!(generated.contains_realm(realm));
        }
    }

    #[test]
    fn the_hand_placed_world_is_the_one_with_structures_to_stand_in() {
        // Stations are built by players and areas mostly are, so the generator emits neither — which is why
        // a test that needs one places it. Pinned both ways: the placed world HAS them, the generated world
        // has NONE, and that is the whole reason both exist.
        let cfg = UniverseConfig::walk_scale();
        let placed = WorldView::hand_placed(&cfg);
        let generated = WorldView::generated(0, &cfg);
        let structures = |w: &WorldView| -> usize {
            w.regions()
                .iter()
                .filter(|r| matches!(r.realm, RealmId::Station(_) | RealmId::Area(_)))
                .count()
        };
        assert_eq!(structures(&generated), 0);
        assert_eq!(structures(&placed), 2);
    }

    #[test]
    fn the_world_answers_neighbourhood_from_its_own_contents() {
        // The remaining questions a world is asked, each equal to the free function it delegates to — so
        // holding a world can never mean a different answer than deriving one, only a cheaper one. It was
        // also asked for a realm's ORIGIN CHAIN; that question no longer exists, because no realm is
        // entitled to know where it sits.
        let cfg = UniverseConfig::visual_scale();
        let world = WorldView::generated(0, &cfg);
        let held = std::collections::BTreeSet::from([SYSTEM_A]);
        assert_eq!(
            world.neighbourhood(&held),
            realm_neighbourhood_for_config(0, &held, &cfg)
        );
        assert_eq!(world.regions(), realm_regions_for_config(0, &cfg));
    }

    #[test]
    fn the_lowered_world_is_the_regions_and_nothing_else() {
        // `WorldView::lowered` hands the connection plane the region forest VERBATIM — the same
        // slice `regions()` answers with — and (by its type) nothing a body knows: the lowered
        // value is `vd_core::worldgen::WorldRealms`, a crate with no path back to an orbit (SL4).
        let cfg = UniverseConfig::visual_scale();
        let world = WorldView::generated(0, &cfg);
        assert_eq!(world.lowered().regions(), world.regions());
    }

    #[test]
    fn no_two_static_siblings_ever_overlap_in_any_shipped_world() {
        // THE FENCE, RUN. A position inside two overlapping siblings has two equally valid owners, and
        // which shard gets you falls out of iteration order — see `siblings_disjoint` for why that is
        // unrepairable downstream. The generator is deterministic, so proving it over the shipped presets
        // and a spread of seeds proves the worlds that can actually be booted.
        //
        // Seeds swept rather than one sampled: the star ring is drawn from the galaxy's own stream, so a
        // seed that happened to draw a crowded galaxy is exactly the case a single-seed test would miss.
        for seed in 0..64_u64 {
            for config in [
                UniverseConfig::visual_scale(),
                UniverseConfig::visual_demand(15.0, 0.02),
            ] {
                let bodies = generate_system_forest(seed, &config);
                assert_eq!(siblings_disjoint(&bodies), Ok(()));
            }
            assert_eq!(
                siblings_disjoint(&generate_walk_forest(&UniverseConfig::walk_scale())),
                Ok(())
            );
        }
    }

    #[test]
    fn the_fence_refuses_two_siblings_that_reach_each_other() {
        // The fence's OWN failing case, so passing above is a fact and not a fence that never says no.
        // Two shells of radius 1 whose centres are 1.5 apart: their surfaces interpenetrate.
        let shell = Boundary::Shell { r: 1.0 };
        let at = |x: f64| Placement::StaticOffset(DVec3::new(x, 0.0, 0.0));
        let body = |realm, placement| GeneratedBody {
            realm,
            parent: Some(GALAXY),
            shape: shell,
            placement,
            photometrics: None,
        };
        let a = RealmId::System(1);
        let b = RealmId::System(2);
        assert_eq!(
            siblings_disjoint(&[body(a, at(0.0)), body(b, at(1.5))]),
            Err(SiblingsOverlap {
                a,
                b,
                parent: GALAXY
            })
        );
        // …and clears once they are pushed apart past the sum of their radii.
        assert_eq!(
            siblings_disjoint(&[body(a, at(0.0)), body(b, at(2.5))]),
            Ok(())
        );
    }

    #[test]
    fn the_first_system_draws_the_stream_its_realm_path_names() {
        // HR1 restated as a measurement: every shard hosting a system must draw the IDENTICAL per-system
        // stream, which holds only if the stream's lineage IS the realm's path from the universe down.
        // Pinned on system zero because that is the one with a named seed to compare against.
        assert_eq!(
            [UNIVERSE_SEED, GALAXY_SEED, system_seed_at(0)],
            SYSTEM_A_LINEAGE
        );
    }

    #[test]
    fn a_planet_is_visible_from_anywhere_inside_its_own_system() {
        // THE REGRESSION THIS PINS (measured 2026-08-07, live): a planet's wake radius was 59.5 m while the
        // system containing it was 150 m in radius, so crossing into a system showed you an EMPTY volume —
        // its planets only existed once you were nearly on top of one.
        //
        // The requirement, stated geometrically: two points inside a sphere of radius R are at most 2R
        // apart, so a planet reaching `2 · (its system's extent)` is visible from ANY point in that system,
        // including the far side of the boundary you just crossed. Derived from the forest, never a pinned
        // literal — so it keeps holding if the geometry is re-tuned, and fails loudly if the ONE visibility
        // angle is widened back.
        let regions = realm_regions_for_config(0, &UniverseConfig::visual_scale());
        let extent_of = |realm: RealmId| {
            regions
                .iter()
                .find(|r| r.realm == realm)
                .map(|r| r.shape.circumscribed_extent())
        };
        let mut planets_checked = 0_u32;
        for r in regions
            .iter()
            .filter(|r| matches!(r.realm, RealmId::Planet(_)))
        {
            let parent = r.parent.expect("a planet always sits inside its system");
            let system_extent = extent_of(parent).expect("the parent system is in the same forest");
            // The reach and the bound as plain values FIRST (HR5: an expression inside a passing
            // assert's message is a region no run evaluates), then a literal-message assert.
            let (reach, must_cross) = (r.aoi.spin_up_r_m(), 2.0 * system_extent);
            assert!(
                reach >= must_cross,
                "a planet's reach must cross its own system"
            );
            planets_checked += 1;
        }
        // …and the loop actually ran, so a forest that stopped emitting planets cannot pass vacuously.
        assert_eq!(planets_checked, VISUAL_N_PLANETS * VISUAL_N_SYSTEMS);
    }

    #[test]
    fn interest_config_build_live() {
        let live = UniverseConfig::visual_scale().interest;
        assert!(live.is_live());
        // spin_up_r = extent × the ONE visibility factor cot(θ/2) (v_child 0 ⇒ tear = spin_up widened by
        // the occupant-speed lead, since spin_up_factor == tear_down_factor collapses the geometric gap).
        let band = live.build(100.0, 0.0).expect("valid live");
        assert_eq!(
            band.spin_up_r_m(),
            100.0 * visibility_factor(VISIBILITY_THETA_MIN_RAD)
        );
    }

    #[test]
    fn to_regions_stamps_per_realm_aoi() {
        // Walk: every region inert (byte-identity — the field never changes containment).
        for r in realm_regions_for(0) {
            assert_eq!(r.aoi, AoiConfig::inert());
        }
        // Visual: a Planet's spin_up = its own extent × the visibility factor; a bigger realm (System)
        // reaches farther (same factor, larger extent).
        let visual = realm_regions_for_config(0, &UniverseConfig::visual_scale());
        let planet = visual
            .iter()
            .find(|r| matches!(r.realm, RealmId::Planet(_)))
            .expect("a planet");
        assert_eq!(
            planet.aoi.spin_up_r_m(),
            planet.shape.finite_extent() * visibility_factor(VISIBILITY_THETA_MIN_RAD)
        );
        let system = visual
            .iter()
            .find(|r| r.realm == SYSTEM_A)
            .expect("system A");
        assert!(system.aoi.spin_up_r_m() > planet.aoi.spin_up_r_m());
    }

    // ===== RLM 5f-4a: the walk-demand LIVE AoI band over the walk forest =================
    // The dev demand-cluster's live AoI dynamics: a 2 m/s brisk walk (move_speed × time_multiplier),
    // 50 Hz (0.02 s/tick). Kept here (test-only) — the composer supplies the live cluster values at boot.
    fn walk_demand_regions() -> Vec<RealmRegion> {
        realm_regions_for_walk_config(0, &UniverseConfig::walk_demand(2.0, 0.02))
    }

    #[test]
    fn walk_keeps_aoi_inert_while_visual_stays_live() {
        // Byte-identity: walk keeps AoI OFF (behaviour unchanged); the compressed-real visual
        // band is LIVE under the ONE cot(θ/2) visibility factor.
        for r in realm_regions_for(0) {
            assert_eq!(r.aoi, AoiConfig::inert(), "walk regions stay AoI-inert");
        }
        assert!(
            UniverseConfig::visual_scale().interest.is_live(),
            "the visual band is live under the visibility factor"
        );
    }

    #[test]
    fn walk_demand_differs_from_walk_only_in_aoi() {
        let walk = realm_regions_for_walk_config(0, &UniverseConfig::walk_scale());
        let demand = walk_demand_regions();
        assert_eq!(walk.len(), demand.len(), "same forest topology");
        for (w, d) in walk.iter().zip(demand.iter()) {
            assert_eq!(w.realm, d.realm, "realm unchanged");
            assert_eq!(w.center, d.center, "center unchanged");
            assert_eq!(w.frame, d.frame, "frame unchanged");
            assert_eq!(w.shape, d.shape, "shape unchanged");
            assert_eq!(w.band, d.band, "containment band unchanged");
            assert_eq!(w.parent, d.parent, "parent unchanged");
            assert_eq!(w.aoi, AoiConfig::inert(), "walk region is AoI-inert");
            assert_ne!(
                d.aoi,
                AoiConfig::inert(),
                "walk-demand region has a LIVE AoI band"
            );
        }
    }

    #[test]
    fn walk_config_builders_delegate_byte_identically() {
        // The existing fns delegate to the config builders with walk_scale ⇒ byte-identical output.
        assert_eq!(
            realm_regions_for(0),
            realm_regions_for_walk_config(0, &UniverseConfig::walk_scale()),
            "realm_regions_for delegates byte-identically"
        );
        let all = realm_regions_for(0);
        let a_planet = all
            .iter()
            .find_map(|r| matches!(r.realm, RealmId::Planet(_)).then_some(r.realm))
            .expect("a planet in the walk forest");
        let sets = [
            std::collections::BTreeSet::from([SYSTEM_A]),
            std::collections::BTreeSet::from([a_planet]),
            std::collections::BTreeSet::from([SYSTEM_A, a_planet]),
        ];
        // The FIXTURE scoper agrees with its own single-realm twin over the same roster. It deliberately
        // does NOT agree with the config scoper any more: that one reads the GENERATED world, which holds
        // no player-built station or area. Asserting they match is what let a fixture masquerade as the
        // world in the first place.
        for held in &sets {
            let scoped = realm_neighbourhood_for_held(0, held);
            for r in held {
                for one in realm_neighbourhood_for(0, *r) {
                    assert!(
                        scoped.iter().any(|s| s.realm == one.realm),
                        "the union over {held:?} covers every member's own neighbourhood"
                    );
                }
            }
        }
    }

    #[test]
    fn direct_child_levels_from_seed() {
        // Visual System A hosts N orbiting planets; the roster is exactly those planet levels.
        let config = UniverseConfig::visual_scale();
        let levels = direct_child_levels(0, &config, SYSTEM_A);
        assert_eq!(levels.len(), VISUAL_N_PLANETS as usize);
        assert!(levels.iter().all(|l| l.kind == RealmKindTag::Planet));
    }

    /// The visual-scale system forest at seed 0 (helper for the tests below).
    fn visual_forest() -> Vec<GeneratedBody> {
        generate_system_forest(0, &UniverseConfig::visual_scale())
    }

    // The parameterized derive helpers evaluated at the compressed-real geometry (the EXACT numbers
    // `visual_geometry` threads) — so the helper-driven assertions below stay DRY and can't drift the args.
    fn vis_planet_soi() -> f64 {
        visual_planet_soi_r_m(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            ECC_SIGMA * ECC_CAP_SIGMAS,
        )
    }
    fn vis_outer_sma() -> f64 {
        visual_outer_sma_render_m(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            ECC_SIGMA * ECC_CAP_SIGMAS,
        )
    }
    fn vis_central_mass() -> f64 {
        visual_central_mass_kg(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            ECC_SIGMA * ECC_CAP_SIGMAS,
            VISUAL_TARGET_OUTER_PERIOD_S,
        )
    }

    // ===== RLM realistic-demo Slice 0: the one visibility constant + compressed-real geometry =========

    /// The single most-distant planet region + its epoch ORBIT DISTANCE — the OUTER planet, the one the
    /// star-view visibility rule culls until an occupant closes in. A moving realm's region.center is ZERO
    /// (position authored via the frame), so the orbit distance is derived from the mover ELEMENTS, not the
    /// region center.
    fn outer_planet_orbit(config: &UniverseConfig) -> (RealmRegion, f64) {
        let regions = realm_regions_for_config(0, config);
        moving_children_for_config(0, config, SYSTEM_A)
            .iter()
            .map(|(realm, el)| {
                let region = *regions
                    .iter()
                    .find(|r| r.realm == *realm)
                    .expect("every mover has a region");
                (region, orbital_state(el, 0.0).position.length())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("a planet mover is present")
    }

    /// FINDING 28's measurement: THE world produces `Placement::Orbital` in production — at least one
    /// per system, through the same `UniverseConfig::world` every shipped shard boots. The claim used
    /// to live as a doc comment saying the opposite ("no production producer") under an
    /// `#[allow(dead_code)]`; a gate can be wrong but it cannot be stale.
    #[test]
    fn the_world_produces_an_orbital_planet_in_every_system() {
        let config = UniverseConfig::world(15.0, 0.02);
        let bodies = generate_system_forest(0, &config);
        let systems: Vec<RealmId> = bodies
            .iter()
            .filter(|b| {
                bodies
                    .iter()
                    .any(|c| c.parent == Some(b.realm) && orbital_of(c.placement).is_some())
            })
            .map(|b| b.realm)
            .collect();
        let stars: Vec<RealmId> = bodies
            .iter()
            .filter(|b| matches!(b.realm, RealmId::System(_)) && b.parent == Some(GALAXY))
            .map(|b| b.realm)
            .collect();
        assert_eq!(
            systems, stars,
            "every star system of THE world holds at least one Orbital planet"
        );
        assert!(!stars.is_empty(), "THE world holds star systems");
    }

    #[test]
    fn visibility_factor_is_cot_half_theta() {
        // cot(θ/2) at θ_min = 1.5° — the ONE visibility constant (≈ 76.390), frozen non-self-referentially.
        assert_eq!(
            visibility_factor(VISIBILITY_THETA_MIN_RAD),
            FROZEN_VISIBILITY_FACTOR
        );
    }

    // ===== The generator visibility check (owner ruling 2026-08-15, item 5 + the re-solve addendum) ==

    /// THE MEASUREMENT ON THE WORLD, pinned verbatim — the two-level bound HOLDS after the
    /// 2026-08-15 shell solve.
    ///
    /// HISTORY (the failing measurement this green pin replaces, kept as provenance). Before the
    /// solve the shell was containment-only (`ring + 2·system_soi = 12_331.398_328_646_887 m`) and
    /// THE world (seed 0) MEASURED FAILING on 2026-08-15: every planet of BOTH ring-placed systems
    /// stayed visible from just outside the galaxy — a 3.954_173_752_999_557_8 m planet visible out
    /// to 302.058_663_384_242_95 m while the shell passed within d_min 161.127_605_697_744_3 …
    /// 280.730_889_197_954 m of the ten ring planets' worst-instant positions (worst_dist
    /// 12_046.713_265_695_933 … 12_166.316_549_196_143 m; the origin system's planets passed). The
    /// owner ruled (addendum to items 5/10): the generator SOLVES the margin as a general
    /// constraint — [`galaxy_shell_r_m`] grows the shell by the worst descendant's two-level
    /// clearance; the ring could not move inward because it already sits at the wake law's LOWER
    /// bound. The exact pre-solve offence stays a live measurement in
    /// `the_guard_refuses_a_shell_that_hugs_its_ring`, which restores the old shell and pins the
    /// first offence verbatim.
    #[test]
    fn the_two_level_bound_re_solved_on_the_world_no_body_is_visible_past_any_two_level_ancestor() {
        let config = UniverseConfig::world(15.0, 0.05);
        // The SOLVED shell, frozen non-self-referentially: ring 12_031.398_328_646_887 m + the
        // two-level clearance (which out-binds the 300 m containment headroom — the interim
        // scale's binding regime).
        assert_eq!(config.scale.galaxy_r_m, FROZEN_GALAXY_SHELL_R_M);
        let pairs = grandchild_visibility_pairs(
            &generate_system_forest(0, &config),
            VISIBILITY_THETA_MIN_RAD,
        );
        // Non-vacuity: every planet is judged against its galaxy AND the universe (15 + 15), every
        // system against the universe (3) — the walk really visited every two-level pair.
        assert_eq!(pairs.len(), 33);
        // The WORST margin across every pair of THE world — a ring system's OUTER planet against
        // the galaxy shell. Positive (the bound holds strictly), larger than the reserved solve
        // margin (the drawn eccentricities sit below the cap the solve bounds against), and pinned
        // EXACTLY so any re-solve of the world flips this loudly and the numbers stay honest
        // in-repo.
        let worst_margin_m = pairs
            .iter()
            .map(|p| p.d_min_m - p.required_m)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(worst_margin_m, FROZEN_TWO_LEVEL_WORST_MARGIN_M);
        assert!(
            worst_margin_m >= VISUAL_SYSTEM_MARGIN_M,
            "the measured margin covers at least the reserved solve margin"
        );
        // …and the check the boot fence runs: no offence anywhere in THE world.
        assert_eq!(
            grandchild_visibility_offences(
                &generate_system_forest(0, &config),
                VISIBILITY_THETA_MIN_RAD,
            ),
            vec![]
        );
        assert_eq!(guard_grandchildren_invisible_outside(0, &config), Ok(()));
    }

    /// The refusal the boot fence makes, measured on the exact PRE-SOLVE geometry: restoring the
    /// containment-only shell (`ring + 2·system_soi` — the world as measured failing 2026-08-15)
    /// makes the guard name the FIRST ring planet with the very numbers of that measurement —
    /// history kept live, and the guard's `Err` arm covered on the boot-facing wrapper.
    #[test]
    fn the_guard_refuses_a_shell_that_hugs_its_ring() {
        let mut config = UniverseConfig::world(15.0, 0.05);
        config.scale.galaxy_r_m =
            config.stellar.system_ring_r_m + 2.0 * config.stellar.system_soi_r_m;
        assert_eq!(
            guard_grandchildren_invisible_outside(0, &config),
            Err(GrandchildVisibleOutside {
                body: RealmId::Planet(2790672799213891506),
                ancestor: GALAXY,
                worst_dist_m: 12_046.713_265_695_933,
                extent_m: 3.954_173_752_999_557_8,
                d_min_m: 280.730_889_197_954,
                required_m: 302.058_663_384_242_95,
            })
        );
    }

    /// The derivation itself, BINDING regime (the interim scale): the worst descendant's two-level
    /// clearance exceeds the containment headroom, so the solved shell is the ring plus the
    /// clearance — the general constraint's terms frozen non-self-referentially at THE world's
    /// numbers (`reach = outer_sma·(1+ecc_cap)`, extent = the planet SOI, θ the ONE visibility
    /// threshold, margin the world's one containment-headroom parameter).
    #[test]
    fn the_shell_solve_binds_on_the_visibility_clearance_at_the_interim_scale() {
        let reach_m = vis_outer_sma() * (1.0 + ECC_SIGMA * ECC_CAP_SIGMAS);
        let clearance_m = two_level_clearance_m(
            reach_m,
            vis_planet_soi(),
            VISIBILITY_THETA_MIN_RAD,
            VISUAL_SYSTEM_MARGIN_M,
        );
        assert_eq!(clearance_m, FROZEN_TWO_LEVEL_CLEARANCE_M);
        assert!(
            clearance_m > 2.0 * VISUAL_SYSTEM_SOI_R_M,
            "the visibility arm binds at the interim scale"
        );
        let cfg = UniverseConfig::visual_scale();
        assert_eq!(
            galaxy_shell_r_m(
                cfg.stellar.system_ring_r_m,
                VISUAL_SYSTEM_SOI_R_M,
                reach_m,
                vis_planet_soi(),
                VISIBILITY_THETA_MIN_RAD,
                VISUAL_SYSTEM_MARGIN_M,
            ),
            FROZEN_GALAXY_SHELL_R_M
        );
        // …and the preset applies EXACTLY this derivation (one solve, no second formula).
        assert_eq!(cfg.scale.galaxy_r_m, FROZEN_GALAXY_SHELL_R_M);
    }

    /// The derivation itself, SLACK regime (owner constraint: the algebra is SCALE-INDEPENDENT —
    /// at near-real scale it must be TRIVIALLY satisfied, not accidentally binding). Near-real
    /// magnitudes: a ~100 AU system SOI (1.5e13 m) dwarfs an Earth-like planet SOI's visibility
    /// reach (9.2e8 m · cot(θ/2) ≈ 7.0e10 m), so the containment arm wins the max and the
    /// two-level bound holds with orders of magnitude to spare.
    #[test]
    fn the_shell_solve_is_slack_at_near_real_scale_containment_binds() {
        let (ring_m, soi_m, reach_m, ext_m) = (3.0e16, 1.5e13, 1.4e13, 9.2e8);
        let clearance_m = two_level_clearance_m(
            reach_m,
            ext_m,
            VISIBILITY_THETA_MIN_RAD,
            VISUAL_SYSTEM_MARGIN_M,
        );
        assert!(
            clearance_m < 2.0 * soi_m,
            "the containment arm binds at near-real scale"
        );
        assert_eq!(
            galaxy_shell_r_m(
                ring_m,
                soi_m,
                reach_m,
                ext_m,
                VISIBILITY_THETA_MIN_RAD,
                VISUAL_SYSTEM_MARGIN_M,
            ),
            ring_m + 2.0 * soi_m
        );
        // The containment-solved shell still clears the general constraint, with room: the worst
        // descendant's visibility range fits far inside the headroom.
        let d_min_m = (ring_m + 2.0 * soi_m) - (ring_m + reach_m) - ext_m;
        assert!(
            d_min_m > ext_m * visibility_factor(VISIBILITY_THETA_MIN_RAD),
            "the two-level bound is trivially satisfied at near-real scale"
        );
    }

    #[test]
    fn worst_hop_excursion_is_the_offset_for_a_static_and_the_apoapsis_for_a_mover() {
        // Static: the authored offset's magnitude, exactly.
        assert_eq!(
            worst_hop_excursion_m(&Placement::StaticOffset(DVec3::new(3.0, 0.0, 4.0))),
            5.0
        );
        // Mover: THE one closed-form worst-instant accessor — never a re-derived `a·(1+e)` beside it.
        let bodies = generate_system_forest(0, &UniverseConfig::world(15.0, 0.05));
        let mover = bodies
            .iter()
            .find(|b| matches!(b.placement, Placement::Orbital(_)))
            .expect("THE world has orbital movers");
        let elements = orbital_of(mover.placement).expect("a mover is Orbital");
        assert_eq!(
            worst_hop_excursion_m(&mover.placement),
            Motion::Kepler(elements).max_excursion_m()
        );
    }

    #[test]
    fn a_visible_grandchild_is_named_with_its_exact_numbers() {
        // A synthetic guilty forest — the arm THE (green) world can never take: root shell 1000 m,
        // a child 100 m off the root's centre, and a 20 m grandchild 50 m off the child's centre.
        // From just outside the root the grandchild can close to 1000 − (100+50) − 20 = 830 m, and
        // the band keeps a 20 m body visible out to 20·cot(θ_min/2) ≈ 1527.8 m — an offence.
        let root = RealmId::System(900);
        let child = RealmId::Planet(901);
        let grand = RealmId::Station(902);
        let bodies = vec![
            GeneratedBody {
                realm: root,
                parent: None,
                shape: Boundary::Shell { r: 1000.0 },
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
            GeneratedBody {
                realm: child,
                parent: Some(root),
                shape: Boundary::Shell { r: 200.0 },
                placement: Placement::StaticOffset(DVec3::new(100.0, 0.0, 0.0)),
                photometrics: None,
            },
            GeneratedBody {
                realm: grand,
                parent: Some(child),
                shape: Boundary::Shell { r: 20.0 },
                placement: Placement::StaticOffset(DVec3::new(0.0, 0.0, 50.0)),
                photometrics: None,
            },
        ];
        let offences = grandchild_visibility_offences(&bodies, VISIBILITY_THETA_MIN_RAD);
        let expected = GrandchildVisibleOutside {
            body: grand,
            ancestor: root,
            worst_dist_m: 150.0,
            extent_m: 20.0,
            d_min_m: 1000.0 - 150.0 - 20.0,
            required_m: 20.0 * visibility_factor(VISIBILITY_THETA_MIN_RAD),
        };
        assert_eq!(offences, vec![expected]);
        // The fail-loud shape: the first offence IS the boot refusal; an empty list is a pass.
        assert_eq!(first_offence(offences), Err(expected));
        assert_eq!(first_offence(Vec::new()), Ok(()));
    }

    #[test]
    fn visual_demand_geometry_is_compressed_real() {
        let c = UniverseConfig::visual_demand(15.0, 0.02);
        // The compressed-real numbers + the DERIVED fields against FROZEN goldens (non-self-referential).
        assert_eq!(c.stellar.system_soi_r_m, VISUAL_SYSTEM_SOI_R_M);
        assert_eq!(c.planet.n_planets, VISUAL_N_PLANETS);
        assert_eq!(c.scale.au_to_render_m, FROZEN_AU_TO_RENDER_M);
        assert_eq!(c.planet.planet_soi_r_m, FROZEN_PLANET_SOI_R_M);
        assert_eq!(c.stellar.central_mass_kg, FROZEN_CENTRAL_MASS_KG);
        assert_eq!(c.scale.galaxy_r_m, FROZEN_GALAXY_SHELL_R_M);
        // The 5 planet orbit distances (semi-major axes, render m): ~15 / 26 / 44 / 75 / 127
        // (the S4 apoapsis-solved compression — the batch review caught this line still carrying
        // the pre-S4 distances beside the re-captured pins).
        let bodies = generate_system_forest(0, &c);
        // ONE star's planets. Orbit distances are a property of a SYSTEM, so a galaxy of several stars
        // must not change them — reading every planet in the galaxy would be reading a different quantity.
        let smas: Vec<f64> = bodies
            .iter()
            .filter(|b| b.parent == Some(SYSTEM_A))
            .map(|b| orbital_of(b.placement).expect("a planet is Orbital").sma)
            .collect();
        assert_eq!(smas, FROZEN_ORBIT_SMA_M.to_vec());
        // The SAME geometry as visual_scale (the static-render twin): one game geometry, two drive modes.
        let vs = UniverseConfig::visual_scale();
        assert_eq!(c.scale.au_to_render_m, vs.scale.au_to_render_m);
        assert_eq!(c.stellar.system_soi_r_m, vs.stellar.system_soi_r_m);
        assert_eq!(c.planet.planet_soi_r_m, vs.planet.planet_soi_r_m);
        assert_eq!(c.stellar.central_mass_kg, vs.stellar.central_mass_kg);
        assert_eq!(c.planet.n_planets, vs.planet.n_planets);
    }

    #[test]
    fn visual_demand_band_is_crossable_between_stars() {
        // NON-VACUITY, the other half of `a_planet_is_visible_from_anywhere_inside_its_own_system`. Together
        // they sandwich the band: a planet is awake everywhere inside its own system (so arriving shows you
        // a populated system), and ASLEEP from the neighbouring star (so the interstellar leg crosses its
        // band and the spin-up machinery is actually exercised). Without this the wider angle could swell
        // until every planet in the galaxy is permanently awake and nothing would ever be measured waking.
        //
        // This USED to read `spin_up < orbit` — the outer planet asleep at its OWN star — which is the very
        // property that made a system look empty on arrival. The band did not stop being crossable; it moved
        // outward, so the crossing is now measured where it belongs, on the way in from another star.
        let cfg = UniverseConfig::visual_demand(15.0, 0.02);
        let (outer, orbit) = outer_planet_orbit(&cfg);
        // The closest a neighbouring star ever gets to this planet: the ring, less its orbit at worst phase.
        let from_the_next_star = cfg.stellar.system_ring_r_m - orbit;
        let reach = outer.aoi.tear_down_r_m();
        assert!(
            reach < from_the_next_star,
            "the outer planet must be asleep from the next star"
        );
    }

    #[test]
    fn equal_visibility_factor_with_zero_v_rel_is_err_not_panic() {
        // THE equal-factor tripwire: under a single visibility factor (spin_up_factor == tear_down_factor)
        // the geometric dead-zone collapses, so a ZERO-relative-velocity band has spin_up == tear_down →
        // Err(InvalidEdges), never a valid inert band (and to_regions' .expect would PANIC at boot on one).
        // The precondition band validity requires occupant_v_max + v_child > 0 — a live occupant. Equality
        // asserted via expect_err (NOT matches!). No epsilon is added to keep the factors exactly equal.
        let factor = visibility_factor(VISIBILITY_THETA_MIN_RAD);
        let err = AoiConfig::for_velocity_safe(
            100.0,
            factor,
            factor,
            0.0,
            AOI_TICK_DT_S,
            VISUAL_AOI_GRACE_TICKS,
            VISUAL_AOI_K_SAFETY_EXTRA,
        )
        .expect_err("equal factors + zero v_rel collapse the dead-zone → InvalidEdges");
        assert_eq!(err, BandError::InvalidEdges);
    }

    #[test]
    fn visual_scale_preset_is_walk_physics_with_derived_visual_geometry() {
        let c = UniverseConfig::visual_scale();
        // Ambient radii (Galaxy/Universe) + render extent + eccentricity physics are REUSED from walk;
        // the System SOI + the four moving-planet fields carry the compressed-real geometry.
        assert_eq!(c.scale.render_extent_m, MAX_RENDERABLE_EXTENT_M);
        assert_eq!(c.stellar.system_soi_r_m, VISUAL_SYSTEM_SOI_R_M);
        // The galaxy is DERIVED to contain the ring of stars, no longer the walk constant: it must hold
        // every system with its reach, or a star sits outside its own galaxy.
        assert!(c.scale.galaxy_r_m > c.stellar.system_ring_r_m + c.stellar.system_soi_r_m);
        // …and it is the SOLVED shell exactly (the 2026-08-15 two-level bound, frozen).
        assert_eq!(c.scale.galaxy_r_m, FROZEN_GALAXY_SHELL_R_M);
        assert_eq!(
            c.planet.ecc_cap,
            ECC_SIGMA * ECC_CAP_SIGMAS,
            "the GEOMETRY cap (4σ), not the solver bound — the compression solves apoapsis at this cap"
        );
        assert_eq!(c.planet.ecc_sigma, ECC_SIGMA);
        assert_eq!(c.planet.incl_sigma, INCL_SIGMA);
        // The overridden fields match the FROZEN compressed-real goldens (non-self-referential — a
        // regression in a derive helper is caught, not silently re-captured). Same goldens as
        // `visual_demand_geometry_is_compressed_real` (visual_scale & visual_demand share visual_geometry).
        assert_eq!(c.scale.au_to_render_m, FROZEN_AU_TO_RENDER_M);
        assert_eq!(c.stellar.central_mass_kg, FROZEN_CENTRAL_MASS_KG);
        assert_eq!(c.planet.planet_soi_r_m, FROZEN_PLANET_SOI_R_M);
        assert_eq!(c.planet.n_planets, VISUAL_N_PLANETS);
    }

    #[test]
    fn generate_system_forest_emits_the_ambient_forest_plus_n_orbital_planets() {
        let bodies = visual_forest();
        // 2 ambient shells + every system + that system's planets. The galaxy's population is DRAWN from
        // its census, so this reads the census rather than restating a number in two places.
        let n_sys = VISUAL_N_SYSTEMS as usize;
        assert_eq!(bodies.len(), 2 + n_sys * (1 + VISUAL_N_PLANETS as usize));
        assert_eq!(orbital_of(bodies[0].placement), None); // Universe
        assert_eq!(orbital_of(bodies[1].placement), None); // Galaxy
        // Every remaining body is either a system (static, under the galaxy) or one of its planets
        // (orbital, under a system) — no third kind, and no planet parented anywhere but a star.
        let systems: Vec<_> = bodies
            .iter()
            .filter(|b| b.parent == Some(GALAXY))
            .map(|b| b.realm)
            .collect();
        assert_eq!(systems.len(), n_sys);
        assert!(
            systems.contains(&SYSTEM_A),
            "system 0 keeps the named identity"
        );
        for b in bodies.iter().skip(2) {
            if systems.contains(&b.realm) {
                assert_eq!(orbital_of(b.placement), None, "a star does not orbit");
            } else {
                assert!(systems.contains(&b.parent.expect("a planet has a star")));
                assert!(orbital_of(b.placement).is_some());
            }
        }
    }

    #[test]
    fn a_galaxy_of_several_systems_gives_each_its_own_seed_place_and_planets() {
        // THE GENERALISATION. The previous shape named ONE system in code and hung the planets off a
        // constant, so a second star could not exist at any scale — which is how the login side and the
        // shard side ended up describing two different worlds. Every system now comes off the same loop.
        let mut cfg = UniverseConfig::visual_scale();
        cfg.galaxy.system_count_lo = 4;
        cfg.galaxy.system_count_hi = 4;
        cfg.stellar.system_ring_r_m = 4.0 * cfg.stellar.system_soi_r_m;
        let bodies = generate_system_forest(0, &cfg);

        let systems: Vec<_> = bodies.iter().filter(|b| b.parent == Some(GALAXY)).collect();
        assert_eq!(systems.len(), 4, "a galaxy is N systems, not one");
        // System 0 keeps the identity every existing fixture and label already names.
        assert_eq!(systems[0].realm, SYSTEM_A);
        // …and no two systems share an id, so their planets can never collide either.
        let ids: std::collections::BTreeSet<_> = systems.iter().map(|s| s.realm).collect();
        assert_eq!(ids.len(), 4, "system identities are distinct: {ids:?}");

        // Each system carries its OWN planets, and a planet belongs to exactly one star.
        for sys in &systems {
            let mine = bodies
                .iter()
                .filter(|b| b.parent == Some(sys.realm))
                .count();
            assert_eq!(
                mine, VISUAL_N_PLANETS as usize,
                "{:?} has its own planets",
                sys.realm
            );
        }
        let planets: std::collections::BTreeSet<_> = bodies
            .iter()
            .filter(|b| b.parent.is_some_and(|p| ids.contains(&p)))
            .map(|b| b.realm)
            .collect();
        assert_eq!(
            planets.len(),
            4 * VISUAL_N_PLANETS as usize,
            "every planet across every system is a distinct realm"
        );

        // A DIFFERENT SEED DRAWS DIFFERENT ORBITS but the SAME structure — the world is a pure function
        // of the seed, so two shards hosting system 3 agree without talking to each other.
        let other = generate_system_forest(99, &cfg);
        assert_eq!(other.len(), bodies.len(), "structure is seed-independent");
        assert_ne!(
            orbital_of(other[3].placement),
            orbital_of(bodies[3].placement),
            "orbits are seed-DERIVED, not fixed"
        );
    }

    #[test]
    fn a_forest_whose_star_systems_overlap_is_refused() {
        // THE FENCE THAT MAKES SEVERAL SYSTEMS SAFE. Authority is "the deepest realm containing you". Two
        // overlapping systems give a position two equally valid owners, and which shard simulates you
        // would come down to iteration order — a coin flip deciding where your input lands.
        let mut cfg = UniverseConfig::visual_scale();
        cfg.galaxy.system_count_lo = 4;
        cfg.galaxy.system_count_hi = 4;

        // A ring TIGHTER than the systems on it: neighbours intersect.
        cfg.stellar.system_ring_r_m = cfg.stellar.system_soi_r_m;
        let overlapping = generate_system_forest(0, &cfg);
        let err = siblings_disjoint(&overlapping).expect_err("touching systems must be refused");
        assert_eq!(
            err.parent, GALAXY,
            "the ambiguity is between children of the galaxy"
        );

        // Spread them and the same forest is accepted — so the refusal is about the GEOMETRY, not about
        // having more than one star.
        cfg.stellar.system_ring_r_m = 4.0 * cfg.stellar.system_soi_r_m;
        assert_eq!(siblings_disjoint(&generate_system_forest(0, &cfg)), Ok(()));

        // And the single-system world every existing rig boots is accepted unchanged.
        assert_eq!(
            siblings_disjoint(&generate_system_forest(0, &UniverseConfig::visual_scale())),
            Ok(())
        );
    }

    #[test]
    fn the_sibling_fence_declines_to_judge_orbits_rather_than_guessing() {
        // AN HONEST LIMIT, pinned so nobody mistakes silence for a guarantee. An orbiting body's region
        // sits at its frame ORIGIN — its position is authored live each tick — so every planet looks
        // co-located to any static comparison. Judging orbits needs their SHELLS compared, which is a
        // separate check over the moving roster; until it exists, orbital overlap is UNCHECKED.
        let cfg = UniverseConfig::visual_scale();
        let bodies = generate_system_forest(0, &cfg);
        let orbiting = bodies
            .iter()
            .filter(|b| orbital_of(b.placement).is_some())
            .count();
        assert_eq!(
            orbiting,
            (VISUAL_N_SYSTEMS * VISUAL_N_PLANETS) as usize,
            "the fixture really does orbit"
        );
        // The planets pass — NOT because they are proven disjoint, but because this fence does not judge
        // orbits at all.
        assert_eq!(siblings_disjoint(&bodies), Ok(()));
    }

    // ---- the render-origin PIN classifier ----

    #[test]
    fn d_fo_7_no_static_region_sits_under_a_varying_ancestor_chain() {
        // D-FO-7 TRIPWIRE: the A4c realm feed ships its movers-only rows and does NOT widen — a STATIC child's
        // authored row rides its frame-local `center`, never a per-tick absolute. That is correct ONLY while no
        // static realm hangs under a MOVING ancestor (which would ride the parent's orbit the center cannot
        // express). Through P4 `generate_system_forest` gives orbiting planets NO children, so the case does not
        // exist. This asserts it: the day P4 first hangs a station under an orbiting planet, THIS fails, and the
        // D-FO-7 decision (parent per-tick rows behind a realm-lane AoI cull, vs the child shard authoring its
        // own box row) must be taken before the widening lands.
        // The invariant, stated as SET EQUALITY (HR5-clean — no branch on the never-true "static-under-varying"
        // condition, whose true arm would be uncoverable): a realm moves-or-inherits-motion IFF the realm is
        // ITSELF a mover. A static realm inheriting a moving ancestor is exactly the case where the two sets
        // diverge. Both filter arms are genuinely exercised — a planet (Orbital ⇒ true) and an ambient body
        // (Static ⇒ false) exist in every forest.
        //
        // This used to ask the question through the seed-derived ORIGIN CHAIN, which is gone: no realm folds
        // its own absolute any more. The question itself survives untouched — it is about the FOREST's shape,
        // not about anybody's absolute — so it is now asked directly of the parent links.
        let moving_anywhere_above = |bodies: &[GeneratedBody], realm: RealmId| -> bool {
            let mut cur = realm;
            for _ in 0..bodies.len() {
                let Some(body) = bodies.iter().find(|b| b.realm == cur) else {
                    return false; // unknown realm — no ancestry to inherit motion from
                };
                if orbital_of(body.placement).is_some() {
                    return true;
                }
                match body.parent {
                    None => return false, // the ambient root — the walk is complete
                    Some(parent) => cur = parent,
                }
            }
            false // hop cap — a cycle, which the boot guard rejects; a safe stop, never a hang
        };
        // The walk's own edge arms, driven where the forest cannot produce them: an UNKNOWN realm has
        // no ancestry to inherit motion from; a CYCLE (which the boot guard rejects at boot) stops at
        // the hop cap instead of hanging — both answer "no motion", never a panic.
        let unknown_only = [GeneratedBody {
            realm: RealmId::System(1),
            parent: None,
            shape: Boundary::Shell { r: 1.0 },
            placement: Placement::StaticOffset(DVec3::ZERO),
            photometrics: None,
        }];
        assert!(
            !moving_anywhere_above(&unknown_only, RealmId::Planet(999)),
            "an unknown realm inherits nothing"
        );
        let cycle = [
            GeneratedBody {
                realm: RealmId::System(1),
                parent: Some(RealmId::System(2)),
                shape: Boundary::Shell { r: 1.0 },
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
            GeneratedBody {
                realm: RealmId::System(2),
                parent: Some(RealmId::System(1)),
                shape: Boundary::Shell { r: 1.0 },
                placement: Placement::StaticOffset(DVec3::ZERO),
                photometrics: None,
            },
        ];
        assert!(
            !moving_anywhere_above(&cycle, RealmId::System(1)),
            "a cycle stops at the hop cap — a safe no, never a hang"
        );
        let config = UniverseConfig::visual_scale();
        for seed in [0u64, 1, 7, 42, 100] {
            let bodies = generate_system_forest(seed, &config);
            let mut varying_chain: Vec<RealmId> = bodies
                .iter()
                .filter(|b| moving_anywhere_above(&bodies, b.realm))
                .map(|b| b.realm)
                .collect();
            let mut movers: Vec<RealmId> = bodies
                .iter()
                .filter(|b| orbital_of(b.placement).is_some())
                .map(|b| b.realm)
                .collect();
            varying_chain.sort();
            movers.sort();
            assert_eq!(
                varying_chain, movers,
                "D-FO-7 (seed {seed}): a realm inherits motion IFF it is itself a mover — a divergence means a \
                 static realm now hangs under a moving ancestor, which the realm feed's movers-only filter \
                 would silently drop. Take the D-FO-7 decision before the widening lands."
            );
        }
    }

    #[test]
    fn realm_regions_for_config_gives_each_moving_planet_a_zero_center() {
        let config = UniverseConfig::visual_scale();
        let bodies = generate_system_forest(0, &config);
        let regions = realm_regions_for_config(0, &config);
        assert_eq!(regions.len(), bodies.len());
        // A moving (Orbital) planet authors its position LIVE through its frame, so its region carries NO
        // baked position — center is the ZERO origin of its own frame. (The crossing-flap fix: a nonzero
        // epoch center would be double-counted against the live frame placement in region_signed_distance.)
        for (body, region) in bodies
            .iter()
            .zip(&regions)
            .filter(|(b, _)| matches!(b.realm, RealmId::Planet(_)))
        {
            assert!(orbital_of(body.placement).is_some(), "a planet is Orbital");
            assert_eq!(region.center.cell(), glam::I64Vec3::ZERO);
            assert_eq!(region.center.offset(), DVec3::ZERO);
        }
    }

    /// FRAME-COHERENT CONTAINMENT (the moving-realm crossing fix): a moving planet's position is authored
    /// ONCE — into the placement book, off its injected `MotionFn` — and its region `center`
    /// is ZERO (the boundary sits at the body's OWN frame origin). So an occupant sitting exactly at the
    /// planet's live orbital position is judged INSIDE its SOI, and an occupant at the star (17.9 m away) is
    /// OUTSIDE — the SAME geometry both the parent shard (planet as a moving child) and the planet's own
    /// shard (planet at the identity) compute, so a crossing cannot flap. Regression guard against the epoch
    /// `center` being double-counted against the frame placement (which put the SOI ~17.9 m off the planet).
    #[test]
    fn a_moving_planet_soi_is_centered_on_its_live_position_not_double_counted() {
        use vd_core::frame::FramePlacement;
        use vd_core::geometry::region_signed_distance;
        use vd_core::kinematics::secs_since_epoch;
        use vd_core::placement::PlacementBook;
        use vd_core::pose::StampedPose;
        let config = UniverseConfig::visual_scale();
        let regions = realm_regions_for_config(0, &config);
        let movers = moving_children_for_config(0, &config, SYSTEM_A);
        let (realm, elements) = movers
            .iter()
            .min_by(|a, b| {
                orbital_state(&a.1, 0.0)
                    .position
                    .length()
                    .total_cmp(&orbital_state(&b.1, 0.0).position.length())
            })
            .cloned()
            .expect("the visual forest has planet movers");
        let region = regions
            .iter()
            .find(|r| r.realm == realm)
            .expect("the mover realm has a region");
        let root = regions
            .iter()
            .find(|r| r.parent.is_none())
            .expect("the forest has a root")
            .frame;
        let tick_hz = 20.0;
        let tick = vd_core::UniverseTick(200);
        // A moving realm carries NO baked position — its center is the origin of its own frame.
        assert_eq!(
            region.center.offset(),
            DVec3::ZERO,
            "a moving planet's region.center must be ZERO (position authored via the frame)",
        );
        // The parent shard's authored book: the planet's row is its live orbital state at this tick —
        // the ONE writer's output, which every consumer (this measurement included) reads as data.
        let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
        let ctx = PlacementBook::new(
            root,
            tick,
            vec![(
                region.frame,
                FramePlacement::moving(state.position, state.velocity),
            )],
        );
        let live = state.position;
        let soi = region.shape.finite_extent();
        // Occupant sitting EXACTLY at the planet's live position (root frame) ⇒ INSIDE the SOI.
        let at_planet = StampedPose::at_rest(root, live, tick);
        let d_at =
            region_signed_distance(&at_planet, region, &ctx).expect("the planet frame resolves");
        // Occupant at the star origin (17.9 m from the planet) ⇒ OUTSIDE.
        let at_star = StampedPose::at_rest(root, DVec3::ZERO, tick);
        let d_star =
            region_signed_distance(&at_star, region, &ctx).expect("the planet frame resolves");
        // An occupant AT the planet's live position is inside its SOI (distance ≈ −SOI). Split asserts
        // (not `a && b`) so neither short-circuit leaves an uncovered branch (HR5).
        assert!(d_at < 0.0);
        assert!((d_at + soi).abs() < 1e-9);
        // An occupant at the star is outside by (orbit − SOI).
        assert!(d_star > 0.0);
        assert!((d_star - (live.length() - soi)).abs() < 1e-6);
    }

    #[test]
    fn moving_children_for_config_lists_every_planet_as_a_mover() {
        let config = UniverseConfig::visual_scale();
        let movers = moving_children_for_config(0, &config, SYSTEM_A);
        assert_eq!(movers.len(), VISUAL_N_PLANETS as usize);
        // Each mover pairs the planet realm with its exact elements (the FIRST non-empty roster —
        // the orbital_of Some-arm + moving_children filter-true in a live path).
        let bodies = generate_system_forest(0, &config);
        for (body, mover) in bodies.iter().skip(3).zip(&movers) {
            assert_eq!(mover.0, body.realm);
            assert_eq!(Some(mover.1), orbital_of(body.placement));
        }
    }

    #[test]
    fn moving_children_for_config_excludes_non_children_and_empty_hosts() {
        let config = UniverseConfig::visual_scale();
        // The Galaxy's only child (System A) is StaticOffset ⇒ no mover (filter-true + orbital_of None).
        assert!(moving_children_for_config(0, &config, GALAXY).is_empty());
        // A realm hosting nothing ⇒ no mover (parent-filter false arm).
        assert!(moving_children_for_config(0, &config, RealmId::System(999)).is_empty());
    }

    #[test]
    fn planet_ecc_is_branchlessly_capped_both_ways() {
        // A HUGE ecc_sigma lets the Rayleigh draw exceed the cap ⇒ `.min` returns the cap exactly.
        let mut hot = UniverseConfig::visual_scale();
        hot.planet.ecc_sigma = 5.0;
        let mut stream = realm_stream(0, &SYSTEM_A_LINEAGE);
        let hot_eccs: Vec<f64> = (0..64)
            .map(|n| planet_elements(&hot, &mut stream, n).ecc)
            .collect();
        assert!(hot_eccs.iter().all(|&e| e <= hot.planet.ecc_cap));
        assert!(
            hot_eccs.contains(&hot.planet.ecc_cap),
            "a large sigma must hit the cap",
        );
        // The real sigma (0.03) draws well below the cap ⇒ `.min` returns the sample.
        let cool = UniverseConfig::visual_scale();
        let mut s2 = realm_stream(0, &SYSTEM_A_LINEAGE);
        for n in 0..VISUAL_N_PLANETS {
            assert!(planet_elements(&cool, &mut s2, n).ecc < cool.planet.ecc_cap);
        }
    }

    #[test]
    fn generate_system_forest_is_deterministic_and_in_domain() {
        // Same seed ⇒ byte-identical forest (the HR1 replay property).
        assert_eq!(
            generate_system_forest(0, &UniverseConfig::visual_scale()),
            generate_system_forest(0, &UniverseConfig::visual_scale()),
        );
        // Every planet's elements are in-domain across seeds (each assert split — no `&&`).
        for seed in [0u64, 1, 42, 999] {
            for body in generate_system_forest(seed, &UniverseConfig::visual_scale())
                .iter()
                .filter(|b| matches!(b.realm, RealmId::Planet(_)))
            {
                let e = orbital_of(body.placement).expect("a planet is Orbital");
                assert!(e.ecc >= 0.0);
                assert!(e.ecc <= KEPLER_ECC_MAX);
                assert!(e.inclination >= 0.0);
                assert!(e.inclination.is_finite());
                assert!(e.raan >= 0.0);
                assert!(e.raan < TAU);
                assert!(e.arg_periapsis < TAU);
                assert!(e.mean_anomaly_epoch < TAU);
            }
        }
    }

    #[test]
    fn generate_system_forest_differs_by_seed() {
        // Genuinely f(seed): different universe seeds yield different orbits/angles.
        assert_ne!(
            generate_system_forest(1, &UniverseConfig::visual_scale()),
            generate_system_forest(2, &UniverseConfig::visual_scale()),
        );
    }

    #[test]
    fn synthetic_central_mass_hits_the_target_outer_period() {
        // The OUTER planet's period is the tuning target (Kepler-3 inversion round-trips).
        let outer = OrbitalElements {
            sma: vis_outer_sma(),
            ecc: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: vis_central_mass(),
        };
        let rel =
            (outer.period() - VISUAL_TARGET_OUTER_PERIOD_S).abs() / VISUAL_TARGET_OUTER_PERIOD_S;
        // No call/expression in the message (it would be an uncovered on-panic-only region, HR5).
        assert!(
            rel < 1e-9,
            "outer period must equal the target within tolerance"
        );
        // Every visual planet's period is seconds-scale — not sub-µs (invisible), not years. Selected by
        // KIND: a fixed offset used to mean "past the ambient shells and the one star", and now lands on
        // another star instead.
        for body in visual_forest()
            .iter()
            .filter(|b| matches!(b.realm, RealmId::Planet(_)))
        {
            let p = orbital_of(body.placement)
                .expect("a planet is Orbital")
                .period();
            assert!(p > 1.0);
            assert!(p <= VISUAL_TARGET_OUTER_PERIOD_S + 1e-6);
        }
    }

    #[test]
    fn visual_geometry_respects_the_far_plane_and_soi_non_overlap() {
        let config = UniverseConfig::visual_scale();
        // Containment order: a system nests strictly inside its galaxy, and a planet inside its system.
        assert!(config.stellar.system_soi_r_m < config.scale.galaxy_r_m);
        // THE GALAXY IS NO LONGER DRAWN, and that is the ruling rather than a regression: a containment
        // boundary is never an object. Once the galaxy has to hold several star systems far enough apart
        // that a neighbour is genuinely ASLEEP until you approach, it necessarily exceeds any box-cull —
        // so "the galaxy is scenery" and "systems wake as you fly to them" cannot both be true. The second
        // is the mechanic; the first was a demo affordance.
        assert!(config.scale.galaxy_r_m > config.scale.render_extent_m);
        assert!(vis_planet_soi() < config.stellar.system_soi_r_m);
        assert!(config.scale.render_extent_m >= config.stellar.system_soi_r_m);
        // The OUTER planet (orbit + SOI) sits STRICTLY inside the System SOI surface (containment).
        assert!(vis_outer_sma() + vis_planet_soi() < config.stellar.system_soi_r_m);
        // NON-OVERLAP: every adjacent orbit gap exceeds two planet SOIs (the smallest gap binds).
        let two_soi = 2.0 * vis_planet_soi();
        for n in 1..VISUAL_N_PLANETS {
            let gap = (orbital_axis_au(n, ORBITAL_A0_AU, ORBITAL_RATIO)
                - orbital_axis_au(n - 1, ORBITAL_A0_AU, ORBITAL_RATIO))
                * config.scale.au_to_render_m;
            assert!(
                gap > two_soi,
                "orbit gap {gap} must exceed two SOIs {two_soi}"
            );
        }
    }

    #[test]
    fn walk_path_is_untouched_and_visual_planet_ids_are_distinct() {
        // Byte-identity: the walk boot path still yields the frozen 7-body forest + empty mover roster
        // (the new generator is uncalled by any walk path).
        assert_eq!(realm_regions_for(0).len(), 7);
        assert!(moving_children_for(0, SYSTEM_A).is_empty());
        // The 5 visual planet ids are mutually distinct and NONE aliases the walk Planet(7) — the
        // child_seed salt/index avalanche keeps them off the roster ids (no silent alias).
        // EVERY planet of EVERY star, not just one star's — the salt/index avalanche must keep them
        // distinct ACROSS systems too, or two stars would quietly claim the same planet realm.
        let ids: Vec<RealmId> = visual_forest()
            .iter()
            .filter(|b| matches!(b.realm, RealmId::Planet(_)))
            .map(|b| b.realm)
            .collect();
        let expect = (VISUAL_N_SYSTEMS * VISUAL_N_PLANETS) as usize;
        assert_eq!(ids.len(), expect);
        let mut distinct = ids.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            expect,
            "every planet id is distinct across every system"
        );
        for id in &ids {
            assert_ne!(*id, PLANET_A);
        }
    }

    #[test]
    fn default_home_realm_walks_root_galaxy_system_and_refuses_a_degenerate_forest() {
        // The login fallback home: the FIRST system under the first galaxy under the root — pure
        // forest-walk, no seed knowledge. A degenerate forest (no root, or no chain below it) is
        // `None`, so a cluster booted on one fails where it can be seen instead of placing every
        // account somewhere arbitrary.
        let world = boot_world_for_tests();
        let home = default_home_realm(world.regions()).expect("THE world has a home system");
        assert!(
            matches!(home, RealmId::System(_)),
            "the fallback home is a star system: {home:?}"
        );
        assert_eq!(default_home_realm(&[]), None, "an empty forest has no home");
        // A root with nothing under it: the galaxy hop refuses.
        let bare_root = [world
            .regions()
            .iter()
            .find(|r| r.parent.is_none())
            .copied()
            .expect("the world has a root")];
        assert_eq!(default_home_realm(&bare_root), None);
    }

    #[test]
    fn the_forest_is_a_valid_single_root_containment_tree() {
        let rs = regions();
        assert_eq!(rs.iter().filter(|r| r.parent.is_none()).count(), 1);
        let mut realms: Vec<RealmId> = rs.iter().map(|r| r.realm).collect();
        realms.sort();
        realms.dedup();
        assert_eq!(realms.len(), rs.len(), "every region has a distinct realm");
        assert_eq!(
            rs.len(),
            7,
            "the 7-region forest (5 shells + Station + Area)"
        );
        // The mandate depths: Universe 0, Galaxy 1, System 2, Planet 3; the sibling system is same-depth.
        assert_eq!(region_depth(&rs, UNIVERSE), 0);
        assert_eq!(region_depth(&rs, GALAXY), 1);
        assert_eq!(region_depth(&rs, SYSTEM_A), 2);
        assert_eq!(region_depth(&rs, PLANET_A), 3);
        assert_eq!(region_depth(&rs, SYSTEM_B), 2);
        // Station A nests directly under System A (depth 3); Area A nests under Planet A (depth 4, the
        // deepest region). The kind-agnostic detector re-homes into either with no station/area code.
        assert_eq!(region_depth(&rs, STATION_A), 3);
        assert_eq!(region_depth(&rs, AREA_A), 4);
        // SIBLING TOPOLOGY LOCKED (task #135 C-6b): System B is a CHILD OF THE GALAXY — a SIBLING of
        // System A reached through the shared parent — NOT a child of System A. This is the canonical forest
        // the LIVE boot uses; the interim `override_regions_for_boundaries` child-of-source model (a
        // playground-only, env-gated fixture) must never drift into it. Depth 2 already implies this, but
        // pin the parent explicitly so a future refactor cannot silently promote the child model.
        assert_eq!(
            rs.iter()
                .find(|r| r.realm == SYSTEM_B)
                .expect("System B present")
                .parent,
            Some(GALAXY),
            "System B is a sibling under the Galaxy, NOT a child of System A",
        );
        // PARENTAGE LOCKED for the first-class Station/Area realms (task #133): a Station nests under its
        // star SYSTEM, an Area under its PLANET (the Area's frame REQUIRES a Planet parent — pin it so a
        // refactor cannot silently re-parent it and break `frame_for_realm`).
        assert_eq!(
            rs.iter()
                .find(|r| r.realm == STATION_A)
                .expect("Station A present")
                .parent,
            Some(SYSTEM_A),
            "Station A nests directly under System A",
        );
        assert_eq!(
            rs.iter()
                .find(|r| r.realm == AREA_A)
                .expect("Area A present")
                .parent,
            Some(PLANET_A),
            "Area A nests under Planet A (its frame provenance)",
        );
    }

    #[test]
    fn region_depth_of_an_unknown_realm_is_zero() {
        assert_eq!(region_depth(&regions(), RealmId::Station(99)), 0);
    }

    #[test]
    fn walk_demand_band_is_crossable_for_every_separated_child() {
        // Every expectation is DERIVED from the generated forest, never hand-typed.
        let regions = walk_demand_regions();
        for r in &regions {
            let ext = r.shape.finite_extent();
            let su = r.aoi.spin_up_r_m();
            // (a) An occupant INSIDE the child (within its own extent) is always in range.
            assert!(
                su > ext,
                "spin-up must exceed the child's own extent (in range inside the child)"
            );
            let Some(parent_id) = r.parent else { continue };
            let parent = regions
                .iter()
                .find(|p| p.realm == parent_id)
                .expect("a region's parent is in the forest");
            let d = r.center.offset().length();
            let td = r.aoi.tear_down_r_m();
            // (c) Releasable: a point inside the parent exists from which the child is out of tear-down
            // range (tear_down < the farthest-in-parent distance = separation + the parent's own extent).
            assert!(
                td < d + parent.shape.finite_extent(),
                "tear-down must release within the parent's reach (child is releasable)"
            );
            if d > 0.0 {
                // (b) SEPARATED child: out of range AT the parent origin ⇒ a walk toward it CROSSES the band.
                assert!(
                    su < d,
                    "a separated child must be out of range at the parent origin (crossable)"
                );
            } else {
                // (d) CO-LOCATED ancestor (offset 0): always in range at the parent origin.
                assert!(
                    su > d,
                    "a co-located child must be in range at the parent origin"
                );
            }
        }
    }

    #[test]
    fn planet_a_is_releasable_at_its_parents_origin() {
        // L3's precondition (spin-DOWN): Planet A releases while the occupant is still AT the parent origin
        // — a STRONGER fact than the uniform (c) criterion (which does NOT hold for the Area: 5.4 vs 5).
        let regions = walk_demand_regions();
        let planet = regions
            .iter()
            .find(|r| matches!(r.realm, RealmId::Planet(_)))
            .expect("planet A in the walk forest");
        let d = planet.center.offset().length();
        assert!(
            planet.aoi.tear_down_r_m() < d,
            "planet A must release at its parent's origin (tear-down < its own offset)"
        );
    }

    #[test]
    fn planet_a_geometric_warmup_precedes_its_boundary() {
        // Gate 1 rests on this: a geometric warm-up margin exists OUTSIDE the child's boundary.
        let regions = walk_demand_regions();
        let planet = regions
            .iter()
            .find(|r| matches!(r.realm, RealmId::Planet(_)))
            .expect("planet A in the walk forest");
        assert!(
            planet.aoi.spin_up_r_m() > planet.shape.finite_extent(),
            "planet A must warm up before its boundary (spin-up radius exceeds its extent)"
        );
    }

    // ===== FA-5 S1: the config-driven VISUAL-scale Orbital generator =====================

    // FROZEN compressed-real geometry goldens — EXACT f64, captured once from the derive helpers at the
    // compressed-real numbers and pinned as literals here (NON-self-referential: a regression in a derive
    // helper is caught, not silently re-captured). Approx: au→render 37.96 / planet SOI 3.95 / orbit
    // semi-major axes 15,26,44,75,127 / synthetic central mass / visibility factor cot(0.75°) ≈ 76.390.
    // RE-CAPTURED at the placement arc S4 (the apoapsis-solved compression + the 0.372 gap fraction) —
    // THE WORLD'S NUMBERS MOVED, deliberately: the outer planet's worst instant now sits exactly at the
    // margin inside its system shell (was: outside it, the S0 tripwire), and a planet's visibility reach
    // still crosses its whole system (302.1 m ≥ 300 m).
    const FROZEN_VISIBILITY_FACTOR: f64 = 76.38983065807547;
    const FROZEN_AU_TO_RENDER_M: f64 = 37.96249762864399;
    const FROZEN_PLANET_SOI_R_M: f64 = 3.9541737529995578;
    const FROZEN_CENTRAL_MASS_KG: f64 = 13407950220013.18;
    const FROZEN_ORBIT_SMA_M: [f64; 5] = [
        15.184999051457595,
        25.814498387477915,
        43.88464725871245,
        74.60390033981116,
        126.82663057767897,
    ];
    // The 2026-08-15 SHELL SOLVE (owner ruling, items 5/10 addendum) — EXACT f64, captured once
    // from the derivation at THE world's numbers and pinned as literals (non-self-referential).
    // The two-level clearance of the worst descendant (the outer planet at the ecc-cap apoapsis,
    // 142.046 m reach + 3.954 m extent × (1 + cot(θ/2)) + the 4 m solve margin ≈ 452.06 m)
    // OUT-BINDS the 300 m containment headroom, so the shell is ring + clearance ≈ 12_483.46 m
    // (was ring + 300 = 12_331.40 m, the 2026-08-15 measured failure). The worst measured margin
    // on THE world (seed 0) is ≈ 11.13 m — the 4 m reserved margin plus the slack of the worst
    // planet's DRAWN eccentricity sitting below the cap the solve bounds against.
    const FROZEN_TWO_LEVEL_CLEARANCE_M: f64 = 452.058663384243;
    const FROZEN_GALAXY_SHELL_R_M: f64 = 12483.45699203113;
    const FROZEN_TWO_LEVEL_WORST_MARGIN_M: f64 = 11.127605697744457;

    // ===== THE WINDOW LANE Slice 0: the per-system photometric draw (the marker datum) =========
    // Owner-approved 2026-08-15/16, docs/design/window_lane.md §2.2/§2.8: a sleeping child's point
    // of light is authored by its parent from the child's OWN generation stream. Slice 0 lands the
    // draw consumer-less (nothing moves); the Slice-A marker emit reads it.

    #[test]
    fn the_worlds_systems_draw_their_pinned_photometrics() {
        // FROZEN per-system draw goldens on THE world (seed 0) — EXACT f64, captured once from the
        // taxonomy chain (sample_imf_mass → classify_spectral → main_sequence_luminosity) at THE
        // world's stellar config and pinned as literals (NON-self-referential: a stream drift, a
        // re-ordered draw, or a retuned IMF is caught, not silently re-captured). All three stars
        // land M-class — the honest Salpeter answer (α = 2.35 concentrates mass draws at the low
        // bound; the u01 that would draw a G star is a ~1e-3 sliver). Sub-solar luma is expected:
        // the marker's DERIVED brightness knob (coordinate-scale model) is a later, separate owe.
        let cfg = UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S);
        let draws = system_photometrics_for_config(0, &cfg);
        assert_eq!(
            draws,
            vec![
                (
                    RealmId::System(7),
                    StarPhotometrics {
                        mass_msun: 0.09287894638451702,
                        class: SpectralClass::M,
                        luma_lsun: 0.0009726074241780799,
                    },
                ),
                (
                    RealmId::System(10487570625701098367),
                    StarPhotometrics {
                        mass_msun: 0.1081418058358058,
                        class: SpectralClass::M,
                        luma_lsun: 0.0013801082634453568,
                    },
                ),
                (
                    RealmId::System(13979593561158050752),
                    StarPhotometrics {
                        mass_msun: 0.16179874709518627,
                        class: SpectralClass::M,
                        luma_lsun: 0.00348634764331354,
                    },
                ),
            ],
        );
        // Provenance: the three pinned realms ARE the seed lineage's systems, in forest order
        // (system 0 keeps the named SYSTEM_A_SEED; the rest avalanche off the galaxy). Plain
        // equality, no destructuring match — a non-System draw fails the vec compare (HR5: no
        // uncoverable panic arm).
        let realms: Vec<RealmId> = draws.iter().map(|(realm, _)| *realm).collect();
        assert_eq!(
            realms,
            vec![
                RealmId::System(system_seed_at(0)),
                RealmId::System(system_seed_at(1)),
                RealmId::System(system_seed_at(2)),
            ]
        );
        assert_eq!(realms[0], RealmId::System(SYSTEM_A_SEED));
        // Coherence: each pinned (class, luma) IS the taxonomy derivation of its pinned mass —
        // the chain cannot silently decouple from the one drawn u01.
        for (_, p) in &draws {
            assert_eq!(
                p.class,
                classify_spectral(p.mass_msun, &SpectralClass::MASS_BOUNDS)
            );
            assert_eq!(
                p.luma_lsun,
                main_sequence_luminosity(p.mass_msun, &cfg.stellar.mlr_segments)
            );
        }
    }

    #[test]
    fn the_marker_luma_bag_frames_the_pinned_draw_through_the_one_shared_codec() {
        // Slice A: the parent's marker datum for a sleeping child rides the ONE window-body codec
        // (`vd_core::look`) — encode every system of THE world, decode through the shared reader,
        // and get back exactly the pinned (class code, luma) pair. A re-framed bag, a transposed
        // field, or a codec fork fails here, not on a live wire.
        let cfg = UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S);
        let draws = system_photometrics_for_config(0, &cfg);
        assert_eq!(draws.len(), 3, "THE world's marker roster is non-vacuous");
        for (_, p) in &draws {
            let bag = marker_luma_bag(p);
            assert_eq!(
                vd_core::look::luma_of(&bag),
                Ok((p.class as u8, p.luma_lsun))
            );
            // A marker bag can never answer for a look (the structural exclusivity, decoded side).
            assert_eq!(
                vd_core::look::look_of(&bag),
                Err(vd_core::tlv::TlvError::MissingRequiredTag(
                    vd_core::look::TAG_LOOK
                ))
            );
        }
    }

    #[test]
    fn the_photometric_draw_is_deterministic_and_dynamics_blind() {
        // Two generations, identical draws (pure f(seed, config) — HR1: every shard hosting the
        // galaxy authors byte-identical markers with no shared state)…
        let cfg = UniverseConfig::world(VISUAL_OCCUPANT_V_MAX_MPS, AOI_TICK_DT_S);
        assert_eq!(
            system_photometrics_for_config(0, &cfg),
            system_photometrics_for_config(0, &cfg)
        );
        // …and blind to the two CLUSTER-dynamics arguments (occupant speed / tick dt): they size
        // the interest band, never the world — the same draw whatever cluster runs it (SL5).
        let other_dynamics = UniverseConfig::world(15.0, 0.02);
        assert_eq!(
            system_photometrics_for_config(0, &cfg),
            system_photometrics_for_config(0, &other_dynamics)
        );
        // A DIFFERENT universe seed draws differently (the stream is real, not a constant): seed 1
        // shares no system seed with seed 0 beyond the named system 0, whose draw must move.
        let seed1 = system_photometrics_for_config(1, &cfg);
        assert_eq!(
            seed1.len(),
            3,
            "the census is config-pinned, not seed-pinned"
        );
        assert_ne!(
            seed1[0].1,
            system_photometrics_for_config(0, &cfg)[0].1,
            "system 0's draw must differ under a different universe seed"
        );
        // The ambient shells and the planets carry NO draw — only systems do (the draw roster is
        // exactly the star roster; the planets' photometric ladder is a later, separate owe).
        // ONE equality over the whole forest (HR5: no matches!/count with uncoverable arms): the
        // bodies carrying a draw are exactly the three systems, in forest order.
        let world = WorldView::generated(0, &cfg);
        let starred: Vec<RealmId> = world
            .bodies
            .iter()
            .filter_map(|b| b.photometrics.map(|_| b.realm))
            .collect();
        assert_eq!(
            starred,
            vec![
                RealmId::System(system_seed_at(0)),
                RealmId::System(system_seed_at(1)),
                RealmId::System(system_seed_at(2)),
            ]
        );
    }
}
