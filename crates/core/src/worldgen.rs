//! Seed-derived realm-region registry — the SINGLE source of truth for realm→region geometry
//! (task #135). Closed-form `f(seed)`: every shard computes the IDENTICAL containment forest at boot from
//! the shared universe seed, so the geometry is REPLICATED BY CONSTRUCTION — no shared mutable state, no
//! inter-shard bytes (HR1). At P3 the bodies are STATIC (identity ephemeris, one universe, WALK-scale);
//! at P4/P5 the per-realm celestial parameters (luminosity/mass/orbital elements) become `f(seed)` and
//! `center` becomes `f(seed, universe_tick)`, ADDITIVELY — the registry SHAPE is frozen (DEFERRED D-44).
//!
//! The P3 forest models the mandate hierarchy **Universe ⊃ Galaxy ⊃ StarSystem ⊃ Planet**, at WALK scale
//! so a player leaves System 7's SOI and is immediately in the GALAXY realm (the between-systems space),
//! then enters System 8's SOI. The star systems are DISJOINT SIBLINGS under the galaxy — a sibling
//! crossing routes through the shared parent (leave a system → the galaxy ancestor → the GALAXY shard
//! detects the entry into the next system), so no shard ever needs a sibling in its local scan
//! ([`realm_neighbourhood_for`]). `RealmId::System`/`Planet` stand in for the levels at P3; dedicated
//! `RealmId::{Universe,Galaxy}` arms + the realistic SOI/AU/ly ephemeris scale land at P4+ (D-44).
//!
//! **Station + Area are FIRST-CLASS realms in this forest** (task #133): a [`RealmId::Station`] BOX nests
//! directly under a star system (a docked/free-floating station volume) and a [`RealmId::Area`] BOX nests
//! under a planet (a city district / spaceport). They are `Aabb` regions (Cartesian volumes, not SOI
//! shells), so the SAME kind-agnostic containment detector re-homes an entity into a Station or an Area
//! with ZERO station/area-specific code (HR3) — the box `signed_distance` feeds the identical
//! `ContainmentBand` the shells use. Planting them here makes the detector LIVE for them (before, the
//! `RealmId::{Station,Area}` arms existed in the taxonomy + frame map but no region carried them, so the
//! detector was inert for those kinds).

use glam::DVec3;
use serde::{Deserialize, Serialize};

use core::f64::consts::TAU;

use crate::celestial::{G, KEPLER_ECC_MAX, OrbitalElements, orbital_state};
use crate::frame::IdentityFrames;
use crate::geometry::{
    AoiConfig, BandError, Boundary, ContainmentBand, RealmRegion, region_depth,
    region_signed_distance,
};
use crate::ids::UniverseTick;
use crate::pose::{FrameRef, LatticePos, RealmId, StampedPose, Tier, frame_for_realm};
use crate::realm_coord::RealmCoord;
use crate::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use crate::rng::{SplitMix64, child_seed, realm_stream};
use crate::taxonomy::{
    FrostThresholds, GalaxyType, SpectralClass, orbital_axis_au, sample_rayleigh,
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
/// Area A's center inside Planet A (Planet A is at +20, r=10). Placed at +25 so the box x∈[22,28] stays
/// within Planet A's sphere yet is OFFSET from the (20,0,0) escape-SOI probe (which must still resolve to
/// Planet 7, not the Area).
const AREA_OFFSET_M: f64 = 25.0;
/// Area A's box half-extent (a small sub-planet district volume).
const AREA_HALF_M: f64 = 3.0;

/// The `RealmId` seed PAYLOADS of the ambient lineage above System A + System A itself — declared as
/// `u64` (the RNG lineage the per-system stream seeds from) and reused as the `RealmId::System(_)`
/// payloads below, so the two never drift. Mirror `realm_path`'s roster (Universe/Galaxy stand-ins 0/1).
const UNIVERSE_SEED: u64 = 0;
const GALAXY_SEED: u64 = 1;
const SYSTEM_A_SEED: u64 = 7;

/// P3 placeholder realm ids for the hierarchy levels that lack a dedicated `RealmId` arm (Universe,
/// Galaxy get one at P4+). The star systems + planet use their real seeds.
const UNIVERSE: RealmId = RealmId::System(UNIVERSE_SEED);
const GALAXY: RealmId = RealmId::System(GALAXY_SEED);
const SYSTEM_A: RealmId = RealmId::System(SYSTEM_A_SEED);
const SYSTEM_B: RealmId = RealmId::System(8);
const PLANET_A: RealmId = RealmId::Planet(7);
/// Station A — a first-class Station realm nested directly under System A (task #133).
const STATION_A: RealmId = RealmId::Station(7);
/// Area A — a first-class sub-planet Area realm nested under Planet A (task #133).
const AREA_A: RealmId = RealmId::Area(7);

/// The largest region extent the CLIENT renders as a box: the Galaxy ([`GALAXY_R_M`] = 180) IS drawn — as
/// the CONTAINING box around the star systems so an entity in the between-space is visibly still inside a
/// realm (never orphaned) — but the ~unbounded Universe ([`UNIVERSE_R_M`] = 1e9) is NOT (it is the ambient
/// fold identity, not a frame). Set between the galaxy (180) and the universe (1e9).
pub const MAX_RENDERABLE_EXTENT_M: f64 = 200.0;

// --- FA-5 (D-45(a)) VISUAL-scale single-system generator — the COMPRESSED-REAL game scale (the ONE
// geometry; `visual_scale` is its static-render expression, `visual_demand` its live demand-cluster
// expression). Every visual geometry number is DERIVED (parameterized helpers below), not a literal. ---
/// `child_seed` salt distinguishing PLANET-kind children under a system (a fixed kind discriminant;
/// `child_seed` avalanches `(parent, salt, index)`, so a distinct salt keeps planet ids off other kinds).
const PLANET_SALT: u64 = 0x504c_414e_4554; // "PLANET"
/// SYSTEM_A's RNG lineage root→leaf `[Universe, Galaxy, System]` — MUST equal
/// `realm_path::system_path(SYSTEM_A_SEED).lineage_seeds()` so every shard hosting System A draws the
/// IDENTICAL per-system stream by construction (HR1); consumed once by [`generate_system_forest`].
const SYSTEM_A_LINEAGE: [u64; 3] = [UNIVERSE_SEED, GALAXY_SEED, SYSTEM_A_SEED];
/// Compressed-real planet count — a 5-planet Kepler system: the inner 3 subtend ≥ the visibility angle
/// from the star (drawn) while the outer 2 fall below it (culled) until an occupant closes in, so the
/// one `cot(θ/2)` rule visibly culls by angular size. Exercises the geometric spacing for n=0..4.
const VISUAL_N_PLANETS: u32 = 5;
/// The compressed-real System SOI radius (render m): System 150 ⊂ Galaxy 180 < cull 200. FLAG: only
/// 20 m of headroom below the cull — no one raises this past ~180 without also moving
/// [`MAX_RENDERABLE_EXTENT_M`] in the same change.
const VISUAL_SYSTEM_SOI_R_M: f64 = 150.0;
/// Headroom (render m) between the OUTER planet's SOI face and the System SOI surface, so the outer
/// body renders STRICTLY inside its System box ([`visual_au_to_render_m`] solves to place it here).
const VISUAL_SYSTEM_MARGIN_M: f64 = 4.0;
/// A planet's SOI radius as a fraction of the SMALLEST inter-orbit gap; `< 0.5` guarantees adjacent
/// SOIs never overlap (the non-overlap invariant is a pinned test, not a hand-tuned coincidence).
const VISUAL_SOI_GAP_FRACTION: f64 = 0.35;
/// The OUTER (slowest) planet's orbital period in seconds — a MAJESTIC-but-visible pace for the human
/// window view (the inner planets are faster by Kepler-3: `T ∝ a^1.5`). Feeds the synthetic central
/// mass via the Kepler-3 inversion. NOT tuned to a frantic few-second orbit: the automated 2-capture
/// render gate samples universe ticks FAR ENOUGH apart to see the sweep, so the period is free to be
/// leisurely for a human watching.
const VISUAL_TARGET_OUTER_PERIOD_S: f64 = 300.0;
/// The minimum angular size (radians) a realm must subtend to enter Area of Interest — 8°, a realm is
/// visible (streams in) out to `extent · cot(θ/2) + velocity-lead`. ONE config constant, no kind-branch.
const VISIBILITY_THETA_MIN_RAD: f64 = 0.139_626;
/// One solar mass (kg) — the canonical/walk INERT central mass (those presets emit no `Orbital` body,
/// so it is never read there; [`UniverseConfig::visual_scale`] overrides it with a synthetic mass).
const CANONICAL_STAR_MASS_KG: f64 = 1.989e30;

/// A body the generator emits before lowering — its realm, parent, shape, and placement. The
/// walk roster uses `StaticOffset` placements (the byte-identity source); real bodies
/// (canonical/seed_derived) use `Orbital`, whose static tick-0 anchor is baked at boot. [`to_regions`]
/// lowers a slice of these to the frozen [`RealmRegion`] forest.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GeneratedBody {
    realm: RealmId,
    parent: Option<RealmId>,
    shape: Boundary,
    placement: Placement,
}

/// Where a body sits in its parent inertial frame.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Placement {
    /// A fixed frame-local offset (the walk roster — the byte-identity source).
    StaticOffset(DVec3),
    /// A Keplerian orbit; its static tick-0 epoch anchor is baked at boot. PLANTED for the
    /// canonical seed-driven generation that goes live at P4 (its bodies need D-41 cells) — no
    /// production producer at P3 (walk uses `StaticOffset` for byte-identity), so the lowering arm
    /// is exercised by tests until then.
    #[allow(dead_code)]
    Orbital(OrbitalElements),
}

/// The region CENTER a body's boundary sits at IN ITS OWN FRAME, as a `cell == ZERO` [`LatticePos`]:
/// - a **moving** (`Orbital`) body authors its position LIVE through its frame
///   ([`LocalFrames::with_moving_child`](crate::frame::LocalFrames::with_moving_child)), so its boundary is
///   at the frame ORIGIN — center **ZERO**, NEVER the epoch. (This is the moving-realm crossing-flap fix: a
///   nonzero epoch center would be DOUBLE-COUNTED against the live frame placement in
///   [`region_signed_distance`](crate::geometry::region_signed_distance) — shifting the SOI ~one orbit off
///   the body, so the parent shard and the body's own shard disagree on containment and a crossing flaps.)
/// - a **`StaticOffset`** body's frame is the identity, so its fixed offset IS the boundary center.
///
/// Walk scale is ALL `StaticOffset` ⇒ unchanged ⇒ byte-identical; only the visual/canonical movers flip to
/// ZERO (whose live pose every other consumer — demand, AoI, feed, render — already reads from the frame).
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
/// keeping the input-side containment seam and the output-side `rebind_pose_to_dest` in agreement.
fn to_regions(bodies: &[GeneratedBody], config: &UniverseConfig) -> Vec<RealmRegion> {
    let band = config
        .band
        .build()
        .expect("containment band edges are valid by construction");
    bodies
        .iter()
        .map(|b| {
            // RLM Step 2: per-realm AoI = factor × the body's OWN finite extent, the dead-zone widened
            // by the occupant speed + THIS child's own orbital closing speed (v_peri; 0 if static).
            let v_child = orbital_of(b.placement).map_or(0.0, |e| e.v_peri());
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
/// from these `OrbitalElements` (`LocalFrames::with_moving_child`), never a static region `center`.
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

/// Direct children of `parent` within `radius` of `occupant_pos`, as `(child RealmCoord, distance)`
/// (RLM Step 2, the C1 spatial-index SEAM). Today a LINEAR fold over the caller's ALREADY-BOUNDED
/// direct-child slice (correct under sparse occupancy — `MAX_REGIONS` caps live direct children); the
/// P6/D-9 spatial index replaces the linear body WITHOUT changing this signature (this is NOT the
/// O(all realms) [`realm_neighbourhood_for`] scan). Positions are FRAME-LOCAL `DVec3` in the parent's
/// OWN frame — occupant and child MUST share a cell through P3 (every pose is cell-ZERO; the cross-cell
/// fold is P4/P5-owed). Yields `RealmCoord` via `parent.child(level)` so a not-yet-spawned child is
/// nameable — NOT the lossy `RealmId`.
pub fn children_within<'a>(
    parent: &'a RealmCoord,
    occupant_pos: DVec3,
    radius: f64,
    direct_children: &'a [(RealmLevel, DVec3)],
) -> impl Iterator<Item = (RealmCoord, f64)> + 'a {
    direct_children.iter().filter_map(move |(level, pos)| {
        aoi_within(*pos, occupant_pos, radius).map(|d| (parent.child(*level), d))
    })
}

/// The monomorphic distance predicate (HR5: the compare lives here; `children_within`'s closure is a
/// branchless map). `Some(d)` iff `d <= radius`.
fn aoi_within(child_pos: DVec3, occupant_pos: DVec3, radius: f64) -> Option<f64> {
    let d = (child_pos - occupant_pos).length();
    (d <= radius).then_some(d)
}

/// A SEED-LINEAGE `RealmId` → its `RealmLevel` (kind + seed), un-lossily: the `System(0)`/`System(1)`
/// stand-ins recover as `Universe`/`Galaxy` (the reverse of `to_realm_id`'s forward map), the keyed
/// kinds pass through. `None` for [`RealmId::Ship`] — a ship is ENTITY-backed (P8, an `EntityId`
/// payload), NOT a seed-lineage realm, so it has no seed `RealmLevel` (and never appears in a seed
/// forest / P3 region). Monomorphic (HR5: the kind match covered once here). A P3 stand-in like
/// `path_for_realm` — a real system with seed 0/1 would alias, but the walk/visual forest uses 7/8.
#[must_use]
pub fn level_of(realm: RealmId) -> Option<RealmLevel> {
    match realm {
        RealmId::System(UNIVERSE_SEED) => {
            Some(RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED))
        }
        RealmId::System(GALAXY_SEED) => Some(RealmLevel::new(RealmKindTag::Galaxy, GALAXY_SEED)),
        RealmId::System(s) => Some(RealmLevel::new(RealmKindTag::System, s)),
        RealmId::Planet(s) => Some(RealmLevel::new(RealmKindTag::Planet, s)),
        RealmId::Station(s) => Some(RealmLevel::new(RealmKindTag::Station, s)),
        RealmId::Area(s) => Some(RealmLevel::new(RealmKindTag::Area, s)),
        RealmId::Ship(_) => None,
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

/// Closed-form inversion of Kepler's third law `T = 2π·√(a³/μ)`, `μ = G·M` → the central mass (kg)
/// that yields orbital period `target_period_s` at semi-major axis `sma_ref_m`. The SYNTHETIC-mass crux
/// for the visual scale: a real star mass at tens-of-metres `sma` gives a sub-µs (invisible) period, so
/// the visual system uses a synthetic mass tuned to a seconds-scale period instead. Straight-line f64.
fn synthetic_central_mass(sma_ref_m: f64, target_period_s: f64) -> f64 {
    TAU * TAU * sma_ref_m.powi(3) / (G * target_period_s * target_period_s)
}

/// The AU→render-metre compression solved so the OUTER planet's orbit + its SOI + `margin` sit EXACTLY
/// at the System SOI surface (containment, vet far-plane fix). Denominator = the outer orbit axis (AU) +
/// the planet SOI expressed in AU (a fraction of the smallest inter-orbit gap). Branchless. Parameterized
/// (RLM realistic-demo Slice 0) so `visual_scale` (static) and `visual_demand` (live) derive the SAME
/// geometry from the same compressed-real numbers.
fn visual_au_to_render_m(
    system_soi: f64,
    margin: f64,
    n: u32,
    gap_fraction: f64,
    a0: f64,
    ratio: f64,
) -> f64 {
    let outer_axis_au = orbital_axis_au(n - 1, a0, ratio);
    let soi_au = gap_fraction * a0 * (ratio - 1.0);
    (system_soi - margin) / (outer_axis_au + soi_au)
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
) -> f64 {
    gap_fraction
        * a0
        * (ratio - 1.0)
        * visual_au_to_render_m(system_soi, margin, n, gap_fraction, a0, ratio)
}

/// The OUTER (slowest) planet's semi-major axis in render metres — the period-tuning reference.
fn visual_outer_sma_render_m(
    system_soi: f64,
    margin: f64,
    n: u32,
    gap_fraction: f64,
    a0: f64,
    ratio: f64,
) -> f64 {
    orbital_axis_au(n - 1, a0, ratio)
        * visual_au_to_render_m(system_soi, margin, n, gap_fraction, a0, ratio)
}

/// The synthetic central mass (kg) placing the OUTER planet's period at `outer_period`.
fn visual_central_mass_kg(
    system_soi: f64,
    margin: f64,
    n: u32,
    gap_fraction: f64,
    a0: f64,
    ratio: f64,
    outer_period: f64,
) -> f64 {
    synthetic_central_mass(
        visual_outer_sma_render_m(system_soi, margin, n, gap_fraction, a0, ratio),
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

/// The config-driven star-system forest: Universe → Galaxy → System A → `config.planet.n_planets`
/// `Orbital` planets (D-45(a) FA-5). The System shell at origin IS the star frame — the planets orbit
/// its center and the star is DATA (`stellar.central_mass_kg`), never a `RealmId::Star` (HR3). The
/// `0..n_planets` range is the ONLY control flow (branchless); `n_planets == 0` (walk/canonical) emits
/// NO planet, so this degenerates to the ambient forest there. Pure `f(seed_universe)`: every planet's
/// elements draw from the ONE per-system [`realm_stream`] in a fixed order (HR1). Makes the
/// `Placement::Orbital` lowering arm LIVE in a real path for the first time (through [`to_regions`]).
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
        },
        // Galaxy: the finite between-systems space, nested in the Universe.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
        },
        // System A: the star's SOI at the origin — the frame the planets orbit (the star is DATA).
        GeneratedBody {
            realm: SYSTEM_A,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: origin,
        },
    ];
    let mut stream = realm_stream(seed_universe, &SYSTEM_A_LINEAGE);
    for n in 0..pl.n_planets {
        bodies.push(GeneratedBody {
            realm: RealmId::Planet(child_seed(SYSTEM_A_SEED, PLANET_SALT, u64::from(n))),
            parent: Some(SYSTEM_A),
            shape: shell(pl.planet_soi_r_m),
            placement: Placement::Orbital(planet_elements(config, &mut stream, n)),
        });
    }
    bodies
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

// ===== RLM Slice 5f-2: the lazy position → full-lineage RealmCoord resolver ==========================

/// Resolve `pos` — a FRAME-LOCAL [`DVec3`] in the Universe root's OWN frame (cell-zero through P3,
/// exactly the pose the containment regions are expressed in; mirrors the test-only `container_at`'s
/// `at(pos)`) — to the FULL-LINEAGE [`RealmCoord`] of the DEEPEST realm whose boundary contains it.
///
/// It DESCENDS from the Universe root: at each level it picks the DEEPEST DIRECT child (a region whose
/// `.parent == Some(current)`) whose boundary contains `pos` (`region_signed_distance <= 0` under
/// [`IdentityFrames`], the P3 no-op reframe), EXTENDS the lineage by that child's [`RealmLevel`], and
/// recurses; it STOPS when no direct child contains `pos`. It ALWAYS returns at least the root coord —
/// the ambient Universe contains all reachable space (the container-fold identity), so an entity is
/// always in ≥ one realm and the result is never empty.
///
/// This returns the WHOLE root→leaf lineage (built by `root.child(level)…`), NOT the lossy [`RealmId`]
/// and NOT [`RealmCoord::lowered`]: the RLM ancestor-closure the orchestrator runs off this needs the
/// UN-collapsed lineage (a `Galaxy`/`Universe` level is LOST through `RealmId`, and a System seed
/// collapses across galaxies). It is the un-lossy `f(seed, config, pos)` twin of the leaf-only
/// [`RealmId`] contract the test-only `container_at` pins.
///
/// HONEST SCOPE (D-41 / P4): this materializes the WHOLE (single-galaxy, walk-scale) containment forest
/// once per call — acceptable at walk / single-galaxy scale. A per-subtree LAZY generator that resolves
/// a 100K-realm / multi-galaxy universe WITHOUT materializing the whole forest is the owed P4 piece;
/// this 5f-2 function nails the LINEAGE shape + the descend structure, and ONLY the generator's laziness
/// is deferred. It resolves against the P3 LIVE containment forest (the walk roster
/// [`generate_walk_forest`]+[`to_regions`], byte-identical to [`realm_regions_for`] at `walk_scale`) —
/// the SAME geometry the sim's `RealmRegions` and the client draw — not the FA-5 visual single-system
/// forest ([`realm_regions_for_config`]); unifying the two under one seed-lazy generator is the P4 owe.
#[must_use]
pub fn container_coord_at(_seed_universe: u64, config: &UniverseConfig, pos: DVec3) -> RealmCoord {
    let regions = to_regions(&generate_walk_forest(config), config);
    let pose = StampedPose::at_rest(
        FrameRef::SystemSpace { system_seed: 0 },
        pos,
        UniverseTick(0),
    );
    // The root is ALWAYS the base of the lineage (an entity is in the Universe even beyond its shell —
    // the container-fold identity). A well-formed forest has exactly one ambient root (`parent: None`).
    let root = regions
        .iter()
        .find(|r| r.parent.is_none())
        .expect("a well-formed forest has exactly one ambient root");
    let mut coord = RealmCoord::from_path(RealmPath::from_levels(vec![
        level_of(root.realm).expect("a seed-forest realm has a RealmLevel"),
    ]))
    .expect("a one-level path always has a leaf");
    let mut current = root.realm;
    // Descend the parent chain: extend into the deepest direct child that contains `pos`, until none.
    while let Some(child) = deepest_containing_child(&regions, current, &pose) {
        coord = coord.child(level_of(child).expect("a seed-forest realm has a RealmLevel"));
        current = child;
    }
    coord
}

/// The full Universe-rooted lineage [`RealmCoord`] of `realm` within a neighbourhood forest — the
/// un-lossy `RealmId`→`RealmCoord` the RLM demand ledger keys on (NOT a `lowered()` single level). This is
/// what lets a source shard, holding only a crossing DEST's `RealmId`, address a `KeepAlive` demand at that
/// dest's WHOLE ancestor chain (`ancestor_close` truncates the coord's parents) so the dest cannot be
/// reaped out from under a player crossing INTO it. `None` only for a [`RealmId::Ship`] (entity-backed, no
/// seed [`RealmLevel`] — a crossing dest is never a ship). Monomorphic (HR5): the ONE realm-KIND decision
/// is [`level_of`]'s already-covered match; this just walks parent pointers root-ward.
///
/// CONTRACT: `realm` MUST be present in `regions` — an unknown realm yields a bogus 1-level coord
/// (`ancestor_realms` returns `[realm]`). Callers satisfy this by only ever passing a dest the container
/// fold produced from these SAME regions (`stub::RealmRegions::coord_of` is the sole caller).
#[must_use]
pub fn coord_of_realm(regions: &[RealmRegion], realm: RealmId) -> Option<RealmCoord> {
    let chain = ancestor_realms(regions, realm); // leaf → root
    let mut levels = Vec::with_capacity(chain.len());
    for r in chain.iter().rev() {
        // root → leaf
        levels.push(level_of(*r)?); // `?` → None only on a ship (never a crossing dest)
    }
    RealmCoord::from_path(RealmPath::from_levels(levels))
}

/// The DEEPEST direct child of `parent` (a region with `.parent == Some(parent)`) whose boundary
/// contains `pose`, or `None` when none does (the descend's stop). In a well-formed containment tree at
/// most one direct child contains a point; [`region_depth`] is a deterministic tiebreak should a
/// malformed forest overlap siblings (never at walk scale). A MONOMORPHIC helper (concrete types) so the
/// containment predicate is covered ONCE here, keeping [`container_coord_at`] a straight-line descend.
fn deepest_containing_child(
    regions: &[RealmRegion],
    parent: RealmId,
    pose: &StampedPose,
) -> Option<RealmId> {
    regions
        .iter()
        .filter(|r| r.parent == Some(parent))
        .filter(|r| region_contains(pose, r))
        .max_by_key(|r| region_depth(regions, r.realm))
        .map(|r| r.realm)
}

/// `pose` is on/inside `region`'s boundary — the P3 instantaneous containment predicate
/// (`signed_distance <= 0`), reframed under [`IdentityFrames`] (a no-op at P3). `.expect` because
/// identity never errors — a real `Err` would be a P4 ephemeris bug and must fail LOUD, never degrade to
/// a wrong containment. Monomorphic (the compare region is covered here, HR5).
fn region_contains(pose: &StampedPose, region: &RealmRegion) -> bool {
    region_signed_distance(pose, region, &IdentityFrames)
        .expect("IdentityFrames never errors at P3")
        <= 0.0
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
        },
        // Galaxy: the finite between-systems space, nested in the Universe.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
        },
        // Star system A: nested in the Galaxy at the origin.
        GeneratedBody {
            realm: SYSTEM_A,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: origin,
        },
        // Planet A: nested in system A, offset from the star.
        GeneratedBody {
            realm: PLANET_A,
            parent: Some(SYSTEM_A),
            shape: shell(pl.planet_soi_r_m),
            placement: at_x(sa.planet_offset_m),
        },
        // Star system B: a DISJOINT sibling of system A under the Galaxy (a walkable galaxy gap between).
        GeneratedBody {
            realm: SYSTEM_B,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: at_x(sa.system_b_offset_m),
        },
        // Station A: a first-class Station BOX under System A (depth 3), on the -X side opposite Planet A.
        GeneratedBody {
            realm: STATION_A,
            parent: Some(SYSTEM_A),
            shape: boxed(sa.station_half_m),
            placement: at_x(sa.station_offset_m),
        },
        // Area A: a first-class sub-planet Area BOX under Planet A (depth 4) — the DEEPEST region.
        GeneratedBody {
            realm: AREA_A,
            parent: Some(PLANET_A),
            shape: boxed(sa.area_half_m),
            placement: at_x(sa.area_offset_m),
        },
    ]
}

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
pub fn realm_neighbourhood_for_held(
    seed_universe: u64,
    held: &std::collections::BTreeSet<RealmId>,
) -> Vec<RealmRegion> {
    realm_neighbourhood_for_held_config(seed_universe, held, &UniverseConfig::walk_scale())
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
    neighbourhood_scope(&realm_regions_for_walk_config(seed_universe, config), held)
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

/// The ancestors-union-direct-children ("never siblings") filter over an ALREADY-BUILT forest — the shared
/// core of both neighbourhood builders (HR3, DRY). A realm is IN-SCOPE iff it is an ancestor of, or a direct
/// child of, ANY held realm. Collect the qualifying realm set first (deduped), then filter the forest ONCE so
/// the output keeps forest order (deterministic boot depth-key/guard results).
fn neighbourhood_scope(
    all: &[RealmRegion],
    held: &std::collections::BTreeSet<RealmId>,
) -> Vec<RealmRegion> {
    let mut scope: std::collections::BTreeSet<RealmId> = std::collections::BTreeSet::new();
    for &hosted in held {
        for a in ancestor_realms(all, hosted) {
            scope.insert(a);
        }
        for r in all.iter().filter(|r| r.parent == Some(hosted)) {
            scope.insert(r.realm);
        }
    }
    all.iter()
        .copied()
        .filter(|r| scope.contains(&r.realm))
        .collect()
}

/// `hosted_realm` + its parent chain up to the ambient root (`parent: None`), as a realm list. Bounded by
/// `all.len()` (a well-formed forest reaches the root well within that). If `hosted_realm` is not in the
/// forest the chain is just `[hosted_realm]` (a shard hosting an unknown realm gets no ancestry → an empty
/// neighbourhood → the detector is inert; a safe degrade).
fn ancestor_realms(all: &[RealmRegion], hosted_realm: RealmId) -> Vec<RealmId> {
    let mut chain = vec![hosted_realm];
    let mut cur = hosted_realm;
    for _ in 0..all.len() {
        match all.iter().find(|r| r.realm == cur).and_then(|r| r.parent) {
            None => break, // reached the root (or an unknown `cur`) — the chain is complete
            Some(p) => {
                chain.push(p);
                cur = p;
            }
        }
    }
    chain
}

// ===== A2: seed-derived origin chain (floating-origin server-authoritative rework) ================
//
// How a Derived realm's shard works out its OWN absolute position: it FOLDS its ancestor chain — re-derived
// FROM SEED, walking UP parent links — and NEVER from its own child list, region center, or live pose (THE
// IRON RULE the reverted attempt broke). Pure additions with no production consumer yet (A3 wires the
// per-tick table), so byte-identical: nothing shipped changes.

/// The PUBLIC lowered projection of a body's [`Placement`] — one link of an origin chain. `Fixed` covers a
/// `StaticOffset`: its frame IS its parent's frame, so it contributes ZERO to the fold (the offset already
/// lives in the region center + the pose value — re-adding it would DOUBLE-COUNT). `Orbital` carries the
/// elements so the fold evaluates the live orbital position + velocity. Exhaustive, no `_`: the P6/P8
/// `Dynamic` arm (a signal-placed realm / a ship — NOT closed-form) is appended THERE, and adding it must be
/// a compile error at every match over this type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OriginLink {
    /// A static body — contributes ZERO to the origin fold (its frame is its parent's).
    Fixed,
    /// An orbiting body — contributes its live orbital position + velocity.
    Orbital(OrbitalElements),
}

/// Lower a [`Placement`] to its public [`OriginLink`] — the monomorphic discriminator (HR5: the match is
/// covered once, here).
fn origin_link_of(placement: Placement) -> OriginLink {
    match placement {
        Placement::StaticOffset(_) => OriginLink::Fixed,
        Placement::Orbital(elements) => OriginLink::Orbital(elements),
    }
}

/// The (position, velocity) one link contributes to the origin fold at `secs`. `Fixed` → ZERO (THE §0.3
/// rule: a static body's frame is its parent's, so NEVER re-add its offset — that offset already lives in
/// the region center); `Orbital` → the live closed-form orbital state. Monomorphic (HR5).
fn origin_link_state(link: OriginLink, secs: f64) -> (DVec3, DVec3) {
    match link {
        OriginLink::Fixed => (DVec3::ZERO, DVec3::ZERO),
        OriginLink::Orbital(elements) => {
            let state = orbital_state(&elements, secs);
            (state.position, state.velocity)
        }
    }
}

/// Fold a root→realm origin chain into the realm's own absolute (position + velocity) in the universe-root
/// frame, at `secs`. Each link adds its parent-relative contribution up the chain: a `Fixed` link adds
/// nothing (its frame is its parent's), an `Orbital` link adds its live position + velocity. These frame
/// placements carry NO rotation, so the fold is a straight positional + velocity SUM (no Coriolis) — the
/// once-per-tick `self_abs` a Derived realm's shard computes LOCALLY from seed, NEVER from its child list
/// (the iron rule). Byte-floor: `compose` does not re-quantize, so through P4 every produced position is
/// `cell == 0` — an all-`Fixed` chain folds to identity, a mover chain to bare `orbital_state`.
#[must_use]
pub fn fold_origin(chain: &[OriginLink], secs: f64) -> (LatticePos, DVec3) {
    let mut pos = LatticePos::local(DVec3::ZERO);
    let mut vel = DVec3::ZERO;
    for &link in chain {
        let (link_pos, link_vel) = origin_link_state(link, secs);
        pos = pos.compose(LatticePos::local(link_pos), Tier::Fine);
        vel += link_vel;
    }
    (pos, vel)
}

/// Does this realm's own absolute VARY over time — i.e. does any ancestor link orbit? A boot constant (the
/// chain is fixed at spin-up): `false` for an all-static chain (a walk shard, a star system — its own frame
/// is fixed), `true` once any ancestor is `Orbital` (a planet shard — its frame rides the orbit).
#[must_use]
pub fn origin_varies(chain: &[OriginLink]) -> bool {
    chain
        .iter()
        .any(|link| matches!(link, OriginLink::Orbital(_)))
}

/// The root→realm origin chain over an explicit body forest: walk PARENT links UPWARD from `realm` to the
/// ambient root, lowering each body's placement, then reverse to root→realm order. Reads ONLY the parent
/// edge of each body it visits — NEVER a realm's child list, region center, or live pose (THE IRON RULE). An
/// unknown `realm` yields an empty chain (a safe degrade — its shard folds to identity). Mirrors
/// [`ancestor_realms`]; bounded by the forest size.
fn origin_chain_over(bodies: &[GeneratedBody], realm: RealmId) -> Vec<OriginLink> {
    let mut chain = Vec::new();
    let mut cur = realm;
    for _ in 0..bodies.len() {
        let Some(body) = bodies.iter().find(|b| b.realm == cur) else {
            return Vec::new(); // unknown realm — no valid ancestry
        };
        chain.push(origin_link_of(body.placement));
        match body.parent {
            None => break, // reached the ambient root — the chain is complete
            Some(parent) => cur = parent,
        }
    }
    chain.reverse();
    chain
}

/// The root→realm origin chain over the config-driven SYSTEM forest — the twin of
/// [`realm_regions_for_config`], built from the SAME `(seed, config)` so the folded absolutes and the regions
/// can never disagree. An orbiting planet's chain ends in an `Orbital` link.
#[must_use]
pub fn origin_chain_for_config(
    seed_universe: u64,
    config: &UniverseConfig,
    realm: RealmId,
) -> Vec<OriginLink> {
    origin_chain_over(&generate_system_forest(seed_universe, config), realm)
}

/// The root→realm origin chain over the WALK forest — the twin of [`realm_regions_for_walk_config`] (the full
/// mandate topology). Every body is `StaticOffset` ⇒ every link `Fixed` ⇒ folds to identity (the byte-floor).
#[must_use]
pub fn origin_chain_for_walk_config(config: &UniverseConfig, realm: RealmId) -> Vec<OriginLink> {
    origin_chain_over(&generate_walk_forest(config), realm)
}

// ===== UniverseConfig (D-45(a) Slice 3b) — the ONE config home ==========================
//
// The ~15 placeholder consts above become NAMED fields of six sub-structs (NOT a god-struct).
// `walk_scale()` reproduces today's EXACT metre-scale geometry (the byte-identity source);
// `canonical()` is the real-scale (AU/ly) tuning — PLANTED but not live as containment regions
// until the D-41 non-zero-cell re-quantization (P4/P5); `seed_derived()` perturbs canonical
// within documented bounds. Nothing consumes this yet — 3c wires the generator onto it.

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
const INCL_SIGMA: f64 = 0.02;
/// Per-system occurrence probability of a station / a sub-planet area district.
const STATION_OCCURRENCE_PROB: f64 = 0.3;
const AREA_OCCURRENCE_PROB: f64 = 0.3;
/// The band by which `seed_derived` jitters the galaxy-type census (peak-to-peak).
const TYPE_MIX_JITTER: f64 = 0.1;

// --- Canonical (real-scale) geometry — planted; live at P4 after D-41 (illustrative values) ---
const CANONICAL_UNIVERSE_R_M: f64 = 8.8e26; // ~observable-universe radius
const CANONICAL_GALAXY_R_M: f64 = 5.0e20; // ~Milky-Way disc radius (~52k ly)
const CANONICAL_RENDER_EXTENT_M: f64 = 1.5e11; // ~1 AU renderable neighbourhood
const CANONICAL_SYSTEM_SOI_R_M: f64 = 1.0e13; // ~system SOI (~65 AU)
const CANONICAL_PLANET_SOI_R_M: f64 = 9.0e8; // ~Earth SOI
const CANONICAL_PLANET_OFFSET_M: f64 = 1.496e11; // ~1 AU
const CANONICAL_SYSTEM_B_OFFSET_M: f64 = 4.0e16; // ~4 ly to the next system
const CANONICAL_STATION_OFFSET_M: f64 = 4.0e8;
const CANONICAL_STATION_HALF_M: f64 = 5.0e3;
const CANONICAL_AREA_OFFSET_M: f64 = 1.0e5;
const CANONICAL_AREA_HALF_M: f64 = 1.0e4;
/// Canonical local galaxy population range (the real galaxy has millions; only a bounded set
/// generates locally — the DENSE ambient scan is the D-45 spatial index).
const CANONICAL_SYSTEM_COUNT_LO: u32 = 1;
const CANONICAL_SYSTEM_COUNT_HI: u32 = 8;
/// The walk forest is exactly two systems (A + its disjoint sibling B).
const WALK_SYSTEM_COUNT: u32 = 2;

/// Ambient-root + galaxy + client-render scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleConfig {
    pub universe_r_m: f64,
    pub galaxy_r_m: f64,
    pub render_extent_m: f64,
    /// AU→render-metre compression (FA-5). VISUAL scale shrinks AU orbits into the render window;
    /// canonical() is the real 1-AU metre factor. Read ONLY on the `Orbital` generator path (inert on
    /// the walk/StaticOffset path), so its value is self-consistent-but-unused on walk_scale().
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
    /// are visible), one solar mass on canonical(). Read ONLY on the `Orbital` path (inert on walk).
    pub central_mass_kg: f64,
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
    /// walk_scale()/canonical() (no planet body ⇒ ambient-only forest, byte-identity); the visual
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
/// (a boot `debug_assert!` cross-checks it, M-2). Named, not inline.
const AOI_TICK_DT_S: f64 = 0.05;
/// Grace ticks a would-be release is held (1 s at the visual 20 Hz).
const VISUAL_AOI_GRACE_TICKS: u32 = 20;
/// Extra velocity-safety margin folded into the dead-zone widening (beyond `K_SAFETY`).
const VISUAL_AOI_K_SAFETY_EXTRA: f64 = 0.5;
/// The visual occupant's max speed (m/s) — MUST equal `StubConfig.move_speed_mps · time_multiplier`
/// (boot `debug_assert!`, M-2), so the anti-thrash pad is measured against the speed the sim integrates.
/// Under the single visibility factor (`spin_up_factor == tear_down_factor`) the geometric dead-zone
/// collapses, so THIS non-zero occupant speed is what keeps `tear_down > spin_up` (band validity requires
/// `occupant_v_max + v_child > 0`; see [`UniverseConfig::visual_demand`]).
const VISUAL_OCCUPANT_V_MAX_MPS: f64 = 2.0;

/// Walk-demand-scale AoI (RLM 5f-4): the LIVE band for the WALK forest, so a WALKING occupant's AoI
/// crosses each separated child's band. Tighter than visual (a walking player over metres, not AU): spin a
/// child up within this multiple of its extent, release past the larger tear-down multiple — HR3
/// proportional, no kind-match. The two DYNAMICS inputs (occupant speed, tick dt) are NOT consts — they are
/// supplied at the composer boot from the LIVE cluster values, which is what closes the M-2 two-home owe (a
/// hardcoded `AOI_TICK_DT_S = 0.05` is wrong at the dev cluster's 50 Hz = 0.02).
const WALK_DEMAND_AOI_SPIN_UP_FACTOR: f64 = 1.2;
const WALK_DEMAND_AOI_TEAR_DOWN_FACTOR: f64 = 1.8;
/// The loiter grace as a DURATION (seconds) — converted to ticks against the live `tick_dt_s` at boot
/// ([`grace_ticks_from_seconds`]), so it is correct at any tick rate. The SINGLE loiter-duration constant in
/// the codebase: the region AoI `grace_ticks` derives from it here, and (VU AoI S2b) the parent's retained-
/// occupant TTL reuses it, so the two are consistent by construction and neither is ever a magic number.
pub const WALK_DEMAND_AOI_GRACE_S: f64 = 1.0;
const WALK_DEMAND_AOI_K_SAFETY_EXTRA: f64 = 0.5;

/// The 1-tick floor for the loiter grace when the tick dt is degenerate — a tripwire (the composer
/// cross-checks the tick pair before this is reached; a non-finite/non-positive dt here is a mis-wired
/// boot). Named, not inline.
const GRACE_TICKS_FLOOR: u32 = 1;

/// Convert a loiter grace measured in SECONDS to ticks against the live `dt_s` (RLM 5f-4 — a duration is
/// correct at ANY tick rate). Monomorphic + saturating: a degenerate `dt_s` (≤0 / non-finite) OR a
/// degenerate quotient (non-finite / below one tick) yields [`GRACE_TICKS_FLOOR`]; an absurd quotient
/// saturates at `u32::MAX`; otherwise the in-range `round()` is a lossless `u32`. `pub` so the VU AoI S2b
/// retained-occupant TTL derives from the SAME converter + loiter constant the region grace uses (DRY).
pub fn grace_ticks_from_seconds(secs: f64, dt_s: f64) -> u32 {
    if !(dt_s > 0.0 && dt_s.is_finite()) {
        return GRACE_TICKS_FLOOR;
    }
    let ticks = (secs / dt_s).round();
    if !(ticks.is_finite() && ticks >= 1.0) {
        GRACE_TICKS_FLOOR
    } else if ticks >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        ticks as u32
    }
}

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
            },
            planet: PlanetConfig {
                planet_soi_r_m: PLANET_SOI_R_M,
                orbital_a0_au: ORBITAL_A0_AU,
                orbital_ratio: ORBITAL_RATIO,
                ecc_sigma: ECC_SIGMA,
                incl_sigma: INCL_SIGMA,
                ecc_cap: KEPLER_ECC_MAX,
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

    /// The canonical real-scale (AU/ly) preset — PLANTED; its bodies go LIVE as containment
    /// regions only after the D-41 cross-cell re-quantization (P4/P5). Physics is shared with
    /// `walk_scale`; only the metre-scale geometry differs.
    #[must_use]
    pub fn canonical() -> UniverseConfig {
        UniverseConfig {
            scale: ScaleConfig {
                universe_r_m: CANONICAL_UNIVERSE_R_M,
                galaxy_r_m: CANONICAL_GALAXY_R_M,
                render_extent_m: CANONICAL_RENDER_EXTENT_M,
                // Real 1-AU metres: the canonical planet offset (1.496e11 m) IS 1 AU = orbital_a0 * this.
                au_to_render_m: CANONICAL_PLANET_OFFSET_M / ORBITAL_A0_AU,
            },
            galaxy: GalaxyConfig {
                type_cumulative: GalaxyType::CANONICAL_CUMULATIVE,
                system_count_lo: CANONICAL_SYSTEM_COUNT_LO,
                system_count_hi: CANONICAL_SYSTEM_COUNT_HI,
            },
            stellar: StellarConfig {
                system_soi_r_m: CANONICAL_SYSTEM_SOI_R_M,
                imf_slope: IMF_SLOPE,
                mass_lo_msun: IMF_MASS_LO_MSUN,
                mass_hi_msun: IMF_MASS_HI_MSUN,
                mlr_segments: SpectralClass::MLR_SEGMENTS,
                central_mass_kg: CANONICAL_STAR_MASS_KG, // one solar mass (real).
            },
            planet: PlanetConfig {
                planet_soi_r_m: CANONICAL_PLANET_SOI_R_M,
                orbital_a0_au: ORBITAL_A0_AU,
                orbital_ratio: ORBITAL_RATIO,
                ecc_sigma: ECC_SIGMA,
                incl_sigma: INCL_SIGMA,
                ecc_cap: KEPLER_ECC_MAX,
                frost_coeff_au: crate::taxonomy::FROST_COEFF_AU,
                m_ocean_lo_mearth: FrostThresholds::CANONICAL.m_ocean_lo_mearth,
                m_gas_mearth: FrostThresholds::CANONICAL.m_gas_mearth,
                m_core_crit_mearth: FrostThresholds::CANONICAL.m_core_crit_mearth,
                n_planets: 0, // ambient-only forest (no Orbital body) — visual_scale() sets N.
            },
            satellite: SatelliteConfig {
                station_prob: STATION_OCCURRENCE_PROB,
                area_prob: AREA_OCCURRENCE_PROB,
                station_half_m: CANONICAL_STATION_HALF_M,
                area_half_m: CANONICAL_AREA_HALF_M,
                station_offset_m: CANONICAL_STATION_OFFSET_M,
                area_offset_m: CANONICAL_AREA_OFFSET_M,
                planet_offset_m: CANONICAL_PLANET_OFFSET_M,
                system_b_offset_m: CANONICAL_SYSTEM_B_OFFSET_M,
            },
            band: BandConfig {
                inset_m: CONTAINMENT_INSET_M,
                outset_m: CONTAINMENT_OUTSET_M,
                k_safety_extra: 0.0,
            },
            interest: InterestConfig::inert(), // canonical: AoI OFF until D-41 (byte-identity).
        }
    }

    /// The COMPRESSED-REAL visual geometry on a `walk_scale()` clone (system SOI 150, 5 Kepler planets,
    /// outer period 300 s) WITHOUT an interest band — the ONE game geometry that `visual_scale` (static
    /// render) and `visual_demand` (live demand cluster) both drive, so the two can NEVER disagree on
    /// geometry (they call the SAME parameterized derive helpers with the SAME compressed-real numbers).
    /// Mutates a `walk_scale()` CLONE (byte-identity: walk/canonical never call these helpers), overriding
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
        );
        cfg.stellar.central_mass_kg = visual_central_mass_kg(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
            VISUAL_TARGET_OUTER_PERIOD_S,
        );
        cfg.planet.planet_soi_r_m = visual_planet_soi_r_m(
            VISUAL_SYSTEM_SOI_R_M,
            VISUAL_SYSTEM_MARGIN_M,
            VISUAL_N_PLANETS,
            VISUAL_SOI_GAP_FRACTION,
            ORBITAL_A0_AU,
            ORBITAL_RATIO,
        );
        cfg.planet.n_planets = VISUAL_N_PLANETS;
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

    /// Perturb `canonical()` deterministically within documented bounds, so different seeds yield
    /// different galaxy morphologies. Physics + geometry stay canonical; only the galaxy-type
    /// census is jittered (kept ordered + within `[0.1, 0.99]`). `ecc_cap` stays `KEPLER_ECC_MAX`.
    #[must_use]
    pub fn seed_derived(seed: u64) -> UniverseConfig {
        let mut rng = SplitMix64::new(seed);
        let mut cfg = UniverseConfig::canonical();
        let jitter = (rng.next_f64() - 0.5) * TYPE_MIX_JITTER;
        let c_spiral = (cfg.galaxy.type_cumulative[0] + jitter).clamp(0.1, 0.85);
        cfg.galaxy.type_cumulative = [c_spiral, (c_spiral + 0.18).min(0.99)];
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::IdentityFrames;
    use crate::geometry::{DepthKey, container, region_depth, region_signed_distance};
    use crate::ids::UniverseTick;
    use crate::pose::{FrameRef, StampedPose};
    use glam::DQuat;

    fn regions() -> Vec<RealmRegion> {
        realm_regions_for(0)
    }

    /// A rest pose at `pos` in the Universe root frame (identity placements at P3 make the frame moot).
    fn at(pos: DVec3) -> StampedPose {
        StampedPose {
            frame: FrameRef::SystemSpace { system_seed: 0 },
            pos: LatticePos::local(pos),
            vel: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            universe_tick: UniverseTick(0),
        }
    }

    /// The instantaneous (hysteresis-free) container realm of `pos`: the members are the regions whose
    /// surface the point is on/inside (`signed_distance <= 0`), folded from the Universe root — the
    /// GEOMETRIC CONTRACT the sim's stateful band membership (C-3) layers hysteresis on top of.
    fn container_at(pos: DVec3) -> RealmId {
        let rs = regions();
        let members: Vec<DepthKey> = rs
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                region_signed_distance(&at(pos), r, &IdentityFrames).expect("identity never errors")
                    <= 0.0
            })
            .map(|(ix, r)| (region_depth(&rs, r.realm), r.realm, ix))
            .collect();
        container(UNIVERSE, &members)
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
    fn containment_resolves_the_full_walk_scale_mandate_chain() {
        // Inside planet A's SOI → the PLANET (deepest container).
        assert_eq!(
            container_at(DVec3::new(PLANET_A_OFFSET_M, 0.0, 0.0)),
            PLANET_A
        );
        // Inside system A but outside planet A (the origin — the star) → the STAR SYSTEM.
        assert_eq!(container_at(DVec3::ZERO), SYSTEM_A);
        // Inside the Station BOX (at (-25,0,0), half=5) → STATION_A: the box is DEEPER (depth 3) than
        // System 7 (depth 2), so the container fold picks the Station — the first-class box realm wins.
        assert_eq!(
            container_at(DVec3::new(STATION_A_OFFSET_M, 0.0, 0.0)),
            STATION_A
        );
        // Inside the Area BOX (at (25,0,0), half=3) → AREA_A: the box (depth 4) is deeper than Planet 7
        // (depth 3) which contains it, so the Area wins — the DEEPEST realm in the whole forest.
        assert_eq!(container_at(DVec3::new(AREA_OFFSET_M, 0.0, 0.0)), AREA_A);
        // In the walkable GAP between the two systems (x=50: outside A's r=40 and B at +100) → the GALAXY
        // (escape the system → immediately the galaxy — the mandate).
        assert_eq!(container_at(DVec3::new(50.0, 0.0, 0.0)), GALAXY);
        // Inside sibling system B → SYSTEM_B (the other side of the round trip).
        assert_eq!(
            container_at(DVec3::new(SYSTEM_B_OFFSET_M, 0.0, 0.0)),
            SYSTEM_B
        );
        // Beyond the galaxy but within the universe → the UNIVERSE root.
        assert_eq!(container_at(DVec3::new(5_000.0, 0.0, 0.0)), UNIVERSE);
        // Beyond EVERYTHING (outside the universe shell) → STILL the Universe, by fold identity.
        assert_eq!(container_at(DVec3::new(1.0e15, 0.0, 0.0)), UNIVERSE);
    }

    /// The FULL-lineage container coord of `pos` under the P3 walk forest (`seed 0`, `walk_scale`) —
    /// the un-lossy `container_coord_at` twin of `container_at`.
    fn coord_at(pos: DVec3) -> RealmCoord {
        container_coord_at(0, &UniverseConfig::walk_scale(), pos)
    }

    /// An expected lineage from `(kind, seed)` pairs, root → leaf.
    fn lineage(levels: &[(RealmKindTag, u64)]) -> RealmPath {
        RealmPath::from_levels(levels.iter().map(|&(k, s)| RealmLevel::new(k, s)).collect())
    }

    #[test]
    fn container_coord_resolves_the_full_lineage_for_the_walk_mandate_chain() {
        use RealmKindTag::{Area, Galaxy, Planet, Station, System, Universe};
        // The WHOLE POINT of this function vs `container_at`: it returns the FULL root→leaf lineage, so
        // every case asserts `.path()` level-for-level (not just the lossy leaf `.lowered()`).

        // Inside Planet A's SOI (20,0,0) → the full [Universe, Galaxy, System A, Planet A] lineage, which
        // LOWERS to the leaf Planet A (lineage + lowered agree).
        let planet = coord_at(DVec3::new(PLANET_A_OFFSET_M, 0.0, 0.0));
        assert_eq!(
            planet.path(),
            &lineage(&[(Universe, 0), (Galaxy, 1), (System, 7), (Planet, 7)])
        );
        assert_eq!(planet.lowered(), PLANET_A);

        // The origin (the star) → [Universe, Galaxy, System A] (inside System A, outside Planet A).
        let origin = coord_at(DVec3::ZERO);
        assert_eq!(
            origin.path(),
            &lineage(&[(Universe, 0), (Galaxy, 1), (System, 7)])
        );
        assert_eq!(origin.lowered(), SYSTEM_A);

        // The Station BOX (-25,0,0) → the Station lineage (depth 3 directly under System A).
        let station = coord_at(DVec3::new(STATION_A_OFFSET_M, 0.0, 0.0));
        assert_eq!(
            station.path(),
            &lineage(&[(Universe, 0), (Galaxy, 1), (System, 7), (Station, 7)])
        );
        assert_eq!(station.lowered(), STATION_A);

        // The Area BOX (25,0,0) → the DEEPEST 5-level lineage [Universe, Galaxy, System A, Planet A,
        // Area A] (the descend recurses past Planet A into its Area child).
        let area = coord_at(DVec3::new(AREA_OFFSET_M, 0.0, 0.0));
        assert_eq!(
            area.path(),
            &lineage(&[
                (Universe, 0),
                (Galaxy, 1),
                (System, 7),
                (Planet, 7),
                (Area, 7)
            ])
        );
        assert_eq!(area.lowered(), AREA_A);

        // Inside the SIBLING System B (130,0,0) → its own [Universe, Galaxy, System B] lineage (System 8,
        // NOT System 7 — the lineage disambiguates the two same-depth systems).
        let system_b = coord_at(DVec3::new(SYSTEM_B_OFFSET_M, 0.0, 0.0));
        assert_eq!(
            system_b.path(),
            &lineage(&[(Universe, 0), (Galaxy, 1), (System, 8)])
        );
        assert_eq!(system_b.lowered(), SYSTEM_B);

        // Outside everything but inside the Universe (5000,0,0: beyond the Galaxy r=180, inside the
        // Universe r=1e9) → JUST the root [Universe] (no direct child of the root contains it).
        let root = coord_at(DVec3::new(5_000.0, 0.0, 0.0));
        assert_eq!(root.path(), &lineage(&[(Universe, 0)]));
        assert_eq!(root.lowered(), UNIVERSE);

        // Beyond the Universe shell too (1e15) → STILL just [Universe] by the container-fold identity
        // (the root is unconditional — its own boundary is never tested).
        let beyond = coord_at(DVec3::new(1.0e15, 0.0, 0.0));
        assert_eq!(beyond.path(), &lineage(&[(Universe, 0)]));
        assert_eq!(beyond.lowered(), UNIVERSE);
    }

    #[test]
    fn container_coord_at_is_a_deterministic_pure_function() {
        // f(seed, config, pos): byte-identical across calls (no `Date::now`/rng), so the orchestrator's
        // ancestor-closure is reproducible cross-host (HR1). Two independent resolves of the deepest
        // point (Area A) are equal, path and all.
        let a = coord_at(DVec3::new(AREA_OFFSET_M, 0.0, 0.0));
        let b = container_coord_at(
            0,
            &UniverseConfig::walk_scale(),
            DVec3::new(AREA_OFFSET_M, 0.0, 0.0),
        );
        assert_eq!(a, b);
        // The lineage length is exactly depth + 1 (root at index 0, leaf last) — the descend appended a
        // level per hop, never truncating (the "full lineage, not lowered()" contract).
        assert_eq!(a.path().levels().len(), 5);
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
        let ship = RealmId::Ship(crate::ids::EntityId::pack(
            crate::entity_kind::EntityKind::Ship,
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
        let sd = region_signed_distance(&at(DVec3::ZERO), system_a, &IdentityFrames).expect("ok");
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
    fn region_depth_of_an_unknown_realm_is_zero() {
        assert_eq!(region_depth(&regions(), RealmId::Station(99)), 0);
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
        assert_eq!(c.scale.galaxy_r_m, GALAXY_R_M);
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
        for c in [
            UniverseConfig::walk_scale(),
            UniverseConfig::visual_scale(),
            UniverseConfig::canonical(),
            UniverseConfig::seed_derived(3),
            UniverseConfig::seed_derived(999),
        ] {
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
        assert!(UniverseConfig::canonical().planet.ecc_cap <= KEPLER_ECC_MAX);
        for seed in [0u64, 1, 42, 999] {
            assert!(UniverseConfig::seed_derived(seed).planet.ecc_cap <= KEPLER_ECC_MAX);
        }
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

    #[test]
    fn seed_derived_is_deterministic_and_bounded() {
        assert_eq!(
            UniverseConfig::seed_derived(7),
            UniverseConfig::seed_derived(7)
        );
        assert_ne!(
            UniverseConfig::seed_derived(1).galaxy.type_cumulative,
            UniverseConfig::seed_derived(2).galaxy.type_cumulative,
        );
        for seed in [0u64, 1, 5, 100, 9999] {
            let c = UniverseConfig::seed_derived(seed).galaxy.type_cumulative;
            assert!(c[0] >= 0.1, "spiral cumulative >= floor: {c:?}");
            assert!(c[0] <= 0.85, "spiral cumulative <= ceiling: {c:?}");
            assert!(c[1] > c[0], "type cumulative ordered: {c:?}");
            assert!(c[1] <= 0.99, "second cumulative capped: {c:?}");
        }
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
                DVec3::new(25.0, 0.0, 0.0),
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
        // ZERO origin of its own frame; its live pose is authored through `LocalFrames::with_moving_child`.
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
        };
        let regions = to_regions(&[body], &UniverseConfig::canonical());
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
        };
        let static_child = GeneratedBody {
            realm: RealmId::Station(2),
            parent: Some(RealmId::System(7)),
            shape: Boundary::Shell { r: 1.0e6 },
            placement: Placement::StaticOffset(DVec3::new(5.0, 0.0, 0.0)),
        };
        let orbital_non_child = GeneratedBody {
            realm: RealmId::Planet(3),
            parent: Some(RealmId::System(99)),
            shape: Boundary::Shell { r: 9.0e8 },
            placement: Placement::Orbital(elements),
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

    #[test]
    fn walk_and_canonical_keep_aoi_inert_while_visual_stays_live() {
        // Byte-identity: walk + canonical keep AoI OFF (behaviour unchanged); the compressed-real visual
        // band is LIVE under the ONE cot(θ/2) visibility factor.
        for r in realm_regions_for(0) {
            assert_eq!(r.aoi, AoiConfig::inert(), "walk regions stay AoI-inert");
        }
        for r in realm_regions_for_walk_config(0, &UniverseConfig::canonical()) {
            assert_eq!(
                r.aoi,
                AoiConfig::inert(),
                "canonical regions stay AoI-inert"
            );
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
        for held in &sets {
            assert_eq!(
                realm_neighbourhood_for_held(0, held),
                realm_neighbourhood_for_held_config(0, held, &UniverseConfig::walk_scale()),
                "realm_neighbourhood_for_held delegates byte-identically"
            );
        }
    }

    #[test]
    fn grace_ticks_from_seconds_converts_saturates_and_floors() {
        assert_eq!(grace_ticks_from_seconds(1.0, 0.02), 50, "1 s at 50 Hz");
        assert_eq!(grace_ticks_from_seconds(1.0, 0.05), 20, "1 s at 20 Hz");
        assert_eq!(
            grace_ticks_from_seconds(1.0, 0.0),
            GRACE_TICKS_FLOOR,
            "dt 0 ⇒ floor"
        );
        assert_eq!(
            grace_ticks_from_seconds(1.0, -0.02),
            GRACE_TICKS_FLOOR,
            "dt < 0 ⇒ floor"
        );
        assert_eq!(
            grace_ticks_from_seconds(1.0, f64::NAN),
            GRACE_TICKS_FLOOR,
            "dt NaN ⇒ floor"
        );
        assert_eq!(
            grace_ticks_from_seconds(1.0, f64::INFINITY),
            GRACE_TICKS_FLOOR,
            "dt +inf ⇒ floor"
        );
        assert_eq!(
            grace_ticks_from_seconds(0.0, 0.02),
            GRACE_TICKS_FLOOR,
            "0 ticks ⇒ floor"
        );
        assert_eq!(
            grace_ticks_from_seconds(f64::INFINITY, 0.02),
            GRACE_TICKS_FLOOR,
            "non-finite quotient ⇒ floor"
        );
        assert_eq!(
            grace_ticks_from_seconds(1e20, 0.02),
            u32::MAX,
            "absurd quotient ⇒ saturate"
        );
    }

    #[test]
    fn aoi_within_boundary() {
        let occ = DVec3::ZERO;
        assert_eq!(aoi_within(DVec3::ZERO, occ, 10.0), Some(0.0));
        assert_eq!(
            aoi_within(DVec3::new(10.0, 0.0, 0.0), occ, 10.0),
            Some(10.0)
        );
        assert_eq!(aoi_within(DVec3::new(11.0, 0.0, 0.0), occ, 10.0), None);
    }

    #[test]
    fn children_within_filters_by_radius() {
        let parent = RealmCoord::from_path(crate::realm_path::RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::Universe, 0),
            RealmLevel::new(RealmKindTag::Galaxy, 1),
            RealmLevel::new(RealmKindTag::System, 7),
        ]))
        .expect("parent coord");
        let children = [
            (
                RealmLevel::new(RealmKindTag::Planet, 10),
                DVec3::new(5.0, 0.0, 0.0),
            ),
            (
                RealmLevel::new(RealmKindTag::Planet, 20),
                DVec3::new(50.0, 0.0, 0.0),
            ),
        ];
        let got: Vec<RealmCoord> = children_within(&parent, DVec3::ZERO, 10.0, &children)
            .map(|(c, _)| c)
            .collect();
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0],
            parent.child(RealmLevel::new(RealmKindTag::Planet, 10))
        );
    }

    #[test]
    fn direct_child_levels_from_seed() {
        // Visual System A hosts N orbiting planets; the roster is exactly those planet levels.
        let config = UniverseConfig::visual_scale();
        let levels = direct_child_levels(0, &config, SYSTEM_A);
        assert_eq!(levels.len(), VISUAL_N_PLANETS as usize);
        assert!(levels.iter().all(|l| l.kind == RealmKindTag::Planet));
    }

    #[test]
    fn level_of_covers_every_kind() {
        assert_eq!(
            level_of(RealmId::System(0)),
            Some(RealmLevel::new(RealmKindTag::Universe, 0))
        );
        assert_eq!(
            level_of(RealmId::System(1)),
            Some(RealmLevel::new(RealmKindTag::Galaxy, 1))
        );
        assert_eq!(
            level_of(RealmId::System(7)),
            Some(RealmLevel::new(RealmKindTag::System, 7))
        );
        assert_eq!(
            level_of(RealmId::Planet(7)),
            Some(RealmLevel::new(RealmKindTag::Planet, 7))
        );
        assert_eq!(
            level_of(RealmId::Station(7)),
            Some(RealmLevel::new(RealmKindTag::Station, 7))
        );
        assert_eq!(
            level_of(RealmId::Area(7)),
            Some(RealmLevel::new(RealmKindTag::Area, 7))
        );
        // A ship is entity-backed (P8), not seed-lineage ⇒ None (the filter_map-dropped case).
        let ship = RealmId::Ship(crate::EntityId::pack(
            crate::entity_kind::EntityKind::Player,
            1,
            1,
            1,
        ));
        assert_eq!(level_of(ship), None);
    }

    // ===== FA-5 S1: the config-driven VISUAL-scale Orbital generator =====================

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
            VISUAL_TARGET_OUTER_PERIOD_S,
        )
    }

    // ===== RLM realistic-demo Slice 0: the one visibility constant + compressed-real geometry =========

    // FROZEN compressed-real geometry goldens — EXACT f64, captured once from the derive helpers at the
    // compressed-real numbers and pinned as literals here (NON-self-referential: a regression in a derive
    // helper is caught, not silently re-captured). Approx: au→render 42.45 / planet SOI 4.16 / orbit
    // semi-major axes 17,29,49,83,142 / synthetic central mass / visibility factor cot(4°) ≈ 14.301.
    const FROZEN_VISIBILITY_FACTOR: f64 = 14.300701209730468;
    const FROZEN_AU_TO_RENDER_M: f64 = 42.456177082969845;
    const FROZEN_PLANET_SOI_R_M: f64 = 4.160705354131045;
    const FROZEN_CENTRAL_MASS_KG: f64 = 18755157416108.99;
    const FROZEN_ORBIT_SMA_M: [f64; 5] = [
        16.98247083318794,
        28.870200416419497,
        49.07934070791314,
        83.43487920345233,
        141.83929464586896,
    ];

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

    #[test]
    fn visibility_factor_is_cot_half_theta() {
        // cot(θ/2) at θ_min = 8° — the ONE visibility constant (≈ 14.301), frozen non-self-referentially.
        assert_eq!(
            visibility_factor(VISIBILITY_THETA_MIN_RAD),
            FROZEN_VISIBILITY_FACTOR
        );
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
        // The 5 planet orbit distances (semi-major axes, render m): ~17 / 29 / 49 / 83 / 142.
        let bodies = generate_system_forest(0, &c);
        let smas: Vec<f64> = bodies
            .iter()
            .skip(3)
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
    fn visual_demand_band_is_crossable_for_the_outer_planet() {
        // Non-vacuous now (unlike the toy where 1.2×extent swallowed everything): the OUTER planet is OUT
        // of spin-up range at the star, so a ship flying out CROSSES its band — spin_up_r < outer orbit.
        let (outer, orbit) = outer_planet_orbit(&UniverseConfig::visual_demand(15.0, 0.02));
        assert!(
            outer.aoi.spin_up_r_m() < orbit,
            "the outer planet is out of spin-up range at the star (crossable)"
        );
    }

    #[test]
    fn visual_demand_is_releasable_at_the_star() {
        // The mirror release fact: from the star the outer planet is past tear-down → released (reapable).
        let (outer, orbit) = outer_planet_orbit(&UniverseConfig::visual_demand(15.0, 0.02));
        assert!(
            outer.aoi.tear_down_r_m() < orbit,
            "the outer planet releases from the star (tear-down < outer orbit distance)"
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
        assert_eq!(c.scale.galaxy_r_m, GALAXY_R_M);
        assert_eq!(c.planet.ecc_cap, KEPLER_ECC_MAX);
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
        assert_eq!(bodies.len(), 3 + VISUAL_N_PLANETS as usize);
        // The 3 ambient bodies are StaticOffset (orbital_of None); each planet is Orbital + System child.
        assert_eq!(orbital_of(bodies[0].placement), None); // Universe
        assert_eq!(orbital_of(bodies[1].placement), None); // Galaxy
        assert_eq!(orbital_of(bodies[2].placement), None); // System A
        for planet in bodies.iter().skip(3) {
            assert_eq!(planet.parent, Some(SYSTEM_A));
            assert!(orbital_of(planet.placement).is_some());
        }
    }

    // ---- A2: seed-derived origin chain + fold (the iron rule) ----

    #[test]
    fn origin_link_derives_are_exercised() {
        let config = UniverseConfig::visual_scale();
        let planet = generate_system_forest(0, &config)
            .into_iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)))
            .expect("a planet");
        let orbital = origin_link_of(planet.placement);
        assert_eq!(OriginLink::Fixed, OriginLink::Fixed);
        assert_ne!(OriginLink::Fixed, orbital);
        assert!(format!("{orbital:?}").contains("Orbital"));
        assert!(format!("{:?}", OriginLink::Fixed).contains("Fixed"));
    }

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
        // condition, whose true arm would be uncoverable): a realm's FULL origin chain varies IFF the realm is
        // ITSELF a mover. A static realm inheriting a varying ancestor is exactly the case where the two sets
        // diverge (its chain varies but its own link is Fixed). Both filter arms are genuinely exercised — a
        // planet (Orbital ⇒ true) and an ambient body (Fixed ⇒ false) exist in every forest.
        let config = UniverseConfig::visual_scale();
        for seed in [0u64, 1, 7, 42, 100] {
            let bodies = generate_system_forest(seed, &config);
            let mut varying_chain: Vec<RealmId> = bodies
                .iter()
                .filter(|b| origin_varies(&origin_chain_over(&bodies, b.realm)))
                .map(|b| b.realm)
                .collect();
            let mut movers: Vec<RealmId> = bodies
                .iter()
                .filter(|b| origin_varies(&[origin_link_of(b.placement)]))
                .map(|b| b.realm)
                .collect();
            varying_chain.sort();
            movers.sort();
            assert_eq!(
                varying_chain, movers,
                "D-FO-7 (seed {seed}): a realm's chain varies IFF it is itself a mover — a divergence means a \
                 static realm now hangs under a moving ancestor, which the realm feed's movers-only filter \
                 would silently drop. Take the D-FO-7 decision before the widening lands."
            );
        }
    }

    #[test]
    fn fold_origin_is_identity_for_an_all_fixed_chain() {
        let (pos, vel) = fold_origin(
            &[OriginLink::Fixed, OriginLink::Fixed, OriginLink::Fixed],
            123.0,
        );
        assert_eq!(pos, LatticePos::local(DVec3::ZERO));
        assert_eq!(vel, DVec3::ZERO);
    }

    #[test]
    fn fold_origin_of_a_planet_chain_equals_its_orbital_state_at_cell_zero() {
        let config = UniverseConfig::visual_scale();
        let bodies = generate_system_forest(0, &config);
        let planet = bodies
            .iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)))
            .expect("a planet");
        let elements = orbital_of(planet.placement).expect("a planet is Orbital");
        let secs = 321.0;
        let (pos, vel) = fold_origin(&origin_chain_for_config(0, &config, planet.realm), secs);
        let state = orbital_state(&elements, secs);
        // Ancestors are all StaticOffset(ZERO) ⇒ the fold is the bare orbital state, cell 0 (byte-floor).
        assert!((pos.offset() - state.position).length() < 1e-9);
        assert_eq!(pos.cell(), glam::I64Vec3::ZERO);
        assert!((vel - state.velocity).length() < 1e-9);
    }

    #[test]
    fn fold_origin_of_a_fixed_under_an_orbital_rides_the_orbit() {
        // A static child (a station) under an orbiting parent: the Fixed link adds nothing, so the child's
        // frame origin folds to the SAME orbital position as its parent — it rides the orbit.
        let config = UniverseConfig::visual_scale();
        let planet = generate_system_forest(0, &config)
            .into_iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)))
            .expect("a planet");
        let secs = 55.0;
        let parent_chain = origin_chain_for_config(0, &config, planet.realm);
        let mut child_chain = parent_chain.clone();
        child_chain.push(OriginLink::Fixed);
        assert_eq!(
            fold_origin(&parent_chain, secs),
            fold_origin(&child_chain, secs)
        );
    }

    #[test]
    fn fold_origin_sums_two_orbital_levels() {
        let config = UniverseConfig::visual_scale();
        let planet = generate_system_forest(0, &config)
            .into_iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)))
            .expect("a planet");
        let e = orbital_of(planet.placement).expect("Orbital");
        let secs = 77.0;
        let (pos, vel) = fold_origin(&[OriginLink::Orbital(e), OriginLink::Orbital(e)], secs);
        let s = orbital_state(&e, secs);
        assert!((pos.offset() - (s.position + s.position)).length() < 1e-9);
        assert!((vel - (s.velocity + s.velocity)).length() < 1e-9);
    }

    #[test]
    fn origin_varies_is_true_for_a_mover_chain_and_false_for_all_fixed() {
        let config = UniverseConfig::visual_scale();
        let planet = generate_system_forest(0, &config)
            .into_iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)))
            .expect("a planet");
        assert!(origin_varies(&origin_chain_for_config(
            0,
            &config,
            planet.realm
        )));
        assert!(!origin_varies(&origin_chain_for_walk_config(
            &UniverseConfig::walk_scale(),
            SYSTEM_A
        )));
        assert!(!origin_varies(&[]));
    }

    #[test]
    fn origin_chain_for_config_ends_in_an_orbital_link_for_a_planet() {
        let config = UniverseConfig::visual_scale();
        let planet = generate_system_forest(0, &config)
            .into_iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)))
            .expect("a planet");
        let chain = origin_chain_for_config(0, &config, planet.realm);
        // Universe > Galaxy > System A > Planet: 4 links, the last Orbital, the rest Fixed.
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0], OriginLink::Fixed);
        assert_eq!(chain[1], OriginLink::Fixed);
        assert_eq!(chain[2], OriginLink::Fixed);
        // the last link is Orbital — the only non-Fixed variant, so assert_ne avoids a matches! false arm.
        assert_ne!(chain[3], OriginLink::Fixed);
    }

    #[test]
    fn origin_chain_for_walk_config_is_all_fixed_and_folds_to_identity() {
        let config = UniverseConfig::walk_scale();
        // Area A is the deepest walk realm: Universe > Galaxy > System A > Planet A > Area A (5 links).
        let chain = origin_chain_for_walk_config(&config, AREA_A);
        assert_eq!(chain.len(), 5);
        assert!(chain.iter().all(|l| *l == OriginLink::Fixed));
        let (pos, vel) = fold_origin(&chain, 999.0);
        assert_eq!(pos, LatticePos::local(DVec3::ZERO));
        assert_eq!(vel, DVec3::ZERO);
    }

    #[test]
    fn origin_chain_reads_no_child_list() {
        // THE IRON RULE: a realm's origin chain walks UP its parents only. Adding a SIBLING (another child of
        // the same parent) must leave the realm's chain AND its fold byte-identical — the chain never reads
        // the parent's child set. (The reverted attempt derived a realm's own position from its children.)
        let config = UniverseConfig::visual_scale();
        let bodies = generate_system_forest(0, &config);
        let planet = bodies
            .iter()
            .find(|b| matches!(b.realm, RealmId::Planet(_)))
            .expect("a planet");
        let planet_realm = planet.realm;
        let planet_parent = planet.parent;
        let elements = orbital_of(planet.placement).expect("Orbital");
        let baseline = origin_chain_over(&bodies, planet_realm);
        // A fresh forest + an appended SIBLING (same parent, different realm) — no clone of a body needed.
        let mut with_sibling = generate_system_forest(0, &config);
        with_sibling.push(GeneratedBody {
            realm: RealmId::Planet(0xDEAD_BEEF),
            parent: planet_parent,
            shape: Boundary::Shell { r: 1.0 },
            placement: Placement::Orbital(elements),
        });
        let after = origin_chain_over(&with_sibling, planet_realm);
        assert_eq!(
            baseline, after,
            "a sibling must not change the realm's origin chain"
        );
        assert_eq!(fold_origin(&baseline, 42.0), fold_origin(&after, 42.0));
    }

    #[test]
    fn origin_chain_of_an_unknown_realm_is_empty() {
        let config = UniverseConfig::visual_scale();
        assert!(origin_chain_for_config(0, &config, RealmId::Planet(0x00C0_FFEE)).is_empty());
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
        for (body, region) in bodies.iter().zip(&regions).skip(3) {
            assert!(orbital_of(body.placement).is_some(), "a planet is Orbital");
            assert_eq!(region.center.cell(), glam::I64Vec3::ZERO);
            assert_eq!(region.center.offset(), DVec3::ZERO);
        }
    }

    /// FRAME-COHERENT CONTAINMENT (the moving-realm crossing fix): a moving planet's position is authored
    /// ONCE — through its live frame placement (`LocalFrames::with_moving_child`) — and its region `center`
    /// is ZERO (the boundary sits at the body's OWN frame origin). So an occupant sitting exactly at the
    /// planet's live orbital position is judged INSIDE its SOI, and an occupant at the star (17.9 m away) is
    /// OUTSIDE — the SAME geometry both the parent shard (planet as a moving child) and the planet's own
    /// shard (planet at the identity) compute, so a crossing cannot flap. Regression guard against the epoch
    /// `center` being double-counted against the frame placement (which put the SOI ~17.9 m off the planet).
    #[test]
    fn a_moving_planet_soi_is_centered_on_its_live_position_not_double_counted() {
        use crate::celestial::secs_since_epoch;
        use crate::frame::LocalFrames;
        use crate::geometry::region_signed_distance;
        use crate::pose::StampedPose;
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
        let tick = crate::UniverseTick(200);
        // A moving realm carries NO baked position — its center is the origin of its own frame.
        assert_eq!(
            region.center.offset(),
            DVec3::ZERO,
            "a moving planet's region.center must be ZERO (position authored via the frame)",
        );
        // The parent shard's ephemeris: the planet is a MOVING child at its live orbital position.
        let ctx = LocalFrames::new(root, tick_hz).with_moving_child(region.frame, elements);
        let live = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz)).position;
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
                .skip(3)
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
        // Every visual planet's period is seconds-scale — not sub-µs (invisible), not years.
        for body in visual_forest().iter().skip(3) {
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
        // Containment order + headroom (System 150 ⊂ Galaxy 180 < cull 200): the System nests strictly
        // inside the renderable Galaxy, which nests strictly under the cull (each assert split — no `&&`).
        assert!(config.stellar.system_soi_r_m < config.scale.galaxy_r_m);
        assert!(config.scale.galaxy_r_m < config.scale.render_extent_m);
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
    fn the_canonical_preset_shares_the_generator_at_real_proportions() {
        // The SAME generate_system_forest serves canonical() — only the config VALUES differ (no-corner).
        let mut canon = UniverseConfig::canonical();
        canon.planet.n_planets = VISUAL_N_PLANETS;
        let bodies = generate_system_forest(0, &canon);
        assert_eq!(bodies.len(), 3 + VISUAL_N_PLANETS as usize);
        let inner = orbital_of(bodies[3].placement).expect("a planet is Orbital");
        // A canonical planet's sma is REAL AU metres; its star is the real solar mass (not synthetic).
        assert_eq!(
            inner.sma,
            orbital_axis_au(0, ORBITAL_A0_AU, ORBITAL_RATIO) * canon.scale.au_to_render_m
        );
        assert_eq!(inner.central_mass, CANONICAL_STAR_MASS_KG);
    }

    #[test]
    fn walk_path_is_untouched_and_visual_planet_ids_are_distinct() {
        // Byte-identity: the walk boot path still yields the frozen 7-body forest + empty mover roster
        // (the new generator is uncalled by any walk path).
        assert_eq!(realm_regions_for(0).len(), 7);
        assert!(moving_children_for(0, SYSTEM_A).is_empty());
        // The 5 visual planet ids are mutually distinct and NONE aliases the walk Planet(7) — the
        // child_seed salt/index avalanche keeps them off the roster ids (no silent alias).
        let ids: Vec<RealmId> = visual_forest().iter().skip(3).map(|b| b.realm).collect();
        assert_eq!(ids.len(), VISUAL_N_PLANETS as usize);
        let mut distinct = ids.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            VISUAL_N_PLANETS as usize,
            "every visual planet id is distinct"
        );
        for id in &ids {
            assert_ne!(*id, PLANET_A);
        }
    }
}
