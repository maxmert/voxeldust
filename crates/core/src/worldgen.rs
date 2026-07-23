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
use crate::geometry::{BandError, Boundary, ContainmentBand, RealmRegion};
use crate::pose::{LatticePos, RealmId, frame_for_realm};
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

// --- FA-5 (D-45(a)) VISUAL-scale single-system generator — the window-friendly synthetic scale whose
// planets ORBIT visibly. Every visual geometry number is DERIVED (helpers below), not a literal. ---
/// `child_seed` salt distinguishing PLANET-kind children under a system (a fixed kind discriminant;
/// `child_seed` avalanches `(parent, salt, index)`, so a distinct salt keeps planet ids off other kinds).
const PLANET_SALT: u64 = 0x504c_414e_4554; // "PLANET"
/// SYSTEM_A's RNG lineage root→leaf `[Universe, Galaxy, System]` — MUST equal
/// `realm_path::system_path(SYSTEM_A_SEED).lineage_seeds()` so every shard hosting System A draws the
/// IDENTICAL per-system stream by construction (HR1); consumed once by [`generate_system_forest`].
const SYSTEM_A_LINEAGE: [u64; 3] = [UNIVERSE_SEED, GALAXY_SEED, SYSTEM_A_SEED];
/// Visual-scale planet count — exercises the geometric spacing [`orbital_axis_au`] for n=0,1,2 (not a
/// single-orbit special case); small so the whole system frames inside the render window.
const VISUAL_N_PLANETS: u32 = 3;
/// Headroom (render m) between the OUTER planet's SOI face and the System SOI surface, so the outer
/// body renders STRICTLY inside its System box ([`visual_au_to_render_m`] solves to place it here).
const VISUAL_SYSTEM_MARGIN_M: f64 = 4.0;
/// A planet's SOI radius as a fraction of the SMALLEST inter-orbit gap; `< 0.5` guarantees adjacent
/// SOIs never overlap (the non-overlap invariant is a pinned test, not a hand-tuned coincidence).
const VISUAL_SOI_GAP_FRACTION: f64 = 0.35;
/// The OUTER (slowest) planet's orbital period in seconds — a MAJESTIC-but-visible pace for the human
/// window view (the inner planets are faster by Kepler-3: `T ∝ a^1.5`, so ~37 s / ~81 s / 180 s for the
/// 3 planets). Feeds the synthetic central mass via the Kepler-3 inversion. NOT tuned to a frantic
/// few-second orbit: the automated 2-capture render gate samples universe ticks FAR ENOUGH apart to see
/// the sweep, so the period is free to be leisurely for a human watching.
const VISUAL_TARGET_OUTER_PERIOD_S: f64 = 180.0;
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

/// The static tick-0 epoch center of a placement as a `cell == ZERO` [`LatticePos`] (D-41: step-1
/// keeps `cell == ZERO`; the moving ephemeris re-derives the live origin per tick at step-2). A
/// branchless shim over [`placement_offset`] (HR5: the one match lives in the monomorphic helper).
fn epoch_offset_in_parent(placement: Placement) -> LatticePos {
    LatticePos::local(placement_offset(placement))
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
        .map(|b| RealmRegion {
            realm: b.realm,
            center: epoch_offset_in_parent(b.placement),
            frame: frame_for_realm(b.realm, b.parent)
                .expect("roster realms have a canonical frame"),
            shape: b.shape,
            band,
            parent: b.parent,
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

/// Closed-form inversion of Kepler's third law `T = 2π·√(a³/μ)`, `μ = G·M` → the central mass (kg)
/// that yields orbital period `target_period_s` at semi-major axis `sma_ref_m`. The SYNTHETIC-mass crux
/// for the visual scale: a real star mass at tens-of-metres `sma` gives a sub-µs (invisible) period, so
/// the visual system uses a synthetic mass tuned to a seconds-scale period instead. Straight-line f64.
fn synthetic_central_mass(sma_ref_m: f64, target_period_s: f64) -> f64 {
    TAU * TAU * sma_ref_m.powi(3) / (G * target_period_s * target_period_s)
}

/// The AU→render-metre compression solved so the OUTER planet's orbit + its SOI + [`VISUAL_SYSTEM_MARGIN_M`]
/// sit EXACTLY at the System SOI surface (containment, vet far-plane fix). Denominator = the outer orbit
/// axis (AU) + the planet SOI expressed in AU (a fraction of the smallest inter-orbit gap). Branchless.
fn visual_au_to_render_m() -> f64 {
    let outer_axis_au = orbital_axis_au(VISUAL_N_PLANETS - 1, ORBITAL_A0_AU, ORBITAL_RATIO);
    let soi_au = VISUAL_SOI_GAP_FRACTION * ORBITAL_A0_AU * (ORBITAL_RATIO - 1.0);
    (SYSTEM_SOI_R_M - VISUAL_SYSTEM_MARGIN_M) / (outer_axis_au + soi_au)
}

/// The planet SOI radius (render m) = the gap-fraction × the SMALLEST inter-orbit gap → adjacent SOIs
/// never overlap by construction. Straight-line f64.
fn visual_planet_soi_r_m() -> f64 {
    VISUAL_SOI_GAP_FRACTION * ORBITAL_A0_AU * (ORBITAL_RATIO - 1.0) * visual_au_to_render_m()
}

/// The OUTER (slowest) planet's semi-major axis in render metres — the period-tuning reference.
fn visual_outer_sma_render_m() -> f64 {
    orbital_axis_au(VISUAL_N_PLANETS - 1, ORBITAL_A0_AU, ORBITAL_RATIO) * visual_au_to_render_m()
}

/// The synthetic central mass (kg) placing the OUTER planet's period at [`VISUAL_TARGET_OUTER_PERIOD_S`].
fn visual_central_mass_kg() -> f64 {
    synthetic_central_mass(visual_outer_sma_render_m(), VISUAL_TARGET_OUTER_PERIOD_S)
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
pub fn realm_regions_for(_seed_universe: u64) -> Vec<RealmRegion> {
    let config = UniverseConfig::walk_scale();
    to_regions(&generate_walk_forest(&config), &config)
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
    let all = realm_regions_for(seed_universe);
    // A realm is IN-SCOPE iff it is an ancestor of, or a child of, ANY held realm. Collect the qualifying
    // realm set first (deduped), then filter the canonical forest ONCE so the output keeps forest order.
    let mut scope: std::collections::BTreeSet<RealmId> = std::collections::BTreeSet::new();
    for &hosted in held {
        for a in ancestor_realms(&all, hosted) {
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

/// THE one config home for the seed universe generator — six named sub-structs (no god-struct),
/// every field seed-derivable and doc-cited (no magic numbers).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UniverseConfig {
    pub scale: ScaleConfig,
    pub galaxy: GalaxyConfig,
    pub stellar: StellarConfig,
    pub planet: PlanetConfig,
    pub satellite: SatelliteConfig,
    pub band: BandConfig,
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
        }
    }

    /// The VISUAL-scale preset (D-45(a) FA-5): the window-friendly synthetic-scale SINGLE system whose
    /// planets ORBIT visibly. Reuses `walk_scale()`'s physics/taxonomy/band + ambient radii VERBATIM
    /// (so the System/Galaxy render at the proven walk sizes, well under the camera far-plane) and
    /// overrides ONLY the four fields the moving planets need: the AU→render compression + planet SOI
    /// (both DERIVED so the outer orbit + SOI + margin land exactly at the System surface — provable
    /// non-overlap + strict containment), the SYNTHETIC central mass (Kepler-3-tuned to a seconds-scale
    /// period), and `n_planets`. Sibling of `walk_scale()`/`canonical()`; the SAME
    /// [`generate_system_forest`] serves all three (canonical differs only in these values — zero new
    /// generator code at P4).
    #[must_use]
    pub fn visual_scale() -> UniverseConfig {
        let mut cfg = UniverseConfig::walk_scale();
        cfg.scale.au_to_render_m = visual_au_to_render_m();
        cfg.stellar.central_mass_kg = visual_central_mass_kg();
        cfg.planet.planet_soi_r_m = visual_planet_soi_r_m();
        cfg.planet.n_planets = VISUAL_N_PLANETS;
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
    fn to_regions_lowers_an_orbital_body_to_a_static_cell_zero_center() {
        // The Orbital placement arm (canonical/seed_derived bodies; not live at walk scale) lowers
        // via orbital_state(e, 0.0) into a cell == ZERO center — the arm the walk gate never hits.
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
        assert_eq!(
            regions[0].center.offset(),
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

    // ===== FA-5 S1: the config-driven VISUAL-scale Orbital generator =====================

    /// The visual-scale system forest at seed 0 (helper for the tests below).
    fn visual_forest() -> Vec<GeneratedBody> {
        generate_system_forest(0, &UniverseConfig::visual_scale())
    }

    #[test]
    fn visual_scale_preset_is_walk_physics_with_derived_visual_geometry() {
        let c = UniverseConfig::visual_scale();
        // Ambient radii + render extent + eccentricity physics are REUSED from walk (proven under the
        // far-plane); only the four moving-planet fields are overridden.
        assert_eq!(c.scale.render_extent_m, MAX_RENDERABLE_EXTENT_M);
        assert_eq!(c.stellar.system_soi_r_m, SYSTEM_SOI_R_M);
        assert_eq!(c.scale.galaxy_r_m, GALAXY_R_M);
        assert_eq!(c.planet.ecc_cap, KEPLER_ECC_MAX);
        assert_eq!(c.planet.ecc_sigma, ECC_SIGMA);
        assert_eq!(c.planet.incl_sigma, INCL_SIGMA);
        // The four overridden fields are the DERIVED helper values (never literals).
        assert_eq!(c.scale.au_to_render_m, visual_au_to_render_m());
        assert_eq!(c.stellar.central_mass_kg, visual_central_mass_kg());
        assert_eq!(c.planet.planet_soi_r_m, visual_planet_soi_r_m());
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

    #[test]
    fn realm_regions_for_config_bakes_each_planet_orbital_epoch_at_cell_zero() {
        let config = UniverseConfig::visual_scale();
        let bodies = generate_system_forest(0, &config);
        let regions = realm_regions_for_config(0, &config);
        assert_eq!(regions.len(), bodies.len());
        // Each planet region lowers its Orbital placement to a cell==ZERO center at the tick-0 epoch —
        // the `Placement::Orbital` lowering arm, now in a REAL (non-test) path.
        for (body, region) in bodies.iter().zip(&regions).skip(3) {
            let elements = orbital_of(body.placement).expect("a planet is Orbital");
            assert_eq!(region.center.cell(), glam::I64Vec3::ZERO);
            assert_eq!(
                region.center.offset(),
                orbital_state(&elements, 0.0).position
            );
        }
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
            sma: visual_outer_sma_render_m(),
            ecc: 0.0,
            inclination: 0.0,
            raan: 0.0,
            arg_periapsis: 0.0,
            mean_anomaly_epoch: 0.0,
            central_mass: visual_central_mass_kg(),
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
        // Far-plane: the largest FINITE renderable region (the System) is well under the vet 120 m cap,
        // and the render extent covers it (each assert split — no `&&`).
        assert!(config.stellar.system_soi_r_m <= 120.0);
        assert!(visual_planet_soi_r_m() <= 120.0);
        assert!(config.scale.render_extent_m >= config.stellar.system_soi_r_m);
        // The OUTER planet (orbit + SOI) sits STRICTLY inside the System SOI surface (containment).
        assert!(
            visual_outer_sma_render_m() + visual_planet_soi_r_m() < config.stellar.system_soi_r_m
        );
        // NON-OVERLAP: every adjacent orbit gap exceeds two planet SOIs (the smallest gap binds).
        let two_soi = 2.0 * visual_planet_soi_r_m();
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
        // The 3 visual planet ids are mutually distinct and NONE aliases the walk Planet(7) — the
        // child_seed salt/index avalanche keeps them off the roster ids (no silent alias).
        let ids: Vec<RealmId> = visual_forest().iter().skip(3).map(|b| b.realm).collect();
        assert_eq!(ids.len(), 3);
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[0], ids[2]);
        for id in &ids {
            assert_ne!(*id, PLANET_A);
        }
    }
}
