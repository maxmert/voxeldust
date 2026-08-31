//! THE WALK-SCALE MANDATE FOREST: a hand-placed topology small enough to walk across.
//!
//! Owns: the walk geometry constants and the forest they build — two disjoint systems with a
//! walkable gap of galaxy between them, a planet inside one, a station and an area inside those.
//!
//! Does NOT own: a reduced world. This is not a smaller variant of the seed universe (SL5 forbids
//! one); it is a DIFFERENT, explicitly hand-authored topology whose whole purpose is that a person
//! can traverse every boundary in it on foot, and it lowers through exactly the same code the
//! generated world does.

use super::{GeneratedBody, Placement, UniverseConfig, to_regions};
use glam::DVec3;
use vd_core::geometry::{Boundary, RealmRegion};
use vd_core::worldgen::{
    AREA_A, GALAXY, PLANET_A, PLANET_B, PLANET_C, STATION_A, SYSTEM_A, SYSTEM_B, UNIVERSE,
};
#[cfg(test)]
use vd_core::worldgen::{GALAXY_SEED, SYSTEM_A_SEED, UNIVERSE_SEED};

// --- P3 WALK-SCALE geometry (placeholders; P4/P5 makes every radius/center `f(seed[, tick])`, D-44). ---
/// The ambient-root (Universe) radius — effectively unbounded; an entity beyond it still resolves to the
/// Universe by the container fold IDENTITY. Non-renderable (far above the render-extent threshold).
pub(crate) const UNIVERSE_R_M: f64 = 1.0e9;
/// The galaxy radius — FINITE (it encloses the star systems), and it is the between-systems space an
/// entity occupies after leaving one system SOI and before entering the next. RENDERABLE (below the extent
/// threshold) so the client draws it as the CONTAINING box around the two systems — an entity in the gap is
/// visibly still inside the Galaxy realm, never orphaned. Contains System B's far face (130 + 40 = 170).
pub(crate) const GALAXY_R_M: f64 = 180.0;
/// A star-system SOI radius (walk scale).
pub(crate) const SYSTEM_SOI_R_M: f64 = 40.0;
/// A planet SOI radius (walk scale), nested inside a system.
pub(crate) const PLANET_SOI_R_M: f64 = 10.0;
/// System B's center on +X — a disjoint sibling of System A with a WALKABLE gap of galaxy between them
/// (System A far-face 40, System B near-face 90 ⇒ a ~50 m pure-Galaxy gap: leaving A you are IN the Galaxy
/// realm until you enter B). The round-trip probe points 0/50/100 still resolve System A / Galaxy / System B.
pub(crate) const SYSTEM_B_OFFSET_M: f64 = 130.0;
/// Planet A's center inside System A (offset from the star at the origin).
pub(crate) const PLANET_A_OFFSET_M: f64 = 20.0;
/// Station A's center inside System A, on the -X side (opposite Planet A on +X), clear of the origin
/// crowd + the round-trip legs at x = 0/50/100. A Cartesian box, not an SOI shell.
pub(crate) const STATION_A_OFFSET_M: f64 = -25.0;
/// Station A's box half-extent (a small docked-station volume). `|-25| + 5 = 30 < 40` ⇒ fully inside
/// System A's r=40 SOI.
pub(crate) const STATION_HALF_M: f64 = 5.0;
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
pub(crate) const AREA_OFFSET_M: f64 = 5.0;
/// Area A's box half-extent (a small sub-planet district volume).
pub(crate) const AREA_HALF_M: f64 = 3.0;

/// SYSTEM_A's RNG lineage root→leaf `[Universe, Galaxy, System]` — MUST equal
/// `realm_path::system_path(SYSTEM_A_SEED).lineage_seeds()` so every shard hosting System A draws the
/// IDENTICAL per-system stream by construction (HR1); consumed once by [`generate_system_forest`].
#[cfg(test)]
pub(crate) const SYSTEM_A_LINEAGE: [u64; 3] = [UNIVERSE_SEED, GALAXY_SEED, SYSTEM_A_SEED];

// ===== THE IN-SYSTEM TRUE-SIZE RE-SOLVE (real-scale design §3.3, landed by the taxonomy arc) ====
//
// The interim compressed in-system world (150 m shells, 3.95 m SOIs, a synthetic Kepler-tuned
// central mass, an AU compression factor) is DELETED. In-system space is TRUE SIZE (χ = 1
// exactly): the ladder is anchored to the star's own luminosity, planet masses are drawn, radii
// and SOIs derived, and each system's shell is SOLVED by the one clearance law.

/// The walk-scale mandate forest as config-driven bodies, in forest order (Universe → Galaxy →
/// System A → Planet A → System B → Station A → Area A). All placements are `StaticOffset`, so the
/// lowering is byte-identical to the pre-generator forest. The GENERIC seed-driven child
/// enumeration (canonical scale) is deferred to P4 — its bodies are not live containment regions
/// until the D-41 non-zero-cell re-quantization.
pub(crate) fn generate_walk_forest(config: &UniverseConfig) -> Vec<GeneratedBody> {
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
        // Never drawn (1e9 m was beyond the render cut since P3): `look = None`, the two-body
        // law stated structurally at the walk root too.
        GeneratedBody {
            realm: UNIVERSE,
            parent: None,
            shape: shell(sc.universe_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: None,
        },
        // Galaxy: the finite between-systems space, nested in the Universe. The WALK galaxy IS
        // drawn (the 180 m containing box every walk gate has always seen) — bound == look at
        // human scale, so the split changes no walk pixel.
        GeneratedBody {
            realm: GALAXY,
            parent: Some(UNIVERSE),
            shape: shell(sc.galaxy_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: Some(shell(sc.galaxy_r_m)),
        },
        // Star system A: nested in the Galaxy at the origin.
        GeneratedBody {
            realm: SYSTEM_A,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: origin,
            photometrics: None,
            taxon: None,
            look: Some(shell(st.system_soi_r_m)),
        },
        // Planet A: nested in system A, offset from the star.
        GeneratedBody {
            realm: PLANET_A,
            parent: Some(SYSTEM_A),
            shape: shell(pl.planet_soi_r_m),
            placement: at_x(sa.planet_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(shell(pl.planet_soi_r_m)),
        },
        // Star system B: a DISJOINT sibling of system A under the Galaxy (a walkable galaxy gap between).
        GeneratedBody {
            realm: SYSTEM_B,
            parent: Some(GALAXY),
            shape: shell(st.system_soi_r_m),
            placement: at_x(sa.system_b_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(shell(st.system_soi_r_m)),
        },
        // Station A: a first-class Station BOX under System A (depth 3), on the -X side opposite Planet A.
        GeneratedBody {
            realm: STATION_A,
            parent: Some(SYSTEM_A),
            shape: boxed(sa.station_half_m),
            placement: at_x(sa.station_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(boxed(sa.station_half_m)),
        },
        // Area A: a first-class sub-planet Area BOX under Planet A (depth 4) — the DEEPEST region.
        GeneratedBody {
            realm: AREA_A,
            parent: Some(PLANET_A),
            shape: boxed(sa.area_half_m),
            placement: at_x(sa.area_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(boxed(sa.area_half_m)),
        },
        // ★ SYSTEM B'S OWN TWO PLANETS (2026-08-31), APPENDED LAST — never inserted. Every existing
        // body keeps its index, so a golden that lists the forest in order still matches on the rows
        // it already had. That is the same append-never-insert discipline the per-system draw stream
        // keeps, and for the same reason: an insertion moves everything after it.
        //
        // System B is the only hand-placed system that is
        // actually SOMEWHERE — A sits at the galaxy's origin — so it is the one a worked example can
        // use to prove that a parent ADDS its child's placement. It held nothing, so that example had
        // to reach for the generator, which since ruling G8 cannot be asked for a two-star galaxy.
        // Two planets, because the story needs a sibling to refuse as well as a subject to place.
        GeneratedBody {
            realm: PLANET_B,
            parent: Some(SYSTEM_B),
            shape: shell(pl.planet_soi_r_m),
            placement: at_x(sa.planet_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(shell(pl.planet_soi_r_m)),
        },
        GeneratedBody {
            realm: PLANET_C,
            parent: Some(SYSTEM_B),
            shape: shell(pl.planet_soi_r_m),
            placement: at_x(-sa.planet_offset_m),
            photometrics: None,
            taxon: None,
            look: Some(shell(pl.planet_soi_r_m)),
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
