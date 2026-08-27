//! THE ROSTER QUESTIONS: what a booting shard asks about the world it is about to host.
//!
//! Owns: the materialised world view, the direct-child roster, the moving-child roster a shard
//! AUTHORS placements for, the marker data its sleeping children glow with, and the neighbourhood a
//! shard evaluates containment against — its own realm plus its ancestor chain, plus a bounded child
//! set.
//!
//! Does NOT own: the hundreds of siblings a crowded parent may have. A shard asks about ITS OWN
//! level and one step down; anything further resolves through the directory, which is the answer
//! that still works when a galaxy is full.

use super::{
    GeneratedBody, Placement, StarPhotometrics, UniverseConfig, generate_system_forest,
    generate_walk_forest, orbital_of, realm_regions_for, to_regions,
};
use crate::celestial::OrbitalElements;
use glam::DVec3;
use vd_core::geometry::RealmRegion;
use vd_core::pose::RealmId;
use vd_core::realm_path::RealmLevel;
use vd_core::worldgen::{ancestor_realms, level_of, neighbourhood_scope};

/// The DIRECT MOVING children a shard hosting `hosted_realm` AUTHORS (D-45(a) realm-unification FA-2b):
/// each direct child (`parent == hosted_realm`) whose placement is a live `Orbital`. Under LAW-1 a
/// passive orbiting body is the ZERO-SIGNAL case — the parent shard re-authors its live pose each tick
/// from these `OrbitalElements` (the boot wraps them as injected `MotionFn`s for the placement
/// writer), never a static region `center`.
/// Returned `(realm, elements)` so the sim keys it against each region by realm. A branchless-shim (HR5):
/// the `Orbital`/`StaticOffset` match lives in the monomorphic [`orbital_of`] helper, not this closure.
/// The walk roster is ALL `StaticOffset`, so this is EMPTY at walk scale (byte-identity); the canonical
/// seed generation (P4/FA-5) is what populates it.
pub(crate) fn moving_children(
    bodies: &[GeneratedBody],
    hosted_realm: RealmId,
) -> Vec<(RealmId, OrbitalElements)> {
    bodies
        .iter()
        .filter(|b| b.parent == Some(hosted_realm))
        .filter_map(|b| orbital_of(b.placement).map(|e| (b.realm, e)))
        .collect()
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

// (`two_level_clearance_m` and `galaxy_shell_r_m` are DELETED — real-scale addendum §9.2: the
// upward interim solve collapsed into the ONE clearance law (`child_clearance_m`, §3.2) and the
// downward storage-fence chain (§A2.2). Their pins (`FROZEN_TWO_LEVEL_CLEARANCE_M`,
// `FROZEN_TWO_LEVEL_WORST_MARGIN_M`) retire with them; the §3.2 identity is the successor.)

// (`synthetic_central_mass`, `visual_au_to_render_m`, `visual_planet_soi_r_m`,
// `visual_outer_sma_render_m`, `visual_central_mass_kg` are DELETED — real-scale design §3.3.7:
// the compressed visual in-system geometry died with the re-solve; orbits are true-size around
// the star's real drawn mass.)

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

/// One sleeping child's marker DATUM for the window lane (Slice A → look_horizon.md slice 1):
/// the spectral class as its stable code (color) plus the main-sequence luminosity (brightness)
/// — exactly the two scalars the owner-ruled R4 datum names. The class→code cast lives HERE and
/// nowhere else. The boot plumbs these datums onto the shard's roster (`vd-sim` `ChildLuma`);
/// the BAG is framed per child by the sim through the one `vd_core::look::marker_bag` codec,
/// which appends the child's circumscribed extent (the presence floor's one radius).
#[must_use]
pub fn marker_datum(p: &StarPhotometrics) -> (u8, f64) {
    (p.class as u8, p.luma_lsun)
}

/// ★ THE STAR CATALOGUE (S11) — every star system in the galaxy, as the client draws it.
///
/// ONE FOLD, and the reason there is exactly one is the owner's second condition on this message:
/// *"a test asserts the encoded catalogue and the folded one are IDENTICAL — one truth, two producers,
/// which will otherwise drift at the first patch and nobody will notice."* Two producers exist because
/// a shard emits from the forest it actually BOOTED while the test folds from the seed; if those ever
/// disagree the shard is serving a different galaxy than the seed describes, and a player would fly to
/// a star that is not there. This function is the single expression both go through, so the only way
/// they can differ is if the boot itself differs — which is exactly what the test is for.
///
/// A row carries no frame: a catalogue is stated in the GALAXY's frame and nowhere else. It carries no
/// sub-cell residual either — measured, every generated system's centre is exactly cell-aligned.
///
/// ORDERED BY REALM so two folds of the same world are byte-identical whatever order the forest was
/// walked in. Determinism here is not a nicety: the generation is derived from these bytes.
#[must_use]
pub fn star_catalogue(
    regions: &[RealmRegion],
    photometrics: &[(RealmId, StarPhotometrics)],
) -> Vec<vd_core::look::StarRow> {
    let galaxy = regions
        .iter()
        .find(|r| matches!(r.realm, RealmId::Galaxy(_)))
        .map(|r| r.realm);
    let mut rows: Vec<vd_core::look::StarRow> = regions
        .iter()
        .filter(|r| matches!(r.realm, RealmId::System(_)) && r.parent == galaxy)
        .map(|r| {
            // A star with no photometric draw still belongs in the sky — it is a place you can fly to.
            // It glows NOT, which the presence floor already says elsewhere: absence of a datum is
            // absence of data, never a default.
            let (class_code, luma_lsun) = photometrics
                .iter()
                .find(|(realm, _)| *realm == r.realm)
                .map_or((0, 0.0), |(_, p)| marker_datum(p));
            vd_core::look::StarRow {
                realm: r.realm,
                cell: r.center.in_parents_frame().cell(),
                class_code,
                luma_lsun,
            }
        })
        .collect();
    rows.sort_by_key(|r| r.realm);
    rows
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

// ===== THE FIXTURE PLANT (look_horizon.md slice 5 — G-IDENTICAL / SL5 fixture-forest doctrine) ==
// Stations and areas are built by PLAYERS; no seed emits one. The pixel gate that proves the look
// horizon is kind-blind (HR4) therefore needs player-built content, and SL5's landed ruling is
// that fixtures may plant player-built regions on THE world. There was no process-tier plant path
// before this (the walk fixture forest is a SEPARATE hand-placed world, retired from the boots),
// so this is the MINIMAL LAWFUL one: a named pair appended to the generated forest through the ONE
// generator/lowering, selected by `UniverseConfig::fixture_plant`, measured by the SAME boot
// fences (`guard_visibility_climb_bounded` at every process boot) as everything else. It is NOT
// build admission (the live build feature does not exist yet — D-LOOK-1): a fixture states the
// content, and the fences judge it exactly as they will judge a player's candidate.

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
    pub(crate) bodies: Vec<GeneratedBody>,
    pub(crate) regions: Vec<RealmRegion>,
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

    /// THE DEFAULT HOME STANDOFF (T2's forced re-derivation of the login pose): the home
    /// realm's own centre stopped being empty space the day the STAR became a body-bearing
    /// child there — a zero-offset spawn would put every new account inside the Star realm.
    /// The pose pushes out along +Z (the I-AXIS-clear polar axis) by TWICE the largest STATIC
    /// child region that CONTAINS the centre (the star: 2× its dust bound). Orbital children
    /// never contain the centre (periapsis − shell > 0, the sibling fences) and static
    /// children clear of the centre contribute nothing — so on a world whose origin IS empty
    /// (the walk fixture) this is ZERO, byte-identical to the pre-T2 spawn. Derived from the
    /// forest this view already holds, never a picked number.
    #[must_use]
    pub fn default_home_offset_m(&self) -> DVec3 {
        let Some(home) = vd_core::worldgen::default_home_realm(self.regions()) else {
            return DVec3::ZERO;
        };
        let clearing_z = self
            .bodies
            .iter()
            .filter(|b| b.parent == Some(home))
            .filter_map(|b| match b.placement {
                Placement::StaticOffset(at) => {
                    let bound = b.shape.finite_extent();
                    (at.length() < bound).then_some(2.0 * bound)
                }
                Placement::Orbital(_) => None,
            })
            .fold(0.0, f64::max);
        DVec3::new(0.0, 0.0, clearing_z)
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
