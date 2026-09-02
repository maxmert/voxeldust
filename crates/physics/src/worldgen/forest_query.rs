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
    generate_walk_forest, orbital_of, realm_regions_for, system_forest_cached, to_regions,
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
    system_forest_cached(seed_universe, config)
        .iter()
        .filter(|b| b.parent == Some(hosted))
        // Every kind has a level since 2026-09-01, so this maps rather than filters. It used to be a
        // `filter_map` that silently DROPPED any child without one — which meant a ship child vanished
        // from this roster instead of being refused, and nothing counted it.
        .map(|b| level_of(b.realm))
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
/// ★ THE SKY, FROM THE SYSTEM LAYER ALONE (owner ruling 2026-08-29) — the star rows a gateway states
/// once at boot, and the generation stamp that names them.
///
/// The sky is a list of STAR SYSTEMS. It has never contained a planet or a moon. This builds only
/// those, through [`generate_system_layer`], instead of folding the whole forest and keeping one row
/// in sixteen: MEASURED on THE world, 3 500 479 objects built to state 233 220 rows.
///
/// The rows are byte-identical to the ones the full fold produced — the layer places every system
/// with the same function, from the same stream, and pushes it with the same rule, which the identity
/// gate beside this measures rather than assumes.
/// ★ WHAT ONE SHARD BOOTS WITH — its neighbourhood and its mover roster, from its OWN SUBTREE
/// (owner ruling 2026-08-29).
///
/// A shard runs ONE realm. It used to call two functions that EACH built the whole forest and
/// filtered: MEASURED on THE world, 3 500 479 bodies built twice to keep 13 rows and one roster.
/// That is the login timeout a player hits — the gateway waits for the home realm and the shard is
/// still copying the galaxy.
///
/// Built ONCE here, read twice, and the generator's own body type never leaves the crate.
#[must_use]
pub fn shard_boot_world(
    seed_universe: u64,
    config: &UniverseConfig,
    held: &std::collections::BTreeSet<RealmId>,
    hosted: RealmId,
    lineage: &std::collections::BTreeSet<RealmId>,
) -> (Vec<RealmRegion>, Vec<(RealmId, OrbitalElements)>) {
    let (regions, movers, _) = shard_boot_world_lit(seed_universe, config, held, hosted, lineage);
    (regions, movers)
}

/// A shard's boot world with its light: the regions it holds, its movers' orbits, and the
/// photometric draw of every lit body.
pub type LitBootWorld = (
    Vec<RealmRegion>,
    Vec<(RealmId, OrbitalElements)>,
    Vec<(RealmId, StarPhotometrics)>,
);

/// ★ THE SHARD BOOT'S WHOLE ANSWER, FROM ONE SUBTREE BUILD (2026-08-30).
///
/// A shard boot needs three things from its own subtree: the region forest, the mover roster, and
/// the marker draw for each child it may state a point of light about. It used to build the subtree
/// TWICE — once here, once in [`subtree_photometrics`] — and on THE world a subtree build folds the
/// star-system layer, 233 222 bodies, each time.
///
/// The doc above `visual_regions_and_movers` already claimed "built once, read twice". This makes it
/// true.
#[must_use]
pub fn shard_boot_world_lit(
    seed_universe: u64,
    config: &UniverseConfig,
    held: &std::collections::BTreeSet<RealmId>,
    hosted: RealmId,
    lineage: &std::collections::BTreeSet<RealmId>,
) -> LitBootWorld {
    shard_boot_world_built(seed_universe, config, held, hosted, lineage, &[], None)
}

/// ★ THE SAME WORLD, PLUS WHAT PEOPLE BUILT HERE (D-MOVE-2; owner rulings 2026-09-01) — THE one
/// implementation; [`shard_boot_world_lit`] is this with no berths, which is what a realm holding no
/// built children has.
///
/// **THE BERTHS GO THROUGH THE SAME LOWERING EVERY GENERATED BODY GOES THROUGH.** They are appended to
/// the subtree as bodies and lowered by `to_regions` with the rest, so a built child gets its band, its
/// interest radius and its frame from exactly the code a planet gets them from. There is no second
/// lowering and no second generator (SL5).
///
/// **APPENDED LAST, so every generated body is byte-identical.** With no berths the forest must be the
/// forest that was there before this existed, to the byte — which is what the gate below asserts.
///
/// ⚠ **THIS IS NOT THE VARIANT THAT WAS BACKED OUT.** That was a setting on the world config, read
/// independently by each process, so two processes held different worlds. These are ROWS a realm read
/// from its own file: the generator is untouched, nothing selects anything, and an empty list is this
/// world exactly.
#[must_use]
pub fn shard_boot_world_built(
    seed_universe: u64,
    config: &UniverseConfig,
    held: &std::collections::BTreeSet<RealmId>,
    hosted: RealmId,
    lineage: &std::collections::BTreeSet<RealmId>,
    berths: &[(RealmId, vd_core::built::Berth)],
    own_row: Option<(
        RealmId,
        vd_core::geometry::Boundary,
        vd_core::geometry::Boundary,
    )>,
) -> LitBootWorld {
    let mut subtree = super::realm_subtree(seed_universe, config, held, lineage);
    for (parent, berth) in berths {
        // A STATIC OFFSET, and only a starting one: from its first tick the parent authors this
        // child's placement from the pushes it states. The berth is where the hull sits before
        // anybody touches the controls.
        subtree.push(built_row(
            berth.child,
            *parent,
            berth.bound,
            berth.look,
            berth.offset_m,
        ));
    }
    // ★ THE SHARD'S OWN ROW, WHEN THE SEED DID NOT MAKE IT (owner ruling 2026-09-01). The realm
    // states WHAT IT IS — its walls and its outline. It states no position and is told none here.
    //
    // Every shard holds a row for ITSELF — `neighbourhood_scope` keeps the hosted realm, its ancestors
    // and its direct children, and the boot refuses a world it cannot find itself in. A seed body gets
    // that row from the generator. A BUILT realm cannot: its berth lives in its PARENT's file, and a
    // sealed shard may not read another realm's file (HR1). So it writes the row from the one record it
    // legitimately holds — its own body — and the caller supplies its parent's name off the lineage the
    // spawn already sent it.
    //
    // ★ AT ZERO, AND THAT IS THE POINT (SL1 clauses 3-5). This field means "where I sit inside my
    // parent", which is the PARENT's number about me and never mine. MEASURED before this was written:
    // no code anywhere reads the HOSTED realm's own centre — every read of it is a parent reading a
    // child (`rebuild_child_index` filters `parent == Some(own)`; `author_book_driven` walks
    // `direct_children`) or a parent reading a sibling it authored. So the zero is never opened, and a
    // realm that never HOLDS its own position cannot state one. That is the law made structural rather
    // than remembered.
    //
    // Guarded on ABSENCE, not on kind (SL4/HR3): a realm the seed already placed keeps the seed's row,
    // so this adds nothing for a planet and the same code serves a station, an area and a ship.
    if let Some((parent, bound, look)) = own_row
        && !subtree.iter().any(|b| b.realm == hosted)
    {
        subtree.push(built_row(hosted, parent, bound, look, DVec3::ZERO));
    }
    let regions = neighbourhood_scope(&to_regions(&subtree, config), held);
    let movers = moving_children(&subtree, hosted);
    let lit = subtree
        .iter()
        .filter_map(|b| b.photometrics.map(|p| (b.realm, p)))
        .collect();
    (regions, movers, lit)
}

/// ONE row for one built thing (HR3) — a berth its parent authored, or a shard's own hull. A built
/// body has no seed stream, so it draws no light and belongs to no taxon: that absence IS what "the
/// seed did not make this" looks like in the data.
fn built_row(
    realm: RealmId,
    parent: RealmId,
    bound: vd_core::geometry::Boundary,
    look: vd_core::geometry::Boundary,
    offset_m: DVec3,
) -> GeneratedBody {
    GeneratedBody {
        realm,
        parent: Some(parent),
        shape: bound,
        placement: Placement::StaticOffset(offset_m),
        photometrics: None,
        taxon: None,
        look: Some(look),
    }
}

#[must_use]
pub fn sky_from_system_layer(
    seed_universe: u64,
    config: &UniverseConfig,
) -> Vec<vd_core::look::StarRow> {
    let layer = super::generate_system_layer(seed_universe, config);
    let photometrics: Vec<(RealmId, StarPhotometrics)> = layer
        .iter()
        .filter_map(|b| b.photometrics.map(|p| (b.realm, p)))
        .collect();
    star_catalogue(&to_regions(&layer, config), &photometrics)
}

/// ★ THE MARKER DRAWS ONE SHARD NEEDS — from its OWN SUBTREE, never the whole galaxy (2026-08-29).
///
/// A shard states a point of light for its held realms and their DIRECT children, and for nothing
/// else. [`system_photometrics_for_config`] answers the same question by building EVERY body in the
/// galaxy — on THE world, 3.5 million of them — and discarding all but a handful.
///
/// ★ MEASURED (2026-08-29). This ran at every shard boot. Once a galaxy shard held its 233 220
/// children, the caller's own per-row scan over the region list turned the pair QUADRATIC: a test on
/// THE world ran for 2 hours 23 minutes at 5.9 GB and had not finished. In the live cluster it is the
/// bulk of a 101-second bring-up.
///
/// `realm_subtree` is the SAME generator reading the SAME streams in the same order, asked only for
/// this shard's own part of the world — so the rows are the rows the full build would have produced.
/// ★ THE WORLD AS ITS STAR SYSTEMS — the layer, lowered, with no planet or moon built (2026-08-29).
///
/// Answers "which star systems exist, and where" without building what is INSIDE them. On THE world
/// that is 233 222 bodies instead of 3 500 479, and the rows for the systems are identical either way
/// — the layer places every system with the same function, from the same stream.
///
/// For a caller that only needs to NAME a system (which realm is home, which galaxy holds it), the
/// full forest is 15× the bodies and about 7 GB of memory for an answer the layer already carries.
#[must_use]
pub fn system_layer_view(seed_universe: u64, config: &UniverseConfig) -> WorldView {
    WorldView::of(super::generate_system_layer(seed_universe, config), config)
}

#[must_use]
pub fn subtree_photometrics(
    seed_universe: u64,
    config: &UniverseConfig,
    held: &std::collections::BTreeSet<RealmId>,
    lineage: &std::collections::BTreeSet<RealmId>,
) -> Vec<(RealmId, StarPhotometrics)> {
    super::realm_subtree(seed_universe, config, held, lineage)
        .iter()
        .filter_map(|b| b.photometrics.map(|p| (b.realm, p)))
        .collect()
}

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
    // ★ THE DRAWS ARE INDEXED ONCE (perf fix 2026-08-29). This searched the WHOLE photometrics list
    // for every system's own star. At three systems that was nine comparisons; THE world holds
    // 279 380, so it became 78 BILLION — and this is the function that builds the sky a client
    // receives, not a corner of a test.
    //
    // Fourth of its kind found today, all the same shape: a loop given ONE thing that looks through
    // EVERYTHING to find what the caller already had. The others were the generator's moon pass, a
    // test's body walk, and the boot's interior reach.
    let by_realm: std::collections::BTreeMap<RealmId, &StarPhotometrics> =
        photometrics.iter().map(|(realm, p)| (*realm, p)).collect();
    let mut rows: Vec<vd_core::look::StarRow> = regions
        .iter()
        .filter(|r| matches!(r.realm, RealmId::System(_)) && r.parent == galaxy)
        .map(|r| {
            // A star with no photometric draw still belongs in the sky — it is a place you can fly to.
            // It glows NOT, which the presence floor already says elsewhere: absence of a datum is
            // absence of data, never a default.
            let (class_code, luma_lsun) =
                by_realm.get(&r.realm).map_or((0, 0.0), |p| marker_datum(p));
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
    to_regions(&system_forest_cached(seed_universe, config), config)
}

/// ★ EVERY MOVING BODY OF THE WORLD, FROM ONE BUILD (2026-08-29).
///
/// [`moving_children_for_config`] answers for ONE parent and builds the whole forest to do it. A
/// caller that wants the movers under EVERY parent therefore rebuilds the galaxy once per parent.
///
/// MEASURED: a window-lane test did exactly that. On THE world it is 233 221 parents, each
/// triggering a 3 500 479-body build — the sim's test binary ran for over eleven minutes inside the
/// system placement and had not reached its first assertion.
///
/// The forest already carries every mover. This reads it once.
#[must_use]
pub fn all_movers_for_config(
    seed_universe: u64,
    config: &UniverseConfig,
) -> std::collections::BTreeMap<RealmId, OrbitalElements> {
    generate_system_forest(seed_universe, config)
        .iter()
        .filter_map(|b| super::orbital_of(b.placement).map(|e| (b.realm, e)))
        .collect()
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
    moving_children(&system_forest_cached(seed_universe, config), hosted)
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

#[cfg(test)]
mod built_boot_tests {
    use super::{shard_boot_world_built, shard_boot_world_lit};
    use crate::worldgen::{HOME_SEED, UniverseConfig};
    use vd_core::built::Berth;
    use vd_core::entity_kind::EntityKind;
    use vd_core::fence::Fence;
    use vd_core::geometry::Boundary;
    use vd_core::glam::DVec3;
    use vd_core::ids::EntityId;
    use vd_core::pose::RealmId;
    use vd_core::worldgen::GALAXY;

    fn world() -> UniverseConfig {
        UniverseConfig::world(500.0, 0.02)
    }
    fn held(realm: RealmId) -> std::collections::BTreeSet<RealmId> {
        std::iter::once(realm).collect()
    }
    fn a_ship() -> RealmId {
        RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 1, 0))
    }

    #[test]
    fn with_no_berths_the_world_is_byte_identical_to_the_world_before_this_existed() {
        // ★ THE GATE THAT KEEPS SL5. A realm holding no built children must get exactly the forest it
        // got before a built realm was expressible — not "equivalent", not "close": the same rows.
        //
        // A world variant was added and backed out on 2026-08-31 precisely because two processes could
        // hold different worlds. This assertion is what makes that impossible here: with nothing built,
        // there is nothing to differ about.
        let cfg = world();
        let h = held(RealmId::System(7));
        let plain = shard_boot_world_lit(HOME_SEED, &cfg, &h, RealmId::System(7), &h);
        let built = shard_boot_world_built(HOME_SEED, &cfg, &h, RealmId::System(7), &h, &[], None);
        assert_eq!(plain.0, built.0, "the regions are identical");
        assert_eq!(plain.1, built.1, "the movers are identical");
        assert_eq!(plain.2.len(), built.2.len(), "the lit bodies are identical");
    }

    #[test]
    fn a_berth_becomes_a_realm_with_the_same_band_a_planet_gets() {
        // ★ ONE LOWERING. A built child must get its band, its interest radius and its frame from the
        // code a generated body gets them from — never from a second path that could drift.
        let cfg = world();
        let parent = RealmId::System(7);
        let h = held(parent);
        let berth = Berth {
            child: a_ship(),
            offset_m: DVec3::new(1000.0, 0.0, 0.0),
            bound: Boundary::Shell { r: 20.0 },
            look: Boundary::Shell { r: 20.0 },
            fence: Fence::GENESIS,
        };
        let (regions, _m, _l) =
            shard_boot_world_built(HOME_SEED, &cfg, &h, parent, &h, &[(parent, berth)], None);
        let ship = regions
            .iter()
            .find(|r| r.realm == a_ship())
            .expect("the berth became a realm in this world");
        assert_eq!(
            ship.parent,
            Some(parent),
            "berthed in the realm that authored it"
        );
        assert!(
            ship.aoi.spin_up_r_m() > 0.0,
            "it wakes by the same rule as everything else: {}",
            ship.aoi.spin_up_r_m()
        );
        assert!(
            ship.aoi.tear_down_r_m() > ship.aoi.spin_up_r_m(),
            "and it sleeps further out than it wakes, so it cannot flap"
        );
    }

    #[test]
    fn a_built_child_is_appended_so_nothing_generated_moves() {
        // Adding a hull must not move, renumber or re-draw a single body that was already there. That
        // is the additive discipline the whole record rests on.
        let cfg = world();
        let parent = RealmId::System(7);
        let h = held(parent);
        let (plain, _, _) = shard_boot_world_lit(HOME_SEED, &cfg, &h, parent, &h);
        let berth = Berth {
            child: a_ship(),
            offset_m: DVec3::new(1000.0, 0.0, 0.0),
            bound: Boundary::Shell { r: 20.0 },
            look: Boundary::Shell { r: 20.0 },
            fence: Fence::GENESIS,
        };
        let (with_ship, _, _) =
            shard_boot_world_built(HOME_SEED, &cfg, &h, parent, &h, &[(parent, berth)], None);
        assert_eq!(
            with_ship.len(),
            plain.len() + 1,
            "exactly one region appended"
        );
        let generated: Vec<_> = with_ship.iter().filter(|r| r.realm != a_ship()).collect();
        for (before, after) in plain.iter().zip(generated) {
            assert_eq!(before, after, "every generated body is untouched");
        }
    }
    #[test]
    fn a_built_realm_states_what_it_is_and_gets_the_row_every_realm_has() {
        // ★ THE MEASURED DEFECT (live, 2026-09-01): a ship shard spawned, read its own body, built its
        // world, found NO row for itself and refused — because its berth lives in its PARENT's file and
        // a sealed shard may not read one. Every other realm has this row from the seed. This is the
        // ship getting the same row, from the one record it legitimately holds.
        let cfg = world();
        let ship = a_ship();
        let parent = RealmId::System(7);
        let lineage = std::collections::BTreeSet::from([RealmId::Universe, GALAXY, parent, ship]);
        let held_ship = held(ship);
        let bound = Boundary::Aabb {
            half: DVec3::new(6.0, 3.0, 20.0),
        };

        // Without its own body row the hull is absent — the exact state that refused.
        let (without, _, _) =
            shard_boot_world_built(HOME_SEED, &cfg, &held_ship, ship, &lineage, &[], None);
        assert!(
            !without.iter().any(|r| r.realm == ship),
            "the seed cannot place a realm it did not make"
        );

        // With it, the hull is on its own map, parented where the lineage says.
        let (with, _, _) = shard_boot_world_built(
            HOME_SEED,
            &cfg,
            &held_ship,
            ship,
            &lineage,
            &[],
            Some((parent, bound, bound)),
        );
        let own = with
            .iter()
            .find(|r| r.realm == ship)
            .expect("a built realm is on its own map");
        assert_eq!(
            own.parent,
            Some(parent),
            "parented by the lineage it was sent"
        );
        assert_eq!(own.shape, bound, "its walls are its own body's walls");
        assert_eq!(own.look, Some(bound), "and it draws its own body's outline");
    }

    #[test]
    fn a_built_realms_own_row_states_no_position_in_its_parent() {
        // ★ SL1 CLAUSES 3-5, AS A MEASUREMENT. "Where I sit inside my parent" is the parent's number
        // about me. A child that never HOLDS it cannot state it, which is the law made structural. The
        // parent authors the real placement from tick one; this row carries a zero nobody reads.
        let cfg = world();
        let ship = a_ship();
        let parent = RealmId::System(7);
        let lineage = std::collections::BTreeSet::from([RealmId::Universe, GALAXY, parent, ship]);
        let bound = Boundary::Shell { r: 20.0 };
        let (with, _, _) = shard_boot_world_built(
            HOME_SEED,
            &cfg,
            &held(ship),
            ship,
            &lineage,
            &[],
            Some((parent, bound, bound)),
        );
        let own = with
            .iter()
            .find(|r| r.realm == ship)
            .expect("a built realm is on its own map");
        assert_eq!(
            own.center.in_parents_frame(),
            vd_core::pose::LatticePos::ORIGIN,
            "a child states no position of its own"
        );
    }

    #[test]
    fn a_seed_realm_keeps_the_seeds_row_even_when_a_body_is_offered() {
        // ★ GUARDED ON ABSENCE, NOT ON KIND (SL4/HR3). Offer a body to a realm the generator already
        // placed and nothing changes: the test is "am I missing?", never "what kind am I?". So a planet
        // is untouched and the same code serves a station, an area and a ship.
        let cfg = world();
        let system = RealmId::System(7);
        let h = held(system);
        let bound = Boundary::Shell { r: 20.0 };
        let (plain, _, _) = shard_boot_world_lit(HOME_SEED, &cfg, &h, system, &h);
        let (offered, _, _) = shard_boot_world_built(
            HOME_SEED,
            &cfg,
            &h,
            system,
            &h,
            &[],
            Some((GALAXY, bound, bound)),
        );
        assert_eq!(plain, offered, "the seed's own row wins, byte for byte");
    }
}
