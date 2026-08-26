//! Realm TOPOLOGY over an already-built region forest — the half of the old worldgen that a sealed
//! shard may hold (the placement arc S5). The split rule, stated once (SL5's guard): anything that
//! READS THE SEED STREAM or MINTS A BODY lives in `vd-physics`; anything here only reads an
//! already-built `&[RealmRegion]` (or converts pure identities/durations). There is still exactly ONE
//! generator — `vd_physics::worldgen` — and what lives here are topology utilities over its output,
//! not a second one.
//!
//! What this module deliberately CANNOT do: name an orbit. `vd-core` carries no edge to the motion
//! crate (`tests/tests/crate_isolation.rs` holds the law), so the crossing/containment path that
//! lives beside this module reads authored placement rows and nothing else (SL4).

use crate::geometry::RealmRegion;
use crate::pose::RealmId;
use crate::realm_coord::RealmCoord;
use crate::realm_path::{RealmKindTag, RealmLevel, RealmPath};

/// The `RealmId` seed PAYLOADS of the ambient lineage above System A + System A itself — declared as
/// `u64` (the RNG lineage the per-system stream seeds from). Since S9 the universe and the galaxy carry
/// their own `RealmId` arms, so these are the SEEDS only — the identities no longer borrow a `System`.
pub const UNIVERSE_SEED: u64 = 0;
pub const GALAXY_SEED: u64 = 1;
pub const SYSTEM_A_SEED: u64 = 7;

/// ★ THE AMBIENT REALMS, NAMING THEMSELVES SINCE S9. These were the placeholders — `System(0)` and
/// `System(1)` — for the two levels that had no `RealmId` arm of their own. Every forest in the program is
/// built from them, so the whole world inherited the borrowed names, and a lineage came out reading
/// "system 0 contains system 1 contains system 7" when it meant "the universe contains a galaxy contains
/// a star system".
///
/// The seeds above are kept: [`UNIVERSE_SEED`] and [`GALAXY_SEED`] still key the RNG lineage the per-system
/// streams draw from, so no draw moves. What changes is only what the level is CALLED — and a star system
/// whose seed happens to be 0 is now a star system rather than the universe.
pub const UNIVERSE: RealmId = RealmId::Universe;
pub const GALAXY: RealmId = RealmId::Galaxy(GALAXY_SEED);
pub const SYSTEM_A: RealmId = RealmId::System(SYSTEM_A_SEED);
pub const SYSTEM_B: RealmId = RealmId::System(8);
pub const PLANET_A: RealmId = RealmId::Planet(7);
/// Station A — a first-class Station realm nested directly under System A (task #133).
pub const STATION_A: RealmId = RealmId::Station(7);
/// Area A — a first-class sub-planet Area realm nested under Planet A (task #133).
pub const AREA_A: RealmId = RealmId::Area(7);

// (`MAX_RENDERABLE_EXTENT_M` is DELETED — real-scale design §3.0 consequence 1, landed with the
// in-system re-solve: "is this drawable" is LOOK PRESENCE plus the angular rule, never a
// magnitude threshold that silently skips. A realm with `RealmRegion.look = None` ships no
// self-look and no marker outline — the ambient Universe/Galaxy are undrawable structurally.)

// (`children_within` + its `aoi_within` predicate are DELETED — real-scale addendum §A4.5: the
// one explicit same-cell assumption in the tree, "occupant and child MUST share a cell through P3".
// The cell activation made that premise false; its future consumer (the P6/D-9 spatial index) is
// built on `Separation`, not on a bare offset subtraction.)

/// A SEED-LINEAGE `RealmId` → its `RealmLevel` (kind + seed), un-lossily: every kind now carries its own
/// identity, so this reads a level rather than recognising a borrowed seed. `None` for [`RealmId::Ship`] — a ship is ENTITY-backed (P8, an `EntityId`
/// payload), NOT a seed-lineage realm, so it has no seed `RealmLevel` (and never appears in a seed
/// forest / P3 region). Monomorphic (HR5: the kind match covered once here).
///
/// ★ IT NO LONGER ALIASES. It used to recover the universe and the galaxy from `System(0)` and
/// `System(1)`, and said so: "a real system with seed 0/1 would alias, but the walk/visual forest uses
/// 7/8." That was a naming convention standing in for a type, and it held only while no seeded world drew
/// a 0 or a 1.
#[must_use]
pub fn level_of(realm: RealmId) -> Option<RealmLevel> {
    match realm {
        // ★ EXACT SINCE S9, WHERE IT USED TO ALIAS. The universe and the galaxy have their own
        // identities now, so recovering their level is reading it rather than recognising a borrowed
        // seed. The two arms below used to be `System(0)` and `System(1)`, and this function's own doc
        // admitted the hole in as many words: "a real system with seed 0/1 would alias, but the
        // walk/visual forest uses 7/8." It aliased on a naming convention, and the convention held only
        // because no seeded world had yet drawn a 0 or a 1.
        RealmId::Universe => Some(RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED)),
        RealmId::Galaxy(s) => Some(RealmLevel::new(RealmKindTag::Galaxy, s)),
        RealmId::System(s) => Some(RealmLevel::new(RealmKindTag::System, s)),
        RealmId::Planet(s) => Some(RealmLevel::new(RealmKindTag::Planet, s)),
        RealmId::Station(s) => Some(RealmLevel::new(RealmKindTag::Station, s)),
        RealmId::Area(s) => Some(RealmLevel::new(RealmKindTag::Area, s)),
        RealmId::Star(s) => Some(RealmLevel::new(RealmKindTag::Star, s)),
        RealmId::Ship(_) => None,
    }
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

/// THE REALM AN ACCOUNT APPEARS IN until something says otherwise: the ambient root's first child, and
/// that child's first child — a star system.
///
/// A NAME, chosen by lineage position, and nothing is measured to choose it. That is the whole reason this
/// is allowed to exist in a world the router holds: naming a realm leaks nothing about where anything is,
/// and the pose that goes with the name is authored in the named realm's OWN frame by
/// [`crate::home::StoredHome::in_realm`], which reads no distance either.
///
/// The fallback pose that pairs with it is the realm's own centre — a home needs no magic offset, and in
/// this world the nearest child's surface sits about twelve metres from that centre, so the origin of a
/// star system is empty space by construction rather than by a chosen number.
///
/// `None` for a forest with no root, or a root with no grandchild — stated rather than substituted, so a
/// cluster booted with a degenerate world fails where it can be seen instead of putting every account
/// somewhere arbitrary.
#[must_use]
pub fn default_home_realm(regions: &[RealmRegion]) -> Option<RealmId> {
    let root = regions.iter().find(|r| r.parent.is_none())?;
    let galaxy = regions.iter().find(|r| r.parent == Some(root.realm))?;
    let system = regions.iter().find(|r| r.parent == Some(galaxy.realm))?;
    Some(system.realm)
}

/// The ancestors-union-direct-children ("never siblings") filter over an ALREADY-BUILT forest — the shared
/// core of both neighbourhood builders (HR3, DRY). A realm is IN-SCOPE iff it is an ancestor of, or a direct
/// child of, ANY held realm. Collect the qualifying realm set first (deduped), then filter the forest ONCE so
/// the output keeps forest order (deterministic boot depth-key/guard results).
pub fn neighbourhood_scope(
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
pub fn ancestor_realms(all: &[RealmRegion], hosted_realm: RealmId) -> Vec<RealmId> {
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

// ===== The render-origin PIN classifier =============================================================
//
/// THE LOWERED WORLD — the containment forest held as a value, answering exactly the questions the
/// connection plane asks of a world: what regions exist, which of them a holder evaluates, and
/// whether a named realm is real. It is `vd_physics::worldgen::WorldView` AFTER lowering — the
/// bodies (and every orbit element) do not survive into it, which is what lets the gateway hold a
/// world without a dependency edge to the motion crate (SL4: the routing plane is a placement
/// CONSUMER; the batch review found the crate-fence carved open for it). Built by the composition
/// root from a `WorldView` (`WorldView::lowered`); never constructed from a seed here — vd-core
/// cannot mint a body (the split rule above).
#[derive(Clone, Debug, PartialEq)]
pub struct WorldRealms {
    regions: Vec<RealmRegion>,
}

impl WorldRealms {
    /// Hold an already-lowered forest. The caller (the composition root) is the party that decided
    /// which world this is; nothing here can generate one.
    #[must_use]
    pub fn new(regions: Vec<RealmRegion>) -> WorldRealms {
        WorldRealms { regions }
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

    /// Is this realm part of this world? (The honest form of the check that once consulted a
    /// DIFFERENT world than the one that produced the answer being checked.)
    #[must_use]
    pub fn contains_realm(&self, realm: RealmId) -> bool {
        self.regions.iter().any(|r| r.realm == realm)
    }
}

// What used to sit here was the seed-derived ORIGIN CHAIN: a realm's shard folded its whole ancestor
// chain up to the universe root to work out its own absolute position, and shipped every occupant at
// that absolute. It is gone, and it must not come back under another name.
//
// Two reasons it had to go. It broke the ground rule — a realm is centred on ITSELF and is never told
// where it sits; only its parent knows that, and the conversion belongs to the parent. And it destroyed
// precision: a planet's surface position became the difference of two universe-scale doubles, so the
// millimetres a player walks in were below the representable step long before they reached anyone. The
// conversion now happens SHARD TO SHARD, one level at a time: each parent applies the single placement IT
// authored for the child on the path and hands the result on, up or down, so no quantity on the path ever
// exceeds the scale of the level applying it and nobody ever walks to the root. (A second attempt put that
// walk in the gateway instead, over a copy of the whole forest. Same breach, one hop further out: a router
// is the parent of nothing. It was deleted.)
//
// What survives is the PIN: which realm a session's whole scene is measured from. That is an identity,
// not a position, so it stays here.

/// Is this realm a STAR-SYSTEM-level container? Today (P3) `RealmId::System` stands in for Universe, Galaxy,
/// AND star systems alike (dedicated `Universe`/`Galaxy` arms land at P4+, D-44); a `Planet`/`Station`/`Area`/
/// `Ship` is never system-level. The one monomorphic discriminator [`pin_realm_of`] branches over (HR5) — its
/// two arms covered by a `System` and a non-`System` realm.
#[must_use]
fn is_system_level(realm: RealmId) -> bool {
    matches!(realm, RealmId::System(_))
}

/// The render-origin realm for a session whose ancestor chain (root→realm order) is `chain_realms`: the
/// player's OWN star system — the DEEPEST system-level ancestor. A `Planet`/`Station`/`Area`/`Ship` is never
/// system-level, so the deepest `System` is the innermost star-system container (the star directly holding the
/// occupant), and since the ambient root is itself system-level a root-only chain folds to the root (the
/// documented fall-back). The ONE centralized pin classifier — no caller writes an inline system-kind test
/// (HR3). The client subtracts this realm's absolute so the render origin sits at the player's own star, NOT
/// the galaxy (a renderable ancestor at visual scale whose selection would push the whole inter-system offset
/// into the f64 residual and defeat the floating origin — verifier FINDING G).
#[must_use]
pub fn pin_realm_of(chain_realms: &[RealmId]) -> RealmId {
    chain_realms
        .iter()
        .rev()
        .copied()
        .find(|&r| is_system_level(r))
        .or_else(|| chain_realms.first().copied())
        .unwrap_or(UNIVERSE)
}

// ===== UniverseConfig (D-45(a) Slice 3b) — the ONE config home ==========================
//
// The ~15 placeholder consts above become NAMED fields of six sub-structs (NOT a god-struct).
// `walk_scale()` reproduces today's EXACT metre-scale geometry (the byte-identity source).
// (The `canonical()`/`seed_derived()` real-scale presets are DELETED — SL5, Stage-C audit :866:
// zero production callers; true astronomical scale is an owed change to THE ONE world, never a
// parallel preset.)

/// The loiter grace as a DURATION (seconds) — converted to ticks against the live `tick_dt_s` at boot
/// ([`grace_ticks_from_seconds`]), so it is correct at any tick rate. The SINGLE loiter-duration constant in
/// the codebase: the region AoI `grace_ticks` derives from it here, and (VU AoI S2b) the parent's retained-
/// occupant TTL reuses it, so the two are consistent by construction and neither is ever a magic number.
pub const WALK_DEMAND_AOI_GRACE_S: f64 = 1.0;
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

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
    fn level_of_covers_every_kind() {
        // ★ THE ALIAS IS GONE, AND THESE TWO ASSERTIONS ARE THE PROOF (slice S9). They used to say that
        // `System(0)` WAS the universe and `System(1)` WAS a galaxy — borrowed names this function
        // recognised by their seed. Its own doc admitted the hole: "a real system with seed 0/1 would
        // alias, but the walk/visual forest uses 7/8". A convention standing in for a type.
        //
        // A star system whose seed is 0 is now a star system whose seed is 0. Nothing else.
        assert_eq!(
            level_of(RealmId::System(0)),
            Some(RealmLevel::new(RealmKindTag::System, 0))
        );
        assert_eq!(
            level_of(RealmId::System(1)),
            Some(RealmLevel::new(RealmKindTag::System, 1))
        );
        // …and the two that DO mean the universe and a galaxy say so themselves.
        assert_eq!(
            level_of(RealmId::Universe),
            Some(RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED))
        );
        assert_eq!(
            level_of(RealmId::Galaxy(4)),
            Some(RealmLevel::new(RealmKindTag::Galaxy, 4))
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

    #[test]
    fn a_lowered_world_answers_the_connection_planes_three_questions() {
        // WorldRealms: regions() is the held forest verbatim; neighbourhood() is the shared
        // ancestors-∪-direct-children scope; contains_realm() is a membership test with both
        // verdicts exercised (the `any` closure's true AND false arms — HR5).
        use crate::geometry::{AoiConfig, Boundary, ContainmentBand, RealmRegion};
        use crate::pose::{FrameRef, LatticePos};
        let region = |realm: RealmId, parent: Option<RealmId>| RealmRegion {
            realm,
            center: crate::geometry::ParentCentre::authored(LatticePos::local(DVec3::ZERO)),
            frame: FrameRef::SystemSpace { system_seed: 0 },
            shape: Boundary::Shell { r: 1.0 },
            look: Some(Boundary::Shell { r: 1.0 }),
            band: ContainmentBand::for_containment_velocity_safe(1.0, 2.0, 0.0, 1.0, 0.0)
                .expect("valid test band"),
            aoi: AoiConfig::inert(),
            parent,
            interior_band: AoiConfig::inert(),
        };
        let forest = vec![
            region(UNIVERSE, None),
            region(GALAXY, Some(UNIVERSE)),
            region(SYSTEM_A, Some(GALAXY)),
            region(SYSTEM_B, Some(GALAXY)),
        ];
        let world = WorldRealms::new(forest.clone());
        assert_eq!(world.regions(), forest.as_slice());
        // Ancestors ∪ direct children of the held realm — never its sibling.
        let held = std::collections::BTreeSet::from([SYSTEM_A]);
        let scoped: Vec<RealmId> = world.neighbourhood(&held).iter().map(|r| r.realm).collect();
        assert_eq!(scoped, vec![UNIVERSE, GALAXY, SYSTEM_A]);
        assert!(world.contains_realm(SYSTEM_B));
        assert!(!world.contains_realm(PLANET_A));
    }

    #[test]
    fn pin_realm_of_selects_the_system_and_falls_back_to_the_root() {
        // A5 render-origin classifier: the DEEPEST system-level ancestor is the player's own star system; a
        // Planet is never system-level (so both `is_system_level` arms are exercised inside one chain).
        // Realistic root→realm chain (Universe, Galaxy, System A, Planet) ⇒ the deepest System (System A) wins.
        assert_eq!(
            pin_realm_of(&[UNIVERSE, GALAXY, SYSTEM_A, RealmId::Planet(42)]),
            SYSTEM_A
        );
        // A root-only chain folds to the root (itself system-level).
        assert_eq!(pin_realm_of(&[UNIVERSE]), UNIVERSE);
        // Defensive: a chain with NO system-level realm falls back to its first entry (find None → first).
        assert_eq!(
            pin_realm_of(&[RealmId::Planet(1), RealmId::Planet(2)]),
            RealmId::Planet(1)
        );
        // Defensive: an empty chain folds to the ambient root (first None → UNIVERSE).
        assert_eq!(pin_realm_of(&[]), UNIVERSE);
    }
}
