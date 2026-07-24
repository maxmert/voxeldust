//! The `RealmPath` derivation layer (D-45(a) Slice 3a) — a realm's ordered lineage from the
//! Universe root, used by the seed universe generator to derive "who is my parent" and "who
//! are my children" WITHOUT scanning a materialized region Vec.
//!
//! The frozen `RealmId`/`FrameRef`/`StampedPose` wire bytes are untouched. [`RealmKindTag`] is a
//! DERIVATION tag distinct from the frozen `RealmId` enum (it can name `Universe`/`Galaxy`, which
//! `RealmId` cannot until Slice 4 — they map to the interim `System(0)`/`System(1)` stand-ins
//! here). NOTE (RLM Step 1): `RealmKindTag`/`RealmLevel`/`RealmPath` are now serde-derived because
//! [`crate::realm_coord::RealmCoord`] lifts them onto the wire behind the frozen
//! `InterShardFlow::RealmDemand` arm; their postcard discriminants are frozen append-only surfaces.
//! The path is the sole source of parent
//! provenance: [`RealmPath::parent_realm`] closes the "a `Planet` cannot name its `System`"
//! gap and supplies the exact `Area`→`Planet` parent that [`crate::pose::frame_for_realm`]
//! needs — read from the path, not a region scan.
//!
//! P3 uses a fixed [`RealmPathBook`] (the `RealmId`→`RealmPath` inversion over the 7-realm
//! roster). The GENERAL inversion at real scale — a flat `RealmId::Planet(p)` naming its
//! `System` without lineage-in-seed-bits or a seed-forest locator — is the load-bearing owed
//! piece, deferred to P4; do NOT assume this book generalizes for free.

use crate::pose::RealmId;
use serde::{Deserialize, Serialize};

/// The interim `RealmId` stand-in for the Universe root (no dedicated wire arm until Slice 4).
const UNIVERSE_STANDIN: RealmId = RealmId::System(0);
/// The interim `RealmId` stand-in for a Galaxy realm (no dedicated wire arm until Slice 4).
const GALAXY_STANDIN: RealmId = RealmId::System(1);

/// The KIND of a realm in a lineage. `Universe`/`Galaxy` have no `RealmId` arm yet (Slice 4);
/// they resolve to the stand-ins. Now serde-derived + `#[repr(u8)]` with explicit discriminants
/// (frozen APPEND-only order, mirroring [`crate::taxonomy::ProfileKind`]) because it rides the
/// wire via [`crate::realm_coord::RealmCoord`] (RLM Step 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RealmKindTag {
    Universe = 0,
    Galaxy = 1,
    System = 2,
    Planet = 3,
    Station = 4,
    Area = 5,
}

impl RealmKindTag {
    /// Every kind in declaration order = the frozen postcard discriminant order (APPEND-only).
    /// Mirrors [`crate::taxonomy::ProfileKind::ALL`]; the drift tripwire test loops it.
    pub const ALL: [RealmKindTag; 6] = [
        RealmKindTag::Universe,
        RealmKindTag::Galaxy,
        RealmKindTag::System,
        RealmKindTag::Planet,
        RealmKindTag::Station,
        RealmKindTag::Area,
    ];
}

/// One level of a lineage: its kind + its deterministic seed. The `seed` doubles as the
/// `RealmId` payload for the keyed kinds (System/Planet/Station/Area) and as the RNG lineage
/// seed fed to [`crate::rng::realm_stream`]; `Universe`/`Galaxy` resolve to the stand-ins
/// regardless of `seed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealmLevel {
    pub kind: RealmKindTag,
    pub seed: u64,
}

impl RealmLevel {
    #[must_use]
    pub fn new(kind: RealmKindTag, seed: u64) -> RealmLevel {
        RealmLevel { kind, seed }
    }

    /// The `RealmId` this level resolves to. Total over the 6 kinds; `Universe`/`Galaxy` use
    /// the interim stand-ins (Slice 4 flips them to dedicated arms).
    #[must_use]
    pub fn to_realm_id(self) -> RealmId {
        match self.kind {
            RealmKindTag::Universe => UNIVERSE_STANDIN,
            RealmKindTag::Galaxy => GALAXY_STANDIN,
            RealmKindTag::System => RealmId::System(self.seed),
            RealmKindTag::Planet => RealmId::Planet(self.seed),
            RealmKindTag::Station => RealmId::Station(self.seed),
            RealmKindTag::Area => RealmId::Area(self.seed),
        }
    }
}

/// A realm's ordered lineage, root → leaf (e.g. `[Universe, Galaxy(g), System(s), Planet(p),
/// Area(a)]`). Off the wire; the derivation-layer source of parent/child structure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealmPath(Vec<RealmLevel>);

impl RealmPath {
    #[must_use]
    pub fn from_levels(levels: Vec<RealmLevel>) -> RealmPath {
        RealmPath(levels)
    }

    #[must_use]
    pub fn levels(&self) -> &[RealmLevel] {
        &self.0
    }

    /// This realm's own `RealmId` (the last level), or `None` for an empty path.
    #[must_use]
    pub fn realm_id(&self) -> Option<RealmId> {
        self.0.last().map(|level| level.to_realm_id())
    }

    /// The parent realm's `RealmId` (the second-last level), or `None` at the root. This is
    /// the sole parent-provenance source — e.g. an `Area` path yields `Some(Planet(..))`, the
    /// exact parent [`crate::pose::frame_for_realm`] requires to form an `AreaLocal` frame.
    #[must_use]
    pub fn parent_realm(&self) -> Option<RealmId> {
        let n = self.0.len();
        if n >= 2 {
            Some(self.0[n - 2].to_realm_id())
        } else {
            None
        }
    }

    /// The lineage seeds root → leaf, fed to [`crate::rng::realm_stream`] for a per-realm
    /// deterministic stream (bit-reproducible across shards by construction, HR1).
    #[must_use]
    pub fn lineage_seeds(&self) -> Vec<u64> {
        self.0.iter().map(|level| level.seed).collect()
    }
}

// ===== The P3 fixed RealmPathBook (the RealmId->RealmPath inversion over the roster) =====

// The 7-realm walk roster seeds (the RealmId payloads). Universe/Galaxy carry their stand-in
// seeds so lineage_seeds() is deterministic; the keyed realms carry seed 7 (or 8 for System B).
const UNIVERSE_SEED: u64 = 0;
const GALAXY_SEED: u64 = 1;
const SYSTEM_A_SEED: u64 = 7;
const SYSTEM_B_SEED: u64 = 8;
const PLANET_A_SEED: u64 = 7;
const STATION_A_SEED: u64 = 7;
const AREA_A_SEED: u64 = 7;

fn universe_path() -> RealmPath {
    RealmPath::from_levels(vec![RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED)])
}

fn galaxy_path() -> RealmPath {
    RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED),
        RealmLevel::new(RealmKindTag::Galaxy, GALAXY_SEED),
    ])
}

fn system_path(seed: u64) -> RealmPath {
    RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED),
        RealmLevel::new(RealmKindTag::Galaxy, GALAXY_SEED),
        RealmLevel::new(RealmKindTag::System, seed),
    ])
}

fn planet_path() -> RealmPath {
    RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED),
        RealmLevel::new(RealmKindTag::Galaxy, GALAXY_SEED),
        RealmLevel::new(RealmKindTag::System, SYSTEM_A_SEED),
        RealmLevel::new(RealmKindTag::Planet, PLANET_A_SEED),
    ])
}

fn station_path() -> RealmPath {
    RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED),
        RealmLevel::new(RealmKindTag::Galaxy, GALAXY_SEED),
        RealmLevel::new(RealmKindTag::System, SYSTEM_A_SEED),
        RealmLevel::new(RealmKindTag::Station, STATION_A_SEED),
    ])
}

fn area_path() -> RealmPath {
    RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, UNIVERSE_SEED),
        RealmLevel::new(RealmKindTag::Galaxy, GALAXY_SEED),
        RealmLevel::new(RealmKindTag::System, SYSTEM_A_SEED),
        RealmLevel::new(RealmKindTag::Planet, PLANET_A_SEED),
        RealmLevel::new(RealmKindTag::Area, AREA_A_SEED),
    ])
}

/// The P3 fixed `RealmId`→`RealmPath` inversion over the walk roster. `None` for any realm not
/// in the roster (loud — never a wrong-lineage guess). The real-scale general inversion is owed
/// to P4 (see the module docs); this fixed book is the "real-enough" P3 stand-in.
#[must_use]
pub fn path_for_realm(realm: RealmId) -> Option<RealmPath> {
    match realm {
        RealmId::System(UNIVERSE_SEED) => Some(universe_path()),
        RealmId::System(GALAXY_SEED) => Some(galaxy_path()),
        RealmId::System(SYSTEM_A_SEED) => Some(system_path(SYSTEM_A_SEED)),
        RealmId::System(SYSTEM_B_SEED) => Some(system_path(SYSTEM_B_SEED)),
        RealmId::Planet(PLANET_A_SEED) => Some(planet_path()),
        RealmId::Station(STATION_A_SEED) => Some(station_path()),
        RealmId::Area(AREA_A_SEED) => Some(area_path()),
        _ => None,
    }
}

/// The realms the P3 [`RealmPathBook`](path_for_realm) knows, in forest order (Universe root →
/// Galaxy → System A / System B → Planet A / Station A → Area A). The generator iterates this
/// to build the walk-scale forest.
pub const ROSTER: [RealmId; 7] = [
    UNIVERSE_STANDIN,
    GALAXY_STANDIN,
    RealmId::System(SYSTEM_A_SEED),
    RealmId::System(SYSTEM_B_SEED),
    RealmId::Planet(PLANET_A_SEED),
    RealmId::Station(STATION_A_SEED),
    RealmId::Area(AREA_A_SEED),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{FrameRef, frame_for_realm};

    #[test]
    fn to_realm_id_covers_every_kind() {
        // Every RealmKindTag resolves to the right RealmId (Universe/Galaxy -> stand-ins).
        assert_eq!(
            RealmLevel::new(RealmKindTag::Universe, 99).to_realm_id(),
            RealmId::System(0)
        );
        assert_eq!(
            RealmLevel::new(RealmKindTag::Galaxy, 99).to_realm_id(),
            RealmId::System(1)
        );
        assert_eq!(
            RealmLevel::new(RealmKindTag::System, 7).to_realm_id(),
            RealmId::System(7)
        );
        assert_eq!(
            RealmLevel::new(RealmKindTag::Planet, 7).to_realm_id(),
            RealmId::Planet(7)
        );
        assert_eq!(
            RealmLevel::new(RealmKindTag::Station, 7).to_realm_id(),
            RealmId::Station(7)
        );
        assert_eq!(
            RealmLevel::new(RealmKindTag::Area, 7).to_realm_id(),
            RealmId::Area(7)
        );
    }

    #[test]
    fn realm_id_reads_the_last_level_and_empty_is_none() {
        let area = area_path();
        assert_eq!(area.realm_id(), Some(RealmId::Area(7)));
        assert_eq!(system_path(8).realm_id(), Some(RealmId::System(8)));
        // The empty path has no realm (the None arm).
        assert_eq!(RealmPath::from_levels(vec![]).realm_id(), None);
        // levels() exposes the ordered lineage (Slice 3c iterates this for ancestor gen).
        assert_eq!(area.levels().len(), 5);
        assert_eq!(area.levels()[0].kind, RealmKindTag::Universe);
        assert_eq!(area.levels()[4].kind, RealmKindTag::Area);
    }

    #[test]
    fn parent_realm_is_pinned_for_every_roster_realm() {
        // The full parent topology — a re-parent anywhere fails loud here.
        assert_eq!(universe_path().parent_realm(), None); // root
        assert_eq!(galaxy_path().parent_realm(), Some(RealmId::System(0))); // -> Universe standin
        assert_eq!(system_path(7).parent_realm(), Some(RealmId::System(1))); // -> Galaxy standin
        assert_eq!(system_path(8).parent_realm(), Some(RealmId::System(1))); // sibling of System 7
        assert_eq!(planet_path().parent_realm(), Some(RealmId::System(7)));
        assert_eq!(station_path().parent_realm(), Some(RealmId::System(7)));
        assert_eq!(area_path().parent_realm(), Some(RealmId::Planet(7)));
    }

    #[test]
    fn area_parent_provenance_forms_the_area_frame_end_to_end() {
        // The exact gap RealmPath closes: the Area's parent (Planet 7) drives frame_for_realm
        // to an AreaLocal frame — provenance from the PATH, not a region scan.
        let area = area_path();
        let parent = area.parent_realm();
        assert_eq!(parent, Some(RealmId::Planet(7)));
        assert_eq!(
            frame_for_realm(RealmId::Area(7), parent),
            Some(FrameRef::AreaLocal {
                planet_seed: 7,
                area_seed: 7
            })
        );
    }

    #[test]
    fn lineage_seeds_are_root_to_leaf() {
        assert_eq!(area_path().lineage_seeds(), vec![0, 1, 7, 7, 7]);
        assert_eq!(system_path(8).lineage_seeds(), vec![0, 1, 8]);
        assert_eq!(
            RealmPath::from_levels(vec![]).lineage_seeds(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn path_book_covers_exactly_the_roster_and_absent_is_none() {
        // Every roster realm resolves to a path whose realm_id() is itself.
        for realm in ROSTER {
            let path = path_for_realm(realm).expect("roster realm has a path");
            assert_eq!(path.realm_id(), Some(realm), "{realm:?} path round-trips");
        }
        assert_eq!(ROSTER.len(), 7);
        // A non-roster realm is loud None (never a wrong-lineage guess).
        assert_eq!(path_for_realm(RealmId::System(999)), None);
        assert_eq!(path_for_realm(RealmId::Planet(999)), None);
    }

    #[test]
    fn realm_kind_tag_all_is_exhaustive() {
        // Drift tripwire: a 7th kind must be added to ALL (this match then fails to compile).
        for k in RealmKindTag::ALL {
            match k {
                RealmKindTag::Universe
                | RealmKindTag::Galaxy
                | RealmKindTag::System
                | RealmKindTag::Planet
                | RealmKindTag::Station
                | RealmKindTag::Area => {}
            }
        }
        assert_eq!(RealmKindTag::ALL.len(), 6);
    }

    #[test]
    fn realm_kind_tag_serde_discriminants_frozen() {
        // Each unit variant's postcard byte IS its append-only discriminant 0..5 — a wire freeze
        // (RealmCoord lifts RealmKindTag onto the RealmDemand arm); `as u8` pins the repr too.
        let expected: [(RealmKindTag, u8); 6] = [
            (RealmKindTag::Universe, 0),
            (RealmKindTag::Galaxy, 1),
            (RealmKindTag::System, 2),
            (RealmKindTag::Planet, 3),
            (RealmKindTag::Station, 4),
            (RealmKindTag::Area, 5),
        ];
        for (kind, disc) in expected {
            assert_eq!(kind as u8, disc, "{kind:?} repr discriminant");
            assert_eq!(
                postcard::to_allocvec(&kind).expect("encode"),
                vec![disc],
                "{kind:?} postcard discriminant byte"
            );
        }
    }
}
