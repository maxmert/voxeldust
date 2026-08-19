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
// `Ord`/`PartialOrd` (RLM Step 2): `RealmPath` is a `BTreeMap`/`BTreeSet` key in the AoI ledger, so the
// whole lineage type stack is totally ordered. A PURE ADDITIVE derive — ordering is NOT serialized, so
// the frozen postcard bytes are untouched; `#[repr(u8)]` explicit discriminants make the kind order
// (Universe < Galaxy < … < Area) match declaration order deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum RealmKindTag {
    Universe = 0,
    Galaxy = 1,
    System = 2,
    Planet = 3,
    Station = 4,
    Area = 5,
    /// A star inside its system (taxonomy arc T2). APPENDED at 6 — inserting it before
    /// `Planet` for lineage aesthetics would RENUMBER the wire and is forbidden (the frozen
    /// APPEND-only order).
    Star = 6,
}

impl RealmKindTag {
    /// Every kind in declaration order = the frozen postcard discriminant order (APPEND-only).
    /// Mirrors [`crate::taxonomy::ProfileKind::ALL`]; the drift tripwire test loops it.
    pub const ALL: [RealmKindTag; 7] = [
        RealmKindTag::Universe,
        RealmKindTag::Galaxy,
        RealmKindTag::System,
        RealmKindTag::Planet,
        RealmKindTag::Station,
        RealmKindTag::Area,
        RealmKindTag::Star,
    ];
}

/// One level of a lineage: its kind + its deterministic seed. The `seed` doubles as the
/// `RealmId` payload for the keyed kinds (System/Planet/Station/Area) and as the RNG lineage
/// seed fed to [`crate::rng::realm_stream`]; `Universe`/`Galaxy` resolve to the stand-ins
/// regardless of `seed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
            RealmKindTag::Star => RealmId::Star(self.seed),
        }
    }
}

/// A realm's ordered lineage, root → leaf (e.g. `[Universe, Galaxy(g), System(s), Planet(p),
/// Area(a)]`). Off the wire; the derivation-layer source of parent/child structure.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

    /// Encode this lineage as a compact env-var string — the `VD_OWN_COORD` transport the real realm
    /// spawner (RLM Step 5) hands a freshly-launched shard so it recovers its UN-collapsed lineage (a
    /// `Galaxy`/`Universe` level is otherwise lost through `RealmId`, dropping `signal_relay`). Lowercase
    /// hex of the FROZEN postcard bytes: it can never drift from the wire form and covers EVERY level
    /// kind with no per-kind vocabulary to maintain (generic by construction). Round-trips exactly via
    /// [`RealmPath::from_env_string`].
    #[must_use]
    pub fn to_env_string(&self) -> String {
        let bytes = postcard::to_allocvec(self).expect("postcard encodes a RealmPath");
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in &bytes {
            s.push(HEX_LOWER[(b >> 4) as usize]);
            s.push(HEX_LOWER[(b & 0x0f) as usize]);
        }
        s
    }

    /// Decode a [`to_env_string`](RealmPath::to_env_string) form. FAILS LOUD on any malformation — a
    /// boot misconfiguration must never silently mis-place a shard (it would author the wrong realm's
    /// frames). Monomorphic body (all branching here, HR5).
    ///
    /// # Errors
    /// [`RealmPathEnvError`] on an odd-length string, a non-hex digit, or bytes that do not decode to a
    /// `RealmPath`.
    pub fn from_env_string(s: &str) -> Result<RealmPath, RealmPathEnvError> {
        let raw = s.as_bytes();
        if !raw.len().is_multiple_of(2) {
            return Err(RealmPathEnvError::OddLength(raw.len()));
        }
        let mut bytes = Vec::with_capacity(raw.len() / 2);
        let mut i = 0;
        while i < raw.len() {
            let hi = hex_digit(raw[i]).ok_or(RealmPathEnvError::NotHex)?;
            let lo = hex_digit(raw[i + 1]).ok_or(RealmPathEnvError::NotHex)?;
            bytes.push((hi << 4) | lo);
            i += 2;
        }
        postcard::from_bytes(&bytes).map_err(|_| RealmPathEnvError::Malformed)
    }
}

/// Lowercase hex alphabet for [`RealmPath::to_env_string`] (a table, so the encode has no `write!`
/// `Result` arm to leave uncovered — HR5). `pub(crate)` so [`crate::incarnation`] shares the ONE hex
/// vocabulary (DRY) rather than re-declaring it.
pub(crate) const HEX_LOWER: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// One lowercase-hex digit → its nibble. Accepts ONLY what [`RealmPath::to_env_string`] emits (`0-9`,
/// `a-f`); anything else is `None` (rejected LOUD). Monomorphic (every arm covered, HR5). `pub(crate)`
/// so [`crate::incarnation`] shares it (DRY).
pub(crate) fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

/// A malformed `VD_OWN_COORD` — rejected LOUD at boot so a shard never silently authors the wrong realm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RealmPathEnvError {
    /// The hex string has an odd number of characters (not whole bytes).
    #[error("VD_OWN_COORD hex has an odd length ({0})")]
    OddLength(usize),
    /// A character outside the lowercase-hex alphabet `[0-9a-f]`.
    #[error("VD_OWN_COORD contains a non-hex character")]
    NotHex,
    /// Well-formed hex whose bytes do not decode to a `RealmPath`.
    #[error("VD_OWN_COORD bytes do not decode to a RealmPath")]
    Malformed,
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
    fn env_string_round_trips_every_lineage_including_galaxy() {
        // Every depth round-trips EXACTLY — crucially `galaxy_path` (a `Galaxy` level would be lost
        // through `RealmId`, dropping `signal_relay`); that preservation is the whole reason for
        // `VD_OWN_COORD`. The paths' hex spans both `0-9` and `a-f` nibbles (both `hex_digit` valid arms).
        for p in [
            universe_path(),
            galaxy_path(),
            system_path(7),
            planet_path(),
            area_path(),
            RealmPath::from_levels(vec![]), // the empty boundary encodes to "00" (postcard len-0)
        ] {
            // The round-trip IS the format proof: `from_env_string` accepts ONLY lowercase hex, so a
            // successful decode back to `p` guarantees `to_env_string` emitted valid lowercase hex.
            let enc = p.to_env_string();
            assert_eq!(RealmPath::from_env_string(&enc), Ok(p), "round-trip {enc}");
        }
    }

    #[test]
    fn env_string_rejects_malformed_input_loud() {
        // Odd length ⇒ not whole bytes.
        assert_eq!(
            RealmPath::from_env_string("abc"),
            Err(RealmPathEnvError::OddLength(3))
        );
        // A non-hex character ⇒ the `hex_digit` reject arm (uppercase is NOT accepted either).
        assert_eq!(
            RealmPath::from_env_string("zz"),
            Err(RealmPathEnvError::NotHex)
        );
        assert_eq!(
            RealmPath::from_env_string("AB"),
            Err(RealmPathEnvError::NotHex)
        );
        // First nibble VALID, second INVALID ⇒ the LOW-nibble reject (a region distinct from the high one).
        assert_eq!(
            RealmPath::from_env_string("az"),
            Err(RealmPathEnvError::NotHex)
        );
        // Well-formed hex whose bytes are NOT a RealmPath (a truncated Vec-length varint) + the empty
        // string (no length byte at all).
        assert_eq!(
            RealmPath::from_env_string("ff"),
            Err(RealmPathEnvError::Malformed)
        );
        assert_eq!(
            RealmPath::from_env_string(""),
            Err(RealmPathEnvError::Malformed)
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
        // Drift tripwire: an 8th kind must be added to ALL (this match then fails to compile).
        for k in RealmKindTag::ALL {
            match k {
                RealmKindTag::Universe
                | RealmKindTag::Galaxy
                | RealmKindTag::System
                | RealmKindTag::Planet
                | RealmKindTag::Station
                | RealmKindTag::Area
                | RealmKindTag::Star => {}
            }
        }
        assert_eq!(RealmKindTag::ALL.len(), 7);
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

    #[test]
    fn realm_path_is_totally_ordered_by_lineage() {
        // RLM Step 2: `RealmPath` is a `BTreeMap`/`BTreeSet` key in the AoI ledger, so the whole type stack
        // is `Ord`. Kind order follows the `#[repr(u8)]` declaration (Universe < Galaxy < … < Area), then
        // seed, then path length (a prefix sorts before its extension) — a deterministic total order.
        let a = RealmLevel::new(RealmKindTag::Universe, 0);
        let b = RealmLevel::new(RealmKindTag::Galaxy, 0);
        assert!(a < b, "kind order follows the repr discriminant");
        let sys7 = system_path(7);
        let sys8 = system_path(8);
        assert!(sys7 < sys8, "same lineage, greater leaf seed sorts later");
        assert!(galaxy_path() < sys7, "a prefix sorts before its extension");
        // A BTreeSet round-trips the ordering (the actual AoI use).
        let mut set = std::collections::BTreeSet::new();
        set.insert(sys8.clone());
        set.insert(galaxy_path());
        set.insert(sys7.clone());
        let ordered: Vec<RealmPath> = set.into_iter().collect();
        assert_eq!(ordered, vec![galaxy_path(), sys7, sys8]);
    }

    /// The STAR tag's own `to_realm_id` arm (T2): a path level tagged `Star` names a
    /// `RealmId::Star` of the same seed — the coord↔id round trip a demand for a star rides.
    #[test]
    fn a_star_level_names_a_star_realm() {
        let level = RealmLevel {
            kind: RealmKindTag::Star,
            seed: 77,
        };
        assert_eq!(level.to_realm_id(), RealmId::Star(77));
    }
}
