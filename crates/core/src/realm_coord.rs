//! `RealmCoord` (RLM Step 1) — the realm-lifecycle addressing that names ANY level.
//!
//! THIS TYPE IS ON THE WIRE (it rides [`crate::realm_path`]'s serde-derived types behind the
//! frozen `InterShardFlow::RealmDemand` arm). It is globally unique BY CONSTRUCTION: two same-seed
//! Systems in different Galaxies get distinct [`RealmPath`]s (the Galaxy level differs), fixing the
//! cross-galaxy aliasing bare [`RealmId`] has. The lowered `RealmId` is LOSSY (Universe/Galaxy
//! collapse to the interim stand-ins, and a System seed collapses across galaxies) — the `path` is
//! the collision-free disambiguator, NEVER [`RealmCoord::lowered`] (see its doc).

use crate::pose::RealmId;
use crate::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use crate::taxonomy::ProfileKind;
use serde::{Deserialize, Serialize};

/// A realm's lifecycle address: its own [`RealmLevel`] (kind + seed) plus the full root→leaf
/// [`RealmPath`] lineage. INVARIANT (maintained by every constructor): `path.levels().last() ==
/// Some(level)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealmCoord {
    level: RealmLevel,
    path: RealmPath,
}

impl RealmCoord {
    /// Build from a full lineage path; the leaf level IS the coord's own level. `None` only if the
    /// path is empty (the sole failure branch). PRODUCER CONTRACT (not runtime-validated, to stay
    /// branchless): the path SHOULD be Universe-rooted + contiguous — a truncated path aliases;
    /// feed the generator's full lineage.
    #[must_use]
    pub fn from_path(path: RealmPath) -> Option<RealmCoord> {
        path.levels()
            .last()
            .copied()
            .map(|level| RealmCoord { level, path })
    }

    /// This coord's own level (kind + seed).
    #[must_use]
    pub fn level(&self) -> RealmLevel {
        self.level
    }

    /// The root→leaf lineage (the globally-unique key).
    #[must_use]
    pub fn path(&self) -> &RealmPath {
        &self.path
    }

    /// The `RealmId` this coord LOWERS to — total, via the leaf's kind. LOSSY: Universe→System(0),
    /// Galaxy→System(1), and every System(seed) across every galaxy collapses by seed alone. NEVER
    /// use as a directory/saga/dedup key at multi-galaxy scale — [`RealmCoord::path`] is the
    /// collision-free key. Provided only where a real `RealmId` is owed (the keyed kinds today).
    #[must_use]
    pub fn lowered(&self) -> RealmId {
        self.level.to_realm_id()
    }

    /// The `ShardProfile` selector for this realm's kind (HR3: a capability config, never a feature
    /// match-on-kind). Universe/Galaxy both collapse to `Galaxy`, mirroring
    /// [`crate::taxonomy::ProfileKind`]'s note (a Universe root and a Galaxy relay want the same
    /// no-voxel/signal-relay caps today); a future split is a deliberate edit of `profile_kind_of`.
    #[must_use]
    pub fn profile_kind(&self) -> ProfileKind {
        profile_kind_of(self.level.kind)
    }

    /// ★ EVERY REALM ON THIS LINEAGE, root → leaf (2026-08-30).
    ///
    /// A coordinate carries each ancestor's KIND and SEED, so it names its ancestors exactly. This is
    /// what lets a shard build a world for a realm the seed alone can never place: a planet's
    /// identifier is a one-way hash of its system's, and a player-built city is not in the seed at
    /// all — the generator emits no station and no area.
    ///
    /// THE PARENT IS THE SOURCE, and it always was: a spawn demand carries a `RealmCoord`, not a
    /// name, so the realm that demanded a child into existence has already told that child who
    /// contains it. This reads what was already sent.
    #[must_use]
    pub fn lineage_realms(&self) -> Vec<RealmId> {
        self.path.levels().iter().map(|l| l.to_realm_id()).collect()
    }

    /// The parent coord (path truncated by one leaf), or `None` at the root (path length ≤ 1).
    #[must_use]
    pub fn parent(&self) -> Option<RealmCoord> {
        let levels = self.path.levels();
        if levels.len() >= 2 {
            RealmCoord::from_path(RealmPath::from_levels(levels[..levels.len() - 1].to_vec()))
        } else {
            None
        }
    }

    /// The child coord: this coord's path EXTENDED by one `level`. The extend direction the AoI
    /// loop needs to name a child it hasn't spawned — built by `parent.child(level)`, NOT by
    /// inverting a `RealmId` (the general inversion is owed to P4). Total: the extended path always
    /// has a leaf, so there is no failure branch (branchless).
    #[must_use]
    pub fn child(&self, level: RealmLevel) -> RealmCoord {
        let mut levels = self.path.levels().to_vec();
        levels.push(level);
        RealmCoord {
            level,
            path: RealmPath::from_levels(levels),
        }
    }
}

/// The ONE `RealmKindTag → ProfileKind` bridge (the monomorphic-helper split, so `profile_kind()`
/// is a branchless one-line delegate — mirrors `RealmLevel::to_realm_id`). Total over the 7 tags
/// only; consistent with `vd_sim::capability::profile_for`'s Universe/Galaxy→`Galaxy` collapse.
/// A STAR is a body with an extent and eventually a surface, so `ProfileKind::Planet` is the
/// right capability set (taxonomy arc §6.2: "new shard types are new values here — zero new
/// feature code"; `ProfileKind` has no inter-shard producer, so this is revisable without wire).
#[must_use]
fn profile_kind_of(kind: RealmKindTag) -> ProfileKind {
    match kind {
        RealmKindTag::Universe => ProfileKind::Galaxy,
        RealmKindTag::Galaxy => ProfileKind::Galaxy,
        RealmKindTag::System => ProfileKind::System,
        RealmKindTag::Planet => ProfileKind::Planet,
        RealmKindTag::Station => ProfileKind::Station,
        RealmKindTag::Area => ProfileKind::Area,
        RealmKindTag::Star => ProfileKind::Planet,
        // The capability side was ready before the lineage side existed: `ProfileKind::Ship` has been
        // here all along, and a ship shard has had a profile to boot with. What was missing was any
        // way to NAME a ship in a lineage, which is what the `Ship` tag adds.
        RealmKindTag::Ship => ProfileKind::Ship,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-level coord (root-only path) for a given kind/seed.
    fn coord1(kind: RealmKindTag, seed: u64) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(kind, seed)]))
            .expect("1-level path has a leaf")
    }

    /// A coord for the full lineage `[Universe, Galaxy(g), System(s)]`.
    fn system_in_galaxy(g: u64, s: u64) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::Universe, 0),
            RealmLevel::new(RealmKindTag::Galaxy, g),
            RealmLevel::new(RealmKindTag::System, s),
        ]))
        .expect("3-level path has a leaf")
    }

    #[test]
    fn lowered_covers_every_kind() {
        // ★ S9: the two ambient kinds lower to their OWN arms. Note the seed 99 — a galaxy now
        // CARRIES it, where the stand-in threw it away and answered `System(1)` whatever it was.
        assert_eq!(
            coord1(RealmKindTag::Universe, 99).lowered(),
            RealmId::Universe
        );
        assert_eq!(
            coord1(RealmKindTag::Galaxy, 99).lowered(),
            RealmId::Galaxy(99)
        );
        assert_eq!(
            coord1(RealmKindTag::System, 7).lowered(),
            RealmId::System(7)
        );
        assert_eq!(
            coord1(RealmKindTag::Planet, 7).lowered(),
            RealmId::Planet(7)
        );
        assert_eq!(
            coord1(RealmKindTag::Station, 7).lowered(),
            RealmId::Station(7)
        );
        assert_eq!(coord1(RealmKindTag::Area, 7).lowered(), RealmId::Area(7));
    }

    #[test]
    fn profile_kind_covers_every_kind() {
        // The kind→ProfileKind mapping (Universe/Galaxy collapse to Galaxy). The "profile_for
        // accepts every one of these" consistency check lives in vd-sim (where profile_for is).
        assert_eq!(
            coord1(RealmKindTag::Universe, 3).profile_kind(),
            ProfileKind::Galaxy
        );
        assert_eq!(
            coord1(RealmKindTag::Galaxy, 3).profile_kind(),
            ProfileKind::Galaxy
        );
        assert_eq!(
            coord1(RealmKindTag::System, 3).profile_kind(),
            ProfileKind::System
        );
        assert_eq!(
            coord1(RealmKindTag::Planet, 3).profile_kind(),
            ProfileKind::Planet
        );
        assert_eq!(
            coord1(RealmKindTag::Station, 3).profile_kind(),
            ProfileKind::Station
        );
        assert_eq!(
            coord1(RealmKindTag::Area, 3).profile_kind(),
            ProfileKind::Area
        );
    }

    #[test]
    fn from_path_leaf_and_empty() {
        let c = system_in_galaxy(2, 7);
        assert_eq!(c.level(), RealmLevel::new(RealmKindTag::System, 7));
        assert_eq!(RealmCoord::from_path(RealmPath::from_levels(vec![])), None);
    }

    #[test]
    fn parent_walks_up_and_root_is_none() {
        let sys = system_in_galaxy(2, 7); // [Universe, Galaxy(2), System(7)]
        let galaxy = sys.parent().expect("system has a galaxy parent");
        assert_eq!(galaxy.level(), RealmLevel::new(RealmKindTag::Galaxy, 2));
        assert_eq!(galaxy.path().levels().len(), 2);
        let universe = galaxy.parent().expect("galaxy has a universe parent");
        assert_eq!(universe.level(), RealmLevel::new(RealmKindTag::Universe, 0));
        assert_eq!(universe.path().levels().len(), 1);
        assert_eq!(universe.parent(), None); // root
    }

    #[test]
    fn child_extends_path() {
        let sys = system_in_galaxy(2, 7);
        let n = sys.path().levels().len();
        let planet_level = RealmLevel::new(RealmKindTag::Planet, 11);
        let planet = sys.child(planet_level);
        assert_eq!(planet.level(), planet_level);
        assert_eq!(planet.path().levels().len(), n + 1);
        // child then parent is the identity (extend inverts truncate).
        assert_eq!(planet.parent(), Some(sys));
    }

    #[test]
    fn level_and_path_accessors() {
        let c = system_in_galaxy(2, 7);
        assert_eq!(c.level(), RealmLevel::new(RealmKindTag::System, 7));
        assert_eq!(c.path().levels().len(), 3);
    }

    /// ★ A COORD NAMES ITS WHOLE LINEAGE, root → leaf. A system inside a galaxy states the
    /// universe, the galaxy and itself, in that order. A shard that spawns a realm reads this list
    /// to learn which realms contain it, without asking anybody.
    #[test]
    fn lineage_realms_names_every_ancestor_root_to_leaf() {
        let sys = system_in_galaxy(2, 7);
        assert_eq!(
            sys.lineage_realms(),
            vec![RealmId::Universe, RealmId::Galaxy(2), RealmId::System(7)]
        );
        // A root-only coord names one realm: itself.
        assert_eq!(
            coord1(RealmKindTag::Galaxy, 5).lineage_realms(),
            vec![RealmId::Galaxy(5)]
        );
    }

    #[test]
    fn realm_coord_postcard_roundtrips() {
        let c = system_in_galaxy(2, 7);
        let bytes = postcard::to_allocvec(&c).expect("encode");
        let back: RealmCoord = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, c);
    }

    #[test]
    fn lowered_aliases_across_levels() {
        // The KNOWN lossy collisions — pinned so no future author keys a directory/saga on lowered().
        //
        // ★ TWO OF THEM ARE GONE (S9). A universe and a galaxy used to lower onto `System(0)` and
        // `System(1)`, borrowing a system's identity because they had none of their own. They have
        // their own arms now, so a galaxy KEEPS ITS SEED and a universe is fieldless — neither can
        // collide with a system any more. Asserted as the non-collisions they now are, because
        // "these used to alias" is exactly the kind of claim that rots into a comment.
        assert_eq!(
            coord1(RealmKindTag::Universe, 0).lowered(),
            RealmId::Universe
        );
        assert_eq!(
            coord1(RealmKindTag::Galaxy, 1).lowered(),
            RealmId::Galaxy(1)
        );
        assert_ne!(
            coord1(RealmKindTag::Galaxy, 1).lowered(),
            coord1(RealmKindTag::System, 1).lowered(),
            "a galaxy and a system of the same seed are no longer the same id"
        );
        assert_ne!(
            coord1(RealmKindTag::Galaxy, 2).lowered(),
            coord1(RealmKindTag::Galaxy, 3).lowered(),
            "and two galaxies are no longer the same id either"
        );
        // ★ THE ONE THAT REMAINS, and the reason this test keeps its name: the same system seed in
        // two different galaxies still lowers identically. That is the collision that makes
        // `lowered()` unsafe as a directory/saga key at multi-galaxy scale, and S9 did not close it —
        // `RealmCoord::path` is still the collision-free key.
        let a = system_in_galaxy(2, 7);
        let b = system_in_galaxy(3, 7);
        assert_eq!(a.lowered(), b.lowered());
        assert_ne!(a.path(), b.path());
    }
}
