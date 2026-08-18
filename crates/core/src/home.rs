//! WHERE AN ACCOUNT APPEARS — **stored, never derived.**
//!
//! A home is two facts, and they are held together because neither is usable alone:
//!
//! ```text
//!   WHICH REALM   a full root→leaf lineage, so demanding it spins the whole chain up
//!   WHERE INSIDE  a pose measured from THAT realm's own centre, in THAT realm's own frame
//! ```
//!
//! ## Why it is stored
//!
//! It used to be derived: an account held one universe-absolute position, and the router walked the whole
//! seed forest downward — subtracting each realm's stored centre as it stepped — to work out which realm
//! that position fell in and what it became once measured from there. Two things were wrong with it.
//!
//! The first is measured. A realm that ORBITS stores its centre as zero, because its live position comes
//! from its parent every tick and belongs on the live lane, not in a stored field. The downward walk
//! subtracted that zero, so every orbiting planet read as sitting exactly on its own star, and a login at
//! a star's centre resolved to *inside a planet*. Standing at the first star's own centre, all five of its
//! planets answered `4.160705354131045 m INSIDE` while the same planets, asked in their own frames through
//! their live placements, answered between `13.76` and `140.31 m OUTSIDE`. Nothing that stands still can
//! show this, which is why it survived every test in the tree.
//!
//! The second is structural, and it is the reason the fix is not "make the walk read live placements".
//! Only a parent may state where its children are; the router owns no realm, so it may not state where any
//! realm is at all. A stored home asks nobody: the realm is a NAME, and the pose is already in that realm's
//! own frame, which is the one frame the realm can read without ever learning where it itself sits.
//!
//! ## What this deliberately is not
//!
//! It is not a login special case. The chain spins up through the ordinary demand mechanics from the
//! lineage below, exactly as any other realm does, and no machinery anywhere asks where the first position
//! came from. Today the registry is populated with one fixed home; the shape is what lets that become a
//! decision later — a busy city choosing between its apartments, say — without a second path appearing.

use std::collections::BTreeMap;

use glam::DVec3;

use crate::geometry::RealmRegion;
use crate::ids::{AccountId, UniverseTick};
use crate::pose::{RealmId, StampedPose};
use crate::realm_coord::RealmCoord;

/// One account's home: the realm it lives in, and where it stands inside that realm.
///
/// The pose is measured from the realm's OWN centre and stamped with the realm's OWN frame. The receiving
/// shard checks that frame and refuses anything else — it cannot convert a foreign number into its own
/// space, because it does not know where it itself sits.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredHome {
    /// The full root→leaf lineage. Demanding it is what spins the chain up, level by level.
    pub realm: RealmCoord,
    /// Where inside that realm, in that realm's own frame, at rest.
    pub pose: StampedPose,
}

impl StoredHome {
    /// The home of `realm`, `offset` metres from that realm's own centre.
    ///
    /// `regions` is consulted for two NAMES and nothing else: the realm's lineage and the label of its own
    /// frame. No distance is read from it, so no realm's position is involved in building a home.
    ///
    /// `None` when the forest does not name that realm — an unknown home is stated as unknown rather than
    /// silently becoming the origin of some other realm.
    #[must_use]
    pub fn in_realm(regions: &[RealmRegion], realm: RealmId, offset: DVec3) -> Option<StoredHome> {
        let frame = regions.iter().find(|r| r.realm == realm)?.frame;
        let coord = crate::worldgen::coord_of_realm(regions, realm)?;
        Some(StoredHome {
            realm: coord,
            pose: StampedPose::at_rest(frame, offset, UniverseTick(0)),
        })
    }
}

/// Every account's home, with one fallback for an account that has none yet.
///
/// A registry rather than a bare map because the fallback is not a special case to be written out at each
/// call site: an account with no stored home has a home, it just is not its own.
#[derive(Clone, Debug, PartialEq)]
pub struct HomeRegistry {
    fallback: StoredHome,
    by_account: BTreeMap<AccountId, StoredHome>,
}

impl HomeRegistry {
    /// A registry whose every account starts at `fallback`.
    #[must_use]
    pub fn new(fallback: StoredHome) -> HomeRegistry {
        HomeRegistry {
            fallback,
            by_account: BTreeMap::new(),
        }
    }

    /// Give one account a home of its own.
    #[must_use]
    pub fn with_account(mut self, account: AccountId, home: StoredHome) -> HomeRegistry {
        self.by_account.insert(account, home);
        self
    }

    /// THE ONLY QUESTION THIS ANSWERS: where does this account appear? One lookup, no arithmetic, and
    /// always an answer.
    #[must_use]
    pub fn home_of(&self, account: AccountId) -> StoredHome {
        self.by_account
            .get(&account)
            .cloned()
            .unwrap_or_else(|| self.fallback.clone())
    }

    /// The home every account without one of its own gets.
    #[must_use]
    pub fn fallback(&self) -> &StoredHome {
        &self.fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{AoiConfig, Boundary, ContainmentBand};
    use crate::pose::{FrameRef, LatticePos};

    fn region(realm: RealmId, parent: Option<RealmId>, frame: FrameRef) -> RealmRegion {
        RealmRegion {
            realm,
            center: LatticePos::local(DVec3::ZERO),
            frame,
            shape: Boundary::Shell { r: 100.0 },
            band: ContainmentBand::for_containment_velocity_safe(1.0, 1.0, 100.0, 0.02, 0.0)
                .expect("a valid containment band"),
            aoi: AoiConfig::inert(),
            parent,
            interior_band: AoiConfig::inert(),
        }
    }

    fn forest() -> Vec<RealmRegion> {
        vec![
            region(
                RealmId::System(0),
                None,
                FrameRef::SystemSpace { system_seed: 0 },
            ),
            region(
                RealmId::System(7),
                Some(RealmId::System(0)),
                FrameRef::SystemSpace { system_seed: 7 },
            ),
        ]
    }

    #[test]
    fn a_home_names_a_lineage_and_a_pose_in_that_realms_own_frame() {
        let home = StoredHome::in_realm(&forest(), RealmId::System(7), DVec3::new(25.0, 0.0, 0.0))
            .expect("the forest names this realm");
        assert_eq!(home.realm.lowered(), RealmId::System(7));
        assert_eq!(home.pose.frame, FrameRef::SystemSpace { system_seed: 7 });
        assert_eq!(home.pose.pos.offset(), DVec3::new(25.0, 0.0, 0.0));
    }

    #[test]
    fn a_realm_the_forest_does_not_name_has_no_home() {
        assert_eq!(
            StoredHome::in_realm(&forest(), RealmId::System(99), DVec3::ZERO),
            None,
        );
    }

    #[test]
    fn an_account_without_a_home_of_its_own_gets_the_fallback() {
        let fallback = StoredHome::in_realm(&forest(), RealmId::System(7), DVec3::ZERO)
            .expect("the forest names this realm");
        let mine = StoredHome::in_realm(&forest(), RealmId::System(0), DVec3::new(1.0, 0.0, 0.0))
            .expect("the forest names this realm");
        let registry = HomeRegistry::new(fallback.clone()).with_account(AccountId(5), mine.clone());

        assert_eq!(registry.home_of(AccountId(5)), mine);
        assert_eq!(registry.home_of(AccountId(6)), fallback);
        assert_eq!(registry.fallback(), &fallback);
    }
}
