//! Seed-derived realm-region registry — the SINGLE source of truth for realm→region geometry
//! (task #135, C-2). Closed-form `f(seed)`: every shard computes the IDENTICAL containment forest at
//! boot from the shared universe seed, so the geometry is REPLICATED BY CONSTRUCTION — no shared
//! mutable state, no inter-shard bytes (HR1). At P3 the bodies are STATIC (identity ephemeris, one
//! universe); at P4/P5 the per-realm celestial parameters (luminosity/mass/orbital elements) become
//! `f(seed)` and `center` becomes `f(seed, universe_tick)`, ADDITIVELY — the registry SHAPE is frozen.
//!
//! The P3 forest models the user-mandated hierarchy **Universe ⊃ Galaxy ⊃ StarSystem ⊃ Planet**, plus
//! a sibling star system, so the containment detector exercises the full mandate chain ("escape the SOI
//! → the star system; escape the star system → the galaxy"). `RealmId::System`/`Planet` stand in for the
//! levels at P3; dedicated `RealmId::{Universe,Galaxy}` arms + real ephemeris land at P4+ (DEFERRED D-44).

use glam::DVec3;

use crate::geometry::{Boundary, ContainmentBand, RealmRegion};
use crate::pose::{LatticePos, RealmId, frame_for_realm};

/// The inner (acquire) edge of the P3 static containment band, metres inside a surface.
const CONTAINMENT_INSET_M: f64 = 1.0;
/// The outer (release) edge of the P3 static containment band, metres outside a surface.
const CONTAINMENT_OUTSET_M: f64 = 2.0;

/// P3 placeholder realm ids for the hierarchy levels that lack a dedicated `RealmId` arm (Universe,
/// Galaxy get one at P4+). The star systems + planet use their real seeds.
const UNIVERSE: RealmId = RealmId::System(0);
const GALAXY: RealmId = RealmId::System(1);
const SYSTEM_A: RealmId = RealmId::System(7);
const SYSTEM_B: RealmId = RealmId::System(8);
const PLANET_A: RealmId = RealmId::Planet(7);

/// The single source of truth for realm→region geometry (see the module docs). At P3 returns the static
/// mandate forest concentric-nested with disjoint sibling systems; `seed_universe` is threaded for the
/// frozen P4/P5 `f(seed)` signature (unused while the bodies are static).
#[must_use]
pub fn realm_regions_for(_seed_universe: u64) -> Vec<RealmRegion> {
    // ONE band for the whole P3 static forest — the geometry is static (v_rel = 0), so the velocity
    // widening is inert; the edges are valid by construction, so the ctor never errors here.
    let band = ContainmentBand::for_containment_velocity_safe(
        CONTAINMENT_INSET_M,
        CONTAINMENT_OUTSET_M,
        0.0,
        1.0,
        0.0,
    )
    .expect("P3 containment band edges are valid by construction");

    let shell = |realm: RealmId, center: DVec3, r: f64, parent: Option<RealmId>| RealmRegion {
        realm,
        center: LatticePos::local(center),
        // The region frame = the realm's canonical authority frame (`frame_for_realm`), so the
        // input-side containment seam and the output-side `rebind_pose_to_dest` agree. System/Planet
        // realms always resolve a frame; only `Area` (absent here) needs a parent.
        frame: frame_for_realm(realm, parent).expect("System/Planet realms have a canonical frame"),
        shape: Boundary::Shell { r },
        band,
        parent,
    };

    vec![
        // Universe: the ambient ROOT (parent None) — its volume contains all reachable space, so an
        // entity is ALWAYS in at least this realm (the container fold identity).
        shell(UNIVERSE, DVec3::ZERO, 1.0e12, None),
        // Galaxy: nested in the Universe; contains the star systems.
        shell(GALAXY, DVec3::ZERO, 1.0e9, Some(UNIVERSE)),
        // Star system A (box A): nested in the Galaxy at the origin, SOI radius 100.
        shell(SYSTEM_A, DVec3::ZERO, 100.0, Some(GALAXY)),
        // Planet A: nested in system A, offset from the star, SOI radius 12.
        shell(PLANET_A, DVec3::new(30.0, 0.0, 0.0), 12.0, Some(SYSTEM_A)),
        // Star system B (box B): a DISJOINT sibling of system A under the Galaxy, offset by 500.
        shell(SYSTEM_B, DVec3::new(500.0, 0.0, 0.0), 100.0, Some(GALAXY)),
    ]
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
    /// surface the point is on/inside (`signed_distance <= 0`), folded from the Universe root. This is
    /// the GEOMETRIC CONTRACT the sim's stateful band membership (C-3) layers hysteresis on top of.
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
        // Exactly one ambient root (parent None).
        assert_eq!(rs.iter().filter(|r| r.parent.is_none()).count(), 1);
        // `.realm` is unique per region (so `region_depth`'s parent walk is deterministic).
        let mut realms: Vec<RealmId> = rs.iter().map(|r| r.realm).collect();
        realms.sort();
        realms.dedup();
        assert_eq!(realms.len(), rs.len(), "every region has a distinct realm");
        // The mandate depths: Universe 0, Galaxy 1, System 2, Planet 3.
        assert_eq!(region_depth(&rs, UNIVERSE), 0);
        assert_eq!(region_depth(&rs, GALAXY), 1);
        assert_eq!(region_depth(&rs, SYSTEM_A), 2);
        assert_eq!(region_depth(&rs, PLANET_A), 3);
        assert_eq!(
            region_depth(&rs, SYSTEM_B),
            2,
            "the sibling system is same-depth"
        );
    }

    #[test]
    fn containment_resolves_the_full_mandate_chain() {
        // Deep inside planet A's SOI (offset 30) → the PLANET (deepest container).
        assert_eq!(container_at(DVec3::new(30.0, 0.0, 0.0)), PLANET_A);
        // Inside system A but outside planet A → the STAR SYSTEM (escape the SOI → the system).
        assert_eq!(container_at(DVec3::new(60.0, 0.0, 0.0)), SYSTEM_A);
        // Outside every system SOI but inside the galaxy → the GALAXY (escape the system → the galaxy).
        assert_eq!(container_at(DVec3::new(300.0, 0.0, 0.0)), GALAXY);
        // Inside sibling system B → SYSTEM_B (the symmetric other side of the round trip).
        assert_eq!(container_at(DVec3::new(500.0, 0.0, 0.0)), SYSTEM_B);
        // Beyond the galaxy but within the universe → the UNIVERSE root (always in a realm).
        assert_eq!(container_at(DVec3::new(1.0e10, 0.0, 0.0)), UNIVERSE);
        // Beyond EVERYTHING (outside the universe shell too) → STILL the Universe, by fold identity.
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
        assert!((sd - (-100.0)).abs() < 1e-9, "center is r_soi inside: {sd}");
    }

    #[test]
    fn region_depth_of_an_unknown_realm_is_zero() {
        // A realm not in the forest ⇒ the parent walk finds nothing ⇒ depth 0 (the dangling-parent arm).
        assert_eq!(region_depth(&regions(), RealmId::Station(99)), 0);
    }
}
