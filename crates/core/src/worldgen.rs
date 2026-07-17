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

use glam::DVec3;

use crate::geometry::{Boundary, ContainmentBand, RealmRegion};
use crate::pose::{LatticePos, RealmId, frame_for_realm};

/// The inner (acquire) edge of the P3 static containment band, metres inside a surface.
const CONTAINMENT_INSET_M: f64 = 1.0;
/// The outer (release) edge of the P3 static containment band, metres outside a surface.
const CONTAINMENT_OUTSET_M: f64 = 2.0;

// --- P3 WALK-SCALE geometry (placeholders; P4/P5 makes every radius/center `f(seed[, tick])`, D-44). ---
/// The ambient-root (Universe) radius — effectively unbounded; an entity beyond it still resolves to the
/// Universe by the container fold IDENTITY. Non-renderable (far above the render-extent threshold).
const UNIVERSE_R_M: f64 = 1.0e9;
/// The galaxy radius — FINITE (it encloses the star systems), and it is the between-systems space an
/// entity occupies after leaving one system SOI and before entering the next. Non-renderable.
const GALAXY_R_M: f64 = 1_000.0;
/// A star-system SOI radius (walk scale).
const SYSTEM_SOI_R_M: f64 = 40.0;
/// A planet SOI radius (walk scale), nested inside a system.
const PLANET_SOI_R_M: f64 = 10.0;
/// System B's center on +X — a disjoint sibling of System A with a WALKABLE gap of galaxy between them.
const SYSTEM_B_OFFSET_M: f64 = 100.0;
/// Planet A's center inside System A (offset from the star at the origin).
const PLANET_A_OFFSET_M: f64 = 20.0;

/// P3 placeholder realm ids for the hierarchy levels that lack a dedicated `RealmId` arm (Universe,
/// Galaxy get one at P4+). The star systems + planet use their real seeds.
const UNIVERSE: RealmId = RealmId::System(0);
const GALAXY: RealmId = RealmId::System(1);
const SYSTEM_A: RealmId = RealmId::System(7);
const SYSTEM_B: RealmId = RealmId::System(8);
const PLANET_A: RealmId = RealmId::Planet(7);

/// The largest region extent the CLIENT renders as a box: finite leaf realms (systems/planets) are drawn,
/// the ~unbounded ambient shells (Galaxy/Universe) are NOT — the between-space is felt, not framed. Set
/// between a system SOI ([`SYSTEM_SOI_R_M`] = 40) and the galaxy ([`GALAXY_R_M`] = 1000).
pub const MAX_RENDERABLE_EXTENT_M: f64 = 100.0;

/// The single source of truth for realm→region geometry (see the module docs). At P3 returns the static
/// WALK-scale mandate forest; `seed_universe` is threaded for the frozen P4/P5 `f(seed)` signature
/// (unused while the bodies are static).
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
        // input-side containment seam and the output-side `rebind_pose_to_dest` agree.
        frame: frame_for_realm(realm, parent).expect("System/Planet realms have a canonical frame"),
        shape: Boundary::Shell { r },
        band,
        parent,
    };

    vec![
        // Universe: the ambient ROOT (parent None) — contains all reachable space (the fold identity).
        shell(UNIVERSE, DVec3::ZERO, UNIVERSE_R_M, None),
        // Galaxy: the finite between-systems space, nested in the Universe.
        shell(GALAXY, DVec3::ZERO, GALAXY_R_M, Some(UNIVERSE)),
        // Star system A: nested in the Galaxy at the origin.
        shell(SYSTEM_A, DVec3::ZERO, SYSTEM_SOI_R_M, Some(GALAXY)),
        // Planet A: nested in system A, offset from the star.
        shell(
            PLANET_A,
            DVec3::new(PLANET_A_OFFSET_M, 0.0, 0.0),
            PLANET_SOI_R_M,
            Some(SYSTEM_A),
        ),
        // Star system B: a DISJOINT sibling of system A under the Galaxy — a walkable gap of galaxy
        // between them (leave A at +40, cross galaxy, enter B at +60).
        shell(
            SYSTEM_B,
            DVec3::new(SYSTEM_B_OFFSET_M, 0.0, 0.0),
            SYSTEM_SOI_R_M,
            Some(GALAXY),
        ),
    ]
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
        // The mandate depths: Universe 0, Galaxy 1, System 2, Planet 3; the sibling system is same-depth.
        assert_eq!(region_depth(&rs, UNIVERSE), 0);
        assert_eq!(region_depth(&rs, GALAXY), 1);
        assert_eq!(region_depth(&rs, SYSTEM_A), 2);
        assert_eq!(region_depth(&rs, PLANET_A), 3);
        assert_eq!(region_depth(&rs, SYSTEM_B), 2);
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
        // System 7's shard: own + ancestors (Galaxy, Universe) + child Planet 7 — NOT sibling System 8.
        let n7: Vec<RealmId> = realm_neighbourhood_for(0, SYSTEM_A)
            .iter()
            .map(|r| r.realm)
            .collect();
        assert!(n7.contains(&SYSTEM_A));
        assert!(n7.contains(&GALAXY));
        assert!(n7.contains(&UNIVERSE));
        assert!(n7.contains(&PLANET_A));
        assert!(
            !n7.contains(&SYSTEM_B),
            "a shard NEVER loads a sibling — the scale-bounded rule",
        );
        assert_eq!(n7.len(), 4);
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
        // A shard hosting an unknown realm ⇒ empty neighbourhood ⇒ the detector is inert (safe degrade).
        assert!(realm_neighbourhood_for(0, RealmId::Station(99)).is_empty());
    }

    #[test]
    fn region_depth_of_an_unknown_realm_is_zero() {
        assert_eq!(region_depth(&regions(), RealmId::Station(99)), 0);
    }
}
