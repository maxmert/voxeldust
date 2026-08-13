//! **WHICH REALM HOLDS THIS POINT HAS EXACTLY ONE ANSWER.** The gate the whole re-home arc rests on.
//!
//! The tree used to hold two, and they disagreed the moment anything moved:
//!
//! ```text
//!   THE PARENT'S DOWNWARD ANSWER          THE SHARD'S OWN-FRAME ANSWER
//!   subtract the child's STORED centre    re-express through the child's LIVE placement,
//!   from a point in my frame              then measure from the child's own origin
//!
//!   an ORBITING child stores centre = ZERO   ⇒   the two differed by exactly how far
//!   (its live position rides its frame)          that child had moved from its parent
//! ```
//!
//! Measured through the shipped boot before the cure: standing at the first star's own centre, all five
//! of its planets answered `4.160705354131045 m INSIDE` to the downward form while answering `13.76`,
//! `26.84`, `42.35`, `79.08` and `140.31 m OUTSIDE` in their own frames. Nothing that stands still can
//! show it — with a stored centre equal to the live placement the two forms are the same arithmetic on the
//! same numbers — which is why every test in the tree passed while a player logging in at a star landed
//! inside a planet and a player entering a planet landed outside it.
//!
//! **The second answer no longer exists.** The downward form and the whole login descent it served are
//! deleted, so the disagreement is now unstateable rather than merely untrue — the compiler holds it, not
//! this file. What is left here are the properties that would have caught the defect, asked of the one
//! surviving answer.
//!
//! ## Why it boots the way it does
//!
//! The audit that first measured the gap TRANSCRIBED the shard's boot into itself, and a transcription can
//! agree with a defect it copied. So this asks the shipped boot and nothing else:
//!
//! - the world comes from [`vd_bins::boot_world`] — the ONE world, the gateway's own entry point;
//! - the star's regions and its moving-child roster come from [`vd_bins::boot_regions_and_movers`] — the
//!   exact call the shard binary makes, with the exact speed and tick the dev cluster flies at;
//! - the frame context comes from the two production builders (`RealmRegions::new` +
//!   `with_moving_children`) and the shard's own `frame_context`, at the same tick as the point.

use std::collections::BTreeSet;

use vd_core::UniverseTick;
use vd_core::frame::{FrameContext, LocalFrames};
use vd_core::geometry::{RealmRegion, region_signed_distance};
use vd_core::glam::DVec3;
use vd_core::pose::{RealmId, StampedPose};
use vd_core::worldgen::WorldView;
use vd_sim::stub::RealmRegions;

/// THE SURFACE, at zero. A signed distance is negative inside a realm, positive outside, and zero exactly
/// on the boundary. No tolerance: a margin here would be a budget for being slightly wrong about which
/// realm holds a player, and which realm holds a player is the whole subject.
const SURFACE_M: f64 = 0.0;

/// The universe seed every process in the dev cluster shares (`VD_UNIVERSE_SEED`, absent ⇒ 0).
const SEED: u64 = 0;

/// The ticks the property is sampled at. Tick 0 is the orbital epoch — every body sits on its authored
/// phase, so a defect that only appears once things have MOVED can hide there. The other two are far
/// enough apart that a single unlucky phase cannot make both agree by accident.
const SAMPLED_TICKS: [UniverseTick; 3] =
    [UniverseTick(0), UniverseTick(1_000), UniverseTick(50_000)];

/// The occupant speed the AoI band is sized against — the dev cluster's flight speed, undilated
/// (`VD_TIME_MULTIPLIER` defaults to 1.0, so `move_speed · multiplier` is `move_speed`).
fn occupant_v_max_mps() -> f64 {
    vd_bins::DEV.move_speed
}

/// THE world, through the gateway's own entry point.
fn the_world() -> WorldView {
    vd_bins::boot_world(SEED, occupant_v_max_mps(), vd_bins::DEV.tick_dt)
}

/// The first star system in the forest that actually holds children, and its direct children. DERIVED
/// from the forest rather than named: the realm seeds are hash avalanches of `(galaxy, salt, index)`, so
/// writing one down would copy a hash into a test and go stale the moment the salt or the index moves.
fn a_star_and_its_children(world: &WorldView) -> (RealmId, Vec<RealmId>) {
    let root = world
        .regions()
        .iter()
        .find(|r| r.parent.is_none())
        .expect("a generated world has exactly one ambient root");
    let galaxy = world
        .regions()
        .iter()
        .find(|r| r.parent == Some(root.realm))
        .expect("the ambient root holds a galaxy");
    let star = world
        .regions()
        .iter()
        .filter(|r| r.parent == Some(galaxy.realm))
        .find(|r| world.regions().iter().any(|c| c.parent == Some(r.realm)))
        .expect("the generated galaxy holds a star system with planets");
    let children = world
        .regions()
        .iter()
        .filter(|r| r.parent == Some(star.realm))
        .map(|r| r.realm)
        .collect();
    (star.realm, children)
}

/// The star shard exactly as its binary boots it: the production regions and mover roster, planted
/// through the production builders.
fn star_shard(star: RealmId) -> (Vec<RealmRegion>, RealmRegions) {
    let held = BTreeSet::from([star]);
    let (regions, moving) = vd_bins::boot_regions_and_movers(
        SEED,
        &held,
        star,
        occupant_v_max_mps(),
        vd_bins::DEV.tick_dt,
    );
    let planted = RealmRegions::new(regions.clone()).with_moving_children(moving);
    (regions, planted)
}

/// The frame the star measures everything in — its own realm's frame, read off the forest it booted with.
fn own_frame(regions: &[RealmRegion], star: RealmId) -> vd_core::pose::FrameRef {
    regions
        .iter()
        .find(|r| r.realm == star)
        .expect("a shard's own realm is in the neighbourhood it boots with")
        .frame
}

/// The star's direct children, as regions.
fn child_regions(regions: &[RealmRegion], star: RealmId) -> Vec<RealmRegion> {
    regions
        .iter()
        .filter(|r| r.parent == Some(star))
        .copied()
        .collect()
}

/// THE SHARD'S OWN-FRAME ANSWER: re-express the same point into the child's own frame through the live
/// placement, then measure from the child's own origin.
fn own_frame_answer(point_in_star: &StampedPose, child: &RealmRegion, ctx: &LocalFrames) -> f64 {
    region_signed_distance(point_in_star, child, ctx).unwrap_or_else(|e| {
        panic!(
            "a star authors every one of its own direct children, so re-expressing a point into \
             {:?} must be available; got {e:?}",
            child.realm
        )
    })
}

/// THE SHAPE OF EVERY CHILD, AND ITS DISTANCE — the two numbers that decide whether a star's own centre
/// can fall inside one of its own planets.
///
/// A planet whose boundary is WIDER than its orbit swallows its own star, and then a point at the star is
/// legitimately inside the planet by the arithmetic — no transform bug required. That is one of exactly two
/// explanations for a live shard reporting `signed_distance = -2.06` for an occupant at the star with the
/// planet's placement 17.90 m away; the other is that the reframe returns the wrong magnitude. `radius`
/// against `distance` here tells them apart, and it is the world's own geometry either way.
#[test]
fn no_child_of_a_star_is_wider_than_its_own_distance_from_that_star() {
    let world = the_world();
    let (star, _) = a_star_and_its_children(&world);
    let (regions, planted) = star_shard(star);
    let mut swallowing = Vec::new();
    for tick in SAMPLED_TICKS {
        let ctx = planted.frame_context(star, f64::from(vd_bins::DEV.tick_hz), tick);
        for child in child_regions(&regions, star) {
            let at = ctx
                .placement(child.frame, tick)
                .expect("a star places every direct child of its own");
            let distance = at.origin.length();
            let radius = child.shape.circumscribed_extent();
            eprintln!(
                "[geometry] tick {} {:?} radius {radius} m at distance {distance} m",
                tick.0, child.realm,
            );
            if radius >= distance {
                swallowing.push(format!(
                    "  tick {}: {:?} reaches {radius} m and sits only {distance} m from its star",
                    tick.0, child.realm,
                ));
            }
        }
    }
    assert_eq!(
        swallowing.join("\n"),
        "",
        "\nA CHILD SWALLOWS ITS OWN PARENT'S CENTRE:\n{}\n",
        swallowing.join("\n"),
    );
}

/// THE GATE: standing at a star's OWN CENTRE, not one of its children reports holding you.
///
/// This is the strongest form of the question. A parent at its own origin is nowhere near any child it has
/// placed away from itself, so every child must answer OUTSIDE — at every tick, however far round its
/// orbit that child has travelled. The deleted downward form answered INSIDE for all five planets at once,
/// which is exactly how a login at the origin ended up inside a planet.
#[test]
fn no_child_of_a_star_claims_to_hold_the_stars_own_centre() {
    let world = the_world();
    let (star, children) = a_star_and_its_children(&world);
    let (regions, planted) = star_shard(star);
    let star_frame = own_frame(&regions, star);
    assert_eq!(
        child_regions(&regions, star).len(),
        children.len(),
        "the star shard must boot with every one of its own children, or this measures nothing",
    );

    let mut claims = Vec::new();
    for tick in SAMPLED_TICKS {
        let ctx = planted.frame_context(star, f64::from(vd_bins::DEV.tick_hz), tick);
        let star_centre = StampedPose::at_rest(star_frame, DVec3::ZERO, tick);
        for child in child_regions(&regions, star) {
            let own = own_frame_answer(&star_centre, &child, &ctx);
            if own <= SURFACE_M {
                claims.push(format!(
                    "  tick {}: {:?} reads the star's own centre as {own} m from its surface",
                    tick.0, child.realm,
                ));
            }
        }
    }

    assert_eq!(
        claims.join("\n"),
        "",
        "\nA CHILD CLAIMS ITS OWN PARENT'S CENTRE:\n{}\n",
        claims.join("\n"),
    );
}

/// THE OTHER PAIR, on the shard's own side: the per-tick feed and the conversion context are TWO
/// expressions of one number — where this shard put each of its children, this tick.
///
/// They are built by different code (`child_placements` solves each child's motion into a pose; the
/// conversion context re-solves it on every lookup), so nothing but arithmetic makes them agree. This
/// states that they do — or measures the day they stop. A second producer that happens to agree today is
/// still a second producer, which is why the table replaces both; this test is what makes that
/// replacement a provable no-change rather than a claimed one.
#[test]
fn the_per_tick_feed_and_the_conversion_context_place_a_child_identically() {
    let world = the_world();
    let (star, _) = a_star_and_its_children(&world);
    let (_, planted) = star_shard(star);

    let mut splits = Vec::new();
    for tick in SAMPLED_TICKS {
        let ctx = planted.frame_context(star, f64::from(vd_bins::DEV.tick_hz), tick);
        for (child, feed_pose) in
            planted.child_placements(star, f64::from(vd_bins::DEV.tick_hz), tick)
        {
            let converted = ctx
                .placement(child.frame, tick)
                .expect("a star holds a placement for every direct child of its own");
            if feed_pose.pos.offset() != converted.origin {
                splits.push(format!(
                    "  tick {}: {:?} — the feed ships {:?}, the conversion uses {:?}",
                    tick.0,
                    child.realm,
                    feed_pose.pos.offset(),
                    converted.origin,
                ));
            }
        }
    }

    assert_eq!(
        splits.join("\n"),
        "",
        "\nTWO PRODUCERS OF ONE PLACEMENT HAVE PARTED COMPANY:\n{}\n",
        splits.join("\n"),
    );
}

/// THE SYMPTOM, stated as a property: an occupant standing where the star says a child IS must be
/// INSIDE that child.
///
/// This is the arrival the owner flew: re-homing into a planet put the player far outside it, and the
/// planet and the star then traded them back and forth. The position is taken from the star's own
/// authored placement — the one party entitled to state where its child is — so nothing here invents a
/// number that production would not produce.
#[test]
fn an_occupant_where_the_star_puts_a_child_is_inside_that_child() {
    let world = the_world();
    let (star, _) = a_star_and_its_children(&world);
    let (regions, planted) = star_shard(star);
    let star_frame = own_frame(&regions, star);
    let tick = SAMPLED_TICKS[1];
    let ctx = planted.frame_context(star, f64::from(vd_bins::DEV.tick_hz), tick);

    let mut misses = Vec::new();
    for child in child_regions(&regions, star) {
        let placement = ctx
            .placement(child.frame, tick)
            .expect("a star holds a placement for every direct child of its own");
        let at_the_child = StampedPose::at_rest(star_frame, placement.origin, tick);
        let own = own_frame_answer(&at_the_child, &child, &ctx);
        if own >= SURFACE_M {
            misses.push(format!(
                "  {:?}: the child's own frame reads its own centre as {own} m outside itself",
                child.realm,
            ));
        }
    }

    assert_eq!(
        misses.join("\n"),
        "",
        "\nARRIVING WHERE A REALM IS MUST LAND INSIDE IT:\n{}\n",
        misses.join("\n"),
    );
}
