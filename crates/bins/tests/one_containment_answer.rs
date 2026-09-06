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
//! - the placement book comes from the two production builders (`RealmRegions::new` +
//!   `with_moving_children`) and the shard's own `author_book`, at the same tick as the point.

use std::collections::BTreeSet;

use vd_core::UniverseTick;
use vd_core::geometry::{RealmRegion, region_signed_distance};
use vd_core::glam::DVec3;
use vd_core::placement::PlacementBook;
use vd_core::pose::{RealmId, StampedPose};
use vd_physics::worldgen::WorldView;
use vd_sim::stub::RealmRegions;

/// THE SURFACE, at zero. A signed distance is negative inside a realm, positive outside, and zero exactly
/// on the boundary. No tolerance: a margin here would be a budget for being slightly wrong about which
/// realm holds a player, and which realm holds a player is the whole subject.
const SURFACE_M: f64 = 0.0;

/// THE PINNED SEED of this whole file — deliberately **0**, and deliberately NOT the shipped
/// default. Since 2026-08-20 a process with no `VD_UNIVERSE_SEED` boots
/// [`vd_physics::worldgen::HOME_SEED`] (2298, the owner's chosen home world), so this const is a
/// PIN, not an echo of the default it used to be. That is HOME_SEED's own stated doctrine — *"tests
/// that pass an explicit seed are unaffected by it, which is why the pinned `f(seed)` suites still
/// pin seed 0"* — and it is what keeps the golden vector below a stable ruler: a home-seed change is
/// a world change, and a ruler that moved with it would measure nothing.
const SEED: u64 = 0;

/// ★ THE GOLDEN'S REGENERATION RECORD — emitted verbatim into the golden's header so the audit
/// trail rides in the file the bits live in, never only in a commit message.
///
/// STATIC ON PURPOSE. Every other header line is a reading of the live config, so it re-derives
/// silently; this one cannot. The next regenerator has to rewrite it, and the diff shows they did.
/// It names BOTH world changes that landed together, including the one that measurably did NOT move
/// these bits — *"it did not move"* is a claim this file must state, not a silence to read into.
const WHY_THESE_BITS_LAST_MOVED: &[&str] = &[
    "# \u{2605} WHY THESE BITS LAST MOVED — regenerated 2026-08-31. EVERY STAR SNAPPED TO THE GRID.",
    "#",
    "#   A galaxy counts in whole 2 m cells, and a star catalogue row carries the CELL and no sub-cell",
    "#     part. The shaped placement returns a CONTINUOUS position, so a star landed between cells and",
    "#     the row silently dropped the remainder — every client then drew that star up to a metre from",
    "#     where the world put it.",
    "#",
    "#   MEASURED before the fix, on the test galaxy: 33 of 48 systems sat off-cell, by up to 1.875 m.",
    "#     After it: 0 of 48. Every star system now lands on the galaxy's own grid, which is the",
    "#     invariant every reader downstream was already written to assume.",
    "#",
    "#   \u{2605} A FIRST ATTEMPT AT THIS WAS WRONG, and the fence is why it was caught. The snap was put",
    "#     inside the CROWDING PUSH, because a float multiply there was a plausible source. It cured",
    "#     only the pushed stars and the fence stayed red. The fault was never the push — it was the",
    "#     placement. Measuring which systems were off-cell answered it in one run.",
    "#",
    "# --- the record from 2026-08-30, whose two causes still stand ---",
    "# \u{2605} TWO CAUSES, BOTH MEASURED.",
    "#",
    "#   CAUSE 1 — THE RADIAL LAW GAINED A DRAW. A galaxy's placement took SIX draws and now takes",
    "#     SEVEN: `radius_b` sits beside `radius` so the radius is a sum of two exponentials — a",
    "#     Gamma(2). Without it the disc was measured as the density of a LINE, not a DISC: 7 844",
    "#     stars in the innermost bin against the 1 776 the shape asks for.",
    "#",
    "#     THE STREAM LAW SAYS WHERE THAT LANDS, AND IT LANDED THERE EXACTLY. A per-system stream is",
    "#     FROZEN for its first five planets' element draws; everything after the placement is",
    "#     APPENDED. So inserting a placement draw moves what follows it and nothing before it.",
    "#",
    "#     MEASURED, on the rows this file records at both samples:",
    "#       Universe -> Galaxy        3 rows, ALL UNCHANGED",
    "#       System 7 -> its children  5 planets UNCHANGED, 3 moved",
    "#     Five is exactly LEGACY_STREAM_PLANETS. The frozen prefix held; the appended tail moved.",
    "#     That is the promise the prefix exists to make, kept.",
    "#",
    "#   CAUSE 2 — THE CROWDING PUSH (ruling G12). Where the shape asks for a density the separation",
    "#     fence refuses, a star is PUSHED OUT rather than dropped. It moves crowded stars, so star",
    "#     system rows move: 6 of the 9 recorded here.",
    "#",
    "#   \u{2605} AND THE SAMPLE CHANGED. This file recorded EVERY anchor and every child. That was",
    "#     154 lines when a galaxy held three star systems; a galaxy now holds 233 220, so the same",
    "#     rule asks for millions of rows and boots a world per anchor — MEASURED, the test ran",
    "#     fifteen minutes at 14 GB without reaching its assertion. It now records ONE anchor per",
    "#     RUNG of the ladder and a bounded slice of its children. Placement maths varies by rung,",
    "#     never by how many siblings a rung holds, so a bit that drifts anywhere still shows.",
    "#",
    "#   \u{2605} A FALSE CLAIM CAUGHT BEFORE IT WAS WRITTEN HERE. The first draft of this record said",
    "#     no number was recomputed and rows were only removed. Comparing the two files disproved it:",
    "#     15 of 36 shared rows had moved. The claim was an argument, not a reading. It is replaced by",
    "#     the counts above, which are readings.",
    "#",
    "# --- the previous record, kept because the world change it states still stands ---",
    "# \u{2605} regenerated 2026-08-28 (slice S12, ruling G1: THE SHAPE",
    "#   COMES FROM THE SEED). The previous record — the same day's shape flag day — is superseded.",
    "#",
    "#   THE CAUSE. The galaxy's shape was ELEVEN NUMBERS A PERSON CHOSE: two arms, a 25 degree",
    "#     pitch, a disc three percent thick, a bulge holding fifteen percent of the stars, the same",
    "#     for every galaxy at every seed. Ruling G1 forbids that — *\"Density also should come from",
    "#     seed, otherwise any tiny change might change positions\"* — so a galaxy now DRAWS its own",
    "#     shape from its own stream, and the chosen table is deleted rather than left as a second",
    "#     source of truth.",
    "#",
    "#   THE DRAW IS APPENDED, so the galaxy's CENSUS and its KIND are byte-identical across this",
    "#     change: they are draws one and two and the shape follows them. What moved is every star's",
    "#     position, because the law that places them now reads drawn numbers.",
    "#",
    "#   \u{2605} EXACTLY SIX DATA ROWS MOVED, AND COUNTING THEM CORRECTED A WRONG CLAIM. The two",
    "#     NON-HOME SYSTEMS, at three sampled ticks each. Nothing else. The home system sits at the",
    "#     galactic origin by law and its row is unchanged zeros; NO planet row moved at all.",
    "#",
    "#     TWO REASONS, and the second is easy to state backwards. First, a planet draws from ITS OWN",
    "#     SYSTEM'S stream, keyed by that system's seed, which the galaxy's stream cannot reach — so",
    "#     no planet's mass, radius, orbit or temperature changed. Second, and this is the half worth",
    "#     writing down: THIS FILE RECORDS EACH ROW IN ITS PARENT'S FRAME. A planet's row is where it",
    "#     sits in ITS OWN STAR'S frame, and moving the star does not move the planet within it. A",
    "#     first draft of this block said the planet rows \"moved by riding\" their systems. They did",
    "#     not, and the diff says so: zero planet rows changed.",
    "#",
    "#   THE SHAPES WERE JUDGED BEFORE THE WORLD TOOK THEM (ruling G13 — the assistant proposes in",
    "#     code, the owner judges the picture). Eight drawn galaxies were rendered and approved. The",
    "#     rendering earned its keep: it showed that an arm stated as an ABSOLUTE ANGLE reached 137",
    "#     degrees and so filled 152 percent of a four-armed galaxy's gap, overlapping its neighbours",
    "#     into one smooth disc. The census said five times more stars in arms than between them and",
    "#     was TRUE; the picture had no arms. An arm is a share of the gap now.",
    "#",
    "#   \u{2605} WHAT IS *NOT* A CAUSE. No wire type, no codec, no lattice tier and no frame changed.",
    "#     The outer radii and the derived mass cap in the lines above are byte-identical.",
];

/// The ticks the property is sampled at. Tick 0 is the orbital epoch — every body sits on its authored
/// phase, so a defect that only appears once things have MOVED can hide there. The other two are far
/// enough apart that a single unlucky phase cannot make both agree by accident.
const SAMPLED_TICKS: [UniverseTick; 3] =
    [UniverseTick(0), UniverseTick(1_000), UniverseTick(50_000)];

/// The occupant speed the AoI band is sized against — the dev cluster's flight speed, undilated
/// (`VD_TIME_MULTIPLIER` defaults to 1.0, so `move_speed · multiplier` is `move_speed`).
/// Flatten a placement anchor to metres (rows ride the lattice NORMALIZED since the cell
/// activation — `.origin` raw is a sub-cell residual, never a position). Bit-exact at these
/// magnitudes, which is what keeps the golden vector's f64 bits UNCHANGED across the activation.
/// A placement, flattened to metres.
///
/// ★ THE `tier` HERE IS THE ANCHOR'S, NEVER THE CHILD'S (slice S9). A `FramePlacement` from a book is
/// stated in the frame of the realm that AUTHORED it — the book's anchor — so that is the unit it
/// counts in. Two callers passed `child.frame.tier()`, which was the same thing while every realm
/// counted in millimetres and is out by 2048× now that a galaxy does not. One of them writes the
/// golden vector, so the wrong ruler would have been blessed into the file as the world's own bits.
fn placement_m(at: &vd_core::frame::FramePlacement, anchor_tier: vd_core::pose::Tier) -> DVec3 {
    at.anchor()
        .delta_m(vd_core::pose::LatticePos::ORIGIN, anchor_tier)
}

fn occupant_v_max_mps() -> f64 {
    vd_bins::DEV.move_speed
}

/// ★ EVERY REALM WITH AT LEAST ONE DIRECT CHILD — ONE PASS (2026-08-30).
///
/// Three places in this file asked this by scanning the WHOLE forest per region. On THE world that
/// is about 3 500 000 rows, so the pair is of the order of 1e13 comparisons; one of them sat at 14 GB
/// for fifteen minutes without reaching an assertion. The parent column already IS the answer.
fn parents_with_children(world: &WorldView) -> std::collections::BTreeSet<RealmId> {
    world.regions().iter().filter_map(|r| r.parent).collect()
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
    let has_children = parents_with_children(world);
    let star = world
        .regions()
        .iter()
        .filter(|r| r.parent == Some(galaxy.realm))
        .find(|r| has_children.contains(&r.realm))
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
    let planted = RealmRegions::new(regions.clone())
        .with_moving_children(vd_physics::motion::kepler_motion_fns(moving));
    (regions, planted)
}

/// How many of an anchor's children the golden records. A galaxy has a quarter of a million; the
/// placement math does not change between the eighth and the eight-thousandth, and a golden that
/// cannot be read is not evidence.
const GOLDEN_CHILDREN_PER_ANCHOR: usize = 8;

/// ★ ONE ANCHOR PER RUNG OF THE LADDER — the golden's sample (2026-08-30).
///
/// DERIVED, never named: the shallowest anchor is the ambient root, then its child, and so on down.
/// Taking the FIRST anchor at each depth in forest order makes the choice deterministic without
/// writing a seed into the test — a realm's seed is a hash avalanche, and copying one here would go
/// stale the moment the salt or the index moves.
fn golden_sample_anchors(world: &WorldView) -> Vec<RealmId> {
    // ★ WALK DOWN, DO NOT ASK EACH REALM ITS DEPTH (2026-08-30). The first spelling of this called
    // `region_depth` once per parent, and that function SCANS the whole forest on every hop — about
    // 233 220 parents over 3 500 000 rows. I wrote that quadratic myself while removing others; it
    // ran eleven minutes before I caught it. Descending from the root asks no realm anything.
    let mut children_of: std::collections::BTreeMap<RealmId, Vec<RealmId>> =
        std::collections::BTreeMap::new();
    let mut root = None;
    for r in world.regions() {
        match r.parent {
            Some(parent) => children_of.entry(parent).or_default().push(r.realm),
            None => root = Some(r.realm),
        }
    }
    // One anchor per rung: the root, then the first of its children that is itself a parent, and so
    // on inward. Forest order makes the choice deterministic without naming a seed.
    let mut anchors = Vec::new();
    let mut cur = root;
    while let Some(realm) = cur {
        let Some(kids) = children_of.get(&realm) else {
            break; // a leaf anchors nothing
        };
        anchors.push(realm);
        cur = kids.iter().copied().find(|k| children_of.contains_key(k));
    }
    anchors
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
fn own_frame_answer(point_in_star: &StampedPose, child: &RealmRegion, book: &PlacementBook) -> f64 {
    region_signed_distance(point_in_star, child, book).unwrap_or_else(|e| {
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
        let ctx = planted.author_book(star, f64::from(vd_bins::DEV.tick_hz), tick);
        for child in child_regions(&regions, star) {
            let at = ctx
                .of(child.frame)
                .expect("a star places every direct child of its own");
            let distance = placement_m(&at, ctx.anchor().tier()).length();
            let radius = child.shape.circumscribed_extent();
            eprintln!(
                "[geometry] tick {} {:?} radius {radius} m at distance {distance} m",
                tick.0, child.realm,
            );
            // THE ONE LAWFUL EXCEPTION (T2): the STAR child sits AT its system's origin and
            // holds the centre BY DESIGN — that is what a Star realm is. Every other child
            // must stand clear of the centre by more than its own reach.
            if matches!(child.realm, vd_core::pose::RealmId::Star(_)) {
                assert_eq!(distance, 0.0, "the star child sits at its system's origin");
                continue;
            }
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
        let ctx = planted.author_book(star, f64::from(vd_bins::DEV.tick_hz), tick);
        let star_centre = StampedPose::at_rest(star_frame, DVec3::ZERO, tick);
        for child in child_regions(&regions, star) {
            let own = own_frame_answer(&star_centre, &child, &ctx);
            // THE ONE LAWFUL CLAIMANT (T2): the STAR child holds its system's centre by
            // design — standing at the star's location IS being in the Star realm. It must
            // answer INSIDE; every other child must answer OUTSIDE.
            if matches!(child.realm, vd_core::pose::RealmId::Star(_)) {
                assert!(
                    own <= SURFACE_M,
                    "tick {}: the Star realm holds its own system's centre: {own}",
                    tick.0,
                );
                continue;
            }
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
        let ctx = planted.author_book(star, f64::from(vd_bins::DEV.tick_hz), tick);
        for (child, feed_pose) in
            planted.child_placements(star, f64::from(vd_bins::DEV.tick_hz), tick)
        {
            let converted = ctx
                .of(child.frame)
                .expect("a star holds a placement for every direct child of its own");
            if feed_pose.pos != converted.anchor() {
                splits.push(format!(
                    "  tick {}: {:?} — the feed ships {:?}, the conversion uses {:?}",
                    tick.0,
                    child.realm,
                    feed_pose.pos,
                    converted.anchor(),
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

/// THE GOLDEN VECTOR (placement arc S0): every anchor's child rows, bit for bit, through the shipped
/// boot. For each realm of THE world that has direct children, boot the shard exactly as its binary
/// would and record where it places every direct child at each sampled tick — every f64 as its raw
/// bits, so a change of ONE ulp anywhere on the placement path fails this diff. A byte-identity claim
/// in the placement arc is a diff of this file, never an argument. Regenerate DELIBERATELY (a slice
/// that changes the world's numbers says so) with `VD_UPDATE_GOLDEN=1`.
///
/// PROVENANCE RIDES IN THE FILE (batch review): the golden's header names WHICH world these bits pin
/// — the seed and the derived world numbers, read off the same config the boot uses — so a diff from
/// a world-numbers change NAMES the moved numbers in its own hunk, and a silent regeneration against
/// a different world is visibly a different world, not 57 indistinguishable data lines. (This golden
/// was FIRST CAPTURED inside the S4 slice that moved the numbers, so it post-dates S4 by
/// construction and is a baseline going forward, not evidence about S4 itself.)
#[test]
fn every_anchors_child_rows_match_the_golden_vector() {
    let world = the_world();
    // ★ A BOUNDED SAMPLE, ONE ANCHOR PER RUNG (2026-08-30). This walked EVERY anchor of THE world
    // and emitted a row per child per tick. That was 154 lines when a galaxy held three star
    // systems; it now holds 233 220, so the same walk asks for millions of rows and boots a world
    // per anchor. MEASURED: the test ran fifteen minutes without reaching its assertion.
    //
    // What the golden PROVES is that the placement bits do not move — and the placement math varies
    // by RUNG, not by how many siblings a rung has. So the sample takes one anchor from each depth
    // of the ladder and a bounded slice of its children. A bit that drifts at any rung still shows.
    let anchors = golden_sample_anchors(&world);
    let mut lines = Vec::new();
    // The provenance header — the WORLD IDENTITY these bits pin, from the shipped config itself.
    let config =
        vd_physics::worldgen::UniverseConfig::world(occupant_v_max_mps(), vd_bins::DEV.tick_dt);
    lines.push(
        "# THE world's placement rows, bit-for-bit, through the shipped boot (placement arc S0)."
            .to_owned(),
    );
    lines.push(format!(
        "# world: seed={SEED} true-size in-system (taxonomy flag day) ecc_cap={} n_planets={} \
         mass_draw_mearth=[{}, min({}, disc {}*M_star/N)]",
        config.planet.ecc_cap,
        config.planet.n_planets,
        config.planet.mass_lo_mearth,
        config.planet.mass_cap_mearth,
        config.planet.disc_mass_fraction,
    ));
    // The OUTER geometry identity (real-scale re-solve, owner rulings 2026-08-18: the four changed
    // numbers + the 3-D seeded placement law) — the stated world-numbers change this golden was
    // last regenerated for.
    lines.push(format!(
        "# outer: universe_r_m={} galaxy_r_m={} placement_r_m={} (3-D seeded placements, Q-B)",
        config.scale.universe_r_m, config.scale.galaxy_r_m, config.stellar.galaxy_rim_r_m,
    ));
    // THE DERIVED STELLAR CAP AND THE GALAXY'S RESERVATION — the two readings of the ONE solve that
    // moved these bits on 2026-08-21. Read off the SAME config the boot uses (the cap enters every
    // star draw as `stellar.mass_hi_msun`), so a future cap move lands in this header's own hunk
    // instead of only in 57 indistinguishable data lines.
    lines.push(format!(
        "# cap: imf_mass_hi_msun={} system_bound_max_m={} (DERIVED — the galaxy reserves its \
         largest possible child)",
        config.stellar.mass_hi_msun,
        vd_physics::worldgen::target_system_bound_max_m(),
    ));
    lines.push(
        "# Regenerate ONLY with a stated world-numbers change (VD_UPDATE_GOLDEN=1); the slice says so."
            .to_owned(),
    );
    // ★ WHY THESE BITS LAST MOVED — the audit trail the regeneration owes. Static text on purpose:
    // the NEXT regenerator must rewrite it, and the diff shows they did. Both world changes that
    // landed in commit e28c8f2 are named, INCLUDING the one that is measurably NOT a mover here —
    // "it did not move" is a claim this file has to state, not a silence to read into.
    for line in WHY_THESE_BITS_LAST_MOVED {
        lines.push((*line).to_owned());
    }
    for anchor in anchors {
        let (regions, planted) = star_shard(anchor);
        for tick in SAMPLED_TICKS {
            let ctx = planted.author_book(anchor, f64::from(vd_bins::DEV.tick_hz), tick);
            for child in child_regions(&regions, anchor)
                .into_iter()
                .take(GOLDEN_CHILDREN_PER_ANCHOR)
            {
                let at = ctx
                    .of(child.frame)
                    .expect("an anchor places every direct child of its own");
                // The FLATTENED value (bit-exact ≤ 2⁵³ cells): the golden's f64 bits survive the
                // activation untouched — the value moved INTO the integer half, and the flatten
                // reproduces it exactly, which is precisely the inertness this golden measures.
                let pos_m = placement_m(&at, ctx.anchor().tier());
                lines.push(format!(
                    "{:?} {:?} tick={} pos={:016x},{:016x},{:016x} vel={:016x},{:016x},{:016x}",
                    anchor,
                    child.realm,
                    tick.0,
                    pos_m.x.to_bits(),
                    pos_m.y.to_bits(),
                    pos_m.z.to_bits(),
                    at.velocity.x.to_bits(),
                    at.velocity.y.to_bits(),
                    at.velocity.z.to_bits(),
                ));
            }
        }
    }
    let got = lines.join("\n") + "\n";
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/goldens/placement_rows.golden"
    );
    if std::env::var_os("VD_UPDATE_GOLDEN").is_some() {
        std::fs::write(path, &got).expect("golden vector written");
        return;
    }
    let want = std::fs::read_to_string(path)
        .expect("the golden vector exists (regenerate deliberately with VD_UPDATE_GOLDEN=1)");
    assert_eq!(
        got, want,
        "\nTHE PLACEMENT ROWS MOVED. If this slice claims byte-identity it is wrong; if it \
         deliberately moves the world's numbers, regenerate with VD_UPDATE_GOLDEN=1 and say so.\n",
    );
}

/// THE RELEASE-EDGE TRIPWIRE (placement arc S0 tripwire 2, re-scoped after S4 — batch review): a
/// planet's WORST instant is its apoapsis `a(1+e)` (deliberately RESTATED here rather than read off
/// `Motion::max_excursion_m`, so a generator regression and an accessor regression are caught by
/// different tests); add the planet's own SOI **and the containment band's release outset** — the
/// edge containment actually RELEASES at, 2 m past the SOI face — and it must still sit inside the
/// system shell. GREEN since the world's numbers moved (the S4 apoapsis-solved compression,
/// D-PLACE-1; this doc used to still call the test "red until the world's numbers move" long after
/// they had — the batch review's stale-known-red class). The generator solves `apoapsis + soi` to
/// `shell − 4 m` at the ecc cap, so the bare SOI-face half holds BY CONSTRUCTION and would only fire
/// on a solve regression; the release-outset term is what this gate adds beyond the solve
/// (146 + 2 = 148 < 150 today) — the tripwire that fires if the outset ever outgrows the solve's
/// stated headroom. The shipped boot fence (`child_reaches` → `guard_regions_nest`) independently
/// judges every mover at its apoapsis; the failure here lists every escaping planet with its numbers.
#[test]
fn every_planets_apoapsis_plus_its_soi_stays_inside_its_systems_shell() {
    // ★ ONE WORLD, ONE PASS (2026-08-30). This asked the same question and paid twice for it.
    //
    // First it found the anchors by asking, for EVERY region, whether ANY region named it as a
    // parent — a full scan of the forest per region, over about 3 500 000 rows.
    //
    // Then it BOOTED A WHOLE WORLD PER ANCHOR, and scanned that world twice more for each mover.
    // On THE world that is a quarter of a million world builds. MEASURED: the test binary sat at
    // 199% CPU and 14.1 GB for fifteen minutes and had reached no assertion.
    //
    // The question needs neither. Every number it wants — a realm's own extent, its parent, and its
    // orbit — is already in the ONE world it built at the top. SL9: a lookup, never a scan.
    let world = the_world();
    let regions = world.regions();
    let extent_of: std::collections::BTreeMap<RealmId, f64> = regions
        .iter()
        .map(|r| (r.realm, r.shape.circumscribed_extent()))
        .collect();
    let cfg =
        vd_physics::worldgen::UniverseConfig::world(occupant_v_max_mps(), vd_bins::DEV.tick_dt);
    // The release outset from the SAME config the boot builds THE world with — the metres past the
    // SOI face at which containment actually lets go of an occupant.
    let release_outset_m = cfg.band.outset_m;
    let movers = vd_physics::worldgen::all_movers_for_config(SEED, &cfg);
    let mut escapes = Vec::new();
    for r in regions {
        // A mover, and the parent whose shell must contain it. A static child has no apoapsis to
        // escape with, and the ambient root has no parent to escape from.
        let (Some(parent), Some(elements)) = (r.parent, movers.get(&r.realm)) else {
            continue;
        };
        let shell = *extent_of
            .get(&parent)
            .expect("a parented region's parent is in the same forest");
        let soi = *extent_of
            .get(&r.realm)
            .expect("every region states its own extent");
        let apoapsis = elements.sma * (1.0 + elements.ecc);
        if apoapsis + soi + release_outset_m > shell {
            escapes.push(format!(
                "  {:?} under {:?}: apoapsis {:.6} m + soi {:.6} m + release outset {:.6} m \
                 = {:.6} m > shell {:.6} m",
                r.realm,
                parent,
                apoapsis,
                soi,
                release_outset_m,
                apoapsis + soi + release_outset_m,
                shell,
            ));
        }
    }
    assert_eq!(
        escapes.join("\n"),
        "",
        "\nA PLANET'S CONTAINMENT RELEASE EDGE LEAVES ITS OWN SYSTEM AT APOAPSIS:\n{}\n",
        escapes.join("\n"),
    );
}

/// THE ZERO-REACH HOLE, closed (batch review, MAJOR): a shard that does NOT host a mover's parent
/// still PLANTS that mover's row — its own row on a planet shard, an ancestor's mover elsewhere —
/// and a mover's stored centre is ZERO by construction. The reach map used to key on the hosted
/// shard's own moving roster, so that row fell to `ChildReach::Fixed(0,0,0)`: the size-only verdict
/// the fence was rebuilt to ban (`RegionNestError::NoReachForChild`'s own words call a defaulted
/// zero banned), and ONE forest got a DIFFERENT verdict per shard. `child_reaches` now derives its
/// roster from THE world for every parent the neighbourhood names, so the mover is judged at its
/// apoapsis on EVERY shard that plants it — an Excursion, never a defaulted Fixed(0).
#[test]
fn a_shard_that_does_not_host_a_movers_parent_still_judges_it_at_apoapsis() {
    let world = the_world();
    let (star, _) = a_star_and_its_children(&world);
    // The mover as its PARENT authors it: the star shard's own roster (the strict verdict's source).
    let held_star = BTreeSet::from([star]);
    let (_, star_movers) = vd_bins::boot_regions_and_movers(
        SEED,
        &held_star,
        star,
        occupant_v_max_mps(),
        vd_bins::DEV.tick_dt,
    );
    let (planet, elements) = star_movers
        .iter()
        .min_by(|a, b| a.1.sma.total_cmp(&b.1.sma))
        .map(|(realm, elements)| (*realm, *elements))
        .expect("THE world's home system authors movers");

    // The PLANET shard's own boot — the shard that does NOT host the mover's parent.
    //
    // ★ AND IT IS NAMED BY ITS PARENT (owner ruling 2026-08-30). A realm below a star system cannot
    // place itself: a planet's identifier is a one-way hash of its system's, so the seed alone never
    // finds it. Its parent says who contains it — in production through the spawn demand's
    // `RealmCoord`, here through the same lineage read off THE world.
    let held = BTreeSet::from([planet]);
    let lineage: BTreeSet<RealmId> = vd_core::worldgen::ancestor_realms(world.regions(), planet)
        .into_iter()
        .collect();
    let (regions, moving) = vd_bins::boot_regions_and_movers_in_lineage(
        SEED,
        &held,
        planet,
        occupant_v_max_mps(),
        vd_bins::DEV.tick_dt,
        &lineage,
    );
    assert!(
        moving.is_empty(),
        "the hole's precondition: a planet of THE world authors no movers of its own, so a reach \
         map keyed on ITS roster would know nothing of the planet's orbit"
    );
    let reaches = vd_bins::child_reaches(
        SEED,
        &held,
        &lineage,
        &regions,
        occupant_v_max_mps(),
        vd_bins::DEV.tick_dt,
    );
    assert_eq!(
        reaches[&planet],
        vd_core::geometry::ChildReach::Excursion(elements.sma * (1.0 + elements.ecc)),
        "the mover's own row is judged at its APOAPSIS on the shard it homes — never the defaulted \
         Fixed(0,0,0) its zeroed stored centre used to become"
    );
    // And the whole fence reaches the SAME verdict this forest gets on the parent's shard: one
    // forest, one answer, on every shard.
    vd_core::geometry::guard_regions_nest(&regions, &reaches)
        .expect("the planet shard's boot fence passes with the mover judged at apoapsis");
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
    let ctx = planted.author_book(star, f64::from(vd_bins::DEV.tick_hz), tick);

    let mut misses = Vec::new();
    for child in child_regions(&regions, star) {
        let placement = ctx
            .of(child.frame)
            .expect("a star holds a placement for every direct child of its own");
        let at_the_child =
            StampedPose::at_rest(star_frame, placement_m(&placement, star_frame.tier()), tick);
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
