//! THE BOUNDARY CONVERSION: the only place a pose changes which realm it is measured from.
//!
//! Owns: the source's half (a departing occupant's own-frame pose, expressed for the destination)
//! and the receiver's half (an arriving pose, measured into this realm's own frame), plus the
//! ancestor path the two halves walk. A conversion that cannot be measured is REFUSED with a typed
//! reason and counted, never silently placed at an approximate spot.
//!
//! Does NOT own: any absolute. Nothing here folds a realm's position from the root — every step is
//! a parent authoring the frame of a direct child, which is the only party that knows it (SL1), and
//! travel is always out into the shared parent and in again, never sibling-to-sibling (SL2).

use super::{RealmRegions, StubConfig, StubStats, arrival_book};
use vd_core::UniverseTick;
use vd_core::frame::{FrameError, transfer_frame};
use vd_core::placement::PlacementLedger;
use vd_core::pose::{FrameRef, RealmId, StampedPose, frame_for_realm};

/// Why an arriving pose could NOT be placed in this shard's frame — a typed refusal, so a hand-off that
/// cannot be measured is dropped loudly instead of being applied at a number nobody computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnplaceableArrival {
    /// This shard cannot name its OWN frame (an `Area` realm booted without its enclosing planet). It has
    /// no space to measure anything in, so it cannot accept a pose at all.
    #[error("this shard cannot name its own frame, so it has no space to place an arrival in")]
    UnnameableOwnFrame,
    /// The pose arrived in a frame that is neither this shard's own nor one of its DIRECT CHILDREN — a
    /// sibling hand-off, a grandchild, or a stranger. Nobody has told this shard where that realm sits, so
    /// there is no arithmetic it could do. This is the case that used to land silently: a pose handed
    /// planet-11 → planet-22 was accepted verbatim and simply relabelled, with no error, no counter and no
    /// log anywhere.
    #[error("the arriving pose's frame is neither mine nor one of my direct children: {0}")]
    ForeignFrame(#[from] FrameError),
    /// The pose's stamp is outside the placement window this shard retains (a stale or future-skewed
    /// sender, or a just-booted receiver whose ledger does not reach back that far). Refused loudly —
    /// the saga times out and re-drives — never placed against a book of the wrong instant.
    #[error("no authored placement book at the arriving pose's instant: {0}")]
    BookMiss(#[from] vd_core::placement::PlacementMiss),
}

/// THE RECEIVER'S HALF OF THE CONVERSION — the mirror of [`flush_pose_for_dest`], and the only place a
/// pose from another shard becomes a position in this one.
///
/// - the pose is ALREADY in my frame → accept it VERBATIM, zero arithmetic. This is the DOWNWARD case: my
///   parent placed me, and re-expressing a pose into the frame it is already in changes nothing.
/// - the pose is in one of my DIRECT CHILDREN's frames → convert: add where I put that child. This is the
///   UPWARD case, and it is the whole point — the child shipped "3 m from my centre" and only I know the
///   145 m that turns it into "148 m from mine".
/// - anything else → refuse, LOUD. Not my child, so I do not know where it is, so I have nothing to add.
///
/// This replaces a `rebind_pose_to_dest` whose `unwrap_or(pose)` swallowed every error: a sibling
/// hand-off, a stale frame, a broken ephemeris and a correct no-op all produced the same silent "keep the
/// pose as it is". Measured before the change: a pose handed from planet 11 to planet 22 landed as
/// `3.0 @ PlanetCentered{22}` with `Ok` at all three hops.
///
/// "MY frame" IS THE DESTINATION REALM'S, NOT THE SHARD'S. `to_realm` is the realm the crossing itself
/// names, and asking `config.realm` instead was the re-home loop the owner saw as a jump. On a shard
/// co-hosting a chain (System ⊃ Planet ⊃ Area) an occupant handed DOWN into the planet arrives already
/// measured from the planet's centre — the star did that subtraction, which is its job — and a receiver
/// that asked its PRIMARY realm found `PlanetCentered ≠ SystemSpace`, saw the planet among its own
/// children, and converted the pose straight back UP. Measured: shipped `-5.0 @ PlanetCentered{7}`,
/// stored `15.0 @ SystemSpace{7}`, tick after tick — the detector saw the same boundary again every time
/// and re-fired the crossing forever (15 completed sagas in a 200-tick window), while a client drawing
/// the dot in the planet's level drew it 20 m away, off the surface it was standing on.
pub(crate) fn place_arriving_pose(
    pose: StampedPose,
    to_realm: RealmId,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    stats: &mut StubStats,
) -> Result<StampedPose, UnplaceableArrival> {
    let Some(own) = arrival_frame(to_realm, config, regions) else {
        return Err(UnplaceableArrival::UnnameableOwnFrame);
    };
    if pose.frame == own {
        // THE CHILD DOES NOTHING. Not an optimisation — it is the rule: my parent already expressed this
        // in my frame, and a shard that "helpfully" re-derived its own position here would be deriving
        // where IT sits, which is exactly what it must never know.
        //
        // STATED, because this is the other end of the owner's symptom. He arrived outside the planet
        // that had accepted him; the number below is what the planet was actually handed, so the two logs
        // together say whether the parent computed it wrong or something changed it in between.
        tracing::info!(
            to = ?to_realm,
            at_tick = pose.universe_tick.0,
            accepted_at = %vd_core::pose::describe(pose.pos, pose.frame),
            accepted_len = pose
                .pos
                .delta_m(vd_core::pose::LatticePos::ORIGIN, pose.frame.tier())
                .length(),
            // Q1 CONDITION 3: a position in a log names its FRAME and its UNIT. A number of
            // metres is meaningless without the unit its integer cell counted, and at three in
            // the morning the difference between two units is the whole incident.
            frame = ?pose.frame,
            unit_m_per_cell = pose.frame.tier().cell_edge_m(),
            "ARRIVAL: accepting my parent's number as given, no arithmetic of my own",
        );
        return Ok(pose);
    }
    // The book holds the destination realm's DIRECT CHILDREN and nothing else, so this lookup is
    // itself the direct-child test: a sibling or an ancestor has no row and comes back as a typed
    // `FrameError` rather than as a number. Selected at the POSE's OWN stamp when the window retains
    // it; an instant OUTSIDE the window (forward ClockSync phase skew, or a redelivered hand-off
    // whose stamp aged past the window while the saga retried) clamps to the HEAD — the receiver
    // reads its world of NOW, counted and measured — because refusing a retried, immutably-stamped
    // hand-off forever is a wedge, not a degrade (see `arrival_book`). `BookMiss` remains only for a
    // receiver with no authored book at all (its first synced tick has not run).
    let Some((book, at)) = arrival_book(placements, to_realm, pose.universe_tick, stats) else {
        stats.placement_book_miss += 1;
        return Err(UnplaceableArrival::BookMiss(
            placements
                .at(to_realm, pose.universe_tick)
                .expect_err("the selector returned None, so the exact selection misses"),
        ));
    };
    let pose = StampedPose {
        universe_tick: at,
        ..pose
    };
    Ok(transfer_frame(&pose, own, book)?)
}

/// The frame the arriving occupant is to be measured in: the DESTINATION realm's own frame.
///
/// Taken from the roster, because that is the only lossless source (an `Area` frame carries its enclosing
/// planet's seed, which `RealmId::Area(seed)` does not). The fallback covers exactly one case — a shard
/// whose neighbourhood has not been planted yet, receiving into ITS OWN realm — where the shard can still
/// name its own frame from its own lineage. A destination this shard carries no region for and that is not
/// its own realm is unnameable: `None`, and the arrival is refused rather than placed at a guess.
/// Monomorphic, so both arms are covered here once (HR5).
#[must_use]
fn arrival_frame(
    to_realm: RealmId,
    config: &StubConfig,
    regions: &RealmRegions,
) -> Option<FrameRef> {
    if let Some(frame) = regions.hosted_frame(to_realm) {
        return Some(frame);
    }
    if to_realm != config.realm {
        return None;
    }
    frame_for_realm(config.realm, config.own_coord.parent().map(|p| p.lowered()))
}

/// THE SOURCE'S HALF OF THE CONVERSION, in the two cases that exist and no third.
///
/// Only a parent knows where its children are, so when this shard hands an entity to another realm there
/// are exactly two possibilities:
///
/// - the destination is one of MY DIRECT CHILDREN. I author where it sits, so the arithmetic is MINE:
///   subtract my child's placement and ship the pose already measured from that child's centre. That is
///   the DOWNWARD direction, and it is the only direction this shard is entitled to compute.
/// - anything else — my parent, an ancestor, a sibling. I have not been told where any of them is, so I
///   ship the dot's pose VERBATIM in MY OWN frame and let whoever does know place it. That is the UPWARD
///   direction, and it costs one copy: no context is built on this arm at all.
///
/// This used to be ONE unconditional `rebind_pose_to_dest` for both. It looked like it converted in both
/// directions; it did not. The upward arm only worked because `transfer_frame` returned
/// `UnknownDestFrame` (this shard cannot place its own parent) and the safe-degrade handed the pose back
/// untouched — the CORRECT answer, reached down the same code path a genuinely broken frame takes. So a
/// real fault and normal operation were indistinguishable, and there was nothing to count or log.
///
/// Now a failure on the child arm is a REAL contradiction — the roster says "my child", the ephemeris says
/// "unplaceable" — and it is counted, logged, and refuses the hand-off (`None` ⇒ no `SourceFlushed` ⇒ the
/// saga times out and aborts ⇒ this shard keeps authority). Monomorphic: every branch is covered once here
/// rather than once per `FrameContext` instantiation (HR5).
///
/// WHICH REALM IS DOING THE HANDING OVER: the one the POSE is measured in, not the one the shard is named
/// after. On a shard hosting a single realm those are the same question. On a shard co-hosting a chain
/// (System ⊃ Planet ⊃ Area) they are not, and asking the shard's primary realm was the second half of the
/// jump: an occupant standing on the Planet and entering the Area is placed by the PLANET, the party that
/// authored where that area sits — but the Area is a GRANDCHILD of `config.realm`, so the child lookup
/// missed, the verbatim arm fired, and the pose went out unconverted. Now the lookup is anchored on the
/// pose's own realm, so the hand-off is always exactly one link and always made by that link's parent.
///
/// The anchor is resolved from the ROSTER BY FRAME (`realm_of_frame`), never from the frame's own label:
/// a pose wearing a frame this shard carries no region for is a realm it was never told the position of,
/// and it must fall back to the primary realm so the conversion below FAILS LOUDLY instead of quietly
/// treating a stranger's frame as an anchor it could measure from.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flush_pose_for_dest(
    pose: StampedPose,
    to_realm: RealmId,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    tick: UniverseTick,
    stats: &mut StubStats,
    // Re-validate that the subject really left this realm and really reached the destination (an
    // occupant's flush) — or neither (an exterior's, whose early start ships while the hull is still
    // inside this realm and still short of the destination's shell, by design). Bitwise `&` with the
    // direction test below, so both arms are one region each (HR5).
    stale_exit_check: bool,
) -> Option<StampedPose> {
    // THE FLUSH READS THE WORLD OF NOW (the placement arc S2 — the B-1 fix). The decision to leave was
    // made ticks ago and a latched dot's stamp FROZE there, while every moving placement swept on: at
    // the measured latch→flush gap the world moved gap × v(mover) under the very departure/entry
    // decisions this function re-validates (measured 1 tick in the in-process cluster — 0.15 m at the
    // production tick rate against the containment inset, per flushed hand-off — the 7.68 m/s inner
    // planet that figure was computed from is the retired compressed geometry's). So the pose is
    // RE-STAMPED to the head book's instant — "the pose ships as re-read NOW" made true rather than
    // claimed — and every link below converts through head books of that same instant.
    let Some(head) = placements.head(config.realm) else {
        stats.placement_book_miss += 1;
        return None;
    };
    let head_at = head.at();
    let pose = StampedPose {
        universe_tick: head_at,
        ..pose
    };
    // THE ANCHOR IS THIS SHARD'S OWN REALM, and it may never be anything else.
    //
    // This used to be read off the OCCUPANT'S LABEL (`realm_of_frame(pose.frame)`). A shard's region set
    // contains its ANCESTORS as well as its own realm, so a pose still carrying an ancestor's frame
    // resolved to that ancestor — and the shard then built its conversion context anchored on a realm it
    // does not author. MEASURED on a live crossing: a shard whose own frame was `PlanetCentered{...}`
    // anchored on `System(7)`, and the placement it subtracted came back as (0,0,0) — because a context
    // anchored on somebody else's realm has no rows for that realm's children. It then handed the
    // occupant to the very realm it was itself hosting, with a zero subtraction: a realm passing somebody
    // to itself, once per tick, which is the flap.
    //
    // Under the ground rule the only frame a shard may author placements in is its own. Naming an
    // ancestor is legitimate; AUTHORING in one is not.
    let labelled = regions.realm_of_frame(pose.frame).unwrap_or(config.realm);
    if labelled != config.realm {
        stats.flush_anchor_not_own += 1;
        tracing::warn!(
            labelled = ?labelled,
            own = ?config.realm,
            "an occupant's pose names a realm this shard does not author — anchoring the hand-off on my \
             own realm, which is the only frame I may do arithmetic in",
        );
    }
    let from_realm = config.realm;
    // THE CONVERSION PATH (crossing-render slice, measured on the co-host suite): the pose starts in
    // the realm its LABEL names (roster-resolved; a stranger's label already fell back to `from_realm`
    // above) and must arrive stated for `to_realm`. On a co-hosting shard those can sit ANY number of
    // links apart — a System shard flushes a planet-standing dot into its AREA (two links down), and
    // the reverse leg lifts an area-standing dot back to the planet (one link up). Each link's
    // conversion belongs to the ONE party that authors it (SL1, both directions): ascending, the
    // link's parent ADDS the placement it authored; descending, it SUBTRACTS. Every such parent's
    // context is hosted here or the path is `None` and the pose ships VERBATIM (the single-realm
    // shard's upward hand-off, converted by the RECEIVER as today). The old one-level child lookup
    // stranded both co-host cases: descents fell to the verbatim arm where the exit re-check rightly
    // refused ("still inside my own realm"), and reverse lifts could not be expressed at all — the
    // pre-existing silent strand (the pre-§4i label-anchor converted one of the two by accident; the
    // anchor fix removed the accident without replacing the law) made loud by Stage B1.
    let start_realm = if regions.regions.iter().any(|r| r.realm == labelled) {
        labelled
    } else {
        from_realm
    };
    let path = conversion_path(regions, from_realm, start_realm, to_realm);
    let Some((ups, downs)) = path else {
        // UPWARD (or sideways): ship what I hold, VERBATIM — in whatever frame the pose already carries.
        // Nothing here rewrites the frame tag (the earlier comment claimed "tagged with my own frame";
        // measured wrong — a foreign label ships as-is, and the destination can only place its own frame
        // or a direct child's). One roster lookup, one copy.
        //
        // RE-VALIDATE THE DEPARTURE FIRST (Stage B1, owner decision 2026-08-12, §4v cure 1). The decision
        // to leave was made ticks ago; the pose ships as re-read NOW, after the freeze drain — and the
        // measured flap's second half was a departure whose occupant was back INSIDE by flush time
        // (pos 2.37 m against a 4.16 m boundary). If this shard's own region would still HOLD the re-read
        // pose — the same band rule the scan reads, hysteretic member side — the departure is no longer
        // true: refuse the flush. No `SourceFlushed` ⇒ the saga aborts PRE-commit within the abort budget
        // and thaws ⇒ this shard keeps authority and the scan re-decides. A self-crossing
        // (`to_realm == from_realm`) skips the check: "still inside myself" is not a stale departure.
        // An EXTERIOR's flush skips it by its caller's word: the early start ships while the hull is
        // still inside, by design (the ruler switch, slice 2).
        if stale_exit_check & (to_realm != from_realm) {
            // The head book IS the departure book: `from_realm` is this shard's own realm, and the
            // pose was just re-stamped to the head's own instant.
            let book = head;
            let still_held = regions
                .regions
                .iter()
                .find(|r| r.realm == from_realm)
                .is_some_and(|own_region| {
                    // THE one band question, integer form (`region_verdict`) — the SAME rule the
                    // scan asks, hysteretic member side. An unplaceable pose reads Err ⇒ not held ⇒
                    // the guard stays out of the way and the pose ships exactly as before this
                    // guard existed (the receiver stays the judge).
                    //
                    // A DEGENERATE SEGMENT (`pose.pos` as its own prior), so this asks the POINT half
                    // of the rule. That is not a shortcut: the only prior reachable here is
                    // bit-identical to the pose being validated, because a subject latched in
                    // `RequestInFlight` is skipped by the re-advance while the scan keeps rewriting
                    // its stored prior every tick. Passing that value would ship a number equal to the
                    // pose while LOOKING like it carried motion. One function, one rule, one set of
                    // arms — the fork this type's own doc forbids never appears.
                    vd_core::geometry::region_verdict(&pose, pose.pos, own_region, book, true)
                        .map(|v| v.member)
                        .unwrap_or(false)
                });
            if still_held {
                stats.flush_stale_exit += 1;
                tracing::warn!(
                    to = ?to_realm,
                    own = ?from_realm,
                    pose_frame = ?pose.frame,
                    at_tick = pose.universe_tick.0,
                    shard_tick = tick.0,
                    "HAND-OFF REFUSED: the occupant is back inside this realm — the departure is no \
                     longer true; the saga aborts and this shard keeps authority",
                );
                return None;
            }
        }
        tracing::info!(
            to = ?to_realm,
            own = ?config.realm,
            labelled = ?labelled,
            pose_frame = ?pose.frame,
            at_tick = pose.universe_tick.0,
            shard_tick = tick.0,
            pos_m = %vd_core::pose::describe(pose.pos, pose.frame),
            "HAND-OFF VERBATIM: the dest is not my direct child — shipping the pose un-converted",
        );
        return Some(pose);
    };
    // CONVERT, one authored link at a time (SL1): ascending links first (each parent ADDS the one
    // placement it authors — child frame → parent frame, in the parent's context), then descending
    // links (each parent SUBTRACTS — parent frame → child frame). A one-link descent is
    // byte-identical to the old single conversion. Both directions are ONE step list — a step is
    // "convert into `target` in `parent`'s context" — so the refusal below is one arm serving both
    // (an ascent cannot produce a conversion error on today's unrotated roster, but a spinning realm
    // must refuse loudly here, never panic — degrade-not-crash). The descending child's region is a
    // stated invariant, not a lookup that can fail: the path's children COME from this roster.
    let steps: Vec<(FrameRef, RealmId)> = ups
        .iter()
        .map(|&parent| (regions.own_frame(parent), parent))
        .chain(downs.iter().map(|&(parent, child)| {
            let child_region = regions
                .regions
                .iter()
                .find(|r| r.realm == child)
                .expect("a conversion path's children come from this same roster");
            (child_region.frame, parent)
        }))
        .collect();
    let mut placed = pose;
    let mut last: Result<(), FrameError> = Ok(());
    for &(target, parent) in &steps {
        // Every link's parent was authored this same tick by the one writer, so each head book
        // speaks at the flush's own instant; a missing one refuses the flush loudly (saga aborts,
        // this shard keeps authority) rather than converting through a placement nobody authored.
        let Ok(book) = placements.at(parent, placed.universe_tick) else {
            stats.placement_book_miss += 1;
            return None;
        };
        match transfer_frame(&placed, target, book) {
            Ok(next) => placed = next,
            Err(err) => {
                last = Err(err);
                break;
            }
        }
    }
    let dest_region = regions
        .regions
        .iter()
        .find(|r| r.realm == to_realm)
        .expect("a conversion path exists only when the destination is in this roster");
    match last {
        Ok(()) => {
            // The entry re-check below measures in the DEST's PARENT's context — the party that
            // authored the final link — exactly as a one-link flush always did. A pure ASCENT
            // (the dest is an ancestor: `downs` empty) has no entry to validate — the occupant is
            // leaving a child INTO the dest, and its exit was the scan's own decision; the dest
            // region trivially holds a pose one level inside it, so the check passes structurally.
            let Ok(book) = placements.at(
                dest_region.parent.unwrap_or(from_realm),
                placed.universe_tick,
            ) else {
                stats.placement_book_miss += 1;
                return None;
            };
            // RE-VALIDATE THE ENTRY (Stage B1, owner decision 2026-08-12, §4v cure 1). The scan decided
            // "inside" at its tick; the pose ships as re-read NOW, after the freeze drain — and the
            // measured mislandings were 12.15 / 23.00 / 20.72 m outside a 4.16 m boundary, each an
            // instant bounce-back. Ask the ONE band rule the destination's scan will ask (hysteretic
            // member side, because the arriving owner carries the owned prior): if the destination
            // would not HOLD this pose, the entry is no longer true — refuse the flush, the saga aborts
            // PRE-commit, this shard keeps authority, and a fast pass-through costs one aborted saga
            // instead of a committed mislanding and a fence-burning flap.
            // THE one band question, integer form — the SAME `region_verdict` function the
            // destination's scan will ask (hysteretic member side, because the arriving owner carries
            // the owned prior); the f64 signed distance stays as the warn gauge.
            //
            // ★ WHAT THIS GUARANTEES, EXACTLY, SINCE THE VERDICT BECAME SWEPT. The guard holds ONE
            // instant and passes a degenerate segment, so it asks the POINT half of the rule. The
            // destination's scan holds two and may sweep. So the guard is no longer the whole of what
            // the destination will decide: it still catches every mislanding where the arriving pose
            // ITSELF sits outside the band, which is the class it was built for and the class that was
            // measured — but an arrival fast enough to pass THROUGH the destination region within one
            // tick would be refused here and accepted there. There is no prior to fix that with: the
            // arriving pose is produced by walking a multi-link conversion chain, and the stored prior
            // carries neither a stamp to convert it through that chain nor a frame tag to say what it
            // was ever relative to. Ledgered rather than papered over.
            let verdict =
                vd_core::geometry::region_verdict(&placed, placed.pos, dest_region, book, true);
            let (held, sd) = verdict
                .map(|v| (v.member, v.signed_distance_m))
                .unwrap_or((false, f64::MAX));
            // ★ THE EARLY START (the ruler switch, slice 1): an exterior's flush ships while the hull
            // is still on its way to the destination's shell — the verdict was made on the LED point,
            // by design — so this re-validation is the occupant's alone, like the exit one above.
            if !held & stale_exit_check {
                stats.flush_stale_entry += 1;
                tracing::warn!(
                    to = ?to_realm,
                    own = ?from_realm,
                    signed_distance = sd,
                    at_tick = pose.universe_tick.0,
                    shard_tick = tick.0,
                    landed_len_m = placed
                        .pos
                        .delta_m(vd_core::pose::LatticePos::ORIGIN, placed.frame.tier())
                        .length(),
                    "HAND-OFF REFUSED: the occupant has left the destination — the entry is no longer \
                     true; the saga aborts and this shard keeps authority",
                );
                return None;
            }
            // THE SUBTRACTION, STATED. Once per hand-off, so it costs nothing and can stay.
            //
            // The owner flew this and reported arriving OUTSIDE the planet that had just taken him, with
            // his position "correct in the system and wrong on the planet". That is a claim about THIS
            // line: the star's own number for where its planet is, at the instant it did the arithmetic.
            // Every in-process gate says this is right; the live cluster says otherwise; so the only
            // useful thing is the actual figures from the actual run, not another argument about them.
            //
            // `child_at` is the placement subtracted at the FINAL link — the one number nobody else
            // can supply and the one that decides whether the occupant lands inside or outside.
            let child_at = book.of(dest_region.frame).map(|p| p.origin);
            // AND WHERE THE OCCUPANT'S OWN FRAME SITS. Without it the line above cannot be checked: a
            // measured run showed `from_pos` at the origin and a landing 27.85 m out when the difference
            // of the two logged vectors is 16.55, which is only possible if the pose is labelled with a
            // frame that is NOT this shard's own. That label, and where this shard puts it, are the two
            // missing numbers.
            let from_at = book.of(pose.frame).map(|p| p.origin);
            tracing::info!(
                to = ?to_realm,
                at_tick = pose.universe_tick.0,
                shard_tick = tick.0,
                from_realm = ?from_realm,
                from_frame = ?pose.frame,
                from_at = ?from_at,
                own_frame = ?regions.own_frame(config.realm),
                // FULL positions, never `offset()` — the sub-cell remainder read as a position produced
                // three retracted root causes in this arc (§4o's binding rule).
                from_pos_m = %vd_core::pose::describe(pose.pos, pose.frame),
                child_at = ?child_at,
                landed_m = %vd_core::pose::describe(placed.pos, placed.frame),
                landed_len_m = placed
                    .pos
                    .delta_m(vd_core::pose::LatticePos::ORIGIN, placed.frame.tier())
                    .length(),
                "HAND-OFF DOWN: subtracting my child's placement from the occupant's position",
            );
            Some(placed)
        }
        Err(err) => {
            stats.flush_unplaceable_child += 1;
            tracing::error!(
                %err,
                ?to_realm,
                own = ?config.realm,
                "the roster calls this realm my descendant but the ephemeris cannot place a pose in \
                 it — refusing the hand-off"
            );
            None
        }
    }
}

/// The roster's conversion path from `start` to `to`, through their nearest common ancestor:
/// `Some((ups, downs))` where `ups` lists each ASCENDING link's converting PARENT (start's parent
/// first, the common ancestor last — each ADDS the placement it authors) and `downs` lists each
/// DESCENDING link as `(parent, child)` pairs from the common ancestor down to `to` (each parent
/// SUBTRACTS). `None` when either realm is not in this shard's roster, they are the same realm, no
/// common ancestor exists within the hop cap, OR — the AUTHORSHIP constraint — any converting
/// parent is a realm this shard does not author placements for: only `own` and its roster
/// DESCENDANTS (the co-hosted chain) qualify. A roster ANCESTOR is present for naming, never for
/// arithmetic: its children's stored centres are ZERO for movers, so "adding" through it would
/// re-create the planets-on-their-star defect — the upward hand-off ships verbatim and the RECEIVER
/// (the true author) adds, exactly as SL1 states. Monomorphic; bounded by the forest depth.
#[must_use]
#[allow(clippy::type_complexity)]
pub(crate) fn conversion_path(
    regions: &RealmRegions,
    own: RealmId,
    start: RealmId,
    to: RealmId,
) -> Option<(Vec<RealmId>, Vec<(RealmId, RealmId)>)> {
    if start == to {
        return None;
    }
    // The ancestor chain of a realm, self first. Hop-capped: a malformed forest (a parent CYCLE,
    // which the boot guard rejects at boot) refuses the whole path — the verbatim arm ships and the
    // receiver judges — rather than handing the LCA search a chain that never reached a root.
    let chain = |realm: RealmId| -> Option<Vec<RealmId>> {
        let mut out = vec![realm];
        let mut cur = realm;
        for _ in 0..64 {
            let Some(region) = regions.regions.iter().find(|r| r.realm == cur) else {
                return None; // a link this shard has no region for — the verbatim arm
            };
            let Some(parent) = region.parent else {
                return Some(out); // the roster root
            };
            out.push(parent);
            cur = parent;
        }
        None // the hop cap — a cycle never reached a root, so there is no chain to speak of
    };
    let up_chain = chain(start)?;
    let down_chain = chain(to)?;
    // The nearest common ancestor: the first realm of `start`'s chain present in `to`'s chain.
    let (lca_ix_up, lca) = up_chain
        .iter()
        .enumerate()
        .find(|(_, r)| down_chain.contains(r))
        .map(|(i, r)| (i, *r))?;
    // Ascending converters: each parent from start's parent up to and including the LCA.
    let ups: Vec<RealmId> = up_chain[1..=lca_ix_up].to_vec();
    // Descending links: from the LCA down to `to`, as (parent, child) pairs.
    let lca_ix_down = down_chain.iter().position(|r| *r == lca)?;
    let mut downs: Vec<(RealmId, RealmId)> = Vec::new();
    for i in (0..lca_ix_down).rev() {
        downs.push((down_chain[i + 1], down_chain[i]));
    }
    // THE AUTHORSHIP CONSTRAINT: every converting parent must be `own` or a roster DESCENDANT of it.
    let authors = |parent: RealmId| -> bool {
        parent == own || chain(parent).is_some_and(|c| c.contains(&own))
    };
    let all_authored = ups.iter().all(|p| authors(*p)) && downs.iter().all(|(p, _)| authors(*p));
    if !all_authored {
        return None;
    }
    Some((ups, downs))
}
