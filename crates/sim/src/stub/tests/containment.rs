//! The CONTAINMENT re-home trigger section (banner 6655, running to 11217): the shell/box region forest fixtures, the ancestor chain a shard folds over its own roster, the child index proving it decides what a full scan decides without touching every child (SL9); then the deepest-container scan (rootless forest, dense crowd, first-class Station/Area and co-hosted realms, banner 9143), the shape- and motion-agnostic G-IDENTICAL fixtures, and the crossing lifecycle (banner 9817: D-38 discharge, ledger eviction, cell-carry shape, grant/abort demuxes, ttl re-drive, exhaustion backoff, demote terminal, dock/undock, boot validation). Largest module; optional further cut at the existing banner 9817 gives [9067,9816]=14 and [9817,11217]=25. Locals that travel: STATION_A, grant_realm_for_at, revoke_realm_for, child_region_shell, child_region_aabb, drive_inward_crossing_feature, child_signed_distance, drive_swept_crossing_feature, config_with_ttl, config_with_ttl_and_budget, XTTL, XBUDGET, drive_outward_exit.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

/// A LOOK-LESS CHILD STATES NOTHING (the bound/look split's structural half): the ambient
/// Universe/Galaxy rows carry no look, so a parent authoring its markers must SKIP them —
/// a marker sized by a containment bound would draw an authority promise as an object,
/// which the owner's law forbids. Measured as an absence: the look-bearing sibling is
/// stated, the look-less one is not, and the realm's own look-less arm states nothing
/// either.
#[test]
fn a_look_less_realm_states_no_body_and_states_nothing_about_its_children() {
    let mut lit = region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0);
    lit.look = Some(Boundary::Shell { r: 250.0 });
    let mut dark = region(RealmId::Station(99), Some(OWN_REALM), DVec3::ZERO, 1000.0);
    dark.look = None;
    let mut own = own_region();
    own.look = None;
    let regions = RealmRegions::new(vec![root_region(), own, lit, dark]);
    let luma = BTreeMap::new();
    let stated: Vec<RealmId> = current_bodies(&config(), &regions, &luma)
        .into_iter()
        .map(|(realm, _)| realm)
        .collect();
    // No own look ⇒ no own body; and a parent states nothing about its children, lit or dark
    // (step 5, 2026-09-04): the lit child draws itself when it runs.
    assert_eq!(stated, Vec::<RealmId>::new());
}

#[test]
fn ancestor_chain_names_self_and_every_ancestor_and_nothing_else() {
    // THE DERIVED HYSTERESIS PRIOR (task #177): being authoritatively in a realm makes you a member of
    // that realm AND of every realm containing it — and of NOTHING else. A sibling must never be
    // implied, or a subject would arrive already "inside" a realm it has never been in.
    //
    // AND THE CEILING IS GONE (SL9, 2026-08-24). This half used to assert the OPPOSITE: with 70
    // children the 70th was indexed past the `u64`'s 64 bits and its own bit was DROPPED, so its chain
    // came back as the root alone. That silent truncation is what capped a parent's child count. The
    // chain names REALMS now, so the seventieth child names itself exactly like the first — and a
    // galaxy's hundred-and-fifty-thousandth star system does too.
    let mut wide: Vec<RealmRegion> = vec![root_region()];
    for i in 0..70u64 {
        wide.push(region(
            RealmId::Station(1000 + i),
            Some(ROOT_REALM),
            DVec3::new(1.0e7 + 1.0e5 * i as f64, 0.0, 0.0),
            10.0,
        ));
    }
    let wide_rr = RealmRegions::new(wide);
    assert_eq!(
        *wide_rr.ancestor_chain_for(RealmId::Station(1069)),
        BTreeSet::from([ROOT_REALM, RealmId::Station(1069)]),
        "the seventieth child names ITSELF and its root — no index, so no width to fall off"
    );
    let regions = vec![root_region(), own_region(), child_region()];
    let rr = RealmRegions::new(regions);

    assert_eq!(
        *rr.ancestor_chain_for(ROOT_REALM),
        BTreeSet::from([ROOT_REALM])
    );
    assert_eq!(
        *rr.ancestor_chain_for(OWN_REALM),
        BTreeSet::from([ROOT_REALM, OWN_REALM])
    );
    assert_eq!(
        *rr.ancestor_chain_for(OTHER_REALM),
        BTreeSet::from([ROOT_REALM, OWN_REALM, OTHER_REALM])
    );
}

#[test]
fn ancestor_chain_excludes_a_sibling_branch() {
    // The anti-vacuity twin of the test above: with TWO children under one parent, each child's chain
    // must contain itself + the chain up, and must NOT contain the other child. A chain built by "every
    // region at or below my depth" (a plausible wrong implementation) would fail exactly here.
    let sibling = region(RealmId::Planet(43), Some(OWN_REALM), DVec3::ZERO, 1000.0);
    let rr = RealmRegions::new(vec![root_region(), own_region(), child_region(), sibling]);

    assert_eq!(
        *rr.ancestor_chain_for(OTHER_REALM),
        BTreeSet::from([ROOT_REALM, OWN_REALM, OTHER_REALM])
    );
    assert_eq!(
        *rr.ancestor_chain_for(RealmId::Planet(43)),
        BTreeSet::from([ROOT_REALM, OWN_REALM, RealmId::Planet(43)])
    );
}

#[test]
fn ancestor_chain_is_empty_for_a_realm_this_shard_does_not_host() {
    // The safe-degrade arm: a pose naming an unhosted realm resolves to NO realms, which reduces to
    // today's blank-prior behaviour rather than inventing membership. Callers treat this as a loud
    // condition, not a normal one — an unhosted realm in a pose means a rebind degraded upstream.
    let rr = RealmRegions::new(vec![root_region(), own_region()]);
    assert_eq!(
        *rr.ancestor_chain_for(RealmId::Planet(9999)),
        BTreeSet::new()
    );
    // And on a forest with no regions at all (the inert default), every lookup is empty.
    assert_eq!(
        *RealmRegions::new(vec![]).ancestor_chain_for(OWN_REALM),
        BTreeSet::new()
    );
}

#[test]
fn ancestor_chain_terminates_on_a_dangling_parent() {
    // `region_depth` stops at a dangling parent rather than hanging; the chain walk mirrors it exactly,
    // so the two caches can never disagree about the forest's shape. The boot guard rejects such a
    // forest — this only proves the walk is safe if one ever slips through.
    let orphan = region(
        RealmId::Planet(77),
        Some(RealmId::Planet(404)),
        DVec3::ZERO,
        10.0,
    );
    let rr = RealmRegions::new(vec![root_region(), orphan]);
    assert_eq!(
        *rr.ancestor_chain_for(RealmId::Planet(77)),
        BTreeSet::from([RealmId::Planet(77)]),
        "the walk stops at the dangling parent, naming only what it reached"
    );
}

#[test]
fn the_child_index_decides_exactly_what_the_full_scan_decides() {
    // THE DIFFERENTIAL PROOF for SL9's second half. The fold now asks a LOOKUP which children are worth
    // evaluating instead of walking all of them, and the only thing that makes that safe is that the
    // two decide identically. This drives a spread of subject positions through the SAME shard, once
    // with the index built (the shipped path) and once with it empty (the full scan it replaced), and
    // requires the emitted crossing destinations to match position for position.
    //
    // Anti-vacuity: the scan half is not a re-run of the same code. An EMPTY index answers for nothing,
    // so the ask list returns true for every region and every child is evaluated — which is exactly
    // the pre-slice behaviour, reached through the shipped code.
    let forest = || {
        let sibling = region(
            RealmId::Planet(43),
            Some(OWN_REALM),
            DVec3::new(4000.0, 0.0, 0.0),
            500.0,
        );
        vec![root_region(), own_region(), child_region(), sibling]
    };
    // The subject positions: inside the child, inside the sibling, in the gap between them, well
    // outside everything, and exactly on each boundary — the places a verdict can differ.
    let probes = [
        DVec3::ZERO,
        DVec3::new(999.0, 0.0, 0.0),
        DVec3::new(1001.0, 0.0, 0.0),
        DVec3::new(2500.0, 0.0, 0.0),
        DVec3::new(3600.0, 0.0, 0.0),
        DVec3::new(4000.0, 0.0, 0.0),
        DVec3::new(4499.0, 0.0, 0.0),
        DVec3::new(4501.0, 0.0, 0.0),
        DVec3::new(50_000.0, 0.0, 0.0),
        DVec3::new(0.0, 900.0, 0.0),
        DVec3::new(0.0, 0.0, 1100.0),
    ];
    let run = |indexed: bool, at: DVec3| -> Vec<RealmId> {
        let mut rig = Rig::new();
        // WITHOUT THIS THE FOLD NEVER RUNS. `evaluate_realm_boundaries` gates on the realm lease, so an
        // ungranted rig returns before the scan and BOTH halves below come back empty — a comparison
        // that could not have failed. Caught by the coverage gate reporting the skip arm as never
        // taken, which is exactly what a vacuous differential looks like from the outside.
        rig.grant_realm();
        let regions = RealmRegions::new(forest());
        *rig.world.resource_mut::<RealmRegions>() = if indexed {
            regions.with_own_realm(OWN_REALM)
        } else {
            regions
        };
        let entity = EntityId(0x5100_0001);
        insert_owned_dot_framed(&mut rig, TRIG_SESSION, entity, frame_of(OWN_REALM), at);
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..6 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        crossing_requests(&all).iter().map(|r| r.to_realm).collect()
    };
    let mut crossings_seen = 0usize;
    for at in probes {
        let indexed = run(true, at);
        crossings_seen += indexed.len();
        assert_eq!(
            indexed,
            run(false, at),
            "the lookup and the full scan disagreed for a subject at {at:?}"
        );
    }
    // NOT VACUOUS: at least one probe must actually have produced a crossing, or the loop above is
    // comparing empty against empty. This assert is here because it already caught exactly that — the
    // rig's realm lease was ungranted, the fold returned before the scan, and every comparison passed
    // while proving nothing.
    assert!(
        crossings_seen > 0,
        "no probe produced a crossing — the differential proved nothing"
    );
    // AND THE INDEX IS NOT VACUOUS: it must actually hold the two static children, or the loop above
    // would be comparing the scan with itself.
    let built = RealmRegions::new(forest()).with_own_realm(OWN_REALM);
    assert_eq!(built.child_index().indexed_len(), 2);
    assert!(built.child_index().answers_for(OTHER_REALM));
    assert!(!built.child_index().answers_for(ROOT_REALM));
    // AND THE SKIP ACTUALLY FIRES — the measurement that stops the loop above from proving nothing. At
    // the far probe the lookup names NEITHER child while answering for both, which is precisely the
    // state in which the fold declines to evaluate them. If the lookup ever answered with everything,
    // the differential loop would still pass and would mean nothing; this line fails instead.
    assert_eq!(
        built.child_index().candidates(
            LatticePos::from_metres(DVec3::new(50_000.0, 0.0, 0.0), vd_core::pose::Tier::Fine),
            vd_core::pose::Tier::Fine,
        ),
        &[] as &[RealmId],
        "a subject far outside every child must be told about none of them"
    );
    // And the near probe NEVER OMITS the child that actually holds the point. It may name more — this
    // forest's two children are comparable in size to the derived cell, so they share one, and the
    // answer here is both. That is the stated degradation, not a defect: the lookup is a SUPERSET and
    // its only hard duty is to never drop the holder. Asserting a singleton here would be asserting a
    // grid-tuning detail, and it would fail the day the band widens for a reason unrelated to this.
    assert!(
        built
            .child_index()
            .candidates(
                LatticePos::from_metres(DVec3::new(4000.0, 0.0, 0.0), vd_core::pose::Tier::Fine),
                vd_core::pose::Tier::Fine,
            )
            .contains(&RealmId::Planet(43)),
        "the lookup dropped the child that holds the point"
    );
}

#[test]
fn a_subject_this_shard_cannot_place_in_its_own_frame_still_gets_the_full_scan() {
    // THE SAFE-DEGRADE ARM of the candidate lookup. The index lives in this shard's own frame, so a
    // subject whose pose names a frame the shard's placement book cannot reach — an ancestor's frame,
    // here the ambient root's — yields NO candidates. That must NOT be read as "near nothing": the
    // fold's skip only ever drops a realm the index positively answers for, and with an unplaceable
    // point every child falls back to being evaluated. Proven the only honest way — by comparing with
    // the same subject on a shard whose index was never built.
    let forest = || vec![root_region(), own_region(), child_region()];
    let run = |indexed: bool| -> Vec<RealmId> {
        let mut rig = Rig::new();
        rig.grant_realm(); // the fold is lease-gated — see the sibling test's note
        let regions = RealmRegions::new(forest());
        *rig.world.resource_mut::<RealmRegions>() = if indexed {
            regions.with_own_realm(OWN_REALM)
        } else {
            regions
        };
        let entity = EntityId(0x5100_0002);
        insert_owned_dot_framed(
            &mut rig,
            TRIG_SESSION,
            entity,
            frame_of(ROOT_REALM),
            DVec3::ZERO,
        );
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..6 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        crossing_requests(&all).iter().map(|r| r.to_realm).collect()
    };
    assert_eq!(
        run(true),
        run(false),
        "an unplaceable subject must be decided exactly as the full scan decides it"
    );
}

#[test]
fn a_moving_child_is_left_out_of_the_index_and_stays_a_candidate() {
    // SL4's clause, made structural: a child that MOVES has no fixed centre to index, so it is omitted
    // and evaluated unconditionally — conservative, and the index never learns that anything moves. The
    // order of the two builders must not matter either, which is why both are asserted.
    // An opaque motion closure — its CONTENTS are irrelevant here and unreadable by design (SL4). The
    // index reads only the KEYS of the moving roster: whether to recompute, never what anyone reads.
    let motion = MotionFn(std::sync::Arc::new(|_| FramePlacement::identity()));
    let moving: BTreeMap<RealmId, MotionFn> = BTreeMap::from([(OTHER_REALM, motion)]);
    let forest = vec![root_region(), own_region(), child_region()];
    let a = RealmRegions::new(forest.clone())
        .with_own_realm(OWN_REALM)
        .with_moving_children(moving.clone());
    let b = RealmRegions::new(forest)
        .with_moving_children(moving)
        .with_own_realm(OWN_REALM);
    for rr in [&a, &b] {
        assert_eq!(rr.child_index().indexed_len(), 0);
        assert!(!rr.child_index().answers_for(OTHER_REALM));
    }
}

#[test]
fn a_rootless_region_forest_is_a_safe_no_op() {
    // DEFENSIVE (HR5): a NON-EMPTY forest with NO ambient root (every region has a parent — a
    // malformed set that `guard_regions_nest` rejects at boot in C-5) leaves `root_realm == None`, so
    // `evaluate_one_subject` cannot seed the `container` fold and returns EARLY — no re-home, never a
    // panic. Covers the `let Some(root_realm) = ctx.root_realm else { return cur }` guard.
    let mut rig = Rig::new();
    rig.grant_realm();
    // A single region whose `parent` is `Some(..)` ⇒ NO `parent: None` root ⇒ `root_realm == None`.
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![own_region()]);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 5);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..6 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all).is_empty(),
        "a rootless forest cannot seed the container fold ⇒ NO re-home",
    );
}

#[test]
fn the_live_containment_scan_holds_a_dense_crowd_in_one_realm_without_a_spurious_re_home() {
    // C-6c SCALE — the "hundreds in ONE location" mandate at the DETECTOR tier. The O(subjects × regions)
    // container fold runs over the WHOLE owned crowd EVERY tick; this proves it stays bounded + CORRECT
    // at crowd scale with the REAL production neighbourhood planted. (The e2e density gate
    // `p1_volume_dense_hundreds_walk_under_invariants` runs the detector INERT — empty `RealmRegions`,
    // early-return — so the live full-scan is only exercised at N ≥ 128 HERE.)
    const CROWD: usize = 128; // the "hundreds in one location" floor (N ≥ 128)
    let mut rig = Rig::new();
    rig.grant_realm();
    // The REAL seed neighbourhood `shard.rs` boots for System 7 — {Universe, Galaxy, System 7, Planet 7,
    // Station 7} (Station 7 is System 7's first-class child, task #133), a 5-region fold per subject — NOT
    // a hand-authored fixture, so this exercises the scan the bins run. The crowd clears the Station box.
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
        vd_physics::worldgen::realm_neighbourhood_for(0, RealmId::System(7)),
    );
    // Pack N dots into a tight ~3.5 m cube at the star (origin): every dot is well inside System 7 (r=40)
    // and ≥ ~16 m from Planet 7's centre (20,0,0) ⇒ its deepest container is System 7 == its owning realm
    // (no re-home). Overlapping positions are fine — a crowd IS hundreds in one place; the entities differ.
    for i in 0..CROWD {
        let offset = DVec3::new(
            (i % 5) as f64 - 2.0,
            ((i / 5) % 5) as f64 - 2.0,
            ((i / 25) % 5) as f64 - 2.0,
        );
        insert_owned_dot(
            &mut rig,
            SessionId(i as u128),
            EntityId::pack(EntityKind::Player, i as u32, 1, i as u32),
            offset,
        );
    }
    // Re-scan the whole crowd for several ticks.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 0..5 {
        rig.set_local_tick(2 + t);
        all.extend(rig.tick(vec![]));
    }
    // The scan TOUCHED every dot (membership computed for all N) — proof the full-scan RAN at scale, not
    // an inert early-return (an empty `RealmRegions` leaves this map empty).
    assert_eq!(
        rig.world.resource::<ContainmentProgress>().0.len(),
        CROWD,
        "the live containment scan evaluated all {CROWD} dots (not an inert early-return)",
    );
    // And it stayed CORRECT at scale: the whole crowd is contained in System 7 ⇒ ZERO re-homes fire.
    let spurious = crossing_requests(&all).len();
    assert_eq!(
        spurious, 0,
        "a dense crowd inside one realm triggers NO spurious re-home under the O(N×regions) scan",
    );
}

#[test]
fn a_dot_moving_into_the_station_box_re_homes_into_the_first_class_station_realm() {
    // THE task #133 headline: the SAME kind-agnostic containment detector re-homes a dot into a
    // first-class STATION realm with ZERO station-specific code (HR3). The shard OWNS System 7 and boots
    // the REAL seed neighbourhood (now {Universe, Galaxy, System 7, Planet 7, STATION 7}). A dot starts
    // at the origin (container == System 7 == owning ⇒ NO re-home), then walks into the Station BOX at
    // (-25,0,0) (a Cartesian `Aabb`, not an SOI shell) — its deepest container flips to Station 7 ≠ the
    // owning System 7, so ONE re-home fires whose `to_realm` is the Station. The box `signed_distance`
    // feeds the identical `ContainmentBand` the shells use — the Station is detected by geometry alone.
    let mut rig = Rig::new(); // owns System 7 (config().realm)
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
        vd_physics::worldgen::realm_neighbourhood_for(0, RealmId::System(7)),
    );
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 33);
    // Origin: well inside System 7 (r=40), clear of the Station box (x∈[-30,-20]) ⇒ container == System 7.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    rig.set_local_tick(2);
    let at_origin = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        at_origin.len(),
        0,
        "a dot at the origin is contained in System 7 (== owning) ⇒ NO re-home",
    );
    // Walk INTO the Station box centre — the deepest container becomes Station 7.
    let mut into: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(-25.0, 0.0, 0.0)); // the Station box centre
        into.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&into);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE re-home into the Station (latched thereafter)"
    );
    assert_eq!(
        reqs[0].to_realm, STATION_A,
        "the kind-agnostic detector re-homes into the first-class Station realm",
    );
    assert_eq!(
        reqs[0].from_realm,
        config().realm,
        "leaving the owning System 7"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
}

#[test]
fn a_station_owning_shard_re_homes_a_dot_that_leaves_the_station_back_to_system_7() {
    // The RETURN leg of the round-trip, proven as a GENUINE emission (symmetric detector, no direction):
    // a shard that OWNS Station 7 boots Station 7's seed neighbourhood — its own realm + its ANCESTOR
    // chain {System 7, Galaxy, Universe}. A dot INSIDE the Station box is contained in Station 7 (==
    // owning ⇒ NO re-home); when it LEAVES the box (to the origin, still inside System 7's r=40 SOI) its
    // deepest container becomes System 7 ≠ the owning Station 7, so ONE re-home fires whose `to_realm` is
    // System 7. This is the same machinery as the inbound gate above, run from the Station's authority —
    // the "both ways" proof that a Station is a first-class realm on the identical kind-agnostic path.
    let station_cfg = StubConfig {
        realm: STATION_A,
        held_realms: StubConfig::single_realm(STATION_A),
        frame: FrameRef::StationLocal { station_seed: 7 },
        // A REAL lineage: this shard hosts Station 7 INSIDE System 7. It used to inherit a ROOT coord
        // while declaring a Station realm — realm and lineage disagreeing — which was harmless only
        // while leaving was a search through the ancestors. Now that leaving hands UP, a shard's own
        // lineage is how it knows who to hand to, so a stub lineage means it hands to nobody.
        own_coord: child_coord_of(RealmId::System(7), STATION_A),
        ..config()
    };
    let station_frame = station_cfg.frame; // FrameRef is Copy — capture before the config move
    let mut rig = Rig::with_config(station_cfg);
    grant_realm_for(&mut rig, STATION_A);
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vd_physics::worldgen::realm_neighbourhood_for(0, STATION_A));
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 34);
    // Inside the Station box, at its CENTRE — which in the Station's OWN frame is zero, because every
    // realm is centred on itself. This used to read -25: the station's position in SYSTEM 7's frame,
    // written on a pose stamped in the STATION's frame. The two spellings agreed on nothing except
    // while the containment maths subtracted the station's position a second time and cancelled the
    // error. An occupant of a station is a few metres from its middle, whatever the station's address.
    insert_owned_dot_framed(&mut rig, TRIG_SESSION, entity, station_frame, DVec3::ZERO);
    rig.set_local_tick(2);
    let inside = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        inside.len(),
        0,
        "a dot inside the Station box is contained in Station 7 (== owning) ⇒ NO re-home",
    );
    // LEAVE the Station box, heading for the star. The station sits 25 m along -X of System 7's centre,
    // so System 7's centre is +25 in the STATION's own frame — well outside the box's ±5, still deep
    // inside System 7's 40 m reach. The dot is now inside nothing this shard holds ⇒ it has left, and
    // the destination is the shard's parent.
    let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(25.0, 0.0, 0.0)); // out of the box, toward the star
        out.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE re-home back to System 7 (latched thereafter)"
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::System(7),
        "leaving the Station re-homes back to the enclosing System 7 (symmetric, same detector)",
    );
    assert_eq!(
        reqs[0].from_realm, STATION_A,
        "leaving the owning Station 7"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
}

#[test]
fn a_dot_moving_into_the_area_box_re_homes_into_the_first_class_area_realm() {
    // The AREA analog of the Station inbound gate (task #133), proving the SAME kind-agnostic detector
    // re-homes into a first-class AREA realm — the DEEPEST region in the seed forest (depth 4). A shard
    // that OWNS Planet 7 boots Planet 7's seed neighbourhood (its own realm + ancestors {System 7,
    // Galaxy, Universe} + its child AREA 7). A dot starts at Planet 7's centre (20,0,0) (container ==
    // Planet 7 == owning ⇒ NO re-home — this is ALSO the escape-SOI probe point, which must NOT resolve
    // to the Area), then walks into the Area BOX at (25,0,0) — its deepest container flips to Area 7 ≠
    // the owning Planet 7, so ONE re-home fires whose `to_realm` is the Area. Zero area-specific code.
    let planet_cfg = StubConfig {
        realm: RealmId::Planet(7),
        held_realms: StubConfig::single_realm(RealmId::Planet(7)),
        frame: FrameRef::PlanetCentered { planet_seed: 7 },
        ..config()
    };
    let planet_frame = planet_cfg.frame; // FrameRef is Copy — capture before the config move
    let mut rig = Rig::with_config(planet_cfg);
    grant_realm_for(&mut rig, RealmId::Planet(7));
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(
        vd_physics::worldgen::realm_neighbourhood_for(0, RealmId::Planet(7)),
    );
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 35);
    // Planet 7's centre, which in the PLANET'S OWN frame is zero — every realm is centred on itself.
    // Inside Planet 7 (r=10) and OUTSIDE the Area box (which sits at +5 from the planet, half 3, so
    // x∈[2,8]) ⇒ container == Planet 7 == owning ⇒ NO re-home. This used to read (20,0,0): the planet's
    // position in SYSTEM 7's frame, written on a pose stamped in the planet's own frame.
    insert_owned_dot_framed(&mut rig, TRIG_SESSION, entity, planet_frame, DVec3::ZERO);
    rig.set_local_tick(2);
    let at_centre = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        at_centre.len(),
        0,
        "the dot at Planet 7's centre is contained in Planet 7 (== owning) ⇒ NO re-home",
    );
    // Walk INTO the Area box centre — +5 along X in the PLANET's own frame, still well inside the
    // planet's 10 m sphere and squarely in the box ⇒ the deepest container becomes Area 7.
    let mut into: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(5.0, 0.0, 0.0)); // the Area box centre
        into.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&into);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE re-home into the Area (latched thereafter)"
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::Area(7),
        "the kind-agnostic detector re-homes into the first-class Area realm (the deepest region)",
    );
    assert_eq!(
        reqs[0].from_realm,
        RealmId::Planet(7),
        "leaving the owning Planet 7"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
}

#[test]
fn a_cohosting_shard_affirms_the_child_realm_head_into_its_own_authority_map() {
    // The multi-realm AFFIRM path (co-hosting): a shard co-hosting Planet 7 must land the Planet-7
    // realm-head reply in `CoHostedAuthority` (independent of the primary `RealmAuthority`), so the
    // short-circuit's `held_here` answers `Some` for the child. This is the grant/affirm half of the cure.
    let mut rig = Rig::with_config(cohost_planet_config());
    grant_realm_for(&mut rig, RealmId::System(7));
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "the primary System-7 realm is held on RealmAuthority",
    );
    assert!(
        !rig.world
            .resource::<CoHostedAuthority>()
            .0
            .contains_key(&RealmId::Planet(7)),
        "the child is NOT held until its own head affirms",
    );
    grant_realm_for(&mut rig, RealmId::Planet(7));
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(1)),
        "the co-hosted Planet-7 head lands in CoHostedAuthority (not RealmAuthority)",
    );
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "the child affirm leaves the primary realm untouched",
    );
    // RE-AFFIRM the SAME child at a NEWER fence: the child is ALREADY in `CoHostedAuthority`, so this hits
    // the `else if ours` REFRESH arm (not the first insert). The stored fence must update to the refreshed
    // value — the periodic round-trip re-arming a still-held co-hosted child head.
    grant_realm_for_at(&mut rig, RealmId::Planet(7), Fence(2));
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(2)),
        "the re-affirm REFRESHES the co-hosted Planet-7 fence to the newer value",
    );
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "the child re-affirm still leaves the primary realm untouched",
    );
}

#[test]
fn a_revoked_cohosted_child_head_is_dropped_from_the_cohost_authority_map() {
    // The FOREIGN/None `else` arm of `affirm_realm_head` (`cohosted.0.remove(&realm)`): a co-hosted CHILD
    // realm this shard held is TAKEN OVER or REVOKED (here `record: None` — the reaper dropped the lease),
    // so its `CoHostedAuthority` entry is DROPPED (no transient loss — a child never anchors this shard's
    // transients; those ride the PRIMARY `RealmAuthority`). This is distinct from the primary self-fence
    // (which drops `RealmAuthority` + declares transients lost). Boot with Planet 7 HELD, then revoke it.
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig); // primary System 7 + co-hosted child Planet 7 both held
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(1)),
        "precondition: the co-hosted Planet-7 head is held",
    );
    revoke_realm_for(&mut rig, RealmId::Planet(7)); // record None ⇒ not ours ⇒ the remove arm
    assert!(
        !rig.world
            .resource::<CoHostedAuthority>()
            .0
            .contains_key(&RealmId::Planet(7)),
        "the revoked co-hosted child is DROPPED from CoHostedAuthority",
    );
    // The PRIMARY realm is UNTOUCHED — the child-revoke path never self-fences the primary lease.
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "revoking the co-hosted child leaves the primary System-7 realm held",
    );
}

#[test]
fn a_dot_re_homing_into_a_cohosted_child_emits_a_crossing_request_with_its_parent() {
    // THE UNIVERSAL re-home assertion (task #149, re-baselined from the old relabel test): a durable dot
    // that walks from System 7 into CO-HOSTED Planet 7's SOI emits the SAME `CrossingRequest` as a
    // foreign crossing — there is NO local short-circuit. `head(Realm(Planet 7))` resolves to THIS node
    // (source==dest), which the ONE orchestrator saga handles as the degenerate case. The request carries
    // Planet 7 as `to_realm` and its enclosing System 7 as `to_parent` — a WIRE-SHAPE pin only: the field
    // is DEAD (its consumer `rebind_pose_to_dest` is deleted, D-PLACE-1/D-WIRE-1; the dest forms the child
    // frame from its own ROSTER via `arrival_frame`). The pose is NOT rewritten in place here (the
    // detector only requests); the frame flips at the dest's adopt (same node on a co-hosted re-home).
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 51);
    // Start at the origin — inside System 7 (r=40), OUTSIDE Planet 7 (centre 20, r=10) ⇒ container ==
    // System 7 == owning ⇒ NO re-home.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    rig.set_local_tick(2);
    let at_origin = crossing_requests(&rig.tick(vec![]));
    assert_eq!(
        at_origin.len(),
        0,
        "at the origin the dot is in System 7 (== owning)"
    );
    // Walk INTO Planet 7's centre (20,0,0) — its deepest container flips to Planet 7, a realm THIS
    // shard CO-HOSTS ⇒ ONE CrossingRequest (the uniform saga; a co-hosted dest is source==dest).
    let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..8 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(20.0, 0.0, 0.0));
        out.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "a re-home into a CO-HOSTED child emits ONE CrossingRequest (source==dest — the uniform saga)",
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::Planet(7),
        "the crossing targets the co-hosted Planet 7 realm",
    );
    assert_eq!(
        reqs[0].to_parent,
        Some(RealmId::System(7)),
        "the request carries Planet 7's enclosing System 7 as to_parent (so the dest's Area/child frame forms)",
    );
}

#[test]
fn a_dot_re_homing_into_a_non_cohosted_child_emits_a_crossing_request() {
    // The SAME co-hosting shard, a dot walking into Station 7 — a child it does NOT co-host (`held_realms`
    // is `{System(7), Planet(7)}`, no Station). Post-task-#149 this is IDENTICAL to the co-hosted case:
    // ONE `CrossingRequest` (the node-placement branch is deleted — held-here vs foreign no longer
    // matters, both are the uniform saga). Kept as a second geometry to prove the request fires for any
    // container change, co-hosted or not.
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig);
    // The region union for {System 7, Planet 7} DOES include Station 7 (a child of the held System 7).
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 52);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO);
    let mut out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..9 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(-25.0, 0.0, 0.0)); // Station 7 box centre
        out.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "a re-home into a NON-co-hosted child emits ONE CrossingRequest (the uniform saga)",
    );
    assert_eq!(
        reqs[0].to_realm,
        RealmId::Station(7),
        "the crossing targets the Station 7 realm",
    );
    assert_eq!(
        reqs[0].to_parent,
        Some(RealmId::System(7)),
        "the request carries Station 7's enclosing System 7 as to_parent",
    );
}

#[test]
fn a_held_transient_re_homing_into_a_cohosted_child_emits_a_transient_request_with_its_parent() {
    // The TRANSIENT twin (HR2 — the SAME machinery, no per-kind fork, no local short-circuit): a held
    // Debris transient inside CO-HOSTED Planet 7 emits ONE `TransientCrossingRequest` carrying Planet 7's
    // enclosing System 7 as `to_parent` — a WIRE-SHAPE pin only (the field is DEAD: its consumer is
    // deleted, D-PLACE-1/D-WIRE-1; the dest places the pose from its own roster at adopt). The
    // pose is not rewritten in place; the frame flips at the dest's adopt.
    let mut rig = Rig::with_config(cohost_planet_config());
    boot_cohost_planet(&mut rig);
    let entity = EntityId::pack(EntityKind::Debris, 10, 1, 61);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            // Inside Planet 7's SOI (centre x=20, r=10) — container == Planet 7 (co-hosted).
            pose: StampedPose::at_rest(
                config().frame,
                DVec3::new(20.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::from_metres(
                DVec3::new(20.0, 0.0, 0.0),
                vd_core::pose::Tier::Fine,
            ),
        },
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..7 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = transient_crossing_requests(&all);
    assert_eq!(
        reqs.len(),
        1,
        "a transient re-home into a CO-HOSTED child emits ONE TransientCrossingRequest (the uniform saga)",
    );
    assert_eq!(reqs[0].to_realm, RealmId::Planet(7));
    assert_eq!(
        reqs[0].to_parent,
        Some(RealmId::System(7)),
        "the transient request carries Planet 7's enclosing System 7 as to_parent",
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transient_crossings_requested,
        1,
    );
}

#[test]
fn an_aborted_re_home_re_fires_while_the_dot_is_still_in_the_region() {
    // POSITIVE proof of the abort RE-FIRE (`on_crossing_aborted` resets `last_commit_tick=None`): a dot
    // whose crossing ABORTED is STILL geometrically in the deeper region, so `container != owning` and
    // `should_rehome` fires AGAIN with a FRESH id — the re-home self-heals without a physical re-cross.
    // A green tree that DISARMS the detector during the abort (the e2e tests) cannot catch a broken
    // re-fire; this drives the abort WITH the regions still armed and asserts the attempt-1 re-emit.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::ZERO); // inside the child ⇒ re-homes
    rig.set_local_tick(2);
    let first = crossing_requests(&rig.tick(vec![]));
    assert_eq!(first.len(), 1, "the in-region dot re-homes once");
    assert_eq!(first[0].attempt, 0, "the first re-home is attempt 0");
    // The MATCHING abort clears the latch + resets the cooldown so the re-home can re-fire.
    let transfer = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
    let mut after: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    rig.set_local_tick(3);
    after.extend(rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Entity(entity),
            transfer,
        }),
    )]));
    rig.set_local_tick(4);
    after.extend(rig.tick(vec![])); // still in the region ⇒ re-fires with a fresh id
    let refired = crossing_requests(&after).iter().any(|r| r.attempt == 1);
    assert!(
        refired,
        "after the abort the still-in-region dot RE-FIRES with attempt==1 (self-heal)",
    );
}

#[test]
fn a_dot_jittering_across_a_region_surface_within_the_band_never_re_homes() {
    // The BAND (not the latch) proves anti-flap. A dot whose signed distance to the child region
    // oscillates ACROSS the surface (sd ∈ [-40, +40]) but stays inside the acquire-hysteresis dead-zone
    // (acquire only at sd ≤ -inset = -50) NEVER acquires membership, so `container` stays the OWN realm
    // and ZERO re-homes fire — latch-INDEPENDENT (no re-home ⇒ no latch to mask a flap). A broken band
    // (naive point membership `sd ≤ 0`) would acquire the child on every sd<0 tick and re-home.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig); // child (Planet 42) at origin, r=1000, band inset=50/outset=100
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 11);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(1040.0, 0.0, 0.0)); // sd_child = +40
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    // Jitter x ∈ [960, 1040] ⇒ sd_child ∈ [-40, +40], crossing the surface but never reaching -inset,
    // for well over `k_dwell` ticks.
    for (i, &x) in [960.0, 1040.0, 970.0, 1030.0, 965.0, 1035.0, 962.0, 1038.0]
        .iter()
        .enumerate()
    {
        rig.set_local_tick(2 + i as u64);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(x, 0.0, 0.0));
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all).is_empty(),
        "the acquire-hysteresis dead-zone keeps the dot a non-member ⇒ ZERO re-homes (band, not latch)",
    );
}

#[test]
fn an_empty_registry_or_no_realm_triggers_nothing() {
    // (a) EMPTY region registry (the production-inert path): no candidates, no emit, no state.
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 5);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert!(crossing_requests(&all).is_empty());
    assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
    assert!(rig.world.resource::<CrossingProgress>().0.is_empty());

    // (b) NO realm authority (never granted): the authority gate short-circuits BEFORE any work —
    // even with a populated registry + an in-band dot. A NON-ZERO `request_ttl_ticks` here so the
    // 3f-D4 `redrive_stranded_crossings` scan ALSO reaches (and covers) its authority-gate `else`
    // arm (past its own `ttl==0` early-return), mirroring the trigger's gate.
    let mut rig2 = Rig::with_config(config_with_ttl(3));
    plant_dock_regions(&mut rig2);
    // A dot cannot normally exist without a realm, but the trigger's gate must be authority-first:
    // force one in and confirm the early-return fires.
    insert_owned_dot(&mut rig2, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    assert_eq!(rig2.world.resource::<RealmAuthority>().0, None);
    let mut all2: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig2.set_local_tick(t);
        all2.extend(rig2.tick(vec![]));
    }
    assert!(
        crossing_requests(&all2).is_empty(),
        "no realm authority ⇒ the trigger + the ttl re-drive both early-return",
    );
    assert!(rig2.world.resource::<CrossingProgress>().0.is_empty());
    assert_eq!(
        rig2.world.resource::<StubStats>().crossings_redriven,
        0,
        "no realm authority ⇒ the ttl scan never re-drives",
    );
}

#[test]
fn eviction_drops_state_for_a_departed_subject() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 6);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    // Deep inside the child ⇒ a re-home fires, so ALL FOUR per-entity ledgers hold a row (the
    // ContainmentProgress bitset is written by the full-scan every tick a subject is evaluated).
    for t in 2..8 {
        rig.set_local_tick(t);
        let _ = rig.tick(vec![]);
    }
    assert!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<ContainmentProgress>()
            .0
            .contains_key(&entity),
        "the per-region membership bitset holds a row for the evaluated subject",
    );
    // The dot logs out (removed): next evaluation tick evicts its per-entity state (the DRY retain).
    rig.world.resource_mut::<Dots>().0.remove(&TRIG_SESSION);
    rig.set_local_tick(8);
    let _ = rig.tick(vec![]);
    assert!(
        !rig.world
            .resource::<CrossingProgress>()
            .0
            .contains_key(&entity)
    );
    assert!(
        !rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert!(
        !rig.world
            .resource::<ContainmentProgress>()
            .0
            .contains_key(&entity)
    );
}

/// ★ SLICE S5's ACCEPTANCE LINE, AT THE SCAN. A subject that clears a child's WHOLE DIAMETER inside
/// one tick still acquires it and still re-homes into it. Before the verdict tested the tick's motion
/// this was impossible in principle, not merely unlikely: acquisition needed a SAMPLE landing at least
/// one inset INSIDE the surface, so no widening of any band could ever buy it.
///
/// The anti-vacuity guarantee is stated as a measurement rather than assumed: BOTH endpoint poses are
/// asserted to sit outside the child, so the point rule cannot have decided this.
#[test]
fn a_subject_that_clears_a_whole_child_in_one_tick_still_re_homes_into_it() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let child = child_region(); // a 1000 m shell at the origin
    let to_realm = child.realm;
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child]);
    let entity = EntityId::pack(EntityKind::Player, 11, 1, 0x4B);

    // Tick 1 — well outside on one side. This is the tick that RECORDS the prior.
    let before = DVec3::new(-9_000.0, 0.0, 0.0);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, before);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);

    // Tick 2 — well outside on the OTHER side, 18 km later: nine child diameters in one tick.
    let after = DVec3::new(9_000.0, 0.0, 0.0);
    rig.set_local_tick(3);
    move_dot(&mut rig, TRIG_SESSION, after);
    let sent = rig.tick(vec![]);

    // ANTI-VACUITY: neither sample is inside the child, so a point rule decides "never a member".
    assert!(
        child_signed_distance(&child, before) > 0.0,
        "the prior sample must be outside the child"
    );
    assert!(
        child_signed_distance(&child, after) > 0.0,
        "the current sample must be outside the child"
    );

    let reqs: Vec<CrossingRequest> = crossing_requests(&sent)
        .into_iter()
        .filter(|r| r.to_realm == to_realm)
        .collect();
    assert_eq!(
        reqs.len(),
        1,
        "the tick's motion crossed the child, so exactly one re-home into it is owed"
    );
}

/// AND IT IS NOT A BLANKET YES. The same jump, offset so the path misses the child, must produce
/// nothing — otherwise the arm above would pass for a verdict that simply said "member" at speed.
#[test]
fn the_same_jump_one_child_radius_off_the_path_re_homes_nowhere() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let child = child_region();
    let to_realm = child.realm;
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child]);
    let entity = EntityId::pack(EntityKind::Player, 12, 1, 0x4C);
    // Same 18 km jump, displaced 2 km off-axis — outside the 1000 m child at every point.
    let before = DVec3::new(-9_000.0, 2_000.0, 0.0);
    let after = DVec3::new(9_000.0, 2_000.0, 0.0);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, before);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    rig.set_local_tick(3);
    move_dot(&mut rig, TRIG_SESSION, after);
    let sent = rig.tick(vec![]);
    let reqs: Vec<CrossingRequest> = crossing_requests(&sent)
        .into_iter()
        .filter(|r| r.to_realm == to_realm)
        .collect();
    assert!(
        reqs.is_empty(),
        "a path that misses the child must acquire nothing: {reqs:?}"
    );
}

/// The HR4/G-IDENTICAL extension proof (the placement arc S6): the IDENTICAL swept-crossing
/// fixture, run with a KEPLER child and an INTEGRATED (ballistic burn) child, on TWO shard kinds —
/// each run commits exactly ONE re-home with identical subject/source/destination/attempt. The
/// motions are built in the motion crate (a dev-dependency: fixtures may plant real motion; the
/// shipped crate cannot name one) and enter ONLY as opaque closures, so the fixture is measurably
/// blind to HOW the child moves and to WHAT KIND of shard runs it.
#[test]
fn a_kepler_child_and_a_thrusting_child_cross_by_identical_code() {
    use vd_physics::motion::{Motion, motion_fn};
    let tick_hz = 20.0;
    // KEPLER: a circular 1000 m orbit phased so the child sits far off the dot at tick 100
    // (√2·1000 m away) and EXACTLY on it at tick 160 (M(8 s) ≡ 0 ⇒ position (1000, 0, 0)).
    let n = std::f64::consts::FRAC_PI_2 / 3.0; // rad/s across the 3 s sweep window
    let kepler = motion_fn(Motion::Kepler(OrbitalElements {
        sma: 1000.0,
        ecc: 0.0,
        inclination: 0.0,
        raan: 0.0,
        arg_periapsis: 0.0,
        mean_anomaly_epoch: -n * (160.0 / tick_hz),
        central_mass: n * n * 1000.0_f64.powi(3) / vd_physics::celestial::G,
    }));
    // INTEGRATED: a ballistic burn along -Y that carries the child from 1200 m off the dot at
    // tick 100 onto it at tick 160 (y(t) = 1600 + 120·t − 40·t²: y(5) = 1200, y(8) = 0).
    let burn = motion_fn(Motion::Integrated {
        placement: FramePlacement::moving(
            DVec3::new(1000.0, 1600.0, 0.0),
            DVec3::new(0.0, 120.0, 0.0),
        ),
        acceleration: DVec3::new(0.0, -80.0, 0.0),
    });

    let (k_reqs, k_start, k_end) = drive_swept_crossing_feature(kepler, NodeKind::StubShard);
    let planet = crate::capability::profiles::planet().expect("planet profile");
    let (b_reqs, b_start, b_end) = drive_swept_crossing_feature(burn, NodeKind::Shard(planet));

    // Anti-vacuity: both children genuinely swept from far OUTSIDE the shell onto the dot.
    assert!(k_start > 200.0);
    assert!(b_start > 200.0);
    assert!(k_end < 1.0);
    assert!(b_end < 1.0);
    // Exactly ONE re-home each — the latch suppressed every later tick of the dwell.
    assert_eq!(k_reqs.len(), 1, "the Kepler sweep commits one re-home");
    assert_eq!(b_reqs.len(), 1, "the burn sweep commits one re-home");
    // IDENTICAL crossing, field by field: the code cannot tell an orbit from a burn, and cannot
    // tell a stub shard from a planet-profile shard (HR4's G-IDENTICAL, both axes at once).
    assert_eq!(k_reqs[0].subject, b_reqs[0].subject);
    assert_eq!(k_reqs[0].from_realm, b_reqs[0].from_realm);
    assert_eq!(k_reqs[0].to_realm, b_reqs[0].to_realm);
    assert_eq!(k_reqs[0].attempt, b_reqs[0].attempt);
    assert_eq!(k_reqs[0].session, b_reqs[0].session);
}

/// DISCHARGES D-38: the G-IDENTICAL assert_feature_anywhere — ONE containment re-home fixture,
/// identical whether the deeper child region is a Spherical (Shell) or a Cartesian (Aabb) volume.
#[test]
fn assert_feature_anywhere() {
    // Run (a): a Spherical PLANET profile ⇔ a Shell child region (SOI descent). The dot descends the
    // diagonal across the shell's acquire edge and commits exactly ONE re-home to OTHER_REALM.
    let planet = crate::capability::profiles::planet().expect("planet profile");
    assert_eq!(
        planet.voxel(),
        Some(crate::capability::VoxelGeometry::Spherical),
        "the Shell run is tied to the Spherical profile",
    );
    let (shell_reqs, shell_start_sd, shell_end_sd, shell_shape) =
        drive_inward_crossing_feature(NodeKind::Shard(planet), child_region_shell);
    // Spherical ⇔ Shell: the run's region IS a Shell (compared by equality, not `matches!`, so there
    // is no uncoverable false arm — HR5(d)). `child_region` uses r 1000.
    assert_eq!(shell_shape, Boundary::Shell { r: 1000.0 });
    // Run (b): a Cartesian STATION profile ⇔ an Aabb child region (station volume). The IDENTICAL body
    // drives the SAME diagonal descent across the box's acquire edge → exactly ONE re-home.
    let station = crate::capability::profiles::station().expect("station profile");
    assert_eq!(
        station.voxel(),
        Some(crate::capability::VoxelGeometry::Cartesian),
        "the Aabb run is tied to the Cartesian profile",
    );
    let (aabb_reqs, aabb_start_sd, aabb_end_sd, aabb_shape) =
        drive_inward_crossing_feature(NodeKind::Shard(station), child_region_aabb);
    // Cartesian ⇔ Aabb: the run's region IS an Aabb (equality, no `matches!` false arm — HR5(d)).
    assert_eq!(
        aabb_shape,
        Boundary::Aabb {
            half: DVec3::new(200.0, 200.0, 200.0),
        }
    );

    // (i) EXACTLY ONE re-home CrossingRequest per run, whose destination is the child's realm — the
    // box/shell actually GATED the re-home (not a bare count of "something fired").
    assert_eq!(
        shell_reqs.len(),
        1,
        "the Shell run emits exactly one re-home"
    );
    assert_eq!(aabb_reqs.len(), 1, "the Aabb run emits exactly one re-home");
    assert_eq!(
        shell_reqs[0].to_realm, OTHER_REALM,
        "the Shell re-home gated to OTHER_REALM"
    );
    assert_eq!(
        aabb_reqs[0].to_realm, OTHER_REALM,
        "the Aabb re-home gated to OTHER_REALM"
    );
    // IDENTICAL feature behavior across the two profiles: same subject, same source realm, same
    // destination, same attempt. The two runs differ ONLY in region shape, never in the feature.
    assert_eq!(shell_reqs[0].subject, aabb_reqs[0].subject);
    assert_eq!(shell_reqs[0].from_realm, aabb_reqs[0].from_realm);
    assert_eq!(shell_reqs[0].to_realm, aabb_reqs[0].to_realm);
    assert_eq!(shell_reqs[0].attempt, aabb_reqs[0].attempt);
    assert_eq!(shell_reqs[0].session, aabb_reqs[0].session);

    // (ii) Anti-vacuity: the signed distance ACTUALLY crossed the acquire edge in BOTH runs — the dot
    // started OUTSIDE the surface (sd > 0) and ended at least `inset` (50 m) INSIDE (sd <= -inset), a
    // real membership acquire, never a no-op that fired on an already-inside dot.
    assert!(
        shell_start_sd > 0.0,
        "Shell start OUTSIDE the surface (sd {shell_start_sd})"
    );
    assert!(
        shell_end_sd <= -50.0,
        "Shell end at least the inset INSIDE (sd {shell_end_sd})"
    );
    assert!(
        aabb_start_sd > 0.0,
        "Aabb start OUTSIDE the surface (sd {aabb_start_sd})"
    );
    assert!(
        aabb_end_sd <= -50.0,
        "Aabb end at least the inset INSIDE (sd {aabb_end_sd})"
    );
}

/// Slice 4a: the ledger-eviction-on-leave gate (the InputLog-leak class). The `retain_live` machinery
/// already exists (`stub.rs` `evaluate_realm_boundaries`); this PROVES all THREE per-entity ledgers
/// evicted by the detector (CrossingProgress + RequestInFlight + ContainmentProgress — the C-3 bitset
/// is the 4th `retain_live` monomorphization that REPLACES the deleted `InterestZones`) are evicted
/// when the subject leaves. Uses `assert_eq!(.get(), None)` (not `assert!(matches!)`) so the None
/// equality is the covered arm.
#[test]
fn slice4a_all_three_ledgers_evict_when_the_subject_leaves() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4B);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    // Deep inside the child ⇒ a durable re-home commits, so CrossingProgress (cooldown), RequestInFlight
    // (latch) AND ContainmentProgress (bitset) all hold a row for the subject.
    for t in 2..8 {
        rig.set_local_tick(t);
        let _ = rig.tick(vec![]);
    }
    // All three ledgers now hold an entry (setup pre-conditions; the load-bearing asserts are the
    // three `None` equalities after the subject leaves).
    assert!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert!(
        rig.world
            .resource::<ContainmentProgress>()
            .0
            .contains_key(&entity)
    );
    // The dot logs out (removed from Dots). Not in OwnedTransients either, so `live` excludes it.
    rig.world.resource_mut::<Dots>().0.remove(&TRIG_SESSION);
    rig.set_local_tick(8);
    let _ = rig.tick(vec![]);
    // All THREE per-entity ledgers no longer contain the subject — split into three `is_none()`
    // asserts (HR5: never a collapsed `assert!(a && b)`), equality-form (not `matches!`).
    assert_eq!(
        rig.world.resource::<CrossingProgress>().0.get(&entity),
        None
    );
    assert_eq!(rig.world.resource::<RequestInFlight>().0.get(&entity), None);
    assert_eq!(
        rig.world.resource::<ContainmentProgress>().0.get(&entity),
        None
    );
}

/// Slice 4a (adversary MEDIUM-3 — the cell-carry SHAPE assertion, NOT an integer-gate flip). Proves
/// the dot's `LatticePos.cell` (a DISTINCT NON-ZERO cell) is CARRIED UNCHANGED through the trigger
/// AND the offset-based crossing still fires. This proves the cell is PRESERVED (shape only). It does
/// NOT prove the "integer in/out DECISION flips across a cell boundary": `should_commit` /
/// `evaluate_one_subject` read ONLY `LatticePos.offset()` today (`stub.rs`), never comparing the cell,
/// so a cross-cell in/out DECISION test is BLOCKED on the P4/P5 rebase math (D-41) and is deliberately
/// NOT written here.
#[test]
fn slice4a_nonzero_cell_is_carried_unchanged_and_the_offset_crossing_still_fires() {
    use vd_core::glam::I64Vec3;
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 0x4C);
    // A DISTINCT non-zero cell anchor. The trigger reads only `offset()`, so the crossing decision is
    // driven by the (in-band) offset exactly as if the cell were zero.
    let cell = I64Vec3::new(5, -7, 11);
    let offset_inside = DVec3::new(100.0, 0.0, 0.0); // sd = 100 - 1000 ≪ -inset ⇒ inside the child
    rig.world.resource_mut::<Dots>().0.insert(
        TRIG_SESSION,
        Dot {
            last_stick: None,
            entity,
            account: AccountId(1),
            session_fence: Fence(1),
            gateway: GATEWAY,
            granted: true,
            input_active: false,
            adopting: false,
            authority: Authority::Owned { fence: Fence(1) },
            departing: false,
            entity_fence: Fence(1),
            pose: StampedPose {
                // Plant the non-zero cell anchor on an otherwise-rest pose (avoids naming DQuat).
                pos: LatticePos::at(cell, offset_inside),
                ..StampedPose::at_rest(config().frame, offset_inside, UniverseTick(100))
            },
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            look_extent_m: vd_core::look::OCCUPANT_FIGURE_EXTENT_M,
            prev_offset: LatticePos::from_metres(offset_inside, vd_core::pose::Tier::Fine),
        },
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    // The OFFSET-based crossing still fires: exactly one inward CrossingRequest.
    assert_eq!(
        crossing_requests(&all).len(),
        1,
        "the offset-based crossing fires regardless of the (non-zero) cell anchor",
    );
    // The non-zero cell is CARRIED UNCHANGED through the trigger — PROVING the cell is PRESERVED
    // (shape). (It is NOT compared in the in/out decision — that cell-aware gate is P4/P5 D-41 math.)
    assert_eq!(
        rig.world
            .resource::<Dots>()
            .0
            .get(&TRIG_SESSION)
            .expect("the owned dot")
            .pose
            .pos
            .cell(),
        cell,
        "the non-zero LatticePos.cell is carried through the trigger unchanged (SHAPE preserved)",
    );
}

#[test]
fn the_grant_demux_flips_a_source_transient_to_crossing() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 10, 1, 7);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: transient_pose().pos,
        },
    );
    let grant = TransientCrossingGrant {
        subject: DirectoryKey::Entity(entity),
        dest: DEST_NODE,
        to_realm: OTHER_REALM,
        dst_realm_fence: Fence(3),
        batch: TransferId(77),
        to_parent: None,
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(grant),
    )]);
    // The grant flipped Held → Crossing; `emit_transient_batch` (later in the SAME schedule tick)
    // drained the Crossing into ONE batch and left it Held{outbound: Some(batch)}. The flip is
    // proven by BOTH the emit having fired and the resulting outbound tag.
    assert_eq!(
        rig.world.resource::<StubStats>().transient_grants_applied,
        1
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 1);
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        TransientStatus::Held {
            outbound: Some(TransferId(77)),
        },
        "the flipped Crossing was emitted, leaving Held{{outbound: Some(batch)}}",
    );
    // A REDELIVERED grant now finds Held{outbound: Some} (not a settled Held{None}) — a counted
    // no-op, so no re-flip + re-emit.
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(grant),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().transient_grant_noop, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().transients_emitted,
        1,
        "the redelivered grant did NOT re-emit",
    );
}

#[test]
fn the_crossing_aborted_demux_clears_the_latch_on_an_id_match_only() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 8);
    // A live owned dot for the entity so the trigger's eviction retain keeps its latch (the empty
    // registry means no crossing is triggered; the dot only keeps the subject alive).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(0.0, 0.0, 0.0));
    let transfer = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
    rig.world
        .resource_mut::<RequestInFlight>()
        .0
        .insert(entity, transfer);
    // A STALE abort (wrong id) preserves the latch (a superseded / re-latched crossing) — but STILL acks
    // (3f-D: the orchestrator keeps its durable entry until the ack, so every delivery must re-ack).
    let stale_out = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Entity(entity),
            transfer: TransferId(999),
        }),
    )]);
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "a mismatched abort does NOT clear the latch",
    );
    assert_eq!(rig.world.resource::<StubStats>().crossing_abort_stale, 1);
    assert_eq!(
        crossing_aborted_acks(&stale_out).len(),
        1,
        "even a stale abort is acked (all paths ack — else the orch entry leaks)",
    );
    // The MATCHING abort clears the latch, acks, and RE-ARMS (bumps the attempt + resets the dwell).
    let match_out = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Entity(entity),
            transfer,
        }),
    )]);
    assert!(
        !rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_latches_cleared,
        1
    );
    assert_eq!(
        crossing_aborted_acks(&match_out).len(),
        1,
        "the match acks too"
    );
    // The re-arm bumped the attempt (0 -> 1), so a re-cross mints a FRESH id (H2).
    assert_eq!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .get(&entity)
            .map(|st| st.crossing_attempt),
        Some(1),
    );
}

#[test]
fn a_stranded_durable_latch_redrives_after_the_ttl() {
    // 3f-D4: a delivered-but-unresolved dest leaves the durable latch STANDING with no rising edge
    // (the dot dwells statically in-band; `evaluate_realm_boundaries`'s `Occupied` arm only
    // suppresses). The per-tick latch-scan MUST re-emit the SAME `CrossingRequest` once the ttl
    // elapses — the C1 fix (the `Entry::Occupied` re-drive the DEFERRED text prescribed is
    // unreachable for a static dweller).
    const TTL: u32 = 3;
    let mut rig = Rig::with_config(config_with_ttl(TTL));
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 42);
    // Deep inside the child from spawn (sd = 100 - 1000 ≪ -inset) so the first commit latches it; there
    // is no orchestrator in this rig, so the latch is never cleared → it STRANDS (the tested condition).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Inside the child from tick 2 ⇒ the re-home commits on the FIRST evaluated tick (the band IS the
    // dwell) → the durable `Vacant` arm emits ONE request + arms `last_commit_tick = 2`.
    let mut first: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..=4 {
        rig.set_local_tick(t);
        first.extend(rig.tick(vec![]));
    }
    let first_reqs = crossing_requests(&first);
    assert_eq!(
        first_reqs.len(),
        1,
        "exactly ONE request from the rising edge"
    );
    assert_eq!(
        first_reqs[0].attempt, 0,
        "the first crossing uses attempt 0"
    );
    let latched_id = crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0);
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&latched_id),
        "the latch is held (no orchestrator ever cleared it)",
    );
    // The commit armed `last_commit_tick = 2`. Ticks 3/4 (already ticked in the loop above) were still
    // < TTL, so no re-drive fired there — proven by `first_reqs.len() == 1`.
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        0,
        "the ttl window has NOT elapsed yet through tick 4 (the `>= ttl` false arm)",
    );

    // Tick 5: local_tick - 2 == 3 >= TTL → the scan re-emits the SAME request (the `>= ttl` TRUE arm).
    rig.set_local_tick(5);
    let redrive_out = rig.tick(vec![]);
    let redrive_reqs = crossing_requests(&redrive_out);
    assert_eq!(
        redrive_reqs.len(),
        1,
        "the ttl re-drive re-emitted exactly one request"
    );
    // The re-drive is BYTE-IDENTICAL to the standing latch: same subject/to_realm/fence/session/attempt.
    assert_eq!(redrive_reqs[0].subject, DirectoryKey::Entity(entity));
    assert_eq!(redrive_reqs[0].to_realm, OTHER_REALM);
    assert_eq!(redrive_reqs[0].from_realm, config().realm);
    assert_eq!(redrive_reqs[0].subject_fence, Fence(1));
    assert_eq!(redrive_reqs[0].session, TRIG_SESSION);
    assert_eq!(redrive_reqs[0].attempt, 0, "same attempt → same id");
    assert_eq!(
        crossing_transfer_id(
            redrive_reqs[0].subject,
            redrive_reqs[0].subject_fence,
            redrive_reqs[0].attempt,
        ),
        latched_id,
        "the re-emitted id equals the standing latch id (idempotent at the orchestrator)",
    );
    assert!(
        rig.world.resource::<StubStats>().crossings_redriven >= 1,
        "the re-drive counter bumped",
    );
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&latched_id),
        "the latch is STILL held after the re-drive (only a saga terminal clears it)",
    );

    // Tick 6: the re-drive re-armed `last_commit_tick = 5`, so 6 - 5 == 1 < TTL → NO third emit yet
    // (the `>= ttl` false arm again, proving the timer re-arms and does not storm every tick).
    rig.set_local_tick(6);
    let after = rig.tick(vec![]);
    assert_eq!(
        crossing_requests(&after).len(),
        0,
        "the re-drive re-armed the ttl timer — no storm before the next window",
    );
}

#[test]
fn an_exhausted_latch_aborts_locally_and_the_next_crossing_fires() {
    // THE D-WORLD-2 headline arc: unresolved-dest request → bounded ttl re-drives (count measured)
    // → exhaustion → the LOCAL pre-CAS abort (latch cleared, attempt bumped, entity stays simulated
    // at the source) → a LATER crossing of the SAME entity fires with a FRESH id. Before the cure
    // this latch stood FOREVER (the permanent strand the walk gate hit live on an unhosted planet).
    let mut rig = Rig::with_config(config_with_ttl_and_budget(XTTL, XBUDGET));
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 44);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Ticks 2..=10: the rising-edge request (t2) + exactly XBUDGET re-drives (t5, t8), all attempt 0.
    let mut before_exhaust: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..=10 {
        rig.set_local_tick(t);
        before_exhaust.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&before_exhaust);
    assert_eq!(
        reqs.len(),
        1 + XBUDGET as usize,
        "one rising edge + exactly the budgeted re-drives before exhaustion",
    );
    assert!(
        reqs.iter().all(|r| r.attempt == 0),
        "every pre-exhaustion emit is the SAME attempt (byte-identical id)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        u64::from(XBUDGET)
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_exhausted, 0);

    // Tick 11: the third expiry finds the budget spent → the LOCAL exhaustion abort.
    rig.set_local_tick(11);
    let exhaust_out = rig.tick(vec![]);
    assert_eq!(
        crossing_requests(&exhaust_out).len(),
        0,
        "exhaustion emits NO further request — it aborts instead",
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_exhausted, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_latches_cleared,
        1,
        "the exhaustion abort rides the ONE pre-CAS latch clear",
    );
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        None,
        "THE LATCH CLEARS — the strand is over",
    );
    let st = *rig
        .world
        .resource::<CrossingProgress>()
        .0
        .get(&entity)
        .expect("the crossing state survives the abort");
    assert_eq!(st.crossing_attempt, 1, "the abort bumped the attempt (H2)");
    assert_eq!(st.latched_crossing, None, "the re-emit payload is dropped");
    assert_eq!(st.redrives_spent, 0, "the budget resets for the next latch");
    assert_eq!(
        st.last_commit_tick,
        Some(TickId(11)),
        "the cooldown arms AT the abort tick — the sit-inside backoff rides k_dwell",
    );
    // THAW: the entity stays simulated at the source (no saga ever started; nothing was frozen).
    assert!(
        rig.world
            .resource::<Dots>()
            .0
            .get(&TRIG_SESSION)
            .expect("the dot is still here")
            .authority
            .simulates(),
        "the entity stays simulated at the source after the local abort",
    );

    // Ticks 12..=15: since_commit 1..4 < k_dwell(5) → the EXISTING dwell suppresses the re-fire.
    let mut cooldown_out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 12..=15 {
        rig.set_local_tick(t);
        cooldown_out.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&cooldown_out).len(),
        0,
        "the k_dwell cooldown bounds the refire — no per-tick abort/latch storm",
    );

    // Tick 16: the cooldown elapsed and the dot is STILL in-band → the SAME entity's crossing
    // FIRES AGAIN, re-latched under a FRESH id (attempt 1) — the strand cure's whole point.
    rig.set_local_tick(16);
    let refire_out = rig.tick(vec![]);
    let refire = crossing_requests(&refire_out);
    assert_eq!(refire.len(), 1, "a later crossing of the SAME entity fires");
    assert_eq!(refire[0].attempt, 1, "the re-fire mints the bumped attempt");
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&crossing_transfer_id(
            DirectoryKey::Entity(entity),
            Fence(1),
            1
        )),
        "the re-latch id is FRESH (attempt-stamped) — a stale abort can never wrong-clear it",
    );
}

#[test]
fn a_sit_inside_dweller_is_bounded_to_the_derived_refire_cycle() {
    // D-WORLD-2 sit-inside backoff: a dot PARKED inside an unresolvable region refires scan →
    // re-drives → abort FOREVER — but at a BOUNDED cadence derived from the existing machinery:
    // one full cycle is `(budget+1)·ttl` (the latch's re-drive life) + `k_dwell` (the containment
    // cooldown the abort arms), and each cycle emits exactly `budget+1` requests. No sleep
    // literal anywhere: the bound is computed from the same config the systems read.
    let cfg = config_with_ttl_and_budget(XTTL, XBUDGET);
    let cycle = (u64::from(XBUDGET) + 1) * u64::from(XTTL) + u64::from(cfg.boundary.k_dwell);
    let per_cycle = u64::from(XBUDGET) + 1;
    let mut rig = Rig::with_config(cfg);
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 45);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Two full cycles, starting at the first evaluated tick (2): 2 ..= 2 + 2·cycle − 1.
    let window = 2 * cycle;
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..(2 + window) {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let total = crossing_requests(&all).len() as u64;
    // The derived bound, computed (never a literal): ⌈window/cycle⌉ cycles × (budget+1) requests.
    let bound = window.div_ceil(cycle) * per_cycle;
    assert!(
        total <= bound,
        "refire count over {window} ticks must stay within the derived bound {bound}, got {total}",
    );
    // Deterministic rig ⇒ the exact count is the bound itself (2 cycles × 3 requests): t2/t5/t8
    // then (post-abort at 11, cooldown to 15) t16/t19/t22 — pinning the cadence, not just the cap.
    assert_eq!(total, 2 * per_cycle);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_exhausted,
        2,
        "one exhaustion abort per cycle",
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        2 * u64::from(XBUDGET),
        "the budgeted re-drives per cycle, twice",
    );
}

#[test]
fn a_graze_continues_unharmed_after_the_exhaustion_abort() {
    // D-WORLD-2 graze: the dot PASSES THROUGH the unresolvable region (the walk gate's live
    // failure: a fly-in grazing an unhosted planet's 3.95 m SOI) and is long gone by exhaustion.
    // The abort clears the latch; the container then MATCHES the owning realm, so nothing
    // re-fires — the graze simply continues, and the entity's later crossings are free again.
    let mut rig = Rig::with_config(config_with_ttl_and_budget(XTTL, XBUDGET));
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 46);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    // Tick 2: in-band → the rising-edge request latches (the graze's entry).
    rig.set_local_tick(2);
    let first = rig.tick(vec![]);
    assert_eq!(crossing_requests(&first).len(), 1);
    // The graze leaves: well past the child's destroy edge (r 1000 + outset), still inside OWN.
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(5000.0, 0.0, 0.0));

    // Through the re-drives (5, 8 — the scan re-emits the LATCHED payload regardless of where the
    // dot now is) and the exhaustion (11), then a long quiet tail.
    let mut rest: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 3..=30 {
        rig.set_local_tick(t);
        rest.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&rest).len(),
        XBUDGET as usize,
        "only the budgeted re-drives — after the abort the departed graze NEVER re-fires",
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_exhausted, 1);
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        None,
        "the latch is gone — the entity's future crossings are unsuppressed",
    );
    let dot = rig
        .world
        .resource::<Dots>()
        .0
        .get(&TRIG_SESSION)
        .expect("the dot is still here");
    assert!(
        dot.authority.simulates(),
        "the graze continues unharmed — still owned and simulated at the source",
    );
}

#[test]
fn an_armed_shard_keepalive_demands_an_inward_crossing_dest() {
    // The PRODUCER gate of the Symptom-B fix, on the LAWFUL shape (finding 37): on an ARMED
    // (demand-scale) shard, every tick a durable crossing latch stands,
    // `redrive_stranded_crossings` emits a KeepAlive `RealmDemand` for an INWARD dest — a DIRECT
    // CHILD, the one crossing shape SL7 lets a shard demand (`ancestor_close` pulls the chain
    // above it at the orchestrator). The child's own AoI band is INERT and only the root's is
    // armed, so the keep-alive is the SOLE demand under test (no AoI-union collision);
    // `request_ttl_ticks == 0` keeps the ttl re-drive inert too.
    let armed_root = RealmRegion {
        aoi: aoi_band(1), // arms `aoi_live` without adding any direct-child AoI demand
        ..region(ROOT_REALM, None, DVec3::ZERO, 1.0e9)
    };
    let forest = RealmRegions::new(vec![
        armed_root,
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 100_000.0),
        region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0),
    ]);
    // `own_coord` comes from the SAME forest the demand coord will (exactly the production boot:
    // one lineage source), or the parent-of-child compare would be measuring two derivations.
    let own_coord = forest.coord_of(OWN_REALM).expect("own realm is rostered");
    let want = forest
        .coord_of(OTHER_REALM)
        .expect("the child is a seed-lineage realm");
    let mut rig = Rig::with_config(StubConfig {
        own_coord,
        ..config()
    });
    grant_realm_for(&mut rig, OWN_REALM);
    *rig.world.resource_mut::<RealmRegions>() = forest;
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 99);
    // INSIDE the child (r=1000, off-centre) ⇒ container == OTHER_REALM ⇒ cross IN, latching the
    // crossing (no orchestrator in the rig ⇒ the latch STRANDS and keeps emitting).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]); // cross IN + latch
    rig.set_local_tick(3);
    let out = rig.tick(vec![]); // the latch stands ⇒ the keep-alive fires this tick
    let dest_keepalives = realm_demands(&out)
        .into_iter()
        .filter(|d| d.child == want)
        .filter(|d| d.verb == DemandVerb::KeepAlive)
        .count();
    assert!(dest_keepalives >= 1);
    // The lawful shape passed the structural gate uncounted; the ttl re-drive stayed inert.
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .demand_refused_not_own_or_child,
        0
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_redriven, 0);
}

#[test]
fn an_outward_crossing_emits_no_demand_and_counts_the_refusal() {
    // The OTHER half of the finding-37 cure, measured: an OUTWARD crossing's dest is this shard's
    // PARENT, which a shard may never demand (SL7 — a realm speaks about itself or a direct child,
    // never upward). `push_demand`'s structural gate refuses it, counted; the latch stands
    // untouched. The parent needs no upward demand: the latch keeps the departing occupant in this
    // shard's observer fold (`speaks_for`), so this realm never reports Empty, arm B of
    // `desired_alive` holds it, and `ancestor_close` pulls the chain — the deterministic twin is
    // `rlm::tests::an_outward_crossings_parent_stays_alive_without_an_upward_demand`, the
    // process-tier experiment the return-crossing gate.
    let mut rig = Rig::new();
    rig.grant_realm();
    let armed = |realm, parent, r| RealmRegion {
        realm,
        center: vd_core::geometry::ParentCentre::authored(LatticePos::ORIGIN),
        frame: frame_of(realm),
        shape: Boundary::Shell { r },
        look: Some(Boundary::Shell { r }),
        band: band(),
        aoi: aoi_band(1),
        parent,
    };
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        armed(ROOT_REALM, None, 1.0e9),
        armed(PARENT_REALM, Some(ROOT_REALM), 100_000.0),
        armed(OWN_REALM, Some(PARENT_REALM), 1000.0),
    ]);
    let parent_coord = rig
        .world
        .resource::<RealmRegions>()
        .coord_of(PARENT_REALM)
        .expect("the parent is a seed-lineage realm");
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 99);
    // OUTSIDE own_small (r=1000) but inside the parent (r=100_000) ⇒ container == PARENT_REALM ⇒
    // cross OUT, latching the crossing (no orchestrator ⇒ the latch STRANDS).
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(5000.0, 0.0, 0.0));
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]); // cross OUT + latch
    rig.set_local_tick(3);
    let out = rig.tick(vec![]); // the latch stands ⇒ the refusal fires this tick
    // The WHOLE demand list is empty — the only candidate emitter this tick was the outward
    // keep-alive (the own realm has no AoI children here), so this is the strongest claim, by
    // equality (a filtering closure would carry an uncoverable never-matching region, HR5).
    assert_eq!(
        realm_demands(&out),
        vec![],
        "no demand at all leaves the shard — the upward emit toward {parent_coord:?} is \
         structurally gone"
    );
    assert!(
        rig.world
            .resource::<StubStats>()
            .demand_refused_not_own_or_child
            >= 1,
        "the refusal is counted where the shape law lives"
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "the crossing latch stands regardless — only a saga terminal clears it"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_redriven, 0);
}

#[test]
fn the_ttl_redrive_is_inert_when_request_ttl_ticks_is_zero() {
    // 3f-D4: with the DEFAULT `request_ttl_ticks == 0` the scan EARLY-RETURNS (the inert default of
    // every current rig) — a held latch is NEVER re-driven, even past many ticks.
    let mut rig = Rig::new(); // request_ttl_ticks == 0
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 43);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Commit on the first in-band tick, then advance well past any plausible ttl window.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..=40 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&all).len(),
        1,
        "exactly the ONE rising-edge request — the ttl scan is inert (never re-drives)",
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_redriven,
        0,
        "the ttl==0 early-return means a held latch is never re-driven",
    );
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "the latch stands (no positive clear), but is never re-emitted",
    );
}

#[test]
fn a_crossing_aborted_for_a_non_entity_subject_still_acks() {
    // The THIRD unconditional-ack path (3f-D): a `CrossingAborted` whose subject is not an Entity (a
    // malformed/realm subject) has no latch to clear, but MUST still ack — else the orchestrator's
    // durable `pending_abort_replies` entry would leak.
    let mut rig = Rig::new();
    rig.grant_realm();
    let out = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Realm(RealmId::System(7)),
            transfer: TransferId(5),
        }),
    )]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_abort_no_entity,
        1
    );
    assert_eq!(
        crossing_aborted_acks(&out).len(),
        1,
        "the no-entity path acks too",
    );
}

#[test]
fn a_saga_demote_clears_the_durable_crossing_latch() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    for t in 2..8 {
        rig.set_local_tick(t);
        let _ = rig.tick(vec![]);
    }
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    // The durable COMMIT terminal at the source (the saga-pushed Demote) POSITIVELY clears it.
    let demote = DemoteCmd {
        transfer: crossing_transfer_id(DirectoryKey::Entity(entity), Fence(1), 0),
        subject: DirectoryKey::Entity(entity),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Demote(demote),
    )]);
    assert!(
        !rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity),
        "the durable Demote terminal cleared the crossing latch",
    );
    assert!(rig.world.resource::<StubStats>().crossing_latches_cleared >= 1);
}

#[test]
fn slice4b_an_undock_re_homes_outward_to_the_parent_realm() {
    // Slice 4b (retargeted to CONTAINMENT) — the UNDOCK leg: the shard OWNS a CHILD realm (OWN_REALM,
    // a small shell nested under PARENT_REALM). An entity leaving the child region (container flips to
    // the parent) re-homes OUTWARD to PARENT_REALM. Symmetric with the dock — the SAME machinery; there
    // is no "direction", only "the container changed".
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 1);
    let out = drive_outward_exit(&mut rig, entity);
    let reqs = crossing_requests(&out);
    assert_eq!(
        reqs.len(),
        1,
        "leaving the owned child region re-homes exactly once",
    );
    // THE LOAD-BEARING ASSERT: the undock re-homes to the PARENT (the deepest region still containing
    // the dot after it left the child). NON-VACUOUS because PARENT_REALM != OWN_REALM != OTHER_REALM.
    assert_eq!(
        reqs[0].to_realm, PARENT_REALM,
        "the undock re-homes OUTWARD to the parent realm (the new deepest container)",
    );
    assert_ne!(
        reqs[0].to_realm, OTHER_REALM,
        "the undock destination is NOT the inward child interior",
    );
    assert_eq!(
        reqs[0].from_realm, OWN_REALM,
        "the re-home leaves the realm the shard owns the subject in",
    );
}

#[test]
fn slice4b_a_dock_and_undock_resolve_contrasting_destinations() {
    // Slice 4b (retargeted to CONTAINMENT) — DOCK vs UNDOCK contrast: entering a DEEPER child region
    // re-homes to the child interior (OTHER_REALM); leaving an OWNED child region re-homes to the
    // parent (PARENT_REALM). Both fall out of `container()`; the two destinations DIFFER (no accidental
    // alias) — proving the symmetric rule distinguishes the two container changes without a direction.

    // DOCK (into a deeper child) — the standard dock forest; the dot inside the child re-homes to it.
    let mut dock = Rig::new();
    dock.grant_realm();
    plant_dock_regions(&mut dock);
    let dock_entity = EntityId::pack(EntityKind::Player, 10, 2, 2);
    // Deep inside the child from spawn (sd ≪ -inset) → container == OTHER_REALM ≠ own ⇒ re-home INWARD.
    insert_owned_dot(
        &mut dock,
        TRIG_SESSION,
        dock_entity,
        DVec3::new(100.0, 0.0, 0.0),
    );
    let mut dock_out: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        dock.set_local_tick(t);
        dock_out.extend(dock.tick(vec![]));
    }
    let dock_reqs = crossing_requests(&dock_out);
    assert_eq!(dock_reqs.len(), 1, "the dock commits exactly one re-home");
    assert_eq!(
        dock_reqs[0].to_realm, OTHER_REALM,
        "the dock docks INWARD to the deeper child region's realm",
    );

    // UNDOCK (out of an owned child) — the escape forest; leaving the own-shell undocks to the parent.
    let mut undock = Rig::new();
    undock.grant_realm();
    let undock_entity = EntityId::pack(EntityKind::Player, 10, 2, 3);
    let undock_out = drive_outward_exit(&mut undock, undock_entity);
    let undock_reqs = crossing_requests(&undock_out);
    assert_eq!(
        undock_reqs.len(),
        1,
        "the undock commits exactly one re-home"
    );
    assert_eq!(
        undock_reqs[0].to_realm, PARENT_REALM,
        "the undock re-homes OUTWARD to the parent realm",
    );

    // THE CONTRAST: the two container changes resolve DISTINCT destinations.
    assert_ne!(
        dock_reqs[0].to_realm, undock_reqs[0].to_realm,
        "a dock and an undock resolve DIFFERENT destinations (deeper child vs parent)",
    );
}

#[test]
fn a_nested_inner_boundary_wins_over_its_parent() {
    // Retargeted to CONTAINMENT (the `deepest_wins` property): a dot inside BOTH the own region
    // (System(7)) AND a coincident DEEPER child (Planet(42), nested under own) has container == the
    // INNER child (depth beats the parent), so the re-home targets the inner realm — exercising the
    // boot-cached `region_depth` argmax.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig); // root ⊃ own(System 7) ⊃ child(Planet 42), all origin-coincident
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 5);
    // At |100| the dot is inside own (100000) AND the child (1000): the deeper child wins.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&all);
    assert_eq!(reqs.len(), 1);
    assert_eq!(
        reqs[0].to_realm, OTHER_REALM,
        "the INNER (deeper) region wins the depth argmax",
    );
}

#[test]
fn a_boundary_hovering_dot_does_not_flap() {
    // Retargeted to CONTAINMENT (`hysteresis_no_flap`): a dot jittering INSIDE the child region's band
    // for many ticks stays a member (its bit never releases), so container is stable and the latch caps
    // the emits at ≤ 1 across the whole window (no per-tick flap).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 2);
    // Hover deep inside the child (sd stays ≪ -inset even with the jitter): a member every tick.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(200.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..15 {
        rig.set_local_tick(t);
        // Jitter within the child region each tick (still a member: sd stays deeply negative).
        let jitter = if t % 2 == 0 { 210.0 } else { 190.0 };
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(jitter, 0.0, 0.0));
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all).len() <= 1,
        "a region-hovering dot emits at most one request (no flap)",
    );
}

#[test]
fn a_non_entity_grant_and_abort_are_counted_no_ops() {
    // The `transfer_subject_entity() == None` arms of the two demuxes (a Realm subject).
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
            subject: DirectoryKey::Realm(RealmId::System(7)),
            dest: DEST_NODE,
            to_realm: OTHER_REALM,
            dst_realm_fence: Fence(3),
            batch: TransferId(1),
            to_parent: None,
        }),
    )]);
    assert_eq!(
        rig.world.resource::<StubStats>().transient_grant_no_entity,
        1
    );
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(CrossingAborted {
            subject: DirectoryKey::Realm(RealmId::System(7)),
            transfer: TransferId(1),
        }),
    )]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_abort_no_entity,
        1
    );
}

#[test]
fn a_grant_for_an_unknown_transient_is_a_counted_no_op() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 10, 2, 3);
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
            subject: DirectoryKey::Entity(entity),
            dest: DEST_NODE,
            to_realm: OTHER_REALM,
            dst_realm_fence: Fence(3),
            batch: TransferId(1),
            to_parent: None,
        }),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().transient_grant_noop, 1);
}

#[test]
fn a_dot_outside_the_deeper_child_but_inside_own_stays() {
    // Retargeted from the portal "StaysOutside is not a candidate": a dot that is OUTSIDE the deeper
    // child region (its bit never acquires) but inside the OWN region has container == OWN_REALM — the
    // realm the shard owns it in — so NO re-home fires and no crossing latch is created.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 4);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(9000.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        // Move around, but always well outside the child region (and inside own).
        move_dot(
            &mut rig,
            TRIG_SESSION,
            DVec3::new(9000.0 + t as f64, 0.0, 0.0),
        );
        all.extend(rig.tick(vec![]));
    }
    assert!(crossing_requests(&all).is_empty());
    assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
    // The subject is evaluated every tick (so it carries a ContainmentProgress bit + a CrossingProgress
    // cooldown row), but since container == own NO re-home ever COMMITTED — the cooldown was never
    // armed (the `last_commit_tick` is None).
    assert_eq!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .get(&entity)
            .and_then(|st| st.last_commit_tick),
        None,
        "no re-home committed ⇒ the cooldown was never armed",
    );
}

#[test]
#[should_panic(expected = "valid BoundaryTuning")]
fn a_zero_dwell_boundary_tuning_fails_loud_at_boot() {
    let bad = StubConfig {
        boundary: BoundaryTuning {
            k_dwell: 0,
            ..BoundaryTuning::DEFAULT
        },
        ..config()
    };
    // The fail-loud validation in `register_stub_shard` (mirrors the tick-pair guard). `k_dwell` is the
    // post-commit cooldown `should_rehome` still reads (§2.7); zero fails `BoundaryTuning::validate`.
    let _ = Rig::with_config(bad);
}

// ===================== the ruler switch, slice 1: a driven child is a subject =====================

/// The escape forest with a HULL berthed inside the own-shell: root ⊃ parent ⊃ own (r = 1000) ⊃ hull
/// (a 20 m shell at x = 500). The hull is a DRIVEN child the shard authors.
fn plant_escape_regions_with_hull(rig: &mut Rig, hull: RealmId) {
    let parent = region(PARENT_REALM, Some(ROOT_REALM), DVec3::ZERO, 100_000.0);
    let own_small = region(OWN_REALM, Some(PARENT_REALM), DVec3::ZERO, 1000.0);
    let berth = region(hull, Some(OWN_REALM), DVec3::new(500.0, 0.0, 0.0), 20.0);
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), parent, own_small, berth]);
}

fn hull_child() -> (RealmId, EntityId) {
    let entity = EntityId::pack(EntityKind::Ship, 1, 1, 0);
    (RealmId::Ship(entity), entity)
}

fn drive_hull(rig: &mut Rig, hull: RealmId, pos_m: DVec3, vel_mps: DVec3) {
    rig.world
        .resource_mut::<crate::stub::drive::DrivenChildren>()
        .0
        .insert(
            hull,
            crate::stub::drive::DrivenChild {
                state: crate::stub::drive::DrivenState {
                    pos_m,
                    vel_mps,
                    orient: DQuat::IDENTITY,
                    spin_radps: DVec3::ZERO,
                },
                drive: ([0; 3], [0; 3]),
                drive_at: (Fence::GENESIS, UniverseTick(0)),
                facts: None,
                facts_at: (Fence::GENESIS, UniverseTick(0)),
                frozen: false,
            },
        );
}

fn exterior_decided(rig: &Rig) -> u64 {
    rig.world.resource::<StubStats>().exterior_crossings_decided
}

#[test]
fn a_driven_child_at_rest_in_its_berth_is_no_crossing_even_inside_its_own_bound() {
    // The hull's own region holds the hull's centre by construction. Without the exclusion the fold
    // would name the hull as its own deepest container and decide a crossing into itself.
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(1));
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    for t in 2..6 {
        rig.set_local_tick(t);
        let _ = rig.tick(vec![]);
    }
    assert_eq!(exterior_decided(&rig), 0);
    assert_eq!(rig.world.resource::<StubStats>().exterior_scan_unleased, 0);
}

#[test]
fn a_driven_child_leaving_the_parents_shell_is_decided_once_per_dwell() {
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(1));
    // Inside first (one in-band tick primes membership), then 4500 m of travel from the berth: the
    // centre sits at x = 5000, outside the own-shell (r = 1000), inside the parent (r = 100 000).
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    drive_hull(&mut rig, hull, DVec3::new(4500.0, 0.0, 0.0), DVec3::ZERO);
    rig.set_local_tick(3);
    let _ = rig.tick(vec![]);
    assert_eq!(exterior_decided(&rig), 1, "the exit is decided");
    // The cooldown holds: the next tick decides nothing new while the hull dwells outside.
    rig.set_local_tick(4);
    let _ = rig.tick(vec![]);
    assert_eq!(
        exterior_decided(&rig),
        1,
        "decided once per dwell, not once per tick"
    );
    // No dot request rode the wire: the exterior arm is its own, and it is pending the owner's word.
    assert_eq!(rig.world.resource::<StubStats>().crossings_requested, 0);
}

#[test]
fn a_driven_child_is_led_by_its_own_velocity_over_the_request_ttl() {
    // The early start: the hull's TRUE point is inside (x = 500), but led by its velocity over the
    // request ttl (20 ticks × 0.05 s = 1 s at 3000 m/s ⇒ x = 3500) it is outside ⇒ decided now.
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.world.resource_mut::<StubConfig>().request_ttl_ticks = 20;
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(1));
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::new(3000.0, 0.0, 0.0));
    rig.set_local_tick(3);
    let _ = rig.tick(vec![]);
    assert_eq!(
        exterior_decided(&rig),
        1,
        "the led point leaves before the hull does"
    );
}

#[test]
fn a_driven_child_whose_exterior_this_realm_does_not_hold_is_skipped_and_counted() {
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    // No exterior lease: this realm does not author the placement, so it may not decide.
    drive_hull(&mut rig, hull, DVec3::new(4500.0, 0.0, 0.0), DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    assert_eq!(exterior_decided(&rig), 0);
    assert_eq!(rig.world.resource::<StubStats>().exterior_scan_unleased, 1);
}

/// The `ExteriorCrossingRequest`s that rode the wire to the orchestrator.
fn exterior_requests(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<vd_wire::intershard::ExteriorCrossingRequest> {
    sent.iter()
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::ExteriorCrossingRequest(r)) => Some(r),
                _ => None,
            },
        )
        .collect()
}

#[test]
fn a_driven_child_entering_a_direct_childs_shell_is_handed_down_to_that_child() {
    // The ruler switch, the INWARD leg: the galaxy carries a hull into System 8's shell and must ask
    // the orchestrator to move the hull's exterior DOWN to System 8. Here: OWN holds the hull and a
    // child Planet(9) 5 km out; the hull is driven into the planet's shell in one swept step.
    let (hull, entity) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    let parent = region(PARENT_REALM, Some(ROOT_REALM), DVec3::ZERO, 1.0e9);
    let own = region(OWN_REALM, Some(PARENT_REALM), DVec3::ZERO, 100_000.0);
    let berth = region(hull, Some(OWN_REALM), DVec3::new(500.0, 0.0, 0.0), 20.0);
    let planet = region(
        RealmId::Planet(9),
        Some(OWN_REALM),
        DVec3::new(5000.0, 0.0, 0.0),
        500.0,
    );
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), parent, own, berth, planet])
            .with_own_realm(OWN_REALM);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(5));
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    // 4500 m of travel from the berth: the centre sits at x = 5000, inside the planet's shell.
    drive_hull(&mut rig, hull, DVec3::new(4500.0, 0.0, 0.0), DVec3::ZERO);
    rig.set_local_tick(3);
    let sent = rig.tick(vec![]);
    assert_eq!(exterior_decided(&rig), 1, "the entry is decided");
    let reqs = exterior_requests(&sent);
    assert_eq!(reqs.len(), 1, "one request, down to the child: {reqs:?}");
    assert_eq!(reqs[0].subject, DirectoryKey::Ship(entity));
    assert_eq!(reqs[0].from_realm, OWN_REALM);
    assert_eq!(reqs[0].to_realm, RealmId::Planet(9));
}

#[test]
fn a_hull_crossing_a_star_systems_shell_at_speed_on_the_galaxys_ruler_is_handed_down() {
    // The ruler switch, the INWARD leg at the GALAXY's own rung (2 m cells): the galaxy carries a hull
    // at 135 m per tick through System 8's 1 km shell — fifteen ticks inside — and must decide the
    // entry on the swept line, with the request naming System 8. This is the fixture's geometry in
    // the unit rig, so a refusal has a counter to name.
    let galaxy = RealmId::Galaxy(1);
    let system_8 = RealmId::System(8);
    let (hull, entity) = hull_child();
    let cfg = StubConfig {
        realm: galaxy,
        own_coord: StubConfig::root_coord(galaxy),
        held_realms: StubConfig::single_realm(galaxy),
        frame: FrameRef::GalaxySpace { galaxy_seed: 1 },
        request_ttl_ticks: 40,
        ..config()
    };
    let mut rig = Rig::with_config(cfg);
    grant_realm_for(&mut rig, galaxy);
    let at = |centre: DVec3, tier: Tier| {
        vd_core::geometry::ParentCentre::authored(LatticePos::from_metres(centre, tier))
    };
    let mk =
        |realm: RealmId, parent: Option<RealmId>, centre: DVec3, tier: Tier, r: f64| RealmRegion {
            realm,
            center: at(centre, tier),
            frame: match realm {
                RealmId::Ship(ship) => FrameRef::ShipLocal { ship },
                RealmId::Galaxy(g) => FrameRef::GalaxySpace { galaxy_seed: g },
                other => frame_of(other),
            },
            shape: Boundary::Shell { r },
            look: Some(Boundary::Shell { r }),
            band: band(),
            aoi: vd_core::geometry::AoiConfig::inert(),
            parent,
        };
    let regions = vec![
        mk(galaxy, None, DVec3::ZERO, Tier::Galaxy, 1.0e12),
        mk(
            system_8,
            Some(galaxy),
            DVec3::new(0.0, 0.0, -50_000.0),
            Tier::Galaxy,
            1_000.0,
        ),
        mk(
            hull,
            Some(galaxy),
            DVec3::new(0.0, 0.0, -1_000.0),
            Tier::Galaxy,
            20.0,
        ),
    ];
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(regions).with_own_realm(galaxy);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(5));
    // The hull is driven by hand along -z at 135 m per tick, its velocity stated so the lead is real.
    let vel = DVec3::new(0.0, 0.0, -2_700.0);
    let mut decided_at = None;
    let mut reqs = Vec::new();
    for i in 0..600u64 {
        let z = -(i as f64) * 135.0;
        drive_hull(&mut rig, hull, DVec3::new(0.0, 0.0, z), vel);
        rig.set_local_tick(2 + i);
        let sent = rig.tick(vec![]);
        reqs.extend(exterior_requests(&sent));
        if decided_at.is_none() && exterior_decided(&rig) > 0 {
            decided_at = Some(i);
        }
        if !reqs.is_empty() {
            break; // the request is out and latched: the demote below answers it
        }
    }
    let stats = rig.world.resource::<StubStats>();
    assert!(
        decided_at.is_some(),
        "the entry was never decided: cross_tier_refused {} prior_unhosted {} book_miss {} unleased {}",
        stats.cross_tier_refused,
        stats.containment_prior_unhosted,
        stats.placement_book_miss,
        stats.exterior_scan_unleased
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Ship(entity));
    assert_eq!(reqs[0].to_realm, system_8, "handed DOWN to System 8");
    // THE SOURCE SIDE OF THE HAND-DOWN: the demote of a Ship subject releases the exterior AND clears
    // the crossing latch the request armed (the ruler switch, slice 2).
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Demote(vd_wire::intershard::DemoteCmd {
            transfer: TransferId(900),
            subject: DirectoryKey::Ship(entity),
            new_owner_fence: Fence(9),
            step_id: vd_wire::intershard::DEMOTE_STEP,
        }),
    )]);
    assert!(
        !rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_latches_cleared,
        1
    );
    assert!(!sent.is_empty(), "the demote is acked");
    // THE RELIABLE CARRIER'S OTHER ARM through the inbox: a child's facts from a node the directory
    // does not place at that child are refused and counted, never applied.
    let coord = rig
        .world
        .resource::<RealmRegions>()
        .coord_of(hull)
        .or_else(|| Some(StubConfig::root_coord(galaxy).child(vd_core::worldgen::level_of(hull))))
        .expect("a coord");
    rig.tick(vec![wire_msg(
        NodeId(4_040),
        MsgClass::Saga,
        &InterShardFlow::ChildFacts(vd_wire::intershard::ChildFacts {
            child: coord,
            child_fence: Fence(1),
            at: UniverseTick(1),
            mass_g: 1,
            cross_section_mm2: 1,
            drag_micro: 1,
            declared: vd_wire::intershard::DeclaredStates::default(),
        }),
    )]);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(
        stats.child_facts_received, 0,
        "facts from an unattested node are never held"
    );
    assert_eq!(
        stats.child_facts_misrouted + stats.child_facts_unattested,
        1
    );
    // A Membership-class frame (the clock's business) is ignored by the stub's dispatch.
    rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Membership,
        &InterShardFlow::PeerLocate(vd_wire::intershard::PeerLocate {
            node: NodeId(1),
            at: UniverseTick(1),
        }),
    )]);
}

#[test]
fn a_driven_childs_exit_rides_the_exterior_request_arm_once_and_is_re_driven_on_the_ttl() {
    let (hull, entity) = hull_child();
    let mut rig = Rig::new();
    rig.world.resource_mut::<StubConfig>().request_ttl_ticks = 3;
    rig.world
        .resource_mut::<StubConfig>()
        .crossing_redrive_budget = 1;
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(5));
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    drive_hull(&mut rig, hull, DVec3::new(4500.0, 0.0, 0.0), DVec3::ZERO);
    rig.set_local_tick(3);
    let sent = rig.tick(vec![]);
    let reqs = exterior_requests(&sent);
    assert_eq!(reqs.len(), 1, "one request per crossing: {reqs:?}");
    assert_eq!(
        reqs[0],
        vd_wire::intershard::ExteriorCrossingRequest {
            subject: DirectoryKey::Ship(entity),
            from_realm: OWN_REALM,
            to_realm: PARENT_REALM,
            subject_fence: Fence(5),
            attempt: 0,
        }
    );
    assert!(
        crossing_requests(&sent).is_empty(),
        "an exterior never rides the session-bearing arm"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .exterior_crossings_requested,
        1
    );
    // Latched: the next tick emits nothing new while the hull dwells outside.
    rig.set_local_tick(4);
    let sent = rig.tick(vec![]);
    assert!(exterior_requests(&sent).is_empty());
    // The ttl passes with no terminal: the SAME request is re-driven, on the SAME arm.
    rig.set_local_tick(6);
    let sent = rig.tick(vec![]);
    let again = exterior_requests(&sent);
    assert_eq!(again.len(), 1, "re-driven once on the ttl: {again:?}");
    assert_eq!(
        again[0], reqs[0],
        "the re-drive re-mints the byte-identical request"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_redriven, 1);
    // The keep-alive for the destination is skipped and counted: this forest can name the parent,
    // so the count stays zero here — the unnamed arm is pinned by the hull-into-hull case below.
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_keepalive_unnamed,
        0
    );
}

// ===================== the ruler switch, slice 2: a child arrives and leaves at runtime =====================

#[test]
fn a_parent_adopts_a_child_region_at_runtime_and_releases_it_with_every_table_in_step() {
    let (hull, _) = hull_child();
    let far = RealmId::Planet(77);
    // own(r=1000) holds a seeded planet; the hull arrives, then leaves; the planet was the LAST row
    // before the hull, so the release swaps it back into the hull's slot and must re-index it.
    let mut regions = RealmRegions::new(vec![
        root_region(),
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 1000.0),
        aoi_child(far, OWN_REALM, 50.0, 0),
    ])
    .with_own_realm(OWN_REALM);
    let berth = region(hull, Some(OWN_REALM), DVec3::new(500.0, 0.0, 0.0), 20.0);
    // Somebody else's child is refused: only MY direct children are mine to adopt.
    regions.adopt_child(region(
        RealmId::Planet(5),
        Some(ROOT_REALM),
        DVec3::ZERO,
        1.0,
    ));
    assert!(
        regions
            .direct_child(ROOT_REALM, RealmId::Planet(5))
            .is_none()
    );
    regions.adopt_child(berth);
    assert!(
        regions.direct_child(OWN_REALM, hull).is_some(),
        "on the roster"
    );
    assert!(
        regions.child_index().answers_for(hull),
        "in the containment grid"
    );
    assert_eq!(
        regions.coord_of(hull).map(|c| c.lowered()),
        Some(hull),
        "its lineage resolves through the forest"
    );
    let ix = regions.ix_of[&hull];
    assert_eq!(regions.depths[ix].1, hull);
    assert_eq!(regions.depths[ix].0, 2, "root → own → hull");
    assert!(regions.ancestor_chain[ix].contains(&OWN_REALM));
    assert_eq!(regions.direct_children(OWN_REALM).count(), 2);
    // A re-driven adopt replaces in place: still one row for the hull.
    regions.adopt_child(berth);
    assert_eq!(regions.direct_children(OWN_REALM).count(), 2);
    assert_eq!(
        regions.regions.iter().filter(|r| r.realm == hull).count(),
        1
    );
    // Release: the hull is gone and every table still names the planet at its new slot.
    regions.release_child(hull);
    assert!(regions.direct_child(OWN_REALM, hull).is_none());
    assert!(!regions.child_index().answers_for(hull));
    assert!(regions.coord_of(hull).is_none());
    assert_eq!(regions.direct_children(OWN_REALM).count(), 1);
    let planet_ix = regions.ix_of[&far];
    assert_eq!(regions.regions[planet_ix].realm, far);
    assert_eq!(regions.depths[planet_ix].1, far);
    assert_eq!(regions.depths[planet_ix].2, planet_ix);
    assert!(regions.direct_child(OWN_REALM, far).is_some());
    assert!(regions.child_index().answers_for(far));
    assert!(regions.aoi_index().answers_for(far));
    // Releasing a realm not on the roster is a no-op (a re-driven release).
    regions.release_child(hull);
    assert_eq!(regions.regions.len(), 3);
}

// ===================== the ruler switch, slice 2: the flush, the adoption, the release =====================

fn exterior_acks(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<TransferAck> {
    to_orch(sent)
        .into_iter()
        .filter_map(|f| match f {
            InterShardFlow::TransferAck(a) => Some(a),
            _ => None,
        })
        .collect()
}

#[test]
fn a_parent_flushes_a_hull_down_into_its_child_before_the_hull_reaches_the_childs_shell() {
    // The ruler switch, the INWARD flush with the EARLY START: the verdict was made on the led point,
    // so at the flush the hull is still short of the child's shell. The flush must SUBTRACT the
    // child's placement and ship — never refuse the hand-off as a stale entry (that re-validation is
    // the occupant's, whose flush happens after the fact). MEASURED on the six-shard fixture: the
    // galaxy's flush toward System 8 came back unplaceable, the saga aborted, and the hull flew on.
    let (hull, entity) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    let parent = region(PARENT_REALM, Some(ROOT_REALM), DVec3::ZERO, 1.0e9);
    let own = region(OWN_REALM, Some(PARENT_REALM), DVec3::ZERO, 100_000.0);
    let berth = region(hull, Some(OWN_REALM), DVec3::new(500.0, 0.0, 0.0), 20.0);
    let planet = region(
        RealmId::Planet(9),
        Some(OWN_REALM),
        DVec3::new(5000.0, 0.0, 0.0),
        500.0,
    );
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), parent, own, berth, planet])
            .with_own_realm(OWN_REALM);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(5));
    // Berth 500 + travel 3500 = 4000: a kilometre short of the planet's centre, outside its shell.
    drive_hull(
        &mut rig,
        hull,
        DVec3::new(3500.0, 0.0, 0.0),
        DVec3::new(200.0, 0.0, 0.0),
    );
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(vd_wire::intershard::FlushSource {
            transfer: TransferId(44),
            subject: DirectoryKey::Ship(entity),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::Planet(9),
            to_parent: Some(OWN_REALM),
        }),
    )]);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(
        (
            stats.exterior_flushed,
            stats.exterior_flush_unplaceable,
            stats.flush_stale_entry
        ),
        (1, 0, 0),
        "shipped, not refused as a stale entry"
    );
    let acks = exterior_acks(&sent);
    let Some(TransferAck::SourceFlushed { pose, .. }) = acks.first() else {
        panic!("no SourceFlushed: {acks:?}");
    };
    assert_eq!(
        pose.frame,
        frame_of(RealmId::Planet(9)),
        "in the child's frame"
    );
    let at = pose.pos.delta_m(LatticePos::ORIGIN, pose.frame.tier());
    assert!(
        (at.x + 1000.0).abs() < 1e-6,
        "the planet's placement subtracted: a kilometre short of its centre: {at:?}"
    );
    assert_eq!(pose.vel, DVec3::new(200.0, 0.0, 0.0));
}

#[test]
fn a_parent_flushes_its_hulls_exterior_verbatim_in_its_own_frame_and_freezes_its_drive() {
    let (hull, entity) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(5));
    drive_hull(
        &mut rig,
        hull,
        DVec3::new(4500.0, 0.0, 0.0),
        DVec3::new(30.0, 0.0, 0.0),
    );
    rig.world
        .resource_mut::<crate::stub::drive::DrivenChildren>()
        .0
        .get_mut(&hull)
        .expect("held")
        .state
        .spin_radps = DVec3::new(0.0, 0.5, 0.0);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    // OUT: the destination is my parent, whose placement I never name — the pose ships in MY frame.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(vd_wire::intershard::FlushSource {
            transfer: TransferId(41),
            subject: DirectoryKey::Ship(entity),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: PARENT_REALM,
            to_parent: None,
        }),
    )]);
    let acks = exterior_acks(&sent);
    let Some(TransferAck::SourceFlushed {
        transfer_id,
        pose,
        state,
        ..
    }) = acks.first()
    else {
        panic!("no SourceFlushed: {acks:?}");
    };
    assert_eq!(*transfer_id, TransferId(41));
    assert_eq!(pose.frame, config().frame, "verbatim, in my own frame");
    let at = pose.pos.delta_m(LatticePos::ORIGIN, pose.frame.tier());
    assert!((at.x - 5000.0).abs() < 1e-6, "berth + travel: {at:?}");
    assert_eq!(pose.vel, DVec3::new(30.0, 0.0, 0.0));
    let blob = vd_wire::intershard::ExteriorState::decode(state).expect("the blob decodes");
    assert_eq!(blob.spin, DVec3::new(0.0, 0.5, 0.0));
    assert_eq!(blob.region.realm, hull);
    assert_eq!(blob.region.parent, None, "no parent named across the wire");
    assert_eq!(
        blob.region.center,
        vd_core::geometry::ParentCentre::ORIGIN,
        "the child's placement in my frame is not the destination's to read"
    );
    assert!(
        rig.world.resource::<crate::stub::drive::DrivenChildren>().0[&hull].frozen,
        "frozen for the hand-over"
    );
    assert_eq!(rig.world.resource::<StubStats>().exterior_flushed, 1);
    // An aborted hand-over thaws it and acks the abort.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(vd_wire::intershard::CrossingAborted {
            subject: DirectoryKey::Ship(entity),
            transfer: TransferId(41),
        }),
    )]);
    assert!(!rig.world.resource::<crate::stub::drive::DrivenChildren>().0[&hull].frozen);
    assert!(
        to_orch(&sent)
            .iter()
            .any(|f| matches!(f, InterShardFlow::CrossingAbortedAck(_)))
    );
    // A flush for an exterior this realm does not hold ships nothing and is counted.
    let stranger = EntityId::pack(EntityKind::Ship, 2, 9, 0);
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(vd_wire::intershard::FlushSource {
            transfer: TransferId(42),
            subject: DirectoryKey::Ship(stranger),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: PARENT_REALM,
            to_parent: None,
        }),
    )]);
    assert!(exterior_acks(&sent).is_empty());
    assert_eq!(rig.world.resource::<StubStats>().exterior_flush_unheld, 1);
}

#[test]
fn a_realm_adopts_an_arriving_exterior_acks_its_promote_and_releases_it_at_the_demote() {
    let (hull, entity) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    // My own forest, with a store to keep the berth in.
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 1000.0),
    ])
    .with_own_realm(OWN_REALM);
    *rig.world
        .resource_mut::<crate::stub::exterior::RealmStore>() =
        crate::stub::exterior::RealmStore(Some(Box::new(crate::io::mem::MemStore::new())));
    // The arriving exterior: the pose is in MY frame (my parent expressed it for me, a hand-DOWN),
    // stamped two ticks ago at 100 m/s along x, and the blob carries the hull's region row and spin.
    let own_frame = config().frame;
    let row = region(hull, None, DVec3::ZERO, 20.0);
    let blob = vd_wire::intershard::ExteriorState {
        region: row,
        spin: DVec3::new(0.0, 0.25, 0.0),
    };
    let pose = StampedPose {
        frame: own_frame,
        pos: LatticePos::from_metres(DVec3::new(300.0, 0.0, 0.0), own_frame.tier()),
        vel: DVec3::new(100.0, 0.0, 0.0),
        orient: DQuat::IDENTITY,
        universe_tick: UniverseTick(98),
    };
    let envelope = |transfer: u128| {
        InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(transfer),
            universe_epoch: vd_core::EpochId(1),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence: Fence(7),
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity,
                from_realm: PARENT_REALM,
                to_realm: OWN_REALM,
                pose,
                state: blob.encode(),
            },
        })
    };
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &envelope(51))]);
    // Adopted: on my roster, driven at the re-advanced pose, leased at the crossing's fence.
    let regions = rig.world.resource::<RealmRegions>();
    let adopted = regions
        .direct_child(OWN_REALM, hull)
        .expect("on the roster");
    let centre = adopted
        .center
        .in_parents_frame()
        .delta_m(LatticePos::ORIGIN, own_frame.tier());
    assert!(
        (centre.x - 310.0).abs() < 1e-6,
        "re-advanced two ticks at 100 m/s (0.05 s each): {centre:?}"
    );
    assert_eq!(adopted.shape, row.shape);
    let driven = rig.world.resource::<crate::stub::drive::DrivenChildren>();
    let held = driven.0.get(&hull).expect("driven");
    assert_eq!(held.state.vel_mps, DVec3::new(100.0, 0.0, 0.0));
    assert_eq!(held.state.spin_radps, DVec3::new(0.0, 0.25, 0.0));
    assert_eq!(held.facts, None, "the hull restates what it is");
    assert_eq!(
        rig.world.resource::<ExteriorAuthority>().0.get(&hull),
        Some(&Fence(7))
    );
    // The berth row is in my store, and the directory is asked who runs the hull.
    let berths = rig
        .world
        .resource::<crate::stub::exterior::RealmStore>()
        .0
        .as_ref()
        .expect("store")
        .scan(&crate::stub::built_store::berth_prefix());
    assert_eq!(berths.len(), 1);
    let berth = crate::stub::built_store::decode_berth(&berths[0].1).expect("decodes");
    assert_eq!(berth.child, hull);
    assert!((berth.offset_m.x - 310.0).abs() < 1e-6);
    assert_eq!(berth.fence, Fence(7));
    let flows = to_orch(&sent);
    assert!(
        flows.contains(&InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(hull),
        }))
    );
    assert!(
        exterior_acks(&sent).contains(&TransferAck::Accepted {
            transfer_id: TransferId(51),
            step_id: STUB_CROSSING_STEP,
        }),
        "the envelope is acked"
    );
    assert_eq!(rig.world.resource::<StubStats>().exterior_adopted, 1);
    // A re-emitted envelope adopts nothing new and acks again.
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &envelope(51))]);
    assert_eq!(rig.world.resource::<StubStats>().exterior_adopted, 1);
    assert_eq!(exterior_acks(&sent).len(), 1);
    // The promote is acked and counted; the dot path is not touched.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Promote(vd_wire::intershard::PromoteCmd {
            transfer: TransferId(51),
            subject: DirectoryKey::Ship(entity),
            new_fence: Fence(7),
            step_id: vd_wire::intershard::PROMOTE_STEP,
            source: NodeId(77),
        }),
    )]);
    assert!(
        to_orch(&sent).contains(&InterShardFlow::SagaAck(TransferControlAck::PromoteAck {
            transfer: TransferId(51)
        }))
    );
    assert_eq!(rig.world.resource::<StubStats>().exterior_promotes, 1);
    assert_eq!(rig.world.resource::<StubStats>().promote_no_dot, 0);
    // A RE-DELIVERED exterior promote is the journal's `AlreadyApplied`: acked again, counted once.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Promote(vd_wire::intershard::PromoteCmd {
            transfer: TransferId(51),
            subject: DirectoryKey::Ship(entity),
            new_fence: Fence(7),
            step_id: vd_wire::intershard::PROMOTE_STEP,
            source: NodeId(77),
        }),
    )]);
    assert!(
        to_orch(&sent).contains(&InterShardFlow::SagaAck(TransferControlAck::PromoteAck {
            transfer: TransferId(51)
        }))
    );
    assert_eq!(rig.world.resource::<StubStats>().exterior_promotes, 1);
    // Rows this parent keeps per child, planted for the hull AND for another child that must survive.
    let other = RealmId::Planet(9);
    {
        let observer = crate::stub::aoi::ObserverId::Dot(SessionId(1));
        let mut aoi = rig.world.resource_mut::<crate::stub::aoi::AoiMembership>();
        aoi.0
            .insert((observer, hull), crate::stub::aoi::AoiState::default());
        aoi.0.insert(
            (crate::stub::aoi::ObserverId::Child(hull), other),
            crate::stub::aoi::AoiState::default(),
        );
        aoi.0
            .insert((observer, other), crate::stub::aoi::AoiState::default());
        let mut latch = rig
            .world
            .resource_mut::<crate::stub::relay::InterestEmitLatch>();
        latch.0.insert(hull);
        latch.0.insert(other);
        let mut band = rig
            .world
            .resource_mut::<crate::stub::relay::InBandVerdict>();
        band.0.insert(hull);
        band.0.insert(other);
        let mut luma = rig.world.resource_mut::<crate::stub::window::ChildLuma>();
        luma.0.insert(hull, (3, 0.5));
        luma.0.insert(other, (4, 0.25));
        let mut owed = rig
            .world
            .resource_mut::<crate::stub::lineage::LineageOwed>();
        owed.0.insert(hull);
        owed.0.insert(other);
    }
    // Later the hull leaves ME: the demote releases it — roster, driven set, lease, berth, every
    // per-child row — and acks.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Demote(vd_wire::intershard::DemoteCmd {
            transfer: TransferId(52),
            subject: DirectoryKey::Ship(entity),
            new_owner_fence: Fence(8),
            step_id: vd_wire::intershard::DEMOTE_STEP,
        }),
    )]);
    assert!(
        rig.world
            .resource::<RealmRegions>()
            .direct_child(OWN_REALM, hull)
            .is_none()
    );
    assert!(
        !rig.world
            .resource::<crate::stub::drive::DrivenChildren>()
            .0
            .contains_key(&hull)
    );
    assert!(
        !rig.world
            .resource::<ExteriorAuthority>()
            .0
            .contains_key(&hull)
    );
    assert!(
        rig.world
            .resource::<crate::stub::exterior::RealmStore>()
            .0
            .as_ref()
            .expect("store")
            .scan(&crate::stub::built_store::berth_prefix())
            .is_empty()
    );
    assert!(
        to_orch(&sent).contains(&InterShardFlow::SagaAck(TransferControlAck::DemoteAck {
            transfer: TransferId(52)
        }))
    );
    assert_eq!(rig.world.resource::<StubStats>().exterior_released, 1);
    // The hull's rows are gone from every per-child table; the other child's rows survive.
    {
        let aoi = rig.world.resource::<crate::stub::aoi::AoiMembership>();
        let watched_or_watching_hull = aoi
            .0
            .keys()
            .filter(|(o, r)| (*r == hull) | (*o == crate::stub::aoi::ObserverId::Child(hull)))
            .count();
        assert_eq!(
            watched_or_watching_hull, 0,
            "no area-of-interest row names the hull"
        );
        assert_eq!(aoi.0.len(), 1, "the other child's row survives");
        // The latch and the verdict are re-derived from the live roster every pass, so the planted
        // `other` row (no rostered child) is pruned by the fold itself; only the hull's absence is theirs.
        let latch = rig
            .world
            .resource::<crate::stub::relay::InterestEmitLatch>();
        assert!(!latch.0.contains(&hull));
        let band = rig.world.resource::<crate::stub::relay::InBandVerdict>();
        assert!(!band.0.contains(&hull));
        let luma = rig.world.resource::<crate::stub::window::ChildLuma>();
        assert_eq!(luma.0.keys().copied().collect::<Vec<_>>(), vec![other]);
        let owed = rig.world.resource::<crate::stub::lineage::LineageOwed>();
        assert_eq!(owed.0.iter().copied().collect::<Vec<_>>(), vec![other]);
    }
    // An exterior for a realm that is not mine, or with an unreadable blob, is refused and counted.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(53),
            universe_epoch: vd_core::EpochId(1),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence: Fence(9),
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity,
                from_realm: PARENT_REALM,
                to_realm: OWN_REALM,
                pose,
                state: vec![1, 2, 3],
            },
        }),
    )]);
    assert_eq!(
        rig.world.resource::<StubStats>().exterior_arrival_refused,
        1
    );
    assert!(
        exterior_acks(&sent).is_empty(),
        "a refused exterior is not acked"
    );
}

// ===================== the ruler switch, slice 3: the hull is told =====================

fn lineage_statements(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
) -> Vec<(NodeId, vd_wire::intershard::LineageStated)> {
    sent.iter()
        .filter_map(
            |(to, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::LineageStated(ls)) => Some((*to, ls)),
                _ => None,
            },
        )
        .collect()
}

#[test]
fn a_parent_states_an_adopted_childs_lineage_on_every_head_read_until_its_facts_arrive() {
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    // The debt begins at an adoption.
    crate::stub::lineage::lineage_owed(
        hull,
        &mut rig
            .world
            .resource_mut::<crate::stub::lineage::LineageOwed>(),
    );
    let head = |node: u64| {
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(DirectoryReply::Head {
                key: DirectoryKey::Realm(hull),
                record: Some(vd_wire::seams::directory::OwnerRecord {
                    authority: AuthorityRef::Shard(NodeId(node)),
                    fence: Fence(2),
                    lease_expires: UniverseTick(1_000),
                    in_transfer: None,
                }),
            }),
        )
    };
    let sent = rig.tick(vec![head(77)]);
    let stated = lineage_statements(&sent);
    assert_eq!(stated.len(), 1, "stated once the node is known: {stated:?}");
    let (to, ls) = &stated[0];
    assert_eq!(*to, NodeId(77));
    assert_eq!(
        ls.child,
        config().own_coord.child(vd_core::worldgen::level_of(hull)),
        "my own coord plus the child's level"
    );
    assert_eq!(
        ls.parent_fence,
        Fence(1),
        "the realm fence the grant recorded"
    );
    assert_eq!(rig.world.resource::<StubStats>().lineage_stated, 1);
    // Re-stated on the next read, while the debt stands.
    let sent = rig.tick(vec![head(77)]);
    assert_eq!(lineage_statements(&sent).len(), 1);
    // The child's facts arrive: the debt is paid and the next read states nothing.
    crate::stub::lineage::lineage_heard(
        hull,
        &mut rig
            .world
            .resource_mut::<crate::stub::lineage::LineageOwed>(),
    );
    let sent = rig.tick(vec![head(77)]);
    assert!(lineage_statements(&sent).is_empty());
    assert_eq!(rig.world.resource::<StubStats>().lineage_stated, 2);
}

#[test]
fn a_hull_applies_a_lineage_only_from_the_node_the_directory_names_as_its_exterior_holder() {
    use crate::stub::lineage::{PendingLineage, apply_pending_lineage, on_lineage_stated};
    let (hull, entity) = hull_child();
    let mut cfg = StubConfig {
        realm: hull,
        own_coord: child_coord_of(OWN_REALM, hull),
        ..config()
    };
    let mut regions = RealmRegions::new(vec![
        root_region(),
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 1000.0),
        region(hull, Some(OWN_REALM), DVec3::new(500.0, 0.0, 0.0), 20.0),
    ])
    .with_own_realm(hull);
    let mut parent_node = ParentRealmNode::default();
    let mut stated = crate::stub::drive::StatedFacts(Some(vd_core::built::BuiltFacts {
        mass_g: 1,
        cross_section_mm2: 1,
        drag_micro: 1,
        max_push_micro_mps2: 1,
        max_turn_micro_radps2: 1,
    }));
    let mut was_occupied = crate::stub::aoi::WasOccupied(true);
    let mut pending = PendingLineage::default();
    let mut exterior = ExteriorAuthority::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let new_coord = StubConfig::root_coord(PARENT_REALM).child(vd_core::worldgen::level_of(hull));
    let statement = |coord: RealmCoord| vd_wire::intershard::LineageStated {
        child: coord,
        parent_fence: Fence(3),
        at: UniverseTick(50),
    };
    let record = |node: u64| vd_wire::seams::directory::OwnerRecord {
        authority: AuthorityRef::Shard(NodeId(node)),
        fence: Fence(3),
        lease_expires: UniverseTick(1_000),
        in_transfer: None,
    };
    // Unattested (nobody resolved yet): held, and my exterior head is re-read.
    on_lineage_stated(
        statement(new_coord.clone()),
        NodeId(70),
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.lineage_held_unattested, 1);
    assert_eq!(stats.lineage_applied, 0);
    assert!(pending.0.is_some());
    let asked: Vec<InterShardFlow> = outbox
        .0
        .iter()
        .filter_map(|(_, _, b, _)| postcard::from_bytes(b).ok())
        .collect();
    assert!(
        asked.contains(&InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Ship(entity),
        }))
    );
    // The directory names node 70: the held statement is applied — I moved house.
    resolve_exterior_head(
        entity,
        Some(&record(70)),
        NodeId(1),
        &cfg,
        &regions,
        &mut parent_node,
        &mut exterior,
    );
    apply_pending_lineage(
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
    );
    assert_eq!(stats.lineage_applied, 1);
    assert_eq!(cfg.own_coord, new_coord);
    assert_eq!(parent_node.0, Some(NodeId(70)));
    assert_eq!(stated.0, None, "the facts go out again, to the new parent");
    assert!(!was_occupied.0);
    assert_eq!(
        regions.parent_of(hull),
        Some(PARENT_REALM),
        "my own row now names my new parent"
    );
    // From the attested node, a statement applies at once.
    let newer = StubConfig::root_coord(OTHER_REALM).child(vd_core::worldgen::level_of(hull));
    on_lineage_stated(
        statement(newer.clone()),
        NodeId(70),
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.lineage_applied, 2);
    assert_eq!(cfg.own_coord, newer);
    // A statement about somebody else is refused; one held from a stranger is discarded when the
    // directory names another node.
    on_lineage_stated(
        statement(
            StubConfig::root_coord(PARENT_REALM).child(vd_core::worldgen::level_of(OTHER_REALM)),
        ),
        NodeId(70),
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.lineage_misrouted, 1);
    on_lineage_stated(
        statement(new_coord.clone()),
        NodeId(71),
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.lineage_held_unattested, 2);
    resolve_exterior_head(
        entity,
        Some(&record(70)),
        NodeId(1),
        &cfg,
        &regions,
        &mut parent_node,
        &mut exterior,
    );
    apply_pending_lineage(
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
    );
    assert_eq!(stats.lineage_discarded, 1);
    assert_eq!(stats.lineage_applied, 2);
    // Nothing pending: the reply arm is a no-op.
    apply_pending_lineage(
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
    );
    assert_eq!(stats.lineage_discarded, 1);
}

// ===================== the ruler switch: the exterior lane's own refusals =====================

#[test]
fn the_exterior_scan_passes_over_a_child_with_no_exterior_key_and_over_one_it_does_not_roster() {
    // Two driven children the scan must leave alone. A PLANET this realm authors has no exterior
    // key — nobody can hand a planet over — so it is never a subject, and it is not even counted as
    // a hull without a lease. A HULL this realm drives but does not roster has no berth to measure
    // against, so the scan says nothing about it until the adopt lands.
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    let stranger = RealmId::Ship(EntityId::pack(EntityKind::Ship, 2, 2, 0));
    drive_hull(
        &mut rig,
        OTHER_REALM,
        DVec3::new(4500.0, 0.0, 0.0),
        DVec3::ZERO,
    );
    drive_hull(
        &mut rig,
        stranger,
        DVec3::new(4500.0, 0.0, 0.0),
        DVec3::ZERO,
    );
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(stranger, Fence(1));
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    assert_eq!(exterior_decided(&rig), 0, "neither child is a subject");
    assert_eq!(
        rig.world.resource::<StubStats>().exterior_scan_unleased,
        0,
        "a planet is not a hull that lacks a lease"
    );
}

#[test]
fn an_abort_for_a_hull_this_realm_never_drove_is_acked_and_thaws_nothing() {
    // The ack is unconditional — the orchestrator must be told the abort landed even when the
    // subject means nothing here. A hull this realm never drove has no drive to thaw, and the
    // key still names its entity, so nothing is counted as a keyless abort.
    let mut rig = Rig::new();
    rig.grant_realm();
    let stranger = EntityId::pack(EntityKind::Ship, 3, 3, 0);
    let abort = vd_wire::intershard::CrossingAborted {
        subject: DirectoryKey::Ship(stranger),
        transfer: TransferId(91),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::CrossingAborted(abort),
    )]);
    assert!(to_orch(&sent).contains(&InterShardFlow::CrossingAbortedAck(abort)));
    assert_eq!(
        rig.world.resource::<StubStats>().crossing_abort_no_entity,
        0,
        "a hull key names its entity"
    );
}

#[test]
fn a_hull_still_outside_after_the_dwell_is_decided_again_and_its_request_is_suppressed() {
    // The dwell passes with the hull still outside, so the verdict is made a second time. The
    // request it would carry is suppressed, because the first one is still in flight: one request
    // per crossing, whatever the scan keeps deciding.
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_escape_regions_with_hull(&mut rig, hull);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(5));
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    drive_hull(&mut rig, hull, DVec3::new(4500.0, 0.0, 0.0), DVec3::ZERO);
    rig.set_local_tick(3);
    let sent = rig.tick(vec![]);
    assert_eq!(exterior_requests(&sent).len(), 1);
    // k_dwell is five ticks: at tick nine the verdict stands again, and the request does not.
    rig.set_local_tick(9);
    let sent = rig.tick(vec![]);
    assert!(exterior_requests(&sent).is_empty(), "no second request");
    assert_eq!(exterior_decided(&rig), 2, "the verdict is made twice");
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .crossings_suppressed_in_flight,
        1
    );
}

#[test]
fn a_crossing_toward_a_realm_this_forest_cannot_name_gets_no_keep_alive_and_is_counted() {
    // The hull leaves a realm whose own parent this shard does not roster, so the destination is a
    // NAME and not a place. The keep-alive that would hold the destination awake is skipped and
    // counted — never assumed, and never an expect that takes the shard down.
    let (hull, _) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    let own_live = RealmRegion {
        aoi: aoi_band(2),
        ..region(OWN_REALM, Some(PARENT_REALM), DVec3::ZERO, 1000.0)
    };
    let berth = region(hull, Some(OWN_REALM), DVec3::new(500.0, 0.0, 0.0), 20.0);
    // The ambient root of this forest is a SMALL shell that holds nothing out where the hull goes,
    // so the fold names no container and the answer is the realm's own parent — which is a name
    // this shard cannot place.
    let small_root = region(ROOT_REALM, None, DVec3::ZERO, 1000.0);
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![small_root, own_live, berth]);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(1));
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    drive_hull(&mut rig, hull, DVec3::new(4500.0, 0.0, 0.0), DVec3::ZERO);
    rig.set_local_tick(3);
    let _ = rig.tick(vec![]);
    assert_eq!(exterior_decided(&rig), 1, "the exit is decided");
    rig.set_local_tick(4);
    let _ = rig.tick(vec![]);
    assert_ne!(
        rig.world.resource::<StubStats>().crossing_keepalive_unnamed,
        0,
        "the unnamed destination is counted, not assumed"
    );
}

// The one hand-written publish, standing in for a middle link whose author has not written its row
// at this instant — the state the counter exists for. Test twin of the ONE writer.
#[allow(clippy::disallowed_methods)]
#[test]
fn a_flush_the_ephemeris_cannot_place_ships_nothing_and_is_counted() {
    // A hand-off DOWN two links — into an area inside a planet this shard co-hosts — adds each
    // link's placement through the book of the realm that authored it. When a link's book holds no
    // row at the instant the pose is stamped, there is no arithmetic to do: nothing is shipped, and
    // the refusal is counted rather than guessed.
    let (hull, entity) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    let planet = region(
        RealmId::Planet(9),
        Some(OWN_REALM),
        DVec3::new(5000.0, 0.0, 0.0),
        500.0,
    );
    let area = region_framed(
        RealmId::Area(99),
        Some(RealmId::Planet(9)),
        DVec3::ZERO,
        100.0,
        FrameRef::AreaLocal {
            planet_seed: 9,
            area_seed: 99,
        },
    );
    let berth = region(hull, Some(OWN_REALM), DVec3::new(500.0, 0.0, 0.0), 20.0);
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), planet, area, berth])
            .with_own_realm(OWN_REALM);
    rig.world
        .resource_mut::<ExteriorAuthority>()
        .0
        .insert(hull, Fence(5));
    drive_hull(&mut rig, hull, DVec3::ZERO, DVec3::ZERO);
    rig.set_local_tick(2);
    let _ = rig.tick(vec![]);
    // This realm's own ephemeris runs one tick ahead of the planet's: the pose ships stamped at the
    // instant its own book holds, and the middle link has no row at that instant.
    let ahead = UniverseTick(rig.world.resource::<ClockSample>().universe_tick.0 + 1);
    let book = rig
        .world
        .resource::<RealmRegions>()
        .author_book(OWN_REALM, 20.0, ahead);
    rig.world
        .resource_mut::<Placements>()
        .0
        .publish(OWN_REALM, book);
    rig.set_local_tick(3);
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(vd_wire::intershard::FlushSource {
            transfer: TransferId(61),
            subject: DirectoryKey::Ship(entity),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::Area(99),
            to_parent: Some(RealmId::Planet(9)),
        }),
    )]);
    assert!(exterior_acks(&sent).is_empty(), "no pose leaves");
    assert_eq!(
        rig.world.resource::<StubStats>().exterior_flush_unplaceable,
        1
    );
}

#[test]
fn a_realm_with_no_store_still_adopts_the_hull_and_simply_keeps_no_berth_row() {
    // Every test rig and every seeded-only shard runs without a store. The adoption must land all
    // the same — on the roster, driven, leased — and the durable berth row is the one thing that
    // does not happen.
    let (hull, entity) = hull_child();
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        region(OWN_REALM, Some(ROOT_REALM), DVec3::ZERO, 1000.0),
    ])
    .with_own_realm(OWN_REALM);
    let own_frame = config().frame;
    let blob = vd_wire::intershard::ExteriorState {
        region: region(hull, None, DVec3::ZERO, 20.0),
        spin: DVec3::ZERO,
    };
    let pose = StampedPose {
        frame: own_frame,
        pos: LatticePos::from_metres(DVec3::new(300.0, 0.0, 0.0), own_frame.tier()),
        vel: DVec3::ZERO,
        orient: DQuat::IDENTITY,
        universe_tick: UniverseTick(100),
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(71),
            universe_epoch: vd_core::EpochId(1),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence: Fence(7),
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity,
                from_realm: PARENT_REALM,
                to_realm: OWN_REALM,
                pose,
                state: blob.encode(),
            },
        }),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().exterior_adopted, 1);
    assert!(
        rig.world
            .resource::<RealmRegions>()
            .direct_child(OWN_REALM, hull)
            .is_some()
    );
    assert!(
        rig.world
            .resource::<crate::stub::exterior::RealmStore>()
            .0
            .is_none()
    );
}

#[test]
fn the_realm_store_says_only_whether_this_shard_holds_one() {
    // The store is a file handle: it cannot be printed, and a debug line that tried would leak the
    // whole realm. The one fact worth printing is whether the shard has a store at all.
    let empty = crate::stub::exterior::RealmStore(None);
    assert_eq!(format!("{empty:?}"), "RealmStore(false)");
    let held = crate::stub::exterior::RealmStore(Some(Box::new(crate::io::mem::MemStore::new())));
    assert_eq!(format!("{held:?}"), "RealmStore(true)");
}

#[test]
fn a_realm_that_is_not_a_hull_holds_an_unattested_lineage_and_a_rootward_move_re_points_nothing() {
    // A star system can be told its lineage too, and it has no exterior head to read: only a hull is
    // named in the directory by a `Ship` key. So the statement is held and NOTHING is asked. And a
    // coord with no parent — the realm now sits at the root of its own forest — is applied without
    // re-pointing the roster: there is no new parent to name.
    use crate::stub::lineage::{PendingLineage, on_lineage_stated};
    let mut cfg = config();
    let mut regions =
        RealmRegions::new(vec![root_region(), own_region()]).with_own_realm(OWN_REALM);
    let mut parent_node = ParentRealmNode::default();
    let mut stated = crate::stub::drive::StatedFacts(None);
    let mut was_occupied = crate::stub::aoi::WasOccupied(true);
    let mut pending = PendingLineage::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let rootward = || vd_wire::intershard::LineageStated {
        child: StubConfig::root_coord(OWN_REALM),
        parent_fence: Fence(3),
        at: UniverseTick(50),
    };
    on_lineage_stated(
        rootward(),
        NodeId(70),
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.lineage_held_unattested, 1);
    assert!(
        outbox.0.is_empty(),
        "only a hull has an exterior head to read"
    );
    // Attested now: the statement applies, and the parentless coord re-points no row.
    parent_node.0 = Some(NodeId(70));
    on_lineage_stated(
        rootward(),
        NodeId(70),
        &mut cfg,
        &mut regions,
        &mut parent_node,
        &mut stated,
        &mut was_occupied,
        &mut pending,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.lineage_applied, 1);
    assert_eq!(cfg.own_coord, StubConfig::root_coord(OWN_REALM));
    assert_eq!(
        regions.parent_of(OWN_REALM),
        Some(ROOT_REALM),
        "no new parent is named, so the roster stands"
    );
}
