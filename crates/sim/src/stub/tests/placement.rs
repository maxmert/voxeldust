//! The ONE WRITER: a parent authoring its direct children's placements and every SL1 conversion built on it. RealmSnap rows (own-frame pose, child-frame edge head, static and moving children), occupant rows restated against the shard's own frame from the placement ledger, the fed-ghost bit-identical rule, and the window lane's two emit gates (kept here because orbit() at 4719 is used at 5125); then the author_book pair and THE WORKED EXAMPLE (banner 7346) — one generated galaxy/system/planet world through the production generator, no realm placing its own parent or a sibling, the parent adding going up and subtracting going down, every arrival/flush conversion, ledger-miss and Stage-B1 re-validation arm, ending with the flush_pose_for_dest refusals at 8524; finally RealmRegions::child_placements. Locals that travel: emit_context, ledger_at, STORY_SHELL_HEADROOM.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

#[test]
fn authored_realm_snaps_computes_a_moving_child_pose_in_the_own_frame_and_ships_static_children() {
    // FA-2c: a moving child's authored `RealmSnap` is its ephemeris pose (position + velocity) at the
    // tick, stamped in the shard's OWN (ambient-root) frame — and the feed ships EVERY direct child,
    // static and moving alike (owner Q3 / D-FO-7: "movers only" was the last rival has-orbit test).
    let elements = orbit();
    let mut moving = BTreeMap::new();
    moving.insert(OTHER_REALM, elements);
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
        .with_moving_children(kepler_motion_fns(moving));
    let (tick_hz, tick) = (20.0, UniverseTick(1_000));
    let snaps =
        regions.authored_realm_snaps(OWN_REALM, &regions.author_book(OWN_REALM, tick_hz, tick));
    let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
    assert_eq!(snaps.len(), 1);
    assert_eq!(snaps[0].realm, OTHER_REALM);
    assert_eq!(
        snaps[0].pose.frame,
        frame_of(OWN_REALM),
        "authored in the SHARD'S OWN frame — the only frame a parent can author its children in, and \
         the one its own occupants are measured in. This asserted the ROOT's frame while calling it \
         the own frame; the two were the same numbers only because every fixture realm sat at the \
         origin, and the mismatch put the realm boxes and the things standing in them in different \
         spaces the moment a shard sat anywhere else."
    );
    assert_eq!(fm(snaps[0].pose.pos), state.position);
    assert_eq!(snaps[0].pose.vel, state.velocity);
    assert_eq!(snaps[0].pose.universe_tick, tick);
    // D-FO-7: a STATIC direct child ships a row too — its stored placement, zero velocity, in the
    // same own frame. This is the arm a re-introduced movers-only filter turns RED (batch review:
    // the old form asserted `is_empty()` over a CHILDLESS forest and called it "the static case",
    // so deleting the filter changed nothing here and re-adding it would not have either).
    let static_forest = RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    let static_snaps = static_forest.authored_realm_snaps(
        OWN_REALM,
        &static_forest.author_book(OWN_REALM, tick_hz, tick),
    );
    assert_eq!(
        static_snaps.len(),
        1,
        "a static direct child ships a row — the feed is every direct child, never movers-only"
    );
    assert_eq!(static_snaps[0].realm, OTHER_REALM);
    assert_eq!(static_snaps[0].pose.frame, frame_of(OWN_REALM));
    assert_eq!(static_snaps[0].pose.vel, DVec3::ZERO);
    // A CHILDLESS forest authors NO rows — empty means "no direct children", never "no movers".
    let bare = RealmRegions::new(vec![root_region(), own_region()]);
    assert!(
        bare.authored_realm_snaps(OWN_REALM, &bare.author_book(OWN_REALM, tick_hz, tick))
            .is_empty()
    );
}

#[test]
fn authored_realm_snaps_ships_the_child_frame_as_the_edge_head() {
    // proto_minor 8: each row is a complete placement EDGE — head (the child's own frame), tail
    // (this shard's own frame, on the pose) and value. A receiver reading it needs no join against
    // the reliable shape lane and no ordering guarantee between the two lanes.
    //
    // The head cannot be recovered from `realm`: `frame_for_realm` needs the PARENT to build an
    // `AreaLocal`, and `FrameRef::realm` throws that parent away going the other way. Deriving it
    // receiver-side is exactly the hierarchy lookup this field exists to remove.
    let mut moving = BTreeMap::new();
    moving.insert(OTHER_REALM, orbit());
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
        .with_moving_children(kepler_motion_fns(moving));
    let snaps = regions.authored_realm_snaps(
        OWN_REALM,
        &regions.author_book(OWN_REALM, 20.0, UniverseTick(1_000)),
    );
    assert_eq!(snaps.len(), 1);
    assert_eq!(
        snaps[0].frame,
        frame_of(OTHER_REALM),
        "the HEAD is the CHILD's own frame — the same frame `frame_context` registers this child \
         under, so the two sides of the ephemeris cannot disagree about what is being placed"
    );
    assert_eq!(
        snaps[0].pose.frame,
        frame_of(OWN_REALM),
        "the TAIL is THIS shard's own frame — the only frame a parent can author a child in"
    );
    assert_ne!(
        snaps[0].frame, snaps[0].pose.frame,
        "head and tail differ on every real row; equal would mean a realm placed inside itself, \
         and a receiver would compose that placement twice"
    );
}

#[test]
fn an_occupant_standing_in_a_child_realm_is_emitted_from_this_shards_own_centre() {
    // SYMPTOM 2, as a unit. An occupant standing inside a realm this shard hosts as a CHILD wears that
    // child's frame — and the realm lane, in the same tick, states where that child sits in THIS
    // shard's frame. Shipping the occupant's row unrestated put one shard's two feeds in two spaces one
    // level apart, so the box and the player standing at its centre drew a whole child-placement
    // apart. Measured at walk scale here: the child sits 300 m out, the occupant stands 7 m from its
    // centre, and the row that leaves says 307 — in this shard's frame, the one it speaks in.
    let child_at = 300.0;
    let inside_child = 7.0;
    let regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(
            OTHER_REALM,
            Some(OWN_REALM),
            DVec3::new(child_at, 0.0, 0.0),
            1000.0,
        ),
    ]);
    let own_frame = emit_context(&regions);
    let entity = EntityId::pack(EntityKind::Player, 10, 11, 11);
    let mut dot = slice6_dot(entity, own_frame, Authority::Owned { fence: Fence(1) });
    dot.pose = StampedPose::at_rest(
        frame_of(OTHER_REALM),
        DVec3::new(inside_child, 0.0, 0.0),
        UniverseTick(100),
    );
    let mut dots = Dots::default();
    dots.0.insert(SessionId(11), dot);

    let mut stats = StubStats::default();
    let entities = emitted_entities(
        &dots,
        &HandoffHolds::default(),
        own_frame,
        &ledger_at(&regions, 20.0, UniverseTick(100)),
        OWN_REALM,
        &mut stats,
    );
    assert_eq!(entities.len(), 1);
    assert_eq!(
        entities[0].pose.frame, own_frame,
        "the row leaves in the ONE frame this shard speaks in",
    );
    assert_eq!(
        fm(entities[0].pose.pos),
        DVec3::new(child_at + inside_child, 0.0, 0.0),
        "this shard added where it put its own child: {child_at} + {inside_child}",
    );
    assert_eq!(
        stats.entity_rows_foreign_labelled, 0,
        "a row this shard CAN place is not a degrade",
    );
}

#[test]
fn a_fed_ghost_ships_with_the_neighbours_frame_label_and_is_never_re_measured_here() {
    // THE FED-GHOST RULE, and it is the one this shard is most likely to get wrong. A fed ghost's pose
    // was authored by the NEIGHBOUR that owns the entity, measured from the NEIGHBOUR's centre and
    // labelled with the NEIGHBOUR's frame. This shard has no idea where that neighbour sits relative to
    // itself unless the neighbour is one of its DIRECT CHILDREN — so it must not touch the number, and
    // it must not quietly restamp the label to its own, which would claim the value is measured from
    // here.
    //
    // It used to be FOLDED here, against a per-tick table of universe-root absolutes keyed on the
    // ghost's own frame. That worked only because the shard believed it knew where every frame was in
    // the universe. With that belief removed, the honest answer is to forward exactly what arrived and
    // let the gateway — which holds both realms' placements — relate them.
    let local_frame = frame_of(OWN_REALM);
    let ghost_frame = frame_of(OTHER_REALM);
    let entity = EntityId::pack(EntityKind::Player, 10, 9, 9);
    let ghost_pose =
        StampedPose::at_rest(ghost_frame, DVec3::new(0.0, 0.0, 7.0), UniverseTick(100));
    let mut dot = slice6_dot(entity, local_frame, Authority::Owned { fence: Fence(1) });
    dot.pose = ghost_pose;
    let mut dots = Dots::default();
    dots.0.insert(SessionId(9), dot);

    // The forest holds this shard's own realm and NOT the ghost's, so the ghost's frame is one this
    // shard has genuinely never been told the position of — the case the rule is about.
    let regions = RealmRegions::new(vec![root_region(), own_region()]);
    let own_frame = emit_context(&regions);
    let mut stats = StubStats::default();
    let entities = emitted_entities(
        &dots,
        &HandoffHolds::default(),
        own_frame,
        &ledger_at(&regions, 20.0, UniverseTick(100)),
        OWN_REALM,
        &mut stats,
    );
    assert_eq!(entities.len(), 1);
    assert_eq!(
        entities[0].pose, ghost_pose,
        "the neighbour's pose ships BIT-IDENTICAL — same value, same neighbour frame label"
    );
    assert_eq!(
        stats.entity_rows_foreign_labelled, 1,
        "and the degrade is COUNTED, not silent",
    );
    assert_eq!(
        entities[0].pose.frame, ghost_frame,
        "the label stays the NEIGHBOUR's; restamping it local would claim the number is measured \
         from here, which is exactly the lie the fold shipped"
    );
}

#[test]
fn emitted_entities_ships_the_emitting_union_bit_identical_and_drops_the_silent() {
    // The EMIT UNION, slice F's shape: an Owned dot emits (the authority truth); a RETAINED
    // source ghost emits ONLY while its hand-off hold is open (the demote→take-over fill —
    // hold closed means the leaver already vanished from bystanders' screens). A pre-grant
    // provisional Ghost, a Frozen dot, and a hold-CLOSED retained ghost emit NOTHING. (The
    // fed-ghost category died with the pose feed.)
    //
    // And every surviving row's pose is BIT-IDENTICAL to the dot's own — value, velocity, tick
    // AND frame label. There is no compose, no re-stamp, no feed to overwrite it. A shard ships
    // what it holds.
    let frame = frame_of(OWN_REALM);
    let owned = EntityId::pack(EntityKind::Player, 10, 1, 1);
    let retained = EntityId::pack(EntityKind::Player, 10, 3, 3);
    let lapsed = EntityId::pack(EntityKind::Player, 10, 2, 2);
    let provisional = EntityId::pack(EntityKind::Player, 10, 4, 4);
    let frozen = EntityId::pack(EntityKind::Player, 10, 5, 5);

    let mut dots = Dots::default();
    dots.0.insert(
        SessionId(1),
        slice6_dot(owned, frame, Authority::Owned { fence: Fence(1) }),
    );
    // A RETAINED source ghost with an OPEN hold: granted, not departing, post-GENESIS fence.
    dots.0.insert(
        SessionId(3),
        slice6_dot(
            retained,
            frame,
            Authority::Ghost {
                source_fence: Fence(1),
                since_tick: TickId(1),
            },
        ),
    );
    // A RETAINED source ghost whose hold CLOSED (no HandoffHolds entry) — silent: the leaver
    // vanished at hold closure and must not reappear frozen.
    dots.0.insert(
        SessionId(2),
        slice6_dot(
            lapsed,
            frame,
            Authority::Ghost {
                source_fence: Fence(2),
                since_tick: TickId(1),
            },
        ),
    );
    // A PRE-GRANT provisional Ghost holding GENESIS — silent (nothing owns it yet).
    let mut provisional_dot = slice6_dot(
        provisional,
        frame,
        Authority::Ghost {
            source_fence: Fence::GENESIS,
            since_tick: TickId(1),
        },
    );
    provisional_dot.granted = false;
    dots.0.insert(SessionId(4), provisional_dot);
    dots.0.insert(
        SessionId(5),
        slice6_dot(
            frozen,
            frame,
            Authority::Frozen {
                transfer: TransferId(1),
                fence: Fence(1),
            },
        ),
    );

    let mut holds = HandoffHolds::default();
    holds.0.insert(
        (retained, HoldRole::Source),
        HandoffHold {
            opened_at: TickId(1),
            takeover_fence: Fence(1),
        },
    );

    // Every dot here already stands in this shard's OWN realm, so the restatement is the identity and
    // "bit-identical" below is asserted through the same-frame short-circuit, not through a degrade.
    let regions = RealmRegions::new(vec![root_region(), own_region()]);
    let own_frame = emit_context(&regions);
    let mut stats = StubStats::default();
    let entities = emitted_entities(
        &dots,
        &holds,
        own_frame,
        &ledger_at(&regions, 20.0, UniverseTick(100)),
        OWN_REALM,
        &mut stats,
    );
    assert_eq!(
        stats.entity_rows_foreign_labelled, 0,
        "an occupant in this shard's own realm is never a degrade",
    );
    let mut got: Vec<EntityId> = entities.iter().map(|e| e.entity).collect();
    got.sort_unstable();
    let mut want = vec![owned, retained];
    want.sort_unstable();
    assert_eq!(
        got, want,
        "Owned | retained-with-open-hold emit; a closed-hold ghost is silent"
    );

    for snap in &entities {
        let dot = dots
            .0
            .values()
            .find(|d| d.entity == snap.entity)
            .expect("every emitted row came from a dot");
        assert_eq!(
            snap.pose, dot.pose,
            "the emitted pose is BIT-IDENTICAL to the held one — frame label included"
        );
    }
}

#[test]
fn emit_realm_frames_states_its_children_only_to_an_open_window_with_authority() {
    // The window lane's two gates, both arms each: a realm states its children's placements
    // when (a) it holds its realm lease and (b) somebody holds a window on it. The old
    // "is a player standing here" gate is GONE with the direct realm datagram (Slice C2):
    // WHO is watching is the subscriber's business, and a childless or unwatched realm simply
    // states nothing.
    let plant = |rig: &mut Rig| {
        let mut moving = BTreeMap::new();
        moving.insert(OTHER_REALM, orbit());
        *rig.world.resource_mut::<RealmRegions>() =
            RealmRegions::new(vec![root_region(), own_region(), child_region()])
                .with_moving_children(kepler_motion_fns(moving));
    };
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
        static_held: None,
    };
    let realms = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<RealmId> {
        window_frames(sent)
            .into_iter()
            .flat_map(|(_, _, _, _, rows)| rows)
            .map(|s| s.realm)
            .collect()
    };

    // (a) HAPPY: granted + a child + an open window ⇒ the child's placement is stated.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant(&mut rig);
    assert_eq!(
        realms(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])),
        vec![OTHER_REALM]
    );

    // (b) NO WINDOW: granted + a child but nobody subscribed ⇒ silent.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant(&mut rig);
    assert!(realms(&rig.tick(vec![])).is_empty());

    // (c) NO CHILD: granted + an open window but an EMPTY roster ⇒ a frame with no rows (the
    // per-tick stamp a leaf still owes its subscriber), so no realm is named.
    let mut rig = Rig::new();
    rig.grant_realm();
    assert!(realms(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])).is_empty());

    // (d) NO AUTHORITY: an ungranted shard is silent even with a child and a window.
    let mut rig = Rig::new();
    plant(&mut rig);
    assert!(realms(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)])).is_empty());
}

#[test]
fn author_book_places_the_anchor_a_child_and_refuses_a_parent() {
    // (a) EMPTY forest → the anchor defaults to the AMBIENT ROOT (the detector short-circuits on
    // `is_empty` before ever authoring, but the default must still be well-formed): the root frame
    // resolves to the identity via the anchor arm, and NO other frame has a row.
    //
    // ★ RE-BASED IN S9: that default is `UniverseSpace`, not `GalaxySpace`. It was a galaxy frame
    // because that was the nearest thing to "outermost" available while the universe had no frame of
    // its own — a stand-in whose meaning had to be remembered. It names the thing it always meant now.
    let empty = RealmRegions::new(vec![]);
    let ebook = empty.author_book(OWN_REALM, 20.0, UniverseTick(0));
    assert_eq!(
        ebook.of(FrameRef::UniverseSpace),
        Some(FramePlacement::identity()),
        "empty-forest anchor defaults to the ambient root ⇒ identity",
    );
    // …and the frame it used to default to is now just another unplaced frame, which is what makes
    // the line above a statement about the root rather than about `GalaxySpace` in particular.
    assert_eq!(ebook.of(FrameRef::GalaxySpace { galaxy_seed: 0 }), None);
    assert_eq!(
        ebook.of(frame_of(OWN_REALM)),
        None,
        "an unplaced frame in an empty forest resolves to None",
    );
    // (b) THE SHIPPED MODEL, stated as three separate facts rather than one sweeping one. This used to
    // assert that EVERY region — root, own and child alike — sat at the identity, which was the retired
    // model: it held only because every fixture realm was parked at the origin, and it is the same
    // assumption that made an arriving traveller find a neighbouring star system empty.
    let child_at = DVec3::new(300.0, -40.0, 7.5);
    let placed_child = region(OTHER_REALM, Some(OWN_REALM), child_at, 1000.0);
    let regions = RealmRegions::new(vec![root_region(), own_region(), placed_child]);
    for tick in [UniverseTick(0), UniverseTick(1_000_000)] {
        let book = regions.author_book(OWN_REALM, 20.0, tick);
        assert_eq!(book.at(), tick, "the instant is a property of the table");
        // The shard's OWN realm: the identity, always. It IS its own origin.
        assert_eq!(
            book.of(own_region().frame),
            Some(FramePlacement::identity()),
            "the shard's own realm is its own origin at tick {}",
            tick.0,
        );
        // A STATIC DIRECT CHILD: at the centre this shard authors for it — NOT the identity. Nothing
        // else green asserts this, which is exactly why registering children at the origin survived so
        // long unnoticed.
        assert_eq!(
            book.of(placed_child.frame)
                .map(|p| (p.origin_cell, p.origin)),
            Some((
                placed_child.center.in_parents_frame().cell(),
                placed_child.center.in_parents_frame().offset()
            )),
            "a static direct child rides the placement its parent authored, at tick {}",
            tick.0,
        );
        // …and that placement's VALUE is exactly the authored centre (normalized carry).
        assert_eq!(
            book.of(placed_child.frame).map(|p| fm(p.anchor())),
            Some(child_at),
        );
        // The PARENT: unknown, deliberately. Nobody has told this shard where its parent is, and under
        // the ground rule nobody ever will — so a conversion involving it must fail loudly rather than
        // quietly assume the identity and answer confidently from the wrong numbers.
        assert_eq!(
            book.of(root_region().frame),
            None,
            "a shard is never told where its parent is, at tick {}",
            tick.0,
        );
    }
}

#[test]
fn author_book_writes_a_registered_moving_childs_row_from_its_ephemeris() {
    // FA-2b, restated on the one writer: a region in the MOVING roster gets its row solved from its
    // `OrbitalElements` at the BOOK's instant; a region ABSENT from the roster stays static at its
    // stored centre (the byte-identity arm). Both arms write the SAME kind of row — downstream
    // cannot tell which ran (SL4). The moving child is `child_region` (`OTHER_REALM` = Planet 42).
    let elements = OrbitalElements {
        sma: 1.5e11,
        ecc: 0.1,
        inclination: 0.4,
        raan: 0.3,
        arg_periapsis: 0.9,
        mean_anomaly_epoch: 0.2,
        central_mass: 1.989e30,
    };
    let mut moving = BTreeMap::new();
    moving.insert(OTHER_REALM, elements);
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()])
        .with_moving_children(kepler_motion_fns(moving));
    let tick_hz = 20.0;
    let tick = UniverseTick(1_000);
    let book = regions.author_book(OWN_REALM, tick_hz, tick);
    // The MOVING child's row: the ephemeris solved at the book's instant (the mover arm).
    let state = orbital_state(&elements, secs_since_epoch(tick.0, tick_hz));
    assert_eq!(
        book.of(frame_of(OTHER_REALM)),
        Some(FramePlacement::moving(state.position, state.velocity)),
        "a registered moving child's row is authored live from its orbit, not the static identity",
    );
    // A region ABSENT from the roster (`own`): the anchor identity — the byte-identity arm.
    assert_eq!(
        book.of(frame_of(OWN_REALM)),
        Some(FramePlacement::identity()),
        "a non-roster region stays at the identity placement (byte-identical to FA-1)",
    );
}

/// THE MOST VALUABLE TEST IN THIS ARC, and the reason it is planted before anything moves: it is what
/// stops a realm learning its own address again under some other name. At EVERY level of the story the
/// shard's authored book knows its own realm and its direct children and NOTHING ELSE — asking it to
/// place its own parent, or a sibling, is `None`, and asking `transfer_frame` to re-express a pose into
/// an ancestor's frame is a TYPED REFUSAL rather than a number.
///
/// The parent is not merely missing from the world: for the galaxy and the system it is a REGION the
/// shard holds and evaluates containment against. It is deliberately absent from the authored book
/// anyway, because containment is a question a realm answers about its own volume and placement is a
/// question only its parent can answer.
#[test]
fn no_shard_can_place_its_own_parent_or_a_sibling() {
    use vd_core::frame::{FrameError, transfer_frame};
    let story = Story::new();
    for (own, parent, sibling) in [
        // The galaxy: its parent is the ambient universe; the deepest realm in the world stands in
        // for "somebody else's realm" (the galaxy has no sibling — the universe holds one galaxy).
        (story.galaxy, story.universe, story.planet),
        (story.system, story.galaxy, story.sibling_system),
        (story.planet, story.system, story.sibling_planet),
    ] {
        let book = story.ctx(own);
        assert_eq!(
            book.of(story.frame(own)),
            Some(FramePlacement::identity()),
            "{own:?} IS its own origin",
        );
        assert_eq!(
            book.of(story.frame(parent)),
            None,
            "{own:?} is never told where its parent {parent:?} is",
        );
        assert_eq!(
            book.of(story.frame(sibling)),
            None,
            "{own:?} is never told where {sibling:?} is either",
        );
    }
    // A planet asking for its own galaxy-frame position must be a typed refusal, not a number.
    assert_eq!(
        transfer_frame(
            &story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.frame(story.galaxy),
            &story.ctx(story.planet),
        )
        .expect_err("a planet must not be able to answer where it is in its galaxy"),
        FrameError::UnknownDestFrame,
    );
    // And a frame no region in this world carries is refused the same way — the refusal is a property
    // of "I was not told", not of "I recognised the name and declined".
    assert_eq!(
        transfer_frame(
            &story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            FrameRef::GalaxySpace { galaxy_seed: 0 },
            &story.ctx(story.planet),
        )
        .expect_err("an unheard-of frame is a refusal, never a guess"),
        FrameError::UnknownDestFrame,
    );
}

/// GOING UP: three separate contexts, three separate additions, each made by the one party that holds
/// that number. The planet ships "3, in my frame"; the SYSTEM adds 145 → 148; the system ships "148,
/// in my frame"; the GALAXY adds 12031 → 12179. Nobody ever learns its own address on the way.
#[test]
fn the_parent_adds_its_childs_placement_going_up() {
    use vd_core::frame::transfer_frame;
    let story = Story::new();

    // The planet's own view: an occupant 3 m from its centre. This is ALL it knows.
    let at_planet = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);

    // The SYSTEM converts, because the system is the only party that holds "I put that planet at 145".
    let at_system = transfer_frame(
        &at_planet,
        story.frame(story.system),
        &story.ctx(story.system),
    )
    .expect("a star can place its own planet");
    assert_eq!(at_system.frame, story.frame(story.system));
    assert_eq!(
        fm(at_system.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the system adds its child's placement: 145 + 3",
    );

    // The GALAXY converts the result, because only the galaxy holds where it put that system.
    // Since the 3-D seeded placement law (owner Q-B) the system's authored centre is a seeded
    // 3-D direction at the story radius — the galaxy's addition is the same VECTOR add the
    // collinear story spelt on one axis: placement + (148, 0, 0).
    let at_galaxy = transfer_frame(
        &at_system,
        story.frame(story.galaxy),
        &story.ctx(story.galaxy),
    )
    .expect("a galaxy can place its own star system");
    assert_eq!(at_galaxy.frame, story.frame(story.galaxy));
    // ★ READ IN THE GALAXY'S OWN UNIT (slice S9). This pose is now counted in two-metre steps, because
    // that is what the frame it was handed into counts in — reading it in millimetres would be out by
    // 2048× and would still look like a plausible distance.
    assert_eq!(
        fm_at(at_galaxy.pos, story.frame(story.galaxy).tier()),
        Story::centre(&story.world, story.system) + DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the galaxy adds its child's placement (the seeded 3-D vector + 148 along x)",
    );
}

/// GOING DOWN: the galaxy computes 12179 − 12031 = 148 and hands it to the system; the system computes
/// 148 − 145 = 3 and hands it to the planet; the planet ACCEPTS 3 and does no arithmetic at all. The
/// last step is asserted BIT-IDENTICAL, because "the child does nothing" is the half of the ground rule
/// that an approximate assertion would let slide.
#[test]
fn the_parent_subtracts_going_down_and_the_child_does_nothing() {
    use vd_core::frame::transfer_frame;
    let story = Story::new();

    // The pose the galaxy holds: the occupant 148 m up the system's own +x, expressed in the
    // galaxy frame through the galaxy's own book (the seeded 3-D placement + the offset — the
    // collinear story's "12179 on one axis" generalized to the vector it always was).
    let at_galaxy = transfer_frame(
        &story.pose_in(story.frame(story.system), story.up_1_m()),
        story.frame(story.galaxy),
        &story.ctx(story.galaxy),
    )
    .expect("a galaxy can place its own star system");

    // The GALAXY subtracts, because it is the party that knows where it put the system —
    // BIT-EXACT: the integer anchors cancel and the residual subtraction is of equal halves.
    let at_system = transfer_frame(
        &at_galaxy,
        story.frame(story.system),
        &story.ctx(story.galaxy),
    )
    .expect("a galaxy can place its own star system");
    assert_eq!(
        fm(at_system.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the galaxy subtracts its child's placement exactly",
    );

    // The SYSTEM subtracts, because it is the party that knows where it put the planet.
    let at_planet = transfer_frame(
        &at_system,
        story.frame(story.planet),
        &story.ctx(story.system),
    )
    .expect("a star can place its own planet");
    assert_eq!(
        fm(at_planet.pos),
        DVec3::new(STORY_OCCUPANT_FROM_PLANET_M, 0.0, 0.0),
        "the system subtracts its child's placement: 148 − 145",
    );

    // The PLANET accepts and does nothing. Same frame in, same frame out, BIT-IDENTICAL — the child
    // never re-derives a number its parent has already measured for it.
    let accepted = transfer_frame(
        &at_planet,
        story.frame(story.planet),
        &story.ctx(story.planet),
    )
    .expect("a realm always knows its own frame");
    assert_eq!(
        accepted, at_planet,
        "the child accepts and does no arithmetic"
    );
}

/// THE DOWNWARD ARRIVAL: my parent already expressed this in my frame, so I accept it and do NOTHING.
/// Asserted BIT-IDENTICAL — "the child does no arithmetic" is the half of the ground rule that an
/// approximate assertion would let slide.
#[test]
fn an_arrival_already_in_my_own_frame_is_accepted_bit_identical() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let arriving = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    assert_eq!(
        place_arriving_pose(
            arriving,
            story.planet,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            &mut stats,
        )
        .expect("a realm always knows its own frame"),
        arriving,
        "the child accepts and does no arithmetic",
    );
}

/// THE UPWARD ARRIVAL, which is the entire point of the change: the planet ships "3, in my frame" and
/// the SYSTEM — the only party that holds "I put that planet at 145" — adds it and gets 148.
#[test]
fn an_arrival_from_a_direct_child_gets_that_childs_placement_added() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let placed = place_arriving_pose(
        story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
        story.system,
        &cfg,
        &story.regions(story.system),
        &story.ledger(story.system),
        &mut stats,
    )
    .expect("a star can place its own planet");
    assert_eq!(placed.frame, story.frame(story.system));
    assert_eq!(
        fm(placed.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the system adds its child's placement: 145 + 3",
    );
}

/// A SIBLING HAND-OFF IS A REFUSAL, not a relabel. Measured before the change: a pose handed
/// planet-11 → planet-22 landed as `3.0 @ PlanetCentered{22}` with `Ok` at all three hops — no error,
/// no counter, no log anywhere, and an occupant silently teleported by the distance between the two
/// planets. Nobody told this planet where its sibling is, so there is nothing it could add.
#[test]
fn an_arrival_from_a_sibling_is_refused_never_relabelled() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    assert_eq!(
        place_arriving_pose(
            story.pose_in(
                story.frame(story.sibling_planet),
                STORY_OCCUPANT_FROM_PLANET_M
            ),
            story.planet,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            &mut stats,
        )
        .expect_err("a planet must not be able to place its sibling's occupants"),
        UnplaceableArrival::ForeignFrame(FrameError::UnknownSourceFrame),
    );
}

/// A shard that booted with NO region forest knows where nothing is — including its own realm — so
/// every arrival it cannot pass through verbatim is refused. Both sides of the conversion are
/// exercised: a child-framed pose fails on the SOURCE side (that child has no placement here), and a
/// pose that happens to arrive in the context's fallback frame fails on the DESTINATION side (this
/// shard's own realm has no placement either). The shipped `shard.rs` always plants a neighbourhood;
/// this is the degenerate boot the guard must survive without inventing a position.
#[test]
fn an_arrival_at_a_shard_with_no_forest_is_refused_on_whichever_side_is_missing() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    assert_eq!(
        place_arriving_pose(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.system,
            &cfg,
            &RealmRegions::default(),
            &bare_ledger(&cfg),
            &mut stats,
        )
        .expect_err("no forest ⇒ no child placements ⇒ nothing to add"),
        UnplaceableArrival::ForeignFrame(FrameError::UnknownSourceFrame),
    );
    // An empty forest has no region to read a frame from, so the context anchors its identity on the
    // ambient-root fallback — which is NOT the frame this shard's realm is named in. So the source
    // resolves and the DESTINATION does not.
    //
    // ★ RE-BASED IN S9: that fallback is `UniverseSpace`. It was `GalaxySpace` only because the
    // universe had no frame to fall back to, and a pose arriving in a galaxy frame no longer lands on
    // the anchor — it is simply a frame this bare shard has never been told about, which fails on the
    // SOURCE side and would have tested nothing new.
    assert_eq!(
        place_arriving_pose(
            story.pose_in(FrameRef::UniverseSpace, STORY_OCCUPANT_FROM_PLANET_M),
            story.system,
            &cfg,
            &RealmRegions::default(),
            &bare_ledger(&cfg),
            &mut stats,
        )
        .expect_err("a shard with no forest cannot place its own realm either"),
        UnplaceableArrival::ForeignFrame(FrameError::UnknownDestFrame),
    );
}

/// An `Area` shard booted WITHOUT its enclosing planet cannot even NAME its own frame (an area frame
/// carries its planet's seed as well as its own), so it has no space to measure an arrival in.
#[test]
fn an_arrival_at_a_shard_that_cannot_name_its_own_frame_is_refused() {
    let mut stats = StubStats::default();
    let story = Story::new();
    let cfg = StubConfig {
        realm: RealmId::Area(9),
        held_realms: StubConfig::single_realm(RealmId::Area(9)),
        own_coord: StubConfig::root_coord(RealmId::Area(9)), // no parent ⇒ no planet seed
        ..config()
    };
    assert_eq!(
        place_arriving_pose(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            RealmId::Area(9),
            &cfg,
            &RealmRegions::default(),
            &bare_ledger(&cfg),
            &mut stats,
        )
        .expect_err("an area with no planet has no frame of its own"),
        UnplaceableArrival::UnnameableOwnFrame,
    );
}

/// THE SOURCE'S DOWNWARD HALF: handing an occupant to one of MY OWN children is arithmetic I hold, so
/// I do it — 148 in my frame becomes 3 in my planet's.
#[test]
fn a_flush_into_my_own_child_subtracts_that_childs_placement() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    let shipped = flush_pose_for_dest(
        story.pose_in(story.frame(story.system), story.up_1_m()),
        story.planet,
        &cfg,
        &story.regions(story.system),
        &story.ledger(story.system),
        UniverseTick(0),
        &mut stats,
    )
    .expect("a star can place its own planet");
    assert_eq!(shipped.frame, story.frame(story.planet));
    assert_eq!(
        fm(shipped.pos),
        DVec3::new(STORY_OCCUPANT_FROM_PLANET_M, 0.0, 0.0),
        "the system subtracts its child's placement: 148 − 145",
    );
    assert_eq!(stats.flush_unplaceable_child, 0);
}

/// THE SOURCE'S UPWARD HALF: handing an occupant to my PARENT ships the pose VERBATIM, in my own
/// frame, because I have never been told where my parent is. Bit-identical — the arm builds no
/// context and does no arithmetic at all.
///
/// This used to reach the same answer down the FAILURE path: one unconditional conversion whose
/// upward arm worked only because it errored and the degrade handed the pose back untouched. Right
/// answer, no way to tell it from a genuine fault.
#[test]
fn a_flush_up_to_my_parent_ships_the_pose_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    // OUTSIDE the release edge: under Stage B1 the departure must still be TRUE at flush time —
    // an occupant still inside the shell is a refusal (its own test below), not a verbatim ship.
    let held = story.pose_in(story.frame(story.planet), story.departed_m());
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.system,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            UniverseTick(0),
            &mut stats,
        )
        .expect("shipping upward never fails — there is nothing to compute"),
        held,
        "the child ships what it holds, tagged with its own frame",
    );
    assert_eq!(
        stats.flush_unplaceable_child, 0,
        "an upward hand-off is not an error and must never be counted as one",
    );
    assert_eq!(
        stats.flush_stale_exit, 0,
        "a departure that is still true at flush time is never refused",
    );
}

/// EVERY LEDGER-MISS ARM, exercised loudly (the placement arc S2 — HR5: a fallible selection's
/// refusal is a region, and an unexercised refusal is a belief). Each case hands the pure function
/// a ledger that genuinely lacks the book it asks for and asserts the LOUD half: the typed refusal
/// or the counted degrade, never a silently substituted instant.
#[test]
fn every_ledger_miss_arm_refuses_loudly() {
    let story = Story::new();
    let t0 = UniverseTick(0);

    // (a) THE FLUSH'S HEAD MISS: no head book for this shard's own realm ⇒ the flush refuses
    // outright (no `SourceFlushed` ⇒ the saga aborts ⇒ this shard keeps authority).
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.planet), story.departed_m()),
            story.system,
            &cfg,
            &story.regions(story.planet),
            &PlacementLedger::new(8),
            t0,
            &mut stats,
        ),
        None,
    );
    assert_eq!(stats.placement_book_miss, 1);

    // A CO-HOSTING shard (galaxy + system) — the multi-link paths live here.
    let held = std::collections::BTreeSet::from([story.galaxy, story.system]);
    let co_regions = RealmRegions::new(story.world.neighbourhood(&held));
    let co_cfg = StubConfig {
        realm: story.galaxy,
        held_realms: held.clone(),
        frame: story.frame(story.galaxy),
        own_coord: vd_core::worldgen::coord_of_realm(
            &story.world.neighbourhood(&held),
            story.galaxy,
        )
        .expect("the galaxy has a lineage"),
        ..config()
    };
    // Test twin of the ONE writer — the same stated exemption from the publish ban.
    #[allow(clippy::disallowed_methods)]
    let ledger_with = |anchors: &[RealmId]| -> PlacementLedger {
        let mut ledger = PlacementLedger::new(8);
        for &anchor in anchors {
            ledger.publish(anchor, co_regions.author_book(anchor, STORY_TICK_HZ, t0));
        }
        ledger
    };

    // (b) A PER-LINK MISS: descending galaxy → system → planet with the SYSTEM's book absent ⇒
    // the second link refuses the flush.
    //
    // ★ WHERE THE POSE STARTS, RE-DERIVED IN S9 — and the comment it replaces was wrong before S9
    // ever touched it. It read "far outside the galaxy so the departure re-validation lets it leave",
    // and 1.0e6 m was never outside a galaxy whose own extent is 4.5e16 m. It was a thousand
    // kilometres from the galaxy's CENTRE, which is a different thing entirely.
    //
    // What the climb exposed is that the real constraint runs the other way. This descent ends in a
    // PLANET's frame, which counts in millimetres, and a millimetre lattice reaches 2.25e15 m — about
    // a quarter of a light year. The galaxy's centre is 0.7 light years from this star system, so a
    // pose parked there has NO millimetre count relative to anything inside the system, and the flush
    // is refused (`BeyondReach`) before any book is consulted. That is the coordinate system telling
    // the truth, not a fault.
    //
    // So the pose starts AT THE SYSTEM — read from the galaxy's own authored placement for it, never
    // typed — which is where an occupant descending into that system would actually be.
    let mut stats = StubStats::default();
    let outside = StampedPose::at_rest(
        story.frame(story.galaxy),
        Story::centre(&story.world, story.system),
        UniverseTick(0),
    );
    assert_eq!(
        flush_pose_for_dest(
            outside,
            story.planet,
            &co_cfg,
            &co_regions,
            &ledger_with(&[story.galaxy]),
            t0,
            &mut stats,
        ),
        None,
    );
    assert_eq!(stats.placement_book_miss, 1);

    // (c) THE ENTRY RE-VALIDATION'S MISS: a pure ASCENT (planet-labelled pose, dest = the galaxy
    // itself) converts through system and galaxy, then re-validates the entry in the galaxy's own
    // PARENT's book (the universe) — absent ⇒ refused. Links hit; only the entry book is missing.
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.galaxy,
            &co_cfg,
            &co_regions,
            &ledger_with(&[story.galaxy, story.system]),
            t0,
            &mut stats,
        ),
        None,
    );
    assert_eq!(stats.placement_book_miss, 1);

    // (d) THE ENTITY FEED'S MISS: a row whose stamp has no book ships VERBATIM under its own
    // label, counted — the same counted degrade a foreign-labelled row takes.
    let mut stats = StubStats::default();
    let entity = EntityId::pack(EntityKind::Player, 10, 9, 9);
    let own_frame = story.frame(story.system);
    let mut dots = Dots::default();
    dots.0.insert(
        SessionId(77),
        slice6_dot(entity, own_frame, Authority::Owned { fence: Fence(1) }),
    );
    let rows = emitted_entities(
        &dots,
        &HandoffHolds::default(),
        own_frame,
        &PlacementLedger::new(8),
        story.system,
        &mut stats,
    );
    assert_eq!(rows.len(), 1, "the row still ships");
    assert_eq!(
        rows[0].pose,
        dots.0[&SessionId(77)].pose,
        "verbatim, never re-spaced"
    );
    assert_eq!(stats.placement_book_miss, 1);
}

/// The DETECTOR's two ledger selections, refused loudly through the FULL schedule: a durable dot
/// standing in a leaf child (an anchor the writer never authors — head miss) and a held transient
/// whose stamp fell behind the retained window (an at() miss). Each is COUNTED and the subject is
/// simply not evaluated that tick — no crossing is invented from a book nobody authored.
#[test]
fn the_detector_counts_a_missing_book_and_skips_the_subject() {
    // (a) Dot in a leaf child's frame: `book_anchor` resolves to the child, which has no children
    // and is not held ⇒ never authored ⇒ head miss.
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    let entity = EntityId::pack(EntityKind::Player, 10, 2, 0x60);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(1.0, 0.0, 0.0));
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&TRIG_SESSION)
        .expect("just inserted")
        .pose
        .frame = frame_of(OTHER_REALM);
    let sent = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<StubStats>().placement_book_miss,
        1,
        "the head miss is counted"
    );
    assert!(
        crossing_requests(&sent).is_empty(),
        "no crossing is invented from a missing book"
    );

    // (b) A held transient whose stamp is older than the retained window.
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region()]);
    let stale = StampedPose::at_rest(
        frame_of(OWN_REALM),
        DVec3::new(1.0, 0.0, 0.0),
        UniverseTick(10),
    );
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        EntityId::pack(EntityKind::Debris, 10, 2, 0x61),
        Transient {
            pose: stale,
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: stale.pos,
        },
    );
    rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<StubStats>().placement_book_miss,
        1,
        "the stale-stamp miss is counted (rig clock 100, window far narrower than 90 ticks)"
    );
}

/// The co-hosted OUTWARD destination: an occupant leaving a realm this shard co-hosts (not its
/// primary) falls to THAT realm's roster parent — one level up from where it stood, never from
/// where the shard is named.
#[test]
fn outward_dest_of_a_cohosted_realm_is_that_realms_own_parent() {
    let story = Story::new();
    let held = std::collections::BTreeSet::from([story.system, story.planet]);
    let regions = story.world.neighbourhood(&held);
    let cfg = story_config(&story, story.system);
    assert_eq!(
        outward_dest(&cfg, &regions, story.universe, story.planet),
        story.system,
        "leaving the co-hosted planet lands in the SYSTEM (its parent), not the shard's own parent"
    );
}

/// A receiver with NO AUTHORED BOOK AT ALL (its first synced tick has not run) is the ONE case
/// the arrival still refuses — typed, counted, and retried by the saga until the writer's first
/// pass lands: nothing can be placed against a world nobody has authored yet.
#[test]
fn an_arrival_before_the_first_authored_book_is_refused_typed() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    let err = place_arriving_pose(
        story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
        story.system,
        &cfg,
        &story.regions(story.system),
        &PlacementLedger::new(8),
        &mut stats,
    )
    .expect_err("no head book ⇒ nothing to place against");
    assert_eq!(
        err,
        UnplaceableArrival::BookMiss(vd_core::placement::PlacementMiss {
            anchor: story.system,
            wanted: UniverseTick(0),
            head: None,
            span: 8,
        })
    );
    assert_eq!(stats.placement_book_miss, 1);
    assert_eq!(stats.placement_skew_clamped, 0);
}

/// FORWARD CLOCK SKEW is CLAMPED and MEASURED, never a refusal and never silent (the process-tier
/// wedge's cure): a message-carried instant AHEAD of this shard's head reads the head book with the
/// instant clamped to the head's own; behind-the-window stays the loud miss; an exact hit stays
/// exact. The arrival lane then accepts the hand-off it used to refuse.
#[test]
fn a_forward_skewed_instant_clamps_to_the_head_counted_and_measured() {
    let story = Story::new();
    let ledger = story.ledger(story.system); // heads at tick 0
    let mut stats = StubStats::default();
    // THE ARRIVAL LANE ACCEPTS THE SKEWED HAND-OFF (the wedge's cure), re-stamped to the
    // receiver's own instant: an upward planet→system hand-off whose pose is stamped one tick
    // ahead of everything the system has authored.
    let cfg = story_config(&story, story.system);
    let mut ahead = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    ahead.universe_tick = UniverseTick(1);
    let placed = place_arriving_pose(
        ahead,
        story.system,
        &cfg,
        &story.regions(story.system),
        &ledger,
        &mut stats,
    )
    .expect("a skewed but healthy hand-off is accepted, never wedged");
    assert_eq!(placed.frame, story.frame(story.system));
    assert_eq!(
        placed.universe_tick,
        UniverseTick(0),
        "the pose speaks at the receiver's own instant after the clamp"
    );
    assert_eq!(
        fm(placed.pos),
        DVec3::new(story.up_1_m(), 0.0, 0.0),
        "the system adds its child's placement exactly as an un-skewed arrival: 145 + 3"
    );
    assert_eq!(stats.placement_skew_clamped, 1);
    assert_eq!(
        stats.placement_skew_ahead_max_ticks, 1,
        "the arrival's 1-tick lead rides the ahead gauge"
    );
    assert_eq!(
        stats.placement_skew_behind_max_ticks, 0,
        "a FORWARD clamp writes the ahead gauge only — the two directions are separate \
         measurements (batch review: one |Δ| magnitude let redelivery staleness drown the \
         span_ahead bound)"
    );

    // …and a RETRIED hand-off whose stamp aged BEHIND the window (the process-tier wedge: an
    // immutably-stamped envelope redelivered while the head advanced) is ALSO accepted at the
    // receiver's now — never refused forever.
    #[allow(clippy::disallowed_methods)] // test twin of the ONE writer
    let aged = {
        let mut ledger = PlacementLedger::new(2);
        for t in [100u64, 101, 102] {
            ledger.publish(
                story.system,
                story.regions(story.system).author_book(
                    story.system,
                    STORY_TICK_HZ,
                    UniverseTick(t),
                ),
            );
        }
        ledger
    };
    let mut stats = StubStats::default();
    let mut stale = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    stale.universe_tick = UniverseTick(10); // 90 ticks behind the head, window 2
    let placed = place_arriving_pose(
        stale,
        story.system,
        &cfg,
        &story.regions(story.system),
        &aged,
        &mut stats,
    )
    .expect("an aged but healthy hand-off is accepted at the receiver's now, never wedged");
    assert_eq!(placed.universe_tick, UniverseTick(102));
    assert_eq!(stats.placement_skew_clamped, 1);
    assert_eq!(
        stats.placement_skew_behind_max_ticks, 92,
        "a BACKWARD (redelivery-staleness) clamp writes the behind gauge only"
    );
    assert_eq!(stats.placement_skew_ahead_max_ticks, 0);
}

/// STAGE B1's mirrored half (§4v cure 1): the departure was decided, and by flush time the occupant
/// is back INSIDE its own realm (the run-1 measurement: 2.37 m against a 4.16 m boundary). The flush
/// refuses — no `SourceFlushed`, the saga aborts pre-commit, this shard keeps authority.
#[test]
fn a_departure_that_is_no_longer_true_is_refused_at_the_flush() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M),
            story.system,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "an occupant still inside its own realm must not be handed to the parent",
    );
    assert_eq!(stats.flush_stale_exit, 1, "the stale departure is counted");
    assert_eq!(
        stats.flush_unplaceable_child, 0,
        "a stale departure is a refusal, never an unplaceable-child fault",
    );
}

/// STAGE B1's first half (§4v cure 1): the entry was decided, and by flush time the occupant has
/// left the destination (the run-1 measurement: landed 12.15 / 23.00 / 20.72 m outside a 4.16 m
/// boundary, each an instant bounce-back). The flush refuses instead of committing a mislanding.
#[test]
fn an_entry_that_is_no_longer_true_is_refused_at_the_flush() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    // The system's own centre is one full orbital radius (145 m) from the planet — far past the
    // shell (36.25 m) plus the release edge, so the destination would never hold it.
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(story.frame(story.system), 0.0),
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "a pose the destination would not hold must not be shipped into it",
    );
    assert_eq!(stats.flush_stale_entry, 1, "the stale entry is counted");
    assert_eq!(
        stats.flush_unplaceable_child, 0,
        "a stale entry is a refusal, never an unplaceable-child fault",
    );
}

/// The self-crossing (`to_realm == config.realm`) skips the departure re-validation: "still inside
/// myself" is not a stale departure, and the pose ships verbatim exactly as before Stage B1.
#[test]
fn a_self_crossing_flush_skips_the_departure_check_and_ships_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    let held = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.planet,
            &cfg,
            &story.regions(story.planet),
            &story.ledger(story.planet),
            UniverseTick(0),
            &mut stats,
        ),
        Some(held),
        "a self-crossing ships what it holds — the launder path stays open",
    );
    assert_eq!(
        stats.flush_stale_exit, 0,
        "a self-crossing is never counted as a stale departure",
    );
}

/// A pose still LABELLED with an ancestor's frame (a realm this shard rosters but does not
/// author) may not anchor the hand-off: the anchor warn counts, the conversion path refuses the
/// ancestor as a converting parent (authorship constraint), and the pose ships VERBATIM for the
/// true author to place.
#[test]
fn a_pose_labelled_with_an_ancestors_frame_warns_and_ships_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    // Labelled with the GALAXY's frame — rostered here (the ambient chain) but never authored.
    let held = story.pose_in(
        story.frame(story.galaxy),
        story.config.stellar.system_ring_r_m + story.up_1_m(),
    );
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        Some(held),
        "an ancestor-labelled pose ships un-converted — the arithmetic is not this shard's",
    );
    assert_eq!(
        stats.flush_anchor_not_own, 1,
        "the mis-anchored label is counted where it is refused"
    );
}

/// A pose whose frame NOBODY in this roster registers cannot descend: the first converting link
/// refuses (a typed frame error, never a guess) and the flush declines the hand-off with the
/// unplaceable-child count — the one arm that serves both directions of the step list.
#[test]
fn a_descent_from_an_unregistered_frame_is_refused_and_counted() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    // A frame no region of this roster carries: the label falls back to the own realm, so the
    // path is a pure descent — whose first conversion cannot place the pose's actual frame.
    let phantom = story.pose_in(FrameRef::PlanetCentered { planet_seed: 999 }, 3.0);
    assert_eq!(
        flush_pose_for_dest(
            phantom,
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "a pose nobody here can measure is refused, never shipped under a guess",
    );
    assert_eq!(stats.flush_unplaceable_child, 1);
}

/// A roster that holds NO region for this shard's own realm cannot re-validate a departure, and the
/// guard steps aside: the pose ships verbatim exactly as before Stage B1 (the receiver stays the
/// judge). Never normal on the seed path — the roster always carries self — but the guard must not
/// invent a refusal from an absence.
#[test]
fn a_departure_check_without_an_own_region_ships_verbatim() {
    let story = Story::new();
    let cfg = story_config(&story, story.planet);
    let mut stats = StubStats::default();
    // A roster holding ONLY the ambient root: no own region, and the dest is not a direct child.
    let roster = RealmRegions::new(vec![
        story
            .regions(story.planet)
            .regions
            .iter()
            .find(|r| r.parent.is_none())
            .copied()
            .expect("every neighbourhood carries the ambient root"),
    ]);
    let held = story.pose_in(story.frame(story.planet), STORY_OCCUPANT_FROM_PLANET_M);
    let clock0 = ClockSample {
        local_tick: vd_core::TickId(0),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    assert_eq!(
        flush_pose_for_dest(
            held,
            story.system,
            &cfg,
            &roster,
            &obs_ledger(&roster, &cfg, &clock0),
            UniverseTick(0),
            &mut stats
        ),
        Some(held),
        "with no own region to measure against, the flush ships as it always did",
    );
    assert_eq!(
        stats.flush_stale_exit, 0,
        "an absent own region is never counted as a stale departure",
    );
}

/// The roster says "my direct child" and the ephemeris still cannot express the outgoing pose in it —
/// here because the dot carries a frame from a realm this shard was never told the position of. That
/// is a REAL internal contradiction: counted, logged, and the hand-off REFUSED, so this shard keeps
/// authority and the saga aborts rather than shipping a number nobody computed.
#[test]
fn a_flush_into_my_child_that_cannot_be_computed_is_refused_and_counted() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let mut stats = StubStats::default();
    assert_eq!(
        flush_pose_for_dest(
            story.pose_in(
                story.frame(story.sibling_system),
                STORY_OCCUPANT_FROM_PLANET_M
            ),
            story.planet,
            &cfg,
            &story.regions(story.system),
            &story.ledger(story.system),
            UniverseTick(0),
            &mut stats,
        ),
        None,
        "a pose this shard cannot measure is not shipped",
    );
    assert_eq!(stats.flush_unplaceable_child, 1);
}

#[test]
fn child_placements_unifies_movers_and_static() {
    // A STATIC direct child of OWN (not in the moving roster) rides its region center — place_child's
    // `None` arm — and is_direct_child accepts ONLY the OWN child (root/own excluded).
    let regions = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(
            OTHER_REALM,
            Some(OWN_REALM),
            DVec3::new(10.0, 0.0, 0.0),
            100.0,
        ),
    ]);
    let placements = regions.child_placements(OWN_REALM, 20.0, UniverseTick(5));
    assert_eq!(placements.len(), 1);
    assert_eq!(placements[0].0.realm, OTHER_REALM);
    assert_eq!(fm(placements[0].1.pos), DVec3::new(10.0, 0.0, 0.0));
    assert_eq!(placements[0].1.vel, DVec3::ZERO);
    // NORMALIZED since the cell activation: 10 m rides the integer half (10 × 1024 cells).
    assert_eq!(placements[0].1.pos.cell(), I64Vec3::new(10_240, 0, 0));
    assert_eq!(placements[0].1.pos.offset(), DVec3::ZERO);
}

#[test]
fn a_static_child_placement_carries_its_whole_cell_anchored_center() {
    // place_child's static arm used to ship `r.center.in_parents_frame().offset()` — the f64 remainder only — so a child
    // authored at a real integer cell anchor was placed as if the anchor were zero. Every region in
    // the forest today sits at cell ZERO, so nothing caught it; the row this produces is the realm
    // lane the gateway keys its placement table on, and one dropped anchor there becomes every
    // occupant in that realm drawn a cell-block away.
    let mut cell_anchored = region(
        OTHER_REALM,
        Some(OWN_REALM),
        DVec3::new(10.0, 0.0, 0.0),
        100.0,
    );
    cell_anchored.center = vd_core::geometry::ParentCentre::authored(LatticePos::at(
        I64Vec3::new(4096, 0, 0),
        DVec3::new(10.0, 0.0, 0.0),
    ));
    let regions = RealmRegions::new(vec![root_region(), own_region(), cell_anchored]);
    let placements = regions.child_placements(OWN_REALM, 20.0, UniverseTick(5));
    // BOTH halves ride the row bit-for-bit (the anchored centre is carried, never re-derived);
    // the VALUE is the total 4096 cells (4 m) + 10 m of raw offset = 14 m.
    assert_eq!(placements[0].1.pos.cell(), I64Vec3::new(4096, 0, 0));
    assert_eq!(placements[0].1.pos.offset(), DVec3::new(10.0, 0.0, 0.0));
    assert_eq!(fm(placements[0].1.pos), DVec3::new(14.0, 0.0, 0.0));
}
