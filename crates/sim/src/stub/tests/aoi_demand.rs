//! RLM Step 2, the demand-driven realm lifecycle (banner 11218): the hysteresis state machine that decides SpinUp/KeepAlive/drop per observer, the scalar primitives the demand fold reduces to (live and predictive distance, the cross-observer union, seed-lineage coord recovery, the counted coord-less ship exclusion), the evaluate_realm_aoi fold end to end through the rig, the downward interest byte of the look horizon (fail-closed admission, down-proxy, rising/beat/falling edges, self-proxy exclusion), the SL7 occupied-child liveness bit as a demand input, and the whole loop's latches, G-IDENTICAL spin-up, recheck cadence, parent/child directory head-reads, node caches, and the ChildLive bit shipped upward with its hand-off hold budget. Locals that travel: interests (13178), HOME_SHARD (13931), drive_aoi_spinup, headreads, resolve_parent_head, parented_aoi_rig, parented_aoi_rig_holding, child_live_bits. Stale empty banners at 14221/14223/14238 can be deleted once render_shape is hoisted.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

#[test]
fn aoi_transition_covers_every_hysteresis_arm() {
    let g = 3;
    // (false,true) ACQUIRE ⇒ SpinUp, grace armed.
    assert_eq!(
        aoi_transition(AoiState::default(), true, g),
        (
            Some(DemandVerb::SpinUp),
            Some(AoiState {
                was_in: true,
                grace_remaining: g
            })
        ),
    );
    // (true,true) HOLD-IN ⇒ KeepAlive, grace re-armed.
    assert_eq!(
        aoi_transition(
            AoiState {
                was_in: true,
                grace_remaining: 1
            },
            true,
            g
        ),
        (
            Some(DemandVerb::KeepAlive),
            Some(AoiState {
                was_in: true,
                grace_remaining: g
            })
        ),
    );
    // (true,false) grace > 0 ⇒ KeepAlive, grace decrements.
    assert_eq!(
        aoi_transition(
            AoiState {
                was_in: true,
                grace_remaining: 2
            },
            false,
            g
        ),
        (
            Some(DemandVerb::KeepAlive),
            Some(AoiState {
                was_in: true,
                grace_remaining: 1
            })
        ),
    );
    // (true,false) grace == 0 ⇒ DROP the key, NO demand (never a TearDown — M-1).
    assert_eq!(
        aoi_transition(
            AoiState {
                was_in: true,
                grace_remaining: 0
            },
            false,
            g
        ),
        (None, None),
    );
    // (false,false) never-in ⇒ nothing.
    assert_eq!(aoi_transition(AoiState::default(), false, g), (None, None));
}

#[test]
fn occupant_child_dist_takes_the_lesser_of_live_and_predictive() {
    let t = vd_core::pose::Tier::Fine;
    let child = LatticePos::ORIGIN;
    // A STATIC occupant (vel 0): pred == live == its distance.
    assert_eq!(
        occupant_child_dist(
            seated_pos(DVec3::new(500.0, 0.0, 0.0)),
            DVec3::ZERO,
            child,
            t,
            1.0
        ),
        500.0
    );
    // A MOVING occupant closing in: pred (300) beats live (1500) — the F7 predictive term.
    assert_eq!(
        occupant_child_dist(
            seated_pos(DVec3::new(1500.0, 0.0, 0.0)),
            DVec3::new(-1200.0, 0.0, 0.0),
            child,
            t,
            1.0
        ),
        300.0
    );
    // THE REGRESSION THIS FUNCTION EXISTS TO SURVIVE: the occupant seated the way the integrator
    // stores a position (whole-number part folded out) must measure the SAME distance as one seated
    // at rest. Reading either endpoint's leftover alone answered ~0 here, which is how every moving
    // player came to be measured as standing at their realm's origin.
    assert_eq!(
        occupant_child_dist(
            seated_pos(DVec3::new(500.0, 0.0, 0.0)),
            DVec3::ZERO,
            child,
            t,
            1.0
        ),
        occupant_child_dist(
            LatticePos::from_metres(DVec3::new(500.0, 0.0, 0.0), vd_core::pose::Tier::Fine),
            DVec3::ZERO,
            child,
            t,
            1.0
        ),
        "where the whole-number part sits cannot change the distance",
    );
    // (The cross-observer MIN/union now lives in `aoi_decide`/`union_verb`, covered by
    // `evaluate_realm_aoi_demand_order_is_stable` + the two-observer tests below.)
}

#[test]
fn union_verb_is_spinup_on_first_demand_keepalive_while_sustained() {
    // The child-level demand = the observer union: SpinUp the tick a child FIRST becomes demanded by
    // anyone, KeepAlive while any observer sustains it, nothing when none want it. (One observer ⇒
    // exactly the per-observer SpinUp→KeepAlive sequence, so the single-observer byte-shape is kept.)
    assert_eq!(union_verb(false, true), Some(DemandVerb::SpinUp)); // newly demanded by someone
    assert_eq!(union_verb(true, true), Some(DemandVerb::KeepAlive)); // still demanded
    assert_eq!(union_verb(true, false), None); // the last observer left ⇒ drop (never a TearDown)
    assert_eq!(union_verb(false, false), None); // never demanded
}

#[test]
fn region_level_recovers_seed_lineage_kinds() {
    use vd_core::realm_path::{RealmKindTag, RealmLevel};
    assert_eq!(
        region_level(&region(
            RealmId::Planet(42),
            Some(OWN_REALM),
            DVec3::ZERO,
            1.0
        )),
        RealmLevel::new(RealmKindTag::Planet, 42)
    );
    assert_eq!(
        region_level(&region(
            RealmId::System(7),
            Some(ROOT_REALM),
            DVec3::ZERO,
            1.0
        )),
        RealmLevel::new(RealmKindTag::System, 7)
    );
    // ★ A SHIP RESOLVES TOO, AND ITS WHOLE MINTED IDENTITY SURVIVES (2026-09-01). This asserted
    // `None` — the typed graceful exclusion, honest while a lineage had no ship tag and held only 64
    // bits. Both changed, so the answer changed with them.
    //
    // The identity is checked WHOLE rather than by its kind alone: a 64-bit level would drop the high
    // bits silently, and two ships minted on different machines would then share one name — two PLACES
    // with one name, which is the collision the widening exists to prevent.
    let id = EntityId::pack(EntityKind::Ship, 1, 7, 3);
    let level = region_level(&region(
        RealmId::Ship(id),
        Some(OWN_REALM),
        DVec3::ZERO,
        1.0,
    ));
    assert_eq!(level.kind, RealmKindTag::Ship);
    assert_eq!(level.seed, id.0, "the whole minted identity, not its low half");
    assert_eq!(level.to_realm_id(), RealmId::Ship(id), "and it round-trips");
}

#[test]
fn a_ship_child_takes_part_in_every_coord_lane_exactly_like_a_planet() {
    // Audit :713 — `region_level` used to `expect` on an entity-backed Ship realm, so the FIRST
    // hosted ship region would abort the whole shard the moment any lane touched it. Until P8
    // gives ships a lineage coordinate (D-SHIP-1), every coord-needing lane must EXCLUDE it:
    // typed, counted, never a panic. Since Slice C2 there are exactly TWO such lanes left (the
    // cascade targeting, the interior fan and the scene reflect died with their messages): the
    // AoI/demand fold (`aoi_decide`) and a `Child`-scope window's hop row
    // (`emit_window_frames`). The ship still counts where no coord is needed — its live bit is
    // a child OBSERVER, so an occupied ship keeps its parent warm (SL7).
    const SHIP_HOME: NodeId = NodeId(77);
    let ship_realm = RealmId::Ship(EntityId::pack(EntityKind::Ship, 1, 7, 3));
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        own_region(),
        child_region(),
        // The hosted ship: a direct child placed far from the dot (never a containment member).
        region(
            ship_realm,
            Some(OWN_REALM),
            DVec3::new(50_000.0, 0.0, 0.0),
            10.0,
        ),
    ]);
    // An occupant (the realm is not Empty), inside OWN only — outside the child and the ship.
    insert_owned_dot(
        &mut rig,
        SessionId(1),
        player(7),
        DVec3::new(5_000.0, 0.0, 0.0),
    );
    // The ship is LIVE (a fresh occupancy bit — it counts as a child observer).
    let now = rig.world.resource::<ClockSample>().local_tick;
    rig.world.resource_mut::<ChildLiveness>().0.insert(
        ship_realm,
        ChildLiveEntry {
            home: SHIP_HOME,
            fence: Fence(1),
            at: UniverseTick(1),
            last_seen: now,
        },
    );
    // …and a subscriber asks for a window scoped to the ship itself — the hop-row lane.
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Child(ship_realm),
        static_held: None,
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    // ★ A SHIP TAKES PART NOW, AND THIS TEST ASSERTED THE OPPOSITE (2026-09-01).
    //
    // It used to prove that every lane needing a lineage SKIPPED this child and counted the skip, and
    // that a window scoped to it was served nothing. That was correct while a ship had no lineage: the
    // graceful skip replaced an `expect` that aborted the whole shard on one unrepresentable region.
    //
    // A ship has a lineage now, so the skip is gone and so is its counter. What must be true instead is
    // the thing the skip made impossible: a ship is served a window like any other child. Without that,
    // a pilot could never see their own hull.
    assert!(
        !window_frames(&sent).is_empty(),
        "a window scoped to a ship is SERVED — this is what the old exclusion made impossible"
    );
}

#[test]
fn evaluate_realm_aoi_inert_without_authority() {
    // No realm lease ⇒ the authority `else` short-circuits ⇒ no demand (even with a live child + dot).
    let mut rig = Rig::new();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    assert!(demands(&rig.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_inert_empty_regions() {
    // Granted, but NO regions ⇒ the `is_empty` guard short-circuits ⇒ no demand.
    let mut rig = Rig::new();
    rig.grant_realm();
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    assert!(demands(&rig.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_unsynced_authors_nothing() {
    // D-Finding-1: a shard whose clock is NOT yet synced authors nothing (`has_synced` skips the
    // system) — no pre-sync demand even with authority + a live child + an in-range occupant.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world.resource_mut::<ClockSample>().synced = false;
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    assert!(demands(&rig.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_empty_self_report() {
    // Zero occupants ⇒ the CHILD shard self-reports its OWN realm holds nobody: exactly one
    // `Empty { child = own_coord }` (Step-3's occupancy authority — a sealed parent cannot see inside).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    assert_eq!(
        demands(&rig.tick(vec![])),
        vec![RealmDemand {
            child: StubConfig::root_coord(OWN_REALM),
            parent_fence: Fence(1),
            verb: DemandVerb::Empty,
            universe_tick: UniverseTick(100),
        }]
    );
}

#[test]
fn evaluate_realm_aoi_spinup_then_keepalive() {
    // AOI-2: an occupant reaching the child emits exactly one SpinUp keyed on `child.path()`; the next
    // tick (still in range) emits KeepAlive. The FULL demand is asserted (child, fence, verb, tick).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let want = child_coord_of(OWN_REALM, OTHER_REALM);
    assert_eq!(
        demands(&rig.tick(vec![])),
        vec![RealmDemand {
            child: want.clone(),
            parent_fence: Fence(1),
            verb: DemandVerb::SpinUp,
            universe_tick: UniverseTick(100),
        }]
    );
    assert_eq!(
        demands(&rig.tick(vec![])),
        vec![RealmDemand {
            child: want,
            parent_fence: Fence(1),
            verb: DemandVerb::KeepAlive,
            universe_tick: UniverseTick(100),
        }]
    );
}

#[test]
fn evaluate_realm_aoi_grace_then_drop() {
    // An occupant LEAVES: while grace remains the child stays demanded (KeepAlive), then its key drops
    // and it stops being demanded — and NO parent TearDown is EVER emitted (M-1 locks REVISION-1 R2).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 2),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let mut seen = Vec::new();
    seen.extend(demands(&rig.tick(vec![]))); // SpinUp (grace armed to 2)
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(3000.0, 0.0, 0.0)); // OUT of the 2000 m tear-down
    seen.extend(demands(&rig.tick(vec![]))); // KeepAlive (grace 2 → 1)
    seen.extend(demands(&rig.tick(vec![]))); // KeepAlive (grace 1 → 0)
    let after_grace = demands(&rig.tick(vec![])); // grace 0 ⇒ drop, silent
    assert!(
        after_grace.is_empty(),
        "grace expired ⇒ the child stops being demanded"
    );
    assert_eq!(
        seen.iter().map(|d| d.verb).collect::<Vec<_>>(),
        vec![
            DemandVerb::SpinUp,
            DemandVerb::KeepAlive,
            DemandVerb::KeepAlive
        ]
    );
    assert!(
        !seen.iter().any(|d| d.verb == DemandVerb::TearDown),
        "Step 2 never emits a parent TearDown"
    );
}

#[test]
fn evaluate_realm_aoi_predictive_spinup() {
    // F7: an occupant OUTSIDE the spin-up radius but whose `pos + vel·horizon` lands inside ⇒ SpinUp
    // (boot latency masked); a STATIC occupant at the same pos ⇒ NO demand.
    let horizon = StubConfig {
        boot_ticks_p99: 20,
        ..config()
    }; // horizon_s = 20 · 0.05 = 1.0
    let mut rig = Rig::with_config(horizon);
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(1500.0, 0.0, 0.0),
    );
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&TRIG_SESSION)
        .expect("the dot")
        .pose
        .vel = DVec3::new(-1000.0, 0.0, 0.0); // pred: 1500 − 1000·1.0 = 500 < 1000
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::SpinUp]
    );

    // STATIC occupant (vel 0) at the same 1500 m ⇒ live == pred == 1500 > 1000 ⇒ silent.
    let mut still = Rig::with_config(StubConfig {
        boot_ticks_p99: 20,
        ..config()
    });
    still.grant_realm();
    plant_aoi(
        &mut still,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut still,
        TRIG_SESSION,
        player(7),
        DVec3::new(1500.0, 0.0, 0.0),
    );
    assert!(demands(&still.tick(vec![])).is_empty());
}

#[test]
fn evaluate_realm_aoi_demand_order_is_stable() {
    // H-2: two children + two occupants — the emitted `Vec<RealmDemand>` is IDENTICAL regardless of the
    // occupants' `Dots` insertion order (occupants reduce to a scalar min BEFORE any emit).
    let build = |sessions: &[(SessionId, DVec3)]| -> Vec<RealmDemand> {
        let mut rig = Rig::new();
        rig.grant_realm();
        plant_aoi(
            &mut rig,
            vec![
                root_region(),
                own_region(),
                aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
                aoi_child(RealmId::Planet(43), OWN_REALM, 100.0, 3),
            ],
        );
        for (i, (s, pos)) in sessions.iter().enumerate() {
            insert_owned_dot(&mut rig, *s, player(i as u32), *pos);
        }
        demands(&rig.tick(vec![]))
    };
    let a = build(&[
        (SessionId(1), DVec3::new(500.0, 0.0, 0.0)),
        (SessionId(2), DVec3::new(700.0, 0.0, 0.0)),
    ]);
    let b = build(&[
        (SessionId(2), DVec3::new(700.0, 0.0, 0.0)),
        (SessionId(1), DVec3::new(500.0, 0.0, 0.0)),
    ]);
    assert_eq!(a, b);
    assert_eq!(
        a.len(),
        2,
        "both children reached ⇒ two SpinUps in a stable order"
    );
}

#[test]
fn evaluate_realm_aoi_evicts_a_departed_child() {
    // The AoI ledger is lazily evicted to the current roster: a child removed from the forest drops its
    // membership entry (retain_live, keyed by RealmPath) — no leak.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 3),
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let _ = rig.tick(vec![]); // SpinUp ⇒ the child's AoI state is recorded
    assert_eq!(rig.world.resource::<AoiMembership>().0.len(), 1);
    // The child leaves the roster (region removed); its state is evicted next tick.
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    let _ = rig.tick(vec![]);
    assert!(
        rig.world.resource::<AoiMembership>().0.is_empty(),
        "a departed child's AoI state is evicted"
    );
}

#[test]
fn evaluate_realm_aoi_ships_a_membership_verdict_on_acquire_and_release() {
    // The per-observer BAND MEMBERSHIP that drives the lifecycle demand is ALSO the parent's
    // SL7 VERDICT — each tick it is diffed against what the window was already told. A child
    // entering the band is ADDED; leaving is REMOVED; an UNCHANGED verdict ships nothing.
    // Ids only: since Slice C2 a parent never states what a child LOOKS like (SL3).
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0), // grace 0 ⇒ leaving is an immediate release
        ],
    );
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(500.0, 0.0, 0.0),
    );
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
        static_held: None,
    };
    // Tick 1: ACQUIRE ⇒ exactly one verdict ADDING the entered child, to the window's opener.
    let v = window_memberships(&rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]));
    assert_eq!(v.len(), 1, "one verdict on acquire");
    assert_eq!(v[0].0, GATEWAY, "routed to the window's opener");
    assert_eq!(v[0].1, WindowId(1));
    assert_eq!(v[0].2, vec![OTHER_REALM], "the entered child is added");
    assert!(v[0].3.is_empty(), "nothing removed on entry");
    // Tick 2: STILL in range (sustained) ⇒ NOTHING (only acquire/release change the verdict).
    assert!(
        window_memberships(&rig.tick(vec![])).is_empty(),
        "a sustained verdict states nothing"
    );
    // Move OUT (grace 0 ⇒ immediate release) ⇒ exactly one verdict REMOVING the departed child.
    move_dot(&mut rig, TRIG_SESSION, DVec3::new(3000.0, 0.0, 0.0));
    let v = window_memberships(&rig.tick(vec![]));
    assert_eq!(v.len(), 1, "one verdict on release");
    assert!(v[0].2.is_empty(), "nothing added on exit");
    assert_eq!(v[0].3, vec![OTHER_REALM], "the departed child is removed");
}

/// A TOMBSTONED-lane frame (the deleted per-occupant `OccupantInterest`) arriving on the
/// SignalDelta carrier is counted undecodable, never retained and never a panic — the
/// discriminant is reserved forever, its meaning is gone.
#[test]
fn a_tombstoned_occupant_interest_frame_is_counted_undecodable() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let interest = InterShardFlow::OccupantInterest(vd_wire::intershard::OccupantInterest {
        observer: AccountId(5),
        to_realm: StubConfig::root_coord(OWN_REALM),
        occupant: StampedPose::at_rest(
            config().frame,
            DVec3::new(10.0, 0.0, 0.0),
            UniverseTick(100),
        ),
        coarsen_level: 0,
    });
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(postcard::to_allocvec(&interest).expect("encode")),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 1);
    // Garbage bytes take the same arm — counted, never a panic.
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(vec![0xFF, 0xFF, 0xFF]),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 2);
}

/// The OTHER tombstoned lane, on its OWN carrier: the deleted per-occupant `ProxySceneSet` rode
/// the Saga class, whose dispatch DECODES it fine (the discriminant is reserved) — so the `Err`
/// fall-through can never count it. The explicit tombstone arm must, or the frame vanishes
/// silently. Pins the wire contract's "a received frame counts `undecodable`" for BOTH halves.
#[test]
fn a_tombstoned_proxy_scene_set_frame_is_counted_undecodable() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let reflect = InterShardFlow::ProxySceneSet(vd_wire::intershard::ProxySceneSet {
        observer: AccountId(5),
        realms: vec![],
    });
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(&reflect).expect("encode")),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 1);
}

/// Step 5 slice A — the SL7 bit's receive arm: attested upsert, mis-route drop, stale-(fence,at)
/// drop, and the lane cure's ADMISSION (findings 0/43): a sender the directory head does not name
/// is refused fail-closed — counted, a head re-read armed, the stored route untouched — until the
/// head is re-read, at which point the re-homed child's new node is believed and last-wins
/// overwrites the home (the down-lanes follow it).
#[test]
fn retain_child_live_upserts_misroutes_and_rejects_stale() {
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
    let cfg = config(); // own realm System(7), own_coord [System(7)]
    let mut store = ChildLiveness::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    // The admission map: the directory head names NodeId(41) for Planet(42).
    let mut attested = ChildRealmNodes(BTreeMap::from([(RealmId::Planet(42), NodeId(41))]));
    // The armed-head-read probe: EVERYTHING the receive pushed, decoded — asserted by equality
    // (never a filtering match: its catch-all arm would be an uncoverable region, HR5).
    let flows = |outbox: &OutboundBox| -> Vec<InterShardFlow> {
        outbox
            .0
            .iter()
            .map(|(_, _, b, _)| {
                postcard::from_bytes::<InterShardFlow>(b).expect("a pushed flow decodes")
            })
            .collect()
    };
    let head_read_for = |realm: RealmId| {
        InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(realm),
        })
    };
    let child_coord = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 7),
        RealmLevel::new(RealmKindTag::Planet, 42),
    ]))
    .expect("two-level path");
    let bit = |fence: u64, at: u64| vd_wire::intershard::ChildLive {
        child: child_coord.clone(),
        fence: Fence(fence),
        at: UniverseTick(at),
    };
    // On-target + attested (the head names the sender) ⇒ upserted, home = the sender.
    retain_child_live(
        &mut store,
        &cfg,
        bit(1, 10),
        vd_core::TickId(5),
        NodeId(41),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_received, 1);
    let held = store.0[&RealmId::Planet(42)];
    assert_eq!(
        (held.home, held.fence, held.at),
        (NodeId(41), Fence(1), UniverseTick(10))
    );
    // STALE by (fence, at): an older heartbeat from the attested node never regresses the entry.
    retain_child_live(
        &mut store,
        &cfg,
        bit(1, 9),
        vd_core::TickId(6),
        NodeId(41),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_stale, 1);
    assert_eq!(
        store.0[&RealmId::Planet(42)].home,
        NodeId(41),
        "the stale bit changed nothing"
    );
    assert_eq!(flows(&outbox), vec![], "no refusal ⇒ no re-read armed");
    // UNATTESTED (the re-homed child's FIRST bit from its new node, before the head re-read):
    // refused fail-closed — counted, the route untouched, ONE head re-read armed for that child.
    retain_child_live(
        &mut store,
        &cfg,
        bit(2, 9),
        vd_core::TickId(7),
        NodeId(44),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_unattested, 1);
    assert_eq!(
        store.0[&RealmId::Planet(42)].home,
        NodeId(41),
        "an unattested bit never refreshes the route — the zombie window is bounded here"
    );
    assert_eq!(
        flows(&outbox),
        vec![head_read_for(RealmId::Planet(42))],
        "the refusal armed the lazy head re-read (D-RLM-6 mechanism C)"
    );
    // THE HEAD RE-READ LANDS (the same reply arm the parent resolve rides): the new node is now
    // the admission answer, and the SAME bit is believed — last-wins overwrites the home.
    attested.0.insert(RealmId::Planet(42), NodeId(44));
    retain_child_live(
        &mut store,
        &cfg,
        bit(2, 9),
        vd_core::TickId(7),
        NodeId(44),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(store.0[&RealmId::Planet(42)].home, NodeId(44));
    // NO HEAD AT ALL (fail closed): a child the directory has no record for is refused too.
    attested.0.remove(&RealmId::Planet(42));
    retain_child_live(
        &mut store,
        &cfg,
        bit(3, 11),
        vd_core::TickId(8),
        NodeId(44),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_unattested, 2);
    assert_eq!(
        store.0[&RealmId::Planet(42)].home,
        NodeId(44),
        "the no-head refusal stored nothing new"
    );
    // MIS-ROUTE: a child whose parent is NOT this realm is dropped + counted (before admission —
    // WHICH REALM precedes WHO SENT).
    let foreign = vd_core::realm_coord::RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 9),
        RealmLevel::new(RealmKindTag::Planet, 42),
    ]))
    .expect("two-level path");
    retain_child_live(
        &mut store,
        &cfg,
        vd_wire::intershard::ChildLive {
            child: foreign,
            fence: Fence(9),
            at: UniverseTick(99),
        },
        vd_core::TickId(8),
        NodeId(45),
        &attested,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.child_live_misrouted, 1);
    assert_eq!(store.0.len(), 1, "the mis-route stored nothing");
}

/// THE INTEREST BYTE's admission (look horizon slice 4, §2 ASK B's fail-closed shape),
/// every refusal arm driven by name: mis-route, unattested sender (which arms the lazy
/// PARENT head re-read — the re-home backstop mirroring `retain_child_live`), an unlawful
/// value, a deposed incarnation's stale byte — and the lawful path holding the LATEST value
/// whole, `0` included, so the fence ordering survives an explicit switch-off.
#[test]
fn the_interest_byte_admits_fail_closed_and_holds_the_latest_lawful_value() {
    let story = Story::new();
    let cfg = story_config(&story, story.system);
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(0),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let parent = ParentRealmNode(Some(NodeId(40)));
    let mut held = InterestHeld::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    // ★ S10: the signal carries a DISTANCE (metres) or `None` for nobody looking, where it carried a
    // 1/0 byte. `look` reads the same at every call site — `1` becomes "someone, close by", `0`
    // becomes "nobody" — so these cases still say what they always said.
    let ri =
        |child: RealmCoord, fence: u64, at: u64, look: u8| vd_wire::intershard::RealmInterest {
            child,
            parent_fence: Fence(fence),
            at: UniverseTick(at),
            look_inside_from_m: (look == 1).then_some(0.0),
        };
    // MIS-ROUTE: a coord lowering to somebody else (the system's own planet) drops counted.
    let planet_coord = cfg
        .own_coord
        .child(level_of(story.planet));
    on_realm_interest(
        ri(planet_coord, 5, 1, 1),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_misrouted, 1);
    assert_eq!(held.0, None, "nothing believed");
    // UNATTESTED: right coord, wrong sender — refused + the lazy parent head re-read armed.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 5, 1, 1),
        NodeId(41),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_unattested, 1);
    assert_eq!(held.0, None);
    let parent_realm = cfg
        .own_coord
        .parent()
        .expect("the story system has a parent")
        .lowered();
    assert_eq!(outbox.0.len(), 1, "the unattested arm arms ONE re-read");
    let (to, _, bytes, _) = &outbox.0[0];
    assert_eq!(*to, cfg.orchestrator);
    assert_eq!(
        postcard::from_bytes::<InterShardFlow>(bytes).expect("the re-read decodes"),
        InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(parent_realm),
        }),
        "…and it is the PARENT's head read (the re-home backstop)"
    );
    // UNATTESTED AT A ROOT: a realm with no parent coord refuses the same way but has no
    // parent head to re-read — the re-read arm is skipped, nothing armed, nothing believed.
    let root_cfg = config(); // own_coord = root_coord(System(7)) — parent() is None
    let outbox_before = outbox.0.len();
    on_realm_interest(
        ri(root_cfg.own_coord.clone(), 5, 1, 1),
        NodeId(41),
        &root_cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_unattested, 2);
    assert_eq!(
        outbox.0.len(),
        outbox_before,
        "a rootward refusal arms no re-read — there is no parent to resolve"
    );
    // UNLAWFUL VALUE. ★ S10: the signal carries a distance, so "unlawful" changed meaning — it was
    // "a byte other than 1 or 0", it is now "a number that is not a distance". A NEGATIVE distance is
    // the case a buggy parent would actually produce (a subtraction that ran the wrong way), and it
    // must be refused rather than clamped, or the receiver would silently wake its whole interior.
    on_realm_interest(
        vd_wire::intershard::RealmInterest {
            child: cfg.own_coord.clone(),
            parent_fence: Fence(5),
            at: UniverseTick(1),
            look_inside_from_m: Some(-1.0),
        },
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_unlawful, 1);
    assert_eq!(held.0, None);
    // …and the OTHER way a number stops being a distance: not finite. Both arms are driven, because a
    // guard with one arm tested is a guard nobody has checked the shape of.
    on_realm_interest(
        vd_wire::intershard::RealmInterest {
            child: cfg.own_coord.clone(),
            parent_fence: Fence(5),
            at: UniverseTick(1),
            look_inside_from_m: Some(f64::NAN),
        },
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_unlawful, 2);
    assert_eq!(held.0, None);
    // LAWFUL 1: held whole.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 5, 2, 1),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_received, 1);
    assert_eq!(
        held.0,
        Some(InterestEntry {
            fence: Fence(5),
            at: UniverseTick(2),
            seen: clock.local_tick,
            from_m: Some(0.0),
        })
    );
    // STALE: a deposed incarnation (fence below the held one) never regresses the entry.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 4, 9, 0),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_stale, 1);
    assert_eq!(
        held.0.map(|e| e.from_m),
        Some(Some(0.0)),
        "the zombie changed nothing"
    );
    // LAWFUL 0 at a fresher stamp: stored WHOLE — the ordering survives the switch-off.
    on_realm_interest(
        ri(cfg.own_coord.clone(), 5, 3, 0),
        NodeId(40),
        &cfg,
        &clock,
        &parent,
        &mut held,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.realm_interest_received, 2);
    assert_eq!(
        held.0,
        Some(InterestEntry {
            fence: Fence(5),
            at: UniverseTick(3),
            seen: clock.local_tick,
            from_m: None,
        }),
        "an explicit 0 is held, not erased — the fence high-water survives"
    );
}

/// ★ A DISTANT LOOKER NO LONGER WAKES THE INTERIOR (slice S10; owner-approved under SL6 2026-08-26).
///
/// THIS IS THE TEST THE CHANGE EXISTS FOR, so it is written to be able to fail. Before S10 the signal
/// carried a yes/no, and a realm that received it inserted a proxy at its own centre with reach equal to
/// its own extent — so every child's distance clamped to ZERO and every child woke, by construction. At
/// the target census that is 150,000 realms woken every tick, and no change of loop shape can help,
/// because every child really is in range.
///
/// The signal carries the looker's DISTANCE now. The direction is absent by law (SL2 forbids a pose
/// crossing a realm boundary), so the honest reading is the NEAREST the looker could be: its distance to
/// me, less the child's distance from my centre. The child's OWN band — already derived from angular
/// size by the generator — then decides. Nothing new judges what is worth waking.
///
/// The same fixture is driven at two distances and the answers must DIFFER. A test that only asserted
/// "close wakes it" would pass against the old wake-everything behaviour unchanged.
#[test]
fn a_distant_looker_wakes_nothing_and_a_near_one_still_wakes_the_interior() {
    // The child sits 100 m out with a band that spins up at 1000 m — so a looker 50 m outside my centre
    // is comfortably inside it, and one 100 km out is nowhere near.
    let build = |from_m: f64| {
        let mut rig = Rig::new();
        rig.grant_realm();
        let child = RealmRegion {
            ..aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0)
        };
        plant_aoi(&mut rig, vec![root_region(), own_region(), child]);
        rig.world
            .resource_mut::<ChildRealmNodes>()
            .0
            .insert(OTHER_REALM, NodeId(70));
        rig.world.resource_mut::<ParentRealmNode>().0 = Some(ANCESTOR);
        let msg = InterShardFlow::RealmInterest(vd_wire::intershard::RealmInterest {
            child: rig.world.resource::<StubConfig>().own_coord.clone(),
            parent_fence: Fence(5),
            at: UniverseTick(50),
            look_inside_from_m: Some(from_m),
        });
        let sent = rig.tick(vec![Inbound::Wire {
            from: ANCESTOR,
            class: MsgClass::Saga,
            bytes: crate::io::bytes(postcard::to_allocvec(&msg).expect("encode")),
        }]);
        demands(&sent)
    };
    let child_coord = child_coord_of(OWN_REALM, OTHER_REALM);
    let woke = |d: &[vd_wire::intershard::RealmDemand]| {
        d.iter()
            .any(|d| (d.child == child_coord) & (d.verb == DemandVerb::SpinUp))
    };

    // NEAR: the looker is 50 m from my centre, the child 100 m the other way — nearest approach is
    // zero, well inside the child's band. It wakes, exactly as the yes/no byte always made it.
    assert!(
        woke(&build(50.0)),
        "a looker at the door wakes the interior"
    );

    // FAR: the looker is 100 km away. The nearest it could be to a child 100 m from my centre is
    // 99,900 m — a hundred times past that child's 1000 m spin-up radius. NOTHING wakes.
    //
    // ★ THIS ASSERTION IS THE WHOLE SLICE. Under the old signal it was FALSE: the proxy's reach was my
    // own extent, every distance clamped to zero, and this child woke at any looker distance at all.
    let far = build(100_000.0);
    assert!(
        !woke(&far),
        "a looker 100 km out must wake nothing — it cannot resolve anything in here: {far:?}"
    );
    // …and it is not vacuously empty because the fold refused to run: the near case above came off the
    // SAME fixture and the same dispatch, and produced a demand.
    assert!(
        far.iter().all(|d| d.child != child_coord),
        "no demand of any verb for the child, not merely no SpinUp"
    );
}

/// THE DOWN-PROXY, AND THE LOOK PASSES ON (owner ruling 2026-08-29 — the cascade cap is DELETED).
///
/// A vacated realm holding a live interest byte inserts ONE synthetic observer at its own centre,
/// and its existing fold demands its in-band interior. The OCCUPANCY truths are unchanged: it still
/// self-reports Empty and ships NO `ChildLive` bit. A byte from outside never manufactures an
/// occupancy bit, and that is what this test guards hardest.
///
/// ★ WHAT CHANGED, AND WHY. This test used to assert the opposite of its last line: a realm woken by
/// a message could not pass the message on. That cap was a HOP COUNT, and SL7 says in plain words
/// "no depth or hop count anywhere". It also fixed the reach at two levels, so a city could wake its
/// districts and a district could wake nothing — no matter how deeply a world nests.
///
/// Depth now limits ITSELF: every level tests the child's OWN radius, and radii shrink inward. What
/// still cannot happen is a realm flagging its own occupied child, and
/// [`a_childs_own_proxy_never_flags_it_interested_but_still_warms_its_sibling`] holds that line.
#[test]
fn a_live_interest_byte_wakes_the_interior_reports_empty_and_passes_the_look_on() {
    let mut rig = Rig::new();
    rig.grant_realm();
    // The child carries a LIVE interior band + a resolved head route: everything an unlawful
    // cascade would need is present, on purpose.
    let child = RealmRegion {
        ..aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0)
    };
    plant_aoi(&mut rig, vec![root_region(), own_region(), child]);
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    rig.world.resource_mut::<ParentRealmNode>().0 = Some(ANCESTOR);
    // The byte arrives through the REAL dispatch (the reliable peer carrier the relay
    // rides), from the resolved parent head — the same admission production runs.
    let byte = InterShardFlow::RealmInterest(vd_wire::intershard::RealmInterest {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        parent_fence: Fence(5),
        at: UniverseTick(50),
        // ★ S10: "someone is looking in, and they are right here" — distance zero is the strongest
        // case the signal can state, and it is what the yes/no byte used to mean by construction.
        look_inside_from_m: Some(0.0),
    });
    let sent = rig.tick(vec![Inbound::Wire {
        from: ANCESTOR,
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(&byte).expect("encode")),
    }]);
    let d = demands(&sent);
    let child_coord = child_coord_of(OWN_REALM, OTHER_REALM);
    assert!(
        d.iter()
            .any(|d| (d.child == child_coord) & (d.verb == DemandVerb::SpinUp)),
        "the down-proxy woke the interior: {d:?}"
    );
    assert!(
        d.iter()
            .any(|d| (d.child.lowered() == OWN_REALM) & (d.verb == DemandVerb::Empty)),
        "…while the realm STILL truthfully reports Empty (occupancy is occupancy): {d:?}"
    );
    let bits = sent
        .iter()
        .filter(|(_, _, b)| {
            matches!(
                postcard::from_bytes::<InterShardFlow>(b),
                Ok(InterShardFlow::ChildLive(_))
            )
        })
        .count();
    assert_eq!(
        bits, 0,
        "a byte from outside must never manufacture an occupancy bit"
    );
    let onward = interests(&sent);
    assert_eq!(
        onward.len(),
        1,
        "the look passes on: route + live radius + in-band synthetic observer {onward:?}"
    );
    assert!(
        onward[0].1.look_inside_from_m.is_some(),
        "and it carries a distance, so the next realm can judge its own children: {onward:?}"
    );
    assert!(
        rig.world
            .resource::<InBandVerdict>()
            .0
            .contains(&OTHER_REALM),
        "the shared verdict names the interior (§3.4.5 — the outside watcher's forward gate)"
    );
    // AN EXPLICIT 0 (still fresh, held whole): powers NO proxy — the fold runs empty.
    let off = InterShardFlow::RealmInterest(vd_wire::intershard::RealmInterest {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        parent_fence: Fence(5),
        at: UniverseTick(51),
        look_inside_from_m: None,
    });
    rig.set_local_tick(2);
    let sent = rig.tick(vec![Inbound::Wire {
        from: ANCESTOR,
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(&off).expect("encode")),
    }]);
    assert_eq!(
        rig.world.resource::<InterestHeld>().0.map(|e| e.from_m),
        Some(None),
        "the switch-off was admitted through the dispatch and held whole"
    );
    assert!(
        demands(&sent)
            .iter()
            .all(|d| d.child != child_coord_of(OWN_REALM, OTHER_REALM)),
        "a held 0 wakes nothing — do-not-assume is as lawful as assume"
    );
    // DECAY: the lane goes silent past the derived TTL ⇒ the byte is gone, the wake ends
    // (the TTL base is the 0-entry's own `seen = 2`).
    rig.set_local_tick(2 + retain_ttl_ticks(&config()) + 1);
    let sent = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<InterestHeld>().0,
        None,
        "silence decays to nobody-is-watching"
    );
    let d = demands(&sent);
    assert!(
        d.iter().all(|d| d.child != child_coord),
        "the expired byte demands nothing: {d:?}"
    );
}

/// THE INTEREST EMISSION's edges (look horizon slice 4, §2 ASK B / G-INTEREST-BAND's unit
/// half): a `1` fires the tick the band is first entered (the transition itself — the
/// finding-41 doctrine), re-asserts on the AoI beat and ONLY the beat while the band holds
/// (the hysteresis hold zone included), and ONE explicit `0` fires on the falling edge; the
/// dead zone between spin-up and tear-down admits nobody fresh (the band-edge gate: it
/// spins up none).
#[test]
fn the_interest_emission_rises_on_entry_beats_and_falls_to_zero_at_the_band() {
    let mut rig = Rig::new();
    rig.grant_realm();
    // A hand-derived interior band: spin-up 400 m, tear-down 400 + 2·0.05·2.5 = 400.25 m.
    let band = vd_core::geometry::AoiConfig::for_velocity_safe(400.0, 1.0, 1.0, 2.0, 0.05, 0, 0.5)
        .expect("a live interior band");
    let child = RealmRegion {
        aoi: band,
        ..aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0)
    };
    plant_aoi(&mut rig, vec![root_region(), own_region(), child]);
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    let child_coord = child_coord_of(OWN_REALM, OTHER_REALM);
    // NO ROUTE YET: the occupant stands in band before the child's head is resolved —
    // nothing is sent, the latch stays untouched, and the NEXT pass (route in hand) still
    // fires the rising edge (fail-closed, self-healing — never through the orchestrator).
    rig.world.resource_mut::<ChildRealmNodes>().0.clear();
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(350.0, 0.0, 0.0),
    );
    rig.set_local_tick(10);
    assert!(
        interests(&rig.tick(vec![])).is_empty(),
        "an unresolved head sends nothing — and must not consume the edge"
    );
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    // RISING EDGE, off-beat: an occupant enters the band ⇒ the 1 ships at once, direct to
    // the child's attested head.
    rig.set_local_tick(11);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(ints.len(), 1, "the rising edge ships immediately");
    assert_eq!(
        ints[0].0,
        NodeId(70),
        "direct to the child's head, nowhere else"
    );
    assert!(ints[0].1.look_inside_from_m.is_some());
    assert_eq!(ints[0].1.child, child_coord);
    // Between beats: send-on-beat holds its tongue.
    rig.set_local_tick(12);
    assert!(
        interests(&rig.tick(vec![])).is_empty(),
        "no beat, no edge, no byte"
    );
    // On the beat: re-asserted (the receiver's TTL is sized against this).
    rig.set_local_tick(20);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(ints.len(), 1);
    assert!(ints[0].1.look_inside_from_m.is_some());
    // THE HOLD ZONE: between spin-up and tear-down an acquired child is kept…
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(400.1, 0.0, 0.0),
    );
    rig.set_local_tick(30);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(ints.len(), 1, "held inside the dead zone");
    // ★ AND THE DISTANCE IS REAL (S10), not a flag wearing a float. The dot sits at 400.1 m; the
    // signal must carry how far it is from the child, or the receiver cannot cull by size and is
    // back to waking everything. PINNED so a future change that ships a constant here goes red.
    assert_eq!(
        ints[0].1.look_inside_from_m,
        Some(400.1),
        "the dot stands at 400.1 m and the signal says so — a flag would read Some(0.0) here"
    );
    // FALLING EDGE, off-beat: past tear-down the explicit 0 ships once.
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(401.0, 0.0, 0.0),
    );
    rig.set_local_tick(31);
    let ints = interests(&rig.tick(vec![]));
    assert_eq!(
        ints.len(),
        1,
        "the falling edge ships the explicit 0 at once"
    );
    assert_eq!(ints[0].1.look_inside_from_m, None);
    // OUT, on a beat: silent (nothing to re-assert).
    rig.set_local_tick(40);
    assert!(interests(&rig.tick(vec![])).is_empty());
    // THE BAND EDGE from outside: the dead zone admits nobody fresh — it spins up none.
    insert_owned_dot(
        &mut rig,
        TRIG_SESSION,
        player(7),
        DVec3::new(400.1, 0.0, 0.0),
    );
    rig.set_local_tick(50);
    assert!(
        interests(&rig.tick(vec![])).is_empty(),
        "between spin-up and tear-down, a FRESH approach is not yet interest"
    );
}

/// THE SELF-PROXY EXCLUSION (the warp gate's red of 2026-08-17, root-caused to this arm):
/// a child's OWN occupied-child proxy never produces interest TO that child — the byte
/// exists to make "something OUTSIDE may be looking in" representable (Q1's exact
/// sentence), and a child's own occupants are inside it, already held, already waking its
/// interior. Without the exclusion the universe flags its occupied galaxy, the galaxy's
/// down-proxy reaches its own full extent, and EVERY system wakes — §2 ASK B's priced and
/// rejected alternative, measured live. The SL7 sibling-warming STAYS: the same proxy
/// still produces interest to a SIBLING inside its band.
#[test]
fn a_childs_own_proxy_never_flags_it_interested_but_still_warms_its_sibling() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let band = vd_core::geometry::AoiConfig::for_velocity_safe(400.0, 1.0, 1.0, 2.0, 0.05, 0, 0.5)
        .expect("a live interior band");
    let child = RealmRegion {
        aoi: band,
        ..aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0)
    };
    let sibling = RealmId::Planet(43);
    let sibling_region = RealmRegion {
        aoi: band,
        ..region(sibling, Some(OWN_REALM), DVec3::new(200.0, 0.0, 0.0), 100.0)
    };
    plant_aoi(
        &mut rig,
        vec![root_region(), own_region(), child, sibling_region],
    );
    {
        let mut nodes = rig.world.resource_mut::<ChildRealmNodes>();
        nodes.0.insert(OTHER_REALM, NodeId(70));
        nodes.0.insert(sibling, NodeId(71));
    }
    // The child's fresh occupancy bit makes it an OCCUPIED-CHILD observer standing at its
    // own placement (the origin), reaching its own 100 m extent.
    rig.world.resource_mut::<ChildLiveness>().0.insert(
        OTHER_REALM,
        ChildLiveEntry {
            home: NodeId(70),
            fence: Fence(1),
            at: UniverseTick(1),
            last_seen: vd_core::TickId(9),
        },
    );
    rig.set_local_tick(10); // a beat
    let ints = interests(&rig.tick(vec![]));
    let child_coord = child_coord_of(OWN_REALM, OTHER_REALM);
    let sibling_coord = child_coord_of(OWN_REALM, sibling);
    assert!(
        ints.iter().all(|(_, ri)| ri.child != child_coord),
        "a child's own proxy must never flag the child itself: {ints:?}"
    );
    let to_sibling: Vec<_> = ints
        .iter()
        .filter(|(_, ri)| ri.child == sibling_coord)
        .collect();
    assert_eq!(
        to_sibling.len(),
        1,
        "…while the SAME proxy still warms the sibling inside its band: {ints:?}"
    );
    assert_eq!(to_sibling[0].0, NodeId(71));
    assert!(to_sibling[0].1.look_inside_from_m.is_some());
}

#[test]
fn ttl_alive_and_retain_ttl_are_derived_and_bridge_one_loss() {
    // TTL is DERIVED, never a magic number: the widest of the 1 s loiter constant (round(1.0 /
    // 0.05) = 20 ticks) and TWO beats of the bit's own cadence plus one tick of slack (finding 41
    // — the bit beats per cadence now, so the TTL must bridge one LOST BEAT, not one lost tick).
    // Here the disarmed recheck derives cadence hz/2 = 10 ⇒ the beats term (21) edges out the
    // loiter term (20).
    let normal = StubConfig {
        tick_dt_s: 0.05,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&normal), 10);
    assert_eq!(retain_ttl_ticks(&normal), 21);
    // The SHIPPED dev profile (50 Hz, recheck 25): loiter 50 vs beats 51 — the old exactly-two-
    // beats-zero-slack coincidence is now a derivation with slack.
    let shipped = StubConfig {
        tick_dt_s: 0.02,
        realm_recheck_interval: 25,
        ..config()
    };
    assert_eq!(retain_ttl_ticks(&shipped), 51);
    // A degenerate dt: the cadence term dominates everything (an absurd dt derives an absurd
    // cadence; every REAL profile has dt > 0, where the two derived terms above decide).
    let degenerate = StubConfig {
        tick_dt_s: 0.0,
        ..config()
    };
    assert_eq!(
        retain_ttl_ticks(&degenerate),
        RETAIN_TTL_CADENCE_BEATS * aoi_recheck_cadence(&degenerate) + 1
    );
    // Alive predicate: age 0 and age == ttl are alive; age == ttl + 1 has expired.
    assert!(ttl_alive(vd_core::TickId(10), vd_core::TickId(10), 3));
    assert!(ttl_alive(vd_core::TickId(10), vd_core::TickId(13), 3));
    assert!(!ttl_alive(vd_core::TickId(10), vd_core::TickId(14), 3));
}

#[test]
fn an_occupied_child_bit_warms_the_sibling_before_the_empty_gate() {
    // SL7's occupied-child proxy (Step 5 slice B): a parent with ZERO local dots still warms an
    // occupied child's NEIGHBOURHOOD — the child observer stands at the placement THIS parent
    // authors, reaching as far as the child's own extent, and folds into `observers` BEFORE the
    // emptiness gate. A live bit on Planet(42) (at the origin, extent 1000) warms the NEARBY
    // sibling Planet(43) at 500 m — and keeps ITSELF demanded, which is what lets the parent's
    // KeepAlive shadow the child's own liveness through a hand-off.
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    let verbs: BTreeMap<RealmId, DemandVerb> = demands(&rig.tick(vec![]))
        .iter()
        .map(|d| (d.child.lowered(), d.verb))
        .collect();
    assert_eq!(
        verbs.get(&RealmId::Planet(43)),
        Some(&DemandVerb::SpinUp),
        "the near sibling warms off the occupied child's bit alone"
    );
    assert_eq!(
        verbs.get(&RealmId::Planet(42)),
        Some(&DemandVerb::SpinUp),
        "the occupied child keeps itself demanded (dist 0, reach its own extent)"
    );
    // ...and the child observer carries its OWN per-child hysteresis latch on the sibling.
    // ★ S10: the latch is keyed by the sibling's own id now, not by its lineage path — the path was
    // a heap list allocated per lookup for a question about identity.
    assert!(
        rig.world
            .resource::<AoiMembership>()
            .0
            .contains_key(&(ObserverId::Child(RealmId::Planet(42)), RealmId::Planet(43))),
        "the child observer has its own latch"
    );
    // A FAR sibling (10 km against a 1000 m band + 1000 m reach) is NOT warmed.
    let mut far_rig = parent_with_two_planet_children(DVec3::new(10_000.0, 0.0, 0.0));
    inject_bit(&mut far_rig, RealmId::Planet(42));
    let far_verbs: BTreeMap<RealmId, DemandVerb> = demands(&far_rig.tick(vec![]))
        .iter()
        .map(|d| (d.child.lowered(), d.verb))
        .collect();
    assert!(
        !far_verbs.contains_key(&RealmId::Planet(43)),
        "a sibling beyond the band + reach is not warmed: {far_verbs:?}"
    );
}

#[test]
fn a_live_child_bit_keeps_the_parent_non_empty() {
    let sibling = DVec3::new(500.0, 0.0, 0.0);
    // No dots, no bit ⇒ the parent self-reports Empty for its own realm.
    let mut rig = parent_with_two_planet_children(sibling);
    assert!(
        demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "a truly empty parent self-reports Empty"
    );
    // A fresh bit ⇒ NOT Empty (SL7: a live child keeps its parent alive); the sibling is demanded.
    let mut rig = parent_with_two_planet_children(sibling);
    inject_bit(&mut rig, RealmId::Planet(42));
    let verbs: BTreeMap<RealmId, DemandVerb> = demands(&rig.tick(vec![]))
        .iter()
        .map(|d| (d.child.lowered(), d.verb))
        .collect();
    assert!(
        !verbs.values().any(|v| *v == DemandVerb::Empty),
        "a live child bit makes the parent non-empty"
    );
    assert_eq!(verbs.get(&RealmId::Planet(43)), Some(&DemandVerb::SpinUp));
    // An EXPIRED bit stops counting: past the TTL the parent is empty again (and the bit is gone).
    let now = rig.world.resource::<ClockSample>().local_tick.0 + retain_ttl_ticks(&config()) + 5;
    rig.set_local_tick(now);
    assert!(
        demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "an expired bit no longer holds the parent open"
    );
    assert!(
        rig.world.resource::<ChildLiveness>().0.is_empty(),
        "the expired bit was pruned"
    );
}

#[test]
fn a_child_observer_never_lands_in_an_occupants_verdict_but_a_dot_does() {
    // The `Occupants` verdict is folded ONLY over DOT observers — an occupied CHILD runs the
    // same band math (it warms its siblings, SL7) but the opener holds no client for it, so it
    // contributes nothing here. Its own band rides a `Child`-scope window instead.
    let sibling = DVec3::new(10_000.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(sibling);
    let open = GatewayToShard::WindowOpen {
        window: WindowId(1),
        scope: WindowScope::Occupants,
        static_held: None,
    };
    // A live bit in range of itself ⇒ the Occupants verdict stays EMPTY (never a render route).
    inject_bit(&mut rig, RealmId::Planet(42));
    let opened = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &open)]);
    assert!(
        window_memberships(&opened)
            .iter()
            .all(|(_, _, added, _)| added.is_empty()),
        "an occupied-child observer never enters an Occupants verdict"
    );
    // But a real local DOT with Planet(42) in its band gets it added.
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::ZERO); // at the origin ⇒ in range
    let added: Vec<RealmId> = window_memberships(&rig.tick(vec![]))
        .into_iter()
        .flat_map(|(_, _, added, _)| added)
        .collect();
    assert!(
        added.contains(&RealmId::Planet(42)),
        "a dot with Planet(42) in its band gets it added"
    );
}

#[test]
fn a_child_bit_survives_a_lost_heartbeat_then_expires() {
    // Step 5 slice B — the anti-blink property, transferred from the pose lane to the bit: the
    // TTL bridges a lost `Unreliable` heartbeat, so one missed datagram never blinks a warmed
    // sibling; past the TTL the bit prunes and stops warming.
    let near = DVec3::new(500.0, 0.0, 0.0);
    let mut rig = parent_with_two_planet_children(near);
    inject_bit(&mut rig, RealmId::Planet(42));
    let sib_verb = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Option<DemandVerb> {
        demands(sent)
            .iter()
            .find(|d| d.child.lowered() == RealmId::Planet(43))
            .map(|d| d.verb)
    };
    // Tick 1 (local_tick 1): the sibling spins up off the occupied child.
    assert_eq!(sib_verb(&rig.tick(vec![])), Some(DemandVerb::SpinUp));
    // A LOST heartbeat: advance one tick WITHOUT refreshing the bit. It is still alive (age 1 ≤
    // TTL) ⇒ the sibling STAYS demanded (KeepAlive) — no blink.
    rig.set_local_tick(2);
    assert_eq!(
        sib_verb(&rig.tick(vec![])),
        Some(DemandVerb::KeepAlive),
        "one lost heartbeat never blinks the warmed sibling"
    );
    // After the TTL lapses ⇒ the bit is pruned ⇒ it no longer warms the sibling.
    let ttl = retain_ttl_ticks(&config());
    rig.set_local_tick(2 + ttl + 5);
    assert_eq!(
        sib_verb(&rig.tick(vec![])),
        None,
        "after the TTL the pruned bit stops warming the sibling"
    );
    assert!(
        rig.world.resource::<ChildLiveness>().0.is_empty(),
        "the bit is pruned once its TTL lapses"
    );
}

#[test]
fn evaluate_realm_aoi_two_observers_latch_independently_and_union() {
    // VU S0: each observer carries its OWN acquire/grace latch; the child-level demand is their UNION.
    // A hands the child off to B (A leaves as B arrives) WITHOUT a thrash — the child stays KeepAlive
    // across the swap (never a re-SpinUp, never a drop), and both latches are tracked independently.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 2),
        ],
    );
    let a = SessionId(1);
    let b = SessionId(2);
    insert_owned_dot(&mut rig, a, player(1), DVec3::new(500.0, 0.0, 0.0)); // A in range (< spin-up)
    insert_owned_dot(&mut rig, b, player(2), DVec3::new(3000.0, 0.0, 0.0)); // B out (past tear-down)
    // Tick 1: A acquires ⇒ exactly ONE SpinUp (the union); ONLY A holds a latch (B is out of range).
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::SpinUp]
    );
    assert_eq!(
        rig.world.resource::<AoiMembership>().0.len(),
        1,
        "only the in-range observer holds a latch"
    );
    // A leaves (into grace) as B arrives: the child stays demanded via the union — ONE KeepAlive, no
    // re-SpinUp, no drop — and NOW both observers hold a latch (A grace-holding, B acquired).
    move_dot(&mut rig, a, DVec3::new(3000.0, 0.0, 0.0));
    move_dot(&mut rig, b, DVec3::new(500.0, 0.0, 0.0));
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::KeepAlive]
    );
    assert_eq!(
        rig.world.resource::<AoiMembership>().0.len(),
        2,
        "both observers hold independent latches (A in grace, B acquired)"
    );
}

#[test]
fn evaluate_realm_aoi_per_observer_hysteresis_is_stricter_than_the_old_global_min() {
    // VU S0 correctness: per-observer latches are STRICTER (more correct) than the old global-min.
    // Observer A acquires then leaves entirely; observer B loiters in the HYSTERESIS band (past spin-up,
    // inside tear-down) but NEVER acquired. Per-observer: once A is gone the child STOPS being demanded
    // (B's release-band distance can't hold a latch it never acquired). The old global-min WOULD have
    // kept it alive (B is within tear-down of the SHARED latch A set) — the flaw this per-observer fixes.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 100.0, 0), // grace 0 ⇒ A drops the instant it is out
        ],
    );
    let a = SessionId(1);
    let b = SessionId(2);
    insert_owned_dot(&mut rig, a, player(1), DVec3::new(500.0, 0.0, 0.0)); // A acquires (< spin-up 1000)
    insert_owned_dot(&mut rig, b, player(2), DVec3::new(1500.0, 0.0, 0.0)); // B loiters in the band
    // Tick 1: A acquires, B (past spin-up, never acquired) does not ⇒ exactly one SpinUp.
    assert_eq!(
        demands(&rig.tick(vec![]))
            .iter()
            .map(|d| d.verb)
            .collect::<Vec<_>>(),
        vec![DemandVerb::SpinUp]
    );
    // A leaves entirely (past tear-down 2000); B stays loitering in the hysteresis band.
    move_dot(&mut rig, a, DVec3::new(3000.0, 0.0, 0.0));
    assert!(
        demands(&rig.tick(vec![])).is_empty(),
        "B never acquired ⇒ once A is gone the child stops being demanded (global-min would wrongly hold)"
    );
}

#[test]
fn assert_realm_aoi_feature_anywhere() {
    // HR4 G-IDENTICAL: the IDENTICAL AoI feature (an occupant reaching a child ⇒ exactly ONE SpinUp
    // keyed on the child's coord) fires byte-identically on a SYSTEM shard (child = Planet) AND a
    // PLANET shard (child = Area). The loop is kind-BLIND — `own_coord.child(level_of(child))`, no
    // match-on-realm-kind — so the two runs differ ONLY in the child realm named.
    let a = drive_aoi_spinup(
        OWN_REALM,
        FrameRef::SystemSpace { system_seed: 7 },
        aoi_child_framed(
            OTHER_REALM,
            Some(OWN_REALM),
            FrameRef::PlanetCentered { planet_seed: 42 },
            3,
        ),
    );
    let b = drive_aoi_spinup(
        RealmId::Planet(42),
        FrameRef::PlanetCentered { planet_seed: 42 },
        aoi_child_framed(
            RealmId::Area(99),
            Some(RealmId::Planet(42)),
            FrameRef::AreaLocal {
                planet_seed: 42,
                area_seed: 99,
            },
            3,
        ),
    );
    // Count the SpinUp feature under test specifically: on the Planet run the (500,0,0) dot ALSO sits
    // in a child it re-homes into, so `redrive_stranded_crossings` now (correctly) emits a KeepAlive to
    // hold that crossing DEST alive — a SEPARATE, universally-correct behaviour (every crossing keeps
    // its dest alive). The keep-alive fires ONLY for a real held latch, so filtering to SpinUp isolates
    // the AoI feature this G-IDENTICAL test asserts.
    let a_spinup: Vec<_> = a.iter().filter(|d| d.verb == DemandVerb::SpinUp).collect();
    let b_spinup: Vec<_> = b.iter().filter(|d| d.verb == DemandVerb::SpinUp).collect();
    assert_eq!(
        a_spinup.len(),
        1,
        "the System shard emits exactly one SpinUp"
    );
    assert_eq!(
        b_spinup.len(),
        1,
        "the Planet shard emits exactly one SpinUp"
    );
    // IDENTICAL structure — same verb, same emitter fence, same tick; only the child KIND differs.
    assert_eq!(a_spinup[0].parent_fence, b_spinup[0].parent_fence);
    assert_eq!(a_spinup[0].universe_tick, b_spinup[0].universe_tick);
    // Each names its OWN child through the coord machinery (System→Planet, Planet→Area).
    assert_eq!(a_spinup[0].child, child_coord_of(OWN_REALM, OTHER_REALM));
    assert_eq!(
        b_spinup[0].child,
        child_coord_of(RealmId::Planet(42), RealmId::Area(99))
    );
}

#[test]
fn aoi_emits_the_parent_headread_when_armed_and_on_cadence() {
    // A PLANET shard (own realm Planet(42)) nested under System(7): its `own_coord` carries the full
    // lineage, so `parent()` is the System. With an ARMED child band + the recheck cadence live, the AoI
    // pass resolves the parent by HeadRead-ing its directory record — its node lands in `ParentRealmNode`,
    // the send target of the UP-lanes (the ChildLive bit + the up-observation rows/outlines; the
    // per-occupant interest up-relay is DELETED, Step 5 slice D).
    let cfg = StubConfig {
        realm: OTHER_REALM,
        held_realms: StubConfig::single_realm(OTHER_REALM),
        frame: frame_of(OTHER_REALM),
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        realm_recheck_interval: 2,
        ..config()
    };
    let mut rig = Rig::with_config(cfg);
    grant_realm_for(&mut rig, OTHER_REALM);
    plant_aoi(
        &mut rig,
        vec![
            region(ROOT_REALM, None, DVec3::ZERO, 1.0e9),
            region_framed(
                OTHER_REALM,
                Some(ROOT_REALM),
                DVec3::ZERO,
                100_000.0,
                frame_of(OTHER_REALM),
            ),
            aoi_child_framed(
                RealmId::Area(99),
                Some(OTHER_REALM),
                FrameRef::AreaLocal {
                    planet_seed: 42,
                    area_seed: 99,
                },
                0,
            ),
        ],
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4); // 4 % 2 == 0 ⇒ due
    let sent = rig.tick(vec![]);
    let parent_key = DirectoryKey::Realm(StubConfig::root_coord(OWN_REALM).lowered());
    assert!(
        headreads(&sent).contains(&parent_key),
        "the AoI pass HeadReads the PARENT realm on the recheck cadence"
    );
}

#[test]
fn aoi_recheck_cadence_uses_the_armed_recheck_else_a_tick_derived_demand_cadence() {
    // ARMED (self-fence recheck > 0): that IS the AoI cadence (one HeadRead round-trip serves both, HR3).
    let armed = StubConfig {
        realm_recheck_interval: 4,
        tick_dt_s: 0.05,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&armed), 4);
    // DISARMED (recheck 0 — the DevTest profile turns the self-fence off): fall back to a tick-DERIVED
    // demand cadence = hz/2 = (1/0.05)/2 = 10 — the up-relay/cascade must still run in demand mode.
    let disarmed = StubConfig {
        realm_recheck_interval: 0,
        tick_dt_s: 0.05,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&disarmed), 10);
    // DISARMED at a very slow tick (hz < 2 ⇒ hz/2 rounds to 0) clamps to 1, never a 0-divisor in
    // `due_this_tick`.
    let slow = StubConfig {
        realm_recheck_interval: 0,
        tick_dt_s: 1.0,
        ..config()
    };
    assert_eq!(aoi_recheck_cadence(&slow), 1);
}

#[test]
fn parent_headread_due_resolves_only_when_parented_armed_and_on_cadence() {
    let clock = ClockSample {
        local_tick: vd_core::TickId(4),
        universe_tick: UniverseTick(100),
        epoch: vd_core::EpochId(1),
        synced: true,
    };
    let armed = RealmRegions::new(vec![aoi_child(RealmId::Planet(99), ROOT_REALM, 1000.0, 0)]);
    let inert = RealmRegions::new(vec![region(
        RealmId::Planet(99),
        Some(ROOT_REALM),
        DVec3::ZERO,
        1000.0,
    )]);
    let parented = || StubConfig {
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        realm_recheck_interval: 2,
        ..config()
    };
    // parented + armed + on cadence (4 % 2 == 0) ⇒ resolve the parent.
    assert_eq!(
        parent_headread_due(&parented(), &armed, &clock),
        Some(StubConfig::root_coord(OWN_REALM)),
    );
    // parented + INERT bands ⇒ `aoi_live()` false ⇒ nothing (the walk/static byte-identity guard).
    assert_eq!(parent_headread_due(&parented(), &inert, &clock), None);
    // parented + armed but OFF cadence (4 % 3 != 0) ⇒ nothing.
    let off = StubConfig {
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        realm_recheck_interval: 3,
        ..config()
    };
    assert_eq!(parent_headread_due(&off, &armed, &clock), None);
    // a ROOT shard (its `own_coord` has no parent) ⇒ nothing, ever.
    let root = StubConfig {
        own_coord: StubConfig::root_coord(OWN_REALM),
        realm_recheck_interval: 2,
        ..config()
    };
    assert_eq!(parent_headread_due(&root, &armed, &clock), None);
}

#[test]
fn update_parent_node_caches_the_parent_shard_overwrites_revokes_and_ignores_non_parents() {
    let cfg = StubConfig {
        own_coord: child_coord_of(OWN_REALM, OTHER_REALM),
        ..config()
    };
    let parent = StubConfig::root_coord(OWN_REALM).lowered();
    let rec = |auth: AuthorityRef| vd_wire::seams::directory::OwnerRecord {
        authority: auth,
        fence: Fence(1),
        lease_expires: UniverseTick(1_000),
        in_transfer: None,
    };
    let mut pn = ParentRealmNode::default();
    // A parent Head with a Shard authority ⇒ the node is cached.
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Shard(NodeId(77)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(pn.0, Some(NodeId(77)));
    // Parent RE-HOME (a later reply naming a new node) ⇒ overwrite.
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Shard(NodeId(88)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(pn.0, Some(NodeId(88)));
    // A reply for a NON-parent realm — here the shard's OWN realm (Planet(42) ≠ the System(7) parent),
    // i.e. its own-realm recheck reply — ⇒ the cache is untouched (the guard's false arm).
    update_parent_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(99)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(
        pn.0,
        Some(NodeId(88)),
        "a non-parent Head never touches the parent cache"
    );
    // A parent record held by a GATEWAY (not a shard) ⇒ cleared (no shard to relay to).
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Gateway(NodeId(5)))),
        &cfg,
        &mut pn,
    );
    assert_eq!(pn.0, None);
    // A parent REVOKE (record gone) ⇒ cleared.
    update_parent_node(
        parent,
        Some(&rec(AuthorityRef::Shard(NodeId(88)))),
        &cfg,
        &mut pn,
    );
    update_parent_node(parent, None, &cfg, &mut pn);
    assert_eq!(pn.0, None);
}

#[test]
fn update_child_node_caches_a_rostered_childs_shard_revokes_and_ignores_non_children() {
    // The downward twin (findings 0/43): a realm-Head reply for a ROSTERED direct child caches its
    // node as the up-lanes' admission answer; a re-home overwrites; a non-Shard or absent record
    // REMOVES (fail closed); a reply for anything not on the roster is a no-op.
    let cfg = config(); // own realm System(7)
    let regions = RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    let rec = |auth: AuthorityRef| vd_wire::seams::directory::OwnerRecord {
        authority: auth,
        fence: Fence(1),
        lease_expires: UniverseTick(1_000),
        in_transfer: None,
    };
    let mut cn = ChildRealmNodes::default();
    // A rostered child's Head with a Shard authority ⇒ cached.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(61)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert_eq!(cn.0.get(&OTHER_REALM), Some(&NodeId(61)));
    // A child RE-HOME (a later reply naming a new node) ⇒ overwrite.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(62)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert_eq!(cn.0.get(&OTHER_REALM), Some(&NodeId(62)));
    // A NON-child realm (here the shard's own) ⇒ the map is untouched (the roster guard's false arm).
    update_child_node(
        OWN_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(63)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert_eq!(cn.0.len(), 1, "a non-child Head never touches the map");
    // A child record held by a GATEWAY (not a shard) ⇒ REMOVED — the up-lanes fail closed.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Gateway(NodeId(5)))),
        &cfg,
        &regions,
        &mut cn,
    );
    assert!(cn.0.is_empty());
    // A child REVOKE (record gone) ⇒ removed too.
    update_child_node(
        OTHER_REALM,
        Some(&rec(AuthorityRef::Shard(NodeId(62)))),
        &cfg,
        &regions,
        &mut cn,
    );
    update_child_node(OTHER_REALM, None, &cfg, &regions, &mut cn);
    assert!(cn.0.is_empty());
}

#[test]
fn admission_head_reads_ride_the_aoi_cadence_for_demanded_children() {
    // Findings 0/43, the EAGER pre-resolve: on the same cadence (and behind the same `aoi_live`
    // gate) as the parent head-read, the shard reads the directory head of every child it is
    // currently demanding or holds a live bit for — so the admission answer is in hand before the
    // child's first bit ever arrives (zero added spin-up-to-visible latency). Off-cadence ticks
    // read nothing.
    let mut rig = Rig::with_config(StubConfig {
        realm_recheck_interval: 2,
        ..config()
    });
    rig.grant_realm();
    plant_aoi(
        &mut rig,
        vec![
            root_region(),
            own_region(),
            aoi_child(OTHER_REALM, OWN_REALM, 1000.0, 0),
        ],
    );
    // An occupant inside the child's band ⇒ the child is demanded every tick.
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4); // on the cadence
    assert!(
        headreads(&rig.tick(vec![])).contains(&DirectoryKey::Realm(OTHER_REALM)),
        "a demanded child's head is read on the cadence beat"
    );
    rig.set_local_tick(5); // off the cadence
    assert!(
        !headreads(&rig.tick(vec![])).contains(&DirectoryKey::Realm(OTHER_REALM)),
        "off-cadence ticks read nothing — the count is bounded by beats, not ticks"
    );
}

#[test]
#[should_panic(expected = "self-fence")]
fn register_stub_shard_rejects_a_parent_that_aliases_its_own_realm_id() {
    use vd_core::realm_path::RealmKindTag;
    // A realm whose PARENT lowers to its own id is unrepresentable, and the boot guard refuses it —
    // otherwise the shard's own parent-Head reply drives the PRIMARY-FOREIGN self-fence branch.
    //
    // ★ RE-BASED IN S9, AND THE WAY IN CHANGED. This used to build `Galaxy(5) → System(1)`, because
    // a galaxy LOWERED to `System(1)` whatever its seed: a perfectly ordinary system, seed 1, hosted
    // under a perfectly ordinary galaxy, collided with its own parent. That was the realistic way to
    // hit this, and S9 closed it — a galaxy keeps its seed now, so `Galaxy(5)` lowers to `Galaxy(5)`
    // and nothing collides.
    //
    // The guard is NOT dead, so it is still driven: with lossless lowering a self-alias needs the
    // same KIND and the same SEED on both levels, which is a nonsense lineage rather than an
    // accident of naming. That is the improvement — the failure went from something a real world
    // could produce to something only a malformed path can.
    let own_coord = RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 1),
        RealmLevel::new(RealmKindTag::System, 1),
    ]))
    .expect("a two-level path has a leaf");
    let bad = StubConfig {
        realm: RealmId::System(1),
        held_realms: StubConfig::single_realm(RealmId::System(1)),
        frame: frame_of(RealmId::System(1)),
        own_coord,
        ..config()
    };
    let _ = Rig::with_config(bad);
}

#[test]
fn a_demoted_dot_goes_silent_to_the_parent_on_the_very_tick_it_is_handed_over() {
    // THE SILENCE the hand-off ledger exists to end — MEASURED, not argued. What crosses upward
    // now is the ONE occupancy bit (slice D deleted the pose relay), and the property is the
    // same: a source shard that has applied the ordered Demote still holds the dot (as a
    // retained Ghost), but the observer fold filters on `speaks_for` — so at a ZERO budget the
    // realm counts nobody, the bit stops on the demote tick, and the parent loses the only
    // liveness that was standing in for the traveller mid-crossing. This is the BEFORE reading;
    // the armed budget below keeps the bit beating through the window.
    const PARENT_NODE: NodeId = NodeId(55);
    let mut rig = parented_aoi_rig(2);
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "while OWNED the realm's bit ships (the occupancy edge here; cadence beats follow)"
    );
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(entity),
            new_owner_fence: Fence(2),
            step_id: DEMOTE_STEP,
        }),
    )]);
    assert_eq!(
        child_live_bits(&sent).len(),
        0,
        "the demote tick is the LAST beat the parent hears — at a zero budget the realm goes silent"
    );
}

#[test]
fn an_armed_shard_keeps_telling_its_parent_where_a_departing_occupant_is() {
    // THE ARMED READING of the silence measured above. Same fixture, same demote, a budget of 8 —
    // and the parent keeps hearing about the traveller for the whole window instead of losing them
    // at the worst possible moment.
    const PARENT_NODE: NodeId = NodeId(55);
    // recheck 1 ⇒ the bit's cadence is every tick (finding 41): this test measures the HOLD
    // budget tick by tick, so it pins the densest beat; the cadence itself has its own gate
    // (`the_bit_beats_on_the_cadence_plus_the_occupancy_edge`).
    let mut rig = parented_aoi_rig_holding(1, 8);
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let owned = child_live_bits(&rig.tick(vec![]));
    assert_eq!(owned.len(), 1, "while OWNED, the bit beats");
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;

    let handed = child_live_bits(&rig.tick(vec![demote_msg(entity, Fence(2))]));
    assert_eq!(
        handed.len(),
        1,
        "the realm keeps counting a subject it has handed away — the bit beats on"
    );
    assert_eq!(handed[0].0, PARENT_NODE);
    // …and it is doing so WITHOUT owning the dot. The hold is what carries the liveness, not
    // ownership.
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates()
    );

    // Past the budget the shard falls silent on its own, so a hand-off that WEDGES rather than
    // completes cannot hold the parent's attention — or, below, its own realm — open forever.
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 8);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "the budget is a real cap, not a formality"
    );
}

#[test]
fn the_take_over_landing_stops_the_relay_before_the_budget_does() {
    // THE NORMAL TERMINAL. The destination's `GhostFlow::Spawn` is its proof that it has taken over —
    // and from that moment IT relays the occupant, so this shard must stop. The budget is only the
    // backstop for a hand-off that never gets here.
    const PARENT_NODE: NodeId = NodeId(55);
    // recheck 1 ⇒ the bit beats every tick (see the hold-budget test above for why).
    let mut rig = parented_aoi_rig_holding(1, 8);
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let _ = rig.tick(vec![demote_msg(entity, Fence(2))]);

    // A REPLAYED take-over from a superseded crossing (an older fence) proves nothing and leaves the
    // hold — and therefore the relay — standing.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(1),
    })]);
    assert_eq!(
        child_live_bits(&sent).len(),
        1,
        "a stale take-over proof does not end this shard's hand-off — the bit beats on"
    );

    // The matching proof does, well inside the budget.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(
        child_live_bits(&sent).is_empty(),
        "once the destination has taken over, the source stops — no lingering double beat"
    );
}

#[test]
fn a_realm_does_not_call_itself_empty_while_somebody_is_still_leaving_it() {
    // THE SECOND CONSUMER, and the behaviour-changing half: emptiness is a claim about occupancy, and
    // a subject mid-hand-off has not finished leaving. Without this a realm can report itself empty in
    // the very window its own hand-off is running — the source-side twin of the arrival race.
    let mut rig = parented_aoi_rig_holding(2, 8);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    assert!(
        !demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "an occupied realm never reports Empty"
    );

    assert!(
        !demands(&rig.tick(vec![demote_msg(entity, Fence(2))]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "nor does it the moment it hands that occupant away"
    );

    // And when the hand-off is over — here by running out of budget, the wedged case — the realm goes
    // back to telling the truth, so an abandoned crossing cannot keep a realm alive indefinitely.
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 8);
    assert!(
        demands(&rig.tick(vec![]))
            .iter()
            .any(|d| d.verb == DemandVerb::Empty),
        "past the budget the realm reports the truth again"
    );
}

#[test]
fn aoi_up_relays_nothing_from_a_root_shard() {
    // A ROOT shard (galaxy free-fly: `own_coord` has no parent) is the TOP of the chain — it relays
    // interest to no one. Armed + occupied, but `own_coord.parent()` is None ⇒ the relay never fires.
    let mut rig = Rig::with_config(StubConfig {
        realm: OWN_REALM,
        held_realms: StubConfig::single_realm(OWN_REALM),
        frame: frame_of(OWN_REALM),
        own_coord: StubConfig::root_coord(OWN_REALM),
        realm_recheck_interval: 2,
        ..config()
    });
    grant_realm_for(&mut rig, OWN_REALM);
    plant_aoi(
        &mut rig,
        vec![
            region(ROOT_REALM, None, DVec3::ZERO, 1.0e9),
            aoi_child(OTHER_REALM, ROOT_REALM, 1000.0, 0),
        ],
    );
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "a root shard has nobody above it to report liveness to"
    );
}

#[test]
fn aoi_resolves_the_parent_but_up_relays_nothing_until_the_node_is_known() {
    // Parented + armed + on cadence, but the parent Head has NOT come back yet: the resolve HeadRead
    // fires (so the node WILL arrive), but NO ChildLive bit until it does (the inner-unresolved arm).
    let mut rig = parented_aoi_rig(2);
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(4);
    let sent = rig.tick(vec![]);
    let parent_key = DirectoryKey::Realm(StubConfig::root_coord(OWN_REALM).lowered());
    assert!(
        headreads(&sent).contains(&parent_key),
        "the parent HeadRead fires (resolving the node)"
    );
    assert!(
        child_live_bits(&sent).is_empty(),
        "no bit until the parent node is resolved — the next tick catches up"
    );
}

#[test]
fn the_bit_beats_on_the_cadence_plus_the_occupancy_edge() {
    // Lane cure, finding 41 — the bit's contract-stated rate, measured: on the AoI cadence, plus
    // immediately when the realm becomes occupied (the adopt edge, derived from the occupancy
    // transition itself — no hook in any adopt path), and re-armed by going empty.
    const PARENT_NODE: NodeId = NodeId(55);
    let mut rig = parented_aoi_rig(4); // cadence 4 — beats land on multiples of 4
    resolve_parent_head(
        &mut rig,
        StubConfig::root_coord(OWN_REALM).lowered(),
        PARENT_NODE,
    );
    // OCCUPANCY EDGE: the first occupied tick ships the bit at once, OFF the cadence.
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(5);
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "the occupancy transition ships immediately, off-cadence"
    );
    // Sustained occupancy off the cadence: silent — the parent has been told, the TTL holds it.
    rig.set_local_tick(6);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "off-cadence, already told ⇒ no beat"
    );
    // The cadence beat re-asserts the level.
    rig.set_local_tick(8);
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "the cadence beat ships"
    );
    // GOING EMPTY re-arms the edge: the next occupant's first tick ships at once again.
    rig.world.resource_mut::<Dots>().0.clear();
    rig.set_local_tick(9);
    assert!(
        child_live_bits(&rig.tick(vec![])).is_empty(),
        "an empty realm ships no bit (it self-reports Empty instead)"
    );
    insert_owned_dot(&mut rig, SESSION, player(8), DVec3::new(500.0, 0.0, 0.0));
    rig.set_local_tick(10);
    assert_eq!(
        child_live_bits(&rig.tick(vec![])).len(),
        1,
        "re-occupied ⇒ the edge fires again, off-cadence"
    );
}

/// ★ DIAGNOSTIC (2026-08-30): HOW MANY INTEREST MESSAGES DOES THE GALAXY SEND, PER BEAT, DURING WARP?
///
/// This is the number the cascade-cap removal owed. The galaxy is the widest realm in the world —
/// 233 220 direct children — and warp is when a looker moves through it fastest, so this is the
/// worst case the mechanism will ever meet.
///
/// EVERY star system is given a resolved route, which is the WORST case by construction: in a live
/// cluster a route resolves only for a child the shard is already demanding, so the real rate is at
/// most this one.
#[test]
#[ignore]
fn diag_the_galaxys_interest_message_rate_during_warp() {
    let cfg_world = vd_physics::worldgen::UniverseConfig::world(
        vd_physics::worldgen::VISUAL_OCCUPANT_V_MAX_MPS,
        vd_physics::worldgen::AOI_TICK_DT_S,
    );
    let galaxy = vd_core::worldgen::GALAXY;
    let held = std::collections::BTreeSet::from([galaxy]);
    let (regions, _) = vd_physics::worldgen::shard_boot_world(
        0,
        &cfg_world,
        &held,
        galaxy,
        &std::collections::BTreeSet::new(),
    );
    let systems: Vec<RealmId> = regions
        .iter()
        .filter(|r| r.parent == Some(galaxy))
        .map(|r| r.realm)
        .collect();
    println!("galaxy holds {} direct children", systems.len());

    let mut stub = config();
    stub.realm = galaxy;
    stub.held_realms = StubConfig::single_realm(galaxy);
    stub.frame = regions
        .iter()
        .find(|r| r.realm == galaxy)
        .expect("the galaxy is in its own boot")
        .frame;
    stub.own_coord = StubConfig::root_coord(galaxy);
    let stub_frame_tier = stub.frame.tier();
    let mut rig = Rig::with_config(stub);
    grant_realm_for(&mut rig, galaxy);
    plant_aoi(&mut rig, regions.clone());
    {
        let mut nodes = rig.world.resource_mut::<ChildRealmNodes>();
        for (i, s) in systems.iter().enumerate() {
            nodes.0.insert(*s, NodeId(1000 + (i as u64 % 64)));
        }
    }
    // A looker flying through the disc. The step is a warp-scale hop per beat.
    // The galaxy's own step, not the finest one: a warp-scale hop counted in fine units overflows an
    // i64 long before it crosses a galaxy.
    let tier = stub_frame_tier;
    let step_m = 3.0e15;
    insert_owned_dot(&mut rig, SESSION, player(7), DVec3::ZERO);
    let mut peak_interests = 0usize;
    let mut peak_demands = 0usize;
    let mut total_interests = 0usize;
    let beats: u32 = 6;
    for beat in 0..beats {
        rig.world
            .resource_mut::<Dots>()
            .0
            .get_mut(&SESSION)
            .expect("the dot is planted")
            .pose
            .pos = LatticePos::from_metres(DVec3::new(step_m * f64::from(beat), 0.0, 0.0), tier);
        rig.set_local_tick(10 * u64::from(beat + 1));
        let sent = rig.tick(vec![]);
        let ints = interests(&sent).len();
        let dems = demands(&sent).len();
        total_interests += ints;
        peak_interests = peak_interests.max(ints);
        peak_demands = peak_demands.max(dems);
        println!("  beat {beat}: interests={ints} demands={dems}");
    }
    println!("PEAK interests per beat: {peak_interests}");
    println!("PEAK demands   per beat: {peak_demands}");
    println!("TOTAL interests over {beats} beats: {total_interests}");
    println!(
        "as a share of the galaxy's children: {:.4}%",
        100.0 * peak_interests as f64 / systems.len() as f64
    );
}
