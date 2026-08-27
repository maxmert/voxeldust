//! The crossing lane itself: Slice 1d.1 pose-only entity-state crossing (banner 5457) — source flush, dest adopt/promote, stale fence/epoch refusals, unplaceable-ingress refusal, retained-ghost emit, the D-4(a) removal fan-out — plus the dead cross-realm ENTITY lane (banner 6512, the SL7 occupancy bit and sealed WindowRelay landing, every tombstoned lane falling through to undecodable); then which realm owns a pose frame, the durable/transient CrossingRequest fan-out with its in-flight latch, and the hand-off hold ledger (open/close/supersede/expire/prune); then the star-system arrival relabel; then the one-hop law (a grandchild's claim moves only into the direct child, a cyclic roster yields no path). Locals that travel: acked (5665), with_child_coord (6523), hold_key (8904).
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

/// The three ingress refusals, driven end to end through the wire (never direct calls): a
/// CROSSING whose pose this shard cannot measure is not adopted (and not acked — an ack would
/// commit authority to a shard holding nothing); a RE-HOME with the same fault reconstructs
/// nothing; a FLUSH for an entity this shard does not hold ships nothing. Each is the loud
/// counted degrade the Stage-A log points instrument.
#[test]
fn unplaceable_ingresses_are_refused_counted_and_never_acked() {
    // (1) The crossing adopt refusal.
    let mut rig = Rig::new();
    rig.grant_realm();
    let foreign = StampedPose::at_rest(
        FrameRef::PlanetCentered { planet_seed: 999 },
        DVec3::new(1.0, 2.0, 3.0),
        UniverseTick(200),
    );
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), foreign)]);
    assert_eq!(rig.world.resource::<StubStats>().arrivals_unplaceable, 1);
    assert!(
        !rig.world
            .resource::<Dots>()
            .0
            .values()
            .any(|d| d.entity == SUBJECT),
        "the unplaceable crossing adopted nothing"
    );
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "an adopt that refused must not claim it landed: {sent:?}"
    );
    // (2) The re-home reconstruction refusal — the same one rule, at the second ingress.
    let mut rig = Rig::new();
    rig.grant_realm();
    let cmd = InterShardFlow::ReHome(ReHomeCmd {
        transfer: TransferId(7),
        universe_epoch: vd_core::EpochId(1),
        subject: DirectoryKey::Entity(SUBJECT),
        new_fence: Fence(2),
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(foreign),
        source: NodeId(99),
    });
    let _ = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &cmd)]);
    assert_eq!(rig.world.resource::<StubStats>().arrivals_unplaceable, 1);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
    // (3) The flush for an entity nobody here holds: nothing ships, loudly.
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![flush_msg(EntityId(0xDEAD))]);
    assert_eq!(
        to_orch(&sent),
        vec![],
        "no SourceFlushed (nothing at all) for an entity this shard does not hold"
    );
    // (4) A flush whose Stage-B1 re-validation refuses (the dot is still HELD by this realm's
    // own band) ships nothing either — the saga aborts pre-commit and this shard keeps
    // authority. The refusal itself is unit-covered on `flush_pose_for_dest`; this drives the
    // `on_flush_source` arm that consumes it.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    let _ = rig.attach();
    let held = rig.world.resource::<Dots>().0[&SESSION].entity;
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(7),
            subject: DirectoryKey::Entity(held),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::System(9),
            to_parent: None,
        }),
    )]);
    assert_eq!(
        to_orch(&sent),
        vec![],
        "a departure that is no longer true ships no SourceFlushed"
    );
    assert_eq!(rig.world.resource::<StubStats>().flush_stale_exit, 1);
    // (5) A crossing into a realm this shard neither is nor hosts has no frame to measure the
    // arrival in — refused counted (the `arrival_frame` None arm).
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(8),
            universe_epoch: vd_core::EpochId(1),
            schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
            fence: Fence(2),
            step_id: STUB_CROSSING_STEP,
            class: vd_core::entity_kind::DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity: SUBJECT,
                from_realm: FROM_REALM,
                to_realm: RealmId::System(9),
                pose: crossing_pose(),
                state: vec![],
            },
        }),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().arrivals_unplaceable, 1);
    assert_eq!(
        to_orch(&sent),
        vec![],
        "an arrival with no nameable frame is never acked"
    );
}

/// THE REMOVE MESSAGE at the band-exit teardown (D-4(a)): the Despawn that tears out the
/// retained ghost — the last emitter of a leaver here — tells every remaining dot's gateway to
/// evict the figure, ONE message per DISTINCT gateway (two bystanders behind one gateway share
/// one; a second gateway gets its own), each stamped with the realm fence and EXACTLY this
/// shard's universe tick (the resurrect guard's whole input — a wrong stamp mis-gates every
/// client refusal). A stale Despawn that removes nothing tells nobody, and a shard whose lease
/// lapsed (self-fenced) withholds the removal LOUDLY — counted, never silent.
#[test]
fn a_despawned_leavers_removal_is_told_once_per_bystander_gateway_and_stamped() {
    const BYSTANDER_A: SessionId = SessionId(0xBB);
    const BYSTANDER_B: SessionId = SessionId(0xBC);
    const BYSTANDER_FAR: SessionId = SessionId(0xBD);
    const OTHER_GATEWAY: NodeId = NodeId(21);
    let mut rig = Rig::new();
    let leaver = make_retained_ghost(&mut rig, Fence(2));
    // THREE bystanders: two behind the default GATEWAY (the dedup half), one behind a second
    // gateway (the multi-gateway half).
    insert_owned_dot(&mut rig, BYSTANDER_A, player(9), DVec3::new(1.0, 0.0, 0.0));
    insert_owned_dot(&mut rig, BYSTANDER_B, player(10), DVec3::new(2.0, 0.0, 0.0));
    insert_owned_dot(
        &mut rig,
        BYSTANDER_FAR,
        player(11),
        DVec3::new(3.0, 0.0, 0.0),
    );
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&BYSTANDER_FAR)
        .expect("just inserted")
        .gateway = OTHER_GATEWAY;
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    let removals = entity_removals(&sent);
    // The stamp is THE shard clock, exactly (the Rig clock reads universe_tick 100).
    assert_eq!(
        removals,
        vec![
            (GATEWAY, leaver, UniverseTick(100)),
            (OTHER_GATEWAY, leaver, UniverseTick(100)),
        ],
        "one removal per DISTINCT gateway, stamped with the shard's exact universe tick"
    );
    // A second Despawn removes nothing (already gone) ⇒ NO second removal fans out.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    assert!(
        entity_removals(&sent).is_empty(),
        "a no-op Despawn tells nobody"
    );
}

/// The despawn emit's NO-LEASE arm: a self-fenced shard (lease lapsed mid-teardown) withholds
/// the removal — an unowned shard is silent — but COUNTS the suppression, because each one is
/// a bystander who may keep a frozen figure until the sub machinery catches up.
#[test]
fn a_despawn_on_a_leaseless_shard_suppresses_the_removal_loudly() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    let leaver = make_retained_ghost(&mut rig, Fence(2));
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    rig.world.resource_mut::<RealmAuthority>().0 = None; // the self-fenced shape
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…but never silently: the suppression is counted"
    );
    // The SAME suppression at the take-over proof: a fresh rig, lease dropped before the
    // exact-fence SpawnV2 lands — the hold closes, the eviction is withheld + counted.
    let mut rig = Rig::new();
    let leaver = make_retained_ghost(&mut rig, Fence(2));
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    rig.world.resource_mut::<RealmAuthority>().0 = None;
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity: leaver,
        source_fence: Fence(2),
    })]);
    assert!(
        !rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(leaver, HoldRole::Source)),
        "the proof still closes the hold"
    );
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent at the proof too"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…and counted"
    );
}

/// THE REMOVE MESSAGE at detach completion (D-4(a)): the directory confirming a logout's
/// revoke is the permanent stop of that avatar here — the remaining bystander's gateway is
/// told to evict it, stamped with EXACTLY the shard's universe tick. The PROVISIONAL drop
/// tells nobody — driven here, not asserted in prose: an ungranted dot's detach removes it
/// with no removal fanned (it never emitted, so no client holds its figure). A lease lapse
/// at the completion instant withholds the removal loudly (counted).
#[test]
fn a_detached_dots_removal_is_told_to_the_bystanders_gateway() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    // Phase 1: detach → departing (held); no removal yet (the dot still emits).
    let detach = GatewayToShard::DetachSession {
        session: SESSION,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    assert!(
        entity_removals(&sent).is_empty(),
        "a departing dot still emits — nobody is told yet"
    );
    // Phase 2: the headless entity head confirms the revoke — despawn + THE removal, stamped
    // with the shard clock exactly (the Rig clock reads universe_tick 100).
    let gone = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: None,
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(gone),
    )]);
    assert_eq!(
        entity_removals(&sent),
        vec![(GATEWAY, entity, UniverseTick(100))],
        "the bystander's gateway is told exactly once, at the shard's exact tick"
    );
}

/// The detach path's two NEGATIVE arms, driven: (a) a PROVISIONAL (ungranted) dot's detach
/// drops it with NO removal — it never emitted, so no client holds a figure to evict;
/// (b) a detach completing on a LEASELESS shard withholds the removal loudly (counted).
#[test]
fn a_provisional_drop_and_a_leaseless_completion_fan_no_removal() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    // (a) the provisional drop: attach_request WITHOUT the grant confirm — an ungranted dot.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    let detach = GatewayToShard::DetachSession {
        session: SESSION,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "the provisional dot is dropped immediately (no directory round-trip)"
    );
    assert!(
        entity_removals(&sent).is_empty(),
        "…and NO removal fans — a pre-grant dot never emitted"
    );
    // (b) the completion on a leaseless shard: granted dot, detach, lease lapses, confirm.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    rig.world.resource_mut::<RealmAuthority>().0 = None;
    let gone = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: None,
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(gone),
    )]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…but the suppression is counted"
    );
}

#[test]
fn flush_source_ships_the_held_dots_pose() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach(); // a granted login dot for SESSION
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;

    // (a) BEFORE any input: the dot is at origin with NO applied seq → drained_seq defaults 0.
    let dot0 = rig.world.resource::<Dots>().0[&SESSION];
    let sent0 = rig.tick(vec![flush_msg(entity)]);
    assert_eq!(
        to_orch(&sent0),
        vec![InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: TransferId(7),
            step_id: FLUSH_SOURCE_STEP,
            pose: dot0.pose,
            drained_seq: 0,
        })],
        "ships the held dot's pose; an unset watermark defaults to 0"
    );

    // (b) AFTER applying seq 1: the pose moved and the watermark is Some(1).
    let _ = rig.tick(vec![input_for(SESSION, 1, GATEWAY)]);
    let dot1 = rig.world.resource::<Dots>().0[&SESSION];
    assert_ne!(fm(dot1.pose.pos), DVec3::ZERO, "the dot moved on input");
    let sent1 = rig.tick(vec![flush_msg(entity)]);
    assert_eq!(
        to_orch(&sent1),
        vec![InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: TransferId(7),
            step_id: FLUSH_SOURCE_STEP,
            pose: dot1.pose,
            drained_seq: 1,
        })],
        "ships the moved pose + the real drain watermark"
    );
}

#[test]
fn flush_source_for_an_unheld_or_non_entity_subject_ships_nothing() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    // An entity this shard does not hold → no ship (counted no-op).
    let sent = rig.tick(vec![flush_msg(EntityId(0xDEAD))]);
    assert!(to_orch(&sent).is_empty(), "no pose for an unheld entity");
    // A non-Entity subject → no ship.
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(7),
            subject: DirectoryKey::Realm(RealmId::System(9)),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::System(0),
            to_parent: None,
        }),
    )]);
    assert!(
        to_orch(&sent).is_empty(),
        "no pose for a non-Entity subject"
    );
}

#[test]
fn an_arriving_crossing_reseeds_the_swept_prior_at_the_arrival_point() {
    // THE STALE PRIOR AFTER A HAND-OFF. A Ghost is minted with its prior at the realm ORIGIN, because
    // a fresh dot has no history. Once membership tests the tick's whole motion segment, leaving it
    // there makes the first scan after arrival sweep a realm-wide line from the destination's centre
    // to wherever the subject actually landed — through every region on that line, none of which it
    // visited — and feed the result straight into a re-home decision. An arrival is a discontinuity,
    // not a movement.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let pose = crossing_pose();
    // The Ghost starts with the prior at the origin, and the arrival pose is somewhere else — so the
    // fixture can distinguish "reseeded" from "happened to already be right".
    let before = rig.world.resource::<Dots>().0[&SESSION].prev_offset;
    assert_eq!(before, LatticePos::ORIGIN);
    assert_ne!(
        pose.sanitized().pos.cell(),
        LatticePos::ORIGIN.cell(),
        "the arrival must be somewhere other than the origin, or this proves nothing"
    );

    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.prev_offset, dot.pose.pos,
        "the arriving dot's swept prior must BE the arrival point"
    );
}

#[test]
fn a_crossing_to_an_adopted_dot_stores_the_pose_stays_ghost_then_promote_flips_owned() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted, still a Ghost (not simulating)
    let pose = crossing_pose();

    // 1d.5b.3b: the crossing STORES the pose but leaves the dot a GHOST — the Ghost→Owned promote
    // RELOCATED to on_saga_promote (strict demote-before-promote). No autonomous flip here.
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(dot.pose, pose.sanitized(), "the crossed pose stored");
    assert!(
        !dot.authority.simulates(),
        "the dot STAYS Ghost after the crossing (the autonomous promote is gone)"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 1);
    assert!(acked(&sent, TransferId(7)), "the crossing step is acked");

    // The saga Promote flips it Ghost→Owned (pose-before-promote satisfied — the crossing landed).
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the Promote flips the dest Ghost→Owned at the recorded CAS fence"
    );

    // A crossing redelivery AFTER the flip still matches `crossing_target` (now Owned, still
    // granted+non-departing) → the journal returns `AlreadyApplied`: re-ack WITHOUT re-applying,
    // NOT re-buffered (no strand).
    let buffered_before = rig.world.resource::<StubStats>().crossings_buffered;
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_applied,
        1,
        "no re-apply on a post-flip redelivery"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_buffered,
        buffered_before,
        "a post-flip redelivery re-acks via the journal — it is NOT re-buffered (no strand)"
    );
    assert!(
        acked(&sent, TransferId(7)),
        "a post-flip redelivery still re-acks"
    );
}

#[test]
fn a_crossing_before_adopt_is_buffered_then_applied_on_the_grant_flip() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot, NOT granted
    let pose = crossing_pose();

    // The crossing arrives BEFORE the adopt flip → BUFFERED (no ack, dot unchanged).
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert!(
        !acked(&sent, TransferId(7)),
        "a buffered crossing is not acked yet"
    );
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        fm(dot.pose.pos),
        DVec3::ZERO,
        "buffered, not applied (still adopting)"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_buffered, 1);
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);

    // A redelivery WHILE STILL BUFFERED (the saga re-emits at-least-once) overwrites the same
    // key and must NOT inflate the buffered count (audit F-3 — the counter is per-crossing, not
    // per-redelivery).
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert!(
        !acked(&sent, TransferId(7)),
        "still buffered, still not acked"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_buffered,
        1,
        "a still-buffered redelivery does not inflate the buffered count"
    );

    // The adopt grant-flip DRAINS the buffer → applies the pose + acks, but the dot STAYS Ghost
    // (1d.5b.3b — the drained path shares `apply_crossing`, whose autonomous promote is gone).
    let sent = rig.tick(vec![adopted_head(Fence(2))]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.pose,
        pose.sanitized(),
        "the buffered crossing applied on the flip"
    );
    assert!(
        !dot.authority.simulates(),
        "the drained crossing leaves the dot a Ghost (the promote relocated to on_saga_promote)"
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 1);
    assert!(acked(&sent, TransferId(7)), "the drained crossing is acked");

    // The saga Promote then flips it Ghost→Owned (the buffered-drain path also satisfies
    // pose-before-promote — the crossing journaled on the drain).
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the Promote flips Owned on the buffered-drain path"
    );

    // A redelivery AFTER the drained crossing was journaled hits the IMMEDIATE path (now Owned,
    // still matched by `crossing_target`) and the journal dedups it across the boundary — re-ack
    // only, NO second apply, NO re-buffer.
    let buffered_before = rig.world.resource::<StubStats>().crossings_buffered;
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), pose)]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_applied,
        1,
        "a redelivery after a DRAINED crossing does not re-apply (journal spans the boundary)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_buffered,
        buffered_before,
        "a post-drain redelivery re-acks via the journal — it is NOT re-buffered",
    );
    assert!(
        acked(&sent, TransferId(7)),
        "the post-drain redelivery still re-acks"
    );
}

#[test]
fn a_stale_fence_crossing_is_dropped() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // adopted at Fence(2)
    // A crossing at Fence(1) is BELOW the dot's recorded authority fence → stale (fence rule 1).
    let sent = rig.tick(vec![crossing_msg(TransferId(7), Fence(1), crossing_pose())]);
    assert_eq!(rig.world.resource::<StubStats>().crossings_stale, 1);
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        fm(dot.pose.pos),
        DVec3::ZERO,
        "a stale crossing does not move the dot"
    );
    assert!(
        !acked(&sent, TransferId(7)),
        "a stale crossing is not acked"
    );
}

#[test]
fn a_non_crossing_transfer_payload_is_a_counted_noop() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    // An InitialSpawn payload is not a 1d.1 dest concern → counted no-op (no apply, no ack).
    let env = InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: TransferId(7),
        universe_epoch: vd_core::EpochId(1),
        schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
        fence: Fence(2),
        step_id: STUB_CROSSING_STEP,
        class: vd_core::entity_kind::DurabilityClass::Durable,
        payload: TransitionPayload::InitialSpawn {
            entity: SUBJECT,
            to_realm: TO_REALM,
            pose: crossing_pose(),
            state: vec![],
        },
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
    assert_eq!(rig.world.resource::<StubStats>().crossings_unhandled, 1);
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
    assert!(
        !acked(&sent, TransferId(7)),
        "an unhandled payload is not acked"
    );
}

#[test]
fn a_stale_epoch_crossing_is_refused() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    // A crossing minted under epoch 2 while this shard's clock epoch is 1 (the rig default) — a
    // delayed redelivery across a re-genesis, or a clock-desync bug. Refused at the ingress
    // BEFORE the payload match and BEFORE find-dot: counted, never applied, never buffered,
    // never acked (transfer_protocol §3.3 fail-safe — no entity placed at a stale celestial
    // position). This exercises the mismatch arm of the epoch gate; every other crossing test
    // exercises the match arm (rig clock epoch == envelope epoch == EpochId(1)).
    let env = InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: TransferId(7),
        universe_epoch: vd_core::EpochId(2),
        schema_version: vd_wire::intershard::TRANSFER_SCHEMA_VERSION,
        fence: Fence(2),
        step_id: STUB_CROSSING_STEP,
        class: vd_core::entity_kind::DurabilityClass::Durable,
        payload: TransitionPayload::StubCrossing {
            entity: SUBJECT,
            from_realm: FROM_REALM,
            to_realm: TO_REALM,
            pose: crossing_pose(),
            state: vec![],
        },
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
    assert_eq!(
        rig.world.resource::<StubStats>().crossings_epoch_mismatch,
        1
    );
    assert_eq!(rig.world.resource::<StubStats>().crossings_applied, 0);
    assert_eq!(rig.world.resource::<StubStats>().crossings_buffered, 0);
    assert_eq!(
        fm(rig.world.resource::<Dots>().0[&SESSION].pose.pos),
        DVec3::ZERO,
        "a stale-epoch crossing does not move the dot"
    );
    assert!(
        !acked(&sent, TransferId(7)),
        "a stale-epoch crossing is not acked"
    );
}

#[test]
fn a_stale_epoch_re_home_is_refused() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let session = SessionId(SUBJECT.0); // the deterministic clientless re-home session key
    // A re-home minted under epoch 2 while this shard's clock epoch is 1 (the rig default) — a delayed
    // redelivery across a re-genesis. Refused at the ingress BEFORE journal/adopt/ack: counted, no dot
    // created, no PromoteAck. The §3.3 fail-safe, UNIFORM with the crossing arm
    // (a_stale_epoch_crossing_is_refused). Exercises the mismatch arm of the re-home epoch gate; every
    // other re-home test exercises the match arm (rig clock epoch == cmd epoch == EpochId(1)).
    let env = InterShardFlow::ReHome(ReHomeCmd {
        transfer: TransferId(7),
        universe_epoch: vd_core::EpochId(2),
        subject: DirectoryKey::Entity(SUBJECT),
        new_fence: Fence(2),
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(crossing_pose()),
        source: NodeId(99),
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &env)]);
    assert_eq!(rig.world.resource::<StubStats>().re_home_epoch_mismatch, 1);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&session),
        "a stale-epoch re-home creates no dot"
    );
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "a stale-epoch re-home is not acked"
    );
}

#[test]
fn a_duplicate_entity_grant_head_is_an_idempotent_noop() {
    // GrantFlip::NoOp: a SECOND grant head for an already-granted dot flips nothing.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // first flip → granted
    let before = rig.world.resource::<Dots>().0[&SESSION];
    let sent = rig.tick(vec![adopted_head(Fence(2))]); // duplicate → NoOp
    let after = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(after, before, "a duplicate grant head changes nothing");
    assert!(!acked(&sent, TransferId(7)), "no ack on a duplicate grant");
}

#[test]
fn producer_less_reliable_flows_push_with_the_retained_marker() {
    // R-6d §7 CONFORMANCE: the three `FlowDurabilityClass::ProducerLessReliable` flows — the
    // source-shard `TransientBatch` emit (D-6 #1), the band-exit `Ghost::Despawn`, and slice F's
    // `Ghost::SpawnV2` take-over proof — have NO scan_deadlines re-driver, so their push MUST
    // carry `Durability::Retained` (the R-6d durable outbox mirrors + replays them across a
    // source crash). The `send`/`push_flow` default is Ephemeral, so THIS test is the guarantee
    // that these sites opted into durability; a future producer-less flow (compile-forced-
    // classified by `durability_class`, R-6d2a) whose author forgets the marker trips this.
    use vd_wire::intershard::FlowDurabilityClass;
    let producer_less = |sent: &[(NodeId, MsgClass, InterShardFlow, Durability)]| {
        sent.iter()
            .find(|(_, _, f, _)| f.durability_class() == FlowDurabilityClass::ProducerLessReliable)
            .map(|(_, _, _, dur)| *dur)
    };

    // --- (a) TransientBatch (the emit_transient_batch producer-less one-shot) ---
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        EntityId::pack(EntityKind::Debris, 1, 7, 1),
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch: TransferId(0xB3),
                to_parent: None,
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    assert_eq!(
        producer_less(&rig.tick_raw(vec![])),
        Some(Durability::Retained),
        "the TransientBatch emit MUST push Durability::Retained (no re-driver, D-6 #1)"
    );

    // --- (b) band-exit Ghost::Despawn (reuses the_dest_feed_despawns...'s setup) ---
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    // (c, folded into (b)'s rig) the PROMOTE tick itself pushes the SpawnV2 proof — read it
    // off the raw outbox before stepping further.
    let promote_sends = rig.tick_raw(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        producer_less(&promote_sends),
        Some(Durability::Retained),
        "the take-over proof Ghost::SpawnV2 MUST push Durability::Retained (no re-driver)"
    );
    rig.set_local_tick(6);
    let _ = rig.tick(vec![]); // in-band: the sweep streams nothing, keeps the registration
    let exit_pos = fm(crossing_pose().pos) + DVec3::new(3.0, 0.0, 0.0);
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION)
        .expect("the owned dot")
        .pose
        .pos = LatticePos::from_metres(exit_pos, vd_core::pose::Tier::Fine);
    assert_eq!(
        producer_less(&rig.tick_raw(vec![])),
        Some(Durability::Retained),
        "the band-exit Ghost::Despawn MUST push Durability::Retained (no re-driver)"
    );
}

/// Slice F's TTL BACKSTOP: a Source hold that EXPIRES (the take-over proof never came — a
/// logout mid-crossing kills the dest before it can send one) still evicts the bystanders'
/// figures: the retained ghost stops emitting the tick the hold dies, and the removal fans at
/// the shard's exact tick. The lane's loss-mode ends in a vanish, never a frozen phantom.
#[test]
fn an_expired_hold_evicts_the_bystanders_like_the_proof_that_never_came() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2)); // arms the budget (1_000 ticks)
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    let sent = rig.tick(vec![]);
    assert!(
        entity_rows_to_gateway(&sent).contains(&entity),
        "the fill emits while the hold lives"
    );
    // Jump past the budget: the prune drops the hold and the eviction fans in the same pass.
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 1_001);
    let sent = rig.tick(vec![]);
    assert_eq!(
        entity_removals(&sent),
        vec![(GATEWAY, entity, UniverseTick(100))],
        "the expiry fans the eviction — the TTL backstop of the take-over proof"
    );
    assert!(
        !entity_rows_to_gateway(&sent).contains(&entity),
        "…and the ghost stopped emitting the same tick"
    );

    // The LEASELESS expiry (the suppression arm at the prune): same setup, lease gone before
    // the budget runs out — the removal is withheld, counted, never silent.
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2));
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    rig.world.resource_mut::<RealmAuthority>().0 = None;
    let opened = rig.world.resource::<HandoffHolds>().0[&(entity, HoldRole::Source)].opened_at;
    rig.set_local_tick(opened.0 + 1_001);
    let sent = rig.tick(vec![]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an unowned shard is silent"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .entity_removals_suppressed_no_lease,
        1,
        "…but the suppression is counted"
    );

    // A DEST-role hold expiring fans nothing (the false arm): only a SOURCE hold guards a
    // retained ghost's emit.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world
        .resource_mut::<StubConfig>()
        .handoff_hold_ttl_ticks = 4;
    let stranger = player(7);
    open_hold(
        &mut rig.world.resource_mut::<HandoffHolds>(),
        (stranger, HoldRole::Dest),
        Fence(2),
        TickId(1),
        4,
    );
    rig.set_local_tick(20);
    let sent = rig.tick(vec![]);
    assert!(
        entity_removals(&sent).is_empty(),
        "an expired Dest hold is bookkeeping, never an eviction"
    );
}

#[test]
fn the_living_up_lanes_dispatch_and_every_dead_scenery_lane_counts_undecodable() {
    // The dispatch arms land in their stores through the REAL inbound path — the receive
    // helpers are covered directly elsewhere; this pins the wiring, and it pins the OTHER
    // half too: after the Slice-C2 deletion the ONLY things a realm still receives about the
    // world are the SL7 occupancy bit and a child's SEALED self-statements. The three old
    // scenery lanes are tombstoned, so a frame of any of them must fall through to
    // `undecodable` ON ITS OWN CARRIER — never a silent apply, never a panic.
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() =
        RealmRegions::new(vec![root_region(), own_region(), child_region()]);
    // The up-lanes admit only the directory-attested child node (findings 0/43) — seed the map
    // the realm-Head reply arm fills.
    rig.world
        .resource_mut::<ChildRealmNodes>()
        .0
        .insert(OTHER_REALM, NodeId(70));
    let child = with_child_coord(&mut rig, OTHER_REALM);
    // LIVING: the SL7 occupancy bit (SignalDelta) and the Q2 relay (reliable Saga).
    let bit = InterShardFlow::ChildLive(vd_wire::intershard::ChildLive {
        child: child.clone(),
        fence: Fence(1),
        at: UniverseTick(5),
    });
    let relay = InterShardFlow::WindowRelay(vd_wire::intershard::WindowRelay {
        child: child.clone(),
        realm_fence: Fence(1),
        own: vd_wire::session_flow::seal_relay_statements(&[]),
        interior: Vec::new(),
    });
    // ★TOMBSTONED (minors 17/19): the up-observation ship, the interim shape lane and the
    // down-cascade all rode SignalDelta; the down-reflect rode the reliable Saga carrier.
    let rows = InterShardFlow::RealmObservation(vd_wire::intershard::RealmObservation {
        child: child.clone(),
        realm_snapshot_bytes: Vec::new(),
    });
    let shapes =
        InterShardFlow::RealmShapeObservation(vd_wire::intershard::RealmShapeObservation {
            child: child.clone(),
            shapes: vec![render_shape(RealmId::Planet(52))],
        });
    let cascade = InterShardFlow::RealmCascade(vd_wire::intershard::RealmCascade {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        realm_snapshot_bytes: Vec::new(),
    });
    let scene_set = InterShardFlow::ChildSceneSet(vd_wire::intershard::ChildSceneSet {
        child: rig.world.resource::<StubConfig>().own_coord.clone(),
        realms: vec![render_shape(RealmId::Planet(53))],
    });
    let signal_msg = |flow: &InterShardFlow| Inbound::Wire {
        from: NodeId(70),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(postcard::to_allocvec(flow).expect("encode")),
    };
    let saga_msg = |flow: &InterShardFlow| Inbound::Wire {
        from: NodeId(70),
        class: MsgClass::Saga,
        bytes: crate::io::bytes(postcard::to_allocvec(flow).expect("encode")),
    };
    let _ = rig.tick(vec![
        signal_msg(&bit),
        signal_msg(&rows),
        signal_msg(&shapes),
        signal_msg(&cascade),
        saga_msg(&relay),
        saga_msg(&scene_set),
    ]);
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.child_live_received, 1, "the bit arm dispatched");
    assert_eq!(
        stats.window_relays_received, 1,
        "the relay arm dispatched on the Saga carrier"
    );
    assert_eq!(
        stats.undecodable, 4,
        "all FOUR tombstoned scenery frames counted undecodable on their own carriers, \
         and none of them reached a store"
    );
    assert_eq!(
        rig.world.resource::<RelayHeld>().0.len(),
        1,
        "and the sealed batch is held"
    );
}

/// A TOMBSTONED entity-lane frame (either leg) arriving on the SignalDelta carrier is counted
/// undecodable, never applied and never a panic — the discriminants are reserved forever, their
/// meaning is gone (Step 5 slice E). Measured per carrier: both legs rode SignalDelta, whose
/// closed fall-through is the counting arm (the slice D lesson — never assert a tombstone's
/// receiver behaviour without driving its actual dispatch).
#[test]
fn a_tombstoned_entity_lane_frame_is_counted_undecodable_on_both_legs() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let relay = vd_wire::intershard::EntityRelay {
        realm: StubConfig::root_coord(OWN_REALM),
        frame: config().frame,
        frame_id: 1,
        universe_tick: UniverseTick(1),
        entities: vec![],
    };
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::EntityInterest(relay.clone())).expect("encode"),
        ),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 1);
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(9),
        class: MsgClass::SignalDelta,
        bytes: crate::io::bytes(
            postcard::to_allocvec(&InterShardFlow::EntityCascade(relay)).expect("encode"),
        ),
    }]);
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 2);
}

#[test]
fn an_unnameable_pose_frame_safe_degrades_to_non_member_never_a_spurious_container() {
    // FA-1 safe-degrade: a subject whose pose frame the shard cannot NAME (not among its regions) has
    // `region_signed_distance` → `Err` → `f64::MAX` for EVERY region, so it is a member of NONE and its
    // deepest container folds to the ambient ROOT — never a spurious INNER container, never a panic. The
    // discriminator: at the origin under the OWN (registered) frame the deepest container is the child
    // `OTHER_REALM` (a re-home INWARD); under an un-nameable frame the child must NOT be entered.
    let mut rig = Rig::new(); // owns System 7 (config().realm == OWN_REALM)
    rig.grant_realm();
    plant_dock_regions(&mut rig); // root(1e9) ⊃ own(1e5) ⊃ child(1000), all shells at the origin
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 77);
    // A Station frame the dock forest NEVER planted — un-nameable to any dock region.
    insert_owned_dot_framed(
        &mut rig,
        TRIG_SESSION,
        entity,
        FrameRef::StationLocal { station_seed: 999 },
        DVec3::ZERO,
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..6 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert!(
        crossing_requests(&all)
            .iter()
            .all(|r| r.to_realm != OTHER_REALM),
        "an un-nameable pose frame safe-degrades to non-member ⇒ NEVER re-homes into the inner child",
    );
}

#[test]
fn the_crossing_decision_does_not_depend_on_whether_this_shard_remembers_the_subject() {
    // THE INVARIANT THAT FIXES THE BOUNDARY RE-FIRE (task #177), stated as a property rather than a
    // scenario. The stored membership bitset is per-shard and RAM-only: the source shard has one, an
    // arriving shard (or any shard after a restart) has a BLANK one. Before the derived prior those two
    // states produced DIFFERENT answers for the same subject at the same place — which is exactly why a
    // subject resting between the acquire and release edges ping-ponged between two shards forever.
    //
    // So: run the identical setup twice, differing ONLY in whether the shard already remembers the
    // subject as a member of its owning realm, and require the emitted crossings to be IDENTICAL.
    //
    // This is the RED control for the fix: with the stored-only prior the blank run acquires the realm
    // from scratch and the seeded run does not, so the two disagree and this fails. It cannot pass
    // vacuously either — `plant_dock_regions` gives a real nested forest and the dot sits where the
    // hysteresis band is genuinely ambiguous, so the prior is load-bearing for the outcome.
    let run = |seed_membership: bool| -> Vec<RealmId> {
        let mut rig = Rig::new(); // owns System 7 == OWN_REALM
        rig.grant_realm();
        plant_dock_regions(&mut rig);
        let entity = EntityId::pack(EntityKind::Player, 10, 1, 78);
        // THE POSITION THAT MATTERS: just inside the inner child's shell (r=1000) but NOT far enough
        // in to ACQUIRE it from scratch — the band's ambiguous zone, where the prior alone decides
        // membership. This is exactly where a player who stops on a boundary ends up.
        //
        // The subject is OWNED BY the child (its pose frame names Planet 42), i.e. it has just been
        // handed off inward. A correct shard therefore keeps it there and emits NO crossing. With the
        // stored-only prior, a shard with no memory of it fails to acquire the child, folds its
        // container out to the parent, and immediately re-homes it BACK OUT — the flap.
        insert_owned_dot_framed(
            &mut rig,
            TRIG_SESSION,
            entity,
            FrameRef::PlanetCentered { planet_seed: 42 },
            DVec3::new(990.0, 0.0, 0.0),
        );
        if seed_membership {
            // The SOURCE shard's state: it already remembers this subject inside the child.
            let mut bits = RegionMembership::default();
            bits.set(ROOT_REALM, true);
            bits.set(OWN_REALM, true);
            // The child — the membership an arriving shard does NOT have. Named by REALM now, not by a
            // position in a bitset, so this seeding says what it means and cannot drift with an index.
            bits.set(OTHER_REALM, true);
            rig.world
                .resource_mut::<ContainmentProgress>()
                .0
                .insert(entity, bits);
        }
        let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
        for t in 2..8 {
            rig.set_local_tick(t);
            all.extend(rig.tick(vec![]));
        }
        crossing_requests(&all).iter().map(|r| r.to_realm).collect()
    };

    let remembered = run(true);
    let blank = run(false);
    assert_eq!(
        blank, remembered,
        "a shard that has never seen this subject must decide EXACTLY as one that remembers it — \
         otherwise the two sides of a hand-off disagree and the subject flaps at the boundary"
    );
    // And the correct shared answer is "no crossing at all": the dot is already in its owning realm.
    assert_eq!(remembered, Vec::<RealmId>::new());
}

#[test]
fn owning_realm_reads_the_pose_frame_when_nameable() {
    // The owning realm is the pose FRAME's realm. There is no longer a fallback to ignore — every frame
    // names one — so what this covers is that each KIND of frame names the right thing.
    assert_eq!(
        super::owning_realm(FrameRef::SystemSpace { system_seed: 7 }),
        RealmId::System(7),
        "a System frame owns System(system_seed), not the config fallback",
    );
    assert_eq!(
        super::owning_realm(FrameRef::PlanetCentered { planet_seed: 3 }),
        RealmId::Planet(3),
        "a Planet frame owns Planet(planet_seed)",
    );
    assert_eq!(
        super::owning_realm(FrameRef::AreaLocal {
            planet_seed: 3,
            area_seed: 8,
        }),
        RealmId::Area(8),
        "an Area frame owns Area(area_seed) — the very frame the to_parent fix makes form",
    );
}

#[test]
fn owning_realm_names_the_galaxy_and_the_universe_too() {
    // ★ REPLACES `owning_realm_falls_back_to_config_realm_for_an_unnameable_frame` (slice S9). That test
    // drove a fallback whose ONLY reachable input was galaxy space, and its own comment said so:
    // "otherwise UNCOVERABLE — no live shard uses GalaxySpace". The fallback is gone because the reason
    // for it is: a galaxy names its realm now, so there is nothing left to fall back FROM.
    //
    // What is asserted instead is the property that replaced it — the two frames that used to have no
    // answer now give one, and it is their own.
    assert_eq!(
        super::owning_realm(FrameRef::GalaxySpace { galaxy_seed: 5 }),
        RealmId::Galaxy(5),
        "a galaxy frame owns the galaxy it names",
    );
    assert_eq!(
        super::owning_realm(FrameRef::UniverseSpace),
        RealmId::Universe,
        "the universe frame owns the universe — there is exactly one",
    );
}

#[test]
fn durable_dot_dwelling_in_band_triggers_exactly_one_crossing_request() {
    let mut rig = Rig::new();
    rig.grant_realm();
    // The dock forest (root ⊃ own(System(7)) ⊃ child(Planet(42))).
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 1);
    // Deep inside the CHILD region's shell (|100| ≪ 1000 - inset): container == OTHER_REALM ⇒ re-home.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));

    // Under containment the band IS the dwell: the re-home fires as soon as the container differs;
    // the RequestInFlight latch then suppresses every later tick → exactly ONE request across the dwell.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = crossing_requests(&all);
    assert_eq!(
        reqs.len(),
        1,
        "exactly ONE CrossingRequest across the dwell"
    );
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
    assert_eq!(reqs[0].from_realm, config().realm);
    assert_eq!(reqs[0].to_realm, OTHER_REALM);
    assert_eq!(reqs[0].subject_fence, Fence(1));
    // Slice 3f: the source threads the dot's session (its `Dots` map key) so the orchestrator's
    // saga can `PrepareSubscribe` to the client's gateway.
    assert_eq!(reqs[0].session, TRIG_SESSION);
    // 3f-D: the FIRST crossing uses attempt 0.
    assert_eq!(reqs[0].attempt, 0);
    // The latch is set to the deterministic id BOTH ends derive (attempt 0).
    assert_eq!(
        rig.world.resource::<RequestInFlight>().0.get(&entity),
        Some(&crossing_transfer_id(
            DirectoryKey::Entity(entity),
            Fence(1),
            0
        )),
    );
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(stats.crossings_requested, 1);
    // The cooldown was armed (`last_commit_tick` set) on the emit.
    assert!(
        rig.world
            .resource::<CrossingProgress>()
            .0
            .get(&entity)
            .expect("progress")
            .last_commit_tick
            .is_some(),
        "the commit armed the cooldown",
    );
}

#[test]
fn the_in_flight_latch_suppresses_a_second_request() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 2);
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(100.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    // Deep inside the child ⇒ the FIRST re-home fires (sets the latch + arms the cooldown).
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    assert_eq!(crossing_requests(&all).len(), 1, "the first crossing fired");
    assert!(
        rig.world
            .resource::<RequestInFlight>()
            .0
            .contains_key(&entity)
    );
    // WITHOUT sending the Demote terminal (so the latch STAYS set), drive a SECOND container change:
    // leave the child region (container reverts to own ⇒ no re-home) long enough for the cooldown to
    // lapse, then re-enter the child (container flips back to OTHER_REALM). This second re-home
    // decision reaches `fan_out_crossing` — but the latch is still held, so it takes the SUPPRESS arm.
    for t in 8..12 {
        rig.set_local_tick(t);
        // Outside the child shell (sd 1000 > outset) but still inside own ⇒ container == own.
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(2000.0, 0.0, 0.0));
        all.extend(rig.tick(vec![]));
    }
    for t in 12..18 {
        rig.set_local_tick(t);
        move_dot(&mut rig, TRIG_SESSION, DVec3::new(100.0, 0.0, 0.0)); // back inside the child
        all.extend(rig.tick(vec![]));
    }
    assert_eq!(
        crossing_requests(&all).len(),
        1,
        "the RequestInFlight latch holds it to ONE request across the second container change",
    );
    assert!(
        rig.world
            .resource::<StubStats>()
            .crossings_suppressed_in_flight
            > 0,
        "the second container change hit the in-flight SUPPRESS arm",
    );
}

#[test]
fn a_transient_crossing_emits_a_transient_crossing_request() {
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    // A Debris entity is Transient → the transient fan-out arm.
    let entity = EntityId::pack(EntityKind::Debris, 10, 1, 3);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: StampedPose::at_rest(
                config().frame,
                DVec3::new(100.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::from_metres(
                DVec3::new(100.0, 0.0, 0.0),
                vd_core::pose::Tier::Fine,
            ),
        },
    );
    // A transient carries NO per-entity latch (its batch journal dedups instead), so the ONLY
    // anti-thrash is the symmetric `k_dwell` cooldown (§2.7). Run WITHIN the cooldown window
    // (commit at tick 2, k_dwell = 5) so the container-differs re-home fires exactly once.
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..7 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    let reqs = transient_crossing_requests(&all);
    assert_eq!(reqs.len(), 1, "exactly ONE TransientCrossingRequest");
    assert_eq!(reqs[0].subject, DirectoryKey::Entity(entity));
    assert_eq!(reqs[0].to_realm, OTHER_REALM);
    assert_eq!(reqs[0].src_realm_fence, Fence(1));
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transient_crossings_requested,
        1
    );
    // Transients are NOT latched in RequestInFlight (the batch journal dedups instead).
    assert!(rig.world.resource::<RequestInFlight>().0.is_empty());
}

#[test]
fn a_durable_tagged_transient_degrades_instead_of_panicking() {
    // REGRESSION (goal-audit L4): a Durable-TAGGED entity in the held-transient set (a kind/loop
    // mismatch — e.g. a mis-tagged batch item) reaches the durable crossing arm from the transient
    // loop, which passes `None` for the session. The arm must DEGRADE (count + emit nothing), NOT
    // panic on `subject_session.expect`, and must leave NO orphan latch.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    // A Player entity is DURABLE, but we place it (wrongly) in the transient set.
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 9);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: StampedPose::at_rest(
                config().frame,
                DVec3::new(100.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::from_metres(
                DVec3::new(100.0, 0.0, 0.0),
                vd_core::pose::Tier::Fine,
            ),
        },
    );
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![])); // must not panic
    }
    assert!(
        crossing_requests(&all).is_empty(),
        "a session-less durable subject emits NO CrossingRequest",
    );
    assert!(
        rig.world.resource::<RequestInFlight>().0.is_empty(),
        "the degradation leaves no orphan latch",
    );
    assert!(
        rig.world
            .resource::<StubStats>()
            .crossing_durable_no_session
            > 0,
        "the kind/loop mismatch is counted, not panicked",
    );
}

#[test]
fn a_dot_inside_only_its_own_realm_triggers_no_re_home() {
    // The "no re-home" case (§4): a dot inside the OWN region (System(7)) but OUTSIDE the deeper child
    // (Planet(42)) has container == OWN_REALM == the realm the shard owns it in ⇒ `should_rehome`
    // returns None. No CrossingRequest, no latch — symmetric with the empty-registry inert path.
    let mut rig = Rig::new();
    rig.grant_realm();
    plant_dock_regions(&mut rig);
    let entity = EntityId::pack(EntityKind::Player, 10, 1, 4);
    // Outside the child shell (sd 3000 > outset) but well inside own ⇒ container == OWN_REALM.
    insert_owned_dot(&mut rig, TRIG_SESSION, entity, DVec3::new(4000.0, 0.0, 0.0));
    let mut all: Vec<(NodeId, MsgClass, Vec<u8>)> = Vec::new();
    for t in 2..8 {
        rig.set_local_tick(t);
        all.extend(rig.tick(vec![]));
    }
    // Split the two emptiness checks (HR5(d): a single `assert!(a && b)` leaves the short-circuit
    // false-branch of `a` uncovered).
    assert!(
        crossing_requests(&all).is_empty(),
        "container == own ⇒ NO durable CrossingRequest",
    );
    assert!(
        transient_crossing_requests(&all).is_empty(),
        "container == own ⇒ NO TransientCrossingRequest",
    );
    assert!(
        rig.world.resource::<RequestInFlight>().0.is_empty(),
        "no re-home ⇒ no latch",
    );
}

#[test]
fn open_hold_is_inert_at_a_zero_ttl() {
    // THE DISARMED ARM. A fully populated call at a zero budget leaves the ledger empty, so a shard
    // that was never given a budget behaves exactly as it did before the ledger existed.
    let mut holds = HandoffHolds::default();
    assert_eq!(
        open_hold(&mut holds, hold_key(7), Fence(2), TickId(10), 0),
        HoldOutcome::Inert
    );
    assert!(holds.0.is_empty());
}

#[test]
fn open_hold_supersedes_a_newer_fence_refuses_an_older_one_and_never_refreshes_the_budget() {
    let mut holds = HandoffHolds::default();
    let k = hold_key(7);
    assert_eq!(
        open_hold(&mut holds, k, Fence(2), TickId(10), 8),
        HoldOutcome::Opened
    );

    // A GENUINELY newer crossing replaces the hold outright — and re-anchors the budget, because it
    // is a different hand-off, not a continuation of the old one.
    assert_eq!(
        open_hold(&mut holds, k, Fence(3), TickId(14), 8),
        HoldOutcome::Superseded
    );
    assert_eq!(holds.0[&k].opened_at, TickId(14));

    // THE IMMORTALITY GUARD: a same-fence re-open (a saga re-driving its demote every tick) leaves the
    // budget anchored where it was. Without this, a wedged hand-off would keep a realm alive forever by
    // heartbeat alone — the one failure the budget exists to make impossible.
    assert_eq!(
        open_hold(&mut holds, k, Fence(3), TickId(17), 8),
        HoldOutcome::Refreshed
    );
    assert_eq!(holds.0[&k].opened_at, TickId(14));

    // A DELAYED REDELIVERY of the FIRST crossing, arriving after the second opened its hold. On an
    // at-least-once mesh this is a real delivery, not a hypothetical, and honouring it would rewind
    // the ledger to a hand-off that has already been superseded.
    assert_eq!(
        open_hold(&mut holds, k, Fence(2), TickId(19), 8),
        HoldOutcome::RefusedStale
    );
    assert_eq!(holds.0[&k].takeover_fence, Fence(3));
    assert_eq!(holds.0[&k].opened_at, TickId(14));
}

#[test]
fn close_hold_at_fence_accepts_the_matching_proof_and_refuses_a_stale_one() {
    let mut holds = HandoffHolds::default();
    let k = hold_key(7);
    // Nothing to close.
    assert!(!close_hold_at_fence(&mut holds, k, Fence(3)));

    open_hold(&mut holds, k, Fence(3), TickId(1), 8);
    // A REPLAYED take-over from the superseded crossing must not close the live hold.
    assert!(!close_hold_at_fence(&mut holds, k, Fence(2)));
    assert!(holds.0.contains_key(&k));
    // The matching proof does.
    assert!(close_hold_at_fence(&mut holds, k, Fence(3)));
    assert!(holds.0.is_empty());
}

#[test]
fn close_hold_is_unconditional_and_reports_whether_there_was_one() {
    // The terminals where the subject is simply gone have no take-over to prove.
    let mut holds = HandoffHolds::default();
    let k = hold_key(7);
    assert!(!close_hold(&mut holds, k));
    open_hold(&mut holds, k, Fence(3), TickId(1), 8);
    assert!(close_hold(&mut holds, k));
    assert!(holds.0.is_empty());
}

#[test]
fn both_ends_of_a_same_node_rehome_are_held_at_once() {
    // Why the key carries the ROLE: on a co-hosting shard a re-home hands the subject from one realm
    // to another WITHOUT leaving the machine, so the same shard is both ends. A bare entity key would
    // silently collapse the two and one end would be lost.
    let mut holds = HandoffHolds::default();
    let e = EntityId::pack(EntityKind::Player, 1, 7, 3);
    open_hold(&mut holds, (e, HoldRole::Source), Fence(2), TickId(1), 8);
    open_hold(&mut holds, (e, HoldRole::Dest), Fence(2), TickId(1), 8);
    assert_eq!(holds.0.len(), 2);
    assert!(close_hold(&mut holds, (e, HoldRole::Source)));
    assert_eq!(
        holds.0.len(),
        1,
        "closing one end leaves the other standing"
    );
}

#[test]
fn handing_over_reads_the_source_end_only_and_expires_with_the_budget() {
    // The consumer-side question, asked of the two ways it can be false — no hold at all, and a hold
    // that has aged past its budget — plus the role split (a DEST hold is not this shard handing away).
    let mut holds = HandoffHolds::default();
    let e = EntityId::pack(EntityKind::Player, 1, 7, 3);
    assert!(
        !handing_over(&holds, e, TickId(5), 4),
        "no hold, no hand-off"
    );
    open_hold(&mut holds, (e, HoldRole::Dest), Fence(2), TickId(5), 4);
    assert!(
        !handing_over(&holds, e, TickId(5), 4),
        "a DEST hold is the other end — this shard is not handing anything away"
    );
    open_hold(&mut holds, (e, HoldRole::Source), Fence(2), TickId(5), 4);
    assert!(handing_over(&holds, e, TickId(5), 4));
    assert!(
        handing_over(&holds, e, TickId(8), 4),
        "age 3 of 4 is still live"
    );
    assert!(
        !handing_over(&holds, e, TickId(9), 4),
        "past the budget the hold stops answering even before the prune sweeps it"
    );
}

#[test]
fn hold_live_covers_both_edges_and_a_backwards_clock() {
    let h = HandoffHold {
        takeover_fence: Fence(2),
        opened_at: TickId(10),
    };
    assert!(hold_live(&h, TickId(10), 4), "age 0 is live");
    assert!(
        hold_live(&h, TickId(13), 4),
        "age 3 against a budget of 4 is live"
    );
    assert!(
        !hold_live(&h, TickId(14), 4),
        "age 4 is EXPIRED — the budget is exclusive"
    );
    assert!(!hold_live(&h, TickId(99), 4));
    // A BACKWARDS clock reads as age zero rather than wrapping to a colossal age and expiring the
    // whole ledger at once.
    assert!(hold_live(&h, TickId(1), 4));
}

#[test]
fn prune_drops_exactly_the_expired_holds_and_reports_how_many() {
    let mut holds = HandoffHolds::default();
    open_hold(&mut holds, hold_key(1), Fence(2), TickId(0), 4);
    open_hold(&mut holds, hold_key(2), Fence(2), TickId(6), 4);
    // The DROPPED KEYS come back (slice F: an expired Source hold is a leaver-vanish moment
    // the caller must fan the eviction for).
    assert_eq!(prune_holds(&mut holds, TickId(8), 4), vec![hold_key(1)]);
    assert_eq!(holds.0.len(), 1);
    assert!(holds.0.contains_key(&hold_key(2)));
    // A second prune with nothing due reports nothing — the arm that proves it is not clearing.
    assert_eq!(prune_holds(&mut holds, TickId(8), 4), Vec::new());
}

#[test]
fn arriving_at_a_star_system_lands_on_its_edge_and_shows_its_planets() {
    // THE GATE for "every realm counts from its own centre", driven through the PRODUCTION path — the
    // same rebase the source shard performs at flush, with the same frame context the shard builds.
    // (An earlier draft called the rebase with the identity stub and so proved nothing about the
    // system; measuring a stand-in is not a measurement.)
    //
    // Two assertions, one arrival, because they are two faces of one defect:
    //   (1) you land ON the boundary you crossed, carrying a position no larger than the realm itself;
    //   (2) standing there, that system's planets are in range — the empty-neighbour-system bug.
    let cfg = vd_physics::worldgen::UniverseConfig::visual_demand(500.0, 0.02);
    let world = vd_physics::worldgen::WorldView::generated(0, &cfg);
    let system = world
        .regions()
        .iter()
        .find(|r| {
            // Flatten the NORMALIZED centre — at the seeded placement radius every component is
            // ulp-coarser than one cell, so the residual `.offset()` is exactly ZERO and only
            // the integer half carries the position.
            // ★ THE FIFTEENTH SITE (slice S9): this read the SYSTEM's own step to flatten a centre
            // the GALAXY authored. It picks "a system away from the origin", and every candidate
            // looked 2048× further out than it is — which happened not to change WHICH system it
            // picked, so nothing ever failed.
            matches!(r.realm, RealmId::System(_))
                && r.centre_m(world.regions())
                    .is_some_and(|c| c.length() > 0.0)
        })
        .copied()
        .expect("a galaxy of several stars has one away from the origin");
    let parent = world
        .regions()
        .iter()
        .find(|r| Some(r.realm) == system.parent)
        .copied()
        .expect("its parent is in the same world");
    let held = BTreeSet::from([system.realm]);
    let regions =
        RealmRegions::new(world.neighbourhood(&held)).with_moving_children(kepler_motion_fns(
            vd_physics::worldgen::moving_children_for_config(0, &cfg, system.realm)
                .into_iter()
                .collect(),
        ));
    // THE SOURCE shard — the one the occupant is leaving, which holds the PARENT realm. It is the only
    // side that can do this conversion, and under the ground rule it is the side that must: it authored
    // where this system sits, so it knows; the system itself does not and never will.
    let src_held = BTreeSet::from([parent.realm]);
    let src_regions =
        RealmRegions::new(world.neighbourhood(&src_held)).with_moving_children(kepler_motion_fns(
            vd_physics::worldgen::moving_children_for_config(0, &cfg, parent.realm)
                .into_iter()
                .collect(),
        ));
    let tick_hz = 50.0;
    let tick = UniverseTick(0);
    let extent = system.shape.circumscribed_extent();
    // Approach along -X and cross at the near face, expressed in the PARENT's frame — where an
    // occupant about to cross in genuinely is.
    // Build the approach position in LATTICE space (centre TRANSLATED by −extent), not as
    // an f64 sum: at the 2.25e15 m galaxy magnitude a flattened `centre − extent` rounds to
    // the 0.25 m grid BEFORE normalization and the exact-relabel claim below would be
    // measuring representability, not the relabel (the H-21 class the activation cures).
    let at_the_face_lattice = system
        .center
        .in_parents_frame()
        .translated(DVec3::new(-extent, 0.0, 0.0), parent.frame.tier());
    let approaching = StampedPose {
        frame: parent.frame,
        pos: at_the_face_lattice,
        vel: DVec3::ZERO,
        orient: glam::DQuat::IDENTITY,
        universe_tick: tick,
    };
    // The PARENT'S authored book does the rebase. An earlier draft built this from the DESTINATION
    // and so asked the arriving realm to place itself — the one thing it cannot do. It answered with
    // the pose unchanged, still measured from the galaxy, which is exactly the 12 km error the owner
    // flew into.
    let src_book = src_regions.author_book(parent.realm, tick_hz, tick);
    let arrived = vd_core::frame::transfer_frame(&approaching, system.frame, &src_book)
        .expect("a galaxy can place its own star system");
    // …and the DESTINATION re-runs the receiver's conversion on receipt, which must be a no-op: the
    // pose already arrives in its frame ("the child does nothing" — the same-frame accept arm).
    let frames = regions.author_book(system.realm, tick_hz, tick);
    assert_eq!(
        vd_core::frame::transfer_frame(&arrived, system.frame, &frames)
            .expect("a same-frame transfer is the identity")
            .pos,
        arrived.pos,
        "the receiver's re-run must be a no-op downward — converting a pose into the frame it is \
         already in cannot move it"
    );

    // (1) ON the edge, on the side we came from. Asserted as an EXACT position, not merely "inside":
    // a bound alone would be satisfied by relocating the occupant to the centre, which is precisely the
    // failure the owner called out — arriving must not move you, only rename where you are. Approaching
    // the near face along -X, that is exactly one extent out on -X in the realm's own frame.
    assert_eq!(
        fm(arrived.pos),
        DVec3::new(-extent, 0.0, 0.0),
        "arriving must RELABEL the occupant, not move it: entering at the near face must read exactly \
         one extent out on the side it came from"
    );
    // …and the crossing is a pure relabel, so nothing about its motion changed either.
    assert_eq!(arrived.vel, approaching.vel);
    // (1b) CONTAINMENT must agree: having arrived on the edge, the occupant is INSIDE this realm.
    // Arriving somewhere the containment engine then denies you are is how a crossing flaps.
    let sd = vd_core::geometry::region_signed_distance(&arrived, &system, &frames)
        .expect("the shard's own frame resolves");
    assert!(
        sd <= 0.0,
        "arrived on the edge but containment says OUTSIDE — the occupant's frame and the \
         boundary's own position disagree"
    );

    // (2) …and from there its planets are visible. This is the owner's live finding: warp to a
    // neighbouring star, arrive, and find the system empty. It works at the HOME system only because
    // that one sits at the origin, where "measured from my star" and "measured from the universe" are
    // the same numbers — which is why every existing proof of streaming planets flies there.
    let placements = regions.child_placements(system.realm, tick_hz, tick);
    assert!(
        !placements.is_empty(),
        "the hosting shard holds this system's planets at all"
    );
    let visible = placements
        .iter()
        .filter(|(region, pose)| {
            let dist =
                occupant_child_dist(arrived.pos, DVec3::ZERO, pose.pos, system.frame.tier(), 0.0);
            region.aoi.in_range(false, dist)
        })
        .count();
    assert!(
        visible > 0,
        "standing on this system's edge, none of its planets are in range"
    );
}

/// A GRANDCHILD's claim on the scanned point (a region whose parent is another child, not the
/// owning realm): the crossing still moves ONE hop — into the DIRECT child — because travel is
/// always through the parent chain, level by level; the grandchild is the NEXT shard's decision.
/// (Measured: the first draft of this test expected the deepest region to win the request and
/// the machinery correctly refused — the one-hop law is the assertion now.) The grandchild's
/// claim still runs the containment fold's non-direct-child branch, which is the log site this
/// pins.
#[test]
fn a_grandchilds_claim_still_crosses_one_hop_into_the_direct_child() {
    let grandchild = RealmId::Station(77);
    let mut rig = Rig::new();
    rig.grant_realm();
    *rig.world.resource_mut::<RealmRegions>() = RealmRegions::new(vec![
        root_region(),
        own_region(),
        region(OTHER_REALM, Some(OWN_REALM), DVec3::ZERO, 1000.0),
        region(grandchild, Some(OTHER_REALM), DVec3::ZERO, 100.0),
    ]);
    insert_owned_dot(&mut rig, TRIG_SESSION, player(7), DVec3::ZERO);
    let sent = rig.tick(vec![]);
    let decided: Vec<RealmId> = crossing_requests(&sent)
        .iter()
        .map(|r| r.to_realm)
        .collect();
    assert_eq!(
        decided,
        vec![OTHER_REALM],
        "one hop, into the DIRECT child — the grandchild is the next shard's decision"
    );
}

/// The conversion path's hop cap: a parent CYCLE (boot-rejected on any real forest) never
/// reaches a root, so there is NO chain — the path refuses and the flush falls to the verbatim
/// arm, never a hang and never an LCA over a chain that lies.
#[test]
fn a_cyclic_roster_yields_no_conversion_path() {
    let cyclic = RealmRegions::new(vec![
        region(
            RealmId::Planet(1),
            Some(RealmId::Planet(2)),
            DVec3::ZERO,
            10.0,
        ),
        region(
            RealmId::Planet(2),
            Some(RealmId::Planet(1)),
            DVec3::new(100.0, 0.0, 0.0),
            10.0,
        ),
    ]);
    assert_eq!(
        conversion_path(
            &cyclic,
            RealmId::Planet(1),
            RealmId::Planet(1),
            RealmId::Planet(2)
        ),
        None,
        "a cycle has no chain, so it has no path"
    );
}
