//! The realm lease and session admission, merged per reader 1's recommendation: boot requests the lease until granted, a foreign grant is refused, a lost lease self-fences, the holder re-reads and renews realm and entity leases on cadence, logout revokes at the recorded fence; then how a session's dot is admitted — spawn-pose resolution and the two-phase attach handshake through the directory, races, wrong-realm routing, idempotent re-attach. Local helper foreign_frame (2407) travels.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

#[test]
fn boot_requests_the_realm_lease_until_granted_then_stops() {
    let mut rig = Rig::new();
    let expected_request =
        postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
            key: DirectoryKey::Realm(config().realm),
            owner: AuthorityRef::Shard(SHARD),
            fence: Fence(1),
        }))
        .expect("encode");
    // Two unanswered ticks: two identical idempotent requests.
    for _ in 0..2 {
        let sent = rig.tick(vec![]);
        assert_eq!(sent, vec![(ORCH, MsgClass::Saga, expected_request.clone())]);
    }
    rig.grant_realm();
    assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
    // Granted: no more requests (and no frames — no sessions yet).
    assert_eq!(rig.tick(vec![]), vec![]);
}

#[test]
fn foreign_realm_grant_is_rejected_loudly_and_authority_stays_none() {
    let mut rig = Rig::new();
    let reply = DirectoryReply::Head {
        key: DirectoryKey::Realm(config().realm),
        record: Some(vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(NodeId(99)),
            fence: Fence(1),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(reply),
    )]);
    assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
}

#[test]
fn a_lost_realm_lease_self_fences_the_shard() {
    // FENCE-1/5/8: once granted, a realm head showing a FOREIGN owner (a P2
    // takeover) or NO record makes the shard drop authority and stop frames —
    // a stale old owner cannot affect clients (fence rule 4).
    let mut rig = Rig::new();
    rig.grant_realm();
    assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
    // A foreign owner head: self-fence.
    let foreign = DirectoryReply::Head {
        key: DirectoryKey::Realm(config().realm),
        record: Some(vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(NodeId(99)),
            fence: Fence(2),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(foreign),
    )]);
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        None,
        "self-fenced"
    );

    // Re-grant, then a headless realm read (record gone): also self-fence.
    rig.grant_realm();
    assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
    let gone = DirectoryReply::Head {
        key: DirectoryKey::Realm(config().realm),
        record: None,
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(gone),
    )]);
    assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
}

#[test]
fn a_granted_shard_periodically_re_reads_its_realm_head() {
    // With a re-check interval, a granted shard sends a HeadRead so a revoked
    // lease is OBSERVED (the self-fence reaction is otherwise unreachable).
    let mut rig = Rig::with_config(StubConfig {
        realm_recheck_interval: 2,
        ..config()
    });
    rig.grant_realm();
    let is_head_read = |sent: &[(NodeId, MsgClass, Vec<u8>)]| {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .any(|(_, _, bytes)| {
                let flow: InterShardFlow =
                    postcard::from_bytes(bytes).expect("directory flow decodes");
                flow == InterShardFlow::Directory(DirectoryOp::HeadRead {
                    key: DirectoryKey::Realm(config().realm),
                })
            })
    };
    // An EVEN tick (local_tick % 2 == 0) re-reads the realm head.
    rig.set_local_tick(2);
    assert!(is_head_read(&rig.tick(vec![])), "even tick re-reads");
    // An ODD tick does not (the interval gate's other branch).
    rig.set_local_tick(3);
    assert!(!is_head_read(&rig.tick(vec![])), "odd tick is quiet");
}

#[test]
fn a_granted_shard_renews_its_realm_and_granted_entities_on_cadence() {
    // D-3 heartbeat: on the renew cadence a granted shard re-sends LeaseRenew for its Realm AND
    // every GRANTED, NON-DEPARTING Entity — never a non-granted (still-granting) or departing
    // (logging-out) dot. INERT off-cadence + when the interval is 0.
    let mut rig = Rig::with_config(StubConfig {
        lease_renew_interval_ticks: 4,
        ..config()
    });
    rig.grant_realm();
    let mk = |entity: EntityId, granted: bool, departing: bool| Dot {
        last_stick: None,
        entity,
        account: AccountId(1),
        session_fence: Fence(1),
        gateway: GATEWAY,
        granted,
        input_active: false,
        adopting: false,
        authority: Authority::Owned { fence: Fence(1) },
        departing,
        entity_fence: Fence(1),
        pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(0)),
        yaw: 0.0,
        pitch: 0.0,
        last_applied_seq: None,
        prev_offset: LatticePos::ORIGIN,
    };
    let granted_e = EntityId::pack(EntityKind::Player, 10, 1, 1);
    let provisional_e = EntityId::pack(EntityKind::Player, 10, 2, 2);
    let departing_e = EntityId::pack(EntityKind::Player, 10, 3, 3);
    {
        let mut dots = rig.world.resource_mut::<Dots>();
        dots.0.insert(SessionId(1), mk(granted_e, true, false));
        dots.0.insert(SessionId(2), mk(provisional_e, false, false)); // emits LeaseGrant
        dots.0.insert(SessionId(3), mk(departing_e, true, true)); // emits LeaseRevoke
    }
    let renew_keys = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<DirectoryKey> {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew { key, fence })) => {
                        assert_eq!(fence, Fence(1), "renews at the held fence");
                        Some(key)
                    }
                    _ => None,
                },
            )
            .collect()
    };
    // A multiple tick renews the Realm + the granted, non-departing entity ONLY.
    rig.set_local_tick(4);
    let on = renew_keys(&rig.tick(vec![]));
    assert!(
        on.contains(&DirectoryKey::Realm(config().realm)),
        "the realm lease is renewed"
    );
    assert!(
        on.contains(&DirectoryKey::Entity(granted_e)),
        "a granted, non-departing entity lease is renewed"
    );
    assert!(
        !on.contains(&DirectoryKey::Entity(provisional_e)),
        "a non-granted (still-granting) entity is NOT renewed"
    );
    assert!(
        !on.contains(&DirectoryKey::Entity(departing_e)),
        "a departing (logging-out) entity is NOT renewed"
    );
    // Off-cadence: no heartbeat at all (the interval gate's modulo branch). The inert branch
    // (interval == 0) is covered by every other granted-shard test (all run config() at interval 0).
    rig.set_local_tick(5);
    assert!(
        renew_keys(&rig.tick(vec![])).is_empty(),
        "an off-cadence tick emits no LeaseRenew"
    );
}

#[test]
fn a_cohosting_shard_renews_its_child_realm_lease_on_cadence() {
    // D-3 heartbeat, co-hosting (task #149): a MULTI-realm shard renews its PRIMARY realm AND every
    // CO-HOSTED CHILD realm it holds (the `cohosted.0` chain in the renewal set). Drives the co-host
    // renewal closure that a single-realm shard never reaches (`cohosted.0` empty). Boot System 7
    // (primary) + Planet 7 (co-hosted child), then a cadence tick renews BOTH realm keys.
    let mut rig = Rig::with_config(StubConfig {
        lease_renew_interval_ticks: 4,
        ..cohost_planet_config()
    });
    boot_cohost_planet(&mut rig); // primary System 7 on RealmAuthority + child Planet 7 on CoHostedAuthority
    assert_eq!(
        rig.world
            .resource::<CoHostedAuthority>()
            .0
            .get(&RealmId::Planet(7)),
        Some(&Fence(1)),
        "precondition: the co-hosted Planet-7 head is held (so the renewal chain has a child to emit)",
    );
    // A still-granting (provisional) dot so the cadence tick ALSO emits a `LeaseGrant` to ORCH — a
    // NON-`LeaseRenew` op that exercises the extractor's fall-through arm (no uncoverable `_ => None`).
    rig.world.resource_mut::<Dots>().0.insert(
        SessionId(1),
        Dot {
            last_stick: None,
            entity: EntityId::pack(EntityKind::Player, 7, 1, 1),
            account: AccountId(1),
            session_fence: Fence(1),
            gateway: GATEWAY,
            granted: false, // provisional ⇒ emits LeaseGrant, NOT LeaseRenew
            input_active: false,
            adopting: false,
            authority: Authority::Owned { fence: Fence(1) },
            departing: false,
            entity_fence: Fence(1),
            pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(0)),
            yaw: 0.0,
            pitch: 0.0,
            last_applied_seq: None,
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let renew_keys = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<DirectoryKey> {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew { key, .. })) => Some(key),
                    _ => None,
                },
            )
            .collect()
    };
    rig.set_local_tick(4); // on the renew cadence
    let on = renew_keys(&rig.tick(vec![]));
    assert!(
        on.contains(&DirectoryKey::Realm(RealmId::System(7))),
        "the PRIMARY realm lease is renewed: {on:?}",
    );
    assert!(
        on.contains(&DirectoryKey::Realm(RealmId::Planet(7))),
        "the CO-HOSTED CHILD realm lease is ALSO renewed (the co-host chain): {on:?}",
    );
}

#[test]
fn logout_revokes_at_the_recorded_entity_fence_not_a_literal() {
    // FENCE-1/5/8: a dot granted at a NON-genesis fence revokes at THAT fence on
    // logout — a hardcoded literal would be Refused and strand the logout.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    // The directory granted the entity at fence 5 (a transfer advanced it).
    let granted_at_5 = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: Some(vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(SHARD),
            fence: Fence(5),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(granted_at_5),
    )]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].entity_fence,
        Fence(5)
    );
    // Detach, then the retry driver revokes at the RECORDED fence 5.
    let detach = GatewayToShard::DetachSession {
        session: SESSION,
        fence: Fence(1),
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &detach)]);
    let sent = rig.tick(vec![]);
    let expected_revoke = InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
        key: DirectoryKey::Entity(entity),
        fence: Fence(5),
    });
    let to_orch: Vec<InterShardFlow> = sent
        .iter()
        .filter(|(to, _, _)| *to == ORCH)
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("flow decodes"))
        .collect();
    assert!(
        to_orch.contains(&expected_revoke),
        "revoke at the recorded fence 5, not a literal: {to_orch:?}"
    );
}

#[test]
fn non_head_directory_replies_are_ignored() {
    // CAS results / clock answers carry no shard obligation (the catch-all arm).
    let mut rig = Rig::new();
    rig.grant_realm();
    let cas = DirectoryReply::CasResult {
        key: DirectoryKey::Realm(config().realm),
        outcome: vd_wire::seams::directory::CasOutcome::Won {
            new_fence: Fence(9),
        },
    };
    let clock = DirectoryReply::ClockNow {
        universe_tick: UniverseTick(5),
        epoch: vd_core::EpochId(1),
    };
    // A non-reply InterShardFlow arm misdirected to the stub on Saga (a SagaAck — the
    // gateway→saga ack) is also ignored: the stub handles ONLY DirectoryReply, every
    // other arm is a no-op (the dispatch `Ok(_) => return`), never a panic or a decode
    // error. (The real Transfer arm to a dest shard lands at Slice 1d.)
    let stray = InterShardFlow::SagaAck(
        vd_wire::seams::transfer_control::TransferControlAck::Committed {
            transfer: vd_core::TransferId(9),
        },
    );
    let _ = rig.tick(vec![
        wire_msg(ORCH, MsgClass::Saga, &InterShardFlow::DirectoryReply(cas)),
        wire_msg(ORCH, MsgClass::Saga, &InterShardFlow::DirectoryReply(clock)),
        wire_msg(ORCH, MsgClass::Saga, &stray),
    ]);
    // Authority unaffected by non-Head replies and the stray arm.
    assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
}

#[test]
fn unrelated_directory_replies_are_ignored() {
    let mut rig = Rig::new();
    // A headless realm read and an entity head: neither grants authority.
    let none_head = DirectoryReply::Head {
        key: DirectoryKey::Realm(config().realm),
        record: None,
    };
    let entity_head = DirectoryReply::Head {
        key: DirectoryKey::Entity(EntityId(1)),
        record: None,
    };
    let _ = rig.tick(vec![
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(none_head),
        ),
        wire_msg(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::DirectoryReply(entity_head),
        ),
    ]);
    assert_eq!(rig.world.resource::<RealmAuthority>().0, None);
}

#[test]
fn undecodable_messages_are_survived() {
    let mut rig = Rig::new();
    let garbage = vec![0xFF, 0x00, 0x13, 0x37];
    let _ = rig.tick(vec![
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: garbage.clone().into(),
        },
        Inbound::Wire {
            from: GATEWAY,
            class: MsgClass::Control,
            bytes: garbage.into(),
        },
        // Non-wire inbound is skipped by the dispatcher.
        Inbound::NodeUnreachable {
            to: GATEWAY,
            class: MsgClass::Snapshot,
            undelivered: MsgId(1),
        },
        // Snapshot/Membership classes carry nothing for the stub dispatcher.
        Inbound::Wire {
            from: GATEWAY,
            class: MsgClass::Snapshot,
            bytes: vec![1].into(),
        },
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Membership,
            bytes: vec![2].into(),
        },
    ]);
    assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
    // Both decode failures (Saga + Control) are COUNTED, never silent (ROB-E2E-1);
    // the Snapshot/Membership garbage is not decoded by the stub, so it adds nothing.
    assert_eq!(rig.world.resource::<StubStats>().undecodable, 2);
}

#[test]
fn a_spawn_pose_offered_in_this_realms_frame_is_taken_verbatim() {
    // The gateway did the conversion — it holds the whole forest and stepped down it subtracting each
    // realm's placement — so the pose arrives already measured from THIS realm's centre. The shard's
    // job is to check the frame and use the number, doing no arithmetic of its own. It cannot do any:
    // it does not know where it itself sits.
    let cfg = config();
    let mut stats = StubStats::default();
    let offered = StampedPose::at_rest(cfg.frame, DVec3::new(3.0, 0.0, 0.0), UniverseTick(0));
    let got = resolve_spawn_pose(
        &cfg,
        AccountId(0x5F3B),
        Some(offered),
        UniverseTick(77),
        &mut stats,
    );
    assert_eq!(got.frame, cfg.frame);
    assert_eq!(fm(got.pos), DVec3::new(3.0, 0.0, 0.0));
    // Re-stamped to the current clock tick; at rest (the offered pose was at rest).
    assert_eq!(got.universe_tick, UniverseTick(77));
    assert_eq!(got.vel, DVec3::ZERO);
    assert_eq!(stats.spawn_poses_refused, 0);
}

#[test]
fn a_stored_spawn_pose_in_the_wrong_frame_is_refused_not_relabelled() {
    // THE FOOT-GUN THIS CLOSES. A pose measured in a frame this realm is not used to be handed to
    // `rebind_pose_to_dest` with an identity context, which renamed the frame and moved no number — so
    // a player stored 3 m above a planet 145 m from its star was planted 3 m from the STAR. There is
    // now no conversion to attempt (only the party holding the whole forest can convert), so the pose
    // is REFUSED, counted, and the avatar births at the origin.
    let mut cfg = config();
    let account = AccountId(0x5F3B);
    let wrong = StampedPose::at_rest(
        foreign_frame(),
        DVec3::new(100.0, 200.0, 300.0),
        UniverseTick(0),
    );
    cfg.spawn_poses.insert(account, wrong);
    let mut stats = StubStats::default();
    let got = resolve_spawn_pose(&cfg, account, None, UniverseTick(77), &mut stats);
    assert_eq!(
        got,
        StampedPose::at_rest(cfg.frame, DVec3::ZERO, UniverseTick(77)),
        "a pose in a frame this realm is not must never be worn — the origin is the honest fallback"
    );
    assert_eq!(stats.spawn_poses_refused, 1);
    // The OFFERED half of the same rule: a wire pose in the wrong frame is refused identically, so a
    // mis-routed login cannot plant an avatar by coming in over the wire instead of the store.
    let mut stats = StubStats::default();
    let got = resolve_spawn_pose(
        &cfg,
        AccountId(0xAB),
        Some(wrong),
        UniverseTick(77),
        &mut stats,
    );
    assert_eq!(
        got,
        StampedPose::at_rest(cfg.frame, DVec3::ZERO, UniverseTick(77))
    );
    assert_eq!(stats.spawn_poses_refused, 1);
}

#[test]
fn the_offered_pose_wins_over_this_realms_own_store() {
    // Order matters and is stated: the gateway's answer is about THIS login, the store is about this
    // account's last known place. A login that carries a position uses it.
    let mut cfg = config();
    let account = AccountId(7);
    cfg.spawn_poses.insert(
        account,
        StampedPose::at_rest(cfg.frame, DVec3::new(9.0, 9.0, 9.0), UniverseTick(0)),
    );
    let mut stats = StubStats::default();
    let offered = StampedPose::at_rest(cfg.frame, DVec3::new(1.0, 0.0, 0.0), UniverseTick(0));
    let got = resolve_spawn_pose(&cfg, account, Some(offered), UniverseTick(5), &mut stats);
    assert_eq!(fm(got.pos), DVec3::new(1.0, 0.0, 0.0));
    assert_eq!(stats.spawn_poses_refused, 0);
}

#[test]
fn resolve_spawn_pose_without_an_entry_is_origin_at_rest_byte_identical() {
    // The `None`/`None` arm: no offered pose and nothing stored ⇒ origin-at-rest in this shard's frame
    // — BYTE-IDENTICAL to the admit literal every rig has always taken.
    let cfg = config(); // empty spawn_poses
    let mut stats = StubStats::default();
    let got = resolve_spawn_pose(&cfg, AccountId(0xAB), None, UniverseTick(42), &mut stats);
    assert_eq!(
        got,
        StampedPose::at_rest(cfg.frame, DVec3::ZERO, UniverseTick(42))
    );
    assert_eq!(stats.spawn_poses_refused, 0);
}

#[test]
fn login_admits_the_dot_at_its_stored_spawn_pose_through_the_one_admit_path() {
    // The admit-path proof (the wiring, not just the helper): a real AttachSession for an account WITH
    // a realm-local stored pose births its dot there — same Ghost-birth admit path, only the pose value
    // differs. `attach_request` attaches AccountId(5), so key the stand-in on it.
    let stored_pos = DVec3::new(11.0, -22.0, 33.0);
    let cfg = config();
    let stored = StampedPose::at_rest(cfg.frame, stored_pos, UniverseTick(0));
    let mut rig = Rig::with_config(StubConfig {
        spawn_poses: BTreeMap::from([(AccountId(5), stored)]),
        ..config()
    });
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    // Born at the STORED pose, re-stamped to the rig clock (universe_tick 100) — NOT origin-at-rest.
    assert_eq!(dot.pose.frame, FrameRef::SystemSpace { system_seed: 7 });
    assert_eq!(fm(dot.pose.pos), stored_pos);
    assert_eq!(dot.pose.universe_tick, UniverseTick(100));
    // `prev_offset` is seeded from the SAME stored offset (not the old hardcoded ZERO).
    assert_eq!(
        dot.prev_offset
            .delta_m(LatticePos::ORIGIN, vd_core::pose::Tier::Fine),
        stored_pos
    );
}

#[test]
fn a_login_is_admitted_at_the_pose_the_gateway_measured_for_it() {
    // THE WIRING, end to end through the real attach message: the gateway put a pose measured from
    // THIS realm's centre in `AttachSession`, and the avatar is born there. This is the path that used
    // to run through a cluster-wide map of universe-absolute positions that the shard relabelled.
    let cfg = config();
    let offered = StampedPose::at_rest(cfg.frame, DVec3::new(3.0, 0.0, 0.0), UniverseTick(0));
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request_with(SESSION, GATEWAY, Some(offered));
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(fm(dot.pose.pos), DVec3::new(3.0, 0.0, 0.0));
    assert_eq!(dot.pose.frame, cfg.frame);
    assert_eq!(rig.world.resource::<StubStats>().spawn_poses_refused, 0);
}

#[test]
fn a_login_routed_to_the_wrong_realm_is_admitted_at_the_origin_and_counted() {
    // The loud arm of the same wiring: a static cluster attaches every login to one fixed shard, which
    // in general is NOT the realm the account's position was measured in. The shard cannot convert
    // (only the party holding the whole forest can), so it says so and births at its own origin —
    // rather than wearing a number that means something else here.
    let elsewhere =
        StampedPose::at_rest(foreign_frame(), DVec3::new(3.0, 0.0, 0.0), UniverseTick(0));
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request_with(SESSION, GATEWAY, Some(elsewhere));
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(fm(dot.pose.pos), DVec3::ZERO);
    assert_eq!(dot.pose.frame, config().frame);
    assert_eq!(rig.world.resource::<StubStats>().spawn_poses_refused, 1);
}

#[test]
fn login_without_a_stored_pose_births_at_the_origin_unchanged() {
    // The admit-path `None` proof: an account with no stand-in entry births origin-at-rest — the
    // byte-identical pre-5f-3b behaviour through the real attach path.
    let mut rig = Rig::new(); // config() ⇒ empty spawn_poses
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.pose,
        StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(100))
    );
    assert_eq!(dot.prev_offset, LatticePos::ORIGIN);
}

#[test]
fn attach_before_realm_grant_is_deferred_and_counted() {
    let mut rig = Rig::new();
    let sent = rig.attach_request(SESSION, GATEWAY);
    // Only the lease re-request went out — no attach reply.
    assert_eq!(sent.len(), 1);
    assert_eq!(rig.world.resource::<StubStats>().attaches_deferred, 1);
    assert_eq!(rig.world.resource::<Dots>().0.len(), 0);
}

#[test]
fn attach_is_two_phase_authority_derives_from_the_directory() {
    let mut rig = Rig::new();
    rig.grant_realm();

    // Phase 1: the attach request spawns a PROVISIONAL dot and asks the
    // directory for its entity grant — no attach reply, no frames yet.
    let sent = rig.attach_request(SESSION, GATEWAY);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert!(!dot.granted, "provisional until the directory records it");
    assert_eq!(dot.account, AccountId(5));
    assert_eq!(dot.gateway, GATEWAY);
    assert_eq!(dot.entity.kind_tag(), EntityKind::Player as u8);
    assert_eq!(dot.entity.mint_shard(), 10, "minted by THIS shard");
    let grant: InterShardFlow = postcard::from_bytes(&sent[0].2).expect("decode");
    assert_eq!(
        grant,
        InterShardFlow::Directory(DirectoryOp::LeaseGrant {
            key: DirectoryKey::Entity(dot.entity),
            owner: AuthorityRef::Shard(SHARD),
            fence: Fence(1),
        })
    );
    assert!(sent.iter().all(|(to, _, _)| *to == ORCH), "directory only");
    assert_eq!(
        decode_frames(&sent).len(),
        0,
        "provisional dots are invisible"
    );
    // The grant is retried every tick until confirmed (idempotent by fence).
    let sent = rig.tick(vec![]);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].0, ORCH);

    // Phase 2: the grant confirmation makes the dot HELD: SessionAttached
    // (with the REAL realm fence) and the first frame flow the same tick.
    // (The retry system also fires one last pre-grant request that tick.)
    let sent = rig.confirm_entity_grant(SESSION);
    assert!(rig.world.resource::<Dots>().0[&SESSION].granted);
    let to_gateway: Vec<ShardToGateway> = sent
        .iter()
        .filter(|(to, class, _)| (*to == GATEWAY) & (*class == MsgClass::Control))
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
        .collect();
    assert_eq!(
        to_gateway,
        vec![ShardToGateway::SessionAttached {
            session: SESSION,
            entity: dot.entity,
            frame: config().frame,
            realm_fence: Fence(1),
        }]
    );
    assert_eq!(decode_frames(&sent).len(), 1, "held dots render");
    // A duplicate grant head is idempotent (no second attach reply).
    let sent = rig.confirm_entity_grant(SESSION);
    let attach_replies = sent
        .iter()
        .filter(|(_, class, _)| *class == MsgClass::Control)
        .count();
    assert_eq!(attach_replies, 0);
}

#[test]
fn foreign_entity_grants_and_pregrant_races_are_survived() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    // The directory says ANOTHER shard owns the entity: loud, not granted.
    let foreign = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: Some(vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(NodeId(99)),
            fence: Fence(1),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(foreign),
    )]);
    assert!(!rig.world.resource::<Dots>().0[&SESSION].granted);
}

#[test]
fn entity_grant_racing_ahead_of_the_realm_lease_waits_for_retry() {
    // The realm lease is NOT granted yet; an entity head arriving anyway
    // cannot activate the dot (no realm fence to stamp) — the per-tick retry
    // resolves it once the realm lease lands.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach_request(SESSION, GATEWAY);
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    rig.world.resource_mut::<RealmAuthority>().0 = None;
    let head = DirectoryReply::Head {
        key: DirectoryKey::Entity(entity),
        record: Some(vd_wire::seams::directory::OwnerRecord {
            authority: AuthorityRef::Shard(SHARD),
            fence: Fence(1),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(head),
    )]);
    assert!(!rig.world.resource::<Dots>().0[&SESSION].granted);
}

#[test]
fn reattach_is_idempotent_and_a_higher_fence_upgrades() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach();
    let first_entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    // Same-fence re-attach: same entity, still one dot.
    let _ = rig.attach();
    assert_eq!(rig.world.resource::<Dots>().0.len(), 1);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].entity,
        first_entity
    );
    // Higher-fence re-attach upgrades the stored fence.
    let msg = GatewayToShard::AttachSession {
        session: SESSION,
        fence: Fence(3),
        account: AccountId(5),
        spawn: None,
    };
    let _ = rig.tick(vec![wire_msg(GATEWAY, MsgClass::Control, &msg)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].session_fence,
        Fence(3)
    );
}
