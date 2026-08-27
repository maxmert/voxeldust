//! The ownership hand-off machinery end to end (banner 3581): the dest-side OpenInputSlot adopt, the applied-steps idempotency journal, the ordered saga Demote/Promote/ReHome consumers including source==dest re-own and the retained-ghost collision, the source-ghost lifecycle (take-over proof, hold closure, band-exit Despawn), then the remaining saga-Promote no-op/defer arms, the deferred promote re-driven after the crossing lands, post-marker input at the resume watermark, the idle re-stamp and the slot's fence/watermark guards. Both ranges must land in ONE module: open_input_slot_f (3606) is used at 5413-5447. Local helper open_input_slot_subj (3586) travels.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

#[test]
fn open_input_slot_adopts_the_subject_input_active_without_attaching() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert!(dot.input_active, "the slot is input-active");
    // 1c.8: the dot ADOPTS the subject — its entity IS the subject id (not a fresh mint),
    // it is `adopting`, NOT granted (the adopt HeadRead flips that), and a Ghost (no simulate).
    assert_eq!(dot.entity, SUBJECT, "the dot adopted the transfer subject");
    assert!(dot.adopting, "the dot is a transfer-dest adopt");
    assert!(
        !dot.granted,
        "the adopt HeadRead has not flipped granted yet"
    );
    assert!(
        !dot.authority.simulates(),
        "the adopt is a Ghost (the frozen mirror) — does not simulate, renders nothing"
    );
    assert_eq!(
        dot.authority,
        Authority::Ghost {
            source_fence: Fence::GENESIS,
            since_tick: TickId(1),
        },
        "the dest adopt is born the GENESIS frozen ghost mirror (Promoted to Owned by the crossing)"
    );
    assert_eq!(
        dot.entity_fence,
        Fence::GENESIS,
        "the adopt HeadRead fills the real fence"
    );
    assert_eq!(
        dot.last_applied_seq,
        Some(5),
        "seeded to the resume watermark"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().last_input_slot_resume,
        Some(5),
        "the as-received resume watermark is latched for the conservation gate"
    );
    // The slot is a SILENT inbound state change: it emits NOTHING — no SessionAttached,
    // no re-home (the source still owns the client connection — R2), and the Ghost adopt dot
    // does not simulate so it renders no snapshot frame either.
    assert!(sent.is_empty(), "the input slot emits nothing back");
}

#[test]
fn an_adopting_dot_head_reads_the_record_never_lease_grants() {
    // 1c.8 HR5: the request_pending_grants 3-way ADOPT arm — an adopting !granted dot emits
    // HeadRead{Entity} (to adopt the record the CAS moved here), NEVER a LeaseGrant (which
    // the directory would Refuse at the post-genesis CAS fence).
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let sent = rig.tick(vec![]);
    let to_orch: Vec<InterShardFlow> = sent
        .iter()
        .filter(|(to, _, _)| *to == ORCH)
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("flow decodes"))
        .collect();
    assert!(
        to_orch.contains(&InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Entity(SUBJECT),
        })),
        "the adopting dot HeadReads its subject: {to_orch:?}"
    );
    // Value-compare (NOT matches!, whose match-success arm would be an uncoverable region):
    // the EXACT LeaseGrant an adopting dot must NEVER send (it adopts via HeadRead instead).
    assert!(
        !to_orch.contains(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
            key: DirectoryKey::Entity(SUBJECT),
            owner: AuthorityRef::Shard(SHARD),
            fence: Fence::GENESIS.next(),
        })),
        "an adopting dot NEVER LeaseGrants its entity: {to_orch:?}"
    );
}

#[test]
fn the_adopt_grant_flip_holds_authority_without_announcing_the_sub_until_promote() {
    // 1d.5b.3b: the adopt grant-flip sets granted + stamps entity_fence + STAYS Ghost, but NO
    // LONGER announces the dest sub — `SubscriptionReady` RELOCATED to on_saga_promote (announced
    // only at the genuine Ghost→Owned promote). So the client stays on the SOURCE sub until then,
    // and demote-before-promote is strict. It still pushes NO SessionAttached (R2).
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let sent = rig.tick(vec![adopted_head(Fence(2))]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    // Split asserts (each &&-short-circuit false arm is uncoverable — HR5).
    assert!(dot.granted, "the adopt flips granted (authority-held)");
    assert_eq!(dot.entity_fence, Fence(2), "stamped the recorded CAS fence");
    assert!(
        !dot.authority.simulates(),
        "an adopt STAYS a Ghost (renders nothing) until the saga Promote flips it Owned"
    );
    assert!(!dot.adopting, "adopting is cleared on the flip");
    // NO SubscriptionReady at the adopt (it moved to the promote) and NO frame rendered.
    assert!(
        gw_replies(&sent).is_empty(),
        "the adopt announces NO gateway reply (the sub moved to the promote): {sent:?}"
    );
    assert_eq!(
        decode_frames(&sent).len(),
        0,
        "an adopted Ghost dot renders nothing (simulates()==false)"
    );

    // After the crossing lands, the saga Promote DOES announce the dest sub (the relocated one).
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(
        gw_replies(&sent),
        vec![ShardToGateway::SubscriptionReady {
            session: SESSION,
            entity: SUBJECT,
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1), // the DEST realm fence (grant_realm set Fence(1))
        }],
        "the Promote announces exactly one SubscriptionReady, never a SessionAttached"
    );
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the Promote flips the dest Ghost→Owned"
    );
}

#[test]
fn a_non_entity_subject_open_input_slot_is_a_counted_noop() {
    // 1c.8 HR5: the DirectoryKey::transfer_subject_entity None arm — a non-Entity subject (e.g.
    // a Realm saga driven through CommitAuthority) does NOT adopt; counted no-op, never a panic.
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![open_input_slot_subj(
        SESSION,
        GATEWAY,
        5,
        Fence(1),
        DirectoryKey::Realm(RealmId::System(99)),
    )]);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "a non-Entity subject mints no dot"
    );
    assert_eq!(rig.world.resource::<StubStats>().input_slots_malformed, 1);
    assert!(sent.is_empty(), "the no-op emits nothing");
}

#[test]
fn applied_step_redelivery_is_idempotent() {
    // 1d.0 PERMANENT GATE: the dest applied_steps journal dedups by the frozen
    // IdempotencyKey::TransferStep (transfer, step_id). The FIRST apply records + returns
    // FirstApply; every redelivery of the SAME key returns AlreadyApplied with no re-effect;
    // a distinct step_id OR transfer is independent.
    let mut steps = AppliedSteps::default();
    let t = TransferId(1);
    assert_eq!(steps.journal_step(t, 0), StepOutcome::FirstApply);
    assert_eq!(
        steps.journal_step(t, 0),
        StepOutcome::AlreadyApplied,
        "a redelivered (transfer, step_id) is a no-op — never a second effect"
    );
    assert_eq!(
        steps.journal_step(t, 1),
        StepOutcome::FirstApply,
        "a distinct step_id is journaled independently"
    );
    assert_eq!(
        steps.journal_step(TransferId(2), 0),
        StepOutcome::FirstApply,
        "a distinct transfer is journaled independently"
    );
}

#[test]
fn the_saga_demote_flips_the_source_to_ghost_and_acks_unconditionally() {
    // 1d.5b.2 SOURCE consumer of the ordered Demote (the SOLE source-demote driver): drive
    // Owned→Frozen→Ghost (REUSING self_fence_foreign_entity) at the new owner fence, then ack
    // DemoteAck. A redelivered Demote finds an already-Ghost dot → the !simulates() counted no-op
    // — but STILL acks (the unconditional ack: never wedge the saga in Demoting).
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach(); // a granted, locally-owned dot at Fence(1)
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    let demote = InterShardFlow::Demote(DemoteCmd {
        transfer: TransferId(7),
        subject: DirectoryKey::Entity(entity),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Ghost {
            source_fence: Fence(2),
            since_tick: TickId(1),
        },
        "the ordered Demote drives Owned{{1}}→Frozen→Ghost at the new owner fence (2)"
    );
    // The demote keys `authority.fence()` to the new owner (2) but does NOT touch `entity_fence`
    // — it stays the dot's OWN old grant fence (1); the intended divergence on a retained Ghost.
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].entity_fence,
        Fence(1),
        "the demote leaves entity_fence at the dot's own grant fence (authority.fence() diverges)"
    );
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "DemoteAck is sent to the orchestrator: {sent:?}"
    );
    // The self-fence is purely LOCAL — no directory write (no LeaseRevoke at the stale fence,
    // no delete of the dest's record): the saga demote's ONLY orch-bound emission is the
    // DemoteAck. (Preserves the deleted poll-era test's no-directory-write guard.)
    assert_eq!(
        sent.iter().filter(|(to, _, _)| *to == ORCH).count(),
        1,
        "the saga demote writes nothing to the directory — only the DemoteAck: {sent:?}"
    );
    // Redelivery on the already-Ghost dot: counted skip, but STILL acks.
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert_eq!(
        rig.world.resource::<StubStats>().self_fence_skipped,
        1,
        "the already-Ghost redelivery is the !simulates() counted no-op"
    );
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "the redelivery STILL acks DemoteAck (never wedge the saga)"
    );
}

#[test]
fn the_saga_demote_on_an_unheld_entity_is_a_clean_noop_but_acks() {
    // The no-match arm of self_fence_foreign_entity (`foreign_takeover_target` finds no dot) —
    // now reachable ONLY via on_saga_demote, since the poll that used to drive it is torn out
    // (1d.5b.2). A Demote for an Entity this shard does not hold is a clean no-op (no flip, no
    // panic) but STILL acks DemoteAck (never wedge the saga in Demoting).
    let mut rig = Rig::new();
    rig.grant_realm(); // realm held, but NO dot attached → this shard holds no entity
    let unheld = EntityId::pack(EntityKind::Player, 99, 99, 99);
    let demote = InterShardFlow::Demote(DemoteCmd {
        transfer: TransferId(7),
        subject: DirectoryKey::Entity(unheld),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert!(
        rig.world.resource::<Dots>().0.is_empty(),
        "an unheld-entity Demote creates or mutates no dot"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().self_fence_skipped,
        0,
        "the no-match arm is distinct from the already-Ghost skip arm"
    );
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "an unheld-entity Demote STILL acks DemoteAck: {sent:?}"
    );
}

#[test]
fn the_saga_demote_on_a_non_entity_subject_is_a_counted_noop_but_acks() {
    // The None arm of subject→entity: a Realm/Session/Ship subject has no local Entity dot to
    // demote — counted (`saga_demote_no_entity`), no flip, but the DemoteAck is STILL sent.
    let mut rig = Rig::new();
    rig.grant_realm();
    let demote = InterShardFlow::Demote(DemoteCmd {
        transfer: TransferId(7),
        subject: DirectoryKey::Realm(config().realm),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &demote)]);
    assert_eq!(rig.world.resource::<StubStats>().saga_demote_no_entity, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::DemoteAck {
                transfer: TransferId(7)
            }
        ),
        "a non-Entity Demote still acks: {sent:?}"
    );
}

#[test]
fn the_saga_promote_flips_owned_announces_the_sub_and_spawns_the_ghost() {
    // 1d.5b.3b: on_saga_promote is the REAL promoter. Set up a transfer-dest Ghost dot (adopt +
    // crossing stores the pose, the dot STAYS Ghost), then the saga Promote: flips Ghost→Owned,
    // announces the dest read-sub (RELOCATED from adopt), registers the source ghost-neighbor,
    // and Spawns the source ghost (the dest DRIVES the feed). Redelivery re-acks only.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot for SUBJECT
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted Ghost
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "still Ghost after the crossing (the autonomous promote is gone)"
    );

    rig.set_local_tick(5); // pins the Spawn's since_tick deterministically
    let source = NodeId(99);
    let sent = rig.tick(vec![promote_msg(Fence(2), source)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the Promote flips the dest Ghost→Owned at the new fence"
    );
    assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "PromoteAck is sent: {sent:?}"
    );
    // The dest read-sub is announced NOW (relocated from the adopt flip).
    assert_eq!(
        gw_replies(&sent),
        vec![ShardToGateway::SubscriptionReady {
            session: SESSION,
            entity: SUBJECT,
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        }],
        "the Promote announces exactly one SubscriptionReady"
    );
    // The source ghost-neighbor is registered (the feed pass has already advanced `seq` once this
    // tick — it runs after process_inbound), and the source ghost is Spawned (the feed started).
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT)
            .map(|n| n.source),
        Some(source),
        "the source ghost-neighbor is registered"
    );
    assert!(
        flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::SpawnV2 {
            entity: SUBJECT,
            source_fence: Fence(2),
        })),
        "the take-over proof is sent — pose-free (slice F): {sent:?}"
    );

    // Redelivery: re-ack only, NO re-flip / re-register / re-proof — the journal returns
    // AlreadyApplied, so `promote_apply` (which holds the flip + register + proof) is NOT
    // entered. `promotes_confirmed` staying 1 proves it.
    let sent = rig.tick(vec![promote_msg(Fence(2), source)]);
    assert_eq!(rig.world.resource::<StubStats>().promotes_redelivered, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().promotes_confirmed,
        1,
        "no re-flip / re-Spawn on redelivery (promote_apply gated on FirstApply)"
    );
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn the_saga_promote_re_owns_a_source_equals_dest_ghost_at_the_exact_cas_fence() {
    // The SOURCE==DEST re-own arm of `promote_apply` (task #149): a co-hosted-child crossing whose
    // `head(Realm(dest))` resolves to THIS node reaches `on_saga_promote` with `cmd.source == self_node`
    // (= the Rig's own SHARD id). Unlike the cross-node case (a GENESIS-fenced Ghost, strictly older than
    // the CAS fence, promoted via `AuthorityCmd::Promote`), the ordered same-node `Demote` already
    // self-fenced THIS dot to `Ghost{source_fence: cmd.new_fence}`, so a strict-newer Promote at the SAME
    // fence would `StaleFence`. The `if cmd.source == self_node` arm RE-OWNS the dot directly at that exact
    // fence — the idempotent route-swap the degenerate saga is. Same Ghost-dot fixture as the cross-node
    // test but with `source = SHARD`, asserting `Owned { fence: <the promote's new_fence> }`.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]); // adopting dot for SUBJECT
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // flip → granted Ghost
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]); // lands STUB_CROSSING_STEP
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "still Ghost after the crossing (source==dest re-own has not run yet)"
    );
    rig.set_local_tick(5);
    // `source == SHARD` (the Rig's own node id) ⇒ `cmd.source == self_node` ⇒ the direct re-own arm.
    let sent = rig.tick(vec![promote_msg(Fence(2), SHARD)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].authority,
        Authority::Owned { fence: Fence(2) },
        "the source==dest Promote RE-OWNS the dot at the exact CAS fence (no strict-newer StaleFence)"
    );
    assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "the source==dest Promote still acks PromoteAck: {sent:?}"
    );
    // The self-ghost tail is skipped on source==dest (no self-feed loop) — mirrors the re-home guard.
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT),
        None,
        "a source==dest promote registers NO self-ghost neighbor",
    );
    assert!(
        flows_to(&sent, SHARD).is_empty(),
        "no GhostFlow::Spawn is sent to self on a source==dest promote: {sent:?}",
    );
}

#[test]
fn on_re_home_creates_an_owned_dot_from_the_pose_acks_and_spawns_the_ghost() {
    // D-37 CELL 2 adopt: the FRESH target receives a `ReHome` and CREATES an Owned dot from the pose
    // (no pre-existing ghost to flip, unlike Promote). Acks PromoteAck, registers + Spawns the source
    // ghost, but emits NO SubscriptionReady (clientless until the session re-homes — D-37/D-36).
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.set_local_tick(5); // pins the Spawn's since_tick deterministically
    let source = NodeId(99);
    let session = SessionId(SUBJECT.0); // the deterministic clientless session key
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        source,
        DirectoryKey::Entity(SUBJECT),
    )]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&session].authority,
        Authority::Owned { fence: Fence(2) },
        "the re-home CREATES an Owned dot born at the new fence"
    );
    assert_eq!(rig.world.resource::<Dots>().0[&session].entity, SUBJECT);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 1);
    assert!(
        saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "PromoteAck is sent: {sent:?}"
    );
    assert!(
        gw_replies(&sent).is_empty(),
        "a clientless re-home announces NO SubscriptionReady"
    );
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT)
            .map(|n| n.source),
        Some(source),
        "the source ghost-neighbor is registered"
    );
    assert!(
        flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::SpawnV2 {
            entity: SUBJECT,
            source_fence: Fence(2),
        })),
        "the take-over proof is sent — pose-free (slice F): {sent:?}"
    );

    // Redelivery: re-ack only, NO re-adopt (journal AlreadyApplied ⇒ re_home_apply not entered).
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        source,
        DirectoryKey::Entity(SUBJECT),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().re_home_redelivered, 1);
    assert_eq!(
        rig.world.resource::<StubStats>().re_home_adopted,
        1,
        "no re-adopt on redelivery (re_home_apply gated on FirstApply)"
    );
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn a_same_node_re_home_registers_no_self_ghost_and_spawns_none() {
    // GHOST-FEED GUARD (task #149): a SOURCE==DEST re-home (the co-hosted-child crossing whose
    // `head(Realm(dest))` resolves to THIS node) reaches `re_home_apply` with `cmd.source == self_node`.
    // `register_and_spawn_source_ghost` must SKIP the self-ghost registration + the `GhostFlow::Spawn`
    // (else a self-feed loop: the owner would `Delta` its own retained copy). The dot is still adopted
    // Owned + acked; ONLY the ghost tail is skipped. `source == SHARD` (the Rig's own node id).
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.set_local_tick(5);
    let session = SessionId(SUBJECT.0);
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        SHARD, // source == this node's id (the source==dest degenerate saga)
        DirectoryKey::Entity(SUBJECT),
    )]);
    // The dot is still adopted Owned + PromoteAck'd (the re-home body ran) …
    assert_eq!(
        rig.world.resource::<Dots>().0[&session].authority,
        Authority::Owned { fence: Fence(2) },
        "a same-node re-home still adopts the dot Owned",
    );
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
    // … but NO self-ghost was registered and NO Spawn was emitted to self.
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT),
        None,
        "a source==dest re-home registers NO self-ghost neighbor",
    );
    assert!(
        flows_to(&sent, SHARD).is_empty(),
        "no GhostFlow::Spawn is sent to self on a same-node re-home: {sent:?}",
    );
}

#[test]
fn a_re_home_colliding_with_the_retained_ghost_flips_it_and_never_mints_a_second_dot() {
    // THE COLLISION (batch review, MAJOR — re_home_apply had no same-entity guard): an upward
    // hand-off demotes this shard's dot to a RETAINED GHOST; the committed dest dies; the
    // forward re-home resolves its target from the flushed pose's frame realm — which on an
    // upward hand-off IS this source realm — so the `ReHome` lands exactly where the ghost
    // still lives. The re-home must FLIP that held dot (same session key, Owned at the CAS
    // fence), never insert a rival under the synthetic `SessionId(entity.0)` key: the orphan
    // ghost's hold would run to TTL and the expiry fan would broadcast `EntityRemoved` for an
    // entity this shard now OWNS and emits.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.set_local_tick(5);
    insert_owned_dot(&mut rig, SESSION, SUBJECT, DVec3::new(3.0, 0.0, 0.0));
    // The outward demote: the dot survives as the retained Ghost under its ORIGINAL session key
    // (0xAA — which also sorts BELOW the synthetic 0xBEEF key, the order that made the doubled
    // state pick the ghost first).
    let _ = rig.tick(vec![demote_msg(SUBJECT, Fence(2))]);
    {
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert!(
            is_retained_ghost(&dot),
            "precondition: the demote left the retained Ghost in place: {dot:?}"
        );
    }

    // The colliding re-home: source == SHARD (this very node — the CELL-2 shape that resolves
    // the target back onto the source).
    let sent = rig.tick(vec![re_home_msg(
        Fence(3),
        SHARD,
        DirectoryKey::Entity(SUBJECT),
    )]);

    // ONE dot for the entity — flipped in place, never doubled.
    let dots = rig.world.resource::<Dots>();
    assert_eq!(
        dots.0.values().filter(|d| d.entity == SUBJECT).count(),
        1,
        "exactly one dot holds the entity after the colliding re-home: {:?}",
        dots.0,
    );
    assert!(
        !dots.0.contains_key(&SessionId(SUBJECT.0)),
        "no rival dot under the synthetic clientless key — the held session key is reused",
    );
    let dot = dots.0[&SESSION];
    assert_eq!(
        dot.authority,
        Authority::Owned { fence: Fence(3) },
        "the held ghost re-owns at the exact CAS fence"
    );
    assert_eq!(dot.entity_fence, Fence(3));
    assert!(dot.granted & !dot.departing & !dot.adopting);
    assert!(
        !is_retained_ghost(&dot),
        "the flipped dot is the live owner — the expiry fan has no retained ghost to evict"
    );
    assert_eq!(
        (dot.account, dot.gateway),
        (AccountId(1), GATEWAY),
        "the flip keeps the client linkage the ghost retained (a fresh build would be clientless)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().re_home_flipped,
        1,
        "the collision shape is counted apart from the fresh-build adopts"
    );
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 1);
    // The saga still gets its landing proof, and no self-ghost is registered (source == self).
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
    assert_eq!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .get(&SUBJECT),
        None,
        "a source==dest re-home registers NO self-ghost neighbor",
    );
}

#[test]
fn on_re_home_no_ops_for_a_non_entity_subject_or_a_target_without_its_realm() {
    // re_home_apply BAILS on a non-Entity subject; the dispatch BAILS (no adopt) when the target does
    // not hold its realm — both counted degrade-never-panic no-ops.
    let mut rig = Rig::new();
    rig.grant_realm();
    let sent = rig.tick(vec![re_home_msg(
        Fence(2),
        NodeId(99),
        DirectoryKey::Realm(config().realm),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().re_home_no_entity, 1);
    assert_eq!(rig.world.resource::<StubStats>().re_home_adopted, 0);
    // AND IT ACKS NOTHING. The ack is the claim "the entity is here now", and nothing landed: there
    // was no entity in the command to land. This assertion used to read the other way, on the
    // reasoning that acking unconditionally "never wedges the saga" — but the saga is not the thing
    // being protected. An ack for an adopt that refused makes the orchestrator commit authority to a
    // shard that holds nothing, and the source, told the hand-off succeeded, lets go: the entity then
    // exists nowhere at all, which is strictly worse than a stalled saga. Unacked, the saga times out
    // and aborts, and whoever held the entity still holds it.
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "a re-home that adopted NOTHING must not claim it landed: {sent:?}"
    );

    // A re-home delivered while the target does NOT hold its realm (no grant_realm) → no adopt.
    let mut rig2 = Rig::new();
    let _ = rig2.tick(vec![re_home_msg(
        Fence(2),
        NodeId(99),
        DirectoryKey::Entity(SUBJECT),
    )]);
    assert_eq!(rig2.world.resource::<StubStats>().re_home_without_realm, 1);
    assert_eq!(rig2.world.resource::<StubStats>().re_home_adopted, 0);
}

/// Slice F: the dest-side pass STREAMS NOTHING — no pose ever crosses back to the source (the
/// Delta feed is dead; the registration only drives the band-exit Despawn). A registration with
/// no Owned dot is still a counted skip.
#[test]
fn the_dest_sweep_streams_no_poses_and_skips_unowned_registrations() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let source = NodeId(99);
    let _ = rig.tick(vec![promote_msg(Fence(2), source)]); // Owned + registered
    // The next ticks send the source NOTHING while the entity stays in-band: no Delta exists
    // to send, and Despawn waits for band-exit.
    rig.set_local_tick(6);
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).is_empty(),
        "the sweep streams no poses to the ghost host: {sent:?}"
    );
    // A registration whose entity is NOT owned here (no dot) is a counted no-op.
    rig.world
        .resource_mut::<GhostColliderRegistration>()
        .0
        .insert(
            EntityId(0xABCD),
            GhostNeighbor {
                source,
                anchor: LatticePos::ORIGIN,
            },
        );
    let skipped_before = rig.world.resource::<StubStats>().ghost_feed_skipped;
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_feed_skipped,
        skipped_before + 1,
        "a registration with no Owned dot is skipped"
    );
}

#[test]
fn the_dest_feed_pass_skips_a_ghost_dot_registration() {
    // The simulates()-gate negative arm: a registration whose entity dot is a Ghost (not Owned)
    // is skipped — only an OWNED entity's live pose is fed.
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2)); // a granted Ghost dot for `entity`
    rig.world
        .resource_mut::<GhostColliderRegistration>()
        .0
        .insert(
            entity,
            GhostNeighbor {
                source: NodeId(88),
                anchor: LatticePos::ORIGIN,
            },
        );
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, NodeId(88)).is_empty(),
        "no feed for a Ghost (non-Owned) dot: {sent:?}"
    );
    assert!(rig.world.resource::<StubStats>().ghost_feed_skipped >= 1);
}

/// Slice F, THE CORE: the pose feed is dead and the take-over proof is pose-free. A tombstoned
/// `Spawn`/`Delta` frame is a counted no-op whose pose NEVER lands (the §4u poison is
/// structurally impossible — its writer is deleted); `SpawnV2` closes the source hold at the
/// exact fence, stops the retained ghost's emit, and evicts the bystanders' figures (the remove
/// message, retimed to hold closure); a stale-fence proof closes nothing; Despawn still tears
/// the dot out; a malformed body is counted, never mis-applied.
#[test]
fn the_take_over_proof_closes_the_hold_stops_the_emit_and_evicts() {
    const BYSTANDER: SessionId = SessionId(0xBB);
    let mut rig = Rig::new();
    let entity = make_retained_ghost(&mut rig, Fence(2)); // retained source Ghost{Fence(2)}
    let demote_pose = rig.world.resource::<Dots>().0[&SESSION].pose;
    insert_owned_dot(&mut rig, BYSTANDER, player(9), DVec3::new(1.0, 0.0, 0.0));
    // The demote opened the SOURCE hold, so the retained ghost EMITS its own-frame demote pose
    // (the fill), alongside the bystander.
    assert!(
        rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "DEBUG: the demote opened the Source hold"
    );
    let sent = rig.tick(vec![]);
    let emitted: Vec<EntityId> = entity_rows_to_gateway(&sent);
    assert!(
        emitted.contains(&entity),
        "the retained ghost fills the hand-off window: {emitted:?}"
    );

    // A TOMBSTONED Spawn (the old pose-carrying proof) is a counted no-op: the pose does not
    // land and the hold does not close.
    let foreign = StampedPose::at_rest(
        FrameRef::PlanetCentered { planet_seed: 42 },
        DVec3::new(1.0, 2.0, 3.0),
        UniverseTick(2),
    );
    let undec_before = rig.world.resource::<StubStats>().undecodable;
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Spawn {
        entity,
        pose: foreign,
        source_fence: Fence(2),
        since_tick: vd_core::TickId(0),
    })]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].pose,
        demote_pose,
        "a tombstoned Spawn's pose NEVER lands — the poison's writer is deleted"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().undecodable,
        undec_before + 1
    );
    // ...and so is a tombstoned Delta.
    let _ = rig.tick(vec![ghost_delta(entity, foreign, Fence(2), 1)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].pose,
        demote_pose,
        "a tombstoned Delta's pose NEVER lands"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().undecodable,
        undec_before + 2
    );
    assert!(
        rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "a tombstoned frame closes no hold"
    );

    // A STALE-fence proof (a replayed take-over from a superseded crossing) closes nothing.
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(1),
    })]);
    assert!(
        rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "a stale-fence proof closes no hold"
    );

    // THE PROOF at the exact fence: the hold closes, the removal fans to the bystander's
    // gateway at the shard's exact tick, and the ghost STOPS emitting — the leaver vanishes at
    // hold closure, never freezing at the boundary.
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(
        !rig.world
            .resource::<HandoffHolds>()
            .0
            .contains_key(&(entity, HoldRole::Source)),
        "the exact-fence proof closes the hold"
    );
    assert_eq!(
        entity_removals(&sent),
        vec![(GATEWAY, entity, UniverseTick(100))],
        "the bystanders' eviction fans at hold closure"
    );
    let emitted: Vec<EntityId> = entity_rows_to_gateway(&sent);
    assert!(
        !emitted.contains(&entity),
        "the retained ghost stopped emitting at hold closure: {emitted:?}"
    );
    assert!(
        rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "the retained DOT stays (the return-crossing target) until band-exit"
    );

    // A REPLAYED proof after the close is a clean no-op (no hold to close, no second eviction).
    let sent = rig.tick(vec![ghost_lifecycle(GhostFlow::SpawnV2 {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(entity_removals(&sent).is_empty(), "a replay evicts nobody");

    // Despawn still TEARS the dot out at band-exit (idempotent; no-host counted).
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity,
        source_fence: Fence(2),
    })]);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "Despawn removes the retained ghost dot (the lifecycle ends)"
    );
    assert_eq!(rig.world.resource::<StubStats>().ghost_despawns, 1);
    let no_host_before = rig.world.resource::<StubStats>().ghost_despawn_no_host;
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity: EntityId(0x12345),
        source_fence: Fence(2),
    })]);
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_despawn_no_host,
        no_host_before + 1,
        "a Despawn for an unhosted entity is a counted no-op"
    );

    // A malformed ghost body is counted undecodable, never mis-applied.
    let undec_before = rig.world.resource::<StubStats>().undecodable;
    let _ = rig.tick(vec![Inbound::Wire {
        from: DEST_OWNER,
        class: MsgClass::GhostDelta,
        bytes: crate::io::bytes(vec![0xFF, 0xFF, 0xFF]),
    }]);
    assert_eq!(
        rig.world.resource::<StubStats>().undecodable,
        undec_before + 1,
        "a malformed ghost body is counted undecodable"
    );
}

#[test]
fn the_dest_feed_despawns_on_band_exit_and_deregisters() {
    // 1d.5b.3c → slice F: the dest (owner) drives the source-ghost lifecycle END. While the
    // owned entity is IN the overlap band (anchored at its crossing) the sweep sends NOTHING
    // (the pose feed is dead); once it walks PAST the band's destroy edge the dest emits
    // GhostFlow::Despawn (reliable) + DEREGISTERS.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let source = NodeId(99);
    // Owned + registered; the anchor is the crossed pose position (the boundary it entered through).
    let _ = rig.tick(vec![promote_msg(Fence(2), source)]);

    // IN-BAND (the dot is at the anchor, distance 0): nothing streams and the registration is
    // KEPT — the false arm of the band-exit decision.
    rig.set_local_tick(6);
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).is_empty(),
        "in-band: the sweep streams nothing (no Delta exists; Despawn waits for band-exit): {sent:?}"
    );
    assert!(
        rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .contains_key(&SUBJECT),
        "in-band: the registration is kept"
    );
    let exits_before = rig.world.resource::<StubStats>().ghost_band_exits;

    // BAND-EXIT: move the owned dot well past the destroy edge from the crossing anchor. The band
    // is `for_motion(move_speed*dt)` = `for_motion(0.1)`, destroy_above = 20*0.1 = 2.0 m; +3 m exits.
    let exit_pos = fm(crossing_pose().pos) + DVec3::new(3.0, 0.0, 0.0);
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION)
        .expect("the owned dot")
        .pose
        .pos = LatticePos::from_metres(exit_pos, vd_core::pose::Tier::Fine);
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).contains(&InterShardFlow::Ghost(GhostFlow::Despawn {
            entity: SUBJECT,
            source_fence: Fence(2),
        })),
        "band-exit: the dest Despawns the ghost on the reliable carrier: {sent:?}"
    );
    assert!(
        !rig.world
            .resource::<GhostColliderRegistration>()
            .0
            .contains_key(&SUBJECT),
        "band-exit: the feed is deregistered"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_band_exits,
        exits_before + 1
    );

    // ...and the feed truly STOPS: a further tick sends nothing to the source (no Delta, no re-Despawn).
    let sent = rig.tick(vec![]);
    assert!(
        flows_to(&sent, source).is_empty(),
        "after deregistration the feed is silent: {sent:?}"
    );
}

#[test]
fn a_band_exit_despawn_refuses_to_remove_a_reowned_owned_dot() {
    // 1d.5b.3c structural refusal — the shard-LOCAL stand-in for the orchestrator in-transfer gate
    // (a `vd-sim` shard cannot see the live-saga set). A stale Despawn for an entity the source has
    // since RE-OWNED removes nothing: only a retained Ghost is torn down, never a live `Owned` dot.
    // Covers `remove_retained_ghost`'s `matches!(Ghost{..})` FALSE arm + the `no_host` counter.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.attach(); // SESSION's dot is granted + Owned (a re-acquisition would land here)
    let entity = rig.world.resource::<Dots>().0[&SESSION].entity;
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "precondition: the dot is Owned"
    );

    let despawns_before = rig.world.resource::<StubStats>().ghost_despawns;
    let _ = rig.tick(vec![ghost_lifecycle(GhostFlow::Despawn {
        entity,
        source_fence: Fence(1),
    })]);
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the re-owned Owned dot is structurally refused (kept, still simulating)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_despawn_no_host,
        1,
        "the stale Despawn is a counted no-op"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().ghost_despawns,
        despawns_before,
        "...and tears nothing down"
    );
}

#[test]
fn a_retained_ghost_self_emits_but_an_unfed_genesis_ghost_does_not() {
    // 1d.5b.3b → slice F: a RETAINED source Ghost SELF-EMITS its own-frame demote pose while
    // its Source HOLD is open (the demote→take-over fill); a pre-promote DEST-adopt Ghost
    // (GENESIS, no real pose) emits NOTHING.
    // (a) retained source ghost, hold open (make_retained_ghost arms the budget) → emits.
    let mut rig = Rig::new();
    let _entity = make_retained_ghost(&mut rig, Fence(2));
    let sent = rig.tick(vec![]);
    assert_eq!(
        decode_frames(&sent).len(),
        1,
        "the retained source Ghost self-emits its last-Owned pose (no vanish): {sent:?}"
    );

    // (b) a pre-promote DEST-adopt Ghost (GENESIS) → emits nothing.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let sent = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost{GENESIS}, no pose
    assert_eq!(
        decode_frames(&sent).len(),
        0,
        "an unfed GENESIS dest-adopt Ghost renders nothing until the Promote"
    );
}

#[test]
fn emit_frames_renders_an_emitting_dot_and_filters_a_silent_one_in_the_same_tick() {
    // Covers the entity-collect filter's FALSE arm: a tick with BOTH a retained source Ghost
    // (emits) AND a pre-grant GENESIS adopt Ghost (silent). emit_frames does NOT early-return
    // (the emitter makes gateways non-empty) AND the entity-collect filter EXCLUDES the silent
    // dot — exactly the emitting one renders.
    let mut rig = Rig::new();
    let emitter = make_retained_ghost(&mut rig, Fence(2)); // SESSION: retained Ghost, emits
    let _ = rig.tick(vec![open_input_slot(SessionId(0xBB), GATEWAY, 5)]); // a pre-grant GENESIS adopt Ghost, silent
    let sent = rig.tick(vec![]);
    let entities: Vec<EntityId> = decode_frames(&sent)
        .into_iter()
        .flat_map(|f| f.entities)
        .map(|e| e.entity)
        .collect();
    assert_eq!(
        entities,
        vec![emitter],
        "only the emitting retained Ghost renders; the silent GENESIS adopt Ghost is filtered out"
    );
}

#[test]
fn the_saga_promote_on_an_unheld_or_non_entity_subject_is_a_counted_noop_but_acks() {
    // on_saga_promote's no-dot arms: a Promote for an entity not held here, and for a non-Entity
    // (Realm) subject, each flip nothing (counted `promote_no_dot`) but STILL ack PromoteAck.
    let mut rig = Rig::new();
    rig.grant_realm();
    // (a) unheld Entity subject.
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]); // SUBJECT not held here
    assert_eq!(rig.world.resource::<StubStats>().promote_no_dot, 1);
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
    // (b) non-Entity (Realm) subject.
    let realm_promote = InterShardFlow::Promote(PromoteCmd {
        transfer: TransferId(8),
        subject: DirectoryKey::Realm(RealmId::System(9)),
        new_fence: Fence(2),
        step_id: PROMOTE_STEP,
        source: NodeId(99),
    });
    let sent = rig.tick(vec![wire_msg(ORCH, MsgClass::Saga, &realm_promote)]);
    assert_eq!(rig.world.resource::<StubStats>().promote_no_dot, 2);
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(8)
        }
    ));
}

#[test]
fn the_saga_promote_before_the_crossing_lands_defers_the_flip_but_acks() {
    // pose-before-promote: a Promote arriving BEFORE the crossing journaled does NOT flip (no
    // poseless origin frame); counted `promote_before_crossing`, still acks. The dot stays Ghost.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost, NO crossing yet
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the Promote does NOT flip a dot whose crossing pose has not landed (stays Ghost)"
    );
    assert_eq!(rig.world.resource::<StubStats>().promote_before_crossing, 1);
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn a_deferred_promote_flips_when_re_driven_after_the_crossing_lands() {
    // FREEZE FIX (the return-crossing total-freeze): a Promote arriving BEFORE its crossing pose DEFERS
    // WITHOUT journaling PROMOTE_STEP, so when the crossing later lands and the saga RE-DRIVES the Promote,
    // the flip completes (Ghost→Owned). The old code journaled PROMOTE_STEP on the deferred delivery, so the
    // re-drive hit AlreadyApplied and the flip NEVER happened — the entity stayed a silent non-emitting
    // Ghost (no feed, input dropped as PendingAuthority), the exact freeze a RETURN to a re-adopting shard
    // triggers when the ordered Promote outruns the crossing pose.
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]); // granted Ghost, NO crossing yet
    // Promote arrives BEFORE the crossing ⇒ DEFER (stays Ghost; PROMOTE_STEP NOT journaled).
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert!(
        !rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "deferred: stays Ghost"
    );
    assert_eq!(rig.world.resource::<StubStats>().promotes_confirmed, 0);
    // The crossing pose LANDS (STUB_CROSSING_STEP journaled for the transfer).
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    // The saga RE-DRIVES the Promote (a redelivery of the same step). Pre-fix this hit AlreadyApplied and
    // never re-ran the flip; with the fix PROMOTE_STEP was withheld on the defer, so the re-run COMPLETES it.
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert!(
        rig.world.resource::<Dots>().0[&SESSION]
            .authority
            .simulates(),
        "the re-driven Promote flips Ghost→Owned once the crossing has landed (no permanent freeze)"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().promotes_confirmed,
        1,
        "exactly one confirmed flip"
    );
    // The flip announces the dest read sub (so the client's feed re-opens) and acks.
    assert!(saga_ack_to_orch(
        &sent,
        TransferControlAck::PromoteAck {
            transfer: TransferId(7)
        }
    ));
}

#[test]
fn the_saga_promote_without_a_realm_is_a_counted_noop_never_a_panic() {
    // The realm-owner guard's None arm (1d.5b.3b audit hardening): a Promote arriving while this
    // shard does NOT hold its realm is DROPPED as a counted no-op (DEGRADE, never panic), no ack
    // — the saga re-drives. Mirrors every sibling handler's degrade-not-crash discipline.
    let mut rig = Rig::new(); // NO grant_realm → the shard holds no realm (authority.0 == None)
    let sent = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]);
    assert_eq!(rig.world.resource::<StubStats>().promote_without_realm, 1);
    assert!(
        !saga_ack_to_orch(
            &sent,
            TransferControlAck::PromoteAck {
                transfer: TransferId(7)
            }
        ),
        "a realm-less Promote is dropped (no ack) — the saga re-drives: {sent:?}"
    );
}

#[test]
fn the_dest_applies_post_marker_input_and_rejects_replays_at_the_marker() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 10)]); // watermark = 10
    // marker+1, marker+2 apply in order; a seq <= marker is a counted DuplicateSeq.
    let _ = rig.tick(vec![
        input_for(SESSION, 11, GATEWAY),
        input_for(SESSION, 12, GATEWAY),
        input_for(SESSION, 9, GATEWAY),
    ]);
    let log = rig.world.resource::<InputLog>();
    assert_eq!(log.applied(), vec![(SESSION, 11), (SESSION, 12)]);
    assert_eq!(
        log.discarded(),
        vec![(SESSION, Some(9), DiscardReason::DuplicateSeq)],
        "a seq <= marker was already applied at the source"
    );
}

/// Stage B4 (§4u refutation 8): an input-idle durable pose must follow the clock. The scan and
/// the flush resolve MOVING-child placements at the POSE's stamp, so a stamp frozen at the last
/// input measured a parked occupant against a frozen world — a planet could sweep through a
/// parked ship with no crossing ever firing. The re-stamp is position-preserving (`Frozen`
/// continuity: stopped means stopped).
#[test]
fn an_input_idle_dot_is_restamped_to_the_current_tick_each_tick() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    let _ = rig.tick(vec![adopted_head(Fence(2))]);
    let _ = rig.tick(vec![crossing_msg(TransferId(7), Fence(2), crossing_pose())]);
    let _ = rig.tick(vec![promote_msg(Fence(2), NodeId(99))]); // Owned — the dot simulates
    let before = rig.world.resource::<Dots>().0[&SESSION].pose;
    rig.world.resource_mut::<ClockSample>().universe_tick =
        UniverseTick(before.universe_tick.0 + 50);
    let _ = rig.tick(vec![]);
    let after = rig.world.resource::<Dots>().0[&SESSION].pose;
    assert_eq!(
        after.universe_tick,
        UniverseTick(before.universe_tick.0 + 50),
        "the idle stamp follows the shard clock",
    );
    assert_eq!(
        after.pos, before.pos,
        "the re-stamp never moves a stopped player"
    );
    assert_eq!(
        after.frame, before.frame,
        "the re-stamp never touches the frame"
    );
}

#[test]
fn open_input_slot_before_the_realm_lease_is_deferred_and_counted() {
    let mut rig = Rig::new(); // NO grant_realm
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    assert!(
        !rig.world.resource::<Dots>().0.contains_key(&SESSION),
        "no slot yet"
    );
    assert_eq!(rig.world.resource::<StubStats>().input_slots_deferred, 1);
    // Stage B2: the early slot is BUFFERED, not dropped — the lease affirm drains it through the
    // identical adopt path, so the adopt completes LATE instead of NEVER (the fence-7 strand,
    // rehome_one_mechanism §4v fact 3). The drained dot is the ordinary adopt-Ghost: input armed,
    // watermark seeded, awaiting its HeadRead grant and the saga Promote.
    rig.grant_realm();
    let stats = rig.world.resource::<StubStats>();
    assert_eq!(
        stats.input_slots_drained, 1,
        "the buffered slot drained at the lease affirm"
    );
    let dot = rig
        .world
        .resource::<Dots>()
        .0
        .get(&SESSION)
        .copied()
        .expect("the lease affirm minted the adopt dot from the buffered slot");
    assert!(
        dot.adopting,
        "the drained slot is a transfer-destination adopt"
    );
    assert!(
        dot.input_active,
        "the drained slot armed input from its owning gateway"
    );
    assert_eq!(
        dot.last_applied_seq,
        Some(5),
        "the drained slot seeded the resume watermark it was buffered with",
    );
}

#[test]
fn open_input_slot_max_merges_the_watermark_and_guards_a_granted_dot() {
    let mut rig = Rig::new();
    rig.grant_realm();
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 20)]);
    // A re-sent slot with a LOWER watermark never lowers it (max-merge).
    let _ = rig.tick(vec![open_input_slot(SESSION, GATEWAY, 5)]);
    assert_eq!(
        rig.world.resource::<Dots>().0[&SESSION].last_applied_seq,
        Some(20)
    );
    // A slot from a NON-owning gateway does not touch the dot (security guard false arm).
    let other_gateway = NodeId(999);
    let _ = rig.tick(vec![open_input_slot(SESSION, other_gateway, 100)]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(
        dot.last_applied_seq,
        Some(20),
        "a foreign gateway cannot move the watermark"
    );
    assert_eq!(dot.gateway, GATEWAY, "ownership unchanged");
}

#[test]
fn open_input_slot_bumps_the_session_fence_then_is_inert_on_a_granted_dot() {
    let mut rig = Rig::new();
    rig.grant_realm();
    // Mint the provisional slot at fence 1, watermark 5.
    let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 5, Fence(1))]);
    // A slot at a HIGHER fence (as 1d/P3 will re-issue under a fresher realm lease) bumps
    // the session fence and MAX-merges the watermark.
    let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 7, Fence(4))]);
    {
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(dot.session_fence, Fence(4), "bumped to the fresher lease");
        assert_eq!(dot.last_applied_seq, Some(7), "watermark advanced");
    }
    // A STALE slot (fence below the dot's session fence — a replay or partitioned old
    // gateway) is dropped + counted, never re-arming input (the day-one stale-gateway rule).
    let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 999, Fence(1))]);
    {
        let dot = rig.world.resource::<Dots>().0[&SESSION];
        assert_eq!(
            dot.session_fence,
            Fence(4),
            "stale slot does not touch the fence"
        );
        assert_eq!(
            dot.last_applied_seq,
            Some(7),
            "stale slot does not move the watermark"
        );
        assert_eq!(rig.world.resource::<StubStats>().input_slots_stale, 1);
    }
    // Once the dot is GRANTED (the 1d promotion), a stray OpenInputSlot is inert —
    // the granted entity owns its own input watermark (the `!granted` guard false arm).
    rig.world
        .resource_mut::<Dots>()
        .0
        .get_mut(&SESSION)
        .expect("the provisional dot was minted above")
        .granted = true;
    let _ = rig.tick(vec![open_input_slot_f(SESSION, GATEWAY, 999, Fence(9))]);
    let dot = rig.world.resource::<Dots>().0[&SESSION];
    assert_eq!(dot.session_fence, Fence(4), "granted dot's fence untouched");
    assert_eq!(
        dot.last_applied_seq,
        Some(7),
        "granted dot's watermark untouched"
    );
}
