//! The whole D-7 transient/debris batched-crossing machinery: status tiers, re-advance of held items, emit/adopt/release/promote/complete, discard/abandon poisoning, re-drive and re-solicit, and the inbound dispatch arms. Also carries the RLM Step-5a capability-inertness gate (504) and five lease/self-fence tests that observe the self-fence THROUGH transient loss (565, 580, 1206, 1247, 1314). Local helper decode_flows (492) travels.
//!
//! Split out of the single `stub::tests` module in slice S10 (the file had reached 16,139 lines).
//! The assertions are VERBATIM; only their module path changed. Every fixture they use still lives
//! in the parent, which is what `use super::*` reaches.

use super::*;

#[test]
fn shard_profile_swap_is_capability_inert_for_every_profile_kind() {
    // RLM Step 5a ACCEPTANCE GATE. The shard boot swaps `NodeKind::StubShard` →
    // `NodeKind::Shard(profile_for(coord.profile_kind()))`. That swap MUST be CAPABILITY-INERT at P1–P3:
    // no system reads the carried `ShardProfile` caps to decide behavior yet, so a shard carrying ANY
    // profile emits BYTE-IDENTICAL authored frames + grants to the zero-cap StubShard shard, at the same
    // synced tick, with identical inputs. PROVEN here (not asserted) over EVERY `ProfileKind` — incl.
    // `Galaxy` (the one that carries `signal_relay`, which will uncorner P9 Signals). A future
    // `match kind` / capability gate that changes authored output would break this guard.
    use vd_core::taxonomy::ProfileKind;
    let baseline = Rig::with_config_and_kind(config(), NodeKind::StubShard).tick(vec![]);
    assert!(
        !baseline.is_empty(),
        "a synced shard authors + self-grants on an empty tick — a non-vacuous inertness baseline"
    );
    for k in ProfileKind::ALL {
        let profile =
            crate::capability::profile_for(k).expect("every ProfileKind yields a profile");
        let out = Rig::with_config_and_kind(config(), NodeKind::Shard(profile)).tick(vec![]);
        assert_eq!(
            out, baseline,
            "Shard({k:?}) authored output diverged from StubShard — the profile swap is NOT inert"
        );
    }
}

#[test]
fn transient_status_is_held_excludes_arriving_and_departing() {
    // Held + the pre-emit Crossing are authoritatively held (counted); Arriving (dest mid-flight)
    // AND Departing (source released, retained) are the two UNCOUNTED tiers (D-7b).
    assert!(TransientStatus::Held { outbound: None }.is_held());
    assert!(
        TransientStatus::Held {
            outbound: Some(TransferId(1))
        }
        .is_held()
    );
    assert!(
        TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(2),
            batch: TransferId(1),
            to_parent: None,
        }
        .is_held()
    );
    assert!(
        !TransientStatus::Arriving {
            batch: TransferId(1)
        }
        .is_held()
    );
    assert!(
        !TransientStatus::Departing {
            batch: TransferId(1)
        }
        .is_held()
    );
}

#[test]
fn realm_lease_lapsed_requires_held_armed_channel_and_stale() {
    // The shared proactive self-fence predicate (HR5 branchless shim). Self-fence iff ALL hold: the
    // realm is HELD, the timer is ARMED (grace != 0), the confirmation CHANNEL exists (recheck != 0),
    // and the last confirmation is STALER than the grace. Each guard alone vetoes; the boundary
    // (`== grace`) is NOT lapsed (a strict `>`), so the holder keeps authority for the whole window.
    use crate::directory::lease_self_fence_due;
    let now = TickId(100);
    assert!(lease_self_fence_due(true, 5, 2, now, TickId(94))); // 6 > 5 ⇒ lapsed
    assert!(!lease_self_fence_due(false, 5, 2, now, TickId(94))); // not held
    assert!(!lease_self_fence_due(true, 0, 2, now, TickId(94))); // timer disarmed (pre-D-3 default)
    assert!(!lease_self_fence_due(true, 5, 0, now, TickId(94))); // no confirmation channel
    assert!(!lease_self_fence_due(true, 5, 2, now, TickId(95))); // 100-95 = 5 == grace ⇒ within, holds
}

#[test]
fn a_partitioned_holder_proactively_self_fences_then_re_arms_on_re_grant() {
    // D-3 Slice 5: the realm is granted (a round-trip confirmation at tick 1), then NO further
    // realm-head reply arrives (a partition from the orchestrator). Once `local_tick - confirmed`
    // exceeds the grace, the holder hard-stops its OWN authority and drops its held transients —
    // before the orchestrator's reassign window opens — so a stale owner can never affect clients.
    // A later re-grant re-arms the deadline, so a re-granted realm never inherits the stale one.
    let mut rig = Rig::with_config(StubConfig {
        realm_recheck_interval: 2, // the round-trip confirmation channel is active
        self_fence_grace_ticks: 5, // rig-local; ttl < grace <= ttl + max is validated orch-side
        ..config()
    });
    rig.grant_realm(); // confirm at local_tick 1 ⇒ RealmConfirmedAt(1), authority Some(Fence(1))
    let debris = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        debris,
        Transient {
            pose: StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(98)),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );

    // Within the grace (local 6 - confirmed 1 = 5, NOT > 5): the holder KEEPS authority.
    rig.set_local_tick(6);
    let _ = rig.tick(vec![]);
    assert_eq!(rig.world.resource::<RealmAuthority>().0, Some(Fence(1)));
    assert_eq!(
        rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
        0
    );
    assert!(
        rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&debris)
    );

    // Past the grace (local 7 - confirmed 1 = 6 > 5): SELF-FENCE — authority dropped, transient lost.
    rig.set_local_tick(7);
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        None,
        "authority hard-stopped"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
        1
    );
    assert_eq!(
        rig.world.resource::<StubStats>().transients_dropped,
        1,
        "the held transient anchored to the lost lease is a counted loss"
    );
    assert!(rig.world.resource::<OwnedTransients>().0.is_empty());

    // A re-grant at local 7 re-confirms (RealmConfirmedAt ⇒ 7); at local 11 (11-7 = 4 <= 5) the
    // holder is STILL authoritative — proof the re-granted realm did NOT inherit the stale deadline
    // (had `confirmed` stayed 1, 11-1 = 10 > 5 would have re-fenced it immediately).
    rig.grant_realm();
    rig.set_local_tick(11);
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<RealmAuthority>().0,
        Some(Fence(1)),
        "re-grant re-armed the timer"
    );
    assert_eq!(
        rig.world.resource::<StubStats>().realm_self_fenced_lapsed,
        1,
        "no second self-fence"
    );
}

#[test]
fn readvance_advances_held_transients_and_skips_the_uncounted_tiers() {
    // D-7b: an AUTHORITATIVELY-held debris re-advances by its closed-form ballistic motion each
    // tick (vel·dt); the uncounted Arriving/Departing tiers are SKIPPED (not rendered, caught up
    // at promote). The Rig clock is universe_tick=100; a pose stamped at tick 98 advances dt =
    // (100-98)·tick_dt_s(0.05) = 0.1s.
    let mut rig = Rig::new();
    rig.grant_realm();
    let held = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let arriving = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    let departing = EntityId::pack(EntityKind::Debris, 1, 7, 3);
    let moving = StampedPose {
        vel: DVec3::new(10.0, 0.0, 0.0),
        ..StampedPose::at_rest(config().frame, DVec3::ZERO, UniverseTick(98))
    };
    {
        let mut owned = rig.world.resource_mut::<OwnedTransients>();
        owned.0.insert(
            held,
            Transient {
                pose: moving,
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: LatticePos::ORIGIN,
            },
        );
        owned.0.insert(
            arriving,
            Transient {
                pose: moving,
                anchor_fence: Fence(1),
                status: TransientStatus::Arriving {
                    batch: TransferId(1),
                },
                prev_offset: LatticePos::ORIGIN,
            },
        );
        owned.0.insert(
            departing,
            Transient {
                pose: moving,
                anchor_fence: Fence(1),
                status: TransientStatus::Departing {
                    batch: TransferId(1),
                },
                prev_offset: LatticePos::ORIGIN,
            },
        );
    }
    let _ = rig.tick(vec![]);
    let owned = rig.world.resource::<OwnedTransients>();
    assert_eq!(
        fm(owned.0[&held].pose.pos),
        DVec3::new(1.0, 0.0, 0.0),
        "the held debris advanced by vel·dt (10 · 0.1)"
    );
    assert_eq!(
        owned.0[&held].pose.universe_tick,
        UniverseTick(100),
        "re-stamped to now"
    );
    assert_eq!(
        fm(owned.0[&arriving].pose.pos),
        DVec3::ZERO,
        "the uncounted Arriving tier is NOT advanced"
    );
    assert_eq!(
        fm(owned.0[&departing].pose.pos),
        DVec3::ZERO,
        "the uncounted Departing tier is NOT advanced"
    );
}

#[test]
fn emit_transient_batch_ships_one_envelope_and_marks_outbound() {
    let mut rig = Rig::new();
    rig.grant_realm(); // authority.0 = Some(Fence(1))
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let batch = TransferId(0xB3);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Crossing {
                dest: DEST_NODE,
                to_realm: RealmId::System(8),
                dst_realm_fence: Fence(2),
                batch,
                to_parent: None,
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![]);
    // Exactly ONE TransientBatch envelope to the dest (G-TIER: one per batch), at the dst fence.
    let to_dest: Vec<_> = sent
        .iter()
        .filter(|(to, _, _)| *to == DEST_NODE)
        .map(|(_, _, bytes)| postcard::from_bytes::<InterShardFlow>(bytes).expect("decode"))
        .collect();
    assert_eq!(
        to_dest,
        vec![InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: batch,
            universe_epoch: vd_core::EpochId(1),
            schema_version: TRANSFER_SCHEMA_VERSION,
            fence: Fence(2),
            step_id: TRANSIENT_BATCH_STEP,
            class: DurabilityClass::Transient,
            payload: TransitionPayload::TransientBatch {
                from_realm: config().realm,
                to_realm: RealmId::System(8),
                src_realm_fence: Fence(1),
                dst_realm_fence: Fence(2),
                source_tick: vd_core::TickId(1),
                items: vec![TransientItem {
                    entity,
                    // D-7b: `readvance_transients` ran BEFORE emit (Crossing is_held), re-stamping
                    // the pose to the source's current universe-tick (100); a rest pose's position
                    // is unchanged (vel ZERO), only the stamp moves.
                    //
                    // The pose then ships VERBATIM in THIS shard's own frame ({7}), because this rig
                    // plants no region forest and so `System(8)` is not one of its direct children —
                    // this shard has not been told where that realm is and has nothing to subtract.
                    // It used to ship stamped `{8}`: the frame flipped, the number did not move, and
                    // the receiver then read a source-frame position as if it were its own.
                    pose: StampedPose {
                        universe_tick: UniverseTick(100),
                        ..transient_pose()
                    },
                    state: vec![],
                }],
            },
        })],
    );
    // The emitted item is now Held{outbound} (still authoritative — adopt-before-drop).
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        TransientStatus::Held {
            outbound: Some(batch)
        }
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 1);
}

#[test]
fn a_transient_whose_flush_refuses_returns_to_held_and_ships_nothing() {
    // Stage B1 reaches the TRANSIENT lane too: a crossing item whose re-read pose is still held
    // by this shard's own band (the departure is no longer true) is NOT shipped — it returns to
    // plain `Held`, this shard keeps authority, and the scan re-decides. Shipping anyway would
    // hand a peer a position nobody vouches for; dropping while marked sent would strand it.
    let mut rig = Rig::new();
    rig.grant_realm();
    // A roster whose OWN region still HOLDS the transient's pose (5 m inside a 1000 m shell).
    plant_aoi(&mut rig, vec![root_region(), own_region()]);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
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
    let sent = rig.tick(vec![]);
    assert!(
        !sent.iter().any(|(to, _, _)| *to == DEST_NODE),
        "the refused item ships nothing: {sent:?}"
    );
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        TransientStatus::Held { outbound: None },
        "the item is back to plain Held — this shard is still its authority"
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 0);
}

#[test]
fn emit_transient_batch_ships_nothing_without_a_realm_lease() {
    let mut rig = Rig::new(); // NO grant_realm → authority.0 = None
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let crossing = TransientStatus::Crossing {
        dest: DEST_NODE,
        to_realm: RealmId::System(8),
        dst_realm_fence: Fence(2),
        batch: TransferId(0xB3),
        to_parent: None,
    };
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: crossing,
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![]);
    assert!(
        sent.iter().all(|(to, _, _)| *to != DEST_NODE),
        "a shard without its realm lease ships no transient batch"
    );
    // The Crossing item is UNCHANGED (it retries when the lease arrives).
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&entity].status,
        crossing
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_emitted, 0);
}

#[test]
fn adopt_transient_batch_adopts_arriving_acks_and_dedups() {
    let batch = TransferId(0xB1);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let cfg = config();
    let regions = RealmRegions::default();
    let placements = PlacementLedger::default();
    let mut owned = OwnedTransients::default();
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let items = vec![TransientItem {
        entity,
        pose: transient_pose(),
        state: vec![],
    }];

    // FIRST delivery: adopt as Arriving (uncounted) anchored to the dst fence + ack BatchAdopted.
    // The pose is in the dest realm's OWN frame, so the receiver conversion's verbatim arm runs.
    adopt_transient_batch(
        batch,
        cfg.realm,
        Fence(5),
        items.clone(),
        &cfg,
        &regions,
        &placements,
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(owned.0[&entity].status, TransientStatus::Arriving { batch });
    assert_eq!(owned.0[&entity].anchor_fence, Fence(5));
    assert_eq!(stats.transients_adopted, 1);
    assert_eq!(stats.transient_arrivals_unplaceable, 0);
    assert_eq!(stats.transients_adopt_redelivered, 0);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: batch,
                step_id: TRANSIENT_BATCH_STEP,
            })
        )]
    );

    // REDELIVERY: no re-adopt, re-ack only (at-least-once — the ack may have been lost).
    adopt_transient_batch(
        batch,
        cfg.realm,
        Fence(5),
        items,
        &cfg,
        &regions,
        &placements,
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.transients_adopted, 1, "not re-adopted");
    assert_eq!(stats.transients_adopt_redelivered, 1);
    assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");
}

#[test]
fn adopt_transient_batch_refuses_an_unplaceable_item_counted_still_acks() {
    // THE RECEIVER-SIDE FRAME GUARD on the transient tier (audit :105/:374/:384 — the adopt used
    // to store the wire pose VERBATIM): an item whose pose this shard cannot measure — here a
    // frame it holds no placement book for — is REFUSED + counted, never inserted, while the
    // batch itself still journals + acks (adopt-before-drop proceeds; the loss is per-item and
    // accounted, the Transient class budget discipline).
    let batch = TransferId(0xB4);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 9);
    let cfg = config();
    let regions = RealmRegions::default();
    let placements = PlacementLedger::default();
    let mut owned = OwnedTransients::default();
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();
    let misframed = StampedPose::at_rest(
        FrameRef::SystemSpace { system_seed: 99 }, // a sibling frame — nobody told us its placement
        DVec3::new(3.0, 0.0, 0.0),
        UniverseTick(5),
    );
    adopt_transient_batch(
        batch,
        cfg.realm,
        Fence(5),
        vec![TransientItem {
            entity,
            pose: misframed,
            state: vec![],
        }],
        &cfg,
        &regions,
        &placements,
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert!(
        owned.0.is_empty(),
        "the mis-framed item is NOT adopted — nothing stored verbatim"
    );
    assert_eq!(
        stats.transient_arrivals_unplaceable, 1,
        "the refusal is counted"
    );
    assert_eq!(stats.transients_adopted, 0);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: batch,
                step_id: TRANSIENT_BATCH_STEP,
            })
        )],
        "the batch still acks — the refusal is per-item, the choreography completes"
    );
}

#[test]
fn transient_release_promote_complete_lifecycle_and_dedup() {
    // The full D-7b source/dest handler lifecycle on ONE mixed owned set (the handlers walk by
    // status; source-vs-dest is just which statuses are present in production): RELEASE flips
    // `Held{Some}→Departing` (uncounted), PROMOTE flips `Arriving→Held` (re-anchored), COMPLETE
    // retires `Departing` — each journaled/state idempotent (redelivery = counted no-op + re-ack).
    let this = TransferId(0xB2);
    let other = TransferId(0xB9);
    let held_this = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    let held_other = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    let held_settled = EntityId::pack(EntityKind::Debris, 1, 7, 3);
    let arriving_this = EntityId::pack(EntityKind::Debris, 1, 7, 4);
    let arriving_other = EntityId::pack(EntityKind::Debris, 1, 7, 5);
    let crossing = EntityId::pack(EntityKind::Debris, 1, 7, 6);
    let departing_other = EntityId::pack(EntityKind::Debris, 1, 7, 7);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(2),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    owned.0.insert(
        held_this,
        mk(TransientStatus::Held {
            outbound: Some(this),
        }),
    );
    owned.0.insert(
        held_other,
        mk(TransientStatus::Held {
            outbound: Some(other),
        }),
    );
    owned
        .0
        .insert(held_settled, mk(TransientStatus::Held { outbound: None }));
    owned
        .0
        .insert(arriving_this, mk(TransientStatus::Arriving { batch: this }));
    owned.0.insert(
        arriving_other,
        mk(TransientStatus::Arriving { batch: other }),
    );
    owned.0.insert(
        crossing,
        mk(TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(9),
            batch: other,
            to_parent: None,
        }),
    );
    owned.0.insert(
        departing_other,
        mk(TransientStatus::Departing { batch: other }),
    );
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let cfg = config();
    let mut outbox = OutboundBox::default();

    // RELEASE (SOURCE): `Held{Some(this)}` → uncounted `Departing`; `Held{Some(other)}` and every
    // other status untouched; ack `DropApplied`(RELEASE_STEP) to the orchestrator.
    let rel = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(5),
    };
    on_transient_release(rel, &mut owned, &mut applied, &mut stats, &cfg, &mut outbox);
    assert_eq!(
        owned.0[&held_this].status,
        TransientStatus::Departing { batch: this },
        "the source's Held item for THIS batch is released to the uncounted Departing tier"
    );
    assert_eq!(
        owned.0[&held_other].status,
        TransientStatus::Held {
            outbound: Some(other)
        },
        "a Held item for ANOTHER batch is untouched"
    );
    assert_eq!(
        owned.0[&held_settled].status,
        TransientStatus::Held { outbound: None }
    );
    assert_eq!(stats.transients_handed_off, 1);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: this,
                step_id: TRANSIENT_RELEASE_STEP,
            })
        )]
    );
    // RELEASE REDELIVERY: re-ack, no re-flip.
    on_transient_release(rel, &mut owned, &mut applied, &mut stats, &cfg, &mut outbox);
    assert_eq!(stats.transient_release_noop, 1);
    assert_eq!(stats.transients_handed_off, 1, "not re-released");
    assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");

    // PROMOTE (DEST): `Arriving{this}` → `Held{None}` re-anchored to the commit fence;
    // `Arriving{other}` untouched; ack `DropApplied`(DROP_STEP).
    let promote = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_DROP_STEP,
        fence: Fence(5),
    };
    on_transient_promote(
        promote,
        &mut owned,
        &mut applied,
        &mut stats,
        &cfg,
        &mut outbox,
    );
    assert_eq!(
        owned.0[&arriving_this].status,
        TransientStatus::Held { outbound: None },
        "THIS batch's Arriving is promoted to authoritative Held"
    );
    assert_eq!(
        owned.0[&arriving_this].anchor_fence,
        Fence(5),
        "the promoted item re-anchors to the batch's commit fence"
    );
    assert_eq!(
        owned.0[&arriving_other].status,
        TransientStatus::Arriving { batch: other },
        "another batch's Arriving is untouched"
    );
    assert!(
        owned.0.contains_key(&crossing),
        "a pending Crossing is untouched"
    );
    assert_eq!(stats.transients_promoted, 1);
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: this,
                step_id: TRANSIENT_DROP_STEP,
            })
        )]
    );
    // PROMOTE REDELIVERY: re-ack, no re-flip.
    on_transient_promote(
        promote,
        &mut owned,
        &mut applied,
        &mut stats,
        &cfg,
        &mut outbox,
    );
    assert_eq!(stats.transient_drop_noop, 1);
    assert_eq!(stats.transients_promoted, 1, "not re-promoted");
    assert_eq!(decode_flows(&mut outbox).len(), 1, "re-acked exactly once");

    // COMPLETE (SOURCE): retire THIS batch's `Departing` copy (held_this); a `Departing` for
    // ANOTHER batch (departing_other) is untouched; D-7d: ALWAYS ack DropApplied(COMPLETE_STEP) —
    // the `SourceRetired` signal that drives the saga's BatchHandoff tail to Done.
    let rc = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(5),
    };
    on_release_complete(rc, &mut owned, &mut stats, &cfg, &mut outbox);
    assert!(
        !owned.0.contains_key(&held_this),
        "the retained Departing copy for THIS batch is retired"
    );
    assert_eq!(
        owned.0[&departing_other].status,
        TransientStatus::Departing { batch: other },
        "a Departing copy for ANOTHER batch is untouched"
    );
    assert_eq!(
        decode_flows(&mut outbox),
        vec![(
            ORCH,
            InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: this,
                step_id: TRANSIENT_COMPLETE_STEP,
            })
        )],
        "ReleaseComplete acks the retire-complete so the saga tail reaches Done (SourceRetired)"
    );
    // COMPLETE REDELIVERY: no Departing for THIS batch → counted no-op, but STILL acks
    // (at-least-once — the orchestrator's tombstoned saga absorbs the duplicate).
    on_release_complete(rc, &mut owned, &mut stats, &cfg, &mut outbox);
    assert_eq!(
        stats.transient_release_noop, 2,
        "the redelivered complete is a counted no-op"
    );
    assert_eq!(
        decode_flows(&mut outbox).len(),
        1,
        "the redelivery still acks (at-least-once)"
    );
}

#[test]
fn self_fence_drops_held_transients_as_a_counted_loss() {
    // A realm takeover (the lease now held by someone else) self-fences the shard AND drops its
    // transients (anchored to the now-lost lease) as a counted LOSS — durable dots are retained.
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 1);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let takeover = DirectoryReply::Head {
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
        &InterShardFlow::DirectoryReply(takeover),
    )]);
    assert!(
        rig.world.resource::<OwnedTransients>().0.is_empty(),
        "transients dropped on self-fence"
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_dropped, 1);
    assert!(
        rig.world.resource::<RealmAuthority>().0.is_none(),
        "the shard self-fenced its realm"
    );
}

#[test]
fn self_fence_drops_held_transients_on_realm_revoke() {
    // The revoked arm (the realm record is GONE) also self-fences + drops transients — the second
    // `self_fence_drop_transients` call site (a revoke vs a takeover).
    let mut rig = Rig::new();
    rig.grant_realm();
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 2);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        entity,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let revoked = DirectoryReply::Head {
        key: DirectoryKey::Realm(config().realm),
        record: None,
    };
    let _ = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(revoked),
    )]);
    assert!(rig.world.resource::<OwnedTransients>().0.is_empty());
    assert_eq!(rig.world.resource::<StubStats>().transients_dropped, 1);
    assert!(rig.world.resource::<RealmAuthority>().0.is_none());
}

#[test]
fn transient_status_is_in_handover_excludes_only_settled_held() {
    // D-7b.3: every tier EXCEPT a settled `Held{outbound: None}` is a handover loss if dropped.
    assert!(
        !TransientStatus::Held { outbound: None }.is_in_handover(),
        "a settled Held is a resident eviction, not a handover loss"
    );
    assert!(
        TransientStatus::Held {
            outbound: Some(TransferId(1))
        }
        .is_in_handover()
    );
    assert!(
        TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(2),
            batch: TransferId(1),
            to_parent: None,
        }
        .is_in_handover()
    );
    assert!(
        TransientStatus::Arriving {
            batch: TransferId(1)
        }
        .is_in_handover()
    );
    assert!(
        TransientStatus::Departing {
            batch: TransferId(1)
        }
        .is_in_handover()
    );
}

#[test]
fn self_fence_buckets_handover_loss_per_kind_and_excludes_resident_and_corrupt() {
    // D-7b.3: a realm self-fence buckets ONLY handover-status items into the per-kind
    // handover-loss counter (the budget gate's input), keyed by kind; a settled `Held{None}` is a
    // resident eviction (NOT bucketed); a corrupt kind tag is counted GROSS but NOT bucketed (the
    // `from_tag` Err arm). The gross `transients_dropped` still counts EVERY tier.
    let t = TransferId(0xC0);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(2),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    // Four Debris in handover (one per tier).
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 1, 0),
        mk(TransientStatus::Held { outbound: Some(t) }),
    );
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 2, 0),
        mk(TransientStatus::Crossing {
            dest: DEST_NODE,
            to_realm: RealmId::System(8),
            dst_realm_fence: Fence(2),
            batch: t,
            to_parent: None,
        }),
    );
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 3, 0),
        mk(TransientStatus::Arriving { batch: t }),
    );
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 4, 0),
        mk(TransientStatus::Departing { batch: t }),
    );
    // A Projectile in handover (proves per-kind disentangling).
    owned.0.insert(
        EntityId::pack(EntityKind::Projectile, 1, 5, 0),
        mk(TransientStatus::Held { outbound: Some(t) }),
    );
    // A SETTLED Debris (resident eviction — NOT a handover loss).
    owned.0.insert(
        EntityId::pack(EntityKind::Debris, 1, 6, 0),
        mk(TransientStatus::Held { outbound: None }),
    );
    // A corrupt kind tag (99) in handover — counted gross, NOT bucketed (the from_tag Err arm).
    owned.0.insert(
        EntityId(99u128 << 120),
        mk(TransientStatus::Departing { batch: t }),
    );
    let mut stats = StubStats::default();

    self_fence_drop_transients(&mut owned, &mut stats);
    assert!(
        owned.0.is_empty(),
        "every transient is dropped on self-fence"
    );
    assert_eq!(stats.transients_dropped, 7, "gross counts EVERY tier");
    assert_eq!(
        stats.transients_lost_in_handover.get(&EntityKind::Debris),
        Some(&4),
        "4 Debris in handover bucketed (the settled one + the corrupt one excluded)"
    );
    assert_eq!(
        stats
            .transients_lost_in_handover
            .get(&EntityKind::Projectile),
        Some(&1),
        "the Projectile is bucketed under its OWN kind (per-kind disentangling)"
    );
}

#[test]
fn on_transient_abandon_drops_the_batch_as_accounted_loss_and_is_idempotent() {
    // D-7d dead-DEST resolution: abandon drops THIS batch's retained items — both `Departing{this}`
    // (already released) AND `Held{Some(this)}` (the dest died before release) — bucketing each into
    // the per-kind loss budget + counting departure_cancelled. A `Held{None}` resident, a
    // `Departing{other}` batch, and a `Held{Some(other)}` batch are UNTOUCHED. A corrupt kind tag is
    // removed but NOT bucketed (the from_tag Err arm). A redelivery is a journaled no-op.
    let this = TransferId(0xD7D);
    let other = TransferId(0xBEEF);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(3),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    let dep_this = EntityId::pack(EntityKind::Debris, 1, 1, 0);
    let held_this = EntityId::pack(EntityKind::Debris, 1, 2, 0);
    let resident = EntityId::pack(EntityKind::Debris, 1, 3, 0);
    let dep_other = EntityId::pack(EntityKind::Debris, 1, 4, 0);
    let held_other = EntityId::pack(EntityKind::Debris, 1, 5, 0);
    let corrupt = EntityId(99u128 << 120); // tag 99 → from_tag Err (removed, not bucketed)
    owned
        .0
        .insert(dep_this, mk(TransientStatus::Departing { batch: this }));
    owned.0.insert(
        held_this,
        mk(TransientStatus::Held {
            outbound: Some(this),
        }),
    );
    owned
        .0
        .insert(resident, mk(TransientStatus::Held { outbound: None }));
    owned
        .0
        .insert(dep_other, mk(TransientStatus::Departing { batch: other }));
    owned.0.insert(
        held_other,
        mk(TransientStatus::Held {
            outbound: Some(other),
        }),
    );
    owned
        .0
        .insert(corrupt, mk(TransientStatus::Departing { batch: this }));
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let abandon = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_ABANDON_STEP,
        fence: Fence(3),
    };

    on_transient_abandon(abandon, &mut owned, &mut applied, &mut stats);
    // THIS batch's Departing + Held{Some} + the corrupt one are dropped.
    assert!(!owned.0.contains_key(&dep_this));
    assert!(!owned.0.contains_key(&held_this));
    assert!(!owned.0.contains_key(&corrupt));
    // The resident + the OTHER batch's copies survive (not this batch).
    assert!(owned.0.contains_key(&resident));
    assert!(owned.0.contains_key(&dep_other));
    assert!(owned.0.contains_key(&held_other));
    assert_eq!(
        stats.transients_departure_cancelled, 3,
        "3 items abandoned (2 Debris + 1 corrupt)"
    );
    assert_eq!(
        stats.transients_lost_in_handover.get(&EntityKind::Debris),
        Some(&2),
        "only the 2 Debris are bucketed (the corrupt kind is removed but not attributable)"
    );

    // REDELIVERY: the journal short-circuits — no double-count, the survivors are untouched.
    on_transient_abandon(abandon, &mut owned, &mut applied, &mut stats);
    assert_eq!(
        stats.transients_departure_cancelled, 3,
        "the redelivery did not re-count"
    );
    assert_eq!(
        stats.transient_release_noop, 1,
        "the redelivery is a counted journal no-op"
    );
}

#[test]
fn on_transient_discard_removes_arriving_poisons_adopt_and_is_idempotent() {
    // R-6d3c NEVER-restart resolution (DEST role): the source died in AwaitAdopt pre-adopt, so the
    // discard REMOVES this batch's `Arriving{this}` items as an accounted loss — a decodable Debris
    // is bucketed, a corrupt kind tag is removed but NOT bucketed (the from_tag Err arm). An
    // `Arriving{other}` (batch mismatch), a `Held{None}` resident, and a `Departing{this}` (non-
    // Arriving) are UNTOUCHED — the false arms. A redelivery is a journaled no-op (loss counts once).
    let this = TransferId(0x6D3C);
    let other = TransferId(0xBEEF);
    let mk = |status| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(4),
        status,
        prev_offset: LatticePos::ORIGIN,
    };
    let mut owned = OwnedTransients::default();
    let arr_this = EntityId::pack(EntityKind::Debris, 1, 1, 0);
    let corrupt = EntityId(99u128 << 120); // tag 99 → from_tag Err (removed, not bucketed)
    let arr_other = EntityId::pack(EntityKind::Debris, 1, 2, 0);
    let resident = EntityId::pack(EntityKind::Debris, 1, 3, 0);
    let dep_this = EntityId::pack(EntityKind::Debris, 1, 4, 0);
    owned
        .0
        .insert(arr_this, mk(TransientStatus::Arriving { batch: this }));
    owned
        .0
        .insert(corrupt, mk(TransientStatus::Arriving { batch: this }));
    owned
        .0
        .insert(arr_other, mk(TransientStatus::Arriving { batch: other }));
    owned
        .0
        .insert(resident, mk(TransientStatus::Held { outbound: None }));
    owned
        .0
        .insert(dep_this, mk(TransientStatus::Departing { batch: this }));
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let discard = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_DISCARD_STEP,
        fence: Fence(4),
    };

    on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
    // THIS batch's Arriving items (the decodable + the corrupt) are removed.
    assert!(!owned.0.contains_key(&arr_this));
    assert!(!owned.0.contains_key(&corrupt));
    // The OTHER batch's Arriving, the settled resident, and a Departing item survive (false arms).
    assert!(owned.0.contains_key(&arr_other));
    assert!(owned.0.contains_key(&resident));
    assert!(owned.0.contains_key(&dep_this));
    assert_eq!(
        stats.transients_discarded_source_crash, 2,
        "2 Arriving items discarded (1 Debris + 1 corrupt)"
    );
    assert_eq!(
        stats.transients_lost_in_handover.get(&EntityKind::Debris),
        Some(&1),
        "only the decodable Debris is bucketed (the corrupt kind is removed but not attributable)"
    );

    // REDELIVERY: the journal short-circuits — no double-count, the survivors are untouched.
    on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
    assert_eq!(
        stats.transients_discarded_source_crash, 2,
        "the redelivery did not re-count"
    );
    assert_eq!(
        stats.transient_release_noop, 1,
        "the redelivery is a counted journal no-op"
    );
    assert_eq!(
        owned.0.len(),
        3,
        "the survivors are untouched by the redelivery"
    );
}

#[test]
fn discard_before_adopt_poisons_so_a_late_replay_never_orphans() {
    // THE target interleave (Defect A closed): the discard fires FIRST on an EMPTY owned set (the
    // dest never received the batch — the source died pre-adopt) → it removes nothing but POISONS
    // `(transfer, TRANSIENT_BATCH_STEP)`. A LATE outbox replay of the batch then adopts as
    // `AlreadyApplied` — inserting NOTHING — so no `Arriving` orphan is ever stranded.
    let this = TransferId(0x6D3C);
    let entity = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let mut owned = OwnedTransients::default();
    let mut applied = AppliedSteps::default();
    let mut stats = StubStats::default();
    let mut outbox = OutboundBox::default();

    let discard = TransientHandoff {
        transfer: this,
        step_id: TRANSIENT_DISCARD_STEP,
        fence: Fence(4),
    };
    on_transient_discard(discard, &mut owned, &mut applied, &mut stats);
    assert_eq!(
        stats.transients_discarded_source_crash, 0,
        "nothing to remove on an empty dest — the discard only poisons the adopt"
    );

    // The LATE batch replay: the adopt hits its `AlreadyApplied` arm (poisoned) — no insert.
    let cfg = config();
    adopt_transient_batch(
        this,
        cfg.realm,
        Fence(5),
        vec![TransientItem {
            entity,
            pose: transient_pose(),
            state: vec![],
        }],
        &cfg,
        &RealmRegions::default(),
        &PlacementLedger::default(),
        &mut owned,
        &mut applied,
        &mut stats,
        &mut outbox,
    );
    assert!(
        owned.0.is_empty(),
        "the poisoned adopt inserts nothing — no Arriving orphan"
    );
    assert_eq!(
        stats.transients_adopt_redelivered, 1,
        "the adopt short-circuited on the poisoned step"
    );
    assert_eq!(stats.transients_adopted, 0, "no item was ever adopted");
}

#[test]
fn transient_discard_flows_through_the_inbound_dispatch() {
    // Covers the `on_directory_reply` DISPATCH arm for `TransientDiscard` (the direct-call tests
    // above cover the handler itself): a DEST holding an `Arriving` copy receives the discard, drops
    // it as an accounted loss, and emits NOTHING (ack-FREE — the resolving saga is terminal).
    let mut rig = Rig::new();
    rig.grant_realm(); // authority.0 = Some(Fence(1))
    let debris = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let batch = TransferId(0xB7);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        debris,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Arriving { batch },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let discard = TransientHandoff {
        transfer: batch,
        step_id: TRANSIENT_DISCARD_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientDiscard(discard),
    )]);
    assert!(
        !rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&debris),
        "the Arriving copy was discarded"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transients_discarded_source_crash,
        1
    );
    assert_eq!(sent, vec![], "the discard is ack-free (terminal saga)");
}

#[test]
fn redrive_pending_adoptions_re_emits_one_ack_per_distinct_arriving_batch() {
    // CA-1 S3/S4 LIVENESS: `redrive_pending_adoptions` re-emits `BatchAdopted` every tick for each
    // DISTINCT `Arriving` batch (dedup — a batch of N items yields ONE ack, the MMO-scale discipline),
    // and SKIPS settled `Held` items. Seed the tiers DIRECTLY + tick with an empty inbox so the ONLY
    // egress is the re-drive (adopt itself is not exercised here).
    let mut rig = Rig::new();
    rig.grant_realm();
    let batch = TransferId(0xB7);
    let other = TransferId(0xB8); // 0xB7 < 0xB8 → deterministic BTreeSet emit order
    let arriving = |b| Transient {
        pose: transient_pose(),
        anchor_fence: Fence(1),
        status: TransientStatus::Arriving { batch: b },
        prev_offset: LatticePos::ORIGIN,
    };
    {
        let mut owned = rig.world.resource_mut::<OwnedTransients>();
        owned
            .0
            .insert(EntityId::pack(EntityKind::Debris, 1, 7, 0), arriving(batch));
        // Second item, SAME batch → still ONE ack for `batch` (distinct-batch dedup).
        owned
            .0
            .insert(EntityId::pack(EntityKind::Debris, 1, 7, 1), arriving(batch));
        owned
            .0
            .insert(EntityId::pack(EntityKind::Debris, 1, 7, 2), arriving(other));
        // A SETTLED resident (Held) is NOT re-driven — the `if let Arriving` false arm.
        owned.0.insert(
            EntityId::pack(EntityKind::Debris, 1, 7, 3),
            Transient {
                pose: transient_pose(),
                anchor_fence: Fence(1),
                status: TransientStatus::Held { outbound: None },
                prev_offset: LatticePos::ORIGIN,
            },
        );
    }
    let sent = rig.tick(vec![]);
    let ack = |t| {
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: t,
            step_id: TRANSIENT_BATCH_STEP,
        }))
        .expect("encode")
    };
    // Exactly ONE ack per DISTINCT batch, in ascending BTreeSet order.
    assert_eq!(
        sent,
        vec![
            (ORCH, MsgClass::Saga, ack(batch)),
            (ORCH, MsgClass::Saga, ack(other)),
        ]
    );
    assert_eq!(rig.world.resource::<StubStats>().batch_adopts_redriven, 2);
}

#[test]
fn redrive_pending_adoptions_is_a_noop_when_nothing_is_arriving() {
    // The empty-`Arriving` path (the `for batch in batches` empty loop + `if let` all-false): a shard
    // holding only a settled `Held` transient re-drives nothing and emits no egress.
    let mut rig = Rig::new();
    rig.grant_realm();
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        EntityId::pack(EntityKind::Debris, 1, 7, 0),
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held { outbound: None },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![]);
    assert_eq!(sent, vec![], "nothing Arriving → no re-drive");
    assert_eq!(rig.world.resource::<StubStats>().batch_adopts_redriven, 0);
}

#[test]
fn re_solicit_batch_is_a_counted_noop_at_the_source() {
    // CA-1 S3: the orchestrator's AwaitAdopt liveness PROBE arriving at a (live) SOURCE is a counted
    // no-op — no state change, no egress (the probe's signal is its SEND outcome at the orchestrator,
    // not this handler). No `Arriving` items, so the re-drive adds nothing to `sent`.
    let mut rig = Rig::new();
    rig.grant_realm();
    let probe = TransientHandoff {
        transfer: TransferId(0xB7),
        step_id: RE_SOLICIT_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::ReSolicitBatch(probe),
    )]);
    assert_eq!(rig.world.resource::<StubStats>().re_solicits_received, 1);
    assert_eq!(
        sent,
        vec![],
        "the probe is a pure no-op — no reply, no state change"
    );
}

#[test]
fn transient_batch_and_drop_flow_through_the_inbound_dispatch() {
    // Covers the inbound DISPATCH into the transient handlers (the direct-call tests above cover
    // the handlers themselves): on_directory_reply → on_transfer_envelope's `TransientBatch` arm
    // (adopt + ack) AND on_directory_reply's `TransientDrop` arm (promote). The DEST adopts a
    // batch, acks BatchAdopted, then promotes Arriving→Held on the drop.
    let mut rig = Rig::new();
    rig.grant_realm(); // authority.0 = Some(Fence(1))
    let debris = EntityId::pack(EntityKind::Debris, 1, 7, 0);
    let batch = TransferId(0xB7);
    let env = TransferEnvelope {
        transfer_id: batch,
        universe_epoch: vd_core::EpochId(1),
        schema_version: TRANSFER_SCHEMA_VERSION,
        fence: Fence(1),
        step_id: TRANSIENT_BATCH_STEP,
        class: DurabilityClass::Transient,
        payload: TransitionPayload::TransientBatch {
            from_realm: RealmId::System(8),
            to_realm: config().realm,
            src_realm_fence: Fence(1),
            dst_realm_fence: Fence(1),
            source_tick: vd_core::TickId(1),
            items: vec![TransientItem {
                entity: debris,
                pose: transient_pose(),
                state: vec![],
            }],
        },
    };
    let sent = rig.tick(vec![wire_msg(
        NodeId(50),
        MsgClass::Saga,
        &InterShardFlow::Transfer(env),
    )]);
    // Adopted as the uncounted Arriving tier, anchored to the envelope's dst realm fence.
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&debris].status,
        TransientStatus::Arriving { batch }
    );
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&debris].anchor_fence,
        Fence(1)
    );
    // The egress is the BatchAdopted ack TWICE (exact-vec equality — no filter/any closure with an
    // uncoverable short-circuit arm, the HR5 test discipline): the adopt handler acks it once (in
    // `process_inbound`), then CA-1 S3/S4's `redrive_pending_adoptions` re-emits it the SAME tick (the
    // item is now `Arriving`) — the liveness re-drive. Both are byte-identical; the orchestrator absorbs
    // the duplicate (idempotent). Order is adopt-ack THEN re-drive (chain order).
    let expected_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: batch,
            step_id: TRANSIENT_BATCH_STEP,
        }))
        .expect("encode");
    assert_eq!(
        sent,
        vec![
            (ORCH, MsgClass::Saga, expected_ack.clone()),
            (ORCH, MsgClass::Saga, expected_ack),
        ]
    );

    // The TransientDrop dispatch arm PROMOTES the Arriving item → Held + acks DropApplied(DROP).
    let promote = TransientHandoff {
        transfer: batch,
        step_id: TRANSIENT_DROP_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientDrop(promote),
    )]);
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&debris].status,
        TransientStatus::Held { outbound: None }
    );
    assert_eq!(rig.world.resource::<StubStats>().transients_promoted, 1);
    let promote_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: batch,
            step_id: TRANSIENT_DROP_STEP,
        }))
        .expect("encode");
    assert_eq!(sent, vec![(ORCH, MsgClass::Saga, promote_ack)]);

    // The TransientRelease + ReleaseComplete dispatch arms (SOURCE role): seed a Held source item
    // for a 2nd batch, release it → Departing + DropApplied(RELEASE) ack, then complete → retired.
    let src_batch = TransferId(0xB8);
    let src_item = EntityId::pack(EntityKind::Debris, 1, 7, 9);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        src_item,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Held {
                outbound: Some(src_batch),
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let rel = TransientHandoff {
        transfer: src_batch,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientRelease(rel),
    )]);
    assert_eq!(
        rig.world.resource::<OwnedTransients>().0[&src_item].status,
        TransientStatus::Departing { batch: src_batch }
    );
    let release_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: src_batch,
            step_id: TRANSIENT_RELEASE_STEP,
        }))
        .expect("encode");
    assert_eq!(sent, vec![(ORCH, MsgClass::Saga, release_ack)]);

    let rc = TransientHandoff {
        transfer: src_batch,
        step_id: TRANSIENT_RELEASE_STEP,
        fence: Fence(1),
    };
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::ReleaseComplete(rc),
    )]);
    assert!(
        !rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&src_item),
        "ReleaseComplete retired the Departing copy"
    );
    let complete_ack =
        postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: src_batch,
            step_id: TRANSIENT_COMPLETE_STEP,
        }))
        .expect("encode");
    assert_eq!(
        sent,
        vec![(ORCH, MsgClass::Saga, complete_ack)],
        "D-7d: ReleaseComplete acks the retire-complete (SourceRetired drives the saga tail to Done)"
    );

    // The D-7d TransientAbandon dispatch arm (SOURCE role): seed a fresh Departing item for a 3rd
    // batch (the dead-dest case), abandon it → dropped + bucketed as an accounted loss, NO ack (the
    // resolving saga is already terminal).
    let abandon_batch = TransferId(0xB9);
    let abandon_item = EntityId::pack(EntityKind::Debris, 1, 8, 9);
    rig.world.resource_mut::<OwnedTransients>().0.insert(
        abandon_item,
        Transient {
            pose: transient_pose(),
            anchor_fence: Fence(1),
            status: TransientStatus::Departing {
                batch: abandon_batch,
            },
            prev_offset: LatticePos::ORIGIN,
        },
    );
    let sent = rig.tick(vec![wire_msg(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::TransientAbandon(TransientHandoff {
            transfer: abandon_batch,
            step_id: TRANSIENT_ABANDON_STEP,
            fence: Fence(1),
        }),
    )]);
    assert!(
        !rig.world
            .resource::<OwnedTransients>()
            .0
            .contains_key(&abandon_item),
        "TransientAbandon dropped the retained Departing copy"
    );
    assert_eq!(
        rig.world
            .resource::<StubStats>()
            .transients_lost_in_handover
            .get(&EntityKind::Debris),
        Some(&1),
        "the abandoned Debris is bucketed as an accounted loss"
    );
    assert!(sent.is_empty(), "TransientAbandon is terminal — no ack");
}
