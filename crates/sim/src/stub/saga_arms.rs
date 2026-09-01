//! THE SAGA'S SHARD-SIDE ARMS: demote, promote, re-home, flush.
//!
//! Owns: this shard's half of an orchestrated hand-off — the fence-enforced `Owned→Frozen→Ghost`
//! demote at the source, the `Ghost→Owned` promote at the destination, the forward re-home adopt,
//! and the authoritative pose flush the orchestrator carries between them. Every arm is
//! fence-guarded and idempotent, because at-least-once delivery means every one of them will be
//! re-delivered.
//!
//! Does NOT own: the ORDER. The saga decides the sequence and the commit point; these arms only
//! apply what they are ordered to, and refuse what arrives stale. Nor do they contain the pose
//! arithmetic (`conversion`) or the trigger that started it all (`containment`).

use super::{
    AppliedSteps, Dot, Dots, GhostColliderRegistration, GhostNeighbor, HandoffHolds, HoldRole,
    RealmRegions, RequestInFlight, StubConfig, StubStats, crossing_target, flush_pose_for_dest,
    open_hold, place_arriving_pose, push_session_reply,
};
use crate::authority::{Authority, AuthorityCmd};
use crate::io::{Durability, MsgClass};
use crate::runtime::{ClockSample, OutboundBox};
use std::collections::BTreeMap;
use vd_core::kinematics::{self};
use vd_core::placement::PlacementLedger;
use vd_core::pose::StampedPose;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TickId, TransferId, UniverseTick};
use vd_wire::intershard::{
    DemoteCmd, FlushSource, GhostFlow, InterShardFlow, PROMOTE_STEP, PromoteCmd, RE_HOME_STEP,
    ReHomeCmd, ReHomeState, STUB_CROSSING_STEP, TransferAck,
};
use vd_wire::seams::transfer_control::TransferControlAck;
use vd_wire::session_flow::ShardToGateway;

/// The SOURCE self-fence machinery (1c.8/1d.4b): DEMOTE the local granted, non-departing holder of
/// `entity` to a RETAINED Ghost (`Owned→Frozen→Ghost`) at `new_owner_fence` (the post-CAS owner
/// fence, strictly newer than the source's own grant fence — the directory CAS is fence-monotone),
/// NO directory write (the record is the dest's now), `granted` KEPT. 1d.4b RETAINS the dot (the
/// first ghost) instead of `dots.remove`-ing it, so it stops emitting (`simulates()==false`) but
/// survives for the 1d.5b ghost-as-collider. Idempotent WITHOUT IllegalTransition-as-control-flow:
/// an already-Ghost redelivery is a counted no-op via the `simulates()` guard; a no-match (the
/// entity is not held here) is a clean no-op. Monomorphic.
///
/// 1d.5b.2 — the saga-pushed ordered `Demote` ([`on_saga_demote`]) is now the SOLE driver of this
/// transition; the 1c.8 cooperative granted-key poll (which DISCOVERED a foreign takeover from a
/// directory head) is TORN OUT, so `transfer` is always the real `Demote` transfer (no inert
/// fallback). 1d.5b.3b RELOCATED the dest's autonomous `apply_crossing` promote into `on_saga_promote`
/// (strict demote-before-promote), so the source demotes to a retained Ghost BEFORE the dest is Owned:
/// there is NO two-holder window (a brief ZERO-Owned handoff gap instead, excused mid-flight by the
/// 1d.5b.3d per-tick oracle). The retained source Ghost is a live collider fed by the dest (the
/// `GhostFlow` feed, 1d.5b.3b) and torn down on band-exit (1d.5b.3c).
fn self_fence_foreign_entity(
    dots: &mut BTreeMap<SessionId, Dot>,
    entity: EntityId,
    new_owner_fence: Fence,
    transfer: TransferId,
    at_tick: TickId,
    stats: &mut StubStats,
) {
    let Some(session) = dots
        .iter()
        .find(|(_, d)| foreign_takeover_target(d, entity))
        .map(|(session, _)| *session)
    else {
        return; // no matching dot: never held here — a clean no-op
    };
    let dot = dots
        .get_mut(&session)
        .expect("foreign_takeover_target just matched this session");
    // Idempotency WITHOUT relying on IllegalTransition as control flow: only an Owned source
    // demotes; an already-Ghost redelivery is a COUNTED no-op (the covered guard arm).
    if !dot.authority.simulates() {
        stats.self_fence_skipped += 1;
        return;
    }
    // Owned → Frozen (always legal) → Ghost (the foreign CAS fence is strictly newer than the
    // source's own grant fence — the directory CAS is fence-monotone). An Err here is a real
    // invariant break, so panic; the refusal arms are proptested in `authority.rs`.
    dot.authority = dot
        .authority
        .apply(AuthorityCmd::Freeze { transfer })
        .and_then(|frozen| {
            frozen.apply(AuthorityCmd::Demote {
                new_owner_fence,
                at_tick,
            })
        })
        .expect("an Owned source freezes infallibly and demotes at the strictly-newer CAS fence");
    tracing::warn!(
        "entity {entity} now held at fence {new_owner_fence} — source self-demoted to a retained Ghost"
    );
    // KEEP the dot (no remove) and KEEP `granted` == true — it survives as the retained Ghost, holding
    // the pose it last had under this shard's authority. That dot IS the subject's pose truth across the
    // hand-off (FG-2), which is why the hand-off hold stores no pose of its own.
}

/// SOURCE consumer of the saga-pushed ordered `Demote` (1d.5b.1, D-2): the binding fence-enforced
/// `Owned→Frozen→Ghost` demote, the FIRST half of demote-before-promote, and (since 1d.5b.2) the
/// SOLE driver of the source self-fence. It REUSES [`self_fence_foreign_entity`]'s machinery (Freeze
/// at the real `cmd.transfer`, then Demote at `cmd.new_owner_fence` — the post-CAS owner fence). An
/// already-Ghost dot (a redelivery) is the `!simulates()` counted no-op (`self_fence_skipped`); an
/// unheld entity is a clean no-match no-op. The `DemoteAck` is sent UNCONDITIONALLY (outside the flip
/// guard): even on those no-op paths the saga MUST still get its ack or it would wedge in `Demoting`.
/// Monomorphic. (A non-Entity subject has no local Entity dot here — counted + skipped, still acked.)
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_saga_demote(
    cmd: DemoteCmd,
    config: &StubConfig,
    clock: &ClockSample,
    dots: &mut Dots,
    in_flight: &mut RequestInFlight,
    holds: &mut HandoffHolds,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    match cmd.subject.transfer_subject_entity() {
        Some(entity) => {
            self_fence_foreign_entity(
                &mut dots.0,
                entity,
                cmd.new_owner_fence,
                cmd.transfer,
                clock.local_tick,
                stats,
            );
            // Slice 3e: the durable COMMIT terminal at the source — POSITIVELY clear the crossing
            // latch (the triggered durable crossing committed; the entity may trigger again). A latch
            // present-and-removed increments the cleared counter; an absent latch is a clean no-op.
            if in_flight.0.remove(&entity).is_some() {
                stats.crossing_latches_cleared += 1;
            }
            // THIS TICK is where the shard used to go silent: the demote it just applied is what stops it
            // relaying the occupant to its parent and what empties its own occupant set. Open a hold so it
            // keeps doing both until the take-over lands (`GhostFlow::Spawn`) or the budget runs out.
            open_hold(
                holds,
                (entity, HoldRole::Source),
                cmd.new_owner_fence,
                clock.local_tick,
                config.handoff_hold_ttl_ticks,
            );
        }
        None => stats.saga_demote_no_entity += 1,
    }
    // Ack DemoteAck UNCONDITIONALLY — the saga's demote-before-promote ordering gates the dest
    // Promote on THIS ack; a missing ack on a no-op path (already-Ghost / unheld) would wedge it.
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::SagaAck(TransferControlAck::DemoteAck {
            transfer: cmd.transfer,
        }),
    );
}

/// DEST consumer of the saga-pushed ordered `Promote` (1d.5b.3b, D-2): the SECOND half of
/// demote-before-promote and — since 1d.5b.3b — the REAL `Ghost→Owned` promoter (the flip RELOCATED
/// here out of `apply_crossing`, so the ordering is STRICT: the dest becomes Owned only on this
/// command, after the source has demoted). On the FIRST delivery it: flips the dest dot Ghost→Owned
/// at `cmd.new_fence` (gated pose-before-promote on the crossing having landed), announces the dest
/// read-sub to the gateway (`SubscriptionReady` — RELOCATED here from the adopt flip, so the
/// client's render authority moves to the dest only NOW, ~promote-time), registers the transfer
/// `source` as a ghost-neighbor, and SPAWNS the source ghost (`GhostFlow::Spawn` → the dest drives
/// the collider feed). It ALWAYS acks `PromoteAck` (outside the FirstApply gate) so a redelivery
/// re-acks without re-flipping/re-spawning. Journal-gate + ack here; the branchy flip lives in the
/// monomorphic [`promote_apply`] (HR5). `realm_fence` is the dest's realm authority (held by the
/// invariant that the orchestrator only routes a Promote to the realm's owner).
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_saga_promote(
    cmd: PromoteCmd,
    config: &StubConfig,
    self_node: NodeId,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    registration: &mut GhostColliderRegistration,
    realm_fence: Fence,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // Journal PROMOTE_STEP only when the flip ACTUALLY lands. A promote that arrives before its precondition —
    // the crossing pose journaled AND a granted crossing-target dot — cannot flip yet; journaling it anyway
    // (the old unconditional `journal_step` here) marked it done FOREVER, so the standing Promoting-timeout
    // re-emit hit `AlreadyApplied` and never re-ran the flip even after the crossing landed, leaving the entity
    // a silent non-emitting Ghost (the return-crossing total-freeze). Withholding the journal on a non-flip
    // keeps the promote RE-DRIVABLE: the next timeout window re-runs `promote_apply` and completes the flip once
    // the fresh-adopt is granted and its buffered crossing has drained. A redelivery AFTER a real flip is the
    // no-op `AlreadyApplied` (protects the strict no-re-flip-on-redelivery invariant).
    if applied.is_applied(cmd.transfer, PROMOTE_STEP) {
        stats.promotes_redelivered += 1;
    } else if promote_apply(
        cmd,
        config,
        self_node,
        dots,
        applied,
        registration,
        realm_fence,
        stats,
        outbox,
    ) == PromoteOutcome::Flipped
    {
        // The flip landed — mark PROMOTE_STEP applied so a later redelivery is the no-op above.
        let _ = applied.journal_step(cmd.transfer, PROMOTE_STEP);
    }
    // Ack UNCONDITIONALLY — the release gate (PromoteAcked AND DestDelivered) needs this ack even on
    // a redelivery / a deferred (pose-not-yet-landed) flip; never wedge the saga in Promoting.
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::SagaAck(TransferControlAck::PromoteAck {
            transfer: cmd.transfer,
        }),
    );
}

/// The outcome of a DEST [`promote_apply`] attempt — whether the `Ghost→Owned` flip actually LANDED. The
/// caller journals `PROMOTE_STEP` ONLY on `Flipped`, so a promote that arrives before its precondition is met
/// (a fresh-adopt still un-granted, or the crossing pose not yet journaled — the case on a RETURN to a shard
/// that despawned the entity's retained ghost and must re-adopt) stays RE-DRIVABLE by the standing Promoting
/// timeout, instead of being journaled-as-done with the entity left a silent non-emitting Ghost FOREVER (the
/// return-crossing total-freeze: no feed, input dropped as `PendingAuthority`, no `SubscriptionReady`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromoteOutcome {
    /// The dot flipped to `Owned` — the caller journals `PROMOTE_STEP` applied.
    Flipped,
    /// The crossing pose has not landed (`STUB_CROSSING_STEP` unjournaled) — NOT journaled; re-drivable.
    Deferred,
    /// No granted crossing-target dot for the subject yet — NOT journaled; re-drivable (the adopt may still land).
    NoDot,
}

/// The DEST promote effect (1d.5b.3b), monomorphic so every branch is covered ONCE here, not in a
/// generic body (HR5). Find the dest Ghost dot for the subject entity; if the crossing pose has not
/// yet landed (`STUB_CROSSING_STEP` not journaled) DEFER (count, no flip — pose-before-promote);
/// else flip Ghost→Owned at `cmd.new_fence` (infallible: the dest Ghost's GENESIS `source_fence` is
/// strictly below the post-CAS fence), announce the dest sub, register the ghost-neighbor, and spawn
/// the source ghost. A non-Entity subject or a missing dot is a counted no-op (`promote_no_dot`).
#[allow(clippy::too_many_arguments)]
fn promote_apply(
    cmd: PromoteCmd,
    config: &StubConfig,
    self_node: NodeId,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    registration: &mut GhostColliderRegistration,
    realm_fence: Fence,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) -> PromoteOutcome {
    let Some(entity) = cmd.subject.transfer_subject_entity() else {
        stats.promote_no_dot += 1;
        return PromoteOutcome::NoDot;
    };
    let Some((session, dot)) = dots.0.iter_mut().find(|(_, d)| crossing_target(d, entity)) else {
        stats.promote_no_dot += 1;
        tracing::warn!(
            %entity,
            transfer = ?cmd.transfer,
            "Promote found NO granted crossing-target dot — the dest adopt never established (the saga re-drives)"
        );
        return PromoteOutcome::NoDot;
    };
    if !applied.is_applied(cmd.transfer, STUB_CROSSING_STEP) {
        // The crossing pose has not landed yet — flipping now would emit a poseless origin frame. DEFER.
        // ⚠️ RECOVERY (DEFERRED.md D-6 #1, the durable entity-STATE crossing as a producer-less phase):
        // re-driving the PROMOTE does NOT cure a crossing that the at-most-once mesh LOST — Promote only
        // re-acks + re-defers here. The lost-crossing case needs the CROSSING re-driven (a Demoting/Promoting
        // Timeout that also re-emits `EmitCrossing`, idempotent via this `STUB_CROSSING_STEP` journal dedup —
        // the proven D-37-2d template) OR the owed redelivering transport. Under the harness at-least-once
        // FaultFabric the crossing always redelivers, so this DEFER is reached only as the transient
        // promote-before-crossing race, never a permanent wedge; the permanent case is a deploy precondition.
        stats.promote_before_crossing += 1;
        tracing::warn!(
            %entity,
            transfer = ?cmd.transfer,
            "Promote DEFERRED — the crossing pose has not landed yet (the saga re-drives the Promote)"
        );
        return PromoteOutcome::Deferred;
    }
    let session = *session;
    // Ghost→Owned at the post-CAS fence. TWO cases, split by whether this is a SOURCE==DEST re-home (task
    // #149) — a co-hosted-child crossing whose `head(Realm(dest))` resolved to THIS node:
    // - CROSS-NODE (`cmd.source != self_node`): the DEST's Ghost holds GENESIS `source_fence`, strictly < the
    //   CAS fence, so the standard strict-newer `AuthorityCmd::Promote` is infallible.
    // - SOURCE==DEST (`cmd.source == self_node`): the ordered `Demote` already self-fenced THIS SAME dot to
    //   `Ghost{source_fence: cmd.new_fence}` (the demote lands the CAS fence), so a strict-newer Promote at
    //   the SAME fence would `StaleFence`. The directory CAS committed THIS node as the owner at
    //   `cmd.new_fence`, so RE-OWN the dot at that exact fence directly (the route swap completing as the
    //   idempotent no-op the source==dest saga is). This monomorphic 2-arm `if` keeps both paths covered
    //   (HR5): the cross-node strict-newer promote AND the source==dest equal-fence re-own.
    dot.authority = if cmd.source == self_node {
        Authority::Owned {
            fence: cmd.new_fence,
        }
    } else {
        dot.authority
            .apply(AuthorityCmd::Promote {
                new_fence: cmd.new_fence,
            })
            .expect(
                "cross-node dest Ghost promotes at the post-CAS fence (strictly newer than GENESIS)",
            )
    };
    let gateway = dot.gateway;
    let pose = dot.pose;
    stats.promotes_confirmed += 1;
    // RELOCATED here from the adopt flip (1d.5b.3b): the dest read-sub is announced ONLY now, at
    // promote — so the client's render authority moves to the dest only after it is genuinely Owned.
    push_session_reply(
        outbox,
        gateway,
        &ShardToGateway::SubscriptionReady {
            session,
            entity,
            frame: config.frame,
            realm_fence,
        },
    );
    // The dest (owner) registers the source as a ghost-neighbor + spawns the source ghost; the ANCHOR is
    // THIS promote pose (= the crossed pose, the boundary the entity entered through), from which the dest
    // measures band membership + Despawns the ghost on band-exit (1d.5b.3c). Shared tail — see the fn doc.
    // `self_node` guards the source==dest same-node re-home (no self-ghost feed loop).
    register_and_spawn_source_ghost(
        entity,
        cmd.source,
        self_node,
        pose,
        cmd.new_fence,
        registration,
        outbox,
    );
    PromoteOutcome::Flipped
}

/// Register the transfer/re-home SOURCE as a ghost-neighbor + SPAWN the source ghost so this shard (the new
/// owner) DRIVES the `GhostFlow` collider feed to the source (owner → ghost-host) — keeping the retained
/// source ghost a live collider + the render seamless (`feed_source_ghosts` streams Delta after). The ANCHOR
/// is the applied (crossed / re-homed) pose. Shared by [`promote_apply`] (a `Ghost→Owned` flip) and
/// [`re_home_apply`] (a fresh Owned build): the two DIVERGE above this tail, which was byte-identical (the
/// old ⚠️ DRY-PIN, now extracted so a [[D-39]].6 ghost-blob / band-driven multi-neighbor edit touches ONE
/// place). A Spawn to a possibly-dead source is harmless FireAndForget (HR1: the shard cannot see the
/// liveness set — never special-case it).
///
/// SOURCE==DEST GUARD (task #149): a same-node re-home (a co-hosted-child crossing whose `head(Realm(dest))`
/// resolves to THIS node) drives the SAME orchestrator saga, so it reaches promote with `source == self_node`.
/// Registering `self` as a ghost-neighbor of itself + Spawning a ghost to itself would create a self-ghost
/// feed loop (the owner would `GhostFlow::Delta` its own retained copy). Skip BOTH here — the dot is already
/// Owned+rendered on this node; there is no foreign owner to feed. A cross-node source (`source != self_node`)
/// takes the register+Spawn path exactly as before. Monomorphic, both arms covered.
#[allow(clippy::too_many_arguments)]
fn register_and_spawn_source_ghost(
    entity: EntityId,
    source: NodeId,
    self_node: NodeId,
    pose: StampedPose,
    source_fence: Fence,
    registration: &mut GhostColliderRegistration,
    outbox: &mut OutboundBox,
) {
    if source == self_node {
        // Same-node re-home: no foreign owner to feed. Skip the self-ghost register + Spawn (else a
        // self-feed loop). The dot is already Owned + rendered here — nothing else is owed.
        return;
    }
    registration.0.insert(
        entity,
        GhostNeighbor {
            source,
            anchor: pose.pos,
        },
    );
    // The pose-free proof (slice F), RETAINED: a promote redelivery re-acks without re-spawning,
    // so nothing re-sends this — a crash between the emit and the send would otherwise leave the
    // source's hold to its TTL and delay every bystander's leaver-vanish.
    outbox.push_flow_durable(
        source,
        MsgClass::GhostReliable,
        &InterShardFlow::Ghost(GhostFlow::SpawnV2 {
            entity,
            source_fence,
        }),
        Durability::Retained,
    );
}

/// D-37 forward re-home ADOPT (the journal gate + the ack, modelled on `on_saga_promote`). The fresh
/// target RECEIVES a `ReHome` after the orchestrator re-homed a permanently-killed owner's committed
/// entity here; it CREATES the entity as an Owned dot from the carried pose (no pre-existing ghost to
/// flip — unlike `Promote`'s `Ghost→Owned`).
///
/// THE ACK IS THE CLAIM "IT LANDED", so it is pushed only where that is true: a FIRST apply that
/// succeeded, or a REDELIVERY of a step that already succeeded (the saga's `Promoting` release gate needs
/// the ack again on a redelivery, else it wedges). A REFUSED adopt — a pose this shard cannot place, a
/// non-Entity subject — acks NOTHING and journals NOTHING, so the saga times out and aborts and the
/// source keeps the entity.
///
/// Both used to sit OUTSIDE the match, unconditionally. That is strictly worse than a misplacement: the
/// orchestrator was told the entity had landed here, committed it, and the source let go — of an entity
/// that this shard had refused and that therefore existed nowhere at all. Journaling was worse still,
/// because it made the refusal STICK: the redelivery read `AlreadyApplied` and acked a second time.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_re_home(
    cmd: ReHomeCmd,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    self_node: NodeId,
    clock: &ClockSample,
    dots: &mut Dots,
    applied: &mut AppliedSteps,
    registration: &mut GhostColliderRegistration,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    // §3.3 epoch fail-safe, UNIFORM with the crossing ingress (`on_transfer_envelope`): a re-home adopt
    // carrying a `universe_epoch` that does not match this shard's current epoch is REFUSED — never
    // journaled, adopted, or acked — so no entity is reconstructed at a stale celestial position. The
    // re-home is the SECOND pose-placing ingress; guarding it here makes §3.3 cover BOTH. A cross-epoch
    // re-home is a defunct OLD-epoch saga (only reachable across a re-genesis / a delayed redelivery): NOT
    // acking is correct — the current-epoch orchestrator is not waiting on it (same rationale as the
    // crossing arm's no-ack-on-mismatch).
    if cmd.universe_epoch != clock.epoch {
        stats.re_home_epoch_mismatch += 1;
        return;
    }
    let transfer = cmd.transfer; // `ReHomeCmd` is Clone-not-Copy (the pose payload) — capture before the move
    // CONSULT (read-only) before the effect and RECORD after it — the same discipline the crossing
    // ingress follows. `journal_step` records as it answers, which is why it cannot be the gate here: it
    // would journal an adopt that then refused.
    if applied.is_applied(transfer, RE_HOME_STEP) {
        stats.re_home_redelivered += 1;
    } else if re_home_apply(
        cmd,
        config,
        regions,
        placements,
        self_node,
        dots,
        registration,
        stats,
        outbox,
    ) {
        applied.journal_step(transfer, RE_HOME_STEP);
    } else {
        return; // refused: nothing landed here, so nothing is acked and nothing is journaled
    }
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::SagaAck(TransferControlAck::PromoteAck { transfer }),
    );
}

/// The D-37 re-home effect (monomorphic so every branch is covered ONCE here — HR5). DIVERGES from
/// `promote_apply` by taking the pose from the PAYLOAD rather than a crossing journal, and it has TWO
/// landing shapes: the target usually holds NOTHING for the subject (the killed dest was elsewhere)
/// and a fresh Owned dot is BUILT; but when the flushed pose's frame realm resolved the re-home right
/// back to the SOURCE shard (an upward hand-off ships the pose verbatim in the source's own frame),
/// the target still holds the subject's RETAINED GHOST — and that dot is FLIPPED in place, never
/// doubled (the batch-review MAJOR: a second dot for one EntityId left an orphan ghost whose expiring
/// hold broadcast `EntityRemoved` for a live entity). A non-Entity subject is a counted no-op
/// (`re_home_no_entity`). A FRESH-built dot is clientless (no session route — the client re-subscribes
/// via the D-37/D-36 connection-plane path, owed): `AccountId(0)` + the orchestrator as an inert reply
/// sentinel, `granted` so it simulates + emits frames, born `Owned` at `cmd.new_fence`; a FLIPPED dot
/// keeps the client linkage it retained.
///
/// Returns whether the entity is ACTUALLY HERE NOW. `false` means this shard refused the adopt and holds
/// nothing — the caller must not journal it and must not ack it (see [`on_re_home`]).
#[allow(clippy::too_many_arguments)]
fn re_home_apply(
    cmd: ReHomeCmd,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    self_node: NodeId,
    dots: &mut Dots,
    registration: &mut GhostColliderRegistration,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) -> bool {
    let Some(entity) = cmd.subject.transfer_subject_entity() else {
        stats.re_home_no_entity += 1;
        return false;
    };
    // Never trust the network: sanitize the carried pose to finite at this ingress. (P7 grows
    // `ReHomeState::Snapshot` — this `let` becomes a `match` whose new arm needs its own coverage. TODO.)
    let ReHomeState::PoseOnly(raw) = cmd.state;
    // THE RECEIVER'S CONVERSION — the SAME helper the crossing ingress uses, because there is ONE rule
    // and this is the second place a pose enters a shard (see `place_arriving_pose`). A pose this shard
    // cannot measure means NO dot is created: reconstructing an entity at an unplaceable position is the
    // failure this ingress exists to prevent.
    // The D-37 forward re-home names no destination realm of its own: the target IS the shard receiving
    // the command, so the realm it re-homes into is that shard's own. (The durable crossing ingress reads
    // its destination off the envelope, because a crossing can name a co-hosted child.)
    let sanitized = raw.sanitized();
    let pose =
        match place_arriving_pose(sanitized, config.realm, config, regions, placements, stats) {
            Ok(placed) => placed,
            Err(err) => {
                stats.arrivals_unplaceable += 1;
                tracing::error!(
                    %err,
                    arriving = ?sanitized.frame,
                    own = ?config.realm,
                    %entity,
                    "refusing a re-home this shard cannot place — the entity is NOT reconstructed"
                );
                return false;
            }
        };
    // SAME-ENTITY GUARD (batch review, MAJOR): the target may ALREADY hold a dot for this entity —
    // the retained Ghost its own outward demote left behind, found here whenever the forward
    // re-home's target resolution (the flushed pose's frame realm) lands the entity back on its
    // SOURCE shard. Inserting under the synthetic key would mint a SECOND dot for one EntityId: the
    // orphan Ghost's hand-off hold then runs to TTL and the expiry fan broadcasts `EntityRemoved`
    // for an entity this shard now OWNS and emits, and a later stale band-exit `Despawn` tears out
    // whichever ghost-shaped dot it finds. So the held dot is FLIPPED in place — same session key,
    // keeping the client linkage it retained — and AUTHORITY-UNIQUE stays ≤ 1 dot per entity.
    if let Some(session) = dots
        .0
        .iter()
        .find(|(_, d)| d.entity == entity)
        .map(|(s, _)| *s)
    {
        let dot = dots
            .0
            .get_mut(&session)
            .expect("the same-entity dot was just found");
        // The directory CAS committed THIS shard as the owner at `cmd.new_fence`, so re-own the held
        // dot at that exact fence — the same direct re-own the source==dest Promote arm performs
        // (a strict-newer FSM Promote would `StaleFence` against the demote that landed this fence).
        dot.authority = Authority::Owned {
            fence: cmd.new_fence,
        };
        dot.entity_fence = cmd.new_fence;
        dot.granted = true;
        dot.departing = false;
        dot.adopting = false;
        dot.pose = pose;
        // Seed to the re-homed pose offset: this tick's swept segment is degenerate.
        dot.prev_offset = pose.pos;
        stats.re_home_flipped += 1;
    } else {
        // Deterministic clientless session key (entity id ↦ session) so seed-replay stays byte-identical and
        // the oracle held-set sees exactly one Owned dot for this entity.
        let session = SessionId(entity.0);
        dots.0.insert(
            session,
            Dot {
            last_stick: None,
                entity,
                account: AccountId(0), // orphan: no client account until the session re-homes (D-37/D-36)
                session_fence: Fence::GENESIS,
                gateway: config.orchestrator, // inert reply sentinel — push_session_reply is never called here
                granted: true,
                input_active: false,
                adopting: false,
                authority: Authority::Owned {
                    fence: cmd.new_fence,
                },
                departing: false,
                entity_fence: cmd.new_fence,
                pose,
                // ★THE RE-HOME FACING (measured by the owner flying, 2026-08-21: "when I'm re-homed
                // the direction I look and the direction W moves me are not aligned").
                //
                // A crossing CONVERTS the pose's orientation into the destination frame — correctly.
                // But the input integrator does not read that orientation; it REBUILDS it each tick
                // from the yaw/pitch pair. Handing the arriving dot a converted pose and a zeroed
                // pair therefore threw the player's facing away on the first input tick after every
                // re-home: the body snapped to the new frame's default heading while the client's
                // own camera kept the heading it had, so looking and moving came apart.
                //
                // The angles are now DERIVED from the converted orientation, so the rebuild
                // reproduces exactly the facing the crossing computed.
                yaw: kinematics::yaw_pitch_from_orient(pose.orient).0,
                pitch: kinematics::yaw_pitch_from_orient(pose.orient).1,
                last_applied_seq: None,
                // Seed to the re-homed pose offset: this tick's swept segment is degenerate.
                prev_offset: pose.pos,
            },
        );
    }
    stats.re_home_adopted += 1;
    // Register the (re-home) source as a ghost-neighbor + spawn its ghost — the target (owner) now drives
    // the collider feed to it, exactly as `promote_apply` does. Shared tail — see the fn doc. `self_node`
    // guards the source==dest same-node re-home (no self-ghost feed loop).
    register_and_spawn_source_ghost(
        entity,
        cmd.source,
        self_node,
        pose,
        cmd.new_fence,
        registration,
        outbox,
    );
    true
}

/// Whether a dot is the local granted, non-departing holder of `entity` (the self-fence target).
/// Monomorphic predicate so the chained `&&`s are covered in one helper, not the reply arm.
///
/// ⚠️ INTENTIONALLY identical-bodied to [`crossing_target`] AND [`flush_target`] — a TRIPLET, DO NOT
/// merge them (the `twin-D1` reconciliation; unrelated to DEFERRED `D-1`). The NAMES carry distinct
/// intents at disjoint call sites/shards: this finds the SOURCE's held dot to DEMOTE on the saga
/// `Demote`; `crossing_target` finds the DEST dot to APPLY a crossing; `flush_target` finds the
/// SOURCE's held dot to SHIP its pose. AUTHORITY-UNIQUE bounds each to ≤1 dot — see [`crossing_target`].
#[must_use]
fn foreign_takeover_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & dot.granted & !dot.departing
}

/// SOURCE side of the 1d.1 pose flush: ship the authoritative pose of the held subject dot back to
/// the orchestrator (`TransferAck::SourceFlushed`), so the saga can stamp the crossing. Read-only —
/// authority is unchanged (the abort-path `ThawSource` is the only stateful source undo). Three
/// total paths, all covered: a non-Entity subject → no-op; the subject not held here → counted
/// no-op (a stale/misrouted flush); held → ship. Monomorphic (the finder + ship are hoisted out of
/// the decode arm — HR5 branchless shim).
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_flush_source(
    flush: FlushSource,
    config: &StubConfig,
    regions: &RealmRegions,
    placements: &PlacementLedger,
    tick: UniverseTick,
    dots: &Dots,
    stats: &mut StubStats,
    outbox: &mut OutboundBox,
) {
    let Some(entity) = flush.subject.transfer_subject_entity() else {
        return; // a non-Entity subject is not a per-entity pose flush
    };
    let Some(dot) = dots.0.values().find(|d| flush_target(d, entity)) else {
        tracing::warn!(%entity, "FlushSource for an entity this shard does not hold — no pose to ship");
        return;
    };
    let Some(pose) = flush_pose_for_dest(
        dot.pose,
        flush.to_realm,
        config,
        regions,
        placements,
        tick,
        stats,
    ) else {
        // A REAL fault, already counted + logged inside the helper. Ship NO `SourceFlushed`: the saga
        // then times out and aborts, and the source keeps authority — the entity stays somewhere real
        // rather than being handed over with a position nobody can vouch for.
        return;
    };
    // THE B-1 MEASUREMENT (placement arc S0→S2): the instant the shipped pose SPEAKS AT against this
    // shard's clock at the flush. Before the fix the pose shipped at its frozen latch stamp and this
    // measured 1 tick in the in-process cluster; the flush now re-reads the world at the head book's
    // instant, so any gap here is a re-opened frozen-instant defect (the crossing e2e pins it to 0).
    stats.flush_stamp_gap_ticks_max = stats
        .flush_stamp_gap_ticks_max
        .max(tick.0.saturating_sub(pose.universe_tick.0));
    outbox.push_flow(
        config.orchestrator,
        MsgClass::Saga,
        &InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: flush.transfer,
            step_id: flush.step_id,
            pose,
            // The source's own input drain watermark (observability; the saga's CAS watermark is
            // the GATEWAY's SourceFrozen seq, not this).
            drained_seq: dot.last_applied_seq.unwrap_or(0),
        }),
    );
}

/// Whether a dot is the local held holder of `entity` whose pose the source flushes. Monomorphic
/// (the `&`s are covered once here, not the decode arm). The THIRD member of the identical-bodied
/// `twin-D1` triplet ([`foreign_takeover_target`], [`crossing_target`]) — DO NOT merge (distinct
/// intent: this SHIPS a pose; disjoint call site; AUTHORITY-UNIQUE bounds it to ≤1 dot).
#[must_use]
fn flush_target(dot: &Dot, entity: EntityId) -> bool {
    (dot.entity == entity) & dot.granted & !dot.departing
}
