//! G-SEALED conformance gate (`docs/design/sealed_shards.md` §1) — lifted from the
//! in-module unit test to the design-named INTEGRATION location so the closed-set
//! guarantee is a per-release gate, not a convention (audit SEAL-2).
//!
//! The closed taxonomy `InterShardFlow` (the FULL current arm set — see `wire/src/lib.rs`
//! for the canonical enumeration; this header does NOT re-list it, and does not COUNT it, to
//! avoid a second copy that drifts, the exact staleness the `arm_tripwire` below structurally
//! prevents) is the ONLY
//! shape that crosses a shard boundary; every arm has a
//! coherent `EffectClass`, and every SIDE-EFFECTING arm carries an idempotency key (so
//! an authority-gating payload can never ride a fire-and-forget channel). Two compile-
//! time tripwires keep this gate honest as arms are added: the exhaustive `match` in
//! `effect_class` (adding an arm without classifying it does not compile) and the
//! `arm_tripwire` below (adding an arm without representing it in `every_arm` here does
//! not compile) — so the per-release surface enumeration can never silently drift.

use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
use vd_core::{
    AccountId, EntityId, EpochId, Fence, NodeId, SessionId, TickId, TransferId, UniverseTick,
};
use vd_wire::intershard::{
    EffectClass, FLUSH_SOURCE_STEP, FlowDurabilityClass, FlushSource, GhostFlow, IdempotencyKey,
    InterShardFlow, STUB_CROSSING_STEP, TransferAck, TransferEnvelope, TransferStepRejectReason,
    TransitionPayload,
};
use vd_wire::seams::directory::{
    AuthorityRef, CasOutcome, DirectoryKey, DirectoryOp, DirectoryReply,
};
use vd_wire::seams::transfer_control::{PrepareResult, TransferControl, TransferControlAck};

fn pose() -> StampedPose {
    StampedPose::at_rest(
        FrameRef::SystemSpace { system_seed: 1 },
        vd_core::glam::DVec3::ZERO,
        UniverseTick(10),
    )
}

fn eid(kind: EntityKind) -> EntityId {
    EntityId::pack(kind, 1, 7, 3)
}

/// EVERY arm of the closed taxonomy, in one place — the per-release surface gate.
fn every_arm() -> Vec<InterShardFlow> {
    let envelope = |payload, class| {
        InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: TransferId(11),
            universe_epoch: EpochId(1),
            schema_version: 1,
            fence: Fence(2),
            step_id: 4,
            class,
            payload,
        })
    };
    vec![
        // ★TOMBSTONE (Step 5 slice F, minor 15) — the old pose-carrying take-over proof. Nothing
        // produces it; the nested discriminant is reserved forever, so its SHAPE stays pinned here.
        InterShardFlow::Ghost(GhostFlow::Spawn {
            entity: eid(EntityKind::Ship),
            pose: pose(),
            source_fence: Fence(1),
            since_tick: TickId(2),
        }),
        // ★TOMBSTONE (Step 5 slice F, minor 15) — the dest→source ghost pose feed; same reservation.
        InterShardFlow::Ghost(GhostFlow::Delta {
            entity: eid(EntityKind::Ship),
            pose: pose(),
            source_fence: Fence(1),
            source_tick: TickId(3),
            seq: 1,
        }),
        InterShardFlow::Ghost(GhostFlow::Despawn {
            entity: eid(EntityKind::Ship),
            source_fence: Fence(1),
        }),
        // The pose-free take-over proof (slice F, minor 15) — Spawn's lawful replacement.
        InterShardFlow::Ghost(GhostFlow::SpawnV2 {
            entity: eid(EntityKind::Ship),
            source_fence: Fence(1),
        }),
        envelope(
            TransitionPayload::InitialSpawn {
                entity: eid(EntityKind::Player),
                to_realm: RealmId::System(1),
                pose: pose(),
                state: vec![],
            },
            DurabilityClass::Durable,
        ),
        envelope(
            TransitionPayload::StubCrossing {
                entity: eid(EntityKind::Player),
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                pose: pose(),
                state: vec![],
            },
            DurabilityClass::Durable,
        ),
        envelope(
            TransitionPayload::TransientBatch {
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                src_realm_fence: Fence(3),
                dst_realm_fence: Fence(4),
                source_tick: TickId(6),
                items: vec![],
            },
            DurabilityClass::Transient,
        ),
        InterShardFlow::Directory(DirectoryOp::LeaseGrant {
            key: DirectoryKey::Realm(RealmId::System(1)),
            owner: AuthorityRef::Shard(NodeId(2)),
            fence: Fence(5),
        }),
        InterShardFlow::Directory(DirectoryOp::CommitCas {
            key: DirectoryKey::Realm(RealmId::System(1)),
            expected: Fence(5),
            transfer: TransferId(6),
            new_owner: AuthorityRef::Shard(NodeId(3)),
        }),
        InterShardFlow::Directory(DirectoryOp::HeadRead {
            key: DirectoryKey::Realm(RealmId::System(1)),
        }),
        // The remaining four `DirectoryOp` variants (audit :650 — `every_arm` used to represent 3 of
        // 7, so the nested-payload halves of the roundtrip + classification gates ran vacuously on
        // the absent ones): renewal + clock are FireAndForget; revoke keys on its fence, abort-CAS
        // on `(expected, transfer)`.
        InterShardFlow::Directory(DirectoryOp::LeaseRenew {
            key: DirectoryKey::Realm(RealmId::System(1)),
            fence: Fence(5),
        }),
        InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Realm(RealmId::System(1)),
            fence: Fence(5),
        }),
        InterShardFlow::Directory(DirectoryOp::AbortCas {
            key: DirectoryKey::Realm(RealmId::System(1)),
            expected: Fence(5),
            transfer: TransferId(6),
        }),
        InterShardFlow::Directory(DirectoryOp::ClockSync {
            universe_tick: UniverseTick(9),
            epoch: EpochId(1),
        }),
        // P2 arms (the route-swap saga) — SIDE-EFFECTING (TransferStep) for the command
        // + its ack; FIRE-AND-FORGET for the directory reply envelope. EVERY nested variant of
        // `TransferControl` (7), `TransferControlAck` (10) and `DirectoryReply` (3) is represented
        // (audit :650 — one apiece used to stand in for the whole enum).
        InterShardFlow::Saga(TransferControl::PrepareSubscribe {
            transfer: TransferId(7),
            session: SessionId(1),
            dest: NodeId(2),
        }),
        InterShardFlow::Saga(TransferControl::RequestCut {
            transfer: TransferId(7),
            session: SessionId(1),
        }),
        InterShardFlow::Saga(TransferControl::FreezeSource {
            transfer: TransferId(7),
            session: SessionId(1),
            marker_seq: 12,
            dest: NodeId(2),
        }),
        InterShardFlow::Saga(TransferControl::CommitAuthority {
            transfer: TransferId(7),
            session: SessionId(1),
            new_fence: Fence(6),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
        }),
        InterShardFlow::Saga(TransferControl::ThawSource {
            transfer: TransferId(7),
            session: SessionId(1),
        }),
        InterShardFlow::Saga(TransferControl::AbortTransfer {
            transfer: TransferId(7),
            session: SessionId(1),
        }),
        InterShardFlow::Saga(TransferControl::ReleaseSubscribe {
            transfer: TransferId(7),
            session: SessionId(1),
            src: NodeId(3),
        }),
        InterShardFlow::SagaAck(TransferControlAck::Prepared {
            transfer: TransferId(8),
            result: PrepareResult::Ready,
        }),
        InterShardFlow::SagaAck(TransferControlAck::CutConfirmed {
            transfer: TransferId(8),
            marker_seq: 12,
        }),
        InterShardFlow::SagaAck(TransferControlAck::SourceFrozen {
            transfer: TransferId(8),
            drained_seq: 12,
        }),
        InterShardFlow::SagaAck(TransferControlAck::Committed {
            transfer: TransferId(8),
        }),
        InterShardFlow::SagaAck(TransferControlAck::SourceThawed {
            transfer: TransferId(8),
        }),
        InterShardFlow::SagaAck(TransferControlAck::Aborted {
            transfer: TransferId(8),
        }),
        InterShardFlow::SagaAck(TransferControlAck::Released {
            transfer: TransferId(8),
        }),
        InterShardFlow::SagaAck(TransferControlAck::DemoteAck {
            transfer: TransferId(8),
        }),
        InterShardFlow::SagaAck(TransferControlAck::PromoteAck {
            transfer: TransferId(8),
        }),
        InterShardFlow::SagaAck(TransferControlAck::DeliveredToObservers {
            transfer: TransferId(8),
        }),
        InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Realm(RealmId::System(1)),
            record: None,
        }),
        InterShardFlow::DirectoryReply(DirectoryReply::CasResult {
            key: DirectoryKey::Realm(RealmId::System(1)),
            outcome: CasOutcome::Won {
                new_fence: Fence(6),
            },
        }),
        InterShardFlow::DirectoryReply(DirectoryReply::ClockNow {
            universe_tick: UniverseTick(9),
            epoch: EpochId(1),
        }),
        // 1d.1 arms: the pose-flush request + the entity-state ack family (SIDE-EFFECTING,
        // TransferStep-keyed by their step phase).
        InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(9),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            step_id: FLUSH_SOURCE_STEP,
            to_realm: RealmId::System(0),
            to_parent: None,
        }),
        InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: TransferId(10),
            step_id: FLUSH_SOURCE_STEP,
            pose: pose(),
            drained_seq: 5,
        }),
        // ALL three TransferAck inner arms ride the per-release surface gate (so a new inner variant
        // is forced through the InterShardFlow postcard roundtrip + effect-class classification, not
        // just the outer arm). Rejected has no producer yet (reserved, D-21) but is sealed here.
        InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: TransferId(10),
            step_id: STUB_CROSSING_STEP,
        }),
        InterShardFlow::TransferAck(TransferAck::Rejected {
            transfer_id: TransferId(10),
            step_id: STUB_CROSSING_STEP,
            reason: TransferStepRejectReason::SpatialPrecondition,
        }),
        // D-7a arm: the dest's TRANSIENT batch-adopt ack (SIDE-EFFECTING, TransferStep-keyed by the
        // transient batch phase) — sealed here so a new inner variant rides the surface gate.
        InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
        }),
        // D-7b arm: the transient handoff DropApplied ack (release-ack / promote-confirm) — sealed
        // so a new TransferAck variant rides the surface gate + the roundtrip/classification.
        InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_RELEASE_STEP,
        }),
        // 1d.5b arms: the saga-pushed ordered demote/promote commands (SIDE-EFFECTING,
        // TransferStep-keyed by DEMOTE_STEP/PROMOTE_STEP).
        InterShardFlow::Demote(vd_wire::intershard::DemoteCmd {
            transfer: TransferId(10),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_owner_fence: Fence(6),
            step_id: vd_wire::intershard::DEMOTE_STEP,
        }),
        InterShardFlow::Promote(vd_wire::intershard::PromoteCmd {
            transfer: TransferId(10),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: vd_wire::intershard::PROMOTE_STEP,
            source: vd_core::NodeId(2),
        }),
        // D-7b arms: the three transient structural drop-before-promote handoff commands (all
        // SIDE-EFFECTING, TransferStep-keyed; the source-only release/complete + the dest-only
        // promote — idempotent by journaled step + local held-status). Share `TransientHandoff`.
        InterShardFlow::TransientRelease(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_RELEASE_STEP,
            fence: Fence(6),
        }),
        InterShardFlow::TransientDrop(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_DROP_STEP,
            fence: Fence(6),
        }),
        InterShardFlow::ReleaseComplete(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_RELEASE_STEP,
            fence: Fence(6),
        }),
        // D-7d arm: the dead-DEST ABANDON (SIDE-EFFECTING, TransferStep-keyed by TRANSIENT_ABANDON_STEP;
        // shares `TransientHandoff` — a proper new action, classified by the same one arm).
        InterShardFlow::TransientAbandon(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_ABANDON_STEP,
            fence: Fence(6),
        }),
        // D-37 arm: the forward re-home adopt (SIDE-EFFECTING, TransferStep-keyed by RE_HOME_STEP) — a
        // DEDICATED arm carrying the `ReHomeState` payload, never a `Promote` reuse (no-repurpose).
        InterShardFlow::ReHome(vd_wire::intershard::ReHomeCmd {
            transfer: TransferId(10),
            universe_epoch: EpochId(1),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: vd_wire::intershard::RE_HOME_STEP,
            state: vd_wire::intershard::ReHomeState::PoseOnly(pose()),
            source: NodeId(2),
        }),
        // R-6d3c arm: the orch→dest DISCARD-poison (SIDE-EFFECTING, TransferStep-keyed by
        // TRANSIENT_DISCARD_STEP; shares `TransientHandoff` — classified ReDriven, NOT producer-less,
        // so the golden pin below still asserts exactly TWO producer-less arms).
        InterShardFlow::TransientDiscard(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::TRANSIENT_DISCARD_STEP,
            fence: Fence(6),
        }),
        // CA-1 S3 arm: the orch→source AwaitAdopt liveness probe (SIDE-EFFECTING, TransferStep-keyed by
        // RE_SOLICIT_STEP; shares `TransientHandoff`). Classified ReDriven, NOT producer-less — so the
        // golden pin below still asserts exactly TWO producer-less arms.
        InterShardFlow::ReSolicitBatch(vd_wire::intershard::TransientHandoff {
            transfer: TransferId(10),
            step_id: vd_wire::intershard::RE_SOLICIT_STEP,
            fence: Fence(6),
        }),
        // Slice 3c arms (spatial transfer-trigger, INERT — planted; consumer routes land later). The two
        // crossing REQUESTS carry NO TransferId (the orch mints it), so they key idempotency on a NON-GENESIS
        // subject/src-realm Fence (`FencedKey` — the roundtrip test asserts `!= Fence::GENESIS`). Classified
        // ReDriven, so the producer-less golden pin below still asserts exactly TWO producer-less arms.
        InterShardFlow::CrossingRequest(vd_wire::intershard::CrossingRequest {
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            from_realm: RealmId::System(1),
            to_realm: RealmId::Planet(2),
            subject_fence: Fence(7),
            session: SessionId(3),
            attempt: 0,
            // The parent-provenance the SOURCE fills for the dest realm's frame (Some here exercises the
            // appended-field roundtrip; None is the non-Area default).
            to_parent: Some(RealmId::System(1)),
        }),
        InterShardFlow::TransientCrossingRequest(vd_wire::intershard::TransientCrossingRequest {
            subject: DirectoryKey::Entity(eid(EntityKind::Debris)),
            from_realm: RealmId::System(1),
            to_realm: RealmId::Planet(2),
            src_realm_fence: Fence(7),
            to_parent: Some(RealmId::System(1)),
        }),
        // The GRANT + ABORTED carry a real minted TransferId (keyed on `(transfer, TRANSIENT_BATCH_STEP)`).
        InterShardFlow::TransientCrossingGrant(vd_wire::intershard::TransientCrossingGrant {
            subject: DirectoryKey::Entity(eid(EntityKind::Debris)),
            dest: NodeId(3),
            to_realm: RealmId::Planet(2),
            dst_realm_fence: Fence(4),
            batch: TransferId(10),
            to_parent: Some(RealmId::System(1)),
        }),
        InterShardFlow::CrossingAborted(vd_wire::intershard::CrossingAborted {
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            transfer: TransferId(10),
        }),
        // Slice 3f arm: the source→orch latch-clear CONFIRM (SIDE-EFFECTING, `(transfer,
        // TRANSIENT_BATCH_STEP)`-keyed — the SAME journal as the `CrossingAborted` it answers). ReDriven,
        // NOT producer-less — so the golden pin below still asserts exactly TWO producer-less arms.
        InterShardFlow::CrossingAbortedAck(vd_wire::intershard::CrossingAborted {
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            transfer: TransferId(10),
        }),
        // RLM Step 1 arm: the realm-lifecycle demand. Universe-rooted child path (not a bare
        // 1-level — no aliasing), non-`GENESIS` parent fence (the roundtrip asserts `!= GENESIS`
        // for `FencedKey`). SIDE-EFFECTING (FencedKey{parent_fence}) / ReDriven — NOT producer-less.
        InterShardFlow::RealmDemand(vd_wire::intershard::RealmDemand {
            child: demand_child_coord(),
            parent_fence: Fence(4),
            verb: vd_wire::intershard::DemandVerb::SpinUp,
            universe_tick: UniverseTick(9),
        }),
        // RLM reactive greeting (shard→gateway). FireAndForget / ReDriven — NOT producer-less, so the
        // golden pin below still asserts exactly TWO producer-less arms. Minimal payload (a local tick).
        InterShardFlow::ShardPresence(vd_wire::intershard::ShardPresence {
            local_tick: vd_core::TickId(11),
        }),
        // ★TOMBSTONE (minor 12): the per-occupant position up-flow — no producer, no consumer; the
        // discriminant is reserved forever, so its SHAPE stays pinned here (a drifted tombstone would
        // silently re-label every later arm).
        InterShardFlow::OccupantInterest(vd_wire::intershard::OccupantInterest {
            observer: AccountId(5),
            to_realm: demand_child_coord(),
            occupant: pose(),
            coarsen_level: 0,
        }),
        // ★TOMBSTONE (minor 12): the per-occupant sibling-scene reflection — replaced by the
        // child-keyed `ChildSceneSet`. Shape pinned for the same reserved-discriminant reason.
        InterShardFlow::ProxySceneSet(vd_wire::intershard::ProxySceneSet {
            observer: AccountId(5),
            realms: vec![],
        }),
        // ★TOMBSTONE (window lane Slice C2, minor 19): the parent's down-cascade of restated scenery
        // — no producer, no consumer; discriminant 26 is reserved forever, so its SHAPE stays pinned
        // here (a drifted tombstone would silently re-label every later arm). Every chain level now
        // states its own rows STRAIGHT to the observer's gateway (`ShardToGateway::WindowFrame`).
        // Classification frozen: FireAndForget / Unreliable, NOT producer-less-reliable — the
        // durability golden pin below still asserts exactly TWO producer-less-reliable arms.
        InterShardFlow::RealmCascade(vd_wire::intershard::RealmCascade {
            child: demand_child_coord(),
            realm_snapshot_bytes: vec![1, 2, 3],
        }),
        // ★TOMBSTONE (Step 5 slice E, minor 13) — the entity lane's UP leg. Nothing produces it; the
        // discriminant is reserved forever, so its SHAPE stays pinned here (a drifted tombstone would
        // silently re-label every later arm). Classification frozen: FireAndForget / Unreliable, NOT
        // producer-less — the golden pin below still asserts exactly TWO.
        InterShardFlow::EntityInterest(vd_wire::intershard::EntityRelay {
            realm: demand_child_coord(),
            frame: FrameRef::SystemSpace { system_seed: 1 },
            frame_id: 5,
            universe_tick: UniverseTick(10),
            entities: vec![vd_wire::channels::EntitySnap {
                entity: eid(EntityKind::Player),
                pose: pose(),
            }],
        }),
        // ★TOMBSTONE (Step 5 slice E, minor 13) — the entity lane's DOWN leg; same reservation, same
        // frozen classification as its up twin above.
        InterShardFlow::EntityCascade(vd_wire::intershard::EntityRelay {
            realm: demand_child_coord(),
            frame: FrameRef::SystemSpace { system_seed: 1 },
            frame_id: 6,
            universe_tick: UniverseTick(11),
            entities: vec![vd_wire::channels::EntitySnap {
                entity: eid(EntityKind::Player),
                pose: pose(),
            }],
        }),
        // The orchestrator's answer to "whose frames may a router read": the nodes the ownership record
        // shows holding a realm. A LEVEL — two nodes here, ascending, so the round-trip also pins that a
        // roster's encoding is one sequence of bytes for one set.
        InterShardFlow::ShardRoster(vd_wire::intershard::ShardRoster {
            nodes: vec![NodeId(1002), NodeId(1004)],
            at: UniverseTick(12),
        }),
        // The SL7 occupancy bit: one heartbeat, presence-is-the-bit, fence+tick ordering guards only.
        InterShardFlow::ChildLive(vd_wire::intershard::ChildLive {
            child: demand_child_coord(),
            fence: Fence(3),
            at: UniverseTick(13),
        }),
        // ★TOMBSTONE (window lane Slice C2, minor 19): the up-observation lane — no producer, no
        // consumer; discriminant 31 is reserved forever, so its SHAPE stays pinned here. Its cargo
        // is carried now by the direct `WindowFrame` (own rows to the gateway) and, for a realm an
        // observer is beside rather than inside, by the SEALED `WindowRelay` below. Classification
        // frozen: FireAndForget / Unreliable, NOT producer-less-reliable.
        InterShardFlow::RealmObservation(vd_wire::intershard::RealmObservation {
            child: demand_child_coord(),
            realm_snapshot_bytes: vec![7, 7, 7],
        }),
        // ★TOMBSTONE (window lane Slice C1, minor 17): the interim shape lane — no producer, no
        // consumer; the discriminant (32) is reserved forever, so its SHAPE stays pinned here (a
        // drifted tombstone would silently re-label every later arm). Its content evolved into
        // `WindowRelay` below (owner-approved 2026-08-16, owner_decisions_2026-08-15.md addendum +
        // window_lane.md §5 RULINGS). Classification frozen: FireAndForget / Unreliable, NOT
        // producer-less — the golden pin below still asserts exactly TWO.
        InterShardFlow::RealmShapeObservation(vd_wire::intershard::RealmShapeObservation {
            child: demand_child_coord(),
            shapes: vec![shape()],
        }),
        // ★TOMBSTONE (window lane Slice C2, minor 19): the per-live-child down-reflect — no producer,
        // no consumer; discriminant 33 is reserved forever, so its SHAPE stays pinned here. This is
        // the lane the SL1 SELF-PLACEMENT FILTER guarded; the filter retires WITH it by the owner's
        // Q3 amendment (window_lane.md §5 RULINGS), replaced by the structural absence pinned in
        // `no_realm_inbound_payload_carries_a_placement_or_a_centre` below. Classification frozen:
        // FireAndForget / ReDriven, NOT producer-less-reliable.
        InterShardFlow::ChildSceneSet(vd_wire::intershard::ChildSceneSet {
            child: demand_child_coord(),
            realms: vec![shape()],
        }),
        // THE Q2 RELAY LEG (window lane Slice C1, minor 17; owner-approved 2026-08-16 —
        // owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS): a live child's
        // VERBATIM self-authored statements, SEALED (the parent forwards these bytes unopened —
        // forward-or-drop), the child's own fence riding outside the seal. FireAndForget /
        // ReDriven — NOT producer-less, so the golden pin below still asserts exactly TWO.
        // Mesh minor 20 (look horizon slice 3, owner-approved 2026-08-17 — look_horizon.md
        // RULINGS + §2 ASK A) renamed `statements` → `own` IN PLACE and APPENDED `interior`:
        // held grandchild batches forwarded sealed, each fenced outside its seal. The fixture
        // carries one so the round-trip pins every field; `InteriorRelay` itself has NO
        // `interior` field — the depth bound is the type.
        InterShardFlow::WindowRelay(vd_wire::intershard::WindowRelay {
            child: demand_child_coord(),
            realm_fence: Fence(6),
            own: vd_wire::session_flow::seal_relay_statements(&[
                vd_wire::session_flow::RelayedStatement::Body {
                    subject: RealmId::System(7),
                    stmt: vd_wire::session_flow::BodyStmt::SelfLook { bag: vec![4, 2] },
                    authored_at: UniverseTick(14),
                },
            ]),
            interior: vec![vd_wire::intershard::InteriorRelay {
                child: RealmId::Planet(3),
                child_fence: Fence(7),
                own: vd_wire::session_flow::seal_relay_statements(&[
                    vd_wire::session_flow::RelayedStatement::Body {
                        subject: RealmId::Planet(3),
                        stmt: vd_wire::session_flow::BodyStmt::SelfLook { bag: vec![5, 1] },
                        authored_at: UniverseTick(15),
                    },
                ]),
            }],
        }),
        // THE INTEREST BIT (look horizon slice 4, minor 21; Q1 APPROVED, owner-approved
        // 2026-08-17 Q1 — look_horizon.md §2 ASK B): one byte, two lawful values, parent → ONE
        // direct child. By construction of the type there is no field a position, identity,
        // direction, distance or count could ride in — the absence pin below covers it the day
        // it lands. FireAndForget / ReDriven — NOT producer-less, so the golden pin still
        // asserts exactly TWO.
        InterShardFlow::RealmInterest(vd_wire::intershard::RealmInterest {
            child: demand_child_coord(),
            parent_fence: Fence(8),
            at: UniverseTick(16),
            look_inside: 1,
        }),
    ]
}

/// One public-geometry outline for the two minor-11 shape-lane fixtures — small but field-complete,
/// so the round-trip pins every field of the shared `RealmShape` payload on both new arms.
fn shape() -> vd_wire::channels::RealmShape {
    vd_wire::channels::RealmShape {
        realm: RealmId::Planet(3),
        frame: FrameRef::PlanetCentered { planet_seed: 3 },
        shape: vd_core::geometry::Boundary::Shell { r: 2.0 },
        parent: Some(RealmId::System(7)),
    }
}

/// A Universe-rooted `[Universe, Galaxy, System]` child coord for the `RealmDemand` fixture — a
/// lineage-rooted path (not a bare 1-level), so the globally-unique-path invariant holds.
fn demand_child_coord() -> RealmCoord {
    RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::Universe, 0),
        RealmLevel::new(RealmKindTag::Galaxy, 2),
        RealmLevel::new(RealmKindTag::System, 7),
    ]))
    .expect("3-level path has a leaf")
}

/// Compile-time tripwire: adding an `InterShardFlow` arm MUST break this exhaustive
/// match (no wildcard), forcing the author back here to also represent it in
/// [`every_arm`] above — so the per-release surface gate can never silently omit an arm
/// (audit HR1-1). This is never called; its body is the structural assertion.
#[allow(dead_code)]
fn arm_tripwire(flow: &InterShardFlow) {
    match flow {
        InterShardFlow::Ghost(_)
        | InterShardFlow::Transfer(_)
        | InterShardFlow::Directory(_)
        | InterShardFlow::Saga(_)
        | InterShardFlow::SagaAck(_)
        | InterShardFlow::DirectoryReply(_)
        | InterShardFlow::FlushSource(_)
        | InterShardFlow::TransferAck(_)
        | InterShardFlow::Demote(_)
        | InterShardFlow::Promote(_)
        | InterShardFlow::TransientRelease(_)
        | InterShardFlow::TransientDrop(_)
        | InterShardFlow::ReleaseComplete(_)
        | InterShardFlow::TransientAbandon(_)
        | InterShardFlow::ReHome(_)
        | InterShardFlow::TransientDiscard(_)
        | InterShardFlow::ReSolicitBatch(_)
        | InterShardFlow::CrossingRequest(_)
        | InterShardFlow::TransientCrossingRequest(_)
        | InterShardFlow::TransientCrossingGrant(_)
        | InterShardFlow::CrossingAborted(_)
        | InterShardFlow::CrossingAbortedAck(_)
        | InterShardFlow::RealmDemand(_)
        | InterShardFlow::ShardPresence(_)
        | InterShardFlow::OccupantInterest(_)
        | InterShardFlow::ProxySceneSet(_)
        | InterShardFlow::RealmCascade(_)
        | InterShardFlow::EntityInterest(_)
        | InterShardFlow::EntityCascade(_)
        | InterShardFlow::ShardRoster(_)
        | InterShardFlow::ChildLive(_)
        | InterShardFlow::RealmObservation(_)
        | InterShardFlow::RealmShapeObservation(_)
        | InterShardFlow::ChildSceneSet(_)
        | InterShardFlow::WindowRelay(_)
        | InterShardFlow::RealmInterest(_) => {}
    }
}

/// NESTED-VARIANT tripwires (R-6d2b review; completed to ALL EIGHT nested enums by the Stage-C audit,
/// finding :650). `arm_tripwire` is exhaustive only at the OUTER `InterShardFlow`
/// level (`Transfer(_)`/`Ghost(_)`), so a new NESTED variant of any payload enum — the natural
/// shape of a FUTURE producer-less durable flow (P9 cross-shard Signal, P6 durable BlockEdit forward) —
/// would NOT break compilation, would be omitted from `every_arm`, and would slip the `durability_class`
/// golden pin (which iterates `every_arm`) VACUOUSLY: it would ship pushed with the `Durability::Ephemeral`
/// default and be silently lost on a source crash (D-6 #1) with a GREEN suite. These wildcard-free matches
/// force a new nested variant to FAIL COMPILATION here until it is classified — an eyes-open edit that (with
/// the golden pin + the stub marker test it points at) forces representing it in `every_arm` and verifying
/// its push-site `Retained` marker. This is the exhaustive-by-construction guard the per-send-`Durability`
/// default relies on instead of the vetted explicit-4-arg. Never called; the body is the assertion.
/// The eight: `TransitionPayload`, `GhostFlow`, `DemandVerb` (below), plus `DirectoryOp`,
/// `DirectoryReply`, `TransferControl`, `TransferControlAck`, `TransferAck` (further below — the five
/// whose only production classification is `durability_class`'s blanket `ReDriven` group).
#[allow(dead_code)]
fn payload_tripwire(p: &TransitionPayload) {
    match p {
        TransitionPayload::InitialSpawn { .. }
        | TransitionPayload::StubCrossing { .. }
        | TransitionPayload::TransientBatch { .. } => {}
    }
}

#[allow(dead_code)]
fn ghost_tripwire(g: &GhostFlow) {
    match g {
        GhostFlow::Spawn { .. }
        | GhostFlow::Delta { .. }
        | GhostFlow::Despawn { .. }
        | GhostFlow::SpawnV2 { .. } => {}
    }
}

/// NESTED-VARIANT tripwire (RLM Step 1): a new `DemandVerb` must be enumerated here or fail to
/// compile — the same guard `payload_tripwire`/`ghost_tripwire` give the nested Transfer/Ghost
/// payloads (a 5th verb would otherwise slip `every_arm` + the golden pin vacuously). Never called.
#[allow(dead_code)]
fn demand_verb_tripwire(v: &vd_wire::intershard::DemandVerb) {
    use vd_wire::intershard::DemandVerb;
    match v {
        DemandVerb::SpinUp | DemandVerb::KeepAlive | DemandVerb::Empty | DemandVerb::TearDown => {}
    }
}

// NESTED-VARIANT tripwires for the five payload enums the wildcard `Directory(_) | Saga(_) |
// SagaAck(_) | DirectoryReply(_) | TransferAck(_)` group in `durability_class` never destructures
// (audit :650 — only 3 of the 8 nested enums carried one). Four of the five would still break SOME
// compile elsewhere (their own wildcard-free accessors); `DirectoryReply` had NO wildcard-free match
// anywhere — a new variant compiled, skipped `every_arm`, and rode the blanket `ReDriven` default
// with a GREEN suite, the exact vacuity the header above describes. Same discipline as
// `payload_tripwire`: never called; the wildcard-free match IS the assertion, forcing the author
// back to `every_arm` (and the durability golden pin) for every new nested variant.

#[allow(dead_code)]
fn directory_op_tripwire(op: &DirectoryOp) {
    match op {
        DirectoryOp::LeaseGrant { .. }
        | DirectoryOp::LeaseRenew { .. }
        | DirectoryOp::LeaseRevoke { .. }
        | DirectoryOp::CommitCas { .. }
        | DirectoryOp::AbortCas { .. }
        | DirectoryOp::HeadRead { .. }
        | DirectoryOp::ClockSync { .. } => {}
    }
}

#[allow(dead_code)]
fn directory_reply_tripwire(r: &DirectoryReply) {
    match r {
        DirectoryReply::Head { .. }
        | DirectoryReply::CasResult { .. }
        | DirectoryReply::ClockNow { .. } => {}
    }
}

#[allow(dead_code)]
fn transfer_control_tripwire(c: &TransferControl) {
    match c {
        TransferControl::PrepareSubscribe { .. }
        | TransferControl::RequestCut { .. }
        | TransferControl::FreezeSource { .. }
        | TransferControl::CommitAuthority { .. }
        | TransferControl::ThawSource { .. }
        | TransferControl::AbortTransfer { .. }
        | TransferControl::ReleaseSubscribe { .. } => {}
    }
}

#[allow(dead_code)]
fn transfer_control_ack_tripwire(a: &TransferControlAck) {
    match a {
        TransferControlAck::Prepared { .. }
        | TransferControlAck::CutConfirmed { .. }
        | TransferControlAck::SourceFrozen { .. }
        | TransferControlAck::Committed { .. }
        | TransferControlAck::SourceThawed { .. }
        | TransferControlAck::Aborted { .. }
        | TransferControlAck::Released { .. }
        | TransferControlAck::DemoteAck { .. }
        | TransferControlAck::PromoteAck { .. }
        | TransferControlAck::DeliveredToObservers { .. } => {}
    }
}

#[allow(dead_code)]
fn transfer_ack_tripwire(a: &TransferAck) {
    match a {
        TransferAck::SourceFlushed { .. }
        | TransferAck::Accepted { .. }
        | TransferAck::Rejected { .. }
        | TransferAck::BatchAdopted { .. }
        | TransferAck::DropApplied { .. } => {}
    }
}

/// THE POSITIONAL PIN the tombstone discipline rests on. Postcard writes a variant's DECLARED
/// index as the envelope's leading varint, so reordering the enum — or deleting a tombstoned
/// arm — re-labels every later arm ON THE WIRE while every same-build roundtrip in this file
/// stays green (encode and decode share the drifted table). The match below is the declaration
/// order stated ONCE as data and asserted against the first byte of the REAL encoding: a drift
/// fails here, in one test, instead of silently in production decode. Wildcard-free, so a new
/// arm must take a pinned index to compile.
#[test]
fn every_arm_encodes_its_declared_discriminant_index() {
    fn declared_index(flow: &InterShardFlow) -> u8 {
        match flow {
            InterShardFlow::Ghost(_) => 0,
            InterShardFlow::Transfer(_) => 1,
            InterShardFlow::Directory(_) => 2,
            InterShardFlow::Saga(_) => 3,
            InterShardFlow::SagaAck(_) => 4,
            InterShardFlow::DirectoryReply(_) => 5,
            InterShardFlow::FlushSource(_) => 6,
            InterShardFlow::TransferAck(_) => 7,
            InterShardFlow::Demote(_) => 8,
            InterShardFlow::Promote(_) => 9,
            InterShardFlow::TransientRelease(_) => 10,
            InterShardFlow::TransientDrop(_) => 11,
            InterShardFlow::ReleaseComplete(_) => 12,
            InterShardFlow::TransientAbandon(_) => 13,
            InterShardFlow::ReHome(_) => 14,
            InterShardFlow::TransientDiscard(_) => 15,
            InterShardFlow::ReSolicitBatch(_) => 16,
            InterShardFlow::CrossingRequest(_) => 17,
            InterShardFlow::TransientCrossingRequest(_) => 18,
            InterShardFlow::TransientCrossingGrant(_) => 19,
            InterShardFlow::CrossingAborted(_) => 20,
            InterShardFlow::CrossingAbortedAck(_) => 21,
            InterShardFlow::RealmDemand(_) => 22,
            InterShardFlow::ShardPresence(_) => 23,
            // The two Step 5 slice D tombstones hold 24 and 25 forever.
            InterShardFlow::OccupantInterest(_) => 24,
            InterShardFlow::ProxySceneSet(_) => 25,
            // The window lane's Slice C2 tombstone holds 26 forever (the down-cascade).
            InterShardFlow::RealmCascade(_) => 26,
            InterShardFlow::EntityInterest(_) => 27,
            InterShardFlow::EntityCascade(_) => 28,
            InterShardFlow::ShardRoster(_) => 29,
            InterShardFlow::ChildLive(_) => 30,
            // The window lane's Slice C2 tombstone holds 31 forever (the up-observation ship).
            InterShardFlow::RealmObservation(_) => 31,
            // The window lane's Slice C1 tombstone holds 32 forever.
            InterShardFlow::RealmShapeObservation(_) => 32,
            // The window lane's Slice C2 tombstone holds 33 forever (the down-reflect).
            InterShardFlow::ChildSceneSet(_) => 33,
            // The Q2 relay leg (mesh minor 17) holds 34 forever.
            InterShardFlow::WindowRelay(_) => 34,
            InterShardFlow::RealmInterest(_) => 35,
        }
    }
    // Every fixture's real leading byte matches its declared index (all indices < 128, so the
    // varint IS the index byte)…
    let mut seen = std::collections::BTreeSet::new();
    for flow in every_arm() {
        let bytes = postcard::to_allocvec(&flow).expect("closed arm encodes");
        assert_eq!(bytes[0], declared_index(&flow));
        seen.insert(bytes[0]);
    }
    // …and the fixture set spans the WHOLE contiguous index space, so a missing fixture (or a
    // gap postcard would assign past a deleted arm) cannot pass vacuously.
    assert_eq!(seen.len(), 36);
    assert_eq!(seen.first().copied(), Some(0));
    assert_eq!(seen.last().copied(), Some(35));
}

/// THE TOMBSTONE GOLDEN SET — the discriminants no producer may ever fill again, stated as data.
///
/// Each entry was a lane somebody deleted on purpose, and each entry's number is load-bearing
/// forever: postcard writes a variant's DECLARED index, so reviving one, renumbering one, or
/// quietly deleting one re-labels every later arm on a live wire. The classifier below is an
/// EXHAUSTIVE match, so a new arm cannot be added without a stated tombstone answer, and the
/// golden count makes a silent revival (a tombstone that grows a producer again) a failing test
/// rather than a review finding.
///
/// Window lane Slice C2 (minor 19, owner-approved 2026-08-16 — `docs/design/window_lane.md`
/// §2.5 + §5 RULINGS) grew the set from FIVE to EIGHT: the three old inter-realm SCENERY lanes
/// (`RealmCascade` 26, `RealmObservation` 31, `ChildSceneSet` 33) joined the interim shape lane
/// (`RealmShapeObservation` 32, minor 17), the two per-occupant lanes (24/25, minor 12) and the
/// entity lane's two legs (27/28, minor 13).
#[test]
fn the_tombstoned_discriminants_are_exactly_this_golden_set() {
    /// Is this arm's lane DELETED (no producer anywhere in the workspace)? Exhaustive by
    /// construction — a new arm does not compile until it answers.
    fn tombstoned(flow: &InterShardFlow) -> bool {
        match flow {
            // ★ The eight deleted lanes.
            InterShardFlow::OccupantInterest(_)          // minor 12, per-occupant pose up-relay
            | InterShardFlow::ProxySceneSet(_)           // minor 12, per-occupant scene reflect
            | InterShardFlow::RealmCascade(_)            // minor 19, the down-cascade of scenery
            | InterShardFlow::EntityInterest(_)          // minor 13, entity lane up-leg
            | InterShardFlow::EntityCascade(_)           // minor 13, entity lane down-leg
            | InterShardFlow::RealmObservation(_)        // minor 19, the up-observation ship
            | InterShardFlow::RealmShapeObservation(_)   // minor 17, the interim shape lane
            | InterShardFlow::ChildSceneSet(_) => true,  // minor 19, the down-reflect
            // Everything else is LIVING: the transfer/saga/directory machinery, the roster, the
            // SL7 occupancy bit, the demand verbs, the reactive greeting, and the window lane's
            // own Q2 relay leg.
            InterShardFlow::Ghost(_)
            | InterShardFlow::Transfer(_)
            | InterShardFlow::Directory(_)
            | InterShardFlow::Saga(_)
            | InterShardFlow::SagaAck(_)
            | InterShardFlow::DirectoryReply(_)
            | InterShardFlow::FlushSource(_)
            | InterShardFlow::TransferAck(_)
            | InterShardFlow::Demote(_)
            | InterShardFlow::Promote(_)
            | InterShardFlow::TransientRelease(_)
            | InterShardFlow::TransientDrop(_)
            | InterShardFlow::ReleaseComplete(_)
            | InterShardFlow::TransientAbandon(_)
            | InterShardFlow::ReHome(_)
            | InterShardFlow::TransientDiscard(_)
            | InterShardFlow::ReSolicitBatch(_)
            | InterShardFlow::CrossingRequest(_)
            | InterShardFlow::TransientCrossingRequest(_)
            | InterShardFlow::TransientCrossingGrant(_)
            | InterShardFlow::CrossingAborted(_)
            | InterShardFlow::CrossingAbortedAck(_)
            | InterShardFlow::RealmDemand(_)
            | InterShardFlow::ShardPresence(_)
            | InterShardFlow::ShardRoster(_)
            | InterShardFlow::ChildLive(_)
            | InterShardFlow::WindowRelay(_)
            | InterShardFlow::RealmInterest(_) => false,
        }
    }
    let mut reserved: Vec<u8> = every_arm()
        .iter()
        .filter(|f| tombstoned(f))
        .map(|f| postcard::to_allocvec(f).expect("closed arm encodes")[0])
        .collect();
    reserved.sort_unstable();
    assert_eq!(
        reserved,
        vec![24, 25, 26, 27, 28, 31, 32, 33],
        "the reserved-forever discriminant set moved: a tombstone was revived, renumbered or \
         deleted — every one of those re-labels a later arm on a live wire"
    );
}

/// THE STRUCTURAL SUCCESSOR OF THE SL1 SELF-PLACEMENT FILTER (owner ruling Q3, 2026-08-16 —
/// `docs/design/window_lane.md` §5 RULINGS; the C2 ledger cites it as retirement BY AMENDMENT,
/// not erosion).
///
/// A realm may never learn where it sits. Until Slice C2 that was guarded BEHAVIOURALLY: the
/// down-reflect lane filtered the sender's own outline out of the set it shipped to a child,
/// because restated one hop down that outline states −(the child's placement) — the child's own
/// address, sign-flipped. The owner approved deleting that filter TOGETHER WITH ITS LANE, on the
/// ground that the hazard itself ceases to exist. This test is what makes that claim a
/// measurement instead of an argument: **no LIVING inter-shard payload can carry a realm's
/// placement at all** — not as a typed field, not as a centre, and not as the opaque
/// pre-serialized datagram the deleted lanes used to hide one inside.
///
/// It is a source scan of the wire crate's own vocabulary (the same discipline the version
/// ledger's citation gate uses), and it can FAIL — the tombstoned payloads still declare those
/// very fields, and the positive control below proves the detector sees them.
#[test]
fn no_living_inter_shard_payload_carries_a_realm_placement_or_a_centre() {
    /// A field type that expresses WHERE A REALM IS. An occupant's own `StampedPose` is
    /// deliberately NOT here: the entity lane is untouched, and a crossing subject's pose is
    /// SL2's permitted case (it rides WITH the subject, about the subject).
    const PLACEMENT_TYPES: &[&str] = &[
        "RealmShape",
        "RealmSnap", // also catches `RealmSnapshotDatagram`
        "FramePlacement",
    ];
    /// Field NAMES that smuggle the same thing past the type check — the opaque
    /// already-serialized datagram the two deleted relay lanes carried, and any centre.
    const PLACEMENT_NAMES: &[&str] = &["center", "centre", "placement", "realm_snapshot_bytes"];
    /// The payload structs whose lane is DELETED. They still declare placement fields — that is
    /// the point of a tombstone (the shape stays pinned) — and they are also this test's proof
    /// that the scan below can fail.
    const TOMBSTONED: &[&str] = &[
        "OccupantInterest",
        "ProxySceneSet",
        "RealmCascade",
        "EntityRelay",
        "RealmObservation",
        "RealmShapeObservation",
        "ChildSceneSet",
    ];

    /// Parse `pub struct NAME { … }` blocks into `(struct, Vec<(field name, field type)>)`.
    /// Doc comments and attributes are skipped; the closing brace at column 0 ends a block.
    fn declared_structs(source: &str) -> Vec<(String, Vec<(String, String)>)> {
        let mut out: Vec<(String, Vec<(String, String)>)> = Vec::new();
        let mut open: Option<(String, Vec<(String, String)>)> = None;
        for line in source.lines() {
            if let Some(rest) = line.strip_prefix("pub struct ") {
                let name = rest
                    .split([' ', '{', '<', '('])
                    .next()
                    .unwrap_or_default()
                    .to_owned();
                open = Some((name, Vec::new()));
                continue;
            }
            let Some((_, fields)) = open.as_mut() else {
                continue;
            };
            if line == "}" {
                out.push(open.take().expect("a block was open"));
                continue;
            }
            let trimmed = line.trim();
            let Some(decl) = trimmed.strip_prefix("pub ") else {
                continue; // a doc line, an attribute, or a private field
            };
            let Some((name, ty)) = decl.split_once(':') else {
                continue;
            };
            fields.push((
                name.trim().to_owned(),
                ty.trim().trim_end_matches(',').to_owned(),
            ));
        }
        out
    }

    /// Does this field state where a REALM is?
    fn is_placement(name: &str, ty: &str) -> bool {
        PLACEMENT_TYPES.iter().any(|t| ty.contains(t))
            | PLACEMENT_NAMES.iter().any(|n| name.contains(n))
    }

    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/intershard.rs"),
    )
    .expect("the wire crate's own source is readable");
    let structs = declared_structs(&source);

    // ANTI-VACUITY (1): the scan really read the file, and really read fields.
    assert!(
        structs.len() >= 20,
        "the struct scan found almost nothing — the parser drifted off the source: {structs:?}"
    );
    assert!(
        structs.iter().all(|(_, fields)| !fields.is_empty()),
        "a scanned struct came back field-less — the parser drifted: {structs:?}"
    );

    // ANTI-VACUITY (2), the POSITIVE CONTROL: the detector fires on the tombstones. If this ever
    // goes quiet the assert below has stopped being able to fail.
    let mut caught: Vec<&str> = structs
        .iter()
        .filter(|(name, fields)| {
            TOMBSTONED.contains(&name.as_str()) & fields.iter().any(|(n, t)| is_placement(n, t))
        })
        .map(|(name, _)| name.as_str())
        .collect();
    caught.sort_unstable();
    assert_eq!(
        caught,
        vec![
            "ChildSceneSet",
            "ProxySceneSet",
            "RealmCascade",
            "RealmObservation",
            "RealmShapeObservation",
        ],
        "the placement detector stopped seeing the placements the DEAD lanes still declare — \
         the living-lane assert below would then pass vacuously. (The other two tombstones, \
         `OccupantInterest` and `EntityRelay`, carry OCCUPANT poses rather than realm \
         placements, and are rightly not caught here — they died for SL2, not for SL1.)"
    );

    // THE PIN: no LIVING payload states where a realm is.
    let offenders: Vec<(String, String, String)> = structs
        .iter()
        .filter(|(name, _)| !TOMBSTONED.contains(&name.as_str()))
        .flat_map(|(name, fields)| {
            fields
                .iter()
                .filter(|(n, t)| is_placement(n, t))
                .map(move |(n, t)| (name.clone(), n.clone(), t.clone()))
        })
        .collect();
    assert_eq!(
        offenders,
        Vec::new(),
        "a LIVING inter-shard message now carries a realm's placement. That is the hazard the \
         SL1 self-placement filter used to catch on one lane; the filter was retired by the \
         owner's Q3 amendment BECAUSE this could no longer happen. Find the local formulation, \
         or take the SL6 ask to the owner (default NO)."
    );
}

/// The NESTED positional pin for `GhostFlow` — the same reorder/deletion hole the outer
/// declared-index table closes for `InterShardFlow`: the tombstoned `Spawn`/`Delta` hold nested
/// slots 0/1 forever, and postcard writes the nested index right after the outer `Ghost` tag, so
/// the second byte of the real encoding is asserted against the declaration order stated as data.
#[test]
fn ghost_flow_encodes_its_declared_nested_discriminant_index() {
    fn declared_index(g: &GhostFlow) -> u8 {
        match g {
            GhostFlow::Spawn { .. } => 0,
            GhostFlow::Delta { .. } => 1,
            GhostFlow::Despawn { .. } => 2,
            GhostFlow::SpawnV2 { .. } => 3,
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for flow in every_arm() {
        let InterShardFlow::Ghost(g) = &flow else {
            continue;
        };
        let bytes = postcard::to_allocvec(&flow).expect("closed arm encodes");
        assert_eq!(bytes[0], 0, "Ghost holds outer slot 0");
        assert_eq!(bytes[1], declared_index(g));
        seen.insert(bytes[1]);
    }
    // The fixture set spans the whole nested index space — a missing fixture cannot pass vacuously.
    assert_eq!(seen.len(), 4);
    assert_eq!(seen.first().copied(), Some(0));
    assert_eq!(seen.last().copied(), Some(3));
}

#[test]
fn every_arm_roundtrips_postcard_and_classifies_coherently() {
    for flow in every_arm() {
        // 1. Closed wire: every arm serializes and decodes byte-identically.
        let bytes = postcard::to_allocvec(&flow).expect("closed arm encodes");
        let back: InterShardFlow = postcard::from_bytes(&bytes).expect("closed arm decodes");
        assert_eq!(back, flow);

        // 2. G-SEALED: side-effecting arms MUST carry an idempotency key; nothing
        //    that mutates authority may ride a fire-and-forget channel.
        match flow.effect_class() {
            EffectClass::SideEffecting { idempotency } => match idempotency {
                IdempotencyKey::TransferStep { transfer, .. }
                | IdempotencyKey::FencedCas { transfer, .. } => {
                    assert_ne!(transfer, TransferId(0), "a real correlation id");
                }
                IdempotencyKey::FencedKey { fence } => {
                    assert_ne!(fence, Fence::GENESIS, "a real fence");
                }
            },
            EffectClass::FireAndForget => {
                // Fire-and-forget arms are re-derivable / latest-wins by construction
                // (ghost deltas, head reads). The exhaustive match in effect_class is
                // what makes this assertion total.
            }
        }
    }
}

#[test]
fn durability_class_pins_the_producer_less_reliable_set() {
    // R-6d §7 conformance: the wildcard-free `durability_class` match makes the classification TOTAL (a new
    // arm / GhostFlow / TransitionPayload variant fails to compile until classified). This golden pin asserts
    // the PRODUCER-LESS-RELIABLE set — the arms whose `push_flow` site MUST carry `Durability::Retained`, or
    // the one-shot is silently lost on a source crash — is EXACTLY {Ghost::Despawn, Ghost::SpawnV2,
    // Transfer(TransientBatch)} (SpawnV2 joined at slice F: the take-over proof is a one-shot at promote
    // with no re-driver — a promote redelivery re-acks without re-spawning). Growing it is a deliberate
    // edit that trips BOTH this pin AND the marker-on-push test — so a future durable Signal / block-edit
    // forward cannot slip in producer-less without a marker.
    let mut producer_less = Vec::new();
    for flow in every_arm() {
        let expect = match &flow {
            InterShardFlow::Ghost(GhostFlow::Despawn { .. })
            | InterShardFlow::Ghost(GhostFlow::SpawnV2 { .. }) => {
                FlowDurabilityClass::ProducerLessReliable
            }
            // ★TOMBSTONE (slice F) — the dead pose feed's frozen class.
            InterShardFlow::Ghost(GhostFlow::Delta { .. }) => FlowDurabilityClass::Unreliable,
            // VU AoI S2a: the occupant-position up-flow is a latest-wins datagram (Unreliable), NOT
            // producer-less — so the golden `producer_less.len() == 2` pin below is unchanged.
            InterShardFlow::OccupantInterest(_) => FlowDurabilityClass::Unreliable,
            // Per-realm observation cascade: latest-wins realm poses (Unreliable), NOT producer-less — so the
            // golden `producer_less.len() == 2` pin below is unchanged.
            InterShardFlow::RealmCascade(_) => FlowDurabilityClass::Unreliable,
            // ★TOMBSTONE (slice E) — the entity lane's frozen class, both legs: Unreliable, NOT
            // producer-less — so the golden `producer_less.len() == 2` pin below is unchanged.
            InterShardFlow::EntityInterest(_) | InterShardFlow::EntityCascade(_) => {
                FlowDurabilityClass::Unreliable
            }
            // The roster is a LEVEL the reconciler re-pushes: losing one leaves a live shard mute, so it
            // is never Unreliable — and it needs no durable outbox, so it is not producer-less either.
            InterShardFlow::ShardRoster(_) => FlowDurabilityClass::ReDriven,
            // The SL7 bit (TTL-bridged, ancestor_close-backstopped) and the up-observation rows
            // (per-tick latest-wins) — both Unreliable, like the lanes they mirror; the golden
            // `producer_less.len() == 2` pin below is unchanged.
            InterShardFlow::ChildLive(_) | InterShardFlow::RealmObservation(_) => {
                FlowDurabilityClass::Unreliable
            }
            // The interior-outline level (minor 11): re-asserted every AoI cadence, TTL-bridged like the
            // bit it rides beside — Unreliable, NOT producer-less (the golden pin below is unchanged).
            // (Its down-going sibling `ChildSceneSet` keeps ProxySceneSet's ReDriven class and lands in
            // the wildcard arm below, exactly as ProxySceneSet does.)
            InterShardFlow::RealmShapeObservation(_) => FlowDurabilityClass::Unreliable,
            InterShardFlow::Transfer(env)
                if matches!(env.payload, TransitionPayload::TransientBatch { .. }) =>
            {
                FlowDurabilityClass::ProducerLessReliable
            }
            _ => FlowDurabilityClass::ReDriven,
        };
        assert_eq!(
            flow.durability_class(),
            expect,
            "arm misclassified: {flow:?}"
        );
        if flow.durability_class() == FlowDurabilityClass::ProducerLessReliable {
            producer_less.push(flow);
        }
    }
    assert_eq!(
        producer_less.len(),
        3,
        "exactly three producer-less-reliable arms today (Ghost::Despawn + Ghost::SpawnV2 + \
         TransientBatch): {producer_less:?}"
    );
}

#[test]
fn side_effecting_transfer_carries_its_step_idempotency() {
    let durable = InterShardFlow::Transfer(TransferEnvelope {
        transfer_id: TransferId(99),
        universe_epoch: EpochId(1),
        schema_version: 1,
        fence: Fence(2),
        step_id: 7,
        class: DurabilityClass::Durable,
        payload: TransitionPayload::InitialSpawn {
            entity: eid(EntityKind::Player),
            to_realm: RealmId::System(1),
            pose: pose(),
            state: vec![],
        },
    });
    assert_eq!(
        durable.effect_class(),
        EffectClass::SideEffecting {
            idempotency: IdempotencyKey::TransferStep {
                transfer: TransferId(99),
                step_id: 7,
            },
        },
    );
}
