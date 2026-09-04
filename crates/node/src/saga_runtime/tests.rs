//! THE SAGA RUNTIME'S UNIT TIER — the assertions that were the tail of `saga_runtime.rs` until
//! the file was split, moved VERBATIM and named identically (the module path did not change: this
//! is still `saga_runtime::tests`).
//!
//! Owns: the fixtures and the assertions for every lane in the `saga_runtime` tree. It reads
//! `super::*`, so it sees the module's whole re-exported surface exactly as a caller outside the
//! crate does, plus the crate-private items the lanes share.
//!
//! Does NOT own: any production behaviour. None of this is compiled into a shipped orchestrator.

use super::*;
use crate::app::{NodeConfig, build_app};
use crate::orchestrator::DirectoryRes;
use crate::orchestrator::{OrchestratorConfig, register_orchestrator_with_store};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::MsgId;
use vd_core::entity_kind::{DurabilityClass, EntityKind};
use vd_core::glam::DVec3;
use vd_core::pose::StampedPose;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{BatchId, NodeId, TransferId};
use vd_core::{EntityId, EpochId, Fence, SessionId, UniverseTick};
use vd_sim::capability::NodeKind;
use vd_sim::capability::{CapRequest, ShardProfile};
use vd_sim::directory::DirectoryCore;
use vd_sim::directory::DirectoryTuning;
use vd_sim::io::mem::{MemHub, MemStore};
use vd_sim::io::{Inbound, MsgClass, Store};
use vd_sim::io::{ShedReason, Transport};
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::saga::BatchHandoffPhase;
use vd_sim::saga::{self, AbortReason, LivenessTuning, SagaCtx, SagaEvent, SagaState, SagaTuning};
use vd_wire::intershard::TRANSIENT_COMPLETE_STEP;
use vd_wire::intershard::{
    CrossingRequest, DEMOTE_STEP, DemoteCmd, FLUSH_SOURCE_STEP, InterShardFlow, PROMOTE_STEP,
    PromoteCmd, RE_HOME_STEP, RE_SOLICIT_STEP, ReHomeCmd, ReHomeState, STUB_CROSSING_STEP,
    TRANSFER_SCHEMA_VERSION, TRANSIENT_ABANDON_STEP, TRANSIENT_DISCARD_STEP, TRANSIENT_DROP_STEP,
    TRANSIENT_RELEASE_STEP, TransferAck, TransferEnvelope, TransientCrossingGrant,
    TransientCrossingRequest, TransientHandoff, TransitionPayload, crossing_transfer_id,
    namespaced_transfer_id,
};
use vd_wire::seams::directory::AuthorityRef;
use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};
use vd_wire::seams::transfer_control::TransferControlAck;
use vd_wire::seams::transfer_control::{
    PrepareReject, PrepareResult, SpatialReject, TransferControl,
};

const FROM_REALM: RealmId = RealmId::System(7);
const TO_REALM: RealmId = RealmId::System(8);

/// A non-origin pose the source "flushes" — distinct components so a test can confirm the
/// EmitCrossing carried THIS pose (not a default).
///
/// It is ALSO what the crossing/re-home now SHIPS, bit for bit: the saga is a courier and moves no
/// numbers. There used to be a second `dest_flushed_pose()` helper here that re-expressed this pose
/// into the dest's frame with the same relabel the builders applied — a helper whose only job was to
/// agree with the bug, so it could never catch it. Assert against THIS pose directly.
fn flushed_pose() -> StampedPose {
    StampedPose::at_rest(
        FrameRef::SystemSpace { system_seed: 7 },
        DVec3::new(1.0, 2.0, 3.0),
        UniverseTick(5),
    )
}

const ORCH: NodeId = NodeId(1);
const SOURCE: NodeId = NodeId(2);
const DEST: NodeId = NodeId(3);
const GATEWAY: NodeId = NodeId(4);
const XFER: TransferId = TransferId(77);
const SESSION: SessionId = SessionId(5);

fn subject_eid() -> EntityId {
    EntityId::pack(EntityKind::Player, 1, 7, 3)
}

fn subject() -> DirectoryKey {
    DirectoryKey::Entity(subject_eid())
}

/// The wire form of one `TransferControl` command as the gateway receives it from the
/// orchestrator (full-value equality target).
fn saga_wire(cmd: TransferControl) -> Inbound {
    Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes: postcard::to_allocvec(&InterShardFlow::Saga(cmd))
            .expect("encode")
            .into(),
    }
}

/// The wire form of the ordered `Demote` as the SOURCE receives it (1d.5b.1).
fn demote_wire(new_owner_fence: Fence) -> Inbound {
    Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes: postcard::to_allocvec(&InterShardFlow::Demote(vd_wire::intershard::DemoteCmd {
            transfer: XFER,
            subject: subject(),
            new_owner_fence,
            step_id: vd_wire::intershard::DEMOTE_STEP,
        }))
        .expect("encode")
        .into(),
    }
}

/// The wire form of the ordered `Promote` as the DEST receives it (1d.5b.1).
fn promote_wire(new_fence: Fence) -> Inbound {
    Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes: postcard::to_allocvec(&InterShardFlow::Promote(vd_wire::intershard::PromoteCmd {
            transfer: XFER,
            subject: subject(),
            new_fence,
            step_id: vd_wire::intershard::PROMOTE_STEP,
            source: SOURCE,
        }))
        .expect("encode")
        .into(),
    }
}

fn ctx(class: vd_core::entity_kind::DurabilityClass, expected_fence: Fence) -> SagaCtx {
    SagaCtx {
        transfer: XFER,
        session: SESSION,
        subject: subject(),
        expected_fence,
        source: SOURCE,
        dest: DEST,
        class,
        needs_provision: false,
        from_realm: FROM_REALM,
        to_realm: TO_REALM,
        // `TO_REALM` is `System(8)` — a one-field frame, no parent needed.
        to_parent: None,
        exterior: false,
    }
}

/// A TRANSIENT batch saga ctx (D-7): the subject is the dest realm (inert provenance — never
/// enters the directory, since `locks_directory_key(Transient)=false`); `expected_fence` is the
/// dest realm-lease fence the batched go-token commits at; `session` is the session-less sentinel
/// (`SessionId::NONE`) — faithful to production, which builds exactly this ctx in
/// `handle_transient_crossing_request` (D-43 #9). The transient short-path never reads it.
fn transient_ctx(expected_fence: Fence) -> SagaCtx {
    SagaCtx {
        exterior: false,
        subject: DirectoryKey::Realm(TO_REALM),
        session: SessionId::NONE,
        ..ctx(DurabilityClass::Transient, expected_fence)
    }
}

/// The wire form of an orchestrator→shard `InterShardFlow` as the source/dest receive it (D-7b:
/// the structural drop-before-promote `TransientRelease`/`TransientDrop`/`ReleaseComplete`).
fn flow_inbound(flow: &InterShardFlow) -> Inbound {
    Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes: postcard::to_allocvec(flow).expect("encode").into(),
    }
}

/// A `TransientHandoff` egress the orchestrator emits (the target asserts it against `flow_inbound`).
fn handoff(
    arm: fn(TransientHandoff) -> InterShardFlow,
    step_id: u32,
    fence: Fence,
) -> InterShardFlow {
    arm(TransientHandoff {
        transfer: XFER,
        step_id,
        fence,
    })
}

/// A stepped orchestrator + a hub the test drives the saga through. The `source`/`dest`
/// endpoints receive the 1d.1 `FlushSource`/`StubCrossing` egress and let the source inject
/// its `SourceFlushed` pose reply.
struct Rig {
    hub: MemHub,
    orch: crate::app::ShardNode<vd_sim::io::mem::MemTransport>,
    gateway: vd_sim::io::mem::MemTransport,
    source: vd_sim::io::mem::MemTransport,
    dest: vd_sim::io::mem::MemTransport,
    /// D-6: the RETAINED durable store — survives a `rebuild` (the kill-9 analog), so the rebuilt
    /// orchestrator re-hydrates its in-flight sagas/directory/clock from it.
    store: MemStore,
}

/// The shared orchestrator config (so `new` + `rebuild` build the IDENTICAL orchestrator).
fn orch_config() -> OrchestratorConfig {
    OrchestratorConfig {
        epoch: EpochId(1),
        reserve_chunk: 1024,
        clock_peers: vec![],
        directory: DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        },
        saga: SagaTuning::default(),
        liveness: LivenessTuning::default(),
        roster: BTreeMap::new(),
        rlm: vd_sim::rlm::RlmTuning::default(),
    }
}

/// A throwaway RLM spawner for the D-6 crash/recover rigs (RLM is INERT in `orch_config` ⇒ never
/// invoked; it only satisfies the `register_orchestrator_with_store` signature).
fn test_spawner() -> Box<dyn vd_sim::io::RealmSpawner + Send + Sync> {
    Box::new(vd_sim::io::mem::MemSpawner::new(
        MemHub::new(),
        NodeId(1_000_000),
        8,
    ))
}

/// HR5 — a TRACE sink so every tracing macro's lazy field closure evaluates on the paths the
/// tests drive (the Stage-A log points); without a subscriber those closures are dead regions.
fn init_test_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::level_filters::LevelFilter::TRACE)
            .with_writer(std::io::sink)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

impl Rig {
    fn new() -> Rig {
        init_test_tracing();
        let hub = MemHub::new();
        let store = MemStore::new();
        let mut orch = build_app(
            NodeConfig {
                node_id: ORCH,
                kind: NodeKind::Orchestrator,
            },
            hub.register(ORCH, 64),
        );
        let (world, schedule) = orch.parts_mut();
        // D-6: build against a RETAINED store (empty ⇒ genesis); the Rig keeps the handle so
        // `rebuild` can re-attach the SAME committed WAL to a fresh orchestrator.
        register_orchestrator_with_store(
            world,
            schedule,
            &orch_config(),
            Box::new(store.clone()),
            test_spawner(),
            // RLM inert in these D-6 crash/recover rigs — no recovered launches.
            crate::rlm_runtime::LaunchSeed::new(),
        );
        let gateway = hub.register(GATEWAY, 64);
        let source = hub.register(SOURCE, 64);
        let dest = hub.register(DEST, 64);
        Rig {
            hub,
            orch,
            gateway,
            source,
            dest,
            store,
        }
    }

    /// KILL-9 + REBUILD the orchestrator (D-6): drop the World (its in-memory saga/directory/clock
    /// state is GONE), re-attach ORCH's transport on the same hub (peers reachable, in-flight queues
    /// lost), and build a FRESH orchestrator that RECOVERS from the retained durable store. The
    /// SOURCE/DEST/GATEWAY peers are untouched (only the orchestrator died).
    fn rebuild(&mut self) {
        let transport = self.hub.reregister(ORCH, 64);
        let mut orch = build_app(
            NodeConfig {
                node_id: ORCH,
                kind: NodeKind::Orchestrator,
            },
            transport,
        );
        let (world, schedule) = orch.parts_mut();
        register_orchestrator_with_store(
            world,
            schedule,
            &orch_config(),
            Box::new(self.store.clone()),
            test_spawner(),
            // RLM inert in these D-6 crash/recover rigs — no recovered launches.
            crate::rlm_runtime::LaunchSeed::new(),
        );
        self.orch = orch; // the old orchestrator World is dropped here — its RAM is lost
    }

    /// Like [`rebuild`](Self::rebuild) but with a specific D-3 liveness tuning in the recover config —
    /// proves a kill-9-rebuilt orchestrator KEEPS its configured dead-vs-slow margin (not the default).
    fn rebuild_with_liveness(&mut self, liveness: LivenessTuning) {
        let transport = self.hub.reregister(ORCH, 64);
        let mut orch = build_app(
            NodeConfig {
                node_id: ORCH,
                kind: NodeKind::Orchestrator,
            },
            transport,
        );
        let (world, schedule) = orch.parts_mut();
        register_orchestrator_with_store(
            world,
            schedule,
            &OrchestratorConfig {
                liveness,
                ..orch_config()
            },
            Box::new(self.store.clone()),
            test_spawner(),
            // RLM inert in these D-6 crash/recover rigs — no recovered launches.
            crate::rlm_runtime::LaunchSeed::new(),
        );
        self.orch = orch;
    }

    /// Deliver pending sends to the orchestrator, run one tick, then deliver its outbound
    /// replies/commands to the peer endpoints (pump → step → pump).
    fn settle(&mut self) {
        self.hub.pump();
        let _ = self.orch.step_tick();
        self.hub.pump();
    }

    /// Grant a directory record for the subject at `fence` (so `lock_transfer` + the CAS
    /// have something to act on), owned by the source shard.
    fn grant_subject(&mut self, fence: Fence) {
        self.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                        key: subject(),
                        owner: AuthorityRef::Shard(SOURCE),
                        fence,
                    }))
                    .expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// Grant an ARBITRARY directory key/owner/fence through the seam (the D-alpha oracle drives
    /// multiple keys + a higher-fence owner replace; `grant_subject` is the `subject()`+SOURCE special
    /// case). Routes through `serve_directory` → `DirectoryCore::grant` like any shard request.
    fn grant_key(&mut self, key: DirectoryKey, owner: AuthorityRef, fence: Fence) {
        self.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseGrant {
                        key,
                        owner,
                        fence,
                    }))
                    .expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// Revoke an arbitrary directory key at its exact fence (the logout / departed path; the D-alpha
    /// oracle uses it to drive the DELETE arm of the incremental reconcile).
    fn revoke_key(&mut self, key: DirectoryKey, fence: Fence) {
        self.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                        key,
                        fence,
                    }))
                    .expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// The SOURCE ships its pose (the reply to `FlushSource`) — the second half of the
    /// pose-before-promote gate.
    fn flush(&mut self) {
        self.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::TransferAck(
                        TransferAck::SourceFlushed {
                            transfer_id: XFER,
                            step_id: FLUSH_SOURCE_STEP,
                            pose: flushed_pose(),
                            drained_seq: 0,
                            state: vec![],
                        },
                    ))
                    .expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// Everything the orchestrator has sent the DEST since the last drain (the `StubCrossing`
    /// crossing egress + the ordered `Promote`).
    fn drain_dest(&mut self) -> Vec<Inbound> {
        self.dest.drain_inbound()
    }

    /// Everything the orchestrator has sent the SOURCE since the last drain (the `FlushSource`
    /// request + the ordered `Demote`).
    fn drain_source(&mut self) -> Vec<Inbound> {
        self.source.drain_inbound()
    }

    fn trigger(&mut self, ctx: SagaCtx) {
        self.orch
            .world_mut()
            .resource_mut::<SagaRuntimeRes>()
            .start_transfer(ctx, GATEWAY);
    }

    /// Deliver one gateway ack to the orchestrator and step it.
    fn ack(&mut self, ack: TransferControlAck) {
        self.gateway
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::SagaAck(ack)).expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// The DEST acks `BatchAdopted` (D-7) — it now holds the batch as `Arriving`. Drives the
    /// orchestrator's D-7b adopt-before-drop `TransientRelease` egress to the source.
    fn batch_adopted(&mut self, transfer: TransferId) {
        self.dest
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::TransferAck(
                        TransferAck::BatchAdopted {
                            transfer_id: transfer,
                            step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
                        },
                    ))
                    .expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// A shard's `DropApplied` proof-of-apply (D-7b): `TRANSIENT_RELEASE_STEP` = the source
    /// released (gates the dest promote); `TRANSIENT_DROP_STEP` = the dest promote-confirmed
    /// (drives the source `ReleaseComplete`). The orchestrator ignores the sender, so it rides
    /// `self.source` regardless of which shard it models.
    fn drop_applied(&mut self, step_id: u32) {
        self.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::DropApplied {
                        transfer_id: XFER,
                        step_id,
                    }))
                    .expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// Everything the orchestrator has sent the gateway since the last drain, as raw
    /// `Inbound` (full-value equality, no destructuring — so there are no never-taken
    /// arms to leave uncovered; the codebase convention).
    fn drain_gateway(&mut self) -> Vec<Inbound> {
        self.gateway.drain_inbound()
    }

    fn subject_head(&mut self) -> vd_wire::seams::directory::OwnerRecord {
        self.orch
            .world_mut()
            .resource::<DirectoryRes>()
            .0
            .head(subject())
            .expect("subject recorded")
    }

    fn live(&mut self) -> usize {
        self.orch.world_mut().resource::<SagaRuntimeRes>().live()
    }

    /// Slice 3f-B: a source shard ships a durable `CrossingRequest` (the boundary-detector egress). The
    /// orchestrator sees it as `Inbound::Wire { from: SOURCE, .. }` (the source is the transport origin).
    fn crossing_request(&mut self, req: CrossingRequest) {
        self.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::CrossingRequest(req)).expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// Slice 3f-C: a source shard ships a transient `TransientCrossingRequest`. The reply GRANT routes
    /// back to the transport origin (`SOURCE`), so a test drains it via [`drain_source`](Self::drain_source).
    /// The ruler switch, slice 1: a parent's request to move its hull's exterior, from `peer`.
    fn exterior_request(
        &mut self,
        req: vd_wire::intershard::ExteriorCrossingRequest,
        from_dest: bool,
    ) {
        let peer = if from_dest {
            &mut self.dest
        } else {
            &mut self.source
        };
        peer.send(
            ORCH,
            MsgClass::Saga,
            vd_sim::io::bytes(
                postcard::to_allocvec(&InterShardFlow::ExteriorCrossingRequest(req))
                    .expect("encode"),
            ),
        )
        .expect("sent");
        self.settle();
    }

    fn transient_crossing_request(&mut self, req: TransientCrossingRequest) {
        self.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::TransientCrossingRequest(req))
                        .expect("encode"),
                ),
            )
            .expect("sent");
        self.settle();
    }

    /// The immutable ctx of the ONE live saga keyed on `transfer` — the crossing tests assert its
    /// source/dest/session/transfer are the ones the resolver latched. Panics if absent (the test's
    /// contract is that the saga started).
    fn saga_ctx(&mut self, transfer: TransferId) -> SagaCtx {
        self.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .sagas
            .get(&transfer)
            .expect("saga live")
            .ctx
    }

    /// The `SagaState` of the ONE live saga keyed on `transfer` (the D-43 #9 transient-start test
    /// asserts it is parked in `BatchHandoff{AwaitAdopt}`). Panics if absent.
    fn saga_state(&mut self, transfer: TransferId) -> SagaState {
        self.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .sagas
            .get(&transfer)
            .expect("saga live")
            .state
    }

    /// Read a `SagaRuntimeRes` counter accessor (the crossing outcome counts).
    fn count(&mut self, read: impl Fn(&SagaRuntimeRes) -> u64) -> u64 {
        read(self.orch.world_mut().resource::<SagaRuntimeRes>())
    }
}

const CROSSING_SESSION: SessionId = SessionId(9);

/// A durable `CrossingRequest` for `subject()` crossing FROM_REALM → TO_REALM at `subject_fence`,
/// carrying `CROSSING_SESSION` (the source-supplied session — 3f-A).
fn crossing_req(subject_fence: Fence) -> CrossingRequest {
    CrossingRequest {
        subject: subject(),
        from_realm: FROM_REALM,
        to_realm: TO_REALM,
        subject_fence,
        session: CROSSING_SESSION,
        attempt: 0,
        // A concrete parent so the threading through `handle_crossing_request` → `ctx.to_parent` is
        // ASSERTED (crossing_request_resolves_and_starts_the_saga). The value is arbitrary here
        // (TO_REALM=System(8) is nameable regardless); the assertion proves the field is not dropped.
        to_parent: Some(FROM_REALM),
    }
}

#[test]
fn durable_happy_path_drives_the_gateway_through_the_direct_commit() {
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
    rig.settle(); // process the start → PrepareSubscribe
    assert_eq!(
        rig.drain_gateway(),
        vec![saga_wire(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: SESSION,
            dest: DEST,
        })]
    );

    rig.ack(TransferControlAck::Prepared {
        transfer: XFER,
        result: PrepareResult::Ready,
    });
    assert_eq!(
        rig.drain_gateway(),
        vec![saga_wire(TransferControl::RequestCut {
            transfer: XFER,
            session: SESSION,
        })]
    );

    rig.ack(TransferControlAck::CutConfirmed {
        transfer: XFER,
        marker_seq: 42,
    });
    assert_eq!(
        rig.drain_gateway(),
        vec![saga_wire(TransferControl::FreezeSource {
            transfer: XFER,
            session: SESSION,
            marker_seq: 42,
            dest: DEST,
        })]
    );

    // The pose-before-promote gate: SourceFrozen alone leaves the saga in Freezing (no CAS).
    rig.ack(TransferControlAck::SourceFrozen {
        transfer: XFER,
        drained_seq: 42,
    });
    assert!(
        rig.drain_gateway().is_empty(),
        "SourceFrozen alone does not commit — the gate awaits the source flush"
    );
    // The SOURCE flushes its pose → BOTH gate conditions met → the DIRECT commit_cas wins
    // in-process → CommitAuthority is sent with the NEW fence, all in ONE tick.
    rig.flush();
    let new_fence = Fence(1).next();
    assert_eq!(
        rig.drain_gateway(),
        vec![saga_wire(TransferControl::CommitAuthority {
            transfer: XFER,
            session: SESSION,
            new_fence,
            subject: subject(),
        })]
    );
    // The entity-STATE crossing rode the SAME commit batch to the DEST, carrying the flushed
    // pose at the new authority fence (the 1d.1 deliverable, asserted end-to-end).
    assert_eq!(
        rig.drain_dest(),
        vec![Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::Transfer(TransferEnvelope {
                transfer_id: XFER,
                universe_epoch: EpochId(1),
                schema_version: TRANSFER_SCHEMA_VERSION,
                fence: new_fence,
                step_id: STUB_CROSSING_STEP,
                class: DurabilityClass::Durable,
                payload: TransitionPayload::StubCrossing {
                    entity: subject_eid(),
                    from_realm: FROM_REALM,
                    to_realm: TO_REALM,
                    pose: flushed_pose(),
                    state: vec![],
                },
            }))
            .expect("encode")
            .into(),
        }],
        "the crossing reached the dest with the flushed pose at the new fence"
    );
    // The directory CAS committed: the dest now owns the subject at the new fence.
    let head = rig.subject_head();
    assert_eq!(head.authority, AuthorityRef::Shard(DEST));
    assert_eq!(head.fence, new_fence);

    // Clear the source buffer of the earlier egress (the grant DirectoryReply + the Freezing
    // FlushSource) so the next drain isolates the ordered Demote.
    let _ = rig.drain_source();

    // Committed (the route swapped) advances to Demoting and PUSHES the ordered Demote to the
    // SOURCE (the fence-enforced Owned→Frozen→Ghost, at the new owner fence) — NOT yet a release.
    rig.ack(TransferControlAck::Committed { transfer: XFER });
    assert_eq!(
        rig.drain_source(),
        vec![demote_wire(new_fence)],
        "Demoting pushes the ordered Demote to the source at the new owner fence"
    );
    assert!(
        rig.drain_gateway().is_empty(),
        "no ReleaseSubscribe before the dest is promoted + delivered (demote-before-promote)"
    );
    assert_eq!(
        rig.live(),
        1,
        "still live in Demoting, awaiting the demote-ack"
    );

    // The source acks DemoteAck (proof-of-freeze) → Promoting + the ordered Promote pushed to
    // the DEST. The demote-ack ALONE advances (breaking R1) — release is NOT yet due.
    rig.ack(TransferControlAck::DemoteAck { transfer: XFER });
    assert_eq!(
        rig.drain_dest(),
        vec![promote_wire(new_fence)],
        "the demote-ack advances to Promoting and pushes the Promote to the dest"
    );
    assert!(
        rig.drain_gateway().is_empty(),
        "promote pushed; the source sub is still held (seamless overlap)"
    );
    assert_eq!(rig.live(), 1);

    // PromoteAck alone holds; DeliveredToObservers completes the seamless gate → ReleaseSubscribe.
    rig.ack(TransferControlAck::PromoteAck { transfer: XFER });
    assert!(
        rig.drain_gateway().is_empty(),
        "promote-ack alone does not release — the delivery half is still pending"
    );
    rig.ack(TransferControlAck::DeliveredToObservers { transfer: XFER });
    assert_eq!(
        rig.drain_gateway(),
        vec![saga_wire(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: SESSION,
            src: SOURCE,
        })],
        "promote-ack AND delivery release the source sub (the seamless gate)"
    );
    assert_eq!(rig.live(), 1);

    // Released → Done → Tombstone: the tail closes, live()→0 with the subject settled at DEST.
    rig.ack(TransferControlAck::Released { transfer: XFER });
    assert_eq!(
        rig.live(),
        0,
        "the ordered demote/promote/release tail reaches Done"
    );
}

#[test]
fn early_delivery_in_demoting_is_carried_into_promoting_and_never_lost() {
    // 1d.5b.1 race fix: the gateway's standing watermark (`DeliveredToObservers`) can arrive
    // while the saga is STILL Demoting — the dest is already delivering from its autonomous
    // adopt-promote, before the source's Demote round-trip completes. The FSM LATCHES it in
    // `Demoting.dest_delivered` and carries it into `Promoting`, so the release gate can never
    // lose it (no park). This drives the full Demoting→Promoting→Releasing→Done tail end-to-end.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    let new_fence = Fence(1).next();
    // Inject a live saga already in Demoting (the post-RouteSwapped entry state).
    {
        let c = ctx(DurabilityClass::Durable, Fence(1));
        let mut runtime = rig.orch.world_mut().resource_mut::<super::SagaRuntimeRes>();
        runtime.sagas.insert(
            XFER,
            LiveSaga {
                ctx: c,
                state: SagaState::Demoting {
                    new_fence,
                    dest_delivered: false,
                },
                gateway: GATEWAY,
                since: UniverseTick(0),
                flushed_pose: None,
                flushed_state: Vec::new(),
                dead_observed_since: None,
                dest_adopted: false,
                opened: UniverseTick(0),
            },
        );
    }
    // Delivery arrives FIRST, while still Demoting: latched, NO Promote yet (awaiting the
    // demote-ack — demote-before-promote), NO release.
    rig.ack(TransferControlAck::DeliveredToObservers { transfer: XFER });
    assert!(
        rig.drain_dest().is_empty(),
        "no Promote while still Demoting — the demote-ack has not landed"
    );
    assert!(
        rig.drain_gateway().is_empty(),
        "no release while still Demoting"
    );
    assert_eq!(rig.live(), 1);

    // DemoteAck → Promoting (carrying the latched delivery) + the ordered Promote to the dest.
    rig.ack(TransferControlAck::DemoteAck { transfer: XFER });
    assert_eq!(
        rig.drain_dest(),
        vec![promote_wire(new_fence)],
        "the demote-ack advances to Promoting and pushes the Promote"
    );
    assert!(
        rig.drain_gateway().is_empty(),
        "delivery already counted, but the promote-ack is still pending"
    );

    // PromoteAck ALONE now completes the gate (the early delivery was carried forward) →
    // ReleaseSubscribe — proving the watermark was never lost across the Demoting→Promoting edge.
    rig.ack(TransferControlAck::PromoteAck { transfer: XFER });
    assert_eq!(
        rig.drain_gateway(),
        vec![saga_wire(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: SESSION,
            src: SOURCE,
        })],
        "the carried delivery + the promote-ack release the source sub"
    );
    rig.ack(TransferControlAck::Released { transfer: XFER });
    assert_eq!(rig.live(), 0, "the ordered tail reached Done");
}

#[test]
fn prepare_rejection_aborts_and_tombstones_with_typed_feedback() {
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
    rig.settle();
    let _ = rig.drain_gateway();

    let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
    rig.ack(TransferControlAck::Prepared {
        transfer: XFER,
        result: PrepareResult::Rejected(reject),
    });
    // The dest is torn down; the typed rejection is recorded for the client (Slice 1c).
    assert_eq!(
        rig.drain_gateway(),
        vec![saga_wire(TransferControl::AbortTransfer {
            transfer: XFER,
            session: SESSION,
        })]
    );
    assert_eq!(
        rig.orch.world_mut().resource::<SagaRuntimeRes>().rejected,
        vec![(XFER, AbortReason::PrepareRejected(reject))]
    );

    // DestAborted → terminal Aborted → tombstoned (GC'd). The player stayed on the source.
    rig.ack(TransferControlAck::Aborted { transfer: XFER });
    assert_eq!(rig.live(), 0, "terminal saga is tombstoned");
    // The subject's authority never moved (the CAS never ran): source still owns it.
    let head = rig.subject_head();
    assert_eq!(head.authority, AuthorityRef::Shard(SOURCE));
    // ✅ FLIPPED (Slice 2a, D-1): the terminal Aborted edge emits ClearTransferLock →
    // abort_clear, so the lock CLEARS — the subject can immediately re-transfer (no wedge).
    assert_eq!(
        head.in_transfer, None,
        "Slice 2a: terminal abort clears the directory lock (D-1 closed)"
    );
}

#[test]
fn cas_loss_unwinds_with_a_thaw() {
    // The saga expects Fence(1) but the directory moved to Fence(5) (someone else won) →
    // the DIRECT commit_cas LOSES → abort-with-thaw fires (the frozen source must thaw).
    let mut rig = Rig::new();
    rig.grant_subject(Fence(5));
    rig.trigger(ctx(DurabilityClass::Durable, Fence(1))); // stale expectation
    rig.settle();
    let _ = rig.drain_gateway();
    rig.ack(TransferControlAck::Prepared {
        transfer: XFER,
        result: PrepareResult::Ready,
    });
    rig.ack(TransferControlAck::CutConfirmed {
        transfer: XFER,
        marker_seq: 9,
    });
    let _ = rig.drain_gateway();

    rig.ack(TransferControlAck::SourceFrozen {
        transfer: XFER,
        drained_seq: 9,
    });
    rig.flush(); // gate: both freeze + flush, so the CAS actually runs (and loses)
    // A lost CAS unwinds with the compensator chain: ThawSource then AbortTransfer.
    assert_eq!(
        rig.drain_gateway(),
        vec![
            saga_wire(TransferControl::ThawSource {
                transfer: XFER,
                session: SESSION,
            }),
            saga_wire(TransferControl::AbortTransfer {
                transfer: XFER,
                session: SESSION,
            }),
        ]
    );
    // The authority never moved (CAS lost); the dest is being torn down + source thawed.
    assert_eq!(rig.subject_head().fence, Fence(5));
    rig.ack(TransferControlAck::SourceThawed { transfer: XFER });
    rig.ack(TransferControlAck::Aborted { transfer: XFER });
    assert_eq!(rig.live(), 0);
    // ✅ FLIPPED (Slice 2a, D-1, audit CPO-2): even the CAS-LOSER's terminal abort clears the
    // lock — `abort_clear` RE-READS the head (this saga's expected fence is stale by definition,
    // so a naive `abort_cas(expected)` would lose too) and clears the lock. FENCE-NEUTRAL: the CAS
    // LOST so authority never moved off the source and no crossing was ever emitted (CasLost never
    // reaches Swapping), so the fence stays put — `directory.fence == source.entity_fence` (FENCE-9).
    let head = rig.subject_head();
    assert_eq!(
        head.in_transfer, None,
        "Slice 2a: the CAS-loser's terminal abort clears the lock via the head re-read (D-1)"
    );
    assert_eq!(
        head.fence,
        Fence(5),
        "abort_clear is fence-neutral: the source still owns at Fence(5), no spurious bump (FENCE-9)"
    );
}

#[test]
fn transient_subject_commits_via_the_go_token_not_a_cas() {
    // HR2 SHORT PATH (D-7): a Transient subject is NOT in the directory (no `grant_subject` —
    // burst isolation), takes NO lock, SKIPS Prepare/Cut/Freeze, and commits the batched go-token at
    // START. It records EXACTLY ONE go-token (G-TIER: one write per batch, never per item), NEVER a
    // per-entity CAS, NEVER a `CommitAuthority`. D-7d: the go-token commit enters the POST-COMMIT
    // `BatchHandoff` TAIL (the saga stays LIVE owning the adopt-before-drop choreography — so a
    // stranded handoff is visible to `scan_deadlines`), tombstoning only at `SourceRetired`.
    let mut rig = Rig::new();
    rig.trigger(transient_ctx(Fence(9)));
    rig.settle(); // process_starts: no lock (Transient) → start → BatchCommitting → go-token → CasWon → BatchHandoff{AwaitAdopt}

    assert_eq!(
        rig.live(),
        1,
        "the short-path transient saga enters the BatchHandoff tail (alive until the choreography completes), NOT tombstoned at commit"
    );
    assert!(
        rig.drain_gateway().is_empty(),
        "a transient drives NO gateway route-swap (no PrepareSubscribe, no CommitAuthority)"
    );
    // Exactly ONE go-token, at the dest realm-lease fence (G-TIER: one orchestrator write per batch).
    assert_eq!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .batch_goes(),
        vec![(BatchId(XFER), Fence(9))],
        "one batched go-token recorded at the commit fence"
    );
    // D-7c G-TIER observable: ONE write (the write COUNT, not just the deduped map size).
    assert_eq!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .batch_go_writes(),
        1,
        "one go-token WRITE per batch (the per-item-regression probe)"
    );

    // The D-7b structural drop-before-promote, driven through the orchestrator (two ordered
    // round-trips). PHASE 1 — the DEST's BatchAdopted → the orchestrator tells the SOURCE ONLY to
    // RELEASE (not a broadcast); the dest gets NOTHING yet (gated on the source's DropApplied).
    let _ = (rig.drain_source(), rig.drain_dest());
    rig.batch_adopted(XFER);
    assert_eq!(
        rig.drain_source(),
        vec![flow_inbound(&handoff(
            InterShardFlow::TransientRelease,
            TRANSIENT_RELEASE_STEP,
            Fence(9)
        ))],
        "the source is told to RELEASE (Held→Departing) — adopt-before-drop phase 1"
    );
    assert!(
        rig.drain_dest().is_empty(),
        "the dest is NOT promoted until the source releases (no both-held window)"
    );

    // PHASE 2 — the source's DropApplied(RELEASE) → the orchestrator PROMOTES the DEST only.
    rig.drop_applied(TRANSIENT_RELEASE_STEP);
    assert_eq!(
        rig.drain_dest(),
        vec![flow_inbound(&handoff(
            InterShardFlow::TransientDrop,
            TRANSIENT_DROP_STEP,
            Fence(9)
        ))],
        "the dest is told to PROMOTE (Arriving→Held) only after the source released"
    );
    assert!(
        rig.drain_source().is_empty(),
        "no source egress on the promote step"
    );

    // PHASE 3 — the dest's DropApplied(DROP, promote-confirm) → the orchestrator tells the SOURCE
    // to retire the retained Departing copy (ReleaseComplete).
    rig.drop_applied(TRANSIENT_DROP_STEP);
    assert_eq!(
        rig.drain_source(),
        vec![flow_inbound(&handoff(
            InterShardFlow::ReleaseComplete,
            TRANSIENT_RELEASE_STEP,
            Fence(9)
        ))],
        "the dest's promote-confirm retires the source's Departing copy"
    );
    // The saga is STILL live through the handoff (the D-7d tail) — NOT tombstoned at commit.
    assert_eq!(
        rig.live(),
        1,
        "the saga owns the handoff to its terminal (awaiting the source's retire-complete)"
    );

    // PHASE 4 (D-7d) — the source's COMPLETE ack (`on_release_complete`'s `DropApplied`(COMPLETE))
    // drives `SourceRetired` → the tail tombstones. NO further egress (the choreography is done).
    rig.drop_applied(TRANSIENT_COMPLETE_STEP);
    assert_eq!(
        rig.live(),
        0,
        "the source's retire-complete drives the BatchHandoff tail to Done (tombstoned)"
    );
    // Split (NOT `&&`) so neither short-circuit edge is an uncoverable branch (HR5).
    assert!(
        rig.drain_source().is_empty(),
        "no source egress after the handoff completes"
    );
    assert!(
        rig.drain_dest().is_empty(),
        "no dest egress after the handoff completes"
    );
}

#[test]
fn a_transient_handoff_ack_with_no_go_token_is_a_loud_noop() {
    // D-7d defensive: a `BatchAdopted` OR a `DropApplied` for a batch with NO LIVE saga (a
    // stray/duplicate, or a batch that never started / already tombstoned) emits NOTHING — a no-op,
    // never a silent drop and never a panic. Covers `deliver`'s early-return on an absent saga
    // (`runtime.sagas.get_mut` → `None`), the same idempotency that absorbs a redelivered ack after
    // the BatchHandoff tail tombstoned.
    let mut rig = Rig::new();
    let _ = (rig.drain_source(), rig.drain_dest());
    rig.batch_adopted(XFER); // no prior trigger → no live saga for XFER
    assert!(rig.drain_source().is_empty(), "no saga ⇒ no release");
    assert!(rig.drain_dest().is_empty(), "no saga ⇒ no promote");
    rig.drop_applied(TRANSIENT_RELEASE_STEP); // also no saga → deliver early-returns (None arm)
    assert!(
        rig.drain_source().is_empty(),
        "a DropApplied with no go-token is a loud no-op (no source egress)"
    );
    assert!(
        rig.drain_dest().is_empty(),
        "a DropApplied with no go-token is a loud no-op (no dest egress)"
    );
}

#[test]
fn durable_saga_parked_in_committing_is_curl_visible() {
    // AAA-1: a parked saga is CURL-VISIBLE through the real admin snapshot — its transfer id, its
    // actual stuck phase, and a staleness anchor, never silently wedged. (Drives both
    // `SagaRuntimeRes::views` and `admin_snapshot`'s saga-population end to end.) D-7 moved this
    // OFF the old transient-park vehicle: a DURABLE saga whose direct CAS is starved (the granted
    // fence races ahead so the CAS would lose) is the wrong vehicle; instead we hold a durable
    // saga in `Demoting` awaiting its `DemoteAck` — a genuine post-commit park.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
    rig.settle();
    rig.ack(TransferControlAck::Prepared {
        transfer: XFER,
        result: PrepareResult::Ready,
    });
    rig.ack(TransferControlAck::CutConfirmed {
        transfer: XFER,
        marker_seq: 3,
    });
    rig.ack(TransferControlAck::SourceFrozen {
        transfer: XFER,
        drained_seq: 3,
    });
    rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping
    rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (parked, awaiting DemoteAck)
    let _ = (rig.drain_gateway(), rig.drain_source());
    assert_eq!(rig.live(), 1);

    let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut(), 0);
    assert_eq!(snap.sagas.len(), 1);
    assert_eq!(snap.sagas[0].transfer, XFER.to_string());
    // The operator sees the REAL parked phase with its fields (the post-CAS authority fence),
    // not a placeholder. Exact equality (no `assert!`-message format arm — HR5 discipline).
    assert_eq!(
        snap.sagas[0].state,
        "Demoting { new_fence: Fence(2), dest_delivered: false }"
    );
    // `since` was set when the saga last advanced and is bounded by the clock.
    assert!(snap.sagas[0].since.0 <= snap.universe_tick);
}

#[test]
fn orchestrator_rehydrates_an_in_flight_durable_saga_across_a_kill_9() {
    // D-6 HEADLINE: a durable saga in flight (Demoting, POST-commit) SURVIVES an orchestrator kill-9.
    // The rebuilt orchestrator re-hydrates it from the durable WAL (NOT vaporized) and the existing
    // Slice-2a Timeout producer re-drives it FORWARD (re-emits the ordered Demote) — never-vanish
    // holds across a restart. Anti-theater: the `rebuild` drops the in-memory World, so the saga can
    // only survive via the durable Store (a no-persist orchestrator would lose it = `live() == 0`).
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
    rig.settle();
    rig.ack(TransferControlAck::Prepared {
        transfer: XFER,
        result: PrepareResult::Ready,
    });
    rig.ack(TransferControlAck::CutConfirmed {
        transfer: XFER,
        marker_seq: 3,
    });
    rig.ack(TransferControlAck::SourceFrozen {
        transfer: XFER,
        drained_seq: 3,
    });
    rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping (durable at the barrier)
    rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (durable at the barrier)
    let _ = (rig.drain_gateway(), rig.drain_source());
    assert_eq!(rig.live(), 1, "the saga is live in Demoting pre-crash");
    let before = crate::orchestrator::admin_snapshot(rig.orch.world_mut(), 0);
    // Static message (no `{}` format arg — a lazily-evaluated arg is an uncoverable region when the
    // assert passes, HR5); the `starts_with` IS the always-evaluated condition.
    assert!(
        before.sagas[0].state.starts_with("Demoting"),
        "pre-crash state is Demoting"
    );

    // KILL-9 + REBUILD: the in-memory saga set dies with the World; the rebuilt orchestrator RECOVERS
    // its in-flight saga + the committed directory + the clock from the retained WAL.
    rig.rebuild();
    assert_eq!(
        rig.live(),
        1,
        "the in-flight saga SURVIVED the orchestrator kill-9 (re-hydrated from the WAL, not vanished)"
    );
    let after = crate::orchestrator::admin_snapshot(rig.orch.world_mut(), 0);
    assert!(
        after.sagas[0].state.starts_with("Demoting"),
        "re-hydrated at the SAME quiescent phase (Demoting)"
    );
    // The directory survived too: the subject's authority committed to the DEST at the CAS fence.
    let head = rig
        .orch
        .world_mut()
        .resource::<DirectoryRes>()
        .0
        .head(subject())
        .expect("the subject's directory record re-hydrated");
    assert_eq!(head.authority, AuthorityRef::Shard(DEST));
    assert_eq!(
        head.fence,
        Fence(2),
        "the committed CAS fence survived the crash"
    );

    // RE-DRIVE: the recovered saga's `since` is armed, so the first post-restart tick fires
    // `scan_deadlines` → Demoting+Timeout re-emits the ordered Demote to the SOURCE (forward-only,
    // the SAME proven producer — no new recovery path). Proves the transfer makes progress post-crash.
    let _ = rig.drain_source();
    rig.settle();
    assert!(
        rig.drain_source().contains(&demote_wire(Fence(2))),
        "the rebuilt orchestrator re-drove the in-flight transfer forward (re-emitted the Demote)"
    );
}

#[test]
fn orchestrator_rehydrates_a_transient_go_token_across_a_kill_9() {
    // D-6 (transient arm): an in-flight transient batch's go-token + its `BatchHandoff` saga survive
    // an orchestrator kill-9. The WAL carries the go-token as ONE record per batch, so rehydrate
    // restores `batch_go_writes` from the map LEN — the G-TIER decouple is preserved on restart, NOT
    // re-driven through the `+= 1` (a regression that re-incremented would read N here, not 1).
    let mut rig = Rig::new();
    rig.trigger(transient_ctx(Fence(9)));
    rig.settle(); // BatchCommitting → the go-token commits (persisted) → BatchHandoff{AwaitAdopt}
    {
        let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
        assert_eq!(
            runtime.live(),
            1,
            "the transient saga is live in the BatchHandoff tail pre-crash"
        );
        assert_eq!(runtime.batch_goes(), vec![(BatchId(XFER), Fence(9))]);
        assert_eq!(runtime.batch_go_writes(), 1);
    }

    rig.rebuild();
    let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
    assert_eq!(
        runtime.live(),
        1,
        "the in-flight transient saga SURVIVED the kill-9 (re-hydrated from the WAL)"
    );
    assert_eq!(
        runtime.batch_goes(),
        vec![(BatchId(XFER), Fence(9))],
        "the committed go-token re-hydrated (the dest's Held authority stays backed)"
    );
    assert_eq!(
        runtime.batch_go_writes(),
        1,
        "batch_go_writes restored from the map len, NOT re-incremented through the +=1 (G-TIER decouple)"
    );
}

#[test]
fn orchestrator_rehydrates_a_transient_saga_mid_await_release() {
    // D-43 #9 crash-safety (FIRST-TIME-REACHABLE persisted state): before the fix, a Transient saga
    // was NEVER persisted (the only `start_transfer` caller was the durable crossing handler), so a
    // `BatchHandoff{AwaitRelease}` `SagaSnapshot` becomes reachable only now. Persist a transient saga
    // mid-`AwaitRelease` (post-`BatchAdopted`), kill-9 + rehydrate, and assert it re-inserts under
    // `batch` with `SessionId::NONE` INTACT, re-drives its pending `TransientRelease` on the next tick,
    // and its go-token is restored — locking the newly-reachable transient persistence path.
    let mut rig = Rig::new();
    rig.trigger(transient_ctx(Fence(9)));
    rig.settle(); // → BatchHandoff{AwaitAdopt}, go-token committed
    rig.batch_adopted(XFER); // dest adopts → AwaitRelease, source told to TransientRelease
    {
        let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
        assert_eq!(runtime.live(), 1, "the transient saga is live pre-crash");
        assert_eq!(
            runtime.sagas.get(&XFER).expect("saga live").state,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitRelease,
                new_fence: Fence(9),
            },
            "parked mid-AwaitRelease (the newly-reachable persisted transient state)"
        );
        assert_eq!(
            runtime.sagas.get(&XFER).expect("saga live").ctx.session,
            SessionId::NONE,
            "the session-less sentinel is what gets persisted"
        );
    }
    let _ = (rig.drain_source(), rig.drain_dest()); // clear the pre-crash egress

    rig.rebuild();
    {
        let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
        assert_eq!(
            runtime.live(),
            1,
            "the in-flight transient saga SURVIVED the kill-9 mid-AwaitRelease"
        );
        let live = runtime.sagas.get(&XFER).expect("re-inserted under batch");
        assert_eq!(
            live.state,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitRelease,
                new_fence: Fence(9),
            },
            "the AwaitRelease phase + go-token fence rehydrated verbatim"
        );
        assert_eq!(
            live.ctx.session,
            SessionId::NONE,
            "the session-less sentinel survived the WAL round-trip intact"
        );
        assert_eq!(
            runtime.batch_goes(),
            vec![(BatchId(XFER), Fence(9))],
            "the committed go-token re-hydrated (the dest's authority stays backed)"
        );
    }

    // The rehydrated saga re-drives on the next tick: `since` is armed to 0, so the first
    // `scan_deadlines` fires a `Timeout` → re-emits the idempotent `TransientRelease` to the source.
    rig.settle();
    assert!(
        rig.drain_source().contains(&flow_inbound(&handoff(
            InterShardFlow::TransientRelease,
            TRANSIENT_RELEASE_STEP,
            Fence(9),
        ))),
        "the rebuilt orchestrator re-drove the in-flight transient handoff (re-emitted TransientRelease)"
    );
}

/// D-6 (D-alpha) DIFFERENTIAL ORACLE: assert the durable Directory family produced by the INCREMENTAL
/// `dirty`-delta reconcile is byte-for-byte identical to a FULL delete-all-then-put-current reconcile.
/// The full reconcile is, by construction, exactly the encoded snapshot of every CURRENT entry — so the
/// expected map is computed directly from `entries()`. Any arm that changed a row without staging its
/// delta (a stale durable snapshot, or a phantom row never deleted) shows up here as a mismatch.
fn assert_incremental_matches_full_reconcile(rig: &mut Rig) {
    let actual: BTreeMap<Vec<u8>, Vec<u8>> = rig
        .store
        .scan(&[StoreKey::DIRECTORY])
        .into_iter()
        .map(|(k, v)| (k, v.to_vec()))
        .collect();
    let expected: BTreeMap<Vec<u8>, Vec<u8>> = rig
        .orch
        .world_mut()
        .resource::<DirectoryRes>()
        .0
        .entries()
        .map(|(k, r)| {
            (
                StoreKey::Directory(*k).bytes(),
                encode(&DirSnapshot {
                    key: *k,
                    record: *r,
                })
                .to_vec(),
            )
        })
        .collect();
    assert_eq!(
        actual, expected,
        "incremental reconcile diverged from the full reconcile"
    );
}

#[test]
fn incremental_directory_reconcile_equals_a_full_reconcile_byte_for_byte() {
    // Drive PUT-new / PUT-refresh / PUT-replace / DELETE through the REAL group-commit barrier and pin
    // the differential oracle after each settle; the kill-9 rebuild proves rehydrate round-trips the
    // incremental durable set with no divergence (the load-bearing COMP-2 correctness for D-gamma).
    let other = DirectoryKey::Realm(RealmId::System(7));
    let mut rig = Rig::new();

    // PUT-new, two distinct keys → a multi-row durable family.
    rig.grant_subject(Fence(1));
    rig.grant_key(other, AuthorityRef::Shard(DEST), Fence(1));
    assert_incremental_matches_full_reconcile(&mut rig);

    // PUT-refresh (idempotent re-grant moves the lease) then PUT-replace (higher fence, new owner) —
    // both must OVERWRITE the prior durable snapshot, never leave a stale one behind.
    rig.grant_subject(Fence(1));
    rig.grant_key(subject(), AuthorityRef::Shard(DEST), Fence(7));
    assert_incremental_matches_full_reconcile(&mut rig);

    // DELETE: revoke the second key (the COMP-2 anti-zombie path through the barrier).
    rig.revoke_key(other, Fence(1));
    assert_incremental_matches_full_reconcile(&mut rig);

    // KILL-9 + REBUILD: rehydrate restores from the incremental durable set; entries() still equals it.
    rig.rebuild();
    assert_incremental_matches_full_reconcile(&mut rig);
}

#[test]
fn directory_store_key_is_the_directory_family_tag_plus_postcard() {
    // D-delta: the crash test computes the sentinel pause prefix via this shim; it MUST equal the bytes
    // the barrier stages for a directory row ([DIRECTORY] ++ postcard(key)) or the writer pause misses.
    let key = DirectoryKey::Realm(RealmId::System(42));
    let bytes = directory_store_key(&key);
    assert_eq!(
        bytes[0],
        StoreKey::DIRECTORY,
        "leads with the DIRECTORY family tag"
    );
    assert_eq!(
        &bytes[1..],
        &postcard::to_allocvec(&key).expect("encode")[..],
        "tail is postcard(key)"
    );
}

#[test]
fn rehydrate_does_not_resurrect_a_revoked_directory_record() {
    // D-6 (audit COMP-2): a directory record REVOKED before a kill-9 (a logged-out / departed owner —
    // the gateway-Bye / stub-departing `LeaseRevoke` path) must NOT be resurrected by rehydrate. The
    // reconcile barrier deletes the durable row; a put-only snapshot would re-install a stale-authority
    // zombie on recover. This exercises the barrier's durable-delete loop + the no-resurrect guarantee.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1)); // SOURCE owns the subject at Fence(1) — durably persisted
    // REVOKE it at its exact fence (the logout / departed path), then settle so the barrier reconciles.
    rig.source
        .send(
            ORCH,
            MsgClass::Saga,
            vd_sim::io::bytes(
                postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
                    key: subject(),
                    fence: Fence(1),
                }))
                .expect("encode"),
            ),
        )
        .expect("revoke sent");
    rig.settle();
    assert!(
        rig.orch
            .world_mut()
            .resource::<DirectoryRes>()
            .0
            .head(subject())
            .is_none(),
        "the record is gone from RAM after the revoke"
    );

    // KILL-9 + REBUILD: the revoked record must STAY gone (not resurrected by rehydrate's restore).
    rig.rebuild();
    assert!(
        rig.orch
            .world_mut()
            .resource::<DirectoryRes>()
            .0
            .head(subject())
            .is_none(),
        "the revoked record is NOT resurrected by rehydrate (COMP-2: the barrier durably deleted it)"
    );
}

#[test]
fn a_handoff_past_its_whole_budget_is_reported_expired_instead_of_shielded() {
    // THE LEAK BOUND, exercised. Every other test reads the shield with an unbounded budget, which
    // only ever takes the "still shielded" side — so the arm that LIFTS the shield had never run.
    // That arm is the one an operator depends on: it is what distinguishes a hand-off that is slow
    // from one that is wedged, and it is what stops a single wedged crossing pinning a realm alive
    // for the lifetime of the process.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.trigger(ctx(
        vd_core::entity_kind::DurabilityClass::Durable,
        Fence(1),
    ));
    rig.settle();

    // Same live crossing, same instant — the ONLY difference is the budget it is measured against.
    let (shielded, expired) = rig
        .orch
        .world_mut()
        .resource::<SagaRuntimeRes>()
        .arriving_dest_realms(UniverseTick(1), u64::MAX);
    assert_eq!(shielded, BTreeSet::from([TO_REALM]), "inside budget: held");
    assert!(expired.is_empty(), "inside budget: nothing to report");

    // A budget of ZERO would prove nothing: that is the DISARMED early return, which reports no
    // arrivals at all. The smallest ARMED budget is what puts a real hand-off past a real deadline.
    let (disarmed, none_reported) = rig
        .orch
        .world_mut()
        .resource::<SagaRuntimeRes>()
        .arriving_dest_realms(UniverseTick(500), 0);
    // SPLIT, not `a && b`: a short-circuit leaves the right-hand side unevaluated whenever the left is
    // false, so one arm can never be reached and the crate cannot hit 100% (the project's own rule).
    assert!(disarmed.is_empty(), "a zero budget shields nothing");
    assert!(
        none_reported.is_empty(),
        "and reports nothing — disarmed is not expired"
    );

    let (shielded, expired) = rig
        .orch
        .world_mut()
        .resource::<SagaRuntimeRes>()
        .arriving_dest_realms(UniverseTick(500), 1);
    assert!(
        shielded.is_empty(),
        "past budget the destination is NO LONGER held — the shield lifts, it does not linger"
    );
    assert_eq!(expired.len(), 1, "and the lift is REPORTED, never silent");
    // What a 2am operator needs to find it: which realm was being held, and for how long.
    assert_eq!(expired[0].realm, TO_REALM);
    // The phase is carried so the report names WHERE it wedged. Asserted as non-empty rather than as a
    // specific phase: which one a settled fixture parks in is an implementation detail of the rig, and
    // pinning it would make this test fail for reasons that have nothing to do with the shield.
    assert!(
        !expired[0].state.is_empty(),
        "the phase it is stuck in is carried"
    );

    // THE AGE IS A CLOCK READING, pinned by DIFFERENCE rather than by a magic number or a `>`. A
    // hardcoded age depends on how many ticks the fixture takes to settle; a comparison leaves a false
    // arm nothing can ever reach (the project's own HR5 rule — prefer an equality). Reading the same
    // wedged hand-off a hundred ticks later must report exactly a hundred ticks more, which is the real
    // property: the age tracks the universe clock tick for tick, and cannot be a constant.
    let (_, later) = rig
        .orch
        .world_mut()
        .resource::<SagaRuntimeRes>()
        .arriving_dest_realms(UniverseTick(600), 1);
    assert_eq!(
        later[0].age_ticks.saturating_sub(expired[0].age_ticks),
        100,
        "the reported age advances with the clock"
    );
}

#[test]
fn rehydrate_restores_the_arrival_shield_set() {
    // RESTART-SAFE WITHOUT EXTRA WORK: the shield needs no durable family of its own, because the
    // saga snapshot already persists the whole context including where the subject is going. A
    // rebuilt orchestrator re-derives the identical arrival set on its first sweep — so a crash
    // mid-crossing cannot leave a landing realm unprotected.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.trigger(ctx(
        vd_core::entity_kind::DurabilityClass::Durable,
        Fence(1),
    ));
    rig.settle();
    assert_eq!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .arriving_dest_realms(UniverseTick(1), u64::MAX)
            .0,
        BTreeSet::from([TO_REALM]),
        "the live crossing shields its destination before the crash"
    );

    rig.rebuild(); // KILL-9: the World's RAM is gone; only the committed WAL survives
    assert_eq!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .arriving_dest_realms(UniverseTick(1), u64::MAX)
            .0,
        BTreeSet::from([TO_REALM]),
        "and it shields the SAME destination after the rebuild, re-derived from the snapshot"
    );
}

#[test]
fn rehydrate_keeps_the_configured_liveness_margin() {
    // D-3 (re-audit wf_4d2ca7ae): a kill-9-rebuilt orchestrator must KEEP its prod dead-vs-slow margin
    // (n = 3), NOT drop to the kill-equivalent n = 1 default — `rehydrate` threads `cfg.liveness` into
    // the rebuilt runtime. (The tracker is still rebuilt EMPTY — the CAP freeze — but its TUNING survives.)
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1)); // persist a directory record + the clock ceiling to the store
    rig.settle();
    rig.rebuild_with_liveness(LivenessTuning {
        n_consecutive_unreachable: 3,
        unreachable_window_ticks: 64,
        retry_delay_ticks_hint: 2,
    });
    // The recovered runtime confirms a peer dead only after 3 notices (the n = 1 default would confirm
    // on the first) — proving the configured margin survived the kill-9 recover.
    let mut runtime = rig.orch.world_mut().resource_mut::<SagaRuntimeRes>();
    runtime.liveness.record_unreachable(SOURCE, UniverseTick(1));
    assert!(
        !runtime.liveness.is_confirmed_dead(SOURCE, UniverseTick(1)),
        "the recovered orchestrator kept its configured n = 3 margin, not the n = 1 default"
    );
}

#[test]
fn a_send_shed_is_counted_and_never_confirms_a_live_peer_dead() {
    // R-4d M3 regression (the false-confirm cure): a LOCAL send-shed toward a LIVE peer is a
    // transport backpressure/oversize refusal — it says NOTHING about that peer's liveness. It
    // must be counted (`sends_shed`) and NEVER routed to `record_unreachable`; otherwise a
    // live-but-ack-stalled peer accrues false death evidence and gets destructively re-homed.
    let mut rig = Rig::new();
    // Default margin is n = 1 (one notice confirms). Drive THREE sheds toward DEST — far past the
    // margin. If ANY leaked to `record_unreachable`, DEST would be confirmed dead.
    {
        let mut inbox = rig.orch.world_mut().resource_mut::<InboundBox>();
        inbox.0 = (0..3u64)
            .map(|i| Inbound::SendShed {
                to: DEST,
                class: MsgClass::Saga,
                undelivered: MsgId(i),
                reason: ShedReason::RetryBufferFull,
            })
            .collect();
    }
    // Run the schedule directly (NOT step_tick — its drain would clobber the seeded inbox, and the
    // MemHub cannot produce a shed): drive_sagas reads the seeded InboundBox in place.
    let (world, schedule) = rig.orch.parts_mut();
    schedule.run(world);

    let now = rig.orch.world_mut().resource::<ClockSample>().universe_tick;
    let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
    assert_eq!(
        runtime.sends_shed(),
        3,
        "every shed is counted on sends_shed"
    );
    assert_eq!(
        runtime.liveness_notices(),
        0,
        "no shed ever incremented liveness_notices — proof none reached record_unreachable"
    );
    assert!(
        !runtime.liveness.is_confirmed_dead(DEST, now),
        "a burst of sheds must NEVER confirm a live peer dead (the false-confirm cure)"
    );
}

// ---- D-3 Slice 4: the expiry reaper + CAP gate -------------------------------------------------

fn rec(node: NodeId, lease_expires: u64) -> vd_wire::seams::directory::OwnerRecord {
    vd_wire::seams::directory::OwnerRecord {
        authority: AuthorityRef::Shard(node),
        fence: Fence(1),
        lease_expires: UniverseTick(lease_expires),
        in_transfer: None,
    }
}

#[test]
fn should_reap_requires_quiesced_past_deadline_and_latched_dead() {
    // The Strong-AND CAP gate: reap iff quiesce-elapsed AND now PAST `lease_expires + max` AND
    // PERSISTENTLY (latched) confirmed dead. Covers all corners incl the `<=` deadline boundary.
    let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1 ⇒ latch on first notice
    let dead = NodeId(5);
    liveness.record_unreachable(dead, UniverseTick(100)); // latched (consecutive 1 >= n 1)
    let now = UniverseTick(200);
    let quiesced = UniverseTick(50); // window elapsed (now >= quiesced)
    let max = 10u64; // reassign horizon = lease_expires + 10
    // All three hold → reap (now 200 > lease_expires 90 + max 10 = 100; latched dead; quiesce elapsed).
    assert!(should_reap(&rec(dead, 90), now, &liveness, quiesced, max));
    // (1) NOT past the quiesce freeze → no reap.
    assert!(!should_reap(
        &rec(dead, 90),
        now,
        &liveness,
        UniverseTick(250),
        max
    ));
    // (2a) NOT past the reassign deadline (now <= lease_expires + max) → no reap.
    assert!(!should_reap(&rec(dead, 200), now, &liveness, quiesced, max));
    // (2b) EXACTLY at the deadline (now == lease_expires + max = 200) → the `<=` still blocks (boundary).
    assert!(!should_reap(&rec(dead, 190), now, &liveness, quiesced, max));
    // (3) NOT latched dead (a node with no unreachable evidence) → no reap.
    assert!(!should_reap(
        &rec(NodeId(99), 90),
        now,
        &liveness,
        quiesced,
        max
    ));
}

#[test]
fn confirmed_dead_latch_sets_at_n_and_survives_a_window_reset_until_ack() {
    // The MONOTONE latch: set the tick consecutive reaches `n`, PRESERVED across a window reset (which
    // expires the freshness-gated pulse), cleared only by a live inbound. This is what lets `should_reap`
    // gate on it at the ttl+max horizon without the pulse having expired.
    let tuning = LivenessTuning {
        n_consecutive_unreachable: 3,
        unreachable_window_ticks: 10,
        retry_delay_ticks_hint: 2,
    };
    let mut lv = LivenessTracker::new(tuning);
    let node = NodeId(5);
    // Below n → not latched (and the pulse is not yet confirmed either).
    lv.record_unreachable(node, UniverseTick(0));
    lv.record_unreachable(node, UniverseTick(1));
    assert!(!lv.is_latched_dead(node), "2 < n=3: not yet latched");
    assert!(!lv.is_confirmed_dead(node, UniverseTick(1)));
    // The 3rd consecutive within the window → LATCHED (the pulse is also true right now).
    lv.record_unreachable(node, UniverseTick(2));
    assert!(
        lv.is_latched_dead(node),
        "consecutive reached n=3 → latched"
    );
    assert!(lv.is_confirmed_dead(node, UniverseTick(2)));
    // A notice FAR past the window RESETS the run (100 - 2 > window 10 ⇒ consecutive back to 1), so the
    // PULSE expires — but the monotone latch stays set (the whole point).
    lv.record_unreachable(node, UniverseTick(100));
    assert!(
        !lv.is_confirmed_dead(node, UniverseTick(100)),
        "pulse expired: consecutive reset to 1 < n"
    );
    assert!(
        lv.is_latched_dead(node),
        "the monotone latch survives the reset"
    );
    // A live inbound CLEARS the whole entry → un-latched.
    lv.record_ack(node);
    assert!(!lv.is_latched_dead(node), "record_ack un-latches");
}

#[test]
fn should_reap_reaps_at_ttl_plus_max_after_the_pulse_expires_never_before() {
    // The Strong-AND split-brain guarantee + attack-1 orphan cure, together: (a) the reassign deadline is
    // `lease_expires + max` (NOT the old lapse@ttl that raced the holder's `grace` self-fence), so a
    // lapsed+latched owner is NOT reaped anywhere in (lease_expires, lease_expires+max]; (b) at the horizon
    // the freshness-gated PULSE has long expired, but the MONOTONE latch persists, so failover COMPLETES
    // (the key IS reaped) — no permanent orphan.
    let tuning = LivenessTuning {
        n_consecutive_unreachable: 1,
        unreachable_window_ticks: 10,
        retry_delay_ticks_hint: 2,
    };
    let mut lv = LivenessTracker::new(tuning);
    let dead = NodeId(5);
    lv.record_unreachable(dead, UniverseTick(5)); // latched (n = 1)
    let max = 250u64; // cloud @50Hz: reassign horizon = lease_expires + 250
    let rec = rec(dead, 100); // lease_expires = 100 ⇒ horizon = 350
    let quiesced = UniverseTick(0);
    // NOT reaped at the OLD ttl-ish point (100) nor anywhere up to and including the horizon (350).
    assert!(
        !should_reap(&rec, UniverseTick(100), &lv, quiesced, max),
        "old lapse@ttl must NOT reap"
    );
    assert!(!should_reap(&rec, UniverseTick(200), &lv, quiesced, max));
    assert!(
        !should_reap(&rec, UniverseTick(350), &lv, quiesced, max),
        "not reaped AT the horizon (<=)"
    );
    // At horizon+1 the PULSE is long dead (351 - 5 = 346 ≫ window 10) ...
    assert!(
        !lv.is_confirmed_dead(dead, UniverseTick(351)),
        "the pulse has expired"
    );
    // ... but the LATCH persists ⇒ should_reap reaps (failover completes, no orphan).
    assert!(should_reap(&rec, UniverseTick(351), &lv, quiesced, max));
}

#[test]
fn reaper_revokes_a_lapsed_confirmed_dead_session_only() {
    // The reaper FULLY revokes a dead Session key but LEAVES a dead Realm key (D-37 re-homes it, not
    // the reaper — revoking a realm into HeldNowhere would freeze it). A lapsed-but-NOT-confirmed
    // session is left (CAP). reaper_interval active; renew INERT (the reaper does not need the heartbeat).
    let dir_tuning = DirectoryTuning {
        lease_ttl_ticks: 10,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    };
    let mut dir = DirectoryCore::new(dir_tuning);
    let dead = NodeId(5);
    let live = NodeId(6);
    // Grant at tick 0 ⇒ lease_expires = 10 (lapsed by now = 100).
    let _ = dir.grant(
        DirectoryKey::Session(SessionId(1)),
        AuthorityRef::Gateway(dead),
        Fence(1),
        UniverseTick(0),
    );
    let _ = dir.grant(
        DirectoryKey::Session(SessionId(2)),
        AuthorityRef::Gateway(live),
        Fence(1),
        UniverseTick(0),
    );
    let _ = dir.grant(
        DirectoryKey::Realm(RealmId::System(7)),
        AuthorityRef::Shard(dead),
        Fence(1),
        UniverseTick(0),
    );
    // A lapsed, confirmed-dead session that is MID-TRANSFER (in_transfer-locked): the reaper TRIES
    // to revoke it (should_reap is true) but `revoke` REFUSES a transfer-locked key — the saga owns
    // it, and the saga's own recovery (scan_deadlines) handles the dead participant, not the reaper.
    let _ = dir.grant(
        DirectoryKey::Session(SessionId(3)),
        AuthorityRef::Gateway(dead),
        Fence(1),
        UniverseTick(0),
    );
    assert!(dir.lock_transfer(DirectoryKey::Session(SessionId(3)), TransferId(7)));
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // n = 1
    runtime.liveness.record_unreachable(dead, UniverseTick(50)); // only `dead` is confirmed
    reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
    assert!(
        dir.head(DirectoryKey::Session(SessionId(1))).is_none(),
        "the lapsed, confirmed-dead session is reaped"
    );
    assert!(
        dir.head(DirectoryKey::Session(SessionId(2))).is_some(),
        "a lapsed but NOT-confirmed-dead session is left (CAP — only confirmed deaths reap)"
    );
    assert!(
        dir.head(DirectoryKey::Realm(RealmId::System(7))).is_some(),
        "a dead Realm is LEFT for D-37 forward re-home, never reaped into HeldNowhere"
    );
    assert!(
        dir.head(DirectoryKey::Session(SessionId(3))).is_some(),
        "an in_transfer-locked dead session is NOT reaped — revoke refuses it (the saga owns the key)"
    );
}

#[test]
fn reaper_is_inert_at_zero_interval_and_respects_its_cadence() {
    let dead = NodeId(5);
    // INERT: reaper_interval == 0 never reaps, even a lapsed + confirmed-dead session.
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10,
        ..DirectoryTuning::default() // reaper_interval_ticks = 0
    });
    let _ = dir.grant(
        DirectoryKey::Session(SessionId(1)),
        AuthorityRef::Gateway(dead),
        Fence(1),
        UniverseTick(0),
    );
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    runtime.liveness.record_unreachable(dead, UniverseTick(50));
    reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(10_000));
    assert!(
        dir.head(DirectoryKey::Session(SessionId(1))).is_some(),
        "an inert reaper (interval 0) never reaps"
    );

    // NOT DUE: interval > 0 but the elapsed since the last sweep is below it → no reap this tick.
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    });
    let _ = dir.grant(
        DirectoryKey::Session(SessionId(1)),
        AuthorityRef::Gateway(dead),
        Fence(1),
        UniverseTick(0),
    );
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    runtime.liveness.record_unreachable(dead, UniverseTick(50));
    runtime.last_reap_tick = UniverseTick(100); // just swept at 100
    reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(103)); // 103 - 100 = 3 < 8 → not due
    assert!(
        dir.head(DirectoryKey::Session(SessionId(1))).is_some(),
        "a sweep below the reaper interval since the last is skipped"
    );
}

#[test]
fn reap_in_freezing_orphan_enqueues_a_pending_rehome_and_arms_a_parked_saga() {
    // D-37 Slice 3 (CELL 3): a confirmed-dead + lapsed + UNLOCKED Entity orphan (the post-abort residual
    // of a SOURCE killed in Freezing) is DETECTED by the reaper (ENQUEUED, not revoked) and ARMED by
    // process_rehome_starts — a fresh re-home saga PARKS in ReHoming{target} (the lowest live capable
    // shard), the key is LOCKED (so the next sweep skips it), authority STAYS at the dead owner
    // (conservative — no CAS), and NOTHING is emitted (HR1: no fabricated pose/adopt — owed Slice 4).
    // Proves detection + capability-matched target selection + the durable parked saga + no fabrication.
    // Reverting the reaper Entity arm / process_rehome_starts / start_rehome turns this RED.
    let dead = NodeId(5);
    let target = NodeId(9);
    let entity = DirectoryKey::Entity(subject_eid());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    });
    let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0)); // lease_expires = 10
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // n = 1
    runtime.set_roster(
        [(
            target,
            ShardProfile::build(CapRequest::default()).expect("empty profile"),
        )]
        .into_iter()
        .collect(),
    );
    runtime.liveness.record_unreachable(dead, UniverseTick(50)); // dead CONFIRMED (n = 1)

    // REAP: the orphan is ENQUEUED (not revoked); the directory record is untouched.
    reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
    assert_eq!(
        runtime.pending_rehome,
        vec![PendingReHome {
            subject: entity,
            dead_owner: dead,
            prev_fence: Fence(3),
        }],
        "the reaper enqueues the orphan for a standing re-home (does NOT revoke it)"
    );

    // ARM: a fresh re-home saga parks in ReHoming; the key is locked; authority STAYS at the
    // corpse — and so does the parked TARGET (Stage-C fix): a pre-flush death has no pose, so
    // no live node can be named placeable; Slice 4 selects pose-aware at adopt time.
    let mut outbox = OutboundBox::default();
    process_rehome_starts(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(100),
    );
    assert!(
        runtime.pending_rehome.is_empty(),
        "the queue is drained the same tick (within-barrier hand-off)"
    );
    let transfer = rehome_transfer_id(entity, Fence(3));
    let live = runtime
        .sagas
        .get(&transfer)
        .expect("a fresh re-home saga was armed");
    assert_eq!(
        live.state,
        SagaState::ReHoming {
            target: dead,
            prev_fence: Fence(3),
        },
        "the saga parks in ReHoming at the DEAD owner (no pose ⇒ no placeable target yet)"
    );
    let head = dir
        .head(entity)
        .expect("the orphan record is still present");
    assert_eq!(
        head.authority,
        AuthorityRef::Shard(dead),
        "authority STAYS at the dead owner (conservative — no CAS, no HeldNowhere strand)"
    );
    assert_eq!(
        head.fence,
        Fence(3),
        "no fence bump (no ReHomeCommit until Slice 4)"
    );
    assert!(
        head.in_transfer.is_some(),
        "the key is LOCKED by the armed re-home saga (the next reaper sweep skips it)"
    );
    assert!(
        outbox.0.is_empty(),
        "NO fabrication: a parked standing re-home emits no ReHome envelope (the adopt is owed Slice 4)"
    );
}

#[test]
fn reaper_leaves_a_locked_dead_entity_for_its_owning_saga() {
    // D-37 Slice 3: an Entity that is dead + lapsed but IN-TRANSFER-LOCKED (a live saga — e.g. an
    // in-flight CELL-1/2 re-home — owns the key) is NOT standing-re-homed; the owning saga's own recovery
    // handles it. Covers the reaper Entity-arm GUARD false branch (in_transfer.is_some()).
    let dead = NodeId(5);
    let entity = DirectoryKey::Entity(subject_eid());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    });
    let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
    assert!(
        dir.lock_transfer(entity, TransferId(1)),
        "a live saga locks the key"
    );
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    runtime.liveness.record_unreachable(dead, UniverseTick(50));
    reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
    assert!(
        runtime.pending_rehome.is_empty(),
        "a LOCKED dead Entity is left to its owning saga, never standing-re-homed"
    );
}

#[test]
fn reaper_leaves_a_dead_entity_a_live_post_commit_saga_still_owns() {
    // AUDIT wf_3b9eb7f0 (HIGH): commit_cas CLEARS in_transfer at the commit point, so a POST-commit
    // Promoting saga's key is UNLOCKED yet still owned by that live in-flight saga (which re-homes the
    // dead committed owner ITSELF via scan_deadlines). The reaper must NOT arm a SECOND standing
    // re-home on it — it cross-checks runtime.sagas (`subject_has_live_saga`), not just in_transfer, so
    // "one re-home arm per key" is an ENFORCED invariant. Reverting the `& !subject_has_live_saga`
    // guard turns this RED (the reaper would double-arm + leak a parked saga). Covers the
    // subject_has_live_saga TRUE arm + the guard's has-live-saga false branch.
    let dead = NodeId(5);
    let entity = DirectoryKey::Entity(subject_eid());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    });
    // The committed key is UNLOCKED (a post-commit saga — commit_cas cleared in_transfer)...
    let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    // ...but a LIVE Promoting saga still owns the subject (ctx.subject == entity).
    inject_saga(
        &mut runtime,
        SagaState::Promoting {
            new_fence: Fence(3),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: None,
        },
        UniverseTick(0),
    );
    runtime.liveness.record_unreachable(dead, UniverseTick(50));
    reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
    assert!(
        runtime.pending_rehome.is_empty(),
        "an unlocked dead-owner Entity a LIVE saga still owns is NOT double-armed (cross-checks sagas)"
    );
}

#[test]
fn process_rehome_parks_when_no_live_target() {
    // D-37 Slice 3 → Stage-C fix: the standing re-home parks UNCONDITIONALLY (even at whole-pool
    // death) — the park names the dead owner, the key LOCKS (no reaper churn), and the future
    // Slice-4 adopt selects its target pose-aware. Nothing waits for a roster node any more,
    // because no roster node can be named placeable without a pose.
    let dead = NodeId(5);
    let entity = DirectoryKey::Entity(subject_eid());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    });
    let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // EMPTY roster (default)
    runtime.liveness.record_unreachable(dead, UniverseTick(50));
    reap_lapsed_leases(&mut runtime, &mut dir, UniverseTick(100));
    assert_eq!(
        runtime.pending_rehome.len(),
        1,
        "the orphan was detected + enqueued"
    );
    let mut outbox = OutboundBox::default();
    process_rehome_starts(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(100),
    );
    assert_eq!(
        runtime.live(),
        1,
        "the standing re-home parks unconditionally — even with an empty roster"
    );
    assert!(
        dir.head(entity)
            .expect("record present")
            .in_transfer
            .is_some(),
        "the key LOCKS with the park (no reaper re-detection churn)"
    );
}

#[test]
fn process_rehome_skips_an_already_locked_key() {
    // D-37 Slice 3: if the orphan key is ALREADY locked (a concurrent arm / a prior sweep's saga) when
    // process_rehome_starts drains it, lock_transfer returns false and the entry is skipped — one key,
    // one re-home saga. Covers the lock-false arm.
    let dead = NodeId(5);
    let target = NodeId(9);
    let entity = DirectoryKey::Entity(subject_eid());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10,
        reaper_interval_ticks: 8,
        ..DirectoryTuning::default()
    });
    let _ = dir.grant(entity, AuthorityRef::Shard(dead), Fence(3), UniverseTick(0));
    assert!(
        dir.lock_transfer(entity, TransferId(1)),
        "pre-lock by another saga"
    );
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    runtime.set_roster(
        [(
            target,
            ShardProfile::build(CapRequest::default()).expect("empty profile"),
        )]
        .into_iter()
        .collect(),
    );
    runtime.pending_rehome.push(PendingReHome {
        subject: entity,
        dead_owner: dead,
        prev_fence: Fence(3),
    });
    let mut outbox = OutboundBox::default();
    process_rehome_starts(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(100),
    );
    assert_eq!(
        runtime.live(),
        0,
        "the already-locked key is skipped — no second re-home saga"
    );
}

#[test]
fn a_re_trigger_of_a_live_transient_saga_does_not_clobber_it() {
    // Audit D6-1: a DUPLICATE trigger for an in-flight transient `BatchId` is a no-op — the live
    // `BatchHandoff` saga + its committed go-token are untouched (one key → one saga, never reset).
    // The durable path is already guarded by `lock_transfer`; this covers the transient path (no lock).
    let mut rig = Rig::new();
    rig.trigger(transient_ctx(Fence(9)));
    rig.settle(); // → BatchHandoff{AwaitAdopt}, go-token committed (batch_go_writes == 1)
    rig.trigger(transient_ctx(Fence(9))); // a spurious/duplicate producer re-trigger
    rig.settle();
    let runtime = rig.orch.world_mut().resource::<SagaRuntimeRes>();
    assert_eq!(
        runtime.live(),
        1,
        "the re-trigger did NOT start a second saga (guarded)"
    );
    assert_eq!(
        runtime.batch_go_writes(),
        1,
        "the live saga was NOT clobbered + its go-token NOT re-committed"
    );
}

#[test]
fn a_stray_ack_to_a_parked_saga_preserves_its_staleness_anchor() {
    // AAA-1-SINCE regression guard: a PARKED saga that receives a duplicate / stray ack the FSM
    // absorbs as same-state must KEEP its `since` — an unconditional bump would reset stuck-saga
    // staleness on every at-least-once redelivery. (Exercises the same-state arm of the
    // `final_state != prior` gate in `commit_result`.) D-7 moved this OFF the old transient
    // CommittingCas park (transients no longer park) onto a DURABLE saga held in `Demoting`
    // awaiting its `DemoteAck`: a stray duplicate `Committed` (route-swap) ack is absorbed there
    // as same-state (`RouteSwapped` is not handled in `Demoting`), the genuine post-commit park.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
    rig.settle();
    rig.ack(TransferControlAck::Prepared {
        transfer: XFER,
        result: PrepareResult::Ready,
    });
    rig.ack(TransferControlAck::CutConfirmed {
        transfer: XFER,
        marker_seq: 3,
    });
    rig.ack(TransferControlAck::SourceFrozen {
        transfer: XFER,
        drained_seq: 3,
    });
    rig.flush(); // both gate conditions → the DIRECT CAS wins → Swapping
    rig.ack(TransferControlAck::Committed { transfer: XFER }); // → Demoting (parked, awaiting DemoteAck)
    let _ = (rig.drain_gateway(), rig.drain_source());
    let since_parked = crate::orchestrator::admin_snapshot(rig.orch.world_mut(), 0).sagas[0].since;

    // Two duplicate Committed acks (at-least-once redelivery of the route-swap ack). The FSM
    // absorbs each as same-state in Demoting (RouteSwapped is not handled there); the clock
    // advances on every step. (Well within the redrive deadline, so the producer does not fire.)
    rig.ack(TransferControlAck::Committed { transfer: XFER });
    rig.ack(TransferControlAck::Committed { transfer: XFER });

    let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut(), 0);
    assert_eq!(
        snap.sagas.len(),
        1,
        "still parked in Demoting, not advanced by stray acks"
    );
    assert_eq!(
        snap.sagas[0].state, "Demoting { new_fence: Fence(2), dest_delivered: false }",
        "state unchanged by the stray acks"
    );
    assert_eq!(
        snap.sagas[0].since.0, since_parked.0,
        "the staleness anchor is NOT reset by a stray same-state ack"
    );
    assert!(
        snap.universe_tick > since_parked.0,
        "the clock DID advance — proving `since` was held, not re-stamped to `now`"
    );
}

#[test]
fn active_transfers_reports_the_live_triple_and_is_empty_when_idle() {
    // 1d.5b.3d: the typed mid-flight ground truth the AUTHORITY-UNIQUE oracle excuses against —
    // each live saga's (subject, source, dest); empty once tombstoned (post-quiesce ⇒ strict).
    let mut rt = SagaRuntimeRes::default();
    assert!(rt.active_transfers().is_empty());
    rt.sagas.insert(
        XFER,
        LiveSaga {
            ctx: ctx(vd_core::entity_kind::DurabilityClass::Durable, Fence(1)),
            state: SagaState::Demoting {
                new_fence: Fence(2),
                dest_delivered: false,
            },
            gateway: GATEWAY,
            since: UniverseTick(0),
            flushed_pose: None,
            flushed_state: Vec::new(),
            dead_observed_since: None,
            dest_adopted: false,
            opened: UniverseTick(0),
        },
    );
    assert_eq!(
        rt.active_transfers(),
        vec![ActiveTransfer {
            subject: subject(),
            source: SOURCE,
            dest: DEST,
        }],
    );
}

/// Build a live saga in `state` for the arrival-classifier tests.
fn live_in(state: SagaState, class: vd_core::entity_kind::DurabilityClass) -> LiveSaga {
    LiveSaga {
        ctx: ctx(class, Fence(1)),
        state,
        gateway: GATEWAY,
        since: UniverseTick(0),
        flushed_pose: None,
        flushed_state: Vec::new(),
        dead_observed_since: None,
        dest_adopted: false,
        opened: UniverseTick(0),
    }
}

#[test]
fn arrival_dest_classifies_every_saga_shape() {
    use vd_core::entity_kind::DurabilityClass::Durable;
    // THE classification table. Every variant of the phase enum appears exactly once, so adding a
    // phase without deciding whether someone is arriving during it breaks this test AND the
    // wildcard-free match. `Done`/`Aborted` are unreachable through the live system (they tombstone
    // in the same barrier they are reached in) — they are covered here or nowhere.
    let arriving = [
        SagaState::AwaitProvision,
        SagaState::Preparing,
        SagaState::Cutting,
        SagaState::Freezing {
            marker_seq: 1,
            frozen_drained: None,
            flushed: false,
        },
        SagaState::CommittingCas {
            marker_seq: 1,
            drained_seq: 1,
        },
        SagaState::BatchCommitting {
            step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
        },
        SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitAdopt,
            new_fence: Fence(2),
        },
        SagaState::Swapping {
            new_fence: Fence(2),
        },
        SagaState::Demoting {
            new_fence: Fence(2),
            dest_delivered: false,
        },
        SagaState::Promoting {
            new_fence: Fence(2),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: None,
        },
        SagaState::Releasing {
            new_fence: Fence(2),
        },
    ];
    for state in arriving {
        assert_eq!(
            arrival_dest(&live_in(state, Durable)),
            Some(TO_REALM),
            "somebody is on their way into the destination during {state:?}"
        );
    }

    let not_arriving = [
        // Compensating / terminal: the subject is going back to the source, or is already settled.
        SagaState::Aborting {
            reason: AbortReason::CasLost,
            awaiting_thaw: false,
            awaiting_abort_ack: false,
        },
        SagaState::Aborted {
            reason: AbortReason::CasLost,
        },
        SagaState::Done {
            new_fence: Fence(2),
        },
        // Re-home: a hand-off between MACHINES, not between PLACES.
        SagaState::ReHoming {
            target: DEST,
            prev_fence: Fence(1),
        },
        SagaState::Promoting {
            new_fence: Fence(2),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: Some(DEST),
        },
    ];
    for state in not_arriving {
        assert_eq!(
            arrival_dest(&live_in(state, Durable)),
            None,
            "no arrival is pending during {state:?}"
        );
    }
}

#[test]
fn arriving_dest_realms_excludes_a_parked_rehome_saga() {
    // THE LEAK THAT WOULD HAVE SHIPPED. A standing re-home fabricates its realm fields and parks
    // indefinitely BY DESIGN, so a naive "every live saga's destination" projection would pin one
    // made-up realm alive forever per orphaned entity, starting at the first machine failure.
    let mut rt = SagaRuntimeRes::default();
    rt.sagas.insert(
        XFER,
        LiveSaga {
            ctx: rehome_ctx(subject(), Fence(1), SOURCE, DEST),
            state: SagaState::ReHoming {
                target: DEST,
                prev_fence: Fence(1),
            },
            gateway: SOURCE,
            since: UniverseTick(0),
            flushed_pose: None,
            flushed_state: Vec::new(),
            dead_observed_since: None,
            dest_adopted: false,
            opened: UniverseTick(0),
        },
    );
    assert_eq!(
        rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
        BTreeSet::new()
    );
}

#[test]
fn arriving_dest_realms_includes_a_transient_batch_dest() {
    // The gap the source-side keep-alive never covered at all: a batch of non-persistent things
    // (dropped items, debris) crossing over has no per-entity latch, so nothing was demanding its
    // destination. The shield is class-blind — ONE machinery, both durability classes (HR2).
    let mut rt = SagaRuntimeRes::default();
    rt.sagas.insert(
        XFER,
        live_in(
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitPromote,
                new_fence: Fence(2),
            },
            vd_core::entity_kind::DurabilityClass::Transient,
        ),
    );
    assert_eq!(
        rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
        BTreeSet::from([TO_REALM]),
        "a transient batch's destination is shielded exactly like a player's"
    );
}

#[test]
fn arriving_dest_realms_is_empty_after_the_saga_tombstones() {
    // The shield has no clearing path to forget because it is derived, not stored: the same
    // `remove` that tombstones the saga retires the shield, in the same barrier, for free.
    let mut rt = SagaRuntimeRes::default();
    rt.sagas.insert(
        XFER,
        live_in(
            SagaState::Promoting {
                new_fence: Fence(2),
                promote_acked: false,
                dest_delivered: false,
                rehome_target: None,
            },
            vd_core::entity_kind::DurabilityClass::Durable,
        ),
    );
    assert_eq!(
        rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
        BTreeSet::from([TO_REALM])
    );
    rt.sagas.remove(&XFER);
    assert_eq!(
        rt.arriving_dest_realms(UniverseTick(1), u64::MAX).0,
        BTreeSet::new()
    );
}

#[test]
fn deadline_for_maps_every_phase_to_its_risk_class() {
    // Slice 2a: pre-freeze/freeze/aborting → the LARGE destructive abort deadline; committing/
    // post-commit → the cheap redrive deadline; terminal → u64::MAX (never fires).
    let t = SagaTuning {
        redrive_deadline_ticks: 8,
        abort_deadline_ticks: 24,
    };
    for s in [
        SagaState::AwaitProvision,
        SagaState::Preparing,
        SagaState::Cutting,
        SagaState::Freezing {
            marker_seq: 0,
            frozen_drained: None,
            flushed: false,
        },
        SagaState::Aborting {
            reason: AbortReason::CutTimeout,
            awaiting_thaw: true,
            awaiting_abort_ack: true,
        },
    ] {
        assert_eq!(deadline_for(&s, &t), 24, "{s:?} uses the abort deadline");
    }
    for s in [
        SagaState::CommittingCas {
            marker_seq: 0,
            drained_seq: 0,
        },
        SagaState::BatchCommitting { step_id: 11 },
        SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitPromote,
            new_fence: Fence(2),
        },
        SagaState::Swapping {
            new_fence: Fence(2),
        },
        SagaState::Demoting {
            new_fence: Fence(2),
            dest_delivered: false,
        },
        SagaState::Promoting {
            new_fence: Fence(2),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: None,
        },
        SagaState::ReHoming {
            target: NodeId(4),
            prev_fence: Fence(2),
        },
        SagaState::Releasing {
            new_fence: Fence(2),
        },
    ] {
        assert_eq!(deadline_for(&s, &t), 8, "{s:?} uses the redrive deadline");
    }
    assert_eq!(
        deadline_for(
            &SagaState::Done {
                new_fence: Fence(2)
            },
            &t
        ),
        u64::MAX
    );
    assert_eq!(
        deadline_for(
            &SagaState::Aborted {
                reason: AbortReason::CutTimeout
            },
            &t
        ),
        u64::MAX
    );
}

/// Inject a live saga directly (controlled state + `since`) for the producer tests.
fn inject_saga(runtime: &mut SagaRuntimeRes, state: SagaState, since: UniverseTick) {
    runtime.sagas.insert(
        XFER,
        LiveSaga {
            ctx: ctx(DurabilityClass::Durable, Fence(1)),
            state,
            gateway: GATEWAY,
            since,
            flushed_pose: None,
            flushed_state: Vec::new(),
            dead_observed_since: None,
            dest_adopted: false,
            opened: UniverseTick(0),
        },
    );
}

fn flows_to_node(outbox: &OutboundBox, node: NodeId) -> Vec<InterShardFlow> {
    outbox
        .0
        .iter()
        .filter(|(to, _, _, _)| *to == node)
        .filter_map(|(_, _, b, _)| postcard::from_bytes(b).ok())
        .collect()
}

#[test]
fn scan_deadlines_re_drives_a_due_saga_re_arms_it_and_skips_a_fresh_one() {
    // THE R1 cure: a post-commit (Demoting) saga whose ack was lost re-drives at the redrive
    // deadline (8); the re-arm (since←now) means it fires at most ONCE per window, never every tick.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8, abort=24
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    inject_saga(
        &mut runtime,
        SagaState::Demoting {
            new_fence: Fence(2),
            dest_delivered: false,
        },
        UniverseTick(0),
    );

    // BELOW the deadline (now=7 < 8): nothing re-driven.
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(7),
    );
    assert!(outbox.0.is_empty(), "below the deadline: no re-drive");

    // AT the deadline (now=8): Timeout → Demoting re-emits the Demote to the SOURCE.
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8),
    );
    let expected = InterShardFlow::Demote(DemoteCmd {
        transfer: XFER,
        subject: subject(),
        new_owner_fence: Fence(2),
        step_id: DEMOTE_STEP,
    });
    assert!(
        flows_to_node(&outbox, SOURCE).contains(&expected),
        "the deadline producer re-drove the Demote: {:?}",
        outbox.0
    );

    // RE-ARM: `since` refreshed to 8, so the next scan within the window (now=9, 9-8=1 < 8) is silent.
    let mut outbox2 = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox2,
        EpochId(1),
        UniverseTick(9),
    );
    assert!(
        outbox2.0.is_empty(),
        "within the re-armed window: no second re-drive (no per-tick storm)"
    );
}

#[test]
fn scan_deadlines_re_drives_a_parked_aborting_saga() {
    // finding #6: a saga PARKED in Aborting (a dropped compensator ack) is re-driven by the
    // producer — at the LARGE abort deadline (24) — re-emitting BOTH outstanding compensators.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    inject_saga(
        &mut runtime,
        SagaState::Aborting {
            reason: AbortReason::FreezeTimeout,
            awaiting_thaw: true,
            awaiting_abort_ack: true,
        },
        UniverseTick(0),
    );

    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(23),
    );
    assert!(outbox.0.is_empty(), "below the abort deadline: no re-drive");

    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(24),
    );
    let to_gateway = flows_to_node(&outbox, GATEWAY);
    assert!(
        to_gateway.contains(&InterShardFlow::Saga(TransferControl::ThawSource {
            transfer: XFER,
            session: SESSION,
        })),
        "the producer re-drove ThawSource: {to_gateway:?}"
    );
    assert!(
        to_gateway.contains(&InterShardFlow::Saga(TransferControl::AbortTransfer {
            transfer: XFER,
            session: SESSION,
        })),
        "the producer re-drove AbortTransfer: {to_gateway:?}"
    );
}

#[test]
fn scan_deadlines_self_promotes_a_demoting_saga_with_a_confirmed_dead_source() {
    // D-37 CELL 1: a DUE post-commit `Demoting` saga whose SOURCE is confirmed-dead self-promotes the
    // already-committed live dest — re-driving the ordered Demote toward the corpse would PARK forever.
    // The producer injects SourceUnreachable (non-destructive, the cheap redrive deadline; n==1 confirms
    // on the first NodeUnreachable), the FSM transitions Demoting→Promoting and emits the Promote to the
    // DEST (NOT a Demote toward the dead source). The saga stays LIVE — Promoting awaits PromoteAck +
    // DestDelivered (the Releasing gate, never Done-on-promote: the D-36 starved-watermark caveat holds).
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    inject_saga(
        &mut runtime,
        SagaState::Demoting {
            new_fence: Fence(2),
            dest_delivered: false,
        },
        UniverseTick(0),
    );
    runtime.liveness.record_unreachable(SOURCE, UniverseTick(8)); // confirmed dead (n == 1)
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8),
    );
    assert!(
        flows_to_node(&outbox, DEST).contains(&InterShardFlow::Promote(PromoteCmd {
            transfer: XFER,
            subject: subject(),
            new_fence: Fence(2),
            step_id: PROMOTE_STEP,
            source: SOURCE,
        })),
        "the dead-source Demoting saga self-promotes the committed dest: {:?}",
        outbox.0
    );
    assert!(
        flows_to_node(&outbox, SOURCE).is_empty(),
        "no Demote is re-driven toward the dead source"
    );
    assert_eq!(
        runtime.live(),
        1,
        "still LIVE — Promoting awaits PromoteAck + DestDelivered (never Done-on-promote, D-36)"
    );
    assert_eq!(runtime.source_unreachable_resolutions(), 1);
}

#[test]
fn select_rehome_target_takes_only_the_pose_realms_live_capable_owner() {
    use std::collections::BTreeMap;
    use vd_sim::capability::{CapRequest, ShardProfile, VoxelGeometry};
    let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
    let cartesian = ShardProfile::build(CapRequest {
        voxel: Some(VoxelGeometry::Cartesian),
        ..CapRequest::default()
    })
    .expect("cartesian profile");
    let roster: BTreeMap<NodeId, ShardProfile> = [
        (NodeId(2), cartesian),
        (NodeId(3), empty),
        (NodeId(4), cartesian),
    ]
    .into_iter()
    .collect();
    let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1
    let req = CapRequest {
        voxel: Some(VoxelGeometry::Cartesian),
        ..CapRequest::default()
    };
    // The pose realm's live, capable owner IS the target — other live capable nodes are NEVER
    // considered (the realm-blind pick was the audit's critical: a receiver refuses a pose it
    // cannot place, so only the placeable owner may be named).
    assert_eq!(
        select_rehome_target(&req, Some(NodeId(4)), &roster, &liveness, UniverseTick(5)),
        Some(NodeId(4)),
        "the pose realm's live capable owner is the target"
    );
    // A DEAD owner ⇒ None (park; the next scan fire re-resolves the head) — even with other
    // live capable nodes in the roster.
    liveness.record_unreachable(NodeId(4), UniverseTick(5));
    assert_eq!(
        select_rehome_target(&req, Some(NodeId(4)), &roster, &liveness, UniverseTick(5)),
        None,
        "a dead owner parks the saga; no realm-blind fallback exists"
    );
    // An INCAPABLE owner (profile cannot satisfy the caps) ⇒ None — never a forced re-home.
    assert_eq!(
        select_rehome_target(&req, Some(NodeId(3)), &roster, &liveness, UniverseTick(5)),
        None,
        "an incapable owner parks the saga"
    );
    // An owner ABSENT from the static roster (a demand-spawned shard) is accepted: hosting the
    // pose's realm is the strongest capability statement available.
    assert_eq!(
        select_rehome_target(&req, Some(NodeId(77)), &roster, &liveness, UniverseTick(5)),
        Some(NodeId(77)),
        "a demand-spawned owner outside the roster is accepted"
    );
}

#[test]
fn select_rehome_target_parks_without_a_pose_realm_owner_and_serves_an_empty_req() {
    use std::collections::BTreeMap;
    use vd_sim::capability::{CapRequest, ShardProfile};
    let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
    let liveness = LivenessTracker::new(LivenessTuning::default());
    let only_stub: BTreeMap<NodeId, ShardProfile> = [(NodeId(3), empty)].into_iter().collect();
    // NO resolvable pose-realm owner (no stashed pose / a frame with no single realm owner /
    // no directory record) ⇒ None — the saga PARKS honestly; the roster is never scanned.
    assert_eq!(
        select_rehome_target(
            &CapRequest::default(),
            None,
            &only_stub,
            &liveness,
            UniverseTick(0)
        ),
        None,
        "no placeable owner ⇒ park; there is no realm-blind fallback"
    );
    // For the P3 EMPTY (bare-point) req, a live stub owner IS the target.
    assert_eq!(
        select_rehome_target(
            &CapRequest::default(),
            Some(NodeId(3)),
            &only_stub,
            &liveness,
            UniverseTick(0)
        ),
        Some(NodeId(3)),
        "an empty bare-point req is satisfied by the live stub owner"
    );
}

#[test]
fn rehome_event_for_promoting_dead_dest_rehomes_past_budget_else_redrives() {
    // D-37 CELL 2 producer — the 4 corners of the Promoting-dead-dest branch (HR5: all covered here in
    // the monomorphic helper). ctx: source=SOURCE(2), dest=DEST(3). abort_deadline=24 (SagaTuning::default).
    let tuning = SagaTuning::default();
    let c = ctx(DurabilityClass::Durable, Fence(1));
    let promoting = SagaState::Promoting {
        new_fence: Fence(2),
        promote_acked: false,
        dest_delivered: false,
        rehome_target: None,
    };
    let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
    let roster: BTreeMap<NodeId, ShardProfile> = [(NodeId(9), empty)].into_iter().collect();
    let req = CapRequest::default();
    let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1

    // (i) dest HEALTHY (not confirmed dead) → Timeout, and a stale budget is cleared.
    let mut dos = Some((DEST, UniverseTick(5)));
    let ev = rehome_event_for(
        &promoting,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(30),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(ev, SagaEvent::Timeout);
    assert_eq!(dos, None, "a healthy dest clears the stale abort budget");

    // Confirm DEST dead (n == 1 → one notice confirms).
    liveness.record_unreachable(DEST, UniverseTick(0));
    // (ii) dest DEAD but WITHIN the abort budget → cheap Timeout re-drive (budget anchored at first fire).
    let mut dos = None;
    let ev = rehome_event_for(
        &promoting,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(0),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(ev, SagaEvent::Timeout);
    assert_eq!(
        dos,
        Some((DEST, UniverseTick(0))),
        "the abort budget anchors on the first dead observation (keyed by the dead node)"
    );
    let ev = rehome_event_for(
        &promoting,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(10),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(
        ev,
        SagaEvent::Timeout,
        "still within the 24-tick budget at tick 10"
    );

    // (iii) dest DEAD, PAST budget, a capable LIVE target exists → ReHomeTo{target}.
    let ev = rehome_event_for(
        &promoting,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(24),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(
        ev,
        SagaEvent::ReHomeTo { target: NodeId(9) },
        "past budget → forward re-home to the live target"
    );

    // (iv) dest DEAD, PAST budget, NO placeable owner (no resolvable pose-realm head) → Timeout
    // (stay PARKED, honest) — the roster is never a fallback.
    let mut dos = Some((DEST, UniverseTick(0)));
    let ev = rehome_event_for(
        &promoting,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(24),
        &roster,
        &req,
        DEST,
        None,
        false,
    );
    assert_eq!(
        ev,
        SagaEvent::Timeout,
        "no placeable live owner → the saga stays parked (honest)"
    );
}

#[test]
fn deliver_rehome_to_commits_to_the_target_and_emits_the_dedicated_adopt() {
    // D-37 CELL 2 executor: a Promoting saga whose committed dest is now dead, delivered ReHomeTo
    // {target}, re-homes onto the live target. ReHomeCommit re-points commit_cas to the target (the
    // fence-monotone bump 1→2 strictly stales the dead dest AND names the target BEFORE the adopt);
    // CasWon → Promoting + ReHomeAdopt emits the DEDICATED ReHome envelope from the stashed pose.
    // Covers the ReHomeCommit + ReHomeAdopt executor arms + emit_rehome's Some arm end-to-end.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    // The subject is committed to the (now-dead) DEST at Fence(1) — the re-home CAS expectation.
    let _ = dir.grant(
        subject(),
        AuthorityRef::Shard(DEST),
        Fence(1),
        UniverseTick(0),
    );
    inject_saga(
        &mut runtime,
        SagaState::Promoting {
            new_fence: Fence(1),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: None,
        },
        UniverseTick(0),
    );
    stash_flush(&mut runtime, XFER, flushed_pose(), Vec::new()); // the adopt payload
    let target = NodeId(9);
    let mut outbox = OutboundBox::default();
    deliver(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(5),
        XFER,
        SagaEvent::ReHomeTo { target },
    );
    // The directory now names the live TARGET at the bumped fence (fence-monotone 1 → 2).
    let head = dir.head(subject()).expect("subject still recorded");
    assert_eq!(
        head.authority,
        AuthorityRef::Shard(target),
        "re-home committed authority to the live target"
    );
    assert_eq!(
        head.fence,
        Fence(2),
        "fence-monotone bump (1→2) strictly stales the dead dest"
    );
    // The DEDICATED ReHome adopt (NOT a Promote) was emitted to the target from the stashed pose.
    let expected = InterShardFlow::ReHome(ReHomeCmd {
        transfer: XFER,
        universe_epoch: EpochId(1),
        subject: subject(),
        new_fence: Fence(2),
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(flushed_pose()),
        source: SOURCE,
    });
    assert!(
        flows_to_node(&outbox, target).contains(&expected),
        "the dedicated ReHome adopt was emitted to the target: {:?}",
        outbox.0
    );
}

#[test]
fn a_rehomed_promoting_redrives_the_adopt_to_the_live_target_via_scan_deadlines() {
    // D-37 Slice 2d END-TO-END — the SELF-SUFFICIENT re-drive proven through the FULL producer chain
    // (scan_deadlines → rehome_event_for → deliver → FSM Some-arm → emit_rehome), NOT just the FSM unit.
    // A Promoting saga that ALREADY re-homed (`rehome_target: Some(target)`, the directory naming the
    // LIVE target at the bumped fence) is driven past the redrive deadline → rehome_event_for returns
    // Timeout (the owner is alive, keyed on `dir.head` not `ctx.dest`) → the FSM Some-arm emits
    // A::ReHomeAdopt → emit_rehome RE-SENDS the dedicated ReHome to the target from the stashed pose.
    // This proves the ORCHESTRATOR OWNS the adopt re-drive (no transport at-least-once needed) and is
    // the regression guard the CELL-2 crash matrix LACKS: the matrix's FIRST adopt lands over the perfect
    // FaultFabric link so it never runs this arm — reverting the Some-arm to always-Promote leaves the
    // matrix green but REDs this test (it would aim a Promote at the dead `ctx.dest`, not a ReHome at the
    // live target). Fence-monotone holds: re-driving the ADOPT re-bumps nothing (only ReHomeCommit bumps).
    let target = NodeId(9);
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive deadline = 8
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    // POST-re-home directory state: the subject is committed to the LIVE target at the bumped Fence(2).
    let _ = dir.grant(
        subject(),
        AuthorityRef::Shard(target),
        Fence(2),
        UniverseTick(0),
    );
    inject_saga(
        &mut runtime,
        SagaState::Promoting {
            new_fence: Fence(2),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: Some(target),
        },
        UniverseTick(0),
    );
    stash_flush(&mut runtime, XFER, flushed_pose(), Vec::new()); // the adopt payload, re-read on every re-drive
    // The producer drives the DUE saga (now=8 >= redrive 8); a LIVE owner ⇒ Timeout (not ReHomeTo).
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8),
    );
    // The dedicated ReHome adopt was RE-EMITTED to the live target from the stashed pose.
    let expected = InterShardFlow::ReHome(ReHomeCmd {
        transfer: XFER,
        universe_epoch: EpochId(1),
        subject: subject(),
        new_fence: Fence(2),
        step_id: RE_HOME_STEP,
        state: ReHomeState::PoseOnly(flushed_pose()),
        source: SOURCE,
    });
    assert!(
        flows_to_node(&outbox, target).contains(&expected),
        "the re-drive re-sent the dedicated ReHome adopt to the live target: {:?}",
        outbox.0
    );
    // NOT a Promote at the (dead) original dest — the explicit regression guard for the Some-arm.
    assert!(
        flows_to_node(&outbox, DEST).is_empty(),
        "the re-drive does NOT aim a Promote at the dead original dest: {:?}",
        outbox.0
    );
    // Re-driving the ADOPT re-bumps nothing; the saga stays live (forward-only re-drive).
    assert_eq!(
        dir.head(subject()).expect("subject recorded").fence,
        Fence(2),
        "re-driving the adopt does not re-bump the fence (only ReHomeCommit bumps)"
    );
    assert_eq!(runtime.live(), 1, "the re-drive keeps the saga live");
}

#[test]
fn rehome_adopt_is_a_loud_no_op_for_a_non_entity_subject_or_missing_pose() {
    // build_rehome / emit_rehome gate on an Entity subject AND a stashed pose (mirrors build/emit_
    // crossing) so the executor ReHomeAdopt arm stays branchless (HR5). A Realm subject (no
    // transfer_subject_entity) ⇒ None; an Entity subject with NO pose ⇒ None; both ⇒ Some.
    let realm_ctx = SagaCtx {
        exterior: false,
        subject: DirectoryKey::Realm(RealmId::System(9)),
        ..ctx(DurabilityClass::Durable, Fence(1))
    };
    assert!(
        build_rehome(&realm_ctx, Fence(2), Some(flushed_pose()), EpochId(1)).is_none(),
        "non-Entity subject ⇒ None"
    );
    let entity_ctx = ctx(DurabilityClass::Durable, Fence(1)); // subject() is an Entity
    assert!(
        build_rehome(&entity_ctx, Fence(2), None, EpochId(1)).is_none(),
        "missing flushed pose ⇒ None"
    );
    assert!(
        build_rehome(&entity_ctx, Fence(2), Some(flushed_pose()), EpochId(1)).is_some(),
        "Entity subject + a stashed pose ⇒ Some"
    );
    // emit_rehome's None arm: a non-Entity re-home adopt emits NOTHING (the LOUD no-op).
    let mut outbox = OutboundBox::default();
    emit_rehome(
        &realm_ctx,
        Fence(2),
        NodeId(9),
        Some(flushed_pose()),
        EpochId(1),
        &mut outbox,
    );
    assert!(
        outbox.0.is_empty(),
        "a non-Entity re-home adopt emits no envelope"
    );
}

#[test]
fn deliver_rehome_to_aborts_cleanly_when_the_cas_is_lost() {
    // D-37 rule 3: if another writer moved the fence past prev_fence, the re-home CAS LOSES — the saga
    // is the loser and tombstones as a clean no-op (the entity is alive at the CAS winner; no
    // compensation, no ReHome envelope). Covers the ReHomeCommit executor's Lost arm end-to-end.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    // Drive the head AHEAD (Fence 2 @ SOURCE) of the saga's expectation (Fence 1) — someone else won.
    let _ = dir.grant(
        subject(),
        AuthorityRef::Shard(DEST),
        Fence(1),
        UniverseTick(0),
    );
    let _ = dir.commit_cas(
        subject(),
        Fence(1),
        AuthorityRef::Shard(SOURCE),
        UniverseTick(0),
    );
    inject_saga(
        &mut runtime,
        SagaState::Promoting {
            new_fence: Fence(1),
            promote_acked: false,
            dest_delivered: false,
            rehome_target: None,
        },
        UniverseTick(0),
    );
    stash_flush(&mut runtime, XFER, flushed_pose(), Vec::new());
    let mut outbox = OutboundBox::default();
    deliver(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(5),
        XFER,
        SagaEvent::ReHomeTo { target: NodeId(9) },
    );
    assert_eq!(
        runtime.live(),
        0,
        "the re-home loser tombstoned (clean no-op)"
    );
    let head = dir.head(subject()).expect("subject still recorded");
    assert_eq!(
        head.authority,
        AuthorityRef::Shard(SOURCE),
        "the CAS winner still owns the entity"
    );
    assert_eq!(
        head.fence,
        Fence(2),
        "the re-home CAS did not bump (it lost)"
    );
    assert!(
        flows_to_node(&outbox, NodeId(9)).is_empty(),
        "no ReHome adopt is emitted to the target when the CAS is lost"
    );
}

#[test]
fn scan_deadlines_resolves_a_batch_handoff_with_a_dead_participant() {
    // D-7d: a DUE `BatchHandoff` saga whose SOURCE is known-dead self-promotes the dest
    // (SourceUnreachable → EmitTransientPromote + Done, counted); whose DEST is dead abandons the
    // source copy (DestUnreachable → EmitTransientAbandon + Done, counted); with NEITHER dead it just
    // RE-DRIVES (Timeout, still live, uncounted). The redrive deadline is 8 (`SagaTuning::default`).
    let mk = || {
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
        inject_saga(
            &mut runtime,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitPromote,
                new_fence: Fence(2),
            },
            UniverseTick(0),
        );
        runtime
    };
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });

    // SOURCE dead in a POST-ADOPT phase (AwaitPromote) → self-promote the dest (TransientDrop to
    // DEST), tombstone, count once. R-6d3c budget-gated the source-dead path too (so a source that
    // RESTARTS within budget delivers via its outbox replay FIRST): the first confirmed scan RE-DRIVES,
    // the resolution fires only past `abort_deadline_ticks` from the first observation.
    let mut runtime = mk();
    runtime.liveness.record_unreachable(SOURCE, UniverseTick(8));
    let mut outbox = OutboundBox::default();
    // First due scan: source CONFIRMED dead, but the restart-race budget has not elapsed → re-drive.
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8),
    );
    assert_eq!(
        runtime.live(),
        1,
        "the source self-promote waits the restart-race budget — not fired on the first observation"
    );
    assert_eq!(runtime.source_unreachable_resolutions(), 0);
    // After `abort_deadline_ticks` from the first observation: the self-promote fires.
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS),
    );
    assert_eq!(
        runtime.live(),
        0,
        "the dead-source resolution tombstoned the saga"
    );
    assert!(
        flows_to_node(&outbox, DEST).contains(&InterShardFlow::TransientDrop(TransientHandoff {
            transfer: XFER,
            step_id: TRANSIENT_DROP_STEP,
            fence: Fence(2),
        })),
        "self-promote to the dest: {:?}",
        outbox.0
    );
    assert_eq!(runtime.source_unreachable_resolutions(), 1);
    assert_eq!(runtime.dest_unreachable_resolutions(), 0);
    assert_eq!(
        runtime.batch_lost_source_crash(),
        0,
        "a POST-adopt source death is a zero-loss self-promote, NOT a counted loss"
    );

    // DEST dead → abandon the source copy — but the DESTRUCTIVE abandon (irreversible accounted loss)
    // is gated behind the LARGE abort budget from the FIRST confirmed-dead observation (the CSCALE-1
    // cure): a dest that recovers before the budget elapses must NOT be abandoned.
    let mut runtime = mk();
    runtime.liveness.record_unreachable(DEST, UniverseTick(8));
    let mut outbox = OutboundBox::default();
    // First due scan: dest CONFIRMED dead, but the abort budget has not elapsed → re-drive, not abandon.
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8),
    );
    assert_eq!(
        runtime.live(),
        1,
        "the abandon waits the abort budget — not fired on the first confirmed observation"
    );
    assert_eq!(runtime.dest_unreachable_resolutions(), 0);
    // After `abort_deadline_ticks` from the first observation: the abandon fires.
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS),
    );
    assert_eq!(runtime.live(), 0);
    assert!(
        flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::TransientAbandon(
            TransientHandoff {
                transfer: XFER,
                step_id: TRANSIENT_ABANDON_STEP,
                fence: Fence(2),
            }
        )),
        "abandon to the source: {:?}",
        outbox.0
    );
    assert_eq!(runtime.dest_unreachable_resolutions(), 1);
    assert_eq!(runtime.source_unreachable_resolutions(), 0);

    // NEITHER dead → a plain Timeout re-drive: still live, no resolution counted.
    let mut runtime = mk();
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8),
    );
    assert_eq!(runtime.live(), 1, "neither dead → re-drive, not resolve");
    assert_eq!(runtime.source_unreachable_resolutions(), 0);
    assert_eq!(runtime.dest_unreachable_resolutions(), 0);
}

#[test]
fn rehome_event_for_await_adopt_source_dead_emits_pre_adopt_past_budget_else_redrives() {
    // R-6d3c producer discrimination (HR5: all corners in the monomorphic helper). A dead SOURCE in
    // BatchHandoff is now BUDGET-gated (so a restart-within-budget wins its outbox-replay race), and
    // the resolution event is PHASE-discriminated: AwaitAdopt → SourceUnreachablePreAdopt (accounted
    // loss); a post-adopt phase → SourceUnreachable (zero-loss self-promote). abort_deadline=24.
    let tuning = SagaTuning::default();
    let c = ctx(DurabilityClass::Transient, Fence(1));
    let await_adopt = SagaState::BatchHandoff {
        phase: BatchHandoffPhase::AwaitAdopt,
        new_fence: Fence(2),
    };
    let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
    let roster: BTreeMap<NodeId, ShardProfile> = [(NodeId(9), empty)].into_iter().collect();
    let req = CapRequest::default();
    let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1

    // (i) source HEALTHY → Timeout, and a stale budget is cleared.
    let mut dos = Some((SOURCE, UniverseTick(5)));
    let ev = rehome_event_for(
        &await_adopt,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(30),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(ev, SagaEvent::Timeout);
    assert_eq!(dos, None, "a healthy source clears the stale budget");

    // Confirm SOURCE dead (n == 1 → one notice confirms).
    liveness.record_unreachable(SOURCE, UniverseTick(0));
    // (ii) source DEAD but WITHIN the restart-race budget → cheap Timeout re-drive (budget anchored).
    let mut dos = None;
    let ev = rehome_event_for(
        &await_adopt,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(0),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(ev, SagaEvent::Timeout);
    assert_eq!(
        dos,
        Some((SOURCE, UniverseTick(0))),
        "the restart-race budget anchors on the first dead observation (keyed by the dead node)"
    );
    let ev = rehome_event_for(
        &await_adopt,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(10),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(
        ev,
        SagaEvent::Timeout,
        "still within the 24-tick budget at tick 10"
    );

    // (iii) source DEAD, PAST budget, PRE-adopt (AwaitAdopt) → SourceUnreachablePreAdopt (accounted
    // loss, discard-to-dest — NEVER self-promote an empty dest).
    let ev = rehome_event_for(
        &await_adopt,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(24),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(ev, SagaEvent::SourceUnreachablePreAdopt);
    assert_eq!(dos, None, "the resolution clears the budget");

    // (iv) source DEAD, PAST budget, POST-adopt (AwaitPromote) → SourceUnreachable (NOT PreAdopt) —
    // closes the mis-pairing gap (a post-adopt phase self-promotes; it must never discard an adopted
    // batch).
    let post_adopt = SagaState::BatchHandoff {
        phase: BatchHandoffPhase::AwaitPromote,
        new_fence: Fence(2),
    };
    let mut dos = Some((SOURCE, UniverseTick(0)));
    let ev = rehome_event_for(
        &post_adopt,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(24),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(
        ev,
        SagaEvent::SourceUnreachable,
        "a post-adopt phase resolves as a zero-loss self-promote, NOT a PreAdopt discard"
    );

    // (v) CA-1 S3/S4 OVER-DISCARD SAFETY: source DEAD, PAST budget, AwaitAdopt, `dest_adopted = true`,
    // and the dest is ALIVE (not confirmed dead) → `defer_to_dest` skips the source resolution and,
    // since the dest is alive, falls to the neutral re-drive → Timeout, NOT the discard. The batch the
    // dest actually holds is never over-discarded; the same-tick drain advances the phase off the
    // latched `BatchAdopted` and the post-adopt self-promote then handles the dead source. Deferring to
    // a LIVE dest clears the (now-moot) source budget anchor.
    let mut dos = Some((SOURCE, UniverseTick(0)));
    let ev = rehome_event_for(
        &await_adopt,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(24),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        true,
    );
    assert_eq!(
        ev,
        SagaEvent::Timeout,
        "dest_adopted + live dest suppresses the pre-adopt discard (defer to the dest, re-drive)"
    );
    assert_eq!(
        dos, None,
        "deferring to a live dest clears the moot source budget anchor"
    );
}

#[test]
fn rehome_event_for_reanchors_the_budget_on_a_source_dest_cause_switch() {
    // R-6d3c budget-gate regression guard (the post-impl review's blocker): `dead_observed_since` is
    // keyed by the confirmed-dead NODE, so a CAUSE-SWITCH (the DEST is confirmed dead → the dest
    // RECOVERS → the SOURCE is confirmed dead) RE-ANCHORS the destructive-resolution budget to `now`
    // instead of measuring the source's restart-race grace from the DEST's stale first-dead tick. A
    // POST-adopt phase (AwaitPromote) is used because its orch->source egress makes
    // `is_confirmed_dead(source)` reachable-NOW — the pre-fix shared anchor would fire early.
    let tuning = SagaTuning::default(); // abort_deadline = 24
    let c = ctx(DurabilityClass::Transient, Fence(1));
    let state = SagaState::BatchHandoff {
        phase: BatchHandoffPhase::AwaitPromote,
        new_fence: Fence(2),
    };
    let empty = ShardProfile::build(CapRequest::default()).expect("empty profile");
    let roster: BTreeMap<NodeId, ShardProfile> = [(NodeId(9), empty)].into_iter().collect();
    let req = CapRequest::default();
    let mut liveness = LivenessTracker::new(LivenessTuning::default()); // n = 1, window = 64
    let mut dos = None;

    // DEST confirmed dead at tick 0 → the budget anchors on DEST, within budget → Timeout.
    liveness.record_unreachable(DEST, UniverseTick(0));
    let ev = rehome_event_for(
        &state,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(0),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(ev, SagaEvent::Timeout);
    assert_eq!(
        dos,
        Some((DEST, UniverseTick(0))),
        "anchored on the dead DEST"
    );

    // CAUSE-SWITCH at tick 30 (already PAST 24 from the DEST's tick-0 anchor): the DEST RECOVERS and
    // the SOURCE is confirmed dead. The shared-anchor BUG would fire SourceUnreachable now (30-0 >= 24);
    // the keyed anchor RE-ANCHORS to (SOURCE, 30) and returns Timeout — the source gets its OWN budget.
    liveness.record_ack(DEST);
    liveness.record_unreachable(SOURCE, UniverseTick(30));
    let ev = rehome_event_for(
        &state,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(30),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(
        ev,
        SagaEvent::Timeout,
        "the cause-switch re-anchors — the source is NOT resolved off the dest's stale budget"
    );
    assert_eq!(
        dos,
        Some((SOURCE, UniverseTick(30))),
        "re-anchored on the newly-dead SOURCE"
    );

    // The source's OWN budget elapses at tick 54 (30 + 24) → the post-adopt self-promote fires.
    let ev = rehome_event_for(
        &state,
        &c,
        &liveness,
        &mut dos,
        &tuning,
        UniverseTick(54),
        &roster,
        &req,
        DEST,
        Some(NodeId(9)),
        false,
    );
    assert_eq!(
        ev,
        SagaEvent::SourceUnreachable,
        "past the source's OWN re-anchored budget → the post-adopt self-promote fires"
    );
    assert_eq!(dos, None, "the resolution clears the budget");
}

#[test]
fn scan_deadlines_resolves_an_await_adopt_batch_as_accounted_loss_and_counts_it() {
    // R-6d3c end-to-end producer (M2 — the scan_deadlines counter arm the `_ => {}` wildcard would
    // silently drop): an AwaitAdopt BatchHandoff whose SOURCE is confirmed dead, PAST the restart-race
    // budget, resolves as an ACCOUNTED loss — it emits `TransientDiscard` to the DEST, tombstones, and
    // increments `batch_lost_source_crash` (NOT the zero-loss `source_unreachable_resolutions`). A late
    // `BatchAdopted` for the tombstoned saga is then absorbed as a no-op (M1 — no second resolution).
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8, abort=24
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    inject_saga(
        &mut runtime,
        SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitAdopt,
            new_fence: Fence(2),
        },
        UniverseTick(0),
    );
    runtime.liveness.record_unreachable(SOURCE, UniverseTick(8));

    // First due scan (tick 8): source confirmed dead but the restart-race budget has not elapsed → the
    // (AwaitAdopt, Timeout) re-drive, which CA-1 S3 makes emit the `ReSolicitBatch` liveness PROBE to the
    // SOURCE (the egress that makes `is_confirmed_dead(source)` reachable — proves the FSM arm + executor).
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8),
    );
    assert_eq!(
        runtime.live(),
        1,
        "the accounted-loss discard waits the restart-race budget"
    );
    assert_eq!(runtime.batch_lost_source_crash(), 0);
    assert!(
        flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::ReSolicitBatch(
            TransientHandoff {
                transfer: XFER,
                step_id: RE_SOLICIT_STEP,
                fence: Fence(2),
            }
        )),
        "AwaitAdopt Timeout emits the source liveness probe: {:?}",
        outbox.0
    );

    // Past `abort_deadline_ticks` from the first observation: the discard-to-dest + count fires.
    let mut outbox = OutboundBox::default();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS),
    );
    assert_eq!(
        runtime.live(),
        0,
        "the accounted-loss resolution tombstoned the saga"
    );
    assert!(
        flows_to_node(&outbox, DEST).contains(&InterShardFlow::TransientDiscard(
            TransientHandoff {
                transfer: XFER,
                step_id: TRANSIENT_DISCARD_STEP,
                fence: Fence(2),
            }
        )),
        "discard-to-dest (poison the late replay): {:?}",
        outbox.0
    );
    assert_eq!(
        runtime.batch_lost_source_crash(),
        1,
        "the never-restart PRE-adopt loss is counted (M2 — the scan_deadlines counter arm)"
    );
    assert_eq!(
        runtime.source_unreachable_resolutions(),
        0,
        "a PRE-adopt loss is NOT a zero-loss self-promote"
    );

    // M1: a LATE BatchAdopted for the now-tombstoned saga is a no-op — no second resolution, no egress.
    let mut outbox = OutboundBox::default();
    deliver(
        &mut runtime,
        &mut dir,
        &mut outbox,
        EpochId(1),
        UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS + 1),
        XFER,
        SagaEvent::BatchAdopted,
    );
    assert_eq!(runtime.live(), 0, "the tombstoned saga stays gone");
    assert_eq!(
        runtime.batch_lost_source_crash(),
        1,
        "no double-count on the late ack"
    );
    assert!(
        outbox.0.is_empty(),
        "no second egress on the late BatchAdopted"
    );
}

#[test]
fn latch_adopted_from_inbox_latches_only_a_present_sagas_batch_adopted() {
    // CA-1 S3/S4 — the pre-scan latch pass sets `dest_adopted` ONLY for a `BatchAdopted` (in a Saga-class
    // Wire) whose transfer names a LIVE saga. Covers every branch: a non-Saga inbound is skipped; a
    // Saga-class non-`BatchAdopted` arm is skipped; a `BatchAdopted` for an ABSENT saga is a no-op; a
    // `BatchAdopted` for the PRESENT XFER saga latches it.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    inject_saga(
        &mut runtime,
        SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitAdopt,
            new_fence: Fence(2),
        },
        UniverseTick(0),
    );
    let batch_adopted = |t| {
        flow_inbound(&InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: t,
            step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
        }))
    };
    let inbox = InboundBox(vec![
        // (1-false) a non-Saga-Wire inbound → skipped.
        Inbound::NodeUnreachable {
            to: SOURCE,
            class: MsgClass::Saga,
            undelivered: MsgId(0),
        },
        // (2-false) a Saga-class Wire that is NOT a BatchAdopted → skipped.
        demote_wire(Fence(2)),
        // (3-false) a BatchAdopted for an ABSENT saga → no-op (no such saga).
        batch_adopted(TransferId(999)),
        // (all-true) a BatchAdopted for the PRESENT XFER saga → latch it.
        batch_adopted(XFER),
    ]);
    latch_adopted_from_inbox(&mut runtime, &inbox);
    assert!(
        runtime.sagas.get(&XFER).expect("saga present").dest_adopted,
        "the present saga's dest_adopted latched from its BatchAdopted"
    );
}

#[test]
fn pre_scan_latch_suppresses_the_await_adopt_over_discard_on_the_maturity_tick() {
    // CA-1 S3/S4 the HARD over-discard guarantee, end-to-end in the intra-tick order `drive_sagas`
    // performs (latch pre-scan → scan). On the EXACT budget-maturity tick, a racing `BatchAdopted` in the
    // inbox is latched BEFORE `scan_deadlines`, so the destructive pre-adopt discard is SUPPRESSED (the
    // saga survives + keeps probing). The CONTROL (no adopt evidence) fires the discard on the SAME tick
    // — proving the latch is precisely what averts the over-discard, not a budget accident.
    let mk = || {
        let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive=8, abort=24
        inject_saga(
            &mut runtime,
            SagaState::BatchHandoff {
                phase: BatchHandoffPhase::AwaitAdopt,
                new_fence: Fence(2),
            },
            UniverseTick(0),
        );
        runtime.liveness.record_unreachable(SOURCE, UniverseTick(8)); // source confirmed dead (n = 1)
        runtime
    };
    let mk_dir = || {
        DirectoryCore::new(DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        })
    };
    // The budget anchors on the FIRST due scan and elapses `abort_deadline_ticks` later (dead_budget_
    // elapsed returns 0 on the anchoring call) — so, exactly like the accounted-loss test, an ANCHOR scan
    // at tick 8 precedes the MATURITY scan at 8 + abort_deadline where the discard is reachable.
    let anchor = UniverseTick(8);
    let maturity = UniverseTick(8 + saga::DEFAULT_ABORT_DEADLINE_TICKS);

    // --- WITH a racing BatchAdopted: the pre-scan latch suppresses the discard on the maturity tick ---
    let mut runtime = mk();
    let mut dir = mk_dir();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut OutboundBox::default(),
        EpochId(1),
        anchor,
    );
    let inbox = InboundBox(vec![flow_inbound(&InterShardFlow::TransferAck(
        TransferAck::BatchAdopted {
            transfer_id: XFER,
            step_id: vd_wire::intershard::TRANSIENT_BATCH_STEP,
        },
    ))]);
    latch_adopted_from_inbox(&mut runtime, &inbox); // the pre-scan pass drive_sagas runs FIRST
    let mut outbox = OutboundBox::default();
    scan_deadlines(&mut runtime, &mut dir, &mut outbox, EpochId(1), maturity);
    assert_eq!(
        runtime.live(),
        1,
        "the racing BatchAdopted latched dest_adopted → the discard is suppressed, the saga survives"
    );
    assert_eq!(runtime.batch_lost_source_crash(), 0, "NO over-discard");
    assert!(
        flows_to_node(&outbox, DEST).is_empty(),
        "no discard (indeed no egress at all) to the dest — the batch it holds is untouched: {:?}",
        outbox.0
    );
    assert!(
        flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::ReSolicitBatch(
            TransientHandoff {
                transfer: XFER,
                step_id: RE_SOLICIT_STEP,
                fence: Fence(2),
            }
        )),
        "the suppressed tick still re-drives the source probe: {:?}",
        outbox.0
    );

    // --- CONTROL: no adopt evidence in the inbox → the GENUINE pre-adopt discard fires the same tick ---
    let mut runtime = mk();
    let mut dir = mk_dir();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut OutboundBox::default(),
        EpochId(1),
        anchor,
    );
    latch_adopted_from_inbox(&mut runtime, &InboundBox::default()); // empty inbox → nothing latched
    let mut outbox = OutboundBox::default();
    scan_deadlines(&mut runtime, &mut dir, &mut outbox, EpochId(1), maturity);
    assert_eq!(
        runtime.live(),
        0,
        "no adopt evidence → the pre-adopt loss discard fires + tombstones"
    );
    assert_eq!(
        runtime.batch_lost_source_crash(),
        1,
        "the accounted loss is counted"
    );
    assert!(
        flows_to_node(&outbox, DEST).contains(&InterShardFlow::TransientDiscard(
            TransientHandoff {
                transfer: XFER,
                step_id: TRANSIENT_DISCARD_STEP,
                fence: Fence(2),
            }
        )),
        "discard-to-dest poisons a late replay: {:?}",
        outbox.0
    );

    // --- DOUBLE-CRASH (dest adopted THEN crashed + source also dead): resolve as the dest-dead abandon,
    // NOT a wedge. This is the regression the post-impl review caught: making the source probe live must
    // NOT let source-death preempt the dest-dead terminal into a perpetual Timeout loop. `defer_to_dest`
    // routes the adopted-then-dead dest to the budget-gated `DestUnreachable` abandon + tombstone.
    let mut runtime = mk();
    runtime.liveness.record_unreachable(DEST, UniverseTick(8)); // the dest is ALSO confirmed dead
    runtime
        .sagas
        .get_mut(&XFER)
        .expect("saga present")
        .dest_adopted = true; // it had adopted before dying
    let mut dir = mk_dir();
    scan_deadlines(
        &mut runtime,
        &mut dir,
        &mut OutboundBox::default(),
        EpochId(1),
        anchor,
    ); // anchors the DEST budget
    let mut outbox = OutboundBox::default();
    scan_deadlines(&mut runtime, &mut dir, &mut outbox, EpochId(1), maturity);
    assert_eq!(
        runtime.live(),
        0,
        "both dead → the dest-dead abandon terminates the saga (no wedge)"
    );
    assert_eq!(
        runtime.dest_unreachable_resolutions(),
        1,
        "counted as a dest-dead resolution (admin-visible), NOT a silent park"
    );
    assert_eq!(
        runtime.batch_lost_source_crash(),
        0,
        "an adopted-then-dead dest is NOT a source-crash discard"
    );
    assert!(
        flows_to_node(&outbox, SOURCE).contains(&InterShardFlow::TransientAbandon(
            TransientHandoff {
                transfer: XFER,
                step_id: TRANSIENT_ABANDON_STEP,
                fence: Fence(2),
            }
        )),
        "abandon-to-source (the promote target is gone): {:?}",
        outbox.0
    );
}

#[test]
fn liveness_tracker_confirms_only_after_n_consecutive_within_the_window() {
    // D-3 CSCALE-1: a peer is CONFIRMED dead only after `n_consecutive_unreachable` notices within the
    // window — a single blip never confirms. Covers: not-in-seen, not-enough, enough+fresh→confirmed,
    // stale (not-fresh), and clear-on-ack.
    let mut t = LivenessTracker::new(LivenessTuning {
        n_consecutive_unreachable: 3,
        unreachable_window_ticks: 10,
        retry_delay_ticks_hint: 2,
    });
    let n = NodeId(5);
    // Not in the tracker → not confirmed (the None arm).
    assert!(!t.is_confirmed_dead(n, UniverseTick(0)));
    // One notice → consecutive 1 < 3 → not enough.
    t.record_unreachable(n, UniverseTick(1));
    assert!(!t.is_confirmed_dead(n, UniverseTick(1)));
    // Two more within the window → consecutive 3 >= 3 + fresh → confirmed (the existing-entry increment).
    t.record_unreachable(n, UniverseTick(2));
    t.record_unreachable(n, UniverseTick(3));
    assert!(t.is_confirmed_dead(n, UniverseTick(3)));
    // STALE: far past the window from the run's first notice (tick 1) → not fresh → not confirmed.
    assert!(!t.is_confirmed_dead(n, UniverseTick(1 + 11)));
    // CLEAR-ON-ACK: a successful inbound un-marks the recovered peer.
    t.record_ack(n);
    assert!(!t.is_confirmed_dead(n, UniverseTick(3)));
    // record_ack is idempotent (clearing an absent node is a no-op).
    t.record_ack(n);
}

#[test]
fn liveness_tracker_resets_a_run_whose_window_elapsed() {
    // A notice arriving AFTER the window since the run's first notice STARTS a fresh run (consecutive
    // 1), not an extension — a long-ago blip is not evidence of a current death (the RESET arm).
    let mut t = LivenessTracker::new(LivenessTuning {
        n_consecutive_unreachable: 2,
        unreachable_window_ticks: 5,
        retry_delay_ticks_hint: 2,
    });
    let n = NodeId(7);
    t.record_unreachable(n, UniverseTick(1)); // run starts: consecutive 1, first = 1
    t.record_unreachable(n, UniverseTick(2)); // within window: consecutive 2
    assert!(t.is_confirmed_dead(n, UniverseTick(2)));
    // A notice at tick 10 (10 - 1 = 9 > window 5) RESETS the run to consecutive 1, first = 10.
    t.record_unreachable(n, UniverseTick(10));
    assert!(
        !t.is_confirmed_dead(n, UniverseTick(10)),
        "the reset run has only 1 notice (< 2) — a stale run is not a confirmation"
    );
}

#[test]
fn set_liveness_tuning_raises_the_confirmation_threshold() {
    // The test-config setter (used by the CSCALE-1 flap cell to run the prod margin n = 3 in-process):
    // re-tuning to n = 3 means a SINGLE NodeUnreachable no longer confirms a peer dead, where the
    // kill-equivalent default (n = 1) would. A fresh runtime has observed zero notices.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // n = 1
    assert_eq!(runtime.liveness_notices(), 0);
    runtime.set_liveness_tuning(LivenessTuning {
        n_consecutive_unreachable: 3,
        unreachable_window_ticks: 64,
        retry_delay_ticks_hint: 2,
    });
    runtime
        .liveness
        .record_unreachable(NodeId(9), UniverseTick(1));
    assert!(
        !runtime
            .liveness
            .is_confirmed_dead(NodeId(9), UniverseTick(1)),
        "after re-tuning to n = 3, one notice is below the confirmation threshold"
    );
}

#[test]
fn trigger_on_an_unrecorded_subject_refuses_to_start() {
    // Per-key serialization: no directory record → lock_transfer FALSE → no saga created.
    let mut rig = Rig::new();
    rig.trigger(ctx(DurabilityClass::Durable, Fence(1)));
    rig.settle();
    assert_eq!(rig.live(), 0, "an unrecorded subject cannot start a saga");
    assert!(rig.drain_gateway().is_empty());
}

#[test]
fn an_ack_for_an_unknown_saga_is_an_idempotent_noop() {
    // A duplicate/stale ack after GC (or for a never-started transfer) is a no-op.
    let mut rig = Rig::new();
    rig.ack(TransferControlAck::Committed {
        transfer: TransferId(999),
    });
    assert_eq!(rig.live(), 0);
    assert!(rig.drain_gateway().is_empty());
}

#[test]
fn ack_to_event_maps_every_ack_phase() {
    let cases: [(TransferControlAck, SagaEvent); 10] = [
        (
            TransferControlAck::Prepared {
                transfer: XFER,
                result: PrepareResult::Ready,
            },
            SagaEvent::Prepared(PrepareResult::Ready),
        ),
        (
            TransferControlAck::CutConfirmed {
                transfer: XFER,
                marker_seq: 1,
            },
            SagaEvent::CutConfirmed { marker_seq: 1 },
        ),
        (
            TransferControlAck::SourceFrozen {
                transfer: XFER,
                drained_seq: 2,
            },
            SagaEvent::SourceFrozen { drained_seq: 2 },
        ),
        (
            TransferControlAck::Committed { transfer: XFER },
            SagaEvent::RouteSwapped,
        ),
        (
            TransferControlAck::SourceThawed { transfer: XFER },
            SagaEvent::SourceThawed,
        ),
        (
            TransferControlAck::Aborted { transfer: XFER },
            SagaEvent::DestAborted,
        ),
        (
            TransferControlAck::Released { transfer: XFER },
            SagaEvent::Released,
        ),
        (
            TransferControlAck::DemoteAck { transfer: XFER },
            SagaEvent::DemoteAcked,
        ),
        (
            TransferControlAck::PromoteAck { transfer: XFER },
            SagaEvent::PromoteAcked,
        ),
        (
            TransferControlAck::DeliveredToObservers { transfer: XFER },
            SagaEvent::DestDelivered,
        ),
    ];
    for (ack, event) in cases {
        assert_eq!(ack_to_event(ack), event);
    }
}

#[test]
fn the_rejection_ledger_is_bounded_with_a_counted_drop() {
    // ROB-1: the un-drained ledger can NEVER grow without limit — beyond the cap the
    // oldest rejections are shed, counted (the loud overflow ALERT), oldest-first.
    let mut runtime = SagaRuntimeRes::default();
    for i in 0..(REJECTION_LEDGER_CAP as u128 + 5) {
        runtime
            .rejected
            .push((TransferId(i), AbortReason::Cancelled));
    }
    bound_rejection_ledger(&mut runtime);
    assert_eq!(runtime.rejected.len(), REJECTION_LEDGER_CAP);
    assert_eq!(runtime.rejections_dropped, 5);
    assert_eq!(
        runtime.rejected.first().expect("non-empty").0,
        TransferId(5),
        "the 5 OLDEST were shed; the newest survive"
    );
}

#[test]
fn emit_crossing_builds_for_an_entity_subject_and_skips_otherwise() {
    let c = ctx(DurabilityClass::Durable, Fence(1));

    // Entity subject + a flushed pose → the crossing is pushed to the DEST, stamped with the
    // given fence, STUB_CROSSING_STEP, and an empty (1d.1) state blob.
    let mut outbox = OutboundBox::default();
    emit_crossing(
        &c,
        Fence(2),
        Some(flushed_pose()),
        &[],
        EpochId(1),
        &mut outbox,
    );
    assert_eq!(
        outbox.0.len(),
        1,
        "an Entity subject with a pose emits a crossing"
    );
    let (to, class, bytes, _) = &outbox.0[0];
    assert_eq!(*to, DEST);
    assert_eq!(*class, MsgClass::Saga);
    assert_eq!(
        postcard::from_bytes::<InterShardFlow>(bytes).expect("decode"),
        InterShardFlow::Transfer(TransferEnvelope {
            transfer_id: XFER,
            universe_epoch: EpochId(1),
            schema_version: TRANSFER_SCHEMA_VERSION,
            fence: Fence(2),
            step_id: STUB_CROSSING_STEP,
            class: DurabilityClass::Durable,
            payload: TransitionPayload::StubCrossing {
                entity: subject_eid(),
                from_realm: FROM_REALM,
                to_realm: TO_REALM,
                pose: flushed_pose(),
                state: vec![],
            },
        })
    );

    // No stashed pose → no-op (the gate makes this unreachable for an Entity subject, but the
    // helper is total — this covers the missing-pose None arm).
    let mut no_pose = OutboundBox::default();
    emit_crossing(&c, Fence(2), None, &[], EpochId(1), &mut no_pose);
    assert!(no_pose.0.is_empty(), "no pose → no crossing");

    // Non-Entity subject (the Realm-subject FSM proptests) → no-op.
    let realm_ctx = SagaCtx {
        exterior: false,
        subject: DirectoryKey::Realm(RealmId::System(9)),
        ..c
    };
    let mut realm_out = OutboundBox::default();
    emit_crossing(
        &realm_ctx,
        Fence(2),
        Some(flushed_pose()),
        &[],
        EpochId(1),
        &mut realm_out,
    );
    assert!(
        realm_out.0.is_empty(),
        "a non-Entity subject emits no crossing"
    );
}

#[test]
fn a_flush_for_an_unknown_saga_is_a_noop() {
    // stash_flush's None arm: a SourceFlushed for an unknown / GC'd transfer mutates nothing.
    let mut runtime = SagaRuntimeRes::default();
    stash_flush(&mut runtime, TransferId(999), flushed_pose(), Vec::new());
    assert_eq!(runtime.live(), 0, "a stray flush creates/mutates no saga");
}

#[test]
fn a_dest_crossing_ack_and_a_non_saga_arm_are_dropped() {
    // drive_sagas decodes but DROPS the DEST's crossing ack (Accepted) and any non-saga-driving
    // arm (here a Saga command, which the orchestrator only ever SENDS): no saga starts,
    // nothing is emitted.
    let mut rig = Rig::new();
    for flow in [
        InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: XFER,
            step_id: STUB_CROSSING_STEP,
        }),
        InterShardFlow::Saga(TransferControl::RequestCut {
            transfer: XFER,
            session: SESSION,
        }),
    ] {
        rig.gateway
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(postcard::to_allocvec(&flow).expect("encode")),
            )
            .expect("sent");
    }
    rig.settle();
    assert_eq!(
        rig.live(),
        0,
        "neither a crossing ack nor a stray arm starts a saga"
    );
    assert!(
        rig.drain_gateway().is_empty(),
        "nothing is emitted in response"
    );
}

// ── Slice 3f-B — the durable `CrossingRequest` consumer ────────────────────────────────────────

#[test]
fn crossing_request_starts_a_tagged_saga() {
    // All THREE heads resolve (subject@F/SOURCE, Realm(to)@DEST, Session@GATEWAY) → a durable crossing
    // saga starts, keyed on the id the SOURCE latched (`crossing_transfer_id(subject, F)` over the WIRE
    // fence), with source/dest/session/gateway resolved from the directory.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1)); // Entity(subject())@F(1) owned by Shard(SOURCE)
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(5),
    );
    rig.grant_key(
        DirectoryKey::Session(CROSSING_SESSION),
        AuthorityRef::Gateway(GATEWAY),
        Fence(3),
    );
    rig.crossing_request(crossing_req(Fence(1))); // tick N: enqueue the start
    rig.settle(); // tick N+1: process_starts inserts the saga

    let latched = crossing_transfer_id(subject(), Fence(1), 0);
    let ctx = rig.saga_ctx(latched);
    assert_eq!(ctx.transfer, latched);
    assert_eq!(ctx.transfer, crossing_transfer_id(subject(), Fence(1), 0));
    assert_eq!(ctx.source, SOURCE);
    assert_eq!(ctx.dest, DEST);
    assert_eq!(ctx.session, CROSSING_SESSION);
    assert_eq!(ctx.expected_fence, Fence(1));
    assert_eq!(ctx.class, DurabilityClass::Durable);
    assert_eq!(ctx.from_realm, FROM_REALM);
    assert_eq!(ctx.to_realm, TO_REALM);
    // The parent-provenance the SOURCE detector supplied was threaded VERBATIM onto the saga ctx —
    // a wire-SHAPE pin: the field is ★DEAD (its consumer is deleted, D-PLACE-1/D-WIRE-1), but the
    // resolver must still carry it unchanged until the flag-day removal.
    assert_eq!(ctx.to_parent, Some(FROM_REALM));
    assert_eq!(rig.live(), 1);
    assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 1);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 0);
}

#[test]
fn crossing_request_unresolved_dest_counted() {
    // Subject + session resolve, but Realm(to) is ABSENT → no saga; counted `crossing_unresolved`.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.grant_key(
        DirectoryKey::Session(CROSSING_SESSION),
        AuthorityRef::Gateway(GATEWAY),
        Fence(3),
    );
    rig.crossing_request(crossing_req(Fence(1)));
    rig.settle();

    assert_eq!(rig.live(), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 1);
    assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 0);
}

#[test]
fn crossing_request_unresolved_session_counted() {
    // Subject + dest realm resolve, but Session is ABSENT → no saga; counted `crossing_unresolved`.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(5),
    );
    rig.crossing_request(crossing_req(Fence(1)));
    rig.settle();

    assert_eq!(rig.live(), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 1);
    assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 0);
}

#[test]
fn crossing_request_vanished_subject_counted() {
    // The subject has NO directory owner (authority already moved/revoked) → `crossing_subject_gone`,
    // even though the dest realm + session are present (the subject arm is checked first).
    let mut rig = Rig::new();
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(5),
    );
    rig.grant_key(
        DirectoryKey::Session(CROSSING_SESSION),
        AuthorityRef::Gateway(GATEWAY),
        Fence(3),
    );
    rig.crossing_request(crossing_req(Fence(1)));
    rig.settle();

    assert_eq!(rig.live(), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 1);
    assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 0);
}

#[test]
fn crossing_request_redelivery_is_one_saga() {
    // A redelivered request for the SAME fenced crossing resolves to ONE saga. Because the first request
    // has already SETTLED (its saga inserted by `process_starts`), the redelivery hits the handler's
    // `contains_key(&transfer)` guard arm → no second `start_transfer`, no second count: `crossings_started`
    // stays 1 and reflects DISTINCT started sagas, not requests seen.
    let mut rig = Rig::new();
    rig.grant_subject(Fence(1));
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(5),
    );
    rig.grant_key(
        DirectoryKey::Session(CROSSING_SESSION),
        AuthorityRef::Gateway(GATEWAY),
        Fence(3),
    );
    rig.crossing_request(crossing_req(Fence(1)));
    rig.settle(); // the saga starts + locks the subject
    rig.crossing_request(crossing_req(Fence(1))); // redelivery
    rig.settle();

    assert_eq!(rig.live(), 1, "the redelivery never spawns a second saga");
    let latched = crossing_transfer_id(subject(), Fence(1), 0);
    assert_eq!(rig.saga_ctx(latched).transfer, latched);
    assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 1);
}

// ── Slice 3f-C — the transient `TransientCrossingRequest` consumer ─────────────────────────────

/// A transient crossing request for `subject()` into TO_REALM at `src_realm_fence`. Carries a concrete
/// `to_parent` so the orchestrator's VERBATIM copy into the `TransientCrossingGrant` is asserted below.
fn transient_req(src_realm_fence: Fence) -> TransientCrossingRequest {
    TransientCrossingRequest {
        subject: subject(),
        from_realm: FROM_REALM,
        to_realm: TO_REALM,
        src_realm_fence,
        to_parent: Some(FROM_REALM),
    }
}

/// The wire form of the `TransientCrossingGrant` the source expects back from the orchestrator.
fn grant_wire(grant: TransientCrossingGrant) -> Inbound {
    Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes: postcard::to_allocvec(&InterShardFlow::TransientCrossingGrant(grant))
            .expect("encode")
            .into(),
    }
}

#[test]
fn transient_request_grants_resolved_dest() {
    // Realm(to) resolves to DEST@RF → ONE grant back to the transport-origin (SOURCE), carrying the
    // resolved dest, the current realm-lease fence, and the deterministic per-subject batch id.
    let mut rig = Rig::new();
    let realm_fence = Fence(4);
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        realm_fence,
    );
    // Discard the `LeaseGrant`'s directory-reply (seeding artifact) so the drain below is the grant only.
    let _ = rig.drain_source();
    rig.transient_crossing_request(transient_req(Fence(2)));

    let batch = crossing_transfer_id(subject(), Fence(2), 0);
    assert_eq!(
        rig.drain_source(),
        vec![grant_wire(TransientCrossingGrant {
            subject: subject(),
            dest: DEST,
            to_realm: TO_REALM,
            dst_realm_fence: realm_fence,
            batch,
            // The orchestrator copied the request's parent VERBATIM into the grant (asserted by this
            // full-value equality) — so the source can stamp it onto `TransientStatus::Crossing`.
            to_parent: Some(FROM_REALM),
        })]
    );
    assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 1);
    assert_eq!(rig.count(SagaRuntimeRes::transient_dest_unresolved), 0);
}

#[test]
fn transient_request_grant_starts_a_batchhandoff_awaitadopt_saga() {
    // D-43 #9 (THE fix): the resolved grant ALSO starts the transient `BatchHandoff` saga keyed on
    // `batch`, parked in `AwaitAdopt` awaiting the dest's `BatchAdopted` — so the downstream adopt
    // lands on a LIVE saga (not the silent None early-return). The go-token committed at the dest
    // realm-lease fence, and the grant emit is UNCHANGED (still counted once).
    let mut rig = Rig::new();
    let realm_fence = Fence(4);
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        realm_fence,
    );
    let _ = rig.drain_source(); // discard the LeaseGrant reply (seeding artifact)
    rig.transient_crossing_request(transient_req(Fence(2))); // tick N: grant + enqueue the start
    rig.settle(); // tick N+1: process_starts inserts the BatchHandoff saga → drives to AwaitAdopt

    let batch = crossing_transfer_id(subject(), Fence(2), 0);
    // EXACTLY ONE live saga, under `batch`.
    assert_eq!(rig.live(), 1, "the grant started exactly one saga");
    let ctx = rig.saga_ctx(batch);
    assert_eq!(
        rig.saga_state(batch),
        SagaState::BatchHandoff {
            phase: BatchHandoffPhase::AwaitAdopt,
            new_fence: realm_fence,
        },
        "parked in AwaitAdopt at the dest realm-lease fence"
    );
    assert_eq!(ctx.class, DurabilityClass::Transient);
    assert_eq!(ctx.source, SOURCE, "source == the transport-origin (from)");
    assert_eq!(ctx.dest, DEST, "dest == the resolved realm owner");
    assert_eq!(
        ctx.session,
        SessionId::NONE,
        "a session-less transient carries the typed sentinel, not a real id"
    );
    // The batched go-token committed at the realm-lease fence (G-TIER: one write per batch).
    assert_eq!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .batch_goes(),
        vec![(BatchId(batch), realm_fence)],
        "the go-token was written (was [] before the fix)"
    );
    // The grant leg is UNCHANGED: still exactly one grant counted.
    assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 1);
}

#[test]
fn transient_request_unknown_realm_drops() {
    // No Realm(to) record → no grant emitted; counted `transient_dest_unresolved`.
    let mut rig = Rig::new();
    rig.transient_crossing_request(transient_req(Fence(2)));

    assert!(
        rig.drain_source().is_empty(),
        "an unresolved dest realm emits no grant"
    );
    assert_eq!(rig.count(SagaRuntimeRes::transient_dest_unresolved), 1);
    assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 0);
}

#[test]
fn transient_request_redelivery_regrants_same_batch() {
    // A redelivered request re-grants the SAME deterministic batch id (idempotent at the source).
    let mut rig = Rig::new();
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(4),
    );
    // Discard the `LeaseGrant`'s directory-reply (seeding artifact) so both drains are the grant only.
    let _ = rig.drain_source();
    rig.transient_crossing_request(transient_req(Fence(2)));
    let first = rig.drain_source();
    rig.transient_crossing_request(transient_req(Fence(2)));
    let second = rig.drain_source();

    assert_eq!(
        first, second,
        "the redelivery re-grants the identical batch"
    );
    assert_eq!(rig.count(SagaRuntimeRes::transient_crossings_granted), 2);
    // D-43 #9 idempotency (the `if !contains_key` FALSE arm): the re-request did NOT start a second
    // saga — one batch, one `BatchHandoff` saga, absorbed by the guard.
    assert_eq!(rig.live(), 1, "one batch, one saga (idempotent re-grant)");
}

// ───────────────────────── Slice 3f-D: durable crossing-abort core (Mechanism Y) ─────────────────────

/// A crossing-origin `TransferId` (id-namespaced `0x39`) the source's latch keys on — the SAME value
/// `handle_crossing_request` mints from `(subject, subject_fence, attempt)`.
fn crossing_xfer() -> TransferId {
    crossing_transfer_id(subject(), Fence(1), 0)
}

/// Inject a live saga under a SPECIFIC transfer id (its `ctx.transfer` set to `transfer`, source
/// SOURCE / dest DEST / subject `subject()`) with the given class + state — for the `commit_result`
/// crossing-abort tests (the id's high byte is the `is_crossing_origin` discriminator).
fn inject_saga_id(
    runtime: &mut SagaRuntimeRes,
    transfer: TransferId,
    class: DurabilityClass,
    state: SagaState,
) {
    let mut ctx = ctx(class, Fence(1));
    ctx.transfer = transfer;
    runtime.sagas.insert(
        transfer,
        LiveSaga {
            ctx,
            state,
            gateway: GATEWAY,
            since: UniverseTick(0),
            flushed_pose: None,
            flushed_state: Vec::new(),
            dead_observed_since: None,
            dest_adopted: false,
            opened: UniverseTick(0),
        },
    );
}

/// The terminal `Aborted` state (a pre-CAS failure) a tombstoning saga carries.
fn aborted_state() -> SagaState {
    SagaState::Aborted {
        reason: AbortReason::CutTimeout,
    }
}

/// Count the `pending_writes` entries staging THIS abort-reply key with the given put/delete shape
/// (`Some` = PUT, `None` = DELETE). Value-equality on the key bytes — no key-parse.
fn abort_reply_writes(runtime: &SagaRuntimeRes, transfer: TransferId, put: bool) -> usize {
    let key = StoreKey::AbortReply(transfer).bytes();
    runtime
        .pending_writes
        .iter()
        .filter(|(k, v)| *k == key && v.is_some() == put)
        .count()
}

#[test]
fn store_key_abort_reply_tag_is_five_and_distinct() {
    // D0: the AbortReply family tag is 5 and disjoint from every prior family (1..=4).
    let bytes = StoreKey::AbortReply(crossing_xfer()).bytes();
    assert_eq!(bytes[0], 5, "AbortReply tag is 5");
    let tags = [
        StoreKey::Saga(XFER).bytes()[0],
        StoreKey::BatchGo(BatchId(XFER)).bytes()[0],
        StoreKey::Directory(subject()).bytes()[0],
        StoreKey::Clock.bytes()[0],
        StoreKey::AbortReply(XFER).bytes()[0],
    ];
    let mut distinct = tags.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        tags.len(),
        "all five family tags are distinct"
    );
}

#[test]
fn pending_abort_reply_roundtrips() {
    // D0: the self-describing record encodes + decodes byte-identically (all three fields).
    let reply = PendingAbortReply {
        transfer: crossing_xfer(),
        source: SOURCE,
        subject: subject(),
    };
    let bytes = encode(&reply);
    let back: PendingAbortReply = postcard::from_bytes(&bytes).expect("decode PendingAbortReply");
    assert_eq!(back, reply);
}

#[test]
fn is_crossing_origin_is_true_only_for_the_crossing_namespace() {
    // D1: the stateless tag-check — `0x39` crossing is EXCLUSIVE of re-home (`0x37`) and the
    // connection-plane's small high-byte-`0x00` ids.
    let mut c = ctx(DurabilityClass::Durable, Fence(1));
    c.transfer = crossing_xfer();
    assert!(is_crossing_origin(&c), "a 0x39 id IS crossing-origin");
    c.transfer = namespaced_transfer_id(0x37, b"rehome"); // re-home namespace
    assert!(
        !is_crossing_origin(&c),
        "a 0x37 re-home id is NOT crossing-origin"
    );
    c.transfer = TransferId(1); // a small connection-plane id (high byte 0x00)
    assert!(
        !is_crossing_origin(&c),
        "a high-byte-0x00 id is NOT crossing-origin"
    );
}

#[test]
fn commit_result_aborted_crossing_inserts_pending_reply_and_stages_both_writes() {
    // D1: a crossing-origin (0x39) DURABLE saga that tombstones ABORTED inserts a PendingAbortReply,
    // stages its PUT, AND stages the UNCONDITIONAL saga-snapshot DELETE — both in the SAME tick's
    // `pending_writes` (co-tick atomicity), keyed on DISTINCT families.
    let mut runtime = SagaRuntimeRes::default();
    let transfer = crossing_xfer();
    inject_saga_id(
        &mut runtime,
        transfer,
        DurabilityClass::Durable,
        aborted_state(),
    );

    commit_result(
        &mut runtime,
        transfer,
        aborted_state(),
        true,
        vec![],
        vec![],
        UniverseTick(0),
    );

    let entry = runtime
        .pending_abort_replies
        .get(&transfer)
        .expect("a pending abort reply was inserted");
    assert_eq!(entry.source, SOURCE);
    assert_eq!(entry.subject, subject());
    assert_eq!(entry.transfer, transfer);
    assert_eq!(
        abort_reply_writes(&runtime, transfer, true),
        1,
        "the AbortReply PUT was staged"
    );
    // The co-tick UNCONDITIONAL saga-snapshot DELETE (a distinct key family) is also staged.
    let saga_delete = StoreKey::Saga(transfer).bytes();
    assert_eq!(
        runtime
            .pending_writes
            .iter()
            .filter(|(k, v)| *k == saga_delete && v.is_none())
            .count(),
        1,
        "the saga-snapshot DELETE rides the same tick"
    );
    assert!(
        !runtime.sagas.contains_key(&transfer),
        "the saga was tombstoned"
    );
}

#[test]
fn commit_result_done_crossing_does_not_insert() {
    // D1: a crossing-origin saga that tombstones DONE (not Aborted) inserts NOTHING — the `aborted`
    // operand is false (the &&-split's second-false case).
    let mut runtime = SagaRuntimeRes::default();
    let transfer = crossing_xfer();
    inject_saga_id(
        &mut runtime,
        transfer,
        DurabilityClass::Durable,
        SagaState::Done {
            new_fence: Fence(2),
        },
    );
    commit_result(
        &mut runtime,
        transfer,
        SagaState::Done {
            new_fence: Fence(2),
        },
        true,
        vec![],
        vec![],
        UniverseTick(0),
    );
    assert!(
        runtime.pending_abort_replies.is_empty(),
        "Done does not owe a reply"
    );
    assert_eq!(abort_reply_writes(&runtime, transfer, true), 0);
}

#[test]
fn commit_result_aborted_rehome_does_not_insert() {
    // D1: a NON-crossing (re-home 0x37) saga that tombstones Aborted inserts NOTHING — the
    // `is_crossing_origin` operand is false (the discriminator-exclusivity / first-false case).
    let mut runtime = SagaRuntimeRes::default();
    let transfer = namespaced_transfer_id(0x37, b"rehome");
    inject_saga_id(
        &mut runtime,
        transfer,
        DurabilityClass::Durable,
        aborted_state(),
    );
    commit_result(
        &mut runtime,
        transfer,
        aborted_state(),
        true,
        vec![],
        vec![],
        UniverseTick(0),
    );
    assert!(
        runtime.pending_abort_replies.is_empty(),
        "a re-home abort owes no crossing reply"
    );
}

#[test]
fn commit_result_aborted_transient_crossing_does_not_insert() {
    // D1: a crossing-namespaced but TRANSIENT saga aborting inserts NOTHING — the `durable` operand is
    // false (the class-gate false arm; a transient carries no per-entity durable latch).
    let mut runtime = SagaRuntimeRes::default();
    let transfer = crossing_xfer();
    inject_saga_id(
        &mut runtime,
        transfer,
        DurabilityClass::Transient,
        aborted_state(),
    );
    commit_result(
        &mut runtime,
        transfer,
        aborted_state(),
        true,
        vec![],
        vec![],
        UniverseTick(0),
    );
    assert!(
        runtime.pending_abort_replies.is_empty(),
        "a transient abort owes no durable reply"
    );
}

/// Scan the abort-reply pending set + re-emits with a fresh outbox, at `now`, using the runtime's
/// tuning (redrive default = 8). Returns the emitted `CrossingAborted` flows to `source`.
/// Run the abort-reply pass with the subject's directory head owned by `owner` (`None` = no record →
/// the subject "left the source", which the head-check reaps). 3f-D: the reap keys on ownership, so the
/// re-emit tests seed `Some(SOURCE)` (still owned → re-emit) and the reap tests seed `None`/`Some(other)`.
fn scan_abort(
    runtime: &mut SagaRuntimeRes,
    now: UniverseTick,
    owner: Option<NodeId>,
) -> Vec<InterShardFlow> {
    let mut dir = DirectoryCore::new(DirectoryTuning {
        lease_ttl_ticks: 10_000,
        ..DirectoryTuning::default()
    });
    if let Some(node) = owner {
        dir.grant(subject(), AuthorityRef::Shard(node), Fence(1), now);
    }
    let mut outbox = OutboundBox::default();
    scan_deadlines(runtime, &mut dir, &mut outbox, EpochId(1), now);
    flows_to_node(&outbox, SOURCE)
}

/// Seed one live pending abort reply (no live saga — the saga already tombstoned; this is the
/// standalone obligation) with `last_abort_reply_emit` armed to 0 (never emitted).
fn seed_pending_reply(runtime: &mut SagaRuntimeRes, transfer: TransferId) {
    runtime.pending_abort_replies.insert(
        transfer,
        PendingAbortReply {
            transfer,
            source: SOURCE,
            subject: subject(),
        },
    );
}

#[test]
fn scan_deadlines_reemits_a_pending_abort_reply_once_per_cadence() {
    // D2: a live-source pending reply re-emits exactly ONE CrossingAborted to the source per
    // redrive-cadence window (8), and the entry stays (awaiting the ack).
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive = 8
    let transfer = crossing_xfer();
    seed_pending_reply(&mut runtime, transfer);

    // First scan at now=8: last_emit=0, 8-0 >= 8 → emit once (source still owns the subject).
    let emitted = scan_abort(&mut runtime, UniverseTick(8), Some(SOURCE));
    assert_eq!(
        emitted,
        vec![InterShardFlow::CrossingAborted(
            vd_wire::intershard::CrossingAborted {
                subject: subject(),
                transfer,
            }
        )],
        "exactly one CrossingAborted re-emitted to the source"
    );
    assert!(
        runtime.pending_abort_replies.contains_key(&transfer),
        "the entry persists until acked"
    );
    assert_eq!(
        runtime.last_abort_reply_emit,
        UniverseTick(8),
        "the cadence gate advanced"
    );
}

#[test]
fn scan_deadlines_throttles_the_abort_reemit_within_the_cadence_window() {
    // D2: within the same redrive window the re-emit is THROTTLED (the not-yet-due arm) — no second
    // egress against an alive-but-ack-stalled source.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive = 8
    let transfer = crossing_xfer();
    seed_pending_reply(&mut runtime, transfer);
    runtime.last_abort_reply_emit = UniverseTick(6); // last emitted at tick 6

    // now=10: 10-6 = 4 < 8 → NOT due, no emit (source still owns → throttled, not reaped).
    let emitted = scan_abort(&mut runtime, UniverseTick(10), Some(SOURCE));
    assert!(
        emitted.is_empty(),
        "within the window: no re-emit (throttled)"
    );
    assert_eq!(
        runtime.last_abort_reply_emit,
        UniverseTick(6),
        "the gate did not advance"
    );
}

#[test]
fn scan_deadlines_reaps_a_pending_reply_whose_subject_left_the_source() {
    // D2: when the subject NO LONGER belongs to the source (head absent — D-37 re-homed it, or it
    // transferred away), the abort is MOOT → REAPED: the entry is removed, a DELETE staged, ZERO
    // CrossingAborted emitted (the reap arm). `None` owner = no directory record for the subject.
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default());
    let transfer = crossing_xfer();
    seed_pending_reply(&mut runtime, transfer);

    let emitted = scan_abort(&mut runtime, UniverseTick(8), None);
    assert!(
        emitted.is_empty(),
        "a subject that left the source draws no re-emit"
    );
    assert!(
        !runtime.pending_abort_replies.contains_key(&transfer),
        "the moot reply was reaped"
    );
    assert_eq!(
        abort_reply_writes(&runtime, transfer, false),
        1,
        "the reap staged a persist-DELETE"
    );
}

#[test]
fn scan_deadlines_reemits_while_owned_then_reaps_when_the_subject_leaves() {
    // D2: two cadence-spaced scans re-emit twice while the source still owns the subject; then the
    // subject leaves the source (re-homed away → head no longer resolves to SOURCE) and the third scan
    // reaps. Proves the re-emit obligation persists on a LIVE-or-recovering owner and only drains once
    // the entity has genuinely moved (the adversary-review-HIGH fix: not the `is_confirmed_dead` pulse).
    let mut runtime = SagaRuntimeRes::with_tuning(SagaTuning::default()); // redrive = 8
    let transfer = crossing_xfer();
    seed_pending_reply(&mut runtime, transfer);

    assert_eq!(
        scan_abort(&mut runtime, UniverseTick(8), Some(SOURCE)).len(),
        1,
        "first re-emit"
    );
    assert_eq!(
        scan_abort(&mut runtime, UniverseTick(16), Some(SOURCE)).len(),
        1,
        "second re-emit"
    );
    // The subject re-homed away from SOURCE (owned by DEST now) → the abort is moot → reaped.
    assert!(
        scan_abort(&mut runtime, UniverseTick(24), Some(DEST)).is_empty(),
        "no re-emit once the subject left the source"
    );
    assert!(
        !runtime.pending_abort_replies.contains_key(&transfer),
        "the third scan reaped the moot reply"
    );
}

#[test]
fn crossing_aborted_ack_drops_the_pending_entry() {
    // D3: the source's `CrossingAbortedAck` drops the pending entry — driven through the full
    // `drive_sagas` inbound path (the new ack arm). Grant the subject's head to SOURCE first, so the
    // ownership-reap does NOT preempt the ack (it would otherwise drain the moot entry before the ack
    // arrives). Then seed the entry, have SOURCE send the ack, and settle.
    let mut rig = Rig::new();
    let transfer = crossing_xfer();
    rig.grant_key(subject(), AuthorityRef::Shard(SOURCE), Fence(1)); // source still owns → no reap
    {
        let mut runtime = rig.orch.world_mut().resource_mut::<SagaRuntimeRes>();
        runtime.pending_abort_replies.insert(
            transfer,
            PendingAbortReply {
                transfer,
                source: SOURCE,
                subject: subject(),
            },
        );
    }
    let ack = |rig: &mut Rig| {
        rig.source
            .send(
                ORCH,
                MsgClass::Saga,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::CrossingAbortedAck(
                        vd_wire::intershard::CrossingAborted {
                            subject: subject(),
                            transfer,
                        },
                    ))
                    .expect("encode"),
                ),
            )
            .expect("ack sent");
        rig.settle();
    };
    ack(&mut rig); // present → `is_some()`-true arm: drops + stages the DELETE
    assert!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .pending_abort_replies
            .is_empty(),
        "the ack dropped the pending entry"
    );
    ack(&mut rig); // redelivery → `is_some()`-false arm: idempotent no-op (no panic, no double-DELETE)
    assert!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .pending_abort_replies
            .is_empty(),
        "a redelivered ack stays a no-op",
    );
}

#[test]
fn crossing_aborted_ack_is_idempotent_on_redelivery() {
    // D3: a REDELIVERED ack (whose entry is already gone) is a `is_some()`-false no-op — no panic, no
    // double-DELETE stage. Unit-drive the arm twice directly on a runtime.
    let mut runtime = SagaRuntimeRes::default();
    let transfer = crossing_xfer();
    seed_pending_reply(&mut runtime, transfer);

    // First remove: present.
    assert!(runtime.pending_abort_replies.remove(&transfer).is_some());
    runtime
        .pending_writes
        .push((StoreKey::AbortReply(transfer).bytes(), None));
    // Second (redelivery): absent → the guard's false arm is a no-op.
    assert!(
        runtime.pending_abort_replies.remove(&transfer).is_none(),
        "a redelivered ack finds nothing"
    );
    assert_eq!(
        abort_reply_writes(&runtime, transfer, false),
        1,
        "only ONE DELETE staged (the redelivery did not double-stage)"
    );
}

#[test]
fn rehydrate_restores_a_persisted_abort_reply_and_reemits_on_first_scan() {
    // D0 + crash-leg: an AbortReply record persisted before a kill-9 is restored by rehydrate, and the
    // FIRST post-restart scan re-emits CrossingAborted (a RAM-only map would have stranded the source).
    let transfer = crossing_xfer();
    let mut store = MemStore::new();
    // The Clock record is the genesis-vs-recover discriminator — seed it so rehydrate recovers.
    store.put(
        &StoreKey::Clock.bytes(),
        &encode(&(EpochId(1), UniverseTick(0))),
    );
    // The persisted abort-reply obligation (what commit_result staged pre-crash).
    let reply = PendingAbortReply {
        transfer,
        source: SOURCE,
        subject: subject(),
    };
    store.put(&StoreKey::AbortReply(transfer).bytes(), &encode(&reply));
    // COMMIT the staged writes (the group-commit barrier the orchestrator would have run pre-crash) —
    // `scan` reads only the committed set.
    store.commit();

    let recovered = rehydrate(
        &store,
        1024,
        SagaTuning::default(),
        LivenessTuning::default(),
        DirectoryTuning {
            lease_ttl_ticks: 10_000,
            ..DirectoryTuning::default()
        },
        vd_sim::rlm::RlmTuning::default(),
    )
    .expect("the store is non-empty (a Clock record present) → recover");
    let mut runtime = recovered.runtime;
    assert_eq!(
        runtime.pending_abort_replies.get(&transfer),
        Some(&reply),
        "rehydrate restored the persisted abort reply"
    );
    // The first post-restart scan (last_abort_reply_emit == 0) re-emits immediately (source still owns).
    let emitted = scan_abort(&mut runtime, UniverseTick(8), Some(SOURCE));
    assert_eq!(
        emitted,
        vec![InterShardFlow::CrossingAborted(
            vd_wire::intershard::CrossingAborted {
                subject: subject(),
                transfer,
            }
        )],
        "the restored obligation re-emits on the first post-restart scan (crash-leg proof)"
    );
}

// ===================== the ruler switch, slice 1: the exterior crossing saga =====================

fn hull_key() -> DirectoryKey {
    DirectoryKey::Ship(EntityId::pack(EntityKind::Ship, 1, 1, 0))
}

fn exterior_req(fence: Fence) -> vd_wire::intershard::ExteriorCrossingRequest {
    vd_wire::intershard::ExteriorCrossingRequest {
        subject: hull_key(),
        from_realm: FROM_REALM,
        to_realm: TO_REALM,
        subject_fence: fence,
        attempt: 0,
    }
}

fn decoded(inbound: &[Inbound]) -> Vec<InterShardFlow> {
    inbound
        .iter()
        .filter_map(|i| match i {
            Inbound::Wire { bytes, .. } => postcard::from_bytes(bytes).ok(),
            _ => None,
        })
        .collect()
}

#[test]
fn an_exterior_request_starts_a_session_less_saga_that_walks_to_done_without_a_gateway() {
    let mut rig = Rig::new();
    // The hull's exterior is held by the SOURCE parent; the destination realm runs on DEST.
    rig.grant_key(hull_key(), AuthorityRef::Shard(SOURCE), Fence(1));
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(5),
    );
    let _ = (rig.drain_source(), rig.drain_dest(), rig.drain_gateway());
    rig.exterior_request(exterior_req(Fence(1)), false);
    rig.settle();
    let latched = crossing_transfer_id(hull_key(), Fence(1), 0);
    let ctx = rig.saga_ctx(latched);
    assert!(ctx.exterior);
    assert_eq!(ctx.session, SessionId::NONE);
    assert_eq!(ctx.source, SOURCE);
    assert_eq!(ctx.dest, DEST);
    assert_eq!(ctx.expected_fence, Fence(1));
    assert_eq!(ctx.from_realm, FROM_REALM);
    assert_eq!(ctx.to_realm, TO_REALM);
    assert_eq!(rig.live(), 1);
    assert_eq!(rig.count(SagaRuntimeRes::exterior_crossings_started), 1);
    assert_eq!(rig.count(SagaRuntimeRes::crossings_started), 0);
    // The walk opens at the flush: the source is asked for the exterior, the gateway for nothing.
    assert!(
        decoded(&rig.drain_source()).contains(&InterShardFlow::FlushSource(
            vd_wire::intershard::FlushSource {
                transfer: latched,
                subject: hull_key(),
                step_id: FLUSH_SOURCE_STEP,
                to_realm: TO_REALM,
                to_parent: None,
            }
        )),
        "the source is asked to flush the exterior"
    );
    assert!(rig.drain_gateway().is_empty(), "no gateway takes part");
    // The source ships the pose: the CAS moves the exterior to DEST at the next fence.
    rig.source
        .send(
            ORCH,
            MsgClass::Saga,
            vd_sim::io::bytes(
                postcard::to_allocvec(&InterShardFlow::TransferAck(TransferAck::SourceFlushed {
                    transfer_id: latched,
                    step_id: FLUSH_SOURCE_STEP,
                    pose: flushed_pose(),
                    drained_seq: 0,
                    state: vec![],
                }))
                .expect("encode"),
            ),
        )
        .expect("sent");
    rig.settle();
    let head = rig
        .orch
        .world_mut()
        .resource::<DirectoryRes>()
        .0
        .head(hull_key())
        .expect("the exterior is recorded");
    assert_eq!(head.authority, AuthorityRef::Shard(DEST));
    let new_fence = head.fence;
    assert!(new_fence > Fence(1));
    // The envelope reaches the destination, naming the hull by its entity; the demote reaches the source.
    let to_dest = decoded(&rig.drain_dest());
    let envelope = to_dest.iter().find_map(|f| match f {
        InterShardFlow::Transfer(env) => Some(env.clone()),
        _ => None,
    });
    let envelope = envelope.expect("the destination receives the exterior's envelope");
    assert_eq!(envelope.fence, new_fence);
    match envelope.payload {
        vd_wire::intershard::TransitionPayload::StubCrossing {
            entity,
            from_realm,
            to_realm,
            ..
        } => {
            assert_eq!(DirectoryKey::Ship(entity), hull_key());
            assert_eq!(from_realm, FROM_REALM);
            assert_eq!(to_realm, TO_REALM);
        }
        other => panic!("not a crossing payload: {other:?}"),
    }
    assert!(
        decoded(&rig.drain_source()).contains(&InterShardFlow::Demote(
            vd_wire::intershard::DemoteCmd {
                transfer: latched,
                subject: hull_key(),
                new_owner_fence: new_fence,
                step_id: vd_wire::intershard::DEMOTE_STEP,
            }
        )),
        "the source is demoted at the new fence"
    );
    assert!(rig.drain_gateway().is_empty(), "still no gateway");
    // The demote ack brings the promote; the promote ack ends the walk with no release.
    rig.ack(TransferControlAck::DemoteAck { transfer: latched });
    assert!(
        decoded(&rig.drain_dest()).contains(&InterShardFlow::Promote(
            vd_wire::intershard::PromoteCmd {
                transfer: latched,
                subject: hull_key(),
                new_fence,
                step_id: vd_wire::intershard::PROMOTE_STEP,
                source: SOURCE,
            }
        )),
        "the destination is promoted"
    );
    rig.ack(TransferControlAck::PromoteAck { transfer: latched });
    assert_eq!(rig.live(), 0, "done at the promote ack");
    assert!(rig.drain_gateway().is_empty(), "no release at a gateway");
    // ★ THE REPARENT NOTE IS TAKEN FROM THE IN-PROCESS COMMIT — the shipped path. MEASURED on the
    // fifth flight (2026-09-03): the note used to read only the caller's own event, the hull's
    // hand-down committed inside the quiescence run, no note was taken, the reconciler never
    // re-keyed the hull and the gateway was never told.
    let hull_realm = vd_core::pose::RealmId::Ship(match hull_key() {
        DirectoryKey::Ship(entity) => entity,
        other => panic!("the hull's exterior key is a Ship key: {other:?}"),
    });
    // The reconciler drains the note on the same schedule step it is taken, so the note itself is
    // gone by now; what remains is the reconciler's own account of it — this rig has no demand cell
    // for the hull, so the move is counted unresolved, once.
    assert!(
        rig.orch
            .world_mut()
            .resource::<SagaRuntimeRes>()
            .pending_reparents
            .is_empty(),
        "drained by the reconciler"
    );
    let rlm = rig.orch.world_mut();
    let rlm = rlm.resource::<crate::rlm_runtime::RlmReconcilerRes>();
    assert_eq!(
        (rlm.reparents_applied, rlm.reparents_unresolved),
        (0, 1),
        "the reconciler was told the hull ({hull_realm:?}) moved house to {TO_REALM:?}"
    );
}

#[test]
fn an_exterior_request_from_a_node_that_does_not_hold_the_exterior_is_dropped() {
    let mut rig = Rig::new();
    rig.grant_key(hull_key(), AuthorityRef::Shard(SOURCE), Fence(1));
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(5),
    );
    // Sent by DEST, which does not hold the exterior.
    rig.exterior_request(exterior_req(Fence(1)), true);
    rig.settle();
    assert_eq!(rig.live(), 0);
    assert_eq!(rig.count(SagaRuntimeRes::exterior_request_unattested), 1);
    assert_eq!(rig.count(SagaRuntimeRes::exterior_crossings_started), 0);
}

#[test]
fn an_exterior_request_with_no_destination_head_or_no_subject_is_counted() {
    let mut rig = Rig::new();
    rig.grant_key(hull_key(), AuthorityRef::Shard(SOURCE), Fence(1));
    rig.exterior_request(exterior_req(Fence(1)), false);
    rig.settle();
    assert_eq!(rig.live(), 0);
    assert_eq!(rig.count(SagaRuntimeRes::crossing_unresolved), 1);
    // A subject nobody recorded.
    let mut rig = Rig::new();
    rig.exterior_request(exterior_req(Fence(1)), false);
    rig.settle();
    assert_eq!(rig.count(SagaRuntimeRes::crossing_subject_gone), 1);
    // A redelivered request for a live saga is one saga.
    let mut rig = Rig::new();
    rig.grant_key(hull_key(), AuthorityRef::Shard(SOURCE), Fence(1));
    rig.grant_key(
        DirectoryKey::Realm(TO_REALM),
        AuthorityRef::Shard(DEST),
        Fence(5),
    );
    rig.exterior_request(exterior_req(Fence(1)), false);
    rig.settle();
    rig.exterior_request(exterior_req(Fence(1)), false);
    rig.settle();
    assert_eq!(rig.live(), 1);
    assert_eq!(rig.count(SagaRuntimeRes::exterior_crossings_started), 1);
}
