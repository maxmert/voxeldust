//! The Transfer Saga RUNTIME (P2 Slice 1b) — the orchestrator-side wrapper around the
//! pure [`vd_sim::saga`] FSM. It is the ONE owner of a transfer's lifecycle at runtime:
//!
//! - **Per-key serialization**: at most one saga per `DirectoryKey`, enforced by
//!   `DirectoryCore::lock_transfer` (the directory's `in_transfer` field) — a duplicate
//!   trigger is refused, not double-started.
//! - **Drives the gateway**: every `SagaAction::Send(TransferControl)` is serialized onto
//!   the `InterShardFlow::Saga` arm to the session's gateway (the gateway never decides a
//!   transfer; HR1 typed seam).
//! - **The single commit point, DIRECT**: `IssueCommitCas` calls `DirectoryCore::commit_cas`
//!   IN PROCESS — the orchestrator owns `DirectoryRes` on this same single-threaded schedule,
//!   so there is no self-addressed wire round-trip (no tick split, no `ResMut` race). The CAS
//!   outcome (`Won`/`Lost`) is fed straight back into the FSM as the next event.
//! - **Class-aware from line one (HR2)**: a `Transient` subject reaches the same commit point
//!   but issues the batched `TransientGo` go-token (a no-op stub until P3), never a per-entity
//!   CAS — the FSM's `commit_action` fan-out, executed here without a shard-kind branch.
//!
//! The sim thread NEVER awaits: all egress is enqueue-only (`OutboundBox`). At-least-once
//! delivery + adaptive timeouts + the CAS re-read loop land at Slice 2; the demote signal
//! (`DemoteComplete`) and the client-facing cut cycle land at Slice 1c/1e. The shard-bound
//! `StubCrossing` state transfer is synthesized at Slice 1d (coupled with the stub decode).
//!
//! ## KNOWN 1b LIMIT (binding Slice-2 work item — pinned by tests below)
//! Every ABORT path currently leaves the subject's directory lock SET (`in_transfer =
//! Some(transfer)` survives the terminal `Aborted`): the FSM emits no abort-CAS action and
//! this runtime never calls `DirectoryCore::abort_cas`, so an aborted subject cannot start
//! another transfer until Slice 2 lands "terminal abort clears the directory lock via
//! `abort_cas`, including the stale-fence re-read" (a CAS-loser's `expected_fence` is stale
//! by definition, so the clear needs the Slice-2 head re-read — it cannot be a naive
//! `abort_cas(expected)` here). The abort tests ASSERT the lock is still set, so the Slice-2
//! fix must deliberately flip those assertions — the leak can never be forgotten silently.
//! (Caught by the Slice-1b audit, CPO-1/CPO-2.)

use std::collections::{BTreeMap, VecDeque};

use bevy_ecs::prelude::{Res, ResMut, Resource};
use vd_core::{NodeId, TransferId, UniverseTick};
use vd_sim::directory::DirectoryCore;
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_sim::saga::{self, AbortReason, SagaAction, SagaCtx, SagaEvent, SagaState};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, CasOutcome};
use vd_wire::seams::transfer_control::TransferControlAck;

use crate::orchestrator::DirectoryRes;

/// One live saga: the pure FSM (ctx + state) plus the wrapper-only routing — the gateway
/// `NodeId` to send `TransferControl` to (resolved at creation from the session's
/// directory record; not the FSM's concern).
struct LiveSaga {
    ctx: SagaCtx,
    state: SagaState,
    gateway: NodeId,
    /// The universe tick of the most recent transition that advanced this saga (set
    /// at creation, refreshed on every write-back). A PARKED saga receives no events,
    /// so `commit_result` is never re-entered for it and `since` stays put — which is
    /// exactly when it entered its current (stuck) state. The admin view reports
    /// `now - since` as staleness so a 2am operator sees how long a saga has been wedged.
    since: UniverseTick,
}

/// A pending create-on-trigger. Enqueued by [`SagaRuntimeRes::start_transfer`] and processed
/// next tick (a stub-shard boundary-crossing request wires the real trigger at Slice 1d).
struct PendingStart {
    ctx: SagaCtx,
    gateway: NodeId,
}

/// The live-saga set + the trigger queue, resource-wrapped (single writer: this schedule).
#[derive(Resource, Default)]
pub struct SagaRuntimeRes {
    sagas: BTreeMap<TransferId, LiveSaga>,
    pending: Vec<PendingStart>,
    /// Typed client-facing rejections awaiting the surfacing system. BOUNDED ring
    /// (`REJECTION_LEDGER_CAP`): the PROPER consumer — the typed rejection→client channel —
    /// lands at Slice 1c.9 (CPO-4) and drains via `std::mem::take` (lifetime ONE cycle,
    /// matching `pending`). Until then this is capped so it can NEVER grow without limit on
    /// the orchestrator at MMO scale (audit ROB-1): on overflow the OLDEST rejections are
    /// shed with a counted + warned drop (`rejections_dropped`) — never an unbounded leak,
    /// never silent.
    pub rejected: Vec<(TransferId, AbortReason)>,
    /// Rejections shed because `rejected` reached `REJECTION_LEDGER_CAP` before the 1c.9
    /// consumer drained it — the loud overflow ALERT (0 in any healthy run).
    pub rejections_dropped: u64,
}

/// Hard ceiling on the un-drained rejection ledger (operational param, ONE home — no inline
/// literal). Sized to absorb a large abort burst between 1c.9 drain cycles; beyond it the
/// oldest rejections are shed (counted) rather than grow the orchestrator without bound.
const REJECTION_LEDGER_CAP: usize = 1024;

impl SagaRuntimeRes {
    /// Trigger a new transfer (create-on-trigger). `ctx.subject` MUST already be a directory
    /// record — [`drive_sagas`] asserts `lock_transfer` succeeds (a missing/already-locked
    /// subject refuses the start). `gateway` is the session's gateway (the caller resolves it
    /// from the `Session` directory record).
    ///
    /// 1c CONTRACT (audit CPO-4): a lock refusal is currently a silent drop (acceptable while
    /// the only caller is a test rig). When the 1c rejection-surfacing channel lands, refusals
    /// MUST route through that SAME channel with a typed reason (SubjectLocked /
    /// SubjectUnrecorded) so a shard-initiated trigger (1d) always learns its fate.
    pub fn start_transfer(&mut self, ctx: SagaCtx, gateway: NodeId) {
        self.pending.push(PendingStart { ctx, gateway });
    }

    /// Live saga count (the oracle/admin surface; bounded — terminal sagas are tombstoned).
    #[must_use]
    pub fn live(&self) -> usize {
        self.sagas.len()
    }

    /// Operator-facing view of every live (in-flight) saga — the admin snapshot's
    /// `sagas` list, the 2am `curl` that AAA-1 promised. Bounded: terminal sagas are
    /// tombstoned, so this is the in-flight set only, in deterministic `TransferId`
    /// order (BTreeMap). `SagaState` is `Debug`-only and lives in `vd-sim`, so the
    /// operator string is its `{:?}` (wire cannot depend on sim — admin.rs §SagaView).
    #[must_use]
    pub fn views(&self) -> Vec<vd_wire::admin::SagaView> {
        self.sagas
            .values()
            .map(|live| vd_wire::admin::SagaView {
                transfer: live.ctx.transfer.to_string(),
                state: format!("{:?}", live.state),
                since: live.since,
            })
            .collect()
    }
}

/// Map a gateway→saga ack to the FSM event it drives. `CasWon`/`CasLost` are NOT here — they
/// come from the DIRECT `commit_cas`, never the wire.
fn ack_to_event(ack: TransferControlAck) -> SagaEvent {
    match ack {
        TransferControlAck::Prepared { result, .. } => SagaEvent::Prepared(result),
        TransferControlAck::CutConfirmed { marker_seq, .. } => {
            SagaEvent::CutConfirmed { marker_seq }
        }
        TransferControlAck::SourceFrozen { drained_seq, .. } => {
            SagaEvent::SourceFrozen { drained_seq }
        }
        // The gateway's `CommitAuthority` ack = the route swapped (post-commit, forward-only).
        TransferControlAck::Committed { .. } => SagaEvent::RouteSwapped,
        TransferControlAck::SourceThawed { .. } => SagaEvent::SourceThawed,
        TransferControlAck::Aborted { .. } => SagaEvent::DestAborted,
        TransferControlAck::Released { .. } => SagaEvent::Released,
    }
}

/// The action executor: run a saga from `(state, actions)` to quiescence, executing each
/// action and feeding any SYNCHRONOUS follow-up event (the direct CAS outcome) back into the
/// FSM. Returns the final state, whether it tombstoned (terminal), and any client rejections.
///
/// ORDERING CONTRACT (audit FF-1): a batch's actions execute IN EMITTED ORDER and the WHOLE
/// batch completes before the next queued event is stepped — so when `CasWon` produces
/// `[PersistCheckpoint, Send(CommitAuthority)]`, the send happens before any later event is
/// processed. The FSM emits at most ONE synchronous-feedback action (the commit) per batch,
/// so the loop depth is bounded (it cannot ping-pong); a second feedback source added later
/// must preserve both properties.
fn run_to_quiescence(
    ctx: &SagaCtx,
    gateway: NodeId,
    mut state: SagaState,
    mut actions: Vec<SagaAction>,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    now: UniverseTick,
) -> (SagaState, bool, Vec<(TransferId, AbortReason)>) {
    let mut events: VecDeque<SagaEvent> = VecDeque::new();
    let mut tombstone = false;
    let mut rejected = Vec::new();
    loop {
        for action in actions {
            match action {
                SagaAction::Send(cmd) => {
                    outbox.push_flow(gateway, MsgClass::Saga, &InterShardFlow::Saga(cmd));
                }
                // THE single commit point, called DIRECTLY (this thread owns the directory).
                // The outcome re-enters the FSM immediately — no wire round-trip, no tick split.
                SagaAction::IssueCommitCas { expected } => {
                    let outcome =
                        dir.commit_cas(ctx.subject, expected, AuthorityRef::Shard(ctx.dest), now);
                    events.push_back(match outcome {
                        CasOutcome::Won { new_fence } => SagaEvent::CasWon { new_fence },
                        CasOutcome::Lost { current } => SagaEvent::CasLost { current },
                    });
                }
                // HR2: a Transient subject commits via the batched go-token (P3); present +
                // classified now. LOUD deferral: the saga will PARK in CommittingCas with the
                // key locked — an operator must see why, never wonder (audit ROB-2/CPO-3).
                SagaAction::IssueTransientGo { .. } => {
                    tracing::warn!(
                        transfer = ctx.transfer.0,
                        "TransientGo is a P3 stub: the transient saga parks in CommittingCas \
                         (no go-token flow yet)"
                    );
                }
                // P3: group-committed redb WAL. LOUD deferral: an orchestrator restart loses
                // every in-flight saga until then — never let a log read as if durability
                // exists (audit ROB-2).
                SagaAction::PersistCheckpoint => {
                    tracing::warn!(
                        transfer = ctx.transfer.0,
                        "PersistCheckpoint is a P3 stub: saga state is IN-MEMORY ONLY \
                         (a restart loses in-flight sagas)"
                    );
                }
                SagaAction::NotifyRejected(reason) => rejected.push((ctx.transfer, reason)),
                SagaAction::Tombstone => tombstone = true,
            }
        }
        match events.pop_front() {
            Some(event) => {
                let (next, acts) = saga::step(ctx, state, event);
                state = next;
                actions = acts;
            }
            None => break,
        }
    }
    (state, tombstone, rejected)
}

/// Apply one event to an existing saga, then run it to quiescence + persist the result. A
/// stale event for an unknown/GC'd saga is an idempotent no-op (at-least-once delivery).
fn deliver(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    now: UniverseTick,
    transfer: TransferId,
    event: SagaEvent,
) {
    let Some(live) = runtime.sagas.get(&transfer) else {
        return;
    };
    let ctx = live.ctx;
    let gateway = live.gateway;
    let (state, actions) = saga::step(&ctx, live.state, event);
    let (final_state, tombstone, rejected) =
        run_to_quiescence(&ctx, gateway, state, actions, dir, outbox, now);
    commit_result(runtime, transfer, final_state, tombstone, rejected, now);
}

/// ROB-1: bound the un-drained rejection ledger to `REJECTION_LEDGER_CAP`, shedding the
/// OLDEST beyond it with a counted, warned drop — never an unbounded leak, never silent.
/// Monomorphic helper (the branch is covered once; `commit_result` stays a straight shim).
fn bound_rejection_ledger(runtime: &mut SagaRuntimeRes) {
    let shed = runtime.rejected.len().saturating_sub(REJECTION_LEDGER_CAP);
    if shed > 0 {
        runtime.rejected.drain(0..shed);
        runtime.rejections_dropped += shed as u64;
        tracing::warn!(
            shed,
            cap = REJECTION_LEDGER_CAP,
            "saga rejection ledger over cap: shed oldest (un-surfaced until Slice 1c.9 CPO-4); \
             never silent"
        );
    }
}

/// Persist a quiescent saga: record rejections, then GC (terminal) or write back the state.
fn commit_result(
    runtime: &mut SagaRuntimeRes,
    transfer: TransferId,
    final_state: SagaState,
    tombstone: bool,
    rejected: Vec<(TransferId, AbortReason)>,
    now: UniverseTick,
) {
    runtime.rejected.extend(rejected);
    bound_rejection_ledger(runtime); // ROB-1: never let the un-drained ledger grow unbounded
    if tombstone {
        runtime.sagas.remove(&transfer);
    } else {
        // Sound because nothing between the lookup and this write-back can touch
        // `runtime.sagas`: `run_to_quiescence` takes only `dir`/`outbox` (the borrow
        // checker enforces it cannot reach the runtime), and this schedule is
        // single-threaded. If a future refactor hands it the runtime, re-prove this.
        let live = runtime
            .sagas
            .get_mut(&transfer)
            .expect("the saga was present at the start of this run");
        // AAA-1 staleness: refresh `since` ONLY on an ACTUAL state change. `deliver` runs
        // for EVERY matching ack — including at-least-once duplicates / stray acks the FSM
        // absorbs as same-state (`saga.rs`'s `(state, _) => (state, vec![])` no-op arm) — so
        // an unconditional bump would RESET a parked saga's staleness on a redelivery,
        // defeating the very stuck-saga metric this field exists for. The stored `live.state`
        // is still the prior state here (single-threaded schedule; nothing touched it between
        // the lookup and now), so the compare is exact.
        if live.state != final_state {
            live.since = now;
        }
        live.state = final_state;
    }
}

/// Process the create-on-trigger queue: per-key-serialize via `lock_transfer`, `start` the
/// FSM, and run its initial actions to quiescence.
fn process_starts(
    runtime: &mut SagaRuntimeRes,
    dir: &mut DirectoryCore,
    outbox: &mut OutboundBox,
    now: UniverseTick,
) {
    for PendingStart { ctx, gateway } in std::mem::take(&mut runtime.pending) {
        // Per-key serialization: `lock_transfer` sets `in_transfer`. FALSE means the subject
        // is absent OR already transferring — refuse to start (a spurious/duplicate trigger
        // is a no-op, never a second concurrent saga on the same key).
        if !dir.lock_transfer(ctx.subject, ctx.transfer) {
            continue;
        }
        let (state, actions) = saga::start(&ctx);
        runtime.sagas.insert(
            ctx.transfer,
            LiveSaga {
                ctx,
                state,
                gateway,
                since: now,
            },
        );
        let (final_state, tombstone, rejected) =
            run_to_quiescence(&ctx, gateway, state, actions, dir, outbox, now);
        commit_result(runtime, ctx.transfer, final_state, tombstone, rejected, now);
    }
}

/// The orchestrator saga-runtime system: process new triggers, then drive every live saga
/// forward on the gateway acks delivered this tick. Runs on the orchestrator's single-threaded
/// schedule; the directory CAS is a direct in-process call (no await, no lock across a send).
pub fn drive_sagas(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    mut dir: ResMut<DirectoryRes>,
    mut runtime: ResMut<SagaRuntimeRes>,
    mut outbox: ResMut<OutboundBox>,
) {
    let now = clock.universe_tick;
    process_starts(&mut runtime, &mut dir.0, &mut outbox, now);
    for msg in &inbox.0 {
        let Inbound::Wire { class, bytes, .. } = msg else {
            continue;
        };
        if *class != MsgClass::Saga {
            continue;
        }
        // Only the SagaAck arm drives the saga here; Directory ops are `serve_directory`'s,
        // and Saga commands flow OUT to the gateway, never in. A non-SagaAck Saga-class
        // message is left for its handler (decode-fail is `serve_directory`'s to log).
        let Ok(InterShardFlow::SagaAck(ack)) = postcard::from_bytes::<InterShardFlow>(bytes) else {
            continue;
        };
        deliver(
            &mut runtime,
            &mut dir.0,
            &mut outbox,
            now,
            ack.transfer(),
            ack_to_event(ack),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{NodeConfig, build_app};
    use crate::orchestrator::{OrchestratorConfig, register_orchestrator};
    use vd_core::entity_kind::{DurabilityClass, EntityKind};
    use vd_core::{EntityId, EpochId, Fence, SessionId};
    use vd_sim::capability::NodeKind;
    use vd_sim::directory::DirectoryTuning;
    use vd_sim::io::Transport;
    use vd_sim::io::mem::MemHub;
    use vd_wire::seams::directory::{DirectoryKey, DirectoryOp};
    use vd_wire::seams::transfer_control::{
        PrepareReject, PrepareResult, SpatialReject, TransferControl,
    };

    const ORCH: NodeId = NodeId(1);
    const SOURCE: NodeId = NodeId(2);
    const DEST: NodeId = NodeId(3);
    const GATEWAY: NodeId = NodeId(4);
    const XFER: TransferId = TransferId(77);
    const SESSION: SessionId = SessionId(5);

    fn subject() -> DirectoryKey {
        DirectoryKey::Entity(EntityId::pack(EntityKind::Player, 1, 7, 3))
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
        }
    }

    /// A stepped orchestrator + a hub the test drives the saga through.
    struct Rig {
        hub: MemHub,
        orch: crate::app::ShardNode<vd_sim::io::mem::MemTransport>,
        gateway: vd_sim::io::mem::MemTransport,
    }

    impl Rig {
        fn new() -> Rig {
            let hub = MemHub::new();
            let mut orch = build_app(
                NodeConfig {
                    node_id: ORCH,
                    kind: NodeKind::Orchestrator,
                },
                hub.register(ORCH, 64),
            );
            let (world, schedule) = orch.parts_mut();
            register_orchestrator(
                world,
                schedule,
                &OrchestratorConfig {
                    epoch: EpochId(1),
                    reserve_chunk: 1024,
                    clock_peers: vec![],
                    directory: DirectoryTuning {
                        lease_ttl_ticks: 10_000,
                    },
                },
            );
            let gateway = hub.register(GATEWAY, 64);
            Rig { hub, orch, gateway }
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
            let mut shard = self.hub.register(SOURCE, 8);
            shard
                .send(
                    ORCH,
                    MsgClass::Saga,
                    vd_sim::io::bytes(
                        postcard::to_allocvec(&InterShardFlow::Directory(
                            DirectoryOp::LeaseGrant {
                                key: subject(),
                                owner: AuthorityRef::Shard(SOURCE),
                                fence,
                            },
                        ))
                        .expect("encode"),
                    ),
                )
                .expect("sent");
            self.settle();
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

        // SourceFrozen → the DIRECT commit_cas wins in-process → CommitAuthority is sent with
        // the NEW fence, all in ONE tick (no wire round-trip for the CAS).
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 42,
        });
        let new_fence = Fence(1).next();
        assert_eq!(
            rig.drain_gateway(),
            vec![saga_wire(TransferControl::CommitAuthority {
                transfer: XFER,
                session: SESSION,
                new_fence,
            })]
        );
        // The directory CAS committed: the dest now owns the subject at the new fence.
        let head = rig.subject_head();
        assert_eq!(head.authority, AuthorityRef::Shard(DEST));
        assert_eq!(head.fence, new_fence);

        // Committed (the route swapped) advances to Demoting (the tail — DemoteComplete +
        // Released — lands with the demote signal at Slice 1c). The saga is still live.
        rig.ack(TransferControlAck::Committed { transfer: XFER });
        assert!(
            rig.drain_gateway().is_empty(),
            "Demoting awaits the demote signal"
        );
        assert_eq!(rig.live(), 1);
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
        // PINNED 1b LIMIT (audit CPO-1): the directory lock survives the abort — the subject
        // is wedged until Slice 2's abort_cas clears it. This assertion EXISTS TO BE FLIPPED
        // by that fix (to `None`); it must never pass silently in either direction.
        assert_eq!(
            head.in_transfer,
            Some(XFER),
            "Slice-2 abort_cas not yet wired: the abort leaves the lock set (see module doc)"
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
        // PINNED 1b LIMIT (audit CPO-2): the CAS-loser's abort also leaves the lock set —
        // and clearing it NEEDS the Slice-2 head re-read (this saga's expected fence is
        // stale, so a naive abort_cas(expected) would lose too). Flipped by the Slice-2 fix.
        assert_eq!(
            rig.subject_head().in_transfer,
            Some(XFER),
            "Slice-2 abort_cas+re-read not yet wired (see module doc)"
        );
    }

    #[test]
    fn transient_subject_commits_via_the_go_token_not_a_cas() {
        // HR2 fan-out on ONE FSM: a Transient subject reaches the commit point and issues the
        // batched go-token (a no-op stub here), NOT a per-entity directory CAS — so the
        // subject's fence is unchanged after SourceFrozen.
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Transient, Fence(1)));
        rig.settle();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 3,
        });
        let _ = rig.drain_gateway();
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });
        // No CommitAuthority (the go-token stub didn't produce a CasWon), no fence bump.
        assert!(
            rig.drain_gateway().is_empty(),
            "Transient issues a go-token, not a CAS → no CommitAuthority yet"
        );
        assert_eq!(rig.subject_head().fence, Fence(1));
        assert_eq!(rig.live(), 1);

        // AAA-1: the parked saga is CURL-VISIBLE through the real admin snapshot — its
        // transfer id, its actual stuck phase, and a staleness anchor, never silently
        // wedged. (This drives both `SagaRuntimeRes::views` and `admin_snapshot`'s
        // saga-population end to end.)
        let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        assert_eq!(snap.sagas.len(), 1);
        assert_eq!(snap.sagas[0].transfer, XFER.to_string());
        // The operator sees the REAL parked phase with its fields (acked marker_seq=3,
        // drained_seq=3), not a placeholder. Exact equality (no `assert!`-message format
        // arm — the HR5 uncoverable-region discipline).
        assert_eq!(
            snap.sagas[0].state,
            "CommittingCas { marker_seq: 3, drained_seq: 3 }"
        );
        // `since` was set when the saga last advanced and is bounded by the clock.
        assert!(snap.sagas[0].since.0 <= snap.universe_tick);
    }

    #[test]
    fn a_stray_ack_to_a_parked_saga_preserves_its_staleness_anchor() {
        // AAA-1-SINCE regression guard: a PARKED saga that receives a duplicate / stray ack
        // the FSM absorbs as same-state must KEEP its `since`. An unconditional bump would
        // reset stuck-saga staleness on every at-least-once redelivery. (This also exercises
        // the same-state arm of the `final_state != prior` gate in `commit_result`.)
        let mut rig = Rig::new();
        rig.grant_subject(Fence(1));
        rig.trigger(ctx(DurabilityClass::Transient, Fence(1)));
        rig.settle();
        rig.ack(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        });
        rig.ack(TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 3,
        });
        let _ = rig.drain_gateway();
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        }); // parks in CommittingCas (the IssueTransientGo stub yields no CasWon)
        let since_parked = crate::orchestrator::admin_snapshot(rig.orch.world_mut()).sagas[0].since;

        // Two duplicate SourceFrozen acks (at-least-once redelivery). The FSM absorbs each
        // as same-state; the clock advances on every step.
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });
        rig.ack(TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 3,
        });

        let snap = crate::orchestrator::admin_snapshot(rig.orch.world_mut());
        assert_eq!(
            snap.sagas.len(),
            1,
            "still parked, not advanced by stray acks"
        );
        assert_eq!(
            snap.sagas[0].state, "CommittingCas { marker_seq: 3, drained_seq: 3 }",
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
        let cases: [(TransferControlAck, SagaEvent); 7] = [
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
}
