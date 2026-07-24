//! Orchestrator-kind systems: the DurableUniverseClock driver, the analytic-clock
//! broadcast, and the Directory service over the frozen seam
//! (`docs/design/transfer_protocol.md` §4, `identity_persistence.md` §clock).
//!
//! The orchestrator is the SINGLE WRITER of the directory and the owner of
//! universe time. In P1 the write-ahead ceiling is confirmed by the in-memory
//! stand-in (exactly what a memory store does); the P3 redb wrapper persists
//! `ReserveCeiling` actions before confirming — the clock discipline is identical.

use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::{EpochId, NodeId};
use vd_sim::directory::{DirectoryCore, DirectoryTuning};
use vd_sim::io::mem::{MemHub, MemSpawner, MemStore};
use vd_sim::io::{Inbound, MsgClass, Store};
use vd_sim::rlm::RlmTuning;
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{DirectoryOp, DirectoryReply};

use crate::rlm_runtime::{RlmReconcilerRes, reconcile_realm_lifecycle, record_realm_demands};
use crate::universe_clock::{CeilingClock, ClockAction};

/// Orchestrator configuration (composer-provided; ONE struct, no inline literals).
#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    pub epoch: EpochId,
    /// Write-ahead reservation chunk for the universe clock.
    pub reserve_chunk: u64,
    /// Nodes that receive the per-tick `ClockSync` broadcast (gateways + shards).
    pub clock_peers: Vec<NodeId>,
    pub directory: DirectoryTuning,
    /// The saga deadline budget (Slice 2a) — the timeout producer's two thresholds. The bin reads
    /// it from env; in-process rigs use `SagaTuning::default()`.
    pub saga: vd_sim::saga::SagaTuning,
    /// The D-3 dead-vs-slow confirmation tuning. The bin reads it from env with a PROD-SAFE default
    /// (`n_consecutive_unreachable >= 3` — the CSCALE-1 margin so a single recoverable blip never
    /// confirms a healthy peer dead); in-process rigs use `LivenessTuning::default()` (kill-equivalent
    /// `n == 1`, so the existing crash cells keep their behavior).
    pub liveness: vd_sim::saga::LivenessTuning,
    /// D-37 forward re-home target roster: `NodeId → ShardProfile` for the shards a re-home may land on.
    /// The cluster builder maps each stub shard to the empty profile; the prod bin leaves it EMPTY for
    /// now (ledgered — prod re-home parks until the per-shard-profile roster config lands; P3 is
    /// harness-driven). An empty roster ⇒ `select_rehome_target` returns `None` ⇒ the saga stays parked.
    pub roster: std::collections::BTreeMap<NodeId, vd_sim::capability::ShardProfile>,
    /// RLM Step 3 — the realm-lifecycle reconciler's timing budget. DEFAULT is INERT
    /// (`reconcile_interval_ticks == 0` ⇒ the reconcile sweep never runs ⇒ byte-identical to a build
    /// without RLM); the live-AoI orchestrator boot sets `RlmTuning::cloud(tick_hz)`.
    pub rlm: RlmTuning,
}

/// The directory, resource-wrapped (single writer: this node's schedule).
#[derive(Resource)]
pub struct DirectoryRes(pub DirectoryCore);

/// The authoritative universe clock, resource-wrapped.
#[derive(Resource)]
pub struct UniverseClockRes(pub CeilingClock);

#[derive(Resource, Clone, Debug)]
struct ClockPeers(Vec<NodeId>);

/// Orchestrator-side honesty counters — tolerated anomalies that must never be silent.
#[derive(Resource, Debug, Default, PartialEq, Eq)]
pub struct OrchestratorStats {
    /// Saga-class frames that failed to decode at the directory ingress (a malformed
    /// or garbage envelope). Dropped, never mis-applied, but counted so a decode
    /// regression is observable rather than log-only (ROB-E2E-1; mirrors the gateway
    /// and stub `undecodable`). 0 in any healthy run.
    pub undecodable: u64,
}

/// Install the orchestrator systems. Genesis-reserves the clock ceiling and
/// confirms it in memory (the durable wrapper arrives with redb at P3).
///
/// # Panics
/// On a zero `reserve_chunk` — an operator configuration error, failed loud at boot.
pub fn register_orchestrator(world: &mut World, schedule: &mut Schedule, cfg: &OrchestratorConfig) {
    // D-6: every orchestrator has a durable Store. The default uses a FRESH MemStore (genesis) — the
    // ~5 unrelated unit-test/bin call sites need no churn, and `drive_sagas`'s `StoreRes` is always
    // present. A caller that needs a RECOVERABLE (kill-9-survivable) orchestrator passes a retained
    // handle via [`register_orchestrator_with_store`]; redb is the io-prod backend (DEFERRED).
    register_orchestrator_with_store(world, schedule, cfg, Box::new(MemStore::new()));
}

/// As [`register_orchestrator`], but against a caller-provided durable [`Store`] (D-6). A NON-EMPTY
/// store (a prior orchestrator's committed WAL) is RE-HYDRATED — the clock resumes forward, the
/// directory + sagas + go-tokens restore, and the existing Slice-2a Timeout producer re-drives each
/// in-flight saga to terminal; an EMPTY store is a fresh genesis. The harness retains the concrete
/// handle to re-attach it to a rebuilt orchestrator (the kill-9 crash cells).
pub fn register_orchestrator_with_store(
    world: &mut World,
    schedule: &mut Schedule,
    cfg: &OrchestratorConfig,
    store: Box<dyn Store + Send + Sync>,
) {
    let (clock, directory, mut runtime) = match crate::saga_runtime::rehydrate(
        store.as_ref(),
        cfg.reserve_chunk,
        cfg.saga,
        cfg.liveness,
        cfg.directory,
    ) {
        // RECOVER: the rebuilt orchestrator resumes its durable state (no in-flight transfer vanishes).
        Some(r) => (r.clock, r.directory, r.runtime),
        // GENESIS: a fresh orchestrator reserves the clock ceiling + starts with an empty directory/saga set.
        None => {
            let (mut clock, ClockAction::ReserveCeiling(ceiling)) =
                CeilingClock::genesis(cfg.epoch, cfg.reserve_chunk)
                    .expect("non-zero reserve chunk");
            clock
                .confirm_ceiling(ceiling)
                .expect("genesis ceiling confirms exactly once");
            (
                clock,
                DirectoryCore::new(cfg.directory),
                crate::saga_runtime::SagaRuntimeRes::with_tunings(cfg.saga, cfg.liveness),
            )
        }
    };
    // D-37: seed the re-home target roster (RAM-only operational config — re-seeded on recover too, like
    // clock_peers; the dead-vs-slow tracker rebuilds empty but its config survives). Single-point wiring.
    runtime.set_roster(cfg.roster.clone());
    world.insert_resource(UniverseClockRes(clock));
    world.insert_resource(DirectoryRes(directory));
    world.insert_resource(ClockPeers(cfg.clock_peers.clone()));
    world.insert_resource(OrchestratorStats::default());
    world.insert_resource(runtime);
    world.insert_resource(crate::saga_runtime::StoreRes(store));
    // RLM Step 3 — the realm-lifecycle reconciler. INERT by default (`cfg.rlm` zero ⇒ the sweep never
    // runs ⇒ byte-identical). The spawner defaults to a fresh in-process `MemSpawner` (mint ids from a
    // high base so they never collide with hand-picked test node ids); a live-AoI / harness boot overwrites
    // this resource with a spawner tied to its own hub (Step 5 supplies the real k8s launcher).
    world.insert_resource(RlmReconcilerRes::new(
        cfg.rlm,
        Box::new(MemSpawner::new(MemHub::new(), NodeId(1_000_000), 8)),
    ));
    // RLM Step 3e chain. `record_realm_demands` folds this tick's re-asserted demands BEFORE `serve_directory`
    // (so they are fresh). `drive_sagas_core` runs the saga FSMs + the reaper + rehome-arm (mutating the
    // directory). `reconcile_realm_lifecycle` then reads the post-CAS heads + liveness and stages its
    // grant/revoke into the SAME `dir.dirty` set. `commit_barrier` is the ONE fsync at the chain tail — it
    // drains every mutation this tick and runs BEFORE the node's flush phase sends `outbox`
    // (persist-before-effect; the D-6 + COMP-2 guarantees are preserved — the barrier was only MOVED out of
    // `drive_sagas`, not reordered).
    schedule.add_systems(
        (
            advance_and_broadcast_clock,
            record_realm_demands,
            serve_directory,
            crate::saga_runtime::drive_sagas_core,
            reconcile_realm_lifecycle,
            crate::saga_runtime::commit_barrier,
        )
            .chain(),
    );
}

/// Advance the authoritative clock one tick and broadcast `ClockSync` to every
/// peer (Membership class — re-derivable, loss-tolerated).
fn advance_and_broadcast_clock(
    mut clock: ResMut<UniverseClockRes>,
    mut sample: ResMut<ClockSample>,
    peers: Res<ClockPeers>,
    mut outbox: ResMut<OutboundBox>,
) {
    let (now, action) = clock
        .0
        .advance()
        .expect("the in-memory wrapper always confirms the ceiling ahead of use");
    if let Some(ClockAction::ReserveCeiling(ceiling)) = action {
        clock
            .0
            .confirm_ceiling(ceiling)
            .expect("in-memory reservation confirms immediately");
    }
    sample.universe_tick = now;
    sample.epoch = clock.0.epoch();

    // The clock broadcast is a small Membership-class cold-path flow; route each peer
    // through the ONE shared encode-and-push (DRY-1). Membership is re-derivable +
    // loss-tolerated, so a per-peer re-encode is immaterial (if profiling ever shows it
    // matters, restore the encode-once/clone-per-peer inline — this is not a hot path).
    let flow = InterShardFlow::Directory(DirectoryOp::ClockSync {
        universe_tick: now,
        epoch: clock.0.epoch(),
    });
    for peer in &peers.0 {
        outbox.push_flow(*peer, MsgClass::Membership, &flow);
    }
}

/// Serve directory operations arriving on the Saga class; replies return to the
/// sender. The directory is authority-of-record — callers compare the returned
/// head against what they requested (hints never decide authority).
fn serve_directory(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    mut dir: ResMut<DirectoryRes>,
    mut stats: ResMut<OrchestratorStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            continue;
        };
        if *class != MsgClass::Saga {
            continue;
        }
        let flow = match postcard::from_bytes::<InterShardFlow>(bytes) {
            Ok(flow) => flow,
            Err(_) => {
                stats.undecodable += 1;
                tracing::error!("undecodable saga-class message from {from}");
                continue;
            }
        };
        // Only Directory ops are this service's; SagaAck (gateway → saga) is `drive_sagas`',
        // and Saga commands flow OUT to the gateway, never in — skip the non-Directory arms.
        let InterShardFlow::Directory(op) = flow else {
            continue;
        };
        if let Some(reply) = apply_directory_op(&mut dir.0, op, &clock) {
            // Wrap the reply in its own InterShardFlow arm (DRY-1 site 4 + the dispatch
            // fix): the requester decodes InterShardFlow once and routes by variant, so a
            // reply and a Saga(TransferControl) on the same MsgClass::Saga never collide.
            outbox.push_flow(
                *from,
                MsgClass::Saga,
                &InterShardFlow::DirectoryReply(reply),
            );
        }
    }
}

/// One directory operation → at most one reply. Fire-and-forget ops (renewals)
/// reply nothing; reads and side-effecting ops return the resulting head/outcome.
fn apply_directory_op(
    dir: &mut DirectoryCore,
    op: DirectoryOp,
    clock: &ClockSample,
) -> Option<DirectoryReply> {
    let now = clock.universe_tick;
    match op {
        DirectoryOp::LeaseGrant { key, owner, fence } => {
            // Outcome is communicated through the resulting head: the caller
            // compares (owner, fence) — a refusal returns the standing record.
            let _ = dir.grant(key, owner, fence, now);
            Some(DirectoryReply::Head {
                key,
                record: dir.head(key),
            })
        }
        DirectoryOp::LeaseRenew { key, fence } => {
            // Soft renewal: loss-tolerated, no reply traffic.
            let _ = dir.renew(key, fence, now);
            None
        }
        DirectoryOp::LeaseRevoke { key, fence } => {
            let _ = dir.revoke(key, fence);
            Some(DirectoryReply::Head {
                key,
                record: dir.head(key),
            })
        }
        DirectoryOp::CommitCas {
            key,
            expected,
            transfer: _,
            new_owner,
        } => Some(DirectoryReply::CasResult {
            key,
            outcome: dir.commit_cas(key, expected, new_owner, now),
        }),
        DirectoryOp::AbortCas {
            key,
            expected,
            transfer: _,
        } => Some(DirectoryReply::CasResult {
            key,
            outcome: dir.abort_cas(key, expected),
        }),
        DirectoryOp::HeadRead { key } => Some(DirectoryReply::Head {
            key,
            record: dir.head(key),
        }),
        DirectoryOp::ClockSync { .. } => {
            // The orchestrator OWNS the clock; an inbound sync is a peer's
            // misdirected broadcast — answer with the authoritative value.
            Some(DirectoryReply::ClockNow {
                universe_tick: now,
                epoch: clock.epoch,
            })
        }
    }
}

/// Build the read-only admin snapshot from a (live or test) orchestrator world —
/// the bin publishes this through `vd-io-prod`'s admin shell after each tick, so a
/// 2am `curl` (and the process-tier parity test) sees the real directory.
#[must_use]
pub fn admin_snapshot(world: &mut bevy_ecs::prelude::World) -> vd_wire::admin::AdminSnapshot {
    let clock = *world.resource::<ClockSample>();
    let mut snapshot =
        vd_wire::admin::AdminSnapshot::shaped_empty(clock.universe_tick, clock.epoch);
    if let Some(dir) = world.get_resource::<DirectoryRes>() {
        snapshot.directory = dir
            .0
            .entries()
            .map(|(key, record)| vd_wire::admin::directory_entry_view(key, record))
            .collect();
        // D-3 Slice 4: per-record lease health, so a lapsed-pending lease (negative ticks_remaining —
        // awaiting reaper confirmation / a D-37 re-home) is VISIBLE to a 2am operator, never a silent wedge.
        snapshot.leases = dir
            .0
            .entries()
            .map(|(_, record)| vd_wire::admin::LeaseHealthView {
                node: record.authority.node(),
                lease_expires: record.lease_expires,
                // Negative = LAPSED. Straight-line cast (no uncoverable fallback branch, HR5); universe
                // ticks are far below i64::MAX so the cast is exact.
                ticks_remaining: record.lease_expires.0 as i64 - clock.universe_tick.0 as i64,
            })
            .collect();
    }
    // AAA-1: the in-flight sagas — the "curl a stuck saga at 2am" promise. A saga that
    // parks (e.g. a P3-stub TransientGo, or a Demoting tail awaiting the band predicate)
    // stays in this set with its state + staleness visible, never silently wedged.
    if let Some(sagas) = world.get_resource::<crate::saga_runtime::SagaRuntimeRes>() {
        snapshot.sagas = sagas.views();
    }
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{NodeConfig, build_app};
    use vd_core::pose::RealmId;
    use vd_core::{Fence, SessionId, UniverseTick};
    use vd_sim::capability::NodeKind;
    use vd_sim::io::Transport;
    use vd_sim::io::mem::MemHub;
    use vd_wire::seams::directory::{AuthorityRef, CasOutcome, DirectoryKey};

    const ORCH: NodeId = NodeId(1);
    const SHARD: NodeId = NodeId(2);
    const GATEWAY: NodeId = NodeId(3);

    fn cfg() -> OrchestratorConfig {
        OrchestratorConfig {
            epoch: EpochId(7),
            reserve_chunk: 64,
            clock_peers: vec![SHARD, GATEWAY],
            directory: DirectoryTuning {
                lease_ttl_ticks: 100,
                ..DirectoryTuning::default()
            },
            saga: vd_sim::saga::SagaTuning::default(),
            liveness: vd_sim::saga::LivenessTuning::default(),
            roster: std::collections::BTreeMap::new(),
            rlm: RlmTuning::default(),
        }
    }

    fn flow_bytes(op: DirectoryOp) -> vd_sim::io::Bytes {
        vd_sim::io::bytes(postcard::to_allocvec(&InterShardFlow::Directory(op)).expect("encode"))
    }

    #[test]
    fn clock_advances_and_broadcasts_to_every_peer() {
        let hub = MemHub::new();
        let mut orch = build_app(
            NodeConfig {
                node_id: ORCH,
                kind: NodeKind::Orchestrator,
            },
            hub.register(ORCH, 64),
        );
        let (world, schedule) = orch.parts_mut();
        register_orchestrator(world, schedule, &cfg());
        let mut shard_t = hub.register(SHARD, 64);
        let mut gateway_t = hub.register(GATEWAY, 64);

        let report = orch.step_tick();
        assert_eq!(report.sent, 2, "one ClockSync per peer");
        hub.pump();
        let expected_sync = Inbound::Wire {
            from: ORCH,
            class: MsgClass::Membership,
            bytes: postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::ClockSync {
                universe_tick: UniverseTick(1),
                epoch: EpochId(7),
            }))
            .expect("encode")
            .into(),
        };
        for t in [&mut shard_t, &mut gateway_t] {
            assert_eq!(t.drain_inbound(), vec![expected_sync.clone()]);
        }
        // The sample tracks the authoritative clock.
        assert_eq!(
            orch.world_mut().resource::<ClockSample>().universe_tick,
            UniverseTick(1)
        );

        // The clock keeps advancing across the reservation boundary (the memory
        // wrapper confirms reservations inline — many chunks' worth).
        for _ in 0..200 {
            let _ = orch.step_tick();
        }
        assert_eq!(
            orch.world_mut().resource::<ClockSample>().universe_tick,
            UniverseTick(201)
        );
    }

    /// Drive one directory op through a stepped orchestrator and return the raw
    /// delivered inbound (tests assert FULL wire values — no destructuring).
    fn roundtrip(op: DirectoryOp) -> Vec<Inbound> {
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
                clock_peers: vec![],
                ..cfg()
            },
        );
        let mut requester = hub.register(SHARD, 64);
        requester
            .send(ORCH, MsgClass::Saga, flow_bytes(op))
            .expect("sent");
        hub.pump();
        let _ = orch.step_tick();
        hub.pump();
        requester.drain_inbound()
    }

    /// The expected wire form of one directory reply from the orchestrator — wrapped in
    /// the InterShardFlow::DirectoryReply envelope (the dispatch split).
    fn reply_wire(reply: &DirectoryReply) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::DirectoryReply(*reply))
                .expect("encode")
                .into(),
        }
    }

    #[test]
    fn grants_reads_and_revokes_reply_with_the_head() {
        let key = DirectoryKey::Realm(RealmId::System(5));
        let replies = roundtrip(DirectoryOp::LeaseGrant {
            key,
            owner: AuthorityRef::Shard(SHARD),
            fence: Fence(1),
        });
        // Full-value equality (no destructuring): the grant happened at the
        // orchestrator's universe tick 1, so the lease expires at 1 + ttl.
        assert_eq!(
            replies,
            vec![reply_wire(&DirectoryReply::Head {
                key,
                record: Some(vd_wire::seams::directory::OwnerRecord {
                    authority: AuthorityRef::Shard(SHARD),
                    fence: Fence(1),
                    lease_expires: UniverseTick(101),
                    in_transfer: None,
                }),
            })]
        );

        assert_eq!(
            roundtrip(DirectoryOp::HeadRead { key }),
            vec![reply_wire(&DirectoryReply::Head { key, record: None })],
            "separate orchestrator: empty directory"
        );

        assert_eq!(
            roundtrip(DirectoryOp::LeaseRevoke {
                key,
                fence: Fence(1),
            }),
            vec![reply_wire(&DirectoryReply::Head { key, record: None })],
            "unknown key revoke leaves nothing"
        );
    }

    #[test]
    fn renewals_are_silent_and_cas_replies_with_the_outcome() {
        assert_eq!(
            roundtrip(DirectoryOp::LeaseRenew {
                key: DirectoryKey::Session(SessionId(1)),
                fence: Fence(1),
            }),
            vec![],
            "fire-and-forget renewal"
        );

        let key = DirectoryKey::Session(SessionId(2));
        let lost = reply_wire(&DirectoryReply::CasResult {
            key,
            outcome: CasOutcome::Lost {
                current: Fence::GENESIS,
            },
        });
        assert_eq!(
            roundtrip(DirectoryOp::CommitCas {
                key,
                expected: Fence(1),
                transfer: vd_core::TransferId(9),
                new_owner: AuthorityRef::Gateway(GATEWAY),
            }),
            vec![lost.clone()],
            "CAS on an unknown key loses at genesis"
        );
        assert_eq!(
            roundtrip(DirectoryOp::AbortCas {
                key,
                expected: Fence(1),
                transfer: vd_core::TransferId(9),
            }),
            vec![lost]
        );
    }

    #[test]
    fn misdirected_clock_sync_answers_with_the_authoritative_clock() {
        assert_eq!(
            roundtrip(DirectoryOp::ClockSync {
                universe_tick: UniverseTick(999),
                epoch: EpochId(1),
            }),
            vec![reply_wire(&DirectoryReply::ClockNow {
                universe_tick: UniverseTick(1),
                epoch: EpochId(7),
            })],
            "the orchestrator's own clock, not the peer's claim"
        );
    }

    #[test]
    fn garbage_and_wrong_class_messages_are_survived() {
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
                clock_peers: vec![],
                ..cfg()
            },
        );
        let mut requester = hub.register(SHARD, 64);
        requester
            .send(ORCH, MsgClass::Saga, vec![0xFF, 0x01].into())
            .expect("sent");
        requester
            .send(
                ORCH,
                MsgClass::Control,
                flow_bytes(DirectoryOp::HeadRead {
                    key: DirectoryKey::Session(SessionId(1)),
                }),
            )
            .expect("sent");
        hub.pump();
        let report = orch.step_tick();
        assert_eq!(report.drained, 2);
        assert_eq!(
            report.sent, 0,
            "garbage replied nothing, wrong class ignored"
        );
        // The Saga-class decode failure is COUNTED, never silent (ROB-E2E-1); the
        // wrong-class (Control) message is skipped before decode, so it adds nothing.
        assert_eq!(
            orch.world_mut().resource::<OrchestratorStats>().undecodable,
            1
        );
    }

    #[test]
    fn admin_snapshot_reflects_the_live_directory() {
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
                clock_peers: vec![],
                ..cfg()
            },
        );
        // Empty-but-shaped before any state exists (the P0 demo promise).
        let _ = orch.step_tick();
        let snap = admin_snapshot(orch.world_mut());
        assert_eq!(snap.universe_tick, 1);
        assert_eq!(snap.epoch, 7);
        assert_eq!(snap.directory, vec![]);
        // No saga has been triggered, so the (present) saga runtime renders an empty
        // list — the saga view is populated from real state, never fabricated (AAA-1).
        assert_eq!(snap.sagas, vec![]);
        assert_eq!(
            snap.leases,
            vec![],
            "no records ⇒ no lease-health rows (D-3)"
        );

        // A granted lease appears in the dump.
        let mut requester = hub.register(SHARD, 64);
        requester
            .send(
                ORCH,
                MsgClass::Saga,
                flow_bytes(DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Realm(vd_core::pose::RealmId::System(5)),
                    owner: AuthorityRef::Shard(SHARD),
                    fence: Fence(1),
                }),
            )
            .expect("sent");
        hub.pump();
        let _ = orch.step_tick();
        let snap = admin_snapshot(orch.world_mut());
        assert_eq!(snap.directory.len(), 1);
        assert_eq!(snap.directory[0].authority, "shard:node-2");
        assert_eq!(snap.directory[0].fence, Fence(1));
        // D-3: the granted lease shows up as a lease-health row — the holder + a POSITIVE ticks_remaining
        // (a fresh lease, not lapsed). A lapsed lease would render a negative value (operator-visible).
        assert_eq!(snap.leases.len(), 1);
        assert_eq!(snap.leases[0].node, SHARD);
        assert!(
            snap.leases[0].ticks_remaining > 0,
            "a freshly-granted lease has positive ticks_remaining (not lapsed)"
        );

        // A world without a directory (non-orchestrator) stays shaped-empty.
        let mut bare = bevy_ecs::prelude::World::new();
        bare.insert_resource(ClockSample::default());
        assert_eq!(admin_snapshot(&mut bare).directory, vec![]);
        assert_eq!(admin_snapshot(&mut bare).leases, vec![]);
    }

    #[test]
    fn unreachable_peer_notices_are_skipped_by_the_dispatcher() {
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
                clock_peers: vec![SHARD],
                ..cfg()
            },
        );
        let _shard_endpoint = hub.register(SHARD, 64);
        hub.kill(SHARD);
        // Tick 1 broadcasts ClockSync toward the dead peer; the failure surfaces
        // on a later drain as NodeUnreachable, which the dispatcher skips.
        let _ = orch.step_tick();
        hub.pump();
        let report = orch.step_tick();
        assert_eq!(report.unreachable, 1, "the notice arrived and was observed");
    }
}
