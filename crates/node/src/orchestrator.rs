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
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, OutboundBox};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{DirectoryOp, DirectoryReply};

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
}

/// The directory, resource-wrapped (single writer: this node's schedule).
#[derive(Resource)]
pub struct DirectoryRes(pub DirectoryCore);

/// The authoritative universe clock, resource-wrapped.
#[derive(Resource)]
pub struct UniverseClockRes(pub CeilingClock);

#[derive(Resource, Clone, Debug)]
struct ClockPeers(Vec<NodeId>);

/// Install the orchestrator systems. Genesis-reserves the clock ceiling and
/// confirms it in memory (the durable wrapper arrives with redb at P3).
///
/// # Panics
/// On a zero `reserve_chunk` — an operator configuration error, failed loud at boot.
pub fn register_orchestrator(world: &mut World, schedule: &mut Schedule, cfg: &OrchestratorConfig) {
    let (mut clock, ClockAction::ReserveCeiling(ceiling)) =
        CeilingClock::genesis(cfg.epoch, cfg.reserve_chunk).expect("non-zero reserve chunk");
    clock
        .confirm_ceiling(ceiling)
        .expect("genesis ceiling confirms exactly once");
    world.insert_resource(UniverseClockRes(clock));
    world.insert_resource(DirectoryRes(DirectoryCore::new(cfg.directory)));
    world.insert_resource(ClockPeers(cfg.clock_peers.clone()));
    world.insert_resource(crate::saga_runtime::SagaRuntimeRes::default());
    // serve_directory then drive_sagas: both read the Saga-class inbound (directory ops vs
    // gateway acks); the saga runtime's direct commit_cas runs after the directory service.
    schedule.add_systems(
        (
            advance_and_broadcast_clock,
            serve_directory,
            crate::saga_runtime::drive_sagas,
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

    let flow = InterShardFlow::Directory(DirectoryOp::ClockSync {
        universe_tick: now,
        epoch: clock.0.epoch(),
    });
    // ONE shared body, refcount-bumped to every clock peer.
    let bytes = vd_sim::io::bytes(
        postcard::to_allocvec(&flow).expect("closed wire enums serialize infallibly"),
    );
    for peer in &peers.0 {
        outbox.0.push((*peer, MsgClass::Membership, bytes.clone()));
    }
}

/// Serve directory operations arriving on the Saga class; replies return to the
/// sender. The directory is authority-of-record — callers compare the returned
/// head against what they requested (hints never decide authority).
fn serve_directory(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    mut dir: ResMut<DirectoryRes>,
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
            let bytes = vd_sim::io::bytes(
                postcard::to_allocvec(&reply).expect("closed wire enums serialize infallibly"),
            );
            outbox.0.push((*from, MsgClass::Saga, bytes));
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
            },
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

    /// The expected wire form of one directory reply from the orchestrator.
    fn reply_wire(reply: &DirectoryReply) -> Inbound {
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(reply).expect("encode").into(),
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

        // A world without a directory (non-orchestrator) stays shaped-empty.
        let mut bare = bevy_ecs::prelude::World::new();
        bare.insert_resource(ClockSample::default());
        assert_eq!(admin_snapshot(&mut bare).directory, vec![]);
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
