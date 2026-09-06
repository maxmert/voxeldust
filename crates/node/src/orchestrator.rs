//! Orchestrator-kind systems: the DurableUniverseClock driver, the analytic-clock
//! broadcast, and the Directory service over the frozen seam
//! (`docs/design/transfer_protocol.md` §4, `identity_persistence.md` §clock).
//!
//! The orchestrator is the SINGLE WRITER of the directory and the owner of
//! universe time. In P1 the write-ahead ceiling is confirmed by the in-memory
//! stand-in (exactly what a memory store does); the P3 redb wrapper persists
//! `ReserveCeiling` actions before confirming — the clock discipline is identical.

use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::{EpochId, NodeId, UniverseTick};
use vd_sim::directory::{DirectoryCore, DirectoryTuning};
use vd_sim::io::mem::{MemHub, MemSpawner, MemStore};
use vd_sim::io::{Inbound, MsgClass, RealmSpawner, Store};
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

/// RLM 5f-4c: the runtime set of DEMAND-SPAWNED shard nodes that ALSO receive the per-tick `ClockSync`
/// broadcast. Republished from the spawner's live roster each reconcile sweep
/// ([`reconcile_realm_lifecycle`](crate::rlm_runtime::reconcile_realm_lifecycle)). A spawned shard gates ALL
/// its authoring on `has_synced`, so without this it would never sync ⇒ never demand its own children,
/// never detect an occupant leaving, never report ready. EMPTY in the inert default (the reconciler never
/// sweeps) ⇒ the broadcast is byte-identical to the static-only one. Disjoint from [`ClockPeers`] by
/// construction (spawned nodes get fresh F2 ids, never a static-roster id).
#[derive(Resource, Clone, Debug, Default)]
pub struct DynamicClockPeers(pub(crate) std::collections::BTreeSet<NodeId>);

/// Orchestrator-side honesty counters — tolerated anomalies that must never be silent.
#[derive(Resource, Debug, Default, PartialEq, Eq)]
pub struct OrchestratorStats {
    /// Saga-class frames that failed to decode at the directory ingress (a malformed
    /// or garbage envelope). Dropped, never mis-applied, but counted so a decode
    /// regression is observable rather than log-only (ROB-E2E-1; mirrors the gateway
    /// and stub `undecodable`). 0 in any healthy run.
    pub undecodable: u64,
    /// `PeerLocate` asks answered from the launch ledger (the peer book).
    pub peer_locates_answered: u64,
    /// `PeerLocate` asks about a node this orchestrator never launched — unanswered, counted.
    pub peer_locates_unknown: u64,
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
    //
    // RLM Step 3/4a — the default realm spawner is a fresh in-process `MemSpawner` (ids minted from a
    // high base so they never collide with hand-picked test node ids). INERT unless `cfg.rlm` is active;
    // a live-AoI / harness / bootstrap boot passes its OWN spawner (Step 5 supplies the real k8s launcher).
    register_orchestrator_with_store(
        world,
        schedule,
        cfg,
        Box::new(MemStore::new()),
        Box::new(MemSpawner::new(MemHub::new(), NodeId(1_000_000), 8)),
        // Genesis in-process boot: no recovered launches (byte-identical).
        crate::rlm_runtime::LaunchSeed::new(),
    );
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
    spawner: Box<dyn RealmSpawner + Send + Sync>,
    // RLM Step 5e: the crash-recovery launch seed (`SpawnCore::launch_ledger_seed()` projected on the
    // concrete spawner BEFORE boxing) — the shards a rebuilt orchestrator recovered, so the first reconcile
    // sweep does not re-spawn a survivor. EMPTY at genesis / for the in-process `MemSpawner` (byte-identical).
    launch_seed: crate::rlm_runtime::LaunchSeed,
) {
    // RLM Step 4a — `rlm_quiesced_until` carries the crash-recovery FREEZE watermark out of the boot
    // discriminant (RECOVER arms it from the recovered ceiling; GENESIS leaves it `0` = unarmed). It must
    // be captured HERE because the recover-vs-genesis distinction is erased once the match collapses.
    let (clock, directory, mut runtime, rlm_quiesced_until) = match crate::saga_runtime::rehydrate(
        store.as_ref(),
        cfg.reserve_chunk,
        cfg.saga,
        cfg.liveness,
        cfg.directory,
        cfg.rlm,
    ) {
        // RECOVER: the rebuilt orchestrator resumes its durable state (no in-flight transfer vanishes). The
        // realm reconciler's RAM demand ledger is EMPTY, so the freeze holds teardown until demands re-accrue.
        Some(r) => (r.clock, r.directory, r.runtime, r.rlm_quiesced_until),
        // GENESIS: a fresh orchestrator reserves the clock ceiling + starts with an empty directory/saga set.
        // Nothing to freeze — no realm was ever running, so teardown is unarmed (`rlm_quiesced_until = 0`).
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
                UniverseTick(0),
            )
        }
    };
    // D-37: seed the re-home target roster (RAM-only operational config — re-seeded on recover too, like
    // clock_peers; the dead-vs-slow tracker rebuilds empty but its config survives). Single-point wiring.
    runtime.set_roster(cfg.roster.clone());
    world.insert_resource(UniverseClockRes(clock));
    world.insert_resource(DirectoryRes(directory));
    world.insert_resource(ClockPeers(cfg.clock_peers.clone()));
    world.insert_resource(DynamicClockPeers::default());
    // The roster publisher's memory of what it last sent. Inserted HERE, beside every other resource the
    // schedule reads, because a system whose parameter is missing does not degrade — bevy fails the whole
    // tick, so the orchestrator dies on its first one and every login hangs at `AwaitingWelcome` with
    // nothing in any counter to explain why. That is precisely what happened when this line was absent.
    world.insert_resource(PublishedShardRoster::default());
    world.insert_resource(OrchestratorStats::default());
    world.insert_resource(runtime);
    world.insert_resource(crate::saga_runtime::StoreRes(store));
    // RLM Step 3 — the realm-lifecycle reconciler. INERT by default (`cfg.rlm` zero ⇒ the sweep never
    // runs ⇒ byte-identical). The caller INJECTS the spawner (register_orchestrator supplies an in-process
    // MemSpawner; a live-AoI / harness / bootstrap boot supplies one tied to its own hub; Step 5 supplies
    // the real k8s launcher) — so this is the ONE construction site and no downstream rig has to re-insert
    // and clobber the boot-armed quiesce (RLM Step 4a — the E2E harness overwrite is gone).
    // The arrival shield's leak bound is the ONE value that needs BOTH budgets — how long a hand-off can
    // take (the saga deadlines) and how fast a realm can be reclaimed (the lifecycle windows). This is
    // the only place they are both in scope, so it is derived here rather than guessed in either.
    // An INERT reconciler leaves it at zero: a disarmed shield on a reconciler that never sweeps.
    let rlm_tuning = vd_sim::rlm::RlmTuning {
        arrival_shield_ticks: if cfg.rlm.reconcile_interval_ticks == 0 {
            0
        } else {
            vd_sim::rlm::derive_arrival_shield_ticks(&cfg.rlm, &cfg.saga)
        },
        ..cfg.rlm
    };
    let mut reconciler = RlmReconcilerRes::with_launch_seed(rlm_tuning, spawner, launch_seed);
    // RLM Step 4a — arm the crash-recovery freeze on RECOVER (a no-op `arm_quiesce(0)` on GENESIS). Teardown
    // is blocked until `now >= rlm_quiesced_until`, so a rebuilt orchestrator with an empty demand ledger
    // never reaps a still-occupied realm before its parent's `KeepAlive` (or a login demand) re-accrues.
    reconciler.arm_quiesce(rlm_quiesced_until);
    world.insert_resource(reconciler);
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
            serve_peer_locates,
            crate::saga_runtime::drive_sagas_core,
            reconcile_realm_lifecycle,
            publish_shard_roster,
            crate::saga_runtime::commit_barrier,
        )
            .chain(),
    );
}

/// The last roster this orchestrator published, so an unchanged one is not re-sent every tick.
///
/// Holding the sent VALUE rather than a dirty flag is deliberate: the roster is derived from the
/// ownership record by a pure read, so "has it changed" is answerable by comparing the answer, and no
/// mutator anywhere has to remember to raise a flag. A flag is a second thing to keep in step, and
/// keeping two things in step is the defect this whole change exists to remove.
#[derive(Resource, Default)]
struct PublishedShardRoster(Vec<NodeId>);

/// Tell every gateway which nodes the ownership record shows holding a realm — the fact a router needs
/// to decide whose frames it may read at all.
///
/// WHY THE ORCHESTRATOR SAYS IT. A node cannot put itself on this list: it gets here by holding a realm
/// in the record, and that record changes only through the fence commit, which this process performs.
/// Node class is authority, and authority is derived from the record — it was the one authority fact in
/// the system that a router was inferring from somewhere else, and the cost was measured: 14,884 frames
/// from a running planet shard discarded on one cluster, including every position of the player who had
/// just re-homed into it.
///
/// SENT ON CHANGE, AS A WHOLE LEVEL. An unchanged roster sends nothing, so a settled cluster pays
/// nothing per tick; a changed one sends the complete set, so a gateway that missed a message is
/// corrected by the next change rather than left permanently deaf to a live shard. The set is sorted and
/// deduplicated at its source, so "changed" is a byte comparison and not a judgement.
///
/// ADDRESSED FROM THE SAME RECORD. The gateways are the nodes the record shows holding SESSIONS — so
/// this needs no roster of gateways in config to drift out of date, and a gateway that holds no session
/// is told nothing because it is routing nobody.
fn publish_shard_roster(
    clock: Res<ClockSample>,
    dir: Res<DirectoryRes>,
    mut published: ResMut<PublishedShardRoster>,
    mut outbox: ResMut<OutboundBox>,
) {
    let nodes = dir.0.realm_holders();
    if nodes == published.0 {
        return; // nothing moved: a settled cluster costs nothing
    }
    published.0.clone_from(&nodes);
    let flow = InterShardFlow::ShardRoster(vd_wire::intershard::ShardRoster {
        nodes,
        at: clock.universe_tick,
    });
    for gateway in dir.0.session_gateways() {
        outbox.push_flow(gateway, MsgClass::Saga, &flow);
    }
}

/// Advance the authoritative clock one tick and broadcast `ClockSync` to every
/// peer (Membership class — re-derivable, loss-tolerated).
fn advance_and_broadcast_clock(
    mut clock: ResMut<UniverseClockRes>,
    mut sample: ResMut<ClockSample>,
    peers: Res<ClockPeers>,
    dynamic: Res<DynamicClockPeers>,
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
    // RLM 5f-4c: the STATIC peers (in cfg order) THEN the runtime DYNAMIC set (demand-spawned shards,
    // sorted). `dynamic` is empty in the inert default ⇒ byte-identical to the static-only broadcast; the
    // two are disjoint by construction (fresh F2 ids vs the static roster) so the chain never double-sends
    // — and a stray overlap would only cost a harmless duplicate Membership ClockSync (loss-tolerated).
    for peer in peers.0.iter().chain(dynamic.0.iter()) {
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
/// ★ THE PEER BOOK'S ANSWER (D-RLM-6 mechanism C): a node with a frame for a node it has no lane to
/// asks on the Membership class; the answer is the launch ledger's live slot, sent back to the asker
/// on the same class. A node this orchestrator never launched (an anchor, a static shard, a stranger)
/// gets no answer and the ask is counted — a wrong address is worse than none, and a static peer was
/// booked at boot by the same hand that wrote its config.
fn serve_peer_locates(
    inbox: Res<InboundBox>,
    clock: Res<ClockSample>,
    rlm: Res<crate::rlm_runtime::RlmReconcilerRes>,
    mut stats: ResMut<OrchestratorStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            continue;
        };
        if *class != MsgClass::Membership {
            continue;
        }
        let Ok(InterShardFlow::PeerLocate(ask)) = postcard::from_bytes::<InterShardFlow>(bytes)
        else {
            continue; // the Membership class also carries ClockSync echoes and nothing this serves
        };
        let Some((ip, port)) = rlm.addr_of(ask.node) else {
            stats.peer_locates_unknown += 1;
            tracing::warn!(
                asker = ?from,
                node = ?ask.node,
                "PEER LOCATE UNANSWERED: no launch record names that node",
            );
            continue;
        };
        stats.peer_locates_answered += 1;
        outbox.push_flow(
            *from,
            MsgClass::Membership,
            &InterShardFlow::PeerLocated(vd_wire::intershard::PeerLocated {
                node: ask.node,
                ip,
                port,
                at: clock.universe_tick,
            }),
        );
    }
}

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
pub fn admin_snapshot(
    world: &mut bevy_ecs::prelude::World,
    world_generation: u64,
) -> vd_wire::admin::AdminSnapshot {
    let clock = *world.resource::<ClockSample>();
    let mut snapshot = vd_wire::admin::AdminSnapshot::shaped_empty(
        clock.universe_tick,
        clock.epoch,
        world_generation,
    );
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
    // RLM 5f-4: the demand-reconciler observability view — counters + gauges + the measured pod boot. All
    // zero on an inert orchestrator (byte-identical to before). Absent on a non-orchestrator world (the
    // shaped-empty default). `pub` counter fields are read directly; boot latency via its getter.
    if let Some(rlm) = world.get_resource::<crate::rlm_runtime::RlmReconcilerRes>() {
        snapshot.rlm = vd_wire::admin::RlmView {
            spins_requested: rlm.spins_requested,
            spins_failed: rlm.spins_failed,
            teardowns_reaped: rlm.teardowns_reaped,
            force_reaps: rlm.force_reaps,
            retired_heads_reaped: rlm.retired_heads_reaped,
            undecodable_demands: rlm.undecodable_demands,
            demand_sender_mismatch: rlm.demand_sender_mismatch,
            desired_gauge: rlm.desired_gauge,
            running_gauge: rlm.running_gauge,
            boot_ticks_observed_max: rlm.boot_ticks_observed_max(),
            arrival_shield_vetoes: rlm.arrival_shield_vetoes,
            arrival_shield_gauge: rlm.arrival_shield_gauge,
        };
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

    /// A spawner that launched exactly one node, at one address (the peer book's source of truth).
    struct OneLaunched(NodeId);
    impl RealmSpawner for OneLaunched {
        fn spawn_realm(
            &self,
            _coord: &vd_core::realm_coord::RealmCoord,
            _at: UniverseTick,
        ) -> Result<NodeId, vd_sim::io::SpawnError> {
            Err(vd_sim::io::SpawnError::LaunchFailed {
                reason: "a fixture spawner launches nothing".into(),
            })
        }
        fn kill_realm(&self, _node: NodeId) -> Result<(), vd_sim::io::SpawnError> {
            Ok(())
        }
        fn live_nodes(&self) -> std::collections::BTreeSet<NodeId> {
            std::collections::BTreeSet::from([self.0])
        }
        fn addr_of(&self, node: NodeId) -> Option<([u8; 16], u16)> {
            (node == self.0).then_some((
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 0, 0, 5],
                7_600,
            ))
        }
    }

    #[test]
    fn a_peer_locate_is_answered_from_the_launch_ledger_and_a_stranger_gets_no_answer() {
        // THE PEER BOOK: a shard with a frame for node 1007 and no lane asks; the orchestrator answers
        // from its live slot. A node it never launched is unanswered and counted.
        let hub = MemHub::new();
        let mut orch = build_app(
            NodeConfig {
                node_id: ORCH,
                kind: NodeKind::Orchestrator,
            },
            hub.register(ORCH, 64),
        );
        let (world, schedule) = orch.parts_mut();
        register_orchestrator_with_store(
            world,
            schedule,
            &OrchestratorConfig {
                clock_peers: vec![],
                ..cfg()
            },
            Box::new(MemStore::new()),
            Box::new(OneLaunched(NodeId(1_007))),
            crate::rlm_runtime::LaunchSeed::new(),
        );
        let mut asker = hub.register(SHARD, 64);
        let ask = |node: NodeId| {
            vd_sim::io::bytes(
                postcard::to_allocvec(&InterShardFlow::PeerLocate(
                    vd_wire::intershard::PeerLocate {
                        node,
                        at: UniverseTick(3),
                    },
                ))
                .expect("encode"),
            )
        };
        // A fixture guard. This spawner LAUNCHES nothing and KILLS nothing, so the address below can
        // only have come from its launch ledger — never from a realm started during this tick.
        let ledger = OneLaunched(NodeId(1_007));
        assert_eq!(
            ledger.live_nodes(),
            std::collections::BTreeSet::from([NodeId(1_007)])
        );
        assert_eq!(ledger.kill_realm(NodeId(1_007)), Ok(()));
        assert_eq!(
            ledger.spawn_realm(&one_system(), UniverseTick(3)),
            Err(vd_sim::io::SpawnError::LaunchFailed {
                reason: "a fixture spawner launches nothing".into()
            })
        );
        asker
            .send(ORCH, MsgClass::Membership, ask(NodeId(1_007)))
            .expect("sent");
        asker
            .send(ORCH, MsgClass::Membership, ask(NodeId(2_222)))
            .expect("sent");
        // The Membership class carries more than the peer book's asks. A clock sync riding the same
        // class is skipped in silence — it is not a question this serves.
        asker
            .send(
                ORCH,
                MsgClass::Membership,
                vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::ClockSync {
                        universe_tick: UniverseTick(3),
                        epoch: EpochId(7),
                    }))
                    .expect("encode"),
                ),
            )
            .expect("sent");
        hub.pump();
        let _ = orch.step_tick();
        hub.pump();
        let at = orch.world_mut().resource::<ClockSample>().universe_tick;
        // Full-value equality, no destructuring (the codebase convention): ONE answer reaches the
        // asker, for the launched node only — nothing for the stranger, nothing for the clock sync.
        assert_eq!(
            asker.drain_inbound(),
            vec![Inbound::Wire {
                from: ORCH,
                class: MsgClass::Membership,
                bytes: vd_sim::io::bytes(
                    postcard::to_allocvec(&InterShardFlow::PeerLocated(
                        vd_wire::intershard::PeerLocated {
                            node: NodeId(1_007),
                            ip: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 0, 0, 5],
                            port: 7_600,
                            at,
                        }
                    ))
                    .expect("encode"),
                ),
            }]
        );
        let stats = orch.world_mut().resource::<OrchestratorStats>();
        assert_eq!(
            (stats.peer_locates_answered, stats.peer_locates_unknown),
            (1, 1)
        );
    }

    /// A one-level lineage — the shape a spawner is asked to launch.
    fn one_system() -> vd_core::realm_coord::RealmCoord {
        vd_core::realm_coord::RealmCoord::from_path(vd_core::realm_path::RealmPath::from_levels(
            vec![vd_core::realm_path::RealmLevel::new(
                vd_core::realm_path::RealmKindTag::System,
                7,
            )],
        ))
        .expect("a one-level lineage has a leaf")
    }

    #[test]
    fn an_armed_reconciler_gets_a_derived_arrival_shield_an_inert_one_gets_none() {
        // THE ONE PLACE the hand-off budget is computed, and until now only its INERT side ever ran in a
        // test — the branch that derives the real number had never executed. That is the branch a live
        // deployment always takes, so the tested path and the shipped path were opposites.
        //
        // It is derived HERE because this is the only place both inputs are in scope: how long a hand-off
        // may legitimately take (the saga deadlines) and how fast a realm can be reclaimed (the lifecycle
        // windows). Both ends of one hand-off read this same number.
        let armed_rlm = vd_sim::rlm::RlmTuning::cloud(20);
        assert_ne!(
            armed_rlm.reconcile_interval_ticks, 0,
            "fixture guard: a cloud tuning must actually sweep, or this test proves nothing"
        );
        let saga = vd_sim::saga::SagaTuning::default();
        let expected = vd_sim::rlm::derive_arrival_shield_ticks(&armed_rlm, &saga);
        assert_ne!(expected, 0, "the derivation yields a real window, not zero");

        for (rlm, want, why) in [
            (
                armed_rlm,
                expected,
                "an armed reconciler derives the budget",
            ),
            (
                RlmTuning::default(),
                0,
                "an inert one leaves the shield disarmed — nothing sweeps, nothing needs shielding",
            ),
        ] {
            let hub = MemHub::new();
            let mut orch = build_app(
                NodeConfig {
                    node_id: ORCH,
                    kind: NodeKind::Orchestrator,
                },
                hub.register(ORCH, 64),
            );
            let (world, schedule) = orch.parts_mut();
            register_orchestrator(world, schedule, &OrchestratorConfig { rlm, saga, ..cfg() });
            assert_eq!(
                world
                    .resource::<crate::rlm_runtime::RlmReconcilerRes>()
                    .tuning()
                    .arrival_shield_ticks,
                want,
                "{why}"
            );
        }
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

    #[test]
    fn the_shard_roster_publishes_on_change_to_session_gateways_and_settles_silent() {
        // Minor 9's roster lane, at its source: the level ships ONLY when the record's realm-holder
        // set changes, addressed to the nodes the record shows holding SESSIONS. A settled cluster
        // pays nothing per tick; a gateway holding no session is told nothing (it routes nobody).
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
        let mut gateway_t = hub.register(GATEWAY, 64);
        // The record: SHARD holds a realm (the roster's content), GATEWAY holds a session (the
        // roster's address list).
        {
            let mut dir = orch.world_mut().resource_mut::<DirectoryRes>();
            let _ = dir.0.grant(
                DirectoryKey::Realm(RealmId::System(7)),
                AuthorityRef::Shard(SHARD),
                Fence(1),
                UniverseTick(0),
            );
            let _ = dir.0.grant(
                DirectoryKey::Session(SessionId(9)),
                AuthorityRef::Gateway(GATEWAY),
                Fence(1),
                UniverseTick(0),
            );
        }
        // Full-inbound VALUE equality, the clock test's own discipline — no filter, no arm to leave
        // dead. Tick 1 delivers the ClockSync (every peer's) and the changed roster; tick 2, with
        // the record settled, delivers the ClockSync alone (send-on-change measured as absence).
        let sync_at = |tick: u64| Inbound::Wire {
            from: ORCH,
            class: MsgClass::Membership,
            bytes: postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::ClockSync {
                universe_tick: UniverseTick(tick),
                epoch: EpochId(7),
            }))
            .expect("encode")
            .into(),
        };
        let roster_msg = Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: postcard::to_allocvec(&InterShardFlow::ShardRoster(
                vd_wire::intershard::ShardRoster {
                    nodes: vec![SHARD],
                    at: UniverseTick(1),
                },
            ))
            .expect("encode")
            .into(),
        };
        let _ = orch.step_tick();
        hub.pump();
        assert_eq!(
            gateway_t.drain_inbound(),
            vec![sync_at(1), roster_msg],
            "the changed level ships once, beside the tick's ClockSync"
        );
        // SETTLED: the next tick ships no roster (send-on-change).
        let _ = orch.step_tick();
        hub.pump();
        assert_eq!(
            gateway_t.drain_inbound(),
            vec![sync_at(2)],
            "a settled record re-ships nothing"
        );
    }

    #[test]
    fn clock_broadcast_also_reaches_dynamic_spawned_peers() {
        // RLM 5f-4c: a demand-spawned shard (in DynamicClockPeers) receives the SAME ClockSync as the static
        // peers, so it can sync + author. The static peers are unaffected (union, not replacement). The
        // inert reconciler (cfg's default RlmTuning) never republishes, so the seeded set survives the tick.
        const SPAWNED: NodeId = NodeId(1000);
        let hub = MemHub::new();
        let mut orch = build_app(
            NodeConfig {
                node_id: ORCH,
                kind: NodeKind::Orchestrator,
            },
            hub.register(ORCH, 64),
        );
        {
            let (world, schedule) = orch.parts_mut();
            register_orchestrator(world, schedule, &cfg());
            world.insert_resource(DynamicClockPeers([SPAWNED].into_iter().collect()));
        }
        let mut shard_t = hub.register(SHARD, 64);
        let mut gateway_t = hub.register(GATEWAY, 64);
        let mut spawned_t = hub.register(SPAWNED, 64);

        let report = orch.step_tick();
        assert_eq!(
            report.sent, 3,
            "static SHARD + GATEWAY + the dynamic spawned node"
        );
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
        for t in [&mut shard_t, &mut gateway_t, &mut spawned_t] {
            assert_eq!(
                t.drain_inbound(),
                vec![expected_sync.clone()],
                "every peer, static + dynamic, receives ClockSync"
            );
        }
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
        let snap = admin_snapshot(orch.world_mut(), 0);
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
        let snap = admin_snapshot(orch.world_mut(), 0);
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
        assert_eq!(admin_snapshot(&mut bare, 0).directory, vec![]);
        assert_eq!(admin_snapshot(&mut bare, 0).leases, vec![]);
        // RLM 5f-4: no reconciler resource ⇒ the zero RLM view (the get_resource None arm).
        assert_eq!(
            admin_snapshot(&mut bare, 0).rlm,
            vd_wire::admin::RlmView::default(),
            "a world without a reconciler renders the zero RLM view"
        );
    }

    #[test]
    fn admin_snapshot_surfaces_the_live_rlm_counters() {
        use crate::rlm_runtime::RlmReconcilerRes;
        use vd_sim::io::mem::{MemHub, MemSpawner};
        let mut world = bevy_ecs::prelude::World::new();
        world.insert_resource(ClockSample::default());
        // A reconciler with observed activity — set the pub counters directly to prove the snapshot READS
        // them (not merely renders defaults). boot_ticks_observed_max stays 0 (no head came up here).
        let mut rlm = RlmReconcilerRes::new(
            RlmTuning::default(),
            Box::new(MemSpawner::new(MemHub::new(), NodeId(1000), 8)),
        );
        rlm.spins_requested = 4;
        rlm.spins_failed = 1;
        rlm.teardowns_reaped = 2;
        rlm.running_gauge = 3;
        rlm.demand_sender_mismatch = 5;
        world.insert_resource(rlm);
        let snap = admin_snapshot(&mut world, 0);
        assert_eq!(snap.rlm.spins_requested, 4);
        assert_eq!(snap.rlm.spins_failed, 1);
        assert_eq!(snap.rlm.teardowns_reaped, 2);
        assert_eq!(snap.rlm.running_gauge, 3);
        assert_eq!(snap.rlm.demand_sender_mismatch, 5);
        assert_eq!(
            snap.rlm.boot_ticks_observed_max, 0,
            "no head up ⇒ no boot measured"
        );
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
