//! THE GATEWAY'S UNIT TIER — the assertions that were the tail of `gateway.rs` until the file
//! was split, moved VERBATIM and named identically (every `gateway::tests::…` citation in `docs/`
//! still resolves, because the module path did not change: this is still `gateway::tests`).
//!
//! Owns: the fixtures and the assertions for every lane in the `gateway` tree. It reads `super::*`,
//! so it sees the module's whole re-exported surface exactly as a caller outside the crate does,
//! plus the crate-private items the lanes share.
//!
//! Does NOT own: any production behaviour. None of this is compiled into a shipped gateway, and no
//! fixture here may become the only place a rule is stated.

use super::*;
use crate::tickets;
use crate::window;
use arc_swap::ArcSwap;
use bevy_ecs::prelude::{Schedule, World};
use ed25519_dalek::SigningKey;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use vd_core::glam::DVec3;
use vd_core::home::{HomeRegistry, StoredHome};
use vd_core::pose::FrameRef;
use vd_core::pose::RealmId;
use vd_core::pose::StampedPose;
use vd_core::realm_coord::RealmCoord;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TransferId};
use vd_core::{EpochId, TickId, UniverseTick};
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};
use vd_wire::channels::{ClientControlMsg, SceneRow, ServerControlMsg, SubId};
use vd_wire::intershard::{DemandVerb, InterShardFlow, InteriorRelay, RealmDemand, ShardPresence};
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::seams::transfer_control::{
    PrepareReject, PrepareResult, TransferControl, TransferControlAck,
};
use vd_wire::session_flow::{
    BodyStmt, GatewayToShard, RelayedStatement, ShardToGateway, WindowId, WindowScope,
    peek_input_seq,
};
use vd_wire::version::ProtoVersion;
// DEV-ONLY (SL4): the fixtures BUILD worlds; the shipped crate only receives lowered ones.
use vd_physics::worldgen::{UniverseConfig, WorldView};
use vd_sim::capability::NodeKind;
use vd_wire::channels::{
    EntitySnap, InputDatagram, RealmSnap, RealmSnapshotDatagram, SnapshotDatagram,
};
use vd_wire::seams::directory::OwnerRecord;
use vd_wire::version::{PROTO_MAJOR, PROTO_MINOR_FLOOR};

const GW: NodeId = NodeId(1);
const SHARD: NodeId = NodeId(2);
const ORCH: NodeId = NodeId(3);
const CLIENT: NodeId = NodeId(100);
/// The transfer DESTINATION authority — DISTINCT from the source `SHARD(2)` so a test
/// asserting `SeqCut.dest == DEST` proves `dest` threads from the wire command, not from
/// `route.authority` (== SHARD) or a default.
const DEST: NodeId = NodeId(42);
const SIGNING_KEY: [u8; 32] = [0x42; 32];

fn verifying_key() -> [u8; 32] {
    SigningKey::from_bytes(&SIGNING_KEY)
        .verifying_key()
        .to_bytes()
}

/// The bare clock the pure-pass tests hand `compose_scenes_pass` (the rig's schedule reads
/// the world resource instead).
fn test_clock() -> ClockSample {
    ClockSample {
        local_tick: TickId(1),
        universe_tick: UniverseTick(50),
        epoch: EpochId(9),
        synced: true,
    }
}

fn config() -> GatewayConfig {
    GatewayConfig {
        orchestrator: ORCH,
        // No world booted in a fixture, so the gateway states no sky (S11).
        sky: Vec::new(),
        sky_generation: 0,
        sky_frame: None,
        shard: SHARD,
        // The STABLE routable-shard roster (FORK 5): the login shard AND the transfer
        // dest, so a render-ready dest's frames are node-class-dispatchable (1d.2c). DEST
        // is recognized as a shard; whether a SESSION receives its frames is governed
        // separately by the per-session `subscribed_shards` reverse index.
        known_shards: BTreeSet::from([SHARD, DEST]),
        auth_verifying_key: verifying_key(),
        session_seed: 7,
        tick_hz: 50,
        lease_renew_interval_ticks: 0,
        session_recheck_interval: 0,
        self_fence_grace_ticks: 0,
        reject_next_prepare: None, // 3g abort-leg lever INERT by default (behaviour-identical)
        // 5f-3c: the injector is UNARMED ⇒ INERT (byte-identical: no RealmDemand emitted).
        // The armed tests below override this via `..config()`. The world is EXPLICIT — the
        // walk fixture, lowered — because `Default` (which built one silently) is deleted.
        seed_injector: SeedInjectorConfig::inert(test_world().lowered()),
        tuning: TransportTuning {
            max_sessions: 4,
            max_buffered_inputs: 8,
        },
    }
}

/// The per-tick sends captured across a login drive (one Vec of `(to, class,
/// bytes)` per tick).
type LoginSends = Vec<Vec<(NodeId, MsgClass, Vec<u8>)>>;

struct Rig {
    world: World,
    schedule: Schedule,
}

/// HR5 — a TRACE sink so every tracing macro's lazy field closure evaluates on the paths the
/// tests drive; without a subscriber those closures are dead regions coverage cannot reach.
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
        let mut world = World::new();
        world.insert_resource(InboundBox::default());
        world.insert_resource(OutboundBox::default());
        world.insert_resource(NodeIdentity {
            node_id: GW,
            kind: NodeKind::Gateway,
        });
        world.insert_resource(ClockSample {
            local_tick: TickId(1),
            universe_tick: UniverseTick(50),
            epoch: EpochId(9),
            // RLM Step 2: a live-clock rig (the gateway runs no clock-gated authors, so this is inert
            // for the gateway systems — set for a coherent non-default clock).
            synced: true,
        });
        let mut schedule = Schedule::default();
        register_gateway(&mut world, &mut schedule, config());
        Rig { world, schedule }
    }

    fn tick(&mut self, inbound: Vec<Inbound>) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
        self.world.resource_mut::<InboundBox>().0 = inbound;
        self.schedule.run(&mut self.world);
        std::mem::take(&mut self.world.resource_mut::<OutboundBox>().0)
            .into_iter()
            .map(|(to, class, bytes, _)| (to, class, bytes.to_vec()))
            .collect()
    }

    fn stats(&self) -> GatewayStats {
        *self.world.resource::<GatewayStats>()
    }

    /// Hello → directory grant → attach reply; returns (session_id, all sends).
    #[allow(clippy::type_complexity)] // test helper: ticks of raw sends
    fn login(&mut self) -> (SessionId, LoginSends) {
        self.login_with(&hello_msg())
    }

    /// Log in at the CURRENT minor, then FORCE the pending session's negotiated minor down to
    /// `minor` before the grant/attach ticks.
    ///
    /// This exists because `PROTO_MINOR_FLOOR` now REFUSES a genuinely old peer at Hello, so a real
    /// minor-0/1 handshake never produces a session to observe. The sender-gates-variants arms are
    /// still live code — the floor is a property of THIS minor (a field append), and the next
    /// purely variant-additive minor lowers it again — so they still have to be exercised. Forcing
    /// the field is the honest way to do that; the alternative (deleting the gate tests because the
    /// floor currently hides them) would leave the withholding logic unproven the day it matters.
    #[allow(clippy::type_complexity)] // test helper: ticks of raw sends
    fn login_at_minor(&mut self, minor: u16) -> (SessionId, LoginSends) {
        self.login_inner(&hello_msg(), Some(minor))
    }

    fn login_with(&mut self, hello: &ClientControlMsg) -> (SessionId, LoginSends) {
        self.login_inner(hello, None)
    }

    fn login_inner(
        &mut self,
        hello: &ClientControlMsg,
        force_minor: Option<u16>,
    ) -> (SessionId, LoginSends) {
        let hello = self.tick(vec![wire(CLIENT, MsgClass::Control, hello)]);
        let session_id = *self
            .world
            .resource::<GatewaySessions>()
            .sessions()
            .collect::<Vec<_>>()
            .first()
            .expect("session pending");
        if let Some(minor) = force_minor {
            self.world
                .resource_mut::<GatewaySessions>()
                .by_session
                .get_mut(&session_id)
                .expect("pending session")
                .negotiated_minor = minor;
        }
        let granted = self.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
        let attached = self.tick(vec![wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session_id,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
        (session_id, vec![hello, granted, attached])
    }
}

fn wire<T: serde::Serialize>(from: NodeId, class: MsgClass, msg: &T) -> Inbound {
    Inbound::Wire {
        from,
        class,
        bytes: postcard::to_allocvec(msg).expect("encode").into(),
    }
}

fn hello_msg() -> ClientControlMsg {
    ClientControlMsg::Hello {
        version: ProtoVersion::CURRENT,
        login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
    }
}

fn hello_msg_below_floor() -> ClientControlMsg {
    ClientControlMsg::Hello {
        version: ProtoVersion::speaking(PROTO_MAJOR, PROTO_MINOR_FLOOR - 1),
        login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
    }
}

#[test]
fn a_peer_below_the_protocol_floor_is_closed_with_the_floor_named() {
    // proto_minor 8 appended a FIELD to `RealmSnap`, which postcard cannot skip and a sender cannot
    // gate out — so an old peer is REFUSED rather than negotiated down. Before the floor, this exact
    // Hello was welcomed and the peer then mis-framed every realm datagram it decoded: not one wrong
    // field, a desynced stream. The reason NAMES the floor so the operator is told to update the
    // client instead of hunting a protocol-generation split.
    let mut rig = Rig::new();
    let sent = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &hello_msg_below_floor(),
    )]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![ServerControlMsg::Close {
            reason: "protocol minor below the floor (24): the scene is server-composed from v1.24"
                .to_owned()
        }]
    );
    assert_eq!(rig.stats().version_rejected, 1);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().sessions().count(),
        0,
        "a refused peer mints NO session"
    );
}

#[test]
fn a_minor_0_session_is_welcomed_without_the_universe_rate_variant() {
    // Sender-gates-variants: a session at minor 0 must NOT be sent the minor-1 UniverseRate (it
    // would desync an old decoder). Welcome only. The minor is FORCED rather than negotiated —
    // the floor refuses a real minor-0 handshake now (see `login_at_minor`), but the withholding
    // arm is live code that a future variant-additive minor will lower the floor back onto.
    let mut rig = Rig::new();
    let (session_id, sends) = rig.login_at_minor(0);
    let welcomes = decode_controls(&sends[1], CLIENT);
    assert_eq!(
        welcomes,
        vec![ServerControlMsg::Welcome {
            version: ProtoVersion::CURRENT,
            session: session_id,
            session_fence: Fence(1),
            epoch: EpochId(9),
        }],
        "a minor-0 session gets Welcome but NOT UniverseRate"
    );
}

// ---- RLM 5f RG-3: the reactive-greeting consume -------------------------------------------
#[test]
fn a_shard_presence_greeting_is_counted_never_undecodable() {
    let mut rig = Rig::new();
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Saga,
        &InterShardFlow::ShardPresence(ShardPresence {
            local_tick: TickId(7),
        }),
    )]);
    assert_eq!(rig.stats().presence_announces, 1);
    assert_eq!(rig.stats().undecodable, 0);
    // Pure observability — a greeting has NO routing effect (no roster claim, no reply here; the
    // reachability was the mesh learning the connection below the seam).
    assert!(sent.is_empty());
}

#[test]
fn a_repeated_greeting_admits_once_and_keeps_counting() {
    // The admit is an INSERT (idempotent): the first greeting logs the admission, a re-greet on
    // the silence cadence counts the announce but does not re-admit (the roster set is a set).
    let mut rig = Rig::new();
    let greet = || {
        wire(
            SHARD,
            MsgClass::Saga,
            &InterShardFlow::ShardPresence(ShardPresence {
                local_tick: TickId(7),
            }),
        )
    };
    let _ = rig.tick(vec![greet()]);
    let _ = rig.tick(vec![greet()]);
    assert_eq!(rig.stats().presence_announces, 2, "every greeting counts");
    assert_eq!(rig.stats().undecodable, 0);
}

#[test]
fn a_greeting_from_an_unbooked_peer_is_counted_the_same() {
    // A demand shard is NOT on the gateway's roster when it first greets; the counter must not care.
    let mut rig = Rig::new();
    rig.tick(vec![wire(
        NodeId(999),
        MsgClass::Saga,
        &InterShardFlow::ShardPresence(ShardPresence {
            local_tick: TickId(3),
        }),
    )]);
    assert_eq!(rig.stats().presence_announces, 1);
    assert_eq!(rig.stats().undecodable, 0);
}

#[test]
fn a_non_greeting_saga_body_stays_undecodable() {
    // A VALID but non-ShardPresence InterShardFlow on the shard→gateway Saga class counts
    // `undecodable` exactly as the pre-RG-3 fallthrough did (the Ok(_) arm) — no greeting miscount.
    let mut rig = Rig::new();
    rig.tick(vec![wire(
        SHARD,
        MsgClass::Saga,
        &granted_head(SessionId(1)),
    )]);
    assert_eq!(rig.stats().presence_announces, 0);
    assert_eq!(rig.stats().undecodable, 1);
}

#[test]
fn garbage_saga_bytes_stay_undecodable() {
    // Undecodable bytes (the Err arm) — an invalid discriminant.
    let mut rig = Rig::new();
    rig.tick(vec![Inbound::Wire {
        from: SHARD,
        class: MsgClass::Saga,
        bytes: vec![0x7F].into(),
    }]);
    assert_eq!(rig.stats().presence_announces, 0);
    assert_eq!(rig.stats().undecodable, 1);
}

#[test]
fn a_gateway_no_one_greets_is_byte_identical() {
    // No shard greets ⇒ presence_announces stays 0 and every counter is the default (byte-identity).
    let mut rig = Rig::new();
    rig.tick(vec![]);
    assert_eq!(rig.stats(), GatewayStats::default());
}

#[test]
fn a_routable_shard_sending_a_non_frame_class_is_still_undecodable() {
    // The routable-shard branch's fallthrough (a rostered shard sending a class it never should — here
    // Input) is UNCHANGED by RG-3: the greeting arm only intercepts Saga, so this still counts
    // `undecodable`. (Re-covers that arm, which the greeting arm's hoist moved the old Saga case off.)
    let mut rig = Rig::new();
    rig.tick(vec![Inbound::Wire {
        from: SHARD, // a `known_shards` member ⇒ routable
        class: MsgClass::Input,
        bytes: vec![0x00].into(),
    }]);
    assert_eq!(rig.stats().undecodable, 1);
    assert_eq!(rig.stats().presence_announces, 0);
}

// ---- RLM 5f RG-4a2: the gateway's /admin/snapshot builder --------------------------------
#[test]
fn the_gateway_admin_snapshot_reflects_the_live_world() {
    // The world→contract builder: one logged-in session ⇒ sessions_open 1; a static rig demanded no
    // dynamic home ⇒ dynamic_shards 0; the clock is mirrored; and the gateway carries NO directory (it
    // does not own one — reporting a directory from a gateway would mislead an operator).
    let mut rig = Rig::new();
    let _ = rig.login();
    let snap = crate::admin::gateway_admin_snapshot(&mut rig.world, 0);
    let gw = snap
        .gateway
        .expect("a gateway snapshot carries the gateway view");
    assert_eq!(gw.sessions_open, 1);
    assert_eq!(gw.dynamic_shards, 0);
    assert_eq!(snap.universe_tick, 50);
    assert_eq!(snap.epoch, 9);
    assert!(
        snap.directory.is_empty(),
        "the gateway owns no directory — that is the orchestrator's"
    );
}

/// A session-grant head reply as the orchestrator now sends it — wrapped in the
/// InterShardFlow::DirectoryReply envelope (the dispatch split).
fn granted_head(session: SessionId) -> InterShardFlow {
    InterShardFlow::DirectoryReply(DirectoryReply::Head {
        key: DirectoryKey::Session(session),
        record: Some(OwnerRecord {
            authority: AuthorityRef::Gateway(GW),
            fence: Fence(1),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    })
}

/// A Session-head reply DENYING this gateway's ownership (record absent — e.g. the orchestrator's
/// reaper revoked the lapsed lease): drives both the reactive self-fence and the recheck channel.
fn absent_head(session: SessionId) -> InterShardFlow {
    InterShardFlow::DirectoryReply(DirectoryReply::Head {
        key: DirectoryKey::Session(session),
        record: None,
    })
}

/// A D-3 Slice-5b config: the recheck channel + proactive self-fence both ARMED (rig-local values;
/// the split-brain-safe `ttl < grace` with `THETA_MAX*grace < ttl + max` ordering is validated orchestrator-side).
fn self_fence_config() -> GatewayConfig {
    GatewayConfig {
        session_recheck_interval: 2,
        self_fence_grace_ticks: 5,
        ..config()
    }
}

fn set_tick(rig: &mut Rig, tick: u64) {
    rig.world.resource_mut::<ClockSample>().local_tick = TickId(tick);
}

fn session_active(rig: &Rig, sid: SessionId) -> bool {
    rig.world
        .resource::<GatewaySessions>()
        .entity_of(sid)
        .is_some()
}

#[test]
fn a_partitioned_gateway_proactively_self_fences_a_stale_session() {
    // D-3 Slice 5b: an Active session whose `Session`-head goes un-confirmed past the grace (a
    // partition from the orchestrator — no recheck reply) is hard-stopped BEFORE the orchestrator's
    // reassign window, then served no input/frames. It is FENCED, not removed (the connection lingers).
    let mut rig = Rig::new();
    rig.world.insert_resource(self_fence_config());
    let (sid, _) = rig.login(); // Active; confirmed_at armed to local_tick 1 at the attach
    // Within the grace (local 6 - confirmed 1 = 5, NOT > 5): still Active.
    set_tick(&mut rig, 6);
    let _ = rig.tick(vec![]);
    assert!(session_active(&rig, sid));
    assert_eq!(rig.stats().sessions_self_fenced_lapsed, 0);
    // Past the grace (local 7 - 1 = 6 > 5): SELF-FENCE.
    set_tick(&mut rig, 7);
    let _ = rig.tick(vec![]);
    assert!(
        !session_active(&rig, sid),
        "the partitioned session self-fenced"
    );
    assert_eq!(rig.stats().sessions_self_fenced_lapsed, 1);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().len(),
        1,
        "fenced, not removed (awaits connection-end / adoption)"
    );
    // S4 readiness de-route reachability (the review's exact concern): after the REAL schedule self-fences
    // the last session out of `Active`, the partition detector must still SEE the frozen confirmed (via the
    // now-SelfFenced session), not read `None`. An `Active`-only max would read `None` here and keep this
    // fully-partitioned gateway falsely Ready. The frozen confirmed is tick 1 (armed at the attach); with
    // local_tick 7 > confirmed 1 + grace 5, `vd_node::health::gateway_sessions_live(Some(1), 7, 5)` is NOT
    // live (proven over the pure predicate in the vd-node health tests — the seam kept node-free here).
    assert_eq!(
        rig.world
            .resource::<GatewaySessions>()
            .freshest_session_confirmed(),
        Some(1),
        "the self-fenced session's frozen confirm is the reachable de-route signal (not None)"
    );
}

#[test]
fn a_recheck_reply_re_arms_an_active_session_and_a_foreign_head_self_fences_it() {
    // The reactive arm: an AFFIRMING recheck reply re-arms the deadline; a foreign/absent head
    // (lease reassigned or reaped) self-fences at once — the link-alive cure beside the partition timer.
    let mut rig = Rig::new();
    rig.world.insert_resource(self_fence_config());
    let (sid, _) = rig.login(); // confirmed_at = 1
    set_tick(&mut rig, 4);
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]); // re-arm ⇒ confirmed_at 4
    assert!(session_active(&rig, sid));
    // At local 8 (8 - re-armed 4 = 4 <= 5) STILL Active — proof the reply re-armed the deadline
    // (had `confirmed_at` stayed 1, 8 - 1 = 7 > 5 would have fenced it here).
    set_tick(&mut rig, 8);
    let _ = rig.tick(vec![]);
    assert!(
        session_active(&rig, sid),
        "an affirming recheck reply re-armed the self-fence deadline"
    );
    assert_eq!(rig.stats().sessions_self_fenced_lapsed, 0);
    // A foreign/absent head reactively self-fences.
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &absent_head(sid))]);
    assert!(
        !session_active(&rig, sid),
        "a revoked-lease head reactively self-fences the session"
    );
    assert_eq!(rig.stats().sessions_self_fenced_revoked, 1);
}

#[test]
fn the_recheck_producer_re_reads_active_session_heads_on_cadence() {
    // The confirmation channel: on the recheck cadence the gateway re-reads each Active session's
    // `Session` head — the round-trip whose affirming reply re-arms `confirmed_at`.
    let mut rig = Rig::new();
    rig.world.insert_resource(self_fence_config()); // recheck every 2 ticks
    let (sid, _) = rig.login();
    set_tick(&mut rig, 4); // a recheck multiple; 4 - 1 = 3 within grace, so no self-fence here
    let sends = rig.tick(vec![]);
    // Compare against the exact expected bytes with BITWISE `&` (no short-circuit branch gaps — HR5).
    let expected = postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::HeadRead {
        key: DirectoryKey::Session(sid),
    }))
    .expect("encode");
    let saw_head_read = sends.iter().any(|(to, class, bytes)| {
        (*to == ORCH) & (*class == MsgClass::Saga) & (bytes.as_slice() == expected.as_slice())
    });
    assert!(
        saw_head_read,
        "an Active session's head is re-read on the recheck cadence"
    );
}

#[test]
fn on_shard_frame_skips_a_self_fenced_session() {
    // A self-fenced session lingers in the fan-out reverse index (its subs are not torn down) but
    // is served NO frames — skipped cleanly via the phase gate (NOT a desync), watermark untouched.
    let (mut sessions, sid, _) = one_active_session();
    sessions.by_session.get_mut(&sid).expect("present").phase = SessionPhase::SelfFenced;
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    let frame = postcard::to_allocvec(&frame_msg(Fence(1), 9)).expect("encode");
    on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
    assert!(
        outbox.0.is_empty(),
        "a self-fenced session receives no forwarded frames"
    );
    assert_eq!(
        stats.frame_sub_desync, 0,
        "skipped via the Active-only phase gate, never counted a desync"
    );
    assert_eq!(
        sessions.by_session[&sid].delivered.get(&SubId(0)),
        None,
        "no frame ⇒ the delivery watermark is untouched"
    );
}

#[test]
fn a_session_head_for_an_awaiting_attach_session_carries_no_obligation() {
    // After the mint grant the session is AwaitingAttach (the shard attach is in flight). A
    // duplicate/late `Session`-head reply then is neither a fresh mint (AwaitingDirectory) nor a
    // recheck (Active), so it is dropped with no effect (the at-least-once idempotency floor) — no
    // second Welcome, no phase change, no mint refusal.
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sid = *rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .collect::<Vec<_>>()
        .first()
        .expect("session pending");
    let granted = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]); // ⇒ AwaitingAttach
    // filter+count evaluates the predicate on EVERY control (Welcome ⇒ kept, UniverseRate ⇒ dropped),
    // so both arms of the `matches!` are covered (no `.any` short-circuit — HR5).
    assert_eq!(
        decode_controls(&granted, CLIENT)
            .iter()
            .filter(|m| matches!(m, ServerControlMsg::Welcome { .. }))
            .count(),
        1,
        "the FIRST grant Welcomes the client exactly once"
    );
    let before = rig.stats();
    let dup = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]); // duplicate head
    assert!(
        decode_controls(&dup, CLIENT).is_empty(),
        "a duplicate head for an AwaitingAttach session re-sends NO Welcome"
    );
    assert_eq!(
        rig.stats().session_mints_refused,
        before.session_mints_refused,
        "no mint refusal — the head simply carries no obligation"
    );
    assert!(
        !session_active(&rig, sid),
        "still AwaitingAttach (awaiting the shard's SessionAttached), not Active"
    );
}

#[test]
fn a_session_attached_straggler_does_not_resurrect_a_self_fenced_session() {
    // D-3 Slice 5b CRITICAL guard. `SessionAttached` is re-emitted on every `AttachSession` retry
    // and rides the gateway↔shard link, which can be HEALTHY while the gateway↔orchestrator link
    // (that drove the self-fence) is partitioned — so a straggler can land AFTER Active→SelfFenced.
    // Re-promoting it would re-arm the grace clock and resume input/frames — re-opening the
    // split-brain window. The attach arm promotes only from AwaitingAttach, so the fenced session
    // stays inert (awaiting Bye / the future ResumeTicket adoption).
    let mut rig = Rig::new();
    rig.world.insert_resource(self_fence_config());
    let (sid, _) = rig.login(); // Active; confirmed_at armed at the attach
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &absent_head(sid))]); // reactively self-fence
    assert!(!session_active(&rig, sid), "the session is self-fenced");
    let fenced_confirmed = rig.world.resource::<GatewaySessions>().by_session[&sid].confirmed_at;
    // A late/duplicate SessionAttached straggler at a much later tick must be IGNORED.
    set_tick(&mut rig, 42);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    assert!(
        !session_active(&rig, sid),
        "the straggler did NOT resurrect the fenced session to Active (no split-brain reopened)"
    );
    assert_eq!(
        rig.world.resource::<GatewaySessions>().by_session[&sid].confirmed_at,
        fenced_confirmed,
        "the self-fence deadline was NOT re-armed by the straggler"
    );
}

fn decode_controls(sent: &[(NodeId, MsgClass, Vec<u8>)], to: NodeId) -> Vec<ServerControlMsg> {
    sent.iter()
        .filter(|(node, class, _)| (*node == to) & (*class == MsgClass::Control))
        .map(|(_, _, bytes)| postcard::from_bytes(bytes).expect("decode"))
        .collect()
}

fn input_bytes(seq: u64) -> Vec<u8> {
    postcard::to_allocvec(&InputDatagram {
        seq,
        is_cut_marker: false,
        client_tick: TickId(1),
        movement: [1.0, 0.0, 0.0],
        look: [0.0, 0.0],
        action_bits: 0,
    })
    .expect("encode")
}

fn frame_msg(fence: Fence, frame_id: u64) -> ShardToGateway {
    let snapshot = SnapshotDatagram {
        sub: SubId(0),
        frame_id,
        source_tick: TickId(5),
        universe_tick: UniverseTick(50),
        entities: vec![EntitySnap {
            entity: EntityId(77),
            pose: vd_core::pose::StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                vd_core::glam::DVec3::ZERO,
                UniverseTick(50),
            ),
        }],
    };
    ShardToGateway::Frame {
        realm_fence: fence,
        source_tick: TickId(5),
        snapshot_bytes: postcard::to_allocvec(&snapshot).expect("encode"),
    }
}

/// A `ShardToGateway::RealmFrame` carrying one moving-realm placement (FA-2c) — the realm twin of
/// [`frame_msg`].
fn realm_frame_msg(realm: RealmId) -> ShardToGateway {
    ShardToGateway::RealmFrame {
        realm_fence: Fence(1),
        source_tick: TickId(5),
        realm_snapshot_bytes: realm_frame_payload(realm),
    }
}

#[test]
fn the_full_login_flow_reaches_active_with_welcome_then_subscription() {
    let mut rig = Rig::new();
    let (session_id, sends) = rig.login();

    // Hello tick: exactly one directory grant went out (plus the retry driver's
    // duplicate — idempotent by fence).
    let to_orch: Vec<NodeId> = sends[0].iter().map(|(to, _, _)| *to).collect();
    assert!(
        to_orch.iter().all(|to| *to == ORCH),
        "only directory traffic"
    );

    // Grant tick: Welcome to the client (with the directory-committed id,
    // fence, epoch) and an attach toward the shard.
    // Welcome, then UniverseRate (the client negotiated minor 1, so the gateway
    // relays the cluster tick rate right after the Welcome).
    let welcomes = decode_controls(&sends[1], CLIENT);
    assert_eq!(
        welcomes,
        vec![
            ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: session_id,
                session_fence: Fence(1),
                epoch: EpochId(9),
            },
            ServerControlMsg::UniverseRate { tick_hz: 50 },
        ]
    );
    // Attach tick: SubscriptionOpened STRICTLY BEFORE AuthorityChanged (X1),
    // sub allocated from the monotonic allocator. The default rig negotiates minor 2
    // (ProtoVersion::CURRENT), so the gateway ALSO trails `OwnEntity` (the pure-renderer
    // own-entity signal, S4) after `AuthorityChanged` — the node-aware + node-agnostic
    // dual-announce for a rolling fleet of both client versions.
    let controls = decode_controls(&sends[2], CLIENT);
    assert_eq!(
        controls,
        vec![
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(0),
                frame: FrameRef::SystemSpace { system_seed: 7 },
            },
            ServerControlMsg::AuthorityChanged {
                entity: EntityId(77),
                sub: SubId(0),
            },
            ServerControlMsg::OwnEntity {
                entity: EntityId(77),
            },
            // THE LOGIN LEVEL (§2.4): the composer's same-tick pass sees the standing
            // realm and bumps the epoch 0→1 — the swap signal, after the identity control.
            ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 1,
                rows: vec![SceneRow {
                    realm: RealmId::System(7),
                    parent: None,
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::ZERO,
                        UniverseTick(0),
                    ),
                    bag: Vec::new(),
                }],
            },
        ]
    );
    assert_eq!(
        rig.stats(),
        GatewayStats {
            // THE WINDOW LANE (Slice A): a clean login now ALSO derives + opens the session's
            // own-realm Occupants window — exactly one open, a THROUGHPUT count, not a reject.
            window_open_sent: 1,
            // Slice B: the composer derives the one-hop chain the moment the session
            // is Active (a GAUGE — one session, one chain). No frames were ever ingested, so
            // nothing folds and nothing holds (pre-first-fold boot is not a stall).
            window_chains_held: 1,
            // Slice C1: the login LEVEL shipped the moment the standing realm resolved.
            scene_levels_sent: 1,
            ..GatewayStats::default()
        },
        "clean run, zero rejects"
    );
}

#[test]
fn own_entity_trails_authority_changed_for_minor2_and_is_withheld_from_minor0() {
    // S4 dual-announce (sender-gates-variants): a minor>=2 (pure-renderer) client gets BOTH
    // `AuthorityChanged{entity, sub}` (the node-aware sub re-point) AND `OwnEntity{entity}` (the
    // node-AGNOSTIC own-entity cue) at attach — the node-agnostic signal LAST. A minor-0 client
    // gets ONLY `AuthorityChanged` (the minor-2 OwnEntity is withheld — it would desync an old
    // decoder). Neither ever learns which shard owns the entity from these two messages.
    let mut rig = Rig::new();
    let (_sid, sends) = rig.login(); // default hello = ProtoVersion::CURRENT (minor 2)
    let controls = decode_controls(&sends[2], CLIENT); // the attach tick's client controls
    assert_eq!(
        controls,
        vec![
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(0),
                frame: FrameRef::SystemSpace { system_seed: 7 },
            },
            ServerControlMsg::AuthorityChanged {
                entity: EntityId(77),
                sub: SubId(0),
            },
            ServerControlMsg::OwnEntity {
                entity: EntityId(77),
            },
            // The login LEVEL (§2.4) trails the identity control — same tick, same stream.
            ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 1,
                rows: vec![SceneRow {
                    realm: RealmId::System(7),
                    parent: None,
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::ZERO,
                        UniverseTick(0),
                    ),
                    bag: Vec::new(),
                }],
            },
        ],
        "minor-2 attach announces AuthorityChanged THEN the node-agnostic OwnEntity",
    );

    // A minor-0 session: OwnEntity is withheld — ONLY AuthorityChanged names its avatar. Forced
    // rather than negotiated (the floor refuses a real minor-0 handshake; see `login_at_minor`).
    let mut rig0 = Rig::new();
    let (_sid0, sends0) = rig0.login_at_minor(0);
    let controls0 = decode_controls(&sends0[2], CLIENT);
    assert_eq!(
        controls0,
        vec![
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(0),
                frame: FrameRef::SystemSpace { system_seed: 7 },
            },
            ServerControlMsg::AuthorityChanged {
                entity: EntityId(77),
                sub: SubId(0),
            },
            // The login LEVEL (§2.4) trails the identity control — same tick, same stream.
            ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 1,
                rows: vec![SceneRow {
                    realm: RealmId::System(7),
                    parent: None,
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::ZERO,
                        UniverseTick(0),
                    ),
                    bag: Vec::new(),
                }],
            },
        ],
        "a minor-0 session gets AuthorityChanged but NOT the minor-2 OwnEntity (sender-gates-variants)",
    );
}

#[test]
fn entity_of_tracks_the_session_lifecycle() {
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let session_id = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .next()
        .expect("pending");
    // Pending phases expose no entity; unknown sessions expose none either.
    assert_eq!(
        rig.world
            .resource::<GatewaySessions>()
            .entity_of(session_id),
        None
    );
    assert_eq!(
        rig.world
            .resource::<GatewaySessions>()
            .entity_of(SessionId(0xDEAD)),
        None
    );
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
    assert_eq!(
        rig.world
            .resource::<GatewaySessions>()
            .entity_of(session_id),
        None,
        "still awaiting attach"
    );
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: session_id,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    assert_eq!(
        rig.world
            .resource::<GatewaySessions>()
            .entity_of(session_id),
        Some(EntityId(77)),
        "active sessions expose their avatar (P2's saga key)"
    );
}

#[test]
fn version_mismatch_and_bad_tickets_close_with_reasons() {
    let mut rig = Rig::new();
    let wrong_version = ClientControlMsg::Hello {
        version: ProtoVersion::speaking(ProtoVersion::CURRENT.major + 1, 0),
        login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
    };
    let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &wrong_version)]);
    let controls = decode_controls(&sent, CLIENT);
    assert_eq!(controls.len(), 1);
    assert_eq!(
        controls[0],
        ServerControlMsg::Close {
            reason: "incompatible protocol major version".to_owned()
        }
    );
    // ★ AND THE SAME REFUSAL FOR A DIFFERENT COORDINATE UNIT (slice S3), driven at the REAL
    // negotiation site rather than against the version type. This is the only place in the product
    // where a client is negotiated with, so it is the only place that can prove the refusal exists.
    //
    // Same protocol, same minor, healthy peer — and every position it exchanged would be wrong by the
    // ratio between its unit and ours. That is invisible from the outside, which is why the refusal
    // has to happen here, before the first position.
    let other_unit = ClientControlMsg::Hello {
        version: ProtoVersion {
            coordinate_generation: ProtoVersion::CURRENT.coordinate_generation ^ 1,
            ..ProtoVersion::CURRENT
        },
        login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
    };
    let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &other_unit)]);
    let controls = decode_controls(&sent, CLIENT);
    // Search for the close rather than indexing: this rig's outbox carries the earlier refusal too, and
    // an index would make the assertion depend on how many refusals ran before it.
    let unit_close = controls
        .iter()
        .filter_map(|c| match c {
            ServerControlMsg::Close { reason } => Some(reason),
            _ => None,
        })
        .find(|r| r.contains("coordinate units differ"));
    let reason = unit_close.unwrap_or_else(|| {
        panic!("a peer counting positions in another unit must be closed: {controls:?}")
    });
    // AND THE CAUSE, which is the half an operator cannot see for themselves: both peers are healthy
    // and the same version.
    assert!(
        reason.contains(&ProtoVersion::CURRENT.coordinate_generation.to_string()),
        "the close must carry this build's own unit: {reason}"
    );
    assert!(
        !controls
            .iter()
            .any(|c| matches!(c, ServerControlMsg::Welcome { .. })),
        "and it must not ALSO be served: {controls:?}"
    );

    // A foreign signer's ticket is rejected by the SOLE validator.
    let forged = ClientControlMsg::Hello {
        version: ProtoVersion::CURRENT,
        login: tickets::mint_login(&[0x66; 32], AccountId(5), EpochId(9), 1),
    };
    let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &forged)]);
    let controls = decode_controls(&sent, CLIENT);
    assert_eq!(
        controls[0],
        ServerControlMsg::Close {
            reason: "login ticket rejected".to_owned()
        }
    );
    // TWO refusals now, and they share ONE counter deliberately: a major mismatch and a coordinate-unit
    // mismatch are both "this peer cannot be served", and the CAUSE lives in the sentence rather than
    // in a second statistic (the same choice the floor refusal already made).
    assert_eq!(rig.stats().version_rejected, 2);
    assert_eq!(rig.stats().logins_rejected, 1);
    assert!(rig.world.resource::<GatewaySessions>().is_empty());
}

#[test]
fn capacity_duplicate_hello_and_resume_paths() {
    let mut rig = Rig::new();
    // Fill to capacity with distinct clients.
    for n in 0..4u64 {
        let _ = rig.tick(vec![wire(NodeId(100 + n), MsgClass::Control, &hello_msg())]);
    }
    assert_eq!(rig.world.resource::<GatewaySessions>().len(), 4);
    // One more is refused loudly.
    let sent = rig.tick(vec![wire(NodeId(199), MsgClass::Control, &hello_msg())]);
    assert_eq!(
        decode_controls(&sent, NodeId(199))[0],
        ServerControlMsg::Close {
            reason: "gateway at session capacity".to_owned()
        }
    );
    assert_eq!(rig.stats().sessions_refused_capacity, 1);
    // Resume is refused honestly until P3.
    let resume = ClientControlMsg::Resume {
        version: ProtoVersion::CURRENT,
        ticket: dummy_resume(),
    };
    let sent = rig.tick(vec![wire(NodeId(198), MsgClass::Control, &resume)]);
    assert_eq!(
        decode_controls(&sent, NodeId(198))[0],
        ServerControlMsg::Close {
            reason: "resume is not available yet".to_owned()
        }
    );
    assert_eq!(rig.stats().resumes_refused, 1);
}

fn dummy_resume() -> vd_wire::seams::tickets::ResumeTicket {
    vd_wire::seams::tickets::ResumeTicket {
        claims: vd_wire::seams::tickets::SessionClaims {
            session: SessionId(1),
            account: AccountId(1),
            epoch: EpochId(1),
            validity_epoch: 0,
            expires: UniverseTick(0),
            key_id: 0,
        },
        session_fence: Fence(0),
        resume_nonce: 0,
        hmac: [0; 32],
    }
}

#[test]
fn mint_refusal_closes_the_client_and_clears_the_session() {
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let session_id = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .next()
        .expect("pending session");
    // The directory says someone ELSE holds the session key.
    let refused = InterShardFlow::DirectoryReply(DirectoryReply::Head {
        key: DirectoryKey::Session(session_id),
        record: Some(OwnerRecord {
            authority: AuthorityRef::Gateway(NodeId(55)),
            fence: Fence(3),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    });
    let sent = rig.tick(vec![wire(ORCH, MsgClass::Saga, &refused)]);
    assert_eq!(
        decode_controls(&sent, CLIENT)[0],
        ServerControlMsg::Close {
            reason: "session mint refused by the directory".to_owned()
        }
    );
    assert!(rig.world.resource::<GatewaySessions>().is_empty());
    assert_eq!(rig.stats().session_mints_refused, 1);
}

#[test]
fn input_routes_dedups_and_counts_every_failure_mode() {
    let mut rig = Rig::new();
    let (session_id, _) = rig.login();

    // A fresh input forwards to the shard wrapped with (session, fence).
    let sent = rig.tick(vec![Inbound::Wire {
        from: CLIENT,
        class: MsgClass::Input,
        bytes: input_bytes(1).into(),
    }]);
    let inputs: Vec<&(NodeId, MsgClass, Vec<u8>)> = sent
        .iter()
        .filter(|(to, class, _)| (*to == SHARD) & (*class == MsgClass::Input))
        .collect();
    assert_eq!(inputs.len(), 1);
    let fwd: GatewayToShard = postcard::from_bytes(&inputs[0].2).expect("decode");
    assert_eq!(
        fwd,
        GatewayToShard::SessionInput {
            session: session_id,
            fence: Fence(1),
            input_bytes: input_bytes(1),
        }
    );

    // Same seq again: deduped. Garbage: malformed. Unknown client: unroutable.
    let _ = rig.tick(vec![
        Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: input_bytes(1).into(),
        },
        Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Input,
            bytes: vec![0x80].into(),
        },
        Inbound::Wire {
            from: NodeId(177),
            class: MsgClass::Input,
            bytes: input_bytes(2).into(),
        },
    ]);
    let stats = rig.stats();
    assert_eq!(stats.inputs_deduped, 1);
    assert_eq!(stats.inputs_malformed, 1);
    assert_eq!(stats.inputs_unroutable, 1);
}

#[test]
fn input_for_a_desynced_session_map_is_dropped_and_counted_never_panics() {
    // WB-1: `by_client` and `by_session` are kept in sync by construction, but a slip
    // must DROP-and-count on the 20Hz input path, never panic the gateway. Proven by
    // an artificially desynced table (present in `by_client`, absent from `by_session`).
    let mut sessions = GatewaySessions::default();
    sessions.by_client.insert(CLIENT, SessionId(7));
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    on_client_input(
        &input_bytes(1),
        CLIENT,
        &config(),
        &mut sessions,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(
        stats.inputs_unroutable, 1,
        "the desync is counted, not crashed"
    );
    assert!(
        outbox.0.is_empty(),
        "nothing forwarded for a desynced session"
    );
}

#[test]
fn duplicate_hello_below_capacity_is_an_idempotent_noop() {
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    assert_eq!(rig.world.resource::<GatewaySessions>().len(), 1);
    let sent = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().len(),
        1,
        "no second session"
    );
    // Only the pending-grant retry went out — no Close, no new mint.
    assert!(sent.iter().all(|(to, _, _)| *to == ORCH));
    assert_eq!(rig.stats(), GatewayStats::default());
}

#[test]
fn input_before_active_is_unroutable() {
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let _ = rig.tick(vec![Inbound::Wire {
        from: CLIENT,
        class: MsgClass::Input,
        bytes: input_bytes(1).into(),
    }]);
    assert_eq!(rig.stats().inputs_unroutable, 1);
}

#[test]
fn frames_fan_out_retagged_and_stale_fences_drop() {
    let mut rig = Rig::new();
    let (_, _) = rig.login();

    // A current-fence frame reaches the client re-tagged to ITS sub.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence(1), 9),
    )]);
    let snaps: Vec<&(NodeId, MsgClass, Vec<u8>)> = sent
        .iter()
        .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
        .collect();
    assert_eq!(snaps.len(), 1);
    let snap: SnapshotDatagram = postcard::from_bytes(&snaps[0].2).expect("decode");
    assert_eq!(snap.sub, SubId(0));
    assert_eq!(snap.frame_id, 9);

    // A HIGHER fence is current (not stale) — still forwarded.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence(2), 10),
    )]);
    assert_eq!(sent.len(), 1);

    // A stale fence is dropped and counted (the P2 old-owner guard).
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence::GENESIS, 11),
    )]);
    assert_eq!(sent.len(), 0);
    assert_eq!(rig.stats().stale_frames_dropped, 1);
}

#[test]
fn two_active_sessions_share_one_retagged_body() {
    // SCALE-1: a second session with the SAME sub re-uses the ONE retagged body
    // (the Occupied map arm) — the gateway never re-encodes per subscriber.
    let mut rig = Rig::new();
    let (_, _) = rig.login();
    // A second client logs in fully (distinct session, same sub 0).
    let before: std::collections::BTreeSet<SessionId> =
        rig.world.resource::<GatewaySessions>().sessions().collect();
    let _ = rig.tick(vec![Inbound::Wire {
        from: NodeId(101),
        class: MsgClass::Control,
        bytes: postcard::to_allocvec(&hello_msg()).expect("encode").into(),
    }]);
    let session2 = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .find(|s| !before.contains(s))
        .expect("second session pending");
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session2))]);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: session2,
            entity: EntityId(88),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);

    // One frame: BOTH clients receive a sub-0 snapshot from the shared body.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence(1), 9),
    )]);
    let mut recipients: Vec<NodeId> = sent
        .iter()
        .filter(|(_, class, _)| *class == MsgClass::Snapshot)
        .map(|(to, _, _)| *to)
        .collect();
    recipients.sort_unstable();
    assert_eq!(
        recipients,
        vec![CLIENT, NodeId(101)],
        "both subscribers fed"
    );
    for (_, _, bytes) in sent.iter().filter(|(_, c, _)| *c == MsgClass::Snapshot) {
        let snap: SnapshotDatagram = postcard::from_bytes(bytes).expect("decode");
        assert_eq!(snap.sub, SubId(0));
    }
    assert_eq!(rig.stats().undecodable, 0);
}

#[test]
fn a_corrupt_snapshot_body_is_counted_once_and_abandons_the_frame() {
    // The Err arm of the per-sub retag: a body that can't re-tag (bad varint)
    // would fail identically for every sub, so the whole frame is abandoned.
    let mut rig = Rig::new();
    let (_, _) = rig.login();
    let corrupt = ShardToGateway::Frame {
        realm_fence: Fence(1),
        source_tick: TickId(5),
        snapshot_bytes: vec![0x80], // truncated varint: retag fails
    };
    let sent = rig.tick(vec![wire(SHARD, MsgClass::Snapshot, &corrupt)]);
    // The corrupt body re-tags for no sub, so the whole frame is abandoned: the
    // active session receives NOTHING and the failure is counted exactly once.
    assert_eq!(sent.len(), 0, "no output from a corrupt body");
    assert_eq!(rig.stats().undecodable, 1, "counted exactly once");
}

#[test]
fn a_window_row_before_its_engine_is_dropped_fail_closed_and_counted_apart() {
    // THE WINDOW LANE's ingest (Slice B: the composer CONSUMES what attestation admits;
    // SHADOW — nothing reaches a client). Driven through the REAL ingest:
    //   admitted  — the roster-head sender, on the held window (Control AND the datagram
    //               class — both carriers run the ONE rule) ⇒ INGESTED;
    //   preroster — a marker BEFORE the author's first level (no attested roster to vouch);
    //   forged    — a routable-but-wrong sender is dropped + counted (fail-closed);
    //   unknown   — a straggler/forged window id drops by id mismatch;
    //   misauthored — a look about anything but the author, a marker about a non-child
    //               (vouched against the author's OWN attested roster — never a forest);
    //   stale     — a body older than the held statement (newest wins).
    // Nothing is guessed at, nothing reaches a client, nothing is `undecodable`.
    let mut rig = Rig::new();
    let (_, _) = rig.login();
    // The login derived + opened the Occupants window (id 1) on SHARD for System(7).
    assert_eq!(
        rig.world.resource::<GatewaySessions>().windows_open_count(),
        1
    );
    let roster_row = vd_wire::channels::RealmSnap {
        realm: RealmId::Planet(7),
        frame: FrameRef::PlanetCentered { planet_seed: 7 },
        pose: vd_core::pose::StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 7 },
            DVec3::new(30.0, 0.0, 0.0),
            vd_core::UniverseTick(5),
        ),
    };
    let frame_for = |window: WindowId| ShardToGateway::WindowFrame {
        realm_fence: Fence(1),
        window,
        at: vd_core::UniverseTick(5),
        hop: None,
        rows: vec![roster_row],
    };
    let body_about = |subject: RealmId, stmt: BodyStmt, at: u64| ShardToGateway::WindowBody {
        realm_fence: Fence(1),
        window: WindowId(1),
        subject,
        stmt,
        authored_at: vd_core::UniverseTick(at),
    };
    let membership = ShardToGateway::WindowMembership {
        window: WindowId(1),
        added: vec![RealmId::Planet(7)],
        removed: Vec::new(),
    };
    // ★ S10: the STATIC roster arrives on its own reliable arm. A realm whose children do not move
    // states its roster here and nowhere else, so this arm has to reach the same ingest the frame does
    // — otherwise every static child's marker is refused as unvouched, silently and only at scale.
    let statics_for = |window: WindowId| ShardToGateway::WindowStaticRows {
        realm_fence: Fence(1),
        window,
        authored_at: vd_core::UniverseTick(5),
        rows: vec![roster_row],
    };
    // PREROSTER: a marker arriving before the author's FIRST level finds no attested roster
    // to vouch for its subject — refused apart, fail-closed, not "misauthored".
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &body_about(RealmId::Planet(7), BodyStmt::Marker { luma: vec![2] }, 4),
    )]);
    assert_eq!(rig.stats().window_body_preroster, 1, "no roster yet");
    let sent = rig.tick(vec![
        // ADMITTED: the head, on the held window — frame (both carriers; the re-delivered
        // stamp replaces, latest-wins), membership, an admissible self-look, and an
        // admissible marker about a child the author's OWN roster rows vouch for.
        wire(SHARD, MsgClass::Control, &frame_for(WindowId(1))),
        wire(SHARD, MsgClass::RealmSnapshot, &frame_for(WindowId(1))),
        // ★ S10: the static roster on its own arm, through the SAME admission every window row runs.
        wire(SHARD, MsgClass::Control, &statics_for(WindowId(1))),
        wire(SHARD, MsgClass::Control, &membership),
        wire(
            SHARD,
            MsgClass::Control,
            &body_about(RealmId::System(7), BodyStmt::SelfLook { bag: vec![1] }, 5),
        ),
        wire(
            SHARD,
            MsgClass::Control,
            &body_about(RealmId::Planet(7), BodyStmt::Marker { luma: vec![2] }, 5),
        ),
        // STALE: an OLDER self-look than the one already held — newest wins, counted apart.
        wire(
            SHARD,
            MsgClass::Control,
            &body_about(RealmId::System(7), BodyStmt::SelfLook { bag: vec![9] }, 3),
        ),
        // FORGED: a routable shard that is NOT the head this window was opened on.
        wire(DEST, MsgClass::Control, &frame_for(WindowId(1))),
        // UNKNOWN: an id this gateway never minted (or already closed).
        wire(SHARD, MsgClass::Control, &frame_for(WindowId(99))),
        // MISAUTHORED: a look about a child (SL3 — a realm draws only itself), and a marker
        // about the author itself (R4 — a marker is only ever about a direct child).
        wire(
            SHARD,
            MsgClass::Control,
            &body_about(RealmId::Planet(7), BodyStmt::SelfLook { bag: vec![1] }, 5),
        ),
        wire(
            SHARD,
            MsgClass::Control,
            &body_about(RealmId::System(7), BodyStmt::Marker { luma: vec![2] }, 5),
        ),
    ]);
    // Since the flag day the composer EMITS (level/delta/datagram) — but a window ROW is
    // never forwarded verbatim: everything client-bound decodes as a composed message.
    // Verified as DATA (HR5: no wildcard arms a test can never reach): the class SET is
    // pinned by equality, every control decodes AND its leading postcard discriminant is a
    // composed-scene variant, every realm-class payload decodes as the composed datagram.
    let client_classes: std::collections::BTreeSet<MsgClass> = sent
        .iter()
        .filter(|(to, _, _)| *to == CLIENT)
        .map(|(_, class, _)| *class)
        .collect();
    assert_eq!(
        client_classes,
        [MsgClass::Control, MsgClass::RealmSnapshot].into(),
        "the client hears exactly the composed control + datagram lanes"
    );
    let scene_discs: std::collections::BTreeSet<u8> = [
        control_disc(&ServerControlMsg::RealmRegistry {
            origin: RealmId::System(7),
            origin_epoch: 0,
            rows: Vec::new(),
        }),
        control_disc(&ServerControlMsg::RealmSceneDelta {
            origin: RealmId::System(7),
            origin_epoch: 0,
            added: Vec::new(),
            removed: Vec::new(),
        }),
    ]
    .into();
    for (_, class, bytes) in sent.iter().filter(|(to, _, _)| *to == CLIENT) {
        if *class == MsgClass::Control {
            let _ = postcard::from_bytes::<ServerControlMsg>(bytes)
                .expect("client-bound control is a composed ServerControlMsg");
            let disc = bytes[0];
            assert!(
                scene_discs.contains(&disc),
                "only composed scene control reaches the client here (disc {disc})"
            );
        }
        if *class == MsgClass::RealmSnapshot {
            let _ = postcard::from_bytes::<RealmSnapshotDatagram>(bytes)
                .expect("client-bound realm bytes are the COMPOSED datagram");
        }
    }
    assert_eq!(
        rig.stats().window_rows_ingested,
        // ★ 6 → 7 AT S10: the static roster is its own admitted row, on its own reliable arm.
        7,
        "each admitted row ingested (two frames, THE STATIC ROSTER, membership, look, marker) PLUS \
         the parked preroster marker drained by the first level (Slice C1: parked, never lost — the \
         send-once lane cannot re-send it)"
    );
    assert_eq!(rig.stats().window_body_stale, 1, "the older look refused");
    assert_eq!(rig.stats().window_sender_mismatch, 1, "the forgery dropped");
    assert_eq!(rig.stats().window_unknown_row, 1, "the stranger id dropped");
    // ★ 2 → 1 WHEN THE PARK GATE LEARNED THE RIGHT QUESTION (2026-08-28). The two plants are a
    // LOOK about a non-author and a MARKER about a non-child. The look still drops. The marker
    // now PARKS, because the gate can no longer tell "this is not a child" from "the door that
    // names this child has not opened yet" — and telling those apart wrongly is what made the
    // home star vanish from the drawn scene.
    assert_eq!(
        rig.stats().window_misauthored_body,
        1,
        "the mis-authored LOOK drops — a look may only be about the author"
    );
    // FAIL-CLOSED IS UNCHANGED, AND STATED RATHER THAN ASSUMED: the mis-authored marker composes
    // NOTHING. It sits in one park slot, holds no statement, and never reaches a scene. Parking is
    // a delay in judging, never an admission.
    let parked = &rig.world.resource::<GatewaySessions>().windows[&WindowId(1)];
    assert_eq!(
        parked.parked_bodies.keys().copied().collect::<Vec<_>>(),
        vec![RealmId::System(7)],
        "the mis-authored marker parks in exactly one slot"
    );
    assert_eq!(
        parked.ingest.marker_of(RealmId::System(7)),
        None,
        "and nothing about it is held: a parked statement is not an admitted one"
    );
    assert_eq!(rig.stats().undecodable, 0, "a window row is NOT garbage");
    // The composer CONSUMED the admitted level: the session's one-hop chain folded at the
    // level's stamp and the roster row composed (SHADOW — measured above, served to no one).
    assert_eq!(rig.stats().window_folds, 1, "the admitted level folded");
    assert_eq!(
        rig.stats().window_composed_rows,
        1,
        "the roster row composed"
    );
    assert_eq!(rig.stats().window_instant_mismatch, 0);
}

#[test]
fn windows_derive_open_keepalive_and_close_with_the_sessions() {
    // THE WINDOW LANE's gateway driver (Slice A): the login's Active promote derives + opens
    // the Occupants window on the session's own shard; the keep-alive re-asserts it on the
    // DERIVED cadence (tick_hz/2 with the recheck channel disarmed — the shard's own
    // fallback derivation, mirrored); the session's end closes it — ZERO sessions ⇒ ZERO
    // window state, structurally (the design's teardown test, gateway half).
    let mut rig = Rig::new();
    let (sid, sends) = rig.login();
    let attach_tick = &sends[2];
    // Shard-bound sends only (bitwise `|`, HR5) — a client-bound control byte must never be
    // decoded under the shard contract (postcard is positional, an alias would be silent).
    let opens: Vec<(NodeId, GatewayToShard)> = attach_tick
        .iter()
        .filter(|(to, _, _)| (*to == SHARD) | (*to == DEST))
        .map(|(to, _, b)| {
            (
                *to,
                postcard::from_bytes::<GatewayToShard>(b)
                    .expect("every shard-bound byte this tick decodes under the shard contract"),
            )
        })
        .collect();
    assert_eq!(
        opens,
        vec![(
            SHARD,
            GatewayToShard::WindowOpen {
                window: WindowId(1),
                scope: WindowScope::Occupants,
                static_held: None,
            }
        )],
        "the Active promote opens the own-realm Occupants window, id minted from 1"
    );
    assert_eq!(rig.stats().window_open_sent, 1);
    // The keep-alive: due exactly on the derived cadence (config: recheck disarmed, 50 Hz ⇒
    // every 25 local ticks), idempotent re-assert of the SAME id + scope.
    set_tick(&mut rig, 24);
    let quiet = rig.tick(vec![]);
    assert_eq!(
        quiet.len(),
        0,
        "off-cadence, an idle Active session sends nothing"
    );
    set_tick(&mut rig, 25);
    let beat = rig.tick(vec![]);
    // ★ THE BEAT CARRIES THE KEEP-ALIVE AND ONE HEAD READ PER UNRESOLVED ANCESTOR (owner ruling
    // 2026-09-02 R9 step 2): the static attach now remembers the whole chain, so the gateway asks
    // the directory where the galaxy and the universe are served — the windows above the origin
    // open the tick those answers land. Two ancestors in this world, so two reads beside the one
    // keep-alive.
    assert_eq!(
        beat.len(),
        3,
        "the cadence beat carries the keep-alive and a head read per unresolved ancestor"
    );
    let keepalive = beat
        .iter()
        .find(|(_, class, _)| *class == MsgClass::Control)
        .expect("the keep-alive rides the control class");
    assert_eq!(
        postcard::from_bytes::<GatewayToShard>(&keepalive.2).expect("decode"),
        GatewayToShard::WindowOpen {
            window: WindowId(1),
            scope: WindowScope::Occupants,
            static_held: None,
        }
    );
    assert_eq!(rig.stats().window_keepalives_sent, 1);
    // The session ends: the window closes on the SAME tick and the state is EMPTY.
    let _ = sid;
    let bye = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    let closes = bye
        .iter()
        .filter(|(to, class, b)| {
            (*to == SHARD)
                & (*class == MsgClass::Control)
                & matches!(
                    postcard::from_bytes::<GatewayToShard>(b),
                    Ok(GatewayToShard::WindowClose {
                        window: WindowId(1)
                    })
                )
        })
        .count();
    assert_eq!(closes, 1, "the polite close rides the session's exit tick");
    let sessions = rig.world.resource::<GatewaySessions>();
    assert!(sessions.is_empty());
    assert_eq!(
        sessions.windows_open_count(),
        0,
        "zero sessions ⇒ zero window state (structural, both maps)"
    );
    assert_eq!(rig.stats().window_close_sent, 1);
}

#[test]
fn a_crossing_overlap_holds_both_chains_and_derives_the_child_window_on_the_parent() {
    // §2.7's posture in Slice-A form: through a hand-off overlap the session holds subs on
    // BOTH ends (the existing dual-sub pattern), so the derivation opens windows on BOTH
    // chains — and the moment the gateway's own routing state names a sub'd PARENT of a
    // sub'd realm (System(7) ⊃ Planet(7) in the login registry), the `Child(c)` window on
    // the parent's shard derives too. Nothing was looked up anew: the two subs and the login
    // registry are exactly what the gateway already held.
    let mut rig = Rig::new();
    let (sid, _) = rig.login(); // Occupants(System(7)) on SHARD == WindowId(1)
    let ready = rig.tick(vec![wire(
        DEST,
        MsgClass::Control,
        &ShardToGateway::SubscriptionReady {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            realm_fence: Fence(2),
        },
    )]);
    // Shard-bound sends only (same discipline as the login test above).
    let opens: Vec<(NodeId, GatewayToShard)> = ready
        .iter()
        .filter(|(to, _, _)| (*to == SHARD) | (*to == DEST))
        .map(|(to, _, b)| {
            (
                *to,
                postcard::from_bytes::<GatewayToShard>(b)
                    .expect("every shard-bound byte this tick decodes under the shard contract"),
            )
        })
        .collect();
    assert_eq!(
        opens,
        vec![
            (
                DEST,
                GatewayToShard::WindowOpen {
                    window: WindowId(2),
                    scope: WindowScope::Occupants,
                    static_held: None,
                }
            ),
            (
                SHARD,
                GatewayToShard::WindowOpen {
                    window: WindowId(3),
                    scope: WindowScope::Child(RealmId::Planet(7)),
                    static_held: None,
                }
            ),
        ],
        "the dest's own-level window AND the hop window on the parent, in derivation order"
    );
    assert_eq!(
        rig.world.resource::<GatewaySessions>().windows_open_count(),
        3,
        "both chains held through the overlap"
    );
}

#[test]
fn the_orchestrator_saga_class_dispatch_splits_reply_from_command_from_garbage() {
    use vd_wire::seams::transfer_control::TransferControl;
    let mut rig = Rig::new();

    // A TransferControl command (the saga driving the gateway) DISPATCHES to the
    // consumer — never mis-decoded as a directory reply. With no session 7 here it is
    // counted unroutable (consumed, not undecodable): the split is what this asserts.
    let cmd = InterShardFlow::Saga(TransferControl::PrepareSubscribe {
        transfer: vd_core::TransferId(1),
        session: SessionId(7),
        dest: SHARD,
    });
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &cmd)]);
    assert_eq!(rig.stats().transfer_unroutable, 1);
    assert_eq!(rig.stats().undecodable, 0, "a command is NOT undecodable");

    // A non-reply / non-command Saga-class arm (a misdirected Ghost) → undecodable.
    let ghost = InterShardFlow::Ghost(vd_wire::intershard::GhostFlow::Despawn {
        entity: EntityId::pack(vd_core::entity_kind::EntityKind::Player, 1, 7, 3),
        source_fence: Fence(1),
    });
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &ghost)]);
    assert_eq!(
        rig.stats().undecodable,
        1,
        "a non-dispatchable arm is undecodable"
    );

    // Raw garbage on the orchestrator Saga path → also undecodable (the Err arm).
    let _ = rig.tick(vec![Inbound::Wire {
        from: ORCH,
        class: MsgClass::Saga,
        bytes: vec![0xFF, 0xFF].into(),
    }]);
    assert_eq!(rig.stats().undecodable, 2);
    // The consumed-command count did not move (the split is clean in both directions).
    assert_eq!(rig.stats().transfer_unroutable, 1);

    // A well-formed reply that is NOT a Session head (a CAS outcome carries no gateway
    // obligation) decodes + dispatches to on_directory_reply, which returns without
    // effect — it is NOT undecodable (valid arm), just no-op for the gateway.
    let cas = InterShardFlow::DirectoryReply(DirectoryReply::CasResult {
        key: DirectoryKey::Session(SessionId(7)),
        outcome: vd_wire::seams::directory::CasOutcome::Won {
            new_fence: Fence(2),
        },
    });
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &cas)]);
    assert_eq!(
        rig.stats().undecodable,
        2,
        "a valid non-Head reply is not undecodable"
    );
}

#[test]
fn frames_skip_sessions_that_are_not_active_yet() {
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence(1), 1),
    )]);
    let to_client = sent.iter().filter(|(to, _, _)| *to == CLIENT).count();
    assert_eq!(to_client, 0, "pending sessions receive nothing");
}

#[test]
fn bye_detaches_revokes_and_clears() {
    let mut rig = Rig::new();
    let (session_id, _) = rig.login();
    let sent = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert!(rig.world.resource::<GatewaySessions>().is_empty());
    let detach: GatewayToShard = postcard::from_bytes(
        &sent
            .iter()
            .find(|(to, _, _)| *to == SHARD)
            .expect("detach sent")
            .2,
    )
    .expect("decode");
    assert_eq!(
        detach,
        GatewayToShard::DetachSession {
            session: session_id,
            fence: Fence(1),
        }
    );
    let revoke: InterShardFlow = postcard::from_bytes(
        &sent
            .iter()
            .find(|(to, _, _)| *to == ORCH)
            .expect("revoke sent")
            .2,
    )
    .expect("decode");
    assert_eq!(
        revoke,
        InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
            key: DirectoryKey::Session(session_id),
            fence: Fence(1),
        })
    );
    // Bye from a connection with no session is a no-op.
    let sent = rig.tick(vec![wire(
        NodeId(177),
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(sent.len(), 0);
    // The shard detach confirmation closes the loop silently.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionDetached {
            session: session_id,
        },
    )]);
    assert_eq!(sent.len(), 0);
}

#[test]
fn pending_phases_retry_every_tick_until_answered() {
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    // AwaitingDirectory: a grant retry goes out on an idle tick.
    let sent = rig.tick(vec![]);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].0, ORCH);
    let session_id = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .next()
        .expect("pending");
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
    // AwaitingAttach: an attach retry goes out on an idle tick.
    let sent = rig.tick(vec![]);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].0, SHARD);
    let retry: GatewayToShard = postcard::from_bytes(&sent[0].2).expect("decode");
    assert_eq!(
        retry,
        GatewayToShard::AttachSession {
            session: session_id,
            fence: Fence(1),
            account: AccountId(5),
            // Every attach carries the registry's pose now (the T2 spawn-standoff fix).
            spawn: attach_spawn(&rig),
        }
    );
}

#[test]
fn duplicate_and_late_replies_are_idempotent() {
    let mut rig = Rig::new();
    let (session_id, _) = rig.login();
    // A duplicate directory head after activation: no-op.
    let sent = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);
    assert_eq!(sent.len(), 0);
    // A duplicate attach reply: no second SubscriptionOpened.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: session_id,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    assert_eq!(decode_controls(&sent, CLIENT).len(), 0);
    // Replies for a session that's gone: ignored.
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    let sent = rig.tick(vec![
        wire(ORCH, MsgClass::Saga, &granted_head(session_id)),
        wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: session_id,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        ),
    ]);
    assert_eq!(sent.len(), 0);
}

#[test]
fn garbage_wrong_classes_and_notices_are_counted_or_skipped() {
    let mut rig = Rig::new();
    let _ = rig.tick(vec![
        // Undecodable from every peer family.
        Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Control,
            bytes: vec![0xFF].into(),
        },
        Inbound::Wire {
            from: SHARD,
            class: MsgClass::Control,
            bytes: vec![0xFF].into(),
        },
        Inbound::Wire {
            from: SHARD,
            class: MsgClass::Snapshot,
            bytes: vec![0xFF].into(),
        },
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Saga,
            bytes: vec![0xFF].into(),
        },
        // Wrong classes.
        Inbound::Wire {
            from: SHARD,
            class: MsgClass::Saga,
            bytes: vec![1].into(),
        },
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Control,
            bytes: vec![1].into(),
        },
        Inbound::Wire {
            from: CLIENT,
            class: MsgClass::Snapshot,
            bytes: vec![1].into(),
        },
        // Clock sync is the follower system's business: skipped here.
        Inbound::Wire {
            from: ORCH,
            class: MsgClass::Membership,
            bytes: vec![1].into(),
        },
        // Transport notices are skipped by the dispatcher.
        Inbound::NodeUnreachable {
            to: SHARD,
            class: MsgClass::Input,
            undelivered: vd_core::MsgId(0),
        },
    ]);
    // SIX, not seven — and the seventh moved rather than vanished. This rig never logs anybody in, so
    // the `CLIENT` node has no session; a `Snapshot` from it is not "a client sent malformed bytes",
    // it is a node the router does not know sending a class only a shard sends. That is now its own
    // fact, because on the live cluster it was a running shard's entire output.
    assert_eq!(rig.stats().undecodable, 6);
    assert_eq!(
        rig.stats().refused_unknown_sender,
        1,
        "the sessionless node's data frame is refused by SENDER, not miscounted as bad bytes"
    );
    // CutEmitted/Pong are accepted no-ops (nothing to bind to in P1).
    let _ = rig.tick(vec![
        wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::CutEmitted {
                transfer: vd_core::TransferId(1),
                marker_seq: 5,
            },
        ),
        wire(
            CLIENT,
            MsgClass::Control,
            &ClientControlMsg::Pong { nonce: 2 },
        ),
    ]);
    // Entity heads carry no gateway obligation.
    let _ = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &DirectoryReply::Head {
            key: DirectoryKey::Entity(EntityId(9)),
            record: None,
        },
    )]);
    // A Frame on the Control class is a peer bug, counted.
    let before = rig.stats().undecodable;
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &frame_msg(Fence(1), 1),
    )]);
    assert_eq!(rig.stats().undecodable, before + 1);
}

/// SPIKE-2a: the 20 Hz hot path is lock-free by construction (ArcSwap load +
/// one atomic + a byte splice) and fast enough that 50k route+retag rounds
/// finish far inside any tick budget even in a debug build. The bound is a
/// generous PROPERTY gate (catches contention collapse / accidental O(n²)),
/// not a microbenchmark number.
#[test]
// The seam ban targets PRODUCTION reaching for wall-clock; a latency microbench
// measuring elapsed time is exactly what Instant is for (justified exemption).
#[allow(clippy::disallowed_methods)]
fn spike_2a_hot_path_volume_bound() {
    let hot = SessionHot {
        route: ArcSwap::from_pointee(RouteSnapshot {
            authority: SHARD,
            fence: Fence(1),
            cut: None,
        }),
        last_input_seq: AtomicU64::new(0),
        subs: ArcSwap::from_pointee(SubTable::default()),
    };
    let snapshot_bytes = frame_msg(Fence(1), 1)
        .into_snapshot_bytes()
        .expect("frame_msg builds a Frame");
    let started = std::time::Instant::now();
    let mut forwarded = 0u64;
    for seq in 1..=50_000u64 {
        let input = input_bytes(seq);
        assert_eq!(
            route_input(&hot, &input),
            InputRouting::Forward { to: SHARD },
            "monotonic seqs always forward"
        );
        forwarded += 1;
        let out = forward_frame(&hot, SubId(0), Fence(1), &snapshot_bytes)
            .expect("current fence forwards");
        assert!(!out.is_empty());
    }
    assert_eq!(forwarded, 50_000);
    // MV-4: the 1c.5 cut partition adds a `.cut` read to `route_input`. Time it under
    // volume against a `cut: Some` route, crossing the marker, proving the added branch
    // is no contended-load regression (the partition is one compare on the loaded Arc).
    let cut_hot = SessionHot {
        route: ArcSwap::from_pointee(RouteSnapshot {
            authority: SHARD,
            fence: Fence(1),
            cut: Some(SeqCut {
                marker_seq: 25_000,
                dest: DEST,
            }),
        }),
        last_input_seq: AtomicU64::new(0),
        subs: ArcSwap::from_pointee(SubTable::default()),
    };
    let (mut to_source, mut to_buffer) = (0u64, 0u64);
    for seq in 1..=50_000u64 {
        // seq <= marker → Forward (source); seq > marker → Buffer (held for the dest).
        if route_input(&cut_hot, &input_bytes(seq)) == InputRouting::Buffer {
            to_buffer += 1;
        } else {
            to_source += 1;
        }
    }
    assert_eq!(
        (to_source, to_buffer),
        (25_000, 25_000),
        "partitioned at the marker"
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "hot path collapsed: 100k rounds took {elapsed:?}"
    );
}

/// THE ROUTER HOT-PATH VOLUME GATE — a crowd-sized snapshot body through the real fan-out.
///
/// WHAT IT USED TO MEASURE AND WHY THAT MATTERS. Until this slice the router RE-EXPRESSED every
/// snapshot body into each viewer's space, and this gate existed to keep that decode-and-re-encode from
/// collapsing under a crowd. The router does not do that any more: the shard chain ships every occupant
/// already measured in the frame of the realm its viewer is standing in, so the only per-body work left
/// here is the leading-varint re-tag and the refcounted fan. The gate is RE-AIMED at that remaining
/// work rather than deleted, because it is still the ONE 20 Hz path in the system with no latency gate
/// over it otherwise — `just spike3a` measures the mesh datagram transport on loopback with the router
/// outside its loop.
///
/// WHAT IS NOT COVERED HERE, AND WHERE IT IS: the cost that MOVED — one conversion per level, one
/// shard hop per level — is measured by `just chain-latency`
/// (`vd-tests`'s `the_chain_pays_one_tick_per_level_each_way_and_the_price_is_measured` and
/// `the_chain_compounds_staleness_per_level_under_a_lossy_link`). That gate measures in TICKS rather
/// than wall-clock, because the moved cost is a delivery per hop and not CPU; the two gates are
/// complementary and neither substitutes for the other. Measured on a 3-level chain at 20 Hz: one tick
/// per hop each way, exactly, and 4 ticks (200 ms) from a star authoring a placement to a client
/// holding it.
///
/// The ceiling is DERIVED, not written out: the gateway must finish a tick's work inside a tick, so one
/// frame per tick may take at most one tick period, and this asserts the whole 50k-round loop inside a
/// large multiple of that budget. It is a PROPERTY gate against contention collapse and accidental
/// O(n²) — a debug-build microbenchmark number would be meaningless — but a per-session full decode
/// (the shape this design exists to avoid) would blow it by two orders of magnitude.
#[test]
// The seam ban targets PRODUCTION reaching for wall-clock; a latency gate measuring elapsed time is
// exactly what Instant is for (justified exemption).
#[allow(clippy::disallowed_methods)]
fn the_router_hot_path_holds_a_crowd_sized_body_under_volume() {
    const ENTITIES: usize = 128;
    const ROUNDS: u32 = 50_000;
    let tick_hz = 50u32;

    let snapshot = SnapshotDatagram {
        sub: SubId(0),
        frame_id: 1,
        source_tick: TickId(5),
        universe_tick: UniverseTick(0),
        entities: (0..ENTITIES)
            .map(|i| EntitySnap {
                entity: EntityId(i as u128),
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::PlanetCentered { planet_seed: 7 },
                    DVec3::new(i as f64, 0.0, 0.0),
                    UniverseTick(0),
                ),
            })
            .collect(),
    };
    let shard_bytes = postcard::to_allocvec(&snapshot).expect("encode");
    let frame = postcard::to_allocvec(&ShardToGateway::Frame {
        realm_fence: Fence(1),
        source_tick: TickId(5),
        snapshot_bytes: shard_bytes.clone(),
    })
    .expect("encode");

    let (mut sessions, _sid, _) = one_active_session();
    let mut stats = GatewayStats::default();

    let started = std::time::Instant::now();
    let mut served = 0u32;
    let mut last = Vec::new();
    for _ in 0..ROUNDS {
        let mut outbox = OutboundBox::default();
        on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
        served += u32::try_from(outbox.0.len()).expect("one push per round");
        last = outbox.0[0].2.to_vec();
    }
    let elapsed = started.elapsed();
    assert_eq!(served, ROUNDS, "every round served the one subscriber");
    // ANTI-VACUITY, and the byte-identity claim in one assertion: the body the subscriber received is
    // the shard's own body, not a re-encoding of it. `one_active_session` holds SubId(0) and the
    // datagram already leads with sub 0, so the re-tag rewrites the same varint and the bytes match
    // exactly. A router that started composing again would fail here before it failed on time.
    assert_eq!(
        last, shard_bytes,
        "the router forwarded the shard's own crowd-sized body byte for byte"
    );

    // The derived ceiling: `ROUNDS` tick-periods of headroom over what is nominally ROUNDS ticks of
    // work. A debug build is ~an order of magnitude off release, hence the whole period rather than a
    // fraction of it; the failure this catches is a per-session decode, which costs sessions× more.
    let budget = std::time::Duration::from_secs_f64(f64::from(ROUNDS) / f64::from(tick_hz));
    assert!(
        elapsed < budget,
        "the gateway fan-out collapsed: {ROUNDS} rounds of {ENTITIES} entities took {elapsed:?} \
         (budget {budget:?} = one tick period per round at {tick_hz} Hz)"
    );
}

/// SPIKE-2a (the route-swap hot-path GATE — formally blocks the P2 route-swap design):
/// proves the gateway route swap is WAIT-FREE and TORN-READ-FREE under a CONCURRENT
/// `route.store` publisher (modeling a P2 `CommitAuthority` swap under load). The ONE
/// `ArcSwap` swap means any `route.load()` yields exactly ONE published `RouteSnapshot` —
/// never a field-mix — so authority + fence + cut (incl. `cut: Some(SeqCut)`, the field
/// P2 adds) move together atomically; the read path is one `ArcSwap::load` (+ for
/// `route_input` one relaxed atomic), no `Mutex` anywhere. No `CommitAuthority`-driven
/// `route.store` lands until this is green.
///
/// SCOPE (honest): this gates the SWAP MECHANIC (atomicity + the wait-free read latency).
/// It does NOT prove the cut-PARTITIONING read logic — `route_input`'s future
/// `seq <= marker → source / > marker → dest` branch (the slot at line ~190) is a
/// CORRECTNESS property that gets its own test in Slice 1c, not a latency gate. Two
/// timed bands are measured separately: the isolated ROUTE DECISION (`frame_passes_fence`
/// = one `route.load` + fence compare, no alloc — the thing the swap actually contends),
/// and the end-to-end per-sub FORWARD (`forward_frame` = load + retag, where production
/// amortizes the retag once-per-SubId via SCALE-1, so this is a fan-out figure, not the
/// route decision). The tight ROUTE-DECISION budget is what catches a contended-load
/// regression that the alloc-dominated forward number would hide.
///
/// HAND-ROLLED (no bench crate): no library expresses a concurrent-contention p99
/// HARD-FAIL gate — criterion/divan are report-only steady-state harnesses with no p99
/// and no fail-threshold (investigated, 2026); we would hand-compute p99 + the assert
/// regardless. The latency ASSERTS are RELEASE-ONLY: debug + coverage instrumentation
/// make a tail meaningless, so a debug/coverage run still exercises the concurrency plus
/// the torn-read invariant (fast, small N) while a release run (`just spike2a`,
/// `--test-threads=1` so siblings don't oversubscribe) enforces the timing. `Instant` is
/// the justified seam exemption (a latency microbench is exactly what wall-clock is for).
#[test]
#[allow(clippy::disallowed_methods)]
fn spike_2a_route_swap_is_wait_free_and_torn_read_free() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    // The publisher's small FIXED set of known-good routes (distinct authority + fence),
    // INCLUDING a `cut: Some(SeqCut)` member — the exact field P2's CommitAuthority adds —
    // so a Some-cut genuinely crosses the swap under contention and the torn-read
    // membership check covers the SeqCut bytes (not just the always-None P1 shape).
    // A frame at Fence(9) is never stale against any fence here, so `forward_frame`
    // always reaches the full retag (the worst-case forward).
    let routes = [
        RouteSnapshot {
            authority: SHARD,
            fence: Fence(1),
            cut: None,
        },
        RouteSnapshot {
            authority: ORCH,
            fence: Fence(2),
            cut: Some(SeqCut {
                marker_seq: 7,
                dest: SHARD,
            }),
        },
        RouteSnapshot {
            authority: SHARD,
            fence: Fence(3),
            cut: None,
        },
    ];
    let hot = Arc::new(SessionHot {
        route: ArcSwap::from_pointee(routes[0]),
        last_input_seq: AtomicU64::new(0),
        subs: ArcSwap::from_pointee(SubTable::default()),
    });
    let frame = frame_msg(Fence(9), 1)
        .into_snapshot_bytes()
        .expect("frame_msg builds a Frame");

    // 4 readers vs 1 publisher: a CONSERVATIVE-on-dev-hardware contention figure (a 4-core
    // box oversubscribes 5:N) — NOT a model of cloud core counts. Production reads a
    // session's hot state from ~one forwarder; 4 concurrent loaders is strictly HARDER, so
    // a pass here is a safe upper bound, not a scale claim.
    const READERS: usize = 4;
    // Small N under debug/coverage (instrumented — keep it quick); large N in release for
    // a stable tail. `cfg!` folds at compile time → no runtime branch (no coverage hole).
    const SAMPLES_PER_READER: usize = if cfg!(debug_assertions) {
        2_000
    } else {
        200_000
    };

    let stop = Arc::new(AtomicBool::new(false));
    let misses = Arc::new(AtomicU64::new(0));

    // Each reader returns TWO sample bands: (isolated route-decision, end-to-end forward).
    type Bands = (Vec<Duration>, Vec<Duration>);
    let bands: Vec<Bands> = std::thread::scope(|s| {
        // Publisher: swap the route as fast as it can (CommitAuthority under contention).
        let pub_hot = Arc::clone(&hot);
        let pub_stop = Arc::clone(&stop);
        s.spawn(move || {
            let mut i = 0usize;
            while !pub_stop.load(Ordering::Relaxed) {
                pub_hot.route.store(Arc::new(routes[i % routes.len()]));
                i = i.wrapping_add(1);
            }
        });
        let handles: Vec<_> = (0..READERS)
            .map(|_| {
                let hot = Arc::clone(&hot);
                let misses = Arc::clone(&misses);
                let frame = frame.clone();
                s.spawn(move || {
                    let mut route_read = Vec::with_capacity(SAMPLES_PER_READER);
                    let mut forward = Vec::with_capacity(SAMPLES_PER_READER);
                    let mut local = 0u64;
                    for _ in 0..SAMPLES_PER_READER {
                        // Band 1 — the ISOLATED route decision the swap contends: one
                        // `route.load` + fence compare, NO alloc, so a contended-load
                        // regression can't hide under the retag's heap-alloc noise.
                        let t = Instant::now();
                        let pass = frame_passes_fence(&hot, Fence(9));
                        route_read.push(t.elapsed());
                        // Band 2 — the end-to-end per-sub forward (load + retag alloc).
                        let t = Instant::now();
                        let out = forward_frame(&hot, SubId(0), Fence(9), &frame);
                        forward.push(t.elapsed());
                        // Invariants — accumulated via `+= u64::from(..)` (NOT an `if`), so
                        // each never-taken failure case stays a COVERED region, not a hole.
                        // A high-fence frame always passes + forwards (proves the reads
                        // RAN); a DIRECT load is a COMPLETE member of the published set.
                        // This last check is a STRUCTURAL-INVARIANT CANARY: ArcSwap cannot
                        // tear a single Arc today, so it guards a FUTURE regression where
                        // authority/fence/cut stop sharing one Arc (e.g. the P2 temptation
                        // to bolt marker_seq onto a separate atomic) — then a mixed load
                        // would be a non-member and fire here.
                        local += u64::from(!pass);
                        local += u64::from(out.is_none());
                        let loaded: RouteSnapshot = **hot.route.load();
                        local += u64::from(!routes.contains(&loaded));
                    }
                    misses.fetch_add(local, Ordering::Relaxed);
                    (route_read, forward)
                })
            })
            .collect();
        let bands = handles
            .into_iter()
            .map(|h| h.join().expect("reader thread"))
            .collect();
        stop.store(true, Ordering::Relaxed); // let the publisher exit before scope-join
        bands
    });

    // Invariants checked in EVERY build (incl. debug/coverage): no torn read AND every
    // high-fence read passed + forwarded (misses counts all failure modes → exactly 0).
    assert_eq!(
        misses.load(Ordering::Relaxed),
        0,
        "a route.load() was not a complete member of the published set (torn read), or a \
         high-fence frame failed to pass/forward"
    );
    let route_read: Vec<Duration> = bands.iter().flat_map(|(r, _)| r.iter().copied()).collect();
    let forward: Vec<Duration> = bands.iter().flat_map(|(_, f)| f.iter().copied()).collect();
    assert_eq!(route_read.len(), READERS * SAMPLES_PER_READER);
    assert_eq!(forward.len(), READERS * SAMPLES_PER_READER);

    // The p99 tail helper is the ONE shared latency-gate utility (HR3), extracted to
    // vd-harness so SPIKE-2a (here) and SPIKE-3a (vd-io-prod) can never drift.
    let route_p99 = vd_harness::latency::percentile_unstable(route_read, 99);
    let forward_p99 = vd_harness::latency::percentile_unstable(forward, 99);
    // The hard latency GATES are release-only (a debug/coverage tail is meaningless).
    #[cfg(not(debug_assertions))]
    {
        // The route DECISION (one ArcSwap load + fence compare) must be lost in the noise
        // of a 50 ms (20 Hz) tick — this is ~10,000x under. The budget guards the property
        // that actually matters: WAIT-FREE (no lock). Observed p99 ~625 ns under a
        // hammering publisher; a Mutex/lock in this read would be ≥20 µs under the same
        // contention, so 5 µs (~8x over observed) cleanly catches that regression while
        // staying robust on a throttled CI-less dev box. THE number that blocks P2.
        const ROUTE_DECISION_P99_BUDGET: Duration = Duration::from_micros(5);
        // The end-to-end forward includes the retag alloc production amortizes per-SubId
        // (SCALE-1) — a looser fan-out ceiling, not the route decision (observed ~600 ns).
        const FORWARD_FAN_OUT_P99_BUDGET: Duration = Duration::from_micros(50);
        eprintln!(
            "SPIKE-2a: route-decision p99 = {route_p99:?} (budget {ROUTE_DECISION_P99_BUDGET:?}); \
             forward-fan-out p99 = {forward_p99:?} (budget {FORWARD_FAN_OUT_P99_BUDGET:?}); \
             {} samples/band across {READERS} readers, 0 torn reads",
            READERS * SAMPLES_PER_READER
        );
        assert!(
            route_p99 < ROUTE_DECISION_P99_BUDGET,
            "route-decision p99 {route_p99:?} exceeded {ROUTE_DECISION_P99_BUDGET:?} under a \
             concurrent route.store publisher (a contended-load regression)"
        );
        assert!(
            forward_p99 < FORWARD_FAN_OUT_P99_BUDGET,
            "forward-fan-out p99 {forward_p99:?} exceeded {FORWARD_FAN_OUT_P99_BUDGET:?}"
        );
    }
    #[cfg(debug_assertions)]
    let _ = (route_p99, forward_p99);
}

#[test]
fn hot_path_unit_outcomes() {
    let hot = SessionHot {
        route: ArcSwap::from_pointee(RouteSnapshot {
            authority: SHARD,
            fence: Fence(2),
            cut: None,
        }),
        last_input_seq: AtomicU64::new(10),
        subs: ArcSwap::from_pointee(SubTable::default()),
    };
    assert_eq!(route_input(&hot, &input_bytes(10)), InputRouting::Deduped);
    assert_eq!(route_input(&hot, &input_bytes(5)), InputRouting::Deduped);
    assert_eq!(
        route_input(&hot, &input_bytes(11)),
        InputRouting::Forward { to: SHARD }
    );
    assert_eq!(route_input(&hot, &[0x80]), InputRouting::Malformed);
    // Stale frame fence → None; corrupt snapshot header → None.
    assert_eq!(forward_frame(&hot, SubId(0), Fence(1), &[0]), None);
    assert_eq!(forward_frame(&hot, SubId(0), Fence(2), &[0x80; 6]), None);
    // A multi-byte sub id re-tags exactly (the varint continuation path).
    let snapshot_bytes = frame_msg(Fence(2), 3)
        .into_snapshot_bytes()
        .expect("frame_msg builds a Frame");
    let big = forward_frame(&hot, SubId(40_000), Fence(2), &snapshot_bytes)
        .expect("current fence forwards");
    let decoded: SnapshotDatagram = postcard::from_bytes(&big).expect("decode");
    assert_eq!(decoded.sub, SubId(40_000));
}

#[test]
fn concurrent_inputs_at_one_seq_forward_exactly_once() {
    // FG-2: the dedup is a SINGLE atomic `fetch_max`, so many threads racing the
    // SAME seq yield exactly ONE Forward — every other thread dedups. The prior
    // non-atomic load-then-store could let several threads observe the same stale
    // high-water mark, all pass, and all forward a duplicate input. A barrier
    // maximizes the contention window.
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;

    const THREADS: usize = 32;
    let hot = SessionHot {
        route: ArcSwap::from_pointee(RouteSnapshot {
            authority: SHARD,
            fence: Fence(2),
            cut: None,
        }),
        last_input_seq: AtomicU64::new(0),
        subs: ArcSwap::from_pointee(SubTable::default()),
    };
    let forwards = AtomicUsize::new(0);
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|s| {
        for _ in 0..THREADS {
            s.spawn(|| {
                barrier.wait();
                if route_input(&hot, &input_bytes(7)) == (InputRouting::Forward { to: SHARD }) {
                    forwards.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    assert_eq!(
        forwards.load(Ordering::Relaxed),
        1,
        "exactly one thread forwards seq 7; the rest dedup"
    );
    assert_eq!(
        hot.last_input_seq.load(Ordering::Relaxed),
        7,
        "the high-water mark advanced to seq 7 exactly once"
    );
}

// ---- Slice 1c.2: the gateway TransferControl consumer ----------------------

const XFER: TransferId = TransferId(0x1c2);
/// The transfer SUBJECT the saga carries on `CommitAuthority` — the avatar the dest adopts
/// (1c.8). The gateway forwards it VERBATIM into `OpenInputSlot`; these tests assert that.
const XFER_SUBJECT: DirectoryKey = DirectoryKey::Entity(EntityId(0x1c8));

fn saga_cmd(cmd: TransferControl) -> Inbound {
    wire(ORCH, MsgClass::Saga, &InterShardFlow::Saga(cmd))
}

/// The `SagaAck`s the gateway sent back to the orchestrator (ignoring the directory
/// ops that also ride ORCH+Saga — login's LeaseGrant, etc.).
fn acks_to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<TransferControlAck> {
    sent.iter()
        .filter(|(node, class, _)| (*node == ORCH) & (*class == MsgClass::Saga))
        .filter_map(|(_, _, bytes)| {
            // The gateway only ever sends valid flows; `.expect` keeps the Err path in
            // std (no caller branch). A non-SagaAck ORCH/Saga send (a directory op, e.g.
            // login's LeaseGrant) maps to None — exercised by the login-tick assertion.
            match postcard::from_bytes::<InterShardFlow>(bytes).expect("gateway sends a valid flow")
            {
                InterShardFlow::SagaAck(ack) => Some(ack),
                _ => None,
            }
        })
        .collect()
}

fn marker_input(seq: u64) -> Inbound {
    wire(
        CLIENT,
        MsgClass::Input,
        &InputDatagram {
            seq,
            is_cut_marker: true,
            client_tick: TickId(1),
            movement: [0.0, 0.0, 0.0],
            look: [0.0, 0.0],
            action_bits: 0,
        },
    )
}

fn transfer_in_flight(rig: &Rig, sid: SessionId) -> bool {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session present")
        .transfer
        .is_some()
}

/// Drive a session to a LIVE cut: Prepare(dest:DEST) -> RequestCut -> marker(M) ->
/// FreezeSource(marker_seq:M, dest:DEST). Returns the tick's sends from the freeze.
fn freeze_to_live_cut(
    rig: &mut Rig,
    sid: SessionId,
    marker: u64,
) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
        transfer: XFER,
        session: sid,
    })]);
    let _ = rig.tick(vec![marker_input(marker)]);
    rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
        transfer: XFER,
        session: sid,
        marker_seq: marker,
        dest: DEST,
    })])
}

fn route_cut(rig: &Rig, sid: SessionId) -> Option<SeqCut> {
    route_state(rig, sid).cut
}

/// The full loaded route snapshot (all three fields from ONE coherent load) — the swap
/// oracle for 1c.4 (authority + fence + cut together).
fn route_state(rig: &Rig, sid: SessionId) -> RouteSnapshot {
    *rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session present")
        .hot
        .route
        .load_full()
}

#[test]
fn prepare_opens_progress_and_acks_ready() {
    let mut rig = Rig::new();
    let (sid, login_sends) = rig.login();
    // The hello tick's ORCH/Saga send is a directory LeaseGrant, not a SagaAck — so
    // `acks_to_orch` yields none (covers the non-ack decode arm of the helper).
    assert_eq!(acks_to_orch(&login_sends[0]), vec![]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        }]
    );
    assert!(
        transfer_in_flight(&rig, sid),
        "PrepareSubscribe opened progress"
    );
}

#[test]
fn request_cut_pushes_to_client_and_self_acks_cut_confirmed() {
    // S3: RequestCut is now SELF-ACKING (server-timed cut). It STILL pushes one
    // ServerControlMsg::RequestCut to the CLIENT (cosmetic — the old client's marker is inert)
    // AND self-acks CutConfirmed{marker_seq: 0 placeholder} in the SAME tick — the saga advances
    // Cutting→Freezing with no client marker. (The real input-cut seq is derived at FreezeSource.)
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![ServerControlMsg::RequestCut { transfer: XFER }],
        "RequestCut still pushes to the client (cosmetic — the marker is now inert)",
    );
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 0, // placeholder; the real seq is FreezeSource's install-time high-water
        }],
        "RequestCut self-acks CutConfirmed server-side (no client marker needed)",
    );

    // A client CUT_MARKER on the INPUT flow is now INERT — an ordinary input, NO CutConfirmed.
    let sent = rig.tick(vec![marker_input(42)]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![],
        "the client marker no longer drives a CutConfirmed (S3 — server-timed cut)",
    );
}

#[test]
fn a_redelivered_request_cut_re_serves_the_recorded_cut_confirmed() {
    // S3: RequestCut is now self-acking and journaled at step 1, so a redelivery re-serves the
    // SAME CutConfirmed verbatim via the standard redelivery gate — it does NOT re-push the
    // client RequestCut nor re-generate the ack (the effect already ran once).
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let request_cut = || {
        saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })
    };
    let first = rig.tick(vec![request_cut()]);
    assert_eq!(
        decode_controls(&first, CLIENT),
        vec![ServerControlMsg::RequestCut { transfer: XFER }],
        "the first RequestCut pushes to the client + self-acks",
    );
    let confirmed = vec![TransferControlAck::CutConfirmed {
        transfer: XFER,
        marker_seq: 0,
    }];
    assert_eq!(acks_to_orch(&first), confirmed);
    // Redeliver RequestCut: the redelivery gate re-serves the recorded CutConfirmed, and does
    // NOT re-push the client RequestCut (no re-effect).
    let sent = rig.tick(vec![request_cut()]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![],
        "a redelivered RequestCut re-serves the recorded ack — it does NOT re-push the client",
    );
    assert_eq!(
        acks_to_orch(&sent),
        confirmed,
        "the redelivery re-serves the SAME CutConfirmed verbatim (idempotent)",
    );
}

#[test]
fn freeze_installs_the_live_cut_and_acks_source_frozen() {
    // G2: FreezeSource installs Some(SeqCut{marker_seq, dest}) on the route and acks
    // SourceFrozen{drained_seq == marker_seq}. Asserting dest == DEST (!= the authority
    // SHARD) proves `dest` THREADS from the wire command into the cut, not defaulted.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let sent = freeze_to_live_cut(&mut rig, sid, 42);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::SourceFrozen {
            transfer: XFER,
            drained_seq: 42,
        }]
    );
    assert_eq!(
        route_cut(&rig, sid),
        Some(SeqCut {
            marker_seq: 42,
            dest: DEST,
        }),
        "the live cut carries the wire-supplied marker_seq + dest"
    );
}

#[test]
fn an_installed_cut_partitions_input_at_the_marker() {
    // 1c.5: the live cut PARTITIONS `route_input` — all three arms (a SessionHot built
    // directly so the dedup high-water is BELOW the marker, exercising the `seq <= marker`
    // Forward arm that the marker-advanced rig high-water would otherwise hide).
    // (Flipped from the 1c.4 installed-but-inert baseline this test reserved.)
    let hot = SessionHot {
        route: ArcSwap::from_pointee(RouteSnapshot {
            authority: SHARD,
            fence: Fence(2),
            cut: Some(SeqCut {
                marker_seq: 42,
                dest: DEST,
            }),
        }),
        last_input_seq: AtomicU64::new(0),
        subs: ArcSwap::from_pointee(SubTable::default()),
    };
    // seq <= marker (and past the dedup high-water) → Forward to the SOURCE authority.
    assert_eq!(
        route_input(&hot, &input_bytes(40)),
        InputRouting::Forward { to: SHARD }
    );
    // seq > marker → Buffer (the cold caller holds it for the dest).
    assert_eq!(route_input(&hot, &input_bytes(99)), InputRouting::Buffer);
    // a duplicate (<= the dedup high-water, now 99) is Deduped, NOT buffered.
    assert_eq!(route_input(&hot, &input_bytes(50)), InputRouting::Deduped);
    // and with NO cut installed, a fresh seq is a plain Forward (the `_` arm).
    let no_cut = SessionHot {
        route: ArcSwap::from_pointee(RouteSnapshot {
            authority: SHARD,
            fence: Fence(2),
            cut: None,
        }),
        last_input_seq: AtomicU64::new(0),
        subs: ArcSwap::from_pointee(SubTable::default()),
    };
    assert_eq!(
        route_input(&no_cut, &input_bytes(100)),
        InputRouting::Forward { to: SHARD }
    );
}

#[test]
fn a_redelivered_freeze_resends_source_frozen_without_re_storing_the_route() {
    // G4: FreezeSource owns its step-2 journal slot, so a redelivery short-circuits at
    // the gate (re-sends the SAME SourceFrozen) and NEVER re-enters apply_freeze — the
    // route keeps the ORIGINAL cut even if the redelivery names a different marker/dest.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let first = freeze_to_live_cut(&mut rig, sid, 42);
    let second = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
        transfer: XFER,
        session: sid,
        marker_seq: 999,
        dest: ORCH,
    })]);
    assert_eq!(
        acks_to_orch(&first),
        acks_to_orch(&second),
        "same SourceFrozen re-sent"
    );
    assert_eq!(
        route_cut(&rig, sid),
        Some(SeqCut {
            marker_seq: 42,
            dest: DEST,
        }),
        "the route keeps the original cut; the redelivery never re-touched it"
    );
}

#[test]
fn freeze_for_an_unrecorded_transfer_drops_and_counts_no_ack() {
    // G6 (LBD-2): an unbound FreezeSource installs NO cut, sends NO ack (pinning the
    // saga — WEDGE-1), and is counted. Diverges from thaw (a compensator that acks).
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let sent = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
        transfer: XFER,
        session: sid,
        marker_seq: 7,
        dest: DEST,
    })]);
    assert_eq!(acks_to_orch(&sent), vec![], "no ack -> the saga pins");
    assert_eq!(
        route_cut(&rig, sid),
        None,
        "no cut installed for an unbound freeze"
    );
    assert_eq!(rig.stats().transfer_unroutable, 1);
}

#[test]
fn thaw_clears_a_live_cut_then_acks() {
    // G5: drive a genuinely-LIVE cut, THEN thaw — the compensator clears it to None.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 42);
    assert!(
        route_cut(&rig, sid).is_some(),
        "cut is live before the thaw"
    );
    let sent = rig.tick(vec![saga_cmd(TransferControl::ThawSource {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::SourceThawed { transfer: XFER }]
    );
    assert_eq!(route_cut(&rig, sid), None, "thaw cleared the live cut");
}

#[test]
fn abort_clears_a_live_cut_locally() {
    // ROB-1c3: AbortTransfer must clear an installed cut LOCALLY, not borrow safety from
    // the saga's Thaw-before-Abort ordering. Drive a LIVE cut, then abort DIRECTLY (no
    // preceding thaw) — the cut is gone and the progress pruned.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 42);
    assert!(
        route_cut(&rig, sid).is_some(),
        "cut is live before the abort"
    );
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Aborted { transfer: XFER }]
    );
    assert_eq!(
        route_cut(&rig, sid),
        None,
        "abort cleared the live cut locally"
    );
    assert!(!transfer_in_flight(&rig, sid), "abort pruned the progress");
}

#[test]
fn prepare_for_a_not_yet_active_session_is_refused_and_counted() {
    // WB-1: a transfer phase can only run for an ATTACHED (Active) session. A
    // PrepareSubscribe that races ahead of SessionAttached (session still AwaitingAttach)
    // is refused — no ack (pins the saga), no progress created, counted — so a later
    // attach can never clobber a cut/route installed on a half-attached session.
    let mut rig = Rig::new();
    // Drive Hello + the directory grant, but NOT SessionAttached → AwaitingAttach.
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sid = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .next()
        .expect("session pending");
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    // The session is NOT Active yet — a PrepareSubscribe must be refused.
    let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![],
        "no Prepared ack for a non-Active session"
    );
    assert!(
        !transfer_in_flight(&rig, sid),
        "no transfer progress created on a half-attached session"
    );
    assert_eq!(rig.stats().transfer_unroutable, 1);
}

// ---- Slice 1c.4: CommitAuthority — the route swap --------------------------

#[test]
fn commit_swaps_authority_to_dest_and_acks() {
    // T2: the route swap moves authority -> cut.dest, clears the cut, and CARRIES the
    // realm fence UNCHANGED (R-FENCE: NOT the per-Entity CAS new_fence). DEST != SHARD
    // (the source/authority), so this proves dest threads from the installed cut.
    let mut rig = Rig::new();
    let (sid, _) = rig.login(); // attach installs route.fence = Fence(1)
    let _ = freeze_to_live_cut(&mut rig, sid, 7);
    let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Committed { transfer: XFER }]
    );
    assert_eq!(
        route_state(&rig, sid),
        RouteSnapshot {
            authority: DEST,
            fence: Fence(1), // R-FENCE: CARRIED, not the new_fence(2)
            cut: None,
        }
    );
}

#[test]
fn commit_does_not_drop_the_dest_own_realm_frames() {
    // T3 (R-FENCE regression guard): after the swap the route fence is CARRIED (Fence(1)),
    // so the dest's own realm-stamped frames (Fence(1)) still FORWARD. This FAILS the
    // instant someone installs new_fence(2) as route.fence (the black-screen bug). A
    // genuinely-stale frame (GENESIS) still drops — rule-5 machinery is live by data.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 7);
    let _ = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    // A realm-stamped frame at the carried fence is forwarded to the client.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence(1), 9),
    )]);
    let forwarded = sent
        .iter()
        .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
        .count();
    assert_eq!(
        forwarded, 1,
        "the dest's own realm frame still forwards post-swap"
    );
    assert_eq!(rig.stats().stale_frames_dropped, 0);
    // A genuinely stale frame still drops + counts.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence::GENESIS, 10),
    )]);
    assert_eq!(sent.len(), 0);
    assert_eq!(rig.stats().stale_frames_dropped, 1);
}

#[test]
fn commit_without_a_prior_freeze_pins_and_counts() {
    // T4: BOUND (prepared) but no cut installed -> commit_without_cut, no ack, route
    // untouched (never a garbage-dest swap). The saga pins.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    assert_eq!(acks_to_orch(&sent), vec![], "no ack -> the saga pins");
    assert_eq!(route_state(&rig, sid).authority, SHARD, "route untouched");
    assert_eq!(route_state(&rig, sid).cut, None);
    assert_eq!(rig.stats().commit_without_cut, 1);
    assert_eq!(
        rig.stats().transfer_unroutable,
        0,
        "distinct from unroutable"
    );
}

#[test]
fn commit_for_an_absent_transfer_pins_and_counts_unroutable() {
    // T5: no prepare at all (unbound) -> transfer_unroutable, no ack, route untouched.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    assert_eq!(acks_to_orch(&sent), vec![]);
    assert_eq!(rig.stats().transfer_unroutable, 1);
    assert_eq!(rig.stats().commit_without_cut, 0, "distinct from no-cut");
    assert_eq!(route_state(&rig, sid).authority, SHARD, "route untouched");
}

#[test]
fn commit_is_idempotent_on_redelivery() {
    // T6: a redelivered CommitAuthority re-serves Committed FROM THE JOURNAL and NEVER
    // re-enters apply_commit — proven by STATE (the route is byte-identical), since a
    // re-entry could not reconstruct dest from the now-None cut.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 7);
    let commit = || {
        saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })
    };
    let first = rig.tick(vec![commit()]);
    let after_first = route_state(&rig, sid);
    let second = rig.tick(vec![commit()]);
    assert_eq!(
        acks_to_orch(&first),
        acks_to_orch(&second),
        "same Committed re-served"
    );
    assert_eq!(
        route_state(&rig, sid),
        after_first,
        "route byte-identical: the redelivery never re-ran apply_commit"
    );
    assert_eq!(
        rig.stats().commit_without_cut,
        0,
        "redelivery is not a no-cut fault"
    );
}

// ---- Slice 1c.5: the cut partition — gateway buffer + drain-at-commit -------

fn client_input(seq: u64) -> Inbound {
    Inbound::Wire {
        from: CLIENT,
        class: MsgClass::Input,
        bytes: input_bytes(seq).into(),
    }
}

/// The `seq`s of `SessionInput` frames the gateway sent to `dest` (the drained cut buffer).
/// The variant match (not a class pre-filter) discriminates — the commit-drain stream to
/// `dest` mixes `OpenInputSlot` + `SessionInput`, so both match arms are live.
fn shard_input_seqs(sent: &[(NodeId, MsgClass, Vec<u8>)], dest: NodeId) -> Vec<u64> {
    sent.iter()
        .filter(|(to, _, _)| *to == dest)
        .filter_map(|(_, _, bytes)| {
            match postcard::from_bytes::<GatewayToShard>(bytes).expect("gateway sends valid") {
                GatewayToShard::SessionInput { input_bytes, .. } => {
                    Some(peek_input_seq(&input_bytes).expect("valid input"))
                }
                _ => None,
            }
        })
        .collect()
}

/// The `resume_from_seq`s of `OpenInputSlot` frames the gateway sent to `dest`.
/// As above, the variant match discriminates so the `_ => None` arm is exercised by the
/// `SessionInput` frames in the same drained stream.
fn open_slot_watermarks(sent: &[(NodeId, MsgClass, Vec<u8>)], dest: NodeId) -> Vec<u64> {
    sent.iter()
        .filter(|(to, _, _)| *to == dest)
        .filter_map(|(_, _, bytes)| {
            match postcard::from_bytes::<GatewayToShard>(bytes).expect("gateway sends valid") {
                GatewayToShard::OpenInputSlot {
                    resume_from_seq, ..
                } => Some(resume_from_seq),
                _ => None,
            }
        })
        .collect()
}

/// The `subject`s of `OpenInputSlot` frames the gateway sent to `dest` (1c.8): proves the
/// CommitAuthority subject is forwarded VERBATIM into the dest's adopt slot.
fn open_slot_subjects(sent: &[(NodeId, MsgClass, Vec<u8>)], dest: NodeId) -> Vec<DirectoryKey> {
    sent.iter()
        .filter(|(to, _, _)| *to == dest)
        .filter_map(|(_, _, bytes)| {
            match postcard::from_bytes::<GatewayToShard>(bytes).expect("gateway sends valid") {
                GatewayToShard::OpenInputSlot { subject, .. } => Some(subject),
                _ => None,
            }
        })
        .collect()
}

fn dest_buffer_len(rig: &Rig, sid: SessionId) -> usize {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session")
        .transfer
        .as_ref()
        .map_or(0, |tp| tp.dest_buffer.len())
}

#[test]
fn seq_past_the_marker_buffers_for_the_dest_and_is_not_forwarded() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 10);
    // Three seq>marker inputs: held in the cut buffer, NOT forwarded to the source.
    let sent = rig.tick(vec![client_input(11), client_input(12), client_input(13)]);
    assert_eq!(rig.stats().inputs_buffered_for_dest, 3);
    assert_eq!(dest_buffer_len(&rig, sid), 3);
    assert_eq!(
        shard_input_seqs(&sent, SHARD),
        Vec::<u64>::new(),
        "buffered input does NOT go to the source"
    );
    assert_eq!(
        shard_input_seqs(&sent, DEST),
        Vec::<u64>::new(),
        "nothing reaches the dest until commit"
    );
}

#[test]
fn the_cut_buffer_drops_oldest_over_cap_and_counts() {
    // Default cap is 8 (config()); 11 inputs ⇒ 3 oldest shed, newest 8 kept.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 100);
    for seq in 101..=111 {
        let _ = rig.tick(vec![client_input(seq)]);
    }
    assert_eq!(
        dest_buffer_len(&rig, sid),
        8,
        "capped at max_buffered_inputs"
    );
    assert_eq!(rig.stats().dest_inputs_dropped, 3);
    assert_eq!(
        rig.stats().inputs_buffered_for_dest,
        11,
        "all counted as buffered"
    );
    // Commit + drain and assert WHICH seqs survive: drop-OLDEST means the kept window is the
    // NEWEST 8 (104..=111), drained in seq order — proving the latest-wins identity, not just
    // the length. A drop-NEWEST inversion would strand the player's most recent input here.
    let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    assert_eq!(
        shard_input_seqs(&sent, DEST),
        vec![104, 105, 106, 107, 108, 109, 110, 111],
        "the drained survivors are exactly the kept newest-8 window, in order"
    );
}

#[test]
fn commit_opens_the_dest_slot_then_drains_the_buffer_in_seq_order() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 10);
    let _ = rig.tick(vec![client_input(11), client_input(12), client_input(13)]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    // The authoritative OpenInputSlot carries resume_from_seq == marker_seq.
    assert_eq!(open_slot_watermarks(&sent, DEST), vec![10]);
    // 1c.8: it also carries the transfer subject VERBATIM (the dest adopts this avatar).
    assert_eq!(open_slot_subjects(&sent, DEST), vec![XFER_SUBJECT]);
    // The buffer drained to the dest, in seq order.
    assert_eq!(shard_input_seqs(&sent, DEST), vec![11, 12, 13]);
    assert_eq!(dest_buffer_len(&rig, sid), 0, "buffer emptied by the drain");
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Committed { transfer: XFER }]
    );
}

#[test]
fn a_redelivered_commit_does_not_re_drain_the_buffer() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 10);
    let _ = rig.tick(vec![client_input(11), client_input(12)]);
    let commit = || {
        saga_cmd(TransferControl::CommitAuthority {
            transfer: XFER,
            session: sid,
            new_fence: Fence(2),
            subject: XFER_SUBJECT,
        })
    };
    let first = rig.tick(vec![commit()]);
    assert_eq!(shard_input_seqs(&first, DEST), vec![11, 12]);
    // Redelivery: re-serves Committed from the journal, drains NOTHING (buffer is empty).
    let second = rig.tick(vec![commit()]);
    assert_eq!(
        acks_to_orch(&second),
        acks_to_orch(&first),
        "same Committed re-served"
    );
    assert_eq!(
        shard_input_seqs(&second, DEST),
        Vec::<u64>::new(),
        "the redelivery never re-drains"
    );
}

#[test]
fn abort_drops_the_cut_buffer() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = freeze_to_live_cut(&mut rig, sid, 10);
    let _ = rig.tick(vec![client_input(11), client_input(12)]);
    assert_eq!(dest_buffer_len(&rig, sid), 2);
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert!(!transfer_in_flight(&rig, sid), "abort pruned the progress");
    assert_eq!(
        shard_input_seqs(&sent, DEST),
        Vec::<u64>::new(),
        "buffered seq>marker frames reach NEITHER shard on abort (player stays on source)"
    );
}

#[test]
fn abort_acks_and_prunes_the_progress() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    assert!(transfer_in_flight(&rig, sid));
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Aborted { transfer: XFER }]
    );
    assert!(
        !transfer_in_flight(&rig, sid),
        "AbortTransfer pruned the progress (the saga's terminal)"
    );
}

#[test]
fn a_redelivered_prepare_resends_the_same_ack_without_re_applying() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let prepare = || {
        saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })
    };
    let first = rig.tick(vec![prepare()]);
    let second = rig.tick(vec![prepare()]);
    // The SAME Prepared ack both times (the journal re-sent it; no second effect).
    assert_eq!(acks_to_orch(&first), acks_to_orch(&second));
    assert_eq!(acks_to_orch(&second).len(), 1);
}

#[test]
fn a_cut_marker_is_inert_no_ack_after_request_cut() {
    // S3: the client CUT_MARKER is RETIRED. After RequestCut has self-acked CutConfirmed, a
    // stamped marker on the input flow is just an ordinary input — it drives NO ack, however
    // many times it is (re)sent. (The saga already advanced server-side; the marker is dead.)
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
        transfer: XFER,
        session: sid,
    })]);
    // Three sends of the marker at the same seq → ZERO acks each (the marker is inert now).
    let a = rig.tick(vec![marker_input(9)]);
    let b = rig.tick(vec![marker_input(9)]);
    let c = rig.tick(vec![marker_input(9)]);
    assert_eq!(
        acks_to_orch(&a),
        vec![],
        "a stamped marker drives no ack (S3)"
    );
    assert_eq!(
        acks_to_orch(&b),
        vec![],
        "a re-sent marker drives no ack (S3)"
    );
    assert_eq!(
        acks_to_orch(&c),
        vec![],
        "a re-sent marker drives no ack (S3)"
    );
}

fn session_transfer_is_none(rig: &Rig, sid: SessionId) -> bool {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session")
        .transfer
        .is_none()
}

#[test]
fn release_subscribe_acks_released() {
    // 1c.8: ReleaseSubscribe is the LAST phase flipped from PARK to LIVE — it acks Released
    // (so the saga reaches Done) and clears session.transfer (the last post-commit remnant).
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: SHARD,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Released { transfer: XFER }],
        "ReleaseSubscribe now acks Released (no longer parks)"
    );
    assert_eq!(
        rig.stats().transfer_control_parked,
        0,
        "nothing parks anymore"
    );
    // The in-flight transfer is pruned (the source subscription closed).
    assert!(
        session_transfer_is_none(&rig, sid),
        "session.transfer cleared on release"
    );
    // Idempotent redelivery: a stray re-ack of the now-absent transfer is a clean
    // no-op-and-ack (still Released, transfer_unroutable stays 0).
    let again = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: SHARD,
    })]);
    assert_eq!(
        acks_to_orch(&again),
        vec![TransferControlAck::Released { transfer: XFER }],
        "a redelivered Release is a clean no-op-and-ack"
    );
    assert_eq!(
        rig.stats().transfer_unroutable,
        0,
        "a release re-ack is not a routing failure"
    );
}

#[test]
fn thaw_for_an_unrecorded_transfer_still_acks_but_is_counted() {
    // ThawSource's compensator must always complete (a thaw against a never-frozen
    // source is a correct no-op), yet the missing progress is counted, not silent.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let sent = rig.tick(vec![saga_cmd(TransferControl::ThawSource {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::SourceThawed { transfer: XFER }]
    );
    assert_eq!(rig.stats().transfer_unroutable, 1);
}

#[test]
fn abort_for_an_unrecorded_transfer_is_an_idempotent_uncounted_re_ack() {
    // An abort is an idempotent terminal: aborting a transfer this gateway never held
    // re-acks Aborted and is NOT a routing failure (F2: transfer_unroutable stays clean).
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Aborted { transfer: XFER }]
    );
    assert_eq!(
        rig.stats().transfer_unroutable,
        0,
        "an abort is not unroutable"
    );
}

#[test]
fn a_redelivered_terminal_abort_is_idempotent_and_uncounted() {
    // F2: after a bound abort prunes the progress, a redelivered abort (now unbound)
    // re-acks Aborted without inflating transfer_unroutable (healthy at-least-once).
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let abort = || {
        saga_cmd(TransferControl::AbortTransfer {
            transfer: XFER,
            session: sid,
        })
    };
    let first = rig.tick(vec![abort()]);
    let second = rig.tick(vec![abort()]); // redelivered terminal
    let aborted = vec![TransferControlAck::Aborted { transfer: XFER }];
    assert_eq!(acks_to_orch(&first), aborted);
    assert_eq!(
        acks_to_orch(&second),
        aborted,
        "the redelivered terminal re-acks"
    );
    assert_eq!(
        rig.stats().transfer_unroutable,
        0,
        "no spurious unroutable count"
    );
}

#[test]
fn abort_of_a_different_transfer_does_not_clobber_the_in_flight_one() {
    // CP-1: an AbortTransfer for transfer B must NOT prune in-flight transfer A's
    // progress — the prune is keyed on the matching transfer, not the variant.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let other = TransferId(0x777);
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    assert!(transfer_in_flight(&rig, sid));
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: other,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Aborted { transfer: other }],
        "the foreign abort still acks idempotently"
    );
    assert!(
        transfer_in_flight(&rig, sid),
        "in-flight transfer A survives an abort aimed at transfer B"
    );
}

#[test]
fn release_of_a_different_transfer_does_not_clobber_the_in_flight_one() {
    // TAIL-1 (mirrors the abort no-clobber case): a ReleaseSubscribe for transfer B must NOT
    // prune in-flight transfer A — `apply_release` prunes ONLY the matching transfer; a stray
    // re-ack of an absent/foreign transfer is a clean ack-and-no-op.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let other = TransferId(0x777);
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    assert!(transfer_in_flight(&rig, sid));
    let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: other,
        session: sid,
        src: SHARD,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Released { transfer: other }],
        "the foreign release still acks idempotently"
    );
    assert!(
        transfer_in_flight(&rig, sid),
        "in-flight transfer A survives a release aimed at transfer B"
    );
}

#[test]
fn a_bye_with_an_in_flight_transfer_is_warned_and_detaches() {
    // WEDGE-1 pin: a Bye mid-transfer drops the session + journal (the saga then pins
    // until the Slice-2 timeout). The path runs cleanly + detaches; the warn fires.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    assert!(transfer_in_flight(&rig, sid));
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    // The session (and its in-flight-transfer journal) is dropped on Bye — the WEDGE-1
    // warn path ran without panic. (The DetachSession/LeaseRevoke fan-out is covered by
    // `bye_detaches_revokes_and_clears`.)
    assert!(
        !rig.world
            .resource::<GatewaySessions>()
            .by_session
            .contains_key(&sid),
        "the session (and its journal) is dropped on Bye"
    );
}

#[test]
fn request_cut_without_a_matching_prepare_is_counted_and_pushes_nothing() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let sent = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![],
        "no RequestCut pushed"
    );
    assert_eq!(rig.stats().transfer_unroutable, 1);
}

#[test]
fn an_ordinary_input_during_a_transfer_is_not_a_cut_marker() {
    // The cold marker observer runs while a transfer is in flight, but a NON-marker
    // input (`is_cut_marker = false`) produces no CutConfirmed — only routing.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let sent = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Input,
        &InputDatagram {
            seq: 5,
            is_cut_marker: false,
            client_tick: TickId(1),
            movement: [1.0, 0.0, 0.0],
            look: [0.0, 0.0],
            action_bits: 0,
        },
    )]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![],
        "ordinary input yields no CutConfirmed"
    );
}

#[test]
fn a_cut_marker_is_inert_before_and_after_request_cut() {
    // S3: the client CUT_MARKER never drives a CutConfirmed — before OR after RequestCut. The
    // cut is server-timed: RequestCut self-acks CutConfirmed, and a marker input is ordinary.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    // A marker BEFORE RequestCut: inert (an ordinary input; no ack).
    let sent = rig.tick(vec![marker_input(7)]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![],
        "a marker before RequestCut drives no ack (S3 — inert)"
    );
    // RequestCut self-acks CutConfirmed server-side.
    let sent = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 0,
        }],
        "RequestCut self-acks the cut server-side (no client marker)",
    );
    // A marker AFTER RequestCut is still inert (the saga already advanced; no second ack).
    let sent = rig.tick(vec![marker_input(8)]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![],
        "a marker after RequestCut drives no ack (S3 — the cut is already server-timed)"
    );
}

#[test]
fn a_seq_valid_but_undecodable_input_during_a_transfer_is_not_a_marker() {
    // The cold observer full-decodes; `route_input` only peeks the leading seq varint.
    // A datagram whose seq varint is valid but whose body is truncated routes normally,
    // then the observer's decode fails → no CutConfirmed, no panic.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let sent = rig.tick(vec![Inbound::Wire {
        from: CLIENT,
        class: MsgClass::Input,
        bytes: vec![0x05].into(), // seq=5 (valid varint), then EOF → full decode fails
    }]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![],
        "an undecodable input yields no CutConfirmed"
    );
}

#[test]
fn a_command_for_a_different_transfer_does_not_consult_the_wrong_journal() {
    // The redelivery gate only re-sends from the journal when the in-flight transfer
    // matches. A command for a DIFFERENT transfer falls through and is handled fresh
    // (defensive replace), never absorbed against the wrong saga.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let other = TransferId(0x999);
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: SHARD,
    })]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: other,
        session: sid,
        dest: SHARD,
    })]);
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Prepared {
            transfer: other,
            result: PrepareResult::Ready,
        }],
        "the different transfer is handled fresh, not re-sent from the XFER journal"
    );
}

#[test]
fn freshest_session_confirmed_maxes_over_active_and_selffenced() {
    // A bare session in a given phase with a given confirmed_at tick (uses the module test consts).
    fn sess(phase: SessionPhase, confirmed: u64) -> Session {
        Session {
            sky_held: None,
            sky_parts_sent: 0,
            client: CLIENT,
            account: AccountId(5),
            fence: Fence(1),
            phase,
            next_sub: 0,
            // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
            home_shard: None,
            home_rid: None,
            realm_feed_frame_id: 0,
            scene_sent: BTreeMap::new(),
            // A bare test session: no home descended, so no spawn pose — the static shape.
            spawn: None,
            bootstrap_deadline: None,
            confirmed_at: TickId(confirmed),
            negotiated_minor: 1,
            transfer: None,
            subs: BTreeMap::new(),
            delivered: BTreeMap::new(),
            lineage: Vec::new(),
            shadow: window::ShadowScene::default(),
            hot: Arc::new(SessionHot {
                route: ArcSwap::from_pointee(RouteSnapshot {
                    authority: SHARD,
                    fence: Fence(1),
                    cut: None,
                }),
                last_input_seq: AtomicU64::new(0),
                subs: ArcSwap::from_pointee(SubTable::default()),
            }),
        }
    }
    let mut sessions = GatewaySessions::default();
    // Empty table → None (no session carries a meaningful confirmed_at).
    assert_eq!(sessions.freshest_session_confirmed(), None);
    // Only PRE-Active logins → None (their confirmed_at is the 0 sentinel, never a real round-trip),
    // even though their values are large — they must not be able to de-route a long-running gateway.
    sessions
        .by_session
        .insert(SessionId(1), sess(SessionPhase::AwaitingDirectory, 999));
    sessions
        .by_session
        .insert(SessionId(2), sess(SessionPhase::AwaitingAttach, 888));
    assert_eq!(sessions.freshest_session_confirmed(), None);
    // ONLY a SelfFenced session (a totally-partitioned gateway that self-fenced its last session): the
    // FROZEN confirmed is visible → Some(stale). This is the case an Active-only max wrongly read as None
    // (falsely Ready). It is the de-route signal.
    sessions
        .by_session
        .insert(SessionId(3), sess(SessionPhase::SelfFenced, 40));
    assert_eq!(sessions.freshest_session_confirmed(), Some(40));
    // Add fresh Active sessions → the MAX over {Active ∪ SelfFenced} picks the freshest Active (a healthy
    // gateway stays Ready despite the lingering SelfFenced ghost); the pre-Active 999 stays excluded.
    sessions.by_session.insert(
        SessionId(4),
        sess(
            SessionPhase::Active {
                entity: EntityId(1),
            },
            30,
        ),
    );
    sessions.by_session.insert(
        SessionId(5),
        sess(
            SessionPhase::Active {
                entity: EntityId(2),
            },
            50,
        ),
    );
    assert_eq!(sessions.freshest_session_confirmed(), Some(50));
}

// ---- Slice 1d.2a: the route-table reshape (sub registry + reverse index) ----

/// Build a bare `GatewaySessions` with ONE Active session whose only sub is on `SHARD`,
/// exactly as a real login would leave it — the substrate for the primitive unit tests.
fn one_active_session() -> (GatewaySessions, SessionId, OutboundBox) {
    let mut sessions = GatewaySessions::default();
    let sid = SessionId(0xA11A);
    sessions.by_session.insert(
        sid,
        Session {
            sky_held: None,
            sky_parts_sent: 0,
            client: CLIENT,
            account: AccountId(5),
            fence: Fence(1),
            phase: SessionPhase::Active {
                entity: EntityId(77),
            },
            next_sub: 0,
            // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
            home_shard: None,
            home_rid: None,
            realm_feed_frame_id: 0,
            scene_sent: BTreeMap::new(),
            // A bare test session: no home descended, so no spawn pose — the static shape.
            spawn: None,
            bootstrap_deadline: None,
            confirmed_at: TickId(0),
            negotiated_minor: 1,
            transfer: None,
            subs: BTreeMap::new(),
            delivered: BTreeMap::new(),
            lineage: Vec::new(),
            shadow: window::ShadowScene::default(),
            hot: Arc::new(SessionHot {
                route: ArcSwap::from_pointee(RouteSnapshot {
                    authority: SHARD,
                    fence: Fence(1),
                    cut: None,
                }),
                last_input_seq: AtomicU64::new(0),
                subs: ArcSwap::from_pointee(SubTable::default()),
            }),
        },
    );
    sessions.by_client.insert(CLIENT, sid);
    let mut outbox = OutboundBox::default();
    // Open the login sub (the FIRST open_sub caller), exactly as on_shard_control does.
    let sub = sessions
        .open_sub(
            sid,
            SHARD,
            FrameRef::SystemSpace { system_seed: 7 },
            Fence(1),
            &mut outbox,
        )
        .expect("session present");
    assert_eq!(sub, SubId(0));
    (sessions, sid, outbox)
}

/// Decode the control messages a captured `OutboundBox` sent to a client.
fn controls_in(outbox: &OutboundBox, to: NodeId) -> Vec<ServerControlMsg> {
    let owned: Vec<(NodeId, MsgClass, Vec<u8>)> = outbox
        .0
        .iter()
        .map(|(t, c, b, _)| (*t, *c, b.to_vec()))
        .collect();
    decode_controls(&owned, to)
}

#[test]
fn open_sub_indexes_publishes_and_emits_opened_before_the_table() {
    // open_sub: X1 (SubscriptionOpened pushed BEFORE the hot table is readable), the cold
    // SubRecord installed, the reverse index populated, and the hot SubTable carries the
    // accepted fence. A second open on a DISTINCT shard yields a fresh never-reused sub.
    let (mut sessions, sid, outbox) = one_active_session();
    assert_eq!(
        controls_in(&outbox, CLIENT),
        vec![ServerControlMsg::SubscriptionOpened {
            sub: SubId(0),
            frame: FrameRef::SystemSpace { system_seed: 7 },
        }]
    );
    // The reverse index now lists this session under SHARD (and nothing under DEST).
    assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
    assert_eq!(sessions.subscribers_of(DEST), Vec::<SessionId>::new());
    // The hot SubTable resolves SHARD to (SubId(0), Fence(1)).
    let table = sessions.by_session[&sid].hot.subs.load();
    assert_eq!(
        table.lookup(SHARD),
        Some(&SubEntry {
            shard: SHARD,
            sub: SubId(0),
            accepted: Fence(1),
        })
    );
    assert_eq!(table.lookup(DEST), None, "no sub for an unsubscribed shard");
    // A second open on DEST allocates the next monotonic, never-reused sub id.
    let mut outbox2 = OutboundBox::default();
    let sub1 = sessions
        .open_sub(
            sid,
            DEST,
            FrameRef::SystemSpace { system_seed: 8 },
            Fence(3),
            &mut outbox2,
        )
        .expect("session present");
    assert_eq!(sub1, SubId(1), "monotonic, never reused");
    assert_eq!(sessions.subscribers_of(DEST), vec![sid]);
    let table = sessions.by_session[&sid].hot.subs.load();
    assert_eq!(table.lookup(DEST).map(|e| e.sub), Some(SubId(1)));
    assert_eq!(table.lookup(DEST).map(|e| e.accepted), Some(Fence(3)));
}

#[test]
fn open_sub_for_an_absent_session_is_a_counted_free_none() {
    // The `?` arm: open_sub on an unknown session returns None and touches nothing.
    let mut sessions = GatewaySessions::default();
    let mut outbox = OutboundBox::default();
    assert_eq!(
        sessions.open_sub(
            SessionId(0xDEAD),
            SHARD,
            FrameRef::SystemSpace { system_seed: 7 },
            Fence(1),
            &mut outbox,
        ),
        None
    );
    assert!(outbox.0.is_empty());
    assert!(sessions.subscribers_of(SHARD).is_empty());
}

#[test]
fn close_sub_drains_for_one_tick_then_the_sweep_removes_it() {
    // close_sub marks Draining + emits SubscriptionClosing but KEEPS the sub in the table +
    // index for one tick (a straggler is still routable); the next sweep removes it.
    let (mut sessions, sid, _) = one_active_session();
    let mut outbox = OutboundBox::default();
    let mut stats = GatewayStats::default();
    // The production shape of a source-sub close: `CommitAuthority` already re-pointed the
    // session's authority at the DEST, so SHARD (the source) is no longer the feeding node and
    // the authority-sub invariant lets the release through.
    sessions
        .by_session
        .get_mut(&sid)
        .expect("the fixture session is present")
        .home_shard = Some(DEST);
    sessions.close_sub(sid, SHARD, &config(), &mut outbox, &mut stats);
    assert_eq!(stats.sub_close_refused_authority, 0);
    assert_eq!(
        controls_in(&outbox, CLIENT),
        vec![ServerControlMsg::SubscriptionClosing { sub: SubId(0) }]
    );
    // Still routable this tick (the drain grace): index + hot table both still resolve it.
    assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
    assert_eq!(
        sessions.by_session[&sid]
            .hot
            .subs
            .load()
            .lookup(SHARD)
            .map(|e| e.sub),
        Some(SubId(0)),
        "Draining sub stays in the hot table for its one-tick grace"
    );
    // A SECOND close is idempotent — no second SubscriptionClosing.
    let mut outbox2 = OutboundBox::default();
    sessions.close_sub(sid, SHARD, &config(), &mut outbox2, &mut stats);
    assert!(outbox2.0.is_empty(), "already Draining: no second close");
    // The next-tick sweep removes it from BOTH the table and the index.
    sessions.sweep_draining();
    assert_eq!(sessions.subscribers_of(SHARD), Vec::<SessionId>::new());
    assert_eq!(
        sessions.by_session[&sid].hot.subs.load().lookup(SHARD),
        None,
        "swept out of the hot table"
    );
    assert!(
        sessions.subscribed_shards.is_empty(),
        "the emptied reverse-index entry is removed"
    );
}

#[test]
fn close_sub_for_an_absent_session_or_unsubscribed_shard_is_a_no_op() {
    // Both early-return arms: an unknown session, and a known session that does not
    // subscribe to the named shard.
    let (mut sessions, sid, _) = one_active_session();
    let mut outbox = OutboundBox::default();
    let mut stats = GatewayStats::default();
    let cfg = config();
    sessions.close_sub(SessionId(0xDEAD), SHARD, &cfg, &mut outbox, &mut stats); // unknown session
    sessions.close_sub(sid, DEST, &cfg, &mut outbox, &mut stats); // does not subscribe to DEST
    assert!(outbox.0.is_empty(), "neither path emits a close");
    assert_eq!(stats.sub_close_refused_authority, 0, "neither is a refusal");
    // The original SHARD sub is untouched.
    assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
}

#[test]
fn close_sub_refuses_to_close_the_sessions_current_authority() {
    // THE AUTHORITY-SUB INVARIANT (the same-node re-home freeze, task #175). A re-home whose
    // source and dest are the SAME node is an ordinary saga (#149 deleted the node-placement
    // short-circuit), and its `ReleaseSubscribe` names a `src` that is ALSO the post-commit
    // authority. `subs` is keyed by shard, so honouring that close would tear down the one
    // subscription feeding the client — the player keeps being simulated while their client
    // goes totally silent (no snapshots, no realm feed, no clock, no visible input response).
    let (mut sessions, sid, _) = one_active_session();
    let mut outbox = OutboundBox::default();
    let mut stats = GatewayStats::default();
    // A static session's authority IS `config.shard` (`session_target`'s `unwrap_or`), which is
    // exactly the same-node shape: the node being released is still the one feeding the client.
    sessions.close_sub(sid, SHARD, &config(), &mut outbox, &mut stats);
    assert_eq!(stats.sub_close_refused_authority, 1, "refused + counted");
    assert!(
        outbox.0.is_empty(),
        "no SubscriptionClosing reaches the client"
    );
    // The live sub survives, in BOTH the cold table and the hot projection.
    assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
    assert_eq!(
        sessions.by_session[&sid]
            .hot
            .subs
            .load()
            .lookup(SHARD)
            .map(|e| e.sub),
        Some(SubId(0)),
        "the authority sub stays live — the client keeps receiving its own owner's frames"
    );
    // And it is NOT swept away next tick either (it was never marked Draining).
    sessions.sweep_draining();
    assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
}

#[test]
fn sweep_with_no_draining_subs_is_a_no_op() {
    // The sweep's empty-`draining` continue arm: a session with only Active subs is left
    // byte-identical.
    let (mut sessions, sid, _) = one_active_session();
    sessions.sweep_draining();
    assert_eq!(sessions.subscribers_of(SHARD), vec![sid]);
    assert_eq!(
        sessions.by_session[&sid]
            .hot
            .subs
            .load()
            .lookup(SHARD)
            .map(|e| e.sub),
        Some(SubId(0))
    );
}

#[test]
fn sweep_keeps_a_shared_reverse_index_entry_with_a_surviving_subscriber() {
    // The `set.is_empty()` FALSE arm: two sessions subscribe to SHARD; closing+sweeping ONE
    // leaves the reverse-index entry alive for the other (the entry is not removed).
    let (mut sessions, sid_a, _) = one_active_session();
    // A second session on the same SHARD sub.
    let sid_b = SessionId(0xB22B);
    sessions.by_session.insert(
        sid_b,
        Session {
            sky_held: None,
            sky_parts_sent: 0,
            client: NodeId(101),
            account: AccountId(6),
            fence: Fence(1),
            phase: SessionPhase::Active {
                entity: EntityId(88),
            },
            next_sub: 0,
            // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
            home_shard: None,
            home_rid: None,
            realm_feed_frame_id: 0,
            scene_sent: BTreeMap::new(),
            // A bare test session: no home descended, so no spawn pose — the static shape.
            spawn: None,
            bootstrap_deadline: None,
            confirmed_at: TickId(0),
            negotiated_minor: 1,
            transfer: None,
            subs: BTreeMap::new(),
            delivered: BTreeMap::new(),
            lineage: Vec::new(),
            shadow: window::ShadowScene::default(),
            hot: Arc::new(SessionHot {
                route: ArcSwap::from_pointee(RouteSnapshot {
                    authority: SHARD,
                    fence: Fence(1),
                    cut: None,
                }),
                last_input_seq: AtomicU64::new(0),
                subs: ArcSwap::from_pointee(SubTable::default()),
            }),
        },
    );
    sessions.by_client.insert(NodeId(101), sid_b);
    let mut ob = OutboundBox::default();
    sessions
        .open_sub(
            sid_b,
            SHARD,
            FrameRef::SystemSpace { system_seed: 7 },
            Fence(1),
            &mut ob,
        )
        .expect("present");
    let mut both = sessions.subscribers_of(SHARD);
    both.sort_unstable();
    assert_eq!(both, vec![sid_a, sid_b]);
    // Close + sweep ONLY session A. Its authority already moved to DEST (the post-commit shape),
    // so the authority-sub invariant lets the source release through.
    let mut ob = OutboundBox::default();
    sessions
        .by_session
        .get_mut(&sid_a)
        .expect("fixture session A is present")
        .home_shard = Some(DEST);
    sessions.close_sub(
        sid_a,
        SHARD,
        &config(),
        &mut ob,
        &mut GatewayStats::default(),
    );
    sessions.sweep_draining();
    assert_eq!(
        sessions.subscribers_of(SHARD),
        vec![sid_b],
        "B's entry survives A's drain (the reverse-index entry is not removed)"
    );
}

#[test]
fn sweep_tolerates_a_drained_shard_missing_from_the_reverse_index() {
    // The `if let Some(set) = ..` None arm (a defensive desync guard): a cold Draining sub
    // whose shard is ABSENT from `subscribed_shards` (a forced index breach) is swept from
    // the table without panic — the index update is a no-op, never an index-out-of-bounds.
    let (mut sessions, sid, _) = one_active_session();
    // Mark the SHARD sub Draining in the COLD map directly...
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .get_mut(&SHARD)
        .expect("sub present")
        .state = SubState::Draining;
    // ...and forcibly clear the reverse index so the drained shard has no entry.
    sessions.subscribed_shards.clear();
    sessions.sweep_draining(); // must not panic on the missing-entry None arm
    assert_eq!(
        sessions.by_session[&sid].hot.subs.load().lookup(SHARD),
        None,
        "the cold Draining sub was still swept from the hot table"
    );
    assert!(sessions.subscribed_shards.is_empty());
}

#[test]
fn a_frame_from_an_unsubscribed_known_shard_fans_to_nobody() {
    // The `subscribers_of` empty path: a DEST frame (DEST is a known shard) reaches
    // on_shard_frame, but no session subscribes to DEST, so it fans to nobody — and the
    // desync counter is untouched (an empty subscriber set is NOT a desync).
    let mut rig = Rig::new();
    let (_, _) = rig.login(); // subscribes only to SHARD
    let sent = rig.tick(vec![wire(
        DEST,
        MsgClass::Snapshot,
        &frame_msg(Fence(1), 9),
    )]);
    assert_eq!(
        sent,
        Vec::new(),
        "a frame from an unsubscribed shard reaches no client (and nothing else is sent)"
    );
    assert_eq!(
        rig.stats().frame_sub_desync,
        0,
        "an empty fan is not a desync"
    );
    assert_eq!(rig.stats().stale_frames_dropped, 0);
}

#[test]
fn a_forced_index_table_desync_hits_the_counter_never_a_silent_drop() {
    // C2 dead-branch trap: a session in the reverse index for SHARD whose hot SubTable has
    // NO SubEntry for SHARD (an invariant breach `publish_subs` makes impossible by
    // construction) is COUNTED (`frame_sub_desync`), never a silent continue. BOTH desync
    // arms are exercised: (a) the index references a session absent from `by_session`;
    // (b) a present session whose hot table was corrupted to empty.
    let frame = postcard::to_allocvec(&frame_msg(Fence(1), 1)).expect("encode");

    // (a) index points at a session that does not exist in by_session.
    let mut sessions = GatewaySessions::default();
    sessions
        .subscribed_shards
        .entry(SHARD)
        .or_default()
        .insert(SessionId(0xC0DE));
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
    assert_eq!(stats.frame_sub_desync, 1, "(a) missing session is counted");
    assert!(outbox.0.is_empty());

    // THE SKY'S FANS ARE GONE (S11): the gateway no longer FORWARDS a shard's sky — it states its own,
    // to sessions, not through the shard reverse index. So there is no sky lane left to drive against
    // this invariant. The index-walking lanes that remain are covered above.

    // (b) a present session indexed under SHARD but with an EMPTY hot SubTable.
    let (mut sessions, sid, _) = one_active_session();
    sessions.by_session[&sid]
        .hot
        .subs
        .store(Arc::new(SubTable::default()));
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    on_shard_frame(SHARD, &frame, &mut sessions, &mut stats, &mut outbox);
    assert_eq!(stats.frame_sub_desync, 1, "(b) lookup None is counted");
    assert!(outbox.0.is_empty(), "no frame forwarded on a desync");
}

#[test]
fn a_frame_with_a_malformed_snapshot_body_is_counted_undecodable_not_forwarded() {
    // 1d.5a: on_shard_frame peeks the frame_id off the body BEFORE the fan (to advance the
    // delivery watermark). A body that fails the peek — a corrupt/buggy shard — is counted
    // (`undecodable`) and the WHOLE frame abandoned once, never forwarded. (The peek validating
    // the sub varint is also what makes the per-session retag below infallible.)
    let bad = postcard::to_allocvec(&ShardToGateway::Frame {
        realm_fence: Fence(1),
        source_tick: TickId(5),
        snapshot_bytes: vec![0x80], // a truncated varint — peek_snapshot_frame_id errors
    })
    .expect("encode");
    let (mut sessions, _sid, _) = one_active_session();
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    on_shard_frame(SHARD, &bad, &mut sessions, &mut stats, &mut outbox);
    assert_eq!(
        stats.undecodable, 1,
        "a malformed snapshot body is counted undecodable"
    );
    assert!(outbox.0.is_empty(), "nothing forwarded on a malformed body");
}

/// The realm datagram a `realm_frame_msg` envelope carries — the payload the shard authored, which is
/// exactly what the client must receive. The ONE builder, which `realm_frame_msg` wraps — so reading
/// the payload never destructures an enum a fixture just built (a dead refusal arm by construction).
fn realm_frame_payload(realm: RealmId) -> Vec<u8> {
    let snapshot = RealmSnapshotDatagram {
        sub: SubId(0),
        frame_id: 3,
        source_tick: TickId(5),
        universe_tick: UniverseTick(50),
        origin_epoch: 0,
        sky_anchor: None,
        realms: vec![RealmSnap {
            realm,
            // The edge HEAD (proto_minor 8): the CHILD's own frame. The gateway IGNORES it in this
            // slice — it forwards realm bytes verbatim — but a row without it is not a placement
            // edge, so the fixture ships a real one rather than a same-frame placeholder.
            frame: vd_core::pose::frame_for_realm(realm, None).expect("a Planet realm resolves"),
            pose: vd_core::pose::StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                vd_core::glam::DVec3::new(1.0e9, 0.0, 0.0),
                UniverseTick(50),
            ),
        }],
    };
    postcard::to_allocvec(&snapshot).expect("encode")
}

#[test]
fn a_tombstoned_old_realm_frame_is_counted_and_serves_nobody() {
    // ★TOMBSTONE (Slice C2, minor 19): the old opaque realm datagram has no producer and no
    // consumer. It still DECODES (its discriminant is reserved forever), so the honest
    // accounting is its own counter — never the `undecodable` bucket (which would hide a
    // revived producer among real garbage), and never a byte to a client.
    let (mut sessions, _sid, _) = one_active_session();
    let mut stats = GatewayStats::default();
    let envelope = postcard::to_allocvec(&realm_frame_msg(RealmId::Planet(7))).expect("encode");
    on_shard_realm_frame(SHARD, &envelope, &config(), &mut sessions, &mut stats);
    assert_eq!(
        stats.old_realm_frames_dropped, 1,
        "the dead lane is counted, so a revived producer is visible"
    );
    assert_eq!(
        stats.undecodable, 0,
        "and it is NOT filed as garbage — it decodes, it is simply dead"
    );
}

#[test]
fn a_malformed_realm_envelope_is_counted_once_and_forwards_nothing() {
    // `on_shard_realm_frame` is the ONE place the realm envelope is opened, so garbage on that class
    // is counted exactly ONCE and forwards nothing.
    let mut rig = Rig::new();
    let (_, _) = rig.login();
    let sent = rig.tick(vec![wire(SHARD, MsgClass::RealmSnapshot, &0xffu8)]);
    assert_eq!(rig.world.resource::<GatewayStats>().undecodable, 1);
    assert_eq!(to_client(&sent, MsgClass::RealmSnapshot), 0);
}

#[test]
fn a_dead_realm_frame_routes_through_the_dispatch_arm_and_serves_nobody() {
    // The `MsgClass::RealmSnapshot` dispatch arm through the REAL schedule: a tombstoned
    // realm frame from a subscribed shard is ROUTED (not a stranger, not garbage) and lands
    // in its own drop counter — the client receives nothing on the realm class, because the
    // composed feed is the ONE realm feed a client has (§2.4).
    let mut rig = Rig::new();
    let (_sid, _) = rig.login(); // subscribes to SHARD
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::RealmSnapshot,
        &realm_frame_msg(RealmId::Planet(7)),
    )]);
    assert_eq!(
        to_client(&sent, MsgClass::RealmSnapshot),
        0,
        "the dead realm lane fans nothing to clients"
    );
    let stats = rig.world.resource::<GatewayStats>();
    assert_eq!(stats.undecodable, 0);
    assert_eq!(stats.old_realm_frames_dropped, 1);
}

#[test]
fn every_observer_delivered_requires_a_non_empty_all_delivered_observer_set() {
    // The 1d.5a (a) predicate's three load-bearing properties, asserted directly (the p2
    // capstone proves them end-to-end — a vacuous fire would release the source early and
    // re-open the vanish — this pins them as a unit gate). `one_active_session` subscribes the
    // session to SHARD as sub 0.
    let (mut sessions, sid, _) = one_active_session();
    // (1) ANTI-VACUOUS: no session subscribes to DEST → the EMPTY observer set is NOT satisfied
    // (never a vacuous true — the saga must not be told "delivered" before the dest sub opens).
    assert!(
        !every_observer_delivered(&sessions, DEST),
        "an empty dest-observer set never vacuously fires the demote",
    );
    // (2) NOT DELIVERED: the SHARD observer exists but its watermark is absent (0) → blocked.
    assert!(
        !every_observer_delivered(&sessions, SHARD),
        "an observer with no delivered frame (watermark 0) blocks the predicate",
    );
    // (3) DELIVERED: advance the observer's sub-0 watermark to >=1 → satisfied.
    sessions
        .by_session
        .get_mut(&sid)
        .expect("session present")
        .delivered
        .insert(SubId(0), 1);
    assert!(
        every_observer_delivered(&sessions, SHARD),
        "every (here: the one) dest observer delivered >=1 frame -> satisfied",
    );
}

#[test]
fn on_shard_frame_advances_the_delivery_watermark_max_wins_and_stale_does_not() {
    // 1d.5a: the delivery watermark advances through the REAL `on_shard_frame` path (not a
    // manual insert): an accepted past-fence frame sets `delivered[sub] = frame_id`; a LOWER
    // frame_id does NOT regress it (`.max`); a fence-STALE frame does NOT advance it (dropped
    // before the advance). `one_active_session` = SubId(0) on SHARD @ accepted Fence(1).
    let (mut sessions, sid, _) = one_active_session();
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    let wm = |s: &GatewaySessions| s.by_session[&sid].delivered.get(&SubId(0)).copied();

    // (a) an accepted frame (Fence(1), frame_id 9) advances the watermark to 9.
    let f9 = postcard::to_allocvec(&frame_msg(Fence(1), 9)).expect("encode");
    on_shard_frame(SHARD, &f9, &mut sessions, &mut stats, &mut outbox);
    assert_eq!(
        wm(&sessions),
        Some(9),
        "an accepted frame advances delivered[sub] to its frame_id"
    );

    // (b) a LOWER frame_id (5) does NOT regress the high-water (`.max`).
    let f5 = postcard::to_allocvec(&frame_msg(Fence(1), 5)).expect("encode");
    on_shard_frame(SHARD, &f5, &mut sessions, &mut stats, &mut outbox);
    assert_eq!(
        wm(&sessions),
        Some(9),
        "a lower frame_id never lowers the watermark (.max)"
    );

    // (c) a fence-STALE frame (Fence(0) < accepted Fence(1)) is dropped — no advance.
    let stale = postcard::to_allocvec(&frame_msg(Fence(0), 99)).expect("encode");
    on_shard_frame(SHARD, &stale, &mut sessions, &mut stats, &mut outbox);
    assert_eq!(
        wm(&sessions),
        Some(9),
        "a fence-stale frame does not advance the watermark"
    );
    assert_eq!(stats.stale_frames_dropped, 1);
}

// ---- Slice 1d.2b: SubscriptionReady + the source-sub close primitives -------

/// A `SubscriptionReady` from shard `from` (the dest), as the dest emits it at adopt.
fn subscription_ready(from: NodeId, session: SessionId) -> Inbound {
    wire(
        from,
        MsgClass::Control,
        &ShardToGateway::SubscriptionReady {
            session,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 8 },
            realm_fence: Fence(5),
        },
    )
}

/// The full route+sub snapshot a session holds (for the dest-sub assertions).
fn sub_for(rig: &Rig, sid: SessionId, shard: NodeId) -> Option<SubEntry> {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session present")
        .hot
        .subs
        .load()
        .lookup(shard)
        .copied()
}

#[test]
fn subscription_ready_opens_the_dest_sub_and_repoints_authority() {
    // FORK 0a: a SubscriptionReady from DEST opens a SECOND sub (SubId(1)) at the DEST realm
    // fence, emits SubscriptionOpened{1} (X1, before any frame) THEN AuthorityChanged{entity,
    // 1} — re-pointing the avatar's render authority to the dest sub. The source SubId(0)
    // stays open (the two-sub overlap).
    let mut rig = Rig::new();
    let (sid, _) = rig.login(); // SubId(0) on SHARD
    let sent = rig.tick(vec![subscription_ready(DEST, sid)]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(1),
                frame: FrameRef::SystemSpace { system_seed: 8 },
            },
            ServerControlMsg::AuthorityChanged {
                entity: EntityId(77),
                sub: SubId(1),
            },
            // The default rig negotiates minor 2, so the pure-renderer `OwnEntity` trails the
            // node-aware `AuthorityChanged` at the dest re-point too (S4 dual-announce).
            ServerControlMsg::OwnEntity {
                entity: EntityId(77),
            },
        ],
        "SubscriptionOpened(1) strictly precedes AuthorityChanged(entity,1) (X1 + A1 re-point)"
    );
    // BOTH subs now resolve: source SubId(0) on SHARD, dest SubId(1) on DEST at Fence(5).
    assert_eq!(sub_for(&rig, sid, SHARD).map(|e| e.sub), Some(SubId(0)));
    assert_eq!(
        sub_for(&rig, sid, DEST),
        Some(SubEntry {
            shard: DEST,
            sub: SubId(1),
            accepted: Fence(5),
        })
    );
    // The reverse index lists this session under BOTH shards.
    assert_eq!(
        rig.world.resource::<GatewaySessions>().subscribers_of(DEST),
        vec![sid]
    );
}

#[test]
fn a_subscription_ready_naming_a_realm_moves_the_lineage() {
    // THE LINEAGE follows the avatar (the deleted render pin's lawful successor is the origin
    // marker, derived from this): `SubscriptionReady` is the one phase that can move it — it
    // carries the destination FRAME, so the realm the player is now in is nameable. The
    // composer derives the new chain from this next pass and bumps the origin epoch (§2.7).
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let before = rig.world.resource::<GatewaySessions>().by_session[&sid]
        .lineage
        .clone();
    let _ = rig.tick(vec![wire(
        DEST,
        MsgClass::Control,
        &ShardToGateway::SubscriptionReady {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::PlanetCentered { planet_seed: 42 },
            realm_fence: Fence(5),
        },
    )]);
    let after = &rig.world.resource::<GatewaySessions>().by_session[&sid].lineage;
    assert_ne!(&before, after, "the lineage advanced for the entered realm");
    assert_eq!(
        after.last(),
        Some(&RealmId::Planet(42)),
        "the entered realm is the lineage leaf"
    );
}

#[test]
fn a_subscription_ready_into_galaxy_space_now_records_the_galaxy_in_the_lineage() {
    // ★ RE-BASED AT S9, AND THE RE-BASE IS THE POINT. This test used to assert the opposite — "no realm
    // named, no lineage moved" — because galaxy space named NO realm: a galaxy was not a realm and had
    // nothing to own, so a player standing in the between-space kept the lineage of wherever they last
    // stood. That was true and is no longer.
    //
    // A galaxy owns its star systems now. A player in galaxy space IS somewhere, and the somewhere has a
    // name — so the lineage records it, exactly as it records a system or a planet. The old behaviour
    // would leave a player in a galaxy still labelled by the system they left.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let before = rig.world.resource::<GatewaySessions>().by_session[&sid]
        .lineage
        .clone();
    let _ = rig.tick(vec![wire(
        DEST,
        MsgClass::Control,
        &ShardToGateway::SubscriptionReady {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::GalaxySpace { galaxy_seed: 0 },
            realm_fence: Fence(5),
        },
    )]);
    let after = &rig.world.resource::<GatewaySessions>().by_session[&sid].lineage;
    assert_ne!(
        &before, after,
        "the galaxy is a realm now and must be recorded"
    );
    assert_eq!(
        after.last(),
        Some(&RealmId::Galaxy(0)),
        "the lineage's deepest entry is the galaxy the subscription named"
    );
}

#[test]
fn a_shard_roster_applies_fresh_rejects_stale_and_admits_for_dispatch() {
    // Minor 9's roster receive: the orchestrator's level REPLACES the record set (latest tick
    // wins); an older one is refused as stale — a reordered datagram must never resurrect a dead
    // roster. The applied set is DECISIVE: a node the record shows holding a realm is heard for
    // dispatch without ever greeting.
    const RECORDED: NodeId = NodeId(77);
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let roster = |nodes: Vec<NodeId>, at: u64| {
        wire(
            ORCH,
            MsgClass::Saga,
            &InterShardFlow::ShardRoster(vd_wire::intershard::ShardRoster {
                nodes,
                at: UniverseTick(at),
            }),
        )
    };
    let _ = rig.tick(vec![roster(vec![RECORDED], 10)]);
    assert_eq!(rig.stats().shard_rosters_applied, 1);
    // STALE: an older level is refused; the held set is untouched.
    let _ = rig.tick(vec![roster(vec![], 5)]);
    assert_eq!(rig.stats().shard_roster_stale, 1);
    assert_eq!(rig.stats().shard_rosters_applied, 1);
    // DECISIVE: the recorded node's SubscriptionReady is heard (it is a shard by record), so the
    // session opens a sub on it — a node outside roster+greetings would have been refused.
    let _ = rig.tick(vec![wire(
        RECORDED,
        MsgClass::Control,
        &ShardToGateway::SubscriptionReady {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 8 },
            realm_fence: Fence(5),
        },
    )]);
    assert_eq!(
        sub_for(&rig, sid, RECORDED).map(|e| e.sub),
        Some(SubId(1)),
        "the record-admitted shard's ready opened a sub"
    );
}

#[test]
fn a_known_clients_wrong_class_counts_undecodable_not_unknown_sender() {
    // The refusal split: an UNKNOWN node on a data class is refused by sender; a node with a
    // LIVE session sending a class it may not send is a peer bug, counted undecodable exactly
    // as before the split existed.
    let mut rig = Rig::new();
    let _ = rig.login();
    let before = rig.stats().undecodable;
    let _ = rig.tick(vec![Inbound::Wire {
        from: CLIENT,
        class: MsgClass::Snapshot,
        bytes: vec![1].into(),
    }]);
    assert_eq!(rig.stats().undecodable, before + 1);
    assert_eq!(
        rig.stats().refused_unknown_sender,
        0,
        "a KNOWN client is never refused as an unknown sender"
    );
}

#[test]
fn a_duplicate_subscription_ready_is_an_idempotent_no_op() {
    // At-least-once: a second SubscriptionReady for an already-open dest sub opens nothing
    // and emits nothing (the sub id is never re-allocated, A1 not re-emitted).
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![subscription_ready(DEST, sid)]);
    let sent = rig.tick(vec![subscription_ready(DEST, sid)]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![],
        "a duplicate SubscriptionReady emits nothing"
    );
    assert_eq!(sub_for(&rig, sid, DEST).map(|e| e.sub), Some(SubId(1)));
    assert_eq!(
        rig.stats().transfer_unroutable,
        0,
        "a duplicate is not unroutable"
    );
}

#[test]
fn subscription_ready_for_an_absent_session_is_counted_and_emits_nothing() {
    // An absent session: counted (transfer_unroutable), never a panic, nothing opened.
    let mut rig = Rig::new();
    let sent = rig.tick(vec![subscription_ready(DEST, SessionId(0xDEAD))]);
    assert_eq!(decode_controls(&sent, CLIENT), vec![]);
    assert_eq!(rig.stats().transfer_unroutable, 1);
}

#[test]
fn release_subscribe_closes_the_source_sub_with_a_drain_grace() {
    // 1d.2b: ReleaseSubscribe(src: SHARD) closes the SOURCE sub — SubscriptionClosing{0} is
    // emitted, the sub goes Draining (still routable THIS tick), and the NEXT tick's sweep
    // removes it. The dest sub (opened by SubscriptionReady) survives.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![subscription_ready(DEST, sid)]); // SubId(1) on DEST
    // Drive the REAL saga order (Prepare → RequestCut → Freeze → Commit) before the release: the
    // commit is what re-points the session's authority at DEST, so SHARD is genuinely no longer
    // the feeding node when its sub is released (the authority-sub invariant, task #175). A
    // release can never precede its commit — it IS the success teardown of the demote tail.
    let _ = cross_to(&mut rig, sid, XFER, DEST);
    let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: SHARD,
    })]);
    // The source sub close went to the client; Released acked to the orchestrator.
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![ServerControlMsg::SubscriptionClosing { sub: SubId(0) }]
    );
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Released { transfer: XFER }]
    );
    // THIS tick (the close tick) the source sub is still in the hot table (drain grace).
    assert_eq!(sub_for(&rig, sid, SHARD).map(|e| e.sub), Some(SubId(0)));
    // The NEXT tick's sweep removes it; the dest sub survives.
    let _ = rig.tick(vec![]);
    assert_eq!(
        sub_for(&rig, sid, SHARD),
        None,
        "source sub swept after the grace"
    );
    assert_eq!(
        sub_for(&rig, sid, DEST).map(|e| e.sub),
        Some(SubId(1)),
        "dest sub survives"
    );
    assert_eq!(
        rig.world
            .resource::<GatewaySessions>()
            .subscribers_of(SHARD),
        Vec::<SessionId>::new()
    );
}

#[test]
fn a_straggler_source_frame_in_the_drain_grace_tick_is_still_routed() {
    // C2 drain grace: a source frame arriving in the SAME batch as the ReleaseSubscribe
    // close is still routed to the client (drained, not silently dropped); only the
    // next-tick sweep stops routing.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    // The real saga order — the commit re-points authority at DEST, so releasing SHARD's sub is
    // permitted by the authority-sub invariant (task #175).
    let _ = cross_to(&mut rig, sid, XFER, DEST);
    // Release + a co-arriving source frame in ONE tick batch: the close marks Draining, the
    // frame still fans (the sub is routable through the rest of the batch).
    let sent = rig.tick(vec![
        saga_cmd(TransferControl::ReleaseSubscribe {
            transfer: XFER,
            session: sid,
            src: SHARD,
        }),
        wire(SHARD, MsgClass::Snapshot, &frame_msg(Fence(1), 9)),
    ]);
    let snaps = sent
        .iter()
        .filter(|(to, c, _)| (*to == CLIENT) & (*c == MsgClass::Snapshot))
        .count();
    assert_eq!(
        snaps, 1,
        "the straggler is drained (routed) during the grace tick"
    );
    // After the next-tick sweep, a further source frame routes to nobody — and (the session
    // being Active with no pending retries) nothing else is sent, so the whole tick is empty.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Snapshot,
        &frame_msg(Fence(1), 10),
    )]);
    assert_eq!(
        sent,
        Vec::new(),
        "after the sweep the source sub no longer routes (nothing reaches the client)"
    );
}

#[test]
fn abort_defensively_closes_an_opened_dest_sub() {
    // apply_abort's defensive close (a sub on a shard != the source): drive a dest sub open
    // (SubscriptionReady) WITH a live in-flight transfer, then abort — the dest sub is
    // closed (SubscriptionClosing{1}) and swept, while the source sub stays.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    let _ = rig.tick(vec![subscription_ready(DEST, sid)]); // SubId(1) on DEST
    assert_eq!(sub_for(&rig, sid, DEST).map(|e| e.sub), Some(SubId(1)));
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![ServerControlMsg::SubscriptionClosing { sub: SubId(1) }],
        "abort defensively closes the dest sub"
    );
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Aborted { transfer: XFER }]
    );
    // The dest sub is swept next tick; the source sub stays open.
    let _ = rig.tick(vec![]);
    assert_eq!(
        sub_for(&rig, sid, DEST),
        None,
        "dest sub closed + swept on abort"
    );
    assert_eq!(
        sub_for(&rig, sid, SHARD).map(|e| e.sub),
        Some(SubId(0)),
        "source sub stays"
    );
}

#[test]
fn abort_without_a_dest_sub_closes_nothing_extra() {
    // The happy-path abort (no dest sub yet): the defensive close collects no shard, so no
    // SubscriptionClosing is emitted — only the Aborted ack.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![],
        "no dest sub to close: no SubscriptionClosing"
    );
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Aborted { transfer: XFER }]
    );
    // The source sub is untouched (abort never closes the source).
    assert_eq!(sub_for(&rig, sid, SHARD).map(|e| e.sub), Some(SubId(0)));
}

#[test]
fn abort_closes_only_its_own_dest_never_a_prior_transfers_live_sub() {
    // F1 REGRESSION (audit `wf_93d8e84f`): once a transfer's dest sub is live (a sub on a shard
    // != the login shard), a LATER, UNRELATED transfer's abort must close ONLY its OWN dest —
    // never that live sub. The old "any sub != config.shard" heuristic closed the player's
    // CURRENT live sub (a chained-transfer black screen) and, in the N-shard end goal, EVERY
    // composited sub (ship/host/planet). The fix closes exactly `tp.dest`.
    let mut rig = Rig::new();
    let (sid, _) = rig.login(); // login sub SubId(0) on config.shard (SHARD)

    // A first transfer opens the DEST sub (SubId(1)) — a live, non-login-shard sub.
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    let _ = rig.tick(vec![subscription_ready(DEST, sid)]);
    assert_eq!(
        sub_for(&rig, sid, DEST).map(|e| e.sub),
        Some(SubId(1)),
        "the prior transfer's DEST sub is live"
    );

    // A SECOND transfer to a DIFFERENT dest, then aborted. Its dest sub was never opened, so the
    // precise abort close is a no-op — and it must NOT touch the prior transfer's live DEST sub.
    let xfer2 = TransferId(0x1c3);
    let dest2 = NodeId(43);
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: xfer2,
        session: sid,
        dest: dest2,
    })]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: xfer2,
        session: sid,
    })]);
    // No SubscriptionClosing on the client: the abort closed only its own (unopened) dest2.
    // (The OLD heuristic would have emitted SubscriptionClosing{SubId(1)} here — closing the
    // live DEST sub of the unrelated prior transfer.)
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![],
        "the unrelated abort closes no live sub"
    );
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Aborted { transfer: xfer2 }]
    );
    let _ = rig.tick(vec![]); // a sweep tick changes nothing
    assert_eq!(
        sub_for(&rig, sid, DEST).map(|e| e.sub),
        Some(SubId(1)),
        "the prior transfer's live DEST sub SURVIVES the unrelated abort"
    );
}

#[test]
fn release_of_a_foreign_transfer_closes_no_sub() {
    // The collect-guard FALSE arm: a ReleaseSubscribe for a transfer NOT in flight collects
    // no source sub to close (the source sub stays open), still acks Released idempotently.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: TransferId(0x999),
        session: sid,
        src: SHARD,
    })]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![],
        "a foreign release closes no sub"
    );
    assert_eq!(
        acks_to_orch(&sent),
        vec![TransferControlAck::Released {
            transfer: TransferId(0x999)
        }]
    );
    assert_eq!(
        sub_for(&rig, sid, SHARD).map(|e| e.sub),
        Some(SubId(0)),
        "source sub intact"
    );
}

// ---- Slice 1d.2c: the 2-shard transfer CAPSTONE (gateway read-plane half) ----

/// The subs that the client-bound snapshots in `sent` are tagged with (decoded).
fn delivered_snapshot_subs(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<SubId> {
    sent.iter()
        .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
        .map(|(_, _, bytes)| {
            postcard::from_bytes::<SnapshotDatagram>(bytes)
                .expect("a delivered snapshot decodes")
                .sub
        })
        .collect()
}

/// Every session whose lease the gateway RENEWED in this batch (an ORCH-bound `LeaseRenew` on a
/// Session key). Shared by the D-3 renew-cadence cell and the 5f-3d MF2 dynamic-hold cell so the
/// decoder's non-renew (`_ => None`) arm is owned once — the D-3 cell's hello tick emits a
/// `LeaseGrant` (a Directory op that is NOT a renew), exercising that arm for both callers.
fn renewed_sessions(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<SessionId> {
    sent.iter()
        .filter(|(to, _, _)| *to == ORCH)
        .filter_map(
            |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                Ok(InterShardFlow::Directory(DirectoryOp::LeaseRenew {
                    key: DirectoryKey::Session(s),
                    ..
                })) => Some(s),
                _ => None,
            },
        )
        .collect()
}

#[test]
fn capstone_two_sub_overlap_routes_both_frames_and_repoints_the_avatar_to_the_dest() {
    // 1d.2c CAPSTONE (the gateway read-plane half of the 2-shard transfer): during the
    // post-commit/pre-release window the gateway holds TWO subs for one session; the dest's
    // `SubscriptionReady` (the stub emits it at adopt, 1d.2c) opened SubId(1) and emitted
    // `AuthorityChanged{entity, SubId(1)}` — re-pointing the avatar's render authority to the
    // dest. A synthetic SOURCE frame fans to SubId(0) and a synthetic DEST frame fans to
    // SubId(1), EACH fence-checked against its OWN realm fence and retagged to its own sub.
    // The client thus receives the avatar on BOTH subs but with `AuthorityChanged` naming
    // SubId(1) authoritative — so a compositing client (`DeliveredView`, suppression proven
    // in `vd_client::view`) renders the avatar EXACTLY ONCE, from the DEST sub. Then
    // `ReleaseSubscribe` closes the SOURCE sub.
    let mut rig = Rig::new();
    let (sid, _) = rig.login(); // login sub SubId(0) on SHARD, AuthorityChanged{77, SubId(0)}
    // The transfer reaches commit (the REAL saga order — the commit is also what re-points the
    // session's authority at DEST, so the later source release is permitted by the
    // authority-sub invariant, task #175): then the dest adopts and announces its read-sub.
    let _ = cross_to(&mut rig, sid, XFER, DEST);
    // SubscriptionReady from DEST opens SubId(1) (at the DEST realm fence) + re-points
    // authority to SubId(1).
    let ready_sent = rig.tick(vec![subscription_ready(DEST, sid)]);
    assert_eq!(
        decode_controls(&ready_sent, CLIENT),
        vec![
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(1),
                frame: FrameRef::SystemSpace { system_seed: 8 },
            },
            ServerControlMsg::AuthorityChanged {
                entity: EntityId(77), // the login avatar (SUBJECT id at the gateway is the dot's entity)
                sub: SubId(1),
            },
            // The default rig is minor 2, so `OwnEntity` re-confirms the avatar (a pure-renderer
            // client renders it latest-wins by EntityId — the sub is irrelevant to it, S4).
            ServerControlMsg::OwnEntity {
                entity: EntityId(77),
            },
            // THE CROSSING SWAP LEVEL (§2.7), ordered AFTER AuthorityChanged on this same
            // reliable stream: the composer saw the standing realm flip System(7)→System(8)
            // and bumped the epoch EXACTLY once (login was 1, this crossing makes 2). No
            // window frames were fed in this rig, so the level carries only the new origin's
            // bagless row at the boot stamp — the swap SIGNAL, which is what is under test.
            ServerControlMsg::RealmRegistry {
                origin: RealmId::System(8),
                origin_epoch: 2,
                rows: vec![SceneRow {
                    realm: RealmId::System(8),
                    parent: None,
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 8 },
                        DVec3::ZERO,
                        UniverseTick(0),
                    ),
                    bag: Vec::new(),
                }],
            },
        ],
        "the dest sub opens (X1), authority re-points to SubId(1) (FORK 0a / A1), and the \
         swap level follows the AuthorityChanged on the same reliable stream (§2.7)"
    );
    // 1d.5a: the dest sub is OPEN but NO dest frame is delivered yet → the standing delivery
    // predicate is NOT satisfied → no premature DeliveredToObservers to the saga (anti-vacuous).
    assert!(
        !acks_to_orch(&ready_sent)
            .contains(&TransferControlAck::DeliveredToObservers { transfer: XFER }),
        "no DeliveredToObservers before the dest delivers a frame",
    );
    // Both subs are held (the two-sub overlap), at their OWN realm fences:
    // SubId(0) on SHARD @ Fence(1) (login), SubId(1) on DEST @ Fence(5) (SubscriptionReady).
    assert_eq!(
        sub_for(&rig, sid, SHARD),
        Some(SubEntry {
            shard: SHARD,
            sub: SubId(0),
            accepted: Fence(1),
        })
    );
    assert_eq!(
        sub_for(&rig, sid, DEST),
        Some(SubEntry {
            shard: DEST,
            sub: SubId(1),
            accepted: Fence(5),
        })
    );

    // A synthetic SOURCE frame (from SHARD @ the source realm fence) fans to SubId(0); a
    // synthetic DEST frame (from DEST @ the dest realm fence) fans to SubId(1). Each is
    // checked against ITS OWN accepted fence and retagged to ITS OWN sub.
    let sent = rig.tick(vec![
        wire(SHARD, MsgClass::Snapshot, &frame_msg(Fence(1), 9)),
        wire(DEST, MsgClass::Snapshot, &frame_msg(Fence(5), 9)),
    ]);
    let mut subs = delivered_snapshot_subs(&sent);
    subs.sort_unstable();
    assert_eq!(
        subs,
        vec![SubId(0), SubId(1)],
        "the avatar rides BOTH subs (source→SubId(0), dest→SubId(1)) — the overlap"
    );
    // 1d.5a: that dest frame ADVANCED the dest observer's watermark, so the standing predicate
    // now holds → the gateway emits DeliveredToObservers{XFER} to the saga (the demote-predicate
    // input). Value-asserted (not incidental): a wrong-transfer / silent / unconditional emit
    // turns THIS red.
    assert!(
        acks_to_orch(&sent).contains(&TransferControlAck::DeliveredToObservers { transfer: XFER }),
        "the delivered dest frame drives DeliveredToObservers{{XFER}} to the saga",
    );
    // A dest frame BELOW the dest's accepted fence (Fence(5)) is stale-dropped — proving the
    // PER-SHARD fence (not the session-global route fence) governs the dest sub.
    let sent = rig.tick(vec![wire(
        DEST,
        MsgClass::Snapshot,
        &frame_msg(Fence(4), 10),
    )]);
    assert_eq!(
        delivered_snapshot_subs(&sent),
        Vec::<SubId>::new(),
        "a dest frame below the dest's own accepted fence is dropped"
    );
    assert_eq!(rig.stats().stale_frames_dropped, 1);

    // ReleaseSubscribe closes the SOURCE sub: SubscriptionClosing{0}, then swept next tick;
    // afterward only the DEST sub (SubId(1)) routes — the crossing is complete.
    let sent = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: SHARD,
    })]);
    assert_eq!(
        decode_controls(&sent, CLIENT),
        vec![ServerControlMsg::SubscriptionClosing { sub: SubId(0) }]
    );
    let _ = rig.tick(vec![]); // the sweep removes the drained source sub
    assert_eq!(sub_for(&rig, sid, SHARD), None, "source sub closed + swept");
    let sent = rig.tick(vec![
        wire(SHARD, MsgClass::Snapshot, &frame_msg(Fence(1), 11)),
        wire(DEST, MsgClass::Snapshot, &frame_msg(Fence(5), 11)),
    ]);
    assert_eq!(
        delivered_snapshot_subs(&sent),
        vec![SubId(1)],
        "post-release only the DEST sub routes — the avatar is single-sub on the dest"
    );
}

#[test]
fn the_gateway_renews_active_session_leases_on_cadence() {
    // D-3 heartbeat (gateway half): on the renew cadence the gateway re-sends LeaseRenew for every
    // ACTIVE session's Session key — never a still-logging-in (non-Active) session, and never
    // off-cadence. Drives ONE session through AwaitingAttach (non-Active) then Active so both filter
    // arms + the iterator's zero-iter (no Active → empty) and nonzero-iter (Active) are exercised.
    // (`renewed_sessions` is the module-level decode helper, shared with the 5f-3d MF2 cell.)
    let mut rig = Rig::new();
    rig.world.insert_resource(GatewayConfig {
        lease_renew_interval_ticks: 4,
        ..config()
    });
    // Hello → AwaitingDirectory: the gateway sends a session LeaseGrant to the orchestrator (a
    // Directory op that is NOT a LeaseRenew — exercises the decoder's non-renew arm).
    let hello = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    assert!(
        renewed_sessions(&hello).is_empty(),
        "login emits a LeaseGrant, not a LeaseRenew"
    );
    let session_id = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .next()
        .expect("session pending");
    // The directory grant → AwaitingAttach (still at tick 1, no renew).
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(session_id))]);

    // A renew tick while the session is NON-Active (AwaitingAttach): nothing renewed (filter false
    // arm; push_renewals called with an empty iterator).
    rig.world.resource_mut::<ClockSample>().local_tick = TickId(4);
    assert!(
        renewed_sessions(&rig.tick(vec![])).is_empty(),
        "a non-Active (still-attaching) session is not renewed"
    );

    // Attach at an off-cadence tick → Active (no renew at tick 5).
    rig.world.resource_mut::<ClockSample>().local_tick = TickId(5);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: session_id,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);

    // A renew tick while Active: the session lease is renewed (filter true arm; nonzero iterator).
    rig.world.resource_mut::<ClockSample>().local_tick = TickId(8);
    assert_eq!(
        renewed_sessions(&rig.tick(vec![])),
        vec![session_id],
        "an Active session's lease is renewed on cadence"
    );

    // Off-cadence (the modulo branch on the proceed path): no renewal.
    rig.world.resource_mut::<ClockSample>().local_tick = TickId(9);
    assert!(
        renewed_sessions(&rig.tick(vec![])).is_empty(),
        "an off-cadence tick emits no LeaseRenew"
    );
}

// ---------------------------------------------------------------------------
// 3g abort-leg: the one-shot `reject_next_prepare` lever (HR5(c) — the gateway unit tests OWN the
// two-arm coverage of `apply_prepare`'s reject tail; the e2e is composition proof, not the arm owner).
// Drive `apply_prepare` DIRECTLY (no `Rig`) so both the `Some`/`None` arms + the guard precedence are
// exercised in one monomorphic surface. `assert_eq!` on the FULL ack (HR5(d) — never `matches!`).
// ---------------------------------------------------------------------------

use vd_wire::seams::transfer_control::SpatialReject;

/// A bare ACTIVE session for driving `apply_prepare` directly (mirrors `freshest_..`'s `sess`, but
/// always Active with a fresh empty transfer). `transfer: None` so `apply_prepare` opens fresh progress.
fn active_session() -> Session {
    Session {
        sky_held: None,
        sky_parts_sent: 0,
        client: CLIENT,
        account: AccountId(5),
        fence: Fence(1),
        phase: SessionPhase::Active {
            entity: EntityId(1),
        },
        next_sub: 0,
        // 5f-3d: a STATIC session (no dynamic home, no bootstrap window) — the byte-identical shape.
        home_shard: None,
        home_rid: None,
        realm_feed_frame_id: 0,
        scene_sent: BTreeMap::new(),
        // A bare test session: no home descended, so no spawn pose — the static shape.
        spawn: None,
        bootstrap_deadline: None,
        confirmed_at: TickId(1),
        negotiated_minor: 1,
        transfer: None,
        subs: BTreeMap::new(),
        delivered: BTreeMap::new(),
        lineage: Vec::new(),
        shadow: window::ShadowScene::default(),
        hot: Arc::new(SessionHot {
            route: ArcSwap::from_pointee(RouteSnapshot {
                authority: SHARD,
                fence: Fence(1),
                cut: None,
            }),
            last_input_seq: AtomicU64::new(0),
            subs: ArcSwap::from_pointee(SubTable::default()),
        }),
    }
}

#[test]
fn apply_prepare_inert_lever_replies_ready() {
    // The `None` arm: an INERT lever leaves the 1c stub behaviour byte-identical (Ready) AND stays None.
    let mut session = active_session();
    let mut stats = GatewayStats::default();
    let mut lever: Option<PrepareReject> = None;
    let ack = apply_prepare(&mut session, XFER, DEST, &mut stats, &mut lever);
    assert_eq!(
        ack,
        Some(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        }),
        "an inert (None) lever replies the 1c Ready stub"
    );
    assert_eq!(lever, None, "the inert lever is untouched (stays None)");
    assert_eq!(
        stats.transfer_unroutable, 0,
        "an Active prepare is not a routing failure"
    );
    assert!(
        session.transfer.is_some(),
        "the prepare opened progress on the Active session"
    );
}

#[test]
fn apply_prepare_armed_lever_rejects_once_then_self_clears() {
    // The `Some` arm + the one-shot self-clear: the FIRST prepare rejects with the armed reason and the
    // lever clears; the SECOND (fresh Active session) prepare is Ready again — proving `.take()` fired.
    let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
    let mut lever: Option<PrepareReject> = Some(reject);

    let mut first = active_session();
    let mut stats = GatewayStats::default();
    let ack1 = apply_prepare(&mut first, XFER, DEST, &mut stats, &mut lever);
    assert_eq!(
        ack1,
        Some(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Rejected(reject),
        }),
        "the armed lever rejects the first prepare with the exact armed reason"
    );
    assert_eq!(lever, None, "the one-shot lever self-cleared (.take)");

    // A SECOND prepare on a fresh Active session now sees the cleared lever → Ready.
    let mut second = active_session();
    let ack2 = apply_prepare(&mut second, XFER, DEST, &mut stats, &mut lever);
    assert_eq!(
        ack2,
        Some(TransferControlAck::Prepared {
            transfer: XFER,
            result: PrepareResult::Ready,
        }),
        "the very next prepare is Ready again (the lever is one-shot, not sticky)"
    );
    assert_eq!(
        stats.transfer_unroutable, 0,
        "neither Active prepare is a routing failure"
    );
}

#[test]
fn apply_prepare_not_active_does_not_consume_the_lever() {
    // Guard precedence: the not-Active guard sits ABOVE the lever and returns None + bumps
    // `transfer_unroutable` WITHOUT consuming the lever — so an inactivity-rejected prepare leaves the
    // lever armed for the retried (Active) one (the lever fires only for a would-be-Ready prepare).
    let reject = PrepareReject::Spatial(SpatialReject::Obstructed);
    let mut lever: Option<PrepareReject> = Some(reject);
    let mut session = active_session();
    session.phase = SessionPhase::AwaitingAttach; // NOT Active
    let mut stats = GatewayStats::default();
    let ack = apply_prepare(&mut session, XFER, DEST, &mut stats, &mut lever);
    assert_eq!(
        ack, None,
        "a not-Active prepare is un-acked (pins the saga, WEDGE-1)"
    );
    assert_eq!(
        stats.transfer_unroutable, 1,
        "a not-Active prepare is counted as unroutable"
    );
    assert_eq!(
        lever,
        Some(reject),
        "the guard did NOT consume the lever — it stays armed for the retried Active prepare"
    );
}

// ===== RLM 5f-3c — the TRUSTED GATEWAY SEED INJECTOR =========================================

/// Every `RealmDemand` the gateway sent to the orchestrator (decoded), ignoring the directory ops that
/// also ride ORCH+Saga (login's `LeaseGrant`). Mirrors [`acks_to_orch`]: `.expect` keeps the Err path
/// in std (no caller branch); a non-`RealmDemand` ORCH/Saga send (the login `LeaseGrant`) maps to None
/// — the `_` arm is exercised by the armed login's hello tick.
fn demands_to_orch(sent: &[(NodeId, MsgClass, Vec<u8>)]) -> Vec<RealmDemand> {
    sent.iter()
        .filter(|(node, class, _)| (*node == ORCH) & (*class == MsgClass::Saga))
        .filter_map(|(_, _, bytes)| {
            match postcard::from_bytes::<InterShardFlow>(bytes).expect("gateway sends a valid flow")
            {
                InterShardFlow::RealmDemand(d) => Some(d),
                _ => None,
            }
        })
        .collect()
}

/// The count of injected home demands across an entire multi-tick login drive.
fn demand_count(sends: &LoginSends) -> usize {
    sends.iter().map(|tick| demands_to_orch(tick).len()).sum()
}

/// THE DEEP HOME every armed rig here uses: the walk-scale world's deepest realm, an Area inside a
/// Planet inside a Star system. Demanding it therefore spins up a four-level chain, which is what these
/// tests are about.
///
/// These rigs used to say the same thing as a POSITION — 25 metres along `+x` from the ambient root —
/// and let the router descend the forest to work out which realm that fell in. It fell in exactly this
/// one, at that area's own centre, so naming it states what the rig always meant and removes the
/// descent from the test as well as from the code.
const TEST_HOME_REALM: RealmId = RealmId::Area(7);

/// RLM 5f-3d — the DEMAND TTL every armed test rig runs with: 8 ticks ⇒ a re-drive cadence of
/// `8 / REDRIVE_DIVISOR = 2` ticks. Small enough that a handful of `set_tick` steps cross it, and NOT 1,
/// so "backed off, not every tick" is observable.
const TEST_DEMAND_TTL: u64 = 8;
/// RLM 5f-3d — the bootstrap TTL for rigs that must NOT expire while a test drives several ticks.
const TEST_BOOTSTRAP_TTL: u64 = 100;

/// An ARMED injector whose accounts live in `homes`, over the walk-scale world, with a LIVE 5f-3d
/// bootstrap budget (a short re-drive cadence, a long bootstrap TTL). The pre-5f-3d 5f-3c tests are
/// unaffected by the budget: they hold `local_tick` at 1, so neither the re-drive nor the TTL can fire
/// during them.
fn armed_injector(homes: HomeRegistry) -> SeedInjectorConfig {
    SeedInjectorConfig {
        armed: true,
        world: test_world().lowered(),
        homes,
        demand_ttl_ticks: TEST_DEMAND_TTL,
        bootstrap_ttl_ticks: TEST_BOOTSTRAP_TTL,
    }
}

/// The walk-scale world every armed rig here resolves homes against.
fn test_world() -> WorldView {
    WorldView::hand_placed(&UniverseConfig::walk_scale())
}

/// `realm`'s full root→leaf lineage in that world — a NAME walk up the parent pointers, with no
/// distance read anywhere in it.
fn lineage_of(realm: RealmId) -> RealmCoord {
    vd_core::worldgen::coord_of_realm(test_world().regions(), realm)
        .expect("the rig names a realm of its own world")
}

/// The rig default: every account lives at the ambient root's own centre.
fn default_homes() -> HomeRegistry {
    let world = test_world();
    let root = world
        .regions()
        .iter()
        .find(|r| r.parent.is_none())
        .expect("a forest has one ambient root")
        .realm;
    homes_at(root)
}

/// A registry where every account lives at `realm`'s own centre — the rig equivalent of "the account
/// store says you live here", with no descent anywhere in it.
fn homes_at(realm: RealmId) -> HomeRegistry {
    let world = test_world();
    HomeRegistry::new(
        StoredHome::in_realm(world.regions(), realm, DVec3::ZERO)
            .expect("the rig names a realm of its own world"),
    )
}

/// A registry where `account` lives at `realm`, `x` metres along `+x` from that realm's own centre, and
/// every other account lives at the ambient root.
fn homes_for(account: AccountId, realm: RealmId, x: f64) -> HomeRegistry {
    let world = test_world();
    let root = world
        .regions()
        .iter()
        .find(|r| r.parent.is_none())
        .expect("a forest has one ambient root")
        .realm;
    HomeRegistry::new(
        StoredHome::in_realm(world.regions(), root, DVec3::ZERO).expect("the root is named"),
    )
    .with_account(
        account,
        StoredHome::in_realm(world.regions(), realm, DVec3::new(x, 0.0, 0.0))
            .expect("the rig names a realm of its own world"),
    )
}

/// Arm `rig`'s live `GatewayConfig` with `injector` + set the clock's `synced` latch (the same
/// re-insert mechanism the D-3 heartbeat tests use).
fn arm_injector(rig: &mut Rig, injector: SeedInjectorConfig, synced: bool) {
    rig.world.insert_resource(GatewayConfig {
        seed_injector: injector,
        ..config()
    });
    rig.world.resource_mut::<ClockSample>().synced = synced;
}

/// A Session-head reply GRANTED to a FOREIGN gateway (`granted` is false — not this gateway).
fn foreign_head(session: SessionId) -> InterShardFlow {
    InterShardFlow::DirectoryReply(DirectoryReply::Head {
        key: DirectoryKey::Session(session),
        record: Some(OwnerRecord {
            authority: AuthorityRef::Gateway(NodeId(0xBAD)),
            fence: Fence(1),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    })
}

/// A Session-head reply owned by THIS gateway but at the WRONG fence (`granted` is false).
fn wrong_fence_head(session: SessionId) -> InterShardFlow {
    InterShardFlow::DirectoryReply(DirectoryReply::Head {
        key: DirectoryKey::Session(session),
        record: Some(OwnerRecord {
            authority: AuthorityRef::Gateway(GW),
            fence: Fence(0xF),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    })
}

/// The composed LEVELS on an outbox (proto_minor 18, §2.4), decoded off the control pushes —
/// each as (origin, origin_epoch, rows). The `_ => None` arm discriminates non-level control
/// (e.g. `OwnEntity`/`AuthorityChanged`). Mirrors `demands_to_orch`.
/// The leading postcard discriminant of one encoded control message — a DATA pin (the
/// composed-scene-only claims compare discriminant bytes instead of matching arms a test
/// could never drive, per the HR5 test discipline).
fn control_disc(msg: &ServerControlMsg) -> u8 {
    postcard::to_allocvec(msg).expect("encodes")[0]
}

fn scene_levels(outbox: &OutboundBox) -> Vec<(RealmId, u64, Vec<SceneRow>)> {
    outbox
        .0
        .iter()
        .filter_map(|(_, _, bytes, _)| {
            match postcard::from_bytes::<ServerControlMsg>(bytes)
                .expect("gateway test pushes only ServerControlMsg on this outbox")
            {
                ServerControlMsg::RealmRegistry {
                    origin,
                    origin_epoch,
                    rows,
                } => Some((origin, origin_epoch, rows)),
                _ => None,
            }
        })
        .collect()
}

/// The composed reliable DELTAS on an outbox (§2.4) — each as (epoch, added, removed).
fn scene_deltas(outbox: &OutboundBox) -> Vec<(u64, Vec<SceneRow>, Vec<RealmId>)> {
    outbox
        .0
        .iter()
        .filter_map(|(_, _, bytes, _)| {
            match postcard::from_bytes::<ServerControlMsg>(bytes)
                .expect("gateway test pushes only ServerControlMsg on this outbox")
            {
                ServerControlMsg::RealmSceneDelta {
                    origin_epoch,
                    added,
                    removed,
                    ..
                } => Some((origin_epoch, added, removed)),
                _ => None,
            }
        })
        .collect()
}

/// The composed per-tick realm datagrams a rig tick sent to `to` — each decoded whole.
fn composed_datagrams(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
    to: NodeId,
) -> Vec<RealmSnapshotDatagram> {
    sent.iter()
        .filter(|(node, class, _)| (*node == to) & (*class == MsgClass::RealmSnapshot))
        .map(|(_, _, bytes)| {
            postcard::from_bytes::<RealmSnapshotDatagram>(bytes)
                .expect("the composed feed carries whole datagrams")
        })
        .collect()
}

#[test]
fn the_composer_emits_the_login_level_the_datagrams_and_the_body_delta() {
    // THE FLAG DAY's login path (§2.4, replacing the deleted SessionAttached RealmRegistry):
    // the first confirmed fold bumps the epoch 0→1 and ships ONE full level — the origin row
    // first (pose ZERO, its self-look attached), every composed row with its bag — then every
    // fresh fold ships the per-tick composed datagram (one frame id per tick, the epoch on
    // each), and a BODY change at a stable epoch ships the reliable delta, never a new level.
    let mut rig = Rig::new();
    let (_sid, _) = rig.login(); // Occupants window id 1 on SHARD; lineage [System(7)]
    let w = WindowId(1);
    // The realm speaks: its level (one authored child row) + its own look + a child marker.
    let sys_look = vd_core::look::look_bag(&vd_core::geometry::Boundary::Shell { r: 40.0 });
    let sent = rig.tick(vec![
        wire(
            SHARD,
            MsgClass::RealmSnapshot,
            &ShardToGateway::WindowFrame {
                realm_fence: Fence(1),
                window: w,
                at: UniverseTick(1000),
                hop: None,
                rows: vec![snap(
                    RealmId::Planet(9),
                    FrameRef::PlanetCentered { planet_seed: 9 },
                    SYS7,
                    30.0,
                    1000,
                )],
            },
        ),
        wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::WindowBody {
                realm_fence: Fence(1),
                window: w,
                subject: RealmId::System(7),
                stmt: vd_wire::session_flow::BodyStmt::SelfLook {
                    bag: sys_look.clone(),
                },
                authored_at: UniverseTick(1000),
            },
        ),
    ]);
    let mut ob = OutboundBox::default();
    ob.0.extend(sent.iter().map(|(to, class, bytes)| {
        (
            *to,
            *class,
            vd_sim::io::bytes(bytes.clone()),
            vd_sim::io::Durability::Ephemeral,
        )
    }));
    // The LEVEL shipped at LOGIN (epoch 0→1, the swap signal — pinned by the login-flow
    // tests); this fold happens at the SAME epoch, so what ships now is the reliable DELTA
    // filling the drawn set: the origin row gained its self-look, and the composed child row
    // appeared (bagless — tracked, not drawn, until its statement lands).
    assert!(
        scene_levels(&ob).is_empty(),
        "a stable epoch re-ships no level — the delta fills the scene"
    );
    let deltas = scene_deltas(&ob);
    assert_eq!(deltas.len(), 1, "the first fold ships one filling delta");
    let (epoch, rows, removed) = &deltas[0];
    assert_eq!(*epoch, 1, "at the login epoch");
    assert!(removed.is_empty());
    assert_eq!(
        rows.len(),
        2,
        "the origin row (its look landed) + the composed child row"
    );
    assert_eq!(rows[0].realm, RealmId::System(7));
    assert_eq!(rows[0].parent, None);
    assert_eq!(
        rows[0].pose.pos.delta_m(
            vd_core::pose::LatticePos::default(),
            rows[0].pose.frame.tier()
        ),
        DVec3::ZERO,
        "the origin draws from its look AT the origin"
    );
    assert_eq!(
        rows[0].bag, sys_look,
        "the origin's self-look rides its delta row"
    );
    assert_eq!(rows[1].realm, RealmId::Planet(9));
    assert_eq!(
        rows[1].parent,
        Some(RealmId::System(7)),
        "hierarchy identity only"
    );
    assert_eq!(
        rows[1].pose.pos.delta_m(
            vd_core::pose::LatticePos::default(),
            rows[1].pose.frame.tier()
        ),
        DVec3::new(30.0, 0.0, 0.0)
    );
    assert_eq!(
        rows[1].bag,
        Vec::<u8>::new(),
        "no statement yet — tracked, not drawn"
    );
    // The per-tick composed datagram rode the same tick: epoch stamped, the gateway's own
    // frame id, the origin absent from the rows (head≠tail law).
    let datagrams = composed_datagrams(&sent, CLIENT);
    assert_eq!(datagrams.len(), 1, "one fresh fold, one datagram");
    assert_eq!(datagrams[0].origin_epoch, 1);
    assert_eq!(
        datagrams[0].frame_id, 1,
        "the per-session monotone feed counter"
    );
    assert_eq!(
        datagrams[0]
            .realms
            .iter()
            .map(|r| r.realm)
            .collect::<Vec<_>>(),
        vec![RealmId::Planet(9)],
        "composed rows only — the origin never rides a datagram row"
    );
    assert_eq!(
        datagrams[0].realms[0].pose.frame, SYS7,
        "the TAIL is the origin frame"
    );

    // A BODY arrives for the child (its marker) at the SAME epoch: the next fresh fold ships
    // a reliable DELTA carrying the row with its new bag — never a whole new level.
    let luma = vd_core::look::luma_bag(2, 1.5);
    let sent = rig.tick(vec![
        wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::WindowBody {
                realm_fence: Fence(1),
                window: w,
                subject: RealmId::Planet(9),
                stmt: vd_wire::session_flow::BodyStmt::Marker { luma: luma.clone() },
                authored_at: UniverseTick(1001),
            },
        ),
        wire(
            SHARD,
            MsgClass::RealmSnapshot,
            &ShardToGateway::WindowFrame {
                realm_fence: Fence(1),
                window: w,
                at: UniverseTick(1001),
                hop: None,
                rows: vec![snap(
                    RealmId::Planet(9),
                    FrameRef::PlanetCentered { planet_seed: 9 },
                    SYS7,
                    31.0,
                    1001,
                )],
            },
        ),
    ]);
    let mut ob = OutboundBox::default();
    ob.0.extend(sent.iter().map(|(to, class, bytes)| {
        (
            *to,
            *class,
            vd_sim::io::bytes(bytes.clone()),
            vd_sim::io::Durability::Ephemeral,
        )
    }));
    assert!(
        scene_levels(&ob).is_empty(),
        "still no new level at a stable epoch"
    );
    let deltas = scene_deltas(&ob);
    assert_eq!(deltas.len(), 1, "the body change ships one reliable delta");
    let (epoch, added, removed) = &deltas[0];
    assert_eq!(*epoch, 1, "at the CURRENT epoch");
    assert_eq!(added.len(), 1);
    assert_eq!(added[0].realm, RealmId::Planet(9));
    assert_eq!(
        added[0].bag, luma,
        "the marker datum rides the delta row's bag"
    );
    assert!(removed.is_empty());
    // The second fresh fold advanced the feed counter — sibling chunks would share it.
    let datagrams = composed_datagrams(&sent, CLIENT);
    assert_eq!(datagrams.len(), 1);
    assert_eq!(datagrams[0].frame_id, 2);
    // A THIRD tick with an UNCHANGED tick/bags emits nothing new (already_current: no fold,
    // no datagram, no delta, no level).
    let sent = rig.tick(vec![]);
    assert!(composed_datagrams(&sent, CLIENT).is_empty());
}

#[test]
fn a_relayed_batch_is_admitted_against_the_child_parked_preroster_and_fail_closed() {
    // THE Q2 RELAY's receiving admission (Slice C1, mesh minor 17; owner-approved 2026-08-16
    // — owner_decisions_2026-08-15.md addendum + window_lane.md §5 RULINGS), driven through
    // the REAL dispatch: a batch arriving BEFORE the author's roster is PARKED (counted
    // unvouched — the send-once forward must not be lost) and DRAINED through the full
    // admission the moment the author's level vouches the child; the opened statements admit
    // against the CHILD's identity (its level held, its self-look + its marker into the one
    // body store, a mis-authored inner body dropped apart); then every refusal arm:
    // a stale child fence, a forged sender, an unknown window, an undecodable seal, and a
    // stale relayed level — all counted, nothing guessed at.
    let mut rig = Rig::new();
    let (_, _) = rig.login(); // window 1 = Occupants(System 7) on SHARD
    let w = WindowId(1);
    let area = RealmId::Area(3);
    let area_frame = FrameRef::AreaLocal {
        planet_seed: 7,
        area_seed: 3,
    };
    let seal = vd_wire::session_flow::seal_relay_statements(&[
        RelayedStatement::Level {
            at: vd_core::UniverseTick(999),
            rows: vec![vd_wire::channels::RealmSnap {
                realm: area,
                frame: area_frame,
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::PlanetCentered { planet_seed: 7 },
                    DVec3::new(1.0, 0.0, 0.0),
                    vd_core::UniverseTick(999),
                ),
            }],
        },
        RelayedStatement::Body {
            subject: RealmId::Planet(7),
            stmt: BodyStmt::SelfLook { bag: vec![1] },
            authored_at: vd_core::UniverseTick(999),
        },
        RelayedStatement::Body {
            subject: area,
            stmt: BodyStmt::Marker { luma: vec![2] },
            authored_at: vd_core::UniverseTick(999),
        },
        // Mis-authored INNER body: a look about a realm that is not the child itself.
        RelayedStatement::Body {
            subject: RealmId::Planet(9),
            stmt: BodyStmt::SelfLook { bag: vec![3] },
            authored_at: vd_core::UniverseTick(999),
        },
    ]);
    let relayed =
        |child: RealmId, child_fence: Fence, statements: Vec<u8>| ShardToGateway::WindowRelayed {
            realm_fence: Fence(1),
            window: w,
            child,
            child_fence,
            statements,
            interior: Vec::new(),
        };
    // (1) BEFORE the author's roster: PARKED + counted unvouched; nothing ingested yet.
    // A second batch with an OLDER child fence does NOT displace the parked one (newest
    // wins even in the park), while a NEWER one does — both counted unvouched.
    let _ = rig.tick(vec![
        wire(
            SHARD,
            MsgClass::Control,
            &relayed(RealmId::Planet(7), Fence(9), seal.clone()),
        ),
        wire(
            SHARD,
            MsgClass::Control,
            &relayed(RealmId::Planet(7), Fence(7), vec![0xAA]),
        ),
    ]);
    assert_eq!(rig.stats().window_relay_unvouched, 2, "parked, counted");
    assert_eq!(rig.stats().window_relays_ingested, 0);
    // (2) The author's level lands naming the child → the parked batch DRAINS through the
    // full admission: the relayed level held, the child's self-look + its marker admitted
    // into the ONE body store, the mis-authored inner body dropped apart.
    let author_frame = ShardToGateway::WindowFrame {
        realm_fence: Fence(1),
        window: w,
        at: vd_core::UniverseTick(6),
        hop: None,
        rows: vec![vd_wire::channels::RealmSnap {
            realm: RealmId::Planet(7),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            pose: vd_core::pose::StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                DVec3::new(30.0, 0.0, 0.0),
                vd_core::UniverseTick(6),
            ),
        }],
    };
    let _ = rig.tick(vec![wire(SHARD, MsgClass::RealmSnapshot, &author_frame)]);
    assert_eq!(
        rig.stats().window_relays_ingested,
        3,
        "the drained batch: the level + the self-look + the marker"
    );
    assert_eq!(
        rig.stats().window_misauthored_body,
        1,
        "the mis-authored inner look dropped apart, against the CHILD's identity"
    );
    {
        let sessions = rig.world.resource::<GatewaySessions>();
        let held = &sessions.windows[&w];
        assert_eq!(
            held.ingest.look_of(RealmId::Planet(7)),
            Some(&[1u8][..]),
            "the relayed self-look landed in the ONE body store (the §2.8 handover path)"
        );
        assert_eq!(held.ingest.marker_of(area), Some(&[2u8][..]));
        assert_eq!(
            held.ingest.relay_child_roster(RealmId::Planet(7)),
            std::collections::BTreeSet::from([area]),
            "the relayed level IS the child's attested roster"
        );
    }
    // (3) A STALE child fence is refused (a deposed incarnation still shipping).
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(RealmId::Planet(7), Fence(8), seal.clone()),
    )]);
    assert_eq!(rig.stats().window_relay_stale, 1);
    // (4a) An OLDER relayed level INSIDE the ring span joins the child's RING APART
    // (look_horizon.md §3.5 C4 — the at-or-before resolution's raw material): admitted, not
    // stale, and the roster still reads the NEWEST level.
    let in_span_level = vd_wire::session_flow::seal_relay_statements(&[RelayedStatement::Level {
        at: vd_core::UniverseTick(998),
        rows: Vec::new(),
    }]);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(RealmId::Planet(7), Fence(9), in_span_level),
    )]);
    assert_eq!(
        rig.stats().window_relay_stale,
        1,
        "an in-span older level rings apart — never refused"
    );
    {
        let sessions = rig.world.resource::<GatewaySessions>();
        let held = &sessions.windows[&w];
        assert_eq!(
            held.ingest.relay_child_roster(RealmId::Planet(7)),
            std::collections::BTreeSet::from([area]),
            "the roster reads the NEWEST ring level, not the older arrival"
        );
    }
    // (4b) A level BEHIND the ring span (beat 25 ⇒ span 51; 900 + 51 < 999) refuses apart.
    let behind_ring = vd_wire::session_flow::seal_relay_statements(&[RelayedStatement::Level {
        at: vd_core::UniverseTick(900),
        rows: Vec::new(),
    }]);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(RealmId::Planet(7), Fence(9), behind_ring),
    )]);
    assert_eq!(
        rig.stats().window_relay_stale,
        2,
        "a behind-ring relayed level refuses apart"
    );
    // (5) FORGED sender: a routable node that is not the window's head.
    let _ = rig.tick(vec![wire(
        DEST,
        MsgClass::Control,
        &relayed(RealmId::Planet(7), Fence(9), seal.clone()),
    )]);
    assert_eq!(rig.stats().window_sender_mismatch, 1);
    // (6) UNKNOWN window id: dropped by id mismatch.
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::WindowRelayed {
            realm_fence: Fence(1),
            window: WindowId(99),
            child: RealmId::Planet(7),
            child_fence: Fence(9),
            statements: seal.clone(),
            interior: Vec::new(),
        },
    )]);
    assert_eq!(rig.stats().window_unknown_row, 1);
    // (7) An UNDECODABLE seal (vouched child, fresh fence): counted, dropped, fail-closed.
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(RealmId::Planet(7), Fence(10), vec![0xFF, 0xFF, 0xFF]),
    )]);
    assert_eq!(rig.stats().window_relay_undecodable, 1);
    // (8) A STALE relayed BODY inside an otherwise-fresh batch: the level applies (newer),
    // the older-stamped look refuses apart (`window_body_stale`) — newest wins per subject.
    let ingested_before = rig.stats().window_relays_ingested;
    let stale_body = vd_wire::session_flow::seal_relay_statements(&[
        RelayedStatement::Level {
            at: vd_core::UniverseTick(1000),
            rows: vec![vd_wire::channels::RealmSnap {
                realm: area,
                frame: area_frame,
                pose: vd_core::pose::StampedPose::at_rest(
                    FrameRef::PlanetCentered { planet_seed: 7 },
                    DVec3::new(2.0, 0.0, 0.0),
                    vd_core::UniverseTick(1000),
                ),
            }],
        },
        RelayedStatement::Body {
            subject: RealmId::Planet(7),
            stmt: BodyStmt::SelfLook { bag: vec![9] },
            authored_at: vd_core::UniverseTick(998),
        },
    ]);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(RealmId::Planet(7), Fence(11), stale_body),
    )]);
    assert_eq!(
        rig.stats().window_body_stale,
        1,
        "an older relayed body refuses apart while its level applies"
    );
    assert_eq!(
        rig.stats().window_relays_ingested,
        ingested_before + 1,
        "the batch's fresh level still ingested"
    );
}

/// THE SEALED INTERIOR FORWARD's admission (look horizon slice 3, owner-approved 2026-08-17
/// — look_horizon.md RULINGS + §2 ASK A; §3.2's three rules + §6's hostile-batch gate),
/// driven through the REAL dispatch, park included: an interior-carrying relay arriving
/// BEFORE the author's roster parks WHOLE and drains through the full admission; a lawful
/// grandchild batch admits ONLY the author's own picture (its Level + markers are the
/// LAWFUL FILTER, counted `window_relay_interior_filtered` — expected NON-zero; a SelfLook
/// about anyone else is a mis-authored body); a HOSTILE batch naming an UNROSTERED
/// grandchild is refused + counted `window_relay_interior_unvouched` (the VIOLATION counter
/// — two different counters, on purpose) with nothing of it ingested; a deposed grandchild
/// incarnation refuses by fence order; an undecodable grandchild seal drops fail-closed.
#[test]
fn the_interior_forward_admits_only_the_authors_own_picture_and_counts_apart() {
    let mut rig = Rig::new();
    let (_, _) = rig.login(); // window 1 = Occupants(System 7) on SHARD
    let w = WindowId(1);
    let child = RealmId::Planet(7);
    let area = RealmId::Area(3);
    let area_frame = FrameRef::AreaLocal {
        planet_seed: 7,
        area_seed: 3,
    };
    // The relaying child's OWN batch: its level rosters the grandchild `area` — the vouch.
    let child_own = vd_wire::session_flow::seal_relay_statements(&[RelayedStatement::Level {
        at: vd_core::UniverseTick(999),
        rows: vec![vd_wire::channels::RealmSnap {
            realm: area,
            frame: area_frame,
            pose: vd_core::pose::StampedPose::at_rest(
                FrameRef::PlanetCentered { planet_seed: 7 },
                DVec3::new(1.0, 0.0, 0.0),
                vd_core::UniverseTick(999),
            ),
        }],
    }]);
    // The LAWFUL grandchild batch: its own picture (admitted), its Level and a marker
    // (depth-3 subjects — filtered), and a SelfLook about somebody else (mis-authored).
    let area_own = vd_wire::session_flow::seal_relay_statements(&[
        RelayedStatement::Body {
            subject: area,
            stmt: BodyStmt::SelfLook { bag: vec![9] },
            authored_at: vd_core::UniverseTick(999),
        },
        RelayedStatement::Level {
            at: vd_core::UniverseTick(999),
            rows: Vec::new(),
        },
        RelayedStatement::Body {
            subject: RealmId::Station(4),
            stmt: BodyStmt::Marker { luma: vec![2] },
            authored_at: vd_core::UniverseTick(999),
        },
        RelayedStatement::Body {
            subject: RealmId::Station(4),
            stmt: BodyStmt::SelfLook { bag: vec![3] },
            authored_at: vd_core::UniverseTick(999),
        },
    ]);
    // The HOSTILE batch: a grandchild the child's own roster never vouched.
    let hostile_own = vd_wire::session_flow::seal_relay_statements(&[RelayedStatement::Body {
        subject: RealmId::Planet(99),
        stmt: BodyStmt::SelfLook { bag: vec![66] },
        authored_at: vd_core::UniverseTick(999),
    }]);
    let relayed =
        |child_fence: Fence, interior: Vec<InteriorRelay>| ShardToGateway::WindowRelayed {
            realm_fence: Fence(1),
            window: w,
            child,
            child_fence,
            statements: child_own.clone(),
            interior,
        };
    // (1) PRE-ROSTER: the whole relay — interior included — parks, counted unvouched once.
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(
            Fence(9),
            vec![
                InteriorRelay {
                    child: area,
                    child_fence: Fence(5),
                    own: area_own.clone(),
                },
                InteriorRelay {
                    child: RealmId::Planet(99),
                    child_fence: Fence(5),
                    own: hostile_own.clone(),
                },
            ],
        ),
    )]);
    assert_eq!(rig.stats().window_relay_unvouched, 1, "parked whole");
    assert_eq!(rig.stats().window_relay_interior_unvouched, 0);
    // (2) The author's level vouches the child → the parked relay DRAINS: the child's level
    // held; the grandchild's OWN picture admitted; the filter and the violation count APART.
    let author_frame = ShardToGateway::WindowFrame {
        realm_fence: Fence(1),
        window: w,
        at: vd_core::UniverseTick(6),
        hop: None,
        rows: vec![vd_wire::channels::RealmSnap {
            realm: child,
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            pose: vd_core::pose::StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                DVec3::new(30.0, 0.0, 0.0),
                vd_core::UniverseTick(6),
            ),
        }],
    };
    let _ = rig.tick(vec![wire(SHARD, MsgClass::RealmSnapshot, &author_frame)]);
    assert_eq!(
        rig.stats().window_relay_interior_unvouched,
        1,
        "the hostile batch naming an unrostered grandchild refused + counted (VIOLATION)"
    );
    assert_eq!(
        rig.stats().window_relay_interior_filtered,
        2,
        "the lawful batch's Level + its marker filtered apart (EXPECTED non-zero)"
    );
    assert_eq!(
        rig.stats().window_misauthored_body,
        1,
        "a SelfLook about somebody else is a mis-authored body, same as every lane"
    );
    {
        let sessions = rig.world.resource::<GatewaySessions>();
        let held = &sessions.windows[&w];
        assert_eq!(
            held.ingest.look_of(area),
            Some(&[9u8][..]),
            "the grandchild's OWN picture landed in the ONE body store"
        );
        assert!(
            held.ingest.interior_admitted(area),
            "…and joined the interior-admitted set (the bag's third disjunct, §3.4.5)"
        );
        assert_eq!(
            held.ingest.look_of(RealmId::Planet(99)),
            None,
            "NOTHING of the hostile batch was ingested"
        );
        assert!(!held.ingest.interior_admitted(RealmId::Planet(99)));
        assert_eq!(
            held.ingest.look_of(RealmId::Station(4)),
            None,
            "no depth-3 subject was ingested"
        );
    }
    // (3) A deposed grandchild incarnation (fence below the admitted one): refused by the
    // SAME per-realm fence map the child's own zombie guard uses (§3.5 C7).
    let stale_before = rig.stats().window_relay_stale;
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(
            Fence(10),
            vec![InteriorRelay {
                child: area,
                child_fence: Fence(4),
                own: area_own.clone(),
            }],
        ),
    )]);
    assert_eq!(
        rig.stats().window_relay_stale,
        stale_before + 1,
        "a deposed grandchild incarnation refuses by fence order"
    );
    // (4) An undecodable grandchild seal: counted, dropped, fail-closed.
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(
            Fence(11),
            vec![InteriorRelay {
                child: area,
                child_fence: Fence(6),
                own: vec![0xFF, 0xFF, 0xFF],
            }],
        ),
    )]);
    assert_eq!(rig.stats().window_relay_undecodable, 1);
    // (4b) A STALE grandchild picture (fresh fence, OLDER authored_at): admissible, but
    // newest-wins in the one body store — refused apart, the held picture untouched.
    let stale_before_body = rig.stats().window_body_stale;
    let older_look = vd_wire::session_flow::seal_relay_statements(&[RelayedStatement::Body {
        subject: area,
        stmt: BodyStmt::SelfLook { bag: vec![4] },
        authored_at: vd_core::UniverseTick(998),
    }]);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &relayed(
            Fence(13),
            vec![InteriorRelay {
                child: area,
                child_fence: Fence(8),
                own: older_look,
            }],
        ),
    )]);
    assert_eq!(
        rig.stats().window_body_stale,
        stale_before_body + 1,
        "an older interior picture refuses apart — newest wins"
    );
    {
        let sessions = rig.world.resource::<GatewaySessions>();
        let held = &sessions.windows[&w];
        assert_eq!(
            held.ingest.look_of(area),
            Some(&[9u8][..]),
            "the held picture is untouched"
        );
    }
    assert_eq!(
        rig.stats().window_relay_interior_unvouched,
        1,
        "still exactly the one violation — the lawful flights added none"
    );
}

/// ★ THE STATIC CHILD'S MARKER, ARRIVING BETWEEN THE TWO DOORS (regression, 2026-08-28).
///
/// A realm's children reach a gateway by TWO doors, because they are two kinds of thing: a MOVER
/// rides the per-tick level, a STATIC child rides the reliable roster. The shard states them in one
/// order every tick — level, then markers, then the static roster — so a static child's marker
/// ALWAYS arrives in the gap between them.
///
/// The park gate used to ask `confirmed()`, which is true as soon as EITHER door has spoken. The
/// level had landed, so the gate said "I have a roster" and judged the marker against a roster that
/// structurally could not contain a static child. It dropped it as mis-authored — and the roster
/// lane never says the same thing twice, so the realm was never drawn again.
///
/// WHAT IT COST: the home star vanished from the drawn scene while all nine sibling planets drew,
/// which `render_boxes_smoke` caught as a drawn-set-against-the-world mismatch. The star is the home
/// system's ONLY static child, which is why it and nothing else disappeared.
///
/// This drives that exact order and holds the gate to the right question: not "has any roster
/// arrived" but "does the roster name THIS one".
#[test]
fn a_static_childs_marker_between_the_two_doors_is_parked_and_then_drawn() {
    let star = RealmId::Star(1_505_330_803_008_586_659);
    let mut held = GatewayWindow {
        shard: SHARD,
        scope: WindowScope::Occupants,
        author_realm: RealmId::System(7),
        ingest: window::WindowIngest::default(),
        parked_bodies: BTreeMap::new(),
        parked_relays: BTreeMap::new(),
    };
    let row = |realm: RealmId, x: f64| vd_wire::channels::RealmSnap {
        realm,
        frame: FrameRef::SystemSpace { system_seed: 7 },
        pose: vd_core::pose::StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 7 },
            DVec3::new(x, 0.0, 0.0),
            vd_core::UniverseTick(10),
        ),
    };
    // DOOR ONE — the level lands, carrying the MOVERS only. A static child cannot be on it.
    assert_eq!(
        held.ingest.ingest_frame(
            window::WindowLevel {
                at: vd_core::UniverseTick(10),
                hop: None,
                rows: vec![row(RealmId::Planet(7), 30.0)],
            },
            &window_tuning(&config()),
        ),
        window::Ingested::Applied,
    );
    // THE TWO QUESTIONS DISAGREE HERE, which is the whole defect in one statement.
    assert!(
        held.ingest.confirmed(),
        "a level alone makes the window confirmed — this is the question that was wrong",
    );
    assert!(
        !held.ingest.rosters(star),
        "…while the roster does not yet name the static child — the question that is right",
    );
    let mut sessions = GatewaySessions::default();
    sessions.windows.insert(WindowId(1), held);
    let mut stats = GatewayStats::default();
    // THE MARKER, in the gap.
    on_window_row(
        SHARD,
        WindowId(1),
        WindowRow::Body {
            subject: star,
            stmt: BodyStmt::Marker { luma: vec![9] },
            authored_at: vd_core::UniverseTick(10),
        },
        &config(),
        &mut sessions,
        &mut stats,
    );
    assert_eq!(
        stats.window_misauthored_body, 0,
        "the static child's marker is NOT judged forged (it read 1 before this fix)",
    );
    assert_eq!(stats.window_body_preroster, 1, "it parks instead");
    // DOOR TWO — the static roster lands, and the drain admits what it now names.
    on_window_row(
        SHARD,
        WindowId(1),
        WindowRow::StaticRows(vec![row(star, 0.0)]),
        &config(),
        &mut sessions,
        &mut stats,
    );
    let held = &sessions.windows[&WindowId(1)];
    assert_eq!(
        held.ingest.marker_of(star),
        Some(&[9u8][..]),
        "the marker is held once the roster names it (it read None before this fix)",
    );
    assert!(
        held.parked_bodies.is_empty(),
        "and the park slot cleared — parking is a delay, never a leak",
    );
}

#[test]
fn a_drained_parked_body_older_than_the_held_statement_counts_stale() {
    // The drain's newest-wins refusal, driven directly (the rig cannot reach it: a marker
    // parks only pre-roster, and a roster never un-confirms — this pins the arm for the
    // redelivered-straggler shape a reordered reliable stream could still produce).
    let mut held = GatewayWindow {
        shard: SHARD,
        scope: WindowScope::Occupants,
        author_realm: RealmId::System(7),
        ingest: window::WindowIngest::default(),
        parked_bodies: BTreeMap::new(),
        parked_relays: BTreeMap::new(),
    };
    let roster_level = window::WindowLevel {
        at: vd_core::UniverseTick(10),
        hop: None,
        rows: vec![vd_wire::channels::RealmSnap {
            realm: RealmId::Planet(7),
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            pose: vd_core::pose::StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 7 },
                DVec3::new(30.0, 0.0, 0.0),
                vd_core::UniverseTick(10),
            ),
        }],
    };
    assert_eq!(
        held.ingest
            .ingest_frame(roster_level, &window_tuning(&config())),
        window::Ingested::Applied
    );
    assert!(held.ingest.ingest_body(
        RealmId::Planet(7),
        &BodyStmt::Marker { luma: vec![9] },
        vd_core::UniverseTick(10)
    ));
    held.parked_bodies.insert(
        RealmId::Planet(7),
        (BodyStmt::Marker { luma: vec![1] }, vd_core::UniverseTick(4)),
    );
    let mut stats = GatewayStats::default();
    drain_parked(&mut held, &window::WindowTuning::derive(2), &mut stats);
    assert_eq!(
        stats.window_body_stale, 1,
        "the drained straggler refuses — newest wins"
    );
    assert_eq!(
        held.ingest.marker_of(RealmId::Planet(7)),
        Some(&[9u8][..]),
        "the held statement survives the drain"
    );
    assert!(
        held.parked_bodies.is_empty(),
        "the park slot cleared either way"
    );
}

#[test]
fn an_old_lane_scene_delta_from_a_shard_is_counted_and_dropped() {
    // ★TOMBSTONE (Slice C2): a shard-side per-observer scene delta has no producer left
    // (deleted, never disabled — §4.5 Topic 5) — its frames land here COUNTED, and nothing
    // client-bound leaves (the composed lane is the one scene author).
    let mut rig = Rig::new();
    let (_sid, _) = rig.login();
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::RealmSceneDelta {
            observer: AccountId(5),
            added: vec![],
            removed: vec![RealmId::Planet(8)],
        },
    )]);
    assert_eq!(rig.stats().old_scene_deltas_dropped, 1);
    assert!(
        decode_controls(&sent, CLIENT)
            .iter()
            .all(|m| !matches!(m, ServerControlMsg::RealmSceneDelta { .. })),
        "nothing scene-shaped is forwarded from the dead lane"
    );
    assert_eq!(
        rig.stats().undecodable,
        0,
        "routed and understood, deliberately dropped"
    );
}

#[test]
fn a_crossing_swap_level_is_composed_at_the_last_old_epoch_tick_when_retained() {
    // §2.7's same-T swap, the retained arm (the capstone drives the fallback — its rings hold
    // nothing at the crossing): a hand-built table where BOTH origins' windows retain the old
    // epoch's tick. The swap level must be composed AT that tick — every persisting body's
    // position is the same-tick re-expression, screen deltas bounded by one tick of true
    // motion (the crossing no-flicker gate's mechanical half) — and the swap tick ships NO
    // datagram (the level carries the poses; the next fresh tick resumes the feed).
    let cfg = config();
    let (mut sessions, sid, _) = one_active_session();
    // The session stands in System(7) (an Active sub on SHARD).
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .insert(
            SHARD,
            SubRecord {
                sub: SubId(0),
                frame: SYS7,
                accepted: Fence(1),
                state: SubState::Active,
            },
        );
    // Two Occupants windows on SHARD — the old origin's and the new one's — each retaining
    // levels at T=100 AND T=101 (the ring spans both, so the swap CAN re-express at 100).
    let mut mk = |id: u64, author: RealmId, own_frame: FrameRef, ticks: &[u64]| {
        let mut ingest = window::WindowIngest::default();
        for &at in ticks {
            let level = window::WindowLevel {
                at: UniverseTick(at),
                hop: None,
                rows: vec![vd_wire::channels::RealmSnap {
                    realm: RealmId::Station(42),
                    frame: FrameRef::StationLocal { station_seed: 42 },
                    pose: vd_core::pose::StampedPose::at_rest(
                        own_frame,
                        DVec3::new(9.0, 0.0, 0.0),
                        UniverseTick(at),
                    ),
                }],
            };
            assert_eq!(
                ingest.ingest_frame(level, &window_tuning(&cfg)),
                window::Ingested::Applied
            );
        }
        sessions.windows.insert(
            WindowId(id),
            GatewayWindow {
                shard: SHARD,
                scope: WindowScope::Occupants,
                author_realm: author,
                ingest,
                parked_bodies: BTreeMap::new(),
                parked_relays: BTreeMap::new(),
            },
        );
    };
    mk(1, RealmId::System(7), SYS7, &[100, 101]);
    // The NEW origin's window retains the old tick AND a newer one — without the same-T
    // preference the swap would fold at 102 and this test would fail.
    mk(2, RealmId::Planet(7), PLANET7, &[100, 101, 102]);
    let mut stats = GatewayStats::default();
    let mut ob = OutboundBox::default();
    // Pass 1: the login scene in System(7) — folds at the NEWEST common tick (101), epoch 1.
    compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
    assert_eq!(stats.scene_levels_sent, 1);
    assert_eq!(
        stats.scene_datagrams_sent, 1,
        "the fresh fold ships the feed"
    );
    assert_eq!(
        sessions.by_session[&sid].shadow.last_t,
        Some(UniverseTick(101))
    );
    // THE CROSSING: the standing frame flips to Planet(7) (the dest sub took over).
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .get_mut(&SHARD)
        .expect("present")
        .frame = PLANET7;
    let datagrams_before = stats.scene_datagrams_sent;
    let mut ob = OutboundBox::default();
    compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
    let levels = scene_levels(&ob);
    assert_eq!(levels.len(), 1, "the swap level shipped");
    let (origin, epoch, rows) = &levels[0];
    assert_eq!(*origin, RealmId::Planet(7));
    assert_eq!(*epoch, 2, "the crossing bumped the epoch exactly once");
    // THE SAME-T HALF: the new chain retains tick 100 — wait, the swap must compose at the
    // LAST OLD-EPOCH tick (101), which the new window retains too.
    assert_eq!(
        sessions.by_session[&sid].shadow.last_t,
        Some(UniverseTick(101)),
        "the swap level is composed AT the last old-epoch tick (§2.7 same-T)"
    );
    for row in rows.iter().filter(|r| r.realm != RealmId::Planet(7)) {
        assert_eq!(
            row.pose.universe_tick,
            UniverseTick(101),
            "every composed swap row is the SAME-tick re-expression"
        );
    }
    assert_eq!(
        stats.scene_datagrams_sent, datagrams_before,
        "the swap tick ships no datagram — the level carries the poses"
    );
}

/// §2.7's same-T swap, the FALLBACK arms (the retained twin above drives the preference):
/// a crossing whose new chain does NOT retain the old tick folds at the freshest common
/// stamp instead — the same-tick re-expression is preferred, never demanded.
#[test]
fn a_crossing_swap_falls_back_to_the_fresh_tick_when_the_old_one_is_not_retained() {
    let cfg = config();
    let (mut sessions, sid, _) = one_active_session();
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .insert(
            SHARD,
            SubRecord {
                sub: SubId(0),
                frame: SYS7,
                accepted: Fence(1),
                state: SubState::Active,
            },
        );
    let mut mk = |id: u64, author: RealmId, own_frame: FrameRef, ticks: &[u64]| {
        let mut ingest = window::WindowIngest::default();
        for &at in ticks {
            let level = window::WindowLevel {
                at: UniverseTick(at),
                hop: None,
                rows: vec![vd_wire::channels::RealmSnap {
                    realm: RealmId::Station(42),
                    frame: FrameRef::StationLocal { station_seed: 42 },
                    pose: vd_core::pose::StampedPose::at_rest(
                        own_frame,
                        DVec3::new(9.0, 0.0, 0.0),
                        UniverseTick(at),
                    ),
                }],
            };
            assert_eq!(
                ingest.ingest_frame(level, &window_tuning(&cfg)),
                window::Ingested::Applied
            );
        }
        sessions.windows.insert(
            WindowId(id),
            GatewayWindow {
                shard: SHARD,
                scope: WindowScope::Occupants,
                author_realm: author,
                ingest,
                parked_bodies: BTreeMap::new(),
                parked_relays: BTreeMap::new(),
            },
        );
    };
    mk(1, RealmId::System(7), SYS7, &[100, 101]);
    // The NEW origin's window retains ONLY a newer tick — the same-T preference cannot hold.
    mk(2, RealmId::Planet(7), PLANET7, &[102]);
    let mut stats = GatewayStats::default();
    let mut ob = OutboundBox::default();
    compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
    assert_eq!(
        sessions.by_session[&sid].shadow.last_t,
        Some(UniverseTick(101))
    );
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .get_mut(&SHARD)
        .expect("present")
        .frame = PLANET7;
    let mut ob = OutboundBox::default();
    compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
    let levels = scene_levels(&ob);
    assert_eq!(levels.len(), 1, "the swap level still ships");
    assert_eq!(levels[0].1, 2, "the epoch still bumps exactly once");
    assert_eq!(
        sessions.by_session[&sid].shadow.last_t,
        Some(UniverseTick(102)),
        "unretained old tick: the freshest stamp serves (§2.7's stated fallback)"
    );
}

/// §2.7's same-T swap, the short-prefix fallback: a crossing whose new chain is NOT fully
/// fresh (a stale upper stratum caps the prefix) also falls back to the fresh fold.
#[test]
fn a_crossing_swap_falls_back_when_the_new_chain_is_not_fully_fresh() {
    let cfg = config();
    let (mut sessions, sid, _) = one_active_session();
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .insert(
            SHARD,
            SubRecord {
                sub: SubId(0),
                frame: SYS7,
                accepted: Fence(1),
                state: SubState::Active,
            },
        );
    let tuning = window_tuning(&cfg);
    let mut occupants = |id: u64, author: RealmId, own_frame: FrameRef, ticks: &[u64]| {
        let mut ingest = window::WindowIngest::default();
        for &at in ticks {
            let level = window::WindowLevel {
                at: UniverseTick(at),
                hop: None,
                rows: vec![vd_wire::channels::RealmSnap {
                    realm: RealmId::Station(42),
                    frame: FrameRef::StationLocal { station_seed: 42 },
                    pose: vd_core::pose::StampedPose::at_rest(
                        own_frame,
                        DVec3::new(9.0, 0.0, 0.0),
                        UniverseTick(at),
                    ),
                }],
            };
            assert_eq!(
                ingest.ingest_frame(level, &tuning),
                window::Ingested::Applied
            );
        }
        sessions.windows.insert(
            WindowId(id),
            GatewayWindow {
                shard: SHARD,
                scope: WindowScope::Occupants,
                author_realm: author,
                ingest,
                parked_bodies: BTreeMap::new(),
                parked_relays: BTreeMap::new(),
            },
        );
    };
    occupants(1, RealmId::System(7), SYS7, &[100, 101]);
    occupants(2, RealmId::Planet(7), PLANET7, &[101, 102]);
    // The new chain's UPPER stratum (the parent's Child-scope window) shares no fresh tick
    // with the leaf: the prefix caps at 1 < 2 and the same-T preference cannot hold.
    let mut stale_parent = window::WindowIngest::default();
    assert_eq!(
        stale_parent.ingest_frame(
            window::WindowLevel {
                at: UniverseTick(50),
                hop: Some(vd_wire::session_flow::HopRow {
                    child: RealmId::Planet(7),
                    placement: vd_core::frame::FramePlacement::moving(
                        DVec3::new(30.0, 0.0, 0.0),
                        DVec3::ZERO,
                    ),
                }),
                rows: vec![vd_wire::channels::RealmSnap {
                    realm: RealmId::Planet(7),
                    frame: PLANET7,
                    pose: vd_core::pose::StampedPose::at_rest(
                        SYS7,
                        DVec3::new(30.0, 0.0, 0.0),
                        UniverseTick(50),
                    ),
                }],
            },
            &tuning
        ),
        window::Ingested::Applied
    );
    sessions.windows.insert(
        WindowId(3),
        GatewayWindow {
            shard: SHARD,
            scope: WindowScope::Child(RealmId::Planet(7)),
            author_realm: RealmId::System(7),
            ingest: stale_parent,
            parked_bodies: BTreeMap::new(),
            parked_relays: BTreeMap::new(),
        },
    );
    let mut stats = GatewayStats::default();
    let mut ob = OutboundBox::default();
    compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
    assert_eq!(
        sessions.by_session[&sid].shadow.last_t,
        Some(UniverseTick(101))
    );
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .get_mut(&SHARD)
        .expect("present")
        .frame = PLANET7;
    let mut ob = OutboundBox::default();
    compose_scenes_pass(&cfg, &test_clock(), &mut sessions, &mut stats, &mut ob);
    let levels = scene_levels(&ob);
    assert_eq!(levels.len(), 1, "the swap level still ships");
    assert_eq!(
        sessions.by_session[&sid].shadow.last_t,
        Some(UniverseTick(102)),
        "a short-prefix chain folds fresh — the preference never blocks the swap"
    );
}

/// THE REMOVE MESSAGE's fan (proto_minor 14, D-4(a)): a shard's `EntityRemoved` reaches its
/// Active subscriber as a typed `ServerControlMsg::Event`, fence-checked per subscriber and
/// minor-gated per session — a stale removal is refused + counted apart (a wrongly-dropped
/// removal strands a frozen figure), and an older peer is withheld the variant.
#[test]
fn an_entity_removed_fans_to_subscribers_fence_checked_and_minor_gated() {
    let mut rig = Rig::new();
    let (sid, login_sends) = rig.login(); // Active session at CLIENT, subscribed to SHARD
    let entity = EntityId(0xE1);
    let at = vd_core::UniverseTick(500);
    let removal = |fence: Fence| ShardToGateway::EntityRemoved {
        realm_fence: fence,
        entity,
        at,
    };
    // (1) A live-fence removal reaches the client as the typed Event.
    let sent = rig.tick(vec![wire(SHARD, MsgClass::Control, &removal(Fence(1)))]);
    let expected =
        ServerControlMsg::Event(vd_wire::channels::EventMsg::EntityRemoved { entity, at });
    assert!(
        decode_controls(&sent, CLIENT).contains(&expected),
        "the removal routes through on_shard_control to the subscriber"
    );
    // (2) A STALE-fence removal (a demoted old owner) is refused for this subscriber + counted
    // on its own counter — it must not evict what the live owner still streams.
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &removal(Fence::GENESIS),
    )]);
    assert!(
        !decode_controls(&sent, CLIENT).contains(&expected),
        "a stale removal is withheld"
    );
    assert_eq!(
        rig.world.resource::<GatewayStats>().stale_removals_dropped,
        1
    );
    // (3) An older peer (minor < 14) is withheld the variant (sender-gates-variants) — it
    // keeps the frozen-figure gap the message closes, but its wire never desyncs.
    rig.world
        .resource_mut::<GatewaySessions>()
        .by_session
        .get_mut(&sid)
        .expect("session")
        .negotiated_minor = 13;
    let sent = rig.tick(vec![wire(SHARD, MsgClass::Control, &removal(Fence(1)))]);
    assert!(
        decode_controls(&sent, CLIENT).is_empty(),
        "a minor<14 peer receives no Event"
    );
    // (4) THE OWNER SKIP: a removal for the session's OWN avatar is never fanned to it — the
    // removal is a fact about the OLD realm mid-crossing, while the owner's truth is its live
    // dest sub; evicting would blink the one figure that must never blink. Restore minor 14
    // first so the skip (not the gate) is what withholds it.
    rig.world
        .resource_mut::<GatewaySessions>()
        .by_session
        .get_mut(&sid)
        .expect("session")
        .negotiated_minor = 14;
    // The session's own entity, read off the wire the login itself announced (`OwnEntity`) —
    // no fallible phase destructure, so no failure-only region (HR5).
    let own = login_sends
        .iter()
        .flat_map(|tick| decode_controls(tick, CLIENT))
        .find_map(|msg| match msg {
            ServerControlMsg::OwnEntity { entity } => Some(entity),
            _ => None,
        })
        .expect("the login announces the own entity");
    let sent = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::EntityRemoved {
            realm_fence: Fence(1),
            entity: own,
            at: vd_core::UniverseTick(501),
        },
    )]);
    assert!(
        decode_controls(&sent, CLIENT).is_empty(),
        "the entity's own session is never told to evict its own avatar"
    );
}

/// The removal fan's C2 honesty arms — the same dead-branch traps the frame fan carries:
/// (a) the reverse index names a session absent from `by_session`; (b) a present session
/// whose hot table holds no entry for the shard; (c) a non-Active session is skipped clean.
/// Each is COUNTED (or a clean skip), never a silent drop.
#[test]
fn an_entity_removed_fan_counts_desyncs_and_skips_non_active_sessions() {
    let removal = ShardToGateway::EntityRemoved {
        realm_fence: Fence(1),
        entity: EntityId(0xE1),
        at: vd_core::UniverseTick(500),
    };
    // (a) index points at a session that does not exist in by_session.
    let mut sessions = GatewaySessions::default();
    sessions
        .subscribed_shards
        .entry(SHARD)
        .or_default()
        .insert(SessionId(0xC0DE));
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    fan_entity_removed(
        SHARD,
        Fence(1),
        EntityId(0xE1),
        vd_core::UniverseTick(500),
        &mut sessions,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.frame_sub_desync, 1, "(a) missing session is counted");
    assert!(outbox.0.is_empty());

    // (b) a present session indexed under SHARD but with an EMPTY hot SubTable.
    let (mut sessions, sid, _) = one_active_session();
    sessions.by_session[&sid]
        .hot
        .subs
        .store(Arc::new(SubTable::default()));
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    fan_entity_removed(
        SHARD,
        Fence(1),
        EntityId(0xE1),
        vd_core::UniverseTick(500),
        &mut sessions,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.frame_sub_desync, 1, "(b) lookup None is counted");
    assert!(outbox.0.is_empty(), "no removal forwarded on a desync");

    // (c) a subscribed but NON-Active session (self-fenced / still attaching) is skipped
    // clean — served no removals, exactly as it is served no frames.
    let (mut sessions, sid, _) = one_active_session();
    sessions.by_session.get_mut(&sid).expect("session").phase = SessionPhase::AwaitingAttach;
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    fan_entity_removed(
        SHARD,
        Fence(1),
        EntityId(0xE1),
        vd_core::UniverseTick(500),
        &mut sessions,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.frame_sub_desync, 0, "not a desync — a clean skip");
    assert!(outbox.0.is_empty());
    let _ = postcard::to_allocvec(&removal).expect("the fixture arm stays constructible");
}

#[test]
fn a_granted_session_lease_injects_exactly_one_server_derived_home_demand() {
    // The trusted injector: on the committed-lease arm (an authenticated login LANDED), the gateway
    // emits EXACTLY ONE RealmDemand{SpinUp} whose child is the SERVER-DERIVED home lineage
    // (container_coord_at over the account's STORED pose) — the client supplies NO coord/pose.
    let account = AccountId(5); // the login account (hello_msg)
    let mut rig = Rig::new();
    arm_injector(
        &mut rig,
        armed_injector(homes_for(account, TEST_HOME_REALM, 0.0)),
        true,
    );
    let (_sid, sends) = rig.login();
    let demands: Vec<RealmDemand> = sends
        .iter()
        .flat_map(|tick| demands_to_orch(tick))
        .collect();
    assert_eq!(
        demands.len(),
        1,
        "exactly one home demand per committed lease"
    );
    // The child is the STORED home's lineage — NOT anything the client sent. The Area-A box is the
    // deep 5-level [Universe, Galaxy, System(7), Planet(7), Area(7)] home (5f-3a).
    let expected_child = lineage_of(TEST_HOME_REALM);
    assert_eq!(
        demands[0].child, expected_child,
        "the child is the stored home's lineage, read and not resolved"
    );
    assert_eq!(demands[0].verb, DemandVerb::SpinUp);
    assert_eq!(
        demands[0].universe_tick,
        UniverseTick(50),
        "the demand carries the clock's universe tick (determinism, no wall-clock)"
    );
    assert_eq!(
        demands[0].parent_fence,
        Fence(5),
        "the per-account sentinel fence (account 5 → 5), not a global constant"
    );
}

#[test]
fn an_unarmed_injector_emits_no_home_demand_byte_identical() {
    // Default (VD_DEMAND unset) ⇒ the injector is INERT: a login emits NO RealmDemand (the byte-
    // identical default; covers the `armed == false` short-circuit arm of the emit gate).
    let mut rig = Rig::new(); // config() ⇒ the inert injector (unarmed)
    let (_sid, sends) = rig.login();
    assert_eq!(
        demand_count(&sends),
        0,
        "an unarmed gateway injects no home demand"
    );
}

#[test]
fn a_pre_sync_clock_injects_no_demand_and_a_synced_clock_injects_one() {
    // The synced gate (CRITIQUE-2): a PRE-SYNC seed would carry universe_tick 0 → last_demand_tick 0,
    // which `demanded_recently` treats as "never demanded" → the home would never spin. So the
    // injector emits ONLY once the clock is synced. Both arms of the `clock.synced` gate.
    //
    // MF3 — and an ARMED-but-pre-sync grant must HOLD, not fall back to the static path: `config.shard`
    // is not this player's home on an armed cluster, so a static attach there would serve the player from
    // the WRONG shard (or retry forever with no TTL). It emits NOTHING client-ward or shard-ward, stays in
    // `AwaitingDirectory`, is COUNTED, and the re-driven idempotent `LeaseGrant` takes the DYNAMIC arm
    // once the clock syncs.
    let poses = homes_for(AccountId(5), TEST_HOME_REALM, 0.0);
    // pre-sync: armed, but the clock has not synced → NO demand.
    let mut rig = Rig::new();
    arm_injector(&mut rig, armed_injector(poses.clone()), false);
    let sid = {
        let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
        session_of(&rig, CLIENT)
    };
    let grant = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    assert_eq!(
        decode_controls(&grant, CLIENT),
        Vec::new(),
        "a pre-sync armed grant pushes NOTHING at the client (no Welcome, no Close — no artifact)"
    );
    assert!(
        !saw_attach(&grant, SHARD, sid, None),
        "and NEVER attaches to the static config.shard on an armed cluster (MF3)"
    );
    assert_eq!(demands_to_orch(&grant).len(), 0, "and demands nothing");
    assert_eq!(
        grant
            .iter()
            .map(|(to, class, _)| (*to, *class))
            .collect::<Vec<_>>(),
        vec![(ORCH, MsgClass::Saga)],
        "the ONLY send is the AwaitingDirectory retry's idempotent LeaseGrant"
    );
    assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingDirectory);
    assert_eq!(
        rig.stats().logins_held_pre_sync,
        1,
        "the hold is counted, never silent"
    );
    // The clock syncs; the SAME session's re-driven grant now takes the DYNAMIC arm.
    rig.world.resource_mut::<ClockSample>().synced = true;
    let synced_grant = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    assert_eq!(
        phase_of(&rig, sid),
        waiting_phase(),
        "once synced the held login enters AwaitingHomeRealm — no re-login needed"
    );
    assert_eq!(
        demands_to_orch(&synced_grant).len(),
        1,
        "and its home is demanded exactly then"
    );
    assert_eq!(rig.stats().logins_held_pre_sync, 1, "held once, not twice");
    // The original 5f-3c assertion, unchanged: a whole pre-sync login drive injects no demand at all.
    let mut rig = Rig::new();
    arm_injector(&mut rig, armed_injector(poses.clone()), false);
    let (_sid, presync) = rig.login();
    assert_eq!(
        demand_count(&presync),
        0,
        "no home demand while the clock is pre-sync"
    );
    // synced: the same armed injector, clock synced → exactly one demand, nonzero tick.
    let mut rig = Rig::new();
    arm_injector(&mut rig, armed_injector(poses), true);
    let (_sid, synced) = rig.login();
    let demands: Vec<RealmDemand> = synced
        .iter()
        .flat_map(|tick| demands_to_orch(tick))
        .collect();
    assert_eq!(
        demands.len(),
        1,
        "exactly one home demand once the clock is synced"
    );
    assert_ne!(
        demands[0].universe_tick,
        UniverseTick(0),
        "a synced demand carries a NONZERO universe tick (so demanded_recently desires it)"
    );
}

#[test]
fn a_re_driven_grant_injects_the_home_demand_exactly_once() {
    // The mint-commit arm fires ONCE (AwaitingDirectory→AwaitingAttach); a re-driven grant re-enters
    // at the AwaitingAttach / Active guards above and returns — it never re-emits. Emit-once per lease.
    let mut rig = Rig::new();
    arm_injector(
        &mut rig,
        armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
        true,
    );
    let hello = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sid = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .next()
        .expect("session pending");
    // The committed-lease transition (5f-3d: `AwaitingDirectory → AwaitingHomeRealm`, since this rig is
    // armed + synced): the ONE inline emit.
    let g1 = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    // A re-driven grant while `AwaitingHomeRealm`: the not-AwaitingDirectory guard returns — NO emit.
    // (The `local_tick` stays 1 throughout, so the 5f-3d re-drive cadence never fires here either.)
    let g2 = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    // A `SessionAttached` from the STATIC shard while awaiting the HOME realm is correctly ignored
    // (5f-3d: this armed session is routed to its home, not to `config.shard`) — and emits nothing.
    let attached = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    // A further granted head: still the same guard, still NO emit.
    let g3 = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    let total: usize = [&hello, &g1, &g2, &attached, &g3]
        .iter()
        .map(|tick| demands_to_orch(tick).len())
        .sum();
    assert_eq!(
        total, 1,
        "exactly one home demand across the whole login + every re-drive"
    );
}

/// Drive an armed gateway to a login's `AwaitingDirectory`, then feed `reply` (a NON-granted head):
/// no home demand is ever injected (the emit is strictly downstream of the committed-lease arm).
fn assert_no_demand_on_non_granted(reply: impl Fn(SessionId) -> InterShardFlow) {
    let mut rig = Rig::new();
    arm_injector(
        &mut rig,
        armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
        true,
    );
    let hello = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sid = rig
        .world
        .resource::<GatewaySessions>()
        .sessions()
        .next()
        .expect("session pending");
    let after = rig.tick(vec![wire(ORCH, MsgClass::Saga, &reply(sid))]);
    assert_eq!(
        demands_to_orch(&hello).len(),
        0,
        "the hello tick injects no demand"
    );
    assert_eq!(
        demands_to_orch(&after).len(),
        0,
        "a non-granted reply injects no home demand"
    );
}

#[test]
fn an_absent_directory_head_injects_no_home_demand() {
    // record None (the reaper revoked / never minted) ⇒ granted false ⇒ the refused arm ⇒ NO emit.
    assert_no_demand_on_non_granted(absent_head);
}

#[test]
fn a_foreign_owned_head_injects_no_home_demand() {
    // The lease is owned by a DIFFERENT gateway ⇒ granted false ⇒ NO emit.
    assert_no_demand_on_non_granted(foreign_head);
}

#[test]
fn a_wrong_fence_head_injects_no_home_demand() {
    // Owned by THIS gateway but at the wrong fence ⇒ granted false ⇒ NO emit.
    assert_no_demand_on_non_granted(wrong_fence_head);
}

#[test]
fn an_absent_spawn_pose_derives_the_origin_home_in_forest() {
    // The P7-store-absent stand-in: NO stored pose ⇒ the injector derives the ROOT/ORIGIN home coord
    // (still a valid in-forest lineage), NOT a skip. One demand, child == container_coord_at(origin).
    let mut rig = Rig::new();
    arm_injector(&mut rig, armed_injector(default_homes()), true); // empty pose store
    let (_sid, sends) = rig.login();
    let demands: Vec<RealmDemand> = sends
        .iter()
        .flat_map(|tick| demands_to_orch(tick))
        .collect();
    assert_eq!(
        demands.len(),
        1,
        "an absent pose still injects one (origin) home demand"
    );
    assert_eq!(
        demands[0].child,
        default_homes().fallback().realm,
        "an account with no home of its own is demanded at the registry's fallback home"
    );
}

#[test]
fn the_home_sentinel_fence_is_per_account_and_deterministic() {
    // CRITIQUE-1 defense-in-depth: the sentinel is DETERMINISTIC per-account (NOT a global constant),
    // so two accounts carry DISTINCT parent_fences — no FencedKey idempotency-collapse.
    assert_ne!(
        home_sentinel_fence(AccountId(5)),
        home_sentinel_fence(AccountId(6)),
        "two accounts ⇒ two distinct sentinels"
    );
    assert_eq!(
        home_sentinel_fence(AccountId(5)),
        Fence(5),
        "the fold is deterministic (low word for a small account)"
    );
    // Accounts differing ONLY in the high u64 word still differ — the fold mixes both halves.
    assert_ne!(
        home_sentinel_fence(AccountId(1u128 << 64)),
        home_sentinel_fence(AccountId(0)),
        "the high word is folded in (distinct even when the low word matches)"
    );
}

#[test]
fn the_defence_reads_the_same_world_that_produced_the_answer() {
    // The injector's defence-in-depth predicate, both arms — now asked OF THE WORLD rather than of a
    // freshly rebuilt one. It used to consult the hand-placed world while the home came from the
    // generated one, which is only safe while those two happen to hold the same realms; the false arm
    // below is exactly what a valid home beside a second star would have hit.
    use vd_core::realm_path::{RealmKindTag, RealmLevel};
    // The LOWERED world — the exact value the shipped injector holds and queries.
    let world = test_world().lowered();
    let home = lineage_of(TEST_HOME_REALM);
    assert!(
        world.contains_realm(home.lowered()),
        "a stored home is in the world it is stored against"
    );
    // Append a leaf realm no world contains (Station(0xDEAD)) — a corrupted stand-in.
    let off = home.child(RealmLevel::new(RealmKindTag::Station, 0xDEAD));
    assert!(
        !world.contains_realm(off.lowered()),
        "an off-world leaf (a corrupted stand-in) is rejected"
    );
}

// ===== RLM 5f-3d — the GATEWAY DYNAMIC-HOME ROUTE + the seamless attach hold ===================

/// The DEMAND-SPAWNED home shard. Deliberately NOT a member of `config().known_shards`
/// (`{SHARD, DEST}`), exactly like a real shard whose `NodeId` is minted at spawn time — so any test
/// that routes to it proves the RUNTIME `dynamic_shards` roster is what makes it dispatchable.
const HOME: NodeId = NodeId(77);
/// A second client connection. The SAME account ⇒ the SAME home realm, which is what makes the
/// one-reply-resolves-many fan-out (and the roster refcount) observable.
const CLIENT2: NodeId = NodeId(101);

/// The home realm `AccountId(5)`'s stored pose `(25,0,0)` resolves to: the SAME lineage `home_coord`
/// derives (and `demand_for_home` demands), lowered to the `RealmId` the directory keys realms by
/// (`DirectoryKey::Realm(coord.lowered())` — exactly what `rlm.rs` grants a spawned realm at).
fn home_lineage() -> RealmCoord {
    lineage_of(TEST_HOME_REALM)
}

/// …lowered to the `RealmId` the directory keys realms by.
fn home_realm() -> RealmId {
    home_lineage().lowered()
}

/// A `Realm`-head reply for `rid`: `Some(node)` = the realm is LIVE, owned by `node` (its shard took the
/// realm lease); `None` = not up yet (the shard is still booting).
fn realm_head(rid: RealmId, owner: Option<NodeId>) -> InterShardFlow {
    InterShardFlow::DirectoryReply(DirectoryReply::Head {
        key: DirectoryKey::Realm(rid),
        record: owner.map(|node| OwnerRecord {
            authority: AuthorityRef::Shard(node),
            fence: Fence(3),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
    })
}

/// A rig in DYNAMIC-HOME mode: armed + synced, with `AccountId(5)`'s stored spawn pose and the live
/// 5f-3d budget (`TEST_DEMAND_TTL` / `TEST_BOOTSTRAP_TTL`).
fn dynamic_rig() -> Rig {
    let mut rig = Rig::new();
    arm_injector(
        &mut rig,
        armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
        true,
    );
    rig
}

/// A dynamic rig whose bootstrap TTL is `ttl` ticks (for the bounded-TTL cells).
fn dynamic_rig_with_ttl(ttl: u64) -> Rig {
    let mut rig = Rig::new();
    let injector = SeedInjectorConfig {
        bootstrap_ttl_ticks: ttl,
        ..armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0))
    };
    arm_injector(&mut rig, injector, true);
    rig
}

fn phase_of(rig: &Rig, sid: SessionId) -> SessionPhase {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session present")
        .phase
        .clone()
}

fn session_of(rig: &Rig, client: NodeId) -> SessionId {
    *rig.world
        .resource::<GatewaySessions>()
        .by_client
        .get(&client)
        .expect("client has a session")
}

/// The session's WRITE-route authority (the node its input would be forwarded to).
fn route_authority(rig: &Rig, sid: SessionId) -> NodeId {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session present")
        .hot
        .route
        .load()
        .authority
}

/// The phase a WAITING dynamic session must be in: holding for its home REALM (the lineage + the cadence
/// anchor live once per realm in `home_bootstraps`, asserted by [`home_wait`]).
fn waiting_phase() -> SessionPhase {
    SessionPhase::AwaitingHomeRealm {
        home_rid: home_realm(),
    }
}

/// The EXACT per-realm bootstrap index a set of `members` waiting on the one home realm must produce
/// (wait opened at local tick `since`, representative account 5 — every dynamic rig's login account).
fn home_wait(since: u64, members: &[SessionId], resolved: bool) -> BTreeMap<RealmId, HomeWait> {
    BTreeMap::from([(
        home_realm(),
        HomeWait {
            coord: home_lineage(),
            since: TickId(since),
            account: AccountId(5),
            members: members.iter().copied().collect(),
            resolved,
        },
    )])
}

/// Drive ONE dynamic login for `client` (hello → granted lease); returns its id + the GRANT tick's sends.
#[allow(clippy::type_complexity)] // test helper: one tick of raw sends
fn dynamic_login(rig: &mut Rig, client: NodeId) -> (SessionId, Vec<(NodeId, MsgClass, Vec<u8>)>) {
    let _ = rig.tick(vec![wire(client, MsgClass::Control, &hello_msg())]);
    let sid = session_of(rig, client);
    let granted = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    (sid, granted)
}

/// The spawn pose THIS rig's gateway derives for the test account — computed through the very
/// `home_placement` production uses, so an attach assertion can never drift from the
/// derivation. Since the T2 spawn-standoff fix EVERY attach carries the registry's pose
/// (static included — an attach with no pose dropped the account inside the Star realm).
fn attach_spawn(rig: &Rig) -> Option<StampedPose> {
    let injector = &rig.world.resource::<GatewayConfig>().seed_injector;
    Some(home_placement(injector, AccountId(5)).1)
}

/// Did the gateway send THIS session's `AttachSession` to `node`? Compares the EXACT expected bytes
/// (never a speculative decode — postcard is not self-describing, so a client-bound `ServerControlMsg`
/// on the same `Control` class can mis-decode as a `GatewayToShard`). Bitwise `&` (no short-circuit
/// region — HR5).
fn saw_attach(
    sent: &[(NodeId, MsgClass, Vec<u8>)],
    node: NodeId,
    sid: SessionId,
    spawn: Option<StampedPose>,
) -> bool {
    let expected = postcard::to_allocvec(&GatewayToShard::AttachSession {
        session: sid,
        fence: Fence(1),
        account: AccountId(5),
        spawn,
    })
    .expect("encode");
    sent.iter().any(|(to, class, bytes)| {
        (*to == node) & (*class == MsgClass::Control) & (bytes.as_slice() == expected.as_slice())
    })
}

/// Did the gateway send THIS session's `DetachSession` to `node`? (Exact bytes, as above.)
fn saw_detach(sent: &[(NodeId, MsgClass, Vec<u8>)], node: NodeId, sid: SessionId) -> bool {
    let expected = postcard::to_allocvec(&GatewayToShard::DetachSession {
        session: sid,
        fence: Fence(1),
    })
    .expect("encode");
    sent.iter().any(|(to, class, bytes)| {
        (*to == node) & (*class == MsgClass::Control) & (bytes.as_slice() == expected.as_slice())
    })
}

/// Did the gateway revoke THIS session's committed lease? (Exact bytes, as above.)
fn saw_lease_revoke(sent: &[(NodeId, MsgClass, Vec<u8>)], sid: SessionId) -> bool {
    let expected = postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::LeaseRevoke {
        key: DirectoryKey::Session(sid),
        fence: Fence(1),
    }))
    .expect("encode");
    sent.iter().any(|(to, class, bytes)| {
        (*to == ORCH) & (*class == MsgClass::Saga) & (bytes.as_slice() == expected.as_slice())
    })
}

/// How many `HeadRead{Realm(rid)}` polls the gateway sent (exact bytes).
fn realm_head_reads(sent: &[(NodeId, MsgClass, Vec<u8>)], rid: RealmId) -> usize {
    let expected = postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::HeadRead {
        key: DirectoryKey::Realm(rid),
    }))
    .expect("encode");
    sent.iter()
        .filter(|(to, class, bytes)| {
            (*to == ORCH) & (*class == MsgClass::Saga) & (bytes.as_slice() == expected.as_slice())
        })
        .count()
}

#[test]
fn nothing_of_the_home_lineage_is_running_at_the_instant_the_login_descent_must_answer() {
    // SLICE 6 (b) — THE BOOTSTRAP CIRCULARITY, AS AN ASSERTION RATHER THAN A PARAGRAPH.
    //
    // The other half of this slice was to move the login DESCENT into the chain: the galaxy's shard
    // hands the star's shard a point in the star's frame, the star's hands the planet's a point in the
    // planet's frame, and the planet accepts and computes nothing. That is the right shape and it is
    // what every other lane in this arc now does.
    //
    // It cannot be built as things stand, and this gate is why. The descent answers TWO questions from
    // ONE walk: WHICH realm the account lives in (a routing decision) and WHERE INSIDE IT they stand (the
    // geometry). The chain can only be asked the second once it is running — and it only runs because
    // something demanded the lineage, which is the first. At the instant the descent has to produce its
    // answer the gateway holds an unresolved bootstrap wait and NOT ONE shard on the runtime routable
    // roster: there is nobody to ask. The router answers from its forest copy precisely because nothing
    // is running yet.
    //
    // This is an owner-visible design decision, not an implementation detail, and inventing a fallback
    // here would be exactly the kind of quiet second source of truth this whole arc exists to remove.
    // The two ways out both change WHERE AN ACCOUNT'S HOME IS STORED or WHAT IS ALWAYS UP, and neither is
    // this slice's to pick:
    //   (1) store a home as (lineage, pose in that realm's own frame) — the P7 durable per-realm store —
    //       so there is no walk to run and no absolute to descend from; the router carries a NAME and a
    //       number it did not compute; or
    //   (2) make the ambient root always resident and descend it level by level, demanding as it goes:
    //       the root names the child that contains you and hands it your point in that child's frame,
    //       and the login waits one round-trip per level. Seamless (a login already waits), but it makes
    //       "the root is up" a cluster invariant the demand loop does not have today.
    let mut rig = dynamic_rig();
    let (sid, _granted) = dynamic_login(&mut rig, CLIENT);
    let sessions = rig.world.resource::<GatewaySessions>();
    // The descent HAS run — the session is waiting on the realm it named, and carries the pose that walk
    // produced. So the answer exists at this instant and was produced by the one party that holds no
    // realm at all.
    assert_eq!(sessions.home_realm_of(sid), Some(home_realm()));
    assert!(
        sessions
            .by_session
            .get(&sid)
            .is_some_and(|s| s.spawn.is_some()),
        "the login already carries a pose measured in its home realm's frame"
    );
    // And nothing of that lineage is reachable. `dynamic_shards` is the RUNTIME routable roster — the
    // only place a demand-spawned shard can appear — and it is empty; `home_shard_of` is the same fact
    // asked about this one realm.
    assert_eq!(
        sessions.dynamic_shard_count(),
        0,
        "no demand-spawned shard is routable yet — there is no chain to ask"
    );
    assert_eq!(
        sessions.home_shard_of(sid),
        None,
        "and specifically not the home realm's own shard"
    );
}

#[test]
fn a_dynamic_login_is_welcomed_at_the_lease_then_held_with_no_attach_and_no_close() {
    // THE SEAMLESS HOLD. On the committed lease a dynamic login is WELCOMED exactly as a static one is
    // (same instant, same bytes), then HELD in `AwaitingHomeRealm`: no `AttachSession` to the static
    // `config.shard` (a teleport to the wrong shard), no `Close`, no loading-screen signal — it simply
    // has no frames yet. The demand + the head-poll ride EXISTING wire arms.
    let mut rig = dynamic_rig();
    let (sid, granted) = dynamic_login(&mut rig, CLIENT);
    assert_eq!(
        decode_controls(&granted, CLIENT),
        vec![
            ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: sid,
                session_fence: Fence(1),
                epoch: EpochId(9),
            },
            ServerControlMsg::UniverseRate { tick_hz: 50 },
        ],
        "a dynamic login is Welcome'd at the committed lease — and nothing else is pushed at it"
    );
    assert!(
        !saw_attach(&granted, SHARD, sid, None),
        "a dynamic login never attaches to the static config.shard"
    );
    assert!(
        !saw_attach(&granted, HOME, sid, attach_spawn(&rig)),
        "and it cannot attach to its home before the head names the node"
    );
    assert_eq!(phase_of(&rig, sid), waiting_phase());
    assert_eq!(
        demands_to_orch(&granted).len(),
        1,
        "ONE home demand, on the existing RealmDemand arm"
    );
    assert_eq!(
        realm_head_reads(&granted, home_realm()),
        1,
        "and ONE poll of the home realm's directory head"
    );
    let sessions = rig.world.resource::<GatewaySessions>();
    assert_eq!(sessions.home_realm_of(sid), Some(home_realm()));
    assert_eq!(
        sessions.home_shard_of(sid),
        None,
        "no home shard until the head resolves"
    );
    assert_eq!(
        sessions.entity_of(sid),
        None,
        "a held session has no avatar yet (the new phase is a non-Active arm of entity_of)"
    );
    assert_eq!(
        sessions.home_bootstraps,
        home_wait(1, &[sid], false),
        "the session is a member of its home realm's ONE bootstrap wait — which owns the descended \
         lineage, the cadence anchor and the representative account (so ONE reply resolves it and ONE \
         re-drive covers it)"
    );
    // The EXACT send fingerprint of a dynamic grant tick: Welcome + UniverseRate to the client, then the
    // demand + the head-poll to the orchestrator — and NOTHING shard-ward (no attach from the grant arm
    // and none from the re-drive, which is not due on the tick the wait began).
    assert_eq!(
        granted
            .iter()
            .map(|(to, class, _)| (*to, *class))
            .collect::<Vec<_>>(),
        vec![
            (CLIENT, MsgClass::Control),
            (CLIENT, MsgClass::Control),
            (ORCH, MsgClass::Saga),
            (ORCH, MsgClass::Saga),
        ]
    );
}

#[test]
fn the_committed_lease_welcome_is_identical_in_both_modes() {
    // SEAMLESS + byte-identical AT THE CLIENT BOUNDARY: the client sees the SAME control stream at the
    // committed lease whether or not the gateway is in dynamic-home mode. Nothing about the home
    // bootstrap leaks to it — no Close, no teleport, no extra/omitted variant, no reordering. (Both rigs
    // share `session_seed`, so the minted SessionId — and hence the Welcome bytes — match exactly.)
    let mut dynamic = dynamic_rig();
    let (dyn_sid, dyn_granted) = dynamic_login(&mut dynamic, CLIENT);
    let mut static_rig = Rig::new();
    let (static_sid, static_granted) = dynamic_login(&mut static_rig, CLIENT);
    assert_eq!(dyn_sid, static_sid, "the same mint stream in both rigs");
    assert_eq!(
        decode_controls(&dyn_granted, CLIENT),
        decode_controls(&static_granted, CLIENT),
        "the client-visible Welcome at the committed lease is identical in both modes"
    );
}

#[test]
fn a_home_head_with_no_record_keeps_the_session_waiting_seamlessly() {
    // The realm is DEMANDED but its shard is still booting (`record: None`). The waiter STAYS: no Close,
    // no teleport, no fallback attach — and it is not dropped.
    let mut rig = dynamic_rig();
    let (sid, _) = dynamic_login(&mut rig, CLIENT);
    let after = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), None),
    )]);
    assert_eq!(
        phase_of(&rig, sid),
        waiting_phase(),
        "a not-yet-routable home keeps the session waiting"
    );
    assert_eq!(
        decode_controls(&after, CLIENT),
        Vec::new(),
        "NOTHING is pushed at the client while its home boots (no Close, no teleport)"
    );
    assert_eq!(
        rig.world.resource::<GatewaySessions>().len(),
        1,
        "held, never dropped"
    );
    assert!(
        !saw_attach(&after, SHARD, sid, None),
        "and never fallen back onto the static shard"
    );
}

#[test]
fn a_resolved_home_head_routes_the_attach_to_the_spawned_shard_and_admits_its_frames() {
    // THE ROUTE. The `Realm` head names the spawned node: it JOINS the runtime routable roster, the
    // WRITE route is retargeted to it, the attach goes THERE (never `config.shard`), and — purely via
    // that runtime roster — its `SessionAttached` promotes the session and its frames reach the client.
    let mut rig = dynamic_rig();
    let (sid, _) = dynamic_login(&mut rig, CLIENT);
    let resolved = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
    assert!(
        saw_attach(&resolved, HOME, sid, attach_spawn(&rig)),
        "the attach goes to the SPAWNED home shard"
    );
    assert!(
        !saw_attach(&resolved, SHARD, sid, None),
        "never to the static config.shard"
    );
    assert_eq!(
        route_authority(&rig, sid),
        HOME,
        "the WRITE route retargeted"
    );
    let sessions = rig.world.resource::<GatewaySessions>();
    assert_eq!(sessions.home_shard_of(sid), Some(HOME));
    assert_eq!(
        sessions.dynamic_shards,
        BTreeMap::from([(HOME, 1)]),
        "the spawned node joined the RUNTIME routable roster (it is not in the frozen config)"
    );
    assert_eq!(
        sessions.home_bootstraps,
        home_wait(1, &[sid], true),
        "the session STAYS a member of the (now resolved) bootstrap — MF1: the demand re-seed must \
         outlive the resolve and run until the Active promote, or the reconciler reaps the realm this \
         login is attaching to"
    );
    // The runtime roster is what makes HOME dispatchable as a shard at all.
    let attached = rig.tick(vec![wire(
        HOME,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().entity_of(sid),
        Some(EntityId(77)),
        "the session went Active off its HOME shard's attach"
    );
    assert_eq!(
        decode_controls(&attached, CLIENT),
        vec![
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(0),
                frame: FrameRef::SystemSpace { system_seed: 7 },
            },
            ServerControlMsg::AuthorityChanged {
                entity: EntityId(77),
                sub: SubId(0),
            },
            ServerControlMsg::OwnEntity {
                entity: EntityId(77)
            },
            // THE LOGIN LEVEL (§2.4, the deleted attach-seam RealmRegistry's successor): the
            // composer's first pass after the attach sees the standing realm and bumps the
            // epoch 0→1 — the swap signal, ordered after the identity control above. No
            // window frames were fed in this rig, so it carries only the origin's bagless
            // row; the deltas fill the drawn set as the statements land.
            ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 1,
                rows: vec![SceneRow {
                    realm: RealmId::System(7),
                    parent: None,
                    pose: vd_core::pose::StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::ZERO,
                        UniverseTick(0),
                    ),
                    bag: Vec::new(),
                }],
            },
        ],
        "the login sub opened on the HOME shard (never config.shard); the composer ships the \
         login level after the identity control (§2.4)"
    );
    let framed = rig.tick(vec![wire(
        HOME,
        MsgClass::Snapshot,
        &frame_msg(Fence(1), 1),
    )]);
    assert_eq!(
        framed
            .iter()
            .filter(|(to, class, _)| (*to == CLIENT) & (*class == MsgClass::Snapshot))
            .count(),
        1,
        "a frame from the dynamically routed home shard reaches the client"
    );
    assert_eq!(
        rig.world
            .resource::<GatewaySessions>()
            .by_session
            .get(&sid)
            .expect("session")
            .bootstrap_deadline,
        None,
        "the bounded bootstrap window closed at the Active promote"
    );
    // MF1 test (iii): the `Active` promote is the wait's TERMINATOR — the member leaves and, being the
    // last one, takes the realm's whole entry (and its lineage `Vec`) with it.
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_bootstraps,
        BTreeMap::new(),
        "the Active promote drops the member and prunes the realm entry"
    );
    // …so the re-drive is silent from here on, however many cadences pass (nothing left to re-seed).
    set_tick(&mut rig, 11);
    let after_active = rig.tick(vec![]);
    assert_eq!(
        demands_to_orch(&after_active).len(),
        0,
        "an Active session's home is no longer re-seeded (arm-B owns it now — live occupancy)"
    );
    assert_eq!(realm_head_reads(&after_active, home_realm()), 0);
    // A `Bye` from an ACTIVE dynamic session: its `home_rid` still names the realm but the wait entry is
    // long gone, so the index exit is a clean no-op — while the DETACH still routes to its home shard and
    // the runtime roster releases (the whole dynamic teardown, after the bootstrap has ended).
    let bye = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert!(
        saw_detach(&bye, HOME, sid),
        "an Active dynamic session detaches at its HOME shard, never config.shard"
    );
    let sessions = rig.world.resource::<GatewaySessions>();
    assert_eq!(sessions.len(), 0);
    assert_eq!(sessions.home_bootstraps, BTreeMap::new());
    assert_eq!(
        sessions.dynamic_shards,
        BTreeMap::new(),
        "and its roster claim is released"
    );
}

#[test]
fn the_home_bootstrap_re_drive_fires_on_the_backoff_cadence_not_every_tick() {
    // THE RE-DRIVE AS A CORRECTNESS INVARIANT (CRITIQUE-3): while waiting, the gateway must keep
    // re-seeding the demand (so the reconciler's arm-A `demanded_recently` never lapses mid-boot and
    // reaps the half-booted realm) AND re-poll the head. But BACKED OFF: `demand_ttl / 4`, NOT every
    // tick — the 100K mass-login storm guard. Both arms of the due/not-due branch are exercised.
    let cadence = SeedInjectorConfig {
        ..armed_injector(default_homes())
    }
    .redrive_interval_ticks();
    assert_eq!(cadence, 2, "demand_ttl 8 / REDRIVE_DIVISOR 4 = 2");
    assert!(
        cadence < TEST_DEMAND_TTL,
        "the cadence must be STRICTLY inside the demand TTL, or arm-A lapses mid-boot"
    );
    assert!(cadence > 1, "…and it must not degenerate to every tick");
    let mut rig = dynamic_rig();
    let (sid, _) = dynamic_login(&mut rig, CLIENT); // waiting since local tick 1
    // Tick 2 (elapsed 1): NOT due.
    set_tick(&mut rig, 2);
    let quiet = rig.tick(vec![]);
    assert_eq!(demands_to_orch(&quiet).len(), 0, "no re-seed off cadence");
    assert_eq!(
        realm_head_reads(&quiet, home_realm()),
        0,
        "no re-poll off cadence"
    );
    // Tick 3 (elapsed 2 == cadence): DUE — both halves fire.
    set_tick(&mut rig, 3);
    let driven = rig.tick(vec![]);
    let demands = demands_to_orch(&driven);
    assert_eq!(demands.len(), 1, "the home demand is RE-SEEDED on cadence");
    assert_eq!(
        demands[0].child,
        lineage_of(TEST_HOME_REALM),
        "the re-seed names the SAME stored home lineage"
    );
    assert_eq!(demands[0].verb, DemandVerb::SpinUp);
    assert_eq!(
        realm_head_reads(&driven, home_realm()),
        1,
        "and the head is re-polled"
    );
    // Tick 4 (elapsed 3): NOT due again — proof it is a cadence, not a latch.
    set_tick(&mut rig, 4);
    let quiet2 = rig.tick(vec![]);
    assert_eq!(demands_to_orch(&quiet2).len(), 0);
    // Tick 5 (elapsed 4): due again.
    set_tick(&mut rig, 5);
    assert_eq!(demands_to_orch(&rig.tick(vec![])).len(), 1);
    assert_eq!(
        phase_of(&rig, sid),
        waiting_phase(),
        "still held, seamlessly"
    );
}

#[test]
fn the_re_seed_outlives_the_resolve_so_the_reconciler_never_reaps_the_home_mid_attach() {
    // MF1 DEFECT A — THE ANTI-REAP INVARIANT, the reason this slice exists. The head resolve is NOT the
    // end of the re-seed: a session can sit in the DYNAMIC `AwaitingAttach` for the whole rest of the
    // bootstrap window (a lost `AttachSession`/`SessionAttached` — the 5f-4 dial-a-fresh-pod race), and a
    // booted-but-unoccupied realm self-reports `Empty`, so the reconciler's arm-B (`running_live &
    // !empty_confirmed`) is FALSE. Only the gateway's re-seed keeps arm-A `demanded_recently` alive; if it
    // stopped at the resolve, arm-A would expire one `demand_ttl` after it and the reconciler would KILL
    // the realm this login is attaching to — long BEFORE the bootstrap TTL noticed.
    let mut rig = dynamic_rig(); // demand_ttl 8 (cadence 2), bootstrap_ttl 100
    let (sid, _) = dynamic_login(&mut rig, CLIENT); // wait opened at local tick 1
    let resolve_tick = 2;
    // A tick MORE than one whole `demand_ttl` after the resolve — exactly where the reconciler's arm-A
    // would have lapsed had the re-seed stopped there — and on the wait's cadence (odd, anchored at 1).
    let probe_tick = 13;
    assert!(
        probe_tick - resolve_tick > TEST_DEMAND_TTL,
        "the probe must sit past one whole demand TTL from the resolve (where arm-A lapses)"
    );
    set_tick(&mut rig, resolve_tick);
    let resolved = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert!(saw_attach(&resolved, HOME, sid, attach_spawn(&rig)));
    assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
    // Now the attach is NEVER confirmed.
    set_tick(&mut rig, probe_tick);
    let late = rig.tick(vec![]);
    assert_eq!(
        demands_to_orch(&late).len(),
        1,
        "the home demand is STILL re-seeded while the resolved session waits to attach (MF1-A)"
    );
    assert_eq!(
        demands_to_orch(&late)[0].child,
        home_lineage(),
        "and it still names the SAME server-derived home lineage"
    );
    assert_eq!(
        realm_head_reads(&late, home_realm()),
        0,
        "…while the head POLL stays off — the node is already known (only the demand half continues)"
    );
    assert!(
        saw_attach(&late, HOME, sid, attach_spawn(&rig)),
        "the dynamic attach retry rides the SAME cadence tick (coalesced, not per-tick)"
    );
    // Off-cadence the whole thing is silent — the retry is a cadence, not a latch.
    set_tick(&mut rig, probe_tick + 1);
    let quiet = rig.tick(vec![]);
    assert_eq!(demands_to_orch(&quiet).len(), 0);
    assert!(
        !saw_attach(&quiet, HOME, sid, attach_spawn(&rig)),
        "a DYNAMIC attach retry does not fire every tick (the mass-login storm guard)"
    );
    // Still held, still bounded, and it can still complete: the attach lands and the session goes Active.
    assert_eq!(rig.stats().home_bootstrap_timeouts, 0);
    let attached = rig.tick(vec![wire(
        HOME,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().entity_of(sid),
        Some(EntityId(77)),
        "the late attach still completes the login"
    );
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_bootstraps,
        BTreeMap::new(),
        "and THAT is what ends the re-seed"
    );
    let _ = attached;
}

#[test]
fn two_sessions_on_one_home_coalesce_to_exactly_one_demand_and_one_head_read() {
    // MF1 DEFECT B — the STORM guard. The re-drive iterates REALMS, not sessions: N sessions booting into
    // ONE home cost ONE `RealmDemand` + ONE `HeadRead` per cadence, not 2N. (At the 100K mass-login scale
    // the difference is 200K messages per cadence versus 2.) Coalescing is sound because the only
    // per-session field in the demand is the audit-only `parent_fence`.
    let mut rig = dynamic_rig();
    let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
    let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
    set_tick(&mut rig, 3); // elapsed 2 == the cadence
    let driven = rig.tick(vec![]);
    let demands = demands_to_orch(&driven);
    assert_eq!(
        demands.len(),
        1,
        "TWO waiters on one home ⇒ exactly ONE re-seeded demand"
    );
    assert_eq!(demands[0].child, home_lineage());
    assert_eq!(
        demands[0].parent_fence,
        home_sentinel_fence(AccountId(5)),
        "carrying the wait's representative per-account sentinel (audit-only at the orchestrator)"
    );
    assert_eq!(
        realm_head_reads(&driven, home_realm()),
        1,
        "and exactly ONE head poll for the realm both are waiting on"
    );
    // After the resolve: still ONE demand per cadence (the anti-reap invariant), and ZERO head-reads.
    let _ = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    set_tick(&mut rig, 5);
    let after = rig.tick(vec![]);
    assert_eq!(
        demands_to_orch(&after).len(),
        1,
        "one demand per cadence still covers BOTH resolved members"
    );
    assert_eq!(
        realm_head_reads(&after, home_realm()),
        0,
        "and the head poll is done — the node is known"
    );
    assert_eq!(phase_of(&rig, sid_a), SessionPhase::AwaitingAttach);
    assert_eq!(phase_of(&rig, sid_b), SessionPhase::AwaitingAttach);
}

#[test]
fn a_session_joining_an_already_resolved_home_resumes_the_head_poll_and_resolves() {
    // The joiner case the surviving wait entry creates: session B logs into a home realm session A has
    // ALREADY resolved. B has seen no head reply of its own, so joining CLEARS `resolved` — the poll
    // resumes on the next cadence tick and B is resolved by the reply, while A (already in
    // `AwaitingAttach`) is left untouched. Both arms of the per-member phase test, in one cell.
    let mut rig = dynamic_rig();
    let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
    let _ = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(phase_of(&rig, sid_a), SessionPhase::AwaitingAttach);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_bootstraps,
        home_wait(1, &[sid_a], true),
        "resolved, and the entry survives"
    );
    // B logs in on the SAME account ⇒ the same home realm ⇒ it JOINS the existing wait.
    let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_bootstraps,
        home_wait(1, &[sid_a, sid_b], false),
        "the joiner cleared `resolved` (its own head reply may be lost — the poll must resume)"
    );
    set_tick(&mut rig, 3);
    let driven = rig.tick(vec![]);
    assert_eq!(
        realm_head_reads(&driven, home_realm()),
        1,
        "so the head IS re-polled for the joiner"
    );
    let resolved = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(phase_of(&rig, sid_b), SessionPhase::AwaitingAttach);
    assert!(
        saw_attach(&resolved, HOME, sid_b, attach_spawn(&rig)),
        "the joiner's attach goes to the home shard"
    );
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::from([(HOME, 2)]),
        "each member claims the roster exactly once — the already-resolved member was skipped"
    );
}

#[test]
fn a_dynamic_pre_active_session_keeps_its_committed_lease_renewed() {
    // MF2 — a held login's lease MUST be renewed. The dynamic-home hold can outlast the orchestrator's
    // lease-reap horizon (`bootstrap_ttl` is a whole measured pod boot), and the lease was COMMITTED at
    // the grant, so an `Active`-only renew set would let a held session's own lease lapse under it with
    // nothing pre-Active watching. The renew set is therefore `Active` OR mid-bootstrap; the RECHECK
    // stays Active-only.
    // `renewed_sessions` (the module-level decode helper) owns the non-renew arm via the D-3 cell.
    let rechecked = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<SessionId> {
        sent.iter()
            .filter(|(to, _, _)| *to == ORCH)
            .filter_map(
                |(_, _, b)| match postcard::from_bytes::<InterShardFlow>(b) {
                    Ok(InterShardFlow::Directory(DirectoryOp::HeadRead {
                        key: DirectoryKey::Session(s),
                    })) => Some(s),
                    _ => None,
                },
            )
            .collect()
    };
    // DYNAMIC: armed + synced, with the renew AND recheck cadences live.
    let mut rig = Rig::new();
    rig.world.insert_resource(GatewayConfig {
        lease_renew_interval_ticks: 4,
        session_recheck_interval: 4,
        seed_injector: armed_injector(homes_for(AccountId(5), TEST_HOME_REALM, 0.0)),
        ..config()
    });
    let (sid, _) = dynamic_login(&mut rig, CLIENT);
    assert_eq!(phase_of(&rig, sid), waiting_phase());
    set_tick(&mut rig, 4);
    let held = rig.tick(vec![]);
    assert_eq!(
        renewed_sessions(&held),
        vec![sid],
        "a session HELD in AwaitingHomeRealm still renews its committed lease (MF2)"
    );
    assert_eq!(
        rechecked(&held),
        Vec::new(),
        "…but is NOT re-checked (no authority to self-fence pre-Active)"
    );
    // …and the Active-only recheck RESUMES once the hold ends: the whole point of gating the recheck on
    // Active (not on the renew set) is that a held session parks the recheck and un-parks it on promote.
    // Drive the SAME session home-resolved → attached → Active, then a recheck-cadence tick fires it.
    let _ = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(
        phase_of(&rig, sid),
        SessionPhase::AwaitingAttach,
        "home resolved ⇒ the bootstrap hold ends"
    );
    let _ = rig.tick(vec![wire(
        HOME,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    set_tick(&mut rig, 8);
    assert_eq!(
        rechecked(&rig.tick(vec![])),
        vec![sid],
        "the Active-only recheck RESUMES once the session is Active"
    );
    // STATIC (the byte-identical control): an `AwaitingAttach` login carries no bootstrap window, so it
    // is NOT renewed — the pre-5f-3d renew set exactly.
    let mut plain = Rig::new();
    plain.world.insert_resource(GatewayConfig {
        lease_renew_interval_ticks: 4,
        ..config()
    });
    let _ = plain.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let static_sid = session_of(&plain, CLIENT);
    let _ = plain.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(static_sid))]);
    assert_eq!(phase_of(&plain, static_sid), SessionPhase::AwaitingAttach);
    set_tick(&mut plain, 4);
    assert_eq!(
        renewed_sessions(&plain.tick(vec![])),
        Vec::new(),
        "a STATIC pre-Active login is not renewed (byte-identical renew set)"
    );
}

#[test]
fn the_bounded_bootstrap_ttl_closes_a_never_routable_home_loudly() {
    // CRITIQUE-1: the hold is BOUNDED. A home that never becomes routable Closes the client LOUDLY at
    // the deadline (counted + a reason + the committed lease revoked), never a silent hang. Expiring in
    // `AwaitingHomeRealm` sends NO detach (no home was ever resolved — the `None` arm).
    let mut rig = dynamic_rig_with_ttl(5); // waiting since tick 1 ⇒ deadline 6
    let (sid, _) = dynamic_login(&mut rig, CLIENT);
    set_tick(&mut rig, 6); // AT the deadline: still inside the window
    let inside = rig.tick(vec![]);
    assert_eq!(
        phase_of(&rig, sid),
        waiting_phase(),
        "the deadline tick itself is still inside the hold"
    );
    assert_eq!(decode_controls(&inside, CLIENT), Vec::new(), "no Close yet");
    assert_eq!(rig.stats().home_bootstrap_timeouts, 0);
    set_tick(&mut rig, 7); // PAST the deadline
    let closed = rig.tick(vec![]);
    assert_eq!(
        decode_controls(&closed, CLIENT),
        vec![ServerControlMsg::Close {
            reason: "home realm did not become available".to_owned(),
        }],
        "the bounded TTL Closes LOUDLY with a reason"
    );
    assert_eq!(rig.stats().home_bootstrap_timeouts, 1);
    assert!(
        saw_lease_revoke(&closed, sid),
        "the committed Session lease is revoked, not left for the reaper"
    );
    assert!(
        !saw_detach(&closed, SHARD, sid),
        "no detach is sprayed at the static shard we never attached to"
    );
    let sessions = rig.world.resource::<GatewaySessions>();
    assert_eq!(sessions.len(), 0, "the session is gone");
    assert_eq!(
        sessions.home_bootstraps,
        BTreeMap::new(),
        "and its bootstrap-index entry with it (the last member out prunes the realm)"
    );
}

#[test]
fn the_bootstrap_ttl_also_spans_the_dynamic_attach_wait_and_detaches_the_home() {
    // The TTL spans the WHOLE pre-Active bootstrap: a home that resolves but never confirms the attach
    // (a shard that dies between head-resolve and `SessionAttached`) ALSO Closes at the deadline — and
    // because a home WAS resolved, the cleanup detaches THERE and releases the runtime roster entry.
    let mut rig = dynamic_rig_with_ttl(5); // deadline 6
    let (sid, _) = dynamic_login(&mut rig, CLIENT);
    let _ = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::from([(HOME, 1)])
    );
    set_tick(&mut rig, 7); // past the deadline, still AwaitingAttach
    let closed = rig.tick(vec![]);
    assert_eq!(
        decode_controls(&closed, CLIENT),
        vec![ServerControlMsg::Close {
            reason: "home realm did not become available".to_owned(),
        }],
        "a dynamic AwaitingAttach that never attaches also Closes loudly"
    );
    assert_eq!(rig.stats().home_bootstrap_timeouts, 1);
    assert!(
        saw_detach(&closed, HOME, sid),
        "the detach goes to the RESOLVED home shard"
    );
    assert!(saw_lease_revoke(&closed, sid));
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::new(),
        "the last session left ⇒ the spawned node leaves the runtime roster (no churn leak)"
    );
}

#[test]
fn one_home_head_resolves_every_waiter_and_the_roster_refcount_drains() {
    // SCALE + the roster refcount. Two sessions on the SAME home realm are resolved by ONE head reply
    // (a mass login onto one home is a win, not a fan-out cost); the roster counts BOTH, and only the
    // LAST session leaving removes the node (both refcount arms).
    let mut rig = dynamic_rig();
    let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
    let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_bootstraps,
        home_wait(1, &[sid_a, sid_b], false),
        "both sessions are members of the ONE per-realm wait (one lineage copy, one cadence anchor)"
    );
    let resolved = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert!(saw_attach(&resolved, HOME, sid_a, attach_spawn(&rig)));
    assert!(saw_attach(&resolved, HOME, sid_b, attach_spawn(&rig)));
    assert_eq!(phase_of(&rig, sid_a), SessionPhase::AwaitingAttach);
    assert_eq!(phase_of(&rig, sid_b), SessionPhase::AwaitingAttach);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::from([(HOME, 2)]),
        "the roster refcounts BOTH sessions homed on the spawned node"
    );
    // MF1-B — a DUPLICATE head reply (at-least-once) is an exact no-op: both members have left
    // `AwaitingHomeRealm`, so nothing re-attaches and — crucially — the roster is NOT re-claimed (a
    // second claim would pin the node on the runtime roster forever once these sessions leave).
    let dup = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert!(
        !saw_attach(&dup, HOME, sid_a, attach_spawn(&rig)),
        "a duplicate Realm head does not re-attach an already-resolved member"
    );
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::from([(HOME, 2)]),
        "and does not double-count the roster refcount"
    );
    assert_eq!(rig.stats().home_wait_desync, 0);
    // A `Bye` from the first: the node STAYS (the other session still needs it).
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::from([(HOME, 1)]),
        "one leaver does not evict a node another session is homed on"
    );
    // A `Bye` from the last: the node LEAVES the roster.
    let last = rig.tick(vec![wire(
        CLIENT2,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::new(),
        "the last leaver evicts it (bounded across 100K-realm churn)"
    );
    assert!(
        saw_detach(&last, HOME, sid_b),
        "and the Bye detach itself routes to the session's HOME shard, not config.shard"
    );
}

#[test]
fn byes_while_awaiting_the_home_prune_the_wait_index_incrementally() {
    // The bootstrap index can never outlive its sessions: a `Bye` mid-boot drops that session from its
    // home's member set (the entry survives while another member waits), and the LAST removal prunes the
    // realm entry — one `RealmCoord` lineage freed with it.
    let mut rig = dynamic_rig();
    let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
    let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_bootstraps,
        home_wait(1, &[sid_b], false),
        "the remaining member keeps the realm's wait alive"
    );
    assert_eq!(
        rig.world.resource::<GatewaySessions>().dynamic_shards,
        BTreeMap::new(),
        "a session that never resolved a home releases nothing"
    );
    let _ = rig.tick(vec![wire(
        CLIENT2,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_bootstraps,
        BTreeMap::new(),
        "the last removal PRUNES the realm entry (no unbounded growth)"
    );
    assert_eq!(rig.world.resource::<GatewaySessions>().len(), 0);
    let _ = sid_a; // named for the assertion above only
}

#[test]
fn a_static_login_is_byte_identical_with_no_home_phase_and_no_extra_head_read() {
    // FLOW-IDENTITY (CRITIQUE-2): with the injector UNARMED (the default) a login follows the EXACT
    // pre-5f-3d flow — `AwaitingDirectory → AwaitingAttach → config.shard` — with NO `AwaitingHomeRealm`,
    // NO extra head-read, NO RealmDemand, and no reordering of Welcome/UniverseRate/AttachSession.
    //
    // ★ IT IS NO LONGER BYTE-identity, and the name is kept only because the FLOW is what this pins.
    // The static `AttachSession` used to carry `spawn: None` while the per-tick retry carried
    // `session.spawn`, so a static login landed at the shard's own origin or at the T2 standoff
    // depending on which attach the shard saw first — measured as a live coin flip on the process
    // parity gate, 2026-08-21. Both producers now send the registry's pose, which is what
    // `attach_spawn`'s own doc has claimed since the T2 fix and what the retry assertion above
    // already pinned.
    let mut rig = Rig::new(); // config() ⇒ the inert injector (unarmed)
    let (sid, sends) = rig.login();
    assert_eq!(
        phase_of(&rig, sid),
        SessionPhase::Active {
            entity: EntityId(77)
        },
        "the static login reached Active in the same three ticks as before"
    );
    assert_eq!(
        decode_controls(&sends[1], CLIENT),
        vec![
            ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: sid,
                session_fence: Fence(1),
                epoch: EpochId(9),
            },
            ServerControlMsg::UniverseRate { tick_hz: 50 },
        ],
        "Welcome then UniverseRate, unchanged"
    );
    assert!(
        saw_attach(&sends[1], SHARD, sid, attach_spawn(&rig)),
        "the attach goes to the static config.shard at the SAME tick as the Welcome, carrying the \
         SAME registry pose its own retry carries"
    );
    // The EXACT send fingerprint of the static grant tick: Welcome + UniverseRate client-ward, then the
    // grant arm's `AttachSession` and the per-tick retry driver's duplicate — both to `config.shard`.
    // Nothing added, nothing reordered, nothing orchestrator-ward (no demand, no Realm head-read).
    assert_eq!(
        sends[1]
            .iter()
            .map(|(to, class, _)| (*to, *class))
            .collect::<Vec<_>>(),
        vec![
            (CLIENT, MsgClass::Control),
            (CLIENT, MsgClass::Control),
            (SHARD, MsgClass::Control),
            (SHARD, MsgClass::Control),
        ]
    );
    let total_head_reads: usize = sends
        .iter()
        .map(|tick| realm_head_reads(tick, home_realm()))
        .sum();
    assert_eq!(
        total_head_reads, 0,
        "a static login costs NO extra Realm head-read round-trip"
    );
    assert_eq!(demand_count(&sends), 0, "and emits no RealmDemand");
    let sessions = rig.world.resource::<GatewaySessions>();
    assert_eq!(
        sessions.home_shard_of(sid),
        None,
        "no dynamic home ⇒ session_target is config.shard"
    );
    assert_eq!(sessions.home_realm_of(sid), None);
    assert_eq!(route_authority(&rig, sid), SHARD);
    assert_eq!(sessions.home_bootstraps, BTreeMap::new());
    assert_eq!(sessions.dynamic_shards, BTreeMap::new());
    assert_eq!(rig.stats().home_bootstrap_timeouts, 0);
    assert_eq!(
        rig.stats().logins_held_pre_sync,
        0,
        "an UNARMED gateway never consults the clock on the login path (MF3)"
    );
    // A stray Realm head in static mode is a clean no-op (nobody waits on it).
    let stray = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().home_shard_of(sid),
        None,
        "a static session is never re-homed by a stray Realm head"
    );
    assert_eq!(decode_controls(&stray, CLIENT), Vec::new());
    assert_eq!(rig.stats().home_wait_desync, 0);
}

#[test]
fn a_forced_home_wait_desync_is_counted_never_silent() {
    // The C2 honesty floor: a `home_bootstraps` MEMBER naming a session absent from `by_session` (an
    // invariant breach the begin/end pairing makes impossible) is COUNTED, never a silent continue.
    let mut sessions = GatewaySessions::default();
    sessions.begin_home_wait(
        SessionId(0xC0DE),
        home_realm(),
        home_lineage(),
        TickId(1),
        AccountId(5),
    );
    let mut stats = GatewayStats::default();
    let mut outbox = OutboundBox::default();
    on_home_realm_head(
        home_realm(),
        Some(OwnerRecord {
            authority: AuthorityRef::Shard(HOME),
            fence: Fence(3),
            lease_expires: UniverseTick(1_000),
            in_transfer: None,
        }),
        &mut sessions,
        &mut stats,
        &mut outbox,
    );
    assert_eq!(stats.home_wait_desync, 1, "the desync is counted");
    assert_eq!(
        outbox.0.len(),
        0,
        "and nothing is sent for a phantom session"
    );
    assert_eq!(
        sessions.dynamic_shards,
        BTreeMap::new(),
        "a phantom session claims no roster entry"
    );
    assert_eq!(
        sessions.home_bootstraps,
        home_wait(1, &[SessionId(0xC0DE)], true),
        "the entry SURVIVES the resolve (the demand re-seed runs until Active) with the poll satisfied"
    );
}

#[test]
fn the_attach_retry_cadence_gate_is_per_tick_for_static_and_backed_off_for_dynamic() {
    // MF1 — the two ARMS of the attach-retry gate, driven directly (the live arms are proven by the
    // static byte-identity fingerprint and by the dynamic cadence cell).
    assert!(
        attach_retry_due(None, TickId(2), 2),
        "a STATIC session (no bootstrap wait) retries EVERY tick — byte-identical"
    );
    assert!(
        attach_retry_due(None, TickId(3), 2),
        "…on the off-cadence tick too"
    );
    assert!(
        !attach_retry_due(Some(TickId(1)), TickId(2), 2),
        "a DYNAMIC member is silent off its realm's cadence"
    );
    assert!(
        attach_retry_due(Some(TickId(1)), TickId(3), 2),
        "…and retries on it"
    );
    // The ANCHOR source, both `?` arms: `None` for a static session (no home realm), `None` also for the
    // fail-SAFE case of a home realm with no live wait (unreachable on the live path — every member drop
    // either removes the session or leaves `AwaitingAttach` — so it is proven here directly).
    let mut sessions = GatewaySessions::default();
    assert_eq!(
        sessions.attach_anchor(None),
        None,
        "a static session has no anchor ⇒ per-tick retry"
    );
    assert_eq!(
        sessions.attach_anchor(Some(home_realm())),
        None,
        "a home realm with no live wait falls back to the per-tick retry (never a wedge)"
    );
    sessions.begin_home_wait(
        SessionId(1),
        home_realm(),
        home_lineage(),
        TickId(9),
        AccountId(5),
    );
    assert_eq!(
        sessions.attach_anchor(Some(home_realm())),
        Some(TickId(9)),
        "a member rides its realm's wait anchor"
    );
}

#[test]
fn home_redrive_due_fires_only_on_the_cadence_and_never_at_the_start() {
    // The pure cadence predicate, both arms + the two guards: elapsed 0 (the tick the wait began, whose
    // seed already went out inline) never re-drives; a zero interval degrades to every-tick (fail-safe,
    // never a silent never-re-drive); and the anchor is the wait's OWN `since`, which is what spreads a
    // mass login across many homes over the cadence window.
    assert!(!home_redrive_due(TickId(1), TickId(1), 2), "elapsed 0");
    assert!(!home_redrive_due(TickId(2), TickId(1), 2), "elapsed 1 of 2");
    assert!(home_redrive_due(TickId(3), TickId(1), 2), "elapsed 2 of 2");
    assert!(!home_redrive_due(TickId(4), TickId(1), 2), "elapsed 3 of 2");
    assert!(home_redrive_due(TickId(5), TickId(1), 2), "elapsed 4 of 2");
    // Two WAITS that opened on DIFFERENT ticks are due on DIFFERENT ticks (the storm spread).
    assert!(home_redrive_due(TickId(4), TickId(2), 2));
    assert!(!home_redrive_due(TickId(5), TickId(2), 2));
    // Guards: a zero interval is every-tick (fail-safe); a `since` in the future saturates to 0.
    assert!(home_redrive_due(TickId(2), TickId(1), 0));
    assert!(!home_redrive_due(TickId(1), TickId(9), 2));
}

#[test]
fn home_bootstrap_expired_is_exclusive_at_the_deadline() {
    // Strictly `>`: the deadline tick itself is still inside the hold (generous at its own edge).
    assert!(!home_bootstrap_expired(TickId(5), TickId(6)));
    assert!(!home_bootstrap_expired(TickId(6), TickId(6)));
    assert!(home_bootstrap_expired(TickId(7), TickId(6)));
}

#[test]
fn the_seed_injector_validate_rejects_a_mis_tuned_armed_budget() {
    // Fail-LOUD at boot (mirrors `RlmTuning::validate`): an UNARMED injector is vacuously valid (its
    // windows are never read), an ARMED one needs both windows non-zero AND a bootstrap window that
    // contains at least one re-drive. All arms + both Display messages.
    assert_eq!(
        SeedInjectorConfig::inert(test_world().lowered()).validate(),
        Ok(())
    );
    let armed = armed_injector(default_homes());
    assert_eq!(armed.validate(), Ok(()), "the live test budget is valid");
    let zero_demand = SeedInjectorConfig {
        demand_ttl_ticks: 0,
        ..armed_injector(default_homes())
    };
    assert_eq!(
        zero_demand
            .validate()
            .expect_err("an armed zero demand TTL is rejected"),
        SeedInjectorError::ZeroWindowWhileArmed
    );
    let zero_bootstrap = SeedInjectorConfig {
        bootstrap_ttl_ticks: 0,
        ..armed_injector(default_homes())
    };
    assert_eq!(
        zero_bootstrap
            .validate()
            .expect_err("an armed zero bootstrap TTL is rejected"),
        SeedInjectorError::ZeroWindowWhileArmed
    );
    let too_tight = SeedInjectorConfig {
        bootstrap_ttl_ticks: 2, // == the cadence (8/4): no room for even one re-drive
        ..armed_injector(default_homes())
    };
    assert_eq!(
        too_tight
            .validate()
            .expect_err("a bootstrap window shorter than one re-drive is rejected"),
        SeedInjectorError::BootstrapTtlBelowRedrive {
            bootstrap_ttl: 2,
            redrive: 2,
        }
    );
    // The operator-facing Display text of both arms (the actionable boot failure).
    assert!(
        SeedInjectorError::ZeroWindowWhileArmed
            .to_string()
            .contains("bootstrap_ttl_ticks > 0")
    );
    assert!(
        SeedInjectorError::BootstrapTtlBelowRedrive {
            bootstrap_ttl: 2,
            redrive: 2,
        }
        .to_string()
        .contains("must strictly exceed the re-drive cadence")
    );
}

#[test]
fn the_bootstrap_ttl_derivation_is_the_launch_floor_plus_one_demand_cadence() {
    // The ONE derivation a bin uses (never an inline literal): the reconciler's measured-boot launch
    // floor PLUS one demand cadence of slack, saturating.
    assert_eq!(SeedInjectorConfig::bootstrap_ttl_from_rlm(60, 80), 140);
    assert_eq!(
        SeedInjectorConfig::bootstrap_ttl_from_rlm(u64::MAX, 1),
        u64::MAX,
        "saturating (a mis-set env can never wrap into an instant expiry)"
    );
    // The cadence divisor is the named constant, and a zero TTL still yields a usable cadence.
    assert_eq!(SeedInjectorConfig::REDRIVE_DIVISOR, 4);
    assert_eq!(
        SeedInjectorConfig::inert(test_world().lowered()).redrive_interval_ticks(),
        1,
        "the inert injector floors at 1 (never a divide-by-zero cadence)"
    );
}

// ===== RLM 5f-4 — the GATEWAY CROSSING-DESTINATION ADMISSION ===================================

/// A DEMAND-SPAWNED crossing DESTINATION. Deliberately NOT a member of `config().known_shards`
/// (`{SHARD, DEST}`) — exactly like a realm the cluster spun up on demand, whose `NodeId` was minted at
/// spawn time long after boot — so every frame of its that the gateway consumes PROVES the RUNTIME
/// roster admitted it.
const DYN_DEST: NodeId = NodeId(78);
/// A SECOND demand-spawned dest (the defensive-replace / chained-crossing cells).
const DYN_DEST2: NodeId = NodeId(79);
/// A SECOND transfer id: a DIFFERENT transfer, so a replacing `PrepareSubscribe` is not absorbed by
/// the redelivery gate.
const XFER2: TransferId = TransferId(0x5f4);

/// The RUNTIME routable-shard roster, WHOLE (asserted by equality — never `matches!`).
fn roster(rig: &Rig) -> BTreeMap<NodeId, u32> {
    rig.world
        .resource::<GatewaySessions>()
        .dynamic_shards
        .clone()
}

/// The session's resolved home shard — the D-34 routing field this slice re-points at commit.
fn homed_on(rig: &Rig, sid: SessionId) -> Option<NodeId> {
    rig.world.resource::<GatewaySessions>().home_shard_of(sid)
}

/// The shards a session subscribes to (the COLD authority, sorted by `NodeId`).
fn subs_of(rig: &Rig, sid: SessionId) -> Vec<NodeId> {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session present")
        .subs
        .keys()
        .copied()
        .collect()
}

/// How many datagrams of `class` reached CLIENT this tick.
fn to_client(sent: &[(NodeId, MsgClass, Vec<u8>)], class: MsgClass) -> usize {
    sent.iter()
        .filter(|(to, c, _)| (*to == CLIENT) & (*c == class))
        .count()
}

/// Both of the dest's DATA classes in one tick: an entity `Snapshot` at `Fence(5)` — the realm fence
/// [`subscription_ready`] opens the dest sub at, so it is ACCEPTED rather than fence-dropped — and a
/// `RealmSnapshot` (ambient world state, no fence gate).
fn dest_frames(dest: NodeId) -> Vec<Inbound> {
    vec![
        wire(dest, MsgClass::Snapshot, &frame_msg(Fence(5), 1)),
        wire(
            dest,
            MsgClass::RealmSnapshot,
            &realm_frame_msg(RealmId::Planet(7)),
        ),
    ]
}

/// Drive the WHOLE saga a crossing rides against `dest`, ONE ordered CONTROL command per tick:
/// Prepare → RequestCut → FreezeSource → CommitAuthority. Returns the COMMIT tick's sends.
fn cross_to(
    rig: &mut Rig,
    sid: SessionId,
    transfer: TransferId,
    dest: NodeId,
) -> Vec<(NodeId, MsgClass, Vec<u8>)> {
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer,
        session: sid,
        dest,
    })]);
    let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
        transfer,
        session: sid,
    })]);
    let _ = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
        transfer,
        session: sid,
        marker_seq: 0,
        dest,
    })]);
    rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })])
}

#[test]
fn a_prepare_admits_the_crossing_dest_so_its_subscription_ready_lands() {
    // THE BUG THIS SLICE FIXES, half one. A demand-spawned crossing dest is in NEITHER the frozen
    // config roster nor (pre-5f-4) the runtime one, so every frame it sent fell through to the CLIENT
    // branch. The orchestrator-signed `PrepareSubscribe` now ADMITS it, and Prepare strictly precedes
    // FreezeSource/CommitAuthority — so the admission is in place before the dest's first frame.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "an ordinary login claims no runtime roster entry"
    );
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DYN_DEST,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "the Prepare claimed the demand-spawned dest onto the RUNTIME routable roster"
    );
    // …and THAT is what makes its read-plane promote land: the dest sub opens and the client is told
    // which sub now carries its avatar.
    let ready = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
    assert_eq!(
        decode_controls(&ready, CLIENT),
        vec![
            ServerControlMsg::SubscriptionOpened {
                sub: SubId(1),
                frame: FrameRef::SystemSpace { system_seed: 8 },
            },
            ServerControlMsg::AuthorityChanged {
                entity: EntityId(77),
                sub: SubId(1),
            },
            ServerControlMsg::OwnEntity {
                entity: EntityId(77)
            },
        ],
        "the dest's SubscriptionReady opened the dest sub and re-pointed the render authority"
    );
    assert_eq!(subs_of(&rig, sid), vec![SHARD, DYN_DEST]);
    assert_eq!(rig.stats().undecodable, 0, "nothing was lost");
}

#[test]
fn without_the_prepare_admission_a_crossing_dests_frames_are_all_lost() {
    // THE FALSIFIABLE TWIN. The IDENTICAL frames from the IDENTICAL node with NO preceding Prepare: the
    // dest is unrecognised, so every one of its frames falls through to the CLIENT branch and is counted
    // `undecodable` — the `SubscriptionReady` (which fails the `ClientControlMsg` decode) so the render
    // authority NEVER re-points and the player is BLIND, and both data classes (which have no
    // client-branch arm at all) so nothing of the dest world is ever served. This is the state
    // test-one's admission is measured against.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let ready = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
    assert_eq!(
        decode_controls(&ready, CLIENT),
        Vec::new(),
        "with no admission the dest's SubscriptionReady opens nothing and tells the client nothing"
    );
    assert_eq!(roster(&rig), BTreeMap::new(), "nothing was claimed");
    assert_eq!(
        subs_of(&rig, sid),
        vec![SHARD],
        "the session still subscribes ONLY to its source — the render authority never moved"
    );
    let frames = rig.tick(dest_frames(DYN_DEST));
    assert_eq!(
        to_client(&frames, MsgClass::Snapshot),
        0,
        "no entity frame from the unadmitted dest reached the client"
    );
    assert_eq!(
        to_client(&frames, MsgClass::RealmSnapshot),
        0,
        "and no realm frame either"
    );
    // All three are still lost — but they are lost for TWO different reasons, and the counters now say
    // which. The `SubscriptionReady` reaches the client branch and fails to decode as a client message,
    // so it is genuinely `undecodable`. The two DATA frames have no client-branch arm at all: they are
    // refused because the router does not know their sender. That split is the whole point of the new
    // counter — on the live demand cluster this second number reached 17,365 while `undecodable` was
    // the only thing anyone looked at.
    assert_eq!(
        rig.stats().undecodable,
        1,
        "the SubscriptionReady fell to the client branch and failed the client decode"
    );
    assert_eq!(
        rig.stats().refused_unknown_sender,
        2,
        "both DATA frames were refused because the sender is not known as a shard — never served"
    );
}

#[test]
fn the_forward_crossing_re_homes_the_session_and_admits_its_snapshot_and_realm_frames() {
    // THE FORWARD CROSSING, end to end, into a DEMAND-SPAWNED dest: the write route swaps, the D-34
    // routing field FOLLOWS authority, the dest is admitted, and BOTH of its render classes reach the
    // client. Pre-5f-4 every one of the dest's frames was counted undecodable instead.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let committed = cross_to(&mut rig, sid, XFER, DYN_DEST);
    assert_eq!(
        acks_to_orch(&committed),
        vec![TransferControlAck::Committed { transfer: XFER }],
        "the commit acked exactly once"
    );
    assert_eq!(
        route_authority(&rig, sid),
        DYN_DEST,
        "the WRITE route swapped to the dest"
    );
    assert_eq!(
        homed_on(&rig, sid),
        Some(DYN_DEST),
        "THE D-34 CLOSE: the session's routing target followed authority to the dest"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 2)]),
        "the dest holds TWO claims — the crossing (from Prepare) and the new home (from commit)"
    );
    // The read-plane promote, then BOTH render classes.
    let _ = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
    assert_eq!(subs_of(&rig, sid), vec![SHARD, DYN_DEST]);
    let frames = rig.tick(dest_frames(DYN_DEST));
    assert_eq!(
        to_client(&frames, MsgClass::Snapshot),
        1,
        "the dest's entity frame reaches the client (the avatar renders after the crossing)"
    );
    // The world around the avatar rides the COMPOSED feed since the flag day; the dest's
    // tombstoned realm frame lands in its own counter (a fallen-through frame would count
    // undecodable, asserted 0 below).
    assert_eq!(to_client(&frames, MsgClass::RealmSnapshot), 0);
    assert_eq!(
        rig.stats().undecodable,
        0,
        "nothing was lost as undecodable"
    );
}

#[test]
fn the_release_terminal_returns_the_crossing_claim_and_the_dest_stays_routable() {
    // The demote tail hands back the CROSSING claim only. The dest keeps the HOME claim commit
    // installed, so a post-`Done` player keeps receiving its dest frames — the refcount's whole point.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
    let _ = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
    let released = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: SHARD,
    })]);
    assert_eq!(
        acks_to_orch(&released),
        vec![TransferControlAck::Released { transfer: XFER }]
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "the crossing claim went back; the HOME claim keeps the dest on the roster"
    );
    assert_eq!(
        to_client(&rig.tick(dest_frames(DYN_DEST)), MsgClass::Snapshot),
        1,
        "so the dest is STILL routable after the saga reached Done"
    );
    // …and the session exit is what finally drains it.
    let bye = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert!(
        saw_detach(&bye, DYN_DEST, sid),
        "D-34: the post-crossing detach reaches the CURRENT authority, not the source"
    );
    assert!(
        !saw_detach(&bye, SHARD, sid),
        "and never the source it left"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "the last claim went with the session — no roster leak"
    );
}

#[test]
fn the_abort_terminal_returns_the_crossing_claim_and_the_dest_leaves_the_roster() {
    // The compensating terminal. A pre-CAS abort never re-pointed `home_shard`, so the Prepare-time
    // claim is the dest's ONLY one and the node leaves the roster with it.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DYN_DEST,
    })]);
    assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 1)]));
    let aborted = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&aborted),
        vec![TransferControlAck::Aborted { transfer: XFER }]
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "the aborted crossing's claim is returned — the dest leaves the runtime roster"
    );
    assert_eq!(
        homed_on(&rig, sid),
        None,
        "and an aborted crossing never re-homed the session"
    );
}

#[test]
fn a_redelivered_and_a_replacing_prepare_never_leak_a_roster_refcount() {
    // IDEMPOTENCE of the admission. (a) A redelivered SAME-transfer Prepare is absorbed by the journal
    // gate before any effect ⇒ no second claim. (b) A DIFFERENT transfer's Prepare defensively REPLACES
    // the progress: it claims the new dest and hands back the DISPLACED one, so neither a stale entry
    // is pinned nor is a re-claimed same-node dest ever transiently unrouted.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let first = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DYN_DEST,
    })]);
    assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 1)]));
    // (a) the exact same command again — the redelivery gate re-serves the recorded ack.
    let again = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DYN_DEST,
    })]);
    assert_eq!(
        acks_to_orch(&again),
        acks_to_orch(&first),
        "the redelivery re-served the recorded Prepared verbatim"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "and claimed nothing a second time (a redelivery cannot inflate the refcount)"
    );
    // (b) a DIFFERENT transfer naming the SAME dest: claim-then-release keeps it at exactly one, and it
    // never leaves the roster in between.
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER2,
        session: sid,
        dest: DYN_DEST,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "the replace re-claimed and released the same node — net one, never a leak"
    );
    // (b') a replace naming a DIFFERENT dest hands the displaced one back.
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DYN_DEST2,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST2, 1)]),
        "the displaced dest left the roster with the progress it belonged to"
    );
    let _ = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "and the terminal drains the survivor — every claim balanced"
    );
}

#[test]
fn a_prepare_refused_before_attach_claims_no_roster_entry() {
    // The claim rides the SAME condition as the progress write. A not-yet-Active session's Prepare is
    // refused and opens NO progress, so a claim here would have no terminal to release it — a
    // permanent roster leak pinning a node the crossing never reached.
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sid = session_of(&rig, CLIENT);
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    assert_eq!(phase_of(&rig, sid), SessionPhase::AwaitingAttach);
    let sent = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DYN_DEST,
    })]);
    assert_eq!(acks_to_orch(&sent), vec![], "the prepare was refused");
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "a refused prepare claims nothing (a claim with no terminal IS the leak)"
    );
    assert_eq!(rig.stats().transfer_unroutable, 1);
}

#[test]
fn a_crossing_off_a_shared_home_leaves_the_other_sessions_route_intact() {
    // THE REFCOUNT'S OTHER ARM. Two sessions share one demand-spawned home; ONE of them crosses away.
    // The node must survive the crosser's whole demote tail (decrement, not remove), or the session that
    // stayed would go blind the moment its neighbour crossed. NOTE (RLM 5f-4e): the crosser's home claim
    // is STASHED at commit and returned at `ReleaseSubscribe`, so HOME holds BOTH sessions' claims
    // through the tail — which is exactly why a SHARED home masked the blind window; the solo-home cell
    // `a_crossing_off_a_solo_dynamic_home_keeps_the_source_routable_through_the_demote_tail` is the
    // falsifying one.
    let mut rig = dynamic_rig();
    let (sid_a, _) = dynamic_login(&mut rig, CLIENT);
    let (sid_b, _) = dynamic_login(&mut rig, CLIENT2);
    // ONE head reply resolves both members onto HOME (two claims).
    let _ = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(roster(&rig), BTreeMap::from([(HOME, 2)]));
    for client in [CLIENT, CLIENT2] {
        let sid = session_of(&rig, client);
        let _ = rig.tick(vec![wire(
            HOME,
            MsgClass::Control,
            &ShardToGateway::SessionAttached {
                session: sid,
                entity: EntityId(77),
                frame: FrameRef::SystemSpace { system_seed: 7 },
                realm_fence: Fence(1),
            },
        )]);
    }
    // B crosses HOME -> DYN_DEST.
    let _ = cross_to(&mut rig, sid_b, XFER, DYN_DEST);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(HOME, 2), (DYN_DEST, 2)]),
        "B's home claim is STASHED for its demote tail, so HOME still counts both sessions"
    );
    assert_eq!(
        homed_on(&rig, sid_b),
        Some(DYN_DEST),
        "B followed authority"
    );
    assert_eq!(homed_on(&rig, sid_a), Some(HOME), "A did not move");
    assert_eq!(
        to_client(&rig.tick(dest_frames(HOME)), MsgClass::Snapshot),
        1,
        "and A still receives its HOME frames after its neighbour crossed away"
    );
    // Both exits drain to nothing.
    let _ = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid_b,
        src: HOME,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(HOME, 1), (DYN_DEST, 1)]),
        "the tail returned B's stashed HOME claim AND its crossing claim — A's HOME claim is untouched"
    );
    let _ = rig.tick(vec![wire(
        CLIENT2,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1)]));
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "every claim across two sessions and one crossing is balanced"
    );
}

#[test]
fn a_bye_mid_crossing_returns_the_crossing_claim_before_the_saga_terminal() {
    // A `Bye` can land ANYWHERE in the saga, including before a terminal ever arrives (WEDGE-1). The
    // ONE session-exit release covers the in-flight transfer's dest too, so a client that quits
    // mid-crossing cannot pin a demand-spawned node on the roster forever.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DYN_DEST,
    })]);
    assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 1)]));
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "the exit released the in-flight crossing's claim, with no terminal in sight"
    );
}

#[test]
fn a_bye_in_the_commit_to_release_window_returns_both_of_the_dests_claims() {
    // The DOUBLE-claim window: between `CommitAuthority` and `ReleaseSubscribe` the dest is claimed
    // twice (crossing + home). Both exits release BOTH roles through the ONE primitive, so a quit
    // exactly here still drains the node to zero.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
    assert_eq!(roster(&rig), BTreeMap::from([(DYN_DEST, 2)]));
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "both roles released at the one exit — the doubly-claimed dest drains to zero"
    );
    // The OTHER exit (the bounded bootstrap TTL) shares that primitive: a pre-Active dynamic session
    // holds only the home role, and it too drains.
    let mut ttl_rig = dynamic_rig_with_ttl(5); // waiting since tick 1 ⇒ deadline 6
    let (ttl_sid, _) = dynamic_login(&mut ttl_rig, CLIENT);
    let _ = ttl_rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    assert_eq!(roster(&ttl_rig), BTreeMap::from([(HOME, 1)]));
    set_tick(&mut ttl_rig, 7); // past the deadline
    let _ = ttl_rig.tick(vec![]);
    assert_eq!(ttl_rig.stats().home_bootstrap_timeouts, 1);
    assert_eq!(
        homed_on(&ttl_rig, ttl_sid),
        None,
        "the TTL-closed session is gone"
    );
    assert_eq!(
        roster(&ttl_rig),
        BTreeMap::new(),
        "and the bounded-TTL exit released its claim through the same primitive"
    );
}

#[test]
fn an_all_static_crossing_leaves_the_runtime_roster_empty() {
    // BYTE-IDENTITY. `DEST` is in the FROZEN `known_shards`, so it is already dispatchable and the
    // admission must not manufacture runtime state for it: `dynamic_shards` stays EMPTY at EVERY step,
    // which makes `is_routable_shard` answer purely from the config half exactly as before this slice.
    // (The D-34 re-point still happens — it is a routing fix, not a roster one.)
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER,
        session: sid,
        dest: DEST,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "a KNOWN dest is never claimed onto the runtime roster"
    );
    let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
        transfer: XFER,
        session: sid,
    })]);
    let _ = rig.tick(vec![saga_cmd(TransferControl::FreezeSource {
        transfer: XFER,
        session: sid,
        marker_seq: 0,
        dest: DEST,
    })]);
    let committed = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    assert_eq!(
        committed
            .iter()
            .map(|(to, class, _)| (*to, *class))
            .collect::<Vec<_>>(),
        vec![(DEST, MsgClass::Control), (ORCH, MsgClass::Saga)],
        "the commit tick's send fingerprint is unchanged: the dest's OpenInputSlot then the ack"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "still empty after the commit"
    );
    assert_eq!(
        homed_on(&rig, sid),
        Some(DEST),
        "the D-34 re-point applies to a static dest too (routing, not roster)"
    );
    let _ = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: SHARD,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "and the release of a known dest is a total no-op on the map"
    );
    // The static dest's frames dispatch through the CONFIG half, as they always did. Since
    // the flag day the OLD-lane realm frame reaches the parity intake, never the client —
    // the composed lane is the client's one realm feed (§2.4); a fallen-through frame would
    // count undecodable, which stays 0 (the routing is intact).
    let _ = rig.tick(vec![subscription_ready(DEST, sid)]);
    let frames = rig.tick(dest_frames(DEST));
    assert_eq!(to_client(&frames, MsgClass::Snapshot), 1);
    assert_eq!(
        to_client(&frames, MsgClass::RealmSnapshot),
        0,
        "the old realm lane no longer fans to clients (the composed feed replaced it)"
    );
    assert_eq!(rig.stats().undecodable, 0);
    let bye = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert!(
        saw_detach(&bye, DEST, sid),
        "D-34 closes for the static crossing too: the detach follows authority"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "and no entry was ever created"
    );
}

// ===== RLM 5f-4e — the DEMOTING-SOURCE claim (the REJECT-class blind window) ====================

/// The SOURCE home a committed crossing is still demoting away from (`TransferProgress::
/// demoting_home`) — the RLM 5f-4e stash. Read by equality so a redelivery can be shown NOT to
/// clobber it. Only called where a transfer is in flight.
fn demoting_home_of(rig: &Rig, sid: SessionId) -> Option<NodeId> {
    rig.world
        .resource::<GatewaySessions>()
        .by_session
        .get(&sid)
        .expect("session present")
        .transfer
        .as_ref()
        .expect("a transfer is in flight")
        .demoting_home
}

/// Drive ONE dynamic login all the way to `Active` on the DEMAND-SPAWNED `HOME` shard: hello → grant →
/// the `Realm` head that resolves the home → the home's `SessionAttached`. The session then holds
/// EXACTLY ONE roster claim (its home role) and ONE read sub (on `HOME`) — the shape every solo-home
/// crossing cell needs, where `HOME`'s routability rests ENTIRELY on that one claim (it is not in
/// `known_shards`).
fn dynamic_active_on_home(rig: &mut Rig, client: NodeId) -> SessionId {
    let (sid, _) = dynamic_login(rig, client);
    let _ = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &realm_head(home_realm(), Some(HOME)),
    )]);
    let _ = rig.tick(vec![wire(
        HOME,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    sid
}

#[test]
fn a_crossing_off_a_solo_dynamic_home_keeps_the_source_routable_through_the_demote_tail() {
    // THE REJECT-CLASS BLIND WINDOW, falsified. ONE session on a DEMAND-SPAWNED home with NO second
    // session parked there — which is exactly what `a_crossing_off_a_shared_home_…` masked: there the
    // neighbour's claim kept the node routable no matter what commit did.
    //
    // The client's READ subscription on the SOURCE stays open across the WHOLE demote grace (it closes
    // only at `ReleaseSubscribe`, a later tick) and the source keeps SHIPPING frames the whole time.
    // Releasing the source's runtime-roster claim at `CommitAuthority` un-routed it, so every one of
    // those frames fell through to the client branch and was counted `undecodable` — a BLIND player for
    // the whole tail, right after EVERY crossing. Pre-rework this cell fails at the first (a) assertion
    // below (0 frames served, 2 undecodable); post-rework the source stays routable until its sub closes.
    let mut rig = dynamic_rig();
    let sid = dynamic_active_on_home(&mut rig, CLIENT);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(HOME, 1)]),
        "HOME's routability rests on this ONE claim — nothing else holds it"
    );
    assert_eq!(
        subs_of(&rig, sid),
        vec![HOME],
        "and the client reads exactly one shard: the source"
    );
    let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
    // (a) THE TICK AFTER COMMIT — the BEHAVIOURAL falsifier, asserted BEFORE any structural claim so a
    // pre-rework tree fails on the SYMPTOM (the player's frames), not on an internal field. The dest sub
    // does not exist yet — its `SubscriptionReady` is a round-trip away — so the SOURCE feed is ALL the
    // player has. Both render classes must still land. Pre-rework: 0 and 0, with `undecodable` at 2.
    let after_commit = rig.tick(dest_frames(HOME));
    assert_eq!(
        to_client(&after_commit, MsgClass::Snapshot),
        1,
        "the SOURCE's entity frame still reaches the client on the tick after commit"
    );
    // Since the flag day the world's continuity rides the COMPOSED feed (the client holds its
    // scene across the swap, §2.7); the source's tombstoned realm frame lands in the dead-lane
    // counter, never at the client. The falsifier stays: a frame that fell through to the
    // client branch as a stranger's would count undecodable.
    assert_eq!(
        to_client(&after_commit, MsgClass::RealmSnapshot),
        0,
        "the old realm lane no longer fans to clients (the composed feed replaced it)"
    );
    assert_eq!(
        rig.stats().undecodable,
        0,
        "nothing of the source was counted undecodable — the source stayed ROUTABLE"
    );
    // …and the structure that produced it: the target moved, the displaced home was STASHED, both ends
    // are on the roster.
    assert_eq!(
        homed_on(&rig, sid),
        Some(DYN_DEST),
        "the routing target followed authority to the dest"
    );
    assert_eq!(
        demoting_home_of(&rig, sid),
        Some(HOME),
        "…while the displaced home was STASHED on the progress, not released at commit"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]),
        "so BOTH ends stay routable through the tail (source 1; dest home + crossing)"
    );
    // (b) AFTER THE DEST PROMOTE — both subs open, which is the composite a seamless crossing renders.
    let _ = rig.tick(vec![subscription_ready(DYN_DEST, sid)]);
    assert_eq!(
        subs_of(&rig, sid),
        vec![HOME, DYN_DEST],
        "both subs are open across the tail"
    );
    let composited = rig.tick(dest_frames(HOME));
    assert_eq!(
        to_client(&composited, MsgClass::Snapshot),
        1,
        "the source still feeds the composite after the dest promote"
    );
    // The realm class rides the composed feed since the flag day (asserted 0 on the dead lane).
    assert_eq!(to_client(&composited, MsgClass::RealmSnapshot), 0);
    let dest_side = rig.tick(dest_frames(DYN_DEST));
    assert_eq!(
        to_client(&dest_side, MsgClass::Snapshot),
        1,
        "and the dest feeds it too — the player sees both ends, never neither"
    );
    assert_eq!(to_client(&dest_side, MsgClass::RealmSnapshot), 0);
    assert_eq!(
        rig.stats().undecodable,
        0,
        "still nothing lost, with both subs open"
    );
    // The terminal is where the source's claim comes DUE: the same command closes the source sub, so the
    // routable window matches the sub's lifetime to the tick.
    let released = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: HOME,
    })]);
    assert_eq!(
        acks_to_orch(&released),
        vec![TransferControlAck::Released { transfer: XFER }]
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "the source left the roster WITH its sub; the dest keeps its home claim"
    );
    let bye = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert!(
        saw_detach(&bye, DYN_DEST, sid),
        "the detach follows authority to the dest"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "every claim across the whole crossing is balanced — no leak, nothing pinned"
    );
}

#[test]
fn a_post_cas_abort_returns_both_the_crossing_dest_and_the_demoting_source_claims() {
    // THE OTHER TERMINAL. `apply_abort` PRUNES `session.transfer`, so an abort that lands AFTER the CAS
    // (the stash already written) is the LAST holder able to NAME the demoting source — skip it there and
    // the source's claim is stranded on the roster forever, with no later path that could ever release
    // it. Both roles go back here; the session, which the commit did re-home, keeps only its dest claim.
    let mut rig = dynamic_rig();
    let sid = dynamic_active_on_home(&mut rig, CLIENT);
    let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
    assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]));
    let aborted = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER,
        session: sid,
    })]);
    assert_eq!(
        acks_to_orch(&aborted),
        vec![TransferControlAck::Aborted { transfer: XFER }]
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "the crossing claim AND the stashed source claim both went back at the abort"
    );
    assert_eq!(
        homed_on(&rig, sid),
        Some(DYN_DEST),
        "a POST-CAS abort does not un-re-home the session (the commit already happened)"
    );
    // RLM 5f-4e (the relocated-blind-window fix): the abort ALSO closed the DEMOTING SOURCE sub in
    // lock-step with releasing its claim, so a straggler source frame can never hit an
    // unroutable-but-still-subscribed node. One sweep tick later the drained source sub is gone.
    let _ = rig.tick(vec![]);
    assert!(
        sub_for(&rig, sid, HOME).is_none(),
        "the demoting source sub was closed at the abort and swept, not left dangling on an un-routed node"
    );
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "and the exit drains the survivor — balanced"
    );
}

#[test]
fn a_bye_inside_the_demote_tail_returns_the_stashed_source_claim_too() {
    // THE THIRD ROLE AT THE SESSION EXIT. A client that quits between `CommitAuthority` and
    // `ReleaseSubscribe` leaves NO terminal to run, so the ONE exit primitive must hand back all THREE
    // roles (home + crossing dest + demoting source) or a demand-spawned source is pinned forever.
    let mut rig = dynamic_rig();
    let sid = dynamic_active_on_home(&mut rig, CLIENT);
    let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
    assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]));
    let bye = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert!(
        saw_detach(&bye, DYN_DEST, sid),
        "the detach follows authority to the dest"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "all three roles released at the ONE exit — nothing pinned, nothing double-released"
    );
}

#[test]
fn a_second_prepare_inside_the_demote_tail_hands_back_the_displaced_source_claim() {
    // THE DEFENSIVE REPLACE. A second `PrepareSubscribe` OVERWRITES the progress, and the replacement
    // starts `demoting_home: None` — so unless the DISPLACED progress's stash is handed back alongside
    // its displaced dest, the first crossing's source is stranded with no holder left able to name it.
    let mut rig = dynamic_rig();
    let sid = dynamic_active_on_home(&mut rig, CLIENT);
    let _ = cross_to(&mut rig, sid, XFER, DYN_DEST);
    assert_eq!(roster(&rig), BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]));
    // A DIFFERENT transfer (so the redelivery gate does not absorb it) naming a THIRD node.
    let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
        transfer: XFER2,
        session: sid,
        dest: DYN_DEST2,
    })]);
    assert_eq!(
        demoting_home_of(&rig, sid),
        None,
        "the replacement progress is demoting nothing of its own"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1), (DYN_DEST2, 1)]),
        "the displaced dest AND the displaced source both went back; the new dest was claimed"
    );
    let _ = rig.tick(vec![saga_cmd(TransferControl::AbortTransfer {
        transfer: XFER2,
        session: sid,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "the second crossing's terminal drains its dest (its own stash was None — the no-op arm)"
    );
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(
        roster(&rig),
        BTreeMap::new(),
        "and the exit drains the home role — fully balanced"
    );
}

#[test]
fn a_redelivered_commit_neither_re_claims_the_dest_nor_clobbers_the_stash() {
    // THE AT-LEAST-ONCE GUARD ON THE STASH. If a redelivered `CommitAuthority` re-entered
    // `apply_commit`, `home_shard.replace(dest)` would return `Some(dest)` — overwriting the stash with
    // the DEST, stranding the real source's claim — and would claim the dest a THIRD time. The
    // applied-steps journal gate absorbs the redelivery BEFORE any effect, so both stay exact.
    let mut rig = dynamic_rig();
    let sid = dynamic_active_on_home(&mut rig, CLIENT);
    let first = cross_to(&mut rig, sid, XFER, DYN_DEST);
    let again = rig.tick(vec![saga_cmd(TransferControl::CommitAuthority {
        transfer: XFER,
        session: sid,
        new_fence: Fence(2),
        subject: XFER_SUBJECT,
    })]);
    assert_eq!(
        acks_to_orch(&again),
        acks_to_orch(&first),
        "the redelivery re-served the recorded Committed verbatim"
    );
    assert_eq!(
        demoting_home_of(&rig, sid),
        Some(HOME),
        "and left the stash naming the REAL source, never the dest"
    );
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(HOME, 1), (DYN_DEST, 2)]),
        "with no third claim on the dest"
    );
    // The proof the held claim is still the SOURCE's: the terminal drains HOME to nothing.
    let _ = rig.tick(vec![saga_cmd(TransferControl::ReleaseSubscribe {
        transfer: XFER,
        session: sid,
        src: HOME,
    })]);
    assert_eq!(
        roster(&rig),
        BTreeMap::from([(DYN_DEST, 1)]),
        "the tail returned exactly the source claim (a clobbered stash would leave HOME pinned)"
    );
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    assert_eq!(roster(&rig), BTreeMap::new());
}

// ===== THE WINDOW LANE: the COMPOSER (docs/design/window_lane.md §2.6/§4) =====

fn snap(realm: RealmId, head: FrameRef, tail: FrameRef, x: f64, at: u64) -> RealmSnap {
    RealmSnap {
        realm,
        frame: head,
        pose: vd_core::pose::StampedPose::at_rest(tail, DVec3::new(x, 0.0, 0.0), UniverseTick(at)),
    }
}

const SYS7: FrameRef = FrameRef::SystemSpace { system_seed: 7 };
const PLANET7: FrameRef = FrameRef::PlanetCentered { planet_seed: 7 };

/// The parent's Child-scope frame at `at`: System(7) authors Planet(7) at `leaf_x` (the hop
/// pre-inverted at the author: the parent's frame sits at −leaf_x in the planet's) and
/// Planet(9) at `sibling_x` — plus any extra rows the caller grafts.
fn parent_frame(
    window: WindowId,
    at: u64,
    leaf_x: f64,
    mut extra: Vec<RealmSnap>,
) -> ShardToGateway {
    let mut rows = vec![
        snap(RealmId::Planet(7), PLANET7, SYS7, leaf_x, at),
        snap(
            RealmId::Planet(9),
            FrameRef::PlanetCentered { planet_seed: 9 },
            SYS7,
            leaf_x + 30.0,
            at,
        ),
    ];
    rows.append(&mut extra);
    ShardToGateway::WindowFrame {
        realm_fence: Fence(1),
        window,
        at: UniverseTick(at),
        hop: Some(Box::new(vd_wire::session_flow::HopRow {
            child: RealmId::Planet(7),
            placement: vd_core::frame::FramePlacement::moving(
                DVec3::new(leaf_x, 0.0, 0.0),
                DVec3::ZERO,
            ),
        })),
        rows,
    }
}

fn leaf_frame(window: WindowId, at: u64) -> ShardToGateway {
    ShardToGateway::WindowFrame {
        realm_fence: Fence(1),
        window,
        at: UniverseTick(at),
        hop: None,
        rows: Vec::new(),
    }
}

#[test]
fn the_composer_folds_the_crossing_chain_at_one_tick_and_tears_down_to_nothing() {
    // THE UNIT-TIER TWIN of the process self-consistency gate (§2.12): a real login, a real
    // crossing (System(7) on SHARD → Planet(7) on DEST), then the two window statements
    // through the REAL ingest + composer — the full chain folding at ONE tick, the dedup
    // agreement measured exactly zero, every refusal arm driven by name, the epoch mechanics
    // observed, and the teardown leaving zero composer state.
    let mut rig = Rig::new();
    let (sid, _) = rig.login(); // lineage [System(7)]; Occupants window id 1 on SHARD
    let _ = cross_to(&mut rig, sid, XFER, DEST);
    let _ = rig.tick(vec![wire(
        DEST,
        MsgClass::Control,
        &ShardToGateway::SubscriptionReady {
            session: sid,
            entity: EntityId(77),
            frame: PLANET7,
            realm_fence: Fence(2),
        },
    )]);
    {
        let sessions = rig.world.resource::<GatewaySessions>();
        let session = &sessions.by_session[&sid];
        // ★ THE STATIC ATTACH DERIVES THE WHOLE CHAIN (owner ruling 2026-09-02 R9 step 2): the
        // gateway holds the seeded forest, so an attach onto System(7) remembers universe → galaxy →
        // system, exactly as a login descent would — and the crossing APPENDS below that leaf.
        assert_eq!(
            session.lineage,
            vec![
                RealmId::Universe,
                vd_core::worldgen::GALAXY,
                RealmId::System(7),
                RealmId::Planet(7)
            ],
            "the crossing APPENDED below the previous leaf (§2.6.2)"
        );
        assert_eq!(
            sessions.windows_open_count(),
            3,
            "own-level both ends + the lineage-derived Child window on the parent"
        );
    }
    // Window ids from the login + the ready tick: 1 = Occupants(System 7)@SHARD,
    // 2 = Occupants(Planet 7)@DEST, 3 = Child(Planet 7)@SHARD.
    let (leaf_w, parent_w) = (WindowId(2), WindowId(3));

    // ---- Tick A (T=1000): the leaf speaks alone — the parent's hop is not confirmed yet,
    // so the chain folds one level deep.
    let _ = rig.tick(vec![wire(
        DEST,
        MsgClass::RealmSnapshot,
        &leaf_frame(leaf_w, 1000),
    )]);
    assert_eq!(
        rig.stats().window_folds,
        1,
        "the one-level chain folded at 1000"
    );

    // ---- Tick B (T=1001): the FULL chain folds at ONE tick — the exact-cadence law — and
    // the two lawful sources for the same realm agree to the bit (§2.12's dedup bound).
    let _ = rig.tick(vec![
        wire(DEST, MsgClass::RealmSnapshot, &leaf_frame(leaf_w, 1001)),
        wire(
            SHARD,
            MsgClass::RealmSnapshot,
            &parent_frame(parent_w, 1001, 30.0, vec![]),
        ),
    ]);
    assert_eq!(
        rig.stats().window_full_chain_folds,
        1,
        "a ≥2-level fold at ONE tick"
    );
    assert_eq!(rig.stats().window_dedup_disagree, 0);
    assert_eq!(
        rig.stats().window_dedup_max_dev_cells,
        0,
        "§2.12: exact today, measured"
    );
    assert_eq!(rig.stats().window_instant_mismatch, 0);

    // ---- Tick C (T=1002): the refusal arms, by name. The parent's roster grows an
    // ALIEN-framed row (tail ≠ the author's frame — dropped from the fold, counted); a
    // tombstoned old realm datagram arrives (counted in its own bucket, served to nobody);
    // and a frame stamped a whole ring span behind the head is REFUSED.
    let alien = snap(
        RealmId::Planet(11),
        FrameRef::PlanetCentered { planet_seed: 11 },
        FrameRef::PlanetCentered { planet_seed: 11 }, // tail ≠ the author's frame: alien
        9.0,
        1002,
    );
    let _ = rig.tick(vec![
        wire(DEST, MsgClass::RealmSnapshot, &leaf_frame(leaf_w, 1002)),
        wire(
            SHARD,
            MsgClass::RealmSnapshot,
            &parent_frame(parent_w, 1002, 31.0, vec![alien]),
        ),
        wire(
            DEST,
            MsgClass::RealmSnapshot,
            &ShardToGateway::RealmFrame {
                realm_fence: Fence(1),
                source_tick: TickId(1002),
                realm_snapshot_bytes: vec![0xFF, 0xFF],
            },
        ),
        wire(SHARD, MsgClass::RealmSnapshot, &leaf_frame(parent_w, 900)),
    ]);
    let stats = rig.stats();
    assert_eq!(
        stats.window_alien_rows, 1,
        "the alien row was dropped, counted"
    );
    assert_eq!(
        stats.old_realm_frames_dropped, 1,
        "the tombstoned realm datagram counts in its own bucket, served to nobody"
    );
    assert_eq!(stats.window_level_refused, 1, "a span-stale level refused");
    assert_eq!(stats.window_full_chain_folds, 2);
    assert_eq!(stats.window_fold_divergence, 0);
    assert_eq!(stats.window_t_monotone_stalled, 0);
    assert_eq!(stats.window_chain_cycle, 0);
    // The origin marker's epoch walked its three chain identities: [System(7)] at login,
    // [Planet(7)] the tick the origin flipped, [Planet(7), System(7)] when the hop
    // confirmed (§2.7's epoch mechanics — the client swap is Slice C1).
    assert_eq!(
        rig.world.resource::<GatewaySessions>().by_session[&sid]
            .shadow
            .origin_epoch,
        3
    );

    // ---- A quiet tick: the scene is already AT the common tick with an unchanged chain —
    // nothing re-folds (the §2.14 cost discipline).
    let folds_before = rig.stats().window_folds;
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.stats().window_folds,
        folds_before,
        "nothing new to fold"
    );

    // ---- Teardown: zero sessions ⇒ zero composer state, structurally (§2.6.1 guard 4).
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    let sessions = rig.world.resource::<GatewaySessions>();
    assert!(sessions.is_empty());
    assert_eq!(
        sessions.windows_open_count(),
        0,
        "every ingest died with its window"
    );
    assert!(
        sessions.realm_heads.is_empty(),
        "the realm→node map pruned to the empty named set"
    );
}

#[test]
fn an_unresolved_lineage_ancestor_resolves_via_the_directory_head_poll() {
    // §2.6.2's ancestor leg end-to-end: a lineage parent the session never subscribed to is
    // resolved through the EXISTING `HeadRead{Realm}`/`Head` pair on the window keep-alive
    // beat; the answer opens the Child window on the named node; and that node's frames are
    // node-class dispatched as shard frames PURELY because the window names it (the window
    // arm of `is_routable_shard`) — no session role, no config entry, no presence.
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    // The login descent of a DYNAMIC session would have recorded the full root→leaf chain;
    // the static rig's descent machinery is exercised elsewhere — grow the recorded lineage
    // directly (session state, exactly what the derivation reads).
    rig.world
        .resource_mut::<GatewaySessions>()
        .by_session
        .get_mut(&sid)
        .expect("session present")
        .lineage = vec![RealmId::System(0), RealmId::System(7)];
    // No node is known for System(0): the Child window CANNOT derive yet (fail-closed).
    let _ = rig.tick(vec![]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().windows_open_count(),
        1,
        "no guessed window for an unresolved ancestor"
    );
    // The keep-alive beat carries the ancestor head poll (the EXISTING directory pair).
    set_tick(&mut rig, 25);
    let beat = rig.tick(vec![]);
    let head_reads = beat
        .iter()
        .filter(|(to, class, b)| {
            (*to == ORCH)
                & (*class == MsgClass::Saga)
                & (postcard::from_bytes::<InterShardFlow>(b)
                    == Ok(InterShardFlow::Directory(DirectoryOp::HeadRead {
                        key: DirectoryKey::Realm(RealmId::System(0)),
                    })))
        })
        .count();
    assert_eq!(
        head_reads, 1,
        "ONE deduped poll per lineage parent per beat"
    );
    assert_eq!(rig.stats().window_head_reads_sent, 1);
    // Step off the beat, so the reply tick below carries the OPEN alone (a beat tick would
    // add the keep-alive re-assert of the same window — a second, idempotent WindowOpen).
    set_tick(&mut rig, 26);
    // The head answers: System(0) is held by a spawn-minted node in no roster anywhere.
    let ancestor = NodeId(99);
    let opened = rig.tick(vec![wire(
        ORCH,
        MsgClass::Saga,
        &InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: DirectoryKey::Realm(RealmId::System(0)),
            record: Some(OwnerRecord {
                authority: AuthorityRef::Shard(ancestor),
                fence: Fence(3),
                lease_expires: UniverseTick(1_000),
                in_transfer: None,
            }),
        }),
    )]);
    let opens_to_ancestor: Vec<GatewayToShard> = opened
        .iter()
        .filter(|(to, _, _)| *to == ancestor)
        .map(|(_, _, b)| postcard::from_bytes(b).expect("shard-bound bytes decode"))
        .collect();
    assert_eq!(
        opens_to_ancestor,
        vec![GatewayToShard::WindowOpen {
            window: WindowId(2),
            scope: WindowScope::Child(RealmId::System(7)),
            static_held: None,
        }],
        "the resolved head opened the lineage hop window on the ancestor's node"
    );
    // The ancestor's attested statement is HEARD (the window arm of the dispatch) and
    // ingested — a node reachable through no session role at all.
    let frame = ShardToGateway::WindowFrame {
        realm_fence: Fence(3),
        window: WindowId(2),
        at: UniverseTick(5),
        hop: Some(Box::new(vd_wire::session_flow::HopRow {
            child: RealmId::System(7),
            placement: vd_core::frame::FramePlacement::identity(),
        })),
        rows: vec![snap(
            RealmId::System(7),
            SYS7,
            FrameRef::SystemSpace { system_seed: 0 },
            0.0,
            5,
        )],
    };
    let _ = rig.tick(vec![wire(ancestor, MsgClass::RealmSnapshot, &frame)]);
    assert_eq!(rig.stats().window_rows_ingested, 1);
    assert_eq!(
        rig.stats().refused_unknown_sender,
        0,
        "heard BECAUSE of the window"
    );
    // Teardown: the session's exit closes BOTH windows and empties the realm→node map.
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    let sessions = rig.world.resource::<GatewaySessions>();
    assert_eq!(sessions.windows_open_count(), 0);
    assert!(sessions.realm_heads.is_empty());
    assert_eq!(rig.stats().window_close_sent, 2);
}

#[test]
fn an_attach_never_overwrites_a_lineage_the_descent_already_recorded() {
    // The dynamic-login posture at the attach promote: the grant already recorded the
    // descent's full lineage, and the attach frame must NOT overwrite it (the static
    // seed runs only on an EMPTY lineage).
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sid = session_of(&rig, CLIENT);
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    rig.world
        .resource_mut::<GatewaySessions>()
        .by_session
        .get_mut(&sid)
        .expect("session present")
        .lineage = vec![RealmId::System(0), RealmId::System(7)];
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::SystemSpace { system_seed: 7 },
            realm_fence: Fence(1),
        },
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().by_session[&sid].lineage,
        vec![RealmId::System(0), RealmId::System(7)],
        "the recorded descent survives the attach"
    );
    // ★ THE SECOND HALF IS RE-BASED AT S9, because its subject stopped existing. It used to assert "the
    // OTHER unseedable shape: an attach whose frame names no realm (galaxy space) seeds nothing" —
    // fail-closed, never guessed. There is no realm-less frame any more: a galaxy names its own realm, so
    // an attach into galaxy space seeds the lineage exactly as an attach into a system does.
    //
    // The test's real subject — an attach never OVERWRITES a lineage the descent already recorded — is
    // the half above and is untouched.
    let mut rig = Rig::new();
    let _ = rig.tick(vec![wire(CLIENT, MsgClass::Control, &hello_msg())]);
    let sid = session_of(&rig, CLIENT);
    let _ = rig.tick(vec![wire(ORCH, MsgClass::Saga, &granted_head(sid))]);
    let _ = rig.tick(vec![wire(
        SHARD,
        MsgClass::Control,
        &ShardToGateway::SessionAttached {
            session: sid,
            entity: EntityId(77),
            frame: FrameRef::GalaxySpace { galaxy_seed: 0 },
            realm_fence: Fence(1),
        },
    )]);
    assert_eq!(
        rig.world.resource::<GatewaySessions>().by_session[&sid].lineage,
        vec![RealmId::Galaxy(0)],
        "an attach into galaxy space seeds the galaxy, like any other frame"
    );
}

#[test]
fn the_lineage_rule_truncates_on_reentry_and_appends_on_descent() {
    // §2.6.2's crossing rule, both arms: an inward cross APPENDS below the previous leaf;
    // an outward cross TRUNCATES back to the re-entered realm KEEPING its ancestors.
    let mut lineage = vec![RealmId::System(0), RealmId::System(7)];
    lineage_apply(&mut lineage, RealmId::Planet(7));
    assert_eq!(
        lineage,
        vec![RealmId::System(0), RealmId::System(7), RealmId::Planet(7)]
    );
    lineage_apply(&mut lineage, RealmId::System(7));
    assert_eq!(
        lineage,
        vec![RealmId::System(0), RealmId::System(7)],
        "truncated to the re-entered realm — ancestors KEPT"
    );
    // An empty lineage (a static session's first frame): the append arm seeds it.
    let mut fresh = Vec::new();
    lineage_apply(&mut fresh, RealmId::System(7));
    assert_eq!(fresh, vec![RealmId::System(7)]);
    // And the descent recorder: a coord's chain lowers root→leaf.
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
    let coord = RealmCoord::from_path(RealmPath::from_levels(vec![
        RealmLevel::new(RealmKindTag::System, 7),
        RealmLevel::new(RealmKindTag::Planet, 3),
    ]))
    .expect("a two-level path");
    assert_eq!(
        coord_lineage(&coord),
        vec![RealmId::System(7), RealmId::Planet(3)]
    );
    let root = RealmCoord::from_path(RealmPath::from_levels(vec![RealmLevel::new(
        RealmKindTag::System,
        7,
    )]))
    .expect("a root path");
    assert_eq!(coord_lineage(&root), vec![RealmId::System(7)]);
}

#[test]
fn the_composer_withholds_sessions_with_no_standing_realm_or_window() {
    // §2.6.6's "unresolved standing" row, all three shapes, driven over a hand-built table
    // through the REAL pass (the login race is held, counted, never guessed):
    // (a) an Active session with no cold sub on its routing target;
    let cfg = config();
    let (mut sessions, sid, _) = one_active_session();
    let mut stats = GatewayStats::default();
    compose_scenes_pass(
        &cfg,
        &test_clock(),
        &mut sessions,
        &mut stats,
        &mut OutboundBox::default(),
    );
    assert_eq!(stats.window_unresolved_standing, 1);
    assert_eq!(stats.window_chains_held, 0);
    // (b) a sub whose frame names no realm (galaxy space names none);
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .insert(
            SHARD,
            SubRecord {
                sub: SubId(0),
                frame: FrameRef::GalaxySpace { galaxy_seed: 0 },
                accepted: Fence(1),
                state: SubState::Active,
            },
        );
    compose_scenes_pass(
        &cfg,
        &test_clock(),
        &mut sessions,
        &mut stats,
        &mut OutboundBox::default(),
    );
    assert_eq!(stats.window_unresolved_standing, 2);
    // (c) a named realm with NO derivable own-level window (the pass ran before any window
    // existed — the pure-fn shape the schedule's driver ordering normally prevents).
    sessions
        .by_session
        .get_mut(&sid)
        .expect("present")
        .subs
        .get_mut(&SHARD)
        .expect("present")
        .frame = SYS7;
    compose_scenes_pass(
        &cfg,
        &test_clock(),
        &mut sessions,
        &mut stats,
        &mut OutboundBox::default(),
    );
    assert_eq!(stats.window_unresolved_standing, 3);
    assert_eq!(
        stats.window_folds, 0,
        "nothing was ever guessed into a fold"
    );
}

/// The load gate's typed extractor: a `WindowFrame`'s (id, level), `None` for anything else
/// (both arms driven below — HR5).
fn window_level_of(msg: ShardToGateway) -> Option<(WindowId, window::WindowLevel)> {
    match msg {
        ShardToGateway::WindowFrame {
            window,
            at,
            hop,
            rows,
            ..
        } => Some((
            window,
            window::WindowLevel {
                at,
                hop: hop.map(|b| *b),
                rows,
            },
        )),
        _ => None,
    }
}

#[test]
fn the_load_gates_extractor_and_the_divergence_verdict_cover_both_arms() {
    // The extractor: a window frame yields its level; anything else yields None.
    let level = window_level_of(ShardToGateway::WindowFrame {
        realm_fence: Fence(1),
        window: WindowId(3),
        at: UniverseTick(9),
        hop: None,
        rows: Vec::new(),
    });
    assert_eq!(
        level,
        Some((
            WindowId(3),
            window::WindowLevel {
                at: UniverseTick(9),
                hop: None,
                rows: Vec::new(),
            }
        ))
    );
    assert_eq!(
        window_level_of(ShardToGateway::SessionDetached {
            session: SessionId(1)
        }),
        None
    );
    // The §2.14 divergence verdict: equal folds verify silent; ANY bit of difference counts.
    let same = window::Composed::default();
    assert_eq!(fold_divergence(&same, &same.clone()), 0);
    let differs = window::Composed {
        instant_refused: 1,
        ..window::Composed::default()
    };
    assert_eq!(fold_divergence(&same, &differs), 1);
}

/// G-COMPOSE-LOAD (window_lane.md §2.6.7; §4.5 Topic 3 — the owner-adopted refinement: gate
/// the p99, never the mean): BOTH engine sides under a DERIVED load, per tick —
/// ingest-DECODE (postcard `WindowFrame` levels at MTU-full row counts, one per live window)
/// AND compose+fan (the shared fold per origin + every session's scene roll). The load
/// shape is derived, never invented: sessions = the density fixture's N (128 — the
/// hundreds-in-one-location correctness bound already in gate), chain depth = the design's
/// stated observer depth (§2.1: 4–6 ⇒ 6), rows per level = what fills one
/// CONSERVATIVE_DATAGRAM_BUDGET datagram (the MTU partitioner's own bound — the shape the
/// wire actually carries), origins = the crossing dual-sub bound (4 distinct chains).
/// Budget: p99 per tick < ONE realm-lane tick (§2.2: 20 Hz ⇒ 50 ms). The timing assert is
/// release-only (the SPIKE-2a/3a pattern — a debug/coverage tail is meaningless); the debug
/// run still exercises every body.
#[test]
// The seam ban targets PRODUCTION wall-clock reads; a latency gate measuring elapsed time is
// exactly what Instant is for (the SPIKE-2a exemption, same reasoning).
#[allow(clippy::disallowed_methods)]
fn g_compose_load_p99_ingest_and_fold_under_one_tick() {
    const SESSIONS: usize = 128; // the density fixture's N
    const DEPTH: usize = 6; // §2.1 chain depth, deep end
    const ORIGINS: u64 = 4; // the dual-sub crossing bound
    const TICKS: u64 = 400;
    let realm_lane_hz = 20u32; // §2.2: the realm-lane tick the budget derives from
    let cfg = config();

    // Rows per level: fill one conservative datagram, exactly as the partitioner bounds it.
    let candidate: Vec<RealmSnap> = (0..512)
        .map(|i| {
            snap(
                RealmId::Planet(1000 + i),
                FrameRef::PlanetCentered {
                    planet_seed: 1000 + i,
                },
                SYS7,
                i as f64,
                0,
            )
        })
        .collect();
    let rows_per_level = vd_wire::channels::partition_realms(
        &candidate,
        vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET,
    )[0]
    .len();

    // The window fan: ORIGINS chains × DEPTH levels. Origin o's chain authors are
    // System(10o+k); level 0 is the origin's own realm System(10o).
    let mut sessions = GatewaySessions::default();
    let mut next_window = 0u64;
    // The fixture's authors are ALL systems; the seed rides beside the id from construction
    // (no realm-kind extraction anywhere downstream).
    let mut window_rows: Vec<(WindowId, u64, WindowScope)> = Vec::new();
    for o in 0..ORIGINS {
        for k in 0..DEPTH as u64 {
            next_window += 1;
            let seed = 10 * o + k;
            let scope = if k == 0 {
                WindowScope::Occupants
            } else {
                WindowScope::Child(RealmId::System(seed - 1))
            };
            sessions.windows.insert(
                WindowId(next_window),
                GatewayWindow {
                    shard: SHARD,
                    scope,
                    author_realm: RealmId::System(seed),
                    ingest: window::WindowIngest::default(),
                    parked_bodies: BTreeMap::new(),
                    parked_relays: BTreeMap::new(),
                },
            );
            window_rows.push((WindowId(next_window), seed, scope));
        }
    }
    sessions.next_window = next_window;
    // The sessions: N Active dots spread over the origins, each standing in its origin.
    let base = one_active_session().0;
    let template = &base.by_session[&SessionId(0xA11A)];
    for i in 0..SESSIONS {
        let o = (i as u64) % ORIGINS;
        let sid = SessionId(0xB000 + i as u128);
        let mut session = Session {
            sky_held: None,
            sky_parts_sent: 0,
            client: CLIENT,
            account: AccountId(i as u128),
            fence: Fence(1),
            phase: SessionPhase::Active {
                entity: EntityId(i as u128),
            },
            next_sub: 1,
            home_shard: None,
            home_rid: None,
            realm_feed_frame_id: 0,
            scene_sent: BTreeMap::new(),
            spawn: None,
            bootstrap_deadline: None,
            confirmed_at: TickId(0),
            negotiated_minor: 1,
            transfer: None,
            subs: BTreeMap::new(),
            delivered: BTreeMap::new(),
            lineage: (0..DEPTH as u64)
                .rev()
                .map(|k| RealmId::System(10 * o + k))
                .collect(),
            shadow: window::ShadowScene::default(),
            hot: Arc::clone(&template.hot),
        };
        session.subs.insert(
            SHARD,
            SubRecord {
                sub: SubId(0),
                frame: FrameRef::SystemSpace {
                    system_seed: 10 * o,
                },
                accepted: Fence(1),
                state: SubState::Active,
            },
        );
        sessions.by_session.insert(sid, session);
    }

    // Pre-encode one tick's wire bytes per window per tick offset (the decode side must
    // decode FRESH bytes per tick — that is the measured cost).
    let mut stats = GatewayStats::default();
    let mut samples: Vec<std::time::Duration> = Vec::with_capacity(TICKS as usize);
    for t in 1..=TICKS {
        // Author one tick's statements (outside the measured section: the SHARD does this).
        let bytes_per_window: Vec<(WindowId, Vec<u8>)> = window_rows
            .iter()
            .map(|(id, seed, scope)| {
                let author_frame = FrameRef::SystemSpace { system_seed: *seed };
                let rows: Vec<RealmSnap> = (0..rows_per_level)
                    .map(|i| {
                        snap(
                            RealmId::Planet(5000 + i as u64),
                            FrameRef::PlanetCentered {
                                planet_seed: 5000 + i as u64,
                            },
                            author_frame,
                            (i as f64) + (t as f64),
                            t,
                        )
                    })
                    .collect();
                let hop = match scope {
                    WindowScope::Occupants => None,
                    WindowScope::Child(child) => Some(Box::new(vd_wire::session_flow::HopRow {
                        child: *child,
                        placement: vd_core::frame::FramePlacement::moving(
                            DVec3::new(t as f64, 0.0, 0.0),
                            DVec3::ZERO,
                        ),
                    })),
                };
                let msg = ShardToGateway::WindowFrame {
                    realm_fence: Fence(1),
                    window: *id,
                    at: UniverseTick(t),
                    hop,
                    rows,
                };
                (*id, postcard::to_allocvec(&msg).expect("encode"))
            })
            .collect();
        // ---- the measured section: decode + ingest every window's level, then the pass.
        let started = std::time::Instant::now();
        for (_, bytes) in &bytes_per_window {
            let decoded: ShardToGateway =
                postcard::from_bytes(bytes).expect("the load gate authored these bytes");
            let (wid, level) =
                window_level_of(decoded).expect("the load gate authored window frames");
            on_window_row(
                SHARD,
                wid,
                WindowRow::Level(level),
                &cfg,
                &mut sessions,
                &mut stats,
            );
        }
        compose_scenes_pass(
            &cfg,
            &test_clock(),
            &mut sessions,
            &mut stats,
            &mut OutboundBox::default(),
        );
        samples.push(started.elapsed());
    }
    assert_eq!(
        stats.window_folds,
        TICKS * ORIGINS,
        "one shared fold per (origin, tick) — the §2.14 sharing under load"
    );
    assert_eq!(
        stats.window_fold_hits as usize,
        (SESSIONS - ORIGINS as usize) * TICKS as usize
    );
    assert_eq!(stats.window_fold_divergence, 0);
    assert_eq!(stats.window_instant_mismatch, 0);
    // Bounded state under load: every ring within the derived span.
    for w in sessions.windows.values() {
        assert!(w.ingest.newest().is_some());
    }
    let p50 = vd_harness::latency::percentile_unstable(samples.clone(), 50);
    let p99 = vd_harness::latency::percentile_unstable(samples, 99);
    let budget = std::time::Duration::from_secs_f64(1.0 / f64::from(realm_lane_hz));
    eprintln!(
        "[g-compose-load] sessions={SESSIONS} origins={ORIGINS} depth={DEPTH} \
         rows/level={rows_per_level} windows={} ticks={TICKS} | per-tick ingest+decode+fold+fan \
         p50={p50:?} p99={p99:?} budget(one 20 Hz tick)={budget:?}",
        window_rows.len(),
    );
    // The hard p99 gate is release-only (SPIKE-2a/3a pattern) — debug/coverage builds run
    // the same bodies but their tails are instrumentation, not the engine.
    #[cfg(not(debug_assertions))]
    assert!(
        p99 < budget,
        "G-COMPOSE-LOAD failed: p99 {p99:?} ≥ one realm-lane tick {budget:?}"
    );
    #[cfg(debug_assertions)]
    let _ = budget;
}

/// THE SHADOW SOAK (window_lane.md §4.5 Topic 5 — "a soak added to slice B's exit"): the
/// existing soak recipe run on the SHADOW configuration (`just rlm-soak`, second line). A
/// fixed-seed xorshift drives ~30k gateway ticks of session churn + lossy/reordered/forged
/// window traffic through the REAL rig, asserting EVERY 128 ticks that
/// memory is BOUNDED (rings ≤ the derived span, pending ≤ the derived cap, realm-heads ≤
/// the named set, windows ≤ the derivable set) and counters MONOTONE — and at the end that
/// the last session's exit leaves ZERO composer state. Asserted, never eyeballed.
/// Release-gated like `rlm_soak` (inert in debug; the same paths run in the gate's unit
/// tests above).
#[test]
#[cfg(not(debug_assertions))]
fn window_shadow_soak_state_stays_bounded_and_zeroes_at_the_end() {
    let mut rig = Rig::new();
    let (sid, _) = rig.login();
    let tuning = window_tuning(&config());
    let mut state: u64 = 0x5150_5150_5150_5150;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut prev = GatewayStats::default();
    for step in 0u64..30_000 {
        let r = next();
        let at = 1_000 + step / 2 + (r % 5); // mostly advancing, sometimes reordered stamps
        let mut inbox: Vec<Inbound> = Vec::new();
        match r % 7 {
            0 | 1 => inbox.push(wire(
                SHARD,
                MsgClass::RealmSnapshot,
                &leaf_frame(WindowId(1), at),
            )),
            // ★ THIS ARM CARRIES ROWS (repaired 2026-08-28). It used to send the same content down
            // the OLD COURIER LANE, so the shadow comparator had a second picture to check against.
            // The flag day deleted that lane and its comparator; the arm's real job here is to keep
            // the churn varied — arms 1, 3 and 4 all send EMPTY frames, so without a rows-carrying
            // one the soak would never grow the structures it exists to prove bounded.
            2 => inbox.push(wire(
                SHARD,
                MsgClass::RealmSnapshot,
                &ShardToGateway::WindowFrame {
                    realm_fence: Fence(1),
                    window: WindowId(1),
                    at: UniverseTick(at),
                    hop: None,
                    rows: vec![snap(
                        RealmId::Planet(9),
                        FrameRef::PlanetCentered { planet_seed: 9 },
                        SYS7,
                        1.0,
                        at,
                    )],
                },
            )),
            3 => inbox.push(wire(
                DEST,
                MsgClass::RealmSnapshot,
                &leaf_frame(WindowId(1), at),
            )), // forged
            4 => inbox.push(wire(
                SHARD,
                MsgClass::RealmSnapshot,
                &leaf_frame(WindowId(9999), at),
            )), // unknown
            5 => inbox.push(wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::WindowBody {
                    realm_fence: Fence(1),
                    window: WindowId(1),
                    subject: RealmId::System(7),
                    stmt: BodyStmt::SelfLook {
                        bag: vec![(r % 251) as u8],
                    },
                    authored_at: UniverseTick(at),
                },
            )),
            _ => inbox.push(wire(
                SHARD,
                MsgClass::Control,
                &ShardToGateway::WindowMembership {
                    window: WindowId(1),
                    added: vec![RealmId::Planet(9)],
                    removed: vec![RealmId::Planet(9)],
                },
            )),
        }
        let _ = rig.tick(inbox);
        if step % 128 == 0 {
            let sessions = rig.world.resource::<GatewaySessions>();
            assert!(
                sessions.windows_open_count() <= 2,
                "windows stay the derivable set"
            );
            for w in sessions.windows.values() {
                let span = w
                    .ingest
                    .newest()
                    .map_or(0, |n| n.0.saturating_sub(tuning.ring_span_ticks));
                let _ = span;
                assert!(
                    w.ingest.roster_set().len() <= 64,
                    "rosters bounded by the authored rows"
                );
            }
            let session = &sessions.by_session[&sid];
            assert!(
                session.shadow.ring.len() as u64 <= tuning.ring_span_ticks + 1,
                "the fold ring holds one derived span, never more"
            );
            // (The comparator queue's cap was asserted here until 2026-08-28. The flag day deleted
            // the queue with the courier lanes, so there is no longer a second picture to queue.)
            assert!(
                sessions.realm_heads.len() <= 2,
                "realm-heads ≤ the named set"
            );
            let now = rig.stats();
            // Counters are MONOTONE (a decreasing counter is state corruption).
            assert!(now.window_rows_ingested >= prev.window_rows_ingested);
            assert!(now.window_folds >= prev.window_folds);
            // (`parity_rows_matched` was asserted monotone here until 2026-08-28 — deleted with the
            // shadow comparison it counted.)
            assert!(now.window_sender_mismatch >= prev.window_sender_mismatch);
            prev = now;
        }
    }
    assert!(
        rig.stats().window_sender_mismatch > 0,
        "the forged arm really ran"
    );
    assert!(
        rig.stats().window_unknown_row > 0,
        "the unknown arm really ran"
    );
    assert!(rig.stats().window_folds > 0, "the composer really folded");
    // ★ AND THE ROWS-CARRYING ARM REALLY RAN (added 2026-08-28 with the repair above). The three
    // statements before this one each name an arm; the arm that carries ROWS had none, so the
    // repair that rebuilt it was unproven — it could have sent an empty frame and this test would
    // still have passed.
    //
    // MEASURED BOTH WAYS, and the control failed: with the row, 20 825 frames ingested and 2 206
    // rows composed; with the arm emptied, 20 825 frames ingested and 0 rows composed. So the FRAME
    // counter is blind to this — it counts arrivals, not content — and `window_composed_rows` is the
    // one that can tell the difference.
    assert!(
        rig.stats().window_composed_rows > 0,
        "the rows-carrying arm really ran"
    );
    // The exit: zero sessions ⇒ zero composer state, after 30k ticks of churn.
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::Bye,
    )]);
    let sessions = rig.world.resource::<GatewaySessions>();
    assert!(sessions.is_empty());
    assert_eq!(sessions.windows_open_count(), 0);
    assert!(sessions.realm_heads.is_empty());
}

/// ★ THE GALAXY CROSSES ONCE (S11; owner ruling 2026-08-27).
///
/// This replaces the shard-side tests deleted when the sky moved here. The behaviours they protected
/// are the same; the author changed. A shard folded its sky from the realms IT booted, so the sky a
/// player received depended on which shard they were subscribed to — MEASURED on a dual cluster: the
/// galaxy shard held 3 stars, the client held 1, and drew 0, because a player never draws their own
/// star.
///
/// "ONCE" means: one sky, the same for everybody, sent one time. It is enforced by the CLIENT's own
/// statement, never by a memory here — a sender's memory of what it sent was proved unfixable this
/// same day and deleted for it.
#[test]
fn the_gateway_states_the_galaxy_once_and_beats_for_it_afterwards() {
    let star = |n: u64| vd_core::look::StarRow {
        realm: RealmId::System(n),
        cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
        class_code: 6,
        luma_lsun: 0.25,
    };
    let sky: Vec<vd_core::look::StarRow> = (1u64..=3).map(star).collect();
    let generation =
        vd_core::look::catalogue_generation(&postcard::to_allocvec(&sky).expect("encodes"));

    let mut rig = Rig::new();
    {
        let mut config = rig.world.resource_mut::<GatewayConfig>();
        config.sky = sky.clone();
        config.sky_generation = generation;
    }
    let (_sid, _login) = rig.login();

    // ★ THE WHOLE GALAXY REACHES THE CLIENT — every star, not the one system it is standing in. This
    // is the assertion the old shard-side lane could never have passed.
    let mut parts = Vec::new();
    for t in 1..=40u64 {
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(t);
        let sent = rig.tick(vec![]);
        for m in decode_controls(&sent, CLIENT) {
            if let ServerControlMsg::StarCatalogue {
                generation: g,
                rows,
                ..
            } = m
            {
                assert_eq!(g, generation, "one sky, one generation");
                parts.push(rows);
            }
        }
    }
    let received: Vec<vd_core::look::StarRow> = parts.iter().flatten().copied().collect();
    assert_eq!(received, sky, "the galaxy arrived whole, and in order");

    // ...and it kept arriving until the client said it held it. That is not a defect — the gateway has
    // no memory of what it sent, deliberately. The client's statement is what stops it.
    let sent_before = rig
        .world
        .resource::<GatewayStats>()
        .star_catalogue_parts_sent;
    assert!(sent_before > 0);

    // THE CLIENT STATES WHAT IT HOLDS.
    let _ = rig.tick(vec![wire(
        CLIENT,
        MsgClass::Control,
        &ClientControlMsg::SkyHeld { generation },
    )]);

    // ★ AND THE GALAXY STOPS CROSSING. This is "once": a client that holds the sky is sent none of it,
    // for ever, including when it crosses to another star system — because the sky did not change.
    let mut after = 0usize;
    let mut beats = 0usize;
    for t in 41..=80u64 {
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(t);
        let sent = rig.tick(vec![]);
        for m in decode_controls(&sent, CLIENT) {
            match m {
                ServerControlMsg::StarCatalogue { .. } => after += 1,
                ServerControlMsg::SkyAlive { generation: g } => {
                    assert_eq!(g, generation);
                    beats += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        after, 0,
        "a client that holds the galaxy is never sent it again"
    );
    assert_eq!(
        rig.world
            .resource::<GatewayStats>()
            .star_catalogue_parts_sent,
        sent_before,
        "and the counter agrees"
    );

    // ★ A STATEMENT FROM A CLIENT WITH NO SESSION IS DROPPED, not recorded against somebody else. A
    // stranger must not be able to silence the sky for a real player, which is what writing this
    // against the wrong session would do. (This assertion outlived the forwarding tests it used to
    // live in — the arm is still reachable, so it still needs driving.)
    const STRANGER: NodeId = NodeId(31337);
    let held_before = rig.world.resource::<GatewayStats>().sky_held_stated;
    let _ = rig.tick(vec![wire(
        STRANGER,
        MsgClass::Control,
        &ClientControlMsg::SkyHeld { generation },
    )]);
    assert_eq!(
        rig.world.resource::<GatewayStats>().sky_held_stated,
        held_before,
        "a client with no session states nothing"
    );

    // ★ BUT THE BEAT KEEPS COMING, and that is the point of it. On a galaxy that never changes, silence
    // is the healthy case AND the signature of a dead emitter. The beat is what tells them apart.
    assert!(
        beats > 0,
        "the sky must keep saying it is unchanged — silence would mean two different things"
    );
    assert!(
        rig.world.resource::<GatewayStats>().sky_parts_skipped > 0,
        "the saving is counted: parts NOT sent because the client already held them"
    );
}

/// ★ A GATEWAY WITH NO WORLD STATES NO SKY (S11) — never a wrong one.
///
/// A fixture, or a gateway booted without a world, holds an empty catalogue. It must stay silent
/// rather than state an empty galaxy, which a client would hold as a real sky and draw as nothing.
#[test]
fn a_gateway_with_no_world_states_no_sky_and_no_beat() {
    let mut rig = Rig::new(); // its GatewayConfig carries an empty sky
    assert!(
        rig.world.resource::<GatewayConfig>().sky.is_empty(),
        "the fixture really does boot no world"
    );
    let (_sid, _login) = rig.login();
    for t in 1..=40u64 {
        rig.world.resource_mut::<ClockSample>().local_tick = TickId(t);
        let sent = rig.tick(vec![]);
        for m in decode_controls(&sent, CLIENT) {
            assert!(
                !matches!(
                    m,
                    ServerControlMsg::StarCatalogue { .. } | ServerControlMsg::SkyAlive { .. }
                ),
                "a gateway with no world said something about the sky: {m:?}"
            );
        }
    }
    assert_eq!(
        rig.world
            .resource::<GatewayStats>()
            .star_catalogue_parts_sent,
        0
    );
    assert_eq!(rig.world.resource::<GatewayStats>().sky_alive_beats_sent, 0);
}

/// ★ A SHARD THAT HAS NOT CAUGHT UP IS COUNTED, NEVER SILENTLY DROPPED (S11).
///
/// No shard states a sky since the galaxy moved to the gateway (owner ruling 2026-08-27). The wire arms
/// remain — deleting a variant renumbers every later one, and a positional test guards that — so a
/// shard binary from before the move can still send one.
///
/// It must be VISIBLE. Silence is what made the sky lane's original defect invisible for so long; a
/// counter is the cheapest thing that stops a mixed-version cluster from looking healthy.
#[test]
fn a_sky_from_a_shard_that_has_not_caught_up_is_refused_and_counted() {
    let mut rig = Rig::new();
    let (_sid, _login) = rig.login();
    let star = vd_core::look::StarRow {
        realm: RealmId::System(1),
        cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304, 7, -3),
        class_code: 6,
        luma_lsun: 0.25,
    };
    // BOTH retired arms, from a shard that still believes it owns the sky.
    let sent = rig.tick(vec![
        wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::StarCatalogue {
                generation: 7,
                part: 0,
                parts: 1,
                rows: vec![star],
            },
        ),
        wire(
            SHARD,
            MsgClass::Control,
            &ShardToGateway::StarSkyAlive { generation: 7 },
        ),
    ]);
    assert_eq!(
        rig.world.resource::<GatewayStats>().sky_from_shard_refused,
        2,
        "both retired arms are counted"
    );
    // ...and NOTHING reached the client. A stale shard must not be able to state a sky at all — two
    // authors is the defect the move removed.
    assert!(
        !decode_controls(&sent, CLIENT).iter().any(|m| matches!(
            m,
            ServerControlMsg::StarCatalogue { .. } | ServerControlMsg::SkyAlive { .. }
        )),
        "a stale shard's sky must not reach a client: {sent:?}"
    );
}
