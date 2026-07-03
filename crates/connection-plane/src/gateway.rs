//! Gateway: the client's single-connection terminus + the P2 transfer-control CONSUMER.
//! (M0 ORIGIN — ONE subscription, ONE authority, `docs/design/connection_plane.md` §M0; the
//! transfer machinery — cut partition, route swap, commit drain, `OpenInputSlot`, release — landed
//! ON that base across P2 Slices 1c.0–1c.8.) The client holds exactly one logical connection;
//! everything server-side routes by in-frame `SessionId` + `Fence`, NEVER by source address (R2).
//!
//! Binding shapes that exist NOW because P2 cannot retrofit them:
//! - WRITE plane: `RouteSnapshot { authority, fence, cut }` behind `ArcSwap`; `route.store` is
//!   the gateway's SOLE route-mutation primitive (P2's `CommitAuthority` drives it). READ plane
//!   (1d.2): per-session `subs: ArcSwap<SubTable>`, with `publish_subs` the SOLE writer.
//! - The 20 Hz hot paths are [`route_input`] (write: one `route` `ArcSwap` load + one `AtomicU64`)
//!   and the read fan [`on_shard_frame`] (per subscribed shard: one `subs` load + a ≤4 `lookup` +
//!   a per-sub fence compare + a once-per-`SubId` byte-level re-tag) — no lock any control path
//!   takes. ([`forward_frame`]/[`frame_passes_fence`] are the SPIKE-2a ROUTE-SWAP-mechanic bench
//!   helpers: they load `route.fence` to measure the swap's wait-free read under contention, which
//!   is DISTINCT from the live read-plane per-sub `SubEntry::accepted` fence the fan checks.)
//! - Every forwarded frame's fence is compared against the per-shard `SubEntry::accepted` fence
//!   (in [`on_shard_frame`]); stale frames are dropped and counted (fence rule 5 — load-bearing the
//!   moment a session subscribes to more than one shard, e.g. across a transfer).
//! - The gateway is the SOLE ticket validator; the session mint COMMITS at the
//!   orchestrator's directory insert (the gateway only proposes entropy).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use bevy_ecs::prelude::{IntoScheduleConfigs, Res, ResMut, Resource, Schedule, World};
use vd_core::pose::FrameRef;
use vd_core::rng::SplitMix64;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox, NodeIdentity, OutboundBox};
use vd_wire::channels::{ClientControlMsg, ServerControlMsg, SubId};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey, DirectoryOp, DirectoryReply};
use vd_wire::seams::transfer_control::{PrepareResult, TransferControl, TransferControlAck};
use vd_wire::session_flow::{
    GatewayToShard, ShardToGateway, peek_input_seq, peek_is_cut_marker, peek_snapshot_frame_id,
    retag_snapshot_sub,
};
use vd_wire::version::ProtoVersion;

use crate::tickets;

/// Gateway operational parameters — ONE reviewed struct, no inline literals.
#[derive(Clone, Copy, Debug)]
pub struct TransportTuning {
    /// Hard cap on concurrent sessions (beyond it, logins are refused loudly).
    pub max_sessions: usize,
    /// Soft cap on a session's `seq > marker` cut buffer (`TransferProgress.dest_buffer`):
    /// beyond it the OLDEST buffered input is dropped (latest-wins) + counted
    /// (`GatewayStats.dest_inputs_dropped`) — never silent, never unbounded. The cap ships
    /// WITH the buffer (NOT Slice 2) because a parked saga (`ReleaseSubscribe` not yet
    /// landed; `Bye`-mid-transfer, D-23) can hold the cut open across unbounded ticks. The
    /// HARD durable backstop (the seq-range interval map) stays 1d/P3 (D-8).
    ///
    /// INVARIANT (drain-burst bound): `apply_commit` drains the WHOLE buffer to the dest in ONE
    /// tick as unreliable `SessionInput` frames, so this cap also bounds that single-tick burst.
    /// It MUST stay well below the transport's per-tick capacity (the dest `BoundedInbox`
    /// capacity and the sender `OutboundStagingCap`) or the conserved resume batch — riding the
    /// unreliable Input class — becomes the designated shed casualty under congestion. The
    /// reliable-carrier-vs-durable-watermark redesign that removes this fragility is owed 1d/P3
    /// (D-8); until then keep this at [`Self::DEFAULT_MAX_BUFFERED_INPUTS`]-scale.
    pub max_buffered_inputs: usize,
}

impl TransportTuning {
    /// Default `max_buffered_inputs`. Sized by the CUT WINDOW (freeze → directory CAS →
    /// commit = a handful of ticks of 20 Hz input, sub-second even under tick-skew), not by the
    /// transport caps: 256 inputs ≈ 12 s of input — a generous margin over any realistic cut
    /// window — while staying below the `OutboundStagingCap` (4096) and the dest `BoundedInbox`
    /// capacity. The dest inbox is `outbound_capacity × 8` with a **`.max(256)` floor**
    /// (`io_prod::mesh`): at the shipped DEV `outbound_cap = 256` that is 2048 (8× headroom), but
    /// the order-of-magnitude margin holds only while `outbound_cap ≥ 256` — below ~32 the inbox
    /// floors at 256 == this drain burst, reintroducing the foot-gun. So treat
    /// `DEFAULT_MAX_BUFFERED_INPUTS ≤ dest_inbound_floor` as a deployment INVARIANT until the
    /// reliable-carrier / durable-watermark redesign removes the unreliable-drain fragility
    /// (D-8, 1d/P3). The earlier 2048 default exactly EQUALLED the DEV dest inbox capacity — a
    /// foot-gun where one full drain plus any co-arriving frame shed a conserved (unreliable)
    /// input.
    pub const DEFAULT_MAX_BUFFERED_INPUTS: usize = 256;
}

/// Gateway configuration (composer-provided).
#[derive(Resource, Clone, Debug)]
pub struct GatewayConfig {
    pub orchestrator: NodeId,
    /// P1: the single stub shard every session lands on (the login shard).
    pub shard: NodeId,
    /// The STABLE set of routable shard `NodeId`s (node-class dispatch — FORK 5). Seeded
    /// from config (the cluster's shard roster is statically known, exactly as `shard` is);
    /// for 1d.2 it is `{shard, the transfer dest}`. It governs ONLY node-class dispatch
    /// (`is_known_shard`) — provably disjoint from client NodeIds — so a per-session
    /// subscription-refcount slip can NEVER mis-class a client datagram as a shard frame.
    /// (DISTINCT from the per-session `subscribed_shards` reverse index, which governs
    /// fan-out only.) `shard` is always a member.
    pub known_shards: BTreeSet<NodeId>,
    /// The auth service's Ed25519 verifying key (login validation).
    pub auth_verifying_key: [u8; tickets::ED25519_KEY_BYTES],
    /// Seed for session-id proposal entropy (the directory insert is the mint).
    pub session_seed: u64,
    /// The cluster's universe-tick rate (Hz), relayed to clients via
    /// `ServerControlMsg::UniverseRate` so they drive the render cursor at the
    /// server's rate. The SAME value the node feeds its `TickPacer` (VD_TICK_HZ).
    pub tick_hz: u32,
    /// D-3 lease-liveness heartbeat cadence (the gateway's LOCAL ticks): how often it re-sends
    /// `LeaseRenew` for every Active session's `Session` key, keeping the session lease alive against
    /// the orchestrator's reaper. `0` = INERT (no heartbeat — the pre-D-3 default). The gateway's local
    /// copy of `DirectoryTuning::lease_renew_interval_ticks` (same env knob), so the producer gates on
    /// `local_tick` without reaching across the directory seam.
    pub lease_renew_interval_ticks: u64,
    /// D-3 Slice 5b — how often (the gateway's LOCAL ticks) to re-read each Active session's `Session`
    /// head: the ROUND-TRIP confirmation channel that re-arms `Session.confirmed_at` (the partition
    /// detector). `0` = INERT (no recheck — the pre-D-3 default); mirrors the shard's
    /// `realm_recheck_interval`. The proactive self-fence is inert without it (no round-trip to measure).
    pub session_recheck_interval: u64,
    /// D-3 Slice 5b — the proactive self-fence grace (the gateway's LOCAL ticks). When an Active session's
    /// lease goes un-confirmed for longer than this (`local_tick - confirmed_at > grace` — a partition
    /// from the orchestrator), the gateway hard-stops acting as that session's authority BEFORE the
    /// orchestrator's reassign window opens. `0` = INERT (the pre-D-3 default). The gateway's local copy of
    /// `DirectoryTuning::self_fence_grace_ticks`; the split-brain-safe ordering is validated orch-side.
    pub self_fence_grace_ticks: u64,
    pub tuning: TransportTuning,
}

impl GatewayConfig {
    /// Is `from` a routable shard (STABLE node-class dispatch — FORK 5)? Seeded from the
    /// cluster's shard roster, provably disjoint from client NodeIds, so it can never
    /// mis-class a client datagram as a shard frame regardless of subscription churn.
    #[must_use]
    fn is_known_shard(&self, from: NodeId) -> bool {
        self.known_shards.contains(&from)
    }
}

/// The WRITE-plane route a session's input follows (authority + the transfer cut).
///
/// **Two independent atomic publishes (1d.2):** `route` (THIS, the input plane) and
/// `subs` (the read plane, [`SubTable`]) are now SEPARATE `ArcSwap`s on [`SessionHot`].
/// The forwarder reads them for orthogonal purposes — `route_input` reads ONLY `route`
/// (never `subs`), the frame-fan reads ONLY `subs` (never `route`) — so any interleaving
/// is tolerated: no consumer needs `route` and `subs` mutually consistent at an instant.
/// (Consequence, NOT a bug given no client prediction: there is a brief window where
/// `route.authority == dest` but the dest sub isn't open yet — its `SubscriptionReady` is
/// a round-trip later — absorbed by the cut buffer; the seamless claim's honest footnote.)
///
/// `fence` is now WRITE-PLANE-ONLY: the read-plane accepted fence moved to
/// [`SubEntry::accepted`] (per-shard), but `fence` stays the WRITE route's fence so the
/// sole-`store_route`-literal (and `store_commit`'s carry) never reshapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteSnapshot {
    pub authority: NodeId,
    /// The WRITE route's fence (carried by commit; the per-shard READ fence is
    /// [`SubEntry::accepted`]).
    pub fence: Fence,
    /// The input seq cut during a transfer — ALWAYS `None` in P1.
    pub cut: Option<SeqCut>,
}

/// Transfer-time input partition (P2; the shape exists so the route never reshapes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqCut {
    pub marker_seq: u64,
    pub dest: NodeId,
}

/// COLD per-session transfer state, owned solely by the single-threaded control plane
/// (`on_transfer_control` + the cut-marker observer). It lives on the cold [`Session`],
/// NEVER on the `Arc<SessionHot>` shared with the (future, threaded) 20 Hz forwarder, so
/// that forwarder can never race it. `Session.transfer` is `None` when no transfer is in
/// flight. ONE in-flight transfer per session: the orchestrator serializes a session's
/// saga on its `DirectoryKey` (`DirectoryCore::lock_transfer`), so a second concurrent
/// transfer on the same subject is refused upstream — no `by_transfer` index is needed.
#[derive(Debug)]
struct TransferProgress {
    /// Binds this progress to ONE transfer; a command/marker for a different transfer on
    /// this session is rejected (counted), never absorbed against the wrong saga.
    transfer: TransferId,
    /// Whether `RequestCut` has been issued for this transfer — the precondition for
    /// confirming a cut marker. A marker that arrives before `RequestCut` (a premature or
    /// forged emit) is dropped, so a `CutConfirmed` can never be journaled before its
    /// issuing command (the saga FSM also gates `CutConfirmed` by state; this is the
    /// gateway-side half of that guard).
    cut_requested: bool,
    /// The transfer's DEST shard, captured from `PrepareSubscribe` (the first command, which
    /// always carries it). Its consumer (1d.2): the PRECISE abort-time read-sub close — on
    /// `AbortTransfer` (which carries no dest of its own) the gateway closes EXACTLY this transfer's
    /// dest sub, mirroring `ReleaseSubscribe`'s precise `src`. NEVER an "any sub != config.shard"
    /// heuristic, which would close the player's CURRENT live sub on a chained transfer and ALL
    /// composited subs (ship/host/planet) in the N-shard end goal. `close_sub` no-ops if the dest
    /// sub is not (yet) open (abort normally runs pre-CAS, before the dest sub exists).
    dest: NodeId,
    /// The applied-steps idempotency journal: `(transfer, step_id) -> the recorded ack`,
    /// re-sent VERBATIM on an at-least-once redelivery (never re-applies the effect),
    /// reached ONLY through [`TransferProgress::recorded`] / [`TransferProgress::journal`]
    /// (the gateway's ONE dedup accessor — never touched inline). The key is the wire's
    /// `IdempotencyKey::TransferStep` — the SAME key the 1d durable redb `applied_steps`
    /// table builds (HR3 one machinery, many stores; only the STORE differs per altitude).
    /// RAM-ONLY by design — the gateway is soft-state (on resume it re-registers with the
    /// saga, never replays from RAM); the DURABLE table is the dest shard's at 1d (DEFERRED
    /// D-22). Bounded: O(phases) per live transfer, dropped whole on terminal/Bye/mint-refusal.
    applied: BTreeMap<(TransferId, u32), TransferControlAck>,
    /// The CUT BUFFER: `seq > marker_seq` client input captured while the cut is open,
    /// drained to the dest as `SessionInput` at `CommitAuthority` (integration.json #1: the
    /// gateway holds it; the dest applies it post-commit). FIFO = seq order (the hot
    /// `route_input` `fetch_max` dedups `seq <= prev` BEFORE the partition, so a buffered
    /// frame is strictly increasing). Bounded by `TransportTuning.max_buffered_inputs` —
    /// over cap the OLDEST is dropped + counted (latest-wins input). COLD/single-threaded
    /// state (never on `Arc<SessionHot>` — HR1; the 20 Hz forwarder must never race it);
    /// dropped whole on the in-flight terminal (`apply_abort` / `Bye`). RAM-only soft-state:
    /// a gateway crash mid-cut — OR a dest that drops `OpenInputSlot` (realm lease late at
    /// commit) — PERMANENTLY loses this take-drained buffer. **1c.5 has NO re-drive producer**
    /// (`OpenInputSlot` is emitted once at `apply_commit`; the saga is forward-only past
    /// `Committed`). The durable seq-range interval-map backstop + the re-drive are owed 1d/P3
    /// (D-8 + the 1c.7 cross-crate conservation gate).
    dest_buffer: VecDeque<Vec<u8>>,
}

impl TransferProgress {
    /// The recorded outcome for this transfer's `step`, or `None` if not yet applied — the
    /// ONE dedup READ (the redelivery gate + the cut-marker observer both call it). Keyed by
    /// `(self.transfer, step)` = the wire's `IdempotencyKey::TransferStep`.
    fn recorded(&self, step: u32) -> Option<TransferControlAck> {
        self.applied.get(&(self.transfer, step)).copied()
    }

    /// Record `ack` at this transfer's `step` — the ONE dedup WRITE, so the recorded value
    /// is re-sent verbatim on a redelivery. Keyed by `(self.transfer, step)`.
    fn journal(&mut self, step: u32, ack: TransferControlAck) {
        self.applied.insert((self.transfer, step), ack);
    }
}

/// The lock-free per-session hot state shared with the (future, threaded) 20 Hz
/// forwarding path. The cold session record owns an `Arc` of this.
///
/// **Per-session, NOT shared (M1):** each session has its OWN `subs` `ArcSwap`. There is
/// NO shared cross-session `SubTable` — an implementer must not "optimize" the per-session
/// `subs.load()` into a shared table, which would re-introduce a cross-session lock.
#[derive(Debug)]
pub struct SessionHot {
    /// WRITE plane (input route) — `route_input` reads ONLY this.
    pub route: ArcSwap<RouteSnapshot>,
    pub last_input_seq: AtomicU64,
    /// READ plane (per-shard accepted subscriptions) — the frame-fan reads ONLY this,
    /// wait-free. Published whole by `publish_subs` (the sole writer); HR1: it holds ONLY
    /// `{shard, sub, accepted}` — never any transfer state (that stays cold on [`Session`]).
    pub subs: ArcSwap<SubTable>,
}

/// An immutable per-session snapshot of the accepted subscriptions, published WHOLE on
/// every membership change and read WAIT-FREE by the forwarder (FORK 3: a boxed sorted
/// slice — ≤ ~4 entries — gives a branch-predictable linear/binary scan with no alloc on
/// the read, and lets a sub be added/removed without a lock, which a map of atomics cannot).
#[derive(Debug, Default)]
pub struct SubTable {
    /// Sorted by shard `NodeId` (so `lookup` can binary-search); ≤ ~4 entries.
    by_shard: Box<[SubEntry]>,
}

impl SubTable {
    /// The accepted subscription THIS session tagged for `shard`, or `None` if the session
    /// does not subscribe to it. Binary search over the ≤4 sorted entries.
    #[must_use]
    fn lookup(&self, shard: NodeId) -> Option<&SubEntry> {
        self.by_shard
            .binary_search_by_key(&shard, |e| e.shard)
            .ok()
            .map(|i| &self.by_shard[i])
    }
}

/// One accepted subscription on the READ plane: which `sub` id this session tagged a
/// `shard`'s frames with, and the fence at which it accepts them. HR1: no transfer state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubEntry {
    pub shard: NodeId,
    pub sub: SubId,
    /// The per-shard accepted frame fence (frames below it are dropped — fence rule 5).
    pub accepted: Fence,
}

/// Where one session is in its login lifecycle.
#[derive(Debug)]
enum SessionPhase {
    /// Session-key grant sent; awaiting the directory head (retried every tick —
    /// idempotent by fence).
    AwaitingDirectory,
    /// Directory granted; attach sent to the shard (retried until attached).
    AwaitingAttach,
    /// Live: input routes shard-ward, frames flow client-ward. The subscription set
    /// lives on `Session.subs` (the cold authority) + `SessionHot.subs` (the hot
    /// projection) — NOT here (1d.2a).
    Active { entity: EntityId },
    /// D-3 Slice 5b — the gateway SELF-FENCED this session (fence rule 4): it lost contact with the
    /// orchestrator (no `Session`-head round-trip within `self_fence_grace_ticks` — a partition), or a
    /// reply revealed the lease reassigned/revoked, so it HARD-STOPS acting as the session's authority
    /// BEFORE the orchestrator's reassign window opens. Input is dropped (the `Active`-only guard in
    /// `route_client_input`), frames are skipped (`on_shard_frame`), the lease is no longer renewed, and
    /// `drive_pending_sessions` leaves it be. It lingers inert until the client's connection ends (`Bye`)
    /// or a future ResumeTicket adoption (D-37/P3) re-homes it — never double-served while fenced.
    SelfFenced,
}

/// The cold authoritative record of ONE accepted subscription (the lifecycle truth; the
/// hot `SubTable` is the forwarding projection — spec's `SessionCold.subscriptions`,
/// `connection_plane.md` §2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SubRecord {
    sub: SubId,
    frame: FrameRef,
    /// The per-shard accepted frame fence (mirrored into [`SubEntry::accepted`]).
    accepted: Fence,
    state: SubState,
}

/// A subscription's lifecycle state. `Draining` = `SubscriptionClosing` sent; it stays in
/// the `SubTable` + reverse index for ONE more tick so an in-flight straggler frame is
/// still routed (drained), then the cold drain-sweep removes it (C2 — never a silent drop).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubState {
    Active,
    Draining,
}

/// One session's cold record (the hot part is the shared `Arc<SessionHot>`).
#[derive(Debug)]
struct Session {
    client: NodeId,
    account: AccountId,
    fence: Fence,
    phase: SessionPhase,
    next_sub: u32,
    /// D-3 Slice 5b — the `local_tick` of the last `Session`-head ROUND-TRIP confirmation (the reply that
    /// affirmed THIS gateway still owns the session lease). The partition detector for the proactive
    /// self-fence: set when the session goes `Active` (the attach IS a confirmation) and re-armed on every
    /// affirming recheck reply; under a partition (no reply) it FREEZES while `local_tick` climbs, and
    /// `lease_self_fence_due` fires once the gap exceeds the grace. Meaningful only while `Active`.
    confirmed_at: TickId,
    /// The negotiated proto minor for this connection (the sender-gates-variants
    /// rule): minor-1+ variants like `UniverseRate` are emitted only when `>= 1`.
    negotiated_minor: u16,
    /// COLD transfer state, `None` until a saga's `PrepareSubscribe` opens one (1c.2).
    /// Dropped whole on the in-flight terminal (`AbortTransfer`) or when the session ends.
    transfer: Option<TransferProgress>,
    /// The COLD authoritative subscription set, keyed by shard `NodeId` (1d.2a; ≤ ~4).
    /// Mutated ONLY through `open_sub`/`close_sub`/the drain-sweep, each followed by
    /// `publish_subs` (the sole `SubTable` writer — HR3). The hot `SubTable` is its
    /// forwarding projection.
    subs: BTreeMap<NodeId, SubRecord>,
    /// 1d.5a — the per-observer-sub delivery high-water: `SubId -> highest delivered frame_id`.
    /// COLD (off the wait-free `SubTable` — HR1), written in `on_shard_frame` at the push instant
    /// (only ACCEPTED, past-fence dest frames advance it), removed when the sub is swept (no leak).
    /// The (a) demote predicate's input: the standing "every current dest observer got >=1 frame"
    /// watermark. Absent ≡ watermark 0 ≡ re-blocks (an observer opening mid-demote is undelivered).
    delivered: BTreeMap<SubId, u64>,
    hot: Arc<SessionHot>,
}

/// The session table: by session id (authoritative) and by client connection
/// (in-process: the client's NodeId IS the connection).
#[derive(Resource, Debug, Default)]
pub struct GatewaySessions {
    by_session: BTreeMap<SessionId, Session>,
    by_client: BTreeMap<NodeId, SessionId>,
    /// The per-session fan-out reverse index `shard -> {sessions subscribing to it}` (FORK 5
    /// / H2), cold-maintained by `open_sub`/`close_sub`/the drain-sweep alongside the hot
    /// `SubTable`. It makes `on_shard_frame` iterate ONLY subscribers-of-`from`, not all
    /// sessions. It governs FAN-OUT only — NEVER node-class dispatch (that is the stable
    /// `config.known_shards`), so a refcount slip cannot mis-route a client. A `Draining`
    /// sub stays indexed for one tick (its straggler is drained), then removed.
    subscribed_shards: BTreeMap<NodeId, BTreeSet<SessionId>>,
}

impl GatewaySessions {
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_session.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_session.is_empty()
    }
    /// The session ids currently active (for tests/oracles).
    pub fn sessions(&self) -> impl Iterator<Item = SessionId> + '_ {
        self.by_session.keys().copied()
    }
    /// The avatar entity a session renders authoritatively (None until Active).
    /// P2's transfer coordinator keys its sagas on this.
    #[must_use]
    pub fn entity_of(&self, session: SessionId) -> Option<EntityId> {
        self.by_session.get(&session).and_then(|s| match s.phase {
            SessionPhase::Active { entity, .. } => Some(entity),
            SessionPhase::AwaitingDirectory
            | SessionPhase::AwaitingAttach
            | SessionPhase::SelfFenced => None,
        })
    }

    /// The sessions subscribing to `shard` (the H2 reverse-index read the frame-fan iterates).
    /// Empty when no session subscribes to it. Returns owned ids so the caller can mutate the
    /// outbox while iterating; the set is ≤ S and only the subscribers, never all sessions.
    fn subscribers_of(&self, shard: NodeId) -> Vec<SessionId> {
        self.subscribed_shards
            .get(&shard)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// THE one open primitive (HR3): allocate a never-reused `sub` id, insert the cold
    /// `SubRecord` (Active), push `SubscriptionOpened` BEFORE publishing the hot table (X1 —
    /// a forwarder can never route a frame for the sub ahead of its opener), publish the sole
    /// `SubTable`, and index the reverse fan-out. The transfer is the FIRST extra caller;
    /// login is the first. Returns the allocated sub id.
    fn open_sub(
        &mut self,
        session_id: SessionId,
        shard: NodeId,
        frame: FrameRef,
        accepted: Fence,
        outbox: &mut OutboundBox,
    ) -> Option<SubId> {
        let session = self.by_session.get_mut(&session_id)?;
        let sub = SubId(session.next_sub);
        session.next_sub += 1;
        session.subs.insert(
            shard,
            SubRecord {
                sub,
                frame,
                accepted,
                state: SubState::Active,
            },
        );
        // X1: SubscriptionOpened strictly precedes any data for the sub.
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::SubscriptionOpened { sub, frame },
        );
        publish_subs(&session.hot, &session.subs); // sole SubTable writer (HR3)
        self.subscribed_shards
            .entry(shard)
            .or_default()
            .insert(session_id);
        Some(sub)
    }

    /// THE one close primitive (HR3): mark the cold `SubRecord` `Draining` and send
    /// `SubscriptionClosing`. It STAYS in the `SubTable` + index for ONE more tick (C2 drain
    /// grace) so an in-flight straggler frame is still routed; the cold drain-sweep removes it
    /// next tick. A close of a shard this session does not subscribe to is a no-op. Called by
    /// `on_transfer_control` at `ReleaseSubscribe` (close the source sub) and defensively at
    /// `AbortTransfer` (close any dest sub).
    fn close_sub(&mut self, session_id: SessionId, shard: NodeId, outbox: &mut OutboundBox) {
        let Some(session) = self.by_session.get_mut(&session_id) else {
            return;
        };
        let Some(rec) = session.subs.get_mut(&shard) else {
            return;
        };
        if rec.state == SubState::Draining {
            return; // already draining (idempotent — never a second SubscriptionClosing)
        }
        rec.state = SubState::Draining;
        let sub = rec.sub;
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::SubscriptionClosing { sub },
        );
        // NOTE: left in `subs` + `subscribed_shards` until the next-tick drain-sweep so a
        // straggler frame from `shard` is drained, not dropped (the hot table still resolves
        // it). No `publish_subs` here — the hot projection keeps the Draining entry routable.
    }

    /// The cold drain-sweep (off the hot path): remove every `Draining` sub, republish the
    /// session's `SubTable`, and drop the reverse-index entry. Runs once per control tick, so a
    /// `Draining` sub lives for exactly one tick (its straggler drained), then is gone.
    fn sweep_draining(&mut self) {
        // Collect first (one mutable borrow at a time): which (session, shard) pairs drained.
        let mut drained: Vec<(SessionId, NodeId)> = Vec::new();
        for (session_id, session) in &mut self.by_session {
            // Capture (shard, sub) up front so the watermark cleanup needs no fallible re-lookup
            // (a `subs.remove` here is always Some — the shard came from this very filter).
            let draining: Vec<(NodeId, SubId)> = session
                .subs
                .iter()
                .filter(|(_, r)| r.state == SubState::Draining)
                .map(|(shard, r)| (*shard, r.sub))
                .collect();
            if draining.is_empty() {
                continue;
            }
            for (shard, sub) in &draining {
                session.subs.remove(shard);
                // 1d.5a: the delivery watermark dies WITH its sub (no leak; a swept observer
                // leaves the conjunction). Removed at the sweep — not at `close_sub` — so a
                // one-tick-Draining sub still counts its delivered frames until it is truly gone.
                session.delivered.remove(sub);
                drained.push((*session_id, *shard));
            }
            publish_subs(&session.hot, &session.subs);
        }
        // Then drop each drained session from its shard's reverse-index entry.
        for (session_id, shard) in drained {
            if let Some(set) = self.subscribed_shards.get_mut(&shard) {
                set.remove(&session_id);
                if set.is_empty() {
                    self.subscribed_shards.remove(&shard);
                }
            }
        }
    }
}

/// Session-id proposal entropy (the directory insert is the authoritative mint).
#[derive(Resource, Debug)]
struct SessionMint(SplitMix64);

/// Honesty counters: everything tolerated-but-rejected is counted, never silent.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GatewayStats {
    pub logins_rejected: u64,
    pub version_rejected: u64,
    pub sessions_refused_capacity: u64,
    pub session_mints_refused: u64,
    pub resumes_refused: u64,
    pub inputs_deduped: u64,
    pub inputs_unroutable: u64,
    pub inputs_malformed: u64,
    pub stale_frames_dropped: u64,
    pub undecodable: u64,
    /// A session in the `subscribed_shards` reverse index for `from` had NO matching
    /// `SubEntry` in its hot `SubTable` (an index/table desync — an invariant breach the
    /// `publish_subs` co-republish makes impossible by construction). Counted, never a silent
    /// `continue` (C2 honesty floor). A straggler from a just-closed source is NOT this — that
    /// is the one-tick `Draining` grace, drained not dropped.
    pub frame_sub_desync: u64,
    /// A `TransferControl` command (or cut marker, or a `SubscriptionReady` read-plane
    /// notice) for an unknown/absent session, or a phase command whose `TransferProgress`
    /// prerequisite is missing — dropped + counted, never panicked (mirrors `inputs_unroutable`).
    pub transfer_unroutable: u64,
    /// A still-parked route phase (after 1c.4: `ReleaseSubscribe`, the demote tail).
    /// Counted + warned, no ack (the saga correctly pins until its handler exists),
    /// never journaled.
    pub transfer_control_parked: u64,
    /// `CommitAuthority` arrived for a BOUND transfer (session + matching `TransferProgress`
    /// present) but `route.cut` is `None` — a FreezeSource-precedes-commit ordered-control
    /// protocol violation. Counted + `tracing::error`-logged; the saga PINS (route untouched),
    /// never a garbage-dest swap. DISTINCT from `transfer_unroutable` (the session or the
    /// in-flight transfer is absent) so the WEDGE-1 pin signal stays unblurred.
    pub commit_without_cut: u64,
    /// A `seq > marker_seq` client input held in the cut buffer for the dest (the partition
    /// fired). Drained to the dest at `CommitAuthority`. 0 outside a transfer's cut window.
    pub inputs_buffered_for_dest: u64,
    /// A buffered input dropped because the cut buffer hit `max_buffered_inputs` (oldest-
    /// first, latest-wins) — the never-silent floor; nonzero only under a parked/stalled
    /// saga holding the cut open (D-23). The durable backstop is 1d/P3 (D-8).
    pub dest_inputs_dropped: u64,
    /// D-3 Slice 5b — PROACTIVE Session self-fences: an Active session whose lease went un-confirmed past
    /// `self_fence_grace_ticks` (a detected partition from the orchestrator) was hard-stopped. Ops
    /// visibility / partition signal; `0` on the happy path and inert (`self_fence_grace_ticks == 0`).
    pub sessions_self_fenced_lapsed: u64,
    /// D-3 Slice 5b — REACTIVE Session self-fences: a `Session`-head recheck reply revealed the lease had
    /// been reassigned/revoked (no longer this gateway at its fence), so the session was hard-stopped
    /// promptly (the link-alive cure, vs the proactive timer's partition cure). `0` on the happy path.
    pub sessions_self_fenced_revoked: u64,
}

/// Install the gateway systems (composed by the harness/bin for `NodeKind::Gateway`).
pub fn register_gateway(world: &mut World, schedule: &mut Schedule, config: GatewayConfig) {
    let session_seed = config.session_seed;
    world.insert_resource(config);
    world.insert_resource(GatewaySessions::default());
    world.insert_resource(SessionMint(SplitMix64::new(session_seed)));
    world.insert_resource(GatewayStats::default());
    schedule.add_systems(
        (
            process_gateway_inbound,
            self_fence_lapsed_sessions,
            drive_pending_sessions,
            renew_and_recheck_sessions,
        )
            .chain(),
    );
}

/// D-3 Slice 5b — the PROACTIVE Session self-fence (fence rule 4, the gateway analog of the shard's
/// `self_fence_lapsed_realm`). For every Active session whose lease has gone un-confirmed past
/// `self_fence_grace_ticks` of the gateway's own `local_tick` — a partition from the orchestrator, where
/// the recheck reply never arrives — hard-stop acting as its authority (phase ⇒ `SelfFenced`) BEFORE the
/// orchestrator's reassign window opens. Runs right after `process_gateway_inbound`, so a `Session`-head
/// reply applied THIS tick (re-arming `confirmed_at`) pre-empts a spurious fence; and before
/// `renew_and_recheck_sessions`, so a just-fenced session is neither renewed nor re-checked. INERT unless
/// `self_fence_grace_ticks > 0` AND `session_recheck_interval > 0` (the shared `lease_self_fence_due`
/// guards both). The hot input path is unaffected — `route_client_input`'s Active-only guard already
/// drops a fenced session's datagrams, so no wait-free route teardown is needed.
fn self_fence_lapsed_sessions(
    config: Res<GatewayConfig>,
    clock: Res<ClockSample>,
    mut sessions: ResMut<GatewaySessions>,
    mut stats: ResMut<GatewayStats>,
) {
    for session in sessions.by_session.values_mut() {
        if vd_sim::directory::lease_self_fence_due(
            matches!(session.phase, SessionPhase::Active { .. }),
            config.self_fence_grace_ticks,
            config.session_recheck_interval,
            clock.local_tick,
            session.confirmed_at,
        ) {
            session.phase = SessionPhase::SelfFenced;
            stats.sessions_self_fenced_lapsed += 1;
        }
    }
}

/// D-3 lease-liveness producers (gateway half), each on the gateway's own LOCAL cadence:
/// (1) the HEARTBEAT — re-send `LeaseRenew` for every Active session's `Session` key, so the lease never
///     lapses while the client is connected (the orchestrator's reaper revokes a lapsed-and-confirmed-dead
///     lease); ONE mechanism with the shard's Realm/Entity heartbeat (the shared `push_renewals` shim —
///     HR3, never a match-on-shard-kind);
/// (2) the RECHECK — re-read every Active session's `Session` head, the ROUND-TRIP CONFIRMATION channel
///     whose affirming reply re-arms `Session.confirmed_at` (and whose foreign/absent reply triggers the
///     reactive self-fence); mirrors the shard's `realm_recheck`, and is what makes the proactive
///     `self_fence_lapsed_sessions` timer non-inert.
/// Both cadences are independent and INERT at interval `0` (the pre-D-3 default). Non-Active sessions have
/// no lease to renew/confirm, so both exclude them.
fn renew_and_recheck_sessions(
    config: Res<GatewayConfig>,
    clock: Res<ClockSample>,
    sessions: Res<GatewaySessions>,
    mut outbox: ResMut<OutboundBox>,
) {
    let tick = clock.local_tick.0;
    if vd_sim::directory::due_this_tick(config.lease_renew_interval_ticks, tick) {
        let renewals = sessions
            .by_session
            .iter()
            .filter(|(_, s)| matches!(s.phase, SessionPhase::Active { .. }))
            .map(|(id, s)| (DirectoryKey::Session(*id), s.fence));
        outbox.push_renewals(renewals, config.orchestrator);
    }
    if vd_sim::directory::due_this_tick(config.session_recheck_interval, tick) {
        for (id, _) in sessions
            .by_session
            .iter()
            .filter(|(_, s)| matches!(s.phase, SessionPhase::Active { .. }))
        {
            push_directory(
                &mut outbox,
                config.orchestrator,
                DirectoryOp::HeadRead {
                    key: DirectoryKey::Session(*id),
                },
            );
        }
    }
}

/// ---------------------------------------------------------------------------
/// THE HOT PATHS (pure, lock-free; SPIKE-2a benches these exact functions)
/// ---------------------------------------------------------------------------
///
/// Route one client input datagram: dedup by the leading seq varint, then read
/// the route via one `ArcSwap` load. No decode of the payload, no locks.
#[must_use]
pub fn route_input(hot: &SessionHot, input_bytes: &[u8]) -> InputRouting {
    let Ok(seq) = peek_input_seq(input_bytes) else {
        return InputRouting::Malformed;
    };
    // Latest-wins dedup: monotonic high-water mark on one atomic. `fetch_max` is a
    // SINGLE atomic RMW — it advances the mark to `max(prev, seq)` and returns the
    // PRIOR value, so the load+compare+store is indivisible. A non-atomic load-then-
    // store (FG-2) would let two concurrent forwarder threads both read the same
    // `last`, both pass, and both forward the same datagram (or store out of order).
    // With `fetch_max` exactly one observes `seq > prev` for any given seq, and the
    // mark never moves backward regardless of arrival interleaving. Still wait-free
    // (one instruction; no lock, no retry loop) — the SPIKE-2a hot-path budget holds.
    let prev = hot.last_input_seq.fetch_max(seq, Ordering::Relaxed);
    if seq <= prev {
        return InputRouting::Deduped;
    }
    let route = hot.route.load();
    // THE CUT PARTITION (1c.5): a `seq > marker_seq` frame during the cut window is held for
    // the dest (the cold caller buffers it; `apply_commit` drains it post-swap). This is ONE
    // `Option` discriminant test + (only when a cut is installed) one `u64` compare, reading
    // `.cut`/`.marker_seq` off the SAME `route` Arc already loaded above — no 2nd load, no
    // lock, no alloc. Steady state (`cut == None`) falls straight to the byte-identical
    // `Forward { authority }` the SPIKE-2a benches assert; the `Some` arm is predicted-not-
    // taken. `fetch_max` (above) still precedes, so a buffered seq is strictly increasing.
    match route.cut {
        Some(SeqCut { marker_seq, .. }) if seq > marker_seq => InputRouting::Buffer,
        _ => InputRouting::Forward {
            to: route.authority,
        },
    }
}

/// What the input hot path decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRouting {
    Forward {
        to: NodeId,
    },
    /// `seq > marker_seq` while a cut is installed: the cold caller holds it in the session's
    /// cut buffer for the dest (drained at commit). Carries no `dest` — the drain reads it
    /// off `route.cut` (keeping this hot return a pure compare).
    Buffer,
    Deduped,
    Malformed,
}

/// SPIKE-2a ROUTE-SWAP-mechanic bench helper — NOT the live read-plane fence check. It loads the
/// WRITE `route.fence` so the bench can measure the route swap's wait-free read under contention
/// (publisher `route.store` vs reader `route.load`). The LIVE read fan ([`on_shard_frame`]) checks
/// the PER-SUB [`SubEntry::accepted`] fence instead (the 1d.2 read/write split) — do NOT confuse
/// the two: this is the swap microbench, that is the production frame-acceptance.
#[must_use]
pub fn frame_passes_fence(hot: &SessionHot, frame_fence: Fence) -> bool {
    !frame_fence.is_stale_against(hot.route.load().fence)
}

/// SPIKE-2a microbench helper (route-swap mechanic): fence-compare against the WRITE route then a
/// byte-level sub re-tag — the single-session forward SHAPE under route-swap contention. NOT the
/// live forwarder: production fans per subscribed shard in [`on_shard_frame`] at the per-sub
/// [`SubEntry::accepted`] fence, sharing the re-tag once-per-`SubId` (SCALE-1).
#[must_use]
pub fn forward_frame(
    hot: &SessionHot,
    sub: SubId,
    frame_fence: Fence,
    snapshot_bytes: &[u8],
) -> Option<Vec<u8>> {
    if !frame_passes_fence(hot, frame_fence) {
        return None;
    }
    retag_snapshot_sub(snapshot_bytes, sub).ok()
}

/// ---------------------------------------------------------------------------
/// Control-plane systems (cold paths)
/// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)] // bevy system: each resource is one parameter
fn process_gateway_inbound(
    config: Res<GatewayConfig>,
    identity: Res<NodeIdentity>,
    clock: Res<ClockSample>,
    inbox: Res<InboundBox>,
    mut sessions: ResMut<GatewaySessions>,
    mut mint: ResMut<SessionMint>,
    mut stats: ResMut<GatewayStats>,
    mut outbox: ResMut<OutboundBox>,
) {
    // Cold drain-sweep FIRST (off the hot path): remove subs that were marked `Draining` in a
    // PRIOR tick. A sub closed (marked Draining) while processing this tick's inbound stays
    // routable through the rest of this tick's batch (its straggler is drained), then is swept
    // at the START of the next tick — the one-tick grace (C2 / X1).
    sessions.sweep_draining();
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            continue;
        };
        let from = *from;
        // Node-class dispatch (FORK 5): orchestrator → known-shard (STABLE roster) →
        // client-fallthrough. The orderING (orchestrator first) keeps a shard NodeId from ever
        // colliding with the orchestrator role; node roles are disjoint by construction. The
        // shard test is the STABLE `config.known_shards`, NEVER the mutable per-session
        // `subscribed_shards` — so a subscription refcount slip cannot mis-class a client.
        if from == config.orchestrator {
            match class {
                // The orchestrator→gateway Saga class carries an InterShardFlow envelope:
                // a DirectoryReply (session-grant head) OR a Saga(TransferControl) command.
                // Decode ONCE and dispatch by variant (the dispatch split that lets them
                // coexist on one class without mis-decoding into each other).
                MsgClass::Saga => match postcard::from_bytes::<InterShardFlow>(bytes) {
                    Ok(InterShardFlow::DirectoryReply(reply)) => on_directory_reply(
                        reply,
                        &config,
                        &identity,
                        &clock,
                        &mut sessions,
                        &mut stats,
                        &mut outbox,
                    ),
                    // The TransferControl consumer (the gateway counterpart to the saga
                    // runtime): the no-authority-move phases (1c.2); the route-touching
                    // phases park until 1c.3/1c.4.
                    Ok(InterShardFlow::Saga(cmd)) => {
                        on_transfer_control(cmd, &config, &mut sessions, &mut stats, &mut outbox)
                    }
                    Ok(_) | Err(_) => stats.undecodable += 1,
                },
                // Membership (clock sync) is consumed by the follower system.
                MsgClass::Membership => {}
                _ => stats.undecodable += 1,
            }
        } else if config.is_known_shard(from) {
            match class {
                MsgClass::Control => on_shard_control(
                    from,
                    bytes,
                    &config,
                    &clock,
                    &mut sessions,
                    &mut stats,
                    &mut outbox,
                ),
                MsgClass::Snapshot => {
                    on_shard_frame(from, bytes, &mut sessions, &mut stats, &mut outbox);
                }
                _ => stats.undecodable += 1,
            }
        } else {
            // A client connection.
            match class {
                MsgClass::Control => on_client_control(
                    bytes,
                    from,
                    &config,
                    &identity,
                    &mut sessions,
                    &mut mint,
                    &mut stats,
                    &mut outbox,
                ),
                MsgClass::Input => {
                    on_client_input(bytes, from, &config, &mut sessions, &mut stats, &mut outbox);
                }
                _ => stats.undecodable += 1,
            }
        }
    }
    // 1d.5a: after draining the tick's inbound, recompute the STANDING delivery watermark for each
    // in-flight transfer and emit `DeliveredToObservers` when satisfied (the (a) demote-predicate
    // input). Run once per tick, NOT per frame — no 20Hz regression (D-24).
    recompute_delivery_watermarks(&sessions, &config, &mut outbox);
}

/// 1d.5a — the STANDING per-observer delivery watermark pass. For each in-flight transfer, emit
/// `DeliveredToObservers` iff the dest observer set is NON-EMPTY (an empty set is NEVER vacuously
/// satisfied — in the window before the dest sub opens the saga must NOT be told delivery is done)
/// AND every current dest observer has received >=1 dest frame. RECOMPUTED each tick (never latched
/// here): an observer opening mid-demote inherits watermark 0 and RE-BLOCKS. Idempotent re-emit
/// while true (the saga latches `dest_delivered` and the FSM absorbs the repeat) — a STANDING
/// per-TICK (not per-frame; no 20Hz regression — D-24) reliable ack for the demote-tail duration;
/// bounded by the tail, rising-edge gating deferred. HR1: the watermark is gateway-internal; only
/// the boolean `DeliveredToObservers` crosses, on the existing SagaAck arm.
fn recompute_delivery_watermarks(
    sessions: &GatewaySessions,
    config: &GatewayConfig,
    outbox: &mut OutboundBox,
) {
    for session in sessions.by_session.values() {
        let Some(progress) = session.transfer.as_ref() else {
            continue;
        };
        if every_observer_delivered(sessions, progress.dest) {
            reply_ack(
                outbox,
                config.orchestrator,
                TransferControlAck::DeliveredToObservers {
                    transfer: progress.transfer,
                },
            );
        }
    }
}

/// Whether the dest observer set is NON-EMPTY AND every observer has delivered >=1 frame on its
/// dest sub. Iterates sessions directly (the `subs.get(&dest)` Some/None is the observer /
/// non-observer split — a domain case, never a cross-lookup desync). Bitwise `&` so neither the
/// found-nor-all arm is a short-circuit-uncoverable region (HR5). Empty observer set ⇒ false.
#[must_use]
fn every_observer_delivered(sessions: &GatewaySessions, dest: NodeId) -> bool {
    let mut found = false;
    let mut all = true;
    for session in sessions.by_session.values() {
        let Some(rec) = session.subs.get(&dest) else {
            continue;
        };
        found = true;
        all &= session.delivered.get(&rec.sub).copied().unwrap_or(0) >= 1;
    }
    found & all
}

fn push_control(outbox: &mut OutboundBox, to: NodeId, msg: &ServerControlMsg) {
    let bytes = postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly");
    // The gateway is a ROUTER — every flow it emits is re-driven/loss-tolerant, so these direct pushes are
    // Ephemeral (R-6d2b review LOW-1). If a future gateway flow is ever producer-less-reliable
    // (`FlowDurabilityClass::ProducerLessReliable`) it MUST route through `OutboundBox::push_flow_durable`
    // with `Retained` (whose debug_assert enforces it), NEVER a bare Ephemeral `.0.push`.
    outbox.0.push((
        to,
        MsgClass::Control,
        vd_sim::io::bytes(bytes),
        vd_sim::io::Durability::Ephemeral,
    ));
}

fn push_to_shard(outbox: &mut OutboundBox, to: NodeId, class: MsgClass, msg: &GatewayToShard) {
    let bytes = postcard::to_allocvec(msg).expect("closed wire enums serialize infallibly");
    outbox.0.push((
        to,
        class,
        vd_sim::io::bytes(bytes),
        vd_sim::io::Durability::Ephemeral,
    ));
}

fn push_directory(outbox: &mut OutboundBox, to: NodeId, op: DirectoryOp) {
    let bytes = postcard::to_allocvec(&InterShardFlow::Directory(op))
        .expect("closed wire enums serialize infallibly");
    outbox.0.push((
        to,
        MsgClass::Saga,
        vd_sim::io::bytes(bytes),
        vd_sim::io::Durability::Ephemeral,
    ));
}

/// The ONE `SagaAck` encode-and-push — the exact inverse of the saga runtime's
/// `outbox.push_flow(gateway, Saga, &InterShardFlow::Saga(cmd))` (DRY: reuses the shared
/// `push_flow`). Every gateway reply to the orchestrator's saga rides this.
fn reply_ack(outbox: &mut OutboundBox, orchestrator: NodeId, ack: TransferControlAck) {
    outbox.push_flow(orchestrator, MsgClass::Saga, &InterShardFlow::SagaAck(ack));
}

/// Handle one client control message. The gateway is the SOLE ticket validator;
/// shards never see a credential.
#[allow(clippy::too_many_arguments)]
fn on_client_control(
    bytes: &[u8],
    client: NodeId,
    config: &GatewayConfig,
    identity: &NodeIdentity,
    sessions: &mut GatewaySessions,
    mint: &mut SessionMint,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<ClientControlMsg>(bytes) else {
        stats.undecodable += 1;
        return;
    };
    match msg {
        ClientControlMsg::Hello { version, login } => {
            let Some(negotiated) = ProtoVersion::negotiate(ProtoVersion::CURRENT, version) else {
                stats.version_rejected += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "incompatible protocol major version".to_owned(),
                    },
                );
                return;
            };
            if tickets::validate_login(&config.auth_verifying_key, &login).is_err() {
                stats.logins_rejected += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "login ticket rejected".to_owned(),
                    },
                );
                return;
            }
            if sessions.by_session.len() >= config.tuning.max_sessions {
                stats.sessions_refused_capacity += 1;
                push_control(
                    outbox,
                    client,
                    &ServerControlMsg::Close {
                        reason: "gateway at session capacity".to_owned(),
                    },
                );
                return;
            }
            if sessions.by_client.contains_key(&client) {
                // A duplicate Hello on a live connection: idempotent no-op (the
                // pending/active session keeps progressing).
                return;
            }
            // Propose a session id; the DIRECTORY INSERT is the authoritative mint.
            let proposed =
                SessionId((u128::from(mint.0.next_u64()) << 64) | u128::from(mint.0.next_u64()));
            let fence = Fence::GENESIS.next();
            sessions.by_session.insert(
                proposed,
                Session {
                    client,
                    account: login.account,
                    fence,
                    phase: SessionPhase::AwaitingDirectory,
                    next_sub: 0,
                    // Armed at the Active transition (the attach); irrelevant while still logging in.
                    confirmed_at: TickId(0),
                    negotiated_minor: negotiated.minor,
                    transfer: None,
                    subs: BTreeMap::new(),
                    delivered: BTreeMap::new(),
                    hot: Arc::new(SessionHot {
                        route: ArcSwap::from_pointee(RouteSnapshot {
                            authority: config.shard,
                            fence: Fence::GENESIS,
                            cut: None,
                        }),
                        last_input_seq: AtomicU64::new(0),
                        subs: ArcSwap::from_pointee(SubTable::default()),
                    }),
                },
            );
            sessions.by_client.insert(client, proposed);
            push_directory(
                outbox,
                config.orchestrator,
                DirectoryOp::LeaseGrant {
                    key: DirectoryKey::Session(proposed),
                    owner: AuthorityRef::Gateway(identity.node_id),
                    fence,
                },
            );
        }
        ClientControlMsg::Resume { .. } => {
            // Resume/adoption lands in P3; refusing is honest, not silent.
            stats.resumes_refused += 1;
            push_control(
                outbox,
                client,
                &ServerControlMsg::Close {
                    reason: "resume is not available yet".to_owned(),
                },
            );
        }
        ClientControlMsg::Bye => {
            let Some(session_id) = sessions.by_client.remove(&client) else {
                return;
            };
            let session = sessions
                .by_session
                .remove(&session_id)
                .expect("session maps are kept in sync");
            // WEDGE-1 (pinned to Slice 2 — DEFERRED D-23): a Bye mid-transfer drops the
            // session + its journal; subsequent saga commands for it then count as
            // `transfer_unroutable` with NO producer to unstick the pinned saga. The real
            // backstop is the Slice-2 saga timeout/abort producer; pin loud here so the
            // dropped in-flight transfer is never silent.
            if let Some(tp) = session.transfer.as_ref() {
                tracing::warn!(
                    session = %session_id,
                    transfer = tp.transfer.0,
                    "client Bye dropped a session with an in-flight transfer — the saga will \
                     pin until the Slice-2 timeout producer lands (D-23)"
                );
            }
            // ⚠️ SCALE (DEFERRED D-34): detaches the single `config.shard`, NOT the session's
            // current authority. After a transfer (player homed on the dest), this leaks the
            // dest's `SessionTable` entry. Correct only for single-login-shard P2; the proper fix
            // is a per-session `home_shard`/`authority` field (set by the orchestrator Spawn
            // Resolver, updated on commit) that this detach + the login landing both route off.
            // NOT a `session.subs.keys()` scan — `subs` is empty at login (would regress login→Bye).
            push_to_shard(
                outbox,
                config.shard,
                MsgClass::Control,
                &GatewayToShard::DetachSession {
                    session: session_id,
                    fence: session.fence,
                },
            );
            push_directory(
                outbox,
                config.orchestrator,
                DirectoryOp::LeaseRevoke {
                    key: DirectoryKey::Session(session_id),
                    fence: session.fence,
                },
            );
        }
        // No transfers in P1: a cut confirmation has nothing to bind to.
        // Pongs are liveness echoes; the P1 gateway sends no pings.
        ClientControlMsg::CutEmitted { .. } | ClientControlMsg::Pong { .. } => {}
    }
}

/// Route one client input datagram (HOT decision + bookkeeping), then COLD-observe the
/// transfer cut marker — strictly OFF the SPIKE-2a hot path (`route_input` is unchanged).
fn on_client_input(
    bytes: &[u8],
    client: NodeId,
    config: &GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Some(session_id) = sessions.by_client.get(&client).copied() else {
        stats.inputs_unroutable += 1;
        return;
    };
    // Guarded lookup, never `[]`: the two session maps are kept in sync by construction,
    // but a desync must DROP-and-count on this 20Hz path, never panic the gateway
    // (R2 lineage — a routing-map slip is a counted unroutable, not a crash).
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        stats.inputs_unroutable += 1;
        return;
    };
    if !matches!(session.phase, SessionPhase::Active { .. }) {
        stats.inputs_unroutable += 1;
        return;
    }
    // HOT: the benched route decision (ArcSwap load + one atomic). Unchanged.
    match route_input(&session.hot, bytes) {
        InputRouting::Forward { to } => {
            push_to_shard(
                outbox,
                to,
                MsgClass::Input,
                &GatewayToShard::SessionInput {
                    session: session_id,
                    fence: session.fence,
                    input_bytes: bytes.to_vec(),
                },
            );
        }
        InputRouting::Buffer => {
            // COLD: hold the seq>marker frame in the session's cut buffer for the dest
            // (drained at CommitAuthority). The hot `route_input` only DECIDED Buffer; the
            // buffer lives on the cold `TransferProgress`, never on `SessionHot` (HR1).
            // `route_input` returns Buffer ONLY when `route.cut` is Some, which `apply_freeze`
            // installs together with the in-flight `transfer` — so `transfer` is Some here by
            // construction (`expect`, the unreachable-arm shape). Bounded: over cap, drop the
            // OLDEST (latest-wins input) + count.
            let tp = session
                .transfer
                .as_mut()
                .expect("an installed cut implies an in-flight transfer (apply_freeze sets both)");
            if tp.dest_buffer.len() >= config.tuning.max_buffered_inputs {
                tp.dest_buffer.pop_front();
                stats.dest_inputs_dropped += 1;
            }
            tp.dest_buffer.push_back(bytes.to_vec());
            stats.inputs_buffered_for_dest += 1;
        }
        InputRouting::Deduped => stats.inputs_deduped += 1,
        InputRouting::Malformed => stats.inputs_malformed += 1,
    }
    // COLD: observe the in-band cut marker ONLY while a transfer is in flight (1c sources
    // it SCRIPTED on the input flow; the client emit is 1e — DEFERRED D-5). `route_input`
    // never decodes `is_cut_marker`; this is a separate cold decode, off the hot path.
    if let Some(tp) = session.transfer.as_mut() {
        on_cut_marker(tp, bytes, config.orchestrator, outbox);
    }
}

/// THE gateway counterpart to the saga runtime: consume one `TransferControl` command.
/// EVERY phase is now LIVE: Prepare/RequestCut/FreezeSource (the cut install, 1c.3)/
/// CommitAuthority (the route swap, 1c.4)/ThawSource/AbortTransfer/ReleaseSubscribe (the demote
/// tail's SUCCESS teardown, 1c.8 — acks `Released` so the saga reaches `Done`). The applied-steps
/// journal gates every command BEFORE any effect (consult-before-effect) and records AFTER
/// (record-after-effect) so an at-least-once redelivery re-sends the recorded ack verbatim and
/// never re-applies.
fn on_transfer_control(
    cmd: TransferControl,
    config: &GatewayConfig,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let transfer = cmd.transfer();
    let step = cmd.step_id();
    let session_id = cmd.session(); // the get_mut key; reused for the dest-bound OpenInputSlot/SessionInput
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        stats.transfer_unroutable += 1; // unknown/absent session: counted, never panic
        return;
    };
    // REDELIVERY GATE (consult-before-effect): a recorded step for THIS transfer re-sends
    // the recorded ack verbatim, no effect — via the ONE dedup accessor `TransferProgress::
    // recorded` (the cut-marker observer reads the SAME way; DRY-1). A let-chain (each link's
    // true/false arm is separately exercised: no-transfer / wrong-transfer / unrecorded-step
    // / recorded-step).
    //
    // RequestCut is EXCLUDED (F1): it is NOT self-acking — its `CutConfirmed` is journaled
    // at the SAME step-1 slot by the cut-marker observer, NOT by RequestCut itself. So a
    // redelivered RequestCut must re-run `apply_request_cut` (an idempotent client re-push
    // the client de-dups), never consult the marker's slot and wrongly answer the command
    // with a `CutConfirmed`. The marker observer owns the step-1 journal exclusively.
    let is_request_cut = matches!(cmd, TransferControl::RequestCut { .. });
    if !is_request_cut
        && let Some(tp) = session.transfer.as_ref()
        && tp.transfer == transfer
        && let Some(prior) = tp.recorded(step)
    {
        reply_ack(outbox, config.orchestrator, prior);
        return;
    }
    // The READ-plane subs to close AFTER the ack/journal (the `&mut Session` borrow must end
    // before `GatewaySessions::close_sub` — the sole sub-close primitive — can run).
    // ReleaseSubscribe closes the SOURCE sub (`src` from the command); AbortTransfer closes
    // EXACTLY this transfer's DEST sub (`tp.dest`, captured at PrepareSubscribe) — never an "any
    // sub != config.shard" heuristic (which would close the player's CURRENT live sub on a chained
    // transfer and ALL composited subs in the N-shard end goal). Inert in the 1d.2 happy path
    // (abort runs pre-CAS, before the dest sub exists; `close_sub` no-ops then) but PRECISE for the
    // commit/flip-window-open ordering and the N-shard future.
    let mut subs_to_close: Vec<NodeId> = Vec::new();
    // Compute the ack (or None for deferred/parked phases), then record-then-send below.
    let ack: Option<TransferControlAck> = match cmd {
        TransferControl::PrepareSubscribe { dest, .. } => {
            apply_prepare(session, transfer, dest, stats)
        }
        TransferControl::RequestCut { .. } => {
            apply_request_cut(session, outbox, transfer, stats);
            None // the ack (CutConfirmed) is deferred to the cut-marker observer
        }
        TransferControl::FreezeSource {
            marker_seq, dest, ..
        } => apply_freeze(session, transfer, marker_seq, dest, stats),
        TransferControl::ThawSource { .. } => apply_thaw(session, transfer, stats),
        TransferControl::AbortTransfer { .. } => {
            // Close EXACTLY this transfer's dest sub (`tp.dest`), mirroring ReleaseSubscribe's
            // precise `src`; a foreign/absent abort matches no `tp` and closes nothing. `close_sub`
            // no-ops if the dest sub is not (yet) open — the normal pre-CAS abort case.
            if let Some(tp) = session
                .transfer
                .as_ref()
                .filter(|tp| tp.transfer == transfer)
            {
                subs_to_close.push(tp.dest);
            }
            apply_abort(session, transfer)
        }
        TransferControl::CommitAuthority {
            new_fence, subject, ..
        } => apply_commit(
            session, transfer, new_fence, subject, session_id, stats, outbox,
        ),
        TransferControl::ReleaseSubscribe { src, .. } => {
            // Close the SOURCE sub ONLY for the matching in-flight transfer (mirroring
            // `apply_release`'s prune); a foreign/absent release closes nothing.
            if session
                .transfer
                .as_ref()
                .is_some_and(|tp| tp.transfer == transfer)
            {
                subs_to_close.push(src);
            }
            apply_release(session, transfer)
        }
    };
    // RECORD-then-SEND for the LIVE acking phases. The journal write is GUARDED to the
    // IN-FLIGHT transfer (`tp.transfer == transfer`): an idempotent re-ack of a phase for a
    // NON-in-flight transfer (e.g. a stray Thaw/Abort while a different transfer is live)
    // must never pollute the live transfer's journal — bounding it to its own steps. (A
    // terminal AbortTransfer prunes `session.transfer` inside `apply_abort`, so `as_mut()`
    // is already `None` there — no record, the abort being idempotently re-ackable anyway.)
    if let Some(ack) = ack {
        if let Some(tp) = session.transfer.as_mut()
            && tp.transfer == transfer
        {
            tp.journal(step, ack);
        }
        reply_ack(outbox, config.orchestrator, ack);
    }
    // The `&mut Session` borrow has ended: close the collected subs through the sole close
    // primitive (Draining grace; the next-tick sweep removes them). `close_sub` is idempotent
    // and a no-op for a shard this session does not subscribe to.
    for shard in subs_to_close {
        sessions.close_sub(session_id, shard, outbox);
    }
}

/// `PrepareSubscribe` (step 0): open the per-session transfer progress + reply `Prepared`.
/// 1c STUB readiness — there is no dest-subscription/ghost machinery yet (one stub shard),
/// so the verdict is `Ready` (the typed `Rejected` path stays WIRED for later bands, not
/// dead live code). DEFENSIVE replace: overwrite any stale progress.
///
/// REQUIRES the session be ACTIVE (WB-1): a transfer can only run for an attached avatar.
/// This single guard ENFORCES "attach strictly precedes transfer" LOCALLY (rather than
/// borrowing it from saga ordering) and is what makes every downstream route writer safe:
/// `session.transfer` is set Some ONLY here, ONLY when Active; Active is monotonic-until-
/// removal; and the attach path early-returns once Active (never re-storing the route). So a
/// cut installed by `apply_freeze` — or a route swapped by 1c.4 `CommitAuthority` — can never
/// be clobbered by a late `SessionAttached`. A not-Active prepare is counted + un-acked (pins
/// the saga, per WEDGE-1), never a route-touching transfer on a half-attached session.
fn apply_prepare(
    session: &mut Session,
    transfer: TransferId,
    dest: NodeId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    if !matches!(session.phase, SessionPhase::Active { .. }) {
        stats.transfer_unroutable += 1;
        return None;
    }
    // RACE-1 (pinned to Slice 2 — DEFERRED D-23): replacing a LIVE, different transfer's
    // progress discards its journal. Impossible today — the orchestrator serializes one
    // saga per session-subject (`DirectoryCore::lock_transfer`), and a redelivered SAME
    // transfer is caught by the redelivery gate before reaching here — so any existing
    // progress here is necessarily a stale DIFFERENT transfer. Becomes reachable only once
    // Slice-2 closes the abort-lock leak (D-1); pinned loud so it is never silently relied on.
    if let Some(displaced) = session.transfer.as_ref() {
        tracing::warn!(
            displaced = displaced.transfer.0,
            opening = transfer.0,
            "PrepareSubscribe replaced a stale in-flight transfer's progress — revisit at \
             Slice 2 (the one-saga-per-subject lock makes this benign today; D-23)"
        );
    }
    session.transfer = Some(TransferProgress {
        transfer,
        cut_requested: false,
        dest, // captured here for the precise abort-time dest-sub close (1d.2)
        applied: BTreeMap::new(),
        dest_buffer: VecDeque::new(),
    });
    // The dest input slot is opened at CommitAuthority (apply_commit) — the gateway BUFFERS
    // seq>marker locally during the cut, so the dest needs nothing until the commit drain.
    // (An early prepare-time open is a P3 gateway-adoption resilience concern, not 1c.5.)
    tracing::debug!(
        transfer = transfer.0,
        "PrepareSubscribe readiness is a 1c stub (Ready)"
    );
    Some(TransferControlAck::Prepared {
        transfer,
        result: PrepareResult::Ready,
    })
}

/// `RequestCut` (step 1): mark the cut as requested + ask the client to emit the in-band
/// cut marker. The ack (`CutConfirmed`) is DEFERRED to the marker observer, so this returns
/// nothing. Re-pushing `RequestCut` is harmless (reliable+ordered CONTROL; the client
/// de-dups), so it is not journaled at command time. Requires the matching progress
/// (`PrepareSubscribe` precedes it); setting `cut_requested` is what later authorizes the
/// marker observer to confirm a cut (F1: no `CutConfirmed` before its `RequestCut`).
fn apply_request_cut(
    session: &mut Session,
    outbox: &mut OutboundBox,
    transfer: TransferId,
    stats: &mut GatewayStats,
) {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // RequestCut without a matching Prepared: drop+count
        return;
    }
    session
        .transfer
        .as_mut()
        .expect("bound implies the in-flight transfer is present")
        .cut_requested = true;
    push_control(
        outbox,
        session.client,
        &ServerControlMsg::RequestCut { transfer },
    );
}

/// `FreezeSource` (step 2): INSTALL the live cut — `seq <= marker_seq` stays bound to the
/// source, `seq > marker_seq` will buffer toward `dest`. The SOLE `cut: Some(..)` writer; its
/// mirror image is `apply_thaw`'s `cut: None`, both via `store_cut`. The `seq > marker`
/// partition READ lands in 1c.5; 1c.3 installs the latent field only (this hot path reads
/// only `.authority`, so the install is inert to routing until then).
///
/// UNBOUND DIVERGES from `apply_thaw`: freeze is a FORWARD phase (its `SourceFrozen` drives
/// the directory CAS), so an unbound freeze must NOT fabricate an ack — acking a
/// never-installed cut would advance the saga to a CAS / route-swap against a session this
/// gateway no longer holds (the Bye-mid-transfer wedge, WEDGE-1). Returning `None` correctly
/// PINS the saga until the Slice-2 abort producer (D-23). Bound-check FIRST, mirroring
/// `apply_request_cut` (the forward sibling), NOT `apply_thaw` (a compensator that no-ops
/// truthfully when unbound). Self-acking, so a redelivery is re-served by the gate, never
/// re-entered here (the route is never re-touched).
fn apply_freeze(
    session: &mut Session,
    transfer: TransferId,
    marker_seq: u64,
    dest: NodeId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // no source to freeze: pin the saga (no ack)
        return None;
    }
    store_cut(&session.hot, Some(SeqCut { marker_seq, dest }));
    // drained_seq = marker_seq: the 1c SINGLE-STUB-SHARD value. By construction of the cut
    // (seq <= marker -> source), the marker IS the last source-bound seq, so "the source
    // applied input through exactly this seq" == marker_seq. NOT last_input_seq (the OBSERVED
    // high-water, which a client racing dest-bound input past the marker pushes ABOVE
    // marker_seq, breaking input-conservation). A real source-applied drain watermark replaces
    // this at 1d (the shard drain oracle); until then nothing READS drained_seq (the FSM
    // carries it unread into CommittingCas), so the stub is observability-only.
    Some(TransferControlAck::SourceFrozen {
        transfer,
        drained_seq: marker_seq,
    })
}

/// THE sole route-mutation primitive: the one `route.store` for every transfer/attach
/// write. `store_cut` (carry authority+fence, swap cut), `store_commit` (move authority→dest,
/// carry fence, clear cut), and the attach path (fresh fence) all funnel here — so a future
/// `RouteSnapshot` field can never silently drop on any of them: the exhaustive no-`..rest`
/// struct literal lives in EXACTLY one place, and a 4th field is a single compile-fix every
/// caller inherits. (Login route BIRTH stays a direct `ArcSwap::from_pointee` — not a store.)
fn store_route(hot: &SessionHot, authority: NodeId, fence: Fence, cut: Option<SeqCut>) {
    hot.route.store(Arc::new(RouteSnapshot {
        authority,
        fence,
        cut,
    }));
}

/// THE sole `SubTable` (READ plane) writer — the read-plane analog of `store_route`: project
/// the cold `subs` map into the immutable hot `SubTable` and publish it whole via one
/// `ArcSwap::store`. The exhaustive no-`..rest` `SubEntry` literal lives in EXACTLY one place
/// (HR3), so a future `SubEntry` field is a single compile-fix every caller inherits. Both
/// `Active` AND `Draining` records are projected (a `Draining` sub stays routable for its
/// one-tick drain grace); the cold map is already sorted by shard `NodeId` (`BTreeMap`), so
/// the boxed slice is sorted for `SubTable::lookup`'s binary search by construction.
fn publish_subs(hot: &SessionHot, subs: &BTreeMap<NodeId, SubRecord>) {
    let by_shard: Box<[SubEntry]> = subs
        .iter()
        .map(|(shard, rec)| SubEntry {
            shard: *shard,
            sub: rec.sub,
            accepted: rec.accepted,
        })
        .collect();
    hot.subs.store(Arc::new(SubTable { by_shard }));
}

/// Carry the route's `authority` + `fence` forward and swap ONLY the `cut` — the cut
/// install/clear policy (`apply_freeze` `Some`, `apply_thaw`/`apply_abort` `None`). Its
/// mirror image is `store_commit`; both go through the lone `store_route` literal.
fn store_cut(hot: &SessionHot, cut: Option<SeqCut>) {
    let current = hot.route.load();
    store_route(hot, current.authority, current.fence, cut); // CARRY authority + fence
}

/// `CommitAuthority`: MOVE authority → `dest`, CARRY the realm fence UNCHANGED, CLEAR the
/// cut — ONE atomic `route.store` (the SPIKE-2a one-publish guarantee). The fence is CARRIED,
/// NOT advanced to the per-Entity CAS `new_fence` (R-FENCE / DEFERRED D-25 — see `apply_commit`).
fn store_commit(hot: &SessionHot, dest: NodeId) {
    let current = hot.route.load();
    store_route(hot, dest, current.fence, None); // MOVE authority→dest, CARRY fence, CLEAR cut
}

/// `CommitAuthority` (step 3): THE route swap. MOVE authority → `dest` (the `SeqCut.dest` the
/// preceding `FreezeSource` installed), CLEAR the cut, in ONE atomic `store_commit`.
///
/// `dest` HAS NO WIRE CARRIER by construction — `CommitAuthority { transfer, session, new_fence }`
/// carries no dest — so the installed `route.cut.dest` is authoritative: the gateway-local
/// crystallization of the saga's durable `SagaCtx.dest` (the SAME value `FreezeSource.dest`
/// rode). FreezeSource strictly precedes commit on reliable+ordered CONTROL (the saga emits
/// CommitAuthority only after `CommittingCas`), so the cut is present at first delivery.
///
/// Bound-check FIRST (mirror `apply_freeze`): a post-CAS forward phase must NOT fabricate
/// `Committed` for a session this gateway no longer holds (WEDGE-1) — pin instead.
///
/// **R-FENCE (DEFERRED D-25): `new_fence` is CARRIED, NOT installed as `route.fence`.** Frames
/// stamp the REALM fence (each sub accepts at its per-shard `SubEntry::accepted` realm fence,
/// checked in `on_shard_frame`), whereas `new_fence` is the per-ENTITY CAS fence
/// (`DirectoryKey::Entity`). Installing the Entity fence here would make the dest's OWN
/// realm-stamped frames stale
/// (`realm_fence.is_stale_against(new_fence) == true`) — a black screen. Fence rule 5 (fence
/// out a demoted REMOTE owner) activates by REALM-fence movement at 1d/mesh when source/dest
/// are distinct realm leases; intra-shard (one realm lease) it correctly does NOT fire.
/// `new_fence` stays threaded through wire+saga (the CAS linearization point); the gateway
/// consumes it without installing it until the dest re-stamps its realm lease (1d/mesh).
fn apply_commit(
    session: &mut Session,
    transfer: TransferId,
    new_fence: Fence,
    subject: DirectoryKey,
    session_id: SessionId,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if !bound {
        stats.transfer_unroutable += 1; // session/transfer absent: pin (WEDGE-1)
        return None;
    }
    // Capture BOTH `dest` and `marker_seq` from the installed cut BEFORE `store_commit`
    // clears it: `marker_seq` seeds the dest's resume watermark; `dest` is the new authority.
    let Some(SeqCut { dest, marker_seq }) = session.hot.route.load().cut else {
        // BOUND but no cut: FreezeSource must precede commit (ordered CONTROL). Loud,
        // counted, route UNTOUCHED — never a swap to a garbage dest; the saga pins.
        stats.commit_without_cut += 1;
        tracing::error!(
            transfer = transfer.0,
            "CommitAuthority with no installed cut — FreezeSource must precede commit; \
             route untouched, saga pins"
        );
        return None;
    };
    let _ = new_fence; // R-FENCE: carried-not-installed in 1c.4 (D-25); 1d/mesh installs on dest re-stamp
    // (a) AUTHORITATIVE OpenInputSlot: the dest seeds last_applied_seq = marker_seq, so the
    // drained resume batch (marker+1..) applies in order and a `seq <= marker` replay is
    // rejected — closing the UnknownSession input-loss hole for the genuinely-distinct dest.
    // 1c.8: it ALSO carries the transfer `subject` (forwarded verbatim from CommitAuthority) so
    // the dest ADOPTS the transferred avatar (its Entity becomes the dot's id). `new_fence` is
    // NOT carried — the dest learns it from its own directory HeadRead (pull-through).
    push_to_shard(
        outbox,
        dest,
        MsgClass::Control,
        &GatewayToShard::OpenInputSlot {
            session: session_id,
            fence: session.fence,
            account: session.account,
            resume_from_seq: marker_seq,
            subject,
        },
    );
    // (b) THE swap: authority:=dest, fence carried, cut:=None (ONE atomic publish).
    store_commit(&session.hot, dest);
    // (c) DRAIN the cut buffer to the now-authoritative dest as ordinary SessionInput, in
    // push order (== seq order: route_input's fetch_max dropped seq<=prev BEFORE buffering,
    // so the buffer is strictly increasing). `std::mem::take` empties it — the STRUCTURAL
    // drain-once guarantee: a redelivered CommitAuthority re-serves Committed via the gate
    // (above) AND finds the buffer empty, so it never re-drains.
    // The bound-check above guarantees `session.transfer` is Some (the in-flight transfer);
    // `expect` is the unreachable-arm shape.
    // Push order == seq order: `route_input`'s `fetch_max` returns `Deduped` for any
    // `seq <= prev` BEFORE the partition, so only a strictly-increasing subsequence is ever
    // buffered. The dest's own `last_applied_seq` dedup is the backstop regardless of order.
    let buffered = std::mem::take(
        &mut session
            .transfer
            .as_mut()
            .expect("bound-check above guarantees the in-flight transfer is present")
            .dest_buffer,
    );
    for input_bytes in buffered {
        push_to_shard(
            outbox,
            dest,
            MsgClass::Input,
            &GatewayToShard::SessionInput {
                session: session_id,
                fence: session.fence, // session-grant fence (NOT new_fence; R-FENCE/D-25)
                input_bytes,
            },
        );
    }
    Some(TransferControlAck::Committed { transfer })
}

/// `ThawSource` (step 4): the `FreezeSource` compensator — input resumes to the source.
/// Clears the cut (via `store_cut`) ONLY for the matching in-flight transfer
/// (ROB-THAW-UNBOUND-STORE): a thaw naming a different/absent transfer must NOT touch the
/// route, else it would clobber the in-flight transfer's LIVE cut (live since 1c.3). A thaw
/// against a never-frozen source is a correct no-op (the source never stopped), so it ALWAYS
/// acks `SourceThawed` (even unbound — counted — so the saga's thaw compensator can always
/// complete), but only the BOUND case stores the route. The live mirror image of `apply_freeze`.
fn apply_thaw(
    session: &mut Session,
    transfer: TransferId,
    stats: &mut GatewayStats,
) -> Option<TransferControlAck> {
    let bound = session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer);
    if bound {
        store_cut(&session.hot, None);
    } else {
        stats.transfer_unroutable += 1;
    }
    Some(TransferControlAck::SourceThawed { transfer })
}

/// `AbortTransfer` (step 5, terminal): `PrepareSubscribe`'s compensator — tear down the dest
/// ghost. In 1c.2 `PrepareSubscribe` built no ghost (`Ready` stub), so teardown is inert on
/// session state and the route is untouched (the source stayed authoritative). NOT a
/// disconnect: the `Session`/`phase` are untouched. Prunes ONLY the matching in-flight
/// transfer (CP-1: an abort for transfer B must never clobber a live transfer A). An abort
/// is an IDEMPOTENT terminal: a redelivered abort, or an abort for a transfer this gateway
/// never held, is a benign no-op re-ack — NOT a routing failure (F2: it is uncounted, so
/// `transfer_unroutable` stays about genuine failures). Always acks `Aborted`.
fn apply_abort(session: &mut Session, transfer: TransferId) -> Option<TransferControlAck> {
    if session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer)
    {
        session.transfer = None; // prune ONLY the matching in-flight transfer
        // ROB-1c3: clear any installed cut LOCALLY. The saga emits ThawSource before
        // AbortTransfer over reliable CONTROL, so today the cut is usually already clear —
        // but abort must be LOCALLY fail-safe, not borrow that ordering: once the cut READ
        // goes live (1c.5) a survived cut becomes a live mis-route, and Slice-2/warp abort
        // producers need not preserve Thaw-before-Abort. `store_cut(None)` is idempotent
        // (a no-op when the cut is already clear), so it is safe on every abort path.
        store_cut(&session.hot, None);
    }
    tracing::debug!(
        transfer = transfer.0,
        "AbortTransfer tears down the dest ghost (inert in 1c.2 — no ghost) + clears any cut"
    );
    Some(TransferControlAck::Aborted { transfer })
}

/// `ReleaseSubscribe` (step 6, the SUCCESS teardown of the demote tail): close the source
/// subscription after demote grace and ack `Released` so the saga reaches `Done`. The cut is
/// already `None` (cleared at `CommitAuthority`'s `store_commit`) and the dest buffer was
/// take-drained there, so the ONLY post-commit remnant to clear is `session.transfer` — pruned
/// here ONLY when bound to THIS transfer (mirroring `apply_abort`'s prune). A stray re-ack of an
/// absent/torn-down transfer is a clean no-op-and-ack: it touches no state and still acks
/// `Released`, so an at-least-once redelivery is idempotent without polluting the journal (the
/// record-then-send guard sees `session.transfer == None` and skips the write, exactly as on
/// `apply_abort`). NOT a routing failure (uncounted) — `transfer_unroutable` stays about genuine
/// failures. Always acks `Released`.
fn apply_release(session: &mut Session, transfer: TransferId) -> Option<TransferControlAck> {
    if session
        .transfer
        .as_ref()
        .is_some_and(|tp| tp.transfer == transfer)
    {
        session.transfer = None; // prune ONLY the matching in-flight transfer (the last remnant)
    }
    tracing::debug!(
        transfer = transfer.0,
        "ReleaseSubscribe closes the source subscription (demote tail) + clears session.transfer"
    );
    Some(TransferControlAck::Released { transfer })
}

/// COLD cut-marker observer (off the hot path): when a transfer is in flight and a client
/// `InputDatagram` carries `is_cut_marker`, derive `CutConfirmed{marker_seq}` and ack it —
/// journaled at step 1 (via the SAME `TransferProgress` accessors as the redelivery gate),
/// so a triple-sent marker re-sends the SAME ack. REQUIRES `RequestCut` to have been issued
/// (`tp.cut_requested`) first (F1): a premature/forged marker must NEVER journal a
/// `CutConfirmed` before its issuing command. 1c sources the marker SCRIPTED (the client
/// emit is 1e — D-5).
fn on_cut_marker(
    tp: &mut TransferProgress,
    bytes: &[u8],
    orchestrator: NodeId,
    outbox: &mut OutboundBox,
) {
    // PEEK the two head fields (seq + is_cut_marker) — never a full InputDatagram decode
    // (D-24 SCALE-CUTDECODE-1): on_cut_marker runs for EVERY input mid-transfer, so it reads
    // only what it needs off the postcard prefix.
    let Ok((seq, is_cut_marker)) = peek_is_cut_marker(bytes) else {
        return; // malformed/truncated/bad-bool = "not a marker"; route_input already counts malformed
    };
    if !is_cut_marker {
        return; // ordinary input
    }
    if !tp.cut_requested {
        return; // F1: a marker before its RequestCut is premature/forged — never confirm it
    }
    const CUT_STEP: u32 = 1; // step 1 = RequestCut/CutConfirmed
    if let Some(prior) = tp.recorded(CUT_STEP) {
        reply_ack(outbox, orchestrator, prior); // triple-sent marker: re-send the SAME ack
        return;
    }
    let ack = TransferControlAck::CutConfirmed {
        transfer: tp.transfer,
        marker_seq: seq,
    };
    tp.journal(CUT_STEP, ack);
    reply_ack(outbox, orchestrator, ack);
}

/// Handle a shard control reply (attach/detach lifecycle), from shard `from`.
fn on_shard_control(
    from: NodeId,
    bytes: &[u8],
    config: &GatewayConfig,
    clock: &ClockSample,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(msg) = postcard::from_bytes::<ShardToGateway>(bytes) else {
        stats.undecodable += 1;
        return;
    };
    match msg {
        ShardToGateway::SessionAttached {
            session: session_id,
            entity,
            frame,
            realm_fence,
        } => {
            {
                let Some(session) = sessions.by_session.get_mut(&session_id) else {
                    // Attach reply for a session that left meanwhile: ignore (the
                    // detach path already ran).
                    return;
                };
                // Promote ONLY from AwaitingAttach. A duplicate reply for an already-Active session is
                // idempotent (as before); CRUCIALLY a late/duplicate `SessionAttached` straggler must
                // NEVER resurrect a `SelfFenced` session (D-3 Slice 5b) — `SessionAttached` is re-emitted
                // on every `AttachSession` retry and rides the gateway↔shard link, which can be HEALTHY
                // while the gateway↔orchestrator link (that drove the self-fence) is partitioned, so a
                // straggler can arrive after `Active → SelfFenced`. Re-promoting it would re-arm the grace
                // clock and resume input/frame egress — re-opening the exact split-brain window the
                // self-fence exists to close. (AwaitingDirectory — an attach before the grant — is
                // likewise not promotable here.) This keeps `Active` monotonic-until-removal.
                if !matches!(session.phase, SessionPhase::AwaitingAttach) {
                    return;
                }
                // THE sole route-mutation primitive: one atomic store (P2 NOW drives this same
                // `store_route` from `CommitAuthority`'s `store_commit`). Attach SETS a fresh
                // realm fence (its distinct carry policy); commit/cut CARRY it.
                let authority = session.hot.route.load().authority;
                store_route(&session.hot, authority, realm_fence, None);
                session.phase = SessionPhase::Active { entity };
                // D-3 Slice 5b: going Active IS a fresh round-trip confirmation (the directory granted
                // and the shard attached) — arm the self-fence deadline from here.
                session.confirmed_at = clock.local_tick;
            }
            // Open the login sub on `config.shard` at the realm fence — the FIRST `open_sub`
            // caller (the transfer dest is the second, 1d.2b). `open_sub` pushes
            // SubscriptionOpened BEFORE publishing the SubTable (X1) and indexes the fan-out.
            let sub = sessions
                .open_sub(session_id, config.shard, frame, realm_fence, outbox)
                .expect("session present (we just held it above this tick)");
            let session = sessions
                .by_session
                .get(&session_id)
                .expect("session present");
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::AuthorityChanged { entity, sub },
            );
        }
        ShardToGateway::SessionDetached { .. } => {
            // The session was already removed on Bye; the confirmation closes the loop.
        }
        ShardToGateway::SubscriptionReady {
            session: session_id,
            entity,
            frame,
            realm_fence,
        } => {
            // FORK 0a (Track R / 1d.2b): the dest (`from`) adopted the crossing entity and is
            // readable. Open a SECOND per-session sub on `from` at the DEST realm fence and
            // RE-POINT the avatar's render authority to it — the read-plane analog of the
            // write-plane `CommitAuthority`. The client then composites the source copy as
            // non-authoritative (renders the avatar ONCE, from the dest sub — invariant A1).
            let Some(session) = sessions.by_session.get(&session_id) else {
                stats.transfer_unroutable += 1; // absent session: counted, never a panic
                return;
            };
            if session.subs.contains_key(&from) {
                return; // duplicate SubscriptionReady (at-least-once): idempotent no-op
            }
            let sub = sessions
                .open_sub(session_id, from, frame, realm_fence, outbox)
                .expect("session present (held immutably just above this tick)");
            let session = sessions
                .by_session
                .get(&session_id)
                .expect("session present");
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::AuthorityChanged { entity, sub },
            );
        }
        ShardToGateway::Frame { .. } => {
            // Frames ride the Snapshot class; one on Control is a peer bug.
            stats.undecodable += 1;
        }
    }
}

/// Fan one shard frame (from shard `from`) out to that shard's subscribers, each at ITS sub
/// id and ITS per-shard accepted fence. The READ-plane heart (1d.2a): iterate ONLY
/// subscribers-of-`from` (H2 reverse index, O(subscribers) not O(all sessions)) and resolve
/// each session's `SubEntry` for `from` off the wait-free hot `SubTable`.
fn on_shard_frame(
    from: NodeId,
    bytes: &[u8],
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let Ok(ShardToGateway::Frame {
        realm_fence,
        source_tick: _,
        snapshot_bytes,
    }) = postcard::from_bytes::<ShardToGateway>(bytes)
    else {
        stats.undecodable += 1;
        return;
    };
    // 1d.5a: peek the `frame_id` ONCE off the wire (the per-observer delivery watermark advances
    // by it). A malformed body fails HERE and the whole frame is abandoned (counted once) — and
    // because the peek validates the leading `sub` varint, the per-session `retag_snapshot_sub`
    // below is then INFALLIBLE (`.expect()`), so there is no second decode-error region.
    let Ok(frame_id) = peek_snapshot_frame_id(&snapshot_bytes) else {
        stats.undecodable += 1;
        return;
    };
    // SCALE-1: re-tag the snapshot body ONCE per distinct sub-id into a SHARED `Arc`; the
    // per-session fan-out is then a cheap fence check + refcount bump, never an O(entities)
    // re-allocation per subscriber.
    let mut retagged: BTreeMap<SubId, vd_sim::io::Bytes> = BTreeMap::new();
    for session_id in sessions.subscribers_of(from) {
        let Some(session) = sessions.by_session.get_mut(&session_id) else {
            // The reverse index and `by_session` are kept in sync by `open_sub`/the
            // drain-sweep; a missing session is an index/table desync (counted, never silent).
            stats.frame_sub_desync += 1;
            continue;
        };
        // D-3 Slice 5b: a SELF-FENCED session no longer acts as authority — it is served NO frames (its
        // subs linger inert in the reverse index until the connection ends or a ResumeTicket adoption
        // re-homes it). A still-attaching session has no subs and is never in this index. Active only.
        if !matches!(session.phase, SessionPhase::Active { .. }) {
            continue;
        }
        // Resolve THIS session's sub + per-shard accepted fence off the wait-free hot `SubTable`,
        // then RELEASE that borrow (the values are `Copy`) so we may advance the cold watermark.
        // A subscriber-in-index ALWAYS has a `SubEntry` (republished together by `publish_subs`);
        // a `None` is an invariant breach, counted (C2 honesty floor), never silent.
        let table = session.hot.subs.load();
        let Some(entry) = table.lookup(from) else {
            stats.frame_sub_desync += 1;
            continue;
        };
        let sub = entry.sub;
        let accepted = entry.accepted;
        let client = session.client;
        drop(table);
        // Per-shard fence (NOT the session-global route fence): the source sub accepts source
        // frames at the source realm fence; the dest sub accepts dest frames at the dest realm
        // fence. A demoted old owner's stale frame is dropped + counted (fence rule 5).
        if realm_fence.is_stale_against(accepted) {
            stats.stale_frames_dropped += 1;
            continue;
        }
        // 1d.5a: an ACCEPTED, past-fence frame ADVANCES this observer-sub's delivery high-water —
        // the standing (a) demote-predicate input. Only delivered (forwarded) frames count; a
        // stale-dropped frame (above) does NOT advance it.
        session
            .delivered
            .entry(sub)
            .and_modify(|w| *w = (*w).max(frame_id))
            .or_insert(frame_id);
        let body =
            retagged.entry(sub).or_insert_with(|| {
                vd_sim::io::bytes(retag_snapshot_sub(&snapshot_bytes, sub).expect(
                    "the frame_id peek validated the sub varint, so the re-tag is infallible",
                ))
            });
        outbox.0.push((
            client,
            MsgClass::Snapshot,
            body.clone(),
            vd_sim::io::Durability::Ephemeral,
        ));
    }
}

/// Handle a directory reply: the Session-key head confirms (or denies) the mint.
fn on_directory_reply(
    reply: DirectoryReply,
    config: &GatewayConfig,
    identity: &NodeIdentity,
    clock: &ClockSample,
    sessions: &mut GatewaySessions,
    stats: &mut GatewayStats,
    outbox: &mut OutboundBox,
) {
    let DirectoryReply::Head {
        key: DirectoryKey::Session(session_id),
        record,
    } = reply
    else {
        return; // entity/realm heads carry no gateway obligation in P1
    };
    let Some(session) = sessions.by_session.get_mut(&session_id) else {
        return; // session left while the reply was in flight
    };
    let granted = record.is_some_and(|r| {
        r.authority == AuthorityRef::Gateway(identity.node_id) && r.fence == session.fence
    });
    // D-3 Slice 5b: an Active session's `Session`-head reply is a RECHECK round-trip (not a mint). An
    // affirming head RE-ARMS the proactive self-fence deadline (we just heard from the directory); a
    // foreign/absent head means the lease was reassigned or reaped, so reactively SELF-FENCE now — the
    // link-alive cure that complements the partition timer (mirrors the shard's reactive realm self-fence).
    if matches!(session.phase, SessionPhase::Active { .. }) {
        if granted {
            session.confirmed_at = clock.local_tick;
        } else {
            session.phase = SessionPhase::SelfFenced;
            stats.sessions_self_fenced_revoked += 1;
        }
        return;
    }
    if !matches!(session.phase, SessionPhase::AwaitingDirectory) {
        return; // AwaitingAttach / already-SelfFenced: no obligation for this head
    }
    if granted {
        // The mint is committed. Welcome the client; attach to the shard.
        session.phase = SessionPhase::AwaitingAttach;
        push_control(
            outbox,
            session.client,
            &ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: session_id,
                session_fence: session.fence,
                epoch: clock.epoch,
            },
        );
        // Relay the cluster tick rate to a minor-1+ client (sender-gates-variants),
        // so it drives its render cursor at the server's rate, not a guess.
        if session.negotiated_minor >= 1 {
            push_control(
                outbox,
                session.client,
                &ServerControlMsg::UniverseRate {
                    tick_hz: config.tick_hz,
                },
            );
        }
        push_to_shard(
            outbox,
            config.shard,
            MsgClass::Control,
            &GatewayToShard::AttachSession {
                session: session_id,
                fence: session.fence,
                account: session.account,
            },
        );
    } else {
        // Mint refused (id collision or foreign holder): close loudly; the client
        // retries login with a fresh Hello.
        stats.session_mints_refused += 1;
        let client = session.client;
        sessions.by_session.remove(&session_id);
        sessions.by_client.remove(&client);
        push_control(
            outbox,
            client,
            &ServerControlMsg::Close {
                reason: "session mint refused by the directory".to_owned(),
            },
        );
    }
}

/// Per-tick retry driver: pending directory grants and shard attaches are
/// re-sent until answered (all idempotent — at-least-once over a lossy fabric).
fn drive_pending_sessions(
    config: Res<GatewayConfig>,
    identity: Res<NodeIdentity>,
    sessions: Res<GatewaySessions>,
    mut outbox: ResMut<OutboundBox>,
) {
    for (session_id, session) in &sessions.by_session {
        match session.phase {
            SessionPhase::AwaitingDirectory => {
                push_directory(
                    &mut outbox,
                    config.orchestrator,
                    DirectoryOp::LeaseGrant {
                        key: DirectoryKey::Session(*session_id),
                        owner: AuthorityRef::Gateway(identity.node_id),
                        fence: session.fence,
                    },
                );
            }
            SessionPhase::AwaitingAttach => {
                push_to_shard(
                    &mut outbox,
                    config.shard,
                    MsgClass::Control,
                    &GatewayToShard::AttachSession {
                        session: *session_id,
                        fence: session.fence,
                        account: session.account,
                    },
                );
            }
            // Active needs no re-drive; a SelfFenced session is deliberately left alone (D-3 Slice 5b) —
            // it is no longer renewed, re-checked, or re-attached, awaiting connection-end / adoption.
            SessionPhase::Active { .. } | SessionPhase::SelfFenced => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use vd_core::pose::FrameRef;
    use vd_core::{EpochId, TickId, UniverseTick};
    use vd_sim::capability::NodeKind;
    use vd_wire::channels::{EntitySnap, InputDatagram, SnapshotDatagram};
    use vd_wire::seams::directory::OwnerRecord;

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

    fn config() -> GatewayConfig {
        GatewayConfig {
            orchestrator: ORCH,
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

    impl Rig {
        fn new() -> Rig {
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

        fn login_with(&mut self, hello: &ClientControlMsg) -> (SessionId, LoginSends) {
            let hello = self.tick(vec![wire(CLIENT, MsgClass::Control, hello)]);
            let session_id = *self
                .world
                .resource::<GatewaySessions>()
                .sessions()
                .collect::<Vec<_>>()
                .first()
                .expect("session pending");
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

    fn hello_msg_minor0() -> ClientControlMsg {
        ClientControlMsg::Hello {
            version: ProtoVersion { major: 1, minor: 0 },
            login: tickets::mint_login(&SIGNING_KEY, AccountId(5), EpochId(9), 1),
        }
    }

    #[test]
    fn a_minor_0_client_is_welcomed_without_the_universe_rate_variant() {
        // Sender-gates-variants: a peer that negotiated minor 0 must NOT be sent the
        // minor-1 UniverseRate (it would desync an old decoder). Welcome only.
        let mut rig = Rig::new();
        let (session_id, sends) = rig.login_with(&hello_msg_minor0());
        let welcomes = decode_controls(&sends[1], CLIENT);
        assert_eq!(
            welcomes,
            vec![ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: session_id,
                session_fence: Fence(1),
                epoch: EpochId(9),
            }],
            "a minor-0 client gets Welcome but NOT UniverseRate"
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
    /// the split-brain-safe `ttl < grace <= ttl + max` ordering is validated orchestrator-side).
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
        let fenced_confirmed =
            rig.world.resource::<GatewaySessions>().by_session[&sid].confirmed_at;
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
        // sub allocated from the monotonic allocator.
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
            ]
        );
        assert_eq!(
            rig.stats(),
            GatewayStats::default(),
            "clean run, zero rejects"
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
            version: ProtoVersion {
                major: ProtoVersion::CURRENT.major + 1,
                minor: 0,
            },
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
        assert_eq!(rig.stats().version_rejected, 1);
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
        assert_eq!(rig.stats().undecodable, 7);
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

        let route_p99 = percentile_unstable(route_read, 99);
        let forward_p99 = percentile_unstable(forward, 99);
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

    /// The p99-style tail of a latency sample set (sort + nearest-rank index). TOTAL — an
    /// empty set is `Duration::ZERO` (no panic), since this is earmarked for extraction to a
    /// shared harness helper for the 2nd hard latency gate (SPIKE-3a, P3) whose caller may not
    /// guarantee non-empty. Hand-rolled (no bench crate — see the spike doc).
    fn percentile_unstable(
        mut samples: Vec<std::time::Duration>,
        pct: usize,
    ) -> std::time::Duration {
        if samples.is_empty() {
            return std::time::Duration::ZERO;
        }
        samples.sort_unstable();
        let rank = samples.len().saturating_mul(pct) / 100;
        samples[rank.min(samples.len() - 1)]
    }

    #[test]
    fn percentile_unstable_total_over_empty_single_and_edges() {
        use std::time::Duration;
        let d = Duration::from_nanos;
        // Empty → ZERO (the total-ness the future harness reuse relies on; no panic).
        assert_eq!(percentile_unstable(Vec::new(), 99), Duration::ZERO);
        // Single element → itself at any percentile.
        assert_eq!(percentile_unstable(vec![d(5)], 99), d(5));
        assert_eq!(percentile_unstable(vec![d(5)], 0), d(5));
        // Nearest-rank over a known set; p100 clamps to the max (no out-of-bounds).
        let s = vec![d(10), d(40), d(20), d(30), d(50)]; // sorts to 10,20,30,40,50
        assert_eq!(percentile_unstable(s.clone(), 99), d(50)); // rank 4
        assert_eq!(percentile_unstable(s.clone(), 100), d(50)); // rank 5 → clamp to 4
        assert_eq!(percentile_unstable(s, 50), d(30)); // rank 2
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
                match postcard::from_bytes::<InterShardFlow>(bytes)
                    .expect("gateway sends a valid flow")
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
    fn request_cut_pushes_to_client_and_defers_the_ack_to_the_marker() {
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        // RequestCut → exactly one ServerControlMsg::RequestCut to the CLIENT, ZERO SagaAck.
        let sent = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![ServerControlMsg::RequestCut { transfer: XFER }]
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "the ack is deferred to the marker"
        );

        // The scripted in-band cut marker on the INPUT flow → CutConfirmed{marker_seq}.
        let sent = rig.tick(vec![marker_input(42)]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::CutConfirmed {
                transfer: XFER,
                marker_seq: 42,
            }]
        );
    }

    #[test]
    fn a_redelivered_request_cut_after_the_marker_re_pushes_never_re_confirms() {
        // F1: RequestCut is NOT self-acking — its CutConfirmed is journaled at step 1 by the
        // marker observer. A redelivered RequestCut must re-push to the client (idempotent),
        // never answer the COMMAND with the marker's CutConfirmed from the shared slot.
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
        let _ = rig.tick(vec![request_cut()]);
        let _ = rig.tick(vec![marker_input(5)]); // CutConfirmed journaled at step 1
        // Redeliver RequestCut: re-pushes to the client, does NOT re-send CutConfirmed.
        let sent = rig.tick(vec![request_cut()]);
        assert_eq!(
            decode_controls(&sent, CLIENT),
            vec![ServerControlMsg::RequestCut { transfer: XFER }],
            "the redelivered RequestCut re-pushes to the client"
        );
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "it never answers the command with the marker's CutConfirmed"
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
    fn a_triple_sent_cut_marker_resends_the_same_cut_confirmed() {
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
        // Three sends of the marker at the same seq (the channel's triple-send) → the SAME
        // CutConfirmed each time (journaled at step 1), never three distinct acks.
        let a = rig.tick(vec![marker_input(9)]);
        let b = rig.tick(vec![marker_input(9)]);
        let c = rig.tick(vec![marker_input(9)]);
        let confirmed = vec![TransferControlAck::CutConfirmed {
            transfer: XFER,
            marker_seq: 9,
        }];
        assert_eq!(acks_to_orch(&a), confirmed);
        assert_eq!(acks_to_orch(&b), confirmed);
        assert_eq!(acks_to_orch(&c), confirmed);
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
    fn a_cut_marker_before_its_request_cut_is_dropped_not_confirmed() {
        // F1: a marker that arrives before `RequestCut` was issued (a premature or forged
        // emit) must NOT be confirmed — no `CutConfirmed`, nothing journaled at step 1.
        let mut rig = Rig::new();
        let (sid, _) = rig.login();
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: SHARD,
        })]);
        // No RequestCut issued yet — feed a cut marker directly.
        let sent = rig.tick(vec![marker_input(7)]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![],
            "a marker before RequestCut is dropped, never confirmed"
        );
        // And a LATER RequestCut + marker still confirms cleanly (the premature one left no
        // poisoned journal entry).
        let _ = rig.tick(vec![saga_cmd(TransferControl::RequestCut {
            transfer: XFER,
            session: sid,
        })]);
        let sent = rig.tick(vec![marker_input(8)]);
        assert_eq!(
            acks_to_orch(&sent),
            vec![TransferControlAck::CutConfirmed {
                transfer: XFER,
                marker_seq: 8,
            }],
            "the real cut (after RequestCut) confirms with its own marker_seq"
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

    // ---- Slice 1d.2a: the route-table reshape (sub registry + reverse index) ----

    /// Build a bare `GatewaySessions` with ONE Active session whose only sub is on `SHARD`,
    /// exactly as a real login would leave it — the substrate for the primitive unit tests.
    fn one_active_session() -> (GatewaySessions, SessionId, OutboundBox) {
        let mut sessions = GatewaySessions::default();
        let sid = SessionId(0xA11A);
        sessions.by_session.insert(
            sid,
            Session {
                client: CLIENT,
                account: AccountId(5),
                fence: Fence(1),
                phase: SessionPhase::Active {
                    entity: EntityId(77),
                },
                next_sub: 0,
                confirmed_at: TickId(0),
                negotiated_minor: 1,
                transfer: None,
                subs: BTreeMap::new(),
                delivered: BTreeMap::new(),
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
        sessions.close_sub(sid, SHARD, &mut outbox);
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
        sessions.close_sub(sid, SHARD, &mut outbox2);
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
        sessions.close_sub(SessionId(0xDEAD), SHARD, &mut outbox); // unknown session
        sessions.close_sub(sid, DEST, &mut outbox); // session does not subscribe to DEST
        assert!(outbox.0.is_empty(), "neither path emits a close");
        // The original SHARD sub is untouched.
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
                client: NodeId(101),
                account: AccountId(6),
                fence: Fence(1),
                phase: SessionPhase::Active {
                    entity: EntityId(88),
                },
                next_sub: 0,
                confirmed_at: TickId(0),
                negotiated_minor: 1,
                transfer: None,
                subs: BTreeMap::new(),
                delivered: BTreeMap::new(),
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
        // Close + sweep ONLY session A.
        let mut ob = OutboundBox::default();
        sessions.close_sub(sid_a, SHARD, &mut ob);
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
        // Open a transfer progress so ReleaseSubscribe is bound.
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
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
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
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
        // The transfer reaches commit: the dest adopts and announces its read-sub.
        let _ = rig.tick(vec![saga_cmd(TransferControl::PrepareSubscribe {
            transfer: XFER,
            session: sid,
            dest: DEST,
        })]);
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
            ],
            "the dest sub opens (X1) and authority re-points to SubId(1) (FORK 0a / A1)"
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
            acks_to_orch(&sent)
                .contains(&TransferControlAck::DeliveredToObservers { transfer: XFER }),
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
        let renewed_sessions = |sent: &[(NodeId, MsgClass, Vec<u8>)]| -> Vec<SessionId> {
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
        };

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
}
