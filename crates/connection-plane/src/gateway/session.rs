//! THE SESSION TABLE: one client's whole server-side record, hot half and cold half.
//!
//! Owns: the cold `Session` record (phase, subscriptions, transfer progress, home bootstrap) and
//! the shared hot half the 20 Hz path reads WAIT-FREE — the route snapshot behind one atomic swap
//! and the immutable subscription table published whole on every membership change.
//!
//! Does NOT own: the writers. Exactly one primitive publishes the route and exactly one publishes
//! the subscription table, and both live in `routing`; keeping them there is what makes "who can
//! mutate the hot plane" answerable by grep rather than by care.

use super::{GatewayConfig, GatewayStats, publish_subs, push_control, session_target};
use crate::window;
use arc_swap::ArcSwap;
use bevy_ecs::prelude::Resource;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use vd_core::UniverseTick;
use vd_core::pose::{FrameRef, RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::rng::SplitMix64;
use vd_core::{AccountId, EntityId, Fence, NodeId, SessionId, TickId, TransferId};
use vd_sim::runtime::OutboundBox;
use vd_wire::channels::{ServerControlMsg, SubId};
use vd_wire::intershard::InteriorRelay;
use vd_wire::seams::transfer_control::TransferControlAck;
use vd_wire::session_flow::{BodyStmt, WindowId, WindowScope};

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
pub(crate) struct TransferProgress {
    /// Binds this progress to ONE transfer; a command/marker for a different transfer on
    /// this session is rejected (counted), never absorbed against the wrong saga.
    pub(crate) transfer: TransferId,
    /// Whether `RequestCut` has been issued for this transfer — the precondition for
    /// confirming a cut marker. A marker that arrives before `RequestCut` (a premature or
    /// forged emit) is dropped, so a `CutConfirmed` can never be journaled before its
    /// issuing command (the saga FSM also gates `CutConfirmed` by state; this is the
    /// gateway-side half of that guard).
    pub(crate) cut_requested: bool,
    /// The transfer's DEST shard, captured from `PrepareSubscribe` (the first command, which
    /// always carries it). Its consumer (1d.2): the PRECISE abort-time read-sub close — on
    /// `AbortTransfer` (which carries no dest of its own) the gateway closes EXACTLY this transfer's
    /// dest sub, mirroring `ReleaseSubscribe`'s precise `src`. NEVER an "any sub != config.shard"
    /// heuristic, which would close the player's CURRENT live sub on a chained transfer and ALL
    /// composited subs (ship/host/planet) in the N-shard end goal. `close_sub` no-ops if the dest
    /// sub is not (yet) open (abort normally runs pre-CAS, before the dest sub exists).
    pub(crate) dest: NodeId,
    /// RLM 5f-4e — the SOURCE home this crossing is DEMOTING away from: the `home_shard` value
    /// `CommitAuthority` displaced when it re-pointed the routing target at `dest`. `Some` EXACTLY inside
    /// the demote tail (`CommitAuthority` → the terminal that closes the source sub); `None` before commit
    /// (a fresh progress) and for a session that never resolved a dynamic home (a static login's
    /// `home_shard` is `None`).
    ///
    /// WHY IT IS STASHED RATHER THAN RELEASED AT COMMIT (the REJECT-class blind window this closes): the
    /// client's READ subscription on the SOURCE stays OPEN across the whole demote grace — it is closed only
    /// at `ReleaseSubscribe`, a later tick — and the source keeps SHIPPING `Snapshot`/`RealmSnapshot` frames
    /// the whole time (the composite the seamless crossing is built on). Dropping the source's runtime-roster
    /// claim at commit un-routes it, so for a DEMAND-SPAWNED source (never in the FROZEN `known_shards`) every
    /// one of those frames falls through to the client branch and is counted `undecodable` — the player goes
    /// BLIND for the entire demote tail, right after every crossing. So the claim is HELD here and handed
    /// back at the terminal that actually closes the sub (`ReleaseSubscribe` / `AbortTransfer`), or at the
    /// session's exit ([`GatewaySessions::release_session_claims`]) if the client quits inside the window.
    ///
    /// It is INTERNAL gateway state — NOT on the wire (no `InterShardFlow` arm, no `TransferControl`
    /// variant, no `PROTO_MINOR` bump); commit already knows the old home locally.
    pub(crate) demoting_home: Option<NodeId>,
    /// The applied-steps idempotency journal: `(transfer, step_id) -> the recorded ack`,
    /// re-sent VERBATIM on an at-least-once redelivery (never re-applies the effect),
    /// reached ONLY through [`TransferProgress::recorded`] / [`TransferProgress::journal`]
    /// (the gateway's ONE dedup accessor — never touched inline). The key is the wire's
    /// `IdempotencyKey::TransferStep` — the SAME key the 1d durable redb `applied_steps`
    /// table builds (HR3 one machinery, many stores; only the STORE differs per altitude).
    /// RAM-ONLY by design — the gateway is soft-state (on resume it re-registers with the
    /// saga, never replays from RAM); the DURABLE table is the dest shard's at 1d (DEFERRED
    /// D-22). Bounded: O(phases) per live transfer, dropped whole on terminal/Bye/mint-refusal.
    pub(crate) applied: BTreeMap<(TransferId, u32), TransferControlAck>,
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
    pub(crate) dest_buffer: VecDeque<Vec<u8>>,
}

/// RLM 5f-4 — the DEFERRED runtime-roster edits ONE `TransferControl` command produced, applied only after
/// the `&mut Session` borrow ends. Exactly the [`GatewaySessions::close_sub`] / `subs_to_close` shape and
/// for the same reason: a `&mut Session` handed out of `by_session` may NEVER be held across a
/// [`GatewaySessions`] mutation, and every roster edit mutates `dynamic_shards`.
///
/// Claims are applied BEFORE releases, so a re-claim of the SAME node — a second `PrepareSubscribe` naming
/// the dest an earlier one already claimed — can never transiently drop that node off the roster (it must
/// stay dispatchable straight through the defensive replace).
#[derive(Debug, Default)]
pub(crate) struct RosterEdits {
    /// Crossing dests to CLAIM, each through the config-gated [`GatewaySessions::claim_crossing_dest`].
    pub(crate) claim: Vec<NodeId>,
    /// Roster claims to RELEASE. `None` — a static session's absent home, or a pre-commit transfer's absent
    /// `demoting_home` — is the no-op arm of [`GatewaySessions::release_dynamic_shard`], so an `Option` role
    /// can be pushed VERBATIM (which is also why adding the RLM 5f-4e demoting-source role introduced no new
    /// branch at any release site). A single command may push SEVERAL: a terminal returns both the crossing
    /// dest and the demoting source; a defensive Prepare replace returns both of the displaced progress's.
    pub(crate) release: Vec<Option<NodeId>>,
}

impl TransferProgress {
    /// The recorded outcome for this transfer's `step`, or `None` if not yet applied — the
    /// ONE dedup READ (the redelivery gate + the cut-marker observer both call it). Keyed by
    /// `(self.transfer, step)` = the wire's `IdempotencyKey::TransferStep`.
    pub(crate) fn recorded(&self, step: u32) -> Option<TransferControlAck> {
        self.applied.get(&(self.transfer, step)).copied()
    }

    /// Record `ack` at this transfer's `step` — the ONE dedup WRITE, so the recorded value
    /// is re-sent verbatim on a redelivery. Keyed by `(self.transfer, step)`.
    pub(crate) fn journal(&mut self, step: u32, ack: TransferControlAck) {
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
    pub(crate) by_shard: Box<[SubEntry]>,
}

impl SubTable {
    /// The accepted subscription THIS session tagged for `shard`, or `None` if the session
    /// does not subscribe to it. Binary search over the ≤4 sorted entries.
    #[must_use]
    pub(crate) fn lookup(&self, shard: NodeId) -> Option<&SubEntry> {
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

/// Where one session is in its login lifecycle. `PartialEq` so tests assert the phase by EQUALITY
/// (`assert_eq!`), never `assert!(matches!(…))` whose false arm is uncoverable (HR5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SessionPhase {
    /// Session-key grant sent; awaiting the directory head (retried every tick —
    /// idempotent by fence).
    AwaitingDirectory,
    /// RLM 5f-3d — the DYNAMIC-HOME hold: the session's lease is COMMITTED (the client is already
    /// `Welcome`d) and its home realm has been DEMANDED, but that realm's shard is still booting, so there
    /// is no node to attach to yet. The gateway holds the client here — SEAMLESSLY: no `Close`, no
    /// teleport, no fallback attach to some other shard, no loading screen; the client is simply welcomed
    /// and frame-less until its own home attaches. While here the gateway (a) RE-SEEDS the home demand and
    /// (b) re-polls `HeadRead{Realm(home.lowered())}` on the [`SeedInjectorConfig::redrive_interval_ticks`]
    /// cadence, and the bounded [`SeedInjectorConfig::bootstrap_ttl_ticks`] guarantees the hold ENDS —
    /// loudly — rather than hanging. Entered ONLY in dynamic mode
    /// ([`GatewayConfig::dynamic_home_mode`]); a static login never sees this phase.
    ///
    /// `home_rid` names the realm being waited on — the KEY into [`GatewaySessions::home_bootstraps`],
    /// where the wait's heavy state lives ONCE PER REALM (the resolved lineage [`RealmCoord`], the cadence
    /// anchor, the member set). At a 100K mass login onto one home that is ONE lineage `Vec` and ONE
    /// re-drive, not one per session. [`Session::home_rid`] is the same id as a standing copy that outlives
    /// this phase (the bounded-TTL diagnostic reads it from `AwaitingAttach`, where the phase payload is
    /// gone).
    AwaitingHomeRealm { home_rid: RealmId },
    /// Directory granted (and, in dynamic mode, the home realm RESOLVED); attach sent to the session's
    /// target shard — `session.home_shard` when dynamically resolved, else the static `config.shard` —
    /// retried until attached.
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
pub(crate) struct SubRecord {
    pub(crate) sub: SubId,
    pub(crate) frame: FrameRef,
    /// The per-shard accepted frame fence (mirrored into [`SubEntry::accepted`]).
    pub(crate) accepted: Fence,
    pub(crate) state: SubState,
}

/// A subscription's lifecycle state. `Draining` = `SubscriptionClosing` sent; it stays in
/// the `SubTable` + reverse index for ONE more tick so an in-flight straggler frame is
/// still routed (drained), then the cold drain-sweep removes it (C2 — never a silent drop).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubState {
    Active,
    Draining,
}

/// One session's cold record (the hot part is the shared `Arc<SessionHot>`).
#[derive(Debug)]
pub(crate) struct Session {
    pub(crate) client: NodeId,
    pub(crate) account: AccountId,
    pub(crate) fence: Fence,
    pub(crate) phase: SessionPhase,
    pub(crate) next_sub: u32,
    /// RLM 5f-3d (the D-34 field) — the session's DYNAMICALLY resolved home shard: `Some` once the
    /// `Realm(home_rid)` head named the node that owns its home realm. THE routing field: every shard-ward
    /// send resolves through [`session_target`] = `home_shard.unwrap_or(config.shard)`, so a static
    /// (unarmed) session — forever `None` — is byte-identical to the pre-5f-3d gateway.
    ///
    /// RLM 5f-4 closes D-34's A1 (AUTHORITY-FOLLOWING DETACH): it is ALSO re-pointed at `CommitAuthority`
    /// (to the transfer's `dest`), so after a crossing the routing target IS the current authority. A session
    /// that never resolved a dynamic home and never crossed stays `None` forever. D-34's COMPOSITED-SUBS
    /// clause remains OWED: this field names ONE node, whereas the source keeps its own read sub (and its
    /// ghost) until `ReleaseSubscribe` — see `TransferProgress::demoting_home`.
    pub(crate) home_shard: Option<NodeId>,
    /// THE SKY THIS SESSION HOLDS (S11) — the generation the client last stated, whether it read the
    /// catalogue from its own disk cache or assembled it from the wire. `None` means it holds nothing,
    /// which is the safe default: an unknown client is SERVED the sky rather than denied it.
    ///
    /// ★ THE MEMORY LIVES HERE BECAUSE THE HOLDER LIVES HERE. The shard used to keep this, keyed by
    /// gateway, and it could not be made correct: a gateway serves many clients and holds one memory,
    /// so forgetting too early re-sent 7.0 MB to a client that had it, and remembering too long sent a
    /// NEW client nothing at all. One row per session has neither failure, because a session is exactly
    /// one holder of exactly one sky.
    pub(crate) sky_held: Option<u64>,
    /// ★ HOW FAR THROUGH THE CATALOGUE THIS SESSION HAS BEEN SERVED (2026-08-29). The sky is sent a
    /// bounded number of parts per beat, so a beat must know where the last one stopped.
    ///
    /// It exists because the unpaced version re-sent the WHOLE sky on every beat until the client
    /// confirmed: MEASURED at S12's census, 10.72 MB in 1 309 parts, per beat, per client. The client
    /// could not ingest that, so it never confirmed, so it was sent again — and the flood starved the
    /// login handshake sharing the same reliable link. A client sat at "authenticating" forever.
    ///
    /// Reset to zero when the sky's generation changes, because a new sky is a new transfer.
    pub(crate) sky_parts_sent: u32,
    /// RLM 5f-3d — the STANDING home-realm identity of a dynamic session: set once at the committed lease
    /// and NEVER cleared, so it outlives the `AwaitingHomeRealm` phase payload. Two live readers: the
    /// bounded-TTL Close diagnostic (which can fire in `AwaitingAttach`, where the phase payload is gone —
    /// this is then the ONLY surviving name of the realm that failed to boot) and the
    /// [`GatewaySessions::end_home_wait`] index removal on every session exit. `None` ⇒ a static session.
    pub(crate) home_rid: Option<RealmId>,
    /// The composed realm feed's per-session monotone frame counter (proto_minor 18, §2.4): the
    /// connection plane stamps it on every composed [`RealmSnapshotDatagram`] it authors — lawful
    /// because a composed row is a NEW row it authors from attested inputs — so the client's
    /// staleness gate is ONE counter (single author) plus the epoch. Chunks of one tick share
    /// one id (the sibling-chunk rule); a fresh tick increments it.
    pub(crate) realm_feed_frame_id: u64,
    /// The reliable scene lane's send-on-change baseline: the (realm → bag) content of the last
    /// level/delta emitted at the CURRENT epoch. Reset (to the full level) on every epoch bump;
    /// diffed per tick into `RealmSceneDelta`s at a stable epoch. Poses deliberately absent —
    /// position changes ride the unreliable per-tick datagram, never the reliable lane.
    pub(crate) scene_sent: BTreeMap<RealmId, Vec<u8>>,
    /// WHERE THIS LOGIN'S AVATAR GOES, measured from its home realm's own centre and stamped with that
    /// realm's frame. Resolved ONCE, in the same descent that resolves the home lineage, and then repeated
    /// verbatim on every `AttachSession` (including the retries) so a re-attach cannot land the player
    /// somewhere else than the first attempt would have.
    ///
    /// `None` for a session with no dynamic home — a static cluster attaches to a fixed shard that is not
    /// in general the realm the stored position is in, and handing it a pose measured somewhere else is
    /// exactly the confusion this field exists to end. The shard then births at its own origin, which is
    /// what every rig does today.
    pub(crate) spawn: Option<StampedPose>,
    /// RLM 5f-3d — the HARD deadline (gateway LOCAL tick) of the BOUNDED pre-Active dynamic-home bootstrap:
    /// `Some` EXACTLY while a dynamic session is booting — spanning `AwaitingHomeRealm` **and** the
    /// dynamic-target `AwaitingAttach` (a freshly spawned shard can die between head-resolve and
    /// `SessionAttached`) — and cleared at the `Active` promote. `None` for a STATIC session, always.
    /// It lives HERE rather than in the phase precisely because it must outlive the `AwaitingHomeRealm`
    /// phase; the re-drive ANCHOR (`since`) conversely lives IN that phase, where it is total (a session in
    /// `AwaitingHomeRealm` always has one — no `Option` arm that no test could ever reach).
    pub(crate) bootstrap_deadline: Option<TickId>,
    /// D-3 Slice 5b — the `local_tick` of the last `Session`-head ROUND-TRIP confirmation (the reply that
    /// affirmed THIS gateway still owns the session lease). The partition detector for the proactive
    /// self-fence: set when the session goes `Active` (the attach IS a confirmation) and re-armed on every
    /// affirming recheck reply; under a partition (no reply) it FREEZES while `local_tick` climbs, and
    /// `lease_self_fence_due` fires once the gap exceeds the grace. Meaningful only while `Active`.
    pub(crate) confirmed_at: TickId,
    /// The negotiated proto minor for this connection (the sender-gates-variants
    /// rule): minor-1+ variants like `UniverseRate` are emitted only when `>= 1`.
    pub(crate) negotiated_minor: u16,
    /// COLD transfer state, `None` until a saga's `PrepareSubscribe` opens one (1c.2).
    /// Dropped whole on the in-flight terminal (`AbortTransfer`) or when the session ends.
    pub(crate) transfer: Option<TransferProgress>,
    /// The COLD authoritative subscription set, keyed by shard `NodeId` (1d.2a; ≤ ~4).
    /// Mutated ONLY through `open_sub`/`close_sub`/the drain-sweep, each followed by
    /// `publish_subs` (the sole `SubTable` writer — HR3). The hot `SubTable` is its
    /// forwarding projection.
    pub(crate) subs: BTreeMap<NodeId, SubRecord>,
    /// 1d.5a — the per-observer-sub delivery high-water: `SubId -> highest delivered frame_id`.
    /// COLD (off the wait-free `SubTable` — HR1), written in `on_shard_frame` at the push instant
    /// (only ACCEPTED, past-fence dest frames advance it), removed when the sub is swept (no leak).
    /// The (a) demote predicate's input: the standing "every current dest observer got >=1 frame"
    /// watermark. Absent ≡ watermark 0 ≡ re-blocks (an observer opening mid-demote is undelivered).
    pub(crate) delivered: BTreeMap<SubId, u64>,
    /// THE WINDOW LANE's session lineage (Slice B — `docs/design/window_lane.md` §2.6.2), the
    /// realm chain root→leaf this session stands under. SESSION HISTORY ONLY, never a forest
    /// read: seeded by the login descent's [`RealmCoord`] (dynamic) or the attach frame's realm
    /// (static), and updated at each crossing's `SubscriptionReady` — a realm already in the
    /// lineage TRUNCATES back to it (an outward cross), anything else APPENDS below the previous
    /// leaf (an inward cross; "travel is always out into the shared parent and in again" — SL2,
    /// so the previous leaf IS the parent). A wrong guess is fail-closed downstream: the opened
    /// `Child` window is refused by the shard (`window_child_unrostered`) and never confirms, so
    /// the chain simply ends there. Empty until the session resolves a home/attach.
    pub(crate) lineage: Vec<RealmId>,
    /// THE WINDOW LANE's per-session composed scene: the derived chain, the origin marker +
    /// epoch, the held strata and the fresh-fold ring the composed emissions read from. Dropped
    /// with the session — zero sessions, zero composer state.
    pub(crate) shadow: window::ShadowScene,
    pub(crate) hot: Arc<SessionHot>,
}

/// The session table: by session id (authoritative) and by client connection
/// (in-process: the client's NodeId IS the connection).
#[derive(Resource, Debug, Default)]
pub struct GatewaySessions {
    pub(crate) by_session: BTreeMap<SessionId, Session>,
    pub(crate) by_client: BTreeMap<NodeId, SessionId>,
    /// ★ THE SKY, CUT INTO WIRE PARTS ONCE (2026-08-29) — the generation it was cut for, and the cut.
    ///
    /// Cutting the catalogue encodes EVERY star to measure it, one heap allocation each, then copies
    /// every row again into the parts. On THE world that is 233 220 stars per cut. The keep-alive beat
    /// did it afresh every time, forever, for a galaxy that cannot change: the sky is folded once at
    /// boot and `GatewayConfig` is immutable thereafter.
    ///
    /// ★ WHY IT SHOWED UP AS THE PLAYER'S POSITION STUTTERING. The cut runs on the gateway's tick, in
    /// front of the pose lane. A tick that spends its time cutting the sky delivers poses late, and a
    /// late pose is a position that jumps. The owner reported exactly that, and reported that it began
    /// when the star count grew — which is the signature of a cost that scales with the census.
    ///
    /// Keyed by generation rather than merely "built once", so a gateway whose sky is ever replaced
    /// re-cuts instead of serving the old galaxy — the failure a bare `Option` would hide.
    pub(crate) sky_cut: Option<(u64, Vec<Vec<vd_core::look::StarRow>>)>,
    /// The per-session fan-out reverse index `shard -> {sessions subscribing to it}` (FORK 5
    /// / H2), cold-maintained by `open_sub`/`close_sub`/the drain-sweep alongside the hot
    /// `SubTable`. It makes `on_shard_frame` iterate ONLY subscribers-of-`from`, not all
    /// sessions. It governs FAN-OUT only — NEVER node-class dispatch (that is the stable
    /// `config.known_shards`), so a refcount slip cannot mis-route a client. A `Draining`
    /// sub stays indexed for one tick (its straggler is drained), then removed.
    pub(crate) subscribed_shards: BTreeMap<NodeId, BTreeSet<SessionId>>,
    /// RLM 5f-3d — the RUNTIME routable-shard roster: `demand-spawned shard -> how many session ROLES are
    /// held on it`. A dynamically spawned shard's `NodeId` is minted at spawn time and can NEVER be in the
    /// FROZEN [`GatewayConfig::known_shards`], so this is the other half of the ONE dispatch predicate
    /// [`is_routable_shard`]. REFCOUNTED (not a grow-only set) so a long-lived gateway does not accumulate
    /// one entry per realm shard the cluster ever spun up across 100K-realm churn: a node LEAVES the roster
    /// when its last role does. Empty (⇒ dispatch byte-identical) unless in dynamic-home mode.
    ///
    /// RLM 5f-4 / 5f-4e — there are now exactly THREE role kinds, each claiming ONE refcount:
    /// - the session's HOME shard — claimed at the login home resolve (`on_home_realm_head`) and re-claimed
    ///   for the new home at `CommitAuthority`; released at the session's exit
    ///   ([`GatewaySessions::release_session_claims`]);
    /// - an in-flight transfer's CROSSING DEST — claimed at `PrepareSubscribe`
    ///   ([`GatewaySessions::claim_crossing_dest`]), released at its terminal (`ReleaseSubscribe` /
    ///   `AbortTransfer`), on a defensive Prepare replace, and at the session's exit;
    /// - a committed transfer's DEMOTING SOURCE home (`TransferProgress::demoting_home`) — the claim commit
    ///   DISPLACED but deliberately did NOT release, because the client's read sub on the source outlives the
    ///   commit by the whole demote grace. Released at the terminal that closes that sub, on a defensive
    ///   Prepare replace, and at the session's exit.
    ///
    /// So mid-demote-tail a crossing session holds TWO refcounts on the dest (home + crossing) and ONE on the
    /// source — deliberately: the dest stays routable when the tail releases the crossing one, and the SOURCE
    /// stays routable for as long as the client is still reading it (without which the player went blind for
    /// the whole tail after every crossing).
    pub(crate) dynamic_shards: BTreeMap<NodeId, u32>,
    /// Nodes that ANNOUNCED themselves as shards over the authenticated mesh, and are therefore heard.
    ///
    /// A SET, not a refcount, because it is not a claim anybody holds — it is a fact about a process
    /// being up. That is exactly why the previous arrangement failed: a transfer's refcount made node
    /// class last as long as one hand-off, so a running shard went mute the moment its hand-off ended,
    /// or never spoke at all if no hand-off ever named it.
    ///
    /// ⚠ REMOVAL IS OWED, and rides with the reaper: nothing takes a node OUT of this set today, so a
    /// long-lived cluster with realm churn accumulates ids. Bounded by nodes ever spawned, which is fine
    /// at the tier this runs at and is NOT fine in a real deployment. The signal to remove on is the same
    /// one the reaper needs — the orchestrator tearing a realm down — so the two land together rather
    /// than growing two half-answers to one lifecycle question.
    pub(crate) announced_shards: BTreeSet<NodeId>,
    /// The nodes the OWNERSHIP RECORD shows holding a realm — the authoritative answer to whose frames
    /// this router may read, pushed by the orchestrator as a whole level (`on_shard_roster`).
    ///
    /// This is the one that DECIDES. `announced_shards` above is what a shard says about itself and is
    /// now only the resync path for the window before a record arrives; a node cannot put itself HERE,
    /// because getting here means holding a realm through the fence commit.
    pub(crate) record_shards: BTreeSet<NodeId>,
    /// The tick of the newest roster applied, so a reordered or redelivered push cannot resurrect a
    /// superseded one. Starts at zero: before any roster arrives this router has been told nothing, and
    /// every level is newer than nothing.
    pub(crate) roster_at: UniverseTick,
    /// RLM 5f-3d — the per-REALM home-BOOTSTRAP index: `home realm -> the ONE wait every session booting
    /// into that realm shares`. Two independent things make it per-REALM rather than per-session, both
    /// load-bearing at 100K scale:
    /// - a `Realm` head reply carries NO session id, so without an index every reply would cost an O(S) scan
    ///   over all sessions — O(S²) across a mass login. With it ONE reply resolves the WHOLE member set at
    ///   O(log R + k).
    /// - the RE-DRIVE iterates this map, so N sessions booting into ONE home cost ONE `RealmDemand` + ONE
    ///   `HeadRead` per cadence instead of 2N. Coalescing is provably safe: the only per-session field a
    ///   demand carries is the audit-only `parent_fence`, which the orchestrator's ledger folds
    ///   order-independently and never reads for a decision (`rlm.rs` `update_fence`); child, verb and tick
    ///   are identical for every member.
    ///
    /// An entry SURVIVES the head resolve (it is not the wait's terminator — the `Active` promote is), which
    /// is what keeps the demand re-seeded across the attach round-trip; see [`HomeWait::resolved`].
    /// Maintained by the [`GatewaySessions::begin_home_wait`] / [`GatewaySessions::end_home_wait`] pair:
    /// EVERY exit (the `Active` promote, the bootstrap-TTL Close, `Bye`) drops that member through
    /// `end_home_wait`, and the realm's entry is PRUNED when its last member leaves — so neither an entry nor
    /// a lineage `Vec` can outlive its sessions. Empty unless in dynamic-home mode (⇒ byte-identical).
    pub(crate) home_bootstraps: BTreeMap<RealmId, HomeWait>,
    /// THE WINDOW LANE (Slice A, docs/design/window_lane.md §2.3): the windows this gateway holds
    /// open, keyed by the id IT minted (monotone, never reused — the `SubId` discipline, so a
    /// straggler row from a closed window drops by id mismatch, never a guess). DERIVED each tick
    /// from the Active sessions' own routing state ([`desired_windows`]) and diffed — so zero
    /// sessions structurally means zero window state (the design's teardown test), and a crossing
    /// re-derives the set with no bespoke hook. Empty on every pre-window rig ⇒ byte-identical.
    pub(crate) windows: BTreeMap<WindowId, GatewayWindow>,
    /// The monotone [`WindowId`] mint for `windows` (starts at 1 — id 0 is never issued, so a
    /// zero-initialized forgery is never a live window).
    pub(crate) next_window: u64,
    /// THE WINDOW LANE's realm→node answers (Slice B — `docs/design/window_lane.md` §2.6.2):
    /// which node currently heads each realm the Active sessions' LINEAGES name. Fed from the
    /// gateway's OWN routing state (every Active sub's `(node, frame)`) and from the directory's
    /// `Realm` head replies (the EXISTING `HeadRead`/`Head` pair, re-polled on the window
    /// keep-alive cadence for lineage ancestors the sessions never subscribed to — e.g. the
    /// galaxy above a logged-in system). NEVER a world model: every entry is a directory answer
    /// or a live subscription, and the map is PRUNED each tick to the realms the Active
    /// sessions' lineages + subs actually name — zero sessions ⇒ empty (the teardown truth).
    pub(crate) realm_heads: BTreeMap<RealmId, NodeId>,
}

/// RLM 5f-3d — ONE realm's pre-Active home bootstrap: the state EVERY session booting into that realm
/// shares. Held once per REALM, which is simultaneously the memory win (one lineage `Vec` for a whole mass
/// login) and the coalescing point (one demand + one head-read per cadence, however many sessions wait).
/// `PartialEq`/`Debug` so tests assert the WHOLE index by equality (never `matches!` — HR5); NO `Clone` —
/// nothing ever copies a wait.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HomeWait {
    /// The SERVER-derived home lineage, descended ONCE per realm. Every re-seed rebuilds its demand off THIS
    /// coord through [`demand_for_home`] — never a fresh forest descend (O(1) per re-drive), and never a
    /// demand naming a different realm than the one being waited on.
    pub(crate) coord: RealmCoord,
    /// The gateway LOCAL tick this wait OPENED: the re-drive cadence anchor, and the anchor a member's
    /// DYNAMIC attach retry rides ([`GatewaySessions::attach_anchor`]). TOTAL (every entry has one — no
    /// unreachable `Option` arm) and PER-REALM, so a mass login across MANY homes spreads its re-drives over
    /// the cadence window instead of spiking them on one tick.
    pub(crate) since: TickId,
    /// The account whose per-account sentinel `parent_fence` the COALESCED demand carries — the first member
    /// to open the wait. ONE representative is correct and deliberate: that fence is audit-only at the
    /// orchestrator (`rlm.rs` `update_fence` never reads it for a decision), so pinning it per REALM keeps
    /// the re-seed's BYTES stable across member churn instead of flapping with whichever waiter is iterated.
    pub(crate) account: AccountId,
    /// The sessions currently booting into this realm — every one of them pre-`Active`. NON-EMPTY by
    /// construction: `end_home_wait` prunes the entry when the last member leaves.
    pub(crate) members: BTreeSet<SessionId>,
    /// Has a `Realm` head reply already named this realm's node for every CURRENT member? It gates ONLY the
    /// head-read half of the re-drive. The DEMAND half keeps running until the last member goes `Active`,
    /// because the reconciler's arm-A `demanded_recently` must stay fresh across the attach round-trip too:
    /// a re-seed that stopped at the resolve lets arm-A lapse mid-bootstrap and the reconciler REAPS the very
    /// realm the login is waiting for (arm-B cannot cover it — a booted-but-unoccupied realm self-reports
    /// `Empty`). Reset to `false` whenever a NEW member joins, so one lost reply cannot wedge that joiner —
    /// the poll simply resumes on the next cadence tick.
    pub(crate) resolved: bool,
}

/// ONE open window as the gateway holds it: where it points (the shard node the gateway
/// resolved as the stating realm's head when it derived the window), what it asks for, which
/// realm is its lawful AUTHOR — the attestation context every inbound row for this window is
/// checked against ([`on_window_row`]) — and, since Slice B, the composer's ingest state (the
/// level ring, the latest bodies, the membership verdict). Closing the window drops the ingest
/// whole: zero windows structurally means zero composer state (§2.6.1 guard 4).
#[derive(Debug, PartialEq)]
pub(crate) struct GatewayWindow {
    /// The shard node this window was opened on — the head the gateway's OWN routing state named
    /// for the author realm (a directory-head answer: `home_shard`/the session's subs/the
    /// window-lane `Realm` head poll). A row from any other sender is forged or stale and is
    /// dropped + counted (fail-closed).
    pub(crate) shard: NodeId,
    pub(crate) scope: WindowScope,
    /// The realm whose statements this window carries — [`WindowScope::Occupants`] ⇒ the realm
    /// the sessions stand in; [`WindowScope::Child`] ⇒ that child's PARENT (the hop author).
    pub(crate) author_realm: RealmId,
    /// Slice B: every attested row this window admitted, decoded once (`docs/design/
    /// window_lane.md` §2.6.2) — the ONLY write path is [`on_window_row`] AFTER attestation
    /// (§2.6.1 guard 2: the composer consumes only admitted rows).
    pub(crate) ingest: window::WindowIngest,
    /// Statements that raced the author's FIRST roster (Slice C1 — load-bearing since the flag
    /// day made bodies pixels): a marker (or a relayed batch) whose vouching roster has not
    /// arrived is PARKED — newest per subject/child, bounded by the roster size by construction —
    /// and re-admitted through the SAME predicates when a level lands. Send-on-change lanes send
    /// ONCE, so a fail-closed drop here would be a permanently invisible body; parking keeps the
    /// drop fail-closed (nothing is served un-attested) without the permanence.
    pub(crate) parked_bodies: BTreeMap<RealmId, (BodyStmt, vd_core::UniverseTick)>,
    /// The parked Q2 relays, newest child-fence wins — drained exactly like the bodies. Since
    /// look horizon slice 3 the parked tuple carries the sealed interior forward too, so a
    /// first-roster race never silently drops a grandchild's picture.
    pub(crate) parked_relays: BTreeMap<RealmId, (Fence, Vec<u8>, Vec<InteriorRelay>)>,
}

impl GatewaySessions {
    /// THE WINDOW LANE's gauge for the admin surface: how many windows this gateway holds open.
    #[must_use]
    pub fn windows_open_count(&self) -> usize {
        self.windows.len()
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_session.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_session.is_empty()
    }
    /// RLM RG-4 — how many demand-spawned home shards are on the RUNTIME routable roster. The direct
    /// process-tier observable for "the dynamic-home resolve fired": `dynamic_shards` is populated in exactly
    /// one place, `claim_dynamic_shard` at the `on_home_realm_head` resolve, so a nonzero count means a login
    /// routed to a spawn-minted node rather than the static `config.shard`.
    #[must_use]
    pub fn dynamic_shard_count(&self) -> usize {
        self.dynamic_shards.len()
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
            | SessionPhase::AwaitingHomeRealm { .. }
            | SessionPhase::AwaitingAttach
            | SessionPhase::SelfFenced => None,
        })
    }

    /// RLM 5f-3d — the session's DYNAMICALLY resolved home shard, or `None` for a STATIC (unarmed) session
    /// / one whose home realm has not become routable yet. The observable half of the D-34 routing field
    /// (the harness/oracles and 5f-4's live route read it to assert WHICH node a session was routed to).
    #[must_use]
    pub fn home_shard_of(&self, session: SessionId) -> Option<NodeId> {
        self.by_session.get(&session).and_then(|s| s.home_shard)
    }

    /// RLM 5f-3d — the session's HOME REALM (the deepest realm containing its stored spawn pose), or `None`
    /// for a STATIC (unarmed) session. Set once at the committed lease and never cleared, so it is
    /// meaningful in every phase from the lease onward.
    #[must_use]
    pub fn home_realm_of(&self, session: SessionId) -> Option<RealmId> {
        self.by_session.get(&session).and_then(|s| s.home_rid)
    }

    /// D-3 / S4 — the local tick of the FRESHEST last-lease-round-trip confirmation (`Session::confirmed_at`)
    /// over sessions that carry a meaningful one: `Active` (re-armed on each affirming `Session`-head recheck
    /// reply) OR `SelfFenced` (FROZEN at the instant the proactive self-fence fired — the gateway's own
    /// evidence it lost the directory path). `None` when no such session exists (only pre-`Active` logins, or
    /// zero sessions). The gateway readiness partition detector — the session-servicing analogue of the
    /// shard's `RealmConfirmedAt`, fed to `vd_node::health::gateway_sessions_live`.
    ///
    /// `SelfFenced` MUST be included: [`self_fence_lapsed_sessions`] fires at the SAME
    /// `local_tick - confirmed > grace` threshold as the readiness de-route and runs EARLIER in the same tick,
    /// so under a TOTAL partition it flips every session `Active → SelfFenced` BEFORE readiness is sampled — an
    /// `Active`-only max would then read `None` and keep a fully-partitioned gateway falsely Ready. A still-fresh
    /// `Active` session dominates the max, so a healthy gateway (or a partial partition where one session still
    /// re-arms) stays Ready; the gateway de-routes only when the freshest over BOTH sets is stale (EVERY session
    /// has frozen — a total directory partition). It re-becomes Ready once a fresh `Active` session attaches or
    /// the frozen `SelfFenced` ghosts clear on connection-end.
    #[must_use]
    pub fn freshest_session_confirmed(&self) -> Option<u64> {
        self.by_session
            .values()
            .filter(|s| {
                matches!(
                    s.phase,
                    SessionPhase::Active { .. } | SessionPhase::SelfFenced
                )
            })
            .map(|s| s.confirmed_at.0)
            .max()
    }

    /// The sessions subscribing to `shard` (the H2 reverse-index read the frame-fan iterates).
    /// Empty when no session subscribes to it. Returns owned ids so the caller can mutate the
    /// outbox while iterating; the set is ≤ S and only the subscribers, never all sessions.
    pub(crate) fn subscribers_of(&self, shard: NodeId) -> Vec<SessionId> {
        self.subscribed_shards
            .get(&shard)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// RLM 5f-3d — THE one entry into the pre-Active home bootstrap (HR3): JOIN `session_id` to its home
    /// realm's [`HomeWait`], creating that wait (with the descended lineage, the cadence anchor and the
    /// representative account) when this session is the first one booting into the realm. Called from the
    /// committed-lease arm in dynamic mode, exactly once per login.
    ///
    /// A joiner ALWAYS clears `resolved`: it has not seen a head reply of its own, so the head poll must
    /// resume even when an earlier member already resolved this realm — otherwise a single lost reply would
    /// hang the joiner until the bounded TTL closed it. `since` is NOT re-anchored (the shared cadence is the
    /// point), and the coord/account of the opening member are kept (the demand is per-realm, not per-member).
    pub(crate) fn begin_home_wait(
        &mut self,
        session_id: SessionId,
        home_rid: RealmId,
        coord: RealmCoord,
        since: TickId,
        account: AccountId,
    ) {
        let wait = self
            .home_bootstraps
            .entry(home_rid)
            .or_insert_with(|| HomeWait {
                coord,
                since,
                account,
                members: BTreeSet::new(),
                resolved: false,
            });
        wait.members.insert(session_id);
        wait.resolved = false;
    }

    /// RLM 5f-3d — THE one exit from the pre-Active home bootstrap (HR3): drop `session_id` from its home
    /// realm's member set and PRUNE the whole realm entry once it empties (no unbounded growth, no orphan
    /// lineage `Vec`). Called on EVERY exit — the `Active` promote, the bootstrap-TTL Close, and `Bye` — so
    /// an index entry can never outlive its sessions. Total, never fallible:
    /// - `home_rid` `None` ⇒ a STATIC session ⇒ no-op (the byte-identical arm every static `Bye` takes);
    /// - the realm absent ⇒ this session already left the bootstrap (the normal arm for a `Bye` AFTER the
    ///   session went `Active`, since `Session::home_rid` is deliberately never cleared) ⇒ no-op.
    pub(crate) fn end_home_wait(&mut self, session_id: SessionId, home_rid: Option<RealmId>) {
        let Some(rid) = home_rid else {
            return;
        };
        let Some(wait) = self.home_bootstraps.get_mut(&rid) else {
            return;
        };
        wait.members.remove(&session_id);
        if wait.members.is_empty() {
            self.home_bootstraps.remove(&rid);
        }
    }

    /// RLM 5f-3d — the cadence ANCHOR a pre-`Active` session's `AwaitingAttach` retry rides, fed to
    /// [`attach_retry_due`]: `Some(since)` of the [`HomeWait`] it belongs to for a DYNAMIC session (so its
    /// attach retry is coalesced onto the SAME backed-off cadence as that realm's re-drive — a mass login
    /// must not re-attach every session every tick at a freshly booted shard), `None` for a STATIC session
    /// (`home_rid` is `None` ⇒ the byte-identical per-tick retry).
    ///
    /// Both `?` arms are total and FAIL-SAFE: `None` means "retry every tick", i.e. the pre-5f-3d behaviour —
    /// never a wedge. The second arm (a `home_rid` with no live wait) cannot occur on the live path — every
    /// path that drops a member either removes the session (`Bye`, the TTL Close) or leaves `AwaitingAttach`
    /// (the `Active` promote) — so it is proven directly by a unit test rather than through a system.
    #[must_use]
    pub(crate) fn attach_anchor(&self, home_rid: Option<RealmId>) -> Option<TickId> {
        let rid = home_rid?;
        Some(self.home_bootstraps.get(&rid)?.since)
    }

    /// RLM 5f-3d — claim `home` for one session on the RUNTIME routable-shard roster (the dispatch half of
    /// the dynamic route): the node becomes node-class-dispatchable as a shard for as long as at least one
    /// session is homed on it.
    pub(crate) fn claim_dynamic_shard(&mut self, home: NodeId) {
        *self.dynamic_shards.entry(home).or_insert(0) += 1;
    }

    /// RLM 5f-3d — release one session's claim on its dynamically resolved home shard. At zero the node
    /// LEAVES the runtime roster, so a long-lived gateway accumulates no entry per ever-spawned realm shard
    /// (the 100K-realm churn leak). `None` — a STATIC session, or a dynamic one that never resolved a home
    /// — is a no-op. All arms (no home / last-out ⇒ remove / others remain) are proven by unit tests.
    pub(crate) fn release_dynamic_shard(&mut self, home: Option<NodeId>) {
        let Some(node) = home else {
            return;
        };
        let remaining = self
            .dynamic_shards
            .get(&node)
            .copied()
            .unwrap_or(0)
            .saturating_sub(1);
        if remaining == 0 {
            self.dynamic_shards.remove(&node);
        } else {
            self.dynamic_shards.insert(node, remaining);
        }
    }

    /// RLM 5f-4 — CLAIM a transfer's CROSSING DEST on the runtime routable roster: THE ADMISSION that makes
    /// a DEMAND-SPAWNED destination realm's shard node-class-dispatchable, so its `SubscriptionReady`, its
    /// `Snapshot` frames and its `RealmSnapshot` frames are CONSUMED instead of falling through to the
    /// client branch and being counted `undecodable` (which left the render authority un-repointed — a
    /// BLIND player after every crossing into a realm the cluster spun up on demand). Driven by the
    /// orchestrator-signed `PrepareSubscribe` (which strictly precedes FreezeSource/CommitAuthority, so the
    /// dest is routable before its first frame) and re-claimed for the home role at `CommitAuthority`.
    ///
    /// GATED to a dest the FROZEN [`GatewayConfig::known_shards`] does not already cover. Two properties
    /// follow, both load-bearing:
    /// - an ALL-STATIC crossing leaves `dynamic_shards` EMPTY, so [`is_routable_shard`] answers purely from
    ///   the config half exactly as it did before this slice (byte-identical dispatch, zero new state);
    /// - the RELEASE side needs NO matching gate, because [`Self::release_dynamic_shard`] of a node that
    ///   holds no entry is a total no-op on the map (`unwrap_or(0).saturating_sub(1) == 0` ⇒ it removes an
    ///   absent key).
    ///
    /// The one-gate design is NOT justified by "a skipped claim leaves nothing to release" — that is FALSE:
    /// a `known_shards` node CAN hold runtime entries, because the LOGIN-time home claim goes through the
    /// RAW, UNGATED [`Self::claim_dynamic_shard`] (`on_home_realm_head`) and `bins/gateway.rs` always puts
    /// the login shard in `known_shards`. The true invariant is a SPLIT by the frozen half:
    /// - for a node IN `known_shards`, its runtime entry is IRRELEVANT to dispatch — [`is_routable_shard`]
    ///   already answers `true` from the config half — so a crossing into it that skips the claim while its
    ///   terminal decrements a login-time entry is unobservable, and can neither un-route the node nor pin it;
    /// - for a node NOT in `known_shards` (the only case dispatch depends on), the gate NEVER skips, so every
    ///   claim site (the raw login claim, this crossing claim, the commit-time home re-claim) is paired by
    ///   count with a release site (the terminal, the defensive replace, the session exit) and the refcount
    ///   is exactly balanced.
    pub(crate) fn claim_crossing_dest(&mut self, config: &GatewayConfig, dest: NodeId) {
        if config.is_known_shard(dest) {
            return; // already dispatchable from the frozen roster — never a spurious dynamic entry
        }
        self.claim_dynamic_shard(dest);
    }

    /// RLM 5f-4 — THE one session-exit roster release (HR3): drop EVERY runtime-roster claim `session`
    /// holds. There are THREE roles (RLM 5f-4e), released here in one place so no exit can leak a refcount
    /// that would pin a demand-spawned node on the roster forever:
    /// 1. its resolved HOME shard (`home_shard`);
    /// 2. an in-flight transfer's CROSSING DEST (`transfer.dest`);
    /// 3. an in-flight transfer's DEMOTING SOURCE home (`transfer.demoting_home`) — the claim
    ///    `CommitAuthority` stashed instead of releasing, so the source stays routable while the client
    ///    still reads it.
    ///
    /// Called from BOTH session exits (`Bye` and the bounded-TTL Close). Releasing all three is what makes
    /// an exit anywhere in the demote tail exact: in the window between `CommitAuthority` and
    /// `ReleaseSubscribe` the dest is claimed TWICE (crossing + new home) and the SOURCE still holds its
    /// stashed one, and the three releases here take every node back to zero. Every arm is total: `None` — a
    /// static session, one with no transfer in flight, or a pre-commit transfer (whose `demoting_home` is
    /// always `None`) — is the no-op arm of [`Self::release_dynamic_shard`]. Role 3 is `None` for the
    /// bounded-TTL exit BY CONSTRUCTION (that deadline is cleared at the `Active` promote and a transfer
    /// requires `Active`), so only `Bye` can observe it `Some`.
    pub(crate) fn release_session_claims(&mut self, session: &Session) {
        self.release_dynamic_shard(session.home_shard);
        self.release_dynamic_shard(session.transfer.as_ref().map(|tp| tp.dest));
        self.release_dynamic_shard(session.transfer.as_ref().and_then(|tp| tp.demoting_home));
    }

    /// THE one open primitive (HR3): allocate a never-reused `sub` id, insert the cold
    /// `SubRecord` (Active), push `SubscriptionOpened` BEFORE publishing the hot table (X1 —
    /// a forwarder can never route a frame for the sub ahead of its opener), publish the sole
    /// `SubTable`, and index the reverse fan-out. The transfer is the FIRST extra caller;
    /// login is the first. Returns the allocated sub id.
    pub(crate) fn open_sub(
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
    ///
    /// **THE AUTHORITY-SUB INVARIANT (the read-plane half of claim-before-release):** never close
    /// the subscription of the node that is CURRENTLY feeding this session
    /// ([`session_target`] — the one definition of "the current authority", HR3). A SAME-NODE
    /// re-home (`source == dest`) is an ordinary saga — task #149 deleted the node-placement
    /// short-circuit precisely so EVERY re-home runs the one cross-node machinery — and on one the
    /// "source sub" IS the live authority sub, because `subs` is keyed by shard `NodeId` (ONE entry
    /// per node). Closing it there unsubscribes the player from their own owner: no snapshots, no
    /// realm feed, no clock, no visible input response — a TOTAL client freeze while the shard
    /// happily simulates them. The runtime roster already survives this case by claiming the dest a
    /// SECOND time at `CommitAuthority` (claim-before-release); this is the same rule for the read
    /// plane. Counted, never silent.
    pub(crate) fn close_sub(
        &mut self,
        session_id: SessionId,
        shard: NodeId,
        config: &GatewayConfig,
        outbox: &mut OutboundBox,
        stats: &mut GatewayStats,
    ) {
        let Some(session) = self.by_session.get_mut(&session_id) else {
            return;
        };
        if session_target(session, config) == shard {
            stats.sub_close_refused_authority += 1;
            tracing::warn!(
                shard = shard.0,
                "refused to close the sub of the session's CURRENT authority (same-node re-home) \
                 — closing it would freeze the client on its own owner"
            );
            return;
        }
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
    pub(crate) fn sweep_draining(&mut self) {
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
pub(crate) struct SessionMint(pub(crate) SplitMix64);
