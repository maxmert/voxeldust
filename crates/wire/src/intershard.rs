//! THE closed shard↔shard / shard↔orchestrator taxonomy (HR1 taxonomy 1 of 2;
//! `docs/design/sealed_shards.md` §0–§1). One reviewed file.
//!
//! Every byte crossing a shard-to-shard or shard-to-orchestrator boundary is exactly
//! one [`InterShardFlow`] arm. There is NO `Raw`, NO `Other`, NO `EcsSync` — to
//! express a new cross-shard flow you MUST add a variant here, under review.
//!
//! Arms freeze INCREMENTALLY with their first consumer (the closed-set guarantee is
//! the per-release conformance test below, not a day-one empty freeze). LANDED (16 arms):
//! - P0: `Ghost`, `Transfer` (Durable class), `Directory`.
//! - P2 route swap: `Saga` (saga→gateway transfer commands) + `SagaAck` (gateway→saga acks).
//! - P2 transfer machinery (1d.1/1d.5b): `DirectoryReply`, `FlushSource`, `TransferAck`, `Demote`,
//!   `Promote` (the saga-pushed ordered demote-before-promote + the entity-state crossing/ack family).
//! - P3 transient (D-7): `Transfer` transient batches + `TransientRelease`/`TransientDrop`/
//!   `ReleaseComplete`/`TransientAbandon` (the structural drop-before-promote handoff + the dead-DEST abandon).
//! - P3 permanent-kill recovery (D-37): `ReHome` (the forward re-home adopt — a DEDICATED arm, never a
//!   `Promote` reuse).
//! - R-6d3c (D-6 #1 NEVER-restart closure): `TransientDiscard` (orch→dest discard-poison — the source
//!   died in `BatchHandoff::AwaitAdopt` pre-adopt, so a late-replayed `Arriving` copy is removed + its
//!   adopt poisoned; reuses `TransientHandoff`).
//! - CA-1 S3: `ReSolicitBatch` (orch→source AwaitAdopt liveness probe; reuses `TransientHandoff`).
//! - Slice 3c (spatial transfer-trigger, INERT — planted here, consumer routes land later): `CrossingRequest`
//!   / `TransientCrossingRequest` (shard→orch: a durable/transient entity crossed a realm boundary — key
//!   idempotency on the subject/src-realm `Fence`, NO `TransferId` at emit), `TransientCrossingGrant`
//!   (orch→source: the resolved dest, its four fields mirror `TransientStatus::Crossing`), `CrossingAborted`
//!   (orch→source: the resolve saga aborted pre-CAS — clear the source's `RequestInFlight` latch).
//!
//! RESERVED (variant lands with its consumer): `BlockEdit` (P6), `Coupling` `EffectFree` ports (P8),
//! `Signal` (P9 cross-shard functional-block signals).
//!
//! Effect classes (the G-SEALED invariant, enforced by `effect_class` + its test):
//! - SIDE-EFFECTING arms carry `(TransferId, step_id)` idempotency and are ack-driven.
//! - FIRE-AND-FORGET arms are effect-free or idempotently re-derivable: they may
//!   NEVER carry a transfer trigger or authority-gating discrete state.

use serde::{Deserialize, Serialize};
use vd_core::entity_kind::DurabilityClass;
use vd_core::pose::{RealmId, StampedPose};
use vd_core::realm_coord::RealmCoord;
use vd_core::{
    AccountId, EntityId, EpochId, Fence, NodeId, SessionId, TickId, TransferId, UniverseTick,
};

use crate::seams::directory::{DirectoryKey, DirectoryOp, DirectoryReply};
use crate::seams::transfer_control::{TransferControl, TransferControlAck};

/// The two `Transfer`-machinery step ids that sit ABOVE the 0–6 `TransferControl` route-swap
/// phases (`seams::transfer_control::step_id`). They key the `(transfer, step_id)` idempotency of
/// the entity-STATE half of a transfer — disjoint from the route-swap phases so a state step can
/// never alias a route-swap phase in any `applied_steps` journal (asserted in tests).
///
/// - [`FLUSH_SOURCE_STEP`] (7): the orchestrator→source pose-flush request AND its source→orch
///   `TransferAck::SourceFlushed` reply (request + ack share a phase, like FreezeSource/SourceFrozen).
/// - [`STUB_CROSSING_STEP`] (8): the orchestrator→dest `StubCrossing` envelope AND its dest→orch
///   `TransferAck::Accepted` reply.
/// - [`DEMOTE_STEP`] (9) / [`PROMOTE_STEP`] (10): the Slice-1d.5b orchestrator→source `Demote` /
///   orchestrator→dest `Promote` ordered-demote commands' idempotency steps (the source/dest journal
///   a redelivery as a no-op at the same step). Their acks ride `SagaAck` (1d.4a `DemoteAck`/`PromoteAck`).
pub const FLUSH_SOURCE_STEP: u32 = 7;
/// See [`FLUSH_SOURCE_STEP`].
pub const STUB_CROSSING_STEP: u32 = 8;
/// See [`FLUSH_SOURCE_STEP`] — the saga-pushed ordered-demote (1d.5b).
pub const DEMOTE_STEP: u32 = 9;
/// See [`FLUSH_SOURCE_STEP`] — the saga-pushed ordered-promote (1d.5b).
pub const PROMOTE_STEP: u32 = 10;
/// The TRANSIENT-class batch handover phases (D-7a), disjoint from the durable steps above and the
/// 0–6 route-swap phases (so a transient step can never alias a durable step in any `(transfer,
/// step)` journal). A whole batch shares ONE `(transfer, step)` key (the batch's `transfer` is its
/// `BatchId`), so the go-token is amortized to one per batch (G-TIER), never per item.
/// - [`TRANSIENT_BATCH_STEP`] (11): the SOURCE→dest `TransientBatch` envelope, its dest→orch
///   `TransferAck::BatchAdopted` reply, AND the orchestrator's batched `TransientGo` go-token
///   (request + ack + commit share a phase, like FreezeSource/SourceFrozen).
/// - [`TRANSIENT_RELEASE_STEP`] (13, D-7b): the orchestrator→SOURCE `TransientRelease` (phase 1 of the
///   STRUCTURAL drop-before-promote — the source flips `Held→Departing`, goes uncounted, and acks
///   `TransferAck::DropApplied`). Mirrors the durable `Demote`/`DemoteAck`.
/// - [`TRANSIENT_DROP_STEP`] (12, D-7b NARROWED): the orchestrator→DEST `TransientDrop` = PROMOTE
///   (`Arriving→Held`), reachable ONLY after the source's `DropApplied` — and the dest's own
///   promote-confirm `DropApplied`, which drives the orchestrator's `ReleaseComplete` to the source
///   (retiring the retained `Departing` copy). The holder set transits `{S}→{}→{D}`, never `{S,D}`.
pub const TRANSIENT_BATCH_STEP: u32 = 11;
/// See [`TRANSIENT_BATCH_STEP`] — the dest PROMOTE phase (`TransientDrop`) + the dest promote-confirm.
pub const TRANSIENT_DROP_STEP: u32 = 12;
/// See [`TRANSIENT_BATCH_STEP`] — the SOURCE release phase (`TransientRelease` + its `DropApplied` ack),
/// D-7b's structural drop-before-promote.
pub const TRANSIENT_RELEASE_STEP: u32 = 13;
/// See [`TRANSIENT_BATCH_STEP`] — the SOURCE retire-complete phase (D-7d): `on_release_complete` now
/// acks `TransferAck::DropApplied`(`TRANSIENT_COMPLETE_STEP`) after dropping the retained `Departing`
/// copy, so the saga's `BatchHandoff` tail reaches `Done` (`SourceRetired`) instead of running ownerless.
pub const TRANSIENT_COMPLETE_STEP: u32 = 14;
/// See [`TRANSIENT_BATCH_STEP`] — the D-7d dead-DEST ABANDON phase: the orchestrator→SOURCE
/// `TransientAbandon` + the source's `DropApplied`(`TRANSIENT_ABANDON_STEP`) ack. Journaled idempotent.
pub const TRANSIENT_ABANDON_STEP: u32 = 15;
/// D-37 forward re-home: the orchestrator→TARGET [`ReHome`](InterShardFlow::ReHome) adopt command + the
/// target's `PromoteAck` (the target becomes the Owned authority). Journaled idempotent by
/// `(transfer, RE_HOME_STEP)` at the target so a redelivery re-acks without re-adopting.
pub const RE_HOME_STEP: u32 = 16;
/// D-6 #1 NEVER-restart closure (R-6d3c): the orchestrator→DEST discard command — the source died
/// in `BatchHandoff::AwaitAdopt` (PRE-adopt) so the batch is being counted lost, but a late outbox
/// replay could still insert `Arriving{batch}` at the dest; the discard REMOVES any such item AND
/// poisons [`TRANSIENT_BATCH_STEP`] so a later replayed adopt is `AlreadyApplied` (never re-inserts an
/// orphan). Journaled idempotent by `(transfer, TRANSIENT_DISCARD_STEP)`. Disjoint from 0–6 route-swap +
/// 7–16 state/transient/rehome steps (asserted in tests); `RE_SOLICIT_STEP` = 18 follows.
pub const TRANSIENT_DISCARD_STEP: u32 = 17;
/// CA-1 S3 (the over-discard DETECTION trigger): the orchestrator→SOURCE `ReSolicitBatch` liveness probe
/// emitted every `BatchHandoff::AwaitAdopt` Timeout. It carries no state — its sole purpose is to give the
/// AwaitAdopt phase an orch→source egress so a DEAD source's send FAILS (`NodeUnreachable`) and
/// `is_confirmed_dead(source)` becomes reachable (the `SourceUnreachablePreAdopt` discard was inert
/// without it). A LIVE source handles it as a counted no-op. 18 is the next free id.
pub const RE_SOLICIT_STEP: u32 = 18;

/// The control-plane schema version stamped on a [`TransferEnvelope`] (postcard, additive under
/// minor negotiation). ONE home — never an inline literal at an emit site (the per-kind
/// version-floor handshake that reads it lands with the TLV blob, P-later).
pub const TRANSFER_SCHEMA_VERSION: u16 = 1;

/// The closed taxonomy. Compiler-forced exhaustive handling everywhere.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InterShardFlow {
    /// Replication into a neighbor's overlap band (spawn/delta/despawn).
    Ghost(GhostFlow),
    /// Authority handoff — both durability classes ride the same arm (HR3).
    Transfer(TransferEnvelope),
    /// Orchestrator-authoritative control ONLY (lease/CAS/clock — never spatial
    /// interest, which is shard-local by design).
    Directory(DirectoryOp),
    /// Saga → gateway transfer commands (P2): the route-swap saga drives the gateway
    /// through the phased, separately-acked command vocabulary (the gateway never
    /// decides a transfer). Side-effecting + ack-driven by `(transfer, phase)`.
    Saga(TransferControl),
    /// Gateway → saga acks for the `Saga` arm (one per command phase).
    SagaAck(TransferControlAck),
    /// Orchestrator → requester REPLY to a `Directory` op (P2): the standing head or the
    /// CAS outcome. Carried as its OWN arm (not bare bytes) so the requester can decode
    /// `InterShardFlow` ONCE and dispatch by variant — a `Saga(TransferControl)` and a
    /// directory reply both ride orchestrator→peer on `MsgClass::Saga`, and postcard is
    /// non-self-describing, so without this they would mis-decode into each other.
    DirectoryReply(DirectoryReply),
    /// Orchestrator → SOURCE shard (Slice 1d.1): "ship the subject's authoritative pose so the
    /// dest can adopt it." Read-only at the source (it ships a copy; authority is unchanged), so
    /// it needs NO compensator — the abort path's `ThawSource` is the only stateful source undo.
    /// Side-effecting + ack-driven by `(transfer, FLUSH_SOURCE_STEP)`; the reply is
    /// `TransferAck::SourceFlushed`.
    FlushSource(FlushSource),
    /// SHARD → orchestrator ack of a `Transfer`-machinery step (Slice 1d.1 gives the long-RESERVED
    /// `TransferAck` its first consumer): the SOURCE's `SourceFlushed` pose reply (phase
    /// `FLUSH_SOURCE_STEP`) and the DEST's `Accepted`/`Rejected` of a `StubCrossing` envelope
    /// (phase `STUB_CROSSING_STEP`). Distinct from `SagaAck` (the gateway↔saga route-swap
    /// vocabulary): this is the entity-STATE ack family, keyed by `(transfer_id, step_id)`.
    TransferAck(TransferAck),
    /// Orchestrator → SOURCE shard ORDERED demote command (Slice 1d.5b): the source demotes the
    /// subject `Owned→Frozen→Ghost` at the post-CAS `new_owner_fence` and acks `DemoteAck` (via
    /// `SagaAck`). The saga-pushed, FENCE-enforced demote that REPLACES the 1c.8 cooperative poll —
    /// the `FlushSource` precedent (a dedicated saga→shard arm, not `TransferControl`). Side-effecting,
    /// ack-driven by `(transfer, DEMOTE_STEP)`. APPENDED at the end (Ghost is discriminant 0; this
    /// preserves every existing postcard discriminant).
    Demote(DemoteCmd),
    /// Orchestrator → DEST shard ORDERED promote command (Slice 1d.5b): the dest flips the subject
    /// `Ghost→Owned` at `new_fence` and acks `PromoteAck`. Reachable in the saga ONLY after
    /// `DemoteAck` (demote-before-promote) — REPLACES the 1c.8 autonomous adopt-flip. Side-effecting,
    /// ack-driven by `(transfer, PROMOTE_STEP)`.
    Promote(PromoteCmd),
    /// Orchestrator → SOURCE shard: phase 1 of the TRANSIENT structural drop-before-promote (D-7b) —
    /// the source flips its `Held{outbound}` batch items to the uncounted+unrendered `Departing` tier
    /// and acks `TransferAck::DropApplied` (the transient twin of `Demote`/`DemoteAck`). Emitted only
    /// after the dest's `BatchAdopted`. Side-effecting, ack-driven by `(transfer, TRANSIENT_RELEASE_STEP)`.
    TransientRelease(TransientHandoff),
    /// Orchestrator → DEST shard: the TRANSIENT PROMOTE (D-7b NARROWED from the D-7a broadcast) — the
    /// dest flips its `Arriving` batch items to authoritative `Held`. Reachable ONLY after the source's
    /// `DropApplied`, so the source is already uncounted: the holder set is never `{S,D}`. Side-effecting,
    /// ack-driven by `(transfer, TRANSIENT_DROP_STEP)` (the dest's promote-confirm `DropApplied`).
    TransientDrop(TransientHandoff),
    /// Orchestrator → SOURCE shard: phase 3 — retire the retained `Departing` copy after the dest's
    /// promote-confirm (a dest-crash before this lets the promote re-drive against the still-extant
    /// source copy). Side-effecting, ack-driven by `(transfer, TRANSIENT_RELEASE_STEP)`. APPENDED
    /// (preserves every existing postcard discriminant).
    ReleaseComplete(TransientHandoff),
    /// Orchestrator → SOURCE shard: D-7d dead-DEST resolution — the dest died mid-handoff so the only
    /// promote target is gone; the source ABANDONS the batch's retained `Departing`/`Held` items as an
    /// accounted loss-within-budget (a PROPER new action, semantically distinct from `ReleaseComplete`'s
    /// clean no-loss retire). Side-effecting, ack-driven by `(transfer, TRANSIENT_ABANDON_STEP)`.
    /// APPENDED (preserves every existing postcard discriminant).
    TransientAbandon(TransientHandoff),
    /// Orchestrator → re-home TARGET shard (D-37 forward re-home): ADOPT the subject as Owned from the
    /// carried [`ReHomeState`] after a permanent kill of its committed owner re-homed it here. A
    /// DEDICATED arm, NOT `Promote` — the re-home adopt RECONSTRUCTS state from a payload at a fresh
    /// target (no pre-existing ghost to flip), so reusing `Promote` would repurpose a Ghost→Owned command
    /// for a create-from-state operation. Side-effecting, ack-driven by `(transfer, RE_HOME_STEP)`; the
    /// target acks `PromoteAck` (it is now the Owned authority). APPENDED (preserves every existing
    /// postcard discriminant).
    ReHome(ReHomeCmd),
    /// Orchestrator → DEST shard (R-6d3c): the source died in `BatchHandoff::AwaitAdopt` (PRE-adopt) so
    /// the batch is counted lost; REMOVE any `Arriving{batch==transfer}` item AND poison `(transfer,
    /// TRANSIENT_BATCH_STEP)` so a late outbox replay's adopt is `AlreadyApplied` (never re-inserts an
    /// orphan). Reuses [`TransientHandoff`] (shares the shape + one classification arm with the other
    /// transient handoff commands, DRY). Side-effecting, ack-FREE (the resolving saga is terminal);
    /// idempotent by `(transfer, TRANSIENT_DISCARD_STEP)`. Classified [`FlowDurabilityClass::ReDriven`]
    /// because it is orchestrator-EMITTED (it does NOT grow the producer-less outbox set / trip the
    /// push-`Ephemeral` debug_assert) — NOT because `scan_deadlines` re-drives it: it is a FIRE-ONCE
    /// terminal egress, its lost-delivery residual is covered by the CA-1/L5 re-solicit (DEFERRED.md),
    /// NOT by a re-driver. APPENDED (preserves every existing postcard discriminant).
    TransientDiscard(TransientHandoff),
    /// Orchestrator → SOURCE shard (CA-1 S3): the AwaitAdopt liveness PROBE. Emitted every
    /// `BatchHandoff::AwaitAdopt` Timeout (the phase's FIRST orch→source egress — before this, AwaitAdopt
    /// was silent, so `is_confirmed_dead(source)` was unreachable and the pre-adopt discard was inert). A
    /// DEAD source ⇒ the send fails ⇒ `NodeUnreachable` ⇒ confirm-dead ⇒ (past budget, `!dest_adopted`) the
    /// `SourceUnreachablePreAdopt` discard fires; a LIVE source handles it as a COUNTED NO-OP (no re-emit —
    /// its regular lease-renewal inbound is what clears stale unreachable evidence). Carries no state; reuses
    /// [`TransientHandoff`] (shares the shape + one classification arm with the transient handoff commands,
    /// DRY). Side-effecting, ack-FREE (the probe's signal is the SEND outcome, not a reply); idempotent by
    /// `(transfer, RE_SOLICIT_STEP)`. Classified [`FlowDurabilityClass::ReDriven`]: orchestrator-EMITTED and
    /// re-driven by `scan_deadlines` on every AwaitAdopt Timeout (it does NOT grow the producer-less outbox
    /// set / trip the push-`Ephemeral` debug_assert). APPENDED (preserves every existing postcard discriminant).
    ReSolicitBatch(TransientHandoff),
    /// SHARD → orchestrator (Slice 3c, spatial transfer-trigger): a DURABLE entity crossed a realm boundary;
    /// the orchestrator resolves the dest for `to_realm` and STARTS the durable transfer saga (the route-swap
    /// plus the FlushSource/StubCrossing/Demote/Promote machinery). INERT this slice — the arm is planted; the
    /// explicit orch route lands in a later slice (the demux wildcards catch it until then). Carries NO
    /// `TransferId` (the orchestrator mints it on start), so `effect_class` keys idempotency on the subject's
    /// `subject_fence` (`FencedKey`) — a redelivered request for the same fenced crossing is a no-op, never a
    /// second saga. Side-effecting. APPENDED (preserves every existing postcard discriminant).
    CrossingRequest(CrossingRequest),
    /// SHARD → orchestrator (Slice 3c, spatial transfer-trigger): a TRANSIENT crossed a realm boundary; the
    /// orchestrator resolves the dest for `to_realm` and GRANTS it back (`TransientCrossingGrant` below), which
    /// the source flips its pending `Crossing` item to. INERT this slice (planted; the explicit orch route
    /// lands later — the demux wildcards catch it). Carries NO `TransferId` at emit (the orchestrator mints the
    /// batch id), so `effect_class` keys idempotency on `src_realm_fence` (`FencedKey`), not `(transfer, step)`.
    /// Side-effecting. APPENDED (preserves every existing postcard discriminant).
    TransientCrossingRequest(TransientCrossingRequest),
    /// ORCHESTRATOR → SOURCE shard (Slice 3c): the GRANT carrying the resolved dest for a
    /// `TransientCrossingRequest`. Its four fields EXACTLY match `TransientStatus::Crossing`
    /// (`sim::stub`), so the source flip is a straight field assign (no repack). INERT this slice
    /// (planted; the source consume route lands later — the demux wildcards catch it). Carries the minted
    /// `batch` `TransferId`, so `effect_class` keys idempotency on `(batch, TRANSIENT_BATCH_STEP)` — the same
    /// phase the batch's whole handoff amortizes to. Side-effecting. APPENDED (preserves every existing
    /// postcard discriminant).
    TransientCrossingGrant(TransientCrossingGrant),
    /// ORCHESTRATOR → SOURCE shard (Slice 3c): the crossing's resolve/start saga ABORTED pre-CAS (e.g. the
    /// dest is unavailable), so the source CLEARS the subject's `RequestInFlight` latch — the entity is free
    /// to re-cross next tick. The PROPER fix (a positive orchestrator signal), never a bounded TTL window that
    /// guesses when to retry. INERT this slice (planted; the source consume route lands later — the demux
    /// wildcards catch it). Carries the aborted `transfer` `TransferId`, so `effect_class` keys idempotency on
    /// `(transfer, TRANSIENT_BATCH_STEP)`. Side-effecting. APPENDED (preserves every existing postcard
    /// discriminant).
    CrossingAborted(CrossingAborted),
    /// SOURCE shard → ORCHESTRATOR (Slice 3f): the latch-clear CONFIRM for a `CrossingAborted` — the source
    /// consumed the abort and cleared (or found already-cleared) the subject's `RequestInFlight` latch, so the
    /// orchestrator drops the DURABLE `pending_abort_replies` entry and stops re-emitting. This closes the
    /// Option-B ack-gate: the orchestrator keeps the aborted-crossing reply alive (re-driven by
    /// `scan_deadlines`, crash-durable via the saga store) until THIS ack lands — so a lost `CrossingAborted`
    /// can never strand the entity. Reuses the `CrossingAborted` `{subject, transfer}` payload; `effect_class`
    /// keys idempotency on the SAME `(transfer, TRANSIENT_BATCH_STEP)` as the abort it answers (request + ack
    /// share one journal). Side-effecting. APPENDED (preserves every existing postcard discriminant).
    CrossingAbortedAck(CrossingAborted),
    /// Realm lifecycle (RLM Step 1): a level-triggered spin-up / keep-alive / empty / teardown
    /// demand between a realm and the orchestrator. Re-asserted every tick, so a dropped verb
    /// self-heals ⇒ `ReDriven`. Authority-gating discrete state keyed on `parent_fence` (the
    /// parent's authority proof), like `CrossingRequest` ⇒ `SideEffecting{FencedKey}`. APPENDED
    /// (preserves every existing postcard discriminant).
    RealmDemand(RealmDemand),
    /// SHARD → its GATEWAY (RLM reactive greeting): a demand-spawned shard's [`NodeId`] is minted at
    /// spawn, so it can NEVER be in the gateway's boot-time peer book; the gateway can only REPLY over a
    /// connection the shard itself opened (the io-prod mesh records the return connection on the FIRST
    /// reliable frame of an accepted stream, then serves both reliable and unreliable traffic over it). This
    /// arm IS that first reliable frame — a pure REACHABILITY primitive, nothing else. It carries NO realm
    /// authority (which shard owns which realm is the orchestrator's fence-CAS directory head, consumed at
    /// the gateway's home-realm resolve; a shard self-asserting ownership here would be a SECOND, unfenced
    /// authority path), so the gateway records only a counter + a log line — never a routing claim. Sent
    /// LEVEL-TRIGGERED on the shard's own silence cadence (only while it has NOT recently heard from the
    /// gateway), so an active realm is silent while a gateway restart / dropped connection self-heals within
    /// one cadence — a `ReDriven` (the shard re-asserts) `FireAndForget` (carries no idempotency-keyed
    /// effect) flow. APPENDED (preserves every existing postcard discriminant).
    ShardPresence(ShardPresence),
    /// CHILD → PARENT shard (VU AoI S2a — the occupant-position UP-flow): relay where an observer IS so a
    /// sealed-off ancestor can cull ITS OWN children (the observer's siblings/cousins) against that position.
    /// The render-AoI is the union of per-level culls up the containment chain; positions flow DOWN today
    /// (parents author children's placements), and THIS is the upward counterpart. A pure re-derivable
    /// POSITION HINT — it carries NO fence and mutates nothing at the parent (the parent authorizes any
    /// `SpinUp` it emits with its OWN realm fence): `FireAndForget` + `Unreliable` (a 20 Hz latest-wins
    /// datagram; a lost hint self-heals next tick, like `GhostFlow::Delta`). A DEDICATED infra arm, NOT the
    /// P9-reserved generic `Signal` bus. APPENDED (preserves every existing postcard discriminant).
    OccupantInterest(OccupantInterest),
    /// VU AoI S2c — a PARENT's reflection of one proxy occupant's CURRENT in-range SIBLING set, sent DOWN to
    /// the home shard that relays the occupant (the up-relay's own return address, so the parent never learns
    /// the client's connection). The home shard reconciles it into the client render stream it already owns.
    /// LEVEL-triggered (the FULL current set, not an add/remove edge): the addressed home changes UNDER a
    /// crossing (a cross-node re-home), so an edge sent there would be delivered-then-dropped; a full set lets
    /// whoever draws the client's world reconcile to the truth after a hand-off / crash / lost message — the
    /// same self-healing shape as `RealmDemand`. Carries ONLY public parent-authored `RealmShape` geometry —
    /// NO NodeId / gateway / session (HR1: the client-connection detail never leaves the home shard).
    /// `FireAndForget` (no fence, mutates no sim state — render bookkeeping only) + `ReDriven` (reliable; a
    /// lost set self-heals on the next change, re-sent from the parent's RAM — no durable outbox). APPENDED.
    ProxySceneSet(ProxySceneSet),
}

/// How an arm participates in side effects: the machine-checkable half of HR1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectClass {
    /// Mutates durable/authoritative state: must carry an idempotency key and be
    /// delivered via acked retry.
    SideEffecting { idempotency: IdempotencyKey },
    /// Latest-wins / re-derivable; loss is tolerated by design. May NEVER carry a
    /// transfer trigger or authority-gating discrete state.
    FireAndForget,
}

/// The idempotency mechanisms a side-effecting arm may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdempotencyKey {
    /// `(TransferId, step_id)` — journaled in `applied_steps` before any effect. THE shared
    /// dedup KEY across every altitude (HR3 one machinery, many stores): the gateway's
    /// per-session RAM journal keys its map on `(transfer, step_id)` (Slice 1c.2); the dest
    /// shard's durable redb `applied_steps` table keys on the same (Slice 1d); cross-shard
    /// durable signals reuse it with a correlation id as the `TransferId` (P9). The store
    /// differs per altitude; the KEY and the consult-before-effect / record-after-effect
    /// discipline do not.
    TransferStep { transfer: TransferId, step_id: u32 },
    /// Idempotent by fence comparison (lease grants/revokes).
    FencedKey { fence: Fence },
    /// A fence CAS within a saga (commit/abort — mutually exclusive by rule 3).
    FencedCas {
        expected: Fence,
        transfer: TransferId,
    },
}

/// Whether an [`InterShardFlow`] arm's SENDER re-drives it after a crash — the axis that decides if a
/// reliable one-shot needs the R-6d durable outbox. ORTHOGONAL to [`EffectClass`] (idempotency) and to the
/// per-entity `DurabilityClass` (Durable-vs-Transient KIND, HR2); this is a per-FLOW recovery property.
/// Classified for every arm by [`InterShardFlow::durability_class`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowDurabilityClass {
    /// A PRODUCER re-drives it on a crash (the orchestrator saga's `scan_deadlines` re-emits the step), so
    /// the in-RAM `ReliableLaneSender.retry` buffer + reconnect replay suffices — no durable outbox needed.
    ReDriven,
    /// A reliable ONE-SHOT with NO re-driver: if the SOURCE process crashes before it is acked, the RAM
    /// retry is lost and nothing re-emits it (D-6 #1). Its `push_flow` site MUST carry `Durability::Retained`
    /// so the R-6d durable outbox mirrors + replays it. The forgotten-marker trap this whole classifier exists
    /// to convert into a build failure.
    ProducerLessReliable,
    /// A latest-wins UNRELIABLE datagram (ghost pose feed): loss is correct by design, never durable.
    Unreliable,
}

impl InterShardFlow {
    /// Classify an arm. EXHAUSTIVE by construction — adding a variant without
    /// classifying it does not compile, which is the G-SEALED conformance hook.
    #[must_use]
    pub fn effect_class(&self) -> EffectClass {
        match self {
            // Ghost spawn/despawn are reliable+acked but idempotently re-derivable
            // from band geometry; deltas are pure latest-wins mirrors.
            InterShardFlow::Ghost(_) => EffectClass::FireAndForget,
            InterShardFlow::Transfer(env) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: env.transfer_id,
                    step_id: env.step_id,
                },
            },
            InterShardFlow::Directory(op) => op.effect_class(),
            // Saga commands + acks are side-effecting + ack-driven; `(transfer, phase)`
            // is their `applied_steps` idempotency key (the gateway/saga no-op a
            // re-delivery at the same phase).
            InterShardFlow::Saga(cmd) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: cmd.transfer(),
                    step_id: cmd.step_id(),
                },
            },
            InterShardFlow::SagaAck(ack) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: ack.transfer(),
                    step_id: ack.step_id(),
                },
            },
            // A reply mutates NOTHING at the receiver: it reports the standing head / CAS
            // outcome, which the receiver treats as a HINT and pulls through to the
            // authority-of-record directory (a lost reply is recovered by re-reading the
            // head — the directory REQUEST it answers carries the real idempotency). So it
            // is re-derivable + loss-tolerated = FireAndForget; it carries no transfer
            // trigger and is never the authority of record (the CAS at the directory is).
            InterShardFlow::DirectoryReply(_) => EffectClass::FireAndForget,
            // The pose-flush request + the entity-state ack family are side-effecting + ack-driven
            // by the universal `(transfer, step_id)` key (the source/dest journal a redelivery as
            // a no-op at the same step).
            InterShardFlow::FlushSource(f) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: f.transfer,
                    step_id: f.step_id,
                },
            },
            InterShardFlow::TransferAck(ack) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: ack.transfer_id(),
                    step_id: ack.step_id(),
                },
            },
            // The saga-pushed ordered demote/promote (1d.5b): side-effecting authority moves at the
            // source/dest, journaled by `(transfer, DEMOTE_STEP|PROMOTE_STEP)` (consult-before-effect).
            InterShardFlow::Demote(cmd) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: cmd.transfer,
                    step_id: cmd.step_id,
                },
            },
            InterShardFlow::Promote(cmd) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: cmd.transfer,
                    step_id: cmd.step_id,
                },
            },
            // The transient structural drop-before-promote handoff commands (D-7b) + the D-7d dead-DEST
            // ABANDON: side-effecting authority moves at the source/dest, journaled by `(transfer,
            // step_id)` (idempotent by local held-status — the batched twin of the demote/promote
            // idempotency). All FIVE share the `TransientHandoff` shape and one classification arm (DRY):
            // the R-6d3c `TransientDiscard` (orch→dest discard-poison) joins the group unchanged.
            InterShardFlow::TransientRelease(h)
            | InterShardFlow::TransientDrop(h)
            | InterShardFlow::ReleaseComplete(h)
            | InterShardFlow::TransientAbandon(h)
            | InterShardFlow::TransientDiscard(h)
            // CA-1 S3: the AwaitAdopt liveness probe joins the group unchanged — correlated by
            // `(transfer, RE_SOLICIT_STEP)`, delivered reliably (its send-outcome IS the signal).
            | InterShardFlow::ReSolicitBatch(h) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: h.transfer,
                    step_id: h.step_id,
                },
            },
            // D-37 forward re-home adopt: side-effecting (the target adopts the subject as Owned),
            // journaled by `(transfer, RE_HOME_STEP)` — the SAME idempotency discipline as `Promote`, on a
            // dedicated arm. An adopt-from-state mutation, never fire-and-forget.
            InterShardFlow::ReHome(cmd) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: cmd.transfer,
                    step_id: cmd.step_id,
                },
            },
            // Slice 3c spatial transfer-trigger: the two crossing REQUESTS carry NO TransferId at emit (the
            // orchestrator mints it when it starts the saga), so they key idempotency on the crossing's fence
            // (`FencedKey`) — a redelivered request for the same fenced crossing is a no-op, never a second
            // saga. The GRANT/ABORTED carry a real minted TransferId, so they key on `(transfer,
            // TRANSIENT_BATCH_STEP)` like the rest of the batch's amortized handoff.
            InterShardFlow::CrossingRequest(r) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedKey {
                    fence: r.subject_fence,
                },
            },
            InterShardFlow::TransientCrossingRequest(r) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedKey {
                    fence: r.src_realm_fence,
                },
            },
            InterShardFlow::TransientCrossingGrant(g) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: g.batch,
                    step_id: TRANSIENT_BATCH_STEP,
                },
            },
            InterShardFlow::CrossingAborted(a) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: a.transfer,
                    step_id: TRANSIENT_BATCH_STEP,
                },
            },
            // The ack answers a specific `CrossingAborted`, so it keys idempotency on the SAME
            // `(transfer, TRANSIENT_BATCH_STEP)` — request + ack share one journal (a redelivered ack for an
            // already-dropped `pending_abort_replies` entry is a covered no-op at the consumer).
            InterShardFlow::CrossingAbortedAck(a) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: a.transfer,
                    step_id: TRANSIENT_BATCH_STEP,
                },
            },
            // Authority-gating discrete state (drives a spawn/kill), keyed on the parent's authority
            // fence — a redelivered demand for the same fenced parent is a covered no-op.
            InterShardFlow::RealmDemand(d) => EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedKey {
                    fence: d.parent_fence,
                },
            },
            // The reactive greeting mutates NOTHING at the gateway (a counter + a log line) and carries no
            // transfer trigger or authority-gating state — it is re-derivable (the shard re-greets) and
            // loss-tolerant (a dropped greeting is re-sent next silence cadence) ⇒ FireAndForget.
            InterShardFlow::ShardPresence(_) => EffectClass::FireAndForget,
            // VU AoI S2a: the occupant-position up-flow mutates NOTHING at the parent (it folds a proxy
            // occupant into its AoI cull; any `SpinUp` it emits is authorized by its OWN realm fence) and
            // carries no transfer trigger or authority-gating state — a pure re-derivable position hint,
            // loss-tolerant (self-heals next tick) ⇒ FireAndForget.
            InterShardFlow::OccupantInterest(_) => EffectClass::FireAndForget,
            // VU AoI S2c — a public-geometry render reflection; no fence, no transfer trigger, mutates no sim
            // state at the home (render bookkeeping only). A re-delivered set is idempotent (full-set reconcile).
            InterShardFlow::ProxySceneSet(_) => EffectClass::FireAndForget,
        }
    }

    /// Classify an arm's crash-recovery property (R-6d §7). EXHAUSTIVE by construction at EVERY nesting
    /// level — NO `_` wildcard anywhere — so adding a variant (or a `GhostFlow`/`TransitionPayload` variant)
    /// does not compile until it is classified. This is the G-SEALED discipline of [`effect_class`] applied
    /// to durability: a `ProducerLessReliable` flow whose `push_flow` site forgets `Durability::Retained` is
    /// silently lost on a source crash, and the only defense is that a NEW producer-less flow cannot be added
    /// without a compiler-forced decision here PLUS the marker test (`intershard_closed.rs`) that pins each
    /// producer-less arm's push site to `Retained`. Sibling of, not reusable from, `effect_class` (which is
    /// payload-blind on `Transfer` and coarse on `Ghost`).
    #[must_use]
    pub fn durability_class(&self) -> FlowDurabilityClass {
        match self {
            // Ghost lifecycle: Spawn re-derives from band geometry (the feed re-spawns); Delta is the 20Hz
            // latest-wins datagram; Despawn is a band-exit ONE-SHOT — a direct shard↔shard emit with NO saga
            // re-driver, so it needs the durable outbox.
            InterShardFlow::Ghost(g) => match g {
                GhostFlow::Spawn { .. } => FlowDurabilityClass::ReDriven,
                GhostFlow::Delta { .. } => FlowDurabilityClass::Unreliable,
                GhostFlow::Despawn { .. } => FlowDurabilityClass::ProducerLessReliable,
            },
            // The entity crossing: InitialSpawn/StubCrossing are saga steps (the orchestrator re-drives them
            // via scan_deadlines); TransientBatch is the SOURCE-shard emit that precedes the saga's AwaitAdopt
            // — no re-driver, the D-6 #1 producer-less case.
            InterShardFlow::Transfer(env) => match &env.payload {
                TransitionPayload::InitialSpawn { .. } => FlowDurabilityClass::ReDriven,
                TransitionPayload::StubCrossing { .. } => FlowDurabilityClass::ReDriven,
                TransitionPayload::TransientBatch { .. } => {
                    FlowDurabilityClass::ProducerLessReliable
                }
            },
            // Every remaining arm is orchestrator/saga-driven — a producer crash is recovered by the saga's
            // scan_deadlines re-drive, so the RAM retry suffices (grouped, DRY — the effect_class discipline).
            InterShardFlow::Directory(_)
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
            // R-6d3c: the orchestrator-EMITTED discard-poison — NOT producer-less (it does not grow the
            // outbox set); a FIRE-ONCE terminal egress, its lost-delivery residual is CA-1/L5-gated.
            | InterShardFlow::TransientDiscard(_)
            // CA-1 S3: the orchestrator-EMITTED AwaitAdopt probe — re-driven by scan_deadlines every
            // Timeout, so the RAM retry suffices (a lost probe is re-emitted next Timeout); NOT producer-less.
            | InterShardFlow::ReSolicitBatch(_)
            // Slice 3c spatial transfer-trigger: the crossing requests are re-EMITTED by the source's ongoing
            // boundary detection (the entity is still over the boundary next tick), and the grant/aborted are
            // orchestrator/saga round-trip control (the saga's scan_deadlines re-drives them) — so the RAM
            // retry + the reliable Saga lane suffice; NONE grows the producer-less outbox set. Slice 3f: the
            // `CrossingAbortedAck` is source-re-driven too (re-sent on every re-delivered `CrossingAborted`),
            // so it is `ReDriven`, never producer-less.
            | InterShardFlow::CrossingRequest(_)
            | InterShardFlow::TransientCrossingRequest(_)
            | InterShardFlow::TransientCrossingGrant(_)
            | InterShardFlow::CrossingAborted(_)
            | InterShardFlow::CrossingAbortedAck(_)
            // RLM Step 1: the parent re-asserts this demand every tick it holds (and the child its
            // Empty report), so a dropped verb self-heals next tick — ReDriven, never producer-less.
            | InterShardFlow::RealmDemand(_)
            // RLM reactive greeting: the shard re-asserts it on its silence cadence, so a dropped greeting
            // self-heals next cadence — ReDriven, never producer-less (a lost greeting needs no durable
            // outbox; the next re-greet re-teaches the gateway's return connection).
            | InterShardFlow::ShardPresence(_) => FlowDurabilityClass::ReDriven,
            // VU AoI S2a: the occupant-position up-flow is a 20 Hz latest-wins datagram (like
            // `GhostFlow::Delta`) — a lost hint self-heals on the next tick's re-assertion, never a durable
            // outbox burden ⇒ Unreliable (NOT producer-less; the golden pin below still asserts exactly TWO).
            InterShardFlow::OccupantInterest(_) => FlowDurabilityClass::Unreliable,
            // VU AoI S2c — RELIABLE (a lost set-change would blink a neighbour) but RE-DRIVEN, not
            // producer-less: on any loss / crash / shed the parent re-sends the full set from its RAM store
            // and the home reconciles it — no durable outbox (the level full-set IS the recovery).
            InterShardFlow::ProxySceneSet(_) => FlowDurabilityClass::ReDriven,
        }
    }
}

/// The lifecycle verb a [`RealmDemand`] carries. LEVEL-TRIGGERED: re-asserted every tick; a dropped
/// verb self-heals on the next re-assertion (⇒ `ReDriven`). `Empty` is the child self-reporting
/// upward that it holds no occupants (it is the occupancy authority — a sealed parent cannot see
/// inside it). APPEND-only after `TearDown` (frozen postcard discriminants).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DemandVerb {
    /// An occupant AoI now reaches this child: it must be running.
    SpinUp = 0,
    /// Still reached — keep it running (steady-state re-assertion).
    KeepAlive = 1,
    /// Child → parent: I hold no occupants (occupancy self-report, RLM R1).
    Empty = 2,
    /// No occupant AoI reaches this child: it may be reclaimed.
    TearDown = 3,
}

/// A level-triggered realm-lifecycle demand (RLM Step 1). Carries the CHILD endpoint as a
/// lineage-anchored [`RealmCoord`] (globally unique — the consumer dedups on `child.path()`, never
/// the lossy `child.lowered()`), the PARENT's authority [`Fence`] (the authority proof; the parent
/// coord is redundant — `child.parent()` derives it — so it is NOT on the wire), the verb, and the
/// tick it was asserted at. Field order is frozen once shipped (positional postcard).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealmDemand {
    pub child: RealmCoord,
    pub parent_fence: Fence,
    pub verb: DemandVerb,
    pub universe_tick: UniverseTick,
}

/// The RLM reactive-greeting payload (shard → gateway). DELIBERATELY MINIMAL: the mechanism is the
/// arrival of a reliable frame (the mesh learns the return connection below the app seam), so the body
/// need carry nothing routing-relevant. It carries the shard's OWN [`TickId`] — its process-local tick,
/// NOT the synced [`UniverseTick`] (which is 0 until the first clock-sync; the greeting fires pre-sync, so
/// a universe tick would be meaningless) — purely for gateway-side observability (the `from` node id and
/// this tick in a log line). No `NodeId` (redundant with the frame's sender), no realm/coord (that is the
/// fenced directory's authority, never a self-assertion), no address (the gateway never dials — it reuses
/// the accepted connection). Field order is frozen once shipped (positional postcard).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardPresence {
    pub local_tick: TickId,
}

/// The occupant-position up-flow payload (VU AoI S2a) — see [`InterShardFlow::OccupantInterest`]. Carries
/// ONLY where an observer is, so a sealed ancestor can cull its own children against it; NO fence (the
/// parent authorizes its own emitted demands with its OWN realm fence — a fence here would make it
/// authority-gating, forcing `SideEffecting`/reliable and breaking the unreliable-scale premise). Field
/// order is frozen once shipped (positional postcard).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccupantInterest {
    /// The DURABLE player id whose scene this feeds (survives re-home; the gateway maps it to the live
    /// session for the render-delta route).
    pub observer: AccountId,
    /// THIS hop's PARENT realm (`child.parent()`) — the routing key. Its `path()` dedups multi-hop relays.
    pub to_realm: RealmCoord,
    /// The observer's pose (position + velocity) expressed in the EMITTER's frame, stamped at its
    /// `universe_tick`. The parent re-expresses it into its own frame (compose with the child's placement).
    pub occupant: StampedPose,
    /// Hops from the observer's home realm (0 at the first parent). Drives the precision ladder: the
    /// deepest ancestors get the full pose; far ancestors need only the lineage (the coarsening is S3).
    pub coarsen_level: u8,
}

/// VU AoI S2c — a parent's LEVEL-triggered reflection of ONE proxy occupant's CURRENT in-range sibling set,
/// sent DOWN to the home shard that relays the occupant. The home shard diffs `realms` against its per-account
/// forwarded baseline to derive the client add/remove — so a lost/shed message or a re-home hand-off self-heals
/// on the next set (an EDGE would be delivered-then-dropped to a route that changes under the crossing). Empty
/// `realms` = "the proxy's AoI holds no sibling now" ⇒ the home reconciles every forwarded id for the account
/// to `removed`. PUBLIC parent-authored geometry ONLY — no NodeId / gateway / session (HR1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxySceneSet {
    /// The DURABLE traveller id — the SAME key the home shard already streams `RealmSceneDelta` under.
    pub observer: AccountId,
    /// The FULL current set of the proxy's in-range sibling outlines (public `RealmShape` geometry).
    pub realms: Vec<crate::channels::RealmShape>,
}

/// Orchestrator → SOURCE shard pose-flush request (Slice 1d.1). The source finds the held dot for
/// `subject` and replies [`TransferAck::SourceFlushed`] with the dot's authoritative pose. The
/// `step_id` is always [`FLUSH_SOURCE_STEP`]; it is carried (not a bare const at the use site) so
/// `effect_class` keys uniformly across every arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlushSource {
    pub transfer: TransferId,
    pub subject: DirectoryKey,
    pub step_id: u32,
}

/// Orchestrator → SOURCE shard ORDERED-demote command (Slice 1d.5b). The source finds the held dot
/// for `subject`, applies `Owned→Frozen→Ghost` at `new_owner_fence` (the dest's post-CAS fence — the
/// per-entity `Authority::apply` gates on `is_stale_against`), retains it as the ghost, and acks
/// `DemoteAck`. The `step_id` is always [`DEMOTE_STEP`] (carried, not inline — `effect_class` keys
/// the orchestrator's `applied_steps` idempotency uniformly by `(transfer, DEMOTE_STEP)`); a SHARD
/// consumer realizes the same step idempotency via its per-entity authority state — an already-Ghost
/// dot re-acks WITHOUT re-flipping (`!simulates()` guard), equivalent to a journal for one transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DemoteCmd {
    pub transfer: TransferId,
    pub subject: DirectoryKey,
    pub new_owner_fence: Fence,
    pub step_id: u32,
}

/// Orchestrator → DEST shard ORDERED-promote command (Slice 1d.5b). Reachable in the saga ONLY
/// after `DemoteAck` (demote-before-promote). `step_id` is always [`PROMOTE_STEP`]. In 1d.5b.3 this
/// is the REAL `Ghost→Owned` promoter (the flip RELOCATED here out of the dest's autonomous
/// adopt-promote, so the ordering is strict — the dest becomes Owned only on this command, after the
/// source has demoted). `source` is the transfer SOURCE node (where the demoted dot retains its
/// kinematic ghost): the dest, on promoting, registers `source` as a ghost-neighbor and DRIVES the
/// `GhostFlow` collider feed to it (owner → ghost-host, per `GhostFlow`'s Spawn/Delta/Despawn) so the
/// retained source ghost stays a live collider + the render stays seamless across the handoff. The
/// dest acks `PromoteAck`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromoteCmd {
    pub transfer: TransferId,
    pub subject: DirectoryKey,
    pub new_fence: Fence,
    pub step_id: u32,
    /// The transfer SOURCE node — the ghost-host the dest feeds via `GhostFlow` after promoting.
    pub source: NodeId,
}

/// Orchestrator → re-home TARGET shard adopt command (D-37 forward re-home). When a transfer's committed
/// owner is permanently KILLED, the orchestrator re-homes the subject onto a LIVE capability-matched
/// shard: it first commits the directory to `target` at `new_fence` (the fence-monotone bump — `new_fence`
/// strictly exceeds the dead owner's recorded fence, so a resurrected corpse self-fences), THEN sends this
/// command so the target ADOPTS the subject from `state`. The target acks `PromoteAck` (it is now the
/// Owned authority) and registers `source` (the demoted ghost-host) as a ghost-neighbor, exactly as
/// `Promote` does. `step_id` is always [`RE_HOME_STEP`] (carried for uniform `effect_class` keying).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReHomeCmd {
    pub transfer: TransferId,
    /// The universe epoch this adopt was minted under (mirrors [`TransferEnvelope::universe_epoch`]).
    /// The re-home target REFUSES a command whose epoch mismatches its current clock epoch
    /// (transfer_protocol §3.3 fail-safe — no entity reconstructed at a stale celestial position). The
    /// re-home adopt is the SECOND pose-placing ingress (the first is the crossing), so this makes the
    /// §3.3 guard UNIFORM across both. Stamped by the orchestrator at `build_rehome` from its clock.
    pub universe_epoch: EpochId,
    pub subject: DirectoryKey,
    pub new_fence: Fence,
    pub step_id: u32,
    /// The adopt payload (D-37): the subject's authoritative state the target reconstructs from.
    pub state: ReHomeState,
    /// The transfer's ghost-host (the demoted source) the target registers as a ghost-neighbor — as
    /// `Promote` does. DEAD in the standing re-home (CELL 3); the registration is then inert.
    pub source: NodeId,
}

/// The re-home adopt payload (D-37). POSE-ONLY today (the stub tier — every realm is points in empty
/// space); the P7 checkpoint slice grows a `Snapshot(Vec<u8>)` TLV-blob arm ADDITIVELY (the new owner
/// opens the RealmId-keyed redb, loads the snapshot, replays the WAL `> up_to_lsn`, restores the player
/// checkpoint honoring durable freeze markers). The typed enum IS the P7 state-reload SEAM — never an
/// opaque escape hatch; only the blob FILL is deferred, so P7 is not painted into a corner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReHomeState {
    /// The subject's authoritative pose (the source's flushed pose, carried verbatim).
    PoseOnly(StampedPose),
}

/// The ONE shape carried by the three TRANSIENT structural drop-before-promote command arms (D-7b):
/// `TransientRelease` (orch→source, `TRANSIENT_RELEASE_STEP` — flip `Held→Departing`), `TransientDrop`
/// (orch→dest, `TRANSIENT_DROP_STEP` — PROMOTE `Arriving→Held`), `ReleaseComplete` (orch→source,
/// `TRANSIENT_RELEASE_STEP` — retire the `Departing` copy). `transfer` IS the `BatchId` (one saga per
/// batch). `step_id` (carried, not inline — `effect_class` keys idempotency uniformly) + the ARM
/// together name the action. `fence` is the batch's commit fence (the dest realm-lease fence the
/// go-token committed at): the dest anchors the promoted item to it (it is NOT yet a stale-reject
/// guard — the in-process saga-ordered path makes a stale handoff unrepresentable; a fence-rule-1
/// reject lands with the cross-host mesh transport, P3+). Dedup is by the journaled `(transfer,
/// step_id)` idempotency at each receiver — at-least-once, redelivery re-acks without re-effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransientHandoff {
    pub transfer: TransferId,
    pub step_id: u32,
    pub fence: Fence,
}

/// SHARD → ORCHESTRATOR (Slice 3c spatial transfer-trigger): a DURABLE entity crossed a realm boundary;
/// the orchestrator resolves the dest for `to_realm` and starts the durable transfer saga. Carries NO
/// `TransferId` — the orchestrator MINTS it on start — so `effect_class` keys idempotency on
/// `subject_fence` ([`IdempotencyKey::FencedKey`]): a redelivered request for the same fenced crossing
/// resolves to a no-op, never a second saga.
///
/// Slice 3f — carries the subject's `session`: the durable crossing saga's FIRST action is a
/// `PrepareSubscribe { session, dest }` routed to the client's gateway, and the orchestrator has no
/// `Entity → Session` reverse index, so the SOURCE (which owns the dot, keyed by `SessionId`) supplies it.
/// Transients need none (their short batch path emits no session-bearing command), so only THIS request
/// grew the field. APPENDED (postcard field-append — preserves the arm's discriminant).
///
/// Slice 3f-D — carries the source's `attempt` (its per-entity crossing-attempt counter) so the
/// orchestrator derives the IDENTICAL per-attempt [`crossing_transfer_id`]: each crossing attempt gets a
/// unique id, so a stale abort for an old attempt can never wrong-clear a same-fence re-cross (H2).
/// APPENDED (postcard field-append).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossingRequest {
    pub subject: DirectoryKey,
    pub from_realm: RealmId,
    pub to_realm: RealmId,
    pub subject_fence: Fence,
    pub session: SessionId,
    pub attempt: u32,
    /// The dest realm's PARENT provenance (the fix for the "Area label never flips" bug). An `AreaLocal`
    /// frame carries `{planet_seed, area_seed}`, so re-expressing a pose into an `Area` needs its enclosing
    /// `Planet` — which only the SOURCE detector knows (the container region's `parent`, a deterministic
    /// worldgen fact). The saga threads it VERBATIM to `rebind_pose_to_dest(.., to_realm, to_parent)` so an
    /// Area frame forms; every other realm kind is a one-field lift and ignores it (`None`). APPENDED
    /// (postcard field-append — preserves the arm's discriminant).
    pub to_parent: Option<RealmId>,
}

/// SHARD → ORCHESTRATOR (Slice 3c spatial transfer-trigger): a TRANSIENT crossed a realm boundary; the
/// orchestrator resolves the dest for `to_realm` and grants it back ([`TransientCrossingGrant`]). Carries
/// NO `TransferId` (the orchestrator mints the batch id), so `effect_class` keys idempotency on
/// `src_realm_fence` ([`IdempotencyKey::FencedKey`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransientCrossingRequest {
    pub subject: DirectoryKey,
    pub from_realm: RealmId,
    pub to_realm: RealmId,
    pub src_realm_fence: Fence,
    /// The dest realm's PARENT provenance — see [`CrossingRequest::to_parent`]. The source detector fills it
    /// from the container region's `parent`; the orchestrator copies it into the [`TransientCrossingGrant`],
    /// which the source stamps onto `TransientStatus::Crossing` so the batch's `rebind_pose_to_dest` forms an
    /// Area frame. `None` for every non-Area dest. APPENDED (postcard field-append).
    pub to_parent: Option<RealmId>,
}

/// ORCHESTRATOR → SOURCE shard (Slice 3c): the GRANT carrying the resolved dest for a
/// [`TransientCrossingRequest`]. Its five fields EXACTLY match `sim::stub::TransientStatus::Crossing`
/// (`dest`/`to_realm`/`dst_realm_fence`/`batch`/`to_parent`), so the source flip is a straight field
/// assign. `batch` is the orchestrator-minted batch id — `effect_class` keys idempotency on
/// `(batch, TRANSIENT_BATCH_STEP)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransientCrossingGrant {
    pub subject: DirectoryKey,
    pub dest: NodeId,
    pub to_realm: RealmId,
    pub dst_realm_fence: Fence,
    pub batch: TransferId,
    /// The dest realm's PARENT provenance — copied VERBATIM from the [`TransientCrossingRequest`] the
    /// orchestrator resolved (see [`CrossingRequest::to_parent`]). The source stamps it onto
    /// `TransientStatus::Crossing` so `emit_transient_batch`'s `rebind_pose_to_dest` forms an Area frame.
    /// `None` for every non-Area dest. APPENDED (postcard field-append).
    pub to_parent: Option<RealmId>,
}

/// ORCHESTRATOR → SOURCE shard (Slice 3c): the crossing's resolve/start saga ABORTED pre-CAS, so the
/// source CLEARS the subject's `RequestInFlight` latch — the entity is free to re-cross. The PROPER fix (a
/// positive orchestrator signal), never a bounded TTL window. `transfer` is the aborted saga id;
/// `effect_class` keys idempotency on `(transfer, TRANSIENT_BATCH_STEP)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossingAborted {
    pub subject: DirectoryKey,
    pub transfer: TransferId,
}

/// The ONE deterministic namespaced-`TransferId` primitive (Slice 3f-D, DRY hoist): FNV-1a over an
/// already-encoded `seed`, with `tag` in the high byte so disjoint namespaces (`0x39` crossing,
/// `0x37` re-home) can never collide with each other or with the small/sequential connection-plane ids
/// (high byte `0x00`). Branchless (HR5: no rng / wall-clock / default hasher — the byte-identical
/// seed-replay canary forbids all three). Callers encode their OWN seed, so this serves both the
/// 3-arg crossing id and the 2-arg re-home id without arity coupling.
#[must_use]
pub fn namespaced_transfer_id(tag: u8, seed: &[u8]) -> TransferId {
    const FNV_OFFSET: u128 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u128 = 0x0000_0100_0000_01b3;
    const LOW_120: u128 = (1u128 << 120) - 1;
    let mut h = FNV_OFFSET;
    for b in seed {
        h = (h ^ u128::from(*b)).wrapping_mul(FNV_PRIME);
    }
    TransferId(((tag as u128) << 120) | (h & LOW_120))
}

/// Deterministic `TransferId` for a geometric crossing — the SAME value the source (its
/// `RequestInFlight` latch) and the orchestrator INDEPENDENTLY derive from `(subject, subject_fence,
/// attempt)`, so a saga terminal (`Demote` / `CrossingAborted`) carrying that id matches the source
/// latch. High-byte namespace-tagged `0x39` (disjoint from `rehome_transfer_id`'s `0x37`).
///
/// Slice 3f-D — the `attempt` is the source's per-entity crossing-attempt counter (bumped ONLY on a new
/// latch, i.e. only AFTER the prior latch cleared on abort/commit). It makes each crossing ATTEMPT's id
/// UNIQUE, so a stale `CrossingAborted` for an old attempt can never wrong-clear a same-fence re-cross's
/// fresh latch (H2). Idempotency is PRESERVED where it matters: a lost-RAM-enqueue keeps the latch held
/// (no new attempt, the ttl-re-drive re-emits the same id), and a crash resets the RAM counter to `0`
/// (the restored in-band dot re-mints the same first-attempt id). Only a post-abort re-latch — exactly
/// where H2 needs a fresh id — advances the attempt.
#[must_use]
pub fn crossing_transfer_id(
    subject: DirectoryKey,
    subject_fence: Fence,
    attempt: u32,
) -> TransferId {
    let seed =
        postcard::to_allocvec(&(subject, subject_fence, attempt)).expect("encode crossing id seed");
    namespaced_transfer_id(0x39, &seed)
}

/// Ghost replication: kinematic mirrors that NEVER independently integrate physics.
///
/// OWED (DEFERRED D-39.6, P11 combat): cross-boundary PvP needs ghosts to carry a small read-only
/// replicated combat-STATE blob (health/shield/anim/pilot-flags, per `transfer_protocol.md:107`) so a
/// shard rendering a ghost owned by another shard can show its health/downed state and gate a hit
/// before forwarding the fire-event to the owner. Today Spawn/Delta carry pose+fences only; the blob
/// lands as a NEW GhostFlow VARIANT (a field-append to Spawn/Delta is NOT postcard-safe), read-only
/// display state — the authoritative hit is still applied at the ghost's OWNER (D-39.1 forward path).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GhostFlow {
    /// RELIABLE delivery ([`MsgClass::GhostReliable`]): the ghost-host inserts a kinematic ghost (a
    /// lost Spawn would strand a never-spawned collider). EffectClass is `FireAndForget` (no
    /// idempotency key); the receiver's insert is idempotent (a redelivery overwrites the same entry).
    Spawn {
        entity: EntityId,
        pose: StampedPose,
        source_fence: Fence,
        since_tick: TickId,
    },
    /// Datagram, 20 Hz, latest-wins. Deltas older than the ghost's `since_tick`
    /// or carrying a stale fence are dropped by data, not by stream ordering.
    Delta {
        entity: EntityId,
        pose: StampedPose,
        source_fence: Fence,
        source_tick: TickId,
        seq: u64,
    },
    /// RELIABLE delivery ([`MsgClass::GhostReliable`]): the owner emits this on BAND-EXIT (the entity
    /// left the host's overlap band) and the host tears the ghost down (a lost Despawn would leak the
    /// collider). EffectClass is `FireAndForget` (no idempotency key); the teardown is IDEMPOTENT —
    /// a redelivery, or a stale Despawn for an entity the host has since re-owned, removes nothing
    /// (the host only tears down a dot still held as a retained Ghost; `vd-sim` `remove_retained_ghost`).
    /// The directory `in_transfer` field is DEAD (cleared at commit-CAS), so mid-transfer protection is
    /// structural: the destroy edge is sized past the handoff window (band-exit is post-release) + the
    /// orchestrator's one-saga-per-key lock. The proper orchestrator-side teardown gate is owed
    /// (`docs/design/DEFERRED.md` D-2, Slice-2).
    Despawn {
        entity: EntityId,
        source_fence: Fence,
    },
}

/// The transfer envelope: ONE shape for every entity kind and both durability
/// classes — the registry (`KindDef`) is the only per-kind variation (HR2/HR3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransferEnvelope {
    pub transfer_id: TransferId,
    pub universe_epoch: EpochId,
    /// Control-plane schema version (postcard, additive under minor negotiation).
    pub schema_version: u16,
    /// The authority fence this message was issued under; stale-fence messages are
    /// rejected by every receiver (fence rule 1).
    pub fence: Fence,
    /// `(transfer_id, step_id)` is the universal side-effect idempotency key.
    pub step_id: u32,
    pub class: DurabilityClass,
    pub payload: TransitionPayload,
}

/// Typed per-transition payloads — the compiler forbids a boarding message carrying
/// warp fields (the R4 god-struct is unrepresentable). Variants are added with their
/// phase; P0 carries what stub shards (P1–P3) need.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TransitionPayload {
    /// First spawn of an entity into a realm (login/bootstrap).
    InitialSpawn {
        entity: EntityId,
        to_realm: RealmId,
        pose: StampedPose,
        /// TLV state blob (`vd_core::tlv`), schema per `KindDef::blob_schema`.
        state: Vec<u8>,
    },
    /// A stub-shard boundary crossing (P2/P3 transfer machinery proving ground;
    /// the real transition classes — boarding/EVA/SOI/warp — land as additive
    /// variants with their phases).
    StubCrossing {
        entity: EntityId,
        from_realm: RealmId,
        to_realm: RealmId,
        pose: StampedPose,
        state: Vec<u8>,
    },
    /// Batched transient handover (debris/projectiles): items share ONE envelope and
    /// ONE batched `TransientGo` go-token — never per-item directory writes.
    TransientBatch {
        from_realm: RealmId,
        to_realm: RealmId,
        src_realm_fence: Fence,
        dst_realm_fence: Fence,
        source_tick: TickId,
        items: Vec<TransientItem>,
    },
}

/// One transient entity inside a batch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransientItem {
    pub entity: EntityId,
    pub pose: StampedPose,
    /// TLV blob, capped by the kind's `max_state_bytes`.
    pub state: Vec<u8>,
}

/// The SHARD → orchestrator ack family for the entity-STATE half of a transfer (Slice 1d.1 gives
/// this long-RESERVED type its first consumer — DEFERRED.md D-21). NOT the gateway↔saga route-swap
/// vocabulary: that is `seams::transfer_control::TransferControlAck`. Every arm is keyed by
/// `(transfer_id, step_id)` for idempotent journaling. The direction (shard→orch) is permanent.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum TransferAck {
    /// SOURCE → orch (phase [`FLUSH_SOURCE_STEP`]): the reply to [`FlushSource`], carrying the
    /// subject's authoritative pose + the source's input drain watermark so the saga can stamp the
    /// crossing. The pose-before-promote gate makes a missing flush block the commit (it never
    /// silently emits a poseless crossing).
    SourceFlushed {
        transfer_id: TransferId,
        step_id: u32,
        pose: StampedPose,
        drained_seq: u64,
    },
    /// DEST → orch (phase [`STUB_CROSSING_STEP`]): accepted and journaled the `StubCrossing` step.
    Accepted {
        transfer_id: TransferId,
        step_id: u32,
    },
    /// DEST → orch: refused; the entity stays authoritative on the source. (A `StubCrossing` is
    /// emitted POST-commit/forward-only, so spatial admissibility is gated PRE-commit at
    /// `PrepareSubscribe`; this arm carries the typed refusal for the non-spatial step rejections.)
    Rejected {
        transfer_id: TransferId,
        step_id: u32,
        reason: TransferStepRejectReason,
    },
    /// DEST → orch (phase [`TRANSIENT_BATCH_STEP`], D-7a): the dest ADOPTED a `TransientBatch` into
    /// its `Arriving` (uncounted) tier and journaled the step. This is the transient twin of the
    /// durable `DemoteAck` — it GATES the orchestrator's `TransientRelease` (adopt-before-drop), so the
    /// source never drops until the dest holds the items. `transfer_id` IS the `BatchId`.
    BatchAdopted {
        transfer_id: TransferId,
        step_id: u32,
    },
    /// SHARD → orch (D-7b): proof a transient handoff step was applied. The SOURCE sends it with
    /// `step_id = TRANSIENT_RELEASE_STEP` (the `Held→Departing` release is done — GATES the dest
    /// promote); the DEST sends it with `step_id = TRANSIENT_DROP_STEP` (the `Arriving→Held` promote is
    /// done — GATES the source's `ReleaseComplete`). The SAME ack variant, distinguished by `step_id`
    /// — the transient twin of `DemoteAck`/`PromoteAck` collapsed onto one phased ack.
    DropApplied {
        transfer_id: TransferId,
        step_id: u32,
    },
}

impl TransferAck {
    /// The transfer this ack belongs to (the `(transfer, step)` idempotency key's first half).
    #[must_use]
    pub fn transfer_id(&self) -> TransferId {
        match self {
            TransferAck::SourceFlushed { transfer_id, .. }
            | TransferAck::Accepted { transfer_id, .. }
            | TransferAck::Rejected { transfer_id, .. }
            | TransferAck::BatchAdopted { transfer_id, .. }
            | TransferAck::DropApplied { transfer_id, .. } => *transfer_id,
        }
    }

    /// The step phase this ack belongs to (the key's second half).
    #[must_use]
    pub fn step_id(&self) -> u32 {
        match self {
            TransferAck::SourceFlushed { step_id, .. }
            | TransferAck::Accepted { step_id, .. }
            | TransferAck::Rejected { step_id, .. }
            | TransferAck::BatchAdopted { step_id, .. }
            | TransferAck::DropApplied { step_id, .. } => *step_id,
        }
    }
}

/// Typed causes a dest shard refuses a `Transfer`-arm STEP (never a stringly-typed
/// warn-and-drop). Distinct from the CLIENT-facing `channels::TransferRejectReason`
/// (which explains a refusal to the player) — this is the shard-internal step-ack
/// reason, paired with [`TransferAck`] and keyed by `step_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferStepRejectReason {
    SpatialPrecondition,
    StaleFence,
    EpochMismatch,
    VersionFloor,
    UnknownKind,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::UniverseTick;
    use vd_core::entity_kind::EntityKind;
    use vd_core::glam::DVec3;
    use vd_core::pose::FrameRef;

    fn pose() -> StampedPose {
        StampedPose::at_rest(
            FrameRef::SystemSpace { system_seed: 1 },
            DVec3::ZERO,
            UniverseTick(10),
        )
    }

    fn eid(kind: EntityKind) -> EntityId {
        EntityId::pack(kind, 1, 7, 3)
    }

    fn envelope(payload: TransitionPayload, class: DurabilityClass) -> TransferEnvelope {
        TransferEnvelope {
            transfer_id: TransferId(11),
            universe_epoch: EpochId(1),
            schema_version: 1,
            fence: Fence(2),
            step_id: 4,
            class,
            payload,
        }
    }

    /// A Universe-rooted `[Universe, Galaxy, System]` lineage (globally-unique path); leaf lowers
    /// to a real `RealmId::System`.
    fn demand_coord() -> RealmCoord {
        use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
        RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::Universe, 0),
            RealmLevel::new(RealmKindTag::Galaxy, 2),
            RealmLevel::new(RealmKindTag::System, 7),
        ]))
        .expect("3-level path has a leaf")
    }

    /// A `[Universe, Galaxy]` lineage — a distinct wire shape (shorter path; leaf lowers via the
    /// Galaxy stand-in), so the round-trip exercises more than one `RealmCoord` encoding.
    fn galaxy_demand_coord() -> RealmCoord {
        use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
        RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::Universe, 0),
            RealmLevel::new(RealmKindTag::Galaxy, 2),
        ]))
        .expect("2-level path has a leaf")
    }

    #[test]
    fn realm_demand_arms_roundtrip() {
        // Every verb × two coord shapes: postcard round-trips the appended arm + its RealmCoord.
        for verb in [
            DemandVerb::SpinUp,
            DemandVerb::KeepAlive,
            DemandVerb::Empty,
            DemandVerb::TearDown,
        ] {
            for child in [demand_coord(), galaxy_demand_coord()] {
                let flow = InterShardFlow::RealmDemand(RealmDemand {
                    child,
                    parent_fence: Fence(4),
                    verb,
                    universe_tick: UniverseTick(9),
                });
                let bytes = postcard::to_allocvec(&flow).expect("encode");
                let back: InterShardFlow = postcard::from_bytes(&bytes).expect("decode");
                assert_eq!(back, flow);
            }
        }
    }

    #[test]
    fn realm_demand_effect_and_durability() {
        // The appended classifier arms (equality, not `matches!`): FencedKey{parent_fence} + ReDriven.
        let flow = InterShardFlow::RealmDemand(RealmDemand {
            child: demand_coord(),
            parent_fence: Fence(4),
            verb: DemandVerb::SpinUp,
            universe_tick: UniverseTick(9),
        });
        assert_eq!(
            flow.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedKey { fence: Fence(4) }
            }
        );
        assert_eq!(flow.durability_class(), FlowDurabilityClass::ReDriven);
    }

    #[test]
    fn occupant_interest_effect_and_durability_and_round_trips() {
        // VU AoI S2a — the appended classifier arms (equality, not `matches!`): FireAndForget + Unreliable,
        // and the postcard roundtrip (a durable observer id, a parent coord, an in-frame pose, a level).
        let flow = InterShardFlow::OccupantInterest(OccupantInterest {
            observer: AccountId(5),
            to_realm: demand_coord(),
            occupant: pose(),
            coarsen_level: 2,
        });
        assert_eq!(flow.effect_class(), EffectClass::FireAndForget);
        assert_eq!(flow.durability_class(), FlowDurabilityClass::Unreliable);
        let bytes = postcard::to_allocvec(&flow).expect("encode");
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
            flow
        );
    }

    #[test]
    fn proxy_scene_set_effect_and_durability_and_round_trips() {
        // VU AoI S2c — the appended classifier arms (equality, not `matches!`): FireAndForget (public geometry,
        // no fence, no home-side sim mutation) + ReDriven (reliable, RAM-re-sent — NOT producer-less, so the
        // golden `producer_less.len() == 2` pin is unchanged). Plus the postcard roundtrip of the new variant.
        let flow = InterShardFlow::ProxySceneSet(ProxySceneSet {
            observer: AccountId(5),
            realms: vec![],
        });
        assert_eq!(flow.effect_class(), EffectClass::FireAndForget);
        assert_eq!(flow.durability_class(), FlowDurabilityClass::ReDriven);
        let bytes = postcard::to_allocvec(&flow).expect("encode");
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
            flow
        );
    }

    /// G-SEALED: every arm has a coherent effect class, and side-effecting arms
    /// expose their idempotency key. The match in `effect_class` is exhaustive, so
    /// adding an arm without classifying it cannot compile.
    #[test]
    fn g_sealed_effect_classes() {
        let ghost = InterShardFlow::Ghost(GhostFlow::Delta {
            entity: eid(EntityKind::Player),
            pose: pose(),
            source_fence: Fence(1),
            source_tick: TickId(5),
            seq: 9,
        });
        assert_eq!(ghost.effect_class(), EffectClass::FireAndForget);

        let durable = InterShardFlow::Transfer(envelope(
            TransitionPayload::InitialSpawn {
                entity: eid(EntityKind::Player),
                to_realm: RealmId::System(1),
                pose: pose(),
                state: vec![],
            },
            DurabilityClass::Durable,
        ));
        assert_eq!(
            durable.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 4
                }
            }
        );

        // Transient batches share the SAME side-effecting machinery (HR3): the
        // batched go-token is their commit point; they are NOT fire-and-forget.
        let transient = InterShardFlow::Transfer(envelope(
            TransitionPayload::TransientBatch {
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                src_realm_fence: Fence(3),
                dst_realm_fence: Fence(4),
                source_tick: TickId(6),
                items: vec![TransientItem {
                    entity: eid(EntityKind::Debris),
                    pose: pose(),
                    state: vec![1],
                }],
            },
            DurabilityClass::Transient,
        ));
        assert_eq!(
            transient.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 4
                }
            }
        );

        // Directory ops classify through the flow wrapper identically to direct calls.
        let directory = InterShardFlow::Directory(DirectoryOp::CommitCas {
            key: crate::seams::directory::DirectoryKey::Realm(RealmId::System(1)),
            expected: Fence(5),
            transfer: TransferId(6),
            new_owner: crate::seams::directory::AuthorityRef::Shard(vd_core::NodeId(2)),
        });
        assert_eq!(
            directory.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::FencedCas {
                    expected: Fence(5),
                    transfer: TransferId(6),
                }
            }
        );

        // Saga commands + acks ride the closed taxonomy as side-effecting arms keyed by
        // (transfer, phase) — the route swap is never untyped bytes (HR1). FreezeSource and
        // its ack SourceFrozen are both phase 2.
        let saga = InterShardFlow::Saga(TransferControl::FreezeSource {
            transfer: TransferId(11),
            session: vd_core::SessionId(3),
            marker_seq: 17,
            dest: vd_core::NodeId(2),
        });
        assert_eq!(
            saga.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 2,
                }
            }
        );
        let saga_ack = InterShardFlow::SagaAck(TransferControlAck::SourceFrozen {
            transfer: TransferId(11),
            drained_seq: 17,
        });
        assert_eq!(
            saga_ack.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: 2,
                }
            }
        );

        // A directory REPLY is a re-derivable report (the receiver pulls through to the
        // authority-of-record directory), never a transfer trigger → FireAndForget.
        let reply = InterShardFlow::DirectoryReply(DirectoryReply::Head {
            key: crate::seams::directory::DirectoryKey::Realm(RealmId::System(1)),
            record: None,
        });
        assert_eq!(reply.effect_class(), EffectClass::FireAndForget);

        // 1d.1: the pose-flush request + the entity-state ack family are side-effecting, keyed by
        // their step phase (7 for the source flush + its reply; 8 for the dest crossing + its ack).
        let flush = InterShardFlow::FlushSource(FlushSource {
            transfer: TransferId(11),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            step_id: FLUSH_SOURCE_STEP,
        });
        assert_eq!(
            flush.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: FLUSH_SOURCE_STEP,
                }
            }
        );
        let flushed = InterShardFlow::TransferAck(TransferAck::SourceFlushed {
            transfer_id: TransferId(11),
            step_id: FLUSH_SOURCE_STEP,
            pose: pose(),
            drained_seq: 17,
        });
        assert_eq!(
            flushed.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: FLUSH_SOURCE_STEP,
                }
            }
        );
        let accepted = InterShardFlow::TransferAck(TransferAck::Accepted {
            transfer_id: TransferId(11),
            step_id: STUB_CROSSING_STEP,
        });
        assert_eq!(
            accepted.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: STUB_CROSSING_STEP,
                }
            }
        );
        // 1d.5b: the saga-pushed ordered demote/promote — side-effecting at (transfer, 9|10).
        let demote = InterShardFlow::Demote(DemoteCmd {
            transfer: TransferId(11),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_owner_fence: Fence(6),
            step_id: DEMOTE_STEP,
        });
        assert_eq!(
            demote.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: DEMOTE_STEP,
                }
            }
        );
        let promote = InterShardFlow::Promote(PromoteCmd {
            transfer: TransferId(11),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: PROMOTE_STEP,
            source: NodeId(2),
        });
        assert_eq!(
            promote.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: PROMOTE_STEP,
                }
            }
        );
        // D-7a/b: the transient batch ack + the structural drop-before-promote handoff commands are
        // side-effecting at their transient phases (11 batch+ack+go-token; 13 release; 12 promote) —
        // keyed by (transfer, step) like every authority-moving arm. The three handoff arms share one
        // classification arm; assert each routes through it (DRY proof).
        let adopted = InterShardFlow::TransferAck(TransferAck::BatchAdopted {
            transfer_id: TransferId(11),
            step_id: TRANSIENT_BATCH_STEP,
        });
        assert_eq!(
            adopted.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: TRANSIENT_BATCH_STEP,
                }
            }
        );
        for (flow, step) in [
            (
                InterShardFlow::TransientRelease(TransientHandoff {
                    transfer: TransferId(11),
                    step_id: TRANSIENT_RELEASE_STEP,
                    fence: Fence(6),
                }),
                TRANSIENT_RELEASE_STEP,
            ),
            (
                InterShardFlow::TransientDrop(TransientHandoff {
                    transfer: TransferId(11),
                    step_id: TRANSIENT_DROP_STEP,
                    fence: Fence(6),
                }),
                TRANSIENT_DROP_STEP,
            ),
            (
                InterShardFlow::ReleaseComplete(TransientHandoff {
                    transfer: TransferId(11),
                    step_id: TRANSIENT_RELEASE_STEP,
                    fence: Fence(6),
                }),
                TRANSIENT_RELEASE_STEP,
            ),
            // R-6d3c: the orch→dest discard-poison rides the SAME classification arm (DRY).
            (
                InterShardFlow::TransientDiscard(TransientHandoff {
                    transfer: TransferId(11),
                    step_id: TRANSIENT_DISCARD_STEP,
                    fence: Fence(6),
                }),
                TRANSIENT_DISCARD_STEP,
            ),
        ] {
            assert_eq!(
                flow.effect_class(),
                EffectClass::SideEffecting {
                    idempotency: IdempotencyKey::TransferStep {
                        transfer: TransferId(11),
                        step_id: step,
                    }
                }
            );
        }
        // The DropApplied ack (D-7b) is side-effecting at its phase too.
        let drop_applied = InterShardFlow::TransferAck(TransferAck::DropApplied {
            transfer_id: TransferId(11),
            step_id: TRANSIENT_RELEASE_STEP,
        });
        assert_eq!(
            drop_applied.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: TRANSIENT_RELEASE_STEP,
                }
            }
        );
        // D-37: the forward re-home adopt is side-effecting at (transfer, RE_HOME_STEP), on its DEDICATED
        // arm (never reuses Promote — the no-repurpose discipline; the target adopts state, not a ghost).
        let rehome = InterShardFlow::ReHome(ReHomeCmd {
            transfer: TransferId(11),
            universe_epoch: EpochId(1),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(pose()),
            source: NodeId(2),
        });
        assert_eq!(
            rehome.effect_class(),
            EffectClass::SideEffecting {
                idempotency: IdempotencyKey::TransferStep {
                    transfer: TransferId(11),
                    step_id: RE_HOME_STEP,
                }
            }
        );
    }

    /// D-37: the dedicated `ReHome` arm + its `ReHomeState` payload survive a postcard roundtrip (the new
    /// types' derived ser/de/clone/debug), and `RE_HOME_STEP` is distinct from every other step phase.
    #[test]
    fn rehome_arm_and_payload_roundtrip() {
        let rehome = InterShardFlow::ReHome(ReHomeCmd {
            transfer: TransferId(11),
            universe_epoch: EpochId(1),
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(pose()),
            source: NodeId(2),
        });
        let bytes = postcard::to_allocvec(&rehome).expect("encode");
        let decoded: InterShardFlow = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded,
            rehome.clone(),
            "ReHome survives a postcard roundtrip"
        );
        assert!(
            format!("{rehome:?}").contains("ReHome"),
            "Debug renders the arm"
        );
        // RE_HOME_STEP is its own phase, disjoint from the route-swap (0–10) + transient (11–15) steps.
        for other in [
            FLUSH_SOURCE_STEP,
            STUB_CROSSING_STEP,
            DEMOTE_STEP,
            PROMOTE_STEP,
            TRANSIENT_ABANDON_STEP,
        ] {
            assert_ne!(RE_HOME_STEP, other);
        }
    }

    /// The entity-STATE step ids are DISJOINT from the 0–6 route-swap phases — so a state step can
    /// never alias a route-swap phase in any `(transfer, step_id)` journal.
    #[test]
    fn transfer_state_step_ids_are_disjoint_from_route_swap_phases() {
        use std::collections::BTreeSet;
        // Every entity-STATE step (flush/crossing/demote/promote/transient/rehome/discard/re-solicit) is
        // disjoint from the 0–6 route-swap phases AND pairwise distinct — so a state step can never alias a
        // phase (or another state step) in any `(transfer, step_id)` journal. The FULL 7–18 step-id space.
        let state_steps = [
            FLUSH_SOURCE_STEP,
            STUB_CROSSING_STEP,
            DEMOTE_STEP,
            PROMOTE_STEP,
            TRANSIENT_BATCH_STEP,
            TRANSIENT_DROP_STEP,
            TRANSIENT_RELEASE_STEP,
            TRANSIENT_COMPLETE_STEP,
            TRANSIENT_ABANDON_STEP,
            RE_HOME_STEP,
            TRANSIENT_DISCARD_STEP,
            RE_SOLICIT_STEP,
        ];
        for phase in 0u32..=6 {
            assert!(
                !state_steps.contains(&phase),
                "state step aliases phase {phase}"
            );
        }
        assert_eq!(
            state_steps.iter().collect::<BTreeSet<_>>().len(),
            state_steps.len(),
            "the entity-state step ids are pairwise distinct",
        );
        assert_eq!(TRANSFER_SCHEMA_VERSION, 1);
    }

    #[test]
    fn saga_arms_roundtrip() {
        for flow in [
            InterShardFlow::Saga(TransferControl::CommitAuthority {
                transfer: TransferId(11),
                session: vd_core::SessionId(3),
                new_fence: Fence(4),
                subject: crate::seams::directory::DirectoryKey::Entity(eid(EntityKind::Player)),
            }),
            InterShardFlow::SagaAck(TransferControlAck::Committed {
                transfer: TransferId(11),
            }),
            // The reply arm must roundtrip distinctly from the Saga arms (the dispatch
            // split depends on the InterShardFlow tag discriminating them).
            InterShardFlow::DirectoryReply(DirectoryReply::CasResult {
                key: crate::seams::directory::DirectoryKey::Session(vd_core::SessionId(3)),
                outcome: crate::seams::directory::CasOutcome::Won {
                    new_fence: Fence(4),
                },
            }),
            // 1d.5b: the saga-pushed ordered-demote command arms roundtrip distinctly (appended at
            // the end — Ghost is discriminant 0, so existing discriminants are unshifted).
            InterShardFlow::Demote(DemoteCmd {
                transfer: TransferId(11),
                subject: crate::seams::directory::DirectoryKey::Entity(eid(EntityKind::Player)),
                new_owner_fence: Fence(6),
                step_id: DEMOTE_STEP,
            }),
            InterShardFlow::Promote(PromoteCmd {
                transfer: TransferId(11),
                subject: crate::seams::directory::DirectoryKey::Entity(eid(EntityKind::Player)),
                new_fence: Fence(6),
                step_id: PROMOTE_STEP,
                source: NodeId(2),
            }),
        ] {
            let bytes = postcard::to_allocvec(&flow).expect("encode");
            assert_eq!(
                postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
                flow
            );
        }
    }

    /// `crossing_transfer_id` is DETERMINISTIC (two calls with the same args agree), STABLE across the
    /// (subject, fence, attempt) triple only (a different subject OR fence OR attempt yields a different id),
    /// and carries the `0x39` namespace tag (disjoint from rehome's `0x37`) in its high byte.
    #[test]
    fn crossing_transfer_id_is_deterministic_and_namespaced() {
        let subject = DirectoryKey::Entity(eid(EntityKind::Player));
        let other_subject = DirectoryKey::Entity(eid(EntityKind::Ship));
        let id = crossing_transfer_id(subject, Fence(4), 0);
        // Deterministic: a second call with the same args is byte-identical.
        assert_eq!(id, crossing_transfer_id(subject, Fence(4), 0));
        // A different fence changes the id.
        assert_ne!(id, crossing_transfer_id(subject, Fence(5), 0));
        // A different subject changes the id.
        assert_ne!(id, crossing_transfer_id(other_subject, Fence(4), 0));
        // 3f-D: a different ATTEMPT changes the id (a post-abort re-cross gets a fresh id — H2).
        assert_ne!(id, crossing_transfer_id(subject, Fence(4), 1));
        // The high byte is the `0x39` crossing namespace tag (attempt lives in the low 120 bits).
        assert_eq!(id.0 >> 120, 0x39);
    }

    #[test]
    fn crossing_arms_roundtrip() {
        for flow in [
            InterShardFlow::CrossingRequest(CrossingRequest {
                subject: DirectoryKey::Entity(eid(EntityKind::Player)),
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                subject_fence: Fence(4),
                session: SessionId(3),
                attempt: 0,
                // None here roundtrips the appended field's absent-parent encoding (the non-Area default).
                to_parent: None,
            }),
            InterShardFlow::TransientCrossingRequest(TransientCrossingRequest {
                subject: DirectoryKey::Entity(eid(EntityKind::Player)),
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                src_realm_fence: Fence(4),
                to_parent: Some(RealmId::System(1)),
            }),
            InterShardFlow::TransientCrossingGrant(TransientCrossingGrant {
                subject: DirectoryKey::Entity(eid(EntityKind::Player)),
                dest: NodeId(2),
                to_realm: RealmId::Planet(2),
                dst_realm_fence: Fence(6),
                batch: TransferId(9),
                to_parent: Some(RealmId::System(1)),
            }),
            InterShardFlow::CrossingAborted(CrossingAborted {
                subject: DirectoryKey::Entity(eid(EntityKind::Player)),
                transfer: crossing_transfer_id(
                    DirectoryKey::Entity(eid(EntityKind::Player)),
                    Fence(4),
                    0,
                ),
            }),
            InterShardFlow::CrossingAbortedAck(CrossingAborted {
                subject: DirectoryKey::Entity(eid(EntityKind::Player)),
                transfer: crossing_transfer_id(
                    DirectoryKey::Entity(eid(EntityKind::Player)),
                    Fence(4),
                    0,
                ),
            }),
        ] {
            let bytes = postcard::to_allocvec(&flow).expect("encode");
            assert_eq!(
                postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
                flow
            );
        }
    }

    /// Slice 3f: the ack answers a `CrossingAborted` on the SAME idempotency journal — both key on
    /// `(transfer, TRANSIENT_BATCH_STEP)`. Pins the request+ack shared-journal invariant (a future refactor
    /// splitting them would break the ack's dedup at the source).
    #[test]
    fn crossing_aborted_ack_shares_the_abort_idempotency_key() {
        let subject = DirectoryKey::Entity(eid(EntityKind::Player));
        let transfer = crossing_transfer_id(subject, Fence(4), 0);
        let abort = InterShardFlow::CrossingAborted(CrossingAborted { subject, transfer });
        let ack = InterShardFlow::CrossingAbortedAck(CrossingAborted { subject, transfer });
        assert_eq!(abort.effect_class(), ack.effect_class());
    }

    #[test]
    fn ghost_flow_roundtrips() {
        let flows = vec![
            GhostFlow::Spawn {
                entity: eid(EntityKind::Ship),
                pose: pose(),
                source_fence: Fence(1),
                since_tick: TickId(2),
            },
            GhostFlow::Delta {
                entity: eid(EntityKind::Ship),
                pose: pose(),
                source_fence: Fence(1),
                source_tick: TickId(3),
                seq: 1,
            },
            GhostFlow::Despawn {
                entity: eid(EntityKind::Ship),
                source_fence: Fence(1),
            },
        ];
        for flow in flows {
            let bytes = postcard::to_allocvec(&flow).expect("encode");
            assert_eq!(
                postcard::from_bytes::<GhostFlow>(&bytes).expect("decode"),
                flow
            );
        }
    }

    #[test]
    fn envelopes_and_acks_roundtrip() {
        let env = envelope(
            TransitionPayload::StubCrossing {
                entity: eid(EntityKind::Player),
                from_realm: RealmId::System(1),
                to_realm: RealmId::Planet(2),
                pose: pose(),
                state: vec![9, 9],
            },
            DurabilityClass::Durable,
        );
        let flow = InterShardFlow::Transfer(env);
        let bytes = postcard::to_allocvec(&flow).expect("encode");
        assert_eq!(
            postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
            flow
        );

        for ack in [
            TransferAck::SourceFlushed {
                transfer_id: TransferId(1),
                step_id: FLUSH_SOURCE_STEP,
                pose: pose(),
                drained_seq: 42,
            },
            TransferAck::Accepted {
                transfer_id: TransferId(1),
                step_id: STUB_CROSSING_STEP,
            },
            TransferAck::Rejected {
                transfer_id: TransferId(1),
                step_id: STUB_CROSSING_STEP,
                reason: TransferStepRejectReason::SpatialPrecondition,
            },
            TransferAck::BatchAdopted {
                transfer_id: TransferId(1),
                step_id: TRANSIENT_BATCH_STEP,
            },
            TransferAck::DropApplied {
                transfer_id: TransferId(1),
                step_id: TRANSIENT_RELEASE_STEP,
            },
        ] {
            // The accessors agree with the constructed key, over every arm.
            let bytes = postcard::to_allocvec(&ack).expect("encode");
            assert_eq!(
                postcard::from_bytes::<TransferAck>(&bytes).expect("decode"),
                ack
            );
            assert_eq!(ack.transfer_id(), TransferId(1));
            // Exercise `step_id()` over EVERY arm (incl. Rejected): every entity-state ack carries
            // a transfer-state step phase (≥ FLUSH_SOURCE_STEP, disjoint from the 0–6 route-swap).
            assert!(
                ack.step_id() >= FLUSH_SOURCE_STEP,
                "every ack carries an entity-state step phase: {ack:?}"
            );
        }

        // The two new InterShardFlow arms roundtrip distinctly (the dispatch split depends on the
        // tag discriminating them from the route-swap arms).
        for flow in [
            InterShardFlow::FlushSource(FlushSource {
                transfer: TransferId(1),
                subject: DirectoryKey::Entity(eid(EntityKind::Player)),
                step_id: FLUSH_SOURCE_STEP,
            }),
            InterShardFlow::TransferAck(TransferAck::SourceFlushed {
                transfer_id: TransferId(1),
                step_id: FLUSH_SOURCE_STEP,
                pose: pose(),
                drained_seq: 42,
            }),
            // D-7a/b: the transient batch ack + the DropApplied ack + the three structural handoff
            // command arms roundtrip distinctly (the dispatch split depends on the tag).
            InterShardFlow::TransferAck(TransferAck::BatchAdopted {
                transfer_id: TransferId(1),
                step_id: TRANSIENT_BATCH_STEP,
            }),
            InterShardFlow::TransferAck(TransferAck::DropApplied {
                transfer_id: TransferId(1),
                step_id: TRANSIENT_DROP_STEP,
            }),
            InterShardFlow::TransientRelease(TransientHandoff {
                transfer: TransferId(1),
                step_id: TRANSIENT_RELEASE_STEP,
                fence: Fence(6),
            }),
            InterShardFlow::TransientDrop(TransientHandoff {
                transfer: TransferId(1),
                step_id: TRANSIENT_DROP_STEP,
                fence: Fence(6),
            }),
            InterShardFlow::ReleaseComplete(TransientHandoff {
                transfer: TransferId(1),
                step_id: TRANSIENT_RELEASE_STEP,
                fence: Fence(6),
            }),
        ] {
            let bytes = postcard::to_allocvec(&flow).expect("encode");
            assert_eq!(
                postcard::from_bytes::<InterShardFlow>(&bytes).expect("decode"),
                flow
            );
        }
    }

    #[test]
    fn reject_reasons_are_typed_and_distinct() {
        let reasons = [
            TransferStepRejectReason::SpatialPrecondition,
            TransferStepRejectReason::StaleFence,
            TransferStepRejectReason::EpochMismatch,
            TransferStepRejectReason::VersionFloor,
            TransferStepRejectReason::UnknownKind,
        ];
        for (i, a) in reasons.iter().enumerate() {
            for (j, b) in reasons.iter().enumerate() {
                assert_eq!(a == b, i == j);
            }
        }
    }
}
