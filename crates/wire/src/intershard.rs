//! THE closed shard↔shard / shard↔orchestrator taxonomy (HR1 taxonomy 1 of 2;
//! `docs/design/sealed_shards.md` §0–§1). One reviewed file.
//!
//! Every byte crossing a shard-to-shard or shard-to-orchestrator boundary is exactly
//! one [`InterShardFlow`] arm. There is NO `Raw`, NO `Other`, NO `EcsSync` — to
//! express a new cross-shard flow you MUST add a variant here, under review.
//!
//! Arms freeze INCREMENTALLY with their first consumer (the closed-set guarantee is
//! the per-release conformance test below, not a day-one empty freeze):
//! - P0 (now): `Ghost`, `Transfer` (Durable class), `Directory`.
//! - P2: `Saga` (the saga→gateway transfer commands) + `SagaAck` (gateway→saga acks) —
//!   the route-swap saga drives the gateway exclusively through these.
//! - P3/P6: `Transfer` transient batches + `BlockEdit`.
//! - P8: `Coupling` (`EffectFree` ports). — variant reserved, payload lands with ships.
//! - P9: `Signal`.
//!
//! Effect classes (the G-SEALED invariant, enforced by `effect_class` + its test):
//! - SIDE-EFFECTING arms carry `(TransferId, step_id)` idempotency and are ack-driven.
//! - FIRE-AND-FORGET arms are effect-free or idempotently re-derivable: they may
//!   NEVER carry a transfer trigger or authority-gating discrete state.

use serde::{Deserialize, Serialize};
use vd_core::entity_kind::DurabilityClass;
use vd_core::pose::{RealmId, StampedPose};
use vd_core::{EntityId, EpochId, Fence, NodeId, TickId, TransferId};

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
            // idempotency). All FOUR share the `TransientHandoff` shape and one classification arm (DRY).
            InterShardFlow::TransientRelease(h)
            | InterShardFlow::TransientDrop(h)
            | InterShardFlow::ReleaseComplete(h)
            | InterShardFlow::TransientAbandon(h) => EffectClass::SideEffecting {
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
        }
    }
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

/// Ghost replication: kinematic mirrors that NEVER independently integrate physics.
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
            subject: DirectoryKey::Entity(eid(EntityKind::Player)),
            new_fence: Fence(6),
            step_id: RE_HOME_STEP,
            state: ReHomeState::PoseOnly(pose()),
            source: NodeId(2),
        });
        let bytes = postcard::to_allocvec(&rehome).expect("encode");
        let decoded: InterShardFlow = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, rehome.clone(), "ReHome survives a postcard roundtrip");
        assert!(format!("{rehome:?}").contains("ReHome"), "Debug renders the arm");
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
        // Every entity-STATE step (flush/crossing/demote/promote) is disjoint from the 0–6 route-swap
        // phases AND from each other — so a state step can never alias a phase in any journal.
        let state_steps = [
            FLUSH_SOURCE_STEP,
            STUB_CROSSING_STEP,
            DEMOTE_STEP,
            PROMOTE_STEP,
            TRANSIENT_BATCH_STEP,
            TRANSIENT_DROP_STEP,
            TRANSIENT_RELEASE_STEP,
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
