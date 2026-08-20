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
//! delivery + adaptive timeouts + the CAS re-read loop land at Slice 2; the client-facing cut
//! cycle landed at Slice 1c/1e. The shard-bound `StubCrossing` state transfer is synthesized at
//! Slice 1d. The ORDERED demote-before-promote tail (1d.5b.1, D-2) pushes `Demote`→source then
//! (on `DemoteAcked`) `Promote`→dest, and releases the source sub only once the dest both acked
//! the promote AND delivered to every observer — the seamless no-vanish gate (`saga.rs` Promoting).
//!
//! ## Slice 2a — the RECOVERY MACHINERY (the R1 cure; closes D-1 + the post-commit park)
//! [`scan_deadlines`] is the production TIMEOUT PRODUCER: a `now - since >= deadline_for(phase)`
//! scan (two thresholds — cheap `redrive` for post-commit, large `abort` for the destructive
//! pre-freeze/compensation phases) that injects `SagaEvent::Timeout`, so a lost saga-ack RE-DRIVES
//! (post-commit) or ABORTS (pre-freeze) instead of PARKING forever. The terminal `Aborted` edge now
//! emits `SagaAction::ClearTransferLock` → [`DirectoryCore::abort_clear`] (the stale-fence head
//! re-read — a CAS-loser's `expected_fence` is stale by definition), so the directory lock CLEARS
//! and an aborted subject can immediately re-transfer (D-1 CLOSED; the two pinned asserts FLIPPED
//! to `None`). ⚠️ HONEST SCOPE: a fixed tick deadline is NOT a crash-vs-slow discriminator (only a
//! permanent `kill` emits `NodeUnreachable`) — a mis-tuned `abort_deadline_ticks` below a deployment's
//! worst-case healthy pre-freeze round-trip WILL false-abort a slow-but-alive saga (the real
//! discriminator is lease-lapse liveness, D-3, owed). The producer cures lost SAGA-ACKS, not a starved
//! standing delivery watermark (a `Promoting` saga whose dest frame delivery is blocked stays
//! half-open — owed, P3) nor an orchestrator CRASH (no durable WAL, D-6). (Caught by Slice-1b audit
//! CPO-1/CPO-2 + Slice-2a design `wf_9f22c70d`.)
//!
//! THE MODULE TREE (one lane per file; each file states what it owns and what it does not):
//! `store` the durable key families, the snapshots and the kill-9 rehydrate · `state` the live-saga
//! set and the queues around it · `liveness` the dead-vs-slow discriminator and the expiry reaper ·
//! `execute` the action executor and the commit of a quiescent saga · `crossing` the entity-state
//! envelope and the two crossing-request consumers · `rehome` the standing re-home · `deadlines` the
//! timeout producer · `systems` the scheduled systems and the group-commit barrier.
//!
//! The whole surface is re-exported here, so `vd_node::saga_runtime::X` resolves exactly as it did
//! when this was one file.

mod crossing;
mod deadlines;
mod execute;
mod liveness;
mod rehome;
mod state;
mod store;
mod systems;

#[cfg(test)]
mod tests;

pub(crate) use crossing::*;
pub(crate) use deadlines::*;
pub(crate) use execute::*;
pub(crate) use liveness::*;
pub(crate) use rehome::*;
pub use state::*;
pub use store::*;
pub use systems::*;
