//! # vd-wire — the frozen wire contract
//!
//! Every byte that crosses a process boundary is expressible ONLY through the closed
//! type families defined here. There is no `Raw(Vec<u8>)`, no `Other`, no escape hatch (HR1).
//!
//! Three orthogonal closed taxonomies (`docs/design/sealed_shards.md` §0; the third
//! freezes incrementally with its first consumer, like `InterShardFlow`'s arms):
//! 1. **Connection plane** (client↔gateway, the families the gateway terminates):
//!    `ControlMsg | InputDatagram | BulkMsg | EventMsg | SnapshotDatagram`.
//! 2. **`InterShardFlow`** (shard↔shard / shard↔orchestrator) — the CURRENT closed set:
//!    `Ghost | Transfer | Directory | Saga | SagaAck | DirectoryReply` — one reviewed
//!    file (`intershard.rs`), arms frozen INCREMENTALLY with their first consumer
//!    (P0: Ghost/Transfer-Durable/Directory; P2: Saga/SagaAck — the route-swap saga —
//!    plus DirectoryReply, the orchestrator→requester directory head/outcome envelope
//!    that keeps a reply distinct from a `Saga(TransferControl)` on the shared Saga
//!    class; RESERVED future arms, added under review with their phases: BlockEdit +
//!    Transfer-Transient at P3/P6, Coupling at P8, Signal at P9).
//! 3. **`session_flow`** (gateway↔shard session-scoped traffic; `session_flow.rs`):
//!    `GatewayToShard | ShardToGateway` — routed ALWAYS by in-frame `SessionId` + `Fence`
//!    (never source address, R2); the gateway forwards snapshot/input payloads as OPAQUE
//!    bytes (a header `sub_id` re-tag / `seq` peek, never a world-state decode). Distinct
//!    from #1 (the gateway terminates client families) and #2 (the gateway is not a shard).
//!
//! Also owns the three contract-tested seams (frozen before any consumer is built):
//! - `TransferControl` — the saga→gateway command/ack vocabulary,
//! - the Ownership Directory query/CAS API,
//! - `SessionTicket`/`ResumeTicket` shapes.
//!
//! Serialization: postcard v1 everywhere; the codec flag bit is reserved (always 0) so
//! bitcode can be added later without a wire break. Entity-state blobs are TLV-framed
//! (defined in `vd-core`) for skip-tolerant evolution.
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

pub mod admin;
pub mod channels;
pub mod framing;
pub mod intershard;
pub mod seams;
pub mod session_flow;
pub mod version;
