//! # vd-wire — the frozen wire contract
//!
//! Every byte that crosses a process boundary is expressible ONLY through the closed
//! type families defined here. There is no `Raw(Vec<u8>)`, no `Other`, no escape hatch (HR1).
//!
//! Three orthogonal closed taxonomies (`docs/design/sealed_shards.md` §0; the third
//! freezes incrementally with its first consumer, like `InterShardFlow`'s arms):
//! 1. **Connection plane** (client↔gateway, the families the gateway terminates):
//!    `ControlMsg | InputDatagram | BulkMsg | EventMsg | SnapshotDatagram`.
//! 2. **`InterShardFlow`** (shard↔shard / shard↔orchestrator) — one reviewed file
//!    (`intershard.rs`), arms frozen INCREMENTALLY with their first consumer. The CURRENT
//!    closed set is 17 arms: `Ghost | Transfer | Directory | Saga | SagaAck | DirectoryReply
//!    | FlushSource | TransferAck | Demote | Promote | TransientRelease | TransientDrop |
//!    ReleaseComplete | TransientAbandon | ReHome | TransientDiscard | ReSolicitBatch` (P0
//!    Ghost/Transfer-Durable/Directory; P2 the route-swap + 1d.1/1d.5b transfer machinery; P3 the D-7
//!    transient handoff + the D-37 `ReHome` forward-re-home adopt; R-6d3c the `TransientDiscard` D-6 #1
//!    never-restart closure; CA-1 S3 the `ReSolicitBatch` AwaitAdopt liveness probe). RESERVED future
//!    arms (added under review with their phases): `BlockEdit` (P6),
//!    `Coupling` `EffectFree` ports (P8), `Signal` (P9). See `intershard.rs`'s header for the
//!    per-phase breakdown + the G-SEALED effect-class invariant.
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
