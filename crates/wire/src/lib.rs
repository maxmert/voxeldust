//! # vd-wire — the frozen wire contract
//!
//! Every byte that crosses a process boundary is expressible ONLY through the closed
//! type families defined here. There is no `Raw(Vec<u8>)`, no `Other`, no escape hatch (HR1).
//!
//! Two orthogonal closed taxonomies (`docs/design/sealed_shards.md` §0):
//! 1. **Connection plane** (client↔gateway↔shard world state):
//!    `ControlMsg | InputDatagram | BulkMsg | EventMsg | SnapshotDatagram`.
//! 2. **`InterShardFlow`** (shard↔shard / shard↔orchestrator):
//!    `Ghost | Transfer | Directory | BlockEdit | Coupling | Signal` — one reviewed file
//!    (`intershard.rs`), arms frozen INCREMENTALLY with their first consumer
//!    (P0: Ghost/Transfer-Durable/Directory; P3/P6: BlockEdit + Transfer-Transient;
//!    P8: Coupling; P9: Signal).
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

pub mod channels;
pub mod framing;
pub mod intershard;
pub mod seams;
pub mod version;
