//! # vd-connection-plane — gateway internals
//!
//! The client holds exactly ONE persistent QUIC connection to a gateway for its entire
//! session; a shard transfer is a server-side route swap the client transport never
//! observes (binding decision #2; `docs/design/connection_plane.md`).
//!
//! Owns:
//! - `SessionState` / `RouteSnapshot`: the 20Hz hot path reads the route via a single
//!   `ArcSwap` load + atomics — NO async lock is ever held across a network send.
//! - The phased, separately-acked, compensatable `TransferControl` consumer
//!   (PrepareSubscribe → RequestCut → FreezeSource → CommitAuthority → ThawSource /
//!   AbortTransfer → ReleaseSubscribe) and the CUT_MARKER seq-range routing.
//! - Subscription lifecycle: monotonic never-reused `sub_id`s, per-sub BULK/EVENTS
//!   streams (opened causally after `SubscriptionOpened` — invariant X1), hysteresis +
//!   min-dwell, the chunk cache keyed by `(shard, region, edit_epoch)`.
//! - `ResumeTicket` (HMAC, single-use directory nonce, session-fence CAS adoption).
//! - BULK pacing from live RTT; snapshot datagram partitioning (one shared function).
//!
//! Everything here is written against the `vd_sim::io` seam — the quinn pool is an
//! `io-prod` implementation detail, so the whole plane is testable in-process.
//!
//! Coverage: Tier-A — 100% region + branch (HR5).

// BULK pacing / multi-sub interest land with P2 per the staged build plan.

pub mod gateway;
pub mod tickets;
