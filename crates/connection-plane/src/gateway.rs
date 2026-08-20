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
//!
//! THE MODULE TREE (one lane per file; each file states what it owns and what it does not):
//! `config` the dials · `stats` the honesty counters · `session` the session table and its records ·
//! `routing` the write plane, the read plane and the egress primitives · `client` the client's own
//! two directions · `shard` what a shard says back · `transfer` the seven-phase transfer-control
//! consumer · `window_lane` the window derivation, admission and composer · `home` the dynamic-home
//! derivation · `directory` the directory replies · `liveness` the self-fence, the retries and the
//! bounded bootstrap · `register` the world build and the inbound dispatch.
//!
//! The whole surface is re-exported here, so `vd_connection_plane::gateway::X` resolves exactly as
//! it did when this was one file.

mod client;
mod config;
mod directory;
mod home;
mod liveness;
mod register;
mod routing;
mod session;
mod shard;
mod stats;
mod transfer;
mod window_lane;

#[cfg(test)]
mod tests;

pub(crate) use client::*;
pub use config::*;
pub(crate) use directory::*;
pub(crate) use home::*;
pub(crate) use liveness::*;
pub use register::*;
pub use routing::*;
pub use session::*;
pub(crate) use shard::*;
pub use stats::*;
pub(crate) use transfer::*;
pub(crate) use window_lane::*;
