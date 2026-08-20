//! THE REACTIVE GREETING: how a shard nobody booked becomes reachable.
//!
//! Owns: the greet-on-silence policy of a DEMAND-SPAWNED shard. Such a shard's `NodeId` was minted
//! at spawn, so no peer has its address; it makes itself reachable by speaking first, and the
//! peer's mesh learns the return connection. It greets a peer only while silent, so an active realm
//! hearing its gateway stays quiet and a dropped connection self-heals within one interval.
//!
//! Does NOT own: addressing, dialling, or any transport fact. It emits one presence message through
//! the same outbox every other lane uses; that no address is needed is the cloud-portable property
//! this lane exists to preserve. ABSENT on a static shard ⇒ the whole lane is inert.

use crate::io::{Inbound, MsgClass};
use crate::runtime::{ClockSample, InboundBox, OutboundBox};
use bevy_ecs::prelude::{Local, Res, ResMut, Resource};
use std::collections::{BTreeMap, BTreeSet};
use vd_core::{NodeId, TickId};
use vd_wire::intershard::{InterShardFlow, ShardPresence};

/// RLM 5f RG-1 — the reactive-greeting policy for a DEMAND-SPAWNED shard. A shard minted at spawn has a
/// `NodeId` its peers never booked, so it makes itself reachable by GREETING every one of them: it sends a
/// reliable [`ShardPresence`] and the peer's mesh learns the return connection (no address needed — the
/// cloud-portable property). `peers` = the shard's full booked set (gateway + ancestor chain + orchestrator,
/// from `VD_PEERS`); it greets each ONLY while SILENT (no contact for `interval_ticks`), so an active realm
/// hearing its gateway stays quiet and a gateway restart / dropped connection self-heals within one interval.
/// Present ONLY on a demand shard (the `--demand` boot inserts it, RG-2); ABSENT ⇒ [`announce_presence`]
/// no-ops ⇒ every static rig is byte-identical.
#[derive(Resource, Clone, Debug)]
pub struct PresenceAnnounce {
    /// The booked peers to greet (all of `VD_PEERS` except self).
    pub peers: BTreeSet<NodeId>,
    /// Greet a peer once it has had no contact (inbound OR a prior greeting) for this many local ticks.
    /// `>= 1` (0 would greet every tick — rejected by [`PresenceAnnounce::new`]).
    pub interval_ticks: u64,
}

impl PresenceAnnounce {
    /// Build the greeting policy, failing LOUD on a zero interval (which would greet every tick — a
    /// misconfiguration, never the intent). The peer set may be empty (an inert no-op, not an error).
    ///
    /// # Errors
    /// `interval_ticks == 0`.
    pub fn new(peers: BTreeSet<NodeId>, interval_ticks: u64) -> Result<PresenceAnnounce, String> {
        if interval_ticks == 0 {
            return Err(
                "PresenceAnnounce interval_ticks must be >= 1 (0 would greet every tick)".into(),
            );
        }
        Ok(PresenceAnnounce {
            peers,
            interval_ticks,
        })
    }
}

/// RLM 5f RG-1 — is `peer` DUE a greeting this tick? `true` when the shard has had NO contact with it
/// (never, or not for `interval` ticks). A monomorphic helper so [`announce_presence`] stays a shim (HR5):
/// the two arms (never-contacted ⇒ greet; contacted ⇒ silence check) live here, covered by both a
/// first-tick greet and a within-interval skip.
fn presence_due(
    last_contact: &BTreeMap<NodeId, TickId>,
    peer: NodeId,
    now: TickId,
    interval: u64,
) -> bool {
    match last_contact.get(&peer) {
        None => true,
        Some(seen) => now.0.saturating_sub(seen.0) >= interval,
    }
}

/// RLM 5f RG-1 — the REACTIVE GREETING. A demand-spawned shard greets every booked peer it has not heard
/// from recently, so the peer's mesh learns the return connection (making this shard — whose spawn-minted
/// `NodeId` no peer booked — reachable WITHOUT any pre-booked address; the cloud-portable property). GREET-
/// ON-SILENCE: an inbound frame from a peer OR a greeting we send it both count as contact, so an active
/// realm hearing its gateway stays quiet while an idle/warm realm greets at most once per `interval_ticks`,
/// and a gateway restart / dropped connection self-heals within one interval. UNGATED on `has_synced` /
/// `RealmAuthority` (the greeting must fire pre-sync, pre-lease — reachability precedes authority). No-ops
/// (the `else` return) when [`PresenceAnnounce`] is absent ⇒ every static rig is byte-identical. The
/// `last_contact` ledger is a per-system `Local` (RAM, rebuilt empty on boot — reachability re-heals from
/// scratch, which is correct after a restart).
pub(crate) fn announce_presence(
    presence: Option<Res<PresenceAnnounce>>,
    clock: Res<ClockSample>,
    inbox: Res<InboundBox>,
    mut last_contact: Local<BTreeMap<NodeId, TickId>>,
    mut outbox: ResMut<OutboundBox>,
) {
    let Some(presence) = presence else {
        return;
    };
    let now = clock.local_tick;
    // Any inbound frame from a peer is contact — it proves the two-way lane is live, so we need not greet.
    for msg in &inbox.0 {
        if let Inbound::Wire { from, .. } = msg {
            last_contact.insert(*from, now);
        }
    }
    // Greet each booked peer that is silent; the greeting itself counts as contact so the cadence is
    // bounded to one per `interval_ticks` even for a peer that never replies (an idle warm realm).
    for &peer in &presence.peers {
        if presence_due(&last_contact, peer, now, presence.interval_ticks) {
            outbox.push_flow(
                peer,
                MsgClass::Saga,
                &InterShardFlow::ShardPresence(ShardPresence { local_tick: now }),
            );
            last_contact.insert(peer, now);
        }
    }
}
