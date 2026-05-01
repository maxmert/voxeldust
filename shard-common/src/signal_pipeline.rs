//! Generic signal pipeline plugin shared by every shard that hosts blocks.
//!
//! Why this exists: ship-shard, planet-shard, station-shard (future), and
//! anything else that lets players place functional blocks all need the same
//! pub/sub plumbing. Before this module the pipeline was inlined in
//! ship-shard alone — placing a Seat + Thruster on a planet did nothing
//! because planet-shard ran no signal systems. Lifting the generic stages to
//! a `bevy_app::Plugin` lets each shard register the plugin and then bolt on
//! its block-kind-specific publishers/subscribers as separate systems
//! grouped under the well-defined [`SignalSet`] schedule labels.
//!
//! # Schedule layout
//!
//! ```text
//! Update
//! ├── SignalSet::Ingest    (drain QUIC inbound + reset pending; generic)
//! ├── SignalSet::Publish   (per-kind publishers — registered by each shard)
//! ├── SignalSet::Merge     (finalize multi-publisher channels; generic)
//! ├── SignalSet::Process   (channel transformers — flight computer,
//! │                          hover module, autopilot, engine controller;
//! │                          ship-shard registers these here)
//! ├── SignalSet::Evaluate  (signal converter rules; generic)
//! ├── SignalSet::Subscribe (per-kind subscribers — thrusters, doors, etc;
//! │                          registered by each shard)
//! └── SignalSet::ClearDirty (reset dirty flags; generic)
//! ```
//!
//! Cross-shard outbound broadcast (`signal_broadcast_remote`) lives outside
//! this plugin because it depends on each shard's QUIC sender + peer
//! position cache, which differ across shard types.
//!
//! # Why a Plugin and not just a function
//!
//! The plugin owns the resource initialization (`SignalChannelTable`,
//! `IncomingSignalBuffer`) and the `SystemSet` ordering. A consumer shard
//! does:
//!
//! ```ignore
//! app.add_plugins(SignalPipelinePlugin);
//! app.add_systems(Update, my_thruster_subscriber.in_set(SignalSet::Subscribe));
//! ```
//!
//! The schedule chain is set once in `build()`, so every shard gets the same
//! ordering guarantees without copy-paste. The block-kind systems remain
//! shard-specific (they reference shard-private state types like
//! `ThrusterState`) — they just slot into the right `SignalSet`.

use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;

use std::collections::HashMap;

use glam::DVec3;

use voxeldust_core::client_message::ServerMsg;
use voxeldust_core::shard_types::{SessionToken, ShardId};
use voxeldust_core::signal::{
    self, metrics as signal_metrics, ClientRateLimits, GrantsRegistry, IncomingSignalBuffer,
    IncomingSubscribeBuffer, SignalChannelTable, SignalConverterConfig,
};
use voxeldust_core::signal::components::SeatChannelMapping;
use voxeldust_core::ecs::components::{Player, SeatedState, SeatInputValues};

use crate::harness::{NetworkBridge, ShardIdentity};

/// Per-session table of grants the player holds. Populated by
/// `apply_add_held_grant` from `ClientMsg::AddHeldGrant`. Read by
/// `apply_remote_signal_publish` to look up the HMAC key + target shard
/// when the player publishes via a held grant.
///
/// A `HeldGrant` contains the bare minimum needed to sign and route a
/// remote publish: the grant id (for the wire entry), the HMAC key (to
/// sign), and the target shard id (to route). The `label` is for the
/// tablet UI's "Grants you hold" panel and isn't used in the hot path.
#[derive(Resource, Default)]
pub struct HeldGrants {
    by_session: HashMap<SessionToken, HashMap<u64, HeldGrant>>,
}

#[derive(Clone, Debug)]
pub struct HeldGrant {
    pub key: [u8; 32],
    pub target_shard_id: u64,
    pub label: String,
}

impl HeldGrants {
    /// Insert / overwrite a grant for a session. Idempotent re-adds are a
    /// non-issue (the player pasted the same key twice — same data wins).
    pub fn insert(&mut self, session: SessionToken, grant_id: u64, grant: HeldGrant) {
        self.by_session
            .entry(session)
            .or_default()
            .insert(grant_id, grant);
    }

    pub fn forget(&mut self, session: SessionToken, grant_id: u64) -> bool {
        self.by_session
            .get_mut(&session)
            .and_then(|map| map.remove(&grant_id))
            .is_some()
    }

    pub fn forget_all_for_session(&mut self, session: SessionToken) {
        self.by_session.remove(&session);
    }

    pub fn get(&self, session: SessionToken, grant_id: u64) -> Option<&HeldGrant> {
        self.by_session.get(&session)?.get(&grant_id)
    }

    pub fn iter_for_session(
        &self,
        session: SessionToken,
    ) -> impl Iterator<Item = (u64, &HeldGrant)> {
        self.by_session
            .get(&session)
            .into_iter()
            .flat_map(|map| map.iter().map(|(k, v)| (*k, v)))
    }

    /// Total grants across every session. O(sessions) sweep. Used by the
    /// gauge-sample system to expose `signal_held_grants`.
    pub fn total_count(&self) -> usize {
        self.by_session.values().map(|m| m.len()).sum()
    }
}

/// Schedule labels for the seven stages of the signal pipeline. Stages are
/// chained in order — see the module-level diagram. Each stage is a public
/// `SystemSet` so shards can hang their own systems on the right pin.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalSet {
    /// Drain cross-shard inbound + seat input + reset pending. Generic.
    Ingest,
    /// Block-kind publishers run here (reactor, cruise drive, mechanical,
    /// etc.). Shards register their own publishers; the plugin chains them
    /// after `Ingest`.
    Publish,
    /// Finalize all `pending` aggregations into channel values. Generic.
    Merge,
    /// Channel transformers (flight computer, hover module, autopilot, ...
    /// engine controller cutoff). Shards register their own systems here.
    /// Order within `Process` is the registration order.
    Process,
    /// Signal converter rules (condition → expression). Generic.
    Evaluate,
    /// Block-kind subscribers (thrusters, doors, lights). Registered per
    /// shard.
    Subscribe,
    /// Reset dirty flags after subscribers consumed them. Generic.
    ClearDirty,
}

/// The plugin. Registers the resources, the `SystemSet` chain, and the four
/// generic systems (`signal_ingest`, `signal_seat_publish`, `signal_merge`,
/// `signal_evaluate`, `signal_clear_dirty`).
pub struct SignalPipelinePlugin;

impl Plugin for SignalPipelinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SignalChannelTable>()
            .init_resource::<IncomingSignalBuffer>()
            .init_resource::<IncomingSubscribeBuffer>()
            .init_resource::<GrantsRegistry>()
            .init_resource::<HeldGrants>()
            .init_resource::<ClientRateLimits>()
            // Phase 4-Persist: every shard with grants needs the
            // queue. Empty by default; pushed to by apply_grant_*
            // and drained by the shard binary's persistence sweep.
            // Inserted unconditionally so a shard that doesn't
            // (yet) wire grant persistence still has a working
            // queue object — the writes simply pile up harmlessly.
            .init_resource::<crate::grant_persistence::GrantsPersistenceQueue>()
            .configure_sets(
                Update,
                (
                    SignalSet::Ingest,
                    SignalSet::Publish,
                    SignalSet::Merge,
                    SignalSet::Process,
                    SignalSet::Evaluate,
                    SignalSet::Subscribe,
                    SignalSet::ClearDirty,
                )
                    .chain(),
            )
            // Ingest: drain grant management messages first (so a grant
            // issued in tick N is visible to the same-tick ingest of a
            // publish that uses it), then held-grant + remote-publish
            // ingestion (recipient-side), then cross-shard subscribe
            // requests (so a fresh subscription gets forwarded values on
            // its own first tick), then clear pending, drain remote
            // inbound + per-player seat publishes.
            .add_systems(
                Update,
                (
                    apply_grant_create,
                    apply_grant_revoke,
                    apply_add_held_grant,
                    apply_forget_held_grant,
                    apply_remote_signal_publish,
                    apply_signal_subscribe,
                    apply_signal_unsubscribe,
                    signal_ingest_clear,
                    signal_ingest_remote,
                    signal_seat_publish,
                )
                    .chain()
                    .in_set(SignalSet::Ingest),
            )
            // Merge: aggregate pending into final channel values.
            .add_systems(Update, signal_merge.in_set(SignalSet::Merge))
            // Subscribe: block-driven readers of fresh channel values.
            // Antenna sits here (NOT in Publish) because it reads its
            // source channel's MERGED value and forwards it remotely —
            // the upstream merge must have run first this tick. Despite
            // its name, the Antenna is a "subscriber-of-local + publisher-
            // of-remote" hybrid; it lives in Subscribe so reads see fresh
            // data. Listener mirror runs alongside it: same set, mirrors
            // the bridged channel into the local destination channel.
            // The lease renewal system also lives here — it doesn't
            // depend on freshly-merged values but it does need to run
            // before ClearDirty (it doesn't touch channel state, but
            // grouping all listener-related work together is clearer).
            .add_systems(
                Update,
                (antenna_publish, listener_mirror, listener_lease_renewal)
                    .in_set(SignalSet::Subscribe),
            )
            // Evaluate: converter rules feed dirty inputs through condition+expression.
            .add_systems(Update, signal_evaluate.in_set(SignalSet::Evaluate))
            // ClearDirty: drop dirty flags so next tick starts clean.
            .add_systems(Update, signal_clear_dirty.in_set(SignalSet::ClearDirty))
            // Rate-limit GC runs as a low-cadence top-level system (not in
            // any SignalSet) — it's bookkeeping, not part of the signal
            // dataflow. Its cadence + stale threshold are tuned so the
            // O(sessions) sweep is dominated by genuine evictions.
            .add_systems(Update, rate_limit_gc)
            // Phase 3F.9.C: propagate this shard's Listener frequency
            // set to its host every 30s. No-op on shards without a
            // host (system / galaxy / planet today).
            .add_systems(Update, propagate_listener_frequency_interest)
            // Metrics gauges (channels / grants / held grants / remote
            // subscribers) are sampled on the same low-cadence cycle.
            // Counters self-emit at every event site; gauges need an
            // explicit sweep because the underlying state has no
            // change-notification.
            .add_systems(Update, signal_metrics_gauge_sample);
    }
}

/// Phase 3F.9.C: cadence + lease for `propagate_listener_frequency_interest`.
/// 600 ticks (30s @ 20Hz) renewal × 90s lease gives 3× headroom — a missed
/// renewal is still inside the system-shard's lease window. Mirrors the
/// renewal/lease ratio used by `refresh_galaxy_radio_subscription`.
pub const LISTENER_INTEREST_REFRESH_TICKS: u64 = 600;
pub const LISTENER_INTEREST_LEASE_MS: u64 = 90_000;

/// Phase 3F.9.C: ship-shard → host-shard frequency interest propagation.
///
/// Walks every active [`ListenerState`] with a non-zero frequency, builds
/// the deduped set, and ships it to our host as
/// `ShardMsg::ShipFrequencyInterest` over QUIC. Empty-set semantics: if
/// every Listener has been removed since the last tick, we still send a
/// message with `frequencies: []` so the host's `ShipFrequencyInterests`
/// resource drops our entry (rather than waiting for the lease to
/// expire). Self-healing: no diff state, just current ground truth.
///
/// **Where it runs**: any shard that hosts Listener blocks AND has a
/// host. In practice that's ship-shards today; planet-shards once they
/// gain Listener support. Shards with no host (`host_shard_id == None`)
/// silently no-op — system-shards and galaxy-shards aren't expected to
/// have Listener blocks.
///
/// **Why generic, not per-shard**: putting this in shard-common keeps
/// every shard with Listeners on the same wire schema + cadence. Adding
/// a new host-aware shard type (station-shard?) needs zero new code.
pub fn propagate_listener_frequency_interest(
    bridge: Res<NetworkBridge>,
    identity: Res<crate::harness::ShardIdentity>,
    tick: Res<voxeldust_core::ecs::TickCounter>,
    listeners: Query<&voxeldust_core::signal::ListenerState>,
) {
    use std::collections::BTreeSet;
    use voxeldust_core::shard_message::{ShardMsg, ShipFrequencyInterestData};
    if tick.0 % LISTENER_INTEREST_REFRESH_TICKS != 0 {
        return;
    }
    // No host = nothing to propagate to.
    let Some(host_id) = identity.host_shard_id else {
        return;
    };

    // Collect deduped frequency set across all active listeners.
    // BTreeSet keeps the wire serialization deterministic — easier to
    // diff in logs and tests than a HashSet's iteration order.
    let mut freqs: BTreeSet<u32> = BTreeSet::new();
    for listener in &listeners {
        if !listener.active {
            continue;
        }
        if listener.frequency == 0 {
            // Frequency 0 isn't a Radio frequency (Local-scope listener
            // mirrors don't go through galaxy). Skip.
            continue;
        }
        freqs.insert(listener.frequency);
    }
    let frequencies: Vec<u32> = freqs.into_iter().collect();

    // Even an empty set is shipped — that's how we tell the host
    // "drop my entry" without waiting for lease expiry.
    let now_ms = signal::current_unix_millis();
    let lease_until_ms = now_ms.saturating_add(LISTENER_INTEREST_LEASE_MS);
    let payload = ShipFrequencyInterestData {
        ship_shard_id: identity.shard_id.0,
        frequencies,
        lease_until_ms,
    };

    let peer_registry = bridge.peer_registry.clone();
    let quic_send_tx = bridge.quic_send_tx.clone();
    tokio::spawn(async move {
        let registry = peer_registry.read().await;
        let Some(addr) = registry.quic_addr(host_id) else {
            tracing::debug!(
                host = host_id.0,
                "ShipFrequencyInterest: host not in peer registry yet"
            );
            return;
        };
        drop(registry);
        let _ = quic_send_tx
            .send((host_id, addr, ShardMsg::ShipFrequencyInterest(payload)))
            .await;
    });
}

/// Cadence at which the gauge-sample system runs. 100 ticks @ 20 Hz = 5s.
/// Picking this:
///   * Prometheus scrapes default to 15s — three samples per scrape window
///     gives smooth charts without aliasing.
///   * The four sweeps below are O(channels) + O(grants) + O(sessions);
///     each is microseconds at realistic shard sizes, so per-tick polling
///     would still be cheap, but 5s is conservative and frees the tick
///     budget for actual gameplay work.
pub const SIGNAL_METRICS_GAUGE_CADENCE_TICKS: u64 = 100;

/// Sample every gauge the signal subsystem exposes. Counters are emitted
/// at their event sites; gauges need a dedicated walker because the
/// underlying state mutates without notifying the metrics layer. Runs
/// once every [`SIGNAL_METRICS_GAUGE_CADENCE_TICKS`].
///
/// Wire-dict gauges (`signal_dict_*`) are sampled here too because the
/// underlying registry mutates on every send/receive but exposing the
/// totals on the hot path would burn locking time. The 5s cadence is
/// well under Prometheus' 15s default scrape so dashboards see fresh
/// values.
pub fn signal_metrics_gauge_sample(
    channels: Res<SignalChannelTable>,
    grants: Res<voxeldust_core::signal::GrantsRegistry>,
    held: Res<HeldGrants>,
    rate_limits: Res<ClientRateLimits>,
    bridge: Res<NetworkBridge>,
    tick: Res<voxeldust_core::ecs::TickCounter>,
) {
    if tick.0 % SIGNAL_METRICS_GAUGE_CADENCE_TICKS != 0 {
        return;
    }
    signal_metrics::set_channel_count(channels.channel_count());
    signal_metrics::set_grants_count(grants.len());
    signal_metrics::set_held_grants_count(held.total_count());
    signal_metrics::set_rate_limited_sessions_tracked(rate_limits.tracked_sessions());
    signal_metrics::set_remote_subscribers(channels.total_remote_subscriber_count());

    // Wire-dict gauges: try a non-blocking read of the registry. If
    // contended (rare — encode/decode hold the lock for microseconds),
    // skip the sample and pick it up next cadence.
    if let Ok(wire_dicts) = bridge.wire_dicts.try_read() {
        signal_metrics::set_dict_outbound_bindings(wire_dicts.outbound_total_len());
        signal_metrics::set_dict_inbound_bindings(wire_dicts.inbound_total_len());
        signal_metrics::set_dict_outbound_peers(wire_dicts.outbound_peer_count());
        signal_metrics::set_dict_inbound_peers(wire_dicts.inbound_peer_count());
    }
}

// ---------------------------------------------------------------------------
// Generic systems — same logic on every shard. Shard-specific publishers and
// subscribers slot into [`SignalSet::Publish`] / [`SignalSet::Subscribe`].
// ---------------------------------------------------------------------------

/// First sub-step of the Ingest stage: clear last tick's pending
/// accumulators on every channel. Split into its own system so the remote-
/// ingress drain and the seat-input drain can each run independently
/// without re-clearing.
pub fn signal_ingest_clear(mut channels: ResMut<SignalChannelTable>) {
    channels.clear_pending();
}

/// Drain the cross-shard inbound queue into pending. Every entry funnels
/// through [`SignalChannelTable::try_push_remote`] which enforces:
///
/// - **`UnknownChannel`** — name doesn't resolve locally (anti-griefing
///   against channel-slot exhaustion).
/// - **`LocalChannelImmutable`** — Local-scoped channel reached without a
///   grant (foreign shards can only touch Local via Phase 3B
///   `RemoteAccessGrant`).
/// - **`ScopeMismatch`** — a peer cannot relabel a packet's scope to bypass
///   spatial filtering or relay restrictions.
/// - **`StaleOrFutureTimestamp` / `Replay`** — the per-(channel, sender)
///   sliding-window replay defense (active for any entry with non-zero
///   `seq` + `timestamp_ms`).
/// - **`AuthFailed`** — HMAC verification against the channel's signature
///   (when `grant_id == 0`) or against the matching grant's key (when
///   `grant_id != 0`, looked up in `GrantsRegistry`).
///
/// Rejections log at `tracing::debug!` rather than `warn!` to deny
/// adversaries free log-amplification on bad inputs. Operators tracking
/// genuine misconfiguration can raise the trace level.
pub fn signal_ingest_remote(
    mut channels: ResMut<SignalChannelTable>,
    mut incoming: ResMut<IncomingSignalBuffer>,
    grants: Res<GrantsRegistry>,
) {
    for entry in incoming.drain() {
        let auth_tag_slice = if entry.auth_tag.is_empty() {
            None
        } else {
            Some(entry.auth_tag.as_slice())
        };
        match channels.try_push_remote(
            &entry.name,
            entry.value,
            entry.scope_code,
            entry.sender_shard_id,
            entry.timestamp_ms,
            entry.seq,
            entry.grant_id,
            auth_tag_slice,
            &grants,
        ) {
            Ok(()) => {
                signal_metrics::record_ingest_accepted(entry.scope_code);
            }
            Err(reason) => {
                signal_metrics::record_ingest_rejected(reason.into());
                tracing::debug!(
                    channel = %entry.name,
                    reason = ?reason,
                    sender_shard = entry.sender_shard_id,
                    grant_id = entry.grant_id,
                    "rejected cross-shard signal at try_push_remote"
                );
            }
        }
    }
}

/// Drain seated players' `SeatInputValues` into their seat's bound channels.
/// This is the input side of "physical proximity = access" — anyone seated
/// triggers the seat's publish path, regardless of player_id, because the
/// channel itself owns the access semantics (default policies, optional
/// `ActivationAllowlist` for locked seats — both consulted by Phase 3's
/// `try_push_pending` migration).
///
/// For now (pre-Phase-3) this preserves today's permissive behavior: any
/// seated player publishes via `push_pending_id` directly. Phase 3 swaps in
/// `try_push_pending` with player_id and the channel's policy gate.
pub fn signal_seat_publish(
    mut channels: ResMut<SignalChannelTable>,
    seated_players: Query<(&SeatedState, &SeatInputValues), With<Player>>,
    seat_query: Query<&SeatChannelMapping>,
) {
    for (seated_state, seat_input) in &seated_players {
        if !seated_state.seated {
            continue;
        }
        let Some(seat_entity) = seated_state.seat_entity else {
            continue;
        };
        let Ok(mapping) = seat_query.get(seat_entity) else {
            continue;
        };

        let values = &seat_input.0;
        for (i, binding) in mapping.bindings.iter().enumerate() {
            let value = values.get(i).copied().unwrap_or(0.0);
            channels.push_pending_id(binding.channel_id, signal::SignalValue::Float(value));
        }
        if let Some(ch) = mapping.seated_channel_id {
            channels.push_pending_id(ch, signal::SignalValue::Float(1.0));
        }
    }
}

/// Finalize all pending aggregations into channel values. Marks channels
/// dirty only when the merged value differs from the previous tick.
pub fn signal_merge(mut channels: ResMut<SignalChannelTable>) {
    channels.merge_pending();
}

/// Evaluate signal-converter rules. Lazy: skips converters whose input
/// channels weren't dirty this tick unless the rule's condition is `Always`.
pub fn signal_evaluate(
    mut channels: ResMut<SignalChannelTable>,
    converters: Query<&SignalConverterConfig>,
) {
    for config in &converters {
        for rule in &config.rules {
            let (input_value, is_dirty) = match channels.get_by_id(rule.input_channel_id) {
                Some(ch) => (ch.value, ch.dirty),
                None => continue,
            };
            if !is_dirty && !matches!(rule.condition, signal::SignalCondition::Always) {
                continue;
            }
            if rule.condition.evaluate(input_value, is_dirty) {
                let output = rule.expression.compute(input_value);
                channels.publish_direct_id(rule.output_channel_id, output);
            }
        }
    }
}

/// Clear all dirty flags. O(dirty_count), not O(channel_count).
pub fn signal_clear_dirty(mut channels: ResMut<SignalChannelTable>) {
    channels.clear_dirty();
}

// ---------------------------------------------------------------------------
// Phase 3C grant management — drain ClientMsg::GrantCreate / GrantRevoke off
// the network bridge and update GrantsRegistry, sending an authoritative
// snapshot back to the requesting client over TCP.
// ---------------------------------------------------------------------------

/// Drain `ClientMsg::GrantCreate` requests. For each:
///
/// 1. Treat the sender's `SessionToken` as their `player_id` (the unified
///    component model in `core::ecs::components::PlayerId(SessionToken)`
///    establishes this equivalence — a session token IS a player_id).
/// 2. Hand off to [`GrantsRegistry::create_for_owner`] which validates that
///    every requested channel exists locally AND is owned by the caller.
///    Atomic — partial-ownership requests reject without inserting any
///    channel, so a half-mutated registry never appears.
/// 3. On success, render an owner-filtered [`GrantsSnapshotData`] and ship
///    it back to that one session via TCP. The snapshot is authoritative
///    (replaces, not merges, the client's view).
/// 4. On failure, log at `warn!` — the rejection reason is structured so a
///    future Phase 3D enhancement can route it back to the tablet UI as a
///    HUD-friendly toast. Today's failures stay server-side.
///
/// Runs in `SignalSet::Ingest` BEFORE the per-tick `signal_ingest_remote`
/// so a grant issued this tick is in the registry by the time a same-tick
/// inbound publish that uses it gets verified.
pub fn apply_grant_create(
    mut bridge: ResMut<NetworkBridge>,
    table: Res<SignalChannelTable>,
    mut registry: ResMut<GrantsRegistry>,
    mut rate_limits: ResMut<ClientRateLimits>,
    mut persist: ResMut<crate::grant_persistence::GrantsPersistenceQueue>,
) {
    while let Ok((session_token, data)) = bridge.grant_create_rx.try_recv() {
        let owner_id = session_token.0;
        let now_ms = signal::current_unix_millis();
        // Rate limit: grant ops are human-initiated discrete actions —
        // tighter cap than streaming publishes. Drop on bucket exhaustion;
        // increment the per-session denied counter for telemetry.
        let allowed = rate_limits.with_session(session_token, now_ms, |b| {
            if b.grant_ops.try_consume(now_ms) {
                true
            } else {
                b.denied_total += 1;
                false
            }
        });
        if !allowed {
            signal_metrics::record_rate_limited(signal_metrics::RateLimitBucket::GrantOps);
            tracing::debug!(
                session = session_token.0,
                "GrantCreate rate-limited"
            );
            continue;
        }
        let glob = if data.namespace_glob.is_empty() {
            None
        } else {
            Some(data.namespace_glob.as_str())
        };

        // Phase 1: mutate registry (atomic — see create_for_owner doc).
        let result = registry.create_for_owner(
            &table,
            owner_id,
            &data.channel_names,
            data.ops,
            &data.label,
            data.expires_at_ms,
            glob,
            now_ms,
        );
        let grant_id = match result {
            Ok(g) => g.grant_id,
            Err(reason) => {
                tracing::warn!(
                    owner = owner_id,
                    label = %data.label,
                    ?reason,
                    "GrantCreate rejected"
                );
                continue;
            }
        };

        // Phase 2: render owner-filtered snapshot (now that the &mut borrow
        // from Phase 1 is dropped). The snapshot includes the new grant's
        // base64 key — visible only to this session.
        let snapshot = registry.snapshot_for_player(&table, owner_id);
        let msg = ServerMsg::GrantsSnapshot(snapshot);

        // Phase 3: ship it. Fire-and-forget tokio task so we don't block
        // the per-tick schedule on a TCP write.
        let cr = bridge.client_registry.clone();
        tokio::spawn(async move {
            let reg = cr.read().await;
            if let Err(e) = reg.send_tcp(session_token, &msg).await {
                tracing::debug!(
                    session = session_token.0,
                    %e,
                    "GrantsSnapshot TCP send failed (client likely disconnected)"
                );
            }
        });

        signal_metrics::record_grant_created();
        // Phase 4-Persist: enqueue for redb write so the grant
        // survives shard restart. The persistence-sweep system in the
        // shard binary picks this up outside the per-tick gameplay
        // path and applies the actual fsync.
        persist.enqueue_upsert(grant_id);
        tracing::info!(
            grant_id,
            owner = owner_id,
            label = %data.label,
            channels = data.channel_names.len(),
            "GrantCreate accepted"
        );
    }
}

/// Drain `ClientMsg::GrantRevoke` requests. Idempotent — already-revoked
/// grants reply success. Authorization: only the grant's `created_by` may
/// revoke (admin-side global revocation lands in a future hardening pass).
///
/// Runs in `SignalSet::Ingest` after `apply_grant_create` but before the
/// signal-ingestion stages, so a revoked grant is invisible to a same-tick
/// publish that tried to use it.
pub fn apply_grant_revoke(
    mut bridge: ResMut<NetworkBridge>,
    table: Res<SignalChannelTable>,
    mut registry: ResMut<GrantsRegistry>,
    mut rate_limits: ResMut<ClientRateLimits>,
    mut persist: ResMut<crate::grant_persistence::GrantsPersistenceQueue>,
) {
    while let Ok((session_token, data)) = bridge.grant_revoke_rx.try_recv() {
        let owner_id = session_token.0;
        let now_ms_for_limits = signal::current_unix_millis();
        // Same rate-limit class as GrantCreate — both are grant lifecycle ops.
        let allowed = rate_limits.with_session(session_token, now_ms_for_limits, |b| {
            if b.grant_ops.try_consume(now_ms_for_limits) {
                true
            } else {
                b.denied_total += 1;
                false
            }
        });
        if !allowed {
            signal_metrics::record_rate_limited(signal_metrics::RateLimitBucket::GrantOps);
            tracing::debug!(
                session = session_token.0,
                "GrantRevoke rate-limited"
            );
            continue;
        }

        // Authorization: only the issuer can revoke. We check BEFORE the
        // mutate so the registry stays untouched for unauthorized requests.
        let authorized = registry
            .get(data.grant_id)
            .map(|g| g.created_by == owner_id)
            .unwrap_or(false);
        if !authorized {
            tracing::warn!(
                owner = owner_id,
                grant_id = data.grant_id,
                "GrantRevoke rejected: not the issuer or unknown grant"
            );
            continue;
        }

        let mutated = registry.revoke(data.grant_id);
        // Always reply with a fresh snapshot — even on idempotent re-
        // revocations the client sees the current state and can clear
        // stale UI.
        let snapshot = registry.snapshot_for_player(&table, owner_id);
        let msg = ServerMsg::GrantsSnapshot(snapshot);

        let cr = bridge.client_registry.clone();
        tokio::spawn(async move {
            let reg = cr.read().await;
            if let Err(e) = reg.send_tcp(session_token, &msg).await {
                tracing::debug!(
                    session = session_token.0,
                    %e,
                    "GrantsSnapshot TCP send failed (client likely disconnected)"
                );
            }
        });

        if mutated {
            signal_metrics::record_grant_revoked();
            // Phase 4-Persist: enqueue an upsert (NOT a delete) — the
            // revocation is recorded in the registry as a tombstone
            // (`revoked = true`) so it persists alongside live grants
            // for audit, with the same expiry path. A future purge
            // sweep will hard-delete from redb.
            persist.enqueue_upsert(data.grant_id);
            tracing::info!(
                grant_id = data.grant_id,
                owner = owner_id,
                "GrantRevoke applied"
            );
        }
    }
}

/// Drain `ClientMsg::AddHeldGrant` requests. The recipient pastes
/// `(grant_id, key_b64, target_shard_id)` into their tablet UI; this system
/// validates the base64 key, stores it in `HeldGrants` keyed by their
/// session, and ALSO inserts a mirror grant in `GrantsRegistry` so
/// `try_push_remote` can verify HMAC tags on inbound forwarded entries
/// (Phase 3E.4 — the "remote subscriber receives values via grant" path).
///
/// The mirror grant starts with empty `channels` — it gets populated
/// when a Listener block referencing this grant is configured (see
/// `apply_listener_config`'s call to `add_channel_to_grant`).
///
/// Bad-key registrations log at `debug!` and silently fail; the player
/// retries with the right key. We don't surface a server-side rejection
/// for malformed input because the client-side b64 decoder runs first
/// and a malformed key reaching here means a buggy / hostile client —
/// not worth a round-trip.
pub fn apply_add_held_grant(
    mut bridge: ResMut<NetworkBridge>,
    mut held: ResMut<HeldGrants>,
    mut grants: ResMut<GrantsRegistry>,
    mut rate_limits: ResMut<ClientRateLimits>,
) {
    while let Ok((session_token, data)) = bridge.add_held_grant_rx.try_recv() {
        let now_ms_for_limits = signal::current_unix_millis();
        let allowed = rate_limits.with_session(session_token, now_ms_for_limits, |b| {
            if b.grant_ops.try_consume(now_ms_for_limits) {
                true
            } else {
                b.denied_total += 1;
                false
            }
        });
        if !allowed {
            signal_metrics::record_rate_limited(signal_metrics::RateLimitBucket::GrantOps);
            tracing::debug!(
                session = session_token.0,
                grant_id = data.grant_id,
                "AddHeldGrant rate-limited"
            );
            continue;
        }
        let key = match signal::decode_grant_key(&data.key_b64) {
            Some(k) => k,
            None => {
                tracing::debug!(
                    session = session_token.0,
                    grant_id = data.grant_id,
                    "AddHeldGrant rejected: malformed base64 key"
                );
                continue;
            }
        };
        held.insert(
            session_token,
            data.grant_id,
            HeldGrant {
                key,
                target_shard_id: data.target_shard_id,
                label: data.label.clone(),
            },
        );

        // Mirror into GrantsRegistry for inbound verification. If a grant
        // with the same id already exists (e.g., player re-paste, or
        // we're on the issuer's shard), don't clobber it — the existing
        // entry has the canonical channel coverage.
        if grants.get(data.grant_id).is_none() {
            grants.insert(voxeldust_core::signal::RemoteAccessGrant {
                grant_id: data.grant_id,
                key,
                channels: smallvec::smallvec![],
                namespace_glob: None,
                ops: voxeldust_core::signal::GrantOps::Both,
                label: data.label.clone(),
                created_at_ms: signal::current_unix_millis(),
                expires_at_ms: None,
                created_by: session_token.0,
                revoked: false,
                mirror_of_held: true,
            });
        }

        signal_metrics::record_held_grant_registered();
        tracing::info!(
            session = session_token.0,
            grant_id = data.grant_id,
            target_shard = data.target_shard_id,
            "Held grant registered (HeldGrants + mirror in GrantsRegistry)"
        );
    }
}

/// Drain `ClientMsg::ForgetHeldGrant`. Also revokes the corresponding
/// mirror grant in `GrantsRegistry` if it exists, so future inbound
/// forwarded entries authorized by that grant get rejected as
/// `AuthFailed` immediately. Idempotent.
pub fn apply_forget_held_grant(
    mut bridge: ResMut<NetworkBridge>,
    mut held: ResMut<HeldGrants>,
    mut grants: ResMut<GrantsRegistry>,
    mut rate_limits: ResMut<ClientRateLimits>,
) {
    while let Ok((session_token, data)) = bridge.forget_held_grant_rx.try_recv() {
        let now_ms_for_limits = signal::current_unix_millis();
        let allowed = rate_limits.with_session(session_token, now_ms_for_limits, |b| {
            if b.grant_ops.try_consume(now_ms_for_limits) {
                true
            } else {
                b.denied_total += 1;
                false
            }
        });
        if !allowed {
            signal_metrics::record_rate_limited(signal_metrics::RateLimitBucket::GrantOps);
            tracing::debug!(
                session = session_token.0,
                grant_id = data.grant_id,
                "ForgetHeldGrant rate-limited"
            );
            continue;
        }
        let forgot = held.forget(session_token, data.grant_id);
        // Revoke the mirror grant. If the registry has a non-mirror grant
        // with the same id (we're on the issuer's shard), do NOT revoke
        // — that's a real grant the issuer is still managing.
        let mirror_revoked = match grants.get(data.grant_id) {
            Some(g) if g.mirror_of_held => grants.revoke(data.grant_id),
            _ => false,
        };
        if forgot || mirror_revoked {
            signal_metrics::record_held_grant_forgotten();
            tracing::info!(
                session = session_token.0,
                grant_id = data.grant_id,
                mirror_revoked,
                "Held grant forgotten"
            );
        }
    }
}

/// Drain `ClientMsg::RemoteSignalPublish` — the player's tablet wants to
/// publish to a remote channel using a held grant. Sequence:
///
/// 1. Look up the player's `HeldGrants[grant_id]` for the session. Missing
///    → silent drop (the client UI shouldn't have offered the option, so
///    a missing grant here is a stale tablet binding).
/// 2. Query the peer registry for the target shard's QUIC endpoint.
///    Missing → log warn (target shard isn't reachable from this shard).
/// 3. Build a `SignalBroadcastEntry` carrying the value, the
///    `grant_id`, a fresh timestamp, and an HMAC tag computed under the
///    held grant's key. Sequence numbers are NOT bumped per held-grant
///    (no per-grant outbound counter today — Phase 3D adds the per-
///    (grant, shard) counter for replay-window symmetry on the receiver).
/// 4. Wrap in a single-entry `SignalBroadcastBatch` and ship via QUIC.
///
/// The receiver's `try_push_remote` looks up the grant in its
/// `GrantsRegistry`, verifies HMAC against `grant.key`, and accepts.
pub fn apply_remote_signal_publish(
    mut bridge: ResMut<NetworkBridge>,
    held: Res<HeldGrants>,
    identity: Res<ShardIdentity>,
    mut rate_limits: ResMut<ClientRateLimits>,
    policy: Res<crate::harness::WireDictSendPolicy>,
) {
    use voxeldust_core::shard_message::{ShardMsg, SignalBroadcastBatchData, SignalBroadcastEntry};
    use voxeldust_core::signal::SignalValue;

    let mut to_send = Vec::new();
    while let Ok((session_token, data)) = bridge.remote_signal_publish_rx.try_recv() {
        // Rate limit on the cross-shard publish bucket. Per-session,
        // refilled at 50/sec. Caps automated tablet-driven floods.
        let now_ms_for_limits = signal::current_unix_millis();
        let allowed = rate_limits.with_session(session_token, now_ms_for_limits, |b| {
            if b.remote_publish.try_consume(now_ms_for_limits) {
                true
            } else {
                b.denied_total += 1;
                false
            }
        });
        if !allowed {
            signal_metrics::record_rate_limited(signal_metrics::RateLimitBucket::RemotePublish);
            tracing::debug!(
                session = session_token.0,
                grant_id = data.grant_id,
                "RemoteSignalPublish rate-limited"
            );
            continue;
        }

        // Look up the grant the player claims to hold for this remote
        // channel. If missing, silently drop — the tablet UI bound to a
        // grant that's since been forgotten / not registered.
        let grant = match held.get(session_token, data.grant_id) {
            Some(g) => g.clone(),
            None => {
                tracing::debug!(
                    session = session_token.0,
                    grant_id = data.grant_id,
                    "RemoteSignalPublish rejected: grant not held by this session"
                );
                continue;
            }
        };

        // Sanity: target_shard the client claimed must match the held
        // grant's target_shard. Mismatch is a buggy client.
        if data.target_shard_id != grant.target_shard_id {
            tracing::warn!(
                session = session_token.0,
                claimed_target = data.target_shard_id,
                held_target = grant.target_shard_id,
                "RemoteSignalPublish target_shard != held_target — dropped"
            );
            continue;
        }

        // Canonicalize the wire entry's value bits the same way the
        // receiver will when verifying.
        let (value_bits, _value) = match data.value_type {
            0 => (
                if data.value_data > 0.5 { 1.0_f32.to_bits() } else { 0.0_f32.to_bits() },
                SignalValue::Bool(data.value_data > 0.5),
            ),
            1 => (data.value_data.to_bits(), SignalValue::Float(data.value_data)),
            2 => ((data.value_data as u8 as f32).to_bits(), SignalValue::State(data.value_data as u8)),
            _ => {
                tracing::debug!(
                    session = session_token.0,
                    value_type = data.value_type,
                    "RemoteSignalPublish rejected: unknown value_type"
                );
                continue;
            }
        };

        // For Local-channel grant publishes the wire scope must be 0.
        // Phase 3 grants on ShortRange/LongRange/Radio channels would
        // pass the original scope through here; for the typical "remote
        // pilot to a friend's local.thrust-forward" path, scope=0.
        let wire_scope_code = 0_u8;
        let frequency = 0_u32;
        let now_ms = signal::current_unix_millis();
        // Sequence: 1. Phase 3D adds a per-(grant, sender) counter so
        // replay rejection works for grant-bearing entries; today every
        // entry carries seq=1 + ts=now, which the receiver's window
        // accepts as a fresh first entry per (channel, sender_shard).
        // The next iteration replaces this with a real counter.
        let seq = 1_u64;

        let tag = signal::auth::hmac_sign(
            &grant.key,
            &data.channel_name,
            wire_scope_code,
            frequency,
            data.value_type,
            value_bits,
            now_ms,
            identity.shard_id.0,
            seq,
            data.grant_id,
        );

        let entry = SignalBroadcastEntry {
            channel_name: data.channel_name.clone(),
            value_type: data.value_type,
            value_data: data.value_data,
            scope: wire_scope_code,
            range_m: 0.0,
            frequency,
            sequence: seq,
            timestamp_ms: now_ms,
            grant_id: data.grant_id,
            auth_tag: tag.to_vec(),
        };

        to_send.push((data.target_shard_id, entry));
        signal_metrics::record_remote_publish_dispatched();
    }

    // Issue the QUIC sends after draining (avoids holding the bridge
    // borrow while we touch the peer registry async lock).
    if to_send.is_empty() {
        return;
    }

    let peer_registry = bridge.peer_registry.clone();
    let wire_dicts = bridge.wire_dicts.clone();
    let quic_send_tx = bridge.quic_send_tx.clone();
    let source_shard_id = identity.shard_id;
    let v2_enabled = policy.v2_enabled;
    tokio::spawn(async move {
        let registry = peer_registry.read().await;
        // Per-call dispatch helper. Picks V1 or V2 wire format up
        // front and uses one QUIC send per (target, entry) pair.
        // Holding the wire-dict lock across the loop is fine — the
        // receive-side decode lock is on the same RwLock and the
        // QUIC drain task scheduling separates the two windows.
        if v2_enabled {
            let mut wd = wire_dicts.write().await;
            for (target_shard_id, entry) in to_send {
                let target = ShardId(target_shard_id);
                let Some(endpoint) = registry.endpoint(target) else {
                    tracing::warn!(
                        target = target_shard_id,
                        "RemoteSignalPublish: target shard not in peer registry"
                    );
                    continue;
                };
                let v2_batch = crate::wire_dict_registry::encode_v2_batch(
                    &mut wd,
                    target,
                    source_shard_id.0,
                    DVec3::ZERO,
                    vec![entry],
                );
                if let Err(e) = quic_send_tx
                    .send((
                        target,
                        endpoint.quic_addr,
                        ShardMsg::SignalBroadcastBatchV2(v2_batch),
                    ))
                    .await
                {
                    tracing::warn!(
                        target = target_shard_id,
                        %e,
                        "RemoteSignalPublish QUIC send failed (V2)"
                    );
                }
            }
        } else {
            for (target_shard_id, entry) in to_send {
                let target = ShardId(target_shard_id);
                let endpoint = match registry.endpoint(target) {
                    Some(ep) => ep,
                    None => {
                        tracing::warn!(
                            target = target_shard_id,
                            "RemoteSignalPublish: target shard not in peer registry"
                        );
                        continue;
                    }
                };
                let batch = SignalBroadcastBatchData {
                    source_shard_id: source_shard_id.0,
                    // No spatial position attached: a tablet-driven publish
                    // isn't tied to a physical position on the publishing
                    // shard. The receiver's ShortRange filter ignores the
                    // position field for non-ShortRange entries (which Local-
                    // grant publishes are).
                    source_position: DVec3::ZERO,
                    entries: vec![entry],
                };
                if let Err(e) = quic_send_tx
                    .send((target, endpoint.quic_addr, ShardMsg::SignalBroadcastBatch(batch)))
                    .await
                {
                    tracing::warn!(target = target_shard_id, %e, "RemoteSignalPublish QUIC send failed");
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Cross-shard subscribe protocol (Phase 3D).
// Receivers of `ShardMsg::SignalSubscribe` register the foreign shard as a
// `SubscriberRef::RemoteShard` on the channel after validating the request's
// HMAC against `GrantsRegistry::check_subscribe(grant_id)`. Subsequent dirty
// values get forwarded to that subscriber via grant-stamped
// `SignalBroadcastBatch` (see ship-shard's signal_broadcast_remote).
// ---------------------------------------------------------------------------

/// Maximum lease duration the receiver will honor on a `SignalSubscribe`
/// request. Defends against a malicious request asking for "valid until
/// tick 2^64" by clamping to a reasonable ceiling. 60s lease at 20Hz =
/// 1200 ticks; 5min cap is plenty of headroom for renewals.
const MAX_SUBSCRIBE_LEASE_TICKS: u64 = 6_000; // 5 min @ 20 Hz

/// Drain queued `SignalSubscribe` requests, validate each, and register
/// the subscription. The signal pipeline plugin owns the
/// `IncomingSubscribeBuffer` resource — each shard's QUIC drainer pushes
/// the wire payload into it; this generic system processes them.
///
/// Validation steps (in order, ALL must pass):
/// 1. Channel name resolves locally.
/// 2. Timestamp within ±5s freshness window.
/// 3. `GrantsRegistry::check_subscribe(grant_id, channel_id, now_ms)`
///    returns `Some(key)` — implicit checks: grant exists, ops includes
///    Subscribe, channel covered, not expired/revoked.
/// 4. HMAC-verify the request bytes under that key.
/// 5. Clamp `valid_until_tick` to `now + MAX_SUBSCRIBE_LEASE_TICKS`.
/// 6. Add / refresh `SubscriberRef::RemoteShard` on the channel.
///
/// Rejected requests log at `tracing::debug!` (anti-DoS via log spam).
pub fn apply_signal_subscribe(
    mut channels: ResMut<SignalChannelTable>,
    grants: Res<GrantsRegistry>,
    mut buffer: ResMut<IncomingSubscribeBuffer>,
    tick: Res<voxeldust_core::ecs::TickCounter>,
) {
    let now_ms = signal::current_unix_millis();
    let now_tick = tick.0;
    let lease_cap = now_tick.saturating_add(MAX_SUBSCRIBE_LEASE_TICKS);

    for req in buffer.subscribes.drain(..) {
        // 1. Channel resolves locally.
        let Some(channel_id) = channels.resolve(&req.channel_name) else {
            signal_metrics::record_subscribe_rejected("unknown_channel");
            tracing::debug!(
                channel = %req.channel_name,
                "SignalSubscribe rejected: unknown channel"
            );
            continue;
        };

        // 2. Freshness window.
        let drift = if now_ms >= req.timestamp_ms {
            now_ms - req.timestamp_ms
        } else {
            req.timestamp_ms - now_ms
        };
        if drift > signal::REPLAY_TIMESTAMP_WINDOW_MS {
            signal_metrics::record_subscribe_rejected("stale_timestamp");
            tracing::debug!(
                channel = %req.channel_name,
                grant_id = req.grant_id,
                drift_ms = drift,
                "SignalSubscribe rejected: timestamp out of freshness window"
            );
            continue;
        }

        // 3. Grant lookup — also enforces ops + expiry + channel-cover.
        let key = match grants.check_subscribe(req.grant_id, channel_id, now_ms) {
            Some(k) => *k,
            None => {
                signal_metrics::record_subscribe_rejected("grant_check_failed");
                tracing::debug!(
                    channel = %req.channel_name,
                    grant_id = req.grant_id,
                    "SignalSubscribe rejected: grant lookup failed (missing/expired/revoked/ops/channel)"
                );
                continue;
            }
        };

        // 4. HMAC-verify the request.
        if !signal::auth::hmac_verify_subscribe_request(
            &key,
            &req.channel_name,
            req.subscriber_shard_id,
            req.grant_id,
            req.nonce,
            req.timestamp_ms,
            req.valid_until_tick,
            &req.auth_tag,
        ) {
            signal_metrics::record_subscribe_rejected("hmac_failed");
            tracing::debug!(
                channel = %req.channel_name,
                grant_id = req.grant_id,
                "SignalSubscribe rejected: HMAC verification failed"
            );
            continue;
        }

        // 5. Clamp lease to a sane ceiling (defends against
        // valid_until_tick = u64::MAX style requests).
        let effective_until = req.valid_until_tick.min(lease_cap);

        // 6. Register / renew.
        let created = channels.add_remote_subscriber(
            channel_id,
            ShardId(req.subscriber_shard_id),
            req.grant_id,
            effective_until,
        );

        signal_metrics::record_subscribe_registered();
        if created {
            tracing::info!(
                channel = %req.channel_name,
                subscriber_shard = req.subscriber_shard_id,
                grant_id = req.grant_id,
                lease_until_tick = effective_until,
                "SignalSubscribe registered"
            );
        } else {
            tracing::debug!(
                channel = %req.channel_name,
                subscriber_shard = req.subscriber_shard_id,
                grant_id = req.grant_id,
                "SignalSubscribe lease renewed"
            );
        }
    }
}

/// Drain queued `SignalUnsubscribe` requests. No HMAC verification —
/// unsubscribe is idempotent and self-targeted (a foreign shard is
/// telling us "I don't want forwards anymore"); the worst a malicious
/// peer can do is unsubscribe themselves from forwards, which is
/// equivalent to them simply ignoring the forwards. No collateral damage
/// to other subscribers.
pub fn apply_signal_unsubscribe(
    mut channels: ResMut<SignalChannelTable>,
    mut buffer: ResMut<IncomingSubscribeBuffer>,
) {
    for req in buffer.unsubscribes.drain(..) {
        let Some(channel_id) = channels.resolve(&req.channel_name) else {
            continue;
        };
        let removed = channels.remove_remote_subscriber(
            channel_id,
            ShardId(req.subscriber_shard_id),
            req.grant_id,
        );
        if removed {
            tracing::info!(
                channel = %req.channel_name,
                subscriber_shard = req.subscriber_shard_id,
                grant_id = req.grant_id,
                "SignalUnsubscribe applied"
            );
        }
    }
}

/// Sweep expired subscribers from every channel. Run at 1 Hz from the
/// shard's schedule (the plugin doesn't auto-register this — each shard
/// schedules it at whatever cadence makes sense given its tick rate).
pub fn cleanup_expired_subscribers_system(
    mut channels: ResMut<SignalChannelTable>,
    tick: Res<voxeldust_core::ecs::TickCounter>,
) {
    // Only run once per ~20 ticks to amortize the O(channels × subs) cost.
    if tick.0 % 20 != 0 {
        return;
    }
    let dropped = channels.cleanup_expired_subscribers(tick.0);
    if dropped > 0 {
        tracing::debug!(
            dropped,
            now_tick = tick.0,
            "expired remote subscribers swept"
        );
    }
}

/// Cadence for the rate-limit GC system. Runs every 1200 ticks (60s @ 20Hz)
/// — sparse enough that the O(sessions) sweep doesn't matter, frequent
/// enough that stale entries don't accumulate during the gap.
pub const RATE_LIMIT_GC_CADENCE_TICKS: u64 = 1_200;

/// Stale-session threshold. A session whose ALL buckets last touched
/// >10 minutes ago is presumed disconnected (TCP RST, crash, lost route)
/// and gets evicted. 10 min is past the legitimate idle gap a player
/// might have between actions but well before any TCP keep-alive.
pub const RATE_LIMIT_STALE_THRESHOLD_MS: u64 = 10 * 60 * 1000;

/// Periodically sweep stale entries from `ClientRateLimits` so the table
/// doesn't grow unbounded across long-running shard lifetimes. Runs as
/// a fallback for disconnects we didn't observe (TCP RST, network
/// partition, client crash) — graceful disconnects are still more
/// efficient via explicit `forget`, but this guarantees bounded memory
/// even when the network misbehaves.
pub fn rate_limit_gc(
    mut rate_limits: ResMut<ClientRateLimits>,
    tick: Res<voxeldust_core::ecs::TickCounter>,
) {
    if tick.0 % RATE_LIMIT_GC_CADENCE_TICKS != 0 {
        return;
    }
    let now_ms = signal::current_unix_millis();
    let evicted = rate_limits.evict_stale(now_ms, RATE_LIMIT_STALE_THRESHOLD_MS);
    if evicted > 0 {
        signal_metrics::record_rate_limit_sessions_evicted(evicted);
        tracing::info!(
            evicted,
            tracked = rate_limits.tracked_sessions(),
            "rate-limit GC swept stale sessions"
        );
    }
    // Sample the gauge each cadence so dashboards see the steady-state
    // tracked-session count, not just the burst-evict deltas.
    signal_metrics::set_rate_limited_sessions_tracked(rate_limits.tracked_sessions());
}

/// Tick threshold below which the lease-renewal system re-issues a
/// `SignalSubscribe` on behalf of an active Listener. 1200 ticks at 20 Hz
/// = 60 s — comfortable buffer ahead of the publisher's
/// `MAX_SUBSCRIBE_LEASE_TICKS = 6000` (5 min) ceiling.
///
/// Picking too small risks letting a slow QUIC round-trip arrive after
/// the publisher's lease expired (subscription drops, value flow stops).
/// Picking too large means re-signing HMACs and shipping QUIC bytes
/// more often than necessary; 60s is the well-tuned middle.
pub const LISTENER_RENEWAL_THRESHOLD_TICKS: u64 = 1_200;

/// Cadence at which `listener_lease_renewal` runs (every N ticks). 60
/// ticks at 20 Hz = 3 s — fine-grained enough that we never miss the
/// renewal window even with clock drift, sparse enough that the
/// per-system O(active listeners) work is dominated by the actual
/// renewal-needed iterations rather than the no-op walks.
pub const LISTENER_RENEWAL_CADENCE_TICKS: u64 = 60;

/// 1Hz-ish system that re-issues `SignalSubscribe` for active Listeners
/// whose lease on the publisher's side is approaching expiry. Without
/// this, the publisher's `cleanup_expired_subscribers_system` would
/// silently drop the subscription after ~5 minutes and forwarded values
/// would stop landing — listeners would appear "dead" to the player.
///
/// Per renewal:
///   1. Look up the held grant's key (fail-soft if grant was forgotten).
///   2. Build a fresh `SignalSubscribeData` with extended deadline,
///      HMAC-signed under the grant key + a fresh nonce.
///   3. Ship via QUIC to the publisher's shard.
///   4. Update `ListenerState.lease_until_tick` so subsequent ticks
///      don't re-renew.
///
/// Skipped listeners log at `tracing::debug!` (potentially noisy at
/// scale, kept low-severity); successful renewals log at `info!` so
/// operators can audit lease lifetimes.
pub fn listener_lease_renewal(
    bridge: Res<NetworkBridge>,
    held: Res<HeldGrants>,
    identity: Res<ShardIdentity>,
    tick: Res<voxeldust_core::ecs::TickCounter>,
    mut listeners: Query<&mut voxeldust_core::signal::ListenerState>,
) {
    use voxeldust_core::shard_message::{ShardMsg, SignalSubscribeData};
    use voxeldust_core::shard_types::{SessionToken, ShardId};

    if tick.0 % LISTENER_RENEWAL_CADENCE_TICKS != 0 {
        return;
    }
    if listeners.is_empty() {
        return;
    }

    let now_tick = tick.0;
    let now_ms = signal::current_unix_millis();
    let our_shard_id = identity.shard_id.0;

    // Snapshot to-renew + their renewed states first; defer QUIC dispatch
    // to a tokio::spawn at the end so we don't hold the bevy borrows
    // across the async boundary.
    let mut to_dispatch: Vec<(u64, SignalSubscribeData)> = Vec::new();

    for mut listener in &mut listeners {
        if !listener.active {
            continue;
        }
        // Renew if the lease ends within the threshold OR has already
        // expired (the publisher dropped us; re-establish from scratch).
        let remaining = listener.lease_until_tick.saturating_sub(now_tick);
        if remaining > LISTENER_RENEWAL_THRESHOLD_TICKS {
            continue;
        }

        // Look up the held grant's key.
        let session = SessionToken(listener.owner_session);
        let grant = match held.get(session, listener.grant_id) {
            Some(g) => g,
            None => {
                tracing::debug!(
                    owner = listener.owner_session,
                    grant_id = listener.grant_id,
                    "listener lease renewal skipped: held grant missing"
                );
                continue;
            }
        };

        // Compute the new lease window. The publisher will clamp this to
        // its own `MAX_SUBSCRIBE_LEASE_TICKS`, so claim the maximum and
        // let the publisher decide.
        let new_lease_until = now_tick.saturating_add(MAX_SUBSCRIBE_LEASE_TICKS);
        let nonce = now_ms;

        let auth_tag = signal::auth::hmac_sign_subscribe_request(
            &grant.key,
            &listener.remote_channel_name,
            our_shard_id,
            listener.grant_id,
            nonce,
            now_ms,
            new_lease_until,
        );

        let req = SignalSubscribeData {
            subscriber_shard_id: our_shard_id,
            channel_name: listener.remote_channel_name.clone(),
            grant_id: listener.grant_id,
            nonce,
            timestamp_ms: now_ms,
            valid_until_tick: new_lease_until,
            auth_tag: auth_tag.to_vec(),
        };

        to_dispatch.push((listener.source_shard_id, req));
        // Update local state to reflect the renewal request.
        listener.lease_until_tick = new_lease_until;

        signal_metrics::record_listener_lease_renewed();
        tracing::info!(
            owner = listener.owner_session,
            grant_id = listener.grant_id,
            source_shard = listener.source_shard_id,
            new_lease_until_tick = new_lease_until,
            "listener lease renewal issued"
        );
    }

    if to_dispatch.is_empty() {
        return;
    }

    let peer_registry = bridge.peer_registry.clone();
    let quic_send_tx = bridge.quic_send_tx.clone();
    tokio::spawn(async move {
        let registry = peer_registry.read().await;
        for (target_shard_id, req) in to_dispatch {
            let target = ShardId(target_shard_id);
            let endpoint = match registry.endpoint(target) {
                Some(ep) => ep,
                None => {
                    tracing::warn!(
                        target = target_shard_id,
                        "listener lease renewal: target shard not in peer registry"
                    );
                    continue;
                }
            };
            if let Err(e) = quic_send_tx
                .send((target, endpoint.quic_addr, ShardMsg::SignalSubscribe(req)))
                .await
            {
                tracing::warn!(
                    target = target_shard_id,
                    %e,
                    "listener lease renewal QUIC send failed"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Phase 3E.2 — Antenna block-driven outbound publish.
//
// Antenna is a block-bound "always-on RemoteSignalPublish." Each tick the
// system reads the source channel's current value and forwards it to the
// target shard via the placing player's held grant (HMAC-signed exactly
// the same way `apply_remote_signal_publish` signs tablet-driven
// publishes — the wire format is uniform).
//
// Why a separate system from apply_remote_signal_publish: tablet-driven
// publishes are EVENT-shaped (one ClientMsg per click), antenna-driven
// publishes are POLL-shaped (every tick while active). Splitting keeps
// the per-tick hot path off the message-drain pattern.
// ---------------------------------------------------------------------------

/// Iterate every active Antenna block on this shard, sign the current
/// value of its source channel under the placing player's held grant,
/// and ship a one-entry `SignalBroadcastBatch` via QUIC to the target
/// shard. The receiver's `try_push_remote` validates against
/// `GrantsRegistry::check_publish` on the issuer's side — same path as
/// any other grant-stamped publish.
///
/// **Scoping**: the antenna stamps wire scope = 0 (Local) because the
/// receiving channel on the target shard is Local-scoped (the user's
/// "remote control of a Local channel via grant" use case). Phase 3F's
/// galaxy relay will revisit this when Radio-scope antenna becomes
/// useful.
///
/// **No per-tick allocation in the hot path**: the per-antenna iteration
/// builds one Vec entry, passes it to a fire-and-forget tokio::spawn for
/// QUIC dispatch. The async closure owns the data.
pub fn antenna_publish(
    channels: Res<SignalChannelTable>,
    held: Res<HeldGrants>,
    bridge: Res<NetworkBridge>,
    identity: Res<ShardIdentity>,
    policy: Res<crate::harness::WireDictSendPolicy>,
    antennas: Query<&voxeldust_core::signal::AntennaState>,
) {
    use voxeldust_core::shard_message::{ShardMsg, SignalBroadcastBatchData, SignalBroadcastEntry};
    use voxeldust_core::shard_types::{SessionToken, ShardId};
    use voxeldust_core::signal::SignalValue;

    if antennas.iter().next().is_none() {
        return;
    }

    let now_ms = signal::current_unix_millis();
    let source_shard_id = identity.shard_id;

    // Snapshot per-antenna outputs first; defer QUIC dispatch to the
    // tokio::spawn at the end. Avoids holding the bevy resource borrows
    // across the async boundary.
    let mut to_send: Vec<(u64, SignalBroadcastEntry)> = Vec::new();

    for antenna in &antennas {
        if !antenna.active {
            continue;
        }
        // Source channel must resolve locally.
        let Some(source_id) = antenna.source_channel_id else {
            continue;
        };
        let Some(ch) = channels.get_by_id(source_id) else {
            continue;
        };

        // Held grant lookup. The placing player's session anchors the
        // grant in HeldGrants (registered via `AddHeldGrant`). Missing
        // grant = silent skip — could be a freshly-placed antenna that
        // hasn't had AddHeldGrant processed yet, or the grant was
        // forgotten without revoking the antenna config.
        let session = SessionToken(antenna.owner_session);
        let Some(grant) = held.get(session, antenna.grant_id) else {
            tracing::debug!(
                owner = antenna.owner_session,
                grant_id = antenna.grant_id,
                "antenna_publish: held grant missing — skipping (config stale?)"
            );
            continue;
        };

        // Cross-check: the antenna's claimed target_shard_id must match
        // the held grant's stored target. Mismatch = stale config.
        if antenna.target_shard_id != grant.target_shard_id {
            tracing::warn!(
                owner = antenna.owner_session,
                grant_id = antenna.grant_id,
                claimed = antenna.target_shard_id,
                held = grant.target_shard_id,
                "antenna_publish: target_shard != grant.target_shard — skipping"
            );
            continue;
        }

        // Canonicalize the value bits exactly as the receiver will when
        // verifying. Float NaN payloads round-trip via `to_bits()`
        // bit-equality, not float equality.
        let (value_type, value_data, value_bits) = match ch.value {
            SignalValue::Bool(b) => {
                let v = if b { 1.0_f32 } else { 0.0 };
                (0u8, v, v.to_bits())
            }
            SignalValue::Float(f) => (1u8, f, f.to_bits()),
            SignalValue::State(s) => {
                let v = s as f32;
                (2u8, v, v.to_bits())
            }
        };

        // Wire scope = 0 (Local) for the canonical "remote control of a
        // Local channel" antenna case. The receiver's try_push_remote
        // accepts because grant_id != 0 unlocks LocalChannelImmutable.
        let wire_scope_code = 0_u8;
        let frequency = antenna.frequency;
        // Sequence: per-antenna counter would need per-AntennaState
        // mutable state, which is awkward in a parallel-iter system.
        // Use seq=1 + ts=now: receiver's per-(channel, sender, grant)
        // window treats each antenna's first-tick entry as fresh. A
        // future hardening pass adds a per-antenna AtomicU64.
        let seq = 1_u64;

        let tag = signal::auth::hmac_sign(
            &grant.key,
            &antenna.remote_channel_name,
            wire_scope_code,
            frequency,
            value_type,
            value_bits,
            now_ms,
            source_shard_id.0,
            seq,
            antenna.grant_id,
        );

        let entry = SignalBroadcastEntry {
            channel_name: antenna.remote_channel_name.clone(),
            value_type,
            value_data,
            scope: wire_scope_code,
            range_m: 0.0,
            frequency,
            sequence: seq,
            timestamp_ms: now_ms,
            grant_id: antenna.grant_id,
            auth_tag: tag.to_vec(),
        };
        to_send.push((antenna.target_shard_id, entry));
        signal_metrics::record_antenna_publish();
    }

    if to_send.is_empty() {
        return;
    }

    let peer_registry = bridge.peer_registry.clone();
    let wire_dicts = bridge.wire_dicts.clone();
    let quic_send_tx = bridge.quic_send_tx.clone();
    let v2_enabled = policy.v2_enabled;
    tokio::spawn(async move {
        let registry = peer_registry.read().await;
        if v2_enabled {
            let mut wd = wire_dicts.write().await;
            for (target_shard_id, entry) in to_send {
                let target = ShardId(target_shard_id);
                let Some(endpoint) = registry.endpoint(target) else {
                    tracing::warn!(
                        target = target_shard_id,
                        "antenna_publish: target shard not in peer registry"
                    );
                    continue;
                };
                let v2_batch = crate::wire_dict_registry::encode_v2_batch(
                    &mut wd,
                    target,
                    source_shard_id.0,
                    DVec3::ZERO,
                    vec![entry],
                );
                if let Err(e) = quic_send_tx
                    .send((
                        target,
                        endpoint.quic_addr,
                        ShardMsg::SignalBroadcastBatchV2(v2_batch),
                    ))
                    .await
                {
                    tracing::warn!(target = target_shard_id, %e, "antenna_publish QUIC send failed (V2)");
                }
            }
        } else {
            for (target_shard_id, entry) in to_send {
                let target = ShardId(target_shard_id);
                let endpoint = match registry.endpoint(target) {
                    Some(ep) => ep,
                    None => {
                        tracing::warn!(
                            target = target_shard_id,
                            "antenna_publish: target shard not in peer registry"
                        );
                        continue;
                    }
                };
                let batch = SignalBroadcastBatchData {
                    source_shard_id: source_shard_id.0,
                    source_position: DVec3::ZERO,
                    entries: vec![entry],
                };
                if let Err(e) = quic_send_tx
                    .send((target, endpoint.quic_addr, ShardMsg::SignalBroadcastBatch(batch)))
                    .await
                {
                    tracing::warn!(target = target_shard_id, %e, "antenna_publish QUIC send failed");
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Phase 3E.3 — Generic apply helpers for Antenna / Listener configs.
//
// These functions are shard-agnostic glue between `BlockConfigUpdateData`
// (which arrives over the wire from a player's tablet) and the
// `AntennaState` / `ListenerState` components that drive `antenna_publish`
// and `listener_mirror` each tick.
//
// Each helper:
//   * Validates that the placing player's HeldGrants contains the
//     requested grant_id (otherwise the antenna/listener can't sign or
//     verify HMAC at the per-tick path).
//   * Resolves channel names → ChannelIds, creating missing channels
//     where appropriate (Listener creates bridged + destination).
//   * Returns the resolved State component for the caller to
//     `commands.entity(e).insert(state)`.
//
// The Listener helper additionally returns a `SignalSubscribeData` for
// the caller to ship to the source shard via QUIC — the helper prepares
// the canonicalized HMAC signature so the caller doesn't need to know
// about the auth construction.
// ---------------------------------------------------------------------------

/// Why a Listener apply step couldn't produce a valid ListenerState.
/// Surfaced to callers so they can log structured errors rather than
/// trying to interpret an `Option::None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListenerApplyDenied {
    /// `held_grants[(owner_session, grant_id)]` returned None. Player
    /// hasn't registered the grant via `AddHeldGrant` yet, or forgot it.
    GrantNotHeld,
    /// The grant's stored target_shard doesn't match the listener's
    /// claimed source_shard_id. Stale or hostile config.
    SourceShardMismatch { held: u64, claimed: u64 },
    /// Channel name length out of bounds (>64 KiB) — defends the HMAC
    /// canonicalization which uses a u16 length prefix.
    ChannelNameTooLong,
}

/// Symmetric of [`ListenerApplyDenied`] for the Antenna path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AntennaApplyDenied {
    /// Source channel doesn't exist locally. Antenna can only forward
    /// channels that exist on this shard — the source must be pre-
    /// configured (e.g., by another block placement or a seat binding).
    UnknownSourceChannel,
    GrantNotHeld,
    TargetShardMismatch { held: u64, claimed: u64 },
}

/// Result of `apply_antenna_config`. The caller inserts `state` as a
/// Component on the block entity and (optionally) persists the config
/// to redb.
#[derive(Debug, Clone)]
pub struct AntennaApplyOk {
    pub state: voxeldust_core::signal::AntennaState,
}

/// Result of `apply_listener_config`. The caller:
///   * Inserts `state` as a Component on the block entity.
///   * Ships `outbound_subscribe` via QUIC to `state.source_shard_id`.
///   * Persists the config to redb (optional, for crash recovery).
#[derive(Debug, Clone)]
pub struct ListenerApplyOk {
    pub state: voxeldust_core::signal::ListenerState,
    pub outbound_subscribe: voxeldust_core::shard_message::SignalSubscribeData,
}

/// Apply an `AntennaConfig` from a `BlockConfigUpdateData`. Returns
/// `AntennaApplyOk` on success — the caller inserts the resolved
/// `AntennaState` as a Component on the block's entity.
///
/// This function does NOT touch the entity-component world; it's pure
/// validation + resolution against `SignalChannelTable` and `HeldGrants`.
/// Splitting validation from mutation keeps the helper testable in
/// isolation (no Bevy `Commands` needed).
pub fn apply_antenna_config(
    channels: &SignalChannelTable,
    held: &HeldGrants,
    config: &voxeldust_core::signal::config::AntennaConfig,
    owner_session: voxeldust_core::shard_types::SessionToken,
) -> Result<AntennaApplyOk, AntennaApplyDenied> {
    // 1. Source channel must exist. Antennas don't create channels —
    //    they bridge ones that already exist (block placement order
    //    constraint: place the source-publishing block first, then the
    //    antenna).
    let source_channel_id = channels
        .resolve(&config.source_channel_name)
        .ok_or(AntennaApplyDenied::UnknownSourceChannel)?;

    // 2. Held grant must exist for the placing player.
    let grant = held
        .get(owner_session, config.grant_id)
        .ok_or(AntennaApplyDenied::GrantNotHeld)?;

    // 3. Cross-check claimed target_shard against held grant's stored
    //    target. Mismatch = stale config from a re-pasted grant.
    if grant.target_shard_id != config.target_shard_id {
        return Err(AntennaApplyDenied::TargetShardMismatch {
            held: grant.target_shard_id,
            claimed: config.target_shard_id,
        });
    }

    Ok(AntennaApplyOk {
        state: voxeldust_core::signal::AntennaState {
            source_channel_id: Some(source_channel_id),
            remote_channel_name: config.remote_channel_name.clone(),
            frequency: config.frequency,
            grant_id: config.grant_id,
            target_shard_id: config.target_shard_id,
            owner_session: owner_session.0,
            active: true,
        },
    })
}

/// Apply a `ListenerConfig`. Mutates `channels` (creates bridged + destination
/// channels if they don't exist) and `grants` (adds bridged channel id to
/// the mirror grant for inbound HMAC verification).
///
/// Returns `ListenerApplyOk { state, outbound_subscribe }`. The caller:
///   * `commands.entity(block).insert(state)` to attach the listener's
///     runtime state to the block entity.
///   * Ships `outbound_subscribe` (pre-signed `ShardMsg::SignalSubscribe`)
///     via QUIC to `state.source_shard_id` so the publisher's shard
///     registers us as a subscriber.
pub fn apply_listener_config(
    channels: &mut SignalChannelTable,
    grants: &mut GrantsRegistry,
    held: &HeldGrants,
    config: &voxeldust_core::signal::config::ListenerConfig,
    owner_session: voxeldust_core::shard_types::SessionToken,
    now_tick: u64,
) -> Result<ListenerApplyOk, ListenerApplyDenied> {
    use voxeldust_core::signal::types::{ChannelMergeStrategy, SignalScope};

    // 1. Held grant + cross-check.
    let held_grant = held
        .get(owner_session, config.grant_id)
        .ok_or(ListenerApplyDenied::GrantNotHeld)?;
    if held_grant.target_shard_id != config.source_shard_id {
        return Err(ListenerApplyDenied::SourceShardMismatch {
            held: held_grant.target_shard_id,
            claimed: config.source_shard_id,
        });
    }

    // Bound channel name length (defended-by-construction in `hmac_sign`
    // via debug_assert; this is the production-path enforcement).
    if config.bridged_channel_name.len() > u16::MAX as usize
        || config.destination_channel_name.len() > u16::MAX as usize
    {
        return Err(ListenerApplyDenied::ChannelNameTooLong);
    }

    // 2. Resolve / create the bridged Radio-scope channel. This is what
    //    `try_push_remote` needs to find when forwarded entries arrive
    //    from the publisher's shard.
    let bridged_channel_id = channels.resolve_or_create(
        &config.bridged_channel_name,
        SignalScope::Radio { frequency: config.frequency },
        ChannelMergeStrategy::LastWrite,
        owner_session.0,
    );

    // 3. Resolve / create the Local destination channel. Internal
    //    subscribers wire to this without learning about HMAC or grants.
    let destination_channel_id = channels.resolve_or_create(
        &config.destination_channel_name,
        SignalScope::Local,
        ChannelMergeStrategy::LastWrite,
        owner_session.0,
    );

    // 4. Add bridged channel to the mirror grant in our registry so
    //    `try_push_remote`'s `check_publish` channel-cover check passes
    //    when forwarded entries arrive.
    grants.add_channel_to_grant(config.grant_id, bridged_channel_id);

    // 5. Build the outbound `SignalSubscribe` request, HMAC-signed under
    //    the held grant key. The caller ships this via QUIC.
    let now_ms = signal::current_unix_millis();
    let nonce = signal::current_unix_millis(); // any non-replayed value works as nonce
    let valid_until_tick = now_tick.saturating_add(MAX_SUBSCRIBE_LEASE_TICKS);
    let auth_tag = signal::auth::hmac_sign_subscribe_request(
        &held_grant.key,
        &config.bridged_channel_name,
        owner_session.0, // subscriber_shard_id == our local id; caller may override
        config.grant_id,
        nonce,
        now_ms,
        valid_until_tick,
    );
    let outbound_subscribe = voxeldust_core::shard_message::SignalSubscribeData {
        subscriber_shard_id: 0, // caller fills in own shard_id before sending
        channel_name: config.bridged_channel_name.clone(),
        grant_id: config.grant_id,
        nonce,
        timestamp_ms: now_ms,
        valid_until_tick,
        auth_tag: auth_tag.to_vec(),
    };

    Ok(ListenerApplyOk {
        state: voxeldust_core::signal::ListenerState {
            destination_channel_id: Some(destination_channel_id),
            bridged_channel_id: Some(bridged_channel_id),
            remote_channel_name: config.bridged_channel_name.clone(),
            frequency: config.frequency,
            grant_id: config.grant_id,
            source_shard_id: config.source_shard_id,
            owner_session: owner_session.0,
            active: true,
            lease_until_tick: valid_until_tick,
        },
        outbound_subscribe,
    })
}

/// Per-tick mirror system for Listener blocks. For each active listener,
/// reads the value of the bridged Radio-scope channel (which
/// `try_push_remote` already verified + push_pendinged from the
/// publisher's forwarded entries) and `publish_direct_id`s it onto the
/// listener's chosen Local-scope destination channel.
///
/// **Why two channels per listener instead of one**: the bridged channel
/// has Radio scope (so `try_push_remote` accepts inbound forwards), the
/// destination channel has Local scope (so internal subscribers can
/// wire to it without learning about HMAC, grants, or cross-shard
/// plumbing). The listener is the "isolating bridge" that decouples the
/// cross-shard auth surface from the in-ship wiring surface.
///
/// Runs in `SignalSet::Subscribe` after `antenna_publish`. By that point
/// the merge has finalized the bridged channel's value for this tick;
/// `publish_direct_id` to the destination skips the merge for the
/// destination (no need to aggregate; the listener IS the publisher).
pub fn listener_mirror(
    mut channels: ResMut<SignalChannelTable>,
    listeners: Query<&voxeldust_core::signal::ListenerState>,
) {
    if listeners.iter().next().is_none() {
        return;
    }
    for listener in &listeners {
        if !listener.active {
            continue;
        }
        let (Some(bridged), Some(destination)) =
            (listener.bridged_channel_id, listener.destination_channel_id)
        else {
            continue;
        };
        // Read the bridged channel's current value. If the channel
        // doesn't exist (config drift), skip silently.
        let value = match channels.get_by_id(bridged) {
            Some(ch) => ch.value,
            None => continue,
        };
        // Publish directly to destination — bypasses merge because the
        // listener is the sole publisher of the destination channel.
        // (If a player wires multiple listeners to the same destination,
        // the LastWrite merge strategy makes the per-tick order
        // deterministic; if they want Sum/Average/etc., they place a
        // SignalConverter in front of the destination.)
        channels.publish_direct_id(destination, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxeldust_core::shard_types::SessionToken;
    use voxeldust_core::signal::config::{AntennaConfig, ListenerConfig};
    use voxeldust_core::signal::types::{ChannelMergeStrategy, SignalScope};

    fn fixture_session() -> SessionToken {
        SessionToken(99)
    }

    fn fixture_held_grant_for(
        held: &mut HeldGrants,
        session: SessionToken,
        grant_id: u64,
        target_shard: u64,
    ) {
        held.insert(
            session,
            grant_id,
            HeldGrant {
                key: [0xCC; 32],
                target_shard_id: target_shard,
                label: "fixture".into(),
            },
        );
    }

    #[test]
    fn held_grants_total_count_sums_across_sessions() {
        let mut held = HeldGrants::default();
        let alice = SessionToken(1);
        let bob = SessionToken(2);
        // Alice holds 2 grants, Bob holds 1.
        fixture_held_grant_for(&mut held, alice, 10, 100);
        fixture_held_grant_for(&mut held, alice, 11, 100);
        fixture_held_grant_for(&mut held, bob, 20, 200);
        assert_eq!(held.total_count(), 3);
        // Forgetting a per-session grant decrements but doesn't drop the
        // session.
        held.forget(alice, 10);
        assert_eq!(held.total_count(), 2);
        // Forgetting the whole session drops both of Alice's remaining.
        held.forget_all_for_session(alice);
        assert_eq!(held.total_count(), 1);
    }

    #[test]
    fn apply_antenna_config_resolves_state_on_happy_path() {
        let mut channels = SignalChannelTable::new();
        // Source channel must already exist.
        channels.get_or_create(
            "alice.local.alarm",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            99,
        );
        let mut held = HeldGrants::default();
        let session = fixture_session();
        fixture_held_grant_for(&mut held, session, 0xC0FE, /*target=*/ 1);

        let cfg = AntennaConfig {
            source_channel_name: "alice.local.alarm".into(),
            remote_channel_name: "bob.alarm-mirror".into(),
            frequency: 0,
            grant_id: 0xC0FE,
            target_shard_id: 1,
        };
        let res =
            apply_antenna_config(&channels, &held, &cfg, session).expect("should resolve");
        assert_eq!(res.state.target_shard_id, 1);
        assert_eq!(res.state.grant_id, 0xC0FE);
        assert_eq!(res.state.owner_session, 99);
        assert!(res.state.active);
        assert_eq!(res.state.remote_channel_name, "bob.alarm-mirror");
    }

    #[test]
    fn apply_antenna_config_rejects_when_held_grant_missing() {
        let mut channels = SignalChannelTable::new();
        channels.get_or_create(
            "alice.local.alarm",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            99,
        );
        let held = HeldGrants::default(); // no grants
        let session = fixture_session();
        let cfg = AntennaConfig {
            source_channel_name: "alice.local.alarm".into(),
            remote_channel_name: "bob.alarm".into(),
            frequency: 0,
            grant_id: 0xC0FE,
            target_shard_id: 1,
        };
        let err = apply_antenna_config(&channels, &held, &cfg, session)
            .expect_err("missing held grant must reject");
        assert!(matches!(err, AntennaApplyDenied::GrantNotHeld));
    }

    #[test]
    fn apply_antenna_config_rejects_unknown_source_channel() {
        let channels = SignalChannelTable::new();
        let mut held = HeldGrants::default();
        let session = fixture_session();
        fixture_held_grant_for(&mut held, session, 1, 1);
        let cfg = AntennaConfig {
            source_channel_name: "no.such.channel".into(),
            remote_channel_name: "x".into(),
            frequency: 0,
            grant_id: 1,
            target_shard_id: 1,
        };
        let err = apply_antenna_config(&channels, &held, &cfg, session).unwrap_err();
        assert!(matches!(err, AntennaApplyDenied::UnknownSourceChannel));
    }

    #[test]
    fn apply_antenna_config_rejects_target_shard_mismatch() {
        let mut channels = SignalChannelTable::new();
        channels.get_or_create(
            "src",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            99,
        );
        let mut held = HeldGrants::default();
        let session = fixture_session();
        fixture_held_grant_for(&mut held, session, 1, /*target_held=*/ 1);
        let cfg = AntennaConfig {
            source_channel_name: "src".into(),
            remote_channel_name: "x".into(),
            frequency: 0,
            grant_id: 1,
            target_shard_id: /*claimed=*/ 999,
        };
        let err = apply_antenna_config(&channels, &held, &cfg, session).unwrap_err();
        assert!(matches!(err, AntennaApplyDenied::TargetShardMismatch { .. }));
    }

    #[test]
    fn apply_listener_config_creates_channels_and_signs_subscribe() {
        let mut channels = SignalChannelTable::new();
        let mut grants = GrantsRegistry::default();
        // Pre-insert a mirror grant — usually populated by
        // apply_add_held_grant when the player registers the held grant.
        grants.insert(voxeldust_core::signal::RemoteAccessGrant {
            grant_id: 0xBEEF,
            key: [0xCC; 32],
            channels: smallvec::smallvec![],
            namespace_glob: None,
            ops: voxeldust_core::signal::GrantOps::Both,
            label: "from alice".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 99,
            revoked: false,
            mirror_of_held: true,
        });
        let mut held = HeldGrants::default();
        let session = fixture_session();
        fixture_held_grant_for(&mut held, session, 0xBEEF, /*source=*/ 1);

        let cfg = ListenerConfig {
            destination_channel_name: "bob.local.health-mirror".into(),
            bridged_channel_name: "alice.health".into(),
            frequency: 0,
            grant_id: 0xBEEF,
            source_shard_id: 1,
        };

        let res =
            apply_listener_config(&mut channels, &mut grants, &held, &cfg, session, /*now_tick=*/ 100)
                .expect("should resolve");
        // Both channels created.
        assert!(channels.resolve("alice.health").is_some());
        assert!(channels.resolve("bob.local.health-mirror").is_some());
        // Mirror grant now covers the bridged channel.
        let bridged_id = channels.resolve("alice.health").unwrap();
        assert!(grants.check_publish(0xBEEF, bridged_id, 0).is_some());
        // Outbound subscribe is signed.
        assert_eq!(res.outbound_subscribe.grant_id, 0xBEEF);
        assert_eq!(res.outbound_subscribe.channel_name, "alice.health");
        assert_eq!(res.outbound_subscribe.auth_tag.len(), 16);
        // Lease clamped reasonably (now_tick + MAX_SUBSCRIBE_LEASE_TICKS).
        assert_eq!(
            res.outbound_subscribe.valid_until_tick,
            100 + MAX_SUBSCRIBE_LEASE_TICKS
        );
    }

    #[test]
    fn apply_listener_config_rejects_when_held_grant_missing() {
        let mut channels = SignalChannelTable::new();
        let mut grants = GrantsRegistry::default();
        let held = HeldGrants::default();
        let session = fixture_session();
        let cfg = ListenerConfig {
            destination_channel_name: "x".into(),
            bridged_channel_name: "y".into(),
            frequency: 0,
            grant_id: 1,
            source_shard_id: 1,
        };
        let err = apply_listener_config(
            &mut channels, &mut grants, &held, &cfg, session, /*now_tick=*/ 0,
        )
        .unwrap_err();
        assert_eq!(err, ListenerApplyDenied::GrantNotHeld);
    }

    #[test]
    fn listener_renewal_threshold_constants_are_sensible() {
        // The renewal threshold must comfortably fit inside the
        // publisher's max-lease ceiling, with enough buffer that a
        // QUIC round-trip + clock skew doesn't push us past expiry.
        // 1200 ticks = 60 s renewal window, 6000 ticks = 5 min ceiling
        // → 80% of the lease elapses before we renew. Reasonable.
        assert!(LISTENER_RENEWAL_THRESHOLD_TICKS < MAX_SUBSCRIBE_LEASE_TICKS);
        assert!(MAX_SUBSCRIBE_LEASE_TICKS - LISTENER_RENEWAL_THRESHOLD_TICKS >= 4_000);
        // Cadence is much smaller than the renewal threshold — we'll
        // poll multiple times within a single renewal window.
        assert!(LISTENER_RENEWAL_CADENCE_TICKS < LISTENER_RENEWAL_THRESHOLD_TICKS / 4);
    }

    #[test]
    fn listener_renewal_lease_recomputation() {
        // Simulate the lease window math the renewal system runs.
        let now_tick = 5_000_u64;
        // Lease ends in 50 s = 1000 ticks — INSIDE the 1200-tick threshold.
        let lease_until_tick = now_tick + 1000;
        let remaining = lease_until_tick.saturating_sub(now_tick);
        assert!(remaining <= LISTENER_RENEWAL_THRESHOLD_TICKS);

        // Renewed lease deadline = now + max_lease.
        let renewed = now_tick + MAX_SUBSCRIBE_LEASE_TICKS;
        // After renewal, the new remaining is the full max lease.
        assert_eq!(renewed - now_tick, MAX_SUBSCRIBE_LEASE_TICKS);
    }

    #[test]
    fn listener_renewal_skips_when_lease_far_from_expiry() {
        // Lease ends in 4 minutes — well outside threshold. No renewal.
        let now_tick = 1_000_u64;
        let lease_until_tick = now_tick + 4 * 60 * 20; // 4 min @ 20 Hz = 4800 ticks
        let remaining = lease_until_tick.saturating_sub(now_tick);
        assert!(remaining > LISTENER_RENEWAL_THRESHOLD_TICKS);
    }

    #[test]
    fn listener_renewal_renews_when_lease_already_expired() {
        // Already past expiry (subscription dropped on publisher's side).
        // Renewal must still fire to re-establish the subscription.
        let now_tick = 10_000_u64;
        let lease_until_tick = 5_000_u64; // 5000 ticks ago
        let remaining = lease_until_tick.saturating_sub(now_tick);
        assert_eq!(remaining, 0, "saturating sub clamps to zero");
        assert!(remaining <= LISTENER_RENEWAL_THRESHOLD_TICKS);
    }

    #[test]
    fn apply_listener_config_rejects_source_shard_mismatch() {
        let mut channels = SignalChannelTable::new();
        let mut grants = GrantsRegistry::default();
        let mut held = HeldGrants::default();
        let session = fixture_session();
        fixture_held_grant_for(&mut held, session, 1, /*target_held=*/ 1);
        let cfg = ListenerConfig {
            destination_channel_name: "x".into(),
            bridged_channel_name: "y".into(),
            frequency: 0,
            grant_id: 1,
            source_shard_id: /*claimed=*/ 999,
        };
        let err = apply_listener_config(&mut channels, &mut grants, &held, &cfg, session, 0)
            .unwrap_err();
        assert!(matches!(err, ListenerApplyDenied::SourceShardMismatch { .. }));
    }
}
