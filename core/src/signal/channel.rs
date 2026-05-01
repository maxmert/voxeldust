//! Signal channel table — named pub/sub channels with merge strategies,
//! scoping, and access control.  Channels are indexed by `ChannelId` (u16)
//! for O(1) hot-path access; a string→id registry handles config-time resolution.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use smallvec::SmallVec;

use crate::shard_types::ShardId;

use super::types::*;

/// Failure modes of `SignalChannelTable::try_push_pending`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishDenied {
    /// Channel name not registered on this shard. Clients must not
    /// auto-create channels by publishing to them.
    UnknownChannel,
    /// Sender's `player_id` not permitted by `publish_policy`.
    Forbidden,
}

/// Why a cross-shard ingress entry was rejected at the `try_push_remote`
/// boundary. Each variant maps to one of the holes the user identified in
/// the audit:
///
/// | Variant | Closes |
/// |---|---|
/// | [`UnknownChannel`] | Auto-creation by malicious publishers (channel slot exhaustion) |
/// | [`LocalChannelImmutable`] | Cross-shard injection into Local channels of the same name |
/// | [`ScopeMismatch`] | Peer relabeling a packet's scope to bypass spatial / relay filters |
/// | [`AuthFailed`] | Forging a publish for a Phase-3 keyed channel |
/// | [`StaleOrFutureTimestamp`] / [`Replay`] | Replay of captured signed entries |
///
/// Phase 1A enforces the first three; the auth + replay checks land in
/// Phase 1C / Phase 3. The receive path (`signal_ingest_remote` in
/// `shard-common`) logs rejections at `tracing::debug!` rather than `warn!`
/// to deny adversaries free log-amplification on bad inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteIngressDenied {
    /// Wire entry references a channel name that doesn't exist on this
    /// shard. Cross-shard publishes must NOT auto-create channels — that
    /// would let any peer exhaust the local `ChannelId` space by publishing
    /// to random names.
    UnknownChannel,
    /// Wire entry targets a channel whose local scope is `Local`. Local
    /// channels are unconditionally unreachable from foreign shards in
    /// Phase 1; Phase 3 introduces grant_id-keyed access so an explicitly-
    /// granted remote actor can reach a Local channel's signature, but
    /// without a grant the channel stays sealed.
    LocalChannelImmutable,
    /// Wire scope code (1=ShortRange, 2=LongRange, 3=Radio) doesn't match
    /// the local channel's scope class. A peer attempting to inject a
    /// LongRange-relayed value into a ShortRange-only channel — or vice
    /// versa — gets dropped here. The receive path can't accept a wire
    /// claim about scope different from the channel's authoritative scope.
    ScopeMismatch,
    /// Phase 3+: HMAC verification of `auth_tag` against the channel's
    /// signature (or a matching grant's key) failed. Stub variant in
    /// Phase 1A — the field exists so callers don't need to refactor when
    /// the check turns on.
    AuthFailed,
    /// Phase 1C+: timestamp_ms differs from `now()` by more than the
    /// allowed window (default ±5 s). Stub variant in Phase 1A.
    StaleOrFutureTimestamp,
    /// Phase 1C+: monotonic-sequence replay window rejected this entry —
    /// it's a duplicate or older than the 64-entry sliding window for the
    /// `(channel, sender_shard_id)` pair. Stub variant in Phase 1A.
    Replay,
}

// ---------------------------------------------------------------------------
// ChannelId
// ---------------------------------------------------------------------------

/// Compact channel identifier — O(1) Vec index on the hot path.
/// Assigned monotonically per `SignalChannelTable`; shard-local (not stable
/// across shards or serialization — names are used on the wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u16);

// ---------------------------------------------------------------------------
// PendingAgg
// ---------------------------------------------------------------------------

/// Running aggregation state — replaces `Vec<SignalValue>` with O(1) inline
/// accumulation.  Each variant holds only the fields relevant to its merge
/// strategy, so there are no ambiguous "unused" fields.
#[derive(Clone, Debug)]
pub enum PendingAgg {
    /// No values pushed this tick (idle state for LastWrite).
    Idle,
    /// Most recent value wins (LastWrite strategy).
    LastWrite(SignalValue),
    /// Running sum and count (Sum and Average strategies).
    Sum { total: f64, count: u32 },
    /// Running extreme value (Max or Min — direction from `ChannelMergeStrategy`).
    Extreme { value: f64, count: u32 },
    /// Running boolean (AnyTrue = OR seed false, AllTrue = AND seed true).
    Bool { result: bool, count: u32 },
}

impl PendingAgg {
    /// Return the idle (zero-value) state appropriate for a merge strategy.
    fn idle_for(merge: ChannelMergeStrategy) -> Self {
        match merge {
            ChannelMergeStrategy::LastWrite => Self::Idle,
            ChannelMergeStrategy::Sum | ChannelMergeStrategy::Average => {
                Self::Sum { total: 0.0, count: 0 }
            }
            ChannelMergeStrategy::Max => Self::Extreme { value: f64::NEG_INFINITY, count: 0 },
            ChannelMergeStrategy::Min => Self::Extreme { value: f64::INFINITY, count: 0 },
            ChannelMergeStrategy::AnyTrue => Self::Bool { result: false, count: 0 },
            ChannelMergeStrategy::AllTrue => Self::Bool { result: true, count: 0 },
        }
    }

    /// Whether any values have been pushed this tick.
    fn has_values(&self) -> bool {
        match self {
            Self::Idle => false,
            Self::LastWrite(_) => true,
            Self::Sum { count, .. }
            | Self::Extreme { count, .. }
            | Self::Bool { count, .. } => *count > 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SignalChannel
// ---------------------------------------------------------------------------

/// Cross-shard subscription reference. A `RemoteShard` entry on a
/// channel's `subscribers` list means: when this channel becomes dirty,
/// forward the value to that shard, HMAC-stamped under the grant's key.
///
/// `valid_until_tick` is the lease deadline — `cleanup_expired_subscribers`
/// drops stale entries at 1 Hz. Lease renewal is just another
/// `SignalSubscribe` message that updates the existing entry's deadline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubscriberRef {
    /// A foreign shard wants forwarded values for this channel via the
    /// named grant. Receiver of the forwarded `SignalBroadcastBatch` will
    /// verify the HMAC tag against `grant.key` in their `GrantsRegistry`.
    RemoteShard {
        shard_id: ShardId,
        grant_id: u64,
        valid_until_tick: u64,
    },
}

/// 64-entry sliding window of accepted sequence numbers from a single
/// sender, plus the most-recent accepted timestamp. Used by
/// [`SignalChannelTable::try_push_remote`] to reject replays of captured
/// `SignalBroadcastEntry` payloads.
///
/// The bitmap encodes which of the most-recent 64 sequence numbers (ending
/// at `high_seq` inclusive) have already been accepted. Bit `i` set means
/// sequence `high_seq - i` was seen. When a fresh `seq` arrives:
/// - If `seq > high_seq`, shift the bitmap left by `(seq - high_seq)`,
///   set the low bit, and update `high_seq`.
/// - If `seq <= high_seq` but `high_seq - seq < 64`, check the bit. Already
///   set → duplicate, reject. Not set → fill it, accept.
/// - Else (`seq <= high_seq - 64`) the seq is older than the window — reject.
///
/// `last_ts_ms` is informational and lets a future audit trail know when
/// this sender was last heard from. The 5-second timestamp window is
/// checked against the current wall clock, not against `last_ts_ms`.
#[derive(Clone, Debug, Default)]
pub struct ReplayWindow {
    pub high_seq: u64,
    pub bitmap: u64,
    pub last_ts_ms: u64,
}

/// Reasons a `ReplayWindow::try_accept` call rejected a wire entry. Returned
/// to `try_push_remote` which translates each variant into the equivalent
/// [`RemoteIngressDenied`] case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayReject {
    /// `|now_ms - timestamp_ms| > 5000` — outside the freshness window.
    StaleOrFutureTimestamp,
    /// Sequence already accepted (bitmap bit set) or older than the
    /// 64-entry sliding window's tail.
    DuplicateOrOutOfWindow,
}

/// Tunable: maximum age (ms) the receiver tolerates between the local clock
/// and the wire timestamp. ±5 s covers normal cross-galaxy QUIC latency
/// plus a generous clock-skew budget; lower would reject legitimate
/// traffic, higher would weaken replay defense.
pub const REPLAY_TIMESTAMP_WINDOW_MS: u64 = 5_000;

/// One channel's snapshot ready to be encoded into a `SignalBroadcastEntry`.
/// Returned by [`SignalChannelTable::drain_remote_dirty`] which bumps the
/// channel's `outbound_seq` and copies the bumped value into [`Self::sequence`]
/// — so the caller can stamp the entry without re-touching the table.
///
/// `signature` is the channel's HMAC key; Phase 3 uses it to compute the
/// `auth_tag`. Phase 1A/2 callers ignore the signature and ship empty
/// `auth_tag` — Phase 3 swap-in is one targeted change in the broadcast
/// stamping code, no API ripple.
#[derive(Clone, Debug)]
pub struct RemoteDirtyEntry {
    pub name: String,
    pub value: SignalValue,
    pub scope: SignalScope,
    pub sequence: u64,
    pub signature: [u8; 32],
}

/// Wall-clock UNIX millis used by replay-window freshness checks. Wraps
/// `SystemTime::now()` and saturates at zero on the (impossible) pre-epoch
/// case, so the call is infallible and inlinable. Cross-shard publishers
/// stamp the same value into `SignalBroadcastEntry.timestamp_ms` (Phase 2)
/// so receivers can compare directly without timezone or monotonic clock
/// gymnastics.
#[inline]
pub fn current_unix_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl ReplayWindow {
    /// Try to accept the wire entry's `(seq, ts)` pair against this
    /// sender's window. Mutates `self` only on success — a rejected entry
    /// leaves the window untouched so attackers can't "burn" sequence
    /// space with malformed inputs.
    ///
    /// `now_ms` is the local wall-clock at receive time; the freshness
    /// check uses absolute difference so both stale and future timestamps
    /// reject (clock-skew defense in both directions).
    pub fn try_accept(&mut self, seq: u64, ts_ms: u64, now_ms: u64) -> Result<(), ReplayReject> {
        // Freshness window check first — cheaper than the seq math, and a
        // stale entry shouldn't even tap the seq bookkeeping.
        let drift = if now_ms >= ts_ms { now_ms - ts_ms } else { ts_ms - now_ms };
        if drift > REPLAY_TIMESTAMP_WINDOW_MS {
            return Err(ReplayReject::StaleOrFutureTimestamp);
        }

        if seq > self.high_seq {
            // Forward slide: shift the bitmap to make room for the new tip.
            // Saturating shift handles the case where the slide exceeds the
            // window (`shift >= 64`) — bitmap empties, only the new tip
            // bit is set.
            let shift = seq - self.high_seq;
            self.bitmap = if shift >= 64 { 0 } else { self.bitmap << shift };
            self.bitmap |= 1; // mark the new tip as seen
            self.high_seq = seq;
        } else {
            let offset = self.high_seq - seq;
            if offset >= 64 {
                return Err(ReplayReject::DuplicateOrOutOfWindow);
            }
            let mask = 1u64 << offset;
            if self.bitmap & mask != 0 {
                return Err(ReplayReject::DuplicateOrOutOfWindow);
            }
            self.bitmap |= mask;
        }

        self.last_ts_ms = ts_ms;
        Ok(())
    }
}

/// A named signal channel with scope, access control, and merge strategy.
#[derive(Clone, Debug)]
pub struct SignalChannel {
    /// Unique ID within this table.
    pub id: ChannelId,
    /// Channel name (kept for UI display and cross-shard serialization).
    pub name: String,
    /// Current aggregated value (after merge).
    pub value: SignalValue,
    /// Previous tick's value (for change detection).
    prev_value: SignalValue,
    /// Running aggregation accumulator.
    pending: PendingAgg,
    /// How multiple publishers merge into one value.
    pub merge: ChannelMergeStrategy,
    /// Whether the value changed this tick.
    pub dirty: bool,
    /// Signal scope.
    pub scope: SignalScope,
    /// Who can publish to this channel.
    pub publish_policy: AccessPolicy,
    /// Who can subscribe to this channel.
    pub subscribe_policy: AccessPolicy,
    /// Who created this channel (player/session ID).
    pub owner_id: u64,
    /// 32-byte cryptographically-strong identity. Generated by `OsRng` at
    /// creation; never derived from name or owner_id (those are not secret).
    /// In Phase 3 this becomes the HMAC key for cross-shard publishes —
    /// possessing the signature is what authorizes a foreign actor to write
    /// to (or read from) the channel. The signature itself never travels
    /// the wire; only HMAC tags computed from it do.
    ///
    /// "Wiring" two blocks in-shard means binding both to the same
    /// `ChannelId` — the signature follows automatically because they're
    /// the same channel. "Wiring" across shards means copying the
    /// signature into a `RemoteAccessGrant` (Phase 3) and shipping the
    /// grant to the recipient.
    pub signature: [u8; 32],
    /// Replay defense: per-`(channel, sender_shard_id, grant_id)`
    /// 64-entry sliding window. Keyed by both sender AND grant so a
    /// publisher's "open" broadcast stream (grant_id=0) and the
    /// grant-stamped forwards from the same sender can share the wire
    /// without colliding on a single window.
    ///
    /// Per-grant separation is critical because Phase 3D's outbound
    /// subscriber forwarding uses a separate per-grant sequence counter
    /// (`outbound_subscriber_seq`) — receiver windows MUST be keyed
    /// symmetrically.
    ///
    /// Memory bound: ~32 bytes per (sender, grant) tuple × few-tuples-
    /// per-channel × thousands of channels ≈ a few MB worst-case.
    pub replay_state: HashMap<(u64, u64), ReplayWindow>,
    /// Monotonic sequence counter we stamp on our own outbound publishes
    /// to this channel. Persisted across reboots in the same redb table
    /// as the channel itself (a regression here would let captured
    /// pre-reboot tags replay; Phase 2's wire format extension persists
    /// this alongside the channel signature).
    pub outbound_seq: u64,
    /// Active cross-shard subscribers. When the channel becomes dirty,
    /// `signal_broadcast_remote` walks this list and forwards the value
    /// to each (HMAC-stamped under the grant's key). Local block-bound
    /// subscribers don't appear here — they read from the merged value
    /// in their own per-tick subscriber systems.
    ///
    /// Per-grant counters for outbound forwarding aren't kept on the
    /// channel itself — Phase 3D bumps a per-(channel, grant) sequence
    /// inside `outbound_subscriber_seq` (separate map) to give each
    /// remote subscriber its own monotonic stream.
    pub subscribers: SmallVec<[SubscriberRef; 8]>,
    /// Per-grant outbound sequence numbers used to stamp forwarded
    /// entries to remote subscribers. Indexed by grant_id; bumped on
    /// each forward so the receiver's replay window grows monotonically.
    /// Separate from `outbound_seq` (which counts our own publishes from
    /// `drain_remote_dirty`) so the two streams don't interleave.
    pub outbound_subscriber_seq: HashMap<u64, u64>,
}

// ---------------------------------------------------------------------------
// SignalChannelTable
// ---------------------------------------------------------------------------

/// All signal channels on a structure (ship, station, planet base).
/// One instance per shard that manages block data.
///
/// Channels are stored in a `Vec` indexed by `ChannelId` for O(1) access.
/// A `HashMap<String, ChannelId>` handles name→id resolution at config time.
#[derive(Resource, Default)]
pub struct SignalChannelTable {
    /// Slot array indexed by `ChannelId.0`.  `None` = freed slot.
    channels: Vec<Option<SignalChannel>>,
    /// Name → id registry for config-time resolution.
    name_to_id: HashMap<String, ChannelId>,
    /// Next id to allocate.
    next_id: u16,
    /// Channels marked dirty this tick. Avoids full-table scans in
    /// `drain_remote_dirty` and `clear_dirty`.
    dirty_set: Vec<ChannelId>,
}

impl SignalChannelTable {
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
            name_to_id: HashMap::new(),
            next_id: 0,
            dirty_set: Vec::new(),
        }
    }

    // -- ID-based hot-path API (O(1)) --------------------------------------

    /// Get a channel by ID (read-only).
    #[inline]
    pub fn get_by_id(&self, id: ChannelId) -> Option<&SignalChannel> {
        self.channels.get(id.0 as usize).and_then(|slot| slot.as_ref())
    }

    /// Get a channel by ID (mutable).
    #[inline]
    pub fn get_by_id_mut(&mut self, id: ChannelId) -> Option<&mut SignalChannel> {
        self.channels.get_mut(id.0 as usize).and_then(|slot| slot.as_mut())
    }

    /// Get a channel's name by ID.
    pub fn name_for_id(&self, id: ChannelId) -> Option<&str> {
        self.get_by_id(id).map(|ch| ch.name.as_str())
    }

    /// Push a value into a channel's running aggregation by ID (O(1)).
    #[inline]
    pub fn push_pending_id(&mut self, id: ChannelId, value: SignalValue) {
        if let Some(ch) = self.channels.get_mut(id.0 as usize).and_then(|s| s.as_mut()) {
            Self::accumulate(&mut ch.pending, ch.merge, value);
        }
    }

    /// Directly publish a value to a channel by ID (bypasses merge).
    pub fn publish_direct_id(&mut self, id: ChannelId, value: SignalValue) {
        if let Some(ch) = self.channels.get_mut(id.0 as usize).and_then(|s| s.as_mut()) {
            ch.prev_value = ch.value;
            ch.value = value;
            if ch.value != ch.prev_value {
                if !ch.dirty {
                    self.dirty_set.push(id);
                }
                ch.dirty = true;
            }
        }
    }

    // -- Name-based API (thin wrappers for config/cross-shard boundaries) --

    /// Resolve a channel name to its ID, or create a new channel.
    /// Primary entry point for config-time name→id resolution.
    pub fn resolve_or_create(
        &mut self,
        name: &str,
        scope: SignalScope,
        merge: ChannelMergeStrategy,
        owner_id: u64,
    ) -> ChannelId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = ChannelId(self.next_id);
        self.next_id += 1;
        let ch = SignalChannel {
            id,
            name: name.to_string(),
            value: SignalValue::default(),
            prev_value: SignalValue::default(),
            pending: PendingAgg::idle_for(merge),
            merge,
            dirty: false,
            scope,
            publish_policy: AccessPolicy::default(),
            subscribe_policy: AccessPolicy::default(),
            owner_id,
            // Mint a fresh 32-byte signature on every channel creation.
            // OsRng (cryptographically strong) is the right choice: the
            // signature anchors Phase 3's capability model, where holders
            // of the signature can authorize foreign actors to publish or
            // subscribe. Predictable signatures (e.g., hash of name) would
            // let an adversary pre-compute access keys.
            signature: {
                use rand::RngCore;
                let mut sig = [0u8; 32];
                rand::rngs::OsRng.fill_bytes(&mut sig);
                sig
            },
            replay_state: HashMap::new(),
            outbound_seq: 0,
            subscribers: SmallVec::new(),
            outbound_subscriber_seq: HashMap::new(),
        };
        if (id.0 as usize) >= self.channels.len() {
            self.channels.resize_with(id.0 as usize + 1, || None);
        }
        self.channels[id.0 as usize] = Some(ch);
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Resolve a name to its ID without creating.
    pub fn resolve(&self, name: &str) -> Option<ChannelId> {
        self.name_to_id.get(name).copied()
    }

    /// Reverse lookup: ID → name (for UI / serialization).
    pub fn name_of(&self, id: ChannelId) -> Option<&str> {
        self.get_by_id(id).map(|ch| ch.name.as_str())
    }

    /// Create or get a channel by name.  Returns `&mut SignalChannel`.
    pub fn get_or_create(
        &mut self,
        name: &str,
        scope: SignalScope,
        merge: ChannelMergeStrategy,
        owner_id: u64,
    ) -> &mut SignalChannel {
        let id = self.resolve_or_create(name, scope, merge, owner_id);
        self.channels[id.0 as usize].as_mut().unwrap()
    }

    /// Get a channel by name (read-only).
    pub fn get(&self, name: &str) -> Option<&SignalChannel> {
        let id = self.name_to_id.get(name)?;
        self.get_by_id(*id)
    }

    /// Get a channel by name (mutable).
    pub fn get_mut(&mut self, name: &str) -> Option<&mut SignalChannel> {
        let id = *self.name_to_id.get(name)?;
        self.get_by_id_mut(id)
    }

    /// Iterate every registered channel as `(name, id, current_value)`.
    /// Used by `WorldState.hud_signals` population — a snapshot-per-tick
    /// of all channels on the shard for client HUD widgets.
    pub fn iter_all(&self) -> impl Iterator<Item = (&str, ChannelId, SignalValue)> {
        self.name_to_id.iter().filter_map(|(name, &id)| {
            self.get_by_id(id).map(|ch| (name.as_str(), id, ch.value))
        })
    }

    /// Push a value by channel name (resolves or auto-creates with defaults).
    pub fn push_pending(&mut self, name: &str, value: SignalValue) {
        let id = self.resolve_or_create(
            name,
            SignalScope::default(),
            ChannelMergeStrategy::default(),
            0,
        );
        self.push_pending_id(id, value);
    }

    /// Permission-checked publish used by the client→server
    /// `SignalPublish` path. Returns:
    /// - `Ok(())` if the channel exists AND `publish_policy.allows`
    ///   the sender (value pushed into the pending aggregation).
    /// - `Err(PublishDenied::UnknownChannel)` if the channel isn't
    ///   registered — publishing to a nonexistent channel is always
    ///   denied (channels must be created by block config first).
    /// - `Err(PublishDenied::Forbidden)` if the policy rejects the
    ///   sender.
    ///
    /// Never auto-creates channels (prevents griefing: a malicious
    /// client spamming `SignalPublish { channel: "<random>" }` would
    /// otherwise exhaust channel slots).
    pub fn try_push_pending(
        &mut self,
        name: &str,
        value: SignalValue,
        sender_id: u64,
    ) -> Result<(), PublishDenied> {
        let Some(&id) = self.name_to_id.get(name) else {
            return Err(PublishDenied::UnknownChannel);
        };
        let Some(ch) = self.channels.get(id.0 as usize).and_then(|s| s.as_ref()) else {
            return Err(PublishDenied::UnknownChannel);
        };
        if !ch.publish_policy.allows(sender_id, ch.owner_id) {
            return Err(PublishDenied::Forbidden);
        }
        self.push_pending_id(id, value);
        Ok(())
    }

    /// Cross-shard ingress: validate a `SignalBroadcastBatch` entry against
    /// the local channel's scope, identity, and (Phase 3+) auth before
    /// accepting it into the pending aggregation. This is the single
    /// chokepoint for all foreign-shard publishes — every wire entry that
    /// arrives over QUIC funnels through here, so closing a hole here
    /// closes it for every shard topology.
    ///
    /// # Phase 1A enforces:
    /// - **`UnknownChannel`** — the wire name doesn't resolve locally.
    /// - **`LocalChannelImmutable`** — the local channel's scope is `Local`
    ///   and no grant covers it (Phase 1A: any Local channel; Phase 3+:
    ///   only when `grant_id == 0`).
    /// - **`ScopeMismatch`** — the `wire_scope_code` doesn't match the
    ///   local channel's scope class. Prevents a peer from re-tagging a
    ///   ShortRange packet as LongRange (or vice versa) to bypass spatial
    ///   filtering or relay restrictions.
    ///
    /// # Phase 1C+ adds:
    /// - **`StaleOrFutureTimestamp`** — `|now - timestamp_ms| > 5000`.
    /// - **`Replay`** — the `(channel, sender_shard_id, seq)` tuple was
    ///   already seen, or `seq` is older than the 64-entry sliding window.
    ///
    /// # Phase 3 adds:
    /// - **`AuthFailed`** — the `auth_tag` doesn't match the HMAC of the
    ///   payload under the channel's signature (or a matching grant's key
    ///   when `grant_id != 0`).
    ///
    /// `grants` is the shard's `GrantsRegistry` resource — consulted when
    /// the wire entry carries a non-zero `grant_id`. An empty / default
    /// registry preserves Phase 3A behavior end-to-end (any non-zero
    /// grant_id reference will fail to match and reject as `AuthFailed`).
    pub fn try_push_remote(
        &mut self,
        name: &str,
        value: SignalValue,
        wire_scope_code: u8,
        sender_shard_id: u64,
        timestamp_ms: u64,
        seq: u64,
        grant_id: u64,
        auth_tag: Option<&[u8]>,
        grants: &super::grants::GrantsRegistry,
    ) -> Result<(), RemoteIngressDenied> {
        let Some(&id) = self.name_to_id.get(name) else {
            return Err(RemoteIngressDenied::UnknownChannel);
        };
        let Some(ch) = self.channels.get(id.0 as usize).and_then(|s| s.as_ref()) else {
            return Err(RemoteIngressDenied::UnknownChannel);
        };

        // A Local-scoped channel only accepts a foreign publish when the
        // entry references an active grant covering it. This is the Phase 3B
        // unlock for the user's "remote control on a Local channel" story:
        // Bob's tablet → grant 0x… → Alice's chair's `local.thrust-forward`
        // is reachable iff Bob's grant is in Alice's `GrantsRegistry` AND
        // covers the channel AND ops includes Publish. The HMAC verification
        // a few lines below confirms Bob actually possesses the grant key.
        if matches!(ch.scope, SignalScope::Local) {
            if grant_id == 0 {
                return Err(RemoteIngressDenied::LocalChannelImmutable);
            }
            // Phase 3B grant-cover check happens in the HMAC block below
            // where we have all the wire fields canonicalized; if the
            // grant doesn't cover this channel (or is expired/revoked),
            // it'll surface as AuthFailed there. Order matters: the wire-
            // scope match below also has to allow the Local case when a
            // grant is presented.
        }

        // Scope-class match. The wire-scope code is what the sender claims;
        // the channel's `scope` field is what this shard authoritatively
        // declared. They must agree, or a peer could re-label packets to
        // dodge spatial / relay enforcement on either end.
        let local_scope_code = match ch.scope {
            SignalScope::Local => 0,
            SignalScope::ShortRange { .. } => 1,
            SignalScope::LongRange => 2,
            SignalScope::Radio { .. } => 3,
        };
        if wire_scope_code != local_scope_code {
            return Err(RemoteIngressDenied::ScopeMismatch);
        }

        // HMAC verification first — must run BEFORE the replay window
        // advances, otherwise a forged-tag adversary can burn legitimate
        // sequence numbers by submitting fresh-seq entries that pass replay
        // but later fail auth (the rejected entry's seq would have already
        // been recorded in the window, blocking the genuine publisher's
        // seq from going through).
        //
        // Cases:
        //   - `auth_tag` empty / `None` → unauthenticated. Accepted at
        //     this layer; channels that REQUIRE authenticated traffic
        //     gate at the access-policy layer (Phase 3 per-channel
        //     subscribe_policy / publish_policy).
        //   - `auth_tag` non-empty + `grant_id == 0` → verify against
        //     `ch.signature` (Phase 3A path for keyed channels whose
        //     signature is the channel-wide secret).
        //   - `auth_tag` non-empty + `grant_id != 0` → look up the grant
        //     in `GrantsRegistry::check_publish` (covers expiry, revoked,
        //     ops, channel match, all in O(1)). Verify against `grant.key`.
        //     This is the path that unlocks foreign-shard publish on a
        //     Local channel (the user's "remote pilot" use case).
        //
        // For Local channels reached via a grant, the wire must claim
        // scope=0 to pass the scope-class match above; the LocalChannel-
        // Immutable gate already skipped because grant_id != 0.
        if let Some(tag) = auth_tag {
            if !tag.is_empty() {
                let ch_ref = self
                    .channels
                    .get(id.0 as usize)
                    .and_then(|s| s.as_ref())
                    .expect("channel existed at the read above");
                let value_bits = match value {
                    SignalValue::Bool(b) => if b { 1.0_f32.to_bits() } else { 0.0_f32.to_bits() },
                    SignalValue::Float(f) => f.to_bits(),
                    SignalValue::State(s) => (s as f32).to_bits(),
                };
                let frequency = match ch_ref.scope {
                    SignalScope::Radio { frequency } => frequency,
                    _ => 0,
                };
                let value_type = match value {
                    SignalValue::Bool(_) => 0,
                    SignalValue::Float(_) => 1,
                    SignalValue::State(_) => 2,
                };

                // Pick the verification key — channel signature for the
                // open-keyed case, grant key for the capability case. The
                // grant lookup ALSO enforces ops/expiry/revoked/channel-
                // match so we don't have to re-check those here.
                let key_for_verify: [u8; 32] = if grant_id != 0 {
                    let now_ms = current_unix_millis();
                    match grants.check_publish(grant_id, id, now_ms) {
                        Some(k) => *k,
                        None => return Err(RemoteIngressDenied::AuthFailed),
                    }
                } else {
                    ch_ref.signature
                };

                if !crate::signal::auth::hmac_verify(
                    &key_for_verify,
                    name,
                    wire_scope_code,
                    frequency,
                    value_type,
                    value_bits,
                    timestamp_ms,
                    sender_shard_id,
                    seq,
                    grant_id,
                    tag,
                ) {
                    return Err(RemoteIngressDenied::AuthFailed);
                }
            } else if grant_id != 0 {
                // grant_id set but no tag → unauthenticated grant claim. Reject.
                return Err(RemoteIngressDenied::AuthFailed);
            }
        } else if grant_id != 0 {
            // grant_id set but no auth_tag at all → reject (capability
            // claim must be backed by HMAC proof).
            return Err(RemoteIngressDenied::AuthFailed);
        }

        // Replay window check + advance — runs ONLY after HMAC succeeded
        // (or was bypassed for an unauthenticated entry). Failed-auth
        // entries never reach this point, so they can't pollute the window.
        //
        // Only fires when both `seq` and `timestamp_ms` are non-zero.
        // Phase 1A wire entries (legacy senders that haven't been upgraded
        // to populate the new fields) carry seq=0 + ts=0 and bypass this
        // check.
        //
        // The timestamp comparison is signed-distance, so future-tagged
        // entries (clock skew or malicious "from the future" attempts)
        // also reject. Local clock comes from the system: a perfect
        // monotonic source isn't required because we tolerate ±5 s drift.
        if seq != 0 || timestamp_ms != 0 {
            let now_ms = current_unix_millis();
            let ch_mut = self
                .channels
                .get_mut(id.0 as usize)
                .and_then(|s| s.as_mut())
                .expect("channel existed at the read above");
            // Per-grant replay windows: each (sender, grant_id) tuple has
            // its own monotonic counter on the publishing side, so the
            // receiver tracks them independently here.
            let window = ch_mut
                .replay_state
                .entry((sender_shard_id, grant_id))
                .or_default();
            match window.try_accept(seq, timestamp_ms, now_ms) {
                Ok(()) => {}
                Err(ReplayReject::StaleOrFutureTimestamp) => {
                    return Err(RemoteIngressDenied::StaleOrFutureTimestamp);
                }
                Err(ReplayReject::DuplicateOrOutOfWindow) => {
                    return Err(RemoteIngressDenied::Replay);
                }
            }
        }

        self.push_pending_id(id, value);
        Ok(())
    }

    /// Directly publish a value by channel name (bypasses merge).
    pub fn publish_direct(&mut self, name: &str, value: SignalValue) {
        if let Some(&id) = self.name_to_id.get(name) {
            self.publish_direct_id(id, value);
        } else {
            // Auto-create with the published value.
            let id = self.resolve_or_create(
                name,
                SignalScope::default(),
                ChannelMergeStrategy::default(),
                0,
            );
            let ch = self.channels[id.0 as usize].as_mut().unwrap();
            ch.value = value;
            ch.dirty = true;
            self.dirty_set.push(id);
        }
    }

    // -- Tick lifecycle -----------------------------------------------------

    /// Update a running accumulator with a new value.
    fn accumulate(pending: &mut PendingAgg, merge: ChannelMergeStrategy, value: SignalValue) {
        match pending {
            PendingAgg::Idle => {
                *pending = PendingAgg::LastWrite(value);
            }
            PendingAgg::LastWrite(last) => {
                *last = value;
            }
            PendingAgg::Sum { total, count } => {
                *total += value.as_f32() as f64;
                *count += 1;
            }
            PendingAgg::Extreme { value: extreme, count } => {
                let v = value.as_f32() as f64;
                *extreme = match merge {
                    ChannelMergeStrategy::Max => extreme.max(v),
                    _ => extreme.min(v),
                };
                *count += 1;
            }
            PendingAgg::Bool { result, count } => {
                let b = value.as_bool();
                *result = match merge {
                    ChannelMergeStrategy::AnyTrue => *result || b,
                    _ => *result && b,
                };
                *count += 1;
            }
        }
    }

    /// Reset all pending accumulators (called at start of publish phase).
    pub fn clear_pending(&mut self) {
        for slot in &mut self.channels {
            if let Some(ch) = slot {
                ch.pending = PendingAgg::idle_for(ch.merge);
            }
        }
    }

    /// Finalize all pending aggregations into channel values.
    /// Marks channels as dirty only if the merged value differs from the previous tick.
    pub fn merge_pending(&mut self) {
        for slot in &mut self.channels {
            let Some(ch) = slot else { continue };
            if !ch.pending.has_values() {
                // No publisher wrote to this channel this tick.
                // Reset to neutral value so stale data doesn't persist.
                let neutral = SignalValue::Float(0.0);
                if ch.value != neutral {
                    ch.prev_value = ch.value;
                    ch.value = neutral;
                    if !ch.dirty {
                        self.dirty_set.push(ch.id);
                    }
                    ch.dirty = true;
                }
                continue;
            }
            let merged = match &ch.pending {
                PendingAgg::Idle => unreachable!(),
                PendingAgg::LastWrite(v) => *v,
                PendingAgg::Sum { total, count } => match ch.merge {
                    ChannelMergeStrategy::Average => {
                        SignalValue::Float((*total / *count as f64) as f32)
                    }
                    _ => SignalValue::Float(*total as f32),
                },
                PendingAgg::Extreme { value, .. } => SignalValue::Float(*value as f32),
                PendingAgg::Bool { result, .. } => SignalValue::Bool(*result),
            };
            ch.prev_value = ch.value;
            ch.value = merged;
            if ch.value != ch.prev_value {
                if !ch.dirty {
                    self.dirty_set.push(ch.id);
                }
                ch.dirty = true;
            }
        }
    }

    /// Clear all dirty flags (called after subscribe phase).
    /// Only iterates channels that were actually dirty — O(dirty_count).
    pub fn clear_dirty(&mut self) {
        for id in self.dirty_set.drain(..) {
            if let Some(Some(ch)) = self.channels.get_mut(id.0 as usize) {
                ch.dirty = false;
            }
        }
    }

    /// Directly set a channel's value (post-merge). Used by custom ship system blocks
    /// (flight computer, hover module, autopilot, engine controller) that read-modify-write
    /// channel values in their processing pass after signal_publish + merge_pending.
    pub fn set_value_direct(&mut self, id: ChannelId, value: SignalValue) {
        if let Some(ch) = self.channels.get_mut(id.0 as usize).and_then(|s| s.as_mut()) {
            if ch.value != value {
                ch.value = value;
                if !ch.dirty {
                    ch.dirty = true;
                    self.dirty_set.push(id);
                }
            }
        }
    }

    /// Read a channel's current value by ID (after merge). Returns Float(0.0) if not found.
    pub fn read_value(&self, id: ChannelId) -> SignalValue {
        self.get_by_id(id).map(|ch| ch.value).unwrap_or(SignalValue::Float(0.0))
    }

    /// Collect all dirty channels with non-Local scope (for network broadcast).
    /// Only iterates the dirty set — O(dirty_count) instead of O(total_channels).
    /// Mutates: bumps each emitted channel's `outbound_seq` by 1, and the
    /// post-bump value is what's stamped onto the wire entry. The same seq
    /// MUST NOT be reused for the same `(channel, sender_shard_id)` pair —
    /// the receiver's 64-entry replay window relies on monotonic increase.
    pub fn drain_remote_dirty(&mut self) -> Vec<RemoteDirtyEntry> {
        // Snapshot the dirty set so we can iterate while we mutate channel
        // entries. Cheap: typically a few entries per tick.
        let ids: Vec<ChannelId> = self.dirty_set.iter().copied().collect();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let Some(slot) = self.channels.get_mut(id.0 as usize) else { continue };
            let Some(ch) = slot.as_mut() else { continue };
            if matches!(ch.scope, SignalScope::Local) {
                continue;
            }
            ch.outbound_seq = ch.outbound_seq.saturating_add(1);
            out.push(RemoteDirtyEntry {
                name: ch.name.clone(),
                value: ch.value,
                scope: ch.scope,
                sequence: ch.outbound_seq,
                signature: ch.signature,
            });
        }
        out
    }

    // -- Introspection ------------------------------------------------------

    // -- Cross-shard subscriber management (Phase 3D) -----------------------

    /// Add a `RemoteShard` subscriber for a channel — or refresh the lease
    /// if the same `(shard_id, grant_id)` pair is already subscribed.
    /// Returns `true` iff the subscription was newly created (vs. renewed),
    /// useful for telemetry / "first subscriber" hooks.
    ///
    /// Caller is responsible for verifying the subscribe request's HMAC
    /// against the grant's key BEFORE calling this — this method trusts
    /// its inputs.
    pub fn add_remote_subscriber(
        &mut self,
        channel_id: ChannelId,
        shard_id: ShardId,
        grant_id: u64,
        valid_until_tick: u64,
    ) -> bool {
        let Some(ch) = self.channels.get_mut(channel_id.0 as usize).and_then(|s| s.as_mut())
        else {
            return false;
        };
        // Renew if (shard, grant) already present — lease updates in place.
        for sub in ch.subscribers.iter_mut() {
            if let SubscriberRef::RemoteShard {
                shard_id: s,
                grant_id: g,
                valid_until_tick: u,
            } = sub
            {
                if *s == shard_id && *g == grant_id {
                    *u = valid_until_tick;
                    return false;
                }
            }
        }
        ch.subscribers.push(SubscriberRef::RemoteShard {
            shard_id,
            grant_id,
            valid_until_tick,
        });
        true
    }

    /// Remove a `RemoteShard` subscriber. Returns `true` if it existed.
    pub fn remove_remote_subscriber(
        &mut self,
        channel_id: ChannelId,
        shard_id: ShardId,
        grant_id: u64,
    ) -> bool {
        let Some(ch) = self.channels.get_mut(channel_id.0 as usize).and_then(|s| s.as_mut())
        else {
            return false;
        };
        let before = ch.subscribers.len();
        ch.subscribers.retain(|sub| match sub {
            SubscriberRef::RemoteShard {
                shard_id: s,
                grant_id: g,
                ..
            } => !(*s == shard_id && *g == grant_id),
        });
        ch.subscribers.len() < before
    }

    /// Sweep every channel and drop subscribers whose lease has expired.
    /// O(channels × subscribers). Run at 1 Hz from the shard's schedule.
    pub fn cleanup_expired_subscribers(&mut self, now_tick: u64) -> usize {
        let mut dropped = 0;
        for slot in self.channels.iter_mut() {
            let Some(ch) = slot else { continue };
            let before = ch.subscribers.len();
            ch.subscribers.retain(|sub| match sub {
                SubscriberRef::RemoteShard { valid_until_tick, .. } => *valid_until_tick > now_tick,
            });
            dropped += before - ch.subscribers.len();
        }
        dropped
    }

    /// Bump the per-grant outbound subscriber sequence for a channel.
    /// Returns the post-bump value to stamp on the forwarded entry's
    /// `sequence` field. The receiver's replay window for
    /// `(channel, source_shard_id=our_id)` grows monotonically.
    pub fn next_subscriber_seq(&mut self, channel_id: ChannelId, grant_id: u64) -> u64 {
        let Some(ch) = self.channels.get_mut(channel_id.0 as usize).and_then(|s| s.as_mut())
        else {
            return 0;
        };
        let entry = ch.outbound_subscriber_seq.entry(grant_id).or_insert(0);
        *entry = entry.saturating_add(1);
        *entry
    }

    /// Iterate (channel_id, channel_ref) for every dirty channel without
    /// mutating the dirty set. Used by `signal_broadcast_remote` to
    /// snapshot per-channel subscribers BEFORE the mutating
    /// `drain_remote_dirty` call so the borrow doesn't conflict.
    pub fn iter_dirty(&self) -> impl Iterator<Item = (ChannelId, &SignalChannel)> {
        self.dirty_set.iter().filter_map(move |id| {
            self.channels
                .get(id.0 as usize)
                .and_then(|s| s.as_ref())
                .map(|ch| (*id, ch))
        })
    }

    /// Number of live channels.
    pub fn channel_count(&self) -> usize {
        self.name_to_id.len()
    }

    /// Number of dirty channels.
    pub fn dirty_count(&self) -> usize {
        self.dirty_set.len()
    }

    /// Total `SubscriberRef::RemoteShard` entries across every channel.
    /// Cheap O(channels) sweep; intended for the periodic gauge-sample
    /// path, not the per-tick hot loop. Used by the metrics subsystem to
    /// expose a `signal_remote_subscribers` gauge that operators can chart
    /// to detect runaway lease growth or stuck subscriptions.
    pub fn total_remote_subscriber_count(&self) -> usize {
        self.channels
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|ch| {
                ch.subscribers
                    .iter()
                    .filter(|s| matches!(s, SubscriberRef::RemoteShard { .. }))
                    .count()
            })
            .sum()
    }

    /// Remove a channel by name.
    pub fn remove(&mut self, name: &str) {
        if let Some(id) = self.name_to_id.remove(name) {
            if let Some(slot) = self.channels.get_mut(id.0 as usize) {
                *slot = None;
            }
        }
    }

    /// All channel names (for UI dropdowns).
    pub fn channel_names(&self) -> impl Iterator<Item = &str> {
        self.name_to_id.keys().map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_get_channel() {
        let mut table = SignalChannelTable::new();
        table.get_or_create("test", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        assert!(table.get("test").is_some());
        assert!(table.get("nonexistent").is_none());
    }

    #[test]
    fn resolve_assigns_stable_ids() {
        let mut table = SignalChannelTable::new();
        let id1 = table.resolve_or_create("a", SignalScope::Local, ChannelMergeStrategy::Sum, 1);
        let id2 = table.resolve_or_create("b", SignalScope::Local, ChannelMergeStrategy::Sum, 1);
        let id1_again = table.resolve_or_create("a", SignalScope::Local, ChannelMergeStrategy::Sum, 1);
        assert_eq!(id1, id1_again);
        assert_ne!(id1, id2);
    }

    #[test]
    fn get_by_id_roundtrip() {
        let mut table = SignalChannelTable::new();
        let id = table.resolve_or_create("ch", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        assert_eq!(table.get_by_id(id).unwrap().name, "ch");
        assert_eq!(table.name_of(id), Some("ch"));
    }

    #[test]
    fn push_and_merge_sum() {
        let mut table = SignalChannelTable::new();
        let id = table.resolve_or_create("thrust", SignalScope::Local, ChannelMergeStrategy::Sum, 1);

        table.clear_pending();
        table.push_pending_id(id, SignalValue::Float(100.0));
        table.push_pending_id(id, SignalValue::Float(200.0));
        table.merge_pending();

        let ch = table.get_by_id(id).unwrap();
        assert_eq!(ch.value, SignalValue::Float(300.0));
        assert!(ch.dirty);
    }

    #[test]
    fn push_pending_by_name() {
        let mut table = SignalChannelTable::new();
        table.get_or_create("thrust", SignalScope::Local, ChannelMergeStrategy::Sum, 1);

        table.clear_pending();
        table.push_pending("thrust", SignalValue::Float(100.0));
        table.push_pending("thrust", SignalValue::Float(200.0));
        table.merge_pending();

        let ch = table.get("thrust").unwrap();
        assert_eq!(ch.value, SignalValue::Float(300.0));
        assert!(ch.dirty);
    }

    #[test]
    fn dirty_only_on_change() {
        let mut table = SignalChannelTable::new();
        table.get_or_create("stable", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);

        // First publish: changes from default (0.0) to 5.0.
        table.clear_pending();
        table.push_pending("stable", SignalValue::Float(5.0));
        table.merge_pending();
        assert!(table.get("stable").unwrap().dirty);

        table.clear_dirty();

        // Second publish: same value → not dirty.
        table.clear_pending();
        table.push_pending("stable", SignalValue::Float(5.0));
        table.merge_pending();
        assert!(!table.get("stable").unwrap().dirty);
    }

    #[test]
    fn publish_direct() {
        let mut table = SignalChannelTable::new();
        table.publish_direct("alarm", SignalValue::Bool(true));

        let ch = table.get("alarm").unwrap();
        assert_eq!(ch.value, SignalValue::Bool(true));
        assert!(ch.dirty);
    }

    #[test]
    fn drain_remote_dirty() {
        let mut table = SignalChannelTable::new();
        table.get_or_create("local", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        table.get_or_create("beacon", SignalScope::ShortRange { range_m: 2000.0 }, ChannelMergeStrategy::LastWrite, 1);

        table.clear_pending();
        table.push_pending("local", SignalValue::Bool(true));
        table.push_pending("beacon", SignalValue::Bool(true));
        table.merge_pending();

        let remote = table.drain_remote_dirty();
        assert_eq!(remote.len(), 1);
        assert_eq!(remote[0].name, "beacon");
        // Each emission bumps the channel's outbound_seq monotonically.
        assert_eq!(remote[0].sequence, 1);
        // Signature was minted at creation and travels with the entry.
        assert_ne!(remote[0].signature, [0u8; 32]);
    }

    #[test]
    fn drain_remote_dirty_bumps_outbound_seq_per_emission() {
        // Each broadcast pass increments the channel's outbound_seq by 1
        // — receivers' replay windows depend on monotonic growth across
        // the (channel, sender_shard_id) pair.
        let mut table = SignalChannelTable::new();
        table.get_or_create(
            "beacon",
            SignalScope::ShortRange { range_m: 2000.0 },
            ChannelMergeStrategy::LastWrite,
            1,
        );

        // Publish + drain three times; assert seq goes 1, 2, 3.
        for expected_seq in 1..=3 {
            table.clear_pending();
            // Alternate values so each tick is genuinely dirty.
            table.push_pending("beacon", SignalValue::Float(expected_seq as f32));
            table.merge_pending();
            let remote = table.drain_remote_dirty();
            assert_eq!(remote.len(), 1);
            assert_eq!(remote[0].sequence, expected_seq);
            table.clear_dirty();
        }
    }

    // -- try_push_remote / RemoteIngressDenied --------------------------------

    /// Helper: build a table with one channel of the given scope.
    fn table_with_channel(name: &str, scope: SignalScope) -> SignalChannelTable {
        let mut t = SignalChannelTable::new();
        t.get_or_create(name, scope, ChannelMergeStrategy::LastWrite, 1);
        t
    }

    #[test]
    fn try_push_remote_rejects_unknown_channel() {
        let mut t = SignalChannelTable::new();
        let err = t
            .try_push_remote(
                "no.such.channel",
                SignalValue::Bool(true),
                /*wire_scope=*/ 1,
                /*sender=*/ 42,
                /*ts=*/ 0,
                /*seq=*/ 0,
                /*grant=*/ 0,
                None,
                &crate::signal::grants::GrantsRegistry::default(),
            )
            .expect_err("publish to unknown channel must be rejected");
        assert_eq!(err, RemoteIngressDenied::UnknownChannel);
    }

    #[test]
    fn try_push_remote_rejects_local_scope_unconditionally() {
        // The user's #1 reported hole: cross-shard signal_publish blindly
        // creating Local channels via push_pending. After Phase 1A any
        // Local channel rejects remote ingress regardless of wire scope.
        let mut t = table_with_channel("priv", SignalScope::Local);
        let err = t
            .try_push_remote("priv", SignalValue::Bool(true), 0, 1, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect_err("Local channels must reject remote ingress");
        assert_eq!(err, RemoteIngressDenied::LocalChannelImmutable);
    }

    #[test]
    fn try_push_remote_rejects_scope_mismatch_short_to_long() {
        // ShortRange channel + wire claim of LongRange (scope=2) should be
        // dropped — a peer must not be able to re-label a packet to bypass
        // spatial filtering on either side.
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        let err = t
            .try_push_remote(
                "beacon",
                SignalValue::Bool(true),
                /*wire_scope=*/ 2,
                1, 0, 0, 0, None,
                &crate::signal::grants::GrantsRegistry::default(),
            )
            .expect_err("scope re-labeling must be rejected");
        assert_eq!(err, RemoteIngressDenied::ScopeMismatch);
    }

    #[test]
    fn try_push_remote_rejects_scope_mismatch_radio_to_short() {
        let mut t = table_with_channel("hail", SignalScope::Radio { frequency: 91100 });
        let err = t
            .try_push_remote("hail", SignalValue::Bool(true), 1, 1, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect_err("Radio channel must not accept ShortRange-tagged wire entries");
        assert_eq!(err, RemoteIngressDenied::ScopeMismatch);
    }

    #[test]
    fn try_push_remote_accepts_matching_short_range() {
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        t.try_push_remote("beacon", SignalValue::Float(0.7), 1, 99, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("matching scope should accept");
        // Run the merge to surface the value.
        t.merge_pending();
        assert_eq!(t.get("beacon").unwrap().value, SignalValue::Float(0.7));
    }

    #[test]
    fn try_push_remote_accepts_matching_long_range() {
        let mut t = table_with_channel("alert", SignalScope::LongRange);
        t.try_push_remote("alert", SignalValue::Bool(true), 2, 99, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("matching scope should accept");
        t.merge_pending();
        assert_eq!(t.get("alert").unwrap().value, SignalValue::Bool(true));
    }

    #[test]
    fn try_push_remote_accepts_matching_radio() {
        let mut t = table_with_channel("hail", SignalScope::Radio { frequency: 91100 });
        t.try_push_remote("hail", SignalValue::Float(0.5), 3, 99, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("matching scope should accept");
        t.merge_pending();
        assert_eq!(t.get("hail").unwrap().value, SignalValue::Float(0.5));
    }

    /// Phase 1A regression test: even though the wire entry claims Local
    /// (scope=0), the receive path rejects unconditionally because Local
    /// channels are sealed regardless of the wire claim. Phase 3 grants
    /// are how "remote control of a Local channel" gets unlocked, not
    /// scope-trickery.
    #[test]
    fn try_push_remote_local_with_local_wire_still_rejects() {
        let mut t = table_with_channel("priv", SignalScope::Local);
        let err = t
            .try_push_remote("priv", SignalValue::Bool(true), 0, 1, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect_err("Local channels stay sealed in Phase 1A");
        assert_eq!(err, RemoteIngressDenied::LocalChannelImmutable);
    }

    #[test]
    fn channel_signatures_are_unique_per_creation() {
        // Two channels with the same name parameters MUST get distinct
        // signatures. Even if some adversary somehow forced reuse of the
        // channel name across shard restarts, the new signature would
        // invalidate any captured Phase-3 HMAC tags from before.
        let mut t1 = SignalChannelTable::new();
        t1.get_or_create("foo", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        let sig1 = t1.get("foo").unwrap().signature;

        let mut t2 = SignalChannelTable::new();
        t2.get_or_create("foo", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        let sig2 = t2.get("foo").unwrap().signature;

        assert_ne!(sig1, sig2, "OsRng must mint distinct signatures across tables");
        assert_ne!(sig1, [0u8; 32], "signature must not be all-zero");
    }

    // -- ReplayWindow -----------------------------------------------------

    #[test]
    fn replay_window_accepts_first_seq() {
        let mut w = ReplayWindow::default();
        w.try_accept(1, 1000, 1000).expect("first seq must accept");
        assert_eq!(w.high_seq, 1);
        assert_eq!(w.last_ts_ms, 1000);
    }

    #[test]
    fn replay_window_rejects_duplicate() {
        let mut w = ReplayWindow::default();
        w.try_accept(5, 1000, 1000).unwrap();
        // Same seq again must reject — even though it's within the window.
        let err = w
            .try_accept(5, 1000, 1000)
            .expect_err("duplicate seq must reject");
        assert_eq!(err, ReplayReject::DuplicateOrOutOfWindow);
    }

    #[test]
    fn replay_window_accepts_out_of_order_within_window() {
        let mut w = ReplayWindow::default();
        // Send 10 first, then 5 (out-of-order but within 64 of 10).
        w.try_accept(10, 1000, 1000).unwrap();
        w.try_accept(5, 1000, 1000).expect("seq within window must accept");
        // Both bits should be set (bit 0 for seq=10, bit 5 for seq=5).
        assert_eq!(w.bitmap & 1, 1);
        assert_eq!(w.bitmap & (1 << 5), 1 << 5);
    }

    #[test]
    fn replay_window_rejects_below_window() {
        let mut w = ReplayWindow::default();
        w.try_accept(100, 1000, 1000).unwrap();
        // seq 35 is `100 - 35 = 65` below high_seq, outside the 64-entry
        // window — must reject.
        let err = w
            .try_accept(35, 1000, 1000)
            .expect_err("seq older than window must reject");
        assert_eq!(err, ReplayReject::DuplicateOrOutOfWindow);
    }

    #[test]
    fn replay_window_rejects_stale_timestamp() {
        let mut w = ReplayWindow::default();
        // ts_ms = 0, now_ms = 10_000 → drift = 10s, outside the 5s window.
        let err = w
            .try_accept(1, 0, 10_000)
            .expect_err("stale timestamp must reject");
        assert_eq!(err, ReplayReject::StaleOrFutureTimestamp);
    }

    #[test]
    fn replay_window_rejects_future_timestamp() {
        let mut w = ReplayWindow::default();
        // ts_ms = 20_000, now_ms = 10_000 → drift = 10s in the future.
        // Defends against clock-skew exploits + malicious "from the future" tags.
        let err = w
            .try_accept(1, 20_000, 10_000)
            .expect_err("future timestamp must reject");
        assert_eq!(err, ReplayReject::StaleOrFutureTimestamp);
    }

    #[test]
    fn replay_window_rejected_entry_does_not_pollute_state() {
        // Sequence-space-burning attack: a malicious peer sends a wildly
        // future seq with a stale timestamp. The window must NOT advance —
        // otherwise legitimate subsequent traffic would fall outside the
        // window and reject as collateral damage.
        let mut w = ReplayWindow::default();
        let now = REPLAY_TIMESTAMP_WINDOW_MS + 10_000;
        // Bootstrap with a fresh entry.
        w.try_accept(10, now, now).unwrap();
        // Try to advance with a stale ts (outside the 5s window).
        let stale_ts = now - REPLAY_TIMESTAMP_WINDOW_MS - 1;
        let _ = w.try_accept(10_000, stale_ts, now);
        // State is unchanged: stale rejection happens BEFORE seq bookkeeping.
        assert_eq!(w.high_seq, 10);
        // Legitimate seq 11 still accepts.
        w.try_accept(11, now, now)
            .expect("post-rejection traffic must still flow");
    }

    #[test]
    fn replay_window_handles_large_forward_jump() {
        // Bitmap shift > 64 should empty the bitmap rather than overflow.
        let mut w = ReplayWindow::default();
        w.try_accept(5, 1000, 1000).unwrap();
        w.try_accept(1_000_000, 1000, 1000).unwrap();
        assert_eq!(w.high_seq, 1_000_000);
        // Only the new tip bit is set.
        assert_eq!(w.bitmap, 1);
    }

    #[test]
    fn try_push_remote_skips_replay_when_seq_and_ts_zero() {
        // Phase 1A compat: legacy senders with seq=0 + ts=0 bypass the
        // replay check. Verifies that two consecutive zero-tagged entries
        // both land (no spurious "duplicate" rejection).
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        t.try_push_remote("beacon", SignalValue::Bool(true), 1, 99, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("first zero-tagged accepts");
        t.try_push_remote("beacon", SignalValue::Bool(true), 1, 99, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("second zero-tagged accepts (replay check skipped)");
    }

    #[test]
    fn try_push_remote_rejects_replay_when_seq_set() {
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        let now = current_unix_millis();
        // First publish at seq=1 with current ts.
        t.try_push_remote("beacon", SignalValue::Bool(true), 1, 99, now, 1, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("first seq=1 must accept");
        // Replay of the same (sender, seq, ts) — duplicate rejection.
        let err = t
            .try_push_remote("beacon", SignalValue::Bool(true), 1, 99, now, 1, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect_err("replay must reject");
        assert_eq!(err, RemoteIngressDenied::Replay);
    }

    #[test]
    fn try_push_remote_per_sender_independent_windows() {
        // Two distinct senders publishing the same seq must not interfere.
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        let now = current_unix_millis();
        t.try_push_remote("beacon", SignalValue::Bool(true), 1, 100, now, 1, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("sender 100 seq 1 accepts");
        t.try_push_remote("beacon", SignalValue::Bool(true), 1, 200, now, 1, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("sender 200 seq 1 also accepts (independent window)");
        // But each sender's own replay still rejects.
        let err = t
            .try_push_remote("beacon", SignalValue::Bool(true), 1, 100, now, 1, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect_err("sender 100 replay must reject");
        assert_eq!(err, RemoteIngressDenied::Replay);
    }

    // -- HMAC verification path -----------------------------------------------

    #[test]
    fn try_push_remote_accepts_valid_hmac_tag() {
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        let key = t.get("beacon").unwrap().signature;
        let now = current_unix_millis();
        let value = SignalValue::Float(0.7);
        let tag = crate::signal::auth::hmac_sign(
            &key,
            "beacon",
            1,
            0,
            1,
            0.7_f32.to_bits(),
            now,
            99,
            1,
            0,
        );
        t.try_push_remote("beacon", value, 1, 99, now, 1, 0, Some(&tag), &crate::signal::grants::GrantsRegistry::default())
            .expect("matching HMAC must accept");
        t.merge_pending();
        assert_eq!(t.get("beacon").unwrap().value, value);
    }

    #[test]
    fn try_push_remote_rejects_tampered_value_with_valid_tag() {
        // Adversary captured a valid (channel, value=0.5) tag and tries to
        // re-submit with a different value. HMAC over the actual payload
        // catches the tamper.
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        let key = t.get("beacon").unwrap().signature;
        let now = current_unix_millis();
        let original_value = SignalValue::Float(0.5);
        let tag = crate::signal::auth::hmac_sign(
            &key,
            "beacon",
            1,
            0,
            1,
            0.5_f32.to_bits(),
            now,
            99,
            1,
            0,
        );
        let tampered = SignalValue::Float(0.99);
        let err = t
            .try_push_remote("beacon", tampered, 1, 99, now, 1, 0, Some(&tag), &crate::signal::grants::GrantsRegistry::default())
            .expect_err("tampered value must reject");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
        // And the genuine value still works.
        t.try_push_remote("beacon", original_value, 1, 99, now, 1, 0, Some(&tag), &crate::signal::grants::GrantsRegistry::default())
            .expect("genuine value with matching tag must accept");
        let _ = original_value; // suppress unused
    }

    #[test]
    fn try_push_remote_rejects_grant_id_without_registry() {
        // Phase 3A: any non-zero grant_id rejects because there's no
        // GrantsRegistry yet to look it up. Phase 3B activates the path.
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        let key = t.get("beacon").unwrap().signature;
        let now = current_unix_millis();
        let tag = crate::signal::auth::hmac_sign(
            &key, "beacon", 1, 0, 1, 0u32, now, 99, 1, 99999,
        );
        let err = t
            .try_push_remote(
                "beacon",
                SignalValue::Float(0.0),
                1,
                99,
                now,
                1,
                /*grant_id=*/ 99999,
                Some(&tag),
                &crate::signal::grants::GrantsRegistry::default(),
            )
            .expect_err("non-zero grant_id rejects in Phase 3A");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
    }

    // -- Phase 3B: grant-based access on Local channels ---------------------

    #[test]
    fn grant_unlocks_local_channel_publish() {
        // The user's headline use case: Bob's tablet publishes to Alice's
        // Local-scoped channel (e.g., chair's `local.thrust-forward`)
        // because Alice issued Bob a grant covering it.
        use crate::signal::grants::{GrantOps, GrantsRegistry, RemoteAccessGrant};
        use smallvec::smallvec;
        let mut t = table_with_channel("alice.thrust-forward", SignalScope::Local);
        let id = t.resolve("alice.thrust-forward").unwrap();
        // Grant key separate from channel signature — that's the point of
        // grants: per-recipient revocation without re-keying the channel.
        let grant_key = [0xBB; 32];
        let grant = RemoteAccessGrant {
            grant_id: 0xCAFE,
            key: grant_key,
            channels: smallvec![id],
            namespace_glob: None,
            ops: GrantOps::Both,
            label: "Bob".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        };
        let mut grants = GrantsRegistry::default();
        grants.insert(grant);

        // Bob computes HMAC under the grant key and sends.
        let now = current_unix_millis();
        let value = SignalValue::Float(1.0);
        let tag = crate::signal::auth::hmac_sign(
            &grant_key,
            "alice.thrust-forward",
            /*scope_code=*/ 0,
            /*frequency=*/ 0,
            /*value_type=*/ 1,
            1.0_f32.to_bits(),
            now,
            /*sender=*/ 99,
            /*seq=*/ 1,
            /*grant_id=*/ 0xCAFE,
        );
        t.try_push_remote(
            "alice.thrust-forward",
            value,
            /*scope=*/ 0, // Local
            /*sender=*/ 99,
            now,
            /*seq=*/ 1,
            /*grant=*/ 0xCAFE,
            Some(&tag),
            &grants,
        )
        .expect("grant-bearing publish to a Local channel must accept");
    }

    #[test]
    fn grant_revoked_locks_local_channel_publish() {
        // Same setup as above, then Alice revokes — the next publish from
        // Bob with the same key fails as AuthFailed.
        use crate::signal::grants::{GrantOps, GrantsRegistry, RemoteAccessGrant};
        use smallvec::smallvec;
        let mut t = table_with_channel("alice.door.open", SignalScope::Local);
        let id = t.resolve("alice.door.open").unwrap();
        let grant_key = [0x33; 32];
        let mut grants = GrantsRegistry::default();
        grants.insert(RemoteAccessGrant {
            grant_id: 7,
            key: grant_key,
            channels: smallvec![id],
            namespace_glob: None,
            ops: GrantOps::Both,
            label: "Bob".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        });
        // Alice revokes.
        assert!(grants.revoke(7));

        let now = current_unix_millis();
        let tag = crate::signal::auth::hmac_sign(
            &grant_key, "alice.door.open", 0, 0, 0, 1.0_f32.to_bits(), now, 99, 1, 7,
        );
        let err = t
            .try_push_remote(
                "alice.door.open",
                SignalValue::Bool(true),
                0,
                99,
                now,
                1,
                7,
                Some(&tag),
                &grants,
            )
            .expect_err("revoked grant must reject");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
    }

    #[test]
    fn grant_publish_only_blocks_local_when_subscribe_only_grant() {
        // A subscribe-only grant must NOT let the holder publish, even on
        // a Local channel they have *some* grant for. Defense in depth.
        use crate::signal::grants::{GrantOps, GrantsRegistry, RemoteAccessGrant};
        use smallvec::smallvec;
        let mut t = table_with_channel("alice.health", SignalScope::Local);
        let id = t.resolve("alice.health").unwrap();
        let grant_key = [0x55; 32];
        let mut grants = GrantsRegistry::default();
        grants.insert(RemoteAccessGrant {
            grant_id: 33,
            key: grant_key,
            channels: smallvec![id],
            namespace_glob: None,
            ops: GrantOps::Subscribe,
            label: "monitor".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        });
        let now = current_unix_millis();
        let tag = crate::signal::auth::hmac_sign(
            &grant_key, "alice.health", 0, 0, 0, 1.0_f32.to_bits(), now, 99, 1, 33,
        );
        let err = t
            .try_push_remote(
                "alice.health", SignalValue::Bool(true), 0, 99, now, 1, 33, Some(&tag),
                &grants,
            )
            .expect_err("subscribe-only grant must reject publish");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
    }

    #[test]
    fn grant_id_with_no_tag_rejects() {
        // grant_id claim without an auth_tag is unauthenticated — reject
        // immediately (a grant claim is a capability assertion that MUST
        // be backed by HMAC proof).
        let mut t = table_with_channel("alice.door", SignalScope::Local);
        let now = current_unix_millis();
        let err = t
            .try_push_remote(
                "alice.door", SignalValue::Bool(true), 0, 99, now, 1, 42, None,
                &crate::signal::grants::GrantsRegistry::default(),
            )
            .expect_err("grant_id without tag must reject");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
    }

    #[test]
    fn grant_id_with_empty_tag_rejects() {
        let mut t = table_with_channel("alice.door", SignalScope::Local);
        let now = current_unix_millis();
        let err = t
            .try_push_remote(
                "alice.door",
                SignalValue::Bool(true),
                0,
                99,
                now,
                1,
                42,
                Some(&[]),
                &crate::signal::grants::GrantsRegistry::default(),
            )
            .expect_err("grant_id with empty tag must reject");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
    }

    #[test]
    fn grant_for_different_channel_rejects() {
        // Bob has a grant covering channel A but tries to publish to
        // channel B (also owned by Alice). The HMAC verifies under the
        // grant key but the channel-cover check rejects.
        use crate::signal::grants::{GrantOps, GrantsRegistry, RemoteAccessGrant};
        use smallvec::smallvec;
        let mut t = table_with_channel("alice.door", SignalScope::Local);
        t.get_or_create("alice.alarm", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        let door_id = t.resolve("alice.door").unwrap();
        let grant_key = [0x77; 32];
        let mut grants = GrantsRegistry::default();
        grants.insert(RemoteAccessGrant {
            grant_id: 11,
            key: grant_key,
            channels: smallvec![door_id],
            namespace_glob: None,
            ops: GrantOps::Both,
            label: "door-only".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        });
        let now = current_unix_millis();
        let tag = crate::signal::auth::hmac_sign(
            &grant_key, "alice.alarm", 0, 0, 0, 1.0_f32.to_bits(), now, 99, 1, 11,
        );
        let err = t
            .try_push_remote(
                "alice.alarm", SignalValue::Bool(true), 0, 99, now, 1, 11, Some(&tag),
                &grants,
            )
            .expect_err("grant for different channel must reject");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
    }

    #[test]
    fn forged_tag_does_not_burn_replay_window() {
        // Regression: HMAC must run before replay-window advance. Otherwise
        // an attacker without the key can submit a fresh-seq forged tag,
        // the replay window advances on what looked like a "first-time"
        // seq, then HMAC rejects — but the genuine publisher's matching
        // seq is now blocked as a duplicate.
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        let key = t.get("beacon").unwrap().signature;
        let now = current_unix_millis();
        // Adversary's bogus tag (random bytes). Replay would normally
        // accept seq=1 as fresh; HMAC must catch this first.
        let bogus = [0xAA_u8; 16];
        let err = t
            .try_push_remote(
                "beacon",
                SignalValue::Float(0.5),
                1,
                99,
                now,
                1,
                0,
                Some(&bogus),
                &crate::signal::grants::GrantsRegistry::default(),
            )
            .expect_err("forged tag must reject");
        assert_eq!(err, RemoteIngressDenied::AuthFailed);
        // The genuine publisher's seq=1 with valid tag must still go through.
        let real_tag = crate::signal::auth::hmac_sign(
            &key, "beacon", 1, 0, 1, 0.5_f32.to_bits(), now, 99, 1, 0,
        );
        t.try_push_remote(
            "beacon",
            SignalValue::Float(0.5),
            1, 99, now, 1, 0,
            Some(&real_tag),
            &crate::signal::grants::GrantsRegistry::default(),
        )
        .expect("genuine seq=1 must pass after the forged-tag rejection");
    }

    #[test]
    fn try_push_remote_unauthenticated_passes_through() {
        // Empty auth_tag means "this sender didn't sign". Today's flow
        // accepts at this layer — Phase 3B can add per-channel "auth
        // required" gating where appropriate (e.g., Radio channels with
        // a key-required policy).
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        t.try_push_remote("beacon", SignalValue::Bool(true), 1, 99, 0, 0, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect("None auth_tag accepts (legacy unauthenticated)");
        t.try_push_remote("beacon", SignalValue::Bool(true), 1, 99, 0, 0, 0, Some(&[]), &crate::signal::grants::GrantsRegistry::default())
            .expect("empty auth_tag accepts (legacy unauthenticated)");
    }

    #[test]
    fn try_push_remote_rejects_stale_timestamp() {
        let mut t = table_with_channel("beacon", SignalScope::ShortRange { range_m: 2000.0 });
        // ts = 0 ms, sender_shard, seq = 1 — clearly stale unless system
        // clock is January 1970 + 5 seconds (it isn't).
        let err = t
            .try_push_remote("beacon", SignalValue::Bool(true), 1, 99, 1, 1, 0, None, &crate::signal::grants::GrantsRegistry::default())
            .expect_err("stale ts must reject");
        assert_eq!(err, RemoteIngressDenied::StaleOrFutureTimestamp);
    }

    // -- SubscriberRef + lease management -------------------------------------

    #[test]
    fn add_remote_subscriber_creates_then_renews() {
        let mut t = table_with_channel("alice.lights", SignalScope::Local);
        let id = t.resolve("alice.lights").unwrap();
        let bob = ShardId(99);

        // First add: created=true.
        let created = t.add_remote_subscriber(id, bob, /*grant=*/ 1, /*until=*/ 100);
        assert!(created, "first add must report created=true");
        assert_eq!(t.get_by_id(id).unwrap().subscribers.len(), 1);

        // Same (shard, grant) again: renewed (created=false), lease updates.
        let renewed = t.add_remote_subscriber(id, bob, 1, 200);
        assert!(!renewed, "re-add of same (shard, grant) is renewal not create");
        assert_eq!(t.get_by_id(id).unwrap().subscribers.len(), 1);
        let SubscriberRef::RemoteShard { valid_until_tick, .. } =
            &t.get_by_id(id).unwrap().subscribers[0];
        assert_eq!(*valid_until_tick, 200, "lease must update in place");

        // Different grant, same shard → distinct subscription.
        let created2 = t.add_remote_subscriber(id, bob, /*grant=*/ 2, 100);
        assert!(created2);
        assert_eq!(t.get_by_id(id).unwrap().subscribers.len(), 2);
    }

    #[test]
    fn total_remote_subscriber_count_sums_across_channels() {
        let mut t = SignalChannelTable::default();
        t.resolve_or_create("a", SignalScope::Local, ChannelMergeStrategy::LastWrite, 0);
        t.resolve_or_create("b", SignalScope::Local, ChannelMergeStrategy::LastWrite, 0);
        t.resolve_or_create("c", SignalScope::Local, ChannelMergeStrategy::LastWrite, 0);
        let ia = t.resolve("a").unwrap();
        let ib = t.resolve("b").unwrap();
        // Channel `a` has two distinct grants → 2 subscriber refs.
        t.add_remote_subscriber(ia, ShardId(1), 100, 1_000);
        t.add_remote_subscriber(ia, ShardId(1), 101, 1_000);
        // Channel `b` has one subscriber.
        t.add_remote_subscriber(ib, ShardId(2), 200, 1_000);
        // Channel `c` has none.
        assert_eq!(t.total_remote_subscriber_count(), 3);

        // After cleanup of expired leases, the gauge tracks the drop.
        let dropped = t.cleanup_expired_subscribers(/*now_tick=*/ 2_000);
        assert_eq!(dropped, 3);
        assert_eq!(t.total_remote_subscriber_count(), 0);
    }

    #[test]
    fn remove_remote_subscriber_removes_only_match() {
        let mut t = table_with_channel("alice.lights", SignalScope::Local);
        let id = t.resolve("alice.lights").unwrap();
        let bob = ShardId(99);
        let charlie = ShardId(100);
        t.add_remote_subscriber(id, bob, 1, 100);
        t.add_remote_subscriber(id, charlie, 2, 100);
        // Remove (bob, grant 1) — Charlie's subscription survives.
        let removed = t.remove_remote_subscriber(id, bob, 1);
        assert!(removed);
        let subs = &t.get_by_id(id).unwrap().subscribers;
        assert_eq!(subs.len(), 1);
        let SubscriberRef::RemoteShard { shard_id, grant_id, .. } = &subs[0];
        assert_eq!(*shard_id, charlie);
        assert_eq!(*grant_id, 2);
        // Removing again is a no-op.
        let removed_again = t.remove_remote_subscriber(id, bob, 1);
        assert!(!removed_again);
    }

    #[test]
    fn cleanup_expired_subscribers_drops_stale() {
        let mut t = table_with_channel("alice.lights", SignalScope::Local);
        let id = t.resolve("alice.lights").unwrap();
        // Two subs — different lease deadlines.
        t.add_remote_subscriber(id, ShardId(1), /*grant=*/ 1, /*until=*/ 50);
        t.add_remote_subscriber(id, ShardId(2), /*grant=*/ 2, /*until=*/ 200);
        let dropped = t.cleanup_expired_subscribers(/*now=*/ 100);
        assert_eq!(dropped, 1);
        let subs = &t.get_by_id(id).unwrap().subscribers;
        assert_eq!(subs.len(), 1);
        let SubscriberRef::RemoteShard { shard_id, .. } = &subs[0];
        assert_eq!(*shard_id, ShardId(2));
    }

    #[test]
    fn next_subscriber_seq_is_per_grant_monotonic() {
        let mut t = table_with_channel("alice.lights", SignalScope::Local);
        let id = t.resolve("alice.lights").unwrap();
        // Per-grant counters bump independently.
        assert_eq!(t.next_subscriber_seq(id, /*grant=*/ 1), 1);
        assert_eq!(t.next_subscriber_seq(id, 1), 2);
        assert_eq!(t.next_subscriber_seq(id, /*grant=*/ 2), 1);
        assert_eq!(t.next_subscriber_seq(id, 1), 3);
        assert_eq!(t.next_subscriber_seq(id, 2), 2);
    }

    #[test]
    fn channel_signature_persists_on_resolve() {
        // Same channel name on the same table → same signature (resolve_or_create
        // returns the existing channel).
        let mut t = SignalChannelTable::new();
        let id = t.resolve_or_create("foo", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        let sig1 = t.get_by_id(id).unwrap().signature;
        let id2 = t.resolve_or_create("foo", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        assert_eq!(id, id2);
        let sig2 = t.get_by_id(id).unwrap().signature;
        assert_eq!(sig1, sig2);
    }
}
