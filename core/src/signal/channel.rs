//! Signal channel table — named pub/sub channels with merge strategies,
//! scoping, and access control.  Channels are indexed by `ChannelId` (u16)
//! for O(1) hot-path access; a string→id registry handles config-time resolution.

use std::collections::HashMap;

use bevy_ecs::prelude::*;

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
}

// ---------------------------------------------------------------------------
// SignalChannelTable
// ---------------------------------------------------------------------------

/// All signal channels on a structure (ship, station, planet base).
/// One instance per shard that manages block data.
///
/// Channels are stored in a `Vec` indexed by `ChannelId` for O(1) access.
/// A `HashMap<String, ChannelId>` handles name→id resolution at config time.
#[derive(Resource)]
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
    pub fn drain_remote_dirty(&self) -> Vec<(String, SignalValue, SignalScope)> {
        self.dirty_set.iter()
            .filter_map(|id| self.channels.get(id.0 as usize)?.as_ref())
            .filter(|ch| !matches!(ch.scope, SignalScope::Local))
            .map(|ch| (ch.name.clone(), ch.value, ch.scope))
            .collect()
    }

    // -- Introspection ------------------------------------------------------

    /// Number of live channels.
    pub fn channel_count(&self) -> usize {
        self.name_to_id.len()
    }

    /// Number of dirty channels.
    pub fn dirty_count(&self) -> usize {
        self.dirty_set.len()
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
        assert_eq!(remote[0].0, "beacon");
    }
}
