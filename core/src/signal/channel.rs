//! Signal channel table — named pub/sub channels with merge strategies,
//! scoping, and access control.

use std::collections::HashMap;

use bevy_ecs::prelude::*;

use super::types::*;

/// A named signal channel with scope, access control, and merge strategy.
#[derive(Clone, Debug)]
pub struct SignalChannel {
    /// Channel name.
    pub name: String,
    /// Current aggregated value (after merge).
    pub value: SignalValue,
    /// Previous tick's value (for change detection).
    prev_value: SignalValue,
    /// Accumulated values from publishers this tick (pre-merge).
    pending_values: Vec<SignalValue>,
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

/// All signal channels on a structure (ship, station, planet base).
/// One instance per shard that manages block data.
#[derive(Resource)]
pub struct SignalChannelTable {
    channels: HashMap<String, SignalChannel>,
}

impl SignalChannelTable {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Create or get a channel. If it doesn't exist, creates with defaults.
    pub fn get_or_create(
        &mut self,
        name: &str,
        scope: SignalScope,
        merge: ChannelMergeStrategy,
        owner_id: u64,
    ) -> &mut SignalChannel {
        self.channels.entry(name.to_string()).or_insert_with(|| SignalChannel {
            name: name.to_string(),
            value: SignalValue::default(),
            prev_value: SignalValue::default(),
            pending_values: Vec::new(),
            merge,
            dirty: false,
            scope,
            publish_policy: AccessPolicy::default(),
            subscribe_policy: AccessPolicy::default(),
            owner_id,
        })
    }

    /// Get a channel by name (read-only).
    pub fn get(&self, name: &str) -> Option<&SignalChannel> {
        self.channels.get(name)
    }

    /// Get a channel by name (mutable).
    pub fn get_mut(&mut self, name: &str) -> Option<&mut SignalChannel> {
        self.channels.get_mut(name)
    }

    /// Push a value to a channel's pending list (pre-merge).
    /// Creates the channel with defaults if it doesn't exist.
    pub fn push_pending(&mut self, name: &str, value: SignalValue) {
        let ch = self.channels.entry(name.to_string()).or_insert_with(|| SignalChannel {
            name: name.to_string(),
            value: SignalValue::default(),
            prev_value: SignalValue::default(),
            pending_values: Vec::new(),
            merge: ChannelMergeStrategy::default(),
            dirty: false,
            scope: SignalScope::default(),
            publish_policy: AccessPolicy::default(),
            subscribe_policy: AccessPolicy::default(),
            owner_id: 0,
        });
        ch.pending_values.push(value);
    }

    /// Clear all pending values (called at start of publish phase).
    pub fn clear_pending(&mut self) {
        for ch in self.channels.values_mut() {
            ch.pending_values.clear();
        }
    }

    /// Merge all pending values using each channel's merge strategy.
    /// Marks channels as dirty only if the merged value differs from the previous tick.
    pub fn merge_pending(&mut self) {
        for ch in self.channels.values_mut() {
            if ch.pending_values.is_empty() {
                continue;
            }
            ch.prev_value = ch.value;
            ch.value = ch.merge.merge(&ch.pending_values);
            ch.dirty = ch.value != ch.prev_value;
        }
    }

    /// Directly publish a value to a channel (bypasses merge — used by converters
    /// which write a single computed value, not accumulated).
    pub fn publish_direct(&mut self, name: &str, value: SignalValue) {
        if let Some(ch) = self.channels.get_mut(name) {
            ch.prev_value = ch.value;
            ch.value = value;
            ch.dirty = ch.value != ch.prev_value;
        } else {
            // Auto-create channel with the published value.
            self.channels.insert(name.to_string(), SignalChannel {
                name: name.to_string(),
                value,
                prev_value: SignalValue::default(),
                pending_values: Vec::new(),
                merge: ChannelMergeStrategy::default(),
                dirty: true,
                scope: SignalScope::default(),
                publish_policy: AccessPolicy::default(),
                subscribe_policy: AccessPolicy::default(),
                owner_id: 0,
            });
        }
    }

    /// Clear all dirty flags (called after subscribe phase).
    pub fn clear_dirty(&mut self) {
        for ch in self.channels.values_mut() {
            ch.dirty = false;
        }
    }

    /// Collect all dirty channels with non-Local scope (for network broadcast).
    pub fn drain_remote_dirty(&self) -> Vec<(String, SignalValue, SignalScope)> {
        self.channels.values()
            .filter(|ch| ch.dirty && !matches!(ch.scope, SignalScope::Local))
            .map(|ch| (ch.name.clone(), ch.value, ch.scope))
            .collect()
    }

    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Number of dirty channels.
    pub fn dirty_count(&self) -> usize {
        self.channels.values().filter(|ch| ch.dirty).count()
    }

    /// Remove a channel by name.
    pub fn remove(&mut self, name: &str) {
        self.channels.remove(name);
    }

    /// All channel names (for UI dropdowns).
    pub fn channel_names(&self) -> impl Iterator<Item = &str> {
        self.channels.keys().map(|s| s.as_str())
    }
}

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
    fn push_and_merge_sum() {
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
