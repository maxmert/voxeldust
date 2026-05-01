//! Capability-based remote access grants — the "no artificial scope
//! boundaries" mechanism from the plan.
//!
//! # The model in one paragraph
//!
//! `SignalScope` decides default routing reach (Local stays in-shard,
//! ShortRange floods nearby peers, LongRange floods the system, Radio
//! floods the galaxy). Access is governed orthogonally: a [`RemoteAccessGrant`]
//! is a keyed token authorizing a foreign shard to publish to (or
//! subscribe to) a specific set of channels. **A Local channel with a
//! grant is reachable from outside; a Local channel without grants is
//! not — no special-casing per scope.**
//!
//! # Why the registry is a Resource, not a field on `SignalChannel`
//!
//! Grants frequently span multiple channels (a "fleet leader" grant covers
//! N member ships' formation channels under one key). Keying by `grant_id`
//! gives O(1) lookup at the cross-shard ingress hot path; the channel-
//! reverse-index is just bookkeeping for revocation cascades.
//!
//! # Wire flow at a glance
//!
//! ```text
//! Owner (Alice's shard):
//!   GrantsRegistry::insert(Grant { id=G, key=K, channels=[door-42] })
//!   → owner UI: "Copy key" → Discord/voice/in-game mail
//!
//! Recipient (Bob's tablet → Bob's primary shard):
//!   ClientMsg::AddHeldGrant { id=G, key=K, target_shard=alice_planet }
//!   stored in Bob's shard's HeldGrants resource
//!
//! Bob clicks his bound HUD button:
//!   ClientMsg::RemoteSignalPublish { ch=door-42.open, grant_id=G, value=true }
//!   → Bob's shard signs HMAC(K, payload) → SignalBroadcastBatch via QUIC
//!     → alice_planet's drain_quic → IncomingSignalEntry { grant_id=G, auth_tag=… }
//!     → try_push_remote(grant_id=G) consults GrantsRegistry, verifies HMAC
//!       against grant.key, advances replay window, push_pending → door opens.
//! ```
//!
//! Phase 3B (this module) builds the registry + verification logic.
//! Phase 3C wires the network/UI plumbing on top.

use bevy_ecs::prelude::*;
use rand::{rngs::OsRng, RngCore};
use smallvec::SmallVec;
use std::collections::HashMap;

use super::channel::ChannelId;

/// What a grant authorizes. `Both` is the typical "remote pilot" case — the
/// grantee can publish commands AND subscribe to the resulting telemetry.
/// Subscribe-only is for monitoring (security camera, fleet status).
/// Publish-only is for one-way control (a station sends docking commands;
/// it doesn't need to read the ship's state from a separate channel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrantOps {
    Publish,
    Subscribe,
    Both,
}

impl GrantOps {
    #[inline]
    pub fn includes_publish(self) -> bool {
        matches!(self, Self::Publish | Self::Both)
    }

    #[inline]
    pub fn includes_subscribe(self) -> bool {
        matches!(self, Self::Subscribe | Self::Both)
    }
}

/// A keyed capability covering one or more channels on the issuing shard.
///
/// `key` is the HMAC secret. `channels` is the snapshot of `ChannelId`s
/// covered at issuance time; if `namespace_glob` is set, future channels
/// matching the pattern auto-extend the grant via
/// [`GrantsRegistry::extend_to_new_channel`] — used for the "approve
/// namespace" tablet UX where Alice grants Bob access to
/// `alice.north-pad.*` and her newly-placed `alice.north-pad.lights-east`
/// is auto-included without a fresh handshake.
///
/// `expires_at_ms` (optional UNIX millis) drives station-docking auto-
/// revocation and grant TTL. `revoked: true` is a tombstone — kept rather
/// than removed so the inverse `by_channel` index can lazily prune.
///
/// `mirror_of_held`: true for grants synthesized from `AddHeldGrant` on
/// the **recipient's** shard. The recipient needs the same `(grant_id,
/// key)` in their `GrantsRegistry` so `try_push_remote` can verify the
/// HMAC tag on inbound forwarded entries from the issuer's shard.
/// Mirror grants are filtered out of `snapshot_for_player` (they would
/// otherwise show up as "issued by me" in the recipient's tablet UI,
/// which is wrong — they were issued by someone else and held by us).
#[derive(Clone, Debug)]
pub struct RemoteAccessGrant {
    pub grant_id: u64,
    pub key: [u8; 32],
    pub channels: SmallVec<[ChannelId; 4]>,
    pub namespace_glob: Option<String>,
    pub ops: GrantOps,
    pub label: String,
    pub created_at_ms: u64,
    pub expires_at_ms: Option<u64>,
    pub created_by: u64,
    pub revoked: bool,
    pub mirror_of_held: bool,
}

impl RemoteAccessGrant {
    /// True iff this grant is non-revoked AND not past its expiry.
    pub fn is_active(&self, now_ms: u64) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(exp) = self.expires_at_ms {
            if now_ms > exp {
                return false;
            }
        }
        true
    }
}

/// Fresh `OsRng` 32-byte HMAC key. Used by tablet UX `[Generate Key]`
/// button and station docking offer auto-mint flows.
pub fn generate_grant_key() -> [u8; 32] {
    let mut k = [0u8; 32];
    OsRng.fill_bytes(&mut k);
    k
}

/// Encode a 32-byte grant key as base64 for tablet UX (Copy-to-clipboard
/// → Discord/voice → recipient pastes into their HUD). Output is exactly
/// 44 ASCII chars — fits in a single short text field. Never logged or
/// emitted server-side except in `GrantsSnapshotData` to the issuing
/// owner; non-owners get an empty string instead.
pub fn encode_grant_key(key: &[u8; 32]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(key)
}

/// Decode a base64 grant key string. Returns `None` on malformed input or
/// wrong length — callers must surface the rejection to the player as a
/// "this isn't a valid key" UI hint rather than crashing.
pub fn decode_grant_key(b64: &str) -> Option<[u8; 32]> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Some(out)
}

/// Fresh `OsRng` 64-bit grant id. Globally unique across the cluster
/// (collision probability ~negligible at game scale: 2⁶⁴ space, ~10⁶
/// grants in flight at peak ⇒ 10⁻¹²+ collision odds per insert).
pub fn generate_grant_id() -> u64 {
    OsRng.next_u64()
}

/// Reasons a `GrantsRegistry::create_for_owner` request was denied.
/// Surfaced to the requesting client so the tablet UI can render a
/// specific error toast ("you don't own all those channels", etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantCreateDenied {
    /// At least one of the requested channel names doesn't resolve to
    /// any channel on this shard. Anti-typo defense; client UX picks
    /// channel names from a server-provided list.
    UnknownChannel(String),
    /// At least one of the requested channels is owned by a different
    /// player. A grant must be issued by the channel's owner only.
    NotOwner { channel: String, owner_id: u64 },
    /// `ops` ordinal didn't decode to a `GrantOps` variant.
    InvalidOps(u8),
    /// Empty `channel_names`. Issuing a grant with no channels is a
    /// configuration mistake.
    NoChannels,
}

/// Per-shard registry of issued grants. Indexed for O(1) `grant_id` lookup
/// (the hot path on cross-shard ingress) plus a reverse `by_channel` index
/// for revoke-cascades and an explicit `glob_index` for namespace-glob
/// auto-extension.
#[derive(Resource, Default)]
pub struct GrantsRegistry {
    by_id: HashMap<u64, RemoteAccessGrant>,
    /// Channel id → list of grant_ids covering it. Maintained alongside
    /// `RemoteAccessGrant.channels` so revocation can cleanly walk
    /// "every grant that touched this channel".
    by_channel: HashMap<ChannelId, SmallVec<[u64; 4]>>,
    /// (namespace_pattern, grant_id) pairs. Consulted by
    /// `extend_to_new_channel` whenever a fresh channel is created so any
    /// grant whose glob matches auto-grows its `channels` snapshot. The
    /// linear scan is fine because typical shards have <100 active glob
    /// grants and channel creation is rare relative to publish traffic.
    glob_index: Vec<(String, u64)>,
}

impl GrantsRegistry {
    /// Insert a grant. Updates all three indexes atomically — callers can
    /// assume the registry is internally consistent after this returns.
    pub fn insert(&mut self, grant: RemoteAccessGrant) {
        let grant_id = grant.grant_id;
        for &ch in &grant.channels {
            self.by_channel.entry(ch).or_default().push(grant_id);
        }
        if let Some(ref pat) = grant.namespace_glob {
            self.glob_index.push((pat.clone(), grant_id));
        }
        self.by_id.insert(grant_id, grant);
    }

    /// Mark a grant as revoked. Returns `true` iff the grant existed and
    /// wasn't already revoked. Tombstones rather than removes — keeps the
    /// channel/glob indexes simple and gives an audit trail. Eventually
    /// a sweep can purge old revoked grants.
    pub fn revoke(&mut self, grant_id: u64) -> bool {
        match self.by_id.get_mut(&grant_id) {
            Some(g) if !g.revoked => {
                g.revoked = true;
                true
            }
            _ => false,
        }
    }

    /// Append a channel id to an existing grant's coverage list. Used by
    /// Listener config apply to register the listener's bridged Local-
    /// scope channel under a previously-mirrored held grant, so
    /// `try_push_remote` finds the channel in the grant's `channels`
    /// list when verifying inbound forwarded entries.
    ///
    /// Idempotent — re-adds are no-ops. Returns `true` if the grant
    /// existed and the channel was either added or already present.
    pub fn add_channel_to_grant(&mut self, grant_id: u64, channel_id: ChannelId) -> bool {
        let Some(g) = self.by_id.get_mut(&grant_id) else {
            return false;
        };
        if !g.channels.contains(&channel_id) {
            g.channels.push(channel_id);
            self.by_channel.entry(channel_id).or_default().push(grant_id);
        }
        true
    }

    /// Auto-extend any glob-matching grant to include the freshly-created
    /// channel. Runs on every channel creation in shards that issued
    /// glob grants. Linear in the number of glob grants — bounded.
    ///
    /// Pattern matching is the simplest "ends in .*" subset (`prefix.*`
    /// matches anything starting with `prefix.`). More complex patterns
    /// can land later without breaking the registry shape.
    pub fn extend_to_new_channel(&mut self, channel_id: ChannelId, channel_name: &str) {
        let mut to_extend = Vec::new();
        for (pat, gid) in &self.glob_index {
            if glob_matches(pat, channel_name) {
                to_extend.push(*gid);
            }
        }
        for gid in to_extend {
            if let Some(g) = self.by_id.get_mut(&gid) {
                if !g.channels.contains(&channel_id) {
                    g.channels.push(channel_id);
                    self.by_channel.entry(channel_id).or_default().push(gid);
                }
            }
        }
    }

    /// Hot-path check for cross-shard publish. Returns `Some(&key)` iff:
    ///   - the grant exists
    ///   - it's active (non-revoked, non-expired)
    ///   - it covers `channel_id`
    ///   - its `ops` includes `Publish`
    /// Otherwise `None` — caller surfaces as `RemoteIngressDenied::AuthFailed`.
    #[inline]
    pub fn check_publish(
        &self,
        grant_id: u64,
        channel_id: ChannelId,
        now_ms: u64,
    ) -> Option<&[u8; 32]> {
        let g = self.by_id.get(&grant_id)?;
        if !g.is_active(now_ms) {
            return None;
        }
        if !g.ops.includes_publish() {
            return None;
        }
        if !g.channels.contains(&channel_id) {
            return None;
        }
        Some(&g.key)
    }

    /// Symmetric for cross-shard subscribe (used by Phase 3C's
    /// `ShardMsg::SignalSubscribe` validator).
    #[inline]
    pub fn check_subscribe(
        &self,
        grant_id: u64,
        channel_id: ChannelId,
        now_ms: u64,
    ) -> Option<&[u8; 32]> {
        let g = self.by_id.get(&grant_id)?;
        if !g.is_active(now_ms) {
            return None;
        }
        if !g.ops.includes_subscribe() {
            return None;
        }
        if !g.channels.contains(&channel_id) {
            return None;
        }
        Some(&g.key)
    }

    pub fn get(&self, grant_id: u64) -> Option<&RemoteAccessGrant> {
        self.by_id.get(&grant_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &RemoteAccessGrant> {
        self.by_id.values()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Server-side helper for `ClientMsg::GrantCreate` ingestion. Resolves
    /// channel names → ids on the supplied table, validates that the
    /// requesting player owns every named channel, mints a fresh
    /// `(grant_id, key)`, inserts the grant, and returns the new grant for
    /// the caller to render into a `GrantsSnapshotData`.
    ///
    /// Mints `created_at_ms` from the supplied `now_ms` (caller passes
    /// `current_unix_millis()` so behaviour stays testable with fixed
    /// timestamps).
    pub fn create_for_owner(
        &mut self,
        table: &super::channel::SignalChannelTable,
        owner_id: u64,
        channel_names: &[String],
        ops_ordinal: u8,
        label: &str,
        expires_at_ms: u64,
        namespace_glob: Option<&str>,
        now_ms: u64,
    ) -> Result<&RemoteAccessGrant, GrantCreateDenied> {
        if channel_names.is_empty() {
            return Err(GrantCreateDenied::NoChannels);
        }
        let ops = match ops_ordinal {
            0 => GrantOps::Publish,
            1 => GrantOps::Subscribe,
            2 => GrantOps::Both,
            other => return Err(GrantCreateDenied::InvalidOps(other)),
        };

        // Resolve all channel names + check ownership UPFRONT — partial
        // grants ("you got rights to half the requested channels") are a
        // worse UX than a clean reject + retry.
        let mut resolved: SmallVec<[ChannelId; 4]> = SmallVec::new();
        for name in channel_names {
            let id = table
                .resolve(name)
                .ok_or_else(|| GrantCreateDenied::UnknownChannel(name.clone()))?;
            let ch = table
                .get_by_id(id)
                .ok_or_else(|| GrantCreateDenied::UnknownChannel(name.clone()))?;
            if ch.owner_id != owner_id {
                return Err(GrantCreateDenied::NotOwner {
                    channel: name.clone(),
                    owner_id: ch.owner_id,
                });
            }
            resolved.push(id);
        }

        let grant = RemoteAccessGrant {
            grant_id: generate_grant_id(),
            key: generate_grant_key(),
            channels: resolved,
            namespace_glob: namespace_glob
                .and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) }),
            ops,
            label: label.to_string(),
            created_at_ms: now_ms,
            expires_at_ms: if expires_at_ms == 0 { None } else { Some(expires_at_ms) },
            created_by: owner_id,
            revoked: false,
            mirror_of_held: false,
        };
        let id = grant.grant_id;
        self.insert(grant);
        Ok(self.by_id.get(&id).expect("just inserted"))
    }

    /// Render the registry as a `GrantsSnapshotData` filtered for the
    /// requesting player. The owner of a grant sees its key (base64-
    /// encoded); non-owners see the grant's existence + metadata but the
    /// `key_b64` field is empty.
    ///
    /// Lives here rather than in shard-common because the registry is in
    /// `core` and the `GrantPublicView` shape is in `core::client_message`
    /// — keeping the rendering function next to the source data avoids a
    /// dependency cycle.
    pub fn snapshot_for_player(
        &self,
        table: &super::channel::SignalChannelTable,
        viewer_player_id: u64,
    ) -> crate::client_message::GrantsSnapshotData {
        let grants = self
            .by_id
            .values()
            // Mirror grants exist only to enable HMAC verification of
            // inbound forwarded entries — they're not issued by anyone
            // on this shard, so they MUST NOT appear in the snapshot
            // (otherwise the recipient sees them as "issued by me",
            // which is wrong + leaks the implementation detail).
            .filter(|g| !g.mirror_of_held)
            .map(|g| {
                let key_b64 = if g.created_by == viewer_player_id {
                    encode_grant_key(&g.key)
                } else {
                    String::new()
                };
                let channel_names = g
                    .channels
                    .iter()
                    .filter_map(|cid| table.name_for_id(*cid).map(|s| s.to_string()))
                    .collect();
                crate::client_message::GrantPublicView {
                    grant_id: g.grant_id,
                    key_b64,
                    channel_names,
                    ops: match g.ops {
                        GrantOps::Publish => 0,
                        GrantOps::Subscribe => 1,
                        GrantOps::Both => 2,
                    },
                    label: g.label.clone(),
                    created_at_ms: g.created_at_ms,
                    expires_at_ms: g.expires_at_ms.unwrap_or(0),
                    created_by: g.created_by,
                    revoked: g.revoked,
                    namespace_glob: g.namespace_glob.clone().unwrap_or_default(),
                }
            })
            .collect();
        crate::client_message::GrantsSnapshotData { grants }
    }
}

/// Minimal glob match: `prefix.*` matches anything starting with `prefix.`.
/// Exact-match patterns also work (no `*`). Anything else returns false —
/// future expansion (multi-segment globs, `?`, `[abc]`) is room-to-grow.
fn glob_matches(pattern: &str, name: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix(".*") {
        // `alice.north-pad.*` matches `alice.north-pad.lights` and
        // `alice.north-pad.lights.east` — anything starting with the
        // prefix followed by `.`.
        let with_dot = format!("{}.", prefix);
        return name.starts_with(&with_dot);
    }
    pattern == name
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_grant(grant_id: u64, channels: Vec<u16>, ops: GrantOps) -> RemoteAccessGrant {
        RemoteAccessGrant {
            grant_id,
            key: [0xAA; 32],
            channels: channels.into_iter().map(ChannelId).collect(),
            namespace_glob: None,
            ops,
            label: "test".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        }
    }

    #[test]
    fn check_publish_accepts_matching_grant() {
        let mut reg = GrantsRegistry::default();
        reg.insert(fixture_grant(42, vec![5], GrantOps::Both));
        assert!(reg.check_publish(42, ChannelId(5), 0).is_some());
    }

    #[test]
    fn check_publish_rejects_unknown_grant_id() {
        let reg = GrantsRegistry::default();
        assert!(reg.check_publish(42, ChannelId(5), 0).is_none());
    }

    #[test]
    fn check_publish_rejects_channel_not_in_grant() {
        let mut reg = GrantsRegistry::default();
        reg.insert(fixture_grant(42, vec![5], GrantOps::Both));
        // Grant covers channel 5, but the publish targets channel 6.
        assert!(reg.check_publish(42, ChannelId(6), 0).is_none());
    }

    #[test]
    fn check_publish_rejects_subscribe_only_grant() {
        let mut reg = GrantsRegistry::default();
        reg.insert(fixture_grant(42, vec![5], GrantOps::Subscribe));
        assert!(reg.check_publish(42, ChannelId(5), 0).is_none());
        // But subscribe still works.
        assert!(reg.check_subscribe(42, ChannelId(5), 0).is_some());
    }

    #[test]
    fn check_publish_rejects_revoked_grant() {
        let mut reg = GrantsRegistry::default();
        reg.insert(fixture_grant(42, vec![5], GrantOps::Both));
        assert!(reg.check_publish(42, ChannelId(5), 0).is_some());
        assert!(reg.revoke(42));
        assert!(reg.check_publish(42, ChannelId(5), 0).is_none());
        // Revoking again is a no-op (returns false).
        assert!(!reg.revoke(42));
    }

    #[test]
    fn check_publish_rejects_expired_grant() {
        let mut g = fixture_grant(42, vec![5], GrantOps::Both);
        g.expires_at_ms = Some(1000);
        let mut reg = GrantsRegistry::default();
        reg.insert(g);
        // Before expiry: ok.
        assert!(reg.check_publish(42, ChannelId(5), 999).is_some());
        // At expiry: ok (inclusive boundary — we use `now > exp` so equal is fine).
        assert!(reg.check_publish(42, ChannelId(5), 1000).is_some());
        // After expiry: rejected.
        assert!(reg.check_publish(42, ChannelId(5), 1001).is_none());
    }

    #[test]
    fn extend_to_new_channel_grows_glob_grant() {
        // Alice has issued a "namespace glob" grant covering
        // alice.north-pad.* — when she places a fresh block whose channels
        // match the pattern, those channel ids must auto-attach to the
        // grant so Bob (the grantee) can reach them without a re-handshake.
        let mut reg = GrantsRegistry::default();
        let mut g = fixture_grant(42, vec![5], GrantOps::Both);
        g.namespace_glob = Some("alice.north-pad.*".into());
        reg.insert(g);

        // New channel 99 named alice.north-pad.lights — should auto-attach.
        reg.extend_to_new_channel(ChannelId(99), "alice.north-pad.lights");
        assert!(reg.check_publish(42, ChannelId(99), 0).is_some());

        // Channel 100 outside the namespace — must NOT attach.
        reg.extend_to_new_channel(ChannelId(100), "alice.south-pad.lights");
        assert!(reg.check_publish(42, ChannelId(100), 0).is_none());
    }

    #[test]
    fn extend_is_idempotent() {
        // Re-running auto-extension with the same name doesn't duplicate
        // the channel id in the grant's channel list (defends against
        // resolve_or_create being called twice for the same name).
        let mut reg = GrantsRegistry::default();
        let mut g = fixture_grant(42, vec![], GrantOps::Both);
        g.namespace_glob = Some("alice.*".into());
        reg.insert(g);

        reg.extend_to_new_channel(ChannelId(7), "alice.beacon");
        reg.extend_to_new_channel(ChannelId(7), "alice.beacon");
        let g = reg.get(42).unwrap();
        assert_eq!(g.channels.len(), 1);
        assert_eq!(g.channels[0], ChannelId(7));
    }

    #[test]
    fn glob_matches_basic() {
        assert!(glob_matches("alice.north-pad.*", "alice.north-pad.lights"));
        assert!(glob_matches("alice.north-pad.*", "alice.north-pad.lights.east"));
        assert!(!glob_matches("alice.north-pad.*", "alice.north-padding"));
        assert!(!glob_matches("alice.north-pad.*", "alice.south-pad.lights"));
        // Exact-match (no glob).
        assert!(glob_matches("alice.beacon", "alice.beacon"));
        assert!(!glob_matches("alice.beacon", "alice.beacon.extra"));
    }

    #[test]
    fn generate_grant_id_is_pseudo_unique() {
        // Trivial smoke test — true uniqueness is a probabilistic property
        // of OsRng. Asserts at least that two consecutive calls don't
        // alias to zero / each other.
        let a = generate_grant_id();
        let b = generate_grant_id();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn generate_grant_key_is_non_zero() {
        let k = generate_grant_key();
        assert_ne!(k, [0u8; 32], "OsRng must not produce all-zero key");
    }

    #[test]
    fn encode_decode_grant_key_roundtrip() {
        let mut k = [0u8; 32];
        for i in 0..32 {
            k[i] = (i as u8).wrapping_mul(11).wrapping_add(7);
        }
        let s = encode_grant_key(&k);
        // Standard base64 of 32 bytes is exactly 44 chars (32 * 4/3 rounded
        // up + padding). Tablet UI assumes 44 in fixed-width fonts.
        assert_eq!(s.len(), 44);
        let decoded = decode_grant_key(&s).expect("roundtrip must decode");
        assert_eq!(decoded, k);
    }

    // -- create_for_owner / snapshot_for_player ----------------------------

    #[test]
    fn create_for_owner_validates_channel_ownership() {
        // Alice (owner_id=1) owns channel "alice.thrust". Bob (owner_id=2)
        // owns "bob.beacon". Bob should NOT be able to issue a grant on
        // alice.thrust.
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope};
        let mut table = SignalChannelTable::new();
        table.get_or_create("alice.thrust", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        table.get_or_create("bob.beacon", SignalScope::Local, ChannelMergeStrategy::LastWrite, 2);

        let mut reg = GrantsRegistry::default();

        // Alice issues a grant on her own channel — succeeds.
        let g = reg
            .create_for_owner(
                &table,
                /*owner_id=*/ 1,
                &["alice.thrust".into()],
                /*ops=*/ 2,
                "Bob (pilot)",
                /*expires_at_ms=*/ 0,
                None,
                /*now_ms=*/ 1000,
            )
            .expect("Alice issuing on her own channel must succeed");
        let g_id = g.grant_id;
        assert_ne!(g_id, 0);
        assert_eq!(g.ops, GrantOps::Both);
        assert_eq!(g.created_by, 1);

        // Bob tries to issue a grant on Alice's channel — must reject.
        let err = reg
            .create_for_owner(
                &table,
                /*owner_id=*/ 2,
                &["alice.thrust".into()],
                2,
                "Bob trying",
                0,
                None,
                1000,
            )
            .expect_err("non-owner must be denied");
        match err {
            GrantCreateDenied::NotOwner { channel, owner_id } => {
                assert_eq!(channel, "alice.thrust");
                assert_eq!(owner_id, 1);
            }
            other => panic!("wrong denial: {:?}", other),
        }
    }

    #[test]
    fn create_for_owner_rejects_unknown_channel() {
        use crate::signal::channel::SignalChannelTable;
        let table = SignalChannelTable::new();
        let mut reg = GrantsRegistry::default();
        let err = reg
            .create_for_owner(
                &table, 1, &["does.not.exist".into()], 2, "x", 0, None, 0,
            )
            .expect_err("unknown channel must reject");
        assert!(matches!(err, GrantCreateDenied::UnknownChannel(_)));
    }

    #[test]
    fn create_for_owner_rejects_empty_channel_list() {
        use crate::signal::channel::SignalChannelTable;
        let table = SignalChannelTable::new();
        let mut reg = GrantsRegistry::default();
        let err = reg
            .create_for_owner(&table, 1, &[], 2, "x", 0, None, 0)
            .expect_err("empty channel list must reject");
        assert_eq!(err, GrantCreateDenied::NoChannels);
    }

    #[test]
    fn create_for_owner_partial_ownership_rejects_atomically() {
        // Alice owns chA, Bob owns chB. Alice asks for a grant covering
        // BOTH. Must reject — and crucially, must NOT have inserted a
        // partial grant covering only chA. Atomic rollback.
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope};
        let mut table = SignalChannelTable::new();
        table.get_or_create("alice.a", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        table.get_or_create("bob.b", SignalScope::Local, ChannelMergeStrategy::LastWrite, 2);
        let mut reg = GrantsRegistry::default();

        let err = reg
            .create_for_owner(
                &table,
                1,
                &["alice.a".into(), "bob.b".into()],
                2,
                "mixed",
                0,
                None,
                0,
            )
            .expect_err("mixed-ownership grant must reject atomically");
        assert!(matches!(err, GrantCreateDenied::NotOwner { .. }));
        assert!(reg.is_empty(), "registry must remain empty after rejection");
    }

    #[test]
    fn snapshot_for_player_filters_keys_by_ownership() {
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope};
        let mut table = SignalChannelTable::new();
        table.get_or_create("alice.thrust", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        let mut reg = GrantsRegistry::default();
        let g = reg
            .create_for_owner(&table, 1, &["alice.thrust".into()], 2, "lbl", 0, None, 100)
            .unwrap();
        let g_id = g.grant_id;

        // Owner sees the key.
        let snap_owner = reg.snapshot_for_player(&table, /*viewer=*/ 1);
        assert_eq!(snap_owner.grants.len(), 1);
        assert_eq!(snap_owner.grants[0].grant_id, g_id);
        assert_eq!(snap_owner.grants[0].key_b64.len(), 44);
        assert_eq!(snap_owner.grants[0].channel_names, vec!["alice.thrust".to_string()]);

        // Non-owner sees the grant exists but the key is hidden.
        let snap_other = reg.snapshot_for_player(&table, /*viewer=*/ 999);
        assert_eq!(snap_other.grants.len(), 1);
        assert!(snap_other.grants[0].key_b64.is_empty());
    }

    #[test]
    fn end_to_end_subscribe_forward_flow() {
        // Phase 3D end-to-end: Bob subscribes to Alice's `alice.health`
        // via a held grant. Alice's `signal_broadcast_remote` (simulated
        // here without ECS) forwards a dirty value, HMAC-stamped under
        // the grant key. Bob's `try_push_remote` verifies and accepts.
        use crate::signal::auth as signal_auth;
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope, SignalValue};
        use crate::shard_types::ShardId;
        use smallvec::smallvec;

        let mut alice_table = SignalChannelTable::new();
        alice_table.get_or_create(
            "alice.health",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            /*owner=*/ 1,
        );
        let alice_channel_id = alice_table.resolve("alice.health").unwrap();

        // Alice issues a Subscribe-only grant for Bob (shard_id=99).
        let mut alice_grants = GrantsRegistry::default();
        let grant = RemoteAccessGrant {
            grant_id: 0xCAFE,
            key: [0xAB; 32],
            channels: smallvec![alice_channel_id],
            namespace_glob: None,
            ops: GrantOps::Subscribe,
            label: "Bob (monitor)".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        };
        let grant_key = grant.key;
        alice_grants.insert(grant);

        // Bob's shard subscribes via SignalSubscribe — Alice's
        // apply_signal_subscribe (logic recreated here without ECS):
        let now_ms = crate::signal::channel::current_unix_millis();
        let bob_shard = ShardId(99);
        let nonce = 42;
        let lease_until = 10_000;
        let sub_tag = signal_auth::hmac_sign_subscribe_request(
            &grant_key,
            "alice.health",
            bob_shard.0,
            0xCAFE,
            nonce,
            now_ms,
            lease_until,
        );
        // Validate the request (mirrors apply_signal_subscribe's checks):
        let key_for_verify = alice_grants
            .check_subscribe(0xCAFE, alice_channel_id, now_ms)
            .expect("grant must allow subscribe");
        assert!(signal_auth::hmac_verify_subscribe_request(
            key_for_verify,
            "alice.health",
            bob_shard.0,
            0xCAFE,
            nonce,
            now_ms,
            lease_until,
            &sub_tag,
        ));
        alice_table.add_remote_subscriber(alice_channel_id, bob_shard, 0xCAFE, lease_until);

        // Alice's channel becomes dirty.
        alice_table.clear_pending();
        alice_table.push_pending("alice.health", SignalValue::Float(0.42));
        alice_table.merge_pending();

        // Forward to Bob (mirrors signal_broadcast_remote's per-subscriber
        // path): bump per-grant seq, sign HMAC, ship.
        let alice_shard_id = 1u64;
        let value = SignalValue::Float(0.42);
        let value_bits = match value {
            SignalValue::Float(f) => f.to_bits(),
            _ => unreachable!(),
        };
        let seq = alice_table.next_subscriber_seq(alice_channel_id, 0xCAFE);
        let pub_tag = signal_auth::hmac_sign(
            &grant_key,
            "alice.health",
            /*scope=Local*/ 0,
            0,
            /*float*/ 1,
            value_bits,
            now_ms,
            alice_shard_id,
            seq,
            0xCAFE,
        );

        // Bob's shard receives + verifies. Bob has the SAME grant in
        // its OWN registry (after AddHeldGrant was processed and a
        // hypothetical sync from Alice). For this test we simulate
        // Bob's registry mirroring Alice's grant.
        let mut bob_table = SignalChannelTable::new();
        bob_table.get_or_create(
            "alice.health",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            1,
        );
        let bob_channel_id = bob_table.resolve("alice.health").unwrap();
        let mut bob_grants = GrantsRegistry::default();
        bob_grants.insert(RemoteAccessGrant {
            grant_id: 0xCAFE,
            key: grant_key,
            channels: smallvec![bob_channel_id],
            namespace_glob: None,
            ops: GrantOps::Subscribe,
            label: "from Alice".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        });

        // Hmm — try_push_remote uses check_publish, not check_subscribe.
        // The grant on Bob's side needs Publish ops for the forwarded
        // entry to verify (since push_remote uses check_publish).
        // In production the grant on Alice's side is Subscribe-only,
        // and the receiver-side validation uses a different grant for
        // forwarded entries. For the test, give Bob's grant Both ops
        // to mirror what production would do (a synthetic "trust the
        // forward" grant). This is a documentation-anchor test, not
        // production logic — the key insight is that the wire-format
        // round-trip + HMAC verify works end-to-end.
        bob_grants.revoke(0xCAFE);
        bob_grants.insert(RemoteAccessGrant {
            grant_id: 0xCAFE,
            key: grant_key,
            channels: smallvec![bob_channel_id],
            namespace_glob: None,
            ops: GrantOps::Both,
            label: "from Alice".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        });

        bob_table
            .try_push_remote(
                "alice.health",
                value,
                /*scope=*/ 0,
                alice_shard_id,
                now_ms,
                seq,
                0xCAFE,
                Some(&pub_tag),
                &bob_grants,
            )
            .expect("subscriber forward must verify on Bob's side");
        bob_table.merge_pending();
        assert_eq!(bob_table.get("alice.health").unwrap().value, value);
    }

    #[test]
    fn mirror_grants_are_filtered_from_snapshot() {
        // Phase 3E.4: a recipient's GrantsRegistry holds verification-
        // only mirror grants that MUST NOT appear in their tablet's
        // "Issued by me" panel.
        use crate::signal::channel::SignalChannelTable;
        let table = SignalChannelTable::new();
        let mut reg = GrantsRegistry::default();

        // Mirror grant for inbound verification — created on AddHeldGrant.
        reg.insert(RemoteAccessGrant {
            grant_id: 0xC0FFEE,
            key: [0xFF; 32],
            channels: smallvec::smallvec![],
            namespace_glob: None,
            ops: GrantOps::Both,
            label: "from Alice".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            // viewer is the bob session — would normally see the key
            created_by: 99,
            revoked: false,
            mirror_of_held: true,
        });

        // Real grant — issued by viewer.
        reg.insert(fixture_grant(7, vec![], GrantOps::Both));

        let snap = reg.snapshot_for_player(&table, /*viewer=*/ 99);
        // Only the non-mirror grant appears.
        assert_eq!(snap.grants.len(), 1, "mirror grant must be filtered out");
        assert_eq!(snap.grants[0].grant_id, 7);

        // But check_publish still finds the mirror grant — it's there
        // for verification purposes.
        // (Add a channel first so the channel-cover check passes.)
        reg.add_channel_to_grant(0xC0FFEE, ChannelId(42));
        assert!(reg.check_publish(0xC0FFEE, ChannelId(42), 0).is_some());
    }

    #[test]
    fn add_channel_to_grant_is_idempotent() {
        let mut reg = GrantsRegistry::default();
        reg.insert(fixture_grant(1, vec![], GrantOps::Both));
        // Cover a channel.
        assert!(reg.add_channel_to_grant(1, ChannelId(5)));
        // Re-add — no duplicate.
        assert!(reg.add_channel_to_grant(1, ChannelId(5)));
        let g = reg.get(1).unwrap();
        assert_eq!(g.channels.len(), 1);
        assert_eq!(g.channels[0], ChannelId(5));

        // Add a different channel — appends.
        assert!(reg.add_channel_to_grant(1, ChannelId(6)));
        let g = reg.get(1).unwrap();
        assert_eq!(g.channels.len(), 2);
    }

    #[test]
    fn add_channel_to_unknown_grant_returns_false() {
        let mut reg = GrantsRegistry::default();
        assert!(!reg.add_channel_to_grant(/*nonexistent=*/ 999, ChannelId(0)));
    }

    #[test]
    fn end_to_end_antenna_publish_flow() {
        // Phase 3E.2 end-to-end: an Antenna block on Bob's shard reads
        // its source channel value, signs HMAC under a held grant key,
        // ships a SignalBroadcastBatch to Alice's shard. Alice's
        // try_push_remote validates against HER GrantsRegistry (where
        // the grant lives) and accepts. The mirror of the published
        // value lands in Alice's local channel.
        //
        // The test exercises the cryptographic + scope + replay flow
        // without ECS — the antenna_publish system is just the per-tick
        // wrapper around exactly this canonical sequence.
        use crate::signal::auth as signal_auth;
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope, SignalValue};
        use smallvec::smallvec;

        // Alice's side: she owns `alice.alarm` (Local) and issued a
        // Publish grant for Bob covering it.
        let mut alice_table = SignalChannelTable::new();
        alice_table.get_or_create(
            "alice.alarm",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            /*alice's player_id=*/ 1,
        );
        let alice_channel_id = alice_table.resolve("alice.alarm").unwrap();

        let grant_key = [0x42; 32];
        let mut alice_grants = GrantsRegistry::default();
        alice_grants.insert(RemoteAccessGrant {
            grant_id: 0xCAFE,
            key: grant_key,
            channels: smallvec![alice_channel_id],
            namespace_glob: None,
            ops: GrantOps::Publish,
            label: "Bob's antenna".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        });

        // Bob's antenna: simulates one tick of antenna_publish for the
        // case "source channel value is `Bool(true)`, target = Alice".
        let bob_shard_id = 99_u64;
        let now_ms = crate::signal::channel::current_unix_millis();
        let value = SignalValue::Bool(true);
        let value_bits = 1.0_f32.to_bits();
        let tag = signal_auth::hmac_sign(
            &grant_key,
            "alice.alarm",
            /*scope=Local*/ 0,
            /*frequency=*/ 0,
            /*value_type=Bool*/ 0,
            value_bits,
            now_ms,
            bob_shard_id,
            /*seq=*/ 1,
            0xCAFE,
        );

        // Alice's shard receives the entry and verifies via her registry.
        alice_table
            .try_push_remote(
                "alice.alarm",
                value,
                /*wire_scope=*/ 0,
                bob_shard_id,
                now_ms,
                /*seq=*/ 1,
                0xCAFE,
                Some(&tag),
                &alice_grants,
            )
            .expect("antenna-stamped entry must verify on the issuer's side");
        alice_table.merge_pending();
        assert_eq!(
            alice_table.get("alice.alarm").unwrap().value,
            SignalValue::Bool(true),
            "the antenna's bool value must materialize on Alice's channel"
        );
    }

    #[test]
    fn end_to_end_listener_via_mirror_grant_flow() {
        // Phase 3E.4 end-to-end. Walks the FULL listener path:
        //
        //   1. Alice issues Subscribe grant on `alice.health` (Local).
        //   2. Bob receives `(grant_id, key)` via AddHeldGrant. His shard
        //      mirrors the grant into HIS GrantsRegistry with
        //      `mirror_of_held=true` so future inbound forwarded entries
        //      can verify against the same key.
        //   3. Bob places a Listener block. Apply step:
        //      - Creates a local Radio-scope channel matching Alice's
        //        `alice.health` so try_push_remote finds it.
        //      - Creates a local Local-scope destination channel.
        //      - Adds the bridged channel id to Bob's mirror grant via
        //        `add_channel_to_grant` so check_publish accepts.
        //   4. Alice's signal_broadcast_remote forwards the value (Bool,
        //      Float, whatever) to Bob's shard, HMAC-stamped under the
        //      grant key.
        //   5. Bob's try_push_remote validates against the mirror grant,
        //      push_pendings the value into the bridged channel.
        //   6. Bob's listener_mirror reads bridged.value, publish_direct
        //      to destination — internal subscribers see a fresh Local
        //      value with the listener as the apparent publisher.
        //
        // The test exercises the cryptographic + scope + replay flow
        // without ECS scheduling — all the building blocks live in
        // core, and this test verifies the contract between them.
        use crate::signal::auth as signal_auth;
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope, SignalValue};
        use smallvec::smallvec;

        // === Alice's side ===
        let mut alice_table = SignalChannelTable::new();
        alice_table.get_or_create(
            "alice.health",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            /*alice's player_id=*/ 1,
        );
        let alice_channel_id = alice_table.resolve("alice.health").unwrap();
        let grant_key = [0xC1; 32];
        let mut alice_grants = GrantsRegistry::default();
        alice_grants.insert(RemoteAccessGrant {
            grant_id: 0xBEEF,
            key: grant_key,
            channels: smallvec![alice_channel_id],
            namespace_glob: None,
            ops: GrantOps::Subscribe,
            label: "Bob (listener)".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 1,
            revoked: false,
            mirror_of_held: false,
        });

        // === Bob's side ===
        // Bob's shard processes AddHeldGrant — it inserts a mirror
        // grant in his GrantsRegistry (simulated here directly).
        let mut bob_table = SignalChannelTable::new();
        let mut bob_grants = GrantsRegistry::default();
        bob_grants.insert(RemoteAccessGrant {
            grant_id: 0xBEEF,
            key: grant_key,
            channels: smallvec![],
            namespace_glob: None,
            ops: GrantOps::Both,
            label: "from Alice".into(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 99, // bob's session
            revoked: false,
            mirror_of_held: true,
        });

        // Listener apply step: create bridged + destination channels,
        // populate the mirror grant's channels list.
        bob_table.get_or_create(
            "alice.health",
            SignalScope::Radio { frequency: 0 },
            ChannelMergeStrategy::LastWrite,
            99,
        );
        let bridged_id = bob_table.resolve("alice.health").unwrap();
        bob_table.get_or_create(
            "bob.local.alice-health-mirror",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            99,
        );
        let destination_id = bob_table.resolve("bob.local.alice-health-mirror").unwrap();
        bob_grants.add_channel_to_grant(0xBEEF, bridged_id);

        // Snapshot for Bob — the mirror grant must NOT appear (filter test).
        let snap_bob = bob_grants.snapshot_for_player(&bob_table, /*viewer=*/ 99);
        assert!(
            snap_bob.grants.is_empty(),
            "mirror grant must not appear in recipient's snapshot"
        );

        // === Alice broadcasts a value ===
        let alice_shard = 1u64;
        let now_ms = crate::signal::channel::current_unix_millis();
        let value = SignalValue::Float(0.65);
        let value_bits = 0.65_f32.to_bits();
        let tag = signal_auth::hmac_sign(
            &grant_key,
            "alice.health",
            /*Radio scope=*/ 3,
            /*frequency=*/ 0,
            /*float=*/ 1,
            value_bits,
            now_ms,
            alice_shard,
            /*seq=*/ 1,
            0xBEEF,
        );

        // === Bob's try_push_remote validates against the mirror grant ===
        bob_table
            .try_push_remote(
                "alice.health",
                value,
                /*wire_scope=Radio*/ 3,
                alice_shard,
                now_ms,
                /*seq=*/ 1,
                0xBEEF,
                Some(&tag),
                &bob_grants,
            )
            .expect("inbound grant-stamped entry must verify via mirror grant");
        bob_table.merge_pending();

        // === Bob's listener_mirror copies bridged → destination ===
        // (Manual — without the ECS system here.)
        let bridged_value = bob_table.get_by_id(bridged_id).unwrap().value;
        bob_table.publish_direct_id(destination_id, bridged_value);

        // === Internal subscribers on Bob's side see the value ===
        assert_eq!(
            bob_table.get_by_id(destination_id).unwrap().value,
            value,
            "destination channel must mirror the bridged value"
        );

        // === Forgetting the grant revokes inbound verification ===
        bob_grants.revoke(0xBEEF);
        let tag2 = signal_auth::hmac_sign(
            &grant_key,
            "alice.health", 3, 0, 1, value_bits, now_ms, alice_shard, /*seq=*/ 2, 0xBEEF,
        );
        let err = bob_table
            .try_push_remote(
                "alice.health", value, 3, alice_shard, now_ms, 2, 0xBEEF, Some(&tag2),
                &bob_grants,
            )
            .expect_err("revoked mirror grant must reject inbound");
        assert_eq!(err, crate::signal::channel::RemoteIngressDenied::AuthFailed);
    }

    #[test]
    fn end_to_end_grant_publish_flow() {
        // Plan's headline use case wired end-to-end:
        //   1. Alice's shard owns `alice.thrust-forward` (Local).
        //   2. Alice issues a grant for Bob covering that channel.
        //   3. snapshot_for_player(alice) shows the key; non-owners see empty.
        //   4. Bob's tablet would receive the snapshot, decode the key,
        //      compute HMAC over a publish payload, and ship it cross-shard.
        //      We simulate that here without the network.
        //   5. try_push_remote on Alice's shard verifies via the grant key
        //      (NOT the channel signature) and accepts.
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope, SignalValue};
        let mut table = SignalChannelTable::new();
        table.get_or_create("alice.thrust-forward", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);

        let mut reg = GrantsRegistry::default();
        let grant = reg
            .create_for_owner(
                &table,
                /*owner=*/ 1,
                &["alice.thrust-forward".into()],
                /*ops=Both*/ 2,
                "Bob",
                /*expires=*/ 0,
                None,
                /*now=*/ 1000,
            )
            .unwrap();
        let grant_id = grant.grant_id;
        let grant_key = grant.key;

        // Render snapshot for Alice → Bob receives a base64 of the key.
        let snap = reg.snapshot_for_player(&table, /*viewer=*/ 1);
        assert_eq!(snap.grants[0].grant_id, grant_id);
        let recovered_key = decode_grant_key(&snap.grants[0].key_b64).unwrap();
        assert_eq!(recovered_key, grant_key);

        // Bob's tablet builds a publish payload + HMAC tag.
        let now_ms = crate::signal::channel::current_unix_millis();
        let value = SignalValue::Float(1.0);
        let tag = crate::signal::auth::hmac_sign(
            &recovered_key,
            "alice.thrust-forward",
            /*scope=Local*/ 0,
            /*frequency=*/ 0,
            /*value_type=*/ 1,
            1.0_f32.to_bits(),
            now_ms,
            /*sender=Bob's shard*/ 99,
            /*seq=*/ 1,
            grant_id,
        );

        // Alice's shard receives the entry and verifies via the registry.
        table.try_push_remote(
            "alice.thrust-forward",
            value,
            /*wire_scope=*/ 0, // Local
            /*sender=*/ 99,
            now_ms,
            /*seq=*/ 1,
            grant_id,
            Some(&tag),
            &reg,
        )
        .expect("end-to-end grant publish must accept");

        // Confirm value materialized.
        table.merge_pending();
        assert_eq!(table.get("alice.thrust-forward").unwrap().value, value);

        // Now Alice revokes — same payload must reject.
        assert!(reg.revoke(grant_id));
        // Use a fresh seq since the old one is in the replay window.
        let tag2 = crate::signal::auth::hmac_sign(
            &recovered_key,
            "alice.thrust-forward",
            0, 0, 1, 1.0_f32.to_bits(), now_ms, 99, 2, grant_id,
        );
        let err = table
            .try_push_remote(
                "alice.thrust-forward",
                value,
                0,
                99,
                now_ms,
                2,
                grant_id,
                Some(&tag2),
                &reg,
            )
            .expect_err("revoked grant must reject");
        assert_eq!(err, crate::signal::channel::RemoteIngressDenied::AuthFailed);
    }

    #[test]
    fn create_for_owner_rejects_invalid_ops_ordinal() {
        use crate::signal::channel::SignalChannelTable;
        use crate::signal::types::{ChannelMergeStrategy, SignalScope};
        let mut table = SignalChannelTable::new();
        table.get_or_create("ch", SignalScope::Local, ChannelMergeStrategy::LastWrite, 1);
        let mut reg = GrantsRegistry::default();
        let err = reg
            .create_for_owner(&table, 1, &["ch".into()], 99, "x", 0, None, 0)
            .expect_err("invalid ops ordinal must reject");
        assert_eq!(err, GrantCreateDenied::InvalidOps(99));
    }

    #[test]
    fn decode_grant_key_rejects_malformed_input() {
        // Empty.
        assert!(decode_grant_key("").is_none());
        // Wrong length (decodes to fewer than 32 bytes).
        assert!(decode_grant_key("abc").is_none());
        // Invalid base64 chars.
        assert!(decode_grant_key("!!! not base64 !!!").is_none());
        // Right base64 but wrong byte length (16 bytes encoded).
        let too_short = encode_grant_key(&[0u8; 32])
            .chars()
            .take(22)
            .collect::<String>();
        assert!(decode_grant_key(&too_short).is_none());
    }
}
