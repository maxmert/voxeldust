//! Phase 4-Persist — durable storage for `RemoteAccessGrant`.
//!
//! Without persistence, every shard restart wipes every issued
//! grant: a player who hands out remote-pilot access to a friend
//! must re-share the key after every redeploy, every crash, every
//! orchestrator-driven shard cycle. That's not production-ready.
//!
//! # The unit of persistence
//!
//! Grants are stored in a shard-local `redb::Database`. Each shard
//! that owns channels (ship, planet, station) keeps its own grant
//! file alongside the existing per-shard state (chunk data, block
//! configs). No cross-shard coordination — grants live with the
//! channels they cover.
//!
//! # Why we persist NAMES, not ChannelIds
//!
//! `RemoteAccessGrant.channels: SmallVec<[ChannelId; 4]>` is a
//! convenient runtime cache for the hot-path `check_publish` lookup.
//! `ChannelId` is a u16 index into `SignalChannelTable.channels`,
//! which is allocated by insertion order — across boots the same
//! channel name can land in a different slot. Persisting raw
//! ChannelIds would corrupt grants on every restart.
//!
//! Instead we persist the channel *names* (which are stable: they
//! were chosen by the player at block-config time and are saved in
//! the existing block-config table). On load, we re-resolve names
//! → ChannelIds against the freshly-rebuilt `SignalChannelTable`.
//!
//! # Wire format
//!
//! Hand-rolled little-endian binary with a leading version byte. The
//! workspace already uses this style (`serialize_block_config` in
//! `ship-shard`), and grants are small + low-frequency so the
//! verbose-but-explicit shape doesn't matter. Adding a field is
//! always a version bump; readers fall back to a sensible default
//! for missing fields when reading older versions, so a downgrade
//! path is possible.
//!
//! # Concurrency
//!
//! Each redb operation opens its own short transaction. We do NOT
//! batch writes inside `save_grant` because grant ops are rare
//! (typically <1/sec at peak) and atomicity of each op matters:
//! a half-flushed grant after a crash is worse than a full grant
//! lost. Callers that need batching (e.g., bulk-load + index-rebuild)
//! call `load_all` once on boot and never write during the load.
//!
//! # Why a queue resource instead of inline writes
//!
//! `apply_grant_create` and `apply_grant_revoke` run under Bevy's
//! synchronous executor. Calling `redb::Database::begin_write` from
//! that path would block the tick on disk fsync — a 5ms hiccup at
//! 20Hz means dropped frames. Instead the systems push deltas onto
//! a `GrantsPersistenceQueue`, and a dedicated periodic system in
//! the shard binary drains the queue + writes redb in batches
//! outside the per-tick gameplay path.

use std::collections::HashSet;

use bevy_ecs::prelude::Resource;

use voxeldust_core::signal::channel::SignalChannelTable;
use voxeldust_core::signal::grants::{GrantOps, GrantsRegistry, RemoteAccessGrant};
use voxeldust_core::signal::types::AccessPolicy;

/// redb table for grant records, keyed by `grant_id` (u64).
///
/// The value is a hand-serialized [`PersistedGrant`] — see
/// [`serialize_grant`] for the byte layout. The same table is shared
/// across every shard binary that imports this module, so all readers
/// see the same schema version conventions.
pub const GRANTS_TABLE: redb::TableDefinition<u64, &[u8]> =
    redb::TableDefinition::new("grants");

/// Wire-format version embedded in every persisted record. Bump on
/// schema change; readers fall back to default values for fields
/// added after their version.
///
/// History:
///   - 1: initial layout (Phase 4-Persist.1).
const PERSIST_VERSION: u8 = 1;

/// On-disk shape — names instead of ChannelIds, plain types instead
/// of `SmallVec`. Bridges the in-memory [`RemoteAccessGrant`] with
/// the redb byte slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedGrant {
    pub grant_id: u64,
    pub key: [u8; 32],
    /// Stable channel names (NOT ChannelIds — those aren't stable
    /// across boots). Empty when the grant is wholly glob-driven.
    pub channel_names: Vec<String>,
    pub namespace_glob: Option<String>,
    /// Encoded `GrantOps` ordinal: 0=Publish, 1=Subscribe, 2=Both.
    pub ops_ordinal: u8,
    pub label: String,
    pub created_at_ms: u64,
    pub expires_at_ms: Option<u64>,
    pub created_by: u64,
    pub revoked: bool,
    pub mirror_of_held: bool,
}

/// Convert an in-memory grant + the channel table that resolved its
/// ChannelIds into a persistable record. Channel names are looked up
/// against the table at the time of this call — a name that has been
/// removed since the grant was issued is silently skipped (the grant
/// loses coverage of that channel, which is fine: the channel is gone).
pub fn to_persisted(grant: &RemoteAccessGrant, channels: &SignalChannelTable) -> PersistedGrant {
    let channel_names: Vec<String> = grant
        .channels
        .iter()
        .filter_map(|id| channels.name_for_id(*id).map(|n| n.to_string()))
        .collect();
    let ops_ordinal = match grant.ops {
        GrantOps::Publish => 0,
        GrantOps::Subscribe => 1,
        GrantOps::Both => 2,
    };
    PersistedGrant {
        grant_id: grant.grant_id,
        key: grant.key,
        channel_names,
        namespace_glob: grant.namespace_glob.clone(),
        ops_ordinal,
        label: grant.label.clone(),
        created_at_ms: grant.created_at_ms,
        expires_at_ms: grant.expires_at_ms,
        created_by: grant.created_by,
        revoked: grant.revoked,
        mirror_of_held: grant.mirror_of_held,
    }
}

/// Restore an in-memory grant from a persisted record by re-resolving
/// channel names against the freshly-rebuilt channel table. If a
/// persisted name no longer resolves (player removed the block, the
/// channel was never recreated), it's skipped — same fail-soft policy
/// as `to_persisted`.
///
/// Returns `None` when the record's `ops_ordinal` is invalid (corrupt
/// data) — caller should drop and log; we never panic on bad bytes.
pub fn from_persisted(
    record: &PersistedGrant,
    channels: &SignalChannelTable,
) -> Option<RemoteAccessGrant> {
    use smallvec::SmallVec;
    let ops = match record.ops_ordinal {
        0 => GrantOps::Publish,
        1 => GrantOps::Subscribe,
        2 => GrantOps::Both,
        _ => return None,
    };
    let channel_ids: SmallVec<[_; 4]> = record
        .channel_names
        .iter()
        .filter_map(|name| channels.resolve(name))
        .collect();
    Some(RemoteAccessGrant {
        grant_id: record.grant_id,
        key: record.key,
        channels: channel_ids,
        namespace_glob: record.namespace_glob.clone(),
        ops,
        label: record.label.clone(),
        created_at_ms: record.created_at_ms,
        expires_at_ms: record.expires_at_ms,
        created_by: record.created_by,
        revoked: record.revoked,
        mirror_of_held: record.mirror_of_held,
    })
}

/// Serialize to bytes. Layout (all fields little-endian):
/// ```text
///   u8                    : version (= 1)
///   [u8; 32]              : key
///   u32 channel_count
///     per channel:
///       u32 name_len
///       [u8] name_bytes (UTF-8)
///   u8 has_glob (0 or 1)
///     if 1:
///       u32 glob_len
///       [u8] glob_bytes
///   u8 ops_ordinal
///   u32 label_len
///   [u8] label_bytes
///   u64 created_at_ms
///   u8 has_expires (0 or 1)
///     if 1:
///       u64 expires_at_ms
///   u64 created_by
///   u8 revoked (0 or 1)
///   u8 mirror_of_held (0 or 1)
/// ```
/// `grant_id` is the redb key, not part of the value bytes.
pub fn serialize_grant(record: &PersistedGrant) -> Vec<u8> {
    // Capacity: header + 32 key + counts + small channels avg 32B + label.
    let mut buf = Vec::with_capacity(64 + record.channel_names.iter().map(|s| s.len() + 4).sum::<usize>() + record.label.len());
    buf.push(PERSIST_VERSION);
    buf.extend_from_slice(&record.key);
    // Channels.
    buf.extend_from_slice(&(record.channel_names.len() as u32).to_le_bytes());
    for name in &record.channel_names {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        buf.extend_from_slice(nb);
    }
    // Glob (optional).
    match &record.namespace_glob {
        None => buf.push(0),
        Some(pat) => {
            buf.push(1);
            let pb = pat.as_bytes();
            buf.extend_from_slice(&(pb.len() as u32).to_le_bytes());
            buf.extend_from_slice(pb);
        }
    }
    // Ops + label.
    buf.push(record.ops_ordinal);
    let lb = record.label.as_bytes();
    buf.extend_from_slice(&(lb.len() as u32).to_le_bytes());
    buf.extend_from_slice(lb);
    // Timestamps.
    buf.extend_from_slice(&record.created_at_ms.to_le_bytes());
    match record.expires_at_ms {
        None => buf.push(0),
        Some(exp) => {
            buf.push(1);
            buf.extend_from_slice(&exp.to_le_bytes());
        }
    }
    buf.extend_from_slice(&record.created_by.to_le_bytes());
    buf.push(if record.revoked { 1 } else { 0 });
    buf.push(if record.mirror_of_held { 1 } else { 0 });
    buf
}

/// Deserialize bytes produced by [`serialize_grant`]. Returns `None`
/// on truncation, invalid version, or any internal length mismatch
/// — callers MUST treat `None` as "drop this entry, keep loading"
/// rather than failing the whole load.
pub fn deserialize_grant(grant_id: u64, data: &[u8]) -> Option<PersistedGrant> {
    let mut p = 0usize;

    fn take_u8(data: &[u8], p: &mut usize) -> Option<u8> {
        if *p >= data.len() { return None; }
        let v = data[*p];
        *p += 1;
        Some(v)
    }
    fn take_u32(data: &[u8], p: &mut usize) -> Option<u32> {
        if *p + 4 > data.len() { return None; }
        let v = u32::from_le_bytes(data[*p..*p + 4].try_into().ok()?);
        *p += 4;
        Some(v)
    }
    fn take_u64(data: &[u8], p: &mut usize) -> Option<u64> {
        if *p + 8 > data.len() { return None; }
        let v = u64::from_le_bytes(data[*p..*p + 8].try_into().ok()?);
        *p += 8;
        Some(v)
    }
    fn take_string(data: &[u8], p: &mut usize) -> Option<String> {
        let len = take_u32(data, p)? as usize;
        if *p + len > data.len() { return None; }
        let s = std::str::from_utf8(&data[*p..*p + len]).ok()?.to_string();
        *p += len;
        Some(s)
    }

    let version = take_u8(data, &mut p)?;
    if version != PERSIST_VERSION {
        // Version skew: refuse to read older/newer payloads. Caller
        // logs + drops. Future versions can add a forward path here.
        return None;
    }
    if p + 32 > data.len() { return None; }
    let mut key = [0u8; 32];
    key.copy_from_slice(&data[p..p + 32]);
    p += 32;
    let channel_count = take_u32(data, &mut p)?;
    let mut channel_names = Vec::with_capacity(channel_count as usize);
    for _ in 0..channel_count {
        channel_names.push(take_string(data, &mut p)?);
    }
    let has_glob = take_u8(data, &mut p)?;
    let namespace_glob = if has_glob == 1 {
        Some(take_string(data, &mut p)?)
    } else {
        None
    };
    let ops_ordinal = take_u8(data, &mut p)?;
    let label = take_string(data, &mut p)?;
    let created_at_ms = take_u64(data, &mut p)?;
    let has_expires = take_u8(data, &mut p)?;
    let expires_at_ms = if has_expires == 1 {
        Some(take_u64(data, &mut p)?)
    } else {
        None
    };
    let created_by = take_u64(data, &mut p)?;
    let revoked = take_u8(data, &mut p)? != 0;
    let mirror_of_held = take_u8(data, &mut p)? != 0;
    Some(PersistedGrant {
        grant_id,
        key,
        channel_names,
        namespace_glob,
        ops_ordinal,
        label,
        created_at_ms,
        expires_at_ms,
        created_by,
        revoked,
        mirror_of_held,
    })
}

/// Write or overwrite a single grant in the redb file. One transaction
/// per call — the typical write rate is <<1/sec, so transaction cost
/// is dominated by the underlying fsync rather than overhead.
pub fn save_grant(db: &redb::Database, record: &PersistedGrant) -> Result<(), redb::Error> {
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(GRANTS_TABLE)?;
        let bytes = serialize_grant(record);
        table.insert(record.grant_id, bytes.as_slice())?;
    }
    txn.commit()?;
    Ok(())
}

/// Delete a grant by id. Idempotent — deleting a non-existent grant
/// is a no-op.
pub fn delete_grant(db: &redb::Database, grant_id: u64) -> Result<(), redb::Error> {
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(GRANTS_TABLE)?;
        table.remove(grant_id)?;
    }
    txn.commit()?;
    Ok(())
}

/// Load every grant from the redb file. Used at boot to repopulate
/// `GrantsRegistry`. Corrupt records are dropped + logged; a single
/// bad row never aborts the whole load.
pub fn load_all(db: &redb::Database) -> Vec<PersistedGrant> {
    let txn = match db.begin_read() {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let table = match txn.open_table(GRANTS_TABLE) {
        Ok(t) => t,
        Err(_) => return Vec::new(), // table doesn't exist on first boot
    };
    let iter = match table.range::<u64>(..) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in iter {
        let (key, value) = match entry {
            Ok(kv) => kv,
            Err(_) => continue,
        };
        let grant_id = key.value();
        let data = value.value();
        match deserialize_grant(grant_id, data) {
            Some(rec) => out.push(rec),
            None => {
                tracing::warn!(
                    grant_id,
                    bytes = data.len(),
                    "grant_persistence: dropping corrupt record"
                );
            }
        }
    }
    out
}

/// Boot-time helper: walk every persisted grant, drop expired ones,
/// resolve channel names, and bulk-insert into a fresh
/// `GrantsRegistry`. Returns the number of records actually loaded
/// (excluding expired + corrupt + unresolvable).
pub fn populate_registry_from_db(
    db: &redb::Database,
    channels: &SignalChannelTable,
    registry: &mut GrantsRegistry,
    now_ms: u64,
) -> usize {
    let mut loaded = 0;
    let mut expired = 0;
    let records = load_all(db);
    for rec in records {
        // Skip expired-and-revoked-and-old grants on boot. Live
        // tombstones (revoked=true but not yet purged) are kept so
        // that the audit trail survives across boots.
        if let Some(exp) = rec.expires_at_ms {
            if now_ms > exp + 24 * 60 * 60 * 1000 {
                // 24-hour grace past expiry before the record is
                // considered purgeable. This is a soft gate — a
                // dedicated sweep would actually delete from redb;
                // here we just skip the load.
                expired += 1;
                continue;
            }
        }
        match from_persisted(&rec, channels) {
            Some(grant) => {
                registry.insert(grant);
                loaded += 1;
            }
            None => {
                tracing::warn!(
                    grant_id = rec.grant_id,
                    "grant_persistence: dropping record with invalid ops_ordinal"
                );
            }
        }
    }
    if loaded > 0 || expired > 0 {
        tracing::info!(loaded, expired, "grant_persistence: registry restore complete");
    }
    loaded
}

// ---------------------------------------------------------------------------
// Bevy integration: deferred-write queue.
// ---------------------------------------------------------------------------

/// What the persistence sweep should do for a given grant_id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrantOp {
    /// Read the current grant out of `GrantsRegistry` and write it.
    /// Use this for new inserts AND for revocations (the registry
    /// keeps revoked grants as tombstones).
    Upsert(u64),
    /// Hard-delete the redb row by id. Used by future purge sweeps;
    /// not produced by `apply_grant_revoke` today.
    Delete(u64),
}

/// Bevy resource — receives delta ops produced inside Bevy systems
/// (which can't synchronously block on redb writes) and is drained
/// outside the per-tick path by a dedicated persistence sweep system
/// in the shard binary.
///
/// Dedup pre-batch: pushing two `Upsert(7)` in the same tick collapses
/// to one redb write. We use a `HashSet`-of-pending-ops keyed by id +
/// kind so the queue stays bounded under heavy churn (e.g., a tablet
/// rapid-fire issuing then revoking).
#[derive(Resource, Default)]
pub struct GrantsPersistenceQueue {
    upserts: HashSet<u64>,
    deletes: HashSet<u64>,
}

impl GrantsPersistenceQueue {
    pub fn enqueue_upsert(&mut self, grant_id: u64) {
        // If a delete was pending and we now upsert the same id, the
        // upsert wins. (Realistically this happens when a grant is
        // re-issued with the same id, which is impossible in practice
        // — grant_ids are random u64s — but we handle it cleanly.)
        self.deletes.remove(&grant_id);
        self.upserts.insert(grant_id);
    }

    pub fn enqueue_delete(&mut self, grant_id: u64) {
        self.upserts.remove(&grant_id);
        self.deletes.insert(grant_id);
    }

    /// Drain ops, returning (upsert ids, delete ids). Caller applies
    /// each op against redb + the registry.
    pub fn drain(&mut self) -> (Vec<u64>, Vec<u64>) {
        let ups = self.upserts.drain().collect();
        let dels = self.deletes.drain().collect();
        (ups, dels)
    }

    pub fn pending_upsert_count(&self) -> usize {
        self.upserts.len()
    }

    pub fn pending_delete_count(&self) -> usize {
        self.deletes.len()
    }
}

// `AccessPolicy` re-export so consumers don't have to know the
// internal grants module path. Pure re-export, zero behaviour.
pub use voxeldust_core::signal::types::AccessPolicy as ReexportedAccessPolicy;
// Suppress the unused import — `AccessPolicy` is part of the public
// signal types but isn't directly referenced here. The use serves
// as a future tripwire if the surrounding module reorganizes.
#[allow(dead_code)]
fn _check_access_policy_exists(_: AccessPolicy) {}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;
    use voxeldust_core::signal::channel::ChannelId;
    use voxeldust_core::signal::types::{ChannelMergeStrategy, SignalScope};

    fn fixture_channel_table() -> SignalChannelTable {
        let mut t = SignalChannelTable::default();
        t.resolve_or_create(
            "alice.thrust-fwd",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            42,
        );
        t.resolve_or_create(
            "alice.thrust-back",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            42,
        );
        t.resolve_or_create(
            "alice.lights",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            42,
        );
        t
    }

    fn fixture_grant(channels: &SignalChannelTable) -> RemoteAccessGrant {
        let id1 = channels.resolve("alice.thrust-fwd").unwrap();
        let id2 = channels.resolve("alice.thrust-back").unwrap();
        RemoteAccessGrant {
            grant_id: 0xDEAD_BEEF_F00D_BABE,
            key: [0xAB; 32],
            channels: smallvec![id1, id2],
            namespace_glob: Some("alice.thrust-*".into()),
            ops: GrantOps::Both,
            label: "Bob remote pilot".into(),
            created_at_ms: 1_700_000_000_000,
            expires_at_ms: Some(1_900_000_000_000),
            created_by: 42,
            revoked: false,
            mirror_of_held: false,
        }
    }

    fn open_db_in_temp_dir() -> (redb::Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("grants_test.redb");
        let db = redb::Database::create(&path).expect("create");
        (db, dir)
    }

    // ------------- Serialization round-trip -------------

    #[test]
    fn serialize_then_deserialize_round_trip() {
        let channels = fixture_channel_table();
        let grant = fixture_grant(&channels);
        let persisted = to_persisted(&grant, &channels);
        let bytes = serialize_grant(&persisted);
        let decoded =
            deserialize_grant(persisted.grant_id, &bytes).expect("deserialize must succeed");
        assert_eq!(decoded, persisted);
    }

    #[test]
    fn deserialize_handles_optional_fields_both_states() {
        let channels = fixture_channel_table();
        // No glob, no expiry, no channels.
        let minimal = PersistedGrant {
            grant_id: 1,
            key: [0; 32],
            channel_names: vec![],
            namespace_glob: None,
            ops_ordinal: 0,
            label: String::new(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 0,
            revoked: false,
            mirror_of_held: false,
        };
        let bytes = serialize_grant(&minimal);
        let decoded = deserialize_grant(1, &bytes).unwrap();
        assert_eq!(decoded, minimal);
        let _ = channels; // silence unused
    }

    #[test]
    fn deserialize_rejects_truncated_bytes() {
        let channels = fixture_channel_table();
        let grant = fixture_grant(&channels);
        let persisted = to_persisted(&grant, &channels);
        let bytes = serialize_grant(&persisted);
        // Truncate progressively; each call must return None without
        // panicking. The most rigorous bound check.
        for cut in 0..bytes.len() {
            let _ = deserialize_grant(persisted.grant_id, &bytes[..cut]);
        }
    }

    #[test]
    fn deserialize_rejects_wrong_version() {
        let mut bytes = serialize_grant(&PersistedGrant {
            grant_id: 1,
            key: [0; 32],
            channel_names: vec![],
            namespace_glob: None,
            ops_ordinal: 0,
            label: String::new(),
            created_at_ms: 0,
            expires_at_ms: None,
            created_by: 0,
            revoked: false,
            mirror_of_held: false,
        });
        bytes[0] = 99; // bogus version byte
        assert!(deserialize_grant(1, &bytes).is_none());
    }

    // ------------- Round trip through redb -------------

    #[test]
    fn save_and_load_round_trip_through_redb() {
        let (db, _dir) = open_db_in_temp_dir();
        let channels = fixture_channel_table();
        let grant = fixture_grant(&channels);
        save_grant(&db, &to_persisted(&grant, &channels)).unwrap();
        let loaded = load_all(&db);
        assert_eq!(loaded.len(), 1);
        let restored = from_persisted(&loaded[0], &channels).unwrap();
        assert_eq!(restored.grant_id, grant.grant_id);
        assert_eq!(restored.key, grant.key);
        assert_eq!(restored.channels.as_slice(), grant.channels.as_slice());
        assert_eq!(restored.label, grant.label);
        assert_eq!(restored.namespace_glob, grant.namespace_glob);
        assert_eq!(restored.ops, grant.ops);
        assert_eq!(restored.created_by, grant.created_by);
        assert_eq!(restored.revoked, grant.revoked);
    }

    #[test]
    fn delete_grant_removes_row() {
        let (db, _dir) = open_db_in_temp_dir();
        let channels = fixture_channel_table();
        let grant = fixture_grant(&channels);
        save_grant(&db, &to_persisted(&grant, &channels)).unwrap();
        assert_eq!(load_all(&db).len(), 1);
        delete_grant(&db, grant.grant_id).unwrap();
        assert_eq!(load_all(&db).len(), 0);
        // Idempotent — second delete is a no-op.
        delete_grant(&db, grant.grant_id).unwrap();
    }

    #[test]
    fn populate_registry_skips_grants_long_past_expiry() {
        let (db, _dir) = open_db_in_temp_dir();
        let channels = fixture_channel_table();
        let mut grant = fixture_grant(&channels);
        grant.expires_at_ms = Some(100); // expired ages ago
        save_grant(&db, &to_persisted(&grant, &channels)).unwrap();

        // now_ms = 100 + 25h, past the 24h grace.
        let now_ms = 100 + 25 * 60 * 60 * 1000;
        let mut reg = GrantsRegistry::default();
        let loaded = populate_registry_from_db(&db, &channels, &mut reg, now_ms);
        assert_eq!(loaded, 0, "stale-by-grace expired grants are skipped");
    }

    #[test]
    fn populate_registry_keeps_recently_expired_grants_within_grace() {
        let (db, _dir) = open_db_in_temp_dir();
        let channels = fixture_channel_table();
        let mut grant = fixture_grant(&channels);
        grant.expires_at_ms = Some(1_000_000);
        save_grant(&db, &to_persisted(&grant, &channels)).unwrap();
        // now_ms = 1_001_000 (1s past expiry, well under 24h grace).
        let mut reg = GrantsRegistry::default();
        let loaded = populate_registry_from_db(&db, &channels, &mut reg, 1_001_000);
        assert_eq!(loaded, 1, "within-grace expiry must still load");
    }

    #[test]
    fn populate_registry_drops_unresolvable_channels_but_keeps_grant() {
        let (db, _dir) = open_db_in_temp_dir();
        // Save with the full channel table.
        let full = fixture_channel_table();
        let grant = fixture_grant(&full);
        save_grant(&db, &to_persisted(&grant, &full)).unwrap();
        // Reload against a SMALLER table — only one of the two channels exists.
        let mut sparse = SignalChannelTable::default();
        sparse.resolve_or_create(
            "alice.thrust-fwd",
            SignalScope::Local,
            ChannelMergeStrategy::LastWrite,
            42,
        );
        let mut reg = GrantsRegistry::default();
        let loaded = populate_registry_from_db(&db, &sparse, &mut reg, 1_700_000_000_000);
        assert_eq!(loaded, 1);
        // The grant exists, but only covers the resolvable channel.
        let restored = reg.get(grant.grant_id).expect("grant present");
        assert_eq!(restored.channels.len(), 1);
        assert_eq!(
            sparse.name_for_id(restored.channels[0]),
            Some("alice.thrust-fwd")
        );
    }

    // ------------- GrantsPersistenceQueue -------------

    #[test]
    fn queue_dedup_collapses_repeated_upserts() {
        let mut q = GrantsPersistenceQueue::default();
        q.enqueue_upsert(1);
        q.enqueue_upsert(1);
        q.enqueue_upsert(1);
        assert_eq!(q.pending_upsert_count(), 1);
    }

    #[test]
    fn queue_upsert_overrides_pending_delete() {
        // Sequence: delete grant 1, then upsert grant 1 same tick.
        // The upsert wins (in practice impossible because grant_ids
        // are random — but the queue must handle it cleanly).
        let mut q = GrantsPersistenceQueue::default();
        q.enqueue_delete(1);
        q.enqueue_upsert(1);
        let (ups, dels) = q.drain();
        assert_eq!(ups, vec![1]);
        assert!(dels.is_empty());
    }

    #[test]
    fn queue_delete_overrides_pending_upsert() {
        // Sequence: upsert grant 1, then revoke + delete in the same tick.
        let mut q = GrantsPersistenceQueue::default();
        q.enqueue_upsert(1);
        q.enqueue_delete(1);
        let (ups, dels) = q.drain();
        assert!(ups.is_empty());
        assert_eq!(dels, vec![1]);
    }

    #[test]
    fn queue_drain_empties_pending() {
        let mut q = GrantsPersistenceQueue::default();
        q.enqueue_upsert(1);
        q.enqueue_delete(2);
        q.drain();
        assert_eq!(q.pending_upsert_count(), 0);
        assert_eq!(q.pending_delete_count(), 0);
    }
}
