use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;

use voxeldust_core::shard_types::{ShardId, ShardInfo};

const SHARD_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("shards");
const LAUNCH_CONFIG_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("launch_configs");

use crate::registry::LaunchConfig;

/// Open or create the orchestrator database.
pub fn open_db(path: &Path) -> Result<Database, Box<dyn std::error::Error>> {
    let db = Database::create(path)?;
    let txn = db.begin_write()?;
    {
        let _ = txn.open_table(SHARD_TABLE)?;
        let _ = txn.open_table(LAUNCH_CONFIG_TABLE)?;
    }
    txn.commit()?;
    Ok(db)
}

/// Save a shard's info to the database.
pub fn save_shard(
    db: &Database,
    id: ShardId,
    info: &ShardInfo,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = serde_json::to_vec(info)?;
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(SHARD_TABLE)?;
        table.insert(id.0, bytes.as_slice())?;
    }
    txn.commit()?;
    Ok(())
}

/// Save a launch config.
pub fn save_launch_config(
    db: &Database,
    id: ShardId,
    config: &LaunchConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = serde_json::to_vec(config)?;
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(LAUNCH_CONFIG_TABLE)?;
        table.insert(id.0, bytes.as_slice())?;
    }
    txn.commit()?;
    Ok(())
}

/// Load all shards from the database.
pub fn load_all_shards(
    db: &Database,
) -> Result<Vec<(ShardId, ShardInfo)>, Box<dyn std::error::Error>> {
    let txn = db.begin_read()?;
    let table = txn.open_table(SHARD_TABLE)?;
    let mut results = Vec::new();
    for entry in table.iter()? {
        let (key, value) = entry?;
        let id = ShardId(key.value());
        let info: ShardInfo = serde_json::from_slice(value.value())?;
        results.push((id, info));
    }
    Ok(results)
}

/// Load all launch configs from the database.
pub fn load_all_launch_configs(
    db: &Database,
) -> Result<Vec<(ShardId, LaunchConfig)>, Box<dyn std::error::Error>> {
    let txn = db.begin_read()?;
    let table = txn.open_table(LAUNCH_CONFIG_TABLE)?;
    let mut results = Vec::new();
    for entry in table.iter()? {
        let (key, value) = entry?;
        let id = ShardId(key.value());
        let config: LaunchConfig = serde_json::from_slice(value.value())?;
        results.push((id, config));
    }
    Ok(results)
}

/// Remove a shard from the database.
pub fn remove_shard(db: &Database, id: ShardId) -> Result<(), Box<dyn std::error::Error>> {
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(SHARD_TABLE)?;
        table.remove(id.0)?;
    }
    {
        let mut table = txn.open_table(LAUNCH_CONFIG_TABLE)?;
        table.remove(id.0)?;
    }
    txn.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use voxeldust_core::shard_types::*;

    fn test_shard_info(id: u64) -> ShardInfo {
        ShardInfo {
            id: ShardId(id),
            shard_type: ShardType::Planet,
            state: ShardState::Ready,
            endpoint: ShardEndpoint {
                tcp_addr: "127.0.0.1:7777".parse::<SocketAddr>().unwrap(),
                udp_addr: "127.0.0.1:7778".parse::<SocketAddr>().unwrap(),
                quic_addr: "127.0.0.1:7779".parse::<SocketAddr>().unwrap(),
            },
            planet_seed: Some(42),
            sectors: Some(vec![0, 1, 2, 3, 4, 5]),
            system_seed: None,
            ship_id: None,
            galaxy_seed: None,
            host_shard_id: None,
            launch_args: vec!["--seed".into(), "42".into()],
        }
    }

    #[test]
    fn roundtrip_shard_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let db = open_db(&db_path).unwrap();

        let info = test_shard_info(1);
        save_shard(&db, ShardId(1), &info).unwrap();

        let loaded = load_all_shards(&db).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, ShardId(1));
        assert_eq!(loaded[0].1.planet_seed, Some(42));
    }

    #[test]
    fn remove_shard_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let db = open_db(&db_path).unwrap();

        save_shard(&db, ShardId(1), &test_shard_info(1)).unwrap();
        save_shard(&db, ShardId(2), &test_shard_info(2)).unwrap();

        remove_shard(&db, ShardId(1)).unwrap();
        let loaded = load_all_shards(&db).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, ShardId(2));
    }
}
