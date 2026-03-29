use std::path::Path;
use std::sync::Mutex;
use std::collections::HashMap;

use voxeldust_core::shard_types::{SessionToken, ShardId};

/// A player session record.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub session_token: SessionToken,
    pub player_name: String,
    pub last_shard_id: ShardId,
    pub target_tcp_addr: String,
    pub target_udp_addr: String,
}

/// In-memory session store. For production, this would be backed by redb,
/// but for Phase 2 an in-memory store is sufficient.
pub struct SessionStore {
    sessions: Mutex<HashMap<String, SessionRecord>>, // keyed by player_name
}

impl SessionStore {
    pub fn open(_db_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            sessions: Mutex::new(HashMap::new()),
        })
    }

    pub fn save(&self, record: SessionRecord) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(record.player_name.clone(), record);
    }

    pub fn find_by_name(&self, name: &str) -> Option<SessionRecord> {
        let sessions = self.sessions.lock().unwrap();
        sessions.get(name).cloned()
    }

    pub fn remove(&self, name: &str) {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(name);
    }
}
