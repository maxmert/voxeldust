//! `ChunkStreamPlugin` — registers all chunk streaming resources +
//! systems.

use bevy::prelude::*;

use crate::chunk::cache::ChunkStorageCache;
use crate::chunk::material::ChunkMaterialCache;
use crate::chunk::stream::{
    ingest_primary_chunks, ingest_secondary_chunks, ChunkIndex, SharedBlockRegistry,
};
use crate::shard::WorldStateIngestSet;

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct ChunkStreamSet;

pub struct ChunkStreamPlugin;

impl Plugin for ChunkStreamPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkIndex>()
            .init_resource::<SharedBlockRegistry>()
            .init_resource::<ChunkMaterialCache>()
            .init_resource::<ChunkStorageCache>()
            .configure_sets(Update, ChunkStreamSet.after(WorldStateIngestSet))
            .add_systems(
                Update,
                (ingest_primary_chunks, ingest_secondary_chunks).in_set(ChunkStreamSet),
            );
    }
}
