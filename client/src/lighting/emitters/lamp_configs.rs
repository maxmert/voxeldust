//! Server-broadcast lamp configurations.
//!
//! `ChunkSnapshotData.lamp_configs` and `ChunkDeltaData.lamp_configs`
//! carry per-`(host_block_pos, face)` lamp settings the player has
//! customised through the F-key config UI (Slice C UI lands in a
//! follow-up). This module ingests both into the [`LampConfigs`]
//! resource and exposes a lookup that
//! `sub_block::spawn_sub_block_emitters` consults when spawning each
//! lamp.
//!
//! The ingest runs once per net event so the resource is current at
//! the moment the chunk re-meshes (and re-spawns its lamp children).
//! No tear-down is needed: when a chunk re-spawns it overwrites the
//! lamp set from scratch using whatever values are in `LampConfigs`
//! at that instant.
//!
//! Storage is server-authoritative: every connected client receives
//! identical `LampConfigEntryData` payloads in their chunk broadcasts,
//! so they all spawn the same lamp colour / channel binding.

use std::collections::HashMap;

use bevy::prelude::*;
use glam::IVec3;

use voxeldust_core::block::palette::CHUNK_SIZE;
use voxeldust_core::block::sub_block::LampConfig;
use voxeldust_core::client_message::LampConfigEntryData;

use crate::net::{GameEvent, NetEvent};
use crate::shard::registry::{PrimaryShard, Secondaries};
use crate::shard::ShardKey;

/// Resource: every player-customised lamp config the client has
/// received. Keyed by `(shard, world block_pos, face)`.
///
/// Lamps without an entry in this map use the default settings from
/// `BlockRegistry::sub_block_light_spec` for the lamp's `SubBlockType`.
#[derive(Resource, Default, Debug)]
pub struct LampConfigs {
    pub by_key: HashMap<(ShardKey, IVec3, u8), LampConfig>,
}

impl LampConfigs {
    /// Look up the player config for a lamp at `(shard, block_pos, face)`.
    /// Returns `None` if the player has not customised it.
    pub fn get(&self, shard: ShardKey, block_pos: IVec3, face: u8) -> Option<&LampConfig> {
        self.by_key.get(&(shard, block_pos, face))
    }

    pub fn set(&mut self, shard: ShardKey, block_pos: IVec3, face: u8, config: LampConfig) {
        self.by_key.insert((shard, block_pos, face), config);
    }
}

pub struct LampConfigsPlugin;

impl Plugin for LampConfigsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LampConfigs>()
            .add_systems(Update, ingest_lamp_configs);
    }
}

/// Drain `ChunkSnapshot` / `ChunkDelta` events into `LampConfigs`. Each
/// entry's local `(bx, by, bz)` is converted to world coordinates by
/// adding `chunk_index × CHUNK_SIZE`. Shard resolution mirrors the
/// chunk-stream pattern: primary chunks attribute to the primary key,
/// secondary chunks to the matching secondary entry.
fn ingest_lamp_configs(
    mut events: MessageReader<GameEvent>,
    mut configs: ResMut<LampConfigs>,
    primary: Res<PrimaryShard>,
    secondaries: Res<Secondaries>,
) {
    let cs = CHUNK_SIZE as i32;
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::ChunkSnapshot(cs_data) => {
                let Some(shard) = primary.current else { continue };
                let chunk_origin = chunk_world_origin(
                    cs_data.chunk_x,
                    cs_data.chunk_y,
                    cs_data.chunk_z,
                    cs,
                );
                for entry in &cs_data.lamp_configs {
                    apply_entry(&mut configs, shard, chunk_origin, entry);
                }
            }
            NetEvent::ChunkDelta(cd) => {
                let Some(shard) = primary.current else { continue };
                let chunk_origin =
                    chunk_world_origin(cd.chunk_x, cd.chunk_y, cd.chunk_z, cs);
                for entry in &cd.lamp_configs {
                    apply_entry(&mut configs, shard, chunk_origin, entry);
                }
            }
            NetEvent::SecondaryChunkSnapshot { seed, data } => {
                let Some(shard) = resolve_secondary(&secondaries, *seed) else {
                    continue;
                };
                let chunk_origin = chunk_world_origin(
                    data.chunk_x,
                    data.chunk_y,
                    data.chunk_z,
                    cs,
                );
                for entry in &data.lamp_configs {
                    apply_entry(&mut configs, shard, chunk_origin, entry);
                }
            }
            NetEvent::SecondaryChunkDelta { seed, data } => {
                let Some(shard) = resolve_secondary(&secondaries, *seed) else {
                    continue;
                };
                let chunk_origin =
                    chunk_world_origin(data.chunk_x, data.chunk_y, data.chunk_z, cs);
                for entry in &data.lamp_configs {
                    apply_entry(&mut configs, shard, chunk_origin, entry);
                }
            }
            _ => {}
        }
    }
}

fn apply_entry(
    configs: &mut LampConfigs,
    shard: ShardKey,
    chunk_origin: IVec3,
    entry: &LampConfigEntryData,
) {
    let world_pos = chunk_origin
        + IVec3::new(entry.bx as i32, entry.by as i32, entry.bz as i32);
    configs.set(shard, world_pos, entry.face, entry.to_lamp_config());
}

fn chunk_world_origin(cx: i32, cy: i32, cz: i32, cs: i32) -> IVec3 {
    IVec3::new(cx * cs, cy * cs, cz * cs)
}

fn resolve_secondary(secondaries: &Secondaries, seed: u64) -> Option<ShardKey> {
    secondaries
        .runtimes
        .keys()
        .find(|k| k.seed == seed)
        .copied()
}
