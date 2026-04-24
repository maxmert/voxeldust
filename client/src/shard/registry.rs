//! `ShardTypeRegistry` + `ShardRegistryPlugin`.
//!
//! The registry stores boxed `ShardTypePlugin` trait objects keyed by
//! wire `shard_type`. The plugin drives lifecycle: on
//! `NetEvent::Connected` / `SecondaryConnected`, spawn a root
//! `ChunkSource` entity + `ShardRuntime` entry. On `Disconnected` /
//! `SecondaryDisconnected`, despawn recursively. Unknown shard_types
//! warn but are still tracked generically (future-proofing: if the
//! server ships a new shard-type before the client has a plugin for
//! it, the chunks still render as a generic secondary).

use std::collections::HashMap;
use std::time::Instant;

use bevy::prelude::*;
use glam::DQuat;

use crate::net::{GameEvent, NetEvent, NetworkBridgeSet};
use crate::shard::origin::ShardOrigin;
use crate::shard::plugin::ShardTypePlugin;
use crate::shard::runtime::{ChunkSource, ShardKey, ShardRuntime};

/// System set for the shard-registry lifecycle stage. Downstream plugins
/// order themselves `.after(ShardRegistrySet)` so they see a consistent
/// `PrimaryShard` / `Secondaries` state.
#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct ShardRegistrySet;

// ─────────────────────────────────────────────────────────────────────
// Registry resource
// ─────────────────────────────────────────────────────────────────────

/// Boxed plugin store. Populated at app-build time via `register(...)`.
/// Lookup via `get(shard_type)` returns `None` for unknown types — the
/// registry plugin logs a warning and falls back to generic spawn /
/// despawn.
#[derive(Resource, Default)]
pub struct ShardTypeRegistry {
    plugins: HashMap<u8, Box<dyn ShardTypePlugin>>,
}

impl ShardTypeRegistry {
    pub fn register(&mut self, plugin: Box<dyn ShardTypePlugin>) {
        let ty = plugin.shard_type();
        if let Some(existing) = self.plugins.insert(ty, plugin) {
            tracing::warn!(
                shard_type = ty,
                old = existing.name(),
                "ShardTypeRegistry::register replaced an existing plugin",
            );
        } else {
            tracing::info!(
                shard_type = ty,
                name = self.plugins[&ty].name(),
                "registered shard-type plugin",
            );
        }
    }

    pub fn get(&self, shard_type: u8) -> Option<&dyn ShardTypePlugin> {
        self.plugins.get(&shard_type).map(|p| p.as_ref())
    }

    pub fn names(&self) -> Vec<(u8, &'static str)> {
        self.plugins.iter().map(|(t, p)| (*t, p.name())).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Shard tracking resources
// ─────────────────────────────────────────────────────────────────────

/// The current primary shard, if connected.
#[derive(Resource, Default, Debug, Clone)]
pub struct PrimaryShard {
    pub current: Option<ShardKey>,
}

/// Every secondary (observer) shard currently pre-connected.
/// Server-driven lifecycle: inserted on `SecondaryConnected`, removed on
/// `SecondaryDisconnected`. No client-side cap.
#[derive(Resource, Default)]
pub struct Secondaries {
    pub runtimes: HashMap<ShardKey, ShardRuntime>,
}

/// Single index to look up a shard's root `ChunkSource` entity (or a
/// sub-grid entity within it) from a `ShardKey` + optional sub-grid id.
/// Maintained in lockstep with spawn / despawn.
#[derive(Resource, Default)]
pub struct SourceIndex {
    pub by_shard: HashMap<ShardKey, Entity>,
    /// Populated in Phase 16 when sub-grids land.
    pub by_sub_grid: HashMap<(ShardKey, u32), Entity>,
}

// ─────────────────────────────────────────────────────────────────────
// Plugin
// ─────────────────────────────────────────────────────────────────────

pub struct ShardRegistryPlugin;

impl Plugin for ShardRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShardTypeRegistry>()
            .init_resource::<PrimaryShard>()
            .init_resource::<Secondaries>()
            .init_resource::<SourceIndex>()
            .configure_sets(Update, ShardRegistrySet.after(NetworkBridgeSet))
            .add_systems(
                Update,
                (handle_connected, handle_secondary_connected, handle_disconnects)
                    .in_set(ShardRegistrySet),
            );
    }
}

// Separate systems per event kind (CLAUDE.md: split systems by access
// pattern). They share `ShardTypeRegistry` (read-only) + the two shard
// maps (write). The write overlap is fine because the three systems
// never process the same ShardKey in the same frame — Connected,
// SecondaryConnected, and Disconnect paths are disjoint.

fn handle_connected(
    mut events: MessageReader<GameEvent>,
    registry: Res<ShardTypeRegistry>,
    mut primary: ResMut<PrimaryShard>,
    mut secondaries: ResMut<Secondaries>,
    mut source_index: ResMut<SourceIndex>,
    mut commands: Commands,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::Connected {
            shard_type,
            seed,
            reference_position,
            reference_rotation,
            ..
        } = ev
        {
            let key = ShardKey::new(*shard_type, *seed);
            // If this shard was a pre-connected secondary, promote in place:
            // reuse its entity + runtime so chunks already streamed remain.
            let runtime = match secondaries.runtimes.remove(&key) {
                Some(rt) => {
                    tracing::info!(%key, "promoted secondary → primary in place");
                    rt
                }
                None => spawn_new_shard_runtime(
                    &mut commands,
                    &mut source_index,
                    &registry,
                    key,
                    *reference_position,
                    *reference_rotation,
                ),
            };
            primary.current = Some(key);
            // Re-insert into secondaries-only-if-needed model — the
            // source_index stays authoritative for "what chunk source
            // entities exist".
            //
            // The promoted primary's runtime lives in PrimaryShard's
            // ad-hoc storage: for now we just carry the key; phase 4 adds
            // a proper PrimaryShardRuntime resource if needed. To keep
            // the runtime available for the origin system, store it in a
            // dedicated single-entry map.
            source_index.by_shard.insert(key, runtime.entity);
            // Ownership of `runtime` ends here; the ChunkSource
            // component on its entity is the authoritative store.
            drop(runtime);
        }
    }
}

fn handle_secondary_connected(
    mut events: MessageReader<GameEvent>,
    registry: Res<ShardTypeRegistry>,
    mut secondaries: ResMut<Secondaries>,
    mut source_index: ResMut<SourceIndex>,
    mut commands: Commands,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::SecondaryConnected {
            shard_type,
            seed,
            reference_position,
            reference_rotation,
        } = ev
        {
            let key = ShardKey::new(*shard_type, *seed);
            if secondaries.runtimes.contains_key(&key) {
                tracing::warn!(%key, "SecondaryConnected for already-known shard");
                continue;
            }
            let runtime = spawn_new_shard_runtime(
                &mut commands,
                &mut source_index,
                &registry,
                key,
                *reference_position,
                *reference_rotation,
            );
            secondaries.runtimes.insert(key, runtime);
        }
    }
}

fn handle_disconnects(
    mut events: MessageReader<GameEvent>,
    registry: Res<ShardTypeRegistry>,
    mut primary: ResMut<PrimaryShard>,
    mut secondaries: ResMut<Secondaries>,
    mut source_index: ResMut<SourceIndex>,
    mut commands: Commands,
) {
    for GameEvent(ev) in events.read() {
        match ev {
            NetEvent::SecondaryDisconnected { seed } => {
                // We don't know the shard_type from just the seed; search
                // the secondaries map by matching seed.
                let key = secondaries
                    .runtimes
                    .keys()
                    .find(|k| k.seed == *seed)
                    .copied();
                if let Some(key) = key {
                    despawn_shard(
                        &mut commands,
                        &mut source_index,
                        &registry,
                        &secondaries.runtimes[&key],
                    );
                    secondaries.runtimes.remove(&key);
                } else {
                    tracing::debug!(%seed, "SecondaryDisconnected for unknown secondary");
                }
            }
            NetEvent::Disconnected(_) => {
                // Full disconnect — tear down the primary (if any) + all
                // secondaries. Scene-context preservation is a concern
                // for shard transitions, NOT full session disconnect.
                if let Some(key) = primary.current.take() {
                    if let Some(entity) = source_index.by_shard.remove(&key) {
                        commands.entity(entity).despawn();
                    }
                }
                for (_, runtime) in secondaries.runtimes.drain() {
                    despawn_shard(&mut commands, &mut source_index, &registry, &runtime);
                }
            }
            _ => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

fn spawn_new_shard_runtime(
    commands: &mut Commands,
    source_index: &mut SourceIndex,
    registry: &ShardTypeRegistry,
    key: ShardKey,
    reference_position: glam::DVec3,
    reference_rotation: DQuat,
) -> ShardRuntime {
    let entity = commands
        .spawn((
            Name::new(format!("ChunkSource[{key}]")),
            ChunkSource { key },
            // f64 canonical pose — read by the origin rebase system to
            // compute per-frame camera-relative f32 translation.
            ShardOrigin::new(reference_position, reference_rotation),
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
        ))
        .id();

    source_index.by_shard.insert(key, entity);

    let runtime = ShardRuntime {
        key,
        entity,
        origin: reference_position,
        rotation: reference_rotation,
        reference_position,
        reference_rotation,
        connected_at: Instant::now(),
    };

    if let Some(plugin) = registry.get(key.shard_type) {
        tracing::info!(%key, plugin = plugin.name(), "shard spawned");
        plugin.on_shard_spawn(commands, &runtime);
    } else {
        tracing::warn!(
            shard_type = key.shard_type,
            "no plugin for shard_type — using generic secondary behavior",
        );
    }

    runtime
}

fn despawn_shard(
    commands: &mut Commands,
    source_index: &mut SourceIndex,
    registry: &ShardTypeRegistry,
    runtime: &ShardRuntime,
) {
    if let Some(plugin) = registry.get(runtime.key.shard_type) {
        plugin.on_shard_despawn(commands, runtime);
    }
    source_index.by_shard.remove(&runtime.key);
    commands.entity(runtime.entity).despawn();
    tracing::info!(key = %runtime.key, "shard despawned");
}
