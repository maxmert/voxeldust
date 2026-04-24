//! GALAXY shard-type plugin (wire shard_type = 3, scene-context).
//!
//! Phase 9 MVP: registers the trait impl. Star catalog ingest + warp
//! parallax land as later phases' `render_contribution`. No blocks in
//! galaxy-space → default raycast contribution (when trait method
//! lands in Phase 14) is empty.

use bevy::prelude::*;
use glam::DVec3;

use crate::shard::{plugin::HudSummaryCtx, ShardRuntime, ShardTypePlugin, ShardTypeRegistry};

pub struct GalaxyShardPlugin;

impl Plugin for GalaxyShardPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut()
            .resource_mut::<ShardTypeRegistry>()
            .register(Box::new(GalaxyShardType));
    }
}

struct GalaxyShardType;

impl ShardTypePlugin for GalaxyShardType {
    fn shard_type(&self) -> u8 {
        GALAXY_SHARD_TYPE
    }
    fn name(&self) -> &'static str {
        "galaxy"
    }
    fn is_scene_context(&self) -> bool {
        true
    }
    /// Galactic-frame ≈ identity at the MVP level (scale differences
    /// handled by `StarCatalog` ingestion + far-field positioning in
    /// the render_contribution phase).
    fn to_system_space(&self, _shard: &ShardRuntime, local: DVec3) -> DVec3 {
        local
    }
    fn from_system_space(&self, _shard: &ShardRuntime, world: DVec3) -> DVec3 {
        world
    }
    fn hud_summary(&self, _ctx: &HudSummaryCtx) -> Vec<(String, String)> {
        // GalaxyWorldState fields (warp_phase, eta, origin/target stars)
        // aren't in ctx yet — lands with H4 when a GalaxyWorldStateData
        // Res is added. MVP: just indicate the warp state.
        vec![("WARP".into(), "idle".into())]
    }
}

pub const GALAXY_SHARD_TYPE: u8 = 3;
