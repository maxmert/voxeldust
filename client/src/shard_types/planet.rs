//! PLANET shard-type plugin (wire shard_type = 0).
//!
//! Phase 8 MVP: registers the trait impl. Tangent-frame camera +
//! CPU sphere-projection vertex transform land in later phases
//! (Phase 11 camera, Phase 5 vertex hook once the trait method lands).

use bevy::prelude::*;

use crate::shard::{plugin::HudSummaryCtx, ShardTypePlugin, ShardTypeRegistry};

pub struct PlanetShardPlugin;

impl Plugin for PlanetShardPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut()
            .resource_mut::<ShardTypeRegistry>()
            .register(Box::new(PlanetShardType));
    }
}

struct PlanetShardType;

impl ShardTypePlugin for PlanetShardType {
    fn shard_type(&self) -> u8 {
        PLANET_SHARD_TYPE
    }
    fn name(&self) -> &'static str {
        "planet"
    }
    fn is_scene_context(&self) -> bool {
        false
    }
    // `to_system_space`: trait default is `origin + rotation * local`,
    // correct for PLANET when `origin` holds the planet's system-space
    // center and `rotation` holds its orientation. Server provides both
    // via `SecondaryConnected { reference_position, reference_rotation }`.

    fn hud_summary(&self, ctx: &HudSummaryCtx) -> Vec<(String, String)> {
        // Altitude = distance from planet center (shard origin) minus
        // planet radius. The server's SecondaryConnected passes the
        // planet's center as `reference_position`; we approximate the
        // radius as `shard.origin.length() - player altitude` isn't
        // reliable without the authoritative radius, so we just report
        // the raw distance from the planet center for MVP. True radius
        // lands with the server's BlockSignalConfig `body_radius` field.
        let distance = (ctx.camera_world - ctx.shard_origin).length();
        vec![
            ("PLANET".into(), "primary".into()),
            ("CAM FROM CENTER".into(), format!("{:.0} m", distance)),
        ]
    }
}

pub const PLANET_SHARD_TYPE: u8 = 0;
