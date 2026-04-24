//! SHIP shard-type plugin (wire shard_type = 2).
//!
//! Phase 6 MVP: registers the trait impl + Bevy plugin so the
//! registry stops warning when it receives SHIP shards.
//!
//! Coordinate mapping uses the trait defaults
//! (`origin + rotation * local` / inverse), which match ship-frame
//! semantics: `origin` is the ship's system-space position,
//! `rotation` is its world rotation. The defaults remain correct for
//! SHIP, so no override needed.
//!
//! Future phases add:
//! - Phase 11: `primary_camera_frame` (ship_rotation + eye_up, pilot lock).
//! - Phase 14: `raycast_contribution` (root grid + sub-grids).
//! - Phase 15: `interaction_schema` (seat-enter / break / place / config).
//! - Phase 10: `depart` / `arrive` capturing ship velocity for grace
//!   extrapolation.
//! - Phase 16: sub-grid transform systems via `render_contribution`.

use bevy::prelude::*;
use glam::DVec3;

use voxeldust_core::client_message::EntityKind;

use crate::shard::{plugin::HudSummaryCtx, ShardTypePlugin, ShardTypeRegistry};

/// Ship shard-type plugin. Registered as a Bevy plugin; wraps a trait
/// object into `ShardTypeRegistry`.
pub struct ShipShardPlugin;

impl Plugin for ShipShardPlugin {
    fn build(&self, app: &mut App) {
        app.world_mut()
            .resource_mut::<ShardTypeRegistry>()
            .register(Box::new(ShipShardType));
    }
}

struct ShipShardType;

impl ShardTypePlugin for ShipShardType {
    fn shard_type(&self) -> u8 {
        SHIP_SHARD_TYPE
    }
    fn name(&self) -> &'static str {
        "ship"
    }
    fn is_scene_context(&self) -> bool {
        false
    }
    // `to_system_space` / `from_system_space` use the trait defaults,
    // which evaluate `origin + rotation * local` / its inverse. That
    // is the correct mapping for ship-local → system-space.

    fn hud_summary(&self, ctx: &HudSummaryCtx) -> Vec<(String, String)> {
        let mut out = Vec::with_capacity(4);
        let ws = match ctx.primary_ws {
            Some(w) => w,
            None => return out,
        };
        if let Some(own_ship) = ws
            .entities
            .iter()
            .find(|e| e.is_own && e.kind == EntityKind::Ship)
        {
            let vel = DVec3::new(
                own_ship.velocity.x,
                own_ship.velocity.y,
                own_ship.velocity.z,
            );
            let speed = vel.length();
            out.push(("SPEED".into(), format!("{:.1} m/s", speed)));
            // Ship world position relative to system-space origin.
            let pos = DVec3::new(
                own_ship.position.x,
                own_ship.position.y,
                own_ship.position.z,
            );
            out.push((
                "POS".into(),
                format!("{:+.0} {:+.0} {:+.0}", pos.x, pos.y, pos.z),
            ));
        } else {
            out.push(("SHIP".into(), "no own-ship in ws".into()));
        }
        out.push(("PILOT SEAT".into(), "F to open panel".into()));
        out
    }
}

/// Wire-level u8 for SHIP shards.
pub const SHIP_SHARD_TYPE: u8 = 2;
