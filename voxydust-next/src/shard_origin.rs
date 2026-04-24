//! `ShardOrigin` — the primary WorldState's system-absolute origin.
//!
//! On every shard type, `WorldState.origin: DVec3` is the world-space
//! anchor point that all entity / chunk positions in that WorldState are
//! expressed relative to. For a Ship-shard primary, this is the ship's
//! current system-absolute position. For a Planet-shard primary, it's the
//! player's floating-origin anchor on the planet. Etc.
//!
//! Celestial bodies streamed from the System shard secondary come in
//! system-absolute coordinates (billions of metres from the star). To
//! render them around the player (who is at Bevy origin), we subtract the
//! primary shard origin from each body's system-absolute position.
//!
//! A separate resource so any system that needs the shift (celestial
//! bodies, future long-range-entity renderers, etc.) can read it without
//! threading it through the WorldState event bus.

use bevy::prelude::*;
use glam::DVec3;

use crate::net_plugin::GameEvent;
use crate::network::NetEvent;

pub struct ShardOriginPlugin;

impl Plugin for ShardOriginPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShardOrigin>()
            .add_systems(Update, update_origin);
    }
}

/// The primary shard's WorldState origin. Updated from every primary
/// `NetEvent::WorldState` arrival. Systems that render in ship-relative
/// coordinates use this to shift system-absolute data (celestial bodies,
/// long-range LOD ships) into the local frame.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct ShardOrigin {
    pub origin: DVec3,
}

fn update_origin(mut origin: ResMut<ShardOrigin>, mut events: MessageReader<GameEvent>) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::WorldState(ws) = ev {
            origin.origin = ws.origin;
        }
    }
}
