//! Client-side config for `SubBlockType::HudPanel` tiles.
//!
//! Each HUD panel sub-block is keyed by `(shard, block_pos, face)`.
//! This resource stores the player's choices (widget kind, channel,
//! property, caption) so that `block_tile::spawn_hud_tiles_for_chunk`
//! can apply them when the tile respawns after a chunk remesh.
//!
//! **Client-side storage, not server-authoritative** — a full AAA
//! implementation would extend the server's `BlockSignalConfig` with
//! per-sub-block HUD payloads and echo them to every client. Until
//! that protocol change lands, this resource lets the player
//! experiment locally: widget kind, channel, property, caption are
//! all configurable from the tablet; the config persists until the
//! client process exits or the sub-block is removed.

use std::collections::HashMap;

use bevy::prelude::*;

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::tile::WidgetKind;
use crate::shard::ShardKey;

/// Composite key for a block-face HUD panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HudPanelKey {
    pub shard: ShardKey,
    pub block_pos: IVec3,
    pub face: u8,
}

/// Per-panel user-editable display settings.
#[derive(Debug, Clone)]
pub struct HudPanelSettings {
    pub kind: WidgetKind,
    pub channel: String,
    pub property: SignalProperty,
    pub caption: String,
    pub opacity: f32,
    /// Master AR overlay toggle. When `true`, the tile additionally
    /// paints AR markers for whichever entity kinds `ar_filter`
    /// admits. When `false`, no AR overlay.
    pub ar_enabled: bool,
    pub ar_filter: crate::hud::ar::ArFilter,
}

impl Default for HudPanelSettings {
    /// Default state for a newly-placed HUD panel: no widget content,
    /// AR overlay ON with every category enabled. Lets the pilot see
    /// celestial / ship / player / debris markers through the canopy
    /// out of the box; the player customises each panel via F → pick
    /// a widget.
    fn default() -> Self {
        Self {
            kind: WidgetKind::None,
            channel: String::new(),
            property: SignalProperty::Throttle,
            caption: String::new(),
            opacity: 0.95,
            ar_enabled: true,
            ar_filter: crate::hud::ar::ArFilter {
                celestial_bodies: true,
                remote_ships: true,
                remote_players: true,
                debris: true,
            },
        }
    }
}

#[derive(Resource, Default, Debug)]
pub struct HudPanelConfigs {
    pub by_key: HashMap<HudPanelKey, HudPanelSettings>,
}

impl HudPanelConfigs {
    pub fn get_or_default(&self, key: HudPanelKey) -> HudPanelSettings {
        self.by_key.get(&key).cloned().unwrap_or_default()
    }
    pub fn set(&mut self, key: HudPanelKey, settings: HudPanelSettings) {
        self.by_key.insert(key, settings);
    }
}

/// When the tablet is open in "HUD panel config" mode (F on a HUD
/// panel sub-block), this resource carries the panel being edited.
/// `None` = tablet is in block-config mode (the default).
#[derive(Resource, Default, Debug)]
pub struct OpenHudPanelConfig {
    pub editing: Option<HudPanelEditState>,
}

#[derive(Debug, Clone)]
pub struct HudPanelEditState {
    pub key: HudPanelKey,
    pub settings: HudPanelSettings,
}

pub struct HudPanelConfigPlugin;

impl Plugin for HudPanelConfigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HudPanelConfigs>()
            .init_resource::<OpenHudPanelConfig>()
            .add_systems(Update, apply_panel_config_to_tiles);
    }
}

/// When `HudPanelConfigs` is updated (user clicks Apply in the HUD
/// panel editor), mutate the live `HudConfig` on any `HudTile` whose
/// attachment matches. No respawn needed — the CPU redraw pipeline
/// picks up the new widget / channel / caption on the next frame.
fn apply_panel_config_to_tiles(
    configs: Res<HudPanelConfigs>,
    mut tiles: Query<(&crate::hud::tile::HudTile, &mut crate::hud::tile::HudConfig)>,
) {
    if !configs.is_changed() {
        return;
    }
    for (tile, mut hud_config) in &mut tiles {
        let crate::hud::tile::HudAttachment::Block {
            shard,
            block_pos,
            face,
        } = tile.attachment
        else {
            continue;
        };
        let key = HudPanelKey {
            shard,
            block_pos,
            face,
        };
        let Some(settings) = configs.by_key.get(&key) else {
            continue;
        };
        hud_config.kind = settings.kind;
        hud_config.channel = settings.channel.clone();
        hud_config.property = settings.property;
        hud_config.caption = settings.caption.clone();
        hud_config.opacity = settings.opacity;
        hud_config.ar_enabled = settings.ar_enabled;
        hud_config.ar_filter = settings.ar_filter;
    }
}
