//! In-world signal-driven HUD sub-blocks.
//!
//! Every HUD element is a world-space tile (a transparent quad parented
//! under a chunk mesh, or positioned camera-relative as a "held tablet")
//! that renders widgets painted onto a per-tile `Image` texture. Content
//! comes from the signal graph — a tile subscribes to a channel +
//! property and displays its current value through its configured widget
//! kind. AR mode additionally overlays entity markers projected through
//! the tile's plane.
//!
//! Everything is occluded by blocks (world-space mesh), transparent
//! where the widget doesn't draw, and input happens through the
//! in-game cursor during focus mode — no OS cursor / floating egui
//! windows required.

pub mod ar;
pub mod block_tile;
pub mod cockpit;
pub mod config;
pub mod focus;
pub mod font;
pub mod material;
pub mod panel_config;
pub mod publish;
pub mod signal_registry;
pub mod tablet;
pub mod tablet_ui;
pub mod texture;
pub mod tile;
pub mod widget;
pub mod widgets;

pub use ar::{ArFilter, ArMarkerPlugin};
pub use focus::{HudClickButton, HudClickEvent, HudFocusPlugin, HudFocusSet, HudFocusState};
pub use signal_registry::{SignalRegistry, SignalRegistryPlugin};
pub use tablet::{HeldTablet, HeldTabletPlugin, SpawnHeldTablet};
pub use texture::{HudTexturePlugin, HudTextureSet};
pub use tile::{HudTile, HudConfig, HudTexture, HudTilePlugin, WidgetKind};
pub use widget::{ClickAction, DrawCtx, HudWidget, HudWidgetRegistry};

use bevy::prelude::*;

/// Top-level HUD plugin — registers every sub-plugin in the right
/// order.
pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SignalRegistryPlugin)
            .add_plugins(HudTilePlugin)
            .add_plugins(HudTexturePlugin)
            .add_plugins(ArMarkerPlugin)
            .add_plugins(HudFocusPlugin)
            .add_plugins(HeldTabletPlugin)
            .add_plugins(tablet_ui::TabletUiPlugin)
            .add_plugins(panel_config::HudPanelConfigPlugin)
            .add_plugins(publish::SignalPublishPlugin)
            .add_plugins(cockpit::CockpitHudPlugin)
            .add_systems(Update, dispatch_hud_clicks);
        // Register built-in widget kinds. Future widgets are added
        // with one `register(...)` line each.
        widgets::register_builtins(app);
    }
}

/// Route `HudClickEvent` to the clicked tile's widget via
/// `HudWidget::on_click`. Emitted `ClickAction::Publish` translates
/// to a `PublishSignalEvent`, which the `SignalPublishPlugin` forwards
/// to the server.
fn dispatch_hud_clicks(
    mut clicks: MessageReader<HudClickEvent>,
    mut publishes: MessageWriter<publish::PublishSignalEvent>,
    registry: Res<HudWidgetRegistry>,
    signals: Res<SignalRegistry>,
    tiles: Query<(&HudTile, &HudConfig)>,
) {
    for click in clicks.read() {
        let Ok((_tile, config)) = tiles.get(click.tile) else { continue };
        let Some(widget) = registry.get(config.kind) else { continue };
        let value = signals.get(&config.channel).map(|r| r.value.clone());
        if let Some(action) = widget.on_click(click.uv, click.button, value.as_ref(), config) {
            match action {
                ClickAction::Publish { channel, value } => {
                    publishes.write(publish::PublishSignalEvent { channel, value });
                }
            }
        }
    }
}
