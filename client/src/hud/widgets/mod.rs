//! Built-in HUD widgets. Adding a new widget kind = new file here +
//! one `register(...)` line in `register_builtins`.

pub mod button;
pub mod config_panel;
pub mod gauge;
pub mod numeric;
pub mod text;
pub mod toggle;

use bevy::prelude::*;

use crate::hud::widget::HudWidgetRegistry;

pub fn register_builtins(app: &mut App) {
    // `HudWidgetRegistry` is initialised by `HudTexturePlugin`; we
    // grab the ResMut directly so built-in registration happens
    // before any tile tries to render.
    app.add_systems(Startup, register_all);
}

fn register_all(mut registry: ResMut<HudWidgetRegistry>) {
    registry.register(Box::new(gauge::GaugeWidget));
    registry.register(Box::new(numeric::NumericWidget));
    registry.register(Box::new(toggle::ToggleWidget));
    registry.register(Box::new(text::TextWidget));
    registry.register(Box::new(config_panel::ConfigPanelWidget));
    registry.register(Box::new(button::ButtonWidget));
    tracing::info!(count = 6, "hud widgets registered");
}
