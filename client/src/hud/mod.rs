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
pub mod camera_focus;
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
pub mod terminal_bridge;
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
pub use widget::{
    ClickAction, DrawCtx, HudTextInput, HudWidget, HudWidgetRegistry, HudWidgetState,
    HudWidgetStateData, WidgetAction,
};

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
            .add_plugins(camera_focus::HudCameraFocusPlugin)
            .add_plugins(terminal_bridge::TerminalBridgePlugin)
            .add_systems(Update, (dispatch_hud_clicks, dispatch_hud_text_input));
        // Register built-in widget kinds. Future widgets are added
        // with one `register(...)` line each.
        widgets::register_builtins(app);
    }
}

/// Route `HudClickEvent` to the clicked tile's widget via
/// `HudWidget::on_click`. The widget's returned [`WidgetAction`] is
/// dispatched into a concrete effect (publish a signal, send a
/// terminal chat line, …). Stateful widgets receive their per-tile
/// [`HudWidgetState`] borrowed mut here so a click can mutate it
/// (e.g. caret movement, selection).
fn dispatch_hud_clicks(
    mut clicks: MessageReader<HudClickEvent>,
    mut publishes: MessageWriter<publish::PublishSignalEvent>,
    registry: Res<HudWidgetRegistry>,
    signals: Res<SignalRegistry>,
    tcp: Res<crate::net::TcpSender>,
    mut tiles: Query<(&HudTile, &HudConfig, Option<&mut HudWidgetState>)>,
) {
    for click in clicks.read() {
        let Ok((_tile, config, mut state)) = tiles.get_mut(click.tile) else { continue };
        let Some(widget) = registry.get(config.kind) else { continue };
        let value = signals.get(&config.channel).map(|r| r.value.clone());
        let state_ref: Option<&mut dyn HudWidgetStateData> =
            state.as_deref_mut().map(|s| s.data.as_mut());
        let Some(action) = widget.on_click(click.uv, click.button, value.as_ref(), state_ref, config) else {
            continue;
        };
        apply_widget_action(action, &mut publishes, &tcp);
    }
}

/// Drain `HudTextInputEvent` and route to the focused tile's widget
/// via `HudWidget::on_text`. Symmetric to `dispatch_hud_clicks` —
/// any future text-capturing widget (multiline editor, console,
/// IRC client, password field, …) just implements `on_text` and
/// participates in the same dispatch.
fn dispatch_hud_text_input(
    mut events: MessageReader<focus::HudTextInputEvent>,
    mut publishes: MessageWriter<publish::PublishSignalEvent>,
    mut focus: ResMut<HudFocusState>,
    registry: Res<HudWidgetRegistry>,
    tcp: Res<crate::net::TcpSender>,
    mut tiles: Query<(&HudTile, &HudConfig, Option<&mut HudWidgetState>)>,
) {
    for ev in events.read() {
        let Ok((_tile, config, mut state)) = tiles.get_mut(ev.tile) else {
            tracing::info!(
                tile = ?ev.tile,
                input = ?ev.input,
                "dispatch_hud_text_input: tile entity gone"
            );
            continue;
        };
        let Some(widget) = registry.get(config.kind) else {
            tracing::info!(
                kind = ?config.kind,
                input = ?ev.input,
                "dispatch_hud_text_input: no widget registered for kind"
            );
            continue;
        };
        let has_state = state.is_some();
        let state_ref: Option<&mut dyn HudWidgetStateData> =
            state.as_deref_mut().map(|s| s.data.as_mut());
        let action = widget.on_text(ev.input.clone(), state_ref, config);
        tracing::info!(
            kind = ?config.kind,
            input = ?ev.input,
            has_state,
            action = ?action,
            "dispatch_hud_text_input"
        );
        let consumed = matches!(action, Some(WidgetAction::Consumed));
        if let Some(action) = action {
            apply_widget_action(action, &mut publishes, &tcp);
        }
        // Default Esc → drop focus, unless the widget consumed it.
        if !consumed && matches!(ev.input, HudTextInput::Escape) {
            focus.active = false;
        }
    }
}

/// Translate a `WidgetAction` into a concrete client-side effect.
/// Centralised so any source of widget actions (clicks today, text
/// inputs now, future programmatic invocations) goes through one
/// path.
fn apply_widget_action(
    action: WidgetAction,
    publishes: &mut MessageWriter<publish::PublishSignalEvent>,
    tcp: &crate::net::TcpSender,
) {
    match action {
        WidgetAction::Publish { channel, value } => {
            publishes.write(publish::PublishSignalEvent { channel, value });
        }
        WidgetAction::SendChat { block_pos, text } => {
            send_terminal_chat(tcp, block_pos, text);
        }
        WidgetAction::Consumed => {}
    }
}

/// Pack and queue a `ClientMsg::TerminalChatSend`. Lives here (the
/// HUD-action dispatch site) rather than inside any specific widget
/// so future widget kinds can emit `WidgetAction::SendChat` with the
/// same plumbing. The widget-side `block_pos` is in Bevy-imported
/// `IVec3` (glam 0.30); the wire format pins glam 0.29 — convert at
/// the boundary.
fn send_terminal_chat(tcp: &crate::net::TcpSender, block_pos: IVec3, text: String) {
    use voxeldust_core::client_message::{ClientMsg, TerminalChatSendData};
    use voxeldust_core::wire_codec;
    let wire_pos = glam::IVec3::new(block_pos.x, block_pos.y, block_pos.z);
    let msg = ClientMsg::TerminalChatSend(TerminalChatSendData {
        block_pos: wire_pos,
        text,
    });
    let data = msg.serialize();
    let mut pkt = Vec::new();
    wire_codec::encode(&data, &mut pkt);
    if tcp.tx.send(pkt).is_err() {
        tracing::warn!("TCP channel closed while sending TerminalChatSend");
    }
}
