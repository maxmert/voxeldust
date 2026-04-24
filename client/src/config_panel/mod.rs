//! F-key config panel for functional blocks.
//!
//! Flow: F on a functional block sends `ClientMsg::BlockEditRequest(INTERACT)`.
//! The server responds with `ServerMsg::BlockConfigState` carrying the
//! block's complete signal config. This module ingests that response,
//! opens a centered egui panel displaying publisher/subscriber bindings
//! + converter rules + kind-specific sub-sections, and on Apply sends
//! back `ClientMsg::BlockConfigUpdate` via TCP.
//!
//! **Cross-shard**: the hit's `ShardKey` is remembered so updates route
//! to the owning shard's TCP (Design Principle #10). Opening a config
//! panel on a ship block while standing on the planet works
//! identically to opening one inside the ship.

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

use voxeldust_core::client_message::ClientMsg;
use voxeldust_core::signal::config::{
    BlockConfigUpdateData, BlockSignalConfig, PublishBindingConfig,
    SubscribeBindingConfig,
};
use voxeldust_core::signal::types::SignalProperty;
use voxeldust_core::wire_codec;

use crate::hud::{tablet::DespawnHeldTablet, SpawnHeldTablet};
use crate::net::{GameEvent, NetEvent, TcpSender};
use crate::shard::ShardKey;

pub struct ConfigPanelPlugin;

impl Plugin for ConfigPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OpenConfigPanel>()
            .add_systems(Update, ingest_block_config_state)
            .add_systems(Update, open_egui_editor_on_focus)
            .add_systems(Update, cursor_ungrab_while_open)
            .add_systems(Update, clear_editable_when_tablet_gone)
            .add_systems(EguiPrimaryContextPass, render_config_panel);
    }
}

/// Clear `editable` when the tablet entity no longer exists so the
/// next F press doesn't flash the PREVIOUS block's config before the
/// server's fresh `BlockConfigState` arrives.
fn clear_editable_when_tablet_gone(
    tablet: Query<(), With<crate::hud::HeldTablet>>,
    mut panel: ResMut<OpenConfigPanel>,
) {
    if tablet.iter().next().is_none() && panel.editable.is_some() {
        panel.editable = None;
    }
}

/// Deprecated — the tablet now renders the interactive editor
/// directly via bevy_egui's render-to-image path (see
/// `hud::tablet_ui`). Kept for API compatibility; the floating
/// egui window only surfaces when something explicitly sets
/// `OpenConfigPanel.state` (no longer auto-done from Tab).
fn open_egui_editor_on_focus(
    _focus: Res<crate::hud::HudFocusState>,
    _panel: ResMut<OpenConfigPanel>,
) {
    // Intentional no-op. The in-world tablet IS the editor now; Tab
    // activates the on-tablet cursor via `hud::focus`. The floating
    // egui window in this module stays available as a debug-only
    // surface.
}

/// Editor state. Two fields:
///
/// * `state` = the **visible** editor. When `Some(_)`, the egui panel
///   renders this frame.
/// * `editable` = a **hidden** editable buffer populated by the
///   latest `BlockConfigState` from the server. `Tab` on the tablet
///   copies this into `state` to make the editor visible.
///
/// Keeping the buffer separate means the tablet always reflects
/// server state, while Tab-driven editing layers the egui form on top
/// without a round-trip.
#[derive(Resource, Default)]
pub struct OpenConfigPanel {
    pub state: Option<ConfigPanelState>,
    pub editable: Option<ConfigPanelState>,
}

impl OpenConfigPanel {
    pub fn is_open(&self) -> bool {
        self.state.is_some()
    }
}

/// Editable copy of a `BlockSignalConfig` plus routing info.
pub struct ConfigPanelState {
    /// Shard that owns the block — `BlockConfigUpdate` must route to
    /// this shard's TCP (for cross-shard edits).
    pub shard: ShardKey,
    /// Local editable copy. Apply serializes this into
    /// `BlockConfigUpdateData` and ships it.
    pub config: BlockSignalConfig,
    /// Pending close — set when user clicks Apply / Close; cleared at
    /// end of frame so the close doesn't conflict with egui event
    /// handling in the same frame.
    pub pending_close: bool,
}

/// Drain `NetEvent::BlockConfigState`: update the editable buffer so
/// the tablet's on-device egui editor picks it up on the next paint
/// pass. Only spawns a NEW tablet entity if there isn't one already —
/// respawning on every server response would churn the egui context
/// entity and leak stale `EguiInputEvent`s (bevy_egui logs
/// `NotSpawned` errors when queued events target a dead context).
fn ingest_block_config_state(
    mut events: MessageReader<GameEvent>,
    mut spawn_tablet: MessageWriter<SpawnHeldTablet>,
    pending_shard: Res<PendingConfigShard>,
    mut panel: ResMut<OpenConfigPanel>,
    existing_tablet: Query<(), With<crate::hud::HeldTablet>>,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::BlockConfigState(cfg) = ev {
            let shard = pending_shard.0.unwrap_or(ShardKey::new(0, 0));

            // Editable buffer updates unconditionally — the tablet's
            // `paint_tablet_ui` reads this on every paint pass, so the
            // on-screen editor will reflect the freshest config
            // without needing a tablet respawn.
            panel.editable = Some(ConfigPanelState {
                shard,
                config: cfg.clone(),
                pending_close: false,
            });

            // Only spawn a new tablet entity when there's none live.
            // Otherwise the existing tablet stays in place and picks
            // up the new editable buffer next paint.
            if existing_tablet.iter().next().is_none() {
                spawn_tablet.write(SpawnHeldTablet {
                    shard,
                    config: cfg.clone(),
                });
            }
            tracing::info!(
                block = ?(cfg.block_pos.x, cfg.block_pos.y, cfg.block_pos.z),
                kind = cfg.kind,
                publish_bindings = cfg.publish_bindings.len(),
                subscribe_bindings = cfg.subscribe_bindings.len(),
                "block config received — editable buffer updated",
            );
        }
    }
}

/// Shard to credit the next incoming `BlockConfigState` to. Set at the
/// moment F triggers the request; cleared once the panel opens.
#[derive(Resource, Default)]
pub struct PendingConfigShard(pub Option<ShardKey>);

/// Ungrab the cursor while the panel is open so the user can click
/// widgets; restore grab on close. Legacy `InputMode::UiPanel` state
/// machine will supersede this in a later iteration.
fn cursor_ungrab_while_open(
    panel: Res<OpenConfigPanel>,
    mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut was_open: Local<bool>,
) {
    let is_open = panel.is_open();
    if is_open == *was_open {
        return;
    }
    *was_open = is_open;
    if let Ok(mut c) = cursors.single_mut() {
        if is_open {
            c.grab_mode = CursorGrabMode::None;
            c.visible = true;
        } else {
            c.grab_mode = CursorGrabMode::Locked;
            c.visible = false;
        }
    }
}

fn render_config_panel(
    mut contexts: EguiContexts,
    mut panel: ResMut<OpenConfigPanel>,
    tcp: Res<TcpSender>,
    mut despawn_tablet: MessageWriter<DespawnHeldTablet>,
) -> Result {
    let Some(state) = panel.state.as_mut() else {
        return Ok(());
    };
    let ctx = contexts.ctx_mut()?;

    let mut close = false;
    let mut apply_and_close = false;
    egui::Window::new("Block Config")
        .default_pos([400.0, 200.0])
        .default_width(500.0)
        .collapsible(false)
        .show(ctx, |ui| {
            // Header.
            ui.label(format!(
                "Block {} at ({}, {}, {})",
                state.config.block_type,
                state.config.block_pos.x,
                state.config.block_pos.y,
                state.config.block_pos.z,
            ));
            ui.label(format!("Shard: {}", state.shard));
            ui.label(format!(
                "Available channels: {}",
                state.config.available_channels.len()
            ));
            ui.separator();

            // Publish bindings editor.
            ui.heading("Publisher bindings");
            let mut remove_publish: Option<usize> = None;
            for (i, b) in state.config.publish_bindings.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}.", i));
                    channel_combo(
                        ui,
                        &format!("pub-{}", i),
                        &mut b.channel_name,
                        &state.config.available_channels,
                    );
                    property_combo(ui, &format!("pub-prop-{}", i), &mut b.property);
                    if ui.button("✕").clicked() {
                        remove_publish = Some(i);
                    }
                });
            }
            if let Some(i) = remove_publish {
                state.config.publish_bindings.remove(i);
            }
            if ui.button("+ add publisher").clicked() {
                state.config.publish_bindings.push(PublishBindingConfig {
                    channel_name: String::new(),
                    property: SignalProperty::Throttle,
                });
            }
            ui.separator();

            // Subscribe bindings editor.
            ui.heading("Subscriber bindings");
            let mut remove_subscribe: Option<usize> = None;
            for (i, b) in state.config.subscribe_bindings.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}.", i));
                    channel_combo(
                        ui,
                        &format!("sub-{}", i),
                        &mut b.channel_name,
                        &state.config.available_channels,
                    );
                    property_combo(ui, &format!("sub-prop-{}", i), &mut b.property);
                    if ui.button("✕").clicked() {
                        remove_subscribe = Some(i);
                    }
                });
            }
            if let Some(i) = remove_subscribe {
                state.config.subscribe_bindings.remove(i);
            }
            if ui.button("+ add subscriber").clicked() {
                state.config.subscribe_bindings.push(SubscribeBindingConfig {
                    channel_name: String::new(),
                    property: SignalProperty::Throttle,
                });
            }
            ui.separator();

            // Converter rules (only meaningful for SignalConverter kind —
            // we display them for any kind that has any; server ignores
            // for non-converter blocks).
            if !state.config.converter_rules.is_empty() {
                ui.heading("Signal converter rules");
                for (i, r) in state.config.converter_rules.iter().enumerate() {
                    ui.label(format!(
                        "{}: `{}` → `{}` (condition + expression are readonly MVP)",
                        i, r.input_channel, r.output_channel,
                    ));
                }
                ui.separator();
            }

            // Kind-specific sections (readonly MVP).
            //
            // Full editing support per-kind is deferred; showing the
            // fields confirms the server config flowed over TCP and
            // makes the panel immediately useful for debugging.
            if let Some(fc) = &state.config.flight_computer {
                ui.heading("Flight computer");
                ui.label(format!(
                    "damping: {:.2}  dead_zone: {:.2}  max_correction: {:.2}",
                    fc.damping_gain, fc.dead_zone, fc.max_correction,
                ));
            }
            if state.config.seat_mappings.len() > 0 {
                ui.heading("Seat bindings");
                for (i, m) in state.config.seat_mappings.iter().enumerate() {
                    ui.label(format!(
                        "{}: `{}` key={} → `{}` (property={:?})",
                        i, m.label, m.key_name, m.channel_name, m.property,
                    ));
                }
            }
            if let Some(ps) = &state.config.power_source {
                ui.heading("Reactor circuits");
                for c in &ps.circuits {
                    ui.label(format!("  `{}`: {:.1}%", c.name, c.fraction * 100.0));
                }
            }
            if let Some(pc) = &state.config.power_consumer {
                ui.heading("Power consumer");
                ui.label(format!(
                    "  reactor_pos: {:?}, circuit: `{}`",
                    pc.reactor_pos, pc.circuit,
                ));
            }
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Apply").clicked() {
                    apply_and_close = true;
                }
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
        });

    if apply_and_close {
        let update = BlockConfigUpdateData {
            block_pos: state.config.block_pos,
            publish_bindings: state.config.publish_bindings.clone(),
            subscribe_bindings: state.config.subscribe_bindings.clone(),
            converter_rules: state.config.converter_rules.clone(),
            seat_mappings: state.config.seat_mappings.clone(),
            seated_channel_name: state.config.seated_channel_name.clone(),
            power_source: state.config.power_source.clone(),
            power_consumer: state.config.power_consumer.clone(),
            flight_computer: state.config.flight_computer.clone(),
            hover_module: state.config.hover_module.clone(),
            autopilot: state.config.autopilot.clone(),
            warp_computer: state.config.warp_computer.clone(),
            engine_controller: state.config.engine_controller.clone(),
            mechanical: state.config.mechanical.clone(),
        };
        let msg = ClientMsg::BlockConfigUpdate(update);
        let data = msg.serialize();
        let mut pkt = Vec::new();
        wire_codec::encode(&data, &mut pkt);
        if tcp.tx.send(pkt).is_err() {
            tracing::warn!("TCP channel closed while sending BlockConfigUpdate");
        }
        close = true;
    }
    if close {
        panel.state = None;
        // Don't despawn the tablet: the player explicitly dismissed
        // the egui editor, not the tablet itself. The tablet stays
        // visible with the (just-applied or unchanged) config. The
        // player toggles the tablet via F.
        let _ = despawn_tablet;
    }

    Ok(())
}

fn channel_combo(
    ui: &mut egui::Ui,
    id: &str,
    current: &mut String,
    available: &[String],
) {
    let label: String = if current.is_empty() {
        "(select)".to_string()
    } else {
        current.clone()
    };
    egui::ComboBox::from_id_salt(id)
        .selected_text(label)
        .show_ui(ui, |ui| {
            for name in available {
                ui.selectable_value(current, name.clone(), name);
            }
            ui.separator();
            ui.label("custom:");
            ui.text_edit_singleline(current);
        });
}

fn property_combo(ui: &mut egui::Ui, id: &str, current: &mut SignalProperty) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(format!("{:?}", current))
        .show_ui(ui, |ui| {
            for prop in [
                SignalProperty::Active,
                SignalProperty::Throttle,
                SignalProperty::Angle,
                SignalProperty::Extension,
                SignalProperty::Pressure,
                SignalProperty::Speed,
                SignalProperty::Level,
                SignalProperty::SwitchState,
                SignalProperty::Boost,
                SignalProperty::Status,
            ] {
                ui.selectable_value(current, prop, format!("{:?}", prop));
            }
        });
}
