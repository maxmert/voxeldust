//! Tablet egui UI — renders the interactive config editor directly
//! onto the held tablet's texture via bevy_egui 0.39's render-to-image
//! support. The in-game cursor (driven by `HudFocusState.cursor_uv`)
//! is mapped to egui pointer events so the OS cursor never ungrabs.
//!
//! Pipeline:
//!   * `TabletPaintPass` is a `ScheduleLabel`. A dedicated
//!     `Camera3d` with `RenderTarget::Image(tablet_texture)` and
//!     `EguiMultipassSchedule::new(TabletPaintPass)` is spawned
//!     alongside the tablet quad.
//!   * `paint_tablet_ui` runs in `TabletPaintPass` — it queries for
//!     the tablet's `EguiContext` and draws the config editor there.
//!   * `drive_tablet_pointer` runs in `Update`
//!     (`EguiInputSet::InitReading`) and feeds `EguiInputEvent`s
//!     derived from the focus-cursor into the tablet context.
//!
//! All OS cursor state stays locked — the player sees their in-world
//! tile cursor painted by `draw_cursor`, egui reacts to that virtual
//! pointer, and nothing about the normal game input loop changes.

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, RenderTarget};
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use bevy_egui::input::{
    EguiContextPointerPosition, EguiInputEvent, FocusedNonWindowEguiContext,
    HoveredNonWindowEguiContext,
};
use bevy_egui::{egui, EguiContext, EguiInputSet, EguiMultipassSchedule, PrimaryEguiContext};

use voxeldust_core::block::block_id::BlockId;
use voxeldust_core::client_message::ClientMsg;
use voxeldust_core::signal::components::{AxisDirection, KeyMode, SeatInputSource};
use voxeldust_core::signal::config::{
    BlockConfigUpdateData, BlockSignalConfig, PublishBindingConfig, SeatInputBindingConfig,
    SignalRuleConfig, SubscribeBindingConfig,
};
use voxeldust_core::signal::converter::{SignalCondition, SignalExpression};
use voxeldust_core::signal::types::SignalProperty;
use voxeldust_core::wire_codec;

use crate::config_panel::OpenConfigPanel;
use crate::hud::focus::{HudClickButton, HudClickEvent, HudFocusState};
use crate::hud::panel_config::{HudPanelConfigs, HudPanelSettings, OpenHudPanelConfig};
use crate::hud::tablet::{DespawnHeldTablet, HeldTablet};
use crate::hud::tile::WidgetKind;
use crate::net::TcpSender;

/// Dedicated schedule that paints egui into the tablet texture. Each
/// frame runs exactly once after the tablet camera begins its render
/// pass.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TabletPaintPass;

/// Marker component on the render-to-image camera driving the tablet
/// egui context. Used by `drive_tablet_pointer` to locate the context
/// entity for pointer routing.
#[derive(Component, Debug, Clone, Copy)]
pub struct TabletEguiContextMarker;

/// Size of the render target. Matches `TABLET_RES` in `tablet.rs`.
/// Kept here as a re-used constant so the pointer-driver can convert
/// tile-UV ∈ [0, 1]² to pixel coords.
pub const TABLET_UI_RES: u32 = 512;

pub struct TabletUiPlugin;

impl Plugin for TabletUiPlugin {
    fn build(&self, app: &mut App) {
        // Paint pass: runs inside the tablet camera's render — this is
        // the only place we may touch the tablet's `EguiContext`.
        app.add_systems(TabletPaintPass, paint_tablet_ui);

        // Pointer driver: translate HUD focus state → egui pointer
        // events. Must run in `EguiInputSet::InitReading` so bevy_egui
        // picks the events up this frame.
        app.add_systems(
            Update,
            drive_tablet_pointer.in_set(EguiInputSet::InitReading),
        );
    }
}

/// Spawn the egui render-to-image camera paired with the tablet
/// texture. Called from `spawn_tablet` right after creating the
/// tablet Image.
///
/// Returns the camera entity (the one carrying the `EguiContext`).
/// The pointer driver maintains a `Local` cache of this entity.
pub fn spawn_tablet_egui_camera(
    commands: &mut Commands,
    image: Handle<Image>,
    owner: Entity,
) -> Entity {
    let image_for_target = image.clone();
    let _ = image;
    commands
        .spawn((
            Name::new("tablet_egui_camera"),
            Camera3d::default(),
            Camera {
                // `order = -1` so it runs before the main camera,
                // populating the tablet texture with fresh content in
                // time for the main 3D pass to sample it on the tablet
                // quad.
                order: -1,
                clear_color: ClearColorConfig::Custom(Color::srgba(
                    10.0 / 255.0,
                    16.0 / 255.0,
                    26.0 / 255.0,
                    1.0,
                )),
                ..default()
            },
            // `RenderTarget` is a separate component in bevy 0.18
            // (not a Camera field). Adding it redirects output to the
            // image.
            RenderTarget::Image(image_for_target.into()),
            // Render nothing from the 3D scene into this camera — its
            // output is entirely egui-sourced. `RenderLayers::none`
            // makes the camera ignore every entity.
            RenderLayers::none(),
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            EguiMultipassSchedule::new(TabletPaintPass),
            TabletEguiContextMarker,
            // Parent to the tablet so despawning the tablet cleans up
            // this camera via the usual despawn chain.
            ChildOf(owner),
        ))
        .id()
}

/// Paint pass — runs inside the tablet camera's render. Queries for
/// the (non-primary) EguiContext on the tablet camera and draws the
/// editor UI. Reading `OpenConfigPanel.editable` gives us the live
/// config copy populated by the last `BlockConfigState`.
fn paint_tablet_ui(
    mut ctx_q: Query<
        &mut EguiContext,
        (With<TabletEguiContextMarker>, Without<PrimaryEguiContext>),
    >,
    mut panel: ResMut<OpenConfigPanel>,
    mut hud_panel_edit: ResMut<OpenHudPanelConfig>,
    mut panel_configs: ResMut<HudPanelConfigs>,
    tcp: Res<TcpSender>,
    focus: Res<HudFocusState>,
    block_registry: Res<crate::chunk::stream::SharedBlockRegistry>,
    mut despawn_tablet: MessageWriter<DespawnHeldTablet>,
) {
    let Ok(mut ctx) = ctx_q.single_mut() else {
        return;
    };
    let ctx = ctx.get_mut();

    // Two modes: HUD panel config (editing a placed HudPanel sub-block)
    // vs block config (editing a functional block's signal bindings).
    // HUD-panel mode takes precedence when active.
    if hud_panel_edit.editing.is_some() {
        paint_hud_panel_editor(ctx, &mut hud_panel_edit, &mut panel_configs, &focus, &mut despawn_tablet);
        return;
    }

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(10, 16, 26)))
        .show(ctx, |ui| {
            style_cockpit(ui);

            let mut close = false;
            let mut apply = false;

            // Empty state — no config loaded yet.
            let Some(state) = panel.editable.as_mut() else {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.heading(
                        egui::RichText::new("NO BLOCK SELECTED")
                            .color(egui::Color32::from_rgb(120, 220, 255))
                            .size(18.0),
                    );
                    ui.add_space(20.0);
                    ui.label(
                        egui::RichText::new("Point at a functional block and press F to load its config.")
                            .color(egui::Color32::from_rgb(180, 200, 220))
                            .size(11.0),
                    );
                    ui.add_space(24.0);
                    ui.label(
                        egui::RichText::new("TAB: focus cursor • F: dismiss")
                            .color(egui::Color32::from_rgb(100, 120, 140))
                            .size(10.0),
                    );
                });
                return;
            };

            // Header: prefer the functional block's human-readable
            // name from the shared BlockRegistry (e.g. "Reactor
            // (Small)", "Cockpit", "Flight Computer"). Falls back to
            // generic "BLOCK" only when the registry has no entry for
            // this id (shouldn't happen for known functional blocks;
            // guard is defence-in-depth against unreleased block
            // types). Position always follows in parentheses so the
            // pilot can identify which block is being edited when
            // several of the same kind sit next to each other.
            let def = block_registry.0.get(BlockId::from_u16(state.config.block_type));
            let title_name = if def.name.is_empty() || def.name == "Undefined" {
                "BLOCK".to_string()
            } else {
                def.name.to_ascii_uppercase()
            };
            ui.horizontal(|ui| {
                ui.heading(
                    egui::RichText::new(format!(
                        "{} ({}, {}, {})",
                        title_name,
                        state.config.block_pos.x,
                        state.config.block_pos.y,
                        state.config.block_pos.z,
                    ))
                    .color(egui::Color32::from_rgb(120, 220, 255))
                    .size(14.0),
                );
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        if ui
                            .add(button_styled("CLOSE"))
                            .on_hover_text("F to dismiss tablet")
                            .clicked()
                        {
                            close = true;
                        }
                        if ui
                            .add(button_styled("APPLY"))
                            .clicked()
                        {
                            apply = true;
                        }
                    },
                );
            });
            ui.add_space(4.0);
            ui.separator();

            // Scrollable body so long configs (many bindings, seat
            // keys, converter rules) don't get clipped below the
            // tablet's bottom edge. The header + APPLY/CLOSE row
            // above this scroll area stays fixed.
            //
            // `max_height(ui.available_height())` pins the scroll
            // container to the remaining vertical space inside the
            // tablet's texture — without this the ScrollArea can
            // expand beyond the texture bounds, painting content
            // that's clipped off the bottom of the rendered image
            // instead of becoming scrollable.
            let scroll_h = ui.available_height();
            egui::ScrollArea::vertical()
                .max_height(scroll_h)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Force every child to use the full available
                    // tablet width — egui groups/rows default to
                    // hugging their content, which leaves dead space
                    // on the right. `set_min_width` pins the
                    // vertical layout's inner rect to the full
                    // content area.
                    ui.set_min_width(ui.available_width());

                    render_publisher_section(ui, &mut state.config);
                    ui.add_space(4.0);
                    render_subscriber_section(ui, &mut state.config);
                    ui.add_space(4.0);
                    render_converter_section(ui, &mut state.config);
                    ui.add_space(4.0);
                    render_seat_section(ui, &mut state.config);

                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("TAB: exit cursor • LMB: click • F: close tablet")
                            .color(egui::Color32::from_rgb(100, 120, 140))
                            .size(9.0),
                    );
                });

            // In-world cursor overlay, painted LAST so it's always
            // on top of the editor chrome. The driver system
            // (drive_tablet_pointer) keeps `focus.cursor_uv` in sync
            // with the mouse delta.
            if focus.active {
                let px = focus.cursor_uv.x * (TABLET_UI_RES as f32 - 1.0);
                let py = focus.cursor_uv.y * (TABLET_UI_RES as f32 - 1.0);
                paint_in_world_cursor(ui, egui::pos2(px, py));
            }

            if apply {
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
                    tracing::warn!(
                        "tablet apply: TCP closed while sending BlockConfigUpdate"
                    );
                } else {
                    tracing::info!(
                        block = ?(state.config.block_pos.x, state.config.block_pos.y, state.config.block_pos.z),
                        pubs = state.config.publish_bindings.len(),
                        subs = state.config.subscribe_bindings.len(),
                        "tablet apply: BlockConfigUpdate sent",
                    );
                }
            }
            if close {
                despawn_tablet.write(DespawnHeldTablet);
            }
        });
}

/// HUD-panel editor mode — edit a placed HudPanel sub-block's widget
/// kind, channel, property, caption. Client-local settings, commits
/// to `HudPanelConfigs` on Apply.
fn paint_hud_panel_editor(
    ctx: &mut egui::Context,
    hud_panel_edit: &mut OpenHudPanelConfig,
    panel_configs: &mut HudPanelConfigs,
    focus: &HudFocusState,
    despawn_tablet: &mut MessageWriter<DespawnHeldTablet>,
) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(10, 16, 26)))
        .show(ctx, |ui| {
            style_cockpit(ui);

            let Some(state) = hud_panel_edit.editing.as_mut() else {
                return;
            };

            let mut apply = false;
            let mut close = false;

            ui.horizontal(|ui| {
                ui.heading(
                    egui::RichText::new(format!(
                        "HUD PANEL ({}, {}, {}) f{}",
                        state.key.block_pos.x,
                        state.key.block_pos.y,
                        state.key.block_pos.z,
                        state.key.face,
                    ))
                    .color(egui::Color32::from_rgb(120, 220, 255))
                    .size(14.0),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(button_styled("CLOSE")).clicked() {
                        close = true;
                    }
                    if ui.add(button_styled("APPLY")).clicked() {
                        apply = true;
                    }
                });
            });
            ui.add_space(4.0);
            ui.separator();

            ui.label(
                egui::RichText::new("WIDGET").color(egui::Color32::from_rgb(180, 220, 240)),
            );
            ui.horizontal_wrapped(|ui| {
                for (kind, label) in [
                    (WidgetKind::None, "NONE"),
                    (WidgetKind::Gauge, "GAUGE"),
                    (WidgetKind::Numeric, "NUMERIC"),
                    (WidgetKind::Toggle, "TOGGLE"),
                    (WidgetKind::Text, "TEXT"),
                    (WidgetKind::Button, "BUTTON"),
                ] {
                    let selected = state.settings.kind == kind;
                    let btn = egui::Button::new(
                        egui::RichText::new(label).color(if selected {
                            egui::Color32::from_rgb(10, 16, 26)
                        } else {
                            egui::Color32::from_rgb(210, 230, 245)
                        }),
                    )
                    .fill(if selected {
                        egui::Color32::from_rgb(120, 220, 255)
                    } else {
                        egui::Color32::from_rgb(22, 34, 50)
                    });
                    if ui.add(btn).clicked() {
                        state.settings.kind = kind;
                    }
                }
            });

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("CHANNEL")
                    .color(egui::Color32::from_rgb(180, 220, 240)),
            );
            ui.label(
                egui::RichText::new("Any signal path — e.g. ship.speed, custom.my_channel").size(9.0).color(
                    egui::Color32::from_rgb(120, 140, 160),
                ),
            );
            ui.add(
                egui::TextEdit::singleline(&mut state.settings.channel)
                    .hint_text("channel name")
                    .desired_width(300.0),
            );
            // Known-channel shortcut chips — click to fill the field
            // without typing. Users can still type anything.
            ui.horizontal_wrapped(|ui| {
                for preset in [
                    "ship.speed",
                    "ship.thrust_tier",
                    "ship.autopilot.phase",
                    "ship.autopilot.eta",
                    "ship.callsign",
                    "ship.altitude",
                    "ship.heading_deg",
                ] {
                    if ui.small_button(preset).clicked() {
                        state.settings.channel = preset.to_string();
                    }
                }
            });

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("PROPERTY").color(egui::Color32::from_rgb(180, 220, 240)),
            );
            property_dropdown(ui, "hud-prop", &mut state.settings.property);

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("CAPTION").color(egui::Color32::from_rgb(180, 220, 240)),
            );
            ui.add(
                egui::TextEdit::singleline(&mut state.settings.caption)
                    .hint_text("caption (any text)")
                    .desired_width(240.0),
            );

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                egui::RichText::new("AR OVERLAY").color(egui::Color32::from_rgb(180, 220, 240)),
            );
            ui.label(
                egui::RichText::new("Project celestial bodies / ships / players / debris through this panel.")
                    .size(9.0)
                    .color(egui::Color32::from_rgb(120, 140, 160)),
            );
            if ui
                .checkbox(&mut state.settings.ar_enabled, "Enable AR markers")
                .changed()
                && state.settings.ar_enabled
                && !(state.settings.ar_filter.celestial_bodies
                    || state.settings.ar_filter.remote_ships
                    || state.settings.ar_filter.remote_players
                    || state.settings.ar_filter.debris)
            {
                // Convenience: if the player enabled AR and no
                // category is picked yet, turn them all on.
                state.settings.ar_filter.celestial_bodies = true;
                state.settings.ar_filter.remote_ships = true;
                state.settings.ar_filter.remote_players = true;
                state.settings.ar_filter.debris = true;
            }
            ui.add_enabled_ui(state.settings.ar_enabled, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(
                        &mut state.settings.ar_filter.celestial_bodies,
                        "Celestial bodies",
                    );
                    ui.checkbox(
                        &mut state.settings.ar_filter.remote_ships,
                        "Remote ships",
                    );
                    ui.checkbox(
                        &mut state.settings.ar_filter.remote_players,
                        "Remote players",
                    );
                    ui.checkbox(&mut state.settings.ar_filter.debris, "Debris");
                });
            });

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                egui::RichText::new("Apply commits to this panel only. Client-side storage for MVP.")
                    .color(egui::Color32::from_rgb(100, 120, 140))
                    .size(9.0),
            );

            // Draw cursor last.
            if focus.active {
                let px = focus.cursor_uv.x * (TABLET_UI_RES as f32 - 1.0);
                let py = focus.cursor_uv.y * (TABLET_UI_RES as f32 - 1.0);
                paint_in_world_cursor(ui, egui::pos2(px, py));
            }

            if apply {
                panel_configs.set(state.key, state.settings.clone());
                tracing::info!(
                    block = ?(state.key.block_pos.x, state.key.block_pos.y, state.key.block_pos.z),
                    face = state.key.face,
                    kind = ?state.settings.kind,
                    channel = %state.settings.channel,
                    property = ?state.settings.property,
                    "HudPanel config saved",
                );
            }
            if close {
                hud_panel_edit.editing = None;
                despawn_tablet.write(DespawnHeldTablet);
            }
        });
}

fn render_publisher_section(ui: &mut egui::Ui, cfg: &mut BlockSignalConfig) {
    ui.heading(
        egui::RichText::new("PUBLISH")
            .color(egui::Color32::from_rgb(60, 220, 120))
            .size(12.0),
    );
    let available = cfg.available_channels.clone();
    let mut remove: Option<usize> = None;
    for (i, b) in cfg.publish_bindings.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("{}.", i + 1));
            channel_text_edit(ui, &format!("pub-{}", i), &mut b.channel_name, &available);
            property_dropdown(ui, &format!("pub-prop-{}", i), &mut b.property);
            if ui.add(button_icon("×")).clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        cfg.publish_bindings.remove(i);
    }
    if ui
        .add(egui::Button::new(
            egui::RichText::new("＋ ADD PUBLISHER").color(egui::Color32::from_rgb(120, 220, 160)),
        ))
        .clicked()
    {
        cfg.publish_bindings.push(PublishBindingConfig {
            // Start empty — user types the channel name freely.
            channel_name: String::new(),
            property: SignalProperty::Throttle,
        });
    }
}

fn render_subscriber_section(ui: &mut egui::Ui, cfg: &mut BlockSignalConfig) {
    ui.heading(
        egui::RichText::new("SUBSCRIBE")
            .color(egui::Color32::from_rgb(240, 200, 60))
            .size(12.0),
    );
    let available = cfg.available_channels.clone();
    let mut remove: Option<usize> = None;
    for (i, b) in cfg.subscribe_bindings.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("{}.", i + 1));
            channel_text_edit(ui, &format!("sub-{}", i), &mut b.channel_name, &available);
            property_dropdown(ui, &format!("sub-prop-{}", i), &mut b.property);
            if ui.add(button_icon("×")).clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        cfg.subscribe_bindings.remove(i);
    }
    if ui
        .add(egui::Button::new(
            egui::RichText::new("＋ ADD SUBSCRIBER").color(egui::Color32::from_rgb(240, 210, 120)),
        ))
        .clicked()
    {
        cfg.subscribe_bindings.push(SubscribeBindingConfig {
            channel_name: String::new(),
            property: SignalProperty::Throttle,
        });
    }
}

fn render_converter_section(ui: &mut egui::Ui, cfg: &mut BlockSignalConfig) {
    ui.heading(
        egui::RichText::new("CONVERT")
            .color(egui::Color32::from_rgb(200, 100, 240))
            .size(12.0),
    );
    let available = cfg.available_channels.clone();
    let mut remove: Option<usize> = None;
    for (i, r) in cfg.converter_rules.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{}.", i + 1));
                ui.label("IN");
                channel_text_edit(ui, &format!("conv-in-{}", i), &mut r.input_channel, &available);
                if ui.add(button_icon("×")).clicked() {
                    remove = Some(i);
                }
            });
            ui.horizontal(|ui| {
                ui.label("   OUT");
                channel_text_edit(ui, &format!("conv-out-{}", i), &mut r.output_channel, &available);
            });
            ui.horizontal(|ui| {
                ui.label("   IF");
                condition_editor(ui, &format!("conv-cond-{}", i), &mut r.condition);
            });
            ui.horizontal(|ui| {
                ui.label("   THEN");
                expression_editor(ui, &format!("conv-expr-{}", i), &mut r.expression);
            });
        });
    }
    if let Some(i) = remove {
        cfg.converter_rules.remove(i);
    }
    if ui
        .add(egui::Button::new(
            egui::RichText::new("＋ ADD RULE")
                .color(egui::Color32::from_rgb(220, 160, 240)),
        ))
        .clicked()
    {
        cfg.converter_rules.push(SignalRuleConfig {
            input_channel: String::new(),
            condition: SignalCondition::Always,
            output_channel: String::new(),
            expression: SignalExpression::PassThrough,
        });
    }
}

fn render_seat_section(ui: &mut egui::Ui, cfg: &mut BlockSignalConfig) {
    ui.heading(
        egui::RichText::new("SEAT KEYS")
            .color(egui::Color32::from_rgb(240, 100, 100))
            .size(12.0),
    );

    // "SEATED" marker channel — the seat publishes 1.0 on this channel
    // while occupied; other blocks can subscribe to know if the pilot
    // is present. Optional; empty means no seated-marker publish.
    ui.horizontal(|ui| {
        ui.label("SEATED →");
        ui.add(
            egui::TextEdit::singleline(&mut cfg.seated_channel_name)
                .id_salt("seat-seated-chan")
                .hint_text("optional seated-marker channel")
                .desired_width(240.0),
        );
    });

    let available = cfg.available_channels.clone();
    let mut remove: Option<usize> = None;
    for (i, m) in cfg.seat_mappings.iter_mut().enumerate() {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{}.", i + 1));
                ui.add(
                    egui::TextEdit::singleline(&mut m.label)
                        .id_salt(format!("seat-lbl-{}", i))
                        .hint_text("label (optional)")
                        .desired_width(140.0),
                );
                source_dropdown(ui, &format!("seat-src-{}", i), &mut m.source);
                if ui.add(button_icon("×")).clicked() {
                    remove = Some(i);
                }
            });
            ui.horizontal(|ui| {
                // Key name is text: "W", "ShiftLeft", "Space", "KeyF".
                // Matches `KeyCode::<name>` on the client.
                ui.label("KEY");
                ui.add(
                    egui::TextEdit::singleline(&mut m.key_name)
                        .id_salt(format!("seat-key-{}", i))
                        .hint_text("e.g. KeyW / Space")
                        .desired_width(140.0),
                );
                key_mode_dropdown(ui, &format!("seat-mode-{}", i), &mut m.key_mode);
                axis_dir_dropdown(ui, &format!("seat-axis-{}", i), &mut m.axis_direction);
            });
            ui.horizontal(|ui| {
                ui.label("CHAN");
                channel_text_edit(
                    ui,
                    &format!("seat-chan-{}", i),
                    &mut m.channel_name,
                    &available,
                );
                property_dropdown(ui, &format!("seat-prop-{}", i), &mut m.property);
            });
        });
    }
    if let Some(i) = remove {
        cfg.seat_mappings.remove(i);
    }
    if ui
        .add(egui::Button::new(
            egui::RichText::new("＋ ADD BINDING")
                .color(egui::Color32::from_rgb(255, 160, 160)),
        ))
        .clicked()
    {
        cfg.seat_mappings.push(SeatInputBindingConfig {
            label: String::new(),
            source: SeatInputSource::Key,
            key_name: String::new(),
            key_mode: KeyMode::Momentary,
            axis_direction: AxisDirection::Positive,
            channel_name: String::new(),
            property: SignalProperty::Throttle,
        });
    }
}

fn source_dropdown(ui: &mut egui::Ui, id: &str, current: &mut SeatInputSource) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(current.label())
        .width(100.0)
        .show_ui(ui, |ui| {
            for s in [
                SeatInputSource::Key,
                SeatInputSource::MouseMoveX,
                SeatInputSource::MouseMoveY,
                SeatInputSource::ScrollWheel,
            ] {
                ui.selectable_value(current, s, s.label());
            }
        });
}

fn key_mode_dropdown(ui: &mut egui::Ui, id: &str, current: &mut KeyMode) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(current.label())
        .width(80.0)
        .show_ui(ui, |ui| {
            for m in [KeyMode::Momentary, KeyMode::Toggle] {
                ui.selectable_value(current, m, m.label());
            }
        });
}

fn axis_dir_dropdown(ui: &mut egui::Ui, id: &str, current: &mut AxisDirection) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(match current {
            AxisDirection::Positive => "Pos",
            AxisDirection::Negative => "Neg",
            AxisDirection::Both => "Both",
        })
        .width(70.0)
        .show_ui(ui, |ui| {
            for d in [
                AxisDirection::Positive,
                AxisDirection::Negative,
                AxisDirection::Both,
            ] {
                let label = match d {
                    AxisDirection::Positive => "Pos",
                    AxisDirection::Negative => "Neg",
                    AxisDirection::Both => "Both",
                };
                ui.selectable_value(current, d, label);
            }
        });
}

fn condition_editor(ui: &mut egui::Ui, id: &str, current: &mut SignalCondition) {
    // Extract the current variant + threshold (0 for variants without one).
    let (variant_idx, mut threshold) = match current {
        SignalCondition::GreaterThan(v) => (0, *v),
        SignalCondition::LessThan(v) => (1, *v),
        SignalCondition::Equals(v) => (2, *v),
        SignalCondition::NotEquals(v) => (3, *v),
        SignalCondition::Changed => (4, 0.0),
        SignalCondition::Always => (5, 0.0),
    };
    let mut new_idx = variant_idx;
    egui::ComboBox::from_id_salt(id)
        .selected_text(match variant_idx {
            0 => ">",
            1 => "<",
            2 => "==",
            3 => "!=",
            4 => "Changed",
            _ => "Always",
        })
        .width(90.0)
        .show_ui(ui, |ui| {
            for (i, label) in [(0, ">"), (1, "<"), (2, "=="), (3, "!="), (4, "Changed"), (5, "Always")] {
                ui.selectable_value(&mut new_idx, i, label);
            }
        });
    if matches!(new_idx, 0 | 1 | 2 | 3) {
        ui.add(egui::DragValue::new(&mut threshold).speed(0.1));
    }
    *current = match new_idx {
        0 => SignalCondition::GreaterThan(threshold),
        1 => SignalCondition::LessThan(threshold),
        2 => SignalCondition::Equals(threshold),
        3 => SignalCondition::NotEquals(threshold),
        4 => SignalCondition::Changed,
        _ => SignalCondition::Always,
    };
}

fn expression_editor(ui: &mut egui::Ui, id: &str, current: &mut SignalExpression) {
    // Extract variant + numeric args.
    let (variant_idx, mut v1, mut v2) = match current {
        SignalExpression::Constant(v) => (0, v.as_f32(), 0.0),
        SignalExpression::PassThrough => (1, 0.0, 0.0),
        SignalExpression::Invert => (2, 0.0, 0.0),
        SignalExpression::Scale(f) => (3, *f, 0.0),
        SignalExpression::Clamp(lo, hi) => (4, *lo, *hi),
    };
    let mut new_idx = variant_idx;
    egui::ComboBox::from_id_salt(id)
        .selected_text(match variant_idx {
            0 => "Const",
            1 => "Pass",
            2 => "Invert",
            3 => "Scale",
            _ => "Clamp",
        })
        .width(90.0)
        .show_ui(ui, |ui| {
            for (i, label) in [(0, "Const"), (1, "Pass"), (2, "Invert"), (3, "Scale"), (4, "Clamp")] {
                ui.selectable_value(&mut new_idx, i, label);
            }
        });
    match new_idx {
        0 | 3 => {
            ui.add(egui::DragValue::new(&mut v1).speed(0.1));
        }
        4 => {
            ui.add(egui::DragValue::new(&mut v1).speed(0.1).prefix("min "));
            ui.add(egui::DragValue::new(&mut v2).speed(0.1).prefix("max "));
        }
        _ => {}
    }
    *current = match new_idx {
        0 => SignalExpression::Constant(
            voxeldust_core::signal::types::SignalValue::Float(v1),
        ),
        1 => SignalExpression::PassThrough,
        2 => SignalExpression::Invert,
        3 => SignalExpression::Scale(v1),
        _ => SignalExpression::Clamp(v1, v2),
    };
}

fn channel_text_edit(
    ui: &mut egui::Ui,
    id: &str,
    current: &mut String,
    available: &[String],
) {
    // Free-text channel entry — user types whatever channel they want
    // (the server will create it on demand or reject if policy
    // forbids). `available` from the server is surfaced as a
    // click-away popup for discovery: click the dropdown icon to pick
    // a known channel, or type directly into the field.
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(current)
                .id_salt(id)
                .hint_text("channel name")
                .desired_width(180.0),
        );
        if !available.is_empty() {
            egui::ComboBox::from_id_salt(format!("{}-pick", id))
                .selected_text("▼")
                .width(24.0)
                .show_ui(ui, |ui| {
                    for name in available {
                        if ui.selectable_label(current == name, name).clicked() {
                            *current = name.clone();
                        }
                    }
                });
        }
    });
}

fn property_dropdown(ui: &mut egui::Ui, id: &str, current: &mut SignalProperty) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(format!("{:?}", current))
        .width(90.0)
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

fn button_styled(label: &str) -> egui::Button<'static> {
    egui::Button::new(
        egui::RichText::new(label)
            .color(egui::Color32::from_rgb(10, 16, 26))
            .size(11.0),
    )
    .fill(egui::Color32::from_rgb(120, 220, 255))
}

fn button_icon(label: &str) -> egui::Button<'static> {
    egui::Button::new(
        egui::RichText::new(label)
            .color(egui::Color32::from_rgb(240, 100, 100))
            .size(13.0),
    )
    .fill(egui::Color32::from_rgb(20, 28, 40))
}

/// Paint the in-world tablet cursor: a crisp white crosshair with a
/// dark outline so it reads against any egui content behind it.
/// Drawn on the top layer of the central panel.
fn paint_in_world_cursor(ui: &mut egui::Ui, pos: egui::Pos2) {
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("tablet_cursor"),
    ));

    // Outline (dark) first, then fill (white) on top — 1 px fatter
    // outline gives the reticle readability against light and dark
    // backgrounds.
    let outline = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 220);
    let fill = egui::Color32::WHITE;
    let arm = 8.0;
    let gap = 2.0;
    for (ofs_scale, color) in [(1.6, outline), (1.0, fill)] {
        let w = 1.5 * ofs_scale;
        let stroke = egui::Stroke::new(w, color);
        // Horizontal arms with a small gap at the centre.
        painter.line_segment(
            [pos + egui::vec2(-arm, 0.0), pos + egui::vec2(-gap, 0.0)],
            stroke,
        );
        painter.line_segment(
            [pos + egui::vec2(gap, 0.0), pos + egui::vec2(arm, 0.0)],
            stroke,
        );
        painter.line_segment(
            [pos + egui::vec2(0.0, -arm), pos + egui::vec2(0.0, -gap)],
            stroke,
        );
        painter.line_segment(
            [pos + egui::vec2(0.0, gap), pos + egui::vec2(0.0, arm)],
            stroke,
        );
    }
    painter.circle_filled(pos, 1.5, fill);
}

fn style_cockpit(ui: &mut egui::Ui) {
    // Cockpit-display aesthetic: dark panel, cyan primary accent,
    // rounded buttons.
    let mut s = ui.style_mut().clone();
    s.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(14, 22, 34);
    s.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(22, 34, 50);
    s.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(40, 80, 120);
    s.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(80, 160, 220);
    s.visuals.extreme_bg_color = egui::Color32::from_rgb(10, 16, 26);
    s.visuals.override_text_color = Some(egui::Color32::from_rgb(210, 230, 245));
    ui.set_style(s);
}

/// Drive the tablet's egui pointer from `HudFocusState.cursor_uv` and
/// the `HudClickEvent` stream. Runs in `EguiInputSet::InitReading`.
///
/// **Emit discipline**:
///   * `PointerMoved` only fires when the cursor UV actually changes
///     OR when we just (re-)acquired the context. Emitting every
///     frame (60/s) queues a backlog that outlives tablet despawn —
///     bevy_egui then logs `NotSpawned` errors when it dequeues
///     events targeting a dead context entity.
///   * `HoveredNonWindowEguiContext` + `FocusedNonWindowEguiContext`
///     are inserted once when focus first activates, removed once
///     when the tablet or focus goes away. No per-frame churn.
#[allow(clippy::too_many_arguments)]
fn drive_tablet_pointer(
    focus: Res<HudFocusState>,
    tablet: Query<Entity, With<HeldTablet>>,
    ctx_q: Query<Entity, With<TabletEguiContextMarker>>,
    mut pos_q: Query<&mut EguiContextPointerPosition>,
    mut clicks: MessageReader<HudClickEvent>,
    mouse_delta: Res<crate::input::FrameMouseDelta>,
    mut writer: MessageWriter<EguiInputEvent>,
    mut commands: Commands,
    mut last_hovered: Local<Option<Entity>>,
    mut last_cursor_uv: Local<Option<bevy::prelude::Vec2>>,
) {
    let tablet_present = tablet.iter().next().is_some();
    let Ok(ctx_entity) = ctx_q.single() else {
        if last_hovered.is_some() {
            commands.remove_resource::<HoveredNonWindowEguiContext>();
            commands.remove_resource::<FocusedNonWindowEguiContext>();
            *last_hovered = None;
            *last_cursor_uv = None;
        }
        return;
    };
    if !tablet_present || !focus.active {
        if last_hovered.is_some() {
            commands.remove_resource::<HoveredNonWindowEguiContext>();
            commands.remove_resource::<FocusedNonWindowEguiContext>();
            *last_hovered = None;
            *last_cursor_uv = None;
        }
        return;
    }

    let px = focus.cursor_uv.x * (TABLET_UI_RES as f32 - 1.0);
    let py = focus.cursor_uv.y * (TABLET_UI_RES as f32 - 1.0);
    let egui_pos = egui::pos2(px, py);

    if let Ok(mut p) = pos_q.get_mut(ctx_entity) {
        p.position = egui_pos;
    }

    let just_acquired = *last_hovered != Some(ctx_entity);
    if just_acquired {
        commands.insert_resource(HoveredNonWindowEguiContext(ctx_entity));
        commands.insert_resource(FocusedNonWindowEguiContext(ctx_entity));
        *last_hovered = Some(ctx_entity);
    }

    // Emit PointerMoved only on change (or on first acquisition so
    // egui gets an initial position). Avoids queuing 60 events/s
    // which would outlive the tablet's despawn.
    let cursor_changed = last_cursor_uv
        .map(|last| last != focus.cursor_uv)
        .unwrap_or(true);
    if cursor_changed || just_acquired {
        writer.write(EguiInputEvent {
            context: ctx_entity,
            event: egui::Event::PointerMoved(egui_pos),
        });
        *last_cursor_uv = Some(focus.cursor_uv);
    }

    for click in clicks.read() {
        if click.tile != tablet.iter().next().unwrap() {
            continue;
        }
        let button = match click.button {
            HudClickButton::Left => egui::PointerButton::Primary,
            HudClickButton::Right => egui::PointerButton::Secondary,
            HudClickButton::Middle => egui::PointerButton::Middle,
        };
        writer.write(EguiInputEvent {
            context: ctx_entity,
            event: egui::Event::PointerButton {
                pos: egui_pos,
                button,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        });
        writer.write(EguiInputEvent {
            context: ctx_entity,
            event: egui::Event::PointerButton {
                pos: egui_pos,
                button,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        });
    }

    // Forward mouse-wheel deltas so the ScrollArea inside the tablet
    // scrolls naturally. `scroll_y` in pixels/tick; egui's
    // `MouseWheel` takes a `Vec2` delta in line-or-pixel units
    // depending on the `unit` field.
    if mouse_delta.scroll_y != 0.0 {
        writer.write(EguiInputEvent {
            context: ctx_entity,
            event: egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: egui::vec2(0.0, mouse_delta.scroll_y),
                modifiers: egui::Modifiers::NONE,
            },
        });
    }
}
