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

use voxeldust_core::block::registry::FunctionalBlockKind;
use voxeldust_core::client_message::ClientMsg;
use voxeldust_core::signal::config::{
    AccessStatusForChannel, AntennaConfig, AntennaSide, BlockConfigUpdateData, BlockSignalConfig,
    HeldGrantSummary, PublishBindingConfig, SubscribeBindingConfig, TerminalConfig,
};
use voxeldust_core::signal::types::{SignalProperty, SignalScope};
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
            // Save-on-close: when the tablet is dismissed (F-key,
            // ESC, Close button → `DespawnHeldTablet`), ship a single
            // `BlockConfigUpdate` carrying the player's full edited
            // config. Runs BEFORE the tablet despawn so the edit
            // leaves the wire while the panel state is still around.
            .add_systems(
                Update,
                save_on_tablet_despawn.before(crate::hud::tablet::TabletDespawnSet),
            )
            .add_systems(EguiPrimaryContextPass, render_config_panel);
    }
}

/// Save-on-close: ship the editable buffer as a single
/// `BlockConfigUpdate` when the tablet is about to despawn. Replaces
/// the previous debounced auto-save-on-change which silently dropped
/// some kinds (e.g. antenna config) because the change-detection
/// state didn't always re-fire when nested fields mutated.
///
/// Idempotency: if the player closes the tablet without editing
/// anything, we still send. The server's apply path is idempotent for
/// equal configs (re-resolving the same channel ids, same grant
/// validation), so the cost is one TCP packet — cheaper than tracking
/// dirty state from every nested egui widget.
fn save_on_tablet_despawn(
    mut events: MessageReader<DespawnHeldTablet>,
    panel: Res<OpenConfigPanel>,
    tcp: Res<TcpSender>,
) {
    let mut should_flush = false;
    for _ in events.read() {
        should_flush = true;
    }
    if !should_flush {
        return;
    }
    let Some(state) = panel.editable.as_ref() else {
        return;
    };
    let bytes = serialize_block_config_update(&state.config);
    if send_block_config_bytes(&tcp, &bytes) {
        tracing::info!(
            block = ?(state.config.block_pos.x, state.config.block_pos.y, state.config.block_pos.z),
            "tablet save-on-close: BlockConfigUpdate flushed",
        );
    }
}

/// Build the `BlockConfigUpdate` wire bytes for a `BlockSignalConfig`
/// snapshot. Used both for diff detection (compare to last sent) and
/// for actually sending. Single source of truth so the comparator and
/// the sender are guaranteed to agree on which fields cross the wire.
pub fn serialize_block_config_update(config: &BlockSignalConfig) -> Vec<u8> {
    let update = BlockConfigUpdateData {
        block_pos: config.block_pos,
        publish_bindings: config.publish_bindings.clone(),
        subscribe_bindings: config.subscribe_bindings.clone(),
        converter_rules: config.converter_rules.clone(),
        seat_mappings: config.seat_mappings.clone(),
        seated_channel_name: config.seated_channel_name.clone(),
        power_source: config.power_source.clone(),
        power_consumer: config.power_consumer.clone(),
        flight_computer: config.flight_computer.clone(),
        hover_module: config.hover_module.clone(),
        autopilot: config.autopilot.clone(),
        warp_computer: config.warp_computer.clone(),
        engine_controller: config.engine_controller.clone(),
        mechanical: config.mechanical.clone(),
        antenna: config.antenna.clone(),
        terminal: config.terminal.clone(),
    };
    let msg = ClientMsg::BlockConfigUpdate(update);
    let data = msg.serialize();
    let mut pkt = Vec::new();
    wire_codec::encode(&data, &mut pkt);
    pkt
}

/// Send pre-encoded `BlockConfigUpdate` bytes via TCP. Returns
/// `true` on enqueue success — the caller updates its
/// `last_sent_bytes` only when the bytes actually leave (so a
/// dropped TCP retries on the next Update tick).
fn send_block_config_bytes(tcp: &TcpSender, pkt: &[u8]) -> bool {
    if tcp.tx.send(pkt.to_vec()).is_err() {
        tracing::warn!("TCP channel closed while sending BlockConfigUpdate");
        return false;
    }
    true
}

/// Clear `editable` when the tablet entity no longer exists so the
/// next F press doesn't flash the PREVIOUS block's config before the
/// server's fresh `BlockConfigState` arrives. Also resets the
/// auto-save bookkeeping so the next-opened tablet starts with a
/// clean baseline.
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
///
/// `state` is the server-authoritative snapshot from the most recent
/// `BlockConfigState`; `editable` is the player's working copy that
/// the egui panel mutates in place. Save-on-close ships `editable` to
/// the server when the tablet is dismissed.
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
    /// Inline "Add held grant" form state — kept on the panel so the
    /// player can paste a multi-line key without losing partial input
    /// across paints. Cleared after a successful Add.
    pub add_grant: AddHeldGrantForm,
}

/// Per-panel inline form for pasting a held grant. String fields stay
/// as String (not parsed mid-edit) so a temporarily-invalid value
/// doesn't get clobbered while the player is typing.
#[derive(Default, Clone, Debug)]
pub struct AddHeldGrantForm {
    pub grant_id: String,
    pub key_b64: String,
    pub target_shard_id: String,
    pub label: String,
    /// Last-attempt status string — surfaces parse errors / send
    /// confirmation ("Added grant 0x…").
    pub status: String,
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
            // Preserve any in-progress "Add held grant" form when the
            // server pushes a fresh BlockConfigState — the player may
            // be mid-paste; clobbering it would feel hostile. Carry
            // forward when the same block is being re-snapshotted.
            let preserved_add_grant = panel
                .editable
                .as_ref()
                .filter(|s| s.config.block_pos == cfg.block_pos)
                .map(|s| s.add_grant.clone())
                .unwrap_or_default();
            panel.editable = Some(ConfigPanelState {
                shard,
                config: cfg.clone(),
                pending_close: false,
                add_grant: preserved_add_grant,
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

            // Publish bindings editor — hidden entirely if this block kind
            // can't publish (e.g., Thrusters). Default property for new
            // bindings is the first entry in the kind's schema.
            let pub_opts = &state.config.publish_property_options;
            if !pub_opts.is_empty() {
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
                        property_combo(ui, &format!("pub-prop-{}", i), &mut b.property, pub_opts);
                        scope_combo(ui, &format!("pub-{}", i), &mut b.scope);
                        if ui.button("✕").clicked() {
                            remove_publish = Some(i);
                        }
                    });
                }
                if let Some(i) = remove_publish {
                    state.config.publish_bindings.remove(i);
                }
                if ui.button("+ add publisher").clicked() {
                    let default_prop = pub_opts
                        .first()
                        .and_then(|(o, _)| SignalProperty::from_ordinal(*o))
                        .unwrap_or(SignalProperty::Active);
                    state.config.publish_bindings.push(PublishBindingConfig {
                        channel_name: String::new(),
                        property: default_prop,
                        scope: None,
                        grant_id: None,
                    });
                }
                ui.separator();
            }

            // Subscribe bindings editor — hidden if this kind can't subscribe.
            let sub_opts = &state.config.subscribe_property_options;
            if !sub_opts.is_empty() {
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
                        property_combo(ui, &format!("sub-prop-{}", i), &mut b.property, sub_opts);
                        scope_combo(ui, &format!("sub-{}", i), &mut b.scope);
                        if ui.button("✕").clicked() {
                            remove_subscribe = Some(i);
                        }
                    });
                }
                if let Some(i) = remove_subscribe {
                    state.config.subscribe_bindings.remove(i);
                }
                if ui.button("+ add subscriber").clicked() {
                    let default_prop = sub_opts
                        .first()
                        .and_then(|(o, _)| SignalProperty::from_ordinal(*o))
                        .unwrap_or(SignalProperty::Active);
                    state.config.subscribe_bindings.push(SubscribeBindingConfig {
                        channel_name: String::new(),
                        property: default_prop,
                        scope: None,
                        grant_id: None,
                    });
                }
                ui.separator();
            }

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

            // Phase D: bidirectional Antenna config. One block carries
            // both a TX side (forward a local channel onto a Radio
            // frequency) and an RX side (subscribe to a Radio frequency,
            // mirror the value onto a local channel). Most antennas use
            // both for full-duplex chat at one frequency; uni-directional
            // antennas leave one side empty.
            if state.config.kind == FunctionalBlockKind::Antenna.as_u8() {
                ui.heading("Antenna config");
                let cfg = state.config.antenna.get_or_insert_with(AntennaConfig::default);

                // Per-side renders: status comes from the server
                // (auth_required, held_grants), edited fields live on
                // the side struct itself.
                // Phase A3 UX: SEND (channel ──▶ frequency) and
                // RECEIVE (frequency ──▶ channel) make the data flow
                // direction obvious without a paragraph of explanation.
                render_antenna_side_panel(
                    ui,
                    "tx",
                    "SEND",
                    "channel ──▶ frequency",
                    "ops=Publish",
                    &mut cfg.tx,
                    state.config.antenna_tx_status.as_ref(),
                );
                ui.separator();

                render_antenna_side_panel(
                    ui,
                    "rx",
                    "RECEIVE",
                    "frequency ──▶ channel",
                    "ops=Subscribe",
                    &mut cfg.rx,
                    state.config.antenna_rx_status.as_ref(),
                );
                ui.label(
                    egui::RichText::new(
                        "Open Radio (channel has no key) needs no grant. \
                         Keyed Radio: pick a held grant or request access.",
                    )
                    .color(egui::Color32::from_rgb(140, 160, 180))
                    .size(11.0),
                );
                ui.separator();
            }

            // Phase D: Terminal config — read+write text via the media
            // pipeline. Either side can be left empty for read-only signs
            // (no publish channel) or input-only kiosks (no subscribe
            // channel). E-key on the placed block opens the in-world
            // chat overlay (scrollback + input).
            if state.config.kind == FunctionalBlockKind::Terminal.as_u8() {
                ui.heading("Terminal config");
                let cfg = state.config.terminal.get_or_insert_with(TerminalConfig::default);
                let mut sub = cfg.subscribe_channel_name.clone().unwrap_or_default();
                ui.label("Subscribe channel (display reads from):");
                if ui.text_edit_singleline(&mut sub).changed() {
                    cfg.subscribe_channel_name =
                        if sub.is_empty() { None } else { Some(sub.clone()) };
                }
                let mut pub_ = cfg.publish_channel_name.clone().unwrap_or_default();
                ui.label("Publish channel (E-key sends to):");
                if ui.text_edit_singleline(&mut pub_).changed() {
                    cfg.publish_channel_name =
                        if pub_.is_empty() { None } else { Some(pub_.clone()) };
                }
                ui.horizontal(|ui| {
                    ui.label("Scrollback lines:");
                    let mut s = cfg
                        .scrollback_lines
                        .unwrap_or(TerminalConfig::DEFAULT_SCROLLBACK)
                        .to_string();
                    if ui.text_edit_singleline(&mut s).changed() {
                        if let Ok(v) = s.parse::<u16>() {
                            cfg.scrollback_lines = Some(v);
                        }
                    }
                });
                ui.label(
                    "Tip: same channel for sub + pub = chat panel. \
                     Sub-only = sign. Pub-only = input kiosk.",
                );
                ui.separator();
            }

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
            antenna: state.config.antenna.clone(),
            terminal: state.config.terminal.clone(),
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

/// Phase D: render one side of a bidirectional Antenna config —
/// channel, frequency, optional shard target, optional grant. Used by
/// both TX and RX (the rendering is symmetric; direction is the slot
/// the side lives in).
fn render_antenna_side_panel(
    ui: &mut egui::Ui,
    id_salt: &str,
    direction_label: &str,
    arrow_hint: &str,
    grant_op_hint: &str,
    side: &mut Option<AntennaSide>,
    status: Option<&AccessStatusForChannel>,
) {
    // "Disabled side" toggle: bind the slot to None when off, defaults
    // when on. Single source of truth for "this side configured?".
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(direction_label)
                .color(egui::Color32::from_rgb(120, 220, 160))
                .strong(),
        );
        ui.label(
            egui::RichText::new(arrow_hint)
                .color(egui::Color32::from_rgb(140, 160, 180)),
        );
        let mut enabled = side.is_some();
        if ui.checkbox(&mut enabled, "enabled").changed() {
            if enabled && side.is_none() {
                *side = Some(AntennaSide::default());
            } else if !enabled {
                *side = None;
            }
        }
    });
    let Some(s) = side.as_mut() else {
        return;
    };
    ui.horizontal(|ui| {
        ui.label("Channel:");
        ui.text_edit_singleline(&mut s.local_channel_name);
    });
    ui.horizontal(|ui| {
        ui.label("Frequency:");
        let mut freq_str = s.frequency.to_string();
        if ui.text_edit_singleline(&mut freq_str).changed() {
            if let Ok(v) = freq_str.parse::<u32>() {
                s.frequency = v;
            }
        }
    });
    render_grant_picker(ui, id_salt, grant_op_hint, &mut s.grant_id, status);
    egui::CollapsingHeader::new("Advanced")
        .id_salt(format!("{}-advanced", id_salt))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Target shard id (blank = relay-routed):");
                let mut t_str = s
                    .remote_shard_id
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                if ui.text_edit_singleline(&mut t_str).changed() {
                    s.remote_shard_id = t_str.parse::<u64>().ok().filter(|v| *v != 0);
                }
            });
        });
}

/// Render the grant picker for an Antenna side or a binding row.
/// Open channel ⇒ small caption; keyed + held grants ⇒ dropdown of
/// labels; keyed + no held grants ⇒ "Request access" button placeholder.
/// Server populates `status` from the channel's resolved auth state.
fn render_grant_picker(
    ui: &mut egui::Ui,
    id_salt: &str,
    op_hint: &str,
    current: &mut Option<u64>,
    status: Option<&AccessStatusForChannel>,
) {
    let id = format!("{}-grant", id_salt);
    let auth_required = status.map(|s| s.auth_required).unwrap_or(false);
    if !auth_required {
        ui.label(
            egui::RichText::new(format!("Open broadcast — no grant required ({op_hint})"))
                .weak(),
        );
        // Keep `*current = None` for open channels so the wire path
        // ships grant_id=0 + empty auth_tag.
        if current.is_some() {
            *current = None;
        }
        return;
    }
    let held: &[HeldGrantSummary] = status.map(|s| s.held_grants.as_slice()).unwrap_or(&[]);
    if held.is_empty() {
        ui.horizontal(|ui| {
            ui.colored_label(
                egui::Color32::from_rgb(220, 200, 80),
                "Keyed channel — no held grants for this side.",
            );
            // Phase D: clicking emits a RequestAccess client message
            // that lands in the channel-owner's tablet inbox. Wired in a
            // follow-up — the placeholder makes the affordance explicit.
            let _ = ui.button("Request access…");
        });
        return;
    }
    let label = match current {
        Some(g) => held
            .iter()
            .find(|s| s.grant_id == *g)
            .map(|s| s.label.clone())
            .unwrap_or_else(|| format!("grant {g}")),
        None => "(pick a held grant)".to_string(),
    };
    egui::ComboBox::from_id_salt(id)
        .selected_text(label)
        .show_ui(ui, |ui| {
            for g in held {
                ui.selectable_value(current, Some(g.grant_id), g.label.clone());
            }
        });
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

/// Renders the property selector for a binding row. Filtered to the set the
/// block kind actually supports (`options` is `BlockSignalConfig::publish_property_options`
/// or `subscribe_property_options` — server-populated from
/// `FunctionalBlockKind::signal_schema()`). UX:
/// - 0 entries: should never reach this fn (the parent hides the whole section).
/// - 1 entry: render as a static label, not a dropdown — there's nothing to choose.
/// - 2+ entries: dropdown with the hint string rendered as a tooltip per option.
fn property_combo(
    ui: &mut egui::Ui,
    id: &str,
    current: &mut SignalProperty,
    options: &[(u8, String)],
) {
    if options.is_empty() {
        // Defensive: parent should hide the section before reaching here.
        ui.label(format!("{:?}", current));
        return;
    }

    // Coerce `current` to the first allowed option if it's somehow out of set
    // (e.g., legacy persisted config that no longer matches the schema).
    if !options.iter().any(|(o, _)| *o == current.as_ordinal()) {
        if let Some(first) = options.first().and_then(|(o, _)| SignalProperty::from_ordinal(*o)) {
            *current = first;
        }
    }

    if options.len() == 1 {
        ui.label(format!("{:?}", current));
        if let Some(hint) = options.first().map(|(_, h)| h).filter(|h| !h.is_empty()) {
            ui.label(egui::RichText::new(format!("({})", hint)).weak());
        }
        return;
    }

    egui::ComboBox::from_id_salt(id)
        .selected_text(format!("{:?}", current))
        .show_ui(ui, |ui| {
            for (ord, hint) in options {
                let Some(prop) = SignalProperty::from_ordinal(*ord) else { continue };
                let label = if hint.is_empty() {
                    format!("{:?}", prop)
                } else {
                    format!("{:?}", prop)
                };
                let resp = ui.selectable_value(current, prop, label);
                if !hint.is_empty() {
                    resp.on_hover_text(hint);
                }
            }
        });
}

/// Renders the scope picker for a binding row. The binding's `scope` is
/// `Option<SignalScope>`: `None` means "use the channel's default scope as
/// established at the home shard"; `Some(_)` is an explicit override at
/// resolve time. UX:
/// - dropdown with: Default · Local · ShortRange · LongRange · Radio
/// - when ShortRange is selected, a meters DragValue appears
/// - when Radio is selected, a frequency DragValue appears
///
/// Override semantics: a publisher can elect to publish under a wider scope
/// than the channel's default (e.g. relay a Local sensor onto Radio via an
/// Antenna binding) iff the channel auth + access policy permit it server-side.
/// Subscribers use the override to filter inbound entries to a specific scope.
fn scope_combo(ui: &mut egui::Ui, id: &str, current: &mut Option<SignalScope>) {
    let tag = match current {
        None => "default",
        Some(SignalScope::Local) => "local",
        Some(SignalScope::ShortRange { .. }) => "short",
        Some(SignalScope::LongRange) => "long",
        Some(SignalScope::Radio { .. }) => "radio",
    };
    egui::ComboBox::from_id_salt(format!("{}-scope", id))
        .selected_text(tag)
        .show_ui(ui, |ui| {
            if ui.selectable_label(matches!(current, None), "default").clicked() {
                *current = None;
            }
            if ui
                .selectable_label(matches!(current, Some(SignalScope::Local)), "local")
                .clicked()
            {
                *current = Some(SignalScope::Local);
            }
            if ui
                .selectable_label(
                    matches!(current, Some(SignalScope::ShortRange { .. })),
                    "short",
                )
                .clicked()
            {
                if !matches!(current, Some(SignalScope::ShortRange { .. })) {
                    *current = Some(SignalScope::ShortRange { range_m: 1000.0 });
                }
            }
            if ui
                .selectable_label(matches!(current, Some(SignalScope::LongRange)), "long")
                .clicked()
            {
                *current = Some(SignalScope::LongRange);
            }
            if ui
                .selectable_label(matches!(current, Some(SignalScope::Radio { .. })), "radio")
                .clicked()
            {
                if !matches!(current, Some(SignalScope::Radio { .. })) {
                    *current = Some(SignalScope::Radio { frequency: 0 });
                }
            }
        });
    match current {
        Some(SignalScope::ShortRange { range_m }) => {
            let mut v = *range_m;
            if ui
                .add(
                    egui::DragValue::new(&mut v)
                        .speed(50.0)
                        .range(0.0..=1.0e9)
                        .suffix(" m"),
                )
                .changed()
            {
                *range_m = v;
            }
        }
        Some(SignalScope::Radio { frequency }) => {
            let mut v = *frequency;
            if ui
                .add(
                    egui::DragValue::new(&mut v)
                        .speed(1.0)
                        .range(0..=u32::MAX)
                        .prefix("ƒ "),
                )
                .changed()
            {
                *frequency = v;
            }
        }
        _ => {}
    }
}
