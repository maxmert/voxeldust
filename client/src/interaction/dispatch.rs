//! Mouse + keyboard → `ClientMsg::BlockEditRequest` dispatch.
//!
//! LMB → BREAK, RMB → PLACE, MMB → pick block_type, E → INTERACT /
//! ENTER_SEAT, F → EXIT_SEAT. The raycast (Phase 14) populates
//! `BlockTarget` with a `ShardKey` identifying the shard that owns
//! the hit; dispatch routes the edit to that shard's TCP.
//!
//! **Transport routing** (Design Principle #10): the owning shard's
//! TCP is the write channel. Phase 15 MVP routes everything via the
//! primary TCP; when the hit is on a secondary we log a warning —
//! the server-side `ObserverConnect` must accept writes for full
//! cross-shard interaction (documented server-side TODO). Behavior
//! stays correct once the server change lands: client flips a single
//! line in `send_block_edit` to dispatch via the per-shard TCP map.

use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use voxeldust_core::client_message::{
    action as edit_action, BlockEditData, ClientMsg, SubBlockEditData,
};
use voxeldust_core::wire_codec;

use crate::config_panel::{OpenConfigPanel, PendingConfigShard};
use voxeldust_core::block::palette::CHUNK_SIZE;
use crate::hud::tablet::{DespawnHeldTablet, HeldTablet, SpawnHeldTablet};
use crate::input::SeatedState;
use crate::interaction::raycast::BlockTarget;
use crate::net::TcpSender;
use crate::shard::{PrimaryShard, ShardKey};

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InteractionSet;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Hotbar>()
            .init_resource::<SubBlockTool>()
            .add_systems(
                Update,
                (toggle_sub_block_tool, dispatch_interactions)
                    .chain()
                    .in_set(InteractionSet),
            );
    }
}

/// Currently-selected placement block. Defaults to Stone (id=1).
/// Phase 20's inventory UI drives this; for now MMB picks + number-
/// key slots drive it.
#[derive(Resource, Debug, Clone, Copy)]
pub struct Hotbar {
    pub block_type: u16,
}

impl Default for Hotbar {
    fn default() -> Self {
        Self { block_type: 1 } // Stone
    }
}

/// Sub-block tool state. T toggles. While active, LMB/RMB send
/// `SubBlockEditRequest` (place/remove on the hit block's face)
/// instead of `BlockEditRequest`. R cycles the rotation. 1-8
/// select the element type from a palette matching legacy voxydust.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SubBlockTool {
    pub active: bool,
    pub element_type: u8,
    pub rotation: u8,
}

impl Default for SubBlockTool {
    fn default() -> Self {
        Self {
            active: false,
            // 0 = PowerWire — see `SubBlockType::from_u8` in
            // core::block::sub_block.
            element_type: 0,
            rotation: 0,
        }
    }
}

fn toggle_sub_block_tool(
    keys: Res<ButtonInput<KeyCode>>,
    mut tool: ResMut<SubBlockTool>,
) {
    if keys.just_pressed(KeyCode::KeyT) {
        tool.active = !tool.active;
        tracing::info!(active = tool.active, "sub-block tool toggled");
    }
    if tool.active {
        if keys.just_pressed(KeyCode::KeyR) {
            tool.rotation = (tool.rotation + 1) & 0b11;
        }
        // Palette keys. Values are wire-level `SubBlockType` ordinals
        // (see `core::block::sub_block::SubBlockType`). Note the
        // non-contiguous values — the enum reserves ranges per
        // category so future additions don't renumber.
        for (key, ty) in [
            (KeyCode::Digit1, 0),   // PowerWire
            (KeyCode::Digit2, 1),   // SignalWire
            (KeyCode::Digit3, 43),  // Cable
            (KeyCode::Digit4, 10),  // Rail
            (KeyCode::Digit5, 30),  // Pipe
            (KeyCode::Digit6, 40),  // Ladder
            (KeyCode::Digit7, 42),  // SurfaceLight
            (KeyCode::Digit8, 50),  // Bracket
            (KeyCode::Digit9, 60),  // HudPanel — in-world signal-driven HUD tile
        ] {
            if keys.just_pressed(key) {
                tool.element_type = ty;
                tracing::info!(element_type = ty, "sub-block element selected");
            }
        }
    }
}

/// Bundle of tablet / HUD-panel config resources + helpers. Groups
/// them into a single `SystemParam` because Bevy's system-fn tuple
/// can't exceed 16 arguments.
#[derive(bevy::ecs::system::SystemParam)]
struct TabletParams<'w, 's> {
    config_panel_state: ResMut<'w, OpenConfigPanel>,
    panel_configs: Res<'w, crate::hud::panel_config::HudPanelConfigs>,
    hud_panel_edit: ResMut<'w, crate::hud::panel_config::OpenHudPanelConfig>,
    storage: Res<'w, crate::chunk::cache::ChunkStorageCache>,
    existing_tablet: Query<'w, 's, Entity, With<HeldTablet>>,
    spawn_tablet: MessageWriter<'w, SpawnHeldTablet>,
    despawn_tablet: MessageWriter<'w, DespawnHeldTablet>,
}

fn dispatch_interactions(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    target: Res<BlockTarget>,
    primary: Res<PrimaryShard>,
    tcp: Res<TcpSender>,
    mut hotbar: ResMut<Hotbar>,
    tool: Res<SubBlockTool>,
    seated: Res<SeatedState>,
    mut pending_config: ResMut<PendingConfigShard>,
    hud_focus: Res<crate::hud::HudFocusState>,
    mut tablet_params: TabletParams,
) {
    let TabletParams {
        ref mut config_panel_state,
        ref panel_configs,
        ref mut hud_panel_edit,
        ref storage,
        ref existing_tablet,
        ref mut spawn_tablet,
        ref mut despawn_tablet,
    } = tablet_params;
    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !grabbed {
        return;
    }

    // While the tablet cursor is active, the player is interacting
    // with the in-world UI — typing into text fields, clicking
    // buttons. Suppress F's "toggle tablet" action so keys typed in
    // a text field don't close the tablet out from under the user.
    // Free-move mode (Tab off) keeps F working.
    if hud_focus.active && keys.just_pressed(KeyCode::KeyF) {
        // No-op: consumed by the tablet's egui context.
    } else if keys.just_pressed(KeyCode::KeyF) {
        if seated.is_seated {
            send_edit(
                &tcp,
                BlockEditData {
                    action: edit_action::EXIT_SEAT,
                    eye: glam::DVec3::ZERO,
                    look: glam::DVec3::ZERO,
                    block_type: 0,
                },
                None,
                &primary,
            );
        } else if existing_tablet.iter().next().is_some() {
            // Tablet currently visible → dismiss.
            despawn_tablet.write(DespawnHeldTablet);
        } else {
            // Tablet currently hidden → summon. Route based on what's
            // under the crosshair:
            //
            //   Hit face has a `HudPanel` sub-block
            //       → open HUD-panel config (widget kind + channel).
            //         No block-config round trip; the panel config is
            //         client-local today.
            //   Hit face has nothing (or the hit is a functional block)
            //       → clear stale editable buffer, send OPEN_CONFIG,
            //         wait for BlockConfigState to populate.
            config_panel_state.editable = None;
            hud_panel_edit.editing = None;

            // Detect HudPanel on the hit face.
            let hud_panel_hit = target.hit.and_then(|hit| {
                let local = bevy::prelude::IVec3::new(
                    hit.block_pos.x.rem_euclid(CHUNK_SIZE as i32),
                    hit.block_pos.y.rem_euclid(CHUNK_SIZE as i32),
                    hit.block_pos.z.rem_euclid(CHUNK_SIZE as i32),
                );
                let chunk_idx = bevy::prelude::IVec3::new(
                    hit.block_pos.x.div_euclid(CHUNK_SIZE as i32),
                    hit.block_pos.y.div_euclid(CHUNK_SIZE as i32),
                    hit.block_pos.z.div_euclid(CHUNK_SIZE as i32),
                );
                let face = face_normal_to_face_u8(hit.face_normal);
                storage.get(hit.shard, chunk_idx).and_then(|c| {
                    c.get_sub_blocks(local.x as u8, local.y as u8, local.z as u8)
                        .iter()
                        .any(|e| {
                            e.face == face
                                && e.element_type
                                    == voxeldust_core::block::sub_block::SubBlockType::HudPanel
                        })
                        .then_some((hit.shard, hit.block_pos, face))
                })
            });

            if let Some((shard, block_pos, face)) = hud_panel_hit {
                let key = crate::hud::panel_config::HudPanelKey {
                    shard,
                    block_pos: bevy::prelude::IVec3::new(block_pos.x, block_pos.y, block_pos.z),
                    face,
                };
                let settings = panel_configs.get_or_default(key);
                hud_panel_edit.editing = Some(crate::hud::panel_config::HudPanelEditState {
                    key,
                    settings,
                });
                spawn_tablet.write(SpawnHeldTablet {
                    shard,
                    config: voxeldust_core::signal::config::BlockSignalConfig::default(),
                });
                tracing::info!(
                    block = ?(block_pos.x, block_pos.y, block_pos.z),
                    face,
                    "F on HudPanel sub-block — opening panel config",
                );
            } else {
                spawn_tablet.write(SpawnHeldTablet {
                    shard: target
                        .hit
                        .map(|h| h.shard)
                        .or(primary.current)
                        .unwrap_or(ShardKey::new(0, 0)),
                    config: voxeldust_core::signal::config::BlockSignalConfig::default(),
                });
                if let Some(hit) = target.hit {
                    pending_config.0 = Some(hit.shard);
                    send_edit(
                        &tcp,
                        BlockEditData {
                            action: edit_action::OPEN_CONFIG,
                            eye: dvec3_from_vec3(hit.eye_local),
                            look: dvec3_from_vec3(hit.look_local),
                            block_type: 0,
                        },
                        Some(hit.shard),
                        &primary,
                    );
                }
            }
        }
    }

    let Some(hit) = target.hit else {
        return;
    };

    // When the tablet cursor is active, every mouse click is meant
    // for the in-world tablet UI — not for the world. Suppress
    // LMB/RMB/MMB world dispatch so clicking an egui button on the
    // tablet doesn't ALSO break / place a block (or sub-block) at the
    // last raycast hit.
    if hud_focus.active {
        return;
    }

    // LMB — break block, or (in sub-block mode) remove the sub-block
    // element the crosshair is on.
    if mouse.just_pressed(MouseButton::Left) {
        if tool.active {
            send_sub_block_edit(
                &tcp,
                SubBlockEditData {
                    block_pos: glam::IVec3::new(
                        hit.block_pos.x,
                        hit.block_pos.y,
                        hit.block_pos.z,
                    ),
                    face: face_normal_to_face_u8(hit.face_normal),
                    element_type: tool.element_type,
                    rotation: tool.rotation,
                    action: edit_action::REMOVE_SUB,
                },
                Some(hit.shard),
                &primary,
            );
        } else {
            send_edit(
                &tcp,
                BlockEditData {
                    action: edit_action::BREAK,
                    eye: dvec3_from_vec3(hit.eye_local),
                    look: dvec3_from_vec3(hit.look_local),
                    block_type: 0,
                },
                Some(hit.shard),
                &primary,
            );
        }
    }

    // RMB — place block (hotbar) or place sub-block element (in
    // sub-block mode).
    if mouse.just_pressed(MouseButton::Right) {
        if tool.active {
            send_sub_block_edit(
                &tcp,
                SubBlockEditData {
                    block_pos: glam::IVec3::new(
                        hit.block_pos.x,
                        hit.block_pos.y,
                        hit.block_pos.z,
                    ),
                    face: face_normal_to_face_u8(hit.face_normal),
                    element_type: tool.element_type,
                    rotation: tool.rotation,
                    action: edit_action::PLACE_SUB,
                },
                Some(hit.shard),
                &primary,
            );
        } else {
            send_edit(
                &tcp,
                BlockEditData {
                    action: edit_action::PLACE,
                    eye: dvec3_from_vec3(hit.eye_local),
                    look: dvec3_from_vec3(hit.look_local),
                    block_type: hotbar.block_type,
                },
                Some(hit.shard),
                &primary,
            );
        }
    }

    // MMB = pick block_type into the hotbar (client-side state).
    if mouse.just_pressed(MouseButton::Middle) {
        hotbar.block_type = hit.block_type.as_u16();
        tracing::info!(block_type = hotbar.block_type, "hotbar: picked from target");
    }

    // E = INTERACT / ENTER_SEAT. Server decides which based on
    // the hit block's functional kind + the player's seated state.
    if keys.just_pressed(KeyCode::KeyE) {
        send_edit(
            &tcp,
            BlockEditData {
                action: edit_action::INTERACT,
                eye: dvec3_from_vec3(hit.eye_local),
                look: dvec3_from_vec3(hit.look_local),
                block_type: 0,
            },
            Some(hit.shard),
            &primary,
        );
    }

    // (F key handled earlier in the top-level branch — applies
    // whether or not the crosshair is on a target.)

    // 1-9 = hotbar slots. Phase 20's inventory will drive this
    // properly; for now 1-9 just pick arbitrary low block IDs so the
    // user can vary placement.
    //
    // Layout intentionally uses consecutive IDs so a developer
    // connecting to a fresh cluster can place different blocks without
    // waiting for the inventory UI.
    for (key, id) in [
        (KeyCode::Digit1, 1u16),
        (KeyCode::Digit2, 2),
        (KeyCode::Digit3, 3),
        (KeyCode::Digit4, 4),
        (KeyCode::Digit5, 5),
    ] {
        if keys.just_pressed(key) {
            hotbar.block_type = id;
            tracing::info!(block_type = id, "hotbar: slot selected");
        }
    }
}

fn send_edit(
    tcp: &TcpSender,
    edit: BlockEditData,
    hit_shard: Option<crate::shard::ShardKey>,
    primary: &PrimaryShard,
) {
    // Route to the owning shard's TCP. MVP: single primary TCP; if the
    // hit is on a secondary, log once and fall through to primary. When
    // server-side observer writes land, extend `TcpSender` to a map
    // keyed by ShardKey and dispatch here.
    if let (Some(hit), Some(cur)) = (hit_shard, primary.current) {
        if hit != cur {
            // Shard-routing mismatch: the edit is for a secondary, but
            // we only have a primary-TCP write channel today. Sending
            // via primary is a no-op server-side (primary doesn't own
            // the target) but avoids client-side dropping the event.
            tracing::warn!(
                %hit, %cur,
                "cross-shard edit routed via primary TCP (server observer-write TODO)",
            );
        }
    }
    // TCP framing: `ClientMsg::serialize` produces the inner flatbuffers
    // payload; `wire_codec::encode` wraps it in the length-prefixed
    // frame the server's TCP reader expects. Sending the raw payload
    // without the wrapper makes the server treat the bytes as garbage
    // and drop the TCP — same framing legacy voxydust uses.
    let msg = ClientMsg::BlockEditRequest(edit);
    let data = msg.serialize();
    let mut pkt = Vec::new();
    wire_codec::encode(&data, &mut pkt);
    if tcp.tx.send(pkt).is_err() {
        tracing::warn!("TCP channel closed while sending BlockEditRequest");
    }
}

fn dvec3_from_vec3(v: glam::Vec3) -> glam::DVec3 {
    glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

/// Send a `SubBlockEditRequest` routed to the owning shard's TCP.
/// Matches `send_edit`'s primary-only fallback semantics.
fn send_sub_block_edit(
    tcp: &TcpSender,
    edit: SubBlockEditData,
    hit_shard: Option<crate::shard::ShardKey>,
    primary: &PrimaryShard,
) {
    if let (Some(hit), Some(cur)) = (hit_shard, primary.current) {
        if hit != cur {
            tracing::warn!(
                %hit, %cur,
                "cross-shard sub-block edit routed via primary TCP (server observer-write TODO)",
            );
        }
    }
    tracing::info!(
        block = ?(edit.block_pos.x, edit.block_pos.y, edit.block_pos.z),
        face = edit.face,
        element_type = edit.element_type,
        rotation = edit.rotation,
        action = edit.action,
        "sub-block edit sent",
    );
    let msg = ClientMsg::SubBlockEdit(edit);
    let data = msg.serialize();
    let mut pkt = Vec::new();
    wire_codec::encode(&data, &mut pkt);
    if tcp.tx.send(pkt).is_err() {
        tracing::warn!("TCP channel closed while sending SubBlockEditRequest");
    }
}

/// Convert a face-normal vector to the 0–5 face id the server + the
/// sub-block mesher use: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5.
fn face_normal_to_face_u8(n: IVec3) -> u8 {
    if n.x > 0 {
        0
    } else if n.x < 0 {
        1
    } else if n.y > 0 {
        2
    } else if n.y < 0 {
        3
    } else if n.z > 0 {
        4
    } else {
        5
    }
}
