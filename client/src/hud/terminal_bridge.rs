//! Bridge between the server-authored Terminal protocol and the
//! client-side `WidgetKind::Terminal` state on a HUD subblock.
//!
//! Responsibilities:
//!
//! 1. **Engage on `OpenTerminalChat`** — server confirms the player
//!    pressed E on a Terminal block; we look up the HUD tile on its
//!    front face, hydrate the tile's `TerminalWidgetState` with the
//!    current channels + recent scrollback, and engage HUD focus on
//!    that tile (which kicks off the camera-focus animation).
//! 2. **Append on `TerminalScrollbackDelta`** — push appended lines
//!    onto every matching Terminal tile's scrollback. This runs
//!    regardless of focus so the on-block display stays live for
//!    bystanders.
//!
//! The Terminal block's outward face today is +Z (face 4) on the
//! starter ship. The bridge looks up the tile by `(block_pos,
//! face=TERMINAL_FACE)`. If there are multiple shards with a
//! Terminal at the same `block_pos` (cross-shard collision), the
//! first match wins — an edge case that's not in scope.

use bevy::prelude::*;

use voxeldust_core::client_message::{OpenTerminalChatData, TerminalScrollbackDeltaData};

use crate::hud::focus::{find_block_face_tile_any_shard, HudFocusState};
use crate::hud::tile::{HudConfig, HudTile, WidgetKind};
use crate::hud::widget::HudWidgetState;
use crate::hud::widgets::terminal::{TerminalWidgetState, MAX_SCROLLBACK};
use crate::net::{GameEvent, NetEvent};

/// Convert the wire-format `glam::IVec3` (workspace pin v0.29) into
/// the Bevy-imported `bevy::prelude::IVec3` (which is glam v0.30
/// re-exported). Same field layout; field-by-field copy is safe and
/// the only stable API across the version split.
fn ivec3_from_wire(v: glam::IVec3) -> IVec3 {
    IVec3::new(v.x, v.y, v.z)
}

/// Outward-facing front of a Terminal block — face 4 (+Z) in the
/// face-index convention used by `block_tile::face_outward_normal`.
/// The starter ship's Terminal is placed with this face exposed.
pub const TERMINAL_FACE: u8 = 4;

pub struct TerminalBridgePlugin;

impl Plugin for TerminalBridgePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (ingest_open_chat, ingest_scrollback_delta));
    }
}

/// Drain `NetEvent::OpenTerminalChat`. For each event:
///   1. Locate the HUD tile entity matching the Terminal block's
///      front face. If the tile hasn't spawned yet (chunk not yet
///      meshed in this view), drop the event — server resends or
///      the player can re-press E.
///   2. Promote the tile's widget kind to `Terminal` if not already
///      (in case the player previously reconfigured it to something
///      else). The texture pipeline's `sync_hud_widget_state`
///      installs a fresh `TerminalWidgetState` next frame.
///   3. Hydrate the per-tile state with channels + scrollback +
///      block_pos.
///   4. Engage HUD focus on the tile — camera-focus animation
///      animates in.
fn ingest_open_chat(
    mut events: MessageReader<GameEvent>,
    mut focus: ResMut<HudFocusState>,
    mut commands: Commands,
    tiles: Query<(Entity, &HudTile)>,
    mut config_q: Query<&mut HudConfig>,
    mut state_q: Query<&mut HudWidgetState>,
) {
    for ev in events.read() {
        let NetEvent::OpenTerminalChat(data) = &ev.0 else {
            continue;
        };
        let block_pos = ivec3_from_wire(data.block_pos);
        let Some(tile) = find_block_face_tile_any_shard(
            block_pos,
            TERMINAL_FACE,
            &tiles,
        ) else {
            tracing::warn!(
                block = ?(block_pos.x, block_pos.y, block_pos.z),
                face = TERMINAL_FACE,
                "OpenTerminalChat: no HudTile found for Terminal face — chunk may not be meshed yet"
            );
            continue;
        };

        // Make sure this tile is configured as a Terminal widget.
        if let Ok(mut config) = config_q.get_mut(tile) {
            if config.kind != WidgetKind::Terminal {
                config.kind = WidgetKind::Terminal;
                // Touch causes change detection so
                // `sync_hud_widget_state` installs a fresh state.
                config.set_changed();
            }
        }

        // Hydrate per-tile state. If the state component already
        // exists (kind didn't change), update it in place; if not,
        // queue an insert with hydrated state — the
        // `sync_hud_widget_state` system would install a default
        // `TerminalWidgetState` next frame; we beat it with the
        // populated version so the player sees scrollback
        // immediately on engage.
        if let Ok(mut state) = state_q.get_mut(tile) {
            if let Some(ts) = state
                .data
                .as_any_mut()
                .downcast_mut::<TerminalWidgetState>()
            {
                hydrate_terminal_state(ts, data);
            } else {
                // Wrong concrete type — replace whole state.
                let mut fresh = TerminalWidgetState::default();
                hydrate_terminal_state(&mut fresh, data);
                state.kind = WidgetKind::Terminal;
                state.data = Box::new(fresh);
            }
        } else {
            let mut fresh = TerminalWidgetState::default();
            hydrate_terminal_state(&mut fresh, data);
            commands.entity(tile).insert(HudWidgetState {
                kind: WidgetKind::Terminal,
                data: Box::new(fresh),
            });
        }

        // Engage focus → triggers camera lerp.
        focus.engage_block_tile(tile);
        tracing::info!(
            block = ?(data.block_pos.x, data.block_pos.y, data.block_pos.z),
            "engaged Terminal HUD subblock"
        );
    }
}

/// Append scrollback delta lines into every Terminal tile matching
/// the broadcast `block_pos`. Runs regardless of focus so the
/// always-visible HUD readout stays current.
fn ingest_scrollback_delta(
    mut events: MessageReader<GameEvent>,
    tiles: Query<(Entity, &HudTile)>,
    mut state_q: Query<&mut HudWidgetState>,
) {
    for ev in events.read() {
        let NetEvent::TerminalScrollbackDelta(data) = &ev.0 else {
            continue;
        };
        let block_pos = ivec3_from_wire(data.block_pos);
        let Some(tile) = find_block_face_tile_any_shard(
            block_pos,
            TERMINAL_FACE,
            &tiles,
        ) else {
            continue;
        };
        let Ok(mut state) = state_q.get_mut(tile) else {
            continue;
        };
        let Some(ts) = state
            .data
            .as_any_mut()
            .downcast_mut::<TerminalWidgetState>()
        else {
            continue;
        };
        append_scrollback(ts, data);
    }
}

fn hydrate_terminal_state(state: &mut TerminalWidgetState, data: &OpenTerminalChatData) {
    state.subscribe_channel = data.subscribe_channel.clone();
    state.publish_channel = data.publish_channel.clone();
    state.block_pos = ivec3_from_wire(data.block_pos);
    state.scrollback = data.recent_lines.clone();
    truncate_scrollback(&mut state.scrollback);
}

fn append_scrollback(state: &mut TerminalWidgetState, data: &TerminalScrollbackDeltaData) {
    state.scrollback.extend(data.appended_lines.iter().cloned());
    truncate_scrollback(&mut state.scrollback);
}

fn truncate_scrollback(lines: &mut Vec<String>) {
    if lines.len() > MAX_SCROLLBACK {
        let drop_n = lines.len() - MAX_SCROLLBACK;
        lines.drain(0..drop_n);
    }
}
