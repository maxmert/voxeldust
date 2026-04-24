//! Client → server block interactions over the reliable TCP channel.
//!
//! Scope: the input actions whose authoritative handling always runs
//! server-side via its own raycast against the ship grid. The client
//! sends an `eye` origin + `look` direction (reused verbatim from the
//! `BlockTarget` computed by `block_raycast`) and the server validates,
//! re-raycasts, and applies or rejects the edit.
//!
//! Ported from legacy voxydust (`voxydust/src/main.rs:3739-3877`). The
//! server-side handler is in `ship-shard/src/main.rs:3700-3870` —
//! BREAK, PLACE, INTERACT (and sub-enums through 7), OPEN_CONFIG, and
//! EXIT_SEAT all flow through the single `BlockEditRequest` TCP path.
//!
//! ## Bindings (match legacy + feedback_hud_interaction_pattern)
//!
//! | Input          | Action                          | Notes                               |
//! |----------------|---------------------------------|-------------------------------------|
//! | LMB            | BREAK (damage → destroy block)  | Progressive damage; server decides  |
//! | RMB            | PLACE (on targeted face)        | Uses `SelectedBlock` for block_type |
//! | MMB            | pick block (local only)         | Sets `SelectedBlock = target.type`  |
//! | E              | INTERACT                        | Seat entry, button, door use        |
//! | F (piloting)   | EXIT_SEAT                       | Stand up from seat                  |
//! | F (walking)    | OPEN_CONFIG                     | Functional-block config panel (#20) |
//!
//! ## Server-authoritative contract
//!
//! `BlockEditRequest { action, eye, look, block_type }`:
//! * `eye/look` in **shard-local coordinates** (same frame as
//!   `BlockTarget.eye/look`). The server runs its own DDA raycast
//!   against the authoritative `ShipGrid` — the client's `BlockTarget`
//!   is used only for the highlight cube and to gate actions to the
//!   visible target; the server re-raycasts for correctness.
//! * `block_type` only meaningful for PLACE / PLACE_SUB.
//! * EXIT_SEAT passes `eye/look = 0` because the handler skips the
//!   raycast (ship-shard:3619-3631).
//!
//! ## Multiplayer
//!
//! Each client dispatches independently. Server orders edits by
//! arrival and applies them atomically to the shared grid — conflicts
//! (e.g. two players simultaneously breaking the same block) are
//! resolved in edit order, and the server re-broadcasts the resulting
//! ChunkDelta to every connected client. No cross-client state here.

use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use glam::DVec3;
use voxeldust_core::block::BlockId;
use voxeldust_core::client_message::{action, shard_type as st, BlockEditData, ClientMsg};

use crate::block_raycast::{BlockRaycastSet, BlockTarget};
use crate::net_plugin::{NetConnection, TcpSender};

pub struct BlockInteractionPlugin;

impl Plugin for BlockInteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SelectedBlock>()
            .add_systems(Update, dispatch_interactions.after(BlockRaycastSet));
    }
}

/// The block type currently armed for placement. MMB on a block copies
/// that block's type here; RMB places a block of this type. Defaults to
/// Stone (id=1) — a sensible starter for build tests. When the
/// inventory / hotbar UI lands (#20) this gets driven from the slot
/// selection instead of MMB pick.
///
/// Server-authoritative: the client only proposes the type; the server
/// validates the player has it in inventory (future work — no
/// inventory enforcement yet).
#[derive(Resource, Debug, Clone, Copy)]
pub struct SelectedBlock(pub BlockId);

impl Default for SelectedBlock {
    fn default() -> Self {
        Self(BlockId::from_u16(1)) // Stone
    }
}

/// Emit `BlockEditRequest` over TCP for mouse + E/F keypresses. Runs
/// `.after(BlockRaycastSet)` so `BlockTarget` is fresh.
fn dispatch_interactions(
    conn: Res<NetConnection>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    target: Res<BlockTarget>,
    mut selected: ResMut<SelectedBlock>,
    tcp: Res<TcpSender>,
    cursors: Query<&bevy::window::CursorOptions, With<bevy::window::PrimaryWindow>>,
) {
    if !conn.connected {
        return;
    }
    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != bevy::window::CursorGrabMode::None)
        .unwrap_or(false);
    if !grabbed {
        return;
    }

    // Per-shard gate: block interactions only valid on SHIP today.
    // PLANET surface building will extend this to shard_type == PLANET
    // when the planet edit path lands.
    if conn.shard_type != st::SHIP {
        return;
    }

    let lmb = mouse.just_pressed(MouseButton::Left);
    let rmb = mouse.just_pressed(MouseButton::Right);
    let mmb = mouse.just_pressed(MouseButton::Middle);
    let e = keys.just_pressed(KeyCode::KeyE);
    let f = keys.just_pressed(KeyCode::KeyF);
    if !lmb && !rmb && !mmb && !e && !f {
        return;
    }
    tracing::info!(
        lmb,
        rmb,
        mmb,
        e,
        f,
        has_target = target.hit.is_some(),
        selected = selected.0.as_u16(),
        "block_interaction input detected"
    );

    // Eye + look come straight from the raycast resource. They describe
    // the same ray the highlight cube is drawn from, so the server's
    // own raycast lands on the same block the player sees targeted.
    let eye = DVec3::new(
        target.eye.x as f64,
        target.eye.y as f64,
        target.eye.z as f64,
    );
    let look = DVec3::new(
        target.look.x as f64,
        target.look.y as f64,
        target.look.z as f64,
    );

    // LMB — BREAK. Server damages the block; multiple LMBs over time
    // progress damage stages until the block is destroyed. No
    // block_type field needed (server reads the type from the grid).
    if lmb {
        send_edit(
            &tcp,
            BlockEditData {
                action: action::BREAK,
                eye,
                look,
                block_type: 0,
            },
            "BREAK",
        );
    }

    // RMB — PLACE. Server places `block_type` on the face the ray
    // entered through. Uses `SelectedBlock` for the type; will be
    // driven by hotbar/inventory when #20 lands.
    if rmb {
        send_edit(
            &tcp,
            BlockEditData {
                action: action::PLACE,
                eye,
                look,
                block_type: selected.0.as_u16(),
            },
            "PLACE",
        );
    }

    // MMB — pick block. Local-only: copies the targeted block's type
    // into `SelectedBlock`. Matches Minecraft-style middle-click pick.
    // Requires a valid `BlockTarget` (otherwise nothing to pick).
    if mmb {
        if let Some(hit) = target.hit.as_ref() {
            // Skip picking air — nothing useful to select.
            if !hit.block_type.is_air() {
                selected.0 = hit.block_type;
                tracing::info!(
                    block_type = hit.block_type.as_u16(),
                    "picked block type from target"
                );
            }
        }
    }

    // E — INTERACT. Server raycasts, finds functional block (seat,
    // door, button, console), dispatches per-block interaction schema.
    // `eye/look` matter here because the server uses them to find the
    // targeted block.
    if e {
        send_edit(
            &tcp,
            BlockEditData {
                action: action::INTERACT,
                eye,
                look,
                block_type: 0,
            },
            "INTERACT",
        );
    }

    // F — EXIT_SEAT. Unconditional: the server's handler no-ops when
    // not seated (`ship-shard:3623`), so the client-side `is_piloting`
    // race doesn't matter. OPEN_CONFIG (the "F while walking" branch)
    // is deferred to #20 when the config panel UI lands — mapping F
    // to two different actions without a reliable client-side state
    // flag is fragile, so we pick EXIT_SEAT (the more common intent
    // today) and move OPEN_CONFIG to a different key when the UI is
    // ready.
    if f {
        send_edit(
            &tcp,
            BlockEditData {
                action: action::EXIT_SEAT,
                eye: DVec3::ZERO,
                look: DVec3::ZERO,
                block_type: 0,
            },
            "EXIT_SEAT",
        );
    }
}

fn send_edit(tcp: &TcpSender, edit: BlockEditData, tag: &'static str) {
    // Wire framing: server expects length-prefixed packets
    // (`wire_codec::encode` = 4-byte BE length header + zstd-wrapped
    // flatbuffer payload). Sending raw `serialize()` bytes without the
    // framing makes the server read the flatbuffer's first 4 bytes as a
    // bogus packet length, try to allocate / read that many bytes, fail,
    // and close the TCP connection — observed as "primary TCP read error
    // — early eof" the instant an edit was sent. Mirrors `send_msg` in
    // network.rs which handles the same framing for `PlayerName` etc.
    let payload = ClientMsg::BlockEditRequest(edit).serialize();
    let mut framed = Vec::with_capacity(payload.len() + 4);
    voxeldust_core::wire_codec::encode(&payload, &mut framed);
    if tcp.tx.send(framed).is_err() {
        tracing::warn!("TCP channel closed — dropped {tag} BlockEditRequest");
    } else {
        tracing::info!(action = tag, "sent BlockEditRequest");
    }
}
