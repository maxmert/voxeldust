//! Server-authoritative player input pipeline.
//!
//! One `PlayerInputData` is assembled per frame and handed to the
//! network thread (`InputSender`), which coalesces to 20 Hz before UDP
//! send. Game-mechanical semantics live on the server — this module
//! only:
//!
//! - Accumulates local look (mouse motion → yaw + pitch) for
//!   responsive camera feel.
//! - Keeps a parallel *server yaw/pitch* snapshot so free-look (Alt)
//!   pans the camera without the server rotating the avatar / ship.
//! - Derives pilot-mode torque rates from mouse motion with a virtual-
//!   spring decay so "hands off the yoke" centers the ship naturally.
//! - Tracks ship-control state the server broadcasts authoritative
//!   replies for: speed tier (keys 1–5), thrust limiter (scroll wheel),
//!   boolean stance bits (sprint / crouch).
//!
//! Seated state comes from `WorldState.players[own_id].seated` —
//! shard-agnostic (turret seats on planets emit seat_values the same
//! way as cockpit seats on ships). See Design Principle #9.
//!
//! Ported from `voxydust-next/src/input_system.rs` with module-path
//! adaptations. Per-shard-type camera frame math layers on top via
//! Phase 6+ trait methods; this module is deliberately shard-agnostic.

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use voxeldust_core::client_message::{input_action_bits, PlayerInputData};

use crate::config::GameConfig;
use crate::net::{GameEvent, InputSender, NetConnection, NetEvent, NetworkBridgeSet};
use crate::seat::SeatValues;
use crate::shard::ShardTransitionSet;

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InputSet;

#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SendInputSet;

// ─────────────────────────────────────────────────────────────────────
// Non-user-facing constants
// ─────────────────────────────────────────────────────────────────────
//
// Values a player might tune live in `config::ControlConfig`. Values
// below are engineering invariants (physics / wire protocol / server
// contract) that don't change through gameplay.

/// Exponential decay for pilot-mode rates (reciprocal seconds).
/// `exp(-PILOT_RATE_DECAY · dt)` per frame → rates decay to zero in
/// ~100 ms after input stops. Frame-rate independent.
pub const PILOT_RATE_DECAY: f32 = 9.75;

/// Maximum absolute rate per axis (pilot mode) — the wire contract
/// with the server caps rates at `[-1, 1]`.
pub const PILOT_RATE_CLAMP: f32 = 1.0;

/// Number of speed tiers (keys 1–5 map to tiers 0–`SPEED_TIER_MAX`).
pub const SPEED_TIER_MAX: u8 = 4;

/// Default thrust-limiter on fresh connect.
pub const THRUST_LIMITER_DEFAULT: f32 = 0.75;

/// Pitch clamp to prevent view flip at poles (0.98 · π/2).
pub const PITCH_CLAMP: f32 = std::f32::consts::FRAC_PI_2 * 0.98;

/// First-person camera eye-offset above the player's origin along the
/// character's local up axis. The server's `StanceCapsule::Standing`
/// is 0.6 cylindrical middle + 0.3 radii caps = 1.2 m total with
/// position at the capsule center — so an eye offset of 1.0 places
/// the view just above the top of the capsule, giving the player a
/// human-scale perspective over 1 m blocks. Crouch / prone stance
/// reductions would lower this proportionally (deferred until the
/// stance system is wired client-side).
pub const EYE_HEIGHT: f32 = 1.0;

// ─────────────────────────────────────────────────────────────────────
// Resources
// ─────────────────────────────────────────────────────────────────────

/// View + server look angles. `yaw`/`pitch` drive the camera; `server_*`
/// tracks the avatar/ship frame sent to the server. They move together
/// unless free-look (Alt) is engaged, at which point server_* freezes.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct LocalLook {
    pub yaw: f32,
    pub pitch: f32,
    pub server_yaw: f32,
    pub server_pitch: f32,
    pub free_look: bool,
    pub snapshot_seeded: bool,
}

/// Pilot-mode torque rates in normalized [-1, 1] units. Accumulated
/// from mouse motion when seated; decayed each frame.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct PilotRates {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

/// Ship-control state (speed tier + thrust limiter). Server-authoritative
/// for what each tier/limiter actually does mechanically.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ShipControlsState {
    pub speed_tier: u8,
    pub thrust_limiter: f32,
}

impl Default for ShipControlsState {
    fn default() -> Self {
        Self {
            speed_tier: 0,
            thrust_limiter: THRUST_LIMITER_DEFAULT,
        }
    }
}

/// Seated flag. Server-authoritative via `WorldState.players[own_id].seated`.
/// Shard-agnostic: planet-turret seat and ship-cockpit seat both set this.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct SeatedState {
    pub is_seated: bool,
}

/// Monotonic tick counter for `PlayerInputData.tick`.
#[derive(Resource, Default)]
pub struct InputTickCounter(pub u64);

/// Aggregated mouse delta for THIS frame (sum across all MouseMotion
/// events). Populated once per frame by `collect_mouse_delta` so every
/// downstream consumer (camera look, seat bindings) reads a single
/// authoritative value instead of racing their own `MessageReader`
/// cursors. Also aggregates scroll wheel for the same reason.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct FrameMouseDelta {
    pub dx: f32,
    pub dy: f32,
    pub scroll_y: f32,
}

// ─────────────────────────────────────────────────────────────────────
// Plugin
// ─────────────────────────────────────────────────────────────────────

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LocalLook>()
            .init_resource::<PilotRates>()
            .init_resource::<ShipControlsState>()
            .init_resource::<SeatedState>()
            .init_resource::<InputTickCounter>()
            .init_resource::<FrameMouseDelta>()
            .configure_sets(
                Update,
                InputSet
                    .after(NetworkBridgeSet)
                    .after(ShardTransitionSet),
            )
            .add_systems(
                Update,
                (
                    // `collect_mouse_delta` drains MouseMotion +
                    // MouseWheel events EXACTLY ONCE into a shared
                    // Resource; every downstream consumer (camera look,
                    // seat bindings, focus-mode cursor) reads the
                    // Resource instead of its own MessageReader. Single
                    // source of truth — impossible for one consumer to
                    // "steal" events from another.
                    collect_mouse_delta,
                    track_seated_state,
                    decay_pilot_rates,
                    accumulate_mouse_look,
                    update_speed_tier,
                    update_thrust_limiter,
                )
                    .chain()
                    .in_set(InputSet),
            )
            .configure_sets(
                Update,
                SendInputSet.after(InputSet).after(ShardTransitionSet),
            )
            .add_systems(Update, build_and_send_input.in_set(SendInputSet));
    }
}

// ─────────────────────────────────────────────────────────────────────
// Systems
// ─────────────────────────────────────────────────────────────────────

/// Track server-authoritative seated state. Source of truth:
/// `WorldStateData.players[own_id].seated`.
fn track_seated_state(
    mut events: MessageReader<GameEvent>,
    conn: Res<NetConnection>,
    mut seated: ResMut<SeatedState>,
) {
    for GameEvent(ev) in events.read() {
        let NetEvent::WorldState(ws) = ev else { continue };
        let seated_now = ws
            .players
            .iter()
            .find(|p| p.player_id == conn.player_id)
            .map(|p| p.seated)
            .unwrap_or(false);
        if seated.is_seated != seated_now {
            seated.is_seated = seated_now;
            tracing::info!(
                seated = seated_now,
                player_id = conn.player_id,
                "seated state changed from server",
            );
        }
    }
}

fn decay_pilot_rates(time: Res<Time>, mut rates: ResMut<PilotRates>) {
    let decay = (-PILOT_RATE_DECAY * time.delta_secs()).exp();
    rates.yaw *= decay;
    rates.pitch *= decay;
    rates.roll *= decay;
}

/// Drain MouseMotion + MouseWheel events into `FrameMouseDelta` once
/// per frame. Runs first in the `InputSet` chain; consumers
/// (camera look, seat bindings, future focus cursor) read the
/// resource instead of their own `MessageReader`.
fn collect_mouse_delta(
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut delta: ResMut<FrameMouseDelta>,
) {
    delta.dx = 0.0;
    delta.dy = 0.0;
    delta.scroll_y = 0.0;
    for ev in motion.read() {
        delta.dx += ev.delta.x;
        delta.dy += ev.delta.y;
    }
    for ev in wheel.read() {
        delta.scroll_y += ev.y;
    }
}

fn accumulate_mouse_look(
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    cfg: Res<GameConfig>,
    seated: Res<SeatedState>,
    delta: Res<FrameMouseDelta>,
    hud_focus: Res<crate::hud::HudFocusState>,
    mut look: ResMut<LocalLook>,
    mut rates: ResMut<PilotRates>,
) {
    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !grabbed {
        return;
    }
    // When HUD focus is active, the mouse drives the in-world cursor
    // on the focused tile — ship / camera must NOT rotate, otherwise
    // the player loses orientation when clicking a button. The focus
    // module itself consumes the delta; we just bail here.
    if hud_focus.active {
        return;
    }

    let alt_held = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);
    // Free-look edge detection (entry / exit snapshots).
    if alt_held && !look.free_look {
        look.server_yaw = look.yaw;
        look.server_pitch = look.pitch;
    }
    if !alt_held && look.free_look {
        look.yaw = look.server_yaw;
        look.pitch = look.server_pitch;
    }
    look.free_look = alt_held;

    let dx_rad = delta.dx * cfg.control.mouse_sensitivity;
    let dy_rad = delta.dy * cfg.control.mouse_sensitivity;
    if dx_rad == 0.0 && dy_rad == 0.0 {
        return;
    }

    // State-machine branch:
    //   SEATED + !Alt  → mouse feeds PilotRates ONLY.
    //                    LocalLook stays frozen (camera forward-locks
    //                    to ship via the camera rotation composer).
    //                    **Mouse-invert flags apply here** — this is
    //                    flight-sim control.
    //   SEATED + Alt   → free-look: mouse pans the camera via
    //                    LocalLook without moving the ship. Uses
    //                    non-inverted (walking) direction so the
    //                    pilot can glance around the cockpit
    //                    intuitively even if they prefer inverted
    //                    flight controls.
    //   WALKING        → mouse pans the camera + sets server_yaw/pitch
    //                    so the server sees the player's head angle.
    //                    Non-inverted (FPS convention).
    if seated.is_seated && !alt_held {
        let inv_x = if cfg.control.mouse_invert_x { -1.0_f32 } else { 1.0_f32 };
        let inv_y = if cfg.control.mouse_invert_y { -1.0_f32 } else { 1.0_f32 };
        rates.yaw = (rates.yaw - inv_x * dx_rad * cfg.control.pilot_rate_gain)
            .clamp(-PILOT_RATE_CLAMP, PILOT_RATE_CLAMP);
        rates.pitch = (rates.pitch - inv_y * dy_rad * cfg.control.pilot_rate_gain)
            .clamp(-PILOT_RATE_CLAMP, PILOT_RATE_CLAMP);
        // LocalLook stays frozen — camera forward-locks to ship.
        return;
    }

    // Walking / free-look: non-inverted natural feel.
    look.yaw += dx_rad;
    look.pitch = (look.pitch - dy_rad).clamp(-PITCH_CLAMP, PITCH_CLAMP);
    if !alt_held {
        look.server_yaw = look.yaw;
        look.server_pitch = look.pitch;
    }
}

fn update_speed_tier(
    keys: Res<ButtonInput<KeyCode>>,
    mut controls: ResMut<ShipControlsState>,
) {
    const KEY_TO_TIER: &[(KeyCode, u8)] = &[
        (KeyCode::Digit1, 0),
        (KeyCode::Digit2, 1),
        (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3),
        (KeyCode::Digit5, 4),
    ];
    for &(key, tier) in KEY_TO_TIER {
        if tier > SPEED_TIER_MAX {
            break;
        }
        if keys.just_pressed(key) {
            controls.speed_tier = tier;
            return;
        }
    }
}

fn update_thrust_limiter(
    delta: Res<FrameMouseDelta>,
    cfg: Res<GameConfig>,
    mut controls: ResMut<ShipControlsState>,
) {
    if delta.scroll_y == 0.0 {
        return;
    }
    controls.thrust_limiter =
        (controls.thrust_limiter + delta.scroll_y * cfg.control.scroll_thrust_step)
            .clamp(0.0, 1.0);
}

fn build_and_send_input(
    conn: Res<NetConnection>,
    keys: Res<ButtonInput<KeyCode>>,
    look: Res<LocalLook>,
    rates: Res<PilotRates>,
    controls: Res<ShipControlsState>,
    seated: Res<SeatedState>,
    seat_values: Res<SeatValues>,
    sender: Res<InputSender>,
    hud_focus: Res<crate::hud::HudFocusState>,
    mut tick: ResMut<InputTickCounter>,
) {
    if !conn.connected {
        return;
    }

    // Movement: zeroed when seated (server routes inputs through
    // seat_values instead of walking movement); zeroed also while
    // tablet cursor mode is active so WASD typed into a text field
    // doesn't move the player / ship. Standard WASD + Space/Ctrl for
    // EVA/fly otherwise.
    let movement = if seated.is_seated || hud_focus.active {
        Vec3::ZERO
    } else {
        let mut m = Vec3::ZERO;
        if keys.pressed(KeyCode::KeyW) {
            m.z += 1.0;
        }
        if keys.pressed(KeyCode::KeyS) {
            m.z -= 1.0;
        }
        if keys.pressed(KeyCode::KeyD) {
            m.x += 1.0;
        }
        if keys.pressed(KeyCode::KeyA) {
            m.x -= 1.0;
        }
        if keys.pressed(KeyCode::Space) {
            m.y += 1.0;
        }
        if keys.pressed(KeyCode::ControlLeft) {
            m.y -= 1.0;
        }
        if m.length_squared() > 1.0 {
            m = m.normalize();
        }
        m
    };

    let mut actions_bits: u32 = 0;
    if keys.pressed(KeyCode::ShiftLeft) {
        actions_bits |= input_action_bits::SPRINT;
    }
    if keys.pressed(KeyCode::KeyC) {
        actions_bits |= input_action_bits::CROUCH;
    }

    // Look: absolute yaw/pitch in walking mode, damped rates when seated.
    let (look_yaw, look_pitch) = if seated.is_seated {
        (rates.yaw, rates.pitch)
    } else {
        (look.server_yaw, look.server_pitch)
    };

    tick.0 = tick.0.wrapping_add(1);
    let input = PlayerInputData {
        movement: [movement.x, movement.y, movement.z],
        look_yaw,
        look_pitch,
        jump: keys.just_pressed(KeyCode::Space),
        fly_toggle: keys.just_pressed(KeyCode::KeyF),
        orbit_stabilizer_toggle: keys.just_pressed(KeyCode::KeyO),
        speed_tier: controls.speed_tier,
        // EXIT_SEAT is a BlockEdit (Phase 15); `action` stays 0 here.
        action: 0,
        // Hotbar-driven block_type is Phase 20 (inventory); 0 means air.
        block_type: 0,
        tick: tick.0,
        thrust_limiter: controls.thrust_limiter,
        roll: 0.0,
        cruise: false,
        atmo_comp: false,
        seat_values: seat_values.0.clone(),
        actions_bits,
    };

    if sender.tx.send(input).is_err() {
        static LAST_WARNED: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = LAST_WARNED.load(std::sync::atomic::Ordering::Relaxed);
        if now_ms.saturating_sub(last) > 1000 {
            LAST_WARNED.store(now_ms, std::sync::atomic::Ordering::Relaxed);
            tracing::warn!("input channel closed — network thread may have died");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers — forward vector / rotation from (yaw, pitch).
//
// Matches the server convention: yaw=0 means forward along +X,
// positive yaw rotates toward +Z, pitch is elevation from horizon.
// Used by the per-shard-type camera-frame math when it lands.
// ─────────────────────────────────────────────────────────────────────

pub fn forward_from_look(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(cy * cp, sp, sy * cp)
}

pub fn rotation_from_look(yaw: f32, pitch: f32) -> Quat {
    let fwd = forward_from_look(yaw, pitch);
    Transform::IDENTITY.looking_to(fwd, Vec3::Y).rotation
}
