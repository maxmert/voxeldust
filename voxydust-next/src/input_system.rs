//! Server-authoritative player input pipeline.
//!
//! One `PlayerInputData` is assembled per frame and handed to the network
//! thread (`InputSender`), which coalesces to 20 Hz before UDP send. All
//! game-mechanical semantics live on the server — this module only:
//!
//! * Accumulates local look (mouse motion → yaw + pitch) for responsive
//!   camera feel.
//! * Keeps a parallel *server yaw/pitch* snapshot so free-look (Alt) can
//!   pan the camera without the server rotating the avatar / ship.
//! * Derives pilot-mode torque rates from mouse motion with a virtual-spring
//!   decay so "hands off the yoke" centers the ship naturally.
//! * Tracks ship-control state the server broadcasts authoritative replies
//!   for: speed tier (keys 1–5), thrust limiter (scroll wheel), boolean
//!   stance bits (sprint / crouch).
//!
//! Camera frame conversion (ship-local vs planet-tangent vs EVA body-frame
//! interpretation of yaw/pitch) is the camera task (#9). Seat bindings that
//! route mouse + key deltas into per-binding `seat_values` while piloting
//! are task #17. Both consumers read `LocalLook` / `PilotRates` /
//! `PilotState` from this module without modification.
//!
//! Multiplayer: entirely per-client. Each connected client runs its own
//! independent input pipeline against its own `InputSender`; nothing in
//! this module references other players. Crew piloting the same ship
//! reach the ship's input authority through the server's seat-bindings
//! broadcast, which will arrive via `SeatBindingsNotify` and flip
//! `PilotState.is_piloting` in #17 — again no cross-client state here.

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use voxeldust_core::client_message::{input_action_bits, PlayerInputData};

use crate::net_plugin::{GameEvent, InputSender, NetConnection};
use crate::network::NetEvent;
use crate::seat_bindings::SeatValues;
use crate::shard_transition::ShardTransitionSet;

/// System set for every input-side system. Camera-frame consumers order
/// themselves `.after(InputSet)` so the camera rotation they compute is
/// based on the mouse motion accumulated THIS frame — without this
/// ordering, `compute_camera_frame` can run before `accumulate_mouse_look`
/// and the camera renders with one frame of input lag (visible as
/// "mouse doesn't work" on machines fast enough that a single frame lag
/// becomes a visible stutter rather than a soft smoothing).
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InputSet;

/// Tight system set around `build_and_send_input` so other plugins (notably
/// `seat_bindings::evaluate_seat_bindings`) can order themselves
/// `.before(SendInputSet)` and guarantee their writes to input-state
/// resources (`SeatValues`, `PilotRates`, …) land in the same-frame
/// outgoing packet. Without this set, build-and-send could run before
/// those writers each frame, producing a steady 50-ms lag on every UDP
/// tick the server receives.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SendInputSet;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------
//
// Every value below is a *tuning default*, exposed as a `Resource` field
// (`InputSettings`) for live override from future settings UI. Constants
// document the default and the reasoning.

/// Radians of yaw/pitch per pixel of mouse motion at the default
/// mouse-sensitivity setting. Matches the feel of legacy voxydust and
/// typical desktop-FPS values (`0.003` rad/px ≈ 18°/full-screen-width at
/// 3840-pixel wide screens, which maps a 180° head-turn to ≈ 0.5 m of
/// mouse travel on a common 1000 DPI mouse).
pub const MOUSE_SENSITIVITY_DEFAULT: f32 = 0.003;

/// Exponential decay coefficient for pilot-mode yaw/pitch rates, in
/// reciprocal seconds. `exp(-PILOT_RATE_DECAY · dt)` is applied each frame
/// so rates returned from mouse motion relax to zero in roughly 100 ms
/// after input stops. Matches the legacy `voxydust::input` tuning (see the
/// `decay` line of `apply_pilot_rate_decay`) which the user signed off on
/// as the SC-style "virtual spring" yoke feel.
pub const PILOT_RATE_DECAY: f32 = 9.75;

/// Mouse-delta → rate scaling (rate units per radian of mouse motion).
/// With the default mouse sensitivity, a full 180° head-turn of the
/// mouse (≈ π rad) drives the rate into the clamp: visible torque on a
/// noticeable mouse push, nothing on a nudge. The product of `PILOT_RATE_GAIN`
/// and `MOUSE_SENSITIVITY_DEFAULT` determines the effective pixel→rate
/// gain (`0.3 × 0.003 ≈ 0.001 rate/px`, i.e. 1000 px mouse travel
/// saturates one axis).
pub const PILOT_RATE_GAIN: f32 = 0.3;

/// Maximum absolute rate sent to the server per axis (pilot mode). The
/// server interprets values in `[-1, 1]` as normalized torque input, so
/// saturating here matches the wire-protocol's domain and avoids
/// sensitivity drift for users with fast mice.
pub const PILOT_RATE_CLAMP: f32 = 1.0;

/// Per-notch scroll-wheel change to the thrust limiter (0..1 slider).
/// Five full notches cover the 0–1 range, matching a "quarter-turn" wheel
/// gesture for a full thrust swing — enough feel without accidental full
/// throttle from single notches.
pub const SCROLL_THRUST_STEP: f32 = 0.2;

/// Number of speed tiers (keys `1`–`5` map to tiers 0–4). Defined as the
/// protocol's documented u8 range start to avoid drifting if the server
/// ever extends the tier count.
pub const SPEED_TIER_MAX: u8 = 4;

/// Default thrust-limiter value on fresh connect; matches legacy so
/// already-authored seat bindings (thrusters hooked to `thrust_limiter`)
/// behave identically to the old client.
pub const THRUST_LIMITER_DEFAULT: f32 = 0.75;

/// Pitch clamp to prevent view flip at the poles — 0.98 of π/2, same
/// bound the legacy client used, matches every mainstream FPS.
pub const PITCH_CLAMP: f32 = std::f32::consts::FRAC_PI_2 * 0.98;

/// Eye offset along the local up axis applied in `apply_server_pose`
/// (player_sync). First-person camera sits this far above the player's
/// origin. Kept here so input + camera modules share one source of truth
/// (camera is what renders, input is what sends — both need to agree on
/// what "look direction" means, which is anchored to this offset).
pub const EYE_HEIGHT: f32 = 0.5;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

#[derive(Resource, Debug, Clone)]
pub struct InputSettings {
    pub mouse_sensitivity: f32,
    pub mouse_invert_y: bool,
    pub scroll_thrust_step: f32,
    pub pilot_rate_gain: f32,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            mouse_sensitivity: MOUSE_SENSITIVITY_DEFAULT,
            mouse_invert_y: false,
            scroll_thrust_step: SCROLL_THRUST_STEP,
            pilot_rate_gain: PILOT_RATE_GAIN,
        }
    }
}

/// View + server look angles. `yaw`/`pitch` are the **view** (what the
/// camera renders); `server_yaw`/`server_pitch` are the latest orientation
/// the server should use for avatar / ship frame. They track together
/// unless free-look is engaged, at which point `server_*` freezes so
/// panning the view never rotates your body.
///
/// `snapshot_seeded` drives a one-shot seeding from the first
/// `WorldState` so we start with the orientation the server spawns us at,
/// not identity (otherwise the very first frame shows the world rotated
/// to whatever `Quat::IDENTITY` implies in the current shard frame).
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct LocalLook {
    pub yaw: f32,
    pub pitch: f32,
    pub server_yaw: f32,
    pub server_pitch: f32,
    pub free_look: bool,
    pub snapshot_seeded: bool,
}

/// Pilot-mode torque rates (in normalized [-1, 1] units). Accumulated from
/// mouse motion when `PilotState.is_piloting` is true; decayed every frame
/// via the virtual-spring formula. Consumed by `build_and_send_input`
/// when piloting — copied into `PlayerInputData.{look_yaw, look_pitch}`
/// in place of the absolute angles.
///
/// `roll` is tracked for future Q/E roll binding (legacy keeps the field
/// but sends 0 today; we do the same).
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct PilotRates {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

/// Ship-control state sent on every input tick. Both fields are edge-
/// updated by key / scroll events and otherwise held constant — the
/// server is authoritative for what "speed tier 2" or "thrust limiter
/// 0.6" actually does mechanically.
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

/// Piloting flag. Flipped to `true` by `SeatBindingsNotify` (task #17 wires
/// that handler; this file treats the field as read-only input). When
/// true, `build_and_send_input` sends torque rates instead of absolute
/// look angles and zeros the walking `movement` vector — the ship
/// controls (and the `seat_values` binding vector, populated in #17) take
/// over.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct PilotState {
    pub is_piloting: bool,
}

/// Monotonic counter for `PlayerInputData.tick`. Server uses it to order
/// and dedupe UDP packets. Simple wrapping add is sufficient — 2⁶⁴
/// frames is ~10¹⁰ years at 1 kHz, so the wraparound never shows up in
/// practice.
#[derive(Resource, Default)]
pub struct InputTickCounter(pub u64);

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InputSettings>()
            .init_resource::<LocalLook>()
            .init_resource::<PilotRates>()
            .init_resource::<ShipControlsState>()
            .init_resource::<PilotState>()
            .init_resource::<InputTickCounter>()
            // Pre-send pipeline: accumulate mouse, decay rates, read
            // keys, update tiers. Stays in InputSet so other plugins
            // can order themselves `.after(InputSet)` when they need
            // fresh values (camera frames, block raycast).
            .add_systems(
                Update,
                (
                    track_pilot_mode,
                    decay_pilot_rates,
                    accumulate_mouse_look,
                    update_speed_tier,
                    update_thrust_limiter,
                )
                    .chain()
                    .in_set(InputSet)
                    .after(ShardTransitionSet),
            )
            // Endpoint: the actual UDP packet assembly. Placed in its
            // own set so cross-plugin writers (`seat_bindings`) can
            // order themselves `.before(SendInputSet)` and guarantee
            // their resources (`SeatValues` etc.) are read fresh in
            // the same-frame outgoing packet.
            .configure_sets(
                Update,
                SendInputSet.after(InputSet).after(ShardTransitionSet),
            )
            .add_systems(Update, build_and_send_input.in_set(SendInputSet));
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Track server-authoritative piloting state from WorldState.
///
/// **Source of truth**: `WorldStateData.players[own_player_id].seated`
/// — the dedicated boolean field on the per-player snapshot, populated
/// authoritatively from the server's `SeatedState` component on every
/// tick. We specifically do NOT rely on `ObservableEntityData.kind`
/// for this: ship-shard only marks the own **ship** with `is_own=true`
/// (player entities always have `is_own=false`), and even the entity
/// kind itself has had recurring bugs where `EntityKind::Seated` gets
/// emitted regardless of state. The `.players[]` path matches on a
/// unique player_id and carries the real flag.
///
/// Writing to `PilotState.is_piloting` here keeps the client's input
/// routing, camera frame-lock, and F-key gating in lockstep with the
/// server's authoritative state — no guessing. Task #17 will
/// additionally populate `ActiveSeatBindings` when `SeatBindingsNotify`
/// arrives so `seat_values` can be produced on mouse/key input, but the
/// piloting *flag itself* is always derived from `.seated` here.
fn track_pilot_mode(
    mut events: MessageReader<GameEvent>,
    conn: Res<NetConnection>,
    mut pilot: ResMut<PilotState>,
) {
    for GameEvent(ev) in events.read() {
        let NetEvent::WorldState(ws) = ev else { continue };
        let seated_now = ws
            .players
            .iter()
            .find(|p| p.player_id == conn.player_id)
            .map(|p| p.seated)
            .unwrap_or(false);
        if pilot.is_piloting != seated_now {
            pilot.is_piloting = seated_now;
            tracing::info!(
                piloting = seated_now,
                player_id = conn.player_id,
                "pilot mode changed from server state"
            );
        }
    }
}

/// Virtual-spring decay of pilot-mode rates. Time-delta scaled so the
/// damping is frame-rate independent — essential at high-refresh displays
/// where naive per-frame multiplicative decay would relax rates orders of
/// magnitude faster at 240 Hz than at 60 Hz.
fn decay_pilot_rates(time: Res<Time>, mut rates: ResMut<PilotRates>) {
    let decay = (-PILOT_RATE_DECAY * time.delta_secs()).exp();
    rates.yaw *= decay;
    rates.pitch *= decay;
    rates.roll *= decay;
}

/// Ingest raw mouse motion into `LocalLook` and (when piloting) into
/// `PilotRates`. Handles free-look engage/release transitions. Only runs
/// when the cursor is grabbed — otherwise we are in a menu / debug focus
/// and must not steal motion events.
fn accumulate_mouse_look(
    mut mouse: MessageReader<MouseMotion>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<InputSettings>,
    pilot: Res<PilotState>,
    mut look: ResMut<LocalLook>,
    mut rates: ResMut<PilotRates>,
) {
    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !grabbed {
        mouse.clear();
        return;
    }

    // Free-look edge detection: snapshot server-look on engage so the
    // server keeps the avatar pointed where it was; snap view back to
    // server-look on release so the camera returns to the avatar's
    // forward direction (typical Star Citizen free-look behaviour).
    let alt_held = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);
    if alt_held && !look.free_look {
        look.server_yaw = look.yaw;
        look.server_pitch = look.pitch;
    }
    if !alt_held && look.free_look {
        look.yaw = look.server_yaw;
        look.pitch = look.server_pitch;
    }
    look.free_look = alt_held;

    // Mouse → yaw/pitch sign convention (matches legacy `main.rs:3594-3599`):
    //   dx > 0 (mouse right) → yaw += dx  (turn right = rotate from +X
    //                                       toward +Z, which is the
    //                                       server's "positive yaw"
    //                                       direction — see kcc_sys.rs)
    //   dy > 0 (mouse down)  → pitch -= dy (look up requires pitch↑,
    //                                       mouse-down feeling configured
    //                                       via mouse_invert_y).
    let invert = if settings.mouse_invert_y { -1.0_f32 } else { 1.0_f32 };
    // Diagnostic: accumulate raw mouse delta this frame and log every ~1s so
    // we can tell whether motion is reaching this system at all (vs.
    // being consumed by egui, blocked by cursor-grab, or never emitted
    // by the window). Rate-limited to avoid spam.
    let mut dx_sum = 0.0_f32;
    let mut dy_sum = 0.0_f32;
    let mut ev_count = 0u32;
    for ev in mouse.read() {
        dx_sum += ev.delta.x;
        dy_sum += ev.delta.y;
        ev_count += 1;
        let dx_rad = ev.delta.x * settings.mouse_sensitivity;
        let dy_rad = ev.delta.y * settings.mouse_sensitivity;
        // View always tracks mouse.
        look.yaw += dx_rad;
        look.pitch = (look.pitch - invert * dy_rad).clamp(-PITCH_CLAMP, PITCH_CLAMP);
        // Server-look and pilot rates only update when *not* free-looking.
        if !alt_held {
            look.server_yaw = look.yaw;
            look.server_pitch = look.pitch;
            if pilot.is_piloting {
                // Pilot rates: legacy feeds `-dx` / `-dy` with a 1.5 gain
                // factor (main.rs:3591-3592). Negative sign because the
                // server interprets positive yaw-rate as "torque the nose
                // right," which from the pilot's POV requires the mouse
                // to move left to correct a rightward drift — standard
                // inverted-yaw for ship controls. Matches the legacy
                // feel the user has already signed off on.
                rates.yaw = (rates.yaw - dx_rad * settings.pilot_rate_gain)
                    .clamp(-PILOT_RATE_CLAMP, PILOT_RATE_CLAMP);
                rates.pitch = (rates.pitch - invert * dy_rad * settings.pilot_rate_gain)
                    .clamp(-PILOT_RATE_CLAMP, PILOT_RATE_CLAMP);
            }
        }
    }
    // Rate-limited diagnostic of the last-frame mouse totals and the
    // current look/rate state. Logs once per second regardless of
    // whether events arrived, so a log line with `events=0` distinguishes
    // "mouse not reaching this system" from "math doesn't update yaw."
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_LOG: AtomicU64 = AtomicU64::new(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = LAST_LOG.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) > 1000 {
            LAST_LOG.store(now_ms, Ordering::Relaxed);
            tracing::info!(
                grabbed,
                alt_held,
                events = ev_count,
                dx_sum,
                dy_sum,
                yaw = look.yaw,
                pitch = look.pitch,
                server_yaw = look.server_yaw,
                server_pitch = look.server_pitch,
                piloting = pilot.is_piloting,
                "mouse-look diag"
            );
        }
    }
}

fn update_speed_tier(
    keys: Res<ButtonInput<KeyCode>>,
    mut controls: ResMut<ShipControlsState>,
) {
    // Keys 1–5 → tiers 0–`SPEED_TIER_MAX`. `just_pressed` edge-triggers so
    // holding the key down does not re-assert. Table-driven so any tier-
    // count change only touches one line.
    const KEY_TO_TIER: &[(KeyCode, u8)] = &[
        (KeyCode::Digit1, 0),
        (KeyCode::Digit2, 1),
        (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3),
        (KeyCode::Digit5, 4),
    ];
    for &(key, tier) in KEY_TO_TIER {
        if tier > SPEED_TIER_MAX { break; }
        if keys.just_pressed(key) {
            controls.speed_tier = tier;
            return;
        }
    }
}

fn update_thrust_limiter(
    mut wheel: MessageReader<MouseWheel>,
    settings: Res<InputSettings>,
    mut controls: ResMut<ShipControlsState>,
) {
    for ev in wheel.read() {
        controls.thrust_limiter =
            (controls.thrust_limiter + ev.y * settings.scroll_thrust_step).clamp(0.0, 1.0);
    }
}

/// Assemble `PlayerInputData` and push it into the outbound channel. Runs
/// every frame; the network thread coalesces to 20 Hz. Sending every frame
/// keeps the sub-frame yaw/pitch snapshot as fresh as possible without
/// any dead-reckoning on the server side.
fn build_and_send_input(
    conn: Res<NetConnection>,
    keys: Res<ButtonInput<KeyCode>>,
    look: Res<LocalLook>,
    rates: Res<PilotRates>,
    controls: Res<ShipControlsState>,
    pilot: Res<PilotState>,
    seat_values: Res<SeatValues>,
    sender: Res<InputSender>,
    mut tick: ResMut<InputTickCounter>,
) {
    if !conn.connected {
        return;
    }

    // Walk-mode movement vector.
    //
    // **Wire convention (audited against `core::character::kcc_sys` +
    // `ship-shard::main::process_input`):**
    //   movement[0] = strafe right   → +1 when D held
    //   movement[1] = vertical       → +1 Space, -1 Ctrl (EVA / fly)
    //   movement[2] = forward        → +1 when W held
    //
    // The server KCC formula at `yaw=0` projects `movement[2]` along +X
    // and `movement[0]` along +Z — so yaw=0 in the *server's* frame
    // means "facing +X." The client's Bevy yaw uses +Y rotation where
    // yaw=0 means facing -Z; translation from client yaw to server yaw
    // is done below at the look-angle assembly step so the two stay in
    // lockstep without changing Bevy's internal basis.
    //
    // When piloting, the server routes WASD into `seat_values` via
    // `SeatBindingsNotify` (task #17), not into `movement`. We zero the
    // vector in that path so the server never double-counts inputs.
    let movement = if pilot.is_piloting {
        Vec3::ZERO
    } else {
        let mut m = Vec3::ZERO;
        if keys.pressed(KeyCode::KeyW) { m.z += 1.0; }
        if keys.pressed(KeyCode::KeyS) { m.z -= 1.0; }
        if keys.pressed(KeyCode::KeyD) { m.x += 1.0; }
        if keys.pressed(KeyCode::KeyA) { m.x -= 1.0; }
        if keys.pressed(KeyCode::Space) { m.y += 1.0; }
        if keys.pressed(KeyCode::ControlLeft) { m.y -= 1.0; }
        if m.length_squared() > 1.0 { m = m.normalize(); }
        m
    };

    let mut actions_bits: u32 = 0;
    if keys.pressed(KeyCode::ShiftLeft) {
        actions_bits |= input_action_bits::SPRINT;
    }
    if keys.pressed(KeyCode::KeyC) {
        actions_bits |= input_action_bits::CROUCH;
    }

    // Look angles. `LocalLook.server_yaw/pitch` are already stored in the
    // server/legacy convention (yaw=0 → shard-local +X, positive yaw → +Z,
    // pitch = elevation from horizon) — `rotation_from_look` builds the
    // Bevy camera rotation from them, and the wire contract expects the
    // same values. No conversion needed.
    //
    // Pilot mode sends rates in [-1, 1] — the server interprets those as
    // torque inputs, not angles, so no frame conversion applies.
    let (look_yaw, look_pitch) = if pilot.is_piloting {
        (rates.yaw, rates.pitch)
    } else {
        (look.server_yaw, look.server_pitch)
    };

    // EXIT_SEAT is NOT consumed from `PlayerInputData.action` — only
    // `BlockEditRequest` (TCP) triggers the server's exit-seat handler
    // (`ship-shard/main.rs:3619-3631`). Putting the action code here is
    // a no-op against the authoritative server. The TCP path lives in
    // the `block_interaction` module.
    let action_code: u8 = 0;

    tick.0 = tick.0.wrapping_add(1);
    let input = PlayerInputData {
        movement: [movement.x, movement.y, movement.z],
        look_yaw,
        look_pitch,
        jump: keys.just_pressed(KeyCode::Space),
        // `fly_toggle` and `orbit_stabilizer_toggle` are legacy-deprecated
        // and the server ignores them; forward for completeness so any
        // future re-introduction does not need client surgery.
        fly_toggle: keys.just_pressed(KeyCode::KeyF),
        orbit_stabilizer_toggle: keys.just_pressed(KeyCode::KeyO),
        speed_tier: controls.speed_tier,
        action: action_code,
        // Block type for placement comes from inventory / hotbar UI (task
        // #20). Placeholder 0 (air) does not harm anything because the
        // `action` field for placement is routed through BlockEdit, not
        // here — this field stays 0 until the inventory wiring sets it.
        block_type: 0,
        tick: tick.0,
        thrust_limiter: controls.thrust_limiter,
        // Roll is reserved for future Q/E roll key binding; legacy keeps
        // this plumbed but at zero.
        roll: 0.0,
        cruise: false,
        atmo_comp: false,
        // Seat-binding outputs. Populated each frame by
        // `seat_bindings::evaluate_seat_bindings` when the player is
        // seated and the server has sent a `SeatBindingsNotify`; empty
        // vec otherwise (server ignores empty). We clone because the
        // UDP sender takes ownership — unavoidable crossing the thread
        // boundary. ~16 f32s per packet at 20 Hz = negligible allocation.
        seat_values: seat_values.0.clone(),
        actions_bits,
    };

    if sender.tx.send(input).is_err() {
        // Channel closed means the network worker died — it will be
        // reconstructed on reconnect. Log once per second to avoid spam.
        static LAST_WARNED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
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

/// Build the local forward direction from (yaw, pitch) in the
/// **server/legacy** convention — yaw=0 means forward along +X, yaw=+π/2
/// means forward along +Z, pitch is elevation measured from the horizon.
/// This is the identical formula used by `voxydust/src/camera.rs` and the
/// server KCC (`core::character::kcc_sys:108-126`), so client and server
/// agree on what "forward" means without any conversion at the wire
/// boundary.
pub fn forward_from_look(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(cy * cp, sp, sy * cp)
}

/// Camera rotation in Bevy's basis (camera default: fwd=-Z, up=+Y) whose
/// `q * (-Z)` equals `forward_from_look(yaw, pitch)`. Used by
/// `camera_frame` to compose the final render rotation: for a SHIP camera
/// the full rotation is `ship_rot * rotation_from_look(yaw, pitch)` and
/// so on for PLANET / SYSTEM / GALAXY.
///
/// Implementation delegates to Bevy's `Transform::looking_to`, which
/// internally builds the correct right-handed orthonormal basis,
/// guarantees a **unit quaternion**, and is the same codepath Bevy
/// itself uses for camera-aim operations. We tried rolling it ourselves
/// via `Mat3::from_cols` + `Quat::from_mat3` and hit a subtle sign bug
/// that produced a reflection matrix (det = -1); `Quat::from_mat3` does
/// NOT panic on non-proper rotations, it silently returns a non-unit
/// quaternion whose action scales every vector by `|q|²`. That presented
/// as "camera rotation partially works but directions are wrong and the
/// forward vector has magnitude 0.5 instead of 1." Using Bevy's tested
/// helper is strictly better — no sign bugs to chase.
///
/// World-up pinned to +Y so the camera stays horizon-stable (no roll)
/// as yaw/pitch vary. `looking_to` is well-defined as long as `fwd` is
/// not colinear with `up` — enforced by `PITCH_CLAMP` < π/2.
pub fn rotation_from_look(yaw: f32, pitch: f32) -> Quat {
    let fwd = forward_from_look(yaw, pitch);
    Transform::IDENTITY.looking_to(fwd, Vec3::Y).rotation
}

/// Extract (yaw, pitch) from a forward vector in the legacy/server
/// convention. Inverse of `forward_from_look`:
///   yaw   = atan2(fwd.z, fwd.x)
///   pitch = asin(fwd.y)
pub fn yaw_pitch_from_forward(fwd: Vec3) -> (f32, f32) {
    let yaw = fwd.z.atan2(fwd.x);
    let pitch = fwd.y.clamp(-1.0, 1.0).asin();
    (yaw, pitch)
}

/// Extract (yaw, pitch) from a **server-convention rotation** — one that
/// rotates +X to the "forward" direction. Used for the authoritative
/// `SpawnPose.rotation` arriving from shard handoffs.
///
/// The server's rotation convention differs from Bevy's camera default:
/// server treats identity as "facing +X" (because legacy/KCC yaw=0 → +X),
/// while Bevy's default camera faces -Z. Applying the server rotation to
/// +X and then running `yaw_pitch_from_forward` on the result returns
/// legacy-convention yaw/pitch, consistent with everything else this
/// module produces.
pub fn yaw_pitch_from_server_rotation(q: Quat) -> (f32, f32) {
    yaw_pitch_from_forward(q * Vec3::X)
}
