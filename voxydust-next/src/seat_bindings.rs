//! Seat-binding evaluation.
//!
//! When the server seats the player in a cockpit/station block it sends a
//! `SeatBindingsNotify { bindings: Vec<SeatInputBindingConfig> }` payload
//! describing how the seat's physical input (keyboard keys, mouse motion
//! axes, scroll wheel) maps into signal-graph channels — a ship's
//! thrusters, rotors, torque, autopilot engage, gear toggle, etc. This
//! module:
//!
//! 1. Receives `SeatBindingsNotify`, parses string `key_name` fields into
//!    strongly-typed `KeyCode` / `MouseButton`, and stores the resolved
//!    bindings in `ActiveSeatBindings`.
//! 2. Clears bindings when the server un-seats the player (detected via
//!    `PilotState.is_piloting` false-edge).
//! 3. Every frame while piloting, evaluates each binding against current
//!    input (keys held, mouse deltas, scroll wheel accumulation) to
//!    produce a `Vec<f32>` in `SeatValues`.
//! 4. `input_system::build_and_send_input` reads `SeatValues` and sends
//!    them in `PlayerInputData.seat_values`.
//!
//! **Multiplayer:** seat bindings are per-seat, per-session. Each client
//! gets its own `SeatBindingsNotify` from the server when it sits; there
//! is no cross-client state. When multiple crewmates sit in different
//! seats on the same ship, each produces its own `seat_values` which
//! the server composes into the ship's signal graph.
//!
//! **Server authority:** the server owns the binding list (authored in
//! the seat block's configuration panel). The client never invents a
//! binding — everything comes from `SeatBindingsNotify`. Values
//! produced here are advisory; the server validates and applies them
//! against the seat's current binding map, re-rejecting any that don't
//! match.
//!
//! Legacy ref: `voxydust/src/input.rs:1-250` + the key-name/KeyCode
//! table that used to live at `input.rs:257-303`. Both are ported
//! essentially verbatim.

use bevy::input::mouse::{MouseButton, MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use voxeldust_core::signal::components::{AxisDirection, KeyMode, SeatInputSource};
use voxeldust_core::signal::config::SeatInputBindingConfig;

use crate::input_system::{
    InputSet, InputSettings, PilotState, SendInputSet, MOUSE_SENSITIVITY_DEFAULT,
};
use crate::net_plugin::GameEvent;
use crate::network::NetEvent;
use crate::shard_transition::ShardTransitionSet;

pub struct SeatBindingsPlugin;

impl Plugin for SeatBindingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveSeatBindings>()
            .init_resource::<SeatValues>()
            .add_systems(
                Update,
                (
                    // 1. Handle incoming seat-bindings-notify (populate
                    //    `ActiveSeatBindings` when server seats us).
                    receive_seat_bindings_notify,
                    // 2. Clear bindings the instant we un-seat.
                    clear_bindings_on_exit,
                    // 3. Evaluate every active binding against this
                    //    frame's input → populate `SeatValues`.
                    evaluate_seat_bindings,
                )
                    .chain()
                    .after(ShardTransitionSet)
                    // After InputSet so track_pilot_mode has flipped
                    // is_piloting for this frame; before SendInputSet
                    // so build_and_send_input reads fresh SeatValues
                    // in the same-frame UDP packet.
                    .after(InputSet)
                    .before(SendInputSet),
            );
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// A single server-authored binding, parsed for client-side evaluation.
///
/// `prev_key_held` / `toggle_state` / `scroll_accumulated` carry
/// per-binding state across frames so Toggle-mode keys latch on rising
/// edges and ScrollWheel sources accumulate notches into a 0..1 slider.
#[derive(Debug)]
pub struct ClientSeatBinding {
    pub source: SeatInputSource,
    pub key_code: Option<KeyCode>,
    pub mouse_button: Option<MouseButton>,
    pub key_mode: KeyMode,
    pub axis_direction: AxisDirection,
    pub toggle_state: bool,
    pub scroll_accumulated: f32,
    pub prev_key_held: bool,
}

impl ClientSeatBinding {
    fn from_config(cfg: &SeatInputBindingConfig) -> Self {
        let (key_code, mouse_button) = if cfg.source == SeatInputSource::Key {
            if voxeldust_core::signal::key_names::is_mouse_button(&cfg.key_name) {
                (None, key_name_to_mouse_button(&cfg.key_name))
            } else {
                (key_name_to_keycode(&cfg.key_name), None)
            }
        } else {
            (None, None)
        };
        Self {
            source: cfg.source,
            key_code,
            mouse_button,
            key_mode: cfg.key_mode,
            axis_direction: cfg.axis_direction,
            toggle_state: false,
            // Scroll wheel initial state matches legacy (`input.rs:68`):
            // 0.75 as a sensible starting throttle for thrust-style
            // bindings that map scroll to 0..1. Non-scroll bindings get 0.
            scroll_accumulated: if cfg.source == SeatInputSource::ScrollWheel { 0.75 } else { 0.0 },
            prev_key_held: false,
        }
    }
}

/// Active seat bindings. Populated when the server sends
/// `SeatBindingsNotify`; cleared on un-seat. Empty vec means not
/// seated (or seated in a seat with no bindings configured, which the
/// server treats as a no-op seat).
#[derive(Resource, Default, Debug)]
pub struct ActiveSeatBindings {
    pub bindings: Vec<ClientSeatBinding>,
}

impl ActiveSeatBindings {
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    fn set_from_configs(&mut self, configs: &[SeatInputBindingConfig]) {
        self.bindings = configs.iter().map(ClientSeatBinding::from_config).collect();
    }

    fn clear(&mut self) {
        self.bindings.clear();
    }
}

/// Per-frame evaluation result — one `f32` per `ActiveSeatBindings.bindings`
/// entry, in the same order. `input_system::build_and_send_input` copies
/// this straight into `PlayerInputData.seat_values`. Empty when not
/// piloting so the server never sees stale values after dismount.
#[derive(Resource, Default, Debug, Clone)]
pub struct SeatValues(pub Vec<f32>);

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

fn receive_seat_bindings_notify(
    mut events: MessageReader<GameEvent>,
    mut active: ResMut<ActiveSeatBindings>,
) {
    for GameEvent(ev) in events.read() {
        if let NetEvent::SeatBindingsNotify(data) = ev {
            active.set_from_configs(&data.bindings);
            tracing::info!(
                count = data.bindings.len(),
                labels = ?data.bindings.iter().map(|b| b.label.clone()).collect::<Vec<_>>(),
                "received SeatBindingsNotify — active seat bindings populated"
            );
        }
    }
}

/// Edge-detect `PilotState.is_piloting` falling to false (server
/// un-seated the player) and clear bindings + values. Running right
/// after `receive_seat_bindings_notify` means a Notify this frame still
/// populates correctly even if `is_piloting` was briefly false — the
/// clear only fires when the edge happens, not whenever the flag is false.
fn clear_bindings_on_exit(
    pilot: Res<PilotState>,
    mut active: ResMut<ActiveSeatBindings>,
    mut values: ResMut<SeatValues>,
    mut prev_piloting: Local<bool>,
) {
    if *prev_piloting && !pilot.is_piloting {
        active.clear();
        values.0.clear();
        tracing::info!("un-seated — cleared active seat bindings");
    }
    *prev_piloting = pilot.is_piloting;
}

/// Evaluate every active binding against this frame's input and write
/// into `SeatValues`. Key bindings handle Momentary/Toggle state
/// machines; mouse axis bindings read pixels-per-sec → normalized rate
/// via the same mouse sensitivity the view look uses; scroll bindings
/// accumulate between frames (clamped 0..1).
fn evaluate_seat_bindings(
    pilot: Res<PilotState>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    settings: Res<InputSettings>,
    mut active: ResMut<ActiveSeatBindings>,
    mut values: ResMut<SeatValues>,
) {
    // Not piloting → guaranteed-empty output. We still drain the event
    // readers here so they don't pile up between sit-down events (the
    // mouse / scroll cursors must stay caught-up regardless).
    if !pilot.is_piloting || active.is_empty() {
        mouse_motion.clear();
        mouse_wheel.clear();
        values.0.clear();
        return;
    }

    // Cursor must be grabbed for seat input to fire — same gate as the
    // view-look accumulation. If the user opened a menu, seat inputs go
    // dead (keys stop, mouse motion doesn't accumulate). Matches legacy.
    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !grabbed {
        mouse_motion.clear();
        mouse_wheel.clear();
        // Keep last-evaluated values — the ship doesn't suddenly lose
        // a toggled landing-gear state just because the player opened
        // a menu. Bindings with axis sources naturally damp to zero
        // next frame because their raw input is zero.
        return;
    }

    // Aggregate this frame's mouse delta (sum across all events this
    // tick) scaled to the same "normalized rate" the view pipeline uses
    // — a full screen-width of mouse travel ≈ 1.0 rate. Applying the
    // same `mouse_sensitivity` scale keeps "how hard am I pushing the
    // stick" consistent with look-mode feel.
    let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
    for ev in mouse_motion.read() {
        dx += ev.delta.x;
        dy += ev.delta.y;
    }
    let sens = settings.mouse_sensitivity;
    let yaw_rate = dx * sens * mouse_sensitivity_to_rate_gain();
    let pitch_rate = dy * sens * mouse_sensitivity_to_rate_gain();

    // Scroll delta this frame. `MouseWheel.y` in "lines" (conventional
    // per-notch units); legacy treats one notch as the per-step binding
    // increment — we pass it straight through to `scroll_accumulated`
    // where the binding clamps into 0..1 at its configured step size.
    // That preserves the legacy feel (one notch ≈ visible slider step).
    let mut scroll = 0.0_f32;
    for ev in mouse_wheel.read() {
        scroll += ev.y;
    }
    // Per-notch scale — legacy defaults to a tuned step in
    // `settings.scroll_thrust_step`. We apply it here so both
    // thrust-limiter (ship controls) and scroll-bound seat inputs feel
    // identical at the same sensitivity.
    let scroll_step = settings.scroll_thrust_step;
    let scroll_delta = scroll * scroll_step;

    // Resize output vec in lock-step with binding count so downstream
    // senders can index by position without bounds-check dance.
    values.0.resize(active.bindings.len(), 0.0);

    for (i, b) in active.bindings.iter_mut().enumerate() {
        values.0[i] = match b.source {
            SeatInputSource::Key => evaluate_key(b, &keys, &mouse_buttons),
            SeatInputSource::MouseMoveX => b.axis_direction.apply(yaw_rate),
            SeatInputSource::MouseMoveY => b.axis_direction.apply(pitch_rate),
            SeatInputSource::ScrollWheel => {
                b.scroll_accumulated = (b.scroll_accumulated + scroll_delta).clamp(0.0, 1.0);
                b.scroll_accumulated
            }
        };
    }

    // Diagnostic: once per second, log which bindings are non-zero so
    // the pilot can confirm their inputs are actually producing values.
    // Without this log it's impossible to tell whether a quiet ship is
    // "bindings aren't firing" vs "bindings fire but server doesn't
    // respond" — they look identical from the user's seat.
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_LOG: AtomicU64 = AtomicU64::new(0);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if now_ms.saturating_sub(LAST_LOG.load(Ordering::Relaxed)) > 1000 {
            LAST_LOG.store(now_ms, Ordering::Relaxed);
            let non_zero: Vec<(usize, SeatInputSource, f32)> = values
                .0
                .iter()
                .enumerate()
                .filter(|(_, v)| v.abs() > 1e-4)
                .map(|(i, v)| (i, active.bindings[i].source, *v))
                .collect();
            tracing::info!(
                binding_count = active.bindings.len(),
                non_zero_count = non_zero.len(),
                non_zero = ?non_zero,
                yaw_rate,
                pitch_rate,
                scroll_delta,
                "seat_bindings diag"
            );
        }
    }
}

/// Key / mouse-button evaluation with Momentary / Toggle state machine.
/// Returns 0.0 or 1.0 — the normalized convention every binding source
/// produces for the signal-graph layer downstream.
fn evaluate_key(
    b: &mut ClientSeatBinding,
    keys: &ButtonInput<KeyCode>,
    mouse_buttons: &ButtonInput<MouseButton>,
) -> f32 {
    let held = b.key_code.map(|k| keys.pressed(k)).unwrap_or(false)
        || b.mouse_button.map(|m| mouse_buttons.pressed(m)).unwrap_or(false);
    match b.key_mode {
        KeyMode::Momentary => {
            b.prev_key_held = held;
            if held { 1.0 } else { 0.0 }
        }
        KeyMode::Toggle => {
            // Rising edge: held now, not held last frame → flip state.
            if held && !b.prev_key_held {
                b.toggle_state = !b.toggle_state;
            }
            b.prev_key_held = held;
            if b.toggle_state { 1.0 } else { 0.0 }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scale factor to convert mouse delta (already-scaled by
/// `mouse_sensitivity`) to the "normalized rate" the server expects for
/// pilot-mode axis bindings. Legacy matched this to `PILOT_RATE_GAIN`
/// (0.3) so mouse-move-X bound to a thruster behaves identically to
/// mouse-move-X bound to ship yaw-rate.
const fn mouse_sensitivity_to_rate_gain() -> f32 {
    // Constant rather than a config field because changing it retunes
    // every seat binding simultaneously — we hold a single "how much
    // does a mouse push mean" knob.
    let _default = MOUSE_SENSITIVITY_DEFAULT; // not used directly; referenced for documentation
    0.3
}

/// Convert a platform-independent `key_name` string (as stored in
/// seat-binding configs) to Bevy / winit's `KeyCode`. This is the
/// inverse of `keycode_to_key_name` on the server side; keeping the
/// table in sync with `voxydust/src/input.rs:257-303` is a pre-condition
/// for new key types becoming bindable.
fn key_name_to_keycode(name: &str) -> Option<KeyCode> {
    Some(match name {
        "KeyA" => KeyCode::KeyA, "KeyB" => KeyCode::KeyB,
        "KeyC" => KeyCode::KeyC, "KeyD" => KeyCode::KeyD,
        "KeyE" => KeyCode::KeyE, "KeyF" => KeyCode::KeyF,
        "KeyG" => KeyCode::KeyG, "KeyH" => KeyCode::KeyH,
        "KeyI" => KeyCode::KeyI, "KeyJ" => KeyCode::KeyJ,
        "KeyK" => KeyCode::KeyK, "KeyL" => KeyCode::KeyL,
        "KeyM" => KeyCode::KeyM, "KeyN" => KeyCode::KeyN,
        "KeyO" => KeyCode::KeyO, "KeyP" => KeyCode::KeyP,
        "KeyQ" => KeyCode::KeyQ, "KeyR" => KeyCode::KeyR,
        "KeyS" => KeyCode::KeyS, "KeyT" => KeyCode::KeyT,
        "KeyU" => KeyCode::KeyU, "KeyV" => KeyCode::KeyV,
        "KeyW" => KeyCode::KeyW, "KeyX" => KeyCode::KeyX,
        "KeyY" => KeyCode::KeyY, "KeyZ" => KeyCode::KeyZ,
        "Digit1" => KeyCode::Digit1, "Digit2" => KeyCode::Digit2,
        "Digit3" => KeyCode::Digit3, "Digit4" => KeyCode::Digit4,
        "Digit5" => KeyCode::Digit5, "Digit6" => KeyCode::Digit6,
        "Digit7" => KeyCode::Digit7, "Digit8" => KeyCode::Digit8,
        "Digit9" => KeyCode::Digit9, "Digit0" => KeyCode::Digit0,
        "F1" => KeyCode::F1, "F2" => KeyCode::F2,
        "F3" => KeyCode::F3, "F4" => KeyCode::F4,
        "F5" => KeyCode::F5, "F6" => KeyCode::F6,
        "F7" => KeyCode::F7, "F8" => KeyCode::F8,
        "F9" => KeyCode::F9, "F10" => KeyCode::F10,
        "F11" => KeyCode::F11, "F12" => KeyCode::F12,
        "Space" => KeyCode::Space, "Enter" => KeyCode::Enter,
        "Tab" => KeyCode::Tab, "Escape" => KeyCode::Escape,
        "Backspace" => KeyCode::Backspace, "Delete" => KeyCode::Delete,
        "Insert" => KeyCode::Insert, "Home" => KeyCode::Home,
        "End" => KeyCode::End, "PageUp" => KeyCode::PageUp,
        "PageDown" => KeyCode::PageDown,
        "ArrowUp" => KeyCode::ArrowUp, "ArrowDown" => KeyCode::ArrowDown,
        "ArrowLeft" => KeyCode::ArrowLeft, "ArrowRight" => KeyCode::ArrowRight,
        "ShiftLeft" => KeyCode::ShiftLeft, "ShiftRight" => KeyCode::ShiftRight,
        "ControlLeft" => KeyCode::ControlLeft, "ControlRight" => KeyCode::ControlRight,
        "AltLeft" => KeyCode::AltLeft, "AltRight" => KeyCode::AltRight,
        "Minus" => KeyCode::Minus, "Equal" => KeyCode::Equal,
        "BracketLeft" => KeyCode::BracketLeft, "BracketRight" => KeyCode::BracketRight,
        "Backslash" => KeyCode::Backslash, "Semicolon" => KeyCode::Semicolon,
        "Quote" => KeyCode::Quote, "Comma" => KeyCode::Comma,
        "Period" => KeyCode::Period, "Slash" => KeyCode::Slash,
        "Backquote" => KeyCode::Backquote,
        _ => return None,
    })
}

fn key_name_to_mouse_button(name: &str) -> Option<MouseButton> {
    Some(match name {
        "MouseLeft" => MouseButton::Left,
        "MouseRight" => MouseButton::Right,
        "MouseMiddle" => MouseButton::Middle,
        "MouseButton4" => MouseButton::Back,
        "MouseButton5" => MouseButton::Forward,
        _ => return None,
    })
}
