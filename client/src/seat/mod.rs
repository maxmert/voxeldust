//! Seat bindings — server-authoritative binding list parsed from
//! `SeatBindingsNotify`, evaluated every frame into a per-binding
//! `f32` vec the input pipeline packs into `PlayerInputData.seat_values`.
//!
//! **Shard-agnostic.** A turret seat on a planet wall, a cockpit seat
//! on a ship, and a future salvage seat on debris all share this code.
//! The server authors bindings; the client evaluates them without
//! caring which shard type the seat lives on. (Design Principle #9.)
//!
//! Ported from `voxydust-next/src/seat_bindings.rs` with module-path
//! adaptations to the new architecture.

use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use voxeldust_core::signal::components::{AxisDirection, KeyMode, SeatInputSource};
use voxeldust_core::signal::config::SeatInputBindingConfig;

use crate::config::GameConfig;
use crate::input::{FrameMouseDelta, InputSet, SeatedState, SendInputSet};
use crate::net::{GameEvent, NetEvent};
use crate::shard::ShardTransitionSet;

pub struct SeatBindingsPlugin;

impl Plugin for SeatBindingsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveSeatBindings>()
            .init_resource::<SeatValues>()
            .add_systems(
                Update,
                (
                    receive_seat_bindings_notify,
                    clear_bindings_on_exit,
                    evaluate_seat_bindings,
                )
                    .chain()
                    .after(ShardTransitionSet)
                    .after(InputSet)
                    .before(SendInputSet),
            );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Resources
// ─────────────────────────────────────────────────────────────────────

/// A single server-authored binding, parsed for client-side evaluation.
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
            scroll_accumulated: if cfg.source == SeatInputSource::ScrollWheel {
                0.75
            } else {
                0.0
            },
            prev_key_held: false,
        }
    }
}

/// Active seat bindings. Populated on `SeatBindingsNotify`; cleared on
/// un-seat. Empty vec means not seated (or seated in a seat with no
/// bindings — server treats as a no-op seat).
#[derive(Resource, Default, Debug)]
pub struct ActiveSeatBindings {
    pub bindings: Vec<ClientSeatBinding>,
}

impl ActiveSeatBindings {
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    fn set_from_configs(&mut self, configs: &[SeatInputBindingConfig]) {
        self.bindings = configs
            .iter()
            .map(ClientSeatBinding::from_config)
            .collect();
    }

    fn clear(&mut self) {
        self.bindings.clear();
    }
}

/// Per-frame evaluation result — one `f32` per active binding, same
/// order. `input::build_and_send_input` clones this into
/// `PlayerInputData.seat_values`.
#[derive(Resource, Default, Debug, Clone)]
pub struct SeatValues(pub Vec<f32>);

// ─────────────────────────────────────────────────────────────────────
// Systems
// ─────────────────────────────────────────────────────────────────────

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
                "received SeatBindingsNotify — active seat bindings populated",
            );
        }
    }
}

/// Clear bindings + values on the falling edge of `SeatedState.is_seated`.
fn clear_bindings_on_exit(
    seated: Res<SeatedState>,
    mut active: ResMut<ActiveSeatBindings>,
    mut values: ResMut<SeatValues>,
    mut prev_seated: Local<bool>,
) {
    if *prev_seated && !seated.is_seated {
        active.clear();
        values.0.clear();
        tracing::info!("un-seated — cleared active seat bindings");
    }
    *prev_seated = seated.is_seated;
}

fn evaluate_seat_bindings(
    seated: Res<SeatedState>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    cursors: Query<&CursorOptions, With<PrimaryWindow>>,
    cfg: Res<GameConfig>,
    frame_delta: Res<FrameMouseDelta>,
    mut active: ResMut<ActiveSeatBindings>,
    mut values: ResMut<SeatValues>,
) {
    if !seated.is_seated || active.is_empty() {
        values.0.clear();
        return;
    }

    let grabbed = cursors
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !grabbed {
        return;
    }

    // Read mouse delta from the shared per-frame Resource populated by
    // `input::collect_mouse_delta`. Single source of truth — both
    // camera-look and seat-bindings read the same authoritative delta.
    //
    // Invert flags: this code path runs only while seated (flight-sim
    // control), so the invert-X / invert-Y config applies here. Walking
    // and free-look consumers use the raw delta for natural FPS feel.
    let inv_x = if cfg.control.mouse_invert_x { -1.0_f32 } else { 1.0_f32 };
    let inv_y = if cfg.control.mouse_invert_y { -1.0_f32 } else { 1.0_f32 };
    let sens = cfg.control.mouse_sensitivity;
    let gain = cfg.control.seat_mouse_gain;
    let yaw_rate = inv_x * frame_delta.dx * sens * gain;
    let pitch_rate = inv_y * frame_delta.dy * sens * gain;
    let scroll_delta = frame_delta.scroll_y * cfg.control.scroll_thrust_step;

    values.0.resize(active.bindings.len(), 0.0);
    for (i, b) in active.bindings.iter_mut().enumerate() {
        values.0[i] = match b.source {
            SeatInputSource::Key => evaluate_key(b, &keys, &mouse_buttons),
            SeatInputSource::MouseMoveX => b.axis_direction.apply(yaw_rate),
            SeatInputSource::MouseMoveY => b.axis_direction.apply(pitch_rate),
            SeatInputSource::ScrollWheel => {
                b.scroll_accumulated =
                    (b.scroll_accumulated + scroll_delta).clamp(0.0, 1.0);
                b.scroll_accumulated
            }
        };
    }
}

fn evaluate_key(
    b: &mut ClientSeatBinding,
    keys: &ButtonInput<KeyCode>,
    mouse_buttons: &ButtonInput<MouseButton>,
) -> f32 {
    let held = b.key_code.map(|k| keys.pressed(k)).unwrap_or(false)
        || b.mouse_button
            .map(|m| mouse_buttons.pressed(m))
            .unwrap_or(false);
    match b.key_mode {
        KeyMode::Momentary => {
            b.prev_key_held = held;
            if held {
                1.0
            } else {
                0.0
            }
        }
        KeyMode::Toggle => {
            if held && !b.prev_key_held {
                b.toggle_state = !b.toggle_state;
            }
            b.prev_key_held = held;
            if b.toggle_state {
                1.0
            } else {
                0.0
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

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
        "Digit0" => KeyCode::Digit0, "Digit1" => KeyCode::Digit1,
        "Digit2" => KeyCode::Digit2, "Digit3" => KeyCode::Digit3,
        "Digit4" => KeyCode::Digit4, "Digit5" => KeyCode::Digit5,
        "Digit6" => KeyCode::Digit6, "Digit7" => KeyCode::Digit7,
        "Digit8" => KeyCode::Digit8, "Digit9" => KeyCode::Digit9,
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
