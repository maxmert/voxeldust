//! Player input processing: key/mouse state to network input packets.
//!
//! When seated with active bindings, evaluates each binding against current input state
//! and sends per-binding float values in `seat_values`. When walking, sends legacy fields.

use std::collections::HashSet;
use tokio::sync::mpsc;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use voxeldust_core::client_message::PlayerInputData;
use voxeldust_core::signal::components::{AxisDirection, KeyMode, SeatInputSource};
use voxeldust_core::signal::config::SeatInputBindingConfig;

// ---------------------------------------------------------------------------
// Client-side seat binding state
// ---------------------------------------------------------------------------

/// A single seat binding resolved to platform-specific input codes.
pub struct ClientSeatBinding {
    pub source: SeatInputSource,
    pub key_code: Option<KeyCode>,
    pub mouse_button: Option<MouseButton>,
    pub key_mode: KeyMode,
    pub axis_direction: AxisDirection,
    /// Current toggle state (for Toggle keys).
    pub toggle_state: bool,
    /// Current accumulated value (for ScrollWheel).
    pub scroll_accumulated: f32,
    /// Previous frame's key held state (for toggle edge detection).
    pub prev_key_held: bool,
}

/// Active seat bindings received from server. Empty when not seated.
#[derive(bevy_ecs::prelude::Resource)]
pub struct ActiveSeatBindings {
    pub bindings: Vec<ClientSeatBinding>,
}

impl Default for ActiveSeatBindings {
    fn default() -> Self {
        Self { bindings: Vec::new() }
    }
}

impl ActiveSeatBindings {
    /// Build from server's SeatBindingsNotify.
    pub fn from_config(configs: &[SeatInputBindingConfig]) -> Self {
        Self {
            bindings: configs.iter().map(|cfg| {
                let (key_code, mouse_button) = if cfg.source == SeatInputSource::Key {
                    if voxeldust_core::signal::key_names::is_mouse_button(&cfg.key_name) {
                        (None, key_name_to_mouse_button(&cfg.key_name))
                    } else {
                        (key_name_to_keycode(&cfg.key_name), None)
                    }
                } else {
                    (None, None)
                };
                ClientSeatBinding {
                    source: cfg.source,
                    key_code,
                    mouse_button,
                    key_mode: cfg.key_mode,
                    axis_direction: cfg.axis_direction,
                    toggle_state: false,
                    scroll_accumulated: if cfg.source == SeatInputSource::ScrollWheel { 0.75 } else { 0.0 },
                    prev_key_held: false,
                }
            }).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }
}

// ---------------------------------------------------------------------------
// Seat binding evaluation
// ---------------------------------------------------------------------------

/// Evaluate all active seat bindings against current input state.
/// Returns a Vec<f32> matching binding order — sent as `seat_values` in PlayerInput.
pub fn evaluate_seat_bindings(
    bindings: &mut [ClientSeatBinding],
    keys_held: &HashSet<KeyCode>,
    mouse_buttons_held: &HashSet<MouseButton>,
    pilot_yaw_rate: f64,
    pilot_pitch_rate: f64,
    scroll_delta: f32,
) -> Vec<f32> {
    bindings.iter_mut().map(|b| {
        match b.source {
            SeatInputSource::Key => {
                let held = b.key_code.map(|k| keys_held.contains(&k)).unwrap_or(false)
                    || b.mouse_button.map(|m| mouse_buttons_held.contains(&m)).unwrap_or(false);

                match b.key_mode {
                    KeyMode::Momentary => {
                        b.prev_key_held = held;
                        if held { 1.0 } else { 0.0 }
                    }
                    KeyMode::Toggle => {
                        // Rising edge: held now but not last frame.
                        if held && !b.prev_key_held {
                            b.toggle_state = !b.toggle_state;
                        }
                        b.prev_key_held = held;
                        if b.toggle_state { 1.0 } else { 0.0 }
                    }
                }
            }
            SeatInputSource::MouseMoveX => {
                b.axis_direction.apply(pilot_yaw_rate as f32)
            }
            SeatInputSource::MouseMoveY => {
                b.axis_direction.apply(pilot_pitch_rate as f32)
            }
            SeatInputSource::ScrollWheel => {
                b.scroll_accumulated = (b.scroll_accumulated + scroll_delta).clamp(0.0, 1.0);
                b.scroll_accumulated
            }
        }
    }).collect()
}

// ---------------------------------------------------------------------------
// Legacy input (walking mode / no active bindings)
// ---------------------------------------------------------------------------

/// Build and send the player input packet based on current key state.
/// When seated with active bindings, sends seat_values. Otherwise sends legacy fields.
pub fn send_input_with_dt(
    input_tx: &Option<mpsc::UnboundedSender<PlayerInputData>>,
    keys_held: &HashSet<KeyCode>,
    mouse_buttons_held: &HashSet<MouseButton>,
    is_piloting: bool,
    camera_yaw: f64,
    camera_pitch: f64,
    pilot_yaw_rate: &mut f64,
    pilot_pitch_rate: &mut f64,
    selected_thrust_tier: &mut u8,
    thrust_limiter: f32,
    frame_count: u64,
    dt: f64,
    active_seat: &mut ActiveSeatBindings,
    scroll_delta: f32,
) {
    if let Some(tx) = input_tx {
        // Decay pilot rates toward zero (virtual spring centering).
        let decay = (-9.75 * dt).exp();
        *pilot_yaw_rate *= decay;
        *pilot_pitch_rate *= decay;

        // Thrust tier selection (1-5 keys) — always active.
        if keys_held.contains(&KeyCode::Digit1) { *selected_thrust_tier = 0; }
        if keys_held.contains(&KeyCode::Digit2) { *selected_thrust_tier = 1; }
        if keys_held.contains(&KeyCode::Digit3) { *selected_thrust_tier = 2; }
        if keys_held.contains(&KeyCode::Digit4) { *selected_thrust_tier = 3; }
        if keys_held.contains(&KeyCode::Digit5) { *selected_thrust_tier = 4; }

        let free_look = keys_held.contains(&KeyCode::AltLeft);

        // When piloting: send yaw/pitch rate (-1 to 1) for ship torque.
        // When walking: send absolute camera yaw/pitch.
        let (look_yaw, look_pitch) = if is_piloting && !free_look {
            (*pilot_yaw_rate as f32, *pilot_pitch_rate as f32)
        } else {
            (camera_yaw as f32, camera_pitch as f32)
        };

        // Evaluate seat bindings when seated.
        let seat_values = if is_piloting && !active_seat.is_empty() {
            evaluate_seat_bindings(
                &mut active_seat.bindings,
                keys_held,
                mouse_buttons_held,
                *pilot_yaw_rate,
                *pilot_pitch_rate,
                scroll_delta,
            )
        } else {
            Vec::new()
        };

        // Legacy movement (for walking mode).
        let mut movement = [0.0f32; 3];
        if !is_piloting {
            if keys_held.contains(&KeyCode::KeyW) { movement[2] += 1.0; }
            if keys_held.contains(&KeyCode::KeyS) { movement[2] -= 1.0; }
            if keys_held.contains(&KeyCode::KeyD) { movement[0] += 1.0; }
            if keys_held.contains(&KeyCode::KeyA) { movement[0] -= 1.0; }
            if keys_held.contains(&KeyCode::Space) { movement[1] += 1.0; }
            if keys_held.contains(&KeyCode::ControlLeft) { movement[1] -= 1.0; }
        }
        let jump = keys_held.contains(&KeyCode::Space);

        let _ = tx.send(PlayerInputData {
            movement,
            look_yaw,
            look_pitch,
            jump,
            fly_toggle: false,
            orbit_stabilizer_toggle: false,
            speed_tier: *selected_thrust_tier,
            action: 0,
            block_type: 0,
            tick: frame_count,
            thrust_limiter,
            roll: 0.0,
            cruise: false,
            atmo_comp: false,
            seat_values,
        });
    }
}

// ---------------------------------------------------------------------------
// Key name ↔ winit conversion
// ---------------------------------------------------------------------------

/// Convert a platform-independent key name to a winit KeyCode.
pub fn key_name_to_keycode(name: &str) -> Option<KeyCode> {
    match name {
        "KeyA" => Some(KeyCode::KeyA), "KeyB" => Some(KeyCode::KeyB),
        "KeyC" => Some(KeyCode::KeyC), "KeyD" => Some(KeyCode::KeyD),
        "KeyE" => Some(KeyCode::KeyE), "KeyF" => Some(KeyCode::KeyF),
        "KeyG" => Some(KeyCode::KeyG), "KeyH" => Some(KeyCode::KeyH),
        "KeyI" => Some(KeyCode::KeyI), "KeyJ" => Some(KeyCode::KeyJ),
        "KeyK" => Some(KeyCode::KeyK), "KeyL" => Some(KeyCode::KeyL),
        "KeyM" => Some(KeyCode::KeyM), "KeyN" => Some(KeyCode::KeyN),
        "KeyO" => Some(KeyCode::KeyO), "KeyP" => Some(KeyCode::KeyP),
        "KeyQ" => Some(KeyCode::KeyQ), "KeyR" => Some(KeyCode::KeyR),
        "KeyS" => Some(KeyCode::KeyS), "KeyT" => Some(KeyCode::KeyT),
        "KeyU" => Some(KeyCode::KeyU), "KeyV" => Some(KeyCode::KeyV),
        "KeyW" => Some(KeyCode::KeyW), "KeyX" => Some(KeyCode::KeyX),
        "KeyY" => Some(KeyCode::KeyY), "KeyZ" => Some(KeyCode::KeyZ),
        "Digit1" => Some(KeyCode::Digit1), "Digit2" => Some(KeyCode::Digit2),
        "Digit3" => Some(KeyCode::Digit3), "Digit4" => Some(KeyCode::Digit4),
        "Digit5" => Some(KeyCode::Digit5), "Digit6" => Some(KeyCode::Digit6),
        "Digit7" => Some(KeyCode::Digit7), "Digit8" => Some(KeyCode::Digit8),
        "Digit9" => Some(KeyCode::Digit9), "Digit0" => Some(KeyCode::Digit0),
        "F1" => Some(KeyCode::F1), "F2" => Some(KeyCode::F2),
        "F3" => Some(KeyCode::F3), "F4" => Some(KeyCode::F4),
        "F5" => Some(KeyCode::F5), "F6" => Some(KeyCode::F6),
        "F7" => Some(KeyCode::F7), "F8" => Some(KeyCode::F8),
        "F9" => Some(KeyCode::F9), "F10" => Some(KeyCode::F10),
        "F11" => Some(KeyCode::F11), "F12" => Some(KeyCode::F12),
        "Space" => Some(KeyCode::Space), "Enter" => Some(KeyCode::Enter),
        "Tab" => Some(KeyCode::Tab), "Escape" => Some(KeyCode::Escape),
        "Backspace" => Some(KeyCode::Backspace), "Delete" => Some(KeyCode::Delete),
        "Insert" => Some(KeyCode::Insert), "Home" => Some(KeyCode::Home),
        "End" => Some(KeyCode::End), "PageUp" => Some(KeyCode::PageUp),
        "PageDown" => Some(KeyCode::PageDown),
        "ArrowUp" => Some(KeyCode::ArrowUp), "ArrowDown" => Some(KeyCode::ArrowDown),
        "ArrowLeft" => Some(KeyCode::ArrowLeft), "ArrowRight" => Some(KeyCode::ArrowRight),
        "ShiftLeft" => Some(KeyCode::ShiftLeft), "ShiftRight" => Some(KeyCode::ShiftRight),
        "ControlLeft" => Some(KeyCode::ControlLeft), "ControlRight" => Some(KeyCode::ControlRight),
        "AltLeft" => Some(KeyCode::AltLeft), "AltRight" => Some(KeyCode::AltRight),
        "Minus" => Some(KeyCode::Minus), "Equal" => Some(KeyCode::Equal),
        "BracketLeft" => Some(KeyCode::BracketLeft), "BracketRight" => Some(KeyCode::BracketRight),
        "Backslash" => Some(KeyCode::Backslash), "Semicolon" => Some(KeyCode::Semicolon),
        "Quote" => Some(KeyCode::Quote), "Comma" => Some(KeyCode::Comma),
        "Period" => Some(KeyCode::Period), "Slash" => Some(KeyCode::Slash),
        "Backquote" => Some(KeyCode::Backquote),
        _ => None,
    }
}

/// Convert a platform-independent key name to a winit MouseButton.
pub fn key_name_to_mouse_button(name: &str) -> Option<MouseButton> {
    match name {
        "MouseLeft" => Some(MouseButton::Left),
        "MouseRight" => Some(MouseButton::Right),
        "MouseMiddle" => Some(MouseButton::Middle),
        "MouseButton4" => Some(MouseButton::Back),
        "MouseButton5" => Some(MouseButton::Forward),
        _ => None,
    }
}

/// Convert a winit KeyCode to a platform-independent key name.
pub fn keycode_to_key_name(code: KeyCode) -> &'static str {
    match code {
        KeyCode::KeyA => "KeyA", KeyCode::KeyB => "KeyB",
        KeyCode::KeyC => "KeyC", KeyCode::KeyD => "KeyD",
        KeyCode::KeyE => "KeyE", KeyCode::KeyF => "KeyF",
        KeyCode::KeyG => "KeyG", KeyCode::KeyH => "KeyH",
        KeyCode::KeyI => "KeyI", KeyCode::KeyJ => "KeyJ",
        KeyCode::KeyK => "KeyK", KeyCode::KeyL => "KeyL",
        KeyCode::KeyM => "KeyM", KeyCode::KeyN => "KeyN",
        KeyCode::KeyO => "KeyO", KeyCode::KeyP => "KeyP",
        KeyCode::KeyQ => "KeyQ", KeyCode::KeyR => "KeyR",
        KeyCode::KeyS => "KeyS", KeyCode::KeyT => "KeyT",
        KeyCode::KeyU => "KeyU", KeyCode::KeyV => "KeyV",
        KeyCode::KeyW => "KeyW", KeyCode::KeyX => "KeyX",
        KeyCode::KeyY => "KeyY", KeyCode::KeyZ => "KeyZ",
        KeyCode::Digit1 => "Digit1", KeyCode::Digit2 => "Digit2",
        KeyCode::Digit3 => "Digit3", KeyCode::Digit4 => "Digit4",
        KeyCode::Digit5 => "Digit5", KeyCode::Digit6 => "Digit6",
        KeyCode::Digit7 => "Digit7", KeyCode::Digit8 => "Digit8",
        KeyCode::Digit9 => "Digit9", KeyCode::Digit0 => "Digit0",
        KeyCode::F1 => "F1", KeyCode::F2 => "F2",
        KeyCode::F3 => "F3", KeyCode::F4 => "F4",
        KeyCode::F5 => "F5", KeyCode::F6 => "F6",
        KeyCode::F7 => "F7", KeyCode::F8 => "F8",
        KeyCode::F9 => "F9", KeyCode::F10 => "F10",
        KeyCode::F11 => "F11", KeyCode::F12 => "F12",
        KeyCode::Space => "Space", KeyCode::Enter => "Enter",
        KeyCode::Tab => "Tab", KeyCode::Escape => "Escape",
        KeyCode::Backspace => "Backspace", KeyCode::Delete => "Delete",
        KeyCode::Insert => "Insert", KeyCode::Home => "Home",
        KeyCode::End => "End", KeyCode::PageUp => "PageUp",
        KeyCode::PageDown => "PageDown",
        KeyCode::ArrowUp => "ArrowUp", KeyCode::ArrowDown => "ArrowDown",
        KeyCode::ArrowLeft => "ArrowLeft", KeyCode::ArrowRight => "ArrowRight",
        KeyCode::ShiftLeft => "ShiftLeft", KeyCode::ShiftRight => "ShiftRight",
        KeyCode::ControlLeft => "ControlLeft", KeyCode::ControlRight => "ControlRight",
        KeyCode::AltLeft => "AltLeft", KeyCode::AltRight => "AltRight",
        KeyCode::Minus => "Minus", KeyCode::Equal => "Equal",
        KeyCode::BracketLeft => "BracketLeft", KeyCode::BracketRight => "BracketRight",
        KeyCode::Backslash => "Backslash", KeyCode::Semicolon => "Semicolon",
        KeyCode::Quote => "Quote", KeyCode::Comma => "Comma",
        KeyCode::Period => "Period", KeyCode::Slash => "Slash",
        KeyCode::Backquote => "Backquote",
        _ => "Unknown",
    }
}

/// Convert a winit MouseButton to a platform-independent key name.
pub fn mouse_button_to_key_name(btn: MouseButton) -> &'static str {
    match btn {
        MouseButton::Left => "MouseLeft",
        MouseButton::Right => "MouseRight",
        MouseButton::Middle => "MouseMiddle",
        MouseButton::Back => "MouseButton4",
        MouseButton::Forward => "MouseButton5",
        MouseButton::Other(_) => "Unknown",
    }
}
