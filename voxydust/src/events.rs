//! Client-side ECS messages and input context.
//!
//! Raw input events from winit are buffered in `RawInputBuffer`, then drained
//! into `MessageWriter<T>` by `bridge_raw_input_system` at the start of the
//! ECS schedule. The `interpret_input_system` reads raw messages, checks
//! `InputContext`, and produces game actions (resource mutations or further
//! messages). This replaces the monolithic `window_event()` key handler.

use bevy_ecs::prelude::*;
use winit::keyboard::KeyCode;

// ---------------------------------------------------------------------------
// Raw input messages (bridge from winit → ECS)
// ---------------------------------------------------------------------------

#[derive(Message)]
pub struct KeyPressedMsg {
    pub key: KeyCode,
}

#[derive(Message)]
pub struct KeyReleasedMsg {
    pub key: KeyCode,
}

#[derive(Message)]
pub struct MouseButtonMsg {
    pub button: winit::event::MouseButton,
    pub pressed: bool,
}

#[derive(Message)]
pub struct MouseMotionMsg {
    pub delta_x: f64,
    pub delta_y: f64,
}

// ---------------------------------------------------------------------------
// Semantic messages (produced by interpret_input_system, consumed by systems)
// ---------------------------------------------------------------------------

/// Request to grab or ungrab the OS cursor.
/// Processed post-update in `apply_cursor_changes()` (needs !Send Window).
#[derive(Message)]
pub struct CursorGrabRequest {
    pub grabbed: bool,
}

/// Request to close the config panel (ESC, Cancel button, Save button).
#[derive(Message)]
pub struct ConfigPanelCloseMsg;

/// Server sent a block config snapshot — open the panel.
#[derive(Message)]
pub struct ConfigPanelOpenMsg {
    pub config: voxeldust_core::signal::config::BlockSignalConfig,
}

// ---------------------------------------------------------------------------
// Input context: single authority for input mode
// ---------------------------------------------------------------------------

/// What the input system should do with raw events.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum InputMode {
    /// Normal gameplay — WASD moves, mouse looks, E/F interact.
    #[default]
    Game,
    /// A UI panel is open — suppress all game input, only ESC works.
    UiPanel,
    /// Camera is animating (config lerp in/out) — suppress all input.
    Animating,
    /// Cursor is ungrabbed (pre-grab state) — LMB re-grabs.
    MenuFocus,
}

/// Single source of truth for current input routing and cursor state.
#[derive(Resource)]
pub struct InputContext {
    pub mode: InputMode,
    pub cursor_grabbed: bool,
}

impl Default for InputContext {
    fn default() -> Self {
        Self {
            mode: InputMode::MenuFocus,
            cursor_grabbed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Raw input buffer (winit → ECS bridge)
// ---------------------------------------------------------------------------

/// Mouse scroll event (for thrust limiter).
#[derive(Message)]
pub struct MouseScrollMsg {
    pub delta_y: f32,
}

/// Accumulates raw input events from `window_event()` / `device_event()`
/// outside the ECS schedule. Drained into `MessageWriter<T>` by
/// `bridge_raw_input_system` at the top of each frame.
#[derive(Resource, Default)]
pub struct RawInputBuffer {
    pub key_presses: Vec<KeyPressedMsg>,
    pub key_releases: Vec<KeyReleasedMsg>,
    pub mouse_buttons: Vec<MouseButtonMsg>,
    pub mouse_motions: Vec<MouseMotionMsg>,
    pub mouse_scrolls: Vec<MouseScrollMsg>,
}
