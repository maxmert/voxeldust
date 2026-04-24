//! HUD focus mode — "walk up to a monitor" interaction.
//!
//! **Default behaviour when the held tablet spawns: cursor mode is
//! ON.** The tablet is an interactive device, so the expected
//! immediate action is click-to-configure. Pressing **Tab** flips to
//! free-move (mouse drives ship / camera again while the tablet
//! remains open on-screen — useful when manoeuvring during config).
//! Tab again returns to cursor mode. OS cursor stays grabbed
//! throughout — no CursorGrabMode toggling.
//!
//! When focus is active:
//!   * Mouse motion accumulates into `HudFocusState.cursor_uv` instead
//!     of ship yaw / camera look.
//!   * An in-world cursor is painted onto the focused tile (via egui
//!     for the held tablet; via `draw_cursor` in `texture.rs` for
//!     CPU-painted block tiles).
//!   * LMB / RMB / MMB at the current `cursor_uv` fires a
//!     `HudClickEvent`; widgets consume via their `HudWidget::on_click`
//!     method OR, for the held tablet, feed egui synthetic pointer
//!     events (see `hud::tablet_ui::drive_tablet_pointer`).

use bevy::input::mouse::MouseButton;
use bevy::input::{keyboard::KeyCode, ButtonInput};
use bevy::prelude::*;

use crate::hud::tablet::HeldTablet;
use crate::input::FrameMouseDelta;

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct HudFocusSet;

/// Focus state. `focused_tile` is the tile currently receiving cursor
/// + click input (today: the held tablet). `cursor_uv` is in [0, 1]^2
/// UV space.
#[derive(Resource, Default, Debug, Clone)]
pub struct HudFocusState {
    pub focused_tile: Option<Entity>,
    pub cursor_uv: Vec2,
    /// When `false`, mouse drives ship / camera. When `true`, mouse
    /// drives the in-world cursor on `focused_tile`.
    pub active: bool,
}

/// Fired once per LMB press while focus is active. Carries the tile
/// entity being clicked and the UV in [0, 1]^2. Widgets read this to
/// decide whether to emit a `PublishSignalEvent` etc.
#[derive(Message, Debug, Clone)]
pub struct HudClickEvent {
    pub tile: Entity,
    pub uv: Vec2,
    pub button: HudClickButton,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudClickButton {
    Left,
    Right,
    Middle,
}

pub struct HudFocusPlugin;

impl Plugin for HudFocusPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HudFocusState>()
            .add_message::<HudClickEvent>()
            .configure_sets(Update, HudFocusSet)
            .add_systems(
                Update,
                (
                    track_focus_target,
                    toggle_focus_mode,
                    update_focus_cursor,
                    emit_clicks,
                )
                    .chain()
                    .in_set(HudFocusSet),
            );
    }
}

/// Keep `focused_tile` in sync with the presence of a held tablet.
/// When a tablet appears, focus/cursor mode activates by default
/// (interactive device expects immediate click-to-configure); when it
/// despawns, focus clears. Future: wall-mounted tiles entered via
/// E-raycast set focused_tile to that tile's entity (see H7 follow-up).
fn track_focus_target(
    mut focus: ResMut<HudFocusState>,
    tablet: Query<Entity, With<HeldTablet>>,
) {
    let target = tablet.single().ok();
    if focus.focused_tile != target {
        focus.focused_tile = target;
        // Cursor mode ON by default when a tablet appears; OFF when it
        // despawns. Tab still toggles mid-session.
        focus.active = target.is_some();
        focus.cursor_uv = Vec2::new(0.5, 0.5);
    }
}

/// Tab toggles focus mode on the currently-focused tile. ESC clears
/// focus (doesn't despawn the tablet — despawn is separately handled
/// by the F-key toggle in interaction::dispatch).
fn toggle_focus_mode(
    mut focus: ResMut<HudFocusState>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if focus.focused_tile.is_none() {
        focus.active = false;
        return;
    }
    if keys.just_pressed(KeyCode::Tab) {
        focus.active = !focus.active;
        if focus.active {
            focus.cursor_uv = Vec2::new(0.5, 0.5);
        }
    }
    if keys.just_pressed(KeyCode::Escape) {
        focus.active = false;
    }
}

/// When focus is active, accumulate mouse delta into `cursor_uv`
/// (clamped to `[0, 1]^2`). Cursor sensitivity: 1 pixel ≈ 0.0025 UV
/// — a 400-pixel sweep crosses the full tile. Kept constant for MVP;
/// future iteration reads `ControlConfig::focus_cursor_sensitivity`.
fn update_focus_cursor(
    mut focus: ResMut<HudFocusState>,
    mouse: Res<FrameMouseDelta>,
) {
    const SENSITIVITY: f32 = 0.0025;
    if !focus.active {
        return;
    }
    focus.cursor_uv.x = (focus.cursor_uv.x + mouse.dx * SENSITIVITY).clamp(0.0, 1.0);
    focus.cursor_uv.y = (focus.cursor_uv.y + mouse.dy * SENSITIVITY).clamp(0.0, 1.0);
}

/// Emit a `HudClickEvent` on LMB/RMB/MMB while focus is active.
fn emit_clicks(
    focus: Res<HudFocusState>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut events: MessageWriter<HudClickEvent>,
) {
    if !focus.active {
        return;
    }
    let Some(tile) = focus.focused_tile else { return };
    if mouse.just_pressed(MouseButton::Left) {
        events.write(HudClickEvent {
            tile,
            uv: focus.cursor_uv,
            button: HudClickButton::Left,
        });
    }
    if mouse.just_pressed(MouseButton::Right) {
        events.write(HudClickEvent {
            tile,
            uv: focus.cursor_uv,
            button: HudClickButton::Right,
        });
    }
    if mouse.just_pressed(MouseButton::Middle) {
        events.write(HudClickEvent {
            tile,
            uv: focus.cursor_uv,
            button: HudClickButton::Middle,
        });
    }
}
