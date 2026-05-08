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
//!   * Keyboard events route through `HudTextInputEvent` to the
//!     focused widget's `HudWidget::on_text` — generic across any
//!     widget that captures text (Terminal today, future console /
//!     editor / chat widgets).
//!
//! ## Two flavours of focus target
//!
//! Today the focus pipeline supports both:
//!   * **The held tablet** — focus turns on the moment a tablet
//!     spawns, off when it despawns (via `track_focus_target`).
//!   * **A block-face HUD subblock** — focus is requested
//!     programmatically (`HudFocusState::engage_block_tile`). This is
//!     how E-key on a Terminal block engages its on-block HUD: the
//!     interaction system finds the matching tile entity and writes
//!     it into focus state.

use bevy::input::keyboard::{Key as KbKey, KeyboardInput};
use bevy::input::mouse::MouseButton;
use bevy::input::{keyboard::KeyCode, ButtonInput};
use bevy::prelude::*;

use crate::hud::tablet::HeldTablet;
use crate::hud::tile::{HudAttachment, HudTile};
use crate::hud::widget::{HudTextInput, HudWidgetRegistry};
use crate::input::FrameMouseDelta;
use crate::shard::ShardKey;

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct HudFocusSet;

/// Focus state. `focused_tile` is the tile currently receiving cursor
/// + click + text input. `cursor_uv` is in [0, 1]^2 UV space.
#[derive(Resource, Default, Debug, Clone)]
pub struct HudFocusState {
    pub focused_tile: Option<Entity>,
    pub cursor_uv: Vec2,
    /// When `false`, mouse drives ship / camera. When `true`, mouse
    /// drives the in-world cursor on `focused_tile` and keyboard
    /// input routes to the focused widget.
    pub active: bool,
    /// Focus origin — distinguishes a tablet engagement from a
    /// block-face engagement. The host uses this to decide whether
    /// to run the camera-focus animation (block face only — the
    /// tablet is camera-relative and doesn't need the lerp).
    pub origin: HudFocusOrigin,
}

/// What kind of tile the focus is currently anchored to. Drives
/// downstream behaviour (camera lerp, cursor grab, …) without
/// widgets having to know.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudFocusOrigin {
    /// No tile focused.
    #[default]
    None,
    /// The held tablet is the focused tile.
    Tablet,
    /// A block-face HUD subblock is the focused tile.
    BlockFace,
}

/// Programmatic focus API. Callers (E-key handler on a Terminal
/// block, future "scripted attention" cinematic, …) use this to
/// engage a tile by entity. Symmetrical `disengage` for releasing.
impl HudFocusState {
    /// Engage a block-face tile. Sets focus, activates cursor mode,
    /// re-centres the cursor at (0.5, 0.5). The camera-focus
    /// pipeline picks up the new origin and animates in.
    pub fn engage_block_tile(&mut self, tile: Entity) {
        self.focused_tile = Some(tile);
        self.active = true;
        self.cursor_uv = Vec2::new(0.5, 0.5);
        self.origin = HudFocusOrigin::BlockFace;
    }

    /// Drop block-face focus. Camera-focus animation runs the
    /// disengage lerp; cursor returns to ship/camera control. No-op
    /// if focus is on a tablet (use the tablet despawn path).
    pub fn disengage_block_tile(&mut self) {
        if matches!(self.origin, HudFocusOrigin::BlockFace) {
            self.focused_tile = None;
            self.active = false;
            self.origin = HudFocusOrigin::None;
        }
    }
}

/// Lookup helper: find the HudTile entity attached to a given block
/// face. Used by the E-key engagement path to resolve `block_pos →
/// tile entity` for `engage_block_tile`. Returns the first match;
/// duplicate (shard, block_pos, face) tiles aren't expected — the
/// chunk-respawn pipeline despawns prior tiles before re-creating.
pub fn find_block_face_tile(
    shard: ShardKey,
    block_pos: IVec3,
    face: u8,
    tiles: &Query<(Entity, &HudTile)>,
) -> Option<Entity> {
    for (entity, tile) in tiles.iter() {
        let HudAttachment::Block {
            shard: s,
            block_pos: p,
            face: f,
        } = tile.attachment
        else {
            continue;
        };
        if s == shard && p == block_pos && f == face {
            return Some(entity);
        }
    }
    None
}

/// Loose lookup: find a HudTile by `(block_pos, face)` regardless of
/// shard. Used when the source message doesn't carry a shard id
/// (e.g. server's `OpenTerminalChat` predates shard-tagging).
/// Returns the first match.
pub fn find_block_face_tile_any_shard(
    block_pos: IVec3,
    face: u8,
    tiles: &Query<(Entity, &HudTile)>,
) -> Option<Entity> {
    for (entity, tile) in tiles.iter() {
        let HudAttachment::Block {
            block_pos: p,
            face: f,
            ..
        } = tile.attachment
        else {
            continue;
        };
        if p == block_pos && f == face {
            return Some(entity);
        }
    }
    None
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

/// Routed keyboard event for the focused widget. Drained by the
/// `dispatch_hud_text_input` system in `hud::mod` and forwarded to
/// the widget's `on_text` method.
///
/// Unlike `HudClickEvent`, the focus pipeline pre-resolves the
/// target tile here (it always equals the currently-focused tile);
/// it's stored so downstream logic doesn't have to re-read
/// `HudFocusState`.
#[derive(Message, Debug, Clone)]
pub struct HudTextInputEvent {
    pub tile: Entity,
    pub input: HudTextInput,
}

pub struct HudFocusPlugin;

impl Plugin for HudFocusPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HudFocusState>()
            .add_message::<HudClickEvent>()
            .add_message::<HudTextInputEvent>()
            .configure_sets(Update, HudFocusSet)
            .add_systems(
                Update,
                (
                    track_focus_target,
                    toggle_focus_mode,
                    update_focus_cursor,
                    emit_clicks,
                    emit_text_inputs,
                )
                    .chain()
                    .in_set(HudFocusSet),
            );
    }
}

/// Keep `focused_tile` in sync with the presence of a held tablet.
/// When a tablet appears, focus/cursor mode activates by default
/// (interactive device expects immediate click-to-configure); when
/// it despawns, focus clears. Block-face engagement is set by
/// `HudFocusState::engage_block_tile`; this system only manages
/// tablet bookkeeping and yields when the player is engaged with a
/// block-face tile (so the tablet appearing on top of an engaged
/// terminal doesn't steal focus away).
fn track_focus_target(
    mut focus: ResMut<HudFocusState>,
    tablet: Query<Entity, With<HeldTablet>>,
) {
    if matches!(focus.origin, HudFocusOrigin::BlockFace) {
        // Block-face focus is sticky — handled by E-key handler /
        // disengage path. Don't fight it from the tablet auto-track.
        return;
    }
    let target = tablet.single().ok();
    if focus.focused_tile != target {
        focus.focused_tile = target;
        // Cursor mode ON by default when a tablet appears; OFF when it
        // despawns. Tab still toggles mid-session.
        focus.active = target.is_some();
        focus.cursor_uv = Vec2::new(0.5, 0.5);
        focus.origin = if target.is_some() {
            HudFocusOrigin::Tablet
        } else {
            HudFocusOrigin::None
        };
    }
}

/// Tab toggles focus mode on the currently-focused tile. ESC clears
/// focus (doesn't despawn the tablet — despawn is separately handled
/// by the F-key toggle in interaction::dispatch). Tab is suppressed
/// on block-face engagements: pressing Tab while at a Terminal would
/// otherwise drop the cursor, which is the opposite of the desired
/// "I'm using this terminal" state. Block-face disengagement is
/// E-press-again or Esc.
fn toggle_focus_mode(
    mut focus: ResMut<HudFocusState>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if focus.focused_tile.is_none() {
        focus.active = false;
        focus.origin = HudFocusOrigin::None;
        return;
    }
    let on_block = matches!(focus.origin, HudFocusOrigin::BlockFace);
    if !on_block && keys.just_pressed(KeyCode::Tab) {
        focus.active = !focus.active;
        if focus.active {
            focus.cursor_uv = Vec2::new(0.5, 0.5);
        }
    }
    // Esc OR E disengages a block-face engagement. Pressing E on
    // a Terminal a second time should let go — Star-Citizen
    // convention. The dispatch system is gated separately so this
    // E-press doesn't also fire INTERACT on the same frame.
    let disengage_block = on_block
        && (keys.just_pressed(KeyCode::Escape) || keys.just_pressed(KeyCode::KeyE));
    if disengage_block {
        focus.disengage_block_tile();
    } else if keys.just_pressed(KeyCode::Escape) {
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

/// Drain Bevy `KeyboardInput`, normalise into `HudTextInput`, and
/// emit `HudTextInputEvent` for the focused tile. Only fires when
/// focus is active AND the focused widget self-reports as
/// interactive — passive widgets never see keyboard input even if
/// their tile is somehow focused (defence-in-depth).
///
/// `keyboard_input` events carry the layout-resolved `Key` (the
/// post-shift, post-IME logical key); `Char(c)` is what egui-style
/// text fields want for typed input. We flatten control keys
/// (Backspace / Enter / Esc / arrows / Home / End / Tab) into
/// dedicated variants so widgets don't have to inspect raw key codes.
///
/// Note: this runs in the focus pipeline, not the tablet input
/// pipeline. The tablet's egui driver consumes its own keyboard
/// events through `EguiInputEvent`s; this drainer is for
/// CPU-painted block-face widgets (Terminal, future console / chat
/// widgets, …) that don't go through egui.
fn emit_text_inputs(
    focus: Res<HudFocusState>,
    mut keyboard_events: MessageReader<KeyboardInput>,
    mut events: MessageWriter<HudTextInputEvent>,
    config_q: Query<&crate::hud::tile::HudConfig>,
    registry: Res<HudWidgetRegistry>,
) {
    // Don't `clear()` the keyboard event channel here — that's a
    // *global* drain and would starve every OTHER system that
    // consumes `KeyboardInput` (player movement, free-look toggles,
    // hotbar binds, …). MessageReader has its own per-reader cursor;
    // returning early just means our reader's cursor lags, which is
    // fine — a future Bevy frame's events are reachable via the
    // catch-up mechanism on the next call.
    if !focus.active {
        return;
    }
    // Block-face origin only — the tablet uses its own egui input
    // routing (`tablet_ui::drive_tablet_pointer`), so emitting
    // HudTextInputEvent on tablet focus would double-fire.
    if !matches!(focus.origin, HudFocusOrigin::BlockFace) {
        return;
    }
    let Some(tile) = focus.focused_tile else {
        return;
    };
    // Gate by widget interactivity — a passive widget on a block
    // face shouldn't capture text even when "focused".
    let kind = config_q.get(tile).ok().map(|c| c.kind);
    let interactive = kind
        .and_then(|k| registry.get(k))
        .map(|w| w.is_interactive())
        .unwrap_or(false);
    if !interactive {
        // Diagnostic: if the focus thinks we're on an interactive
        // tile but the lookup says otherwise, that's a routing bug
        // worth logging once.
        let pending = keyboard_events.read().count();
        if pending > 0 {
            tracing::info!(
                ?tile,
                ?kind,
                pending_keyboard_events = pending,
                "emit_text_inputs: focused tile is NOT interactive — dropping keys"
            );
        }
        return;
    }
    for ev in keyboard_events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        // Trace the raw event so we can correlate user keypresses
        // with widget on_text invocations.
        tracing::info!(
            ?tile,
            ?kind,
            logical_key = ?ev.logical_key,
            repeat = ev.repeat,
            "emit_text_inputs: routing key to widget"
        );
        let mapped = match &ev.logical_key {
            KbKey::Character(s) => {
                // Multi-char strings (extremely rare from a single
                // key press, e.g. some IME outputs) emit one
                // `Char` per character.
                for ch in s.chars() {
                    if !ch.is_control() {
                        events.write(HudTextInputEvent {
                            tile,
                            input: HudTextInput::Char(ch),
                        });
                    }
                }
                continue;
            }
            KbKey::Space => Some(HudTextInput::Char(' ')),
            KbKey::Enter => Some(HudTextInput::Enter),
            KbKey::Backspace => Some(HudTextInput::Backspace),
            KbKey::Delete => Some(HudTextInput::Delete),
            KbKey::Escape => Some(HudTextInput::Escape),
            KbKey::Tab => Some(HudTextInput::Tab),
            KbKey::ArrowLeft => Some(HudTextInput::ArrowLeft),
            KbKey::ArrowRight => Some(HudTextInput::ArrowRight),
            KbKey::ArrowUp => Some(HudTextInput::ArrowUp),
            KbKey::ArrowDown => Some(HudTextInput::ArrowDown),
            KbKey::Home => Some(HudTextInput::Home),
            KbKey::End => Some(HudTextInput::End),
            _ => None,
        };
        if let Some(input) = mapped {
            events.write(HudTextInputEvent { tile, input });
        }
    }
}
