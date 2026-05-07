//! `HudWidget` trait + registry. Adding a new widget kind is a single
//! module under `widgets/` plus one `register(...)` line.
//!
//! ## Stateful widgets
//!
//! Widgets can opt into per-tile mutable state by overriding
//! `init_state` and reading the same `state` argument from `draw`,
//! `on_click`, and `on_text`. The host stores it as a
//! `HudWidgetState` component on the tile entity; lifecycle is
//! automatic — created on first interaction, despawned with the tile.
//!
//! Stateless widgets (gauge, numeric, toggle, button, text readout)
//! return `None` from `init_state` and ignore the `state` argument —
//! the existing implementations stay just as small as before.
//!
//! ## Input surface
//!
//! Beyond mouse clicks, widgets can opt into keyboard input via
//! `on_text`. The host's focus pipeline drains `KeyboardInput` events
//! while a tile is focused and routes a normalised
//! [`HudTextInput`] sequence to the widget — this works for the
//! tablet, for in-world block-face HUD subblocks, or for any future
//! tile carrier (helmet HUD, ship cockpit MFDs, …) without each
//! widget knowing where it lives.
//!
//! ## Generic action
//!
//! Both `on_click` and `on_text` return [`WidgetAction`], a
//! widget-agnostic contract for the host's dispatcher: publish a
//! signal, send a chat-style block message, or do nothing. New action
//! kinds plug in here without touching widgets that don't emit them.

use std::collections::HashMap;

use bevy::prelude::*;

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, WidgetKind};

/// Context passed to a widget's `draw()` each redraw pass. Gives the
/// widget the tile's pixel buffer, its current config, the value to
/// render, and the tile size.
pub struct DrawCtx<'a> {
    /// RGBA8 pixel buffer, `size * size * 4` bytes. Transparent
    /// background already cleared; widget writes opaque pixels where
    /// it wants content.
    pub pixels: &'a mut [u8],
    pub size: u32,
    pub caption: &'a str,
    pub opacity: f32,
}

/// Per-tile widget state — opaque, owned by the host. A widget that
/// declares `init_state` returns a `Box<dyn HudWidgetStateData>`
/// which the host keeps alive in a [`HudWidgetState`] component on
/// the tile entity. Widget impls downcast `&mut dyn HudWidgetStateData`
/// via [`HudWidgetStateData::as_any_mut`] in their `draw`/`on_*`
/// methods.
///
/// The bound is `Any + Send + Sync + 'static` because Bevy's
/// component storage requires both Send + Sync; `Any` enables the
/// downcast.
pub trait HudWidgetStateData: std::any::Any + Send + Sync + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Blanket impl: every `Send + Sync + 'static` type is automatically
/// usable as widget state. Widget authors just write a normal struct
/// (no trait derive) and the box-it-up pattern works.
impl<T: std::any::Any + Send + Sync + 'static> HudWidgetStateData for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Bevy component storing per-tile widget state.
///
/// One per stateful tile. Inserted lazily by the texture redraw
/// pipeline the first frame after `kind` resolves to a widget that
/// returns `Some(_)` from `init_state`. Cleared (and re-initialised)
/// when the tile's `WidgetKind` changes — preserving state across
/// kind swaps would let a Terminal's scrollback leak into a Gauge.
#[derive(Component)]
pub struct HudWidgetState {
    /// Which widget kind owns this state — a sanity tag so the
    /// dispatcher can detect kind swaps and re-init.
    pub kind: WidgetKind,
    /// Type-erased payload. Always non-empty for components that
    /// exist; we never insert `None`.
    pub data: Box<dyn HudWidgetStateData>,
}

/// Normalised text-input event delivered to a focused widget's
/// `on_text` method. Keyboard layouts / IME / repeat are resolved by
/// the host before dispatch; widgets only see the final character or
/// named key.
#[derive(Debug, Clone)]
pub enum HudTextInput {
    /// A user-typed character (post-layout; e.g. shifted symbols
    /// already resolved). Excludes control characters — those arrive
    /// as `Key` variants.
    Char(char),
    /// Backspace: delete one character to the left of the caret.
    Backspace,
    /// Delete: delete one character to the right of the caret.
    Delete,
    /// Enter / Return: submit / newline (widget-defined).
    Enter,
    /// Escape: a widget-meaningful cancel (close suggestion, clear
    /// input, etc.). The host's focus pipeline ALSO uses Escape as a
    /// "drop focus" signal — widgets that want to swallow Escape
    /// before it propagates can return `Some(WidgetAction::Consumed)`.
    Escape,
    /// Tab: focus next field within the widget (widget-defined).
    Tab,
    /// Arrow keys: caret movement (widget-defined).
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    /// Home / End within the input.
    Home,
    End,
}

/// Pure trait — no Bevy types, so widgets can be unit-tested without a
/// world. The registry stores `Box<dyn HudWidget>`.
pub trait HudWidget: Send + Sync + 'static {
    fn kind(&self) -> WidgetKind;
    fn name(&self) -> &'static str;

    /// Which SignalProperties this widget can render. The config UI
    /// filters the widget dropdown to those matching the selected
    /// channel's property type.
    fn supported_properties(&self) -> &'static [SignalProperty];

    /// Whether this widget is interactive — i.e. should be eligible
    /// for HUD focus when the player presses E on its host block. The
    /// default is `false` (gauges, numerics, text readouts are passive
    /// displays). Buttons / sliders / Terminal / future text-input
    /// widgets override to `true`.
    fn is_interactive(&self) -> bool {
        false
    }

    /// Build the per-tile state when the widget first attaches. Return
    /// `None` for stateless widgets. Default: stateless.
    ///
    /// Called once per (tile, widget-kind) pair: when the tile is
    /// spawned with this widget's kind, OR when an existing tile's
    /// kind changes to this widget. The state outlives a single
    /// engagement — Terminal scrollback persists when the player
    /// disengages and re-engages.
    fn init_state(
        &self,
        _config: &HudConfig,
    ) -> Option<Box<dyn HudWidgetStateData>> {
        None
    }

    /// Paint the widget onto the tile pixel buffer given the current
    /// signal value + config + per-tile state. Value is `None` when
    /// no signal is subscribed yet or the channel isn't in the
    /// registry — widget renders a "no signal" / idle state. State is
    /// `None` for stateless widgets.
    fn draw(
        &self,
        ctx: DrawCtx,
        value: Option<SignalValue>,
        state: Option<&mut dyn HudWidgetStateData>,
        config: &HudConfig,
    );

    /// Handle a click at tile-UV `uv` in `[0, 1]^2`. Default no-op;
    /// passive widgets ignore clicks. State is `None` for stateless
    /// widgets.
    fn on_click(
        &self,
        _uv: bevy::prelude::Vec2,
        _button: crate::hud::focus::HudClickButton,
        _value: Option<&SignalValue>,
        _state: Option<&mut dyn HudWidgetStateData>,
        _config: &HudConfig,
    ) -> Option<WidgetAction> {
        None
    }

    /// Handle a normalised keyboard event. Default no-op; only
    /// widgets that capture text override this. Widgets returning a
    /// non-`None` action bypass the host's default Escape→drop-focus
    /// behaviour for that single event.
    fn on_text(
        &self,
        _event: HudTextInput,
        _state: Option<&mut dyn HudWidgetStateData>,
        _config: &HudConfig,
    ) -> Option<WidgetAction> {
        None
    }
}

/// Result of `HudWidget::on_click` / `on_text`. The dispatcher
/// translates these into concrete effects (`PublishSignalEvent`,
/// terminal chat send, etc.). New variants are added as new widget
/// classes need them — no widget that doesn't emit a variant has to
/// know it exists.
#[derive(Debug, Clone)]
pub enum WidgetAction {
    /// Publish a signal value to a channel. Server validates
    /// `publish_policy` before accepting.
    Publish {
        channel: String,
        value: voxeldust_core::signal::types::SignalValue,
    },
    /// Send a Terminal-style chat line targeted at a specific block.
    /// The dispatcher translates this into
    /// `ClientMsg::TerminalChatSend`. Generic across any widget that
    /// needs an "Enter to send" affordance bound to a block (today:
    /// `TerminalWidget`; tomorrow: console / log-tail / IRC widgets).
    SendChat { block_pos: IVec3, text: String },
    /// Acknowledge that the widget consumed the event and the host
    /// should NOT apply default behaviour (e.g. don't drop focus on
    /// Escape if the widget used Escape to dismiss its own
    /// suggestion popup).
    Consumed,
}

/// Boxed-trait-object registry. Keyed on `WidgetKind`.
#[derive(Resource, Default)]
pub struct HudWidgetRegistry {
    by_kind: HashMap<WidgetKind, Box<dyn HudWidget>>,
}

impl HudWidgetRegistry {
    pub fn register(&mut self, widget: Box<dyn HudWidget>) {
        let kind = widget.kind();
        if self.by_kind.insert(kind, widget).is_some() {
            tracing::warn!(?kind, "HudWidgetRegistry::register replaced an existing widget");
        }
    }

    pub fn get(&self, kind: WidgetKind) -> Option<&dyn HudWidget> {
        self.by_kind.get(&kind).map(|b| b.as_ref())
    }

    pub fn kinds(&self) -> impl Iterator<Item = WidgetKind> + '_ {
        self.by_kind.keys().copied()
    }
}

/// Backwards-compat alias. Older call sites used `ClickAction`; new
/// code uses `WidgetAction`. Removed once the rename has propagated
/// through any downstream forks.
pub type ClickAction = WidgetAction;
