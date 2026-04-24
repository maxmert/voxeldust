//! `HudWidget` trait + registry. Adding a new widget kind is a single
//! module under `widgets/` plus one `register(...)` line.

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

/// Pure trait — no Bevy types, so widgets can be unit-tested without a
/// world. The registry stores `Box<dyn HudWidget>`.
pub trait HudWidget: Send + Sync + 'static {
    fn kind(&self) -> WidgetKind;
    fn name(&self) -> &'static str;

    /// Which SignalProperties this widget can render. The config UI
    /// filters the widget dropdown to those matching the selected
    /// channel's property type.
    fn supported_properties(&self) -> &'static [SignalProperty];

    /// Paint the widget onto the tile pixel buffer given the current
    /// signal value + config. Value is `None` when no signal is
    /// subscribed yet or the channel isn't in the registry — widget
    /// renders a "no signal" / idle state.
    fn draw(&self, ctx: DrawCtx, value: Option<SignalValue>, config: &HudConfig);

    /// Handle a click at tile-UV `uv` in `[0, 1]^2`. The default
    /// implementation is a no-op (static readouts like `GaugeWidget`
    /// ignore clicks). Button / slider / toggle widgets override this
    /// to emit a `PublishSignalEvent` for the bound channel.
    ///
    /// Returning `Some(ClickAction::Publish { … })` instructs the
    /// dispatcher to emit a publish; `None` means the click missed
    /// any active hotspot on this widget.
    fn on_click(
        &self,
        _uv: bevy::prelude::Vec2,
        _button: crate::hud::focus::HudClickButton,
        _value: Option<&SignalValue>,
        _config: &HudConfig,
    ) -> Option<ClickAction> {
        None
    }
}

/// Result of `HudWidget::on_click`. The dispatcher translates these
/// into concrete effects (`PublishSignalEvent`, focus-exit, etc.).
#[derive(Debug, Clone)]
pub enum ClickAction {
    /// Publish a signal value to a channel. Server validates
    /// `publish_policy` before accepting.
    Publish {
        channel: String,
        value: voxeldust_core::signal::types::SignalValue,
    },
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
