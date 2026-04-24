//! HUD tile components + lifecycle.

use bevy::prelude::*;

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::ar::ArFilter;
use crate::shard::ShardKey;

/// Marker component on every HUD tile entity — world-placed sub-block
/// tiles, held-tablet tiles, or anything rendering HUD widget content
/// onto a texture. Carries the owning shard + block-space identity so
/// input / config routing can address it.
#[derive(Component, Debug, Clone, Copy)]
pub struct HudTile {
    /// `None` for the held-tablet (not block-attached).
    pub attachment: HudAttachment,
}

#[derive(Debug, Clone, Copy)]
pub enum HudAttachment {
    /// Attached to a block face on a shard.
    Block {
        shard: ShardKey,
        block_pos: IVec3,
        face: u8,
    },
    /// Camera-relative held tablet.
    Tablet,
}

/// Declarative widget kind — the discriminant picked in the config UI.
/// `HudWidgetRegistry` maps each variant to its `HudWidget` trait impl.
/// Adding a kind = adding a variant + registering the impl.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WidgetKind {
    /// No widget content. The tile is transparent except for any AR
    /// markers the config has enabled. Useful when the player wants
    /// a panel to be AR-only (e.g. a cockpit window framed with a
    /// compass band) or a placeholder before picking a real widget.
    None,
    Gauge,
    Numeric,
    Toggle,
    Text,
    ConfigPanel,
    /// Publisher button — clicking the tile while in focus mode
    /// publishes `press_value` to the configured channel; releasing
    /// publishes `release_value`. For now only the press fires;
    /// release-tracking lands when we thread mouse-up through.
    Button,
    // Future: Graph, Compass, Speedo, …
}

/// Per-tile config. Server-authored via `SubBlockConfigState` for block
/// tiles; set programmatically for held tablets.
#[derive(Component, Debug, Clone)]
pub struct HudConfig {
    pub kind: WidgetKind,
    pub channel: String,
    pub property: SignalProperty,
    pub caption: String,
    /// Widget content opacity 0..1 (background stays fully transparent).
    pub opacity: f32,
    /// When `true`, AR markers overlay the widget content.
    pub ar_enabled: bool,
    /// What kinds of entities to mark when `ar_enabled`.
    pub ar_filter: ArFilter,
    /// Optional tile-specific payload — e.g., a `ConfigPanel` widget
    /// stashes the target block's `BlockSignalConfig` here.
    pub payload: HudPayload,
}

impl Default for HudConfig {
    fn default() -> Self {
        Self {
            kind: WidgetKind::Numeric,
            channel: String::new(),
            property: SignalProperty::Throttle,
            caption: String::new(),
            opacity: 0.85,
            ar_enabled: false,
            ar_filter: ArFilter::default(),
            payload: HudPayload::None,
        }
    }
}

/// Variant payload attached to a tile's config. Widgets that need
/// richer per-tile data than a single channel/property pair store it
/// here.
#[derive(Debug, Clone, Default)]
pub enum HudPayload {
    #[default]
    None,
    /// For `ConfigPanel` widget — the target block's current server-
    /// authored config to edit.
    ConfigPanel(Box<voxeldust_core::signal::config::BlockSignalConfig>),
}

/// GPU texture the widget draws into each redraw pass. Size is fixed at
/// tile spawn (configured via `ControlConfig` later; 256×256 default).
///
/// We also keep the material handle here so the redraw system can
/// **bump it via `materials.get_mut(&handle)` each frame**. That is
/// the known workaround for Bevy issue #17350 — modifying an Image
/// asset alone doesn't propagate to GPU; bumping the material forces
/// change-detection that cascades into re-uploading the texture.
#[derive(Component, Debug, Clone)]
pub struct HudTexture {
    pub handle: Handle<Image>,
    pub material: Handle<StandardMaterial>,
    pub size: u32,
    /// World-space size of this tile's rectangle mesh in metres.
    /// Used by AR projection to map ray-plane hits into UV space —
    /// a held tablet reads 0.34×0.23, block-face HUD panels ~0.9×0.9.
    pub world_size: Vec2,
    /// Last `SignalRegistry.last_tick` the widget drew at — used to
    /// skip redraws when nothing changed.
    pub last_draw_tick: u64,
    /// Floor cadence for redraw (ms) even if no signal changed, so
    /// animations / cursors / AR markers stay fresh.
    pub redraw_floor_ms: u64,
    pub last_draw_at: std::time::Instant,
}

pub struct HudTilePlugin;

impl Plugin for HudTilePlugin {
    fn build(&self, _app: &mut App) {
        // No systems yet; presence of the plugin exists so
        // `HudPlugin::build` orders add correctly.
    }
}
