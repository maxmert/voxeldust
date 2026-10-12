//! ★ THE TEMPORAL PATH'S MEASUREMENT SWITCHES (slice T0 of
//! `docs/investigation/2026-09-28/temporal_path_design.md`; ruling B24). Read ONCE at startup, by the
//! window and by the headless capture alike, because the picture census (G-PICTURE-HOLES) runs on the
//! capture's own picture camera. None is a product setting: each prices one piece of the frame, or
//! prepares a picture for a meter, and T8 ends with one path (the owner's law: one world, no second
//! implementation chosen by a flag).
//!
//! - `VD_RENDER_SCALE=<s>` (0 < s ≤ 1): the main passes draw the top-left `s·w × s·h` rectangle of
//!   the camera's targets — the engine's own override (`MainPassResolutionOverride`, the design's F4),
//!   inserted as the engine's DLSS plug-in inserts it. The post passes do not rescale, so the picture
//!   sits in a corner: T0 reads only the GPU clock at each scale (M-2), never the picture.
//! - `VD_PREPASS=1`: the engine's depth and motion prepass on the picture's camera, with MSAA off —
//!   route G3's price and its holes (M-3), and T1's motion: the ground's prepass writes each pixel's
//!   motion, the motion target clears to the motion census's sentinel, and the headless capture reads
//!   the motion and the depth back beside each picture (`motion_capture`). Nothing draws from the
//!   motion yet (T2 does).
//! - `VD_PREPASS=depth`: the depth prepass alone, with MSAA off: the motion's price is the clock's
//!   difference to `VD_PREPASS=1` (T1, row V of the design's budget).
//! - `VD_STARS=0`: the star cloud is not built (M-4; the picture census wants no star on its sky).
//! - `VD_CLEAR_SENTINEL=1`: the picture clears to the census's sentinel (pure cyan,
//!   `vd_client_harness::picture_census::SENTINEL_RGB8`), a colour no lit ground can take, so a pixel
//!   that no geometry wrote keeps it (the picture census; run it with `VD_SKY=0`).
//! - `VD_TEMPORAL=1` (T2): THE RECONSTRUCTION — the picture camera draws a jittered render picture
//!   through the depth and motion prepass, MSAA off, and `reconstruction` rebuilds the display picture
//!   from it and its history; with `VD_RENDER_SCALE` the render picture is smaller than the display
//!   (held by the floor law) and the reconstruction up-scales it, so the picture fills the window.
//!   ★ THE CAPTURE KEEPS ITS GRID (the design §12.1): under both switches the headless capture's
//!   render picture stays the reference view (1 284 × 720) and its display picture grows to that
//!   grid ÷ s ([`RenderMeasure::capture_display`], [`RenderGrid`]).
//! - `VD_SHARPEN=<0..1>` (T2, the owner's look at D7): the sharpen's strength after the tone map under
//!   the reconstruction; 0 turns it off; unset, the settings' PROPOSED strength.
//! - `VD_PRICE_G2_TARGETS=1`: one pass a frame clears and stores four display-size textures in the
//!   deferred route's formats, its viewport at the drawn rectangle — row W of the design's budget,
//!   priced before T3 builds the route (M-2b).
//!
//! The census's control (a rung drawn one depth step nearer in the prepass than in the main pass) is
//! the terrain's own switch, `VD_CENSUS_CONTROL_RUNG` (`terrain::TerrainConfig::census_control_rung`).
//!
//! Example: the owner's stand on foot, full screen, MSAA off, `VD_RENDER_SCALE=0.5`: the ground and
//! the air draw 1 728 × 1 084 pixels in the window's top-left quarter, and the GPU clock reads what
//! half the width and half the height cost.

use bevy::camera::MainPassResolutionOverride;
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::ScheduleSystem;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
};
use bevy::render::render_resource::{Texture, TextureView};
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice};
use bevy::render::{Render, RenderApp, RenderSystems};

const RENDER_SCALE_ENV: &str = "VD_RENDER_SCALE";
const PREPASS_ENV: &str = "VD_PREPASS";
const STARS_ENV: &str = "VD_STARS";
const SENTINEL_ENV: &str = "VD_CLEAR_SENTINEL";
const PRICE_G2_ENV: &str = "VD_PRICE_G2_TARGETS";
const TEMPORAL_ENV: &str = "VD_TEMPORAL";
const SHARPEN_ENV: &str = "VD_SHARPEN";

/// What the measurement switches said at startup (ONE struct, read once).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct RenderMeasure {
    /// The drawn rectangle's share of each side, if a measurement names one.
    pub render_scale: Option<f32>,
    pub prepass: Prepass,
    pub stars: bool,
    pub sentinel: bool,
    pub price_g2_targets: bool,
    /// ★ THE RECONSTRUCTION (T2, `VD_TEMPORAL=1`): the picture camera draws a jittered render
    /// picture through the depth and motion prepass, MSAA off, at the render scale held by the floor
    /// law, and `reconstruction` rebuilds the display picture from it.
    pub temporal: bool,
    /// The sharpen's strength under the reconstruction (`VD_SHARPEN`; the owner's D7).
    pub sharpen: f32,
}

impl Default for RenderMeasure {
    fn default() -> RenderMeasure {
        RenderMeasure {
            render_scale: None,
            prepass: Prepass::Off,
            stars: true,
            sentinel: false,
            price_g2_targets: false,
            temporal: false,
            sharpen: vd_client::reconstruction::ReconstructionSettings::PROPOSED.sharpen as f32,
        }
    }
}

/// ★ THE WINDOW'S RENDER SCALE BY DEFAULT (ruling B90): the scale the owner judged in flight under the reconstruction (B29:
/// "flying at 0.67 — All good!"). MEASURED in the window at 2560 × 1440 on the owner's M4 Pro (2026-10-12): a frame every
/// 16.8 ms (60 a second) against 32 ms at full size with 4× MSAA, where the ground and the sky's per-sample march pass the
/// display's beat.
pub(crate) const WINDOW_RENDER_SCALE: f32 = 0.67;

impl RenderMeasure {
    /// ★ THE WINDOW'S OWN SWITCHES (ruling B90, the owner: "let's take the most optimized version please"): the
    /// reconstruction ON by default at [`WINDOW_RENDER_SCALE`]; `VD_TEMPORAL=0` turns it off for a run and
    /// `VD_RENDER_SCALE` states another scale. The headless capture keeps [`RenderMeasure::from_env`]'s defaults: its
    /// pictures are the gates' references, which change only on the owner's word (B71).
    pub(crate) fn from_env_window() -> RenderMeasure {
        let mut measure = RenderMeasure::from_env();
        measure.temporal = std::env::var(TEMPORAL_ENV).map_or(true, |v| v.trim() != "0");
        if measure.temporal && measure.render_scale.is_none() {
            measure.render_scale = Some(WINDOW_RENDER_SCALE);
        }
        measure
    }

    pub(crate) fn from_env() -> RenderMeasure {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v.trim() == "1");
        RenderMeasure {
            render_scale: std::env::var(RENDER_SCALE_ENV)
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|s| s.is_finite() && *s > 0.0 && *s <= 1.0),
            prepass: Prepass::from_switch(std::env::var(PREPASS_ENV).ok().as_deref()),
            stars: std::env::var(STARS_ENV).map_or(true, |v| v.trim() != "0"),
            sentinel: on(SENTINEL_ENV),
            price_g2_targets: on(PRICE_G2_ENV),
            temporal: on(TEMPORAL_ENV),
            sharpen: std::env::var(SHARPEN_ENV)
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|s| (0.0..=1.0).contains(s))
                .unwrap_or(RenderMeasure::default().sharpen),
        }
    }

    /// ★ THE CAPTURE'S DISPLAY SIZE for its render grid (the design §12.1): the grid ÷ s under the
    /// reconstruction with a render scale (`vd_client::reconstruction::display_size`), else the grid.
    pub(crate) fn capture_display(self, grid: UVec2) -> UVec2 {
        match (self.temporal, self.render_scale) {
            (true, Some(s)) => UVec2::from_array(vd_client::reconstruction::display_size(
                grid.to_array(),
                f64::from(s),
            )),
            _ => grid,
        }
    }

    /// The picture's clear colour: the census's sentinel under its switch (named once, in the
    /// census's own module, so the renderer and the gate cannot state two colours), else the
    /// caller's own. ★ Under the reconstruction its alpha is NOUGHT (the design §7.7, change one): the
    /// sky share of a render pixel is one less its alpha, so the clear's alpha must say "all sky"; the
    /// reconstruction writes the display picture's alpha one again (change two).
    pub(crate) fn clear_color(self, own_srgb: [f32; 3]) -> ClearColor {
        let color = if self.sentinel {
            let [r, g, b] = vd_client_harness::picture_census::SENTINEL_RGB8;
            Color::srgb_u8(r, g, b)
        } else {
            let [r, g, b] = own_srgb;
            Color::srgb(r, g, b)
        };
        ClearColor(if self.temporal {
            color.with_alpha(0.0)
        } else {
            color
        })
    }

    /// The picture camera's extra components under the prepass switch: the engine's depth prepass,
    /// with MSAA off (the design: route G3's price is taken without MSAA). The motion prepass rides
    /// beside it ([`RenderMeasure::motion_prepass`]).
    pub(crate) fn prepass_bundle(
        self,
    ) -> Option<(bevy::core_pipeline::prepass::DepthPrepass, Msaa)> {
        (self.temporal || self.prepass != Prepass::Off)
            .then_some((bevy::core_pipeline::prepass::DepthPrepass, Msaa::Off))
    }

    /// The engine's motion prepass under `VD_PREPASS=1` (T1: the ground writes its motion).
    pub(crate) fn motion_prepass(
        self,
    ) -> Option<bevy::core_pipeline::prepass::MotionVectorPrepass> {
        (self.temporal || self.prepass == Prepass::Motion)
            .then_some(bevy::core_pipeline::prepass::MotionVectorPrepass)
    }

    /// The picture camera's reconstruction under `VD_TEMPORAL=1`: its jitter, its mark, and the sharpen
    /// after the tone map (the engine's contrast-adaptive sharpening at `VD_SHARPEN`'s strength; the
    /// owner's D7).
    pub(crate) fn reconstruction_bundle(
        self,
    ) -> Option<(
        bevy::render::camera::TemporalJitter,
        crate::reconstruction::ReconstructionCamera,
        bevy::anti_alias::contrast_adaptive_sharpening::ContrastAdaptiveSharpening,
    )> {
        self.temporal.then_some((
            bevy::render::camera::TemporalJitter::default(),
            crate::reconstruction::ReconstructionCamera,
            bevy::anti_alias::contrast_adaptive_sharpening::ContrastAdaptiveSharpening {
                enabled: self.sharpen > 0.0,
                sharpening_strength: self.sharpen,
                denoise: false,
            },
        ))
    }

    /// The probe camera's jitter under `VD_TEMPORAL=1`: it takes the picture's (the design §12.1).
    pub(crate) fn probe_jitter_bundle(
        self,
    ) -> Option<(
        bevy::render::camera::TemporalJitter,
        crate::reconstruction::JitterFollower,
    )> {
        self.temporal.then_some((
            bevy::render::camera::TemporalJitter::default(),
            crate::reconstruction::JitterFollower,
        ))
    }
}

/// ★ A CAMERA'S FIXED RENDER GRID (the capture, the design §12.1): its main passes draw exactly this
/// top-left rectangle of its display-size targets, whatever the scale's rounding.
#[derive(Component, Clone, Copy, bevy::render::extract_component::ExtractComponent)]
pub(crate) struct RenderGrid(pub UVec2);

/// ★ WHICH PREPASS the picture's camera draws (`VD_PREPASS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Prepass {
    /// No prepass (the default): the ground draws in the main pass alone, with the window's MSAA.
    Off,
    /// `VD_PREPASS=depth`: the depth prepass alone, MSAA off.
    Depth,
    /// `VD_PREPASS=1`: the depth and the motion prepass, MSAA off.
    Motion,
}

impl Prepass {
    fn from_switch(value: Option<&str>) -> Prepass {
        match value.map(str::trim) {
            Some("1") => Prepass::Motion,
            Some("depth") => Prepass::Depth,
            _ => Prepass::Off,
        }
    }
}

/// The switches' plug-in: the resource in both worlds, the render scale's override, the pricing
/// pass, and the adapter report (the design's T0 item 4).
pub(crate) struct RenderMeasurePlugin(pub RenderMeasure);

impl Plugin for RenderMeasurePlugin {
    fn build(&self, app: &mut App) {
        let measure = self.0;
        if measure != RenderMeasure::default() {
            tracing::info!(
                ?measure,
                "the temporal path's measurement switches are set (T0)"
            );
        }
        app.insert_resource(measure).add_plugins(
            bevy::render::extract_component::ExtractComponentPlugin::<RenderGrid>::default(),
        );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(measure)
            .add_systems(Render, render_scale_system());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        report_adapter(render_app.world().resource::<RenderAdapter>());
        if self.0.price_g2_targets {
            render_app.insert_resource(G2Targets::default());
            let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
            graph.add_node(PriceG2Label, PriceG2Node::default());
            graph.add_node_edge(bevy::render::graph::CameraDriverLabel, PriceG2Label);
        }
    }
}

/// The render scale's place in the render schedule: with the views, before the engine sizes the views'
/// targets. The reconstruction reads the size it sets AFTER it, in the same frame
/// (`reconstruction::view_systems`).
pub(crate) fn render_scale_system() -> ScheduleConfigs<ScheduleSystem> {
    apply_render_scale
        .in_set(RenderSystems::ManageViews)
        .before(bevy::render::view::prepare_view_targets)
}

/// ★ THE RENDER SCALE: every picture camera's main passes draw `s` of each side (the probe camera,
/// order −1, keeps its own size: it is a meter's picture, not the frame's); a camera with a fixed
/// render grid (the capture's picture and its probe) draws exactly that grid.
pub(crate) fn apply_render_scale(
    mut commands: Commands,
    measure: Res<RenderMeasure>,
    views: Query<(Entity, &ExtractedCamera, Option<&RenderGrid>)>,
) {
    let Some(s) = measure.render_scale else {
        return;
    };
    for (entity, camera, grid) in &views {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        // A fixed grid first: the capture's probe takes its picture's path exactly (the design §12.1).
        if let Some(grid) = grid {
            commands
                .entity(entity)
                .insert(MainPassResolutionOverride(grid.0));
            continue;
        }
        if camera.order < 0 {
            continue;
        }
        // Under the reconstruction the scale is held by the floor law (`vd_client::reconstruction`):
        // one render pixel never covers more than one pixel of the reference view.
        let scale = if measure.temporal {
            vd_client::reconstruction::lawful_scale(f64::from(s), size.y) as f32
        } else {
            s
        };
        let side = |n: u32| ((n as f32 * scale).round() as u32).max(1);
        commands
            .entity(entity)
            .insert(MainPassResolutionOverride(UVec2::new(
                side(size.x),
                side(size.y),
            )));
    }
}

/// ★ THE ADAPTER REPORT (T0 item 4): what the reconstruction needs from this GPU, as wgpu reports
/// it — the storage formats it may read AND write in one pass, the texture atomics, the half floats,
/// and the shader model.
fn report_adapter(adapter: &RenderAdapter) {
    use wgpu::{Features, TextureFormat, TextureFormatFeatureFlags};
    let features = adapter.features();
    let read_write = |format: TextureFormat| {
        adapter
            .get_texture_format_features(format)
            .flags
            .contains(TextureFormatFeatureFlags::STORAGE_READ_WRITE)
    };
    let info = adapter.get_info();
    tracing::info!(
        name = %info.name,
        backend = ?info.backend,
        shader_f16 = features.contains(Features::SHADER_F16),
        texture_atomic = features.contains(Features::TEXTURE_ATOMIC),
        texture_int64_atomic = features.contains(Features::TEXTURE_INT64_ATOMIC),
        rw_r32uint = read_write(TextureFormat::R32Uint),
        rw_r32float = read_write(TextureFormat::R32Float),
        rw_rgba16float = read_write(TextureFormat::Rgba16Float),
        rw_rgba32float = read_write(TextureFormat::Rgba32Float),
        shader_model = ?adapter.get_downlevel_capabilities().shader_model,
        "THE ADAPTER REPORT (the temporal path's T0): what the reconstruction may use"
    );
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct PriceG2Label;

/// The four textures of the pricing pass, made at the display size and made again when it changes.
#[derive(Resource, Default)]
struct G2Targets {
    size: UVec2,
    textures: Vec<(Texture, TextureView)>,
}

/// ★ M-2b: the deferred route's full-size targets, priced before T3 builds them. Each frame one
/// pass clears and stores four textures of the route's formats — the surface words (Rgba32Uint), the
/// light pass's id (R8Uint), the prepass depth copy (Depth16Unorm) and the motion (Rg16Float): 23
/// bytes a pixel — with its viewport at the drawn rectangle, as the engine's passes set theirs. The
/// GPU clock reads the difference with and without it.
#[derive(Default)]
struct PriceG2Node {
    views: Option<
        QueryState<(
            &'static ExtractedCamera,
            Option<&'static MainPassResolutionOverride>,
        )>,
    >,
}

impl Node for PriceG2Node {
    fn update(&mut self, world: &mut World) {
        let views = self.views.get_or_insert_with(|| QueryState::new(world));
        views.update_archetypes(world);
        // The display size of the first picture camera: the textures follow it.
        let size = views
            .iter_manual(world)
            .filter(|(c, _)| c.order >= 0)
            .find_map(|(c, _)| c.physical_target_size);
        let Some(size) = size else {
            return;
        };
        if world.resource::<G2Targets>().size == size {
            return;
        }
        let device = world.resource::<RenderDevice>().clone();
        let make = |format: wgpu::TextureFormat| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vd_price_g2"),
                size: wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let textures = vec![
            make(wgpu::TextureFormat::Rgba32Uint),
            make(wgpu::TextureFormat::R8Uint),
            make(wgpu::TextureFormat::Rg16Float),
            make(wgpu::TextureFormat::Depth16Unorm),
        ];
        *world.resource_mut::<G2Targets>() = G2Targets { size, textures };
    }

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let targets = world.resource::<G2Targets>();
        let [words, id, motion, depth] = targets.textures.as_slice() else {
            return Ok(());
        };
        // The drawn rectangle: the override's size where one is set, else the whole target.
        let rect = self.views.as_ref().and_then(|views| {
            views
                .iter_manual(world)
                .filter(|(c, _)| c.order >= 0)
                .find_map(|(c, o)| o.map(|o| **o).or(c.physical_target_size))
        });
        let colour = |view: &'w TextureView| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut pass =
            render_context
                .command_encoder()
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("vd_price_g2_targets"),
                    color_attachments: &[colour(&words.1), colour(&id.1), colour(&motion.1)],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth.1,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
        if let Some(rect) = rect {
            pass.set_viewport(0.0, 0.0, rect.x as f32, rect.y as f32, 0.0, 1.0);
        }
        drop(pass);
        Ok(())
    }
}
