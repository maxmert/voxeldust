//! ★ THE RECONSTRUCTION (the temporal path's slice T2; `docs/investigation/2026-09-28/
//! temporal_path_design.md` §4.2–§4.4, §8.2): our own temporal up-scaler on AMD FSR 2.2's structure
//! (MIT, credited), behind the switch `VD_TEMPORAL=1` (`measure::RenderMeasure::temporal`). The
//! picture camera draws a JITTERED render picture — at the render scale's size, `VD_RENDER_SCALE`,
//! held by the floor law — and this pass rebuilds each DISPLAY pixel from this frame's samples and
//! the history of the frames before (`reconstruction.wgsl`). The rules are Tier-A
//! (`vd_client::reconstruction`); this file is the render world's half: the jitter on the pass's
//! own clock, the reset decision and its count (G-RESET), the history's textures, and the node.
//!
//! The node stands where the engine places its own temporal pass: after the main passes, before
//! bloom and the tone map.
//!
//! **Example.** The owner stands on foot at full screen with `VD_TEMPORAL=1 VD_RENDER_SCALE=0.5`: the
//! ground and the air draw 1 728 × 1 084 jittered pixels over 32 phases, and this pass writes the
//! 3 456 × 2 168 display picture from them and from its history.
//!
//! ★ THE DEPTH TEST (the design §4.2 steps 1.3 and 2): before the pass, a compute pass rebuilds the last
//! frame's nearest depth from this frame's surfaces (one word per render pixel, cleared each frame), and
//! the pass lowers the count of a pixel that a nearer surface hid in the last frame.
//!
//! ★ THE PROBE TAKES THE PICTURE'S JITTER (the design §12.1): a camera marked [`JitterFollower`] (the
//! capture's probe) draws with the same projection offset as the picture, so the probe and the render
//! picture sample the same points.
//!
//! The points pass (`points`) stands after this pass: the stars and the sun at the display size.
//!
//! ★ THE SKY'S MOTION (the design §6.3): a sky pixel reprojects by rotation alone, through the two
//! views turned by the air's rotation step ([`SkyTurn`]).
//!
//! ★ THE REACTIVE MASK (the design §4.2 step 3): a copy of the render picture after the sky and before the
//! transparent pass, which the pass compares with the finished render picture.
//!
//! NOT YET (T2's next steps): a tool's jump as a reset.

use std::collections::{BTreeMap, BTreeSet};

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::camera::MainPassResolutionOverride;
use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::prepass::{PreviousViewData, ViewPrepassTextures};
use bevy::ecs::error::BevyError;
use bevy::ecs::query::QueryItem;
use bevy::image::{BevyDefault as _, ToExtents};
use bevy::prelude::*;
use bevy::render::camera::{ExtractedCamera, TemporalJitter};
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraph, RenderGraphContext, RenderGraphError, RenderGraphExt, RenderLabel,
    ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_depth_2d,
    uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer, BufferDescriptor,
    BufferUsages, CachedComputePipelineId, CachedRenderPipelineId, Canonical, ColorTargetState,
    ColorWrites, ComputePassDescriptor, ComputePipelineDescriptor, FilterMode, FragmentState,
    Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    ShaderType, Specializer, SpecializerKey, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, UniformBuffer, Variants,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use vd_client::reconstruction::{
    DEPTH_FLOOR, JitterClock, ReconstructionSettings, ResetCause, corner_factor, depth_clip_power,
    depth_separation, jitter, jitter_phases, kernel_bias, reset_cause,
};

/// ★ THE PICTURE CAMERA'S MARK: this camera draws through the reconstruction (the main world, moved
/// into the render world every frame).
#[derive(Component, Clone, Copy, ExtractComponent)]
pub(crate) struct ReconstructionCamera;

/// ★ THE AIR'S ROTATION STEP (the design §6.3): the rotation part of the air body's realm step — this
/// frame's render space to last frame's, for the body whose air the picture draws (`vd_client::motion::
/// RealmStep`) — set by the terrain's placement every frame; the identity where the air's body had no
/// placement last frame or no air is drawn. A sky pixel (no surface) reprojects by the two views
/// turned by it. On the home planet, the origin, it is the identity; at a boarding it turns the planet's
/// axes into the hull's.
///
/// **Example.** A hull berthed 90° off the planet's axes departs: in the swap frame this step turns the
/// history of every sky pixel by those 90°, and the air stays still on the screen.
#[derive(Resource, Clone, Copy, Default, bevy::render::extract_resource::ExtractResource)]
pub(crate) struct SkyTurn(pub(crate) Quat);

/// ★ THE PICTURE'S RECONSTRUCTION ROW for the capture's stamp (the gates, the design §12.3): the frame's
/// jitter (to reproject a jittered picture exactly), the two sizes, and the history's restarts by cause
/// (G-RESET) — written each frame for the first reconstructed view (the capture holds one picture camera),
/// read by the capture's copier onto the frame's own stamp.
#[derive(Resource, Default, Clone, Copy)]
pub(crate) struct ReconstructionReadout(pub(crate) Option<vd_devproto::DevReconstruction>);

/// ★ A CAMERA THAT TAKES THE PICTURE'S JITTER (the capture's probe, the design §12.1): its projection
/// offset is the reconstructed picture's, every frame. The capture holds one picture camera; with more,
/// the first one leads.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub(crate) struct JitterFollower;

/// The plug-in: the marks' extraction, the pipeline, the jitter and the textures, and the node.
pub(crate) struct ReconstructionPlugin;

impl Plugin for ReconstructionPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "reconstruction.wgsl");
        app.init_resource::<SkyTurn>().add_plugins((
            ExtractComponentPlugin::<ReconstructionCamera>::default(),
            ExtractComponentPlugin::<JitterFollower>::default(),
            bevy::render::extract_resource::ExtractResourcePlugin::<SkyTurn>::default(),
        ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(ReconstructionViews::default())
            .init_resource::<ReconstructionReadout>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                (
                    jitter_views
                        .in_set(RenderSystems::ManageViews)
                        .before(bevy::render::view::prepare_view_targets),
                    prepare_pipelines.in_set(RenderSystems::Prepare),
                    prepare_textures.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<ReconstructionNode>>(
                Core3d,
                ReconstructionLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPassPostProcessing,
                    ReconstructionLabel,
                    Node3d::Bloom,
                ),
            )
            // ★ THE REACTIVE MASK'S COPY: after the transmissive pass and the sky (the sky's edge in
            // `finish`: the air adds its nodes there), before the transparent pass.
            .add_render_graph_node::<ViewNodeRunner<OpaqueCopyNode>>(Core3d, OpaqueCopyLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainTransmissivePass,
                    OpaqueCopyLabel,
                    Node3d::MainTransparentPass,
                ),
            );
    }

    /// The copy stands after the sky pass, which the air's plug-in adds in its own `finish` (the sky and
    /// the transmissive pass have no order between them in the engine's graph). An adapter the air
    /// refuses has no sky node, and then no sky to wait for.
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        let Some(core) = graph.get_sub_graph_mut(Core3d) else {
            return;
        };
        match core.try_add_node_edge(bevy::pbr::AtmosphereNode::RenderSky, OpaqueCopyLabel) {
            Ok(()) | Err(RenderGraphError::EdgeAlreadyExists(_)) => {}
            Err(e) => tracing::info!(%e, "the reactive mask's copy: no sky pass to follow"),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct OpaqueCopyLabel;

/// ★ THE OPAQUE COPY (the reactive mask, the design §4.2 step 3): the render rectangle of the picture as it
/// stands after the sky, before the transparent pass — FSR 2's "opaque only" colour — copied at the render
/// size; the pass compares it with the finished render picture.
#[derive(Default)]
struct OpaqueCopyNode;

impl ViewNode for OpaqueCopyNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ReconstructionFrame,
        &'static ReconstructionTextures,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (target, frame, textures): QueryItem<Self::ViewQuery>,
        _world: &World,
    ) -> Result<(), NodeRunError> {
        render_context.command_encoder().copy_texture_to_texture(
            target.main_texture().as_image_copy(),
            textures.opaque.texture.as_image_copy(),
            frame.render_size.to_extents(),
        );
        Ok(())
    }
}

/// The pass's uniform: `Params` in `reconstruction.wgsl`, one layout.
#[derive(ShaderType, Clone, Copy, Default)]
struct ReconstructionUniform {
    sky_reproject: Mat4,
    render_size: Vec2,
    display_size: Vec2,
    jitter: Vec2,
    reset: u32,
    motion_sentinel: f32,
    still_floor: f32,
    moving_floor: f32,
    still_motion_px: f32,
    moving_motion_px: f32,
    box_deviations: f32,
    kernel_bias: f32,
    near: f32,
    depth_floor: f32,
    depth_separation: f32,
    depth_clip_power: f32,
    rebuild_still_display_px: f32,
    bilinear_tap_floor: f32,
    reactive_scale: f32,
    reactive_threshold: f32,
    reactive_value: f32,
    upsample_weight_min: f32,
}

/// ★ ONE VIEW'S STATE across frames (the render world keeps a camera's entity): the jitter's clock,
/// the history's size, the frames and the resets by cause (G-RESET), the uniform's buffer, and the
/// rebuilt previous depth (one word per render pixel, made again when the render size changes).
struct ViewState {
    clock: JitterClock,
    has_history: bool,
    history_size: UVec2,
    frames: u64,
    resets: [u64; 4],
    uniform: UniformBuffer<ReconstructionUniform>,
    rebuilt: Option<(Buffer, UVec2)>,
}

impl ViewState {
    fn new() -> Self {
        ViewState {
            clock: JitterClock::default(),
            has_history: false,
            history_size: UVec2::ZERO,
            frames: 0,
            resets: [0; 4],
            uniform: UniformBuffer::default(),
            rebuilt: None,
        }
    }
}

/// Every reconstructed view's state, by its render-world entity.
#[derive(Resource, Default)]
struct ReconstructionViews(BTreeMap<Entity, ViewState>);

/// This frame's decision for a view: the textures and the node read it.
#[derive(Component, Clone, Copy)]
pub(crate) struct ReconstructionFrame {
    display_size: UVec2,
    /// The render picture's size (the top-left rectangle the main passes draw) and its format's range.
    render_size: UVec2,
    hdr: bool,
    /// Which of the two history textures this frame writes (they alternate).
    write_first: bool,
}

/// The reset cause's slot in a view's count.
fn cause_slot(cause: ResetCause) -> usize {
    match cause {
        ResetCause::FirstFrame => 0,
        ResetCause::DeviceLost => 1,
        ResetCause::DisplaySize => 2,
        ResetCause::ToolJump => 3,
    }
}

/// ★ THE JITTER ON THE PASS'S OWN CLOCK, AND THE RESET (the render world's view management, before
/// the view's uniforms take the jitter): the reset decision, the phase, the projection's offset,
/// the sky's reprojection, and the pass's uniform. The engine's own temporal pass takes the frame
/// count; this clock starts at nought at each reset, so two captures of one stand see one sequence.
#[allow(clippy::type_complexity, clippy::too_many_arguments)] // a Bevy system: injected queries and resources
fn jitter_views(
    mut commands: Commands,
    mut state: ResMut<ReconstructionViews>,
    mut views: Query<
        (
            Entity,
            &ExtractedCamera,
            &ExtractedView,
            &mut TemporalJitter,
            Option<&MainPassResolutionOverride>,
            Option<&PreviousViewData>,
        ),
        With<ReconstructionCamera>,
    >,
    mut followers: Query<
        &mut TemporalJitter,
        (With<JitterFollower>, Without<ReconstructionCamera>),
    >,
    sky_turn: Option<Res<SkyTurn>>,
    mut readout: ResMut<ReconstructionReadout>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let settings = ReconstructionSettings::PROPOSED;
    let mut seen = BTreeSet::new();
    let mut lead = None;
    readout.0 = None;
    for (entity, camera, view, mut offset, resolution, previous) in &mut views {
        let Some(display) = camera.physical_viewport_size else {
            continue;
        };
        let render = resolution.map_or(display, |r| r.0);
        let st = state.0.entry(entity).or_insert_with(ViewState::new);
        let cause = reset_cause(
            st.has_history,
            false,
            display.to_array(),
            st.history_size.to_array(),
            false,
        );
        if let Some(cause) = cause {
            st.resets[cause_slot(cause)] += 1;
            tracing::info!(?cause, resets = ?st.resets, "the reconstruction's history restarts");
        }
        let scale = f64::from(render.y) / f64::from(display.y.max(1));
        let phase = st.clock.advance(cause.is_some());
        let [jx, jy] = jitter(phase, jitter_phases(scale));
        offset.offset = Vec2::new(jx as f32, jy as f32);
        lead.get_or_insert(offset.offset);
        readout.0.get_or_insert(vd_devproto::DevReconstruction {
            jitter_px: offset.offset.to_array(),
            render_px: render.to_array(),
            display_px: display.to_array(),
            resets: st.resets,
        });
        if st.rebuilt.as_ref().is_none_or(|(_, size)| *size != render) {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("vd_reconstruction_rebuilt_depth"),
                size: 4 * u64::from(render.x) * u64::from(render.y),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            st.rebuilt = Some((buffer, render));
        }
        // The depth test's numbers from the camera's own (unjittered) projection: its near plane and
        // its field's corner factor.
        let projection = view.clip_from_view;
        let corner = corner_factor(
            f64::from(projection.x_axis.x),
            f64::from(projection.y_axis.y),
        );
        // The sky's motion: last frame's unjittered clip from this frame's, through a point at
        // infinity, turned by the air's rotation step (both are the camera's own projection; the
        // jitter enters only the view uniform).
        let world_from_clip = view.world_from_view.to_matrix() * view.clip_from_view.inverse();
        let turn = Mat4::from_quat(sky_turn.as_ref().map_or(Quat::IDENTITY, |t| t.0));
        let sky_reproject = previous.map_or(Mat4::IDENTITY, |p| {
            p.clip_from_world * turn * world_from_clip
        });
        st.uniform.set(ReconstructionUniform {
            sky_reproject,
            render_size: render.as_vec2(),
            display_size: display.as_vec2(),
            jitter: offset.offset,
            reset: u32::from(cause.is_some()),
            motion_sentinel: vd_client_harness::motion_census::MOTION_SENTINEL,
            still_floor: settings.still_floor as f32,
            moving_floor: settings.moving_floor as f32,
            still_motion_px: settings.still_motion_px as f32,
            moving_motion_px: settings.moving_motion_px as f32,
            box_deviations: settings.box_deviations as f32,
            kernel_bias: kernel_bias(scale) as f32,
            near: projection.w_axis.z,
            depth_floor: DEPTH_FLOOR as f32,
            depth_separation: depth_separation(render.to_array(), corner, &settings) as f32,
            depth_clip_power: depth_clip_power(render.to_array()) as f32,
            rebuild_still_display_px: settings.rebuild_still_display_px as f32,
            bilinear_tap_floor: settings.bilinear_tap_floor as f32,
            reactive_scale: settings.reactive_scale as f32,
            reactive_threshold: settings.reactive_threshold as f32,
            reactive_value: settings.reactive_value as f32,
            upsample_weight_min: settings.upsample_weight_min as f32,
        });
        st.uniform.write_buffer(&device, &queue);
        st.has_history = true;
        st.history_size = display;
        st.frames += 1;
        commands.entity(entity).insert(ReconstructionFrame {
            display_size: display,
            render_size: render,
            hdr: view.hdr,
            write_first: st.frames.is_multiple_of(2),
        });
        seen.insert(entity);
    }
    state.0.retain(|entity, _| seen.contains(entity));
    if let Some(lead) = lead {
        for mut follower in &mut followers {
            follower.offset = lead;
        }
    }
}

/// The history's two pairs of textures (the colour with the sky share, and the count), alternating.
#[derive(Component)]
pub(crate) struct ReconstructionTextures {
    /// The render picture before the transparent pass (the reactive mask's copy), at the render size.
    opaque: CachedTexture,
    read: CachedTexture,
    /// This frame's new history: the points pass reads its sky share (the fourth channel).
    pub(crate) write: CachedTexture,
    read_count: CachedTexture,
    write_count: CachedTexture,
}

/// The history's textures at the display size (a render-scale change keeps them: the design §4.4).
fn prepare_textures(
    mut commands: Commands,
    mut cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ReconstructionFrame)>,
) {
    for (entity, frame) in &views {
        let size = frame.display_size.to_extents();
        let mut texture = |label: &'static str, format: TextureFormat| {
            cache.get(
                &device,
                TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                },
            )
        };
        let one = texture("vd_reconstruction_history_1", TextureFormat::Rgba16Float);
        let two = texture("vd_reconstruction_history_2", TextureFormat::Rgba16Float);
        let one_count = texture("vd_reconstruction_count_1", TextureFormat::R16Float);
        let two_count = texture("vd_reconstruction_count_2", TextureFormat::R16Float);
        let opaque = cache.get(
            &device,
            TextureDescriptor {
                label: Some("vd_reconstruction_opaque"),
                size: frame.render_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: if frame.hdr {
                    ViewTarget::TEXTURE_FORMAT_HDR
                } else {
                    TextureFormat::bevy_default()
                },
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
        );
        let textures = if frame.write_first {
            ReconstructionTextures {
                opaque,
                read: two,
                write: one,
                read_count: two_count,
                write_count: one_count,
            }
        } else {
            ReconstructionTextures {
                opaque,
                read: one,
                write: two,
                read_count: one_count,
                write_count: two_count,
            }
        };
        commands.entity(entity).insert(textures);
    }
}

/// The pass's pipelines: the bind group's layout, the sampler, the variants by the view's format, and
/// the rebuild's compute pipeline with its own layout.
#[derive(Resource)]
struct ReconstructionPipeline {
    layout: BindGroupLayoutDescriptor,
    linear_sampler: Sampler,
    variants: Variants<RenderPipeline, ReconstructionSpecializer>,
    rebuild_layout: BindGroupLayoutDescriptor,
    rebuild: CachedComputePipelineId,
}

/// The rebuild's workgroup side (`@workgroup_size(8, 8, 1)` in `reconstruction.wgsl`).
const REBUILD_GROUP: u32 = 8;

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    fullscreen: Res<FullscreenShader>,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let linear_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("vd_reconstruction_linear"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..SamplerDescriptor::default()
    });
    let layout = BindGroupLayoutDescriptor::new(
        "vd_reconstruction_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                // The render picture (the view target's source), the history, the count, the motion.
                (0, texture_2d(TextureSampleType::Float { filterable: true })),
                (1, texture_2d(TextureSampleType::Float { filterable: true })),
                (2, texture_2d(TextureSampleType::Float { filterable: true })),
                (3, texture_2d(TextureSampleType::Float { filterable: true })),
                // The prepass's depth.
                (4, texture_depth_2d()),
                (5, sampler(SamplerBindingType::Filtering)),
                (6, uniform_buffer::<ReconstructionUniform>(false)),
                // The rebuilt previous depth, read.
                (8, storage_buffer_read_only_sized(false, None)),
                // The render picture before the transparent pass (the reactive mask).
                (9, texture_2d(TextureSampleType::Float { filterable: true })),
            ),
        ),
    );
    let rebuild_layout = BindGroupLayoutDescriptor::new(
        "vd_reconstruction_rebuild_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (3, texture_2d(TextureSampleType::Float { filterable: true })),
                (4, texture_depth_2d()),
                (6, uniform_buffer::<ReconstructionUniform>(false)),
                (7, storage_buffer_sized(false, None)),
            ),
        ),
    );
    let shader = load_embedded_asset!(assets.as_ref(), "reconstruction.wgsl");
    let rebuild = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("vd_reconstruction_rebuild_pipeline".into()),
        layout: vec![rebuild_layout.clone()],
        shader: shader.clone(),
        entry_point: Some("rebuild".into()),
        ..default()
    });
    let variants = Variants::new(
        ReconstructionSpecializer,
        RenderPipelineDescriptor {
            label: Some("vd_reconstruction_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader,
                entry_point: Some("fragment".into()),
                ..default()
            }),
            ..default()
        },
    );
    commands.insert_resource(ReconstructionPipeline {
        layout,
        linear_sampler,
        variants,
        rebuild_layout,
        rebuild,
    });
}

struct ReconstructionSpecializer;

#[derive(PartialEq, Eq, Hash, Clone, SpecializerKey)]
struct ReconstructionKey {
    hdr: bool,
}

impl Specializer<RenderPipeline> for ReconstructionSpecializer {
    type Key = ReconstructionKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        let fragment = descriptor.fragment_mut()?;
        let format = if key.hdr {
            fragment.shader_defs.push("TONEMAP".into());
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };
        let target = |format| ColorTargetState {
            format,
            blend: None,
            write_mask: ColorWrites::ALL,
        };
        fragment.set_target(0, target(format));
        fragment.set_target(1, target(TextureFormat::Rgba16Float));
        fragment.set_target(2, target(TextureFormat::R16Float));
        Ok(key)
    }
}

#[derive(Component)]
struct ReconstructionPipelineId(CachedRenderPipelineId);

fn prepare_pipelines(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    mut pipeline: ResMut<ReconstructionPipeline>,
    views: Query<(Entity, &ExtractedView), With<ReconstructionFrame>>,
) -> Result<(), BevyError> {
    for (entity, view) in &views {
        let id = pipeline
            .variants
            .specialize(&cache, ReconstructionKey { hdr: view.hdr })?;
        commands.entity(entity).insert(ReconstructionPipelineId(id));
    }
    Ok(())
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct ReconstructionLabel;

/// ★ THE PASS: the display picture, the new history and the new count, from the render picture,
/// the prepass's motion and depth, and the history.
#[derive(Default)]
struct ReconstructionNode;

impl ViewNode for ReconstructionNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewPrepassTextures,
        &'static ReconstructionTextures,
        &'static ReconstructionPipelineId,
    );

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (target, prepass, textures, pipeline_id): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let (Some(pipelines), Some(cache)) = (
            world.get_resource::<ReconstructionPipeline>(),
            world.get_resource::<PipelineCache>(),
        ) else {
            return Ok(());
        };
        let (Some(pipeline), Some(rebuild), Some(motion), Some(depth)) = (
            cache.get_render_pipeline(pipeline_id.0),
            cache.get_compute_pipeline(pipelines.rebuild),
            &prepass.motion_vectors,
            &prepass.depth,
        ) else {
            return Ok(());
        };
        let Some(view) = world
            .get_resource::<ReconstructionViews>()
            .and_then(|views| views.0.get(&graph.view_entity()))
        else {
            return Ok(());
        };
        let (Some(uniform), Some((rebuilt, render))) = (view.uniform.binding(), &view.rebuilt)
        else {
            return Ok(());
        };
        // ★ THE REBUILT PREVIOUS DEPTH: cleared (nought: no surface), then written by every render pixel.
        let device = render_context.render_device().clone();
        let rebuild_group = device.create_bind_group(
            "vd_reconstruction_rebuild_bind_group",
            &cache.get_bind_group_layout(&pipelines.rebuild_layout),
            &BindGroupEntries::with_indices((
                (3, &motion.texture.default_view),
                (4, &depth.texture.default_view),
                (6, uniform.clone()),
                (7, rebuilt.as_entire_binding()),
            )),
        );
        let encoder = render_context.command_encoder();
        encoder.clear_buffer(rebuilt, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("vd_reconstruction_rebuild"),
                timestamp_writes: None,
            });
            pass.set_pipeline(rebuild);
            pass.set_bind_group(0, &rebuild_group, &[]);
            pass.dispatch_workgroups(
                render.x.div_ceil(REBUILD_GROUP),
                render.y.div_ceil(REBUILD_GROUP),
                1,
            );
        }
        let post = target.post_process_write();
        let bind_group = device.create_bind_group(
            "vd_reconstruction_bind_group",
            &cache.get_bind_group_layout(&pipelines.layout),
            &BindGroupEntries::with_indices((
                (0, post.source),
                (1, &textures.read.default_view),
                (2, &textures.read_count.default_view),
                (3, &motion.texture.default_view),
                (4, &depth.texture.default_view),
                (5, &pipelines.linear_sampler),
                (6, uniform),
                (8, rebuilt.as_entire_binding()),
                (9, &textures.opaque.default_view),
            )),
        );
        let attachment = |view| {
            Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })
        };
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("vd_reconstruction"),
            color_attachments: &[
                attachment(post.destination),
                attachment(&textures.write.default_view),
                attachment(&textures.write_count.default_view),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ THE JITTER'S SIGN, ON THE ENGINE'S OWN PROJECTION: under a projection offset `o` the engine's
    /// reverse-Z perspective draws a scene point at `p − o` (x to the right, y down), so render pixel
    /// `i` samples `i + 0.5 + o` — the convention `vd_client::reconstruction::jitter` states and
    /// `reconstruction.wgsl` reads. An engine upgrade that flips it turns this test red.
    #[test]
    fn the_engine_draws_a_point_at_its_place_less_the_projection_offset() {
        let size = Vec2::new(1_284.0, 720.0);
        let unjittered = Mat4::perspective_infinite_reverse_rh(0.8, size.x / size.y, 0.1);
        let pixel = |clip_from_view: Mat4, view: Vec3| {
            let clip = clip_from_view * view.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            Vec2::new((ndc.x + 1.0) * 0.5 * size.x, (1.0 - ndc.y) * 0.5 * size.y)
        };
        for offset in [Vec2::new(0.25, -0.375), Vec2::new(-0.5, 0.125)] {
            let mut jittered = unjittered;
            TemporalJitter { offset }.jitter_projection(&mut jittered, size);
            for view in [Vec3::new(3.0, -1.0, -20.0), Vec3::new(-40.0, 25.0, -900.0)] {
                let moved = pixel(jittered, view) - pixel(unjittered, view);
                assert!((moved + offset).length() < 1e-3, "{offset} {view} {moved}");
            }
        }
    }
}
