//! Deferred-path eclipse — fullscreen post-process pass that runs after
//! Bevy's [`DeferredLightingPass`] and applies the same surgical
//! sun-only attenuation the forward extension applies per-fragment.
//!
//! When SSR (or any other feature requiring the deferred pipeline) is
//! enabled on a camera, Bevy switches that camera's opaque draws to
//! the deferred path: prepass writes a G-buffer; a single fullscreen
//! `DeferredLightingPass` reads the G-buffer and computes lit colors
//! into the view target. The forward material extension's lighting
//! branch never runs, so the per-fragment eclipse subtraction would
//! be missing without this pass.
//!
//! Architecture:
//!
//!   1. Bevy's `DeferredLightingPass` writes the lit radiance buffer
//!      (still HDR linear in space, before tonemap).
//!   2. **This pass** ([`EclipseDeferredNode`]) runs next. It reads
//!      the G-buffer (`deferred_prepass_texture`) + depth +
//!      `mesh_view_bindings::view`, reconstructs `PbrInput` via the
//!      same `pbr_input_from_deferred_gbuffer` helper Bevy uses,
//!      computes the eclipse factor at each pixel from the shared
//!      [`super::eclipse::EclipseState`] resource, re-derives the
//!      would-be directional-sun contribution via `sun_direct_contribution`
//!      from `voxeldust::eclipse_lib`, and subtracts
//!      `(1 - factor) × sun_term × view.exposure` from the radiance.
//!   3. Subsequent passes (`MainOpaquePass`, transparents, tonemap)
//!      see the eclipsed buffer.
//!
//! Forward views (no `DeferredPrepass`) skip the pass — gated by
//! `Has<DeferredPrepass>` on the [`ViewNode`] query — because their
//! material extension already handled eclipse per-fragment.
//!
//! The pipeline mirrors Bevy's `DeferredLightingLayout::specialize`
//! so every shader-def the imported PBR helpers depend on (HDR,
//! environment maps, IBL, shadow filter method, atmosphere, etc.)
//! lines up with how Bevy compiled `pbr_input_from_deferred_gbuffer`
//! and `lighting::directional_light` — keeping eclipse and Bevy's
//! lighting in lock-step across configurations.

use bevy::{
    asset::AssetServer,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        prepass::DeferredPrepass,
        tonemapping::{DebandDither, Tonemapping},
        FullscreenShader,
    },
    ecs::query::{Has, QueryItem},
    image::BevyDefault as _,
    light::{EnvironmentMapLight, IrradianceVolume, ShadowFilteringMethod},
    pbr::{
        graph::NodePbr, DistanceFog, ExtractedAtmosphere, MeshPipeline, MeshPipelineKey,
        MeshViewBindGroup, RenderViewLightProbes, ScreenSpaceAmbientOcclusion,
        ScreenSpaceReflectionsUniform, ViewEnvironmentMapUniformOffset, ViewFogUniformOffset,
        ViewLightProbesUniformOffset, ViewLightsUniformOffset,
        ViewScreenSpaceReflectionsUniformOffset,
    },
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode,
            ViewNodeRunner,
        },
        render_resource::{
            binding_types::{sampler, texture_2d, uniform_buffer},
            BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, MultisampleState,
            Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
            SpecializedRenderPipeline, SpecializedRenderPipelines, TextureSampleType, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::{ExtractedView, ViewTarget, ViewUniformOffset},
        Render, RenderApp, RenderSystems,
    },
    shader::{Shader, ShaderDefVal, ShaderRef},
};

use super::eclipse::{EclipseState, EclipseUniform};

// ───── Render-graph label ─────────────────────────────────────────────────

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct EclipseDeferredLabel;

// ───── Plugin ─────────────────────────────────────────────────────────────

pub struct EclipseDeferredPlugin;

impl Plugin for EclipseDeferredPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<EclipseDeferredLayout>>()
            .init_resource::<EclipseUniformBuffer>()
            .add_systems(bevy::render::RenderStartup, init_eclipse_deferred_layout)
            .add_systems(
                Render,
                (
                    write_eclipse_uniform_buffer.in_set(RenderSystems::Prepare),
                    prepare_eclipse_deferred_pipelines.in_set(RenderSystems::Prepare),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<EclipseDeferredNode>>(
                Core3d,
                EclipseDeferredLabel,
            )
            // Insert AFTER Bevy's deferred-lighting writes the radiance
            // buffer, BEFORE the main opaque (forward) pass that
            // already handled eclipse per-fragment in the material
            // extension. `add_render_graph_edges` walks the array in
            // adjacent pairs, so `(a, b, c)` creates `a → b` and
            // `b → c` only — no implicit `a → c` edge.
            .add_render_graph_edges(
                Core3d,
                (
                    NodePbr::DeferredLightingPass,
                    EclipseDeferredLabel,
                    Node3d::MainOpaquePass,
                ),
            );
    }
}

// ───── Render-world resources ─────────────────────────────────────────────

/// Render-world uniform buffer holding the per-frame [`EclipseUniform`].
/// Re-uploaded each frame from the extracted [`EclipseState`] resource;
/// bound as `@group(2) @binding(2)` in our shader.
#[derive(Resource, Default)]
pub struct EclipseUniformBuffer {
    pub buffer: UniformBuffer<EclipseUniform>,
}

fn write_eclipse_uniform_buffer(
    state: Option<Res<EclipseState>>,
    mut buf: ResMut<EclipseUniformBuffer>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // `EclipseState` is extracted from the main world via
    // `ExtractResourcePlugin`. Until the main-world resource initialises
    // on the first frame the render-world copy may be missing — write
    // an INACTIVE uniform so the shader short-circuits to no-occlusion.
    let value = state
        .map(|s| s.uniform)
        .unwrap_or(EclipseUniform::INACTIVE);
    buf.buffer.set(value);
    buf.buffer.write_buffer(&device, &queue);
}

/// Pipeline layout + cached resources for the deferred eclipse pass.
/// One layout per render world; specialised per-view (HDR /
/// environment-map / etc. shader-def variants).
#[derive(Resource)]
pub struct EclipseDeferredLayout {
    pub mesh_pipeline: MeshPipeline,
    pub bind_group_layout_2: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
    pub sampler: Sampler,
    pub shader: Handle<Shader>,
}

fn init_eclipse_deferred_layout(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    render_device: Res<RenderDevice>,
) {
    // Layout: source radiance texture (the previous color attachment;
    // we ping-pong via `ViewTarget::post_process_write`), a sampler,
    // and the per-frame eclipse uniform.
    let bind_group_layout_2 = BindGroupLayoutDescriptor::new(
        "eclipse_deferred_group_2",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<EclipseUniform>(false),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    commands.insert_resource(EclipseDeferredLayout {
        mesh_pipeline: mesh_pipeline.clone(),
        bind_group_layout_2,
        fullscreen_shader: fullscreen_shader.clone(),
        sampler,
        shader: asset_server.load("shaders/eclipse_deferred.wgsl"),
    });
}

// ───── Pipeline specialisation ────────────────────────────────────────────

impl SpecializedRenderPipeline for EclipseDeferredLayout {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Mirror `DeferredLightingLayout::specialize` shader-defs so
        // the imported PBR helpers (`pbr_input_from_deferred_gbuffer`,
        // `lighting::directional_light`, `shadows::fetch_directional_shadow`,
        // etc.) compile with the exact same configuration Bevy used
        // when it produced the radiance buffer we're modulating.
        let mut shader_defs: Vec<ShaderDefVal> = Vec::new();

        // Always inside the deferred path here.
        shader_defs.push("DEFERRED_LIGHTING_PIPELINE".into());
        shader_defs.push("DEFERRED_PREPASS".into());

        if key.contains(MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION) {
            shader_defs.push("SCREEN_SPACE_AMBIENT_OCCLUSION".into());
        }
        if key.contains(MeshPipelineKey::ENVIRONMENT_MAP) {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }
        if key.contains(MeshPipelineKey::IRRADIANCE_VOLUME) {
            shader_defs.push("IRRADIANCE_VOLUME".into());
        }
        if key.contains(MeshPipelineKey::NORMAL_PREPASS) {
            shader_defs.push("NORMAL_PREPASS".into());
        }
        if key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            shader_defs.push("DEPTH_PREPASS".into());
        }
        if key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            shader_defs.push("MOTION_VECTOR_PREPASS".into());
        }
        if key.contains(MeshPipelineKey::SCREEN_SPACE_REFLECTIONS) {
            shader_defs.push("SCREEN_SPACE_REFLECTIONS".into());
        }
        if key.contains(MeshPipelineKey::DISTANCE_FOG) {
            shader_defs.push("DISTANCE_FOG".into());
        }
        if key.contains(MeshPipelineKey::ATMOSPHERE) {
            shader_defs.push("ATMOSPHERE".into());
        }

        let shadow_filter_method =
            key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
        if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
            shader_defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN {
            shader_defs.push("SHADOW_FILTER_METHOD_GAUSSIAN".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL {
            shader_defs.push("SHADOW_FILTER_METHOD_TEMPORAL".into());
        }
        if self.mesh_pipeline.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
            shader_defs.push("MULTIPLE_LIGHTMAPS_IN_ARRAY".into());
        }

        let view_layout = self.mesh_pipeline.get_view_layout(key.into());

        RenderPipelineDescriptor {
            label: Some("eclipse_deferred_pipeline".into()),
            layout: vec![
                view_layout.main_layout.clone(),
                view_layout.binding_array_layout.clone(),
                self.bind_group_layout_2.clone(),
            ],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: if key.contains(MeshPipelineKey::HDR) {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        bevy::render::render_resource::TextureFormat::bevy_default()
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            ..default()
        }
    }
}

// ───── Per-view pipeline cache ────────────────────────────────────────────

/// Per-view cached pipeline ID for the deferred eclipse pass. Mirrors
/// Bevy's `DeferredLightingPipeline` pattern.
#[derive(Component)]
pub struct EclipseDeferredPipeline {
    pub pipeline_id: CachedRenderPipelineId,
}

#[allow(clippy::too_many_arguments)]
fn prepare_eclipse_deferred_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<EclipseDeferredLayout>>,
    layout: Res<EclipseDeferredLayout>,
    views: Query<(
        Entity,
        &ExtractedView,
        Option<&Tonemapping>,
        Option<&DebandDither>,
        Option<&ShadowFilteringMethod>,
        (
            Has<ScreenSpaceAmbientOcclusion>,
            Has<ScreenSpaceReflectionsUniform>,
            Has<DistanceFog>,
        ),
        Has<bevy::core_pipeline::prepass::NormalPrepass>,
        Has<bevy::core_pipeline::prepass::DepthPrepass>,
        Has<bevy::core_pipeline::prepass::MotionVectorPrepass>,
        Has<DeferredPrepass>,
        Has<RenderViewLightProbes<EnvironmentMapLight>>,
        Has<RenderViewLightProbes<IrradianceVolume>>,
        Has<ExtractedAtmosphere>,
    )>,
) {
    for (
        entity,
        view,
        tonemapping,
        dither,
        shadow_filter_method,
        (ssao, ssr, distance_fog),
        normal_prepass,
        depth_prepass,
        motion_vector_prepass,
        deferred_prepass,
        has_environment_maps,
        has_irradiance_volumes,
        has_atmosphere,
    ) in &views
    {
        // Forward views skip the pass — their per-fragment material
        // extension already applied eclipse. Drop any stale pipeline
        // (covers the case where a view turns SSR off mid-session).
        if !deferred_prepass {
            commands.entity(entity).remove::<EclipseDeferredPipeline>();
            continue;
        }

        let mut view_key = MeshPipelineKey::from_hdr(view.hdr);
        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }
        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }
        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }
        if has_atmosphere {
            view_key |= MeshPipelineKey::ATMOSPHERE;
        }
        if view.invert_culling {
            view_key |= MeshPipelineKey::INVERT_CULLING;
        }
        view_key |= MeshPipelineKey::DEFERRED_PREPASS;
        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;
                view_key |= match tonemapping {
                    Tonemapping::None => MeshPipelineKey::TONEMAP_METHOD_NONE,
                    Tonemapping::Reinhard => MeshPipelineKey::TONEMAP_METHOD_REINHARD,
                    Tonemapping::ReinhardLuminance => {
                        MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE
                    }
                    Tonemapping::AcesFitted => MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED,
                    Tonemapping::AgX => MeshPipelineKey::TONEMAP_METHOD_AGX,
                    Tonemapping::SomewhatBoringDisplayTransform => {
                        MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
                    }
                    Tonemapping::TonyMcMapface => MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
                    Tonemapping::BlenderFilmic => MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
                };
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= MeshPipelineKey::DEBAND_DITHER;
            }
        }
        if ssao {
            view_key |= MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }
        if ssr {
            view_key |= MeshPipelineKey::SCREEN_SPACE_REFLECTIONS;
        }
        if distance_fog {
            view_key |= MeshPipelineKey::DISTANCE_FOG;
        }
        if has_environment_maps {
            view_key |= MeshPipelineKey::ENVIRONMENT_MAP;
        }
        if has_irradiance_volumes {
            view_key |= MeshPipelineKey::IRRADIANCE_VOLUME;
        }
        match shadow_filter_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Gaussian => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
            }
            ShadowFilteringMethod::Temporal => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
            }
        }

        let pipeline_id = pipelines.specialize(&pipeline_cache, &layout, view_key);
        commands
            .entity(entity)
            .insert(EclipseDeferredPipeline { pipeline_id });
    }
}

// ───── ViewNode ───────────────────────────────────────────────────────────

#[derive(Default)]
pub struct EclipseDeferredNode;

impl ViewNode for EclipseDeferredNode {
    type ViewQuery = (
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewFogUniformOffset,
        &'static ViewLightProbesUniformOffset,
        &'static ViewScreenSpaceReflectionsUniformOffset,
        &'static ViewEnvironmentMapUniformOffset,
        &'static MeshViewBindGroup,
        &'static ViewTarget,
        &'static EclipseDeferredPipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            view_uniform_offset,
            view_lights_offset,
            view_fog_offset,
            view_light_probes_offset,
            view_ssr_offset,
            view_environment_map_offset,
            mesh_view_bind_group,
            target,
            pipeline_marker,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<EclipseDeferredLayout>();
        let uniform_buffer = world.resource::<EclipseUniformBuffer>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_marker.pipeline_id)
        else {
            return Ok(());
        };
        let Some(eclipse_binding) = uniform_buffer.buffer.binding() else {
            return Ok(());
        };

        // Ping-pong source/destination for read-then-write on the
        // same view target. `post_process_write()` flips the active
        // texture so the next pass sees `destination` as the new
        // source.
        let post_process = target.post_process_write();

        let bind_group_2 = render_context.render_device().create_bind_group(
            "eclipse_deferred_group_2",
            &pipeline_cache.get_bind_group_layout(&layout.bind_group_layout_2),
            &BindGroupEntries::sequential((
                post_process.source,
                &layout.sampler,
                eclipse_binding,
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("eclipse_deferred_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(
            0,
            &mesh_view_bind_group.main,
            &[
                view_uniform_offset.offset,
                view_lights_offset.offset,
                view_fog_offset.offset,
                **view_light_probes_offset,
                **view_ssr_offset,
                **view_environment_map_offset,
            ],
        );
        render_pass.set_bind_group(1, &mesh_view_bind_group.binding_array, &[]);
        render_pass.set_bind_group(2, &bind_group_2, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}
