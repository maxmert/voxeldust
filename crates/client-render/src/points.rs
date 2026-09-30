//! ★ THE POINTS PASS (the temporal path's slice T2; `docs/investigation/2026-09-28/temporal_path_design.md`
//! §7.7; ruling B24): the sun's disc and the star cloud drawn AFTER the reconstruction, at the display size,
//! added to the display picture before bloom (`points.wgsl`).
//!
//! Under the temporal path the star cloud stands on its own render layer ([`POINTS_LAYER`]), so no camera
//! draws it in the main passes, and the sky pass adds no disc (the vendored fifth change,
//! `AtmosphereSettings::sun_disk_on_sky`); this pass draws both, each dimmed by the air's transmittance table
//! along its own direction and by the sky share that the reconstruction accumulated. A star's sprite is the
//! one law's (`star_law.wgsl`), in display pixels, never jittered.
//!
//! **Example.** The owner stands at the dusk stand with `VD_TEMPORAL=1 VD_RENDER_SCALE=0.5`: the render
//! picture is 1 728 × 1 084 and jittered; the stars and the sun are drawn at 3 456 × 2 168, each exactly
//! today's sprite, and a star that sets behind the belt's ridge dims over the last display pixel.

use std::collections::{BTreeMap, BTreeSet};

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::BevyDefault as _;
use bevy::mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout};
use bevy::pbr::resources::{AtmosphereSampler, AtmosphereTextures, GpuAtmosphere};
use bevy::pbr::{GpuAtmosphereSettings, GpuLights, LightMeta, ViewLightsUniformOffset};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{
    ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
};
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, uniform_buffer};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendComponent,
    BlendFactor, BlendOperation, BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites,
    FragmentState, PipelineCache, PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor,
    SamplerBindingType, ShaderStages, ShaderType, SpecializedMeshPipeline,
    SpecializedMeshPipelineError, SpecializedMeshPipelines, SpecializedRenderPipeline,
    SpecializedRenderPipelines, TextureFormat, TextureSampleType, UniformBuffer, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::StarSkyParams;
use crate::reconstruction::{ReconstructionFrame, ReconstructionLabel, ReconstructionTextures};

/// ★ THE STAR CLOUD'S LAYER under the temporal path: no camera sees it (the picture camera draws layer
/// nought, the probe [`crate::PROBE_LAYER`], the sun's shadow cascades [`crate::SHADOW_LAYER`]), so the
/// main passes never draw the cloud jittered; the points pass draws its mesh itself.
pub(crate) const POINTS_LAYER: usize = crate::SHADOW_LAYER + 1;

/// ★ THE MARK ON THE STAR CLOUD under the temporal path: its law's numbers (the material's own), which the
/// points pass binds for its sprites.
#[derive(Component, Clone)]
pub(crate) struct PointsCloud(pub(crate) StarSkyParams);

/// The cloud in the render world: its mesh, its placement and its law.
#[derive(Component, Clone)]
pub(crate) struct ExtractedPointsCloud {
    mesh: AssetId<Mesh>,
    world_from_local: Mat4,
    params: StarSkyParams,
}

impl ExtractComponent for PointsCloud {
    type QueryData = (
        &'static PointsCloud,
        &'static Mesh3d,
        &'static GlobalTransform,
    );
    type QueryFilter = ();
    type Out = ExtractedPointsCloud;

    fn extract_component(
        (cloud, mesh, transform): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(ExtractedPointsCloud {
            mesh: mesh.id(),
            world_from_local: transform.to_matrix(),
            params: cloud.0.clone(),
        })
    }
}

/// The plug-in: the cloud's extraction, the pipelines, the per-frame uniforms, and the node after the
/// reconstruction. Added after [`crate::reconstruction::ReconstructionPlugin`], whose node it follows.
pub(crate) struct PointsPlugin;

impl Plugin for PointsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "points.wgsl");
        app.add_plugins(ExtractComponentPlugin::<PointsCloud>::default());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PointsFrame>()
            .init_resource::<SpecializedRenderPipelines<PointsPipeline>>()
            .init_resource::<SpecializedMeshPipelines<PointsPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(Render, prepare_points.in_set(RenderSystems::Prepare))
            .add_render_graph_node::<ViewNodeRunner<PointsNode>>(Core3d, PointsLabel)
            .add_render_graph_edges(Core3d, (ReconstructionLabel, PointsLabel, Node3d::Bloom));
    }
}

/// The pass's uniform: `Points` in `points.wgsl`.
#[derive(ShaderType, Clone, Copy, Default)]
struct PointsUniform {
    clip_from_view: Mat4,
    view_from_clip: Mat4,
    world_from_view: Mat4,
    view_from_world: Mat4,
    world_from_local: Mat4,
}

/// ★ THIS FRAME'S POINTS: the cloud's mesh, the law's uniform, and each reconstructed view's uniform (by its
/// render-world entity).
#[derive(Resource, Default)]
struct PointsFrame {
    mesh: Option<AssetId<Mesh>>,
    star: Option<UniformBuffer<StarSkyParams>>,
    views: BTreeMap<Entity, UniformBuffer<PointsUniform>>,
}

/// The pass's pipelines: the two bind group layouts (with the air's table and without), the shader and the
/// full-screen vertex stage.
#[derive(Resource, Clone)]
struct PointsPipeline {
    air_layout: BindGroupLayoutDescriptor,
    bare_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    fullscreen: VertexState,
}

fn init_pipeline(
    mut commands: Commands,
    fullscreen: Res<FullscreenShader>,
    assets: Res<AssetServer>,
) {
    let stages = ShaderStages::VERTEX_FRAGMENT;
    // The air's own bindings, at the air's own numbers (`bevy_pbr::atmosphere::bindings`): the air, its
    // settings, the view, the lights, the transmittance table and its sampler.
    let air_layout = BindGroupLayoutDescriptor::new(
        "vd_points_air_layout",
        &BindGroupLayoutEntries::with_indices(
            stages,
            (
                (0, uniform_buffer::<GpuAtmosphere>(true)),
                (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                (3, uniform_buffer::<ViewUniform>(true)),
                (4, uniform_buffer::<GpuLights>(true)),
                (8, texture_2d(TextureSampleType::default())),
                (12, sampler(SamplerBindingType::Filtering)),
                (20, uniform_buffer::<PointsUniform>(false)),
                (21, uniform_buffer::<StarSkyParams>(false)),
                (
                    22,
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
            ),
        ),
    );
    let bare_layout = BindGroupLayoutDescriptor::new(
        "vd_points_bare_layout",
        &BindGroupLayoutEntries::with_indices(
            stages,
            (
                (20, uniform_buffer::<PointsUniform>(false)),
                (21, uniform_buffer::<StarSkyParams>(false)),
                (
                    22,
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
            ),
        ),
    );
    commands.insert_resource(PointsPipeline {
        air_layout,
        bare_layout,
        shader: load_embedded_asset!(assets.as_ref(), "points.wgsl"),
        fullscreen: fullscreen.to_vertex_state(),
    });
}

/// A pipeline's key: the view's format, and whether the view holds air.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PointsKey {
    hdr: bool,
    air: bool,
}

impl PointsPipeline {
    /// The descriptor both pipelines share: the layout, the fragment's target with TRUE ADDITION of the
    /// colour and the picture's alpha kept, and no depth.
    fn descriptor(&self, key: PointsKey, entry: &'static str) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();
        if key.air {
            shader_defs.push("AIR".into());
        }
        let format = if key.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };
        RenderPipelineDescriptor {
            label: Some(format!("vd_points_{entry}").into()),
            layout: vec![if key.air {
                self.air_layout.clone()
            } else {
                self.bare_layout.clone()
            }],
            vertex: self.fullscreen.clone(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some(entry.into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
            ..default()
        }
    }
}

/// ★ THE SUN'S PIPELINE: one full-screen triangle.
impl SpecializedRenderPipeline for PointsPipeline {
    type Key = PointsKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        self.descriptor(key, "sun_fragment")
    }
}

/// ★ THE STARS' PIPELINE: the cloud's own mesh, its four attributes at the star material's locations.
impl SpecializedMeshPipeline for PointsPipeline {
    type Key = PointsKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let buffers: VertexBufferLayout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            crate::ATTRIBUTE_STAR_CORNER.at_shader_location(1),
            crate::ATTRIBUTE_STAR_COLOR.at_shader_location(2),
            crate::ATTRIBUTE_STAR_BASE_R.at_shader_location(3),
        ])?;
        let mut descriptor = self.descriptor(key, "star_fragment");
        descriptor.vertex = VertexState {
            shader: self.shader.clone(),
            shader_defs: descriptor
                .fragment
                .as_ref()
                .map(|f| f.shader_defs.clone())
                .unwrap_or_default(),
            entry_point: Some("star_vertex".into()),
            buffers: vec![buffers],
        };
        Ok(descriptor)
    }
}

/// A view's two pipelines this frame: the sun's (with air only) and the stars' (with a cloud only).
#[derive(Component, Clone, Copy)]
struct PointsViewPipelines {
    sun: Option<CachedRenderPipelineId>,
    stars: Option<CachedRenderPipelineId>,
}

/// ★ THE PREPARE: the cloud's law into its uniform, each reconstructed view's matrices into its own (the
/// UNJITTERED projection: the points never take the jitter), and each view's pipelines.
#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources and queries
fn prepare_points(
    mut commands: Commands,
    mut frame: ResMut<PointsFrame>,
    clouds: Query<&ExtractedPointsCloud>,
    meshes: Res<RenderAssets<RenderMesh>>,
    views: Query<(Entity, &ExtractedView, Has<GpuAtmosphereSettings>), With<ReconstructionFrame>>,
    pipeline: Res<PointsPipeline>,
    cache: Res<PipelineCache>,
    mut suns: ResMut<SpecializedRenderPipelines<PointsPipeline>>,
    mut stars: ResMut<SpecializedMeshPipelines<PointsPipeline>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let cloud = clouds.iter().next();
    frame.mesh = cloud.map(|c| c.mesh);
    match cloud {
        Some(c) => {
            let star = frame
                .star
                .get_or_insert_with(|| UniformBuffer::from(c.params.clone()));
            star.set(c.params.clone());
            star.write_buffer(&device, &queue);
        }
        None => frame.star = None,
    }
    let layout = cloud
        .and_then(|c| meshes.get(c.mesh))
        .map(|m| m.layout.clone());
    let mut seen = BTreeSet::new();
    for (entity, view, air) in &views {
        let key = PointsKey { hdr: view.hdr, air };
        let sun = air.then(|| suns.specialize(&cache, &pipeline, key));
        let stars_id = match &layout {
            Some(layout) => match stars.specialize(&cache, &pipeline, key, layout) {
                Ok(id) => Some(id),
                Err(e) => {
                    tracing::warn!(error = %e, "the points pass's star pipeline cannot be made");
                    None
                }
            },
            None => None,
        };
        commands.entity(entity).insert(PointsViewPipelines {
            sun,
            stars: stars_id,
        });
        let world_from_view = view.world_from_view.to_matrix();
        let uniform = frame.views.entry(entity).or_default();
        uniform.set(PointsUniform {
            clip_from_view: view.clip_from_view,
            view_from_clip: view.clip_from_view.inverse(),
            world_from_view,
            view_from_world: world_from_view.inverse(),
            world_from_local: cloud.map_or(Mat4::IDENTITY, |c| c.world_from_local),
        });
        uniform.write_buffer(&device, &queue);
        seen.insert(entity);
    }
    frame.views.retain(|entity, _| seen.contains(entity));
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct PointsLabel;

/// ★ THE PASS: onto the display picture (loaded, never cleared), the sun's triangle, then the cloud.
#[derive(Default)]
struct PointsNode;

impl ViewNode for PointsNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ReconstructionTextures,
        &'static PointsViewPipelines,
        &'static ViewUniformOffset,
        Option<&'static ViewLightsUniformOffset>,
        Option<(
            &'static AtmosphereTextures,
            &'static DynamicUniformIndex<GpuAtmosphere>,
            &'static DynamicUniformIndex<GpuAtmosphereSettings>,
        )>,
    );

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, target, textures, pipelines, view_offset, lights_offset, air): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let (Some(frame), Some(cache), Some(pipeline)) = (
            world.get_resource::<PointsFrame>(),
            world.get_resource::<PipelineCache>(),
            world.get_resource::<PointsPipeline>(),
        ) else {
            return Ok(());
        };
        let (Some(points), Some(star)) = (
            frame
                .views
                .get(&graph.view_entity())
                .and_then(UniformBuffer::binding),
            frame.star.as_ref().and_then(UniformBuffer::binding),
        ) else {
            return Ok(());
        };
        let history = &textures.write.default_view;
        let device = render_context.render_device().clone();
        // The air's bindings when the view holds air, with their dynamic offsets in binding order.
        let bound: Option<(BindGroup, [u32; 4])> = match (air, lights_offset) {
            (Some((air_textures, air_index, settings_index)), Some(lights_offset)) => {
                let resources = (
                    world.get_resource::<ComponentUniforms<GpuAtmosphere>>(),
                    world.get_resource::<ComponentUniforms<GpuAtmosphereSettings>>(),
                    world.get_resource::<ViewUniforms>(),
                    world.get_resource::<LightMeta>(),
                    world.get_resource::<AtmosphereSampler>(),
                );
                let (Some(atmospheres), Some(settings), Some(views), Some(lights), Some(sampler)) =
                    resources
                else {
                    return Ok(());
                };
                let (Some(atmosphere), Some(settings), Some(view), Some(lights)) = (
                    atmospheres.binding(),
                    settings.binding(),
                    views.uniforms.binding(),
                    lights.view_gpu_lights.binding(),
                ) else {
                    return Ok(());
                };
                let group = device.create_bind_group(
                    "vd_points_air_bind_group",
                    &cache.get_bind_group_layout(&pipeline.air_layout),
                    &BindGroupEntries::with_indices((
                        (0, atmosphere),
                        (1, settings),
                        (3, view),
                        (4, lights),
                        (8, &air_textures.transmittance_lut.default_view),
                        (12, &**sampler),
                        (20, points.clone()),
                        (21, star.clone()),
                        (22, history),
                    )),
                );
                Some((
                    group,
                    [
                        air_index.index(),
                        settings_index.index(),
                        view_offset.offset,
                        lights_offset.offset,
                    ],
                ))
            }
            _ => None,
        };
        let bare;
        let (group, offsets): (&BindGroup, &[u32]) = match &bound {
            Some((group, offsets)) => (group, offsets),
            None => {
                bare = device.create_bind_group(
                    "vd_points_bare_bind_group",
                    &cache.get_bind_group_layout(&pipeline.bare_layout),
                    &BindGroupEntries::with_indices(((20, points), (21, star), (22, history))),
                );
                (&bare, &[])
            }
        };
        let sun = pipelines.sun.and_then(|id| cache.get_render_pipeline(id));
        let stars = pipelines.stars.and_then(|id| cache.get_render_pipeline(id));
        if sun.is_none() && stars.is_none() {
            return Ok(());
        }
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("vd_points"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = camera.viewport.as_ref() {
            pass.set_camera_viewport(viewport);
        }
        if let Some(sun) = sun {
            pass.set_render_pipeline(sun);
            pass.set_bind_group(0, group, offsets);
            pass.draw(0..3, 0..1);
        }
        let (Some(stars), Some(mesh)) = (stars, frame.mesh) else {
            return Ok(());
        };
        let (Some(allocator), Some(meshes)) = (
            world.get_resource::<MeshAllocator>(),
            world.get_resource::<RenderAssets<RenderMesh>>(),
        ) else {
            return Ok(());
        };
        let (Some(vertices), Some(render_mesh)) =
            (allocator.mesh_vertex_slice(&mesh), meshes.get(mesh))
        else {
            return Ok(());
        };
        pass.set_render_pipeline(stars);
        pass.set_bind_group(0, group, offsets);
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        match &render_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                count,
                index_format,
            } => {
                let Some(indices) = allocator.mesh_index_slice(&mesh) else {
                    return Ok(());
                };
                pass.set_index_buffer(indices.buffer.slice(..), *index_format);
                pass.draw_indexed(
                    indices.range.start..indices.range.start + count,
                    vertices.range.start as i32,
                    0..1,
                );
            }
            RenderMeshBufferInfo::NonIndexed => pass.draw(vertices.range.clone(), 0..1),
        }
        Ok(())
    }
}
