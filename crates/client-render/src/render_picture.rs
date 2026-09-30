//! ★ THE RENDER PICTURE FOR THE GATES (the temporal path's slice T2; `docs/investigation/2026-09-28/
//! temporal_path_design.md` §12.3): under the reconstruction the headless capture writes, beside each display
//! picture, the frame's RENDER picture — the jittered picture the main passes drew, BEFORE the history —
//! through the gate's tone map, on the render grid beside the probe (`render_picture.wgsl`). The seam gate reads
//! it, so the history can never hide a pop from the gate; the display picture gets a reading of its own.
//!
//! The copy stands after the main passes and before the reconstruction, and acts only when the capture set its
//! image ([`RenderPictureImage`]); a window has none.
//!
//! **Example.** The hull flies the 240 m/s leg under `VD_TEMPORAL=1 VD_RENDER_SCALE=0.5`: frame 1 706 writes
//! `pop-hull-240-m-s-010-0001.png` (the display, 2 568 × 1 440), and `….render.png` and `….probe.png` (both the
//! render grid, 1 284 × 720); the seam gate compares the render pictures of frames 1 705 and 1 706 through the probe.

use bevy::asset::{embedded_asset, load_embedded_asset};
use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::BevyDefault as _;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::texture_2d;
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, FragmentState, Operations, PipelineCache,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    TextureFormat, TextureSampleType,
};
use bevy::render::renderer::RenderContext;
use bevy::render::texture::GpuImage;
use bevy::render::view::ViewTarget;
use bevy::render::{RenderApp, RenderStartup};

use crate::reconstruction::{ReconstructionFrame, ReconstructionLabel};

/// ★ THE CAPTURE'S RENDER-PICTURE IMAGE: the render grid, eight-bit sRGB, set by the capture's setup under the
/// reconstruction; the render world draws the frame's render picture into it, and a copier reads it back.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct RenderPictureImage(pub(crate) Handle<Image>);

/// The plug-in: the image's extraction, the pipeline, and the node between the main passes and the
/// reconstruction. Added after [`crate::reconstruction::ReconstructionPlugin`], whose node it precedes.
pub(crate) struct RenderPicturePlugin;

impl Plugin for RenderPicturePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "render_picture.wgsl");
        app.add_plugins(ExtractResourcePlugin::<RenderPictureImage>::default());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_pipeline)
            .add_render_graph_node::<ViewNodeRunner<RenderPictureNode>>(Core3d, RenderPictureLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPassPostProcessing,
                    RenderPictureLabel,
                    ReconstructionLabel,
                ),
            );
    }
}

/// The copy's pipeline: one full-screen triangle through the gate's tone map.
#[derive(Resource)]
struct RenderPicturePipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
}

fn init_pipeline(
    mut commands: Commands,
    fullscreen: Res<FullscreenShader>,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "vd_render_picture_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    );
    let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("vd_render_picture".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: load_embedded_asset!(assets.as_ref(), "render_picture.wgsl"),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::bevy_default(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(RenderPicturePipeline { layout, id });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct RenderPictureLabel;

/// ★ THE COPY: the view's render picture (its main texture before the reconstruction) into the capture's image.
#[derive(Default)]
struct RenderPictureNode;

impl ViewNode for RenderPictureNode {
    type ViewQuery = (&'static ViewTarget, &'static ReconstructionFrame);

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (target, _frame): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let (Some(image), Some(pipeline), Some(cache), Some(images)) = (
            world.get_resource::<RenderPictureImage>(),
            world.get_resource::<RenderPicturePipeline>(),
            world.get_resource::<PipelineCache>(),
            world.get_resource::<RenderAssets<GpuImage>>(),
        ) else {
            return Ok(());
        };
        let (Some(render_pipeline), Some(gpu_image)) =
            (cache.get_render_pipeline(pipeline.id), images.get(&image.0))
        else {
            return Ok(());
        };
        let group = render_context.render_device().create_bind_group(
            "vd_render_picture_bind_group",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::single(target.main_texture_view()),
        );
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("vd_render_picture"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &gpu_image.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_render_pipeline(render_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
