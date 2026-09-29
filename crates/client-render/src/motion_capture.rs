//! ★ THE MOTION'S RENDER-WORLD PARTS (the temporal path's slice T1 items 5 and 6;
//! `docs/investigation/2026-09-28/temporal_path_design.md` §11): under the motion prepass
//! (`VD_PREPASS=1`),
//! - THE SENTINEL: every view's motion target clears to the motion census's sentinel
//!   (`vd_client_harness::motion_census::MOTION_SENTINEL`) instead of the engine's nought, so a pixel
//!   that no surface wrote keeps a value that no surface writes;
//! - THE READBACK (the headless capture only): after the frame's own passes, the picture camera's
//!   motion target and its depth are copied into one buffer and sent to the capture beside the
//!   picture and the probe, which writes them as `<stem>.motion.bin`.
//!
//! The engine makes its motion target without the copy usage, so under the readback the picture's
//! view draws its motion into a target of the same size and format that the copy may read; the
//! engine's bind groups are made after the swap, so every pass reads the one target.
//!
//! The copy is recorded in the frame's own command stream after the camera driver, so it reads THIS
//! frame's motion and depth. Example: the census flight stands on the hill; the capture of frame 900
//! writes `hill.png`, `hill.probe.png` and `hill.motion.bin`, and the census counts the ground pixels
//! of frame 900 that hold a depth and keep the sentinel.

use std::sync::atomic::{AtomicBool, Ordering};

use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
};
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, MapMode, Origin3d, PollType, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureUsages,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::{ColorAttachment, TextureCache};
use bevy::render::view::ViewDepthTexture;
use bevy::render::{Render, RenderApp, RenderSystems};
use vd_client_harness::motion_census::{MOTION_SENTINEL, MOTION_TEXEL_BYTES};

/// The plug-in: the sentinel on every motion target, and the readback when `readback` is set (the
/// headless capture; a window has no capture to write it to).
pub(crate) struct MotionCapturePlugin {
    pub readback: bool,
}

impl Plugin for MotionCapturePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            motion_target_sentinel
                .in_set(RenderSystems::PrepareBindGroups)
                .before(bevy::pbr::prepare_mesh_view_bind_groups),
        );
        if !self.readback {
            return;
        }
        render_app
            .insert_resource(MotionReadback::default())
            .add_systems(Render, receive_motion.after(RenderSystems::Render));
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(MotionCopyLabel, MotionCopyNode::default());
        graph.add_node_edge(bevy::render::graph::CameraDriverLabel, MotionCopyLabel);
    }
}

/// The readback's buffer and whether this frame's copy was recorded.
#[derive(Resource, Default)]
struct MotionReadback {
    /// The buffer, the picture's size and the padded row, made for the picture camera's target and
    /// made again when its size changes.
    buffer: Option<(Buffer, UVec2, u32)>,
    /// Set by the copy node when it recorded this frame's copy; cleared at each update.
    copied: AtomicBool,
}

/// The picture camera's motion target: a camera drawn at order nought or later (the probe camera
/// draws first, at −1) that has one.
fn picture_motion<'a>(
    camera: &ExtractedCamera,
    textures: &'a ViewPrepassTextures,
) -> Option<&'a ColorAttachment> {
    textures
        .motion_vectors
        .as_ref()
        .filter(|_| camera.order >= 0)
}

/// ★ THE SENTINEL (item 5), and under the readback the copyable target: the engine made this
/// frame's prepass textures in its prepare stage; this swaps the motion's attachment before any bind
/// group reads it.
fn motion_target_sentinel(
    mut views: Query<(&ExtractedCamera, &mut ViewPrepassTextures)>,
    readback: Option<Res<MotionReadback>>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
) {
    let clear = LinearRgba::new(MOTION_SENTINEL, MOTION_SENTINEL, 0.0, 0.0);
    for (camera, mut textures) in &mut views {
        let Some(motion) = textures.motion_vectors.as_ref() else {
            continue;
        };
        let copyable = readback.is_some() && picture_motion(camera, &textures).is_some();
        let texture = if copyable {
            let own = &motion.texture.texture;
            texture_cache.get(
                &device,
                TextureDescriptor {
                    label: Some("vd_motion_target_copyable"),
                    size: own.size(),
                    mip_level_count: 1,
                    sample_count: own.sample_count(),
                    dimension: TextureDimension::D2,
                    format: own.format(),
                    usage: own.usage() | TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )
        } else {
            motion.texture.clone()
        };
        textures.motion_vectors = Some(ColorAttachment::new(texture, None, None, Some(clear)));
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct MotionCopyLabel;

/// ★ THE COPY (item 6): the picture camera's motion target and its depth into the readback's
/// buffer, the motion rows first and the depth rows after them, each row padded as a copy wants it.
#[derive(Default)]
struct MotionCopyNode {
    views: Option<
        QueryState<(
            &'static ExtractedCamera,
            &'static ViewPrepassTextures,
            &'static ViewDepthTexture,
        )>,
    >,
}

impl Node for MotionCopyNode {
    fn update(&mut self, world: &mut World) {
        let views = self.views.get_or_insert_with(|| QueryState::new(world));
        views.update_archetypes(world);
        let size = views.iter_manual(world).find_map(|(camera, textures, _)| {
            picture_motion(camera, textures).map(|m| {
                let s = m.texture.texture.size();
                UVec2::new(s.width, s.height)
            })
        });
        let device = world.resource::<RenderDevice>().clone();
        let mut readback = world.resource_mut::<MotionReadback>();
        readback.copied.store(false, Ordering::Relaxed);
        let Some(size) = size else {
            return;
        };
        if readback.buffer.as_ref().is_some_and(|(_, s, _)| *s == size) {
            return;
        }
        let padded_row =
            RenderDevice::align_copy_bytes_per_row(size.x as usize * MOTION_TEXEL_BYTES) as u32;
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("vd motion readback"),
            size: 2 * u64::from(padded_row) * u64::from(size.y),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        readback.buffer = Some((buffer, size, padded_row));
    }

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(views) = &self.views else {
            return Ok(());
        };
        let readback = world.resource::<MotionReadback>();
        let Some((buffer, size, padded_row)) = &readback.buffer else {
            return Ok(());
        };
        let Some((motion, depth)) =
            views
                .iter_manual(world)
                .find_map(|(camera, textures, depth)| {
                    picture_motion(camera, textures).map(|m| (m, depth))
                })
        else {
            return Ok(());
        };
        let own = &motion.texture.texture;
        let extent = own.size();
        // Only a single-sampled, copyable target of the buffer's size, with a depth of the same
        // size; anything else is not copied, and the capture writes no motion file for the frame.
        let fits = (UVec2::new(extent.width, extent.height) == *size)
            & (own.sample_count() == 1)
            & own.usage().contains(TextureUsages::COPY_SRC)
            & (depth.texture.size() == extent)
            & depth.texture.usage().contains(TextureUsages::COPY_SRC);
        if !fits {
            return Ok(());
        }
        let at = |offset: u64| TexelCopyBufferInfo {
            buffer,
            layout: TexelCopyBufferLayout {
                offset,
                bytes_per_row: Some(*padded_row),
                rows_per_image: None,
            },
        };
        let encoder = render_context.command_encoder();
        encoder.copy_texture_to_buffer(own.as_image_copy(), at(0), extent);
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &depth.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::DepthOnly,
            },
            at(u64::from(*padded_row) * u64::from(size.y)),
            extent,
        );
        readback.copied.store(true, Ordering::Relaxed);
        Ok(())
    }
}

/// ★ THE RECEIVE: after the frame's submission, the buffer mapped and sent to the capture with the
/// frame and stamp the picture's copiers carry. A frame with no copy sends an empty readback, so a
/// capture waiting for its motion is answered and writes no motion file.
fn receive_motion(
    readback: Res<MotionReadback>,
    copiers: Res<super::ImageCopiers>,
    device: Res<RenderDevice>,
    sender: Res<super::RenderWorldSender>,
) {
    let copied = readback.copied.load(Ordering::Relaxed);
    let bytes = match &readback.buffer {
        Some((buffer, _, _)) if copied => {
            let slice = buffer.slice(..);
            let (tx, rx) = crossbeam_channel::bounded(1);
            slice.map_async(MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            if device.poll(PollType::wait_indefinitely()).is_err() {
                return;
            }
            if !matches!(rx.recv(), Ok(Ok(()))) {
                return;
            }
            let data = slice.get_mapped_range().to_vec();
            buffer.unmap();
            data
        }
        _ => Vec::new(),
    };
    let _ = sender.send((
        super::COPIER_MOTION,
        copiers.frame,
        copiers.stamp.clone(),
        bytes,
    ));
}
