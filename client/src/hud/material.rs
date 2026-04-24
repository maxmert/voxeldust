//! Shared alpha-blended `StandardMaterial` preset for every HUD tile.
//! One material per tile because each tile has its own texture handle.
//! `unlit: true` so the HUD reads consistently regardless of lighting;
//! `double_sided: true` so you can see it from both sides of the face.

use bevy::prelude::*;

pub fn tile_material(image: Handle<Image>) -> StandardMaterial {
    // Opaque for MVP. True transparency is an H10 polish item.
    //
    // `unlit: true` in Bevy's PBR shader SKIPS the emissive add —
    // only `base_color × texture` reaches the output. If the texture
    // is black (empty), the tablet is black regardless of emissive
    // value. So:
    //   * `unlit: false` so emissive contributes.
    //   * WHITE base_color so `base_color × texture` = `texture`
    //     verbatim (no darkening).
    //   * High emissive so the panel reads well even if the texture
    //     hasn't been painted yet.
    StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(image),
        emissive: LinearRgba::new(0.4, 0.5, 0.7, 1.0),
        alpha_mode: AlphaMode::Opaque,
        unlit: false,
        double_sided: true,
        cull_mode: None,
        // Reduce roughness so some highlight shows up on ship lights.
        perceptual_roughness: 0.3,
        ..default()
    }
}

/// Create a new per-HUD-tile Image. Initialised as solid near-black so
/// the tablet reads as a dark panel until the widget paints its
/// content. RGBA8 sRGB, `MAIN_WORLD | RENDER_WORLD` usage so we can
/// write to it from the redraw system and have it sync to GPU.
pub fn new_hud_image(images: &mut Assets<Image>, size: u32) -> Handle<Image> {
    use bevy::asset::RenderAssetUsages;
    use bevy::image::Image;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    let usage = RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
    let img = Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        // Dark panel background: R=8, G=12, B=20, A=255. Slight blue
        // tint under emissive.
        &[8, 12, 20, 255],
        TextureFormat::Rgba8UnormSrgb,
        usage,
    );
    images.add(img)
}

/// Create a fully-transparent Image for block-face HUD panels.
/// Every pixel starts at RGBA = `(0, 0, 0, 0)`. Widgets paint opaque
/// pixels over this alpha-zero canvas, so the host block stays
/// visible through the empty regions of the tile. Paired with
/// `StandardMaterial { alpha_mode: AlphaMode::Blend, .. }` on the
/// tile entity.
pub fn new_transparent_hud_image(images: &mut Assets<Image>, size: u32) -> Handle<Image> {
    use bevy::asset::RenderAssetUsages;
    use bevy::image::Image;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    let usage = RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
    let img = Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        usage,
    );
    images.add(img)
}

/// Create a new Image destined for an egui render-to-texture camera.
/// Adds `RENDER_ATTACHMENT` on top of the standard HUD-image usage so
/// bevy's render graph can bind it as a colour target. Used by the
/// held tablet so bevy_egui 0.39 can render an interactive UI directly
/// onto the tablet surface.
pub fn new_egui_target_image(images: &mut Assets<Image>, size: u32) -> Handle<Image> {
    use bevy::asset::RenderAssetUsages;
    use bevy::image::Image;
    use bevy::render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureUsages,
    };
    let usage = RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
    let mut img = Image::new_fill(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[8, 12, 20, 255],
        TextureFormat::Rgba8UnormSrgb,
        usage,
    );
    img.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT
        | TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING;
    images.add(img)
}
