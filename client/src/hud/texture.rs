//! Per-tile texture redraw pass.
//!
//! Runs once per frame per tile with a change-detection + floor-cadence
//! check: if the subscribed signal value didn't change AND the floor
//! cadence (`redraw_floor_ms`) hasn't elapsed, skip the redraw. Else,
//! clear the tile texture, let the widget paint, overlay AR markers
//! (if enabled), draw the in-game cursor (if focused).

use std::time::Instant;

use bevy::prelude::*;

use crate::hud::ar::{draw_ar_markers, ArDrawCtx};
use crate::hud::focus::HudFocusState;
use crate::hud::signal_registry::SignalRegistry;
use crate::hud::tablet::HeldTablet;
use crate::hud::tile::{HudConfig, HudPanelLayout, HudTexture, HudTile, HudWidgetSlot, WidgetKind};
use crate::hud::widget::{DrawCtx, HudWidgetRegistry};
use crate::remote::{RemoteDebris, RemotePlayers, RemoteShips};
use crate::shard::{CameraWorldPos, PrimaryWorldState, SecondaryWorldStates};

#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct HudTextureSet;

pub struct HudTexturePlugin;

impl Plugin for HudTexturePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HudWidgetRegistry>()
            .configure_sets(Update, HudTextureSet)
            .add_systems(Update, redraw_hud_textures.in_set(HudTextureSet));
    }
}

fn redraw_hud_textures(
    registry: Res<HudWidgetRegistry>,
    signals: Res<SignalRegistry>,
    focus: Res<HudFocusState>,
    camera_world: Res<CameraWorldPos>,
    primary_ws: Res<PrimaryWorldState>,
    secondary_ws: Res<SecondaryWorldStates>,
    remote_ships: Res<RemoteShips>,
    remote_players: Res<RemotePlayers>,
    remote_debris: Res<RemoteDebris>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut q: Query<(
        Entity,
        &HudTile,
        &HudConfig,
        &mut HudTexture,
        &GlobalTransform,
        Option<&HeldTablet>,
    )>,
) {
    let now = Instant::now();
    for (entity, _tile, config, mut tex, tile_gt, is_tablet) in q.iter_mut() {
        // The tablet renders its surface via the egui render-to-image
        // camera in `tablet_ui::TabletPaintPass` — don't also overwrite
        // it from this CPU-painted pipeline.
        if is_tablet.is_some() {
            continue;
        }
        // Floor-cadence gate.
        if now.saturating_duration_since(tex.last_draw_at).as_millis()
            < tex.redraw_floor_ms as u128
        {
            continue;
        }
        tex.last_draw_at = now;

        let size = tex.size;
        let pixel_count = (size * size) as usize;
        let byte_count = pixel_count * 4;

        // Build a FRESH pixel buffer every redraw. Writing to a
        // scratch Vec and then assigning it into the Image asset via
        // `image.data = Some(new)` unambiguously marks the asset
        // dirty, sidestepping any Bevy edge case where in-place Vec
        // mutation fails to propagate to GPU.
        let mut buf = vec![0u8; byte_count];

        // Widget paint — Single = one widget over the whole face;
        // Quad = four widgets, each filling a 2×2 quadrant. For Quad
        // we paint each slot into a temp sub-buffer at `size/2` and
        // blit the resulting pixels into the corresponding quadrant
        // of the main buffer. `AR overlay` and `cursor` are applied
        // on top of the composited result using the full face.
        match config.layout {
            HudPanelLayout::Single => {
                paint_slot(
                    buf.as_mut_slice(),
                    size,
                    0,
                    0,
                    size,
                    &registry,
                    &signals,
                    config.kind,
                    &config.channel,
                    &config.caption,
                    config.opacity,
                    config,
                );
            }
            HudPanelLayout::Quad => {
                let half = size / 2;
                // Slot 0 = outer HudConfig (TL).
                paint_slot(
                    buf.as_mut_slice(), size, 0,     0,     half,
                    &registry, &signals,
                    config.kind, &config.channel, &config.caption,
                    config.opacity, config,
                );
                if let Some(extras) = &config.extra_slots {
                    let [tr, bl, br] = extras.as_ref();
                    paint_slot_from(&mut buf, size, half, 0,    half, &registry, &signals, tr, config.opacity, config);
                    paint_slot_from(&mut buf, size, 0,    half, half, &registry, &signals, bl, config.opacity, config);
                    paint_slot_from(&mut buf, size, half, half, half, &registry, &signals, br, config.opacity, config);
                }
            }
        }

        // AR overlay.
        if config.ar_enabled {
            let ctx = ArDrawCtx {
                tile_transform: tile_gt,
                tile_world_size: tex.world_size,
                camera_world: camera_world.pos,
                primary_ws: primary_ws.latest.as_ref(),
                secondary_ws: &secondary_ws,
                remote_ships: &remote_ships,
                remote_players: &remote_players,
                remote_debris: &remote_debris,
            };
            draw_ar_markers(buf.as_mut_slice(), size, config, &ctx);
        }

        // In-game cursor.
        if focus.focused_tile == Some(entity) {
            draw_cursor(buf.as_mut_slice(), size, focus.cursor_uv);
        }

        let _ = pixel_count; // kept: used in future change-detection path

        // Upload strategy: mutate the EXISTING Image's data in place,
        // then touch the material via `materials.get_mut()` to force
        // change-detection. Per bevy#17350 workaround.
        //
        // We also mutate a material field (not just call get_mut) to
        // ensure Bevy's Prepare pass sees a real modification — the
        // `get_mut` call alone always fires `AssetEvent::Modified`
        // but the render pipeline's material-preparation step may
        // short-circuit based on value equality.
        if let Some(image) = images.get_mut(&tex.handle) {
            image.data = Some(buf);
        }
        if let Some(mat) = materials.get_mut(&tex.material) {
            // Cheapest real mutation: toggle the texture handle
            // clone-ref to itself. `base_color_texture` is Option<Handle>;
            // reassignment with the same Some(..) forces the binding
            // layer to re-extract this material into the render world.
            mat.base_color_texture = Some(tex.handle.clone());
        }
    }
}

/// Paint one widget slot into a sub-rectangle of the full tile
/// pixel buffer. Used by both Single layouts (sub-rect = the whole
/// face) and Quad layouts (sub-rect = one quadrant). Widgets write
/// into a temp buffer sized `sub_size × sub_size`; result is blitted
/// row-by-row into the main buffer starting at `(x, y)`.
#[allow(clippy::too_many_arguments)]
fn paint_slot(
    full_buf: &mut [u8],
    full_size: u32,
    x: u32,
    y: u32,
    sub_size: u32,
    registry: &HudWidgetRegistry,
    signals: &SignalRegistry,
    kind: WidgetKind,
    channel: &str,
    caption: &str,
    opacity: f32,
    config: &HudConfig,
) {
    if kind == WidgetKind::None {
        return;
    }
    let Some(widget) = registry.get(kind) else { return };
    let value = signals.get(channel).map(|r| r.value.clone());
    let mut sub = vec![0u8; (sub_size * sub_size * 4) as usize];
    let ctx = DrawCtx {
        pixels: sub.as_mut_slice(),
        size: sub_size,
        caption,
        opacity,
    };
    widget.draw(ctx, value, config);
    blit(full_buf, full_size, &sub, x, y, sub_size);
}

/// Variant for Quad slots 1-3 (read from `extra_slots`).
#[allow(clippy::too_many_arguments)]
fn paint_slot_from(
    full_buf: &mut [u8],
    full_size: u32,
    x: u32,
    y: u32,
    sub_size: u32,
    registry: &HudWidgetRegistry,
    signals: &SignalRegistry,
    slot: &HudWidgetSlot,
    opacity: f32,
    config: &HudConfig,
) {
    paint_slot(
        full_buf, full_size, x, y, sub_size, registry, signals,
        slot.kind, &slot.channel, &slot.caption, opacity, config,
    );
}

/// Copy `sub_size × sub_size` RGBA pixels from `src` into the full
/// buffer at origin `(x, y)`. Overwrites; assumes `src` is tightly
/// packed row-major at `sub_size` stride.
fn blit(dst: &mut [u8], dst_size: u32, src: &[u8], x: u32, y: u32, sub_size: u32) {
    let dst_stride = dst_size as usize * 4;
    let src_stride = sub_size as usize * 4;
    for row in 0..sub_size as usize {
        let src_off = row * src_stride;
        let dst_off = (y as usize + row) * dst_stride + x as usize * 4;
        dst[dst_off..dst_off + src_stride]
            .copy_from_slice(&src[src_off..src_off + src_stride]);
    }
}

/// Draw a small arrow/crosshair cursor at the given UV (0..1) on the
/// tile texture. Written last so it's always on top.
fn draw_cursor(pixels: &mut [u8], size: u32, uv: Vec2) {
    let cx = (uv.x.clamp(0.0, 1.0) * (size - 1) as f32) as i32;
    let cy = (uv.y.clamp(0.0, 1.0) * (size - 1) as f32) as i32;
    let color = [255u8, 255, 255, 255];
    let radius = 4_i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let d2 = dx * dx + dy * dy;
            // Thin cross.
            let inside_cross = (dx.abs() <= 1 && dy.abs() <= radius)
                || (dy.abs() <= 1 && dx.abs() <= radius);
            if !inside_cross || d2 > radius * radius {
                continue;
            }
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
                continue;
            }
            let idx = (y as usize * size as usize + x as usize) * 4;
            if idx + 3 >= pixels.len() {
                continue;
            }
            pixels[idx] = color[0];
            pixels[idx + 1] = color[1];
            pixels[idx + 2] = color[2];
            pixels[idx + 3] = color[3];
        }
    }
}
