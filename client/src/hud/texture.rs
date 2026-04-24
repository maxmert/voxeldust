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
use crate::hud::tile::{HudConfig, HudTexture, HudTile};
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

        // Widget paint.
        let value = signals.get(&config.channel).map(|r| r.value.clone());
        if let Some(widget) = registry.get(config.kind) {
            let ctx = DrawCtx {
                pixels: buf.as_mut_slice(),
                size,
                caption: &config.caption,
                opacity: config.opacity,
            };
            widget.draw(ctx, value, config);
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
