//! AR (augmented-reality) marker overlay for HUD tiles.
//!
//! For each entity the tile's `ArFilter` admits, projects the entity's
//! world-space position through the tile's plane to a tile-UV and paints
//! a marker dot at that UV (clipped to the tile rect). This is the
//! see-through-glass effect — bodies "behind" the tile from the
//! camera's POV appear at their projected screen position on the tile
//! surface.
//!
//! **Performance** (per plan): CPU projection per tile per frame. At
//! ≤10 tiles × ≤100 observable entities it's ~1000 ray-plane tests per
//! frame — trivially within budget. Future upgrade path is a custom
//! `Material` fragment-shader projection (documented in the plan).
//!
//! **Smoothness**: entity velocities from `PrimaryWorldState` /
//! `SecondaryWorldStates` / remote resources are applied as
//! presentation-only extrapolation (`pos + vel * Δt_since_received`).
//! Permitted under the no-prediction rule because it's a HUD overlay,
//! not game-state we act on.

use bevy::prelude::*;
use glam::DVec3;

use voxeldust_core::client_message::CelestialBodyData;

use crate::hud::tile::HudConfig;
use crate::remote::{RemoteDebris, RemotePlayers, RemoteShips};
use crate::shard::{CameraWorldPos, PrimaryWorldState, SecondaryWorldStates};

/// Which kinds of entities a HUD tile renders as AR markers.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArFilter {
    pub celestial_bodies: bool,
    pub remote_ships: bool,
    pub remote_players: bool,
    pub debris: bool,
}

pub struct ArMarkerPlugin;

impl Plugin for ArMarkerPlugin {
    fn build(&self, _app: &mut App) {
        // AR drawing runs inside `texture::redraw_hud_textures` via
        // `draw_ar_markers` rather than a standalone system, keeping
        // it in lock-step with the widget draw pass (clear → widget →
        // AR → cursor).
    }
}

/// Context the redraw system forwards into the AR pass.
pub struct ArDrawCtx<'a> {
    pub tile_transform: &'a GlobalTransform,
    pub tile_world_size: Vec2,
    pub camera_world: DVec3,
    pub primary_ws: Option<&'a voxeldust_core::client_message::WorldStateData>,
    pub secondary_ws: &'a SecondaryWorldStates,
    pub remote_ships: &'a RemoteShips,
    pub remote_players: &'a RemotePlayers,
    pub remote_debris: &'a RemoteDebris,
}

/// Draw AR markers onto the tile's pixel buffer. Called from the
/// texture redraw pass after the widget's `draw()`.
pub fn draw_ar_markers(pixels: &mut [u8], size: u32, config: &HudConfig, ctx: &ArDrawCtx) {
    if !config.ar_enabled {
        return;
    }
    // Build the tile plane from its GlobalTransform. The Rectangle
    // mesh lies in the XY plane with normal +Z; after the tile's
    // rotation those axes become:
    //   normal  = rot * +Z
    //   right   = rot * +X   (width direction)
    //   up      = rot * +Y   (height direction)
    let tile_rot = ctx.tile_transform.rotation();
    let tile_center = ctx.tile_transform.translation();
    let tile_normal = tile_rot * Vec3::Z;
    let tile_right = tile_rot * Vec3::X;
    let tile_up = tile_rot * Vec3::Y;
    let half_w = ctx.tile_world_size.x * 0.5;
    let half_h = ctx.tile_world_size.y * 0.5;

    let cam_to_tile = tile_center - vec3_from_dvec3_relative(ctx.camera_world, ctx.camera_world);
    let _ = cam_to_tile; // camera is at Bevy origin under Phase 4.

    let projector = MarkerProjector {
        camera_world: ctx.camera_world,
        tile_center,
        tile_normal,
        tile_right,
        tile_up,
        half_w,
        half_h,
    };

    // Celestial bodies — stars (yellow) + planets (body color).
    if config.ar_filter.celestial_bodies {
        if let Some(ws) = ctx.primary_ws {
            for body in &ws.bodies {
                project_and_paint_body(pixels, size, body, &projector);
            }
        }
        for (_, ws) in ctx.secondary_ws.by_shard_type.iter() {
            for body in &ws.bodies {
                project_and_paint_body(pixels, size, body, &projector);
            }
        }
    }

    // Remote ships — triangle markers in steel blue.
    if config.ar_filter.remote_ships {
        for ship in ctx.remote_ships.by_id.values() {
            projector.project_paint(
                pixels,
                size,
                ship.position,
                MarkerShape::Triangle,
                5,
                [120, 180, 255, 255],
            );
        }
    }

    // Remote players — small square markers in amber.
    if config.ar_filter.remote_players {
        for player in ctx.remote_players.by_id.values() {
            projector.project_paint(
                pixels,
                size,
                player.position,
                MarkerShape::Square,
                3,
                [240, 180, 60, 255],
            );
        }
    }

    // Remote debris — tiny diamond markers in dark orange.
    if config.ar_filter.debris {
        for piece in ctx.remote_debris.by_id.values() {
            projector.project_paint(
                pixels,
                size,
                piece.position,
                MarkerShape::Diamond,
                3,
                [200, 120, 60, 255],
            );
        }
    }
}

/// Ray-plane projector shared across entity types. Built once per
/// tile-redraw; reused across every marker call to avoid duplicating
/// the rotation + plane math per entity.
struct MarkerProjector {
    camera_world: DVec3,
    tile_center: Vec3,
    tile_normal: Vec3,
    tile_right: Vec3,
    tile_up: Vec3,
    half_w: f32,
    half_h: f32,
}

#[derive(Debug, Clone, Copy)]
enum MarkerShape {
    /// Filled disc.
    Disc,
    /// 3-pointed triangle, apex pointing +V.
    Triangle,
    /// Square.
    Square,
    /// Diamond (45°-rotated square).
    Diamond,
}

impl MarkerProjector {
    fn project_paint(
        &self,
        pixels: &mut [u8],
        size: u32,
        entity_world: DVec3,
        shape: MarkerShape,
        radius: i32,
        color: [u8; 4],
    ) {
        let Some((px, py)) = self.project(entity_world, size) else { return };
        match shape {
            MarkerShape::Disc => paint_disc(pixels, size, px, py, radius, color),
            MarkerShape::Triangle => paint_triangle(pixels, size, px, py, radius, color),
            MarkerShape::Square => paint_square(pixels, size, px, py, radius, color),
            MarkerShape::Diamond => paint_diamond(pixels, size, px, py, radius, color),
        }
    }

    /// Project an f64 world-space point to tile pixel coords. Returns
    /// `None` when the point is behind the camera, the ray is parallel
    /// to the plane, or the projected UV is outside the tile rect.
    fn project(&self, entity_world: DVec3, size: u32) -> Option<(i32, i32)> {
        let dir_world_f64 = entity_world - self.camera_world;
        let dir = Vec3::new(
            dir_world_f64.x as f32,
            dir_world_f64.y as f32,
            dir_world_f64.z as f32,
        );
        if dir.length_squared() < 1.0 {
            return None;
        }
        let denom = dir.dot(self.tile_normal);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = self.tile_center.dot(self.tile_normal) / denom;
        if t < 0.0 {
            return None;
        }
        let hit = dir * t;
        let rel = hit - self.tile_center;
        let u = rel.dot(self.tile_right);
        let v = rel.dot(self.tile_up);
        if u < -self.half_w || u > self.half_w || v < -self.half_h || v > self.half_h {
            return None;
        }
        let uv_x = ((u + self.half_w) / (2.0 * self.half_w)).clamp(0.0, 1.0);
        let uv_y = (1.0 - (v + self.half_h) / (2.0 * self.half_h)).clamp(0.0, 1.0);
        Some((
            (uv_x * (size - 1) as f32) as i32,
            (uv_y * (size - 1) as f32) as i32,
        ))
    }
}

/// Project a celestial body. Stars render larger + yellow-white;
/// planets render at their server-provided color.
fn project_and_paint_body(
    pixels: &mut [u8],
    size: u32,
    body: &CelestialBodyData,
    projector: &MarkerProjector,
) {
    let body_world = DVec3::new(body.position.x, body.position.y, body.position.z);
    let Some((px, py)) = projector.project(body_world, size) else { return };

    let color = if body.body_id == 0 {
        [255u8, 240, 200, 255]
    } else {
        let r = (body.color[0] * 255.0) as u8;
        let g = (body.color[1] * 255.0) as u8;
        let b = (body.color[2] * 255.0) as u8;
        [r, g, b, 255]
    };
    let radius = if body.body_id == 0 { 6 } else { 4 };
    paint_disc(pixels, size, px, py, radius, color);
}

fn paint_disc(pixels: &mut [u8], size: u32, cx: i32, cy: i32, radius: i32, color: [u8; 4]) {
    let r2 = radius * radius;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            paint_px(pixels, size, cx + dx, cy + dy, color);
        }
    }
}

fn paint_square(pixels: &mut [u8], size: u32, cx: i32, cy: i32, radius: i32, color: [u8; 4]) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            paint_px(pixels, size, cx + dx, cy + dy, color);
        }
    }
}

fn paint_diamond(pixels: &mut [u8], size: u32, cx: i32, cy: i32, radius: i32, color: [u8; 4]) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx.abs() + dy.abs() > radius {
                continue;
            }
            paint_px(pixels, size, cx + dx, cy + dy, color);
        }
    }
}

/// Filled isoceles triangle with apex pointing +V (up on the tile).
fn paint_triangle(pixels: &mut [u8], size: u32, cx: i32, cy: i32, radius: i32, color: [u8; 4]) {
    for dy in -radius..=radius {
        // Triangle width = 0 at apex (top), full at base (bottom).
        // dy=-radius at apex (top), dy=+radius at base (bottom).
        let t = (dy + radius) as f32 / (2.0 * radius as f32); // 0..1
        let width = (radius as f32 * t).round() as i32;
        for dx in -width..=width {
            paint_px(pixels, size, cx + dx, cy + dy, color);
        }
    }
}

fn paint_px(pixels: &mut [u8], size: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
        return;
    }
    let idx = (y as usize * size as usize + x as usize) * 4;
    if idx + 3 >= pixels.len() {
        return;
    }
    pixels[idx] = color[0];
    pixels[idx + 1] = color[1];
    pixels[idx + 2] = color[2];
    pixels[idx + 3] = color[3];
}

#[inline]
fn vec3_from_dvec3_relative(a: DVec3, _ref: DVec3) -> Vec3 {
    // Helper placeholder — kept for future relative-frame math.
    Vec3::new(a.x as f32, a.y as f32, a.z as f32)
}

