//! Toggle indicator: "ON" / "OFF" caption with a glowing dot. Large
//! text reads at any tile size; dot colour tracks state.

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::font;
use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, WidgetKind};
use crate::hud::widget::{DrawCtx, HudWidget};

pub struct ToggleWidget;

impl HudWidget for ToggleWidget {
    fn kind(&self) -> WidgetKind {
        WidgetKind::Toggle
    }
    fn name(&self) -> &'static str {
        "Toggle"
    }
    fn supported_properties(&self) -> &'static [SignalProperty] {
        &[SignalProperty::Active]
    }
    fn draw(&self, ctx: DrawCtx, value: Option<SignalValue>, _config: &HudConfig) {
        let on = value.map(|v| v.as_bool()).unwrap_or(false);
        let w = ctx.size as usize;
        let h = ctx.size as usize;
        let alpha = (255.0 * ctx.opacity) as u8;
        let caption_color = [180u8, 200, 220, alpha];
        let state_color = if on {
            [80u8, 240, 140, alpha]
        } else {
            [160u8, 160, 170, alpha]
        };

        // Caption top.
        if !ctx.caption.is_empty() {
            let scale = small_scale(w);
            font::draw_text(
                ctx.pixels,
                w,
                h,
                ctx.caption,
                (scale * 2).max(4),
                (scale * 2).max(4),
                scale,
                caption_color,
            );
        }

        // Centered "ON" / "OFF".
        let label = if on { "ON" } else { "OFF" };
        let scale = large_scale(w);
        let tw = font::text_width(label, scale);
        let th = font::GLYPH_H as i32 * scale;
        font::draw_text(
            ctx.pixels,
            w,
            h,
            label,
            (w as i32 - tw) / 2,
            (h as i32 - th) / 2,
            scale,
            state_color,
        );

        // Status dot bottom-right.
        let dot_radius = (w / 16).max(4);
        paint_disc(
            ctx.pixels,
            w,
            h,
            w as i32 - dot_radius as i32 * 3,
            h as i32 - dot_radius as i32 * 3,
            dot_radius as i32,
            state_color,
        );
    }
}

fn small_scale(w: usize) -> i32 {
    match w {
        0..=127 => 1,
        128..=255 => 2,
        _ => 3,
    }
}

fn large_scale(w: usize) -> i32 {
    match w {
        0..=127 => 2,
        128..=255 => 4,
        256..=383 => 5,
        _ => 6,
    }
}

fn paint_disc(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    color: [u8; 4],
) {
    let r2 = radius * radius;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            let idx = (y as usize * w + x as usize) * 4;
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
