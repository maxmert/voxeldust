//! Linear gauge: draws a horizontal bar at the bottom of the tile
//! filled to the value's 0..1 extent. Caption on top. Transparent
//! background. Suitable for `Throttle`, `Level`, `Extension`,
//! `Pressure` (treated as normalized 0..1 unless server publishes
//! min/max — future iteration).

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::font;
use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, WidgetKind};
use crate::hud::widget::{DrawCtx, HudWidget};

pub struct GaugeWidget;

impl HudWidget for GaugeWidget {
    fn kind(&self) -> WidgetKind {
        WidgetKind::Gauge
    }
    fn name(&self) -> &'static str {
        "Gauge"
    }
    fn supported_properties(&self) -> &'static [SignalProperty] {
        &[
            SignalProperty::Throttle,
            SignalProperty::Level,
            SignalProperty::Extension,
            SignalProperty::Pressure,
            SignalProperty::Boost,
        ]
    }
    fn draw(&self, mut ctx: DrawCtx, value: Option<SignalValue>, _config: &HudConfig) {
        let v = value.map(|v| v.as_f32().clamp(0.0, 1.0)).unwrap_or(0.0);
        let w = ctx.size as usize;
        let h = ctx.size as usize;
        let alpha_fg = (255.0 * ctx.opacity) as u8;
        let alpha_bg = (128.0 * ctx.opacity) as u8;
        let fg = [60u8, 220, 80, alpha_fg];
        let bg = [30u8, 30, 40, alpha_bg];
        let text_color = [220u8, 235, 245, alpha_fg];

        // Caption band (top).
        if !ctx.caption.is_empty() {
            let scale = pick_scale(w);
            font::draw_text(
                ctx.pixels,
                w,
                h,
                ctx.caption,
                (scale * 2).max(4),
                (scale * 2).max(4),
                scale,
                text_color,
            );
        }

        // Horizontal bar: bottom 20 % of tile, x ∈ [0, v].
        let bar_y0 = (h as f32 * 0.70) as usize;
        let bar_y1 = (h as f32 * 0.90) as usize;
        let fill_x = (w as f32 * v) as usize;
        for y in bar_y0..bar_y1 {
            for x in 0..w {
                let color = if x < fill_x { fg } else { bg };
                put_px(&mut ctx.pixels, w, x, y, color);
            }
        }

        // Numeric readout underneath (percentage).
        let pct = (v * 100.0).round() as i32;
        let label = format!("{}%", pct);
        let scale = pick_scale(w);
        let tw = font::text_width(&label, scale);
        font::draw_text(
            ctx.pixels,
            w,
            h,
            &label,
            (w as i32 - tw) / 2,
            bar_y1 as i32 + scale * 2,
            scale,
            text_color,
        );
    }
}

/// Pick an integer scale so glyphs read well against the tile
/// resolution without a text-file lookup. Empirically: scale 2 for
/// 128 px, scale 3 for 256, scale 4 for 384+, scale 5 for 512+.
fn pick_scale(width_px: usize) -> i32 {
    match width_px {
        0..=127 => 1,
        128..=191 => 2,
        192..=319 => 3,
        320..=447 => 4,
        _ => 5,
    }
}

fn put_px(pixels: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 4]) {
    let idx = (y * width + x) * 4;
    if idx + 3 >= pixels.len() {
        return;
    }
    pixels[idx] = color[0];
    pixels[idx + 1] = color[1];
    pixels[idx + 2] = color[2];
    pixels[idx + 3] = color[3];
}
