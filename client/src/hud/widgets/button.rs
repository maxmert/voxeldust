//! Publisher button widget. Renders a large button face; clicking it
//! (in HUD focus mode) publishes `1.0` to the configured channel.
//! Value encoded as `SignalValue::Float` so it pairs with
//! `SignalProperty::Throttle` / `Active` channels. A future iteration
//! threads press + release distinct values.

use voxeldust_core::signal::types::{SignalProperty, SignalValue as CoreSignalValue};

use crate::hud::focus::HudClickButton;
use crate::hud::font;
use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, WidgetKind};
use crate::hud::widget::{ClickAction, DrawCtx, HudWidget};

pub struct ButtonWidget;

impl HudWidget for ButtonWidget {
    fn kind(&self) -> WidgetKind {
        WidgetKind::Button
    }
    fn name(&self) -> &'static str {
        "Button"
    }
    fn supported_properties(&self) -> &'static [SignalProperty] {
        &[SignalProperty::Active, SignalProperty::Throttle]
    }
    fn draw(&self, ctx: DrawCtx, value: Option<SignalValue>, _config: &HudConfig) {
        let w = ctx.size as usize;
        let h = ctx.size as usize;
        let alpha = (255.0 * ctx.opacity) as u8;

        let pressed = value.map(|v| v.as_bool()).unwrap_or(false);
        let bg = if pressed {
            [20u8, 120, 60, alpha]
        } else {
            [40u8, 70, 120, alpha]
        };
        let border = if pressed {
            [120u8, 255, 180, alpha]
        } else {
            [180u8, 220, 255, alpha]
        };
        let caption_color = [235u8, 240, 245, alpha];

        // Flood fill the button face.
        for y in 0..h {
            for x in 0..w {
                put_px(ctx.pixels, w, x, y, bg);
            }
        }
        // 4-pixel bezel.
        for y in 0..h {
            for x in 0..w {
                let on_border = x < 4 || x + 4 >= w || y < 4 || y + 4 >= h;
                if on_border {
                    put_px(ctx.pixels, w, x, y, border);
                }
            }
        }

        // Label centered. Uppercase the caption if any; fall back to
        // the channel name so players can tell which button they're
        // looking at even before they set a caption.
        let label_src = if !ctx.caption.is_empty() {
            ctx.caption
        } else {
            "PUBLISH"
        };
        let label: String = label_src.to_ascii_uppercase();
        let scale = button_scale(w);
        let tw = font::text_width(&label, scale);
        let th = font::GLYPH_H as i32 * scale;
        font::draw_text(
            ctx.pixels,
            w,
            h,
            &label,
            (w as i32 - tw) / 2,
            (h as i32 - th) / 2,
            scale,
            caption_color,
        );
    }

    fn on_click(
        &self,
        _uv: bevy::prelude::Vec2,
        button: HudClickButton,
        _value: Option<&SignalValue>,
        config: &HudConfig,
    ) -> Option<ClickAction> {
        if config.channel.is_empty() {
            return None;
        }
        // LMB publishes 1.0 (press). RMB publishes 0.0 (release /
        // reset). Middle-click ignored.
        let value = match button {
            HudClickButton::Left => CoreSignalValue::Float(1.0),
            HudClickButton::Right => CoreSignalValue::Float(0.0),
            HudClickButton::Middle => return None,
        };
        Some(ClickAction::Publish {
            channel: config.channel.clone(),
            value,
        })
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

fn button_scale(w: usize) -> i32 {
    match w {
        0..=127 => 2,
        128..=255 => 4,
        256..=383 => 5,
        _ => 6,
    }
}
