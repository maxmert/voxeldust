//! Text readout — reads a `SignalValue::Text` string from the
//! registry and displays it. Text is server-authoritative (a
//! `SignalProperty::Text` / `SignalValue::Text` variant the server
//! publishes directly) — no client-side lookups.

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::font;
use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, WidgetKind};
use crate::hud::widget::{DrawCtx, HudWidget};

pub struct TextWidget;

impl HudWidget for TextWidget {
    fn kind(&self) -> WidgetKind {
        WidgetKind::Text
    }
    fn name(&self) -> &'static str {
        "Text"
    }
    fn supported_properties(&self) -> &'static [SignalProperty] {
        // `SignalProperty::Text` variant lands server-side as part of
        // the HUD protocol prerequisites; until then, any current
        // property is nominally accepted so registry/panel work.
        &[SignalProperty::Status]
    }
    fn draw(&self, ctx: DrawCtx, value: Option<SignalValue>, _config: &HudConfig) {
        let w = ctx.size as usize;
        let h = ctx.size as usize;
        let alpha = (255.0 * ctx.opacity) as u8;
        let caption_color = [180u8, 200, 220, alpha];
        let value_color = [235u8, 235, 245, alpha];
        let empty_color = [120u8, 130, 150, alpha];

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

        // Body text (value).
        let scale = body_scale(w);
        match value.as_ref().and_then(|v| v.as_text()) {
            Some(text) => {
                let th = font::GLYPH_H as i32 * scale;
                let lines = wrap_lines(text, w, scale);
                let total_h = lines.len() as i32 * th + (lines.len() as i32 - 1).max(0) * scale;
                let mut y = (h as i32 - total_h) / 2;
                for line in &lines {
                    let tw = font::text_width(line, scale);
                    font::draw_text(
                        ctx.pixels,
                        w,
                        h,
                        line,
                        (w as i32 - tw) / 2,
                        y,
                        scale,
                        value_color,
                    );
                    y += th + scale;
                }
            }
            None => {
                let label = "NO SIGNAL";
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
                    empty_color,
                );
            }
        }
    }
}

fn wrap_lines(text: &str, width_px: usize, scale: i32) -> Vec<String> {
    // 6 = 5 wide glyph + 1 advance; match font::ADVANCE.
    let max_cols = (width_px as i32 - 8) / (6 * scale);
    if max_cols <= 1 {
        return vec![text.chars().take(1).collect()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let trial_len = if current.is_empty() {
            word.len()
        } else {
            current.len() + 1 + word.len()
        };
        if (trial_len as i32) > max_cols && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn small_scale(w: usize) -> i32 {
    match w {
        0..=127 => 1,
        128..=255 => 2,
        _ => 3,
    }
}

fn body_scale(w: usize) -> i32 {
    match w {
        0..=127 => 2,
        128..=255 => 3,
        _ => 4,
    }
}
