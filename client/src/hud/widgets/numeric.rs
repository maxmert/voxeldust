//! Numeric readout: renders the signal value as text centered on the
//! tile, large-scale. Handles float-valued properties and clamps text
//! length so the label always fits.

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::font;
use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, WidgetKind};
use crate::hud::widget::{DrawCtx, HudWidget};

pub struct NumericWidget;

impl HudWidget for NumericWidget {
    fn kind(&self) -> WidgetKind {
        WidgetKind::Numeric
    }
    fn name(&self) -> &'static str {
        "Numeric"
    }
    fn supported_properties(&self) -> &'static [SignalProperty] {
        &[
            SignalProperty::Speed,
            SignalProperty::Pressure,
            SignalProperty::Angle,
            SignalProperty::Throttle,
            SignalProperty::Level,
            SignalProperty::Extension,
            SignalProperty::Boost,
        ]
    }
    fn draw(&self, ctx: DrawCtx, value: Option<SignalValue>, config: &HudConfig) {
        let w = ctx.size as usize;
        let h = ctx.size as usize;
        let alpha = (255.0 * ctx.opacity) as u8;
        let caption_color = [180u8, 200, 220, alpha];
        let value_color = [240u8, 240, 245, alpha];

        // Caption on top.
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

        // Large numeric readout centered.
        let text = format_value(value.as_ref(), config.property);
        let scale = large_scale(w);
        let tw = font::text_width(&text, scale);
        let th = font::GLYPH_H as i32 * scale;
        font::draw_text(
            ctx.pixels,
            w,
            h,
            &text,
            (w as i32 - tw) / 2,
            (h as i32 - th) / 2,
            scale,
            value_color,
        );
    }
}

fn format_value(v: Option<&SignalValue>, property: SignalProperty) -> String {
    let Some(v) = v else {
        return "-".to_string();
    };
    if let Some(s) = v.as_text() {
        return s.chars().take(10).collect();
    }
    let f = v.as_f32();
    match property {
        SignalProperty::Speed => format!("{:.0}", f),
        SignalProperty::Angle => format!("{:.0}", f),
        SignalProperty::Pressure => format!("{:.1}", f),
        _ => {
            // 0-1 normalised properties — display as percent.
            let pct = (f.clamp(0.0, 1.0) * 100.0).round() as i32;
            format!("{}%", pct)
        }
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
