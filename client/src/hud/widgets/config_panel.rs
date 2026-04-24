//! In-world config panel widget — the tablet surface the held tablet
//! summons when the player opens a block config. Renders a dark
//! cockpit-display styled panel with four sections (PUBLISH,
//! SUBSCRIBE, CONVERT, SEAT KEYS), each section listing its configured
//! bindings by channel + property.
//!
//! Fully server-authoritative: every rendered value comes from the
//! `BlockSignalConfig` payload delivered via `HudPayload::ConfigPanel`.
//! The client only displays; editing lands with H7 v2 (in-place
//! row click → selector popup).

use voxeldust_core::signal::config::BlockSignalConfig;
use voxeldust_core::signal::types::SignalProperty;

use crate::hud::font;
use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, HudPayload, WidgetKind};
use crate::hud::widget::{DrawCtx, HudWidget};

pub struct ConfigPanelWidget;

impl HudWidget for ConfigPanelWidget {
    fn kind(&self) -> WidgetKind {
        WidgetKind::ConfigPanel
    }
    fn name(&self) -> &'static str {
        "Config Panel"
    }
    fn supported_properties(&self) -> &'static [SignalProperty] {
        &[]
    }
    fn draw(&self, mut ctx: DrawCtx, _value: Option<SignalValue>, config: &HudConfig) {
        let w = ctx.size as usize;
        let h = ctx.size as usize;
        let alpha = 255u8;

        let bg = [10u8, 16, 24, alpha];
        let bg_row_alt = [14u8, 22, 34, alpha];
        let accent = [80u8, 200, 255, alpha];
        let accent_dim = [40u8, 120, 170, alpha];
        let label_color = [200u8, 220, 235, alpha];
        let value_color = [235u8, 245, 250, alpha];
        let empty_color = [90u8, 110, 130, alpha];

        // Panel background.
        fill(ctx.pixels, w, h, 0, 0, w, h, bg);

        // Accent bezel (3-pixel).
        for y in 0..h {
            for x in 0..w {
                let on_border = x < 3 || x + 3 >= w || y < 3 || y + 3 >= h;
                if on_border {
                    put_px(ctx.pixels, w, x, y, accent);
                }
            }
        }

        // Corner brackets — the Star Citizen-style angle brackets at
        // each corner to give the panel a cockpit look.
        paint_corner_brackets(ctx.pixels, w, h, 22, 4, accent);

        // Title band.
        let band_h = (h as f32 * 0.10).max(24.0) as usize;
        let title_scale = title_scale(w);
        let title = if !ctx.caption.is_empty() {
            ctx.caption.to_string()
        } else {
            "CONFIG PANEL".to_string()
        };
        let th = font::GLYPH_H as i32 * title_scale;
        font::draw_text(
            ctx.pixels,
            w,
            h,
            &title.to_ascii_uppercase(),
            16,
            (band_h as i32 - th) / 2,
            title_scale,
            accent,
        );
        // Separator under title.
        draw_hline(ctx.pixels, w, 6, w - 6, band_h, accent);
        // Dim sub-separator just under.
        draw_hline(ctx.pixels, w, 6, w - 6, band_h + 2, accent_dim);

        // Section layout — four rows with labeled sections.
        let Some(cfg) = extract_cfg(&config.payload) else {
            paint_empty_hint(ctx.pixels, w, h, band_h, empty_color, label_color);
            return;
        };

        // Detect an all-zero default config (freshly-summoned tablet
        // without a server-side `BlockConfigState` response yet) and
        // show an explanatory hint INSTEAD of empty rows, so the
        // player isn't left guessing whether anything is wrong.
        let is_default = cfg.block_pos.x == 0
            && cfg.block_pos.y == 0
            && cfg.block_pos.z == 0
            && cfg.publish_bindings.is_empty()
            && cfg.subscribe_bindings.is_empty()
            && cfg.converter_rules.is_empty()
            && cfg.seat_mappings.is_empty();
        if is_default {
            paint_empty_hint(ctx.pixels, w, h, band_h, empty_color, label_color);
            return;
        }

        let row_scale = row_scale(w);
        let section_top = band_h as i32 + 10;
        let section_h = (h as i32 - section_top - 20) / 4;

        let sections: [(&str, Vec<SectionRow>, [u8; 4]); 4] = [
            ("PUBLISH", publish_rows(cfg), accent),
            ("SUBSCRIBE", subscribe_rows(cfg), accent),
            ("CONVERT", convert_rows(cfg), accent),
            ("SEAT KEYS", seat_rows(cfg), accent),
        ];

        for (i, (label, rows, section_accent)) in sections.iter().enumerate() {
            let y0 = section_top + (i as i32) * section_h;
            paint_section(
                ctx.pixels,
                w,
                h,
                y0,
                section_h,
                label,
                rows,
                *section_accent,
                accent_dim,
                label_color,
                value_color,
                empty_color,
                row_scale,
                bg_row_alt,
            );
        }

        // Bottom footer hint.
        let footer = "TAB: FOCUS  LMB: CLICK  F: CLOSE";
        let sc = 1;
        let tw = font::text_width(footer, sc);
        font::draw_text(
            ctx.pixels,
            w,
            h,
            footer,
            (w as i32 - tw) / 2,
            h as i32 - (font::GLYPH_H as i32 * sc) - 6,
            sc,
            empty_color,
        );
    }
}

fn extract_cfg(payload: &HudPayload) -> Option<&BlockSignalConfig> {
    match payload {
        HudPayload::ConfigPanel(cfg) => Some(cfg.as_ref()),
        _ => None,
    }
}

/// Paint a centered "no block selected" hint when the tablet is
/// summoned without a matching `BlockConfigState` (i.e., F was
/// pressed in empty space or the server hasn't replied yet).
fn paint_empty_hint(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    band_h: usize,
    empty_color: [u8; 4],
    label_color: [u8; 4],
) {
    let lines = [
        ("NO BLOCK SELECTED", true),
        ("", false),
        ("POINT AT A FUNCTIONAL BLOCK", false),
        ("AND PRESS F TO LOAD ITS CONFIG.", false),
        ("", false),
        ("TAB: ENTER CURSOR MODE", false),
        ("F: DISMISS TABLET", false),
    ];
    let sc_big = 3.min(scale_for_width(w, 3)).max(1);
    let sc_sm = 2.min(scale_for_width(w, 2)).max(1);
    let total_h = lines
        .iter()
        .map(|(_, big)| {
            (font::GLYPH_H as i32) * if *big { sc_big } else { sc_sm } + sc_big * 3
        })
        .sum::<i32>();
    let mut y = (band_h as i32 + (h as i32 - band_h as i32 - total_h) / 2).max(band_h as i32 + 8);
    for (line, big) in lines.iter() {
        let sc = if *big { sc_big } else { sc_sm };
        let color = if *big { label_color } else { empty_color };
        let tw = font::text_width(line, sc);
        if !line.is_empty() {
            font::draw_text(pixels, w, h, line, (w as i32 - tw) / 2, y, sc, color);
        }
        y += (font::GLYPH_H as i32) * sc + sc_big * 3;
    }
}

fn scale_for_width(w: usize, desired: i32) -> i32 {
    // Small tiles can't fit 3x-scale text; fall back to 2x or 1x.
    let max_fit_for_50_chars = (w as i32 - 20) / (50 * 6);
    desired.min(max_fit_for_50_chars.max(1))
}

struct SectionRow {
    lhs: String,
    rhs: String,
}

fn publish_rows(cfg: &BlockSignalConfig) -> Vec<SectionRow> {
    cfg.publish_bindings
        .iter()
        .map(|b| SectionRow {
            lhs: b.channel_name.clone(),
            rhs: format!("{:?}", b.property).to_ascii_uppercase(),
        })
        .collect()
}

fn subscribe_rows(cfg: &BlockSignalConfig) -> Vec<SectionRow> {
    cfg.subscribe_bindings
        .iter()
        .map(|b| SectionRow {
            lhs: b.channel_name.clone(),
            rhs: format!("{:?}", b.property).to_ascii_uppercase(),
        })
        .collect()
}

fn convert_rows(cfg: &BlockSignalConfig) -> Vec<SectionRow> {
    cfg.converter_rules
        .iter()
        .map(|r| SectionRow {
            lhs: format!("{}  \u{2192}  {}", r.input_channel, r.output_channel),
            rhs: format!("{:?}", r.expression).to_ascii_uppercase(),
        })
        .collect()
}

fn seat_rows(cfg: &BlockSignalConfig) -> Vec<SectionRow> {
    cfg.seat_mappings
        .iter()
        .map(|s| {
            let key = if !s.label.is_empty() {
                s.label.clone()
            } else {
                format!("{:?}/{}", s.source, s.key_name)
            };
            SectionRow {
                lhs: key,
                rhs: s.channel_name.clone(),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn paint_section(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    y0: i32,
    section_h: i32,
    label: &str,
    rows: &[SectionRow],
    section_accent: [u8; 4],
    dim_accent: [u8; 4],
    label_color: [u8; 4],
    value_color: [u8; 4],
    empty_color: [u8; 4],
    row_scale: i32,
    bg_alt: [u8; 4],
) {
    // Section header — left-aligned label, count on the right.
    let header_scale = row_scale.max(2);
    let header_y = y0 + 2;
    font::draw_text(
        pixels,
        w,
        h,
        label,
        16,
        header_y,
        header_scale,
        section_accent,
    );
    let count_text = format!("{}", rows.len());
    let cw = font::text_width(&count_text, header_scale);
    font::draw_text(
        pixels,
        w,
        h,
        &count_text,
        w as i32 - cw - 16,
        header_y,
        header_scale,
        section_accent,
    );
    let header_h = font::GLYPH_H as i32 * header_scale + 4;
    draw_hline(
        pixels,
        w,
        14,
        w - 14,
        (header_y + header_h) as usize,
        dim_accent,
    );

    // Rows (up to 3 per section, truncated by space).
    let row_area_y0 = header_y + header_h + 2;
    let row_h = font::GLYPH_H as i32 * row_scale + 6;
    let max_rows = ((section_h - header_h - 4) / row_h).max(1) as usize;
    if rows.is_empty() {
        let msg = "—";
        let sc = row_scale;
        font::draw_text(pixels, w, h, msg, 20, row_area_y0, sc, empty_color);
        return;
    }
    for (i, row) in rows.iter().take(max_rows).enumerate() {
        let ry = row_area_y0 + (i as i32) * row_h;
        // Alternating row background stripe.
        if i & 1 == 1 {
            fill(
                pixels,
                w,
                h,
                12,
                ry as usize,
                w.saturating_sub(12),
                (ry + row_h) as usize,
                bg_alt,
            );
        }
        // lhs (channel / mapping).
        let lhs = truncate(&row.lhs, max_cols_for_width((w as i32 - 32) / 2, row_scale));
        font::draw_text(pixels, w, h, &lhs, 20, ry + 2, row_scale, label_color);
        // rhs (property / expression) right-aligned.
        let rhs = truncate(&row.rhs, max_cols_for_width(w as i32 / 3, row_scale));
        let rw = font::text_width(&rhs, row_scale);
        font::draw_text(
            pixels,
            w,
            h,
            &rhs,
            w as i32 - rw - 20,
            ry + 2,
            row_scale,
            value_color,
        );
    }
    // "+N MORE" indicator when truncated.
    if rows.len() > max_rows {
        let msg = format!("+{} MORE", rows.len() - max_rows);
        let sc = 1;
        let tw = font::text_width(&msg, sc);
        font::draw_text(
            pixels,
            w,
            h,
            &msg,
            w as i32 - tw - 20,
            row_area_y0 + (max_rows as i32) * row_h - font::GLYPH_H as i32,
            sc,
            empty_color,
        );
    }
}

fn truncate(s: &str, max_cols: i32) -> String {
    let mx = max_cols.max(1) as usize;
    s.chars().take(mx).collect()
}

fn max_cols_for_width(width_px: i32, scale: i32) -> i32 {
    (width_px / (6 * scale)).max(1)
}

fn title_scale(w: usize) -> i32 {
    match w {
        0..=191 => 2,
        192..=383 => 3,
        _ => 4,
    }
}

fn row_scale(w: usize) -> i32 {
    match w {
        0..=191 => 1,
        192..=383 => 2,
        _ => 3,
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

fn fill(
    pixels: &mut [u8],
    w: usize,
    _h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color: [u8; 4],
) {
    for y in y0..y1 {
        for x in x0..x1 {
            put_px(pixels, w, x, y, color);
        }
    }
}

fn draw_hline(pixels: &mut [u8], w: usize, x0: usize, x1: usize, y: usize, color: [u8; 4]) {
    for x in x0..x1 {
        put_px(pixels, w, x, y, color);
    }
}

/// Paint corner brackets of `size` pixels long, `thickness` thick, in
/// the given color. Classic cockpit-display aesthetic.
fn paint_corner_brackets(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    size: usize,
    thickness: usize,
    color: [u8; 4],
) {
    // Top-left.
    fill(pixels, w, h, 0, 0, size, thickness, color);
    fill(pixels, w, h, 0, 0, thickness, size, color);
    // Top-right.
    fill(pixels, w, h, w.saturating_sub(size), 0, w, thickness, color);
    fill(pixels, w, h, w.saturating_sub(thickness), 0, w, size, color);
    // Bottom-left.
    fill(pixels, w, h, 0, h.saturating_sub(thickness), size, h, color);
    fill(pixels, w, h, 0, h.saturating_sub(size), thickness, h, color);
    // Bottom-right.
    fill(
        pixels,
        w,
        h,
        w.saturating_sub(size),
        h.saturating_sub(thickness),
        w,
        h,
        color,
    );
    fill(
        pixels,
        w,
        h,
        w.saturating_sub(thickness),
        h.saturating_sub(size),
        w,
        h,
        color,
    );
}
