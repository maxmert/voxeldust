//! Terminal widget — on-block chat / log display.
//!
//! Renders directly into the tile's pixel buffer (no separate render
//! target, no render-to-image camera). State is per-tile and lives
//! across engagements: scrollback survives disengage→re-engage so
//! the player doesn't lose context every time they look away.
//!
//! ## Server protocol bridge
//!
//! - `ServerMsg::OpenTerminalChat`: full snapshot at engagement —
//!   subscribe / publish channel names plus most-recent scrollback.
//!   Hydrates state for the matching tile.
//! - `ServerMsg::TerminalScrollbackDelta`: incremental append —
//!   pushes lines onto `scrollback`, capped by `MAX_SCROLLBACK`.
//!
//! Both messages are routed by `terminal_bridge.rs` (hud module
//! sibling) which finds the tile entity by `(block_pos, face)` and
//! mutates the `TerminalWidgetState` directly. The widget itself
//! stays passive — `draw` just reads state, `on_text` mutates input
//! buffer, `on_text(Enter)` returns `WidgetAction::SendChat`.
//!
//! ## Engagement gates writing
//!
//! Typing only mutates state when the tile is HUD-focused (the
//! `on_text` path is only invoked then). Without focus we still
//! paint scrollback so the terminal acts as an always-visible HUD
//! display — anyone walking past a Terminal sees the latest chat
//! lines on its face, exactly like a Star Citizen MFD.

use voxeldust_core::signal::types::SignalProperty;

use crate::hud::focus::HudClickButton;
use crate::hud::font;
use crate::hud::signal_registry::SignalValue;
use crate::hud::tile::{HudConfig, WidgetKind};
use crate::hud::widget::{
    DrawCtx, HudTextInput, HudWidget, HudWidgetStateData, WidgetAction,
};

/// Maximum scrollback lines retained on the client. Mirrors
/// `TerminalState::DEFAULT_MAX_LINES` on the server so an engagement
/// of any length doesn't grow client memory unbounded.
pub const MAX_SCROLLBACK: usize = 64;

/// Maximum length of one player-typed line. Server-side
/// `TerminalChatSendData.text` accepts up to 4 KiB; we cap lower
/// because the visible width is ~80 cols at our font scale and
/// longer entries would need horizontal scrolling we don't have.
pub const MAX_INPUT_LEN: usize = 256;

/// Per-tile state for `WidgetKind::Terminal`. Created via
/// `init_state` on the first redraw after the tile takes the kind;
/// mutated by `on_text` during input; mutated externally by the
/// scrollback bridge (`bridge::ingest_scrollback_*`).
#[derive(Debug, Default)]
pub struct TerminalWidgetState {
    /// Bounded ring of received chat lines, oldest at index 0.
    pub scrollback: Vec<String>,
    /// Active input buffer. Cleared on Enter (after submit).
    pub input: String,
    /// Most-recent subscribe channel name (purely for the header
    /// render — also set on engage so the widget can show "READ:
    /// shipchat" before any scrollback delta lands).
    pub subscribe_channel: Option<String>,
    /// Most-recent publish channel name (for the header + the
    /// "writeable?" gate of the input field).
    pub publish_channel: Option<String>,
    /// Block position of the Terminal block this state belongs to.
    /// Stored at hydrate time so `on_text(Enter)` can fill the
    /// `WidgetAction::SendChat` block_pos field without the widget
    /// needing access to the tile's `HudAttachment`. Set by
    /// `terminal_bridge::ingest_open_chat`.
    pub block_pos: bevy::prelude::IVec3,
}

pub struct TerminalWidget;

impl HudWidget for TerminalWidget {
    fn kind(&self) -> WidgetKind {
        WidgetKind::Terminal
    }
    fn name(&self) -> &'static str {
        "Terminal"
    }
    fn supported_properties(&self) -> &'static [SignalProperty] {
        // Terminal is text-only — `Status` carries text in the
        // current schema; the property dropdown filters to this.
        &[SignalProperty::Status]
    }
    fn is_interactive(&self) -> bool {
        true
    }
    fn init_state(
        &self,
        _config: &HudConfig,
    ) -> Option<Box<dyn HudWidgetStateData>> {
        Some(Box::new(TerminalWidgetState::default()))
    }

    fn draw(
        &self,
        ctx: DrawCtx,
        _value: Option<SignalValue>,
        state: Option<&mut dyn HudWidgetStateData>,
        _config: &HudConfig,
    ) {
        let w = ctx.size as usize;
        let h = ctx.size as usize;
        let alpha = (255.0 * ctx.opacity) as u8;
        let bg = [4u8, 8, 14, alpha];
        let bezel = [180u8, 140, 60, alpha];
        let bezel_dim = [70u8, 50, 25, alpha];
        let header_color = [220u8, 180, 80, alpha];
        let scrollback_color = [220u8, 180, 80, alpha];
        let prompt_color = [220u8, 180, 80, alpha];
        let chrome_color = [120u8, 130, 150, alpha];
        let dim_color = [80u8, 100, 120, alpha];

        // Background — fully opaque dark "screen" colour so this
        // reads as a CRT regardless of ambient lighting.
        fill(ctx.pixels, w, h, 0, 0, w, h, bg);

        // Bezel — 3-pixel amber border; subtle inner bezel.
        bezel_rect(ctx.pixels, w, h, bezel, 3);
        bezel_rect(ctx.pixels, w, h, bezel_dim, 5);

        // Pull state. Stateless render path (state == None) still
        // paints chrome + "no engagement" hint, so the panel reads
        // sensibly even when never engaged. The `init_state`
        // override means this only happens during the one-frame
        // gap before `sync_hud_widget_state` inserts the state
        // component.
        //
        // `Option::as_deref` reborrows the `&mut dyn HudWidgetStateData`
        // as `&dyn HudWidgetStateData` whose lifetime is tied to the
        // outer parameter (so the downcast result doesn't escape a
        // temporary). Using `Option::map` here would move the inner
        // reference into a closure, ending its lifetime at closure
        // exit and triggering E0515 / E0521.
        let state_ref: Option<&TerminalWidgetState> = state
            .as_deref()
            .and_then(|s| s.as_any().downcast_ref::<TerminalWidgetState>());

        // Per-section font scale. The terminal is denser than other
        // widgets — we want maximum scrollback rows visible — so we
        // hard-code rather than going through `pick_scale`. `font`
        // glyphs are 5×7; at our 256-px texture and 0.9 m face viewed
        // from ~1 m away, scale=2 (10×14 px ≈ 1.9° × 2.7°) is the
        // smallest that still reads easily as monospace; chrome /
        // channel-indicator labels drop to scale=1 (under 0.5° tall —
        // legible peripheral info, not primary content).
        let header_scale: i32 = 2;
        let body_scale: i32 = 2;
        let chrome_scale: i32 = 1;

        // Header band.
        let header_h = font::GLYPH_H as i32 * header_scale + 8;
        let header_label = "TERMINAL";
        font::draw_text(
            ctx.pixels,
            w,
            h,
            header_label,
            10,
            6,
            header_scale,
            header_color,
        );
        if let Some(s) = state_ref {
            let chans = format!(
                "R:{}  W:{}",
                s.subscribe_channel
                    .as_deref()
                    .unwrap_or("(none)"),
                s.publish_channel
                    .as_deref()
                    .unwrap_or("(none)"),
            );
            let tw = font::text_width(&chans, chrome_scale);
            font::draw_text(
                ctx.pixels,
                w,
                h,
                &chans,
                w as i32 - tw - 10,
                10,
                chrome_scale,
                chrome_color,
            );
        }
        // Header separator.
        draw_hline(ctx.pixels, w, 8, w - 8, header_h as usize + 4, bezel_dim);

        // Scrollback area.
        let line_h = font::GLYPH_H as i32 * body_scale + 3;
        let prompt_band_h = (font::GLYPH_H as i32 * body_scale + 12).max(24);
        let body_y0 = header_h + 10;
        let body_y1 = h as i32 - prompt_band_h - 6;
        let max_lines = ((body_y1 - body_y0) / line_h).max(1) as usize;

        // Render newest at the bottom (pin-to-bottom semantics).
        let lines: &[String] = state_ref.map(|s| s.scrollback.as_slice()).unwrap_or(&[]);
        if lines.is_empty() {
            let msg = "(awaiting transmissions)";
            let tw = font::text_width(msg, body_scale);
            font::draw_text(
                ctx.pixels,
                w,
                h,
                msg,
                (w as i32 - tw) / 2,
                body_y0 + (body_y1 - body_y0 - line_h) / 2,
                body_scale,
                dim_color,
            );
        } else {
            let start = lines.len().saturating_sub(max_lines);
            let visible = &lines[start..];
            let total_h = visible.len() as i32 * line_h;
            let mut y = body_y1 - total_h;
            for line in visible {
                let truncated = truncate_to_width(line, w as i32 - 24, body_scale);
                font::draw_text(
                    ctx.pixels,
                    w,
                    h,
                    &truncated,
                    12,
                    y,
                    body_scale,
                    scrollback_color,
                );
                y += line_h;
            }
        }

        // Prompt band separator.
        draw_hline(ctx.pixels, w, 8, w - 8, body_y1 as usize + 2, bezel_dim);

        // Input row — leading prompt arrow + the buffered input.
        let prompt_y = body_y1 + 8;
        let prompt = ">";
        font::draw_text(
            ctx.pixels,
            w,
            h,
            prompt,
            12,
            prompt_y,
            body_scale,
            prompt_color,
        );
        let prompt_w = font::text_width(prompt, body_scale);
        let input_x = 12 + prompt_w + (8 * body_scale);
        let input_buf: &str = state_ref.map(|s| s.input.as_str()).unwrap_or("");
        let writeable = state_ref
            .and_then(|s| s.publish_channel.as_deref())
            .map(|c| !c.is_empty())
            .unwrap_or(false);
        if input_buf.is_empty() {
            let hint = if writeable {
                "type — Enter to send"
            } else {
                "(read-only — set a publish channel via F)"
            };
            font::draw_text(ctx.pixels, w, h, hint, input_x, prompt_y, body_scale, dim_color);
        } else {
            let truncated = truncate_to_width(
                input_buf,
                w as i32 - input_x - 16,
                body_scale,
            );
            font::draw_text(
                ctx.pixels,
                w,
                h,
                &truncated,
                input_x,
                prompt_y,
                body_scale,
                prompt_color,
            );
            // Caret blinks at end of input. Position is past the
            // truncated text, clamped to the right margin.
            let tw = font::text_width(&truncated, body_scale);
            let caret_x = (input_x + tw).min(w as i32 - 14);
            // Hard-on caret for now (no time-based blink) — keeps
            // the redraw cadence cheap; could blink later with a
            // floor-cadence-driven phase.
            for dy in 0..(font::GLYPH_H as i32 * body_scale) {
                put_px(
                    ctx.pixels,
                    w,
                    caret_x as usize,
                    (prompt_y + dy) as usize,
                    prompt_color,
                );
            }
        }
    }

    fn on_click(
        &self,
        _uv: bevy::prelude::Vec2,
        _button: HudClickButton,
        _value: Option<&SignalValue>,
        _state: Option<&mut dyn HudWidgetStateData>,
        _config: &HudConfig,
    ) -> Option<WidgetAction> {
        // Click currently focuses the input (always focused while
        // engaged) — nothing to publish on click. A future
        // refinement could place the caret at the click UV.
        None
    }

    fn on_text(
        &self,
        event: HudTextInput,
        state: Option<&mut dyn HudWidgetStateData>,
        _config: &HudConfig,
    ) -> Option<WidgetAction> {
        let state = state.and_then(|s| {
            let any: &mut dyn std::any::Any = s.as_any_mut();
            any.downcast_mut::<TerminalWidgetState>()
        })?;
        match event {
            HudTextInput::Char(c) => {
                if state.input.len() + c.len_utf8() <= MAX_INPUT_LEN {
                    state.input.push(c);
                }
                None
            }
            HudTextInput::Backspace => {
                state.input.pop();
                None
            }
            HudTextInput::Enter => {
                let text = std::mem::take(&mut state.input);
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return None;
                }
                let publish = state.publish_channel.as_deref().unwrap_or("");
                if publish.is_empty() {
                    // Read-only terminal — drop submission silently;
                    // the input field already showed the read-only
                    // hint.
                    return None;
                }
                Some(WidgetAction::SendChat {
                    block_pos: state.block_pos,
                    text: trimmed.to_string(),
                })
            }
            // Esc / Tab: don't capture — let the focus pipeline
            // apply default behaviour (Esc = drop focus).
            HudTextInput::Escape | HudTextInput::Tab => None,
            // Movement keys: caret positioning isn't implemented —
            // ignore for now. Widgets that want them implement
            // them; we don't synthesise behaviour the user can't
            // see.
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel-buffer helpers
// ---------------------------------------------------------------------------

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

fn bezel_rect(pixels: &mut [u8], w: usize, h: usize, color: [u8; 4], thickness: usize) {
    for y in 0..h {
        for x in 0..w {
            let on_border = x < thickness
                || x + thickness >= w
                || y < thickness
                || y + thickness >= h;
            if on_border {
                put_px(pixels, w, x, y, color);
            }
        }
    }
}

fn truncate_to_width(s: &str, width_px: i32, scale: i32) -> String {
    // Per-glyph advance in pixels, derived from the font helper so
    // we don't lock in private font internals.
    let advance = font::text_width("X", scale).max(1);
    let max_chars = (width_px / advance).max(1) as usize;
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let cut = max_chars.saturating_sub(1);
    let mut out: String = s.chars().take(cut).collect();
    out.push_str("…");
    out
}
