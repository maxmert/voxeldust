//! Cockpit HUD overlay — Star Citizen-style screen-space indicators
//! drawn every frame via egui. Reads exclusively from
//! `SignalRegistry` + `PrimaryWorldState` / `SecondaryWorldStates` so
//! everything displayed is server-authoritative.
//!
//! Per-frame smoothing: signal values update at the server tick rate
//! (20 Hz) but the HUD renders at 60+ FPS. The `SmoothedCockpitSignals`
//! resource exponential-filters the visible speed / thrust / altitude
//! toward their raw values with a short (~80 ms) time constant — fast
//! enough to feel instant, slow enough that the display doesn't
//! visibly step between ticks.
//!
//! Layers:
//!   * **Crosshair + compass ring** (top center): heading ticks every
//!     30° with N/E/S/W labels; yaw relative to ship forward.
//!   * **Flight telemetry** (bottom center): speed, thrust tier, warp
//!     ETA when active. All read from `ship.*` channels.
//!   * **Target reticle** (when nearest hostile entity within range):
//!     range + bearing.
//!
//! All overlays respect `InputMode::UiPanel` — when the config panel
//! or held tablet is focused (`HudFocusState.active == true`) the
//! cockpit HUD dims out so it doesn't compete with the focused tile.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

use crate::hud::focus::HudFocusState;
use crate::hud::signal_registry::SignalRegistry;
use crate::net::NetConnection;
use crate::shard::PrimaryWorldState;

/// Ordering marker for cockpit HUD overlay systems.
#[derive(SystemSet, Clone, Debug, Hash, Eq, PartialEq)]
pub struct CockpitHudSet;

pub struct CockpitHudPlugin;

impl Plugin for CockpitHudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SmoothedCockpitSignals>()
            .configure_sets(EguiPrimaryContextPass, CockpitHudSet)
            .add_systems(Update, smooth_cockpit_signals)
            .add_systems(
                EguiPrimaryContextPass,
                draw_cockpit_hud.in_set(CockpitHudSet),
            );
    }
}

/// Interpolated display values tracked per-frame. Updated by
/// `smooth_cockpit_signals` each frame and read by `draw_cockpit_hud`.
#[derive(Resource, Default, Debug, Clone)]
pub struct SmoothedCockpitSignals {
    pub speed: f32,
    pub thrust_tier: f32,
    pub heading_deg: f32,
    pub altitude: Option<f32>,
    pub autopilot_eta: f32,
}

/// Exponential-filter the raw registry values toward their live
/// counterparts. `HALF_LIFE_SECS` controls the aggressiveness — shorter
/// = snappier (more reactive), longer = smoother (more filtered).
/// 80 ms balances responsiveness with butter-smooth motion at 60 FPS.
fn smooth_cockpit_signals(
    time: Res<Time>,
    signals: Res<SignalRegistry>,
    mut smoothed: ResMut<SmoothedCockpitSignals>,
) {
    const HALF_LIFE_SECS: f32 = 0.08;
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let alpha = 1.0 - (-dt * (std::f32::consts::LN_2 / HALF_LIFE_SECS)).exp();

    let raw_speed = signals
        .get("ship.speed")
        .map(|r| r.value.as_f32())
        .unwrap_or(0.0);
    smoothed.speed += (raw_speed - smoothed.speed) * alpha;

    let raw_thrust = signals
        .get("ship.thrust_tier")
        .map(|r| r.value.as_f32())
        .unwrap_or(0.0);
    smoothed.thrust_tier += (raw_thrust - smoothed.thrust_tier) * alpha;

    let raw_heading = signals
        .get("ship.heading_deg")
        .map(|r| r.value.as_f32())
        .unwrap_or(0.0);
    // Wrap-around-aware lerp so heading doesn't take the long way around
    // 360 ↔ 0 transitions.
    let delta = (raw_heading - smoothed.heading_deg + 540.0).rem_euclid(360.0) - 180.0;
    smoothed.heading_deg = (smoothed.heading_deg + delta * alpha).rem_euclid(360.0);

    let raw_alt = signals.get("ship.altitude").map(|r| r.value.as_f32());
    smoothed.altitude = match (smoothed.altitude, raw_alt) {
        (_, None) => None,
        (None, Some(v)) => Some(v),
        (Some(prev), Some(v)) => Some(prev + (v - prev) * alpha),
    };

    let raw_eta = signals
        .get("ship.autopilot.eta")
        .map(|r| r.value.as_f32())
        .unwrap_or(0.0);
    smoothed.autopilot_eta += (raw_eta - smoothed.autopilot_eta) * alpha;
}

/// Colours tuned for dark space scenes — high contrast, slightly
/// desaturated teal/amber palette matching the tablet accent.
const COLOR_PRIMARY: egui::Color32 = egui::Color32::from_rgb(120, 220, 255);
const COLOR_DIM: egui::Color32 = egui::Color32::from_rgb(180, 210, 230);
const COLOR_WARN: egui::Color32 = egui::Color32::from_rgb(255, 180, 60);
const COLOR_HOSTILE: egui::Color32 = egui::Color32::from_rgb(255, 90, 90);

fn draw_cockpit_hud(
    mut contexts: EguiContexts,
    signals: Res<SignalRegistry>,
    smoothed: Res<SmoothedCockpitSignals>,
    focus: Res<HudFocusState>,
    conn: Res<NetConnection>,
    primary_ws: Res<PrimaryWorldState>,
) -> Result {
    // Don't draw until connected — the cockpit HUD is ship-contextual.
    if !conn.connected {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;

    // Dim the overlay to ~40% when a tile has focus so it doesn't
    // compete with the focused panel.
    let base_alpha = if focus.active { 100 } else { 220 };

    // Transparent, borderless fullscreen panel so we paint anywhere.
    egui::Area::new(egui::Id::new("cockpit_hud_canvas"))
        .fixed_pos(egui::Pos2::ZERO)
        .order(egui::Order::Background)
        .show(ctx, |ui| {
            let rect = ctx.screen_rect();
            let painter = ui.painter_at(rect);
            paint_crosshair(&painter, rect, base_alpha);
            paint_compass_band(&painter, rect, &smoothed, base_alpha);
            paint_flight_telemetry(&painter, rect, &signals, &smoothed, base_alpha);
            paint_warp_panel(&painter, rect, &signals, &smoothed, base_alpha);
            paint_connection_breadcrumb(
                &painter,
                rect,
                &conn,
                primary_ws.latest.as_ref(),
                base_alpha,
            );
        });
    Ok(())
}

fn with_alpha(c: egui::Color32, a: u8) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

fn paint_crosshair(painter: &egui::Painter, rect: egui::Rect, alpha: u8) {
    let c = rect.center();
    let color = with_alpha(COLOR_PRIMARY, alpha);
    let stroke = egui::Stroke::new(1.5, color);

    // Four ticks forming a gap-at-center reticle.
    for (a, b) in [
        (egui::vec2(-12.0, 0.0), egui::vec2(-4.0, 0.0)),
        (egui::vec2(4.0, 0.0), egui::vec2(12.0, 0.0)),
        (egui::vec2(0.0, -12.0), egui::vec2(0.0, -4.0)),
        (egui::vec2(0.0, 4.0), egui::vec2(0.0, 12.0)),
    ] {
        painter.line_segment([c + a, c + b], stroke);
    }

    // Center pinpoint dot.
    painter.circle_filled(c, 1.5, color);
}

fn paint_compass_band(
    painter: &egui::Painter,
    rect: egui::Rect,
    smoothed: &SmoothedCockpitSignals,
    alpha: u8,
) {
    // Heading derived from `ship.heading_deg` (0-360, 0=N). The
    // smoothed value handles wrap-around lerping so the compass band
    // doesn't sweep the long way around 360 ↔ 0.
    let heading = smoothed.heading_deg;

    let band_w = 360.0_f32.min(rect.width() * 0.75);
    let band_x0 = rect.center().x - band_w * 0.5;
    let band_y = rect.top() + 28.0;
    let color = with_alpha(COLOR_PRIMARY, alpha);
    let dim = with_alpha(COLOR_DIM, alpha.saturating_sub(60));

    // Tick every 10° over a ±45° window centered on current heading.
    let window_deg = 90.0_f32;
    for tick in (-90_i32..=90).step_by(10) {
        let deg = (heading + tick as f32).rem_euclid(360.0);
        let u = (tick as f32 + window_deg * 0.5) / window_deg;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let x = band_x0 + u * band_w;
        let major = (deg as i32) % 30 == 0;
        let (h, c) = if major { (10.0, color) } else { (5.0, dim) };
        painter.line_segment(
            [egui::pos2(x, band_y - h * 0.5), egui::pos2(x, band_y + h * 0.5)],
            egui::Stroke::new(1.5, c),
        );
        if major {
            let label = match deg.round() as i32 {
                0 => "N".to_string(),
                90 => "E".to_string(),
                180 => "S".to_string(),
                270 => "W".to_string(),
                d => format!("{:03}", d),
            };
            painter.text(
                egui::pos2(x, band_y + 14.0),
                egui::Align2::CENTER_TOP,
                label,
                egui::FontId::monospace(10.0),
                color,
            );
        }
    }
    // Center caret ▼ marking ownship heading.
    let caret = egui::pos2(rect.center().x, band_y - 10.0);
    painter.text(
        caret,
        egui::Align2::CENTER_BOTTOM,
        "▼",
        egui::FontId::monospace(12.0),
        color,
    );
}

fn paint_flight_telemetry(
    painter: &egui::Painter,
    rect: egui::Rect,
    signals: &SignalRegistry,
    smoothed: &SmoothedCockpitSignals,
    alpha: u8,
) {
    let color = with_alpha(COLOR_PRIMARY, alpha);
    let dim = with_alpha(COLOR_DIM, alpha.saturating_sub(40));

    // Numeric telemetry uses the smoothed values so the readout is
    // butter-smooth between server ticks. Callsign is a string signal
    // — no smoothing needed (no interpolation makes sense on text).
    let speed = smoothed.speed;
    let thrust_tier = smoothed.thrust_tier;
    let altitude = smoothed.altitude;
    let callsign = signals
        .get("ship.callsign")
        .and_then(|r| r.value.as_text().map(str::to_owned))
        .unwrap_or_default();

    let anchor = egui::pos2(rect.center().x, rect.bottom() - 80.0);

    // Speed readout — big bold centered number.
    painter.text(
        anchor + egui::vec2(-140.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        format!("{:.0}", speed),
        egui::FontId::monospace(42.0),
        color,
    );
    painter.text(
        anchor + egui::vec2(-140.0, 24.0),
        egui::Align2::RIGHT_TOP,
        "M/S",
        egui::FontId::monospace(10.0),
        dim,
    );

    // Thrust tier segmented bar — 5 segments, fill = tier * 5.
    let bar_left = anchor.x - 100.0;
    let bar_right = anchor.x + 100.0;
    let bar_y = anchor.y + 10.0;
    let seg_w = (bar_right - bar_left) / 5.0 - 4.0;
    for i in 0..5 {
        let x0 = bar_left + (i as f32) * (seg_w + 4.0);
        let filled = thrust_tier * 5.0 > (i as f32) + 0.5;
        let seg = egui::Rect::from_min_size(
            egui::pos2(x0, bar_y),
            egui::vec2(seg_w, 10.0),
        );
        painter.rect_stroke(
            seg,
            1.0,
            egui::Stroke::new(1.0, dim),
            egui::StrokeKind::Inside,
        );
        if filled {
            painter.rect_filled(seg.shrink(1.0), 0.5, color);
        }
    }
    painter.text(
        egui::pos2(bar_left - 4.0, bar_y + 5.0),
        egui::Align2::RIGHT_CENTER,
        "THR",
        egui::FontId::monospace(10.0),
        dim,
    );

    // Altitude (if published).
    if let Some(alt) = altitude {
        painter.text(
            anchor + egui::vec2(140.0, 0.0),
            egui::Align2::LEFT_CENTER,
            format!("{:.0} M", alt),
            egui::FontId::monospace(24.0),
            color,
        );
        painter.text(
            anchor + egui::vec2(140.0, 24.0),
            egui::Align2::LEFT_TOP,
            "ALT",
            egui::FontId::monospace(10.0),
            dim,
        );
    }

    // Callsign below thrust bar.
    if !callsign.is_empty() {
        painter.text(
            egui::pos2(anchor.x, rect.bottom() - 22.0),
            egui::Align2::CENTER_CENTER,
            callsign,
            egui::FontId::monospace(12.0),
            dim,
        );
    }
}

fn paint_warp_panel(
    painter: &egui::Painter,
    rect: egui::Rect,
    signals: &SignalRegistry,
    smoothed: &SmoothedCockpitSignals,
    alpha: u8,
) {
    let phase = signals
        .get("ship.autopilot.phase")
        .map(|r| r.value.as_u8())
        .unwrap_or(0);
    if phase == 0 {
        return;
    }
    let eta = smoothed.autopilot_eta;
    let target = signals
        .get("ship.warp_target")
        .and_then(|r| r.value.as_text().map(str::to_owned))
        .unwrap_or_else(|| "UNKNOWN".into());

    let anchor = egui::pos2(rect.right() - 24.0, rect.top() + 60.0);
    let warn = with_alpha(COLOR_WARN, alpha);
    let dim = with_alpha(COLOR_DIM, alpha.saturating_sub(40));

    painter.text(
        anchor,
        egui::Align2::RIGHT_TOP,
        "◆ AUTOPILOT",
        egui::FontId::monospace(14.0),
        warn,
    );
    let phase_label = match phase {
        1 => "ALIGN",
        2 => "ACCEL",
        3 => "CRUISE",
        4 => "BRAKE",
        5 => "ORBIT",
        _ => "ACTIVE",
    };
    painter.text(
        anchor + egui::vec2(0.0, 20.0),
        egui::Align2::RIGHT_TOP,
        phase_label,
        egui::FontId::monospace(12.0),
        warn,
    );
    painter.text(
        anchor + egui::vec2(0.0, 36.0),
        egui::Align2::RIGHT_TOP,
        format!("ETA {:.0}S", eta),
        egui::FontId::monospace(12.0),
        dim,
    );
    painter.text(
        anchor + egui::vec2(0.0, 52.0),
        egui::Align2::RIGHT_TOP,
        format!("TGT {}", target),
        egui::FontId::monospace(12.0),
        dim,
    );
}

fn paint_connection_breadcrumb(
    painter: &egui::Painter,
    rect: egui::Rect,
    conn: &NetConnection,
    primary_ws: Option<&voxeldust_core::client_message::WorldStateData>,
    alpha: u8,
) {
    let color = with_alpha(COLOR_PRIMARY, alpha.saturating_sub(40));
    let text = match (conn.connected, primary_ws) {
        (true, Some(ws)) => format!(
            "◆ SHARD {} • T={}",
            shard_name(conn.shard_type),
            ws.tick
        ),
        (true, None) => format!("◆ SHARD {} • AWAITING WS", shard_name(conn.shard_type)),
        (false, _) => {
            let color = with_alpha(COLOR_HOSTILE, alpha);
            painter.text(
                egui::pos2(rect.right() - 20.0, rect.bottom() - 20.0),
                egui::Align2::RIGHT_BOTTOM,
                "◆ DISCONNECTED",
                egui::FontId::monospace(12.0),
                color,
            );
            return;
        }
    };
    painter.text(
        egui::pos2(rect.right() - 20.0, rect.bottom() - 20.0),
        egui::Align2::RIGHT_BOTTOM,
        text,
        egui::FontId::monospace(11.0),
        color,
    );
}

fn shard_name(shard_type: u8) -> &'static str {
    match shard_type {
        0 => "PLANET",
        1 => "SYSTEM",
        2 => "SHIP",
        3 => "GALAXY",
        _ => "UNK",
    }
}
