//! `vd-client` — the renderer-FREE client logic core (P1.5 Slice 1).
//!
//! Everything the client does that is NOT pixels lives here, as a Tier-A library
//! at 100% region+branch coverage: talk to the gateway over the `sim::io`
//! `Transport` seam, decode snapshots through the shared `vd_wire` §6.3 gate,
//! interpolate delivered poses on a continuous render cursor, and assemble 20 Hz
//! input. The winit/wgpu/egui glue (and the real QUIC transport + dev-control
//! listener) is the Tier-B `client-bin` that WRAPS this core; the core itself
//! touches no clock, no socket, no renderer (clippy-enforced).
//!
//! ## No client-side prediction (a binding mandate)
//! The render path only ever INTERPOLATES between delivered snapshots on a
//! 100-150 ms buffer; it never extrapolates. This is structural, not a comment:
//! the interpolation sample is a [`interp::RenderPose`] with NO velocity field, and
//! `StampedPose::advanced_ballistic`/`.vel` are never read on the render path.

pub mod input;
pub mod interp;
pub mod net;
pub mod realm_scene;
pub mod realm_view;
pub mod render_clock;
pub mod render_snapshot;
pub mod star_sky;
pub mod tuning;
pub mod view;

/// Universe ticks as f64 for the continuous render cursor. Exact for any realistic
/// runtime (a 20 Hz universe reaches 2^53 ticks only after ~14 million years), and
/// interpolation uses small tick DIFFERENCES so sub-tick precision holds regardless.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn tick_to_f64(tick: vd_core::UniverseTick) -> f64 {
    tick.0 as f64
}
