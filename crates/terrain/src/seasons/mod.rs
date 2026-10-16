//! ★ THE SEASONS' CLIMATE — generator 27's rain and winds by law (ruling B99 step 1; the design
//! `docs/investigation/2026-10-15/rain_seasons_design.md`). BENCH ONLY until the owner's look: the game's solve still
//! reads [`crate::climate`]; the bench copy of the macro solve reads this module first (ruling B38).
//!
//! The chain, one law a module:
//! 1. [`sun`] — the daily sunlight by latitude and season (exact geometry, the orbit's precession-averaged year).
//! 2. [`grid`] — the climate grid: a latitude–longitude grid on the spin axis (orthogonal, so its diffusion is exact).
//! 3. [`balance`] — the seasonal moist energy balance in implicit steps through the year.
//! 4. [`wind`] — the rain belt (the energy-flux equator) and the surface winds of each step; [`gill`] — the winds from
//!    the season's heating (Gill's damped balance: the cells, the monsoon's pull, the winter continent's push).
//! 5. [`vapour`] — the evaporation, the vapour carried by the wind and the storms, and the rain of each season.
//! 6. [`orography`] — the mountains' rain on the 8 km nodes: the windward ramp and the lee's shadow.
//! 7. [`chain`] — the six laws as the solve's climate: today's rows, warm from pass to pass.
//!
//! **Example.** On the home planet (tilt 55°) the north pole gets no sun for a quarter of the year and more sun than the
//! equator at midsummer; the table this chain starts from holds both, so the rain belt can follow the sun far north.

// ★ THE SEASONS MAY DIVIDE: they run inside the macro solve on the CPU, never in a kernel (as `crate::climate`).
#![allow(
    clippy::integer_division,
    clippy::modulo_arithmetic,
    reason = "the seasons run inside the macro solve on the CPU, never in a kernel; integer-exact on every host"
)]

pub mod balance;
pub mod chain;
pub mod gill;
pub mod grid;
pub mod orography;
pub mod sun;
pub mod vapour;
pub mod wind;
