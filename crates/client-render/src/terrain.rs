//! ★ THE TERRAIN ON SCREEN (the voxel foundation, slice 7; ruling V12; slice 8 step 2, ruling V14) —
//! the engine's side of the chunk lane: it ASKS the library for the ladder's chunks, HARVESTS the
//! finished ones, and DRAWS them as children of their realm's row. It never calls the generator,
//! never picks a vertex, never decides a rung: the ladder view (`vd_client::ladder_view`) does.
//!
//! ```text
//!   every frame
//!   ───────────
//!   1. every realm row with a SURFACE statement → the lane builds its body (once)
//!   2. the eye in that body's frame (from the delivered row and the delivered eye — display
//!      arithmetic, no pose derived) → THE WANTED SET: every ring of the ladder from the rung under
//!      the eye out to the horizon, coarsest first (recomputed when the eye has moved half a metre)
//!      → request the new ones; release a chunk no longer wanted only when every wanted chunk over
//!      its footprint has ARRIVED (coarse before fine — no hole while a finer rung is building)
//!   3. harvest finished chunks → a mesh each, a child of the row (floating origin), and in Capture
//!      mode a TWIN of each on the probe layer (slice 8p)
//!   4. every chunk's transform = row placement ⊕ facing · chunk origin, reduced against the eye in
//!      f64 and narrowed once
//!   5. the sun: one directional light from the brightest luminous row in the window
//!   6. THE STAMP and THE RULER (slice 8p): what this frame measured about the ground under the
//!      eye, for the picture and the state; a ball of known size where the centre ray meets ground
//! ```
//!
//! The workers are THREADS behind the library's seam (`std::thread` + `crossbeam-channel`, no new
//! library): the render thread never generates.
//!
//! **Example.** The pilot's hull sits on the home planet. The planet's row carries its surface
//! statement; the lane holds its body; the ladder view names the metre cells for the first 869 m,
//! the two-metre cells out to 1.7 km, and so on to the 6.6 km horizon and the peaks behind it; the
//! workers build the coarse rings first, and the hills are on screen to the horizon, lit by the
//! star's row, turning with the planet because they are children of its row. The stamp on the
//! picture says the eye stands 3.4 m over the recipe, rungs 0 to 9 are drawn, and the star is 15°
//! up; the ruler ball 29 m ahead is a metre across, and the probe says so pixel by pixel.

use std::collections::BTreeMap;
use std::sync::Arc;

/// The ruler ball's icosphere subdivisions: facets of about half a degree, so a silhouette facet's
/// depth stands under a quarter of a cell past the true limb on every pinned stand (see the ball's
/// construction).
const RULER_BALL_SUBDIVISIONS: u32 = 7;

use bevy::prelude::*;
use crossbeam_channel::{Receiver, bounded};
use std::collections::BTreeSet;
use std::sync::Mutex;

use bevy::camera::visibility::RenderLayers;
use vd_client::ask_pace::{AskPace, FrameClock, PeakHold, Throughput};
use vd_client::card_budget::{CARD_BUDGET_FRACTION, CardBudget, clamped_fraction};
use vd_client::chunks::{
    ChunkJob, ChunkLane, ChunkReady, ChunkWorkers, Ruler, body_frame_point, eye_surface,
    geometry_from, geometry_with, oct_encode, ruler_on_surface,
};
use vd_client::ladder_view::{
    AskBound, AskRate, Column, FADE_ALWAYS_IN, FADE_ALWAYS_OUT, LadderView, ShadowReach, WantedSet,
    rung_for_distance, switch_m,
};
use vd_client::realm_scene::{BoxShape, RealmBox, RealmScene};
use vd_client_harness::probe::{
    PROBE_KIND_RULER, PROBE_KIND_TERRAIN, horizon_dip_rad, horizon_m, star_angles,
};
use vd_core::geometry::Boundary;
use vd_core::glam::{DQuat, DVec3};
use vd_core::pose::RealmId;
use vd_devproto::{DevRuler, DevStarAngles, DevTerrainStamp};
use vd_terrain::chunk::ChunkKey;

use super::{GroundMaterial, LadderFade, PROBE_LAYER, ProbeMaterial, SHADOW_LAYER};

/// Flat shading instead of smooth (a debug switch; both are style, ruling S6-5).
pub const FLAT_ENV: &str = "VD_TERRAIN_FLAT";
/// THE DEFAULT harvest cap: finished chunks harvested per frame — a bounded harvest, never a stall.
/// A ladder to the horizon is thousands of chunks (MEASURED: about 4 300 from the ground), so the
/// harvest is wide enough to
/// land one in a few seconds and narrow enough to keep a frame.
const HARVEST_PER_FRAME: usize = 48;
/// The harvest's byte budget: twenty-four near chunks at 400 KB — above the ground stand's mean
/// chunk (305 KB, §18.2) and its largest (about 320 KB with the skirts) — so a frame of near
/// chunks uploads what it uploaded under the old count cap of 24, and a frame of far ones (200
/// KB each) uploads 48 of them under the count cap. MEASURED at 24 × 305 KB: the budget bound
/// under the real near chunk, the 528 m/s harvest fell from 355 to 300 chunks a second and the
/// queue grew to 2 551 — a budget must stand above the count cap's worth, never at it.
/// ★ 2026-09-12: the packed vertex grew from 32 to 36 bytes (the morph normal, item 20), so the
/// same chunk charges an eighth more; the budget grows by the same eighth (24 × 450 KB) to admit
/// the same chunks a frame (refutation: an unchanged budget would have cut the harvest by 11 %).
const HARVEST_BYTES_PER_FRAME: u64 = (24 * 450) << 10;
const HARVEST_BYTES_ENV: &str = "VD_TERRAIN_HARVEST_BYTES";
/// The harvest's count cap as a setting (`VD_TERRAIN_HARVEST_PER_FRAME=<n>`), for the wall's
/// measurement: MEASURED on the 528 m/s leg at fourteen workers, the harvest filled its cap on 593
/// of 1 841 frames while the workers idled half the time — the cap, not the builders, held the
/// finished chunks back (the bounded done queue then parked the workers behind it).
const HARVEST_PER_FRAME_ENV: &str = "VD_TERRAIN_HARVEST_PER_FRAME";
/// THE DONE QUEUE'S BOUND, in frames of the harvest cap (D-TERRAIN-5 item 19): a finished chunk
/// waits in memory with its whole geometry until the harvest takes it, so the workers may run
/// ahead of the harvest by this many frames' worth and no further — a worker that finishes a
/// chunk while that many wait pauses on the hand-over. MEASURED unbounded (the 528 m/s leg under
/// the byte budget): the queue grew to 2 551 finished chunks and the client's small allocations
/// by 2.2 GB over the minute. The bound holds four frames of the cap: 192 chunks, under 80 MB at
/// the ground stand's mean chunk.
const DONE_QUEUE_FRAMES: usize = 4;
/// How far the eye moves before the wanted set is recomputed, in metres: a still stand computes it
/// once; a hull at 528 m/s recomputes every frame.
const EYE_STEP_M: f64 = 0.5;
/// THE SHADOW'S REACH: the cascaded shadow map covers the two nearest rings (out to the second
/// switch distance, 3.5 km), so every hill the eye can tell a shadow on lies inside it. Four
/// cascades, the first ending at a 64th of the reach — the engine's own default ratio. Style: no
/// vertex moves.
const SHADOW_REACH_RUNG: u8 = 2;
const SHADOW_CASCADES: usize = 4;
const SHADOW_FIRST_CASCADE_SHARE: f32 = 1.0 / 64.0;
/// THE NORMAL BIAS, DERIVED FROM THE SUN'S INCIDENCE (M8-L): a shadow map compares depths in texel
/// steps, and a face lit at an angle `i` from its normal spans `tan(i)` texels of depth per texel of
/// width, so a bias smaller than that shadows the face on itself ("shadow acne" — MEASURED on the
/// first re-lit pictures: a 15° star put the whole patch in its own shadow). The bias is the engine's
/// default plus `tan(i)` at the eye's own up, capped where the star is on the horizon.
const SHADOW_BIAS_TAN_CAP: f32 = 8.0;
/// THE RULER'S PAINT (slice 8p): a matte red ball, lit like the ground it stands on and casting its
/// own shadow — the shadow is the second orienter, it says the ball touches the ground. Style.
pub const RULER_SRGB: [f32; 3] = [0.85, 0.12, 0.10];
const RULER_ROUGHNESS: f32 = 0.6;

/// THE SUN'S ILLUMINANCE, FROM THE CAMERA'S OWN EXPOSURE: the lux at which a white face turned
/// square to the sun renders white — `π / exposure`, the inverse of the engine's own pipeline
/// (`out = albedo · E / π · exposure`). Both cameras keep the engine's default exposure, which every
/// star sprite and every marker was measured under; a physical 100 000 lux at that exposure rendered
/// the ground thirty times white (MEASURED on the third ground picture: a cream field, no relief).
/// A picture's brightness is style; nothing here moves a vertex.
fn sun_lux(exposure: &bevy::camera::Exposure) -> f32 {
    std::f32::consts::PI / exposure.exposure()
}

/// What the flags said at startup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainConfig {
    pub flat: bool,
    /// THE PARENT CACHE'S MEMORY BUDGET, in bytes: the cache keeps parent meshes while their
    /// measured bytes fit, never fewer than one working set of the workers.
    pub parent_cache_bytes: usize,
    /// Finished chunks harvested per frame, at most.
    pub harvest_per_frame: usize,
    /// THE HARVEST'S BYTE BUDGET per frame (ruling V15 item 2): the upload stops when the
    /// chunks' bytes reach it, so a frame of small far chunks takes more of them and a frame of
    /// large near ones fewer. Default `HARVEST_BYTES_PER_FRAME`; `VD_TERRAIN_HARVEST_BYTES`
    /// overrides it for a measurement.
    pub harvest_bytes_per_frame: u64,
    /// THE FIRST RUNG WHOSE MESHES KEEP THE ENGINE'S EXACT NORMAL (ruling V18): below it the
    /// packed four-byte normal, from it the twelve-byte one. MEASURED on the orbit stand: under
    /// any 16-bit normal one pixel at the planet's limb (rungs 11–12) moved three levels — the
    /// lighting there divides by a near-zero view angle and reads a normal's error a
    /// hundredfold — while every near stand stayed within one level. Default `EXACT_NORMAL_RUNG`.
    pub exact_normal_rung: u8,
    /// THE FAR-RUNG SPLAT (D8-8's LOOK measurement, `VD_TERRAIN_SPLATS=<rung>`): from this rung
    /// up a chunk is drawn as one camera-facing square per surface vertex, one cell wide, with
    /// the vertex's own normal, morph and radial — the ladder's own voxels drawn as splats instead
    /// of the extracted mesh. `None` (the default, the shipped look) draws meshes on every rung.
    /// A measurement of the LOOK and the frame rate; the bytes of this form are four copies of
    /// each vertex and come DOWN only with vertex pulling, which follows the owner's look ruling.
    pub splat_rung: Option<u8>,
    /// THE ABLATION SWITCHES (D8-8, the frame's wall named by taking work away): the sun casts
    /// shadows (`VD_TERRAIN_SHADOWS=0` turns them off), and chunks from `hide_rung` up are
    /// spawned hidden (`VD_TERRAIN_HIDE_RUNG=<rung>`: built and counted, never drawn). Dev
    /// switches for a measurement, never a gate's path.
    pub shadows: bool,
    pub hide_rung: Option<u8>,
    /// THE SHADOW'S SHAPE (D8-8's ablation named the sun's shadow as the still stand's wall,
    /// §19.6): how far it reaches (the rung whose switch distance ends it), how many cascades
    /// draw it, and each cascade map's width in pixels. Defaults: the constants below; the
    /// environment overrides them for a measurement (`VD_TERRAIN_SHADOW_REACH_RUNG`,
    /// `VD_TERRAIN_SHADOW_CASCADES`, `VD_TERRAIN_SHADOW_MAP`).
    pub shadow_reach_rung: u8,
    pub shadow_cascades: usize,
    pub shadow_map_px: u32,
    /// The shadow's two halves, each a switch for the ablation: the chunks CAST into the maps
    /// (`VD_TERRAIN_SHADOW_CAST=0` marks every chunk a non-caster: the maps stay empty) and the
    /// chunks RECEIVE from them (`VD_TERRAIN_SHADOW_RECEIVE=0` marks every chunk a non-receiver:
    /// the maps are drawn and never read).
    pub shadow_cast: bool,
    pub shadow_receive: bool,
    /// The first rung whose chunks cast no shadow (`VD_TERRAIN_SHADOW_CAST_RUNG=<rung>`): the
    /// casters' cost by rung, for the ablation; `None` lets every rung within the reach cast.
    pub shadow_cast_rung: Option<u8>,
    /// THE SHADOW LADDER (D8-8's shadow cost, MEASURED §19.6–19.9: the casters' vertex count is
    /// the still stand's wall): from `shadow_coarse_from_rung` up, a drawn chunk casts no shadow
    /// itself; the chunk `shadow_coarse_step` rungs coarser that holds it is built on the shadow
    /// layer and casts instead — sixteen times fewer caster vertices per area at two steps. Zero
    /// steps turns the ladder off (every drawn chunk casts). Environment:
    /// `VD_TERRAIN_SHADOW_COARSE_STEP`, `VD_TERRAIN_SHADOW_COARSE_FROM`.
    pub shadow_coarse_step: u8,
    pub shadow_coarse_from_rung: u8,
    /// The chunk workers' thread count; `0` = the machine's SHARE (ruling F6: a quarter of the
    /// cores, at least two — the rest belong to the game that is not built yet).
    pub workers: usize,
    /// ★ THE BOUNDED ASK (ruling F9 item 1): the client measures its builders' throughput and asks
    /// the finest ring only as far ahead as they can deliver it before the ground reaches the
    /// screen. The product default is ON; `VD_TERRAIN_BOUND=0` switches it off for the comparison
    /// flight alone.
    pub ask_bound: bool,
    /// THE THROUGHPUT'S WINDOW, seconds: the exponential average over which the builders' CAPACITY
    /// is read. The capacity is a property of the MACHINE (its worker count over its mean build
    /// time), not of the flight, so the window may be long; what must follow the flight quickly is
    /// the SPEED, and that has its own one-second peak hold. MEASURED at three seconds (the third
    /// and fourth bounded flights): the reading still wandered 183 to 225 chunks a second over one
    /// leg, and the horizon wandered a tenth either side of its mean with it, which asked and
    /// cancelled the same finest-ring chunks frame after frame. TEN seconds is what ships, and it
    /// holds about two thousand builds. One reading may carry at most half of the average
    /// ([`vd_client::ask_pace::THROUGHPUT_ALPHA_MAX`]), so a long idle cannot throw the window
    /// away (review item 10).
    pub throughput_window_s: f64,
    /// THE HORIZON'S HYSTERESIS: how far the horizon IN FORCE must stand FROM THE ONE THE
    /// MEASUREMENT ASKS FOR before it slides toward it at all
    /// ([`vd_client::ladder_view::ASK_BOUND_HYSTERESIS`]). It gates the DISTANCE, never the
    /// per-frame step: a step is proportional to the frame's own seconds, and a gate on the step
    /// refused every step above about 54 frames a second, which left the bound never binding at
    /// all on a fast machine (review item 1, THE BLOCKER). The horizon never jumps — it SLIDES at
    /// [`vd_client::ladder_view::ASK_BOUND_SLEW_PER_S`], because the crossfade band rides on it
    /// and a jumped band is a pop.
    pub ask_bound_hysteresis: f64,
    /// HOW FAR A HORIZON MUST MOVE before the crossfade's three material families follow it
    /// ([`vd_client::ladder_view::ASK_BOUND_REBIND`]). A material rewrite costs the engine a bind
    /// group, and the horizon wanders.
    pub ask_bound_rebind: f64,
    /// HOW FAR THE HORIZON MAY SLIDE before the DESCENT follows it
    /// ([`vd_client::ladder_view::ASK_BOUND_BRACKET`]). The descent is the costliest thing on the
    /// main thread, and it asks a ring wider by [`vd_client::ladder_view::ASK_BOUND_SLACK`] —
    /// DERIVED from this fraction and the materials' own together — so no band it did not cover is
    /// ever drawn.
    pub ask_bound_bracket: f64,
    /// HOW OFTEN THE BOUND MAY FORCE A DESCENT, seconds, per realm
    /// ([`vd_client::ladder_view::ASK_BOUND_DESCENT_S`]). ZERO as shipped — the bracket alone
    /// paces the descent, MEASURED — and a nonzero value caps the descents a realm's bound can
    /// force a second.
    pub ask_bound_descent_s: f64,
    /// ★ THE CARD AS A SECOND BUILDER (ruling F9 item 2): whether the card may build chunks at
    /// all. The CPU share is the default and the fallback; `VD_TERRAIN_GPU=0` keeps the card out,
    /// and a card the self-check does not trust builds nothing whatever this says.
    pub gpu_card: bool,
    /// THE CARD'S SHARE OF A FRAME (`VD_TERRAIN_GPU_BUDGET=<fraction>`): how much of each frame the
    /// card may spend building, so it keeps the rest to DRAW. The default is
    /// [`CARD_BUDGET_FRACTION`], a quarter, whose measured reason is stated there; `1.0` lets the
    /// card build all it can, which is the flight that says what the budget buys. A typed value is
    /// CLAMPED to none-or-all, and a value that had to be clamped is said in the log.
    pub gpu_budget: f64,
    /// ★ THE HEAD-OF-LINE RULE (review item 9, `VD_TERRAIN_GPU_SKIP=<n>`): how many of the queue's
    /// most urgent requests the card leaves to the CPU workers. `None` reads the measured default
    /// ([`CARD_SKIP`], one).
    pub gpu_skip: Option<usize>,
    /// ★ WHETHER THE CARD'S CAPACITY IS SUMMED INTO THE BOUNDED ASK (`VD_TERRAIN_GPU_BOUND=1`).
    ///
    /// OFF by default, and the default is MEASURED (§26.4, three flights of one binary at the
    /// average machine's three workers, the 528 m/s leg): summed, the bound reads 222 to 445 chunks
    /// a second, pulls the finest ring's horizon out from 405 m to 569 m, and the band goes MORE
    /// incomplete, not less — 442 urgent chunks at the worst against 125 with no card at all, and
    /// the frames fall to 43.3. NOT summed, the very same card and the very same chunks read 87
    /// urgent chunks, a queue of 529 against 1 442, and 45.7 frames a second. The card's chunks are
    /// worth more spent on the ask the eye already has than on a wider one.
    pub gpu_bound: bool,
    /// ★ HOW MANY BOXES THE CARD KEEPS IN FLIGHT (`VD_TERRAIN_GPU_FLIGHTS=<n>`): the default is
    /// [`CARD_IN_FLIGHT`], whose measured reason is stated there.
    pub gpu_flights: usize,
    /// ★ THE DRIFT HUNT (`VD_TERRAIN_GPU_VERIFY=1`): every box the card builds is built AGAIN on
    /// the CPU and compared cell for cell, and the first difference is told with its KEY and its
    /// CELL. SL10 asks for no drift as a MEASUREMENT; this is that measurement on the shipped path.
    /// It doubles the geometry stage's work, so it is a hunt's knob and never a default.
    pub gpu_verify: bool,
    /// THE SPEED'S HOLD, seconds: the window the LARGEST reading of the lead's sawtooth is kept
    /// over ([`vd_client::ask_pace::PeakHold`]). The sawtooth's period is the snapshot interval
    /// (0.05 s at the 20 Hz universe tick), so a second holds twenty of its teeth — and because
    /// the hold is a true maximum over a window and not a decay, a hull that stops reads a still
    /// eye within that second instead of reading 194 m/s (review item 5).
    pub speed_hold_s: f64,
}
const SHADOW_COARSE_STEP_ENV: &str = "VD_TERRAIN_SHADOW_COARSE_STEP";
const SHADOW_COARSE_FROM_ENV: &str = "VD_TERRAIN_SHADOW_COARSE_FROM";
/// The shadow ladder's defaults (the owner's acceptance of the look, 2026-09-11, §19.10): one
/// rung coarser from rung 0 up — the ring at the eye's feet casts from 2 m cells. MEASURED: the
/// ground stand at the frame runner's cap (56 frames a second against 29 with every drawn chunk
/// casting), the hill 47 against 27; the look within a dozen far-shadow-edge pixels of the exact
/// one. Two rungs from rung 1 read 44.5 and 39.8 (the near ring's own casting was the last four
/// milliseconds); two rungs from rung 0 read the same as one rung with 75 MB fewer casters, but
/// a 4 m caster at the feet loses a metre-wide rock's shadow that a 2 m one keeps half of.
const SHADOW_COARSE_STEP: u8 = 1;
const SHADOW_COARSE_FROM_RUNG: u8 = 0;
/// A coarse caster's request priority: the margin class at its rung, BEHIND every margin chunk
/// of that rung (the index field at its widest) — a missing caster is a missing shadow, never a
/// hole, so no drawn chunk waits for one.
fn shadow_priority(rung: u8) -> u32 {
    (2u32 << 30) | (u32::from(63 - rung.min(63)) << 24) | 0x00FF_FFFF
}
/// THE WORKER COUNT (`VD_TERRAIN_WORKERS=<n>`; `0` or unset = the machine's own parallelism): the
/// threads that build chunks. A setting, so the throughput wall (§16.3) can be flown with the worker
/// count as the only change.
const WORKERS_ENV: &str = "VD_TERRAIN_WORKERS";
const SHADOW_CAST_RUNG_ENV: &str = "VD_TERRAIN_SHADOW_CAST_RUNG";
const SHADOW_CAST_ENV: &str = "VD_TERRAIN_SHADOW_CAST";
const SHADOW_RECEIVE_ENV: &str = "VD_TERRAIN_SHADOW_RECEIVE";

/// The environment switches of the shadow's shape.
const SHADOW_REACH_ENV: &str = "VD_TERRAIN_SHADOW_REACH_RUNG";
const SHADOW_CASCADES_ENV: &str = "VD_TERRAIN_SHADOW_CASCADES";
const SHADOW_MAP_ENV: &str = "VD_TERRAIN_SHADOW_MAP";
/// The engine's own default width of a cascade's map, in pixels.
const SHADOW_MAP_PX: u32 = 2048;

/// A number from the environment, or the default.
pub(crate) fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

/// WHETHER A SWITCH IS ON. A switch is on unless it is turned off by name, and the names of NO are
/// the ones a person writes: `0`, `false`, `off`, `no`, in any case (review item 9). One reading
/// for every switch, so `VD_TERRAIN_BOUND=false` and `VD_TERRAIN_SHADOWS=off` mean what they say
/// instead of silently meaning ON.
fn env_on(name: &str) -> bool {
    let off = ["0", "false", "off", "no"];
    !std::env::var(name)
        .map(|v| v.trim().to_ascii_lowercase())
        .is_ok_and(|v| off.contains(&v.as_str()))
}

/// WHETHER A SWITCH THAT IS OFF BY DEFAULT WAS ARMED BY NAME. The mirror of [`env_on`], for the
/// switches whose default the measurement decides against ([`GPU_CARD_ENV`], [`GPU_BOUND_ENV`]):
/// each is off unless a
/// person writes `1`, `true`, `on` or `yes`, in any case. Two readers, because there are two kinds
/// of default and a switch must mean what it says either way.
fn env_armed(name: &str) -> bool {
    let yes = ["1", "true", "on", "yes"];
    std::env::var(name)
        .map(|v| v.trim().to_ascii_lowercase())
        .is_ok_and(|v| yes.contains(&v.as_str()))
}

/// The environment switch of the far-rung splat.
const SPLAT_ENV: &str = "VD_TERRAIN_SPLATS";
/// The environment switches of the ablation.
const SHADOWS_ENV: &str = "VD_TERRAIN_SHADOWS";
const HIDE_RUNG_ENV: &str = "VD_TERRAIN_HIDE_RUNG";

/// THE DEFAULT memory budget of the parent cache. The working set on a flight is every ring's
/// LEADING EDGE, not one worker's neighbourhood: the pool builds hundreds of parents a second
/// (ESTIMATED from the stamp's parent builds: about 600 a second at 224 entries) and the next
/// column of a ring arrives about half a second later (ESTIMATED from the ring's chunk size over
/// the speed), so the entries must outlive that. MEASURED on the M8-1 flight at 240 m/s with 14
/// workers (§16.6): 224 entries hit 53 % at 35 ms a chunk, 448 hit 76 % at 22 ms, 896 hit 89 % at
/// 16 ms. 256 MB holds about a thousand of the shrunk parent meshes (step 5, D-TERRAIN-5 item
/// 10: MEASURED 239 KB at rung 1 on the home planet, against 500 KB before), so the 89 % setting
/// fits the budget; the cache bounds itself by the meshes' own bytes, not by a count at an
/// estimated size.
pub const PARENT_CACHE_BYTES: usize = 256 << 20;

impl TerrainConfig {
    /// Read the flags from the environment; the budgets are the defaults.
    #[must_use]
    pub fn from_env() -> TerrainConfig {
        let flat = std::env::var(FLAT_ENV).is_ok_and(|v| v == "1");
        TerrainConfig {
            flat,
            parent_cache_bytes: PARENT_CACHE_BYTES,
            harvest_per_frame: env_or(HARVEST_PER_FRAME_ENV, HARVEST_PER_FRAME),
            harvest_bytes_per_frame: env_or(HARVEST_BYTES_ENV, HARVEST_BYTES_PER_FRAME),
            exact_normal_rung: EXACT_NORMAL_RUNG,
            splat_rung: std::env::var(SPLAT_ENV)
                .ok()
                .and_then(|v| v.parse::<u8>().ok()),
            shadows: env_on(SHADOWS_ENV),
            hide_rung: std::env::var(HIDE_RUNG_ENV)
                .ok()
                .and_then(|v| v.parse::<u8>().ok()),
            shadow_reach_rung: env_or(SHADOW_REACH_ENV, SHADOW_REACH_RUNG),
            shadow_cascades: env_or(SHADOW_CASCADES_ENV, SHADOW_CASCADES),
            shadow_map_px: env_or(SHADOW_MAP_ENV, SHADOW_MAP_PX),
            shadow_cast: env_on(SHADOW_CAST_ENV),
            shadow_receive: env_on(SHADOW_RECEIVE_ENV),
            shadow_cast_rung: std::env::var(SHADOW_CAST_RUNG_ENV)
                .ok()
                .and_then(|v| v.parse::<u8>().ok()),
            shadow_coarse_step: env_or(SHADOW_COARSE_STEP_ENV, SHADOW_COARSE_STEP),
            shadow_coarse_from_rung: env_or(SHADOW_COARSE_FROM_ENV, SHADOW_COARSE_FROM_RUNG),
            workers: env_or(WORKERS_ENV, 0),
            ask_bound: env_on(ASK_BOUND_ENV),
            throughput_window_s: THROUGHPUT_WINDOW_S,
            ask_bound_hysteresis: vd_client::ladder_view::ASK_BOUND_HYSTERESIS,
            ask_bound_rebind: vd_client::ladder_view::ASK_BOUND_REBIND,
            ask_bound_bracket: vd_client::ladder_view::ASK_BOUND_BRACKET,
            ask_bound_descent_s: vd_client::ladder_view::ASK_BOUND_DESCENT_S,
            speed_hold_s: SPEED_HOLD_S,
            gpu_card: env_armed(GPU_CARD_ENV),
            gpu_budget: clamped_fraction(env_or(GPU_BUDGET_ENV, CARD_BUDGET_FRACTION)),
            gpu_skip: std::env::var(GPU_SKIP_ENV)
                .ok()
                .and_then(|v| v.parse::<usize>().ok()),
            gpu_bound: env_armed(GPU_BOUND_ENV),
            gpu_flights: env_or(GPU_FLIGHTS_ENV, CARD_IN_FLIGHT).max(1),
            gpu_verify: env_armed(GPU_VERIFY_ENV),
        }
    }
}

/// THE FRAME'S WORK, the pieces by name (the frame bar's instrument, ruling F9 item 1): the
/// builders' throughput read, the bound's own arithmetic (the speed hold, `ask_bound`, the slew
/// and the two hysteresis tests), the wanted set's DESCENT, and the crossfade materials' rewrite.
const WORK_THROUGHPUT: &str = "throughput";
const WORK_BOUND: &str = "bound";
const WORK_DESCENT: &str = "descent";
const WORK_REBIND: &str = "rebind";

/// ★ HOW LONG A PIECE'S WORST FRAME IS REMEMBERED FOR: one second (review item 8). The peak in
/// the stamp is a ROLLING one, not the run's own, so a flight reading the stamp can name the worst
/// frame OF A LEG instead of the worst frame since the client started. The moving-eye flight
/// samples every 40 ms, so a one-second window is read about twenty-five times over and no frame
/// of a leg escapes it.
const WORK_PEAK_S: f64 = 1.0;

/// ONE PIECE OF THE FRAME'S WORK: its wall nanoseconds since the client started, how many times it
/// ran, and the worst single frame of the last [`WORK_PEAK_S`] seconds.
#[derive(Default)]
struct WorkPiece {
    total_ns: u64,
    runs: u64,
    peak: PeakHold,
    peak_ns: f64,
}

/// The lead sawtooth's peak is held for this many seconds (see `TerrainConfig::speed_hold_s`).
const SPEED_HOLD_S: f64 = 1.0;

/// THE BOUNDED ASK's switch (`VD_TERRAIN_BOUND=0`): off for the comparison flight, on everywhere
/// else. Ruling F9 item 1.
const ASK_BOUND_ENV: &str = "VD_TERRAIN_BOUND";
/// The builders' throughput is read over this many seconds (see `TerrainConfig`).
const THROUGHPUT_WINDOW_S: f64 = 10.0;
/// THE CHUNKS A COLUMN HOLDS before any descent has measured it (ruling F9 item 1): two. MEASURED
/// on the home planet (§17.1's census: 6 659 chunks over the ground stand's columns): a column
/// holds one chunk where the surface crosses one, and two where it crosses a chunk boundary.
const CHUNKS_PER_COLUMN: f64 = 2.0;

/// THE THREADED WORKERS: a PRIORITY QUEUE served by `threads` threads, a done channel back. The
/// library's seam, filled in by the binary (HR5: the library's own tests use the inline workers).
/// The queue orders by the job's priority (the wanted set's own order, ruling V15), then by
/// arrival; a worker takes the first. A cancel REMOVES the job from the queue (refutation T-3:
/// a marker left a job with a poor priority in the map for ever); a job already taken runs, and
/// the lane's poll drops what it no longer wants. A re-request moves a waiting job to its new
/// priority (refutation T-1).
pub struct ThreadedWorkers {
    queue: Arc<(Mutex<JobQueue>, std::sync::Condvar)>,
    done: Receiver<ChunkReady>,
    /// The jobs the workers ran (with or without a geometry), and the wall nanoseconds they spent
    /// on them, a wait on a sibling's parent build included (M8-2a). The CARD's own chunks are
    /// counted apart, in [`CardMeter`], because the two builders' mean times must never be mixed.
    built: Arc<std::sync::atomic::AtomicU64>,
    build_nanos: Arc<std::sync::atomic::AtomicU64>,
    /// ★ THE WORST SINGLE CHUNK (2026-09-16, the walk-gap measurement): the longest wall time one
    /// build took and the chunk it took it on. Two workers that beat the peak in the same instant
    /// may leave either key: the instrument names a dense chunk, it does not order them.
    build_peak: Arc<Mutex<(u64, Option<ChunkKey>)>>,
    /// ★ THE CARD'S SEAM (ruling F9 item 2): the same queue and the same done channel, kept for
    /// the moment the renderer's own device exists.
    card: CardSeam,
}

/// The queue: jobs by (priority, arrival), an index from the chunk to its place, and the close
/// flag the workers leave on.
#[derive(Default)]
struct JobQueue {
    jobs: BTreeMap<(u32, u64), ChunkJob>,
    index: BTreeMap<(RealmId, ChunkKey), (u32, u64)>,
    seq: u64,
    closed: bool,
}

impl JobQueue {
    /// Put a job at `priority`: a job already waiting for the same chunk is moved.
    fn place(&mut self, job: ChunkJob) {
        let at = (job.realm, job.key);
        if let Some(old) = self.index.remove(&at) {
            self.jobs.remove(&old);
        }
        self.seq += 1;
        let slot = (job.priority, self.seq);
        self.index.insert(at, slot);
        self.jobs.insert(slot, job);
    }

    /// Take a waiting job out.
    fn withdraw(&mut self, realm: RealmId, key: ChunkKey) -> Option<ChunkJob> {
        let slot = self.index.remove(&(realm, key))?;
        self.jobs.remove(&slot)
    }
}

/// The parents one chunk reads at most (`vd_client::chunks::parent_keys`): its own and the
/// lateral neighbours' on every side. One working set of the workers is `threads` times this.
const PARENTS_PER_CHUNK: usize = 8;

impl ThreadedWorkers {
    /// Start `threads` workers whose finished chunks wait in a queue of at most `done_bound`
    /// (item 19): a worker that finishes a chunk while the queue is full pauses on the hand-over
    /// until the harvest takes one, so the finished geometry in memory is bounded.
    #[must_use]
    pub fn start(threads: usize, done_bound: usize) -> ThreadedWorkers {
        let queue: Arc<(Mutex<JobQueue>, std::sync::Condvar)> =
            Arc::new((Mutex::new(JobQueue::default()), std::sync::Condvar::new()));
        let (done_tx, done) = bounded::<ChunkReady>(done_bound.max(1));
        let built = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let build_nanos = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let build_peak: Arc<Mutex<(u64, Option<ChunkKey>)>> = Arc::new(Mutex::new((0, None)));
        let mut n = 0;
        while n < threads.max(1) {
            let queue = Arc::clone(&queue);
            let tx = done_tx.clone();
            let built_by_me = Arc::clone(&built);
            let nanos_by_me = Arc::clone(&build_nanos);
            let peak_by_me = Arc::clone(&build_peak);
            std::thread::Builder::new()
                .name(format!("terrain-worker-{n}"))
                .spawn(move || {
                    loop {
                        // Take the first job by priority; wait while the queue is empty; leave
                        // as soon as it is closed (refutation T-29: a close builds nothing more).
                        // ONE rule, for the CPU workers and for the card alike (ruling F9 item 2).
                        let Some(job) = next_job(&queue) else {
                            return;
                        };
                        let started = std::time::Instant::now();
                        let geometry = geometry_with(
                            &job.body,
                            job.artifact.as_deref(),
                            job.realm,
                            job.key,
                            &job.parents,
                        );
                        let took = started.elapsed().as_nanos() as u64;
                        nanos_by_me.fetch_add(took, std::sync::atomic::Ordering::Relaxed);
                        {
                            let mut peak = peak_by_me
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            if took > peak.0 {
                                *peak = (took, Some(job.key));
                            }
                        }
                        built_by_me.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if let Some(geometry) = geometry {
                            let _ = tx.send(ChunkReady {
                                realm: job.realm,
                                geometry,
                            });
                        }
                    }
                })
                .expect("a worker thread starts");
            n += 1;
        }
        ThreadedWorkers {
            card: CardSeam {
                queue: Arc::clone(&queue),
                done: done_tx,
                meter: Arc::new(CardMeter::default()),
            },
            queue,
            done,
            built,
            build_nanos,
            build_peak,
        }
    }

    /// ★ THE CARD'S SEAM ON THESE WORKERS (ruling F9 item 2): the client keeps it and fills it in
    /// once the renderer's device exists.
    #[must_use]
    pub fn card_seam(&self) -> CardSeam {
        self.card.clone()
    }

    /// How many jobs wait in the queue.
    #[must_use]
    pub fn waiting(&self) -> usize {
        let (lock, _) = &*self.queue;
        lock.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .jobs
            .len()
    }
}

impl Drop for ThreadedWorkers {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.queue;
        {
            let mut q = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            q.closed = true;
            q.jobs.clear();
            q.index.clear();
        }
        cvar.notify_all();
        // The card waits on its own grant, never on the queue, so it is closed by its own flag.
        self.card
            .meter
            .closed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.card.meter.granted.notify_all();
    }
}

impl ChunkWorkers for ThreadedWorkers {
    fn built(&self) -> vd_client::chunks::BuildCount {
        let peak = *self
            .build_peak
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        vd_client::chunks::BuildCount {
            chunks: self.built.load(std::sync::atomic::Ordering::Relaxed),
            nanos: self.build_nanos.load(std::sync::atomic::Ordering::Relaxed),
            peak_nanos: peak.0,
            peak_key: peak.1,
        }
    }

    fn submit(&mut self, job: ChunkJob) {
        let (lock, cvar) = &*self.queue;
        lock.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .place(job);
        // ★ EVERY BUILDER IS WOKEN, never one of them (review item 9's own defect, MEASURED: the
        // walk's settle hung for ever on ONE pending chunk). The card waits for the request AFTER
        // the workers' next few, so a queue of one holds nothing FOR THE CARD — and a `notify_one`
        // that happened to wake the card left the CPU workers asleep beside a job they could have
        // taken. Two kinds of waiter on one queue means every wake is a broadcast.
        cvar.notify_all();
    }

    fn reprioritise(&mut self, realm: RealmId, key: ChunkKey, priority: u32) {
        let (lock, _) = &*self.queue;
        let mut q = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(mut job) = q.withdraw(realm, key) {
            job.priority = priority;
            q.place(job);
        }
    }

    fn cancel(&mut self, realm: RealmId, key: ChunkKey) {
        let (lock, _) = &*self.queue;
        let _ = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .withdraw(realm, key);
    }

    fn drain(&mut self, out: &mut Vec<ChunkReady>, max: usize) {
        let mut n = 0;
        while n < max {
            match self.done.try_recv() {
                Ok(ready) => out.push(ready),
                Err(_) => return,
            }
            n += 1;
        }
    }
}

/// ★ THE FRAME METER (ruling F9 item 2, step 1): the frames the terrain system has run, and the
/// worst single frame of the last second in nanoseconds — shared, so an INSTRUMENT ON ANOTHER
/// THREAD can read what the renderer is doing while it works. The seam probe is that instrument:
/// it is the one thing that must say what a worker on the renderer's own device costs the frames,
/// and it cannot ask the main thread for them.
#[derive(Default)]
pub struct FrameMeter {
    /// The frames the terrain system has run since the client started.
    pub frames: std::sync::atomic::AtomicU64,
    /// The worst single frame of the last second, in nanoseconds.
    pub peak_ns: std::sync::atomic::AtomicU64,
}

/// ★ THE CARD'S KNOBS (ruling F9 item 2). `VD_TERRAIN_GPU=1/true/on/yes` makes the card a second
/// builder — and the default is OFF for a MEASURED reason, though no longer the first one. At speed
/// the card pays (§26.8: the band's worst gap at 528 m/s falls from 222 urgent chunks to 39 and the
/// queue from 1 660 to 368, and the turning leg's gap from 106 to 0), and the still stands are
/// CURED by the stand-down rule ([`vd_client::card_gate`]). ⚠ WHAT KEEPS IT OFF is a defect of the
/// card's own box: the picture gate's SEAM stand draws the same 6 401 chunks with 14 959 pixels of
/// nothing under the drawn ground, at one box in flight as well as two — so the card's box is not
/// the CPU's for every key, and the boot self-check compares eight golden keys (§26.9); `VD_TERRAIN_GPU_BUDGET=<fraction>` is its share of a frame (the default is
/// [`CARD_BUDGET_FRACTION`], a quarter, clamped to none-or-all by
/// [`vd_client::card_budget::clamped_fraction`]); `VD_TERRAIN_GPU_SKIP=<n>` is the head-of-line
/// rule below; `VD_TERRAIN_GPU_BOUND=1` SUMS the card's capacity into the bounded ask, which the
/// measurement refuses as a default (see [`GPU_BOUND_ENV`]'s own note).
const GPU_CARD_ENV: &str = "VD_TERRAIN_GPU";
const GPU_BUDGET_ENV: &str = "VD_TERRAIN_GPU_BUDGET";
const GPU_SKIP_ENV: &str = "VD_TERRAIN_GPU_SKIP";
const GPU_FLIGHTS_ENV: &str = "VD_TERRAIN_GPU_FLIGHTS";
const GPU_VERIFY_ENV: &str = "VD_TERRAIN_GPU_VERIFY";
const GPU_BOUND_ENV: &str = "VD_TERRAIN_GPU_BOUND";

/// ★ THE HEAD-OF-LINE RULE (review item 9): how many of the queue's most urgent requests the card
/// LEAVES ALONE. The card holds a chunk longer than a CPU worker does — its box waits on the
/// device while a worker's box is pure arithmetic — so a card that takes the single most urgent
/// request delays exactly the chunk the picture is waiting for. The card therefore takes the
/// request AFTER the workers' next few.
///
/// ★ THE DEFAULT IS ONE, and THREE flights of one binary pick it (§26.4). Against a card that
/// takes the head of the queue it reads 87 urgent chunks at 528 m/s against 113, and 10 against 65
/// at 240 m/s. Against a card that skips THE WORKER COUNT — tried because that card stays out of
/// the tail of a still stand — it reads 87 against 97 at 528 m/s, 50 against 58 turning, and, on
/// the slow hull's leg, **a band that holds on every frame against 3 394 urgent chunks**: a card
/// that skips three never enters that leg's queue at all, so the leg reads what no card at all
/// reads. And the still stand it was meant to cure stayed RED either way (the hill stand settled at
/// tick 2 446 at one and 2 408 at three, against a capture tick of 2 400), which is why the card
/// ships behind [`GPU_CARD_ENV`] and this default follows the flights.
/// `VD_TERRAIN_GPU_SKIP=<n>` names another; `0` gives the card the head of the queue again.
const CARD_SKIP: usize = 1;

/// ★ HOW MANY BOXES THE CARD KEEPS IN FLIGHT (the owner's step after Step 15): TWO.
///
/// The card's own three passes cost it 0.08 ms by the device's own clock; the ROUND TRIP that
/// carries the answer home — the submit, the device's queue, the map back — costs 1.8 ms of wall
/// time. A builder that waits for each trip before it starts the next is bounded by the TRIP, not
/// by the card: it can never build more than about 550 boxes a second however cheap the arithmetic
/// is. Submitting the next box before the last one is read back hides the wait behind the next
/// box's work.
///
/// ★ MEASURED by the seam probe alone, on this machine, over the eight golden boxes (`just
/// gpu-seam` at `VD_TERRAIN_GPU_FLIGHTS=1/2/4`, the renderer drawing beside it):
///
/// | boxes in flight | the probe's boxes a second | the worst box | the renderer's frames |
/// |---|---|---|---|
/// | one | 547 | 7.64 ms | 101.1 % of quiet |
/// | TWO | **1 005** | 9.23 ms | 101.3 % of quiet |
/// | four | 1 522 | 10.91 ms | 101.8 % of quiet |
///
/// Two nearly doubles the rate, and four raises it by half again — the round trip WAS the ceiling,
/// exactly as the 0.08 ms box against the 1.8 ms trip said. ★ But the BUILDER's own ceiling is not
/// the trip: it is the GEOMETRY STAGE, one thread at about 11 ms a chunk, which is why the card's
/// stated capacity in the flights is about 90 chunks a second and not 500. So the lanes buy the
/// device's idleness back and nothing more, and TWO is what ships: it keeps the device fed while
/// one answer is mapped home, and costs one extra set of every buffer (about 2.5 MB at the finest
/// rung) instead of three. More geometry threads are the next lever, and they are not measured.
///
/// ⚠ AND THE CARD'S OWN CLOCK READS HIGH WITH LANES: a box's two timestamps bracket whatever else
/// the device ran between them, so the probe reads 0.07 ms a box at one lane, 0.16 at two and 0.24
/// at four for the same arithmetic. The budget therefore rations an UPPER BOUND on the card's own
/// seconds, which is the safe direction — the card can only spend less than it is charged.
const CARD_IN_FLIGHT: usize = 2;

/// HOW LONG THE CARD WAITS FOR ITS NEXT GRANT before it looks again. The frames grant, and a frame
/// is about 22 ms; a tenth of a second is the longest the card can sleep past a grant, and it is
/// only ever reached where the frames have STOPPED (a client shutting down), which is exactly when
/// the card must wake to see the close flag.
const CARD_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

/// ★ THE CARD'S METER: what the card built, what it cost, and what the frames granted it — shared
/// by the card's own threads and the frame that grants (ruling F9 item 2).
#[derive(Default)]
pub struct CardMeter {
    /// The chunks the card sampled and handed to the geometry stage.
    boxes: std::sync::atomic::AtomicU64,
    /// The wall nanoseconds THE CARD ITSELF took over those boxes (never the host's stages).
    nanos: std::sync::atomic::AtomicU64,
    /// The nanoseconds the frames actually added to the card's allowance since the client started.
    granted_nanos: std::sync::atomic::AtomicU64,
    /// The budget's arithmetic, in the Tier-A library.
    budget: Mutex<CardBudget>,
    /// ★ THE STAND-DOWN RULE'S ANSWER (the owner's step after Step 15): HOW MANY CHUNKS THE CARD
    /// MAY HOLD AT ONCE — zero where the queue is shallow or the eye is still, so the card takes
    /// nothing at all. The FRAME writes it, from the pure rule in the Tier-A library
    /// ([`vd_client::card_gate::QueueDepth::card_may_hold`]); the card's builder reads it. It
    /// starts ZERO, so a card no frame has judged yet takes nothing.
    hold_cap: std::sync::atomic::AtomicU64,
    /// WHAT THE CARD HOLDS RIGHT NOW: the chunks it has taken and not yet handed to the harvest —
    /// the boxes in flight on the device and the ones waiting for the geometry stage.
    holding: std::sync::atomic::AtomicU64,
    /// How many frames judged the queue, and how many of them stood the card down: the stamp and
    /// the flight state both, so a card that never builds says WHY.
    judged: std::sync::atomic::AtomicU64,
    stood_down: std::sync::atomic::AtomicU64,
    /// The workers are closing: the card leaves.
    closed: std::sync::atomic::AtomicBool,
    /// Whether the card's own time is READ FROM THE DEVICE's timestamps, or taken from the host's
    /// wall clock around the submit (review item 1a). The stamp states which.
    device_timed: std::sync::atomic::AtomicBool,
    /// A failure is told once, never on every box.
    told: std::sync::atomic::AtomicBool,
    /// The card's thread waits on this for its next grant.
    granted: std::sync::Condvar,
}

impl CardMeter {
    /// THE FRAME GRANTS the card its share and wakes it.
    ///
    /// The counter adds WHAT THE ALLOWANCE ACTUALLY TOOK (review item 7): a frame that granted into
    /// a full allowance added nothing, and the stamp says so, so the budget's use is read against
    /// the time the card could really have spent.
    fn grant(&self, frame_s: f64, fraction: f64) {
        let added = self
            .budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .grant(frame_s, fraction);
        self.granted_nanos
            .fetch_add((added * 1.0e9) as u64, std::sync::atomic::Ordering::Relaxed);
        self.granted.notify_all();
    }

    /// ★ THE FRAME JUDGES THE QUEUE (the owner's step after Step 15): the pure stand-down rule
    /// over the readings the bounded ask already holds. The answer is counted, so the flight can
    /// say how often the card stood down and the stamp can state it.
    fn judge(&self, depth: vd_client::card_gate::QueueDepth) {
        let stage_s = self
            .budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stage_s();
        let cap = depth.card_may_hold(stage_s);
        self.hold_cap
            .store(cap as u64, std::sync::atomic::Ordering::Relaxed);
        self.judged
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if cap == 0 {
            self.stood_down
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.granted.notify_all();
    }

    /// THE CARD TAKES ONE MORE CHUNK: it holds it until the geometry stage hands it to the harvest.
    fn hold(&self) {
        self.holding
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// THE CARD LETS ONE GO: the chunk reached the harvest, or the job went back to the queue.
    fn release(&self) {
        self.holding
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |held| Some(held.saturating_sub(1)),
            )
            .unwrap_or_default();
    }

    /// Whether the card may take one more chunk: the frame's own cap against what the card holds.
    fn has_room(&self) -> bool {
        self.holding.load(std::sync::atomic::Ordering::Relaxed)
            < self.hold_cap.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What the stand-down rule said: how many frames judged, and how many stood the card down.
    fn stand_down(&self) -> (u64, u64) {
        (
            self.judged.load(std::sync::atomic::Ordering::Relaxed),
            self.stood_down.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// ★ MAY THE CARD START A BOX RIGHT NOW — asked WITHOUT waiting and WITHOUT spending, so the
    /// builder can fill its lanes without ever holding a job it may not build. The card's builder
    /// is the only thread that spends, so what this answers is still true when it takes.
    fn can_take(&self) -> bool {
        if self.closed.load(std::sync::atomic::Ordering::Relaxed) | !self.has_room() {
            return false;
        }
        self.budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .can_take()
    }

    /// THE BUILDER SPENDS one box's own time, once it holds the job (never blocks).
    fn take_now(&self) -> bool {
        self.budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// THE CARD WAITS for a queue deep enough to want it AND a grant that covers one box. `false`
    /// means the workers closed, or the card has left.
    fn take_blocking(&self) -> bool {
        let mut budget = self
            .budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if self.closed.load(std::sync::atomic::Ordering::Relaxed) | budget.is_detached() {
                return false;
            }
            // ★ THE STAND-DOWN RULE FIRST: a card the frame has left no room for takes no job, so
            // a still stand's short queue is the CPU workers' alone.
            if self.has_room() && budget.take() {
                return true;
            }
            budget = self
                .granted
                .wait_timeout(budget, CARD_WAIT)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    /// ★ THE CARD HAS LEFT (review item 4): a dispatch failed. The budget detaches, the capacity is
    /// zero from here on, the client keeps drawing and the CPU share carries the ladder alone. The
    /// failure is told ONCE.
    fn detach(&self, why: &str) {
        self.budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .detach();
        if !self.told.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                why,
                "THE CARD HAS LEFT: a box failed on the device, so the card builds nothing more — \
                 the CPU workers carry the whole ladder and the client goes on drawing"
            );
        }
        self.granted.notify_all();
    }

    /// What the card built and what it cost, for the stamp: the chunks, the card's own nanoseconds,
    /// the nanoseconds granted, the measured time of one box in milliseconds, how many boxes this
    /// frame's own budget allows, the card's own capacity, and whether the card's time is the
    /// DEVICE's own reading.
    fn read(&self) -> (u64, u64, u64, f64, f64, f64, bool) {
        let budget = self
            .budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            self.boxes.load(std::sync::atomic::Ordering::Relaxed),
            self.nanos.load(std::sync::atomic::Ordering::Relaxed),
            self.granted_nanos
                .load(std::sync::atomic::Ordering::Relaxed),
            budget.per_box_s() * 1.0e3,
            budget.boxes_per_frame(),
            budget.capacity_per_s(),
            self.device_timed.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// The card's own capacity, chunks a second, as the bounded ask must read it. Zero once the
    /// card has left.
    fn capacity_per_s(&self) -> f64 {
        self.budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .capacity_per_s()
    }
}

/// ★ THE CARD'S SEAM ON THE WORKERS (ruling F9 item 2): everything the card's threads need to take
/// jobs from the SAME queue by the SAME priority and hand their chunks to the SAME harvest. The
/// client holds one of these from the moment the workers start, and fills it in at start-up — the
/// renderer's own device does not exist until then.
#[derive(Clone)]
pub struct CardSeam {
    queue: Arc<(Mutex<JobQueue>, std::sync::Condvar)>,
    done: crossbeam_channel::Sender<ChunkReady>,
    meter: Arc<CardMeter>,
}

/// One sampled box on its way from the card to the geometry stage: the job it belongs to, the plan
/// that names the box, and the RAW bytes the card wrote. The decode is the geometry stage's work.
struct SampledBox {
    job: ChunkJob,
    plan: vd_terrain::gpu::BoxPlan,
    cells: Vec<u8>,
    dirs: Vec<u8>,
}

impl CardSeam {
    /// ★ THE CARD BECOMES A BUILDER, in TWO STAGES (ruling F9 item 2; review item 1c).
    ///
    /// THE CARD THREAD plans a box's topology, uploads it into the pooled buffers, submits the
    /// three passes and reads the bytes back — and does nothing else, so the time the budget
    /// rations is the card's own and the thread is free the moment the device is.
    /// THE GEOMETRY THREAD decodes those bytes into the box (`BoxPlan::box_of`) and runs the SAME
    /// geometry step the CPU workers run (`vd_client::chunks::geometry_from`), then sends the chunk
    /// down the same done channel to the same harvest.
    ///
    /// **Example.** The pilot flies at 528 m/s. The wanted set asks for a ring of metre-rung
    /// chunks; the CPU workers take the most urgent ones and the card takes the one after them
    /// (`skip`), computes its 262 144 cells and 4 096 column directions — byte for byte what the
    /// shard's CPU writes — and hands the bytes on while it starts the next box.
    pub fn attach(
        &self,
        device: wgpu::Device,
        queue: wgpu::Queue,
        skip: usize,
        done_bound: usize,
        lanes: usize,
        verify: bool,
    ) {
        let (tx, rx) = bounded::<SampledBox>(done_bound.max(1));
        // THE GEOMETRY STAGE.
        let done = self.done.clone();
        let meter = Arc::clone(&self.meter);
        std::thread::Builder::new()
            .name("terrain-card-geometry".to_owned())
            .spawn(move || {
                let mut told = 0u32;
                let mut next_spot = std::time::Instant::now();
                while let Ok(sampled) = rx.recv() {
                    let at = std::time::Instant::now();
                    let samples = sampled
                        .plan
                        .box_of(&cells_of(&sampled.cells), &dirs_of(&sampled.dirs));
                    // ★★ THE SPOT-CHECK, and the DRIFT HUNT (`VD_TERRAIN_GPU_VERIFY=1`): the same
                    // box on the CPU, cell for cell. The hunt checks EVERY box; the shipped path
                    // checks ONE A SECOND. Either runs BEFORE the geometry step, so a box that
                    // drifted is named with the key that drew it — and on the shipped path the
                    // card LEAVES, because a shape that is not the CPU's may not reach the screen.
                    let now = std::time::Instant::now();
                    let spot = now >= next_spot;
                    if verify | spot {
                        next_spot = now + CARD_SPOT_CHECK;
                        let drifted = verify_card_box(
                            &sampled.job.body,
                            sampled.job.key,
                            &samples,
                            told < VERIFY_TOLD_MAX,
                        );
                        told += u32::from(drifted);
                        if drifted {
                            meter.detach("a card box was not the CPU's");
                        }
                    }
                    // The card never takes a job with an artifact (it holds no `Z`), so the
                    // field here is always none: the box is the recipe's own.
                    let geometry = geometry_from(
                        &sampled.job.body,
                        sampled.job.artifact.as_deref(),
                        sampled.job.realm,
                        sampled.job.key,
                        &samples,
                        &sampled.job.parents,
                    );
                    meter
                        .budget
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .note_geometry(at.elapsed().as_secs_f64());
                    if let Some(geometry) = geometry {
                        let _ = done.send(ChunkReady {
                            realm: sampled.job.realm,
                            geometry,
                        });
                    }
                    // ★ THE CARD LETS THE CHUNK GO (the owner's step after Step 15): what the card
                    // HOLDS is what it has taken and not yet delivered, and the frame's cap is
                    // read against exactly that.
                    meter.release();
                }
            })
            .expect("the card's geometry thread starts");
        // THE CARD THREAD.
        let jobs = Arc::clone(&self.queue);
        let meter = Arc::clone(&self.meter);
        std::thread::Builder::new()
            .name("terrain-card".to_owned())
            .spawn(move || {
                // ★ ONE GEAR PER BOX IN FLIGHT (the owner's step after Step 15): the card submits
                // into the next lane while the last one is still on the device, so the throughput
                // is the bus's and not the round trip's.
                let mut gears: Vec<crate::gpu_check::BoxGear> = (0..lanes.max(1))
                    .map(|_| crate::gpu_check::BoxGear::new(device.clone(), queue.clone()))
                    .collect();
                let device_timed = gears[0].device_timed();
                meter
                    .device_timed
                    .store(device_timed, std::sync::atomic::Ordering::Relaxed);
                meter
                    .budget
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .set_lanes(gears.len());
                tracing::info!(
                    device_timed,
                    skip,
                    lanes = gears.len(),
                    "THE CARD'S BUILDER starts: it takes the request after the workers' next few \
                     when the queue is deep enough to want it, it keeps several boxes in flight, \
                     and its own time is read from the device where the device offers it"
                );
                // The boxes on the device, oldest first: which lane each flies in, and what it is.
                let mut flying: std::collections::VecDeque<(
                    usize,
                    ChunkJob,
                    vd_terrain::gpu::BoxPlan,
                )> = std::collections::VecDeque::new();
                // The lanes with nothing in flight.
                let mut idle: Vec<usize> = (0..gears.len()).rev().collect();
                loop {
                    // ★ FILL THE LANES. A builder with NOTHING in flight waits — for a queue deep
                    // enough to want it, for its grant, and for a job. One that already holds a box
                    // never waits: it takes what is there and goes back to collect.
                    while let Some(lane) = idle.pop() {
                        let job = if flying.is_empty() {
                            // THE BUDGET AND THE STAND-DOWN RULE FIRST: the card never holds a job
                            // it may not build, so a job it cannot afford stays for a CPU worker.
                            if !meter.take_blocking() {
                                return;
                            }
                            let Some(job) = next_job_for_card(&jobs, skip) else {
                                return;
                            };
                            meter.hold();
                            job
                        } else {
                            // The same order, asked and never waited on: the peek answers what the
                            // spend would, because this thread is the only spender.
                            if !meter.can_take() {
                                idle.push(lane);
                                break;
                            }
                            let Some(job) = take_job_now(&jobs, skip) else {
                                idle.push(lane);
                                break;
                            };
                            if !meter.take_now() {
                                // Nothing else spends, so this cannot happen after a peek — and
                                // if it ever did, the job goes back rather than being held.
                                give_back(&jobs, job);
                                idle.push(lane);
                                break;
                            }
                            meter.hold();
                            job
                        };
                        // A key outside the body's ladder is the seam's own refusal: it is NOT a
                        // box, so it is neither timed nor counted (review item 3).
                        let Some(plan) = vd_terrain::gpu::plan(&job.body, job.key) else {
                            meter.release();
                            idle.push(lane);
                            continue;
                        };
                        gears[lane].submit(&plan);
                        flying.push_back((lane, job, plan));
                    }
                    // ★ THE OLDEST BOX COMES HOME, while the others stay on the device.
                    let Some((lane, job, plan)) = flying.pop_front() else {
                        continue;
                    };
                    let run = match gears[lane].collect() {
                        Ok(run) => run,
                        Err(why) => {
                            // Every job the card holds goes back to the queue for a CPU worker,
                            // and the card leaves.
                            give_back(&jobs, job);
                            meter.release();
                            for (_, held, _) in flying {
                                give_back(&jobs, held);
                                meter.release();
                            }
                            meter.detach(&why);
                            return;
                        }
                    };
                    idle.push(lane);
                    meter
                        .budget
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .note_box(run.device_s, run.trip_s);
                    meter
                        .boxes
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    meter.nanos.fetch_add(
                        (run.device_s * 1.0e9) as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    if tx
                        .send(SampledBox {
                            job,
                            plan,
                            cells: run.cells,
                            dirs: run.dirs,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            })
            .expect("the card's builder thread starts");
    }
}

/// How many differing boxes the drift hunt names in full before it only counts them.
const VERIFY_TOLD_MAX: u32 = 8;

/// ★★ THE SPOT-CHECK'S PACE: one card box a second, rebuilt on the CPU and compared (the drift
/// hunt, 2026-09-14). SL10 asks for no drift as a MEASUREMENT, and the boot self-check measures it
/// ONCE, on eight keys; this measures it FOR EVER, on the keys the player's own flight asks for.
/// A box costs the geometry thread about 11 ms, so one a second is about a hundredth of that
/// thread and nothing at all of the frame. A box that is not the CPU's DETACHES the card — the
/// same path a failed dispatch takes — so a drift costs the picture nothing but the card.
const CARD_SPOT_CHECK: std::time::Duration = std::time::Duration::from_secs(1);

/// ★ THE DRIFT HUNT'S COMPARE (`VD_TERRAIN_GPU_VERIFY=1`): the card's box against the CPU's, cell
/// for cell and direction for direction. Answers whether this box DIFFERED, and tells the first
/// difference with the key that drew it, so a hunt names a key and not a stand.
fn verify_card_box(
    body: &vd_terrain::BodyDefinition,
    key: ChunkKey,
    card: &vd_terrain::lattice::SampleBox,
    tell: bool,
) -> bool {
    let Some(cpu) = vd_terrain::lattice::sample_box(body, None, key) else {
        return false;
    };
    let mut cells = 0usize;
    let mut first: Option<(usize, String, String)> = None;
    let mut i = 0;
    while i < cpu.cells.len().min(card.cells.len()) {
        if cpu.cells[i] != card.cells[i] {
            cells += 1;
            if first.is_none() {
                first = Some((
                    i,
                    format!("{:?}", card.cells[i]),
                    format!("{:?}", cpu.cells[i]),
                ));
            }
        }
        i += 1;
    }
    let mut dirs = 0usize;
    let mut j = 0;
    while j < cpu.dirs.len().min(card.dirs.len()) {
        dirs += usize::from(cpu.dirs[j] != card.dirs[j]);
        j += 1;
    }
    let lengths = (cpu.cells.len() != card.cells.len()) | (cpu.dirs.len() != card.dirs.len());
    let differed = (cells > 0) | (dirs > 0) | lengths;
    if differed & tell {
        let (index, on_card, on_cpu) = first.unwrap_or((0, String::new(), String::new()));
        tracing::error!(
            face = ?key.face,
            rung = key.rung,
            x = key.x,
            y = key.y,
            z = key.z,
            cells,
            dirs,
            card_cells = card.cells.len(),
            cpu_cells = cpu.cells.len(),
            card_dirs = card.dirs.len(),
            cpu_dirs = cpu.dirs.len(),
            index,
            on_card,
            on_cpu,
            "THE CARD'S BOX IS NOT THE CPU'S"
        );
    }
    differed
}

/// The cells of a readback, decoded on the GEOMETRY stage's own thread.
fn cells_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
        .collect()
}

/// The column directions of a readback, decoded on the GEOMETRY stage's own thread.
fn dirs_of(bytes: &[u8]) -> Vec<[vd_recipe::Gi; 3]> {
    bytes
        .chunks_exact(24)
        .map(|row| {
            let word = |k: usize| {
                vd_recipe::Gi::new(i64::from_le_bytes(
                    row[k * 8..k * 8 + 8].try_into().unwrap_or([0; 8]),
                ))
            };
            [word(0), word(1), word(2)]
        })
        .collect()
}

/// The first job of the queue by priority, waiting while the queue is empty; `None` once the
/// workers close. The card and the CPU workers take from this one queue by this one rule.
fn next_job(jobs: &Arc<(Mutex<JobQueue>, std::sync::Condvar)>) -> Option<ChunkJob> {
    take_job(jobs, 0)
}

/// ★ THE CARD'S OWN TAKE (review item 9): the same queue and the same order, but the card leaves
/// the `skip` most urgent requests to the CPU workers and takes the one after them. A queue that
/// holds no more than `skip` requests holds nothing for the card, and it waits.
fn next_job_for_card(
    jobs: &Arc<(Mutex<JobQueue>, std::sync::Condvar)>,
    skip: usize,
) -> Option<ChunkJob> {
    take_job(jobs, skip)
}

/// ★ ONE JOB OUT OF THE QUEUE WITHOUT WAITING, `skip` places down the priority order: `None` where
/// the queue holds no more than `skip` requests, and `None` once the workers close. The card's fill
/// loop takes this way while it already holds a box in flight — a builder that waits there would
/// leave the device idle.
fn take_job_now(
    jobs: &Arc<(Mutex<JobQueue>, std::sync::Condvar)>,
    skip: usize,
) -> Option<ChunkJob> {
    let (lock, _) = &**jobs;
    let mut q = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if q.closed {
        return None;
    }
    let slot = card_slot(&q, skip)?;
    let job = q.jobs.remove(&slot).expect("the slot was just read");
    q.index.remove(&(job.realm, job.key));
    Some(job)
}

/// ★ THE SLOT THE CARD MAY TAKE: `skip` places down the order, and then the first job WITHOUT an
/// artifact — a chunk that reads an artifact's field is the CPU builders' alone, because the card
/// holds no `Z` (slice 8c stage C4c; ruling F9, the card builder parked). `None` where every job
/// past the skip carries one, or the queue holds no more than `skip`.
fn card_slot(q: &JobQueue, skip: usize) -> Option<(u32, u64)> {
    q.jobs
        .iter()
        .skip(skip)
        .find(|(_, job)| job.artifact.is_none())
        .map(|(slot, _)| *slot)
}

/// A JOB GOES BACK to the queue for a CPU worker, at its own priority, and every waiter is woken.
fn give_back(jobs: &Arc<(Mutex<JobQueue>, std::sync::Condvar)>, job: ChunkJob) {
    let (lock, cvar) = &**jobs;
    lock.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .place(job);
    cvar.notify_all();
}

/// One job out of the queue, `skip` places down the priority order; `None` once the workers close.
fn take_job(jobs: &Arc<(Mutex<JobQueue>, std::sync::Condvar)>, skip: usize) -> Option<ChunkJob> {
    let (lock, cvar) = &**jobs;
    let mut q = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if q.closed {
            return None;
        }
        // The CPU workers (skip 0) take the first job whatever it reads; the card takes the first
        // past its skip that reads no artifact.
        let slot = if skip == 0 {
            q.jobs.keys().next().copied()
        } else {
            card_slot(&q, skip)
        };
        if let Some(slot) = slot {
            let job = q.jobs.remove(&slot).expect("the slot was just read");
            q.index.remove(&(job.realm, job.key));
            return Some(job);
        }
        q = cvar
            .wait(q)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

/// The least chunk key in the map's order (the face first, then the rung, then the cell): where a
/// realm's keys start in the drawn map.
const FIRST_CHUNK_KEY: ChunkKey = ChunkKey {
    face: vd_seed::bend::Face::PosX,
    rung: 0,
    x: i32::MIN,
    y: i32::MIN,
    z: i32::MIN,
};

/// THE SHADOW'S REACH for the ladder's casting set (item 18): the cascades' reach, the sun's
/// tangent as last placed (the bias cap's worth before the sun is born — the longest shadows, so
/// no caster is missed), and the shadow ladder's step. `None` while the shadow ladder is off (no
/// shadows, no casters, or a zero step): every chunk may then ask for a caster, as before.
fn shadow_reach(config: &TerrainConfig, sun_tan_i: Option<f32>) -> Option<ShadowReach> {
    (config.shadows && config.shadow_cast && config.shadow_coarse_step > 0).then(|| ShadowReach {
        reach_m: switch_m(config.shadow_reach_rung),
        tan_i: f64::from(sun_tan_i.unwrap_or(SHADOW_BIAS_TAN_CAP)),
        coarse_step: config.shadow_coarse_step,
    })
}

/// One drawn chunk: which realm's row it rides, and its rung.
#[derive(Component)]
pub struct TerrainChunk {
    pub realm: RealmId,
    pub rung: u8,
}

/// A chunk's PROBE TWIN (slice 8p): the same mesh on the probe layer, drawn with the probe material.
/// Placed by the same system as its chunk; despawned with it.
#[derive(Component)]
pub struct ProbeTwin;

/// THE RULER BALL (slice 8p) and its probe twin: both carry this marker so one query places them.
#[derive(Component)]
pub struct RulerBall;

/// THE SUN: the one directional light, from the brightest luminous row.
#[derive(Component)]
pub struct TerrainSun;

/// The fill light opposite the sun — RETIRED (M8-L): a fill on the shadow side flattened every
/// picture; one light, one shadow. The marker stays so a stale entity of an older build is still
/// addressable by the query.
#[derive(Component)]
pub struct TerrainFill;

/// The ruler's entities: the visible ball and its probe twin, and the rung they were built for.
#[derive(Clone, Copy)]
struct RulerEntities {
    ball: Entity,
    twin: Entity,
    rung: u8,
}

/// The last ruler computation's inputs and answer: the march is re-run only when the eye or the
/// nose moves — a stand recomputes once.
struct RulerCache {
    eye_body: [f64; 3],
    forward_body: [f64; 3],
    ruler: Option<(Ruler, u8)>,
}

/// One realm's ladder: its view (the spans it read), the eye its wanted set was computed for, and
/// that wanted set.
#[derive(Default)]
struct RealmLadder {
    view: LadderView,
    eye: Option<[f64; 3]>,
    wanted: WantedSet,
    /// ★ THE HORIZON AS IT STANDS NOW, AND THE THREE RATES THAT READ IT (ruling F9 item 1): the
    /// pure pace, in the Tier-A library, where every arm of it is a unit test
    /// ([`vd_client::ask_pace::AskPace`]). The render crate only wires it: the descent reads
    /// `pace.asked()`, the crossfade's materials read `pace.drawn()`, and the slew moves
    /// `pace.held()` every frame.
    pace: AskPace,
    /// ★ THE EYE'S SPEED through this body, metres a second, as the bounded ask reads it (ruling
    /// F9 item 1): the LARGEST reading of the last `speed_hold_s` seconds
    /// ([`vd_client::ask_pace::PeakHold`]), because the lead's metres are a sawtooth whose mean is
    /// half the true speed. The hold is a true maximum over a window and reaches ZERO when the
    /// hull stops; the decay it replaced still read 194 m/s a second after a stop from 528.
    speed: PeakHold,
    /// The speed the hold last read, for the stamp.
    speed_mps: f64,
    /// ★ THE ROW'S GRACE (2026-09-14, the boarding cure): the display moment at which this
    /// realm last HAD a row in the drawn scene. A realm's row can be absent for a frame or two
    /// while the gateway re-composes a picture in a new origin, and forgetting the ladder there
    /// releases every chunk the realm holds — the whole band, rebuilt from nothing. The ladder
    /// is forgotten only once the row has been away longer than the interpolation buffer, which
    /// is the wire's own statement of how stale a delivered picture may legitimately be.
    /// `None` until the first frame the realm is seen with a row.
    row_seen_s: Option<f64>,
    /// ★ THE LAST DESCENT'S OWN SET (2026-09-16, the walk-gap measurement): the wanted set this
    /// ladder held before the newest descent replaced it, moved across for nothing. A missing
    /// urgent chunk reads its class here: `absent` means this very descent first wanted it, so no
    /// builder could have had it, and `urgent` means the ring asked earlier and a builder is late.
    prev: WantedSet,
    /// ★ THE DESCENT'S OWN PACE (the same measurement): the display moment of the last descent,
    /// how far the LEAD eye had moved from the last descent's before this frame judged it, the
    /// seconds since the last descent, and whether this frame re-cut the ring.
    last_descent_s: Option<f64>,
    drift_m: f64,
    since_descent_s: f64,
    descent_ran: bool,
}

/// One drawn chunk: its entity, its probe twin in Capture mode, and its morph counts (targets
/// that fell back to the field, vertices on a face seam, vertices), which the stamp sums.
struct Drawn {
    entity: Entity,
    twin: Option<Entity>,
    counts: [u64; 3],
    /// The mesh's bytes as the engine uploads them (M8-2): the vertex buffer and the indices.
    bytes: u64,
    /// The coarse caster this chunk asked for (the shadow ladder), released with it.
    caster: Option<ChunkKey>,
}

/// A coarse caster on the shadow layer (the shadow ladder): its entity and its bytes.
struct Caster {
    entity: Entity,
    bytes: u64,
}

/// Every placed chunk entity: the drawn ones, their probe twins and the sun's casters.
type PlacedChunks<'w, 's> = Query<
    'w,
    's,
    (
        &'static TerrainChunk,
        &'static ChunkOrigin,
        Option<&'static ProbeTwin>,
        Option<&'static ShadowCaster>,
        &'static mut Transform,
    ),
>;

/// THE CASTER MARKER: a chunk entity that exists for the sun alone. Placed by the chunks'
/// system like any chunk, never counted in the stamp's nearest and farthest.
#[derive(Component)]
pub struct ShadowCaster;

/// The bytes a mesh costs the engine: its vertex stride times its vertices, and its indices.
fn mesh_bytes(mesh: &Mesh) -> u64 {
    let indices = match mesh.indices() {
        Some(bevy::mesh::Indices::U16(v)) => v.len() * 2,
        Some(bevy::mesh::Indices::U32(v)) => v.len() * 4,
        None => 0,
    };
    mesh.get_vertex_size() * mesh.count_vertices() as u64 + indices as u64
}

/// The box the engine culls a chunk by, grown to wherever the vertex stage can put a vertex: its
/// own position, its morph target, and its position less its whole sink (the library's own
/// bounds, step 5).
fn moved_bounds(
    geometry: &vd_client::chunks::ChunkGeometry,
    margin_m: f32,
) -> bevy::camera::primitives::Aabb {
    let (lo, hi) = geometry.bounds;
    let m = Vec3::splat(margin_m);
    bevy::camera::primitives::Aabb::from_min_max(Vec3::from_array(lo) - m, Vec3::from_array(hi) + m)
}

/// The terrain's state on the engine side.
#[derive(Resource)]
pub struct Terrain {
    pub config: TerrainConfig,
    pub lane: ChunkLane,
    /// The declared world identity (the generator tag), stated on the stamp.
    declared: u64,
    /// Each drawn chunk.
    entities: BTreeMap<(RealmId, ChunkKey), Drawn>,
    /// The ladder per realm with a body.
    ladders: BTreeMap<RealmId, RealmLadder>,
    /// The ground's materials, one per realm and rung with that rung's crossfade bands, built on
    /// first use.
    materials: BTreeMap<(RealmId, u8), Handle<GroundMaterial>>,
    /// The probe materials, one per realm, kind and rung, built on first use.
    probe_materials: BTreeMap<(RealmId, u8, u8), Handle<ProbeMaterial>>,
    /// THE SHADOW LADDER'S state: the light-caster materials per realm and rung; how many
    /// drawn chunks want each coarse caster (a caster leaves with its last wanter); the casters
    /// on the shadow layer, and their bytes.
    shadow_materials: BTreeMap<(RealmId, u8), Handle<GroundMaterial>>,
    shadow_wanted: BTreeMap<(RealmId, ChunkKey), u32>,
    shadow_casters: BTreeMap<(RealmId, ChunkKey), Caster>,
    shadow_bytes: u64,
    /// The drawn chunks' morph counts, summed: fallbacks to the field, seam vertices, vertices.
    morph_totals: [u64; 3],
    /// The drawn chunks' mesh bytes, summed (M8-2's census).
    bytes_drawn: u64,
    /// ★ THE BAND'S LAST GAP (2026-09-16, the walk-gap measurement): the last frame whose band
    /// went incomplete, LATCHED. A gap lasts one frame and the stamp is polled every forty-five
    /// milliseconds, so the live count misses most of them.
    last_gap: Option<vd_devproto::DevBandGap>,
    /// THE FRAMES WITH A GAP: how many frames, since the start, drew with an urgent chunk
    /// missing. A gate reads the difference across a leg and misses no frame, where a poll at 20
    /// Hz sees one frame in three (refutation R4-6).
    urgent_frames: u64,
    /// The frames this system ran (M8-2a): the frame rate, against the build and harvest rates.
    frames: u64,
    /// ★ THE BOARDING INSTRUMENT (2026-09-14, the walk-aboard blank): how many times a realm's
    /// ladder was forgotten because that realm had no row in a frame's scene, and what the last
    /// forget cost. The release loop below a forget drops EVERY chunk of that realm, so a single
    /// frame without the planet's row rebuilds a pilot's whole band.
    ladders_forgotten: u64,
    last_forget: Option<vd_devproto::DevLadderForget>,
    /// ★ THE BOARDING INSTRUMENT, second half (2026-09-14): how many frames released more than
    /// [`WHOLESALE_RELEASE`] of ONE realm's chunks at once, and what the last such frame read. A
    /// forget is not the only way a band dies — a wanted set computed from an eye in the wrong
    /// frame releases every drawn chunk through the ordinary release loop below.
    band_releases: u64,
    last_release: Option<vd_devproto::DevBandRelease>,
    /// ★ THE FOREIGN EYE (2026-09-14, the boarding cure): the frames whose own pose was stated in
    /// a realm's frame OTHER than the one the picture is composed in. The eye is the own pose
    /// flattened, so on such a frame the eye is a number measured from another realm's centre —
    /// a pilot who has boarded reads a few metres from the hull's centre while the picture is
    /// still drawn from the planet's, and the eye lands at the planet's CENTRE. A wanted set
    /// computed there is the far view, and the whole band is released. The descent REFUSES such
    /// a frame (SL1 clause 6: a stale reading is refused, never used); the count says how long
    /// the disagreement lasts.
    eye_foreign_frames: u64,
    /// ★ THE BUILDERS' THROUGHPUT for the bounded ask (ruling F9 item 1): how many workers build,
    /// the smoothed CAPACITY in chunks a second (the worker count over the mean wall time of a
    /// build — never the chunks they happened to finish, which on a walk is the ask and not the
    /// ceiling), and the counter and the moment the last reading was taken at.
    workers: usize,
    throughput: Throughput,
    /// ★ THE CARD AS A SECOND BUILDER (ruling F9 item 2): the seam the client fills in once the
    /// renderer's device exists, and the meter both sides read. `attached` says whether a card
    /// builder is running at all, so a client with no trusted card grants nothing.
    card: CardSeam,
    card_attached: bool,
    /// THE WORST SINGLE FRAME of the last [`WORK_PEAK_S`] seconds: the terrain system times the
    /// gap between its own runs, which is the frame itself. The engine's own frame-time diagnostic
    /// is a SMOOTHED mean and hides one long frame; a second builder on the renderer's own device
    /// is judged by exactly that frame.
    frame_peak: PeakHold,
    frame_peak_ms: f64,
    /// The frames and the worst frame, shared with an instrument on another thread.
    frame_meter: Arc<FrameMeter>,
    /// This frame's own seconds, which is how far the deliverable horizons may slide (ruling F9
    /// item 1).
    frame_clock: FrameClock,
    /// ★ THE FRAME'S OWN WORK, named piece by piece (ruling F9 item 1's frame bar): for each
    /// piece the wall NANOSECONDS it has cost since the client started, how many times it RAN,
    /// and the worst SINGLE FRAME of it IN THE LAST [`WORK_PEAK_S`] SECONDS. A frame rate that
    /// falls with the bound on is either one of these pieces or none of them, and this is how the
    /// flight tells which — per leg, because the peak is a rolling window and not the run's own
    /// (review item 8).
    work_ns: BTreeMap<&'static str, WorkPiece>,
    /// The main thread's nanoseconds in the harvest loop since the start (M8-2a).
    harvest_nanos: u64,
    pub(crate) sun: Option<Entity>,
    /// ★ THE SKY'S INPUTS (slice 8s), listed every frame for `sky::sync_sky`: every body in the
    /// window with a charter, the brightest luminous row's luminosity, the sun disc last sized,
    /// the body whose air the camera holds.
    pub(crate) sky_bodies: Vec<crate::sky::SkyBody>,
    /// The brightest luminous row's luminosity and the EYE's distance to it, metres (the sun disc
    /// is the eye's own, S6).
    pub(crate) sun_star: Option<(f64, f64)>,
    pub(crate) sun_disk: Option<f32>,
    pub(crate) sky_realm: Option<RealmId>,
    /// The tangent of the sun's incidence at the eye, as the sun was last placed (capped as the
    /// shadow bias caps it): the shadow's reach for the casters reads it (item 18).
    sun_tan_i: Option<f32>,
    /// The ruler on screen, its shared assets, and its cached placement.
    ruler: Option<RulerEntities>,
    ruler_assets: Option<(Handle<Mesh>, Handle<StandardMaterial>)>,
    ruler_cache: Option<RulerCache>,
    /// This frame's stamp, assembled by `sync_terrain`, completed and published by `place_chunks`.
    pub(crate) stamp: Option<DevTerrainStamp>,
}

impl Terrain {
    /// The engine's terrain, for a client whose declared recipe tag is `declared`, over the threaded
    /// workers: every client draws the ladder of every body in its window (slice 8 step 2 — no flag).
    #[must_use]
    pub fn new(config: TerrainConfig, declared: u64) -> Terrain {
        let threads = if config.workers > 0 {
            config.workers
        } else {
            worker_share(std::thread::available_parallelism().map_or(4, |n| n.get()))
        };
        tracing::info!(threads, "terrain workers");
        // The done queue holds `DONE_QUEUE_FRAMES` frames of the harvest cap (item 19).
        let threaded =
            ThreadedWorkers::start(threads, config.harvest_per_frame * DONE_QUEUE_FRAMES);
        let card = threaded.card_seam();
        let workers: Box<dyn ChunkWorkers> = Box::new(threaded);
        let lane = ChunkLane::new(workers, declared);
        // The parent cache holds what the memory budget allows, in the meshes' own bytes, never
        // less than one working set of the workers (ruling V15; refutation T-8: a size that
        // followed the cores alone had no ceiling).
        lane.parents()
            .set_budget_bytes(config.parent_cache_bytes, threads * PARENTS_PER_CHUNK);
        Terrain {
            config,
            lane,
            declared,
            entities: BTreeMap::new(),
            shadow_materials: BTreeMap::new(),
            shadow_wanted: BTreeMap::new(),
            shadow_casters: BTreeMap::new(),
            shadow_bytes: 0,
            ladders: BTreeMap::new(),
            materials: BTreeMap::new(),
            probe_materials: BTreeMap::new(),
            morph_totals: [0; 3],
            bytes_drawn: 0,
            last_gap: None,
            urgent_frames: 0,
            frames: 0,
            ladders_forgotten: 0,
            last_forget: None,
            band_releases: 0,
            last_release: None,
            eye_foreign_frames: 0,
            workers: threads,
            throughput: Throughput::default(),
            card,
            card_attached: false,
            frame_peak: PeakHold::default(),
            frame_peak_ms: 0.0,
            frame_meter: Arc::new(FrameMeter::default()),
            frame_clock: FrameClock::default(),
            work_ns: BTreeMap::new(),
            harvest_nanos: 0,
            sun: None,
            sky_bodies: Vec::new(),
            sun_star: None,
            sun_disk: None,
            sky_realm: None,
            sun_tan_i: None,
            ruler: None,
            ruler_assets: None,
            ruler_cache: None,
            stamp: None,
        }
    }

    /// ONE PIECE'S WALL TIME THIS FRAME, added to its running total, and offered to its rolling
    /// peak (the frame bar's instrument).
    fn note_work(&mut self, name: &'static str, dt: std::time::Duration, runs: u64, now_s: f64) {
        let ns = dt.as_nanos() as u64;
        let piece = self.work_ns.entry(name).or_default();
        piece.total_ns += ns;
        piece.runs += runs;
        piece.peak_ns = piece.peak.read(ns as f64, now_s, WORK_PEAK_S);
    }

    /// THE FRAME'S WORK as the stamp states it: the piece, its nanoseconds in all, its worst
    /// single frame of the last [`WORK_PEAK_S`] seconds, and how many times it ran.
    fn frame_work(&self) -> Vec<(String, u64, u64, u64)> {
        self.work_ns
            .iter()
            .map(|(name, piece)| {
                (
                    (*name).to_owned(),
                    piece.total_ns,
                    piece.peak_ns as u64,
                    piece.runs,
                )
            })
            .collect()
    }

    /// ★ THE BUILDERS' THROUGHPUT, read as a CAPACITY (ruling F9 item 1, and ruling F9 item 2's
    /// second builder): the pure reader in the Tier-A library
    /// ([`vd_client::ask_pace::Throughput`]) over the lane's own counters, PLUS the card's own
    /// capacity inside its time budget ([`vd_client::card_budget::CardBudget`]). The bounded ask
    /// sizes its horizon against everything that builds, so a card that builds widens the horizon
    /// and a card that does not leaves it exactly where it was.
    fn read_throughput(&mut self, now_s: f64) -> f64 {
        let built = self.lane.built();
        self.throughput
            .set_card(if self.card_attached & self.config.gpu_bound {
                self.card.meter.capacity_per_s()
            } else {
                0.0
            });
        self.throughput.read(
            built.chunks,
            built.nanos,
            now_s,
            self.workers,
            self.config.throughput_window_s,
        )
    }

    /// ★ THE CARD BECOMES A BUILDER (ruling F9 item 2): the renderer's own device and queue, once
    /// they exist and once the self-check has TRUSTED them. A card that is not trusted builds
    /// nothing; so does `VD_TERRAIN_GPU=0`; the CPU share is the default and the fallback.
    pub fn attach_card(&mut self, device: wgpu::Device, queue: wgpu::Queue, trusted: bool) {
        if !self.config.gpu_card {
            tracing::info!("THE CARD BUILDS NOTHING: VD_TERRAIN_GPU is off");
            return;
        }
        if !trusted {
            tracing::warn!(
                "THE CARD BUILDS NOTHING: the recipe's self-check did not trust this GPU — the \
                 CPU workers build the whole ladder"
            );
            return;
        }
        let skip = self.config.gpu_skip.unwrap_or(CARD_SKIP);
        let lanes = self.config.gpu_flights;
        self.card.attach(
            device,
            queue,
            skip,
            self.config.harvest_per_frame * DONE_QUEUE_FRAMES,
            lanes,
            self.config.gpu_verify,
        );
        self.card_attached = true;
        tracing::info!(
            budget = self.config.gpu_budget,
            skip,
            lanes,
            summed = self.config.gpu_bound,
            "THE CARD IS A SECOND BUILDER: it takes chunks from the same queue by the same \
             priority, inside its share of every frame, while the queue is deep enough to want it"
        );
    }

    /// The frames and the worst frame, for an instrument on another thread (the seam probe).
    #[must_use]
    pub fn frame_meter(&self) -> Arc<FrameMeter> {
        Arc::clone(&self.frame_meter)
    }

    /// ★ THE FRAME JUDGES THE QUEUE FOR THE CARD (the owner's step after Step 15): the pure
    /// stand-down rule ([`vd_client::card_gate::QueueDepth`]) over the readings the bounded ask
    /// already holds — the requests the lane is waiting on, the CPU WORKERS' own capacity, the
    /// eye's delivered speed and the lead the client asks ahead by. Below the depth the card takes
    /// nothing, so a still stand's short queue is the CPU workers' alone.
    ///
    /// The speed and the lead are the FASTEST body's: the eye is inside one realm, and a queue is
    /// urgent because THAT realm's ground is on its way to the screen.
    fn judge_queue(&mut self, workers_per_s: f64, speed_mps: f64, lead_m: f64) {
        if !self.card_attached {
            return;
        }
        self.card.meter.judge(vd_client::card_gate::QueueDepth {
            pending: self.lane.pending_count(),
            workers_per_s,
            speed_mps,
            lead_m,
        });
    }

    /// THE FRAME GRANTS THE CARD its share, and the frame's own worst reading is kept. The answer
    /// is what the stamp states about the card.
    fn grant_card(
        &mut self,
        frame_s: f64,
        now_s: f64,
    ) -> (u64, u64, u64, f64, f64, f64, bool, u64, u64) {
        self.frame_peak_ms = self.frame_peak.read(frame_s * 1.0e3, now_s, WORK_PEAK_S);
        self.frame_meter.peak_ns.store(
            (self.frame_peak_ms * 1.0e6) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        if self.card_attached {
            self.card.meter.grant(frame_s, self.config.gpu_budget);
        }
        let read = self.card.meter.read();
        let (judged, stood_down) = self.card.meter.stand_down();
        (
            read.0, read.1, read.2, read.3, read.4, read.5, read.6, judged, stood_down,
        )
    }

    /// ★ THE CROSSFADE FOLLOWS THE BOUND (ruling F9 item 1): a realm whose deliverable horizons
    /// moved rewrites its own materials in place — the ground's, the probe's and the caster's —
    /// because all three carry the rung's bands as a uniform read from the same effective switch
    /// distances the descent asks at. Nothing is rebuilt and nothing is respawned.
    fn rebind_bands(
        &mut self,
        realm: RealmId,
        body: &vd_terrain::BodyDefinition,
        ground: &mut Assets<GroundMaterial>,
        probes: Option<&mut Assets<ProbeMaterial>>,
    ) {
        let rungs = body.ladder().rungs;
        let Some(ladder) = self.ladders.get(&realm) else {
            return;
        };
        let bound = ladder.pace.drawn().clone();
        let coarse_step = self.config.shadow_coarse_step;
        for ((_, rung), handle) in self.materials.range((realm, 0u8)..=(realm, u8::MAX)) {
            if let Some(m) = ground.get_mut(handle) {
                m.extension.set_bands(
                    bound.fade_bands(*rung, rungs),
                    bound.sink_end_m(body, *rung, rungs),
                );
            }
        }
        // A caster's bands are the DRAWN rung's — the rung it casts for (§24.5) — so they follow
        // the bound at that rung, not at the caster's own.
        for ((_, rung), handle) in self.shadow_materials.range((realm, 0u8)..=(realm, u8::MAX)) {
            let drawn = rung.saturating_sub(coarse_step);
            if let Some(m) = ground.get_mut(handle) {
                m.extension.set_bands(
                    bound.fade_bands(drawn, rungs),
                    bound.sink_end_m(body, drawn, rungs),
                );
            }
        }
        if let Some(probes) = probes {
            for ((_, kind, rung), handle) in self
                .probe_materials
                .range((realm, 0u8, 0u8)..=(realm, u8::MAX, u8::MAX))
            {
                if *kind != PROBE_KIND_TERRAIN {
                    continue;
                }
                if let Some(m) = probes.get_mut(handle) {
                    m.set_bands(
                        bound.fade_bands(*rung, rungs),
                        bound.sink_end_m(body, *rung, rungs),
                    );
                }
            }
        }
    }

    /// The probe material for a kind at a rung, built once: the terrain's carries the rung's
    /// crossfade bands, the ruler's none (it is never faded).
    fn probe_material(
        &mut self,
        assets: &mut Assets<ProbeMaterial>,
        realm: RealmId,
        kind: u8,
        rung: u8,
        body: &vd_terrain::BodyDefinition,
    ) -> Handle<ProbeMaterial> {
        let bound = self.bound_of(realm);
        self.probe_materials
            .entry((realm, kind, rung))
            .or_insert_with(|| {
                let rungs = body.ladder().rungs;
                let (bands, sink_end, sink) = if kind == PROBE_KIND_TERRAIN {
                    (
                        bound.fade_bands(rung, rungs),
                        bound.sink_end_m(body, rung, rungs),
                        vd_client::chunks::sink_m(body, rung),
                    )
                } else {
                    ((FADE_ALWAYS_IN, FADE_ALWAYS_OUT), FADE_ALWAYS_IN[1], 0.0)
                };
                assets.add(ProbeMaterial::new(kind, rung, bands, sink_end, sink))
            })
            .clone()
    }

    /// The ground's material for a realm's rung, built once with the rung's crossfade bands.
    fn ground_material(
        &mut self,
        assets: &mut Assets<GroundMaterial>,
        realm: RealmId,
        rung: u8,
        body: &vd_terrain::BodyDefinition,
    ) -> Handle<GroundMaterial> {
        let bound = self.bound_of(realm);
        self.materials
            .entry((realm, rung))
            .or_insert_with(|| {
                let rungs = body.ladder().rungs;
                assets.add(GroundMaterial {
                    base: StandardMaterial {
                        base_color: Color::srgb(0.55, 0.50, 0.42),
                        perceptual_roughness: 0.95,
                        metallic: 0.0,
                        cull_mode: Some(bevy::render::render_resource::Face::Back),
                        // Masked, never cut (the paint is opaque): the mask makes the engine run
                        // the extension's prepass fragment stage, which ends the rung past its
                        // far edge in the shadow map too.
                        alpha_mode: AlphaMode::Mask(0.5),
                        ..default()
                    },
                    extension: LadderFade::new(
                        bound.fade_bands(rung, rungs),
                        bound.sink_end_m(body, rung, rungs),
                        vd_client::chunks::sink_m(body, rung),
                        f64::from(vd_seed::ladder::cell_m(rung)),
                    ),
                })
            })
            .clone()
    }

    /// THE BOUNDED ASK in force for a realm (ruling F9 item 1): the tier rule's own radii while
    /// the realm has no ladder yet.
    fn bound_of(&self, realm: RealmId) -> AskBound {
        self.ladders
            .get(&realm)
            .map_or_else(AskBound::unbounded, |l| l.pace.drawn().clone())
    }

    /// THE CASTER'S MATERIAL for a realm and rung (the shadow ladder): the rung's own material as
    /// a light caster, sunk by the two rungs' bound — the recipe's own bound between the drawn
    /// rung and the caster's (`dropped_bound_m`), plus a cell of each for the extractors'
    /// placement — so the drawn ground never stands under its caster.
    /// THE CASTER'S SINK for a caster at `rung`: the recipe's own bound between the caster's rung
    /// and the drawn rung it casts for (`dropped_bound_m`, the octaves the coarser rung drops),
    /// plus a cell of each for the extractors' placement.
    fn caster_sink_m(&self, body: &vd_terrain::BodyDefinition, rung: u8) -> f64 {
        let fine = rung.saturating_sub(self.config.shadow_coarse_step);
        body.dropped_bound_m(rung) - body.dropped_bound_m(fine)
            + f64::from(vd_seed::ladder::cell_m(rung))
            + f64::from(vd_seed::ladder::cell_m(fine))
    }

    fn shadow_material(
        &mut self,
        assets: &mut Assets<GroundMaterial>,
        realm: RealmId,
        rung: u8,
        body: &vd_terrain::BodyDefinition,
    ) -> Handle<GroundMaterial> {
        let bound_m = self.caster_sink_m(body, rung);
        // THE CASTER'S CROSSFADE (2026-09-12): the caster's bands are the DRAWN rung's — the rung it
        // casts for, `coarse_step` below — so its sink scales with that rung's wholeness in the
        // shadow pass and reaches zero where the finer chunk has morphed onto this surface. MEASURED
        // before it (the pop detector): the caster left unsunk in one frame at the handover, a step
        // of about 50 levels along the shadow's edge.
        let drawn_rung = rung.saturating_sub(self.config.shadow_coarse_step);
        let bound = self.bound_of(realm);
        self.shadow_materials
            .entry((realm, rung))
            .or_insert_with(|| {
                let rungs = body.ladder().rungs;
                assets.add(GroundMaterial {
                    base: StandardMaterial {
                        base_color: Color::srgb(0.55, 0.50, 0.42),
                        perceptual_roughness: 0.95,
                        metallic: 0.0,
                        cull_mode: Some(bevy::render::render_resource::Face::Back),
                        alpha_mode: AlphaMode::Mask(0.5),
                        ..default()
                    },
                    extension: LadderFade::new(
                        bound.fade_bands(drawn_rung, rungs),
                        bound.sink_end_m(body, drawn_rung, rungs),
                        vd_client::chunks::sink_m(body, rung),
                        f64::from(vd_seed::ladder::cell_m(rung)),
                    )
                    .into_light_caster(bound_m),
                })
            })
            .clone()
    }

    /// THE SHADOW LADDER'S INVARIANT: one key, one residency in the lane. A key the ladder DRAWS
    /// casts itself while some finer chunk wants it as a caster, and is a non-caster otherwise
    /// (its own coarse caster casts for it). A key the ladder does not draw is built as a coarse
    /// caster on the shadow layer while wanted. When the ladder comes to want a key held as a
    /// caster, the caster is dropped and the key rebuilt as a drawn chunk (`convert_casters`).
    /// (Refutation of the ladder, findings 1–3: a caster and a drawn chunk shared one residency
    /// slot, so a crossfade band lost its coarse rung or leaked an entity.)
    ///
    /// A finer chunk asks for its caster: the count rises; on the first want a drawn key starts
    /// casting itself, else the key is requested unless a caster already stands.
    fn want_caster(&mut self, commands: &mut Commands, realm: RealmId, ckey: ChunkKey) {
        let n = self.shadow_wanted.entry((realm, ckey)).or_insert(0);
        *n += 1;
        if *n > 1 {
            return;
        }
        if let Some(drawn) = self.entities.get(&(realm, ckey)) {
            commands
                .entity(drawn.entity)
                .remove::<bevy::light::NotShadowCaster>();
        } else if !self.shadow_casters.contains_key(&(realm, ckey)) {
            self.lane.request(realm, ckey, shadow_priority(ckey.rung));
        }
    }

    /// A finer chunk leaves: the count falls; at zero a drawn key stops casting itself and a
    /// caster is despawned and released — the lane's residency of a DRAWN key is never touched.
    fn unwant_caster(&mut self, commands: &mut Commands, realm: RealmId, ckey: ChunkKey) {
        match self.shadow_wanted.get_mut(&(realm, ckey)) {
            Some(n) if *n > 1 => {
                *n -= 1;
                return;
            }
            Some(_) => {}
            None => return,
        }
        self.shadow_wanted.remove(&(realm, ckey));
        if let Some(drawn) = self.entities.get(&(realm, ckey)) {
            commands
                .entity(drawn.entity)
                .insert(bevy::light::NotShadowCaster);
            return;
        }
        if let Some(c) = self.shadow_casters.remove(&(realm, ckey)) {
            commands.entity(c.entity).despawn();
            self.shadow_bytes -= c.bytes;
        }
        self.lane.release(realm, ckey);
    }

    /// The chunks on screen per rung, finest first.
    fn drawn_per_rung(&self) -> Vec<(u8, u64)> {
        let mut counts: BTreeMap<u8, u64> = BTreeMap::new();
        for (_, key) in self.entities.keys() {
            *counts.entry(key.rung).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }
}

/// A realm box's look as the boundary the lane wants.
fn look_of(rbox: &RealmBox) -> Boundary {
    match rbox.shape {
        BoxShape::Sphere { r } => Boundary::Shell { r },
        BoxShape::Box { half } => Boundary::Aabb { half },
    }
}

/// The row's facing as a rotation.
fn facing_of(rbox: &RealmBox) -> DQuat {
    // At the box's own precision (f64 since step 6: a narrowed facing on a 6 371 km lever moved
    // the stamped eye by 0.7 m between two frames of a turning hull).
    DQuat::from_array(rbox.facing).normalize()
}

/// The triangles' indices at 16 bits where the vertices fit (step 5: exact, half the bytes), else
/// at 32.
fn packed_indices(geometry: &vd_client::chunks::ChunkGeometry) -> bevy::mesh::Indices {
    let flat = geometry.triangles.iter().flatten().copied();
    if geometry.vertices.len() <= usize::from(u16::MAX) {
        bevy::mesh::Indices::U16(flat.map(|i| i as u16).collect())
    } else {
        bevy::mesh::Indices::U32(flat.collect())
    }
}

/// A Bevy mesh from a chunk's geometry: positions and normals relative to the chunk's origin, the
/// morph metre per vertex, the extractor's triangles as indices. `flat` duplicates the vertices
/// and takes one normal per face.
fn mesh_of(
    geometry: &vd_client::chunks::ChunkGeometry,
    flat: bool,
    exact_normal_rung: u8,
    splat: bool,
) -> Mesh {
    if splat {
        return splat_mesh_of(geometry, exact_normal_rung);
    }
    // RENDER WORLD ONLY (step 5, refutation P-16): the engine keeps a mesh in the main world too
    // by default, and nothing reads a chunk's mesh back on the client — the culling box is the
    // library's own and the geometry stays on the lane. One copy, in the render world.
    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, geometry.vertices.clone())
    .with_inserted_attribute(super::ATTRIBUTE_MORPH, geometry.morph_m.clone())
    .with_inserted_attribute(super::ATTRIBUTE_RADIAL, geometry.radials.clone())
    .with_inserted_attribute(
        super::ATTRIBUTE_MORPH_NORMAL,
        bevy::mesh::VertexAttributeValues::Snorm16x2(geometry.morph_normals.clone()),
    )
    .with_inserted_indices(packed_indices(geometry));
    // THE NORMAL BY RUNG (ruling V18, `TerrainConfig::exact_normal_rung`): the worker's packed
    // four bytes on the near rungs, the engine's twelve on the far ones; nothing is packed here
    // (refutation N-2: the harvest loop is the main thread's).
    let packed = geometry.key.rung < exact_normal_rung;
    if flat {
        // THE FLAT LOOK (a dev switch): the engine recomputes a normal per face on the main
        // thread, and the packing follows it here — the one place the encoder runs off the
        // worker, by the switch's own nature.
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, geometry.normals.clone());
        mesh.duplicate_vertices();
        mesh.compute_flat_normals();
        if packed {
            let flat_normals: Vec<[i16; 2]> = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
                Some(bevy::mesh::VertexAttributeValues::Float32x3(v)) => {
                    v.iter().map(|n| oct_encode(*n)).collect()
                }
                other => panic!("the flat normals are the engine's Float32x3, not {other:?}"),
            };
            mesh.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
            mesh.insert_attribute(
                super::ATTRIBUTE_OCT_NORMAL,
                bevy::mesh::VertexAttributeValues::Snorm16x2(flat_normals),
            );
        }
    } else if packed {
        mesh.insert_attribute(
            super::ATTRIBUTE_OCT_NORMAL,
            bevy::mesh::VertexAttributeValues::Snorm16x2(geometry.packed_normals.clone()),
        );
    } else {
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, geometry.normals.clone());
    }
    mesh
}

/// THE SPLAT MESH (D8-8's measurement): every surface vertex four times, with the corner it
/// stands at; the shader spreads the four into a camera-facing square one cell wide. The
/// normal, the morph and the radial ride along unchanged, so the crossfade and the light are
/// the mesh's own; the skirt vertices are left out (a splat has no edge to hide).
fn splat_mesh_of(geometry: &vd_client::chunks::ChunkGeometry, exact_normal_rung: u8) -> Mesh {
    let n = geometry.skirt_start as usize;
    let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let mut positions = Vec::with_capacity(n * 4);
    let mut morph = Vec::with_capacity(n * 4);
    let mut radials = Vec::with_capacity(n * 4);
    let mut corner = Vec::with_capacity(n * 4);
    let mut packed = Vec::with_capacity(n * 4);
    let mut normals = Vec::with_capacity(n * 4);
    let mut morph_normals = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);
    for i in 0..n {
        let base = (i * 4) as u32;
        for c in corners {
            positions.push(geometry.vertices[i]);
            morph.push(geometry.morph_m[i]);
            radials.push(geometry.radials[i]);
            corner.push(c);
            packed.push(geometry.packed_normals[i]);
            normals.push(geometry.normals[i]);
            morph_normals.push(geometry.morph_normals[i]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(super::ATTRIBUTE_MORPH, morph)
    .with_inserted_attribute(super::ATTRIBUTE_RADIAL, radials)
    .with_inserted_attribute(super::ATTRIBUTE_SPLAT_CORNER, corner)
    .with_inserted_attribute(
        super::ATTRIBUTE_MORPH_NORMAL,
        bevy::mesh::VertexAttributeValues::Snorm16x2(morph_normals),
    )
    .with_inserted_indices(bevy::mesh::Indices::U32(indices));
    if geometry.key.rung < exact_normal_rung {
        mesh.insert_attribute(
            super::ATTRIBUTE_OCT_NORMAL,
            bevy::mesh::VertexAttributeValues::Snorm16x2(packed),
        );
    } else {
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    }
    mesh
}

/// THE FRAME'S ANATOMY (D8-8): every render pass the engine timed this frame, by name, with its
/// CPU milliseconds (encoding) and its GPU milliseconds (running; zero without timestamps) —
/// smoothed, from the render diagnostics' `render/<pass>/elapsed_cpu|elapsed_gpu` paths.
fn render_passes_ms(store: &bevy::diagnostic::DiagnosticsStore) -> Vec<(String, f32, f32)> {
    let mut by_pass: BTreeMap<String, (f32, f32)> = BTreeMap::new();
    for d in store.iter() {
        let path = d.path().as_str();
        let Some(rest) = path.strip_prefix("render/") else {
            continue;
        };
        let Some((pass, field)) = rest.rsplit_once('/') else {
            continue;
        };
        let v = d.smoothed().unwrap_or(0.0) as f32;
        let e = by_pass.entry(pass.to_owned()).or_insert((0.0, 0.0));
        match field {
            "elapsed_cpu" => e.0 = v,
            "elapsed_gpu" => e.1 = v,
            _ => {}
        }
    }
    by_pass.into_iter().map(|(p, (c, g))| (p, c, g)).collect()
}

/// The default first rung whose meshes keep the engine's exact normal (ruling V18, MEASURED on
/// the orbit stand: `TerrainConfig::exact_normal_rung`). Cells of 512 m and up: the limb of a
/// body seen from high.
pub const EXACT_NORMAL_RUNG: u8 = 9;

/// The two terrain lights' transforms: the sun and its fill, re-aimed together every frame.
type LightQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Transform,
    (
        Or<(With<TerrainSun>, With<TerrainFill>)>,
        Without<TerrainChunk>,
        Without<RulerBall>,
    ),
>;

/// The picture's camera transform, read for the nose (the stamp's star angles and the ruler's ray).
type CameraQuery<'w, 's> = Query<
    'w,
    's,
    &'static Transform,
    (
        With<super::FollowCam>,
        Without<TerrainChunk>,
        Without<TerrainSun>,
        Without<TerrainFill>,
        Without<RulerBall>,
    ),
>;

/// The ruler ball and its twin, placed every frame.
type RulerQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Transform,
    (
        With<RulerBall>,
        Without<TerrainChunk>,
        Without<TerrainSun>,
        Without<TerrainFill>,
    ),
>;

/// A render-frame f64 point narrowed once into a transform at a scale.
fn placed(p: DVec3, scale: f64) -> Transform {
    Transform::from_translation(Vec3::new(p.x as f32, p.y as f32, p.z as f32))
        .with_scale(Vec3::splat(scale as f32))
}

/// The own entity's drawn point among composited poses (at any cursor), if it is among them.
fn own_world<S>(
    snap: &vd_client::render_snapshot::RenderSnapshot,
    own: Option<vd_core::EntityId>,
    poses: &[(vd_core::EntityId, S, vd_client::interp::RenderPose)],
) -> Option<DVec3> {
    poses
        .iter()
        .find(|(id, _, _)| Some(*id) == own)
        .map(|(_, _, p)| snap.world_pos(p))
}

/// The frame the own entity's pose is stated in at a cursor, as a label — `"-"` when the own
/// entity has no pose among the composited ones there (the boarding instrument's frame pair).
fn own_frame_of<S>(
    own: Option<vd_core::EntityId>,
    poses: &[(vd_core::EntityId, S, vd_client::interp::RenderPose)],
) -> String {
    poses
        .iter()
        .find(|(id, _, _)| Some(*id) == own)
        .map_or_else(|| "-".to_owned(), |(_, _, p)| format!("{:?}", p.frame))
}

/// Whether the own entity's pose at a cursor is stated in the realm the picture is composed in —
/// the eye and the rows measured from ONE centre. `true` when the own entity has no pose there or
/// the picture names no origin (nothing to disagree with). See the refusal in [`sync_terrain`].
fn eye_at_home<S>(
    own: Option<vd_core::EntityId>,
    poses: &[(vd_core::EntityId, S, vd_client::interp::RenderPose)],
    picture_realm: Option<RealmId>,
) -> bool {
    let stated = poses
        .iter()
        .find(|(id, _, _)| Some(*id) == own)
        .map(|(_, _, p)| p.frame.realm());
    match (stated, picture_realm) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

/// ★ A WHOLESALE RELEASE (2026-09-14): more than this many of ONE realm's chunks released in one
/// frame is not a moving eye trimming its band — it is the band itself going. The instrument
/// names such a frame; the flight reads it.
const WHOLESALE_RELEASE: u64 = 64;

/// ★ HOW MANY MISSING CHUNKS A GAP ROW NAMES (2026-09-16, the walk-gap measurement): the walk's
/// worst gap read two, a hull's at 528 m/s read thousands, and a stamp is a diagnostic message on
/// one connection. Eight names the walk's gap whole and bounds the hull's row.
const GAP_KEYS_CAP: usize = 8;

/// A body in the window with the two eyes in its frame: the DRAWN eye (the stamp's, the ruler's)
/// and the LEAD eye (the wanted set's), both in metres from the body's centre.
struct EyeBody {
    realm: RealmId,
    body: Arc<vd_terrain::BodyDefinition>,
    eye: [f64; 3],
    lead: [f64; 3],
}

/// Whether the eye has moved more than [`EYE_STEP_M`] from the last one.
fn moved(last: Option<[f64; 3]>, eye: [f64; 3]) -> bool {
    last.is_none_or(|l| (DVec3::from_array(l) - DVec3::from_array(eye)).length() > EYE_STEP_M)
}

/// THE TERRAIN SYSTEM: runs after the realm boxes are placed, with the same eye.
#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources/queries
pub(crate) fn sync_terrain(
    net: Res<super::Net>,
    render_eye: Res<super::RenderEye>,
    mut terrain: ResMut<Terrain>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ground_materials: ResMut<Assets<GroundMaterial>>,
    mut probe_materials: Option<ResMut<Assets<ProbeMaterial>>>,
    mut commands: Commands,
    mut light_tf: LightQuery,
    mut ruler_tf: RulerQuery,
    cam: CameraQuery,
    boxes: Res<super::RealmBoxEntities>,
    mut visibility: Query<&mut Visibility>,
    key_light: Query<Entity, With<super::KeyLight>>,
    exposure: Query<&bevy::camera::Exposure, With<super::FollowCam>>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) {
    // Every frame this system runs (M8-2a; refutation T-15: counted only under a body before).
    terrain.frames += 1;
    terrain
        .frame_meter
        .frames
        .store(terrain.frames, std::sync::atomic::Ordering::Relaxed);
    // THE FRAME'S MOMENT: the sample the camera was placed from (step 6), never a second one.
    let Some((now_s, snap)) = render_eye.moment.clone() else {
        terrain.stamp = None;
        return;
    };
    let scene = snap.scene_now(now_s);
    // THE LEAD (slice 8 step 4, the residency band): the scene and the own eye at the LEAD
    // cursor — the freshest delivered moment, one interpolation buffer ahead of what the picture
    // draws. The wanted set is computed for the eye THERE, so every chunk is asked for one
    // buffer before the picture needs it, from delivered data alone (never a speed the client
    // derived, SL10 clause 7). The offset from the drawn eye to the lead eye is a difference of
    // two delivered poses at two cursors; the row's centre at the lead cursor is the lead scene's.
    let lead_cursor = snap.lead_cursor(now_s);
    let scene_lead = lead_cursor.map(|c| snap.scene_at(c));
    let own = snap.own_entity();
    let rendered_now = snap.rendered(now_s);
    let rendered_lead = lead_cursor.map(|c| snap.rendered_at(c));
    let own_now = own_world(&snap, own, &rendered_now);
    let own_lead = rendered_lead
        .as_ref()
        .and_then(|r| own_world(&snap, own, r));
    let lead_offset = match (own_now, own_lead) {
        (Some(a), Some(b)) => b - a,
        _ => DVec3::ZERO,
    };
    // ★ THE BOARDING INSTRUMENT (2026-09-14): the frame the own pose is DELIVERED in at each of
    // the two cursors. A boarding pilot's drawn pose and lead pose may be stated in two different
    // realms' frames for a frame — the planet's and the hull's — and a descent that reads the
    // pair computes a wanted set for an eye that stands nowhere.
    let own_frame_now = own_frame_of(own, &rendered_now);
    let own_frame_lead = rendered_lead
        .as_ref()
        .map_or_else(|| "-".to_owned(), |r| own_frame_of(own, r));
    // ★ THE EYE MUST STAND IN THE PICTURE'S OWN FRAME (2026-09-14, the boarding cure; SL1 clause
    // 6: a stale reading is REFUSED, never used). The eye is the own pose flattened, and the
    // rows are composed in the origin realm's frame. Between a pilot's crossing and the level
    // that carries the new origin the two disagree for a beat: the pose already reads a few
    // metres from the hull's centre while the picture is still drawn from the planet's, so the
    // eye lands at the PLANET'S CENTRE, the descent runs with an altitude of minus six thousand
    // kilometres, and the wanted set it computes holds none of the 7 288 chunks that are drawn —
    // the release loop below then drops every one of them. A frame whose eye is not a reading in
    // the picture's own frame recomputes NOTHING: the wanted set of the last agreeing frame
    // stands, the band stays whole, and the next agreeing frame moves it.
    let picture_realm = snap.origin();
    let eye_at_home_now = eye_at_home(own, &rendered_now, picture_realm);
    let eye_at_home_lead = rendered_lead
        .as_ref()
        .is_none_or(|r| eye_at_home(own, r, picture_realm));
    let eye_refused = !(eye_at_home_now & eye_at_home_lead);
    terrain.eye_foreign_frames += u64::from(eye_refused);
    // 1. Every stated surface becomes a body (once); the eye-relative centre of every row, drawn
    //    and at the lead.
    let mut centres: BTreeMap<RealmId, (DVec3, DQuat)> = BTreeMap::new();
    let mut lead_centres: BTreeMap<RealmId, (DVec3, DQuat)> = BTreeMap::new();
    let mut brightest: Option<(f64, DVec3)> = None;
    let mut charters: BTreeMap<RealmId, vd_core::look::BodyCharter> = BTreeMap::new();
    for (realm, rbox) in scene.iter() {
        if let Some(charter) = rbox.charter {
            charters.insert(realm, charter);
        }
        let draw_center = super::draw_center_of(rbox, &render_eye, &snap, now_s);
        centres.insert(realm, (draw_center, facing_of(rbox)));
        if let Some(lead_box) = scene_lead.as_ref().and_then(|s| s.get(realm)) {
            let lead_center = super::draw_center_of(lead_box, &render_eye, &snap, now_s);
            lead_centres.insert(realm, (lead_center, facing_of(lead_box)));
        }
        if let Some(surface) = rbox.surface {
            terrain
                .lane
                .state_surface(realm, &surface, rbox.charter.as_ref(), &look_of(rbox));
        }
        // ★ THE ARTIFACT the realm shipped (slice 8c stage C4c), as the client's book holds it
        // this frame: the lane refreshes its pointer when a level or a tile landed.
        if let Some(cache) = snap.artifact(realm) {
            terrain.lane.state_artifact(realm, cache);
        }
        if let Some((_, lux)) = rbox.luma
            && brightest.is_none_or(|(b, _)| lux > b)
        {
            brightest = Some((lux, draw_center));
        }
    }
    // 2. THE WANTED SET per realm with a body: the ladder from the LEAD eye to the horizon,
    //    coarsest first, recomputed when that eye has moved. The drawn eye (the stamp's, the
    //    ruler's) is the second point of the pair.
    let with_bodies: Vec<EyeBody> = centres
        .iter()
        .filter_map(|(realm, (centre, facing))| {
            terrain.lane.body(*realm).map(|b| {
                let eye_body = body_frame_point(
                    [0.0, 0.0, 0.0],
                    [centre.x, centre.y, centre.z],
                    [facing.x, facing.y, facing.z, facing.w],
                );
                let (lead_centre, lead_facing) = lead_centres
                    .get(realm)
                    .copied()
                    .unwrap_or((*centre, *facing));
                let eye_lead = body_frame_point(
                    [lead_offset.x, lead_offset.y, lead_offset.z],
                    [lead_centre.x, lead_centre.y, lead_centre.z],
                    [lead_facing.x, lead_facing.y, lead_facing.z, lead_facing.w],
                );
                EyeBody {
                    realm: *realm,
                    body: Arc::clone(b),
                    eye: eye_body,
                    lead: eye_lead,
                }
            })
        })
        .collect();
    // ★ THE SKY'S INPUTS (slice 8s): every body with a charter, its centre in the render frame
    // (the same centre the chunks are placed from), for `sky::sync_sky` after this system.
    terrain.sky_bodies = with_bodies
        .iter()
        .filter_map(|eb| {
            let charter = *charters.get(&eb.realm)?;
            let (centre, _) = centres.get(&eb.realm)?;
            Some(crate::sky::SkyBody {
                realm: eb.realm,
                centre: *centre,
                radius_m: eb.body.ladder().radius_m(),
                charter,
            })
        })
        .collect();
    terrain.sun_star = brightest.map(|(lux, centre)| (lux, centre.length()));
    // THE SHADOW'S REACH for the casters (item 18): the cascades' reach, the sun's tangent as it
    // was last placed (the cap's worth before the sun is born: the longest shadows, so no caster
    // is missed), and the ladder's step. A change past the hysteresis recomputes the wanted set.
    let shadow = shadow_reach(&terrain.config, terrain.sun_tan_i);
    // ★ THE BOUNDED ASK (ruling F9 item 1): the builders' measured capacity, and the eye's own
    // speed through each body — the LEAD offset over the buffer's own seconds, both of them
    // differences of DELIVERED poses (SL10 clause 7: the client never coasts a pose forward).
    let work_started = std::time::Instant::now();
    let throughput = terrain.read_throughput(now_s);
    let frame_s = terrain.frame_clock.seconds(now_s);
    // ★ THE CARD'S SHARE OF THIS FRAME (ruling F9 item 2), and the frame's own worst reading.
    let card = terrain.grant_card(frame_s, now_s);
    let work_throughput = work_started.elapsed();
    let lead_s = lead_cursor.map_or(0.0, |_| snap.lead_seconds());
    let bound_on = terrain.config.ask_bound;
    let hysteresis = terrain.config.ask_bound_hysteresis;
    let rebind_fraction = terrain.config.ask_bound_rebind;
    let bracket = terrain.config.ask_bound_bracket;
    let descent_every_s = terrain.config.ask_bound_descent_s;
    let speed_hold_s = terrain.config.speed_hold_s;
    let mut recomputed: Vec<RealmId> = Vec::new();
    let mut rebind: Vec<RealmId> = Vec::new();
    // ★ THE FASTEST BODY'S OWN EYE, for the card's stand-down rule (the owner's step after
    // Step 15): the eye is inside one realm, and a queue is urgent because THAT realm's ground is
    // on its way to the screen.
    let mut fastest_mps = 0.0f64;
    // ★ THE FRAME'S OWN WORK (ruling F9 item 1's frame bar): the wall time this frame spends in
    // each piece the bounded ask added, and in the descent it may re-run. The flight reads them.
    let mut work_bound = std::time::Duration::ZERO;
    let mut work_descent = std::time::Duration::ZERO;
    let mut descents = 0u64;
    // ★ THE BOARDING INSTRUMENT (2026-09-14): each ladder's eye and wanted size BEFORE this
    // frame's descent, so a wholesale release can name what the descent changed.
    let before: BTreeMap<RealmId, (Option<[f64; 3]>, u64)> = terrain
        .ladders
        .iter()
        .map(|(r, l)| (*r, (l.eye, l.wanted.keys.len() as u64)))
        .collect();
    for eb in &with_bodies {
        let piece = std::time::Instant::now();
        let rungs = eb.body.ladder().rungs;
        let reading = if lead_s > 0.0 {
            (DVec3::from_array(eb.lead) - DVec3::from_array(eb.eye)).length() / lead_s
        } else {
            0.0
        };
        let ladder = terrain.ladders.entry(eb.realm).or_default();
        // THE SAWTOOTH'S PEAK (see `RealmLadder::speed`): the largest reading of the last
        // `speed_hold_s` seconds, which reaches ZERO when the hull stops.
        ladder.speed_mps = ladder.speed.read(reading, now_s, speed_hold_s);
        let speed_mps = ladder.speed_mps;
        fastest_mps = fastest_mps.max(speed_mps);
        // The bound reads the LAST DESCENT's own altitude and its own chunks-a-column: both are
        // measurements the descent already made, and the recipe is not run a second time for them.
        // Before the first descent there is no measurement, and the ask is the tier rule's.
        // ★ AND THE BOUND READS THE BODY (ruling T7 rules 2 and 3): the body's own octave table
        // states each handover's step, which floors the switch distance and widens the crossfade
        // band where a dropped octave would stand over the ladder's own tolerance. Inert on every
        // body whose table drops under a cell, which is every body the recipe draws today.
        let candidate = if bound_on & !ladder.wanted.is_empty() {
            vd_client::ladder_view::ask_bound(
                rungs,
                AskRate {
                    chunks_per_s: throughput,
                    speed_mps,
                    altitude_m: ladder.wanted.altitude_m,
                    chunks_per_column: ladder.wanted.chunks_per_column(CHUNKS_PER_COLUMN),
                },
            )
        } else {
            AskBound::unbounded()
        }
        .for_body(&eb.body, rungs);
        ladder.pace.slew(&candidate, rungs, hysteresis, frame_s);
        if ladder.pace.take_rebind(rungs, rebind_fraction) {
            rebind.push(eb.realm);
        }
        let bound_changed = ladder
            .pace
            .take_descent(rungs, bracket, descent_every_s, now_s);
        if bound_changed {
            ladder.view.bound = ladder.pace.asked().clone();
        }
        let reach_changed = match (ladder.view.shadow, shadow) {
            (Some(a), Some(b)) => !a.same_as(b),
            (a, b) => a.is_some() != b.is_some(),
        };
        let wanted_move = moved(ladder.eye, eb.lead) || reach_changed || bound_changed;
        let run_descent = wanted_move & !eye_refused;
        work_bound += piece.elapsed();
        // ★ THE DESCENT'S PACE, READ (2026-09-16, the walk-gap measurement): how far the LEAD
        // eye has moved from the eye the standing ring was cut for, and how long ago it was cut.
        ladder.drift_m = ladder.eye.map_or(0.0, |last| {
            (DVec3::from_array(last) - DVec3::from_array(eb.lead)).length()
        });
        ladder.since_descent_s = ladder.last_descent_s.map_or(0.0, |t| now_s - t);
        ladder.descent_ran = run_descent;
        if run_descent {
            let descent = std::time::Instant::now();
            ladder.view.shadow = shadow;
            let fresh = ladder.view.wanted(&eb.body, eb.lead);
            ladder.prev = std::mem::replace(&mut ladder.wanted, fresh);
            ladder.last_descent_s = Some(now_s);
            ladder.eye = Some(eb.lead);
            recomputed.push(eb.realm);
            work_descent += descent.elapsed();
            descents += 1;
        }
    }
    // ★ THE CARD'S STAND-DOWN RULE (the owner's step after Step 15): the frame judges whether the
    // queue is deep enough to want a second builder at all. THE LEAD IS THE HELD SPEED'S OWN — the
    // lead's metres are a sawtooth that reads zero the instant a row lands, and the speed is that
    // sawtooth's held peak, so the pair the rule divides must come from the same reading.
    let workers_per_s = terrain.throughput.workers_value();
    terrain.judge_queue(workers_per_s, fastest_mps, fastest_mps * lead_s);
    // THE CROSSFADE FOLLOWS THE BOUND: the realm's three material families carry the same
    // effective switch distances the descent asked at, rewritten in place.
    let rebinds = rebind.len() as u64;
    let piece = std::time::Instant::now();
    for realm in &rebind {
        if let Some(body) = terrain.lane.body(*realm).map(Arc::clone) {
            terrain.rebind_bands(
                *realm,
                &body,
                &mut ground_materials,
                probe_materials.as_deref_mut(),
            );
        }
    }
    let work_rebind = piece.elapsed();
    terrain.note_work(WORK_THROUGHPUT, work_throughput, 1, now_s);
    terrain.note_work(WORK_BOUND, work_bound, 1, now_s);
    terrain.note_work(WORK_DESCENT, work_descent, descents, now_s);
    terrain.note_work(WORK_REBIND, work_rebind, rebinds, now_s);
    // THE CASTING DELTAS (item 18): a drawn chunk that left the casting set releases its caster;
    // one that entered it asks for its caster.
    let coarse_step = terrain.config.shadow_coarse_step;
    let coarse_from = terrain.config.shadow_coarse_from_rung;
    let mut caster_changes: Vec<(RealmId, ChunkKey, Option<ChunkKey>)> = Vec::new();
    for realm in &recomputed {
        let Some(ladder) = terrain.ladders.get(realm) else {
            continue;
        };
        let Some(body) = terrain.lane.body(*realm) else {
            continue;
        };
        // The realm's own keys alone (SL9: never a walk of every realm's chunks).
        let first = (*realm, FIRST_CHUNK_KEY);
        for ((r, key), drawn) in terrain.entities.range(first..) {
            if r != realm {
                break;
            }
            if coarse_step == 0 || key.rung < coarse_from {
                continue;
            }
            match (drawn.caster, ladder.wanted.casts(*key)) {
                (Some(_), false) => caster_changes.push((*realm, *key, None)),
                (None, true) => {
                    if let Some(ckey) = vd_client::chunks::coarse_key(body, *key, coarse_step) {
                        caster_changes.push((*realm, *key, Some(ckey)));
                    }
                }
                _ => {}
            }
        }
    }
    for (realm, key, want) in caster_changes {
        let Some(drawn) = terrain.entities.get(&(realm, key)) else {
            continue;
        };
        let had = drawn.caster;
        match want {
            Some(ckey) => terrain.want_caster(&mut commands, realm, ckey),
            None => {
                if let Some(ckey) = had {
                    terrain.unwant_caster(&mut commands, realm, ckey);
                }
            }
        }
        if let Some(drawn) = terrain.entities.get_mut(&(realm, key)) {
            drawn.caster = want;
        }
    }
    // The realms whose row left the window: forget their ladder (their chunks go below) — but
    // ★ ONLY AFTER THE ROW'S GRACE (2026-09-14, the boarding cure). A row absent from ONE
    // frame's scene is not a realm that left: at a boarding the gateway composes the first
    // picture in the hull's frame, and the planet's row can miss a beat while it does. The
    // grace is the interpolation buffer — the wire's own contract for how old a delivered
    // picture may be — never a frame count this crate chose. Example: the pilot steps aboard a
    // berthed hull; the planet's row is away for two frames; its 7 288 chunks stay drawn and
    // the ground never blinks.
    let grace_s = snap.lead_seconds();
    for (realm, ladder) in terrain.ladders.iter_mut() {
        if centres.contains_key(realm) {
            ladder.row_seen_s = Some(now_s);
        }
    }
    let gone: Vec<RealmId> = terrain
        .ladders
        .iter()
        .filter(|(r, l)| {
            vd_client::ladder_view::row_lapsed(
                centres.contains_key(r),
                l.row_seen_s,
                now_s,
                grace_s,
            )
        })
        .map(|(r, _)| *r)
        .collect();
    for realm in gone {
        // ★ THE BOARDING INSTRUMENT (2026-09-14): NAME the forget. The release loop below drops
        // every chunk this realm holds, so this line is the whole cost of one frame without a
        // row. `still_a_parent` says whether the delivered scene still names the realm as some
        // row's parent — a hull whose row states the planet as its parent while the planet's own
        // row is absent.
        let held = terrain.entities.keys().filter(|(r, _)| *r == realm).count() as u64;
        let pending = terrain
            .lane
            .pending_all()
            .into_iter()
            .filter(|(r, _)| *r == realm)
            .count() as u64;
        let still_a_parent = scene.iter().any(|(_, b)| b.parent == Some(realm));
        terrain.ladders_forgotten += 1;
        terrain.last_forget = Some(vd_devproto::DevLadderForget {
            realm: format!("{realm:?}"),
            frame: terrain.frames,
            held,
            pending,
            rows: centres.len() as u64,
            still_a_parent,
        });
        tracing::warn!(
            ?realm,
            frame = terrain.frames,
            held,
            pending,
            rows = centres.len(),
            still_a_parent,
            origin = ?snap.origin(),
            forgets = terrain.ladders_forgotten,
            "the terrain forgets a realm's ladder: no row for it in this frame's scene"
        );
        terrain.ladders.remove(&realm);
    }
    // Release what is no longer wanted — with THE HOLD: a chunk stays while a wanted chunk over its
    // footprint is still building (coarse before fine, SL8). Then ask for what is wanted and not
    // held, in the wanted set's own order: the coarsest ring first.
    let held: Vec<(RealmId, ChunkKey)> = terrain.entities.keys().copied().collect();
    // ★ THE BOARDING INSTRUMENT (2026-09-14): how many DRAWN chunks each realm loses this frame.
    let mut released_count: BTreeMap<RealmId, u64> = BTreeMap::new();
    let unwant = {
        let Terrain {
            lane,
            ladders,
            entities,
            morph_totals,
            bytes_drawn,
            shadow_wanted,
            shadow_casters,
            shadow_bytes,
            ..
        } = &mut *terrain;
        let mut unwant: Vec<(RealmId, ChunkKey)> = Vec::new();
        for (realm, key) in held {
            let wanted = ladders.get(&realm).map(|l| &l.wanted);
            let keep = wanted.is_some_and(|w| {
                w.contains(key)
                    || w.overlapping_missing(Column::of(key), &|k| lane.is_resident(realm, k))
            });
            if !keep {
                lane.release(realm, key);
                if let Some(drawn) = entities.remove(&(realm, key)) {
                    *released_count.entry(realm).or_default() += 1;
                    commands.entity(drawn.entity).despawn();
                    if let Some(twin) = drawn.twin {
                        commands.entity(twin).despawn();
                    }
                    let mut i = 0;
                    while i < 3 {
                        morph_totals[i] -= drawn.counts[i];
                        i += 1;
                    }
                    *bytes_drawn -= drawn.bytes;
                    // THE SHADOW LADDER: the caster leaves with its last wanter (below, once the
                    // lane is free again).
                    if let Some(ckey) = drawn.caster {
                        unwant.push((realm, ckey));
                    }
                }
            }
        }
        // THE LADDER COMES TO WANT A KEY HELD AS A CASTER: the caster is dropped and released, so
        // the request below rebuilds the key as a drawn chunk (which casts itself while wanted).
        for (realm, ladder) in ladders.iter() {
            for key in ladder.wanted.keys.iter() {
                if let Some(c) = shadow_casters.remove(&(*realm, *key)) {
                    commands.entity(c.entity).despawn();
                    *shadow_bytes -= c.bytes;
                    lane.release(*realm, *key);
                }
            }
        }
        // A chunk still BUILDING that is no longer wanted is withdrawn from the workers (nothing is
        // drawn for it, so no hold): a moving eye leaves no stale job in the queue. A coarse
        // caster still wanted is not.
        for (realm, key) in lane.pending_all() {
            let wanted = ladders.get(&realm).is_some_and(|l| l.wanted.contains(key))
                || shadow_wanted.contains_key(&(realm, key));
            if !wanted {
                lane.release(realm, key);
            }
        }
        // Every wanted chunk not yet resident is asked for at its priority — a job already
        // waiting moves to this frame's priority (the eye moved; its class may have changed).
        for (realm, ladder) in ladders.iter() {
            for (index, key) in ladder.wanted.keys.iter().enumerate() {
                if !lane.is_resident(*realm, *key) {
                    lane.request(*realm, *key, ladder.wanted.priority_of(index, *key));
                }
            }
        }
        unwant
    };
    for (realm, ckey) in unwant {
        terrain.unwant_caster(&mut commands, realm, ckey);
    }
    // ★ THE BOARDING INSTRUMENT (2026-09-14): a frame that released a WHOLESALE count of one
    // realm's chunks names itself — the realm, the count, whether the descent ran, the eye in the
    // body's own frame before and after it, the lead offset, the wanted set's size on both sides,
    // the drawn and lead scenes' row counts, the origin, and the frame the own pose was delivered
    // in at each cursor. A band that dies without a forget dies HERE, and this line says why.
    for (realm, released) in &released_count {
        if *released <= WHOLESALE_RELEASE {
            continue;
        }
        let (eye_before, wanted_before) = before.get(realm).copied().unwrap_or((None, 0));
        let (eye_after, wanted_after) = terrain.ladders.get(realm).map_or(([0.0; 3], 0), |l| {
            (l.eye.unwrap_or([0.0; 3]), l.wanted.keys.len() as u64)
        });
        let descent = recomputed.contains(realm);
        terrain.band_releases += 1;
        let row = vd_devproto::DevBandRelease {
            realm: format!("{realm:?}"),
            frame: terrain.frames,
            released: *released,
            descent,
            eye_before,
            eye_after,
            lead_offset_m: lead_offset.length(),
            wanted_before,
            wanted_after,
            rows: centres.len() as u64,
            lead_rows: scene_lead.as_ref().map_or(0, RealmScene::len) as u64,
            origin: format!("{:?}", snap.origin()),
            own_frame: own_frame_now.clone(),
            own_lead_frame: own_frame_lead.clone(),
        };
        tracing::warn!(
            realm = %row.realm,
            frame = row.frame,
            released = row.released,
            descent = row.descent,
            eye_before = ?row.eye_before,
            eye_after = ?row.eye_after,
            lead_offset_m = row.lead_offset_m,
            wanted_before = row.wanted_before,
            wanted_after = row.wanted_after,
            rows = row.rows,
            lead_rows = row.lead_rows,
            origin = %row.origin,
            own_frame = %row.own_frame,
            own_lead_frame = %row.own_lead_frame,
            releases = terrain.band_releases,
            eye_foreign_frames = terrain.eye_foreign_frames,
            "the terrain releases a realm's band wholesale in one frame"
        );
        terrain.last_release = Some(row);
    }
    // 3. Harvest finished chunks — each with its rung's crossfade material — and their probe twins
    //    where a probe exists (Capture mode).
    let flat = terrain.config.flat;
    let exact_normal_rung = terrain.config.exact_normal_rung;
    let splat_rung = terrain.config.splat_rung;
    let hide_rung = terrain.config.hide_rung;
    let shadow_cast = terrain.config.shadow_cast;
    let shadow_cast_rung = terrain.config.shadow_cast_rung;
    let shadow_receive = terrain.config.shadow_receive;
    let coarse_step = terrain.config.shadow_coarse_step;
    let coarse_from = terrain.config.shadow_coarse_from_rung;
    let harvest_cap = terrain.config.harvest_per_frame;
    let harvest_bytes = terrain.config.harvest_bytes_per_frame;
    // THE UPLOAD's cost on the main thread (M8-2a): the mesh conversion, the asset, the entity —
    // per harvested chunk, so the harvest's own wall is named in milliseconds.
    let harvest_started = std::time::Instant::now();
    for ready in terrain.lane.poll_within(harvest_cap, harvest_bytes) {
        let realm = ready.realm;
        let key = ready.geometry.key;
        let wanted = terrain
            .ladders
            .get(&realm)
            .is_some_and(|l| l.wanted.contains(key));
        let caster_wanted = terrain.shadow_wanted.contains_key(&(realm, key));
        if !wanted && !caster_wanted {
            terrain.lane.release(realm, key);
            continue;
        }
        let Some(body) = terrain.lane.body(realm).map(Arc::clone) else {
            // No body: released; a caster's wants are forgotten with it (nothing re-asks).
            terrain.lane.release(realm, key);
            terrain.shadow_wanted.remove(&(realm, key));
            continue;
        };
        if !wanted {
            // THE SHADOW LADDER: a coarse caster lands on the shadow layer, for the sun alone,
            // its culling box grown by its own sink (the caster stands under the drawn ground).
            let material = terrain.shadow_material(&mut ground_materials, realm, key.rung, &body);
            let sink_m = terrain.caster_sink_m(&body, key.rung);
            let built = mesh_of(&ready.geometry, false, exact_normal_rung, false);
            let bytes = mesh_bytes(&built);
            let mesh = meshes.add(built);
            let bounds = moved_bounds(&ready.geometry, sink_m as f32);
            let entity = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    Transform::default(),
                    TerrainChunk {
                        realm,
                        rung: key.rung,
                    },
                    ChunkOrigin(ready.geometry.origin_m),
                    bounds,
                    RenderLayers::layer(SHADOW_LAYER),
                    bevy::light::NotShadowReceiver,
                    ShadowCaster,
                ))
                .id();
            terrain.shadow_bytes += bytes;
            if let Some(old) = terrain
                .shadow_casters
                .insert((realm, key), Caster { entity, bytes })
            {
                commands.entity(old.entity).despawn();
                terrain.shadow_bytes -= old.bytes;
            }
            continue;
        }
        // THE SHADOW LADDER: a drawn chunk from the coarse rung up asks for the coarse chunk that
        // holds it, which casts for it and for its neighbours; it casts nothing itself unless a
        // finer chunk wants IT as a caster (the invariant, `want_caster`).
        // ... while its caster may throw a shadow within the sun's reach (item 18).
        let casts = terrain
            .ladders
            .get(&realm)
            .is_none_or(|l| l.wanted.casts(key));
        let caster = (coarse_step > 0 && key.rung >= coarse_from && casts)
            .then(|| vd_client::chunks::coarse_key(&body, key, coarse_step))
            .flatten();
        if let Some(ckey) = caster {
            terrain.want_caster(&mut commands, realm, ckey);
        }
        // A caster that stood for this key gives way to the drawn chunk, which casts itself.
        if let Some(c) = terrain.shadow_casters.remove(&(realm, key)) {
            commands.entity(c.entity).despawn();
            terrain.shadow_bytes -= c.bytes;
        }
        let casts_itself = terrain.shadow_wanted.contains_key(&(realm, key));
        let material = terrain.ground_material(&mut ground_materials, realm, key.rung, &body);
        let splat = splat_rung.is_some_and(|r| key.rung >= r);
        let built = mesh_of(&ready.geometry, flat, exact_normal_rung, splat);
        let bytes = mesh_bytes(&built);
        let mesh = meshes.add(built);
        // THE BOUNDS the engine culls by, grown to where the vertex stage can move a vertex: its
        // morph target and its whole sink (the engine reads the box from the positions alone).
        // A splat reaches one cell past its vertex on the screen's plane.
        let margin_m = if splat {
            vd_seed::ladder::cell_m(key.rung) as f32
        } else {
            0.0
        };
        let bounds = moved_bounds(&ready.geometry, margin_m);
        let entity = commands
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material),
                Transform::default(),
                TerrainChunk {
                    realm,
                    rung: key.rung,
                },
                ChunkOrigin(ready.geometry.origin_m),
                bounds,
            ))
            .id();
        if hide_rung.is_some_and(|r| key.rung >= r) {
            commands.entity(entity).insert(Visibility::Hidden);
        }
        if !shadow_cast
            || shadow_cast_rung.is_some_and(|r| key.rung >= r)
            || (caster.is_some() && !casts_itself)
        {
            commands.entity(entity).insert(bevy::light::NotShadowCaster);
        }
        if !shadow_receive {
            commands
                .entity(entity)
                .insert(bevy::light::NotShadowReceiver);
        }
        let twin = probe_materials.as_mut().map(|pm| {
            let probe = terrain.probe_material(pm, realm, PROBE_KIND_TERRAIN, key.rung, &body);
            commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(probe),
                    Transform::default(),
                    TerrainChunk {
                        realm,
                        rung: key.rung,
                    },
                    ChunkOrigin(ready.geometry.origin_m),
                    bounds,
                    ProbeTwin,
                    RenderLayers::layer(PROBE_LAYER),
                    bevy::light::NotShadowCaster,
                ))
                .id()
        });
        let counts = [
            u64::from(ready.geometry.morph_fallbacks),
            u64::from(ready.geometry.morph_seam),
            ready.geometry.vertices.len() as u64,
        ];
        let mut i = 0;
        while i < 3 {
            terrain.morph_totals[i] += counts[i];
            i += 1;
        }
        terrain.bytes_drawn += bytes;
        if let Some(old) = terrain.entities.insert(
            (realm, key),
            Drawn {
                entity,
                twin,
                counts,
                bytes,
                caster,
            },
        ) {
            // A key drawn twice (refutation of the ladder, finding 2): the old entity, its twin,
            // its counts and its caster's want all leave with it.
            commands.entity(old.entity).despawn();
            if let Some(t) = old.twin {
                commands.entity(t).despawn();
            }
            let mut i = 0;
            while i < 3 {
                terrain.morph_totals[i] -= old.counts[i];
                i += 1;
            }
            terrain.bytes_drawn -= old.bytes;
            if let Some(ckey) = old.caster {
                terrain.unwant_caster(&mut commands, realm, ckey);
            }
        }
    }
    terrain.harvest_nanos += harvest_started.elapsed().as_nanos() as u64;
    // 4. Every chunk rides its row: transform = draw_center + facing · origin, in f64, narrowed once
    //    (`place_chunks`, below, which runs after the spawn commands apply). The count on screen goes
    //    to the diagnosis surface.
    net.terrain_drawn.store(
        terrain.entities.len() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    net.terrain_pending.store(
        terrain.lane.pending_count() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    // ONE DRAWING PER REALM: while a realm's terrain is on screen its proxy outline is hidden (the
    // ladder reaches the horizon and past it, so the outline would paint over the ground); the
    // moment the terrain is gone the outline is back.
    let drawn_realms: BTreeSet<RealmId> = terrain.entities.keys().map(|(r, _)| *r).collect();
    for (realm, (entity, _)) in boxes.0.iter() {
        let has_terrain = drawn_realms.contains(realm);
        if let Ok(mut vis) = visibility.get_mut(*entity) {
            let want = if has_terrain {
                Visibility::Hidden
            } else {
                Visibility::Inherited
            };
            if *vis != want {
                *vis = want;
            }
        }
    }
    // 5. The sun: from the brightest luminous row; when the window holds none (the star out of the
    //    planet's window, MEASURED on the first ground picture: a black ground under a black sky),
    //    a WORK LIGHT from straight above the observer — the radial of the body under the eye — so
    //    the relief is seen. A work light is style; it moves no vertex.
    //
    // THE BODY UNDER THE EYE is the one whose ladder floor is NEAREST the eye (the floor stands a
    // few kilometres under the recipe; as a RANKING between bodies thousands of kilometres apart it
    // is the same answer as the surface) — not the first in the window. MEASURED on the first
    // stamped picture (slice 8p): the window holds every planet of the system with a surface
    // statement, the first by id was a sibling 29 000 km away, and the stamp read an altitude of
    // 29 416 km over it. The instrument found the fault the work light had carried silently.
    let under_eye: Option<(RealmId, Arc<vd_terrain::BodyDefinition>, [f64; 3])> = with_bodies
        .iter()
        .map(|eb| {
            let over = DVec3::from_array(eb.eye).length() - eb.body.ladder().radius_m();
            (over, eb)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, eb)| (eb.realm, Arc::clone(&eb.body), eb.eye));
    // THE LEAD, MEASURED where the band runs: the distance from the drawn eye to the lead eye in
    // the frame of the body UNDER THE EYE. Two earlier readings of this stamp, before this form,
    // chose it: the own pose's offset alone read "lead 0.0 m" at 240 m/s (a pilot stands still
    // inside a flying hull while the hull moves the eye through the planet), and the widest lead
    // over every body read "lead 4 310 m" on a walk (a moon's: in the spinning planet's frame a
    // moon moves kilometres per buffer — that moon's band, not this ground's). The shipped form
    // reads 14–30 m at 240 m/s (M8-1, the ninth run).
    let lead_m = under_eye
        .as_ref()
        .and_then(|(realm, _, eye_body)| {
            with_bodies
                .iter()
                .find(|eb| eb.realm == *realm)
                .map(|eb| (DVec3::from_array(*eye_body) - DVec3::from_array(eb.lead)).length())
        })
        .unwrap_or(0.0);
    let overhead = under_eye.as_ref().map(|(realm, _, _)| {
        let (centre, _) = centres[realm];
        -centre.normalize_or_zero()
    });
    let sun_from = brightest.map(|(_, star)| star).or(overhead);
    if let Some(star) = sun_from {
        let dir = -star;
        if dir.length_squared() > 0.0 {
            let dir = Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32).normalize();
            let transform = Transform::default().looking_to(dir, Vec3::Y);
            // The incidence at the eye: the angle between the light and the local up. Read at
            // every placement for the casters' reach (item 18); the bias reads it at the birth.
            let up = overhead.map_or(Vec3::Y, |u| Vec3::new(u.x as f32, u.y as f32, u.z as f32));
            let cos_i = (-dir).dot(up).clamp(0.0, 1.0);
            let tan_i = ((1.0 - cos_i * cos_i).sqrt() / cos_i.max(1e-3)).min(SHADOW_BIAS_TAN_CAP);
            terrain.sun_tan_i = Some(tan_i);
            match terrain.sun {
                Some(sun) => {
                    if let Ok(mut t) = light_tf.get_mut(sun) {
                        *t = transform;
                    }
                }
                None => {
                    let lux = exposure
                        .iter()
                        .next()
                        .map_or_else(|| sun_lux(&bevy::camera::Exposure::default()), sun_lux);
                    // THE SHADOW (M8-L): the sun casts one, over the two nearest rings.
                    let reach_m = switch_m(terrain.config.shadow_reach_rung) as f32;
                    // The map's width is an engine resource; set with the sun.
                    commands.insert_resource(bevy::light::DirectionalLightShadowMap {
                        size: terrain.config.shadow_map_px as usize,
                    });
                    let cascades = bevy::light::CascadeShadowConfigBuilder {
                        num_cascades: terrain.config.shadow_cascades,
                        first_cascade_far_bound: reach_m * SHADOW_FIRST_CASCADE_SHARE,
                        maximum_distance: reach_m,
                        ..default()
                    }
                    .build();
                    let sun = commands
                        .spawn((
                            DirectionalLight {
                                illuminance: lux,
                                shadows_enabled: terrain.config.shadows,
                                shadow_normal_bias: DirectionalLight::DEFAULT_SHADOW_NORMAL_BIAS
                                    + tan_i,
                                ..default()
                            },
                            cascades,
                            transform,
                            TerrainSun,
                            // The sun sees the picture's layer and the shadow ladder's.
                            RenderLayers::from_layers(&[0, SHADOW_LAYER]),
                        ))
                        .id();
                    terrain.sun = Some(sun);
                    // ONE SUN: the stub world's fixed key light retires the moment the sun is born
                    // (it lit the ground from a direction no star stands in, at three times white).
                    for key in &key_light {
                        commands.entity(key).despawn();
                    }
                }
            }
        }
    }
    // 6. THE STAMP AND THE RULER (slice 8p, ruling V14 D8-7): for the body under the eye (above).
    //    The stamp is assembled here and completed by `place_chunks` (the nearest and farthest
    //    chunk); the ruler is a ball where the picture's centre ray meets the drawn ground, at the
    //    rung the tier rule names for its distance, only where a probe exists to read it.
    terrain.stamp = None;
    let nose = cam.iter().next().map(|t| {
        let f = t.forward();
        DVec3::new(f64::from(f.x), f64::from(f.y), f64::from(f.z))
    });
    // The camera's rotation in the render frame, widened once: the stamp states it in the body's
    // frame (step 6), beside the drawn eye.
    let camera_rotation = cam.iter().next().map(|t| {
        let q = t.rotation;
        DQuat::from_xyzw(
            f64::from(q.x),
            f64::from(q.y),
            f64::from(q.z),
            f64::from(q.w),
        )
    });
    let mut ruler_now: Option<(DVec3, f64, u8, DevRuler)> = None;
    if let Some((realm, body, eye_body)) = under_eye.as_ref() {
        let (centre, facing) = centres[realm];
        let eye_body = *eye_body;
        if let Some(ground) = eye_surface(body, eye_body) {
            let rungs = body.ladder().rungs;
            let (reach, rung_min, rung_max) = terrain.ladders.get(realm).map_or((0.0, 0, 0), |l| {
                (l.wanted.reach_m, l.wanted.rung_min, l.wanted.rung_max)
            });
            let up = -centre.normalize_or_zero();
            let star = brightest.and_then(|(_, star)| {
                nose.map(|f| {
                    let (elevation, off_nose) = star_angles(star, up, f);
                    DevStarAngles {
                        elevation_deg: elevation.to_degrees(),
                        off_nose_deg: off_nose.to_degrees(),
                        direction_body: (facing.inverse() * star.normalize_or_zero()).to_array(),
                    }
                })
            });
            if let (Some(f), true) = (nose, probe_materials.is_some()) {
                let forward_body = (facing.inverse() * f).to_array();
                let fresh = terrain
                    .ruler_cache
                    .as_ref()
                    .is_some_and(|c| c.eye_body == eye_body && c.forward_body == forward_body);
                if !fresh {
                    // Two passes: march at the rung under the eye, then at the rung the rule names
                    // for the hit's distance, so the ball stands on the ground that is drawn there.
                    let first_rung = rung_for_distance(ground.altitude_m, rungs);
                    let first = ruler_on_surface(body, eye_body, forward_body, first_rung, reach);
                    let ruler = first.and_then(|r| {
                        let rung = rung_for_distance(r.distance_m, rungs);
                        let again = if rung == first_rung {
                            Some(r)
                        } else {
                            ruler_on_surface(body, eye_body, forward_body, rung, reach)
                        };
                        again.map(|r| (r, rung))
                    });
                    terrain.ruler_cache = Some(RulerCache {
                        eye_body,
                        forward_body,
                        ruler,
                    });
                }
                if let Some((r, rung)) = terrain.ruler_cache.as_ref().and_then(|c| c.ruler) {
                    let p = centre + facing * DVec3::from_array(r.centre_m);
                    ruler_now = Some((
                        p,
                        r.radius_m,
                        rung,
                        DevRuler {
                            centre_m: p.to_array(),
                            radius_m: r.radius_m,
                            distance_m: r.distance_m,
                            rung,
                            cell_m: f64::from(vd_seed::ladder::cell_m(rung)),
                        },
                    ));
                }
            }
            // THE BAND'S GAP, once per frame: the urgent chunks not yet harvested, by rung and in
            // all (a chunk harvested this frame is spawned at the schedule's end and draws next
            // frame: the gap reads one frame early, never late).
            let (gap_per_rung, gap_revealed) = {
                let Terrain { lane, ladders, .. } = &*terrain;
                let mut counts: BTreeMap<u8, u64> = BTreeMap::new();
                let mut revealed = 0;
                for (r, l) in ladders {
                    for (rung, n) in l
                        .wanted
                        .urgent_missing_per_rung(&|k| lane.is_resident(*r, k))
                    {
                        *counts.entry(rung).or_insert(0) += n;
                    }
                    revealed += l.wanted.revealed_missing(&|k| lane.is_resident(*r, k)) as u64;
                }
                (counts.into_iter().collect::<Vec<(u8, u64)>>(), revealed)
            };
            let gap: u64 = gap_per_rung.iter().map(|(_, n)| *n).sum();
            terrain.urgent_frames += u64::from(gap > 0);
            // ★ THE GAP, NAMED AND LATCHED (2026-09-16, the walk-gap measurement): which chunks
            // the band lacks, what the LAST descent called each of them, and what the DRAWN eye
            // reads of each one's column. A chunk the last set called `absent` was first wanted by
            // this frame's own descent; one it called `urgent` was asked earlier and is late.
            if gap > 0 {
                let misses: Vec<vd_devproto::DevBandMiss> = {
                    let Terrain { lane, ladders, .. } = &*terrain;
                    ladders.get(realm).map_or_else(Vec::new, |l| {
                        l.wanted
                            .urgent_missing_keys(&|k| lane.is_resident(*realm, k), GAP_KEYS_CAP)
                            .into_iter()
                            .map(|key| {
                                let probe = l.view.probe(body, eye_body, key);
                                vd_devproto::DevBandMiss {
                                    key: format!("{key:?}"),
                                    rung: key.rung,
                                    was: l.prev.class_of(key).name().to_owned(),
                                    near_drawn_m: probe.near_m,
                                    territory_m: probe
                                        .territory_m
                                        .is_finite()
                                        .then_some(probe.territory_m),
                                    horizon_m: probe.horizon_m,
                                    drawn_urgent: probe.urgent,
                                }
                            })
                            .collect()
                    })
                };
                let pace = terrain
                    .ladders
                    .get(realm)
                    .map_or((false, 0.0, 0.0), |l: &RealmLadder| {
                        (l.descent_ran, l.drift_m, l.since_descent_s)
                    });
                terrain.last_gap = Some(vd_devproto::DevBandGap {
                    realm: format!("{realm:?}"),
                    frame: terrain.frames,
                    urgent: gap,
                    misses,
                    descent: pace.0,
                    drift_m: pace.1,
                    step_m: EYE_STEP_M,
                    since_descent_s: pace.2,
                    lead_m,
                });
            }
            let built = terrain.lane.built();
            let counters = terrain.lane.counters();
            let parents = terrain.lane.parent_stats();
            let camera_body = camera_rotation.map_or(DQuat::IDENTITY, |c| facing.inverse() * c);
            terrain.stamp = Some(DevTerrainStamp {
                realm: format!("{realm:?}"),
                rung_min,
                rung_max,
                eye_body_m: eye_body,
                camera_body_xyzw: camera_body.to_array(),
                chunks_per_rung: terrain.drawn_per_rung(),
                surface_m: ground.surface_m,
                altitude_m: ground.altitude_m,
                horizon_m: horizon_m(ground.surface_m, ground.altitude_m),
                horizon_dip_deg: horizon_dip_rad(ground.surface_m, ground.altitude_m).to_degrees(),
                drawn_radius_m: reach,
                chunk_nearest_m: 0.0,
                chunk_farthest_m: 0.0,
                chunks_drawn: terrain.entities.len() as u64,
                chunks_pending: terrain.lane.pending_count() as u64,
                chunks_urgent: gap,
                chunks_revealed: gap_revealed,
                urgent_per_rung: gap_per_rung,
                urgent_frames: terrain.urgent_frames,
                frames: terrain.frames,
                built_chunks: built.chunks,
                build_nanos: built.nanos,
                build_peak_nanos: built.peak_nanos,
                build_peak_key: built
                    .peak_key
                    .map_or_else(String::new, |k| format!("{k:?}")),
                harvested: counters.harvested,
                harvest_full: counters.harvest_full,
                harvest_nanos: terrain.harvest_nanos,
                parent_hits: parents.hits,
                parent_builds: parents.builds,
                parent_waits: parents.waits,
                lead_m,
                // THE BOUNDED ASK (ruling F9 item 1): the builders' capacity, the eye's speed
                // through the body under it, and that body's deliverable horizons.
                build_rate_per_s: terrain.throughput.value(),
                eye_speed_mps: terrain
                    .ladders
                    .get(realm)
                    .map_or(0.0, |l: &RealmLadder| l.speed_mps),
                ask_horizon_m: terrain
                    .ladders
                    .get(realm)
                    .map(|l: &RealmLadder| l.view.bound.horizons_m())
                    .unwrap_or_default(),
                // ★ THE CARD AS A SECOND BUILDER (ruling F9 item 2): what it built, what its
                // boxes cost it, and what the frames granted it.
                card_boxes: card.0,
                card_nanos: card.1,
                card_budget_nanos: card.2,
                card_per_box_ms: card.3,
                card_boxes_per_frame: card.4,
                card_capacity_per_s: card.5,
                card_device_timed: card.6,
                card_judged: card.7,
                card_stood_down: card.8,
                frame_peak_ms: terrain.frame_peak_ms as f32,
                frame_work_ns: terrain.frame_work(),
                morph_fallbacks: terrain.morph_totals[0],
                morph_seam: terrain.morph_totals[1],
                vertices: terrain.morph_totals[2],
                bytes_drawn: terrain.bytes_drawn,
                shadow_casters: terrain.shadow_casters.len() as u64,
                shadow_bytes: terrain.shadow_bytes,
                hud_rect_px: [0.0; 4],
                frame_ms: diagnostics
                    .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FRAME_TIME)
                    .and_then(bevy::diagnostic::Diagnostic::smoothed)
                    .unwrap_or(0.0) as f32,
                passes_ms: render_passes_ms(&diagnostics),
                star,
                biome: format!("{:?}", ground.biome),
                world: format!("{:#x}", terrain.declared),
                tick: snap.freshest_tick(),
                ruler: ruler_now.as_ref().map(|(_, _, _, d)| *d),
                // ★ THE BOARDING INSTRUMENT (2026-09-14): the forgets, and the scene swap the
                // view last took (`place_camera` runs before this system, in the same chain).
                ladders_forgotten: terrain.ladders_forgotten,
                last_forget: terrain.last_forget.clone(),
                scene_swaps: render_eye.swaps,
                swap: render_eye.swap.clone(),
                band_releases: terrain.band_releases,
                last_release: terrain.last_release.clone(),
                eye_foreign_frames: terrain.eye_foreign_frames,
                // ★ THE CAMERA'S REFUSAL AND THE EYE'S JUMP (2026-09-15): decided in
                // `place_camera`, which runs before this system in the same chain.
                eye_refusals: render_eye.track.refusals,
                eye_jump_m: render_eye.track.jump_max_m,
                eye_step_m: render_eye.track.step_max_m,
                last_gap: terrain.last_gap.clone(),
                sky: None,
            });
        }
    }
    let realm_under = under_eye
        .as_ref()
        .map(|(realm, _, _)| *realm)
        .unwrap_or(RealmId::System(0));
    let realm_under = &realm_under;
    let body_under = under_eye.as_ref().map(|(_, b, _)| Arc::clone(b));
    // The ruler's entities follow the answer: born with it, placed every frame, gone without it or
    // rebuilt when its rung (its probe material) changes.
    if let (Some((_, _, rung, _)), Some(e)) = (ruler_now, terrain.ruler)
        && e.rung != rung
    {
        commands.entity(e.ball).despawn();
        commands.entity(e.twin).despawn();
        terrain.ruler = None;
    }
    match (ruler_now, terrain.ruler) {
        (Some((p, r, _, _)), Some(e)) => {
            for entity in [e.ball, e.twin] {
                if let Ok(mut t) = ruler_tf.get_mut(entity) {
                    *t = placed(p, r);
                }
            }
        }
        (Some((p, r, rung, _)), None) => {
            let (mesh, paint) = match &terrain.ruler_assets {
                Some(a) => a.clone(),
                None => {
                    // The ball's mesh carries the morph and radial attributes too (a zero
                    // metre and an upward radial: a ball never morphs, and its material's
                    // sink is zero), because the probe's material asks every mesh for them.
                    // ★ THE BALL'S FACETS ARE THE RULER'S OWN ERROR (slice 8b stage 7, MEASURED on the
                    // orbit stand): the default icosphere (five subdivisions, facets spanning about
                    // 2°) puts a silhouette facet's depth up to `r·sin 2°` past the true limb — 4.4 km
                    // on a 125 km ball, 1.06 cells of its rung, and one stray pixel read 2.1 cells
                    // past the rim against the gate's 2-cell tolerance. Seven subdivisions (facets of
                    // about 0.5°) bound that excess at `r·sin 0.5°` — a quarter of a cell there, and
                    // under a hundredth of a cell on the ground's metre ball — so the probe's rim
                    // reading is the limb's to within its own quantisation. The tolerance is not
                    // moved; the instrument is.
                    let mut ball = Sphere::new(1.0)
                        .mesh()
                        .ico(RULER_BALL_SUBDIVISIONS)
                        .expect("an icosphere of seven subdivisions");
                    let count = ball.count_vertices();
                    // The ball's normals packed like the ground's BESIDE the engine's own: the
                    // ball itself is lit by the engine's standard material (which reads the
                    // standard normal), its probe twin by the probe's shader (which reads the
                    // packed one). MEASURED without the standard normal: a flat dark disc.
                    let packed: Vec<[i16; 2]> = match ball.attribute(Mesh::ATTRIBUTE_NORMAL) {
                        Some(bevy::mesh::VertexAttributeValues::Float32x3(v)) => {
                            v.iter().map(|n| oct_encode(*n)).collect()
                        }
                        other => panic!("the sphere's normals are Float32x3, not {other:?}"),
                    };
                    ball.insert_attribute(
                        super::ATTRIBUTE_OCT_NORMAL,
                        bevy::mesh::VertexAttributeValues::Snorm16x2(packed.clone()),
                    );
                    // The ball morphs to itself: its morph normal is its own.
                    ball.insert_attribute(
                        super::ATTRIBUTE_MORPH_NORMAL,
                        bevy::mesh::VertexAttributeValues::Snorm16x2(packed),
                    );
                    ball.insert_attribute(super::ATTRIBUTE_MORPH, vec![0.0f32; count]);
                    ball.insert_attribute(super::ATTRIBUTE_RADIAL, vec![[0.0f32, 1.0, 0.0]; count]);
                    let a = (
                        meshes.add(ball),
                        materials.add(StandardMaterial {
                            base_color: Color::srgb(RULER_SRGB[0], RULER_SRGB[1], RULER_SRGB[2]),
                            perceptual_roughness: RULER_ROUGHNESS,
                            metallic: 0.0,
                            ..default()
                        }),
                    );
                    terrain.ruler_assets = Some(a.clone());
                    a
                }
            };
            let probe = probe_materials
                .as_mut()
                .map(|pm| {
                    let body = body_under.as_ref().expect("the ruler stands on a body");
                    terrain.probe_material(pm, *realm_under, PROBE_KIND_RULER, rung, body)
                })
                .expect("the ruler exists only with a probe");
            let ball = commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(paint),
                    placed(p, r),
                    RulerBall,
                ))
                .id();
            let twin = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(probe),
                    placed(p, r),
                    RulerBall,
                    ProbeTwin,
                    RenderLayers::layer(PROBE_LAYER),
                    bevy::light::NotShadowCaster,
                ))
                .id();
            terrain.ruler = Some(RulerEntities { ball, twin, rung });
        }
        (None, Some(e)) => {
            commands.entity(e.ball).despawn();
            commands.entity(e.twin).despawn();
            terrain.ruler = None;
        }
        (None, None) => {}
    }
}

/// A chunk's origin in its realm's frame, kept on the entity so its transform can be re-derived
/// every frame from the row without the lane.
#[derive(Component)]
pub struct ChunkOrigin(pub [f64; 3]);

/// Place every drawn chunk (and its twin) against the eye: `draw_center(row) + facing · origin`, in
/// f64, narrowed once. Runs after `sync_terrain` so a chunk spawned this frame is placed this frame.
/// Completes this frame's stamp with the nearest and farthest chunk and publishes it.
pub(crate) fn place_chunks(
    net: Res<super::Net>,
    render_eye: Res<super::RenderEye>,
    mut terrain: ResMut<Terrain>,
    mut chunks: PlacedChunks,
    mut frame: Local<u32>,
) {
    // THE STAMP STAYS IN THE RESOURCE (slice 8 step 6): the frame's completed stamp is
    // published for the dev-control poll AND kept here, so the extract of this very frame
    // carries it beside the pixels (a captured frame's dump then states the drawn camera of the
    // picture it sits beside). MEASURED with `take()` here: the extract found none, the dump fell
    // back to the poll's stamp of a random later frame, and no recorded pair was ever consecutive
    // in the terrain's own count (0, +2, +3) while the capture frames were.
    let mut stamp = terrain.stamp.clone();
    // Nothing drawn: nothing to place, and no scene to clone (a flight in open space used to walk
    // every row of the window for no pixel — SL9). The stamp still goes out.
    if chunks.is_empty() {
        *net.terrain_stamp
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = stamp;
        return;
    }
    // THE FRAME'S MOMENT (step 6): the camera's own sample, so the chunks stand where the
    // stamped eye looked from.
    let Some((now_s, snap)) = render_eye.moment.clone() else {
        *net.terrain_stamp
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = stamp;
        return;
    };
    let scene = snap.scene_now(now_s);
    let mut rows: BTreeMap<RealmId, (DVec3, DQuat)> = BTreeMap::new();
    for (realm, rbox) in scene.iter() {
        rows.insert(
            realm,
            (
                super::draw_center_of(rbox, &render_eye, &snap, now_s),
                facing_of(rbox),
            ),
        );
    }
    let mut nearest = f64::INFINITY;
    let mut farthest = 0.0_f64;
    let mut placed = 0u32;
    let mut first: Option<(RealmId, [f64; 3], DVec3)> = None;
    for (chunk, origin, twin, caster, mut transform) in &mut chunks {
        if let Some((centre, facing)) = rows.get(&chunk.realm) {
            let o = DVec3::new(origin.0[0], origin.0[1], origin.0[2]);
            let p = *centre + *facing * o;
            transform.translation = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            transform.rotation = Quat::from_xyzw(
                facing.x as f32,
                facing.y as f32,
                facing.z as f32,
                facing.w as f32,
            );
            // The twins ride the same rows but are not counted twice; the casters ride them and
            // are not counted at all (the sun's, not the picture's).
            if twin.is_none() && caster.is_none() {
                nearest = nearest.min(p.length());
                farthest = farthest.max(p.length());
                placed += 1;
                if first.is_none() {
                    first = Some((chunk.realm, origin.0, p));
                }
            }
        }
    }
    if let Some(t) = &mut stamp {
        t.chunk_nearest_m = if placed > 0 { nearest } else { 0.0 };
        t.chunk_farthest_m = farthest;
    }
    *net.terrain_stamp
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = stamp.clone();
    terrain.stamp = stamp;
    *frame += 1;
    if frame.is_multiple_of(DIAG_EVERY) {
        for (realm, (centre, facing)) in &rows {
            tracing::debug!(
                ?realm,
                centre = ?centre,
                facing = ?facing,
                dist = centre.length(),
                "terrain diag: row"
            );
        }
        tracing::debug!(
            placed,
            nearest,
            farthest,
            first = ?first,
            eye = ?render_eye.eye,
            eye_tier = ?render_eye.eye_lattice.map(|(_, t)| t),
            view = ?render_eye.view,
            "terrain diag: chunks"
        );
    }
}

/// Frames between two diagnostic lines (`RUST_LOG=vd_client_render=debug` shows them: every row's
/// drawn centre and facing, the chunks' nearest and farthest distance, the eye and its unit — the
/// lines that found the 230 km column, kept for the next such hunt).
pub(crate) const DIAG_EVERY: u32 = 60;

/// THE CORE SHARE (ruling F6, 2026-09-12; the owner: *"we are building a game that can be run on an
/// average machine… those cores will need to do a bunch of other stuff"*): the terrain workers take
/// a quarter of the machine's cores and never fewer than two. On an eight-core machine that is two
/// workers; on this Mac's fourteen, three. The knob (`VD_TERRAIN_WORKERS`) overrides it for a
/// measurement; the ceiling measurements of §24.4 were taken at the whole machine and stay as the
/// instrument's numbers, not the target's.
#[must_use]
pub const fn worker_share(cores: usize) -> usize {
    let quarter = cores / 4;
    if quarter < 2 { 2 } else { quarter }
}

#[cfg(test)]
mod worker_tests {
    use super::*;

    /// ★★ THE SPOT-CHECK'S OWN COMPARE (§26.10, the drift hunt): the shipped path rebuilds one
    /// card box a second on the CPU and compares it cell for cell, and a box that is not the CPU's
    /// DETACHES the card. This states that the compare answers what it is asked — a box that IS
    /// the CPU's passes, and ONE CELL of a different substance fails.
    ///
    /// **Example.** The pilot flies over the seam. The card hands the geometry stage a box whose
    /// deep bedrock came out as the last box's limestone; the compare says so, the card leaves,
    /// and the CPU workers carry the ladder — the pilot never flies at a hill the shard has not
    /// got.
    #[test]
    fn the_spot_check_names_a_box_that_is_not_the_cpus() {
        let body = vd_terrain::home::home_planet();
        let key = vd_terrain::digest::self_check_key(&body, vd_terrain::GOLDEN_SELF_CHECK_KEYS[0]);
        let mut box_of =
            vd_terrain::lattice::sample_box(&body, None, key).expect("the box is on the ladder");
        assert!(
            !verify_card_box(&body, key, &box_of, false),
            "the CPU's own box is the CPU's box"
        );
        // ONE CELL a step out of place is the whole difference a hole is made of.
        box_of.cells[0].gap = box_of.cells[0].gap.wrapping_add(1);
        assert!(
            verify_card_box(&body, key, &box_of, false),
            "one differing cell is a box that is not the CPU's"
        );
    }

    #[test]
    fn the_worker_share_is_a_quarter_of_the_cores_and_at_least_two() {
        assert_eq!(worker_share(1), 2);
        assert_eq!(worker_share(4), 2);
        assert_eq!(worker_share(8), 2);
        assert_eq!(worker_share(12), 3);
        assert_eq!(worker_share(14), 3);
        assert_eq!(worker_share(16), 4);
        assert_eq!(worker_share(64), 16);
    }
    use vd_client::chunks::ParentCache;
    use vd_seed::bend::Face;

    fn job(body: &Arc<vd_terrain::BodyDefinition>, x: i32, priority: u32) -> ChunkJob {
        ChunkJob {
            realm: RealmId::Planet(body.seed()),
            body: Arc::clone(body),
            key: ChunkKey {
                face: Face::PosX,
                rung: 3,
                x,
                y: 5,
                z: vd_terrain::digest::surface_chunk_z(body, Face::PosX, 3, x, 5),
            },
            parents: Arc::new(ParentCache::default()),
            priority,
            artifact: None,
        }
    }

    fn drain_all(workers: &mut ThreadedWorkers, want: usize) -> Vec<i32> {
        let mut out = Vec::new();
        let started = std::time::Instant::now();
        while out.len() < want && started.elapsed() < std::time::Duration::from_secs(60) {
            workers.drain(&mut out, 8);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        out.iter().map(|r| r.geometry.key.x).collect()
    }

    /// THE ORDER (ruling V15): with one worker busy on a first job, three jobs submitted out of
    /// order come back by priority; a re-request moves a waiting job; a cancel removes it; a drop
    /// closes the queue and builds nothing more.
    #[test]
    fn the_pool_serves_by_priority_moves_and_withdraws_waiting_jobs_and_closes() {
        let body = Arc::new(vd_terrain::home::home_planet());
        let mut workers = ThreadedWorkers::start(1, HARVEST_PER_FRAME * DONE_QUEUE_FRAMES);
        // A blocker the one worker takes at once, so the next three wait in the queue.
        workers.submit(job(&body, 1, 0));
        std::thread::sleep(std::time::Duration::from_millis(50));
        workers.submit(job(&body, 5, 5));
        workers.submit(job(&body, 2, 9));
        workers.submit(job(&body, 3, 3));
        // The chunk at x = 2 waits at 9: moved to 1, it goes first; x = 5 is withdrawn.
        workers.reprioritise(RealmId::Planet(body.seed()), job(&body, 2, 0).key, 1);
        workers.cancel(RealmId::Planet(body.seed()), job(&body, 5, 0).key);
        assert!(workers.waiting() <= 2);
        let order = drain_all(&mut workers, 3);
        assert_eq!(order, vec![1, 2, 3]);
        let count = workers.built();
        assert_eq!(count.chunks, 3);
        assert!(count.nanos > 0);
        // A re-request of a chunk not waiting changes nothing; a cancel of one not waiting too.
        workers.reprioritise(RealmId::Planet(body.seed()), job(&body, 7, 0).key, 1);
        workers.cancel(RealmId::Planet(body.seed()), job(&body, 7, 0).key);
        assert_eq!(workers.waiting(), 0);
        // A job submitted twice sits once, at its last priority.
        workers.submit(job(&body, 8, 4));
        workers.submit(job(&body, 8, 2));
        assert!(workers.waiting() <= 1);
        // The drop closes the queue with jobs waiting: no hang.
        drop(workers);
    }
}
