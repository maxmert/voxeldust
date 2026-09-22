//! ★ THE LADDER VIEW (the voxel foundation, slice 8 step 2; ruling V14 D8-1, D8-6) — which rung the
//! client draws at every distance, and the set of chunks that follows from it, out to the horizon.
//!
//! **The tier rule (D8-1 A).** A column of ground at distance `d` from the eye is drawn at the FINEST
//! rung `L` whose cell still stands one pixel high at the reference view: `cell_m(L) ≥ d · pixel_rad`.
//! The pixel is the drawable floor every realm's reach already reads (`vd_core::geometry`). One rule
//! for every realm kind — a planet, a moon, a hull with terrain — never a branch on what the realm is
//! (HR3, HR4).
//!
//! ```text
//!   distance from the eye →
//!   |- rung 0 -|-- rung 1 --|--- rung 2 ---|---- rung 3 ----| ... |-- the top rung --| the horizon
//!   0      switch(0)     switch(1)      switch(2)        switch(3)
//!           869 m         1.7 km         3.5 km           7.0 km        (switch(L) = cell(L) / pixel)
//! ```
//!
//! **The rings.** Rung `L` covers the ring of distances `(switch(L−1), switch(L)]`; the top rung
//! covers everything past its inner edge out to the REACH: the horizon from the eye's height plus the
//! horizon of the tallest ground the recipe can raise (its amplitude sum), so a peak behind the
//! geometric horizon is still wanted. PAST THE GEOMETRIC HORIZON A COLUMN IS WANTED ONLY WHEN IT
//! CAN SHOW OVER THE SKYLINE the near ground raises (`crate::skyline`): every column inside the
//! horizon raises a wall — its guaranteed floor, less a cell and the sink its mesh may stand
//! under, over its footprint quad on the eye's chart — per ray, and a far column is culled only
//! when its peak bound stands under the lowest wall at every azimuth its cap spans. MEASURED before the skyline: the reach alone wanted 6 556 chunks from
//! the ground, most of them hidden by the curve of the planet; the sphere's own tangent as the
//! line of sight culled a valley floor the eye could see into (47 pixels of sky between two
//! crests), and the tangent of the sphere lowered by the whole relief bound culled nothing (8 775
//! chunks). The outermost ring IS the globe beyond the band (D8-6): the body's own recipe at the
//! rung above the coarsest inner ring, never
//! black, never a proxy outline.
//!
//! **Coarse before fine.** The wanted set lists the coarsest ring first, so the workers build the
//! coarse ground before the fine, and a chunk that is no longer wanted is released only when every
//! wanted chunk over its footprint has ARRIVED — a column never shows a hole while its finer rung is
//! still building (SL8).
//!
//! **The crossfade (step 3, D8-2, §5).** Around every switch distance `s_L` lies a BAND,
//! `(HYSTERESIS_IN · s_L, HYSTERESIS_OUT · s_L)`, in which BOTH rung `L` and rung `L + 1` are drawn:
//! rung `L` MORPHED onto rung `L + 1` — every vertex of a rung-`L` chunk carries a second position,
//! where its own radial meets the rung-`L + 1` MESH (`chunks::ParentMesh`), and slides from its own
//! position to that one across the band on its own distance from the eye — and rung `L + 1` SUNK
//! beneath it, dropping under the finer surface by the recipe's bound across the same band
//! (`chunks::ChunkGeometry::{morph, sink}`, the `ladder_fade.wgsl` vertex stage). At the band's far
//! edge the finer vertices lie on the coarser triangles, so the finer ends there without a pop;
//! nearer, the coarser lies under the finer and never shows through; no two rungs ever meet at a
//! mesh edge, so the hairline crack of step 2 has nothing to open on; and no pixel is ever left to
//! nobody, because the coarser is always there. (A dither that gives every pixel to one of the two
//! rungs was MEASURED first and refused: two surfaces that stand apart along a pixel's ray cannot
//! share a weight — 116 holes; a weight read against a reference sphere is no distance where the
//! ground has relief — 895 holes on the horizon rows; and a finer hill seen over a near crest
//! dithers out onto a coarser surface under the sightline — 47 holes.) The band's two edges ARE
//! the hysteresis: nothing flips at either, because the weight there is one or zero.
//! [`fade_bands`] states a rung's two bands; the shaders compute the weights from them.
//!
//! **The cost, sized.** A ring's outer edge is `switch(L)` and a chunk at rung `L` is `62 · cell(L)`,
//! so every ring is the same annulus in its own chunks — inner 7.0, outer 14.0 chunks — about 460
//! columns, and the rung-0 disc about 620. From the ground to the 6.6 km horizon that is four rings:
//! about 2 000 columns. M8-2 measures it; the mesh packing (D8-4) is the lever.
//!
//! **Example.** The pilot stands on the home planet. Columns within 869 m are rung 0, out to 1.7 km
//! rung 1, out to 3.5 km rung 2, out to 7.0 km rung 3, and the horizon at 6.6 km lies inside rung 3.
//! From 60 km up the ground under her is rung 7, and the ring that reaches the 877 km horizon is
//! rung 10.

use std::collections::{BTreeMap, BTreeSet};
use vd_core::glam::{DVec2, DVec3};

use crate::artifact_book::ArtifactCache;
use crate::skyline::{DISC_MARGIN, EyeFrame, Skyline, WALL_MAX_HALF_ANGLE};
use vd_seed::bend::{Face, direction, face_coords, face_of, unbend};
use vd_seed::ladder::{cell_m, face_param, index_of};
use vd_terrain::BodyDefinition;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::digest::ColumnSpan;

/// THE HYSTERESIS PAIR (D8-2, starting values; the pop detector fixes them): the crossfade band
/// around a switch distance `s` runs from `HYSTERESIS_IN · s`, where the finer rung starts to fade
/// out and the coarser to fade in, to `HYSTERESIS_OUT · s`, where the finer is gone and the coarser
/// whole. A column moving in switches to the finer rung's full weight at the inner edge, one moving
/// out to the coarser's at the outer edge, and neither edge flips anything.
pub const HYSTERESIS_IN: f64 = 0.9;
/// The outer edge is the terrain crate's word (`ASK_HYSTERESIS_OUT`): the shard and the gateway
/// ship tiles to the same edge the ladder asks at (2026-09-21, the far ring's tiles).
pub const HYSTERESIS_OUT: f64 = vd_terrain::artifact::ASK_HYSTERESIS_OUT;
/// A band that is always passed: a rung 0 has no finer rung to fade in from, the top rung no
/// coarser one to fade out to. Stated as a band far below or far above every distance so the one
/// weight formula serves every rung (the shader clamps).
pub const FADE_ALWAYS_IN: [f64; 2] = [-2.0, -1.0];
pub const FADE_ALWAYS_OUT: [f64; 2] = [1.0e30, 2.0e30];

/// THE BANDS of a rung: `(in, out)` — the distances over which a chunk of `rung` fades in (the
/// band around the switch below it) and out (the band around its own switch). Rung 0 is always
/// in; the top rung is always out-of-fade (never fades out).
#[must_use]
pub fn fade_bands(rung: u8, rungs: u8) -> ([f64; 2], [f64; 2]) {
    AskBound::unbounded().fade_bands(rung, rungs)
}

/// WHERE THE SINK RAMP ENDS for a rung: past its fade-in edge `in_hi` (the finer rung's fade-out
/// edge, where the finer ends), so that AT the edge the rung still stands one finer cell under
/// the finer surface. The morph is exact at the finer VERTICES; between them a finer triangle
/// that spans a coarser crease is the chord under it, by up to a finer cell, and MEASURED with the
/// ramp ending at the edge the coarser mesh showed through the chord there (dark specks at every
/// fade-out edge). The ramp runs from `in_lo` (the whole sink) to this end (none), linearly, so the
/// residual at `in_hi` is one finer cell; past the edge only this rung is drawn and it rises to
/// its own surface with distance — continuous, never a pop. Rung 0 sinks nowhere: its edge.
#[must_use]
pub fn sink_end_m(body: &BodyDefinition, rung: u8, rungs: u8) -> f64 {
    AskBound::unbounded().sink_end_m(body, rung, rungs)
}

/// Whether a distance lies inside a crossfade band of some rung: where two rungs share the ground.
#[must_use]
pub fn in_fade_band(d_m: f64, rungs: u8) -> bool {
    let rung = rung_for_distance(d_m, rungs);
    let (fade_in, fade_out) = fade_bands(rung, rungs);
    ((d_m > fade_in[0]) & (d_m < fade_in[1])) | ((d_m > fade_out[0]) & (d_m < fade_out[1]))
}
// ★ THE CLIENT HAS NO FAR EDGE (owner 2026-09-15, *"agree"*): THE SERVER'S VISIBILITY RADIUS IS
// THE ONLY RULE. One radius per realm, tested by its parent; a realm has a row in the pilot's
// window only inside it, and the ladder draws a body at any distance while the body has a row.
//
// WHY THE CLIENT'S OWN EDGE DIED. It was `drawn_reach_m` — "one top-rung chunk column stands one
// pixel". Since the ladder was extended (owner 2026-09-15) a top-rung column is wider than the body
// itself, so that line landed at about 2 200 body radii, 29 times OUTSIDE the realm's own stated
// radius of 76.39: it could never fire, because no parent ships a row out there. The measurement
// that built it — a far body costing about 5 000 chunks — is SPENT: the extended ladder draws a far
// globe with SIX chunks, one tile a cube face.
//
// Example. The pilot lifts her hull off the home planet and keeps climbing. The ground stays the
// recipe's own the whole way out and coarsens by the descent's own rules — a few hundred chunks at
// 1.5 radii, a few dozen at 10, and SIX from 34 radii out to wherever the parent still ships the
// planet's row. Past the parent's radius there is no row, so there is nothing to draw.

// The descent's roots are EVERY top-rung column of EVERY face (ONE per edge by the ladder's
// construction since 2026-09-15, so SIX on any body), each tested by its nearest point against the
// reach. MEASURED before this (the refuter's finding): the roots were a square around the eye's
// foot folded through the foot's own face, whose bend reaches 76° from that face's centre — a
// hull 2 000 km over a point near a face edge sees ground out to 83°, and 7° of arc, 780 km
// inside the horizon, was never a candidate.

/// The reference view's pixel, in radians: the drawable floor.
#[must_use]
pub fn pixel_rad() -> f64 {
    vd_core::geometry::drawable_theta_min_rad()
}

/// The distance at which one cell of `rung` stands one pixel high: `cell_m(rung) / pixel_rad`.
#[must_use]
pub fn switch_m(rung: u8) -> f64 {
    vd_terrain::artifact::switch_m(rung, pixel_rad())
}

/// ★ THE STEP A HANDOVER MAKES, in metres — ruling T7 rules 2 and 3 both read this ONE number, and
/// it comes from the BODY'S OWN OCTAVE TABLE, never from a per-rung literal.
///
/// Where rung `rung` hands the ground to rung `rung + 1`, the picture changes by exactly the octaves
/// the coarser rung drops (`vd_terrain::BodyDefinition::octaves_at`), at the per-column roughness
/// factor's ceiling of ONE — the same ceiling the ladder's own band is sized at, so 8c's macro field
/// may replace the factor without moving this number.
///
/// ★ **THE BODY STATES IT** (`vd_terrain::BodyDefinition::step_bound_m`, slice 8a stage 4), and the
/// client never re-derives it from two bounds. Since the cap-rock bench landed the step is no longer
/// a plain difference of amplitude sums: the terrace AMPLIFIES the dropped octaves by its own rung's
/// Lipschitz constant and adds its fade's own step, and only the body knows both.
///
/// **Example.** On the home planet the rung 7 → 8 handover drops the 3 125 m crest: 198.8 m against
/// a 256 m cell of the rung that takes over. The rung 0 → 1 handover drops a 30 cm ripple.
#[must_use]
pub fn handover_step_m(body: &BodyDefinition, rung: u8) -> f64 {
    body.step_bound_m(rung).max(0.0)
}

/// ★ THAT STEP IN PIXELS — the ladder's own tolerance, and the line the judge
/// (`cargo run -p vd-bins --example rung_disagreement`) measures on: ONE CELL OF THE RUNG THAT
/// TAKES OVER, which is one pixel at that rung's own switch distance.
///
/// A rung's surface is the finer surface with the octaves its own cells cannot carry left out
/// (ruling V9), so a rung may stand a cell of its own away from the rung below it and no more.
/// Under one, the handover cannot be seen; over one, the eye catches the ground move.
#[must_use]
pub fn step_px(step_m: f64, rung: u8) -> f64 {
    step_m / f64::from(cell_m(rung + 1))
}

/// ★ RULE 2 — HOW MUCH WIDER THE CROSSFADE BAND STANDS, as a factor on its own half width
/// (ruling T7 rule 2, owner 2026-09-17).
///
/// The crossfade turns the handover's step into a RAMP: the band's width is the ground over which
/// the step is paid off, so the rate the eye sees is the step over the band. The ladder's own band
/// (`HYSTERESIS_IN`…`HYSTERESIS_OUT`, a fifth of the switch distance) was sized on a step of ONE
/// cell; a step of `r` cells needs `r` times the ground to fall at that same rate. So the band
/// widens by exactly [`step_px`], and never narrows.
///
/// ★ **IT IS INERT TODAY, BY MEASUREMENT.** After ruling T7 rule 1 every pair of the home planet's
/// table drops under one cell — the worst reads 0.999, at the rung the cap-rock bench fades over
/// (slice 8a stage 4; it was 0.88 at rung 9 → 10 before the bench) — so this answers ONE everywhere
/// and no band moves. It wakes by itself if a later term (the cap-rock bench, the terrace, 8c's
/// macro field) pushes a pair over, which is why it reads the table instead of a frozen number.
///
/// ★ **AND IT IS NOT DEAD CODE:** the recipe already draws a body it wakes on — the rock of seed
/// 382 at a 300 km look radius reads 1.52 cells at its rung 3 → 4 handover, which is the body the
/// crate's own test flies (`a_body_over_the_tolerance_widens_its_band_and_pushes_its_switch_out`).
#[must_use]
pub fn band_widen(step_m: f64, rung: u8) -> f64 {
    step_px(step_m, rung).max(1.0)
}

/// ★ RULE 3 — THE SWITCH DISTANCE FLOOR (ruling T7 rule 3, owner 2026-09-17): a rung hands the
/// ground over at the LARGER of two distances — where its own cell stands one pixel (today's rule,
/// [`switch_m`]) and where THE STEP THE HANDOVER MAKES stands under the ladder's own tolerance.
///
/// **The derivation.** The tolerance is one cell of the rung that takes over ([`step_px`]). A cell
/// of rung `L + 1` is twice a cell of rung `L`, so at the handover distance `D` the tolerance is
/// `2 · D · pixel_rad` metres, and the step stands inside it while `D ≥ step_m / (2 · pixel_rad)` —
/// which is the same line as "the step is one pixel at the COARSER rung's own switch distance",
/// because that distance is twice this one. One bound, two rules.
///
/// ★ **IT IS INERT TODAY** on the home planet, for the same measured reason [`band_widen`] is, and
/// it protects the picture where the band alone would have to stretch too far. It is a FLOOR: the
/// bounded ask (ruling F9) may pull a rung's horizon in for deliverability, but never so near that
/// the handover's step shows — a missing chunk is covered by the next rung, a step over the
/// tolerance is a seam with nothing under it.
#[must_use]
pub fn switch_floor_m(step_m: f64) -> f64 {
    step_m / (2.0 * pixel_rad())
}

/// THE TIER RULE: the finest rung whose cell is at least one pixel at distance `d_m`, clamped to the
/// body's top rung. A distance at or under the pixel's own size reads rung 0.
#[must_use]
pub fn rung_for_distance(d_m: f64, rungs: u8) -> u8 {
    let need_m = d_m.max(0.0) * pixel_rad();
    let rung = if need_m <= 1.0 {
        0
    } else {
        need_m.log2().ceil() as u8
    };
    rung.min(rungs.saturating_sub(1))
}

/// ★ THE BOUNDED ASK (ruling F9 item 1, 2026-09-13) — the client asks the finest ring only as far
/// ahead as its own builders can deliver it before the ground reaches the screen; past that it
/// asks the NEXT rung, which stands whole, and the crossfade blends the finer rung in as it lands.
///
/// The bound is ONE NUMBER PER RUNG: the rung's EFFECTIVE SWITCH DISTANCE, never farther than the
/// tier rule's own [`switch_m`]. Every part of the ladder already reads a rung's switch distance —
/// the descent's split, the rung's territory, the crossfade's bands, the sink's ramp — so moving
/// that one number moves all of them together, and the picture stays a crossfade instead of a cut.
///
/// **Example.** The pilot's hull crosses the home planet at 528 m/s, a kilometre up, on a machine
/// whose terrain share is three workers. The tier rule wants metre cells out to 869 m, two-metre
/// cells to 1.7 km and four-metre cells to 3.5 km; three workers build about 195 chunks a second
/// against an ask of about 400. The bound answers: four-metre cells out to about 2.1 km, and the
/// eight-metre ring takes over there. The pilot sees whole ground one rung coarser instead of holes
/// at every rung; when the hull slows to a walk the horizons grow back to the tier rule's own and
/// the metre cells fade in.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AskBound {
    /// Each rung's effective switch distance, in metres, finest first. `None` is the tier rule's
    /// own radii — the bound does not bind, and the ladder is the one every earlier flight flew.
    switches: Option<Vec<f64>>,
    /// ★ THE DESCENT'S SLACK (ruling F9 item 1, the frame bar): the fraction each crossfade band's
    /// edges are pushed OUT by. The DESCENT carries it; the bound the materials carry does not.
    ///
    /// Why it exists. The horizon SLIDES every frame, and the descent may not follow it every
    /// frame — MEASURED (§25.7): a descent costs 3 to 5 ms and re-running one per body per frame
    /// cost 8.4 ms a frame at 528 m/s, which is the whole of the frame rate's fall. So the descent
    /// runs on the horizon as it stood, and asks for a ring WIDER than that horizon's own bands by
    /// this fraction — wide enough to cover every band the drawn horizon may slide to before the
    /// next descent. A hole is ruled out by the DERIVATION, not by the shape: the fraction is
    /// [`ASK_BOUND_SLACK`], computed from the descent's bracket and the materials' rebind together
    /// and asserted against both at compile time, because the picture and the descent read the
    /// horizon at two different tolerances and the worst stand of the two is their product.
    slack: f64,
    /// ★ THE BODY'S OWN HANDOVER STEPS (ruling T7 rules 2 and 3), finest first, in metres —
    /// [`handover_step_m`] read once per body ([`AskBound::for_body`]). Rule 3's FLOOR and rule 2's
    /// WIDENING are both derived from this one row, so a bound that has read a body draws the
    /// body's own ladder and one that has not draws the tier rule's.
    ///
    /// EMPTY means no body has been read: every step is zero, the floor is zero and the widening is
    /// one, which is the ladder every earlier flight flew.
    steps_m: Vec<f64>,
}

/// ★ HOW FAST A DELIVERABLE HORIZON MAY MOVE: a quarter of its own length every second.
///
/// A horizon carries a CROSSFADE BAND with it (`AskBound::fade_bands`), and a band that JUMPS is a
/// pop — every chunk in it changes its morph weight in one frame, and a rung-5 chunk's morph metre
/// is tens of metres. A band that SLIDES is what the crossfade was built for: the band is a fifth
/// of the switch distance wide, so a quarter of a length a second carries a chunk across the whole
/// band in about 0.8 s — fifty frames of fade at 60 Hz, softer than the fade an approaching eye
/// makes.
/// (The eye's own crossing at 528 m/s and a 3.5 km switch takes about 1.3 s, so the horizon's own
/// motion is of the same order and never faster than a few times it.)
///
/// A horizon at rest may still move: a rung's step is read from its horizon PLUS its own column's
/// width, so a rung bound in to nothing can always grow back.
///
/// MEASURED at half a length a second (§25.7): the horizon then crossed the descent's own bracket
/// about five times a second for every body in the window, and each crossing costs a descent. The
/// quarter shipped here halves those crossings, and the band traversal it makes is softer than the
/// half was, not harsher.
pub const ASK_BOUND_SLEW_PER_S: f64 = 0.25;

/// How far a rung's horizon must move before the client adopts the step at all: half a percent of
/// the horizon. The measured build rate and the measured speed wander frame to frame, so the
/// horizon they ask for wanders too; without this the descent would run and every material's bands
/// would be rewritten on every frame for a change no eye can see. The SLEW above is what keeps the
/// horizon from flapping; this is what keeps it from churning once it has arrived.
pub const ASK_BOUND_HYSTERESIS: f64 = 0.005;

/// ★ HOW FAR A HORIZON MUST MOVE BEFORE THE CROSSFADE'S MATERIALS FOLLOW IT: a twentieth.
///
/// The DESCENT follows the horizon every frame — it runs every frame at speed anyway — but the
/// three material families carry the bands as a UNIFORM, and rewriting a material makes the engine
/// prepare its bind group again. MEASURED (the third bounded flight, 2026-09-13): the horizon the
/// measurement asks for wanders about a tenth either side of its own mean from frame to frame (the
/// build rate reads 182 to 226 chunks a second over one leg), so the materials were rewritten on
/// nearly every frame of the 528 m/s leg and the frame rate fell from 43.2 to 38.7.
///
/// A twentieth is well inside the crossfade band's own width (a fifth of the switch distance).
/// What makes the difference safe is not that it is small but that [`ASK_BOUND_SLACK`] is derived
/// from THIS fraction and the descent's bracket together, so the ring the descent asked for covers
/// the band the picture draws at the worst stand of both.
pub const ASK_BOUND_REBIND: f64 = 0.05;

/// WHAT THE CLIENT MEASURED about its builders and its own motion, for the bounded ask. Every
/// number here is a MEASUREMENT the client already holds; nothing is derived from a pose the
/// client made up (SL10 clause 7: the speed comes from two DELIVERED poses, the same pair the
/// lead eye is made of).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AskRate {
    /// THE BUILDERS' THROUGHPUT, chunks a second: the workers' count over the mean wall time of a
    /// build, smoothed. Read as a CAPACITY, not as what the builders happened to do — on a walk
    /// the builders are idle and the chunks they finished are the ask, not the ceiling.
    pub chunks_per_s: f64,
    /// THE EYE'S SPEED through the body's frame, metres a second, from delivered poses alone.
    pub speed_mps: f64,
    /// The eye's height over the surface under it, metres.
    pub altitude_m: f64,
    /// The chunks one column holds, measured on the last descent (a column of the home planet
    /// holds one or two: the surface crosses one chunk, two where it crosses a chunk boundary).
    pub chunks_per_column: f64,
}

/// A COLUMN'S FOOTPRINT at a rung, in metres: a chunk column is [`CHUNK_EDGE`] cells across.
#[must_use]
pub fn column_span_m(rung: u8) -> f64 {
    CHUNK_EDGE as f64 * f64::from(cell_m(rung))
}

/// THE GROUND CIRCLE an eye `altitude_m` over the surface reaches at slant distance `slant_m`:
/// its radius along the ground. Zero where the slant does not reach the ground at all — an eye a
/// kilometre up has NO ground within 869 m of it, which is why the finest ring is empty at
/// altitude and the bound there costs nothing.
fn ground_radius_m(slant_m: f64, altitude_m: f64) -> f64 {
    (slant_m * slant_m - altitude_m * altitude_m)
        .max(0.0)
        .sqrt()
}

/// THE ASK RATE of one rung, chunks a second, when the rung is asked out to `reach_m` (a slant
/// distance): an eye moving at `speed` uncovers a strip `2 · ground_radius` wide of new ground
/// every second, and each new column of the rung costs `chunks_per_column` builds.
///
/// An ESTIMATE, and a deliberately generous one: it counts the whole circle's leading edge, where
/// the skyline and the horizon cull part of it, so the bound binds a little sooner than the true
/// ask needs. Completeness first (ruling F9 item 1); the flight is the judge.
fn ask_rate_per_s(rung: u8, reach_m: f64, rate: AskRate) -> f64 {
    let span = column_span_m(rung);
    2.0 * ground_radius_m(reach_m, rate.altitude_m) * rate.speed_mps * rate.chunks_per_column
        / (span * span)
}

/// THE REACH one rung can hold at `budget` chunks a second: the inverse of [`ask_rate_per_s`],
/// as a slant distance from the eye.
fn ask_reach_m(rung: u8, budget: f64, rate: AskRate) -> f64 {
    let span = column_span_m(rung);
    let ground = budget * span * span / (2.0 * rate.speed_mps * rate.chunks_per_column);
    (ground * ground + rate.altitude_m * rate.altitude_m).sqrt()
}

/// THE DELIVERABLE HORIZONS of a ladder of `rungs` rungs, from what the client measured
/// ([`AskRate`]).
///
/// The walk is COARSEST FIRST, with the builders' throughput as a budget. A coarse ring is cheap
/// (its columns hold four times the area of the ring below it and it is only twice as wide), so it
/// is served first and costs little; the budget that is left buys the finer rings. The rung where
/// the budget runs out keeps the reach the budget pays for, and every finer rung follows the
/// ladder's own half-and-double from there — so no two rungs ever share a crossfade band.
///
/// The bound NEVER BINDS where the builders cover the ask: a still stand (no speed, no ask), a
/// walk, a strong machine, or an eye whose rings are empty all read the tier rule's own radii and
/// the returned bound is [`AskBound::unbounded`], byte for byte the ladder of every earlier flight.
#[must_use]
pub fn ask_bound(rungs: u8, rate: AskRate) -> AskBound {
    let measured = (rate.chunks_per_s > 0.0)
        & (rate.speed_mps > 0.0)
        & (rate.chunks_per_column > 0.0)
        & (rungs > 0);
    if !measured {
        return AskBound::unbounded();
    }
    let mut switches = vec![0.0f64; usize::from(rungs)];
    let mut budget = rate.chunks_per_s;
    let mut binds = false;
    let mut rung = rungs;
    while rung > 0 {
        rung -= 1;
        let full = switch_m(rung);
        let need = ask_rate_per_s(rung, HYSTERESIS_OUT * full, rate);
        let fits = need <= budget;
        // The rung the budget covers keeps the tier rule's radius; the rung it runs out on keeps
        // what the rest of the budget pays for, read back through the same strip.
        let reach = ask_reach_m(rung, budget, rate) / HYSTERESIS_OUT;
        switches[usize::from(rung)] = if fits { full } else { reach.min(full) };
        budget = if fits { budget - need } else { 0.0 };
        binds |= !fits;
    }
    if !binds {
        return AskBound::unbounded();
    }
    // THE LADDER'S OWN SHAPE under the binding rung: a finer rung's switch is at most half its
    // coarser neighbour's, exactly as the tier rule's own are. Without it two bound rungs can land
    // on one distance, and two rungs that share a crossfade band draw a half-transparent shell.
    let mut rung = usize::from(rungs) - 1;
    while rung > 0 {
        rung -= 1;
        switches[rung] = switches[rung].min(switches[rung + 1] * 0.5);
    }
    AskBound {
        switches: Some(switches),
        slack: 0.0,
        // The builders' own bound reads no body; the caller states the body's row
        // ([`AskBound::for_body`], ruling T7 rules 2 and 3).
        steps_m: Vec::new(),
    }
}

/// ★ HOW FAR THE DRAWN HORIZON MAY SLIDE BEFORE THE DESCENT FOLLOWS IT: a tenth.
///
/// The descent is the costliest thing the terrain system does on the main thread — MEASURED on the
/// 528 m/s leg (§25.7): 9.7 ms a frame with the bound off, and 18.1 ms with it on, because the
/// sliding horizon forced a descent for every body on every frame. The DRAWN bands must still
/// slide (a jumped band is a pop), so the two are separated: the bands slide every frame, and the
/// descent re-runs only when the horizon has left the ring it last asked for. At the slew's quarter
/// of a length a second a tenth is about two fifths of a second.
///
/// The descent's own ask is widened by [`ASK_BOUND_SLACK`] ([`AskBound::with_slack`]) — DERIVED
/// from this fraction and the materials' own, never equal to this one — so the ring it asked for
/// covers the bands the picture draws until the next descent.
///
/// ★ AND THE DESCENT MAY BE RATE-LIMITED BESIDES ([`ASK_BOUND_DESCENT_S`], zero as shipped), which
/// would cap the descents a body's bound can force a second whatever the slew or the speed.
pub const ASK_BOUND_BRACKET: f64 = 0.10;

/// HOW OFTEN THE BOUND MAY FORCE A DESCENT, in seconds, per body: ZERO — the bracket alone paces
/// it. MEASURED (§25.7): a limit of half a second read 40.1 frames a second against 41.8 without
/// one, because the crossings were never the cost; the guard stays as the lever it is, and its
/// other arm is tested (`ask_pace::tests::the_rate_limit_refuses_a_descent_until_its_interval_has_passed`).
///
/// The descent is the costliest thing the terrain system does on the main thread (§25.7), and it
/// already runs whenever the eye moves half a metre. At ZERO the pacing is the BRACKET's alone:
/// the descent follows when the horizon has left the ring it asked for, and the slack below is
/// what makes that safe. A nonzero value would cap the descents a body's bound can force per
/// second, at the cost of a horizon the picture may outrun — which is why the slack would have to
/// grow with it.
pub const ASK_BOUND_DESCENT_S: f64 = 0.0;

/// ★ HOW MUCH WIDER THE DESCENT ASKS THAN THE HORIZON IT RAN ON: about a sixth, DERIVED from the
/// two tolerances it must cover (review item 3). It is not a taste.
///
/// The picture and the descent read the horizon at two different tolerances. The materials follow
/// the held horizon within [`ASK_BOUND_REBIND`], so the DRAWN base may stand as far out as
/// `1 / (1 − REBIND)` of the held one. The descent follows within [`ASK_BOUND_BRACKET`], so the
/// ASKED base may stand as far in as `1 − BRACKET` of it. At the worst of both at once the drawn
/// base is `1 / ((1 − REBIND)(1 − BRACKET))` — about 1.170 — of the asked base, and every band is
/// a fixed multiple of its base, so the drawn band's outer edge stands 1.170 times the asked one's
/// unless the ask is widened by exactly that much.
///
/// A HOLE IS THEREFORE NOT "IMPOSSIBLE BY CONSTRUCTION" BY ITSELF — it is impossible because this
/// number is derived from the two tolerances and asserted against them below. With the slack at
/// the bracket alone (a tenth) the drawn edge could reach 1.170 times the asked base against an
/// asked outer edge of 1.10, and the picture could draw a band the descent never asked for.
/// The derivation reads 0.1696 at today's two tolerances; the number is rounded up to a
/// seventeen-hundredth so it is a figure a person can hold, and the assertions below fail the
/// BUILD if either tolerance ever moves past it.
pub const ASK_BOUND_SLACK: f64 = vd_terrain::artifact::ASK_SLACK;

// THE DERIVATION, ASSERTED WHERE IT CANNOT ROT: the slack covers the worst simultaneous stand of
// the two tolerances (the drawn edge inside the asked edge), and it is never less than their sum.
const _: () =
    assert!((1.0 + ASK_BOUND_SLACK) * (1.0 - ASK_BOUND_REBIND) * (1.0 - ASK_BOUND_BRACKET) >= 1.0);
const _: () = assert!(ASK_BOUND_SLACK >= ASK_BOUND_BRACKET + ASK_BOUND_REBIND);

impl AskBound {
    /// The tier rule's own radii: the bound does not bind.
    #[must_use]
    pub fn unbounded() -> AskBound {
        AskBound {
            switches: None,
            slack: 0.0,
            steps_m: Vec::new(),
        }
    }

    /// ★ THE SAME BOUND, HAVING READ A BODY (ruling T7 rules 2 and 3): every rung's handover step
    /// from the body's own octave table, so the floor and the widening are the body's own. A rung
    /// with no coarser neighbour has no handover and no step.
    #[must_use]
    pub fn for_body(mut self, body: &BodyDefinition, rungs: u8) -> AskBound {
        let mut steps = Vec::with_capacity(usize::from(rungs));
        let mut rung = 0u8;
        while rung + 1 < rungs {
            steps.push(handover_step_m(body, rung));
            rung += 1;
        }
        self.steps_m = steps;
        self
    }

    /// The handover step at a rung, in metres: zero where no body has been read and zero at the top
    /// rung, which hands over to nobody.
    #[must_use]
    pub fn step_m(&self, rung: u8) -> f64 {
        self.steps_m.get(usize::from(rung)).copied().unwrap_or(0.0)
    }

    /// A BOUND STATED OUTRIGHT, one effective switch distance per rung, finest first — what
    /// [`ask_bound`] builds, and what a test builds to stand a horizon exactly where it wants it.
    #[must_use]
    pub fn from_switches(switches: Vec<f64>) -> AskBound {
        AskBound {
            switches: Some(switches),
            slack: 0.0,
            steps_m: Vec::new(),
        }
    }

    /// THE SAME BOUND, ASKED WIDER: every band's edges pushed out by `slack` (see the field). The
    /// descent carries this; the bound the crossfade's materials carry never does.
    #[must_use]
    pub fn with_slack(mut self, slack: f64) -> AskBound {
        self.slack = slack.max(0.0);
        self
    }

    /// Whether the bound moves any rung in from the tier rule's own radius.
    #[must_use]
    pub fn binds(&self) -> bool {
        self.switches.is_some()
    }

    /// A rung's EFFECTIVE switch distance, in metres: the tier rule's own where the bound does not
    /// bind, and a rung past the bound's own ladder reads the tier rule's too — never nearer than
    /// the body's own floor ([`switch_floor_m`], ruling T7 rule 3), which is zero until a body has
    /// been read and inert on the home planet's table.
    #[must_use]
    pub fn switch_m(&self, rung: u8) -> f64 {
        let base = match &self.switches {
            None => switch_m(rung),
            Some(s) => s
                .get(usize::from(rung))
                .copied()
                .unwrap_or_else(|| switch_m(rung)),
        };
        base.max(switch_floor_m(self.step_m(rung)))
    }

    /// Every rung's effective switch distance, finest first — the stamp's readout. Empty where the
    /// bound does not bind.
    #[must_use]
    pub fn horizons_m(&self) -> Vec<f64> {
        self.switches.clone().unwrap_or_default()
    }

    /// Whether `other` is the same bound for the ask's purposes: every rung of the two within
    /// `fraction` of the larger of the pair ([`ASK_BOUND_HYSTERESIS`]). Symmetric, and never
    /// zero-width, so a horizon that jitters is never adopted and one that really moves always is.
    #[must_use]
    pub fn same_as(&self, other: &AskBound, rungs: u8, fraction: f64) -> bool {
        let mut rung = 0u8;
        let mut same = true;
        while rung < rungs {
            let (a, b) = (self.switch_m(rung), other.switch_m(rung));
            same &= (a - b).abs() <= fraction * a.abs().max(b.abs());
            rung += 1;
        }
        same
    }

    /// ★ THE HORIZON SLIDES, IT NEVER JUMPS (ruling F9 item 1): the bound this one becomes after
    /// `dt_s` seconds of moving toward `target`, each rung by at most [`ASK_BOUND_SLEW_PER_S`] of
    /// its own horizon plus its own column's width. A rung already at its target stays there
    /// exactly, so a settled bound stops changing and the descent stops recomputing.
    ///
    /// **Example.** The pilot's hull pushes from a hover to 528 m/s. The measurement asks at once
    /// for four-metre cells at 2.1 km instead of 3.5 km; the horizon walks the 1.4 km in about a
    /// second and a half, and the pilot sees the eight-metre ring fade in over the four-metre one
    /// exactly as it fades in when the hull flies toward it.
    #[must_use]
    pub fn slewed_toward(&self, target: &AskBound, rungs: u8, dt_s: f64) -> AskBound {
        let mut switches = vec![0.0f64; usize::from(rungs)];
        let mut free = true;
        let mut rung = 0u8;
        while rung < rungs {
            let now = self.switch_m(rung);
            let want = target.switch_m(rung);
            let step = (now + column_span_m(rung)) * ASK_BOUND_SLEW_PER_S * dt_s.max(0.0);
            let gap = want - now;
            let next = if gap.abs() <= step {
                want
            } else {
                now + step.copysign(gap)
            };
            switches[usize::from(rung)] = next;
            free &= next >= switch_m(rung);
            rung += 1;
        }
        if free {
            return AskBound {
                switches: None,
                slack: 0.0,
                steps_m: target.steps_m.clone(),
            };
        }
        AskBound {
            switches: Some(switches),
            slack: 0.0,
            // ★ THE BODY'S OWN ROW COMES FROM THE TARGET (ruling T7): the bound we move toward is
            // the one built from the body this frame, and the body's table does not slide.
            steps_m: target.steps_m.clone(),
        }
    }

    /// THE BANDS of a rung under this bound — [`fade_bands`] read at the effective switch
    /// distances. A rung whose finer neighbour is bound in to nothing has no finer rung to fade in
    /// from at all, and is always in, exactly as rung 0 is.
    #[must_use]
    pub fn fade_bands(&self, rung: u8, rungs: u8) -> ([f64; 2], [f64; 2]) {
        // THE SLACK GOES ONLY WHERE THE HORIZON MOVES: a rung the bound left at the tier rule's
        // own radius never slides, so widening it buys nothing and costs the descent every column
        // of the widening. MEASURED (§25.7): widening every rung cost 0.23 ms of a 3.53 ms
        // descent, and the coarse rungs — whose rings are the widest of all — never moved.
        let slack = |s: f64, rung: u8| {
            if s < switch_m(rung) { self.slack } else { 0.0 }
        };
        // ★ THE BAND WIDENS BY THE HANDOVER'S OWN STEP (ruling T7 rule 2): the ladder's own half
        // width was sized on a step of one cell, so a step of `r` cells is paid off over `r` times
        // the ground and falls at the same rate on the screen. INERT where the step stands under a
        // cell, which is every pair of the home planet's table after rule 1.
        let band = |s: f64, r: u8| {
            let k = slack(s, r);
            // The widening, as the EXTRA over the ladder's own band: at one it is zero and the two
            // edges are the constants themselves, bit for bit, which is why no flight moves.
            let extra = band_widen(self.step_m(r), r) - 1.0;
            let lo = (HYSTERESIS_IN - (1.0 - HYSTERESIS_IN) * extra).max(0.0);
            let hi = HYSTERESIS_OUT + (HYSTERESIS_OUT - 1.0) * extra;
            [lo * s * (1.0 - k), hi * s * (1.0 + k)]
        };
        let fade_in = if rung == 0 {
            FADE_ALWAYS_IN
        } else {
            let finer = self.switch_m(rung - 1);
            if finer > 0.0 {
                band(finer, rung - 1)
            } else {
                FADE_ALWAYS_IN
            }
        };
        let fade_out = if rung + 1 >= rungs {
            FADE_ALWAYS_OUT
        } else {
            band(self.switch_m(rung), rung)
        };
        (fade_in, fade_out)
    }

    /// ★ THE OUTER EDGE OF A RUNG'S OWN TERRITORY, in metres: the distance past which the coarser
    /// rung draws the ground instead. It is the rung's effective switch distance — and for the TOP
    /// rung there is no coarser rung to hand to, so its territory reaches EVERYWHERE, exactly as
    /// its fade-out band is [`FADE_ALWAYS_OUT`] and the tier rule's own
    /// [`rung_for_distance`] clamps to it. ONE branch, the same one [`AskBound::fade_bands`]
    /// already makes, never a test on how far the eye stands.
    ///
    /// MEASURED before this (the far-eye probe, 2026-09-15): an eye 1.5 radii over the home planet
    /// wanted 2 015 top-rung chunks and called 1 857 of them MARGIN — the lowest request class,
    /// the one that means "a finer ring already covers this ground" — because the top rung's own
    /// switch distance is 0.56 radii and every column stood past it. The band could never read
    /// incomplete aloft, whatever was missing, and the workers built the ground the pilot was
    /// looking at last.
    #[must_use]
    pub fn territory_m(&self, rung: u8, rungs: u8) -> f64 {
        if rung + 1 >= rungs {
            return f64::INFINITY;
        }
        self.switch_m(rung)
    }

    /// WHERE THE SINK RAMP ENDS for a rung under this bound — [`sink_end_m`] read at the effective
    /// switch distances.
    #[must_use]
    pub fn sink_end_m(&self, body: &BodyDefinition, rung: u8, rungs: u8) -> f64 {
        let (fade_in, _) = self.fade_bands(rung, rungs);
        if fade_in[0] == FADE_ALWAYS_IN[0] {
            return fade_in[1];
        }
        let crease = f64::from(cell_m(rung - 1));
        let sink = crate::chunks::sink_m(body, rung);
        // sink > crease always: the sink holds a cell of each rung and the gap bound.
        fade_in[1] + (fade_in[1] - fade_in[0]) * crease / (sink - crease)
    }
}

/// THE HORIZON of a smooth sphere of `radius_m` seen from `altitude_m` over it, in metres along the
/// line of sight: `√(2Rh + h²)`. A height under the surface reads as zero. The one formula the stamp,
/// the gate and the reach share.
#[must_use]
pub fn horizon_m(radius_m: f64, altitude_m: f64) -> f64 {
    vd_terrain::artifact::horizon_m(radius_m, altitude_m)
}

/// THE EYE'S HEIGHT over the ground it stands on, in metres: the pilot camera lifts the eye by
/// this over the avatar's feet (the harness's own offset reads it from here), and it is THE FLOOR
/// of the WANTED SET's altitude (slice 8 step 6, D-TERRAIN-5 item 12): an eye that the recipe's
/// surface stands above — in a dip the mesh cuts under the field, in a cave, for the frame of a
/// hard landing — wants what an eye standing on that surface wants, never a zero horizon's set.
/// The stamp still states the eye's true altitude and its own horizon, unfloored.
/// MEASURED before the floor (M8-1's first run): 1 930 chunks wanted on a walk from an eye the
/// field stood over, the horizon zero and every column past it a skyline candidate.
pub const EYE_HEIGHT_M: f64 = 1.6;

/// The wanted set's altitude: the eye's height over the recipe's surface under it, never under
/// the eye's own height ([`EYE_HEIGHT_M`]).
#[must_use]
pub fn floored_altitude_m(altitude_m: f64) -> f64 {
    altitude_m.max(EYE_HEIGHT_M)
}

/// THE REACH of the ladder from an eye `altitude_m` over a surface of `surface_m`: its own horizon
/// plus the horizon of the tallest ground the recipe can raise (`relief_m`), so a peak standing
/// behind the geometric horizon is still wanted.
#[must_use]
pub fn reach_m(surface_m: f64, altitude_m: f64, relief_m: f64) -> f64 {
    vd_terrain::artifact::reach_m(surface_m, altitude_m, relief_m)
}

/// The tallest ground the recipe can raise over its radius, in metres: the recipe's own bound.
#[must_use]
pub fn relief_m(body: &BodyDefinition) -> f64 {
    body.relief_bound_m(0)
}

/// Where a wanted chunk stands for the band: in the hysteresis margin past its rung's switch, in
/// the rung's own territory inside the eye's horizon (URGENT), or in that territory past the
/// horizon (REVEALED).
#[derive(Clone, Copy)]
enum Territory {
    Margin,
    Urgent,
    Revealed,
}

impl Territory {
    /// The request order's first key: the picture's need now, then the peaks it sees, then the
    /// bands' overlap a finer ring already covers.
    fn rank(self) -> u8 {
        match self {
            Territory::Urgent => 0,
            Territory::Revealed => 1,
            Territory::Margin => 2,
        }
    }
}

/// THE REQUEST ORDER of one wanted chunk (ruling V15, M8-2a): the class, then the coarser rung
/// first (a missing coarse chunk is a hole, a missing fine chunk a coarser patch), then the
/// PARENT column along a MORTON curve over its face, then the chunk — so the four children of one
/// parent and the neighbours on every side build back to back, and the workers' parent cache holds
/// their parents. MEASURED: 60 ms a chunk on a flight with the cache missing, 5 ms warm; with the
/// parents ordered nearest-first the fourteenth run still paid 46 ms, because two parents at one
/// distance stand anywhere around the eye and their eight-parent neighbourhoods rarely overlap.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RequestOrder {
    class: u8,
    depth: u8,
    face: Face,
    parent_morton: u64,
}

/// The Morton code of a column pair: the bits of `x` and `y` interleaved, so two codes close in
/// value are two columns close on the face (in both axes, most of the time). A column index is
/// never negative (a face's columns count from zero); a negative is clamped to zero, so it can
/// never break the order, and it is not a column. The order groups by FACE: a chunk on a face
/// edge reads a parent across the edge (`parent_keys`), and that parent is ordered with the other
/// face — the one seam the curve does not cover.
#[must_use]
pub fn morton(x: i32, y: i32) -> u64 {
    let mut code = 0u64;
    let (x, y) = (x.max(0) as u64, y.max(0) as u64);
    let mut bit = 0;
    while bit < 32 {
        code |= ((x >> bit) & 1) << (2 * bit);
        code |= ((y >> bit) & 1) << (2 * bit + 1);
        bit += 1;
    }
    code
}

/// A column of chunks: a face, a rung, and the chunk index across the face. Two columns on one face
/// OVERLAP when one's footprint holds the other's (the same column, or an ancestor at a coarser
/// rung); columns on different faces never overlap, the faces tile the sphere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Column {
    pub face: Face,
    pub rung: u8,
    pub x: i32,
    pub y: i32,
}

impl Column {
    /// The column of a chunk.
    #[must_use]
    pub fn of(key: ChunkKey) -> Column {
        Column {
            face: key.face,
            rung: key.rung,
            x: key.x,
            y: key.y,
        }
    }

    /// This column's index at a coarser rung `up` rungs above it.
    #[must_use]
    pub fn coarser(self, up: u8) -> (i32, i32) {
        (self.x >> up, self.y >> up)
    }
}

/// The wanted chunks of one face and rung, by column.
type ColumnIndex = BTreeMap<(i32, i32), Vec<ChunkKey>>;

/// THE SHADOW'S REACH, for the casters (D-TERRAIN-5 item 18): how far the sun's cascades reach from
/// the eye, the tangent of the sun's incidence at the eye, and how many rungs coarser a caster
/// stands than the chunk that asks for it. A drawn chunk asks for its coarse caster only while that
/// caster's shadow can fall on ground the cascades cover — see [`ShadowReach::caster_bound_m`].
/// MEASURED without it (§19.10): 858 casters, 221 MB, at the ground stand, most past the shadow's
/// reach and never cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowReach {
    /// The cascades' farthest distance from the eye, in metres.
    pub reach_m: f64,
    /// The tangent of the sun's incidence at the eye (zero overhead; capped by the caller at a
    /// low sun, as the shadow bias caps it).
    pub tan_i: f64,
    /// How many rungs coarser a caster stands than the drawn chunk that asks for it.
    pub coarse_step: u8,
}

/// How much the sun's tangent may change before the casting set is recomputed: a tenth of the
/// larger tangent, plus [`SHADOW_TAN_FLOOR`] (so a sun at the zenith, whose tangent is zero, does
/// not recompute on every frame's rounding).
pub const SHADOW_TAN_HYSTERESIS: f64 = 0.1;
pub const SHADOW_TAN_FLOOR: f64 = 0.01;

impl ShadowReach {
    /// The rung the caster of a drawn chunk of `rung` stands at, on a ladder whose top rung is
    /// `top`: never past the top (refutation, 2026-09-12: a clamp at the global `RUNG_MAX` sampled
    /// columns on rungs the body does not have). A chunk of the top rung has no caster.
    #[must_use]
    pub fn caster_rung(&self, rung: u8, top: u8) -> u8 {
        rung.saturating_add(self.coarse_step).min(top)
    }

    /// THE CASTER BOUND for a caster column that peaks at `peak_m` (a radius) over ground within
    /// the reach no lower than `low_m` (a radius): the farthest the CASTER'S nearest point may
    /// stand from the eye and still throw a shadow onto that ground. The shadow of a peak
    /// `peak − low` over the ground reaches `(peak − low) × tan` along it — nothing when the peak
    /// stands under the ground. Both terms over-estimate, so a caster that could cast is always
    /// asked for. MEASURED with the whole relief in place of the peak (the ground stand, the sun
    /// 15° up): 858 → 838 casters — a 5 km relief at a low sun reaches 40 km, so the peak's own
    /// height is the bound that bites; and with the caster's diagonal added to the bound in place
    /// of its own nearest distance, every coarse chunk within a rung-12 caster's 227 km asked.
    #[must_use]
    pub fn caster_bound_m(&self, peak_m: f64, low_m: f64) -> f64 {
        let shadow_m = (peak_m - low_m).max(0.0) * self.tan_i;
        self.reach_m + shadow_m
    }

    /// Whether `other` is the same reach for the casting set's purposes: the same reach and step,
    /// and tangents within [`SHADOW_TAN_HYSTERESIS`] of the larger one plus [`SHADOW_TAN_FLOOR`]
    /// — symmetric, and never zero-width.
    #[must_use]
    pub fn same_as(&self, other: ShadowReach) -> bool {
        let width =
            SHADOW_TAN_HYSTERESIS * self.tan_i.abs().max(other.tan_i.abs()) + SHADOW_TAN_FLOOR;
        (self.reach_m == other.reach_m)
            & (self.coarse_step == other.coarse_step)
            & ((self.tan_i - other.tan_i).abs() <= width)
    }
}

/// THE WANTED SET: the chunks the ladder wants for one body, coarsest ring first, indexed by column
/// so the release hold is a lookup and never a scan (SL9).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WantedSet {
    /// Every wanted chunk, in request order: the coarsest ring first.
    pub keys: Vec<ChunkKey>,
    /// The wanted chunks by face and rung, then by column.
    index: BTreeMap<(Face, u8), ColumnIndex>,
    set: BTreeSet<ChunkKey>,
    /// THE URGENT CHUNKS (slice 8 step 4): the wanted chunks of columns INSIDE THE EYE'S HORIZON
    /// whose nearest point lies inside their rung's own territory, nearer than the rung's switch
    /// distance — the ground the picture draws at that rung NOW, which the eye's motion carries
    /// it into. The rest of a ring is the hysteresis margin past the switch (a fifth of it),
    /// where the coarser rung still stands: ground asked for ahead of need. A missing urgent
    /// chunk is the residency band incomplete; a missing margin chunk is not. M8-1 reads the
    /// count of missing urgent chunks per frame on a moving eye.
    urgent: BTreeSet<ChunkKey>,
    /// THE REVEALED CHUNKS: the wanted chunks of columns PAST THE HORIZON inside their rung's
    /// territory — peaks the skyline admits. One that is not resident is a peak the eye can see
    /// before it is built (a reveal over a crest): not the band's motion, and not predictable
    /// by any lead; the want margin builds most before they show, and the pop detector (step
    /// 6) judges the rest. Counted apart, so the band's verdict stays the band's.
    revealed: BTreeSet<ChunkKey>,
    /// THE CASTING CHUNKS (item 18): the wanted chunks whose coarse caster may throw a shadow
    /// onto ground within the sun's reach ([`ShadowReach::caster_bound_m`]); meaningful only
    /// while `bounded` — a descent without a shadow reach lets every chunk cast.
    casting: BTreeSet<ChunkKey>,
    bounded: bool,
    /// THE ALTITUDE THIS DESCENT RAN AT, in metres over the recipe's surface under the eye,
    /// floored as [`floored_altitude_m`] floors it. The bounded ask reads it (ruling F9 item 1):
    /// a ring whose slant distance does not reach the ground has no ask at all, and only the
    /// RECIPE'S surface says where the ground is — the ladder's own radius is its FLOOR, which
    /// stands kilometres under the recipe, and reading that in its place made every near ring look
    /// empty and the bound inert (MEASURED, the second bounded flight of 2026-09-13).
    pub altitude_m: f64,
    /// How far the ladder reaches, in metres, and the rungs it holds.
    pub reach_m: f64,
    pub rung_min: u8,
    pub rung_max: u8,
}

/// ★ WHAT A DESCENT CALLED A CHUNK (2026-09-16, the walk-gap instrument), stated outside this
/// module: `Urgent` is ground the picture draws at that rung now, `Revealed` a peak past the
/// horizon, `Margin` ground a coarser rung still stands over, and `Absent` a chunk the set does
/// not want at all. [`Territory`] stays private: it is the descent's own word, and this is the
/// instrument's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandClass {
    Absent,
    Margin,
    Revealed,
    Urgent,
}

impl BandClass {
    /// The class's own word, for a stamp that carries strings.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            BandClass::Absent => "absent",
            BandClass::Margin => "margin",
            BandClass::Revealed => "revealed",
            BandClass::Urgent => "urgent",
        }
    }
}

/// ★ ONE CHUNK'S READING FROM ONE EYE (see [`LadderView::probe`]): the column's nearest and
/// farthest points from that eye in metres, the rung's own territory edge, the eye's horizon, and
/// whether the pair of them call the chunk urgent. All zero, and not urgent, for an eye at the
/// body's own centre.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BandProbe {
    pub near_m: f64,
    pub far_m: f64,
    pub territory_m: f64,
    pub horizon_m: f64,
    pub urgent: bool,
}

impl WantedSet {
    /// A wanted set from keys, in the order given.
    #[must_use]
    pub fn from_keys(keys: Vec<ChunkKey>, reach_m: f64) -> WantedSet {
        let mut out = WantedSet {
            reach_m,
            ..WantedSet::default()
        };
        for key in keys {
            out.push(key);
        }
        out
    }

    fn push(&mut self, key: ChunkKey) {
        let fresh = self.set.insert(key);
        let first = self.keys.is_empty();
        // Branchless bounds: the first key sets both, a later key widens.
        self.rung_min = if first {
            key.rung
        } else {
            self.rung_min.min(key.rung)
        };
        self.rung_max = self.rung_max.max(key.rung);
        if fresh {
            self.keys.push(key);
            self.index
                .entry((key.face, key.rung))
                .or_default()
                .entry((key.x, key.y))
                .or_default()
                .push(key);
        }
    }

    /// Whether the set wants a chunk.
    #[must_use]
    pub fn contains(&self, key: ChunkKey) -> bool {
        self.set.contains(&key)
    }

    /// Whether a wanted chunk may ask for its coarse caster (item 18): every chunk while the
    /// descent had no shadow reach, else the casting set's own.
    #[must_use]
    pub fn casts(&self, key: ChunkKey) -> bool {
        !self.bounded | self.casting.contains(&key)
    }

    /// How many wanted chunks may ask for a caster.
    #[must_use]
    pub fn casting_count(&self) -> usize {
        if self.bounded {
            self.casting.len()
        } else {
            self.keys.len()
        }
    }

    /// Mark a wanted chunk's territory.
    fn mark(&mut self, key: ChunkKey, territory: Territory) {
        match territory {
            Territory::Margin => {}
            Territory::Urgent => {
                self.urgent.insert(key);
            }
            Territory::Revealed => {
                self.revealed.insert(key);
            }
        }
    }

    /// Whether a wanted chunk is urgent.
    #[must_use]
    pub fn is_urgent(&self, key: ChunkKey) -> bool {
        self.urgent.contains(&key)
    }

    /// ★ WHAT THIS SET CALLED A CHUNK (2026-09-16, the walk-gap instrument): the class the descent
    /// gave it, or [`BandClass::Absent`] where the set does not want it at all. The band's gap
    /// reads the PREVIOUS descent's set through this, so a missing urgent chunk can say whether
    /// the ring had already asked for it (the builders are late) or whether this very descent
    /// first wanted it (the ask is late). Example: the walker crosses a crest, a rung-3 column
    /// the skyline hid comes inside the horizon, and the frame that first wants it counts it
    /// missing — `Absent` names that, `Urgent` would name a slow builder.
    #[must_use]
    pub fn class_of(&self, key: ChunkKey) -> BandClass {
        if self.urgent.contains(&key) {
            BandClass::Urgent
        } else if self.revealed.contains(&key) {
            BandClass::Revealed
        } else if self.set.contains(&key) {
            BandClass::Margin
        } else {
            BandClass::Absent
        }
    }

    /// THE JOB'S PRIORITY (the lower builds first) of the chunk at `index` in `keys`: a GLOBAL
    /// order, the same across every realm's set (refutation T-2: an index alone let a moon's
    /// margin chunk outrank the planet's urgent one) — the class in the top two bits, then the
    /// depth (the coarser rung first) in six, then the index in the set, which the request order
    /// already sorted by parent along the Morton curve.
    #[must_use]
    pub fn priority_of(&self, index: usize, key: ChunkKey) -> u32 {
        let class: u32 = if self.urgent.contains(&key) {
            0
        } else if self.revealed.contains(&key) {
            1
        } else {
            2
        };
        let depth = 63u32.saturating_sub(u32::from(key.rung));
        (class << 30) | (depth << 24) | (index as u32 & 0x00FF_FFFF)
    }

    /// How many chunks are urgent.
    #[must_use]
    pub fn urgent_count(&self) -> usize {
        self.urgent.len()
    }

    /// How many chunks are revealed peaks past the horizon.
    #[must_use]
    pub fn revealed_count(&self) -> usize {
        self.revealed.len()
    }

    /// THE BAND'S GAP: how many urgent chunks have NOT `arrived` — zero when every chunk the
    /// picture draws now, inside the horizon, is resident. A lookup per urgent chunk, never a
    /// scan of the lane.
    #[must_use]
    pub fn urgent_missing(&self, arrived: &dyn Fn(ChunkKey) -> bool) -> usize {
        self.urgent.iter().filter(|k| !arrived(**k)).count()
    }

    /// THE BAND'S GAP PER RUNG: the urgent chunks that have NOT `arrived`, counted by rung
    /// (rungs with no gap are absent) — which ring of the ladder a moving eye outruns.
    #[must_use]
    pub fn urgent_missing_per_rung(&self, arrived: &dyn Fn(ChunkKey) -> bool) -> Vec<(u8, u64)> {
        let mut counts: BTreeMap<u8, u64> = BTreeMap::new();
        for key in self.urgent.iter().filter(|k| !arrived(**k)) {
            *counts.entry(key.rung).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// ★ THE BAND'S GAP, NAMED (2026-09-16, the walk-gap measurement): the urgent chunks that have
    /// NOT `arrived`, at most `cap` of them, coarsest rung first. The counts alone say how many the
    /// band lacks; a cure needs to know WHICH, so the gap line can read each one's class in the
    /// previous descent's set and its distance from the drawn eye.
    #[must_use]
    pub fn urgent_missing_keys(
        &self,
        arrived: &dyn Fn(ChunkKey) -> bool,
        cap: usize,
    ) -> Vec<ChunkKey> {
        let mut out: Vec<ChunkKey> = self
            .urgent
            .iter()
            .filter(|k| !arrived(**k))
            .take(cap)
            .copied()
            .collect();
        out.sort_by(|a, b| b.rung.cmp(&a.rung).then(a.cmp(b)));
        out
    }

    /// THE REVEALS' GAP: how many revealed chunks have NOT `arrived`.
    #[must_use]
    pub fn revealed_missing(&self, arrived: &dyn Fn(ChunkKey) -> bool) -> usize {
        self.revealed.iter().filter(|k| !arrived(**k)).count()
    }

    /// ★ THE MARGIN'S GAP (the coast flight's hole instrument, 2026-09-20): the wanted chunks of
    /// the margin class — outside their rung's own territory, inside the widened ask — that have
    /// NOT `arrived`; a scan of the set, for an instrument.
    #[must_use]
    pub fn margin_missing(&self, arrived: &dyn Fn(ChunkKey) -> bool) -> usize {
        self.set
            .iter()
            .filter(|k| !self.urgent.contains(*k) && !self.revealed.contains(*k) && !arrived(**k))
            .count()
    }

    /// How many chunks are wanted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// HOW MANY COLUMNS the set holds — the wanted chunks divided by this is the chunks a column
    /// holds, which the bounded ask reads as the cost of uncovering one new column (ruling F9
    /// item 1). One on ground the surface crosses once, two where it crosses a chunk boundary.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.index.values().map(BTreeMap::len).sum()
    }

    /// THE CHUNKS A COLUMN HOLDS, as this descent measured it; `default` while nothing is wanted.
    #[must_use]
    pub fn chunks_per_column(&self, default: f64) -> f64 {
        let columns = self.columns();
        if columns == 0 {
            return default;
        }
        self.keys.len() as f64 / columns as f64
    }

    /// Whether nothing is wanted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The chunk counts per rung, finest first.
    #[must_use]
    pub fn per_rung(&self) -> Vec<(u8, u64)> {
        let mut counts: BTreeMap<u8, u64> = BTreeMap::new();
        for key in &self.keys {
            *counts.entry(key.rung).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// THE RELEASE HOLD: whether any wanted chunk over `col`'s footprint has NOT `arrived` — while
    /// one is still building, the chunk drawn there today stays (coarse before fine, SL8).
    #[must_use]
    pub fn overlapping_missing(&self, col: Column, arrived: &dyn Fn(ChunkKey) -> bool) -> bool {
        for ((_, rung), columns) in self.index.range((col.face, 0)..=(col.face, u8::MAX)) {
            if *rung >= col.rung {
                // The wanted column at this coarser-or-equal rung that holds `col`.
                let up = rung - col.rung;
                if columns
                    .get(&col.coarser(up))
                    .is_some_and(|keys| keys.iter().any(|k| !arrived(*k)))
                {
                    return true;
                }
            } else {
                // The wanted finer columns inside `col`'s footprint: a range over x, filtered on y.
                let down = col.rung - rung;
                let (x0, y0) = (col.x << down, col.y << down);
                let (x1, y1) = ((col.x + 1) << down, (col.y + 1) << down);
                let missing = columns
                    .range((x0, i32::MIN)..(x1, i32::MIN))
                    .filter(|((_, y), _)| (*y >= y0) & (*y < y1))
                    .any(|(_, keys)| keys.iter().any(|k| !arrived(*k)));
                if missing {
                    return true;
                }
            }
        }
        false
    }
}

/// THE LADDER VIEW of one body: the surface spans it has read (a span is a function of the seed,
/// read once per column and kept WHILE THE COLUMN IS VISITED — after every descent the spans of
/// columns the descent did not touch are dropped, so a flight never holds more than one descent's
/// columns; SL9, the refuter's finding), and the wanted set it computes for an eye.
#[derive(Debug, Default)]
pub struct LadderView {
    /// ★ A SPAN IS READ ON THE FIELD THE CHUNK IS BUILT ON (2026-09-20, the owner's coast flight;
    /// `vd_terrain::digest::surface_column_field`): with an artifact, the column's slices come
    /// from the artifact's own column read, never from the recipe's relief the artifact replaced
    /// — MEASURED 2.1 km apart on average at the coast stand, which asked slices with no ground
    /// in most fine columns and drew the sea sheet through the empty chunks. A span read while
    /// the rung's own field was not whole for the column (its tiles not here) is PROVISIONAL —
    /// taken from the finest whole pyramid level, or the recipe — so the column stays wanted and
    /// the coarser rung stands (ruling F9); it is read again at every descent until the field is.
    spans: BTreeMap<Column, SpanEntry>,
    generation: u64,
    /// THE KEPT COLUMNS (slice 8 step 4): every column the last descent wanted. A column past
    /// the horizon that was wanted is KEPT while its peak stands within [`KEEP_MARGIN_RAD`] under
    /// the skyline, and a new one is wanted only when it stands within [`WANT_MARGIN_RAD`] under
    /// it: hysteresis on the skyline's verdict. MEASURED on the walk of M8-1 without it: a
    /// column just at the skyline flipped between hidden and seen as the eye moved half a
    /// metre, and each flip released and rebuilt it — 4 chunks missing on 80 of 1 383 samples.
    /// Keeping is free (the chunk is resident); rebuilding is not.
    kept: BTreeSet<Column>,
    /// THE SHADOW'S REACH for the casters (item 18), the renderer's to set from the sun and its
    /// cascades before a descent; `None` lets every wanted chunk ask for its caster.
    pub shadow: Option<ShadowReach>,
    /// THE BOUNDED ASK (ruling F9 item 1), the renderer's to set from its measured throughput and
    /// the eye's delivered speed before a descent. The default is [`AskBound::unbounded`]: the
    /// tier rule's own radii, the ladder every earlier flight flew.
    pub bound: AskBound,
    /// THE CULLED COLUMNS: every column past the horizon the last descent judged hidden. Such a
    /// column stays skyline-judged until it lies well inside the horizon
    /// ([`HORIZON_HYSTERESIS`]): the horizon moves with the eye's height (a walker over a bump,
    /// 1.8 m to 2.5 m, moves it from 4.8 km to 5.7 km), and MEASURED on the walk of M8-1 a
    /// column hidden by the skyline at one step stood inside the horizon at the next, was
    /// wanted unconditionally, and was built for nothing — 2 chunks missing on 17 of 1 291
    /// samples.
    culled: BTreeSet<Column>,
}

/// One column's span in the view: the span, the descent that last visited it, and whether it was
/// read on a stand-in field (see [`LadderView::spans`]).
#[derive(Clone, Copy, Debug)]
struct SpanEntry {
    span: ColumnSpan,
    generation: u64,
    provisional: bool,
}

/// THE SPAN OF A COLUMN ON WHAT THE CLIENT HOLDS: the artifact's field at the column's rung when
/// it is whole for the column (final), else the finest whole pyramid level (provisional), else
/// the recipe's own relief (provisional with an artifact, final without one). Returns the span
/// and whether it is provisional.
#[must_use]
pub fn column_span(
    body: &BodyDefinition,
    artifact: Option<&ArtifactCache>,
    col: Column,
) -> (ColumnSpan, bool) {
    let recipe = || vd_terrain::digest::surface_column(body, col.face, col.rung, col.x, col.y);
    let Some(lattice) = body.macro_lattice() else {
        return (recipe(), false);
    };
    // ★ NO ARTIFACT YET: a body with a macro lattice STATES one (every solved body does), so its
    // recipe span is a stand-in until the head lands — PROVISIONAL, read again at every descent.
    // MEASURED before this (the coast flight's hole instrument, 2026-09-20): the first descents
    // ran before the head arrived, their recipe spans were kept as final, and 1 849 columns held
    // nothing but empty chunks at the recipe's slices, at rest, in every flight.
    let Some(artifact) = artifact else {
        return (recipe(), true);
    };
    let on = |field: &dyn vd_terrain::artifact::ZField| {
        vd_terrain::digest::surface_column_field(body, field, col.face, col.rung, col.x, col.y)
    };
    if let Some(field) = artifact.field_at_rung(&lattice, col.rung)
        && let Some(span) = on(&*field)
    {
        return (span, false);
    }
    let whole = (1..=artifact.head.levels)
        .find_map(|k| artifact.level(k))
        .and_then(|level| on(&**level));
    (whole.unwrap_or_else(recipe), true)
}

/// How far inside the horizon a column the skyline culled must lie before the horizon alone
/// wants it: a quarter of the horizon's distance.
pub const HORIZON_HYSTERESIS: f64 = 0.25;

/// How far under the skyline a KEPT far column's peak may stand and stay wanted, in radians:
/// about 3°, more than the near walls swing per frame at a hull's speed over a planet (a step
/// of 8 m at a wall 300 m off is 1.6°).
pub const KEEP_MARGIN_RAD: f64 = 0.05;
/// How far under the skyline a NEW far column's peak may stand and be wanted, in radians: about
/// 0.6°, so a peak a walk is about to reveal over a crest is built before it shows.
pub const WANT_MARGIN_RAD: f64 = 0.01;

/// THE GEOMETRY OF A COLUMN as the eye sees it: its centre's straight distance bounds (the nearest
/// and farthest point, by the circumscribed disc), its centre's central angle and azimuth from
/// the eye's foot, its footprint on the eye's chart (a wall is raised over it), and the
/// circumscribed CAP's angular radius a far column is judged over, with the margin that covers
/// chords read as arcs. The corners are the face's own bent directions at the column's four
/// corners, so a column near a face edge, wider than its nominal cell, is never under-read.
#[derive(Clone, Copy, Debug)]
struct ColumnGeometry {
    near: f64,
    far: f64,
    /// The centre's central angle from the eye's foot.
    phi: f64,
    az: f64,
    quad: [DVec2; 4],
    /// The circumscribed cap's angular radius, from the chord at the surface's radius.
    rho: f64,
}

fn column_geometry(
    ladder: &vd_seed::ladder::Ladder,
    frame: &EyeFrame,
    eye: DVec3,
    col: Column,
    surface: f64,
) -> ColumnGeometry {
    let n_l = ladder.cells_per_edge(col.rung);
    let edge = CHUNK_EDGE as i32;
    let at = |i: i32, j: i32| -> DVec3 {
        let i = i.min(n_l as i32);
        let j = j.min(n_l as i32);
        DVec3::from_array(direction(col.face, face_param(i, n_l), face_param(j, n_l))) * surface
    };
    let (x0, y0) = (col.x * edge, col.y * edge);
    let centre = at(x0 + edge / 2, y0 + edge / 2);
    let corners = [
        at(x0, y0),
        at(x0 + edge, y0),
        at(x0 + edge, y0 + edge),
        at(x0, y0 + edge),
    ];
    let mut r_out: f64 = 0.0;
    let mut quad = [DVec2::ZERO; 4];
    let mut i = 0;
    while i < 4 {
        r_out = r_out.max((corners[i] - centre).length());
        quad[i] = frame.chart(corners[i] / surface);
        i += 1;
    }
    let r_out = r_out * (1.0 + DISC_MARGIN);
    let dist = (centre - eye).length();
    let (arc_m, az) = frame.ground(centre / surface);
    ColumnGeometry {
        near: dist - r_out,
        far: dist + r_out,
        phi: arc_m / frame.radius_m,
        az,
        quad,
        rho: (r_out / surface).min(1.0).asin(),
    }
}

/// THE SWEEP of one descent: the ladder and body it walks, and the keys of each rung with their
/// territory, filled by both passes (the set lists the coarsest first).
struct Sweep<'a> {
    ladder: &'a vd_seed::ladder::Ladder,
    body: &'a BodyDefinition,
    /// THE BOUNDED ASK (ruling F9 item 1): the effective switch distance of every rung. The
    /// descent splits, emits and classes against these, never against the tier rule's own.
    bound: &'a AskBound,
    /// Each rung's keys with their territory.
    per_rung: Vec<Vec<(ChunkKey, Territory)>>,
    /// The shadow's reach (item 18), `None` when every chunk may cast.
    shadow: Option<ShadowReach>,
    /// THE LOWEST GROUND WITHIN THE SHADOW'S REACH, a sampled radius in metres, over the columns
    /// the descent met inside the reach: the ground a caster must stand over to shade.
    low_within_reach_m: f64,
}

impl Sweep<'_> {
    /// ONE STEP OF THE DESCENT for a column that is seen: split it into its four children while
    /// a child could carry weight (some point of this node lies short of the end of the band
    /// below it), and draw it — its span's chunks into the rung's own list — while some point of
    /// it carries weight.
    fn descend(
        &mut self,
        col: Column,
        geo: &ColumnGeometry,
        span: &ColumnSpan,
        inside_horizon: bool,
        next: &mut Vec<Column>,
    ) {
        let ladder = self.ladder;
        let body = self.body;
        let bound = self.bound;
        let per_rung = &mut self.per_rung;
        let edge = CHUNK_EDGE as i32;
        // THE BOUNDED ASK (ruling F9 item 1): a column SPLITS into the finer rung only while it
        // stands inside that rung's DELIVERABLE HORIZON; past the horizon it does not split, and
        // its own rung — whose territory now reaches in to the horizon — stands whole instead.
        let (fade_in, fade_out) = bound.fade_bands(col.rung, ladder.rungs);
        if (col.rung > 0) & (geo.near < fade_in[1]) {
            let child_rung = col.rung - 1;
            let last = (ladder.cells_per_edge(child_rung) as i32 - 1) / edge;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let child = Column {
                    face: col.face,
                    rung: child_rung,
                    x: col.x * 2 + dx,
                    y: col.y * 2 + dy,
                };
                if (child.x <= last) & (child.y <= last) {
                    next.push(child);
                }
            }
        }
        if (geo.far > fade_in[0]) & (geo.near < fade_out[1]) {
            // Inside the rung's OWN territory while some point of the column lies nearer than its
            // switch distance (the top rung's is past everything) AND past the finer rung's far edge
            // (the finer covers the ground up to there, so a column entering at its fade-in edge is
            // under the finer rung and not yet the picture's need — MEASURED on the walk of M8-1: a
            // coarser column crossing its fade-in edge counted as two missing chunks for a frame,
            // under a finer rung that stood whole). URGENT for a column inside the eye's horizon
            // (the ground the motion carries the eye into), REVEALED for one past it (a peak the
            // skyline admits).
            let inside =
                (geo.near < bound.territory_m(col.rung, ladder.rungs)) & (geo.far > fade_in[1]);
            let territory = match (inside, inside_horizon) {
                (false, _) => Territory::Margin,
                (true, true) => Territory::Urgent,
                (true, false) => Territory::Revealed,
            };
            let top_z = vd_terrain::digest::top_chunk_z(body, col.rung);
            // The lowest ground within the shadow's reach, for the casting set (item 18).
            if self.shadow.is_some_and(|s| geo.near <= s.reach_m) {
                self.low_within_reach_m = self.low_within_reach_m.min(span.sampled_low_m);
            }
            let mut z = span.lo;
            while z <= span.hi.min(top_z) {
                per_rung[col.rung as usize].push((
                    ChunkKey {
                        face: col.face,
                        rung: col.rung,
                        x: col.x,
                        y: col.y,
                        z,
                    },
                    territory,
                ));
                z += 1;
            }
        }
    }
}

impl LadderView {
    /// The surface span and peak of a column, read once per residency, stamped with this descent.
    fn span(
        &mut self,
        body: &BodyDefinition,
        artifact: Option<&ArtifactCache>,
        col: Column,
    ) -> ColumnSpan {
        let generation = self.generation;
        let entry = self
            .spans
            .entry(col)
            .and_modify(|e| {
                // A provisional span is read again: the tiles may be here now.
                if e.provisional {
                    let (span, provisional) = column_span(body, artifact, col);
                    e.span = span;
                    e.provisional = provisional;
                }
            })
            .or_insert_with(|| {
                let (span, provisional) = column_span(body, artifact, col);
                SpanEntry {
                    span,
                    generation,
                    provisional,
                }
            });
        entry.generation = generation;
        entry.span
    }

    /// How many spans the view holds.
    #[must_use]
    pub fn spans_held(&self) -> usize {
        self.spans.len()
    }

    /// How many of them are provisional (read on a stand-in field): a renderer re-runs the
    /// descent when the artifact changes while any is.
    #[must_use]
    pub fn provisional_spans(&self) -> usize {
        self.spans.values().filter(|e| e.provisional).count()
    }

    /// THE WANTED SET for an eye at `eye_m` in the body's frame: a DESCENT from the top rung. Every
    /// top-rung column of every face is a root; a node is DRAWN when some point of it has a weight
    /// (its farthest point past its fade-in band's start, its nearest point short of its fade-out
    /// band's end), and SPLIT into its four children while some point of it lies short of the end
    /// of the band below it (a child could have a weight); a node whose nearest point lies past the
    /// reach, or past the horizon with a peak under the sightline, is neither. Inside a band a
    /// point is covered by two rungs, whose dithers are complementary; outside by one. MEASURED
    /// before the descent: each rung's columns were chosen by their own centres, a coarse column
    /// whose centre said "finer" was dropped while two of its children said "coarser" and were
    /// dropped too — black rectangles on the hill and aloft pictures at every ring boundary, 4 818
    /// and 16 297 probe pixels of nothing.
    ///
    /// Empty for an eye at the body's own centre, where the descent has no radial to stand on.
    /// THAT IS THE ONLY GUARD (owner 2026-09-15, *"agree"*): the client has NO far edge, because
    /// the server's visibility radius is the only rule — a realm has a row in the pilot's window
    /// only inside it, so the ladder is never asked for a body the parent does not ship. At every
    /// distance the descent's own rules coarsen the body: past the top rung's fade-in band no
    /// column splits, so a far eye gets the top rung's six columns and nothing finer.
    /// ★ ONE CHUNK READ FROM ONE EYE (2026-09-16, the walk-gap instrument): where the chunk's
    /// column stands for an eye at `eye_m`, and whether THAT eye's own ring calls it urgent - its
    /// nearest point inside the rung's territory, its farthest past the finer rung's far edge, and
    /// its nearest inside the eye's own horizon. The descent asks at the LEAD eye and the band
    /// judges the same frame, so a gap line reads this at the DRAWN eye to say whether the picture
    /// needs the chunk yet.
    ///
    /// It runs ONE column's geometry and no descent: the skyline is not raised, so a column the
    /// skyline hides reads urgent here while a descent would not want it. The instrument states
    /// that; it is not the band's rule.
    #[must_use]
    pub fn probe(&self, body: &BodyDefinition, eye_m: [f64; 3], key: ChunkKey) -> BandProbe {
        let eye = DVec3::from_array(eye_m);
        let len = eye.length();
        let ladder = *body.ladder();
        if len.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            // An eye at the body's own centre has no radial: the descent wants nothing there.
            return BandProbe::default();
        }
        let d = eye / len;
        let surface = vd_terrain::height::height_m(body, [d.x, d.y, d.z], 0);
        let altitude = floored_altitude_m(len - surface);
        let horizon = horizon_m(surface, altitude);
        let frame = EyeFrame::new(eye);
        let col = Column::of(key);
        let geo = column_geometry(&ladder, &frame, eye, col, surface);
        let (fade_in, _) = self.bound.fade_bands(col.rung, ladder.rungs);
        let territory = self.bound.territory_m(col.rung, ladder.rungs);
        let inside = (geo.near < territory) & (geo.far > fade_in[1]);
        BandProbe {
            near_m: geo.near,
            far_m: geo.far,
            territory_m: territory,
            horizon_m: horizon,
            urgent: inside & (geo.near <= horizon),
        }
    }

    pub fn wanted(
        &mut self,
        body: &BodyDefinition,
        eye_m: [f64; 3],
        artifact: Option<&ArtifactCache>,
    ) -> WantedSet {
        let eye = DVec3::from_array(eye_m);
        let len = eye.length();
        let ladder = *body.ladder();
        if len.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            // Nothing wanted, nothing kept.
            self.spans.clear();
            self.kept.clear();
            self.culled.clear();
            return WantedSet::default();
        }
        let d = eye / len;
        let surface = vd_terrain::height::height_m(body, [d.x, d.y, d.z], 0);
        // THE FLOOR (item 12): an eye under the recipe's surface wants what an eye standing on
        // it wants.
        let altitude = floored_altitude_m(len - surface);
        let horizon = horizon_m(surface, altitude);
        let reach = reach_m(surface, altitude, relief_m(body));
        let edge = CHUNK_EDGE as i32;
        let rungs = ladder.rungs;
        let top = rungs.saturating_sub(1);
        self.generation += 1;
        let frame = EyeFrame::new(eye);
        let mut skyline = Skyline::new(len);
        let bound = self.bound.clone();
        let mut sweep = Sweep {
            ladder: &ladder,
            body,
            bound: &bound,
            per_rung: vec![Vec::new(); usize::from(rungs)],
            shadow: self.shadow,
            low_within_reach_m: f64::MAX,
        };
        // The roots: every top-rung column of every face.
        let n_top = ladder.cells_per_edge(top) as i32;
        let chunks_top = (n_top - 1) / edge + 1;
        let mut level: Vec<Column> = Vec::new();
        for face in Face::ALL {
            let mut y = 0;
            while y < chunks_top {
                let mut x = 0;
                while x < chunks_top {
                    level.push(Column {
                        face,
                        rung: top,
                        x,
                        y,
                    });
                    x += 1;
                }
                y += 1;
            }
        }
        // PASS A — inside the horizon: every column whose nearest point lies within the eye's own
        // horizon is seen (wanting a hidden one is safe), and every one raises its wall on the
        // skyline. A column whose nearest point lies past the horizon waits for pass B; one past
        // the reach is nothing. One rung per pass, the descent.
        let mut far: Vec<Column> = Vec::new();
        while !level.is_empty() {
            let mut next: Vec<Column> = Vec::new();
            for col in level {
                let geo = column_geometry(&ladder, &frame, eye, col, surface);
                if geo.near > reach {
                    continue;
                }
                // Inside the horizon — but a column the skyline culled last time stays with the
                // skyline until it lies well inside (the horizon's own hysteresis).
                let boundary = if self.culled.contains(&col) {
                    horizon * (1.0 - HORIZON_HYSTERESIS)
                } else {
                    horizon
                };
                if geo.near > boundary {
                    far.push(col);
                    continue;
                }
                let span = self.span(body, artifact, col);
                // The guaranteed floor of the DRAWN ground: the lowest sample less the bounds the
                // peak adds (the field's), less a cell (the extractor's mesh stands within a
                // cell of the field) and the sink its mesh may stand under while a finer rung is
                // drawn over it. Only a small column raises a wall: a wide one's chart quad
                // over-claims ground, and its floor is loose anyway.
                if geo.rho <= WALL_MAX_HALF_ANGLE {
                    let floor = span.sampled_low_m
                        - (span.peak_m - span.sampled_high_m)
                        - f64::from(cell_m(col.rung))
                        - crate::chunks::sink_m(body, col.rung);
                    skyline.raise(&geo.quad, floor);
                }
                sweep.descend(col, &geo, &span, true, &mut next);
            }
            level = next;
        }
        // PASS B — past the horizon, against the skyline the near ground raised: a column is seen
        // while its peak bound can show over the lowest wall at some azimuth it spans. A child of
        // a far column is far too (it lies inside its parent), and is judged on its own.
        let mut culled: BTreeSet<Column> = BTreeSet::new();
        let mut cleared: BTreeSet<Column> = BTreeSet::new();
        level = far;
        while !level.is_empty() {
            let mut next: Vec<Column> = Vec::new();
            for col in level {
                let geo = column_geometry(&ladder, &frame, eye, col, surface);
                if geo.near > reach {
                    continue;
                }
                let span = self.span(body, artifact, col);
                let margin = if self.kept.contains(&col) {
                    KEEP_MARGIN_RAD
                } else {
                    WANT_MARGIN_RAD
                };
                let clears = skyline.clears(geo.phi, geo.az, geo.rho, span.peak_m, margin);
                if clears {
                    cleared.insert(col);
                } else {
                    culled.insert(col);
                }
                // ★ INSIDE THE TRUE HORIZON THE SKYLINE DECIDES URGENCY, NEVER THE ASK
                // (2026-09-16, the walk-gap cure). A column the skyline hides and that stands
                // PAST the horizon is nothing: it is neither drawn now nor about to be. A hidden
                // column INSIDE the horizon is different - the eye's next step may uncover it,
                // and pass A already wants every column inside the horizon without asking the
                // skyline at all ("wanting a hidden one is safe"). The horizon's own hysteresis
                // holds such a column in this pass, so before this line it was not asked for at
                // all, and the frame it cleared the skyline it entered the ring URGENT from
                // NOTHING - missing on the very frame it was first wanted.
                //
                // MEASURED (the walk flight of 2026-09-16): the band went incomplete on 18 frames
                // of a minute's walk, 2 chunks at the worst; EVERY missing chunk read "was absent"
                // in the descent one frame before, the DRAWN eye already wanted it, the descent had
                // re-cut the ring 22 ms earlier with a drift of 0.000 m, and the builders stood at
                // 400 chunks a second. Each stood at rung 3 or 4, between 0.89 and 0.99 of the
                // eye's own horizon - the skyline's own edge. So the chunk is now ASKED while it is
                // hidden, as a REVEAL (the class for ground a crest may show), and the frame the
                // skyline clears it, it is already resident.
                if !clears & (geo.near > horizon) {
                    continue;
                }
                // The territory reads the TRUE horizon: a column the hysteresis sent here from
                // inside the horizon (between three quarters of it and the horizon) is ground the
                // picture draws now, and its chunks are urgent, never "revealed" (refutation
                // R4-1: the gap under-read it as a reveal) - while the skyline clears it.
                sweep.descend(col, &geo, &span, clears & (geo.near <= horizon), &mut next);
            }
            level = next;
        }
        let mut out = WantedSet {
            reach_m: reach,
            altitude_m: altitude,
            bounded: sweep.shadow.is_some(),
            ..WantedSet::default()
        };
        let mut emitted: Vec<(RequestOrder, ChunkKey, Territory)> = Vec::new();
        for rung_keys in &mut sweep.per_rung {
            for (key, territory) in rung_keys.drain(..) {
                let parent = Column {
                    face: key.face,
                    rung: key.rung + 1,
                    x: key.x.div_euclid(2),
                    y: key.y.div_euclid(2),
                };
                let order = RequestOrder {
                    class: territory.rank(),
                    depth: rungs - key.rung,
                    face: key.face,
                    parent_morton: morton(parent.x, parent.y),
                };
                emitted.push((order, key, territory));
            }
        }
        emitted.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        // THE CASTING SET (item 18): a chunk asks for its caster while its column's nearest point
        // lies within the caster bound read from the CASTER column's own peak over the lowest
        // ground within the reach (no ground within the reach: nothing to shade, the reach and the
        // diagonal alone). With no shadow reach, every chunk.
        let shadow = sweep.shadow;
        let low_m = sweep.low_within_reach_m;
        for (_, key, territory) in emitted {
            out.push(key);
            out.mark(key, territory);
            let casting = shadow.is_none_or(|s| {
                let caster_rung = s.caster_rung(key.rung, top);
                // The top rung has no coarser rung to cast for it: it never asks.
                if caster_rung == key.rung {
                    return false;
                }
                let step = caster_rung - key.rung;
                let caster = Column {
                    face: key.face,
                    rung: caster_rung,
                    x: key.x.div_euclid(1 << step),
                    y: key.y.div_euclid(1 << step),
                };
                let peak_m = self.span(body, artifact, caster).peak_m;
                let low_m = if low_m == f64::MAX { peak_m } else { low_m };
                // The caster's own nearest point, never nearer than the eye's height over the
                // relief (a column wider than the eye is high reads under the eye by the disc).
                let caster_near_m = column_geometry(&ladder, &frame, eye, caster, surface)
                    .near
                    .max(altitude - relief_m(body));
                caster_near_m <= s.caster_bound_m(peak_m, low_m)
            });
            if casting {
                out.casting.insert(key);
            }
        }
        // The columns this descent wanted or judged clear of the skyline, and the ones it culled,
        // for the next one's hysteresis. A far column that only descends (its own rung emits
        // nothing there) is kept through the skyline it cleared, so its whole subtree does not
        // hang on the narrower want margin (refutation R4-9).
        self.kept = out.keys.iter().map(|k| Column::of(*k)).collect();
        self.kept.extend(cleared);
        self.culled = culled;
        // The spans this descent did not visit are dropped.
        let generation = self.generation;
        self.spans.retain(|_, e| e.generation == generation);
        out
    }
}

/// THE COLUMN UNDER A DIRECTION at a rung: which chunk column of the face the direction falls in.
#[must_use]
pub fn column_under(body: &BodyDefinition, dir: [f64; 3], rung: u8) -> Column {
    let edge = CHUNK_EDGE as i32;
    let face = face_of(dir);
    let (t, s) = face_coords(face, dir);
    let rung = rung.min(vd_seed::ladder::RUNG_MAX);
    let n_l = body.ladder().cells_per_edge(rung);
    Column {
        face,
        rung,
        x: index_of(unbend(t), n_l) / edge,
        y: index_of(unbend(s), n_l) / edge,
    }
}

/// ★ THE ROW'S GRACE (2026-09-14, the boarding cure): may a realm's ladder be forgotten?
///
/// A ladder is forgotten when the realm's row leaves the drawn scene, and forgetting it releases
/// every chunk that realm holds — a whole band of ground, rebuilt from nothing. But a row absent
/// from ONE frame's scene is not a realm that left. At a boarding the gateway composes the first
/// picture in the hull's frame, and the planet's row can miss a beat while it does; a realm that
/// truly left never comes back, so waiting costs nothing but a beat of memory.
///
/// So the answer is YES only when the row is absent AND it has been absent longer than `grace_s`.
/// The caller's grace is the INTERPOLATION BUFFER — the wire's own contract for how old a
/// delivered picture may legitimately be — never a frame count a renderer chose.
///
/// `last_seen_s` is the display moment the realm last HAD a row, and `None` means it never did.
///
/// Example: the pilot walks aboard a berthed hull; the planet's row is away for two frames; its
/// 7 288 chunks stay drawn, and the ground under the hull never blinks.
#[must_use]
pub fn row_lapsed(has_row: bool, last_seen_s: Option<f64>, now_s: f64, grace_s: f64) -> bool {
    let away = match last_seen_s {
        Some(seen) => now_s - seen > grace_s,
        None => true,
    };
    !has_row & away
}

/// ★ WHERE THE CAMERA STANDS AND WHICH WAY IT LOOKS — the whole of one frame's placement, so a
/// frame that refuses its own pose can re-draw the LAST one exactly (position AND facing).
///
/// The eye rides twice: as metres (`eye`, what the camera transform needs) and on the lattice
/// (`lattice` with its `tier`, what every drawn position reduces against). Both are kept, because
/// re-deriving one from the other is the rounding this slice exists to remove.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeStand {
    /// The eye in the picture's own frame, metres.
    pub eye: vd_core::glam::DVec3,
    /// The same eye on the lattice, in `tier` units.
    pub lattice: vd_core::pose::LatticePos,
    /// The unit `lattice` is counted in.
    pub tier: vd_core::pose::Tier,
    /// Where the camera looks, and which way is up — the facing, kept whole.
    pub direction: vd_core::glam::DVec3,
    pub up: vd_core::glam::DVec3,
}

/// Is the own pose a reading in the realm the picture is composed in?
///
/// The camera places the eye by flattening the own pose into the picture's frame. The two are
/// different statements on different lanes — the pose rides the entity lane, the picture's origin
/// rides the window lane — and at a boarding they disagree for two to five frames: the pose already
/// reads a few metres from the HULL's centre while the picture is still drawn from the PLANET's.
/// Flattened anyway, that pose puts the eye at the planet's centre, six thousand kilometres under
/// the ground the pilot is standing on.
///
/// A picture that names no origin has nothing to disagree with (the pre-login state).
#[must_use]
pub fn pose_at_home(
    stated: vd_core::pose::RealmId,
    picture: Option<vd_core::pose::RealmId>,
) -> bool {
    match picture {
        Some(realm) => realm == stated,
        None => true,
    }
}

/// ★ THE CAMERA REFUSES A POSE STATED IN ANOTHER REALM (2026-09-15, the boarding's last seam;
/// SL1 clause 6: a stale reading is REFUSED, never used).
///
/// Given the realm the own pose names, the realm the picture is composed in, the stand this
/// frame's pose would place, and the stand the camera placed LAST frame, answers with the stand to
/// place and whether the pose was refused.
///
/// * They agree — the camera places the eye from the pose, as it always has.
/// * They disagree and there IS a last stand — the pose is refused and the last stand is placed
///   again, unchanged: the last delivered eye, RE-DRAWN. Never moved forward, because moving it
///   forward would be the client predicting where the pilot is, which it may not do.
/// * They disagree and there is no last stand — nothing has been drawn yet, so there is nothing to
///   hold; the delivered stand is placed and the frame is not counted as a refusal.
///
/// The first frame a pose in the picture's frame lands, the camera places from it and the hold
/// ends — there is no timer here, and nothing decays.
///
/// Example: the pilot walks aboard a berthed hull. For three frames their pose says `ShipLocal`
/// while the window still composes the planet's picture. The camera holds the eye on the ground
/// where it stood, looking the way the pilot looked; the fourth frame's picture is the hull's, the
/// pose agrees, and the eye steps aboard.
#[must_use]
pub fn camera_stand(
    stated: vd_core::pose::RealmId,
    picture: Option<vd_core::pose::RealmId>,
    delivered: EyeStand,
    held: Option<EyeStand>,
) -> (EyeStand, bool) {
    match held {
        Some(last) if !pose_at_home(stated, picture) => (last, true),
        _ => (delivered, false),
    }
}

/// ★ THE CAMERA'S EYE, FRAME AFTER FRAME (2026-09-15) — the whole of the camera's memory, so the
/// renderer only WIRES the rule and never decides it.
///
/// It holds the stand the camera last placed (what a refusal re-draws), the two readings the jump
/// and its bound are measured between, and the three numbers the dev stamp carries.
///
/// Example: the pilot walks aboard. Three frames refuse their pose and the eye stays on the
/// ground; `refusals` reads 3, `jump_max_m` stays at a walking step, and the frame the hull's
/// picture arrives the eye steps aboard by the walking those three frames delivered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EyeTrack {
    /// The stand the camera placed last frame, whole.
    held: Option<EyeStand>,
    /// The placed eye and the realm the picture was composed in, last frame.
    placed_at: Option<(vd_core::pose::RealmId, vd_core::glam::DVec3)>,
    /// The own pose's own point the last time it was stated in the picture's own frame. KEPT
    /// across a hold: a hold's catching-up step must be measured against the walking the pilot
    /// really did over the same frames, or the bound would refuse the very step it allows.
    delivered_at: Option<(vd_core::pose::RealmId, vd_core::glam::DVec3)>,
    /// How many frames the camera refused a pose stated in another realm.
    pub refusals: u64,
    /// The largest single-frame jump of the PLACED eye, metres.
    pub jump_max_m: f64,
    /// The largest single-frame displacement of the DELIVERED own pose, metres — the bound the
    /// jump is judged against.
    pub step_max_m: f64,
}

impl EyeTrack {
    /// Place this frame's eye: the delivered stand when the own pose names the picture's own
    /// realm, the held one when it names another. Reads the two distances on the way and answers
    /// with the stand the camera must draw from.
    ///
    /// The BOUND is the delivered stand's own eye, not the pilot's point: a pilot who turns on the
    /// spot moves the eye by the eye's own height, and a bound that read the point alone would
    /// call that turn a jump.
    pub fn place(
        &mut self,
        stated: vd_core::pose::RealmId,
        picture: Option<vd_core::pose::RealmId>,
        delivered: EyeStand,
    ) -> EyeStand {
        let (stand, refused) = camera_stand(stated, picture, delivered, self.held);
        self.refusals += u64::from(refused);
        let placed_now = picture.map(|realm| (realm, stand.eye));
        if let Some(jump) = frame_step_m(self.placed_at, placed_now) {
            self.jump_max_m = self.jump_max_m.max(jump);
        }
        let delivered_now = if pose_at_home(stated, picture) {
            picture.map(|realm| (realm, delivered.eye))
        } else {
            None
        };
        if let Some(step) = frame_step_m(self.delivered_at, delivered_now) {
            self.step_max_m = self.step_max_m.max(step);
        }
        if delivered_now.is_some() {
            self.delivered_at = delivered_now;
        }
        self.placed_at = placed_now;
        self.held = Some(stand);
        stand
    }
}

/// How far a point moved between two frames, when both readings are stated in the SAME realm's
/// frame — `None` otherwise, because a distance between two frames' coordinates is not a distance.
///
/// The eye's JUMP and the pilot's own DELIVERED step are both this measurement, so the gate that
/// compares them compares two readings of one shape. A frame change is not a jump: at a boarding
/// the picture's origin moves from the planet's centre to the hull's, and every coordinate in it
/// moves by the planet's radius without anything moving at all.
///
/// Example: the eye stands on the planet at 6 371 004 m from its centre and the next frame reads
/// 6 371 004.02 m — a step of two centimetres, one frame of a 1.4 m/s walk. The frame after that
/// the picture is the hull's, and the pair is refused rather than read as 6 371 km.
#[must_use]
pub fn frame_step_m(
    before: Option<(vd_core::pose::RealmId, vd_core::glam::DVec3)>,
    after: Option<(vd_core::pose::RealmId, vd_core::glam::DVec3)>,
) -> Option<f64> {
    let (was, at) = (before?, after?);
    if was.0 == at.0 {
        Some((at.1 - was.1).length())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_core::pose::RealmId;
    use vd_terrain::home::home_planet;

    /// ★ A SPAN IS READ ON THE FIELD THE CHUNK IS BUILT ON (2026-09-20): with a head alone the
    /// span is the recipe's and provisional; with the pyramid it is the finest whole level's,
    /// still provisional at a rung that reads the tiles and final at one that reads a level; with
    /// the tiles it is the tiles' own, final. Without an artifact, or on a body with no macro
    /// lattice, it is the recipe's, final. The view reads a provisional span again at the next
    /// descent and keeps a final one.
    #[test]
    fn a_span_is_read_on_the_artifacts_field_and_provisional_until_the_tiles_land() {
        use crate::artifact_book::ArtifactReceiver;
        use vd_terrain::digest::{surface_column, surface_column_field};
        use vd_terrain::home::{
            HOME_MOON_SEED, HOME_SYSTEM_AGE_YR, home_moon, home_moon_solve_words,
        };
        use vd_terrain::solve::{Schedule, solve_full};
        use vd_wire::channels::BulkMsg;
        let moon = home_moon();
        let lattice = moon.macro_lattice().expect("a lattice");
        let words = home_moon_solve_words();
        let (state, facies, _) =
            solve_full(&moon, &words, Schedule::standard(HOME_SYSTEM_AGE_YR)).expect("a solve");
        let climate =
            vd_terrain::climate::climate(&moon, &lattice, &words, &state.z, Some(state.sea_z));
        let artifact =
            vd_terrain::artifact::Artifact::of(&state, &facies, &climate, words.water_km3 > 0);
        let moon = moon.with_sea_m(artifact.sea());
        let realm = RealmId::Planet(HOME_MOON_SEED);
        let mut rx = ArtifactReceiver::default();
        rx.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels: artifact.pyramid.len() as u32,
            sea_m: artifact.sea_m,
            coast_parts: 0,
        });
        let col = Column {
            face: Face::PosZ,
            rung: 3,
            x: 40,
            y: 41,
        };
        let slices = |s: ColumnSpan| (s.lo, s.hi, s.peak_m);
        // A head alone: the recipe's span, provisional.
        let head_only = rx.book().get(realm).cloned().expect("a cache");
        let (span, provisional) = column_span(&moon, Some(&head_only), col);
        assert!(provisional);
        assert_eq!(
            slices(span),
            slices(surface_column(&moon, col.face, col.rung, col.x, col.y))
        );
        // The pyramid: the finest whole level's span, provisional where the rung reads the tiles.
        for (k, level) in artifact.pyramid.iter().enumerate().rev() {
            rx.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: k as u32 + 1,
                part: 0,
                parts: 1,
                z_m: level.clone(),
                water_m: artifact.pyramid_water[k].clone(),
            });
        }
        let pyramid = rx.book().get(realm).cloned().expect("a cache");
        let (span, provisional) = column_span(&moon, Some(&pyramid), col);
        assert!(provisional);
        let level1 = pyramid.level(1).expect("level 1");
        let expect = surface_column_field(&moon, &**level1, col.face, col.rung, col.x, col.y)
            .expect("a span");
        assert_eq!(slices(span), slices(expect));
        // A rung that reads a level: final.
        let coarse = Column {
            face: Face::PosZ,
            rung: 11,
            x: 1,
            y: 1,
        };
        assert!(!column_span(&moon, Some(&pyramid), coarse).1);
        // ★ THE LEVEL THE RUNG READS IS NOT HERE: the head and the coarser levels landed, the
        // level this rung reads did not. The span falls back to the finest whole level and stays
        // PROVISIONAL, so the view reads the column again at the next descent.
        let levels = artifact.pyramid.len() as u32;
        let deep_rung = (0..=14u8)
            .find(|r| vd_terrain::artifact::PyramidField::level_for(&lattice, levels, *r) == 1)
            .expect("a rung that reads level 1");
        let mut late = ArtifactReceiver::default();
        late.accept(BulkMsg::ArtifactHead {
            realm,
            world_tag: 9,
            version: artifact.version,
            edge: artifact.edge,
            digest: artifact.digest(),
            tiles_per_edge: artifact.tiles_per_edge(),
            levels,
            sea_m: artifact.sea_m,
            coast_parts: 0,
        });
        for (k, level) in artifact
            .pyramid
            .iter()
            .enumerate()
            .rev()
            .take(levels as usize - 1)
        {
            late.accept(BulkMsg::ArtifactPyramid {
                realm,
                level: k as u32 + 1,
                part: 0,
                parts: 1,
                z_m: level.clone(),
                water_m: artifact.pyramid_water[k].clone(),
            });
        }
        let late = late.book().get(realm).cloned().expect("a cache");
        assert_eq!(late.level(1), None, "the rung's own level waits");
        let deep = Column {
            face: Face::PosZ,
            rung: deep_rung,
            x: 1,
            y: 1,
        };
        let (span, provisional) = column_span(&moon, Some(&late), deep);
        assert!(provisional);
        let coarser = late.level(2).expect("level 2");
        let expect = surface_column_field(&moon, &**coarser, deep.face, deep.rung, deep.x, deep.y)
            .expect("a span");
        assert_eq!(slices(span), slices(expect));
        // The tiles, every face's (a descent visits the columns across a seam too): the tiles'
        // own span, final.
        for face in Face::ALL {
            for tx in 0..artifact.tiles_per_edge() {
                for ty in 0..artifact.tiles_per_edge() {
                    rx.accept(BulkMsg::ArtifactTile {
                        realm,
                        face: face.index(),
                        tx,
                        ty,
                        rows: artifact.tile(face, tx, ty).to_bytes(),
                    });
                }
            }
        }
        let whole = rx.book().get(realm).cloned().expect("a cache");
        let (span, provisional) = column_span(&moon, Some(&whole), col);
        assert!(!provisional);
        let expect = surface_column_field(&moon, &artifact, col.face, col.rung, col.x, col.y)
            .expect("a span");
        assert_eq!(slices(span), slices(expect));
        // No artifact yet on a body that will state one: the recipe's, PROVISIONAL; a body with
        // no macro lattice: the recipe's, final.
        let (span, provisional) = column_span(&moon, None, col);
        assert!(provisional);
        assert_eq!(
            slices(span),
            slices(surface_column(&moon, col.face, col.rung, col.x, col.y))
        );
        let rock = moon.without_macro_lattice();
        assert!(!column_span(&rock, Some(&whole), col).1);
        assert!(!column_span(&rock, None, col).1);
        // The view: over the column, the pyramid alone leaves provisional spans, the tiles none;
        // a final span is kept across descents.
        let n_l = moon.ladder().cells_per_edge(col.rung);
        let edge = CHUNK_EDGE as i32;
        let d = DVec3::from_array(direction(
            col.face,
            face_param(col.x * edge + edge / 2, n_l),
            face_param(col.y * edge + edge / 2, n_l),
        ))
        .normalize();
        let eye = (d * (moon.radius_m() + 3_000.0)).to_array();
        let mut view = LadderView::default();
        let on_pyramid = view.wanted(&moon, eye, Some(&pyramid));
        assert!(!on_pyramid.keys.is_empty());
        assert!(view.provisional_spans() > 0);
        let on_tiles = view.wanted(&moon, eye, Some(&whole));
        assert!(!on_tiles.keys.is_empty());
        assert_eq!(view.provisional_spans(), 0);
        let held = view.spans_held();
        view.wanted(&moon, eye, Some(&whole));
        assert_eq!(view.spans_held(), held);
    }

    /// ★ THE FACTS OF THE 300 km TEST BODY the two ruling-T7 tests below read (slice 8b stage 3).
    /// COMPUTED at 3 000 kg/m³ by `g = (4/3)πGρR`: `g = 0.2516 m/s²`, so 251 mm/s². The body is
    /// SHAPE-limited — its strength arm is 636 km against a shape arm of 23 100 m — which is what a
    /// 300 km body is.
    const SMALL_BODY_FACTS: vd_terrain::BodyFacts = vd_terrain::BodyFacts::new(251, 3_000);

    /// ★ THE ROW'S GRACE: a row that is THERE never lapses; an absent row lapses only once it has
    /// been away longer than the grace; a realm that never had a row lapses at once.
    #[test]
    fn a_rows_grace_holds_a_ladder_for_one_interpolation_buffer() {
        // Exact binary values, so the comparison's own edge is the thing measured and not a
        // decimal's rounding.
        let grace = 0.25;
        // The planet's row is in this frame's scene: nothing lapses, however old the last sight.
        assert!(!row_lapsed(true, Some(1.0), 99.0, grace));
        assert!(!row_lapsed(true, None, 99.0, grace));
        // The row is away, but for less than the grace: the ladder stays.
        assert!(!row_lapsed(false, Some(10.0), 10.125, grace));
        // Exactly the grace is still within it (the comparison is strict).
        assert!(!row_lapsed(false, Some(10.0), 10.25, grace));
        // Away for longer than the grace: the realm really left.
        assert!(row_lapsed(false, Some(10.0), 10.5, grace));
        // A realm that never had a row has nothing to wait for.
        assert!(row_lapsed(false, None, 10.0, grace));
    }

    /// A stand for the tests: the eye, its lattice twin and the facing, all named by one number
    /// so a held stand and a delivered one can never be confused.
    fn stand(x: f64) -> EyeStand {
        let eye = DVec3::new(x, 0.0, 0.0);
        EyeStand {
            eye,
            lattice: vd_core::pose::LatticePos::from_metres(eye, vd_core::pose::Tier::Fine),
            tier: vd_core::pose::Tier::Fine,
            direction: DVec3::new(0.0, 0.0, -x),
            up: DVec3::new(0.0, x, 0.0),
        }
    }

    /// ★ THE CAMERA REFUSES A POSE STATED IN ANOTHER REALM. The pilot walks aboard the hull: for
    /// a beat their pose says the hull and the picture still says the planet. The camera holds the
    /// eye and the facing it last placed; the first pose in the picture's own frame ends the hold.
    #[test]
    fn the_camera_holds_its_eye_while_the_pose_and_the_picture_name_two_realms() {
        let planet = RealmId::Planet(7);
        let hull = RealmId::Ship(vd_core::EntityId::pack(
            vd_core::entity_kind::EntityKind::Ship,
            1,
            1,
            0,
        ));
        let (ground, aboard) = (stand(1.0), stand(2.0));
        // The pose names the realm the picture is composed in: the camera places from it.
        assert_eq!(
            camera_stand(planet, Some(planet), ground, None),
            (ground, false),
            "an agreeing pose places the eye"
        );
        assert_eq!(
            camera_stand(planet, Some(planet), aboard, Some(ground)),
            (aboard, false),
            "an agreeing pose places the eye even with a stand to hold"
        );
        // The pose names the HULL while the picture is still the PLANET's: the eye and the facing
        // of the last frame stand again, unchanged.
        assert_eq!(
            camera_stand(hull, Some(planet), aboard, Some(ground)),
            (ground, true),
            "a foreign pose is refused and the held stand is placed again"
        );
        // Nothing drawn yet: there is no stand to hold, so the delivered one is placed and the
        // frame is no refusal.
        assert_eq!(
            camera_stand(hull, Some(planet), aboard, None),
            (aboard, false),
            "with nothing held there is nothing to hold"
        );
        // The first pose in the picture's own frame ends the hold at once.
        assert_eq!(
            camera_stand(hull, Some(hull), aboard, Some(ground)),
            (aboard, false),
            "the first agreeing pose after a hold places the eye"
        );
        // A picture that names no origin has nothing to disagree with.
        assert_eq!(
            camera_stand(hull, None, aboard, Some(ground)),
            (aboard, false),
            "no origin, no disagreement"
        );
        assert_eq!(
            (
                pose_at_home(hull, None),
                pose_at_home(hull, Some(hull)),
                pose_at_home(hull, Some(planet)),
            ),
            (true, true, false),
            "a pose is at home in the picture's own realm, and in no other"
        );
    }

    /// ★ THE HOLD, FRAME AFTER FRAME, AND THE JUMP IT MAY NOT EXCEED (2026-09-15). The pilot
    /// walks on the planet, their pose names the hull for two frames while the picture is still
    /// the planet's, and then the picture is the hull's too. The eye is held through the two
    /// frames and never moves farther in one frame than the delivered pose moved over the same
    /// frames.
    #[test]
    fn the_track_holds_the_eye_through_a_boarding_and_never_jumps_farther_than_the_walk() {
        let planet = RealmId::Planet(7);
        let hull = RealmId::Ship(vd_core::EntityId::pack(
            vd_core::entity_kind::EntityKind::Ship,
            1,
            1,
            0,
        ));
        let mut track = EyeTrack::default();
        // Frame 1: the first delivered pose. Nothing to measure against yet.
        let first = stand(0.0);
        assert_eq!(track.place(planet, Some(planet), first), first);
        assert_eq!(
            (track.refusals, track.jump_max_m, track.step_max_m),
            (0, 0.0, 0.0),
            "the first frame has no earlier reading to measure against"
        );
        // Frame 2: a walking step on the planet. The numbers are exact in binary, so the
        // comparison measures the rule and not a decimal's rounding.
        let second = stand(0.25);
        assert_eq!(track.place(planet, Some(planet), second), second);
        assert_eq!(
            (track.refusals, track.jump_max_m, track.step_max_m),
            (0, 0.25, 0.25),
            "the eye moved exactly the walk"
        );
        // Frames 3 and 4: the pose names the HULL while the picture is still the PLANET's. The
        // stand that pose would place stands a thousand metres away — the boarding's own defect in
        // miniature. The eye is held; it moves nowhere, and the delivered reading is not read.
        let aboard = stand(1000.0);
        assert_eq!(
            track.place(hull, Some(planet), aboard),
            second,
            "the held stand is placed again"
        );
        assert_eq!(
            track.place(hull, Some(planet), aboard),
            second,
            "and again, for as long as they disagree"
        );
        assert_eq!(
            (track.refusals, track.jump_max_m, track.step_max_m),
            (2, 0.25, 0.25),
            "two refusals, and the eye stood still through both"
        );
        // Frame 5: the pose names the planet again, two frames of walking further on. The eye
        // catches up by exactly what was delivered — the jump never passes the bound.
        let third = stand(0.75);
        assert_eq!(track.place(planet, Some(planet), third), third);
        assert_eq!(
            (track.refusals, track.jump_max_m, track.step_max_m),
            (2, 0.5, 0.5),
            "the catching-up jump equals the walking the hold covered"
        );
        // Frame 6: the picture is the hull's now and the pose agrees. The origin changed, so
        // neither distance is read: a coordinate in the hull's frame is not a distance from one
        // in the planet's.
        let inside = stand(3.0);
        assert_eq!(track.place(hull, Some(hull), inside), inside);
        assert_eq!(
            (track.refusals, track.jump_max_m, track.step_max_m),
            (2, 0.5, 0.5),
            "an origin swap is not a jump"
        );
        // Frame 7: a picture that names no origin (the pre-login state) measures nothing.
        let nowhere = stand(9.0);
        assert_eq!(track.place(hull, None, nowhere), nowhere);
        assert_eq!(
            (track.refusals, track.jump_max_m, track.step_max_m),
            (2, 0.5, 0.5),
            "no origin, nothing to measure"
        );
    }

    /// ★ A STEP IS READ ONLY BETWEEN TWO READINGS OF ONE FRAME. Two metres of walking on the
    /// planet is two metres; the planet's centre against the hull's centre is not a distance at
    /// all, and the pair is refused.
    #[test]
    fn a_step_is_measured_only_between_two_readings_of_one_realms_frame() {
        let planet = RealmId::Planet(7);
        let hull = RealmId::Ship(vd_core::EntityId::pack(
            vd_core::entity_kind::EntityKind::Ship,
            1,
            1,
            0,
        ));
        let a = DVec3::new(0.0, 0.0, 0.0);
        let b = DVec3::new(3.0, 4.0, 0.0);
        assert_eq!(
            frame_step_m(Some((planet, a)), Some((planet, b))),
            Some(5.0),
            "two readings of the planet's frame are five metres apart"
        );
        assert_eq!(
            frame_step_m(Some((planet, a)), Some((hull, b))),
            None,
            "the picture's origin changed: there is no distance to read"
        );
        assert_eq!(
            frame_step_m(None, Some((planet, b))),
            None,
            "no first reading"
        );
        assert_eq!(
            frame_step_m(Some((planet, a)), None),
            None,
            "no second reading"
        );
    }

    /// THE CASTING SET (item 18): without a shadow reach every wanted chunk may cast; with one,
    /// the chunks near the eye cast and the far ones do not, the bound growing with the sun's
    /// tangent and with the rung's relief; the same reach within a tenth in the tangent is the
    /// same for the recompute's purposes.
    #[test]
    fn a_shadow_reach_bounds_which_chunks_ask_for_a_caster() {
        let body = home_planet();
        let r = body.ladder().radius_m();
        let eye = [r + EYE_HEIGHT_M, 0.0, 0.0];
        let free = LadderView::default().wanted(&body, eye, None);
        assert!(free.keys.iter().all(|k| free.casts(*k)));
        assert_eq!(free.casting_count(), free.keys.len());
        let mut view = LadderView::default();
        let reach = ShadowReach {
            reach_m: switch_m(2),
            tan_i: 0.0,
            coarse_step: 2,
        };
        view.shadow = Some(reach);
        let bounded = view.wanted(&body, eye, None);
        let casting = bounded.casting_count();
        assert!(casting > 0);
        assert!(casting < bounded.keys.len());
        // The finest ring stands at the eye: every one of its chunks casts. The chunks that do
        // not cast stand farther than every one that does (their columns' nearest points).
        let finest: Vec<ChunkKey> = bounded
            .keys
            .iter()
            .filter(|k| k.rung == bounded.rung_min)
            .copied()
            .collect();
        assert!(!finest.is_empty());
        assert!(finest.iter().all(|k| bounded.casts(*k)));
        let surface = body.ladder().radius_m();
        let frame = EyeFrame::new(DVec3::from_array(eye));
        let top = body.ladder().rungs - 1;
        // The CASTER column's nearest point, for a drawn key.
        let caster_nearest = |k: &ChunkKey| {
            let caster_rung = reach.caster_rung(k.rung, top);
            let step = caster_rung - k.rung;
            column_geometry(
                body.ladder(),
                &frame,
                DVec3::from_array(eye),
                Column {
                    face: k.face,
                    rung: caster_rung,
                    x: k.x.div_euclid(1 << step),
                    y: k.y.div_euclid(1 << step),
                },
                surface,
            )
            .near
        };
        // Rung by rung, below the top: the casters that are not asked for stand farther than
        // every one that is (the sun overhead, so the bound is the reach alone).
        for rung in bounded.rung_min..top {
            let casting_max = bounded
                .keys
                .iter()
                .filter(|k| (k.rung == rung) & bounded.casts(**k))
                .map(caster_nearest)
                .fold(f64::MIN, f64::max);
            let silent_min = bounded
                .keys
                .iter()
                .filter(|k| (k.rung == rung) & !bounded.casts(**k))
                .map(caster_nearest)
                .fold(f64::MAX, f64::min);
            assert!(
                silent_min >= casting_max,
                "rung {rung}: {silent_min} vs {casting_max}"
            );
        }
        // A lower sun throws longer shadows: more chunks cast.
        view.shadow = Some(ShadowReach {
            tan_i: 8.0,
            ..reach
        });
        let low_sun = view.wanted(&body, eye, None);
        assert!(low_sun.casting_count() >= casting);
        // The bound: the reach, the peak's shadow over the low ground, the caster's diagonal; a
        // peak under the ground shades nothing (the reach and the diagonal alone), and the
        // diagonal grows with the rung.
        let sun = ShadowReach {
            reach_m: 100.0,
            tan_i: 2.0,
            coarse_step: 2,
        };
        assert!((sun.caster_bound_m(1_030.0, 1_000.0) - 160.0).abs() < 1e-9);
        assert!((sun.caster_bound_m(990.0, 1_000.0) - 100.0).abs() < 1e-9);
        // The caster's rung never passes the body's top rung.
        assert_eq!(sun.caster_rung(0, 12), 2);
        assert_eq!(sun.caster_rung(11, 12), 12);
        assert_eq!(sun.caster_rung(12, 12), 12);
        // The ladder's top rung has no caster: none of its chunks asks (a low sun, so the bound
        // alone would let them).
        assert!(
            low_sun
                .keys
                .iter()
                .filter(|k| k.rung == top)
                .all(|k| !low_sun.casts(*k))
        );
        // ★ THE SAME CLAIM READ WHERE THE TOP RUNG IS ACTUALLY DRAWN (the extended ladder,
        // 2026-09-15). From an eye 1.8 m up the descent never reaches rung 18, so the filter above
        // is EMPTY and proves nothing — the "no coarser rung to cast for me" arm was never run.
        // From 34 body radii the whole globe IS the top rung, one chunk a face, and every one of
        // them is silent while the same low sun stands.
        let far_up = view.wanted(&body, [r * 34.0, 0.0, 0.0], None);
        assert_eq!(far_up.keys.len(), 6);
        assert_eq!((far_up.rung_min, far_up.rung_max), (top, top));
        assert_eq!(far_up.casting_count(), 0);
        // The hysteresis is symmetric and has a floor: a zenith sun's tangents agree.
        let sym = ShadowReach {
            reach_m: 100.0,
            tan_i: 1.0,
            coarse_step: 2,
        };
        assert!(ShadowReach { tan_i: 0.0, ..sym }.same_as(ShadowReach {
            tan_i: 0.005,
            ..sym
        }));
        assert_eq!(
            sym.same_as(ShadowReach { tan_i: 1.2, ..sym }),
            ShadowReach { tan_i: 1.2, ..sym }.same_as(sym)
        );
        // NO GROUND WITHIN THE REACH (an eye 60 km up, the reach a few kilometres): nothing to
        // shade, so no chunk asks for a caster.
        let aloft = view.wanted(&body, [r + 60_000.0, 0.0, 0.0], None);
        assert!(!aloft.keys.is_empty());
        assert_eq!(aloft.casting_count(), 0);
        // The same reach within a tenth of the tangent; a different one past it.
        let base = ShadowReach {
            reach_m: 100.0,
            tan_i: 1.0,
            coarse_step: 2,
        };
        assert!(base.same_as(ShadowReach {
            tan_i: 1.05,
            ..base
        }));
        assert!(!base.same_as(ShadowReach { tan_i: 1.2, ..base }));
        assert!(!base.same_as(ShadowReach {
            reach_m: 200.0,
            ..base
        }));
        assert!(!base.same_as(ShadowReach {
            coarse_step: 1,
            ..base
        }));
    }

    /// ★ THE HANDOVER'S STEP STANDS UNDER THE LADDER'S OWN TOLERANCE ON THE HOME PLANET — which is
    /// what makes ruling T7 rules 2 and 3 INERT today (they are machinery, not a change to the
    /// picture), and it is an assertion that could fail: before rule 1 the rung 6 → 7 handover
    /// dropped the 3 125 m crest, 198.84 m against a 128 m cell — 1.55 of a cell.
    ///
    /// The tolerance is ONE CELL OF THE RUNG THAT TAKES OVER, the line the judge
    /// (`cargo run --release -p vd-bins --example rung_disagreement`) measures on.
    #[test]
    fn the_home_planets_table_leaves_both_the_band_and_the_floor_inert() {
        let body = home_planet();
        let rungs = body.ladder().rungs;
        let bound = AskBound::unbounded().for_body(&body, rungs);
        let mut worst = 0.0f64;
        let mut rung = 0u8;
        while rung + 1 < rungs {
            let step = handover_step_m(&body, rung);
            assert_eq!(bound.step_m(rung), step, "rung {rung}");
            worst = worst.max(step_px(step, rung));
            // Rule 2 is inert: the band is the ladder's own, bit for bit.
            assert_eq!(band_widen(step, rung), 1.0, "rung {rung}");
            assert_eq!(
                bound.fade_bands(rung, rungs),
                fade_bands(rung, rungs),
                "rung {rung}"
            );
            // Rule 3 is inert: the floor stands inside the tier rule's own switch distance.
            assert!(switch_floor_m(step) < switch_m(rung), "rung {rung}");
            assert_eq!(bound.switch_m(rung), switch_m(rung), "rung {rung}");
            rung += 1;
        }
        // The top rung hands over to nobody, so it has no step at all.
        assert_eq!(bound.step_m(rungs - 1), 0.0);
        assert_eq!(bound.step_m(rungs), 0.0);
        // The line is not slack: the worst pair stands at 0.999 of a cell — rung 3 → 4, the rung the
        // CAP-ROCK BENCH fades over (slice 8a stage 4). The bench's strength is SOLVED against this
        // very line, so the pair sits on it by construction, with the ladder's own margin of two gap
        // steps and no more. Before the bench the worst pair was rung 9 → 10 at 0.88 of a cell.
        assert!((0.5..1.0).contains(&worst), "the worst pair reads {worst}");
    }

    /// ★ RULES 2 AND 3 WAKE BY THEMSELVES ON A BODY WHOSE TABLE NEEDS THEM — MEASURED, not
    /// invented: the recipe draws the body of seed 382 at a look radius of 300 km with a rung
    /// 3 → 4 handover of 1.52 cells, over the ladder's own tolerance. No number in this test is
    /// typed: the step comes from that body's own octave table, exactly as it does in the picture.
    ///
    /// ★ RED BEFORE ruling T7 rules 2 and 3: the band stood at the ladder's own fifth whatever the
    /// step, and the switch distance was the cell's alone — this body's rung 3 surface handed over
    /// where its own step stood over a pixel and a half, and the crossfade had a pixel and a half
    /// to hide in a band sized for one.
    #[test]
    fn a_body_over_the_tolerance_widens_its_band_and_pushes_its_switch_out() {
        let body = vd_terrain::BodyDefinition::from_seed(382, 300_000.0, SMALL_BODY_FACTS)
            .expect("a real body");
        let rungs = body.ladder().rungs;
        let bound = AskBound::unbounded().for_body(&body, rungs);
        let step = handover_step_m(&body, 3);
        let ratio = step_px(step, 3);
        assert!(ratio > 1.0, "the measurement that made this test: {ratio}");
        // Rule 2: the band is wider by exactly the ratio, around the same switch distance.
        assert_eq!(band_widen(step, 3), ratio);
        let (_, out) = bound.fade_bands(3, rungs);
        let s = bound.switch_m(3);
        assert!((out[0] - s * (1.0 - (1.0 - HYSTERESIS_IN) * ratio)).abs() < 1e-9);
        assert!((out[1] - s * (1.0 + (HYSTERESIS_OUT - 1.0) * ratio)).abs() < 1e-9);
        // The band still widens about the switch distance itself: the eye crosses where it did.
        assert!((out[0] + out[1] - 2.0 * s).abs() < 1e-6);
        // Rule 3: the switch distance is the floor, which stands past the tier rule's own.
        assert_eq!(s, switch_floor_m(step));
        assert!(s > switch_m(3));
        assert!(
            (s / switch_m(3) - ratio).abs() < 1e-9,
            "by exactly the ratio"
        );
        // And the rungs the body's own table leaves alone read the tier rule's, bit for bit.
        assert_eq!(bound.switch_m(0), switch_m(0));
        assert_eq!(bound.fade_bands(0, rungs), fade_bands(0, rungs));
        // A bound that never read a body draws the tier ladder, whatever the body's table says.
        assert_eq!(AskBound::unbounded().switch_m(3), switch_m(3));
    }

    /// ★ THE BODY'S ROW SURVIVES THE SLEW AND THE SLACK (ruling T7): the horizon slides and the
    /// descent asks wider, and both keep the floor the body's own table states — otherwise the
    /// picture would draw a band the rule had already refused.
    #[test]
    fn the_bodys_row_rides_the_slew_and_the_slack() {
        let body = vd_terrain::BodyDefinition::from_seed(382, 300_000.0, SMALL_BODY_FACTS)
            .expect("a real body");
        let rungs = body.ladder().rungs;
        let want = AskBound::unbounded().for_body(&body, rungs);
        let floor = switch_floor_m(handover_step_m(&body, 3));
        // A held bound that has read no body slews toward the one that has, and takes its row.
        let held = AskBound::unbounded().slewed_toward(&want, rungs, 1.0);
        assert_eq!(held.step_m(3), want.step_m(3));
        assert_eq!(held.switch_m(3), floor);
        assert_eq!(
            held.clone().with_slack(ASK_BOUND_SLACK).step_m(3),
            want.step_m(3)
        );
        // And a bound the builders really bind still never hands over nearer than the floor.
        let bound = AskBound::from_switches(vec![1.0; usize::from(rungs)]).for_body(&body, rungs);
        assert_eq!(bound.switch_m(3), floor);
        // Even at rung 0, where the body's own step is under a pixel, the floor outlives an ask of
        // one metre: a step the eye can see is never handed over at arm's length.
        assert_eq!(bound.switch_m(0), switch_floor_m(handover_step_m(&body, 0)));
        assert!(bound.switch_m(0) > 1.0);
        // A slew that lands on the tier rule's own radii keeps the row too.
        let free = AskBound::unbounded()
            .for_body(&body, rungs)
            .slewed_toward(&want, rungs, 1.0e9);
        assert_eq!(free.step_m(3), want.step_m(3));
        assert_eq!(free.switch_m(3), floor);
    }

    #[test]
    fn the_tier_rule_is_the_finest_rung_with_a_cell_of_one_pixel() {
        let px = pixel_rad();
        assert!((px - 2.0 * (std::f64::consts::FRAC_PI_4 * 0.5).tan() / 720.0).abs() < 1e-15);
        assert_eq!(rung_for_distance(0.0, 13), 0);
        assert_eq!(rung_for_distance(-5.0, 13), 0);
        assert_eq!(rung_for_distance(3.4, 13), 0);
        // Just inside and just past the first switch.
        assert_eq!(rung_for_distance(switch_m(0) * 0.999, 13), 0);
        assert_eq!(rung_for_distance(switch_m(0) * 1.001, 13), 1);
        assert_eq!(rung_for_distance(switch_m(2), 13), 2);
        assert_eq!(rung_for_distance(60_000.0, 13), 7);
        // Clamped to the top rung; a ladder of no rungs reads rung 0.
        assert_eq!(rung_for_distance(1e9, 13), 12);
        assert_eq!(rung_for_distance(1e9, 0), 0);
        assert!((switch_m(0) - 1.0 / px).abs() < 1e-9);
        assert!((switch_m(3) - 8.0 / px).abs() < 1e-9);
        assert!((HYSTERESIS_IN - 0.9).abs() < 1e-12);
        assert!((HYSTERESIS_OUT - 1.1).abs() < 1e-12);
        // The bands: rung 0 is always in, the top rung never fades out, a middle rung fades in over
        // the band below and out over its own; the weights are complementary across a band.
        let (in0, out0) = fade_bands(0, 13);
        assert_eq!(in0, FADE_ALWAYS_IN);
        assert!((out0[0] - 0.9 * switch_m(0)).abs() < 1e-9);
        assert!((out0[1] - 1.1 * switch_m(0)).abs() < 1e-9);
        let (in12, out12) = fade_bands(12, 13);
        assert!((in12[0] - 0.9 * switch_m(11)).abs() < 1e-6);
        assert_eq!(out12, FADE_ALWAYS_OUT);
        let s3 = switch_m(3);
        // A rung's fade-in band is the rung below's fade-out band: one line, `in_hi = out_hi`.
        let (in4, _) = fade_bands(4, 13);
        let (_, out3) = fade_bands(3, 13);
        assert_eq!(in4, out3);
        // Inside the band, outside it, and the always-passed ends.
        assert!(in_fade_band(1.0 * s3, 13));
        assert!(in_fade_band(0.95 * s3, 13));
        assert!(!in_fade_band(0.7 * s3, 13));
        assert!(!in_fade_band(3.4, 13));
    }

    #[test]
    fn the_sink_ramp_ends_past_the_edge_by_one_finer_cell_of_residual() {
        let body = home_planet();
        let rungs = body.ladder().rungs;
        // Rung 0 sinks nowhere: its fade-in edge.
        assert_eq!(sink_end_m(&body, 0, rungs), fade_bands(0, rungs).0[1]);
        // Rung 1: the ramp from in_lo to the end passes in_hi with one finer cell (1 m) left of
        // the sink.
        let (fade_in, _) = fade_bands(1, rungs);
        let end = sink_end_m(&body, 1, rungs);
        assert!(end > fade_in[1]);
        let sink = crate::chunks::sink_m(&body, 1);
        let at_edge = sink * (end - fade_in[1]) / (end - fade_in[0]);
        assert!((at_edge - 1.0).abs() < 1e-9, "{at_edge}");
    }

    #[test]
    fn the_horizon_and_the_reach() {
        let r = 6_371_000.0;
        assert!((horizon_m(r, 3.4) - 6_582.0).abs() < 1.0);
        assert_eq!(horizon_m(r, -1.0), 0.0);
        // The reach adds the horizon of the tallest ground.
        assert!((reach_m(r, 3.4, 0.0) - horizon_m(r, 3.4)).abs() < 1e-9);
        assert!(reach_m(r, 3.4, 100.0) > horizon_m(r, 3.4) + 35_000.0);
        // The relief is the amplitude sum: positive, under a body's radius by far.
        let body = home_planet();
        let relief = relief_m(&body);
        assert!(relief > 0.0);
        assert!(relief < 50_000.0, "{relief}");
    }

    #[test]
    fn a_column_knows_its_chunk_and_its_coarser_index() {
        let key = ChunkKey {
            face: Face::PosX,
            rung: 2,
            x: 13,
            y: 7,
            z: 1,
        };
        let col = Column::of(key);
        assert_eq!(
            col,
            Column {
                face: Face::PosX,
                rung: 2,
                x: 13,
                y: 7
            }
        );
        assert_eq!(col.coarser(0), (13, 7));
        assert_eq!(col.coarser(2), (3, 1));
    }

    fn key(face: Face, rung: u8, x: i32, y: i32, z: i32) -> ChunkKey {
        ChunkKey {
            face,
            rung,
            x,
            y,
            z,
        }
    }

    #[test]
    fn the_release_hold_finds_a_missing_chunk_over_a_footprint_at_any_rung() {
        // Wanted: a coarse column (rung 2, 1,1) with two chunks, and four fine columns (rung 0) inside
        // the footprint of the rung-1 column (2,2), plus one on another face.
        let keys = vec![
            key(Face::PosX, 2, 1, 1, 0),
            key(Face::PosX, 2, 1, 1, 1),
            key(Face::PosX, 0, 4, 4, 0),
            key(Face::PosX, 0, 5, 4, 0),
            key(Face::PosX, 0, 4, 5, 0),
            key(Face::PosX, 0, 5, 5, 0),
            key(Face::NegY, 1, 2, 2, 0),
            key(Face::PosX, 2, 1, 1, 0), // a duplicate is kept once
        ];
        let w = WantedSet::from_keys(keys, 1000.0);
        assert_eq!(w.len(), 7);
        assert!(!w.is_empty());
        assert!(w.contains(key(Face::PosX, 0, 5, 5, 0)));
        assert!(!w.contains(key(Face::PosX, 0, 6, 5, 0)));
        assert_eq!((w.rung_min, w.rung_max), (0, 2));
        assert_eq!(w.per_rung(), vec![(0, 4), (1, 1), (2, 2)]);
        assert!((w.reach_m - 1000.0).abs() < 1e-12);
        // The held rung-1 column (2,2): its footprint holds the four fine columns and lies inside the
        // coarse (1,1). Nothing arrived: held.
        let held = Column {
            face: Face::PosX,
            rung: 1,
            x: 2,
            y: 2,
        };
        assert!(w.overlapping_missing(held, &|_| false));
        // Everything arrived: released.
        assert!(!w.overlapping_missing(held, &|_| true));
        // Only the coarse chunk (1,1,z=1) missing: held (the coarser-or-equal arm).
        assert!(w.overlapping_missing(held, &|k| k != key(Face::PosX, 2, 1, 1, 1)));
        // Only one fine chunk missing: held (the finer arm, filtered on y).
        assert!(w.overlapping_missing(held, &|k| k != key(Face::PosX, 0, 4, 5, 0)));
        // A fine chunk OUTSIDE the footprint missing: released — (6,5) is not wanted, so make (5,5)
        // arrive and pretend a neighbour column is the only miss by holding a column elsewhere.
        let elsewhere = Column {
            face: Face::PosX,
            rung: 1,
            x: 9,
            y: 9,
        };
        assert!(!w.overlapping_missing(elsewhere, &|_| false));
        // The other face's column never overlaps a PosX column.
        let other = Column {
            face: Face::NegY,
            rung: 0,
            x: 4,
            y: 4,
        };
        assert!(w.overlapping_missing(other, &|_| false));
        assert!(!w.overlapping_missing(other, &|k| k.face == Face::NegY));
        // A fine wanted column whose y lies outside the footprint's y range is not an overlap.
        let narrow = WantedSet::from_keys(vec![key(Face::PosX, 0, 4, 9, 0)], 1.0);
        assert!(!narrow.overlapping_missing(held, &|_| false));
        assert_eq!(WantedSet::default().per_rung(), Vec::<(u8, u64)>::new());
        assert!(WantedSet::default().is_empty());
    }

    /// THE WALK the gap was measured on: the pilot's step between two descents, and how many
    /// steps the test walks. The flight measured about one chunk born urgent every six metres of
    /// walking, so forty metres names several and costs a fraction of a second.
    const WALK_STEP_M: f64 = 1.0;
    const WALK_STEPS: usize = 250;
    const WALK_ALTITUDE_M: f64 = 3.4;

    /// THE CHUNKS BORN URGENT between two descents: the ones this ring calls URGENT that the
    /// descent before did not want AT ALL. A chunk born urgent cannot be resident on the frame it
    /// is first wanted, so the band reads a gap however fast the builders run.
    fn born_urgent(before: &WantedSet, now: &WantedSet) -> Vec<ChunkKey> {
        now.keys
            .iter()
            .copied()
            .filter(|k| {
                (now.class_of(*k) == BandClass::Urgent) & (before.class_of(*k) == BandClass::Absent)
            })
            .collect()
    }

    #[test]
    fn nothing_enters_the_ring_urgent_from_nothing_on_a_walk() {
        // MEASURED, the walk flight of 2026-09-16 (`scratchpad/walk_gap_fix.md`): the band went
        // incomplete on 18 frames of a minute's walk, and EVERY missing chunk read `was absent` -
        // the descent one frame earlier did not want it at all, while the DRAWN eye already wanted
        // it and the builders stood idle at 400 chunks a second. A chunk first wanted this frame
        // cannot be resident this frame, so the ask bought nothing.
        //
        // THE RULE: a chunk the ring calls URGENT must have been ASKED FOR before - as a reveal or
        // as a margin chunk. Nothing may enter the ring urgent from nothing. The walk here is the
        // flight's own: the pilot walks along the ground of the home planet at her own height over
        // the recipe's surface, and a descent runs at every step.
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let up = DVec3::new(d[0], d[1], d[2]);
        let east = up.cross(DVec3::Z).normalize();
        let radius = body.ladder().radius_m();
        let mut view = LadderView::default();
        let mut prev: Option<WantedSet> = None;
        let mut first: Option<WantedSet> = None;
        let mut born: Vec<(usize, ChunkKey)> = Vec::new();
        let mut step = 0usize;
        while step < WALK_STEPS {
            let arc = step as f64 * WALK_STEP_M / radius;
            let dir = (up * arc.cos() + east * arc.sin()).normalize();
            let surface = vd_terrain::height::height_m(&body, [dir.x, dir.y, dir.z], 0);
            let eye = (dir * (surface + WALK_ALTITUDE_M)).to_array();
            let now = view.wanted(&body, eye, None);
            if let Some(before) = prev.as_ref() {
                born.extend(born_urgent(before, &now).into_iter().map(|k| (step, k)));
            }
            first = first.or_else(|| Some(now.clone()));
            prev = Some(now);
            step += 1;
        }
        assert_eq!(
            born.iter().map(|(_, k)| *k).collect::<Vec<ChunkKey>>(),
            Vec::new(),
            "chunks entered the ring urgent from nothing over {:.0} m of walking: {born:?}",
            WALK_STEPS as f64 * WALK_STEP_M
        );
        // ★ THE DETECTOR ITSELF FIRES when there IS a birth, so an empty result above is the rule
        // holding and never a reader that cannot read. Against NO descent at all, every urgent
        // chunk of the walker's first ring is born urgent — which is exactly why the rule is about
        // two CONSECUTIVE descents and never about the first one: the first frame has nothing
        // behind it, and the pilot stands still while it is built.
        let first = first.expect("a walk of at least one step");
        assert_eq!(
            born_urgent(&WantedSet::default(), &first).len(),
            first.urgent_count()
        );
        assert!(first.urgent_count() > 0);
    }

    #[test]
    fn the_gap_names_its_chunks_and_one_eye_reads_one_column() {
        // THE WALK-GAP INSTRUMENT (2026-09-16): a missing urgent chunk states its class in a set
        // and its column's reading from an eye, so a gap line can say whether the ring first
        // wanted the chunk this very frame or asked for it earlier.
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [d[0], d[1], d[2]];
        let surface = vd_terrain::height::height_m(&body, dir, 0);
        let eye = [
            d[0] * (surface + 3.4),
            d[1] * (surface + 3.4),
            d[2] * (surface + 3.4),
        ];
        let mut view = LadderView::default();
        let w = view.wanted(&body, eye, None);
        // EVERY CLASS, counted through `class_of`: the urgent ones, the revealed peaks, and the
        // rest of every ring, which is the margin.
        let class_count = |c: BandClass| w.keys.iter().filter(|k| w.class_of(**k) == c).count();
        assert_eq!(class_count(BandClass::Urgent), w.urgent_count());
        assert_eq!(class_count(BandClass::Revealed), w.revealed_count());
        assert_eq!(
            class_count(BandClass::Margin),
            w.len() - w.urgent_count() - w.revealed_count()
        );
        assert_eq!(class_count(BandClass::Absent), 0);
        // A chunk the set does not hold is absent. The key stands over a face the eye is not on.
        let away = key(Face::NegX, 0, 0, 0, 0);
        assert!(!w.contains(away));
        assert_eq!(w.class_of(away), BandClass::Absent);
        assert_eq!(BandClass::Absent.name(), "absent");
        assert_eq!(BandClass::Margin.name(), "margin");
        assert_eq!(BandClass::Revealed.name(), "revealed");
        assert_eq!(BandClass::Urgent.name(), "urgent");
        // THE GAP, NAMED: with nothing arrived the keys are the urgent ones, capped, coarsest
        // rung first; with everything arrived there are none.
        let named = w.urgent_missing_keys(&|_| false, 4);
        assert_eq!(named.len(), 4);
        assert_eq!(named.iter().filter(|k| w.is_urgent(**k)).count(), 4);
        let mut rungs: Vec<u8> = named.iter().map(|k| k.rung).collect();
        let mut sorted = rungs.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(rungs, sorted);
        rungs.dedup();
        assert_eq!(w.urgent_missing_keys(&|_| true, 4), Vec::new());
        assert_eq!(
            w.urgent_missing_keys(&|_| false, w.len() + 1).len(),
            w.urgent_count()
        );
        // THE MARGIN'S GAP: with nothing arrived, every chunk that is neither urgent nor
        // revealed; with everything arrived, none.
        assert_eq!(
            w.margin_missing(&|_| false),
            w.len() - w.urgent_count() - w.revealed_count()
        );
        assert_eq!(w.margin_missing(&|_| true), 0);
        // ONE COLUMN READ FROM ONE EYE: an urgent chunk's column stands inside its rung's
        // territory and inside the horizon, so the same eye's own reading calls it urgent.
        let hot = named[named.len() - 1];
        let read = view.probe(&body, eye, hot);
        assert!(read.urgent);
        assert!(read.near_m <= read.horizon_m, "{read:?}");
        assert!(read.near_m < read.territory_m, "{read:?}");
        assert!(read.far_m > read.near_m, "{read:?}");
        // A REVEALED peak stands PAST the horizon, so the same reading does not call it urgent.
        let peak = *w
            .keys
            .iter()
            .find(|k| w.class_of(**k) == BandClass::Revealed)
            .expect("a peak past the horizon");
        let seen = view.probe(&body, eye, peak);
        assert!(!seen.urgent);
        // An eye at the body's own centre has no radial and reads nothing.
        assert_eq!(
            view.probe(&body, [0.0, 0.0, 0.0], hot),
            BandProbe::default()
        );
    }

    #[test]
    fn the_urgent_chunks_are_the_rungs_own_territory_and_the_gap_counts_the_missing() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [d[0], d[1], d[2]];
        let surface = vd_terrain::height::height_m(&body, dir, 0);
        let eye = [
            d[0] * (surface + 3.4),
            d[1] * (surface + 3.4),
            d[2] * (surface + 3.4),
        ];
        let mut view = LadderView::default();
        let w = view.wanted(&body, eye, None);
        // Some chunks are urgent and some are not (the margin past every switch), and every
        // urgent chunk is wanted.
        let urgent = w.urgent_count();
        assert!(urgent > 0);
        assert!(urgent < w.len(), "{urgent} of {}", w.len());
        assert_eq!(w.keys.iter().filter(|k| w.is_urgent(**k)).count(), urgent);
        // The column under the eye is urgent at rung 0; a rung-0 column past the switch is not.
        let foot = column_under(&body, d, 0);
        assert!(
            w.keys
                .iter()
                .any(|k| Column::of(*k) == foot && w.is_urgent(*k))
        );
        let margin = w
            .keys
            .iter()
            .filter(|k| k.rung == 0 && !w.is_urgent(**k))
            .count();
        assert!(margin > 0, "no rung-0 chunk in the margin");
        // The gap: with nothing arrived every urgent chunk is missing; with everything, none;
        // with the margin alone arrived, still every urgent one.
        assert_eq!(w.urgent_missing(&|_| false), urgent);
        assert_eq!(w.urgent_missing(&|_| true), 0);
        assert_eq!(w.urgent_missing(&|k| !w.is_urgent(k)), urgent);
        // The gap per rung sums to the gap, holds no empty rung, and is empty when all arrived.
        let per_rung = w.urgent_missing_per_rung(&|_| false);
        assert_eq!(
            per_rung.iter().map(|(_, n)| *n as usize).sum::<usize>(),
            urgent
        );
        assert!(per_rung.iter().all(|(_, n)| *n > 0));
        assert_eq!(w.urgent_missing_per_rung(&|_| true), Vec::new());
        // The peaks past the horizon are revealed, not urgent, and their gap counts apart.
        let revealed = w.revealed_count();
        assert!(revealed > 0, "no revealed peak from the ground");
        assert_eq!(w.revealed_missing(&|_| false), revealed);
        assert_eq!(w.revealed_missing(&|_| true), 0);
        // No urgent chunk is revealed: with every revealed chunk but this one arrived, nothing of
        // it is missing (the sets are disjoint).
        for k in w.keys.iter().filter(|k| w.is_urgent(**k)) {
            assert_eq!(w.revealed_missing(&|r| r != *k), 0);
        }
        // A set built from keys alone holds no territory.
        let plain = WantedSet::from_keys(w.keys.clone(), w.reach_m);
        assert_eq!(plain.urgent_count(), 0);
        assert_eq!(plain.revealed_count(), 0);
        // HYSTERESIS: the descent keeps the columns it wanted; a second descent from the same eye
        // wants the same set, and a step of half a metre changes it little — never the whole far
        // ring (MEASURED without the kept set: whole far columns flipped per step).
        let emitted: BTreeSet<Column> = w.keys.iter().map(|k| Column::of(*k)).collect();
        assert!(view.kept.is_superset(&emitted));
        assert!(
            view.kept.len() > emitted.len(),
            "no far column was kept for its clearance"
        );
        let again = view.wanted(&body, eye, None);
        assert_eq!(again.keys, w.keys);
        assert!(
            !view.culled.is_empty(),
            "no column was culled from the ground"
        );
        assert!(view.culled.is_disjoint(&view.kept));
        let step = [eye[0] + 0.4, eye[1] + 0.2, eye[2] - 0.1];
        let stepped = view.wanted(&body, step, None);
        let before: BTreeSet<ChunkKey> = w.keys.iter().copied().collect();
        let after: BTreeSet<ChunkKey> = stepped.keys.iter().copied().collect();
        let churn = before.symmetric_difference(&after).count();
        let len = w.len();
        assert!(
            churn * 50 < len,
            "{churn} chunks changed on a half-metre step of {len}"
        );
        // A higher eye pushes the horizon out over columns the skyline culled: they stay
        // skyline-judged until well inside it, and the ones it clears are urgent (the true
        // horizon), never revealed.
        let culled_before = view.culled.clone();
        let raised = view.wanted(
            &body,
            [
                d[0] * (surface + 12.0),
                d[1] * (surface + 12.0),
                d[2] * (surface + 12.0),
            ],
            None,
        );
        assert!(raised.len() > w.len());
        assert!(
            raised
                .keys
                .iter()
                .any(|k| culled_before.contains(&Column::of(*k)) && raised.is_urgent(*k)),
            "no column the ground culled is urgent from 12 m up"
        );
    }

    #[test]
    fn the_morton_curve_interleaves_the_bits_and_keeps_neighbours_close() {
        assert_eq!(morton(0, 0), 0);
        assert_eq!(morton(1, 0), 1);
        assert_eq!(morton(0, 1), 2);
        assert_eq!(morton(1, 1), 3);
        assert_eq!(morton(2, 0), 4);
        assert_eq!(morton(3, 5), 0b100111);
        assert_eq!(morton(-4, -9), 0);
        assert_eq!(morton(i32::MAX, i32::MAX), (1u64 << 62) - 1);
        // The four children of one 2×2 block are consecutive.
        let block: Vec<u64> = [(4, 6), (5, 6), (4, 7), (5, 7)]
            .iter()
            .map(|(x, y)| morton(*x, *y))
            .collect();
        assert_eq!(block, vec![56, 57, 58, 59]);
    }

    /// THE FLOOR (item 12): an eye ten metres UNDER the recipe's surface wants the same set as an
    /// eye standing on it — the horizon of the eye's own height, never zero.
    #[test]
    fn an_eye_under_the_surface_wants_what_an_eye_on_it_wants() {
        assert!((floored_altitude_m(-10.0) - EYE_HEIGHT_M).abs() < 1e-12);
        assert!((floored_altitude_m(0.0) - EYE_HEIGHT_M).abs() < 1e-12);
        assert!((floored_altitude_m(300.0) - 300.0).abs() < 1e-12);
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [d[0], d[1], d[2]];
        let surface = vd_terrain::height::height_m(&body, dir, 0);
        let at = |h: f64| {
            [
                d[0] * (surface + h),
                d[1] * (surface + h),
                d[2] * (surface + h),
            ]
        };
        let under = LadderView::default().wanted(&body, at(-10.0), None);
        let standing = LadderView::default().wanted(&body, at(EYE_HEIGHT_M), None);
        assert!((under.reach_m - standing.reach_m).abs() < 1e-6);
        assert_eq!(under.rung_min, standing.rung_min);
        // The skyline keeps its own truth: from under the ground the far rings are walled off
        // (MEASURED: the coarsest ring rung 3 against the standing eye's rung 10), so the set is
        // bounded by the standing eye's and never the reach-wide flood.
        assert!(under.rung_max <= standing.rung_max);
        let (a, b) = (under.keys.len(), standing.keys.len());
        assert!(a > 0, "under {a}, standing {b}");
        assert!(a <= b, "under {a}, standing {b}");
    }

    #[test]
    fn the_ladder_from_the_ground_reaches_the_horizon_coarse_first() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [d[0], d[1], d[2]];
        let surface = vd_terrain::height::height_m(&body, dir, 0);
        let eye = [
            d[0] * (surface + 3.4),
            d[1] * (surface + 3.4),
            d[2] * (surface + 3.4),
        ];
        let mut view = LadderView::default();
        let w = view.wanted(&body, eye, None);
        // The reach passes the 6.6 km horizon; the rings run from rung 0 to the ring that holds it.
        assert!(w.reach_m > horizon_m(surface, 3.4), "{}", w.reach_m);
        assert_eq!(w.rung_min, 0);
        // The coarsest ring holds the horizon at least, and never passes the reach's rung (a far
        // ring with no visible peak is empty and names no rung).
        assert!(w.rung_max >= rung_for_distance(horizon_m(surface, 3.4), body.ladder().rungs));
        // ★ IT IS THE DRAWN GROUND THAT THE REACH BOUNDS, NOT THE SHADOW CASTERS (re-read
        // 2026-09-18, slice 8b stage 3). A caster is a COARSER column a chunk asks for so its
        // shadow has something to fall on; its rung is a fixed step over the chunk's own
        // (`Shadow::caster_rung`) and reads no distance at all. MEASURED here: the reach is
        // 349 096 m, whose own rung is 9, and rungs 0 to 9 carry every urgent and every revealed
        // chunk — while 22 chunks stand at rung 10 and not one of them is drawn.
        //
        // The two agreed until this stage only because the reach was longer: the relief law halved
        // the home planet's mountains, so the peak a standing eye can see is lower and the reach
        // fell from over 445 km to 349 km, which is one rung.
        let drawn_max = w
            .keys
            .iter()
            .filter(|k| w.is_urgent(**k) | w.revealed.contains(*k))
            .map(|k| k.rung)
            .max()
            .expect("the ground under a standing eye names a rung");
        assert!(
            drawn_max <= rung_for_distance(w.reach_m, body.ladder().rungs),
            "the drawn ground reaches rung {drawn_max} at a reach of {} m",
            w.reach_m
        );
        // The reading is taken into a word BEFORE the assertion: an expression inside a passing
        // assertion's message never runs, and llvm counts it as a miss (HR5).
        let over = i32::from(w.rung_max) - i32::from(drawn_max);
        assert!(
            over <= 1,
            "the casters stand {over} rungs over the drawn ground"
        );
        // THE REQUEST ORDER (ruling V15): the classes never go back (urgent, then revealed, then
        // margin); within a class the rungs never rise (coarse first); within a class and a rung
        // each parent's children stand together (one run per parent).
        let rank = |k: &ChunkKey| -> u8 {
            if w.is_urgent(*k) {
                0
            } else if w.revealed.contains(k) {
                1
            } else {
                2
            }
        };
        let mut last_rank = 0u8;
        let mut last_rung = u8::MAX;
        // Per (class, rung): the runs of consecutive parents, and the distinct parents.
        type Runs = BTreeMap<(u8, u8), (usize, BTreeSet<(Face, i32, i32)>)>;
        let mut runs: Runs = BTreeMap::new();
        let mut last_parent: Option<(u8, u8, Face, i32, i32)> = None;
        for k in &w.keys {
            let r = rank(k);
            assert!(r >= last_rank, "a class went back in the key order");
            if r != last_rank {
                last_rung = u8::MAX;
            }
            assert!(
                k.rung <= last_rung,
                "a finer chunk before a coarser one in one class"
            );
            let parent = (r, k.rung, k.face, k.x.div_euclid(2), k.y.div_euclid(2));
            let entry = runs.entry((r, k.rung)).or_insert((0, BTreeSet::new()));
            if last_parent != Some(parent) {
                entry.0 += 1;
            }
            entry.1.insert((k.face, parent.3, parent.4));
            last_parent = Some(parent);
            last_rank = r;
            last_rung = k.rung;
        }
        for ((r, rung), (run_count, parents)) in &runs {
            assert_eq!(
                *run_count,
                parents.len(),
                "class {r} rung {rung}: a parent's children are split across the order"
            );
        }
        assert_eq!(rank(&w.keys[0]), 0, "the first chunk asked for is urgent");
        // THE PRIORITY follows the key order, and the class leads it across sets: a margin chunk
        // at index zero of another set ranks after an urgent chunk at the end of this one.
        let mut last_priority = 0u32;
        for (i, k) in w.keys.iter().enumerate() {
            let p = w.priority_of(i, *k);
            assert!(p >= last_priority, "the priority went back at {i}");
            last_priority = p;
        }
        let urgent_last = w
            .keys
            .iter()
            .rposition(|k| w.is_urgent(*k))
            .expect("an urgent chunk");
        let margin_first = w
            .keys
            .iter()
            .position(|k| !w.is_urgent(*k) && !w.revealed.contains(k))
            .expect("a margin chunk");
        assert!(
            w.priority_of(0, w.keys[margin_first])
                > w.priority_of(urgent_last, w.keys[urgent_last])
        );
        // The column under the eye is wanted at rung 0, and its chunks hold the surface.
        let foot = Column {
            face: face_of(d),
            rung: 0,
            x: 0,
            y: 0,
        };
        let (t, s) = face_coords(foot.face, d);
        let n0 = body.ladder().cells_per_edge(0);
        let foot = Column {
            x: index_of(unbend(t), n0) / CHUNK_EDGE as i32,
            y: index_of(unbend(s), n0) / CHUNK_EDGE as i32,
            ..foot
        };
        assert!(w.keys.iter().any(|k| Column::of(*k) == foot), "{foot:?}");
        // The rung-0 disc: roughly π·(switch(0)/chunk)² columns — between 400 and 1 200 — and every
        // ring a few hundred columns, each one or two chunks tall. Sized in the module doc; measured.
        let columns0: BTreeSet<(i32, i32)> = w
            .keys
            .iter()
            .filter(|k| k.rung == 0)
            .map(|k| (k.x, k.y))
            .collect();
        let columns0_n = columns0.len();
        assert!(
            (400..=1_200).contains(&columns0_n),
            "rung 0 holds {columns0_n} columns"
        );
        let per = w.per_rung();
        eprintln!(
            "ground ladder: {per:?}, reach {} m, columns0 {}",
            w.reach_m,
            columns0.len()
        );
        let rung0 = per.iter().find(|(r, _)| *r == 0).map_or(0, |(_, n)| *n);
        // ★ TWO OR THREE CHUNKS A COLUMN SINCE THE CAP-ROCK BENCH, WHERE IT WAS ONE OR TWO (slice
        // 8a stage 4, MEASURED on this very fixture with the bench's strength toggled off: 1 668
        // chunks over 877 columns before, 1 943 after — 1.90 a column against 2.22). The terrace
        // amplifies every variation under it by 1.4358, so the rung-0 column bound stands at 32.98 m
        // where it stood at 22.97 m, and twice it is over a 62-cell chunk: NO rung-0 column can fit
        // in one chunk any more. It is the bench's own price and it is stated, not hidden.
        assert!(
            rung0 < 3 * columns0_n as u64,
            "rung 0: {rung0} chunks over {columns0_n} columns"
        );
        // Past the horizon only peaks are wanted, and every ring carries its crossfade band (a
        // fifth more chunks than the ring alone): the whole ladder stays under THIRTEEN thousand
        // chunks (MEASURED 2026-09-09: 4 107 before the bands, 7 073 with them; 10 987 since slice
        // 8a stage 2; 12 391 since the cap-rock bench).
        //
        // ★ WHY THE CEILING MOVED FROM EIGHT THOUSAND TO TWELVE, AND IT IS A MEASUREMENT, NOT AN
        // ARGUMENT. Stage 2 makes five octaves RIDGED, and a ridged octave has a kink, so
        // `vd_terrain::digest::column_bound` must bound it by its FIRST derivative instead of its
        // curvature (`slice_8a_design.md` §2.2). The bound at rung 0 grows, every column's chunk
        // SPAN grows with it, and the ladder wants more chunks. The two causes were separated by
        // running this very fixture on the ridged field with the OLD curvature bound: 6 613 chunks,
        // which is FEWER than the 7 073 of 2026-09-09. So the rougher GROUND costs nothing here;
        // the honest BOUND costs 4 374 chunks, a growth of 66 %.
        //
        // ⚠ OWED, the arc's call and not this stage's: whether a ridged octave can carry a tighter
        // bound than its first derivative (a curvature term away from the kink plus a kink term). A
        // bound that is honest and loose costs chunks; a bound that is tight and wrong is a HOLE.
        //
        // ★ THIRTEEN THOUSAND SINCE THE CAP-ROCK BENCH (slice 8a stage 4), and the cause is the same
        // shape: a bound, not the ground. MEASURED on this fixture with the bench's strength toggled
        // off, 11 586 chunks; with it, 12 391 — a growth of 6.9 %, all of it in the column SPAN the
        // terrace's Lipschitz constant widens. The span already takes the LESSER of the terrace's
        // two honest bounds (`vd_terrain::digest::column_bound`), which is what keeps the rise at
        // 6.9 % instead of the 11.0 % the amplification alone asked for.
        let total = w.len();
        let reach = w.reach_m;
        assert!(total < 13_000, "{total} chunks: {per:?}, reach {reach} m");
        // The spans are kept for the columns the descent visits — a second call from the same
        // eye reads none anew and holds the same set; at the body's own centre nothing stays.
        let held = view.spans_held();
        let again = view.wanted(&body, eye, None);
        assert_eq!(again, w);
        assert_eq!(view.spans_held(), held);
        assert_eq!(view.wanted(&body, [0.0, 0.0, 0.0], None).len(), 0);
        assert_eq!(view.spans_held(), 0);
        // Every wanted chunk is in the ladder.
        for k in &w.keys {
            assert!(vd_terrain::lattice::in_ladder(&body, *k), "{k:?}");
        }
        // THE TILING: a direction inside the horizon is covered by EXACTLY ONE wanted column — no
        // hole, no double — at every distance from under the eye to the horizon (MEASURED before
        // the descent: holes at every ring boundary).
        assert_tiled(&body, &w, d, surface, 3.4);
    }

    /// A spiral of 400 directions from under the eye out to the horizon's ARC (the angle at the
    /// centre between the foot and the tangent point, `acos(R / (R + h))`), in every direction
    /// around the foot `d`: each is covered by exactly one wanted column, or by exactly two inside
    /// a crossfade band (the finer and the coarser rung of that band).
    fn assert_tiled(
        body: &BodyDefinition,
        w: &WantedSet,
        d: [f64; 3],
        surface: f64,
        altitude: f64,
    ) {
        let wanted_columns: BTreeSet<Column> = w.keys.iter().map(|k| Column::of(*k)).collect();
        let up = DVec3::from_array(d);
        let east = up.cross(DVec3::Z).normalize();
        let north = up.cross(east).normalize();
        let theta_max = (surface / (surface + altitude)).acos();
        let mut step = 0;
        while step < 400 {
            let theta = theta_max * (step as f64 + 0.5) / 400.0;
            let dist = theta * surface;
            let angle = step as f64 * 2.399_963; // the golden angle: no two on one radial
            let along = east * angle.cos() + north * angle.sin();
            // The point on the sphere at that arc.
            let p = (up * theta.cos() + along * theta.sin()).normalize();
            let mut covering = 0;
            let mut rung = 0u8;
            while rung < body.ladder().rungs {
                let col = column_under(body, p.to_array(), rung);
                covering += i32::from(wanted_columns.contains(&col));
                rung += 1;
            }
            // The point's own distance from the eye names the band it may be in. A chunk that
            // straddles a band's edge is wanted whole (its fragments past the edge carry no
            // weight and are discarded), so within a chunk's diagonal of a band two columns may
            // cover the point; elsewhere exactly one, and never none.
            let rungs = body.ladder().rungs;
            let eye = up * (surface + altitude);
            let d_eye = (p * surface - eye).length();
            // A chunk straddling a band's edge is wanted whole, so within one diagonal of the
            // coarser rung's chunk of a band two columns may cover the point: the interval around
            // the point, a diagonal each way, overlaps a band of the rule's rung or a neighbour.
            let rule = rung_for_distance(d_eye, rungs);
            let coarser = rule.saturating_add(1).min(rungs - 1);
            let slack = f64::from(cell_m(coarser)) * CHUNK_EDGE as f64 * std::f64::consts::SQRT_2;
            let mut near_band = false;
            let mut r = rule.saturating_sub(1);
            while r <= coarser {
                let (fade_in, fade_out) = fade_bands(r, rungs);
                for band in [fade_in, fade_out] {
                    near_band |= (band[0] < d_eye + slack) & (band[1] > d_eye - slack);
                }
                r += 1;
            }
            assert!(
                covering >= 1,
                "a point {dist:.0} m out ({d_eye:.0} m from the eye) is covered by no column"
            );
            assert!(
                covering <= 1 + i32::from(near_band),
                "a point {dist:.0} m out ({d_eye:.0} m from the eye) is covered by {covering} \
                 columns"
            );
            step += 1;
        }
    }

    #[test]
    fn the_ladder_from_orbit_is_the_globe_in_four_coarse_rings_and_no_pose_wants_nothing() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([0.2, 0.9, 0.4]);
        let r = body.ladder().radius_m();
        let mut view = LadderView::default();
        // 2 000 km up: the whole visible cap, coarsening outward by the tier rule alone.
        let orbit = [d[0] * (r + 2.0e6), d[1] * (r + 2.0e6), d[2] * (r + 2.0e6)];
        let w = view.wanted(&body, orbit, None);
        let top = body.ladder().rungs - 1;
        // ★ RE-MEASURED 2026-09-15 (the extended ladder). The cap spans FOUR rungs — 11 under the
        // nadir out to 14 at the horizon — and the ladder's own top (18, a 262 km cell) is six rungs
        // COARSER than any orbit needs. Before the extension the top rung was 12 and the cap stood
        // on it alone, because the ladder had nothing coarser to fall to.
        assert_eq!((w.rung_min, w.rung_max), (11, 14));
        assert!(
            w.rung_max < top,
            "an orbit needs no top rung: {} of {top}",
            w.rung_max
        );
        assert!(w.len() > 200, "{}", w.len());
        assert!(w.len() < 6_000, "{}", w.len());
        // The cap crosses face edges: more than one face is wanted, and the cap tiles out to the
        // horizon on every face it crosses.
        let faces: BTreeSet<Face> = w.keys.iter().map(|k| k.face).collect();
        assert!(faces.len() > 1, "{faces:?}");
        let dir = [d[0], d[1], d[2]];
        let surface = vd_terrain::height::height_m(&body, dir, 0);
        assert_tiled(&body, &w, d, surface, r + 2.0e6 - surface);
        // From orbit over a point near a FACE EDGE, 43° from the face's centre, the cap reaches
        // 83° from that centre (the refuter's finding: a fold through one face stopped at 76°).
        let edge_d = vd_seed::bend::normalize([1.0, 0.95, 0.05]);
        let edge_eye = [
            edge_d[0] * (r + 2.0e6),
            edge_d[1] * (r + 2.0e6),
            edge_d[2] * (r + 2.0e6),
        ];
        let w_edge = view.wanted(&body, edge_eye, None);
        let edge_dir = [edge_d[0], edge_d[1], edge_d[2]];
        let edge_surface = vd_terrain::height::height_m(&body, edge_dir, 0);
        assert_tiled(
            &body,
            &w_edge,
            edge_d,
            edge_surface,
            r + 2.0e6 - edge_surface,
        );
        // At the centre, and at a pose that is no pose: nothing. THREE RADII IS NOT FAR — the
        // globe stands there (`a_far_eye_draws_the_globe_at_every_distance`).
        assert_eq!(view.wanted(&body, [0.0, 0.0, 0.0], None).len(), 0);
        assert_eq!(view.wanted(&body, [f64::NAN, 0.0, 0.0], None).len(), 0);
    }

    /// ★ THE FAR EYE DRAWS THE GLOBE AT EVERY DISTANCE — THE CLIENT HAS NO FAR EDGE (owner
    /// 2026-09-15, *"agree"*: the server's visibility radius is the only rule; a realm has a row in
    /// the window only inside it, and the ladder draws a body at any distance while the body has a
    /// row). The pilot lifts her hull off the home planet and keeps climbing: the ground never
    /// leaves the screen, and it coarsens to the top rung by the descent's own rules — no branch
    /// on how far the eye stands.
    ///
    /// MEASURED BEFORE THIS (the owner's window flight of 2026-09-15): 2 706 chunks at 5 856 km of
    /// altitude, ZERO at 7 932 km with the body's plain outline standing in, 2 526 again at
    /// 5 086 km. The cutoff was two body radii. Its replacement — the ladder's own far edge, "one
    /// top-rung chunk column stands one pixel" — is deleted by the same ruling: the extended ladder
    /// put it at about 2 200 radii, 29 times outside the realm's stated 76.39, so it could never
    /// fire.
    #[test]
    fn a_far_eye_draws_the_globe_at_every_distance() {
        let body = home_planet();
        let r = body.ladder().radius_m();
        let top = body.ladder().rungs - 1;
        let d = vd_seed::bend::normalize([0.2, 0.9, 0.4]);
        let at = |k: f64| [d[0] * r * k, d[1] * r * k, d[2] * r * k];
        let set = |k: f64| LadderView::default().wanted(&body, at(k), None);
        // ★ THE COUNT FALLS WITH DISTANCE, and it falls to SIX (the extended ladder, owner
        // 2026-09-15). MEASURED on this direction: 1 128 chunks at 1.5 radii, 432 at 2, 216 at 3,
        // 121 at 5, 61 at 10, 27 at 20, and SIX — the whole globe, one chunk a face — from 34 radii
        // out, at every distance. Before the extension the same walk ROSE instead: 2 015, 2 867,
        // 3 716, 4 381, 4 875, 5 219, because the ladder's top rung held 9 600 columns and a far
        // eye wanted the visible hemisphere of them. (The reading was 877 before the crust was
        // rounded up to a whole top-rung cell. Only the NEAREST stand moved; every farther reading
        // is unchanged. WHY the nearest one moved is UNMEASURED.) ★ The nearest stand moved AGAIN,
        // 1 124 to 1 128, at slice 8a stage 2: the ridged band widens the column bound, so four more
        // columns of the nearest stand carry a second chunk. Every farther reading is unchanged,
        // because a coarse rung keeps no ridged octave. ★ AND BACK, 1 128 to 1 124, at stage 3: the
        // per-column roughness factor LOWERS the fine half of most columns, so the ground that
        // reached into a second chunk no longer does. Every farther reading is unchanged again, for
        // the same reason — a coarse rung keeps no fine octave, so the factor cannot reach it.
        // ★ AND THE NEAREST THREE STANDS MOVED at slice 8b stage 3, THE RELIEF LAW: 1 124 to 1 015
        // at 1.5 radii, 432 to 425 at 2, 216 to 214 at 3, 121 to 120 at 5. The relief halved, so a
        // column's ground reaches into fewer chunks of the band; the three farthest readings (61,
        // 27, 6) are unchanged, because from ten radii out the globe is drawn by its coarse rungs
        // and a coarse rung's chunk count is the ladder's own, not the ground's.
        let walk: Vec<usize> = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 34.0]
            .iter()
            .map(|k| set(*k).len())
            .collect();
        assert_eq!(walk, vec![1_015, 425, 214, 120, 61, 27, 6]);
        for pair in walk.windows(2) {
            assert!(pair[1] <= pair[0], "the count rose with distance: {walk:?}");
        }
        // THE TOP RUNG ALONE, ONE CHUNK A COLUMN, from 34 radii out — and the six columns are the
        // six faces, so the globe is drawn by one tile a face. The top rung's own chunk count is
        // SIX by the ladder's construction (one chunk per face edge since 2026-09-15).
        let per_edge = body.ladder().cells_per_edge(top) as usize;
        let per_edge = per_edge.div_ceil(vd_terrain::chunk::CHUNK_EDGE);
        let top_columns = 6 * per_edge * per_edge;
        assert_eq!(top_columns, 6, "the top rung holds {top_columns} columns");
        for k in [34.0, 100.0, 1_000.0] {
            let w = set(k);
            assert_eq!((w.rung_min, w.rung_max), (top, top), "{k} radii");
            assert_eq!((w.len(), w.columns()), (6, 6), "{k} radii");
            assert_eq!(
                w.urgent_count(),
                6,
                "{k} radii: every one is the picture's own need"
            );
            let faces: BTreeSet<Face> = w.keys.iter().map(|k| k.face).collect();
            assert_eq!(faces.len(), 6, "{k} radii: {faces:?}");
        }
        // ★ THE CLIENT HAS NO FAR EDGE (owner 2026-09-15, *"agree"*). An eye at a hundred and at a
        // thousand body radii still wants a NON-EMPTY set, at most the top rung's own six chunks,
        // and a thousand radii never wants more than ten radii does. The realm's stated visibility
        // radius is 76.39 radii — its parent ships no row past that — so a thousand radii is a
        // claim about the ladder's own arithmetic and never a picture anybody sees.
        let reach_radii =
            vd_core::geometry::visibility_reach_m(r, vd_core::geometry::VISIBILITY_THETA_MIN_RAD)
                / r;
        assert!((76.0..77.0).contains(&reach_radii), "{reach_radii} radii");
        let at_ten = set(10.0).len();
        for k in [100.0, 1_000.0] {
            let n = set(k).len();
            assert!((1..=top_columns).contains(&n), "{k} radii: {n}");
            assert!(n <= at_ten, "{k} radii: {n} over ten radii's {at_ten}");
        }
    }

    /// ★ THE TOP RUNG'S TERRITORY REACHES EVERYWHERE, because there is no coarser rung to hand the
    /// ground to — the same branch [`AskBound::fade_bands`] makes for the fade-out band.
    ///
    /// MEASURED BEFORE THIS (the far-eye probe): an eye 1.5 radii over the home planet wanted
    /// 2 015 top-rung chunks and called 1 857 of them MARGIN — "a finer ring already covers this
    /// ground" — because the top rung's own switch distance is 0.56 radii. The band could never
    /// read incomplete aloft, and the workers built the ground the pilot had already left.
    #[test]
    fn the_top_rungs_territory_has_no_outer_edge_and_every_other_rungs_is_its_switch() {
        let rungs = 13u8;
        let free = AskBound::unbounded();
        assert_eq!(free.territory_m(rungs - 1, rungs), f64::INFINITY);
        assert_eq!(free.territory_m(0, rungs), switch_m(0));
        assert_eq!(free.territory_m(rungs - 2, rungs), switch_m(rungs - 2));
        // A bound rung reads the bound's own radius; the top rung still reaches everywhere.
        let bound = AskBound::from_switches(vec![100.0; usize::from(rungs)]);
        assert_eq!(bound.territory_m(0, rungs), 100.0);
        assert_eq!(bound.territory_m(rungs - 1, rungs), f64::INFINITY);
        // The ground on the TOP RUNG is URGENT, never MARGIN: from 34 radii the top rung is the
        // only rung the descent reaches, and every one of its six chunks is the picture's own need.
        // ★ RE-MEASURED 2026-09-15: 1.5 radii no longer reads the top rung at all — the extended
        // ladder answers there with rungs 12 to 14, whose territory IS their own switch distance,
        // so a margin chunk there is the rule working and not the defect this test pins.
        let body = home_planet();
        let r = body.ladder().radius_m();
        let d = vd_seed::bend::normalize([0.2, 0.9, 0.4]);
        let top_only = LadderView::default().wanted(
            &body,
            [d[0] * r * 34.0, d[1] * r * 34.0, d[2] * r * 34.0],
            None,
        );
        assert_eq!(top_only.rung_min, body.ladder().rungs - 1);
        assert_eq!(
            top_only.urgent_count(),
            top_only.len(),
            "a top-rung chunk was classed MARGIN"
        );
        let aloft = LadderView::default().wanted(
            &body,
            [d[0] * r * 1.5, d[1] * r * 1.5, d[2] * r * 1.5],
            None,
        );
        assert_eq!((aloft.rung_min, aloft.rung_max), (12, 14));
    }

    #[test]
    fn a_stand_near_a_face_corner_wants_columns_on_three_faces() {
        let body = home_planet();
        // The corner where +X, +Y and +Z meet, a little inside the +X face.
        let d = vd_seed::bend::normalize([1.0, 0.97, 0.97]);
        let dir = [d[0], d[1], d[2]];
        let surface = vd_terrain::height::height_m(&body, dir, 0);
        // From 60 km up the horizon is 877 km: the three faces at the corner are all inside it.
        let eye = [
            d[0] * (surface + 60_000.0),
            d[1] * (surface + 60_000.0),
            d[2] * (surface + 60_000.0),
        ];
        let w = LadderView::default().wanted(&body, eye, None);
        let faces: BTreeSet<Face> = w.keys.iter().map(|k| k.face).collect();
        assert_eq!(faces.len(), 3, "{faces:?}");
        for k in &w.keys {
            assert!(vd_terrain::lattice::in_ladder(&body, *k), "{k:?}");
        }
        // The three faces tile the cap around the corner out to the horizon.
        assert_tiled(&body, &w, d, surface, 60_000.0);
    }
}

/// ★ THE BOUNDED ASK's own tests (ruling F9 item 1): the horizon arithmetic, both arms of every
/// branch, and the invariant the descent owes — inside a rung's horizon the tier rule's own rung is
/// asked, past it the next rung, and no ground is ever left unasked.
#[cfg(test)]
mod ask_bound_tests {
    use super::*;
    use vd_terrain::home::home_planet;

    /// A rate that binds on the home planet: a hull at 528 m/s over the ground, three workers.
    fn fast() -> AskRate {
        AskRate {
            chunks_per_s: 195.0,
            speed_mps: 528.0,
            altitude_m: EYE_HEIGHT_M,
            chunks_per_column: 2.0,
        }
    }

    /// THE BOUND NEVER BINDS where the builders cover the ask: nothing measured, nothing moving,
    /// no column, no ladder — and a machine whose throughput is large. Every one of them reads the
    /// tier rule's own switch distance at every rung.
    #[test]
    fn a_covered_ask_reads_the_tier_rules_own_radii() {
        let rungs = 12u8;
        let covered = [
            AskRate {
                chunks_per_s: 0.0,
                ..fast()
            },
            AskRate {
                speed_mps: 0.0,
                ..fast()
            },
            AskRate {
                chunks_per_column: 0.0,
                ..fast()
            },
            AskRate {
                chunks_per_s: 1.0e9,
                ..fast()
            },
        ];
        for rate in covered {
            let bound = ask_bound(rungs, rate);
            assert!(!bound.binds(), "{rate:?} bound the ask");
            assert_eq!(bound.horizons_m(), Vec::<f64>::new());
            let mut rung = 0u8;
            while rung < rungs {
                assert_eq!(bound.switch_m(rung), switch_m(rung));
                rung += 1;
            }
        }
        // A ladder with no rung at all: nothing to bound.
        assert!(!ask_bound(0, fast()).binds());
    }

    /// AT SPEED ON A SMALL SHARE the bound binds: the finest rungs come in, every rung stays
    /// inside the tier rule's own radius, and no two rungs share a distance (a shared crossfade
    /// band would draw a half-transparent shell).
    #[test]
    fn a_share_that_cannot_keep_up_brings_the_finest_rungs_in() {
        let rungs = 12u8;
        let bound = ask_bound(rungs, fast());
        assert!(bound.binds());
        let horizons = bound.horizons_m();
        assert_eq!(horizons.len(), usize::from(rungs));
        assert!(horizons[0] < switch_m(0), "{horizons:?}");
        let mut rung = 0u8;
        while rung < rungs {
            assert!(bound.switch_m(rung) <= switch_m(rung), "rung {rung}");
            assert!(bound.switch_m(rung) > 0.0, "rung {rung}");
            rung += 1;
        }
        rung = 1;
        while rung < rungs {
            // Every finer rung's switch is at most half its coarser neighbour's, as the tier
            // rule's own are.
            let finer = bound.switch_m(rung - 1);
            let coarser = bound.switch_m(rung);
            assert!(
                finer <= coarser * 0.5 + 1.0e-9,
                "rung {rung}: {finer}, {coarser}"
            );
            rung += 1;
        }
        // A rung past the bound's own ladder reads the tier rule's own.
        assert_eq!(bound.switch_m(rungs), switch_m(rungs));
    }

    /// THE BUILDERS' THROUGHPUT MOVES THE HORIZON OUT, and the eye's speed moves it in.
    #[test]
    fn the_horizon_follows_the_throughput_and_the_speed() {
        let slow = ask_bound(12, fast());
        let faster_builders = ask_bound(
            12,
            AskRate {
                chunks_per_s: 400.0,
                ..fast()
            },
        );
        assert!(faster_builders.switch_m(0) > slow.switch_m(0));
        let faster_eye = ask_bound(
            12,
            AskRate {
                speed_mps: 1056.0,
                ..fast()
            },
        );
        assert!(faster_eye.switch_m(0) < slow.switch_m(0));
    }

    /// AN EYE ALOFT pays nothing for its empty finest rings: a ring whose slant distance does not
    /// reach the ground has no ask at all, so the budget it leaves buys the coarser rings.
    #[test]
    fn a_ring_that_reaches_no_ground_costs_the_builders_nothing() {
        // A kilometre up, the rung-0 ring (869 m) holds no ground at all.
        let aloft = AskRate {
            altitude_m: 1_000.0,
            ..fast()
        };
        assert_eq!(ground_radius_m(switch_m(0), aloft.altitude_m), 0.0);
        assert!(ground_radius_m(switch_m(4), aloft.altitude_m) > 0.0);
        assert_eq!(ask_rate_per_s(0, switch_m(0), aloft), 0.0);
        let bound = ask_bound(12, aloft);
        assert!(bound.binds());
        // The rings the eye can actually see stand farther out than the same rings do for an eye
        // on the ground, whose rung-0 disc is full of ground and eats the budget.
        assert!(bound.switch_m(1) > ask_bound(12, fast()).switch_m(1));
        assert!(bound.switch_m(2) > ask_bound(12, fast()).switch_m(2));
    }

    /// ★ THE FLIGHT'S OWN OPERATING POINTS (the moving eye at the average machine's three
    /// workers, MEASURED 2026-09-13): the bound must be INERT where the builders keep up and must
    /// BIND where they cannot, at the very numbers the flight reads off the stamp.
    #[test]
    fn the_bound_is_inert_at_240_and_binds_at_528() {
        let body = home_planet();
        let rungs = body.ladder().rungs;
        // A column of the home planet holds one chunk where the surface crosses one; the flight's
        // own descents read it, and both points below use that.
        let column = 1.0;
        // The 240 m/s leg: 223 m/s read from the lead, 1 117 m up, 170 chunks a second built.
        let steady = ask_bound(
            rungs,
            AskRate {
                chunks_per_s: 170.0,
                speed_mps: 223.0,
                altitude_m: 1_117.0,
                chunks_per_column: column,
            },
        );
        assert!(!steady.binds(), "{:?}", steady.horizons_m());
        // The 528 m/s leg: 515 m/s, 766 m up, 193 chunks a second. The builders cannot hold the
        // ask, and the finest rungs come in.
        let fast = ask_bound(
            rungs,
            AskRate {
                chunks_per_s: 193.0,
                speed_mps: 515.0,
                altitude_m: 766.0,
                chunks_per_column: column,
            },
        );
        assert!(fast.binds());
        assert!(fast.switch_m(1) < switch_m(1), "{:?}", fast.horizons_m());
    }

    /// THE HYSTERESIS reads two bounds as the same while every rung is within its fraction, and
    /// as different past it — symmetric, and never zero-width.
    #[test]
    fn the_hysteresis_keeps_a_jittering_horizon_and_adopts_a_moved_one() {
        let rungs = 12u8;
        let a = ask_bound(rungs, fast());
        let hair = ask_bound(
            rungs,
            AskRate {
                chunks_per_s: 200.0,
                ..fast()
            },
        );
        let far = ask_bound(
            rungs,
            AskRate {
                chunks_per_s: 60.0,
                ..fast()
            },
        );
        assert!(a.same_as(&hair, rungs, 0.25));
        assert!(hair.same_as(&a, rungs, 0.25));
        assert!(!a.same_as(&hair, rungs, ASK_BOUND_HYSTERESIS));
        assert!(!a.same_as(&far, rungs, 0.25));
        assert!(!far.same_as(&a, rungs, 0.25));
        // The unbounded pair is the same as itself, at every rung.
        let free = AskBound::unbounded();
        assert!(free.same_as(&AskBound::unbounded(), rungs, 0.0));
        assert!(!free.same_as(&a, rungs, ASK_BOUND_HYSTERESIS));
    }

    /// ★ THE HORIZON SLIDES, IT NEVER JUMPS: a step of one frame moves each rung by a slice of its
    /// own horizon, a long step lands exactly on the target, a settled bound stops moving, and the
    /// way back to the tier rule's own radii ends in [`AskBound::unbounded`].
    #[test]
    fn the_horizon_slides_toward_its_target_and_settles_on_it() {
        let rungs = 12u8;
        let target = ask_bound(rungs, fast());
        let free = AskBound::unbounded();
        // One frame of a fortieth of a second: a step, not the target.
        let frame = 1.0 / 40.0;
        let one = free.slewed_toward(&target, rungs, frame);
        assert!(one.binds());
        assert!(one.switch_m(0) < free.switch_m(0));
        assert!(one.switch_m(0) > target.switch_m(0));
        // Enough seconds to cross the whole distance: exactly the target.
        let landed = free.slewed_toward(&target, rungs, 1_000.0);
        assert_eq!(landed.horizons_m(), target.horizons_m());
        // A bound already at its target does not move, at any step.
        assert_eq!(
            target.slewed_toward(&target, rungs, frame).horizons_m(),
            target.horizons_m()
        );
        // And the way home ends unbounded, never at a bound that merely equals the tier rule's.
        assert!(!target.slewed_toward(&free, rungs, 1_000.0).binds());
        // A rung bound in to nothing can still grow: its own column's width is its floor step.
        let nothing = ask_bound(
            rungs,
            AskRate {
                chunks_per_s: 1.0e-9,
                altitude_m: 0.0,
                ..fast()
            },
        );
        assert_eq!(nothing.switch_m(0), 0.0);
        assert!(nothing.slewed_toward(&free, rungs, frame).switch_m(0) > 0.0);
        // A moment that went backwards moves nothing.
        assert_eq!(
            free.slewed_toward(&target, rungs, -1.0).horizons_m(),
            free.slewed_toward(&target, rungs, 0.0).horizons_m()
        );
    }

    /// THE BANDS FOLLOW THE BOUND: a rung's fade-out band sits on its own effective switch, its
    /// fade-in band on the finer rung's, rung 0 is always in, and the top rung never fades out.
    #[test]
    fn the_crossfade_bands_read_the_effective_switch_distances() {
        let body = home_planet();
        let rungs = body.ladder().rungs;
        let bound = ask_bound(rungs, fast());
        let (in0, out0) = bound.fade_bands(0, rungs);
        assert_eq!(in0, FADE_ALWAYS_IN);
        assert_eq!(out0[0], HYSTERESIS_IN * bound.switch_m(0));
        assert_eq!(out0[1], HYSTERESIS_OUT * bound.switch_m(0));
        let (in1, _) = bound.fade_bands(1, rungs);
        assert_eq!(in1[0], HYSTERESIS_IN * bound.switch_m(0));
        let (_, top_out) = bound.fade_bands(rungs - 1, rungs);
        assert_eq!(top_out, FADE_ALWAYS_OUT);
        // The sink's ramp moves with the band, and rung 0 sinks nowhere.
        assert_eq!(bound.sink_end_m(&body, 0, rungs), FADE_ALWAYS_IN[1]);
        assert!(bound.sink_end_m(&body, 1, rungs) > in1[1]);
        // The unbounded bound is the free functions, step for step.
        let free = AskBound::unbounded();
        assert_eq!(free.fade_bands(1, rungs), fade_bands(1, rungs));
        assert_eq!(
            free.sink_end_m(&body, 1, rungs),
            sink_end_m(&body, 1, rungs)
        );
    }

    /// ★ THE DESCENT'S SLACK COVERS THE SLIDE (ruling F9 item 1, the frame bar; review item 2).
    ///
    /// The picture and the descent read the horizon at two tolerances: the materials follow the
    /// held horizon within [`ASK_BOUND_REBIND`], the descent within [`ASK_BOUND_BRACKET`]. This
    /// walks THE WORST STAND OF BOTH AT ONCE — a drawn horizon a rebind OUT and an asked horizon a
    /// bracket IN — and asserts that every band the picture draws lies inside the ring the descent
    /// asked for. It asserts the guard itself, so the walk cannot go vacuous, and it asserts that
    /// the bracket ALONE falls short, which is why [`ASK_BOUND_SLACK`] is derived from the two.
    #[test]
    fn the_descents_slack_covers_every_band_the_horizon_may_slide_to() {
        let rungs = 12u8;
        let held = ask_bound(rungs, fast());
        assert!(
            held.binds(),
            "the fixture must bind for this to mean anything"
        );
        // The slack is not the bound: the same switches read as the same bound.
        let asked = held.clone().with_slack(ASK_BOUND_SLACK);
        assert!(asked.same_as(&held, rungs, 0.0));
        // THE WORST STAND OF BOTH TOLERANCES AT ONCE, at every rung, built outright instead of
        // hoping some rate lands there.
        let scaled = |bound: &AskBound, k: f64| {
            let mut switches = Vec::new();
            let mut rung = 0u8;
            while rung < rungs {
                switches.push(bound.switch_m(rung) * k);
                rung += 1;
            }
            AskBound::from_switches(switches)
        };
        // A HAIR inside each tolerance, so the guard's own comparison is decided by the
        // arithmetic and not by the last bit of a double at the exact boundary.
        let hair = 1.0e-12;
        let worst_asked = scaled(&held, 1.0 - ASK_BOUND_BRACKET + hair).with_slack(ASK_BOUND_SLACK);
        let worst_drawn = scaled(&held, 1.0 / (1.0 - ASK_BOUND_REBIND) - hair);
        // THE GUARD ITSELF: these two really are a bracket and a rebind from the held horizon, so
        // the walk below is a stand the pace can produce and not a pair that never happens.
        assert!(
            worst_asked.same_as(&held, rungs, ASK_BOUND_BRACKET),
            "the asked horizon must sit inside the descent's own bracket"
        );
        assert!(
            worst_drawn.same_as(&held, rungs, ASK_BOUND_REBIND),
            "the drawn horizon must sit inside the materials' own rebind"
        );
        let mut checked = 0u32;
        let mut rung = 0u8;
        while rung < rungs {
            let (ask_in, ask_out) = worst_asked.fade_bands(rung, rungs);
            let (drawn_in, drawn_out) = worst_drawn.fade_bands(rung, rungs);
            assert!(ask_in[0] <= drawn_in[0], "rung {rung}: the fade-in's start");
            assert!(ask_in[1] >= drawn_in[1], "rung {rung}: the fade-in's end");
            assert!(
                ask_out[0] <= drawn_out[0],
                "rung {rung}: the fade-out's start"
            );
            assert!(
                ask_out[1] >= drawn_out[1],
                "rung {rung}: the fade-out's end"
            );
            checked += 1;
            rung += 1;
        }
        assert_eq!(checked, u32::from(rungs), "every rung must be walked");
        // AND THE BRACKET ALONE FALLS SHORT: the same worst stand asked with a slack of the
        // bracket draws a band outside the ring the descent asked for. This is the defect the
        // derivation cures, asserted so it cannot come back.
        let short = scaled(&held, 1.0 - ASK_BOUND_BRACKET + hair).with_slack(ASK_BOUND_BRACKET);
        let (_, short_out) = short.fade_bands(0, rungs);
        let (_, drawn_out) = worst_drawn.fade_bands(0, rungs);
        assert!(
            short_out[1] < drawn_out[1],
            "the bracket alone must fall short of the drawn band"
        );
        // No slack is the bands themselves.
        let exact = ask_bound(rungs, fast());
        assert_eq!(
            exact.fade_bands(2, rungs),
            exact.clone().with_slack(0.0).fade_bands(2, rungs)
        );
    }

    /// A RUNG BOUND IN TO NOTHING has no finer rung to fade in from, so it is always in and sinks
    /// nowhere — exactly as rung 0 is. (An eye AT the surface whose builders cannot hold even the
    /// coarsest ring: the arithmetic's own floor.)
    #[test]
    fn a_rung_with_no_finer_neighbour_is_always_in() {
        let body = home_planet();
        let rungs = body.ladder().rungs;
        let none = AskRate {
            chunks_per_s: 1.0e-9,
            altitude_m: 0.0,
            ..fast()
        };
        let bound = ask_bound(rungs, none);
        assert!(bound.binds());
        assert_eq!(bound.switch_m(0), 0.0);
        let (fade_in, _) = bound.fade_bands(1, rungs);
        assert_eq!(fade_in, FADE_ALWAYS_IN);
        assert_eq!(bound.sink_end_m(&body, 1, rungs), FADE_ALWAYS_IN[1]);
    }

    /// THE CHUNKS A COLUMN HOLDS, as the descent measures it: the default while nothing is wanted,
    /// and the set's own ratio once it is.
    #[test]
    fn a_wanted_set_states_the_chunks_a_column_holds() {
        let empty = WantedSet::default();
        assert_eq!(empty.columns(), 0);
        assert_eq!(empty.chunks_per_column(2.0), 2.0);
        let body = home_planet();
        let r = body.ladder().radius_m();
        let set = LadderView::default().wanted(&body, [r + EYE_HEIGHT_M, 0.0, 0.0], None);
        assert!(set.columns() > 0);
        let per = set.chunks_per_column(2.0);
        assert!((1.0..=4.0).contains(&per), "{per}");
        assert_eq!(per, set.len() as f64 / set.columns() as f64);
    }

    /// ★ THE INVARIANT (ruling F9 item 1): with the bound in force, a column of the finest rung
    /// INSIDE that rung's deliverable horizon is still asked at the tier rule's own rung; one
    /// BEYOND the horizon is asked at the next rung instead; and no ground the unbounded ask
    /// wanted is left unasked.
    #[test]
    fn inside_the_horizon_the_tier_rules_rung_is_asked_and_beyond_it_the_next() {
        let body = home_planet();
        let rungs = body.ladder().rungs;
        let r = body.ladder().radius_m();
        let eye = [r + EYE_HEIGHT_M, 0.0, 0.0];
        let free = LadderView::default().wanted(&body, eye, None);
        let bound = ask_bound(rungs, fast());
        assert!(bound.binds());
        let mut view = LadderView {
            bound: bound.clone(),
            ..LadderView::default()
        };
        let held = view.wanted(&body, eye, None);
        assert!(!held.is_empty());
        // The bounded ask is the smaller one, and it starts no finer than the free one.
        assert!(held.len() < free.len(), "{} {}", held.len(), free.len());
        assert!(held.rung_min >= free.rung_min);
        let ladder = *body.ladder();
        let surface = vd_terrain::height::height_m(&body, [1.0, 0.0, 0.0], 0);
        let frame = EyeFrame::new(DVec3::from_array(eye));
        let near_of = |col: Column| {
            column_geometry(&ladder, &frame, DVec3::from_array(eye), col, surface).near
        };
        let mut inside = 0u32;
        let mut beyond = 0u32;
        let columns: BTreeSet<Column> = free.keys.iter().map(|k| Column::of(*k)).collect();
        for col in columns {
            // NEVER UNASKED: some wanted chunk of the bounded ask stands over this ground.
            assert!(
                held.overlapping_missing(col, &|_| false),
                "{col:?} was left unasked"
            );
            let at_own_rung = held.keys.iter().any(|k| {
                (k.rung == col.rung) & (k.x == col.x) & (k.y == col.y) & (k.face == col.face)
            });
            let near = near_of(col);
            let horizon = bound.switch_m(col.rung);
            // Inside the rung's own deliverable horizon: the tier rule's rung still stands.
            if near < horizon {
                assert!(at_own_rung, "{col:?} inside {horizon} m was not asked");
                inside += 1;
            }
            // Past the crossfade band around that horizon: the next rung stands instead.
            if near > HYSTERESIS_OUT * horizon {
                assert!(!at_own_rung, "{col:?} past {horizon} m was asked anyway");
                beyond += 1;
            }
        }
        // Both arms were exercised: the bound bit, and it did not take everything.
        assert!(inside > 0);
        assert!(beyond > 0);
    }
}
