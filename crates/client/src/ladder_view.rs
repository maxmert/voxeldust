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

use crate::skyline::{DISC_MARGIN, EyeFrame, Skyline, WALL_MAX_HALF_ANGLE};
use vd_seed::bend::{Face, direction, face_coords, face_of, unbend};
use vd_seed::ladder::{cell_m, face_param, index_of};
use vd_terrain::BodyDefinition;
use vd_terrain::Gf;
use vd_terrain::chunk::{CHUNK_EDGE, ChunkKey};
use vd_terrain::digest::ColumnSpan;

/// THE HYSTERESIS PAIR (D8-2, starting values; the pop detector fixes them): the crossfade band
/// around a switch distance `s` runs from `HYSTERESIS_IN · s`, where the finer rung starts to fade
/// out and the coarser to fade in, to `HYSTERESIS_OUT · s`, where the finer is gone and the coarser
/// whole. A column moving in switches to the finer rung's full weight at the inner edge, one moving
/// out to the coarser's at the outer edge, and neither edge flips anything.
pub const HYSTERESIS_IN: f64 = 0.9;
pub const HYSTERESIS_OUT: f64 = 1.1;
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
    let band = |r: u8| [HYSTERESIS_IN * switch_m(r), HYSTERESIS_OUT * switch_m(r)];
    let fade_in = if rung == 0 {
        FADE_ALWAYS_IN
    } else {
        band(rung - 1)
    };
    let fade_out = if rung + 1 >= rungs {
        FADE_ALWAYS_OUT
    } else {
        band(rung)
    };
    (fade_in, fade_out)
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
    let (fade_in, _) = fade_bands(rung, rungs);
    if rung == 0 {
        return fade_in[1];
    }
    let crease = f64::from(cell_m(rung - 1));
    let sink = crate::chunks::sink_m(body, rung);
    // sink > crease always: the sink holds a cell of each rung and the gap bound.
    fade_in[1] + (fade_in[1] - fade_in[0]) * crease / (sink - crease)
}

/// Whether a distance lies inside a crossfade band of some rung: where two rungs share the ground.
#[must_use]
pub fn in_fade_band(d_m: f64, rungs: u8) -> bool {
    let rung = rung_for_distance(d_m, rungs);
    let (fade_in, fade_out) = fade_bands(rung, rungs);
    ((d_m > fade_in[0]) & (d_m < fade_in[1])) | ((d_m > fade_out[0]) & (d_m < fade_out[1]))
}
/// How many body radii from the centre an eye may stand and still get the ladder. Past it the
/// whole hemisphere is in view and the body's proxy outline stands in (the handover is a seam the
/// pop detector measures in step 6; MEASURED on the first ground picture: two planets 1.5 × 10¹¹ m
/// away each cost a column of chunks placed where nobody looks).
pub const FAR_EYE_RADII: f64 = 2.0;
// The descent's roots are EVERY top-rung column of EVERY face (at most 64 per edge by the ladder's
// construction, so at most 24 576 on any body), each tested by its nearest point against the
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
    f64::from(cell_m(rung)) / pixel_rad()
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

/// THE HORIZON of a smooth sphere of `radius_m` seen from `altitude_m` over it, in metres along the
/// line of sight: `√(2Rh + h²)`. A height under the surface reads as zero. The one formula the stamp,
/// the gate and the reach share.
#[must_use]
pub fn horizon_m(radius_m: f64, altitude_m: f64) -> f64 {
    let h = altitude_m.max(0.0);
    (2.0 * radius_m * h + h * h).sqrt()
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
    horizon_m(surface_m, altitude_m) + horizon_m(surface_m, relief_m)
}

/// The tallest ground the recipe can raise over its radius, in metres: the recipe's own bound.
#[must_use]
pub fn relief_m(body: &BodyDefinition) -> f64 {
    body.relief_bound_m(0).to_f64()
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
    /// How far the ladder reaches, in metres, and the rungs it holds.
    pub reach_m: f64,
    pub rung_min: u8,
    pub rung_max: u8,
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

    /// THE REVEALS' GAP: how many revealed chunks have NOT `arrived`.
    #[must_use]
    pub fn revealed_missing(&self, arrived: &dyn Fn(ChunkKey) -> bool) -> usize {
        self.revealed.iter().filter(|k| !arrived(**k)).count()
    }

    /// How many chunks are wanted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
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
    spans: BTreeMap<Column, (ColumnSpan, u64)>,
    generation: u64,
    /// THE KEPT COLUMNS (slice 8 step 4): every column the last descent wanted. A column past
    /// the horizon that was wanted is KEPT while its peak stands within [`KEEP_MARGIN_RAD`] under
    /// the skyline, and a new one is wanted only when it stands within [`WANT_MARGIN_RAD`] under
    /// it: hysteresis on the skyline's verdict. MEASURED on the walk of M8-1 without it: a
    /// column just at the skyline flipped between hidden and seen as the eye moved half a
    /// metre, and each flip released and rebuilt it — 4 chunks missing on 80 of 1 383 samples.
    /// Keeping is free (the chunk is resident); rebuilding is not.
    kept: BTreeSet<Column>,
    /// THE CULLED COLUMNS: every column past the horizon the last descent judged hidden. Such a
    /// column stays skyline-judged until it lies well inside the horizon
    /// ([`HORIZON_HYSTERESIS`]): the horizon moves with the eye's height (a walker over a bump,
    /// 1.8 m to 2.5 m, moves it from 4.8 km to 5.7 km), and MEASURED on the walk of M8-1 a
    /// column hidden by the skyline at one step stood inside the horizon at the next, was
    /// wanted unconditionally, and was built for nothing — 2 chunks missing on 17 of 1 291
    /// samples.
    culled: BTreeSet<Column>,
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
    per_rung: Vec<Vec<(ChunkKey, Territory)>>,
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
        let per_rung = &mut self.per_rung;
        let edge = CHUNK_EDGE as i32;
        let (fade_in, fade_out) = fade_bands(col.rung, ladder.rungs);
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
            let inside = (geo.near < switch_m(col.rung)) & (geo.far > fade_in[1]);
            let territory = match (inside, inside_horizon) {
                (false, _) => Territory::Margin,
                (true, true) => Territory::Urgent,
                (true, false) => Territory::Revealed,
            };
            let top_z = vd_terrain::digest::top_chunk_z(body, col.rung);
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
    fn span(&mut self, body: &BodyDefinition, col: Column) -> ColumnSpan {
        let generation = self.generation;
        let entry = self.spans.entry(col).or_insert_with(|| {
            (
                vd_terrain::digest::surface_column(body, col.face, col.rung, col.x, col.y),
                generation,
            )
        });
        entry.1 = generation;
        entry.0
    }

    /// How many spans the view holds.
    #[must_use]
    pub fn spans_held(&self) -> usize {
        self.spans.len()
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
    /// Empty for an eye at the centre, or farther than [`FAR_EYE_RADII`] radii.
    pub fn wanted(&mut self, body: &BodyDefinition, eye_m: [f64; 3]) -> WantedSet {
        let eye = DVec3::from_array(eye_m);
        let len = eye.length();
        let ladder = *body.ladder();
        if len.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            || len > ladder.radius_m() * FAR_EYE_RADII
        {
            // Nothing wanted, nothing kept.
            self.spans.clear();
            self.kept.clear();
            self.culled.clear();
            return WantedSet::default();
        }
        let d = eye / len;
        let surface = vd_terrain::height::height_m(
            body,
            [Gf::from_f64(d.x), Gf::from_f64(d.y), Gf::from_f64(d.z)],
            0,
        )
        .to_f64();
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
        let mut sweep = Sweep {
            ladder: &ladder,
            body,
            per_rung: vec![Vec::new(); usize::from(rungs)],
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
                let span = self.span(body, col);
                // The guaranteed floor of the DRAWN ground: the lowest sample less the bounds the
                // peak adds (the field's), less a cell (the extractor's mesh stands within a
                // cell of the field) and the sink its mesh may stand under while a finer rung is
                // drawn over it. Only a small column raises a wall: a wide one's chart quad
                // over-claims ground, and its floor is loose anyway.
                if geo.rho <= WALL_MAX_HALF_ANGLE {
                    let floor = span.sampled_low_m.to_f64()
                        - (span.peak_m - span.sampled_high_m).to_f64()
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
                let span = self.span(body, col);
                let margin = if self.kept.contains(&col) {
                    KEEP_MARGIN_RAD
                } else {
                    WANT_MARGIN_RAD
                };
                if !skyline.clears(geo.phi, geo.az, geo.rho, span.peak_m.to_f64(), margin) {
                    culled.insert(col);
                    continue;
                }
                cleared.insert(col);
                // The territory reads the TRUE horizon: a column the hysteresis sent here from
                // inside the horizon (between three quarters of it and the horizon) is ground the
                // picture draws now, and its chunks are urgent, never "revealed" (refutation
                // R4-1: the gap under-read it as a reveal).
                sweep.descend(col, &geo, &span, geo.near <= horizon, &mut next);
            }
            level = next;
        }
        let mut out = WantedSet {
            reach_m: reach,
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
        for (_, key, territory) in emitted {
            out.push(key);
            out.mark(key, territory);
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
        self.spans.retain(|_, (_, g)| *g == generation);
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

#[cfg(test)]
mod tests {
    use super::*;
    use vd_terrain::home::home_planet;

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

    #[test]
    fn the_urgent_chunks_are_the_rungs_own_territory_and_the_gap_counts_the_missing() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([1.0, 0.31, -0.22]);
        let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
        let surface = vd_terrain::height::height_m(&body, dir, 0).to_f64();
        let eye = [
            d[0] * (surface + 3.4),
            d[1] * (surface + 3.4),
            d[2] * (surface + 3.4),
        ];
        let mut view = LadderView::default();
        let w = view.wanted(&body, eye);
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
        let again = view.wanted(&body, eye);
        assert_eq!(again.keys, w.keys);
        assert!(
            !view.culled.is_empty(),
            "no column was culled from the ground"
        );
        assert!(view.culled.is_disjoint(&view.kept));
        let step = [eye[0] + 0.4, eye[1] + 0.2, eye[2] - 0.1];
        let stepped = view.wanted(&body, step);
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
        let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
        let surface = vd_terrain::height::height_m(&body, dir, 0).to_f64();
        let at = |h: f64| {
            [
                d[0] * (surface + h),
                d[1] * (surface + h),
                d[2] * (surface + h),
            ]
        };
        let under = LadderView::default().wanted(&body, at(-10.0));
        let standing = LadderView::default().wanted(&body, at(EYE_HEIGHT_M));
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
        let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
        let surface = vd_terrain::height::height_m(&body, dir, 0).to_f64();
        let eye = [
            d[0] * (surface + 3.4),
            d[1] * (surface + 3.4),
            d[2] * (surface + 3.4),
        ];
        let mut view = LadderView::default();
        let w = view.wanted(&body, eye);
        // The reach passes the 6.6 km horizon; the rings run from rung 0 to the ring that holds it.
        assert!(w.reach_m > horizon_m(surface, 3.4), "{}", w.reach_m);
        assert_eq!(w.rung_min, 0);
        // The coarsest ring holds the horizon at least, and never passes the reach's rung (a far
        // ring with no visible peak is empty and names no rung).
        assert!(w.rung_max >= rung_for_distance(horizon_m(surface, 3.4), body.ladder().rungs));
        assert!(w.rung_max <= rung_for_distance(w.reach_m, body.ladder().rungs));
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
        assert!(
            rung0 < 2 * columns0_n as u64,
            "rung 0: {rung0} chunks over {columns0_n} columns"
        );
        // Past the horizon only peaks are wanted, and every ring carries its crossfade band (a
        // fifth more chunks than the ring alone): the whole ladder stays under eight thousand
        // chunks (MEASURED 2026-09-09: 4 107 before the bands, 7 073 with them).
        let total = w.len();
        let reach = w.reach_m;
        assert!(total < 8_000, "{total} chunks: {per:?}, reach {reach} m");
        // The spans are kept for the columns the descent visits — a second call from the same
        // eye reads none anew and holds the same set; from far away nothing stays.
        let held = view.spans_held();
        let again = view.wanted(&body, eye);
        assert_eq!(again, w);
        assert_eq!(view.spans_held(), held);
        let r = body.ladder().radius_m();
        assert!(view.wanted(&body, [r * 3.0, 0.0, 0.0]).is_empty());
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
    fn the_ladder_from_orbit_is_the_globe_at_the_top_rung_and_from_far_away_nothing() {
        let body = home_planet();
        let d = vd_seed::bend::normalize([0.2, 0.9, 0.4]);
        let r = body.ladder().radius_m();
        let mut view = LadderView::default();
        // 2 000 km up: the nadir is at the top rung, so the whole visible cap is one ring.
        let orbit = [d[0] * (r + 2.0e6), d[1] * (r + 2.0e6), d[2] * (r + 2.0e6)];
        let w = view.wanted(&body, orbit);
        let top = body.ladder().rungs - 1;
        // The top rung holds the cap; the rung below fades in where its band reaches the nadir.
        assert_eq!(w.rung_max, top);
        assert!(w.rung_min + 1 >= top, "{}", w.rung_min);
        assert!(w.len() > 200, "{}", w.len());
        assert!(w.len() < 6_000, "{}", w.len());
        // The cap crosses face edges: more than one face is wanted, and the cap tiles out to the
        // horizon on every face it crosses.
        let faces: BTreeSet<Face> = w.keys.iter().map(|k| k.face).collect();
        assert!(faces.len() > 1, "{faces:?}");
        let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
        let surface = vd_terrain::height::height_m(&body, dir, 0).to_f64();
        assert_tiled(&body, &w, d, surface, r + 2.0e6 - surface);
        // From orbit over a point near a FACE EDGE, 43° from the face's centre, the cap reaches
        // 83° from that centre (the refuter's finding: a fold through one face stopped at 76°).
        let edge_d = vd_seed::bend::normalize([1.0, 0.95, 0.05]);
        let edge_eye = [
            edge_d[0] * (r + 2.0e6),
            edge_d[1] * (r + 2.0e6),
            edge_d[2] * (r + 2.0e6),
        ];
        let w_edge = view.wanted(&body, edge_eye);
        let edge_dir = [
            Gf::from_f64(edge_d[0]),
            Gf::from_f64(edge_d[1]),
            Gf::from_f64(edge_d[2]),
        ];
        let edge_surface = vd_terrain::height::height_m(&body, edge_dir, 0).to_f64();
        assert_tiled(
            &body,
            &w_edge,
            edge_d,
            edge_surface,
            r + 2.0e6 - edge_surface,
        );
        // Beyond two radii: nothing; at the centre: nothing.
        let far = [d[0] * r * 3.0, d[1] * r * 3.0, d[2] * r * 3.0];
        assert!(view.wanted(&body, far).is_empty());
        assert!(view.wanted(&body, [0.0, 0.0, 0.0]).is_empty());
        assert!(view.wanted(&body, [f64::NAN, 0.0, 0.0]).is_empty());
    }

    #[test]
    fn a_stand_near_a_face_corner_wants_columns_on_three_faces() {
        let body = home_planet();
        // The corner where +X, +Y and +Z meet, a little inside the +X face.
        let d = vd_seed::bend::normalize([1.0, 0.97, 0.97]);
        let dir = [Gf::from_f64(d[0]), Gf::from_f64(d[1]), Gf::from_f64(d[2])];
        let surface = vd_terrain::height::height_m(&body, dir, 0).to_f64();
        // From 60 km up the horizon is 877 km: the three faces at the corner are all inside it.
        let eye = [
            d[0] * (surface + 60_000.0),
            d[1] * (surface + 60_000.0),
            d[2] * (surface + 60_000.0),
        ];
        let w = LadderView::default().wanted(&body, eye);
        let faces: BTreeSet<Face> = w.keys.iter().map(|k| k.face).collect();
        assert_eq!(faces.len(), 3, "{faces:?}");
        for k in &w.keys {
            assert!(vd_terrain::lattice::in_ladder(&body, *k), "{k:?}");
        }
        // The three faces tile the cap around the corner out to the horizon.
        assert_tiled(&body, &w, d, surface, 60_000.0);
    }
}
