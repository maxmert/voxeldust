//! ★ THE ARRIVAL RAMP (the far-water design §2.7 item 3 and §10, slice R1c; ruling B18 Q7: half a
//! second).
//!
//! A finer chunk that lands while its coarser stand-in is drawn in its place used to appear at its
//! own place in one frame: MEASURED by the code map (the design §10), the ground rose by
//! `w · (s_{L+1} − m)` — up to twice the coarser rung's sink, always a rise — and its shade jumped
//! from the coarser rung's to its own. Now the chunk ENTERS ON THE STAND-IN'S DRAWN SURFACE and
//! rises into its own place over [`ARRIVAL_RAMP_S`].
//!
//! THE STAND-IN'S DRAWN SURFACE along a finer vertex's radial is the coarser surface there (the
//! vertex's morph target `T`) sunk by the coarser rung's sink times `1 − risen_{L+1}(d)`. The two
//! rungs' bands are one band (`fade_bands`: rung L's fade-out IS rung L+1's fade-in, and the sink
//! ramp ends at its edge), so `1 − risen_{L+1}(d) = whole_L(d)`, and the start is
//! `T − r · s_{L+1} · whole_L(d)` — every word of which the finer chunk already holds.
//!
//! This module is the rule the shaders run (`ladder_fade.wgsl`, the prepass, `probe.wgsl`), in
//! `f64`, so its arms are unit tests: the clock the band table carries, the ramp's weight, and the
//! placing of a vertex.
//!
//! **Example.** A hull dives over the belt and a rung-9 chunk lands late, 30 % into its fade-out
//! band. In its first frame each of its vertices stands where the sunk rung-10 ground stood; over
//! the next thirty frames it rises by a thirtieth of the old one-frame step each frame.

use vd_core::glam::DVec3;

/// The ramp's time (ruling B18 Q7): a late chunk rises into its own place over half a second.
pub const ARRIVAL_RAMP_S: f64 = 0.5;

/// The clock's resolution: ticks a second. The clock's 28 bits of them wrap after three days; the
/// ramp reads a wrapping difference, so a wrap moves nothing.
pub const CLOCK_TICKS_PER_S: f64 = 1024.0;

/// ★ THE TAG a chunk carries (its `MeshTag`): the clock of the frame it APPEARED in, in the low 28
/// bits ([`CLOCK_MASK`]), and [`RAMP_FLAG`] when it rises from its stand-in; without the flag it
/// enters whole. An untagged mesh reads nought: whole, and never new (the clock never reads nought).
/// Every chunk thus states when it appeared, a rebuilt key too, so the probe marks the chunks that
/// are new in a frame (the temporal path's T1: the engine gives a new entity no previous place).
/// Three more flags carry THE STAND-IN AT ITS OWN HEIGHT ([`OWN_HEIGHT_FLAG`], [`STAND_IN_FLAG`],
/// [`COVERED_FLAG`]; `crate::stand_in`).
pub const RAMP_FLAG: u32 = 1 << 31;
/// ★ THE FLAG OF A CHUNK WHOSE STAND-IN STANDS AT ITS OWN HEIGHT (2026-09-30; the owner: "Own
/// height"): its ramp starts on the coarser surface itself, with none of the stand-in's sink
/// ([`start_sink_share`]).
pub const OWN_HEIGHT_FLAG: u32 = 1 << 30;
/// ★ THE FLAG OF A CHUNK DRAWN AS A STAND-IN (`crate::stand_in`): it stands at its own height — its
/// sink's weight nought, whatever its band says ([`sink_weight`]) — until [`COVERED_FLAG`].
pub const STAND_IN_FLAG: u32 = 1 << 29;
/// ★ THE FLAG OF A STAND-IN WHOSE FINER GROUND HAS ALL ARRIVED: from its tag's clock it sinks under
/// that ground over the ramp ([`covered_stamp`]), and then the release hold lets it go.
pub const COVERED_FLAG: u32 = 1 << 28;
/// The clock's bits in a tag.
pub const CLOCK_MASK: u32 = COVERED_FLAG - 1;

/// ★ THE CLOCK in ticks of the client's own seconds, 28 bits, wrapping, NEVER NOUGHT (nought is an
/// untagged mesh's tag). A moment before the start reads the first tick.
#[must_use]
pub fn clock_ticks(now_s: f64) -> u32 {
    let ticks = (now_s.max(0.0) * CLOCK_TICKS_PER_S) as u64 as u32 & CLOCK_MASK;
    ticks.max(1)
}

/// ★ THE TAG a chunk appears with: this frame's clock, with [`RAMP_FLAG`] when it enters on its
/// stand-in's surface and rises — and without it, so it enters whole, when it has no stand-in or
/// needs no ramp: a key that was already drawn (a rebuild replaces a chunk the picture shows at its
/// own place), and a chunk of the TOP rung, which no coarser chunk stands in for (starting it on its
/// own unsunk surface would lift it over the finer rung it lies under inside its fade-in band).
#[must_use]
pub fn arrival_stamp(already_drawn: bool, rung: u8, rungs: u8, now: u32) -> u32 {
    let ramps = u32::from(!already_drawn & (u16::from(rung) + 1 < u16::from(rungs)));
    (ramps * RAMP_FLAG) | (now & CLOCK_MASK)
}

/// ★ THE TAG a chunk appears with, in full (2026-09-30; `crate::stand_in`): [`arrival_stamp`], with
/// [`OWN_HEIGHT_FLAG`] when the stand-in it lands on stands at its own height, and [`STAND_IN_FLAG`]
/// when it is drawn as a stand-in itself.
#[must_use]
pub fn landing_stamp(
    already_drawn: bool,
    (rung, rungs): (u8, u8),
    now: u32,
    own_height: bool,
    stand_in: bool,
) -> u32 {
    arrival_stamp(already_drawn, rung, rungs, now)
        | (u32::from(own_height) * OWN_HEIGHT_FLAG)
        | (u32::from(stand_in) * STAND_IN_FLAG)
}

/// ★ THE TAG OF A COVERED STAND-IN from clock `now`: it sinks under its finer ground over the ramp
/// from `now` on, and it ramps no more (its own arrival is done).
#[must_use]
pub fn covered_stamp(now: u32) -> u32 {
    STAND_IN_FLAG | COVERED_FLAG | (now & CLOCK_MASK)
}

/// Whether a tag's chunk is drawn as a stand-in.
#[must_use]
pub fn tag_stand_in(tag: u32) -> bool {
    tag & STAND_IN_FLAG != 0
}

/// Whether a tag's stand-in is covered (its finer ground has all arrived).
#[must_use]
pub fn tag_covered(tag: u32) -> bool {
    tag & COVERED_FLAG != 0
}

/// The clock of the frame a tag's chunk appeared in.
#[must_use]
pub fn tag_clock(tag: u32) -> u32 {
    tag & CLOCK_MASK
}

/// Whether a tag's chunk rises from its stand-in (else it entered whole).
#[must_use]
pub fn tag_ramps(tag: u32) -> bool {
    tag & RAMP_FLAG != 0
}

/// ★ Whether a tag's chunk is NEW in the frame of clock `now`: it appeared in that very frame (the
/// probe's new mark, `probe.wgsl`).
#[must_use]
pub fn appeared_at(tag: u32, now: u32) -> bool {
    tag_clock(tag) == now
}

/// The ramp's length in ticks.
#[must_use]
pub fn ramp_ticks() -> u32 {
    (ARRIVAL_RAMP_S * CLOCK_TICKS_PER_S) as u32
}

/// The tag's age at clock `now` over the ramp, clamped to `[0, 1]`: read as a wrapping 28-bit
/// difference, so the clock's wrap moves nothing.
#[must_use]
pub fn ramp_age(now: u32, tag: u32, ramp: u32) -> f64 {
    let age = f64::from(now.wrapping_sub(tag_clock(tag)) & CLOCK_MASK) / f64::from(ramp.max(1));
    age.clamp(0.0, 1.0)
}

/// ★ THE RAMP'S WEIGHT `a` (the shaders' `arrival`, step for step): 1 for a chunk that entered
/// whole (no [`RAMP_FLAG`]), else the tag's age over the ramp ([`ramp_age`]).
#[must_use]
pub fn ramp_weight(now: u32, tag: u32, ramp: u32) -> f64 {
    let ramps = u8::from(tag_ramps(tag));
    f64::from(1 - ramps) + f64::from(ramps) * ramp_age(now, tag, ramp)
}

/// ★ THE SINK'S WEIGHT (the shaders' `sink_weight`, step for step): one for a chunk of its own ring
/// (its band sinks it as before), nought for a stand-in at its own height, and for a covered
/// stand-in its age over the ramp — so it sinks under its finer ground over half a second.
#[must_use]
pub fn sink_weight(now: u32, tag: u32, ramp: u32) -> f64 {
    let stand_in = u8::from(tag_stand_in(tag));
    let covered = u8::from(tag_covered(tag));
    f64::from(1 - stand_in) + f64::from(stand_in * covered) * ramp_age(now, tag, ramp)
}

/// ★ THE SHARE OF THE STAND-IN'S SINK a ramp starts under (the shaders' `start_share`): nought when
/// the stand-in stands at its own height ([`OWN_HEIGHT_FLAG`]), else one.
#[must_use]
pub fn start_sink_share(tag: u32) -> f64 {
    f64::from(u8::from(tag & OWN_HEIGHT_FLAG == 0))
}

/// How whole a rung is at distance `d` (the shaders' `whole`): one below its fade-out band, nought
/// past it. `row` is the band table's row, `(in_lo, sink_end, out_lo, out_hi)`.
#[must_use]
pub fn whole(row: [f32; 4], d: f64) -> f64 {
    let (lo, hi) = (f64::from(row[2]), f64::from(row[3]));
    ((hi - d) / (hi - lo)).clamp(0.0, 1.0)
}

/// How far a rung has risen from under the finer one at distance `d` (the shaders' `risen`).
#[must_use]
pub fn risen(row: [f32; 4], d: f64) -> f64 {
    let (lo, end) = (f64::from(row[0]), f64::from(row[1]));
    ((d - lo) / (end - lo)).clamp(0.0, 1.0)
}

/// One vertex as the vertex stage reads it: its own place (the eye at the origin, so `own`'s
/// length is its distance), its unit radial, its metre to the coarser surface, its rung's sink,
/// the stand-in's sink (the next coarser rung's; nought at the top rung), and its tag's two
/// stand-in words: its sink's weight ([`sink_weight`]) and the share of the stand-in's sink its
/// ramp starts under ([`start_sink_share`]) — both one for a chunk of its own ring.
#[derive(Clone, Copy, Debug)]
pub struct RampVertex {
    pub own: DVec3,
    pub radial: DVec3,
    pub morph_m: f64,
    pub sink_m: f64,
    pub stand_in_sink_m: f64,
    pub sink_weight: f64,
    pub start_share: f64,
}

impl RampVertex {
    /// The morph target: the coarser surface on the vertex's radial.
    #[must_use]
    pub fn target(&self) -> DVec3 {
        self.own + self.radial * self.morph_m
    }

    /// ★ ITS OWN DRAWN PLACE under band row `row` (the vertex stage before R1c): the morph toward
    /// the coarser surface across the fade-out band, and the sink under the finer rung across the
    /// fade-in band, at the sink's weight (nought for a stand-in at its own height).
    #[must_use]
    pub fn own_place(&self, row: [f32; 4]) -> DVec3 {
        let d = self.own.length();
        self.target().lerp(self.own, whole(row, d))
            - self.radial * self.sink_m * (1.0 - risen(row, d)) * self.sink_weight
    }

    /// ★ THE STAND-IN'S DRAWN SURFACE on this vertex's radial: the coarser surface sunk by the
    /// stand-in's sink times this rung's wholeness (which is the stand-in's `1 − risen`), at the
    /// start's share (nought over a stand-in at its own height).
    #[must_use]
    pub fn stand_in_place(&self, row: [f32; 4]) -> DVec3 {
        self.target()
            - self.radial * self.stand_in_sink_m * whole(row, self.own.length()) * self.start_share
    }

    /// ★ THE DRAWN PLACE with the ramp at weight `a`: the stand-in's surface at `a = 0`, its own
    /// place at `a = 1`, and the straight line between.
    #[must_use]
    pub fn placed(&self, row: [f32; 4], a: f64) -> DVec3 {
        self.stand_in_place(row).lerp(self.own_place(row), a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ladder_view::{band_row, fade_bands};

    /// The home planet's ladder is 19 rungs; any rung with a coarser one serves.
    const RUNGS: u8 = 19;

    /// A band row of rung `rung` on the unbounded ladder, with the sink ramp ending at its edge.
    fn row(rung: u8) -> [f32; 4] {
        let (fade_in, fade_out) = fade_bands(rung, RUNGS);
        band_row((fade_in, fade_out), fade_in[1])
    }

    /// THE CLOCK: ticks of 1/1024 s, never nought, 28 bits wrapping; the ramp is 512 ticks.
    #[test]
    fn the_clock_counts_ticks_and_never_reads_nought() {
        assert_eq!(clock_ticks(0.0), 1);
        assert_eq!(clock_ticks(-3.0), 1);
        assert_eq!(clock_ticks(2.0), 2048);
        assert_eq!(ramp_ticks(), 512);
        // 2^28 ticks later the clock reads what it read: the wrap, below the tag's flags.
        assert_eq!(clock_ticks(10.0 + 262_144.0), clock_ticks(10.0));
        assert_eq!(clock_ticks(262_143.5) & !CLOCK_MASK, 0);
    }

    /// ★ THE STAND-IN'S WORDS (2026-09-30): a chunk of its own ring sinks at weight one and starts
    /// its ramp under the whole stand-in's sink; a stand-in stands at its own height (weight nought)
    /// until covered, then sinks over the ramp; a chunk landing on a stand-in at its own height
    /// starts under none of its sink. The flags leave the clock alone.
    #[test]
    fn a_stand_in_stands_at_its_own_height_until_covered_then_sinks_over_the_ramp() {
        let ramp = ramp_ticks();
        let ring = landing_stamp(false, (9, RUNGS), 7_000, false, false);
        assert_eq!(ring, arrival_stamp(false, 9, RUNGS, 7_000));
        assert_eq!(
            (sink_weight(9_000, ring, ramp), start_sink_share(ring)),
            (1.0, 1.0)
        );
        let stand_in = landing_stamp(false, (9, RUNGS), 7_000, true, true);
        assert_eq!(tag_clock(stand_in), 7_000);
        assert!(tag_stand_in(stand_in) & !tag_covered(stand_in) & tag_ramps(stand_in));
        assert_eq!(sink_weight(9_000, stand_in, ramp), 0.0);
        assert_eq!(start_sink_share(stand_in), 0.0);
        let covered = covered_stamp(8_000);
        assert!(tag_covered(covered) & tag_stand_in(covered) & !tag_ramps(covered));
        assert_eq!(ramp_weight(8_000, covered, ramp), 1.0);
        assert_eq!(sink_weight(8_000, covered, ramp), 0.0);
        assert_eq!(sink_weight(8_256, covered, ramp), 0.5);
        assert_eq!(sink_weight(9_000, covered, ramp), 1.0);
        assert_eq!(covered_stamp(CLOCK_MASK + 5) & CLOCK_MASK, 4);
    }

    /// ★ OVER A STAND-IN AT ITS OWN HEIGHT the ramp starts on the coarser surface itself; a stand-in's
    /// own place is its surface unsunk, and a covered one's at half its ramp is half sunk.
    #[test]
    fn a_ramp_over_an_own_height_stand_in_starts_on_the_coarser_surface() {
        let fine = row(9);
        let lo = f64::from(fine[2]);
        let radial = DVec3::new(0.3, -0.4, 0.866_025_403_784_438_6).normalize();
        let side = radial.cross(DVec3::Z).normalize();
        // Deep inside the finer ring (a stand-in's ground): wholly this rung, wholly sunk by band.
        let own = side * (lo * 0.25);
        let v = RampVertex {
            own,
            radial,
            morph_m: 25.0,
            sink_m: 40.0,
            stand_in_sink_m: 90.0,
            sink_weight: 0.0,
            start_share: 0.0,
        };
        assert_eq!(
            (whole(fine, own.length()), risen(fine, own.length())),
            (1.0, 0.0)
        );
        assert!((v.placed(fine, 0.0) - v.target()).length() < 1e-9);
        assert!((v.own_place(fine) - own).length() < 1e-9);
        let half = RampVertex {
            sink_weight: 0.5,
            ..v
        };
        assert!((v.own_place(fine) - half.own_place(fine) - radial * 20.0).length() < 1e-9);
    }

    /// ★ THE WEIGHT: a whole chunk reads one at once (any clock, or an untagged nought); a ramping
    /// chunk reads nought at its clock, a half at half the ramp, one at its end and after; and
    /// across the clock's wrap the same.
    #[test]
    fn the_ramp_weight_runs_from_nought_to_one_over_the_ramp_and_across_the_wrap() {
        let ramp = ramp_ticks();
        let t = |clock: u32| RAMP_FLAG | clock;
        assert_eq!(ramp_weight(5_000, 0, ramp), 1.0);
        assert_eq!(ramp_weight(5_000, 4_990, ramp), 1.0);
        assert_eq!(ramp_weight(5_000, t(5_000), ramp), 0.0);
        assert_eq!(ramp_weight(5_256, t(5_000), ramp), 0.5);
        assert_eq!(ramp_weight(5_512, t(5_000), ramp), 1.0);
        assert_eq!(ramp_weight(9_000, t(5_000), ramp), 1.0);
        // Stamped 100 ticks before the wrap, read 156 ticks after it: 256 ticks old.
        assert_eq!(ramp_weight(156, t(CLOCK_MASK - 99), ramp), 0.5);
        // A ramp of nought ticks is a ramp of one: no division by nought.
        assert_eq!(ramp_weight(5_001, t(5_000), 0), 1.0);
    }

    /// ★ THE TWO RUNGS' BANDS ARE ONE BAND: rung L's wholeness IS the stand-in's `1 − risen`
    /// at every distance, so the start this module states is the surface the stand-in draws.
    #[test]
    fn a_rungs_wholeness_is_its_stand_ins_unrisen_share_at_every_distance() {
        for rung in [0u8, 3, 9, 13] {
            let (fine, coarse) = (row(rung), row(rung + 1));
            let (lo, hi) = (f64::from(fine[2]), f64::from(fine[3]));
            let mut k = 0;
            while k <= 40 {
                let d = lo - 0.25 * (hi - lo) + f64::from(k) * 0.0375 * (hi - lo);
                let gap = whole(fine, d) - (1.0 - risen(coarse, d));
                assert!(gap.abs() < 1e-9, "rung {rung} at {d} m: {gap}");
                k += 1;
            }
        }
    }

    /// ★ THE RAMP STARTS ON THE STAND-IN AND ENDS ON ITS OWN PLACE — and THE CONTROL: with no ramp
    /// the arrival frame moves the vertex by `w · (s_{L+1} − m)`, the pop the ramp spreads over
    /// half a second.
    #[test]
    fn the_ramp_starts_on_the_stand_ins_surface_and_ends_on_its_own_place() {
        let rung = 9u8;
        let fine = row(rung);
        let coarse = row(rung + 1);
        let (lo, hi) = (f64::from(fine[2]), f64::from(fine[3]));
        let radial = DVec3::new(0.3, -0.4, 0.866_025_403_784_438_6).normalize();
        let side = radial.cross(DVec3::Z).normalize();
        let (s_fine, s_coarse, m) = (40.0, 90.0, 25.0);
        // A vertex 30 % into its fade-out band, along a line of sight across its radial.
        let d = lo + 0.3 * (hi - lo);
        let own = side * d;
        let v = RampVertex {
            own,
            radial,
            morph_m: m,
            sink_m: s_fine,
            stand_in_sink_m: s_coarse,
            sink_weight: 1.0,
            start_share: 1.0,
        };
        let w = whole(fine, own.length());
        assert!((w - 0.7).abs() < 1e-9, "{w}");
        // At a = 0: the stand-in's own placing of the coarser surface on this radial.
        let stand_in = v.target() - radial * s_coarse * (1.0 - risen(coarse, own.length()));
        assert!((v.placed(fine, 0.0) - stand_in).length() < 1e-6);
        // At a = 1: the vertex's own place, as before R1c.
        assert!((v.placed(fine, 1.0) - v.own_place(fine)).length() < 1e-9);
        // Between, the straight line: a half is the midpoint.
        let mid = v.placed(fine, 0.5);
        assert!((mid - (stand_in + v.own_place(fine)) * 0.5).length() < 1e-6);
        // THE CONTROL: the one-frame step without a ramp, along the radial, is w · (s_{L+1} − m).
        let step = (v.own_place(fine) - stand_in).dot(radial);
        assert!((step - w * (s_coarse - m)).abs() < 1e-6, "{step}");
        assert!(step > 0.0, "the arrival is a rise");
    }

    /// ★ WHO RAMPS: a chunk that lands for a key not drawn ramps; a rebuilt key and a chunk of the
    /// top rung (no stand-in) enter whole — and every one of them states the clock it appeared at.
    #[test]
    fn a_landing_chunk_ramps_unless_it_was_drawn_or_is_the_top_rung() {
        assert_eq!(arrival_stamp(false, 9, RUNGS, 7_000), RAMP_FLAG | 7_000);
        assert_eq!(arrival_stamp(false, 0, RUNGS, 7_000), RAMP_FLAG | 7_000);
        assert_eq!(arrival_stamp(true, 9, RUNGS, 7_000), 7_000);
        assert_eq!(arrival_stamp(false, RUNGS - 1, RUNGS, 7_000), 7_000);
        assert_eq!(arrival_stamp(true, RUNGS - 1, RUNGS, 7_000), 7_000);
        // A whole tag reads whole at once.
        assert_eq!(
            ramp_weight(7_000, arrival_stamp(true, 9, RUNGS, 7_000), ramp_ticks()),
            1.0
        );
        assert!(tag_ramps(arrival_stamp(false, 9, RUNGS, 7_000)));
        assert!(!tag_ramps(arrival_stamp(true, 9, RUNGS, 7_000)));
    }

    /// ★ NEW IN ITS FRAME: a landing chunk and a rebuilt one are both new at the clock they appeared
    /// at, and at no other; an untagged mesh is never new (the clock never reads nought).
    #[test]
    fn every_chunk_is_new_at_the_clock_it_appeared_at_and_at_no_other() {
        for tag in [
            arrival_stamp(false, 9, RUNGS, 7_000),
            arrival_stamp(true, 9, RUNGS, 7_000),
        ] {
            assert_eq!(tag_clock(tag), 7_000);
            assert!(appeared_at(tag, 7_000));
            assert!(!appeared_at(tag, 7_017));
        }
        assert!(!appeared_at(0, clock_ticks(0.0)));
    }
}
