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

/// The clock's resolution: ticks a second. The clock's 31 bits of them wrap after 24 days; the ramp
/// reads a wrapping difference, so a wrap moves nothing.
pub const CLOCK_TICKS_PER_S: f64 = 1024.0;

/// ★ THE TAG a chunk carries (its `MeshTag`): the clock of the frame it APPEARED in, in the low 31
/// bits ([`CLOCK_MASK`]), and [`RAMP_FLAG`] when it rises from its stand-in; without the flag it
/// enters whole. An untagged mesh reads nought: whole, and never new (the clock never reads nought).
/// Every chunk thus states when it appeared, a rebuilt key too, so the probe marks the chunks that
/// are new in a frame (the temporal path's T1: the engine gives a new entity no previous place).
pub const RAMP_FLAG: u32 = 1 << 31;
/// The clock's bits in a tag.
pub const CLOCK_MASK: u32 = RAMP_FLAG - 1;

/// ★ THE CLOCK in ticks of the client's own seconds, 31 bits, wrapping, NEVER NOUGHT (nought is an
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

/// ★ THE RAMP'S WEIGHT `a` (the shaders' `arrival`, step for step): 1 for a chunk that entered
/// whole (no [`RAMP_FLAG`]), else the tag's age over the ramp, clamped to `[0, 1]` — the age read
/// as a wrapping 31-bit difference, so the clock's wrap moves nothing.
#[must_use]
pub fn ramp_weight(now: u32, tag: u32, ramp: u32) -> f64 {
    let age = f64::from(now.wrapping_sub(tag_clock(tag)) & CLOCK_MASK) / f64::from(ramp.max(1));
    let ramps = u8::from(tag_ramps(tag));
    f64::from(1 - ramps) + f64::from(ramps) * age.clamp(0.0, 1.0)
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
/// and the stand-in's sink (the next coarser rung's; nought at the top rung).
#[derive(Clone, Copy, Debug)]
pub struct RampVertex {
    pub own: DVec3,
    pub radial: DVec3,
    pub morph_m: f64,
    pub sink_m: f64,
    pub stand_in_sink_m: f64,
}

impl RampVertex {
    /// The morph target: the coarser surface on the vertex's radial.
    #[must_use]
    pub fn target(&self) -> DVec3 {
        self.own + self.radial * self.morph_m
    }

    /// ★ ITS OWN DRAWN PLACE under band row `row` (the vertex stage before R1c): the morph toward
    /// the coarser surface across the fade-out band, and the sink under the finer rung across the
    /// fade-in band.
    #[must_use]
    pub fn own_place(&self, row: [f32; 4]) -> DVec3 {
        let d = self.own.length();
        self.target().lerp(self.own, whole(row, d))
            - self.radial * self.sink_m * (1.0 - risen(row, d))
    }

    /// ★ THE STAND-IN'S DRAWN SURFACE on this vertex's radial: the coarser surface sunk by the
    /// stand-in's sink times this rung's wholeness (which is the stand-in's `1 − risen`).
    #[must_use]
    pub fn stand_in_place(&self, row: [f32; 4]) -> DVec3 {
        self.target() - self.radial * self.stand_in_sink_m * whole(row, self.own.length())
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

    /// THE CLOCK: ticks of 1/1024 s, never nought, 31 bits wrapping; the ramp is 512 ticks.
    #[test]
    fn the_clock_counts_ticks_and_never_reads_nought() {
        assert_eq!(clock_ticks(0.0), 1);
        assert_eq!(clock_ticks(-3.0), 1);
        assert_eq!(clock_ticks(2.0), 2048);
        assert_eq!(ramp_ticks(), 512);
        // 2^31 ticks later the clock reads what it read: the wrap, below the tag's flag.
        assert_eq!(clock_ticks(10.0 + 2_097_152.0), clock_ticks(10.0));
        assert_eq!(clock_ticks(2_097_151.5) & RAMP_FLAG, 0);
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
