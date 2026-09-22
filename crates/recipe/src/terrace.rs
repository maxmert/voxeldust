//! ★ THE CAP-ROCK BENCH — slice 8a stage 4 (`slice_8a_design.md` §1.4 and §2.4;
//! `03_erosion_rivers.md` §7.3 rule 2; `04_detail_rungs.md` §4.5). A SHAPE, not a rock map: the
//! column's surface is pulled toward the nearest BED TOP, and a bed top stands at a FIXED RADIUS.
//!
//! **Why a fixed radius, and not a depth.** The strata drape at a constant DEPTH under the surface
//! (`vd_terrain::strata`), so a soft layer follows the hill and can never make a bench. A bed top at
//! a fixed radius CUTS the hill, so the soft bed wears back at ONE HEIGHT along a whole hillside —
//! which is what a cap-rock escarpment is, and what the reference picture's near cliff band is.
//!
//! ```text
//!    S           the bed spacing, a per-body length          (a soft bed over the hard bed under it)
//!    k           the bed index of h against the DATUM radius
//!    d           h − (the nearer of the two bed tops)        SIGNED
//!    u           |d| / (S/2)                                 so u runs over the whole [0, 1]
//!    m           the hardness of THAT bed top, in [0, 1]     a draw from the seed and the index
//!    terrace(h)  = h − q · m · d · f(u),  f(u) = 1 − fade(u)
//! ```
//!
//! **Why the HALF spacing.** With `u` taken against one whole thickness the falloff never reaches
//! zero before the next bed top, so the height JUMPS where the nearest top changes — COMPUTED in
//! `04` §4.5 at `q = 0.6` on a 40 m band, the jump is 12 m, six cells at rung 1. With half the
//! spacing `f(1) = 0` and `f'(1) = 0`, so the pull fades to nothing at the midpoint from BOTH sides
//! and the terrace is continuous and smooth everywhere.
//!
//! **The falloff is the QUINTIC the noise already carries** ([`fade`]), not `04`'s cubic: its
//! derivative is zero at both ends, which is strictly better for the half-spacing argument, and no
//! new polynomial enters the float fence. Its Lipschitz constant is therefore re-derived rather than
//! copied (`vd_terrain::body::TERRACE_LIP_NUM` over `TERRACE_LIP_DEN`, seven ninths, and stage 4's
//! own test scans for it).
//!
//! ★ **EACH BED HAS ITS OWN HARDNESS** — slice 8d step 1 (`slice_8d_design.md` §3.4; ruling W1 of
//! `owner_decisions_2026-09-21_water.md`, which pulled this piece forward). The body strength is now
//! a CEILING, not the pull: the pull toward a bed top is that top's own hardness share of it. A soft
//! top pulls nothing at all, so a hillside of one soft rock carries no tread. MEASURED defect it
//! cures: from 10 km over the day-side belt every slope showed a contour stripe at every bed,
//! because one body-wide strength pulled at all of them.
//!
//! ★ **WHAT IT DOES NOT DELIVER.** No overhang: the cell fold takes `greater` and can only REMOVE
//! ([`crate::cell`]), so a cap wider than its shaft is withdrawn — 8f owns that question. No mesa
//! whose LID is a different rock: that is 8d's rock map (§3.5), which reads the bed index for a
//! SUBSTANCE and is a separate piece of work. This kernel moves no address, because every bound is
//! derived at the body strength's own ceiling and a hardness share never passes one.
//!
//! **Example.** The pilot walks up a hillside. She crosses four bed tops and stands on TWO treads,
//! because two of those four beds drew hard and two drew soft. Each tread runs along the whole
//! hillside at one height, because a bed top stands at a fixed radius.

use crate::gi::Gi;
use crate::noise::{NOISE_BITS, NOISE_ONE, fade, unit_value};
use crate::rng::corner_hash;

/// The fraction bits of the bed spacing's two reciprocals: a whole word less the headroom the
/// two-word product needs, the same place [`crate::root::recip_pow2`] is read everywhere else, so
/// the bed index and the falloff's argument are each ONE multiply and the kernel divides nothing.
pub const TERRACE_RECIP_BITS: u32 = 62;

/// The length format's fraction bits — the same as the noise's, which is why a surface radius and a
/// relief add without a shift (the rule [`crate::cell`] states).
const LENGTH_BITS: u32 = NOISE_BITS;

/// The sign bit of the word: an arithmetic shift by it spreads the sign over the whole word, which
/// is how [`terrace`] picks the nearer bed top WITHOUT a branch.
const SIGN_SHIFT: u32 = i64::BITS - 1;

/// ★ THE CAP-ROCK BENCH's own words, drawn ONCE per body and per rung. `repr(C)`: EIGHT words —
/// five of its own and three of explicit padding — so the row's stride is 64 bytes, the same power
/// of two [`crate::height::Octave`] stands at, and a card reads the row the host wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Terrace {
    /// THE DATUM RADIUS the bed stack is counted from, in gap steps at the length format. The draw
    /// puts it UNDER the deepest surface the body can have, so the bed index is never negative and
    /// the truncation of the reciprocal's product IS the floor.
    pub datum: Gi,
    /// The bed spacing `S` in WHOLE gap steps. Always EVEN, so the half spacing below is exact.
    pub spacing: Gi,
    /// `floor(2^TERRACE_RECIP_BITS / spacing)` — the bed index, one two-word product.
    pub spacing_recip: Gi,
    /// `floor(2^TERRACE_RECIP_BITS / (spacing / 2))` — the falloff's argument, one two-word product.
    pub half_recip: Gi,
    /// THE STRENGTH `q` at [`NOISE_BITS`], ALREADY FADED for this rung. It is the body's CEILING,
    /// not the pull: [`bed_hardness`] takes its own share of it at each bed top. Zero at a rung
    /// whose cells are too coarse to carry a tread, and the kernel then answers its argument
    /// unchanged whatever the beds drew.
    pub strength: Gi,
    /// THE HARDNESS DRAW'S OWN SEED as a word — the body's bench salt, drawn once per body. The
    /// kernel folds it with the bed index and reads one share, so nothing about the beds crosses a
    /// lane and the card needs no new binding.
    pub hardness_seed: Gi,
    /// EXPLICIT PADDING to eight words. No kernel reads it; it stands so the row's stride is 64
    /// bytes on every host.
    pub pad: [Gi; 2],
}

impl Terrace {
    /// NO BENCH AT ALL: the strength is zero, so [`terrace`] answers its argument word for word.
    /// The spacing is one metre of gap steps and its reciprocals are real, so the arithmetic in
    /// front of the zero still reads a sane bed — a charter that carries no bench is a charter with
    /// no bench, never a charter with a degenerate divisor.
    pub const NONE: Terrace = Terrace {
        datum: Gi::ZERO,
        // 128 gap steps is one metre ([`crate::height::GAP_STEPS_PER_CELL`]); 64 is its half.
        spacing: Gi::new(128),
        spacing_recip: Gi::new(crate::root::recip_pow2(128, TERRACE_RECIP_BITS) as i64),
        half_recip: Gi::new(crate::root::recip_pow2(64, TERRACE_RECIP_BITS) as i64),
        strength: Gi::ZERO,
        hardness_seed: Gi::ZERO,
        pad: [Gi::ZERO; 2],
    };
}

/// ★ THE CAP THRESHOLD — the share a bed's draw must pass before it caps anything: ONE HALF.
///
/// **It is an ASSUMPTION, and it is named as one.** Ruling T9 forbids a drawn tuning number, so the
/// share was sought in the reference (`01_reference_target.md` §2), which states two lengths: bedded
/// rock in a cliff face is "many bands, 1–20 m thick", and a cliff is "20–200 m of vertical". They
/// give two different shares and the two disagree by a factor of thirty:
///
/// * ONE CAP PER CLIFF FACE, the face itself many bands: `sqrt(20 · 200) / sqrt(1 · 20)` = 14.1
///   bands to a cap, a share of 1 in 14;
/// * THE CAPS AVERAGE THE REFERENCE'S CLIFF SPACING: `sqrt(20 · 200)` = 63 m of vertical against
///   this body's 138 m bed spacing, which is a share OVER ONE — every bed a cap, today's picture.
///
/// They disagree because THIS BODY'S BED IS NOT THE REFERENCE'S BAND. The spacing is two of the
/// body's own sediment (`vd_terrain::body::BEDS_PER_SPACING`, 138 m on the home planet), which
/// already stands inside the reference's CLIFF range and thirty times over its BAND range. The
/// reference therefore states no share for a bed stack as coarse as this one.
///
/// So the smallest-information law stands in: an EVEN SPLIT. Half the bed tops cap and half do not.
/// It is owed to the owner as a number to look at in a picture, not as a number derived from one.
pub const CAP_THRESHOLD: Gi = Gi::new(1 << (NOISE_BITS - 1));

/// ★ THE HARDNESS OF ONE BED TOP, a share in `[0, 1)` at [`NOISE_BITS`]. A SEED IDENTITY DRAW under
/// ruling T9: WHICH beds are hard is one of the seed's own identity choices, like the body's tilt,
/// and no physical number is drawn here.
///
/// The draw is the recipe's own hash — [`corner_hash`] folds the body's bench seed with the bed
/// index through SplitMix64's three constants, and [`unit_value`] reads the top bits as a share.
/// ONE implementation: the noise draws its corners the same way, and the GPU spike measured that
/// exact arithmetic against the CPU on 3 936 256 corners with 0 differing.
///
/// **The threshold is SOFT, so the share reaches both ends.** `2 · raw − 1` is zero at
/// [`CAP_THRESHOLD`] and one at the top of the draw; under the threshold it is clamped to zero
/// WITHOUT A BRANCH, by spreading the sign over a word and subtracting the word from itself where
/// that sign is set. A bed just over the threshold therefore caps weakly, and no bed pulls harder
/// than the body's own solved strength.
///
/// **Example.** The pilot's hillside crosses the bed tops 412, 413 and 414. Bed 413 draws 0.81, so
/// its top pulls at 0.62 of the body strength and she stands on a tread. The other two draw under a
/// half, so they pull nothing and she walks straight over them.
#[must_use]
pub fn bed_hardness(seed: u64, bed: Gi) -> Gi {
    let raw = unit_value(corner_hash(seed, bed.raw(), 0, 0));
    // The rescale `1 / (1 − CAP_THRESHOLD)` is ONE SHIFT, and it is exact BECAUSE the threshold is a
    // half. A threshold the owner later moves off a half needs a reciprocal word on the charter.
    let over = (raw - CAP_THRESHOLD) << 1;
    over - (over & (over >> SIGN_SHIFT))
}

/// ★ WHICH ROCK OF A BED'S OWN HALF (slice 8d step 2): ZERO or ONE, the top bit of a SECOND draw on
/// the same bed. A SEED IDENTITY DRAW under ruling T9, like [`bed_hardness`]: which of a province's
/// two hard rocks a hard bed is made of is one of the seed's identity choices, and no physical
/// number is drawn. The draw takes `1` on its second axis where the hardness takes `0`, so the two
/// answers are independent words of one hash and the kernel pays one more round.
///
/// **Example.** The pilot's hillside crosses bed 413, which drew hard. Its pick draws one, so on a
/// folded belt the riser is quartzite and not granite — the same rock along the whole hillside,
/// because the bed index stands at a fixed radius.
#[must_use]
pub fn bed_pick(seed: u64, bed: Gi) -> Gi {
    Gi::new((corner_hash(seed, bed.raw(), 1, 0) >> (u64::BITS - 1)) as i64)
}

/// ★ THE BED INDEX OF A RADIUS, the floor of `(h − datum) / spacing`: one shift and one two-word
/// product against the spacing's stored reciprocal, so the kernel divides nothing. The datum stands
/// under every surface the body's relief can reach, so `h − datum` is never negative and the
/// product's truncation toward zero IS the floor.
///
/// ONE WRITER: [`terrace`] reads the index to pick the nearer bed top, and the cell kernel reads it
/// to pick the bed's ROCK ([`crate::cell::CellCharter::bed_rock`]). Two spellings of one index would
/// put a riser's rock at one radius and its tread at another.
///
/// **Example.** A miner digs at 6 341 900 m on a body whose beds stand 138 m apart from a datum at
/// 6 300 000 m. The index reads 303, and every cell of that bed along the whole hillside reads 303.
#[must_use]
pub fn bed_of(datum: Gi, spacing_recip: Gi, h: Gi) -> Gi {
    ((h - datum) >> LENGTH_BITS).mul_shr(spacing_recip, TERRACE_RECIP_BITS)
}

/// ★ THE TERRACE OF ONE COLUMN: the surface radius `h` pulled toward the nearest bed top, in gap
/// steps at the length format. A STRAIGHT LINE — no loop, no runtime-indexed table, no divide, and
/// NO BRANCH AT ALL.
///
/// **The nearer bed top, by an ARITHMETIC MASK.** `h` stands between the tops `lo` and `hi`; the
/// nearer is `lo` while `(h − lo) < (hi − h)`, which is the same line as `(h − lo) + (h − hi) < 0`.
/// So the sign of that ONE sum, spread over a word by an arithmetic shift, selects the difference:
/// one shift, one `and`, two adds and a subtract. `root.rs` and `cell.rs` each record a MEASURED
/// fault where a shader compiler mis-spelled a value a branch assigned, and stage 3 measured a third
/// (3 364 016 differing cells); this kernel offers no such shape.
///
/// **The bed index is a FLOOR because the datum stands under every surface.** `datum` is drawn
/// below the deepest the body's own relief can reach, so `h − datum` is never negative and
/// [`Gi::mul_shr`]'s truncation toward zero and a floor are the same answer. Where `h − datum` is an
/// exact multiple of the spacing the floored reciprocal may answer one bed low; the answer does not
/// move, because the pair `{lo, hi}` then shifts by one whole bed and the nearer of the two is the
/// same top, at the same distance.
///
/// **The bits.** `q · d` passes 63 bits on a real body (`d` reaches half a spacing at the length
/// format), so both products go through the TWO-WORD [`Gi::mul_shr`], which TRUNCATES TOWARD ZERO —
/// stated here so no reader guesses. Truncation only ever SHRINKS the pull, so every bound derived
/// at the strength's ceiling holds strictly.
///
/// ★ **THE STRENGTH IS THE BED'S OWN, AND THE SAME MASK PICKS IT** (slice 8d step 1). The mask that
/// selects the nearer bed top also selects that top's INDEX — `k` when the mask stands, `k + 1` when
/// it does not — and [`bed_hardness`] reads that index's share. The body strength is a CEILING: the
/// share never passes one, so the value bound, the difference bound and the Lipschitz factor the
/// body derives at that ceiling all hold strictly, and not one address moves.
///
/// **Example.** The pilot's column stands 40 m over a bed top on a body whose beds are 138 m apart.
/// `u` reads 0.58 and the falloff answers 0.31. If that top's bed drew hard the ground under her is
/// pulled 7 m down toward the tread below; if it drew soft she stands where the octaves put her.
#[must_use]
pub fn terrace(t: &Terrace, h: Gi) -> Gi {
    // The bed index: whole gap steps over the spacing, one two-word product.
    let k = bed_of(t.datum, t.spacing_recip, h);
    let lo = t.datum + ((k * t.spacing) << LENGTH_BITS);
    let hi = lo + (t.spacing << LENGTH_BITS);
    let below = h - lo;
    let above = h - hi;
    // The nearer of the two, SIGNED, on a mask and never on a branch.
    let nearer = (below + above) >> SIGN_SHIFT;
    let d = above + (nearer & (below - above));
    // THE NEARER TOP'S BED INDEX, on the same mask: `k` while the mask stands (all ones, so the sum
    // is `k + 1 − 1`), `k + 1` while it does not. The hardness of THAT bed scales the pull, so a
    // soft top pulls nothing at all.
    let bed = k + Gi::ONE + nearer;
    let q = (t.strength * bed_hardness(t.hardness_seed.raw() as u64, bed)) >> NOISE_BITS;
    // u = |d| / (S/2), at the noise's bits; it reaches ONE at the midpoint and never passes it.
    let mag = Gi::new(d.unsigned_abs() as i64);
    let u = mag.mul_shr(t.half_recip, TERRACE_RECIP_BITS);
    let f = NOISE_ONE - fade(u);
    h - q.mul_shr(d, NOISE_BITS).mul_shr(f, NOISE_BITS)
}

#[cfg(test)]
mod tests {
    //! ★ A TEST MAY DIVIDE (ruling F7's rule is about the SHIPPED path, not the measurement).
    #![allow(
        clippy::float_arithmetic,
        clippy::integer_division,
        clippy::modulo_arithmetic,
        reason = "a test states an exact quotient or reads the shape as real numbers; never a kernel"
    )]
    use super::*;
    use crate::root::recip_pow2;

    /// One metre in gap steps, as the length format's own unit.
    const STEPS_PER_M: i64 = 128;

    /// ★ A HARDNESS SEED WHOSE FIRST TWO BED TOPS BOTH DRAW HARD — 0.99894 and 0.99906, FOUND by a
    /// search over the draw itself. The kernel's own shape (the falloff, the Lipschitz constant, the
    /// midpoint) is a statement about a bed that CAPS, so the shape tests stand on this seed and read
    /// essentially the whole body strength. The hardness's own two arms are tested apart, below.
    const HARD_BEDS_SEED: i64 = 1_888_601;

    /// ★ A HARDNESS SEED WHOSE FIRST BED TOP DRAWS SOFT — exactly zero, under the even split. The
    /// soft arm stands on it.
    const SOFT_BED_SEED: i64 = 3;

    /// A bench of `spacing_m` metres at strength `q`, with its datum at `datum_m` metres, on the
    /// hardness seed `seed`.
    fn bench_seeded(spacing_m: i64, datum_m: i64, q: f64, seed: i64) -> Terrace {
        let spacing = spacing_m * STEPS_PER_M;
        Terrace {
            datum: Gi::new(datum_m * STEPS_PER_M) << LENGTH_BITS,
            spacing: Gi::new(spacing),
            spacing_recip: Gi::new(recip_pow2(spacing as u64, TERRACE_RECIP_BITS) as i64),
            half_recip: Gi::new(recip_pow2((spacing / 2) as u64, TERRACE_RECIP_BITS) as i64),
            strength: Gi::new((q * f64::from(1u32 << NOISE_BITS)) as i64),
            hardness_seed: Gi::new(seed),
            pad: [Gi::ZERO; 2],
        }
    }

    /// A bench of `spacing_m` metres at strength `q` whose first two beds both CAP.
    fn bench(spacing_m: i64, datum_m: i64, q: f64) -> Terrace {
        bench_seeded(spacing_m, datum_m, q, HARD_BEDS_SEED)
    }

    /// A bed's hardness as a real share.
    fn share_of(word: Gi) -> f64 {
        word.raw() as f64 / f64::from(1u32 << NOISE_BITS)
    }

    /// A height in metres as the length format's word.
    fn height(metres: f64) -> Gi {
        Gi::new((metres * (STEPS_PER_M << NOISE_BITS) as f64) as i64)
    }

    /// The word back in metres.
    fn metres(h: Gi) -> f64 {
        h.raw() as f64 / (STEPS_PER_M << NOISE_BITS) as f64
    }

    /// ★ FAILING FIRST (1): THE PULL FADES TO NOTHING AT THE MIDPOINT FROM BOTH SIDES, and it never
    /// carries the surface PAST the bed top it pulls toward.
    ///
    /// Four statements, each of which could fail. (1) At a bed top the pull is exactly zero. (2) At
    /// the midpoint between two tops the pull is zero AND its first difference is zero, approached
    /// from each side — which is what the half spacing buys. (3) Everywhere between, the pull is
    /// TOWARD the nearer top and never past it. (4) The answer is continuous where the nearest top
    /// changes: the two columns either side of a midpoint differ by under a gap step.
    #[test]
    fn the_terrace_pulls_toward_a_bed_top_and_never_past_it() {
        let t = bench(100, 1_000_000, 0.6);
        let datum_m = 1_000_000.0;
        // (1) AT A BED TOP the pull is nothing at all.
        for bed in 0..8 {
            let top = datum_m + f64::from(bed) * 100.0;
            assert_eq!(metres(terrace(&t, height(top))), top, "bed {bed}");
        }
        // (2) AT THE MIDPOINT the pull and its first difference are both zero, from each side.
        let mid = datum_m + 50.0;
        let step = 1.0 / f64::from(STEPS_PER_M as u32);
        // The falloff is computed in WHOLE WORDS, so at the midpoint it answers a few units rather
        // than a clean zero: `(S/2) × 6` units of the length format is a millionth of a metre on a
        // 100 m bed (`vd_terrain::body::TERRACE_FADE_ROUND_UNITS` derives the six). The pull is
        // under it, and the two differences below are what says the falloff really reached zero.
        let slack = 2e-6;
        assert!(
            (metres(terrace(&t, height(mid))) - mid).abs() < slack,
            "the midpoint"
        );
        let before = metres(terrace(&t, height(mid - step))) - (mid - step);
        let after = metres(terrace(&t, height(mid + step))) - (mid + step);
        assert!(before.abs() < 1e-4, "the pull below the midpoint {before}");
        assert!(after.abs() < 1e-4, "the pull above the midpoint {after}");
        // (3) BETWEEN THEM the pull is toward the nearer top and never past it.
        let mut over = 0;
        let mut under = 0;
        let mut moved_far = 0;
        for i in 1..1000 {
            let x = datum_m + f64::from(i) * 0.1;
            let raw = x - datum_m;
            let bed = (raw / 100.0).floor() * 100.0;
            let near = if raw - bed < 50.0 { bed } else { bed + 100.0 };
            let got = metres(terrace(&t, height(x))) - datum_m;
            let moved = got - raw;
            let toward = near - raw;
            // TWO ARMS and never three: a column exactly ON a bed top is not in this scan (the
            // step is a tenth of a metre and a bed is a hundred), so a third arm would be a line
            // nobody walks. Each `&&` is split, because the short-circuit's false side is a region
            // a passing assertion never reaches (HR5's own rule (d)).
            if toward > 0.0 {
                assert!(moved >= -1e-9, "at {x}: {moved}");
                assert!(moved <= toward + 1e-9, "at {x}: {moved}");
                over += 1;
            } else {
                assert!(moved <= 1e-9, "at {x}: {moved}");
                assert!(moved >= toward - 1e-9, "at {x}: {moved}");
                under += i32::from(toward < 0.0);
            }
            assert!(moved.abs() <= 50.0);
            moved_far += i32::from(moved.abs() > 1.0);
        }
        assert!(
            moved_far > 500,
            "the pull is real on most columns: {moved_far}"
        );
        assert!(over > 100, "columns pulled up: {over}");
        assert!(under > 100, "columns pulled down: {under}");
        // (4) CONTINUOUS where the nearest top changes.
        let a = metres(terrace(&t, height(mid - step)));
        let b = metres(terrace(&t, height(mid + step)));
        assert!(
            (b - a - 2.0 * step).abs() < 1e-4,
            "the midpoint step {a} {b}"
        );
    }

    /// ★ FAILING FIRST (2): THE LIPSCHITZ CONSTANT IS THE QUINTIC'S OWN, `1 + (7/9)·q`, and not the
    /// cubic's `1 + 0.6875·q` that `04_detail_rungs.md` §4.5 derived. A DENSE SCAN of the terrace's
    /// own first difference over a whole bed, against both numbers: the quintic's bound holds and
    /// the cubic's is BROKEN, so a bound copied from `04` would be a bound quietly short.
    ///
    /// ★ THE STRENGTH THE SCAN MEASURES IS THE BED'S OWN (slice 8d step 1), so the bound is read
    /// at `q · m` and not at `q`. The scan covers ONE bed, whose two halves read two different bed
    /// tops, so the extreme stands in whichever half drew harder — hence the fold below.
    #[test]
    fn the_terrace_amplifies_by_the_quintics_own_lipschitz_constant() {
        let seed = HARD_BEDS_SEED as u64;
        let m = share_of(std::cmp::max(
            bed_hardness(seed, Gi::ZERO),
            bed_hardness(seed, Gi::ONE),
        ));
        let q = 0.6 * m;
        let t = bench(100, 1_000_000, 0.6);
        let step = 1.0 / f64::from(STEPS_PER_M as u32);
        let mut worst = 0.0f64;
        let mut at = 0.0f64;
        for i in 0..12_800 {
            let x = 1_000_000.0 + f64::from(i) * step;
            let a = metres(terrace(&t, height(x)));
            let b = metres(terrace(&t, height(x + step)));
            let slope = (b - a) / step;
            if slope > worst {
                worst = slope;
                at = (x - 1_000_000.0) / 50.0;
            }
        }
        let quintic = 1.0 + q * 7.0 / 9.0;
        let cubic = 1.0 + q * 0.6875;
        assert!(worst <= quintic + 1e-3, "the quintic's bound {worst}");
        assert!(worst > cubic, "the cubic's bound is BROKEN by {worst}");
        assert!(
            (worst - quintic).abs() < 5e-3,
            "tight: {worst} vs {quintic}"
        );
        // The extreme stands where the derivation puts it: at two thirds of the half spacing,
        // measured from whichever bed top is the nearer — the fold that reads either half.
        let u_extreme = 1.0 - (at - 1.0).abs();
        assert!(
            (u_extreme - 2.0 / 3.0).abs() < 0.02,
            "the extreme at u = {u_extreme} (the scan's {at})"
        );
    }

    /// ★ FAILING FIRST (3): A STRENGTH OF ZERO IS THE IDENTITY, word for word — the shape a rung too
    /// coarse to carry a tread reads, and the shape [`Terrace::NONE`] states.
    #[test]
    fn a_strength_of_zero_answers_the_height_word_for_word() {
        let t = bench(100, 1_000_000, 0.0);
        for i in 0..400 {
            let h = height(1_000_000.0 + f64::from(i) * 0.37);
            assert_eq!(terrace(&t, h), h, "sample {i}");
            assert_eq!(terrace(&Terrace::NONE, h), h, "NONE at sample {i}");
        }
        assert_eq!(Terrace::NONE.strength, Gi::ZERO);
        assert_eq!(Terrace::NONE.spacing, Gi::new(128));
    }

    /// ★ THE KERNEL'S SHAPE, stated as a test: the falloff's argument never passes ONE, so the
    /// quintic is only ever read on the range it is defined over, and the bed index is never
    /// negative. Both are what the datum's placement buys, and a datum over a column would break
    /// them.
    #[test]
    fn the_falloff_argument_stays_inside_its_range_over_a_whole_bed() {
        let t = bench(138, 1_000, 0.56);
        let base = 1_000.0 * f64::from(STEPS_PER_M as u32);
        let mut worst_u = 0.0f64;
        for i in 0..20_000 {
            let h = Gi::new(((base + f64::from(i) * 3.7) * f64::from(1u32 << NOISE_BITS)) as i64);
            let rel = h - t.datum;
            assert!(!rel.is_negative(), "the bed index at sample {i}");
            let k = (rel >> LENGTH_BITS).mul_shr(t.spacing_recip, TERRACE_RECIP_BITS);
            let lo = t.datum + ((k * t.spacing) << LENGTH_BITS);
            let hi = lo + (t.spacing << LENGTH_BITS);
            let below = h - lo;
            let above = h - hi;
            let nearer = (below + above) >> SIGN_SHIFT;
            let d = above + (nearer & (below - above));
            let u = Gi::new(d.unsigned_abs() as i64).mul_shr(t.half_recip, TERRACE_RECIP_BITS);
            assert!(u.raw() >= 0, "u at sample {i}: {u:?}");
            assert!(u <= NOISE_ONE, "u at sample {i}: {u:?}");
            let share = u.raw() as f64 / f64::from(1u32 << NOISE_BITS);
            if share > worst_u {
                worst_u = share;
            }
        }
        assert!(worst_u > 0.99, "the scan reaches the midpoint: {worst_u}");
    }

    /// ★ FAILING FIRST (4), slice 8d step 1: THE TWO ARMS OF THE HARDNESS. A SOFT bed top pulls
    /// NOTHING — the kernel answers its argument word for word, exactly as a strength of zero does —
    /// and a HARD bed top pulls the WHOLE body strength. Before this step both arms pulled the body
    /// strength, which is the contour stripe on every slope the owner measured from 10 km.
    #[test]
    fn a_soft_bed_pulls_nothing_and_a_hard_bed_pulls_the_body_strength() {
        let q = 0.6;
        let soft = bench_seeded(100, 1_000_000, q, SOFT_BED_SEED);
        let hard = bench_seeded(100, 1_000_000, q, HARD_BEDS_SEED);
        assert_eq!(bed_hardness(SOFT_BED_SEED as u64, Gi::ZERO), Gi::ZERO);
        let m = share_of(bed_hardness(HARD_BEDS_SEED as u64, Gi::ZERO));
        assert!(m > 0.998, "the hard bed's share {m}");
        // The lower half of bed 0: the nearer top is bed 0's, so both arms read the same top.
        for i in 1..500 {
            let x = 1_000_000.0 + f64::from(i) * 0.1;
            let h = height(x);
            // THE SOFT ARM: word for word, and nothing else could make that true.
            assert_eq!(terrace(&soft, h), h, "the soft bed at {x}");
            // THE HARD ARM: the real pull `q · d · (1 − fade(u))`, toward the top below.
            let d = x - 1_000_000.0;
            let u = d / 50.0;
            let f = 1.0 - (u * u * u * (u * (6.0 * u - 15.0) + 10.0));
            let want = q * d * f;
            let got = x - metres(terrace(&hard, h));
            // The slack: the share falls 0.0011 short of one, and the integer falloff is out by a
            // few units of the length format.
            assert!(
                (got - want).abs() <= (1.0 - m) * d + 1e-3,
                "the hard bed at {x}: {got} against {want}"
            );
            assert!(got > 0.0, "the hard bed pulls at {x}");
        }
    }

    /// ★ FAILING FIRST (5), slice 8d step 1: THE HARDNESS IS A SHARE, AND THE BODY STRENGTH IS ITS
    /// CEILING. Three statements over ten thousand beds: every share stands inside `[0, 1]`; the
    /// strength the kernel reaches never passes the body's own solved strength (which is what keeps
    /// the value bound, the difference bound and the Lipschitz factor derived at that ceiling
    /// valid); and about HALF the beds cap, which is the even split [`CAP_THRESHOLD`] states.
    #[test]
    fn a_beds_hardness_is_a_share_and_the_body_strength_is_its_ceiling() {
        let strength = Gi::new(3 << (NOISE_BITS - 2));
        let mut caps = 0i32;
        let mut biggest = Gi::ZERO;
        for bed in 0..10_000i64 {
            let m = bed_hardness(0x5EED, Gi::new(bed));
            assert!(m >= Gi::ZERO, "bed {bed}: {m:?}");
            assert!(m <= NOISE_ONE, "bed {bed}: {m:?}");
            let q = (strength * m) >> NOISE_BITS;
            assert!(q <= strength, "bed {bed}: {q:?}");
            caps += i32::from(m > Gi::ZERO);
            if m > biggest {
                biggest = m;
            }
        }
        // The even split, four standard deviations wide (50 on ten thousand draws).
        let share = f64::from(caps) / 10_000.0;
        assert!(share > 0.48, "the cap share {share}");
        assert!(share < 0.52, "the cap share {share}");
        // The draw REACHES the top of its range, so "a hard bed pulls the body strength" is real.
        assert!(share_of(biggest) > 0.999, "the hardest bed {biggest:?}");
    }

    /// ★ FAILING FIRST (slice 8d step 2): THE BED INDEX IS THE FLOOR, AND THE PICK IS A COIN.
    ///
    /// Three statements. (1) [`bed_of`] answers the bed a radius stands in, and its floored
    /// reciprocal reads ONE BED LOW exactly on a bed top — which is a stated reading, not a defect,
    /// because [`terrace`] reads the same index there and its pair of tops is the same pair.
    /// (2) [`bed_pick`] answers only zero or one. (3) It answers each of them on about half the
    /// beds, so a province's two rocks of a half are both real.
    #[test]
    fn the_bed_index_floors_and_the_pick_is_one_bit() {
        let spacing = 100 * STEPS_PER_M;
        let recip = Gi::new(recip_pow2(spacing as u64, TERRACE_RECIP_BITS) as i64);
        let datum = Gi::new(1_000 * STEPS_PER_M) << LENGTH_BITS;
        let at = |metres: i64| datum + (Gi::new(metres * STEPS_PER_M) << LENGTH_BITS);
        assert_eq!(bed_of(datum, recip, at(0) + Gi::ONE), Gi::ZERO);
        assert_eq!(bed_of(datum, recip, at(99)), Gi::ZERO);
        assert_eq!(bed_of(datum, recip, at(101)), Gi::ONE);
        assert_eq!(bed_of(datum, recip, at(250)), Gi::new(2));
        // Exactly ON a bed top the floored reciprocal reads one bed low, and it is stated.
        assert_eq!(bed_of(datum, recip, at(100)), Gi::ZERO);
        let mut ones = 0i32;
        for bed in 0..10_000i64 {
            let pick = bed_pick(0x5EED, Gi::new(bed));
            assert!(pick >= Gi::ZERO, "bed {bed}: {pick:?}");
            assert!(pick <= Gi::ONE, "bed {bed}: {pick:?}");
            ones += pick.raw() as i32;
        }
        assert!(ones > 4_800, "the pick's ones {ones}");
        assert!(ones < 5_200, "the pick's ones {ones}");
    }

    /// ★ THE HARDNESS SWITCHES WHERE THE PULL IS ALREADY ZERO, so a bed that caps beside one that
    /// does not makes NO STEP. The nearer top changes at the midpoint, and the falloff is zero
    /// there from both sides — so the two columns either side of every midpoint of a whole bed
    /// stack differ by the step and no more, whatever the two beds drew.
    #[test]
    fn a_hard_bed_beside_a_soft_one_makes_no_step() {
        let t = bench_seeded(100, 1_000_000, 0.6, 0x5EED);
        let step = 1.0 / f64::from(STEPS_PER_M as u32);
        let mut switches = 0i32;
        for bed in 0..200i64 {
            let mid = 1_000_000.0 + (bed as f64) * 100.0 + 50.0;
            let a = metres(terrace(&t, height(mid - step)));
            let b = metres(terrace(&t, height(mid + step)));
            assert!(
                (b - a - 2.0 * step).abs() < 1e-4,
                "the midpoint of bed {bed}"
            );
            let low = bed_hardness(0x5EED, Gi::new(bed));
            let high = bed_hardness(0x5EED, Gi::new(bed + 1));
            switches += i32::from((low > Gi::ZERO) != (high > Gi::ZERO));
        }
        // The scan really crossed a hard bed standing beside a soft one, many times over.
        assert!(switches > 50, "hard beside soft: {switches}");
    }
}
