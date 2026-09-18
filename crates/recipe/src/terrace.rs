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
//!    terrace(h)  = h − q · d · f(u),      f(u) = 1 − fade(u)
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
//! ★ **WHAT IT DOES NOT DELIVER.** No overhang: the cell fold takes `greater` and can only REMOVE
//! ([`crate::cell`]), so a cap wider than its shaft is withdrawn — 8f owns that question. No mesa
//! whose LID is a different rock: that is 8d's rock map, which replaces this ONE body-wide strength
//! with a per-bed hardness in `[0, 1]` and moves no address, because every bound here is derived at
//! the strength's own ceiling.
//!
//! **Example.** The pilot walks up a hillside. Every 138 m of altitude the ground flattens into a
//! tread she can stand on and then rises in a riser she must climb, because the bed tops stand at
//! fixed radii and her surface crosses them.

use crate::gi::Gi;
use crate::noise::{NOISE_BITS, NOISE_ONE, fade};

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
    /// THE STRENGTH `q` at [`NOISE_BITS`], ALREADY FADED for this rung. Zero at a rung whose cells
    /// are too coarse to carry a tread, and the kernel then answers its argument unchanged.
    pub strength: Gi,
    /// EXPLICIT PADDING to eight words. No kernel reads it; it stands so the row's stride is 64
    /// bytes on every host.
    pub pad: [Gi; 3],
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
        pad: [Gi::ZERO; 3],
    };
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
/// **Example.** The pilot's column stands 40 m over a bed top on a body whose beds are 138 m apart:
/// `u` reads 0.58, the falloff answers 0.31, and the ground under her is pulled 7 m down toward the
/// tread below.
#[must_use]
pub fn terrace(t: &Terrace, h: Gi) -> Gi {
    let rel = h - t.datum;
    // The bed index: whole gap steps over the spacing, one two-word product.
    let k = (rel >> LENGTH_BITS).mul_shr(t.spacing_recip, TERRACE_RECIP_BITS);
    let lo = t.datum + ((k * t.spacing) << LENGTH_BITS);
    let hi = lo + (t.spacing << LENGTH_BITS);
    let below = h - lo;
    let above = h - hi;
    // The nearer of the two, SIGNED, on a mask and never on a branch.
    let nearer = (below + above) >> SIGN_SHIFT;
    let d = above + (nearer & (below - above));
    // u = |d| / (S/2), at the noise's bits; it reaches ONE at the midpoint and never passes it.
    let mag = Gi::new(d.unsigned_abs() as i64);
    let u = mag.mul_shr(t.half_recip, TERRACE_RECIP_BITS);
    let f = NOISE_ONE - fade(u);
    h - t.strength.mul_shr(d, NOISE_BITS).mul_shr(f, NOISE_BITS)
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

    /// A bench of `spacing_m` metres at strength `q`, with its datum at `datum_m` metres.
    fn bench(spacing_m: i64, datum_m: i64, q: f64) -> Terrace {
        let spacing = spacing_m * STEPS_PER_M;
        Terrace {
            datum: Gi::new(datum_m * STEPS_PER_M) << LENGTH_BITS,
            spacing: Gi::new(spacing),
            spacing_recip: Gi::new(recip_pow2(spacing as u64, TERRACE_RECIP_BITS) as i64),
            half_recip: Gi::new(recip_pow2((spacing / 2) as u64, TERRACE_RECIP_BITS) as i64),
            strength: Gi::new((q * f64::from(1u32 << NOISE_BITS)) as i64),
            pad: [Gi::ZERO; 3],
        }
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
    #[test]
    fn the_terrace_amplifies_by_the_quintics_own_lipschitz_constant() {
        let q = 0.6;
        let t = bench(100, 1_000_000, q);
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
        // The extreme stands where the derivation puts it: at two thirds of the half spacing.
        assert!((at - 2.0 / 3.0).abs() < 0.02, "the extreme at u = {at}");
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
}
