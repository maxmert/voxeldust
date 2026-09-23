//! ★ THE FOUR AMPLIFICATION OPERATORS — Schott, Galin, Guérin, Peytavie & Paris 2024, *Terrain
//! Amplification using Multi-scale Erosion*, ACM TOG 43(4) ([10.1145/3658200](https://doi.org/10.1145/3658200)),
//! written on the fenced integer word. THE BENCH ONLY (recommendation R1,
//! `docs/investigation/2026-09-22/fine_terrain_models_and_crates.md` §5): these operators are
//! measured before any plan for the fine ground is written. Nothing in the shipped generator calls
//! them yet.
//!
//! ★ **WHY THIS CRATE AND NOT `vd-terrain` (the task asked for the reason).** Ruling F7 states the
//! recipe goes integer-only and *"the GPU does the client's chunk work to the maximum — one source,
//! two targets"*; ruling F8 decision 5 states *"rust-gpu, one crate two compilations"*; ruling F9
//! makes the card a BUDGETED SECOND BUILDER. Recommendation R3 then asks for the fine ground to be
//! DERIVED below the shipped level *"the same kernel, the same crate, compiled into the server and
//! the client"* (SL10: one generator, a port is forbidden). An operator that must one day run in a
//! compute shader cannot live in `vd-terrain`: that crate names `Gf`, holds `Vec`, reads the store's
//! rows and depends on `vd-seed`, and a shader holds none of those. So the OPERATORS live here —
//! each one a pure function of a cell and its eight neighbours, no allocation, no float, no
//! division, no `std` — and the HOST LOOP that walks a tile, holds the buffers and ships the halo
//! lives in `crates/bins/examples/erosion_bench.rs`. That is the same split the noise and the octave
//! sum already use.
//!
//! ★ **WHAT WE COPIED AND WHAT WE WROTE.** We copied NO code. We read the reference
//! implementation's four GLSL shaders through their raw URLs
//! ([`H-Schott/MultiScaleErosion`](https://github.com/H-Schott/MultiScaleErosion), MIT) for the
//! OPERATOR DEFINITIONS and THE CONSTANTS, and wrote our own in integers. Their uniforms, read out
//! of the source: `flow_p = 1.3`, `k = 0.0005`, `p_sa = 0.8`, `p_sl = 2.0`, `max_spe = 10000`,
//! `dt = 1.0` (`erosion.glsl`); `eps = 0.00005`, `tanThresholdAngle = 0.57` (`thermal.glsl`);
//! `deposition_strength = 1.0` and the two `0.1` shares (`deposition.glsl`).
//!
//! ★ **THE THREE PLACES WE DEPART FROM THE REFERENCE, EACH STATED.**
//!
//! 1. **The routing exponent is `p = 2`, not the published `1.3`.** The task asks for an INTEGER
//!    exponent, and 1.3 is not one. The published `1.3` is Holmgren 1994's, chosen *"to avoid sharp
//!    fluvial incision produced by high exponents"*. Hyväluoma, Thorne & Turunen 2017 measured the
//!    grid-isotropy optimum of the same exponent at `W ≈ 1.3–4.1`, so `2` sits inside the published
//!    band. A square is one multiply; 1.3 is a transcendental the fence forbids anyway.
//! 2. **The drainage exponent is `m = 3/4`, not the published `0.8`.** `4/5` needs a fifth root the
//!    recipe does not hold; `3/4` is TWO of the root it does hold (`a^{3/4} = √a · √√a`), exact on
//!    every host. Our `m/n` is then `0.375` against the shader's `0.4` and the paper's *"typically"*
//!    `0.5` — inside the same band, and closer to the shipped shader than the paper's own text is.
//! 3. **The thermal exchange is written as a PAIR, not as a count.** Their shader accumulates a
//!    `receiveMul` and a `distributeMul`; ours computes the Musgrave, Kolb & Mace 1989 excess
//!    `max(s − s₀, 0)` once per pair and gives it the opposite sign on each side, so the height the
//!    upper cell loses is exactly the height the lower cell gains. Mass is conserved by
//!    construction, which a count is not.
//!
//! ★ **THE FORMATS.** Every word here is a [`Gi`] with a stated number of fraction bits, and every
//! kernel says which it reads and which it writes. A height is [`H_BITS`] (1/256 m, 3.9 mm); a slope
//! is [`S_BITS`] (1/65 536, dimensionless); a routing weight is [`W_BITS`] of one; a drainage is
//! [`A_BITS`] of one cell's own catch; the reciprocal of a chord is [`D_BITS`] per metre; a rate is
//! [`K_BITS`].
//!
//! ★ **THE TIE RULE, STATED ONCE.** Multiple-flow routing needs NO tie rule: a neighbour's weight is
//! a function of its DROP alone, so two neighbours at the same height below a cell take exactly the
//! same share, whatever order a host walks the stencil in. The only place an order can be seen is
//! the STEEPEST-descent receiver the incision floor and the Hack fit read, and there the rule is
//! **the lowest stencil slot wins** ([`crate::amplify::Routing::steepest_slot`]) — the stencil's own
//! order, the row below first, left to right, the same order `MacroLattice::neighbours` fills.
//!
//! **Example.** A pilot flies the belt at 1 448 km. Under her one macro node is 8 192 m across and
//! the artifact's row says how high it stands, which way its water leaves and how much water that
//! is. These four operators take that row, upsample it, and cut a valley into it that JOINS the
//! valley of the node beside it — a river the pilot can follow to the sea, not a ripple that stops
//! at a cell.

use crate::gi::Gi;
use crate::root::{isqrt, recip_pow2};

/// The fraction bits of a HEIGHT: 1/65 536 m, 15 micrometres. A ±64 km relief is ±2³² steps, far
/// inside the word, and the sediment field shares the format so a deposit is a height without a
/// conversion.
///
/// ★ **WHY SO FINE — A MEASUREMENT, 2026-09-22, R1's first run.** This was 8 bits (1/256 m, 3.9 mm)
/// and the WHOLE LOOP DID NOTHING: over 300 iterations the tile's ground moved *"0.000 m on average,
/// 0.0 m at the worst"*. The cause is the format, not the operators. One iteration's cut on an
/// ordinary cell — a drainage of a hundred cells at a slope of a tenth — is `80 × 838 >> 24`, which
/// TRUNCATES TO ZERO at 8 fraction bits; only a cell carrying a real river cut anything at all, and
/// the rest of the tile was frozen for ever. **An incremental loop's step must be representable, or
/// the loop is a no-op that looks like a converged answer.** Eight more fraction bits make the same
/// cut read one step, and the loop moves. The rule this states for any later kernel: a field a loop
/// adds to needs the resolution of ONE STEP, not of the answer.
pub const H_BITS: u32 = 16;
/// The fraction bits of a SLOPE (a rise over a run, dimensionless): 1/65 536. A talus angle of 30°
/// is `tan 30° = 0.577`, which is 37 837 of these.
pub const S_BITS: u32 = 16;
/// The fraction bits of a ROUTING WEIGHT: the eight weights of a cell sum to `1 << W_BITS`, to
/// within the rounding of the reciprocal.
pub const W_BITS: u32 = 20;
/// The fraction bits of a DRAINAGE, counted in the cell's own catch: 1/256 of one cell.
pub const A_BITS: u32 = 8;
/// The fraction bits of the RECIPROCAL OF A CHORD, per metre: a 64 m cell's reciprocal is
/// `2²⁰ / 64 = 16 384`. The host computes the two a level needs (the axial chord and the diagonal)
/// ONCE per level with [`crate::root::recip_pow2`], never per cell.
pub const D_BITS: u32 = 20;
/// The fraction bits of a RATE — the erodibility `k`, the thermal rate `k_γ`, the deposition
/// strength and the two shares.
pub const K_BITS: u32 = 24;
/// The fraction bits the routing's own reciprocal is taken at. The sum of eight squared slopes
/// stays under 2²⁶ for any slope the recipe can hold, so `2⁴⁰ / sum` never leaves the word.
pub const RECIP_SHIFT: u32 = 40;

/// The stencil's width: the eight neighbours, in `MacroLattice::STENCIL` order.
pub const NEIGHBOURS: usize = 8;

/// ★ OUR ROUTING EXPONENT, an integer (the module's doc states why 2 and not the published 1.3).
pub const FLOW_P: u32 = 2;
/// The slope exponent of the stream power — the shader's `p_sl`, and an integer already.
pub const SLOPE_N: u32 = 2;

/// The slot word for a cell that drains to nobody: a pit, or a cell whose every neighbour stands at
/// or above it. Eight slots fit in three bits, so this is the first word outside them.
pub const NO_SLOT: u32 = 8;

/// What one cell's routing pre-pass answers. The host stores [`Routing::inv_sum`] for every cell,
/// because the GATHER at a cell needs its NEIGHBOURS' normalisers, not its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Routing {
    /// `Σ drop_k^FLOW_P` over the lower neighbours, at [`S_BITS`]. Zero at a pit.
    pub sum: Gi,
    /// The steepest drop, at [`S_BITS`]; zero at a pit.
    pub steepest: Gi,
    /// The stencil slot of the steepest drop, the LOWEST slot on a tie; [`NO_SLOT`] at a pit.
    pub steepest_slot: u32,
    /// `floor(2^RECIP_SHIFT / max(sum, 1))` — what a weight multiplies by instead of dividing.
    pub inv_sum: Gi,
}

/// THE DROP from a cell to one neighbour, at [`S_BITS`]: `(h − h_k) · inv_d_k`, FLOORED at zero, so
/// an uphill neighbour reads exactly nothing and takes no share. `present` is the neighbour mask —
/// bit `k` set where slot `k` exists (a cube corner has seven neighbours, never eight), and an
/// absent slot reads zero by the same mask, with no branch.
///
/// **Example.** A cell of the belt stands 40 m over the cell 64 m downhill of it. The drop is
/// `40 · 2⁸ · (2²⁰/64) >> (8 + 20 − 16)` — 0.625 at 1/65 536, which is 40 960.
#[must_use]
pub fn drop_to(h: Gi, h_k: Gi, inv_d_k: Gi, present: u32, slot: usize) -> Gi {
    let raw = (h - h_k).mul_shr(inv_d_k, H_BITS + D_BITS - S_BITS);
    // The mask: all ones where the slot exists AND the drop is positive, all zeros otherwise.
    let live = (present >> slot) & 1;
    let keep = Gi::new(0i64.wrapping_sub(i64::from(live)));
    let positive = Gi::new(0i64.wrapping_sub(i64::from(!raw.is_negative())));
    raw & keep & positive
}

/// ① ★ MULTIPLE-FLOW-DIRECTION ROUTING, the pre-pass (Freeman 1991; Quinn et al. 1991; Holmgren
/// 1994; Schott et al. 2024 §4.1). The weight of a lower neighbour is `s_k^p / Σ s_j^p` with
/// `p = `[`FLOW_P`]`= 2`; this answers the SUM and its reciprocal, so the gather can multiply.
///
/// NO tie rule is needed for the weights (the module's doc). The steepest slot, which the incision
/// floor reads, breaks a tie on the LOWEST STENCIL SLOT.
///
/// **Example.** A ridge cell has three neighbours below it and five above. The three below share the
/// cell's water in proportion to the SQUARES of their drops, so a neighbour twice as steep takes
/// four times the water; the five above take nothing, and no branch was written to say so.
#[must_use]
pub fn route(h: Gi, nb: &[Gi; NEIGHBOURS], inv_d: &[Gi; NEIGHBOURS], present: u32) -> Routing {
    let mut sum = Gi::ZERO;
    let mut steepest = Gi::ZERO;
    let mut steepest_slot = NO_SLOT;
    let mut slot = 0usize;
    while slot < NEIGHBOURS {
        let drop = drop_to(h, nb[slot], inv_d[slot], present, slot);
        sum += weight_of(drop);
        // The LOWEST slot wins a tie: only a STRICTLY greater drop replaces the answer.
        if drop > steepest {
            steepest = drop;
            steepest_slot = slot as u32;
        }
        slot += 1;
    }
    let d = sum.unsigned_abs().max(1);
    Routing {
        sum,
        steepest,
        steepest_slot,
        inv_sum: Gi::new(recip_pow2(d, RECIP_SHIFT) as i64),
    }
}

/// A drop raised to [`FLOW_P`], at [`S_BITS`]: the square, with the format's own shift.
#[must_use]
pub fn weight_of(drop: Gi) -> Gi {
    drop.mul_shr(drop, S_BITS)
}

/// ★ THE WEIGHT of the flow from the neighbour in slot `k` INTO this cell, at [`W_BITS`]. The
/// gather reads it, so the drop runs the other way — from the neighbour DOWN to this cell — and the
/// normaliser is the NEIGHBOUR's, which the host stored in its pre-pass. The chord is the same word
/// either way, which is why one `inv_d` table serves both directions.
#[must_use]
pub fn weight_in(h: Gi, h_k: Gi, inv_d_k: Gi, inv_sum_k: Gi, present: u32, slot: usize) -> Gi {
    let drop = drop_to(h_k, h, inv_d_k, present, slot);
    weight_of(drop).mul_shr(inv_sum_k, RECIP_SHIFT - W_BITS)
}

/// ① ★ THE GATHER: one parallel iteration of the drainage, `a_{i+1}(p) = seed(p) + Σ_k a_i(q_k) ·
/// w(q_k → p)`, at [`A_BITS`]. Every cell reads the PREVIOUS iteration and writes its own, which is
/// what makes the operator a stencil and the answer independent of the order a host walks the tile
/// in (the C1 determinism constraint the droplet models fail).
///
/// The same routine carries the SEDIMENT in operator ④ — the sediment rides the same weights, so a
/// caller passes the sediment field as `vals` and zero as `seed`.
///
/// **Example.** The macro row under this patch says the node carries a river of a known discharge.
/// The host seeds every fine cell of that node with the node's own share, so the patch STARTS with
/// the globally correct water and the iterations only move it locally — which is what turns a
/// whole-planet problem into a tile-local one.
#[must_use]
pub fn gather(
    seed: Gi,
    h: Gi,
    nb: &[Gi; NEIGHBOURS],
    vals: &[Gi; NEIGHBOURS],
    inv_d: &[Gi; NEIGHBOURS],
    inv_sum_nb: &[Gi; NEIGHBOURS],
    present: u32,
) -> Gi {
    let mut acc = seed;
    let mut slot = 0usize;
    while slot < NEIGHBOURS {
        let w = weight_in(h, nb[slot], inv_d[slot], inv_sum_nb[slot], present, slot);
        acc += vals[slot].mul_shr(w, W_BITS);
        slot += 1;
    }
    acc
}

/// `a^{3/4}` at [`A_BITS`], by the two roots the recipe already holds: `√a · √√a`. The module's doc
/// states why 3/4 and not the shipped shader's 0.8.
///
/// **Example.** A cell that catches 10 000 of its neighbours' cells reads 1 000 here, so the river
/// that drains a hundred times the land cuts about thirty-two times as hard, not a hundred.
#[must_use]
pub fn drainage_power(a: Gi) -> Gi {
    let raw = a.unsigned_abs();
    // √a at A_BITS: √(raw · 2^A_BITS) = √a_real · 2^A_BITS.
    let r2 = isqrt(raw << A_BITS);
    // √√a at A_BITS, from the same identity applied to r2.
    let r4 = isqrt(r2 << A_BITS);
    Gi::new(((r2 * r4) >> A_BITS) as i64)
}

/// ② ★ CLAMPED STREAM POWER (Schott et al. 2024 §4.2): `ẽ = min(a^m, a_max^m) · min(s^n, s_max^n)`,
/// at [`A_BITS`]. The CLAMPS are the paper's own contribution — without them *"plunging erosion
/// features appear on steep slopes, whereas the peak regions lack erosion landmarks"*. The slope
/// term is additionally held at one, as the shipped shader's
/// `clamp(pow(steepest_slope, p_sl), 0., 1.)` does.
///
/// `a` is the routed drainage at [`A_BITS`], `slope` the steepest drop at [`S_BITS`], `a_max` and
/// `s_max` the two clamps in the same formats, `spe_max` the shader's `max_spe` at [`A_BITS`].
///
/// **Example.** A cliff face of the belt stands at a slope of three. Without the clamp the square of
/// three cuts nine times as hard as a slope of one and the cliff plunges into a slot; with it the
/// cliff cuts exactly as hard as a slope of one, and the peak beside it keeps its shape.
#[must_use]
pub fn stream_power(a: Gi, slope: Gi, a_max: Gi, s_max: Gi, spe_max: Gi) -> Gi {
    let s = min_gi(slope, s_max);
    let one = Gi::ONE << S_BITS;
    let s_pow = min_gi(s.mul_shr(s, S_BITS), one);
    let a_pow = min_gi(drainage_power(a), drainage_power(a_max));
    min_gi(a_pow.mul_shr(s_pow, S_BITS), spe_max)
}

/// ② The CUT one iteration takes out of a cell, at [`H_BITS`]: the stream power times the
/// erodibility `k` (at [`K_BITS`]) times the rock's own share.
///
/// ★ **THE HARDNESS FIELD IS THE ROCK PROVINCE.** Schott et al. write `k(p) = k · (1 − ρ(p))` with
/// `ρ` a fractal noise, and say plainly that *"introducing randomness in the hardness function also
/// reduces the axis-aligned artifacts produced by the regular grid discretization"*. We do not need
/// a noise: the artifact row already carries the ROCK PROVINCE (`ARTIFACT_VERSION 5`, ruling W4),
/// and `Province::erodibility_q8` already turns it into a share of `k` in 1/256 — a shield of
/// granite and quartzite cutting about a tenth as fast as a shale shelf. One field, two defects.
///
/// **Example.** A river crosses from a folded belt onto a shale shelf. The same water cuts the shale
/// several times faster, so the valley widens where the rock softens — which is what a real river
/// does, and what a noise hardness can only imitate.
#[must_use]
pub fn cut(spe: Gi, k: Gi, hardness_q8: Gi) -> Gi {
    let scaled = k.mul_shr(hardness_q8, 8);
    spe.mul_shr(scaled, K_BITS + A_BITS - H_BITS)
}

/// ② The INCISION, with the reference's own floor: `h − cut`, never below the height of the cell it
/// drains into. The shipped shader's last line is `new_height = max(new_height, receiver_height)`,
/// and it is what stops a cell cutting a hole under its own outlet.
#[must_use]
pub fn incise(h: Gi, cut: Gi, h_receiver: Gi) -> Gi {
    max_gi(h - cut, h_receiver)
}

/// ③ ★ THERMAL STABILISATION (Musgrave, Kolb & Mace 1989; Schott et al. 2024 §4.3):
/// `∂h/∂t = −k_γ · max(s − s₀, 0)`, with `s₀ = tan γ₀` the talus angle. Material moves DOWN every
/// slope steeper than the talus, and the height a cell loses to a neighbour is exactly the height
/// that neighbour gains (the module's doc, departure 3): the excess of a PAIR is one number, and it
/// enters this cell's answer with one sign and the neighbour's with the other.
///
/// `talus` is `tan γ₀` at [`S_BITS`] (the shader's `tanThresholdAngle = 0.57`, which is 37 356);
/// `rate` is `k_γ` at [`K_BITS`], a height per unit of excess slope per iteration.
///
/// **Example.** A scree slope above the pilot's landing site stands at 40°, over a talus of 30°. Each
/// iteration moves the excess ten degrees' worth of rock downhill, and after enough iterations the
/// scree stands at exactly 30° — which is the angle a real scree stands at, for exactly this reason.
#[must_use]
pub fn thermal(
    h: Gi,
    nb: &[Gi; NEIGHBOURS],
    inv_d: &[Gi; NEIGHBOURS],
    present: u32,
    talus: Gi,
    rate: Gi,
) -> Gi {
    let mut net = Gi::ZERO;
    let mut slot = 0usize;
    while slot < NEIGHBOURS {
        // The excess in each direction: at most one of the two is non-zero, and both read zero
        // where the slot is absent, because `drop_to` masks it.
        let down = excess(drop_to(h, nb[slot], inv_d[slot], present, slot), talus);
        let up = excess(drop_to(nb[slot], h, inv_d[slot], present, slot), talus);
        net += up - down;
        slot += 1;
    }
    h + net.mul_shr(rate, K_BITS + S_BITS - H_BITS)
}

/// `max(s − s₀, 0)` at [`S_BITS`] — the talus excess, floored at zero by a mask, with no branch.
#[must_use]
pub fn excess(slope: Gi, talus: Gi) -> Gi {
    let over = slope - talus;
    let positive = Gi::new(0i64.wrapping_sub(i64::from(!over.is_negative())));
    over & positive
}

/// ④ ★ DEPOSITION (Schott et al. 2024 §4.4), as the shipped `deposition.glsl` writes it: the
/// suspended sediment is created in proportion to the fluvial erosion, transported by the SAME
/// routing weights, and settles where the transport capacity falls under what is carried.
///
/// The shader's three lines, in our formats: `sed += share · spe`, then, where
/// `strength · sed > spe`, `deposit = min(sed, (strength · sed − spe) · share)`. `sed_in` is what
/// the gather brought (at [`H_BITS`], the sediment riding the same weights as the drainage), `spe_h`
/// the cut this iteration's stream power would take (also [`H_BITS`], from [`cut`]), `strength` the
/// shader's `deposition_strength = 1.0` at [`K_BITS`] and `share` its two `0.1` at [`K_BITS`].
///
/// It answers `(deposit, sediment kept)`, both at [`H_BITS`]. **This is the operator that makes a
/// valley floor FLAT instead of a slot**, which is the whole difference between a drawn ravine and
/// a place a pilot can set a hull down in.
#[must_use]
pub fn deposit(sed_in: Gi, spe_h: Gi, strength: Gi, share: Gi) -> (Gi, Gi) {
    let sed = sed_in + spe_h.mul_shr(share, K_BITS);
    let capacity = sed.mul_shr(strength, K_BITS) - spe_h;
    let over = {
        let positive = Gi::new(0i64.wrapping_sub(i64::from(!capacity.is_negative())));
        capacity & positive
    };
    let d = min_gi(sed, over.mul_shr(share, K_BITS));
    (d, sed - d)
}

/// The smaller of two words. Written out: the recipe's fence keeps `Ord::min` out of a kernel for
/// the same reason `root.rs` writes its steps out — a shader compiler must not be asked to spell a
/// generic the CPU folds away.
#[must_use]
pub fn min_gi(a: Gi, b: Gi) -> Gi {
    if a < b { a } else { b }
}

/// The larger of two words.
#[must_use]
pub fn max_gi(a: Gi, b: Gi) -> Gi {
    if a > b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64 m cell's chord reciprocals, the two words a host computes ONCE per level with
    /// [`recip_pow2`]: the axial chord (2²⁰/64, exact) and the diagonal (2²⁰/90, floored).
    const INV_64: i64 = 16_384;
    const INV_90: i64 = 11_650;

    fn metres(m: i64) -> Gi {
        Gi::new(m << H_BITS)
    }

    fn flat_inv() -> [Gi; NEIGHBOURS] {
        [
            Gi::new(INV_90),
            Gi::new(INV_64),
            Gi::new(INV_90),
            Gi::new(INV_64),
            Gi::new(INV_64),
            Gi::new(INV_90),
            Gi::new(INV_64),
            Gi::new(INV_90),
        ]
    }

    /// ★ OPERATOR ① — A STATED FIELD, THE ANSWER COMPUTED BY HAND.
    ///
    /// The cell stands at 100 m. Slot 1 (the cell below it, 64 m away) stands at 68 m; slot 4 (the
    /// cell to its right, 64 m away) stands at 84 m. Every other neighbour stands at 100 m.
    ///
    /// By hand: the drop to slot 1 is `32/64 = 0.5`; to slot 4, `16/64 = 0.25`. Squared, `0.25` and
    /// `0.0625`; their sum is `0.3125`. So slot 1 takes `0.25/0.3125 = 0.8` of the water and slot 4
    /// takes `0.2` — EXACTLY four to one, because the weight goes as the SQUARE and the drop is
    /// twice as big. The steepest drop is 0.5, at slot 1.
    #[test]
    fn the_routing_splits_the_water_by_the_square_of_the_drop() {
        let h = metres(100);
        let mut nb = [metres(100); NEIGHBOURS];
        nb[1] = metres(68);
        nb[4] = metres(84);
        let inv = flat_inv();
        let r = route(h, &nb, &inv, 0xFF);

        // The two drops, at S_BITS: 0.5 is 32 768 and 0.25 is 16 384, EXACTLY — a 64 m chord's
        // reciprocal at D_BITS is 2²⁰/64 = 16 384 with no remainder, so nothing is floored here.
        let drop1 = drop_to(h, nb[1], inv[1], 0xFF, 1);
        let drop4 = drop_to(h, nb[4], inv[4], 0xFF, 4);
        assert_eq!(drop1.raw(), 32_768);
        assert_eq!(drop4.raw(), 16_384);
        // An uphill neighbour reads EXACTLY nothing.
        assert_eq!(drop_to(h, nb[0], inv[0], 0xFF, 0), Gi::ZERO);
        // The steepest, and its slot.
        assert_eq!(r.steepest, drop1);
        assert_eq!(r.steepest_slot, 1);
        // The sum is the two squares: 0.25 + 0.0625 = 0.3125, which is 20 480 at S_BITS.
        assert_eq!(r.sum.raw(), 20_480);

        // The two weights, gathered from the other side: four to one, to one part in 2²⁰.
        let w1 = weight_in(nb[1], h, inv[1], r.inv_sum, 0xFF, 1);
        let w4 = weight_in(nb[4], h, inv[4], r.inv_sum, 0xFF, 4);
        assert_eq!(w1.raw(), 838_860);
        assert_eq!(w4.raw(), 209_715);
        // 0.8 of one at W_BITS is 838 861; 0.2 is 209 715. Both land within 25 units of 2²⁰.
        assert!((w1.raw() - 838_861).abs() < 32);
        assert!((w4.raw() - 209_715).abs() < 32);
        // And they sum to one, to the same tolerance: the reciprocal normalised them.
        assert!(((w1 + w4).raw() - (1 << W_BITS)).abs() < 48);
    }

    /// ★ A PIT drains to nobody, and says so without a branch anywhere in the caller.
    #[test]
    fn a_pit_holds_its_water_and_names_no_slot() {
        let h = metres(10);
        let nb = [metres(40); NEIGHBOURS];
        let r = route(h, &nb, &flat_inv(), 0xFF);
        assert_eq!(r.sum, Gi::ZERO);
        assert_eq!(r.steepest, Gi::ZERO);
        assert_eq!(r.steepest_slot, NO_SLOT);
        // The reciprocal of a zero sum reads the reciprocal of one: 2⁴⁰, and every weight is zero.
        assert_eq!(r.inv_sum.raw(), 1i64 << RECIP_SHIFT);
        assert_eq!(
            weight_in(nb[1], h, Gi::new(INV_64), r.inv_sum, 0xFF, 1),
            Gi::ZERO
        );
    }

    /// ★ THE TIE RULE, and that the WEIGHTS do not need one. Two neighbours at the same height
    /// below the cell take exactly the same share; the steepest SLOT is the lower one.
    #[test]
    fn two_level_neighbours_share_alike_and_the_lowest_slot_wins_the_tie() {
        let h = metres(100);
        let mut nb = [metres(100); NEIGHBOURS];
        nb[3] = metres(50);
        nb[4] = metres(50);
        let inv = flat_inv();
        let r = route(h, &nb, &inv, 0xFF);
        assert_eq!(r.steepest_slot, 3);
        let w3 = weight_in(nb[3], h, inv[3], r.inv_sum, 0xFF, 3);
        let w4 = weight_in(nb[4], h, inv[4], r.inv_sum, 0xFF, 4);
        assert_eq!(w3, w4);
        assert!(((w3 + w4).raw() - (1 << W_BITS)).abs() < 48);
    }

    /// ★ AN ABSENT SLOT — a cube corner has SEVEN neighbours — takes nothing, and the other seven
    /// still share the whole.
    #[test]
    fn an_absent_neighbour_takes_no_water() {
        let h = metres(100);
        let nb = [metres(50); NEIGHBOURS];
        let inv = flat_inv();
        let present = 0x7F;
        let r = route(h, &nb, &inv, present);
        assert_eq!(drop_to(h, nb[7], inv[7], present, 7), Gi::ZERO);
        let mut total = Gi::ZERO;
        for slot in 0..NEIGHBOURS {
            total += weight_in(nb[slot], h, inv[slot], r.inv_sum, present, slot);
        }
        assert!((total.raw() - (1 << W_BITS)).abs() < 64);
    }

    /// ★ THE GATHER adds the seed to what the routing brings, and a cell with no upslope neighbour
    /// reads exactly its seed.
    #[test]
    fn the_gather_is_the_seed_plus_what_the_slopes_bring() {
        // The cell stands at 50 m; slot 1 stands at 82 m (32 m above, 64 m away) and drains ONLY
        // into it — every other neighbour of slot 1 is, by construction of this stated field, at
        // 82 m, so slot 1's whole normaliser is that one drop and its weight into us is ONE.
        let h = metres(50);
        let mut nb = [metres(50); NEIGHBOURS];
        nb[1] = metres(82);
        let inv = flat_inv();
        // Slot 1's own routing: its only lower neighbour is us, at the same chord.
        let mut ours_from_1 = [metres(82); NEIGHBOURS];
        ours_from_1[6] = h; // the opposite slot: the row above, centre
        let r1 = route(nb[1], &ours_from_1, &inv, 0xFF);
        let mut inv_sum_nb = [Gi::new(1i64 << RECIP_SHIFT); NEIGHBOURS];
        inv_sum_nb[1] = r1.inv_sum;
        let mut vals = [Gi::ZERO; NEIGHBOURS];
        vals[1] = Gi::new(100 << A_BITS);
        let seed = Gi::new(1 << A_BITS);
        let a = gather(seed, h, &nb, &vals, &inv, &inv_sum_nb, 0xFF);
        // One of its own plus the whole hundred the neighbour carried, to one part in 2²⁰.
        assert_eq!(a.raw(), 25_856);
        assert!((a.raw() - (101 << A_BITS)).abs() < 8);
        // With nothing above it the same cell reads exactly its seed.
        let flat = [h; NEIGHBOURS];
        assert_eq!(gather(seed, h, &flat, &vals, &inv, &inv_sum_nb, 0xFF), seed);
    }

    /// ★ `a^{3/4}`, BY HAND: 16 to the three quarters is 8; 256 to the three quarters is 64; and
    /// one is one. The two roots, exact.
    #[test]
    fn the_drainage_power_is_the_three_quarter_root() {
        assert_eq!(drainage_power(Gi::new(16 << A_BITS)).raw(), 8 << A_BITS);
        assert_eq!(drainage_power(Gi::new(256 << A_BITS)).raw(), 64 << A_BITS);
        assert_eq!(drainage_power(Gi::new(1 << A_BITS)).raw(), 1 << A_BITS);
        assert_eq!(drainage_power(Gi::ZERO).raw(), 0);
        // 10 000^0.75 = 1 000, exactly: the reason the example in the doc reads a thousand.
        assert_eq!(
            drainage_power(Gi::new(10_000 << A_BITS)).raw(),
            1_000 << A_BITS
        );
    }

    /// ★ OPERATOR ② — A STATED CELL, THE ANSWER COMPUTED BY HAND.
    ///
    /// The cell catches 10 000 cells and stands at a slope of 0.5. By hand:
    /// `a^{3/4} = 1 000`; `s² = 0.25`; `ẽ = 250`. With `k = 0.0005` and a shale shelf's hardness
    /// (256/256, the reference rock), the cut is `0.125` m per iteration — 8 192 of the
    /// 1/65 536 m steps.
    #[test]
    fn the_stream_power_is_the_drainage_to_the_three_quarters_times_the_squared_slope() {
        let a = Gi::new(10_000 << A_BITS);
        let s = Gi::new(1 << (S_BITS - 1)); // 0.5
        let a_max = Gi::new(1_000_000 << A_BITS);
        let s_max = Gi::new(4 << S_BITS);
        let spe_max = Gi::new(10_000 << A_BITS);
        let e = stream_power(a, s, a_max, s_max, spe_max);
        assert_eq!(e.raw(), 250 << A_BITS);
        // k = 0.0005 at K_BITS is 8 388 (floor of 0.0005 · 2²⁴).
        let k = Gi::new(8_388);
        let cut_h = cut(e, k, Gi::new(256));
        assert_eq!(cut_h.raw(), 8_191);
        // 0.125 m is 8 192 of the 1/65 536 m steps; the floor takes one.
        assert!((cut_h.raw() - 8_192).abs() <= 1);
        // A shield of granite and quartzite (erodibility 27/256) cuts about a tenth as hard.
        assert_eq!(cut(e, k, Gi::new(27)).raw(), 863);
    }

    /// ★ THE TWO CLAMPS, each able to fail: a slope past `s_max` cuts exactly as a slope AT the
    /// clamp, and the slope term never passes one whatever the slope.
    #[test]
    fn the_clamps_hold_the_cliff_and_the_flood() {
        let a = Gi::new(16 << A_BITS);
        let a_max = Gi::new(1_000_000 << A_BITS);
        let spe_max = Gi::new(10_000 << A_BITS);
        let s_max = Gi::new(1 << S_BITS);
        // A slope of three and a slope of one give the SAME answer once the clamp is one.
        let steep = stream_power(a, Gi::new(3 << S_BITS), a_max, s_max, spe_max);
        let unit = stream_power(a, Gi::new(1 << S_BITS), a_max, s_max, spe_max);
        assert_eq!(steep, unit);
        // And the slope term is held at one: 16^0.75 = 8, so ẽ reads 8 and no more.
        assert_eq!(steep.raw(), 8 << A_BITS);
        // The DRAINAGE clamp: a flood past `a_max` reads `a_max`'s own power.
        let flood = stream_power(
            Gi::new(1_000_000_000 << A_BITS),
            Gi::new(1 << S_BITS),
            Gi::new(16 << A_BITS),
            s_max,
            spe_max,
        );
        assert_eq!(flood.raw(), 8 << A_BITS);
        // And the reference's own outer clamp, `max_spe`.
        let capped = stream_power(a, Gi::new(1 << S_BITS), a_max, s_max, Gi::new(3 << A_BITS));
        assert_eq!(capped.raw(), 3 << A_BITS);
    }

    /// ★ THE INCISION FLOOR: a cell never cuts below the cell it drains into.
    #[test]
    fn the_incision_stops_at_the_receiver() {
        let h = metres(100);
        assert_eq!(incise(h, metres(5), metres(80)), metres(95));
        assert_eq!(incise(h, metres(50), metres(80)), metres(80));
    }

    /// ★ OPERATOR ③ — A STATED PAIR, THE ANSWER COMPUTED BY HAND.
    ///
    /// The cell stands at 100 m; slot 1, 64 m away, stands at 68 m. The drop is 0.5; the talus is
    /// `tan 30° = 0.577`. **0.5 is UNDER the talus, so nothing moves at all** — and that is the test
    /// that can fail, because a wrong sign or a missing floor would move material here.
    ///
    /// Raise the drop over the talus and the material moves DOWNHILL: the cell loses, and the
    /// neighbour computing its own answer gains exactly the same, which the second half asserts.
    #[test]
    fn the_thermal_moves_nothing_under_the_talus_and_moves_a_pair_alike_over_it() {
        let talus = Gi::new(37_356); // tan 30°, the shader's 0.57
        let rate = Gi::new(1 << K_BITS); // one height unit per unit of excess slope
        let inv = flat_inv();

        // Under the talus: the cell is unchanged, bit for bit.
        let h = metres(100);
        let mut nb = [metres(100); NEIGHBOURS];
        nb[1] = metres(68);
        assert_eq!(thermal(h, &nb, &inv, 0xFF, talus, rate), h);

        // Over it: the drop to slot 1 is 64/64 = 1.0, an excess of 0.430 over the talus (the
        // shader's 0.57 is 37 356 at S_BITS, so the excess is 65 536 − 37 356 = 28 180).
        nb[1] = metres(36);
        let moved = thermal(h, &nb, &inv, 0xFF, talus, rate);
        let lost = h - moved;
        assert_eq!(lost.raw(), 28_180);
        // At the rate of one height unit per unit of excess slope, 0.430 of a slope is 0.430 m,
        // which is 28 180 of the 1/65 536 m steps.
        assert!((lost.raw() - 28_180).abs() <= 4);

        // ★ THE PAIR: the neighbour, computing ITS own answer from the same field, GAINS exactly
        // what this cell lost. Mass is conserved by construction, not by care.
        let mut from_1 = [metres(36); NEIGHBOURS];
        from_1[6] = h; // the opposite stencil slot
        let gained = thermal(nb[1], &from_1, &inv, 0xFF, talus, rate) - nb[1];
        assert_eq!(gained, lost);
    }

    /// ★ OPERATOR ④ — A STATED CELL, THE ANSWER COMPUTED BY HAND.
    ///
    /// The shader's constants: `deposition_strength = 1.0`, and the two shares `0.1`.
    ///
    /// A cell carries 100 height units of sediment and its stream power would cut 10. By hand:
    /// `sed = 100 + 0.1·10 = 101`; `strength·sed − spe = 101 − 10 = 91`; the deposit is
    /// `min(101, 0.1·91) = 9.1`. The cell rises by 9.1 and keeps 91.9.
    ///
    /// And where the stream power EXCEEDS what is carried, nothing settles: the river is still
    /// cutting.
    #[test]
    fn the_deposition_settles_the_excess_and_nothing_where_the_river_still_cuts() {
        let strength = Gi::new(1 << K_BITS);
        // 0.1 at K_BITS, floored: floor(2²⁴/10) = 1 677 721.
        let share = Gi::new(1_677_721);
        let (d, kept) = deposit(
            Gi::new(100 << H_BITS),
            Gi::new(10 << H_BITS),
            strength,
            share,
        );
        // 9.1 m at H_BITS is 596 377; 91.9 m is 6 022 758.
        assert_eq!(d.raw(), 596_377);
        assert_eq!(kept.raw(), 6_022_758);
        assert!((d.raw() - 596_377).abs() < 64);
        assert!((kept.raw() - 6_022_758).abs() < 64);
        // The sediment is conserved: what settled plus what is carried is what arrived plus what
        // the erosion created.
        // (the share is floor(2²⁴/10), one part in 1.7 million low, so the sum reads a few steps
        // short of 101 m — a stated floor, not a drift; a step is 15 micrometres).
        assert!(((d + kept).raw() - (101 << H_BITS)).abs() < 64);

        // A river still cutting deposits NOTHING.
        let (none, carried) = deposit(
            Gi::new(1 << H_BITS),
            Gi::new(1_000 << H_BITS),
            strength,
            share,
        );
        assert_eq!(none, Gi::ZERO);
        // It keeps what it arrived with plus the hundred metres the erosion put into it, less the
        // share's own floor.
        assert!((carried.raw() - ((1 << H_BITS) + (100 << H_BITS))).abs() < 64);
    }

    /// The two comparisons the kernels use instead of a generic `min`/`max`.
    #[test]
    fn the_smaller_and_the_larger_word() {
        assert_eq!(min_gi(Gi::new(3), Gi::new(7)), Gi::new(3));
        assert_eq!(min_gi(Gi::new(7), Gi::new(3)), Gi::new(3));
        assert_eq!(max_gi(Gi::new(3), Gi::new(7)), Gi::new(7));
        assert_eq!(max_gi(Gi::new(7), Gi::new(3)), Gi::new(7));
        assert_eq!(excess(Gi::new(5), Gi::new(9)), Gi::ZERO);
        assert_eq!(excess(Gi::new(9), Gi::new(5)), Gi::new(4));
    }
}
